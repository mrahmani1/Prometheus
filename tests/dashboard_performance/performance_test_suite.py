"""
Dashboard Performance Test Suite

Comprehensive performance testing for the causal dashboard including
load testing, memory usage, rendering performance, and user interaction responsiveness.
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import statistics
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for dashboard operations"""
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    network_requests: int = 0
    network_bytes: int = 0
    dom_nodes: int = 0
    javascript_heap_mb: float = 0.0
    render_time_ms: float = 0.0
    interaction_delay_ms: float = 0.0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadTestResult:
    """Result from load testing"""
    test_name: str
    concurrent_users: int
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    requests_per_second: float
    errors: List[str] = field(default_factory=list)
    memory_peak_mb: float = 0.0
    cpu_peak_percent: float = 0.0


class DashboardPerformanceTestSuite:
    """
    Comprehensive performance testing suite for the causal dashboard
    """
    
    def __init__(self, base_url: str = "http://localhost:3000", output_dir: str = "performance_results"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        self.performance_metrics: List[PerformanceMetrics] = []
        self.load_test_results: List[LoadTestResult] = []
        
        # Performance thresholds
        self.thresholds = {
            'page_load_time_ms': 3000,
            'interaction_response_ms': 500,
            'memory_usage_mb': 500,
            'cpu_usage_percent': 80,
            'network_visualization_render_ms': 2000,
            'data_update_latency_ms': 1000,
            'concurrent_users_supported': 50
        }
        
        # Test configurations
        self.test_scenarios = [
            'dashboard_loading',
            'causal_network_rendering',
            'real_time_data_streaming',
            'large_dataset_visualization',
            'multi_user_interaction',
            'memory_stress_test',
            'network_performance_test'
        ]
    
    async def setup_browser(self, headless: bool = True):
        """Setup Playwright browser for testing"""
        playwright = await async_playwright().start()
        
        self.browser = await playwright.chromium.launch(
            headless=headless,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--allow-running-insecure-content',
                '--enable-precise-memory-info'
            ]
        )
        
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            permissions=['geolocation'],
            geolocation={'latitude': 45.0, 'longitude': -120.0}
        )
        
        # Enable performance monitoring
        await self.context.add_init_script("""
            window.performanceMetrics = {
                navigationStart: performance.now(),
                interactions: [],
                networkRequests: [],
                memoryUsage: [],
                renderTimes: []
            };
            
            // Monitor network requests
            const originalFetch = window.fetch;
            window.fetch = function(...args) {
                const startTime = performance.now();
                return originalFetch.apply(this, args).then(response => {
                    const endTime = performance.now();
                    window.performanceMetrics.networkRequests.push({
                        url: args[0],
                        duration: endTime - startTime,
                        status: response.status,
                        size: response.headers.get('content-length') || 0
                    });
                    return response;
                });
            };
            
            // Monitor interactions
            document.addEventListener('click', (e) => {
                window.performanceMetrics.interactions.push({
                    type: 'click',
                    timestamp: performance.now(),
                    target: e.target.tagName + (e.target.id ? '#' + e.target.id : '')
                });
            });
            
            // Monitor memory usage
            setInterval(() => {
                if (performance.memory) {
                    window.performanceMetrics.memoryUsage.push({
                        timestamp: performance.now(),
                        used: performance.memory.usedJSHeapSize,
                        total: performance.memory.totalJSHeapSize,
                        limit: performance.memory.jsHeapSizeLimit
                    });
                }
            }, 1000);
        """)
        
        self.page = await self.context.new_page()
        self.page.set_default_timeout(30000)
    
    async def cleanup_browser(self):
        """Cleanup browser resources"""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
    
    def _start_performance_monitoring(self, operation_name: str) -> PerformanceMetrics:
        """Start monitoring performance for an operation"""
        process = psutil.Process()
        
        return PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            end_time=datetime.now(),  # Will be updated
            duration_ms=0.0,
            memory_usage_mb=process.memory_info().rss / 1024 / 1024,
            cpu_usage_percent=process.cpu_percent()
        )
    
    async def _end_performance_monitoring(self, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """End performance monitoring and collect final metrics"""
        metrics.end_time = datetime.now()
        metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
        
        process = psutil.Process()
        final_memory = process.memory_info().rss / 1024 / 1024
        final_cpu = process.cpu_percent()
        
        # Get browser performance metrics
        if self.page:
            try:
                browser_metrics = await self.page.evaluate("""
                    () => {
                        const metrics = window.performanceMetrics || {};
                        const memoryInfo = performance.memory || {};
                        
                        return {
                            networkRequests: metrics.networkRequests ? metrics.networkRequests.length : 0,
                            networkBytes: metrics.networkRequests ? 
                                metrics.networkRequests.reduce((sum, req) => sum + (parseInt(req.size) || 0), 0) : 0,
                            domNodes: document.querySelectorAll('*').length,
                            javascriptHeapMB: memoryInfo.usedJSHeapSize ? memoryInfo.usedJSHeapSize / 1024 / 1024 : 0,
                            interactions: metrics.interactions ? metrics.interactions.length : 0
                        };
                    }
                """)
                
                metrics.network_requests = browser_metrics.get('networkRequests', 0)
                metrics.network_bytes = browser_metrics.get('networkBytes', 0)
                metrics.dom_nodes = browser_metrics.get('domNodes', 0)
                metrics.javascript_heap_mb = browser_metrics.get('javascriptHeapMB', 0)
                
            except Exception as e:
                logger.warning(f"Failed to collect browser metrics: {e}")
        
        metrics.metadata.update({
            'final_memory_mb': final_memory,
            'final_cpu_percent': final_cpu,
            'memory_delta_mb': final_memory - metrics.memory_usage_mb,
            'cpu_delta_percent': final_cpu - metrics.cpu_usage_percent
        })
        
        self.performance_metrics.append(metrics)
        return metrics
    
    async def test_dashboard_loading_performance(self) -> PerformanceMetrics:
        """Test dashboard initial loading performance"""
        metrics = self._start_performance_monitoring('dashboard_loading')
        
        try:
            # Navigate to dashboard
            start_time = time.time()
            await self.page.goto(f"{self.base_url}/dashboard")
            
            # Wait for critical components to load
            await self.page.wait_for_selector('[data-testid="causal-network-container"]', timeout=15000)
            await self.page.wait_for_selector('[data-testid="real-time-monitor"]', timeout=15000)
            await self.page.wait_for_selector('[data-testid="data-source-status"]', timeout=15000)
            
            load_time = (time.time() - start_time) * 1000
            
            # Get detailed performance metrics
            performance_data = await self.page.evaluate("""
                () => {
                    const navigation = performance.getEntriesByType('navigation')[0];
                    const paint = performance.getEntriesByType('paint');
                    
                    return {
                        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
                        loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
                        firstPaint: paint.find(p => p.name === 'first-paint')?.startTime || 0,
                        firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
                        largestContentfulPaint: performance.getEntriesByType('largest-contentful-paint')[0]?.startTime || 0
                    };
                }
            """)
            
            metrics.render_time_ms = load_time
            metrics.metadata.update({
                'dom_content_loaded_ms': performance_data.get('domContentLoaded', 0),
                'load_complete_ms': performance_data.get('loadComplete', 0),
                'first_paint_ms': performance_data.get('firstPaint', 0),
                'first_contentful_paint_ms': performance_data.get('firstContentfulPaint', 0),
                'largest_contentful_paint_ms': performance_data.get('largestContentfulPaint', 0)
            })
            
            # Check if loading meets performance threshold
            if load_time > self.thresholds['page_load_time_ms']:
                metrics.error_count += 1
                metrics.metadata['performance_warning'] = f"Load time {load_time:.0f}ms exceeds threshold {self.thresholds['page_load_time_ms']}ms"
            
        except Exception as e:
            metrics.error_count += 1
            metrics.metadata['error'] = str(e)
            logger.error(f"Dashboard loading test failed: {e}")
        
        return await self._end_performance_monitoring(metrics)
    
    async def test_causal_network_rendering_performance(self, node_count: int = 100, edge_count: int = 200) -> PerformanceMetrics:
        """Test causal network visualization rendering performance"""
        metrics = self._start_performance_monitoring(f'causal_network_rendering_{node_count}nodes')
        
        try:
            # Navigate to causal network view
            await self.page.goto(f"{self.base_url}/dashboard/causal-network")
            
            # Wait for network container
            await self.page.wait_for_selector('[data-testid="causal-network-svg"]', timeout=15000)
            
            # Simulate loading large network data
            start_time = time.time()
            
            # Inject large network data for testing
            await self.page.evaluate(f"""
                () => {{
                    // Generate test network data
                    const nodes = [];
                    const edges = [];
                    
                    for (let i = 0; i < {node_count}; i++) {{
                        nodes.push({{
                            id: `node_${{i}}`,
                            label: `Node ${{i}}`,
                            x: Math.random() * 800,
                            y: Math.random() * 600,
                            type: Math.random() > 0.5 ? 'seismic' : 'volcanic'
                        }});
                    }}
                    
                    for (let i = 0; i < {edge_count}; i++) {{
                        const source = Math.floor(Math.random() * {node_count});
                        const target = Math.floor(Math.random() * {node_count});
                        if (source !== target) {{
                            edges.push({{
                                source: `node_${{source}}`,
                                target: `node_${{target}}`,
                                strength: Math.random(),
                                confidence: Math.random()
                            }});
                        }}
                    }}
                    
                    // Trigger network rendering
                    if (window.updateCausalNetwork) {{
                        window.updateCausalNetwork({{ nodes, edges }});
                    }}
                    
                    return {{ nodeCount: nodes.length, edgeCount: edges.length }};
                }}
            """)
            
            # Wait for rendering to complete
            await asyncio.sleep(2)
            
            render_time = (time.time() - start_time) * 1000
            metrics.render_time_ms = render_time
            
            # Test interaction responsiveness
            interaction_start = time.time()
            
            # Click on network area
            await self.page.click('[data-testid="causal-network-svg"]')
            
            # Wait for interaction response
            await asyncio.sleep(0.1)
            
            interaction_time = (time.time() - interaction_start) * 1000
            metrics.interaction_delay_ms = interaction_time
            
            # Get rendering performance metrics
            rendering_metrics = await self.page.evaluate("""
                () => {
                    const svg = document.querySelector('[data-testid="causal-network-svg"]');
                    const nodes = svg ? svg.querySelectorAll('.network-node') : [];
                    const edges = svg ? svg.querySelectorAll('.network-edge') : [];
                    
                    return {
                        renderedNodes: nodes.length,
                        renderedEdges: edges.length,
                        svgElements: svg ? svg.children.length : 0
                    };
                }
            """)
            
            metrics.metadata.update({
                'target_nodes': node_count,
                'target_edges': edge_count,
                'rendered_nodes': rendering_metrics.get('renderedNodes', 0),
                'rendered_edges': rendering_metrics.get('renderedEdges', 0),
                'svg_elements': rendering_metrics.get('svgElements', 0),
                'render_time_ms': render_time,
                'interaction_delay_ms': interaction_time
            })
            
            # Check performance thresholds
            if render_time > self.thresholds['network_visualization_render_ms']:
                metrics.error_count += 1
                metrics.metadata['performance_warning'] = f"Render time {render_time:.0f}ms exceeds threshold"
            
            if interaction_time > self.thresholds['interaction_response_ms']:
                metrics.error_count += 1
                metrics.metadata['interaction_warning'] = f"Interaction delay {interaction_time:.0f}ms exceeds threshold"
            
        except Exception as e:
            metrics.error_count += 1
            metrics.metadata['error'] = str(e)
            logger.error(f"Network rendering test failed: {e}")
        
        return await self._end_performance_monitoring(metrics)
    
    async def test_real_time_data_streaming_performance(self, duration_seconds: int = 30) -> PerformanceMetrics:
        """Test real-time data streaming performance"""
        metrics = self._start_performance_monitoring(f'real_time_streaming_{duration_seconds}s')
        
        try:
            # Navigate to real-time view
            await self.page.goto(f"{self.base_url}/dashboard/real-time")
            
            # Wait for streaming components
            await self.page.wait_for_selector('[data-testid="real-time-stream"]', timeout=15000)
            
            # Monitor streaming performance
            start_time = time.time()
            initial_data_count = 0
            
            try:
                initial_data_count = await self.page.evaluate("""
                    () => document.querySelectorAll('[data-testid="stream-data-point"]').length
                """)
            except:
                pass
            
            # Monitor for specified duration
            data_updates = []
            memory_samples = []
            
            while time.time() - start_time < duration_seconds:
                await asyncio.sleep(1)
                
                # Sample current data count
                try:
                    current_data_count = await self.page.evaluate("""
                        () => document.querySelectorAll('[data-testid="stream-data-point"]').length
                    """)
                    data_updates.append(current_data_count)
                except:
                    data_updates.append(0)
                
                # Sample memory usage
                try:
                    memory_info = await self.page.evaluate("""
                        () => performance.memory ? performance.memory.usedJSHeapSize / 1024 / 1024 : 0
                    """)
                    memory_samples.append(memory_info)
                except:
                    memory_samples.append(0)
            
            total_duration = time.time() - start_time
            
            # Calculate streaming metrics
            final_data_count = data_updates[-1] if data_updates else 0
            new_data_points = final_data_count - initial_data_count
            data_rate = new_data_points / total_duration if total_duration > 0 else 0
            
            # Memory growth analysis
            memory_growth = memory_samples[-1] - memory_samples[0] if len(memory_samples) > 1 else 0
            memory_growth_rate = memory_growth / total_duration if total_duration > 0 else 0
            
            # Data update latency test
            latency_start = time.time()
            
            # Trigger manual data update
            try:
                await self.page.click('[data-testid="refresh-data-button"]')
                await self.page.wait_for_selector('[data-testid="data-updated-indicator"]', timeout=5000)
                update_latency = (time.time() - latency_start) * 1000
            except:
                update_latency = 5000  # Timeout
            
            metrics.metadata.update({
                'streaming_duration_s': total_duration,
                'initial_data_count': initial_data_count,
                'final_data_count': final_data_count,
                'new_data_points': new_data_points,
                'data_rate_per_second': data_rate,
                'memory_growth_mb': memory_growth,
                'memory_growth_rate_mb_per_s': memory_growth_rate,
                'update_latency_ms': update_latency,
                'memory_samples': len(memory_samples)
            })
            
            # Check performance thresholds
            if update_latency > self.thresholds['data_update_latency_ms']:
                metrics.error_count += 1
                metrics.metadata['latency_warning'] = f"Update latency {update_latency:.0f}ms exceeds threshold"
            
            if memory_growth_rate > 10:  # More than 10MB/s growth
                metrics.error_count += 1
                metrics.metadata['memory_warning'] = f"Memory growth rate {memory_growth_rate:.1f}MB/s too high"
            
        except Exception as e:
            metrics.error_count += 1
            metrics.metadata['error'] = str(e)
            logger.error(f"Real-time streaming test failed: {e}")
        
        return await self._end_performance_monitoring(metrics)
    
    async def test_memory_stress_performance(self, iterations: int = 10) -> PerformanceMetrics:
        """Test memory usage under stress conditions"""
        metrics = self._start_performance_monitoring(f'memory_stress_{iterations}iterations')
        
        try:
            await self.page.goto(f"{self.base_url}/dashboard")
            
            memory_samples = []
            
            for i in range(iterations):
                logger.info(f"Memory stress iteration {i+1}/{iterations}")
                
                # Navigate between different views to stress memory
                views = [
                    '/dashboard/causal-network',
                    '/dashboard/real-time',
                    '/dashboard/volcanic-systems',
                    '/dashboard/data-quality'
                ]
                
                for view in views:
                    await self.page.goto(f"{self.base_url}{view}")
                    await self.page.wait_for_load_state('domcontentloaded')
                    
                    # Sample memory usage
                    try:
                        memory_info = await self.page.evaluate("""
                            () => {
                                if (performance.memory) {
                                    return {
                                        used: performance.memory.usedJSHeapSize / 1024 / 1024,
                                        total: performance.memory.totalJSHeapSize / 1024 / 1024,
                                        limit: performance.memory.jsHeapSizeLimit / 1024 / 1024
                                    };
                                }
                                return { used: 0, total: 0, limit: 0 };
                            }
                        """)
                        memory_samples.append(memory_info)
                    except:
                        memory_samples.append({'used': 0, 'total': 0, 'limit': 0})
                    
                    await asyncio.sleep(0.5)
                
                # Force garbage collection
                await self.page.evaluate("if (window.gc) window.gc();")
                await asyncio.sleep(1)
            
            # Analyze memory usage patterns
            if memory_samples:
                used_memory = [sample['used'] for sample in memory_samples]
                total_memory = [sample['total'] for sample in memory_samples]
                
                metrics.metadata.update({
                    'iterations': iterations,
                    'memory_samples_count': len(memory_samples),
                    'min_memory_used_mb': min(used_memory),
                    'max_memory_used_mb': max(used_memory),
                    'avg_memory_used_mb': statistics.mean(used_memory),
                    'memory_growth_mb': max(used_memory) - min(used_memory),
                    'memory_variance': statistics.variance(used_memory) if len(used_memory) > 1 else 0,
                    'peak_total_memory_mb': max(total_memory) if total_memory else 0
                })
                
                # Check for memory leaks
                if len(used_memory) > 5:
                    # Check if memory consistently grows
                    recent_avg = statistics.mean(used_memory[-3:])
                    early_avg = statistics.mean(used_memory[:3])
                    
                    if recent_avg > early_avg * 1.5:  # 50% growth
                        metrics.error_count += 1
                        metrics.metadata['memory_leak_warning'] = f"Potential memory leak detected: {early_avg:.1f}MB -> {recent_avg:.1f}MB"
                
                # Check memory threshold
                if max(used_memory) > self.thresholds['memory_usage_mb']:
                    metrics.error_count += 1
                    metrics.metadata['memory_threshold_warning'] = f"Peak memory {max(used_memory):.1f}MB exceeds threshold"
            
        except Exception as e:
            metrics.error_count += 1
            metrics.metadata['error'] = str(e)
            logger.error(f"Memory stress test failed: {e}")
        
        return await self._end_performance_monitoring(metrics)
    
    async def test_concurrent_user_simulation(self, concurrent_users: int = 10, duration_seconds: int = 60) -> LoadTestResult:
        """Simulate concurrent users accessing the dashboard"""
        logger.info(f"Starting concurrent user simulation: {concurrent_users} users for {duration_seconds}s")
        
        start_time = time.time()
        user_tasks = []
        
        # Create tasks for concurrent users
        for user_id in range(concurrent_users):
            task = asyncio.create_task(self._simulate_user_session(user_id, duration_seconds))
            user_tasks.append(task)
        
        # Wait for all user sessions to complete
        user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
        
        total_duration = time.time() - start_time
        
        # Analyze results
        successful_sessions = 0
        failed_sessions = 0
        total_requests = 0
        response_times = []
        errors = []
        
        for result in user_results:
            if isinstance(result, Exception):
                failed_sessions += 1
                errors.append(str(result))
            else:
                successful_sessions += 1
                total_requests += result.get('requests', 0)
                response_times.extend(result.get('response_times', []))
        
        # Calculate statistics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        median_response_time = statistics.median(response_times) if response_times else 0
        
        # Calculate percentiles
        if response_times:
            sorted_times = sorted(response_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
            p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else sorted_times[-1]
        else:
            p95_response_time = 0
            p99_response_time = 0
        
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        
        # Get system resource usage
        process = psutil.Process()
        memory_peak = process.memory_info().rss / 1024 / 1024
        cpu_peak = process.cpu_percent()
        
        result = LoadTestResult(
            test_name=f'concurrent_users_{concurrent_users}',
            concurrent_users=concurrent_users,
            duration_seconds=total_duration,
            total_requests=total_requests,
            successful_requests=successful_sessions,
            failed_requests=failed_sessions,
            average_response_time_ms=avg_response_time,
            median_response_time_ms=median_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            requests_per_second=requests_per_second,
            errors=errors,
            memory_peak_mb=memory_peak,
            cpu_peak_percent=cpu_peak
        )
        
        self.load_test_results.append(result)
        
        logger.info(f"Concurrent user test completed: {successful_sessions}/{concurrent_users} successful sessions")
        
        return result
    
    async def _simulate_user_session(self, user_id: int, duration_seconds: int) -> Dict[str, Any]:
        """Simulate a single user session"""
        session_start = time.time()
        requests = 0
        response_times = []
        
        try:
            # Create separate browser context for this user
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            # User behavior simulation
            while time.time() - session_start < duration_seconds:
                # Random navigation
                views = [
                    '/dashboard',
                    '/dashboard/causal-network',
                    '/dashboard/real-time',
                    '/dashboard/volcanic-systems'
                ]
                
                view = np.random.choice(views)
                
                request_start = time.time()
                await page.goto(f"{self.base_url}{view}")
                await page.wait_for_load_state('domcontentloaded')
                request_time = (time.time() - request_start) * 1000
                
                requests += 1
                response_times.append(request_time)
                
                # Random interactions
                try:
                    # Click random elements
                    clickable_elements = await page.query_selector_all('button, [data-testid*="button"], .clickable')
                    if clickable_elements:
                        element = np.random.choice(clickable_elements)
                        await element.click()
                        await asyncio.sleep(0.5)
                except:
                    pass
                
                # Random wait time between actions
                await asyncio.sleep(np.random.uniform(1, 3))
            
            await page.close()
            await context.close()
            await browser.close()
            
            return {
                'user_id': user_id,
                'requests': requests,
                'response_times': response_times,
                'session_duration': time.time() - session_start
            }
            
        except Exception as e:
            logger.error(f"User {user_id} session failed: {e}")
            raise
    
    async def run_comprehensive_performance_tests(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite"""
        logger.info("Starting comprehensive performance tests")
        
        await self.setup_browser()
        
        try:
            # Dashboard loading performance
            loading_metrics = await self.test_dashboard_loading_performance()
            
            # Network rendering performance (different sizes)
            network_small = await self.test_causal_network_rendering_performance(50, 100)
            network_medium = await self.test_causal_network_rendering_performance(200, 400)
            network_large = await self.test_causal_network_rendering_performance(500, 1000)
            
            # Real-time streaming performance
            streaming_metrics = await self.test_real_time_data_streaming_performance(30)
            
            # Memory stress test
            memory_metrics = await self.test_memory_stress_performance(5)
            
            # Concurrent user tests
            concurrent_5 = await self.test_concurrent_user_simulation(5, 30)
            concurrent_10 = await self.test_concurrent_user_simulation(10, 30)
            concurrent_20 = await self.test_concurrent_user_simulation(20, 30)
            
        finally:
            await self.cleanup_browser()
        
        # Generate comprehensive report
        report = self._generate_performance_report()
        
        # Save report
        report_file = self.output_dir / f"performance_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance tests completed. Report saved to {report_file}")
        
        return report
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Analyze performance metrics
        total_tests = len(self.performance_metrics)
        failed_tests = sum(1 for m in self.performance_metrics if m.error_count > 0)
        
        # Performance statistics
        load_times = [m.render_time_ms for m in self.performance_metrics if m.render_time_ms > 0]
        memory_usage = [m.javascript_heap_mb for m in self.performance_metrics if m.javascript_heap_mb > 0]
        interaction_delays = [m.interaction_delay_ms for m in self.performance_metrics if m.interaction_delay_ms > 0]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_performance_tests': total_tests,
                'failed_tests': failed_tests,
                'success_rate': (total_tests - failed_tests) / total_tests if total_tests > 0 else 0,
                'average_load_time_ms': statistics.mean(load_times) if load_times else 0,
                'average_memory_usage_mb': statistics.mean(memory_usage) if memory_usage else 0,
                'average_interaction_delay_ms': statistics.mean(interaction_delays) if interaction_delays else 0
            },
            'performance_metrics': [
                {
                    'operation': m.operation_name,
                    'duration_ms': m.duration_ms,
                    'render_time_ms': m.render_time_ms,
                    'memory_usage_mb': m.javascript_heap_mb,
                    'interaction_delay_ms': m.interaction_delay_ms,
                    'error_count': m.error_count,
                    'metadata': m.metadata
                }
                for m in self.performance_metrics
            ],
            'load_test_results': [
                {
                    'test_name': r.test_name,
                    'concurrent_users': r.concurrent_users,
                    'success_rate': r.successful_requests / (r.successful_requests + r.failed_requests) if (r.successful_requests + r.failed_requests) > 0 else 0,
                    'average_response_time_ms': r.average_response_time_ms,
                    'p95_response_time_ms': r.p95_response_time_ms,
                    'requests_per_second': r.requests_per_second,
                    'memory_peak_mb': r.memory_peak_mb
                }
                for r in self.load_test_results
            ],
            'threshold_analysis': self._analyze_thresholds(),
            'recommendations': self._generate_performance_recommendations()
        }
        
        return report
    
    def _analyze_thresholds(self) -> Dict[str, Any]:
        """Analyze performance against thresholds"""
        threshold_analysis = {}
        
        for metric in self.performance_metrics:
            violations = []
            
            if metric.render_time_ms > self.thresholds['page_load_time_ms']:
                violations.append(f"Load time {metric.render_time_ms:.0f}ms > {self.thresholds['page_load_time_ms']}ms")
            
            if metric.interaction_delay_ms > self.thresholds['interaction_response_ms']:
                violations.append(f"Interaction delay {metric.interaction_delay_ms:.0f}ms > {self.thresholds['interaction_response_ms']}ms")
            
            if metric.javascript_heap_mb > self.thresholds['memory_usage_mb']:
                violations.append(f"Memory usage {metric.javascript_heap_mb:.1f}MB > {self.thresholds['memory_usage_mb']}MB")
            
            if violations:
                threshold_analysis[metric.operation_name] = violations
        
        return threshold_analysis
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Analyze load times
        slow_operations = [m for m in self.performance_metrics if m.render_time_ms > self.thresholds['page_load_time_ms']]
        if slow_operations:
            recommendations.append(f"Optimize loading performance for: {', '.join(m.operation_name for m in slow_operations)}")
        
        # Analyze memory usage
        high_memory_operations = [m for m in self.performance_metrics if m.javascript_heap_mb > self.thresholds['memory_usage_mb']]
        if high_memory_operations:
            recommendations.append(f"Reduce memory usage for: {', '.join(m.operation_name for m in high_memory_operations)}")
        
        # Analyze interaction delays
        slow_interactions = [m for m in self.performance_metrics if m.interaction_delay_ms > self.thresholds['interaction_response_ms']]
        if slow_interactions:
            recommendations.append(f"Improve interaction responsiveness for: {', '.join(m.operation_name for m in slow_interactions)}")
        
        # Analyze load test results
        failed_load_tests = [r for r in self.load_test_results if r.successful_requests / (r.successful_requests + r.failed_requests) < 0.95]
        if failed_load_tests:
            recommendations.append("Improve system stability under concurrent load")
        
        # Check for memory leaks
        memory_leak_operations = [m for m in self.performance_metrics if 'memory_leak_warning' in m.metadata]
        if memory_leak_operations:
            recommendations.append("Investigate potential memory leaks in dashboard components")
        
        return recommendations


# CLI interface
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run dashboard performance tests')
    parser.add_argument('--base-url', default='http://localhost:3000', help='Dashboard base URL')
    parser.add_argument('--output-dir', default='performance_results', help='Output directory')
    parser.add_argument('--concurrent-users', type=int, default=10, help='Number of concurrent users for load test')
    parser.add_argument('--test-duration', type=int, default=60, help='Test duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = DashboardPerformanceTestSuite(
        base_url=args.base_url,
        output_dir=args.output_dir
    )
    
    try:
        # Run comprehensive tests
        report = await test_suite.run_comprehensive_performance_tests()
        
        print("\n" + "="*80)
        print("DASHBOARD PERFORMANCE TEST RESULTS")
        print("="*80)
        print(f"Total Tests: {report['summary']['total_performance_tests']}")
        print(f"Success Rate: {report['summary']['success_rate']:.2%}")
        print(f"Average Load Time: {report['summary']['average_load_time_ms']:.0f}ms")
        print(f"Average Memory Usage: {report['summary']['average_memory_usage_mb']:.1f}MB")
        print(f"Average Interaction Delay: {report['summary']['average_interaction_delay_ms']:.0f}ms")
        
        if report.get('recommendations'):
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print(f"\nDetailed report saved to: {test_suite.output_dir}")
        
    except Exception as e:
        print(f"Performance tests failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))