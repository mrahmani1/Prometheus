"""
Playwright Tests for Real Data Integration Dashboard

End-to-end testing for the causal dashboard interface with real data integration.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlaywrightRealDataTests:
    """
    Playwright-based tests for real data integration dashboard
    """
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.base_url = "http://localhost:3000"  # React development server
        self.api_base_url = "http://localhost:8000"  # FastAPI backend
        
    async def setup_browser(self, headless: bool = True):
        """Setup Playwright browser"""
        playwright = await async_playwright().start()
        
        # Launch browser with specific configuration
        self.browser = await playwright.chromium.launch(
            headless=headless,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--allow-running-insecure-content'
            ]
        )
        
        # Create browser context with viewport and permissions
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            permissions=['geolocation'],
            geolocation={'latitude': 45.0, 'longitude': -120.0}
        )
        
        # Enable request/response logging
        self.context.on('request', self._log_request)
        self.context.on('response', self._log_response)
        
        # Create new page
        self.page = await self.context.new_page()
        
        # Set longer timeout for data loading
        self.page.set_default_timeout(30000)
        
    async def cleanup_browser(self):
        """Cleanup browser resources"""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
    
    def _log_request(self, request):
        """Log HTTP requests"""
        if '/api/' in request.url:
            logger.info(f"Request: {request.method} {request.url}")
    
    def _log_response(self, response):
        """Log HTTP responses"""
        if '/api/' in response.url and response.status >= 400:
            logger.error(f"Response Error: {response.status} {response.url}")
    
    async def test_dashboard_loading_with_real_data(self) -> Dict[str, Any]:
        """Test dashboard loading with real data sources"""
        test_result = {
            'test_name': 'dashboard_loading_real_data',
            'success': False,
            'start_time': datetime.now(),
            'metrics': {},
            'errors': []
        }
        
        try:
            # Navigate to dashboard
            start_time = time.time()
            await self.page.goto(f"{self.base_url}/dashboard")
            
            # Wait for main dashboard components to load
            await self.page.wait_for_selector('[data-testid="causal-network-container"]', timeout=15000)
            await self.page.wait_for_selector('[data-testid="real-time-monitor"]', timeout=15000)
            await self.page.wait_for_selector('[data-testid="data-source-status"]', timeout=15000)
            
            load_time = (time.time() - start_time) * 1000
            test_result['metrics']['page_load_time_ms'] = load_time
            
            # Check for real data indicators
            real_data_indicators = await self.page.query_selector_all('[data-testid*="real-data"]')
            test_result['metrics']['real_data_indicators_count'] = len(real_data_indicators)
            
            # Verify no placeholder/mock data warnings
            mock_warnings = await self.page.query_selector_all('[data-testid*="mock"], [data-testid*="placeholder"]')
            test_result['metrics']['mock_warnings_count'] = len(mock_warnings)
            
            if len(mock_warnings) > 0:
                test_result['errors'].append(f"Found {len(mock_warnings)} mock/placeholder data warnings")
            
            # Check data source connection status
            data_sources = await self.page.query_selector_all('[data-testid="data-source-status"] .status-indicator')
            connected_sources = 0
            
            for source in data_sources:
                status_class = await source.get_attribute('class')
                if 'connected' in status_class or 'online' in status_class:
                    connected_sources += 1
            
            test_result['metrics']['connected_data_sources'] = connected_sources
            test_result['metrics']['total_data_sources'] = len(data_sources)
            
            # Verify causal network has real data
            network_nodes = await self.page.query_selector_all('[data-testid="causal-network"] .network-node')
            test_result['metrics']['causal_network_nodes'] = len(network_nodes)
            
            if len(network_nodes) == 0:
                test_result['errors'].append("No causal network nodes found")
            
            # Check for real-time data updates
            initial_timestamp = await self.page.text_content('[data-testid="last-update-timestamp"]')
            await asyncio.sleep(5)  # Wait for potential updates
            updated_timestamp = await self.page.text_content('[data-testid="last-update-timestamp"]')
            
            test_result['metrics']['real_time_updates'] = initial_timestamp != updated_timestamp
            
            # Performance metrics
            performance_metrics = await self.page.evaluate("""
                () => {
                    const navigation = performance.getEntriesByType('navigation')[0];
                    return {
                        dom_content_loaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
                        load_complete: navigation.loadEventEnd - navigation.loadEventStart,
                        first_paint: performance.getEntriesByType('paint').find(p => p.name === 'first-paint')?.startTime || 0,
                        first_contentful_paint: performance.getEntriesByType('paint').find(p => p.name === 'first-contentful-paint')?.startTime || 0
                    };
                }
            """)
            
            test_result['metrics']['performance'] = performance_metrics
            
            # Success criteria
            success_criteria = [
                load_time < 10000,  # Page loads in under 10 seconds
                len(mock_warnings) == 0,  # No mock data warnings
                connected_sources > 0,  # At least one data source connected
                len(network_nodes) > 0  # Causal network has nodes
            ]
            
            test_result['success'] = all(success_criteria)
            
            if not test_result['success']:
                test_result['errors'].append("Failed success criteria check")
            
        except Exception as e:
            test_result['errors'].append(f"Test execution error: {str(e)}")
            logger.error(f"Dashboard loading test failed: {e}")
        
        test_result['end_time'] = datetime.now()
        test_result['duration_ms'] = (test_result['end_time'] - test_result['start_time']).total_seconds() * 1000
        
        return test_result
    
    async def test_causal_network_interaction(self) -> Dict[str, Any]:
        """Test causal network visualization interaction with real data"""
        test_result = {
            'test_name': 'causal_network_interaction',
            'success': False,
            'start_time': datetime.now(),
            'metrics': {},
            'errors': []
        }
        
        try:
            # Navigate to causal network view
            await self.page.goto(f"{self.base_url}/dashboard/causal-network")
            
            # Wait for network to load
            await self.page.wait_for_selector('[data-testid="causal-network-svg"]', timeout=15000)
            
            # Get initial network state
            initial_nodes = await self.page.query_selector_all('.network-node')
            initial_edges = await self.page.query_selector_all('.network-edge')
            
            test_result['metrics']['initial_nodes'] = len(initial_nodes)
            test_result['metrics']['initial_edges'] = len(initial_edges)
            
            # Test node selection
            if len(initial_nodes) > 0:
                first_node = initial_nodes[0]
                await first_node.click()
                
                # Check if node details panel appears
                await self.page.wait_for_selector('[data-testid="node-details-panel"]', timeout=5000)
                
                # Verify real data in details panel
                node_data = await self.page.text_content('[data-testid="node-data-content"]')
                test_result['metrics']['node_has_real_data'] = 'mock' not in node_data.lower() and 'placeholder' not in node_data.lower()
            
            # Test network filtering
            filter_button = await self.page.query_selector('[data-testid="network-filter-button"]')
            if filter_button:
                await filter_button.click()
                
                # Apply seismic data filter
                seismic_filter = await self.page.query_selector('[data-testid="filter-seismic"]')
                if seismic_filter:
                    await seismic_filter.click()
                    
                    # Wait for network update
                    await asyncio.sleep(2)
                    
                    # Check filtered results
                    filtered_nodes = await self.page.query_selector_all('.network-node')
                    test_result['metrics']['filtered_nodes'] = len(filtered_nodes)
                    test_result['metrics']['filter_working'] = len(filtered_nodes) != len(initial_nodes)
            
            # Test zoom and pan functionality
            network_svg = await self.page.query_selector('[data-testid="causal-network-svg"]')
            if network_svg:
                # Get initial transform
                initial_transform = await network_svg.get_attribute('transform')
                
                # Simulate zoom in
                await network_svg.hover()
                await self.page.mouse.wheel(0, -100)  # Scroll up to zoom in
                
                await asyncio.sleep(1)
                
                # Check if transform changed
                new_transform = await network_svg.get_attribute('transform')
                test_result['metrics']['zoom_working'] = initial_transform != new_transform
            
            # Test real-time network updates
            initial_update_count = await self.page.evaluate("""
                () => window.networkUpdateCount || 0
            """)
            
            await asyncio.sleep(10)  # Wait for potential updates
            
            final_update_count = await self.page.evaluate("""
                () => window.networkUpdateCount || 0
            """)
            
            test_result['metrics']['network_updates_received'] = final_update_count - initial_update_count
            
            # Test causal strength visualization
            edges_with_strength = await self.page.query_selector_all('.network-edge[data-strength]')
            test_result['metrics']['edges_with_strength'] = len(edges_with_strength)
            
            # Verify strength values are realistic (not mock data)
            if len(edges_with_strength) > 0:
                strength_values = []
                for edge in edges_with_strength[:5]:  # Check first 5 edges
                    strength = await edge.get_attribute('data-strength')
                    try:
                        strength_val = float(strength)
                        strength_values.append(strength_val)
                    except:
                        pass
                
                test_result['metrics']['realistic_strength_values'] = all(0 <= s <= 1 for s in strength_values)
            
            # Success criteria
            success_criteria = [
                len(initial_nodes) > 0,  # Network has nodes
                len(initial_edges) > 0,  # Network has edges
                test_result['metrics'].get('node_has_real_data', False),  # Nodes contain real data
                test_result['metrics'].get('edges_with_strength', 0) > 0  # Edges have strength values
            ]
            
            test_result['success'] = all(success_criteria)
            
        except Exception as e:
            test_result['errors'].append(f"Test execution error: {str(e)}")
            logger.error(f"Causal network interaction test failed: {e}")
        
        test_result['end_time'] = datetime.now()
        test_result['duration_ms'] = (test_result['end_time'] - test_result['start_time']).total_seconds() * 1000
        
        return test_result
    
    async def test_real_time_data_streaming(self) -> Dict[str, Any]:
        """Test real-time data streaming in dashboard"""
        test_result = {
            'test_name': 'real_time_data_streaming',
            'success': False,
            'start_time': datetime.now(),
            'metrics': {},
            'errors': []
        }
        
        try:
            # Navigate to real-time monitoring view
            await self.page.goto(f"{self.base_url}/dashboard/real-time")
            
            # Wait for streaming components to load
            await self.page.wait_for_selector('[data-testid="real-time-stream"]', timeout=15000)
            
            # Monitor data stream for updates
            initial_data_count = await self.page.evaluate("""
                () => document.querySelectorAll('[data-testid="stream-data-point"]').length
            """)
            
            # Wait for streaming updates
            await asyncio.sleep(15)
            
            final_data_count = await self.page.evaluate("""
                () => document.querySelectorAll('[data-testid="stream-data-point"]').length
            """)
            
            test_result['metrics']['initial_data_points'] = initial_data_count
            test_result['metrics']['final_data_points'] = final_data_count
            test_result['metrics']['new_data_received'] = final_data_count > initial_data_count
            
            # Check streaming indicators
            streaming_indicators = await self.page.query_selector_all('[data-testid="streaming-active"]')
            test_result['metrics']['active_streams'] = len(streaming_indicators)
            
            # Verify data freshness
            timestamps = await self.page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('[data-timestamp]');
                    return Array.from(elements).map(el => el.getAttribute('data-timestamp'));
                }
            """)
            
            if timestamps:
                latest_timestamp = max(timestamps)
                current_time = datetime.now()
                try:
                    data_time = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
                    age_seconds = (current_time - data_time).total_seconds()
                    test_result['metrics']['data_age_seconds'] = age_seconds
                    test_result['metrics']['data_is_fresh'] = age_seconds < 300  # Less than 5 minutes old
                except:
                    test_result['metrics']['data_is_fresh'] = False
            
            # Test stream controls
            pause_button = await self.page.query_selector('[data-testid="stream-pause-button"]')
            if pause_button:
                await pause_button.click()
                
                # Wait and check if streaming stopped
                await asyncio.sleep(5)
                
                paused_data_count = await self.page.evaluate("""
                    () => document.querySelectorAll('[data-testid="stream-data-point"]').length
                """)
                
                test_result['metrics']['stream_pause_working'] = paused_data_count == final_data_count
                
                # Resume streaming
                resume_button = await self.page.query_selector('[data-testid="stream-resume-button"]')
                if resume_button:
                    await resume_button.click()
            
            # Check for data source health indicators
            health_indicators = await self.page.query_selector_all('[data-testid="source-health-indicator"]')
            healthy_sources = 0
            
            for indicator in health_indicators:
                status = await indicator.get_attribute('data-status')
                if status == 'healthy' or status == 'connected':
                    healthy_sources += 1
            
            test_result['metrics']['healthy_data_sources'] = healthy_sources
            test_result['metrics']['total_monitored_sources'] = len(health_indicators)
            
            # Test alert generation
            alerts = await self.page.query_selector_all('[data-testid="real-time-alert"]')
            test_result['metrics']['active_alerts'] = len(alerts)
            
            # Verify alerts contain real data
            if len(alerts) > 0:
                first_alert = alerts[0]
                alert_content = await first_alert.text_content()
                test_result['metrics']['alerts_have_real_data'] = 'mock' not in alert_content.lower()
            
            # Success criteria
            success_criteria = [
                test_result['metrics'].get('new_data_received', False),  # New data received
                test_result['metrics'].get('active_streams', 0) > 0,  # Active streams present
                test_result['metrics'].get('data_is_fresh', False),  # Data is fresh
                test_result['metrics'].get('healthy_data_sources', 0) > 0  # Healthy sources
            ]
            
            test_result['success'] = all(success_criteria)
            
        except Exception as e:
            test_result['errors'].append(f"Test execution error: {str(e)}")
            logger.error(f"Real-time streaming test failed: {e}")
        
        test_result['end_time'] = datetime.now()
        test_result['duration_ms'] = (test_result['end_time'] - test_result['start_time']).total_seconds() * 1000
        
        return test_result
    
    async def test_data_quality_indicators(self) -> Dict[str, Any]:
        """Test data quality indicators in dashboard"""
        test_result = {
            'test_name': 'data_quality_indicators',
            'success': False,
            'start_time': datetime.now(),
            'metrics': {},
            'errors': []
        }
        
        try:
            # Navigate to data quality view
            await self.page.goto(f"{self.base_url}/dashboard/data-quality")
            
            # Wait for quality indicators to load
            await self.page.wait_for_selector('[data-testid="quality-dashboard"]', timeout=15000)
            
            # Check completeness indicators
            completeness_scores = await self.page.query_selector_all('[data-testid="completeness-score"]')
            test_result['metrics']['completeness_indicators'] = len(completeness_scores)
            
            # Verify completeness scores are realistic
            completeness_values = []
            for score_element in completeness_scores:
                score_text = await score_element.text_content()
                try:
                    score = float(score_text.replace('%', '')) / 100
                    completeness_values.append(score)
                except:
                    pass
            
            if completeness_values:
                test_result['metrics']['average_completeness'] = sum(completeness_values) / len(completeness_values)
                test_result['metrics']['realistic_completeness'] = all(0 <= s <= 1 for s in completeness_values)
            
            # Check accuracy indicators
            accuracy_scores = await self.page.query_selector_all('[data-testid="accuracy-score"]')
            test_result['metrics']['accuracy_indicators'] = len(accuracy_scores)
            
            # Check timeliness indicators
            timeliness_scores = await self.page.query_selector_all('[data-testid="timeliness-score"]')
            test_result['metrics']['timeliness_indicators'] = len(timeliness_scores)
            
            # Test quality alerts
            quality_alerts = await self.page.query_selector_all('[data-testid="quality-alert"]')
            test_result['metrics']['quality_alerts'] = len(quality_alerts)
            
            # Check for quality trends
            trend_charts = await self.page.query_selector_all('[data-testid="quality-trend-chart"]')
            test_result['metrics']['quality_trend_charts'] = len(trend_charts)
            
            # Verify data source quality breakdown
            source_quality_panels = await self.page.query_selector_all('[data-testid="source-quality-panel"]')
            test_result['metrics']['source_quality_panels'] = len(source_quality_panels)
            
            # Test quality filtering
            quality_filter = await self.page.query_selector('[data-testid="quality-filter"]')
            if quality_filter:
                await quality_filter.select_option('low_quality')
                await asyncio.sleep(2)
                
                low_quality_items = await self.page.query_selector_all('[data-testid="low-quality-item"]')
                test_result['metrics']['low_quality_items_found'] = len(low_quality_items)
            
            # Success criteria
            success_criteria = [
                test_result['metrics'].get('completeness_indicators', 0) > 0,
                test_result['metrics'].get('accuracy_indicators', 0) > 0,
                test_result['metrics'].get('timeliness_indicators', 0) > 0,
                test_result['metrics'].get('realistic_completeness', False)
            ]
            
            test_result['success'] = all(success_criteria)
            
        except Exception as e:
            test_result['errors'].append(f"Test execution error: {str(e)}")
            logger.error(f"Data quality indicators test failed: {e}")
        
        test_result['end_time'] = datetime.now()
        test_result['duration_ms'] = (test_result['end_time'] - test_result['start_time']).total_seconds() * 1000
        
        return test_result
    
    async def test_performance_under_load(self) -> Dict[str, Any]:
        """Test dashboard performance under load"""
        test_result = {
            'test_name': 'performance_under_load',
            'success': False,
            'start_time': datetime.now(),
            'metrics': {},
            'errors': []
        }
        
        try:
            # Navigate to dashboard
            await self.page.goto(f"{self.base_url}/dashboard")
            
            # Wait for initial load
            await self.page.wait_for_load_state('networkidle')
            
            # Simulate heavy interaction load
            start_time = time.time()
            
            # Rapid navigation between views
            views = [
                '/dashboard/causal-network',
                '/dashboard/real-time',
                '/dashboard/volcanic-systems',
                '/dashboard/data-quality',
                '/dashboard/multi-scale'
            ]
            
            navigation_times = []
            for view in views:
                nav_start = time.time()
                await self.page.goto(f"{self.base_url}{view}")
                await self.page.wait_for_load_state('domcontentloaded')
                nav_time = (time.time() - nav_start) * 1000
                navigation_times.append(nav_time)
            
            test_result['metrics']['navigation_times_ms'] = navigation_times
            test_result['metrics']['average_navigation_time_ms'] = sum(navigation_times) / len(navigation_times)
            test_result['metrics']['max_navigation_time_ms'] = max(navigation_times)
            
            # Test memory usage
            memory_info = await self.page.evaluate("""
                () => {
                    if (performance.memory) {
                        return {
                            used: performance.memory.usedJSHeapSize,
                            total: performance.memory.totalJSHeapSize,
                            limit: performance.memory.jsHeapSizeLimit
                        };
                    }
                    return null;
                }
            """)
            
            if memory_info:
                test_result['metrics']['memory_usage_mb'] = memory_info['used'] / 1024 / 1024
                test_result['metrics']['memory_usage_percent'] = (memory_info['used'] / memory_info['limit']) * 100
            
            # Test rapid interactions
            await self.page.goto(f"{self.base_url}/dashboard/causal-network")
            await self.page.wait_for_selector('[data-testid="causal-network-svg"]')
            
            # Rapid zoom operations
            network_svg = await self.page.query_selector('[data-testid="causal-network-svg"]')
            if network_svg:
                await network_svg.hover()
                
                zoom_start = time.time()
                for _ in range(10):
                    await self.page.mouse.wheel(0, -50)  # Zoom in
                    await asyncio.sleep(0.1)
                    await self.page.mouse.wheel(0, 50)   # Zoom out
                    await asyncio.sleep(0.1)
                
                zoom_time = (time.time() - zoom_start) * 1000
                test_result['metrics']['zoom_interaction_time_ms'] = zoom_time
            
            # Test concurrent data loading
            concurrent_start = time.time()
            
            # Open multiple data panels simultaneously
            await self.page.evaluate("""
                () => {
                    // Simulate opening multiple data panels
                    const buttons = document.querySelectorAll('[data-testid*="open-panel"]');
                    buttons.forEach(button => button.click());
                }
            """)
            
            await asyncio.sleep(3)  # Wait for panels to load
            
            concurrent_time = (time.time() - concurrent_start) * 1000
            test_result['metrics']['concurrent_loading_time_ms'] = concurrent_time
            
            # Check for performance warnings
            performance_warnings = await self.page.query_selector_all('[data-testid="performance-warning"]')
            test_result['metrics']['performance_warnings'] = len(performance_warnings)
            
            # Test responsiveness during heavy load
            responsiveness_start = time.time()
            
            # Click a button and measure response time
            test_button = await self.page.query_selector('[data-testid="test-interaction-button"]')
            if test_button:
                await test_button.click()
                await self.page.wait_for_selector('[data-testid="interaction-response"]', timeout=5000)
                
                responsiveness_time = (time.time() - responsiveness_start) * 1000
                test_result['metrics']['responsiveness_time_ms'] = responsiveness_time
            
            total_test_time = (time.time() - start_time) * 1000
            test_result['metrics']['total_test_time_ms'] = total_test_time
            
            # Success criteria
            success_criteria = [
                test_result['metrics'].get('average_navigation_time_ms', float('inf')) < 3000,  # < 3s navigation
                test_result['metrics'].get('memory_usage_percent', 100) < 80,  # < 80% memory usage
                test_result['metrics'].get('performance_warnings', 1) == 0,  # No performance warnings
                test_result['metrics'].get('responsiveness_time_ms', float('inf')) < 1000  # < 1s response
            ]
            
            test_result['success'] = all(success_criteria)
            
        except Exception as e:
            test_result['errors'].append(f"Test execution error: {str(e)}")
            logger.error(f"Performance under load test failed: {e}")
        
        test_result['end_time'] = datetime.now()
        test_result['duration_ms'] = (test_result['end_time'] - test_result['start_time']).total_seconds() * 1000
        
        return test_result
    
    async def run_all_playwright_tests(self, headless: bool = True) -> Dict[str, Any]:
        """Run all Playwright tests"""
        logger.info("Starting Playwright real data integration tests")
        
        await self.setup_browser(headless=headless)
        
        test_results = []
        
        try:
            # Run all tests
            tests = [
                self.test_dashboard_loading_with_real_data,
                self.test_causal_network_interaction,
                self.test_real_time_data_streaming,
                self.test_data_quality_indicators,
                self.test_performance_under_load
            ]
            
            for test_func in tests:
                logger.info(f"Running test: {test_func.__name__}")
                result = await test_func()
                test_results.append(result)
                logger.info(f"Test {test_func.__name__} completed: {'PASS' if result['success'] else 'FAIL'}")
            
        finally:
            await self.cleanup_browser()
        
        # Generate summary report
        total_tests = len(test_results)
        successful_tests = sum(1 for r in test_results if r['success'])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': total_tests - successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0
            },
            'test_results': test_results,
            'recommendations': []
        }
        
        # Generate recommendations
        for result in test_results:
            if not result['success']:
                report['recommendations'].append(
                    f"Fix issues in {result['test_name']}: {', '.join(result['errors'])}"
                )
        
        logger.info(f"Playwright tests completed. Success rate: {report['summary']['success_rate']:.2%}")
        
        return report


# Test execution functions
async def run_playwright_dashboard_tests():
    """Run Playwright dashboard tests"""
    test_runner = PlaywrightRealDataTests()
    return await test_runner.run_all_playwright_tests(headless=True)


async def run_playwright_visual_tests():
    """Run Playwright tests with visual browser for debugging"""
    test_runner = PlaywrightRealDataTests()
    return await test_runner.run_all_playwright_tests(headless=False)


if __name__ == "__main__":
    # Run Playwright tests
    async def main():
        test_runner = PlaywrightRealDataTests()
        report = await test_runner.run_all_playwright_tests()
        
        # Save report
        with open('playwright_real_data_test_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Playwright tests completed. Report saved to playwright_real_data_test_report.json")
        print(f"Success rate: {report['summary']['success_rate']:.2%}")
    
    asyncio.run(main())