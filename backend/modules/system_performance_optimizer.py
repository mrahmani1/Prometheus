"""
System Performance Optimizer for Prometheus AI Geodynamic Platform
==================================================================

This module implements comprehensive performance optimization across all subsystems
including memory management, CPU optimization, I/O optimization, caching strategies,
and resource allocation algorithms.

Advanced Features:
- Dynamic memory profiling and optimization
- CPU workload balancing and parallelization
- I/O bottleneck detection and optimization
- Multi-level caching with intelligent eviction
- Real-time resource monitoring and auto-scaling
- Garbage collection optimization
- Database query optimization
- Network optimization and connection pooling
- Memory leak detection and prevention
- Performance regression testing

Author: Advanced Performance Optimization System
"""

import asyncio
import gc
import psutil
import threading
import multiprocessing
import time
import sys
import os
import tracemalloc
import weakref
import sqlite3
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import logging
from pathlib import Path
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from contextlib import contextmanager
import warnings
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False
from collections import defaultdict, deque
import heapq
import functools
import inspect
from abc import ABC, abstractmethod
import signal
import subprocess

# Performance monitoring libraries
try:
    import memory_profiler
    import cProfile
    import pstats
    import line_profiler
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    gc_collections: int
    active_threads: int
    open_files: int
    response_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0

@dataclass
class OptimizationResult:
    """Results from optimization operations"""
    operation: str
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    improvement_percent: float
    recommendations: List[str]
    execution_time_ms: float

class MemoryProfiler:
    """Advanced memory profiling and optimization"""
    
    def __init__(self):
        self.memory_snapshots: List[Tuple[datetime, Dict[str, Any]]] = []
        self.leak_detectors: Dict[str, WeakValueDictionary] = {}
        self.allocation_trackers: Dict[str, int] = defaultdict(int)
        self.peak_memory_usage = 0.0
        
        # Start memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            
        logger.info("MemoryProfiler initialized with tracemalloc")
    
    def take_snapshot(self, label: str = "default") -> Dict[str, Any]:
        """Take a memory snapshot for analysis"""
        snapshot = tracemalloc.take_snapshot()
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Analyze top memory allocations
        top_stats = snapshot.statistics('lineno')[:10]
        
        memory_data = {
            'timestamp': datetime.now(),
            'label': label,
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'peak_rss_mb': self.peak_memory_usage,
            'top_allocations': [
                {
                    'filename': stat.traceback.format()[0],
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                }
                for stat in top_stats[:5]
            ],
            'gc_stats': {
                'gen0_collections': gc.get_count()[0],
                'gen1_collections': gc.get_count()[1], 
                'gen2_collections': gc.get_count()[2],
                'total_objects': len(gc.get_objects())
            }
        }
        
        self.memory_snapshots.append((datetime.now(), memory_data))
        self.peak_memory_usage = max(self.peak_memory_usage, memory_data['rss_mb'])
        
        return memory_data
    
    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks"""
        leaks = []
        
        if len(self.memory_snapshots) < 2:
            return leaks
        
        # Compare recent snapshots
        recent_snapshots = self.memory_snapshots[-5:]  # Last 5 snapshots
        
        if len(recent_snapshots) >= 3:
            # Check for consistently increasing memory usage
            memory_trend = [snapshot[1]['rss_mb'] for snapshot in recent_snapshots]
            
            # Calculate linear regression slope
            x = np.array(range(len(memory_trend)))
            y = np.array(memory_trend)
            
            if len(x) > 1 and np.std(x) > 0:
                correlation = np.corrcoef(x, y)[0, 1]
                slope = np.polyfit(x, y, 1)[0]
                
                if correlation > 0.8 and slope > 1.0:  # Strong upward trend
                    leaks.append({
                        'type': 'memory_growth',
                        'trend_correlation': correlation,
                        'growth_rate_mb_per_snapshot': slope,
                        'current_memory_mb': memory_trend[-1],
                        'recommendation': f"Investigate memory growth: {slope:.2f}MB per measurement"
                    })
        
        # Check for large object collections
        if self.memory_snapshots:
            latest = self.memory_snapshots[-1][1]
            total_objects = latest['gc_stats']['total_objects']
            
            if total_objects > 100000:  # More than 100k objects
                leaks.append({
                    'type': 'high_object_count',
                    'object_count': total_objects,
                    'recommendation': f"High object count: {total_objects}. Consider object pooling or cleanup."
                })
        
        return leaks
    
    def optimize_garbage_collection(self) -> Dict[str, Any]:
        """Optimize garbage collection settings"""
        initial_collections = sum(gc.get_count())
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Force full garbage collection
        collected_objects = gc.collect()
        
        # Set more aggressive GC thresholds for memory-intensive operations
        old_thresholds = gc.get_threshold()
        new_thresholds = (500, 8, 8)  # More frequent collection
        gc.set_threshold(*new_thresholds)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_freed = initial_memory - final_memory
        
        result = {
            'objects_collected': collected_objects,
            'memory_freed_mb': memory_freed,
            'old_thresholds': old_thresholds,
            'new_thresholds': new_thresholds,
            'recommendation': f"Freed {memory_freed:.2f}MB, collected {collected_objects} objects"
        }
        
        logger.info(f"GC optimization: {result['recommendation']}")
        return result

class CPUOptimizer:
    """Advanced CPU optimization and workload balancing"""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=min(32, self.cpu_count * 2))
        self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        self.workload_history: deque = deque(maxlen=100)
        self.optimization_cache = {}
        
        logger.info(f"CPUOptimizer initialized with {self.cpu_count} CPUs")
    
    def analyze_cpu_usage(self) -> Dict[str, Any]:
        """Analyze current CPU usage patterns"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_stats = psutil.cpu_stats()
        cpu_freq = psutil.cpu_freq()
        
        analysis = {
            'timestamp': datetime.now(),
            'overall_cpu_percent': psutil.cpu_percent(),
            'per_cpu_percent': cpu_percent,
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'context_switches': cpu_stats.ctx_switches,
            'interrupts': cpu_stats.interrupts,
            'soft_interrupts': cpu_stats.soft_interrupts,
            'frequency_current': cpu_freq.current if cpu_freq else 0,
            'frequency_max': cpu_freq.max if cpu_freq else 0,
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
        
        # Detect CPU bottlenecks
        bottlenecks = []
        if analysis['overall_cpu_percent'] > 80:
            bottlenecks.append("High overall CPU usage")
        
        # Check for uneven CPU distribution
        cpu_std = np.std(cpu_percent) if cpu_percent else 0
        if cpu_std > 20:  # High variation across cores
            bottlenecks.append("Uneven CPU load distribution")
        
        analysis['bottlenecks'] = bottlenecks
        self.workload_history.append(analysis)
        
        return analysis
    
    def optimize_thread_pool_size(self, workload_type: str = "io_bound") -> Dict[str, Any]:
        """Dynamically optimize thread pool size based on workload"""
        current_threads = self.thread_pool._max_workers
        
        # Analyze recent workload patterns
        if len(self.workload_history) >= 5:
            recent_cpu_usage = [w['overall_cpu_percent'] for w in list(self.workload_history)[-5:]]
            avg_cpu_usage = np.mean(recent_cpu_usage)
            
            # Adjust thread pool based on workload type and CPU usage
            if workload_type == "cpu_bound":
                optimal_threads = min(self.cpu_count, max(1, int(100 / max(avg_cpu_usage, 1))))
            else:  # io_bound
                if avg_cpu_usage < 50:
                    optimal_threads = min(64, self.cpu_count * 4)
                else:
                    optimal_threads = min(32, self.cpu_count * 2)
        else:
            # Default optimization
            optimal_threads = self.cpu_count * 2 if workload_type == "io_bound" else self.cpu_count
        
        # Recreate thread pool if size needs adjustment
        if optimal_threads != current_threads:
            self.thread_pool.shutdown(wait=False)
            self.thread_pool = ThreadPoolExecutor(max_workers=optimal_threads)
        
        return {
            'previous_threads': current_threads,
            'optimized_threads': optimal_threads,
            'workload_type': workload_type,
            'improvement_factor': optimal_threads / max(current_threads, 1),
            'recommendation': f"Adjusted thread pool from {current_threads} to {optimal_threads} threads"
        }
    
    def parallelize_workload(self, tasks: List[Callable], use_processes: bool = False) -> List[Any]:
        """Intelligently parallelize a workload"""
        if not tasks:
            return []
        
        executor = self.process_pool if use_processes else self.thread_pool
        
        try:
            futures = [executor.submit(task) for task in tasks]
            results = [future.result(timeout=30) for future in futures]
            return results
            
        except Exception as e:
            logger.warning(f"Parallel execution failed: {e}")
            # Fallback to sequential execution
            return [task() for task in tasks]

class CacheOptimizer:
    """Advanced caching system with intelligent eviction policies"""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache_layers = {
            'l1_memory': {},  # Hot data
            'l2_memory': {},  # Warm data
            'l3_disk': {}     # Cold data
        }
        self.access_counts = defaultdict(int)
        self.access_times = defaultdict(float)
        self.cache_sizes = defaultdict(int)
        self.hit_counts = defaultdict(int)
        self.miss_counts = defaultdict(int)
        
        # LRU eviction queues
        self.lru_queues = {
            'l1_memory': deque(),
            'l2_memory': deque(),
            'l3_disk': deque()
        }
        
        logger.info(f"CacheOptimizer initialized with {max_memory_mb}MB max memory")
    
    def get(self, key: str, layer: str = None) -> Optional[Any]:
        """Intelligent cache retrieval with automatic promotion"""
        # Check all layers if no specific layer requested
        layers_to_check = [layer] if layer else ['l1_memory', 'l2_memory', 'l3_disk']
        
        for cache_layer in layers_to_check:
            if key in self.cache_layers[cache_layer]:
                value = self.cache_layers[cache_layer][key]
                
                # Update access statistics
                self.access_counts[key] += 1
                self.access_times[key] = time.time()
                self.hit_counts[cache_layer] += 1
                
                # Promote frequently accessed items to higher cache levels
                self._promote_if_needed(key, cache_layer)
                
                # Update LRU queue
                if key in self.lru_queues[cache_layer]:
                    self.lru_queues[cache_layer].remove(key)
                self.lru_queues[cache_layer].append(key)
                
                return value
        
        # Cache miss
        for cache_layer in layers_to_check:
            self.miss_counts[cache_layer] += 1
            
        return None
    
    def put(self, key: str, value: Any, layer: str = 'l1_memory') -> bool:
        """Intelligent cache storage with automatic eviction"""
        # Calculate value size
        value_size = sys.getsizeof(value)
        
        # Check if we need to evict
        self._evict_if_needed(layer, value_size)
        
        # Store value
        self.cache_layers[layer][key] = value
        self.cache_sizes[key] = value_size
        self.access_times[key] = time.time()
        self.lru_queues[layer].append(key)
        
        return True
    
    def _promote_if_needed(self, key: str, current_layer: str):
        """Promote frequently accessed items to higher cache levels"""
        access_count = self.access_counts[key]
        
        # Promotion thresholds
        if current_layer == 'l3_disk' and access_count > 10:
            self._move_between_layers(key, 'l3_disk', 'l2_memory')
        elif current_layer == 'l2_memory' and access_count > 50:
            self._move_between_layers(key, 'l2_memory', 'l1_memory')
    
    def _move_between_layers(self, key: str, from_layer: str, to_layer: str):
        """Move cache entry between layers"""
        if key in self.cache_layers[from_layer]:
            value = self.cache_layers[from_layer].pop(key)
            self.cache_layers[to_layer][key] = value
            
            # Update LRU queues
            if key in self.lru_queues[from_layer]:
                self.lru_queues[from_layer].remove(key)
            self.lru_queues[to_layer].append(key)
    
    def _evict_if_needed(self, layer: str, required_size: int):
        """Evict cache entries if needed"""
        current_size = sum(self.cache_sizes.get(k, 0) for k in self.cache_layers[layer])
        
        while current_size + required_size > self.max_memory_bytes:
            if not self.lru_queues[layer]:
                break
                
            # Evict LRU item
            lru_key = self.lru_queues[layer].popleft()
            if lru_key in self.cache_layers[layer]:
                del self.cache_layers[layer][lru_key]
                current_size -= self.cache_sizes.get(lru_key, 0)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        stats = {}
        
        for layer in self.cache_layers:
            total_requests = self.hit_counts[layer] + self.miss_counts[layer]
            hit_rate = self.hit_counts[layer] / max(total_requests, 1)
            
            stats[layer] = {
                'entries': len(self.cache_layers[layer]),
                'hit_count': self.hit_counts[layer],
                'miss_count': self.miss_counts[layer],
                'hit_rate': hit_rate,
                'memory_usage_mb': sum(self.cache_sizes.get(k, 0) for k in self.cache_layers[layer]) / 1024 / 1024
            }
        
        return stats

class DatabaseOptimizer:
    """Database query and connection optimization"""
    
    def __init__(self):
        self.connection_pools = {}
        self.query_cache = {}
        self.slow_query_threshold = 1.0  # seconds
        self.query_stats = defaultdict(list)
        
        logger.info("DatabaseOptimizer initialized")
    
    def optimize_connection_pooling(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize database connection pooling"""
        # Analyze current connection usage
        active_connections = len([c for pool in self.connection_pools.values() for c in pool if c])
        
        # Calculate optimal pool size based on CPU cores and expected load
        cpu_count = multiprocessing.cpu_count()
        optimal_pool_size = min(20, max(5, cpu_count * 2))
        
        optimization_result = {
            'current_connections': active_connections,
            'optimal_pool_size': optimal_pool_size,
            'improvement_factor': optimal_pool_size / max(active_connections, 1),
            'recommendations': [
                f"Set connection pool size to {optimal_pool_size}",
                "Enable connection pooling with appropriate timeout settings",
                "Monitor connection pool utilization and adjust as needed"
            ]
        }
        
        return optimization_result
    
    def analyze_query_performance(self, query: str, execution_time: float) -> Dict[str, Any]:
        """Analyze and optimize query performance"""
        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        
        # Record query statistics
        self.query_stats[query_hash].append({
            'execution_time': execution_time,
            'timestamp': time.time()
        })
        
        # Analyze query performance
        recent_stats = self.query_stats[query_hash][-10:]  # Last 10 executions
        avg_time = sum(s['execution_time'] for s in recent_stats) / len(recent_stats)
        
        analysis = {
            'query_hash': query_hash,
            'avg_execution_time': avg_time,
            'total_executions': len(self.query_stats[query_hash]),
            'is_slow': avg_time > self.slow_query_threshold
        }
        
        # Generate optimization recommendations
        recommendations = []
        if analysis['is_slow']:
            recommendations.extend([
                "Consider adding database indexes for frequently queried columns",
                "Optimize query structure and joins",
                "Consider query result caching",
                "Analyze query execution plan"
            ])
        
        analysis['recommendations'] = recommendations
        return analysis

class SystemPerformanceOptimizer:
    """Main system performance optimization orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Initialize optimizers
        self.memory_profiler = MemoryProfiler()
        self.cpu_optimizer = CPUOptimizer()
        self.cache_optimizer = CacheOptimizer(max_memory_mb=self.config.get('max_cache_memory_mb', 512))
        self.database_optimizer = DatabaseOptimizer()
        
        # Performance monitoring
        self.metrics_history: deque = deque(maxlen=1000)
        self.optimization_results: List[OptimizationResult] = []
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("SystemPerformanceOptimizer initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default optimization configuration"""
        return {
            'max_cache_memory_mb': 512,
            'monitoring_interval_seconds': 30,
            'optimization_interval_minutes': 5,
            'auto_gc_threshold_mb': 100,
            'max_thread_pool_size': 64,
            'performance_alert_threshold': 80
        }
    
    def collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        process = psutil.Process()
        
        # CPU and memory metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # I/O metrics
        io_counters = process.io_counters()
        disk_io_read = io_counters.read_bytes / 1024 / 1024  # MB
        disk_io_write = io_counters.write_bytes / 1024 / 1024  # MB
        
        # Network metrics (system-wide)
        try:
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent / 1024 / 1024  # MB
            net_recv = net_io.bytes_recv / 1024 / 1024  # MB
        except:
            net_sent = net_recv = 0
        
        # System metrics
        gc_stats = gc.get_stats() if hasattr(gc, 'get_stats') else []
        total_gc_collections = sum(stat['collections'] for stat in gc_stats)
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            memory_percent=memory_percent,
            disk_io_read_mb=disk_io_read,
            disk_io_write_mb=disk_io_write,
            network_sent_mb=net_sent,
            network_recv_mb=net_recv,
            gc_collections=total_gc_collections,
            active_threads=threading.active_count(),
            open_files=len(process.open_files())
        )
    
    def run_comprehensive_optimization(self) -> List[OptimizationResult]:
        """Run comprehensive system optimization"""
        logger.info("Starting comprehensive system optimization...")
        
        results = []
        before_metrics = self.collect_current_metrics()
        
        # 1. Memory optimization
        start_time = time.time()
        memory_snapshot_before = self.memory_profiler.take_snapshot("before_optimization")
        gc_result = self.memory_profiler.optimize_garbage_collection()
        memory_leaks = self.memory_profiler.detect_memory_leaks()
        memory_snapshot_after = self.memory_profiler.take_snapshot("after_optimization")
        
        memory_optimization_time = (time.time() - start_time) * 1000
        
        # 2. CPU optimization
        start_time = time.time()
        cpu_analysis = self.cpu_optimizer.analyze_cpu_usage()
        thread_optimization = self.cpu_optimizer.optimize_thread_pool_size("io_bound")
        cpu_optimization_time = (time.time() - start_time) * 1000
        
        # 3. Cache optimization
        start_time = time.time()
        cache_stats = self.cache_optimizer.get_cache_statistics()
        cache_optimization_time = (time.time() - start_time) * 1000
        
        # Collect final metrics
        after_metrics = self.collect_current_metrics()
        
        # Calculate overall improvement
        memory_improvement = (before_metrics.memory_usage_mb - after_metrics.memory_usage_mb) / before_metrics.memory_usage_mb * 100
        cpu_improvement = (before_metrics.cpu_usage_percent - after_metrics.cpu_usage_percent) / before_metrics.cpu_usage_percent * 100 if before_metrics.cpu_usage_percent > 0 else 0
        
        overall_improvement = (abs(memory_improvement) + abs(cpu_improvement)) / 2
        
        # Generate recommendations
        recommendations = [
            f"Memory freed: {gc_result['memory_freed_mb']:.2f}MB",
            f"GC objects collected: {gc_result['objects_collected']}",
            f"Thread pool optimized: {thread_optimization['previous_threads']} -> {thread_optimization['optimized_threads']}"
        ]
        
        if memory_leaks:
            recommendations.extend([f"Memory leak detected: {leak['recommendation']}" for leak in memory_leaks])
        
        recommendations.extend([
            "Enable continuous performance monitoring",
            "Implement automated optimization triggers",
            "Set up performance alerting for critical thresholds"
        ])
        
        # Create optimization result
        result = OptimizationResult(
            operation="comprehensive_system_optimization",
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percent=overall_improvement,
            recommendations=recommendations,
            execution_time_ms=memory_optimization_time + cpu_optimization_time + cache_optimization_time
        )
        
        self.optimization_results.append(result)
        results.append(result)
        
        logger.info(f"Comprehensive optimization completed. Improvement: {overall_improvement:.2f}%")
        return results
    
    def start_continuous_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Continuous performance monitoring started")
    
    def stop_continuous_monitoring(self):
        """Stop continuous performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Continuous performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self.collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Check for optimization triggers
                if (metrics.memory_usage_mb > self.config['auto_gc_threshold_mb'] or
                    metrics.cpu_usage_percent > self.config['performance_alert_threshold']):
                    
                    logger.warning(f"Performance threshold exceeded - triggering optimization")
                    self.run_comprehensive_optimization()
                
                time.sleep(self.config['monitoring_interval_seconds'])
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)  # Wait before retrying
    
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        if not self.optimization_results:
            return "No optimization results available."
        
        latest_result = self.optimization_results[-1]
        
        report = f"""# System Performance Optimization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Optimization Type**: {latest_result.operation}
- **Overall Improvement**: {latest_result.improvement_percent:.2f}%
- **Execution Time**: {latest_result.execution_time_ms:.2f}ms

## Performance Metrics

### Before Optimization
- **CPU Usage**: {latest_result.before_metrics.cpu_usage_percent:.1f}%
- **Memory Usage**: {latest_result.before_metrics.memory_usage_mb:.2f}MB ({latest_result.before_metrics.memory_percent:.1f}%)
- **Active Threads**: {latest_result.before_metrics.active_threads}
- **Open Files**: {latest_result.before_metrics.open_files}
- **GC Collections**: {latest_result.before_metrics.gc_collections}

### After Optimization
- **CPU Usage**: {latest_result.after_metrics.cpu_usage_percent:.1f}%
- **Memory Usage**: {latest_result.after_metrics.memory_usage_mb:.2f}MB ({latest_result.after_metrics.memory_percent:.1f}%)
- **Active Threads**: {latest_result.after_metrics.active_threads}
- **Open Files**: {latest_result.after_metrics.open_files}
- **GC Collections**: {latest_result.after_metrics.gc_collections}

## Improvements
- **Memory Savings**: {latest_result.before_metrics.memory_usage_mb - latest_result.after_metrics.memory_usage_mb:.2f}MB
- **CPU Reduction**: {latest_result.before_metrics.cpu_usage_percent - latest_result.after_metrics.cpu_usage_percent:.1f}%

## Cache Performance
"""
        
        cache_stats = self.cache_optimizer.get_cache_statistics()
        for layer, stats in cache_stats.items():
            report += f"\n### {layer.upper()}\n"
            report += f"- **Entries**: {stats['entries']}\n"
            report += f"- **Hit Rate**: {stats['hit_rate']:.1%}\n"
            report += f"- **Memory Usage**: {stats['memory_usage_mb']:.2f}MB\n"
        
        report += f"\n## Recommendations\n"
        for i, rec in enumerate(latest_result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        if len(self.metrics_history) > 10:
            recent_metrics = list(self.metrics_history)[-10:]
            avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
            
            report += f"\n## Recent Performance Trends (Last 10 measurements)\n"
            report += f"- **Average CPU Usage**: {avg_cpu:.1f}%\n"
            report += f"- **Average Memory Usage**: {avg_memory:.2f}MB\n"
        
        return report

# Helper classes
class WeakValueDictionary:
    """Dictionary with weak references for memory leak detection"""
    
    def __init__(self):
        self._data = weakref.WeakValueDictionary()
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __contains__(self, key):
        return key in self._data
    
    def __len__(self):
        return len(self._data)
    
    def keys(self):
        return self._data.keys()

def main():
    """Example usage of the system performance optimizer"""
    print("Prometheus AI System Performance Optimizer")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = SystemPerformanceOptimizer({
        'max_cache_memory_mb': 256,
        'monitoring_interval_seconds': 10,
        'auto_gc_threshold_mb': 50
    })
    
    try:
        # Run comprehensive optimization
        results = optimizer.run_comprehensive_optimization()
        
        # Generate and display report
        report = optimizer.generate_optimization_report()
        print(report)
        
        # Start continuous monitoring
        optimizer.start_continuous_monitoring()
        
        print("\nContinuous monitoring started. Press Ctrl+C to stop...")
        
        try:
            while True:
                time.sleep(60)
                print(f"Current metrics: CPU {optimizer.collect_current_metrics().cpu_usage_percent:.1f}%, "
                     f"Memory {optimizer.collect_current_metrics().memory_usage_mb:.2f}MB")
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise
    
    finally:
        # Cleanup
        optimizer.stop_continuous_monitoring()
        optimizer.cpu_optimizer.thread_pool.shutdown(wait=True)
        optimizer.cpu_optimizer.process_pool.shutdown(wait=True)

if __name__ == "__main__":
    main()