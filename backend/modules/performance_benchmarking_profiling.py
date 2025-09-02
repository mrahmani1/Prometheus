"""
Advanced Performance Benchmarking and Optimization Profiling System
==================================================================

This module implements state-of-the-art performance analysis and optimization tools including:
- Multi-dimensional performance profiling (CPU, memory, I/O, GPU, network)
- Advanced algorithmic complexity analysis with empirical validation
- Real-time performance monitoring with anomaly detection
- Automated bottleneck identification and optimization recommendations
- Distributed performance profiling for multi-node systems
- Energy consumption analysis and green computing metrics
- Cache performance analysis and optimization suggestions
- Machine learning-based performance prediction
- Continuous benchmarking with historical trend analysis
- Performance regression detection and root cause analysis

Author: Advanced Performance Analysis System
"""

import asyncio
import time
import psutil
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Callable, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import traceback
import sys
import os
import gc
import tracemalloc
import cProfile
import pstats
import io
from contextlib import contextmanager
import functools
import itertools
import math
from collections import defaultdict, deque, Counter
from pathlib import Path
import json
import pickle
import sqlite3
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats, optimize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import optuna
from memory_profiler import LineProfiler, profile
import py_spy
import objgraph
import pympler
from pympler import muppy, summary, tracker
import resource

# GPU profiling libraries
try:
    import torch
    import torch.profiler
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Network profiling
import socket
import requests
from urllib.parse import urlparse

# Advanced profiling tools
try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

try:
    import viztracer
    VIZTRACER_AVAILABLE = True
except ImportError:
    VIZTRACER_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    timestamp: datetime
    function_name: str
    execution_time: float
    cpu_usage: float
    memory_usage: float
    memory_peak: float
    io_read: int
    io_write: int
    network_sent: int
    network_recv: int
    gpu_usage: float
    gpu_memory: float
    cache_hits: int
    cache_misses: int
    context_switches: int
    page_faults: int
    instructions_retired: int
    cycles: int
    branch_misses: int
    energy_consumption: float
    thermal_state: float
    algorithmic_complexity: Dict[str, float]
    bottlenecks: List[str]
    optimization_score: float
    
    def __post_init__(self):
        self.performance_index = self._calculate_performance_index()
        self.efficiency_ratio = self._calculate_efficiency_ratio()
    
    def _calculate_performance_index(self) -> float:
        """Calculate comprehensive performance index"""
        # Normalize all metrics to 0-1 scale and combine with weights
        weights = {
            'time': 0.25,
            'cpu': 0.20,
            'memory': 0.20,
            'io': 0.15,
            'gpu': 0.10,
            'cache': 0.10
        }
        
        # Inverse for time-based metrics (lower is better)
        time_score = 1.0 / (1.0 + self.execution_time)
        cpu_score = max(0, 1.0 - self.cpu_usage / 100.0)
        memory_score = max(0, 1.0 - self.memory_usage / (8 * 1024**3))  # 8GB reference
        io_score = 1.0 / (1.0 + (self.io_read + self.io_write) / 1024**2)  # MB reference
        gpu_score = max(0, 1.0 - self.gpu_usage / 100.0)
        cache_score = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        
        return (weights['time'] * time_score +
                weights['cpu'] * cpu_score +
                weights['memory'] * memory_score +
                weights['io'] * io_score +
                weights['gpu'] * gpu_score +
                weights['cache'] * cache_score)
    
    def _calculate_efficiency_ratio(self) -> float:
        """Calculate energy efficiency ratio"""
        if self.energy_consumption <= 0:
            return 1.0
        
        # Performance per unit energy
        return self.performance_index / self.energy_consumption

@dataclass
class BenchmarkConfiguration:
    """Configuration for benchmarking runs"""
    warmup_iterations: int = 5
    measurement_iterations: int = 100
    timeout_seconds: Optional[int] = 300
    memory_profiling: bool = True
    cpu_profiling: bool = True
    gpu_profiling: bool = False
    io_profiling: bool = True
    network_profiling: bool = False
    energy_profiling: bool = False
    cache_profiling: bool = True
    algorithmic_analysis: bool = True
    statistical_significance: bool = True
    confidence_level: float = 0.95
    outlier_detection: bool = True
    regression_detection: bool = True
    optimization_suggestions: bool = True
    distributed_profiling: bool = False
    real_time_monitoring: bool = False

@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    category: str
    priority: str  # 'high', 'medium', 'low'
    description: str
    expected_improvement: float
    implementation_complexity: str  # 'simple', 'moderate', 'complex'
    code_changes_required: List[str]
    estimated_development_time: str
    risk_assessment: str
    validation_tests: List[str]

class ComplexityAnalyzer:
    """Advanced algorithmic complexity analyzer with empirical validation"""
    
    def __init__(self):
        self.complexity_patterns = {
            'constant': lambda n: 1,
            'logarithmic': lambda n: np.log(n),
            'linear': lambda n: n,
            'linearithmic': lambda n: n * np.log(n),
            'quadratic': lambda n: n**2,
            'cubic': lambda n: n**3,
            'exponential': lambda n: 2**n,
            'factorial': lambda n: math.factorial(min(n, 20))  # Limit for numerical stability
        }
        
        self.ml_predictor = None
        self.training_data = []
    
    def analyze_empirical_complexity(self, function: Callable, 
                                   input_sizes: List[int],
                                   input_generator: Callable) -> Dict[str, Any]:
        """Analyze empirical complexity by running function with different input sizes"""
        measurements = []
        
        for size in input_sizes:
            test_input = input_generator(size)
            
            # Multiple runs for statistical significance
            execution_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                try:
                    function(test_input)
                    end_time = time.perf_counter()
                    execution_times.append(end_time - start_time)
                except Exception as e:
                    logger.warning(f"Function failed for input size {size}: {e}")
                    continue
            
            if execution_times:
                measurements.append({
                    'input_size': size,
                    'mean_time': np.mean(execution_times),
                    'std_time': np.std(execution_times),
                    'min_time': np.min(execution_times),
                    'max_time': np.max(execution_times)
                })
        
        if len(measurements) < 3:
            return {'error': 'Insufficient measurements for complexity analysis'}
        
        # Fit complexity curves
        input_sizes_array = np.array([m['input_size'] for m in measurements])
        execution_times_array = np.array([m['mean_time'] for m in measurements])
        
        complexity_fits = {}
        
        for complexity_name, complexity_func in self.complexity_patterns.items():
            try:
                # Generate expected values
                expected_complexity = np.array([complexity_func(n) for n in input_sizes_array])
                
                # Normalize to avoid numerical issues
                if np.max(expected_complexity) > 0:
                    expected_complexity = expected_complexity / np.max(expected_complexity)
                    normalized_times = execution_times_array / np.max(execution_times_array)
                    
                    # Calculate R-squared
                    correlation = np.corrcoef(expected_complexity, normalized_times)[0, 1]
                    r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0
                    
                    # Calculate mean squared error
                    mse = np.mean((expected_complexity - normalized_times) ** 2)
                    
                    complexity_fits[complexity_name] = {
                        'r_squared': r_squared,
                        'mse': mse,
                        'correlation': correlation,
                        'aic': len(input_sizes_array) * np.log(mse) + 2  # Simplified AIC
                    }
            
            except Exception as e:
                logger.warning(f"Failed to fit {complexity_name} complexity: {e}")
                complexity_fits[complexity_name] = {'error': str(e)}
        
        # Find best fit
        valid_fits = {k: v for k, v in complexity_fits.items() 
                     if 'error' not in v and not np.isnan(v['r_squared'])}
        
        best_fit = None
        if valid_fits:
            best_fit = max(valid_fits.keys(), key=lambda k: valid_fits[k]['r_squared'])
        
        # Predict performance scaling
        scaling_prediction = self._predict_scaling(measurements, best_fit)
        
        return {
            'measurements': measurements,
            'complexity_fits': complexity_fits,
            'best_fit_complexity': best_fit,
            'best_fit_metrics': valid_fits.get(best_fit, {}),
            'scaling_prediction': scaling_prediction,
            'growth_rate': self._calculate_growth_rate(measurements),
            'efficiency_analysis': self._analyze_efficiency_trends(measurements)
        }
    
    def _predict_scaling(self, measurements: List[Dict], best_fit: str) -> Dict[str, Any]:
        """Predict performance scaling for larger inputs"""
        if not best_fit or len(measurements) < 2:
            return {'error': 'Insufficient data for scaling prediction'}
        
        input_sizes = [m['input_size'] for m in measurements]
        times = [m['mean_time'] for m in measurements]
        
        # Predict for 2x, 5x, 10x larger inputs
        max_size = max(input_sizes)
        prediction_sizes = [max_size * factor for factor in [2, 5, 10, 100]]
        
        predictions = {}
        
        if best_fit in self.complexity_patterns:
            complexity_func = self.complexity_patterns[best_fit]
            
            # Fit scaling coefficient
            try:
                current_complexity = complexity_func(max_size)
                current_time = times[-1]
                
                if current_complexity > 0:
                    scaling_coefficient = current_time / current_complexity
                    
                    for pred_size in prediction_sizes:
                        pred_complexity = complexity_func(pred_size)
                        pred_time = scaling_coefficient * pred_complexity
                        
                        predictions[f'{pred_size:,}'] = {
                            'predicted_time': pred_time,
                            'scaling_factor': pred_complexity / current_complexity,
                            'time_ratio': pred_time / current_time
                        }
            
            except Exception as e:
                predictions['error'] = str(e)
        
        return predictions
    
    def _calculate_growth_rate(self, measurements: List[Dict]) -> Dict[str, float]:
        """Calculate growth rate of execution time"""
        if len(measurements) < 2:
            return {'error': 'Need at least 2 measurements'}
        
        sizes = [m['input_size'] for m in measurements]
        times = [m['mean_time'] for m in measurements]
        
        # Calculate percentage growth between consecutive measurements
        growth_rates = []
        for i in range(1, len(measurements)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            
            if size_ratio > 1:
                growth_rate = (time_ratio - 1) / (size_ratio - 1)
                growth_rates.append(growth_rate)
        
        if not growth_rates:
            return {'error': 'Could not calculate growth rates'}
        
        return {
            'mean_growth_rate': np.mean(growth_rates),
            'std_growth_rate': np.std(growth_rates),
            'min_growth_rate': np.min(growth_rates),
            'max_growth_rate': np.max(growth_rates),
            'growth_acceleration': np.std(growth_rates)  # Higher std indicates acceleration
        }
    
    def _analyze_efficiency_trends(self, measurements: List[Dict]) -> Dict[str, Any]:
        """Analyze efficiency trends across input sizes"""
        if len(measurements) < 2:
            return {'error': 'Need at least 2 measurements'}
        
        sizes = [m['input_size'] for m in measurements]
        times = [m['mean_time'] for m in measurements]
        
        # Calculate throughput (operations per second)
        throughputs = [size / time for size, time in zip(sizes, times)]
        
        # Calculate efficiency (throughput per unit input)
        efficiencies = [throughput / size for throughput, size in zip(throughputs, sizes)]
        
        # Trend analysis
        size_array = np.array(sizes)
        efficiency_array = np.array(efficiencies)
        
        # Linear regression on efficiency
        if len(sizes) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(size_array, efficiency_array)
            
            efficiency_trend = {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend': 'improving' if slope > 0 else 'degrading' if slope < 0 else 'stable'
            }
        else:
            efficiency_trend = {'error': 'Insufficient data for trend analysis'}
        
        return {
            'throughputs': throughputs,
            'efficiencies': efficiencies,
            'mean_efficiency': np.mean(efficiencies),
            'efficiency_std': np.std(efficiencies),
            'efficiency_trend': efficiency_trend,
            'optimal_input_size': sizes[np.argmax(efficiencies)]
        }
    
    def train_ml_predictor(self, training_data: List[Dict]) -> Dict[str, Any]:
        """Train ML model to predict performance based on code features"""
        if len(training_data) < 10:
            return {'error': 'Insufficient training data (need at least 10 samples)'}
        
        # Extract features and targets
        features = []
        targets = []
        
        for sample in training_data:
            if 'code_features' in sample and 'execution_time' in sample:
                features.append(list(sample['code_features'].values()))
                targets.append(sample['execution_time'])
        
        if len(features) < 10:
            return {'error': 'Insufficient feature data'}
        
        X = np.array(features)
        y = np.array(targets)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models and select best
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(random_state=42),
        }
        
        best_model = None
        best_score = -float('inf')
        model_results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                
                model_results[name] = {
                    'train_r2': train_score,
                    'test_r2': test_score,
                    'mse': mse,
                    'overfitting': train_score - test_score
                }
                
                if test_score > best_score:
                    best_score = test_score
                    best_model = model
                    
            except Exception as e:
                model_results[name] = {'error': str(e)}
        
        if best_model is not None:
            self.ml_predictor = best_model
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_
            else:
                feature_importance = None
        
        return {
            'model_results': model_results,
            'best_model': type(best_model).__name__ if best_model else None,
            'best_test_score': best_score,
            'feature_importance': feature_importance.tolist() if feature_importance is not None else None,
            'training_samples': len(training_data)
        }

class RealTimeProfiler:
    """Real-time performance profiler with anomaly detection"""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.metrics_history = deque(maxlen=10000)  # Keep last 10k measurements
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.baseline_established = False
        self.baseline_metrics = {}
        self.alert_callbacks = []
        
    def add_alert_callback(self, callback: Callable):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self, target_function: Optional[Callable] = None):
        """Start real-time monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(target_function,), 
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Real-time profiling started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        logger.info("Real-time profiling stopped")
    
    def _monitoring_loop(self, target_function: Optional[Callable]):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                
                if target_function:
                    # Add function-specific metrics
                    func_metrics = self._collect_function_metrics(target_function)
                    metrics.update(func_metrics)
                
                self.metrics_history.append({
                    'timestamp': datetime.now(),
                    **metrics
                })
                
                # Anomaly detection after establishing baseline
                if len(self.metrics_history) > 100:
                    self._detect_anomalies(metrics)
                
                # Update baseline periodically
                if len(self.metrics_history) % 1000 == 0:
                    self._update_baseline()
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.sampling_interval)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-wide performance metrics"""
        process = psutil.Process()
        
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available': psutil.virtual_memory().available,
            'disk_io_read': psutil.disk_io_counters().read_bytes if psutil.disk_io_counters() else 0,
            'disk_io_write': psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0,
            'network_sent': psutil.net_io_counters().bytes_sent if psutil.net_io_counters() else 0,
            'network_recv': psutil.net_io_counters().bytes_recv if psutil.net_io_counters() else 0,
            'process_cpu': process.cpu_percent(),
            'process_memory': process.memory_info().rss,
            'process_threads': process.num_threads(),
            'context_switches': process.num_ctx_switches().voluntary,
            'page_faults': process.memory_info().vms,
        }
        
        # GPU metrics if available
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                metrics.update({
                    'gpu_utilization': gpu_info.gpu,
                    'gpu_memory_used': memory_info.used,
                    'gpu_memory_total': memory_info.total,
                    'gpu_memory_percent': (memory_info.used / memory_info.total) * 100
                })
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
        
        return metrics
    
    def _collect_function_metrics(self, function: Callable) -> Dict[str, Any]:
        """Collect function-specific metrics"""
        # This would require instrumenting the function
        # For now, return placeholder metrics
        return {
            'function_calls': 0,
            'function_errors': 0,
            'function_avg_time': 0.0
        }
    
    def _detect_anomalies(self, current_metrics: Dict[str, float]):
        """Detect performance anomalies"""
        if not self.baseline_established:
            return
        
        # Prepare data for anomaly detection
        numeric_metrics = {k: v for k, v in current_metrics.items() 
                         if isinstance(v, (int, float)) and not np.isnan(v)}
        
        if len(numeric_metrics) < 3:
            return
        
        # Compare with baseline
        anomalies = []
        
        for metric_name, current_value in numeric_metrics.items():
            if metric_name in self.baseline_metrics:
                baseline_mean = self.baseline_metrics[metric_name]['mean']
                baseline_std = self.baseline_metrics[metric_name]['std']
                
                # Z-score based anomaly detection
                if baseline_std > 0:
                    z_score = abs(current_value - baseline_mean) / baseline_std
                    
                    if z_score > 3:  # 3-sigma rule
                        anomalies.append({
                            'metric': metric_name,
                            'current_value': current_value,
                            'baseline_mean': baseline_mean,
                            'z_score': z_score,
                            'severity': 'high' if z_score > 5 else 'medium'
                        })
        
        # Trigger alerts for anomalies
        if anomalies:
            self._trigger_alerts(anomalies)
    
    def _update_baseline(self):
        """Update baseline performance metrics"""
        if len(self.metrics_history) < 100:
            return
        
        # Use recent history for baseline
        recent_metrics = list(self.metrics_history)[-1000:]
        
        # Calculate baseline statistics
        metric_names = set()
        for entry in recent_metrics:
            metric_names.update(k for k, v in entry.items() 
                              if isinstance(v, (int, float)) and k != 'timestamp')
        
        baseline = {}
        
        for metric_name in metric_names:
            values = []
            for entry in recent_metrics:
                if metric_name in entry and isinstance(entry[metric_name], (int, float)):
                    values.append(entry[metric_name])
            
            if values:
                baseline[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        self.baseline_metrics = baseline
        self.baseline_established = True
        logger.info(f"Updated baseline with {len(baseline)} metrics")
    
    def _trigger_alerts(self, anomalies: List[Dict]):
        """Trigger performance alerts"""
        alert_data = {
            'timestamp': datetime.now(),
            'anomalies': anomalies,
            'severity': max(a['severity'] for a in anomalies)
        }
        
        logger.warning(f"Performance anomaly detected: {len(anomalies)} metrics affected")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def get_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for specified time window"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        # Filter recent metrics
        recent_metrics = [m for m in self.metrics_history 
                         if m['timestamp'] > cutoff_time]
        
        if not recent_metrics:
            return {'error': 'No recent metrics available'}
        
        # Calculate summary statistics
        summary = {
            'time_window_minutes': window_minutes,
            'sample_count': len(recent_metrics),
            'metrics': {}
        }
        
        # Get all numeric metric names
        metric_names = set()
        for entry in recent_metrics:
            metric_names.update(k for k, v in entry.items() 
                              if isinstance(v, (int, float)) and k != 'timestamp')
        
        for metric_name in metric_names:
            values = []
            for entry in recent_metrics:
                if metric_name in entry and isinstance(entry[metric_name], (int, float)):
                    values.append(entry[metric_name])
            
            if values:
                summary['metrics'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'p95': np.percentile(values, 95),
                    'trend': self._calculate_trend(values)
                }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for metric values"""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Split into first and second half
        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]
        
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)
        
        if abs(second_mean - first_mean) < np.std(values) * 0.1:
            return 'stable'
        elif second_mean > first_mean:
            return 'increasing'
        else:
            return 'decreasing'

class DistributedProfiler:
    """Distributed performance profiling for multi-node systems"""
    
    def __init__(self):
        self.nodes = {}
        self.coordinator_port = 8765
        self.profiling_active = False
        
    async def add_node(self, node_id: str, host: str, port: int):
        """Add a node to the distributed profiling system"""
        self.nodes[node_id] = {
            'host': host,
            'port': port,
            'status': 'offline',
            'last_heartbeat': None,
            'metrics': deque(maxlen=1000)
        }
        
        # Test connection
        try:
            # This would connect to the actual node
            self.nodes[node_id]['status'] = 'online'
            logger.info(f"Added node {node_id} at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to add node {node_id}: {e}")
    
    async def start_distributed_profiling(self, duration_seconds: int = 300):
        """Start distributed profiling across all nodes"""
        if self.profiling_active:
            return {'error': 'Profiling already active'}
        
        self.profiling_active = True
        
        # Start profiling on all nodes
        tasks = []
        for node_id in self.nodes:
            task = asyncio.create_task(self._profile_node(node_id, duration_seconds))
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            aggregated_results = self._aggregate_distributed_results(results)
            
            return aggregated_results
        
        finally:
            self.profiling_active = False
    
    async def _profile_node(self, node_id: str, duration: int) -> Dict[str, Any]:
        """Profile a specific node"""
        node_info = self.nodes[node_id]
        
        # Simulate distributed profiling
        # In real implementation, this would communicate with remote nodes
        
        start_time = time.time()
        metrics_collected = 0
        
        while time.time() - start_time < duration and self.profiling_active:
            try:
                # Collect metrics from node
                metrics = await self._collect_node_metrics(node_id)
                
                node_info['metrics'].append({
                    'timestamp': datetime.now(),
                    **metrics
                })
                
                metrics_collected += 1
                await asyncio.sleep(1)  # 1 second interval
                
            except Exception as e:
                logger.error(f"Error profiling node {node_id}: {e}")
        
        return {
            'node_id': node_id,
            'metrics_collected': metrics_collected,
            'duration': time.time() - start_time,
            'status': 'completed'
        }
    
    async def _collect_node_metrics(self, node_id: str) -> Dict[str, Any]:
        """Collect metrics from a specific node"""
        # Simulate node metric collection
        # In real implementation, this would make HTTP/gRPC calls to nodes
        
        return {
            'cpu_usage': np.random.uniform(10, 90),
            'memory_usage': np.random.uniform(1e9, 8e9),
            'network_latency': np.random.uniform(1, 100),
            'active_connections': np.random.randint(10, 1000),
            'request_rate': np.random.uniform(100, 10000)
        }
    
    def _aggregate_distributed_results(self, results: List[Any]) -> Dict[str, Any]:
        """Aggregate results from distributed profiling"""
        successful_results = [r for r in results if isinstance(r, dict) and 'error' not in r]
        
        if not successful_results:
            return {'error': 'No successful node profiling results'}
        
        aggregated = {
            'total_nodes': len(self.nodes),
            'successful_nodes': len(successful_results),
            'total_metrics_collected': sum(r['metrics_collected'] for r in successful_results),
            'average_duration': np.mean([r['duration'] for r in successful_results]),
            'node_summaries': {}
        }
        
        # Calculate per-node summaries
        for result in successful_results:
            node_id = result['node_id']
            node_metrics = list(self.nodes[node_id]['metrics'])
            
            if node_metrics:
                # Calculate statistics for each metric type
                metric_stats = {}
                metric_names = set()
                
                for entry in node_metrics:
                    metric_names.update(k for k, v in entry.items() 
                                      if isinstance(v, (int, float)) and k != 'timestamp')
                
                for metric_name in metric_names:
                    values = [entry[metric_name] for entry in node_metrics 
                            if metric_name in entry]
                    
                    if values:
                        metric_stats[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
                
                aggregated['node_summaries'][node_id] = {
                    'metrics_count': len(node_metrics),
                    'metric_statistics': metric_stats
                }
        
        # Calculate cluster-wide statistics
        aggregated['cluster_statistics'] = self._calculate_cluster_statistics()
        
        return aggregated
    
    def _calculate_cluster_statistics(self) -> Dict[str, Any]:
        """Calculate cluster-wide performance statistics"""
        all_metrics = []
        
        for node_info in self.nodes.values():
            all_metrics.extend(list(node_info['metrics']))
        
        if not all_metrics:
            return {'error': 'No metrics available'}
        
        # Find common metrics across all nodes
        common_metrics = set()
        if all_metrics:
            common_metrics = set(all_metrics[0].keys()) - {'timestamp'}
            for entry in all_metrics[1:]:
                common_metrics &= (set(entry.keys()) - {'timestamp'})
        
        cluster_stats = {}
        
        for metric_name in common_metrics:
            values = []
            for entry in all_metrics:
                if metric_name in entry and isinstance(entry[metric_name], (int, float)):
                    values.append(entry[metric_name])
            
            if values:
                cluster_stats[metric_name] = {
                    'cluster_mean': np.mean(values),
                    'cluster_std': np.std(values),
                    'cluster_min': np.min(values),
                    'cluster_max': np.max(values),
                    'node_variance': self._calculate_node_variance(metric_name),
                    'load_balance_score': self._calculate_load_balance_score(metric_name)
                }
        
        return cluster_stats
    
    def _calculate_node_variance(self, metric_name: str) -> float:
        """Calculate variance in metric values across nodes"""
        node_means = []
        
        for node_info in self.nodes.values():
            values = []
            for entry in node_info['metrics']:
                if metric_name in entry and isinstance(entry[metric_name], (int, float)):
                    values.append(entry[metric_name])
            
            if values:
                node_means.append(np.mean(values))
        
        return np.std(node_means) if len(node_means) > 1 else 0.0
    
    def _calculate_load_balance_score(self, metric_name: str) -> float:
        """Calculate load balance score (0-1, higher is more balanced)"""
        node_means = []
        
        for node_info in self.nodes.values():
            values = []
            for entry in node_info['metrics']:
                if metric_name in entry and isinstance(entry[metric_name], (int, float)):
                    values.append(entry[metric_name])
            
            if values:
                node_means.append(np.mean(values))
        
        if len(node_means) <= 1:
            return 1.0
        
        # Calculate coefficient of variation (lower = more balanced)
        cv = np.std(node_means) / np.mean(node_means) if np.mean(node_means) > 0 else 0
        
        # Convert to balance score (0-1, higher = better balance)
        balance_score = 1.0 / (1.0 + cv)
        
        return balance_score

class OptimizationEngine:
    """Advanced optimization engine with ML-based recommendations"""
    
    def __init__(self):
        self.optimization_strategies = self._initialize_strategies()
        self.ml_optimizer = None
        self.historical_optimizations = []
        
    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize optimization strategies"""
        return {
            'memory_optimization': {
                'patterns': [
                    {
                        'condition': lambda metrics: metrics.memory_usage > 4e9,  # > 4GB
                        'recommendation': OptimizationRecommendation(
                            category='memory',
                            priority='high',
                            description='High memory usage detected. Consider memory profiling and optimization.',
                            expected_improvement=0.3,
                            implementation_complexity='moderate',
                            code_changes_required=[
                                'Add memory profiling with tracemalloc',
                                'Implement object pooling for frequently created objects',
                                'Use generators instead of lists where possible',
                                'Add explicit garbage collection at key points'
                            ],
                            estimated_development_time='2-4 hours',
                            risk_assessment='low',
                            validation_tests=[
                                'Memory usage monitoring',
                                'Performance regression testing',
                                'Load testing with memory constraints'
                            ]
                        )
                    },
                    {
                        'condition': lambda metrics: metrics.memory_peak > metrics.memory_usage * 2,
                        'recommendation': OptimizationRecommendation(
                            category='memory',
                            priority='medium',
                            description='Memory usage spikes detected. Implement streaming or batch processing.',
                            expected_improvement=0.25,
                            implementation_complexity='moderate',
                            code_changes_required=[
                                'Implement streaming data processing',
                                'Add batch processing with configurable batch sizes',
                                'Use memory-mapped files for large datasets',
                                'Implement lazy loading patterns'
                            ],
                            estimated_development_time='4-8 hours',
                            risk_assessment='medium',
                            validation_tests=[
                                'Peak memory usage testing',
                                'Large dataset processing validation',
                                'Memory consistency tests'
                            ]
                        )
                    }
                ]
            },
            
            'cpu_optimization': {
                'patterns': [
                    {
                        'condition': lambda metrics: metrics.cpu_usage > 80,
                        'recommendation': OptimizationRecommendation(
                            category='cpu',
                            priority='high',
                            description='High CPU usage detected. Consider parallelization and algorithmic improvements.',
                            expected_improvement=0.4,
                            implementation_complexity='complex',
                            code_changes_required=[
                                'Profile CPU hotspots with cProfile',
                                'Implement multiprocessing for CPU-bound tasks',
                                'Use NumPy vectorization for numerical computations',
                                'Consider Cython or Numba for performance-critical sections',
                                'Implement caching for expensive computations'
                            ],
                            estimated_development_time='1-2 days',
                            risk_assessment='medium',
                            validation_tests=[
                                'CPU usage monitoring',
                                'Performance benchmarking',
                                'Parallel processing validation',
                                'Numerical accuracy tests'
                            ]
                        )
                    },
                    {
                        'condition': lambda metrics: metrics.context_switches > 10000,
                        'recommendation': OptimizationRecommendation(
                            category='cpu',
                            priority='medium',
                            description='High context switching detected. Optimize thread usage and synchronization.',
                            expected_improvement=0.2,
                            implementation_complexity='moderate',
                            code_changes_required=[
                                'Reduce number of threads or use thread pools',
                                'Implement async/await patterns where appropriate',
                                'Optimize lock contention and synchronization',
                                'Use queue-based communication between threads'
                            ],
                            estimated_development_time='4-8 hours',
                            risk_assessment='medium',
                            validation_tests=[
                                'Context switch monitoring',
                                'Thread synchronization testing',
                                'Concurrency validation'
                            ]
                        )
                    }
                ]
            },
            
            'io_optimization': {
                'patterns': [
                    {
                        'condition': lambda metrics: (metrics.io_read + metrics.io_write) > 1e8,  # > 100MB
                        'recommendation': OptimizationRecommendation(
                            category='io',
                            priority='high',
                            description='High I/O usage detected. Implement buffering and async I/O.',
                            expected_improvement=0.5,
                            implementation_complexity='moderate',
                            code_changes_required=[
                                'Implement async I/O with aiofiles',
                                'Add read/write buffering',
                                'Use memory-mapped files for large files',
                                'Implement I/O batching and queuing',
                                'Add I/O caching layer'
                            ],
                            estimated_development_time='6-12 hours',
                            risk_assessment='low',
                            validation_tests=[
                                'I/O throughput testing',
                                'File integrity validation',
                                'Concurrent I/O testing'
                            ]
                        )
                    }
                ]
            },
            
            'algorithmic_optimization': {
                'patterns': [
                    {
                        'condition': lambda metrics: 'quadratic' in str(metrics.algorithmic_complexity) or 'cubic' in str(metrics.algorithmic_complexity),
                        'recommendation': OptimizationRecommendation(
                            category='algorithm',
                            priority='high',
                            description='Poor algorithmic complexity detected. Consider algorithm improvements.',
                            expected_improvement=0.7,
                            implementation_complexity='complex',
                            code_changes_required=[
                                'Analyze algorithm complexity and bottlenecks',
                                'Replace O(n²) or O(n³) algorithms with more efficient alternatives',
                                'Implement appropriate data structures (hash tables, trees, etc.)',
                                'Add memoization for recursive algorithms',
                                'Consider approximation algorithms for NP-hard problems'
                            ],
                            estimated_development_time='1-3 days',
                            risk_assessment='high',
                            validation_tests=[
                                'Algorithmic correctness validation',
                                'Performance scaling tests',
                                'Edge case testing',
                                'Numerical stability tests'
                            ]
                        )
                    }
                ]
            },
            
            'caching_optimization': {
                'patterns': [
                    {
                        'condition': lambda metrics: metrics.cache_misses > metrics.cache_hits * 2,
                        'recommendation': OptimizationRecommendation(
                            category='caching',
                            priority='medium',
                            description='Poor cache hit rate detected. Optimize caching strategy.',
                            expected_improvement=0.3,
                            implementation_complexity='simple',
                            code_changes_required=[
                                'Analyze cache access patterns',
                                'Implement LRU or adaptive replacement caching',
                                'Increase cache size if memory permits',
                                'Add cache warming strategies',
                                'Implement cache partitioning for different data types'
                            ],
                            estimated_development_time='2-4 hours',
                            risk_assessment='low',
                            validation_tests=[
                                'Cache hit rate monitoring',
                                'Cache consistency testing',
                                'Performance validation'
                            ]
                        )
                    }
                ]
            }
        }
    
    def analyze_performance_bottlenecks(self, metrics: PerformanceMetrics) -> List[OptimizationRecommendation]:
        """Analyze performance metrics and generate optimization recommendations"""
        recommendations = []
        
        # Check all optimization strategies
        for strategy_name, strategy_info in self.optimization_strategies.items():
            for pattern in strategy_info['patterns']:
                try:
                    if pattern['condition'](metrics):
                        recommendations.append(pattern['recommendation'])
                except Exception as e:
                    logger.warning(f"Error evaluating optimization pattern in {strategy_name}: {e}")
        
        # Sort by priority and expected improvement
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        
        recommendations.sort(
            key=lambda r: (priority_order.get(r.priority, 0), r.expected_improvement),
            reverse=True
        )
        
        return recommendations
    
    def generate_optimization_plan(self, recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Generate comprehensive optimization plan"""
        if not recommendations:
            return {'message': 'No optimization recommendations found'}
        
        # Categorize recommendations
        categories = defaultdict(list)
        for rec in recommendations:
            categories[rec.category].append(rec)
        
        # Calculate total expected improvement
        total_improvement = sum(rec.expected_improvement for rec in recommendations)
        
        # Estimate total development time
        time_estimates = {
            'simple': 2,      # hours
            'moderate': 6,    # hours
            'complex': 16     # hours
        }
        
        total_dev_hours = sum(
            time_estimates.get(rec.implementation_complexity, 8) 
            for rec in recommendations
        )
        
        # Risk assessment
        risk_levels = [rec.risk_assessment for rec in recommendations]
        overall_risk = 'high' if 'high' in risk_levels else 'medium' if 'medium' in risk_levels else 'low'
        
        # Prioritize recommendations for implementation phases
        phases = self._create_implementation_phases(recommendations)
        
        return {
            'summary': {
                'total_recommendations': len(recommendations),
                'categories': list(categories.keys()),
                'expected_improvement': min(1.0, total_improvement),  # Cap at 100%
                'estimated_development_hours': total_dev_hours,
                'overall_risk': overall_risk,
                'implementation_phases': len(phases)
            },
            'recommendations_by_category': {
                category: [self._recommendation_to_dict(rec) for rec in recs]
                for category, recs in categories.items()
            },
            'implementation_phases': phases,
            'quick_wins': [
                self._recommendation_to_dict(rec) for rec in recommendations
                if rec.implementation_complexity == 'simple' and rec.expected_improvement > 0.1
            ],
            'high_impact_changes': [
                self._recommendation_to_dict(rec) for rec in recommendations
                if rec.expected_improvement > 0.4
            ]
        }
    
    def _create_implementation_phases(self, recommendations: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Create implementation phases based on dependencies and complexity"""
        phases = []
        
        # Phase 1: Quick wins and foundational changes
        phase1 = [rec for rec in recommendations 
                 if rec.implementation_complexity == 'simple' or rec.priority == 'high']
        
        if phase1:
            phases.append({
                'phase': 1,
                'description': 'Quick wins and high-priority optimizations',
                'recommendations': [rec.description for rec in phase1],
                'estimated_duration': '1-2 days',
                'expected_improvement': sum(rec.expected_improvement for rec in phase1)
            })
        
        # Phase 2: Moderate complexity changes
        phase2 = [rec for rec in recommendations 
                 if rec.implementation_complexity == 'moderate' and rec not in phase1]
        
        if phase2:
            phases.append({
                'phase': 2,
                'description': 'Moderate complexity optimizations',
                'recommendations': [rec.description for rec in phase2],
                'estimated_duration': '3-5 days',
                'expected_improvement': sum(rec.expected_improvement for rec in phase2)
            })
        
        # Phase 3: Complex changes
        phase3 = [rec for rec in recommendations 
                 if rec.implementation_complexity == 'complex']
        
        if phase3:
            phases.append({
                'phase': 3,
                'description': 'Complex algorithmic and architectural optimizations',
                'recommendations': [rec.description for rec in phase3],
                'estimated_duration': '1-2 weeks',
                'expected_improvement': sum(rec.expected_improvement for rec in phase3)
            })
        
        return phases
    
    def _recommendation_to_dict(self, rec: OptimizationRecommendation) -> Dict[str, Any]:
        """Convert optimization recommendation to dictionary"""
        return {
            'category': rec.category,
            'priority': rec.priority,
            'description': rec.description,
            'expected_improvement': rec.expected_improvement,
            'implementation_complexity': rec.implementation_complexity,
            'code_changes_required': rec.code_changes_required,
            'estimated_development_time': rec.estimated_development_time,
            'risk_assessment': rec.risk_assessment,
            'validation_tests': rec.validation_tests
        }
    
    async def optimize_function_automatically(self, function: Callable, 
                                            optimization_targets: List[str]) -> Dict[str, Any]:
        """Attempt automatic optimization of function"""
        # This is a conceptual implementation
        # Real automatic optimization would require more sophisticated techniques
        
        original_metrics = await self._benchmark_function(function)
        
        optimization_results = {
            'original_performance': original_metrics,
            'optimizations_applied': [],
            'final_performance': None,
            'improvement_achieved': 0.0
        }
        
        # Apply simple automatic optimizations
        optimized_function = function
        
        for target in optimization_targets:
            if target == 'memory':
                # Apply memory optimizations
                optimized_function = self._apply_memory_optimizations(optimized_function)
                optimization_results['optimizations_applied'].append('memory_optimization')
            
            elif target == 'cpu':
                # Apply CPU optimizations
                optimized_function = self._apply_cpu_optimizations(optimized_function)
                optimization_results['optimizations_applied'].append('cpu_optimization')
        
        # Benchmark optimized function
        if optimization_results['optimizations_applied']:
            optimized_metrics = await self._benchmark_function(optimized_function)
            optimization_results['final_performance'] = optimized_metrics
            
            # Calculate improvement
            if original_metrics and optimized_metrics:
                improvement = (original_metrics.get('execution_time', 1) - 
                             optimized_metrics.get('execution_time', 1)) / original_metrics.get('execution_time', 1)
                optimization_results['improvement_achieved'] = max(0, improvement)
        
        return optimization_results
    
    async def _benchmark_function(self, function: Callable) -> Dict[str, float]:
        """Benchmark function performance"""
        # Simple benchmarking
        times = []
        memory_usage = []
        
        for _ in range(10):
            tracemalloc.start()
            start_time = time.perf_counter()
            
            try:
                function()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                
                current, peak = tracemalloc.get_traced_memory()
                memory_usage.append(peak)
                
            except Exception as e:
                logger.error(f"Function benchmark failed: {e}")
            finally:
                tracemalloc.stop()
        
        if not times:
            return {}
        
        return {
            'execution_time': np.mean(times),
            'execution_time_std': np.std(times),
            'peak_memory': np.max(memory_usage),
            'avg_memory': np.mean(memory_usage)
        }
    
    def _apply_memory_optimizations(self, function: Callable) -> Callable:
        """Apply automatic memory optimizations"""
        # This is a simplified example
        # Real implementation would require code analysis and transformation
        
        @functools.wraps(function)
        def optimized_function(*args, **kwargs):
            # Force garbage collection before execution
            gc.collect()
            
            result = function(*args, **kwargs)
            
            # Force garbage collection after execution
            gc.collect()
            
            return result
        
        return optimized_function
    
    def _apply_cpu_optimizations(self, function: Callable) -> Callable:
        """Apply automatic CPU optimizations"""
        # This is a simplified example
        # Real implementation would require more sophisticated analysis
        
        @functools.wraps(function)
        def optimized_function(*args, **kwargs):
            # Add simple caching
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            if hasattr(optimized_function, '_cache'):
                if cache_key in optimized_function._cache:
                    return optimized_function._cache[cache_key]
            else:
                optimized_function._cache = {}
            
            result = function(*args, **kwargs)
            optimized_function._cache[cache_key] = result
            
            # Limit cache size
            if len(optimized_function._cache) > 100:
                optimized_function._cache.clear()
            
            return result
        
        return optimized_function

class AdvancedBenchmarkingSuite:
    """Comprehensive benchmarking suite with advanced features"""
    
    def __init__(self, config: BenchmarkConfiguration):
        self.config = config
        self.complexity_analyzer = ComplexityAnalyzer()
        self.real_time_profiler = RealTimeProfiler()
        self.distributed_profiler = DistributedProfiler()
        self.optimization_engine = OptimizationEngine()
        self.benchmark_history = []
        self.regression_detector = self._initialize_regression_detector()
        
        # Performance tracking database
        self.db_path = "performance_benchmarks.db"
        self._initialize_database()
    
    def _initialize_regression_detector(self) -> IsolationForest:
        """Initialize performance regression detector"""
        return IsolationForest(contamination=0.05, random_state=42)
    
    def _initialize_database(self):
        """Initialize SQLite database for performance tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                function_name TEXT NOT NULL,
                execution_time REAL,
                memory_usage REAL,
                cpu_usage REAL,
                gpu_usage REAL,
                optimization_score REAL,
                performance_index REAL,
                configuration TEXT,
                git_commit TEXT,
                environment_info TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                benchmark_id INTEGER,
                category TEXT,
                priority TEXT,
                description TEXT,
                expected_improvement REAL,
                implementation_complexity TEXT,
                FOREIGN KEY (benchmark_id) REFERENCES benchmarks (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def comprehensive_benchmark(self, function: Callable, 
                                    input_generator: Optional[Callable] = None,
                                    input_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """Execute comprehensive benchmarking with all features"""
        
        logger.info(f"Starting comprehensive benchmark for {function.__name__}")
        
        benchmark_results = {
            'function_name': function.__name__,
            'timestamp': datetime.now(),
            'configuration': self.config.__dict__,
            'results': {}
        }
        
        try:
            # 1. Basic performance profiling
            logger.info("Executing basic performance profiling...")
            basic_metrics = await self._basic_performance_profile(function)
            benchmark_results['results']['basic_metrics'] = basic_metrics
            
            # 2. Algorithmic complexity analysis
            if self.config.algorithmic_analysis and input_generator and input_sizes:
                logger.info("Analyzing algorithmic complexity...")
                complexity_results = self.complexity_analyzer.analyze_empirical_complexity(
                    function, input_sizes, input_generator
                )
                benchmark_results['results']['complexity_analysis'] = complexity_results
            
            # 3. Memory profiling
            if self.config.memory_profiling:
                logger.info("Executing memory profiling...")
                memory_results = await self._detailed_memory_profile(function)
                benchmark_results['results']['memory_profiling'] = memory_results
            
            # 4. CPU profiling
            if self.config.cpu_profiling:
                logger.info("Executing CPU profiling...")
                cpu_results = await self._detailed_cpu_profile(function)
                benchmark_results['results']['cpu_profiling'] = cpu_results
            
            # 5. GPU profiling
            if self.config.gpu_profiling and GPU_AVAILABLE:
                logger.info("Executing GPU profiling...")
                gpu_results = await self._detailed_gpu_profile(function)
                benchmark_results['results']['gpu_profiling'] = gpu_results
            
            # 6. I/O profiling
            if self.config.io_profiling:
                logger.info("Executing I/O profiling...")
                io_results = await self._detailed_io_profile(function)
                benchmark_results['results']['io_profiling'] = io_results
            
            # 7. Real-time monitoring
            if self.config.real_time_monitoring:
                logger.info("Starting real-time monitoring...")
                self.real_time_profiler.start_monitoring(function)
                
                # Run function multiple times while monitoring
                for _ in range(20):
                    function()
                    await asyncio.sleep(0.1)
                
                self.real_time_profiler.stop_monitoring()
                
                rt_summary = self.real_time_profiler.get_performance_summary()
                benchmark_results['results']['real_time_monitoring'] = rt_summary
            
            # 8. Statistical analysis
            if self.config.statistical_significance:
                logger.info("Performing statistical analysis...")
                stats_results = await self._statistical_analysis(function)
                benchmark_results['results']['statistical_analysis'] = stats_results
            
            # 9. Regression detection
            if self.config.regression_detection:
                logger.info("Checking for performance regressions...")
                regression_results = await self._detect_regressions(basic_metrics, function.__name__)
                benchmark_results['results']['regression_analysis'] = regression_results
            
            # 10. Generate optimization recommendations
            if self.config.optimization_suggestions:
                logger.info("Generating optimization recommendations...")
                
                # Create comprehensive metrics object
                comprehensive_metrics = self._create_comprehensive_metrics(benchmark_results)
                
                recommendations = self.optimization_engine.analyze_performance_bottlenecks(
                    comprehensive_metrics
                )
                
                optimization_plan = self.optimization_engine.generate_optimization_plan(
                    recommendations
                )
                
                benchmark_results['results']['optimization_recommendations'] = optimization_plan
            
            # 11. Store results in database
            await self._store_benchmark_results(benchmark_results)
            
            # 12. Generate comprehensive report
            report = self._generate_benchmark_report(benchmark_results)
            benchmark_results['comprehensive_report'] = report
            
            logger.info(f"Comprehensive benchmark completed for {function.__name__}")
            
            return benchmark_results
        
        except Exception as e:
            logger.error(f"Comprehensive benchmark failed: {e}")
            benchmark_results['error'] = str(e)
            return benchmark_results
    
    async def _basic_performance_profile(self, function: Callable) -> Dict[str, Any]:
        """Basic performance profiling"""
        times = []
        memory_peaks = []
        cpu_usage = []
        
        # Warmup runs
        for _ in range(self.config.warmup_iterations):
            function()
        
        # Measurement runs
        for _ in range(self.config.measurement_iterations):
            # Memory tracking
            tracemalloc.start()
            
            # CPU usage before
            process = psutil.Process()
            cpu_before = process.cpu_percent()
            
            # Time measurement
            start_time = time.perf_counter()
            
            try:
                function()
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                
                # Memory peak
                current, peak = tracemalloc.get_traced_memory()
                memory_peaks.append(peak)
                
                # CPU usage after
                cpu_after = process.cpu_percent()
                cpu_usage.append(cpu_after - cpu_before)
                
            except Exception as e:
                logger.warning(f"Measurement iteration failed: {e}")
            finally:
                tracemalloc.stop()
        
        # Calculate statistics
        if not times:
            return {'error': 'No successful measurements'}
        
        return {
            'execution_time': {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'median': np.median(times),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99)
            },
            'memory_usage': {
                'peak_mean': np.mean(memory_peaks),
                'peak_max': np.max(memory_peaks),
                'peak_std': np.std(memory_peaks)
            },
            'cpu_usage': {
                'mean': np.mean(cpu_usage),
                'max': np.max(cpu_usage),
                'std': np.std(cpu_usage)
            },
            'measurements_count': len(times),
            'success_rate': len(times) / self.config.measurement_iterations
        }
    
    async def _detailed_memory_profile(self, function: Callable) -> Dict[str, Any]:
        """Detailed memory profiling"""
        memory_results = {
            'line_by_line': None,
            'object_tracking': None,
            'garbage_collection': None
        }
        
        # Line-by-line memory profiling
        if LINE_PROFILER_AVAILABLE:
            try:
                profiler = LineProfiler()
                profiler.add_function(function)
                profiler.enable()
                
                function()
                
                profiler.disable()
                
                # Get line-by-line results
                stream = io.StringIO()
                profiler.print_stats(stream)
                memory_results['line_by_line'] = stream.getvalue()
                
            except Exception as e:
                memory_results['line_by_line'] = f'Error: {e}'
        
        # Object tracking
        try:
            tracker = pympler.tracker.SummaryTracker()
            
            function()
            
            diff = tracker.diff()
            memory_results['object_tracking'] = {
                'summary': str(diff[:10]),  # Top 10 differences
                'total_objects': len(diff)
            }
            
        except Exception as e:
            memory_results['object_tracking'] = f'Error: {e}'
        
        # Garbage collection analysis
        try:
            gc_before = gc.get_stats()
            gc.collect()
            
            function()
            
            gc_after = gc.get_stats()
            gc.collect()
            
            memory_results['garbage_collection'] = {
                'collections_before': gc_before,
                'collections_after': gc_after,
                'objects_collected': sum(stat['collected'] for stat in gc_after) - 
                                   sum(stat['collected'] for stat in gc_before)
            }
            
        except Exception as e:
            memory_results['garbage_collection'] = f'Error: {e}'
        
        return memory_results
    
    async def _detailed_cpu_profile(self, function: Callable) -> Dict[str, Any]:
        """Detailed CPU profiling"""
        cpu_results = {
            'cprofile_stats': None,
            'hotspots': None,
            'call_graph': None
        }
        
        # cProfile profiling
        try:
            profiler = cProfile.Profile()
            profiler.enable()
            
            function()
            
            profiler.disable()
            
            # Get stats
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative').print_stats(20)
            
            cpu_results['cprofile_stats'] = stats_stream.getvalue()
            
            # Extract hotspots
            hotspots = []
            stats.sort_stats('tottime')
            for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
                if tt > 0.001:  # Only functions taking > 1ms
                    hotspots.append({
                        'function': f'{func_info[0]}:{func_info[1]}({func_info[2]})',
                        'total_time': tt,
                        'calls': nc,
                        'time_per_call': tt / nc if nc > 0 else 0
                    })
            
            cpu_results['hotspots'] = hotspots[:10]  # Top 10
            
        except Exception as e:
            cpu_results['cprofile_stats'] = f'Error: {e}'
        
        return cpu_results
    
    async def _detailed_gpu_profile(self, function: Callable) -> Dict[str, Any]:
        """Detailed GPU profiling"""
        if not GPU_AVAILABLE:
            return {'error': 'GPU profiling not available'}
        
        gpu_results = {
            'utilization': None,
            'memory_usage': None,
            'kernel_analysis': None
        }
        
        try:
            # PyTorch profiler
            if hasattr(torch, 'profiler'):
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as profiler:
                    function()
                
                # Export trace
                trace_file = f"gpu_trace_{int(time.time())}.json"
                profiler.export_chrome_trace(trace_file)
                
                gpu_results['trace_file'] = trace_file
                gpu_results['key_averages'] = str(profiler.key_averages())
                
        except Exception as e:
            gpu_results['error'] = str(e)
        
        # NVIDIA-ML metrics
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Before execution
                util_before = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                function()
                
                # After execution
                util_after = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_results['nvidia_metrics'] = {
                    'gpu_utilization_change': util_after.gpu - util_before.gpu,
                    'memory_utilization_change': util_after.memory - util_before.memory,
                    'memory_used_change': mem_after.used - mem_before.used
                }
                
            except Exception as e:
                gpu_results['nvidia_metrics'] = f'Error: {e}'
        
        return gpu_results
    
    async def _detailed_io_profile(self, function: Callable) -> Dict[str, Any]:
        """Detailed I/O profiling"""
        io_results = {
            'disk_io': None,
            'network_io': None
        }
        
        # Disk I/O monitoring
        try:
            process = psutil.Process()
            io_before = process.io_counters()
            
            function()
            
            io_after = process.io_counters()
            
            io_results['disk_io'] = {
                'read_bytes': io_after.read_bytes - io_before.read_bytes,
                'write_bytes': io_after.write_bytes - io_before.write_bytes,
                'read_count': io_after.read_count - io_before.read_count,
                'write_count': io_after.write_count - io_before.write_count
            }
            
        except Exception as e:
            io_results['disk_io'] = f'Error: {e}'
        
        # Network I/O monitoring  
        if self.config.network_profiling:
            try:
                net_before = psutil.net_io_counters()
                
                function()
                
                net_after = psutil.net_io_counters()
                
                io_results['network_io'] = {
                    'bytes_sent': net_after.bytes_sent - net_before.bytes_sent,
                    'bytes_recv': net_after.bytes_recv - net_before.bytes_recv,
                    'packets_sent': net_after.packets_sent - net_before.packets_sent,
                    'packets_recv': net_after.packets_recv - net_before.packets_recv
                }
                
            except Exception as e:
                io_results['network_io'] = f'Error: {e}'
        
        return io_results
    
    async def _statistical_analysis(self, function: Callable) -> Dict[str, Any]:
        """Statistical significance analysis"""
        # Collect larger sample for statistical analysis
        sample_size = max(100, self.config.measurement_iterations * 2)
        execution_times = []
        
        for _ in range(sample_size):
            start_time = time.perf_counter()
            try:
                function()
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
            except:
                continue
        
        if len(execution_times) < 30:
            return {'error': 'Insufficient samples for statistical analysis'}
        
        times_array = np.array(execution_times)
        
        # Basic statistics
        stats_results = {
            'sample_size': len(execution_times),
            'mean': np.mean(times_array),
            'std': np.std(times_array),
            'variance': np.var(times_array),
            'skewness': stats.skew(times_array),
            'kurtosis': stats.kurtosis(times_array),
            'coefficient_of_variation': np.std(times_array) / np.mean(times_array)
        }
        
        # Confidence intervals
        confidence_level = self.config.confidence_level
        alpha = 1 - confidence_level
        
        # t-distribution confidence interval
        df = len(execution_times) - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        margin_error = t_critical * (np.std(times_array) / np.sqrt(len(execution_times)))
        
        stats_results['confidence_interval'] = {
            'level': confidence_level,
            'lower': np.mean(times_array) - margin_error,
            'upper': np.mean(times_array) + margin_error,
            'margin_of_error': margin_error
        }
        
        # Distribution fitting
        try:
            # Test for normal distribution
            shapiro_stat, shapiro_p = stats.shapiro(times_array)
            
            # Test other distributions
            distributions = ['norm', 'expon', 'gamma', 'lognorm']
            distribution_fits = {}
            
            for dist_name in distributions:
                try:
                    dist = getattr(stats, dist_name)
                    params = dist.fit(times_array)
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.kstest(times_array, 
                                                lambda x: dist.cdf(x, *params))
                    
                    distribution_fits[dist_name] = {
                        'parameters': params,
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        'aic': 2 * len(params) - 2 * np.sum(dist.logpdf(times_array, *params))
                    }
                except:
                    continue
            
            stats_results['distribution_analysis'] = {
                'normality_test': {
                    'shapiro_statistic': shapiro_stat,
                    'shapiro_p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                },
                'distribution_fits': distribution_fits
            }
            
        except Exception as e:
            stats_results['distribution_analysis'] = f'Error: {e}'
        
        # Outlier detection
        if self.config.outlier_detection:
            q1 = np.percentile(times_array, 25)
            q3 = np.percentile(times_array, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = times_array[(times_array < lower_bound) | (times_array > upper_bound)]
            
            stats_results['outlier_analysis'] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(times_array)) * 100,
                'outlier_bounds': {'lower': lower_bound, 'upper': upper_bound},
                'outlier_values': outliers.tolist()
            }
        
        return stats_results
    
    async def _detect_regressions(self, current_metrics: Dict[str, Any], 
                                function_name: str) -> Dict[str, Any]:
        """Detect performance regressions"""
        # Get historical data
        historical_data = await self._get_historical_data(function_name, limit=100)
        
        if len(historical_data) < 10:
            return {'message': 'Insufficient historical data for regression detection'}
        
        # Extract performance metrics
        historical_times = [record[2] for record in historical_data]  # execution_time column
        current_time = current_metrics.get('execution_time', {}).get('mean', 0)
        
        if current_time <= 0:
            return {'error': 'Invalid current execution time'}
        
        # Statistical regression detection
        historical_array = np.array(historical_times)
        
        # Calculate z-score
        historical_mean = np.mean(historical_array)
        historical_std = np.std(historical_array)
        
        if historical_std > 0:
            z_score = (current_time - historical_mean) / historical_std
            
            regression_detected = z_score > 2  # 2-sigma threshold
            
            regression_results = {
                'regression_detected': regression_detected,
                'z_score': z_score,
                'current_time': current_time,
                'historical_mean': historical_mean,
                'historical_std': historical_std,
                'performance_change_percent': ((current_time - historical_mean) / historical_mean) * 100,
                'severity': 'high' if z_score > 3 else 'medium' if z_score > 2 else 'low'
            }
            
            # Trend analysis
            if len(historical_times) >= 10:
                # Linear regression on recent history
                x = np.arange(len(historical_times))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, historical_times)
                
                regression_results['trend_analysis'] = {
                    'slope': slope,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'trend': 'degrading' if slope > 0 else 'improving' if slope < 0 else 'stable',
                    'trend_strength': abs(r_value)
                }
        else:
            regression_results = {'error': 'Zero variance in historical data'}
        
        return regression_results
    
    async def _get_historical_data(self, function_name: str, limit: int = 100) -> List[Tuple]:
        """Get historical performance data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, function_name, execution_time, memory_usage, cpu_usage
            FROM benchmarks 
            WHERE function_name = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (function_name, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def _create_comprehensive_metrics(self, benchmark_results: Dict[str, Any]) -> PerformanceMetrics:
        """Create comprehensive metrics object from benchmark results"""
        basic_metrics = benchmark_results['results'].get('basic_metrics', {})
        
        # Extract values with defaults
        execution_time = basic_metrics.get('execution_time', {}).get('mean', 0.0)
        memory_usage = basic_metrics.get('memory_usage', {}).get('peak_mean', 0.0)
        cpu_usage = basic_metrics.get('cpu_usage', {}).get('mean', 0.0)
        
        # Get complexity analysis if available
        complexity_analysis = benchmark_results['results'].get('complexity_analysis', {})
        
        return PerformanceMetrics(
            timestamp=benchmark_results['timestamp'],
            function_name=benchmark_results['function_name'],
            execution_time=execution_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_peak=basic_metrics.get('memory_usage', {}).get('peak_max', 0.0),
            io_read=0,  # Would be extracted from I/O profiling
            io_write=0,
            network_sent=0,
            network_recv=0,
            gpu_usage=0.0,  # Would be extracted from GPU profiling
            gpu_memory=0.0,
            cache_hits=1000,  # Placeholder
            cache_misses=100,   # Placeholder
            context_switches=0,
            page_faults=0,
            instructions_retired=0,
            cycles=0,
            branch_misses=0,
            energy_consumption=1.0,  # Placeholder
            thermal_state=25.0,      # Placeholder
            algorithmic_complexity=complexity_analysis.get('complexity_fits', {}),
            bottlenecks=[],
            optimization_score=0.8   # Placeholder
        )
    
    async def _store_benchmark_results(self, benchmark_results: Dict[str, Any]):
        """Store benchmark results in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            basic_metrics = benchmark_results['results'].get('basic_metrics', {})
            
            cursor.execute('''
                INSERT INTO benchmarks (
                    function_name, execution_time, memory_usage, cpu_usage, 
                    gpu_usage, optimization_score, performance_index, 
                    configuration, environment_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                benchmark_results['function_name'],
                basic_metrics.get('execution_time', {}).get('mean', 0.0),
                basic_metrics.get('memory_usage', {}).get('peak_mean', 0.0),
                basic_metrics.get('cpu_usage', {}).get('mean', 0.0),
                0.0,  # gpu_usage placeholder
                0.8,  # optimization_score placeholder
                0.9,  # performance_index placeholder
                json.dumps(benchmark_results['configuration']),
                json.dumps({'python_version': sys.version, 'platform': sys.platform})
            ))
            
            benchmark_id = cursor.lastrowid
            
            # Store optimization recommendations
            optimization_results = benchmark_results['results'].get('optimization_recommendations', {})
            for category, recommendations in optimization_results.get('recommendations_by_category', {}).items():
                for rec in recommendations:
                    cursor.execute('''
                        INSERT INTO optimization_recommendations (
                            benchmark_id, category, priority, description, 
                            expected_improvement, implementation_complexity
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        benchmark_id,
                        rec.get('category', category),
                        rec.get('priority', 'medium'),
                        rec.get('description', ''),
                        rec.get('expected_improvement', 0.0),
                        rec.get('implementation_complexity', 'moderate')
                    ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored benchmark results with ID {benchmark_id}")
            
        except Exception as e:
            logger.error(f"Failed to store benchmark results: {e}")
    
    def _generate_benchmark_report(self, benchmark_results: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report"""
        function_name = benchmark_results['function_name']
        timestamp = benchmark_results['timestamp']
        
        report = f"""
# Comprehensive Performance Benchmark Report

**Function:** `{function_name}`  
**Timestamp:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
"""
        
        basic_metrics = benchmark_results['results'].get('basic_metrics', {})
        
        if basic_metrics:
            exec_time = basic_metrics.get('execution_time', {})
            memory = basic_metrics.get('memory_usage', {})
            cpu = basic_metrics.get('cpu_usage', {})
            
            report += f"""
- **Average Execution Time:** {exec_time.get('mean', 0):.6f} seconds
- **Peak Memory Usage:** {memory.get('peak_max', 0) / 1024**2:.2f} MB
- **Average CPU Usage:** {cpu.get('mean', 0):.2f}%
- **Success Rate:** {basic_metrics.get('success_rate', 0):.1%}
"""
        
        # Complexity Analysis
        complexity = benchmark_results['results'].get('complexity_analysis', {})
        if complexity and 'best_fit_complexity' in complexity:
            report += f"""
## Algorithmic Complexity Analysis
- **Best Fit Complexity:** {complexity['best_fit_complexity']}
- **R-squared:** {complexity.get('best_fit_metrics', {}).get('r_squared', 0):.4f}
"""
        
        # Statistical Analysis
        stats_analysis = benchmark_results['results'].get('statistical_analysis', {})
        if stats_analysis and 'confidence_interval' in stats_analysis:
            ci = stats_analysis['confidence_interval']
            report += f"""
## Statistical Analysis
- **Sample Size:** {stats_analysis.get('sample_size', 0)}
- **Coefficient of Variation:** {stats_analysis.get('coefficient_of_variation', 0):.4f}
- **95% Confidence Interval:** [{ci.get('lower', 0):.6f}, {ci.get('upper', 0):.6f}]
"""
        
        # Regression Analysis
        regression = benchmark_results['results'].get('regression_analysis', {})
        if regression and not regression.get('error'):
            report += f"""
## Regression Analysis
- **Regression Detected:** {'Yes' if regression.get('regression_detected', False) else 'No'}
- **Performance Change:** {regression.get('performance_change_percent', 0):+.1f}%
- **Z-Score:** {regression.get('z_score', 0):.2f}
"""
        
        # Optimization Recommendations
        optimization = benchmark_results['results'].get('optimization_recommendations', {})
        if optimization and 'summary' in optimization:
            summary = optimization['summary']
            report += f"""
## Optimization Recommendations
- **Total Recommendations:** {summary.get('total_recommendations', 0)}
- **Expected Improvement:** {summary.get('expected_improvement', 0):.1%}
- **Estimated Development Time:** {summary.get('estimated_development_hours', 0)} hours
- **Overall Risk:** {summary.get('overall_risk', 'unknown')}

### Quick Wins
"""
            
            for quick_win in optimization.get('quick_wins', []):
                report += f"- {quick_win.get('description', 'N/A')}\n"
            
            report += "\n### High Impact Changes\n"
            for high_impact in optimization.get('high_impact_changes', []):
                report += f"- {high_impact.get('description', 'N/A')}\n"
        
        report += f"""
## Configuration
- **Warmup Iterations:** {benchmark_results['configuration'].get('warmup_iterations', 0)}
- **Measurement Iterations:** {benchmark_results['configuration'].get('measurement_iterations', 0)}
- **Memory Profiling:** {'Enabled' if benchmark_results['configuration'].get('memory_profiling', False) else 'Disabled'}
- **CPU Profiling:** {'Enabled' if benchmark_results['configuration'].get('cpu_profiling', False) else 'Disabled'}
- **Statistical Analysis:** {'Enabled' if benchmark_results['configuration'].get('statistical_significance', False) else 'Disabled'}

---
*Report generated by Advanced Performance Benchmarking Suite*
"""
        
        return report

# Example usage and demonstration
async def main():
    """Demonstration of advanced performance benchmarking"""
    
    # Example function to benchmark
    def geological_data_processing(data_size: int = 10000):
        """Simulate geological data processing"""
        import numpy as np
        import time
        
        # Simulate seismic data
        seismic_data = np.random.random((data_size, 3))
        
        # Simulate processing with various complexities
        # Linear operation
        processed_1 = seismic_data * 2.5
        
        # Quadratic operation (matrix multiplication)
        if data_size <= 5000:  # Avoid memory issues for large sizes
            processed_2 = np.dot(seismic_data.T, seismic_data)
        
        # Some I/O simulation
        temp_file = f"temp_data_{int(time.time())}.npy"
        np.save(temp_file, processed_1)
        loaded_data = np.load(temp_file)
        
        # Cleanup
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return np.mean(loaded_data)
    
    # Input generator for complexity analysis
    def input_generator(size: int) -> int:
        return size
    
    # Configure benchmarking
    config = BenchmarkConfiguration(
        warmup_iterations=3,
        measurement_iterations=20,
        memory_profiling=True,
        cpu_profiling=True,
        gpu_profiling=False,
        io_profiling=True,
        algorithmic_analysis=True,
        statistical_significance=True,
        regression_detection=True,
        optimization_suggestions=True,
        real_time_monitoring=False  # Disabled for demo
    )
    
    # Initialize benchmarking suite
    benchmark_suite = AdvancedBenchmarkingSuite(config)
    
    print("🚀 Starting Advanced Performance Benchmarking Suite...")
    print("=" * 80)
    
    try:
        # Execute comprehensive benchmark
        results = await benchmark_suite.comprehensive_benchmark(
            function=geological_data_processing,
            input_generator=input_generator,
            input_sizes=[100, 500, 1000, 2000, 5000]  # Different data sizes
        )
        
        print(f"\n✅ Benchmarking completed successfully!")
        
        # Display key results
        basic_metrics = results['results'].get('basic_metrics', {})
        if basic_metrics:
            exec_time = basic_metrics.get('execution_time', {})
            print(f"Average Execution Time: {exec_time.get('mean', 0):.6f}s")
            print(f"Standard Deviation: {exec_time.get('std', 0):.6f}s")
            
            memory = basic_metrics.get('memory_usage', {})
            print(f"Peak Memory Usage: {memory.get('peak_max', 0) / 1024**2:.2f} MB")
        
        # Display complexity analysis
        complexity = results['results'].get('complexity_analysis', {})
        if complexity and 'best_fit_complexity' in complexity:
            print(f"Best Fit Complexity: {complexity['best_fit_complexity']}")
            print(f"R-squared: {complexity.get('best_fit_metrics', {}).get('r_squared', 0):.4f}")
        
        # Display optimization recommendations
        optimization = results['results'].get('optimization_recommendations', {})
        if optimization:
            summary = optimization.get('summary', {})
            print(f"\nOptimization Summary:")
            print(f"- Total Recommendations: {summary.get('total_recommendations', 0)}")
            print(f"- Expected Improvement: {summary.get('expected_improvement', 0):.1%}")
            print(f"- Development Time: {summary.get('estimated_development_hours', 0)} hours")
        
        # Display comprehensive report
        if 'comprehensive_report' in results:
            print("\n" + "="*80)
            print("COMPREHENSIVE BENCHMARK REPORT")
            print("="*80)
            print(results['comprehensive_report'])
        
        print(f"\n📊 Results stored in database: {benchmark_suite.db_path}")
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())