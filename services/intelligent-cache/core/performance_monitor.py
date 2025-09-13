"""
Advanced Cache Performance Monitoring and Analytics System

This module implements comprehensive performance monitoring, analytics, and
automated optimization for the intelligent caching system with geological
domain awareness.

Features:
- Real-time performance monitoring and metrics collection
- Advanced analytics and trend analysis
- Automated performance optimization recommendations
- Geological domain-aware performance analysis
- Cost optimization and resource management
- Predictive performance modeling
- Anomaly detection and alerting
- Performance visualization and reporting
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .cache_manager import CacheLevel, DataType, IntelligentCacheManager


class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    HIT_RATE = "hit_rate"
    MEMORY_USAGE = "memory_usage"
    NETWORK_IO = "network_io"
    CPU_USAGE = "cpu_usage"
    COST = "cost"
    GEOLOGICAL_ACCURACY = "geological_accuracy"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    metric_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    cache_level: Optional[CacheLevel] = None
    data_type: Optional[DataType] = None
    geological_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    severity: AlertSeverity
    metric_type: MetricType
    message: str
    threshold_value: float
    actual_value: float
    timestamp: datetime
    geological_context: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    recommendation_id: str
    category: str
    priority: int  # 1-10, 10 being highest
    description: str
    expected_improvement: float
    implementation_effort: str  # "low", "medium", "high"
    geological_impact: str
    cost_impact: float
    metrics_affected: List[MetricType]
    implementation_steps: List[str] = field(default_factory=list)


class GeologicalPerformanceAnalyzer:
    """Analyzes performance with geological domain awareness"""
    
    def __init__(self):
        self.geological_performance_patterns = {
            'volcanic_monitoring': {
                'expected_latency': 0.1,  # 100ms for real-time monitoring
                'expected_hit_rate': 0.85,
                'critical_data_types': [DataType.REAL_TIME_STREAM, DataType.VOLCANIC_STATE],
                'peak_usage_hours': [8, 9, 10, 16, 17, 18]  # Business hours
            },
            'seismic_analysis': {
                'expected_latency': 0.5,  # 500ms for seismic analysis
                'expected_hit_rate': 0.75,
                'critical_data_types': [DataType.SEISMIC_WAVEFORM],
                'peak_usage_hours': [6, 7, 8, 14, 15, 16]  # Analysis hours
            },
            'geological_modeling': {
                'expected_latency': 2.0,  # 2s for complex models
                'expected_hit_rate': 0.90,
                'critical_data_types': [DataType.GEOLOGICAL_MODEL, DataType.SIMULATION_RESULT],
                'peak_usage_hours': [9, 10, 11, 13, 14, 15]  # Modeling hours
            },
            'resource_assessment': {
                'expected_latency': 1.0,  # 1s for resource queries
                'expected_hit_rate': 0.80,
                'critical_data_types': [DataType.GEOLOGICAL_MODEL, DataType.SIMULATION_RESULT],
                'peak_usage_hours': [8, 9, 10, 13, 14, 15, 16]  # Business hours
            }
        }
        
        self.volcanic_alert_thresholds = {
            'low': {'latency': 0.2, 'hit_rate': 0.7},
            'moderate': {'latency': 0.15, 'hit_rate': 0.75},
            'high': {'latency': 0.1, 'hit_rate': 0.8},
            'critical': {'latency': 0.05, 'hit_rate': 0.9}
        }
    
    def analyze_geological_performance(self, metrics: List[PerformanceMetric],
                                     geological_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance with geological domain context"""
        analysis = {
            'domain_performance': {},
            'critical_issues': [],
            'optimization_opportunities': [],
            'geological_impact_score': 0.0
        }
        
        domain = geological_context.get('domain', 'general')
        volcano_id = geological_context.get('volcano_id')
        alert_level = geological_context.get('alert_level', 0)
        
        # Get expected performance for domain
        expected_perf = self.geological_performance_patterns.get(domain, {})
        
        # Analyze latency performance
        latency_metrics = [m for m in metrics if m.metric_type == MetricType.LATENCY]
        if latency_metrics and expected_perf:
            avg_latency = statistics.mean([m.value for m in latency_metrics])
            expected_latency = expected_perf.get('expected_latency', 1.0)
            
            latency_ratio = avg_latency / expected_latency
            analysis['domain_performance']['latency_ratio'] = latency_ratio
            
            if latency_ratio > 2.0:  # More than 2x expected
                analysis['critical_issues'].append({
                    'type': 'high_latency',
                    'severity': 'critical' if alert_level >= 3 else 'warning',
                    'description': f"Latency {latency_ratio:.1f}x higher than expected for {domain}",
                    'geological_impact': 'High - May affect real-time monitoring capabilities'
                })
        
        # Analyze hit rate performance
        hit_rate_metrics = [m for m in metrics if m.metric_type == MetricType.HIT_RATE]
        if hit_rate_metrics and expected_perf:
            avg_hit_rate = statistics.mean([m.value for m in hit_rate_metrics])
            expected_hit_rate = expected_perf.get('expected_hit_rate', 0.8)
            
            analysis['domain_performance']['hit_rate_ratio'] = avg_hit_rate / expected_hit_rate
            
            if avg_hit_rate < expected_hit_rate * 0.8:  # Less than 80% of expected
                analysis['critical_issues'].append({
                    'type': 'low_hit_rate',
                    'severity': 'warning',
                    'description': f"Hit rate {avg_hit_rate:.2f} below expected {expected_hit_rate:.2f}",
                    'geological_impact': 'Medium - May increase data retrieval times'
                })
        
        # Volcanic alert level specific analysis
        if volcano_id and alert_level > 0:
            alert_level_str = ['low', 'moderate', 'high', 'critical'][min(alert_level-1, 3)]
            thresholds = self.volcanic_alert_thresholds.get(alert_level_str, {})
            
            if latency_metrics:
                avg_latency = statistics.mean([m.value for m in latency_metrics])
                threshold_latency = thresholds.get('latency', 1.0)
                
                if avg_latency > threshold_latency:
                    analysis['critical_issues'].append({
                        'type': 'volcanic_alert_latency',
                        'severity': 'critical',
                        'description': f"Latency {avg_latency:.3f}s exceeds {alert_level_str} alert threshold {threshold_latency:.3f}s",
                        'geological_impact': 'Critical - May delay eruption warnings'
                    })
        
        # Calculate geological impact score
        impact_score = 1.0
        for issue in analysis['critical_issues']:
            if issue['severity'] == 'critical':
                impact_score *= 0.5
            elif issue['severity'] == 'warning':
                impact_score *= 0.8
        
        analysis['geological_impact_score'] = impact_score
        
        return analysis


class PerformanceMonitor:
    """Advanced performance monitoring system"""
    
    def __init__(self, cache_manager: IntelligentCacheManager):
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics_buffer: deque = deque(maxlen=10000)  # Last 10k metrics
        self.metrics_history: Dict[MetricType, List[PerformanceMetric]] = defaultdict(list)
        
        # Alert system
        self.alerts: List[PerformanceAlert] = []
        self.alert_thresholds: Dict[MetricType, Dict[str, float]] = {
            MetricType.LATENCY: {'warning': 1.0, 'error': 5.0, 'critical': 10.0},
            MetricType.HIT_RATE: {'warning': 0.7, 'error': 0.5, 'critical': 0.3},
            MetricType.MEMORY_USAGE: {'warning': 0.8, 'error': 0.9, 'critical': 0.95},
            MetricType.CPU_USAGE: {'warning': 0.7, 'error': 0.85, 'critical': 0.95}
        }
        
        # Performance analysis
        self.geological_analyzer = GeologicalPerformanceAnalyzer()
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self.collection_interval = 30  # seconds
        self.retention_period = timedelta(days=7)
        
        # Performance tracking
        self.request_times: deque = deque(maxlen=1000)
        self.cache_operations: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Cost tracking
        self.cost_metrics = {
            'storage_cost': 0.0,
            'compute_cost': 0.0,
            'network_cost': 0.0,
            'total_cost': 0.0
        }
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_enabled:
            asyncio.create_task(self._monitoring_loop())
            asyncio.create_task(self._analysis_loop())
            self.logger.info("Performance monitoring started")
    
    async def record_metric(self, metric_type: MetricType, value: float,
                          cache_level: Optional[CacheLevel] = None,
                          data_type: Optional[DataType] = None,
                          geological_context: Dict[str, Any] = None,
                          metadata: Dict[str, Any] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            metric_id=str(uuid4()),
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            cache_level=cache_level,
            data_type=data_type,
            geological_context=geological_context or {},
            metadata=metadata or {}
        )
        
        # Add to buffer and history
        self.metrics_buffer.append(metric)
        self.metrics_history[metric_type].append(metric)
        
        # Check for alerts
        await self._check_alert_thresholds(metric)
        
        # Clean old metrics
        self._clean_old_metrics()
    
    async def record_cache_operation(self, operation: str, duration: float,
                                   success: bool, cache_level: CacheLevel,
                                   data_type: DataType,
                                   geological_context: Dict[str, Any] = None):
        """Record cache operation performance"""
        self.cache_operations.append({
            'operation': operation,
            'duration': duration,
            'success': success,
            'cache_level': cache_level,
            'data_type': data_type,
            'timestamp': datetime.now(),
            'geological_context': geological_context or {}
        })
        
        # Record as metrics
        await self.record_metric(
            MetricType.LATENCY, duration, cache_level, data_type, geological_context
        )
        
        if not success:
            self.error_counts[f"{operation}_{cache_level.value}"] += 1
    
    async def record_request_performance(self, request_id: str, total_duration: float,
                                       cache_hit: bool, data_size: int,
                                       geological_context: Dict[str, Any] = None):
        """Record overall request performance"""
        self.request_times.append({
            'request_id': request_id,
            'duration': total_duration,
            'cache_hit': cache_hit,
            'data_size': data_size,
            'timestamp': datetime.now(),
            'geological_context': geological_context or {}
        })
        
        # Calculate throughput (requests per second)
        recent_requests = [
            r for r in self.request_times
            if (datetime.now() - r['timestamp']).total_seconds() < 60
        ]
        
        throughput = len(recent_requests) / 60.0 if recent_requests else 0.0
        
        await self.record_metric(
            MetricType.THROUGHPUT, throughput,
            geological_context=geological_context
        )
        
        # Calculate hit rate
        cache_hits = sum(1 for r in recent_requests if r['cache_hit'])
        hit_rate = cache_hits / len(recent_requests) if recent_requests else 0.0
        
        await self.record_metric(
            MetricType.HIT_RATE, hit_rate,
            geological_context=geological_context
        )
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_enabled:
            try:
                await self._collect_system_metrics()
                await self._update_cost_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _analysis_loop(self):
        """Performance analysis loop"""
        while self.monitoring_enabled:
            try:
                await self._analyze_performance_trends()
                await self._generate_optimization_recommendations()
                await asyncio.sleep(300)  # Analyze every 5 minutes
            except Exception as e:
                self.logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(300)
    
    async def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        try:
            # Memory usage
            if hasattr(self.cache_manager, 'get_metrics'):
                cache_metrics = self.cache_manager.get_metrics()
                
                await self.record_metric(
                    MetricType.MEMORY_USAGE,
                    cache_metrics.memory_usage / (1024 * 1024 * 1024)  # GB
                )
                
                await self.record_metric(
                    MetricType.HIT_RATE,
                    cache_metrics.hit_rate
                )
            
            # CPU usage (mock - would use psutil in real implementation)
            cpu_usage = np.random.uniform(0.1, 0.8)  # Mock CPU usage
            await self.record_metric(MetricType.CPU_USAGE, cpu_usage)
            
            # Network I/O (mock)
            network_io = np.random.uniform(100, 1000)  # MB/s
            await self.record_metric(MetricType.NETWORK_IO, network_io)
            
        except Exception as e:
            self.logger.error(f"System metrics collection error: {e}")
    
    async def _update_cost_metrics(self):
        """Update cost-related metrics"""
        try:
            # Calculate storage costs
            if hasattr(self.cache_manager, 'get_metrics'):
                cache_metrics = self.cache_manager.get_metrics()
                storage_gb = cache_metrics.memory_usage / (1024 * 1024 * 1024)
                
                # Mock cost calculation (would use actual cloud pricing)
                storage_cost_per_gb_hour = 0.023  # AWS pricing example
                self.cost_metrics['storage_cost'] = storage_gb * storage_cost_per_gb_hour
            
            # Calculate compute costs based on CPU usage
            recent_cpu_metrics = [
                m for m in self.metrics_history[MetricType.CPU_USAGE]
                if (datetime.now() - m.timestamp).total_seconds() < 3600
            ]
            
            if recent_cpu_metrics:
                avg_cpu = statistics.mean([m.value for m in recent_cpu_metrics])
                compute_cost_per_hour = 0.10  # Mock compute cost
                self.cost_metrics['compute_cost'] = avg_cpu * compute_cost_per_hour
            
            # Calculate network costs
            recent_network_metrics = [
                m for m in self.metrics_history[MetricType.NETWORK_IO]
                if (datetime.now() - m.timestamp).total_seconds() < 3600
            ]
            
            if recent_network_metrics:
                avg_network_gb = statistics.mean([m.value / 1024 for m in recent_network_metrics])
                network_cost_per_gb = 0.09  # Mock network cost
                self.cost_metrics['network_cost'] = avg_network_gb * network_cost_per_gb
            
            # Total cost
            self.cost_metrics['total_cost'] = (
                self.cost_metrics['storage_cost'] +
                self.cost_metrics['compute_cost'] +
                self.cost_metrics['network_cost']
            )
            
            await self.record_metric(MetricType.COST, self.cost_metrics['total_cost'])
            
        except Exception as e:
            self.logger.error(f"Cost metrics update error: {e}")
    
    async def _check_alert_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds alert thresholds"""
        thresholds = self.alert_thresholds.get(metric.metric_type, {})
        
        for severity_str, threshold in thresholds.items():
            severity = AlertSeverity(severity_str)
            
            # Check threshold based on metric type
            alert_triggered = False
            
            if metric.metric_type == MetricType.HIT_RATE:
                alert_triggered = metric.value < threshold  # Low hit rate is bad
            else:
                alert_triggered = metric.value > threshold  # High values are bad
            
            if alert_triggered:
                alert = PerformanceAlert(
                    alert_id=str(uuid4()),
                    severity=severity,
                    metric_type=metric.metric_type,
                    message=f"{metric.metric_type.value} {severity_str}: {metric.value:.3f}",
                    threshold_value=threshold,
                    actual_value=metric.value,
                    timestamp=metric.timestamp,
                    geological_context=metric.geological_context,
                    recommendations=self._generate_alert_recommendations(metric, severity)
                )
                
                self.alerts.append(alert)
                self.logger.warning(f"Performance alert: {alert.message}")
                
                # Keep only recent alerts
                cutoff = datetime.now() - timedelta(hours=24)
                self.alerts = [a for a in self.alerts if a.timestamp > cutoff]
                
                break  # Only trigger highest severity alert
    
    def _generate_alert_recommendations(self, metric: PerformanceMetric,
                                      severity: AlertSeverity) -> List[str]:
        """Generate recommendations for performance alerts"""
        recommendations = []
        
        if metric.metric_type == MetricType.LATENCY:
            if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                recommendations.extend([
                    "Increase cache memory allocation",
                    "Enable data compression",
                    "Implement query optimization",
                    "Consider adding more cache levels"
                ])
            else:
                recommendations.extend([
                    "Monitor query patterns",
                    "Consider prefetching optimization"
                ])
        
        elif metric.metric_type == MetricType.HIT_RATE:
            recommendations.extend([
                "Analyze cache eviction policies",
                "Increase cache size",
                "Improve prefetching strategies",
                "Optimize TTL settings"
            ])
        
        elif metric.metric_type == MetricType.MEMORY_USAGE:
            recommendations.extend([
                "Enable data compression",
                "Implement cache eviction",
                "Optimize data structures",
                "Consider distributed caching"
            ])
        
        # Add geological context specific recommendations
        domain = metric.geological_context.get('domain')
        if domain == 'volcanic' and severity == AlertSeverity.CRITICAL:
            recommendations.append("Priority: Volcanic monitoring requires real-time performance")
        elif domain == 'seismic':
            recommendations.append("Consider seismic-specific optimization strategies")
        
        return recommendations
    
    async def _analyze_performance_trends(self):
        """Analyze performance trends and patterns"""
        try:
            # Analyze latency trends
            recent_latency = [
                m for m in self.metrics_history[MetricType.LATENCY]
                if (datetime.now() - m.timestamp).total_seconds() < 3600
            ]
            
            if len(recent_latency) > 10:
                latency_values = [m.value for m in recent_latency]
                
                # Calculate trend
                x = np.arange(len(latency_values))
                trend = np.polyfit(x, latency_values, 1)[0]  # Linear trend
                
                if trend > 0.001:  # Increasing latency trend
                    self.logger.warning(f"Increasing latency trend detected: {trend:.6f}s per request")
            
            # Analyze hit rate patterns by geological domain
            domain_hit_rates = defaultdict(list)
            
            recent_hit_rates = [
                m for m in self.metrics_history[MetricType.HIT_RATE]
                if (datetime.now() - m.timestamp).total_seconds() < 3600
            ]
            
            for metric in recent_hit_rates:
                domain = metric.geological_context.get('domain', 'general')
                domain_hit_rates[domain].append(metric.value)
            
            # Identify domains with poor performance
            for domain, hit_rates in domain_hit_rates.items():
                if hit_rates:
                    avg_hit_rate = statistics.mean(hit_rates)
                    expected_hit_rate = self.geological_analyzer.geological_performance_patterns.get(
                        domain, {}
                    ).get('expected_hit_rate', 0.8)
                    
                    if avg_hit_rate < expected_hit_rate * 0.8:
                        self.logger.warning(
                            f"Poor hit rate for {domain}: {avg_hit_rate:.3f} "
                            f"(expected: {expected_hit_rate:.3f})"
                        )
        
        except Exception as e:
            self.logger.error(f"Performance trend analysis error: {e}")
    
    async def _generate_optimization_recommendations(self):
        """Generate optimization recommendations based on performance analysis"""
        try:
            recommendations = []
            
            # Analyze recent performance
            recent_metrics = [
                m for m in self.metrics_buffer
                if (datetime.now() - m.timestamp).total_seconds() < 1800  # Last 30 minutes
            ]
            
            if not recent_metrics:
                return
            
            # Group metrics by type
            metrics_by_type = defaultdict(list)
            for metric in recent_metrics:
                metrics_by_type[metric.metric_type].append(metric)
            
            # Latency optimization recommendations
            if MetricType.LATENCY in metrics_by_type:
                latency_metrics = metrics_by_type[MetricType.LATENCY]
                avg_latency = statistics.mean([m.value for m in latency_metrics])
                
                if avg_latency > 1.0:  # High latency
                    recommendations.append(OptimizationRecommendation(
                        recommendation_id=str(uuid4()),
                        category="latency_optimization",
                        priority=8,
                        description="Implement advanced caching strategies to reduce latency",
                        expected_improvement=0.4,  # 40% improvement
                        implementation_effort="medium",
                        geological_impact="High - Improves real-time monitoring capabilities",
                        cost_impact=-0.2,  # 20% cost reduction
                        metrics_affected=[MetricType.LATENCY, MetricType.THROUGHPUT],
                        implementation_steps=[
                            "Enable L1 memory caching for frequently accessed data",
                            "Implement predictive prefetching",
                            "Optimize query execution plans",
                            "Add compression for large datasets"
                        ]
                    ))
            
            # Hit rate optimization recommendations
            if MetricType.HIT_RATE in metrics_by_type:
                hit_rate_metrics = metrics_by_type[MetricType.HIT_RATE]
                avg_hit_rate = statistics.mean([m.value for m in hit_rate_metrics])
                
                if avg_hit_rate < 0.7:  # Low hit rate
                    recommendations.append(OptimizationRecommendation(
                        recommendation_id=str(uuid4()),
                        category="cache_efficiency",
                        priority=7,
                        description="Improve cache hit rate through better eviction policies",
                        expected_improvement=0.3,  # 30% improvement
                        implementation_effort="low",
                        geological_impact="Medium - Better data availability",
                        cost_impact=-0.15,  # 15% cost reduction
                        metrics_affected=[MetricType.HIT_RATE, MetricType.LATENCY],
                        implementation_steps=[
                            "Implement geological-aware eviction policies",
                            "Increase cache size for critical data types",
                            "Optimize TTL settings based on data type",
                            "Implement intelligent prefetching"
                        ]
                    ))
            
            # Memory optimization recommendations
            if MetricType.MEMORY_USAGE in metrics_by_type:
                memory_metrics = metrics_by_type[MetricType.MEMORY_USAGE]
                avg_memory = statistics.mean([m.value for m in memory_metrics])
                
                if avg_memory > 0.8:  # High memory usage
                    recommendations.append(OptimizationRecommendation(
                        recommendation_id=str(uuid4()),
                        category="memory_optimization",
                        priority=6,
                        description="Optimize memory usage through compression and efficient data structures",
                        expected_improvement=0.25,  # 25% improvement
                        implementation_effort="medium",
                        geological_impact="Low - Primarily infrastructure optimization",
                        cost_impact=-0.3,  # 30% cost reduction
                        metrics_affected=[MetricType.MEMORY_USAGE, MetricType.COST],
                        implementation_steps=[
                            "Enable advanced data compression",
                            "Implement data deduplication",
                            "Optimize data serialization formats",
                            "Implement memory-efficient data structures"
                        ]
                    ))
            
            # Cost optimization recommendations
            if MetricType.COST in metrics_by_type:
                cost_metrics = metrics_by_type[MetricType.COST]
                avg_cost = statistics.mean([m.value for m in cost_metrics])
                
                if avg_cost > 10.0:  # High cost per hour
                    recommendations.append(OptimizationRecommendation(
                        recommendation_id=str(uuid4()),
                        category="cost_optimization",
                        priority=5,
                        description="Reduce operational costs through resource optimization",
                        expected_improvement=0.4,  # 40% cost reduction
                        implementation_effort="high",
                        geological_impact="Low - Cost optimization with maintained performance",
                        cost_impact=-0.4,  # 40% cost reduction
                        metrics_affected=[MetricType.COST],
                        implementation_steps=[
                            "Implement auto-scaling based on demand",
                            "Optimize resource allocation",
                            "Use spot instances for non-critical workloads",
                            "Implement intelligent data archiving"
                        ]
                    ))
            
            # Update recommendations list
            self.optimization_recommendations.extend(recommendations)
            
            # Keep only recent recommendations
            cutoff = datetime.now() - timedelta(days=1)
            self.optimization_recommendations = [
                r for r in self.optimization_recommendations
                if any(m.timestamp > cutoff for m in recent_metrics)
            ]
            
        except Exception as e:
            self.logger.error(f"Optimization recommendation generation error: {e}")
    
    def _clean_old_metrics(self):
        """Clean old metrics to manage memory usage"""
        cutoff = datetime.now() - self.retention_period
        
        for metric_type in self.metrics_history:
            self.metrics_history[metric_type] = [
                m for m in self.metrics_history[metric_type]
                if m.timestamp > cutoff
            ]
    
    def get_performance_summary(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if time_range is None:
            time_range = timedelta(hours=1)
        
        cutoff = datetime.now() - time_range
        recent_metrics = [m for m in self.metrics_buffer if m.timestamp > cutoff]
        
        summary = {
            'time_range': str(time_range),
            'total_metrics': len(recent_metrics),
            'metrics_by_type': {},
            'alerts': len([a for a in self.alerts if a.timestamp > cutoff]),
            'cost_metrics': self.cost_metrics.copy(),
            'recommendations': len(self.optimization_recommendations),
            'geological_analysis': {}
        }
        
        # Summarize metrics by type
        for metric_type in MetricType:
            type_metrics = [m for m in recent_metrics if m.metric_type == metric_type]
            if type_metrics:
                values = [m.value for m in type_metrics]
                summary['metrics_by_type'][metric_type.value] = {
                    'count': len(values),
                    'avg': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0
                }
        
        # Geological domain analysis
        domain_metrics = defaultdict(list)
        for metric in recent_metrics:
            domain = metric.geological_context.get('domain', 'general')
            domain_metrics[domain].append(metric)
        
        for domain, metrics in domain_metrics.items():
            if metrics:
                geological_analysis = self.geological_analyzer.analyze_geological_performance(
                    metrics, {'domain': domain}
                )
                summary['geological_analysis'][domain] = geological_analysis
        
        return summary
    
    def get_optimization_recommendations(self, priority_threshold: int = 5) -> List[OptimizationRecommendation]:
        """Get optimization recommendations above priority threshold"""
        return [
            r for r in self.optimization_recommendations
            if r.priority >= priority_threshold
        ]
    
    def get_recent_alerts(self, severity: Optional[AlertSeverity] = None,
                         time_range: Optional[timedelta] = None) -> List[PerformanceAlert]:
        """Get recent performance alerts"""
        if time_range is None:
            time_range = timedelta(hours=24)
        
        cutoff = datetime.now() - time_range
        alerts = [a for a in self.alerts if a.timestamp > cutoff]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)


# Configuration and factory
class PerformanceMonitorConfig(BaseModel):
    """Performance monitor configuration"""
    enabled: bool = Field(default=True, description="Enable performance monitoring")
    collection_interval: int = Field(default=30, description="Metrics collection interval in seconds")
    retention_days: int = Field(default=7, description="Metrics retention period in days")
    alert_thresholds: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "latency": {"warning": 1.0, "error": 5.0, "critical": 10.0},
            "hit_rate": {"warning": 0.7, "error": 0.5, "critical": 0.3},
            "memory_usage": {"warning": 0.8, "error": 0.9, "critical": 0.95}
        },
        description="Alert thresholds for different metrics"
    )
    geological_optimization: bool = Field(default=True, description="Enable geological domain optimization")


async def create_performance_monitor(cache_manager: IntelligentCacheManager,
                                   config: PerformanceMonitorConfig) -> PerformanceMonitor:
    """Factory function to create performance monitor"""
    monitor = PerformanceMonitor(cache_manager)
    monitor.monitoring_enabled = config.enabled
    monitor.collection_interval = config.collection_interval
    monitor.retention_period = timedelta(days=config.retention_days)
    
    # Update alert thresholds
    for metric_name, thresholds in config.alert_thresholds.items():
        try:
            metric_type = MetricType(metric_name)
            monitor.alert_thresholds[metric_type] = {
                AlertSeverity(severity): threshold
                for severity, threshold in thresholds.items()
            }
        except ValueError:
            monitor.logger.warning(f"Invalid metric type or severity: {metric_name}")
    
    await monitor.start_monitoring()
    return monitor


# Example usage and testing
if __name__ == "__main__":
    async def test_performance_monitor():
        """Test the performance monitor"""
        from .cache_manager import create_cache_manager, CacheConfig
        
        # Create cache manager
        cache_config = CacheConfig()
        cache_manager = await create_cache_manager(cache_config)
        
        # Create performance monitor
        monitor_config = PerformanceMonitorConfig()
        monitor = await create_performance_monitor(cache_manager, monitor_config)
        
        # Simulate some metrics
        geological_context = {
            "domain": "volcanic",
            "volcano_id": "kilauea_001",
            "alert_level": 3
        }
        
        # Record some test metrics
        await monitor.record_metric(MetricType.LATENCY, 0.5, geological_context=geological_context)
        await monitor.record_metric(MetricType.HIT_RATE, 0.85, geological_context=geological_context)
        await monitor.record_metric(MetricType.MEMORY_USAGE, 0.6, geological_context=geological_context)
        
        # Record cache operation
        await monitor.record_cache_operation(
            "get", 0.1, True, CacheLevel.L1_MEMORY, DataType.VOLCANIC_STATE, geological_context
        )
        
        # Record request performance
        await monitor.record_request_performance(
            "test_request", 0.2, True, 1024, geological_context
        )
        
        # Wait for analysis
        await asyncio.sleep(2)
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        print(f"Performance summary: {json.dumps(summary, indent=2, default=str)}")
        
        # Get recommendations
        recommendations = monitor.get_optimization_recommendations()
        print(f"Optimization recommendations: {len(recommendations)}")
        
        for rec in recommendations:
            print(f"- {rec.category}: {rec.description} (Priority: {rec.priority})")
        
        # Get alerts
        alerts = monitor.get_recent_alerts()
        print(f"Recent alerts: {len(alerts)}")
        
        for alert in alerts:
            print(f"- {alert.severity.value}: {alert.message}")
    
    # Run test
    asyncio.run(test_performance_monitor())