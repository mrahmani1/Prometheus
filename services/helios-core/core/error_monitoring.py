"""
Monitoring and metrics module for hierarchical error management system

This module provides comprehensive monitoring, metrics collection, and
reporting capabilities for the Prometheus AI error management system.

Requirements: 12.3, 12.4
"""

import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

try:
    from .error_management import ErrorInfo, ErrorCategory, ErrorSeverity, ServiceDegradationLevel
except ImportError:
    from error_management import ErrorInfo, ErrorCategory, ErrorSeverity, ServiceDegradationLevel

logger = logging.getLogger(__name__)


@dataclass
class ErrorMetrics:
    """Error metrics data structure"""
    
    # Basic counts
    total_errors: int = 0
    resolved_errors: int = 0
    unresolved_errors: int = 0
    
    # Category breakdown
    category_counts: Dict[str, int] = field(default_factory=dict)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    
    # Timing metrics
    average_resolution_time: float = 0.0
    max_resolution_time: float = 0.0
    min_resolution_time: float = float('inf')
    
    # Recovery metrics
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    recovery_success_rate: float = 0.0
    
    # Service metrics
    service_error_counts: Dict[str, int] = field(default_factory=dict)
    operation_error_counts: Dict[str, int] = field(default_factory=dict)
    
    # Time-based metrics
    errors_per_hour: Dict[int, int] = field(default_factory=dict)
    errors_per_day: Dict[str, int] = field(default_factory=dict)
    
    # Degradation metrics
    degradation_events: int = 0
    current_degradation_level: str = "full_service"
    time_in_degradation: float = 0.0


@dataclass
class SystemHealthMetrics:
    """System health metrics"""
    
    overall_health_score: float = 100.0  # 0-100 scale
    error_rate_24h: float = 0.0
    critical_error_count: int = 0
    system_availability: float = 100.0  # Percentage
    mean_time_to_recovery: float = 0.0
    mean_time_between_failures: float = 0.0
    
    # Component health scores
    component_health: Dict[str, float] = field(default_factory=dict)
    
    # Trend indicators
    error_trend: str = "stable"  # increasing, decreasing, stable
    performance_trend: str = "stable"
    
    # Alerts
    active_alerts: int = 0
    alert_history_24h: int = 0


class ErrorMetricsCollector:
    """Collects and aggregates error metrics"""
    
    def __init__(self, retention_hours: int = 168):  # 7 days default
        self.retention_hours = retention_hours
        self.error_history: deque = deque(maxlen=10000)
        self.metrics_cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=5)
        
        # Real-time counters
        self.realtime_counters = {
            'total_errors': 0,
            'resolved_errors': 0,
            'failed_recoveries': 0,
            'degradation_events': 0
        }
        
        # Time-series data
        self.hourly_error_counts = defaultdict(int)
        self.daily_error_counts = defaultdict(int)
        
        logger.info("Error metrics collector initialized")
    
    def record_error(self, error_info: ErrorInfo):
        """Record an error for metrics collection"""
        self.error_history.append({
            'error_info': error_info,
            'recorded_at': datetime.now()
        })
        
        # Update real-time counters
        self.realtime_counters['total_errors'] += 1
        
        # Update time-series data
        now = datetime.now()
        hour_key = now.hour
        day_key = now.strftime('%Y-%m-%d')
        
        self.hourly_error_counts[hour_key] += 1
        self.daily_error_counts[day_key] += 1
        
        # Clear cache to force recalculation
        self._clear_cache()
        
        logger.debug(f"Recorded error {error_info.error_id} for metrics")
    
    def record_error_resolution(self, error_id: str, resolution_time: float, success: bool):
        """Record error resolution for metrics"""
        if success:
            self.realtime_counters['resolved_errors'] += 1
        else:
            self.realtime_counters['failed_recoveries'] += 1
        
        # Update error history with resolution info
        for record in reversed(self.error_history):
            if record['error_info'].error_id == error_id:
                record['resolved_at'] = datetime.now()
                record['resolution_time'] = resolution_time
                record['resolution_success'] = success
                break
        
        self._clear_cache()
        logger.debug(f"Recorded resolution for error {error_id}: success={success}, time={resolution_time:.2f}s")
    
    def record_degradation_event(self, from_level: ServiceDegradationLevel, to_level: ServiceDegradationLevel):
        """Record service degradation event"""
        self.realtime_counters['degradation_events'] += 1
        
        degradation_record = {
            'timestamp': datetime.now(),
            'from_level': from_level.value,
            'to_level': to_level.value,
            'event_type': 'degradation'
        }
        
        self.error_history.append({
            'degradation_event': degradation_record,
            'recorded_at': datetime.now()
        })
        
        self._clear_cache()
        logger.info(f"Recorded degradation event: {from_level.value} -> {to_level.value}")
    
    def get_error_metrics(self, time_window_hours: int = 24) -> ErrorMetrics:
        """Get comprehensive error metrics"""
        cache_key = f"error_metrics_{time_window_hours}h"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]
        
        # Calculate metrics
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        relevant_records = [
            record for record in self.error_history
            if record['recorded_at'] >= cutoff_time and 'error_info' in record
        ]
        
        metrics = ErrorMetrics()
        
        if not relevant_records:
            self._cache_result(cache_key, metrics)
            return metrics
        
        # Basic counts
        metrics.total_errors = len(relevant_records)
        resolved_records = [r for r in relevant_records if r.get('resolution_success') is not None]
        metrics.resolved_errors = len([r for r in resolved_records if r['resolution_success']])
        metrics.unresolved_errors = metrics.total_errors - len(resolved_records)
        
        # Category and severity breakdown
        for record in relevant_records:
            error_info = record['error_info']
            
            category = error_info.category.value
            severity = error_info.severity.value
            
            metrics.category_counts[category] = metrics.category_counts.get(category, 0) + 1
            metrics.severity_counts[severity] = metrics.severity_counts.get(severity, 0) + 1
            
            # Service and operation counts
            if error_info.context:
                service = error_info.context.service_name
                operation = error_info.context.operation
                
                metrics.service_error_counts[service] = metrics.service_error_counts.get(service, 0) + 1
                metrics.operation_error_counts[operation] = metrics.operation_error_counts.get(operation, 0) + 1
        
        # Timing metrics
        resolution_times = [r['resolution_time'] for r in resolved_records if 'resolution_time' in r]
        if resolution_times:
            metrics.average_resolution_time = sum(resolution_times) / len(resolution_times)
            metrics.max_resolution_time = max(resolution_times)
            metrics.min_resolution_time = min(resolution_times)
        
        # Recovery metrics
        successful_recoveries = len([r for r in resolved_records if r.get('resolution_success')])
        failed_recoveries = len([r for r in resolved_records if not r.get('resolution_success')])
        
        metrics.successful_recoveries = successful_recoveries
        metrics.failed_recoveries = failed_recoveries
        
        if resolved_records:
            metrics.recovery_success_rate = successful_recoveries / len(resolved_records) * 100
        
        # Time-based distribution
        for record in relevant_records:
            hour = record['recorded_at'].hour
            day = record['recorded_at'].strftime('%Y-%m-%d')
            
            metrics.errors_per_hour[hour] = metrics.errors_per_hour.get(hour, 0) + 1
            metrics.errors_per_day[day] = metrics.errors_per_day.get(day, 0) + 1
        
        # Degradation metrics
        degradation_records = [
            record for record in self.error_history
            if record['recorded_at'] >= cutoff_time and 'degradation_event' in record
        ]
        metrics.degradation_events = len(degradation_records)
        
        self._cache_result(cache_key, metrics)
        return metrics
    
    def get_system_health_metrics(self) -> SystemHealthMetrics:
        """Get system health metrics"""
        cache_key = "system_health"
        
        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]
        
        error_metrics = self.get_error_metrics(24)  # 24-hour window
        health_metrics = SystemHealthMetrics()
        
        # Calculate error rate
        health_metrics.error_rate_24h = error_metrics.total_errors / 24.0
        health_metrics.critical_error_count = error_metrics.severity_counts.get('critical', 0)
        
        # Calculate overall health score (0-100)
        health_score = 100.0
        
        # Deduct points for errors
        health_score -= min(error_metrics.total_errors * 0.5, 30)  # Max 30 points for error count
        health_score -= min(health_metrics.critical_error_count * 5, 25)  # Max 25 points for critical errors
        health_score -= min(error_metrics.failed_recoveries * 2, 20)  # Max 20 points for failed recoveries
        health_score -= min(error_metrics.degradation_events * 3, 15)  # Max 15 points for degradation
        
        health_metrics.overall_health_score = max(0, health_score)
        
        # Calculate availability (assuming 99.9% baseline minus error impact)
        error_impact = min(error_metrics.total_errors * 0.01, 5.0)  # Max 5% impact
        health_metrics.system_availability = max(95.0, 99.9 - error_impact)
        
        # Calculate MTTR (Mean Time To Recovery)
        if error_metrics.resolved_errors > 0:
            health_metrics.mean_time_to_recovery = error_metrics.average_resolution_time
        
        # Calculate component health scores
        for service, error_count in error_metrics.service_error_counts.items():
            component_score = max(0, 100 - (error_count * 2))
            health_metrics.component_health[service] = component_score
        
        # Determine trends
        health_metrics.error_trend = self._calculate_error_trend()
        health_metrics.performance_trend = self._calculate_performance_trend()
        
        self._cache_result(cache_key, health_metrics)
        return health_metrics
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns and anomalies"""
        cache_key = "error_patterns"
        
        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]
        
        patterns = {
            'temporal_patterns': self._analyze_temporal_patterns(),
            'service_patterns': self._analyze_service_patterns(),
            'error_correlations': self._analyze_error_correlations(),
            'anomalies': self._detect_anomalies(),
            'recommendations': self._generate_pattern_recommendations()
        }
        
        self._cache_result(cache_key, patterns)
        return patterns
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal error patterns"""
        # Get hourly distribution
        hourly_dist = defaultdict(int)
        daily_dist = defaultdict(int)
        
        cutoff_time = datetime.now() - timedelta(hours=168)  # 7 days
        for record in self.error_history:
            if record['recorded_at'] >= cutoff_time and 'error_info' in record:
                hour = record['recorded_at'].hour
                day = record['recorded_at'].strftime('%A')
                
                hourly_dist[hour] += 1
                daily_dist[day] += 1
        
        # Find peak hours and days
        peak_hour = max(hourly_dist.items(), key=lambda x: x[1]) if hourly_dist else (0, 0)
        peak_day = max(daily_dist.items(), key=lambda x: x[1]) if daily_dist else ("Unknown", 0)
        
        return {
            'hourly_distribution': dict(hourly_dist),
            'daily_distribution': dict(daily_dist),
            'peak_error_hour': peak_hour[0],
            'peak_error_day': peak_day[0],
            'temporal_variance': self._calculate_temporal_variance(hourly_dist)
        }
    
    def _analyze_service_patterns(self) -> Dict[str, Any]:
        """Analyze service-specific error patterns"""
        service_patterns = defaultdict(lambda: {
            'error_count': 0,
            'error_types': defaultdict(int),
            'avg_resolution_time': 0,
            'success_rate': 0
        })
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        relevant_records = [
            record for record in self.error_history
            if record['recorded_at'] >= cutoff_time and 'error_info' in record
        ]
        
        for record in relevant_records:
            error_info = record['error_info']
            if error_info.context:
                service = error_info.context.service_name
                pattern = service_patterns[service]
                
                pattern['error_count'] += 1
                pattern['error_types'][error_info.category.value] += 1
                
                if 'resolution_time' in record:
                    pattern['avg_resolution_time'] += record['resolution_time']
                
                if record.get('resolution_success'):
                    pattern['success_rate'] += 1
        
        # Calculate averages
        for service, pattern in service_patterns.items():
            if pattern['error_count'] > 0:
                pattern['avg_resolution_time'] /= pattern['error_count']
                pattern['success_rate'] = (pattern['success_rate'] / pattern['error_count']) * 100
        
        return dict(service_patterns)
    
    def _analyze_error_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between different error types"""
        # Simple correlation analysis
        correlations = {}
        
        # Analyze category co-occurrence
        category_pairs = defaultdict(int)
        
        # Group errors by time windows (5-minute windows)
        time_windows = defaultdict(list)
        
        for record in self.error_history:
            if 'error_info' in record:
                window_key = record['recorded_at'].replace(minute=record['recorded_at'].minute // 5 * 5, second=0, microsecond=0)
                time_windows[window_key].append(record['error_info'].category.value)
        
        # Count co-occurrences
        for window_errors in time_windows.values():
            unique_categories = set(window_errors)
            for cat1 in unique_categories:
                for cat2 in unique_categories:
                    if cat1 != cat2:
                        pair_key = tuple(sorted([cat1, cat2]))
                        category_pairs[pair_key] += 1
        
        correlations['category_correlations'] = dict(category_pairs)
        return correlations
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous error patterns"""
        anomalies = []
        
        # Get recent metrics
        recent_metrics = self.get_error_metrics(1)  # Last hour
        baseline_metrics = self.get_error_metrics(24)  # Last 24 hours
        
        # Check for error rate spikes
        recent_rate = recent_metrics.total_errors
        baseline_rate = baseline_metrics.total_errors / 24
        
        if recent_rate > baseline_rate * 3:  # 3x normal rate
            anomalies.append({
                'type': 'error_rate_spike',
                'severity': 'high',
                'description': f'Error rate spike detected: {recent_rate} errors in last hour vs {baseline_rate:.1f} average',
                'detected_at': datetime.now()
            })
        
        # Check for new error categories
        recent_categories = set(recent_metrics.category_counts.keys())
        baseline_categories = set(baseline_metrics.category_counts.keys())
        new_categories = recent_categories - baseline_categories
        
        if new_categories:
            anomalies.append({
                'type': 'new_error_categories',
                'severity': 'medium',
                'description': f'New error categories detected: {", ".join(new_categories)}',
                'detected_at': datetime.now()
            })
        
        # Check for critical error clusters
        if recent_metrics.severity_counts.get('critical', 0) > 3:
            anomalies.append({
                'type': 'critical_error_cluster',
                'severity': 'critical',
                'description': f'Multiple critical errors detected: {recent_metrics.severity_counts["critical"]} in last hour',
                'detected_at': datetime.now()
            })
        
        return anomalies
    
    def _generate_pattern_recommendations(self) -> List[str]:
        """Generate recommendations based on error patterns"""
        recommendations = []
        
        error_metrics = self.get_error_metrics(24)
        
        # High error rate recommendations
        if error_metrics.total_errors > 50:
            recommendations.append(
                "High error rate detected. Consider implementing additional monitoring and alerting."
            )
        
        # Low recovery rate recommendations
        if error_metrics.recovery_success_rate < 70:
            recommendations.append(
                "Low recovery success rate. Review and improve error recovery strategies."
            )
        
        # Service-specific recommendations
        for service, count in error_metrics.service_error_counts.items():
            if count > 20:
                recommendations.append(
                    f"High error count in {service} service. Consider service-specific optimizations."
                )
        
        # Category-specific recommendations
        if error_metrics.category_counts.get('data_quality', 0) > 15:
            recommendations.append(
                "High data quality error rate. Implement additional data validation and preprocessing."
            )
        
        if error_metrics.category_counts.get('model_convergence', 0) > 10:
            recommendations.append(
                "Frequent model convergence issues. Review model architecture and training parameters."
            )
        
        return recommendations
    
    def _calculate_error_trend(self) -> str:
        """Calculate error trend (increasing, decreasing, stable)"""
        # Compare last 6 hours vs previous 6 hours
        now = datetime.now()
        recent_cutoff = now - timedelta(hours=6)
        previous_cutoff = now - timedelta(hours=12)
        
        recent_errors = len([
            r for r in self.error_history
            if r['recorded_at'] >= recent_cutoff and 'error_info' in r
        ])
        
        previous_errors = len([
            r for r in self.error_history
            if previous_cutoff <= r['recorded_at'] < recent_cutoff and 'error_info' in r
        ])
        
        if recent_errors > previous_errors * 1.2:
            return "increasing"
        elif recent_errors < previous_errors * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend based on resolution times"""
        # Compare recent resolution times with historical average
        cutoff_time = datetime.now() - timedelta(hours=6)
        recent_times = [
            r['resolution_time'] for r in self.error_history
            if r['recorded_at'] >= cutoff_time and 'resolution_time' in r
        ]
        
        if not recent_times:
            return "stable"
        
        recent_avg = sum(recent_times) / len(recent_times)
        
        # Compare with overall average
        all_times = [
            r['resolution_time'] for r in self.error_history
            if 'resolution_time' in r
        ]
        
        if not all_times:
            return "stable"
        
        overall_avg = sum(all_times) / len(all_times)
        
        if recent_avg > overall_avg * 1.3:
            return "degrading"
        elif recent_avg < overall_avg * 0.7:
            return "improving"
        else:
            return "stable"
    
    def _calculate_temporal_variance(self, hourly_dist: Dict[int, int]) -> float:
        """Calculate temporal variance in error distribution"""
        if not hourly_dist:
            return 0.0
        
        values = list(hourly_dist.values())
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        
        return variance
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        return (
            cache_key in self.metrics_cache and
            cache_key in self.cache_expiry and
            datetime.now() < self.cache_expiry[cache_key]
        )
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache a result with expiry time"""
        self.metrics_cache[cache_key] = result
        self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
    
    def _clear_cache(self):
        """Clear all cached results"""
        self.metrics_cache.clear()
        self.cache_expiry.clear()
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        metrics = {
            'error_metrics': self.get_error_metrics(24),
            'system_health': self.get_system_health_metrics(),
            'error_patterns': self.get_error_patterns(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        if format.lower() == "json":
            return json.dumps(metrics, default=str, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global metrics collector instance
metrics_collector = ErrorMetricsCollector()


# Convenience functions
def record_error(error_info: ErrorInfo):
    """Record an error for metrics collection"""
    metrics_collector.record_error(error_info)


def record_error_resolution(error_id: str, resolution_time: float, success: bool):
    """Record error resolution for metrics"""
    metrics_collector.record_error_resolution(error_id, resolution_time, success)


def get_error_metrics(time_window_hours: int = 24) -> ErrorMetrics:
    """Get error metrics for specified time window"""
    return metrics_collector.get_error_metrics(time_window_hours)


def get_system_health_metrics() -> SystemHealthMetrics:
    """Get system health metrics"""
    return metrics_collector.get_system_health_metrics()


def get_error_patterns() -> Dict[str, Any]:
    """Get error pattern analysis"""
    return metrics_collector.get_error_patterns()


def export_metrics(format: str = "json") -> str:
    """Export all metrics"""
    return metrics_collector.export_metrics(format)


if __name__ == "__main__":
    # Example usage
    print("Error Monitoring and Metrics")
    print("=" * 40)
    
    # Get current metrics
    health_metrics = get_system_health_metrics()
    print(f"Overall health score: {health_metrics.overall_health_score:.1f}")
    print(f"Error rate (24h): {health_metrics.error_rate_24h:.2f} errors/hour")
    print(f"System availability: {health_metrics.system_availability:.2f}%")
    print(f"Error trend: {health_metrics.error_trend}")
    
    # Get error patterns
    patterns = get_error_patterns()
    if patterns['anomalies']:
        print(f"\nAnomalies detected: {len(patterns['anomalies'])}")
        for anomaly in patterns['anomalies']:
            print(f"  - {anomaly['type']}: {anomaly['description']}")
    
    if patterns['recommendations']:
        print(f"\nRecommendations:")
        for rec in patterns['recommendations']:
            print(f"  - {rec}")