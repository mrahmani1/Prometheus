"""
Data Source Health Monitor

Provides comprehensive health monitoring and failover mechanisms for all
real data source connectors with automated recovery and alerting.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import logging
from enum import Enum
from dataclasses import dataclass, field
import json

from .base_connector import BaseDataConnector, ConnectionStatus

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetrics:
    """Health metrics for a data source"""
    status: HealthStatus
    response_time_ms: float
    success_rate: float
    error_count: int
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 0.0
    throughput_per_minute: float = 0.0
    
    
@dataclass
class FailoverConfig:
    """Failover configuration for data sources"""
    primary_source: str
    fallback_sources: List[str] = field(default_factory=list)
    max_failures: int = 3
    failure_window_minutes: int = 15
    recovery_check_interval: int = 300  # 5 minutes
    auto_failback: bool = True


class DataSourceHealthMonitor:
    """Comprehensive health monitoring for real data sources"""
    
    def __init__(self):
        self.connectors: Dict[str, BaseDataConnector] = {}
        self.health_metrics: Dict[str, HealthMetrics] = {}
        self.failover_configs: Dict[str, FailoverConfig] = {}
        self.active_sources: Dict[str, str] = {}  # service -> active_source_id
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring configuration
        self.check_interval = 60  # seconds
        self.health_history_hours = 24
        self.alert_thresholds = {
            'response_time_ms': 5000,
            'success_rate': 0.95,
            'consecutive_failures': 3
        }
    
    def register_connector(self, source_id: str, connector: BaseDataConnector, 
                          failover_config: Optional[FailoverConfig] = None):
        """Register a data source connector for monitoring"""
        self.connectors[source_id] = connector
        self.health_metrics[source_id] = HealthMetrics(
            status=HealthStatus.UNKNOWN,
            response_time_ms=0.0,
            success_rate=1.0,
            error_count=0
        )
        
        if failover_config:
            self.failover_configs[source_id] = failover_config
            self.active_sources[failover_config.primary_source] = source_id
        
        # Start monitoring task
        task = asyncio.create_task(self._monitor_source(source_id))
        self.monitoring_tasks[source_id] = task
        
        logger.info(f"Registered data source {source_id} for health monitoring")
    
    def add_alert_callback(self, callback: Callable[[str, HealthStatus, Dict[str, Any]], None]):
        """Add callback function for health alerts"""
        self.alert_callbacks.append(callback)
    
    async def _monitor_source(self, source_id: str):
        """Monitor health of a specific data source"""
        connector = self.connectors[source_id]
        metrics = self.health_metrics[source_id]
        
        while True:
            try:
                # Perform health check
                start_time = datetime.now()
                health_result = await connector.health_check()
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Update metrics
                if health_result.get('status') == 'healthy':
                    await self._record_success(source_id, response_time)
                else:
                    await self._record_failure(source_id, health_result.get('error', 'Unknown error'))
                
                # Check for failover conditions
                await self._check_failover_conditions(source_id)
                
                # Send alerts if necessary
                await self._check_alert_conditions(source_id)
                
            except Exception as e:
                logger.error(f"Error monitoring source {source_id}: {e}")
                await self._record_failure(source_id, str(e))
            
            await asyncio.sleep(self.check_interval)
    
    async def _record_success(self, source_id: str, response_time_ms: float):
        """Record successful health check"""
        metrics = self.health_metrics[source_id]
        
        metrics.last_success = datetime.now()
        metrics.consecutive_failures = 0
        metrics.response_time_ms = response_time_ms
        
        # Update success rate (exponential moving average)
        metrics.success_rate = 0.9 * metrics.success_rate + 0.1 * 1.0
        
        # Determine health status
        if response_time_ms > self.alert_thresholds['response_time_ms']:
            metrics.status = HealthStatus.DEGRADED
        else:
            metrics.status = HealthStatus.HEALTHY
        
        logger.debug(f"Source {source_id} health check successful: {response_time_ms:.1f}ms")
    
    async def _record_failure(self, source_id: str, error_message: str):
        """Record failed health check"""
        metrics = self.health_metrics[source_id]
        
        metrics.last_failure = datetime.now()
        metrics.consecutive_failures += 1
        metrics.error_count += 1
        
        # Update success rate
        metrics.success_rate = 0.9 * metrics.success_rate + 0.1 * 0.0
        
        # Determine health status based on consecutive failures
        if metrics.consecutive_failures >= 5:
            metrics.status = HealthStatus.CRITICAL
        elif metrics.consecutive_failures >= 3:
            metrics.status = HealthStatus.UNHEALTHY
        else:
            metrics.status = HealthStatus.DEGRADED
        
        logger.warning(f"Source {source_id} health check failed: {error_message}")
    
    async def _check_failover_conditions(self, source_id: str):
        """Check if failover should be triggered"""
        metrics = self.health_metrics[source_id]
        
        # Find failover config for this source
        failover_config = None
        for config in self.failover_configs.values():
            if config.primary_source == source_id or source_id in config.fallback_sources:
                failover_config = config
                break
        
        if not failover_config:
            return
        
        # Check if primary source needs failover
        if (source_id == failover_config.primary_source and 
            metrics.consecutive_failures >= failover_config.max_failures):
            
            await self._trigger_failover(failover_config)
        
        # Check if we can fail back to primary
        elif (failover_config.auto_failback and 
              source_id == failover_config.primary_source and
              metrics.status == HealthStatus.HEALTHY and
              self.active_sources.get(failover_config.primary_source) != source_id):
            
            await self._trigger_failback(failover_config)
    
    async def _trigger_failover(self, config: FailoverConfig):
        """Trigger failover to backup data source"""
        primary_id = config.primary_source
        
        # Find healthy fallback source
        for fallback_id in config.fallback_sources:
            if fallback_id in self.health_metrics:
                fallback_metrics = self.health_metrics[fallback_id]
                if fallback_metrics.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                    # Switch to fallback source
                    self.active_sources[primary_id] = fallback_id
                    
                    logger.warning(f"Failover triggered: {primary_id} -> {fallback_id}")
                    
                    # Send alert
                    await self._send_alert(
                        primary_id, 
                        HealthStatus.CRITICAL,
                        {
                            'event': 'failover_triggered',
                            'primary_source': primary_id,
                            'fallback_source': fallback_id,
                            'reason': 'consecutive_failures'
                        }
                    )
                    return
        
        # No healthy fallback found
        logger.error(f"No healthy fallback source available for {primary_id}")
        await self._send_alert(
            primary_id,
            HealthStatus.CRITICAL,
            {
                'event': 'failover_failed',
                'primary_source': primary_id,
                'reason': 'no_healthy_fallback'
            }
        )
    
    async def _trigger_failback(self, config: FailoverConfig):
        """Trigger failback to primary data source"""
        primary_id = config.primary_source
        current_active = self.active_sources.get(primary_id)
        
        if current_active != primary_id:
            self.active_sources[primary_id] = primary_id
            
            logger.info(f"Failback triggered: {current_active} -> {primary_id}")
            
            await self._send_alert(
                primary_id,
                HealthStatus.HEALTHY,
                {
                    'event': 'failback_triggered',
                    'primary_source': primary_id,
                    'previous_active': current_active,
                    'reason': 'primary_recovered'
                }
            )
    
    async def _check_alert_conditions(self, source_id: str):
        """Check if alerts should be sent"""
        metrics = self.health_metrics[source_id]
        
        # Alert on status changes
        if hasattr(metrics, '_previous_status'):
            if metrics.status != metrics._previous_status:
                await self._send_alert(source_id, metrics.status, {
                    'event': 'status_change',
                    'previous_status': metrics._previous_status.value,
                    'current_status': metrics.status.value
                })
        
        metrics._previous_status = metrics.status
        
        # Alert on threshold breaches
        if (metrics.response_time_ms > self.alert_thresholds['response_time_ms'] and
            metrics.status == HealthStatus.HEALTHY):
            
            await self._send_alert(source_id, HealthStatus.DEGRADED, {
                'event': 'high_response_time',
                'response_time_ms': metrics.response_time_ms,
                'threshold_ms': self.alert_thresholds['response_time_ms']
            })
        
        if metrics.success_rate < self.alert_thresholds['success_rate']:
            await self._send_alert(source_id, HealthStatus.DEGRADED, {
                'event': 'low_success_rate',
                'success_rate': metrics.success_rate,
                'threshold': self.alert_thresholds['success_rate']
            })
    
    async def _send_alert(self, source_id: str, status: HealthStatus, details: Dict[str, Any]):
        """Send health alert to registered callbacks"""
        alert_data = {
            'source_id': source_id,
            'status': status.value,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        
        for callback in self.alert_callbacks:
            try:
                await callback(source_id, status, alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_active_connector(self, service_name: str) -> Optional[BaseDataConnector]:
        """Get the currently active connector for a service"""
        active_source_id = self.active_sources.get(service_name)
        if active_source_id:
            return self.connectors.get(active_source_id)
        return None
    
    def get_health_status(self, source_id: str) -> Optional[HealthMetrics]:
        """Get current health status for a data source"""
        return self.health_metrics.get(source_id)
    
    def get_all_health_status(self) -> Dict[str, HealthMetrics]:
        """Get health status for all monitored sources"""
        return self.health_metrics.copy()
    
    async def force_failover(self, service_name: str, target_source_id: str):
        """Manually trigger failover to specific source"""
        if target_source_id in self.connectors:
            self.active_sources[service_name] = target_source_id
            logger.info(f"Manual failover: {service_name} -> {target_source_id}")
            
            await self._send_alert(service_name, HealthStatus.UNKNOWN, {
                'event': 'manual_failover',
                'target_source': target_source_id,
                'reason': 'administrator_action'
            })
        else:
            raise ValueError(f"Unknown source ID: {target_source_id}")
    
    async def shutdown(self):
        """Shutdown health monitoring"""
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        
        logger.info("Health monitoring shutdown complete")
    
    def export_health_report(self) -> Dict[str, Any]:
        """Export comprehensive health report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'active_sources': self.active_sources.copy(),
            'summary': {
                'total_sources': len(self.connectors),
                'healthy_sources': 0,
                'degraded_sources': 0,
                'unhealthy_sources': 0,
                'critical_sources': 0
            }
        }
        
        for source_id, metrics in self.health_metrics.items():
            report['sources'][source_id] = {
                'status': metrics.status.value,
                'response_time_ms': metrics.response_time_ms,
                'success_rate': metrics.success_rate,
                'error_count': metrics.error_count,
                'consecutive_failures': metrics.consecutive_failures,
                'last_success': metrics.last_success.isoformat() if metrics.last_success else None,
                'last_failure': metrics.last_failure.isoformat() if metrics.last_failure else None,
                'uptime_percentage': metrics.uptime_percentage
            }
            
            # Update summary counts
            if metrics.status == HealthStatus.HEALTHY:
                report['summary']['healthy_sources'] += 1
            elif metrics.status == HealthStatus.DEGRADED:
                report['summary']['degraded_sources'] += 1
            elif metrics.status == HealthStatus.UNHEALTHY:
                report['summary']['unhealthy_sources'] += 1
            elif metrics.status == HealthStatus.CRITICAL:
                report['summary']['critical_sources'] += 1
        
        return report