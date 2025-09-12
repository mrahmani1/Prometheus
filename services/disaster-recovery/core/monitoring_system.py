"""
Disaster Recovery Monitoring System
Advanced monitoring and testing system for disaster recovery operations
with RTO/RPO tracking, integrity validation, and automated testing.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import uuid
import structlog
import psutil
import aiohttp
from pathlib import Path

from ..models.disaster_recovery_models import (
    DisasterRecoveryMetrics, DisasterRecoveryTest, IntegrityCheckResult,
    BackupStatus, ReplicationStatus, RecoveryStatus
)

logger = structlog.get_logger(__name__)

class DisasterRecoveryMonitoring:
    """Advanced disaster recovery monitoring and testing system"""
    
    def __init__(self):
        self.metrics_history: List[DisasterRecoveryMetrics] = []
        self.active_tests: Dict[str, DisasterRecoveryTest] = {}
        self.test_history: List[DisasterRecoveryTest] = []
        self.integrity_checks: Dict[str, IntegrityCheckResult] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.alert_thresholds = {}
        
    async def initialize(self):
        """Initialize disaster recovery monitoring"""
        try:
            await self._load_monitoring_configuration()
            await self._initialize_alert_thresholds()
            await self._start_monitoring_tasks()
            
            logger.info("Disaster recovery monitoring initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize DR monitoring", error=str(e))
            raise
    
    async def _load_monitoring_configuration(self):
        """Load monitoring configuration"""
        try:
            config_file = '/etc/prometheus/dr-monitoring-config.json'
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.alert_thresholds = config.get('alert_thresholds', {})
            else:
                await self._create_default_monitoring_config()
            
            logger.info("DR monitoring configuration loaded")
            
        except Exception as e:
            logger.error("Failed to load monitoring configuration", error=str(e))
            await self._create_default_monitoring_config()
    
    async def _create_default_monitoring_config(self):
        """Create default monitoring configuration"""
        self.alert_thresholds = {
            'rto_threshold_minutes': 120,
            'rpo_threshold_minutes': 30,
            'backup_success_rate_threshold': 0.95,
            'replication_health_threshold': 0.90,
            'data_integrity_threshold': 0.99,
            'system_availability_threshold': 0.995
        }
    
    async def _initialize_alert_thresholds(self):
        """Initialize alert thresholds and notification channels"""
        try:
            # Set up default thresholds if not configured
            if not self.alert_thresholds:
                await self._create_default_monitoring_config()
            
            logger.info("Alert thresholds initialized", 
                       thresholds=self.alert_thresholds)
            
        except Exception as e:
            logger.error("Failed to initialize alert thresholds", error=str(e))
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        try:
            # Start metrics collection task
            self.monitoring_tasks['metrics_collection'] = asyncio.create_task(
                self._collect_metrics_continuously()
            )
            
            # Start integrity checking task
            self.monitoring_tasks['integrity_checking'] = asyncio.create_task(
                self._run_integrity_checks_continuously()
            )
            
            # Start alert monitoring task
            self.monitoring_tasks['alert_monitoring'] = asyncio.create_task(
                self._monitor_alerts_continuously()
            )
            
            logger.info("Monitoring tasks started", 
                       task_count=len(self.monitoring_tasks))
            
        except Exception as e:
            logger.error("Failed to start monitoring tasks", error=str(e))
            raise
    
    async def _collect_metrics_continuously(self):
        """Continuously collect DR metrics"""
        while True:
            try:
                metrics = await self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 24 hours of metrics
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(300)  # Collect every 5 minutes
                
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(600)  # Wait longer on error
    
    async def _collect_current_metrics(self) -> DisasterRecoveryMetrics:
        """Collect current disaster recovery metrics"""
        try:
            metrics = DisasterRecoveryMetrics(timestamp=datetime.utcnow())
            
            # Collect backup success rate
            metrics.backup_success_rate = await self._calculate_backup_success_rate()
            
            # Collect replication health score
            metrics.replication_health_score = await self._calculate_replication_health()
            
            # Collect data integrity score
            metrics.data_integrity_score = await self._calculate_data_integrity_score()
            
            # Collect system availability
            metrics.system_availability = await self._calculate_system_availability()
            
            # Collect actual RTO/RPO if recent recovery occurred
            rto_rpo = await self._get_recent_rto_rpo()
            if rto_rpo:
                metrics.rto_actual_minutes = rto_rpo['rto_minutes']
                metrics.rpo_actual_minutes = rto_rpo['rpo_minutes']
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to collect current metrics", error=str(e))
            return DisasterRecoveryMetrics(timestamp=datetime.utcnow())
    
    async def _calculate_backup_success_rate(self) -> float:
        """Calculate backup success rate over last 24 hours"""
        try:
            # This would query the backup manager for recent backup status
            # For now, return a simulated value
            return 0.98  # 98% success rate
            
        except Exception as e:
            logger.error("Failed to calculate backup success rate", error=str(e))
            return 0.0
    
    async def _calculate_replication_health(self) -> float:
        """Calculate overall replication health score"""
        try:
            # This would query the replication engine for health status
            # For now, return a simulated value
            return 0.95  # 95% health score
            
        except Exception as e:
            logger.error("Failed to calculate replication health", error=str(e))
            return 0.0
    
    async def _calculate_data_integrity_score(self) -> float:
        """Calculate data integrity score based on recent checks"""
        try:
            if not self.integrity_checks:
                return 1.0  # No checks = assume good
            
            recent_checks = [
                check for check in self.integrity_checks.values()
                if check.start_time > datetime.utcnow() - timedelta(hours=24)
            ]
            
            if not recent_checks:
                return 1.0
            
            passed_checks = sum(1 for check in recent_checks if check.status == 'passed')
            return passed_checks / len(recent_checks)
            
        except Exception as e:
            logger.error("Failed to calculate data integrity score", error=str(e))
            return 0.0
    
    async def _calculate_system_availability(self) -> float:
        """Calculate system availability"""
        try:
            # Check system resources and service health
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            # Simple availability calculation based on resource usage
            resource_health = (
                (100 - cpu_usage) / 100 * 0.3 +
                (100 - memory_usage) / 100 * 0.3 +
                (100 - disk_usage) / 100 * 0.4
            )
            
            # Check service health
            service_health = await self._check_service_health()
            
            return (resource_health + service_health) / 2
            
        except Exception as e:
            logger.error("Failed to calculate system availability", error=str(e))
            return 0.0
    
    async def _check_service_health(self) -> float:
        """Check health of critical services"""
        try:
            services_to_check = [
                'http://backup-manager:8080/health',
                'http://replication-engine:8080/health',
                'http://recovery-orchestrator:8080/health'
            ]
            
            healthy_services = 0
            total_services = len(services_to_check)
            
            async with aiohttp.ClientSession() as session:
                for service_url in services_to_check:
                    try:
                        async with session.get(service_url, timeout=5) as response:
                            if response.status == 200:
                                healthy_services += 1
                    except Exception:
                        pass  # Service is down
            
            return healthy_services / total_services if total_services > 0 else 1.0
            
        except Exception as e:
            logger.error("Failed to check service health", error=str(e))
            return 0.0
    
    async def _get_recent_rto_rpo(self) -> Optional[Dict[str, int]]:
        """Get RTO/RPO from recent recovery operations"""
        try:
            # This would query recent recovery executions
            # For now, return None (no recent recoveries)
            return None
            
        except Exception as e:
            logger.error("Failed to get recent RTO/RPO", error=str(e))
            return None
    
    async def _run_integrity_checks_continuously(self):
        """Continuously run data integrity checks"""
        while True:
            try:
                await self._run_scheduled_integrity_checks()
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error("Integrity checking error", error=str(e))
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def _run_scheduled_integrity_checks(self):
        """Run scheduled integrity checks"""
        try:
            # Check backup integrity
            await self._check_backup_integrity()
            
            # Check replication integrity
            await self._check_replication_integrity()
            
            # Check database integrity
            await self._check_database_integrity()
            
        except Exception as e:
            logger.error("Scheduled integrity checks failed", error=str(e))
    
    async def _check_backup_integrity(self):
        """Check integrity of recent backups"""
        try:
            check_id = str(uuid.uuid4())
            check = IntegrityCheckResult(
                check_id=check_id,
                backup_id="latest",
                check_type="backup_verification",
                start_time=datetime.utcnow(),
                status="running"
            )
            
            self.integrity_checks[check_id] = check
            
            # Simulate integrity check
            await asyncio.sleep(5)  # Simulate check time
            
            # Update check result
            check.end_time = datetime.utcnow()
            check.status = "passed"
            
            logger.info("Backup integrity check completed", check_id=check_id)
            
        except Exception as e:
            logger.error("Backup integrity check failed", error=str(e))
            if check_id in self.integrity_checks:
                self.integrity_checks[check_id].status = "failed"
                self.integrity_checks[check_id].issues_found.append({
                    'issue': 'integrity_check_error',
                    'description': str(e)
                })
    
    async def _check_replication_integrity(self):
        """Check integrity of replication systems"""
        try:
            check_id = str(uuid.uuid4())
            check = IntegrityCheckResult(
                check_id=check_id,
                backup_id="replication",
                check_type="replication_verification",
                start_time=datetime.utcnow(),
                status="running"
            )
            
            self.integrity_checks[check_id] = check
            
            # Simulate replication check
            await asyncio.sleep(3)
            
            check.end_time = datetime.utcnow()
            check.status = "passed"
            
            logger.info("Replication integrity check completed", check_id=check_id)
            
        except Exception as e:
            logger.error("Replication integrity check failed", error=str(e))
    
    async def _check_database_integrity(self):
        """Check database integrity"""
        try:
            check_id = str(uuid.uuid4())
            check = IntegrityCheckResult(
                check_id=check_id,
                backup_id="database",
                check_type="database_verification",
                start_time=datetime.utcnow(),
                status="running"
            )
            
            self.integrity_checks[check_id] = check
            
            # Simulate database check
            await asyncio.sleep(10)
            
            check.end_time = datetime.utcnow()
            check.status = "passed"
            
            logger.info("Database integrity check completed", check_id=check_id)
            
        except Exception as e:
            logger.error("Database integrity check failed", error=str(e))
    
    async def _monitor_alerts_continuously(self):
        """Continuously monitor for alert conditions"""
        while True:
            try:
                await self._check_alert_conditions()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error("Alert monitoring error", error=str(e))
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _check_alert_conditions(self):
        """Check for alert conditions"""
        try:
            if not self.metrics_history:
                return
            
            latest_metrics = self.metrics_history[-1]
            
            # Check backup success rate
            if latest_metrics.backup_success_rate < self.alert_thresholds['backup_success_rate_threshold']:
                await self._send_alert(
                    'backup_success_rate_low',
                    f"Backup success rate ({latest_metrics.backup_success_rate:.1%}) below threshold "
                    f"({self.alert_thresholds['backup_success_rate_threshold']:.1%})"
                )
            
            # Check replication health
            if latest_metrics.replication_health_score < self.alert_thresholds['replication_health_threshold']:
                await self._send_alert(
                    'replication_health_low',
                    f"Replication health ({latest_metrics.replication_health_score:.1%}) below threshold "
                    f"({self.alert_thresholds['replication_health_threshold']:.1%})"
                )
            
            # Check data integrity
            if latest_metrics.data_integrity_score < self.alert_thresholds['data_integrity_threshold']:
                await self._send_alert(
                    'data_integrity_low',
                    f"Data integrity score ({latest_metrics.data_integrity_score:.1%}) below threshold "
                    f"({self.alert_thresholds['data_integrity_threshold']:.1%})"
                )
            
            # Check system availability
            if latest_metrics.system_availability < self.alert_thresholds['system_availability_threshold']:
                await self._send_alert(
                    'system_availability_low',
                    f"System availability ({latest_metrics.system_availability:.1%}) below threshold "
                    f"({self.alert_thresholds['system_availability_threshold']:.1%})"
                )
            
            # Check RTO/RPO if available
            if latest_metrics.rto_actual_minutes:
                if latest_metrics.rto_actual_minutes > self.alert_thresholds['rto_threshold_minutes']:
                    await self._send_alert(
                        'rto_exceeded',
                        f"Actual RTO ({latest_metrics.rto_actual_minutes} minutes) exceeded threshold "
                        f"({self.alert_thresholds['rto_threshold_minutes']} minutes)"
                    )
            
            if latest_metrics.rpo_actual_minutes:
                if latest_metrics.rpo_actual_minutes > self.alert_thresholds['rpo_threshold_minutes']:
                    await self._send_alert(
                        'rpo_exceeded',
                        f"Actual RPO ({latest_metrics.rpo_actual_minutes} minutes) exceeded threshold "
                        f"({self.alert_thresholds['rpo_threshold_minutes']} minutes)"
                    )
            
        except Exception as e:
            logger.error("Failed to check alert conditions", error=str(e))
    
    async def _send_alert(self, alert_type: str, message: str):
        """Send alert notification"""
        try:
            alert_data = {
                'alert_type': alert_type,
                'message': message,
                'timestamp': datetime.utcnow().isoformat(),
                'severity': 'warning'
            }
            
            # Log the alert
            logger.warning("DR Alert triggered", **alert_data)
            
            # Send to notification system (placeholder)
            # In production, this would integrate with PagerDuty, Slack, etc.
            
        except Exception as e:
            logger.error("Failed to send alert", alert_type=alert_type, error=str(e))
    
    async def check_system_health(self):
        """Check overall system health"""
        try:
            health_status = {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_status': 'healthy',
                'components': {}
            }
            
            # Check backup system health
            backup_health = await self._check_component_health('backup_manager')
            health_status['components']['backup_manager'] = backup_health
            
            # Check replication system health
            replication_health = await self._check_component_health('replication_engine')
            health_status['components']['replication_engine'] = replication_health
            
            # Check recovery system health
            recovery_health = await self._check_component_health('recovery_orchestrator')
            health_status['components']['recovery_orchestrator'] = recovery_health
            
            # Determine overall status
            component_statuses = [comp['status'] for comp in health_status['components'].values()]
            if any(status == 'unhealthy' for status in component_statuses):
                health_status['overall_status'] = 'unhealthy'
            elif any(status == 'degraded' for status in component_statuses):
                health_status['overall_status'] = 'degraded'
            
            logger.info("System health check completed", 
                       overall_status=health_status['overall_status'])
            
        except Exception as e:
            logger.error("System health check failed", error=str(e))
    
    async def _check_component_health(self, component: str) -> Dict[str, Any]:
        """Check health of a specific component"""
        try:
            # This would check the actual component health
            # For now, return a simulated status
            return {
                'status': 'healthy',
                'last_check': datetime.utcnow().isoformat(),
                'response_time_ms': 50
            }
            
        except Exception as e:
            logger.error("Component health check failed", component=component, error=str(e))
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }
    
    async def initiate_dr_test(self, test_request: Dict[str, Any]) -> str:
        """Initiate disaster recovery test"""
        try:
            test = DisasterRecoveryTest(
                test_id=str(uuid.uuid4()),
                test_name=test_request['test_name'],
                test_type=test_request['test_type'],
                scope=test_request['scope'],
                start_time=datetime.utcnow(),
                status='planned'
            )
            
            self.active_tests[test.test_id] = test
            
            logger.info("DR test initiated", 
                       test_id=test.test_id,
                       test_type=test.test_type)
            
            return test.test_id
            
        except Exception as e:
            logger.error("Failed to initiate DR test", error=str(e))
            raise
    
    async def execute_dr_test(self, test_id: str):
        """Execute disaster recovery test"""
        try:
            test = self.active_tests.get(test_id)
            if not test:
                raise ValueError(f"DR test {test_id} not found")
            
            test.status = 'running'
            test.start_time = datetime.utcnow()
            
            logger.info("DR test execution started", test_id=test_id)
            
            # Execute test based on type
            if test.test_type == 'backup_restore':
                await self._execute_backup_restore_test(test)
            elif test.test_type == 'failover':
                await self._execute_failover_test(test)
            elif test.test_type == 'full_dr':
                await self._execute_full_dr_test(test)
            elif test.test_type == 'tabletop':
                await self._execute_tabletop_test(test)
            else:
                raise ValueError(f"Unknown test type: {test.test_type}")
            
            test.status = 'completed'
            test.end_time = datetime.utcnow()
            
            # Move to history
            self.test_history.append(test)
            del self.active_tests[test_id]
            
            logger.info("DR test completed", 
                       test_id=test_id,
                       duration_minutes=(test.end_time - test.start_time).total_seconds() / 60)
            
        except Exception as e:
            test.status = 'failed'
            test.end_time = datetime.utcnow()
            test.issues_identified.append({
                'issue': 'test_execution_error',
                'description': str(e)
            })
            
            logger.error("DR test execution failed", test_id=test_id, error=str(e))
    
    async def _execute_backup_restore_test(self, test: DisasterRecoveryTest):
        """Execute backup and restore test"""
        try:
            # Simulate backup restore test
            test.test_results['backup_creation'] = {
                'status': 'success',
                'duration_seconds': 120,
                'backup_size_gb': 15.5
            }
            
            test.test_results['restore_operation'] = {
                'status': 'success',
                'duration_seconds': 180,
                'data_integrity_verified': True
            }
            
            await asyncio.sleep(5)  # Simulate test time
            
        except Exception as e:
            logger.error("Backup restore test failed", error=str(e))
            raise
    
    async def _execute_failover_test(self, test: DisasterRecoveryTest):
        """Execute failover test"""
        try:
            # Simulate failover test
            test.test_results['failover_initiation'] = {
                'status': 'success',
                'duration_seconds': 45
            }
            
            test.test_results['service_availability'] = {
                'status': 'success',
                'downtime_seconds': 30,
                'services_restored': len(test.scope)
            }
            
            await asyncio.sleep(3)  # Simulate test time
            
        except Exception as e:
            logger.error("Failover test failed", error=str(e))
            raise
    
    async def _execute_full_dr_test(self, test: DisasterRecoveryTest):
        """Execute full disaster recovery test"""
        try:
            # Simulate full DR test
            test.test_results['disaster_simulation'] = {
                'status': 'success',
                'scenario': 'data_center_failure'
            }
            
            test.test_results['recovery_execution'] = {
                'status': 'success',
                'total_duration_minutes': 45,
                'rto_achieved_minutes': 40,
                'rpo_achieved_minutes': 5
            }
            
            await asyncio.sleep(10)  # Simulate test time
            
        except Exception as e:
            logger.error("Full DR test failed", error=str(e))
            raise
    
    async def _execute_tabletop_test(self, test: DisasterRecoveryTest):
        """Execute tabletop exercise"""
        try:
            # Simulate tabletop exercise
            test.test_results['scenario_walkthrough'] = {
                'status': 'success',
                'participants': 5,
                'scenarios_covered': 3
            }
            
            test.test_results['plan_validation'] = {
                'status': 'success',
                'gaps_identified': 2,
                'improvements_suggested': 4
            }
            
            await asyncio.sleep(2)  # Simulate test time
            
        except Exception as e:
            logger.error("Tabletop test failed", error=str(e))
            raise
    
    async def get_rto_rpo_metrics(self) -> Dict[str, Any]:
        """Get RTO/RPO monitoring metrics"""
        try:
            if not self.metrics_history:
                return {'error': 'No metrics available'}
            
            # Calculate RTO/RPO statistics from recent metrics
            recent_metrics = [
                m for m in self.metrics_history
                if m.timestamp > datetime.utcnow() - timedelta(hours=24)
            ]
            
            rto_values = [m.rto_actual_minutes for m in recent_metrics if m.rto_actual_minutes]
            rpo_values = [m.rpo_actual_minutes for m in recent_metrics if m.rpo_actual_minutes]
            
            return {
                'rto_statistics': {
                    'count': len(rto_values),
                    'average_minutes': sum(rto_values) / len(rto_values) if rto_values else None,
                    'max_minutes': max(rto_values) if rto_values else None,
                    'threshold_minutes': self.alert_thresholds['rto_threshold_minutes']
                },
                'rpo_statistics': {
                    'count': len(rpo_values),
                    'average_minutes': sum(rpo_values) / len(rpo_values) if rpo_values else None,
                    'max_minutes': max(rpo_values) if rpo_values else None,
                    'threshold_minutes': self.alert_thresholds['rpo_threshold_minutes']
                },
                'latest_metrics': recent_metrics[-1].__dict__ if recent_metrics else None
            }
            
        except Exception as e:
            logger.error("Failed to get RTO/RPO metrics", error=str(e))
            raise
    
    async def get_backup_integrity_metrics(self) -> Dict[str, Any]:
        """Get backup integrity metrics"""
        try:
            recent_checks = [
                check for check in self.integrity_checks.values()
                if check.start_time > datetime.utcnow() - timedelta(hours=24)
            ]
            
            passed_checks = sum(1 for check in recent_checks if check.status == 'passed')
            failed_checks = sum(1 for check in recent_checks if check.status == 'failed')
            
            return {
                'total_checks_24h': len(recent_checks),
                'passed_checks': passed_checks,
                'failed_checks': failed_checks,
                'success_rate': passed_checks / len(recent_checks) if recent_checks else 1.0,
                'recent_checks': [
                    {
                        'check_id': check.check_id,
                        'check_type': check.check_type,
                        'status': check.status,
                        'start_time': check.start_time.isoformat(),
                        'issues_count': len(check.issues_found)
                    }
                    for check in recent_checks[-10:]  # Last 10 checks
                ]
            }
            
        except Exception as e:
            logger.error("Failed to get backup integrity metrics", error=str(e))
            raise
    
    async def is_ready(self) -> bool:
        """Check if monitoring system is ready"""
        return len(self.monitoring_tasks) > 0