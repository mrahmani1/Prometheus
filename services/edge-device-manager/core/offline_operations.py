"""
Main offline operations coordinator that integrates all offline capabilities
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import pickle

from .offline_processor import OfflineDataProcessor, OfflineAnalysisResult
from .offline_alerting import OfflineAlertGenerator, Alert, AlertSeverity
from .local_storage import LocalStorageManager, StorageType
from .connectivity_manager import ConnectivityManager, ConnectivityStatus
from models.config import EdgeDeviceConfig
from models.device import DeviceMetrics

logger = logging.getLogger(__name__)


@dataclass
class OfflineOperationStatus:
    """Status of offline operations"""
    is_offline: bool
    offline_since: Optional[datetime]
    connectivity_score: float
    data_processing_active: bool
    alert_generation_active: bool
    storage_usage_percent: float
    pending_sync_count: int
    active_alerts_count: int
    last_analysis_time: Optional[datetime]


class OfflineOperationsCoordinator:
    """Coordinates all offline operation capabilities"""
    
    def __init__(self, config: EdgeDeviceConfig, storage_path: str = "./offline_operations"):
        self.config = config
        self.storage_path = storage_path
        
        # Initialize core components
        self.storage_manager = LocalStorageManager(
            storage_root=f"{storage_path}/storage",
            max_storage_gb=config.max_storage_usage if hasattr(config, 'max_storage_usage') else 5.0
        )
        
        self.offline_processor = OfflineDataProcessor(
            config=config,
            storage_path=f"{storage_path}/processor"
        )
        
        self.alert_generator = OfflineAlertGenerator(
            config_path=f"{storage_path}/alert_config.json"
        )
        
        self.connectivity_manager = ConnectivityManager(
            config=config,
            storage_manager=self.storage_manager
        )
        
        # Operation state
        self.is_running = False
        self.offline_mode = True
        self.offline_since: Optional[datetime] = None
        
        # Background tasks
        self._processing_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._maintenance_task: Optional[asyncio.Task] = None
        
        # Event callbacks
        self.status_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Statistics
        self.operation_stats = {
            "total_data_processed": 0,
            "total_alerts_generated": 0,
            "offline_time_hours": 0.0,
            "data_processing_rate": 0.0,
            "storage_efficiency": 0.0,
            "last_maintenance": None
        }
        
        self._setup_component_callbacks()
    
    def _setup_component_callbacks(self):
        """Setup callbacks between components"""
        # Connectivity callbacks
        self.connectivity_manager.register_connectivity_callback(self._on_connectivity_change)
        
        # Alert callbacks
        self.alert_generator.register_callback("new_alert", self._on_new_alert)
        self.alert_generator.register_callback("alert_resolved", self._on_alert_resolved)
    
    async def start(self):
        """Start offline operations"""
        try:
            if self.is_running:
                logger.warning("Offline operations already running")
                return
            
            logger.info("Starting offline operations coordinator...")
            
            # Start core components
            await self.connectivity_manager.start()
            
            # Start background tasks
            self._processing_task = asyncio.create_task(self._data_processing_loop())
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
            
            self.is_running = True
            self.offline_since = datetime.utcnow()
            
            logger.info("Offline operations coordinator started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start offline operations: {e}")
            raise
    
    async def stop(self):
        """Stop offline operations"""
        try:
            if not self.is_running:
                return
            
            logger.info("Stopping offline operations coordinator...")
            
            self.is_running = False
            
            # Cancel background tasks
            tasks = [self._processing_task, self._monitoring_task, self._maintenance_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Stop core components
            await self.connectivity_manager.stop()
            await self.storage_manager.shutdown()
            await self.offline_processor.shutdown()
            
            logger.info("Offline operations coordinator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping offline operations: {e}")
    
    async def _data_processing_loop(self):
        """Main data processing loop"""
        while self.is_running:
            try:
                # Process offline data
                analysis_results = await self.offline_processor.process_offline_data()
                
                if analysis_results:
                    # Store analysis results
                    for result in analysis_results:
                        await self._store_analysis_result(result)
                        
                        # Generate alerts based on analysis
                        await self._process_analysis_for_alerts(result)
                    
                    self.operation_stats["total_data_processed"] += len(analysis_results)
                    self.operation_stats["data_processing_rate"] = len(analysis_results) / 60.0  # per minute
                
                # Sleep between processing cycles
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in data processing loop: {e}")
                await asyncio.sleep(30)  # Longer sleep on error
    
    async def _monitoring_loop(self):
        """System monitoring loop"""
        while self.is_running:
            try:
                # Update operation statistics
                await self._update_operation_statistics()
                
                # Check system health
                await self._check_system_health()
                
                # Notify status callbacks
                await self._notify_status_callbacks()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _maintenance_loop(self):
        """Maintenance and cleanup loop"""
        while self.is_running:
            try:
                # Cleanup old data
                await self.storage_manager.cleanup_expired_data()
                await self.offline_processor.cleanup_old_data()
                await self.alert_generator.cleanup_old_alerts()
                
                # Update maintenance timestamp
                self.operation_stats["last_maintenance"] = datetime.utcnow()
                
                # Sleep for 1 hour between maintenance cycles
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(3600)
    
    async def _on_connectivity_change(self, status: ConnectivityStatus, score: float):
        """Handle connectivity status changes"""
        try:
            previous_offline = self.offline_mode
            self.offline_mode = status != ConnectivityStatus.ONLINE
            
            if previous_offline and not self.offline_mode:
                # Coming back online
                logger.info("Transitioning from offline to online mode")
                await self._transition_to_online()
                self.offline_since = None
                
            elif not previous_offline and self.offline_mode:
                # Going offline
                logger.info("Transitioning from online to offline mode")
                await self._transition_to_offline()
                self.offline_since = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error handling connectivity change: {e}")
    
    async def _transition_to_online(self):
        """Handle transition to online mode"""
        try:
            # Force synchronization of pending data
            sync_result = await self.connectivity_manager.force_sync()
            logger.info(f"Online transition sync result: {sync_result}")
            
            # Update alert thresholds from cloud if possible
            await self._update_alert_thresholds_from_cloud()
            
        except Exception as e:
            logger.error(f"Error in online transition: {e}")
    
    async def _transition_to_offline(self):
        """Handle transition to offline mode"""
        try:
            # Cache current models and thresholds
            await self._cache_models_and_thresholds()
            
            # Prepare for extended offline operation
            logger.info("Prepared for offline operation")
            
        except Exception as e:
            logger.error(f"Error in offline transition: {e}")
    
    async def ingest_sensor_data(self, data_type: str, data: Dict[str, Any], location: Optional[Dict[str, float]] = None) -> str:
        """Ingest sensor data for offline processing"""
        try:
            # Serialize data
            data_payload = pickle.dumps(data)
            
            # Store in local storage
            metadata = {
                "data_type": data_type,
                "location": location or {},
                "ingestion_time": datetime.utcnow().isoformat(),
                "device_id": self.config.device_id
            }
            
            record_id = await self.storage_manager.store_data(
                storage_type=StorageType.SENSOR_DATA,
                data=data_payload,
                metadata=metadata,
                retention_days=self.config.data_retention_days
            )
            
            # Queue for offline processing
            await self.offline_processor.store_sensor_data(data_type, data_payload, metadata)
            
            # Immediate alert evaluation for critical data
            if data_type in ["seismic", "gas", "thermal"]:
                alerts = await self.alert_generator.evaluate_data(data_type, data, location)
                if alerts:
                    logger.info(f"Generated {len(alerts)} immediate alerts for {data_type} data")
            
            return record_id
            
        except Exception as e:
            logger.error(f"Error ingesting sensor data: {e}")
            raise
    
    async def _store_analysis_result(self, result: OfflineAnalysisResult):
        """Store analysis result in local storage"""
        try:
            # Serialize result
            result_data = {
                "timestamp": result.timestamp.isoformat(),
                "analysis_type": result.analysis_type,
                "input_data_hash": result.input_data_hash,
                "results": result.results,
                "confidence_score": result.confidence_score,
                "alerts_generated": result.alerts_generated,
                "processing_time_ms": result.processing_time_ms
            }
            
            metadata = {
                "analysis_type": result.analysis_type,
                "confidence_score": result.confidence_score,
                "device_id": self.config.device_id
            }
            
            await self.storage_manager.store_data(
                storage_type=StorageType.ANALYSIS_RESULTS,
                data=result_data,
                metadata=metadata,
                retention_days=30  # Keep analysis results longer
            )
            
        except Exception as e:
            logger.error(f"Error storing analysis result: {e}")
    
    async def _process_analysis_for_alerts(self, result: OfflineAnalysisResult):
        """Process analysis result for alert generation"""
        try:
            # Extract relevant data for alert evaluation
            alert_data = result.results.copy()
            alert_data.update({
                "confidence_score": result.confidence_score,
                "processing_time_ms": result.processing_time_ms
            })
            
            # Generate alerts
            alerts = await self.alert_generator.evaluate_data(
                result.analysis_type,
                alert_data
            )
            
            if alerts:
                self.operation_stats["total_alerts_generated"] += len(alerts)
                
                # Store alerts
                for alert in alerts:
                    await self._store_alert(alert)
            
        except Exception as e:
            logger.error(f"Error processing analysis for alerts: {e}")
    
    async def _store_alert(self, alert: Alert):
        """Store alert in local storage"""
        try:
            alert_data = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity.value,
                "status": alert.status.value,
                "title": alert.title,
                "message": alert.message,
                "data_source": alert.data_source,
                "trigger_value": alert.trigger_value,
                "threshold_value": alert.threshold_value,
                "location": alert.location,
                "metadata": alert.metadata
            }
            
            metadata = {
                "alert_type": alert.data_source,
                "severity": alert.severity.value,
                "device_id": self.config.device_id
            }
            
            await self.storage_manager.store_data(
                storage_type=StorageType.ALERT_HISTORY,
                data=alert_data,
                metadata=metadata,
                retention_days=90  # Keep alerts for 3 months
            )
            
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    async def _on_new_alert(self, alert: Alert):
        """Handle new alert generation"""
        try:
            logger.warning(f"NEW OFFLINE ALERT: {alert.title} - {alert.message}")
            
            # Notify alert callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
        except Exception as e:
            logger.error(f"Error handling new alert: {e}")
    
    async def _on_alert_resolved(self, alert: Alert):
        """Handle alert resolution"""
        try:
            logger.info(f"ALERT RESOLVED: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error handling alert resolution: {e}")
    
    async def _check_system_health(self):
        """Check system health and generate system alerts if needed"""
        try:
            # Get system metrics (placeholder - would integrate with actual system monitoring)
            system_metrics = {
                "cpu_usage_percent": 45.0,  # Would get from actual system
                "memory_usage_percent": 60.0,
                "storage_usage_percent": 70.0,
                "temperature_celsius": 35.0
            }
            
            # Evaluate system health alerts
            alerts = await self.alert_generator.evaluate_data("system", system_metrics)
            
            if alerts:
                for alert in alerts:
                    await self._store_alert(alert)
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
    
    async def _update_operation_statistics(self):
        """Update operation statistics"""
        try:
            # Calculate offline time
            if self.offline_since:
                offline_duration = datetime.utcnow() - self.offline_since
                self.operation_stats["offline_time_hours"] = offline_duration.total_seconds() / 3600
            
            # Get storage statistics
            storage_stats = await self.storage_manager.get_storage_statistics()
            self.operation_stats["storage_efficiency"] = storage_stats.get("compression_ratio", 0.0)
            
            # Get processing statistics
            processing_stats = await self.offline_processor.get_processing_statistics()
            self.operation_stats.update(processing_stats.get("processing_stats", {}))
            
        except Exception as e:
            logger.error(f"Error updating operation statistics: {e}")
    
    async def _notify_status_callbacks(self):
        """Notify status change callbacks"""
        try:
            status = await self.get_operation_status()
            
            for callback in self.status_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(status)
                    else:
                        callback(status)
                except Exception as e:
                    logger.error(f"Error in status callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error notifying status callbacks: {e}")
    
    async def _update_alert_thresholds_from_cloud(self):
        """Update alert thresholds from cloud service when online"""
        try:
            # This would fetch updated thresholds from cloud service
            # For now, just log the attempt
            logger.info("Would update alert thresholds from cloud service")
            
        except Exception as e:
            logger.error(f"Error updating alert thresholds from cloud: {e}")
    
    async def _cache_models_and_thresholds(self):
        """Cache models and thresholds for offline operation"""
        try:
            # This would cache the latest models and thresholds
            # For now, just log the attempt
            logger.info("Would cache models and thresholds for offline operation")
            
        except Exception as e:
            logger.error(f"Error caching models and thresholds: {e}")
    
    def register_status_callback(self, callback: Callable):
        """Register callback for status changes"""
        self.status_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alert events"""
        self.alert_callbacks.append(callback)
    
    async def get_operation_status(self) -> OfflineOperationStatus:
        """Get current offline operation status"""
        try:
            # Get connectivity status
            connectivity_status = await self.connectivity_manager.get_connectivity_status()
            
            # Get storage statistics
            storage_stats = await self.storage_manager.get_storage_statistics()
            
            # Get processing statistics
            processing_stats = await self.offline_processor.get_processing_statistics()
            
            # Get alert statistics
            alert_stats = self.alert_generator.get_statistics()
            
            return OfflineOperationStatus(
                is_offline=self.offline_mode,
                offline_since=self.offline_since,
                connectivity_score=connectivity_status["connectivity_score"],
                data_processing_active=self.is_running,
                alert_generation_active=self.is_running,
                storage_usage_percent=storage_stats["usage_percentage"],
                pending_sync_count=connectivity_status["sync_queue_size"],
                active_alerts_count=alert_stats["active_alerts_count"],
                last_analysis_time=processing_stats.get("processing_stats", {}).get("last_processing_time")
            )
            
        except Exception as e:
            logger.error(f"Error getting operation status: {e}")
            return OfflineOperationStatus(
                is_offline=True,
                offline_since=self.offline_since,
                connectivity_score=0.0,
                data_processing_active=False,
                alert_generation_active=False,
                storage_usage_percent=0.0,
                pending_sync_count=0,
                active_alerts_count=0,
                last_analysis_time=None
            )
    
    async def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts"""
        return self.alert_generator.get_active_alerts(severity)
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        return await self.alert_generator.acknowledge_alert(alert_id)
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        return await self.alert_generator.resolve_alert(alert_id)
    
    async def force_data_sync(self) -> Dict[str, Any]:
        """Force synchronization of pending data"""
        return await self.connectivity_manager.force_sync()
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status"""
        return await self.connectivity_manager.get_sync_status()
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all offline operations"""
        try:
            operation_status = await self.get_operation_status()
            connectivity_status = await self.connectivity_manager.get_connectivity_status()
            sync_status = await self.connectivity_manager.get_sync_status()
            storage_stats = await self.storage_manager.get_storage_statistics()
            processing_stats = await self.offline_processor.get_processing_statistics()
            alert_stats = self.alert_generator.get_statistics()
            
            return {
                "operation_status": {
                    "is_offline": operation_status.is_offline,
                    "offline_since": operation_status.offline_since.isoformat() if operation_status.offline_since else None,
                    "data_processing_active": operation_status.data_processing_active,
                    "alert_generation_active": operation_status.alert_generation_active
                },
                "connectivity": connectivity_status,
                "synchronization": sync_status,
                "storage": storage_stats,
                "data_processing": processing_stats,
                "alerts": alert_stats,
                "operation_statistics": self.operation_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive status: {e}")
            return {"error": str(e)}