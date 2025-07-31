"""
Health monitoring and diagnostics for edge devices
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import uuid

import aioredis
import httpx
from models.config import EdgeManagerConfig
from models.device import (
    EdgeDevice, DeviceStatus, HealthCheckResult, DeviceMetrics
)

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitors health and diagnostics of edge devices"""
    
    def __init__(self, config: EdgeManagerConfig):
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
        self.health_checks: Dict[str, HealthCheckResult] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._ready = False
        self._monitoring = False
        
        # Health monitoring settings
        self.health_check_interval = config.health_check_interval
        self.device_timeout = config.device_timeout
        
        # Health thresholds
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 95.0
        self.memory_warning_threshold = 80.0
        self.memory_critical_threshold = 95.0
        self.storage_warning_threshold = 85.0
        self.storage_critical_threshold = 95.0
        self.temperature_warning_threshold = 70.0
        self.temperature_critical_threshold = 85.0
        
    async def initialize(self):
        """Initialize the health monitor"""
        try:
            # Connect to Redis
            self.redis = aioredis.from_url(self.config.redis_url)
            await self.redis.ping()
            
            # Load existing health checks
            await self._load_health_checks_from_storage()
            
            self._ready = True
            logger.info("Health monitor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize health monitor: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if the health monitor is ready"""
        return self._ready
    
    async def start_monitoring(self):
        """Start health monitoring for all devices"""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        # Start global monitoring task
        asyncio.create_task(self._global_health_monitor())
        
        logger.info("Started health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._monitoring = False
        
        # Cancel all device monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        self.monitoring_tasks.clear()
        logger.info("Stopped health monitoring")
    
    async def add_device_monitoring(self, device: EdgeDevice):
        """Add a device to health monitoring"""
        if device.device_id in self.monitoring_tasks:
            return  # Already monitoring
        
        # Create monitoring task for this device
        task = asyncio.create_task(
            self._monitor_device_health(device.device_id)
        )
        self.monitoring_tasks[device.device_id] = task
        
        logger.info(f"Started monitoring device {device.device_id}")
    
    async def remove_device_monitoring(self, device_id: str):
        """Remove a device from health monitoring"""
        if device_id in self.monitoring_tasks:
            self.monitoring_tasks[device_id].cancel()
            del self.monitoring_tasks[device_id]
            
            # Remove health check data
            if device_id in self.health_checks:
                del self.health_checks[device_id]
                await self._remove_health_check_from_storage(device_id)
            
            logger.info(f"Stopped monitoring device {device_id}")
    
    async def perform_health_check(self, device: EdgeDevice) -> HealthCheckResult:
        """Perform comprehensive health check on a device"""
        start_time = datetime.utcnow()
        
        try:
            # Perform various health checks
            system_health = await self._check_system_health(device)
            model_health = await self._check_model_health(device)
            connectivity_health = await self._check_connectivity_health(device)
            
            # Calculate response time
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Determine overall status
            status = self._determine_device_status(system_health, model_health, connectivity_health)
            
            # Generate issues and recommendations
            issues, warnings, recommendations = self._analyze_health_results(
                system_health, model_health, connectivity_health
            )
            
            # Create health check result
            health_result = HealthCheckResult(
                device_id=device.device_id,
                timestamp=datetime.utcnow(),
                status=status,
                response_time_ms=response_time,
                system_health=system_health,
                model_health=model_health,
                connectivity_health=connectivity_health,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations
            )
            
            # Store result
            self.health_checks[device.device_id] = health_result
            await self._save_health_check_to_storage(health_result)
            
            return health_result
            
        except Exception as e:
            logger.error(f"Health check failed for device {device.device_id}: {e}")
            
            # Create error health result
            error_result = HealthCheckResult(
                device_id=device.device_id,
                timestamp=datetime.utcnow(),
                status=DeviceStatus.ERROR,
                response_time_ms=0,
                system_health={"error": str(e)},
                model_health={"error": str(e)},
                connectivity_health={"error": str(e)},
                issues=[f"Health check failed: {e}"],
                warnings=[],
                recommendations=["Check device connectivity and status"]
            )
            
            self.health_checks[device.device_id] = error_result
            return error_result
    
    async def get_device_health(self, device_id: str) -> Optional[HealthCheckResult]:
        """Get latest health check result for a device"""
        return self.health_checks.get(device_id)
    
    async def get_health_summary(self) -> Dict:
        """Get health summary for all monitored devices"""
        total_devices = len(self.health_checks)
        healthy_devices = len([h for h in self.health_checks.values() if h.status == DeviceStatus.ONLINE])
        warning_devices = len([h for h in self.health_checks.values() if h.warnings])
        critical_devices = len([h for h in self.health_checks.values() if h.issues])
        
        # Get recent issues
        recent_issues = []
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for health_check in self.health_checks.values():
            if health_check.timestamp > cutoff_time and health_check.issues:
                recent_issues.extend([
                    {
                        "device_id": health_check.device_id,
                        "timestamp": health_check.timestamp,
                        "issue": issue
                    } for issue in health_check.issues
                ])
        
        return {
            "total_devices": total_devices,
            "healthy_devices": healthy_devices,
            "warning_devices": warning_devices,
            "critical_devices": critical_devices,
            "recent_issues": recent_issues[-10:],  # Last 10 issues
            "last_updated": datetime.utcnow()
        }
    
    async def get_device_diagnostics(self, device_id: str) -> Dict:
        """Get detailed diagnostics for a specific device"""
        health_check = self.health_checks.get(device_id)
        if not health_check:
            
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def __init__(self):
        self.processors = {
            'time_series': TimeSeriesProcessor(),
            'spatial': SpatialDataProcessor(),
            'spectral': SpectralAnalysisProcessor(),
            'statistical': StatisticalProcessor(),
            'ml_preprocessing': MLPreprocessor()
        }
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', str(uuid.uuid4())),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps based on configuration
        for step_name, step_config in processing_config.items():
            if step_name in self.processors:
                try:
                    processor = self.processors[step_name]
                    processed_data, step_results = processor.process(data, step_config)
                    
                    data = processed_data
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'success',
                        'results': step_results,
                        'output_size': len(data) if isinstance(data, list) else 'N/A'
                    })
                    
                except Exception as e:
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results

class TimeSeriesProcessor:
    """Advanced time series processing for geological data"""
    
    def process(self, data: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process time series data"""
        
        if not isinstance(data, (list, np.ndarray)):
            return data, {'error': 'Invalid data type for time series processing'}
        
        data_array = np.array(data)
        results = {}
        
        # Detrending
        if config.get('detrend', False):
            from scipy import signal
            data_array = signal.detrend(data_array, type=config.get('detrend_type', 'linear'))
            results['detrended'] = True
        
        # Filtering
        if 'filter' in config:
            filter_config = config['filter']
            data_array = self._apply_filter(data_array, filter_config)
            results['filtered'] = filter_config
        
        # Resampling
        if 'resample' in config:
            resample_config = config['resample']
            data_array = self._resample_data(data_array, resample_config)
            results['resampled'] = resample_config
        
        # Anomaly detection
        if config.get('detect_anomalies', False):
            anomalies = self._detect_anomalies(data_array, config.get('anomaly_config', {}))
            results['anomalies'] = anomalies
        
        return data_array.tolist(), results
    
    def _apply_filter(self, data: np.ndarray, filter_config: Dict[str, Any]) -> np.ndarray:
        """Apply digital filter to data"""
        from scipy import signal
        
        filter_type = filter_config.get('type', 'bandpass')
        sampling_rate = filter_config.get('sampling_rate', 100.0)
        
        if filter_type == 'bandpass':
            low_freq = filter_config.get('low_freq', 1.0)
            high_freq = filter_config.get('high_freq', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                [low_freq/nyquist, high_freq/nyquist],
                btype='band'
            )
            return signal.filtfilt(b, a, data)
        
        elif filter_type == 'lowpass':
            cutoff = filter_config.get('cutoff', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                cutoff/nyquist,
                btype='low'
            )
            return signal.filtfilt(b, a, data)
        
        return data
    
    def _resample_data(self, data: np.ndarray, resample_config: Dict[str, Any]) -> np.ndarray:
        """Resample data to different sampling rate"""
        from scipy import signal
        
        original_rate = resample_config.get('original_rate', 100.0)
        target_rate = resample_config.get('target_rate', 50.0)
        
        if original_rate != target_rate:
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        
        return data
    
    def _detect_anomalies(self, data: np.ndarray, anomaly_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        method = anomaly_config.get('method', 'statistical')
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = anomaly_config.get('threshold', 3.0)
            
            anomaly_indices = np.where(np.abs(data - mean_val) > threshold * std_val)[0]
            
            return {
                'method': 'statistical',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'threshold_used': threshold
            }
        
        elif method == 'isolation_forest':
            # Machine learning based anomaly detection
            from sklearn.ensemble import IsolationForest
            
            contamination = anomaly_config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            
            data_reshaped = data.reshape(-1, 1)
            anomaly_labels = clf.fit_predict(data_reshaped)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'contamination': contamination
            }
        
        return {'method': 'none', 'anomaly_count': 0}

        
        # Get historical metrics if available
        historical_metrics = await self._get_historical_metrics(device_id)
        
        return {
            "device_id": device_id,
            "current_status": health_check.status.value,
            "last_check": health_check.timestamp,
            "response_time_ms": health_check.response_time_ms,
            "system_health": health_check.system_health,
            "model_health": health_check.model_health,
            "connectivity_health": health_check.connectivity_health,
            "issues": health_check.issues,
            "warnings": health_check.warnings,
            "recommendations": health_check.recommendations,
            "historical_metrics": historical_metrics
        }
    
    async def _global_health_monitor(self):
        """Global health monitoring task"""
        while self._monitoring:
            try:
                # Perform periodic cleanup and analysis
                await self._cleanup_old_health_data()
                await self._analyze_health_trends()
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in global health monitor: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _monitor_device_health(self, device_id: str):
        """Monitor health of a specific device"""
        while self._monitoring:
            try:
                # This would be implemented to get device reference
                # For now, skip if we can't get device info
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring device {device_id}: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_system_health(self, device: EdgeDevice) -> Dict:
        """Check system health metrics"""
        system_health = {
            "status": "unknown",
            "cpu_usage": 0,
            "memory_usage": 0,
            "storage_usage": 0,
            "temperature": None,
            "uptime": None
        }
        
        try:
            # Get current metrics from device
            if device.current_metrics:
                metrics = device.current_metrics
                system_health.update({
                    "cpu_usage": metrics.cpu_usage_percent,
                    "memory_usage": metrics.memory_usage_percent,
                    "storage_usage": metrics.storage_usage_percent,
                    "temperature": metrics.temperature_celsius
                })
                
                # Determine system status based on thresholds
                if (metrics.cpu_usage_percent > self.cpu_critical_threshold or
                    metrics.memory_usage_percent > self.memory_critical_threshold or
                    metrics.storage_usage_percent > self.storage_critical_threshold):
                    system_health["status"] = "critical"
                elif (metrics.cpu_usage_percent > self.cpu_warning_threshold or
                      metrics.memory_usage_percent > self.memory_warning_threshold or
                      metrics.storage_usage_percent > self.storage_warning_threshold):
                    system_health["status"] = "warning"
                else:
                    system_health["status"] = "healthy"
            
            # Check uptime
            if device.last_seen:
                uptime = (datetime.utcnow() - device.last_seen).total_seconds()
                system_health["uptime"] = uptime
            
        except Exception as e:
            logger.error(f"Failed to check system health for device {device.device_id}: {e}")
            system_health["status"] = "error"
            system_health["error"] = str(e)
        
        return system_health
    
    async def _check_model_health(self, device: EdgeDevice) -> Dict:
        """Check health of deployed models"""
        model_health = {
            "status": "unknown",
            "active_models": 0,
            "failed_models": 0,
            "model_details": []
        }
        
        try:
            # Check deployed models
            active_models = len(device.deployed_models)
            model_health["active_models"] = active_models
            
            if active_models > 0:
                model_health["status"] = "healthy"
            else:
                model_health["status"] = "no_models"
            
            # Add model details (would be populated from actual model status)
            for model_id in device.deployed_models:
                model_health["model_details"].append({
                    "model_id": model_id,
                    "status": "running",  # Would check actual status
                    "health": "healthy"
                })
            
        except Exception as e:
            logger.error(f"Failed to check model health for device {device.device_id}: {e}")
            model_health["status"] = "error"
            model_health["error"] = str(e)
        
        return model_health
    
    async def _check_connectivity_health(self, device: EdgeDevice) -> Dict:
        """Check network connectivity health"""
        connectivity_health = {
            "status": "unknown",
            "ping_response_ms": None,
            "last_seen": None,
            "network_quality": "unknown"
        }
        
        try:
            # Check last seen time
            if device.last_seen:
                time_since_seen = (datetime.utcnow() - device.last_seen).total_seconds()
                connectivity_health["last_seen"] = time_since_seen
                
                if time_since_seen < 60:  # Less than 1 minute
                    connectivity_health["status"] = "excellent"
                elif time_since_seen < 300:  # Less than 5 minutes
                    connectivity_health["status"] = "good"
                elif time_since_seen < 900:  # Less than 15 minutes
                    connectivity_health["status"] = "poor"
                else:
                    connectivity_health["status"] = "disconnected"
            
            # Check network metrics
            if device.current_metrics and device.current_metrics.network_latency_ms:
                latency = device.current_metrics.network_latency_ms
                connectivity_health["ping_response_ms"] = latency
                
                if latency < 50:
                    connectivity_health["network_quality"] = "excellent"
                elif latency < 150:
                    connectivity_health["network_quality"] = "good"
                elif latency < 300:
                    connectivity_health["network_quality"] = "fair"
                else:
                    connectivity_health["network_quality"] = "poor"
            
        except Exception as e:
            logger.error(f"Failed to check connectivity health for device {device.device_id}: {e}")
            connectivity_health["status"] = "error"
            connectivity_health["error"] = str(e)
        
        return connectivity_health
    
    def _determine_device_status(self, system_health: Dict, model_health: Dict, 
                               connectivity_health: Dict) -> DeviceStatus:
        """Determine overall device status from health checks"""
        # Check for errors
        if (system_health.get("status") == "error" or
            model_health.get("status") == "error" or
            connectivity_health.get("status") == "error"):
            return DeviceStatus.ERROR
        
        # Check for critical issues
        if (system_health.get("status") == "critical" or
            connectivity_health.get("status") == "disconnected"):
            return DeviceStatus.OFFLINE
        
        # Check for warnings
        if (system_health.get("status") == "warning" or
            connectivity_health.get("status") == "poor"):
            return DeviceStatus.MAINTENANCE
        
        # Default to online if no issues
        return DeviceStatus.ONLINE
    
    def _analyze_health_results(self, system_health: Dict, model_health: Dict, 
                              connectivity_health: Dict) -> tuple:
        """Analyze health results and generate issues, warnings, and recommendations"""
        issues = []
        warnings = []
        recommendations = []
        
        # System health analysis
        if system_health.get("status") == "critical":
            if system_health.get("cpu_usage", 0) > self.cpu_critical_threshold:
                issues.append(f"Critical CPU usage: {system_health['cpu_usage']:.1f}%")
                recommendations.append("Reduce computational load or upgrade CPU")
            
            if system_health.get("memory_usage", 0) > self.memory_critical_threshold:
                issues.append(f"Critical memory usage: {system_health['memory_usage']:.1f}%")
                recommendations.append("Free memory or add more RAM")
            
            if system_health.get("storage_usage", 0) > self.storage_critical_threshold:
                issues.append(f"Critical storage usage: {system_health['storage_usage']:.1f}%")
                recommendations.append("Clean up storage or add more disk space")
        
        elif system_health.get("status") == "warning":
            if system_health.get("cpu_usage", 0) > self.cpu_warning_threshold:
                warnings.append(f"High CPU usage: {system_health['cpu_usage']:.1f}%")
            
            if system_health.get("memory_usage", 0) > self.memory_warning_threshold:
                warnings.append(f"High memory usage: {system_health['memory_usage']:.1f}%")
            
            if system_health.get("storage_usage", 0) > self.storage_warning_threshold:
                warnings.append(f"High storage usage: {system_health['storage_usage']:.1f}%")
        
        # Temperature checks
        temp = system_health.get("temperature")
        if temp:
            if temp > self.temperature_critical_threshold:
                issues.append(f"Critical temperature: {temp:.1f}°C")
                recommendations.append("Check cooling system and reduce load")
            elif temp > self.temperature_warning_threshold:
                warnings.append(f"High temperature: {temp:.1f}°C")
                recommendations.append("Monitor temperature and ensure proper ventilation")
        
        # Connectivity analysis
        if connectivity_health.get("status") == "disconnected":
            issues.append("Device disconnected")
            recommendations.append("Check network connectivity and device power")
        elif connectivity_health.get("status") == "poor":
            warnings.append("Poor network connectivity")
            recommendations.append("Check network connection quality")
        
        # Model health analysis
        if model_health.get("failed_models", 0) > 0:
            issues.append(f"{model_health['failed_models']} failed models")
            recommendations.append("Restart failed models or check model configurations")
        
        return issues, warnings, recommendations
    
    async def _get_historical_metrics(self, device_id: str) -> List[Dict]:
        """Get historical metrics for a device"""
        if not self.redis:
            return []
        
        try:
            # Get metrics from last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            cutoff_timestamp = int(cutoff_time.timestamp())
            
            pattern = f"metrics:{device_id}:*"
            keys = await self.redis.keys(pattern)
            
            metrics = []
            for key in keys:
                # Extract timestamp from key
                key_str = key.decode() if isinstance(key, bytes) else key
                timestamp_str = key_str.split(":")[-1]
                try:
                    timestamp = int(timestamp_str)
                    if timestamp >= cutoff_timestamp:
                        metric_data = await self.redis.get(key)
                        if metric_data:
                            metrics.append(json.loads(metric_data))
                except ValueError:
                    continue
            
            # Sort by timestamp
            metrics.sort(key=lambda x: x.get('timestamp', ''))
            return metrics[-100:]  # Return last 100 metrics
            
        except Exception as e:
            logger.error(f"Failed to get historical metrics for device {device_id}: {e}")
            return []
    
    async def _cleanup_old_health_data(self):
        """Clean up old health check data"""
        # Remove health checks older than 7 days
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        
        to_remove = []
        for device_id, health_check in self.health_checks.items():
            if health_check.timestamp < cutoff_time:
                to_remove.append(device_id)
        
        for device_id in to_remove:
            del self.health_checks[device_id]
            await self._remove_health_check_from_storage(device_id)
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old health check records")
    
    async def _analyze_health_trends(self):
        """Analyze health trends across all devices"""
        # This could implement trend analysis, anomaly detection, etc.
        # For now, just log summary statistics
        
        if not self.health_checks:
            return
        
        total_devices = len(self.health_checks)
        healthy_count = len([h for h in self.health_checks.values() if h.status == DeviceStatus.ONLINE])
        
        health_percentage = (healthy_count / total_devices) * 100
        
        logger.info(f"Health trend: {healthy_count}/{total_devices} devices healthy ({health_percentage:.1f}%)")
    
    async def _load_health_checks_from_storage(self):
        """Load health checks from Redis storage"""
        if not self.redis:
            return
        
        try:
            health_keys = await self.redis.keys("health:*")
            for key in health_keys:
                health_data = await self.redis.get(key)
                if health_data:
                    health_dict = json.loads(health_data)
                    health_check = HealthCheckResult(**health_dict)
                    self.health_checks[health_check.device_id] = health_check
            
            logger.info(f"Loaded {len(self.health_checks)} health checks from storage")
            
        except Exception as e:
            logger.error(f"Failed to load health checks from storage: {e}")
    
    async def _save_health_check_to_storage(self, health_check: HealthCheckResult):
        """Save health check to Redis storage"""
        if not self.redis:
            return
        
        try:
            health_key = f"health:{health_check.device_id}"
            health_data = health_check.model_dump_json()
            await self.redis.setex(health_key, 604800, health_data)  # Expire after 7 days
            
        except Exception as e:
            logger.error(f"Failed to save health check for device {health_check.device_id}: {e}")
    
    async def _remove_health_check_from_storage(self, device_id: str):
        """Remove health check from Redis storage"""
        if not self.redis:
            return
        
        try:
            health_key = f"health:{device_id}"
            await self.redis.delete(health_key)
            
        except Exception as e:
            logger.error(f"Failed to remove health check for device {device_id}: {e}")