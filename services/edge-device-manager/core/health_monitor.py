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
            return {"error": "No health data available"}
        
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