"""
Core device management functionality
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import uuid

import aioredis
from models.config import EdgeManagerConfig
from models.device import (
    EdgeDevice, DeviceStatus, DeviceRegistrationRequest, 
    DeviceRegistrationResponse, DeviceMetrics, EdgeDeviceConfig,
    DeviceCapabilities, DeviceLocation
)

logger = logging.getLogger(__name__)


class EdgeDeviceManager:
    """Manages edge device registration, configuration, and lifecycle"""
    
    def __init__(self, config: EdgeManagerConfig):
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
        self.devices: Dict[str, EdgeDevice] = {}
        self.device_heartbeats: Dict[str, datetime] = {}
        self._ready = False
        
        # Device management settings
        self.heartbeat_interval = config.device_heartbeat_interval
        self.device_timeout = config.device_timeout
        self.max_devices_per_region = config.max_devices_per_region
        
    async def initialize(self):
        """Initialize the device manager"""
        try:
            # Connect to Redis
            self.redis = aioredis.from_url(self.config.redis_url)
            await self.redis.ping()
            
            # Load existing devices from Redis
            await self._load_devices_from_storage()
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._device_cleanup_task())
            
            self._ready = True
            logger.info(f"Device manager initialized with {len(self.devices)} devices")
            
        except Exception as e:
            logger.error(f"Failed to initialize device manager: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis:
            await self.redis.close()
        self._ready = False
    
    def is_ready(self) -> bool:
        """Check if the device manager is ready"""
        return self._ready
    
    async def register_device(self, request: DeviceRegistrationRequest) -> DeviceRegistrationResponse:
        """Register a new edge device"""
        try:
            # Generate unique device ID
            device_id = f"edge-{uuid.uuid4().hex[:12]}"
            
            # Check regional limits
            region_devices = await self._count_devices_in_region(request.location.region)
            if region_devices >= self.max_devices_per_region:
                raise ValueError(f"Maximum devices ({self.max_devices_per_region}) reached for region {request.location.region}")
            
            # Generate authentication token
            auth_token = self._generate_auth_token(device_id)
            
            # Create device configuration
            device_config = EdgeDeviceConfig(
                device_id=device_id,
                device_name=request.device_name,
                device_type=request.device_type,
                location=request.location,
                capabilities=request.capabilities,
                ip_address=request.ip_address,
                port=request.port,
                ssl_enabled=request.ssl_enabled,
                auth_token=auth_token
            )
            
            # Create device record
            device = EdgeDevice(
                device_id=device_id,
                device_name=request.device_name,
                device_type=request.device_type,
                location=request.location,
                capabilities=request.capabilities,
                config=device_config,
                status=DeviceStatus.ONLINE,
                registered_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                firmware_version=request.firmware_version,
                software_version=request.software_version
            )
            
            # Store device
            self.devices[device_id] = device
            await self._save_device_to_storage(device)
            
            # Update heartbeat
            self.device_heartbeats[device_id] = datetime.utcnow()
            
            logger.info(f"Registered new device: {device_id} ({request.device_name})")
            
            # Create response
            response = DeviceRegistrationResponse(
                device_id=device_id,
                auth_token=auth_token,
                config=device_config,
                management_endpoint=f"http://localhost:{self.config.consul_port}/api/v1/devices",
                sync_endpoint=f"http://localhost:{self.config.consul_port}/api/v1/sync",
                model_registry_endpoint=self.config.model_registry_url
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to register device: {e}")
            raise
    
    async def update_device_heartbeat(self, device_id: str, metrics: Optional[DeviceMetrics] = None):
        """Update device heartbeat and metrics"""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")
        
        # Update heartbeat timestamp
        self.device_heartbeats[device_id] = datetime.utcnow()
        
        # Update device status
        device = self.devices[device_id]
        device.status = DeviceStatus.ONLINE
        device.last_seen = datetime.utcnow()
        
        # Update metrics if provided
        if metrics:
            device.current_metrics = metrics
            await self._store_device_metrics(device_id, metrics)
        
        # Save to storage
        await self._save_device_to_storage(device)
        
        logger.debug(f"Updated heartbeat for device {device_id}")
    
    async def get_device(self, device_id: str) -> Optional[EdgeDevice]:
        """Get device by ID"""
        return self.devices.get(device_id)
    
    async def list_devices(self, 
                          region: Optional[str] = None,
                          device_type: Optional[str] = None,
                          status: Optional[DeviceStatus] = None) -> List[EdgeDevice]:
        """List devices with optional filters"""
        devices = list(self.devices.values())
        
        if region:
            devices = [d for d in devices if d.location.region == region]
        
        if device_type:
            devices = [d for d in devices if d.device_type == device_type]
        
        if status:
            devices = [d for d in devices if d.status == status]
        
        return devices
    
    async def update_device_config(self, device_id: str, config_updates: Dict) -> EdgeDevice:
        """Update device configuration"""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")
        
        device = self.devices[device_id]
        
        # Update configuration fields
        for key, value in config_updates.items():
            if hasattr(device.config, key):
                setattr(device.config, key, value)
        
        device.last_updated = datetime.utcnow()
        
        # Save to storage
        await self._save_device_to_storage(device)
        
        logger.info(f"Updated configuration for device {device_id}")
        return device
    
    async def remove_device(self, device_id: str):
        """Remove a device from management"""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")
        
        # Remove from memory
        del self.devices[device_id]
        if device_id in self.device_heartbeats:
            del self.device_heartbeats[device_id]
        
        # Remove from storage
        await self._remove_device_from_storage(device_id)
        
        logger.info(f"Removed device {device_id}")
    
    async def get_metrics(self) -> Dict:
        """Get device manager metrics"""
        online_devices = len([d for d in self.devices.values() if d.status == DeviceStatus.ONLINE])
        offline_devices = len([d for d in self.devices.values() if d.status == DeviceStatus.OFFLINE])
        
        # Group by region
        regions = {}
        for device in self.devices.values():
            region = device.location.region
            if region not in regions:
                regions[region] = {"total": 0, "online": 0, "offline": 0}
            regions[region]["total"] += 1
            if device.status == DeviceStatus.ONLINE:
                regions[region]["online"] += 1
            else:
                regions[region]["offline"] += 1
        
        return {
            "total_devices": len(self.devices),
            "online_devices": online_devices,
            "offline_devices": offline_devices,
            "regions": regions,
            "device_types": self._get_device_type_counts()
        }
    
    def _generate_auth_token(self, device_id: str) -> str:
        """Generate authentication token for device"""
        # In production, use proper JWT token generation
        import hashlib
        import time
        
        payload = f"{device_id}:{time.time()}:{self.config.jwt_secret}"
        return hashlib.sha256(payload.encode()).hexdigest()
    
    async def _count_devices_in_region(self, region: str) -> int:
        """Count devices in a specific region"""
        return len([d for d in self.devices.values() if d.location.region == region])
    
    def _get_device_type_counts(self) -> Dict[str, int]:
        """Get count of devices by type"""
        type_counts = {}
        for device in self.devices.values():
            device_type = device.device_type
            type_counts[device_type] = type_counts.get(device_type, 0) + 1
        return type_counts
    
    async def _heartbeat_monitor(self):
        """Monitor device heartbeats and update status"""
        while self._ready:
            try:
                current_time = datetime.utcnow()
                timeout_threshold = current_time - timedelta(seconds=self.device_timeout)
                
                for device_id, last_heartbeat in self.device_heartbeats.items():
                    if last_heartbeat < timeout_threshold:
                        # Device timed out
                        if device_id in self.devices:
                            device = self.devices[device_id]
                            if device.status != DeviceStatus.OFFLINE:
                                device.status = DeviceStatus.OFFLINE
                                await self._save_device_to_storage(device)
                                logger.warning(f"Device {device_id} marked as offline (timeout)")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _device_cleanup_task(self):
        """Periodic cleanup of old device data"""
        while self._ready:
            try:
                # Clean up old metrics (keep last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                for device_id in self.devices.keys():
                    await self._cleanup_old_metrics(device_id, cutoff_time)
                
                # Sleep for 1 hour before next cleanup
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in device cleanup task: {e}")
                await asyncio.sleep(3600)
    
    async def _load_devices_from_storage(self):
        """Load devices from Redis storage"""
        if not self.redis:
            return
        
        try:
            device_keys = await self.redis.keys("device:*")
            for key in device_keys:
                device_data = await self.redis.get(key)
                if device_data:
                    device_dict = json.loads(device_data)
                    device = EdgeDevice(**device_dict)
                    self.devices[device.device_id] = device
                    
                    # Initialize heartbeat
                    if device.last_seen:
                        self.device_heartbeats[device.device_id] = device.last_seen
            
            logger.info(f"Loaded {len(self.devices)} devices from storage")
            
        except Exception as e:
            logger.error(f"Failed to load devices from storage: {e}")
    
    async def _save_device_to_storage(self, device: EdgeDevice):
        """Save device to Redis storage"""
        if not self.redis:
            return
        
        try:
            device_key = f"device:{device.device_id}"
            device_data = device.model_dump_json()
            await self.redis.set(device_key, device_data)
            
        except Exception as e:
            logger.error(f"Failed to save device {device.device_id} to storage: {e}")
    
    async def _remove_device_from_storage(self, device_id: str):
        """Remove device from Redis storage"""
        if not self.redis:
            return
        
        try:
            device_key = f"device:{device_id}"
            await self.redis.delete(device_key)
            
            # Also remove metrics
            metrics_pattern = f"metrics:{device_id}:*"
            metrics_keys = await self.redis.keys(metrics_pattern)
            if metrics_keys:
                await self.redis.delete(*metrics_keys)
            
        except Exception as e:
            logger.error(f"Failed to remove device {device_id} from storage: {e}")
    
    async def _store_device_metrics(self, device_id: str, metrics: DeviceMetrics):
        """Store device metrics in Redis"""
        if not self.redis:
            return
        
        try:
            metrics_key = f"metrics:{device_id}:{int(metrics.timestamp.timestamp())}"
            metrics_data = metrics.model_dump_json()
            await self.redis.setex(metrics_key, 86400, metrics_data)  # Expire after 24 hours
            
        except Exception as e:
            logger.error(f"Failed to store metrics for device {device_id}: {e}")
    
    async def _cleanup_old_metrics(self, device_id: str, cutoff_time: datetime):
        """Clean up old metrics for a device"""
        if not self.redis:
            return
        
        try:
            cutoff_timestamp = int(cutoff_time.timestamp())
            pattern = f"metrics:{device_id}:*"
            
            keys = await self.redis.keys(pattern)
            old_keys = []
            
            for key in keys:
                # Extract timestamp from key
                key_str = key.decode() if isinstance(key, bytes) else key
                timestamp_str = key_str.split(":")[-1]
                try:
                    timestamp = int(timestamp_str)
                    if timestamp < cutoff_timestamp:
                        old_keys.append(key)
                except ValueError:
                    continue
            
            if old_keys:
                await self.redis.delete(*old_keys)
                logger.debug(f"Cleaned up {len(old_keys)} old metrics for device {device_id}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics for device {device_id}: {e}")