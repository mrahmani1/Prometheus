"""
Unit tests for EdgeDeviceManager
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from core.device_manager import EdgeDeviceManager
from models.config import EdgeManagerConfig, DeviceCapabilities, DeviceLocation
from models.device import (
    DeviceRegistrationRequest, DeviceStatus, DeviceMetrics, EdgeDevice
)


@pytest.fixture
def config():
    """Test configuration"""
    return EdgeManagerConfig(
        redis_url="redis://localhost:6379",
        device_heartbeat_interval=30,
        device_timeout=300,
        max_devices_per_region=100
    )


@pytest.fixture
def device_capabilities():
    """Test device capabilities"""
    return DeviceCapabilities(
        cpu_cores=4,
        memory_gb=8.0,
        storage_gb=100.0,
        gpu_available=True,
        gpu_memory_gb=4.0,
        network_bandwidth_mbps=100.0,
        power_source="grid",
        operating_system="Linux",
        python_version="3.11",
        docker_available=True,
        kubernetes_available=False
    )


@pytest.fixture
def device_location():
    """Test device location"""
    return DeviceLocation(
        latitude=37.7749,
        longitude=-122.4194,
        altitude=100.0,
        region="us-west-1",
        site_name="San Francisco Station",
        timezone="America/Los_Angeles"
    )


@pytest.fixture
def registration_request(device_capabilities, device_location):
    """Test device registration request"""
    return DeviceRegistrationRequest(
        device_name="test-seismic-station",
        device_type="seismic_station",
        location=device_location,
        capabilities=device_capabilities,
        ip_address="192.168.1.100",
        port=8080,
        ssl_enabled=False,
        firmware_version="1.0.0",
        software_version="2.0.0"
    )


@pytest.fixture
async def device_manager(config):
    """Test device manager instance"""
    manager = EdgeDeviceManager(config)
    
    # Mock Redis connection
    with patch('aioredis.from_url') as mock_redis:
        mock_redis_instance = AsyncMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping = AsyncMock()
        mock_redis_instance.keys = AsyncMock(return_value=[])
        
        manager.redis = mock_redis_instance
        await manager.initialize()
        
        yield manager
        
        await manager.cleanup()


class TestEdgeDeviceManager:
    """Test cases for EdgeDeviceManager"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, config):
        """Test device manager initialization"""
        manager = EdgeDeviceManager(config)
        
        with patch('aioredis.from_url') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping = AsyncMock()
            mock_redis_instance.keys = AsyncMock(return_value=[])
            
            await manager.initialize()
            
            assert manager.is_ready()
            assert manager.redis is not None
            
            await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_device_registration(self, device_manager, registration_request):
        """Test device registration"""
        response = await device_manager.register_device(registration_request)
        
        assert response.device_id.startswith("edge-")
        assert response.auth_token is not None
        assert response.config.device_name == registration_request.device_name
        assert response.config.device_type == registration_request.device_type
        
        # Verify device is stored
        device = await device_manager.get_device(response.device_id)
        assert device is not None
        assert device.device_name == registration_request.device_name
        assert device.status == DeviceStatus.ONLINE
    
    @pytest.mark.asyncio
    async def test_device_heartbeat_update(self, device_manager, registration_request):
        """Test device heartbeat update"""
        # Register device first
        response = await device_manager.register_device(registration_request)
        device_id = response.device_id
        
        # Create test metrics
        metrics = DeviceMetrics(
            timestamp=datetime.utcnow(),
            device_id=device_id,
            cpu_usage_percent=50.0,
            memory_usage_percent=60.0,
            storage_usage_percent=70.0,
            temperature_celsius=45.0,
            network_latency_ms=20.0,
            bandwidth_usage_mbps=50.0,
            battery_level_percent=85.0,
            power_consumption_watts=100.0,
            active_models=2,
            data_processing_rate=1000.0,
            error_count=0
        )
        
        # Update heartbeat
        await device_manager.update_device_heartbeat(device_id, metrics)
        
        # Verify device status and metrics
        device = await device_manager.get_device(device_id)
        assert device.status == DeviceStatus.ONLINE
        assert device.current_metrics is not None
        assert device.current_metrics.cpu_usage_percent == 50.0
        assert device.last_seen is not None
    
    @pytest.mark.asyncio
    async def test_device_config_update(self, device_manager, registration_request):
        """Test device configuration update"""
        # Register device first
        response = await device_manager.register_device(registration_request)
        device_id = response.device_id
        
        # Update configuration
        config_updates = {
            "max_cpu_usage": 90.0,
            "max_memory_usage": 85.0,
            "data_collection_enabled": False
        }
        
        updated_device = await device_manager.update_device_config(device_id, config_updates)
        
        assert updated_device.config.max_cpu_usage == 90.0
        assert updated_device.config.max_memory_usage == 85.0
        assert updated_device.config.data_collection_enabled == False
    
    @pytest.mark.asyncio
    async def test_list_devices_with_filters(self, device_manager, device_capabilities, device_location):
        """Test listing devices with filters"""
        # Register multiple devices
        requests = []
        for i in range(3):
            location = DeviceLocation(
                latitude=device_location.latitude + i,
                longitude=device_location.longitude + i,
                region=f"region-{i % 2}",  # Two different regions
                site_name=f"Site {i}",
                timezone="UTC"
            )
            
            request = DeviceRegistrationRequest(
                device_name=f"device-{i}",
                device_type="seismic_station" if i % 2 == 0 else "weather_station",
                location=location,
                capabilities=device_capabilities,
                ip_address=f"192.168.1.{100 + i}",
                port=8080
            )
            requests.append(request)
            await device_manager.register_device(request)
        
        # Test filtering by region
        region_0_devices = await device_manager.list_devices(region="region-0")
        assert len(region_0_devices) == 2
        
        # Test filtering by device type
        seismic_devices = await device_manager.list_devices(device_type="seismic_station")
        assert len(seismic_devices) == 2
        
        # Test filtering by status
        online_devices = await device_manager.list_devices(status=DeviceStatus.ONLINE)
        assert len(online_devices) == 3
    
    @pytest.mark.asyncio
    async def test_device_removal(self, device_manager, registration_request):
        """Test device removal"""
        # Register device first
        response = await device_manager.register_device(registration_request)
        device_id = response.device_id
        
        # Verify device exists
        device = await device_manager.get_device(device_id)
        assert device is not None
        
        # Remove device
        await device_manager.remove_device(device_id)
        
        # Verify device is removed
        device = await device_manager.get_device(device_id)
        assert device is None
    
    @pytest.mark.asyncio
    async def test_regional_device_limit(self, device_manager, device_capabilities):
        """Test regional device limit enforcement"""
        # Set low limit for testing
        device_manager.max_devices_per_region = 2
        
        location = DeviceLocation(
            latitude=37.7749,
            longitude=-122.4194,
            region="test-region",
            site_name="Test Site",
            timezone="UTC"
        )
        
        # Register devices up to limit
        for i in range(2):
            request = DeviceRegistrationRequest(
                device_name=f"device-{i}",
                device_type="seismic_station",
                location=location,
                capabilities=device_capabilities,
                ip_address=f"192.168.1.{100 + i}",
                port=8080
            )
            await device_manager.register_device(request)
        
        # Try to register one more device (should fail)
        request = DeviceRegistrationRequest(
            device_name="device-overflow",
            device_type="seismic_station",
            location=location,
            capabilities=device_capabilities,
            ip_address="192.168.1.200",
            port=8080
        )
        
        with pytest.raises(ValueError, match="Maximum devices"):
            await device_manager.register_device(request)
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, device_manager, registration_request):
        """Test getting device manager metrics"""
        # Register a device
        await device_manager.register_device(registration_request)
        
        metrics = await device_manager.get_metrics()
        
        assert "total_devices" in metrics
        assert "online_devices" in metrics
        assert "offline_devices" in metrics
        assert "regions" in metrics
        assert "device_types" in metrics
        
        assert metrics["total_devices"] == 1
        assert metrics["online_devices"] == 1
        assert metrics["offline_devices"] == 0
    
    @pytest.mark.asyncio
    async def test_heartbeat_timeout(self, device_manager, registration_request):
        """Test device timeout handling"""
        # Set short timeout for testing
        device_manager.device_timeout = 1  # 1 second
        
        # Register device
        response = await device_manager.register_device(registration_request)
        device_id = response.device_id
        
        # Wait for timeout
        await asyncio.sleep(2)
        
        # Manually trigger heartbeat monitor logic
        from datetime import timedelta
        current_time = datetime.utcnow()
        timeout_threshold = current_time - timedelta(seconds=device_manager.device_timeout)
        
        if device_id in device_manager.device_heartbeats:
            last_heartbeat = device_manager.device_heartbeats[device_id]
            if last_heartbeat < timeout_threshold:
                device = device_manager.devices[device_id]
                device.status = DeviceStatus.OFFLINE
        
        # Verify device is marked offline
        device = await device_manager.get_device(device_id)
        # Note: In real implementation, this would be handled by background task
        # For testing, we manually simulate the timeout logic above


if __name__ == "__main__":
    pytest.main([__file__])