#!/usr/bin/env python3
"""
Example script demonstrating edge device registration and management
"""

import asyncio
from datetime import datetime
from typing import Any, Dict

import aiohttp
import pandas as pd
import torch.nn.functional as F

class EdgeDeviceClient:
    """Client for interacting with Edge Device Manager API"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
        self.device_id = None
        self.auth_token = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def register_device(self, device_config: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new edge device"""
        url = f"{self.base_url}/api/v1/devices/register"

        async with self.session.post(url, json=device_config) as response:
            if response.status == 200:
                result = await response.json()
                self.device_id = result["device_id"]
                self.auth_token = result["auth_token"]
                return result
            else:
                error_text = await response.text()
                raise Exception(
                    f"Registration failed: {response.status} - {error_text}"
                )

    async def send_heartbeat(self, metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send heartbeat with optional metrics"""
        if not self.device_id:
            raise Exception("Device not registered")

        url = f"{self.base_url}/api/v1/devices/{self.device_id}/heartbeat"

        payload = {}
        if metrics:
            payload = metrics

        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Heartbeat failed: {response.status} - {error_text}")

    async def get_device_status(self) -> Dict[str, Any]:
        """Get device status"""
        if not self.device_id:
            raise Exception("Device not registered")

        url = f"{self.base_url}/api/v1/devices/{self.device_id}/status"

        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(
                    f"Status check failed: {response.status} - {error_text}"
                )

    async def update_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update device configuration"""
        if not self.device_id:
            raise Exception("Device not registered")

        url = f"{self.base_url}/api/v1/devices/{self.device_id}/config"

        async with self.session.put(url, json=config_updates) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(
                    f"Config update failed: {response.status} - {error_text}"
                )

async def simulate_seismic_station():
    """Simulate a seismic monitoring station"""
    print("🌋 Starting Seismic Station Simulation")

    # Device configuration
    device_config = {
        "device_name": "yellowstone-seismic-01",
        "device_type": "seismic_station",
        "location": {
            "latitude": 44.4280,
            "longitude": -110.5885,
            "altitude": 2400.0,
            "region": "us-west-1",
            "site_name": "Yellowstone National Park - Old Faithful",
            "timezone": "America/Denver",
        },
        "capabilities": {
            "cpu_cores": 4,
            "memory_gb": 8.0,
            "storage_gb": 500.0,
            "gpu_available": False,
            "network_bandwidth_mbps": 50.0,
            "power_source": "solar",
            "operating_system": "Linux",
            "python_version": "3.11",
            "docker_available": True,
            "kubernetes_available": False,
        },
        "ip_address": "192.168.100.10",
        "port": 8080,
        "ssl_enabled": True,
        "firmware_version": "2.1.0",
        "software_version": "1.5.2",
    }

    async with EdgeDeviceClient() as client:
        try:
            # Register device
            print("📝 Registering seismic station...")
            registration_result = await client.register_device(device_config)
            print(f"✅ Device registered: {registration_result['device_id']}")
            print(f"🔑 Auth token: {registration_result['auth_token'][:20]}...")

            # Send initial heartbeat
            print("\n💓 Sending initial heartbeat...")
            initial_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "device_id": client.device_id,
                "cpu_usage_percent": 25.0,
                "memory_usage_percent": 40.0,
                "storage_usage_percent": 15.0,
                "temperature_celsius": 22.0,
                "network_latency_ms": 45.0,
                "bandwidth_usage_mbps": 5.0,
                "power_consumption_watts": 85.0,
                "active_models": 1,
                "data_processing_rate": 100.0,
                "error_count": 0,
            }

            heartbeat_result = await client.send_heartbeat(initial_metrics)
            print(f"✅ Heartbeat sent: {heartbeat_result['status']}")

            # Get device status
            print("\n📊 Checking device status...")
            status = await client.get_device_status()
            print(f"Status: {status['status']}")
            print(f"Last seen: {status['last_seen']}")
            print(f"Deployed models: {status['deployed_models']}")

            # Simulate some activity
            print("\n🔄 Simulating seismic activity detection...")
            for i in range(3):
                await asyncio.sleep(2)

                # Simulate varying metrics during seismic activity
                activity_metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "device_id": client.device_id,
                    "cpu_usage_percent": 45.0 + (i * 10),
                    "memory_usage_percent": 55.0 + (i * 5),
                    "storage_usage_percent": 15.0 + (i * 2),
                    "temperature_celsius": 25.0 + (i * 3),
                    "network_latency_ms": 45.0,
                    "bandwidth_usage_mbps": 15.0 + (i * 5),
                    "power_consumption_watts": 120.0 + (i * 10),
                    "active_models": 2,
                    "data_processing_rate": 500.0 + (i * 100),
                    "error_count": 0,
                }

                await client.send_heartbeat(activity_metrics)
                print(
                    f"📈 Activity cycle {i+1}: CPU {activity_metrics['cpu_usage_percent']:.1f}%, "
                    f"Processing {activity_metrics['data_processing_rate']:.0f} samples/sec"
                )

            # Update configuration
            print("\n⚙️ Updating device configuration...")
            config_updates = {
                "max_cpu_usage": 85.0,
                "data_collection_enabled": True,
                "max_concurrent_models": 3,
            }

            update_result = await client.update_config(config_updates)
            print(f"✅ Configuration updated: {update_result['status']}")

            print("\n🎉 Seismic station simulation completed successfully!")

        except Exception as e:
            print(f"❌ Error during simulation: {e}")

async def simulate_weather_station():
    """Simulate a weather monitoring station"""
    print("🌤️ Starting Weather Station Simulation")

    device_config = {
        "device_name": "mt-rainier-weather-01",
        "device_type": "weather_station",
        "location": {
            "latitude": 46.8523,
            "longitude": -121.7603,
            "altitude": 4392.0,
            "region": "us-west-1",
            "site_name": "Mount Rainier - Paradise",
            "timezone": "America/Los_Angeles",
        },
        "capabilities": {
            "cpu_cores": 2,
            "memory_gb": 4.0,
            "storage_gb": 100.0,
            "gpu_available": False,
            "network_bandwidth_mbps": 25.0,
            "power_source": "battery",
            "operating_system": "Linux",
            "python_version": "3.11",
            "docker_available": True,
            "kubernetes_available": False,
        },
        "ip_address": "192.168.100.20",
        "port": 8080,
        "ssl_enabled": False,
        "firmware_version": "1.8.0",
        "software_version": "1.2.1",
    }

    async with EdgeDeviceClient() as client:
        try:
            # Register device
            print("📝 Registering weather station...")
            registration_result = await client.register_device(device_config)
            print(f"✅ Device registered: {registration_result['device_id']}")

            # Simulate weather monitoring
            print("\n🌡️ Starting weather monitoring...")
            for i in range(5):
                await asyncio.sleep(1)

                # Simulate weather data collection
                weather_metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "device_id": client.device_id,
                    "cpu_usage_percent": 15.0 + (i * 2),
                    "memory_usage_percent": 30.0 + (i * 3),
                    "storage_usage_percent": 25.0,
                    "temperature_celsius": -5.0 + (i * 2),  # Cold mountain weather
                    "network_latency_ms": 80.0,  # Higher latency due to remote location
                    "bandwidth_usage_mbps": 2.0,
                    "battery_level_percent": 85.0 - (i * 2),  # Battery draining
                    "power_consumption_watts": 25.0,
                    "active_models": 1,
                    "data_processing_rate": 50.0,
                    "error_count": 0,
                }

                await client.send_heartbeat(weather_metrics)
                print(
                    f"🌨️ Weather reading {i+1}: Temp {weather_metrics['temperature_celsius']:.1f}°C, "
                    f"Battery {weather_metrics['battery_level_percent']:.0f}%"
                )

            print("\n🎉 Weather station simulation completed!")

        except Exception as e:
            print(f"❌ Error during weather simulation: {e}")

async def list_all_devices():
    """List all registered devices"""
    print("📋 Listing all registered devices...")

    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8080/api/v1/devices/"

        async with session.get(url) as response:
            if response.status == 200:
                devices = await response.json()
                print(f"\n📊 Found {len(devices)} registered devices:")

                for device in devices:
                    print(f"  🔹 {device['device_name']} ({device['device_id']})")
                    print(f"     Type: {device['device_type']}")
                    print(f"     Status: {device['status']}")
                    print(f"     Location: {device['location']['site_name']}")
                    print(f"     Last seen: {device.get('last_seen', 'Never')}")
                    print()
            else:
                error_text = await response.text()
                print(f"❌ Failed to list devices: {response.status} - {error_text}")

async def main():
    """Main demonstration function"""
    print("🚀 Edge Device Manager Demo")
    print("=" * 50)

    try:
        # Run simulations concurrently
        await asyncio.gather(simulate_seismic_station(), simulate_weather_station())

        # Wait a moment then list all devices
        await asyncio.sleep(2)
        await list_all_devices()

    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
