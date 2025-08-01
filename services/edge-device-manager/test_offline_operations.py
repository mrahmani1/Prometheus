#!/usr/bin/env python3
"""
Test script for offline operations functionality
"""

import asyncio
import logging
from datetime import datetime

from core.offline_operations import OfflineOperationsCoordinator
from models.config import EdgeDeviceConfig
from models.device import DeviceCapabilities, DeviceLocation

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def test_offline_operations():
    """Test offline operations functionality"""

    # Create test configuration
    config = EdgeDeviceConfig(
        device_id="test-edge-001",
        device_name="Test Edge Device",
        device_type="seismic_station",
        location=DeviceLocation(
            latitude=45.0, longitude=-122.0, region="test-region", site_name="Test Site"
        ),
        capabilities=DeviceCapabilities(
            cpu_cores=4,
            memory_gb=8.0,
            storage_gb=100.0,
            gpu_available=False,
            operating_system="Linux",
            python_version="3.11",
            docker_available=True,
        ),
        ip_address="192.168.1.100",
        auth_token="test-token-123",
        data_retention_days=7,
    )

    # Initialize offline operations coordinator
    coordinator = OfflineOperationsCoordinator(config, "./test_offline_data")

    try:
        # Start offline operations
        logger.info("Starting offline operations coordinator...")
        await coordinator.start()

        # Wait for initialization
        await asyncio.sleep(2)

        # Test 1: Ingest sensor data
        logger.info("Test 1: Ingesting sensor data...")

        # Seismic data
        seismic_data = {
            "magnitude": 3.2,
            "depth": 15.0,
            "duration": 45.0,
            "frequency_content": {"dominant_freq": 2.5, "bandwidth": 1.0},
        }

        record_id1 = await coordinator.ingest_sensor_data(
            data_type="seismic",
            data=seismic_data,
            location={"latitude": 45.0, "longitude": -122.0},
        )
        logger.info(f"Ingested seismic data: {record_id1}")

        # Thermal data
        thermal_data = {
            "temperature": 85.0,
            "baseline_temperature": 25.0,
            "temperature_anomaly": 60.0,
        }

        record_id2 = await coordinator.ingest_sensor_data(
            data_type="thermal",
            data=thermal_data,
            location={"latitude": 45.0, "longitude": -122.0},
        )
        logger.info(f"Ingested thermal data: {record_id2}")

        # Gas data
        gas_data = {"composition": {"CO2": 2000.0, "SO2": 150.0, "H2S": 75.0}}

        record_id3 = await coordinator.ingest_sensor_data(
            data_type="gas",
            data=gas_data,
            location={"latitude": 45.0, "longitude": -122.0},
        )
        logger.info(f"Ingested gas data: {record_id3}")

        # Wait for processing
        logger.info("Waiting for data processing...")
        await asyncio.sleep(15)

        # Test 2: Check operation status
        logger.info("Test 2: Checking operation status...")
        status = await coordinator.get_operation_status()
        logger.info(
            f"Operation status: offline={status.is_offline}, "
            f"processing_active={status.data_processing_active}, "
            f"alerts={status.active_alerts_count}"
        )

        # Test 3: Get active alerts
        logger.info("Test 3: Getting active alerts...")
        alerts = await coordinator.get_active_alerts()
        logger.info(f"Found {len(alerts)} active alerts")

        for alert in alerts:
            logger.info(
                f"Alert: {alert.title} - {alert.message} (severity: {alert.severity.value})"
            )

        # Test 4: Get comprehensive status
        logger.info("Test 4: Getting comprehensive status...")
        comprehensive_status = await coordinator.get_comprehensive_status()

        # Print key statistics
        if "storage" in comprehensive_status:
            storage_stats = comprehensive_status["storage"]
            logger.info(
                f"Storage: {storage_stats.get('total_records', 0)} records, "
                f"{storage_stats.get('usage_percentage', 0):.1f}% used"
            )

        if "data_processing" in comprehensive_status:
            processing_stats = comprehensive_status["data_processing"]
            logger.info(
                f"Processing: {processing_stats.get('total_processed', 0)} records processed"
            )

        if "alerts" in comprehensive_status:
            alert_stats = comprehensive_status["alerts"]
            logger.info(
                f"Alerts: {alert_stats.get('total_alerts', 0)} total, "
                f"{alert_stats.get('active_alerts_count', 0)} active"
            )

        # Test 5: Test alert acknowledgment
        if alerts:
            logger.info("Test 5: Acknowledging first alert...")
            first_alert = alerts[0]
            success = await coordinator.acknowledge_alert(first_alert.alert_id)
            logger.info(f"Alert acknowledgment: {'success' if success else 'failed'}")

        # Test 6: Test connectivity and sync status
        logger.info("Test 6: Checking connectivity and sync status...")
        connectivity_status = (
            await coordinator.connectivity_manager.get_connectivity_status()
        )
        logger.info(
            f"Connectivity: {connectivity_status['status']}, "
            f"score: {connectivity_status['connectivity_score']:.2f}"
        )

        sync_status = await coordinator.get_sync_status()
        logger.info(
            f"Sync: queue={sync_status['queue_size']}, "
            f"in_progress={sync_status['in_progress']}"
        )

        # Test 7: Force sync (will fail due to no actual cloud service)
        logger.info("Test 7: Testing force sync...")
        sync_result = await coordinator.force_data_sync()
        logger.info(f"Force sync result: {sync_result}")

        logger.info("All tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

    finally:
        # Stop offline operations
        logger.info("Stopping offline operations coordinator...")
        await coordinator.stop()

async def test_individual_components():
    """Test individual components separately"""

    logger.info("Testing individual components...")

    # Test configuration
    config = EdgeDeviceConfig(
        device_id="test-edge-002",
        device_name="Component Test Device",
        device_type="test_station",
        location=DeviceLocation(latitude=40.0, longitude=-120.0, region="test-region"),
        capabilities=DeviceCapabilities(
            cpu_cores=2,
            memory_gb=4.0,
            storage_gb=50.0,
            operating_system="Linux",
            python_version="3.11",
        ),
        ip_address="192.168.1.101",
        auth_token="test-token-456",
    )

    # Test offline processor

    processor = OfflineDataProcessor(config, "./test_processor")

    # Store some test data
    test_data = {"test": "data", "value": 123}

import torch.nn as nn
import torch.nn.functional as F

    data_payload = pickle.dumps(test_data)
    metadata = {"type": "test", "timestamp": datetime.utcnow().isoformat()}

    record_id = await processor.store_sensor_data("test", data_payload, metadata)
    logger.info(f"Stored test data: {record_id}")

    # Process data
    results = await processor.process_offline_data()
    logger.info(f"Processing results: {len(results)} items")

    # Get statistics
    stats = await processor.get_processing_statistics()
    logger.info(f"Processing stats: {stats}")

    # Cleanup
    await processor.shutdown()

    logger.info("Individual component tests completed!")

if __name__ == "__main__":
    print("Starting offline operations tests...")

    # Run main test
    asyncio.run(test_offline_operations())

    print("\nRunning individual component tests...")

    # Run component tests
    asyncio.run(test_individual_components())

    print("\nAll tests completed!")
