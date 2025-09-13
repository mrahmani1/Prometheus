"""
Validation script for Monitoring Network Registry

This script validates the core functionality of the monitoring network registry
without requiring all optional dependencies.
"""

import sys
import os
import asyncio
from datetime import datetime
from typing import Dict, Any

# Add modules path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def test_imports():
    """Test that core classes can be imported"""
    try:
        from monitoring_network_registry import (
            MonitoringNetworkType,
            InstrumentType,
            DataStreamStatus,
            NetworkStatus,
            GeographicBounds,
            InstrumentSpecification,
            DataStreamInfo,
            NetworkCoverageMetrics,
            PerformanceMetrics
        )
        print("✓ Core classes imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_data_structures():
    """Test data structure creation"""
    try:
        from monitoring_network_registry import (
            GeographicBounds,
            NetworkCoverageMetrics,
            PerformanceMetrics,
            MonitoringNetworkType
        )
        
        # Test GeographicBounds
        bounds = GeographicBounds(44.0, 45.0, -121.0, -120.0)
        assert bounds.contains_point(44.5, -120.5) == True
        assert bounds.contains_point(50.0, -130.0) == False
        print("✓ GeographicBounds working correctly")
        
        # Test NetworkCoverageMetrics
        coverage = NetworkCoverageMetrics(
            total_area_km2=1000.0,
            monitored_area_km2=800.0,
            coverage_percentage=80.0,
            station_density=0.05,
            average_inter_station_distance=25.0,
            coverage_gaps=[],
            redundancy_score=0.7,
            temporal_coverage=95.0
        )
        assert coverage.coverage_percentage == 80.0
        print("✓ NetworkCoverageMetrics working correctly")
        
        # Test PerformanceMetrics
        performance = PerformanceMetrics(
            uptime_percentage=98.5,
            data_completeness=95.2,
            latency_p95=150.0,
            error_rate=0.02,
            false_alarm_rate=0.01,
            detection_capability=0.92,
            response_time=120.0,
            maintenance_frequency=4.0
        )
        assert performance.uptime_percentage == 98.5
        print("✓ PerformanceMetrics working correctly")
        
        # Test Enums
        assert MonitoringNetworkType.SEISMIC.value == "seismic"
        assert MonitoringNetworkType.GPS_GNSS.value == "gps_gnss"
        print("✓ Enums working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Data structure test failed: {e}")
        return False

def test_registry_initialization():
    """Test registry initialization with minimal configuration"""
    try:
        from monitoring_network_registry import MonitoringNetworkRegistry
        
        # Test with SQLite (no external dependencies)
        registry = MonitoringNetworkRegistry(
            database_url="sqlite:///:memory:",
            redis_url=None,
            kafka_config=None
        )
        
        print("✓ Registry initialization successful")
        return True
        
    except Exception as e:
        print(f"✗ Registry initialization failed: {e}")
        return False

async def test_basic_operations():
    """Test basic registry operations"""
    try:
        from monitoring_network_registry import MonitoringNetworkRegistry
        
        # Initialize registry
        registry = MonitoringNetworkRegistry(
            database_url="sqlite:///:memory:",
            redis_url=None,
            kafka_config=None
        )
        
        # Test observatory registration
        observatory_data = {
            'name': 'Test Observatory',
            'country': 'Test Country',
            'latitude': 45.0,
            'longitude': -120.0
        }
        
        obs_id = await registry.register_volcanic_observatory(observatory_data)
        assert obs_id is not None
        print("✓ Observatory registration successful")
        
        # Test observatory query
        observatories = await registry.query_observatories()
        assert len(observatories) == 1
        assert observatories[0]['name'] == 'Test Observatory'
        print("✓ Observatory query successful")
        
        # Test station registration
        station_data = {
            'station_code': 'TEST.STA1',
            'name': 'Test Station',
            'observatory_id': obs_id,
            'latitude': 45.1,
            'longitude': -119.9,
            'elevation': 1000,
            'installation_date': datetime(2020, 1, 1),
            'station_type': 'seismic'
        }
        
        station_id = await registry.register_monitoring_station(station_data)
        assert station_id is not None
        print("✓ Station registration successful")
        
        # Test station query
        stations = await registry.query_monitoring_stations()
        assert len(stations) == 1
        assert stations[0]['station_code'] == 'TEST.STA1'
        print("✓ Station query successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic operations test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("Monitoring Network Registry Validation")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Structures Test", test_data_structures),
        ("Registry Initialization Test", test_registry_initialization),
        ("Basic Operations Test", lambda: asyncio.run(test_basic_operations()))
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print(f"\n" + "=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Monitoring Network Registry is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)