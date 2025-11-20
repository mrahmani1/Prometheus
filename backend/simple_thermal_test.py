#!/usr/bin/env python3
"""
Simple test for Advanced Thermal Analysis System classes and enums
"""

import sys
import os
from datetime import datetime
import numpy as np

# Test basic imports
try:
    # Test enum imports
    from core.advanced_thermal_analysis_system import ThermalDataSource, ThermalAnomalyType, HazardType
    print("✓ Enums imported successfully")
    
    # Test data class imports
    from core.advanced_thermal_analysis_system import ThermalObservation, ThermalAnomaly, SubsurfaceHeatModel, ThermalHazardAssessment
    print("✓ Data classes imported successfully")
    
    # Test main class import
    from core.advanced_thermal_analysis_system import AdvancedThermalAnalysisSystem
    print("✓ Main class imported successfully")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_enums():
    """Test enum functionality"""
    print("\n=== Testing Enums ===")
    
    # Test ThermalDataSource
    assert ThermalDataSource.MODIS_TERRA.value == "modis_terra"
    assert ThermalDataSource.VIIRS_NPP.value == "viirs_npp"
    assert ThermalDataSource.LANDSAT_8.value == "landsat_8"
    print("✓ ThermalDataSource enum working")
    
    # Test ThermalAnomalyType
    assert ThermalAnomalyType.VOLCANIC_HOTSPOT.value == "volcanic_hotspot"
    assert ThermalAnomalyType.LAVA_FLOW.value == "lava_flow"
    assert ThermalAnomalyType.FUMAROLE_FIELD.value == "fumarole_field"
    print("✓ ThermalAnomalyType enum working")
    
    # Test HazardType
    assert HazardType.LAVA_FLOW_HAZARD.value == "lava_flow_hazard"
    assert HazardType.PYROCLASTIC_FLOW_HAZARD.value == "pyroclastic_flow_hazard"
    print("✓ HazardType enum working")

def test_data_classes():
    """Test data class functionality"""
    print("\n=== Testing Data Classes ===")
    
    # Test ThermalObservation
    obs = ThermalObservation(
        observation_id="test_001",
        timestamp=datetime.utcnow(),
        latitude=45.0,
        longitude=-120.0,
        elevation=1500.0,
        temperature_k=350.0,
        radiance=15.5,
        emissivity=0.95,
        pixel_size_m=30.0,
        data_source=ThermalDataSource.MODIS_TERRA,
        satellite_zenith_angle=15.0,
        atmospheric_correction=True,
        uncertainty_k=2.5
    )
    assert obs.observation_id == "test_001"
    assert obs.temperature_k == 350.0
    assert obs.data_source == ThermalDataSource.MODIS_TERRA
    print("✓ ThermalObservation data class working")
    
    # Test ThermalAnomaly
    anomaly = ThermalAnomaly(
        anomaly_id="anomaly_001",
        detection_timestamp=datetime.utcnow(),
        centroid_lat=45.0,
        centroid_lon=-120.0,
        area_m2=10000.0,
        max_temperature_k=450.0,
        mean_temperature_k=420.0,
        temperature_excess_k=50.0,
        radiant_flux_w=1000000.0,
        anomaly_type=ThermalAnomalyType.VOLCANIC_HOTSPOT,
        confidence_score=0.85,
        spatial_extent={"area_m2": 10000.0}
    )
    assert anomaly.anomaly_id == "anomaly_001"
    assert anomaly.max_temperature_k == 450.0
    assert anomaly.anomaly_type == ThermalAnomalyType.VOLCANIC_HOTSPOT
    print("✓ ThermalAnomaly data class working")

def test_system_initialization():
    """Test system initialization"""
    print("\n=== Testing System Initialization ===")
    
    config = {
        'thermal_detection': {
            'background_window_days': 30,
            'anomaly_threshold_std': 3.0,
            'minimum_area_m2': 900.0
        },
        'subsurface_modeling': {
            'grid_resolution_m': 100.0,
            'max_depth_m': 5000.0
        },
        'data_quality': {
            'temperature_range_k': (200.0, 2000.0),
            'uncertainty_threshold_k': 10.0
        }
    }
    
    try:
        thermal_system = AdvancedThermalAnalysisSystem(config)
        print("✓ System initialized successfully")
        
        # Test configuration access
        assert thermal_system.config['thermal_detection']['background_window_days'] == 30
        assert thermal_system.config['subsurface_modeling']['grid_resolution_m'] == 100.0
        print("✓ Configuration loaded correctly")
        
        # Test method existence
        methods_to_check = [
            'integrate_satellite_thermal_data',
            'detect_thermal_anomalies',
            'create_subsurface_heat_model',
            'assess_thermal_hazards',
            'create_multispectral_thermal_visualization',
            'validate_thermal_data_quality',
            'integrate_monitoring_networks'
        ]
        
        for method_name in methods_to_check:
            if hasattr(thermal_system, method_name):
                print(f"✓ Method {method_name} exists")
            else:
                print(f"✗ Method {method_name} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ System initialization failed: {e}")
        return False

def main():
    """Main test function"""
    print("Advanced Thermal Analysis System - Simple Test Suite")
    print("=" * 60)
    
    try:
        # Test enums
        test_enums()
        
        # Test data classes
        test_data_classes()
        
        # Test system initialization
        success = test_system_initialization()
        
        if success:
            print("\n🎉 All basic tests passed! The Advanced Thermal Analysis System structure is correct.")
            return 0
        else:
            print("\n❌ Some tests failed.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)