#!/usr/bin/env python3
"""
Test script for submarine and ice-covered volcano monitoring functionality
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

try:
    # Try direct import first
    from advanced_thermal_analysis_system import AdvancedThermalAnalysisSystem
except ImportError:
    try:
        # Try from core module
        from core.advanced_thermal_analysis_system import AdvancedThermalAnalysisSystem
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure the advanced_thermal_analysis_system.py file exists")
        sys.exit(1)

async def test_hydroacoustic_data_integration():
    """Test hydroacoustic data integration for submarine volcano detection"""
    print("Testing hydroacoustic data integration...")
    
    # Initialize the thermal analysis system
    thermal_system = AdvancedThermalAnalysisSystem()
    
    # Sample hydroacoustic data
    hydroacoustic_data = [
        {
            'frequency_hz': 15.0,
            'amplitude_db': 125.0,
            'timestamp': datetime.utcnow().isoformat(),
            'depth_m': 2000.0
        },
        {
            'frequency_hz': 8.0,
            'amplitude_db': 110.0,
            'timestamp': (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
            'depth_m': 2000.0
        },
        {
            'frequency_hz': 25.0,
            'amplitude_db': 95.0,
            'timestamp': (datetime.utcnow() + timedelta(minutes=10)).isoformat(),
            'depth_m': 2000.0
        }
    ]
    
    # Test location (submarine volcano in Pacific)
    location = (40.0, -130.0)  # Latitude, Longitude
    
    try:
        results = await thermal_system.integrate_hydroacoustic_data(hydroacoustic_data, location)
        
        print(f"✓ Processed {len(results['processed_signals'])} hydroacoustic signals")
        print(f"✓ Detection confidence: {results['detection_confidence']:.2f}")
        print(f"✓ Temporal analysis pattern: {results['temporal_analysis']['pattern_type']}")
        print(f"✓ Recommendations: {len(results['recommended_actions'])} actions")
        
        return True
        
    except Exception as e:
        print(f"✗ Hydroacoustic data integration failed: {e}")
        return False

async def test_ice_volcano_interactions():
    """Test ice-volcano interaction modeling"""
    print("\nTesting ice-volcano interaction modeling...")
    
    thermal_system = AdvancedThermalAnalysisSystem()
    
    # Test parameters for subglacial volcano
    volcano_location = (64.0, -17.0)  # Iceland coordinates
    ice_thickness_m = 200.0  # 200m thick ice
    magma_temperature_k = 1373.15  # 1100°C
    eruption_rate_m3_s = 10.0  # 10 m³/s eruption rate
    
    try:
        results = await thermal_system.model_ice_volcano_interactions(
            volcano_location, ice_thickness_m, magma_temperature_k, eruption_rate_m3_s
        )
        
        print(f"✓ Ice melting rate: {results['ice_melting_dynamics']['ice_melting_rate_m3_s']:.2f} m³/s")
        print(f"✓ Time to breakthrough: {results['ice_melting_dynamics']['time_to_breakthrough_hours']:.1f} hours")
        print(f"✓ Flood likely: {results['glacial_outburst_flood_prediction']['flood_likely']}")
        print(f"✓ Peak discharge: {results['glacial_outburst_flood_prediction']['peak_discharge_m3_s']:.1f} m³/s")
        print(f"✓ Hazard count: {len(results['hazard_assessment'])} hazards identified")
        
        return True
        
    except Exception as e:
        print(f"✗ Ice-volcano interaction modeling failed: {e}")
        return False

async def test_tsunami_risk_assessment():
    """Test tsunami risk assessment from submarine eruption"""
    print("\nTesting tsunami risk assessment...")
    
    thermal_system = AdvancedThermalAnalysisSystem()
    
    # Test parameters for submarine eruption
    eruption_location = (35.0, 139.0)  # Near Japan
    eruption_magnitude = 4.5  # VEI 4.5
    water_depth_m = 500.0  # 500m water depth
    eruption_volume_m3 = 1e8  # 100 million m³
    
    try:
        results = await thermal_system.assess_tsunami_risk_from_submarine_eruption(
            eruption_location, eruption_magnitude, water_depth_m, eruption_volume_m3
        )
        
        tsunami_gen = results['tsunami_generation_potential']
        wave_prop = results['wave_propagation_model']
        
        print(f"✓ Tsunami likely: {tsunami_gen['tsunami_likely']}")
        print(f"✓ Maximum wave height: {tsunami_gen['maximum_wave_height_m']:.1f} m")
        print(f"✓ Dominant mechanism: {tsunami_gen['dominant_mechanism']}")
        print(f"✓ Affected coastlines: {len(wave_prop['affected_coastlines'])}")
        print(f"✓ Overall risk level: {results['overall_tsunami_risk_level']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Tsunami risk assessment failed: {e}")
        return False

async def test_3d_seafloor_visualization():
    """Test 3D seafloor volcanic feature visualization"""
    print("\nTesting 3D seafloor visualization...")
    
    thermal_system = AdvancedThermalAnalysisSystem()
    
    # Test region bounds (min_lat, max_lat, min_lon, max_lon)
    region_bounds = (35.0, 37.0, 138.0, 140.0)  # Tokyo Bay area
    
    # Sample volcanic features
    volcanic_features = [
        {
            'type': 'seamount',
            'location': (36.0, 139.0),
            'depth_m': -1500.0,
            'height_m': 800.0,
            'base_radius_m': 3000.0
        },
        {
            'type': 'hydrothermal_vent',
            'location': (36.2, 139.2),
            'depth_m': -2000.0,
            'vent_height_m': 15.0,
            'vent_radius_m': 3.0,
            'temperature_k': 573.15
        },
        {
            'type': 'volcanic_cone',
            'location': (36.5, 139.5),
            'depth_m': -800.0,
            'height_m': 400.0,
            'base_radius_m': 1500.0,
            'crater_radius_m': 150.0
        }
    ]
    
    try:
        results = await thermal_system.create_3d_seafloor_volcanic_visualization(
            region_bounds, volcanic_features
        )
        
        metadata = results['metadata']
        
        print(f"✓ Grid resolution: {metadata['grid_resolution']:.4f} degrees")
        print(f"✓ Feature count: {metadata['feature_count']}")
        print(f"✓ Depth range: {metadata['depth_range_m'][0]:.0f} to {metadata['depth_range_m'][1]:.0f} m")
        print(f"✓ Mesh vertices: {len(results['mesh_data']['vertices']) if 'vertices' in results.get('mesh_data', {}) else 'N/A'}")
        print(f"✓ Interactive elements: {len(results['interactive_elements'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ 3D seafloor visualization failed: {e}")
        return False

async def main():
    """Run all submarine and ice-covered volcano monitoring tests"""
    print("=" * 60)
    print("SUBMARINE AND ICE-COVERED VOLCANO MONITORING TESTS")
    print("=" * 60)
    
    tests = [
        test_hydroacoustic_data_integration,
        test_ice_volcano_interactions,
        test_tsunami_risk_assessment,
        test_3d_seafloor_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All submarine and ice-covered volcano monitoring tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)