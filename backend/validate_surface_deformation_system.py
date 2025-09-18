#!/usr/bin/env python3
"""
Simple validation script for Surface Deformation Correlation System
Tests core functionality without external dependencies
"""

import sys
import os
import numpy as np
from datetime import datetime
import asyncio
from unittest.mock import AsyncMock, MagicMock

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def test_data_structures():
    """Test the data structure definitions"""
    print("Testing data structures...")
    
    try:
        from surface_deformation_correlation_system import (
            InSARDeformationData,
            GPSDeformationData,
            TiltStrainData,
            SillGeometryModel
        )
        
        # Test SillGeometryModel
        sill_model = SillGeometryModel(
            sill_id="test_sill",
            center_coordinates=(-122.5, 37.8, 2000),
            length=5000,
            width=3000,
            thickness=50,
            strike=45,
            dip=10,
            opening=0.5,
            pressure_change=1e6,
            elastic_moduli=(30e9, 0.25),
            host_rock_properties={'density': 2700},
            emplacement_time=datetime.now(),
            growth_rate=0.1,
            geometry_uncertainty={'depth': 200},
            validation_metrics={}
        )
        
        assert sill_model.sill_id == "test_sill"
        assert sill_model.length == 5000
        assert sill_model.center_coordinates[2] == 2000
        print("✓ SillGeometryModel creation successful")
        
        # Test InSARDeformationData
        deformation_map = np.random.normal(0, 0.01, (50, 50))
        uncertainty_map = np.full((50, 50), 0.005)
        coherence_map = np.random.uniform(0.3, 0.9, (50, 50))
        
        lon_grid = np.linspace(-122.6, -122.4, 50)
        lat_grid = np.linspace(37.7, 37.9, 50)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        insar_data = InSARDeformationData(
            acquisition_date=datetime.now(),
            satellite="Sentinel-1A",
            track=144,
            frame=350,
            look_angle=23.0,
            heading_angle=-13.3,
            wavelength=0.056,
            spatial_resolution=20.0,
            temporal_baseline=12.0,
            perpendicular_baseline=150.0,
            coherence_threshold=0.3,
            unwrapping_method="snaphu",
            atmospheric_correction="gacos",
            deformation_map=deformation_map,
            uncertainty_map=uncertainty_map,
            coherence_map=coherence_map,
            coordinates=(lon_mesh, lat_mesh),
            reference_point=(-122.5, 37.8),
            processing_parameters={'multilook': '4x1'},
            quality_metrics={'mean_coherence': 0.65}
        )
        
        assert insar_data.satellite == "Sentinel-1A"
        assert insar_data.deformation_map.shape == (50, 50)
        print("✓ InSARDeformationData creation successful")
        
        # Test GPSDeformationData
        gps_data = GPSDeformationData(
            station_id="GPS01",
            coordinates=(-122.5, 37.8, 100),
            measurement_date=datetime.now(),
            displacement_vector=(0.01, 0.02, 0.05),
            velocity_vector=(0.001, 0.002, 0.005),
            uncertainty_vector=(0.002, 0.002, 0.005),
            correlation_matrix=np.eye(3) * 0.1,
            measurement_type='continuous',
            processing_strategy='PPP',
            reference_frame='ITRF2014',
            antenna_type='AOAD/M_T',
            monument_type='concrete_pillar',
            data_quality={'rms': 0.003},
            environmental_corrections={'troposphere': 0.001}
        )
        
        assert gps_data.station_id == "GPS01"
        assert len(gps_data.displacement_vector) == 3
        print("✓ GPSDeformationData creation successful")
        
        # Test TiltStrainData
        tilt_data = TiltStrainData(
            instrument_id="TILT01",
            instrument_type="tiltmeter",
            coordinates=(-122.5, 37.8, 50),
            measurement_date=datetime.now(),
            tilt_components=(10.0, 5.0),
            strain_tensor=None,
            shear_strain=None,
            volumetric_strain=None,
            uncertainty_tilt=(1.0, 1.0),
            uncertainty_strain=None,
            temperature_correction=0.001,
            pressure_correction=0.0005,
            installation_depth=50,
            calibration_date=datetime.now(),
            drift_correction=0.0001,
            data_quality_flags=['good']
        )
        
        assert tilt_data.instrument_id == "TILT01"
        assert tilt_data.tilt_components[0] == 10.0
        print("✓ TiltStrainData creation successful")
        
        print("All data structures validated successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing data structures: {e}")
        return False

def test_correlation_calculations():
    """Test correlation calculation methods"""
    print("\nTesting correlation calculations...")
    
    try:
        # Mock the system to avoid database dependencies
        config = {
            'database': {},
            'processing': {'max_workers': 2},
            'validation': {'min_correlation': 0.3}
        }
        
        # Import with mocked dependencies
        import sys
        from unittest.mock import patch
        
        with patch.dict('sys.modules', {
            'asyncpg': MagicMock(),
            'redis.asyncio': MagicMock(),
            'influxdb_client.client.influxdb_client_async': MagicMock(),
            'neo4j': MagicMock()
        }):
            from surface_deformation_correlation_system import SurfaceDeformationCorrelationSystem
            
            system = SurfaceDeformationCorrelationSystem(config)
            
            # Test weighted correlation calculation
            observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            predicted = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
            weights = np.array([1.0, 2.0, 1.5, 1.0, 2.5])
            
            correlation = system._calculate_weighted_correlation(observed, predicted, weights)
            
            assert isinstance(correlation, float)
            assert -1 <= correlation <= 1
            print(f"✓ Weighted correlation calculation: {correlation:.3f}")
            
            # Test with perfect correlation
            perfect_correlation = system._calculate_weighted_correlation(
                observed, observed, weights
            )
            assert abs(perfect_correlation - 1.0) < 1e-10
            print("✓ Perfect correlation test passed")
            
            # Test correlation significance
            significance = system._calculate_correlation_significance(0.8, 100)
            assert 0 <= significance <= 1
            print(f"✓ Correlation significance calculation: {significance:.3f}")
            
            # Test parameter conversion
            params = np.array([-122.5, 37.8, 2000, 5000, 3000, 45, 10, 0.5])
            sill_model = system._params_to_sill_model(params, None)
            
            assert sill_model.center_coordinates[0] == params[0]
            assert sill_model.length == params[3]
            print("✓ Parameter conversion test passed")
            
        print("All correlation calculations validated successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing correlation calculations: {e}")
        return False

def test_forward_modeling_components():
    """Test forward modeling components"""
    print("\nTesting forward modeling components...")
    
    try:
        with patch.dict('sys.modules', {
            'asyncpg': MagicMock(),
            'redis.asyncio': MagicMock(),
            'influxdb_client.client.influxdb_client_async': MagicMock(),
            'neo4j': MagicMock()
        }):
            from surface_deformation_correlation_system import SurfaceDeformationCorrelationSystem
            
            config = {'database': {}, 'processing': {}, 'validation': {}}
            system = SurfaceDeformationCorrelationSystem(config)
            
            # Test Mogi model calculation
            source_x, source_y, source_depth = 0, 0, 2000
            volume_change = 1e6  # m³
            obs_coords = np.array([[1000, 1000], [2000, 0], [0, 2000]])
            poisson_ratio = 0.25
            
            # This would normally be async, but we'll test the calculation logic
            # by calling the method synchronously for validation
            
            # Test simple forward model (synchronous version)
            from surface_deformation_correlation_system import SillGeometryModel
            
            sill_model = SillGeometryModel(
                sill_id="test",
                center_coordinates=(0, 0, 2000),
                length=5000, width=3000, thickness=50,
                strike=0, dip=0, opening=1.0,
                pressure_change=1e6, elastic_moduli=(30e9, 0.25),
                host_rock_properties={}, emplacement_time=datetime.now(),
                growth_rate=0, geometry_uncertainty={}, validation_metrics={}
            )
            
            # Test simple forward model calculation
            predicted = system._simple_forward_model(sill_model, obs_coords)
            
            assert len(predicted) == len(obs_coords)
            assert np.all(np.isfinite(predicted))
            assert np.all(predicted >= 0)  # Uplift should be positive
            print(f"✓ Simple forward model: max displacement = {np.max(predicted)*1000:.1f} mm")
            
            # Test misfit calculation
            observed = predicted + np.random.normal(0, 0.01, len(predicted))
            obs_data = {
                'observations': observed,
                'uncertainties': np.full(len(observed), 0.01),
                'weights': np.full(len(observed), 100)
            }
            
            params = np.array([0, 0, 2000, 5000, 3000, 0, 0, 1.0])
            misfit = system._calculate_misfit(params, obs_data, obs_coords, {})
            
            assert misfit >= 0
            print(f"✓ Misfit calculation: {misfit:.6f}")
            
        print("Forward modeling components validated successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing forward modeling: {e}")
        return False

async def test_async_functionality():
    """Test async functionality with mocked dependencies"""
    print("\nTesting async functionality...")
    
    try:
        with patch.dict('sys.modules', {
            'asyncpg': MagicMock(),
            'redis.asyncio': MagicMock(),
            'influxdb_client.client.influxdb_client_async': MagicMock(),
            'neo4j': MagicMock()
        }):
            from surface_deformation_correlation_system import (
                SurfaceDeformationCorrelationSystem,
                SillGeometryModel
            )
            
            config = {'database': {}, 'processing': {}, 'validation': {}}
            system = SurfaceDeformationCorrelationSystem(config)
            
            # Mock database connections
            system.postgres_pool = AsyncMock()
            system.redis_client = AsyncMock()
            system.influx_client = AsyncMock()
            system.neo4j_driver = AsyncMock()
            
            # Test connection initialization
            await system.initialize_connections()
            print("✓ Connection initialization test passed")
            
            # Test connection cleanup
            await system.close_connections()
            print("✓ Connection cleanup test passed")
            
        print("Async functionality validated successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing async functionality: {e}")
        return False

def main():
    """Run all validation tests"""
    print("Surface Deformation Correlation System Validation")
    print("=" * 55)
    
    tests = [
        test_data_structures,
        test_correlation_calculations,
        test_forward_modeling_components
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    # Test async functionality
    try:
        async_result = asyncio.run(test_async_functionality())
        results.append(async_result)
    except Exception as e:
        print(f"✗ Async test failed: {e}")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 55)
    print("VALIDATION SUMMARY")
    print("=" * 55)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All validation tests PASSED!")
        print("\nThe Surface Deformation Correlation System is ready for use.")
        print("\nKey features validated:")
        print("  • Data structure definitions")
        print("  • Correlation calculations")
        print("  • Forward modeling components")
        print("  • Async functionality")
        print("  • Database connection management")
    else:
        print("✗ Some validation tests FAILED!")
        print("Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)