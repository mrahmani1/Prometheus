"""
Comprehensive unit tests for PlateEvolutionEngine module

Tests all 11 recently implemented advanced helper methods and core functionality.
"""

import pytest
import numpy as np
import h5py
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'modules'))

# Mock missing dependencies
import unittest.mock
sys.modules['cartopy'] = unittest.mock.MagicMock()
sys.modules['cartopy.crs'] = unittest.mock.MagicMock()
sys.modules['cartopy.feature'] = unittest.mock.MagicMock()
sys.modules['cupy'] = unittest.mock.MagicMock()
sys.modules['cupyx'] = unittest.mock.MagicMock()
sys.modules['cupyx.scipy'] = unittest.mock.MagicMock()
sys.modules['cupyx.scipy.sparse'] = unittest.mock.MagicMock()

try:
    from plate_evolution_engine import (
        PlateEvolutionEngine, PlateType, BoundaryType, SubductionType, PlumeType,
        PlateProperties
    )
    IMPORT_SUCCESS = True
except Exception as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


class TestPlateEvolutionEngineHelpers:
    """Test class for the 11 advanced helper methods"""
    
    def setup_method(self):
        """Skip tests if module import failed"""
        if not IMPORT_SUCCESS:
            pytest.skip(f"Could not import plate_evolution_engine: {IMPORT_ERROR}")
    
    @pytest.fixture
    def engine(self):
        """Create PlateEvolutionEngine instance for testing"""
        config = {
            'grid_resolution': (100, 100),
            'time_step': 1000,  # years
            'max_time': 100000,  # years
            'enable_gpu': False,  # Disable GPU for testing
            'boundary_conditions': 'periodic',
            'rheology_model': 'viscoelastic'
        }
        return PlateEvolutionEngine(config)
    
    @pytest.fixture
    def sample_reconstruction_data(self):
        """Create sample plate reconstruction data"""
        return {
            'time': 50.0,  # Ma
            'longitude': np.linspace(-180, 180, 100),
            'latitude': np.linspace(-90, 90, 50),
            'velocity_x': np.random.uniform(-5, 5, (50, 100)),
            'velocity_y': np.random.uniform(-5, 5, (50, 100)),
            'age': np.random.uniform(0, 200, (50, 100)),
            'topography': np.random.uniform(-5000, 3000, (50, 100)),
            'bathymetry': np.random.uniform(-6000, 0, (50, 100)),
            'crustal_thickness': np.random.uniform(5, 50, (50, 100)),
            'plate_ids': np.random.randint(1, 20, (50, 100))
        }
    
    @pytest.fixture
    def temp_hdf5_file(self):
        """Create temporary HDF5 file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            yield f.name
        # Cleanup
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass

    def test_extract_spatial_data_from_reconstruction(self, engine, sample_reconstruction_data):
        """Test _extract_spatial_data_from_reconstruction method"""
        result = engine._extract_spatial_data_from_reconstruction(
            sample_reconstruction_data, 
            target_time=50.0
        )
        
        # Verify structure
        assert isinstance(result, dict)
        expected_keys = [
            'velocity_field', 'age_field', 'topographic_field', 
            'geological_features', 'spatial_metrics', 'quality_metrics'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify velocity field components
        velocity_field = result['velocity_field']
        assert 'divergence' in velocity_field
        assert 'vorticity' in velocity_field
        assert 'magnitude' in velocity_field
        
        # Verify data shapes
        expected_shape = sample_reconstruction_data['velocity_x'].shape
        assert velocity_field['divergence'].shape == expected_shape
        assert velocity_field['vorticity'].shape == expected_shape
        
        # Verify geological features
        features = result['geological_features']
        assert 'ridges' in features
        assert 'trenches' in features
        assert 'transforms' in features
    
    def test_calculate_velocity_divergence(self, engine):
        """Test _calculate_velocity_divergence method"""
        # Create test velocity field
        vx = np.random.uniform(-5, 5, (50, 100))
        vy = np.random.uniform(-5, 5, (50, 100))
        lon = np.linspace(-180, 180, 100)
        lat = np.linspace(-90, 90, 50)
        
        result = engine._calculate_velocity_divergence(vx, vy, lon, lat)
        
        # Verify structure
        assert isinstance(result, dict)
        expected_keys = [
            'divergence', 'finite_diff_metrics', 'spherical_corrections',
            'boundary_conditions', 'quality_assessment'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify divergence field shape
        assert result['divergence'].shape == vx.shape
        
        # Verify spherical corrections were applied
        corrections = result['spherical_corrections']
        assert 'latitude_factor' in corrections
        assert 'longitude_factor' in corrections
    
    def test_calculate_velocity_vorticity(self, engine):
        """Test _calculate_velocity_vorticity method"""
        # Create test velocity field
        vx = np.random.uniform(-5, 5, (50, 100))
        vy = np.random.uniform(-5, 5, (50, 100))
        lon = np.linspace(-180, 180, 100)
        lat = np.linspace(-90, 90, 50)
        
        result = engine._calculate_velocity_vorticity(vx, vy, lon, lat)
        
        # Verify structure
        assert isinstance(result, dict)
        expected_keys = [
            'vorticity', 'curl_components', 'rotation_analysis',
            'plate_boundaries', 'deformation_metrics'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify vorticity field shape
        assert result['vorticity'].shape == vx.shape
        
        # Verify rotation analysis
        rotation = result['rotation_analysis']
        assert 'angular_velocity' in rotation
        assert 'rotation_poles' in rotation
    
    def test_calculate_age_gradients(self, engine):
        """Test _calculate_age_gradients method"""
        # Create test age field
        age_field = np.random.uniform(0, 200, (50, 100))
        lon = np.linspace(-180, 180, 100)
        lat = np.linspace(-90, 90, 50)
        
        result = engine._calculate_age_gradients(age_field, lon, lat)
        
        # Verify structure
        assert isinstance(result, dict)
        expected_keys = [
            'age_gradients', 'spreading_analysis', 'uncertainty_propagation',
            'gradient_magnitude', 'seafloor_fabric'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify gradient shapes
        gradients = result['age_gradients']
        assert gradients['d_age_d_lon'].shape == age_field.shape
        assert gradients['d_age_d_lat'].shape == age_field.shape
        
        # Verify spreading analysis
        spreading = result['spreading_analysis']
        assert 'half_spreading_rates' in spreading
        assert 'ridge_orientations' in spreading
    
    def test_calculate_dynamic_topography(self, engine):
        """Test _calculate_dynamic_topography method"""
        # Create test density field
        density_anomaly = np.random.uniform(-100, 100, (20, 50, 100))  # 3D field
        coordinates = {
            'longitude': np.linspace(-180, 180, 100),
            'latitude': np.linspace(-90, 90, 50),
            'depth': np.linspace(0, 2900, 20)
        }
        
        result = engine._calculate_dynamic_topography(density_anomaly, coordinates)
        
        # Verify structure
        assert isinstance(result, dict)
        expected_keys = [
            'dynamic_topography', 'mantle_flow', 'density_integration',
            'isostatic_response', 'surface_deformation'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify topography field shape (should be 2D)
        topo = result['dynamic_topography']
        assert topo.shape == (50, 100)  # lat x lon
        
        # Verify mantle flow components
        flow = result['mantle_flow']
        assert 'velocity_field' in flow
        assert 'streamlines' in flow
    
    def test_calculate_isostatic_correction(self, engine):
        """Test _calculate_isostatic_correction method"""
        # Create test topography and crustal properties
        topography = np.random.uniform(-3000, 3000, (50, 100))
        crustal_thickness = np.random.uniform(5, 50, (50, 100))
        crustal_density = np.full((50, 100), 2800.0)
        
        result = engine._calculate_isostatic_correction(
            topography, crustal_thickness, crustal_density
        )
        
        # Verify structure
        assert isinstance(result, dict)
        expected_keys = [
            'isostatic_correction', 'flexural_response', 'local_compensation',
            'effective_elastic_thickness', 'compensation_analysis'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify correction field shape
        correction = result['isostatic_correction']
        assert correction.shape == topography.shape
        
        # Verify flexural response
        flexural = result['flexural_response']
        assert 'deflection' in flexural
        assert 'bending_moment' in flexural
    
    def test_classify_deformation_style(self, engine):
        """Test _classify_deformation_style method"""
        # Create test velocity field and stress tensors
        velocity_field = {
            'vx': np.random.uniform(-5, 5, (50, 100)),
            'vy': np.random.uniform(-5, 5, (50, 100))
        }
        stress_tensors = np.random.uniform(-1e8, 1e8, (50, 100, 2, 2))
        
        result = engine._classify_deformation_style(velocity_field, stress_tensors)
        
        # Verify structure
        assert isinstance(result, dict)
        expected_keys = [
            'deformation_classification', 'strain_analysis', 'stress_analysis',
            'rheological_regime', 'deformation_mechanisms'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify classification field
        classification = result['deformation_classification']
        assert classification.shape == (50, 100)
        
        # Verify strain analysis components
        strain = result['strain_analysis']
        assert 'strain_rate_tensor' in strain
        assert 'principal_strains' in strain
    
    def test_calculate_magmatic_flux(self, engine):
        """Test _calculate_magmatic_flux method"""
        # Create test temperature and pressure fields
        temperature = np.random.uniform(1000, 1600, (20, 50, 100))
        pressure = np.random.uniform(0, 5e9, (20, 50, 100))
        composition = {
            'SiO2': np.random.uniform(45, 75, (20, 50, 100)),
            'MgO': np.random.uniform(3, 15, (20, 50, 100))
        }
        
        result = engine._calculate_magmatic_flux(temperature, pressure, composition)
        
        # Verify structure
        assert isinstance(result, dict)
        expected_keys = [
            'melt_fraction', 'melt_productivity', 'magmatic_flux',
            'volcanic_output', 'melt_transport'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify melt fraction shape
        melt_fraction = result['melt_fraction']
        assert melt_fraction.shape == temperature.shape
        
        # Verify flux is positive or zero
        flux = result['magmatic_flux']
        assert np.all(flux >= 0)
    
    def test_calculate_interpolation_quality(self, engine):
        """Test _calculate_interpolation_quality method"""
        # Create test data grid and interpolated values
        data_points = np.array([
            [-120, 40, 2.5],  # lon, lat, value
            [-119, 41, 2.8],
            [-118, 42, 3.1]
        ])
        grid_coordinates = {
            'longitude': np.linspace(-125, -115, 50),
            'latitude': np.linspace(38, 44, 30)
        }
        interpolated_values = np.random.uniform(2.0, 3.5, (30, 50))
        
        result = engine._calculate_interpolation_quality(
            data_points, grid_coordinates, interpolated_values
        )
        
        # Verify structure
        assert isinstance(result, dict)
        expected_keys = [
            'quality_metrics', 'spatial_resolution', 'uncertainty_map',
            'confidence_intervals', 'validation_statistics'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify uncertainty map shape
        uncertainty = result['uncertainty_map']
        assert uncertainty.shape == interpolated_values.shape
        
        # Verify quality metrics are valid
        metrics = result['quality_metrics']
        assert 'mean_error' in metrics
        assert 'rmse' in metrics
        assert metrics['rmse'] >= 0
    
    def test_save_fallback_spatial_grids(self, engine, temp_hdf5_file):
        """Test _save_fallback_spatial_grids method"""
        # Create test spatial grids
        spatial_grids = {
            'velocity_x': np.random.uniform(-5, 5, (50, 100)),
            'velocity_y': np.random.uniform(-5, 5, (50, 100)),
            'temperature': np.random.uniform(273, 1600, (50, 100)),
            'pressure': np.random.uniform(0, 1e9, (50, 100))
        }
        metadata = {
            'time_step': 1000,
            'resolution': [1.0, 1.0],
            'units': {'velocity': 'cm/yr', 'temperature': 'K'}
        }
        
        result = engine._save_fallback_spatial_grids(
            spatial_grids, temp_hdf5_file, metadata
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        expected_keys = [
            'save_status', 'file_info', 'compression_stats',
            'dataset_hierarchy', 'optimization_metrics'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify file was created and contains data
        assert os.path.exists(temp_hdf5_file)
        
        with h5py.File(temp_hdf5_file, 'r') as f:
            assert 'spatial_grids' in f
            grids_group = f['spatial_grids']
            for grid_name in spatial_grids.keys():
                assert grid_name in grids_group
                saved_data = grids_group[grid_name][:]
                np.testing.assert_array_equal(saved_data, spatial_grids[grid_name])
    
    def test_save_fallback_uncertainty_estimates(self, engine, temp_hdf5_file):
        """Test _save_fallback_uncertainty_estimates method"""
        # Create test uncertainty data
        uncertainty_data = {
            'velocity_uncertainty': np.random.uniform(0, 2, (50, 100)),
            'temperature_uncertainty': np.random.uniform(0, 50, (50, 100)),
            'model_variance': np.random.uniform(0, 1, (50, 100))
        }
        confidence_intervals = {
            'lower_bound': np.random.uniform(0, 5, (50, 100)),
            'upper_bound': np.random.uniform(5, 10, (50, 100))
        }
        
        result = engine._save_fallback_uncertainty_estimates(
            uncertainty_data, confidence_intervals, temp_hdf5_file
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        expected_keys = [
            'save_status', 'statistical_summary', 'monte_carlo_results',
            'confidence_analysis', 'validation_metrics'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Verify file contains uncertainty data
        assert os.path.exists(temp_hdf5_file)
        
        with h5py.File(temp_hdf5_file, 'r') as f:
            assert 'uncertainty_estimates' in f
            uncertainty_group = f['uncertainty_estimates']
            for data_name in uncertainty_data.keys():
                assert data_name in uncertainty_group
        
        # Verify statistical summary
        stats = result['statistical_summary']
        assert 'mean_uncertainty' in stats
        assert 'uncertainty_range' in stats
        assert stats['mean_uncertainty'] >= 0


class TestPlateEvolutionEngineCore:
    """Test core functionality of PlateEvolutionEngine"""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for engine initialization"""
        return {
            'grid_resolution': (50, 50),
            'time_step': 1000,
            'max_time': 10000,
            'enable_gpu': False,
            'boundary_conditions': 'periodic'
        }
    
    def test_engine_initialization(self, basic_config):
        """Test PlateEvolutionEngine initialization"""
        engine = PlateEvolutionEngine(basic_config)
        
        # Verify basic attributes
        assert engine.config == basic_config
        assert hasattr(engine, 'grid_resolution')
        assert hasattr(engine, 'time_step')
        assert engine.grid_resolution == (50, 50)
        assert engine.time_step == 1000
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration"""
        with pytest.raises((ValueError, KeyError)):
            PlateEvolutionEngine({})  # Empty config
        
        with pytest.raises(ValueError):
            PlateEvolutionEngine({'grid_resolution': (-10, -10)})  # Invalid resolution


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def engine(self):
        """Create engine for error testing"""
        config = {
            'grid_resolution': (10, 10),
            'time_step': 1000,
            'max_time': 10000,
            'enable_gpu': False
        }
        return PlateEvolutionEngine(config)
    
    def test_empty_data_handling(self, engine):
        """Test handling of empty or None data"""
        # Test with None input
        result = engine._extract_spatial_data_from_reconstruction(None, target_time=0)
        assert result is not None
        assert 'error_status' in result or 'fallback_data' in result
        
        # Test with empty dictionary
        result = engine._extract_spatial_data_from_reconstruction({}, target_time=0)
        assert result is not None
    
    def test_invalid_array_shapes(self, engine):
        """Test handling of mismatched array shapes"""
        # Mismatched velocity components
        vx = np.random.uniform(-5, 5, (10, 20))
        vy = np.random.uniform(-5, 5, (15, 25))  # Different shape
        lon = np.linspace(-180, 180, 20)
        lat = np.linspace(-90, 90, 10)
        
        result = engine._calculate_velocity_divergence(vx, vy, lon, lat)
        # Should handle gracefully or raise specific error
        assert result is not None
    
    def test_extreme_values(self, engine):
        """Test handling of extreme or invalid values"""
        # Very large velocities
        vx = np.full((10, 10), 1e6)
        vy = np.full((10, 10), 1e6)
        lon = np.linspace(-180, 180, 10)
        lat = np.linspace(-90, 90, 10)
        
        result = engine._calculate_velocity_divergence(vx, vy, lon, lat)
        assert result is not None
        
        # NaN values
        vx_nan = np.full((10, 10), np.nan)
        vy_nan = np.full((10, 10), np.nan)
        
        result = engine._calculate_velocity_divergence(vx_nan, vy_nan, lon, lat)
        assert result is not None


class TestIntegration:
    """Integration tests between different methods"""
    
    @pytest.fixture
    def engine(self):
        """Create engine for integration testing"""
        config = {
            'grid_resolution': (20, 20),
            'time_step': 1000,
            'max_time': 10000,
            'enable_gpu': False
        }
        return PlateEvolutionEngine(config)
    
    def test_spatial_data_to_divergence_workflow(self, engine):
        """Test workflow from spatial data extraction to divergence calculation"""
        # Create reconstruction data
        reconstruction_data = {
            'time': 10.0,
            'longitude': np.linspace(-180, 180, 20),
            'latitude': np.linspace(-90, 90, 20),
            'velocity_x': np.random.uniform(-5, 5, (20, 20)),
            'velocity_y': np.random.uniform(-5, 5, (20, 20)),
            'age': np.random.uniform(0, 200, (20, 20))
        }
        
        # Extract spatial data
        spatial_result = engine._extract_spatial_data_from_reconstruction(
            reconstruction_data, target_time=10.0
        )
        
        # Use extracted data for divergence calculation
        velocity_field = spatial_result['velocity_field']
        divergence_result = engine._calculate_velocity_divergence(
            velocity_field['u_component'],
            velocity_field['v_component'],
            reconstruction_data['longitude'],
            reconstruction_data['latitude']
        )
        
        # Verify integration worked
        assert divergence_result is not None
        assert 'divergence' in divergence_result
        assert divergence_result['divergence'].shape == (20, 20)


class TestPerformance:
    """Performance tests for advanced methods"""
    
    @pytest.fixture
    def engine(self):
        """Create engine for performance testing"""
        config = {
            'grid_resolution': (100, 100),
            'time_step': 1000,
            'max_time': 10000,
            'enable_gpu': False
        }
        return PlateEvolutionEngine(config)
    
    @pytest.mark.performance
    def test_large_grid_performance(self, engine):
        """Test performance with large grid sizes"""
        import time
        
        # Large velocity field
        vx = np.random.uniform(-5, 5, (200, 300))
        vy = np.random.uniform(-5, 5, (200, 300))
        lon = np.linspace(-180, 180, 300)
        lat = np.linspace(-90, 90, 200)
        
        start_time = time.time()
        result = engine._calculate_velocity_divergence(vx, vy, lon, lat)
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert execution_time < 10.0  # 10 seconds max
        assert result is not None
        assert 'divergence' in result
    
    @pytest.mark.performance
    def test_memory_efficiency(self, engine):
        """Test memory efficiency with large datasets"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large dataset
        large_data = {
            'velocity_x': np.random.uniform(-5, 5, (500, 500)),
            'velocity_y': np.random.uniform(-5, 5, (500, 500)),
            'temperature': np.random.uniform(273, 1600, (500, 500))
        }
        
        with tempfile.NamedTemporaryFile(suffix='.h5') as temp_file:
            result = engine._save_fallback_spatial_grids(
                large_data, temp_file.name, {}
            )
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 1GB)
        assert memory_increase < 1e9  # 1GB
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])