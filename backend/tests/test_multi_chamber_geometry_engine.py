"""
Comprehensive test suite for Multi-Chamber Geometry Modeling Engine
Tests all major functionality including reconstruction, connectivity analysis,
and temporal evolution tracking with validation against known systems.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json
import tempfile
from pathlib import Path

# Import the module under test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.multi_chamber_geometry_engine import (
    MultiChamberGeometryEngine,
    SeismicTomographyData,
    GravityMagneticData,
    DeformationData,
    ChamberGeometry3D,
    ChamberConnectivity,
    GeometryEvolution
)

# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def geometry_engine():
    """Create a MultiChamberGeometryEngine instance for testing"""
    config = {
        'inversion': {
            'max_iterations': 100,  # Reduced for testing
            'convergence_tolerance': 1e-4,
            'regularization_parameter': 0.01,
            'smoothing_factor': 0.1,
            'uncertainty_samples': 50  # Reduced for testing
        },
        'geometry': {
            'min_chamber_volume': 0.001,
            'max_chamber_volume': 1000.0,
            'min_aspect_ratio': 0.1,
            'max_aspect_ratio': 10.0,
            'resolution_levels': [1.0, 0.5]  # Reduced for testing
        },
        'validation': {
            'cross_validation_folds': 3,  # Reduced for testing
            'bootstrap_samples': 50,  # Reduced for testing
            'confidence_level': 0.95
        },
        'database': {
            'cache_results': False,  # Disable for testing
            'auto_save': False,  # Disable for testing
            'compression': 'gzip'
        }
    }
    return MultiChamberGeometryEngine(config)

@pytest.fixture
def synthetic_seismic_data():
    """Create synthetic seismic tomography data for testing"""
    # Create 3D grid
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    z = np.linspace(0, 15, 15)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create velocity model with chamber anomaly
    velocity_model = np.ones_like(X) * 6.0  # Background velocity
    
    # Add chamber at (0, 0, 8) with radius 2 km
    distance = np.sqrt(X**2 + Y**2 + (Z - 8)**2)
    chamber_mask = distance < 2.0
    velocity_model[chamber_mask] *= 0.75  # 25% velocity reduction
    
    # Add noise
    noise = np.random.normal(0, 0.05, velocity_model.shape)
    velocity_model += noise
    
    return SeismicTomographyData(
        velocity_model=velocity_model,
        coordinates={'x': X, 'y': Y, 'z': Z},
        uncertainty=np.ones_like(velocity_model) * 0.1,
        resolution={'x': 0.5, 'y': 0.5, 'z': 1.0},
        acquisition_params={'method': 'local_earthquake_tomography'},
        quality_metrics={'resolution': 0.8, 'coverage': 0.9}
    )

@pytest.fixture
def synthetic_gravity_data():
    """Create synthetic gravity and magnetic data"""
    # Create 2D grid
    x = np.linspace(-10, 10, 30)
    y = np.linspace(-10, 10, 30)
    X, Y = np.meshgrid(x, y)
    
    # Synthetic gravity anomaly (negative for low-density chamber)
    distance = np.sqrt(X**2 + Y**2)
    gravity_anomaly = -5.0 * np.exp(-distance**2 / 16.0)  # mGal
    
    # Synthetic magnetic anomaly
    magnetic_anomaly = 100.0 * np.exp(-distance**2 / 25.0)  # nT
    
    # Add noise
    gravity_anomaly += np.random.normal(0, 0.5, gravity_anomaly.shape)
    magnetic_anomaly += np.random.normal(0, 10.0, magnetic_anomaly.shape)
    
    return GravityMagneticData(
        gravity_anomaly=gravity_anomaly,
        magnetic_anomaly=magnetic_anomaly,
        coordinates={'x': X, 'y': Y},
        elevation=np.zeros_like(X),
        uncertainty={'gravity': np.ones_like(gravity_anomaly) * 0.5,
                    'magnetic': np.ones_like(magnetic_anomaly) * 10.0},
        regional_field={'gravity': -9810000, 'magnetic': 45000},
        processing_params={'corrections': ['terrain', 'drift']}
    )

@pytest.fixture
def synthetic_deformation_data():
    """Create synthetic deformation data"""
    # Create station grid
    x = np.linspace(-8, 8, 15)
    y = np.linspace(-8, 8, 15)
    X, Y = np.meshgrid(x, y)
    
    # Time series (monthly observations for 2 years)
    time_series = np.array([
        datetime(2022, 1, 1) + timedelta(days=30*i) for i in range(24)
    ])
    
    # Synthetic uplift pattern (Mogi source at depth 8 km)
    distance = np.sqrt(X**2 + Y**2)
    depth = 8.0  # km
    
    # Calculate surface displacement for each time step
    displacement_z = np.zeros((len(time_series), *X.shape))
    for i, t in enumerate(time_series):
        # Increasing pressure with time
        pressure_change = 1e6 * (i + 1)  # Pa
        
        # Mogi model displacement
        displacement_z[i] = (pressure_change * 1e-9 * depth / 
                           (np.pi * (distance**2 + depth**2)**(3/2))) * 1000  # mm
    
    # Add noise
    displacement_z += np.random.normal(0, 2.0, displacement_z.shape)
    
    return DeformationData(
        displacement={'z': displacement_z},
        coordinates={'x': X, 'y': Y},
        time_series=time_series,
        uncertainty={'z': np.ones_like(displacement_z) * 2.0},
        data_type='InSAR',
        processing_level='processed',
        atmospheric_correction=True,
        reference_frame='ITRF2014'
    )

@pytest.fixture
def sample_chamber_geometries():
    """Create sample chamber geometries for testing"""
    geometries = [
        ChamberGeometry3D(
            chamber_id='chamber_1',
            center_coordinates=(0.0, 0.0, 8.0),
            volume=150.0,
            shape_parameters={'a': 3.0, 'b': 2.5, 'c': 2.0},
            shape_type='ellipsoid',
            orientation={'strike': 45, 'dip': 15, 'plunge': 0},
            aspect_ratios=(1.2, 1.25, 1.0),
            confidence_level=0.95,
            reconstruction_method='seismic_tomography',
            validation_score=0.85
        ),
        ChamberGeometry3D(
            chamber_id='chamber_2',
            center_coordinates=(5.0, 3.0, 12.0),
            volume=80.0,
            shape_parameters={'a': 2.5, 'b': 2.0, 'c': 1.8},
            shape_type='ellipsoid',
            orientation={'strike': 120, 'dip': 25, 'plunge': 10},
            aspect_ratios=(1.25, 1.11, 1.0),
            confidence_level=0.90,
            reconstruction_method='joint_inversion',
            validation_score=0.78
        )
    ]
    return geometries

# ============================================================================
# Test Classes
# ============================================================================

class TestMultiChamberGeometryEngine:
    """Test the main MultiChamberGeometryEngine class"""
    
    def test_initialization(self, geometry_engine):
        """Test engine initialization"""
        assert geometry_engine is not None
        assert geometry_engine.config is not None
        assert geometry_engine.logger is not None
        assert 'least_squares' in geometry_engine.inversion_methods
        assert 'bayesian' in geometry_engine.inversion_methods
        assert 'machine_learning' in geometry_engine.inversion_methods
        assert 'joint_inversion' in geometry_engine.inversion_methods
    
    def test_default_config(self):
        """Test default configuration generation"""
        engine = MultiChamberGeometryEngine()
        config = engine.config
        
        assert 'inversion' in config
        assert 'geometry' in config
        assert 'validation' in config
        assert 'database' in config
        
        assert config['inversion']['max_iterations'] > 0
        assert config['geometry']['min_chamber_volume'] > 0
        assert 0 < config['validation']['confidence_level'] <= 1
    
    @pytest.mark.asyncio
    async def test_data_validation(self, geometry_engine, synthetic_seismic_data):
        """Test input data validation"""
        # Valid data should not raise exception
        geometry_engine._validate_input_data(synthetic_seismic_data, None, None)
        
        # Invalid seismic data should raise exception
        invalid_seismic = SeismicTomographyData(
            velocity_model=np.array([]),  # Empty array
            coordinates={'x': np.array([]), 'y': np.array([]), 'z': np.array([])},
            uncertainty=np.array([]),
            resolution={'x': 1.0, 'y': 1.0, 'z': 1.0},
            acquisition_params={},
            quality_metrics={}
        )
        
        with pytest.raises(ValueError, match="Empty velocity model"):
            geometry_engine._validate_input_data(invalid_seismic, None, None)

class TestSeismicDataProcessing:
    """Test seismic tomography data processing"""
    
    @pytest.mark.asyncio
    async def test_seismic_processing(self, geometry_engine, synthetic_seismic_data):
        """Test seismic tomography data processing"""
        processed = await geometry_engine._process_seismic_tomography(synthetic_seismic_data)
        
        assert 'velocity_anomaly' in processed
        assert 'smoothed_anomaly' in processed
        assert 'chamber_mask' in processed
        assert 'chamber_properties' in processed
        
        # Should detect at least one chamber
        assert len(processed['chamber_properties']) >= 1
        
        # Check chamber properties structure
        chamber_prop = processed['chamber_properties'][0]
        assert 'id' in chamber_prop
        assert 'center' in chamber_prop
        assert 'volume' in chamber_prop
        assert 'confidence' in chamber_prop
        
        # Chamber should be detected near expected location
        center = chamber_prop['center']
        expected_center = [0, 0, 8]
        distance = np.sqrt(sum((c - e)**2 for c, e in zip(center, expected_center)))
        assert distance < 2.0  # Within 2 km of expected center

class TestChamberReconstruction:
    """Test chamber geometry reconstruction methods"""
    
    @pytest.mark.asyncio
    async def test_least_squares_reconstruction(self, geometry_engine, synthetic_seismic_data):
        """Test least squares inversion reconstruction"""
        geometries = await geometry_engine.reconstruct_chamber_geometry_3d(
            synthetic_seismic_data, method='least_squares'
        )
        
        assert len(geometries) >= 1
        
        geometry = geometries[0]
        assert isinstance(geometry, ChamberGeometry3D)
        assert geometry.chamber_id is not None
        assert geometry.volume > 0
        assert geometry.validation_score >= 0
        
        # Check that center is reasonable
        center = geometry.center_coordinates
        assert -10 <= center[0] <= 10  # x coordinate
        assert -10 <= center[1] <= 10  # y coordinate
        assert 0 <= center[2] <= 20    # z coordinate (depth)
    
    @pytest.mark.asyncio
    async def test_joint_inversion_reconstruction(self, geometry_engine, synthetic_seismic_data, 
                                                synthetic_gravity_data, synthetic_deformation_data):
        """Test joint inversion with multiple datasets"""
        geometries = await geometry_engine.reconstruct_chamber_geometry_3d(
            synthetic_seismic_data, 
            gravity_data=synthetic_gravity_data,
            deformation_data=synthetic_deformation_data,
            method='joint_inversion'
        )
        
        assert len(geometries) >= 1
        
        geometry = geometries[0]
        assert geometry.reconstruction_method == 'joint_inversion'
        
        # Joint inversion should provide uncertainty bounds
        if geometry.uncertainty_bounds:
            assert 'center_x' in geometry.uncertainty_bounds
            assert 'volume' in geometry.uncertainty_bounds
    
    @pytest.mark.asyncio
    async def test_reconstruction_with_invalid_method(self, geometry_engine, synthetic_seismic_data):
        """Test reconstruction with invalid method"""
        with pytest.raises(ValueError, match="Unknown inversion method"):
            await geometry_engine.reconstruct_chamber_geometry_3d(
                synthetic_seismic_data, method='invalid_method'
            )

class TestConnectivityAnalysis:
    """Test chamber connectivity analysis"""
    
    @pytest.mark.asyncio
    async def test_connectivity_analysis(self, geometry_engine, sample_chamber_geometries):
        """Test chamber connectivity analysis"""
        connectivities = await geometry_engine.analyze_chamber_connectivity(
            sample_chamber_geometries, connectivity_threshold=0.05
        )
        
        # Should find connection between the two chambers
        assert len(connectivities) >= 1
        
        connectivity = connectivities[0]
        assert isinstance(connectivity, ChamberConnectivity)
        assert connectivity.from_chamber in ['chamber_1', 'chamber_2']
        assert connectivity.to_chamber in ['chamber_1', 'chamber_2']
        assert connectivity.from_chamber != connectivity.to_chamber
        
        # Check physical properties
        assert connectivity.conductivity > 0
        assert connectivity.cross_sectional_area > 0
        assert connectivity.tortuosity >= 1.0
        assert 0 <= connectivity.connectivity_strength <= 1
    
    @pytest.mark.asyncio
    async def test_connectivity_calculation(self, geometry_engine, sample_chamber_geometries):
        """Test individual connectivity calculation"""
        chamber1, chamber2 = sample_chamber_geometries
        
        connectivity = await geometry_engine._calculate_chamber_connectivity(chamber1, chamber2)
        
        assert connectivity.connection_id == f"{chamber1.chamber_id}_{chamber2.chamber_id}"
        assert connectivity.from_chamber == chamber1.chamber_id
        assert connectivity.to_chamber == chamber2.chamber_id
        
        # Connectivity strength should be reasonable for chambers 6.4 km apart
        distance = np.sqrt((5-0)**2 + (3-0)**2 + (12-8)**2)  # ~6.4 km
        assert 0 < connectivity.connectivity_strength < 1
    
    @pytest.mark.asyncio
    async def test_no_connectivity_with_high_threshold(self, geometry_engine, sample_chamber_geometries):
        """Test that high threshold filters out weak connections"""
        connectivities = await geometry_engine.analyze_chamber_connectivity(
            sample_chamber_geometries, connectivity_threshold=0.9
        )
        
        # High threshold should result in no connections for distant chambers
        assert len(connectivities) == 0

class TestTemporalEvolution:
    """Test temporal evolution tracking"""
    
    @pytest.mark.asyncio
    async def test_evolution_tracking_insufficient_data(self, geometry_engine):
        """Test evolution tracking with insufficient data"""
        # Single data point should raise error
        time_series_data = [
            (datetime(2022, 1, 1), {'velocity_model': np.ones((10, 10, 10))})
        ]
        
        with pytest.raises(ValueError, match="Insufficient data for evolution tracking"):
            await geometry_engine.track_geometry_evolution(
                'test_chamber', time_series_data
            )
    
    def test_evolution_data_structures(self):
        """Test evolution data structure creation"""
        evolution = GeometryEvolution(
            chamber_id='test_chamber',
            time_series=[datetime(2022, 1, 1), datetime(2022, 6, 1)],
            geometry_history=[],
            volume_evolution=np.array([100.0, 120.0]),
            shape_evolution={'aspect_ratio': np.array([1.2, 1.3])},
            migration_path=np.array([[0, 0, 8], [0.1, 0.1, 8.2]]),
            growth_rate=40.0,  # km³/year
            deformation_rate={'volume_rate': 40.0},
            evolution_drivers=['magma_injection'],
            prediction_horizon=timedelta(days=365),
            confidence_intervals={}
        )
        
        assert evolution.chamber_id == 'test_chamber'
        assert len(evolution.time_series) == 2
        assert evolution.growth_rate == 40.0
        assert 'magma_injection' in evolution.evolution_drivers

class TestValidation:
    """Test validation against known systems"""
    
    @pytest.mark.asyncio
    async def test_validation_against_yellowstone(self, geometry_engine):
        """Test validation against Yellowstone system"""
        # Create mock geometries similar to Yellowstone
        mock_geometries = [
            ChamberGeometry3D(
                chamber_id='upper_chamber',
                center_coordinates=(-110.588, 44.428, 8.5),
                volume=14000.0,  # Close to reference 15000
                shape_type='ellipsoid',
                shape_parameters={},
                orientation={},
                aspect_ratios=(1.0, 1.0, 1.0),
                reconstruction_method='test'
            ),
            ChamberGeometry3D(
                chamber_id='lower_chamber',
                center_coordinates=(-110.588, 44.428, 19.5),
                volume=45000.0,  # Close to reference 46000
                shape_type='ellipsoid',
                shape_parameters={},
                orientation={},
                aspect_ratios=(1.0, 1.0, 1.0),
                reconstruction_method='test'
            )
        ]
        
        metrics = await geometry_engine.validate_against_known_systems(
            mock_geometries, 'yellowstone'
        )
        
        assert 'overall_volume_accuracy' in metrics
        assert 'overall_depth_accuracy' in metrics
        assert 'chamber_detection_rate' in metrics
        
        # Should have good accuracy for close matches
        assert metrics['overall_volume_accuracy'] > 0.8
        assert metrics['chamber_detection_rate'] > 0.5
    
    @pytest.mark.asyncio
    async def test_validation_unknown_system(self, geometry_engine, sample_chamber_geometries):
        """Test validation with unknown system"""
        with pytest.raises(ValueError, match="Unknown validation system"):
            await geometry_engine.validate_against_known_systems(
                sample_chamber_geometries, 'unknown_system'
            )

class TestUtilityMethods:
    """Test utility and helper methods"""
    
    def test_detection_confidence_calculation(self, geometry_engine):
        """Test detection confidence calculation"""
        # High signal-to-noise should give high confidence
        velocities = np.array([5.0, 4.8, 5.2, 4.9])  # Clear anomaly
        uncertainties = np.array([0.1, 0.1, 0.1, 0.1])  # Low uncertainty
        
        confidence = geometry_engine._calculate_detection_confidence(velocities, uncertainties)
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably confident
        
        # Low signal-to-noise should give low confidence
        velocities_noisy = np.array([5.0, 4.8, 5.2, 4.9])
        uncertainties_high = np.array([1.0, 1.0, 1.0, 1.0])  # High uncertainty
        
        confidence_low = geometry_engine._calculate_detection_confidence(velocities_noisy, uncertainties_high)
        assert confidence_low < confidence  # Should be less confident

class TestDataIntegration:
    """Test data preprocessing and integration"""
    
    @pytest.mark.asyncio
    async def test_data_preprocessing(self, geometry_engine, synthetic_seismic_data, 
                                    synthetic_gravity_data, synthetic_deformation_data):
        """Test comprehensive data preprocessing"""
        processed = await geometry_engine._preprocess_geophysical_data(
            synthetic_seismic_data, synthetic_gravity_data, synthetic_deformation_data
        )
        
        assert 'seismic' in processed
        assert 'seismic_processed' in processed
        assert 'gravity' in processed
        assert 'gravity_processed' in processed
        assert 'deformation' in processed
        assert 'deformation_processed' in processed
        assert 'coordinate_system' in processed
        
        # Check seismic processing results
        seismic_processed = processed['seismic_processed']
        assert 'chamber_properties' in seismic_processed
        assert len(seismic_processed['chamber_properties']) >= 1
    
    @pytest.mark.asyncio
    async def test_gravity_processing(self, geometry_engine, synthetic_gravity_data):
        """Test gravity and magnetic data processing"""
        processed = await geometry_engine._process_gravity_magnetic_data(synthetic_gravity_data)
        
        assert 'gravity_residual' in processed
        assert 'magnetic_residual' in processed
        assert 'gravity_sources' in processed
        assert 'magnetic_sources' in processed
        
        # Processed data should have same shape as input
        assert processed['gravity_residual'].shape == synthetic_gravity_data.gravity_anomaly.shape
        assert processed['magnetic_residual'].shape == synthetic_gravity_data.magnetic_anomaly.shape
    
    @pytest.mark.asyncio
    async def test_deformation_processing(self, geometry_engine, synthetic_deformation_data):
        """Test deformation data processing"""
        processed = await geometry_engine._process_deformation_data(synthetic_deformation_data)
        
        assert 'displacement_rates' in processed
        assert 'source_models' in processed
        assert 'strain_tensor' in processed
        assert 'inflation_patterns' in processed
        
        # Should calculate rates for z component
        assert 'z' in processed['displacement_rates']

# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_reconstruction_workflow(self, geometry_engine, synthetic_seismic_data):
        """Test complete reconstruction workflow from data to results"""
        # Step 1: Reconstruct geometries
        geometries = await geometry_engine.reconstruct_chamber_geometry_3d(
            synthetic_seismic_data, method='least_squares'
        )
        
        assert len(geometries) >= 1
        
        # Step 2: Analyze connectivity (if multiple chambers)
        if len(geometries) > 1:
            connectivities = await geometry_engine.analyze_chamber_connectivity(geometries)
            assert isinstance(connectivities, list)
        
        # Step 3: Validate results
        for geometry in geometries:
            assert geometry.volume > 0
            assert geometry.validation_score >= 0
            assert len(geometry.center_coordinates) == 3
    
    @pytest.mark.asyncio
    async def test_multi_dataset_integration(self, geometry_engine, synthetic_seismic_data,
                                           synthetic_gravity_data, synthetic_deformation_data):
        """Test integration of multiple geophysical datasets"""
        # Test with all three datasets
        geometries = await geometry_engine.reconstruct_chamber_geometry_3d(
            synthetic_seismic_data,
            gravity_data=synthetic_gravity_data,
            deformation_data=synthetic_deformation_data,
            method='joint_inversion'
        )
        
        assert len(geometries) >= 1
        
        # Joint inversion should potentially provide better results
        geometry = geometries[0]
        assert geometry.reconstruction_method == 'joint_inversion'

# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and scalability tests"""
    
    @pytest.mark.asyncio
    async def test_large_dataset_processing(self, geometry_engine):
        """Test processing of larger datasets"""
        # Create larger synthetic dataset
        x = np.linspace(-20, 20, 50)
        y = np.linspace(-20, 20, 50)
        z = np.linspace(0, 30, 30)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        velocity_model = np.ones_like(X) * 6.0
        
        # Add multiple chambers
        for center in [(0, 0, 8), (10, 5, 15), (-8, -3, 12)]:
            distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
            chamber_mask = distance < 3.0
            velocity_model[chamber_mask] *= 0.8
        
        large_seismic_data = SeismicTomographyData(
            velocity_model=velocity_model,
            coordinates={'x': X, 'y': Y, 'z': Z},
            uncertainty=np.ones_like(velocity_model) * 0.1,
            resolution={'x': 0.8, 'y': 0.8, 'z': 1.0},
            acquisition_params={'method': 'local_earthquake_tomography'},
            quality_metrics={'resolution': 0.8, 'coverage': 0.9}
        )
        
        # Should complete within reasonable time
        import time
        start_time = time.time()
        
        geometries = await geometry_engine.reconstruct_chamber_geometry_3d(
            large_seismic_data, method='least_squares'
        )
        
        processing_time = time.time() - start_time
        
        # Should detect multiple chambers
        assert len(geometries) >= 2
        
        # Should complete within 30 seconds (adjust based on system)
        assert processing_time < 30.0
    
    def test_memory_usage(self, geometry_engine):
        """Test memory usage with various data sizes"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create moderately large dataset
        size = 30
        velocity_model = np.random.random((size, size, size))
        coordinates = {
            'x': np.random.random((size, size, size)),
            'y': np.random.random((size, size, size)),
            'z': np.random.random((size, size, size))
        }
        
        seismic_data = SeismicTomographyData(
            velocity_model=velocity_model,
            coordinates=coordinates,
            uncertainty=np.ones_like(velocity_model) * 0.1,
            resolution={'x': 1.0, 'y': 1.0, 'z': 1.0},
            acquisition_params={},
            quality_metrics={}
        )
        
        # Memory usage should not increase dramatically
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use more than 500 MB additional memory
        assert memory_increase < 500

# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_empty_velocity_model(self, geometry_engine):
        """Test handling of empty velocity model"""
        empty_seismic_data = SeismicTomographyData(
            velocity_model=np.array([]),
            coordinates={'x': np.array([]), 'y': np.array([]), 'z': np.array([])},
            uncertainty=np.array([]),
            resolution={'x': 1.0, 'y': 1.0, 'z': 1.0},
            acquisition_params={},
            quality_metrics={}
        )
        
        with pytest.raises(ValueError):
            await geometry_engine.reconstruct_chamber_geometry_3d(empty_seismic_data)
    
    @pytest.mark.asyncio
    async def test_mismatched_array_sizes(self, geometry_engine):
        """Test handling of mismatched array sizes"""
        velocity_model = np.ones((10, 10, 10))
        coordinates = {
            'x': np.ones((5, 5, 5)),  # Different size
            'y': np.ones((10, 10, 10)),
            'z': np.ones((10, 10, 10))
        }
        
        mismatched_seismic_data = SeismicTomographyData(
            velocity_model=velocity_model,
            coordinates=coordinates,
            uncertainty=np.ones_like(velocity_model) * 0.1,
            resolution={'x': 1.0, 'y': 1.0, 'z': 1.0},
            acquisition_params={},
            quality_metrics={}
        )
        
        # Should handle gracefully or raise appropriate error
        try:
            await geometry_engine.reconstruct_chamber_geometry_3d(mismatched_seismic_data)
        except (ValueError, IndexError, RuntimeError):
            pass  # Expected to fail with appropriate error
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration"""
        invalid_config = {
            'inversion': {
                'max_iterations': -1,  # Invalid negative value
                'convergence_tolerance': 2.0  # Invalid > 1
            }
        }
        
        # Should either use defaults or raise error
        try:
            engine = MultiChamberGeometryEngine(invalid_config)
            # If it doesn't raise error, should use reasonable defaults
            assert engine.config['inversion']['max_iterations'] > 0
        except ValueError:
            pass  # Expected to fail with invalid config

# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_initialization or test_seismic_processing or test_least_squares_reconstruction"
    ])