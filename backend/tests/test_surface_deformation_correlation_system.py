"""
Comprehensive test suite for Surface Deformation Correlation System

This test suite validates all components of the surface deformation correlation system
including InSAR correlation, GPS analysis, tilt/strain correlation, forward modeling,
parameter inversion, temporal analysis, uncertainty quantification, and validation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from typing import Dict, List, Any

# Import the system under test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from surface_deformation_correlation_system import (
    SurfaceDeformationCorrelationSystem,
    InSARDeformationData,
    GPSDeformationData,
    TiltStrainData,
    SillGeometryModel,
    DeformationCorrelationResult
)

class TestSurfaceDeformationCorrelationSystem:
    """Test class for Surface Deformation Correlation System"""
    
    @pytest.fixture
    async def correlation_system(self):
        """Create a test instance of the correlation system"""
        config = {
            'database': {
                'postgres_host': 'localhost',
                'postgres_port': 5432,
                'postgres_user': 'test_user',
                'postgres_password': 'test_pass',
                'postgres_db': 'test_db',
                'redis_host': 'localhost',
                'redis_port': 6379,
                'influx_url': 'http://localhost:8086',
                'influx_token': 'test-token',
                'influx_org': 'test-org',
                'neo4j_uri': 'bolt://localhost:7687',
                'neo4j_user': 'neo4j',
                'neo4j_password': 'test-pass'
            },
            'processing': {
                'max_workers': 2,
                'chunk_size': 100
            },
            'validation': {
                'min_correlation': 0.3,
                'significance_level': 0.05
            }
        }
        
        system = SurfaceDeformationCorrelationSystem(config)
        
        # Mock database connections
        system.postgres_pool = AsyncMock()
        system.redis_client = AsyncMock()
        system.influx_client = AsyncMock()
        system.neo4j_driver = AsyncMock()
        
        return system
    
    @pytest.fixture
    def sample_sill_model(self):
        """Create a sample sill geometry model for testing"""
        return SillGeometryModel(
            sill_id="test_sill_001",
            center_coordinates=(-122.5, 37.8, 2000),  # lon, lat, depth
            length=5000,  # meters
            width=3000,   # meters
            thickness=50, # meters
            strike=45,    # degrees
            dip=10,       # degrees
            opening=0.5,  # meters
            pressure_change=1e6,  # Pa
            elastic_moduli=(30e9, 0.25),  # Young's modulus, Poisson's ratio
            host_rock_properties={'density': 2700},
            emplacement_time=datetime.now(),
            growth_rate=0.1,  # m/year
            geometry_uncertainty={'depth': 200, 'length': 500},
            validation_metrics={}
        )
    
    @pytest.fixture
    def sample_insar_data(self):
        """Create sample InSAR deformation data"""
        # Create synthetic deformation map
        nx, ny = 50, 50
        x = np.linspace(-10000, 10000, nx)
        y = np.linspace(-10000, 10000, ny)
        X, Y = np.meshgrid(x, y)
        
        # Synthetic Mogi-like deformation pattern
        R = np.sqrt(X**2 + Y**2 + 2000**2)
        deformation = 0.1 * 2000 / R**3  # Simplified Mogi model
        
        # Add noise
        deformation += np.random.normal(0, 0.005, deformation.shape)
        
        # Create uncertainty and coherence maps
        uncertainty = np.full_like(deformation, 0.01)
        coherence = np.random.uniform(0.3, 0.9, deformation.shape)
        
        # Convert to lon/lat coordinates (approximate)
        lon_grid = -122.5 + X / 111000  # Approximate conversion
        lat_grid = 37.8 + Y / 111000
        
        return InSARDeformationData(
            acquisition_date=datetime.now(),
            satellite="Sentinel-1A",
            track=144,
            frame=350,
            look_angle=23.0,
            heading_angle=-13.3,
            wavelength=0.056,  # C-band
            spatial_resolution=20.0,
            temporal_baseline=12.0,
            perpendicular_baseline=150.0,
            coherence_threshold=0.3,
            unwrapping_method="snaphu",
            atmospheric_correction="gacos",
            deformation_map=deformation,
            uncertainty_map=uncertainty,
            coherence_map=coherence,
            coordinates=(lon_grid, lat_grid),
            reference_point=(-122.5, 37.8),
            processing_parameters={'multilook': '4x1', 'filter': 'goldstein'},
            quality_metrics={'mean_coherence': 0.65, 'unwrapping_errors': 0.02}
        )
    
    @pytest.fixture
    def sample_gps_data(self):
        """Create sample GPS deformation data"""
        gps_stations = []
        
        # Create 5 GPS stations around the sill
        station_positions = [
            (-122.51, 37.81, 100),
            (-122.49, 37.79, 150),
            (-122.52, 37.82, 80),
            (-122.48, 37.78, 120),
            (-122.50, 37.80, 90)
        ]
        
        for i, (lon, lat, elev) in enumerate(station_positions):
            # Calculate synthetic displacement (simplified Mogi)
            dx = (lon + 122.5) * 111000
            dy = (lat - 37.8) * 111000
            r = np.sqrt(dx**2 + dy**2)
            R = np.sqrt(r**2 + 2000**2)
            
            # Mogi displacement
            ur = 0.1 * r / R**3
            uz = 0.1 * 2000 / R**3
            
            # Convert radial to east-north
            ux = ur * dx / (r + 1e-10)
            uy = ur * dy / (r + 1e-10)
            
            # Add noise
            ux += np.random.normal(0, 0.002)
            uy += np.random.normal(0, 0.002)
            uz += np.random.normal(0, 0.005)
            
            gps_station = GPSDeformationData(
                station_id=f"GPS{i+1:02d}",
                coordinates=(lon, lat, elev),
                measurement_date=datetime.now(),
                displacement_vector=(ux, uy, uz),
                velocity_vector=(ux/365, uy/365, uz/365),  # Convert to daily
                uncertainty_vector=(0.002, 0.002, 0.005),
                correlation_matrix=np.eye(3) * 0.1,
                measurement_type='continuous',
                processing_strategy='PPP',
                reference_frame='ITRF2014',
                antenna_type='AOAD/M_T',
                monument_type='concrete_pillar',
                data_quality={'rms': 0.003, 'availability': 0.95},
                environmental_corrections={'troposphere': 0.001, 'multipath': 0.0005}
            )
            gps_stations.append(gps_station)
        
        return gps_stations
    
    @pytest.fixture
    def sample_tilt_strain_data(self):
        """Create sample tilt and strain data"""
        instruments = []
        
        # Create 3 tiltmeters and 2 strainmeters
        instrument_configs = [
            ('TILT01', 'tiltmeter', -122.505, 37.805, 50),
            ('TILT02', 'tiltmeter', -122.495, 37.795, 30),
            ('TILT03', 'tiltmeter', -122.515, 37.815, 40),
            ('STRAIN01', 'strainmeter', -122.490, 37.790, 100),
            ('STRAIN02', 'borehole_strainmeter', -122.510, 37.810, 200)
        ]
        
        for inst_id, inst_type, lon, lat, depth in instrument_configs:
            # Calculate synthetic tilt/strain (simplified)
            dx = (lon + 122.5) * 111000
            dy = (lat - 37.8) * 111000
            r = np.sqrt(dx**2 + dy**2)
            
            if 'tilt' in inst_type:
                # Synthetic tilt components (microradians)
                tilt_x = np.random.normal(0, 5) + 10 * dx / (r**2 + 1e6)
                tilt_y = np.random.normal(0, 5) + 10 * dy / (r**2 + 1e6)
                tilt_components = (tilt_x, tilt_y)
                strain_tensor = None
                uncertainty_tilt = (1.0, 1.0)
                uncertainty_strain = None
            else:
                # Synthetic strain tensor
                exx = np.random.normal(0, 1e-8) + 1e-7 / (r + 1000)
                eyy = np.random.normal(0, 1e-8) + 1e-7 / (r + 1000)
                exy = np.random.normal(0, 5e-9)
                strain_tensor = np.array([[exx, exy], [exy, eyy]])
                tilt_components = None
                uncertainty_tilt = None
                uncertainty_strain = np.array([[1e-9, 5e-10], [5e-10, 1e-9]])
            
            instrument = TiltStrainData(
                instrument_id=inst_id,
                instrument_type=inst_type,
                coordinates=(lon, lat, depth),
                measurement_date=datetime.now(),
                tilt_components=tilt_components,
                strain_tensor=strain_tensor,
                shear_strain=exy if strain_tensor is not None else None,
                volumetric_strain=exx + eyy if strain_tensor is not None else None,
                uncertainty_tilt=uncertainty_tilt,
                uncertainty_strain=uncertainty_strain,
                temperature_correction=0.001,
                pressure_correction=0.0005,
                installation_depth=depth,
                calibration_date=datetime.now() - timedelta(days=30),
                drift_correction=0.0001,
                data_quality_flags=['good']
            )
            instruments.append(instrument)
        
        return instruments

    @pytest.mark.asyncio
    async def test_insar_deformation_correlation(self, correlation_system, sample_sill_model, sample_insar_data):
        """Test InSAR deformation correlation functionality"""
        
        # Mock database storage
        correlation_system._store_insar_correlation_results = AsyncMock()
        correlation_system._store_residuals_timeseries = AsyncMock()
        
        # Test correlation
        correlation_params = {
            'spatial_window': 15000,
            'temporal_window': 365,
            'coherence_threshold': 0.3
        }
        
        result = await correlation_system.correlate_insar_deformation(
            [sample_insar_data], sample_sill_model, correlation_params
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check that at least one dataset was processed
        dataset_key = list(result.keys())[0]
        dataset_result = result[dataset_key]
        
        assert 'correlation_coefficient' in dataset_result
        assert 'weighted_correlation' in dataset_result
        assert 'rms_misfit' in dataset_result
        assert 'statistical_significance' in dataset_result
        assert 'valid_pixels' in dataset_result
        
        # Verify correlation coefficient is reasonable
        assert -1 <= dataset_result['correlation_coefficient'] <= 1
        assert dataset_result['valid_pixels'] > 0
        assert dataset_result['rms_misfit'] >= 0
        
        # Verify database storage was called
        correlation_system._store_insar_correlation_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_gps_deformation_analysis(self, correlation_system, sample_sill_model, sample_gps_data):
        """Test GPS deformation analysis functionality"""
        
        # Mock database storage
        correlation_system._store_gps_correlation_results = AsyncMock()
        
        # Test GPS analysis
        analysis_params = {
            'max_distance': 50000,
            'min_stations': 3,
            'velocity_threshold': 1e-3
        }
        
        result = await correlation_system.analyze_gps_deformation(
            sample_gps_data, sample_sill_model, analysis_params
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert 'east' in result
        assert 'north' in result
        assert 'up' in result
        assert 'overall' in result
        
        # Check component results
        for component in ['east', 'north', 'up']:
            comp_result = result[component]
            assert 'correlation_coefficient' in comp_result
            assert 'weighted_correlation' in comp_result
            assert 'rms_misfit' in comp_result
            assert 'statistical_significance' in comp_result
            
            # Verify correlation coefficient is reasonable
            assert -1 <= comp_result['correlation_coefficient'] <= 1
            assert comp_result['rms_misfit'] >= 0
        
        # Check overall results
        overall = result['overall']
        assert 'total_correlation' in overall
        assert 'num_stations' in overall
        assert overall['num_stations'] == len(sample_gps_data)
        
        # Verify database storage was called
        correlation_system._store_gps_correlation_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_tilt_strain_correlation(self, correlation_system, sample_sill_model, sample_tilt_strain_data):
        """Test tilt and strain correlation functionality"""
        
        # Mock database storage
        correlation_system._store_tilt_strain_correlation_results = AsyncMock()
        
        # Test tilt/strain correlation
        correlation_params = {
            'max_distance': 20000,
            'depth_weighting': True
        }
        
        result = await correlation_system.correlate_tilt_strain_data(
            sample_tilt_strain_data, sample_sill_model, correlation_params
        )
        
        # Verify results
        assert isinstance(result, dict)
        
        # Should have tilt and strain results
        if 'tilt' in result:
            tilt_result = result['tilt']
            assert 'tilt_x' in tilt_result
            assert 'tilt_y' in tilt_result
            
            for component in ['tilt_x', 'tilt_y']:
                comp_result = tilt_result[component]
                assert 'correlation' in comp_result
                assert 'rms_misfit' in comp_result
                assert 'num_instruments' in comp_result
                assert -1 <= comp_result['correlation'] <= 1
        
        if 'strain' in result:
            strain_result = result['strain']
            for component in ['exx', 'eyy', 'exy']:
                if component in strain_result:
                    comp_result = strain_result[component]
                    assert 'correlation' in comp_result
                    assert 'rms_misfit' in comp_result
                    assert -1 <= comp_result['correlation'] <= 1
        
        # Verify database storage was called
        correlation_system._store_tilt_strain_correlation_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_forward_model_surface_deformation(self, correlation_system, sample_sill_model):
        """Test forward modeling of surface deformation"""
        
        # Mock database storage
        correlation_system._store_forward_model_results = AsyncMock()
        
        # Create observation points
        observation_coords = np.array([
            [-122.51, 37.81],
            [-122.49, 37.79],
            [-122.52, 37.82],
            [-122.48, 37.78],
            [-122.50, 37.80]
        ])
        
        # Test forward modeling
        model_params = {
            'elastic_parameters': {
                'poisson_ratio': 0.25,
                'youngs_modulus': 30e9
            }
        }
        
        result = await correlation_system.forward_model_surface_deformation(
            sample_sill_model, observation_coords, model_params
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert 'mogi_model' in result
        assert 'finite_model' in result
        assert 'combined_model' in result
        assert 'model_weights' in result
        assert 'observation_coordinates' in result
        assert 'model_parameters' in result
        
        # Check deformation components
        for model_type in ['mogi_model', 'finite_model', 'combined_model']:
            model_result = result[model_type]
            assert 'east' in model_result
            assert 'north' in model_result
            assert 'up' in model_result
            
            # Verify array shapes
            assert len(model_result['east']) == len(observation_coords)
            assert len(model_result['north']) == len(observation_coords)
            assert len(model_result['up']) == len(observation_coords)
            
            # Verify finite values
            assert np.all(np.isfinite(model_result['east']))
            assert np.all(np.isfinite(model_result['north']))
            assert np.all(np.isfinite(model_result['up']))
        
        # Verify database storage was called
        correlation_system._store_forward_model_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_parameter_inversion(self, correlation_system, sample_sill_model):
        """Test sill parameter inversion functionality"""
        
        # Mock database storage
        correlation_system._store_inversion_results = AsyncMock()
        
        # Create synthetic observation data
        observation_coords = np.array([
            [-122.51, 37.81],
            [-122.49, 37.79],
            [-122.52, 37.82],
            [-122.48, 37.78]
        ])
        
        # Generate synthetic deformation with known model
        forward_result = await correlation_system.forward_model_surface_deformation(
            sample_sill_model, observation_coords, {}
        )
        
        # Add noise to create "observations"
        deformation_obs = {
            'insar': {
                'deformation': forward_result['combined_model']['up'] + np.random.normal(0, 0.01, len(observation_coords)),
                'uncertainty': np.full(len(observation_coords), 0.01)
            }
        }
        
        # Test parameter inversion
        inversion_params = {
            'method': 'differential_evolution',
            'max_iterations': 100,
            'tolerance': 1e-4,
            'uncertainty_estimation': False  # Skip for speed
        }
        
        result = await correlation_system.invert_sill_parameters(
            deformation_obs, observation_coords, sample_sill_model, inversion_params
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert 'optimized_sill_model' in result
        assert 'optimization_result' in result
        assert 'forward_predictions' in result
        assert 'residuals' in result
        assert 'model_statistics' in result
        
        # Check optimization result
        opt_result = result['optimization_result']
        assert 'success' in opt_result
        assert 'final_misfit' in opt_result
        assert opt_result['final_misfit'] >= 0
        
        # Check model statistics
        stats = result['model_statistics']
        assert 'rms_misfit' in stats
        assert 'aic' in stats
        assert 'bic' in stats
        assert stats['rms_misfit'] >= 0
        
        # Verify optimized model is reasonable
        opt_model = result['optimized_sill_model']
        assert opt_model.length > 0
        assert opt_model.width > 0
        assert opt_model.center_coordinates[2] > 0  # Depth should be positive
        
        # Verify database storage was called
        correlation_system._store_inversion_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_temporal_deformation_analysis(self, correlation_system, sample_sill_model):
        """Test temporal deformation analysis functionality"""
        
        # Mock database storage
        correlation_system._store_temporal_analysis_results = AsyncMock()
        
        # Create synthetic time series data
        time_series_data = {
            'gps': []
        }
        
        # Generate GPS time series over 1 year
        base_date = datetime.now() - timedelta(days=365)
        for i in range(12):  # Monthly measurements
            measurement_date = base_date + timedelta(days=i*30)
            
            # Synthetic temporal evolution (linear trend + noise)
            trend_factor = i * 0.01  # 1 cm/year trend
            noise = np.random.normal(0, 0.005)
            
            gps_measurement = GPSDeformationData(
                station_id="GPS01",
                coordinates=(-122.50, 37.80, 100),
                measurement_date=measurement_date,
                displacement_vector=(0.001 + trend_factor + noise, 
                                   0.002 + trend_factor + noise,
                                   0.005 + trend_factor*2 + noise),
                velocity_vector=(0.001/365, 0.002/365, 0.005/365),
                uncertainty_vector=(0.002, 0.002, 0.005),
                correlation_matrix=np.eye(3) * 0.1,
                measurement_type='campaign',
                processing_strategy='PPP',
                reference_frame='ITRF2014',
                antenna_type='AOAD/M_T',
                monument_type='concrete_pillar',
                data_quality={'rms': 0.003, 'availability': 0.95},
                environmental_corrections={'troposphere': 0.001}
            )
            time_series_data['gps'].append(gps_measurement)
        
        # Test temporal analysis
        temporal_params = {
            'time_window': 365,
            'smoothing_window': 30,
            'trend_analysis': True,
            'change_detection': True
        }
        
        result = await correlation_system.analyze_temporal_deformation(
            time_series_data, sample_sill_model, temporal_params
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert 'gps' in result
        
        gps_results = result['gps']
        
        # Should have results for each GPS component
        for component in ['east', 'north', 'up']:
            if component in gps_results:
                comp_result = gps_results[component]
                assert 'times' in comp_result
                assert 'values' in comp_result
                assert 'smoothed_values' in comp_result
                assert 'trend_analysis' in comp_result
                assert 'velocity_results' in comp_result
                
                # Verify array lengths match
                assert len(comp_result['times']) == len(comp_result['values'])
                assert len(comp_result['smoothed_values']) == len(comp_result['values'])
        
        # Verify database storage was called
        correlation_system._store_temporal_analysis_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_uncertainty_quantification(self, correlation_system):
        """Test uncertainty quantification functionality"""
        
        # Mock database storage
        correlation_system._store_uncertainty_results = AsyncMock()
        
        # Create mock correlation results
        correlation_results = {
            'inversion_results': {
                'optimized_sill_model': SillGeometryModel(
                    sill_id="test_sill",
                    center_coordinates=(-122.5, 37.8, 2000),
                    length=5000, width=3000, thickness=50,
                    strike=45, dip=10, opening=0.5,
                    pressure_change=1e6, elastic_moduli=(30e9, 0.25),
                    host_rock_properties={}, emplacement_time=datetime.now(),
                    growth_rate=0.1, geometry_uncertainty={}, validation_metrics={}
                ),
                'parameter_uncertainties': {
                    'depth_std': 100,
                    'length_std': 200,
                    'opening_std': 0.05
                }
            }
        }
        
        model_parameters = {'sill_id': 'test_sill'}
        
        uncertainty_params = {
            'confidence_level': 0.95,
            'monte_carlo_samples': 100,
            'bootstrap_samples': 50
        }
        
        # Mock the uncertainty calculation methods
        correlation_system._propagate_parameter_uncertainties = AsyncMock(return_value={
            'depth': (1900, 2100),
            'length': (4800, 5200)
        })
        correlation_system._propagate_data_uncertainties = AsyncMock(return_value={
            'deformation': (0.01, 0.02)
        })
        correlation_system._assess_model_uncertainties = AsyncMock(return_value={
            'model_error': (0.005, 0.015)
        })
        correlation_system._bootstrap_uncertainty_estimation = AsyncMock(return_value={
            'bootstrap_depth': (1950, 2050)
        })
        correlation_system._combine_uncertainty_sources = MagicMock(return_value={
            'depth': (1900, 2100),
            'length': (4800, 5200),
            'opening': (0.45, 0.55)
        })
        correlation_system._prepare_uncertainty_visualization = MagicMock(return_value={
            'confidence_ellipses': {},
            'parameter_distributions': {}
        })
        
        # Test uncertainty quantification
        result = await correlation_system.quantify_uncertainties(
            correlation_results, model_parameters, uncertainty_params
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check that uncertainty bounds are tuples
        for param, bounds in result.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2
            assert bounds[0] <= bounds[1]  # Lower bound <= upper bound
        
        # Verify all mock methods were called
        correlation_system._propagate_parameter_uncertainties.assert_called_once()
        correlation_system._propagate_data_uncertainties.assert_called_once()
        correlation_system._assess_model_uncertainties.assert_called_once()
        correlation_system._bootstrap_uncertainty_estimation.assert_called_once()
        correlation_system._combine_uncertainty_sources.assert_called_once()
        correlation_system._prepare_uncertainty_visualization.assert_called_once()
        
        # Verify database storage was called
        correlation_system._store_uncertainty_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_validation_against_monitored_events(self, correlation_system):
        """Test validation against monitored sill intrusion events"""
        
        # Mock database storage
        correlation_system._store_validation_results = AsyncMock()
        
        # Create mock correlation results
        correlation_results = {
            'correlation_id': 'test_correlation_001',
            'sill_model': SillGeometryModel(
                sill_id="test_sill",
                center_coordinates=(-122.5, 37.8, 2000),
                length=5000, width=3000, thickness=50,
                strike=45, dip=10, opening=0.5,
                pressure_change=1e6, elastic_moduli=(30e9, 0.25),
                host_rock_properties={}, emplacement_time=datetime.now(),
                growth_rate=0.1, geometry_uncertainty={}, validation_metrics={}
            )
        }
        
        # Create mock monitored events
        monitored_events = [
            {
                'event_id': 'EVENT_001',
                'location': (-122.51, 37.81, 1950),  # Close to sill
                'event_time': datetime.now() - timedelta(days=10),
                'magnitude': 0.8,
                'event_type': 'sill_intrusion',
                'deformation_signature': {'max_uplift': 0.05}
            },
            {
                'event_id': 'EVENT_002',
                'location': (-122.45, 37.75, 3000),  # Further from sill
                'event_time': datetime.now() - timedelta(days=50),
                'magnitude': 0.3,
                'event_type': 'sill_intrusion',
                'deformation_signature': {'max_uplift': 0.02}
            }
        ]
        
        validation_params = {
            'temporal_tolerance': 30,  # days
            'spatial_tolerance': 5000,  # meters
            'magnitude_tolerance': 0.5
        }
        
        # Mock validation methods
        correlation_system._validate_spatial_correlation = MagicMock(side_effect=[0.8, 0.3])
        correlation_system._validate_temporal_correlation = MagicMock(side_effect=[0.9, 0.6])
        correlation_system._validate_magnitude_correlation = MagicMock(side_effect=[0.7, 0.4])
        correlation_system._validate_deformation_pattern = AsyncMock(side_effect=[0.8, 0.5])
        correlation_system._calculate_validation_score = MagicMock(side_effect=[0.8, 0.45])
        correlation_system._calculate_overall_validation_statistics = MagicMock(return_value={
            'num_events': 2,
            'mean_validation_score': 0.625,
            'spatial_accuracy': 0.55,
            'temporal_accuracy': 0.75,
            'pattern_accuracy': 0.65
        })
        
        # Test validation
        result = await correlation_system.validate_against_monitored_events(
            correlation_results, monitored_events, validation_params
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert 'num_events' in result
        assert 'mean_validation_score' in result
        assert result['num_events'] == 2
        assert 0 <= result['mean_validation_score'] <= 1
        
        # Verify all validation methods were called
        assert correlation_system._validate_spatial_correlation.call_count == 2
        assert correlation_system._validate_temporal_correlation.call_count == 2
        assert correlation_system._validate_magnitude_correlation.call_count == 2
        correlation_system._validate_deformation_pattern.assert_awaited()
        assert correlation_system._calculate_validation_score.call_count == 2
        correlation_system._calculate_overall_validation_statistics.assert_called_once()
        
        # Verify database storage was called
        correlation_system._store_validation_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_weighted_correlation_calculation(self, correlation_system):
        """Test weighted correlation coefficient calculation"""
        
        # Create test data
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        weights = np.array([1.0, 2.0, 1.5, 1.0, 2.5])
        
        # Calculate weighted correlation
        result = correlation_system._calculate_weighted_correlation(
            observed, predicted, weights
        )
        
        # Verify result
        assert isinstance(result, float)
        assert -1 <= result <= 1
        assert not np.isnan(result)
        
        # Test with perfect correlation
        perfect_predicted = observed.copy()
        perfect_result = correlation_system._calculate_weighted_correlation(
            observed, perfect_predicted, weights
        )
        assert abs(perfect_result - 1.0) < 1e-10
        
        # Test with zero variance
        constant_observed = np.ones(5)
        constant_result = correlation_system._calculate_weighted_correlation(
            constant_observed, predicted, weights
        )
        assert constant_result == 0.0

    def test_correlation_significance_calculation(self, correlation_system):
        """Test statistical significance calculation for correlations"""
        
        # Test with strong correlation
        strong_correlation = 0.8
        large_sample = 100
        significance = correlation_system._calculate_correlation_significance(
            strong_correlation, large_sample
        )
        
        assert 0 <= significance <= 1
        assert significance > 0.95  # Should be highly significant
        
        # Test with weak correlation
        weak_correlation = 0.2
        small_sample = 10
        weak_significance = correlation_system._calculate_correlation_significance(
            weak_correlation, small_sample
        )
        
        assert 0 <= weak_significance <= 1
        assert weak_significance < significance  # Should be less significant
        
        # Test edge cases
        zero_correlation = correlation_system._calculate_correlation_significance(0.0, 50)
        assert zero_correlation < 0.05  # Should not be significant
        
        small_sample_result = correlation_system._calculate_correlation_significance(0.5, 2)
        assert small_sample_result == 0.0  # Too small sample

    @pytest.mark.asyncio
    async def test_database_connection_initialization(self, correlation_system):
        """Test database connection initialization"""
        
        # Mock successful connections
        with patch('asyncpg.create_pool') as mock_postgres, \
             patch('redis.asyncio.Redis') as mock_redis, \
             patch('influxdb_client.client.influxdb_client_async.InfluxDBClientAsync') as mock_influx, \
             patch('neo4j.AsyncGraphDatabase.driver') as mock_neo4j:
            
            mock_postgres.return_value = AsyncMock()
            mock_redis.return_value = AsyncMock()
            mock_influx.return_value = AsyncMock()
            mock_neo4j.return_value = AsyncMock()
            
            # Test initialization
            await correlation_system.initialize_connections()
            
            # Verify all connections were attempted
            mock_postgres.assert_called_once()
            mock_redis.assert_called_once()
            mock_influx.assert_called_once()
            mock_neo4j.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_cleanup(self, correlation_system):
        """Test proper cleanup of database connections"""
        
        # Mock connections
        correlation_system.postgres_pool = AsyncMock()
        correlation_system.redis_client = AsyncMock()
        correlation_system.influx_client = AsyncMock()
        correlation_system.neo4j_driver = AsyncMock()
        
        # Test cleanup
        await correlation_system.close_connections()
        
        # Verify all connections were closed
        correlation_system.postgres_pool.close.assert_called_once()
        correlation_system.redis_client.close.assert_called_once()
        correlation_system.influx_client.close.assert_called_once()
        correlation_system.neo4j_driver.close.assert_called_once()

    def test_sill_model_parameter_conversion(self, correlation_system, sample_sill_model):
        """Test conversion between parameter arrays and sill models"""
        
        # Test params to model conversion
        params = np.array([
            -122.5, 37.8, 2000,  # x, y, depth
            5000, 3000,           # length, width
            45, 10,               # strike, dip
            0.5                   # opening
        ])
        
        converted_model = correlation_system._params_to_sill_model(params, sample_sill_model)
        
        # Verify conversion
        assert converted_model.center_coordinates[0] == params[0]
        assert converted_model.center_coordinates[1] == params[1]
        assert converted_model.center_coordinates[2] == params[2]
        assert converted_model.length == params[3]
        assert converted_model.width == params[4]
        assert converted_model.strike == params[5]
        assert converted_model.dip == params[6]
        assert converted_model.opening == params[7]
        
        # Test with None reference model
        minimal_model = correlation_system._params_to_sill_model(params, None)
        assert minimal_model.sill_id == "inversion_model"
        assert minimal_model.center_coordinates == (params[0], params[1], params[2])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])