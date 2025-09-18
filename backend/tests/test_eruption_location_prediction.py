"""
Comprehensive tests for Eruption Location Prediction System

This test suite validates all components of the advanced eruption location prediction
system including stress field analysis, failure criteria, topographic controls,
probabilistic mapping, temporal evolution, and historical validation.

Author: Prometheus AI Platform
Version: 1.0.0
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Import system components
from backend.modules.eruption_location_prediction_system import (
    EruptionLocationPredictionSystem,
    SillGeometry,
    StressFieldParameters,
    FailureCriterion,
    TopographicControls,
    EruptionPrediction,
    HazardZone
)

class TestEruptionLocationPredictionSystem:
    """Test suite for the main prediction system"""
    
    @pytest.fixture
    async def prediction_system(self):
        """Create a test prediction system instance"""
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_prometheus_volcanic',
            'user': 'test_user',
            'password': 'test_password',
            'redis_host': 'localhost',
            'redis_port': 6379
        }
        
        system = EruptionLocationPredictionSystem(db_config)
        
        # Mock database connections for testing
        system.engine = AsyncMock()
        system.redis_client = AsyncMock()
        
        return system
    
    @pytest.fixture
    def sample_sill_geometries(self):
        """Create sample sill geometries for testing"""
        return [
            SillGeometry(
                sill_id="test_sill_001",
                center_coordinates=(0, 0, -1500),
                length=3000, width=2000, thickness=50,
                dip=15, strike=45, volume=3e8,
                aspect_ratio=1.5, shape_factor=0.8, connectivity_index=0.6
            ),
            SillGeometry(
                sill_id="test_sill_002",
                center_coordinates=(1000, 500, -2000),
                length=2500, width=1800, thickness=40,
                dip=20, strike=60, volume=1.8e8,
                aspect_ratio=1.4, shape_factor=0.7, connectivity_index=0.4
            )
        ]
    
    @pytest.fixture
    def sample_stress_field_params(self):
        """Create sample stress field parameters"""
        return StressFieldParameters(
            principal_stress_1=50e6,  # 50 MPa
            principal_stress_2=30e6,  # 30 MPa
            principal_stress_3=10e6,  # 10 MPa
            stress_orientation=(45, 30, 0),
            differential_stress=40e6,
            mean_stress=30e6,
            stress_ratio=0.5,
            deviatoric_stress=np.array([[20e6, 0, 0], [0, 0, 0], [0, 0, -20e6]]),
            stress_invariants=(90e6, 400e12, 8e18),
            von_mises_stress=35e6
        )

class TestStressFieldAnalysis:
    """Test stress field analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_stress_field_calculation(self, prediction_system, sample_sill_geometries, sample_stress_field_params):
        """Test stress field calculation around sill complexes"""
        
        # Mock the database storage method
        prediction_system._store_stress_field_analysis = AsyncMock()
        
        # Mock the individual stress calculation methods
        prediction_system._calculate_sill_induced_stress = AsyncMock(return_value={
            'xx': np.ones((10, 10, 5)) * 1e6,
            'yy': np.ones((10, 10, 5)) * 1e6,
            'zz': np.ones((10, 10, 5)) * 1e6,
            'xy': np.zeros((10, 10, 5)),
            'xz': np.zeros((10, 10, 5)),
            'yz': np.zeros((10, 10, 5))
        })
        
        prediction_system._calculate_principal_stresses = AsyncMock(return_value={
            'sigma1': np.ones((10, 10, 5)) * 50e6,
            'sigma2': np.ones((10, 10, 5)) * 30e6,
            'sigma3': np.ones((10, 10, 5)) * 10e6
        })
        
        prediction_system._calculate_von_mises_stress = AsyncMock(return_value=np.ones((10, 10, 5)) * 35e6)
        prediction_system._calculate_stress_gradients = AsyncMock(return_value=np.ones((10, 10, 5, 3)) * 1e3)
        
        # Test the stress field analysis
        result = await prediction_system.analyze_stress_field_around_sill_complex(
            sample_sill_geometries, sample_stress_field_params, (10, 10, 5)
        )
        
        # Verify results structure
        assert 'grid_coordinates' in result
        assert 'stress_tensor' in result
        assert 'principal_stresses' in result
        assert 'von_mises_stress' in result
        assert 'stress_gradients' in result
        assert 'analysis_timestamp' in result
        
        # Verify database storage was called
        prediction_system._store_stress_field_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sill_induced_stress_calculation(self, prediction_system):
        """Test individual sill stress calculation"""
        
        sill = SillGeometry(
            sill_id="test_sill",
            center_coordinates=(0, 0, -1000),
            length=2000, width=1500, thickness=30,
            dip=10, strike=30, volume=9e7,
            aspect_ratio=1.33, shape_factor=0.75, connectivity_index=0.5
        )
        
        # Create test grid
        X, Y, Z = np.meshgrid(
            np.linspace(-1000, 1000, 5),
            np.linspace(-1000, 1000, 5),
            np.linspace(-500, 0, 3),
            indexing='ij'
        )
        
        result = await prediction_system._calculate_sill_induced_stress(sill, X, Y, Z)
        
        # Verify result structure
        assert all(key in result for key in ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'])
        assert all(isinstance(result[key], np.ndarray) for key in result)
        assert all(result[key].shape == X.shape for key in result)
        
        # Verify stress values are reasonable (positive and finite)
        for key in result:
            assert np.all(np.isfinite(result[key]))
            assert np.all(result[key] >= 0)  # Compressive stress convention

class TestFailureCriteriaAnalysis:
    """Test failure criteria analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_mohr_coulomb_failure(self, prediction_system):
        """Test Mohr-Coulomb failure criterion calculation"""
        
        # Create test principal stresses
        principal_stresses = {
            'sigma1': np.array([60e6, 40e6, 80e6]),  # Various stress levels
            'sigma2': np.array([40e6, 30e6, 50e6]),
            'sigma3': np.array([20e6, 10e6, 30e6])
        }
        
        cohesion = 10e6  # 10 MPa
        friction_angle = np.radians(30)  # 30 degrees
        
        result = await prediction_system._calculate_mohr_coulomb_failure(
            principal_stresses, cohesion, friction_angle
        )
        
        # Verify result properties
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.all(result >= 0)  # Failure probability should be non-negative
        assert np.all(result <= 1)  # Failure probability should not exceed 1
        
        # Higher differential stress should give higher failure probability
        diff_stress = principal_stresses['sigma1'] - principal_stresses['sigma3']
        assert result[2] > result[1]  # Highest differential stress case
    
    @pytest.mark.asyncio
    async def test_griffith_failure(self, prediction_system):
        """Test Griffith failure criterion calculation"""
        
        # Test both tensile and compressive conditions
        principal_stresses = {
            'sigma1': np.array([10e6, -5e6, 50e6]),  # Mixed conditions
            'sigma2': np.array([5e6, -3e6, 30e6]),
            'sigma3': np.array([0, -8e6, 10e6])  # Including tensile stress
        }
        
        tensile_strength = 5e6  # 5 MPa
        
        result = await prediction_system._calculate_griffith_failure(
            principal_stresses, tensile_strength
        )
        
        # Verify result properties
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        
        # Tensile condition (index 1) should show failure
        assert result[1] > 0.5  # Should indicate high failure probability
    
    @pytest.mark.asyncio
    async def test_complete_failure_analysis(self, prediction_system):
        """Test complete failure criterion analysis"""
        
        # Mock stress field data
        stress_field = {
            'stress_tensor': {
                'xx': np.ones((5, 5, 3)) * 40e6,
                'yy': np.ones((5, 5, 3)) * 30e6,
                'zz': np.ones((5, 5, 3)) * 20e6,
                'xy': np.zeros((5, 5, 3)),
                'xz': np.zeros((5, 5, 3)),
                'yz': np.zeros((5, 5, 3))
            },
            'principal_stresses': {
                'sigma1': np.ones((5, 5, 3)) * 40e6,
                'sigma2': np.ones((5, 5, 3)) * 30e6,
                'sigma3': np.ones((5, 5, 3)) * 20e6
            },
            'grid_coordinates': {
                'x': np.ones((5, 5, 3)),
                'y': np.ones((5, 5, 3)),
                'z': np.ones((5, 5, 3))
            }
        }
        
        rock_properties = {
            'cohesion': 10e6,
            'friction_angle': np.radians(30),
            'tensile_strength': 5e6,
            'ucs': 100e6
        }
        
        # Mock helper methods
        prediction_system._calculate_mohr_coulomb_failure = AsyncMock(return_value=np.ones((5, 5, 3)) * 0.3)
        prediction_system._calculate_griffith_failure = AsyncMock(return_value=np.ones((5, 5, 3)) * 0.2)
        prediction_system._calculate_hoek_brown_failure = AsyncMock(return_value=np.ones((5, 5, 3)) * 0.25)
        prediction_system._calculate_drucker_prager_failure = AsyncMock(return_value=np.ones((5, 5, 3)) * 0.35)
        prediction_system._calculate_failure_gradient = AsyncMock(return_value=np.ones((5, 5, 3, 3)) * 0.1)
        prediction_system._identify_eruption_pathways = AsyncMock(return_value=[])
        prediction_system._find_critical_stress_locations = AsyncMock(return_value=[])
        prediction_system._store_failure_analysis = AsyncMock()
        
        result = await prediction_system.analyze_failure_criteria(stress_field, rock_properties)
        
        # Verify result structure
        assert 'mohr_coulomb_failure' in result
        assert 'griffith_failure' in result
        assert 'hoek_brown_failure' in result
        assert 'drucker_prager_failure' in result
        assert 'combined_failure_probability' in result
        assert 'failure_gradient' in result
        assert 'eruption_pathways' in result
        assert 'analysis_timestamp' in result
        
        # Verify database storage was called
        prediction_system._store_failure_analysis.assert_called_once()

class TestTopographicAnalysis:
    """Test topographic and structural controls analysis"""
    
    @pytest.mark.asyncio
    async def test_topographic_analysis(self, prediction_system):
        """Test topographic and structural controls analysis"""
        
        analysis_area = (-5000, -5000, 5000, 5000)  # 10km x 10km area
        dem_resolution = 30.0
        
        # Mock all the helper methods
        prediction_system._load_dem_data = AsyncMock(return_value={
            'elevation': np.random.rand(100, 100) * 1000
        })
        prediction_system._calculate_slope = AsyncMock(return_value=np.random.rand(100, 100) * 45)
        prediction_system._calculate_aspect = AsyncMock(return_value=np.random.rand(100, 100) * 360)
        prediction_system._calculate_curvature = AsyncMock(return_value=np.random.rand(100, 100) * 0.01)
        prediction_system._calculate_roughness = AsyncMock(return_value=np.random.rand(100, 100) * 10)
        prediction_system._load_structural_data = AsyncMock(return_value={
            'faults': [], 'fractures': [], 'lineaments': []
        })
        prediction_system._calculate_fault_density = AsyncMock(return_value=np.random.rand(100, 100) * 5)
        prediction_system._calculate_fracture_density = AsyncMock(return_value=np.random.rand(100, 100) * 10)
        prediction_system._calculate_lineament_density = AsyncMock(return_value=np.random.rand(100, 100) * 3)
        prediction_system._extract_drainage_network = AsyncMock(return_value=[])
        prediction_system._calculate_drainage_density = AsyncMock(return_value=np.random.rand(100, 100) * 2)
        prediction_system._calculate_topographic_stress = AsyncMock(return_value=np.random.rand(100, 100) * 1e6)
        prediction_system._analyze_lineament_orientations = AsyncMock(return_value=[])
        prediction_system._calculate_structural_control_index = AsyncMock(return_value=np.random.rand(100, 100))
        prediction_system._calculate_topographic_favorability = AsyncMock(return_value=np.random.rand(100, 100))
        prediction_system._store_topographic_analysis = AsyncMock()
        
        result = await prediction_system.analyze_topographic_structural_controls(
            analysis_area, dem_resolution
        )
        
        # Verify result structure
        assert 'dem_data' in result
        assert 'structural_data' in result
        assert 'drainage_network' in result
        assert 'topographic_stress' in result
        assert 'structural_control_index' in result
        assert 'topographic_favorability' in result
        assert 'analysis_parameters' in result
        
        # Verify all helper methods were called
        prediction_system._load_dem_data.assert_called_once()
        prediction_system._calculate_slope.assert_called_once()
        prediction_system._store_topographic_analysis.assert_called_once()

class TestProbabilisticMapping:
    """Test probabilistic eruption location mapping"""
    
    @pytest.mark.asyncio
    async def test_probabilistic_mapping(self, prediction_system, sample_sill_geometries):
        """Test probabilistic eruption location mapping"""
        
        # Mock input data
        stress_field = {
            'grid_coordinates': {
                'x': np.random.rand(50, 50, 10),
                'y': np.random.rand(50, 50, 10),
                'z': np.random.rand(50, 50, 10)
            },
            'von_mises_stress': np.random.rand(50, 50, 10) * 50e6
        }
        
        failure_analysis = {
            'combined_failure_probability': np.random.rand(50, 50, 10)
        }
        
        topographic_controls = {
            'topographic_favorability': np.random.rand(100, 100),
            'structural_control_index': np.random.rand(100, 100),
            'dem_data': {'elevation': np.random.rand(100, 100) * 1000}
        }
        
        # Mock helper methods
        prediction_system._interpolate_to_grid = AsyncMock(return_value=np.random.rand(1000))
        prediction_system._calculate_sill_proximity_factor = AsyncMock(return_value=np.random.rand(1000))
        prediction_system._apply_ml_enhancement = AsyncMock(return_value=np.random.rand(1000))
        prediction_system._calculate_uncertainty_bounds = AsyncMock(return_value=(np.random.rand(1000), np.random.rand(1000)))
        prediction_system._identify_high_probability_zones = AsyncMock(return_value=[])
        prediction_system._create_probability_contours = AsyncMock(return_value=[])
        prediction_system._store_probabilistic_mapping = AsyncMock()
        
        result = await prediction_system.create_probabilistic_eruption_location_mapping(
            stress_field, failure_analysis, topographic_controls, sample_sill_geometries
        )
        
        # Verify result structure
        assert 'surface_coordinates' in result
        assert 'eruption_probability' in result
        assert 'uncertainty_bounds' in result
        assert 'contributing_factors' in result
        assert 'high_probability_zones' in result
        assert 'probability_contours' in result
        assert 'analysis_weights' in result
        assert 'confidence_level' in result
        
        # Verify database storage was called
        prediction_system._store_probabilistic_mapping.assert_called_once()

class TestTemporalEvolution:
    """Test temporal evolution modeling"""
    
    @pytest.mark.asyncio
    async def test_temporal_evolution(self, prediction_system, sample_sill_geometries):
        """Test temporal evolution of eruption potential"""
        
        growth_rates = {'test_sill_001': 1000.0, 'test_sill_002': 800.0}
        time_horizon = 365.0
        
        # Mock helper methods
        prediction_system._calculate_stress_evolution = AsyncMock(return_value={})
        prediction_system._calculate_failure_evolution = AsyncMock(return_value={})
        prediction_system._calculate_eruption_potential_evolution = AsyncMock(return_value={})
        prediction_system._identify_critical_events = AsyncMock(return_value=[])
        prediction_system._generate_eruption_forecasts = AsyncMock(return_value={})
        prediction_system._store_temporal_evolution = AsyncMock()
        
        result = await prediction_system.model_temporal_evolution_eruption_potential(
            sample_sill_geometries, growth_rates, time_horizon
        )
        
        # Verify result structure
        assert 'time_steps' in result
        assert 'sill_evolution' in result
        assert 'stress_evolution' in result
        assert 'failure_evolution' in result
        assert 'eruption_potential_evolution' in result
        assert 'critical_events' in result
        assert 'eruption_forecasts' in result
        
        # Verify time steps are reasonable
        assert len(result['time_steps']) > 0
        assert result['time_steps'][0] == 0
        assert result['time_steps'][-1] <= time_horizon
        
        # Verify sill evolution data
        for sill in sample_sill_geometries:
            assert sill.sill_id in result['sill_evolution']
            sill_data = result['sill_evolution'][sill.sill_id]
            assert 'volume' in sill_data
            assert 'pressure' in sill_data
            assert 'thickness' in sill_data
            assert 'lateral_extent' in sill_data

class TestEruptionStylePrediction:
    """Test eruption style prediction"""
    
    @pytest.mark.asyncio
    async def test_eruption_style_prediction(self, prediction_system, sample_sill_geometries):
        """Test eruption style prediction based on sill characteristics"""
        
        magma_properties = {
            'viscosity': 1e4,
            'gas_content': 0.03,
            'temperature': 1200,
            'silica_content': 0.65,
            'density': 2500
        }
        
        stress_field = {'von_mises_stress': np.random.rand(10, 10, 5) * 50e6}
        
        # Mock helper methods
        prediction_system._estimate_discharge_rate = AsyncMock(return_value=100.0)
        prediction_system._estimate_column_height = AsyncMock(return_value=15000.0)
        prediction_system._estimate_eruption_duration = AsyncMock(return_value=24.0)
        prediction_system._store_eruption_style_predictions = AsyncMock()
        
        result = await prediction_system.predict_eruption_style(
            sample_sill_geometries, magma_properties, stress_field
        )
        
        # Verify result structure
        for sill in sample_sill_geometries:
            assert sill.sill_id in result
            prediction = result[sill.sill_id]
            
            assert 'eruption_style' in prediction
            assert 'vei_estimate' in prediction
            assert 'explosive_potential' in prediction
            assert 'discharge_rate' in prediction
            assert 'column_height' in prediction
            assert 'duration_estimate' in prediction
            assert 'contributing_factors' in prediction
            assert 'confidence' in prediction
            assert 'uncertainty_range' in prediction
            
            # Verify reasonable values
            assert prediction['vei_estimate'] >= 0
            assert prediction['vei_estimate'] <= 7
            assert 0 <= prediction['explosive_potential'] <= 1
            assert 0 <= prediction['confidence'] <= 1
            assert prediction['eruption_style'] in ['Effusive', 'Strombolian', 'Vulcanian', 'Sub-Plinian', 'Plinian']

class TestHazardZoneMapping:
    """Test hazard zone mapping functionality"""
    
    @pytest.mark.asyncio
    async def test_hazard_zone_creation(self, prediction_system):
        """Test hazard zone mapping creation"""
        
        # Mock input data
        probabilistic_mapping = {
            'high_probability_zones': [
                {'center': (0, 0), 'probability': 0.8, 'area': 1000000}
            ]
        }
        
        eruption_style_predictions = {
            'test_sill_001': {
                'eruption_style': 'Vulcanian',
                'vei_estimate': 3,
                'explosive_potential': 0.6
            }
        }
        
        population_data = {'total': 50000, 'density': 200}
        infrastructure_data = ['hospital', 'school', 'airport']
        
        # Mock helper methods
        prediction_system._determine_dominant_eruption_style = AsyncMock(return_value={
            'eruption_style': 'Vulcanian',
            'vei_estimate': 3
        })
        prediction_system._calculate_hazard_parameters = AsyncMock(return_value={
            'hazard_level': 'High',
            'impact_radius': 5000,
            'type_probability': 0.7
        })
        prediction_system._create_hazard_geometry = AsyncMock(return_value=Mock())
        prediction_system._assess_population_at_risk = AsyncMock(return_value={'total': 10000})
        prediction_system._assess_infrastructure_at_risk = AsyncMock(return_value=['hospital'])
        prediction_system._calculate_evacuation_time = AsyncMock(return_value=6.0)
        prediction_system._calculate_monitoring_priority = AsyncMock(return_value=4)
        prediction_system._create_composite_hazard_map = AsyncMock(return_value={})
        prediction_system._calculate_risk_metrics = AsyncMock(return_value={})
        prediction_system._generate_evacuation_plans = AsyncMock(return_value={})
        prediction_system._create_monitoring_recommendations = AsyncMock(return_value={})
        prediction_system._store_hazard_mapping = AsyncMock()
        
        result = await prediction_system.create_hazard_zone_mapping(
            probabilistic_mapping, eruption_style_predictions, 
            population_data, infrastructure_data
        )
        
        # Verify result structure
        assert 'hazard_zones' in result
        assert 'composite_hazard_map' in result
        assert 'risk_metrics' in result
        assert 'evacuation_plans' in result
        assert 'monitoring_recommendations' in result
        
        # Verify hazard zones have required fields
        if result['hazard_zones']:
            hazard_zone = result['hazard_zones'][0]
            assert 'zone_id' in hazard_zone
            assert 'hazard_level' in hazard_zone
            assert 'hazard_type' in hazard_zone
            assert 'probability' in hazard_zone
            assert 'population_at_risk' in hazard_zone

class TestHistoricalValidation:
    """Test historical validation functionality"""
    
    @pytest.mark.asyncio
    async def test_historical_validation(self, prediction_system):
        """Test validation against historical eruptions"""
        
        prediction_results = {
            'surface_coordinates': {'x': np.random.rand(100), 'y': np.random.rand(100)},
            'eruption_probability': np.random.rand(100),
            'eruption_forecasts': {},
            'eruption_style_predictions': {}
        }
        
        # Mock helper methods
        prediction_system._load_historical_eruption_data = AsyncMock(return_value=[])
        prediction_system._filter_sill_related_eruptions = AsyncMock(return_value=[
            {'eruption_id': '1', 'latitude': 0.1, 'longitude': 0.1, 'vei': 3, 'eruption_style': 'Vulcanian'}
        ])
        prediction_system._validate_spatial_accuracy = AsyncMock(return_value={'accuracy': 0.75})
        prediction_system._validate_probability_calibration = AsyncMock(return_value={'calibration_score': 0.8})
        prediction_system._validate_temporal_accuracy = AsyncMock(return_value={'temporal_score': 0.7})
        prediction_system._validate_style_predictions = AsyncMock(return_value={'style_accuracy': 0.85})
        prediction_system._calculate_overall_performance = AsyncMock(return_value={'composite_score': 0.78})
        prediction_system._generate_validation_report = AsyncMock(return_value="Validation report")
        prediction_system._generate_model_improvement_recommendations = AsyncMock(return_value=[])
        prediction_system._store_validation_results = AsyncMock()
        
        result = await prediction_system.validate_against_historical_eruptions(prediction_results)
        
        # Verify result structure
        assert 'validation_metrics' in result
        assert 'historical_eruptions_used' in result
        assert 'validation_report' in result
        assert 'model_performance_score' in result
        assert 'recommendations' in result
        
        # Verify validation metrics structure
        validation_metrics = result['validation_metrics']
        assert 'spatial_accuracy' in validation_metrics
        assert 'probability_calibration' in validation_metrics
        assert 'overall_performance' in validation_metrics
        
        # Verify performance score is reasonable
        assert 0 <= result['model_performance_score'] <= 1

# Integration tests
class TestSystemIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_complete_prediction_workflow(self, prediction_system, sample_sill_geometries, sample_stress_field_params):
        """Test complete prediction workflow integration"""
        
        # Mock all database operations
        prediction_system._store_stress_field_analysis = AsyncMock()
        prediction_system._store_failure_analysis = AsyncMock()
        prediction_system._store_topographic_analysis = AsyncMock()
        prediction_system._store_probabilistic_mapping = AsyncMock()
        prediction_system._store_temporal_evolution = AsyncMock()
        prediction_system._store_eruption_style_predictions = AsyncMock()
        prediction_system._store_hazard_mapping = AsyncMock()
        prediction_system._store_validation_results = AsyncMock()
        
        # Mock all complex calculations to return reasonable test data
        with patch.multiple(
            prediction_system,
            _calculate_sill_induced_stress=AsyncMock(return_value={
                'xx': np.ones((10, 10, 5)) * 1e6, 'yy': np.ones((10, 10, 5)) * 1e6,
                'zz': np.ones((10, 10, 5)) * 1e6, 'xy': np.zeros((10, 10, 5)),
                'xz': np.zeros((10, 10, 5)), 'yz': np.zeros((10, 10, 5))
            }),
            _calculate_principal_stresses=AsyncMock(return_value={
                'sigma1': np.ones((10, 10, 5)) * 50e6,
                'sigma2': np.ones((10, 10, 5)) * 30e6,
                'sigma3': np.ones((10, 10, 5)) * 10e6
            }),
            _calculate_von_mises_stress=AsyncMock(return_value=np.ones((10, 10, 5)) * 35e6),
            _calculate_stress_gradients=AsyncMock(return_value=np.ones((10, 10, 5, 3)) * 1e3),
            _load_dem_data=AsyncMock(return_value={'elevation': np.random.rand(50, 50) * 1000}),
            _calculate_slope=AsyncMock(return_value=np.random.rand(50, 50) * 45),
            _calculate_aspect=AsyncMock(return_value=np.random.rand(50, 50) * 360),
            _load_structural_data=AsyncMock(return_value={'faults': [], 'fractures': [], 'lineaments': []}),
            _interpolate_to_grid=AsyncMock(return_value=np.random.rand(100)),
            _apply_ml_enhancement=AsyncMock(return_value=np.random.rand(100)),
            _load_historical_eruption_data=AsyncMock(return_value=[]),
            _filter_sill_related_eruptions=AsyncMock(return_value=[])
        ):
            
            # Run complete workflow (simplified version)
            stress_results = await prediction_system.analyze_stress_field_around_sill_complex(
                sample_sill_geometries, sample_stress_field_params, (10, 10, 5)
            )
            
            assert stress_results is not None
            assert 'analysis_timestamp' in stress_results
            
            # Verify that the workflow completed without errors
            prediction_system._store_stress_field_analysis.assert_called_once()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])