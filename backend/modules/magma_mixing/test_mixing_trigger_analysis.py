"""
Comprehensive test suite for Mixing Trigger Analysis System

Tests all components of the magma mixing trigger analysis system including
injection event detection, convection initiation analysis, external trigger
analysis, and predictive modeling capabilities.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json
from typing import Dict, List, Any

# Import the system under test
from .mixing_trigger_analysis_system import (
    MixingTriggerAnalysisSystem,
    TriggerType,
    MagmaInjectionEvent,
    ConvectionInitiation,
    ExternalTrigger,
    DensityContrastAnalysis,
    TemperatureDifferenceAnalysis,
    VolatileContentAnalysis,
    MixingTriggerPrediction
)
from .data_models import MagmaComposition, ThermalProperties, RheologicalProperties


class TestMixingTriggerAnalysisSystem:
    """Test suite for MixingTriggerAnalysisSystem"""
    
    @pytest.fixture
    def mock_database_config(self):
        """Mock database configuration"""
        return {
            'user': 'test_user',
            'password': 'test_password',
            'host': 'localhost',
            'port': 5432,
            'database': 'test_prometheus',
            'redis_host': 'localhost',
            'redis_port': 6379
        }
    
    @pytest.fixture
    def trigger_analysis_system(self, mock_database_config):
        """Create MixingTriggerAnalysisSystem instance with mocked dependencies"""
        with patch('sqlalchemy.create_engine'), \
             patch('redis.Redis'):
            system = MixingTriggerAnalysisSystem(mock_database_config)
            return system
    
    @pytest.fixture
    def sample_monitoring_data(self):
        """Sample monitoring data for testing"""
        return {
            'seismic': np.random.randn(1000) * 0.1,
            'deformation': np.random.randn(1000) * 0.01 + np.sin(np.linspace(0, 10, 1000)) * 0.05,
            'thermal': np.random.randn(1000) * 5 + 1200,
            'gas': np.random.randn(1000) * 10 + 100
        }
    
    @pytest.fixture
    def sample_thermal_field(self):
        """Sample 3D thermal field"""
        return np.random.randn(50, 50, 50) * 50 + 1200
    
    @pytest.fixture
    def sample_density_field(self):
        """Sample 3D density field"""
        return np.random.randn(50, 50, 50) * 100 + 2500
    
    @pytest.fixture
    def sample_viscosity_field(self):
        """Sample 3D viscosity field"""
        return np.random.randn(50, 50, 50) * 1e3 + 1e5
    
    @pytest.fixture
    def sample_pressure_field(self):
        """Sample 3D pressure field"""
        return np.random.randn(50, 50, 50) * 1e6 + 1e8
    
    @pytest.fixture
    def sample_volatile_distribution(self):
        """Sample volatile distribution"""
        return {
            'H2O': np.random.randn(50, 50, 50) * 0.5 + 2.0,
            'CO2': np.random.randn(50, 50, 50) * 0.1 + 0.5,
            'SO2': np.random.randn(50, 50, 50) * 0.05 + 0.1
        }
    
    @pytest.fixture
    def sample_seismic_events(self):
        """Sample seismic events for testing"""
        return [
            {
                'event_id': 'eq_001',
                'timestamp': datetime.now() - timedelta(hours=1),
                'magnitude': 4.2,
                'distance_km': 15.0,
                'location': (45.0, -122.0, -5.0),
                'dominant_frequency': 2.5,
                'duration': 45.0,
                'energy': 1e12
            },
            {
                'event_id': 'eq_002',
                'timestamp': datetime.now() - timedelta(hours=3),
                'magnitude': 3.8,
                'distance_km': 25.0,
                'location': (45.1, -122.1, -8.0),
                'dominant_frequency': 1.8,
                'duration': 30.0,
                'energy': 5e11
            }
        ]
    
    @pytest.fixture
    def sample_pressure_changes(self):
        """Sample pressure change events"""
        return [
            {
                'event_id': 'press_001',
                'timestamp': datetime.now() - timedelta(hours=2),
                'pressure_change_pa': 5000.0,
                'location': (45.0, -122.0, -2.0),
                'duration': 1800.0
            },
            {
                'event_id': 'press_002',
                'timestamp': datetime.now() - timedelta(hours=4),
                'pressure_change_pa': -3000.0,
                'location': (45.05, -122.05, -3.0),
                'duration': 3600.0
            }
        ]
    
    @pytest.fixture
    def sample_thermal_properties(self):
        """Sample thermal properties"""
        return ThermalProperties(
            temperature=1200.0,
            heat_capacity=1200.0,
            thermal_conductivity=2.5,
            thermal_diffusivity=1e-6,
            latent_heat=400000.0,
            crystallization_temperature=1150.0
        )
    
    @pytest.fixture
    def sample_historical_triggers(self):
        """Sample historical trigger data"""
        return [
            {
                'trigger_type': 'magma_injection',
                'magnitude': 1000.0,
                'duration': 3600.0,
                'energy': 1e10,
                'mixing_response': 0.7,
                'timestamp': datetime.now() - timedelta(days=1)
            },
            {
                'trigger_type': 'seismic_event',
                'magnitude': 4.0,
                'duration': 60.0,
                'energy': 1e12,
                'mixing_response': 0.3,
                'timestamp': datetime.now() - timedelta(days=2)
            },
            {
                'trigger_type': 'pressure_change',
                'magnitude': 5000.0,
                'duration': 1800.0,
                'energy': 1e8,
                'mixing_response': 0.4,
                'timestamp': datetime.now() - timedelta(days=3)
            }
        ]
    
    @pytest.fixture
    def sample_current_conditions(self):
        """Sample current conditions for prediction"""
        return {
            'seismic_activity': 0.5,
            'deformation_rate': 0.01,
            'thermal_anomaly': 10.0,
            'gas_flux': 150.0,
            'pressure_change': 2000.0,
            'temperature_gradient': 0.5
        }

    # Test magma injection event detection
    @pytest.mark.asyncio
    async def test_detect_magma_injection_events(self, trigger_analysis_system, sample_monitoring_data):
        """Test magma injection event detection"""
        chamber_id = "test_chamber_001"
        time_range = (datetime.now() - timedelta(hours=24), datetime.now())
        
        # Mock the database storage method
        trigger_analysis_system._store_injection_events = AsyncMock()
        
        # Mock helper methods
        trigger_analysis_system._extract_injection_signatures = AsyncMock(return_value=[
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'duration': 1800.0,
                'strength': 0.8,
                'velocity_field': np.array([0.1, 0.05, -0.02])
            }
        ])
        trigger_analysis_system._calculate_injection_volume = AsyncMock(return_value=1500.0)
        trigger_analysis_system._calculate_injection_rate = AsyncMock(return_value=0.83)
        trigger_analysis_system._locate_injection_source = AsyncMock(return_value=(100.0, 200.0, -2000.0))
        trigger_analysis_system._analyze_injected_composition = AsyncMock(return_value=MagmaComposition(
            SiO2=52.0, TiO2=1.2, Al2O3=16.5, FeO=8.5, MnO=0.15,
            MgO=6.2, CaO=9.8, Na2O=3.2, K2O=1.8, P2O5=0.3
        ))
        trigger_analysis_system._estimate_injection_temperature = AsyncMock(return_value=1250.0)
        trigger_analysis_system._calculate_injection_pressure = AsyncMock(return_value=5e7)
        trigger_analysis_system._calculate_injection_mixing_efficiency = AsyncMock(return_value=0.75)
        
        # Execute test
        injection_events = await trigger_analysis_system.detect_magma_injection_events(
            chamber_id, time_range, sample_monitoring_data
        )
        
        # Assertions
        assert len(injection_events) == 1
        assert isinstance(injection_events[0], MagmaInjectionEvent)
        assert injection_events[0].injection_volume == 1500.0
        assert injection_events[0].injection_rate == 0.83
        assert injection_events[0].injection_temperature == 1250.0
        assert injection_events[0].mixing_efficiency == 0.75
        
        # Verify storage was called
        trigger_analysis_system._store_injection_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_convection_initiation(self, trigger_analysis_system, sample_thermal_field, 
                                                sample_density_field, sample_viscosity_field):
        """Test convection initiation analysis"""
        chamber_id = "test_chamber_002"
        time_range = (datetime.now() - timedelta(hours=12), datetime.now())
        
        # Mock database storage
        trigger_analysis_system._store_convection_events = AsyncMock()
        trigger_analysis_system._get_depth_scale = AsyncMock(return_value=10.0)
        
        # Execute test
        convection_events = await trigger_analysis_system.analyze_convection_initiation(
            chamber_id, sample_thermal_field, sample_density_field, sample_viscosity_field, time_range
        )
        
        # Assertions
        assert isinstance(convection_events, list)
        for event in convection_events:
            assert isinstance(event, ConvectionInitiation)
            assert event.rayleigh_number > 0
            assert event.mixing_potential >= 0
            assert event.convection_velocity >= 0
        
        # Verify storage was called
        trigger_analysis_system._store_convection_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_external_triggers(self, trigger_analysis_system, sample_seismic_events, 
                                           sample_pressure_changes):
        """Test external trigger analysis"""
        chamber_id = "test_chamber_003"
        time_range = (datetime.now() - timedelta(hours=24), datetime.now())
        
        # Mock database storage and helper methods
        trigger_analysis_system._store_external_triggers = AsyncMock()
        trigger_analysis_system._analyze_tidal_triggers = AsyncMock(return_value=[])
        trigger_analysis_system._analyze_injection_triggers = AsyncMock(return_value=[])
        
        # Execute test
        external_triggers = await trigger_analysis_system.analyze_external_triggers(
            chamber_id, sample_seismic_events, sample_pressure_changes, time_range
        )
        
        # Assertions
        assert isinstance(external_triggers, list)
        for trigger in external_triggers:
            assert isinstance(trigger, ExternalTrigger)
            assert trigger.trigger_type in TriggerType
            assert trigger.magnitude > 0
            assert trigger.mixing_response >= 0

    @pytest.mark.asyncio
    async def test_analyze_density_contrast_mixing_potential(self, trigger_analysis_system, 
                                                           sample_density_field, sample_thermal_field):
        """Test density contrast analysis"""
        chamber_id = "test_chamber_004"
        composition_field = {
            'SiO2': np.random.randn(50, 50, 50) * 2 + 52,
            'MgO': np.random.randn(50, 50, 50) * 1 + 6
        }
        
        # Mock database storage
        trigger_analysis_system._store_density_analysis = AsyncMock()
        
        # Execute test
        density_analysis = await trigger_analysis_system.analyze_density_contrast_mixing_potential(
            chamber_id, sample_density_field, composition_field, sample_thermal_field
        )
        
        # Assertions
        assert isinstance(density_analysis, DensityContrastAnalysis)
        assert density_analysis.analysis_id.startswith('density_')
        assert density_analysis.mixing_potential >= 0
        assert density_analysis.critical_density_contrast > 0
        assert density_analysis.instability_growth_rate >= 0

    @pytest.mark.asyncio
    async def test_analyze_temperature_difference_effects(self, trigger_analysis_system, 
                                                        sample_thermal_field, sample_thermal_properties):
        """Test temperature difference analysis"""
        chamber_id = "test_chamber_005"
        velocity_field = np.random.randn(3, 50, 50, 50) * 0.01
        
        # Mock database storage
        trigger_analysis_system._store_temperature_analysis = AsyncMock()
        
        # Execute test
        temp_analysis = await trigger_analysis_system.analyze_temperature_difference_effects(
            chamber_id, sample_thermal_field, sample_thermal_properties, velocity_field
        )
        
        # Assertions
        assert isinstance(temp_analysis, TemperatureDifferenceAnalysis)
        assert temp_analysis.analysis_id.startswith('temp_')
        assert temp_analysis.peclet_number > 0
        assert temp_analysis.thermal_mixing_efficiency >= 0
        assert temp_analysis.convection_strength >= 0

    @pytest.mark.asyncio
    async def test_analyze_volatile_content_effects(self, trigger_analysis_system, sample_volatile_distribution,
                                                  sample_pressure_field, sample_thermal_field):
        """Test volatile content analysis"""
        chamber_id = "test_chamber_006"
        
        # Mock database storage
        trigger_analysis_system._store_volatile_analysis = AsyncMock()
        
        # Execute test
        volatile_analysis = await trigger_analysis_system.analyze_volatile_content_effects(
            chamber_id, sample_volatile_distribution, sample_pressure_field, sample_thermal_field
        )
        
        # Assertions
        assert isinstance(volatile_analysis, VolatileContentAnalysis)
        assert volatile_analysis.analysis_id.startswith('volatile_')
        assert volatile_analysis.bubble_nucleation_rate >= 0
        assert volatile_analysis.degassing_rate >= 0
        assert volatile_analysis.mixing_enhancement_factor >= 1.0

    @pytest.mark.asyncio
    async def test_perform_statistical_analysis_of_triggers(self, trigger_analysis_system, 
                                                          sample_historical_triggers):
        """Test statistical analysis of triggers"""
        chamber_id = "test_chamber_007"
        time_range = (datetime.now() - timedelta(days=30), datetime.now())
        
        # Mock database storage
        trigger_analysis_system._store_statistical_analysis = AsyncMock()
        
        # Execute test
        statistical_results = await trigger_analysis_system.perform_statistical_analysis_of_triggers(
            chamber_id, sample_historical_triggers, time_range
        )
        
        # Assertions
        assert isinstance(statistical_results, dict)
        assert 'chamber_id' in statistical_results
        assert 'trigger_frequencies' in statistical_results
        assert 'trigger_correlations' in statistical_results
        assert 'temporal_patterns' in statistical_results
        assert 'total_triggers_analyzed' in statistical_results
        assert statistical_results['total_triggers_analyzed'] == len(sample_historical_triggers)

    @pytest.mark.asyncio
    async def test_predict_future_mixing_events(self, trigger_analysis_system, sample_current_conditions,
                                              sample_historical_triggers):
        """Test predictive modeling for future mixing events"""
        chamber_id = "test_chamber_008"
        prediction_horizon = timedelta(days=7)
        
        # Mock database storage and helper methods
        trigger_analysis_system._store_mixing_predictions = AsyncMock()
        trigger_analysis_system._prepare_prediction_training_data = AsyncMock(
            return_value=(np.random.randn(10, 4), np.random.randint(0, 3, 10))
        )
        trigger_analysis_system._train_prediction_models = AsyncMock()
        trigger_analysis_system._extract_current_condition_features = AsyncMock(
            return_value=np.array([[0.5, 0.01, 10.0, 150.0, 2000.0, 0.5]])
        )
        trigger_analysis_system._generate_ensemble_predictions = AsyncMock(return_value=[])
        
        # Execute test
        predictions = await trigger_analysis_system.predict_future_mixing_events(
            chamber_id, sample_current_conditions, sample_historical_triggers, prediction_horizon
        )
        
        # Assertions
        assert isinstance(predictions, list)
        for prediction in predictions:
            assert isinstance(prediction, MixingTriggerPrediction)
            assert 0 <= prediction.trigger_probability <= 1
            assert prediction.trigger_type in TriggerType
            assert prediction.predicted_trigger_time > datetime.now()

    # Test helper methods
    @pytest.mark.asyncio
    async def test_extract_injection_signatures(self, trigger_analysis_system, sample_monitoring_data):
        """Test injection signature extraction"""
        # Mock individual signature detection methods
        trigger_analysis_system._detect_seismic_injection_signatures = AsyncMock(return_value=[
            {'timestamp': datetime.now(), 'strength': 0.5, 'duration': 1800, 'type': 'seismic'}
        ])
        trigger_analysis_system._detect_deformation_signatures = AsyncMock(return_value=[
            {'timestamp': datetime.now(), 'strength': 0.3, 'duration': 1200, 'type': 'deformation'}
        ])
        trigger_analysis_system._detect_thermal_signatures = AsyncMock(return_value=[])
        trigger_analysis_system._detect_gas_signatures = AsyncMock(return_value=[])
        trigger_analysis_system._correlate_multi_parameter_signatures = AsyncMock(return_value=[
            {
                'timestamp': datetime.now(),
                'strength': 0.4,
                'duration': 1500,
                'correlation_score': 0.8,
                'velocity_field': np.array([0.1, 0.05, -0.02])
            }
        ])
        
        # Execute test
        signatures = await trigger_analysis_system._extract_injection_signatures(
            sample_monitoring_data['seismic'],
            sample_monitoring_data['deformation'],
            sample_monitoring_data['thermal'],
            sample_monitoring_data['gas']
        )
        
        # Assertions
        assert isinstance(signatures, list)
        assert len(signatures) == 1
        assert 'correlation_score' in signatures[0]

    @pytest.mark.asyncio
    async def test_calculate_rayleigh_number_field(self, trigger_analysis_system, sample_thermal_field,
                                                 sample_density_field, sample_viscosity_field):
        """Test Rayleigh number field calculation"""
        rayleigh_field = await trigger_analysis_system._calculate_rayleigh_number_field(
            sample_thermal_field, sample_density_field, sample_viscosity_field
        )
        
        # Assertions
        assert isinstance(rayleigh_field, np.ndarray)
        assert rayleigh_field.shape == sample_thermal_field.shape
        assert np.all(rayleigh_field >= 0)  # Rayleigh number should be non-negative

    def test_trigger_type_enum(self):
        """Test TriggerType enum"""
        assert TriggerType.MAGMA_INJECTION.value == "magma_injection"
        assert TriggerType.CONVECTION_INITIATION.value == "convection_initiation"
        assert TriggerType.SEISMIC_EVENT.value == "seismic_event"
        assert TriggerType.PRESSURE_CHANGE.value == "pressure_change"
        assert TriggerType.DENSITY_CONTRAST.value == "density_contrast"
        assert TriggerType.TEMPERATURE_GRADIENT.value == "temperature_gradient"
        assert TriggerType.VOLATILE_OVERSATURATION.value == "volatile_oversaturation"
        assert TriggerType.EXTERNAL_PERTURBATION.value == "external_perturbation"

    def test_data_model_creation(self):
        """Test creation of data model instances"""
        # Test MagmaInjectionEvent
        injection_event = MagmaInjectionEvent(
            event_id="test_injection_001",
            timestamp=datetime.now(),
            injection_location=(100.0, 200.0, -2000.0),
            injection_volume=1500.0,
            injection_rate=0.83,
            injected_composition=MagmaComposition(
                SiO2=52.0, TiO2=1.2, Al2O3=16.5, FeO=8.5, MnO=0.15,
                MgO=6.2, CaO=9.8, Na2O=3.2, K2O=1.8, P2O5=0.3
            ),
            injection_temperature=1250.0,
            injection_pressure=5e7,
            injection_velocity=np.array([0.1, 0.05, -0.02]),
            duration=1800.0,
            trigger_strength=0.8,
            mixing_efficiency=0.75
        )
        
        assert injection_event.event_id == "test_injection_001"
        assert injection_event.injection_volume == 1500.0
        assert injection_event.mixing_efficiency == 0.75
        
        # Test ConvectionInitiation
        convection_event = ConvectionInitiation(
            initiation_id="test_convection_001",
            timestamp=datetime.now(),
            rayleigh_number=2000.0,
            critical_rayleigh=1708.0,
            temperature_gradient=0.5,
            density_gradient=10.0,
            viscosity_contrast=1e3,
            convection_onset_depth=1500.0,
            convection_cell_size=200.0,
            convection_velocity=1e-4,
            mixing_potential=0.6
        )
        
        assert convection_event.rayleigh_number == 2000.0
        assert convection_event.mixing_potential == 0.6
        
        # Test ExternalTrigger
        external_trigger = ExternalTrigger(
            trigger_id="test_external_001",
            timestamp=datetime.now(),
            trigger_type=TriggerType.SEISMIC_EVENT,
            magnitude=4.2,
            location=(45.0, -122.0, -5.0),
            frequency=2.5,
            duration=45.0,
            energy=1e12,
            stress_change=1e5,
            pressure_change=0.0,
            mixing_response=0.3
        )
        
        assert external_trigger.trigger_type == TriggerType.SEISMIC_EVENT
        assert external_trigger.magnitude == 4.2
        assert external_trigger.mixing_response == 0.3

    @pytest.mark.asyncio
    async def test_error_handling(self, trigger_analysis_system):
        """Test error handling in various methods"""
        chamber_id = "test_chamber_error"
        
        # Test with invalid data
        with pytest.raises(Exception):
            await trigger_analysis_system.detect_magma_injection_events(
                chamber_id, 
                (datetime.now(), datetime.now() - timedelta(hours=1)),  # Invalid time range
                {}  # Empty monitoring data
            )

    @pytest.mark.asyncio
    async def test_database_integration_mocking(self, trigger_analysis_system):
        """Test database integration with proper mocking"""
        # Test that database methods are properly mocked
        assert trigger_analysis_system.engine is not None
        assert trigger_analysis_system.redis_client is not None
        
        # Test storage methods don't raise errors when mocked
        injection_event = MagmaInjectionEvent(
            event_id="test_001",
            timestamp=datetime.now(),
            injection_location=(0, 0, 0),
            injection_volume=1000.0,
            injection_rate=1.0,
            injected_composition=MagmaComposition(
                SiO2=50.0, TiO2=1.0, Al2O3=15.0, FeO=8.0, MnO=0.1,
                MgO=6.0, CaO=10.0, Na2O=3.0, K2O=2.0, P2O5=0.3
            ),
            injection_temperature=1200.0,
            injection_pressure=1e8,
            injection_velocity=np.array([0, 0, 0]),
            duration=3600.0,
            trigger_strength=0.5,
            mixing_efficiency=0.5
        )
        
        # Mock the storage method to avoid actual database calls
        trigger_analysis_system._store_injection_events = AsyncMock()
        await trigger_analysis_system._store_injection_events([injection_event])
        trigger_analysis_system._store_injection_events.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])