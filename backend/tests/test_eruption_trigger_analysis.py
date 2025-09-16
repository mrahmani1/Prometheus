"""
Comprehensive tests for Eruption Trigger Analysis Framework
Tests all components with realistic volcanic data and scenarios
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import json
from typing import Dict, List

# Import the framework components
from backend.core.eruption_trigger_analysis import (
    PressureThresholdAnalyzer,
    VolatileOversaturationModeler,
    ExternalTriggerAnalyzer,
    CascadingFailureAnalyzer,
    ProbabilisticEruptionForecaster,
    EruptionCharacteristicsPredictor,
    TemporalEruptionSequenceModeler,
    HistoricalEruptionValidator,
    EruptionTriggerAnalysisFramework,
    PressureThreshold,
    VolatileOversaturation,
    ExternalTrigger,
    CascadingFailure,
    EruptionForecast,
    EruptionCharacteristics
)

@pytest.fixture
def mock_db_engine():
    """Mock database engine for testing"""
    engine = AsyncMock()
    return engine

@pytest.fixture
def sample_chamber_network():
    """Sample multi-chamber volcanic system for testing"""
    return {
        'id': 'test_network_001',
        'center_location': (45.0, -122.0, 0.0),  # lat, lon, elevation
        'chambers': {
            'chamber_1': {
                'id': 'chamber_1',
                'location': (45.0, -122.0, -5000.0),  # 5 km depth
                'depth': 5000.0,
                'radius': 1000.0,
                'volume': 4.19e9,  # ~1 km³
                'current_pressure': 150e6,  # 150 MPa
                'temperature': 1000.0,  # °C
                'composition': {
                    'SiO2': 70.0, 'Al2O3': 15.0, 'FeO': 3.0, 'MgO': 2.0,
                    'CaO': 4.0, 'Na2O': 4.0, 'K2O': 2.0,
                    'CO2': 1000.0, 'H2O': 4.0, 'SO2': 500.0
                },
                'magma_supply_rate': 1e-3  # m³/s
            },
            'chamber_2': {
                'id': 'chamber_2',
                'location': (45.01, -122.01, -8000.0),  # 8 km depth
                'depth': 8000.0,
                'radius': 1500.0,
                'volume': 1.41e10,  # ~3.4 km³
                'current_pressure': 200e6,  # 200 MPa
                'temperature': 1100.0,  # °C
                'composition': {
                    'SiO2': 65.0, 'Al2O3': 16.0, 'FeO': 5.0, 'MgO': 4.0,
                    'CaO': 6.0, 'Na2O': 3.0, 'K2O': 1.0,
                    'CO2': 2000.0, 'H2O': 5.0, 'SO2': 800.0
                },
                'magma_supply_rate': 2e-3  # m³/s
            }
        }
    }

@pytest.fixture
def sample_stress_field():
    """Sample stress field for testing"""
    return np.array([
        [50e6, 10e6, 5e6],    # σxx, σxy, σxz
        [10e6, 40e6, 8e6],    # σyx, σyy, σyz
        [5e6, 8e6, 60e6]      # σzx, σzy, σzz
    ])

@pytest.fixture
def sample_material_properties():
    """Sample material properties for testing"""
    return {
        'tensile_strength': 5e6,  # Pa
        'cohesion': 10e6,  # Pa
        'friction_angle': 30.0,  # degrees
        'crack_velocity_coeff': 1e-6,  # m/s
        'stress_corrosion_index': 20.0,
        'youngs_modulus': 50e9,  # Pa
        'poisson_ratio': 0.25
    }

@pytest.fixture
def sample_monitoring_data():
    """Sample monitoring data for testing"""
    base_time = datetime.utcnow()
    
    return {
        'seismic': {
            'earthquakes': [
                {
                    'magnitude': 4.5,
                    'location': (45.05, -122.05, -2000.0),
                    'timestamp': base_time - timedelta(hours=6),
                    'event_type': 'volcano-tectonic'
                },
                {
                    'magnitude': 3.2,
                    'location': (44.98, -121.98, -3000.0),
                    'timestamp': base_time - timedelta(hours=2),
                    'event_type': 'long-period'
                }
            ]
        },
        'deformation': {
            'insar': [
                {
                    'displacement': 0.05,  # m
                    'timestamp': base_time - timedelta(days=1),
                    'location': (45.0, -122.0)
                },
                {
                    'displacement': 0.08,  # m
                    'timestamp': base_time,
                    'location': (45.0, -122.0)
                }
            ]
        },
        'gas': {
            'co2_flux': [100.0, 120.0, 150.0, 180.0, 200.0],  # tons/day
            'so2_flux': [50.0, 60.0, 80.0, 100.0, 120.0]  # tons/day
        },
        'thermal': {
            'temperature': [
                {
                    'temperature': 85.0,  # °C
                    'timestamp': base_time - timedelta(hours=12),
                    'location': (45.0, -122.0)
                },
                {
                    'temperature': 92.0,  # °C
                    'timestamp': base_time,
                    'location': (45.0, -122.0)
                }
            ]
        },
        'injection': {
            'magma_injections': [
                {
                    'volume': 1e6,  # m³
                    'timestamp': base_time - timedelta(hours=8),
                    'location': (45.0, -122.0, -6000.0),
                    'volume_rate': 0.1,  # m³/s
                    'source_volume': 1e9  # m³
                }
            ]
        },
        'atmospheric': {
            'pressure_changes': [
                {
                    'pressure_change': -2000.0,  # Pa
                    'timestamp': base_time - timedelta(hours=4),
                    'duration': 3600.0  # seconds
                }
            ]
        }
    }

@pytest.fixture
def sample_historical_data():
    """Sample historical eruption data for testing"""
    base_time = datetime(2000, 1, 1)
    
    return {
        'eruptions': [
            {
                'timestamp': base_time,
                'vei': 3,
                'magnitude': 3.2,
                'duration': 86400.0,  # 1 day
                'volume': 1e8,  # m³
                'style': 'vulcanian',
                'chambers_involved': ['chamber_1']
            },
            {
                'timestamp': base_time + timedelta(days=365),
                'vei': 2,
                'magnitude': 2.5,
                'duration': 43200.0,  # 12 hours
                'volume': 1e7,  # m³
                'style': 'strombolian',
                'chambers_involved': ['chamber_1', 'chamber_2']
            },
            {
                'timestamp': base_time + timedelta(days=730),
                'vei': 4,
                'magnitude': 4.1,
                'duration': 172800.0,  # 2 days
                'volume': 1e9,  # m³
                'style': 'sub-plinian',
                'chambers_involved': ['chamber_2']
            }
        ]
    }

class TestPressureThresholdAnalyzer:
    """Test pressure threshold analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_analyze_pressure_thresholds(self, mock_db_engine, sample_chamber_network, 
                                             sample_stress_field, sample_material_properties):
        """Test comprehensive pressure threshold analysis"""
        
        analyzer = PressureThresholdAnalyzer(mock_db_engine)
        
        # Mock database storage
        mock_db_engine.execute = AsyncMock()
        
        with patch.object(analyzer, '_store_pressure_thresholds', new_callable=AsyncMock):
            thresholds = await analyzer.analyze_pressure_thresholds(
                sample_chamber_network['chambers'],
                sample_stress_field,
                sample_material_properties
            )
        
        # Verify results
        assert len(thresholds) == 8  # 2 chambers × 4 failure mechanisms
        
        for threshold in thresholds:
            assert isinstance(threshold, PressureThreshold)
            assert threshold.critical_pressure > 0
            assert threshold.pressure_gradient > 0
            assert threshold.threshold_uncertainty > 0
            assert threshold.time_to_failure >= 0
            assert threshold.failure_mechanism in ['tensile', 'shear', 'mixed_mode', 'hydraulic']
            assert threshold.chamber_id in ['chamber_1', 'chamber_2']
    
    def test_failure_criteria(self, mock_db_engine, sample_material_properties):
        """Test different failure criteria calculations"""
        
        analyzer = PressureThresholdAnalyzer(mock_db_engine)
        
        # Test stress conditions
        sigma_1, sigma_2, sigma_3 = 100e6, 80e6, 60e6  # Pa
        tensile_strength = sample_material_properties['tensile_strength']
        cohesion = sample_material_properties['cohesion']
        friction_angle = sample_material_properties['friction_angle']
        
        # Test tensile failure
        tensile_crit = analyzer._tensile_failure_criterion(
            sigma_1, sigma_2, sigma_3, tensile_strength, cohesion, friction_angle
        )
        assert tensile_crit > 0
        
        # Test shear failure
        shear_crit = analyzer._shear_failure_criterion(
            sigma_1, sigma_2, sigma_3, tensile_strength, cohesion, friction_angle
        )
        assert shear_crit > 0
        
        # Test mixed mode failure
        mixed_crit = analyzer._mixed_mode_failure_criterion(
            sigma_1, sigma_2, sigma_3, tensile_strength, cohesion, friction_angle
        )
        assert mixed_crit > 0
        assert mixed_crit <= min(tensile_crit, shear_crit)
        
        # Test hydraulic failure
        hydraulic_crit = analyzer._hydraulic_failure_criterion(
            sigma_1, sigma_2, sigma_3, tensile_strength, cohesion, friction_angle
        )
        assert hydraulic_crit > 0

class TestVolatileOversaturationModeler:
    """Test volatile oversaturation modeling functionality"""
    
    @pytest.mark.asyncio
    async def test_model_volatile_oversaturation(self, mock_db_engine, sample_chamber_network):
        """Test comprehensive volatile oversaturation modeling"""
        
        modeler = VolatileOversaturationModeler(mock_db_engine)
        
        chamber = sample_chamber_network['chambers']['chamber_1']
        composition = chamber['composition']
        p_t_conditions = {
            'pressure': chamber['current_pressure'],
            'temperature': chamber['temperature']
        }
        
        with patch.object(modeler, '_store_oversaturation_data', new_callable=AsyncMock):
            oversaturation = await modeler.model_volatile_oversaturation(
                chamber, composition, p_t_conditions
            )
        
        # Verify results
        assert isinstance(oversaturation, VolatileOversaturation)
        assert oversaturation.co2_saturation > 0
        assert oversaturation.h2o_saturation > 0
        assert oversaturation.so2_saturation > 0
        assert oversaturation.bubble_nucleation_rate >= 0
        assert oversaturation.bubble_growth_rate >= 0
        assert 0 <= oversaturation.coalescence_efficiency <= 1
        assert 0 <= oversaturation.gas_volume_fraction <= 1
        assert oversaturation.degassing_rate >= 0
    
    def test_solubility_models(self, mock_db_engine):
        """Test volatile solubility models"""
        
        modeler = VolatileOversaturationModeler(mock_db_engine)
        
        composition = {'SiO2': 70.0, 'Al2O3': 15.0}
        pressure = 100e6  # Pa
        temperature = 1000.0  # °C
        
        # Test CO2 solubility
        co2_solubility = modeler._co2_solubility_model(composition, pressure, temperature)
        assert co2_solubility > 0
        
        # Test H2O solubility
        h2o_solubility = modeler._h2o_solubility_model(composition, pressure, temperature)
        assert h2o_solubility > 0
        
        # Test SO2 solubility
        so2_solubility = modeler._so2_solubility_model(composition, pressure, temperature)
        assert so2_solubility > 0
        
        # Test pressure dependence
        higher_pressure = 200e6  # Pa
        co2_high_p = modeler._co2_solubility_model(composition, higher_pressure, temperature)
        assert co2_high_p > co2_solubility  # Higher pressure = higher solubility
    
    def test_bubble_nucleation_calculation(self, mock_db_engine):
        """Test bubble nucleation rate calculation"""
        
        modeler = VolatileOversaturationModeler(mock_db_engine)
        
        # Test with oversaturation
        oversaturation_ratios = {'co2': 2.0, 'h2o': 1.5, 'so2': 1.8}
        p_t_conditions = {'pressure': 100e6, 'temperature': 1000.0}
        
        nucleation_rate = modeler._calculate_bubble_nucleation_rate(
            oversaturation_ratios, p_t_conditions
        )
        assert nucleation_rate > 0
        
        # Test without oversaturation
        no_oversaturation = {'co2': 0.8, 'h2o': 0.9, 'so2': 0.7}
        nucleation_rate_low = modeler._calculate_bubble_nucleation_rate(
            no_oversaturation, p_t_conditions
        )
        assert nucleation_rate_low == 0.0

class TestExternalTriggerAnalyzer:
    """Test external trigger analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_analyze_external_triggers(self, mock_db_engine, sample_monitoring_data, 
                                           sample_chamber_network):
        """Test comprehensive external trigger analysis"""
        
        analyzer = ExternalTriggerAnalyzer(mock_db_engine)
        
        time_window = (
            datetime.utcnow() - timedelta(days=1),
            datetime.utcnow()
        )
        
        with patch.object(analyzer, '_store_external_triggers', new_callable=AsyncMock):
            triggers = await analyzer.analyze_external_triggers(
                sample_monitoring_data, sample_chamber_network, time_window
            )
        
        # Verify results
        assert len(triggers) > 0
        
        for trigger in triggers:
            assert isinstance(trigger, ExternalTrigger)
            assert trigger.trigger_type in ['seismic', 'tidal', 'injection', 'thermal', 'atmospheric']
            assert trigger.magnitude > 0
            assert len(trigger.location) == 3
            assert trigger.stress_change != 0  # Should have some stress change
            assert trigger.influence_radius > 0
            assert 0 <= trigger.coupling_strength <= 1
    
    @pytest.mark.asyncio
    async def test_seismic_trigger_analysis(self, mock_db_engine, sample_monitoring_data, 
                                          sample_chamber_network):
        """Test seismic trigger analysis specifically"""
        
        analyzer = ExternalTriggerAnalyzer(mock_db_engine)
        
        time_window = (
            datetime.utcnow() - timedelta(days=1),
            datetime.utcnow()
        )
        
        seismic_triggers = await analyzer._analyze_seismic_trigger(
            sample_monitoring_data, sample_chamber_network, time_window
        )
        
        assert len(seismic_triggers) > 0
        
        for trigger in seismic_triggers:
            assert trigger.trigger_type == 'seismic'
            assert trigger.magnitude > 0  # Earthquake magnitude
            assert trigger.stress_change != 0  # Coulomb stress change
    
    def test_distance_calculation(self, mock_db_engine):
        """Test 3D distance calculation"""
        
        analyzer = ExternalTriggerAnalyzer(mock_db_engine)
        
        loc1 = (45.0, -122.0, -5000.0)
        loc2 = (45.01, -122.01, -8000.0)
        
        distance = analyzer._calculate_distance(loc1, loc2)
        
        assert distance > 0
        assert distance < 10000.0  # Should be reasonable distance in meters

class TestCascadingFailureAnalyzer:
    """Test cascading failure analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_analyze_cascading_failure(self, mock_db_engine, sample_chamber_network):
        """Test comprehensive cascading failure analysis"""
        
        analyzer = CascadingFailureAnalyzer(mock_db_engine)
        
        # Create sample pressure thresholds
        pressure_thresholds = [
            PressureThreshold(
                critical_pressure=180e6, pressure_gradient=1e5, threshold_uncertainty=10e6,
                time_to_failure=3600.0, failure_mechanism='tensile', 
                chamber_id='chamber_1', depth=5000.0
            ),
            PressureThreshold(
                critical_pressure=220e6, pressure_gradient=1.2e5, threshold_uncertainty=15e6,
                time_to_failure=7200.0, failure_mechanism='tensile',
                chamber_id='chamber_2', depth=8000.0
            )
        ]
        
        # Create sample external triggers
        external_triggers = [
            ExternalTrigger(
                trigger_type='seismic', magnitude=4.5, 
                location=(45.05, -122.05, -2000.0),
                timestamp=datetime.utcnow(), stress_change=5e6,
                strain_rate=1e-6, influence_radius=20000.0, coupling_strength=0.8
            )
        ]
        
        initial_conditions = {
            'chamber_1': {'pressure': 150e6},
            'chamber_2': {'pressure': 180e6}
        }
        
        with patch.object(analyzer, '_store_cascading_failure', new_callable=AsyncMock):
            cascading_failure = await analyzer.analyze_cascading_failure(
                sample_chamber_network, pressure_thresholds, 
                external_triggers, initial_conditions
            )
        
        # Verify results
        assert isinstance(cascading_failure, CascadingFailure)
        assert len(cascading_failure.failure_sequence) >= 0
        assert len(cascading_failure.failure_times) == len(cascading_failure.failure_sequence)
        assert cascading_failure.stress_transfer_matrix.shape == (2, 2)  # 2 chambers
        assert len(cascading_failure.failure_probabilities) == 2
        assert cascading_failure.total_system_failure_time >= 0
    
    @pytest.mark.asyncio
    async def test_stress_transfer_matrix(self, mock_db_engine, sample_chamber_network):
        """Test stress transfer matrix calculation"""
        
        analyzer = CascadingFailureAnalyzer(mock_db_engine)
        
        stress_matrix = await analyzer._build_stress_transfer_matrix(sample_chamber_network)
        
        # Verify matrix properties
        assert stress_matrix.shape == (2, 2)  # 2 chambers
        assert np.all(stress_matrix >= 0)  # All positive values
        assert np.all(np.diag(stress_matrix) == 0)  # Zero diagonal (no self-interaction)

class TestProbabilisticEruptionForecaster:
    """Test probabilistic eruption forecasting functionality"""
    
    @pytest.mark.asyncio
    async def test_generate_eruption_forecast(self, mock_db_engine, sample_chamber_network,
                                            sample_monitoring_data):
        """Test comprehensive eruption forecasting"""
        
        forecaster = ProbabilisticEruptionForecaster(mock_db_engine)
        
        # Create sample input data
        pressure_thresholds = [
            PressureThreshold(
                critical_pressure=180e6, pressure_gradient=1e5, threshold_uncertainty=10e6,
                time_to_failure=3600.0, failure_mechanism='tensile',
                chamber_id='chamber_1', depth=5000.0
            )
        ]
        
        external_triggers = [
            ExternalTrigger(
                trigger_type='seismic', magnitude=4.5,
                location=(45.05, -122.05, -2000.0),
                timestamp=datetime.utcnow(), stress_change=5e6,
                strain_rate=1e-6, influence_radius=20000.0, coupling_strength=0.8
            )
        ]
        
        cascading_failure = CascadingFailure(
            failure_sequence=['chamber_1', 'chamber_2'],
            failure_times=[3600.0, 7200.0],
            stress_transfer_matrix=np.array([[0, 0.1], [0.2, 0]]),
            failure_probabilities=[0.7, 0.5],
            total_system_failure_time=7200.0,
            critical_path=['chamber_1', 'chamber_2']
        )
        
        with patch.object(forecaster, '_store_eruption_forecast', new_callable=AsyncMock):
            forecast = await forecaster.generate_eruption_forecast(
                sample_chamber_network, pressure_thresholds, external_triggers,
                cascading_failure, sample_monitoring_data
            )
        
        # Verify results
        assert isinstance(forecast, EruptionForecast)
        assert 0 <= forecast.eruption_probability <= 1
        assert len(forecast.time_window) == 2
        assert forecast.time_window[0] <= forecast.time_window[1]
        assert len(forecast.confidence_interval) == 2
        assert forecast.confidence_interval[0] <= forecast.confidence_interval[1]
        assert forecast.magnitude_estimate >= 0
        assert forecast.style_prediction in ['effusive', 'strombolian', 'vulcanian', 'sub-plinian', 'plinian']
        assert 0 <= forecast.forecast_skill <= 1
    
    @pytest.mark.asyncio
    async def test_feature_extraction(self, mock_db_engine, sample_chamber_network, 
                                    sample_monitoring_data):
        """Test feature extraction for ML models"""
        
        forecaster = ProbabilisticEruptionForecaster(mock_db_engine)
        
        # Create sample data
        pressure_thresholds = [
            PressureThreshold(
                critical_pressure=180e6, pressure_gradient=1e5, threshold_uncertainty=10e6,
                time_to_failure=3600.0, failure_mechanism='tensile',
                chamber_id='chamber_1', depth=5000.0
            )
        ]
        
        external_triggers = []
        
        cascading_failure = CascadingFailure(
            failure_sequence=['chamber_1'],
            failure_times=[3600.0],
            stress_transfer_matrix=np.array([[0, 0.1], [0.2, 0]]),
            failure_probabilities=[0.7, 0.5],
            total_system_failure_time=3600.0,
            critical_path=['chamber_1']
        )
        
        features = await forecaster._extract_forecast_features(
            sample_chamber_network, pressure_thresholds, external_triggers,
            cascading_failure, sample_monitoring_data
        )
        
        # Verify features
        assert len(features) == 20  # Fixed feature size
        assert np.all(np.isfinite(features))  # No NaN or infinite values

class TestEruptionCharacteristicsPredictor:
    """Test eruption characteristics prediction functionality"""
    
    @pytest.mark.asyncio
    async def test_predict_eruption_characteristics(self, mock_db_engine, sample_chamber_network):
        """Test comprehensive eruption characteristics prediction"""
        
        predictor = EruptionCharacteristicsPredictor(mock_db_engine)
        
        # Create sample input data
        eruption_forecast = EruptionForecast(
            eruption_probability=0.7,
            time_window=(3600.0, 86400.0),
            confidence_interval=(0.5, 0.9),
            magnitude_estimate=3.5,
            style_prediction='vulcanian',
            uncertainty_bounds={'probability': 0.1, 'magnitude': 0.5},
            forecast_skill=0.8
        )
        
        volatile_data = [
            VolatileOversaturation(
                co2_saturation=1500.0, h2o_saturation=4.5, so2_saturation=600.0,
                bubble_nucleation_rate=1e12, bubble_growth_rate=1e-4,
                coalescence_efficiency=0.3, gas_volume_fraction=0.15,
                degassing_rate=100.0
            )
        ]
        
        pressure_data = [
            PressureThreshold(
                critical_pressure=180e6, pressure_gradient=1e5, threshold_uncertainty=10e6,
                time_to_failure=3600.0, failure_mechanism='tensile',
                chamber_id='chamber_1', depth=5000.0
            )
        ]
        
        with patch.object(predictor, '_store_eruption_characteristics', new_callable=AsyncMock):
            characteristics = await predictor.predict_eruption_characteristics(
                sample_chamber_network, eruption_forecast, volatile_data, pressure_data
            )
        
        # Verify results
        assert isinstance(characteristics, EruptionCharacteristics)
        assert 0 <= characteristics.vei_estimate <= 8
        assert characteristics.discharge_rate > 0
        assert characteristics.eruption_duration > 0
        assert 0 <= characteristics.explosivity_index <= 1
        assert characteristics.column_height > 0
        assert 0 <= characteristics.pyroclastic_flow_potential <= 1
        assert 0 <= characteristics.lava_flow_potential <= 1
        assert characteristics.ash_dispersal_range > 0

class TestTemporalEruptionSequenceModeler:
    """Test temporal eruption sequence modeling functionality"""
    
    @pytest.mark.asyncio
    async def test_model_temporal_eruption_sequence(self, mock_db_engine, sample_chamber_network,
                                                  sample_historical_data):
        """Test comprehensive temporal sequence modeling"""
        
        modeler = TemporalEruptionSequenceModeler(mock_db_engine)
        
        # Create sample input data
        cascading_failure = CascadingFailure(
            failure_sequence=['chamber_1', 'chamber_2'],
            failure_times=[3600.0, 7200.0],
            stress_transfer_matrix=np.array([[0, 0.1], [0.2, 0]]),
            failure_probabilities=[0.7, 0.5],
            total_system_failure_time=7200.0,
            critical_path=['chamber_1', 'chamber_2']
        )
        
        eruption_characteristics = [
            EruptionCharacteristics(
                vei_estimate=3.0, discharge_rate=100.0, eruption_duration=86400.0,
                explosivity_index=0.6, column_height=15000.0,
                pyroclastic_flow_potential=0.4, lava_flow_potential=0.6,
                ash_dispersal_range=100000.0
            )
        ]
        
        with patch.object(modeler, '_store_temporal_model', new_callable=AsyncMock):
            temporal_model = await modeler.model_temporal_eruption_sequence(
                sample_chamber_network, cascading_failure, eruption_characteristics,
                sample_historical_data
            )
        
        # Verify results
        assert 'sequence_model' in temporal_model
        assert 'eruption_sequences' in temporal_model
        assert 'sequence_probabilities' in temporal_model
        assert 'temporal_patterns' in temporal_model
        assert 'repose_intervals' in temporal_model
        assert 'cumulative_hazard' in temporal_model
        
        # Verify sequence model structure
        sequence_model = temporal_model['sequence_model']
        assert 'chamber_ids' in sequence_model
        assert 'transition_matrix' in sequence_model
        assert 'temporal_parameters' in sequence_model
        
        # Verify eruption sequences
        sequences = temporal_model['eruption_sequences']
        assert len(sequences) > 0
        
        for sequence in sequences[:5]:  # Check first 5 sequences
            assert 'events' in sequence
            assert 'total_duration' in sequence
            assert 'n_eruptions' in sequence
            assert 'chambers_involved' in sequence

class TestHistoricalEruptionValidator:
    """Test historical eruption validation functionality"""
    
    @pytest.mark.asyncio
    async def test_validate_against_historical_eruptions(self, mock_db_engine, sample_historical_data):
        """Test comprehensive historical validation"""
        
        validator = HistoricalEruptionValidator(mock_db_engine)
        
        # Create sample predictions to validate
        eruption_forecasts = [
            EruptionForecast(
                eruption_probability=0.7,
                time_window=(0.0, 86400.0),
                confidence_interval=(0.5, 0.9),
                magnitude_estimate=3.0,
                style_prediction='vulcanian',
                uncertainty_bounds={'probability': 0.1},
                forecast_skill=0.8
            )
        ]
        
        eruption_characteristics = [
            EruptionCharacteristics(
                vei_estimate=3.0, discharge_rate=100.0, eruption_duration=86400.0,
                explosivity_index=0.6, column_height=15000.0,
                pyroclastic_flow_potential=0.4, lava_flow_potential=0.6,
                ash_dispersal_range=100000.0
            )
        ]
        
        temporal_model = {
            'repose_intervals': {
                'predicted_repose_mean': 31536000.0,  # 1 year
                'predicted_repose_std': 15768000.0    # 6 months
            },
            'temporal_patterns': {
                'duration_mean': 86400.0,
                'clustering_coefficient': 0.3
            }
        }
        
        historical_database = {
            'Yellowstone': sample_historical_data,
            'Mount St. Helens': sample_historical_data
        }
        
        with patch.object(validator, '_store_validation_results', new_callable=AsyncMock):
            validation_report = await validator.validate_against_historical_eruptions(
                eruption_forecasts, eruption_characteristics, temporal_model, historical_database
            )
        
        # Verify results
        assert 'forecast_validation' in validation_report
        assert 'magnitude_validation' in validation_report
        assert 'temporal_validation' in validation_report
        assert 'model_performance' in validation_report
        
        # Verify validation metrics
        forecast_validation = validation_report['forecast_validation']
        assert 'probability_accuracy' in forecast_validation
        assert 'timing_accuracy' in forecast_validation
        assert 'skill_score' in forecast_validation
        
        magnitude_validation = validation_report['magnitude_validation']
        assert 'vei_accuracy' in magnitude_validation
        
        model_performance = validation_report['model_performance']
        assert 'overall_score' in model_performance
        assert 'performance_category' in model_performance

class TestEruptionTriggerAnalysisFramework:
    """Test complete eruption trigger analysis framework"""
    
    @pytest.mark.asyncio
    async def test_run_complete_analysis(self, sample_chamber_network, sample_monitoring_data,
                                       sample_historical_data):
        """Test complete end-to-end analysis"""
        
        # Mock database configuration
        db_config = {
            'connection_string': 'postgresql+asyncpg://test:test@localhost/test'
        }
        
        framework = EruptionTriggerAnalysisFramework(db_config)
        
        # Mock all database operations
        with patch.object(framework.pressure_analyzer, '_store_pressure_thresholds', new_callable=AsyncMock), \
             patch.object(framework.volatile_modeler, '_store_oversaturation_data', new_callable=AsyncMock), \
             patch.object(framework.external_analyzer, '_store_external_triggers', new_callable=AsyncMock), \
             patch.object(framework.cascading_analyzer, '_store_cascading_failure', new_callable=AsyncMock), \
             patch.object(framework.eruption_forecaster, '_store_eruption_forecast', new_callable=AsyncMock), \
             patch.object(framework.characteristics_predictor, '_store_eruption_characteristics', new_callable=AsyncMock), \
             patch.object(framework.temporal_modeler, '_store_temporal_model', new_callable=AsyncMock), \
             patch.object(framework.historical_validator, '_store_validation_results', new_callable=AsyncMock):
            
            analysis_config = {
                'stress_field': np.zeros((3, 3)),
                'material_properties': {
                    'tensile_strength': 5e6,
                    'cohesion': 10e6,
                    'friction_angle': 30.0
                },
                'time_window': (
                    datetime.utcnow() - timedelta(days=30),
                    datetime.utcnow()
                ),
                'initial_conditions': {
                    'chamber_1': {'pressure': 150e6},
                    'chamber_2': {'pressure': 180e6}
                },
                'forecast_horizon': 86400.0 * 365.0  # 1 year
            }
            
            results = await framework.run_complete_analysis(
                sample_chamber_network, sample_monitoring_data,
                sample_historical_data, analysis_config
            )
        
        # Verify comprehensive results
        assert 'pressure_thresholds' in results
        assert 'volatile_oversaturation' in results
        assert 'external_triggers' in results
        assert 'cascading_failure' in results
        assert 'eruption_forecast' in results
        assert 'eruption_characteristics' in results
        assert 'temporal_model' in results
        assert 'validation_report' in results
        assert 'analysis_timestamp' in results
        assert 'chamber_network_id' in results
        
        # Verify data types
        assert isinstance(results['pressure_thresholds'], list)
        assert isinstance(results['volatile_oversaturation'], list)
        assert isinstance(results['external_triggers'], list)
        assert isinstance(results['cascading_failure'], CascadingFailure)
        assert isinstance(results['eruption_forecast'], EruptionForecast)
        assert isinstance(results['eruption_characteristics'], EruptionCharacteristics)
        assert isinstance(results['temporal_model'], dict)
        assert isinstance(results['validation_report'], dict)
        
        # Clean up
        await framework.close()

class TestIntegrationScenarios:
    """Test realistic volcanic scenarios"""
    
    @pytest.mark.asyncio
    async def test_yellowstone_scenario(self):
        """Test analysis with Yellowstone-like multi-chamber system"""
        
        # Yellowstone-like chamber network
        yellowstone_network = {
            'id': 'yellowstone_test',
            'center_location': (44.428, -110.588, 0.0),
            'chambers': {
                'upper_chamber': {
                    'id': 'upper_chamber',
                    'location': (44.428, -110.588, -8000.0),  # 8 km depth
                    'depth': 8000.0,
                    'radius': 2000.0,
                    'volume': 3.35e10,  # ~8 km³
                    'current_pressure': 200e6,
                    'temperature': 750.0,  # Rhyolitic
                    'composition': {
                        'SiO2': 77.0, 'Al2O3': 12.5, 'FeO': 1.5, 'MgO': 0.5,
                        'CaO': 1.0, 'Na2O': 4.0, 'K2O': 3.5,
                        'CO2': 500.0, 'H2O': 6.0, 'SO2': 200.0
                    },
                    'magma_supply_rate': 0.1e-3
                },
                'lower_chamber': {
                    'id': 'lower_chamber',
                    'location': (44.428, -110.588, -50000.0),  # 50 km depth
                    'depth': 50000.0,
                    'radius': 10000.0,
                    'volume': 4.19e12,  # ~1000 km³
                    'current_pressure': 1500e6,
                    'temperature': 1200.0,  # Basaltic
                    'composition': {
                        'SiO2': 50.0, 'Al2O3': 18.0, 'FeO': 10.0, 'MgO': 8.0,
                        'CaO': 10.0, 'Na2O': 2.5, 'K2O': 1.5,
                        'CO2': 3000.0, 'H2O': 2.0, 'SO2': 1000.0
                    },
                    'magma_supply_rate': 1e-3
                }
            }
        }
        
        # High-risk monitoring data
        high_risk_monitoring = {
            'seismic': {
                'earthquakes': [
                    {'magnitude': 5.2, 'location': (44.43, -110.59, -5000.0),
                     'timestamp': datetime.utcnow() - timedelta(hours=2)},
                    {'magnitude': 4.8, 'location': (44.42, -110.58, -8000.0),
                     'timestamp': datetime.utcnow() - timedelta(hours=1)},
                    {'magnitude': 6.1, 'location': (44.45, -110.60, -12000.0),
                     'timestamp': datetime.utcnow() - timedelta(minutes=30)}
                ]
            },
            'deformation': {
                'insar': [
                    {'displacement': 0.5, 'timestamp': datetime.utcnow() - timedelta(days=7)},
                    {'displacement': 1.2, 'timestamp': datetime.utcnow()}
                ]
            },
            'gas': {
                'co2_flux': [500.0, 800.0, 1200.0, 1800.0, 2500.0],
                'so2_flux': [200.0, 400.0, 600.0, 900.0, 1200.0]
            },
            'thermal': {
                'temperature': [
                    {'temperature': 95.0, 'timestamp': datetime.utcnow() - timedelta(hours=6)},
                    {'temperature': 105.0, 'timestamp': datetime.utcnow()}
                ]
            }
        }
        
        # Mock framework
        db_config = {'connection_string': 'test'}
        framework = EruptionTriggerAnalysisFramework(db_config)
        
        # Mock all database operations
        with patch.object(framework.pressure_analyzer, '_store_pressure_thresholds', new_callable=AsyncMock), \
             patch.object(framework.volatile_modeler, '_store_oversaturation_data', new_callable=AsyncMock), \
             patch.object(framework.external_analyzer, '_store_external_triggers', new_callable=AsyncMock), \
             patch.object(framework.cascading_analyzer, '_store_cascading_failure', new_callable=AsyncMock), \
             patch.object(framework.eruption_forecaster, '_store_eruption_forecast', new_callable=AsyncMock), \
             patch.object(framework.characteristics_predictor, '_store_eruption_characteristics', new_callable=AsyncMock), \
             patch.object(framework.temporal_modeler, '_store_temporal_model', new_callable=AsyncMock), \
             patch.object(framework.historical_validator, '_store_validation_results', new_callable=AsyncMock):
            
            analysis_config = {
                'stress_field': np.array([
                    [30e6, 5e6, 2e6],
                    [5e6, 25e6, 3e6],
                    [2e6, 3e6, 35e6]
                ]),
                'material_properties': {
                    'tensile_strength': 3e6,  # Weaker due to hydrothermal alteration
                    'cohesion': 8e6,
                    'friction_angle': 25.0
                },
                'time_window': (datetime.utcnow() - timedelta(days=30), datetime.utcnow()),
                'initial_conditions': {
                    'upper_chamber': {'pressure': 200e6},
                    'lower_chamber': {'pressure': 1500e6}
                },
                'forecast_horizon': 86400.0 * 30.0  # 30 days
            }
            
            results = await framework.run_complete_analysis(
                yellowstone_network, high_risk_monitoring, {}, analysis_config
            )
        
        # Verify high-risk scenario results
        assert results['eruption_forecast'].eruption_probability > 0.1  # Should show elevated risk
        assert results['eruption_characteristics'].vei_estimate >= 3.0  # Large system
        
        await framework.close()

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])