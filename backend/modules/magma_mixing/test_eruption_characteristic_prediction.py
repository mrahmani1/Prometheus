"""
Test suite for Eruption Characteristic Prediction System

This module provides comprehensive tests for the eruption characteristic prediction
system, including unit tests, integration tests, and validation tests against
historical eruption data.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from .eruption_characteristic_prediction_system import (
    EruptionCharacteristicPredictionSystem,
    EruptionIntensityPrediction,
    ExplosivityForecast,
    DischargeRatePrediction,
    EruptionStylePrediction,
    TemporalEruptionEvolution,
    MixingTriggeredHazardAssessment,
    UncertaintyQuantification,
    HistoricalValidationResult,
    EruptionCharacteristicPredictionResult
)
from .data_models import RheologicalProperties


class TestEruptionCharacteristicPredictionSystem:
    """Test class for eruption characteristic prediction system"""
    
    @pytest.fixture
    async def prediction_system(self):
        """Create prediction system for testing"""
        database_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'test_user',
            'password': 'test_password',
            'database': 'test_volcanic_db'
        }
        
        system = EruptionCharacteristicPredictionSystem(database_config)
        
        # Mock database connection
        system.db_pool = AsyncMock()
        
        return system
    
    @pytest.mark.asyncio
    async def test_predict_eruption_intensity(self, prediction_system):
        """Test eruption intensity prediction"""
        # Test data
        mixing_degree = 0.65
        magma_properties = {
            'volatile_content': 0.045,
            'temperature': 1373,
            'viscosity': 1e5,
            'crystallinity': 0.25
        }
        chamber_conditions = {
            'pressure': 1e8
        }
        
        # Run prediction
        result = await prediction_system.predict_eruption_intensity(
            mixing_degree, magma_properties, chamber_conditions
        )
        
        # Assertions
        assert isinstance(result, EruptionIntensityPrediction)
        assert 0 <= result.predicted_intensity <= 8
        assert 0 <= result.intensity_confidence <= 1
        assert len(result.intensity_bounds) == 2
        assert result.intensity_bounds[0] <= result.intensity_bounds[1]
        assert isinstance(result.contributing_factors, dict)
        assert result.mixing_degree == mixing_degree
    
    @pytest.mark.asyncio
    async def test_forecast_explosivity(self, prediction_system):
        """Test explosivity forecasting"""
        # Test data
        volatile_evolution = {
            'H2O': [0.02, 0.03, 0.045, 0.05],
            'CO2': [0.005, 0.007, 0.008, 0.01],
            'SO2': [0.001, 0.0015, 0.002, 0.0025]
        }
        pressure_evolution = [8e7, 9e7, 1e8, 1.1e8]
        temperature_evolution = [1200, 1250, 1300, 1350]
        time_steps = [
            datetime.now() - timedelta(hours=3),
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=1),
            datetime.now()
        ]
        
        # Run forecast
        result = await prediction_system.forecast_explosivity(
            volatile_evolution, pressure_evolution, temperature_evolution, time_steps
        )
        
        # Assertions
        assert isinstance(result, ExplosivityForecast)
        assert 0 <= result.predicted_explosivity <= 1
        assert 0 <= result.explosivity_index <= 8
        assert result.gas_pressure_buildup >= 0
        assert 0 <= result.explosivity_confidence <= 1
        assert isinstance(result.forecast_horizon, timedelta)
        assert isinstance(result.volatile_content, dict)
    
    @pytest.mark.asyncio
    async def test_predict_discharge_rate(self, prediction_system):
        """Test discharge rate prediction"""
        # Test data
        rheological_properties = RheologicalProperties(
            viscosity=1e5,
            yield_strength=1e4,
            shear_rate=0.1,
            strain_rate=0.01,
            non_newtonian_index=0.8,
            thixotropy_parameter=0.1
        )
        conduit_geometry = {
            'diameter': 100,
            'length': 5000,
            'roughness': 0.01,
            'chamber_volume': 1e8
        }
        pressure_gradient = 1e4
        gas_content = 0.03
        
        # Run prediction
        result = await prediction_system.predict_discharge_rate(
            rheological_properties, conduit_geometry, pressure_gradient, gas_content
        )
        
        # Assertions
        assert isinstance(result, DischargeRatePrediction)
        assert result.predicted_discharge_rate >= 0
        assert result.peak_discharge_rate >= result.sustained_discharge_rate
        assert result.discharge_duration >= 0
        assert isinstance(result.rheological_controls, dict)
        assert result.viscosity == rheological_properties.viscosity
        assert result.yield_strength == rheological_properties.yield_strength
    
    @pytest.mark.asyncio
    async def test_predict_eruption_style(self, prediction_system):
        """Test eruption style prediction"""
        # Test data for explosive eruption
        mixed_magma_properties = {
            'gas_content': 0.06,
            'viscosity': 5e6,
            'temperature': 1123,
            'ascent_rate': 0.5,
            'crystallinity': 0.3
        }
        environmental_conditions = {
            'water_interaction': 0.05,
            'depth_to_water': 500,
            'external_water': False
        }
        
        # Run prediction
        result = await prediction_system.predict_eruption_style(
            mixed_magma_properties, environmental_conditions
        )
        
        # Assertions
        assert isinstance(result, EruptionStylePrediction)
        assert result.predicted_style in ['effusive', 'explosive', 'mixed', 'phreatomagmatic']
        assert isinstance(result.style_probability, dict)
        assert all(0 <= prob <= 1 for prob in result.style_probability.values())
        assert abs(sum(result.style_probability.values()) - 1.0) < 0.01  # Should sum to 1
        assert 0 <= result.style_confidence <= 1
        assert isinstance(result.controlling_parameters, dict)
    
    @pytest.mark.asyncio
    async def test_model_temporal_eruption_evolution(self, prediction_system):
        """Test temporal eruption evolution modeling"""
        # Test data
        mixing_progression = [0.2, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3]
        time_steps = [
            datetime.now() + timedelta(hours=i) for i in range(len(mixing_progression))
        ]
        initial_conditions = {
            'gas_content': 0.02,
            'viscosity': 1e5,
            'temperature': 1200,
            'crystallinity': 0.2,
            'pressure': 1e8
        }
        
        # Run modeling
        result = await prediction_system.model_temporal_eruption_evolution(
            mixing_progression, time_steps, initial_conditions
        )
        
        # Assertions
        assert isinstance(result, TemporalEruptionEvolution)
        assert len(result.evolution_timeline) == len(mixing_progression)
        assert len(result.intensity_evolution) == len(mixing_progression)
        assert result.mixing_progression == mixing_progression
        assert isinstance(result.peak_activity_time, datetime)
        assert isinstance(result.eruption_duration, timedelta)
        assert isinstance(result.evolution_uncertainty, dict)
    
    @pytest.mark.asyncio
    async def test_assess_mixing_triggered_hazards(self, prediction_system):
        """Test hazard assessment for mixing-triggered eruptions"""
        # Test data
        eruption_characteristics = {
            'intensity': 4.5,
            'explosivity': 0.7,
            'discharge_rate': 500,
            'style': 'explosive',
            'gas_emission_rate': 2000
        }
        location = {
            'lat': 40.821,
            'lon': 14.426,
            'elevation': 1281
        }
        population_data = {
            'total_population': 500000,
            'population_density': 200,
            'infrastructure': {
                'buildings_density': 50,
                'roads_density': 2,
                'utilities_density': 5
            }
        }
        
        # Run assessment
        result = await prediction_system.assess_mixing_triggered_hazards(
            eruption_characteristics, location, population_data
        )
        
        # Assertions
        assert isinstance(result, MixingTriggeredHazardAssessment)
        assert isinstance(result.hazard_zones, dict)
        assert isinstance(result.population_exposure, dict)
        assert isinstance(result.infrastructure_risk, dict)
        assert isinstance(result.evacuation_recommendations, dict)
        assert isinstance(result.hazard_probabilities, dict)
        assert isinstance(result.impact_scenarios, list)
        assert isinstance(result.risk_mitigation_strategies, list)
        
        # Check that hazard zones are properly calculated
        for hazard_type, zones in result.hazard_zones.items():
            assert isinstance(zones, dict)
            for zone_name, zone_data in zones.items():
                assert 'probability' in zone_data
                assert 0 <= zone_data['probability'] <= 1
    
    @pytest.mark.asyncio
    async def test_quantify_prediction_uncertainty(self, prediction_system):
        """Test uncertainty quantification"""
        # Test data
        prediction_results = {
            'intensity': 4.2,
            'explosivity': 0.65,
            'discharge_rate': 300,
            'style_probability': 0.8
        }
        model_parameters = {
            'data_coverage': 0.85,
            'intensity_validation_score': 0.75,
            'explosivity_validation_score': 0.70,
            'mixing_degree': 0.6,
            'volatile_content': 0.04
        }
        
        # Run uncertainty quantification
        result = await prediction_system.quantify_prediction_uncertainty(
            prediction_results, model_parameters
        )
        
        # Assertions
        assert isinstance(result, UncertaintyQuantification)
        assert isinstance(result.aleatory_uncertainty, dict)
        assert isinstance(result.epistemic_uncertainty, dict)
        assert isinstance(result.model_uncertainty, dict)
        assert isinstance(result.parameter_uncertainty, dict)
        assert isinstance(result.total_uncertainty, dict)
        assert isinstance(result.confidence_intervals, dict)
        assert isinstance(result.sensitivity_analysis, dict)
        
        # Check that uncertainties are reasonable
        for uncertainty_dict in [result.aleatory_uncertainty, result.epistemic_uncertainty,
                               result.model_uncertainty, result.parameter_uncertainty,
                               result.total_uncertainty]:
            for param, uncertainty in uncertainty_dict.items():
                assert 0 <= uncertainty <= 2  # Reasonable uncertainty range
    
    @pytest.mark.asyncio
    async def test_validate_against_historical_eruptions(self, prediction_system):
        """Test validation against historical eruptions"""
        # Create mock prediction result
        prediction_result = Mock()
        prediction_result.intensity_prediction = Mock()
        prediction_result.intensity_prediction.predicted_intensity = 3.5
        prediction_result.explosivity_forecast = Mock()
        prediction_result.explosivity_forecast.predicted_explosivity = 0.75
        prediction_result.discharge_rate_prediction = Mock()
        prediction_result.discharge_rate_prediction.predicted_discharge_rate = 120
        prediction_result.style_prediction = Mock()
        prediction_result.style_prediction.predicted_style = 'explosive'
        
        # Run validation
        result = await prediction_system.validate_against_historical_eruptions(
            prediction_result, 'mount_vesuvius_1944'
        )
        
        # Assertions
        assert isinstance(result, HistoricalValidationResult)
        assert result.eruption_name == 'mount_vesuvius_1944'
        assert isinstance(result.eruption_date, datetime)
        assert isinstance(result.predicted_characteristics, dict)
        assert isinstance(result.observed_characteristics, dict)
        assert isinstance(result.prediction_accuracy, dict)
        assert isinstance(result.residuals, dict)
        assert isinstance(result.validation_metrics, dict)
        assert isinstance(result.lessons_learned, list)
        
        # Check accuracy values
        for param, accuracy in result.prediction_accuracy.items():
            assert 0 <= accuracy <= 1
    
    def test_calculate_gas_volume_fraction(self, prediction_system):
        """Test gas volume fraction calculation"""
        volatile_content = {
            'H2O': 0.04,
            'CO2': 0.01,
            'SO2': 0.002
        }
        pressure = 1e8
        temperature = 1200
        
        result = prediction_system._calculate_gas_volume_fraction(
            volatile_content, pressure, temperature
        )
        
        assert 0 <= result <= 1
        assert isinstance(result, float)
    
    def test_evolve_magma_properties(self, prediction_system):
        """Test magma property evolution"""
        initial_conditions = {
            'gas_content': 0.02,
            'viscosity': 1e5,
            'temperature': 1200,
            'crystallinity': 0.2
        }
        mixing_degree = 0.6
        time_fraction = 0.5
        
        result = prediction_system._evolve_magma_properties(
            initial_conditions, mixing_degree, time_fraction
        )
        
        assert isinstance(result, dict)
        assert 'gas_content' in result
        assert 'viscosity' in result
        assert 'temperature' in result
        assert 'crystallinity' in result
        
        # Check that properties evolved reasonably
        assert result['gas_content'] >= initial_conditions['gas_content']
        assert result['viscosity'] >= initial_conditions['viscosity']
        assert result['crystallinity'] >= initial_conditions['crystallinity']
    
    def test_calculate_prediction_confidence(self, prediction_system):
        """Test prediction confidence calculation"""
        primary_parameter = 3.5
        supporting_data = {
            'param1': 1.0, 'param2': 2.0, 'param3': 3.0,
            'param4': 4.0, 'param5': 5.0
        }
        prediction_type = 'intensity'
        
        result = prediction_system._calculate_prediction_confidence(
            primary_parameter, supporting_data, prediction_type
        )
        
        assert 0 <= result <= 1
        assert isinstance(result, float)
    
    @pytest.mark.asyncio
    async def test_store_prediction_results(self, prediction_system):
        """Test storing prediction results in database"""
        # Create mock prediction result
        prediction_result = Mock()
        prediction_result.prediction_id = 'test_prediction_123'
        prediction_result.chamber_id = 'test_chamber_456'
        prediction_result.prediction_timestamp = datetime.now()
        prediction_result.mixing_degree = 0.65
        prediction_result.validation_score = 0.8
        prediction_result.confidence_level = 0.75
        
        # Mock sub-objects
        prediction_result.intensity_prediction = Mock()
        prediction_result.intensity_prediction.predicted_intensity = 3.5
        prediction_result.intensity_prediction.intensity_confidence = 0.8
        prediction_result.intensity_prediction.intensity_bounds = (2.5, 4.5)
        prediction_result.intensity_prediction.contributing_factors = {'mixing': 0.6}
        
        prediction_result.explosivity_forecast = Mock()
        prediction_result.explosivity_forecast.predicted_explosivity = 0.7
        prediction_result.explosivity_forecast.explosivity_index = 4
        prediction_result.explosivity_forecast.gas_pressure_buildup = 1e7
        prediction_result.explosivity_forecast.explosivity_confidence = 0.75
        
        prediction_result.discharge_rate_prediction = Mock()
        prediction_result.discharge_rate_prediction.predicted_discharge_rate = 200
        prediction_result.discharge_rate_prediction.peak_discharge_rate = 400
        prediction_result.discharge_rate_prediction.sustained_discharge_rate = 120
        prediction_result.discharge_rate_prediction.rheological_controls = {'viscosity': 0.5}
        
        prediction_result.style_prediction = Mock()
        prediction_result.style_prediction.predicted_style = 'explosive'
        prediction_result.style_prediction.style_probability = {'explosive': 0.8}
        prediction_result.style_prediction.controlling_parameters = {'gas_content': 0.6}
        prediction_result.style_prediction.style_confidence = 0.8
        
        prediction_result.hazard_assessment = Mock()
        prediction_result.hazard_assessment.hazard_zones = {'lava_flows': {}}
        prediction_result.hazard_assessment.population_exposure = {'high_risk': 1000}
        prediction_result.hazard_assessment.evacuation_recommendations = {'radius': 5000}
        
        prediction_result.uncertainty_quantification = Mock()
        prediction_result.uncertainty_quantification.total_uncertainty = {'intensity': 0.2}
        prediction_result.uncertainty_quantification.confidence_intervals = {'intensity': (3.0, 4.0)}
        
        # Mock database connection
        mock_conn = AsyncMock()
        prediction_system.db_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Run storage
        result = await prediction_system.store_prediction_results(prediction_result)
        
        # Assertions
        assert result == prediction_result.prediction_id
        mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_integration_full_prediction_workflow(self, prediction_system):
        """Test complete prediction workflow integration"""
        # Test data representing a complete mixing scenario
        mixing_degree = 0.7
        magma_properties = {
            'volatile_content': 0.05,
            'temperature': 1300,
            'viscosity': 2e6,
            'crystallinity': 0.3,
            'gas_content': 0.05,
            'ascent_rate': 0.3
        }
        chamber_conditions = {
            'pressure': 1.2e8
        }
        
        # Run intensity prediction
        intensity_result = await prediction_system.predict_eruption_intensity(
            mixing_degree, magma_properties, chamber_conditions
        )
        
        # Run style prediction
        environmental_conditions = {
            'water_interaction': 0.1,
            'depth_to_water': 200,
            'external_water': True
        }
        style_result = await prediction_system.predict_eruption_style(
            magma_properties, environmental_conditions
        )
        
        # Verify integration
        assert isinstance(intensity_result, EruptionIntensityPrediction)
        assert isinstance(style_result, EruptionStylePrediction)
        
        # Check that results are consistent
        if intensity_result.predicted_intensity > 4:
            assert style_result.predicted_style in ['explosive', 'phreatomagmatic', 'mixed']
        
        # Check that high gas content leads to explosive style
        if magma_properties['gas_content'] > 0.04:
            assert style_result.style_probability.get('explosive', 0) > 0.3


if __name__ == '__main__':
    pytest.main([__file__])