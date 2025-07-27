"""
Tests for Probabilistic Eruption Forecasting
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from core.eruption_forecasting import ProbabilisticEruptionForecaster
from models.volcanic_data import EruptionForecast

@pytest.fixture
def forecaster():
    """Create eruption forecaster for testing"""
    return ProbabilisticEruptionForecaster()

@pytest.fixture
def sample_monitoring_data():
    """Create sample monitoring data for testing"""
    return {
        'seismic': [],
        'deformation': [],
        'geochemical': [],
        'thermal': [],
        'quality_score': 0.8
    }

@pytest.mark.asyncio
async def test_generate_probabilistic_forecast(forecaster, sample_monitoring_data):
    """Test probabilistic eruption forecast generation"""
    volcano_id = "test_volcano"
    
    forecast = await forecaster.generate_probabilistic_forecast(
        volcano_id, sample_monitoring_data
    )
    
    assert isinstance(forecast, EruptionForecast)
    assert forecast.volcano_id == volcano_id
    assert isinstance(forecast.timestamp, datetime)
    assert isinstance(forecast.probability_by_timeframe, dict)
    assert 0.0 <= forecast.uncertainty <= 1.0
    assert 0.0 <= forecast.confidence <= 1.0
    assert isinstance(forecast.key_indicators, list)
    assert isinstance(forecast.model_ensemble_results, dict)
    
    # Check that all timeframes are present
    expected_timeframes = ["1_hour", "6_hours", "24_hours", "1_week", "1_month", "1_year"]
    for timeframe in expected_timeframes:
        assert timeframe in forecast.probability_by_timeframe
        probability = forecast.probability_by_timeframe[timeframe]
        assert 0.0 <= probability <= 1.0

@pytest.mark.asyncio
async def test_statistical_forecast(forecaster):
    """Test statistical forecasting component"""
    volcano_id = "kilauea"
    
    # Test with historical data
    historical_data = [
        {'date': datetime(2020, 12, 20), 'vei': 2},
        {'date': datetime(2021, 9, 29), 'vei': 1},
        {'date': datetime(2022, 11, 27), 'vei': 1}
    ]
    
    from core.eruption_forecasting import ForecastTimeframe
    timeframe = ForecastTimeframe("24_hours", 24)
    
    probability = await forecaster._statistical_forecast(
        volcano_id, timeframe, historical_data
    )
    
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 0.95  # Capped at 95%
    
    # Test with no historical data
    empty_probability = await forecaster._statistical_forecast(
        volcano_id, timeframe, []
    )
    
    assert isinstance(empty_probability, float)
    assert 0.0 <= empty_probability <= 1.0

@pytest.mark.asyncio
async def test_machine_learning_forecast(forecaster, sample_monitoring_data):
    """Test machine learning forecasting component"""
    volcano_id = "test_volcano"
    
    from core.eruption_forecasting import ForecastTimeframe
    timeframe = ForecastTimeframe("24_hours", 24)
    
    probability = await forecaster._machine_learning_forecast(
        volcano_id, timeframe, sample_monitoring_data
    )
    
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 0.9  # Capped at 90%

@pytest.mark.asyncio
async def test_physics_informed_forecast(forecaster, sample_monitoring_data):
    """Test physics-informed forecasting component"""
    volcano_id = "test_volcano"
    
    from core.eruption_forecasting import ForecastTimeframe
    timeframe = ForecastTimeframe("24_hours", 24)
    
    probability = await forecaster._physics_informed_forecast(
        volcano_id, timeframe, sample_monitoring_data
    )
    
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0

def test_extract_ml_features(forecaster, sample_monitoring_data):
    """Test ML feature extraction"""
    features = forecaster._extract_ml_features(sample_monitoring_data)
    
    assert isinstance(features, dict)
    
    expected_features = [
        'seismic_rate', 'avg_magnitude', 'deformation_rate',
        'gas_flux_change', 'thermal_anomaly'
    ]
    
    for feature in expected_features:
        assert feature in features
        assert isinstance(features[feature], (int, float))
        assert features[feature] >= 0.0

def test_combine_forecasts(forecaster):
    """Test forecast ensemble combination"""
    model_forecasts = {
        'statistical': 0.1,
        'machine_learning': 0.2,
        'physics_informed': 0.15
    }
    
    ensemble_forecast, uncertainty = forecaster._combine_forecasts(model_forecasts)
    
    assert isinstance(ensemble_forecast, float)
    assert isinstance(uncertainty, float)
    assert 0.0 <= ensemble_forecast <= 0.95
    assert 0.0 <= uncertainty <= 1.0
    
    # Should be weighted average
    expected = (0.3 * 0.1 + 0.4 * 0.2 + 0.3 * 0.15)
    assert abs(ensemble_forecast - expected) < 0.001

def test_calculate_forecast_confidence(forecaster, sample_monitoring_data):
    """Test forecast confidence calculation"""
    historical_data = [
        {'date': datetime(2020, 1, 1), 'vei': 2},
        {'date': datetime(2021, 1, 1), 'vei': 1},
        {'date': datetime(2022, 1, 1), 'vei': 1}
    ]
    
    confidence = forecaster._calculate_forecast_confidence(
        sample_monitoring_data, historical_data
    )
    
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 0.95

def test_identify_key_indicators(forecaster, sample_monitoring_data):
    """Test key indicator identification"""
    indicators = forecaster._identify_key_indicators(sample_monitoring_data)
    
    assert isinstance(indicators, list)
    assert len(indicators) > 0
    
    for indicator in indicators:
        assert isinstance(indicator, str)
        assert len(indicator) > 0

@pytest.mark.asyncio
async def test_volcano_type_mapping(forecaster):
    """Test volcano type mapping"""
    test_cases = [
        ("yellowstone", "caldera"),
        ("kilauea", "shield"),
        ("etna", "stratovolcano"),
        ("unknown_volcano", "stratovolcano")  # Default
    ]
    
    for volcano_id, expected_type in test_cases:
        volcano_type = await forecaster._get_volcano_type(volcano_id)
        assert volcano_type == expected_type

@pytest.mark.asyncio
async def test_historical_eruption_data(forecaster):
    """Test historical eruption data retrieval"""
    volcano_id = "kilauea"
    
    historical_data = await forecaster._get_historical_eruption_data(volcano_id)
    
    assert isinstance(historical_data, list)
    
    for eruption in historical_data:
        assert isinstance(eruption, dict)
        assert 'date' in eruption
        assert 'vei' in eruption
        assert isinstance(eruption['date'], datetime)
        assert isinstance(eruption['vei'], int)
        assert 0 <= eruption['vei'] <= 8

if __name__ == "__main__":
    pytest.main([__file__])