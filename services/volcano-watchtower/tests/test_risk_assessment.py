"""
Tests for Volcanic Risk Assessment Engine
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np

from core.risk_assessment import VolcanicRiskAssessmentEngine
from models.volcanic_data import (
    SeismicEvent, DeformationMeasurement, GeochemicalSample, 
    ThermalMeasurement, GeologicalLocation
)

@pytest.fixture
def risk_engine():
    """Create risk assessment engine for testing"""
    return VolcanicRiskAssessmentEngine()

@pytest.fixture
def sample_seismic_data():
    """Create sample seismic data for testing"""
    location = GeologicalLocation(44.428, -110.588, 2400, depth=5000)
    
    events = []
    for i in range(10):
        event = SeismicEvent(
            timestamp=datetime.utcnow() - timedelta(hours=i),
            location=location,
            magnitude=2.0 + i * 0.1,
            depth=5000 + i * 100,
            event_type="volcanic",
            confidence=0.9
        )
        events.append(event)
    
    return events

@pytest.fixture
def sample_deformation_data():
    """Create sample deformation data for testing"""
    location = GeologicalLocation(44.428, -110.588, 2400)
    
    measurements = []
    for i in range(5):
        measurement = DeformationMeasurement(
            timestamp=datetime.utcnow() - timedelta(days=i),
            location=location,
            displacement_vector=np.array([0.001 * i, 0.001 * i, 0.002 * i]),
            measurement_type="GPS",
            uncertainty=np.array([0.0001, 0.0001, 0.0002])
        )
        measurements.append(measurement)
    
    return measurements

@pytest.mark.asyncio
async def test_assess_volcanic_risk(risk_engine):
    """Test basic volcanic risk assessment"""
    volcano_id = "test_volcano"
    
    risk_assessment = await risk_engine.assess_volcanic_risk(volcano_id)
    
    assert risk_assessment is not None
    assert risk_assessment.volcano_id == volcano_id
    assert isinstance(risk_assessment.timestamp, datetime)
    assert 0 <= risk_assessment.alert_level.level <= 5
    assert 0.0 <= risk_assessment.composite_risk_score <= 1.0
    assert 0.0 <= risk_assessment.model_ensemble_agreement <= 1.0
    assert isinstance(risk_assessment.contributing_factors, list)
    assert isinstance(risk_assessment.eruption_probability, dict)

@pytest.mark.asyncio
async def test_seismic_risk_assessment(risk_engine, sample_seismic_data):
    """Test seismic risk component assessment"""
    risk_score = await risk_engine._assess_seismic_risk(sample_seismic_data)
    
    assert isinstance(risk_score, float)
    assert 0.0 <= risk_score <= 1.0
    
    # Test with empty data
    empty_risk = await risk_engine._assess_seismic_risk([])
    assert empty_risk == 0.1  # Baseline risk

@pytest.mark.asyncio
async def test_deformation_risk_assessment(risk_engine, sample_deformation_data):
    """Test deformation risk component assessment"""
    risk_score = await risk_engine._assess_deformation_risk(sample_deformation_data)
    
    assert isinstance(risk_score, float)
    assert 0.0 <= risk_score <= 1.0
    
    # Test with insufficient data
    insufficient_risk = await risk_engine._assess_deformation_risk(sample_deformation_data[:1])
    assert insufficient_risk == 0.1  # Baseline risk

@pytest.mark.asyncio
async def test_alert_level_determination(risk_engine):
    """Test alert level determination from risk scores"""
    # Test different risk levels
    test_cases = [
        (0.0, 0),   # Normal
        (0.3, 1),   # Advisory
        (0.5, 2),   # Watch
        (0.7, 3),   # Warning
        (0.9, 4),   # Alert
        (0.98, 5)   # Emergency
    ]
    
    for risk_score, expected_level in test_cases:
        alert_level = risk_engine._determine_alert_level(risk_score)
        assert alert_level.level == expected_level
        assert isinstance(alert_level.description, str)
        assert isinstance(alert_level.recommended_actions, list)
        assert len(alert_level.recommended_actions) > 0

@pytest.mark.asyncio
async def test_composite_risk_calculation(risk_engine):
    """Test composite risk score calculation"""
    risk_components = {
        'seismic': 0.5,
        'deformation': 0.3,
        'geochemical': 0.2,
        'thermal': 0.1,
        'historical': 0.4
    }
    
    composite_risk = risk_engine._calculate_composite_risk(risk_components)
    
    assert isinstance(composite_risk, float)
    assert 0.0 <= composite_risk <= 1.0
    
    # Should be weighted average
    expected = (0.25 * 0.5 + 0.25 * 0.3 + 0.20 * 0.2 + 0.15 * 0.1 + 0.15 * 0.4)
    assert abs(composite_risk - expected) < 0.001

@pytest.mark.asyncio
async def test_hazard_zone_generation(risk_engine):
    """Test hazard zone generation"""
    from models.volcanic_data import AlertLevel
    
    # Create test alert level
    alert_level = AlertLevel(
        level=3,
        description="Warning",
        recommended_actions=["Evacuate high-risk areas"],
        validity_period=(datetime.utcnow(), datetime.utcnow() + timedelta(hours=6)),
        confidence=0.8
    )
    
    # Mock eruption forecast
    eruption_forecast = type('MockForecast', (), {
        'probability_by_timeframe': {'24_hours': 0.3}
    })()
    
    hazard_zones = await risk_engine._generate_hazard_zones(
        "test_volcano", alert_level, eruption_forecast
    )
    
    assert isinstance(hazard_zones, dict)
    
    # Should have hazard zones for warning level
    if alert_level.level >= 2:
        assert len(hazard_zones) > 0
        for hazard_type, zones in hazard_zones.items():
            assert isinstance(hazard_type, str)
            assert isinstance(zones, np.ndarray)

if __name__ == "__main__":
    pytest.main([__file__])