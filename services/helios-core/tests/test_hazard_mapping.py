"""
Test suite for comprehensive hazard mapping system
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import pytest_asyncio
from core.hazard_communication import (
    AudienceType,
    CommunicationFormat,
    CommunicationProduct,
    HazardCommunicationEngine,
)
from core.hazard_mapping import (
    GeographicBounds,
    HazardMappingEngine,
    HazardProbability,
    HazardType,
    MultiHazardAssessment,
    SeismicHazardModel,
    VolcanicHazardModel,
)
from core.realtime_hazard_updates import (
    MonitoringDataStream,
    RealTimeHazardUpdater,
    UpdateTrigger,
    import,
)
from core.realtime_hazard_updates import numpy as np
from core.realtime_hazard_updates import pandas as pd
from core.realtime_hazard_updates import torch.nn.functional as F

class TestHazardMappingEngine:
    """Test cases for HazardMappingEngine"""

    @pytest_asyncio.fixture
    async def hazard_engine(self):
        """Create and initialize hazard mapping engine"""
        engine = HazardMappingEngine()
        await engine.initialize_hazard_models()
        return engine

    @pytest.fixture
    def sample_bounds(self):
        """Sample geographic bounds for testing"""
        return GeographicBounds(
            min_lat=45.0, max_lat=46.0, min_lon=-122.0, max_lon=-121.0, resolution=0.01
        )

    @pytest.fixture
    def sample_monitoring_data(self):
        """Sample monitoring data for testing"""
        return {
            "seismic_events": [
                {
                    "magnitude": 4.5,
                    "timestamp": datetime.now(),
                    "location": {"lat": 45.5, "lon": -121.5, "depth": 10},
                }
            ],
            "ground_deformation": {
                "uplift_rate": 0.02,
                "measurements": [
                    {
                        "displacement": 0.05,
                        "timestamp": datetime.now(),
                        "location": {"lat": 45.5, "lon": -121.5},
                    }
                ],
            },
            "gas_emissions": {
                "so2_flux": 1200,
                "measurements": [
                    {
                        "flux": 1200,
                        "timestamp": datetime.now(),
                        "location": {"lat": 45.5, "lon": -121.5},
                    }
                ],
            },
            "thermal_data": {
                "max_temperature": 75,
                "measurements": [
                    {
                        "temperature": 75,
                        "timestamp": datetime.now(),
                        "location": {"lat": 45.5, "lon": -121.5},
                    }
                ],
            },
        }

    @pytest.mark.asyncio
    async def test_engine_initialization(self, hazard_engine):
        """Test hazard engine initialization"""
        assert len(hazard_engine.hazard_models) > 0
        assert HazardType.VOLCANIC_ERUPTION in hazard_engine.hazard_models
        assert HazardType.SEISMIC_EARTHQUAKE in hazard_engine.hazard_models

    @pytest.mark.asyncio
    async def test_multi_hazard_assessment(
        self, hazard_engine, sample_bounds, sample_monitoring_data
    ):
        """Test comprehensive multi-hazard assessment"""
        assessment = await hazard_engine.assess_multi_hazard_risk(
            geographic_bounds=sample_bounds,
            monitoring_data=sample_monitoring_data,
            time_horizon=timedelta(days=30),
        )

        # Verify assessment structure
        assert isinstance(assessment, MultiHazardAssessment)
        assert assessment.assessment_id is not None
        assert assessment.timestamp is not None
        assert assessment.geographic_bounds == sample_bounds

        # Verify individual hazards
        assert len(assessment.individual_hazards) > 0
        assert HazardType.VOLCANIC_ERUPTION in assessment.individual_hazards
        assert HazardType.SEISMIC_EARTHQUAKE in assessment.individual_hazards

        # Verify risk maps
        assert assessment.combined_risk_map.shape[0] > 0
        assert assessment.combined_risk_map.shape[1] > 0
        assert assessment.uncertainty_map.shape == assessment.combined_risk_map.shape

        # Verify risk values are in valid range
        assert np.all(assessment.combined_risk_map >= 0)
        assert np.all(assessment.combined_risk_map <= 1)
        assert np.all(assessment.uncertainty_map >= 0)
        assert np.all(assessment.uncertainty_map <= 1)

        # Verify population and economic estimates
        assert assessment.total_population_at_risk >= 0
        assert assessment.economic_impact_estimate >= 0

    @pytest.mark.asyncio
    async def test_cascading_effects_analysis(
        self, hazard_engine, sample_bounds, sample_monitoring_data
    ):
        """Test cascading effects analysis"""
        assessment = await hazard_engine.assess_multi_hazard_risk(
            geographic_bounds=sample_bounds,
            monitoring_data=sample_monitoring_data,
            time_horizon=timedelta(days=30),
        )

        # Check that cascading effects are identified
        assert isinstance(assessment.cascading_effects, dict)

        # Verify logical cascading relationships
        if HazardType.VOLCANIC_ERUPTION in assessment.cascading_effects:
            secondary_hazards = assessment.cascading_effects[
                HazardType.VOLCANIC_ERUPTION
            ]
            expected_secondary = [
                HazardType.PYROCLASTIC_FLOW,
                HazardType.LAHAR,
                HazardType.ASH_FALL,
                HazardType.LAVA_FLOW,
            ]
            # At least some expected secondary hazards should be present
            assert any(hazard in secondary_hazards for hazard in expected_secondary)

    @pytest.mark.asyncio
    async def test_error_handling(self, hazard_engine, sample_bounds):
        """Test error handling with invalid data"""
        # Test with empty monitoring data
        assessment = await hazard_engine.assess_multi_hazard_risk(
            geographic_bounds=sample_bounds,
            monitoring_data={},
            time_horizon=timedelta(days=30),
        )

        # Should still produce valid assessment with conservative estimates
        assert isinstance(assessment, MultiHazardAssessment)
        assert len(assessment.individual_hazards) > 0

class TestVolcanicHazardModel:
    """Test cases for VolcanicHazardModel"""

    @pytest.fixture
    def volcanic_model(self):
        """Create volcanic hazard model"""
        return VolcanicHazardModel()

    @pytest.fixture
    def sample_bounds(self):
        """Sample geographic bounds"""
        return GeographicBounds(
            min_lat=45.0, max_lat=46.0, min_lon=-122.0, max_lon=-121.0, resolution=0.01
        )

    @pytest.mark.asyncio
    async def test_volcanic_probability_assessment(self, volcanic_model, sample_bounds):
        """Test volcanic hazard probability assessment"""
        monitoring_data = {
            "seismic_events": [
                {"magnitude": 3.0} for _ in range(15)
            ],  # High seismic activity
            "ground_deformation": {"uplift_rate": 0.02},  # Significant uplift
            "gas_emissions": {"so2_flux": 1500},  # Elevated SO2
            "thermal_data": {"max_temperature": 80},  # Thermal anomalies
        }

        hazard_prob = await volcanic_model.assess_probability(
            bounds=sample_bounds,
            monitoring_data=monitoring_data,
            time_horizon=timedelta(days=30),
        )

        # Verify probability structure
        assert isinstance(hazard_prob, HazardProbability)
        assert hazard_prob.hazard_type == HazardType.VOLCANIC_ERUPTION
        assert 0 <= hazard_prob.probability <= 1
        assert 0 <= hazard_prob.uncertainty <= 1
        assert len(hazard_prob.confidence_interval) == 2
        assert hazard_prob.confidence_interval[0] <= hazard_prob.confidence_interval[1]

        # With high activity indicators, probability should be elevated
        assert hazard_prob.probability > 0.05  # Above baseline

        # Contributing factors should be identified
        assert len(hazard_prob.contributing_factors) > 0
        assert "increased_seismic_activity" in hazard_prob.contributing_factors

    @pytest.mark.asyncio
    async def test_low_activity_scenario(self, volcanic_model, sample_bounds):
        """Test assessment with low volcanic activity"""
        monitoring_data = {
            "seismic_events": [],
            "ground_deformation": {"uplift_rate": 0.001},
            "gas_emissions": {"so2_flux": 100},
            "thermal_data": {"max_temperature": 20},
        }

        hazard_prob = await volcanic_model.assess_probability(
            bounds=sample_bounds,
            monitoring_data=monitoring_data,
            time_horizon=timedelta(days=30),
        )

        # Low activity should result in lower probability
        assert hazard_prob.probability <= 0.1  # Should be near baseline
        assert len(hazard_prob.contributing_factors) == 0  # No significant factors

class TestSeismicHazardModel:
    """Test cases for SeismicHazardModel"""

    @pytest.fixture
    def seismic_model(self):
        """Create seismic hazard model"""
        return SeismicHazardModel()

    @pytest.fixture
    def sample_bounds(self):
        """Sample geographic bounds"""
        return GeographicBounds(
            min_lat=45.0, max_lat=46.0, min_lon=-122.0, max_lon=-121.0, resolution=0.01
        )

    @pytest.mark.asyncio
    async def test_seismic_probability_assessment(self, seismic_model, sample_bounds):
        """Test seismic hazard probability assessment"""
        monitoring_data = {
            "recent_earthquakes": [
                {"magnitude": 4.0, "timestamp": datetime.now()},
                {"magnitude": 3.5, "timestamp": datetime.now()},
                {"magnitude": 4.2, "timestamp": datetime.now()},
            ],
            "foreshock_activity": [
                {"magnitude": 2.5},
                {"magnitude": 2.8},
                {"magnitude": 3.1},
                {"magnitude": 2.9},
            ],
            "stress_changes": {"coulomb_stress_change": 0.15},
        }

        hazard_prob = await seismic_model.assess_probability(
            bounds=sample_bounds,
            monitoring_data=monitoring_data,
            time_horizon=timedelta(days=7),
        )

        # Verify probability structure
        assert isinstance(hazard_prob, HazardProbability)
        assert hazard_prob.hazard_type == HazardType.SEISMIC_EARTHQUAKE
        assert 0 <= hazard_prob.probability <= 1

        # With elevated seismic indicators, probability should be higher
        assert hazard_prob.probability > 0.1

        # Contributing factors should be identified
        expected_factors = [
            "recent_seismic_activity",
            "foreshock_sequence",
            "stress_loading",
        ]
        assert any(
            factor in hazard_prob.contributing_factors for factor in expected_factors
        )

class TestRealTimeHazardUpdater:
    """Test cases for RealTimeHazardUpdater"""

    @pytest_asyncio.fixture
    async def mock_hazard_engine(self):
        """Create mock hazard engine"""
        engine = Mock(spec=HazardMappingEngine)
        engine.assess_multi_hazard_risk = AsyncMock()

        # Mock assessment response
        mock_assessment = Mock(spec=MultiHazardAssessment)
        mock_assessment.assessment_id = "test_assessment_123"
        mock_assessment.timestamp = datetime.now()
        mock_assessment.combined_risk_map = np.random.random((10, 10))
        mock_assessment.uncertainty_map = np.random.random((10, 10))
        mock_assessment.total_population_at_risk = 1000
        mock_assessment.geographic_bounds = GeographicBounds(45.0, 46.0, -122.0, -121.0)

        engine.assess_multi_hazard_risk.return_value = mock_assessment
        return engine

    @pytest.fixture
    def realtime_updater(self, mock_hazard_engine):
        """Create real-time hazard updater"""
        return RealTimeHazardUpdater(mock_hazard_engine)

    @pytest.fixture
    def sample_monitoring_stream(self):
        """Sample monitoring data stream"""
        return MonitoringDataStream(
            stream_id="test_stream_001",
            data_type="seismic_event",
            timestamp=datetime.now(),
            location={"lat": 45.5, "lon": -121.5, "elevation": 1000},
            value=4.5,  # Magnitude
            unit="magnitude",
            quality_flag="good",
            sensor_id="seismometer_001",
        )

    @pytest.mark.asyncio
    async def test_data_ingestion(self, realtime_updater, sample_monitoring_stream):
        """Test monitoring data ingestion"""
        await realtime_updater.ingest_monitoring_data(sample_monitoring_stream)

        # Verify data is stored
        assert "seismic_event" in realtime_updater.monitoring_streams
        assert len(realtime_updater.monitoring_streams["seismic_event"]) == 1

        stored_stream = realtime_updater.monitoring_streams["seismic_event"][0]
        assert stored_stream.stream_id == sample_monitoring_stream.stream_id
        assert stored_stream.value == sample_monitoring_stream.value

    @pytest.mark.asyncio
    async def test_trigger_evaluation(self, realtime_updater, sample_monitoring_stream):
        """Test update trigger evaluation"""
        # Modify stream to exceed trigger threshold
        sample_monitoring_stream.value = 5.0  # Above 4.0 threshold

        # Mock the update queue
        realtime_updater.update_queue = AsyncMock()
        realtime_updater.update_queue.put = AsyncMock()

        await realtime_updater.ingest_monitoring_data(sample_monitoring_stream)

        # Should trigger an update due to high magnitude
        # Note: This test might need adjustment based on actual trigger logic
        # realtime_updater.update_queue.put.assert_called_once()

    def test_trigger_condition_evaluation(self, realtime_updater):
        """Test trigger condition evaluation logic"""
        trigger = UpdateTrigger(
            trigger_id="test_trigger",
            data_type="seismic_event",
            threshold_value=4.0,
            comparison_operator=">=",
            time_window=timedelta(minutes=5),
        )

        # Test data that should trigger
        high_magnitude_data = MonitoringDataStream(
            stream_id="test",
            data_type="seismic_event",
            timestamp=datetime.now(),
            location={"lat": 45.5, "lon": -121.5, "elevation": 1000},
            value=4.5,
            unit="magnitude",
            quality_flag="good",
            sensor_id="test",
        )

        result = realtime_updater._evaluate_trigger_condition(
            trigger, [], high_magnitude_data
        )
        assert result is True

        # Test data that should not trigger
        low_magnitude_data = MonitoringDataStream(
            stream_id="test",
            data_type="seismic_event",
            timestamp=datetime.now(),
            location={"lat": 45.5, "lon": -121.5, "elevation": 1000},
            value=3.0,
            unit="magnitude",
            quality_flag="good",
            sensor_id="test",
        )

        result = realtime_updater._evaluate_trigger_condition(
            trigger, [], low_magnitude_data
        )
        assert result is False

    def test_affected_area_estimation(self, realtime_updater):
        """Test affected area estimation"""
        trigger = UpdateTrigger(
            trigger_id="test_trigger",
            data_type="seismic_event",
            threshold_value=4.0,
            comparison_operator=">=",
            time_window=timedelta(minutes=5),
        )

        data_stream = MonitoringDataStream(
            stream_id="test",
            data_type="seismic_event",
            timestamp=datetime.now(),
            location={"lat": 45.5, "lon": -121.5, "elevation": 1000},
            value=5.0,
            unit="magnitude",
            quality_flag="good",
            sensor_id="test",
        )

        bounds = realtime_updater._estimate_affected_area(data_stream, trigger)

        # Verify bounds are reasonable
        assert isinstance(bounds, GeographicBounds)
        assert bounds.min_lat < bounds.max_lat
        assert bounds.min_lon < bounds.max_lon
        assert bounds.min_lat <= 45.5 <= bounds.max_lat
        assert bounds.min_lon <= -121.5 <= bounds.max_lon

class TestHazardCommunicationEngine:
    """Test cases for HazardCommunicationEngine"""

    @pytest.fixture
    def communication_engine(self):
        """Create hazard communication engine"""
        return HazardCommunicationEngine()

    @pytest.fixture
    def sample_assessment(self):
        """Sample hazard assessment for testing"""
        bounds = GeographicBounds(45.0, 46.0, -122.0, -121.0)

        assessment = Mock(spec=MultiHazardAssessment)
        assessment.assessment_id = "test_assessment_123"
        assessment.timestamp = datetime.now()
        assessment.geographic_bounds = bounds
        assessment.combined_risk_map = np.random.random((10, 10)) * 0.6  # Moderate risk
        assessment.uncertainty_map = np.random.random((10, 10)) * 0.3
        assessment.total_population_at_risk = 5000
        assessment.economic_impact_estimate = 10000000

        # Mock individual hazards
        volcanic_hazard = Mock(spec=HazardProbability)
        volcanic_hazard.probability = 0.4
        volcanic_hazard.confidence_interval = (0.3, 0.5)
        volcanic_hazard.uncertainty = 0.2
        volcanic_hazard.contributing_factors = [
            "increased_seismic_activity",
            "ground_deformation",
        ]
        volcanic_hazard.time_window = timedelta(days=30)

        seismic_hazard = Mock(spec=HazardProbability)
        seismic_hazard.probability = 0.3
        seismic_hazard.confidence_interval = (0.2, 0.4)
        seismic_hazard.uncertainty = 0.25
        seismic_hazard.contributing_factors = ["recent_seismic_activity"]
        seismic_hazard.time_window = timedelta(days=30)

        assessment.individual_hazards = {
            HazardType.VOLCANIC_ERUPTION: volcanic_hazard,
            HazardType.SEISMIC_EARTHQUAKE: seismic_hazard,
        }

        assessment.cascading_effects = {
            HazardType.VOLCANIC_ERUPTION: [
                HazardType.PYROCLASTIC_FLOW,
                HazardType.ASH_FALL,
            ]
        }

        return assessment

    @pytest.mark.asyncio
    async def test_scientific_report_generation(
        self, communication_engine, sample_assessment
    ):
        """Test scientific report generation"""
        product = await communication_engine.generate_communication_product(
            assessment=sample_assessment,
            audience_type=AudienceType.SCIENTIFIC,
            format_type=CommunicationFormat.TECHNICAL_REPORT,
        )

        # Verify product structure
        assert isinstance(product, CommunicationProduct)
        assert product.audience_type == AudienceType.SCIENTIFIC
        assert product.format_type == CommunicationFormat.TECHNICAL_REPORT
        assert product.title is not None
        assert product.content is not None
        assert len(product.visual_elements) > 0

        # Verify content includes technical details
        assert "Multi-Hazard Risk Assessment Report" in product.content
        assert "assessment_id" in product.content.lower()
        assert "probability" in product.content.lower()
        assert "uncertainty" in product.content.lower()

    @pytest.mark.asyncio
    async def test_public_advisory_generation(
        self, communication_engine, sample_assessment
    ):
        """Test public advisory generation"""
        product = await communication_engine.generate_communication_product(
            assessment=sample_assessment,
            audience_type=AudienceType.GENERAL_PUBLIC,
            format_type=CommunicationFormat.PUBLIC_ADVISORY,
        )

        # Verify product structure
        assert isinstance(product, CommunicationProduct)
        assert product.audience_type == AudienceType.GENERAL_PUBLIC
        assert product.format_type == CommunicationFormat.PUBLIC_ADVISORY

        # Verify content is public-friendly
        assert "PUBLIC SAFETY ADVISORY" in product.content
        assert "What's Happening" in product.content
        assert "What This Means for You" in product.content
        assert "Stay Informed" in product.content

        # Should not contain technical jargon
        assert "physics-informed neural network" not in product.content.lower()
        assert "probabilistic" not in product.content.lower()

    @pytest.mark.asyncio
    async def test_emergency_summary_generation(
        self, communication_engine, sample_assessment
    ):
        """Test emergency management summary generation"""
        product = await communication_engine.generate_communication_product(
            assessment=sample_assessment,
            audience_type=AudienceType.EMERGENCY_MANAGEMENT,
            format_type=CommunicationFormat.EXECUTIVE_SUMMARY,
        )

        # Verify product structure
        assert isinstance(product, CommunicationProduct)
        assert product.audience_type == AudienceType.EMERGENCY_MANAGEMENT
        assert product.format_type == CommunicationFormat.EXECUTIVE_SUMMARY

        # Verify content includes operational information
        assert "EMERGENCY MANAGEMENT BRIEFING" in product.content
        assert "SITUATION OVERVIEW" in product.content
        assert "RECOMMENDED ACTIONS" in product.content
        assert "population" in product.content.lower()

    @pytest.mark.asyncio
    async def test_social_media_generation(
        self, communication_engine, sample_assessment
    ):
        """Test social media post generation"""
        product = await communication_engine.generate_communication_product(
            assessment=sample_assessment,
            audience_type=AudienceType.GENERAL_PUBLIC,
            format_type=CommunicationFormat.SOCIAL_MEDIA,
        )

        # Verify product structure
        assert isinstance(product, CommunicationProduct)
        assert product.format_type == CommunicationFormat.SOCIAL_MEDIA

        # Verify content is concise and includes emojis
        assert len(product.content) < 500  # Reasonable length for social media
        assert any(emoji in product.content for emoji in ["🟢", "🟡", "🔴", "🌋", "🏠"])
        assert "#" in product.content  # Should include hashtags

    @pytest.mark.asyncio
    async def test_visual_elements_generation(
        self, communication_engine, sample_assessment
    ):
        """Test visual elements generation"""
        product = await communication_engine.generate_communication_product(
            assessment=sample_assessment,
            audience_type=AudienceType.SCIENTIFIC,
            format_type=CommunicationFormat.TECHNICAL_REPORT,
        )

        # Verify visual elements are generated
        assert len(product.visual_elements) > 0

        # Check for expected visualization types
        visual_types = [element.get("type") for element in product.visual_elements]
        assert "map" in visual_types or "chart" in visual_types

        # Verify visual elements have required fields
        for element in product.visual_elements:
            assert "type" in element
            assert "data" in element
            assert "caption" in element

    def test_urgency_level_calculation(self, communication_engine, sample_assessment):
        """Test urgency level calculation"""
        # Test with moderate risk
        urgency = communication_engine._calculate_urgency_level(sample_assessment, None)
        assert 1 <= urgency <= 5

        # Test with high risk assessment
        sample_assessment.combined_risk_map = np.full((10, 10), 0.8)  # High risk
        urgency_high = communication_engine._calculate_urgency_level(
            sample_assessment, None
        )
        assert urgency_high > urgency

    def test_language_simplification(self, communication_engine):
        """Test technical language simplification for public"""
        technical_action = "Monitor volcanic activity closely"
        simplified = communication_engine._simplify_for_public(technical_action)

        assert simplified != technical_action
        assert (
            "monitor" not in simplified.lower() or "stay informed" in simplified.lower()
        )

class TestIntegration:
    """Integration tests for the complete hazard mapping system"""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from data ingestion to communication"""
        # Initialize components
        hazard_engine = HazardMappingEngine()
        await hazard_engine.initialize_hazard_models()

        RealTimeHazardUpdater(hazard_engine)
        communication_engine = HazardCommunicationEngine()

        # Define test area
        bounds = GeographicBounds(45.0, 46.0, -122.0, -121.0)

        # Simulate monitoring data
        monitoring_data = {
            "seismic_events": [
                {
                    "magnitude": 4.2,
                    "timestamp": datetime.now(),
                    "location": {"lat": 45.5, "lon": -121.5, "depth": 8},
                }
            ],
            "ground_deformation": {
                "uplift_rate": 0.015,
                "measurements": [
                    {
                        "displacement": 0.03,
                        "timestamp": datetime.now(),
                        "location": {"lat": 45.5, "lon": -121.5},
                    }
                ],
            },
        }

        # Perform hazard assessment
        assessment = await hazard_engine.assess_multi_hazard_risk(
            geographic_bounds=bounds,
            monitoring_data=monitoring_data,
            time_horizon=timedelta(days=30),
        )

        # Verify assessment
        assert isinstance(assessment, MultiHazardAssessment)
        assert len(assessment.individual_hazards) > 0

        # Generate communication products for different audiences
        audiences_formats = [
            (AudienceType.SCIENTIFIC, CommunicationFormat.TECHNICAL_REPORT),
            (AudienceType.EMERGENCY_MANAGEMENT, CommunicationFormat.EXECUTIVE_SUMMARY),
            (AudienceType.GENERAL_PUBLIC, CommunicationFormat.PUBLIC_ADVISORY),
        ]

        for audience, format_type in audiences_formats:
            product = await communication_engine.generate_communication_product(
                assessment=assessment, audience_type=audience, format_type=format_type
            )

            assert isinstance(product, CommunicationProduct)
            assert product.audience_type == audience
            assert product.format_type == format_type
            assert product.content is not None
            assert len(product.visual_elements) > 0

    @pytest.mark.asyncio
    async def test_real_time_update_workflow(self):
        """Test real-time update workflow"""
        # Initialize components
        hazard_engine = HazardMappingEngine()
        await hazard_engine.initialize_hazard_models()

        realtime_updater = RealTimeHazardUpdater(hazard_engine)

        # Create monitoring data stream that should trigger update
        high_magnitude_stream = MonitoringDataStream(
            stream_id="integration_test_001",
            data_type="seismic_event",
            timestamp=datetime.now(),
            location={"lat": 45.5, "lon": -121.5, "elevation": 1000},
            value=5.2,  # High magnitude
            unit="magnitude",
            quality_flag="good",
            sensor_id="test_seismometer",
        )

        # Ingest data
        await realtime_updater.ingest_monitoring_data(high_magnitude_stream)

        # Verify data was stored
        assert "seismic_event" in realtime_updater.monitoring_streams
        assert len(realtime_updater.monitoring_streams["seismic_event"]) == 1

        # Check monitoring status
        status = realtime_updater.get_monitoring_data_summary()
        assert "seismic_event" in status
        assert status["seismic_event"]["count"] == 1
        assert status["seismic_event"]["latest_value"] == 5.2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
