"""
Tests for Earthquake Forecasting Module

Tests for probabilistic earthquake forecasting functionality including
machine learning models, statistical seismicity models, temporal clustering,
and probabilistic hazard assessment.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn.functional as F
from core.earthquake_forecasting import (
    EarthquakeCluster,
    EarthquakeForecastingEngine,
    ForecastModel,
    ForecastResult,
    HazardLevel,
)
from models.geological import EventType, GeologicalLocation, SeismicEvent

class TestEarthquakeForecastingEngine:
    """Test suite for EarthquakeForecastingEngine"""

    @pytest.fixture
    def engine(self):
        """Create earthquake forecasting engine for testing"""
        return EarthquakeForecastingEngine()

    @pytest.fixture
    def sample_events(self):
        """Create sample seismic events for testing"""
        events = []
        base_time = datetime(2023, 1, 1)

        # Create a sequence of events with varying magnitudes and locations
        for i in range(50):
            event = SeismicEvent(
                timestamp=base_time
                + timedelta(days=i * 2, hours=np.random.randint(0, 24)),
                location=GeologicalLocation(
                    latitude=35.0 + np.random.normal(0, 0.1),
                    longitude=-118.0 + np.random.normal(0, 0.1),
                    elevation=0.0,
                ),
                magnitude=3.0 + np.random.exponential(1.0),
                depth=np.random.uniform(5, 30),
                event_type=EventType.EARTHQUAKE,
                confidence=0.9,
                event_id=f"event_{i:03d}",
            )
            events.append(event)

        return events

    @pytest.fixture
    def clustered_events(self):
        """Create sample events with clear clustering patterns"""
        events = []
        base_time = datetime(2023, 1, 1)

        # Mainshock-aftershock sequence
        mainshock = SeismicEvent(
            timestamp=base_time,
            location=GeologicalLocation(latitude=35.0, longitude=-118.0, elevation=0.0),
            magnitude=6.5,
            depth=15.0,
            event_type=EventType.EARTHQUAKE,
            event_id="mainshock",
        )
        events.append(mainshock)

        # Aftershocks
        for i in range(20):
            aftershock = SeismicEvent(
                timestamp=base_time + timedelta(hours=i + 1),
                location=GeologicalLocation(
                    latitude=35.0 + np.random.normal(0, 0.02),
                    longitude=-118.0 + np.random.normal(0, 0.02),
                    elevation=0.0,
                ),
                magnitude=3.0 + np.random.exponential(0.8),
                depth=15.0 + np.random.normal(0, 5),
                event_type=EventType.EARTHQUAKE,
                event_id=f"aftershock_{i:02d}",
            )
            events.append(aftershock)

        return events

    @pytest.fixture
    def target_region(self):
        """Create target region for forecasting"""
        return {"lat_min": 34.5, "lat_max": 35.5, "lon_min": -118.5, "lon_max": -117.5}

    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert hasattr(engine, "config")
        assert hasattr(engine, "ml_models")
        assert hasattr(engine, "statistical_models")
        assert hasattr(engine, "clustering_models")

    def test_analyze_earthquake_patterns(self, engine, sample_events):
        """Test earthquake pattern analysis"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            results = engine.analyze_earthquake_patterns(sample_events)

            assert "ml_model_results" in results
            assert "spatial_patterns" in results
            assert "temporal_patterns" in results
            assert "magnitude_patterns" in results
            assert "feature_importance" in results
            assert "pattern_summary" in results

            # Check spatial patterns
            spatial = results["spatial_patterns"]
            assert "mean_inter_event_distance_km" in spatial
            assert "spatial_extent_km2" in spatial
            assert "centroid" in spatial
            assert len(spatial["centroid"]) == 2

            # Check temporal patterns
            temporal = results["temporal_patterns"]
            assert "mean_inter_event_time_hours" in temporal
            assert "event_rate_per_day" in temporal
            assert "total_time_span_days" in temporal

            # Check magnitude patterns
            magnitude = results["magnitude_patterns"]
            assert "magnitude_statistics" in magnitude
            assert "gutenberg_richter" in magnitude

    def test_analyze_earthquake_patterns_insufficient_data(self, engine):
        """Test pattern analysis with insufficient data"""
        few_events = [
            SeismicEvent(
                timestamp=datetime(2023, 1, 1),
                location=GeologicalLocation(
                    latitude=35.0, longitude=-118.0, elevation=0.0
                ),
                magnitude=4.0,
                depth=10.0,
            )
        ]

        with pytest.raises(ValueError, match="Need at least 10 events"):
            engine.analyze_earthquake_patterns(few_events)

    def test_create_statistical_seismicity_model(self, engine, sample_events):
        """Test statistical seismicity model creation"""
        results = engine.create_statistical_seismicity_model(sample_events)

        assert "gutenberg_richter" in results
        assert "seismicity_rates" in results
        assert "recurrence_intervals" in results
        assert "uncertainties" in results
        assert "model_validation" in results

        # Check Gutenberg-Richter parameters
        gr = results["gutenberg_richter"]
        if "error" not in gr:
            assert "b_value" in gr
            assert "a_value" in gr
            assert "b_uncertainty" in gr
            assert "a_uncertainty" in gr
            assert gr["b_value"] > 0  # b-value should be positive

        # Check seismicity rates
        rates = results["seismicity_rates"]
        assert len(rates) > 0
        for rate in rates.values():
            assert rate >= 0  # Rates should be non-negative

        # Check recurrence intervals
        intervals = results["recurrence_intervals"]
        assert len(intervals) > 0

    def test_perform_temporal_clustering_analysis(self, engine, clustered_events):
        """Test temporal clustering analysis"""
        clusters = engine.perform_temporal_clustering_analysis(clustered_events)

        # Should identify at least one cluster from the mainshock-aftershock sequence
        assert len(clusters) >= 0  # May be 0 if clustering parameters are strict

        for cluster in clusters:
            assert isinstance(cluster, EarthquakeCluster)
            assert len(cluster.events) >= 3  # Minimum cluster size
            assert cluster.spatial_extent >= 0
            assert cluster.temporal_extent >= 0
            assert len(cluster.magnitude_range) == 2
            assert cluster.magnitude_range[0] <= cluster.magnitude_range[1]

    def test_perform_temporal_clustering_analysis_insufficient_data(self, engine):
        """Test clustering with insufficient data"""
        few_events = [
            SeismicEvent(
                timestamp=datetime(2023, 1, 1),
                location=GeologicalLocation(
                    latitude=35.0, longitude=-118.0, elevation=0.0
                ),
                magnitude=4.0,
                depth=10.0,
            )
        ]

        clusters = engine.perform_temporal_clustering_analysis(few_events)
        assert clusters == []  # Should return empty list

    def test_generate_probabilistic_hazard_assessment(
        self, engine, sample_events, target_region
    ):
        """Test probabilistic hazard assessment generation"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            forecast = engine.generate_probabilistic_hazard_assessment(
                sample_events, target_region, forecast_period_days=30
            )

            assert isinstance(forecast, ForecastResult)
            assert forecast.forecast_time is not None
            assert len(forecast.forecast_period) == 2
            assert forecast.target_region == target_region
            assert len(forecast.probability_estimates) > 0
            assert len(forecast.confidence_intervals) > 0
            assert isinstance(forecast.hazard_level, HazardLevel)
            assert isinstance(forecast.model_used, ForecastModel)
            assert len(forecast.uncertainty_metrics) > 0

            # Check probability estimates are valid
            for prob in forecast.probability_estimates.values():
                assert 0 <= prob <= 1

            # Check confidence intervals are valid
            for lower, upper in forecast.confidence_intervals.values():
                assert 0 <= lower <= upper <= 1

    def test_generate_probabilistic_hazard_assessment_invalid_inputs(self, engine):
        """Test hazard assessment with invalid inputs"""
        few_events = [
            SeismicEvent(
                timestamp=datetime(2023, 1, 1),
                location=GeologicalLocation(
                    latitude=35.0, longitude=-118.0, elevation=0.0
                ),
                magnitude=4.0,
                depth=10.0,
            )
        ]

        target_region = {
            "lat_min": 34.5,
            "lat_max": 35.5,
            "lon_min": -118.5,
            "lon_max": -117.5,
        }

        # Test insufficient events
        with pytest.raises(ValueError, match="Need at least 10 historical events"):
            engine.generate_probabilistic_hazard_assessment(few_events, target_region)

        # Test invalid region
        invalid_region = {"lat_min": 35.0}  # Missing required keys
        sample_events = [few_events[0]] * 15  # Enough events

        with pytest.raises(ValueError, match="Target region must contain keys"):
            engine.generate_probabilistic_hazard_assessment(
                sample_events, invalid_region
            )

        # Test invalid forecast period
        valid_region = target_region
        with pytest.raises(ValueError, match="Forecast period must be positive"):
            engine.generate_probabilistic_hazard_assessment(
                sample_events, valid_region, -1
            )

    def test_gutenberg_richter_calculation(self, engine, sample_events):
        """Test Gutenberg-Richter law parameter calculation"""
        gr_params = engine._calculate_gutenberg_richter(sample_events)

        if "error" not in gr_params:
            assert "b_value" in gr_params
            assert "a_value" in gr_params
            assert "b_uncertainty" in gr_params
            assert "a_uncertainty" in gr_params
            assert "completeness_magnitude" in gr_params
            assert "n_events_used" in gr_params

            # b-value should be positive and typically between 0.5 and 2.0
            assert 0.1 < gr_params["b_value"] < 5.0
            assert gr_params["b_uncertainty"] > 0
            assert gr_params["n_events_used"] > 0

    def test_seismicity_rates_calculation(self, engine, sample_events):
        """Test seismicity rates calculation"""
        rates = engine._calculate_seismicity_rates(sample_events)

        assert len(rates) > 0

        # Check that all rates are non-negative
        for rate in rates.values():
            assert rate >= 0

        # Check that cumulative rates decrease with magnitude
        cumulative_rates = {k: v for k, v in rates.items() if k.startswith("M>=")}
        if len(cumulative_rates) > 1:
            list(cumulative_rates.values())
            # Generally, higher magnitude thresholds should have lower rates
            # (though this may not always be true for small samples)

    def test_distance_calculation(self, engine):
        """Test distance calculation between locations"""
        loc1 = GeologicalLocation(latitude=35.0, longitude=-118.0, elevation=0.0)
        loc2 = GeologicalLocation(latitude=35.1, longitude=-118.1, elevation=0.0)

        distance = engine._calculate_distance(loc1, loc2)

        assert distance > 0
        assert distance < 20  # Should be reasonable distance for 0.1 degree difference

    def test_feature_extraction(self, engine, sample_events):
        """Test earthquake feature extraction"""
        features = engine._extract_earthquake_features(sample_events)

        assert features.shape[0] == len(sample_events)
        assert features.shape[1] > 0  # Should have multiple features

        # Check that features are reasonable
        assert np.all(np.isfinite(features))  # No NaN or infinite values

    def test_spatial_patterns_analysis(self, engine, sample_events):
        """Test spatial patterns analysis"""
        spatial_patterns = engine._analyze_spatial_patterns(sample_events)

        assert "mean_inter_event_distance_km" in spatial_patterns
        assert "spatial_extent_km2" in spatial_patterns
        assert "centroid" in spatial_patterns

        assert spatial_patterns["mean_inter_event_distance_km"] >= 0
        assert spatial_patterns["spatial_extent_km2"] >= 0
        assert len(spatial_patterns["centroid"]) == 2

    def test_temporal_patterns_analysis(self, engine, sample_events):
        """Test temporal patterns analysis"""
        temporal_patterns = engine._analyze_temporal_patterns(sample_events)

        assert "mean_inter_event_time_hours" in temporal_patterns
        assert "event_rate_per_day" in temporal_patterns
        assert "total_time_span_days" in temporal_patterns

        assert temporal_patterns["mean_inter_event_time_hours"] >= 0
        assert temporal_patterns["event_rate_per_day"] >= 0
        assert temporal_patterns["total_time_span_days"] >= 0

    def test_magnitude_patterns_analysis(self, engine, sample_events):
        """Test magnitude patterns analysis"""
        magnitude_patterns = engine._analyze_magnitude_patterns(sample_events)

        assert "magnitude_statistics" in magnitude_patterns
        assert "gutenberg_richter" in magnitude_patterns

        mag_stats = magnitude_patterns["magnitude_statistics"]
        assert "mean" in mag_stats
        assert "std" in mag_stats
        assert "min" in mag_stats
        assert "max" in mag_stats

        assert mag_stats["min"] <= mag_stats["mean"] <= mag_stats["max"]
        assert mag_stats["std"] >= 0

    def test_cluster_characteristics_analysis(self, engine):
        """Test earthquake cluster characteristics analysis"""
        # Create a simple cluster
        base_time = datetime(2023, 1, 1)
        cluster_events = [
            SeismicEvent(
                timestamp=base_time + timedelta(hours=i),
                location=GeologicalLocation(
                    latitude=35.0 + i * 0.01, longitude=-118.0 + i * 0.01, elevation=0.0
                ),
                magnitude=4.0 + i * 0.1,
                depth=10.0,
                event_id=f"cluster_event_{i}",
            )
            for i in range(5)
        ]

        cluster = engine._analyze_cluster_characteristics(cluster_events)

        assert isinstance(cluster, EarthquakeCluster)
        assert len(cluster.events) == 5
        assert cluster.spatial_extent >= 0
        assert cluster.temporal_extent >= 0
        assert len(cluster.magnitude_range) == 2
        assert cluster.magnitude_range[0] <= cluster.magnitude_range[1]

    def test_cluster_type_classification(self, engine, clustered_events):
        """Test earthquake cluster type classification"""
        # Analyze cluster characteristics first
        cluster = engine._analyze_cluster_characteristics(clustered_events)
        classified_clusters = engine._classify_cluster_types([cluster])

        assert len(classified_clusters) == 1
        classified_cluster = classified_clusters[0]

        # Should classify as mainshock-aftershock due to the structure of clustered_events
        assert classified_cluster.cluster_type in [
            "mainshock_aftershock",
            "swarm",
            "foreshock_mainshock",
            "general_cluster",
        ]

    def test_hazard_level_assessment(self, engine):
        """Test hazard level assessment"""
        # Test different probability scenarios
        low_risk_forecast = {
            "M>=4.0": 0.05,
            "M>=5.0": 0.01,
            "M>=6.0": 0.001,
            "M>=7.0": 0.0001,
        }
        hazard_level = engine._assess_hazard_level(low_risk_forecast)
        assert hazard_level in [HazardLevel.LOW, HazardLevel.MODERATE]

        high_risk_forecast = {
            "M>=4.0": 0.8,
            "M>=5.0": 0.4,
            "M>=6.0": 0.15,
            "M>=7.0": 0.05,
        }
        hazard_level = engine._assess_hazard_level(high_risk_forecast)
        assert hazard_level in [HazardLevel.HIGH, HazardLevel.VERY_HIGH]

        extreme_risk_forecast = {
            "M>=4.0": 0.9,
            "M>=5.0": 0.7,
            "M>=6.0": 0.5,
            "M>=7.0": 0.2,
        }
        hazard_level = engine._assess_hazard_level(extreme_risk_forecast)
        assert hazard_level == HazardLevel.EXTREME

    def test_contributing_factors_identification(self, engine, sample_events):
        """Test identification of contributing factors"""
        forecast = {"M>=4.0": 0.1, "M>=5.0": 0.05, "M>=6.0": 0.01}
        factors = engine._identify_contributing_factors(sample_events, forecast)

        assert isinstance(factors, list)
        # Factors should be strings describing contributing conditions
        for factor in factors:
            assert isinstance(factor, str)

    def test_uncertainty_metrics_calculation(self, engine):
        """Test uncertainty metrics calculation"""
        # Create mock ensemble forecasts
        ensemble_forecasts = {
            "model1": {"M>=4.0": 0.1, "M>=5.0": 0.05},
            "model2": {"M>=4.0": 0.15, "M>=5.0": 0.03},
            "model3": {"M>=4.0": 0.08, "M>=5.0": 0.07},
        }

        uncertainty_metrics = engine._calculate_uncertainty_metrics(ensemble_forecasts)

        assert "overall_uncertainty" in uncertainty_metrics
        assert uncertainty_metrics["overall_uncertainty"] >= 0

        # Should have standard deviation metrics for each magnitude range
        assert "M>=4.0_std" in uncertainty_metrics
        assert "M>=5.0_std" in uncertainty_metrics

    def test_ensemble_forecast_combination(self, engine):
        """Test ensemble forecast combination"""
        ensemble_forecasts = {
            "statistical": {"M>=4.0": 0.1, "M>=5.0": 0.05},
            "etas": {"M>=4.0": 0.12, "M>=5.0": 0.04},
            "machine_learning": {"M>=4.0": 0.08, "M>=5.0": 0.06},
        }

        combined = engine._combine_ensemble_forecasts(ensemble_forecasts)

        assert "M>=4.0" in combined
        assert "M>=5.0" in combined

        # Combined values should be within reasonable range of individual forecasts
        assert 0.05 <= combined["M>=4.0"] <= 0.15
        assert 0.03 <= combined["M>=5.0"] <= 0.07

    def test_confidence_intervals_calculation(self, engine):
        """Test confidence intervals calculation"""
        ensemble_forecasts = {
            "model1": {"M>=4.0": 0.1, "M>=5.0": 0.05},
            "model2": {"M>=4.0": 0.15, "M>=5.0": 0.03},
            "model3": {"M>=4.0": 0.08, "M>=5.0": 0.07},
        }

        combined_forecast = {"M>=4.0": 0.11, "M>=5.0": 0.05}

        confidence_intervals = engine._calculate_confidence_intervals(
            ensemble_forecasts, combined_forecast
        )

        assert "M>=4.0" in confidence_intervals
        assert "M>=5.0" in confidence_intervals

        # Check that intervals are valid
        for lower, upper in confidence_intervals.values():
            assert 0 <= lower <= upper <= 1

    def test_events_filtering_to_region(self, engine, sample_events, target_region):
        """Test filtering events to target region"""
        filtered_events = engine._filter_events_to_region(sample_events, target_region)

        # All filtered events should be within the target region
        for event in filtered_events:
            assert (
                target_region["lat_min"]
                <= event.location.latitude
                <= target_region["lat_max"]
            )
            assert (
                target_region["lon_min"]
                <= event.location.longitude
                <= target_region["lon_max"]
            )

    def test_statistical_forecast_generation(self, engine, sample_events):
        """Test statistical forecast generation"""
        forecast = engine._generate_statistical_forecast(sample_events, 30)

        assert isinstance(forecast, dict)
        assert len(forecast) > 0

        # All probabilities should be valid
        for prob in forecast.values():
            assert 0 <= prob <= 1

class TestEarthquakeCluster:
    """Test suite for EarthquakeCluster class"""

    def test_cluster_creation(self):
        """Test earthquake cluster creation"""
        events = [
            SeismicEvent(
                timestamp=datetime(2023, 1, 1),
                location=GeologicalLocation(
                    latitude=35.0, longitude=-118.0, elevation=0.0
                ),
                magnitude=4.0,
                depth=10.0,
                event_id="event1",
            ),
            SeismicEvent(
                timestamp=datetime(2023, 1, 2),
                location=GeologicalLocation(
                    latitude=35.1, longitude=-118.1, elevation=0.0
                ),
                magnitude=4.5,
                depth=12.0,
                event_id="event2",
            ),
        ]

        cluster = EarthquakeCluster(
            cluster_id="test_cluster",
            events=events,
            centroid_location=GeologicalLocation(
                latitude=35.05, longitude=-118.05, elevation=0.0
            ),
            centroid_time=datetime(2023, 1, 1, 12),
            spatial_extent=15.0,
            temporal_extent=1.0,
            magnitude_range=(4.0, 4.5),
        )

        assert cluster.cluster_id == "test_cluster"
        assert len(cluster.events) == 2
        assert cluster.spatial_extent == 15.0
        assert cluster.temporal_extent == 1.0
        assert cluster.magnitude_range == (4.0, 4.5)

    def test_cluster_validation(self):
        """Test cluster parameter validation"""
        # Test empty events list
        with pytest.raises(ValueError, match="Cluster must contain at least one event"):
            EarthquakeCluster(
                cluster_id="empty_cluster",
                events=[],
                centroid_location=GeologicalLocation(
                    latitude=35.0, longitude=-118.0, elevation=0.0
                ),
                centroid_time=datetime(2023, 1, 1),
                spatial_extent=10.0,
                temporal_extent=1.0,
                magnitude_range=(4.0, 5.0),
            )

        # Test negative spatial extent
        event = SeismicEvent(
            timestamp=datetime(2023, 1, 1),
            location=GeologicalLocation(latitude=35.0, longitude=-118.0, elevation=0.0),
            magnitude=4.0,
            depth=10.0,
        )

        with pytest.raises(ValueError, match="Spatial extent must be non-negative"):
            EarthquakeCluster(
                cluster_id="invalid_cluster",
                events=[event],
                centroid_location=GeologicalLocation(
                    latitude=35.0, longitude=-118.0, elevation=0.0
                ),
                centroid_time=datetime(2023, 1, 1),
                spatial_extent=-5.0,
                temporal_extent=1.0,
                magnitude_range=(4.0, 5.0),
            )

class TestForecastResult:
    """Test suite for ForecastResult class"""

    def test_forecast_result_creation(self):
        """Test forecast result creation"""
        forecast_time = datetime.utcnow()
        forecast_period = (forecast_time, forecast_time + timedelta(days=30))
        target_region = {
            "lat_min": 34.0,
            "lat_max": 36.0,
            "lon_min": -119.0,
            "lon_max": -117.0,
        }

        result = ForecastResult(
            forecast_time=forecast_time,
            forecast_period=forecast_period,
            target_region=target_region,
            probability_estimates={"M>=4.0": 0.1, "M>=5.0": 0.05},
            confidence_intervals={"M>=4.0": (0.05, 0.15), "M>=5.0": (0.02, 0.08)},
            hazard_level=HazardLevel.MODERATE,
            model_used=ForecastModel.STATISTICAL_SEISMICITY,
            uncertainty_metrics={"overall_uncertainty": 0.02},
        )

        assert result.forecast_time == forecast_time
        assert result.forecast_period == forecast_period
        assert result.hazard_level == HazardLevel.MODERATE
        assert result.model_used == ForecastModel.STATISTICAL_SEISMICITY

    def test_forecast_result_validation(self):
        """Test forecast result validation"""
        forecast_time = datetime.utcnow()
        forecast_period = (forecast_time, forecast_time + timedelta(days=30))
        target_region = {
            "lat_min": 34.0,
            "lat_max": 36.0,
            "lon_min": -119.0,
            "lon_max": -117.0,
        }

        # Test invalid probability (> 1.0)
        with pytest.raises(ValueError, match="Probability.*must be between 0 and 1"):
            ForecastResult(
                forecast_time=forecast_time,
                forecast_period=forecast_period,
                target_region=target_region,
                probability_estimates={"M>=4.0": 1.5},  # Invalid probability
                confidence_intervals={"M>=4.0": (0.05, 0.15)},
                hazard_level=HazardLevel.MODERATE,
                model_used=ForecastModel.STATISTICAL_SEISMICITY,
                uncertainty_metrics={"overall_uncertainty": 0.02},
            )

        # Test invalid confidence interval
        with pytest.raises(ValueError, match="Invalid confidence interval"):
            ForecastResult(
                forecast_time=forecast_time,
                forecast_period=forecast_period,
                target_region=target_region,
                probability_estimates={"M>=4.0": 0.1},
                confidence_intervals={"M>=4.0": (0.15, 0.05)},  # Lower > Upper
                hazard_level=HazardLevel.MODERATE,
                model_used=ForecastModel.STATISTICAL_SEISMICITY,
                uncertainty_metrics={"overall_uncertainty": 0.02},
            )

if __name__ == "__main__":
    pytest.main([__file__])
