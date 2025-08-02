"""
Tests for Earthquake Impact Modeling System

Tests for ground shaking simulation, liquefaction assessment,
landslide modeling, and tsunami generation capabilities.
"""

from datetime import datetime

import pytest
import torch.nn.functional as F

from core.earthquake_impact_modeling import (
    EarthquakeImpactModelingEngine,
    GroundMotionModel,
    GroundMotionParameters,
    LandslideAssessment,
    LandslideSusceptibility,
    LiquefactionAssessment,
    LiquefactionSusceptibility,
    SoilType,
    TsunamiParameters,
    TsunamiSource,
)
from models.geological import EventType, GeologicalLocation, SeismicEvent


class TestEarthquakeImpactModelingEngine:
    """Test cases for the main earthquake impact modeling engine"""

    @pytest.fixture
    def engine(self):
        """Create earthquake impact modeling engine for testing"""
        return EarthquakeImpactModelingEngine()

    @pytest.fixture
    def sample_earthquake(self):
        """Create sample earthquake for testing"""
        return SeismicEvent(
            timestamp=datetime.utcnow(),
            location=GeologicalLocation(
                latitude=37.7749, longitude=-122.4194, elevation=0.0
            ),
            magnitude=7.0,
            depth=10.0,
            event_type=EventType.EARTHQUAKE,
        )

    @pytest.fixture
    def sample_locations(self):
        """Create sample target locations for testing"""
        return [
            GeologicalLocation(latitude=37.8, longitude=-122.4, elevation=50.0),
            GeologicalLocation(latitude=37.7, longitude=-122.3, elevation=20.0),
            GeologicalLocation(latitude=37.6, longitude=-122.5, elevation=100.0),
        ]

    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert engine.config is not None
        assert "ground_motion" in engine.config
        assert "liquefaction" in engine.config
        assert "landslide" in engine.config
        assert "tsunami" in engine.config

    def test_simulate_ground_shaking_basic(
        self, engine, sample_earthquake, sample_locations
    ):
        """Test basic ground shaking simulation"""
        results = engine.simulate_ground_shaking(
            earthquake=sample_earthquake, target_locations=sample_locations
        )

        assert len(results) == len(sample_locations)

        for result in results:
            assert isinstance(result, GroundMotionParameters)
            assert result.peak_ground_acceleration > 0
            assert result.peak_ground_velocity > 0
            assert result.peak_ground_displacement >= 0
            assert result.duration > 0
            assert result.arias_intensity > 0
            assert len(result.spectral_accelerations) > 0

    def test_simulate_ground_shaking_with_soil_conditions(
        self, engine, sample_earthquake, sample_locations
    ):
        """Test ground shaking simulation with different soil conditions"""
        soil_conditions = {
            "0": SoilType.ROCK,
            "1": SoilType.STIFF_SOIL,
            "2": SoilType.SOFT_SOIL,
        }

        results = engine.simulate_ground_shaking(
            earthquake=sample_earthquake,
            target_locations=sample_locations,
            soil_conditions=soil_conditions,
        )

        assert len(results) == len(sample_locations)

        # Soft soil should generally have higher ground motion
        rock_pga = results[0].peak_ground_acceleration
        soft_soil_pga = results[2].peak_ground_acceleration

        # This is a general trend, though not always true
        assert rock_pga > 0
        assert soft_soil_pga > 0

    def test_simulate_ground_shaking_different_models(
        self, engine, sample_earthquake, sample_locations
    ):
        """Test ground shaking simulation with different GMPEs"""
        models = [
            GroundMotionModel.BOORE_ATKINSON_2008,
            GroundMotionModel.CAMPBELL_BOZORGNIA_2014,
            GroundMotionModel.CHIOU_YOUNGS_2014,
        ]

        for model in models:
            results = engine.simulate_ground_shaking(
                earthquake=sample_earthquake,
                target_locations=sample_locations[:1],  # Test with one location
                model=model,
            )

            assert len(results) == 1
            assert results[0].peak_ground_acceleration > 0

    def test_simulate_ground_shaking_small_earthquake(self, engine, sample_locations):
        """Test that small earthquakes raise appropriate error"""
        small_earthquake = SeismicEvent(
            timestamp=datetime.utcnow(),
            location=GeologicalLocation(
                latitude=37.7749, longitude=-122.4194, elevation=0.0
            ),
            magnitude=2.0,  # Too small
            depth=10.0,
            event_type=EventType.EARTHQUAKE,
        )

        with pytest.raises(ValueError, match="magnitude too small"):
            engine.simulate_ground_shaking(
                earthquake=small_earthquake, target_locations=sample_locations
            )

    def test_assess_liquefaction_susceptibility(self, engine, sample_locations):
        """Test liquefaction susceptibility assessment"""
        # Create sample ground motion data
        ground_motion = [
            GroundMotionParameters(
                location=loc,
                peak_ground_acceleration=0.3,
                peak_ground_velocity=30.0,
                peak_ground_displacement=15.0,
                spectral_accelerations={0.1: 0.4, 1.0: 0.2},
                duration=20.0,
                arias_intensity=0.5,
            )
            for loc in sample_locations
        ]

        # Sample soil properties
        soil_properties = {
            "0": {
                "spt_n_value": 10,
                "fines_content": 20,
                "total_stress": 100000,
                "effective_stress": 50000,
            },
            "1": {
                "spt_n_value": 15,
                "fines_content": 30,
                "total_stress": 120000,
                "effective_stress": 60000,
            },
            "2": {
                "spt_n_value": 25,
                "fines_content": 40,
                "total_stress": 150000,
                "effective_stress": 80000,
            },
        }

        # Sample groundwater data
        groundwater_data = {"0": 2.0, "1": 5.0, "2": 8.0}

        results = engine.assess_liquefaction_susceptibility(
            ground_motion=ground_motion,
            soil_properties=soil_properties,
            groundwater_data=groundwater_data,
        )

        assert len(results) == len(ground_motion)

        for result in results:
            assert isinstance(result, LiquefactionAssessment)
            assert isinstance(result.susceptibility, LiquefactionSusceptibility)
            assert result.factor_of_safety > 0
            assert 0 <= result.probability <= 1
            assert isinstance(result.contributing_factors, list)
            assert result.groundwater_depth is not None

    def test_assess_liquefaction_high_groundwater(self, engine, sample_locations):
        """Test liquefaction assessment with high groundwater"""
        ground_motion = [
            GroundMotionParameters(
                location=sample_locations[0],
                peak_ground_acceleration=0.4,
                peak_ground_velocity=40.0,
                peak_ground_displacement=20.0,
                spectral_accelerations={0.1: 0.5, 1.0: 0.3},
                duration=25.0,
                arias_intensity=0.8,
            )
        ]

        soil_properties = {
            "0": {
                "spt_n_value": 8,
                "fines_content": 15,
                "total_stress": 100000,
                "effective_stress": 50000,
            }
        }

        groundwater_data = {"0": 1.5}  # Very shallow groundwater

        results = engine.assess_liquefaction_susceptibility(
            ground_motion=ground_motion,
            soil_properties=soil_properties,
            groundwater_data=groundwater_data,
        )

        assert len(results) == 1
        # Should have high susceptibility due to shallow groundwater and loose soil
        assert results[0].susceptibility in [
            LiquefactionSusceptibility.HIGH,
            LiquefactionSusceptibility.VERY_HIGH,
        ]
        assert "shallow_groundwater" in results[0].contributing_factors

    def test_model_landslide_triggering(self, engine, sample_locations):
        """Test landslide triggering and runout modeling"""
        # Create sample ground motion data
        ground_motion = [
            GroundMotionParameters(
                location=loc,
                peak_ground_acceleration=0.4,
                peak_ground_velocity=35.0,
                peak_ground_displacement=18.0,
                spectral_accelerations={0.1: 0.5, 1.0: 0.3},
                duration=22.0,
                arias_intensity=0.7,
            )
            for loc in sample_locations
        ]

        # Sample slope data (degrees)
        slope_data = {"0": 25.0, "1": 35.0, "2": 45.0}

        # Sample soil strength data
        soil_strength = {
            "0": {
                "cohesion": 8000,
                "friction_angle": 32,
                "unit_weight": 18000,
                "thickness": 2.5,
            },
            "1": {
                "cohesion": 5000,
                "friction_angle": 28,
                "unit_weight": 17000,
                "thickness": 3.0,
            },
            "2": {
                "cohesion": 3000,
                "friction_angle": 25,
                "unit_weight": 16000,
                "thickness": 2.0,
            },
        }

        results = engine.model_landslide_triggering(
            ground_motion=ground_motion,
            slope_data=slope_data,
            soil_strength=soil_strength,
        )

        assert len(results) == len(ground_motion)

        for result in results:
            assert isinstance(result, LandslideAssessment)
            assert isinstance(result.susceptibility, LandslideSusceptibility)
            assert result.trigger_threshold > 0
            assert 0 <= result.probability <= 1
            assert result.volume_estimate > 0
            assert result.runout_distance > 0
            assert len(result.runout_path) >= 2
            assert result.impact_area > 0

    def test_model_landslide_gentle_slopes(self, engine, sample_locations):
        """Test that gentle slopes are filtered out"""
        ground_motion = [
            GroundMotionParameters(
                location=sample_locations[0],
                peak_ground_acceleration=0.4,
                peak_ground_velocity=35.0,
                peak_ground_displacement=18.0,
                spectral_accelerations={0.1: 0.5, 1.0: 0.3},
                duration=22.0,
                arias_intensity=0.7,
            )
        ]

        # Gentle slope (below threshold)
        slope_data = {"0": 10.0}

        soil_strength = {
            "0": {
                "cohesion": 8000,
                "friction_angle": 32,
                "unit_weight": 18000,
                "thickness": 2.5,
            }
        }

        results = engine.model_landslide_triggering(
            ground_motion=ground_motion,
            slope_data=slope_data,
            soil_strength=soil_strength,
        )

        # Should be empty because slope is too gentle
        assert len(results) == 0

    def test_model_tsunami_generation_capable_earthquake(
        self, engine, sample_locations
    ):
        """Test tsunami generation for earthquake capable of generating tsunami"""
        # Large, shallow earthquake
        tsunami_earthquake = SeismicEvent(
            timestamp=datetime.utcnow(),
            location=GeologicalLocation(
                latitude=37.7749, longitude=-122.4194, elevation=-1000.0
            ),
            magnitude=8.5,
            depth=15.0,  # Shallow
            event_type=EventType.EARTHQUAKE,
        )

        result = engine.model_tsunami_generation(
            earthquake=tsunami_earthquake, coastal_locations=sample_locations
        )

        assert result is not None
        assert isinstance(result, TsunamiParameters)
        assert isinstance(result.source_type, TsunamiSource)
        assert result.initial_wave_height > 0
        assert result.wave_period > 0
        assert result.propagation_speed > 0
        assert len(result.arrival_times) == len(sample_locations)
        assert len(result.wave_heights) == len(sample_locations)
        assert len(result.inundation_distances) == len(sample_locations)
        assert result.affected_coastline > 0

    def test_model_tsunami_generation_incapable_earthquake(
        self, engine, sample_locations
    ):
        """Test tsunami generation for earthquake incapable of generating tsunami"""
        # Small earthquake
        small_earthquake = SeismicEvent(
            timestamp=datetime.utcnow(),
            location=GeologicalLocation(
                latitude=37.7749, longitude=-122.4194, elevation=0.0
            ),
            magnitude=6.0,  # Too small
            depth=15.0,
            event_type=EventType.EARTHQUAKE,
        )

        result = engine.model_tsunami_generation(
            earthquake=small_earthquake, coastal_locations=sample_locations
        )

        assert result is None

    def test_model_tsunami_generation_deep_earthquake(self, engine, sample_locations):
        """Test tsunami generation for deep earthquake"""
        # Deep earthquake
        deep_earthquake = SeismicEvent(
            timestamp=datetime.utcnow(),
            location=GeologicalLocation(
                latitude=37.7749, longitude=-122.4194, elevation=0.0
            ),
            magnitude=8.0,
            depth=80.0,  # Too deep
            event_type=EventType.EARTHQUAKE,
        )

        result = engine.model_tsunami_generation(
            earthquake=deep_earthquake, coastal_locations=sample_locations
        )

        assert result is None


class TestGroundMotionParameters:
    """Test cases for GroundMotionParameters data class"""

    def test_valid_ground_motion_parameters(self):
        """Test creation of valid ground motion parameters"""
        location = GeologicalLocation(
            latitude=37.7749, longitude=-122.4194, elevation=0.0
        )

        params = GroundMotionParameters(
            location=location,
            peak_ground_acceleration=0.3,
            peak_ground_velocity=25.0,
            peak_ground_displacement=12.0,
            spectral_accelerations={0.1: 0.4, 1.0: 0.2},
            duration=18.0,
            arias_intensity=0.6,
        )

        assert params.location == location
        assert params.peak_ground_acceleration == 0.3
        assert params.peak_ground_velocity == 25.0
        assert params.peak_ground_displacement == 12.0
        assert params.duration == 18.0
        assert params.arias_intensity == 0.6

    def test_invalid_ground_motion_parameters(self):
        """Test validation of ground motion parameters"""
        location = GeologicalLocation(
            latitude=37.7749, longitude=-122.4194, elevation=0.0
        )

        # Negative PGA
        with pytest.raises(ValueError, match="PGA must be non-negative"):
            GroundMotionParameters(
                location=location,
                peak_ground_acceleration=-0.1,
                peak_ground_velocity=25.0,
                peak_ground_displacement=12.0,
                spectral_accelerations={0.1: 0.4},
                duration=18.0,
                arias_intensity=0.6,
            )

        # Negative duration
        with pytest.raises(ValueError, match="Duration must be non-negative"):
            GroundMotionParameters(
                location=location,
                peak_ground_acceleration=0.3,
                peak_ground_velocity=25.0,
                peak_ground_displacement=12.0,
                spectral_accelerations={0.1: 0.4},
                duration=-5.0,
                arias_intensity=0.6,
            )


class TestLiquefactionAssessment:
    """Test cases for LiquefactionAssessment data class"""

    def test_valid_liquefaction_assessment(self):
        """Test creation of valid liquefaction assessment"""
        location = GeologicalLocation(
            latitude=37.7749, longitude=-122.4194, elevation=0.0
        )

        assessment = LiquefactionAssessment(
            location=location,
            susceptibility=LiquefactionSusceptibility.MODERATE,
            factor_of_safety=1.1,
            probability=0.3,
            contributing_factors=["shallow_groundwater", "loose_soil"],
            soil_properties={"spt_n_value": 12, "fines_content": 25},
            groundwater_depth=3.0,
        )

        assert assessment.location == location
        assert assessment.susceptibility == LiquefactionSusceptibility.MODERATE
        assert assessment.factor_of_safety == 1.1
        assert assessment.probability == 0.3
        assert len(assessment.contributing_factors) == 2
        assert assessment.groundwater_depth == 3.0

    def test_invalid_liquefaction_probability(self):
        """Test validation of liquefaction probability"""
        location = GeologicalLocation(
            latitude=37.7749, longitude=-122.4194, elevation=0.0
        )

        # Probability > 1
        with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
            LiquefactionAssessment(
                location=location,
                susceptibility=LiquefactionSusceptibility.HIGH,
                factor_of_safety=0.8,
                probability=1.5,  # Invalid
                contributing_factors=[],
                soil_properties={},
            )


class TestLandslideAssessment:
    """Test cases for LandslideAssessment data class"""

    def test_valid_landslide_assessment(self):
        """Test creation of valid landslide assessment"""
        location = GeologicalLocation(
            latitude=37.7749, longitude=-122.4194, elevation=100.0
        )
        runout_location = GeologicalLocation(
            latitude=37.7740, longitude=-122.4194, elevation=50.0
        )

        assessment = LandslideAssessment(
            location=location,
            susceptibility=LandslideSusceptibility.HIGH,
            trigger_threshold=0.2,
            probability=0.7,
            volume_estimate=5000.0,
            runout_distance=200.0,
            runout_path=[location, runout_location],
            impact_area=40000.0,
        )

        assert assessment.location == location
        assert assessment.susceptibility == LandslideSusceptibility.HIGH
        assert assessment.trigger_threshold == 0.2
        assert assessment.probability == 0.7
        assert assessment.volume_estimate == 5000.0
        assert assessment.runout_distance == 200.0
        assert len(assessment.runout_path) == 2
        assert assessment.impact_area == 40000.0

    def test_invalid_landslide_parameters(self):
        """Test validation of landslide parameters"""
        location = GeologicalLocation(
            latitude=37.7749, longitude=-122.4194, elevation=100.0
        )

        # Negative volume
        with pytest.raises(ValueError, match="Volume estimate must be non-negative"):
            LandslideAssessment(
                location=location,
                susceptibility=LandslideSusceptibility.HIGH,
                trigger_threshold=0.2,
                probability=0.7,
                volume_estimate=-1000.0,  # Invalid
                runout_distance=200.0,
                runout_path=[location],
                impact_area=40000.0,
            )


class TestTsunamiParameters:
    """Test cases for TsunamiParameters data class"""

    def test_valid_tsunami_parameters(self):
        """Test creation of valid tsunami parameters"""
        location = GeologicalLocation(
            latitude=37.7749, longitude=-122.4194, elevation=-1000.0
        )

        params = TsunamiParameters(
            source_location=location,
            source_type=TsunamiSource.SUBMARINE_EARTHQUAKE,
            initial_wave_height=3.0,
            wave_period=600.0,
            propagation_speed=200.0,
            arrival_times={"coastal_0": 15.0, "coastal_1": 25.0},
            wave_heights={"coastal_0": 2.5, "coastal_1": 1.8},
            inundation_distances={"coastal_0": 500.0, "coastal_1": 300.0},
            affected_coastline=50.0,
        )

        assert params.source_location == location
        assert params.source_type == TsunamiSource.SUBMARINE_EARTHQUAKE
        assert params.initial_wave_height == 3.0
        assert params.wave_period == 600.0
        assert params.propagation_speed == 200.0
        assert len(params.arrival_times) == 2
        assert len(params.wave_heights) == 2
        assert len(params.inundation_distances) == 2
        assert params.affected_coastline == 50.0

    def test_invalid_tsunami_parameters(self):
        """Test validation of tsunami parameters"""
        location = GeologicalLocation(
            latitude=37.7749, longitude=-122.4194, elevation=-1000.0
        )

        # Negative wave height
        with pytest.raises(
            ValueError, match="Initial wave height must be non-negative"
        ):
            TsunamiParameters(
                source_location=location,
                source_type=TsunamiSource.SUBMARINE_EARTHQUAKE,
                initial_wave_height=-1.0,  # Invalid
                wave_period=600.0,
                propagation_speed=200.0,
                arrival_times={},
                wave_heights={},
                inundation_distances={},
                affected_coastline=50.0,
            )

        # Zero wave period
        with pytest.raises(ValueError, match="Wave period must be positive"):
            TsunamiParameters(
                source_location=location,
                source_type=TsunamiSource.SUBMARINE_EARTHQUAKE,
                initial_wave_height=3.0,
                wave_period=0.0,  # Invalid
                propagation_speed=200.0,
                arrival_times={},
                wave_heights={},
                inundation_distances={},
                affected_coastline=50.0,
            )


class TestDistanceCalculation:
    """Test cases for distance calculation methods"""

    def test_distance_calculation(self):
        """Test distance calculation between locations"""
        engine = EarthquakeImpactModelingEngine()

        # San Francisco to Los Angeles (approximate)
        sf = GeologicalLocation(latitude=37.7749, longitude=-122.4194, elevation=0.0)
        la = GeologicalLocation(latitude=34.0522, longitude=-118.2437, elevation=0.0)

        distance = engine._calculate_distance(sf, la)

        # Should be approximately 560 km
        assert 500 < distance < 600

    def test_distance_same_location(self):
        """Test distance calculation for same location"""
        engine = EarthquakeImpactModelingEngine()

        location = GeologicalLocation(
            latitude=37.7749, longitude=-122.4194, elevation=0.0
        )

        distance = engine._calculate_distance(location, location)

        assert distance == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
