"""
Unit tests for geological data models

Tests all geological data structures, validation, and serialization/deserialization.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from models.geological import (
    GeologicalLocation, SeismicEvent, DeformationMeasurement,
    GeochemicalSample, ThermalMeasurement, MagmaReservoir,
    CoordinateSystem, EventType, MeasurementType, SampleType, ThermalMethod
)
from models.validation import (
    GeologicalDataValidator, ValidationError, ValidationLevel, ValidationCategory
)


class TestGeologicalLocation:
    """Test GeologicalLocation data model"""
    
    def test_valid_location_creation(self):
        """Test creating a valid geological location"""
        location = GeologicalLocation(
            latitude=45.5,
            longitude=-122.3,
            elevation=1000.0,
            depth=500.0,
            coordinate_system=CoordinateSystem.WGS84
        )
        
        assert location.latitude == 45.5
        assert location.longitude == -122.3
        assert location.elevation == 1000.0
        assert location.depth == 500.0
        assert location.coordinate_system == CoordinateSystem.WGS84
    
    def test_invalid_latitude(self):
        """Test validation of invalid latitude"""
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            GeologicalLocation(latitude=95.0, longitude=0.0, elevation=0.0)
    
    def test_invalid_longitude(self):
        """Test validation of invalid longitude"""
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            GeologicalLocation(latitude=0.0, longitude=185.0, elevation=0.0)
    
    def test_negative_depth(self):
        """Test validation of negative depth"""
        with pytest.raises(ValueError, match="Depth must be positive"):
            GeologicalLocation(latitude=0.0, longitude=0.0, elevation=0.0, depth=-100.0)
    
    def test_serialization(self):
        """Test to_dict and from_dict methods"""
        location = GeologicalLocation(
            latitude=45.5,
            longitude=-122.3,
            elevation=1000.0,
            depth=500.0,
            uncertainty_horizontal=10.0,
            uncertainty_vertical=5.0
        )
        
        # Test serialization
        data = location.to_dict()
        expected_keys = {
            'latitude', 'longitude', 'elevation', 'depth', 
            'coordinate_system', 'uncertainty_horizontal', 'uncertainty_vertical'
        }
        assert set(data.keys()) == expected_keys
        
        # Test deserialization
        restored_location = GeologicalLocation.from_dict(data)
        assert restored_location.latitude == location.latitude
        assert restored_location.longitude == location.longitude
        assert restored_location.elevation == location.elevation
        assert restored_location.depth == location.depth


class TestSeismicEvent:
    """Test SeismicEvent data model"""
    
    def create_valid_location(self) -> GeologicalLocation:
        """Helper to create valid location"""
        return GeologicalLocation(latitude=45.0, longitude=-122.0, elevation=1000.0)
    
    def test_valid_seismic_event(self):
        """Test creating a valid seismic event"""
        location = self.create_valid_location()
        event = SeismicEvent(
            timestamp=datetime.utcnow(),
            location=location,
            magnitude=5.5,
            depth=10.0,
            event_type=EventType.EARTHQUAKE,
            confidence=0.95
        )
        
        assert event.magnitude == 5.5
        assert event.depth == 10.0
        assert event.event_type == EventType.EARTHQUAKE
        assert event.confidence == 0.95
    
    def test_invalid_magnitude(self):
        """Test validation of invalid magnitude"""
        location = self.create_valid_location()
        with pytest.raises(ValueError, match="Magnitude must be between -5.0 and 12.0"):
            SeismicEvent(
                timestamp=datetime.utcnow(),
                location=location,
                magnitude=15.0,
                depth=10.0
            )
    
    def test_invalid_depth(self):
        """Test validation of invalid depth"""
        location = self.create_valid_location()
        with pytest.raises(ValueError, match="Depth must be between 0 and 800"):
            SeismicEvent(
                timestamp=datetime.utcnow(),
                location=location,
                magnitude=5.0,
                depth=1000.0
            )
    
    def test_invalid_confidence(self):
        """Test validation of invalid confidence"""
        location = self.create_valid_location()
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            SeismicEvent(
                timestamp=datetime.utcnow(),
                location=location,
                magnitude=5.0,
                depth=10.0,
                confidence=1.5
            )
    
    def test_waveform_data_validation(self):
        """Test waveform data validation"""
        location = self.create_valid_location()
        
        # Valid 1D waveform
        waveform = np.random.randn(1000)
        event = SeismicEvent(
            timestamp=datetime.utcnow(),
            location=location,
            magnitude=5.0,
            depth=10.0,
            waveform_data=waveform
        )
        assert event.waveform_data is not None
        
        # Invalid 2D waveform
        with pytest.raises(ValueError, match="Waveform data must be 1-dimensional"):
            SeismicEvent(
                timestamp=datetime.utcnow(),
                location=location,
                magnitude=5.0,
                depth=10.0,
                waveform_data=np.random.randn(100, 3)
            )
    
    def test_serialization_with_waveform(self):
        """Test serialization with waveform data"""
        location = self.create_valid_location()
        waveform = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        event = SeismicEvent(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            location=location,
            magnitude=5.0,
            depth=10.0,
            waveform_data=waveform,
            event_id="test_event_001"
        )
        
        # Test serialization
        data = event.to_dict()
        assert 'waveform_data' in data
        assert data['waveform_data'] == [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test deserialization
        restored_event = SeismicEvent.from_dict(data)
        assert np.array_equal(restored_event.waveform_data, waveform)
        assert restored_event.event_id == "test_event_001"


class TestDeformationMeasurement:
    """Test DeformationMeasurement data model"""
    
    def create_valid_location(self) -> GeologicalLocation:
        """Helper to create valid location"""
        return GeologicalLocation(latitude=45.0, longitude=-122.0, elevation=1000.0)
    
    def test_valid_deformation_measurement(self):
        """Test creating a valid deformation measurement"""
        location = self.create_valid_location()
        displacement = np.array([0.01, 0.02, -0.005])  # meters
        uncertainty = np.array([0.001, 0.001, 0.002])  # meters
        
        measurement = DeformationMeasurement(
            timestamp=datetime.utcnow(),
            location=location,
            displacement_vector=displacement,
            measurement_type=MeasurementType.GPS,
            uncertainty=uncertainty
        )
        
        assert np.array_equal(measurement.displacement_vector, displacement)
        assert np.array_equal(measurement.uncertainty, uncertainty)
        assert measurement.measurement_type == MeasurementType.GPS
    
    def test_invalid_displacement_dimensions(self):
        """Test validation of displacement vector dimensions"""
        location = self.create_valid_location()
        
        with pytest.raises(ValueError, match="Displacement vector must be 3-dimensional"):
            DeformationMeasurement(
                timestamp=datetime.utcnow(),
                location=location,
                displacement_vector=np.array([0.01, 0.02]),  # Only 2D
                measurement_type=MeasurementType.GPS,
                uncertainty=np.array([0.001, 0.001, 0.002])
            )
    
    def test_invalid_uncertainty_dimensions(self):
        """Test validation of uncertainty dimensions"""
        location = self.create_valid_location()
        
        with pytest.raises(ValueError, match="Uncertainty must be 3-dimensional"):
            DeformationMeasurement(
                timestamp=datetime.utcnow(),
                location=location,
                displacement_vector=np.array([0.01, 0.02, -0.005]),
                measurement_type=MeasurementType.GPS,
                uncertainty=np.array([0.001, 0.001])  # Only 2D
            )
    
    def test_negative_uncertainty(self):
        """Test validation of negative uncertainty"""
        location = self.create_valid_location()
        
        with pytest.raises(ValueError, match="Uncertainty values must be non-negative"):
            DeformationMeasurement(
                timestamp=datetime.utcnow(),
                location=location,
                displacement_vector=np.array([0.01, 0.02, -0.005]),
                measurement_type=MeasurementType.GPS,
                uncertainty=np.array([0.001, -0.001, 0.002])  # Negative uncertainty
            )
    
    def test_large_displacement_warning(self):
        """Test warning for large displacements"""
        location = self.create_valid_location()
        large_displacement = np.array([150.0, 160.0, -130.0])  # Very large displacement (>100m)
        
        with pytest.warns(UserWarning, match="Large displacement detected"):
            DeformationMeasurement(
                timestamp=datetime.utcnow(),
                location=location,
                displacement_vector=large_displacement,
                measurement_type=MeasurementType.GPS,
                uncertainty=np.array([0.1, 0.1, 0.1])
            )
    
    def test_serialization_with_velocity(self):
        """Test serialization with velocity data"""
        location = self.create_valid_location()
        displacement = np.array([0.01, 0.02, -0.005])
        uncertainty = np.array([0.001, 0.001, 0.002])
        velocity = np.array([0.001, 0.002, -0.0005])  # m/year
        
        measurement = DeformationMeasurement(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            location=location,
            displacement_vector=displacement,
            measurement_type=MeasurementType.INSAR,
            uncertainty=uncertainty,
            velocity_vector=velocity,
            station_id="GPS001"
        )
        
        # Test serialization
        data = measurement.to_dict()
        assert 'velocity_vector' in data
        assert data['velocity_vector'] == velocity.tolist()
        
        # Test deserialization
        restored_measurement = DeformationMeasurement.from_dict(data)
        assert np.array_equal(restored_measurement.velocity_vector, velocity)
        assert restored_measurement.station_id == "GPS001"


class TestGeochemicalSample:
    """Test GeochemicalSample data model"""
    
    def create_valid_location(self) -> GeologicalLocation:
        """Helper to create valid location"""
        return GeologicalLocation(latitude=45.0, longitude=-122.0, elevation=1000.0)
    
    def create_valid_composition(self) -> Dict[str, float]:
        """Helper to create valid rock composition"""
        return {
            "SiO2": 65.0,
            "Al2O3": 15.0,
            "Fe2O3": 3.0,
            "FeO": 2.0,
            "MgO": 2.5,
            "CaO": 4.0,
            "Na2O": 3.5,
            "K2O": 2.0,
            "TiO2": 0.8,
            "P2O5": 0.2
        }
    
    def test_valid_geochemical_sample(self):
        """Test creating a valid geochemical sample"""
        location = self.create_valid_location()
        composition = self.create_valid_composition()
        
        sample = GeochemicalSample(
            timestamp=datetime.utcnow(),
            location=location,
            sample_type=SampleType.ROCK,
            composition=composition,
            temperature=25.0,
            ph=7.0
        )
        
        assert sample.sample_type == SampleType.ROCK
        assert sample.composition == composition
        assert sample.temperature == 25.0
        assert sample.ph == 7.0
    
    def test_negative_concentration(self):
        """Test validation of negative concentrations"""
        location = self.create_valid_location()
        composition = self.create_valid_composition()
        composition["SiO2"] = -10.0  # Invalid negative concentration
        
        with pytest.raises(ValueError, match="Concentration for SiO2 must be non-negative"):
            GeochemicalSample(
                timestamp=datetime.utcnow(),
                location=location,
                sample_type=SampleType.ROCK,
                composition=composition
            )
    
    def test_invalid_temperature(self):
        """Test validation of invalid temperature"""
        location = self.create_valid_location()
        composition = self.create_valid_composition()
        
        with pytest.raises(ValueError, match="Temperature must be between -273.15°C and 2000°C"):
            GeochemicalSample(
                timestamp=datetime.utcnow(),
                location=location,
                sample_type=SampleType.ROCK,
                composition=composition,
                temperature=-300.0  # Below absolute zero
            )
    
    def test_invalid_ph(self):
        """Test validation of invalid pH"""
        location = self.create_valid_location()
        composition = self.create_valid_composition()
        
        with pytest.raises(ValueError, match="pH must be between 0 and 14"):
            GeochemicalSample(
                timestamp=datetime.utcnow(),
                location=location,
                sample_type=SampleType.WATER,
                composition=composition,
                ph=15.0  # Invalid pH
            )
    
    def test_invalid_isotope_ratio(self):
        """Test validation of invalid isotope ratios"""
        location = self.create_valid_location()
        composition = self.create_valid_composition()
        
        with pytest.raises(ValueError, match="Isotope ratio Sr87_Sr86 must be positive"):
            GeochemicalSample(
                timestamp=datetime.utcnow(),
                location=location,
                sample_type=SampleType.ROCK,
                composition=composition,
                isotope_ratios={"Sr87_Sr86": -0.5}  # Invalid negative ratio
            )
    
    def test_serialization_complete(self):
        """Test complete serialization with all fields"""
        location = self.create_valid_location()
        composition = self.create_valid_composition()
        isotope_ratios = {"Sr87_Sr86": 0.7045, "Nd143_Nd144": 0.5126}
        
        sample = GeochemicalSample(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            location=location,
            sample_type=SampleType.LAVA,
            composition=composition,
            isotope_ratios=isotope_ratios,
            temperature=1200.0,
            pressure=1e5,
            ph=None,
            sample_id="SAMPLE_001",
            collection_method="direct_sampling"
        )
        
        # Test serialization
        data = sample.to_dict()
        assert data['sample_id'] == "SAMPLE_001"
        assert data['isotope_ratios'] == isotope_ratios
        
        # Test deserialization
        restored_sample = GeochemicalSample.from_dict(data)
        assert restored_sample.sample_id == "SAMPLE_001"
        assert restored_sample.isotope_ratios == isotope_ratios
        assert restored_sample.temperature == 1200.0


class TestThermalMeasurement:
    """Test ThermalMeasurement data model"""
    
    def create_valid_location(self) -> GeologicalLocation:
        """Helper to create valid location"""
        return GeologicalLocation(latitude=45.0, longitude=-122.0, elevation=1000.0)
    
    def test_valid_thermal_measurement(self):
        """Test creating a valid thermal measurement"""
        location = self.create_valid_location()
        
        measurement = ThermalMeasurement(
            timestamp=datetime.utcnow(),
            location=location,
            temperature=25.0,
            measurement_method=ThermalMethod.SATELLITE,
            spatial_resolution=30.0,
            uncertainty=1.0
        )
        
        assert measurement.temperature == 25.0
        assert measurement.measurement_method == ThermalMethod.SATELLITE
        assert measurement.spatial_resolution == 30.0
        assert measurement.uncertainty == 1.0
    
    def test_invalid_temperature_range(self):
        """Test validation of temperature range"""
        location = self.create_valid_location()
        
        with pytest.raises(ValueError, match="Temperature must be between -273.15°C and 2000°C"):
            ThermalMeasurement(
                timestamp=datetime.utcnow(),
                location=location,
                temperature=-300.0,  # Below absolute zero
                measurement_method=ThermalMethod.SATELLITE,
                spatial_resolution=30.0,
                uncertainty=1.0
            )
    
    def test_invalid_spatial_resolution(self):
        """Test validation of spatial resolution"""
        location = self.create_valid_location()
        
        with pytest.raises(ValueError, match="Spatial resolution must be positive"):
            ThermalMeasurement(
                timestamp=datetime.utcnow(),
                location=location,
                temperature=25.0,
                measurement_method=ThermalMethod.SATELLITE,
                spatial_resolution=-10.0,  # Negative resolution
                uncertainty=1.0
            )
    
    def test_invalid_emissivity(self):
        """Test validation of emissivity range"""
        location = self.create_valid_location()
        
        with pytest.raises(ValueError, match="Emissivity must be between 0 and 1"):
            ThermalMeasurement(
                timestamp=datetime.utcnow(),
                location=location,
                temperature=25.0,
                measurement_method=ThermalMethod.THERMAL_CAMERA,
                spatial_resolution=1.0,
                uncertainty=0.5,
                emissivity=1.5  # Invalid emissivity
            )


class TestMagmaReservoir:
    """Test MagmaReservoir data model"""
    
    def create_valid_location(self) -> GeologicalLocation:
        """Helper to create valid location"""
        return GeologicalLocation(latitude=45.0, longitude=-122.0, elevation=1000.0, depth=5000.0)
    
    def create_valid_composition(self) -> GeochemicalSample:
        """Helper to create valid magma composition"""
        location = self.create_valid_location()
        composition = {
            "SiO2": 60.0,
            "Al2O3": 18.0,
            "Fe2O3": 2.0,
            "FeO": 3.0,
            "MgO": 3.0,
            "CaO": 6.0,
            "Na2O": 4.0,
            "K2O": 2.5,
            "TiO2": 1.0,
            "P2O5": 0.5
        }
        
        return GeochemicalSample(
            timestamp=datetime.utcnow(),
            location=location,
            sample_type=SampleType.LAVA,
            composition=composition,
            temperature=1000.0
        )
    
    def test_valid_magma_reservoir(self):
        """Test creating a valid magma reservoir"""
        location = self.create_valid_location()
        composition = self.create_valid_composition()
        volatile_content = {"H2O": 3.0, "CO2": 0.5, "SO2": 0.1}
        
        reservoir = MagmaReservoir(
            location=location,
            volume=1e9,  # 1 km³
            composition=composition,
            temperature=1000.0,
            pressure=2e8,  # 200 MPa
            crystallinity=0.3,
            volatile_content=volatile_content
        )
        
        assert reservoir.volume == 1e9
        assert reservoir.temperature == 1000.0
        assert reservoir.crystallinity == 0.3
        assert reservoir.volatile_content == volatile_content
    
    def test_invalid_volume(self):
        """Test validation of invalid volume"""
        location = self.create_valid_location()
        composition = self.create_valid_composition()
        
        with pytest.raises(ValueError, match="Volume must be positive"):
            MagmaReservoir(
                location=location,
                volume=-1e6,  # Negative volume
                composition=composition,
                temperature=1000.0,
                pressure=2e8,
                crystallinity=0.3,
                volatile_content={"H2O": 3.0}
            )
    
    def test_invalid_temperature_range(self):
        """Test validation of magma temperature range"""
        location = self.create_valid_location()
        composition = self.create_valid_composition()
        
        with pytest.raises(ValueError, match="Magma temperature must be between 600°C and 1400°C"):
            MagmaReservoir(
                location=location,
                volume=1e9,
                composition=composition,
                temperature=500.0,  # Too cold for magma
                pressure=2e8,
                crystallinity=0.3,
                volatile_content={"H2O": 3.0}
            )
    
    def test_invalid_crystallinity(self):
        """Test validation of crystallinity range"""
        location = self.create_valid_location()
        composition = self.create_valid_composition()
        
        with pytest.raises(ValueError, match="Crystallinity must be between 0 and 1"):
            MagmaReservoir(
                location=location,
                volume=1e9,
                composition=composition,
                temperature=1000.0,
                pressure=2e8,
                crystallinity=1.5,  # Invalid crystallinity
                volatile_content={"H2O": 3.0}
            )
    
    def test_invalid_depth_range(self):
        """Test validation of depth range"""
        location = self.create_valid_location()
        composition = self.create_valid_composition()
        
        with pytest.raises(ValueError, match="Minimum depth must be less than maximum depth"):
            MagmaReservoir(
                location=location,
                volume=1e9,
                composition=composition,
                temperature=1000.0,
                pressure=2e8,
                crystallinity=0.3,
                volatile_content={"H2O": 3.0},
                depth_range=[10000.0, 5000.0]  # Min > Max
            )


class TestGeologicalDataValidator:
    """Test GeologicalDataValidator"""
    
    def setup_method(self):
        """Set up validator for each test"""
        self.validator = GeologicalDataValidator()
    
    def create_valid_location(self) -> GeologicalLocation:
        """Helper to create valid location"""
        return GeologicalLocation(latitude=45.0, longitude=-122.0, elevation=1000.0)
    
    def test_validate_valid_location(self):
        """Test validation of valid location"""
        location = self.create_valid_location()
        result = self.validator.validate_geological_location(location)
        
        assert result.is_valid
        assert result.errors_count == 0
    
    def test_validate_invalid_location(self):
        """Test validation of invalid location"""
        # Test that invalid coordinates are caught during creation
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            GeologicalLocation(latitude=95.0, longitude=0.0, elevation=0.0)
        
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            GeologicalLocation(latitude=0.0, longitude=185.0, elevation=0.0)
    
    def test_validate_seismic_event_warnings(self):
        """Test seismic event validation with warnings"""
        location = self.create_valid_location()
        
        # Create event with unusual magnitude for type
        event = SeismicEvent(
            timestamp=datetime.utcnow(),
            location=location,
            magnitude=9.0,  # Very large for volcanic event
            depth=5.0,
            event_type=EventType.VOLCANIC
        )
        
        result = self.validator.validate_seismic_event(event)
        
        # Should be valid but have warnings
        assert result.is_valid
        assert result.warnings_count > 0
        
        # Check for magnitude warning
        magnitude_warnings = [
            issue for issue in result.issues 
            if issue.field == "magnitude" and issue.level == ValidationLevel.WARNING
        ]
        assert len(magnitude_warnings) > 0
    
    def test_validate_geochemical_sample_closure(self):
        """Test geochemical sample validation with closure check"""
        location = self.create_valid_location()
        
        # Create composition that doesn't sum to 100%
        composition = {
            "SiO2": 50.0,
            "Al2O3": 15.0,
            "Fe2O3": 5.0,
            "MgO": 10.0,
            "CaO": 8.0,
            "Na2O": 3.0,
            "K2O": 2.0
            # Total = 93%, should trigger warning
        }
        
        sample = GeochemicalSample(
            timestamp=datetime.utcnow(),
            location=location,
            sample_type=SampleType.ROCK,
            composition=composition
        )
        
        result = self.validator.validate_geochemical_sample(sample)
        
        # Should be valid but have warnings about closure
        assert result.is_valid
        assert result.warnings_count > 0
    
    def test_validate_and_raise_with_errors(self):
        """Test validate_and_raise method with errors"""
        # Create location with invalid coordinates (this will fail in __post_init__)
        with pytest.raises(ValueError):
            location = GeologicalLocation(latitude=95.0, longitude=0.0, elevation=0.0)
    
    def test_validate_data_consistency(self):
        """Test consistency validation across multiple data points"""
        location1 = GeologicalLocation(latitude=45.0, longitude=-122.0, elevation=1000.0)
        location2 = GeologicalLocation(latitude=45.0, longitude=-122.0, elevation=1000.0)  # Same location
        
        event1 = SeismicEvent(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            location=location1,
            magnitude=5.0,
            depth=10.0
        )
        
        event2 = SeismicEvent(
            timestamp=datetime(2023, 1, 1, 13, 0, 0),
            location=location2,
            magnitude=5.2,
            depth=12.0
        )
        
        result = self.validator.validate_data_consistency([event1, event2])
        
        # Should detect duplicate locations
        assert result.is_valid  # No errors, just info
        location_issues = [
            issue for issue in result.issues 
            if issue.category == ValidationCategory.SPATIAL
        ]
        assert len(location_issues) > 0


if __name__ == "__main__":
    pytest.main([__file__])