"""
Geological Data Models

Core data structures for representing geological entities and measurements
with scientific constraints and validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import numpy as np
import json
from pydantic import BaseModel, Field, validator, root_validator


class CoordinateSystem(str, Enum):
    """Supported coordinate systems"""
    WGS84 = "WGS84"
    NAD83 = "NAD83"
    ITRF2014 = "ITRF2014"
    UTM = "UTM"


class EventType(str, Enum):
    """Types of seismic events"""
    EARTHQUAKE = "earthquake"
    VOLCANIC = "volcanic"
    EXPLOSION = "explosion"
    LANDSLIDE = "landslide"
    ROCKFALL = "rockfall"
    UNKNOWN = "unknown"


class MeasurementType(str, Enum):
    """Types of deformation measurements"""
    GPS = "GPS"
    INSAR = "InSAR"
    TILTMETER = "tiltmeter"
    STRAINMETER = "strainmeter"
    LEVELING = "leveling"
    EDM = "EDM"  # Electronic Distance Measurement


class SampleType(str, Enum):
    """Types of geological samples"""
    ROCK = "rock"
    GAS = "gas"
    WATER = "water"
    SOIL = "soil"
    LAVA = "lava"
    ASH = "ash"
    FUMAROLE = "fumarole"


class ThermalMethod(str, Enum):
    """Thermal measurement methods"""
    SATELLITE = "satellite"
    GROUND_BASED = "ground_based"
    BOREHOLE = "borehole"
    THERMAL_CAMERA = "thermal_camera"
    THERMOCOUPLE = "thermocouple"


@dataclass
class GeologicalLocation:
    """
    Represents a geological location with coordinates and metadata.
    
    Attributes:
        latitude: Latitude in decimal degrees (-90 to 90)
        longitude: Longitude in decimal degrees (-180 to 180)
        elevation: Elevation above sea level in meters
        depth: Depth below surface in meters (positive downward)
        coordinate_system: Reference coordinate system
        uncertainty_horizontal: Horizontal position uncertainty in meters
        uncertainty_vertical: Vertical position uncertainty in meters
    """
    latitude: float
    longitude: float
    elevation: float
    depth: Optional[float] = None
    coordinate_system: CoordinateSystem = CoordinateSystem.WGS84
    uncertainty_horizontal: Optional[float] = None
    uncertainty_vertical: Optional[float] = None
    
    def __post_init__(self):
        """Validate coordinates after initialization"""
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {self.longitude}")
        if self.depth is not None and self.depth < 0:
            raise ValueError(f"Depth must be positive (downward), got {self.depth}")
        if self.uncertainty_horizontal is not None and self.uncertainty_horizontal < 0:
            raise ValueError("Horizontal uncertainty must be non-negative")
        if self.uncertainty_vertical is not None and self.uncertainty_vertical < 0:
            raise ValueError("Vertical uncertainty must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
            "depth": self.depth,
            "coordinate_system": self.coordinate_system.value,
            "uncertainty_horizontal": self.uncertainty_horizontal,
            "uncertainty_vertical": self.uncertainty_vertical
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeologicalLocation':
        """Create from dictionary"""
        return cls(
            latitude=data["latitude"],
            longitude=data["longitude"],
            elevation=data["elevation"],
            depth=data.get("depth"),
            coordinate_system=CoordinateSystem(data.get("coordinate_system", "WGS84")),
            uncertainty_horizontal=data.get("uncertainty_horizontal"),
            uncertainty_vertical=data.get("uncertainty_vertical")
        )


@dataclass
class SeismicEvent:
    """
    Represents a seismic event with location, magnitude, and waveform data.
    
    Attributes:
        timestamp: Event occurrence time (UTC)
        location: Geological location of the event
        magnitude: Event magnitude (various scales)
        magnitude_type: Type of magnitude scale (Mw, ML, Ms, etc.)
        depth: Focal depth in kilometers
        event_type: Classification of seismic event
        waveform_data: Optional seismic waveform data
        confidence: Confidence in event parameters (0-1)
        agency: Reporting agency or network
        event_id: Unique identifier for the event
    """
    timestamp: datetime
    location: GeologicalLocation
    magnitude: float
    depth: float
    event_type: EventType = EventType.EARTHQUAKE
    magnitude_type: str = "Mw"
    waveform_data: Optional[np.ndarray] = None
    confidence: float = 1.0
    agency: Optional[str] = None
    event_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate seismic event parameters"""
        if not -5.0 <= self.magnitude <= 12.0:
            raise ValueError(f"Magnitude must be between -5.0 and 12.0, got {self.magnitude}")
        if not 0 <= self.depth <= 800:  # Maximum depth for earthquakes
            raise ValueError(f"Depth must be between 0 and 800 km, got {self.depth}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.waveform_data is not None and len(self.waveform_data.shape) != 1:
            raise ValueError("Waveform data must be 1-dimensional array")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "location": self.location.to_dict(),
            "magnitude": self.magnitude,
            "magnitude_type": self.magnitude_type,
            "depth": self.depth,
            "event_type": self.event_type.value,
            "waveform_data": self.waveform_data.tolist() if self.waveform_data is not None else None,
            "confidence": self.confidence,
            "agency": self.agency,
            "event_id": self.event_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SeismicEvent':
        """Create from dictionary"""
        waveform_data = None
        if data.get("waveform_data") is not None:
            waveform_data = np.array(data["waveform_data"])
        
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            location=GeologicalLocation.from_dict(data["location"]),
            magnitude=data["magnitude"],
            magnitude_type=data.get("magnitude_type", "Mw"),
            depth=data["depth"],
            event_type=EventType(data.get("event_type", "earthquake")),
            waveform_data=waveform_data,
            confidence=data.get("confidence", 1.0),
            agency=data.get("agency"),
            event_id=data.get("event_id")
        )


@dataclass
class DeformationMeasurement:
    """
    Represents ground deformation measurements from various sources.
    
    Attributes:
        timestamp: Measurement time (UTC)
        location: Measurement location
        displacement_vector: 3D displacement [x, y, z] in meters
        measurement_type: Type of measurement system
        uncertainty: Uncertainty in each dimension (meters)
        reference_frame: Geodetic reference frame
        velocity_vector: Optional velocity [vx, vy, vz] in m/year
        station_id: Identifier for measurement station
    """
    timestamp: datetime
    location: GeologicalLocation
    displacement_vector: np.ndarray
    measurement_type: MeasurementType
    uncertainty: np.ndarray
    reference_frame: str = "ITRF2014"
    velocity_vector: Optional[np.ndarray] = None
    station_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate deformation measurement parameters"""
        if self.displacement_vector.shape != (3,):
            raise ValueError("Displacement vector must be 3-dimensional [x, y, z]")
        if self.uncertainty.shape != (3,):
            raise ValueError("Uncertainty must be 3-dimensional [σx, σy, σz]")
        if np.any(self.uncertainty < 0):
            raise ValueError("Uncertainty values must be non-negative")
        if self.velocity_vector is not None and self.velocity_vector.shape != (3,):
            raise ValueError("Velocity vector must be 3-dimensional [vx, vy, vz]")
        
        # Check for reasonable displacement values (< 100m for most applications)
        if np.any(np.abs(self.displacement_vector) > 100):
            import warnings
            warnings.warn("Large displacement detected (>100m), please verify data")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "location": self.location.to_dict(),
            "displacement_vector": self.displacement_vector.tolist(),
            "measurement_type": self.measurement_type.value,
            "uncertainty": self.uncertainty.tolist(),
            "reference_frame": self.reference_frame,
            "velocity_vector": self.velocity_vector.tolist() if self.velocity_vector is not None else None,
            "station_id": self.station_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeformationMeasurement':
        """Create from dictionary"""
        velocity_vector = None
        if data.get("velocity_vector") is not None:
            velocity_vector = np.array(data["velocity_vector"])
        
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            location=GeologicalLocation.from_dict(data["location"]),
            displacement_vector=np.array(data["displacement_vector"]),
            measurement_type=MeasurementType(data["measurement_type"]),
            uncertainty=np.array(data["uncertainty"]),
            reference_frame=data.get("reference_frame", "ITRF2014"),
            velocity_vector=velocity_vector,
            station_id=data.get("station_id")
        )


@dataclass
class GeochemicalSample:
    """
    Represents geochemical sample data with composition and metadata.
    
    Attributes:
        timestamp: Sample collection time (UTC)
        location: Sample location
        sample_type: Type of geological sample
        composition: Element/compound concentrations (ppm, wt%, etc.)
        isotope_ratios: Isotopic ratios for dating and source identification
        temperature: Sample temperature in Celsius
        pressure: Sample pressure in Pa
        ph: pH value for water/gas samples
        sample_id: Unique sample identifier
        collection_method: How the sample was collected
    """
    timestamp: datetime
    location: GeologicalLocation
    sample_type: SampleType
    composition: Dict[str, float]
    isotope_ratios: Optional[Dict[str, float]] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    ph: Optional[float] = None
    sample_id: Optional[str] = None
    collection_method: Optional[str] = None
    
    def __post_init__(self):
        """Validate geochemical sample parameters"""
        # Validate composition values are non-negative
        for element, concentration in self.composition.items():
            if concentration < 0:
                raise ValueError(f"Concentration for {element} must be non-negative, got {concentration}")
        
        # Validate temperature range
        if self.temperature is not None:
            if not -273.15 <= self.temperature <= 2000:  # Absolute zero to magma temps
                raise ValueError(f"Temperature must be between -273.15°C and 2000°C, got {self.temperature}")
        
        # Validate pressure
        if self.pressure is not None and self.pressure < 0:
            raise ValueError(f"Pressure must be non-negative, got {self.pressure}")
        
        # Validate pH
        if self.ph is not None and not 0 <= self.ph <= 14:
            raise ValueError(f"pH must be between 0 and 14, got {self.ph}")
        
        # Validate isotope ratios
        if self.isotope_ratios:
            for ratio_name, ratio_value in self.isotope_ratios.items():
                if ratio_value <= 0:
                    raise ValueError(f"Isotope ratio {ratio_name} must be positive, got {ratio_value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "location": self.location.to_dict(),
            "sample_type": self.sample_type.value,
            "composition": self.composition,
            "isotope_ratios": self.isotope_ratios,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "ph": self.ph,
            "sample_id": self.sample_id,
            "collection_method": self.collection_method
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeochemicalSample':
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            location=GeologicalLocation.from_dict(data["location"]),
            sample_type=SampleType(data["sample_type"]),
            composition=data["composition"],
            isotope_ratios=data.get("isotope_ratios"),
            temperature=data.get("temperature"),
            pressure=data.get("pressure"),
            ph=data.get("ph"),
            sample_id=data.get("sample_id"),
            collection_method=data.get("collection_method")
        )


@dataclass
class ThermalMeasurement:
    """
    Represents thermal measurements from various sources.
    
    Attributes:
        timestamp: Measurement time (UTC)
        location: Measurement location
        temperature: Temperature in Celsius
        measurement_method: Method used for measurement
        spatial_resolution: Spatial resolution in meters
        uncertainty: Temperature uncertainty in Celsius
        emissivity: Surface emissivity (for thermal infrared)
        atmospheric_correction: Whether atmospheric correction was applied
        instrument_id: Identifier for measuring instrument
    """
    timestamp: datetime
    location: GeologicalLocation
    temperature: float
    measurement_method: ThermalMethod
    spatial_resolution: float
    uncertainty: float
    emissivity: Optional[float] = None
    atmospheric_correction: bool = False
    instrument_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate thermal measurement parameters"""
        if not -273.15 <= self.temperature <= 2000:
            raise ValueError(f"Temperature must be between -273.15°C and 2000°C, got {self.temperature}")
        if self.spatial_resolution <= 0:
            raise ValueError(f"Spatial resolution must be positive, got {self.spatial_resolution}")
        if self.uncertainty < 0:
            raise ValueError(f"Uncertainty must be non-negative, got {self.uncertainty}")
        if self.emissivity is not None and not 0 <= self.emissivity <= 1:
            raise ValueError(f"Emissivity must be between 0 and 1, got {self.emissivity}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "location": self.location.to_dict(),
            "temperature": self.temperature,
            "measurement_method": self.measurement_method.value,
            "spatial_resolution": self.spatial_resolution,
            "uncertainty": self.uncertainty,
            "emissivity": self.emissivity,
            "atmospheric_correction": self.atmospheric_correction,
            "instrument_id": self.instrument_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThermalMeasurement':
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            location=GeologicalLocation.from_dict(data["location"]),
            temperature=data["temperature"],
            measurement_method=ThermalMethod(data["measurement_method"]),
            spatial_resolution=data["spatial_resolution"],
            uncertainty=data["uncertainty"],
            emissivity=data.get("emissivity"),
            atmospheric_correction=data.get("atmospheric_correction", False),
            instrument_id=data.get("instrument_id")
        )


@dataclass
class MagmaReservoir:
    """
    Represents a magma reservoir with physical and chemical properties.
    
    Attributes:
        location: Reservoir location (center point)
        volume: Reservoir volume in cubic meters
        composition: Magma composition
        temperature: Magma temperature in Celsius
        pressure: Magma pressure in Pa
        crystallinity: Fraction of crystals (0-1)
        volatile_content: Volatile concentrations (H2O, CO2, SO2, etc.)
        connectivity: List of connected reservoir IDs
        reservoir_id: Unique reservoir identifier
        depth_range: Depth range [min_depth, max_depth] in meters
        shape_parameters: Geometric parameters for reservoir shape
    """
    location: GeologicalLocation
    volume: float
    composition: GeochemicalSample
    temperature: float
    pressure: float
    crystallinity: float
    volatile_content: Dict[str, float]
    connectivity: List[str] = field(default_factory=list)
    reservoir_id: Optional[str] = None
    depth_range: Optional[List[float]] = None
    shape_parameters: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate magma reservoir parameters"""
        if self.volume <= 0:
            raise ValueError(f"Volume must be positive, got {self.volume}")
        if not 600 <= self.temperature <= 1400:  # Typical magma temperature range
            raise ValueError(f"Magma temperature must be between 600°C and 1400°C, got {self.temperature}")
        if self.pressure <= 0:
            raise ValueError(f"Pressure must be positive, got {self.pressure}")
        if not 0 <= self.crystallinity <= 1:
            raise ValueError(f"Crystallinity must be between 0 and 1, got {self.crystallinity}")
        
        # Validate volatile content
        for volatile, concentration in self.volatile_content.items():
            if concentration < 0:
                raise ValueError(f"Volatile concentration for {volatile} must be non-negative")
        
        # Validate depth range
        if self.depth_range is not None:
            if len(self.depth_range) != 2:
                raise ValueError("Depth range must contain exactly 2 values [min, max]")
            if self.depth_range[0] >= self.depth_range[1]:
                raise ValueError("Minimum depth must be less than maximum depth")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "location": self.location.to_dict(),
            "volume": self.volume,
            "composition": self.composition.to_dict(),
            "temperature": self.temperature,
            "pressure": self.pressure,
            "crystallinity": self.crystallinity,
            "volatile_content": self.volatile_content,
            "connectivity": self.connectivity,
            "reservoir_id": self.reservoir_id,
            "depth_range": self.depth_range,
            "shape_parameters": self.shape_parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MagmaReservoir':
        """Create from dictionary"""
        return cls(
            location=GeologicalLocation.from_dict(data["location"]),
            volume=data["volume"],
            composition=GeochemicalSample.from_dict(data["composition"]),
            temperature=data["temperature"],
            pressure=data["pressure"],
            crystallinity=data["crystallinity"],
            volatile_content=data["volatile_content"],
            connectivity=data.get("connectivity", []),
            reservoir_id=data.get("reservoir_id"),
            depth_range=data.get("depth_range"),
            shape_parameters=data.get("shape_parameters")
        )