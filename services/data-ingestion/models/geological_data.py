"""Geological data models for the ingestion service"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DataQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


@dataclass
class GeologicalLocation:
    """Represents a geographical location with geological context"""

    latitude: float
    longitude: float
    elevation: float
    depth: Optional[float] = None
    coordinate_system: str = "WGS84"
    uncertainty_m: Optional[float] = None


@dataclass
class SeismicEvent:
    """Represents a seismic event (earthquake, volcanic, etc.)"""

    timestamp: datetime
    location: GeologicalLocation
    magnitude: float
    depth: float
    event_type: str  # earthquake, volcanic, explosion, etc.
    magnitude_type: str = "unknown"
    event_id: str = ""
    source: str = ""
    confidence: float = 1.0
    waveform_data: Optional[np.ndarray] = None
    quality: DataQuality = DataQuality.UNKNOWN


@dataclass
class VolcanicEvent:
    """Represents a volcanic event"""

    timestamp: datetime
    location: GeologicalLocation
    event_type: str  # eruption, gas_emission, thermal_anomaly, etc.
    intensity: float
    duration_seconds: Optional[float] = None
    event_id: str = ""
    source: str = ""
    confidence: float = 1.0
    quality: DataQuality = DataQuality.UNKNOWN


@dataclass
class DeformationMeasurement:
    """Represents ground deformation measurement"""

    timestamp: datetime
    location: GeologicalLocation
    displacement_vector: np.ndarray  # [x, y, z] displacement in meters
    measurement_type: str  # GPS, InSAR, tiltmeter, etc.
    uncertainty: np.ndarray  # uncertainty in each dimension
    reference_frame: str = "ITRF2014"
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN


@dataclass
class GeochemicalSample:
    """Represents a geochemical sample"""

    timestamp: datetime
    location: GeologicalLocation
    sample_type: str  # rock, gas, water, etc.
    composition: Dict[str, float]  # element/compound concentrations
    isotope_ratios: Optional[Dict[str, float]] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    ph: Optional[float] = None
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN


@dataclass
class ThermalMeasurement:
    """Represents thermal measurement"""

    timestamp: datetime
    location: GeologicalLocation
    temperature: float  # Kelvin
    measurement_method: str  # satellite, ground-based, borehole, etc.
    spatial_resolution: float  # meters
    uncertainty: float
    confidence: float = 1.0
    fire_radiative_power: Optional[float] = None
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN


@dataclass
class SatelliteObservation:
    """Represents satellite observation data"""

    timestamp: datetime
    location: GeologicalLocation
    satellite: str
    instrument: str
    bands: List[str]
    spatial_resolution: float  # meters
    cloud_cover: float  # percentage
    scene_id: str = ""
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN


@dataclass
class FiberOpticMeasurement:
    """Represents fiber optic sensing measurement"""

    timestamp: datetime
    location: GeologicalLocation
    measurement_type: str  # das, dts
    temperature: Optional[float] = None
    temperature_std: Optional[float] = None
    strain_rate: Optional[float] = None
    strain_rate_std: Optional[float] = None
    channel_index: int = 0
    spatial_resolution: float = 1.0  # meters
    sampling_rate: float = 1.0  # Hz
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN


@dataclass
class StrainMeasurement:
    """Represents strain measurement from DAS"""

    timestamp: datetime
    location: GeologicalLocation
    strain_rate: float  # 1/s
    strain_rate_std: float
    strain_rate_max: float
    measurement_type: str = "das"
    channel_index: int = 0
    gauge_length: float = 10.0  # meters
    sampling_rate: float = 1000.0  # Hz
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN


@dataclass
class AtmosphericMeasurement:
    """Represents atmospheric measurement"""

    timestamp: datetime
    location: GeologicalLocation
    pressure: Optional[float] = None  # Pa
    temperature: Optional[float] = None  # K
    humidity: Optional[float] = None  # %
    wind_speed: Optional[float] = None  # m/s
    wind_direction: Optional[float] = None  # degrees
    gas_concentrations: Optional[Dict[str, float]] = None  # ppm/ppb
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN


@dataclass
class HydroacousticMeasurement:
    """Represents hydroacoustic measurement"""

    timestamp: datetime
    location: GeologicalLocation
    pressure: float  # Pa
    frequency_hz: float
    amplitude: float
    measurement_type: str = "hydrophone"
    water_depth: Optional[float] = None  # meters
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN


@dataclass
class GravityMeasurement:
    """Represents gravity measurement"""

    timestamp: datetime
    location: GeologicalLocation
    gravity_value: float  # m/s²
    uncertainty: float
    measurement_type: str  # absolute, relative, satellite
    reference_datum: str = "WGS84"
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN


@dataclass
class MagneticMeasurement:
    """Represents magnetic field measurement"""

    timestamp: datetime
    location: GeologicalLocation
    total_field: float  # nT
    declination: Optional[float] = None  # degrees
    inclination: Optional[float] = None  # degrees
    horizontal_intensity: Optional[float] = None  # nT
    vertical_intensity: Optional[float] = None  # nT
    measurement_type: str = "total_field"
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN


@dataclass
class ProcessedDataBatch:
    """Represents a batch of processed geological data"""

    batch_id: str
    timestamp: datetime
    data_type: str
    source_connector: str
    record_count: int
    processing_time_ms: float
    quality_metrics: Dict[str, float]
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
