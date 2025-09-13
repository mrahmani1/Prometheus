"""
Standardized geological data formats and schemas
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field


class DataType(Enum):
    """Geological data types"""
    SEISMIC = "seismic"
    VOLCANIC = "volcanic"
    GEOCHEMICAL = "geochemical"
    DEFORMATION = "deformation"
    THERMAL = "thermal"
    MAGNETIC = "magnetic"
    GRAVITY = "gravity"
    GEOLOGICAL_SURVEY = "geological_survey"
    MTSAR = "mtsar"
    GPS = "gps"


class CoordinateSystem(Enum):
    """Supported coordinate systems"""
    WGS84 = "EPSG:4326"
    UTM_NORTH = "UTM_N"
    UTM_SOUTH = "UTM_S"
    LOCAL_CARTESIAN = "LOCAL"
    GEOGRAPHIC = "GEOGRAPHIC"


class TemporalResolution(Enum):
    """Temporal resolution types"""
    MILLISECOND = "millisecond"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


@dataclass
class Point3D:
    """3D point with coordinate system information"""
    x: float
    y: float
    z: float
    coordinate_system: CoordinateSystem
    uncertainty: Optional[Dict[str, float]] = None


@dataclass
class TimeRange:
    """Time range with timezone information"""
    start: datetime
    end: datetime
    timezone: str = "UTC"


@dataclass
class QualityMetrics:
    """Data quality assessment metrics"""
    completeness: float  # 0-1
    accuracy: float  # 0-1
    consistency: float  # 0-1
    timeliness: float  # 0-1
    overall_score: float  # 0-1
    flags: List[str] = field(default_factory=list)


@dataclass
class ProcessingHistory:
    """Record of data processing steps"""
    step_id: str
    timestamp: datetime
    operation: str
    parameters: Dict[str, Any]
    input_hash: str
    output_hash: str
    processor_version: str


class StandardizedDataFormat(BaseModel):
    """Standardized format for all geological data"""
    
    # Core identification
    data_id: str = Field(..., description="Unique identifier for this data record")
    data_type: DataType = Field(..., description="Type of geological data")
    
    # Spatial information
    location: Dict[str, float] = Field(..., description="Spatial coordinates")
    coordinate_system: CoordinateSystem = Field(default=CoordinateSystem.WGS84)
    spatial_uncertainty: Optional[Dict[str, float]] = None
    
    # Temporal information
    timestamp: datetime = Field(..., description="Observation timestamp in UTC")
    temporal_resolution: TemporalResolution = Field(default=TemporalResolution.SECOND)
    temporal_uncertainty: Optional[float] = None  # seconds
    
    # Data values
    values: Dict[str, Union[float, List[float], np.ndarray]] = Field(..., description="Measured values")
    units: Dict[str, str] = Field(..., description="Units for each value")
    value_uncertainty: Optional[Dict[str, float]] = None
    
    # Metadata
    source: str = Field(..., description="Data source identifier")
    instrument: Optional[str] = None
    method: Optional[str] = None
    processing_level: str = Field(default="raw", description="Level of data processing")
    
    # Quality information
    quality_metrics: Optional[QualityMetrics] = None
    quality_flags: List[str] = Field(default_factory=list)
    
    # Processing history
    processing_history: List[ProcessingHistory] = Field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class GeologicalDataSchema:
    """Schema definitions for different geological data types"""
    
    @staticmethod
    def seismic_schema() -> Dict[str, Any]:
        """Schema for seismic data"""
        return {
            "required_fields": ["amplitude", "frequency", "phase"],
            "optional_fields": ["magnitude", "depth", "azimuth"],
            "units": {
                "amplitude": "m/s",
                "frequency": "Hz",
                "phase": "radians",
                "magnitude": "Ml",
                "depth": "km",
                "azimuth": "degrees"
            },
            "coordinate_systems": [CoordinateSystem.WGS84, CoordinateSystem.UTM_NORTH],
            "temporal_resolution": [TemporalResolution.MILLISECOND, TemporalResolution.SECOND]
        }
    
    @staticmethod
    def volcanic_schema() -> Dict[str, Any]:
        """Schema for volcanic monitoring data"""
        return {
            "required_fields": ["parameter_type", "value"],
            "optional_fields": ["alert_level", "confidence", "trend"],
            "parameter_types": [
                "deformation_rate", "gas_emission", "thermal_anomaly", 
                "seismic_energy", "tremor_amplitude"
            ],
            "units": {
                "deformation_rate": "mm/year",
                "gas_emission": "tonnes/day",
                "thermal_anomaly": "K",
                "seismic_energy": "J",
                "tremor_amplitude": "m/s"
            },
            "coordinate_systems": [CoordinateSystem.WGS84],
            "temporal_resolution": [TemporalResolution.MINUTE, TemporalResolution.HOUR]
        }
    
    @staticmethod
    def geochemical_schema() -> Dict[str, Any]:
        """Schema for geochemical data"""
        return {
            "required_fields": ["element", "concentration"],
            "optional_fields": ["isotope_ratio", "mineral_phase", "analysis_method"],
            "units": {
                "concentration": "ppm",
                "isotope_ratio": "ratio",
                "temperature": "C",
                "pressure": "MPa"
            },
            "coordinate_systems": [CoordinateSystem.WGS84],
            "temporal_resolution": [TemporalResolution.DAY, TemporalResolution.YEAR]
        }
    
    @staticmethod
    def mtsar_schema() -> Dict[str, Any]:
        """Schema for MT-SAR deformation data"""
        return {
            "required_fields": ["displacement", "coherence", "orbit_direction"],
            "optional_fields": ["baseline_length", "temporal_baseline", "atmospheric_delay"],
            "units": {
                "displacement": "mm",
                "coherence": "unitless",
                "baseline_length": "m",
                "temporal_baseline": "days",
                "atmospheric_delay": "mm"
            },
            "coordinate_systems": [CoordinateSystem.WGS84, CoordinateSystem.UTM_NORTH],
            "temporal_resolution": [TemporalResolution.DAY, TemporalResolution.WEEK]
        }
    
    @staticmethod
    def gps_schema() -> Dict[str, Any]:
        """Schema for GPS deformation data"""
        return {
            "required_fields": ["displacement_east", "displacement_north", "displacement_up"],
            "optional_fields": ["velocity_east", "velocity_north", "velocity_up"],
            "units": {
                "displacement_east": "mm",
                "displacement_north": "mm", 
                "displacement_up": "mm",
                "velocity_east": "mm/year",
                "velocity_north": "mm/year",
                "velocity_up": "mm/year"
            },
            "coordinate_systems": [CoordinateSystem.WGS84],
            "temporal_resolution": [TemporalResolution.HOUR, TemporalResolution.DAY]
        }


@dataclass
class TransformationConfig:
    """Configuration for data transformation pipeline"""
    target_coordinate_system: CoordinateSystem = CoordinateSystem.WGS84
    target_temporal_resolution: TemporalResolution = TemporalResolution.MINUTE
    target_timezone: str = "UTC"
    interpolation_method: str = "linear"
    aggregation_method: str = "mean"
    quality_threshold: float = 0.7
    fill_gaps: bool = True
    max_gap_duration: float = 3600  # seconds
    export_format: str = "netcdf"
    include_uncertainty: bool = True
    preserve_raw_data: bool = True


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quality_score: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)