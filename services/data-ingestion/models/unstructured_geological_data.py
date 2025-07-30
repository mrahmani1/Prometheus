"""Extended geological data models for unstructured data storage in MongoDB"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import numpy as np
from enum import Enum

from .geological_data import GeologicalLocation, DataQuality

class RockType(str, Enum):
    IGNEOUS = "igneous"
    SEDIMENTARY = "sedimentary"
    METAMORPHIC = "metamorphic"
    VOLCANIC = "volcanic"
    PLUTONIC = "plutonic"
    UNKNOWN = "unknown"

class MetamorphicGrade(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"
    UNKNOWN = "unknown"

class CrystalSystem(str, Enum):
    CUBIC = "cubic"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"
    UNKNOWN = "unknown"

@dataclass
class RockSample:
    """Represents a rock sample with detailed geological information"""
    timestamp: datetime
    location: GeologicalLocation
    sample_id: str
    rock_type: RockType
    formation: Optional[str] = None
    age_ma: Optional[float] = None  # Age in millions of years
    texture: Optional[str] = None
    grain_size: Optional[str] = None
    color: Optional[str] = None
    mineral_assemblage: List[str] = field(default_factory=list)
    metamorphic_grade: Optional[MetamorphicGrade] = None
    alteration_type: Optional[str] = None
    alteration_intensity: Optional[str] = None
    weathering_degree: Optional[str] = None
    porosity: Optional[float] = None
    permeability: Optional[float] = None
    density: Optional[float] = None  # g/cm³
    magnetic_susceptibility: Optional[float] = None
    structural_features: List[str] = field(default_factory=list)
    field_notes: Optional[str] = None
    collector: Optional[str] = None
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MineralAnalysis:
    """Represents detailed mineral analysis data"""
    timestamp: datetime
    location: GeologicalLocation
    sample_id: str
    mineral_name: str
    chemical_formula: Optional[str] = None
    crystal_system: Optional[CrystalSystem] = None
    hardness: Optional[float] = None  # Mohs scale
    specific_gravity: Optional[float] = None
    color: Optional[str] = None
    streak: Optional[str] = None
    luster: Optional[str] = None
    cleavage: Optional[str] = None
    fracture: Optional[str] = None
    optical_properties: Dict[str, Any] = field(default_factory=dict)
    chemical_composition: Dict[str, float] = field(default_factory=dict)
    trace_elements: Dict[str, float] = field(default_factory=dict)
    isotope_ratios: Dict[str, float] = field(default_factory=dict)
    xrd_data: Optional[Dict[str, Any]] = None  # X-ray diffraction data
    sem_data: Optional[Dict[str, Any]] = None  # SEM analysis data
    microprobe_data: Optional[Dict[str, Any]] = None
    analysis_method: str = ""
    analyst: Optional[str] = None
    laboratory: Optional[str] = None
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FluidSample:
    """Represents fluid sample analysis (water, gas, etc.)"""
    timestamp: datetime
    location: GeologicalLocation
    sample_id: str
    fluid_type: str  # water, gas, oil, etc.
    temperature: Optional[float] = None  # Celsius
    ph: Optional[float] = None
    conductivity: Optional[float] = None  # µS/cm
    salinity: Optional[float] = None  # ppt
    dissolved_oxygen: Optional[float] = None  # mg/L
    major_ions: Dict[str, float] = field(default_factory=dict)  # mg/L
    trace_elements: Dict[str, float] = field(default_factory=dict)  # µg/L
    isotope_ratios: Dict[str, float] = field(default_factory=dict)
    gas_composition: Dict[str, float] = field(default_factory=dict)  # vol%
    organic_compounds: Dict[str, float] = field(default_factory=dict)
    bacterial_content: Optional[Dict[str, Any]] = None
    sampling_method: Optional[str] = None
    preservation_method: Optional[str] = None
    analysis_date: Optional[datetime] = None
    laboratory: Optional[str] = None
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CoreSample:
    """Represents drill core sample data"""
    timestamp: datetime
    location: GeologicalLocation
    core_id: str
    hole_id: str
    depth_from: float  # meters
    depth_to: float  # meters
    recovery_percentage: Optional[float] = None
    rqd: Optional[float] = None  # Rock Quality Designation
    lithology: Optional[str] = None
    alteration: Optional[str] = None
    mineralization: Optional[str] = None
    structure: Optional[str] = None
    hardness: Optional[str] = None
    color: Optional[str] = None
    texture: Optional[str] = None
    grain_size: Optional[str] = None
    fracture_frequency: Optional[float] = None  # fractures per meter
    fracture_orientation: List[Dict[str, float]] = field(default_factory=list)
    geotechnical_properties: Dict[str, float] = field(default_factory=dict)
    assay_results: Dict[str, float] = field(default_factory=dict)
    photos: List[str] = field(default_factory=list)  # photo file paths
    geologist: Optional[str] = None
    logging_date: Optional[datetime] = None
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeologicalSurvey:
    """Represents geological survey data"""
    timestamp: datetime
    location: GeologicalLocation
    survey_id: str
    survey_type: str  # mapping, geophysical, geochemical, etc.
    survey_method: str
    surveyor: Optional[str] = None
    organization: Optional[str] = None
    equipment_used: List[str] = field(default_factory=list)
    weather_conditions: Optional[str] = None
    observations: Dict[str, Any] = field(default_factory=dict)
    measurements: Dict[str, float] = field(default_factory=dict)
    samples_collected: List[str] = field(default_factory=list)
    photos: List[str] = field(default_factory=list)
    sketches: List[str] = field(default_factory=list)
    gps_waypoints: List[Dict[str, float]] = field(default_factory=list)
    geological_units: List[Dict[str, Any]] = field(default_factory=list)
    structural_measurements: List[Dict[str, Any]] = field(default_factory=list)
    hazards_identified: List[str] = field(default_factory=list)
    recommendations: Optional[str] = None
    report_reference: Optional[str] = None
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FieldObservation:
    """Represents field geological observations"""
    timestamp: datetime
    location: GeologicalLocation
    observation_id: str
    observer: str
    observation_type: str  # outcrop, structure, mineral, etc.
    description: str
    geological_unit: Optional[str] = None
    rock_type: Optional[RockType] = None
    strike: Optional[float] = None  # degrees
    dip: Optional[float] = None  # degrees
    trend: Optional[float] = None  # degrees
    plunge: Optional[float] = None  # degrees
    thickness: Optional[float] = None  # meters
    width: Optional[float] = None  # meters
    length: Optional[float] = None  # meters
    color: Optional[str] = None
    texture: Optional[str] = None
    grain_size: Optional[str] = None
    weathering: Optional[str] = None
    vegetation_cover: Optional[str] = None
    accessibility: Optional[str] = None
    confidence_level: Optional[str] = None
    photos: List[str] = field(default_factory=list)
    sketches: List[str] = field(default_factory=list)
    related_observations: List[str] = field(default_factory=list)
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LaboratoryResult:
    """Represents laboratory analysis results"""
    timestamp: datetime
    sample_id: str
    laboratory: str
    analysis_type: str
    analysis_method: str
    analysis_date: datetime
    analyst: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    detection_limits: Dict[str, float] = field(default_factory=dict)
    uncertainties: Dict[str, float] = field(default_factory=dict)
    quality_control: Dict[str, Any] = field(default_factory=dict)
    standards_used: List[str] = field(default_factory=list)
    blanks_analyzed: List[str] = field(default_factory=list)
    duplicates_analyzed: List[str] = field(default_factory=list)
    instrument_used: Optional[str] = None
    calibration_date: Optional[datetime] = None
    data_files: List[str] = field(default_factory=list)
    report_reference: Optional[str] = None
    certification: Optional[str] = None
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeophysicalSurvey:
    """Represents geophysical survey data"""
    timestamp: datetime
    location: GeologicalLocation
    survey_id: str
    survey_type: str  # magnetic, gravity, electrical, seismic, etc.
    method: str
    equipment: str
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN
    operator: Optional[str] = None
    survey_parameters: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[Dict[str, Any]] = None
    processed_data: Optional[Dict[str, Any]] = None
    data_quality_metrics: Dict[str, float] = field(default_factory=dict)
    processing_software: Optional[str] = None
    processing_parameters: Dict[str, Any] = field(default_factory=dict)
    interpretation: Optional[str] = None
    anomalies_identified: List[Dict[str, Any]] = field(default_factory=list)
    data_files: List[str] = field(default_factory=list)
    maps_generated: List[str] = field(default_factory=list)
    report_reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BoreholeData:
    """Represents borehole/well data"""
    timestamp: datetime
    location: GeologicalLocation
    borehole_id: str
    total_depth: float  # meters
    drilling_method: str
    purpose: str  # exploration, monitoring, water, etc.
    source: str = ""
    quality: DataQuality = DataQuality.UNKNOWN
    drilling_date: Optional[datetime] = None
    driller: Optional[str] = None
    diameter: Optional[float] = None  # mm
    casing_details: List[Dict[str, Any]] = field(default_factory=list)
    lithology_log: List[Dict[str, Any]] = field(default_factory=list)
    geophysical_logs: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    water_levels: List[Dict[str, Any]] = field(default_factory=list)
    flow_rates: List[Dict[str, Any]] = field(default_factory=list)
    water_quality: List[Dict[str, Any]] = field(default_factory=list)
    completion_details: Dict[str, Any] = field(default_factory=dict)
    abandonment_details: Optional[Dict[str, Any]] = None
    permits: List[str] = field(default_factory=list)
    reports: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)