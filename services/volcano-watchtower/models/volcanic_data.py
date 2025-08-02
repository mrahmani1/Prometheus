"""
Volcanic Data Models
Data structures for volcanic monitoring and risk assessment
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GeologicalLocation:
    """Geological location with coordinates and elevation"""

    latitude: float
    longitude: float
    elevation: float
    depth: Optional[float] = None
    coordinate_system: str = "WGS84"


@dataclass
class SeismicEvent:
    """Seismic event data structure"""

    timestamp: datetime
    location: GeologicalLocation
    magnitude: float
    depth: float
    event_type: str  # earthquake, volcanic, explosion, etc.
    waveform_data: Optional[np.ndarray] = None
    confidence: float = 1.0
    source_network: str = "unknown"


@dataclass
class DeformationMeasurement:
    """Ground deformation measurement"""

    timestamp: datetime
    location: GeologicalLocation
    displacement_vector: np.ndarray  # [x, y, z] displacement in meters
    measurement_type: str  # GPS, InSAR, tiltmeter, etc.
    uncertainty: np.ndarray  # uncertainty in each dimension
    reference_frame: str = "ITRF2014"
    quality_flag: str = "good"


@dataclass
class GeochemicalSample:
    """Geochemical sample data"""

    timestamp: datetime
    location: GeologicalLocation
    sample_type: str  # rock, gas, water, etc.
    composition: Dict[str, float]  # element/compound concentrations
    isotope_ratios: Optional[Dict[str, float]] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    collection_method: str = "unknown"


@dataclass
class ThermalMeasurement:
    """Thermal measurement data"""

    timestamp: datetime
    location: GeologicalLocation
    temperature: float
    measurement_method: str  # satellite, ground-based, borehole, etc.
    spatial_resolution: float  # meters
    uncertainty: float
    calibration_date: Optional[datetime] = None


@dataclass
class AlertLevel:
    """Volcanic alert level definition"""

    level: int  # 0-5 scale
    description: str
    recommended_actions: List[str]
    validity_period: Tuple[datetime, datetime]  # (start, end) timestamps
    confidence: float


@dataclass
class EruptionForecast:
    """Probabilistic eruption forecast"""

    volcano_id: str
    timestamp: datetime
    probability_by_timeframe: Dict[str, float]  # timeframe -> probability
    uncertainty: float
    confidence: float
    key_indicators: List[str]
    model_ensemble_results: Dict[str, Dict[str, float]]


@dataclass
class MagmaReservoirParameters:
    """Magma reservoir parameter estimates"""

    volcano_id: str
    timestamp: datetime
    location: GeologicalLocation
    depth: float
    volume: float  # cubic meters
    volume_change_rate: float  # m³/s
    pressure: float  # Pa
    pressure_change_rate: float  # Pa/s
    temperature: float  # Celsius
    composition: Dict[str, float]  # element/compound percentages
    crystallinity: float  # fraction of crystals (0-1)
    volatile_content: Dict[str, float]  # H2O, CO2, SO2, etc.
    connectivity_index: float  # 0-1 scale of reservoir connectivity
    connected_reservoirs: List[str]  # IDs of connected reservoirs
    estimation_confidence: float
    estimation_methods: List[str]
    uncertainty_bounds: Dict[str, float]


@dataclass
class VolcanicRiskAssessment:
    """Comprehensive volcanic risk assessment"""

    volcano_id: str
    timestamp: datetime
    alert_level: AlertLevel
    eruption_probability: Dict[str, float]  # timeframe -> probability
    hazard_zones: Dict[str, np.ndarray]  # hazard type -> affected areas
    contributing_factors: List[str]
    model_ensemble_agreement: float
    composite_risk_score: float
    reservoir_parameters: Optional[MagmaReservoirParameters] = None
    confidence_metrics: Dict[str, float] = None
    data_quality_assessment: Dict[str, float] = None


class NotificationChannel(Enum):
    """Notification delivery channels"""

    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    EMERGENCY_BROADCAST = "emergency_broadcast"


@dataclass
class NotificationTarget:
    """Notification target configuration"""

    target_id: str
    target_type: str  # person, organization, system
    channels: List[NotificationChannel]
    alert_level_threshold: int  # minimum alert level to notify
    contact_info: Dict[str, str]  # channel -> contact details
    priority: int = 1  # 1=highest, 5=lowest
    active: bool = True


@dataclass
class MonitoringStation:
    """Volcanic monitoring station"""

    station_id: str
    location: GeologicalLocation
    station_type: str  # seismic, GPS, gas, thermal, etc.
    instruments: List[str]
    operational_status: str  # active, maintenance, offline
    last_data_timestamp: Optional[datetime] = None
    data_quality_score: float = 1.0
    communication_status: str = "online"


@dataclass
class VolcanicHazardZone:
    """Volcanic hazard zone definition"""

    hazard_type: str  # pyroclastic_flow, ash_fall, lahar, etc.
    geometry: np.ndarray  # polygon coordinates
    risk_level: str  # low, moderate, high, extreme
    population_at_risk: int
    infrastructure_at_risk: List[str]
    evacuation_time_required: timedelta
    last_updated: datetime


@dataclass
class EmergencyResponse:
    """Emergency response configuration"""

    response_level: int  # 1-5 scale
    response_type: str  # evacuation, shelter, monitoring
    affected_areas: List[str]
    estimated_population: int
    resources_required: List[str]
    coordination_agencies: List[str]
    activation_criteria: Dict[str, Any]
    estimated_duration: timedelta


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""

    completeness: float  # 0-1 scale
    accuracy: float  # 0-1 scale
    timeliness: float  # 0-1 scale
    consistency: float  # 0-1 scale
    validity: float  # 0-1 scale
    overall_score: float  # 0-1 scale
    last_assessment: datetime
    issues_identified: List[str]


@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking"""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    last_validation: datetime
    validation_dataset: str
    performance_trend: str  # improving, stable, degrading


@dataclass
class SystemHealthMetrics:
    """System health monitoring"""

    component_name: str
    status: str  # healthy, warning, critical, offline
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    error_rate: float
    uptime: timedelta
    last_health_check: datetime


# Utility functions for data validation and conversion


def validate_alert_level(level: int) -> bool:
    """Validate alert level is within valid range"""
    return 0 <= level <= 5


def validate_probability(probability: float) -> bool:
    """Validate probability is within valid range"""
    return 0.0 <= probability <= 1.0


def validate_coordinates(latitude: float, longitude: float) -> bool:
    """Validate geographic coordinates"""
    return -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0


def calculate_distance(loc1: GeologicalLocation, loc2: GeologicalLocation) -> float:
    """Calculate distance between two geological locations in meters"""
    # Haversine formula for great circle distance
    R = 6371000  # Earth radius in meters

    lat1_rad = np.radians(loc1.latitude)
    lat2_rad = np.radians(loc2.latitude)
    delta_lat = np.radians(loc2.latitude - loc1.latitude)
    delta_lon = np.radians(loc2.longitude - loc1.longitude)

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    # Add elevation difference if available
    if loc1.elevation is not None and loc2.elevation is not None:
        elevation_diff = abs(loc2.elevation - loc1.elevation)
        distance = np.sqrt(distance**2 + elevation_diff**2)

    return distance


def convert_magnitude_to_energy(magnitude: float) -> float:
    """Convert seismic magnitude to energy in Joules"""
    # Gutenberg-Richter relation: log10(E) = 1.5*M + 9.1
    return 10 ** (1.5 * magnitude + 9.1)


def calculate_volcanic_explosivity_index(
    eruption_volume: float, column_height: float
) -> int:
    """Calculate Volcanic Explosivity Index (VEI) from eruption parameters"""
    # Simplified VEI calculation based on eruption volume
    if eruption_volume < 1e4:  # < 10,000 m³
        return 0
    elif eruption_volume < 1e6:  # < 1,000,000 m³
        return 1
    elif eruption_volume < 1e7:  # < 10,000,000 m³
        return 2
    elif eruption_volume < 1e8:  # < 100,000,000 m³
        return 3
    elif eruption_volume < 1e9:  # < 1,000,000,000 m³
        return 4
    elif eruption_volume < 1e10:  # < 10,000,000,000 m³
        return 5
    elif eruption_volume < 1e11:  # < 100,000,000,000 m³
        return 6
    elif eruption_volume < 1e12:  # < 1,000,000,000,000 m³
        return 7
    else:
        return 8


def estimate_ash_fall_area(vei: int, wind_speed: float = 10.0) -> float:
    """Estimate ash fall area in km² based on VEI and wind conditions"""
    # Simplified ash fall area estimation
    base_areas = {
        0: 1,
        1: 10,
        2: 100,
        3: 1000,
        4: 10000,
        5: 100000,
        6: 1000000,
        7: 10000000,
        8: 100000000,
    }

    base_area = base_areas.get(vei, 1000)

    # Adjust for wind speed (higher wind spreads ash further)
    wind_factor = 1 + (wind_speed - 10) * 0.1

    return base_area * wind_factor


def calculate_evacuation_time(population: int, evacuation_routes: int = 1) -> timedelta:
    """Estimate evacuation time based on population and available routes"""
    # Simplified evacuation time calculation
    # Assume 1000 people per hour per route
    evacuation_rate = 1000 * evacuation_routes  # people per hour

    hours_needed = max(1, population / evacuation_rate)

    return timedelta(hours=hours_needed)


# Data validation schemas for API endpoints

SEISMIC_EVENT_SCHEMA = {
    "type": "object",
    "required": ["timestamp", "location", "magnitude", "depth", "event_type"],
    "properties": {
        "timestamp": {"type": "string", "format": "date-time"},
        "location": {
            "type": "object",
            "required": ["latitude", "longitude", "elevation"],
            "properties": {
                "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                "longitude": {"type": "number", "minimum": -180, "maximum": 180},
                "elevation": {"type": "number"},
                "depth": {"type": "number", "minimum": 0},
            },
        },
        "magnitude": {"type": "number", "minimum": 0, "maximum": 10},
        "depth": {"type": "number", "minimum": 0},
        "event_type": {
            "type": "string",
            "enum": ["earthquake", "volcanic", "explosion", "landslide"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

ALERT_LEVEL_SCHEMA = {
    "type": "object",
    "required": [
        "level",
        "description",
        "recommended_actions",
        "validity_period",
        "confidence",
    ],
    "properties": {
        "level": {"type": "integer", "minimum": 0, "maximum": 5},
        "description": {"type": "string"},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "validity_period": {
            "type": "object",
            "required": ["start", "end"],
            "properties": {
                "start": {"type": "string", "format": "date-time"},
                "end": {"type": "string", "format": "date-time"},
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}
