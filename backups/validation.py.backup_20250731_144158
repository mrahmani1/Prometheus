"""
Data Validation Module

Provides comprehensive validation for geological data with scientific constraints
and quality control measures.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np
import warnings

from .geological import (
    GeologicalLocation, SeismicEvent, DeformationMeasurement,
    GeochemicalSample, ThermalMeasurement, MagmaReservoir
)


class ValidationLevel(str, Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationCategory(str, Enum):
    """Categories of validation checks"""
    SCIENTIFIC_CONSTRAINTS = "scientific_constraints"
    DATA_QUALITY = "data_quality"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in data"""
    level: ValidationLevel
    category: ValidationCategory
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int = 0
    errors_count: int = 0
    
    def __post_init__(self):
        """Count issues by level"""
        self.warnings_count = sum(1 for issue in self.issues if issue.level == ValidationLevel.WARNING)
        self.errors_count = sum(1 for issue in self.issues if issue.level == ValidationLevel.ERROR)
        self.is_valid = self.errors_count == 0


class ValidationError(Exception):
    """Raised when validation fails with errors"""
    def __init__(self, message: str, validation_result: ValidationResult):
        super().__init__(message)
        self.validation_result = validation_result


class GeologicalDataValidator:
    """
    Comprehensive validator for geological data with scientific constraints
    """
    
    def __init__(self):
        """Initialize validator with scientific constraints"""
        # Define scientific constraints
        self.MAGNITUDE_RANGES = {
            "earthquake": (-2.0, 10.0),
            "volcanic": (-1.0, 8.0),
            "explosion": (-1.0, 6.0),
            "landslide": (-1.0, 5.0)
        }
        
        self.DEPTH_RANGES = {
            "earthquake": (0, 800),  # km
            "volcanic": (0, 50),     # km
            "explosion": (0, 10),    # km
        }
        
        self.COMMON_ELEMENTS = {
            # Major elements (typical ranges in wt%)
            "SiO2": (35, 80),
            "Al2O3": (8, 25),
            "Fe2O3": (0, 15),
            "FeO": (0, 20),
            "MgO": (0, 50),
            "CaO": (0, 20),
            "Na2O": (0, 10),
            "K2O": (0, 15),
            "TiO2": (0, 5),
            "P2O5": (0, 2),
            "MnO": (0, 1),
            
            # Trace elements (typical ranges in ppm)
            "Rb": (0, 500),
            "Sr": (0, 2000),
            "Ba": (0, 3000),
            "Zr": (0, 1000),
            "Nb": (0, 200),
            "Y": (0, 100),
            "Cr": (0, 3000),
            "Ni": (0, 2000),
            "V": (0, 500),
            "Cu": (0, 1000),
            "Zn": (0, 500),
            "Pb": (0, 100)
        }
        
        self.VOLATILE_RANGES = {
            "H2O": (0, 10),    # wt%
            "CO2": (0, 5),     # wt%
            "SO2": (0, 2),     # wt%
            "H2S": (0, 1),     # wt%
            "HCl": (0, 0.5),   # wt%
            "HF": (0, 0.1)     # wt%
        }
    
    def validate_geological_location(self, location: GeologicalLocation) -> ValidationResult:
        """Validate geological location data"""
        issues = []
        
        # Check coordinate ranges
        if not -90 <= location.latitude <= 90:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                message=f"Latitude {location.latitude} outside valid range [-90, 90]",
                field="latitude",
                value=location.latitude
            ))
        
        if not -180 <= location.longitude <= 180:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                message=f"Longitude {location.longitude} outside valid range [-180, 180]",
                field="longitude",
                value=location.longitude
            ))
        
        # Check elevation reasonableness
        if location.elevation < -11000:  # Mariana Trench depth
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.DATA_QUALITY,
                message=f"Elevation {location.elevation}m is below deepest ocean point",
                field="elevation",
                value=location.elevation
            ))
        
        if location.elevation > 9000:  # Above Everest
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.DATA_QUALITY,
                message=f"Elevation {location.elevation}m is above highest mountain",
                field="elevation",
                value=location.elevation
            ))
        
        # Check depth consistency
        if location.depth is not None:
            if location.depth < 0:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                    message="Depth must be positive (downward direction)",
                    field="depth",
                    value=location.depth
                ))
            
            if location.depth > 800000:  # 800 km - deepest earthquakes
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.DATA_QUALITY,
                    message=f"Depth {location.depth}m exceeds typical geological processes",
                    field="depth",
                    value=location.depth
                ))
        
        return ValidationResult(is_valid=len([i for i in issues if i.level == ValidationLevel.ERROR]) == 0, issues=issues)
    
    def validate_seismic_event(self, event: SeismicEvent) -> ValidationResult:
        """Validate seismic event data with scientific constraints"""
        issues = []
        
        # Validate location first
        location_result = self.validate_geological_location(event.location)
        issues.extend(location_result.issues)
        
        # Check magnitude ranges by event type
        event_type_str = event.event_type.value
        if event_type_str in self.MAGNITUDE_RANGES:
            min_mag, max_mag = self.MAGNITUDE_RANGES[event_type_str]
            if not min_mag <= event.magnitude <= max_mag:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                    message=f"Magnitude {event.magnitude} unusual for {event_type_str} events (typical range: {min_mag}-{max_mag})",
                    field="magnitude",
                    value=event.magnitude
                ))
        
        # Check depth ranges by event type
        if event_type_str in self.DEPTH_RANGES:
            min_depth, max_depth = self.DEPTH_RANGES[event_type_str]
            if not min_depth <= event.depth <= max_depth:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                    message=f"Depth {event.depth}km unusual for {event_type_str} events (typical range: {min_depth}-{max_depth}km)",
                    field="depth",
                    value=event.depth
                ))
        
        # Check confidence range
        if not 0 <= event.confidence <= 1:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.DATA_QUALITY,
                message=f"Confidence {event.confidence} must be between 0 and 1",
                field="confidence",
                value=event.confidence
            ))
        
        # Check temporal reasonableness
        now = datetime.utcnow()
        if event.timestamp > now + timedelta(hours=1):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.TEMPORAL,
                message="Event timestamp is in the future",
                field="timestamp",
                value=event.timestamp
            ))
        
        # Check waveform data if present
        if event.waveform_data is not None:
            if len(event.waveform_data) == 0:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.DATA_QUALITY,
                    message="Waveform data is empty",
                    field="waveform_data"
                ))
            
            # Check for reasonable amplitude ranges
            if np.any(np.abs(event.waveform_data) > 1e6):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.DATA_QUALITY,
                    message="Waveform amplitudes seem unusually large",
                    field="waveform_data"
                ))
        
        return ValidationResult(is_valid=len([i for i in issues if i.level == ValidationLevel.ERROR]) == 0, issues=issues)
    
    def validate_deformation_measurement(self, measurement: DeformationMeasurement) -> ValidationResult:
        """Validate deformation measurement data"""
        issues = []
        
        # Validate location
        location_result = self.validate_geological_location(measurement.location)
        issues.extend(location_result.issues)
        
        # Check displacement reasonableness
        displacement_magnitude = np.linalg.norm(measurement.displacement_vector)
        if displacement_magnitude > 10:  # 10 meters
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.DATA_QUALITY,
                message=f"Large displacement detected: {displacement_magnitude:.2f}m",
                field="displacement_vector",
                value=displacement_magnitude,
                suggestion="Verify measurement accuracy and units"
            ))
        
        # Check uncertainty values
        if np.any(measurement.uncertainty <= 0):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.DATA_QUALITY,
                message="Uncertainty values must be positive",
                field="uncertainty"
            ))
        
        # Check if uncertainty is reasonable compared to displacement
        for i, (disp, unc) in enumerate(zip(measurement.displacement_vector, measurement.uncertainty)):
            if abs(disp) > 0 and unc > abs(disp):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.DATA_QUALITY,
                    message=f"Uncertainty in dimension {i} exceeds displacement magnitude",
                    field="uncertainty",
                    suggestion="Check measurement precision"
                ))
        
        # Check velocity if present
        if measurement.velocity_vector is not None:
            velocity_magnitude = np.linalg.norm(measurement.velocity_vector)
            if velocity_magnitude > 1:  # 1 m/year is very fast for geological processes
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                    message=f"Very high velocity detected: {velocity_magnitude:.3f}m/year",
                    field="velocity_vector",
                    value=velocity_magnitude
                ))
        
        return ValidationResult(is_valid=len([i for i in issues if i.level == ValidationLevel.ERROR]) == 0, issues=issues)
    
    def validate_geochemical_sample(self, sample: GeochemicalSample) -> ValidationResult:
        """Validate geochemical sample data with scientific constraints"""
        issues = []
        
        # Validate location
        location_result = self.validate_geological_location(sample.location)
        issues.extend(location_result.issues)
        
        # Check composition values
        total_major_elements = 0
        for element, concentration in sample.composition.items():
            if concentration < 0:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                    message=f"Negative concentration for {element}: {concentration}",
                    field="composition",
                    value=concentration
                ))
            
            # Check against typical ranges
            if element in self.COMMON_ELEMENTS:
                min_val, max_val = self.COMMON_ELEMENTS[element]
                if not min_val <= concentration <= max_val:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                        message=f"{element} concentration {concentration} outside typical range [{min_val}, {max_val}]",
                        field="composition",
                        value=concentration
                    ))
            
            # Sum major elements for closure check
            if element in ["SiO2", "Al2O3", "Fe2O3", "FeO", "MgO", "CaO", "Na2O", "K2O", "TiO2", "P2O5", "MnO"]:
                total_major_elements += concentration
        
        # Check major element closure (should sum to ~100% for whole rock analysis)
        if sample.sample_type.value == "rock" and 80 <= total_major_elements <= 120:
            if not 95 <= total_major_elements <= 105:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.DATA_QUALITY,
                    message=f"Major elements sum to {total_major_elements:.1f}% (expected ~100%)",
                    field="composition",
                    value=total_major_elements,
                    suggestion="Check analytical totals and loss on ignition"
                ))
        
        # Validate temperature
        if sample.temperature is not None:
            if sample.sample_type.value == "lava" and not 800 <= sample.temperature <= 1400:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                    message=f"Lava temperature {sample.temperature}°C outside typical range [800-1400°C]",
                    field="temperature",
                    value=sample.temperature
                ))
        
        # Validate pH for appropriate sample types
        if sample.ph is not None:
            if sample.sample_type.value in ["water", "gas"] and not 0 <= sample.ph <= 14:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                    message=f"pH {sample.ph} outside valid range [0-14]",
                    field="ph",
                    value=sample.ph
                ))
        
        return ValidationResult(is_valid=len([i for i in issues if i.level == ValidationLevel.ERROR]) == 0, issues=issues)
    
    def validate_thermal_measurement(self, measurement: ThermalMeasurement) -> ValidationResult:
        """Validate thermal measurement data"""
        issues = []
        
        # Validate location
        location_result = self.validate_geological_location(measurement.location)
        issues.extend(location_result.issues)
        
        # Check temperature reasonableness
        if measurement.temperature < -100:  # Very cold
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.DATA_QUALITY,
                message=f"Very low temperature: {measurement.temperature}°C",
                field="temperature",
                value=measurement.temperature
            ))
        
        if measurement.temperature > 1500:  # Very hot
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.DATA_QUALITY,
                message=f"Very high temperature: {measurement.temperature}°C",
                field="temperature",
                value=measurement.temperature
            ))
        
        # Check uncertainty relative to measurement
        if measurement.uncertainty > abs(measurement.temperature) * 0.5:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.DATA_QUALITY,
                message="Temperature uncertainty is very large relative to measurement",
                field="uncertainty",
                suggestion="Check instrument calibration"
            ))
        
        # Check spatial resolution reasonableness
        if measurement.spatial_resolution > 10000:  # 10 km
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                category=ValidationCategory.DATA_QUALITY,
                message=f"Low spatial resolution: {measurement.spatial_resolution}m",
                field="spatial_resolution"
            ))
        
        return ValidationResult(is_valid=len([i for i in issues if i.level == ValidationLevel.ERROR]) == 0, issues=issues)
    
    def validate_magma_reservoir(self, reservoir: MagmaReservoir) -> ValidationResult:
        """Validate magma reservoir data with petrological constraints"""
        issues = []
        
        # Validate location
        location_result = self.validate_geological_location(reservoir.location)
        issues.extend(location_result.issues)
        
        # Validate composition
        composition_result = self.validate_geochemical_sample(reservoir.composition)
        issues.extend(composition_result.issues)
        
        # Check temperature-crystallinity relationship
        if reservoir.temperature < 800 and reservoir.crystallinity < 0.8:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                message=f"Low temperature ({reservoir.temperature}°C) with low crystallinity ({reservoir.crystallinity}) is unusual",
                field="temperature",
                suggestion="Check temperature-crystallinity relationship"
            ))
        
        # Check volatile content
        total_volatiles = sum(reservoir.volatile_content.values())
        if total_volatiles > 15:  # 15 wt% is very high
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                message=f"Very high total volatile content: {total_volatiles:.1f}wt%",
                field="volatile_content",
                value=total_volatiles
            ))
        
        # Check individual volatile ranges
        for volatile, concentration in reservoir.volatile_content.items():
            if volatile in self.VOLATILE_RANGES:
                min_val, max_val = self.VOLATILE_RANGES[volatile]
                if concentration > max_val:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category=ValidationCategory.SCIENTIFIC_CONSTRAINTS,
                        message=f"{volatile} concentration {concentration}wt% exceeds typical maximum {max_val}wt%",
                        field="volatile_content"
                    ))
        
        # Check volume reasonableness
        if reservoir.volume > 1e12:  # 1000 km³
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.DATA_QUALITY,
                message=f"Very large reservoir volume: {reservoir.volume/1e9:.1f} km³",
                field="volume",
                suggestion="Verify volume calculation"
            ))
        
        return ValidationResult(is_valid=len([i for i in issues if i.level == ValidationLevel.ERROR]) == 0, issues=issues)
    
    def validate_data_consistency(self, data_list: List[Union[SeismicEvent, DeformationMeasurement, 
                                                           GeochemicalSample, ThermalMeasurement]]) -> ValidationResult:
        """Validate consistency across multiple data points"""
        issues = []
        
        if len(data_list) < 2:
            return ValidationResult(is_valid=True, issues=[])
        
        # Check temporal consistency
        timestamps = [getattr(data, 'timestamp', None) for data in data_list if hasattr(data, 'timestamp')]
        if timestamps:
            timestamps.sort()
            for i in range(1, len(timestamps)):
                time_diff = timestamps[i] - timestamps[i-1]
                if time_diff.total_seconds() < 0:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category=ValidationCategory.TEMPORAL,
                        message="Timestamps are not in chronological order",
                        field="timestamp"
                    ))
                    break
        
        # Check spatial consistency
        locations = [getattr(data, 'location', None) for data in data_list if hasattr(data, 'location')]
        if len(locations) > 1:
            # Check for duplicate locations
            for i in range(len(locations)):
                for j in range(i+1, len(locations)):
                    if (abs(locations[i].latitude - locations[j].latitude) < 1e-6 and
                        abs(locations[i].longitude - locations[j].longitude) < 1e-6):
                        issues.append(ValidationIssue(
                            level=ValidationLevel.INFO,
                            category=ValidationCategory.SPATIAL,
                            message="Duplicate or very close locations detected",
                            field="location"
                        ))
        
        return ValidationResult(is_valid=len([i for i in issues if i.level == ValidationLevel.ERROR]) == 0, issues=issues)
    
    def validate_and_raise(self, data: Any) -> ValidationResult:
        """Validate data and raise exception if errors found"""
        if isinstance(data, GeologicalLocation):
            result = self.validate_geological_location(data)
        elif isinstance(data, SeismicEvent):
            result = self.validate_seismic_event(data)
        elif isinstance(data, DeformationMeasurement):
            result = self.validate_deformation_measurement(data)
        elif isinstance(data, GeochemicalSample):
            result = self.validate_geochemical_sample(data)
        elif isinstance(data, ThermalMeasurement):
            result = self.validate_thermal_measurement(data)
        elif isinstance(data, MagmaReservoir):
            result = self.validate_magma_reservoir(data)
        else:
            raise ValueError(f"Unsupported data type for validation: {type(data)}")
        
        if not result.is_valid:
            error_messages = [issue.message for issue in result.issues if issue.level == ValidationLevel.ERROR]
            raise ValidationError(f"Validation failed: {'; '.join(error_messages)}", result)
        
        # Issue warnings for warning-level issues
        for issue in result.issues:
            if issue.level == ValidationLevel.WARNING:
                warnings.warn(f"Data quality warning: {issue.message}")
        
        return result