"""
Metadata harmonization across different data sources
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re

from .schemas import StandardizedDataFormat, DataType, QualityMetrics


logger = logging.getLogger(__name__)


@dataclass
class MetadataMapping:
    """Mapping between source metadata and standardized metadata"""
    source_field: str
    target_field: str
    transformation: Optional[str] = None  # Function name for transformation
    required: bool = False
    default_value: Optional[Any] = None


@dataclass
class HarmonizationResult:
    """Result of metadata harmonization"""
    harmonized_metadata: Dict[str, Any]
    mapping_applied: List[MetadataMapping]
    unmapped_fields: List[str]
    quality_score: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class MetadataHarmonizer:
    """
    Harmonizes metadata across different geological data sources
    """
    
    def __init__(self):
        self.source_mappings = self._initialize_source_mappings()
        self.standard_metadata_schema = self._initialize_standard_schema()
        self.transformation_functions = self._initialize_transformations()
        self.vocabulary_mappings = self._initialize_vocabulary_mappings()
        
    def _initialize_source_mappings(self) -> Dict[str, List[MetadataMapping]]:
        """Initialize metadata mappings for different data sources"""
        return {
            "IRIS": [
                MetadataMapping("network", "network_code", required=True),
                MetadataMapping("station", "station_code", required=True),
                MetadataMapping("channel", "channel_code", required=True),
                MetadataMapping("location", "location_code"),
                MetadataMapping("starttime", "start_time", "parse_datetime"),
                MetadataMapping("endtime", "end_time", "parse_datetime"),
                MetadataMapping("sampling_rate", "sampling_rate", "float"),
                MetadataMapping("delta", "sample_interval", "float"),
                MetadataMapping("npts", "sample_count", "int"),
                MetadataMapping("calib", "calibration_factor", "float"),
                MetadataMapping("response", "instrument_response"),
            ],
            
            "USGS": [
                MetadataMapping("eventid", "event_id", required=True),
                MetadataMapping("time", "event_time", "parse_datetime", required=True),
                MetadataMapping("latitude", "latitude", "float", required=True),
                MetadataMapping("longitude", "longitude", "float", required=True),
                MetadataMapping("depth", "depth_km", "float"),
                MetadataMapping("mag", "magnitude", "float"),
                MetadataMapping("magType", "magnitude_type"),
                MetadataMapping("place", "location_description"),
                MetadataMapping("type", "event_type"),
                MetadataMapping("status", "review_status"),
                MetadataMapping("updated", "last_updated", "parse_datetime"),
            ],
            
            "NASA_EarthData": [
                MetadataMapping("granule_id", "granule_id", required=True),
                MetadataMapping("collection", "collection_name", required=True),
                MetadataMapping("begin_date", "start_time", "parse_datetime"),
                MetadataMapping("end_date", "end_time", "parse_datetime"),
                MetadataMapping("bounding_box", "spatial_bounds", "parse_bbox"),
                MetadataMapping("cloud_cover", "cloud_coverage", "float"),
                MetadataMapping("instrument", "instrument_name"),
                MetadataMapping("platform", "platform_name"),
                MetadataMapping("processing_level", "processing_level"),
                MetadataMapping("format", "data_format"),
            ],
            
            "Copernicus": [
                MetadataMapping("product_id", "product_id", required=True),
                MetadataMapping("product_type", "product_type", required=True),
                MetadataMapping("sensing_start", "acquisition_start", "parse_datetime"),
                MetadataMapping("sensing_end", "acquisition_end", "parse_datetime"),
                MetadataMapping("orbit_number", "orbit_number", "int"),
                MetadataMapping("orbit_direction", "orbit_direction"),
                MetadataMapping("polarisation", "polarization"),
                MetadataMapping("instrument_mode", "instrument_mode"),
                MetadataMapping("resolution", "spatial_resolution", "float"),
                MetadataMapping("cloud_coverage", "cloud_coverage", "float"),
            ],
            
            "ASF_Hyp3": [
                MetadataMapping("job_id", "job_id", required=True),
                MetadataMapping("job_type", "processing_type", required=True),
                MetadataMapping("reference_granule", "reference_scene"),
                MetadataMapping("secondary_granule", "secondary_scene"),
                MetadataMapping("temporal_baseline", "temporal_baseline_days", "int"),
                MetadataMapping("perpendicular_baseline", "perpendicular_baseline_m", "float"),
                MetadataMapping("looks", "number_of_looks", "int"),
                MetadataMapping("dem_name", "dem_source"),
                MetadataMapping("processing_date", "processing_date", "parse_datetime"),
                MetadataMapping("pixel_spacing", "pixel_spacing_m", "float"),
            ],
            
            "EarthChem": [
                MetadataMapping("sample_id", "sample_id", required=True),
                MetadataMapping("material", "sample_type", required=True),
                MetadataMapping("latitude", "latitude", "float", required=True),
                MetadataMapping("longitude", "longitude", "float", required=True),
                MetadataMapping("elevation", "elevation_m", "float"),
                MetadataMapping("age", "age_ma", "float"),
                MetadataMapping("age_min", "age_min_ma", "float"),
                MetadataMapping("age_max", "age_max_ma", "float"),
                MetadataMapping("rock_class", "rock_classification"),
                MetadataMapping("tectonic_setting", "tectonic_setting"),
                MetadataMapping("method", "analytical_method"),
            ],
            
            "GVP": [  # Global Volcanism Program
                MetadataMapping("volcano_number", "volcano_id", required=True),
                MetadataMapping("volcano_name", "volcano_name", required=True),
                MetadataMapping("latitude", "latitude", "float", required=True),
                MetadataMapping("longitude", "longitude", "float", required=True),
                MetadataMapping("elevation", "elevation_m", "float"),
                MetadataMapping("type", "volcano_type"),
                MetadataMapping("status", "activity_status"),
                MetadataMapping("last_eruption", "last_eruption_year", "int"),
                MetadataMapping("region", "volcanic_region"),
                MetadataMapping("subregion", "volcanic_subregion"),
                MetadataMapping("country", "country"),
            ],
        }
    
    def _initialize_standard_schema(self) -> Dict[str, Dict[str, Any]]:
        """Initialize standard metadata schema"""
        return {
            "identification": {
                "data_id": {"type": "string", "required": True},
                "source_id": {"type": "string", "required": True},
                "collection_id": {"type": "string"},
                "version": {"type": "string"},
                "doi": {"type": "string"},
            },
            
            "temporal": {
                "acquisition_time": {"type": "datetime"},
                "start_time": {"type": "datetime"},
                "end_time": {"type": "datetime"},
                "processing_time": {"type": "datetime"},
                "temporal_resolution": {"type": "string"},
                "temporal_coverage": {"type": "object"},
            },
            
            "spatial": {
                "latitude": {"type": "float", "range": [-90, 90]},
                "longitude": {"type": "float", "range": [-180, 180]},
                "elevation": {"type": "float"},
                "depth": {"type": "float"},
                "spatial_resolution": {"type": "float"},
                "coordinate_system": {"type": "string"},
                "spatial_bounds": {"type": "object"},
            },
            
            "instrument": {
                "platform": {"type": "string"},
                "instrument": {"type": "string"},
                "sensor": {"type": "string"},
                "instrument_mode": {"type": "string"},
                "calibration_date": {"type": "datetime"},
                "instrument_response": {"type": "object"},
            },
            
            "processing": {
                "processing_level": {"type": "string"},
                "processing_date": {"type": "datetime"},
                "processing_software": {"type": "string"},
                "processing_version": {"type": "string"},
                "processing_parameters": {"type": "object"},
                "algorithm": {"type": "string"},
            },
            
            "quality": {
                "quality_flag": {"type": "string"},
                "quality_score": {"type": "float", "range": [0, 1]},
                "completeness": {"type": "float", "range": [0, 1]},
                "accuracy": {"type": "float", "range": [0, 1]},
                "uncertainty": {"type": "object"},
                "validation_status": {"type": "string"},
            },
            
            "geological": {
                "geological_age": {"type": "float"},
                "rock_type": {"type": "string"},
                "mineral_composition": {"type": "object"},
                "tectonic_setting": {"type": "string"},
                "volcanic_system": {"type": "string"},
                "eruption_style": {"type": "string"},
            },
            
            "access": {
                "access_url": {"type": "string"},
                "download_url": {"type": "string"},
                "access_protocol": {"type": "string"},
                "access_constraints": {"type": "string"},
                "usage_rights": {"type": "string"},
                "citation": {"type": "string"},
            },
        }
    
    def _initialize_transformations(self) -> Dict[str, callable]:
        """Initialize transformation functions"""
        return {
            "parse_datetime": self._parse_datetime,
            "parse_bbox": self._parse_bounding_box,
            "float": self._safe_float,
            "int": self._safe_int,
            "normalize_string": self._normalize_string,
            "parse_coordinate": self._parse_coordinate,
            "standardize_units": self._standardize_units,
        }
    
    def _initialize_vocabulary_mappings(self) -> Dict[str, Dict[str, str]]:
        """Initialize controlled vocabulary mappings"""
        return {
            "volcano_type": {
                "stratovolcano": "stratovolcano",
                "shield": "shield_volcano",
                "cinder cone": "cinder_cone",
                "caldera": "caldera",
                "complex": "complex_volcano",
                "fissure vent": "fissure_vent",
                "lava dome": "lava_dome",
                "maar": "maar",
                "pyroclastic cone": "pyroclastic_cone",
                "submarine": "submarine_volcano",
            },
            
            "tectonic_setting": {
                "subduction zone": "subduction_zone",
                "mid-ocean ridge": "mid_ocean_ridge",
                "hotspot": "hotspot",
                "rift zone": "continental_rift",
                "back-arc": "back_arc",
                "intraplate": "intraplate",
                "transform fault": "transform_fault",
            },
            
            "rock_type": {
                "basalt": "basalt",
                "andesite": "andesite",
                "dacite": "dacite",
                "rhyolite": "rhyolite",
                "trachyte": "trachyte",
                "phonolite": "phonolite",
                "basaltic andesite": "basaltic_andesite",
                "rhyodacite": "rhyodacite",
            },
            
            "processing_level": {
                "L0": "raw",
                "L1A": "reconstructed_unprocessed",
                "L1B": "reconstructed_calibrated",
                "L2": "derived_geophysical",
                "L3": "derived_gridded",
                "L4": "model_output",
                "raw": "raw",
                "processed": "processed",
                "calibrated": "calibrated",
            },
        }
    
    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse various datetime formats"""
        if isinstance(value, datetime):
            return value
        
        if not isinstance(value, str):
            return None
        
        # Common datetime formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse datetime: {value}")
        return None
    
    def _parse_bounding_box(self, value: Any) -> Optional[Dict[str, float]]:
        """Parse bounding box from various formats"""
        if isinstance(value, dict):
            return value
        
        if isinstance(value, str):
            # Try to parse comma-separated values: "minlon,minlat,maxlon,maxlat"
            try:
                coords = [float(x.strip()) for x in value.split(',')]
                if len(coords) == 4:
                    return {
                        "min_longitude": coords[0],
                        "min_latitude": coords[1],
                        "max_longitude": coords[2],
                        "max_latitude": coords[3]
                    }
            except ValueError:
                pass
        
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                return {
                    "min_longitude": float(value[0]),
                    "min_latitude": float(value[1]),
                    "max_longitude": float(value[2]),
                    "max_latitude": float(value[3])
                }
            except (ValueError, TypeError):
                pass
        
        logger.warning(f"Could not parse bounding box: {value}")
        return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to int"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def _normalize_string(self, value: Any) -> Optional[str]:
        """Normalize string value"""
        if not isinstance(value, str):
            return str(value) if value is not None else None
        
        # Remove extra whitespace and normalize case
        normalized = re.sub(r'\s+', ' ', value.strip())
        return normalized.lower() if normalized else None
    
    def _parse_coordinate(self, value: Any) -> Optional[float]:
        """Parse coordinate value"""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Handle degree/minute/second format
            if '°' in value or "'" in value or '"' in value:
                return self._parse_dms_coordinate(value)
            
            # Try direct float conversion
            try:
                return float(value)
            except ValueError:
                pass
        
        return None
    
    def _parse_dms_coordinate(self, dms_str: str) -> Optional[float]:
        """Parse degree/minute/second coordinate format"""
        try:
            # Remove extra spaces and normalize
            dms_str = re.sub(r'\s+', ' ', dms_str.strip())
            
            # Extract degrees, minutes, seconds
            pattern = r"(\d+)°?\s*(\d+)?'?\s*(\d+(?:\.\d+)?)?\"?\s*([NSEW])?"
            match = re.match(pattern, dms_str, re.IGNORECASE)
            
            if not match:
                return None
            
            degrees = float(match.group(1))
            minutes = float(match.group(2)) if match.group(2) else 0
            seconds = float(match.group(3)) if match.group(3) else 0
            direction = match.group(4).upper() if match.group(4) else None
            
            # Convert to decimal degrees
            decimal = degrees + minutes/60 + seconds/3600
            
            # Apply direction
            if direction in ['S', 'W']:
                decimal = -decimal
            
            return decimal
            
        except Exception as e:
            logger.warning(f"Could not parse DMS coordinate {dms_str}: {e}")
            return None
    
    def _standardize_units(self, value: Any) -> Optional[str]:
        """Standardize unit strings"""
        if not isinstance(value, str):
            return str(value) if value is not None else None
        
        # Common unit standardizations
        unit_mappings = {
            "meters": "m",
            "metres": "m",
            "kilometers": "km",
            "kilometres": "km",
            "degrees": "deg",
            "celsius": "C",
            "kelvin": "K",
            "pascals": "Pa",
        }
        
        normalized = value.lower().strip()
        return unit_mappings.get(normalized, value)
    
    def harmonize_metadata(
        self,
        source_metadata: Dict[str, Any],
        source_name: str,
        data_type: Optional[DataType] = None
    ) -> HarmonizationResult:
        """
        Harmonize metadata from a specific source
        
        Args:
            source_metadata: Original metadata from source
            source_name: Name of the data source
            data_type: Type of geological data
            
        Returns:
            Harmonization result with standardized metadata
        """
        harmonized = {}
        applied_mappings = []
        unmapped_fields = []
        warnings = []
        errors = []
        
        # Get mappings for this source
        mappings = self.source_mappings.get(source_name, [])
        
        # Apply mappings
        for mapping in mappings:
            if mapping.source_field in source_metadata:
                source_value = source_metadata[mapping.source_field]
                
                # Apply transformation if specified
                if mapping.transformation and mapping.transformation in self.transformation_functions:
                    try:
                        transformed_value = self.transformation_functions[mapping.transformation](source_value)
                        if transformed_value is not None:
                            harmonized[mapping.target_field] = transformed_value
                            applied_mappings.append(mapping)
                        elif mapping.required:
                            errors.append(f"Required field {mapping.target_field} transformation failed")
                    except Exception as e:
                        error_msg = f"Transformation failed for {mapping.source_field}: {e}"
                        if mapping.required:
                            errors.append(error_msg)
                        else:
                            warnings.append(error_msg)
                else:
                    # Direct mapping
                    harmonized[mapping.target_field] = source_value
                    applied_mappings.append(mapping)
            
            elif mapping.required:
                if mapping.default_value is not None:
                    harmonized[mapping.target_field] = mapping.default_value
                    warnings.append(f"Using default value for required field {mapping.target_field}")
                else:
                    errors.append(f"Required field {mapping.source_field} not found in source metadata")
        
        # Identify unmapped fields
        mapped_source_fields = {m.source_field for m in applied_mappings}
        unmapped_fields = [field for field in source_metadata.keys() 
                          if field not in mapped_source_fields]
        
        # Apply vocabulary standardization
        harmonized = self._apply_vocabulary_standardization(harmonized, warnings)
        
        # Add source information
        harmonized['source_name'] = source_name
        harmonized['original_metadata_hash'] = self._calculate_metadata_hash(source_metadata)
        harmonized['harmonization_timestamp'] = datetime.now().isoformat()
        
        # Calculate quality score
        quality_score = self._calculate_harmonization_quality(
            source_metadata, harmonized, applied_mappings, errors, warnings
        )
        
        return HarmonizationResult(
            harmonized_metadata=harmonized,
            mapping_applied=applied_mappings,
            unmapped_fields=unmapped_fields,
            quality_score=quality_score,
            warnings=warnings,
            errors=errors
        )
    
    def _apply_vocabulary_standardization(
        self,
        metadata: Dict[str, Any],
        warnings: List[str]
    ) -> Dict[str, Any]:
        """Apply controlled vocabulary standardization"""
        standardized = metadata.copy()
        
        for field, value in metadata.items():
            if isinstance(value, str):
                # Check if field has vocabulary mapping
                for vocab_type, vocab_mapping in self.vocabulary_mappings.items():
                    if vocab_type in field.lower():
                        normalized_value = value.lower().strip()
                        if normalized_value in vocab_mapping:
                            standardized[field] = vocab_mapping[normalized_value]
                        else:
                            # Try partial matching
                            for key, standard_value in vocab_mapping.items():
                                if key in normalized_value or normalized_value in key:
                                    standardized[field] = standard_value
                                    warnings.append(f"Partial vocabulary match for {field}: {value} -> {standard_value}")
                                    break
        
        return standardized
    
    def _calculate_metadata_hash(self, metadata: Dict[str, Any]) -> str:
        """Calculate hash of metadata for change detection"""
        metadata_str = json.dumps(metadata, sort_keys=True, default=str)
        return hashlib.md5(metadata_str.encode()).hexdigest()
    
    def _calculate_harmonization_quality(
        self,
        source_metadata: Dict[str, Any],
        harmonized_metadata: Dict[str, Any],
        applied_mappings: List[MetadataMapping],
        errors: List[str],
        warnings: List[str]
    ) -> float:
        """Calculate quality score for harmonization"""
        if not source_metadata:
            return 0.0
        
        # Base score on successful mappings
        total_fields = len(source_metadata)
        mapped_fields = len(applied_mappings)
        mapping_score = mapped_fields / total_fields if total_fields > 0 else 0.0
        
        # Penalize errors and warnings
        error_penalty = len(errors) * 0.1
        warning_penalty = len(warnings) * 0.05
        
        # Bonus for required fields
        required_mappings = [m for m in applied_mappings if m.required]
        required_bonus = len(required_mappings) * 0.1
        
        quality_score = mapping_score + required_bonus - error_penalty - warning_penalty
        
        return max(0.0, min(1.0, quality_score))
    
    def harmonize_data_format(
        self,
        data: StandardizedDataFormat,
        source_name: Optional[str] = None
    ) -> StandardizedDataFormat:
        """
        Harmonize metadata in standardized data format
        
        Args:
            data: Data with metadata to harmonize
            source_name: Source name (uses data.source if None)
            
        Returns:
            Data with harmonized metadata
        """
        if source_name is None:
            source_name = data.source
        
        # Harmonize metadata
        harmonization_result = self.harmonize_metadata(
            data.metadata, source_name, data.data_type
        )
        
        # Create new data with harmonized metadata
        harmonized_data = data.copy(deep=True)
        harmonized_data.metadata = harmonization_result.harmonized_metadata
        
        # Update quality metrics
        if harmonized_data.quality_metrics is None:
            harmonized_data.quality_metrics = QualityMetrics(
                completeness=0.0,
                accuracy=0.0,
                consistency=0.0,
                timeliness=0.0,
                overall_score=0.0
            )
        
        # Update consistency score based on harmonization quality
        harmonized_data.quality_metrics.consistency = harmonization_result.quality_score
        
        # Add quality flags for warnings and errors
        for warning in harmonization_result.warnings:
            harmonized_data.quality_flags.append(f"metadata_warning: {warning}")
        
        for error in harmonization_result.errors:
            harmonized_data.quality_flags.append(f"metadata_error: {error}")
        
        # Add processing history
        processing_step = {
            'step_id': f"metadata_harmonization_{datetime.now().isoformat()}",
            'timestamp': datetime.now(),
            'operation': 'metadata_harmonization',
            'parameters': {
                'source_name': source_name,
                'mappings_applied': len(harmonization_result.mapping_applied),
                'unmapped_fields': len(harmonization_result.unmapped_fields),
                'quality_score': harmonization_result.quality_score
            },
            'input_hash': str(hash(str(data.metadata))),
            'output_hash': str(hash(str(harmonized_data.metadata))),
            'processor_version': '1.0.0'
        }
        
        harmonized_data.processing_history.append(processing_step)
        
        return harmonized_data
    
    def batch_harmonize_metadata(
        self,
        data_list: List[StandardizedDataFormat]
    ) -> List[StandardizedDataFormat]:
        """
        Harmonize metadata for a batch of data records
        
        Args:
            data_list: List of data to harmonize
            
        Returns:
            List of data with harmonized metadata
        """
        harmonized_data = []
        
        for data in data_list:
            try:
                harmonized = self.harmonize_data_format(data)
                harmonized_data.append(harmonized)
            except Exception as e:
                logger.error(f"Failed to harmonize metadata for data {data.data_id}: {e}")
                # Keep original data with error flag
                data_copy = data.copy(deep=True)
                data_copy.quality_flags.append(f"metadata_harmonization_failed: {str(e)}")
                harmonized_data.append(data_copy)
        
        return harmonized_data