"""
Main data transformation and standardization pipeline
"""

import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time

from .schemas import (
    StandardizedDataFormat, TransformationConfig, ValidationResult,
    TemporalResolution, CoordinateSystem, DataType
)
from .coordinate_transformer import CoordinateTransformer
from .temporal_synchronizer import TemporalSynchronizer
from .unit_converter import UnitConverter
from .metadata_harmonizer import MetadataHarmonizer
from .interpolation_engine import InterpolationEngine
from .aggregation_engine import AggregationEngine, AggregationConfig
from .export_manager import ExportManager, ExportConfig


logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of complete pipeline execution"""
    success: bool
    processed_data: List[StandardizedDataFormat]
    original_count: int
    processed_count: int
    processing_time_seconds: float
    transformations_applied: List[str]
    quality_improvement: float
    validation_results: ValidationResult
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineStats:
    """Statistics from pipeline execution"""
    coordinate_transformations: int = 0
    temporal_synchronizations: int = 0
    unit_conversions: int = 0
    metadata_harmonizations: int = 0
    interpolated_points: int = 0
    aggregated_records: int = 0
    quality_flags_added: int = 0
    processing_steps_total: int = 0


class DataTransformationPipeline:
    """
    Main pipeline for geological data transformation and standardization
    """
    
    def __init__(self, config: Optional[TransformationConfig] = None):
        self.config = config or TransformationConfig()
        
        # Initialize transformation components
        self.coordinate_transformer = CoordinateTransformer()
        self.temporal_synchronizer = TemporalSynchronizer()
        self.unit_converter = UnitConverter()
        self.metadata_harmonizer = MetadataHarmonizer()
        self.interpolation_engine = InterpolationEngine()
        self.aggregation_engine = AggregationEngine()
        self.export_manager = ExportManager()
        
        # Pipeline statistics
        self.stats = PipelineStats()
        
    async def process_data(
        self,
        raw_data: List[Dict[str, Any]],
        source_name: str,
        data_type: DataType,
        custom_config: Optional[TransformationConfig] = None
    ) -> PipelineResult:
        """
        Process raw geological data through complete transformation pipeline
        
        Args:
            raw_data: Raw data from geological sources
            source_name: Name of data source
            data_type: Type of geological data
            custom_config: Custom configuration (overrides default)
            
        Returns:
            Complete pipeline processing result
        """
        start_time = time.time()
        config = custom_config or self.config
        warnings = []
        errors = []
        transformations_applied = []
        
        try:
            # Step 1: Convert to standardized format
            logger.info(f"Converting {len(raw_data)} records to standardized format")
            standardized_data = await self._convert_to_standard_format(
                raw_data, source_name, data_type, warnings, errors
            )
            
            if not standardized_data:
                return self._create_error_result(
                    "No data could be converted to standardized format", 
                    len(raw_data), start_time, errors
                )
            
            # Step 2: Coordinate system transformation
            if config.target_coordinate_system != CoordinateSystem.WGS84:
                logger.info("Applying coordinate system transformation")
                standardized_data = await self._apply_coordinate_transformation(
                    standardized_data, config, warnings, errors
                )
                transformations_applied.append("coordinate_transformation")
                self.stats.coordinate_transformations += len(standardized_data)
            
            # Step 3: Temporal synchronization
            if config.target_temporal_resolution != TemporalResolution.MINUTE:
                logger.info("Applying temporal synchronization")
                standardized_data = await self._apply_temporal_synchronization(
                    standardized_data, config, warnings, errors
                )
                transformations_applied.append("temporal_synchronization")
                self.stats.temporal_synchronizations += len(standardized_data)
            
            # Step 4: Unit conversion and standardization
            logger.info("Applying unit standardization")
            standardized_data = await self._apply_unit_standardization(
                standardized_data, warnings, errors
            )
            transformations_applied.append("unit_standardization")
            self.stats.unit_conversions += len(standardized_data)
            
            # Step 5: Metadata harmonization
            logger.info("Applying metadata harmonization")
            standardized_data = await self._apply_metadata_harmonization(
                standardized_data, source_name, warnings, errors
            )
            transformations_applied.append("metadata_harmonization")
            self.stats.metadata_harmonizations += len(standardized_data)
            
            # Step 6: Gap filling and interpolation (if enabled)
            if config.fill_gaps:
                logger.info("Applying gap filling and interpolation")
                standardized_data = await self._apply_gap_filling(
                    standardized_data, config, warnings, errors
                )
                transformations_applied.append("gap_filling")
            
            # Step 7: Data aggregation (if requested)
            if hasattr(config, 'aggregation_enabled') and config.aggregation_enabled:
                logger.info("Applying data aggregation")
                standardized_data = await self._apply_aggregation(
                    standardized_data, config, warnings, errors
                )
                transformations_applied.append("aggregation")
            
            # Step 8: Final validation
            logger.info("Performing final validation")
            validation_result = await self._validate_processed_data(
                standardized_data, warnings, errors
            )
            
            # Calculate quality improvement
            quality_improvement = self._calculate_quality_improvement(
                raw_data, standardized_data
            )
            
            processing_time = time.time() - start_time
            
            return PipelineResult(
                success=len(errors) == 0,
                processed_data=standardized_data,
                original_count=len(raw_data),
                processed_count=len(standardized_data),
                processing_time_seconds=processing_time,
                transformations_applied=transformations_applied,
                quality_improvement=quality_improvement,
                validation_results=validation_result,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            processing_time = time.time() - start_time
            
            return PipelineResult(
                success=False,
                processed_data=[],
                original_count=len(raw_data),
                processed_count=0,
                processing_time_seconds=processing_time,
                transformations_applied=transformations_applied,
                quality_improvement=0.0,
                validation_results=ValidationResult(
                    is_valid=False,
                    errors=[str(e)],
                    warnings=warnings
                ),
                warnings=warnings,
                errors=[str(e)]
            )
    
    async def _convert_to_standard_format(
        self,
        raw_data: List[Dict[str, Any]],
        source_name: str,
        data_type: DataType,
        warnings: List[str],
        errors: List[str]
    ) -> List[StandardizedDataFormat]:
        """Convert raw data to standardized format"""
        standardized_data = []
        
        for i, raw_record in enumerate(raw_data):
            try:
                # Create standardized data record
                standardized = self._create_standardized_record(
                    raw_record, source_name, data_type, i
                )
                
                if standardized:
                    standardized_data.append(standardized)
                else:
                    warnings.append(f"Could not standardize record {i}")
                    
            except Exception as e:
                error_msg = f"Failed to convert record {i}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        return standardized_data
    
    def _create_standardized_record(
        self,
        raw_record: Dict[str, Any],
        source_name: str,
        data_type: DataType,
        record_index: int
    ) -> Optional[StandardizedDataFormat]:
        """Create standardized data record from raw data"""
        try:
            # Extract basic information
            data_id = raw_record.get('id', f"{source_name}_{record_index}")
            
            # Extract timestamp
            timestamp_field = self._find_timestamp_field(raw_record)
            if timestamp_field:
                timestamp = self._parse_timestamp(raw_record[timestamp_field])
            else:
                timestamp = datetime.now()
            
            # Extract location
            location = self._extract_location(raw_record)
            
            # Extract values and units
            values, units = self._extract_values_and_units(raw_record, data_type)
            
            # Create standardized format
            standardized = StandardizedDataFormat(
                data_id=data_id,
                data_type=data_type,
                location=location,
                timestamp=timestamp,
                values=values,
                units=units,
                source=source_name,
                metadata=raw_record.copy()
            )
            
            return standardized
            
        except Exception as e:
            logger.error(f"Failed to create standardized record: {e}")
            return None
    
    def _find_timestamp_field(self, raw_record: Dict[str, Any]) -> Optional[str]:
        """Find timestamp field in raw record"""
        timestamp_fields = [
            'timestamp', 'time', 'datetime', 'date_time',
            'acquisition_time', 'observation_time', 'event_time',
            'starttime', 'start_time', 'sensing_start'
        ]
        
        for field in timestamp_fields:
            if field in raw_record:
                return field
        
        return None
    
    def _parse_timestamp(self, timestamp_value: Any) -> datetime:
        """Parse timestamp from various formats"""
        if isinstance(timestamp_value, datetime):
            return timestamp_value
        
        if isinstance(timestamp_value, str):
            # Try common formats
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_value, fmt)
                except ValueError:
                    continue
        
        # Default to current time
        return datetime.now()
    
    def _extract_location(self, raw_record: Dict[str, Any]) -> Dict[str, float]:
        """Extract location information from raw record"""
        location = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        # Common field mappings
        x_fields = ['longitude', 'lon', 'x', 'easting']
        y_fields = ['latitude', 'lat', 'y', 'northing']
        z_fields = ['elevation', 'altitude', 'depth', 'z', 'height']
        
        for field in x_fields:
            if field in raw_record:
                try:
                    location['x'] = float(raw_record[field])
                    break
                except (ValueError, TypeError):
                    continue
        
        for field in y_fields:
            if field in raw_record:
                try:
                    location['y'] = float(raw_record[field])
                    break
                except (ValueError, TypeError):
                    continue
        
        for field in z_fields:
            if field in raw_record:
                try:
                    location['z'] = float(raw_record[field])
                    break
                except (ValueError, TypeError):
                    continue
        
        return location
    
    def _extract_values_and_units(
        self,
        raw_record: Dict[str, Any],
        data_type: DataType
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Extract data values and units from raw record"""
        values = {}
        units = {}
        
        # Skip metadata fields
        skip_fields = {
            'id', 'timestamp', 'time', 'datetime', 'date_time',
            'latitude', 'longitude', 'elevation', 'depth', 'altitude',
            'lat', 'lon', 'x', 'y', 'z', 'easting', 'northing',
            'source', 'network', 'station', 'channel', 'location'
        }
        
        for key, value in raw_record.items():
            if key.lower() not in skip_fields:
                # Check if it's a numerical value
                if isinstance(value, (int, float)):
                    values[key] = value
                    # Try to infer units
                    units[key] = self._infer_units(key, data_type)
                elif isinstance(value, str):
                    # Try to parse as number
                    try:
                        numeric_value = float(value)
                        values[key] = numeric_value
                        units[key] = self._infer_units(key, data_type)
                    except ValueError:
                        # Keep as string value
                        values[key] = value
                        units[key] = "text"
        
        return values, units
    
    def _infer_units(self, field_name: str, data_type: DataType) -> str:
        """Infer units based on field name and data type"""
        field_lower = field_name.lower()
        
        # Common unit patterns
        if 'temperature' in field_lower or 'temp' in field_lower:
            return 'C'
        elif 'pressure' in field_lower:
            return 'Pa'
        elif 'velocity' in field_lower or 'vel' in field_lower:
            return 'm/s'
        elif 'acceleration' in field_lower or 'acc' in field_lower:
            return 'm/s²'
        elif 'displacement' in field_lower or 'disp' in field_lower:
            return 'm'
        elif 'frequency' in field_lower or 'freq' in field_lower:
            return 'Hz'
        elif 'magnitude' in field_lower or 'mag' in field_lower:
            return 'Ml'
        elif 'energy' in field_lower:
            return 'J'
        elif 'concentration' in field_lower or 'conc' in field_lower:
            return 'ppm'
        elif 'coherence' in field_lower:
            return 'unitless'
        elif 'phase' in field_lower:
            return 'rad'
        else:
            return 'unknown'
    
    async def _apply_coordinate_transformation(
        self,
        data_list: List[StandardizedDataFormat],
        config: TransformationConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[StandardizedDataFormat]:
        """Apply coordinate system transformation"""
        try:
            transformed_data = self.coordinate_transformer.batch_transform_coordinates(
                data_list, config.target_coordinate_system
            )
            return transformed_data
        except Exception as e:
            errors.append(f"Coordinate transformation failed: {e}")
            return data_list
    
    async def _apply_temporal_synchronization(
        self,
        data_list: List[StandardizedDataFormat],
        config: TransformationConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[StandardizedDataFormat]:
        """Apply temporal synchronization"""
        try:
            sync_result = self.temporal_synchronizer.synchronize_data_formats(
                data_list, config.target_temporal_resolution, config.target_timezone
            )
            
            if sync_result.warnings:
                warnings.extend(sync_result.warnings)
            
            return sync_result.synchronized_data
        except Exception as e:
            errors.append(f"Temporal synchronization failed: {e}")
            return data_list
    
    async def _apply_unit_standardization(
        self,
        data_list: List[StandardizedDataFormat],
        warnings: List[str],
        errors: List[str]
    ) -> List[StandardizedDataFormat]:
        """Apply unit standardization"""
        try:
            standardized_data = self.unit_converter.batch_standardize_units(data_list)
            return standardized_data
        except Exception as e:
            errors.append(f"Unit standardization failed: {e}")
            return data_list
    
    async def _apply_metadata_harmonization(
        self,
        data_list: List[StandardizedDataFormat],
        source_name: str,
        warnings: List[str],
        errors: List[str]
    ) -> List[StandardizedDataFormat]:
        """Apply metadata harmonization"""
        try:
            harmonized_data = self.metadata_harmonizer.batch_harmonize_metadata(data_list)
            return harmonized_data
        except Exception as e:
            errors.append(f"Metadata harmonization failed: {e}")
            return data_list
    
    async def _apply_gap_filling(
        self,
        data_list: List[StandardizedDataFormat],
        config: TransformationConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[StandardizedDataFormat]:
        """Apply gap filling and interpolation"""
        try:
            interp_result = self.interpolation_engine.interpolate_temporal_gaps(
                data_list,
                method=config.interpolation_method,
                max_gap_duration=config.max_gap_duration
            )
            
            self.stats.interpolated_points += interp_result.interpolated_points
            
            return interp_result.interpolated_data
        except Exception as e:
            errors.append(f"Gap filling failed: {e}")
            return data_list
    
    async def _apply_aggregation(
        self,
        data_list: List[StandardizedDataFormat],
        config: TransformationConfig,
        warnings: List[str],
        errors: List[str]
    ) -> List[StandardizedDataFormat]:
        """Apply data aggregation"""
        try:
            agg_config = AggregationConfig(
                target_resolution=config.target_temporal_resolution,
                aggregation_method=config.aggregation_method
            )
            
            agg_result = self.aggregation_engine.aggregate_temporal_data(
                data_list, agg_config
            )
            
            self.stats.aggregated_records += agg_result.original_count - agg_result.aggregated_count
            
            return agg_result.aggregated_data
        except Exception as e:
            errors.append(f"Data aggregation failed: {e}")
            return data_list
    
    async def _validate_processed_data(
        self,
        data_list: List[StandardizedDataFormat],
        warnings: List[str],
        errors: List[str]
    ) -> ValidationResult:
        """Validate processed data"""
        validation_errors = []
        validation_warnings = []
        
        # Check data completeness
        if not data_list:
            validation_errors.append("No data records after processing")
        
        # Check data quality
        quality_scores = []
        for data in data_list:
            if data.quality_metrics:
                quality_scores.append(data.quality_metrics.overall_score)
            
            # Check for required fields
            if not data.timestamp:
                validation_errors.append(f"Missing timestamp in record {data.data_id}")
            
            if not data.location:
                validation_errors.append(f"Missing location in record {data.data_id}")
            
            if not data.values:
                validation_warnings.append(f"No data values in record {data.data_id}")
        
        # Calculate overall quality
        overall_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return ValidationResult(
            is_valid=len(validation_errors) == 0,
            errors=validation_errors,
            warnings=validation_warnings,
            quality_score=overall_quality
        )
    
    def _calculate_quality_improvement(
        self,
        raw_data: List[Dict[str, Any]],
        processed_data: List[StandardizedDataFormat]
    ) -> float:
        """Calculate quality improvement from processing"""
        if not processed_data:
            return 0.0
        
        # Simple heuristic based on standardization and processing steps
        base_improvement = 0.2  # Base improvement from standardization
        
        # Additional improvement from transformations
        transformation_improvement = len(self.stats.__dict__) * 0.05
        
        # Quality based on processing success rate
        success_rate = len(processed_data) / len(raw_data) if raw_data else 0.0
        
        return min(1.0, base_improvement + transformation_improvement * success_rate)
    
    def _create_error_result(
        self,
        error_message: str,
        original_count: int,
        start_time: float,
        errors: List[str]
    ) -> PipelineResult:
        """Create error result for pipeline failure"""
        processing_time = time.time() - start_time
        
        return PipelineResult(
            success=False,
            processed_data=[],
            original_count=original_count,
            processed_count=0,
            processing_time_seconds=processing_time,
            transformations_applied=[],
            quality_improvement=0.0,
            validation_results=ValidationResult(
                is_valid=False,
                errors=[error_message] + errors
            ),
            errors=[error_message] + errors
        )
    
    async def export_processed_data(
        self,
        processed_data: List[StandardizedDataFormat],
        export_config: ExportConfig
    ) -> Any:
        """Export processed data to specified format"""
        try:
            export_result = self.export_manager.export_data(processed_data, export_config)
            return export_result
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            raise
    
    def get_pipeline_statistics(self) -> PipelineStats:
        """Get current pipeline statistics"""
        return self.stats
    
    def reset_statistics(self):
        """Reset pipeline statistics"""
        self.stats = PipelineStats()