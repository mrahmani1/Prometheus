"""Data processor for geological data validation and standardization"""

import signal
import uuid
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import structlog
import torch.nn as nn
from models.geological_data import (
    DataQuality,
    FiberOpticMeasurement,
    ProcessedDataBatch,
    SatelliteObservation,
    SeismicEvent,
    StrainMeasurement,
    ThermalMeasurement,
)
from storage.data_storage import DataStorage
from streaming.kafka_producer import KafkaDataProducer

from .completeness_validator import CompletenessValidator
from .format_standardizer import FormatStandardizer
from .outlier_detector import OutlierDetector
from .quality_validator import DataQualityValidator

logger = structlog.get_logger()

class DataProcessor:
    """Main data processor for geological data validation and standardization"""

    def __init__(self, config: Any, data_storage: DataStorage):
        self.config = config
        self.data_storage = data_storage
        self.is_initialized = False

        # Quality control thresholds
        self.seismic_quality_threshold = config.seismic_quality_threshold
        self.deformation_quality_threshold = config.deformation_quality_threshold
        self.atmospheric_quality_threshold = config.atmospheric_quality_threshold

        # Processing statistics
        self.total_records_processed = 0
        self.total_records_rejected = 0
        self.processing_errors = 0

        # Batch processing
        self.batch_size = config.batch_size
        self.pending_batches = {}

        # Quality validation modules
        self.quality_validator = DataQualityValidator(config)
        self.format_standardizer = FormatStandardizer(config)
        self.outlier_detector = OutlierDetector(config)
        self.completeness_validator = CompletenessValidator(config)

        # Streaming integration
        self.kafka_producer = KafkaDataProducer(config)

    async def initialize(self):
        """Initialize the data processor"""
        # Initialize quality validation modules
        await self.quality_validator.initialize()
        await self.format_standardizer.initialize()
        await self.outlier_detector.initialize()
        await self.completeness_validator.initialize()

        # Initialize streaming
        await self.kafka_producer.initialize()

        self.is_initialized = True
        logger.info("Data processor initialized")

    async def is_ready(self) -> bool:
        """Check if the data processor is ready"""
        return self.is_initialized and await self.data_storage.is_healthy()

    async def _validate_seismic_event(self, event: SeismicEvent) -> DataQuality:
        """Validate seismic event data quality using advanced validation"""
        try:
            quality_level, quality_score, issues = (
                await self.quality_validator.validate_seismic_event(event)
            )

            if issues:
                logger.debug(
                    "Seismic event validation issues",
                    event_id=getattr(event, "event_id", "unknown"),
                    issues=issues,
                )

            return quality_level

        except Exception as e:
            logger.warning("Error validating seismic event", error=str(e))
            return DataQuality.UNKNOWN

    async def _validate_thermal_measurement(
        self, measurement: ThermalMeasurement
    ) -> DataQuality:
        """Validate thermal measurement data quality using advanced validation"""
        try:
            quality_level, quality_score, issues = (
                await self.quality_validator.validate_thermal_measurement(measurement)
            )

            if issues:
                logger.debug("Thermal measurement validation issues", issues=issues)

            return quality_level

        except Exception as e:
            logger.warning("Error validating thermal measurement", error=str(e))
            return DataQuality.UNKNOWN

    async def _standardize_coordinates(self, location) -> None:
        """Standardize coordinate system to WGS84"""
        try:
            if location.coordinate_system != "WGS84":
                # Use format standardizer for coordinate transformation
                standardized_location = (
                    self.format_standardizer.standardize_coordinates(
                        longitude=location.longitude,
                        latitude=location.latitude,
                        elevation=location.elevation,
                        from_crs=location.coordinate_system,
                    )
                )

                # Update location with standardized coordinates
                location.latitude = standardized_location.latitude
                location.longitude = standardized_location.longitude
                location.coordinate_system = "WGS84"

                logger.debug(
                    f"Converted coordinates from {location.coordinate_system} to WGS84"
                )
        except Exception as e:
            logger.warning("Error standardizing coordinates", error=str(e))

    async def _detect_outliers(
        self, values: List[float], threshold: float = 3.0
    ) -> List[bool]:
        """Detect outliers using z-score method"""
        if len(values) < 3:
            return [False] * len(values)

        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)

        if std == 0:
            return [False] * len(values)

        z_scores = np.abs((values_array - mean) / std)
        return z_scores > threshold

    async def process_seismic_event(self, event: SeismicEvent) -> bool:
        """Process and validate seismic event"""
        try:
            start_time = datetime.utcnow()

            # Standardize coordinates
            await self._standardize_coordinates(event.location)

            # Validate data quality
            event.quality = await self._validate_seismic_event(event)

            # Reject poor quality data if below threshold
            if event.quality == DataQuality.POOR:
                quality_score = 0.3  # Assumed score for poor quality
                if quality_score < self.seismic_quality_threshold:
                    logger.debug(
                        "Rejecting seismic event due to poor quality",
                        event_id=event.event_id,
                    )
                    self.total_records_rejected += 1
                    return False

            # Store the processed event
            await self.data_storage.store_seismic_event(event)

            # Send to streaming pipeline
            await self.kafka_producer.send_seismic_event(event)

            # Update statistics
            self.total_records_processed += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.debug(
                "Processed seismic event",
                event_id=event.event_id,
                magnitude=event.magnitude,
                quality=event.quality.value,
                processing_time_ms=processing_time,
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to process seismic event",
                event_id=getattr(event, "event_id", "unknown"),
                error=str(e),
            )
            self.processing_errors += 1
            return False

    async def process_thermal_measurement(
        self, measurement: ThermalMeasurement
    ) -> bool:
        """Process and validate thermal measurement"""
        try:
            start_time = datetime.utcnow()

            # Standardize coordinates
            await self._standardize_coordinates(measurement.location)

            # Validate data quality
            measurement.quality = await self._validate_thermal_measurement(measurement)

            # Store the processed measurement
            await self.data_storage.store_thermal_measurement(measurement)

            # Send to streaming pipeline
            await self.kafka_producer.send_thermal_measurement(measurement)

            # Update statistics
            self.total_records_processed += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.debug(
                "Processed thermal measurement",
                temperature=measurement.temperature,
                quality=measurement.quality.value,
                processing_time_ms=processing_time,
            )

            return True

        except Exception as e:
            logger.error("Failed to process thermal measurement", error=str(e))
            self.processing_errors += 1
            return False

    async def process_satellite_observation(
        self, observation: SatelliteObservation
    ) -> bool:
        """Process and validate satellite observation"""
        try:
            start_time = datetime.utcnow()

            # Standardize coordinates
            await self._standardize_coordinates(observation.location)

            # Basic quality assessment
            quality_score = 1.0
            if observation.cloud_cover > 50:
                quality_score *= 0.6
            if observation.spatial_resolution > 100:
                quality_score *= 0.8

            if quality_score >= 0.8:
                observation.quality = DataQuality.EXCELLENT
            elif quality_score >= 0.6:
                observation.quality = DataQuality.GOOD
            else:
                observation.quality = DataQuality.FAIR

            # Store the processed observation
            await self.data_storage.store_satellite_observation(observation)

            # Update statistics
            self.total_records_processed += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.debug(
                "Processed satellite observation",
                satellite=observation.satellite,
                scene_id=observation.scene_id,
                quality=observation.quality.value,
                processing_time_ms=processing_time,
            )

            return True

        except Exception as e:
            logger.error("Failed to process satellite observation", error=str(e))
            self.processing_errors += 1
            return False

    async def process_fiber_optic_measurement(
        self, measurement: FiberOpticMeasurement
    ) -> bool:
        """Process and validate fiber optic measurement"""
        try:
            start_time = datetime.utcnow()

            # Standardize coordinates
            await self._standardize_coordinates(measurement.location)

            # Basic quality assessment
            measurement.quality = (
                DataQuality.GOOD
            )  # Fiber optic data is typically high quality

            # Store the processed measurement
            await self.data_storage.store_fiber_optic_measurement(measurement)

            # Update statistics
            self.total_records_processed += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.debug(
                "Processed fiber optic measurement",
                measurement_type=measurement.measurement_type,
                channel=measurement.channel_index,
                processing_time_ms=processing_time,
            )

            return True

        except Exception as e:
            logger.error("Failed to process fiber optic measurement", error=str(e))
            self.processing_errors += 1
            return False

    async def process_strain_measurement(self, measurement: StrainMeasurement) -> bool:
        """Process and validate strain measurement"""
        try:
            start_time = datetime.utcnow()

            # Standardize coordinates
            await self._standardize_coordinates(measurement.location)

            # Quality assessment based on signal-to-noise ratio
            if measurement.strain_rate_std > 0:
                snr = abs(measurement.strain_rate) / measurement.strain_rate_std
                if snr > 10:
                    measurement.quality = DataQuality.EXCELLENT
                elif snr > 5:
                    measurement.quality = DataQuality.GOOD
                elif snr > 2:
                    measurement.quality = DataQuality.FAIR
                else:
                    measurement.quality = DataQuality.POOR
            else:
                measurement.quality = DataQuality.GOOD

            # Store the processed measurement
            await self.data_storage.store_strain_measurement(measurement)

            # Send to streaming pipeline
            await self.kafka_producer.send_strain_measurement(measurement)

            # Update statistics
            self.total_records_processed += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.debug(
                "Processed strain measurement",
                strain_rate=measurement.strain_rate,
                channel=measurement.channel_index,
                quality=measurement.quality.value,
                processing_time_ms=processing_time,
            )

            return True

        except Exception as e:
            logger.error("Failed to process strain measurement", error=str(e))
            self.processing_errors += 1
            return False

    async def process_gravity_measurement(self, measurement: Dict[str, Any]) -> bool:
        """Process and validate gravity measurement"""
        try:
            start_time = datetime.utcnow()

            # Store the processed measurement
            await self.data_storage.store_gravity_measurement(measurement)

            # Update statistics
            self.total_records_processed += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.debug(
                "Processed gravity measurement", processing_time_ms=processing_time
            )

            return True

        except Exception as e:
            logger.error("Failed to process gravity measurement", error=str(e))
            self.processing_errors += 1
            return False

    async def create_processing_batch(
        self, data_type: str, source_connector: str, records: List[Any]
    ) -> ProcessedDataBatch:
        """Create a processing batch for multiple records"""

        batch_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        batch = ProcessedDataBatch(
            batch_id=batch_id,
            timestamp=start_time,
            data_type=data_type,
            source_connector=source_connector,
            record_count=len(records),
            processing_time_ms=0.0,
            quality_metrics={},
            errors=[],
            metadata={},
        )

        return batch

    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "total_records_processed": self.total_records_processed,
            "total_records_rejected": self.total_records_rejected,
            "processing_errors": self.processing_errors,
            "success_rate": (
                self.total_records_processed
                / (
                    self.total_records_processed
                    + self.total_records_rejected
                    + self.processing_errors
                )
                if (
                    self.total_records_processed
                    + self.total_records_rejected
                    + self.processing_errors
                )
                > 0
                else 0.0
            ),
            "pending_batches": len(self.pending_batches),
        }

    async def process_batch_with_quality_control(
        self, data_batch: List[Any], data_type: str, source_connector: str
    ) -> ProcessedDataBatch:
        """Process a batch of data with comprehensive quality control"""

        batch_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        try:
            # Initialize batch
            batch = ProcessedDataBatch(
                batch_id=batch_id,
                timestamp=start_time,
                data_type=data_type,
                source_connector=source_connector,
                record_count=len(data_batch),
                processing_time_ms=0.0,
                quality_metrics={},
                errors=[],
                metadata={},
            )

            if not data_batch:
                batch.processing_time_ms = (
                    datetime.utcnow() - start_time
                ).total_seconds() * 1000
                return batch

            # Step 1: Outlier detection
            outlier_flags = []
            if data_type == "seismic":
                outlier_flags = await self.outlier_detector.detect_seismic_outliers(
                    data_batch
                )
            elif data_type == "thermal":
                outlier_flags = await self.outlier_detector.detect_thermal_outliers(
                    data_batch
                )
            else:
                outlier_flags = [False] * len(data_batch)

            # Step 2: Clean outliers
            cleaned_data = await self.outlier_detector.clean_outliers(
                data_batch, outlier_flags, cleaning_method="flag"
            )

            # Step 3: Completeness validation
            completeness_report = (
                await self.completeness_validator.generate_completeness_report(
                    cleaned_data, data_type
                )
            )

            # Step 4: Process individual records
            processed_count = 0
            rejected_count = 0

            for record in cleaned_data:
                try:
                    if data_type == "seismic":
                        success = await self.process_seismic_event(record)
                    elif data_type == "thermal":
                        success = await self.process_thermal_measurement(record)
                    elif data_type == "strain":
                        success = await self.process_strain_measurement(record)
                    elif data_type == "satellite":
                        success = await self.process_satellite_observation(record)
                    elif data_type == "fiber_optic":
                        success = await self.process_fiber_optic_measurement(record)
                    else:
                        success = False
                        batch.errors.append(f"Unknown data type: {data_type}")

                    if success:
                        processed_count += 1
                    else:
                        rejected_count += 1

                except Exception as e:
                    batch.errors.append(f"Error processing record: {str(e)}")
                    rejected_count += 1

            # Update batch metrics
            batch.quality_metrics = {
                "outliers_detected": sum(outlier_flags),
                "outlier_percentage": (sum(outlier_flags) / len(data_batch)) * 100,
                "completeness_score": completeness_report.get("overall_score", 0),
                "processed_count": processed_count,
                "rejected_count": rejected_count,
                "success_rate": (
                    (processed_count / len(data_batch)) * 100 if data_batch else 0
                ),
            }

            batch.metadata = {
                "completeness_report": completeness_report,
                "outlier_summary": await self.outlier_detector.get_outlier_summary(
                    data_batch, outlier_flags
                ),
            }

            # Send batch metadata to streaming pipeline
            await self.kafka_producer.send_processing_batch(batch)

            # Update global statistics
            self.total_records_processed += processed_count
            self.total_records_rejected += rejected_count

            batch.processing_time_ms = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000

            logger.info(
                "Batch processing completed",
                batch_id=batch_id,
                data_type=data_type,
                total_records=len(data_batch),
                processed=processed_count,
                rejected=rejected_count,
                outliers=sum(outlier_flags),
                processing_time_ms=batch.processing_time_ms,
            )

            return batch

        except Exception as e:
            batch.errors.append(f"Batch processing error: {str(e)}")
            batch.processing_time_ms = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000
            logger.error("Batch processing failed", batch_id=batch_id, error=str(e))
            return batch

    async def cleanup(self):
        """Cleanup the data processor"""
        # Cleanup quality validation modules
        (
            await self.quality_validator.cleanup()
            if hasattr(self.quality_validator, "cleanup")
            else None
        )
        (
            await self.format_standardizer.cleanup()
            if hasattr(self.format_standardizer, "cleanup")
            else None
        )
        (
            await self.outlier_detector.cleanup()
            if hasattr(self.outlier_detector, "cleanup")
            else None
        )
        (
            await self.completeness_validator.cleanup()
            if hasattr(self.completeness_validator, "cleanup")
            else None
        )

        # Cleanup streaming
        await self.kafka_producer.cleanup()

        self.is_initialized = False
        logger.info("Data processor cleanup completed")
