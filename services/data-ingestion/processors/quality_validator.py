"""Data quality validation algorithms for geological data"""

import signal
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import structlog
import torch.nn as nn
import torch.nn.functional as F
from models.geological_data import (
    DataQuality,
    GeologicalLocation,
    SeismicEvent,
    StrainMeasurement,
    ThermalMeasurement,
)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

class DataQualityValidator:
    """Automated data quality assessment algorithms"""

    def __init__(self, config: Any):
        self.config = config

        # Quality thresholds
        self.seismic_quality_threshold = config.seismic_quality_threshold
        self.deformation_quality_threshold = config.deformation_quality_threshold
        self.atmospheric_quality_threshold = config.atmospheric_quality_threshold

        # Outlier detection models
        self.outlier_detectors = {}
        self.scalers = {}

        # Historical data for comparison
        self.historical_stats = {}

    async def initialize(self):
        """Initialize quality validation models"""
        # Initialize outlier detection models for different data types
        self.outlier_detectors = {
            "seismic": IsolationForest(contamination=0.1, random_state=42),
            "thermal": IsolationForest(contamination=0.05, random_state=42),
            "deformation": IsolationForest(contamination=0.08, random_state=42),
            "strain": IsolationForest(contamination=0.12, random_state=42),
        }

        self.scalers = {
            data_type: StandardScaler() for data_type in self.outlier_detectors.keys()
        }

        logger.info("Data quality validator initialized")

    def _calculate_location_quality(self, location: GeologicalLocation) -> float:
        """Calculate quality score for geographical location"""
        quality_score = 1.0

        # Check coordinate validity
        if abs(location.latitude) > 90:
            quality_score *= 0.0  # Invalid latitude
        elif abs(location.latitude) > 85:
            quality_score *= 0.7  # Extreme latitude (polar regions)

        if abs(location.longitude) > 180:
            quality_score *= 0.0  # Invalid longitude

        # Check elevation reasonableness
        if location.elevation < -11000 or location.elevation > 9000:  # meters
            quality_score *= 0.6  # Extreme elevation

        # Check depth if provided
        if location.depth is not None:
            if location.depth < 0 or location.depth > 700:  # km
                quality_score *= 0.7  # Unusual depth

        # Check uncertainty if provided
        if hasattr(location, "uncertainty_m") and location.uncertainty_m is not None:
            if location.uncertainty_m > 10000:  # meters
                quality_score *= 0.5  # Very high uncertainty

        return quality_score

    def _calculate_temporal_quality(self, timestamp: datetime) -> float:
        """Calculate quality score for timestamp"""
        quality_score = 1.0

        now = datetime.utcnow()

        # Check if timestamp is in the future
        if timestamp > now + timedelta(minutes=5):
            quality_score *= 0.3  # Future timestamp (allowing small clock skew)

        # Check if timestamp is too old (more than 10 years)
        if timestamp < now - timedelta(days=3650):
            quality_score *= 0.8  # Very old data

        return quality_score

    async def validate_seismic_event(
        self, event: SeismicEvent
    ) -> Tuple[DataQuality, float, List[str]]:
        """Validate seismic event data quality"""
        quality_score = 1.0
        issues = []

        try:
            # Location quality
            location_quality = self._calculate_location_quality(event.location)
            quality_score *= location_quality
            if location_quality < 0.8:
                issues.append("Location quality issues detected")

            # Temporal quality
            temporal_quality = self._calculate_temporal_quality(event.timestamp)
            quality_score *= temporal_quality
            if temporal_quality < 0.8:
                issues.append("Temporal quality issues detected")

            # Magnitude validation
            if event.magnitude < -3.0 or event.magnitude > 10.0:
                quality_score *= 0.4
                issues.append(f"Unusual magnitude: {event.magnitude}")
            elif event.magnitude < 0.0 or event.magnitude > 9.0:
                quality_score *= 0.7
                issues.append(f"Extreme magnitude: {event.magnitude}")

            # Depth validation
            if event.depth < 0 or event.depth > 700:  # km
                quality_score *= 0.5
                issues.append(f"Unusual depth: {event.depth} km")
            elif event.depth > 300:
                quality_score *= 0.8
                issues.append(f"Deep event: {event.depth} km")

            # Confidence validation
            if event.confidence < 0.0 or event.confidence > 1.0:
                quality_score *= 0.3
                issues.append(f"Invalid confidence: {event.confidence}")
            else:
                quality_score *= event.confidence

            # Event type validation
            valid_event_types = [
                "earthquake",
                "volcanic",
                "explosion",
                "quarry_blast",
                "nuclear_test",
                "landslide",
                "rockfall",
                "ice_quake",
            ]
            if event.event_type not in valid_event_types:
                quality_score *= 0.9
                issues.append(f"Unknown event type: {event.event_type}")

            # Magnitude type validation
            valid_mag_types = ["ML", "Mw", "Ms", "mb", "Md", "Mc", "unknown"]
            if event.magnitude_type not in valid_mag_types:
                quality_score *= 0.95
                issues.append(f"Unknown magnitude type: {event.magnitude_type}")

            # Cross-validation checks
            if event.event_type == "volcanic" and event.depth > 50:
                quality_score *= 0.8
                issues.append("Deep volcanic event (unusual)")

            if event.magnitude > 7.0 and event.depth < 10:
                quality_score *= 0.9
                issues.append("Large shallow earthquake (verify)")

            # Determine quality level
            if quality_score >= 0.9:
                quality_level = DataQuality.EXCELLENT
            elif quality_score >= 0.7:
                quality_level = DataQuality.GOOD
            elif quality_score >= 0.5:
                quality_level = DataQuality.FAIR
            else:
                quality_level = DataQuality.POOR

            return quality_level, quality_score, issues

        except Exception as e:
            logger.error("Error validating seismic event", error=str(e))
            return DataQuality.UNKNOWN, 0.0, [f"Validation error: {str(e)}"]

    async def validate_thermal_measurement(
        self, measurement: ThermalMeasurement
    ) -> Tuple[DataQuality, float, List[str]]:
        """Validate thermal measurement data quality"""
        quality_score = 1.0
        issues = []

        try:
            # Location quality
            location_quality = self._calculate_location_quality(measurement.location)
            quality_score *= location_quality
            if location_quality < 0.8:
                issues.append("Location quality issues detected")

            # Temporal quality
            temporal_quality = self._calculate_temporal_quality(measurement.timestamp)
            quality_score *= temporal_quality
            if temporal_quality < 0.8:
                issues.append("Temporal quality issues detected")

            # Temperature validation (Kelvin)
            if measurement.temperature < 150 or measurement.temperature > 500:
                quality_score *= 0.3
                issues.append(f"Extreme temperature: {measurement.temperature} K")
            elif measurement.temperature < 200 or measurement.temperature > 400:
                quality_score *= 0.7
                issues.append(f"Unusual temperature: {measurement.temperature} K")

            # Uncertainty validation
            if measurement.uncertainty < 0:
                quality_score *= 0.2
                issues.append("Negative uncertainty")
            elif measurement.uncertainty > 20:  # K
                quality_score *= 0.6
                issues.append(f"High uncertainty: {measurement.uncertainty} K")
            elif measurement.uncertainty > 10:
                quality_score *= 0.8
                issues.append(f"Moderate uncertainty: {measurement.uncertainty} K")

            # Spatial resolution validation
            if measurement.spatial_resolution <= 0:
                quality_score *= 0.2
                issues.append("Invalid spatial resolution")
            elif measurement.spatial_resolution > 50000:  # meters
                quality_score *= 0.6
                issues.append(
                    f"Low spatial resolution: {measurement.spatial_resolution} m"
                )

            # Confidence validation
            if measurement.confidence < 0.0 or measurement.confidence > 1.0:
                quality_score *= 0.3
                issues.append(f"Invalid confidence: {measurement.confidence}")
            else:
                quality_score *= measurement.confidence

            # Method validation
            valid_methods = [
                "modis_terra",
                "modis_aqua",
                "landsat_8",
                "landsat_9",
                "aster",
                "avhrr",
                "viirs",
                "ground_based",
                "thermal_camera",
                "infrared_thermometer",
                "borehole",
            ]
            if measurement.measurement_method not in valid_methods:
                quality_score *= 0.95
                issues.append(
                    f"Unknown measurement method: {measurement.measurement_method}"
                )

            # Cross-validation checks
            if "satellite" in measurement.measurement_method.lower():
                if measurement.spatial_resolution < 10:  # meters
                    quality_score *= 0.8
                    issues.append(
                        "Unusually high resolution for satellite thermal data"
                    )

            # Fire radiative power validation (if present)
            if (
                hasattr(measurement, "fire_radiative_power")
                and measurement.fire_radiative_power is not None
            ):
                if measurement.fire_radiative_power < 0:
                    quality_score *= 0.5
                    issues.append("Negative fire radiative power")
                elif measurement.fire_radiative_power > 10000:  # MW
                    quality_score *= 0.8
                    issues.append(
                        f"Very high fire radiative power: {measurement.fire_radiative_power} MW"
                    )

            # Determine quality level
            if quality_score >= 0.9:
                quality_level = DataQuality.EXCELLENT
            elif quality_score >= 0.7:
                quality_level = DataQuality.GOOD
            elif quality_score >= 0.5:
                quality_level = DataQuality.FAIR
            else:
                quality_level = DataQuality.POOR

            return quality_level, quality_score, issues

        except Exception as e:
            logger.error("Error validating thermal measurement", error=str(e))
            return DataQuality.UNKNOWN, 0.0, [f"Validation error: {str(e)}"]

    async def validate_strain_measurement(
        self, measurement: StrainMeasurement
    ) -> Tuple[DataQuality, float, List[str]]:
        """Validate strain measurement data quality"""
        quality_score = 1.0
        issues = []

        try:
            # Location quality
            location_quality = self._calculate_location_quality(measurement.location)
            quality_score *= location_quality
            if location_quality < 0.8:
                issues.append("Location quality issues detected")

            # Temporal quality
            temporal_quality = self._calculate_temporal_quality(measurement.timestamp)
            quality_score *= temporal_quality
            if temporal_quality < 0.8:
                issues.append("Temporal quality issues detected")

            # Strain rate validation (1/s)
            if abs(measurement.strain_rate) > 1e-3:  # Very high strain rate
                quality_score *= 0.4
                issues.append(f"Extreme strain rate: {measurement.strain_rate}")
            elif abs(measurement.strain_rate) > 1e-6:  # High strain rate
                quality_score *= 0.7
                issues.append(f"High strain rate: {measurement.strain_rate}")

            # Standard deviation validation
            if measurement.strain_rate_std < 0:
                quality_score *= 0.2
                issues.append("Negative strain rate standard deviation")
            elif measurement.strain_rate_std > abs(measurement.strain_rate) * 2:
                quality_score *= 0.6
                issues.append("High noise level in strain measurement")

            # Signal-to-noise ratio
            if measurement.strain_rate_std > 0:
                snr = abs(measurement.strain_rate) / measurement.strain_rate_std
                if snr < 1:
                    quality_score *= 0.5
                    issues.append(f"Low signal-to-noise ratio: {snr:.2f}")
                elif snr < 3:
                    quality_score *= 0.8
                    issues.append(f"Moderate signal-to-noise ratio: {snr:.2f}")

            # Maximum strain rate validation
            if measurement.strain_rate_max < abs(measurement.strain_rate):
                quality_score *= 0.7
                issues.append("Maximum strain rate less than mean strain rate")

            # Gauge length validation
            if measurement.gauge_length <= 0:
                quality_score *= 0.3
                issues.append("Invalid gauge length")
            elif measurement.gauge_length > 100:  # meters
                quality_score *= 0.9
                issues.append(f"Large gauge length: {measurement.gauge_length} m")

            # Sampling rate validation
            if measurement.sampling_rate <= 0:
                quality_score *= 0.2
                issues.append("Invalid sampling rate")
            elif measurement.sampling_rate < 100:  # Hz
                quality_score *= 0.8
                issues.append(f"Low sampling rate: {measurement.sampling_rate} Hz")
            elif measurement.sampling_rate > 10000:  # Hz
                quality_score *= 0.9
                issues.append(
                    f"Very high sampling rate: {measurement.sampling_rate} Hz"
                )

            # Channel index validation
            if measurement.channel_index < 0:
                quality_score *= 0.3
                issues.append("Invalid channel index")

            # Determine quality level
            if quality_score >= 0.9:
                quality_level = DataQuality.EXCELLENT
            elif quality_score >= 0.7:
                quality_level = DataQuality.GOOD
            elif quality_score >= 0.5:
                quality_level = DataQuality.FAIR
            else:
                quality_level = DataQuality.POOR

            return quality_level, quality_score, issues

        except Exception as e:
            logger.error("Error validating strain measurement", error=str(e))
            return DataQuality.UNKNOWN, 0.0, [f"Validation error: {str(e)}"]

    async def detect_outliers_batch(
        self, data_type: str, measurements: List[Any]
    ) -> List[bool]:
        """Detect outliers in a batch of measurements using machine learning"""

        if len(measurements) < 10:  # Need minimum samples for outlier detection
            return [False] * len(measurements)

        try:
            # Extract features based on data type
            features = []

            if data_type == "seismic":
                features = [
                    [
                        m.magnitude,
                        m.depth,
                        m.location.latitude,
                        m.location.longitude,
                        m.confidence,
                    ]
                    for m in measurements
                ]
            elif data_type == "thermal":
                features = [
                    [
                        m.temperature,
                        m.uncertainty,
                        m.spatial_resolution,
                        m.confidence,
                        m.location.latitude,
                        m.location.longitude,
                    ]
                    for m in measurements
                ]
            elif data_type == "strain":
                features = [
                    [
                        m.strain_rate,
                        m.strain_rate_std,
                        m.strain_rate_max,
                        m.gauge_length,
                        m.sampling_rate,
                        m.location.latitude,
                        m.location.longitude,
                    ]
                    for m in measurements
                ]
            else:
                return [False] * len(measurements)

            # Convert to numpy array
            features_array = np.array(features)

            # Handle missing values
            features_array = np.nan_to_num(features_array)

            # Scale features
            if data_type not in self.scalers:
                self.scalers[data_type] = StandardScaler()

            features_scaled = self.scalers[data_type].fit_transform(features_array)

            # Detect outliers
            if data_type not in self.outlier_detectors:
                self.outlier_detectors[data_type] = IsolationForest(
                    contamination=0.1, random_state=42
                )

            outlier_predictions = self.outlier_detectors[data_type].fit_predict(
                features_scaled
            )

            # Convert to boolean (True = outlier)
            is_outlier = outlier_predictions == -1

            logger.debug(
                f"Outlier detection completed for {data_type}",
                total_samples=len(measurements),
                outliers_detected=sum(is_outlier),
            )

            return is_outlier.tolist()

        except Exception as e:
            logger.error(f"Error in outlier detection for {data_type}", error=str(e))
            return [False] * len(measurements)

    async def calculate_completeness_score(
        self, measurements: List[Any], expected_fields: List[str]
    ) -> float:
        """Calculate data completeness score"""

        if not measurements:
            return 0.0

        total_score = 0.0

        for measurement in measurements:
            field_scores = []

            for field in expected_fields:
                if hasattr(measurement, field):
                    value = getattr(measurement, field)
                    if value is not None:
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            field_scores.append(1.0)
                        elif isinstance(value, str) and value.strip():
                            field_scores.append(1.0)
                        elif value:  # Other truthy values
                            field_scores.append(1.0)
                        else:
                            field_scores.append(0.0)
                    else:
                        field_scores.append(0.0)
                else:
                    field_scores.append(0.0)

            measurement_score = np.mean(field_scores) if field_scores else 0.0
            total_score += measurement_score

        return total_score / len(measurements)

    async def calculate_consistency_score(self, measurements: List[Any]) -> float:
        """Calculate data consistency score"""

        if len(measurements) < 2:
            return 1.0  # Single measurement is consistent by definition

        try:
            consistency_scores = []

            # Check temporal consistency (measurements should be in chronological order)
            timestamps = [m.timestamp for m in measurements if hasattr(m, "timestamp")]
            if len(timestamps) > 1:
                sorted_timestamps = sorted(timestamps)
                temporal_consistency = sum(
                    1 for i, ts in enumerate(timestamps) if ts == sorted_timestamps[i]
                ) / len(timestamps)
                consistency_scores.append(temporal_consistency)

            # Check spatial consistency (measurements from same source should be nearby)
            locations = [
                (m.location.latitude, m.location.longitude)
                for m in measurements
                if hasattr(m, "location")
            ]
            if len(locations) > 1:
                # Calculate spatial spread
                lats, lons = zip(*locations)
                lat_range = max(lats) - min(lats)
                lon_range = max(lons) - min(lons)

                # Normalize by expected range (adjust based on data type)
                expected_range = 1.0  # degrees
                spatial_consistency = max(
                    0, 1 - (lat_range + lon_range) / (2 * expected_range)
                )
                consistency_scores.append(spatial_consistency)

            # Check value consistency (similar measurements should have similar values)
            if hasattr(measurements[0], "magnitude"):  # Seismic events
                magnitudes = [m.magnitude for m in measurements]
                mag_std = np.std(magnitudes)
                mag_consistency = max(
                    0, 1 - mag_std / 3.0
                )  # Normalize by typical range
                consistency_scores.append(mag_consistency)

            return np.mean(consistency_scores) if consistency_scores else 1.0

        except Exception as e:
            logger.error("Error calculating consistency score", error=str(e))
            return 0.5  # Neutral score on error

    async def get_quality_summary(
        self, measurements: List[Any], data_type: str
    ) -> Dict[str, Any]:
        """Get comprehensive quality summary for a batch of measurements"""

        if not measurements:
            return {
                "total_measurements": 0,
                "quality_distribution": {},
                "completeness_score": 0.0,
                "consistency_score": 0.0,
                "outlier_count": 0,
                "issues": [],
            }

        try:
            # Quality distribution
            quality_counts = {}
            all_issues = []

            for measurement in measurements:
                quality = getattr(measurement, "quality", DataQuality.UNKNOWN)
                quality_counts[quality.value] = quality_counts.get(quality.value, 0) + 1

            # Completeness score
            expected_fields = {
                "seismic": [
                    "timestamp",
                    "location",
                    "magnitude",
                    "depth",
                    "event_type",
                ],
                "thermal": ["timestamp", "location", "temperature", "uncertainty"],
                "strain": ["timestamp", "location", "strain_rate", "strain_rate_std"],
            }.get(data_type, ["timestamp", "location"])

            completeness_score = await self.calculate_completeness_score(
                measurements, expected_fields
            )

            # Consistency score
            consistency_score = await self.calculate_consistency_score(measurements)

            # Outlier detection
            outliers = await self.detect_outliers_batch(data_type, measurements)
            outlier_count = sum(outliers)

            return {
                "total_measurements": len(measurements),
                "quality_distribution": quality_counts,
                "completeness_score": completeness_score,
                "consistency_score": consistency_score,
                "outlier_count": outlier_count,
                "outlier_percentage": (outlier_count / len(measurements)) * 100,
                "issues": all_issues,
            }

        except Exception as e:
            logger.error("Error generating quality summary", error=str(e))
            return {"total_measurements": len(measurements), "error": str(e)}
