"""Data completeness and consistency validation"""

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import structlog
import torch.nn as nn
import torch.nn.functional as F
from models.geological_data import DataQuality

logger = structlog.get_logger()

class CompletenessValidator:
    """Data completeness and consistency validation"""

    def __init__(self, config: Any):
        self.config = config

        # Required fields for different data types
        self.required_fields = {
            "seismic": {
                "mandatory": ["timestamp", "location", "magnitude", "depth"],
                "optional": ["event_type", "magnitude_type", "confidence", "source"],
                "location_fields": ["latitude", "longitude"],
            },
            "thermal": {
                "mandatory": ["timestamp", "location", "temperature"],
                "optional": [
                    "measurement_method",
                    "spatial_resolution",
                    "uncertainty",
                    "confidence",
                ],
                "location_fields": ["latitude", "longitude"],
            },
            "strain": {
                "mandatory": ["timestamp", "location", "strain_rate"],
                "optional": [
                    "strain_rate_std",
                    "gauge_length",
                    "sampling_rate",
                    "channel_index",
                ],
                "location_fields": ["latitude", "longitude"],
            },
            "satellite": {
                "mandatory": ["timestamp", "location", "satellite", "instrument"],
                "optional": ["bands", "spatial_resolution", "cloud_cover", "scene_id"],
                "location_fields": ["latitude", "longitude"],
            },
            "fiber_optic": {
                "mandatory": ["timestamp", "location", "measurement_type"],
                "optional": [
                    "temperature",
                    "strain_rate",
                    "channel_index",
                    "sampling_rate",
                ],
                "location_fields": ["latitude", "longitude"],
            },
        }

        # Data type constraints
        self.data_constraints = {
            "seismic": {
                "magnitude": {"min": -3.0, "max": 10.0},
                "depth": {"min": 0.0, "max": 700.0},
                "confidence": {"min": 0.0, "max": 1.0},
            },
            "thermal": {
                "temperature": {"min": 150.0, "max": 500.0},  # Kelvin
                "uncertainty": {"min": 0.0, "max": 50.0},
                "spatial_resolution": {"min": 0.1, "max": 100000.0},
            },
            "strain": {
                "strain_rate": {"min": -1e-3, "max": 1e-3},
                "sampling_rate": {"min": 1.0, "max": 50000.0},
                "gauge_length": {"min": 0.1, "max": 1000.0},
            },
        }

        # Temporal consistency thresholds
        self.temporal_thresholds = {
            "max_future_offset": timedelta(minutes=5),  # Allow small clock skew
            "max_past_offset": timedelta(days=3650),  # 10 years
            "expected_intervals": {
                "seismic": timedelta(seconds=1),  # Real-time seismic data
                "thermal": timedelta(hours=12),  # Satellite thermal data
                "strain": timedelta(seconds=0.1),  # High-frequency strain data
                "atmospheric": timedelta(minutes=10),  # Weather data
            },
        }

    async def initialize(self):
        """Initialize the completeness validator"""
        logger.info("Completeness validator initialized")

    def validate_field_completeness(
        self, data: List[Any], data_type: str
    ) -> Dict[str, Any]:
        """Validate field completeness for a dataset"""

        if data_type not in self.required_fields:
            logger.warning(
                f"Unknown data type for completeness validation: {data_type}"
            )
            return {"error": f"Unknown data type: {data_type}"}

        try:
            field_config = self.required_fields[data_type]
            mandatory_fields = field_config["mandatory"]
            optional_fields = field_config["optional"]
            location_fields = field_config["location_fields"]

            total_records = len(data)
            if total_records == 0:
                return {
                    "total_records": 0,
                    "completeness_score": 0.0,
                    "field_completeness": {},
                    "missing_mandatory": [],
                    "issues": ["No data to validate"],
                }

            # Track field completeness
            field_completeness = {}
            missing_mandatory = []
            issues = []

            # Check each field
            all_fields = mandatory_fields + optional_fields

            for field in all_fields:
                present_count = 0

                for record in data:
                    if field == "location":
                        # Special handling for location field
                        if hasattr(record, "location") and record.location is not None:
                            location = record.location
                            if all(
                                hasattr(location, lf)
                                and getattr(location, lf) is not None
                                for lf in location_fields
                            ):
                                present_count += 1
                    else:
                        # Regular field check
                        if hasattr(record, field):
                            value = getattr(record, field)
                            if value is not None:
                                if isinstance(value, str) and value.strip():
                                    present_count += 1
                                elif isinstance(value, (int, float)) and not np.isnan(
                                    value
                                ):
                                    present_count += 1
                                elif value:  # Other truthy values
                                    present_count += 1

                completeness_percentage = (present_count / total_records) * 100
                field_completeness[field] = {
                    "present_count": present_count,
                    "missing_count": total_records - present_count,
                    "completeness_percentage": completeness_percentage,
                }

                # Check mandatory fields
                if field in mandatory_fields and completeness_percentage < 100:
                    missing_mandatory.append(field)
                    issues.append(
                        f"Mandatory field '{field}' missing in {total_records - present_count} records"
                    )

            # Calculate overall completeness score
            mandatory_scores = [
                field_completeness[field]["completeness_percentage"]
                for field in mandatory_fields
            ]
            optional_scores = [
                field_completeness[field]["completeness_percentage"]
                for field in optional_fields
            ]

            # Weighted score (mandatory fields have higher weight)
            mandatory_weight = 0.8
            optional_weight = 0.2

            mandatory_avg = np.mean(mandatory_scores) if mandatory_scores else 0
            optional_avg = np.mean(optional_scores) if optional_scores else 0

            completeness_score = (
                mandatory_weight * mandatory_avg + optional_weight * optional_avg
            )

            return {
                "total_records": total_records,
                "completeness_score": completeness_score,
                "field_completeness": field_completeness,
                "missing_mandatory": missing_mandatory,
                "mandatory_completeness": mandatory_avg,
                "optional_completeness": optional_avg,
                "issues": issues,
            }

        except Exception as e:
            logger.error("Error validating field completeness", error=str(e))
            return {"error": str(e)}

    def validate_data_consistency(
        self, data: List[Any], data_type: str
    ) -> Dict[str, Any]:
        """Validate data consistency within a dataset"""

        try:
            total_records = len(data)
            if total_records < 2:
                return {
                    "total_records": total_records,
                    "consistency_score": 100.0,
                    "issues": (
                        []
                        if total_records == 1
                        else ["Insufficient data for consistency check"]
                    ),
                }

            issues = []
            consistency_scores = []

            # Temporal consistency
            temporal_score, temporal_issues = self._validate_temporal_consistency(
                data, data_type
            )
            consistency_scores.append(temporal_score)
            issues.extend(temporal_issues)

            # Spatial consistency
            spatial_score, spatial_issues = self._validate_spatial_consistency(data)
            consistency_scores.append(spatial_score)
            issues.extend(spatial_issues)

            # Value consistency
            value_score, value_issues = self._validate_value_consistency(
                data, data_type
            )
            consistency_scores.append(value_score)
            issues.extend(value_issues)

            # Source consistency
            source_score, source_issues = self._validate_source_consistency(data)
            consistency_scores.append(source_score)
            issues.extend(source_issues)

            # Overall consistency score
            overall_score = np.mean(consistency_scores)

            return {
                "total_records": total_records,
                "consistency_score": overall_score,
                "temporal_consistency": temporal_score,
                "spatial_consistency": spatial_score,
                "value_consistency": value_score,
                "source_consistency": source_score,
                "issues": issues,
            }

        except Exception as e:
            logger.error("Error validating data consistency", error=str(e))
            return {"error": str(e)}

    def _validate_temporal_consistency(
        self, data: List[Any], data_type: str
    ) -> Tuple[float, List[str]]:
        """Validate temporal consistency"""

        try:
            timestamps = []
            issues = []

            # Extract timestamps
            for record in data:
                if hasattr(record, "timestamp") and record.timestamp:
                    timestamps.append(record.timestamp)

            if len(timestamps) < 2:
                return 100.0, []

            # Check for chronological order
            sorted_timestamps = sorted(timestamps)
            chronological_score = (
                sum(1 for i, ts in enumerate(timestamps) if ts == sorted_timestamps[i])
                / len(timestamps)
                * 100
            )

            if chronological_score < 90:
                issues.append(
                    f"Timestamps not in chronological order ({chronological_score:.1f}% ordered)"
                )

            # Check for reasonable time ranges
            now = datetime.utcnow()
            future_count = sum(
                1
                for ts in timestamps
                if ts > now + self.temporal_thresholds["max_future_offset"]
            )
            old_count = sum(
                1
                for ts in timestamps
                if ts < now - self.temporal_thresholds["max_past_offset"]
            )

            if future_count > 0:
                issues.append(f"{future_count} timestamps in the future")
            if old_count > 0:
                issues.append(
                    f"{old_count} timestamps older than {self.temporal_thresholds['max_past_offset'].days} days"
                )

            # Check for expected intervals (if applicable)
            if data_type in self.temporal_thresholds["expected_intervals"]:
                expected_interval = self.temporal_thresholds["expected_intervals"][
                    data_type
                ]

                # Calculate actual intervals
                intervals = []
                for i in range(1, len(sorted_timestamps)):
                    interval = sorted_timestamps[i] - sorted_timestamps[i - 1]
                    intervals.append(interval)

                if intervals:
                    median_interval = sorted(intervals)[len(intervals) // 2]

                    # Check if median interval is reasonable
                    if median_interval > expected_interval * 10:
                        issues.append(
                            f"Large gaps in data (median interval: {median_interval})"
                        )
                    elif median_interval < expected_interval / 10:
                        issues.append(
                            f"Very frequent data (median interval: {median_interval})"
                        )

            # Calculate temporal consistency score
            temporal_score = max(0, chronological_score - len(issues) * 10)

            return temporal_score, issues

        except Exception as e:
            logger.error("Error validating temporal consistency", error=str(e))
            return 0.0, [f"Temporal validation error: {str(e)}"]

    def _validate_spatial_consistency(self, data: List[Any]) -> Tuple[float, List[str]]:
        """Validate spatial consistency"""

        try:
            locations = []
            issues = []

            # Extract locations
            for record in data:
                if hasattr(record, "location") and record.location:
                    loc = record.location
                    if hasattr(loc, "latitude") and hasattr(loc, "longitude"):
                        locations.append((loc.latitude, loc.longitude))

            if len(locations) < 2:
                return 100.0, []

            # Check coordinate validity
            valid_coords = 0
            for lat, lon in locations:
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    valid_coords += 1

            coord_validity_score = (valid_coords / len(locations)) * 100

            if coord_validity_score < 100:
                invalid_count = len(locations) - valid_coords
                issues.append(f"{invalid_count} records with invalid coordinates")

            # Check for spatial clustering (detect if all points are too close or too spread)
            if len(locations) > 5:
                lats, lons = zip(*locations)
                lat_range = max(lats) - min(lats)
                lon_range = max(lons) - min(lons)

                # Very tight clustering might indicate duplicate data
                if lat_range < 0.001 and lon_range < 0.001:  # ~100m
                    issues.append(
                        "All locations very tightly clustered (possible duplicates)"
                    )

                # Very wide spread might indicate data quality issues
                elif lat_range > 90 or lon_range > 180:
                    issues.append("Locations span very large geographic area")

            # Calculate spatial consistency score
            spatial_score = max(0, coord_validity_score - len(issues) * 15)

            return spatial_score, issues

        except Exception as e:
            logger.error("Error validating spatial consistency", error=str(e))
            return 0.0, [f"Spatial validation error: {str(e)}"]

    def _validate_value_consistency(
        self, data: List[Any], data_type: str
    ) -> Tuple[float, List[str]]:
        """Validate value consistency"""

        try:
            issues = []
            value_scores = []

            if data_type not in self.data_constraints:
                return 100.0, []

            constraints = self.data_constraints[data_type]

            # Check each constrained field
            for field, constraint in constraints.items():
                values = []

                # Extract values
                for record in data:
                    if hasattr(record, field):
                        value = getattr(record, field)
                        if value is not None and not (
                            isinstance(value, float) and np.isnan(value)
                        ):
                            values.append(value)

                if not values:
                    continue

                # Check range constraints
                min_val = constraint.get("min")
                max_val = constraint.get("max")

                valid_count = 0
                for value in values:
                    if min_val is not None and value < min_val:
                        continue
                    if max_val is not None and value > max_val:
                        continue
                    valid_count += 1

                field_score = (valid_count / len(values)) * 100
                value_scores.append(field_score)

                if field_score < 100:
                    invalid_count = len(values) - valid_count
                    issues.append(
                        f"{invalid_count} {field} values outside valid range [{min_val}, {max_val}]"
                    )

                # Check for suspicious patterns
                if len(values) > 10:
                    # Check for too many identical values
                    value_counts = Counter(values)
                    most_common_count = value_counts.most_common(1)[0][1]
                    if most_common_count > len(values) * 0.8:
                        issues.append(
                            f"Too many identical {field} values ({most_common_count}/{len(values)})"
                        )

                    # Check for unrealistic precision
                    if field in ["magnitude", "temperature", "strain_rate"]:
                        # Count decimal places
                        decimal_places = []
                        for value in values:
                            str_val = str(value)
                            if "." in str_val:
                                decimal_places.append(len(str_val.split(".")[1]))
                            else:
                                decimal_places.append(0)

                        avg_precision = np.mean(decimal_places)
                        if avg_precision > 6:  # More than 6 decimal places
                            issues.append(
                                f"Unrealistic precision in {field} values (avg: {avg_precision:.1f} decimal places)"
                            )

            # Calculate overall value consistency score
            value_score = np.mean(value_scores) if value_scores else 100.0
            value_score = max(0, value_score - len(issues) * 10)

            return value_score, issues

        except Exception as e:
            logger.error("Error validating value consistency", error=str(e))
            return 0.0, [f"Value validation error: {str(e)}"]

    def _validate_source_consistency(self, data: List[Any]) -> Tuple[float, List[str]]:
        """Validate source consistency"""

        try:
            sources = []
            issues = []

            # Extract sources
            for record in data:
                if hasattr(record, "source") and record.source:
                    sources.append(record.source)

            if not sources:
                return 100.0, []

            # Check source distribution
            source_counts = Counter(sources)
            total_sources = len(source_counts)

            # Check for reasonable source distribution
            if total_sources == 1:
                # All from same source - good consistency
                source_score = 100.0
            else:
                # Multiple sources - check if distribution is reasonable
                max_source_count = max(source_counts.values())
                min_source_count = min(source_counts.values())

                # If one source dominates too much, it might indicate issues
                if max_source_count > len(sources) * 0.95:
                    issues.append("One source dominates dataset")
                    source_score = 80.0
                elif min_source_count == 1 and total_sources > 10:
                    issues.append("Many sources with single records")
                    source_score = 70.0
                else:
                    source_score = 90.0

            # Check for unknown or suspicious sources
            known_sources = {
                "usgs",
                "nasa",
                "esa",
                "modis",
                "landsat",
                "fiber_optic",
                "seismometer",
                "atmospheric",
                "hydroacoustic",
            }

            unknown_sources = set(sources) - known_sources
            if unknown_sources:
                issues.append(f"Unknown sources detected: {list(unknown_sources)}")
                source_score -= 10

            return max(0, source_score), issues

        except Exception as e:
            logger.error("Error validating source consistency", error=str(e))
            return 0.0, [f"Source validation error: {str(e)}"]

    def validate_data_gaps(
        self, data: List[Any], expected_interval: timedelta, tolerance: float = 2.0
    ) -> Dict[str, Any]:
        """Validate data for temporal gaps"""

        try:
            timestamps = []

            # Extract timestamps
            for record in data:
                if hasattr(record, "timestamp") and record.timestamp:
                    timestamps.append(record.timestamp)

            if len(timestamps) < 2:
                return {
                    "total_records": len(timestamps),
                    "gaps_detected": 0,
                    "gap_details": [],
                    "coverage_percentage": 100.0 if len(timestamps) == 1 else 0.0,
                }

            # Sort timestamps
            sorted_timestamps = sorted(timestamps)

            # Find gaps
            gaps = []
            for i in range(1, len(sorted_timestamps)):
                actual_interval = sorted_timestamps[i] - sorted_timestamps[i - 1]
                expected_duration = expected_interval.total_seconds()
                actual_duration = actual_interval.total_seconds()

                if actual_duration > expected_duration * tolerance:
                    gaps.append(
                        {
                            "start_time": sorted_timestamps[i - 1],
                            "end_time": sorted_timestamps[i],
                            "gap_duration": actual_interval,
                            "expected_duration": expected_interval,
                            "gap_ratio": actual_duration / expected_duration,
                        }
                    )

            # Calculate coverage percentage
            if len(sorted_timestamps) >= 2:
                total_duration = sorted_timestamps[-1] - sorted_timestamps[0]
                gap_duration = sum(gap["gap_duration"] for gap in gaps)
                coverage_percentage = max(
                    0,
                    (1 - gap_duration.total_seconds() / total_duration.total_seconds())
                    * 100,
                )
            else:
                coverage_percentage = 100.0

            return {
                "total_records": len(timestamps),
                "time_span": (
                    sorted_timestamps[-1] - sorted_timestamps[0]
                    if len(sorted_timestamps) >= 2
                    else timedelta(0)
                ),
                "gaps_detected": len(gaps),
                "gap_details": gaps,
                "coverage_percentage": coverage_percentage,
                "expected_interval": expected_interval,
            }

        except Exception as e:
            logger.error("Error validating data gaps", error=str(e))
            return {"error": str(e)}

    async def generate_completeness_report(
        self, data: List[Any], data_type: str
    ) -> Dict[str, Any]:
        """Generate comprehensive completeness and consistency report"""

        try:
            # Field completeness validation
            completeness_result = self.validate_field_completeness(data, data_type)

            # Data consistency validation
            consistency_result = self.validate_data_consistency(data, data_type)

            # Data gaps validation (if applicable)
            gaps_result = {}
            if data_type in self.temporal_thresholds["expected_intervals"]:
                expected_interval = self.temporal_thresholds["expected_intervals"][
                    data_type
                ]
                gaps_result = self.validate_data_gaps(data, expected_interval)

            # Overall quality assessment
            completeness_score = completeness_result.get("completeness_score", 0)
            consistency_score = consistency_result.get("consistency_score", 0)
            coverage_percentage = gaps_result.get("coverage_percentage", 100)

            # Weighted overall score
            overall_score = (
                0.4 * completeness_score
                + 0.4 * consistency_score
                + 0.2 * coverage_percentage
            )

            # Determine quality level
            if overall_score >= 90:
                quality_level = DataQuality.EXCELLENT
            elif overall_score >= 75:
                quality_level = DataQuality.GOOD
            elif overall_score >= 60:
                quality_level = DataQuality.FAIR
            else:
                quality_level = DataQuality.POOR

            return {
                "data_type": data_type,
                "total_records": len(data),
                "overall_score": overall_score,
                "quality_level": quality_level.value,
                "completeness": completeness_result,
                "consistency": consistency_result,
                "gaps": gaps_result,
                "timestamp": datetime.utcnow(),
                "recommendations": self._generate_recommendations(
                    completeness_result, consistency_result, gaps_result
                ),
            }

        except Exception as e:
            logger.error("Error generating completeness report", error=str(e))
            return {"error": str(e)}

    def _generate_recommendations(
        self,
        completeness_result: Dict[str, Any],
        consistency_result: Dict[str, Any],
        gaps_result: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations based on validation results"""

        recommendations = []

        try:
            # Completeness recommendations
            if completeness_result.get("completeness_score", 100) < 80:
                missing_mandatory = completeness_result.get("missing_mandatory", [])
                if missing_mandatory:
                    recommendations.append(
                        f"Fill missing mandatory fields: {', '.join(missing_mandatory)}"
                    )

                recommendations.append(
                    "Improve data collection procedures to ensure field completeness"
                )

            # Consistency recommendations
            if consistency_result.get("consistency_score", 100) < 70:
                if consistency_result.get("temporal_consistency", 100) < 70:
                    recommendations.append(
                        "Review timestamp accuracy and chronological ordering"
                    )

                if consistency_result.get("spatial_consistency", 100) < 70:
                    recommendations.append(
                        "Validate coordinate systems and location accuracy"
                    )

                if consistency_result.get("value_consistency", 100) < 70:
                    recommendations.append(
                        "Review data validation rules and acceptable value ranges"
                    )

            # Gap recommendations
            if gaps_result.get("coverage_percentage", 100) < 80:
                recommendations.append("Address temporal gaps in data collection")
                recommendations.append(
                    "Consider interpolation or gap-filling techniques"
                )

            if not recommendations:
                recommendations.append(
                    "Data quality is good - maintain current collection standards"
                )

        except Exception as e:
            logger.error("Error generating recommendations", error=str(e))
            recommendations.append(
                "Unable to generate specific recommendations due to analysis error"
            )

        return recommendations

    async def cleanup(self):
        """Cleanup completeness validator"""
        logger.info("Completeness validator cleanup completed")
