"""
Data Quality Validation Framework for Geological Observations

This module implements comprehensive data quality assessment algorithms for geological
observations, including cross-source validation, anomaly detection, completeness
analysis, uncertainty quantification, and provenance tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityFlag(Enum):
    """Quality flags for geological observations"""
    VALID = "valid"
    SUSPICIOUS = "suspicious"
    INVALID = "invalid"
    MISSING = "missing"
    INTERPOLATED = "interpolated"
    OUTLIER = "outlier"
    INCONSISTENT = "inconsistent"
    UNCERTAIN = "uncertain"


class DataType(Enum):
    """Types of geological data"""
    SEISMIC = "seismic"
    VOLCANIC = "volcanic"
    GEOCHEMICAL = "geochemical"
    DEFORMATION = "deformation"
    THERMAL = "thermal"
    MAGNETIC = "magnetic"
    GRAVITY = "gravity"
    METEOROLOGICAL = "meteorological"


@dataclass
class QualityMetrics:
    """Quality metrics for geological data"""
    completeness_score: float  # 0-1, percentage of expected data present
    accuracy_score: float      # 0-1, accuracy against reference data
    consistency_score: float   # 0-1, consistency across sources
    timeliness_score: float    # 0-1, data freshness score
    uncertainty_level: float   # 0-1, uncertainty in measurements
    anomaly_score: float       # 0-1, likelihood of being anomalous
    overall_quality: float     # 0-1, weighted overall quality score
    confidence_interval: Tuple[float, float]  # Confidence bounds
    
    def __post_init__(self):
        """Calculate overall quality score"""
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.20,
            'timeliness': 0.15,
            'uncertainty': 0.10,
            'anomaly': 0.05
        }
        
        self.overall_quality = (
            self.completeness_score * weights['completeness'] +
            self.accuracy_score * weights['accuracy'] +
            self.consistency_score * weights['consistency'] +
            self.timeliness_score * weights['timeliness'] +
            (1 - self.uncertainty_level) * weights['uncertainty'] +
            (1 - self.anomaly_score) * weights['anomaly']
        )


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    quality_metrics: QualityMetrics
    quality_flags: List[QualityFlag]
    validation_timestamp: datetime
    validation_method: str
    details: Dict[str, Any]
    recommendations: List[str]


@dataclass
class ProvenanceRecord:
    """Data provenance tracking record"""
    data_id: str
    source_system: str
    collection_timestamp: datetime
    processing_steps: List[Dict[str, Any]]
    quality_assessments: List[ValidationResult]
    lineage_hash: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Generate lineage hash for tracking"""
        if not self.lineage_hash:
            lineage_data = {
                'source': self.source_system,
                'timestamp': self.collection_timestamp.isoformat(),
                'steps': self.processing_steps
            }
            self.lineage_hash = hashlib.sha256(
                json.dumps(lineage_data, sort_keys=True).encode()
            ).hexdigest()[:16]


@dataclass
class GeologicalObservation:
    """Geological observation with quality metadata"""
    observation_id: str
    timestamp: datetime
    location: Tuple[float, float, float]  # lat, lon, elevation
    data_type: DataType
    value: Union[float, np.ndarray, Dict[str, Any]]
    uncertainty: Optional[float]
    source_system: str
    quality_flags: List[QualityFlag] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[ProvenanceRecord] = None


class DataQualityValidator:
    """Comprehensive data quality validation for geological observations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.anomaly_detectors = {}
        self.reference_datasets = {}
        self.quality_thresholds = self.config.get('quality_thresholds', {})
        self._initialize_anomaly_detectors()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for validation framework"""
        return {
            'quality_thresholds': {
                'completeness_min': 0.80,
                'accuracy_min': 0.85,
                'consistency_min': 0.75,
                'timeliness_max_hours': 24,
                'uncertainty_max': 0.20,
                'anomaly_max': 0.10
            },
            'anomaly_detection': {
                'contamination': 0.05,
                'n_estimators': 100,
                'random_state': 42
            },
            'cross_validation': {
                'min_sources': 2,
                'agreement_threshold': 0.80,
                'temporal_window_hours': 1
            }
        }
    
    def _initialize_anomaly_detectors(self):
        """Initialize anomaly detection models for different data types"""
        for data_type in DataType:
            self.anomaly_detectors[data_type] = IsolationForest(
                contamination=self.config['anomaly_detection']['contamination'],
                n_estimators=self.config['anomaly_detection']['n_estimators'],
                random_state=self.config['anomaly_detection']['random_state']
            )
    
    async def validate_observation(
        self, 
        observation: GeologicalObservation,
        reference_data: Optional[List[GeologicalObservation]] = None
    ) -> ValidationResult:
        """
        Comprehensive validation of a geological observation
        
        Args:
            observation: The observation to validate
            reference_data: Optional reference data for cross-validation
            
        Returns:
            ValidationResult with quality metrics and flags
        """
        logger.info(f"Validating observation {observation.observation_id}")
        
        # Perform individual quality assessments
        completeness = await self._assess_completeness(observation)
        accuracy = await self._assess_accuracy(observation, reference_data)
        consistency = await self._assess_consistency(observation, reference_data)
        timeliness = await self._assess_timeliness(observation)
        uncertainty = await self._quantify_uncertainty(observation)
        anomaly_score = await self._detect_anomalies(observation)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            accuracy, uncertainty, consistency
        )
        
        # Create quality metrics
        quality_metrics = QualityMetrics(
            completeness_score=completeness,
            accuracy_score=accuracy,
            consistency_score=consistency,
            timeliness_score=timeliness,
            uncertainty_level=uncertainty,
            anomaly_score=anomaly_score,
            confidence_interval=confidence_interval
        )
        
        # Determine quality flags
        quality_flags = self._determine_quality_flags(quality_metrics, observation)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_metrics, quality_flags)
        
        # Determine overall validity
        is_valid = self._determine_validity(quality_metrics)
        
        return ValidationResult(
            is_valid=is_valid,
            quality_metrics=quality_metrics,
            quality_flags=quality_flags,
            validation_timestamp=datetime.utcnow(),
            validation_method="comprehensive_geological_validation",
            details={
                'observation_id': observation.observation_id,
                'data_type': observation.data_type.value,
                'source_system': observation.source_system,
                'validation_components': {
                    'completeness': completeness,
                    'accuracy': accuracy,
                    'consistency': consistency,
                    'timeliness': timeliness,
                    'uncertainty': uncertainty,
                    'anomaly_score': anomaly_score
                }
            },
            recommendations=recommendations
        )    a
sync def _assess_completeness(self, observation: GeologicalObservation) -> float:
        """
        Assess data completeness based on expected fields and values
        
        Args:
            observation: Geological observation to assess
            
        Returns:
            Completeness score (0-1)
        """
        required_fields = ['observation_id', 'timestamp', 'location', 'value', 'source_system']
        optional_fields = ['uncertainty', 'metadata']
        
        # Check required fields
        missing_required = 0
        for field in required_fields:
            if not hasattr(observation, field) or getattr(observation, field) is None:
                missing_required += 1
        
        # Check optional fields
        missing_optional = 0
        for field in optional_fields:
            if not hasattr(observation, field) or getattr(observation, field) is None:
                missing_optional += 1
        
        # Check value completeness based on data type
        value_completeness = self._assess_value_completeness(observation)
        
        # Calculate overall completeness
        required_weight = 0.7
        optional_weight = 0.2
        value_weight = 0.1
        
        required_score = (len(required_fields) - missing_required) / len(required_fields)
        optional_score = (len(optional_fields) - missing_optional) / len(optional_fields)
        
        completeness = (
            required_score * required_weight +
            optional_score * optional_weight +
            value_completeness * value_weight
        )
        
        return max(0.0, min(1.0, completeness))
    
    def _assess_value_completeness(self, observation: GeologicalObservation) -> float:
        """Assess completeness of the observation value based on data type"""
        if observation.value is None:
            return 0.0
        
        if observation.data_type == DataType.SEISMIC:
            # For seismic data, expect waveform or amplitude values
            if isinstance(observation.value, (int, float)):
                return 1.0 if not np.isnan(observation.value) else 0.0
            elif isinstance(observation.value, np.ndarray):
                return 1.0 - (np.isnan(observation.value).sum() / len(observation.value))
        
        elif observation.data_type == DataType.GEOCHEMICAL:
            # For geochemical data, expect composition dictionary
            if isinstance(observation.value, dict):
                total_elements = len(observation.value)
                valid_elements = sum(1 for v in observation.value.values() 
                                   if v is not None and not np.isnan(float(v)))
                return valid_elements / total_elements if total_elements > 0 else 0.0
        
        # Default case for other data types
        return 1.0 if observation.value is not None else 0.0
    
    async def _assess_accuracy(
        self, 
        observation: GeologicalObservation,
        reference_data: Optional[List[GeologicalObservation]] = None
    ) -> float:
        """
        Assess accuracy against reference datasets or expected ranges
        
        Args:
            observation: Observation to assess
            reference_data: Optional reference observations for comparison
            
        Returns:
            Accuracy score (0-1)
        """
        # If no reference data, use physical plausibility checks
        if not reference_data:
            return self._assess_physical_plausibility(observation)
        
        # Find matching reference observations
        matching_refs = self._find_matching_references(observation, reference_data)
        
        if not matching_refs:
            return self._assess_physical_plausibility(observation)
        
        # Calculate accuracy against references
        accuracies = []
        for ref in matching_refs:
            accuracy = self._calculate_pairwise_accuracy(observation, ref)
            accuracies.append(accuracy)
        
        # Return weighted average accuracy
        return np.mean(accuracies) if accuracies else 0.5
    
    def _assess_physical_plausibility(self, observation: GeologicalObservation) -> float:
        """Assess if observation values are physically plausible"""
        plausibility_ranges = {
            DataType.SEISMIC: {'min': -10, 'max': 10},  # Richter scale
            DataType.VOLCANIC: {'min': 0, 'max': 8},    # VEI scale
            DataType.GEOCHEMICAL: {'min': 0, 'max': 100},  # Percentage composition
            DataType.DEFORMATION: {'min': -1000, 'max': 1000},  # mm/year
            DataType.THERMAL: {'min': -50, 'max': 1500},  # Celsius
            DataType.MAGNETIC: {'min': -100000, 'max': 100000},  # nT
            DataType.GRAVITY: {'min': -1000, 'max': 1000},  # mGal
        }
        
        if observation.data_type not in plausibility_ranges:
            return 0.8  # Default moderate confidence
        
        range_info = plausibility_ranges[observation.data_type]
        
        if isinstance(observation.value, (int, float)):
            if range_info['min'] <= observation.value <= range_info['max']:
                return 1.0
            else:
                # Calculate how far outside the range
                if observation.value < range_info['min']:
                    deviation = (range_info['min'] - observation.value) / abs(range_info['min'])
                else:
                    deviation = (observation.value - range_info['max']) / abs(range_info['max'])
                
                return max(0.0, 1.0 - min(deviation, 1.0))
        
        return 0.8  # Default for complex values
    
    def _find_matching_references(
        self, 
        observation: GeologicalObservation,
        reference_data: List[GeologicalObservation]
    ) -> List[GeologicalObservation]:
        """Find reference observations that match spatially and temporally"""
        matches = []
        spatial_threshold = 10.0  # km
        temporal_threshold = timedelta(hours=1)
        
        for ref in reference_data:
            if ref.data_type != observation.data_type:
                continue
            
            # Check spatial proximity
            distance = self._calculate_distance(observation.location, ref.location)
            if distance > spatial_threshold:
                continue
            
            # Check temporal proximity
            time_diff = abs(observation.timestamp - ref.timestamp)
            if time_diff > temporal_threshold:
                continue
            
            matches.append(ref)
        
        return matches
    
    def _calculate_distance(self, loc1: Tuple[float, float, float], loc2: Tuple[float, float, float]) -> float:
        """Calculate distance between two locations in km"""
        # Simplified haversine distance calculation
        lat1, lon1, _ = loc1
        lat2, lon2, _ = loc2
        
        R = 6371  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _calculate_pairwise_accuracy(
        self, 
        obs1: GeologicalObservation, 
        obs2: GeologicalObservation
    ) -> float:
        """Calculate accuracy between two observations"""
        if isinstance(obs1.value, (int, float)) and isinstance(obs2.value, (int, float)):
            # For scalar values, use relative error
            if obs2.value == 0:
                return 1.0 if obs1.value == 0 else 0.0
            
            relative_error = abs(obs1.value - obs2.value) / abs(obs2.value)
            return max(0.0, 1.0 - relative_error)
        
        elif isinstance(obs1.value, dict) and isinstance(obs2.value, dict):
            # For dictionary values (e.g., geochemical compositions)
            common_keys = set(obs1.value.keys()) & set(obs2.value.keys())
            if not common_keys:
                return 0.0
            
            accuracies = []
            for key in common_keys:
                val1, val2 = obs1.value[key], obs2.value[key]
                if val2 == 0:
                    acc = 1.0 if val1 == 0 else 0.0
                else:
                    relative_error = abs(val1 - val2) / abs(val2)
                    acc = max(0.0, 1.0 - relative_error)
                accuracies.append(acc)
            
            return np.mean(accuracies)
        
        return 0.5  # Default for incomparable types
    
    async def _assess_consistency(
        self, 
        observation: GeologicalObservation,
        reference_data: Optional[List[GeologicalObservation]] = None
    ) -> float:
        """
        Assess consistency with other observations from different sources
        
        Args:
            observation: Observation to assess
            reference_data: Reference observations for comparison
            
        Returns:
            Consistency score (0-1)
        """
        if not reference_data:
            return 0.8  # Default moderate consistency without references
        
        # Find observations from different sources
        different_source_obs = [
            ref for ref in reference_data 
            if (ref.source_system != observation.source_system and 
                ref.data_type == observation.data_type)
        ]
        
        if not different_source_obs:
            return 0.8  # No cross-source data available
        
        # Find spatially and temporally close observations
        nearby_obs = []
        for ref in different_source_obs:
            distance = self._calculate_distance(observation.location, ref.location)
            time_diff = abs(observation.timestamp - ref.timestamp)
            
            if distance <= 50.0 and time_diff <= timedelta(hours=6):  # Relaxed thresholds
                nearby_obs.append(ref)
        
        if not nearby_obs:
            return 0.7  # No nearby cross-source observations
        
        # Calculate consistency scores
        consistency_scores = []
        for ref in nearby_obs:
            consistency = self._calculate_pairwise_accuracy(observation, ref)
            # Weight by spatial and temporal proximity
            distance = self._calculate_distance(observation.location, ref.location)
            time_diff = abs(observation.timestamp - ref.timestamp).total_seconds() / 3600
            
            spatial_weight = max(0.1, 1.0 - distance / 50.0)
            temporal_weight = max(0.1, 1.0 - time_diff / 6.0)
            
            weighted_consistency = consistency * spatial_weight * temporal_weight
            consistency_scores.append(weighted_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.7   
 async def _assess_timeliness(self, observation: GeologicalObservation) -> float:
        """
        Assess timeliness of the observation based on collection and processing time
        
        Args:
            observation: Observation to assess
            
        Returns:
            Timeliness score (0-1)
        """
        current_time = datetime.utcnow()
        observation_age = current_time - observation.timestamp
        
        # Define acceptable age thresholds by data type
        age_thresholds = {
            DataType.SEISMIC: timedelta(minutes=5),      # Real-time seismic data
            DataType.VOLCANIC: timedelta(hours=1),       # Volcanic monitoring
            DataType.DEFORMATION: timedelta(days=1),     # GPS/InSAR data
            DataType.GEOCHEMICAL: timedelta(days=30),    # Laboratory analysis
            DataType.THERMAL: timedelta(hours=6),        # Thermal imagery
            DataType.MAGNETIC: timedelta(days=7),        # Magnetic surveys
            DataType.GRAVITY: timedelta(days=30),        # Gravity surveys
        }
        
        threshold = age_thresholds.get(observation.data_type, timedelta(hours=24))
        
        if observation_age <= threshold:
            return 1.0
        else:
            # Exponential decay of timeliness score
            decay_factor = observation_age.total_seconds() / threshold.total_seconds()
            return max(0.0, np.exp(-decay_factor + 1))
    
    async def _quantify_uncertainty(self, observation: GeologicalObservation) -> float:
        """
        Quantify uncertainty in the observation
        
        Args:
            observation: Observation to assess
            
        Returns:
            Uncertainty level (0-1, where 0 is certain, 1 is highly uncertain)
        """
        # Start with reported uncertainty if available
        if observation.uncertainty is not None:
            base_uncertainty = min(1.0, observation.uncertainty)
        else:
            # Estimate uncertainty based on data type and source
            base_uncertainty = self._estimate_base_uncertainty(observation)
        
        # Adjust uncertainty based on data quality indicators
        quality_adjustments = []
        
        # Age-based uncertainty increase
        age_hours = (datetime.utcnow() - observation.timestamp).total_seconds() / 3600
        age_uncertainty = min(0.2, age_hours / 168)  # Increase over a week
        quality_adjustments.append(age_uncertainty)
        
        # Source reliability adjustment
        source_reliability = self._get_source_reliability(observation.source_system)
        reliability_uncertainty = (1.0 - source_reliability) * 0.3
        quality_adjustments.append(reliability_uncertainty)
        
        # Metadata completeness adjustment
        metadata_completeness = len(observation.metadata) / 10  # Assume 10 ideal metadata fields
        metadata_uncertainty = (1.0 - min(1.0, metadata_completeness)) * 0.1
        quality_adjustments.append(metadata_uncertainty)
        
        # Combine uncertainties
        total_uncertainty = base_uncertainty + sum(quality_adjustments)
        return min(1.0, total_uncertainty)
    
    def _estimate_base_uncertainty(self, observation: GeologicalObservation) -> float:
        """Estimate base uncertainty for different data types"""
        base_uncertainties = {
            DataType.SEISMIC: 0.05,      # High precision instruments
            DataType.VOLCANIC: 0.10,     # Moderate uncertainty in volcanic data
            DataType.DEFORMATION: 0.02,  # High precision GPS/InSAR
            DataType.GEOCHEMICAL: 0.15,  # Laboratory analysis variability
            DataType.THERMAL: 0.08,      # Thermal sensor accuracy
            DataType.MAGNETIC: 0.03,     # Magnetometer precision
            DataType.GRAVITY: 0.05,      # Gravimeter accuracy
        }
        
        return base_uncertainties.get(observation.data_type, 0.10)
    
    def _get_source_reliability(self, source_system: str) -> float:
        """Get reliability score for data source"""
        # This would typically be loaded from a configuration or database
        source_reliabilities = {
            'IRIS': 0.95,
            'USGS': 0.95,
            'NASA': 0.90,
            'Copernicus': 0.90,
            'ASF-Hyp3': 0.85,
            'EarthChem': 0.88,
            'GEOROC': 0.85,
            'local_station': 0.70,
            'manual_entry': 0.60,
        }
        
        return source_reliabilities.get(source_system, 0.75)  # Default moderate reliability
    
    async def _detect_anomalies(self, observation: GeologicalObservation) -> float:
        """
        Detect anomalies in the observation using statistical methods
        
        Args:
            observation: Observation to assess
            
        Returns:
            Anomaly score (0-1, where 0 is normal, 1 is highly anomalous)
        """
        # Statistical anomaly detection based on data type
        if isinstance(observation.value, (int, float)):
            return self._detect_scalar_anomaly(observation)
        elif isinstance(observation.value, np.ndarray):
            return self._detect_array_anomaly(observation)
        elif isinstance(observation.value, dict):
            return self._detect_dict_anomaly(observation)
        else:
            return 0.0  # Cannot assess unknown data types
    
    def _detect_scalar_anomaly(self, observation: GeologicalObservation) -> float:
        """Detect anomalies in scalar values"""
        value = float(observation.value)
        
        # Use z-score based on expected ranges for data type
        expected_ranges = {
            DataType.SEISMIC: {'mean': 2.0, 'std': 2.0},
            DataType.VOLCANIC: {'mean': 2.0, 'std': 1.5},
            DataType.DEFORMATION: {'mean': 0.0, 'std': 10.0},
            DataType.THERMAL: {'mean': 20.0, 'std': 50.0},
            DataType.MAGNETIC: {'mean': 50000.0, 'std': 5000.0},
            DataType.GRAVITY: {'mean': 0.0, 'std': 100.0},
        }
        
        if observation.data_type not in expected_ranges:
            return 0.0
        
        expected = expected_ranges[observation.data_type]
        z_score = abs(value - expected['mean']) / expected['std']
        
        # Convert z-score to anomaly probability
        anomaly_score = min(1.0, z_score / 3.0)  # 3-sigma rule
        return anomaly_score
    
    def _detect_array_anomaly(self, observation: GeologicalObservation) -> float:
        """Detect anomalies in array data (e.g., waveforms)"""
        data = np.array(observation.value)
        
        if len(data) == 0:
            return 1.0  # Empty array is anomalous
        
        # Check for NaN or infinite values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return 0.8
        
        # Statistical tests
        anomaly_indicators = []
        
        # Check for extreme values
        z_scores = np.abs(stats.zscore(data))
        extreme_values = np.sum(z_scores > 3) / len(data)
        anomaly_indicators.append(extreme_values)
        
        # Check for constant values (sensor failure)
        if np.std(data) < 1e-10:
            anomaly_indicators.append(0.9)
        
        # Check for sudden jumps
        if len(data) > 1:
            diffs = np.abs(np.diff(data))
            jump_threshold = np.std(data) * 5
            jumps = np.sum(diffs > jump_threshold) / len(diffs)
            anomaly_indicators.append(jumps)
        
        return min(1.0, np.mean(anomaly_indicators))
    
    def _detect_dict_anomaly(self, observation: GeologicalObservation) -> float:
        """Detect anomalies in dictionary data (e.g., geochemical compositions)"""
        data = observation.value
        
        if not isinstance(data, dict) or not data:
            return 1.0
        
        anomaly_indicators = []
        
        # Check for negative values in compositions
        negative_values = sum(1 for v in data.values() if isinstance(v, (int, float)) and v < 0)
        anomaly_indicators.append(negative_values / len(data))
        
        # Check for unrealistic total composition (should sum to ~100% for oxides)
        if observation.data_type == DataType.GEOCHEMICAL:
            numeric_values = [v for v in data.values() if isinstance(v, (int, float))]
            if numeric_values:
                total = sum(numeric_values)
                # Expect total between 95-105% for major oxides
                if total < 95 or total > 105:
                    anomaly_indicators.append(min(1.0, abs(total - 100) / 50))
                else:
                    anomaly_indicators.append(0.0)
        
        return min(1.0, np.mean(anomaly_indicators)) if anomaly_indicators else 0.0
    
    def _calculate_confidence_interval(
        self, 
        accuracy: float, 
        uncertainty: float, 
        consistency: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for quality assessment"""
        # Base confidence on accuracy and consistency
        base_confidence = (accuracy + consistency) / 2
        
        # Adjust for uncertainty
        confidence_width = uncertainty * 0.5
        
        lower_bound = max(0.0, base_confidence - confidence_width)
        upper_bound = min(1.0, base_confidence + confidence_width)
        
        return (lower_bound, upper_bound)
    
    def _determine_quality_flags(
        self, 
        metrics: QualityMetrics, 
        observation: GeologicalObservation
    ) -> List[QualityFlag]:
        """Determine quality flags based on metrics"""
        flags = []
        
        # Check individual metrics against thresholds
        thresholds = self.quality_thresholds
        
        if metrics.completeness_score < thresholds.get('completeness_min', 0.8):
            flags.append(QualityFlag.MISSING)
        
        if metrics.accuracy_score < thresholds.get('accuracy_min', 0.85):
            flags.append(QualityFlag.SUSPICIOUS)
        
        if metrics.consistency_score < thresholds.get('consistency_min', 0.75):
            flags.append(QualityFlag.INCONSISTENT)
        
        if metrics.uncertainty_level > thresholds.get('uncertainty_max', 0.2):
            flags.append(QualityFlag.UNCERTAIN)
        
        if metrics.anomaly_score > thresholds.get('anomaly_max', 0.1):
            flags.append(QualityFlag.OUTLIER)
        
        # Check for interpolated data
        if 'interpolated' in observation.metadata:
            flags.append(QualityFlag.INTERPOLATED)
        
        # If no issues found, mark as valid
        if not flags:
            flags.append(QualityFlag.VALID)
        elif metrics.overall_quality < 0.5:
            flags.append(QualityFlag.INVALID)
        
        return flags
    
    def _generate_recommendations(
        self, 
        metrics: QualityMetrics, 
        flags: List[QualityFlag]
    ) -> List[str]:
        """Generate recommendations for improving data quality"""
        recommendations = []
        
        if QualityFlag.MISSING in flags:
            recommendations.append("Complete missing data fields or metadata")
        
        if QualityFlag.SUSPICIOUS in flags:
            recommendations.append("Verify data against additional sources")
        
        if QualityFlag.INCONSISTENT in flags:
            recommendations.append("Cross-validate with other data sources")
        
        if QualityFlag.UNCERTAIN in flags:
            recommendations.append("Obtain additional measurements to reduce uncertainty")
        
        if QualityFlag.OUTLIER in flags:
            recommendations.append("Investigate potential measurement errors or unusual conditions")
        
        if QualityFlag.INTERPOLATED in flags:
            recommendations.append("Consider obtaining direct measurements if possible")
        
        if metrics.timeliness_score < 0.7:
            recommendations.append("Update data more frequently for better timeliness")
        
        if metrics.overall_quality < 0.6:
            recommendations.append("Consider excluding this data from critical analyses")
        
        return recommendations
    
    def _determine_validity(self, metrics: QualityMetrics) -> bool:
        """Determine if observation is valid based on quality metrics"""
        # Minimum thresholds for validity
        min_overall_quality = 0.6
        min_accuracy = 0.5
        max_anomaly_score = 0.5
        
        return (
            metrics.overall_quality >= min_overall_quality and
            metrics.accuracy_score >= min_accuracy and
            metrics.anomaly_score <= max_anomaly_score
        )