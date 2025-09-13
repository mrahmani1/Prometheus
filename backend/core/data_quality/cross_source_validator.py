"""
Cross-Source Validation and Consistency Checking

This module implements cross-source validation algorithms that compare geological
observations from multiple data sources to identify inconsistencies and validate
data reliability across different systems.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .validation_framework import (
    GeologicalObservation, DataType, QualityFlag, ValidationResult,
    QualityMetrics, ProvenanceRecord
)

logger = logging.getLogger(__name__)


@dataclass
class CrossSourceComparison:
    """Result of cross-source validation comparison"""
    primary_observation: GeologicalObservation
    reference_observations: List[GeologicalObservation]
    agreement_score: float
    disagreement_details: Dict[str, Any]
    consensus_value: Optional[Any]
    confidence_level: float
    validation_timestamp: datetime


@dataclass
class ConsensusResult:
    """Result of consensus analysis across multiple sources"""
    consensus_value: Any
    confidence_score: float
    participating_sources: List[str]
    outlier_sources: List[str]
    statistical_metrics: Dict[str, float]
    uncertainty_bounds: Tuple[float, float]


class CrossSourceValidator:
    """Cross-source validation and consistency checking for geological data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.source_reliability_scores = {}
        self.historical_agreements = defaultdict(list)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for cross-source validation"""
        return {
            'spatial_tolerance_km': 10.0,
            'temporal_tolerance_hours': 2.0,
            'min_sources_for_consensus': 3,
            'agreement_threshold': 0.80,
            'outlier_detection_threshold': 2.0,  # Standard deviations
            'confidence_weights': {
                'source_reliability': 0.3,
                'temporal_proximity': 0.2,
                'spatial_proximity': 0.2,
                'statistical_agreement': 0.3
            }
        }
    
    async def validate_cross_source_consistency(
        self,
        target_observation: GeologicalObservation,
        reference_observations: List[GeologicalObservation]
    ) -> CrossSourceComparison:
        """
        Validate consistency of an observation against multiple reference sources
        
        Args:
            target_observation: Primary observation to validate
            reference_observations: List of reference observations from other sources
            
        Returns:
            CrossSourceComparison with validation results
        """
        logger.info(f"Cross-source validation for observation {target_observation.observation_id}")
        
        # Filter relevant reference observations
        relevant_refs = self._filter_relevant_observations(
            target_observation, reference_observations
        )
        
        if len(relevant_refs) < 2:
            logger.warning("Insufficient reference observations for cross-source validation")
            return CrossSourceComparison(
                primary_observation=target_observation,
                reference_observations=relevant_refs,
                agreement_score=0.5,  # Neutral score when insufficient data
                disagreement_details={'reason': 'insufficient_references'},
                consensus_value=None,
                confidence_level=0.3,
                validation_timestamp=datetime.utcnow()
            )
        
        # Calculate agreement scores
        agreement_scores = []
        disagreement_details = {}
        
        for ref in relevant_refs:
            agreement = await self._calculate_agreement_score(target_observation, ref)
            agreement_scores.append(agreement)
            
            if agreement < self.config['agreement_threshold']:
                disagreement_details[ref.source_system] = {
                    'agreement_score': agreement,
                    'value_difference': self._calculate_value_difference(
                        target_observation.value, ref.value
                    ),
                    'spatial_distance_km': self._calculate_distance(
                        target_observation.location, ref.location
                    ),
                    'temporal_difference_hours': abs(
                        target_observation.timestamp - ref.timestamp
                    ).total_seconds() / 3600
                }
        
        # Calculate overall agreement
        overall_agreement = np.mean(agreement_scores) if agreement_scores else 0.0
        
        # Generate consensus value
        consensus_result = await self._generate_consensus(
            target_observation, relevant_refs
        )
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(
            target_observation, relevant_refs, agreement_scores
        )
        
        return CrossSourceComparison(
            primary_observation=target_observation,
            reference_observations=relevant_refs,
            agreement_score=overall_agreement,
            disagreement_details=disagreement_details,
            consensus_value=consensus_result.consensus_value if consensus_result else None,
            confidence_level=confidence_level,
            validation_timestamp=datetime.utcnow()
        )
    
    def _filter_relevant_observations(
        self,
        target: GeologicalObservation,
        references: List[GeologicalObservation]
    ) -> List[GeologicalObservation]:
        """Filter reference observations that are relevant for comparison"""
        relevant = []
        
        spatial_tolerance = self.config['spatial_tolerance_km']
        temporal_tolerance = timedelta(hours=self.config['temporal_tolerance_hours'])
        
        for ref in references:
            # Must be same data type
            if ref.data_type != target.data_type:
                continue
            
            # Must be from different source
            if ref.source_system == target.source_system:
                continue
            
            # Check spatial proximity
            distance = self._calculate_distance(target.location, ref.location)
            if distance > spatial_tolerance:
                continue
            
            # Check temporal proximity
            time_diff = abs(target.timestamp - ref.timestamp)
            if time_diff > temporal_tolerance:
                continue
            
            relevant.append(ref)
        
        return relevant
    
    def _calculate_distance(
        self, 
        loc1: Tuple[float, float, float], 
        loc2: Tuple[float, float, float]
    ) -> float:
        """Calculate distance between two locations in km"""
        lat1, lon1, _ = loc1
        lat2, lon2, _ = loc2
        
        # Haversine formula
        R = 6371  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    async def _calculate_agreement_score(
        self,
        obs1: GeologicalObservation,
        obs2: GeologicalObservation
    ) -> float:
        """Calculate agreement score between two observations"""
        if isinstance(obs1.value, (int, float)) and isinstance(obs2.value, (int, float)):
            return self._calculate_scalar_agreement(obs1.value, obs2.value, obs1.data_type)
        
        elif isinstance(obs1.value, np.ndarray) and isinstance(obs2.value, np.ndarray):
            return self._calculate_array_agreement(obs1.value, obs2.value)
        
        elif isinstance(obs1.value, dict) and isinstance(obs2.value, dict):
            return self._calculate_dict_agreement(obs1.value, obs2.value)
        
        else:
            return 0.0  # Cannot compare different types
    
    def _calculate_scalar_agreement(self, val1: float, val2: float, data_type: DataType) -> float:
        """Calculate agreement between scalar values"""
        # Define tolerance levels by data type
        tolerance_levels = {
            DataType.SEISMIC: 0.1,       # 10% tolerance for seismic magnitudes
            DataType.VOLCANIC: 0.5,      # 0.5 VEI units tolerance
            DataType.DEFORMATION: 2.0,   # 2mm tolerance for deformation
            DataType.THERMAL: 5.0,       # 5°C tolerance for temperature
            DataType.GEOCHEMICAL: 5.0,   # 5% tolerance for compositions
            DataType.MAGNETIC: 100.0,    # 100nT tolerance for magnetic field
            DataType.GRAVITY: 10.0,      # 10mGal tolerance for gravity
        }
        
        tolerance = tolerance_levels.get(data_type, abs(val2) * 0.1)  # Default 10%
        
        if tolerance == 0:
            return 1.0 if val1 == val2 else 0.0
        
        difference = abs(val1 - val2)
        agreement = max(0.0, 1.0 - (difference / tolerance))
        
        return agreement
    
    def _calculate_array_agreement(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate agreement between array values (e.g., waveforms)"""
        if len(arr1) != len(arr2):
            # Resample to same length if different
            min_len = min(len(arr1), len(arr2))
            arr1 = arr1[:min_len]
            arr2 = arr2[:min_len]
        
        if len(arr1) == 0:
            return 0.0
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(arr1, arr2)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Calculate normalized RMSE
        rmse = np.sqrt(np.mean((arr1 - arr2)**2))
        max_range = max(np.ptp(arr1), np.ptp(arr2))
        normalized_rmse = rmse / max_range if max_range > 0 else 1.0
        
        # Combine correlation and RMSE
        agreement = (abs(correlation) + (1.0 - min(1.0, normalized_rmse))) / 2.0
        
        return max(0.0, min(1.0, agreement))
    
    def _calculate_dict_agreement(self, dict1: Dict, dict2: Dict) -> float:
        """Calculate agreement between dictionary values"""
        common_keys = set(dict1.keys()) & set(dict2.keys())
        
        if not common_keys:
            return 0.0
        
        agreements = []
        for key in common_keys:
            val1, val2 = dict1[key], dict2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Use relative tolerance for numeric values
                tolerance = max(abs(val2) * 0.1, 0.01)  # 10% or minimum 0.01
                difference = abs(val1 - val2)
                agreement = max(0.0, 1.0 - (difference / tolerance))
                agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.0
    
    def _calculate_value_difference(self, val1: Any, val2: Any) -> Dict[str, Any]:
        """Calculate detailed value differences for reporting"""
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return {
                'type': 'scalar',
                'absolute_difference': abs(val1 - val2),
                'relative_difference': abs(val1 - val2) / abs(val2) if val2 != 0 else float('inf'),
                'values': {'primary': val1, 'reference': val2}
            }
        
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            return {
                'type': 'array',
                'rmse': np.sqrt(np.mean((val1 - val2)**2)) if len(val1) == len(val2) else None,
                'correlation': np.corrcoef(val1, val2)[0, 1] if len(val1) == len(val2) else None,
                'length_difference': len(val1) - len(val2)
            }
        
        elif isinstance(val1, dict) and isinstance(val2, dict):
            common_keys = set(val1.keys()) & set(val2.keys())
            differences = {}
            for key in common_keys:
                if isinstance(val1[key], (int, float)) and isinstance(val2[key], (int, float)):
                    differences[key] = abs(val1[key] - val2[key])
            
            return {
                'type': 'dictionary',
                'common_keys': list(common_keys),
                'key_differences': differences,
                'missing_in_primary': list(set(val2.keys()) - set(val1.keys())),
                'missing_in_reference': list(set(val1.keys()) - set(val2.keys()))
            }
        
        else:
            return {
                'type': 'incompatible',
                'primary_type': type(val1).__name__,
                'reference_type': type(val2).__name__
            }
    
    async def _generate_consensus(
        self,
        target: GeologicalObservation,
        references: List[GeologicalObservation]
    ) -> Optional[ConsensusResult]:
        """Generate consensus value from multiple observations"""
        if len(references) < self.config['min_sources_for_consensus']:
            return None
        
        all_observations = [target] + references
        
        if isinstance(target.value, (int, float)):
            return self._generate_scalar_consensus(all_observations)
        elif isinstance(target.value, dict):
            return self._generate_dict_consensus(all_observations)
        else:
            return None  # Cannot generate consensus for other types
    
    def _generate_scalar_consensus(
        self, 
        observations: List[GeologicalObservation]
    ) -> ConsensusResult:
        """Generate consensus for scalar values"""
        values = [float(obs.value) for obs in observations]
        sources = [obs.source_system for obs in observations]
        
        # Remove outliers using IQR method
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_indices = [
            i for i, val in enumerate(values) 
            if val < lower_bound or val > upper_bound
        ]
        
        clean_values = [val for i, val in enumerate(values) if i not in outlier_indices]
        clean_sources = [src for i, src in enumerate(sources) if i not in outlier_indices]
        outlier_sources = [sources[i] for i in outlier_indices]
        
        if not clean_values:
            clean_values = values  # Use all if all are outliers
            clean_sources = sources
            outlier_sources = []
        
        # Calculate consensus statistics
        consensus_value = np.median(clean_values)  # Use median for robustness
        std_dev = np.std(clean_values)
        confidence_score = max(0.0, 1.0 - (std_dev / abs(consensus_value))) if consensus_value != 0 else 0.5
        
        # Calculate uncertainty bounds
        uncertainty = 1.96 * std_dev / np.sqrt(len(clean_values))  # 95% confidence interval
        uncertainty_bounds = (consensus_value - uncertainty, consensus_value + uncertainty)
        
        statistical_metrics = {
            'mean': np.mean(clean_values),
            'median': consensus_value,
            'std_dev': std_dev,
            'min': np.min(clean_values),
            'max': np.max(clean_values),
            'count': len(clean_values)
        }
        
        return ConsensusResult(
            consensus_value=consensus_value,
            confidence_score=confidence_score,
            participating_sources=clean_sources,
            outlier_sources=outlier_sources,
            statistical_metrics=statistical_metrics,
            uncertainty_bounds=uncertainty_bounds
        )
    
    def _generate_dict_consensus(
        self, 
        observations: List[GeologicalObservation]
    ) -> ConsensusResult:
        """Generate consensus for dictionary values (e.g., geochemical compositions)"""
        all_dicts = [obs.value for obs in observations if isinstance(obs.value, dict)]
        sources = [obs.source_system for obs in observations if isinstance(obs.value, dict)]
        
        if not all_dicts:
            return None
        
        # Find common keys across all dictionaries
        common_keys = set(all_dicts[0].keys())
        for d in all_dicts[1:]:
            common_keys &= set(d.keys())
        
        consensus_dict = {}
        statistical_metrics = {}
        outlier_sources = []
        
        for key in common_keys:
            values = [d[key] for d in all_dicts if isinstance(d[key], (int, float))]
            
            if values:
                # Generate consensus for this key
                median_val = np.median(values)
                std_val = np.std(values)
                
                consensus_dict[key] = median_val
                statistical_metrics[key] = {
                    'mean': np.mean(values),
                    'median': median_val,
                    'std_dev': std_val,
                    'count': len(values)
                }
        
        # Calculate overall confidence
        if statistical_metrics:
            relative_stds = [
                metrics['std_dev'] / abs(metrics['median']) 
                for metrics in statistical_metrics.values() 
                if metrics['median'] != 0
            ]
            avg_relative_std = np.mean(relative_stds) if relative_stds else 0.5
            confidence_score = max(0.0, 1.0 - avg_relative_std)
        else:
            confidence_score = 0.0
        
        return ConsensusResult(
            consensus_value=consensus_dict,
            confidence_score=confidence_score,
            participating_sources=sources,
            outlier_sources=outlier_sources,
            statistical_metrics=statistical_metrics,
            uncertainty_bounds=(0.0, 1.0)  # Placeholder for dict uncertainty
        )
    
    def _calculate_confidence_level(
        self,
        target: GeologicalObservation,
        references: List[GeologicalObservation],
        agreement_scores: List[float]
    ) -> float:
        """Calculate confidence level for cross-source validation"""
        weights = self.config['confidence_weights']
        
        # Source reliability component
        source_reliability = self._get_source_reliability(target.source_system)
        
        # Temporal proximity component (closer in time = higher confidence)
        temporal_proximities = [
            1.0 - min(1.0, abs(target.timestamp - ref.timestamp).total_seconds() / 
                     (self.config['temporal_tolerance_hours'] * 3600))
            for ref in references
        ]
        avg_temporal_proximity = np.mean(temporal_proximities) if temporal_proximities else 0.0
        
        # Spatial proximity component
        spatial_proximities = [
            1.0 - min(1.0, self._calculate_distance(target.location, ref.location) / 
                     self.config['spatial_tolerance_km'])
            for ref in references
        ]
        avg_spatial_proximity = np.mean(spatial_proximities) if spatial_proximities else 0.0
        
        # Statistical agreement component
        avg_agreement = np.mean(agreement_scores) if agreement_scores else 0.0
        
        # Weighted confidence calculation
        confidence = (
            source_reliability * weights['source_reliability'] +
            avg_temporal_proximity * weights['temporal_proximity'] +
            avg_spatial_proximity * weights['spatial_proximity'] +
            avg_agreement * weights['statistical_agreement']
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _get_source_reliability(self, source_system: str) -> float:
        """Get reliability score for a data source"""
        # This would typically be loaded from configuration or learned from historical data
        default_reliabilities = {
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
        
        return default_reliabilities.get(source_system, 0.75)
    
    async def update_source_reliability(
        self, 
        source_system: str, 
        validation_results: List[ValidationResult]
    ):
        """Update source reliability based on validation history"""
        if not validation_results:
            return
        
        # Calculate average quality metrics
        avg_accuracy = np.mean([r.quality_metrics.accuracy_score for r in validation_results])
        avg_consistency = np.mean([r.quality_metrics.consistency_score for r in validation_results])
        
        # Update reliability score
        new_reliability = (avg_accuracy + avg_consistency) / 2.0
        
        # Smooth update with existing score
        if source_system in self.source_reliability_scores:
            existing_score = self.source_reliability_scores[source_system]
            self.source_reliability_scores[source_system] = (
                0.7 * existing_score + 0.3 * new_reliability
            )
        else:
            self.source_reliability_scores[source_system] = new_reliability
        
        logger.info(f"Updated reliability for {source_system}: {self.source_reliability_scores[source_system]:.3f}")