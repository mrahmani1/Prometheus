"""
Data aggregation and downsampling for multi-resolution analysis
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from collections import defaultdict

from .schemas import StandardizedDataFormat, TemporalResolution, DataType


logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of data aggregation"""
    aggregated_data: List[StandardizedDataFormat]
    original_count: int
    aggregated_count: int
    aggregation_method: str
    temporal_resolution: TemporalResolution
    quality_preservation: float


@dataclass
class AggregationConfig:
    """Configuration for data aggregation"""
    target_resolution: TemporalResolution
    aggregation_method: str = "mean"
    preserve_extremes: bool = True
    include_statistics: bool = True
    weight_by_quality: bool = True
    min_points_per_bin: int = 1
    uncertainty_propagation: str = "quadrature"  # 'quadrature', 'max', 'mean'


class AggregationEngine:
    """
    Handles data aggregation and downsampling for multi-resolution geological analysis
    """
    
    def __init__(self):
        self.aggregation_methods = {
            'mean': self._mean_aggregation,
            'median': self._median_aggregation,
            'max': self._max_aggregation,
            'min': self._min_aggregation,
            'sum': self._sum_aggregation,
            'std': self._std_aggregation,
            'rms': self._rms_aggregation,
            'percentile_95': lambda x, w=None: self._percentile_aggregation(x, 95, w),
            'percentile_05': lambda x, w=None: self._percentile_aggregation(x, 5, w),
            'first': self._first_aggregation,
            'last': self._last_aggregation,
            'count': self._count_aggregation,
            'range': self._range_aggregation,
            'weighted_mean': self._weighted_mean_aggregation,
        }
        
        self.resolution_seconds = {
            TemporalResolution.MILLISECOND: 0.001,
            TemporalResolution.SECOND: 1.0,
            TemporalResolution.MINUTE: 60.0,
            TemporalResolution.HOUR: 3600.0,
            TemporalResolution.DAY: 86400.0,
            TemporalResolution.WEEK: 604800.0,
            TemporalResolution.MONTH: 2629746.0,
            TemporalResolution.YEAR: 31556952.0,
        }
        
        # Data type specific aggregation preferences
        self.datatype_preferences = {
            DataType.SEISMIC: {
                'amplitude': 'rms',
                'velocity': 'mean',
                'acceleration': 'rms',
                'frequency': 'mean',
                'magnitude': 'max',
                'energy': 'sum',
            },
            DataType.VOLCANIC: {
                'deformation_rate': 'mean',
                'gas_emission': 'mean',
                'temperature': 'max',
                'seismic_energy': 'sum',
                'tremor_amplitude': 'rms',
            },
            DataType.GEOCHEMICAL: {
                'concentration': 'mean',
                'isotope_ratio': 'mean',
                'temperature': 'mean',
                'pressure': 'mean',
            },
            DataType.DEFORMATION: {
                'displacement': 'mean',
                'velocity': 'mean',
                'strain': 'mean',
                'strain_rate': 'mean',
            },
            DataType.MTSAR: {
                'displacement': 'mean',
                'coherence': 'mean',
                'phase': 'mean',
            },
            DataType.GPS: {
                'displacement_east': 'mean',
                'displacement_north': 'mean',
                'displacement_up': 'mean',
                'velocity_east': 'mean',
                'velocity_north': 'mean',
                'velocity_up': 'mean',
            },
        }
    
    def aggregate_temporal_data(
        self,
        data_list: List[StandardizedDataFormat],
        config: AggregationConfig
    ) -> AggregationResult:
        """
        Aggregate temporal data to target resolution
        
        Args:
            data_list: Data to aggregate
            config: Aggregation configuration
            
        Returns:
            Aggregation result with downsampled data
        """
        if not data_list:
            return AggregationResult(
                aggregated_data=[],
                original_count=0,
                aggregated_count=0,
                aggregation_method=config.aggregation_method,
                temporal_resolution=config.target_resolution,
                quality_preservation=0.0
            )
        
        # Sort data by timestamp
        sorted_data = sorted(data_list, key=lambda x: x.timestamp)
        
        # Create time bins
        time_bins = self._create_time_bins(sorted_data, config.target_resolution)
        
        # Group data into bins
        binned_data = self._bin_data_by_time(sorted_data, time_bins)
        
        # Aggregate each bin
        aggregated_data = []
        quality_scores = []
        
        for bin_start, bin_data in binned_data.items():
            if len(bin_data) >= config.min_points_per_bin:
                try:
                    aggregated_point = self._aggregate_bin(bin_data, config)
                    if aggregated_point:
                        aggregated_data.append(aggregated_point)
                        
                        # Calculate quality preservation
                        quality_score = self._calculate_aggregation_quality(bin_data, aggregated_point)
                        quality_scores.append(quality_score)
                        
                except Exception as e:
                    logger.error(f"Failed to aggregate bin starting at {bin_start}: {e}")
        
        # Calculate overall quality preservation
        overall_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return AggregationResult(
            aggregated_data=aggregated_data,
            original_count=len(data_list),
            aggregated_count=len(aggregated_data),
            aggregation_method=config.aggregation_method,
            temporal_resolution=config.target_resolution,
            quality_preservation=overall_quality
        )
    
    def _create_time_bins(
        self,
        data_list: List[StandardizedDataFormat],
        target_resolution: TemporalResolution
    ) -> List[datetime]:
        """Create time bins for aggregation"""
        if not data_list:
            return []
        
        start_time = data_list[0].timestamp
        end_time = data_list[-1].timestamp
        
        # Get bin size in seconds
        bin_size_seconds = self.resolution_seconds[target_resolution]
        
        # Align start time to resolution boundary
        if target_resolution == TemporalResolution.HOUR:
            aligned_start = start_time.replace(minute=0, second=0, microsecond=0)
        elif target_resolution == TemporalResolution.DAY:
            aligned_start = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif target_resolution == TemporalResolution.WEEK:
            # Align to Monday
            days_since_monday = start_time.weekday()
            aligned_start = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
            aligned_start -= timedelta(days=days_since_monday)
        elif target_resolution == TemporalResolution.MONTH:
            aligned_start = start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif target_resolution == TemporalResolution.YEAR:
            aligned_start = start_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            aligned_start = start_time
        
        # Create bins
        bins = []
        current_time = aligned_start
        
        while current_time <= end_time:
            bins.append(current_time)
            
            if target_resolution == TemporalResolution.MONTH:
                # Handle month boundaries
                if current_time.month == 12:
                    current_time = current_time.replace(year=current_time.year + 1, month=1)
                else:
                    current_time = current_time.replace(month=current_time.month + 1)
            elif target_resolution == TemporalResolution.YEAR:
                current_time = current_time.replace(year=current_time.year + 1)
            else:
                current_time += timedelta(seconds=bin_size_seconds)
        
        return bins
    
    def _bin_data_by_time(
        self,
        data_list: List[StandardizedDataFormat],
        time_bins: List[datetime]
    ) -> Dict[datetime, List[StandardizedDataFormat]]:
        """Group data into time bins"""
        binned_data = defaultdict(list)
        
        for data_point in data_list:
            # Find appropriate bin
            bin_index = self._find_time_bin(data_point.timestamp, time_bins)
            if bin_index is not None and bin_index < len(time_bins):
                binned_data[time_bins[bin_index]].append(data_point)
        
        return dict(binned_data)
    
    def _find_time_bin(self, timestamp: datetime, time_bins: List[datetime]) -> Optional[int]:
        """Find the appropriate time bin for a timestamp"""
        for i in range(len(time_bins) - 1):
            if time_bins[i] <= timestamp < time_bins[i + 1]:
                return i
        
        # Handle last bin
        if len(time_bins) > 0 and timestamp >= time_bins[-1]:
            return len(time_bins) - 1
        
        return None
    
    def _aggregate_bin(
        self,
        bin_data: List[StandardizedDataFormat],
        config: AggregationConfig
    ) -> Optional[StandardizedDataFormat]:
        """Aggregate data within a single time bin"""
        if not bin_data:
            return None
        
        # Use first data point as template
        template = bin_data[0]
        aggregated = template.copy(deep=True)
        
        # Set aggregated timestamp (bin center or representative time)
        timestamps = [d.timestamp for d in bin_data]
        if config.aggregation_method in ['first']:
            aggregated.timestamp = min(timestamps)
        elif config.aggregation_method in ['last']:
            aggregated.timestamp = max(timestamps)
        else:
            # Use mean timestamp
            total_seconds = sum((ts - timestamps[0]).total_seconds() for ts in timestamps)
            mean_offset = total_seconds / len(timestamps)
            aggregated.timestamp = timestamps[0] + timedelta(seconds=mean_offset)
        
        # Generate new data ID
        aggregated.data_id = f"{template.data_id}_agg_{aggregated.timestamp.isoformat()}"
        
        # Aggregate values
        aggregated.values = {}
        aggregated.value_uncertainty = {}
        
        # Get data type specific preferences
        data_type_prefs = self.datatype_preferences.get(template.data_type, {})
        
        for key in template.values.keys():
            # Determine aggregation method for this field
            field_method = data_type_prefs.get(key, config.aggregation_method)
            
            # Extract values and weights
            values = []
            weights = []
            uncertainties = []
            
            for data_point in bin_data:
                if key in data_point.values:
                    value = data_point.values[key]
                    if isinstance(value, (int, float)):
                        values.append(value)
                        
                        # Calculate weight based on quality
                        if config.weight_by_quality and data_point.quality_metrics:
                            weight = data_point.quality_metrics.overall_score
                        else:
                            weight = 1.0
                        weights.append(weight)
                        
                        # Extract uncertainty
                        if data_point.value_uncertainty and key in data_point.value_uncertainty:
                            uncertainties.append(data_point.value_uncertainty[key])
                        else:
                            uncertainties.append(0.0)
            
            if values:
                # Apply aggregation method
                if field_method in self.aggregation_methods:
                    if field_method == 'weighted_mean' and config.weight_by_quality:
                        aggregated_value = self.aggregation_methods[field_method](values, weights)
                    else:
                        aggregated_value = self.aggregation_methods[field_method](values)
                    
                    aggregated.values[key] = aggregated_value
                    
                    # Propagate uncertainty
                    if uncertainties:
                        propagated_uncertainty = self._propagate_uncertainty(
                            uncertainties, weights, config.uncertainty_propagation
                        )
                        aggregated.value_uncertainty[key] = propagated_uncertainty
        
        # Add statistics if requested
        if config.include_statistics:
            aggregated.metadata['aggregation_statistics'] = self._calculate_bin_statistics(bin_data)
        
        # Update quality metrics
        aggregated.quality_metrics = self._aggregate_quality_metrics(bin_data)
        
        # Add processing history
        processing_step = {
            'step_id': f"temporal_aggregation_{datetime.now().isoformat()}",
            'timestamp': datetime.now(),
            'operation': 'temporal_aggregation',
            'parameters': {
                'method': config.aggregation_method,
                'target_resolution': config.target_resolution.value,
                'bin_size': len(bin_data),
                'weight_by_quality': config.weight_by_quality,
                'preserve_extremes': config.preserve_extremes
            },
            'input_hash': str(hash(tuple(d.data_id for d in bin_data))),
            'output_hash': str(hash(aggregated.data_id)),
            'processor_version': '1.0.0'
        }
        
        aggregated.processing_history.append(processing_step)
        
        # Add quality flags
        aggregated.quality_flags.append("temporally_aggregated")
        aggregated.quality_flags.append(f"aggregation_method:{config.aggregation_method}")
        aggregated.quality_flags.append(f"bin_size:{len(bin_data)}")
        
        return aggregated
    
    def _propagate_uncertainty(
        self,
        uncertainties: List[float],
        weights: List[float],
        method: str
    ) -> float:
        """Propagate uncertainty through aggregation"""
        if not uncertainties:
            return 0.0
        
        uncertainties = np.array(uncertainties)
        weights = np.array(weights)
        
        if method == 'quadrature':
            # Quadrature sum weighted by normalized weights
            normalized_weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
            return np.sqrt(np.sum((normalized_weights * uncertainties) ** 2))
        
        elif method == 'max':
            return np.max(uncertainties)
        
        elif method == 'mean':
            return np.mean(uncertainties)
        
        else:
            # Default to quadrature
            return np.sqrt(np.sum(uncertainties ** 2) / len(uncertainties))
    
    def _calculate_bin_statistics(self, bin_data: List[StandardizedDataFormat]) -> Dict[str, Any]:
        """Calculate statistics for aggregated bin"""
        return {
            'count': len(bin_data),
            'time_span_seconds': (max(d.timestamp for d in bin_data) - 
                                min(d.timestamp for d in bin_data)).total_seconds(),
            'quality_range': {
                'min': min(d.quality_metrics.overall_score for d in bin_data 
                          if d.quality_metrics),
                'max': max(d.quality_metrics.overall_score for d in bin_data 
                          if d.quality_metrics),
            } if any(d.quality_metrics for d in bin_data) else None,
        }
    
    def _aggregate_quality_metrics(self, bin_data: List[StandardizedDataFormat]):
        """Aggregate quality metrics from bin data"""
        from .schemas import QualityMetrics
        
        quality_data = [d.quality_metrics for d in bin_data if d.quality_metrics]
        
        if not quality_data:
            return None
        
        return QualityMetrics(
            completeness=np.mean([q.completeness for q in quality_data]),
            accuracy=np.mean([q.accuracy for q in quality_data]),
            consistency=np.mean([q.consistency for q in quality_data]),
            timeliness=np.mean([q.timeliness for q in quality_data]),
            overall_score=np.mean([q.overall_score for q in quality_data]),
            flags=[f"aggregated_from_{len(bin_data)}_points"]
        )
    
    def _calculate_aggregation_quality(
        self,
        original_data: List[StandardizedDataFormat],
        aggregated_data: StandardizedDataFormat
    ) -> float:
        """Calculate quality preservation score for aggregation"""
        if not original_data:
            return 0.0
        
        # Base quality on data count and temporal coverage
        count_factor = min(1.0, len(original_data) / 10)  # More data = better quality
        
        # Quality based on original data quality
        original_qualities = [d.quality_metrics.overall_score for d in original_data 
                            if d.quality_metrics]
        if original_qualities:
            quality_factor = np.mean(original_qualities)
        else:
            quality_factor = 0.5
        
        # Temporal coverage factor
        if len(original_data) > 1:
            time_span = (max(d.timestamp for d in original_data) - 
                        min(d.timestamp for d in original_data)).total_seconds()
            coverage_factor = min(1.0, time_span / 3600)  # Up to 1 hour gets full score
        else:
            coverage_factor = 0.5
        
        return count_factor * quality_factor * coverage_factor
    
    # Aggregation method implementations
    def _mean_aggregation(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Calculate weighted or unweighted mean"""
        if weights:
            return np.average(values, weights=weights)
        return np.mean(values)
    
    def _median_aggregation(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Calculate median (weights ignored for median)"""
        return np.median(values)
    
    def _max_aggregation(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Calculate maximum value"""
        return np.max(values)
    
    def _min_aggregation(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Calculate minimum value"""
        return np.min(values)
    
    def _sum_aggregation(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Calculate sum"""
        return np.sum(values)
    
    def _std_aggregation(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Calculate standard deviation"""
        return np.std(values)
    
    def _rms_aggregation(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Calculate root mean square"""
        return np.sqrt(np.mean(np.array(values) ** 2))
    
    def _percentile_aggregation(self, values: List[float], percentile: float, weights: Optional[List[float]] = None) -> float:
        """Calculate percentile"""
        return np.percentile(values, percentile)
    
    def _first_aggregation(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Return first value"""
        return values[0] if values else 0.0
    
    def _last_aggregation(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Return last value"""
        return values[-1] if values else 0.0
    
    def _count_aggregation(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Return count of values"""
        return float(len(values))
    
    def _range_aggregation(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Calculate range (max - min)"""
        return np.max(values) - np.min(values) if values else 0.0
    
    def _weighted_mean_aggregation(self, values: List[float], weights: List[float]) -> float:
        """Calculate weighted mean"""
        if not weights or len(weights) != len(values):
            return np.mean(values)
        return np.average(values, weights=weights)
    
    def create_multi_resolution_dataset(
        self,
        data_list: List[StandardizedDataFormat],
        resolutions: List[TemporalResolution],
        base_config: Optional[AggregationConfig] = None
    ) -> Dict[TemporalResolution, List[StandardizedDataFormat]]:
        """
        Create multi-resolution dataset with different temporal aggregations
        
        Args:
            data_list: Original high-resolution data
            resolutions: Target resolutions for aggregation
            base_config: Base configuration (will be modified for each resolution)
            
        Returns:
            Dictionary mapping resolutions to aggregated datasets
        """
        if base_config is None:
            base_config = AggregationConfig(
                target_resolution=TemporalResolution.MINUTE,
                aggregation_method='mean'
            )
        
        multi_resolution_data = {}
        
        for resolution in resolutions:
            config = AggregationConfig(
                target_resolution=resolution,
                aggregation_method=base_config.aggregation_method,
                preserve_extremes=base_config.preserve_extremes,
                include_statistics=base_config.include_statistics,
                weight_by_quality=base_config.weight_by_quality,
                min_points_per_bin=base_config.min_points_per_bin,
                uncertainty_propagation=base_config.uncertainty_propagation
            )
            
            try:
                result = self.aggregate_temporal_data(data_list, config)
                multi_resolution_data[resolution] = result.aggregated_data
                logger.info(f"Created {resolution.value} resolution dataset with {len(result.aggregated_data)} points")
            except Exception as e:
                logger.error(f"Failed to create {resolution.value} resolution dataset: {e}")
                multi_resolution_data[resolution] = []
        
        return multi_resolution_data