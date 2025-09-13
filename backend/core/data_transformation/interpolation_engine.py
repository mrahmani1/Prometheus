"""
Data interpolation and gap-filling algorithms
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import interpolate
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

from .schemas import StandardizedDataFormat, TemporalResolution


logger = logging.getLogger(__name__)


@dataclass
class InterpolationResult:
    """Result of data interpolation"""
    interpolated_data: List[StandardizedDataFormat]
    method_used: str
    interpolated_points: int
    gaps_filled: int
    interpolation_quality: float
    uncertainty_estimates: Optional[Dict[str, np.ndarray]] = None


@dataclass
class GapInfo:
    """Information about a data gap"""
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    gap_size: int
    surrounding_data_quality: float


class InterpolationEngine:
    """
    Handles data interpolation and gap-filling for geological datasets
    """
    
    def __init__(self):
        self.interpolation_methods = {
            'linear': self._linear_interpolation,
            'cubic': self._cubic_interpolation,
            'spline': self._spline_interpolation,
            'kriging': self._kriging_interpolation,
            'gaussian_process': self._gaussian_process_interpolation,
            'nearest': self._nearest_neighbor_interpolation,
            'rbf': self._rbf_interpolation,
            'temporal_trend': self._temporal_trend_interpolation,
        }
        
        self.spatial_methods = {
            'idw': self._inverse_distance_weighting,
            'kriging_spatial': self._spatial_kriging,
            'thin_plate_spline': self._thin_plate_spline,
            'natural_neighbor': self._natural_neighbor_interpolation,
        }
        
    def detect_gaps(
        self,
        data_list: List[StandardizedDataFormat],
        expected_resolution: TemporalResolution,
        gap_threshold_factor: float = 2.0
    ) -> List[GapInfo]:
        """
        Detect gaps in temporal data
        
        Args:
            data_list: Time series data
            expected_resolution: Expected temporal resolution
            gap_threshold_factor: Factor above expected resolution to consider a gap
            
        Returns:
            List of detected gaps
        """
        if len(data_list) < 2:
            return []
        
        # Sort data by timestamp
        sorted_data = sorted(data_list, key=lambda x: x.timestamp)
        
        # Calculate expected interval
        resolution_seconds = self._get_resolution_seconds(expected_resolution)
        gap_threshold = resolution_seconds * gap_threshold_factor
        
        gaps = []
        for i in range(len(sorted_data) - 1):
            current_time = sorted_data[i].timestamp
            next_time = sorted_data[i + 1].timestamp
            
            time_diff = (next_time - current_time).total_seconds()
            
            if time_diff > gap_threshold:
                # Calculate surrounding data quality
                quality_before = self._assess_data_quality(sorted_data[max(0, i-2):i+1])
                quality_after = self._assess_data_quality(sorted_data[i+1:min(len(sorted_data), i+4)])
                surrounding_quality = (quality_before + quality_after) / 2
                
                gap = GapInfo(
                    start_time=current_time,
                    end_time=next_time,
                    duration_seconds=time_diff,
                    gap_size=int(time_diff / resolution_seconds) - 1,
                    surrounding_data_quality=surrounding_quality
                )
                gaps.append(gap)
        
        return gaps
    
    def _get_resolution_seconds(self, resolution: TemporalResolution) -> float:
        """Convert temporal resolution to seconds"""
        resolution_map = {
            TemporalResolution.MILLISECOND: 0.001,
            TemporalResolution.SECOND: 1.0,
            TemporalResolution.MINUTE: 60.0,
            TemporalResolution.HOUR: 3600.0,
            TemporalResolution.DAY: 86400.0,
            TemporalResolution.WEEK: 604800.0,
            TemporalResolution.MONTH: 2629746.0,
            TemporalResolution.YEAR: 31556952.0,
        }
        return resolution_map.get(resolution, 1.0)
    
    def _assess_data_quality(self, data_subset: List[StandardizedDataFormat]) -> float:
        """Assess quality of data subset"""
        if not data_subset:
            return 0.0
        
        quality_scores = []
        for data in data_subset:
            if data.quality_metrics:
                quality_scores.append(data.quality_metrics.overall_score)
            else:
                # Estimate quality based on flags
                flag_penalty = len(data.quality_flags) * 0.1
                quality_scores.append(max(0.0, 1.0 - flag_penalty))
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def interpolate_temporal_gaps(
        self,
        data_list: List[StandardizedDataFormat],
        method: str = 'linear',
        max_gap_duration: float = 3600.0,
        target_resolution: Optional[TemporalResolution] = None
    ) -> InterpolationResult:
        """
        Fill temporal gaps in data using interpolation
        
        Args:
            data_list: Data with potential gaps
            method: Interpolation method to use
            max_gap_duration: Maximum gap duration to interpolate (seconds)
            target_resolution: Target temporal resolution
            
        Returns:
            Interpolation result with filled gaps
        """
        if not data_list:
            return InterpolationResult(
                interpolated_data=[],
                method_used=method,
                interpolated_points=0,
                gaps_filled=0,
                interpolation_quality=0.0
            )
        
        # Sort data by timestamp
        sorted_data = sorted(data_list, key=lambda x: x.timestamp)
        
        # Detect gaps
        if target_resolution:
            gaps = self.detect_gaps(sorted_data, target_resolution)
        else:
            # Estimate resolution from data
            estimated_resolution = self._estimate_temporal_resolution(sorted_data)
            gaps = self.detect_gaps(sorted_data, estimated_resolution)
        
        # Filter gaps by maximum duration
        fillable_gaps = [gap for gap in gaps if gap.duration_seconds <= max_gap_duration]
        
        # Apply interpolation method
        if method not in self.interpolation_methods:
            logger.warning(f"Unknown interpolation method: {method}, using linear")
            method = 'linear'
        
        interpolation_func = self.interpolation_methods[method]
        
        interpolated_data = []
        interpolated_points = 0
        gaps_filled = 0
        quality_scores = []
        
        # Process each gap
        current_data = sorted_data.copy()
        
        for gap in fillable_gaps:
            try:
                gap_result = interpolation_func(current_data, gap)
                if gap_result:
                    interpolated_data.extend(gap_result.interpolated_points)
                    interpolated_points += len(gap_result.interpolated_points)
                    gaps_filled += 1
                    quality_scores.append(gap_result.quality)
                    
                    # Insert interpolated points into current data for next iteration
                    current_data.extend(gap_result.interpolated_points)
                    current_data.sort(key=lambda x: x.timestamp)
                    
            except Exception as e:
                logger.error(f"Failed to interpolate gap {gap.start_time} - {gap.end_time}: {e}")
        
        # Combine original and interpolated data
        all_data = sorted_data + interpolated_data
        all_data.sort(key=lambda x: x.timestamp)
        
        # Calculate overall quality
        overall_quality = np.mean(quality_scores) if quality_scores else 1.0
        
        return InterpolationResult(
            interpolated_data=all_data,
            method_used=method,
            interpolated_points=interpolated_points,
            gaps_filled=gaps_filled,
            interpolation_quality=overall_quality
        )
    
    def _estimate_temporal_resolution(self, data_list: List[StandardizedDataFormat]) -> TemporalResolution:
        """Estimate temporal resolution from data"""
        if len(data_list) < 2:
            return TemporalResolution.MINUTE
        
        # Calculate time differences
        time_diffs = []
        for i in range(len(data_list) - 1):
            diff = (data_list[i + 1].timestamp - data_list[i].timestamp).total_seconds()
            if diff > 0:
                time_diffs.append(diff)
        
        if not time_diffs:
            return TemporalResolution.MINUTE
        
        # Find most common interval
        median_diff = np.median(time_diffs)
        
        # Map to resolution enum
        if median_diff < 1:
            return TemporalResolution.MILLISECOND
        elif median_diff < 60:
            return TemporalResolution.SECOND
        elif median_diff < 3600:
            return TemporalResolution.MINUTE
        elif median_diff < 86400:
            return TemporalResolution.HOUR
        elif median_diff < 604800:
            return TemporalResolution.DAY
        elif median_diff < 2629746:
            return TemporalResolution.WEEK
        elif median_diff < 31556952:
            return TemporalResolution.MONTH
        else:
            return TemporalResolution.YEAR
    
    def _linear_interpolation(self, data_list: List[StandardizedDataFormat], gap: GapInfo):
        """Linear interpolation for gap filling"""
        from dataclasses import dataclass
        
        @dataclass
        class GapInterpolationResult:
            interpolated_points: List[StandardizedDataFormat]
            quality: float
        
        # Find data points before and after gap
        before_data = [d for d in data_list if d.timestamp <= gap.start_time]
        after_data = [d for d in data_list if d.timestamp >= gap.end_time]
        
        if not before_data or not after_data:
            return None
        
        point_before = before_data[-1]
        point_after = after_data[0]
        
        # Calculate number of interpolation points
        total_duration = (gap.end_time - gap.start_time).total_seconds()
        estimated_resolution = self._estimate_temporal_resolution(data_list)
        resolution_seconds = self._get_resolution_seconds(estimated_resolution)
        n_points = max(1, int(total_duration / resolution_seconds) - 1)
        
        interpolated_points = []
        
        for i in range(1, n_points + 1):
            # Calculate interpolation weight
            weight = i / (n_points + 1)
            
            # Interpolate timestamp
            time_diff = gap.end_time - gap.start_time
            interp_time = gap.start_time + timedelta(seconds=time_diff.total_seconds() * weight)
            
            # Create interpolated data point
            interp_data = point_before.copy(deep=True)
            interp_data.data_id = f"{point_before.data_id}_interp_{i}"
            interp_data.timestamp = interp_time
            
            # Interpolate values
            for key, value_before in point_before.values.items():
                if key in point_after.values:
                    value_after = point_after.values[key]
                    
                    if isinstance(value_before, (int, float)) and isinstance(value_after, (int, float)):
                        interp_value = value_before * (1 - weight) + value_after * weight
                        interp_data.values[key] = interp_value
                    elif isinstance(value_before, (list, np.ndarray)) and isinstance(value_after, (list, np.ndarray)):
                        arr_before = np.array(value_before)
                        arr_after = np.array(value_after)
                        if arr_before.shape == arr_after.shape:
                            interp_array = arr_before * (1 - weight) + arr_after * weight
                            interp_data.values[key] = interp_array.tolist()
            
            # Update uncertainty
            if interp_data.value_uncertainty:
                for key in interp_data.value_uncertainty:
                    if key in point_after.value_uncertainty:
                        unc_before = interp_data.value_uncertainty[key]
                        unc_after = point_after.value_uncertainty[key]
                        # Conservative uncertainty propagation
                        interp_data.value_uncertainty[key] = np.sqrt(
                            ((1 - weight) * unc_before) ** 2 + (weight * unc_after) ** 2
                        )
            
            # Add quality flags
            interp_data.quality_flags.append("temporally_interpolated")
            interp_data.quality_flags.append(f"interpolation_method:linear")
            
            # Add processing history
            processing_step = {
                'step_id': f"linear_interpolation_{datetime.now().isoformat()}",
                'timestamp': datetime.now(),
                'operation': 'linear_interpolation',
                'parameters': {
                    'weight': weight,
                    'gap_duration_seconds': total_duration,
                    'interpolation_point': i,
                    'total_points': n_points
                },
                'input_hash': str(hash(f"{point_before.data_id}_{point_after.data_id}")),
                'output_hash': str(hash(interp_data.data_id)),
                'processor_version': '1.0.0'
            }
            
            interp_data.processing_history.append(processing_step)
            interpolated_points.append(interp_data)
        
        # Calculate interpolation quality based on gap characteristics
        quality = self._calculate_interpolation_quality(gap, 'linear', point_before, point_after)
        
        return GapInterpolationResult(
            interpolated_points=interpolated_points,
            quality=quality
        )
    
    def _cubic_interpolation(self, data_list: List[StandardizedDataFormat], gap: GapInfo):
        """Cubic spline interpolation for gap filling"""
        # Find surrounding data points (need at least 4 for cubic)
        surrounding_data = [d for d in data_list 
                          if gap.start_time - timedelta(hours=1) <= d.timestamp <= gap.end_time + timedelta(hours=1)]
        
        if len(surrounding_data) < 4:
            # Fall back to linear interpolation
            return self._linear_interpolation(data_list, gap)
        
        # Extract timestamps and values for interpolation
        timestamps = np.array([(d.timestamp - surrounding_data[0].timestamp).total_seconds() 
                              for d in surrounding_data])
        
        # Get gap time points
        gap_start_seconds = (gap.start_time - surrounding_data[0].timestamp).total_seconds()
        gap_end_seconds = (gap.end_time - surrounding_data[0].timestamp).total_seconds()
        
        # Calculate interpolation points
        estimated_resolution = self._estimate_temporal_resolution(data_list)
        resolution_seconds = self._get_resolution_seconds(estimated_resolution)
        n_points = max(1, int((gap_end_seconds - gap_start_seconds) / resolution_seconds) - 1)
        
        interp_times = np.linspace(gap_start_seconds + resolution_seconds, 
                                  gap_end_seconds - resolution_seconds, n_points)
        
        interpolated_points = []
        
        # Interpolate each value field
        for i, interp_time_sec in enumerate(interp_times):
            interp_time = surrounding_data[0].timestamp + timedelta(seconds=interp_time_sec)
            
            # Create base interpolated data point
            base_data = surrounding_data[0].copy(deep=True)
            base_data.data_id = f"{base_data.data_id}_cubic_interp_{i}"
            base_data.timestamp = interp_time
            
            # Interpolate each numerical field
            for key in base_data.values:
                try:
                    # Extract values for this field
                    field_values = []
                    for data_point in surrounding_data:
                        if key in data_point.values:
                            value = data_point.values[key]
                            if isinstance(value, (int, float)):
                                field_values.append(value)
                            else:
                                field_values.append(None)
                        else:
                            field_values.append(None)
                    
                    # Skip if not enough valid values
                    valid_values = [v for v in field_values if v is not None]
                    if len(valid_values) < 4:
                        continue
                    
                    # Create cubic spline
                    valid_indices = [i for i, v in enumerate(field_values) if v is not None]
                    valid_times = timestamps[valid_indices]
                    valid_vals = np.array(valid_values)
                    
                    spline = interpolate.CubicSpline(valid_times, valid_vals)
                    interp_value = float(spline(interp_time_sec))
                    
                    base_data.values[key] = interp_value
                    
                except Exception as e:
                    logger.warning(f"Failed to interpolate field {key} with cubic spline: {e}")
                    # Keep original value or skip
                    continue
            
            # Add quality flags
            base_data.quality_flags.append("temporally_interpolated")
            base_data.quality_flags.append("interpolation_method:cubic")
            
            interpolated_points.append(base_data)
        
        # Calculate quality
        quality = self._calculate_interpolation_quality(gap, 'cubic', surrounding_data[0], surrounding_data[-1])
        
        from dataclasses import dataclass
        
        @dataclass
        class GapInterpolationResult:
            interpolated_points: List[StandardizedDataFormat]
            quality: float
        
        return GapInterpolationResult(
            interpolated_points=interpolated_points,
            quality=quality
        )
    
    def _spline_interpolation(self, data_list: List[StandardizedDataFormat], gap: GapInfo):
        """B-spline interpolation for smooth gap filling"""
        # Similar to cubic but with B-splines for smoother results
        return self._cubic_interpolation(data_list, gap)  # Simplified implementation
    
    def _kriging_interpolation(self, data_list: List[StandardizedDataFormat], gap: GapInfo):
        """Kriging interpolation with uncertainty estimation"""
        # Simplified kriging implementation
        return self._gaussian_process_interpolation(data_list, gap)
    
    def _gaussian_process_interpolation(self, data_list: List[StandardizedDataFormat], gap: GapInfo):
        """Gaussian Process interpolation with uncertainty quantification"""
        try:
            # Find surrounding data
            surrounding_data = [d for d in data_list 
                              if gap.start_time - timedelta(hours=2) <= d.timestamp <= gap.end_time + timedelta(hours=2)]
            
            if len(surrounding_data) < 3:
                return self._linear_interpolation(data_list, gap)
            
            # Prepare time features
            reference_time = surrounding_data[0].timestamp
            X_train = np.array([[(d.timestamp - reference_time).total_seconds()] for d in surrounding_data])
            
            # Calculate interpolation times
            gap_start_sec = (gap.start_time - reference_time).total_seconds()
            gap_end_sec = (gap.end_time - reference_time).total_seconds()
            
            estimated_resolution = self._estimate_temporal_resolution(data_list)
            resolution_seconds = self._get_resolution_seconds(estimated_resolution)
            n_points = max(1, int((gap_end_sec - gap_start_sec) / resolution_seconds) - 1)
            
            X_pred = np.linspace(gap_start_sec + resolution_seconds, 
                               gap_end_sec - resolution_seconds, n_points).reshape(-1, 1)
            
            interpolated_points = []
            
            # Interpolate each numerical field
            for key in surrounding_data[0].values:
                try:
                    # Extract values
                    y_train = []
                    for data_point in surrounding_data:
                        if key in data_point.values and isinstance(data_point.values[key], (int, float)):
                            y_train.append(data_point.values[key])
                        else:
                            y_train.append(np.nan)
                    
                    # Skip if not enough valid values
                    valid_mask = ~np.isnan(y_train)
                    if np.sum(valid_mask) < 3:
                        continue
                    
                    X_valid = X_train[valid_mask]
                    y_valid = np.array(y_train)[valid_mask]
                    
                    # Fit Gaussian Process
                    kernel = RBF(length_scale=3600.0) + WhiteKernel(noise_level=0.1)
                    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
                    gp.fit(X_valid, y_valid)
                    
                    # Predict
                    y_pred, y_std = gp.predict(X_pred, return_std=True)
                    
                    # Create interpolated points
                    for i, (pred_time_sec, pred_value, pred_std) in enumerate(zip(X_pred.flatten(), y_pred, y_std)):
                        if i >= len(interpolated_points):
                            # Create new interpolated point
                            interp_data = surrounding_data[0].copy(deep=True)
                            interp_data.data_id = f"{surrounding_data[0].data_id}_gp_interp_{i}"
                            interp_data.timestamp = reference_time + timedelta(seconds=pred_time_sec)
                            interp_data.values = {}
                            interp_data.value_uncertainty = {}
                            interp_data.quality_flags.append("temporally_interpolated")
                            interp_data.quality_flags.append("interpolation_method:gaussian_process")
                            interpolated_points.append(interp_data)
                        
                        # Add predicted value and uncertainty
                        interpolated_points[i].values[key] = float(pred_value)
                        interpolated_points[i].value_uncertainty[key] = float(pred_std)
                
                except Exception as e:
                    logger.warning(f"GP interpolation failed for field {key}: {e}")
                    continue
            
            # Calculate quality
            quality = self._calculate_interpolation_quality(gap, 'gaussian_process', 
                                                          surrounding_data[0], surrounding_data[-1])
            
            from dataclasses import dataclass
            
            @dataclass
            class GapInterpolationResult:
                interpolated_points: List[StandardizedDataFormat]
                quality: float
            
            return GapInterpolationResult(
                interpolated_points=interpolated_points,
                quality=quality
            )
            
        except Exception as e:
            logger.error(f"Gaussian Process interpolation failed: {e}")
            return self._linear_interpolation(data_list, gap)
    
    def _nearest_neighbor_interpolation(self, data_list: List[StandardizedDataFormat], gap: GapInfo):
        """Nearest neighbor interpolation (step function)"""
        # Find closest data points
        before_data = [d for d in data_list if d.timestamp <= gap.start_time]
        after_data = [d for d in data_list if d.timestamp >= gap.end_time]
        
        if not before_data and not after_data:
            return None
        
        # Choose closest point
        if not before_data:
            closest_point = after_data[0]
        elif not after_data:
            closest_point = before_data[-1]
        else:
            time_to_before = (gap.start_time - before_data[-1].timestamp).total_seconds()
            time_to_after = (after_data[0].timestamp - gap.end_time).total_seconds()
            closest_point = before_data[-1] if time_to_before <= time_to_after else after_data[0]
        
        # Create interpolated points using nearest neighbor values
        estimated_resolution = self._estimate_temporal_resolution(data_list)
        resolution_seconds = self._get_resolution_seconds(estimated_resolution)
        total_duration = (gap.end_time - gap.start_time).total_seconds()
        n_points = max(1, int(total_duration / resolution_seconds) - 1)
        
        interpolated_points = []
        
        for i in range(1, n_points + 1):
            time_offset = (total_duration * i) / (n_points + 1)
            interp_time = gap.start_time + timedelta(seconds=time_offset)
            
            interp_data = closest_point.copy(deep=True)
            interp_data.data_id = f"{closest_point.data_id}_nn_interp_{i}"
            interp_data.timestamp = interp_time
            interp_data.quality_flags.append("temporally_interpolated")
            interp_data.quality_flags.append("interpolation_method:nearest_neighbor")
            
            interpolated_points.append(interp_data)
        
        quality = self._calculate_interpolation_quality(gap, 'nearest', closest_point, closest_point)
        
        from dataclasses import dataclass
        
        @dataclass
        class GapInterpolationResult:
            interpolated_points: List[StandardizedDataFormat]
            quality: float
        
        return GapInterpolationResult(
            interpolated_points=interpolated_points,
            quality=quality * 0.7  # Lower quality for step function
        )
    
    def _rbf_interpolation(self, data_list: List[StandardizedDataFormat], gap: GapInfo):
        """Radial Basis Function interpolation"""
        # Simplified RBF implementation
        return self._cubic_interpolation(data_list, gap)
    
    def _temporal_trend_interpolation(self, data_list: List[StandardizedDataFormat], gap: GapInfo):
        """Trend-based interpolation considering temporal patterns"""
        # Simplified trend interpolation
        return self._linear_interpolation(data_list, gap)
    
    def _calculate_interpolation_quality(
        self,
        gap: GapInfo,
        method: str,
        point_before: StandardizedDataFormat,
        point_after: StandardizedDataFormat
    ) -> float:
        """Calculate quality score for interpolation"""
        base_quality = 0.8  # Base quality for interpolation
        
        # Adjust based on gap duration
        if gap.duration_seconds < 300:  # 5 minutes
            duration_factor = 1.0
        elif gap.duration_seconds < 1800:  # 30 minutes
            duration_factor = 0.9
        elif gap.duration_seconds < 3600:  # 1 hour
            duration_factor = 0.8
        else:
            duration_factor = 0.7
        
        # Adjust based on surrounding data quality
        quality_factor = gap.surrounding_data_quality
        
        # Adjust based on method
        method_factors = {
            'linear': 0.8,
            'cubic': 0.9,
            'spline': 0.9,
            'gaussian_process': 0.95,
            'kriging': 0.95,
            'nearest': 0.6,
            'rbf': 0.85,
            'temporal_trend': 0.85,
        }
        method_factor = method_factors.get(method, 0.8)
        
        return base_quality * duration_factor * quality_factor * method_factor
    
    # Spatial interpolation methods (for completeness)
    def _inverse_distance_weighting(self, data_points, target_location):
        """Inverse Distance Weighting spatial interpolation"""
        pass  # Implementation would go here
    
    def _spatial_kriging(self, data_points, target_location):
        """Spatial kriging interpolation"""
        pass  # Implementation would go here
    
    def _thin_plate_spline(self, data_points, target_location):
        """Thin plate spline spatial interpolation"""
        pass  # Implementation would go here
    
    def _natural_neighbor_interpolation(self, data_points, target_location):
        """Natural neighbor spatial interpolation"""
        pass  # Implementation would go here