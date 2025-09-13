"""
Temporal synchronization and time zone standardization
"""

import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timezone, timedelta
import pytz
from dataclasses import dataclass
import logging

from .schemas import StandardizedDataFormat, TemporalResolution, TimeRange


logger = logging.getLogger(__name__)


@dataclass
class TemporalSyncResult:
    """Result of temporal synchronization"""
    synchronized_data: List[StandardizedDataFormat]
    common_time_grid: np.ndarray
    synchronization_accuracy: float
    interpolated_points: int
    gaps_filled: int


class TemporalSynchronizer:
    """
    Handles temporal synchronization and time zone standardization for geological data
    """
    
    def __init__(self):
        self.timezone_cache = {}
        self.resolution_seconds = self._initialize_resolution_mapping()
        
    def _initialize_resolution_mapping(self) -> Dict[TemporalResolution, float]:
        """Initialize mapping of temporal resolutions to seconds"""
        return {
            TemporalResolution.MILLISECOND: 0.001,
            TemporalResolution.SECOND: 1.0,
            TemporalResolution.MINUTE: 60.0,
            TemporalResolution.HOUR: 3600.0,
            TemporalResolution.DAY: 86400.0,
            TemporalResolution.WEEK: 604800.0,
            TemporalResolution.MONTH: 2629746.0,  # Average month
            TemporalResolution.YEAR: 31556952.0   # Average year
        }
    
    def standardize_timezone(
        self,
        timestamp: datetime,
        source_timezone: Optional[str] = None,
        target_timezone: str = "UTC"
    ) -> datetime:
        """
        Convert timestamp to target timezone
        
        Args:
            timestamp: Input timestamp
            source_timezone: Source timezone (if None, assumes UTC)
            target_timezone: Target timezone
            
        Returns:
            Timestamp in target timezone
        """
        try:
            # Handle timezone-naive timestamps
            if timestamp.tzinfo is None:
                if source_timezone:
                    source_tz = pytz.timezone(source_timezone)
                    timestamp = source_tz.localize(timestamp)
                else:
                    timestamp = pytz.UTC.localize(timestamp)
            
            # Convert to target timezone
            target_tz = pytz.timezone(target_timezone)
            return timestamp.astimezone(target_tz)
            
        except Exception as e:
            logger.error(f"Timezone conversion failed: {e}")
            # Return original timestamp with warning
            return timestamp
    
    def create_time_grid(
        self,
        start_time: datetime,
        end_time: datetime,
        resolution: TemporalResolution,
        timezone_str: str = "UTC"
    ) -> np.ndarray:
        """
        Create regular time grid for synchronization
        
        Args:
            start_time: Start of time grid
            end_time: End of time grid
            resolution: Temporal resolution
            timezone_str: Timezone for grid
            
        Returns:
            Array of datetime objects
        """
        # Ensure times are in correct timezone
        start_time = self.standardize_timezone(start_time, target_timezone=timezone_str)
        end_time = self.standardize_timezone(end_time, target_timezone=timezone_str)
        
        # Get resolution in seconds
        resolution_sec = self.resolution_seconds[resolution]
        
        # Calculate number of points
        total_seconds = (end_time - start_time).total_seconds()
        n_points = int(total_seconds / resolution_sec) + 1
        
        # Create time grid
        time_grid = []
        current_time = start_time
        
        for i in range(n_points):
            time_grid.append(current_time)
            current_time += timedelta(seconds=resolution_sec)
            if current_time > end_time:
                break
        
        return np.array(time_grid)
    
    def synchronize_to_grid(
        self,
        data_list: List[StandardizedDataFormat],
        time_grid: np.ndarray,
        interpolation_method: str = "linear",
        max_gap_seconds: float = 3600.0
    ) -> TemporalSyncResult:
        """
        Synchronize data to common time grid
        
        Args:
            data_list: List of data to synchronize
            time_grid: Target time grid
            interpolation_method: Method for interpolation
            max_gap_seconds: Maximum gap to interpolate across
            
        Returns:
            Synchronized data result
        """
        synchronized_data = []
        interpolated_points = 0
        gaps_filled = 0
        
        # Group data by type and location for synchronization
        data_groups = self._group_data_for_sync(data_list)
        
        for group_key, group_data in data_groups.items():
            try:
                sync_result = self._synchronize_group(
                    group_data, time_grid, interpolation_method, max_gap_seconds
                )
                synchronized_data.extend(sync_result.data)
                interpolated_points += sync_result.interpolated
                gaps_filled += sync_result.gaps_filled
                
            except Exception as e:
                logger.error(f"Failed to synchronize group {group_key}: {e}")
                # Add original data with quality flags
                for data in group_data:
                    data_copy = data.copy(deep=True)
                    data_copy.quality_flags.append(f"synchronization_failed: {str(e)}")
                    synchronized_data.append(data_copy)
        
        # Calculate synchronization accuracy
        accuracy = self._calculate_sync_accuracy(data_list, synchronized_data, time_grid)
        
        return TemporalSyncResult(
            synchronized_data=synchronized_data,
            common_time_grid=time_grid,
            synchronization_accuracy=accuracy,
            interpolated_points=interpolated_points,
            gaps_filled=gaps_filled
        )
    
    def _group_data_for_sync(
        self,
        data_list: List[StandardizedDataFormat]
    ) -> Dict[str, List[StandardizedDataFormat]]:
        """Group data by type and location for synchronization"""
        groups = {}
        
        for data in data_list:
            # Create group key based on data type, source, and location
            location_key = f"{data.location['x']:.6f}_{data.location['y']:.6f}"
            group_key = f"{data.data_type.value}_{data.source}_{location_key}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(data)
        
        return groups
    
    def _synchronize_group(
        self,
        group_data: List[StandardizedDataFormat],
        time_grid: np.ndarray,
        interpolation_method: str,
        max_gap_seconds: float
    ) -> 'GroupSyncResult':
        """Synchronize a group of related data"""
        from dataclasses import dataclass
        
        @dataclass
        class GroupSyncResult:
            data: List[StandardizedDataFormat]
            interpolated: int
            gaps_filled: int
        
        # Sort data by timestamp
        sorted_data = sorted(group_data, key=lambda x: x.timestamp)
        
        # Extract timestamps and values
        timestamps = np.array([data.timestamp for data in sorted_data])
        
        # Create synchronized data for each time point in grid
        synchronized_group = []
        interpolated_count = 0
        gaps_filled_count = 0
        
        for target_time in time_grid:
            # Find closest data points
            time_diffs = np.array([(target_time - ts).total_seconds() for ts in timestamps])
            closest_idx = np.argmin(np.abs(time_diffs))
            closest_diff = time_diffs[closest_idx]
            
            # If exact match or very close, use original data
            if abs(closest_diff) < 1.0:  # Within 1 second
                original_data = sorted_data[closest_idx]
                synced_data = original_data.copy(deep=True)
                synced_data.timestamp = target_time
                synchronized_group.append(synced_data)
                
            # If within interpolation range, interpolate
            elif abs(closest_diff) <= max_gap_seconds:
                interpolated_data = self._interpolate_data_point(
                    sorted_data, target_time, interpolation_method
                )
                if interpolated_data:
                    synchronized_group.append(interpolated_data)
                    interpolated_count += 1
                    
                    # Check if this filled a gap
                    if self._is_gap_fill(sorted_data, target_time):
                        gaps_filled_count += 1
            
            # Otherwise, skip this time point (gap too large)
        
        return GroupSyncResult(
            data=synchronized_group,
            interpolated=interpolated_count,
            gaps_filled=gaps_filled_count
        )
    
    def _interpolate_data_point(
        self,
        data_list: List[StandardizedDataFormat],
        target_time: datetime,
        method: str
    ) -> Optional[StandardizedDataFormat]:
        """Interpolate data at target time"""
        if not data_list:
            return None
        
        # Find surrounding data points
        timestamps = np.array([data.timestamp for data in data_list])
        time_diffs = np.array([(target_time - ts).total_seconds() for ts in timestamps])
        
        # Find points before and after target time
        before_mask = time_diffs <= 0
        after_mask = time_diffs >= 0
        
        if not np.any(before_mask) or not np.any(after_mask):
            return None  # Cannot interpolate at boundary
        
        before_idx = np.where(before_mask)[0][-1]  # Latest before
        after_idx = np.where(after_mask)[0][0]     # Earliest after
        
        if before_idx == after_idx:
            # Exact match
            return data_list[before_idx]
        
        data_before = data_list[before_idx]
        data_after = data_list[after_idx]
        
        # Calculate interpolation weights
        dt_total = (data_after.timestamp - data_before.timestamp).total_seconds()
        dt_target = (target_time - data_before.timestamp).total_seconds()
        
        if dt_total == 0:
            weight_after = 0.5
        else:
            weight_after = dt_target / dt_total
        weight_before = 1.0 - weight_after
        
        # Create interpolated data
        interpolated = data_before.copy(deep=True)
        interpolated.timestamp = target_time
        
        # Interpolate numerical values
        for key, value in data_before.values.items():
            if key in data_after.values:
                try:
                    if isinstance(value, (int, float)):
                        interpolated_value = (
                            weight_before * value + 
                            weight_after * data_after.values[key]
                        )
                        interpolated.values[key] = interpolated_value
                    elif isinstance(value, (list, np.ndarray)):
                        value_array = np.array(value)
                        after_array = np.array(data_after.values[key])
                        if value_array.shape == after_array.shape:
                            interpolated_array = (
                                weight_before * value_array + 
                                weight_after * after_array
                            )
                            interpolated.values[key] = interpolated_array.tolist()
                except Exception as e:
                    logger.warning(f"Failed to interpolate value {key}: {e}")
                    # Keep original value
                    pass
        
        # Update uncertainty if available
        if interpolated.value_uncertainty:
            for key in interpolated.value_uncertainty:
                if key in data_after.value_uncertainty:
                    # Propagate uncertainty (conservative approach)
                    unc_before = interpolated.value_uncertainty[key]
                    unc_after = data_after.value_uncertainty[key]
                    interpolated.value_uncertainty[key] = np.sqrt(
                        (weight_before * unc_before) ** 2 + 
                        (weight_after * unc_after) ** 2
                    )
        
        # Add processing history
        processing_step = {
            'step_id': f"temporal_interpolation_{datetime.now().isoformat()}",
            'timestamp': datetime.now(),
            'operation': 'temporal_interpolation',
            'parameters': {
                'method': method,
                'weight_before': weight_before,
                'weight_after': weight_after,
                'time_before': data_before.timestamp.isoformat(),
                'time_after': data_after.timestamp.isoformat()
            },
            'input_hash': str(hash(f"{data_before.data_id}_{data_after.data_id}")),
            'output_hash': str(hash(interpolated.data_id)),
            'processor_version': '1.0.0'
        }
        
        interpolated.processing_history.append(processing_step)
        interpolated.quality_flags.append("temporally_interpolated")
        
        return interpolated
    
    def _is_gap_fill(self, data_list: List[StandardizedDataFormat], target_time: datetime) -> bool:
        """Check if interpolation fills a significant gap"""
        timestamps = [data.timestamp for data in data_list]
        
        # Find gaps larger than expected resolution
        for i in range(len(timestamps) - 1):
            gap_start = timestamps[i]
            gap_end = timestamps[i + 1]
            
            if gap_start <= target_time <= gap_end:
                gap_duration = (gap_end - gap_start).total_seconds()
                return gap_duration > 300  # 5 minutes threshold
        
        return False
    
    def _calculate_sync_accuracy(
        self,
        original_data: List[StandardizedDataFormat],
        synchronized_data: List[StandardizedDataFormat],
        time_grid: np.ndarray
    ) -> float:
        """Calculate synchronization accuracy"""
        if not original_data or not synchronized_data:
            return 0.0
        
        # Count exact matches vs interpolated points
        exact_matches = sum(1 for data in synchronized_data 
                          if "temporally_interpolated" not in data.quality_flags)
        
        total_points = len(synchronized_data)
        
        if total_points == 0:
            return 0.0
        
        # Base accuracy on ratio of exact matches
        base_accuracy = exact_matches / total_points
        
        # Adjust for temporal coverage
        original_timespan = self._calculate_timespan(original_data)
        grid_timespan = (time_grid[-1] - time_grid[0]).total_seconds()
        
        coverage_ratio = min(1.0, original_timespan / grid_timespan if grid_timespan > 0 else 0.0)
        
        return base_accuracy * coverage_ratio
    
    def _calculate_timespan(self, data_list: List[StandardizedDataFormat]) -> float:
        """Calculate timespan covered by data in seconds"""
        if len(data_list) < 2:
            return 0.0
        
        timestamps = [data.timestamp for data in data_list]
        return (max(timestamps) - min(timestamps)).total_seconds()
    
    def synchronize_data_formats(
        self,
        data_list: List[StandardizedDataFormat],
        target_resolution: TemporalResolution,
        target_timezone: str = "UTC",
        time_range: Optional[TimeRange] = None
    ) -> TemporalSyncResult:
        """
        Synchronize list of data to common temporal grid
        
        Args:
            data_list: Data to synchronize
            target_resolution: Target temporal resolution
            target_timezone: Target timezone
            time_range: Optional time range (uses data extent if None)
            
        Returns:
            Temporal synchronization result
        """
        if not data_list:
            return TemporalSyncResult(
                synchronized_data=[],
                common_time_grid=np.array([]),
                synchronization_accuracy=0.0,
                interpolated_points=0,
                gaps_filled=0
            )
        
        # Standardize all timestamps to target timezone
        for data in data_list:
            data.timestamp = self.standardize_timezone(
                data.timestamp, target_timezone=target_timezone
            )
        
        # Determine time range
        if time_range is None:
            timestamps = [data.timestamp for data in data_list]
            start_time = min(timestamps)
            end_time = max(timestamps)
        else:
            start_time = self.standardize_timezone(
                time_range.start, target_timezone=target_timezone
            )
            end_time = self.standardize_timezone(
                time_range.end, target_timezone=target_timezone
            )
        
        # Create time grid
        time_grid = self.create_time_grid(
            start_time, end_time, target_resolution, target_timezone
        )
        
        # Synchronize to grid
        return self.synchronize_to_grid(data_list, time_grid)