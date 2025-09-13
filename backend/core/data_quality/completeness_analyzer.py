"""
Data Completeness Analysis and Gap Identification

This module implements comprehensive analysis of data completeness for geological
observations, including temporal gap detection, spatial coverage assessment,
and data availability analysis across different sources and time periods.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .validation_framework import GeologicalObservation, DataType, QualityFlag

logger = logging.getLogger(__name__)


@dataclass
class TemporalGap:
    """Represents a temporal gap in data coverage"""
    start_time: datetime
    end_time: datetime
    duration: timedelta
    data_type: DataType
    source_system: str
    expected_observations: int
    severity: str  # 'minor', 'moderate', 'severe', 'critical'
    impact_assessment: Dict[str, Any]


@dataclass
class SpatialGap:
    """Represents a spatial gap in data coverage"""
    region_bounds: Tuple[float, float, float, float]  # min_lat, max_lat, min_lon, max_lon
    area_km2: float
    data_type: DataType
    expected_stations: int
    actual_stations: int
    coverage_percentage: float
    priority_level: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class CompletenessReport:
    """Comprehensive data completeness analysis report"""
    analysis_timestamp: datetime
    time_range: Tuple[datetime, datetime]
    spatial_bounds: Tuple[float, float, float, float]
    
    # Overall metrics
    overall_completeness: float
    temporal_completeness: float
    spatial_completeness: float
    source_completeness: float
    
    # Detailed analysis
    temporal_gaps: List[TemporalGap]
    spatial_gaps: List[SpatialGap]
    missing_data_summary: Dict[str, Any]
    data_availability_matrix: pd.DataFrame
    
    # Recommendations
    priority_gaps: List[Dict[str, Any]]
    improvement_recommendations: List[str]
    estimated_cost_impact: Dict[str, float]


@dataclass
class DataAvailabilityMetrics:
    """Metrics for data availability analysis"""
    total_expected_observations: int
    actual_observations: int
    availability_percentage: float
    uptime_percentage: float
    mean_time_between_observations: timedelta
    longest_gap: timedelta
    gap_frequency: float  # gaps per day
    reliability_score: float


class CompletenessAnalyzer:
    """Comprehensive data completeness analysis for geological observations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.expected_frequencies = self._load_expected_frequencies()
        self.critical_regions = self._load_critical_regions()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for completeness analysis"""
        return {
            'temporal_analysis': {
                'gap_severity_thresholds': {
                    'minor': timedelta(hours=1),
                    'moderate': timedelta(hours=6),
                    'severe': timedelta(days=1),
                    'critical': timedelta(days=7)
                },
                'expected_uptime': 0.95  # 95% uptime expected
            },
            'spatial_analysis': {
                'grid_resolution_km': 50,  # 50km grid for spatial analysis
                'coverage_thresholds': {
                    'good': 0.80,
                    'acceptable': 0.60,
                    'poor': 0.40,
                    'critical': 0.20
                }
            },
            'priority_weighting': {
                'volcanic_regions': 3.0,
                'seismic_zones': 2.5,
                'populated_areas': 2.0,
                'remote_areas': 1.0
            }
        }
    
    def _load_expected_frequencies(self) -> Dict[DataType, timedelta]:
        """Load expected observation frequencies for different data types"""
        return {
            DataType.SEISMIC: timedelta(seconds=1),      # Real-time seismic monitoring
            DataType.VOLCANIC: timedelta(minutes=10),    # Volcanic parameter monitoring
            DataType.DEFORMATION: timedelta(hours=6),    # GPS measurements
            DataType.GEOCHEMICAL: timedelta(days=30),    # Laboratory analysis
            DataType.THERMAL: timedelta(hours=3),        # Thermal monitoring
            DataType.MAGNETIC: timedelta(hours=1),       # Magnetic field monitoring
            DataType.GRAVITY: timedelta(days=7),         # Gravity measurements
            DataType.METEOROLOGICAL: timedelta(minutes=15)  # Weather data
        }
    
    def _load_critical_regions(self) -> List[Dict[str, Any]]:
        """Load definitions of critical monitoring regions"""
        return [
            {
                'name': 'Ring of Fire - Pacific',
                'bounds': (-60, 60, 120, -60),  # lat_min, lat_max, lon_min, lon_max
                'priority': 'critical',
                'types': [DataType.SEISMIC, DataType.VOLCANIC]
            },
            {
                'name': 'Mediterranean Volcanic Arc',
                'bounds': (30, 50, -10, 40),
                'priority': 'high',
                'types': [DataType.VOLCANIC, DataType.SEISMIC]
            },
            {
                'name': 'Mid-Atlantic Ridge',
                'bounds': (-60, 70, -50, 10),
                'priority': 'medium',
                'types': [DataType.SEISMIC, DataType.MAGNETIC]
            }
        ]
    
    async def analyze_completeness(
        self,
        observations: List[GeologicalObservation],
        time_range: Tuple[datetime, datetime],
        spatial_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> CompletenessReport:
        """
        Perform comprehensive completeness analysis
        
        Args:
            observations: List of geological observations to analyze
            time_range: Time range for analysis (start_time, end_time)
            spatial_bounds: Optional spatial bounds (min_lat, max_lat, min_lon, max_lon)
            
        Returns:
            CompletenessReport with detailed analysis results
        """
        logger.info(f"Starting completeness analysis for {len(observations)} observations")
        
        start_time, end_time = time_range
        
        # Determine spatial bounds if not provided
        if spatial_bounds is None:
            spatial_bounds = self._calculate_spatial_bounds(observations)
        
        # Perform temporal analysis
        temporal_gaps = await self._analyze_temporal_gaps(observations, time_range)
        temporal_completeness = self._calculate_temporal_completeness(
            observations, time_range, temporal_gaps
        )
        
        # Perform spatial analysis
        spatial_gaps = await self._analyze_spatial_gaps(observations, spatial_bounds)
        spatial_completeness = self._calculate_spatial_completeness(
            observations, spatial_bounds, spatial_gaps
        )
        
        # Analyze source completeness
        source_completeness = self._analyze_source_completeness(observations)
        
        # Calculate overall completeness
        overall_completeness = (
            temporal_completeness * 0.4 +
            spatial_completeness * 0.4 +
            source_completeness * 0.2
        )
        
        # Generate data availability matrix
        availability_matrix = self._generate_availability_matrix(observations, time_range)
        
        # Create missing data summary
        missing_data_summary = self._create_missing_data_summary(
            observations, temporal_gaps, spatial_gaps
        )
        
        # Identify priority gaps
        priority_gaps = self._identify_priority_gaps(temporal_gaps, spatial_gaps)
        
        # Generate recommendations
        recommendations = self._generate_improvement_recommendations(
            temporal_gaps, spatial_gaps, overall_completeness
        )
        
        # Estimate cost impact
        cost_impact = self._estimate_cost_impact(temporal_gaps, spatial_gaps)
        
        return CompletenessReport(
            analysis_timestamp=datetime.utcnow(),
            time_range=time_range,
            spatial_bounds=spatial_bounds,
            overall_completeness=overall_completeness,
            temporal_completeness=temporal_completeness,
            spatial_completeness=spatial_completeness,
            source_completeness=source_completeness,
            temporal_gaps=temporal_gaps,
            spatial_gaps=spatial_gaps,
            missing_data_summary=missing_data_summary,
            data_availability_matrix=availability_matrix,
            priority_gaps=priority_gaps,
            improvement_recommendations=recommendations,
            estimated_cost_impact=cost_impact
        )
    
    def _calculate_spatial_bounds(
        self, 
        observations: List[GeologicalObservation]
    ) -> Tuple[float, float, float, float]:
        """Calculate spatial bounds from observations"""
        if not observations:
            return (-90, 90, -180, 180)  # Global bounds
        
        lats = [obs.location[0] for obs in observations]
        lons = [obs.location[1] for obs in observations]
        
        return (min(lats), max(lats), min(lons), max(lons))
    
    async def _analyze_temporal_gaps(
        self,
        observations: List[GeologicalObservation],
        time_range: Tuple[datetime, datetime]
    ) -> List[TemporalGap]:
        """Analyze temporal gaps in data coverage"""
        gaps = []
        start_time, end_time = time_range
        
        # Group observations by data type and source
        grouped_obs = defaultdict(lambda: defaultdict(list))
        for obs in observations:
            grouped_obs[obs.data_type][obs.source_system].append(obs)
        
        # Analyze gaps for each data type and source combination
        for data_type, sources in grouped_obs.items():
            expected_frequency = self.expected_frequencies.get(data_type, timedelta(hours=1))
            
            for source_system, obs_list in sources.items():
                # Sort observations by timestamp
                obs_list.sort(key=lambda x: x.timestamp)
                
                # Find gaps between consecutive observations
                for i in range(len(obs_list) - 1):
                    current_obs = obs_list[i]
                    next_obs = obs_list[i + 1]
                    
                    gap_duration = next_obs.timestamp - current_obs.timestamp
                    
                    # Check if gap exceeds expected frequency
                    if gap_duration > expected_frequency * 2:  # Allow 2x tolerance
                        severity = self._determine_gap_severity(gap_duration)
                        expected_obs = int(gap_duration.total_seconds() / 
                                         expected_frequency.total_seconds())
                        
                        impact = self._assess_gap_impact(
                            current_obs.timestamp, next_obs.timestamp,
                            data_type, source_system
                        )
                        
                        gap = TemporalGap(
                            start_time=current_obs.timestamp,
                            end_time=next_obs.timestamp,
                            duration=gap_duration,
                            data_type=data_type,
                            source_system=source_system,
                            expected_observations=expected_obs,
                            severity=severity,
                            impact_assessment=impact
                        )
                        gaps.append(gap)
                
                # Check for gaps at the beginning and end of time range
                if obs_list:
                    first_obs = obs_list[0]
                    if first_obs.timestamp > start_time + expected_frequency:
                        gap_duration = first_obs.timestamp - start_time
                        severity = self._determine_gap_severity(gap_duration)
                        expected_obs = int(gap_duration.total_seconds() / 
                                         expected_frequency.total_seconds())
                        
                        gap = TemporalGap(
                            start_time=start_time,
                            end_time=first_obs.timestamp,
                            duration=gap_duration,
                            data_type=data_type,
                            source_system=source_system,
                            expected_observations=expected_obs,
                            severity=severity,
                            impact_assessment={'type': 'start_gap'}
                        )
                        gaps.append(gap)
                    
                    last_obs = obs_list[-1]
                    if last_obs.timestamp < end_time - expected_frequency:
                        gap_duration = end_time - last_obs.timestamp
                        severity = self._determine_gap_severity(gap_duration)
                        expected_obs = int(gap_duration.total_seconds() / 
                                         expected_frequency.total_seconds())
                        
                        gap = TemporalGap(
                            start_time=last_obs.timestamp,
                            end_time=end_time,
                            duration=gap_duration,
                            data_type=data_type,
                            source_system=source_system,
                            expected_observations=expected_obs,
                            severity=severity,
                            impact_assessment={'type': 'end_gap'}
                        )
                        gaps.append(gap)
        
        return gaps
    
    def _determine_gap_severity(self, gap_duration: timedelta) -> str:
        """Determine severity level of a temporal gap"""
        thresholds = self.config['temporal_analysis']['gap_severity_thresholds']
        
        if gap_duration >= thresholds['critical']:
            return 'critical'
        elif gap_duration >= thresholds['severe']:
            return 'severe'
        elif gap_duration >= thresholds['moderate']:
            return 'moderate'
        else:
            return 'minor'
    
    def _assess_gap_impact(
        self,
        start_time: datetime,
        end_time: datetime,
        data_type: DataType,
        source_system: str
    ) -> Dict[str, Any]:
        """Assess the impact of a temporal gap"""
        gap_duration = end_time - start_time
        
        # Check if gap occurred during critical periods
        critical_periods = self._identify_critical_periods(start_time, end_time, data_type)
        
        # Assess impact based on data type importance
        importance_weights = {
            DataType.SEISMIC: 0.9,      # High importance for earthquake monitoring
            DataType.VOLCANIC: 0.95,    # Critical for eruption forecasting
            DataType.DEFORMATION: 0.7,  # Important for long-term monitoring
            DataType.GEOCHEMICAL: 0.5,  # Lower frequency, less critical gaps
            DataType.THERMAL: 0.8,      # Important for volcanic monitoring
        }
        
        base_impact = importance_weights.get(data_type, 0.6)
        
        # Adjust impact based on critical periods
        if critical_periods:
            base_impact *= 1.5  # Increase impact if during critical period
        
        return {
            'base_impact_score': base_impact,
            'critical_periods': critical_periods,
            'duration_hours': gap_duration.total_seconds() / 3600,
            'affected_analyses': self._identify_affected_analyses(data_type, gap_duration)
        }
    
    def _identify_critical_periods(
        self,
        start_time: datetime,
        end_time: datetime,
        data_type: DataType
    ) -> List[str]:
        """Identify if gap occurred during critical monitoring periods"""
        critical_periods = []
        
        # This would typically check against a database of known events
        # For now, we'll use some heuristics
        
        # Check for volcanic activity periods (simplified)
        if data_type in [DataType.VOLCANIC, DataType.SEISMIC, DataType.DEFORMATION]:
            # Check if gap occurred during typical volcanic unrest periods
            # This is a simplified example - real implementation would check event databases
            pass
        
        return critical_periods
    
    def _identify_affected_analyses(self, data_type: DataType, gap_duration: timedelta) -> List[str]:
        """Identify analyses that would be affected by the data gap"""
        affected = []
        
        if data_type == DataType.SEISMIC:
            if gap_duration > timedelta(minutes=10):
                affected.append('earthquake_detection')
            if gap_duration > timedelta(hours=1):
                affected.append('seismic_hazard_assessment')
        
        elif data_type == DataType.VOLCANIC:
            if gap_duration > timedelta(hours=1):
                affected.append('eruption_forecasting')
            if gap_duration > timedelta(days=1):
                affected.append('volcanic_hazard_mapping')
        
        elif data_type == DataType.DEFORMATION:
            if gap_duration > timedelta(days=1):
                affected.append('ground_deformation_analysis')
            if gap_duration > timedelta(weeks=1):
                affected.append('long_term_trend_analysis')
        
        return affected   
 async def _analyze_spatial_gaps(
        self,
        observations: List[GeologicalObservation],
        spatial_bounds: Tuple[float, float, float, float]
    ) -> List[SpatialGap]:
        """Analyze spatial gaps in data coverage"""
        gaps = []
        min_lat, max_lat, min_lon, max_lon = spatial_bounds
        
        # Group observations by data type
        grouped_obs = defaultdict(list)
        for obs in observations:
            grouped_obs[obs.data_type].append(obs)
        
        # Analyze spatial coverage for each data type
        for data_type, obs_list in grouped_obs.items():
            if not obs_list:
                continue
            
            # Create spatial grid
            grid_resolution = self.config['spatial_analysis']['grid_resolution_km']
            lat_step = grid_resolution / 111.0  # Approximate km to degrees
            lon_step = grid_resolution / (111.0 * np.cos(np.radians((min_lat + max_lat) / 2)))
            
            lat_grid = np.arange(min_lat, max_lat + lat_step, lat_step)
            lon_grid = np.arange(min_lon, max_lon + lon_step, lon_step)
            
            # Create coverage matrix
            coverage_matrix = np.zeros((len(lat_grid) - 1, len(lon_grid) - 1))
            
            # Mark grid cells with observations
            for obs in obs_list:
                lat, lon, _ = obs.location
                lat_idx = int((lat - min_lat) / lat_step)
                lon_idx = int((lon - min_lon) / lon_step)
                
                if (0 <= lat_idx < len(lat_grid) - 1 and 
                    0 <= lon_idx < len(lon_grid) - 1):
                    coverage_matrix[lat_idx, lon_idx] = 1
            
            # Identify gaps (uncovered regions)
            gap_regions = self._identify_gap_regions(
                coverage_matrix, lat_grid, lon_grid, data_type
            )
            
            gaps.extend(gap_regions)
        
        return gaps
    
    def _identify_gap_regions(
        self,
        coverage_matrix: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        data_type: DataType
    ) -> List[SpatialGap]:
        """Identify contiguous regions with no data coverage"""
        gaps = []
        
        # Find connected components of uncovered cells
        from scipy.ndimage import label
        
        # Invert coverage matrix to find gaps (0 becomes 1, 1 becomes 0)
        gap_matrix = 1 - coverage_matrix
        
        # Label connected components
        labeled_gaps, num_gaps = label(gap_matrix)
        
        for gap_id in range(1, num_gaps + 1):
            # Find cells belonging to this gap
            gap_cells = np.where(labeled_gaps == gap_id)
            
            if len(gap_cells[0]) == 0:
                continue
            
            # Calculate gap bounds
            min_lat_idx, max_lat_idx = np.min(gap_cells[0]), np.max(gap_cells[0])
            min_lon_idx, max_lon_idx = np.min(gap_cells[1]), np.max(gap_cells[1])
            
            gap_min_lat = lat_grid[min_lat_idx]
            gap_max_lat = lat_grid[max_lat_idx + 1]
            gap_min_lon = lon_grid[min_lon_idx]
            gap_max_lon = lon_grid[max_lon_idx + 1]
            
            # Calculate area
            area_km2 = self._calculate_area_km2(
                gap_min_lat, gap_max_lat, gap_min_lon, gap_max_lon
            )
            
            # Skip very small gaps
            if area_km2 < 100:  # Less than 100 km²
                continue
            
            # Determine expected and actual stations
            expected_stations = self._estimate_expected_stations(area_km2, data_type)
            actual_stations = 0  # By definition, gaps have no stations
            
            coverage_percentage = 0.0
            priority_level = self._determine_spatial_priority(area_km2, data_type, gap_min_lat, gap_max_lat, gap_min_lon, gap_max_lon)
            
            gap = SpatialGap(
                region_bounds=(gap_min_lat, gap_max_lat, gap_min_lon, gap_max_lon),
                area_km2=area_km2,
                data_type=data_type,
                expected_stations=expected_stations,
                actual_stations=actual_stations,
                coverage_percentage=coverage_percentage,
                priority_level=priority_level
            )
            
            gaps.append(gap)
        
        return gaps
    
    def _calculate_area_km2(
        self, 
        min_lat: float, 
        max_lat: float, 
        min_lon: float, 
        max_lon: float
    ) -> float:
        """Calculate area of a geographic region in km²"""
        # Simplified calculation using spherical approximation
        R = 6371  # Earth radius in km
        
        lat_diff = np.radians(max_lat - min_lat)
        lon_diff = np.radians(max_lon - min_lon)
        avg_lat = np.radians((min_lat + max_lat) / 2)
        
        area = R**2 * lat_diff * lon_diff * np.cos(avg_lat)
        return abs(area)
    
    def _estimate_expected_stations(self, area_km2: float, data_type: DataType) -> int:
        """Estimate expected number of monitoring stations for an area"""
        # Station density per 1000 km² by data type
        station_densities = {
            DataType.SEISMIC: 2.0,      # 2 stations per 1000 km²
            DataType.VOLCANIC: 5.0,     # 5 stations per 1000 km² in volcanic areas
            DataType.DEFORMATION: 1.0,  # 1 GPS station per 1000 km²
            DataType.GEOCHEMICAL: 0.1,  # 1 sampling site per 10,000 km²
            DataType.THERMAL: 0.5,      # 1 thermal sensor per 2000 km²
            DataType.MAGNETIC: 0.2,     # 1 magnetometer per 5000 km²
            DataType.GRAVITY: 0.1,      # 1 gravimeter per 10,000 km²
        }
        
        density = station_densities.get(data_type, 1.0)
        expected = int(area_km2 / 1000.0 * density)
        
        return max(1, expected)  # At least 1 station expected
    
    def _determine_spatial_priority(
        self,
        area_km2: float,
        data_type: DataType,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float
    ) -> str:
        """Determine priority level for spatial gap"""
        base_priority = 1.0
        
        # Check if gap is in critical regions
        for region in self.critical_regions:
            region_bounds = region['bounds']
            if (min_lat >= region_bounds[0] and max_lat <= region_bounds[1] and
                min_lon >= region_bounds[2] and max_lon <= region_bounds[3]):
                if data_type in region['types']:
                    if region['priority'] == 'critical':
                        base_priority *= 3.0
                    elif region['priority'] == 'high':
                        base_priority *= 2.0
                    elif region['priority'] == 'medium':
                        base_priority *= 1.5
        
        # Adjust for area size
        if area_km2 > 10000:  # Large gaps are higher priority
            base_priority *= 1.5
        elif area_km2 > 50000:  # Very large gaps
            base_priority *= 2.0
        
        # Convert to priority level
        if base_priority >= 3.0:
            return 'critical'
        elif base_priority >= 2.0:
            return 'high'
        elif base_priority >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_temporal_completeness(
        self,
        observations: List[GeologicalObservation],
        time_range: Tuple[datetime, datetime],
        temporal_gaps: List[TemporalGap]
    ) -> float:
        """Calculate overall temporal completeness score"""
        start_time, end_time = time_range
        total_duration = (end_time - start_time).total_seconds()
        
        if total_duration == 0:
            return 1.0
        
        # Calculate total gap duration
        total_gap_duration = sum(gap.duration.total_seconds() for gap in temporal_gaps)
        
        # Calculate completeness
        completeness = max(0.0, 1.0 - (total_gap_duration / total_duration))
        
        return completeness
    
    def _calculate_spatial_completeness(
        self,
        observations: List[GeologicalObservation],
        spatial_bounds: Tuple[float, float, float, float],
        spatial_gaps: List[SpatialGap]
    ) -> float:
        """Calculate overall spatial completeness score"""
        min_lat, max_lat, min_lon, max_lon = spatial_bounds
        total_area = self._calculate_area_km2(min_lat, max_lat, min_lon, max_lon)
        
        if total_area == 0:
            return 1.0
        
        # Calculate total gap area
        total_gap_area = sum(gap.area_km2 for gap in spatial_gaps)
        
        # Calculate completeness
        completeness = max(0.0, 1.0 - (total_gap_area / total_area))
        
        return completeness
    
    def _analyze_source_completeness(self, observations: List[GeologicalObservation]) -> float:
        """Analyze completeness across different data sources"""
        if not observations:
            return 0.0
        
        # Group by data type and source
        source_coverage = defaultdict(set)
        for obs in observations:
            source_coverage[obs.data_type].add(obs.source_system)
        
        # Expected minimum sources per data type
        expected_sources = {
            DataType.SEISMIC: 2,      # At least 2 seismic networks
            DataType.VOLCANIC: 3,     # Multiple volcanic monitoring sources
            DataType.DEFORMATION: 2,  # GPS and InSAR sources
            DataType.GEOCHEMICAL: 2,  # Multiple laboratory sources
            DataType.THERMAL: 2,      # Multiple thermal monitoring sources
        }
        
        completeness_scores = []
        for data_type, sources in source_coverage.items():
            expected = expected_sources.get(data_type, 1)
            actual = len(sources)
            score = min(1.0, actual / expected)
            completeness_scores.append(score)
        
        return np.mean(completeness_scores) if completeness_scores else 0.0
    
    def _generate_availability_matrix(
        self,
        observations: List[GeologicalObservation],
        time_range: Tuple[datetime, datetime]
    ) -> pd.DataFrame:
        """Generate data availability matrix showing coverage over time"""
        start_time, end_time = time_range
        
        # Create time bins (daily resolution)
        time_bins = pd.date_range(start=start_time, end=end_time, freq='D')
        
        # Get unique combinations of data type and source
        type_source_combinations = set()
        for obs in observations:
            type_source_combinations.add((obs.data_type.value, obs.source_system))
        
        # Create matrix
        matrix_data = {}
        for data_type, source in type_source_combinations:
            column_name = f"{data_type}_{source}"
            matrix_data[column_name] = [0] * len(time_bins)
        
        # Fill matrix with availability data
        for obs in observations:
            column_name = f"{obs.data_type.value}_{obs.source_system}"
            # Find appropriate time bin
            time_bin_idx = pd.Index(time_bins).get_loc(obs.timestamp, method='nearest')
            if 0 <= time_bin_idx < len(time_bins):
                matrix_data[column_name][time_bin_idx] = 1
        
        return pd.DataFrame(matrix_data, index=time_bins)
    
    def _create_missing_data_summary(
        self,
        observations: List[GeologicalObservation],
        temporal_gaps: List[TemporalGap],
        spatial_gaps: List[SpatialGap]
    ) -> Dict[str, Any]:
        """Create summary of missing data patterns"""
        # Count observations by type and source
        obs_counts = defaultdict(lambda: defaultdict(int))
        for obs in observations:
            obs_counts[obs.data_type.value][obs.source_system] += 1
        
        # Summarize temporal gaps
        temporal_summary = defaultdict(lambda: {'count': 0, 'total_duration_hours': 0})
        for gap in temporal_gaps:
            key = f"{gap.data_type.value}_{gap.source_system}"
            temporal_summary[key]['count'] += 1
            temporal_summary[key]['total_duration_hours'] += gap.duration.total_seconds() / 3600
        
        # Summarize spatial gaps
        spatial_summary = defaultdict(lambda: {'count': 0, 'total_area_km2': 0})
        for gap in spatial_gaps:
            key = gap.data_type.value
            spatial_summary[key]['count'] += 1
            spatial_summary[key]['total_area_km2'] += gap.area_km2
        
        return {
            'observation_counts': dict(obs_counts),
            'temporal_gap_summary': dict(temporal_summary),
            'spatial_gap_summary': dict(spatial_summary),
            'total_temporal_gaps': len(temporal_gaps),
            'total_spatial_gaps': len(spatial_gaps),
            'most_affected_data_type': self._find_most_affected_data_type(temporal_gaps, spatial_gaps)
        }
    
    def _find_most_affected_data_type(
        self, 
        temporal_gaps: List[TemporalGap], 
        spatial_gaps: List[SpatialGap]
    ) -> str:
        """Find the data type most affected by gaps"""
        gap_counts = defaultdict(int)
        
        for gap in temporal_gaps:
            gap_counts[gap.data_type.value] += 1
        
        for gap in spatial_gaps:
            gap_counts[gap.data_type.value] += 1
        
        if not gap_counts:
            return "none"
        
        return max(gap_counts.items(), key=lambda x: x[1])[0]
    
    def _identify_priority_gaps(
        self, 
        temporal_gaps: List[TemporalGap], 
        spatial_gaps: List[SpatialGap]
    ) -> List[Dict[str, Any]]:
        """Identify highest priority gaps for attention"""
        priority_gaps = []
        
        # Add critical temporal gaps
        for gap in temporal_gaps:
            if gap.severity in ['critical', 'severe']:
                priority_gaps.append({
                    'type': 'temporal',
                    'severity': gap.severity,
                    'data_type': gap.data_type.value,
                    'source_system': gap.source_system,
                    'duration_hours': gap.duration.total_seconds() / 3600,
                    'expected_observations': gap.expected_observations,
                    'priority_score': self._calculate_temporal_priority_score(gap)
                })
        
        # Add high priority spatial gaps
        for gap in spatial_gaps:
            if gap.priority_level in ['critical', 'high']:
                priority_gaps.append({
                    'type': 'spatial',
                    'priority_level': gap.priority_level,
                    'data_type': gap.data_type.value,
                    'area_km2': gap.area_km2,
                    'expected_stations': gap.expected_stations,
                    'priority_score': self._calculate_spatial_priority_score(gap)
                })
        
        # Sort by priority score
        priority_gaps.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return priority_gaps[:10]  # Return top 10 priority gaps
    
    def _calculate_temporal_priority_score(self, gap: TemporalGap) -> float:
        """Calculate priority score for temporal gap"""
        severity_weights = {'minor': 1, 'moderate': 2, 'severe': 4, 'critical': 8}
        base_score = severity_weights.get(gap.severity, 1)
        
        # Adjust for data type importance
        type_weights = {
            DataType.VOLCANIC: 3.0,
            DataType.SEISMIC: 2.5,
            DataType.DEFORMATION: 2.0,
            DataType.THERMAL: 1.5,
            DataType.GEOCHEMICAL: 1.0
        }
        
        type_weight = type_weights.get(gap.data_type, 1.0)
        
        return base_score * type_weight
    
    def _calculate_spatial_priority_score(self, gap: SpatialGap) -> float:
        """Calculate priority score for spatial gap"""
        priority_weights = {'low': 1, 'medium': 2, 'high': 4, 'critical': 8}
        base_score = priority_weights.get(gap.priority_level, 1)
        
        # Adjust for area size
        area_factor = min(2.0, gap.area_km2 / 10000)  # Larger gaps get higher scores
        
        return base_score * area_factor
    
    def _generate_improvement_recommendations(
        self,
        temporal_gaps: List[TemporalGap],
        spatial_gaps: List[SpatialGap],
        overall_completeness: float
    ) -> List[str]:
        """Generate recommendations for improving data completeness"""
        recommendations = []
        
        if overall_completeness < 0.7:
            recommendations.append("Overall data completeness is below acceptable threshold (70%). Immediate action required.")
        
        # Temporal gap recommendations
        critical_temporal = [g for g in temporal_gaps if g.severity == 'critical']
        if critical_temporal:
            recommendations.append(f"Address {len(critical_temporal)} critical temporal gaps immediately.")
            
            # Identify most problematic sources
            problem_sources = defaultdict(int)
            for gap in critical_temporal:
                problem_sources[gap.source_system] += 1
            
            worst_source = max(problem_sources.items(), key=lambda x: x[1])
            recommendations.append(f"Source '{worst_source[0]}' has {worst_source[1]} critical gaps - investigate system reliability.")
        
        # Spatial gap recommendations
        critical_spatial = [g for g in spatial_gaps if g.priority_level == 'critical']
        if critical_spatial:
            recommendations.append(f"Deploy monitoring equipment in {len(critical_spatial)} critical spatial gaps.")
            
            total_critical_area = sum(g.area_km2 for g in critical_spatial)
            recommendations.append(f"Critical spatial gaps cover {total_critical_area:.0f} km² - prioritize largest gaps first.")
        
        # Data type specific recommendations
        type_gap_counts = defaultdict(int)
        for gap in temporal_gaps + spatial_gaps:
            type_gap_counts[gap.data_type if hasattr(gap, 'data_type') else gap.data_type] += 1
        
        if type_gap_counts:
            worst_type = max(type_gap_counts.items(), key=lambda x: x[1])
            recommendations.append(f"Data type '{worst_type[0].value if hasattr(worst_type[0], 'value') else worst_type[0]}' has the most gaps ({worst_type[1]}) - review monitoring strategy.")
        
        return recommendations
    
    def _estimate_cost_impact(
        self,
        temporal_gaps: List[TemporalGap],
        spatial_gaps: List[SpatialGap]
    ) -> Dict[str, float]:
        """Estimate cost impact of data gaps"""
        # Simplified cost estimation - would be more sophisticated in practice
        
        # Cost per missing observation by data type (in arbitrary units)
        observation_costs = {
            DataType.SEISMIC: 0.1,
            DataType.VOLCANIC: 0.5,
            DataType.DEFORMATION: 1.0,
            DataType.GEOCHEMICAL: 10.0,
            DataType.THERMAL: 0.3
        }
        
        temporal_cost = 0.0
        for gap in temporal_gaps:
            cost_per_obs = observation_costs.get(gap.data_type, 1.0)
            temporal_cost += gap.expected_observations * cost_per_obs
        
        # Cost per missing station (in thousands of units)
        station_costs = {
            DataType.SEISMIC: 50.0,
            DataType.VOLCANIC: 100.0,
            DataType.DEFORMATION: 30.0,
            DataType.GEOCHEMICAL: 5.0,
            DataType.THERMAL: 20.0
        }
        
        spatial_cost = 0.0
        for gap in spatial_gaps:
            cost_per_station = station_costs.get(gap.data_type, 50.0)
            spatial_cost += gap.expected_stations * cost_per_station
        
        return {
            'temporal_gap_cost': temporal_cost,
            'spatial_gap_cost': spatial_cost,
            'total_estimated_cost': temporal_cost + spatial_cost,
            'currency_unit': 'arbitrary_units'
        }