"""
Comprehensive Hazard Mapping System for Prometheus AI Platform

This module implements multi-hazard risk assessment combining volcanic and seismic risks,
probabilistic hazard mapping with uncertainty visualization, real-time hazard map updates,
and hazard communication for different audiences.

Requirements addressed:
    - 6.4: Produce detailed volcanic hazard maps
- 8.4: Produce real-time risk assessments for different earthquake scenarios
- 4.5: Produce high-resolution resource prospectivity and volcanic hazard maps
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HazardType(Enum):
    """Types of geological hazards"""

    VOLCANIC_ERUPTION = "volcanic_eruption"
    EARTHQUAKE = "earthquake"
    LANDSLIDE = "landslide"
    TSUNAMI = "tsunami"
    LIQUEFACTION = "liquefaction"
    GROUND_DEFORMATION = "ground_deformation"
    PYROCLASTIC_FLOW = "pyroclastic_flow"
    LAHAR = "lahar"
    ASH_FALL = "ash_fall"
    LAVA_FLOW = "lava_flow"


class RiskLevel(Enum):
    """Risk levels for hazard assessment"""

    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5
    EXTREME = 6


class AudienceType(Enum):
    """Different audience types for hazard communication"""

    SCIENTIFIC = "scientific"
    EMERGENCY_MANAGEMENT = "emergency_management"
    PUBLIC = "public"
    MEDIA = "media"
    GOVERNMENT = "government"
    INDUSTRY = "industry"


@dataclass
class GeographicBounds:
    """Geographic boundaries for hazard mapping"""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    resolution: float = 0.001  # degrees


@dataclass
class UncertaintyQuantification:
    """Uncertainty quantification for hazard estimates"""

    mean_value: float
    std_deviation: float
    confidence_interval_95: Tuple[float, float]
    epistemic_uncertainty: float  # Model uncertainty
    aleatory_uncertainty: float  # Natural variability
    total_uncertainty: float


@dataclass
class HazardMap:
    """Complete hazard map with metadata"""

    hazard_type: HazardType
    bounds: GeographicBounds
    grid_data: np.ndarray
    uncertainty_grid: np.ndarray
    metadata: Dict[str, Any]
    creation_time: datetime
    valid_until: datetime
    confidence_level: float


class HazardMappingSystem:
    """
    Comprehensive hazard mapping system that combines volcanic and seismic risks
    with probabilistic assessment and uncertainty quantification.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the hazard mapping system"""
        self.config = config or {}
        self.hazard_models = {}
        self.uncertainty_models = {}
        self.real_time_data_cache = {}
        self.hazard_maps_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Initialize default parameters
        self.default_bounds = GeographicBounds(
            min_lat=-90, max_lat=90, min_lon=-180, max_lon=180, resolution=0.01
        )

        logger.info("Hazard Mapping System initialized")

    async def create_multi_hazard_assessment(
        self,
        bounds: GeographicBounds,
        hazard_types: List[HazardType],
        monitoring_data: Dict[str, Any],
        historical_data: Dict[str, Any],
    ) -> Dict[str, HazardMap]:
        """
        Create comprehensive multi-hazard risk assessment combining volcanic and seismic risks

        Args:
            bounds: Geographic boundaries for assessment
            hazard_types: List of hazard types to assess
            monitoring_data: Real-time monitoring data
            historical_data: Historical hazard data

        Returns:
            Dictionary of hazard maps for each hazard type
        """
        logger.info(
            f"Creating multi-hazard assessment for {len(hazard_types)} hazard types"
        )

        hazard_maps = {}

        # Process each hazard type
        for hazard_type in hazard_types:
            try:
                hazard_map = await self._create_individual_hazard_map(
                    hazard_type, bounds, monitoring_data, historical_data
                )
                hazard_maps[hazard_type.value] = hazard_map

            except Exception as e:
                logger.error(f"Error creating hazard map for {hazard_type}: {e}")
                continue

        # Create combined multi-hazard map
        if len(hazard_maps) > 1:
            combined_map = await self._create_combined_hazard_map(hazard_maps, bounds)
            hazard_maps["combined"] = combined_map

        logger.info(f"Multi-hazard assessment completed with {len(hazard_maps)} maps")
        return hazard_maps

    async def _create_individual_hazard_map(
        self,
        hazard_type: HazardType,
        bounds: GeographicBounds,
        monitoring_data: Dict[str, Any],
        historical_data: Dict[str, Any],
    ) -> HazardMap:
        """Create hazard map for individual hazard type"""

        # Create coordinate grid
        lats = np.arange(bounds.min_lat, bounds.max_lat, bounds.resolution)
        lons = np.arange(bounds.min_lon, bounds.max_lon, bounds.resolution)
        lat_grid, lon_grid = np.meshgrid(lats, lons)

        # Initialize hazard and uncertainty grids
        hazard_grid = np.zeros_like(lat_grid)
        uncertainty_grid = np.zeros_like(lat_grid)

        if hazard_type == HazardType.VOLCANIC_ERUPTION:
            hazard_grid, uncertainty_grid = await self._calculate_volcanic_hazard(
                lat_grid, lon_grid, monitoring_data, historical_data
            )
        elif hazard_type == HazardType.EARTHQUAKE:
            hazard_grid, uncertainty_grid = await self._calculate_seismic_hazard(
                lat_grid, lon_grid, monitoring_data, historical_data
            )
        elif hazard_type in [HazardType.LANDSLIDE, HazardType.LIQUEFACTION]:
            hazard_grid, uncertainty_grid = await self._calculate_secondary_hazard(
                hazard_type, lat_grid, lon_grid, monitoring_data, historical_data
            )

        # Create hazard map object
        hazard_map = HazardMap(
            hazard_type=hazard_type,
            bounds=bounds,
            grid_data=hazard_grid,
            uncertainty_grid=uncertainty_grid,
            metadata={
                "data_sources": list(monitoring_data.keys()),
                "historical_events": len(historical_data.get("events", [])),
                "grid_shape": hazard_grid.shape,
                "max_hazard_value": float(np.max(hazard_grid)),
                "mean_uncertainty": float(np.mean(uncertainty_grid)),
            },
            creation_time=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=24),
            confidence_level=0.85,
        )

        return hazard_map

    async def _calculate_volcanic_hazard(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        monitoring_data: Dict[str, Any],
        historical_data: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate volcanic hazard values for grid points"""

        hazard_grid = np.zeros_like(lat_grid)
        uncertainty_grid = np.zeros_like(lat_grid)

        # Get volcanic sources from monitoring data
        volcanic_sources = monitoring_data.get("volcanic_sources", [])
        seismic_events = monitoring_data.get("seismic_events", [])
        deformation_data = monitoring_data.get("deformation", [])

        for i in range(lat_grid.shape[0]):
            for j in range(lat_grid.shape[1]):
                lat, lon = lat_grid[i, j], lon_grid[i, j]

                # Calculate hazard from volcanic sources
                volcanic_hazard = 0.0
                volcanic_uncertainty = 0.0

                for source in volcanic_sources:
                    source_lat = source.get("latitude", 0)
                    source_lon = source.get("longitude", 0)
                    source_magnitude = source.get("magnitude", 1.0)

                    # Calculate distance
                    distance = np.sqrt(
                        (lat - source_lat) ** 2 + (lon - source_lon) ** 2
                    )

                    # Distance decay model
                    if distance > 0:
                        hazard_contribution = source_magnitude / (distance**2 + 1)
                        volcanic_hazard += hazard_contribution

                        # Uncertainty increases with distance
                        uncertainty_contribution = 0.1 * distance + 0.05
                        volcanic_uncertainty += uncertainty_contribution

                # Incorporate seismic activity
                seismic_influence = self._calculate_seismic_influence(
                    lat, lon, seismic_events
                )
                volcanic_hazard += seismic_influence * 0.3

                # Incorporate deformation data
                deformation_influence = self._calculate_deformation_influence(
                    lat, lon, deformation_data
                )
                volcanic_hazard += deformation_influence * 0.2

                hazard_grid[i, j] = min(volcanic_hazard, 1.0)  # Cap at 1.0
                uncertainty_grid[i, j] = min(volcanic_uncertainty, 0.5)  # Cap at 0.5

        return hazard_grid, uncertainty_grid

    async def _calculate_seismic_hazard(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        monitoring_data: Dict[str, Any],
        historical_data: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate seismic hazard values for grid points"""

        hazard_grid = np.zeros_like(lat_grid)
        uncertainty_grid = np.zeros_like(lat_grid)

        # Get seismic sources and historical earthquakes
        seismic_events = monitoring_data.get("seismic_events", [])
        historical_earthquakes = historical_data.get("earthquakes", [])
        fault_lines = monitoring_data.get("fault_lines", [])

        for i in range(lat_grid.shape[0]):
            for j in range(lat_grid.shape[1]):
                lat, lon = lat_grid[i, j], lon_grid[i, j]

                # Calculate hazard from recent seismic activity
                seismic_hazard = 0.0
                seismic_uncertainty = 0.0

                # Recent seismic events
                for event in seismic_events:
                    event_lat = event.get("latitude", 0)
                    event_lon = event.get("longitude", 0)
                    magnitude = event.get("magnitude", 0)
                    depth = event.get("depth", 10)  # km

                    # Calculate distance
                    distance = np.sqrt((lat - event_lat) ** 2 + (lon - event_lon) ** 2)

                    # Ground motion prediction equation (simplified)
                    if distance > 0 and magnitude > 0:
                        log_pga = (
                            magnitude - 3.5 * np.log10(distance + 5) - 0.006 * depth
                        )
                        pga = 10**log_pga  # Peak ground acceleration

                        # Convert to hazard probability
                        hazard_contribution = min(pga / 100.0, 1.0)  # Normalize
                        seismic_hazard += hazard_contribution

                        # Uncertainty based on distance and magnitude uncertainty
                        uncertainty_contribution = (
                            0.1 + 0.02 * distance + 0.05 / magnitude
                        )
                        seismic_uncertainty += uncertainty_contribution

                # Historical seismicity
                historical_influence = self._calculate_historical_seismic_influence(
                    lat, lon, historical_earthquakes
                )
                seismic_hazard += historical_influence * 0.4

                # Fault proximity
                fault_influence = self._calculate_fault_influence(lat, lon, fault_lines)
                seismic_hazard += fault_influence * 0.3

                hazard_grid[i, j] = min(seismic_hazard, 1.0)  # Cap at 1.0
                uncertainty_grid[i, j] = min(seismic_uncertainty, 0.6)  # Cap at 0.6

        return hazard_grid, uncertainty_grid

    async def _calculate_secondary_hazard(
        self,
        hazard_type: HazardType,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        monitoring_data: Dict[str, Any],
        historical_data: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate secondary hazards (landslides, liquefaction, etc.)"""

        hazard_grid = np.zeros_like(lat_grid)
        uncertainty_grid = np.zeros_like(lat_grid)

        # Get relevant data
        topography = monitoring_data.get("topography", {})
        geology = monitoring_data.get("geology", {})
        precipitation = monitoring_data.get("precipitation", {})

        for i in range(lat_grid.shape[0]):
            for j in range(lat_grid.shape[1]):
                lat, lon = lat_grid[i, j], lon_grid[i, j]

                if hazard_type == HazardType.LANDSLIDE:
                    # Landslide susceptibility based on slope, geology, precipitation
                    slope = topography.get("slope", {}).get(f"{lat},{lon}", 0)
                    rock_type = geology.get("rock_type", {}).get(
                        f"{lat},{lon}", "stable"
                    )
                    rainfall = precipitation.get("annual_mm", {}).get(f"{lat},{lon}", 0)

                    # Simple landslide susceptibility model
                    slope_factor = min(slope / 45.0, 1.0)  # Normalize by 45 degrees
                    geology_factor = 0.8 if rock_type in ["clay", "shale"] else 0.3
                    rainfall_factor = min(rainfall / 2000.0, 1.0)  # Normalize by 2000mm

                    landslide_hazard = slope_factor * geology_factor * rainfall_factor
                    hazard_grid[i, j] = landslide_hazard
                    uncertainty_grid[i, j] = 0.3  # Moderate uncertainty

                elif hazard_type == HazardType.LIQUEFACTION:
                    # Liquefaction susceptibility based on soil type and groundwater
                    soil_type = geology.get("soil_type", {}).get(f"{lat},{lon}", "rock")
                    groundwater_depth = geology.get("groundwater_depth", {}).get(
                        f"{lat},{lon}", 100
                    )

                    # Liquefaction susceptibility
                    if soil_type in ["sand", "silt"] and groundwater_depth < 10:
                        liquefaction_hazard = 0.8
                    elif soil_type in ["sand", "silt"] and groundwater_depth < 30:
                        liquefaction_hazard = 0.5
                    else:
                        liquefaction_hazard = 0.1

                    hazard_grid[i, j] = liquefaction_hazard
                    uncertainty_grid[i, j] = 0.4  # Higher uncertainty for liquefaction

        return hazard_grid, uncertainty_grid

    def _calculate_seismic_influence(
        self, lat: float, lon: float, seismic_events: List[Dict]
    ) -> float:
        """Calculate seismic influence on volcanic hazard"""
        influence = 0.0

        for event in seismic_events:
            event_lat = event.get("latitude", 0)
            event_lon = event.get("longitude", 0)
            magnitude = event.get("magnitude", 0)

            distance = np.sqrt((lat - event_lat) ** 2 + (lon - event_lon) ** 2)

            if distance < 0.5 and magnitude > 3.0:  # Within 50km and M>3
                influence += magnitude / (distance + 1) ** 2

        return min(influence / 10.0, 1.0)  # Normalize

    def _calculate_deformation_influence(
        self, lat: float, lon: float, deformation_data: List[Dict]
    ) -> float:
        """Calculate ground deformation influence on volcanic hazard"""
        influence = 0.0

        for measurement in deformation_data:
            meas_lat = measurement.get("latitude", 0)
            meas_lon = measurement.get("longitude", 0)
            displacement = measurement.get("displacement_mm", 0)

            distance = np.sqrt((lat - meas_lat) ** 2 + (lon - meas_lon) ** 2)

            if distance < 0.2 and abs(displacement) > 10:  # Within 20km and >10mm
                influence += abs(displacement) / (distance + 1) ** 2

        return min(influence / 100.0, 1.0)  # Normalize

    def _calculate_historical_seismic_influence(
        self, lat: float, lon: float, historical_earthquakes: List[Dict]
    ) -> float:
        """Calculate influence from historical seismicity"""
        influence = 0.0

        for earthquake in historical_earthquakes:
            eq_lat = earthquake.get("latitude", 0)
            eq_lon = earthquake.get("longitude", 0)
            magnitude = earthquake.get("magnitude", 0)

            distance = np.sqrt((lat - eq_lat) ** 2 + (lon - eq_lon) ** 2)

            if distance < 2.0 and magnitude > 5.0:  # Within 200km and M>5
                influence += magnitude / (distance + 1)

        return min(influence / 50.0, 1.0)  # Normalize

    def _calculate_fault_influence(
        self, lat: float, lon: float, fault_lines: List[Dict]
    ) -> float:
        """Calculate influence from proximity to fault lines"""
        min_distance = float("in")

        for fault in fault_lines:
            fault_coords = fault.get("coordinates", [])

            for coord in fault_coords:
                fault_lat, fault_lon = coord
                distance = np.sqrt((lat - fault_lat) ** 2 + (lon - fault_lon) ** 2)
                min_distance = min(min_distance, distance)

        if min_distance == float("inf"):
            return 0.0

        # Higher influence for closer faults
        influence = 1.0 / (min_distance + 0.1)
        return min(influence, 1.0)
