"""
Earthquake Impact Modeling System

Implements comprehensive earthquake impact modeling including:
- Ground shaking simulation using physics-based models
- Liquefaction susceptibility mapping and modeling
- Landslide triggering and runout simulation
- Tsunami generation and propagation modeling

Requirements: 8.2, 8.6
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import interpolate, integrate
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import warnings

from models.geological import SeismicEvent, GeologicalLocation


class GroundMotionModel(str, Enum):
    """Ground motion prediction equation models"""
    BOORE_ATKINSON_2008 = "boore_atkinson_2008"
    CAMPBELL_BOZORGNIA_2014 = "campbell_bozorgnia_2014"
    CHIOU_YOUNGS_2014 = "chiou_youngs_2014"
    ASK_2014 = "ask_2014"
    ZHAO_2006 = "zhao_2006"


class SoilType(str, Enum):
    """Soil classification types"""
    ROCK = "rock"  # Vs30 > 760 m/s
    STIFF_SOIL = "stiff_soil"  # 360 < Vs30 <= 760 m/s
    SOFT_SOIL = "soft_soil"  # 180 < Vs30 <= 360 m/s
    VERY_SOFT_SOIL = "very_soft_soil"  # Vs30 <= 180 m/s


class LiquefactionSusceptibility(str, Enum):
    """Liquefaction susceptibility levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class LandslideSusceptibility(str, Enum):
    """Landslide susceptibility levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TsunamiSource(str, Enum):
    """Tsunami generation mechanisms"""
    SUBMARINE_EARTHQUAKE = "submarine_earthquake"
    COASTAL_EARTHQUAKE = "coastal_earthquake"
    LANDSLIDE = "landslide"
    VOLCANIC = "volcanic"


@dataclass
class GroundMotionParameters:
    """
    Ground motion parameters for a specific location.
    
    Attributes:
        location: Geographic location
        peak_ground_acceleration: PGA in g
        peak_ground_velocity: PGV in cm/s
        peak_ground_displacement: PGD in cm
        spectral_accelerations: SA at different periods (s -> g)
        duration: Strong motion duration in seconds
        arias_intensity: Arias intensity in m/s
        uncertainty: Uncertainty in ground motion parameters
    """
    location: GeologicalLocation
    peak_ground_acceleration: float
    peak_ground_velocity: float
    peak_ground_displacement: float
    spectral_accelerations: Dict[float, float]
    duration: float
    arias_intensity: float
    uncertainty: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate ground motion parameters"""
        if self.peak_ground_acceleration < 0:
            raise ValueError("PGA must be non-negative")
        if self.peak_ground_velocity < 0:
            raise ValueError("PGV must be non-negative")
        if self.peak_ground_displacement < 0:
            raise ValueError("PGD must be non-negative")
        if self.duration < 0:
            raise ValueError("Duration must be non-negative")
        if self.arias_intensity < 0:
            raise ValueError("Arias intensity must be non-negative")


@dataclass
class LiquefactionAssessment:
    """
    Liquefaction susceptibility assessment for a location.
    
    Attributes:
        location: Geographic location
        susceptibility: Liquefaction susceptibility level
        factor_of_safety: Factor of safety against liquefaction
        probability: Probability of liquefaction occurrence
        contributing_factors: Factors influencing liquefaction
        soil_properties: Relevant soil properties
        groundwater_depth: Depth to groundwater table in meters
    """
    location: GeologicalLocation
    susceptibility: LiquefactionSusceptibility
    factor_of_safety: float
    probability: float
    contributing_factors: List[str]
    soil_properties: Dict[str, float]
    groundwater_depth: Optional[float] = None
    
    def __post_init__(self):
        """Validate liquefaction assessment"""
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1")
        if self.factor_of_safety < 0:
            raise ValueError("Factor of safety must be non-negative")


@dataclass
class LandslideAssessment:
    """
    Landslide susceptibility and runout assessment.
    
    Attributes:
        location: Geographic location (initiation point)
        susceptibility: Landslide susceptibility level
        trigger_threshold: Ground motion threshold for triggering
        probability: Probability of landslide occurrence
        volume_estimate: Estimated landslide volume in cubic meters
        runout_distance: Estimated runout distance in meters
        runout_path: Predicted runout path coordinates
        impact_area: Area potentially impacted by landslide
    """
    location: GeologicalLocation
    susceptibility: LandslideSusceptibility
    trigger_threshold: float
    probability: float
    volume_estimate: float
    runout_distance: float
    runout_path: List[GeologicalLocation]
    impact_area: float
    
    def __post_init__(self):
        """Validate landslide assessment"""
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1")
        if self.trigger_threshold < 0:
            raise ValueError("Trigger threshold must be non-negative")
        if self.volume_estimate < 0:
            raise ValueError("Volume estimate must be non-negative")
        if self.runout_distance < 0:
            raise ValueError("Runout distance must be non-negative")
        if self.impact_area < 0:
            raise ValueError("Impact area must be non-negative")


@dataclass
class TsunamiParameters:
    """
    Tsunami generation and propagation parameters.
    
    Attributes:
        source_location: Tsunami source location
        source_type: Type of tsunami generation mechanism
        initial_wave_height: Initial wave height in meters
        wave_period: Dominant wave period in seconds
        propagation_speed: Wave propagation speed in m/s
        arrival_times: Arrival times at different locations
        wave_heights: Wave heights at different locations
        inundation_distances: Inland inundation distances
        affected_coastline: Length of affected coastline in km
    """
    source_location: GeologicalLocation
    source_type: TsunamiSource
    initial_wave_height: float
    wave_period: float
    propagation_speed: float
    arrival_times: Dict[str, float]  # location_id -> arrival_time_minutes
    wave_heights: Dict[str, float]  # location_id -> wave_height_meters
    inundation_distances: Dict[str, float]  # location_id -> distance_meters
    affected_coastline: float
    
    def __post_init__(self):
        """Validate tsunami parameters"""
        if self.initial_wave_height < 0:
            raise ValueError("Initial wave height must be non-negative")
        if self.wave_period <= 0:
            raise ValueError("Wave period must be positive")
        if self.propagation_speed <= 0:
            raise ValueError("Propagation speed must be positive")
        if self.affected_coastline < 0:
            raise ValueError("Affected coastline must be non-negative")


class EarthquakeImpactModelingEngine:
    """
    Main engine for earthquake impact modeling and assessment.
    
    Implements physics-based models for ground shaking, liquefaction,
    landslides, and tsunami generation and propagation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the earthquake impact modeling engine.
        
        Args:
            config: Configuration parameters for impact models
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize model parameters
        self.ground_motion_models = {}
        self.soil_database = {}
        self.topography_data = {}
        self.bathymetry_data = {}
        
        self.logger.info("Earthquake impact modeling engine initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration parameters"""
        return {
            "ground_motion": {
                "default_model": GroundMotionModel.BOORE_ATKINSON_2008,
                "site_amplification": True,
                "directivity_effects": True,
                "hanging_wall_effects": True
            },
            "liquefaction": {
                "groundwater_threshold": 10.0,  # meters
                "critical_csr": 0.65,  # Critical cyclic stress ratio
                "magnitude_scaling_factor": True
            },
            "landslide": {
                "slope_threshold": 15.0,  # degrees
                "cohesion_threshold": 5000.0,  # Pa
                "friction_angle_threshold": 30.0,  # degrees
                "runout_model": "empirical"
            },
            "tsunami": {
                "minimum_magnitude": 7.0,
                "minimum_depth": 50.0,  # km
                "coastal_amplification": True,
                "shoaling_effects": True
            }
        }
    
    def simulate_ground_shaking(
        self,
        earthquake: SeismicEvent,
        target_locations: List[GeologicalLocation],
        soil_conditions: Optional[Dict[str, SoilType]] = None,
        model: GroundMotionModel = GroundMotionModel.BOORE_ATKINSON_2008
    ) -> List[GroundMotionParameters]:
        """
        Simulate ground shaking using physics-based models.
        
        Args:
            earthquake: Source earthquake event
            target_locations: Locations to calculate ground motion
            soil_conditions: Soil conditions at target locations
            model: Ground motion prediction equation to use
            
        Returns:
            List of ground motion parameters for each location
        """
        self.logger.info(f"Simulating ground shaking for M{earthquake.magnitude} earthquake")
        
        if earthquake.magnitude < 3.0:
            raise ValueError("Earthquake magnitude too small for meaningful ground motion simulation")
        
        ground_motion_results = []
        
        for i, location in enumerate(target_locations):
            # Calculate source-to-site distance
            distance = self._calculate_distance(earthquake.location, location)
            
            # Get soil conditions
            soil_type = SoilType.ROCK  # Default
            if soil_conditions and str(i) in soil_conditions:
                soil_type = soil_conditions[str(i)]
            
            # Calculate ground motion parameters using selected model
            if model == GroundMotionModel.BOORE_ATKINSON_2008:
                gm_params = self._boore_atkinson_2008(earthquake, location, distance, soil_type)
            elif model == GroundMotionModel.CAMPBELL_BOZORGNIA_2014:
                gm_params = self._campbell_bozorgnia_2014(earthquake, location, distance, soil_type)
            elif model == GroundMotionModel.CHIOU_YOUNGS_2014:
                gm_params = self._chiou_youngs_2014(earthquake, location, distance, soil_type)
            else:
                # Default to Boore-Atkinson 2008
                gm_params = self._boore_atkinson_2008(earthquake, location, distance, soil_type)
            
            ground_motion_results.append(gm_params)
        
        self.logger.info(f"Calculated ground motion for {len(target_locations)} locations")
        return ground_motion_results
    
    def assess_liquefaction_susceptibility(
        self,
        ground_motion: List[GroundMotionParameters],
        soil_properties: Dict[str, Dict[str, float]],
        groundwater_data: Optional[Dict[str, float]] = None
    ) -> List[LiquefactionAssessment]:
        """
        Assess liquefaction susceptibility and probability.
        
        Args:
            ground_motion: Ground motion parameters at each location
            soil_properties: Soil properties for each location
            groundwater_data: Groundwater depth data
            
        Returns:
            List of liquefaction assessments
        """
        self.logger.info(f"Assessing liquefaction susceptibility for {len(ground_motion)} locations")
        
        liquefaction_results = []
        
        for i, gm in enumerate(ground_motion):
            location_id = str(i)
            
            # Get soil properties
            if location_id not in soil_properties:
                self.logger.warning(f"No soil properties for location {i}, skipping")
                continue
            
            soil_props = soil_properties[location_id]
            
            # Get groundwater depth
            gw_depth = groundwater_data.get(location_id, 5.0) if groundwater_data else 5.0
            
            # Calculate cyclic stress ratio (CSR)
            csr = self._calculate_cyclic_stress_ratio(gm, soil_props, gw_depth)
            
            # Calculate cyclic resistance ratio (CRR)
            crr = self._calculate_cyclic_resistance_ratio(soil_props)
            
            # Calculate factor of safety
            factor_of_safety = crr / max(csr, 1e-6)  # Avoid division by zero
            
            # Calculate probability of liquefaction
            probability = self._calculate_liquefaction_probability(factor_of_safety, gm)
            
            # Determine susceptibility level
            susceptibility = self._determine_liquefaction_susceptibility(
                factor_of_safety, probability, soil_props, gw_depth
            )
            
            # Identify contributing factors
            contributing_factors = self._identify_liquefaction_factors(
                gm, soil_props, gw_depth, factor_of_safety
            )
            
            assessment = LiquefactionAssessment(
                location=gm.location,
                susceptibility=susceptibility,
                factor_of_safety=factor_of_safety,
                probability=probability,
                contributing_factors=contributing_factors,
                soil_properties=soil_props,
                groundwater_depth=gw_depth
            )
            
            liquefaction_results.append(assessment)
        
        self.logger.info(f"Completed liquefaction assessment for {len(liquefaction_results)} locations")
        return liquefaction_results
    
    def model_landslide_triggering(
        self,
        ground_motion: List[GroundMotionParameters],
        slope_data: Dict[str, float],
        soil_strength: Dict[str, Dict[str, float]],
        topography: Optional[Dict[str, np.ndarray]] = None
    ) -> List[LandslideAssessment]:
        """
        Model landslide triggering and runout simulation.
        
        Args:
            ground_motion: Ground motion parameters
            slope_data: Slope angles for each location (degrees)
            soil_strength: Soil strength parameters
            topography: Digital elevation model data
            
        Returns:
            List of landslide assessments
        """
        self.logger.info(f"Modeling landslide triggering for {len(ground_motion)} locations")
        
        landslide_results = []
        
        for i, gm in enumerate(ground_motion):
            location_id = str(i)
            
            # Get slope data
            if location_id not in slope_data:
                continue
            
            slope_angle = slope_data[location_id]
            
            # Skip if slope is too gentle
            if slope_angle < self.config["landslide"]["slope_threshold"]:
                continue
            
            # Get soil strength parameters
            if location_id not in soil_strength:
                continue
            
            strength_params = soil_strength[location_id]
            
            # Calculate factor of safety using infinite slope model
            factor_of_safety = self._calculate_slope_stability(
                slope_angle, strength_params, gm
            )
            
            # Calculate trigger threshold
            trigger_threshold = self._calculate_landslide_trigger_threshold(
                slope_angle, strength_params
            )
            
            # Calculate probability of landslide
            probability = self._calculate_landslide_probability(
                gm, factor_of_safety, trigger_threshold
            )
            
            # Estimate landslide volume
            volume_estimate = self._estimate_landslide_volume(
                slope_angle, strength_params, gm
            )
            
            # Calculate runout distance and path
            runout_distance, runout_path = self._calculate_landslide_runout(
                gm.location, slope_angle, volume_estimate, topography
            )
            
            # Calculate impact area
            impact_area = self._calculate_landslide_impact_area(
                runout_distance, volume_estimate
            )
            
            # Determine susceptibility
            susceptibility = self._determine_landslide_susceptibility(
                factor_of_safety, probability, slope_angle
            )
            
            assessment = LandslideAssessment(
                location=gm.location,
                susceptibility=susceptibility,
                trigger_threshold=trigger_threshold,
                probability=probability,
                volume_estimate=volume_estimate,
                runout_distance=runout_distance,
                runout_path=runout_path,
                impact_area=impact_area
            )
            
            landslide_results.append(assessment)
        
        self.logger.info(f"Completed landslide assessment for {len(landslide_results)} locations")
        return landslide_results
    
    def model_tsunami_generation(
        self,
        earthquake: SeismicEvent,
        coastal_locations: List[GeologicalLocation],
        bathymetry: Optional[Dict[str, np.ndarray]] = None
    ) -> Optional[TsunamiParameters]:
        """
        Model tsunami generation and propagation.
        
        Args:
            earthquake: Source earthquake event
            coastal_locations: Coastal locations for impact assessment
            bathymetry: Bathymetry data for wave propagation
            
        Returns:
            Tsunami parameters if tsunami is generated, None otherwise
        """
        self.logger.info(f"Modeling tsunami generation for M{earthquake.magnitude} earthquake")
        
        # Check if earthquake can generate tsunami
        if not self._can_generate_tsunami(earthquake):
            self.logger.info("Earthquake unlikely to generate significant tsunami")
            return None
        
        # Determine tsunami source type
        source_type = self._determine_tsunami_source_type(earthquake)
        
        # Calculate initial wave height using empirical relationships
        initial_wave_height = self._calculate_initial_tsunami_height(earthquake)
        
        # Calculate wave period
        wave_period = self._calculate_tsunami_period(earthquake)
        
        # Calculate propagation speed (shallow water approximation)
        propagation_speed = self._calculate_tsunami_speed(earthquake.depth)
        
        # Calculate arrival times at coastal locations
        arrival_times = {}
        wave_heights = {}
        inundation_distances = {}
        
        for i, location in enumerate(coastal_locations):
            location_id = f"coastal_{i}"
            
            # Calculate distance from source
            distance = self._calculate_distance(earthquake.location, location)
            
            # Calculate arrival time
            arrival_time = distance * 1000 / propagation_speed / 60  # minutes
            arrival_times[location_id] = arrival_time
            
            # Calculate wave height at location (with geometric spreading)
            wave_height = self._calculate_tsunami_wave_height(
                initial_wave_height, distance, earthquake.depth, bathymetry
            )
            wave_heights[location_id] = wave_height
            
            # Calculate inundation distance
            inundation_distance = self._calculate_tsunami_inundation(
                wave_height, location, bathymetry
            )
            inundation_distances[location_id] = inundation_distance
        
        # Calculate affected coastline length
        affected_coastline = self._calculate_affected_coastline(
            earthquake, coastal_locations, wave_heights
        )
        
        tsunami_params = TsunamiParameters(
            source_location=earthquake.location,
            source_type=source_type,
            initial_wave_height=initial_wave_height,
            wave_period=wave_period,
            propagation_speed=propagation_speed,
            arrival_times=arrival_times,
            wave_heights=wave_heights,
            inundation_distances=inundation_distances,
            affected_coastline=affected_coastline
        )
        
        self.logger.info(f"Generated tsunami with initial height {initial_wave_height:.2f}m")
        return tsunami_params
    
    def _calculate_distance(self, loc1: GeologicalLocation, loc2: GeologicalLocation) -> float:
        """Calculate distance between two locations in km"""
        # Haversine formula
        R = 6371  # Earth's radius in km
        
        lat1, lon1 = np.radians(loc1.latitude), np.radians(loc1.longitude)
        lat2, lon2 = np.radians(loc2.latitude), np.radians(loc2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _boore_atkinson_2008(
        self,
        earthquake: SeismicEvent,
        location: GeologicalLocation,
        distance: float,
        soil_type: SoilType
    ) -> GroundMotionParameters:
        """
        Calculate ground motion using Boore-Atkinson 2008 GMPE.
        
        This is a simplified implementation of the Boore-Atkinson 2008
        ground motion prediction equation.
        """
        M = earthquake.magnitude
        R = distance
        
        # Vs30 values for different soil types
        vs30_map = {
            SoilType.ROCK: 800.0,
            SoilType.STIFF_SOIL: 500.0,
            SoilType.SOFT_SOIL: 270.0,
            SoilType.VERY_SOFT_SOIL: 150.0
        }
        
        Vs30 = vs30_map[soil_type]
        
        # Simplified PGA calculation (natural log units, then convert to g)
        # This is a very simplified version of Boore-Atkinson 2008
        ln_pga = (
            -3.512 + 0.904 * M - 1.328 * np.log(R + 8.5) +
            0.149 * np.exp(0.647 * M) * 0.1 + 0.2 * np.log(Vs30/760)
        )
        
        pga = np.exp(ln_pga)  # Convert to g
        pga = max(min(pga, 2.0), 0.001)  # Clamp between reasonable values
        
        # Simplified PGV calculation
        ln_pgv = (
            -2.895 + 0.510 * M - 1.303 * np.log(R + 5.0) +
            0.270 * np.log(Vs30/760)
        )
        
        pgv = np.exp(ln_pgv)  # Convert to cm/s
        pgv = max(min(pgv, 200.0), 0.1)  # Clamp between reasonable values
        
        # Simplified PGD calculation
        pgd = pgv * 0.5  # Rough approximation
        
        # Calculate spectral accelerations at different periods
        periods = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        spectral_accelerations = {}
        
        for T in periods:
            if T <= 0.5:
                # Short period approximation
                sa = pga * (1 + 2 * T)
            else:
                # Long period approximation
                sa = pga * (2.5 / T)
            
            spectral_accelerations[T] = max(sa, 0.01)  # Minimum SA
        
        # Calculate duration (simplified)
        duration = 5.0 + 0.5 * M + 0.1 * R
        
        # Calculate Arias intensity
        arias_intensity = pga**2 * duration * np.pi / (2 * 9.81)
        
        # Estimate uncertainties
        uncertainty = {
            "pga": 0.3 * pga,  # 30% uncertainty
            "pgv": 0.3 * pgv,
            "pgd": 0.4 * pgd,
            "duration": 0.2 * duration
        }
        
        return GroundMotionParameters(
            location=location,
            peak_ground_acceleration=pga,
            peak_ground_velocity=pgv,
            peak_ground_displacement=pgd,
            spectral_accelerations=spectral_accelerations,
            duration=duration,
            arias_intensity=arias_intensity,
            uncertainty=uncertainty
        )
    
    def _campbell_bozorgnia_2014(
        self,
        earthquake: SeismicEvent,
        location: GeologicalLocation,
        distance: float,
        soil_type: SoilType
    ) -> GroundMotionParameters:
        """
        Calculate ground motion using Campbell-Bozorgnia 2014 GMPE.
        
        Simplified implementation for demonstration.
        """
        # For now, use Boore-Atkinson with slight modifications
        ba_result = self._boore_atkinson_2008(earthquake, location, distance, soil_type)
        
        # Apply CB14 adjustments (simplified)
        ba_result.peak_ground_acceleration *= 1.1
        ba_result.peak_ground_velocity *= 1.05
        
        return ba_result
    
    def _chiou_youngs_2014(
        self,
        earthquake: SeismicEvent,
        location: GeologicalLocation,
        distance: float,
        soil_type: SoilType
    ) -> GroundMotionParameters:
        """
        Calculate ground motion using Chiou-Youngs 2014 GMPE.
        
        Simplified implementation for demonstration.
        """
        # For now, use Boore-Atkinson with slight modifications
        ba_result = self._boore_atkinson_2008(earthquake, location, distance, soil_type)
        
        # Apply CY14 adjustments (simplified)
        ba_result.peak_ground_acceleration *= 0.95
        ba_result.peak_ground_velocity *= 1.02
        
        return ba_result
    
    def _calculate_cyclic_stress_ratio(
        self,
        ground_motion: GroundMotionParameters,
        soil_properties: Dict[str, float],
        groundwater_depth: float
    ) -> float:
        """Calculate cyclic stress ratio for liquefaction analysis"""
        # Simplified CSR calculation
        amax = ground_motion.peak_ground_acceleration
        sigma_v = soil_properties.get("total_stress", 100000)  # Pa
        sigma_v_prime = soil_properties.get("effective_stress", 50000)  # Pa
        
        # Stress reduction factor
        rd = 1.0 - 0.00765 * groundwater_depth if groundwater_depth < 9.15 else 0.174
        
        csr = 0.65 * (amax / 9.81) * (sigma_v / sigma_v_prime) * rd
        
        return csr
    
    def _calculate_cyclic_resistance_ratio(self, soil_properties: Dict[str, float]) -> float:
        """Calculate cyclic resistance ratio"""
        # Simplified CRR calculation based on SPT N-value
        n1_60 = soil_properties.get("spt_n_value", 10)
        fines_content = soil_properties.get("fines_content", 10)  # %
        
        if n1_60 < 30:
            crr = (n1_60 / 34 - 0.040) * (10 / (fines_content + 45))**2 + (1 / 130)
        else:
            crr = 0.3  # High resistance for dense soils
        
        return max(crr, 0.05)  # Minimum CRR
    
    def _calculate_liquefaction_probability(
        self,
        factor_of_safety: float,
        ground_motion: GroundMotionParameters
    ) -> float:
        """Calculate probability of liquefaction occurrence"""
        # Simplified probability calculation
        if factor_of_safety >= 1.5:
            probability = 0.0
        elif factor_of_safety <= 0.8:
            probability = 1.0
        else:
            # Linear interpolation between 0.8 and 1.5
            probability = (1.5 - factor_of_safety) / 0.7
        
        # Adjust for ground motion intensity
        intensity_factor = min(ground_motion.peak_ground_acceleration / 0.3, 2.0)
        probability *= intensity_factor
        
        return min(max(probability, 0.0), 1.0)
    
    def _determine_liquefaction_susceptibility(
        self,
        factor_of_safety: float,
        probability: float,
        soil_properties: Dict[str, float],
        groundwater_depth: float
    ) -> LiquefactionSusceptibility:
        """Determine liquefaction susceptibility level"""
        if groundwater_depth > 10:
            return LiquefactionSusceptibility.VERY_LOW
        
        if factor_of_safety > 1.5:
            return LiquefactionSusceptibility.VERY_LOW
        elif factor_of_safety > 1.2:
            return LiquefactionSusceptibility.LOW
        elif factor_of_safety > 1.0:
            return LiquefactionSusceptibility.MODERATE
        elif factor_of_safety > 0.8:
            return LiquefactionSusceptibility.HIGH
        else:
            return LiquefactionSusceptibility.VERY_HIGH
    
    def _identify_liquefaction_factors(
        self,
        ground_motion: GroundMotionParameters,
        soil_properties: Dict[str, float],
        groundwater_depth: float,
        factor_of_safety: float
    ) -> List[str]:
        """Identify factors contributing to liquefaction susceptibility"""
        factors = []
        
        if groundwater_depth < 5:
            factors.append("shallow_groundwater")
        
        if ground_motion.peak_ground_acceleration > 0.2:
            factors.append("high_ground_acceleration")
        
        if ground_motion.duration > 20:
            factors.append("long_duration_shaking")
        
        if soil_properties.get("spt_n_value", 20) < 15:
            factors.append("loose_soil")
        
        if soil_properties.get("fines_content", 10) < 35:
            factors.append("clean_sand")
        
        if factor_of_safety < 1.0:
            factors.append("low_factor_of_safety")
        
        return factors
    
    def _calculate_slope_stability(
        self,
        slope_angle: float,
        strength_params: Dict[str, float],
        ground_motion: GroundMotionParameters
    ) -> float:
        """Calculate slope stability factor of safety"""
        # Infinite slope model with seismic loading
        slope_rad = np.radians(slope_angle)
        
        cohesion = strength_params.get("cohesion", 5000)  # Pa
        friction_angle = strength_params.get("friction_angle", 30)  # degrees
        unit_weight = strength_params.get("unit_weight", 18000)  # N/m³
        thickness = strength_params.get("thickness", 2.0)  # m
        
        friction_rad = np.radians(friction_angle)
        
        # Static factor of safety
        fs_static = (
            (cohesion / (unit_weight * thickness * np.sin(slope_rad))) +
            (np.tan(friction_rad) / np.tan(slope_rad))
        )
        
        # Seismic coefficient
        kh = ground_motion.peak_ground_acceleration / 2  # Horizontal seismic coefficient
        
        # Seismic factor of safety
        fs_seismic = fs_static - (kh / np.tan(slope_rad))
        
        return max(fs_seismic, 0.1)
    
    def _calculate_landslide_trigger_threshold(
        self,
        slope_angle: float,
        strength_params: Dict[str, float]
    ) -> float:
        """Calculate ground motion threshold for landslide triggering"""
        # Simplified threshold based on slope angle and soil strength
        base_threshold = 0.1  # g
        
        slope_factor = (slope_angle / 45) ** 2
        strength_factor = 1.0 / max(strength_params.get("friction_angle", 30) / 30, 0.5)
        
        threshold = base_threshold * slope_factor * strength_factor
        
        return min(max(threshold, 0.05), 0.5)  # Clamp between 0.05g and 0.5g
    
    def _calculate_landslide_probability(
        self,
        ground_motion: GroundMotionParameters,
        factor_of_safety: float,
        trigger_threshold: float
    ) -> float:
        """Calculate probability of landslide occurrence"""
        if ground_motion.peak_ground_acceleration < trigger_threshold:
            return 0.0
        
        if factor_of_safety > 1.2:
            return 0.0
        elif factor_of_safety < 0.8:
            return 1.0
        else:
            # Probability based on factor of safety and ground motion
            fs_prob = (1.2 - factor_of_safety) / 0.4
            gm_prob = min(ground_motion.peak_ground_acceleration / trigger_threshold, 2.0) - 1.0
            
            probability = fs_prob * gm_prob
            
        return min(max(probability, 0.0), 1.0)
    
    def _estimate_landslide_volume(
        self,
        slope_angle: float,
        strength_params: Dict[str, float],
        ground_motion: GroundMotionParameters
    ) -> float:
        """Estimate landslide volume"""
        # Empirical volume estimation
        base_volume = 1000  # m³
        
        slope_factor = (slope_angle / 30) ** 1.5
        intensity_factor = (ground_motion.peak_ground_acceleration / 0.2) ** 0.5
        
        volume = base_volume * slope_factor * intensity_factor
        
        return max(volume, 100)  # Minimum 100 m³
    
    def _calculate_landslide_runout(
        self,
        source_location: GeologicalLocation,
        slope_angle: float,
        volume: float,
        topography: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[float, List[GeologicalLocation]]:
        """Calculate landslide runout distance and path"""
        # Simplified runout calculation
        # In practice, this would use detailed topography and flow models
        
        # Empirical runout distance
        runout_distance = 50 * (volume / 1000) ** 0.3 * (slope_angle / 30) ** 0.5
        
        # Simplified runout path (straight line downslope)
        # In reality, this would follow topographic gradients
        runout_path = [
            source_location,
            GeologicalLocation(
                latitude=source_location.latitude - 0.001 * runout_distance / 111,
                longitude=source_location.longitude,
                elevation=source_location.elevation - runout_distance * np.sin(np.radians(slope_angle))
            )
        ]
        
        return runout_distance, runout_path
    
    def _calculate_landslide_impact_area(self, runout_distance: float, volume: float) -> float:
        """Calculate landslide impact area"""
        # Simplified impact area calculation
        width = (volume / 1000) ** 0.5 * 10  # Empirical width estimation
        area = runout_distance * width
        
        return max(area, 100)  # Minimum 100 m²
    
    def _determine_landslide_susceptibility(
        self,
        factor_of_safety: float,
        probability: float,
        slope_angle: float
    ) -> LandslideSusceptibility:
        """Determine landslide susceptibility level"""
        if slope_angle < 15:
            return LandslideSusceptibility.VERY_LOW
        
        if factor_of_safety > 1.5:
            return LandslideSusceptibility.VERY_LOW
        elif factor_of_safety > 1.2:
            return LandslideSusceptibility.LOW
        elif factor_of_safety > 1.0:
            return LandslideSusceptibility.MODERATE
        elif factor_of_safety > 0.8:
            return LandslideSusceptibility.HIGH
        else:
            return LandslideSusceptibility.VERY_HIGH
    
    def _can_generate_tsunami(self, earthquake: SeismicEvent) -> bool:
        """Check if earthquake can generate significant tsunami"""
        # Basic criteria for tsunami generation
        min_magnitude = self.config["tsunami"]["minimum_magnitude"]
        max_depth = self.config["tsunami"]["minimum_depth"]
        
        # Must be large enough and shallow enough
        if earthquake.magnitude < min_magnitude:
            return False
        
        if earthquake.depth > max_depth:
            return False
        
        # Should be near or under water (simplified check)
        # In practice, would check if epicenter is offshore
        return True
    
    def _determine_tsunami_source_type(self, earthquake: SeismicEvent) -> TsunamiSource:
        """Determine tsunami source mechanism"""
        if earthquake.depth < 20:
            return TsunamiSource.SUBMARINE_EARTHQUAKE
        elif earthquake.depth < 50:
            return TsunamiSource.COASTAL_EARTHQUAKE
        else:
            return TsunamiSource.SUBMARINE_EARTHQUAKE
    
    def _calculate_initial_tsunami_height(self, earthquake: SeismicEvent) -> float:
        """Calculate initial tsunami wave height"""
        # Empirical relationship based on magnitude
        M = earthquake.magnitude
        
        if M < 7.5:
            height = 0.1 * (M - 6.5)
        elif M < 8.5:
            height = 0.5 + 2.0 * (M - 7.5)
        else:
            height = 2.5 + 5.0 * (M - 8.5)
        
        return max(height, 0.1)  # Minimum 0.1m
    
    def _calculate_tsunami_period(self, earthquake: SeismicEvent) -> float:
        """Calculate dominant tsunami wave period"""
        # Empirical relationship
        M = earthquake.magnitude
        period = 300 + 100 * (M - 7.0)  # seconds
        
        return max(period, 300)  # Minimum 5 minutes
    
    def _calculate_tsunami_speed(self, water_depth: float) -> float:
        """Calculate tsunami propagation speed"""
        # Shallow water wave speed: c = sqrt(g * h)
        g = 9.81  # m/s²
        h = max(water_depth * 1000, 100)  # Convert km to m, minimum 100m
        
        speed = np.sqrt(g * h)
        
        return speed
    
    def _calculate_tsunami_wave_height(
        self,
        initial_height: float,
        distance: float,
        source_depth: float,
        bathymetry: Optional[Dict[str, np.ndarray]] = None
    ) -> float:
        """Calculate tsunami wave height at distance"""
        # Simplified geometric spreading
        # In practice, would use detailed bathymetry and wave propagation models
        
        # Geometric spreading factor
        spreading_factor = 1.0 / np.sqrt(max(distance, 1.0))
        
        # Depth factor (deeper sources produce smaller waves)
        depth_factor = 1.0 / (1.0 + source_depth / 20.0)
        
        wave_height = initial_height * spreading_factor * depth_factor
        
        return max(wave_height, 0.01)  # Minimum 1cm
    
    def _calculate_tsunami_inundation(
        self,
        wave_height: float,
        location: GeologicalLocation,
        bathymetry: Optional[Dict[str, np.ndarray]] = None
    ) -> float:
        """Calculate tsunami inundation distance"""
        # Simplified inundation calculation
        # In practice, would use detailed coastal topography and hydrodynamic models
        
        # Empirical relationship based on wave height and coastal slope
        coastal_slope = 0.02  # Assumed 2% slope
        
        # Inundation distance (simplified)
        inundation_distance = wave_height / coastal_slope * 0.5
        
        return max(inundation_distance, 0.0)
    
    def _calculate_affected_coastline(
        self,
        earthquake: SeismicEvent,
        coastal_locations: List[GeologicalLocation],
        wave_heights: Dict[str, float]
    ) -> float:
        """Calculate length of affected coastline"""
        # Count locations with significant wave heights
        significant_locations = [
            loc_id for loc_id, height in wave_heights.items() if height > 0.5
        ]
        
        # Estimate coastline length (simplified)
        if len(significant_locations) > 1:
            # Rough estimate based on number of affected locations
            coastline_length = len(significant_locations) * 10  # km per location
        else:
            coastline_length = 10  # Minimum 10 km
        
        return coastline_length