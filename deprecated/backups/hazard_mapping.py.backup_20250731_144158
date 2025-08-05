"""
Comprehensive Hazard Mapping System for Prometheus AI Platform

This module implements multi-hazard risk assessment combining volcanic and seismic risks
with probabilistic hazard mapping and uncertainty visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class HazardType(Enum):
    """Types of geological hazards"""
    VOLCANIC_ERUPTION = "volcanic_eruption"
    SEISMIC_EARTHQUAKE = "seismic_earthquake"
    LANDSLIDE = "landslide"
    TSUNAMI = "tsunami"
    PYROCLASTIC_FLOW = "pyroclastic_flow"
    LAHAR = "lahar"
    ASH_FALL = "ash_fall"
    LAVA_FLOW = "lava_flow"
    GROUND_DEFORMATION = "ground_deformation"
    LIQUEFACTION = "liquefaction"


class RiskLevel(Enum):
    """Risk assessment levels"""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5
    EXTREME = 6


@dataclass
class GeographicBounds:
    """Geographic boundaries for hazard mapping"""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    resolution: float = 0.001  # degrees


@dataclass
class HazardProbability:
    """Probabilistic hazard assessment"""
    hazard_type: HazardType
    probability: float  # 0.0 to 1.0
    confidence_interval: Tuple[float, float]
    time_window: timedelta
    uncertainty: float
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class HazardZone:
    """Geographic hazard zone with risk assessment"""
    zone_id: str
    hazard_type: HazardType
    risk_level: RiskLevel
    probability: float
    affected_area: np.ndarray  # 2D array of lat/lon coordinates
    population_at_risk: int
    infrastructure_at_risk: List[str]
    evacuation_routes: List[Tuple[float, float]]
    last_updated: datetime


@dataclass
class MultiHazardAssessment:
    """Combined assessment of multiple hazard types"""
    assessment_id: str
    timestamp: datetime
    geographic_bounds: GeographicBounds
    individual_hazards: Dict[HazardType, HazardProbability]
    combined_risk_map: np.ndarray
    uncertainty_map: np.ndarray
    cascading_effects: Dict[HazardType, List[HazardType]]
    total_population_at_risk: int
    economic_impact_estimate: float


class HazardMappingEngine:
    """Core engine for comprehensive hazard mapping"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.hazard_models = {}
        self.monitoring_data_cache = {}
        self.historical_data = {}
        
    async def initialize_hazard_models(self):
        """Initialize all hazard assessment models"""
        logger.info("Initializing hazard mapping models...")
        
        # Initialize volcanic hazard models
        self.hazard_models[HazardType.VOLCANIC_ERUPTION] = VolcanicHazardModel()
        self.hazard_models[HazardType.PYROCLASTIC_FLOW] = PyroclasticFlowModel()
        self.hazard_models[HazardType.LAHAR] = LaharModel()
        self.hazard_models[HazardType.ASH_FALL] = AshFallModel()
        self.hazard_models[HazardType.LAVA_FLOW] = LavaFlowModel()
        
        # Initialize seismic hazard models
        self.hazard_models[HazardType.SEISMIC_EARTHQUAKE] = SeismicHazardModel()
        self.hazard_models[HazardType.LANDSLIDE] = LandslideModel()
        self.hazard_models[HazardType.TSUNAMI] = TsunamiModel()
        self.hazard_models[HazardType.LIQUEFACTION] = LiquefactionModel()
        self.hazard_models[HazardType.GROUND_DEFORMATION] = DeformationModel()
        
        # Initialize all models
        for model in self.hazard_models.values():
            await model.initialize()
            
        logger.info("Hazard mapping models initialized successfully")
    
    async def assess_multi_hazard_risk(
        self,
        geographic_bounds: GeographicBounds,
        monitoring_data: Dict[str, Any],
        time_horizon: timedelta = timedelta(days=30)
    ) -> MultiHazardAssessment:
        """
        Perform comprehensive multi-hazard risk assessment
        
        Args:
            geographic_bounds: Area to assess
            monitoring_data: Real-time monitoring data
            time_horizon: Time window for assessment
            
        Returns:
            MultiHazardAssessment with combined risk analysis
        """
        logger.info(f"Starting multi-hazard assessment for bounds: {geographic_bounds}")
        
        # Assess individual hazards in parallel
        hazard_tasks = []
        for hazard_type, model in self.hazard_models.items():
            task = asyncio.create_task(
                self._assess_individual_hazard(
                    hazard_type, model, geographic_bounds, monitoring_data, time_horizon
                )
            )
            hazard_tasks.append(task)
        
        individual_assessments = await asyncio.gather(*hazard_tasks)
        individual_hazards = {
            assessment.hazard_type: assessment 
            for assessment in individual_assessments
        }
        
        # Analyze cascading effects
        cascading_effects = self._analyze_cascading_effects(individual_hazards)
        
        # Generate combined risk map
        combined_risk_map, uncertainty_map = self._generate_combined_risk_map(
            individual_hazards, geographic_bounds, cascading_effects
        )
        
        # Calculate population and economic impact
        population_at_risk = self._calculate_population_at_risk(
            combined_risk_map, geographic_bounds
        )
        economic_impact = self._estimate_economic_impact(
            combined_risk_map, geographic_bounds, individual_hazards
        )
        
        assessment = MultiHazardAssessment(
            assessment_id=f"mha_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            geographic_bounds=geographic_bounds,
            individual_hazards=individual_hazards,
            combined_risk_map=combined_risk_map,
            uncertainty_map=uncertainty_map,
            cascading_effects=cascading_effects,
            total_population_at_risk=population_at_risk,
            economic_impact_estimate=economic_impact
        )
        
        logger.info(f"Multi-hazard assessment completed: {assessment.assessment_id}")
        return assessment
    
    async def _assess_individual_hazard(
        self,
        hazard_type: HazardType,
        model: 'BaseHazardModel',
        bounds: GeographicBounds,
        monitoring_data: Dict[str, Any],
        time_horizon: timedelta
    ) -> HazardProbability:
        """Assess individual hazard probability"""
        try:
            return await model.assess_probability(
                bounds, monitoring_data, time_horizon
            )
        except Exception as e:
            logger.error(f"Error assessing {hazard_type}: {e}")
            # Return conservative estimate
            return HazardProbability(
                hazard_type=hazard_type,
                probability=0.1,
                confidence_interval=(0.05, 0.2),
                time_window=time_horizon,
                uncertainty=0.5,
                contributing_factors=["assessment_error"]
            )
    
    def _analyze_cascading_effects(
        self, 
        individual_hazards: Dict[HazardType, HazardProbability]
    ) -> Dict[HazardType, List[HazardType]]:
        """Analyze potential cascading effects between hazards"""
        cascading_effects = {}
        
        # Define cascading relationships
        cascading_rules = {
            HazardType.SEISMIC_EARTHQUAKE: [
                HazardType.LANDSLIDE,
                HazardType.TSUNAMI,
                HazardType.LIQUEFACTION,
                HazardType.VOLCANIC_ERUPTION  # earthquake-triggered eruption
            ],
            HazardType.VOLCANIC_ERUPTION: [
                HazardType.PYROCLASTIC_FLOW,
                HazardType.LAHAR,
                HazardType.ASH_FALL,
                HazardType.LAVA_FLOW
            ],
            HazardType.PYROCLASTIC_FLOW: [
                HazardType.LAHAR  # when interacting with water
            ],
            HazardType.LANDSLIDE: [
                HazardType.TSUNAMI  # if landslide enters water body
            ]
        }
        
        for primary_hazard, secondary_hazards in cascading_rules.items():
            if primary_hazard in individual_hazards:
                primary_prob = individual_hazards[primary_hazard].probability
                if primary_prob > 0.1:  # Only consider significant probabilities
                    cascading_effects[primary_hazard] = []
                    for secondary_hazard in secondary_hazards:
                        # Calculate conditional probability
                        conditional_prob = self._calculate_conditional_probability(
                            primary_hazard, secondary_hazard, primary_prob
                        )
                        if conditional_prob > 0.05:
                            cascading_effects[primary_hazard].append(secondary_hazard)
        
        return cascading_effects
    
    def _calculate_conditional_probability(
        self,
        primary: HazardType,
        secondary: HazardType,
        primary_prob: float
    ) -> float:
        """Calculate conditional probability of secondary hazard given primary"""
        # Simplified conditional probability model
        # In practice, this would use more sophisticated physics-based models
        conditional_factors = {
            (HazardType.SEISMIC_EARTHQUAKE, HazardType.LANDSLIDE): 0.3,
            (HazardType.SEISMIC_EARTHQUAKE, HazardType.TSUNAMI): 0.1,
            (HazardType.SEISMIC_EARTHQUAKE, HazardType.LIQUEFACTION): 0.2,
            (HazardType.SEISMIC_EARTHQUAKE, HazardType.VOLCANIC_ERUPTION): 0.05,
            (HazardType.VOLCANIC_ERUPTION, HazardType.PYROCLASTIC_FLOW): 0.7,
            (HazardType.VOLCANIC_ERUPTION, HazardType.LAHAR): 0.4,
            (HazardType.VOLCANIC_ERUPTION, HazardType.ASH_FALL): 0.9,
            (HazardType.VOLCANIC_ERUPTION, HazardType.LAVA_FLOW): 0.6,
        }
        
        factor = conditional_factors.get((primary, secondary), 0.1)
        return primary_prob * factor
    
    def _generate_combined_risk_map(
        self,
        individual_hazards: Dict[HazardType, HazardProbability],
        bounds: GeographicBounds,
        cascading_effects: Dict[HazardType, List[HazardType]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate combined risk and uncertainty maps"""
        # Create grid based on geographic bounds
        lat_points = int((bounds.max_lat - bounds.min_lat) / bounds.resolution)
        lon_points = int((bounds.max_lon - bounds.min_lon) / bounds.resolution)
        
        combined_risk = np.zeros((lat_points, lon_points))
        uncertainty_map = np.zeros((lat_points, lon_points))
        
        # Combine individual hazard probabilities
        for hazard_type, hazard_prob in individual_hazards.items():
            # Generate spatial distribution for this hazard
            hazard_map = self._generate_hazard_spatial_distribution(
                hazard_type, hazard_prob, bounds, lat_points, lon_points
            )
            
            # Add to combined risk using probabilistic combination
            combined_risk = 1 - (1 - combined_risk) * (1 - hazard_map)
            
            # Add uncertainty
            uncertainty_contribution = np.full_like(hazard_map, hazard_prob.uncertainty)
            uncertainty_map = np.sqrt(uncertainty_map**2 + uncertainty_contribution**2)
        
        # Account for cascading effects
        for primary_hazard, secondary_hazards in cascading_effects.items():
            primary_prob = individual_hazards[primary_hazard].probability
            for secondary_hazard in secondary_hazards:
                conditional_prob = self._calculate_conditional_probability(
                    primary_hazard, secondary_hazard, primary_prob
                )
                # Add cascading risk
                cascading_map = np.full((lat_points, lon_points), conditional_prob)
                combined_risk = 1 - (1 - combined_risk) * (1 - cascading_map)
        
        # Normalize to 0-1 range
        combined_risk = np.clip(combined_risk, 0, 1)
        uncertainty_map = np.clip(uncertainty_map, 0, 1)
        
        return combined_risk, uncertainty_map
    
    def _generate_hazard_spatial_distribution(
        self,
        hazard_type: HazardType,
        hazard_prob: HazardProbability,
        bounds: GeographicBounds,
        lat_points: int,
        lon_points: int
    ) -> np.ndarray:
        """Generate spatial distribution for a specific hazard"""
        # Create base probability map
        base_prob = np.full((lat_points, lon_points), hazard_prob.probability)
        
        # Apply spatial variations based on hazard type
        if hazard_type in [HazardType.VOLCANIC_ERUPTION, HazardType.PYROCLASTIC_FLOW]:
            # Higher probability near volcanic centers
            center_lat = (bounds.min_lat + bounds.max_lat) / 2
            center_lon = (bounds.min_lon + bounds.max_lon) / 2
            
            lat_grid = np.linspace(bounds.min_lat, bounds.max_lat, lat_points)
            lon_grid = np.linspace(bounds.min_lon, bounds.max_lon, lon_points)
            
            for i, lat in enumerate(lat_grid):
                for j, lon in enumerate(lon_grid):
                    distance = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
                    # Exponential decay with distance
                    distance_factor = np.exp(-distance * 10)
                    base_prob[i, j] *= (0.1 + 0.9 * distance_factor)
        
        elif hazard_type == HazardType.SEISMIC_EARTHQUAKE:
            # Higher probability near fault lines (simplified)
            # In practice, this would use actual fault line data
            noise = np.random.random((lat_points, lon_points)) * 0.3
            base_prob *= (0.7 + noise)
        
        return base_prob
    
    def _calculate_population_at_risk(
        self,
        risk_map: np.ndarray,
        bounds: GeographicBounds
    ) -> int:
        """Calculate population at risk based on risk map"""
        # Simplified population density model
        # In practice, this would use actual population data
        avg_population_density = 100  # people per km²
        cell_area = (bounds.resolution * 111) ** 2  # km² per cell (approx)
        
        high_risk_cells = np.sum(risk_map > 0.5)
        moderate_risk_cells = np.sum((risk_map > 0.2) & (risk_map <= 0.5))
        
        population_at_risk = int(
            high_risk_cells * cell_area * avg_population_density +
            moderate_risk_cells * cell_area * avg_population_density * 0.5
        )
        
        return population_at_risk
    
    def _estimate_economic_impact(
        self,
        risk_map: np.ndarray,
        bounds: GeographicBounds,
        individual_hazards: Dict[HazardType, HazardProbability]
    ) -> float:
        """Estimate economic impact of hazards"""
        # Simplified economic impact model
        base_economic_density = 1000000  # USD per km²
        cell_area = (bounds.resolution * 111) ** 2  # km² per cell
        
        total_impact = 0.0
        for hazard_type, hazard_prob in individual_hazards.items():
            # Different hazards have different economic impact factors
            impact_factors = {
                HazardType.VOLCANIC_ERUPTION: 5.0,
                HazardType.SEISMIC_EARTHQUAKE: 3.0,
                HazardType.TSUNAMI: 4.0,
                HazardType.PYROCLASTIC_FLOW: 6.0,
                HazardType.LANDSLIDE: 2.0,
                HazardType.ASH_FALL: 1.5,
                HazardType.LAVA_FLOW: 4.0,
                HazardType.LAHAR: 3.5,
                HazardType.LIQUEFACTION: 2.5,
                HazardType.GROUND_DEFORMATION: 1.0
            }
            
            factor = impact_factors.get(hazard_type, 1.0)
            hazard_impact = (
                np.sum(risk_map) * cell_area * base_economic_density * 
                factor * hazard_prob.probability
            )
            total_impact += hazard_impact
        
        return total_impact


class BaseHazardModel:
    """Base class for individual hazard models"""
    
    async def initialize(self):
        """Initialize the hazard model"""
        pass
    
    async def assess_probability(
        self,
        bounds: GeographicBounds,
        monitoring_data: Dict[str, Any],
        time_horizon: timedelta
    ) -> HazardProbability:
        """Assess probability for this hazard type"""
        raise NotImplementedError


class VolcanicHazardModel(BaseHazardModel):
    """Model for volcanic eruption hazard assessment"""
    
    async def assess_probability(
        self,
        bounds: GeographicBounds,
        monitoring_data: Dict[str, Any],
        time_horizon: timedelta
    ) -> HazardProbability:
        # Analyze volcanic indicators
        seismic_activity = monitoring_data.get('seismic_events', [])
        deformation_data = monitoring_data.get('ground_deformation', {})
        gas_emissions = monitoring_data.get('gas_emissions', {})
        thermal_anomalies = monitoring_data.get('thermal_data', {})
        
        # Calculate base probability from historical data
        base_probability = 0.05  # 5% baseline
        
        # Adjust based on monitoring indicators
        if len(seismic_activity) > 10:  # Increased seismic activity
            base_probability *= 2.0
        
        if deformation_data.get('uplift_rate', 0) > 0.01:  # Significant uplift
            base_probability *= 1.5
        
        if gas_emissions.get('so2_flux', 0) > 1000:  # Elevated SO2
            base_probability *= 1.3
        
        if thermal_anomalies.get('max_temperature', 0) > 50:  # Thermal anomalies
            base_probability *= 1.2
        
        # Cap probability at reasonable maximum
        probability = min(base_probability, 0.8)
        
        # Calculate uncertainty based on data quality
        data_quality_score = self._assess_data_quality(monitoring_data)
        uncertainty = 0.5 - (data_quality_score * 0.3)
        
        # Calculate confidence interval
        margin = uncertainty * probability
        confidence_interval = (
            max(0, probability - margin),
            min(1, probability + margin)
        )
        
        contributing_factors = []
        if len(seismic_activity) > 10:
            contributing_factors.append("increased_seismic_activity")
        if deformation_data.get('uplift_rate', 0) > 0.01:
            contributing_factors.append("ground_deformation")
        if gas_emissions.get('so2_flux', 0) > 1000:
            contributing_factors.append("elevated_gas_emissions")
        if thermal_anomalies.get('max_temperature', 0) > 50:
            contributing_factors.append("thermal_anomalies")
        
        return HazardProbability(
            hazard_type=HazardType.VOLCANIC_ERUPTION,
            probability=probability,
            confidence_interval=confidence_interval,
            time_window=time_horizon,
            uncertainty=uncertainty,
            contributing_factors=contributing_factors
        )
    
    def _assess_data_quality(self, monitoring_data: Dict[str, Any]) -> float:
        """Assess quality of monitoring data (0.0 to 1.0)"""
        quality_score = 0.0
        total_indicators = 4
        
        if monitoring_data.get('seismic_events'):
            quality_score += 0.25
        if monitoring_data.get('ground_deformation'):
            quality_score += 0.25
        if monitoring_data.get('gas_emissions'):
            quality_score += 0.25
        if monitoring_data.get('thermal_data'):
            quality_score += 0.25
        
        return quality_score


class SeismicHazardModel(BaseHazardModel):
    """Model for seismic earthquake hazard assessment"""
    
    async def assess_probability(
        self,
        bounds: GeographicBounds,
        monitoring_data: Dict[str, Any],
        time_horizon: timedelta
    ) -> HazardProbability:
        # Analyze seismic indicators
        recent_earthquakes = monitoring_data.get('recent_earthquakes', [])
        stress_changes = monitoring_data.get('stress_changes', {})
        foreshock_activity = monitoring_data.get('foreshock_activity', [])
        
        # Calculate base probability from seismic hazard maps
        base_probability = 0.1  # 10% baseline for active regions
        
        # Adjust based on recent activity
        if len(recent_earthquakes) > 5:
            base_probability *= 1.5
        
        if len(foreshock_activity) > 3:
            base_probability *= 2.0
        
        if stress_changes.get('coulomb_stress_change', 0) > 0.1:
            base_probability *= 1.3
        
        # Time-dependent factors
        days_in_horizon = time_horizon.days
        if days_in_horizon < 7:  # Short-term elevated risk
            base_probability *= 1.2
        
        probability = min(base_probability, 0.9)
        
        # Calculate uncertainty
        data_completeness = len(recent_earthquakes) / 10.0  # Normalize
        uncertainty = 0.4 - (min(data_completeness, 1.0) * 0.2)
        
        margin = uncertainty * probability
        confidence_interval = (
            max(0, probability - margin),
            min(1, probability + margin)
        )
        
        contributing_factors = []
        if len(recent_earthquakes) > 5:
            contributing_factors.append("recent_seismic_activity")
        if len(foreshock_activity) > 3:
            contributing_factors.append("foreshock_sequence")
        if stress_changes.get('coulomb_stress_change', 0) > 0.1:
            contributing_factors.append("stress_loading")
        
        return HazardProbability(
            hazard_type=HazardType.SEISMIC_EARTHQUAKE,
            probability=probability,
            confidence_interval=confidence_interval,
            time_window=time_horizon,
            uncertainty=uncertainty,
            contributing_factors=contributing_factors
        )


# Additional hazard models (simplified implementations)
class PyroclasticFlowModel(BaseHazardModel):
    async def assess_probability(self, bounds, monitoring_data, time_horizon):
        # Depends heavily on volcanic eruption probability
        volcanic_indicators = monitoring_data.get('volcanic_activity', {})
        base_prob = 0.02
        if volcanic_indicators.get('eruption_probability', 0) > 0.3:
            base_prob = 0.4
        
        return HazardProbability(
            hazard_type=HazardType.PYROCLASTIC_FLOW,
            probability=base_prob,
            confidence_interval=(base_prob * 0.5, base_prob * 1.5),
            time_window=time_horizon,
            uncertainty=0.3,
            contributing_factors=["volcanic_activity"]
        )


class LaharModel(BaseHazardModel):
    async def assess_probability(self, bounds, monitoring_data, time_horizon):
        base_prob = 0.01
        precipitation = monitoring_data.get('precipitation', 0)
        volcanic_activity = monitoring_data.get('volcanic_activity', {})
        
        if precipitation > 50:  # Heavy rainfall
            base_prob *= 3.0
        if volcanic_activity.get('eruption_probability', 0) > 0.2:
            base_prob *= 2.0
        
        return HazardProbability(
            hazard_type=HazardType.LAHAR,
            probability=min(base_prob, 0.6),
            confidence_interval=(base_prob * 0.6, min(base_prob * 1.4, 0.8)),
            time_window=time_horizon,
            uncertainty=0.4,
            contributing_factors=["precipitation", "volcanic_activity"]
        )


class AshFallModel(BaseHazardModel):
    async def assess_probability(self, bounds, monitoring_data, time_horizon):
        volcanic_activity = monitoring_data.get('volcanic_activity', {})
        wind_patterns = monitoring_data.get('wind_data', {})
        
        base_prob = volcanic_activity.get('eruption_probability', 0.05) * 0.8
        
        return HazardProbability(
            hazard_type=HazardType.ASH_FALL,
            probability=base_prob,
            confidence_interval=(base_prob * 0.7, base_prob * 1.3),
            time_window=time_horizon,
            uncertainty=0.2,
            contributing_factors=["volcanic_activity", "wind_patterns"]
        )


class LavaFlowModel(BaseHazardModel):
    async def assess_probability(self, bounds, monitoring_data, time_horizon):
        volcanic_activity = monitoring_data.get('volcanic_activity', {})
        base_prob = volcanic_activity.get('eruption_probability', 0.05) * 0.6
        
        return HazardProbability(
            hazard_type=HazardType.LAVA_FLOW,
            probability=base_prob,
            confidence_interval=(base_prob * 0.8, base_prob * 1.2),
            time_window=time_horizon,
            uncertainty=0.25,
            contributing_factors=["volcanic_activity"]
        )


class LandslideModel(BaseHazardModel):
    async def assess_probability(self, bounds, monitoring_data, time_horizon):
        precipitation = monitoring_data.get('precipitation', 0)
        seismic_activity = len(monitoring_data.get('recent_earthquakes', []))
        slope_stability = monitoring_data.get('slope_stability', 0.8)
        
        base_prob = 0.03
        if precipitation > 100:
            base_prob *= 2.5
        if seismic_activity > 3:
            base_prob *= 2.0
        if slope_stability < 0.5:
            base_prob *= 3.0
        
        return HazardProbability(
            hazard_type=HazardType.LANDSLIDE,
            probability=min(base_prob, 0.7),
            confidence_interval=(base_prob * 0.6, min(base_prob * 1.4, 0.9)),
            time_window=time_horizon,
            uncertainty=0.35,
            contributing_factors=["precipitation", "seismic_activity", "slope_stability"]
        )


class TsunamiModel(BaseHazardModel):
    async def assess_probability(self, bounds, monitoring_data, time_horizon):
        earthquake_prob = monitoring_data.get('earthquake_probability', 0.1)
        underwater_activity = monitoring_data.get('underwater_seismic', [])
        
        base_prob = earthquake_prob * 0.1  # Only some earthquakes generate tsunamis
        if len(underwater_activity) > 2:
            base_prob *= 3.0
        
        return HazardProbability(
            hazard_type=HazardType.TSUNAMI,
            probability=min(base_prob, 0.5),
            confidence_interval=(base_prob * 0.5, min(base_prob * 2.0, 0.8)),
            time_window=time_horizon,
            uncertainty=0.5,
            contributing_factors=["seismic_activity", "underwater_activity"]
        )


class LiquefactionModel(BaseHazardModel):
    async def assess_probability(self, bounds, monitoring_data, time_horizon):
        earthquake_prob = monitoring_data.get('earthquake_probability', 0.1)
        soil_conditions = monitoring_data.get('soil_liquefaction_susceptibility', 0.3)
        groundwater_level = monitoring_data.get('groundwater_level', 0.5)
        
        base_prob = earthquake_prob * soil_conditions * groundwater_level
        
        return HazardProbability(
            hazard_type=HazardType.LIQUEFACTION,
            probability=base_prob,
            confidence_interval=(base_prob * 0.7, base_prob * 1.3),
            time_window=time_horizon,
            uncertainty=0.3,
            contributing_factors=["seismic_activity", "soil_conditions", "groundwater"]
        )


class DeformationModel(BaseHazardModel):
    async def assess_probability(self, bounds, monitoring_data, time_horizon):
        current_deformation = monitoring_data.get('ground_deformation', {})
        deformation_rate = current_deformation.get('rate', 0)
        
        base_prob = min(deformation_rate * 10, 0.8)  # Convert rate to probability
        
        return HazardProbability(
            hazard_type=HazardType.GROUND_DEFORMATION,
            probability=base_prob,
            confidence_interval=(base_prob * 0.8, base_prob * 1.2),
            time_window=time_horizon,
            uncertainty=0.2,
            contributing_factors=["current_deformation_trends"]
        )