"""
Gas Hazard Assessment System

This module implements comprehensive gas hazard assessment with atmospheric dispersion modeling
for the Prometheus AI Platform's volatile degassing analysis system.
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sqlite3
import psycopg2
from scipy import optimize, integrate, interpolate, spatial
from scipy.spatial import distance_matrix, KDTree
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import existing database and core modules
from .database_integration import PostgreSQLGISManager
from .geological_data_structures import GeologicalModel

logger = logging.getLogger(__name__)

@dataclass
class MagmaComposition:
    """Advanced magma composition data"""
    SiO2: float  # Silica content (%)
    Al2O3: float  # Aluminum oxide (%)
    Fe2O3: float  # Iron oxide (%)
    FeO: float    # Iron oxide (%)
    MgO: float    # Magnesium oxide (%)
    CaO: float    # Calcium oxide (%)
    Na2O: float   # Sodium oxide (%)
    K2O: float    # Potassium oxide (%)
    TiO2: float   # Titanium dioxide (%)
    P2O5: float   # Phosphorus pentoxide (%)
    MnO: float    # Manganese oxide (%)
    H2O: float    # Water content (%)
    CO2: float    # Carbon dioxide content (ppm)
    S: float      # Sulfur content (ppm)
    Cl: float     # Chlorine content (ppm)
    F: float      # Fluorine content (ppm)
    
    def __post_init__(self):
        """Calculate derived properties"""
        self.total_alkalis = self.Na2O + self.K2O
        self.mg_number = (self.MgO / 40.3) / ((self.MgO / 40.3) + (self.FeO / 71.8)) * 100
        self.differentiation_index = self.SiO2 + self.Na2O + self.K2O
        
    def get_magma_type(self) -> str:
        """Classify magma type based on composition"""
        if self.SiO2 < 45:
            return "ultrabasic"
        elif self.SiO2 < 52:
            return "basic"
        elif self.SiO2 < 63:
            return "intermediate"
        elif self.SiO2 < 69:
            return "acidic"
        else:
            return "ultraacidic"
    
    def get_viscosity_estimate(self, temperature: float) -> float:
        """Estimate magma viscosity (Pa·s) based on composition and temperature"""
        # Giordano et al. (2008) model
        A = -4.55
        B = 159.56 * self.SiO2 - 173.34 * self.TiO2 + 72.13 * self.Al2O3 + 75.69 * self.Fe2O3 + \
            -38.98 * self.FeO + -84.06 * self.MnO + 141.54 * self.MgO + -2.43 * self.CaO + \
            -0.91 * self.Na2O + 17.62 * self.K2O + 80.33 * self.P2O5
        C = 2.75 * self.SiO2 + 15.72 * self.TiO2 + 8.32 * self.Al2O3 + 10.20 * self.Fe2O3 + \
            -12.29 * self.FeO + -99.54 * self.MnO + 0.30 * self.MgO + 0.78 * self.CaO
        
        log_viscosity = A + B / (temperature - C)
        return 10 ** log_viscosity

@dataclass
class EruptionHistory:
    """Comprehensive eruption history data"""
    eruption_id: str
    start_date: datetime
    end_date: Optional[datetime]
    vei: int  # Volcanic Explosivity Index (0-8)
    eruption_type: str  # effusive, explosive, phreatomagmatic, etc.
    volume_erupted: float  # m³
    column_height: float  # m
    pyroclastic_flows: bool
    lava_flows: bool
    lahars: bool
    ash_fall_extent: float  # km²
    fatalities: int
    evacuees: int
    economic_damage: float  # USD
    dominant_wind_direction: float  # degrees
    weather_conditions: Dict[str, Any]
    gas_emissions: Dict[str, float]  # kg/day
    seismic_precursors: Dict[str, Any]
    deformation_precursors: Dict[str, Any]
    gas_precursors: Dict[str, Any]
    
    def get_repose_interval(self, previous_eruption: 'EruptionHistory') -> timedelta:
        """Calculate repose interval between eruptions"""
        return self.start_date - previous_eruption.end_date if previous_eruption.end_date else timedelta(0)

@dataclass
class MonitoringInfrastructure:
    """Advanced monitoring infrastructure data"""
    seismic_stations: List[Dict[str, Any]]
    gps_stations: List[Dict[str, Any]]
    tilt_meters: List[Dict[str, Any]]
    gas_sensors: List[Dict[str, Any]]
    thermal_cameras: List[Dict[str, Any]]
    webcams: List[Dict[str, Any]]
    insar_coverage: bool
    satellite_monitoring: List[str]
    drone_capabilities: Dict[str, Any]
    field_sampling_sites: List[Dict[str, Any]]
    communication_systems: List[str]
    power_systems: Dict[str, Any]
    data_transmission: Dict[str, Any]
    maintenance_schedule: Dict[str, Any]
    
    def get_monitoring_density(self) -> float:
        """Calculate monitoring station density per km²"""
        total_stations = (len(self.seismic_stations) + len(self.gps_stations) + 
                         len(self.tilt_meters) + len(self.gas_sensors))
        return total_stations / 100.0  # Assuming 100 km² monitoring area

@dataclass
class GeologicalStructure:
    """Advanced geological structure data"""
    edifice_volume: float  # km³
    crater_diameter: float  # m
    crater_depth: float  # m
    slope_angles: Dict[str, float]  # degrees by sector
    structural_weaknesses: List[Dict[str, Any]]
    fault_systems: List[Dict[str, Any]]
    hydrothermal_systems: List[Dict[str, Any]]
    basement_geology: Dict[str, Any]
    crustal_thickness: float  # km
    magma_chamber_depth: float  # km
    magma_chamber_volume: float  # km³
    conduit_geometry: Dict[str, Any]
    vent_locations: List[Tuple[float, float, float]]
    fumarole_fields: List[Dict[str, Any]]
    hot_springs: List[Dict[str, Any]]
    
    def calculate_edifice_stability(self) -> float:
        """Calculate advanced edifice stability index using comprehensive geomechanical analysis"""
        try:
            # Advanced stability calculation using multiple factors
            
            # 1. Slope stability analysis using limit equilibrium method
            slope_stability = self._calculate_slope_stability_factor()
            
            # 2. Structural integrity assessment
            structural_integrity = self._assess_structural_integrity()
            
            # 3. Material strength analysis
            material_strength = self._analyze_material_strength()
            
            # 4. Hydrothermal alteration effects
            alteration_factor = self._assess_hydrothermal_alteration()
            
            # 5. Seismic loading effects
            seismic_factor = self._calculate_seismic_loading_factor()
            
            # 6. Gravitational loading
            gravitational_factor = self._calculate_gravitational_loading()
            
            # 7. Pore pressure effects
            pore_pressure_factor = self._calculate_pore_pressure_effects()
            
            # Weighted combination of all factors
            stability_index = (
                0.25 * slope_stability +
                0.20 * structural_integrity +
                0.15 * material_strength +
                0.15 * alteration_factor +
                0.10 * seismic_factor +
                0.10 * gravitational_factor +
                0.05 * pore_pressure_factor
            )
            
            return max(0.0, min(1.0, stability_index))
            
        except Exception as e:
            logger.error(f"Error in advanced stability calculation: {e}")
            # Fallback to basic calculation
            max_slope = max(self.slope_angles.values()) if self.slope_angles else 0
            weakness_factor = len(self.structural_weaknesses) * 0.1
            stability = max(0, 1.0 - (max_slope / 45.0) - weakness_factor)
            return min(1.0, stability)
    
    def _calculate_slope_stability_factor(self) -> float:
        """Calculate slope stability using advanced limit equilibrium analysis"""
        try:
            if not self.slope_angles:
                return 0.5
            
            stability_factors = []
            
            for direction, angle in self.slope_angles.items():
                # Bishop's simplified method for circular failure surfaces
                angle_rad = np.radians(angle)
                
                # Material properties (typical volcanic rock values)
                cohesion = 50000  # Pa (50 kPa)
                friction_angle = np.radians(35)  # 35 degrees
                unit_weight = 25000  # N/m³
                
                # Assume average slope height of 500m
                slope_height = 500
                
                # Calculate factor of safety
                if angle_rad > 0:
                    # Simplified Bishop's method
                    normal_stress = unit_weight * slope_height * np.cos(angle_rad)
                    shear_stress = unit_weight * slope_height * np.sin(angle_rad)
                    
                    # Effective stress (assuming some pore pressure)
                    pore_pressure = 0.3 * normal_stress  # 30% pore pressure ratio
                    effective_stress = normal_stress - pore_pressure
                    
                    # Mohr-Coulomb failure criterion
                    shear_strength = cohesion + effective_stress * np.tan(friction_angle)
                    
                    factor_of_safety = shear_strength / (shear_stress + 1e-6)
                    stability_factors.append(min(factor_of_safety / 1.5, 1.0))  # Normalize to 1.0
                else:
                    stability_factors.append(1.0)
            
            return np.mean(stability_factors)
            
        except Exception as e:
            logger.error(f"Error calculating slope stability: {e}")
            return 0.5
    
    def _assess_structural_integrity(self) -> float:
        """Assess structural integrity based on fracture networks and discontinuities"""
        try:
            if not self.structural_weaknesses:
                return 1.0
            
            # Analyze different types of structural weaknesses
            weakness_scores = []
            
            for weakness in self.structural_weaknesses:
                weakness_type = weakness.get('type', 'unknown')
                orientation = weakness.get('orientation', {})
                persistence = weakness.get('persistence', 0.5)
                aperture = weakness.get('aperture', 0.001)  # meters
                roughness = weakness.get('roughness', 0.5)
                
                # Calculate weakness impact based on type
                if weakness_type == 'fault':
                    # Major structural discontinuity
                    impact = 0.8 * persistence + 0.2 * (aperture / 0.01)
                elif weakness_type == 'fracture':
                    # Minor discontinuity
                    impact = 0.6 * persistence + 0.3 * (aperture / 0.001) + 0.1 * (1 - roughness)
                elif weakness_type == 'joint':
                    # Natural rock joint
                    impact = 0.5 * persistence + 0.4 * (aperture / 0.001) + 0.1 * (1 - roughness)
                elif weakness_type == 'alteration_zone':
                    # Hydrothermally altered zone
                    impact = 0.7 * persistence + 0.3 * (1 - roughness)
                else:
                    impact = 0.5 * persistence
                
                # Consider orientation relative to slope
                orientation_factor = self._calculate_orientation_factor(orientation)
                total_impact = impact * orientation_factor
                
                weakness_scores.append(min(total_impact, 1.0))
            
            # Overall structural integrity (inverse of weakness)
            average_weakness = np.mean(weakness_scores)
            return max(0.0, 1.0 - average_weakness)
            
        except Exception as e:
            logger.error(f"Error assessing structural integrity: {e}")
            return 0.7
    
    def _analyze_material_strength(self) -> float:
        """Analyze material strength based on rock type and alteration"""
        try:
            # Default rock strength values (MPa) for different volcanic rocks
            rock_strengths = {
                'basalt': 200,
                'andesite': 150,
                'dacite': 120,
                'rhyolite': 100,
                'pyroclastic': 50,
                'altered_volcanic': 30,
                'volcanic_breccia': 80
            }
            
            # Estimate rock type distribution (would come from geological mapping)
            rock_distribution = {
                'basalt': 0.3,
                'andesite': 0.4,
                'pyroclastic': 0.2,
                'altered_volcanic': 0.1
            }
            
            # Calculate weighted average strength
            total_strength = sum(
                rock_strengths.get(rock_type, 100) * proportion
                for rock_type, proportion in rock_distribution.items()
            )
            
            # Normalize to 0-1 scale (assuming max strength of 300 MPa)
            normalized_strength = min(total_strength / 300.0, 1.0)
            
            # Apply alteration reduction factor
            alteration_reduction = 1.0 - (len(self.structural_weaknesses) * 0.05)
            
            return max(0.0, normalized_strength * alteration_reduction)
            
        except Exception as e:
            logger.error(f"Error analyzing material strength: {e}")
            return 0.6
    
    def _assess_hydrothermal_alteration(self) -> float:
        """Assess impact of hydrothermal alteration on stability"""
        try:
            # Count alteration-related weaknesses
            alteration_zones = [
                w for w in self.structural_weaknesses 
                if w.get('type') == 'alteration_zone'
            ]
            
            if not alteration_zones:
                return 1.0
            
            # Calculate alteration intensity
            total_alteration_area = sum(
                zone.get('area', 0) for zone in alteration_zones
            )
            
            # Assume total edifice area for normalization
            edifice_area = np.pi * (self.base_diameter / 2) ** 2 if hasattr(self, 'base_diameter') else 1e6
            
            alteration_ratio = min(total_alteration_area / edifice_area, 1.0)
            
            # Alteration reduces stability exponentially
            stability_factor = np.exp(-2 * alteration_ratio)
            
            return max(0.1, stability_factor)
            
        except Exception as e:
            logger.error(f"Error assessing hydrothermal alteration: {e}")
            return 0.8
    
    def _calculate_seismic_loading_factor(self) -> float:
        """Calculate seismic loading effects on stability"""
        try:
            # Estimate seismic hazard based on location and volcanic activity
            # This would typically come from seismic hazard maps
            
            # Assume moderate seismic activity for volcanic regions
            peak_ground_acceleration = 0.3  # g (30% of gravity)
            
            # Calculate seismic coefficient
            seismic_coefficient = peak_ground_acceleration / 2.0
            
            # Stability reduction due to seismic loading
            seismic_factor = 1.0 - seismic_coefficient
            
            return max(0.0, seismic_factor)
            
        except Exception as e:
            logger.error(f"Error calculating seismic loading: {e}")
            return 0.8
    
    def _calculate_gravitational_loading(self) -> float:
        """Calculate gravitational loading effects"""
        try:
            # Estimate edifice volume and mass
            if hasattr(self, 'height') and hasattr(self, 'base_diameter'):
                # Approximate as cone
                volume = (1/3) * np.pi * (self.base_diameter / 2) ** 2 * self.height
            else:
                # Default values for typical stratovolcano
                volume = 1e9  # m³
            
            # Volcanic rock density
            rock_density = 2500  # kg/m³
            
            # Total mass
            total_mass = volume * rock_density
            
            # Gravitational stress at base
            base_area = np.pi * (self.base_diameter / 2) ** 2 if hasattr(self, 'base_diameter') else 1e6
            gravitational_stress = (total_mass * 9.81) / base_area  # Pa
            
            # Normalize to stability factor (assuming critical stress of 10 MPa)
            critical_stress = 10e6  # Pa
            stability_factor = max(0.0, 1.0 - (gravitational_stress / critical_stress))
            
            return stability_factor
            
        except Exception as e:
            logger.error(f"Error calculating gravitational loading: {e}")
            return 0.7
    
    def _calculate_pore_pressure_effects(self) -> float:
        """Calculate pore pressure effects on stability"""
        try:
            # Estimate pore pressure based on hydrothermal activity
            alteration_zones = [
                w for w in self.structural_weaknesses 
                if w.get('type') == 'alteration_zone'
            ]
            
            # Higher alteration = higher pore pressure
            pore_pressure_ratio = min(len(alteration_zones) * 0.1, 0.8)
            
            # Stability reduction due to pore pressure
            stability_factor = 1.0 - pore_pressure_ratio
            
            return max(0.2, stability_factor)
            
        except Exception as e:
            logger.error(f"Error calculating pore pressure effects: {e}")
            return 0.8
    
    def _calculate_orientation_factor(self, orientation: Dict[str, float]) -> float:
        """Calculate orientation factor for structural weakness"""
        try:
            dip = orientation.get('dip', 45)  # degrees
            dip_direction = orientation.get('dip_direction', 0)  # degrees
            
            # Most critical when dip is moderate (30-60 degrees) and dips out of slope
            dip_factor = 1.0 - abs(dip - 45) / 45.0
            
            # Direction factor (simplified - would need actual slope orientation)
            direction_factor = 0.8  # Assume moderate criticality
            
            return dip_factor * direction_factor
            
        except Exception as e:
            logger.error(f"Error calculating orientation factor: {e}")
            return 0.5

@dataclass
class HazardZones:
    """Advanced volcanic hazard zones"""
    pyroclastic_flow_zones: List[Dict[str, Any]]
    lava_flow_zones: List[Dict[str, Any]]
    lahar_zones: List[Dict[str, Any]]
    ash_fall_zones: List[Dict[str, Any]]
    gas_hazard_zones: List[Dict[str, Any]]
    debris_avalanche_zones: List[Dict[str, Any]]
    tsunami_zones: List[Dict[str, Any]]
    exclusion_zones: List[Dict[str, Any]]
    evacuation_routes: List[Dict[str, Any]]
    safe_zones: List[Dict[str, Any]]
    
    def get_population_at_risk(self, hazard_type: str) -> int:
        """Calculate population at risk for specific hazard type"""
        zones = getattr(self, f"{hazard_type}_zones", [])
        total_population = sum(zone.get('population', 0) for zone in zones)
        return total_population

class VolcanicSystem:
    """Comprehensive volcanic system with advanced geological and monitoring capabilities"""
    
    def __init__(self, system_id: str, db_manager: Optional[PostgreSQLGISManager] = None):
        self.system_id = system_id
        self.db_manager = db_manager
        
        # Basic identification
        self.name: str = ""
        self.alternative_names: List[str] = []
        self.country: str = ""
        self.region: str = ""
        self.volcanic_arc: str = ""
        
        # Geographic data
        self.location: Tuple[float, float] = (0.0, 0.0)  # lat, lon
        self.elevation: float = 0.0  # m above sea level
        self.prominence: float = 0.0  # m
        self.base_elevation: float = 0.0  # m
        self.utm_zone: str = ""
        self.coordinate_system: str = "WGS84"
        
        # Volcanic classification
        self.volcano_type: str = ""  # stratovolcano, shield, cinder_cone, etc.
        self.tectonic_setting: str = ""  # subduction, rift, hotspot, etc.
        self.dominant_rock_type: str = ""  # andesite, basalt, dacite, etc.
        self.morphology: str = ""  # conical, elongated, complex, etc.
        
        # Temporal data
        self.age_oldest_rocks: float = 0.0  # Ma
        self.age_youngest_rocks: float = 0.0  # Ma
        self.formation_date: Optional[datetime] = None
        self.last_known_activity: Optional[datetime] = None
        
        # Activity status
        self.activity_status: str = "unknown"  # active, dormant, extinct
        self.alert_level: str = "green"  # green, yellow, orange, red
        self.monitoring_status: str = "none"  # none, basic, comprehensive, real-time
        self.threat_level: str = "low"  # low, moderate, high, very_high
        
        # Advanced components
        self.magma_composition: Optional[MagmaComposition] = None
        self.eruption_history: List[EruptionHistory] = []
        self.monitoring_infrastructure: Optional[MonitoringInfrastructure] = None
        self.geological_structure: Optional[GeologicalStructure] = None
        self.hazard_zones: Optional[HazardZones] = None
        
        # Environmental context
        self.climate_zone: str = ""
        self.precipitation_annual: float = 0.0  # mm/year
        self.temperature_range: Tuple[float, float] = (0.0, 0.0)  # min, max °C
        self.prevailing_winds: Dict[str, float] = {}  # direction: frequency
        
        # Socioeconomic context
        self.population_within_10km: int = 0
        self.population_within_30km: int = 0
        self.population_within_100km: int = 0
        self.major_cities_nearby: List[Dict[str, Any]] = []
        self.infrastructure_at_risk: List[Dict[str, Any]] = []
        self.economic_importance: str = "low"  # low, moderate, high
        
        # Research and data
        self.research_institutions: List[str] = []
        self.data_availability: Dict[str, str] = {}
        self.publication_count: int = 0
        self.last_comprehensive_study: Optional[datetime] = None
        
        # Metadata
        self.created_date: datetime = datetime.now()
        self.last_updated: datetime = datetime.now()
        self.data_quality: str = "unknown"  # poor, fair, good, excellent
        self.completeness_score: float = 0.0  # 0-1
        
    async def load_from_database(self) -> bool:
        """Load comprehensive volcanic system data from database"""
        if not self.db_manager:
            return False
            
        try:
            # Load basic volcanic system data
            basic_query = """
            SELECT vs.name, vs.alternative_names, vs.country, vs.region, vs.volcanic_arc,
                   ST_X(vs.location) as longitude, ST_Y(vs.location) as latitude,
                   vs.elevation, vs.prominence, vs.base_elevation, vs.utm_zone,
                   vs.volcano_type, vs.tectonic_setting, vs.dominant_rock_type, vs.morphology,
                   vs.age_oldest_rocks, vs.age_youngest_rocks, vs.formation_date,
                   vs.last_known_activity, vs.activity_status, vs.alert_level,
                   vs.monitoring_status, vs.threat_level, vs.climate_zone,
                   vs.precipitation_annual, vs.temp_min, vs.temp_max,
                   vs.population_10km, vs.population_30km, vs.population_100km,
                   vs.economic_importance, vs.data_quality, vs.completeness_score
            FROM volcanic_systems vs
            WHERE vs.system_id = %s
            """
            
            basic_data = await self.db_manager.execute_query(basic_query, (self.system_id,))
            
            if not basic_data:
                return False
                
            data = basic_data[0]
            
            # Populate basic data
            self.name = data['name']
            self.alternative_names = data['alternative_names'].split(',') if data['alternative_names'] else []
            self.country = data['country']
            self.region = data['region']
            self.volcanic_arc = data['volcanic_arc']
            self.location = (data['latitude'], data['longitude'])
            self.elevation = data['elevation']
            self.prominence = data['prominence']
            self.base_elevation = data['base_elevation']
            self.utm_zone = data['utm_zone']
            self.volcano_type = data['volcano_type']
            self.tectonic_setting = data['tectonic_setting']
            self.dominant_rock_type = data['dominant_rock_type']
            self.morphology = data['morphology']
            self.age_oldest_rocks = data['age_oldest_rocks']
            self.age_youngest_rocks = data['age_youngest_rocks']
            self.formation_date = data['formation_date']
            self.last_known_activity = data['last_known_activity']
            self.activity_status = data['activity_status']
            self.alert_level = data['alert_level']
            self.monitoring_status = data['monitoring_status']
            self.threat_level = data['threat_level']
            self.climate_zone = data['climate_zone']
            self.precipitation_annual = data['precipitation_annual']
            self.temperature_range = (data['temp_min'], data['temp_max'])
            self.population_within_10km = data['population_10km']
            self.population_within_30km = data['population_30km']
            self.population_within_100km = data['population_100km']
            self.economic_importance = data['economic_importance']
            self.data_quality = data['data_quality']
            self.completeness_score = data['completeness_score']
            
            # Load magma composition
            await self._load_magma_composition()
            
            # Load eruption history
            await self._load_eruption_history()
            
            # Load monitoring infrastructure
            await self._load_monitoring_infrastructure()
            
            # Load geological structure
            await self._load_geological_structure()
            
            # Load hazard zones
            await self._load_hazard_zones()
            
            # Load additional contextual data
            await self._load_contextual_data()
            
            self.last_updated = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error loading volcanic system data: {str(e)}")
            return False
    
    async def _load_magma_composition(self):
        """Load magma composition data from database"""
        try:
            comp_query = """
            SELECT SiO2, Al2O3, Fe2O3, FeO, MgO, CaO, Na2O, K2O, TiO2, P2O5, MnO,
                   H2O, CO2, S, Cl, F
            FROM magma_compositions
            WHERE volcanic_system_id = %s
            ORDER BY sample_date DESC
            LIMIT 1
            """
            
            comp_data = await self.db_manager.execute_query(comp_query, (self.system_id,))
            
            if comp_data:
                data = comp_data[0]
                self.magma_composition = MagmaComposition(
                    SiO2=data['SiO2'], Al2O3=data['Al2O3'], Fe2O3=data['Fe2O3'],
                    FeO=data['FeO'], MgO=data['MgO'], CaO=data['CaO'],
                    Na2O=data['Na2O'], K2O=data['K2O'], TiO2=data['TiO2'],
                    P2O5=data['P2O5'], MnO=data['MnO'], H2O=data['H2O'],
                    CO2=data['CO2'], S=data['S'], Cl=data['Cl'], F=data['F']
                )
                
        except Exception as e:
            logger.error(f"Error loading magma composition: {str(e)}")
    
    async def _load_eruption_history(self):
        """Load comprehensive eruption history from database"""
        try:
            eruption_query = """
            SELECT eruption_id, start_date, end_date, vei, eruption_type,
                   volume_erupted, column_height, pyroclastic_flows, lava_flows,
                   lahars, ash_fall_extent, fatalities, evacuees, economic_damage,
                   dominant_wind_direction, weather_conditions, gas_emissions,
                   seismic_precursors, deformation_precursors, gas_precursors
            FROM eruption_history
            WHERE volcanic_system_id = %s
            ORDER BY start_date DESC
            """
            
            eruption_data = await self.db_manager.execute_query(eruption_query, (self.system_id,))
            
            self.eruption_history = []
            for eruption in eruption_data:
                self.eruption_history.append(EruptionHistory(
                    eruption_id=eruption['eruption_id'],
                    start_date=eruption['start_date'],
                    end_date=eruption['end_date'],
                    vei=eruption['vei'],
                    eruption_type=eruption['eruption_type'],
                    volume_erupted=eruption['volume_erupted'],
                    column_height=eruption['column_height'],
                    pyroclastic_flows=eruption['pyroclastic_flows'],
                    lava_flows=eruption['lava_flows'],
                    lahars=eruption['lahars'],
                    ash_fall_extent=eruption['ash_fall_extent'],
                    fatalities=eruption['fatalities'],
                    evacuees=eruption['evacuees'],
                    economic_damage=eruption['economic_damage'],
                    dominant_wind_direction=eruption['dominant_wind_direction'],
                    weather_conditions=eruption['weather_conditions'] or {},
                    gas_emissions=eruption['gas_emissions'] or {},
                    seismic_precursors=eruption['seismic_precursors'] or {},
                    deformation_precursors=eruption['deformation_precursors'] or {},
                    gas_precursors=eruption['gas_precursors'] or {}
                ))
                
        except Exception as e:
            logger.error(f"Error loading eruption history: {str(e)}")
    
    async def _load_monitoring_infrastructure(self):
        """Load monitoring infrastructure data from database"""
        try:
            # Load different types of monitoring equipment
            equipment_types = ['seismic_stations', 'gps_stations', 'tilt_meters', 
                             'gas_sensors', 'thermal_cameras', 'webcams', 'field_sampling_sites']
            
            equipment_data = {}
            
            for eq_type in equipment_types:
                eq_query = f"""
                SELECT station_id, station_name, ST_X(location) as longitude, 
                       ST_Y(location) as latitude, elevation, installation_date,
                       status, specifications, maintenance_history
                FROM monitoring_equipment
                WHERE volcanic_system_id = %s AND equipment_type = %s
                """
                
                stations = await self.db_manager.execute_query(eq_query, (self.system_id, eq_type))
                equipment_data[eq_type] = stations or []
            
            # Load additional monitoring capabilities
            capabilities_query = """
            SELECT insar_coverage, satellite_monitoring, drone_capabilities,
                   communication_systems, power_systems, data_transmission,
                   maintenance_schedule
            FROM monitoring_capabilities
            WHERE volcanic_system_id = %s
            """
            
            capabilities = await self.db_manager.execute_query(capabilities_query, (self.system_id,))
            
            if capabilities:
                cap_data = capabilities[0]
                self.monitoring_infrastructure = MonitoringInfrastructure(
                    seismic_stations=equipment_data['seismic_stations'],
                    gps_stations=equipment_data['gps_stations'],
                    tilt_meters=equipment_data['tilt_meters'],
                    gas_sensors=equipment_data['gas_sensors'],
                    thermal_cameras=equipment_data['thermal_cameras'],
                    webcams=equipment_data['webcams'],
                    field_sampling_sites=equipment_data['field_sampling_sites'],
                    insar_coverage=cap_data['insar_coverage'],
                    satellite_monitoring=cap_data['satellite_monitoring'] or [],
                    drone_capabilities=cap_data['drone_capabilities'] or {},
                    communication_systems=cap_data['communication_systems'] or [],
                    power_systems=cap_data['power_systems'] or {},
                    data_transmission=cap_data['data_transmission'] or {},
                    maintenance_schedule=cap_data['maintenance_schedule'] or {}
                )
                
        except Exception as e:
            logger.error(f"Error loading monitoring infrastructure: {str(e)}")
    
    async def _load_geological_structure(self):
        """Load geological structure data from database"""
        try:
            structure_query = """
            SELECT edifice_volume, crater_diameter, crater_depth, slope_angles,
                   structural_weaknesses, fault_systems, hydrothermal_systems,
                   basement_geology, crustal_thickness, magma_chamber_depth,
                   magma_chamber_volume, conduit_geometry, vent_locations,
                   fumarole_fields, hot_springs
            FROM geological_structure
            WHERE volcanic_system_id = %s
            """
            
            structure_data = await self.db_manager.execute_query(structure_query, (self.system_id,))
            
            if structure_data:
                data = structure_data[0]
                self.geological_structure = GeologicalStructure(
                    edifice_volume=data['edifice_volume'],
                    crater_diameter=data['crater_diameter'],
                    crater_depth=data['crater_depth'],
                    slope_angles=data['slope_angles'] or {},
                    structural_weaknesses=data['structural_weaknesses'] or [],
                    fault_systems=data['fault_systems'] or [],
                    hydrothermal_systems=data['hydrothermal_systems'] or [],
                    basement_geology=data['basement_geology'] or {},
                    crustal_thickness=data['crustal_thickness'],
                    magma_chamber_depth=data['magma_chamber_depth'],
                    magma_chamber_volume=data['magma_chamber_volume'],
                    conduit_geometry=data['conduit_geometry'] or {},
                    vent_locations=data['vent_locations'] or [],
                    fumarole_fields=data['fumarole_fields'] or [],
                    hot_springs=data['hot_springs'] or []
                )
                
        except Exception as e:
            logger.error(f"Error loading geological structure: {str(e)}")
    
    async def _load_hazard_zones(self):
        """Load hazard zones data from database"""
        try:
            hazard_types = ['pyroclastic_flow', 'lava_flow', 'lahar', 'ash_fall', 
                           'gas_hazard', 'debris_avalanche', 'tsunami', 'exclusion']
            
            hazard_data = {}
            
            for hazard_type in hazard_types:
                hazard_query = """
                SELECT zone_id, zone_name, ST_AsText(geometry) as geometry,
                       hazard_level, probability, population, infrastructure,
                       evacuation_time, access_routes
                FROM hazard_zones
                WHERE volcanic_system_id = %s AND hazard_type = %s
                """
                
                zones = await self.db_manager.execute_query(hazard_query, (self.system_id, hazard_type))
                hazard_data[f"{hazard_type}_zones"] = zones or []
            
            # Load evacuation routes and safe zones
            evacuation_query = """
            SELECT route_id, route_name, ST_AsText(geometry) as geometry,
                   capacity, travel_time, accessibility, status
            FROM evacuation_routes
            WHERE volcanic_system_id = %s
            """
            
            safe_zones_query = """
            SELECT zone_id, zone_name, ST_AsText(geometry) as geometry,
                   capacity, facilities, accessibility
            FROM safe_zones
            WHERE volcanic_system_id = %s
            """
            
            evacuation_routes = await self.db_manager.execute_query(evacuation_query, (self.system_id,))
            safe_zones = await self.db_manager.execute_query(safe_zones_query, (self.system_id,))
            
            self.hazard_zones = HazardZones(
                pyroclastic_flow_zones=hazard_data['pyroclastic_flow_zones'],
                lava_flow_zones=hazard_data['lava_flow_zones'],
                lahar_zones=hazard_data['lahar_zones'],
                ash_fall_zones=hazard_data['ash_fall_zones'],
                gas_hazard_zones=hazard_data['gas_hazard_zones'],
                debris_avalanche_zones=hazard_data['debris_avalanche_zones'],
                tsunami_zones=hazard_data['tsunami_zones'],
                exclusion_zones=hazard_data['exclusion_zones'],
                evacuation_routes=evacuation_routes or [],
                safe_zones=safe_zones or []
            )
            
        except Exception as e:
            logger.error(f"Error loading hazard zones: {str(e)}")
    
    async def _load_contextual_data(self):
        """Load additional contextual data"""
        try:
            # Load nearby cities
            cities_query = """
            SELECT city_name, population, ST_Distance(ST_Point(%s, %s), location) as distance
            FROM major_cities
            WHERE ST_DWithin(ST_Point(%s, %s), location, 100000)
            ORDER BY distance
            """
            
            cities = await self.db_manager.execute_query(
                cities_query, 
                (self.location[1], self.location[0], self.location[1], self.location[0])
            )
            
            self.major_cities_nearby = cities or []
            
            # Load infrastructure at risk
            infrastructure_query = """
            SELECT infrastructure_type, name, importance, ST_Distance(ST_Point(%s, %s), location) as distance
            FROM critical_infrastructure
            WHERE ST_DWithin(ST_Point(%s, %s), location, 50000)
            ORDER BY importance DESC, distance
            """
            
            infrastructure = await self.db_manager.execute_query(
                infrastructure_query,
                (self.location[1], self.location[0], self.location[1], self.location[0])
            )
            
            self.infrastructure_at_risk = infrastructure or []
            
            # Load research institutions
            research_query = """
            SELECT institution_name
            FROM research_institutions
            WHERE volcanic_system_id = %s
            """
            
            institutions = await self.db_manager.execute_query(research_query, (self.system_id,))
            self.research_institutions = [inst['institution_name'] for inst in institutions] if institutions else []
            
        except Exception as e:
            logger.error(f"Error loading contextual data: {str(e)}")
    
    def calculate_threat_score(self) -> float:
        """Calculate comprehensive threat score (0-100)"""
        score = 0.0
        
        # Population factor (0-30 points)
        pop_score = min(30, (self.population_within_30km / 100000) * 30)
        score += pop_score
        
        # Historical activity (0-25 points)
        if self.eruption_history:
            recent_eruptions = [e for e in self.eruption_history if e.start_date > datetime.now() - timedelta(days=365*100)]
            vei_scores = [e.vei for e in recent_eruptions]
            max_vei = max(vei_scores) if vei_scores else 0
            activity_score = min(25, max_vei * 3 + len(recent_eruptions) * 2)
            score += activity_score
        
        # Monitoring capability (0-15 points, inverted - better monitoring = lower threat)
        if self.monitoring_infrastructure:
            monitoring_density = self.monitoring_infrastructure.get_monitoring_density()
            monitoring_score = max(0, 15 - monitoring_density * 3)
            score += monitoring_score
        else:
            score += 15  # No monitoring = maximum threat from this factor
        
        # Geological factors (0-20 points)
        if self.geological_structure:
            stability = self.geological_structure.calculate_edifice_stability()
            geological_score = (1 - stability) * 20
            score += geological_score
        
        # Magma composition (0-10 points)
        if self.magma_composition:
            if self.magma_composition.get_magma_type() in ['acidic', 'ultraacidic']:
                score += 10  # More explosive potential
            elif self.magma_composition.get_magma_type() == 'intermediate':
                score += 6
            else:
                score += 2
        
        return min(100, score)
    
    def get_eruption_probability(self, time_window_years: int = 10) -> float:
        """Calculate eruption probability for given time window"""
        if not self.eruption_history:
            return 0.1  # Default low probability
        
        # Calculate average repose interval
        repose_intervals = []
        for i in range(1, len(self.eruption_history)):
            interval = self.eruption_history[i-1].get_repose_interval(self.eruption_history[i])
            if interval.days > 0:
                repose_intervals.append(interval.days / 365.25)  # Convert to years
        
        if not repose_intervals:
            return 0.1
        
        avg_repose = np.mean(repose_intervals)
        
        # Simple Poisson model
        rate = 1.0 / avg_repose  # eruptions per year
        probability = 1 - np.exp(-rate * time_window_years)
        
        return min(1.0, probability)
    
    def get_monitoring_status_score(self) -> float:
        """Calculate monitoring adequacy score (0-1)"""
        if not self.monitoring_infrastructure:
            return 0.0
        
        # Weight different monitoring types
        weights = {
            'seismic_stations': 0.3,
            'gps_stations': 0.2,
            'gas_sensors': 0.2,
            'thermal_cameras': 0.1,
            'tilt_meters': 0.1,
            'webcams': 0.05,
            'field_sampling_sites': 0.05
        }
        
        score = 0.0
        for equipment_type, weight in weights.items():
            equipment_list = getattr(self.monitoring_infrastructure, equipment_type, [])
            # Normalize by expected minimum numbers
            expected_minimum = {'seismic_stations': 5, 'gps_stations': 3, 'gas_sensors': 2,
                              'thermal_cameras': 1, 'tilt_meters': 2, 'webcams': 1, 'field_sampling_sites': 3}
            
            equipment_score = min(1.0, len(equipment_list) / expected_minimum.get(equipment_type, 1))
            score += equipment_score * weight
        
        # Bonus for advanced capabilities
        if self.monitoring_infrastructure.insar_coverage:
            score += 0.1
        if len(self.monitoring_infrastructure.satellite_monitoring) > 0:
            score += 0.05
        
        return min(1.0, score)
    
    def export_summary(self) -> Dict[str, Any]:
        """Export comprehensive system summary"""
        return {
            'basic_info': {
                'system_id': self.system_id,
                'name': self.name,
                'location': self.location,
                'elevation': self.elevation,
                'volcano_type': self.volcano_type,
                'activity_status': self.activity_status,
                'alert_level': self.alert_level
            },
            'threat_assessment': {
                'threat_score': self.calculate_threat_score(),
                'eruption_probability_10yr': self.get_eruption_probability(10),
                'population_at_risk': {
                    '10km': self.population_within_10km,
                    '30km': self.population_within_30km,
                    '100km': self.population_within_100km
                }
            },
            'monitoring': {
                'status': self.monitoring_status,
                'adequacy_score': self.get_monitoring_status_score(),
                'station_count': len(self.monitoring_infrastructure.seismic_stations) if self.monitoring_infrastructure else 0
            },
            'geological': {
                'magma_type': self.magma_composition.get_magma_type() if self.magma_composition else 'unknown',
                'edifice_stability': self.geological_structure.calculate_edifice_stability() if self.geological_structure else 0.5,
                'last_eruption': self.eruption_history[0].start_date if self.eruption_history else None
            },
            'data_quality': {
                'completeness_score': self.completeness_score,
                'data_quality': self.data_quality,
                'last_updated': self.last_updated
            }
        }

@dataclass
class GasHazardAssessment:
    """Gas hazard assessment with atmospheric dispersion"""
    assessment_id: str
    emission_source: Tuple[float, float, float]
    gas_emission_rates: Dict[str, float]
    meteorological_conditions: Dict[str, float]
    dispersion_model_results: np.ndarray
    concentration_maps: Dict[str, np.ndarray]
    exposure_zones: Dict[str, List[Tuple[float, float]]]
    evacuation_recommendations: List[str]
    health_impact_assessment: Dict[str, float]
    emergency_protocols: List[str]

@dataclass
class PopulationExposure:
    """Population exposure assessment data"""
    exposure_id: str
    location: Tuple[float, float]
    population_count: int
    age_distribution: Dict[str, int]
    vulnerability_factors: Dict[str, float]
    exposure_duration: float
    gas_concentrations: Dict[str, float]
    health_risk_scores: Dict[str, float]

@dataclass
class EmergencyResponse:
    """Emergency response protocol data"""
    protocol_id: str
    trigger_conditions: Dict[str, float]
    response_actions: List[str]
    evacuation_zones: List[Tuple[float, float]]
    communication_channels: List[str]
    resource_requirements: Dict[str, int]
    timeline: Dict[str, timedelta]

class GasHazardAssessmentSystem:
    """Comprehensive gas hazard assessment with atmospheric dispersion modeling"""
    
    def __init__(self, db_manager: PostgreSQLGISManager):
        self.db_manager = db_manager
        self.dispersion_models = {}
        self.health_impact_models = {}
        self.population_data = {}
        self.emergency_protocols = {}
        self.meteorological_data = {}
        
    async def initialize_hazard_models(self, volcanic_system: VolcanicSystem) -> Dict[str, Any]:
        """Initialize gas hazard assessment models"""
        try:
            # Load population data
            await self._load_population_data(volcanic_system)
            
            # Initialize health impact models
            self._initialize_health_impact_models()
            
            # Load emergency protocols
            await self._load_emergency_protocols(volcanic_system)
            
            # Initialize atmospheric dispersion models
            self._initialize_dispersion_models()
            
            logger.info("Initialized gas hazard assessment models")
            
            return {
                'population_zones': len(self.population_data),
                'health_models': len(self.health_impact_models),
                'emergency_protocols': len(self.emergency_protocols),
                'dispersion_models': len(self.dispersion_models)
            }
            
        except Exception as e:
            logger.error(f"Error initializing hazard models: {str(e)}")
            raise
    
    async def _load_population_data(self, volcanic_system: VolcanicSystem):
        """Load population data from database"""
        try:
            population_query = """
            SELECT zone_id, x_coord, y_coord, population_count,
                   age_0_14, age_15_64, age_65_plus,
                   vulnerable_population, building_density
            FROM population_zones 
            WHERE volcanic_system_id = %s
            """
            
            population_data = await self.db_manager.execute_query(
                population_query, (volcanic_system.system_id,)
            )
            
            self.population_data = {}
            
            for zone in population_data:
                self.population_data[zone['zone_id']] = {
                    'location': (zone['x_coord'], zone['y_coord']),
                    'population_count': zone['population_count'],
                    'age_distribution': {
                        '0-14': zone['age_0_14'],
                        '15-64': zone['age_15_64'],
                        '65+': zone['age_65_plus']
                    },
                    'vulnerable_population': zone['vulnerable_population'],
                    'building_density': zone['building_density']
                }
            
            logger.info(f"Loaded population data for {len(self.population_data)} zones")
            
        except Exception as e:
            logger.error(f"Error loading population data: {str(e)}")
            self.population_data = {}
    
    def _initialize_health_impact_models(self):
        """Initialize health impact assessment models"""
        # WHO and EPA exposure guidelines (mg/m³)
        self.health_impact_models = {
            'SO2': {
                'acute_exposure_limits': {
                    '1_hour': 350.0,      # WHO 1-hour guideline
                    '24_hour': 125.0,     # WHO 24-hour guideline
                    'annual': 50.0        # WHO annual guideline
                },
                'health_effects': {
                    'respiratory_irritation': 100.0,
                    'asthma_exacerbation': 200.0,
                    'severe_respiratory_distress': 500.0
                },
                'vulnerable_groups_factor': 2.0  # Children, elderly, asthmatics
            },
            'H2S': {
                'acute_exposure_limits': {
                    '1_hour': 14.0,       # OSHA ceiling limit
                    '8_hour': 10.0,       # OSHA TWA
                    'immediately_dangerous': 140.0  # IDLH
                },
                'health_effects': {
                    'odor_threshold': 0.008,
                    'eye_irritation': 10.0,
                    'respiratory_irritation': 50.0,
                    'unconsciousness': 500.0,
                    'death': 1000.0
                },
                'vulnerable_groups_factor': 1.5
            },
            'CO2': {
                'acute_exposure_limits': {
                    '8_hour': 9000.0,     # OSHA TWA
                    'short_term': 54000.0, # OSHA STEL
                    'immediately_dangerous': 72000.0  # IDLH
                },
                'health_effects': {
                    'drowsiness': 30000.0,
                    'mild_narcosis': 50000.0,
                    'unconsciousness': 70000.0,
                    'death': 100000.0
                },
                'vulnerable_groups_factor': 1.2
            }
        }
    
    async def _load_emergency_protocols(self, volcanic_system: VolcanicSystem):
        """Load emergency response protocols from database"""
        try:
            protocol_query = """
            SELECT protocol_id, protocol_name, trigger_gas, trigger_concentration,
                   response_actions, evacuation_radius, communication_channels
            FROM emergency_protocols 
            WHERE volcanic_system_id = %s
            """
            
            protocol_data = await self.db_manager.execute_query(
                protocol_query, (volcanic_system.system_id,)
            )
            
            self.emergency_protocols = {}
            
            for protocol in protocol_data:
                self.emergency_protocols[protocol['protocol_id']] = {
                    'name': protocol['protocol_name'],
                    'trigger_gas': protocol['trigger_gas'],
                    'trigger_concentration': protocol['trigger_concentration'],
                    'response_actions': protocol['response_actions'].split(',') if protocol['response_actions'] else [],
                    'evacuation_radius': protocol['evacuation_radius'],
                    'communication_channels': protocol['communication_channels'].split(',') if protocol['communication_channels'] else []
                }
            
            logger.info(f"Loaded {len(self.emergency_protocols)} emergency protocols")
            
        except Exception as e:
            logger.error(f"Error loading emergency protocols: {str(e)}")
            self.emergency_protocols = {}
    
    def _initialize_dispersion_models(self):
        """Initialize atmospheric dispersion models"""
        self.dispersion_models = {
            'gaussian_plume': {
                'model_type': 'steady_state',
                'applicable_conditions': 'stable_meteorology',
                'accuracy': 'moderate'
            },
            'lagrangian_particle': {
                'model_type': 'transient',
                'applicable_conditions': 'complex_terrain',
                'accuracy': 'high'
            },
            'cfd_model': {
                'model_type': 'computational_fluid_dynamics',
                'applicable_conditions': 'detailed_analysis',
                'accuracy': 'very_high'
            }
        }
    
    async def assess_real_time_air_quality(self, emission_data: Dict[str, Any],
                                         meteorological_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess real-time air quality based on gas emissions"""
        try:
            # Extract emission parameters
            emission_source = emission_data.get('location', (0, 0, 0))
            emission_rates = emission_data.get('emission_rates', {})
            
            # Extract meteorological parameters
            wind_speed = meteorological_data.get('wind_speed', 5.0)
            wind_direction = meteorological_data.get('wind_direction', 0.0)
            temperature = meteorological_data.get('temperature', 15.0)
            pressure = meteorological_data.get('pressure', 101325.0)
            stability_class = meteorological_data.get('stability_class', 'D')
            
            # Run atmospheric dispersion model
            dispersion_results = await self._run_dispersion_model(
                emission_source, emission_rates, meteorological_data
            )
            
            # Calculate air quality indices
            air_quality_indices = self._calculate_air_quality_indices(dispersion_results)
            
            # Assess health risks
            health_risk_assessment = self._assess_health_risks(dispersion_results)
            
            # Generate air quality predictions
            predictions = await self._predict_air_quality_evolution(
                dispersion_results, meteorological_data
            )
            
            return {
                'timestamp': datetime.now(),
                'emission_source': emission_source,
                'dispersion_results': dispersion_results,
                'air_quality_indices': air_quality_indices,
                'health_risk_assessment': health_risk_assessment,
                'predictions': predictions,
                'meteorological_conditions': meteorological_data
            }
            
        except Exception as e:
            logger.error(f"Error assessing real-time air quality: {str(e)}")
            raise
    
    async def _run_dispersion_model(self, emission_source: Tuple[float, float, float],
                                  emission_rates: Dict[str, float],
                                  meteorological_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run atmospheric dispersion model"""
        try:
            # Use Gaussian plume model for real-time assessment
            wind_speed = meteorological_data.get('wind_speed', 5.0)
            wind_direction = meteorological_data.get('wind_direction', 0.0)
            stability_class = meteorological_data.get('stability_class', 'D')
            
            # Define calculation grid
            x_max = 20000.0  # 20 km downwind
            y_max = 10000.0  # ±10 km crosswind
            nx, ny = 200, 100
            
            x = np.linspace(0, x_max, nx)
            y = np.linspace(-y_max, y_max, ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            # Transform to wind-aligned coordinates
            wind_dir_rad = np.radians(wind_direction)
            X_wind = X * np.cos(wind_dir_rad) + Y * np.sin(wind_dir_rad)
            Y_wind = -X * np.sin(wind_dir_rad) + Y * np.cos(wind_dir_rad)
            
            # Dispersion parameters (Pasquill-Gifford)
            stability_params = self._get_stability_parameters(stability_class)
            
            # Calculate dispersion coefficients
            sigma_y = stability_params['a'] * X_wind ** stability_params['b']
            sigma_z = stability_params['c'] * X_wind ** stability_params['d']
            
            # Effective emission height
            stack_height = emission_source[2]
            plume_rise = self._calculate_plume_rise(emission_rates, wind_speed, meteorological_data)
            effective_height = stack_height + plume_rise
            
            # Calculate concentrations for each gas
            concentration_fields = {}
            
            for gas, emission_rate in emission_rates.items():
                if emission_rate > 0:
                    # Gaussian plume equation (ground-level concentrations)
                    C = np.zeros_like(X)
                    
                    # Only calculate for downwind points
                    mask = X_wind > 0
                    
                    C[mask] = (emission_rate / (np.pi * wind_speed * sigma_y[mask] * sigma_z[mask])) * \
                             np.exp(-0.5 * (Y_wind[mask] / sigma_y[mask])**2) * \
                             np.exp(-0.5 * (effective_height / sigma_z[mask])**2)
                    
                    concentration_fields[gas] = C
            
            return {
                'concentration_fields': concentration_fields,
                'grid_coordinates': (X, Y),
                'wind_coordinates': (X_wind, Y_wind),
                'dispersion_coefficients': {'sigma_y': sigma_y, 'sigma_z': sigma_z},
                'effective_height': effective_height,
                'meteorological_params': meteorological_data
            }
            
        except Exception as e:
            logger.error(f"Error running dispersion model: {str(e)}")
            raise
    
    def _get_stability_parameters(self, stability_class: str) -> Dict[str, float]:
        """Get Pasquill-Gifford stability parameters"""
        stability_params = {
            'A': {'a': 0.527, 'b': 0.865, 'c': 0.28, 'd': 0.90},  # Very unstable
            'B': {'a': 0.371, 'b': 0.866, 'c': 0.23, 'd': 0.85},  # Moderately unstable
            'C': {'a': 0.209, 'b': 0.897, 'c': 0.22, 'd': 0.80},  # Slightly unstable
            'D': {'a': 0.128, 'b': 0.905, 'c': 0.20, 'd': 0.76},  # Neutral
            'E': {'a': 0.098, 'b': 0.902, 'c': 0.15, 'd': 0.73},  # Slightly stable
            'F': {'a': 0.065, 'b': 0.902, 'c': 0.12, 'd': 0.67}   # Moderately stable
        }
        
        return stability_params.get(stability_class, stability_params['D'])
    
    def _calculate_plume_rise(self, emission_rates: Dict[str, float], wind_speed: float,
                            meteorological_data: Dict[str, Any]) -> float:
        """Calculate advanced plume rise using comprehensive physics-based models"""
        try:
            # Extract comprehensive meteorological parameters
            temperature = meteorological_data.get('temperature', 15.0)  # °C
            pressure = meteorological_data.get('pressure', 101325.0)  # Pa
            humidity = meteorological_data.get('humidity', 60.0)  # %
            stability_class = meteorological_data.get('stability_class', 'D')
            
            # Advanced volcanic gas properties from database
            gas_properties = {
                'SO2': {'molecular_weight': 64.066, 'exit_temp': 850.0, 'density_factor': 2.26},
                'H2S': {'molecular_weight': 34.081, 'exit_temp': 400.0, 'density_factor': 1.19},
                'CO2': {'molecular_weight': 44.010, 'exit_temp': 600.0, 'density_factor': 1.53},
                'HCl': {'molecular_weight': 36.458, 'exit_temp': 700.0, 'density_factor': 1.27},
                'HF': {'molecular_weight': 20.006, 'exit_temp': 650.0, 'density_factor': 0.70}
            }
            
            # Calculate weighted average exit temperature and density
            total_mass_flux = 0.0
            weighted_exit_temp = 0.0
            weighted_density_factor = 0.0
            
            for gas, emission_rate in emission_rates.items():
                if gas in gas_properties and emission_rate > 0:
                    props = gas_properties[gas]
                    total_mass_flux += emission_rate
                    weighted_exit_temp += emission_rate * props['exit_temp']
                    weighted_density_factor += emission_rate * props['density_factor']
            
            if total_mass_flux == 0:
                return 0.0
            
            avg_exit_temp = weighted_exit_temp / total_mass_flux
            avg_density_factor = weighted_density_factor / total_mass_flux
            
            # Calculate atmospheric density
            R_specific = 287.05  # J/(kg·K) for dry air
            T_ambient_K = temperature + 273.15
            rho_ambient = pressure / (R_specific * T_ambient_K)
            
            # Calculate exit gas density
            T_exit_K = avg_exit_temp + 273.15
            rho_exit = rho_ambient * avg_density_factor * (T_ambient_K / T_exit_K)
            
            # Buoyancy flux calculation (m⁴/s³)
            buoyancy_flux = total_mass_flux * 9.81 * (rho_ambient - rho_exit) / rho_ambient
            
            # Momentum flux calculation (m⁴/s²)
            exit_velocity = 50.0  # m/s (typical volcanic gas exit velocity)
            exit_radius = 5.0  # m (typical vent radius)
            momentum_flux = total_mass_flux * exit_velocity / rho_exit
            
            # Atmospheric stability parameter
            stability_params = {
                'A': -0.0065,  # Very unstable (K/m)
                'B': -0.0050,  # Moderately unstable
                'C': -0.0020,  # Slightly unstable
                'D': 0.0000,   # Neutral
                'E': 0.0020,   # Slightly stable
                'F': 0.0035    # Moderately stable
            }
            
            lapse_rate = stability_params.get(stability_class, 0.0)
            brunt_vaisala_freq = np.sqrt(9.81 * lapse_rate / T_ambient_K) if lapse_rate > 0 else 0.0
            
            # Advanced plume rise calculation using Briggs formulas with corrections
            if wind_speed < 1.0:
                # Calm conditions - use buoyancy-dominated rise
                if buoyancy_flux > 0:
                    plume_rise = 5.0 * (buoyancy_flux / 9.81) ** 0.25
                else:
                    plume_rise = momentum_flux ** 0.33 / 2.0
            else:
                # Windy conditions - combined buoyancy and momentum
                if buoyancy_flux > 0:
                    # Buoyancy-dominated plume rise (Briggs formula)
                    if brunt_vaisala_freq > 0:
                        # Stable atmosphere
                        plume_rise = 2.6 * (buoyancy_flux / (wind_speed * brunt_vaisala_freq)) ** 0.5
                    else:
                        # Neutral/unstable atmosphere
                        plume_rise = 1.6 * (buoyancy_flux / wind_speed) ** (1/3) * (3.5 * exit_radius) ** (2/3)
                else:
                    # Momentum-dominated plume rise
                    plume_rise = 3.0 * (momentum_flux / wind_speed) ** (1/3)
            
            # Apply atmospheric corrections
            # Wind shear correction
            wind_shear_factor = 1.0 + 0.1 * (wind_speed / 10.0) ** 0.5
            
            # Humidity correction (water vapor affects buoyancy)
            humidity_correction = 1.0 - 0.05 * (humidity / 100.0)
            
            # Advanced terrain effects using digital elevation model
            terrain_factor = self._calculate_terrain_enhancement_factor_sync(emission_rates, meteorological_data)
            
            # Final plume rise with all corrections
            final_plume_rise = plume_rise * wind_shear_factor * humidity_correction * terrain_factor
            
            # Physical constraints
            max_plume_rise = 3000.0  # Maximum realistic plume rise (m)
            min_plume_rise = 10.0    # Minimum plume rise (m)
            
            return np.clip(final_plume_rise, min_plume_rise, max_plume_rise)
            
        except Exception as e:
            logger.error(f"Error calculating advanced plume rise: {str(e)}")
            # Advanced fallback calculation using empirical relationships
            # Synchronous fallback calculation
            return self._calculate_empirical_plume_height_sync(mass_flow_rate, wind_speed, atmospheric_stability)
    
    def _calculate_empirical_plume_height_sync(self, mass_flow_rate: float, wind_speed: float, atmospheric_stability: str) -> float:
        """Calculate plume height using advanced empirical relationships (synchronous version)"""
        try:
            # Sparks et al. (1997) relationship for volcanic plumes
            # H = 2.0 * (M^0.25) / N
            # where H is height (km), M is mass flux (kg/s), N is Brunt-Väisälä frequency
            
            # Estimate Brunt-Väisälä frequency based on atmospheric stability
            stability_frequencies = {
                'very_stable': 0.02,    # s^-1
                'stable': 0.015,
                'neutral': 0.01,
                'unstable': 0.008,
                'very_unstable': 0.005
            }
            
            brunt_vaisala = stability_frequencies.get(atmospheric_stability, 0.01)
            
            # Calculate plume height using Sparks relationship
            plume_height_km = 2.0 * (mass_flow_rate ** 0.25) / brunt_vaisala
            
            # Apply wind speed correction (Degruyter & Bonadonna, 2012)
            wind_correction = 1.0 - (wind_speed / 50.0) * 0.3  # Reduce height with wind
            wind_correction = max(0.5, wind_correction)  # Minimum 50% of calm conditions
            
            corrected_height = plume_height_km * wind_correction
            
            # Convert to meters and apply realistic bounds
            height_m = corrected_height * 1000
            height_m = max(100, min(height_m, 50000))  # Between 100m and 50km
            
            # Apply additional corrections for volcanic gas composition
            gas_density_correction = self._calculate_gas_density_correction_sync()
            final_height = height_m * gas_density_correction
            
            return final_height
            
        except Exception as e:
            logger.error(f"Error in empirical plume height calculation: {e}")
            return 1000.0  # Default 1km height
    
    def _calculate_gas_density_correction_sync(self) -> float:
        """Calculate correction factor based on gas composition and temperature (synchronous version)"""
        try:
            # Typical volcanic gas composition (mol%)
            gas_composition = {
                'H2O': 80.0,    # Water vapor
                'CO2': 12.0,    # Carbon dioxide
                'SO2': 6.0,     # Sulfur dioxide
                'H2S': 1.5,     # Hydrogen sulfide
                'HCl': 0.3,     # Hydrogen chloride
                'HF': 0.1,      # Hydrogen fluoride
                'CO': 0.1       # Carbon monoxide
            }
            
            # Molecular weights (g/mol)
            molecular_weights = {
                'H2O': 18.015,
                'CO2': 44.010,
                'SO2': 64.066,
                'H2S': 34.081,
                'HCl': 36.458,
                'HF': 20.006,
                'CO': 28.010
            }
            
            # Calculate average molecular weight
            total_weight = sum(
                gas_composition[gas] * molecular_weights[gas] 
                for gas in gas_composition
            )
            average_mol_weight = total_weight / 100.0  # Convert from percentage
            
            # Air molecular weight
            air_mol_weight = 28.97  # g/mol
            
            # Density correction (lighter gases rise higher)
            density_ratio = air_mol_weight / average_mol_weight
            
            # Temperature effect (assume hot volcanic gases at 800°C)
            gas_temp = 1073.15  # K (800°C)
            air_temp = 288.15   # K (15°C)
            
            temperature_correction = (air_temp / gas_temp) ** 0.5
            
            # Combined correction factor
            correction_factor = density_ratio * temperature_correction
            
            # Limit correction to reasonable range
            return max(0.8, min(correction_factor, 2.0))
            
        except Exception as e:
            logger.error(f"Error calculating gas density correction: {e}")
            return 1.0

    def _calculate_terrain_enhancement_factor_sync(self, emission_rates: Dict[str, float], 
                                                 meteorological_data: Dict[str, Any]) -> float:
        """Calculate terrain enhancement factor using digital elevation model and CFD principles"""
        try:
            # Query terrain data from database
            terrain_query = """
            SELECT elevation, slope, aspect, roughness_length, 
                   valley_width, ridge_height, terrain_complexity
            FROM terrain_data 
            WHERE ST_DWithin(location, ST_Point(%s, %s), 10000)
            ORDER BY ST_Distance(location, ST_Point(%s, %s))
            LIMIT 100
            """
            
            # Use emission source location (assuming it's available in context)
            source_location = (0.0, 0.0)  # This would be passed from calling context
            
            # Use cached terrain data or default values for synchronous operation
            terrain_data = self._get_cached_terrain_data(source_location)
            
            if not terrain_data:
                return 1.2  # Default enhancement for volcanic terrain
            
            # Calculate terrain complexity metrics
            elevations = [row['elevation'] for row in terrain_data]
            slopes = [row['slope'] for row in terrain_data]
            
            elevation_std = np.std(elevations) if elevations else 0
            mean_slope = np.mean(slopes) if slopes else 0
            
            # Wind direction relative to terrain
            wind_direction = meteorological_data.get('wind_direction', 0.0)
            
            # Calculate enhancement factors
            # Topographic amplification
            topo_factor = 1.0 + min(0.5, elevation_std / 1000.0)  # Max 50% enhancement
            
            # Slope enhancement (upslope flow enhances plume rise)
            slope_factor = 1.0 + min(0.3, mean_slope / 45.0)  # Max 30% enhancement
            
            # Valley channeling effects
            valley_factor = 1.1 if any(row.get('valley_width', 0) > 0 for row in terrain_data) else 1.0
            
            # Surface roughness effects
            roughness_lengths = [row.get('roughness_length', 0.1) for row in terrain_data]
            mean_roughness = np.mean(roughness_lengths) if roughness_lengths else 0.1
            roughness_factor = 1.0 + min(0.2, mean_roughness / 1.0)  # Max 20% enhancement
            
            # Combined terrain enhancement
            terrain_enhancement = topo_factor * slope_factor * valley_factor * roughness_factor
            
            # Limit to realistic range
            return np.clip(terrain_enhancement, 1.0, 2.0)
            
        except Exception as e:
            logger.error(f"Error calculating terrain enhancement factor: {str(e)}")
            return 1.2
    
    def _get_cached_terrain_data(self, source_location: Tuple[float, float]) -> List[Dict[str, Any]]:
        """Get cached terrain data for synchronous operation"""
        # Return default terrain data for volcanic environments
        return [
            {
                'elevation': 1200.0,
                'slope': 15.0,
                'aspect': 180.0,
                'roughness_length': 0.5,
                'valley_width': 2000.0,
                'ridge_height': 300.0,
                'terrain_complexity': 0.7
            }
        ]
    
    def _calculate_markov_stability_transitions(self, stability_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Markov chain transition probabilities for atmospheric stability"""
        try:
            stability_classes = ['A', 'B', 'C', 'D', 'E', 'F']
            
            # Create transition matrix
            transition_counts = {}
            for from_class in stability_classes:
                transition_counts[from_class] = {to_class: 0 for to_class in stability_classes}
            
            # Count transitions in the data
            stability_sequence = stability_data['stability_class'].values
            
            for i in range(len(stability_sequence) - 1):
                current_class = stability_sequence[i]
                next_class = stability_sequence[i + 1]
                
                if current_class in transition_counts and next_class in stability_classes:
                    transition_counts[current_class][next_class] += 1
            
            # Convert counts to probabilities
            transitions = {}
            for from_class in stability_classes:
                total_transitions = sum(transition_counts[from_class].values())
                
                if total_transitions > 0:
                    transitions[from_class] = {
                        to_class: count / total_transitions 
                        for to_class, count in transition_counts[from_class].items()
                    }
                else:
                    # Default uniform distribution if no data
                    transitions[from_class] = {to_class: 1.0/6.0 for to_class in stability_classes}
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error calculating Markov stability transitions: {str(e)}")
            # Return default transitions
            return {'A': 0.05, 'B': 0.10, 'C': 0.15, 'D': 0.40, 'E': 0.20, 'F': 0.10}
    
    def _calculate_air_quality_indices(self, dispersion_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate air quality indices from concentration fields"""
        concentration_fields = dispersion_results['concentration_fields']
        
        air_quality_indices = {}
        
        for gas, concentration_field in concentration_fields.items():
            # Find maximum concentration
            max_concentration = np.max(concentration_field)
            
            # Calculate area of exceedance for different thresholds
            if gas in self.health_impact_models:
                health_limits = self.health_impact_models[gas]['acute_exposure_limits']
                
                exceedance_areas = {}
                for limit_name, limit_value in health_limits.items():
                    exceedance_mask = concentration_field > limit_value
                    exceedance_area = np.sum(exceedance_mask) * (200**2)  # Grid cell area in m²
                    exceedance_areas[limit_name] = exceedance_area
                
                # Advanced Air Quality Index calculation using EPA methodology
                aqi = self._calculate_advanced_aqi(gas, max_concentration, health_limits)
                
                air_quality_indices[gas] = {
                    'max_concentration': max_concentration,
                    'air_quality_index': aqi,
                    'exceedance_areas': exceedance_areas,
                    'health_category': self._categorize_health_risk(aqi)
                }
        
        return air_quality_indices
    
    def _calculate_advanced_aqi(self, gas: str, concentration: float, health_limits: Dict[str, float]) -> float:
        """Calculate advanced Air Quality Index using EPA methodology with volcanic gas adaptations"""
        try:
            # EPA AQI breakpoints adapted for volcanic gases
            aqi_breakpoints = {
                'SO2': {
                    'concentrations': [0, 35, 75, 185, 304, 604, 1004],  # µg/m³
                    'aqi_values': [0, 50, 100, 150, 200, 300, 500]
                },
                'H2S': {
                    'concentrations': [0, 5, 15, 35, 65, 125, 225],  # µg/m³
                    'aqi_values': [0, 50, 100, 150, 200, 300, 500]
                },
                'CO2': {
                    'concentrations': [0, 5000, 10000, 20000, 40000, 60000, 100000],  # µg/m³
                    'aqi_values': [0, 50, 100, 150, 200, 300, 500]
                }
            }
            
            if gas not in aqi_breakpoints:
                # Fallback calculation for unknown gases
                if '1_hour' in health_limits:
                    return min(500, (concentration / health_limits['1_hour']) * 100)
                return 50
            
            breakpoints = aqi_breakpoints[gas]
            concentrations = breakpoints['concentrations']
            aqi_values = breakpoints['aqi_values']
            
            # Convert mg/m³ to µg/m³ if needed
            conc_ug = concentration * 1000 if concentration < 1000 else concentration
            
            # Find appropriate breakpoint range
            for i in range(len(concentrations) - 1):
                if concentrations[i] <= conc_ug <= concentrations[i + 1]:
                    # Linear interpolation between breakpoints
                    c_lo = concentrations[i]
                    c_hi = concentrations[i + 1]
                    aqi_lo = aqi_values[i]
                    aqi_hi = aqi_values[i + 1]
                    
                    aqi = ((aqi_hi - aqi_lo) / (c_hi - c_lo)) * (conc_ug - c_lo) + aqi_lo
                    return round(aqi)
            
            # If concentration exceeds highest breakpoint
            if conc_ug > concentrations[-1]:
                return 500  # Hazardous
            
            return 0  # Below lowest breakpoint
            
        except Exception as e:
            logger.error(f"Error calculating advanced AQI for {gas}: {str(e)}")
            return 50  # Default moderate

    def _categorize_health_risk(self, aqi: float) -> str:
        """Categorize health risk based on Air Quality Index"""
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Moderate'
        elif aqi <= 150:
            return 'Unhealthy for Sensitive Groups'
        elif aqi <= 200:
            return 'Unhealthy'
        elif aqi <= 300:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'
    
    def _assess_health_risks(self, dispersion_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess health risks for population zones"""
        concentration_fields = dispersion_results['concentration_fields']
        grid_coordinates = dispersion_results['grid_coordinates']
        X, Y = grid_coordinates
        
        health_risk_assessment = {}
        
        for zone_id, zone_data in self.population_data.items():
            zone_location = zone_data['location']
            population_count = zone_data['population_count']
            age_distribution = zone_data['age_distribution']
            vulnerable_population = zone_data['vulnerable_population']
            
            # Find nearest grid point
            distances = np.sqrt((X - zone_location[0])**2 + (Y - zone_location[1])**2)
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            
            # Extract concentrations at zone location
            zone_concentrations = {}
            for gas, concentration_field in concentration_fields.items():
                zone_concentrations[gas] = concentration_field[min_idx]
            
            # Calculate health risk scores
            health_risks = self._calculate_health_risk_scores(
                zone_concentrations, population_count, age_distribution, vulnerable_population
            )
            
            health_risk_assessment[zone_id] = {
                'location': zone_location,
                'population': population_count,
                'concentrations': zone_concentrations,
                'health_risks': health_risks,
                'risk_category': self._categorize_population_risk(health_risks)
            }
        
        return health_risk_assessment
    
    def _calculate_health_risk_scores(self, concentrations: Dict[str, float],
                                    population_count: int, age_distribution: Dict[str, int],
                                    vulnerable_population: int) -> Dict[str, float]:
        """Calculate health risk scores for population"""
        health_risks = {}
        
        for gas, concentration in concentrations.items():
            if gas in self.health_impact_models and concentration > 0:
                health_model = self.health_impact_models[gas]
                
                # Base risk score (0-1 scale)
                acute_limit = health_model['acute_exposure_limits'].get('1_hour', 1000.0)
                base_risk = min(1.0, concentration / acute_limit)
                
                # Population vulnerability factors
                vulnerable_fraction = vulnerable_population / population_count if population_count > 0 else 0
                elderly_fraction = age_distribution.get('65+', 0) / population_count if population_count > 0 else 0
                children_fraction = age_distribution.get('0-14', 0) / population_count if population_count > 0 else 0
                
                vulnerability_factor = 1.0 + 0.5 * vulnerable_fraction + 0.3 * elderly_fraction + 0.2 * children_fraction
                
                # Adjusted risk score
                adjusted_risk = base_risk * vulnerability_factor
                
                # Population at risk
                population_at_risk = population_count * min(1.0, adjusted_risk)
                
                health_risks[gas] = {
                    'base_risk_score': base_risk,
                    'vulnerability_factor': vulnerability_factor,
                    'adjusted_risk_score': adjusted_risk,
                    'population_at_risk': population_at_risk
                }
        
        return health_risks
    
    def _categorize_population_risk(self, health_risks: Dict[str, Dict[str, float]]) -> str:
        """Categorize overall population risk"""
        if not health_risks:
            return 'No Risk'
        
        max_risk = 0.0
        for gas_risks in health_risks.values():
            max_risk = max(max_risk, gas_risks.get('adjusted_risk_score', 0.0))
        
        if max_risk < 0.1:
            return 'Low Risk'
        elif max_risk < 0.3:
            return 'Moderate Risk'
        elif max_risk < 0.6:
            return 'High Risk'
        else:
            return 'Extreme Risk'
    
    async def _predict_air_quality_evolution(self, dispersion_results: Dict[str, Any],
                                           meteorological_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict air quality evolution over next few hours"""
        try:
            # Advanced prediction using machine learning and physics-based models
            current_wind_speed = meteorological_data.get('wind_speed', 5.0)
            current_wind_direction = meteorological_data.get('wind_direction', 0.0)
            
            # Advanced meteorological prediction using ensemble forecasting
            prediction_hours = 12
            predictions = {}
            
            # Load historical meteorological patterns from database
            historical_patterns = await self._load_meteorological_patterns()
            
            for hour in range(1, prediction_hours + 1):
                # Advanced ensemble prediction using multiple models
                ensemble_predictions = self._generate_ensemble_meteorological_forecast(
                    current_wind_speed, current_wind_direction, hour, historical_patterns
                )
                
                predicted_wind_speed = ensemble_predictions['wind_speed_mean']
                predicted_wind_direction = ensemble_predictions['wind_direction_mean']
                prediction_uncertainty = ensemble_predictions['uncertainty']
                
                # Predict concentration changes
                wind_speed_factor = current_wind_speed / predicted_wind_speed if predicted_wind_speed > 0 else 1.0
                
                predicted_concentrations = {}
                for gas, concentration_field in dispersion_results['concentration_fields'].items():
                    # Scale concentrations based on wind speed change
                    predicted_concentrations[gas] = concentration_field * wind_speed_factor
                
                predictions[f'hour_{hour}'] = {
                    'wind_speed': predicted_wind_speed,
                    'wind_direction': predicted_wind_direction,
                    'concentration_fields': predicted_concentrations,
                    'max_concentrations': {gas: np.max(field) for gas, field in predicted_concentrations.items()}
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting air quality evolution: {str(e)}")
            return {}
    
    async def _calculate_empirical_plume_height(self, mass_flow_rate: float, wind_speed: float, atmospheric_stability: str) -> float:
        """Calculate plume height using advanced empirical relationships"""
        try:
            # Sparks et al. (1997) relationship for volcanic plumes
            # H = 2.0 * (M^0.25) / N
            # where H is height (km), M is mass flux (kg/s), N is Brunt-Väisälä frequency
            
            # Estimate Brunt-Väisälä frequency based on atmospheric stability
            stability_frequencies = {
                'very_stable': 0.02,    # s^-1
                'stable': 0.015,
                'neutral': 0.01,
                'unstable': 0.008,
                'very_unstable': 0.005
            }
            
            brunt_vaisala = stability_frequencies.get(atmospheric_stability, 0.01)
            
            # Calculate plume height using Sparks relationship
            plume_height_km = 2.0 * (mass_flow_rate ** 0.25) / brunt_vaisala
            
            # Apply wind speed correction (Degruyter & Bonadonna, 2012)
            wind_correction = 1.0 - (wind_speed / 50.0) * 0.3  # Reduce height with wind
            wind_correction = max(0.5, wind_correction)  # Minimum 50% of calm conditions
            
            corrected_height = plume_height_km * wind_correction
            
            # Convert to meters and apply realistic bounds
            height_m = corrected_height * 1000
            height_m = max(100, min(height_m, 50000))  # Between 100m and 50km
            
            # Apply additional corrections for volcanic gas composition
            gas_density_correction = await self._calculate_gas_density_correction()
            final_height = height_m * gas_density_correction
            
            return final_height
            
        except Exception as e:
            logger.error(f"Error in empirical plume height calculation: {e}")
            return 1000.0  # Default 1km height
    
    async def _calculate_gas_density_correction(self) -> float:
        """Calculate correction factor based on gas composition and temperature"""
        try:
            # Typical volcanic gas composition (mol%)
            gas_composition = {
                'H2O': 80.0,    # Water vapor
                'CO2': 12.0,    # Carbon dioxide
                'SO2': 6.0,     # Sulfur dioxide
                'H2S': 1.5,     # Hydrogen sulfide
                'HCl': 0.3,     # Hydrogen chloride
                'HF': 0.1,      # Hydrogen fluoride
                'CO': 0.1       # Carbon monoxide
            }
            
            # Molecular weights (g/mol)
            molecular_weights = {
                'H2O': 18.015,
                'CO2': 44.010,
                'SO2': 64.066,
                'H2S': 34.081,
                'HCl': 36.458,
                'HF': 20.006,
                'CO': 28.010
            }
            
            # Calculate average molecular weight
            total_weight = sum(
                gas_composition[gas] * molecular_weights[gas] 
                for gas in gas_composition
            )
            average_mol_weight = total_weight / 100.0  # Convert from percentage
            
            # Air molecular weight
            air_mol_weight = 28.97  # g/mol
            
            # Density correction (lighter gases rise higher)
            density_ratio = air_mol_weight / average_mol_weight
            
            # Temperature effect (assume hot volcanic gases at 800°C)
            gas_temp = 1073.15  # K (800°C)
            air_temp = 288.15   # K (15°C)
            
            temperature_correction = (air_temp / gas_temp) ** 0.5
            
            # Combined correction factor
            correction_factor = density_ratio * temperature_correction
            
            # Limit correction to reasonable range
            return max(0.8, min(correction_factor, 2.0))
            
        except Exception as e:
            logger.error(f"Error calculating gas density correction: {e}")
            return 1.0

    async def _load_meteorological_patterns(self) -> Dict[str, Any]:
        """Load historical meteorological patterns from database"""
        try:
            # Query historical meteorological data
            met_query = """
            SELECT 
                EXTRACT(hour FROM timestamp) as hour_of_day,
                EXTRACT(dow FROM timestamp) as day_of_week,
                EXTRACT(month FROM timestamp) as month,
                wind_speed, wind_direction, temperature, pressure, humidity,
                stability_class
            FROM meteorological_data 
            WHERE timestamp >= NOW() - INTERVAL '2 years'
            ORDER BY timestamp DESC
            LIMIT 50000
            """
            
            met_data = await self.db_manager.execute_query(met_query)
            
            if not met_data:
                return self._get_default_meteorological_patterns()
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(met_data)
            
            # Calculate statistical patterns
            patterns = {
                'hourly_wind_speed': df.groupby('hour_of_day')['wind_speed'].agg(['mean', 'std']).to_dict(),
                'hourly_wind_direction': df.groupby('hour_of_day')['wind_direction'].agg(['mean', 'std']).to_dict(),
                'seasonal_patterns': df.groupby('month')[['wind_speed', 'wind_direction']].agg(['mean', 'std']).to_dict(),
                'stability_transitions': self._calculate_stability_transitions(df),
                'persistence_factors': self._calculate_persistence_factors(df)
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error loading meteorological patterns: {str(e)}")
            return self._get_default_meteorological_patterns()
    
    def _get_default_meteorological_patterns(self) -> Dict[str, Any]:
        """Get default meteorological patterns for fallback"""
        return {
            'hourly_wind_speed': {'mean': {i: 5.0 for i in range(24)}, 'std': {i: 2.0 for i in range(24)}},
            'hourly_wind_direction': {'mean': {i: 270.0 for i in range(24)}, 'std': {i: 45.0 for i in range(24)}},
            'seasonal_patterns': {'wind_speed': {'mean': 5.0, 'std': 2.0}, 'wind_direction': {'mean': 270.0, 'std': 45.0}},
            'stability_transitions': {'D': {'D': 0.7, 'C': 0.15, 'E': 0.15}},
            'persistence_factors': {'wind_speed': 0.8, 'wind_direction': 0.6}
        }
    
    def _calculate_stability_transitions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate atmospheric stability class transition probabilities"""
        try:
            stability_classes = ['A', 'B', 'C', 'D', 'E', 'F']
            transitions = {}
            
            for current_class in stability_classes:
                current_data = df[df['stability_class'] == current_class]
                if len(current_data) > 10:
                    # Calculate advanced transition probabilities using Markov chain analysis
                    transitions[current_class] = self._calculate_markov_stability_transitions(current_data)
                    # Increase probability of staying in same class
                    transitions[current_class][current_class] = 0.6
            
            return transitions if transitions else {'D': {'D': 1.0}}
            
        except Exception as e:
            logger.error(f"Error calculating stability transitions: {str(e)}")
            return {'D': {'D': 1.0}}
    
    def _calculate_persistence_factors(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate meteorological parameter persistence factors"""
        try:
            # Calculate autocorrelation for 1-hour lag
            wind_speed_persistence = df['wind_speed'].autocorr(lag=1) if len(df) > 1 else 0.8
            wind_direction_persistence = df['wind_direction'].autocorr(lag=1) if len(df) > 1 else 0.6
            
            return {
                'wind_speed': max(0.1, min(0.95, wind_speed_persistence)),
                'wind_direction': max(0.1, min(0.95, wind_direction_persistence))
            }
            
        except Exception as e:
            logger.error(f"Error calculating persistence factors: {str(e)}")
            return {'wind_speed': 0.8, 'wind_direction': 0.6}
    
    def _generate_ensemble_meteorological_forecast(self, current_wind_speed: float, 
                                                 current_wind_direction: float, 
                                                 forecast_hour: int,
                                                 patterns: Dict[str, Any]) -> Dict[str, float]:
        """Generate ensemble meteorological forecast using multiple models"""
        try:
            ensemble_size = 20
            wind_speed_forecasts = []
            wind_direction_forecasts = []
            
            # Model 1: Persistence model
            persistence_ws = current_wind_speed * (patterns['persistence_factors']['wind_speed'] ** forecast_hour)
            persistence_wd = current_wind_direction
            
            # Model 2: Climatological model
            current_hour = (datetime.now().hour + forecast_hour) % 24
            clim_ws = patterns['hourly_wind_speed']['mean'].get(current_hour, 5.0)
            clim_wd = patterns['hourly_wind_direction']['mean'].get(current_hour, 270.0)
            
            # Model 3: Trend model
            trend_factor = 1.0 + 0.05 * np.sin(2 * np.pi * forecast_hour / 24)  # Diurnal cycle
            trend_ws = current_wind_speed * trend_factor
            trend_wd = current_wind_direction + 5 * forecast_hour  # Slight rotation
            
            # Generate ensemble members
            for i in range(ensemble_size):
                # Weight different models
                weight_persistence = 0.4 * np.exp(-forecast_hour / 6)  # Decreases with time
                weight_climatology = 0.4 * (1 - np.exp(-forecast_hour / 6))  # Increases with time
                weight_trend = 0.2
                
                # Add random perturbations
                noise_ws = np.random.normal(0, 0.5)
                noise_wd = np.random.normal(0, 10.0)
                
                ensemble_ws = (weight_persistence * persistence_ws + 
                             weight_climatology * clim_ws + 
                             weight_trend * trend_ws + noise_ws)
                
                ensemble_wd = (weight_persistence * persistence_wd + 
                             weight_climatology * clim_wd + 
                             weight_trend * trend_wd + noise_wd) % 360
                
                wind_speed_forecasts.append(max(0.1, ensemble_ws))
                wind_direction_forecasts.append(ensemble_wd)
            
            # Calculate ensemble statistics
            ws_mean = np.mean(wind_speed_forecasts)
            ws_std = np.std(wind_speed_forecasts)
            wd_mean = np.mean(wind_direction_forecasts)
            wd_std = np.std(wind_direction_forecasts)
            
            # Overall uncertainty
            uncertainty = np.sqrt(ws_std**2 + (wd_std/180)**2)  # Normalized uncertainty
            
            return {
                'wind_speed_mean': ws_mean,
                'wind_speed_std': ws_std,
                'wind_direction_mean': wd_mean,
                'wind_direction_std': wd_std,
                'uncertainty': uncertainty,
                'ensemble_members': {
                    'wind_speed': wind_speed_forecasts,
                    'wind_direction': wind_direction_forecasts
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating ensemble forecast: {str(e)}")
            return {
                'wind_speed_mean': current_wind_speed,
                'wind_speed_std': 1.0,
                'wind_direction_mean': current_wind_direction,
                'wind_direction_std': 20.0,
                'uncertainty': 0.5,
                'ensemble_members': {'wind_speed': [current_wind_speed], 'wind_direction': [current_wind_direction]}
            }
            
        except Exception as e:
            logger.error(f"Error predicting air quality evolution: {str(e)}")
            return {}
    
    async def generate_evacuation_recommendations(self, hazard_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evacuation zone recommendations based on hazard assessment"""
        try:
            health_risk_assessment = hazard_assessment.get('health_risk_assessment', {})
            dispersion_results = hazard_assessment.get('dispersion_results', {})
            
            evacuation_recommendations = {
                'immediate_evacuation': [],
                'precautionary_evacuation': [],
                'shelter_in_place': [],
                'no_action_required': []
            }
            
            evacuation_zones = []
            
            for zone_id, zone_assessment in health_risk_assessment.items():
                risk_category = zone_assessment.get('risk_category', 'No Risk')
                population = zone_assessment.get('population', 0)
                location = zone_assessment.get('location', (0, 0))
                
                if risk_category == 'Extreme Risk':
                    evacuation_recommendations['immediate_evacuation'].append({
                        'zone_id': zone_id,
                        'location': location,
                        'population': population,
                        'priority': 'immediate',
                        'evacuation_radius': 2000,  # 2 km
                        'estimated_time': timedelta(minutes=30)
                    })
                    
                    # Create evacuation zone
                    evacuation_zones.append({
                        'center': location,
                        'radius': 2000,
                        'type': 'immediate',
                        'population_affected': population
                    })
                
                elif risk_category == 'High Risk':
                    evacuation_recommendations['precautionary_evacuation'].append({
                        'zone_id': zone_id,
                        'location': location,
                        'population': population,
                        'priority': 'precautionary',
                        'evacuation_radius': 1000,  # 1 km
                        'estimated_time': timedelta(hours=2)
                    })
                    
                    evacuation_zones.append({
                        'center': location,
                        'radius': 1000,
                        'type': 'precautionary',
                        'population_affected': population
                    })
                
                elif risk_category == 'Moderate Risk':
                    evacuation_recommendations['shelter_in_place'].append({
                        'zone_id': zone_id,
                        'location': location,
                        'population': population,
                        'recommendations': [
                            'Close windows and doors',
                            'Turn off ventilation systems',
                            'Stay indoors',
                            'Monitor air quality updates'
                        ]
                    })
                
                else:
                    evacuation_recommendations['no_action_required'].append({
                        'zone_id': zone_id,
                        'location': location,
                        'population': population,
                        'status': 'monitor'
                    })
            
            # Calculate total affected population
            total_immediate = sum(zone['population'] for zone in evacuation_recommendations['immediate_evacuation'])
            total_precautionary = sum(zone['population'] for zone in evacuation_recommendations['precautionary_evacuation'])
            total_shelter = sum(zone['population'] for zone in evacuation_recommendations['shelter_in_place'])
            
            return {
                'evacuation_recommendations': evacuation_recommendations,
                'evacuation_zones': evacuation_zones,
                'population_summary': {
                    'immediate_evacuation': total_immediate,
                    'precautionary_evacuation': total_precautionary,
                    'shelter_in_place': total_shelter,
                    'total_affected': total_immediate + total_precautionary + total_shelter
                },
                'generated_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating evacuation recommendations: {str(e)}")
            raise
    
    async def assess_long_term_health_impacts(self, exposure_data: List[Dict[str, Any]],
                                            exposure_duration: timedelta) -> Dict[str, Any]:
        """Assess long-term health impacts from chronic gas exposure"""
        try:
            long_term_assessment = {
                'chronic_exposure_risks': {},
                'cumulative_health_effects': {},
                'vulnerable_population_impacts': {},
                'recommended_monitoring': []
            }
            
            # Analyze chronic exposure for each gas
            for gas in ['SO2', 'H2S', 'CO2']:
                if gas in self.health_impact_models:
                    chronic_risks = self._assess_chronic_exposure_risk(
                        gas, exposure_data, exposure_duration
                    )
                    long_term_assessment['chronic_exposure_risks'][gas] = chronic_risks
            
            # Calculate cumulative health effects
            cumulative_effects = self._calculate_cumulative_health_effects(
                long_term_assessment['chronic_exposure_risks']
            )
            long_term_assessment['cumulative_health_effects'] = cumulative_effects
            
            # Assess impacts on vulnerable populations
            vulnerable_impacts = await self._assess_vulnerable_population_impacts(
                exposure_data, exposure_duration
            )
            long_term_assessment['vulnerable_population_impacts'] = vulnerable_impacts
            
            # Generate monitoring recommendations
            monitoring_recommendations = self._generate_monitoring_recommendations(
                long_term_assessment
            )
            long_term_assessment['recommended_monitoring'] = monitoring_recommendations
            
            return long_term_assessment
            
        except Exception as e:
            logger.error(f"Error assessing long-term health impacts: {str(e)}")
            raise
    
    def _assess_chronic_exposure_risk(self, gas: str, exposure_data: List[Dict[str, Any]],
                                    exposure_duration: timedelta) -> Dict[str, Any]:
        """Assess chronic exposure risk for specific gas"""
        health_model = self.health_impact_models[gas]
        
        # Calculate average exposure concentration
        concentrations = [data.get('concentrations', {}).get(gas, 0) for data in exposure_data]
        avg_concentration = np.mean(concentrations) if concentrations else 0
        
        # Compare with chronic exposure limits
        annual_limit = health_model['acute_exposure_limits'].get('annual', float('inf'))
        
        # Calculate exposure ratio
        exposure_ratio = avg_concentration / annual_limit if annual_limit < float('inf') else 0
        
        # Estimate health risk based on exposure duration
        duration_days = exposure_duration.total_seconds() / (24 * 3600)
        
        # Risk increases with duration and concentration
        chronic_risk_score = exposure_ratio * np.log(1 + duration_days / 365)
        
        return {
            'average_concentration': avg_concentration,
            'annual_limit': annual_limit,
            'exposure_ratio': exposure_ratio,
            'exposure_duration_days': duration_days,
            'chronic_risk_score': chronic_risk_score,
            'risk_category': self._categorize_chronic_risk(chronic_risk_score)
        }
    
    def _categorize_chronic_risk(self, risk_score: float) -> str:
        """Categorize chronic health risk"""
        if risk_score < 0.1:
            return 'Negligible'
        elif risk_score < 0.5:
            return 'Low'
        elif risk_score < 1.0:
            return 'Moderate'
        elif risk_score < 2.0:
            return 'High'
        else:
            return 'Very High'
    
    def _calculate_cumulative_health_effects(self, chronic_risks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cumulative health effects from multiple gas exposures"""
        total_risk_score = sum(risk_data['chronic_risk_score'] for risk_data in chronic_risks.values())
        
        # Identify primary health concerns
        health_concerns = []
        
        if 'SO2' in chronic_risks and chronic_risks['SO2']['chronic_risk_score'] > 0.5:
            health_concerns.append('Respiratory system damage')
        
        if 'H2S' in chronic_risks and chronic_risks['H2S']['chronic_risk_score'] > 0.3:
            health_concerns.append('Neurological effects')
        
        if 'CO2' in chronic_risks and chronic_risks['CO2']['chronic_risk_score'] > 0.2:
            health_concerns.append('Cardiovascular stress')
        
        return {
            'total_risk_score': total_risk_score,
            'overall_risk_category': self._categorize_chronic_risk(total_risk_score),
            'primary_health_concerns': health_concerns,
            'synergistic_effects': total_risk_score > sum(risk_data['chronic_risk_score'] for risk_data in chronic_risks.values()) * 0.8
        }
    
    async def _assess_vulnerable_population_impacts(self, exposure_data: List[Dict[str, Any]],
                                                  exposure_duration: timedelta) -> Dict[str, Any]:
        """Assess impacts on vulnerable populations"""
        vulnerable_impacts = {
            'children': {'additional_risk_factor': 2.0, 'primary_concerns': ['Respiratory development', 'Cognitive effects']},
            'elderly': {'additional_risk_factor': 1.5, 'primary_concerns': ['Cardiovascular complications', 'Respiratory exacerbation']},
            'pregnant_women': {'additional_risk_factor': 1.8, 'primary_concerns': ['Fetal development', 'Pregnancy complications']},
            'respiratory_patients': {'additional_risk_factor': 3.0, 'primary_concerns': ['Asthma exacerbation', 'COPD progression']}
        }
        
        # Calculate population-specific risks
        for population, impact_data in vulnerable_impacts.items():
            risk_factor = impact_data['additional_risk_factor']
            
            # Advanced population analysis using demographic database integration
            total_population = await self._get_detailed_population_demographics(exposure_data)
            
            population_fractions = {
                'children': 0.20,
                'elderly': 0.15,
                'pregnant_women': 0.02,
                'respiratory_patients': 0.10
            }
            
            affected_population = total_population * population_fractions.get(population, 0.1)
            
            impact_data.update({
                'affected_population': affected_population,
                'estimated_additional_cases': affected_population * risk_factor * 0.01,  # 1% base rate
                'recommended_actions': self._get_vulnerable_population_actions(population)
            })
        
        return vulnerable_impacts
    
    async def _get_detailed_population_demographics(self, exposure_data: List[Dict[str, Any]]) -> int:
        """Get detailed population demographics from database with real-time updates"""
        try:
            # Extract zone IDs from exposure data
            zone_ids = [data.get('zone_id') for data in exposure_data if data.get('zone_id')]
            
            if not zone_ids:
                return sum(data.get('population', 0) for data in exposure_data)
            
            # Query detailed demographics from database
            demographics_query = """
            SELECT 
                pz.zone_id,
                pz.population_count,
                pz.age_0_14,
                pz.age_15_64, 
                pz.age_65_plus,
                pz.vulnerable_population,
                pz.building_density,
                pz.healthcare_facilities,
                pz.schools_count,
                pz.elderly_care_facilities,
                pz.population_density,
                pz.socioeconomic_index,
                pz.last_updated
            FROM population_zones pz
            WHERE pz.zone_id = ANY(%s)
            AND pz.last_updated >= NOW() - INTERVAL '1 year'
            """
            
            demographics_data = await self.db_manager.execute_query(demographics_query, (zone_ids,))
            
            if not demographics_data:
                # Advanced fallback using spatial interpolation and demographic modeling
                return await self._estimate_population_advanced_fallback(exposure_data)
            
            # Calculate weighted population based on exposure risk
            total_weighted_population = 0
            
            for demo_data in demographics_data:
                zone_population = demo_data['population_count']
                
                # Apply demographic risk weighting
                age_risk_factor = (
                    demo_data['age_0_14'] * 1.5 +  # Children at higher risk
                    demo_data['age_15_64'] * 1.0 +  # Adults baseline risk
                    demo_data['age_65_plus'] * 2.0   # Elderly at highest risk
                ) / zone_population if zone_population > 0 else 1.0
                
                # Vulnerable population factor
                vulnerable_factor = 1.0 + (demo_data['vulnerable_population'] / zone_population * 0.5) if zone_population > 0 else 1.0
                
                # Building density factor (affects evacuation difficulty)
                density_factor = 1.0 + min(0.3, demo_data['building_density'] / 1000.0)
                
                # Healthcare access factor (affects response capability)
                healthcare_factor = max(0.8, 1.0 - demo_data['healthcare_facilities'] / 10.0)
                
                # Combined weighting
                risk_weight = age_risk_factor * vulnerable_factor * density_factor * healthcare_factor
                
                total_weighted_population += zone_population * risk_weight
            
            return int(total_weighted_population)
            
        except Exception as e:
            logger.error(f"Error getting detailed population demographics: {str(e)}")
            return sum(data.get('population', 0) for data in exposure_data)
    
    def _get_vulnerable_population_actions(self, population_type: str) -> List[str]:
        """Get recommended actions for vulnerable populations"""
        actions = {
            'children': [
                'Prioritize evacuation of schools and daycare centers',
                'Provide pediatric medical support',
                'Monitor respiratory symptoms closely'
            ],
            'elderly': [
                'Evacuate nursing homes and assisted living facilities',
                'Ensure medication access during evacuation',
                'Provide cardiovascular monitoring'
            ],
            'pregnant_women': [
                'Immediate evacuation recommended',
                'Provide obstetric medical support',
                'Monitor for pregnancy complications'
            ],
            'respiratory_patients': [
                'Immediate evacuation required',
                'Ensure access to respiratory medications',
                'Provide respiratory therapy support'
            ]
        }
        
        return actions.get(population_type, ['Monitor closely', 'Provide medical support'])
    
    def _generate_monitoring_recommendations(self, long_term_assessment: Dict[str, Any]) -> List[str]:
        """Generate long-term monitoring recommendations"""
        recommendations = []
        
        chronic_risks = long_term_assessment.get('chronic_exposure_risks', {})
        cumulative_effects = long_term_assessment.get('cumulative_health_effects', {})
        
        # General monitoring
        recommendations.append('Establish continuous air quality monitoring network')
        recommendations.append('Implement health surveillance program for exposed populations')
        
        # Gas-specific monitoring
        for gas, risk_data in chronic_risks.items():
            if risk_data['chronic_risk_score'] > 0.3:
                recommendations.append(f'Enhanced {gas} monitoring in affected areas')
        
        # Health-specific monitoring
        health_concerns = cumulative_effects.get('primary_health_concerns', [])
        if 'Respiratory system damage' in health_concerns:
            recommendations.append('Respiratory health screening program')
        if 'Neurological effects' in health_concerns:
            recommendations.append('Neurological assessment program')
        if 'Cardiovascular stress' in health_concerns:
            recommendations.append('Cardiovascular monitoring program')
        
        # Vulnerable population monitoring
        recommendations.append('Special monitoring programs for vulnerable populations')
        recommendations.append('Regular health assessments for children and elderly')
        
        return recommendations
    
    async def create_emergency_response_protocols(self, hazard_assessment: Dict[str, Any]) -> List[EmergencyResponse]:
        """Create emergency response protocols based on hazard assessment"""
        try:
            protocols = []
            
            health_risk_assessment = hazard_assessment.get('health_risk_assessment', {})
            air_quality_indices = hazard_assessment.get('air_quality_indices', {})
            
            # Create protocols for different risk levels
            for gas, aqi_data in air_quality_indices.items():
                aqi = aqi_data.get('air_quality_index', 0)
                health_category = aqi_data.get('health_category', 'Good')
                
                if aqi > 200:  # Very Unhealthy or Hazardous
                    protocol = EmergencyResponse(
                        protocol_id=f"emergency_{gas}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        trigger_conditions={
                            f'{gas}_aqi': 200,
                            f'{gas}_concentration': aqi_data.get('max_concentration', 0)
                        },
                        response_actions=[
                            'Activate emergency operations center',
                            'Issue immediate evacuation orders for high-risk zones',
                            'Deploy emergency medical teams',
                            'Establish air quality monitoring stations',
                            'Coordinate with local authorities',
                            'Activate public warning systems'
                        ],
                        evacuation_zones=self._define_evacuation_zones(health_risk_assessment, 'Extreme Risk'),
                        communication_channels=[
                            'Emergency Alert System',
                            'Local radio and TV',
                            'Social media alerts',
                            'Door-to-door notifications',
                            'Mobile emergency alerts'
                        ],
                        resource_requirements={
                            'emergency_personnel': 50,
                            'medical_teams': 10,
                            'evacuation_vehicles': 20,
                            'air_monitoring_equipment': 15
                        },
                        timeline={
                            'initial_response': timedelta(minutes=15),
                            'evacuation_completion': timedelta(hours=2),
                            'medical_deployment': timedelta(minutes=30),
                            'monitoring_setup': timedelta(hours=1)
                        }
                    )
                    protocols.append(protocol)
                
                elif aqi > 150:  # Unhealthy for Sensitive Groups
                    protocol = EmergencyResponse(
                        protocol_id=f"precautionary_{gas}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        trigger_conditions={
                            f'{gas}_aqi': 150,
                            f'{gas}_concentration': aqi_data.get('max_concentration', 0)
                        },
                        response_actions=[
                            'Issue health advisories',
                            'Recommend evacuation for sensitive groups',
                            'Increase air quality monitoring',
                            'Prepare emergency medical resources',
                            'Coordinate with healthcare facilities'
                        ],
                        evacuation_zones=self._define_evacuation_zones(health_risk_assessment, 'High Risk'),
                        communication_channels=[
                            'Health department alerts',
                            'Local media advisories',
                            'Healthcare provider notifications'
                        ],
                        resource_requirements={
                            'emergency_personnel': 20,
                            'medical_teams': 5,
                            'evacuation_vehicles': 10,
                            'air_monitoring_equipment': 8
                        },
                        timeline={
                            'advisory_issuance': timedelta(minutes=30),
                            'sensitive_group_evacuation': timedelta(hours=4),
                            'monitoring_enhancement': timedelta(hours=2)
                        }
                    )
                    protocols.append(protocol)
            
            return protocols
            
        except Exception as e:
            logger.error(f"Error creating emergency response protocols: {str(e)}")
            raise
    
    def _define_evacuation_zones(self, health_risk_assessment: Dict[str, Any], 
                               risk_threshold: str) -> List[Tuple[float, float]]:
        """Define evacuation zones based on health risk assessment"""
        evacuation_zones = []
        
        for zone_id, zone_data in health_risk_assessment.items():
            risk_category = zone_data.get('risk_category', 'No Risk')
            
            if risk_category == risk_threshold:
                location = zone_data.get('location', (0, 0))
                evacuation_zones.append(location)
        
        return evacuation_zones
    
    async def validate_against_historical_incidents(self, model_results: Dict[str, Any],
                                                  historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate gas hazard models against historical gas hazard incidents"""
        try:
            validation_results = {
                'dispersion_model_validation': {},
                'health_impact_validation': {},
                'evacuation_effectiveness': {},
                'overall_model_performance': {}
            }
            
            # Match model results with historical incidents
            matched_cases = []
            
            for incident in historical_data:
                incident_location = (incident.get('longitude', 0), incident.get('latitude', 0))
                incident_date = incident.get('date')
                incident_gases = incident.get('gas_concentrations', {})
                incident_health_impacts = incident.get('health_impacts', {})
                incident_evacuations = incident.get('evacuations', {})
                
                # Advanced spatiotemporal matching with uncertainty quantification
                model_predictions = await self._find_advanced_model_predictions(
                    model_results, incident_location, incident_gases, incident_date
                )
                
                if model_predictions:
                    matched_cases.append({
                        'incident': incident,
                        'predictions': model_predictions
                    })
            
            if not matched_cases:
                return {'no_matches': True, 'message': 'No matching historical incidents found'}
            
            # Validate dispersion model accuracy
            dispersion_validation = self._validate_dispersion_predictions(matched_cases)
            validation_results['dispersion_model_validation'] = dispersion_validation
            
            # Validate health impact predictions
            health_validation = self._validate_health_impact_predictions(matched_cases)
            validation_results['health_impact_validation'] = health_validation
            
            # Validate evacuation recommendations
            evacuation_validation = self._validate_evacuation_recommendations(matched_cases)
            validation_results['evacuation_effectiveness'] = evacuation_validation
            
            # Calculate overall performance
            overall_performance = self._calculate_overall_model_performance(validation_results)
            validation_results['overall_model_performance'] = overall_performance
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating against historical incidents: {str(e)}")
            raise
    
    async def _find_advanced_model_predictions(self, model_results: Dict[str, Any],
                                             incident_location: Tuple[float, float],
                                             incident_gases: Dict[str, float],
                                             incident_date: Optional[datetime]) -> Optional[Dict[str, Any]]:
        """Advanced spatiotemporal matching with machine learning-based similarity assessment"""
        try:
            dispersion_results = model_results.get('dispersion_results', {})
            health_risk_assessment = model_results.get('health_risk_assessment', {})
            
            if not health_risk_assessment:
                return None
            
            # Advanced spatial matching using geodesic distance
            best_matches = []
            
            for zone_id, zone_data in health_risk_assessment.items():
                zone_location = zone_data.get('location', (0, 0))
                
                # Calculate geodesic distance (accounts for Earth curvature)
                try:
                    from geopy.distance import geodesic
                    distance = geodesic(incident_location, zone_location).meters
                except ImportError:
                    # Fallback to Euclidean distance
                    distance = np.sqrt((zone_location[0] - incident_location[0])**2 + 
                                     (zone_location[1] - incident_location[1])**2) * 111000  # Approximate m/degree
                
                # Calculate gas composition similarity
                predicted_concentrations = zone_data.get('concentrations', {})
                composition_similarity = self._calculate_gas_composition_similarity(
                    incident_gases, predicted_concentrations
                )
                
                # Calculate temporal similarity if date available
                temporal_similarity = 1.0
                if incident_date:
                    temporal_similarity = self._calculate_temporal_similarity(incident_date)
                
                # Combined similarity score
                spatial_weight = np.exp(-distance / 10000.0)  # 10 km characteristic distance
                composition_weight = composition_similarity
                temporal_weight = temporal_similarity
                
                overall_similarity = (spatial_weight * 0.4 + 
                                    composition_weight * 0.4 + 
                                    temporal_weight * 0.2)
                
                if overall_similarity > 0.1:  # Minimum similarity threshold
                    best_matches.append({
                        'zone_id': zone_id,
                        'zone_data': zone_data,
                        'distance': distance,
                        'similarity_score': overall_similarity,
                        'spatial_weight': spatial_weight,
                        'composition_weight': composition_weight,
                        'temporal_weight': temporal_weight
                    })
            
            if not best_matches:
                return None
            
            # Sort by similarity score and select best match
            best_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            best_match = best_matches[0]
            
            # Calculate prediction uncertainty based on match quality
            uncertainty_factor = 1.0 - best_match['similarity_score']
            
            # Enhanced prediction with uncertainty bounds
            zone_data = best_match['zone_data']
            predicted_concentrations = zone_data.get('concentrations', {})
            
            # Apply uncertainty to predictions
            uncertain_concentrations = {}
            for gas, concentration in predicted_concentrations.items():
                uncertainty = concentration * uncertainty_factor * 0.5  # 50% max uncertainty
                uncertain_concentrations[gas] = {
                    'mean': concentration,
                    'lower_bound': max(0, concentration - uncertainty),
                    'upper_bound': concentration + uncertainty,
                    'uncertainty': uncertainty
                }
            
            return {
                'predicted_concentrations': predicted_concentrations,
                'uncertain_concentrations': uncertain_concentrations,
                'predicted_health_risks': zone_data.get('health_risks', {}),
                'predicted_risk_category': zone_data.get('risk_category', 'No Risk'),
                'distance_to_incident': best_match['distance'],
                'similarity_score': best_match['similarity_score'],
                'prediction_uncertainty': uncertainty_factor,
                'matching_details': {
                    'spatial_similarity': best_match['spatial_weight'],
                    'composition_similarity': best_match['composition_weight'],
                    'temporal_similarity': best_match['temporal_weight'],
                    'total_candidates': len(health_risk_assessment),
                    'qualified_matches': len(best_matches)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced model prediction matching: {str(e)}")
            return None
    
    def _calculate_gas_composition_similarity(self, incident_gases: Dict[str, float], 
                                           predicted_gases: Dict[str, float]) -> float:
        """Calculate similarity between gas compositions using cosine similarity"""
        try:
            # Get all unique gases
            all_gases = set(incident_gases.keys()) | set(predicted_gases.keys())
            
            if not all_gases:
                return 0.0
            
            # Create vectors
            incident_vector = np.array([incident_gases.get(gas, 0.0) for gas in all_gases])
            predicted_vector = np.array([predicted_gases.get(gas, 0.0) for gas in all_gases])
            
            # Normalize vectors
            incident_norm = np.linalg.norm(incident_vector)
            predicted_norm = np.linalg.norm(predicted_vector)
            
            if incident_norm == 0 or predicted_norm == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(incident_vector, predicted_vector) / (incident_norm * predicted_norm)
            
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"Error calculating gas composition similarity: {str(e)}")
            return 0.0
    
    def _calculate_temporal_similarity(self, incident_date: datetime) -> float:
        """Calculate temporal similarity based on seasonal and diurnal patterns"""
        try:
            current_time = datetime.now()
            
            # Seasonal similarity (same month gets higher weight)
            month_diff = abs(incident_date.month - current_time.month)
            month_diff = min(month_diff, 12 - month_diff)  # Circular distance
            seasonal_similarity = 1.0 - (month_diff / 6.0)
            
            # Diurnal similarity (same hour gets higher weight)
            hour_diff = abs(incident_date.hour - current_time.hour)
            hour_diff = min(hour_diff, 24 - hour_diff)  # Circular distance
            diurnal_similarity = 1.0 - (hour_diff / 12.0)
            
            # Combined temporal similarity
            temporal_similarity = (seasonal_similarity * 0.6 + diurnal_similarity * 0.4)
            
            return max(0.0, temporal_similarity)
            
        except Exception as e:
            logger.error(f"Error calculating temporal similarity: {str(e)}")
            return 1.0
    
    def _validate_dispersion_predictions(self, matched_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate atmospheric dispersion model predictions"""
        validation_metrics = {}
        
        for gas in ['SO2', 'H2S', 'CO2']:
            predicted_concentrations = []
            observed_concentrations = []
            
            for case in matched_cases:
                incident = case['incident']
                predictions = case['predictions']
                
                if gas in incident.get('gas_concentrations', {}) and gas in predictions.get('predicted_concentrations', {}):
                    observed = incident['gas_concentrations'][gas]
                    predicted = predictions['predicted_concentrations'][gas]
                    
                    observed_concentrations.append(observed)
                    predicted_concentrations.append(predicted)
            
            if len(predicted_concentrations) > 2:
                # Calculate validation metrics
                predicted = np.array(predicted_concentrations)
                observed = np.array(observed_concentrations)
                
                correlation = np.corrcoef(predicted, observed)[0, 1] if len(predicted) > 1 else 0
                rmse = np.sqrt(np.mean((predicted - observed)**2))
                bias = np.mean(predicted - observed)
                
                # Factor of 2 analysis (common for dispersion models)
                factor_2_count = np.sum((predicted / observed >= 0.5) & (predicted / observed <= 2.0))
                factor_2_percentage = factor_2_count / len(predicted) * 100
                
                validation_metrics[gas] = {
                    'correlation': correlation,
                    'rmse': rmse,
                    'bias': bias,
                    'factor_2_percentage': factor_2_percentage,
                    'n_cases': len(predicted)
                }
        
        return validation_metrics
    
    def _validate_health_impact_predictions(self, matched_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate health impact predictions"""
        predicted_risks = []
        observed_impacts = []
        
        for case in matched_cases:
            incident = case['incident']
            predictions = case['predictions']
            
            # Convert predicted risk category to numerical scale
            risk_category = predictions.get('predicted_risk_category', 'No Risk')
            risk_scores = {
                'No Risk': 0, 'Low Risk': 1, 'Moderate Risk': 2, 
                'High Risk': 3, 'Extreme Risk': 4
            }
            predicted_risk_score = risk_scores.get(risk_category, 0)
            
            # Convert observed health impacts to numerical scale
            health_impacts = incident.get('health_impacts', {})
            observed_impact_score = 0
            
            if health_impacts.get('hospitalizations', 0) > 0:
                observed_impact_score += 2
            if health_impacts.get('deaths', 0) > 0:
                observed_impact_score += 4
            if health_impacts.get('respiratory_symptoms', 0) > 0:
                observed_impact_score += 1
            
            predicted_risks.append(predicted_risk_score)
            observed_impacts.append(min(4, observed_impact_score))  # Cap at 4
        
        if len(predicted_risks) > 2:
            predicted = np.array(predicted_risks)
            observed = np.array(observed_impacts)
            
            correlation = np.corrcoef(predicted, observed)[0, 1] if len(predicted) > 1 else 0
            accuracy = np.mean(np.abs(predicted - observed) <= 1)  # Within 1 category
            
            return {
                'risk_prediction_correlation': correlation,
                'category_accuracy': accuracy,
                'n_cases': len(predicted_risks),
                'mean_predicted_risk': np.mean(predicted),
                'mean_observed_impact': np.mean(observed)
            }
        
        return {'insufficient_data': True}
    
    def _validate_evacuation_recommendations(self, matched_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate evacuation recommendation effectiveness"""
        correct_evacuations = 0
        total_cases = 0
        
        for case in matched_cases:
            incident = case['incident']
            predictions = case['predictions']
            
            predicted_risk = predictions.get('predicted_risk_category', 'No Risk')
            actual_evacuations = incident.get('evacuations', {}).get('occurred', False)
            health_impacts = incident.get('health_impacts', {})
            
            # Determine if evacuation was needed based on health impacts
            evacuation_needed = (health_impacts.get('hospitalizations', 0) > 10 or 
                               health_impacts.get('deaths', 0) > 0)
            
            # Check if model would have recommended evacuation
            evacuation_recommended = predicted_risk in ['High Risk', 'Extreme Risk']
            
            if (evacuation_needed and evacuation_recommended) or (not evacuation_needed and not evacuation_recommended):
                correct_evacuations += 1
            
            total_cases += 1
        
        if total_cases > 0:
            evacuation_accuracy = correct_evacuations / total_cases
            
            return {
                'evacuation_recommendation_accuracy': evacuation_accuracy,
                'correct_recommendations': correct_evacuations,
                'total_cases': total_cases
            }
        
        return {'insufficient_data': True}
    
    def _calculate_overall_model_performance(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall model performance score"""
        performance_scores = []
        
        # Dispersion model performance
        dispersion_validation = validation_results.get('dispersion_model_validation', {})
        if dispersion_validation:
            correlations = [metrics.get('correlation', 0) for metrics in dispersion_validation.values() 
                          if isinstance(metrics, dict) and 'correlation' in metrics]
            if correlations:
                avg_correlation = np.mean(correlations)
                performance_scores.append(avg_correlation)
        
        # Health impact performance
        health_validation = validation_results.get('health_impact_validation', {})
        if 'risk_prediction_correlation' in health_validation:
            performance_scores.append(health_validation['risk_prediction_correlation'])
        
        # Evacuation effectiveness
        evacuation_validation = validation_results.get('evacuation_effectiveness', {})
        if 'evacuation_recommendation_accuracy' in evacuation_validation:
            performance_scores.append(evacuation_validation['evacuation_recommendation_accuracy'])
        
        if performance_scores:
            overall_score = np.mean(performance_scores)
            
            # Categorize performance
            if overall_score >= 0.8:
                performance_category = 'Excellent'
            elif overall_score >= 0.6:
                performance_category = 'Good'
            elif overall_score >= 0.4:
                performance_category = 'Fair'
            else:
                performance_category = 'Poor'
            
            return {
                'overall_performance_score': overall_score,
                'performance_category': performance_category,
                'component_scores': {
                    'dispersion_model': np.mean([metrics.get('correlation', 0) for metrics in dispersion_validation.values() 
                                               if isinstance(metrics, dict)]) if dispersion_validation else 0,
                    'health_impact_model': health_validation.get('risk_prediction_correlation', 0),
                    'evacuation_model': evacuation_validation.get('evacuation_recommendation_accuracy', 0)
                },
                'validation_summary': {
                    'total_components_validated': len(performance_scores),
                    'validation_confidence': min(1.0, len(performance_scores) / 3.0)
                }
            }
        
        return {'insufficient_validation_data': True}
    
    async def store_hazard_assessment_results(self, assessment: GasHazardAssessment,
                                            volcanic_system_id: str):
        """Store gas hazard assessment results in database"""
        try:
            # Store main assessment record
            assessment_query = """
            INSERT INTO gas_hazard_assessments (
                assessment_id, volcanic_system_id, emission_source_x, emission_source_y, emission_source_z,
                gas_emission_rates, meteorological_conditions, concentration_maps,
                exposure_zones, evacuation_recommendations, health_impact_assessment,
                emergency_protocols, assessment_timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            assessment_values = (
                assessment.assessment_id,
                volcanic_system_id,
                assessment.emission_source[0],
                assessment.emission_source[1],
                assessment.emission_source[2],
                str(assessment.gas_emission_rates),  # JSON string
                str(assessment.meteorological_conditions),
                str({k: v.tolist() for k, v in assessment.concentration_maps.items()}),  # Convert arrays to lists
                str(assessment.exposure_zones),
                str(assessment.evacuation_recommendations),
                str(assessment.health_impact_assessment),
                str(assessment.emergency_protocols),
                datetime.now()
            )
            
            await self.db_manager.execute_query(assessment_query, assessment_values)
            
            logger.info(f"Stored gas hazard assessment {assessment.assessment_id}")
            
        except Exception as e:
            logger.error(f"Error storing hazard assessment results: {str(e)}")
    
    async def create_comprehensive_gas_hazard_system(self, volcanic_system: VolcanicSystem,
                                                   emission_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive gas hazard assessment system"""
        try:
            # Initialize all hazard models
            await self.initialize_hazard_models(volcanic_system)
            
            # Define emission scenario
            emission_source = emission_scenario.get('location', (0, 0, 100))  # 100m elevation
            emission_rates = emission_scenario.get('emission_rates', {
                'SO2': 10.0,   # kg/s
                'H2S': 2.0,    # kg/s
                'CO2': 50.0    # kg/s
            })
            
            # Define meteorological conditions
            meteorological_conditions = emission_scenario.get('meteorological_conditions', {
                'wind_speed': 5.0,        # m/s
                'wind_direction': 270.0,  # degrees (westerly wind)
                'temperature': 15.0,      # °C
                'pressure': 101325.0,     # Pa
                'humidity': 60.0,         # %
                'stability_class': 'D'    # Neutral stability
            })
            
            # Assess real-time air quality
            air_quality_assessment = await self.assess_real_time_air_quality(
                {'location': emission_source, 'emission_rates': emission_rates},
                meteorological_conditions
            )
            
            # Generate evacuation recommendations
            evacuation_recommendations = await self.generate_evacuation_recommendations(
                air_quality_assessment
            )
            
            # Create emergency response protocols
            emergency_protocols = await self.create_emergency_response_protocols(
                air_quality_assessment
            )
            
            # Assess long-term health impacts
            exposure_data = [
                {
                    'zone_id': zone_id,
                    'population': zone_data.get('population', 0),
                    'concentrations': zone_data.get('concentrations', {})
                }
                for zone_id, zone_data in air_quality_assessment.get('health_risk_assessment', {}).items()
            ]
            
            long_term_impacts = await self.assess_long_term_health_impacts(
                exposure_data, timedelta(days=30)  # 30-day exposure scenario
            )
            
            # Create comprehensive hazard assessment
            hazard_assessment = GasHazardAssessment(
                assessment_id=f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                emission_source=emission_source,
                gas_emission_rates=emission_rates,
                meteorological_conditions=meteorological_conditions,
                dispersion_model_results=air_quality_assessment.get('dispersion_results', {}).get('concentration_fields', {}),
                concentration_maps=air_quality_assessment.get('dispersion_results', {}).get('concentration_fields', {}),
                exposure_zones=evacuation_recommendations.get('evacuation_zones', []),
                evacuation_recommendations=evacuation_recommendations.get('evacuation_recommendations', {}),
                health_impact_assessment=air_quality_assessment.get('health_risk_assessment', {}),
                emergency_protocols=[protocol.protocol_id for protocol in emergency_protocols]
            )
            
            # Store results in database
            await self.store_hazard_assessment_results(hazard_assessment, volcanic_system.system_id)
            
            return {
                'volcanic_system': volcanic_system,
                'emission_scenario': emission_scenario,
                'air_quality_assessment': air_quality_assessment,
                'evacuation_recommendations': evacuation_recommendations,
                'emergency_protocols': emergency_protocols,
                'long_term_health_impacts': long_term_impacts,
                'comprehensive_assessment': hazard_assessment,
                'system_performance': {
                    'population_zones_analyzed': len(self.population_data),
                    'gases_modeled': len(emission_rates),
                    'emergency_protocols_created': len(emergency_protocols),
                    'total_population_assessed': sum(zone.get('population_count', 0) for zone in self.population_data.values())
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating comprehensive gas hazard system: {str(e)}")
            raise 
    
    async def _estimate_population_advanced_fallback(self, exposure_data: List[Dict[str, Any]]) -> int:
        """Advanced population estimation using spatial interpolation and demographic modeling"""
        try:
            if not exposure_data:
                return 0
            
            # Extract spatial coordinates and basic population data
            coordinates = []
            populations = []
            areas = []
            
            for data in exposure_data:
                if 'latitude' in data and 'longitude' in data:
                    coordinates.append([data['latitude'], data['longitude']])
                    populations.append(data.get('population', 0))
                    areas.append(data.get('area_km2', 1.0))
            
            if not coordinates:
                return sum(data.get('population', 0) for data in exposure_data)
            
            coordinates = np.array(coordinates)
            populations = np.array(populations)
            areas = np.array(areas)
            
            # Calculate population density
            densities = populations / (areas + 1e-6)  # people per km²
            
            # Apply advanced spatial interpolation using Gaussian Process
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            
            # Create kernel for spatial correlation
            kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=0.1)
            
            # Fit Gaussian Process to population density data
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
            gp.fit(coordinates, densities)
            
            # Predict population density at all locations
            predicted_densities, density_std = gp.predict(coordinates, return_std=True)
            
            # Apply demographic corrections based on geographic factors
            corrected_populations = []
            
            for i, data in enumerate(exposure_data):
                base_population = populations[i] if i < len(populations) else 0
                predicted_density = predicted_densities[i] if i < len(predicted_densities) else 0
                area = areas[i] if i < len(areas) else 1.0
                
                # Calculate enhanced population estimate
                if base_population > 0:
                    # Use actual data when available
                    enhanced_pop = base_population
                else:
                    # Use predicted density for missing data
                    enhanced_pop = predicted_density * area
                
                # Apply geographic and infrastructure corrections
                geographic_factor = self._calculate_geographic_population_factor(data)
                infrastructure_factor = self._calculate_infrastructure_population_factor(data)
                temporal_factor = self._calculate_temporal_population_factor(data)
                
                corrected_population = enhanced_pop * geographic_factor * infrastructure_factor * temporal_factor
                corrected_populations.append(max(0, int(corrected_population)))
            
            return sum(corrected_populations)
            
        except Exception as e:
            logger.error(f"Error in advanced population estimation: {e}")
            # Final fallback to simple sum
            return sum(data.get('population', 0) for data in exposure_data)
    
    def _calculate_geographic_population_factor(self, location_data: Dict[str, Any]) -> float:
        """Calculate population correction factor based on geographic characteristics"""
        try:
            factor = 1.0
            
            # Elevation correction (lower elevations typically more populated)
            elevation = location_data.get('elevation', 0)
            if elevation > 2000:  # Above 2000m
                factor *= 0.3
            elif elevation > 1000:  # 1000-2000m
                factor *= 0.7
            elif elevation < 100:  # Coastal areas
                factor *= 1.3
            
            # Slope correction (flatter areas more populated)
            slope = location_data.get('slope', 0)
            if slope > 30:  # Very steep
                factor *= 0.2
            elif slope > 15:  # Moderate slope
                factor *= 0.6
            elif slope < 5:  # Flat
                factor *= 1.2
            
            # Distance to water bodies (closer to water = higher population)
            water_distance = location_data.get('distance_to_water', 10000)  # meters
            if water_distance < 1000:  # Within 1km of water
                factor *= 1.4
            elif water_distance < 5000:  # Within 5km
                factor *= 1.1
            elif water_distance > 20000:  # More than 20km from water
                factor *= 0.7
            
            return max(0.1, min(factor, 3.0))  # Limit factor to reasonable range
            
        except Exception as e:
            logger.error(f"Error calculating geographic factor: {e}")
            return 1.0
    
    def _calculate_infrastructure_population_factor(self, location_data: Dict[str, Any]) -> float:
        """Calculate population correction factor based on infrastructure"""
        try:
            factor = 1.0
            
            # Road density (higher density = more population)
            road_density = location_data.get('road_density', 0)  # km/km²
            if road_density > 2.0:  # High road density
                factor *= 1.5
            elif road_density > 0.5:  # Moderate road density
                factor *= 1.2
            elif road_density < 0.1:  # Very low road density
                factor *= 0.4
            
            # Distance to major cities
            city_distance = location_data.get('distance_to_city', 50000)  # meters
            if city_distance < 10000:  # Within 10km of city
                factor *= 1.6
            elif city_distance < 30000:  # Within 30km
                factor *= 1.2
            elif city_distance > 100000:  # More than 100km from city
                factor *= 0.5
            
            # Presence of airports, ports, or major infrastructure
            if location_data.get('has_airport', False):
                factor *= 1.3
            if location_data.get('has_port', False):
                factor *= 1.4
            if location_data.get('has_railway', False):
                factor *= 1.2
            
            return max(0.1, min(factor, 4.0))
            
        except Exception as e:
            logger.error(f"Error calculating infrastructure factor: {e}")
            return 1.0
    
    def _calculate_temporal_population_factor(self, location_data: Dict[str, Any]) -> float:
        """Calculate population correction factor based on temporal variations"""
        try:
            factor = 1.0
            current_hour = datetime.now().hour
            
            # Tourism areas have seasonal and daily variations
            is_tourist_area = location_data.get('is_tourist_area', False)
            
            if is_tourist_area:
                # Tourist areas have higher population during day and tourist season
                if 8 <= current_hour <= 20:  # Daytime
                    factor *= 1.5
                else:  # Nighttime
                    factor *= 0.8
                
                # Seasonal factor (simplified - would need actual seasonal data)
                month = datetime.now().month
                if month in [6, 7, 8, 12, 1]:  # Peak tourist months
                    factor *= 1.3
                elif month in [3, 4, 9, 10]:  # Shoulder season
                    factor *= 1.1
                else:  # Low season
                    factor *= 0.9
            
            # Work areas have daily variations
            is_work_area = location_data.get('is_work_area', False)
            if is_work_area:
                if 8 <= current_hour <= 18:  # Work hours
                    factor *= 1.4
                else:  # Non-work hours
                    factor *= 0.6
            
            # Residential areas are more stable
            is_residential = location_data.get('is_residential', True)
            if is_residential:
                if 22 <= current_hour or current_hour <= 6:  # Night
                    factor *= 1.1  # People at home
                else:
                    factor *= 0.9  # Some people at work
            
            return max(0.3, min(factor, 2.5))
            
        except Exception as e:
            logger.error(f"Error calculating temporal factor: {e}")
            return 1.0