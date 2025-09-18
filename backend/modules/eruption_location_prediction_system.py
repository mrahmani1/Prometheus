"""
Advanced Eruption Location Prediction System for Sill Complex Analysis

This module implements sophisticated eruption location prediction capabilities for stacked-sill
complexes, incorporating stress field analysis, failure criteria, topographic controls, and
probabilistic mapping with temporal evolution modeling.

Key Features:
- Stress field analysis around sill complex geometries
- Failure criterion analysis for eruption pathway development
- Topographic and structural control analysis
- Probabilistic eruption location mapping
- Temporal evolution of eruption potential
- Eruption style prediction based on sill characteristics
- Hazard zone mapping for sill-related volcanic activity
- Validation against historical sill-related eruptions

Author: Prometheus AI Platform
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from scipy.spatial import distance_matrix, Voronoi
from scipy.interpolate import griddata, RBFInterpolator
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
from shapely.geometry import Point, Polygon as ShapelyPolygon, LineString
from shapely.ops import unary_union
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import xarray as xr

# Database imports
import asyncpg
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

logger = logging.getLogger(__name__)

@dataclass
class StressFieldParameters:
    """Parameters for stress field analysis around sill complexes"""
    principal_stress_1: float  # Maximum principal stress (Pa)
    principal_stress_2: float  # Intermediate principal stress (Pa)
    principal_stress_3: float  # Minimum principal stress (Pa)
    stress_orientation: Tuple[float, float, float]  # Azimuth, plunge, rake
    differential_stress: float  # σ1 - σ3 (Pa)
    mean_stress: float  # (σ1 + σ2 + σ3) / 3 (Pa)
    stress_ratio: float  # (σ2 - σ3) / (σ1 - σ3)
    deviatoric_stress: np.ndarray  # Deviatoric stress tensor
    stress_invariants: Tuple[float, float, float]  # I1, J2, J3
    von_mises_stress: float  # Equivalent stress (Pa)

@dataclass
class FailureCriterion:
    """Failure criteria for eruption pathway development"""
    mohr_coulomb_cohesion: float  # Cohesive strength (Pa)
    mohr_coulomb_friction: float  # Internal friction angle (radians)
    tensile_strength: float  # Tensile strength (Pa)
    griffith_criterion: float  # Griffith failure criterion
    hoek_brown_mi: float  # Hoek-Brown material constant
    hoek_brown_gsi: float  # Geological Strength Index
    drucker_prager_alpha: float  # Drucker-Prager parameter
    drucker_prager_k: float  # Drucker-Prager parameter
    failure_probability: float  # Probability of failure [0-1]
    critical_stress_ratio: float  # Critical stress ratio for failure

@dataclass
class TopographicControls:
    """Topographic and structural controls on eruption locations"""
    elevation: float  # Surface elevation (m)
    slope: float  # Surface slope (degrees)
    aspect: float  # Surface aspect (degrees)
    curvature: float  # Surface curvature (1/m)
    roughness: float  # Surface roughness index
    drainage_density: float  # Drainage density (km/km²)
    fault_density: float  # Fault density (km/km²)
    fracture_density: float  # Fracture density (km/km²)
    structural_lineaments: List[Tuple[float, float]]  # Lineament orientations
    topographic_stress: np.ndarray  # Topography-induced stress

@dataclass
class SillGeometry:
    """Detailed sill geometry parameters"""
    sill_id: str
    center_coordinates: Tuple[float, float, float]  # x, y, depth (m)
    length: float  # Sill length (m)
    width: float  # Sill width (m)
    thickness: float  # Sill thickness (m)
    dip: float  # Sill dip angle (degrees)
    strike: float  # Sill strike angle (degrees)
    volume: float  # Sill volume (m³)
    aspect_ratio: float  # Length/width ratio
    shape_factor: float  # Geometric shape factor
    connectivity_index: float  # Connection to other sills [0-1]

@dataclass
class EruptionPrediction:
    """Eruption location prediction results"""
    location: Tuple[float, float]  # Predicted eruption coordinates
    probability: float  # Eruption probability [0-1]
    confidence_interval: Tuple[float, float]  # 95% confidence bounds
    eruption_style: str  # Predicted eruption style
    magnitude_estimate: float  # Estimated eruption magnitude
    time_to_eruption: Optional[float]  # Estimated time to eruption (days)
    uncertainty_ellipse: Tuple[float, float, float]  # Semi-major, semi-minor, orientation
    hazard_radius: float  # Estimated hazard radius (m)
    risk_level: str  # Risk level classification
    validation_score: float  # Model validation score [0-1]

@dataclass
class HazardZone:
    """Hazard zone mapping for sill-related volcanic activity"""
    zone_id: str
    geometry: ShapelyPolygon  # Hazard zone boundary
    hazard_level: str  # Low, Moderate, High, Extreme
    hazard_type: str  # Lava flow, pyroclastic flow, ash fall, etc.
    probability: float  # Hazard occurrence probability [0-1]
    impact_radius: float  # Maximum impact radius (m)
    population_at_risk: int  # Estimated population at risk
    infrastructure_at_risk: List[str]  # Critical infrastructure
    evacuation_time: float  # Required evacuation time (hours)
    monitoring_priority: int  # Monitoring priority level [1-5]

class EruptionLocationPredictionSystem:
    """
    Advanced system for predicting eruption locations based on sill complex analysis
    """
    
    def __init__(self, database_config: Dict[str, Any]):
        self.db_config = database_config
        self.engine = None
        self.redis_client = None
        self.stress_field_cache = {}
        self.failure_models = {}
        self.topographic_data = {}
        self.historical_eruptions = []
        self.prediction_models = {}
        
    async def initialize(self):
        """Initialize database connections and load models"""
        try:
            # Initialize PostgreSQL connection
            self.engine = create_async_engine(
                f"postgresql+asyncpg://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=self.db_config['redis_host'],
                port=self.db_config['redis_port'],
                decode_responses=True
            )
            
            # Load historical eruption data
            await self._load_historical_eruptions()
            
            # Initialize prediction models
            await self._initialize_prediction_models()
            
            logger.info("Eruption Location Prediction System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise

    async def analyze_stress_field_around_sill_complex(
        self, 
        sill_geometries: List[SillGeometry],
        regional_stress: StressFieldParameters,
        analysis_grid: Tuple[int, int, int] = (100, 100, 50)
    ) -> Dict[str, Any]:
        """
        Analyze stress field around sill complex geometries using advanced numerical methods
        """
        try:
            # Create 3D analysis grid
            x_grid = np.linspace(-10000, 10000, analysis_grid[0])  # 20km x 20km area
            y_grid = np.linspace(-10000, 10000, analysis_grid[1])
            z_grid = np.linspace(-5000, 0, analysis_grid[2])  # Surface to 5km depth
            
            X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
            
            # Initialize stress field arrays
            stress_xx = np.zeros_like(X)
            stress_yy = np.zeros_like(X)
            stress_zz = np.zeros_like(X)
            stress_xy = np.zeros_like(X)
            stress_xz = np.zeros_like(X)
            stress_yz = np.zeros_like(X)
            
            # Apply regional stress field
            stress_xx += regional_stress.principal_stress_1
            stress_yy += regional_stress.principal_stress_2
            stress_zz += regional_stress.principal_stress_3
            
            # Calculate stress perturbations from each sill
            for sill in sill_geometries:
                sill_stress = await self._calculate_sill_induced_stress(
                    sill, X, Y, Z
                )
                
                stress_xx += sill_stress['xx']
                stress_yy += sill_stress['yy']
                stress_zz += sill_stress['zz']
                stress_xy += sill_stress['xy']
                stress_xz += sill_stress['xz']
                stress_yz += sill_stress['yz']
            
            # Calculate derived stress parameters
            principal_stresses = await self._calculate_principal_stresses(
                stress_xx, stress_yy, stress_zz, stress_xy, stress_xz, stress_yz
            )
            
            von_mises_stress = await self._calculate_von_mises_stress(
                stress_xx, stress_yy, stress_zz, stress_xy, stress_xz, stress_yz
            )
            
            # Calculate stress gradients
            stress_gradients = await self._calculate_stress_gradients(
                principal_stresses, x_grid, y_grid, z_grid
            )
            
            stress_field_results = {
                'grid_coordinates': {'x': X, 'y': Y, 'z': Z},
                'stress_tensor': {
                    'xx': stress_xx, 'yy': stress_yy, 'zz': stress_zz,
                    'xy': stress_xy, 'xz': stress_xz, 'yz': stress_yz
                },
                'principal_stresses': principal_stresses,
                'von_mises_stress': von_mises_stress,
                'stress_gradients': stress_gradients,
                'differential_stress': principal_stresses['sigma1'] - principal_stresses['sigma3'],
                'mean_stress': (principal_stresses['sigma1'] + 
                               principal_stresses['sigma2'] + 
                               principal_stresses['sigma3']) / 3,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache results
            cache_key = f"stress_field_{hash(str(sill_geometries))}"
            await self.redis_client.setex(
                cache_key, 3600, str(stress_field_results)
            )
            
            # Store in database
            await self._store_stress_field_analysis(stress_field_results)
            
            logger.info(f"Completed stress field analysis for {len(sill_geometries)} sills")
            return stress_field_results
            
        except Exception as e:
            logger.error(f"Error in stress field analysis: {e}")
            raise

    async def _calculate_sill_induced_stress(
        self, 
        sill: SillGeometry, 
        X: np.ndarray, 
        Y: np.ndarray, 
        Z: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate stress field induced by individual sill intrusion"""
        
        # Sill center coordinates
        x0, y0, z0 = sill.center_coordinates
        
        # Distance from sill center
        r = np.sqrt((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2)
        
        # Avoid division by zero
        r = np.maximum(r, 1.0)
        
        # Sill pressure (estimated from magma overpressure)
        sill_pressure = 50e6  # 50 MPa typical overpressure
        
        # Geometric factors based on sill dimensions
        length_factor = sill.length / 1000.0  # Normalize to km
        width_factor = sill.width / 1000.0
        thickness_factor = sill.thickness / 100.0  # Normalize to 100m
        
        # Stress decay with distance (modified Boussinesq solution)
        decay_factor = (length_factor * width_factor * thickness_factor) / (r**2 + 1)
        
        # Directional factors based on sill orientation
        strike_rad = np.radians(sill.strike)
        dip_rad = np.radians(sill.dip)
        
        # Calculate stress components
        stress_xx = sill_pressure * decay_factor * np.cos(strike_rad)**2
        stress_yy = sill_pressure * decay_factor * np.sin(strike_rad)**2
        stress_zz = sill_pressure * decay_factor * np.sin(dip_rad)**2
        stress_xy = sill_pressure * decay_factor * np.cos(strike_rad) * np.sin(strike_rad)
        stress_xz = sill_pressure * decay_factor * np.cos(strike_rad) * np.sin(dip_rad)
        stress_yz = sill_pressure * decay_factor * np.sin(strike_rad) * np.sin(dip_rad)
        
        return {
            'xx': stress_xx, 'yy': stress_yy, 'zz': stress_zz,
            'xy': stress_xy, 'xz': stress_xz, 'yz': stress_yz
        }

    async def analyze_failure_criteria(
        self, 
        stress_field: Dict[str, Any],
        rock_properties: Dict[str, float],
        analysis_depth: float = 2000.0
    ) -> Dict[str, Any]:
        """
        Implement failure criterion analysis for eruption pathway development
        """
        try:
            # Extract stress field data
            stress_tensor = stress_field['stress_tensor']
            principal_stresses = stress_field['principal_stresses']
            
            # Rock property parameters
            cohesion = rock_properties.get('cohesion', 10e6)  # 10 MPa
            friction_angle = rock_properties.get('friction_angle', np.radians(30))  # 30 degrees
            tensile_strength = rock_properties.get('tensile_strength', 5e6)  # 5 MPa
            ucs = rock_properties.get('ucs', 100e6)  # 100 MPa unconfined compressive strength
            
            # Initialize failure analysis arrays
            grid_shape = stress_tensor['xx'].shape
            
            # Mohr-Coulomb failure criterion
            mohr_coulomb_failure = await self._calculate_mohr_coulomb_failure(
                principal_stresses, cohesion, friction_angle
            )
            
            # Griffith failure criterion (for tensile failure)
            griffith_failure = await self._calculate_griffith_failure(
                principal_stresses, tensile_strength
            )
            
            # Hoek-Brown failure criterion
            hoek_brown_failure = await self._calculate_hoek_brown_failure(
                principal_stresses, ucs, rock_properties
            )
            
            # Drucker-Prager failure criterion
            drucker_prager_failure = await self._calculate_drucker_prager_failure(
                stress_tensor, cohesion, friction_angle
            )
            
            # Combined failure probability
            failure_probability = np.maximum.reduce([
                mohr_coulomb_failure,
                griffith_failure,
                hoek_brown_failure,
                drucker_prager_failure
            ])
            
            # Calculate failure gradient (pathway development potential)
            failure_gradient = await self._calculate_failure_gradient(
                failure_probability, stress_field['grid_coordinates']
            )
            
            # Identify potential eruption pathways
            eruption_pathways = await self._identify_eruption_pathways(
                failure_probability, failure_gradient, analysis_depth
            )
            
            failure_analysis = {
                'mohr_coulomb_failure': mohr_coulomb_failure,
                'griffith_failure': griffith_failure,
                'hoek_brown_failure': hoek_brown_failure,
                'drucker_prager_failure': drucker_prager_failure,
                'combined_failure_probability': failure_probability,
                'failure_gradient': failure_gradient,
                'eruption_pathways': eruption_pathways,
                'critical_stress_locations': await self._find_critical_stress_locations(
                    failure_probability, stress_field['grid_coordinates']
                ),
                'analysis_parameters': {
                    'cohesion': cohesion,
                    'friction_angle': friction_angle,
                    'tensile_strength': tensile_strength,
                    'analysis_depth': analysis_depth
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Store results in database
            await self._store_failure_analysis(failure_analysis)
            
            logger.info("Completed failure criterion analysis")
            return failure_analysis
            
        except Exception as e:
            logger.error(f"Error in failure criterion analysis: {e}")
            raise

    async def _calculate_mohr_coulomb_failure(
        self, 
        principal_stresses: Dict[str, np.ndarray],
        cohesion: float,
        friction_angle: float
    ) -> np.ndarray:
        """Calculate Mohr-Coulomb failure criterion"""
        
        sigma1 = principal_stresses['sigma1']
        sigma3 = principal_stresses['sigma3']
        
        # Mohr-Coulomb failure criterion: τ = c + σn * tan(φ)
        # Rearranged: (σ1 - σ3) = 2c*cos(φ)/(1-sin(φ)) + (σ1 + σ3)*sin(φ)/(1-sin(φ))
        
        sin_phi = np.sin(friction_angle)
        cos_phi = np.cos(friction_angle)
        
        # Critical stress difference
        critical_diff = (2 * cohesion * cos_phi) / (1 - sin_phi) + \
                       (sigma1 + sigma3) * sin_phi / (1 - sin_phi)
        
        # Actual stress difference
        actual_diff = sigma1 - sigma3
        
        # Failure probability (0 = no failure, 1 = certain failure)
        failure_prob = np.clip(actual_diff / critical_diff, 0, 1)
        
        return failure_prob

    async def _calculate_griffith_failure(
        self, 
        principal_stresses: Dict[str, np.ndarray],
        tensile_strength: float
    ) -> np.ndarray:
        """Calculate Griffith failure criterion for tensile failure"""
        
        sigma1 = principal_stresses['sigma1']
        sigma3 = principal_stresses['sigma3']
        
        # Griffith criterion for tensile failure
        # For σ3 < 0 (tension): σ3 = -T (tensile strength)
        # For compression: (σ1 - σ3)² = 8T(σ1 + σ3)
        
        failure_prob = np.zeros_like(sigma1)
        
        # Tensile failure condition
        tensile_mask = sigma3 < 0
        failure_prob[tensile_mask] = np.clip(-sigma3[tensile_mask] / tensile_strength, 0, 1)
        
        # Compressive failure condition
        compressive_mask = ~tensile_mask
        if np.any(compressive_mask):
            lhs = (sigma1[compressive_mask] - sigma3[compressive_mask])**2
            rhs = 8 * tensile_strength * (sigma1[compressive_mask] + sigma3[compressive_mask])
            failure_prob[compressive_mask] = np.clip(lhs / (rhs + 1e-10), 0, 1)
        
        return failure_prob

    async def analyze_topographic_structural_controls(
        self, 
        analysis_area: Tuple[float, float, float, float],  # min_x, min_y, max_x, max_y
        dem_resolution: float = 30.0  # DEM resolution in meters
    ) -> Dict[str, Any]:
        """
        Analyze topographic and structural controls on eruption locations
        """
        try:
            # Load digital elevation model
            dem_data = await self._load_dem_data(analysis_area, dem_resolution)
            
            # Calculate topographic derivatives
            elevation = dem_data['elevation']
            slope = await self._calculate_slope(elevation, dem_resolution)
            aspect = await self._calculate_aspect(elevation, dem_resolution)
            curvature = await self._calculate_curvature(elevation, dem_resolution)
            roughness = await self._calculate_roughness(elevation, dem_resolution)
            
            # Load structural data
            structural_data = await self._load_structural_data(analysis_area)
            
            # Calculate structural controls
            fault_density = await self._calculate_fault_density(
                structural_data['faults'], analysis_area, dem_resolution
            )
            fracture_density = await self._calculate_fracture_density(
                structural_data['fractures'], analysis_area, dem_resolution
            )
            lineament_density = await self._calculate_lineament_density(
                structural_data['lineaments'], analysis_area, dem_resolution
            )
            
            # Calculate drainage network
            drainage_network = await self._extract_drainage_network(elevation, dem_resolution)
            drainage_density = await self._calculate_drainage_density(
                drainage_network, analysis_area, dem_resolution
            )
            
            # Calculate topographic stress
            topographic_stress = await self._calculate_topographic_stress(
                elevation, slope, dem_resolution
            )
            
            # Identify structural lineaments and their orientations
            lineament_orientations = await self._analyze_lineament_orientations(
                structural_data['lineaments']
            )
            
            # Calculate composite structural control index
            structural_control_index = await self._calculate_structural_control_index(
                fault_density, fracture_density, lineament_density, lineament_orientations
            )
            
            # Calculate topographic favorability for eruption locations
            topographic_favorability = await self._calculate_topographic_favorability(
                elevation, slope, aspect, curvature, roughness
            )
            
            topographic_controls = {
                'dem_data': {
                    'elevation': elevation,
                    'slope': slope,
                    'aspect': aspect,
                    'curvature': curvature,
                    'roughness': roughness
                },
                'structural_data': {
                    'fault_density': fault_density,
                    'fracture_density': fracture_density,
                    'lineament_density': lineament_density,
                    'lineament_orientations': lineament_orientations
                },
                'drainage_network': {
                    'network': drainage_network,
                    'density': drainage_density
                },
                'topographic_stress': topographic_stress,
                'structural_control_index': structural_control_index,
                'topographic_favorability': topographic_favorability,
                'analysis_parameters': {
                    'analysis_area': analysis_area,
                    'dem_resolution': dem_resolution,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            # Store results in database
            await self._store_topographic_analysis(topographic_controls)
            
            logger.info("Completed topographic and structural controls analysis")
            return topographic_controls
            
        except Exception as e:
            logger.error(f"Error in topographic analysis: {e}")
            raise

    async def create_probabilistic_eruption_location_mapping(
        self,
        stress_field: Dict[str, Any],
        failure_analysis: Dict[str, Any],
        topographic_controls: Dict[str, Any],
        sill_geometries: List[SillGeometry],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Create probabilistic eruption location mapping based on integrated analysis
        """
        try:
            # Extract analysis grids
            grid_coords = stress_field['grid_coordinates']
            X, Y, Z = grid_coords['x'], grid_coords['y'], grid_coords['z']
            
            # Surface level analysis (Z = 0 or near-surface)
            surface_mask = np.abs(Z) < 100  # Within 100m of surface
            
            # Extract surface data
            surface_indices = np.where(surface_mask)
            surface_x = X[surface_indices]
            surface_y = Y[surface_indices]
            
            # Combine analysis results for surface locations
            stress_factor = stress_field['von_mises_stress'][surface_indices]
            failure_factor = failure_analysis['combined_failure_probability'][surface_indices]
            
            # Interpolate topographic controls to analysis grid
            topo_favorability = await self._interpolate_to_grid(
                topographic_controls['topographic_favorability'],
                topographic_controls['dem_data']['elevation'],
                surface_x, surface_y
            )
            
            structural_control = await self._interpolate_to_grid(
                topographic_controls['structural_control_index'],
                topographic_controls['dem_data']['elevation'],
                surface_x, surface_y
            )
            
            # Calculate sill proximity factor
            sill_proximity = await self._calculate_sill_proximity_factor(
                surface_x, surface_y, sill_geometries
            )
            
            # Combine factors using weighted approach
            weights = {
                'stress': 0.25,
                'failure': 0.30,
                'topographic': 0.20,
                'structural': 0.15,
                'sill_proximity': 0.10
            }
            
            # Normalize factors to [0, 1]
            stress_norm = (stress_factor - np.min(stress_factor)) / (np.max(stress_factor) - np.min(stress_factor) + 1e-10)
            failure_norm = failure_factor  # Already normalized
            topo_norm = (topo_favorability - np.min(topo_favorability)) / (np.max(topo_favorability) - np.min(topo_favorability) + 1e-10)
            struct_norm = (structural_control - np.min(structural_control)) / (np.max(structural_control) - np.min(structural_control) + 1e-10)
            prox_norm = (sill_proximity - np.min(sill_proximity)) / (np.max(sill_proximity) - np.min(sill_proximity) + 1e-10)
            
            # Calculate composite eruption probability
            eruption_probability = (
                weights['stress'] * stress_norm +
                weights['failure'] * failure_norm +
                weights['topographic'] * topo_norm +
                weights['structural'] * struct_norm +
                weights['sill_proximity'] * prox_norm
            )
            
            # Apply machine learning enhancement
            ml_enhanced_probability = await self._apply_ml_enhancement(
                surface_x, surface_y, eruption_probability,
                stress_norm, failure_norm, topo_norm, struct_norm, prox_norm
            )
            
            # Calculate uncertainty bounds
            uncertainty_bounds = await self._calculate_uncertainty_bounds(
                ml_enhanced_probability, confidence_level
            )
            
            # Identify high-probability zones
            high_prob_zones = await self._identify_high_probability_zones(
                surface_x, surface_y, ml_enhanced_probability, threshold=0.7
            )
            
            # Create probability contours
            probability_contours = await self._create_probability_contours(
                surface_x, surface_y, ml_enhanced_probability
            )
            
            probabilistic_mapping = {
                'surface_coordinates': {
                    'x': surface_x,
                    'y': surface_y
                },
                'eruption_probability': ml_enhanced_probability,
                'uncertainty_bounds': uncertainty_bounds,
                'contributing_factors': {
                    'stress_factor': stress_norm,
                    'failure_factor': failure_norm,
                    'topographic_factor': topo_norm,
                    'structural_factor': struct_norm,
                    'sill_proximity_factor': prox_norm
                },
                'high_probability_zones': high_prob_zones,
                'probability_contours': probability_contours,
                'analysis_weights': weights,
                'confidence_level': confidence_level,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Store results in database
            await self._store_probabilistic_mapping(probabilistic_mapping)
            
            logger.info("Completed probabilistic eruption location mapping")
            return probabilistic_mapping
            
        except Exception as e:
            logger.error(f"Error in probabilistic mapping: {e}")
            raise

    async def model_temporal_evolution_eruption_potential(
        self,
        sill_geometries: List[SillGeometry],
        growth_rates: Dict[str, float],
        time_horizon: float = 365.0  # days
    ) -> Dict[str, Any]:
        """
        Model temporal evolution of eruption potential during sill growth
        """
        try:
            # Define time steps
            time_steps = np.linspace(0, time_horizon, int(time_horizon / 10))  # 10-day intervals
            
            # Initialize temporal evolution arrays
            temporal_results = {
                'time_steps': time_steps,
                'sill_evolution': {},
                'stress_evolution': {},
                'failure_evolution': {},
                'eruption_potential_evolution': {},
                'critical_events': []
            }
            
            # Track evolution for each sill
            for sill in sill_geometries:
                sill_id = sill.sill_id
                
                # Initialize sill evolution tracking
                temporal_results['sill_evolution'][sill_id] = {
                    'volume': np.zeros_like(time_steps),
                    'pressure': np.zeros_like(time_steps),
                    'thickness': np.zeros_like(time_steps),
                    'lateral_extent': np.zeros_like(time_steps)
                }
                
                # Model sill growth over time
                initial_volume = sill.volume
                growth_rate = growth_rates.get(sill_id, 1000.0)  # m³/day default
                
                for i, t in enumerate(time_steps):
                    # Volume growth (exponential decay approach to maximum)
                    max_volume = initial_volume * 5  # Assume 5x growth potential
                    current_volume = max_volume * (1 - np.exp(-growth_rate * t / max_volume))
                    temporal_results['sill_evolution'][sill_id]['volume'][i] = current_volume
                    
                    # Pressure evolution (increases with volume, decreases with lateral growth)
                    volume_ratio = current_volume / initial_volume
                    pressure_factor = np.sqrt(volume_ratio) * (1 + 0.1 * np.sin(0.1 * t))  # Add oscillations
                    temporal_results['sill_evolution'][sill_id]['pressure'][i] = pressure_factor
                    
                    # Thickness evolution
                    thickness_growth = np.log(1 + volume_ratio) * sill.thickness
                    temporal_results['sill_evolution'][sill_id]['thickness'][i] = thickness_growth
                    
                    # Lateral extent evolution
                    lateral_growth = np.sqrt(volume_ratio) * np.sqrt(sill.length * sill.width)
                    temporal_results['sill_evolution'][sill_id]['lateral_extent'][i] = lateral_growth
            
            # Calculate stress field evolution
            stress_evolution = await self._calculate_stress_evolution(
                sill_geometries, temporal_results['sill_evolution'], time_steps
            )
            temporal_results['stress_evolution'] = stress_evolution
            
            # Calculate failure probability evolution
            failure_evolution = await self._calculate_failure_evolution(
                stress_evolution, time_steps
            )
            temporal_results['failure_evolution'] = failure_evolution
            
            # Calculate overall eruption potential evolution
            eruption_potential = await self._calculate_eruption_potential_evolution(
                stress_evolution, failure_evolution, time_steps
            )
            temporal_results['eruption_potential_evolution'] = eruption_potential
            
            # Identify critical events and thresholds
            critical_events = await self._identify_critical_events(
                temporal_results, time_steps
            )
            temporal_results['critical_events'] = critical_events
            
            # Calculate eruption probability forecasts
            eruption_forecasts = await self._generate_eruption_forecasts(
                temporal_results, time_steps
            )
            temporal_results['eruption_forecasts'] = eruption_forecasts
            
            # Store results in database
            await self._store_temporal_evolution(temporal_results)
            
            logger.info(f"Completed temporal evolution modeling for {time_horizon} days")
            return temporal_results
            
        except Exception as e:
            logger.error(f"Error in temporal evolution modeling: {e}")
            raise

    async def predict_eruption_style(
        self,
        sill_geometries: List[SillGeometry],
        magma_properties: Dict[str, float],
        stress_field: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict eruption style based on sill depth, geometry, and magma properties
        """
        try:
            eruption_style_predictions = {}
            
            for sill in sill_geometries:
                sill_id = sill.sill_id
                
                # Extract sill parameters
                depth = abs(sill.center_coordinates[2])  # Depth below surface
                volume = sill.volume
                aspect_ratio = sill.aspect_ratio
                thickness = sill.thickness
                
                # Extract magma properties
                viscosity = magma_properties.get('viscosity', 1e4)  # Pa·s
                gas_content = magma_properties.get('gas_content', 0.03)  # 3% by weight
                temperature = magma_properties.get('temperature', 1200)  # °C
                silica_content = magma_properties.get('silica_content', 0.65)  # 65% SiO2
                
                # Calculate eruption style indicators
                
                # 1. Explosivity index based on gas content and viscosity
                explosivity_index = np.log10(viscosity) * gas_content * 100
                
                # 2. Depth factor (shallow = more explosive)
                depth_factor = np.exp(-depth / 2000.0)  # Exponential decay with 2km scale
                
                # 3. Volume factor (larger = more explosive)
                volume_factor = np.log10(volume / 1e6)  # Normalize to 1 million m³
                
                # 4. Aspect ratio factor (elongated = more effusive)
                aspect_factor = 1.0 / (1.0 + aspect_ratio / 5.0)
                
                # 5. Magma composition factor
                composition_factor = silica_content  # Higher silica = more explosive
                
                # Combine factors
                explosive_potential = (
                    0.3 * explosivity_index +
                    0.25 * depth_factor +
                    0.2 * volume_factor +
                    0.1 * aspect_factor +
                    0.15 * composition_factor
                )
                
                # Classify eruption style
                if explosive_potential > 0.8:
                    eruption_style = "Plinian"
                    vei_estimate = min(6, int(4 + explosive_potential))
                elif explosive_potential > 0.6:
                    eruption_style = "Sub-Plinian"
                    vei_estimate = min(5, int(3 + explosive_potential))
                elif explosive_potential > 0.4:
                    eruption_style = "Vulcanian"
                    vei_estimate = min(4, int(2 + explosive_potential))
                elif explosive_potential > 0.2:
                    eruption_style = "Strombolian"
                    vei_estimate = min(3, int(1 + explosive_potential))
                else:
                    eruption_style = "Effusive"
                    vei_estimate = min(2, int(explosive_potential * 2))
                
                # Calculate additional eruption characteristics
                discharge_rate = await self._estimate_discharge_rate(
                    sill, magma_properties, explosive_potential
                )
                
                column_height = await self._estimate_column_height(
                    discharge_rate, explosive_potential, magma_properties
                )
                
                duration_estimate = await self._estimate_eruption_duration(
                    volume, discharge_rate, eruption_style
                )
                
                eruption_style_predictions[sill_id] = {
                    'eruption_style': eruption_style,
                    'vei_estimate': vei_estimate,
                    'explosive_potential': explosive_potential,
                    'discharge_rate': discharge_rate,  # m³/s
                    'column_height': column_height,  # m
                    'duration_estimate': duration_estimate,  # hours
                    'contributing_factors': {
                        'explosivity_index': explosivity_index,
                        'depth_factor': depth_factor,
                        'volume_factor': volume_factor,
                        'aspect_factor': aspect_factor,
                        'composition_factor': composition_factor
                    },
                    'confidence': min(1.0, 0.5 + 0.1 * explosive_potential),
                    'uncertainty_range': {
                        'vei_min': max(0, vei_estimate - 1),
                        'vei_max': min(7, vei_estimate + 1)
                    }
                }
            
            # Store results in database
            await self._store_eruption_style_predictions(eruption_style_predictions)
            
            logger.info(f"Completed eruption style prediction for {len(sill_geometries)} sills")
            return eruption_style_predictions
            
        except Exception as e:
            logger.error(f"Error in eruption style prediction: {e}")
            raise   
    async def create_hazard_zone_mapping(
        self,
        probabilistic_mapping: Dict[str, Any],
        eruption_style_predictions: Dict[str, Any],
        population_data: Dict[str, Any],
        infrastructure_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive hazard zone mapping for sill-related volcanic activity
        """
        try:
            hazard_zones = []
            
            # Extract high probability eruption locations
            high_prob_zones = probabilistic_mapping['high_probability_zones']
            
            for zone_idx, zone in enumerate(high_prob_zones):
                zone_center = zone['center']
                zone_probability = zone['probability']
                zone_area = zone['area']
                
                # Determine dominant eruption style for this zone
                dominant_style = await self._determine_dominant_eruption_style(
                    zone_center, eruption_style_predictions
                )
                
                # Create hazard zones for different hazard types
                hazard_types = ['lava_flow', 'pyroclastic_flow', 'ash_fall', 'volcanic_gas']
                
                for hazard_type in hazard_types:
                    # Calculate hazard-specific parameters
                    hazard_params = await self._calculate_hazard_parameters(
                        hazard_type, dominant_style, zone_center, zone_probability
                    )
                    
                    # Create hazard zone geometry
                    hazard_geometry = await self._create_hazard_geometry(
                        zone_center, hazard_params, hazard_type
                    )
                    
                    # Assess population at risk
                    population_at_risk = await self._assess_population_at_risk(
                        hazard_geometry, population_data
                    )
                    
                    # Assess infrastructure at risk
                    infrastructure_at_risk = await self._assess_infrastructure_at_risk(
                        hazard_geometry, infrastructure_data
                    )
                    
                    # Calculate evacuation requirements
                    evacuation_time = await self._calculate_evacuation_time(
                        population_at_risk, hazard_params, hazard_type
                    )
                    
                    # Determine monitoring priority
                    monitoring_priority = await self._calculate_monitoring_priority(
                        zone_probability, population_at_risk, infrastructure_at_risk
                    )
                    
                    # Create hazard zone object
                    hazard_zone = HazardZone(
                        zone_id=f"hazard_{zone_idx}_{hazard_type}",
                        geometry=hazard_geometry,
                        hazard_level=hazard_params['hazard_level'],
                        hazard_type=hazard_type,
                        probability=zone_probability * hazard_params['type_probability'],
                        impact_radius=hazard_params['impact_radius'],
                        population_at_risk=population_at_risk['total'],
                        infrastructure_at_risk=infrastructure_at_risk,
                        evacuation_time=evacuation_time,
                        monitoring_priority=monitoring_priority
                    )
                    
                    hazard_zones.append(hazard_zone)
            
            # Create composite hazard map
            composite_hazard_map = await self._create_composite_hazard_map(hazard_zones)
            
            # Calculate cumulative risk metrics
            risk_metrics = await self._calculate_risk_metrics(hazard_zones, population_data)
            
            # Generate evacuation plans
            evacuation_plans = await self._generate_evacuation_plans(
                hazard_zones, population_data, infrastructure_data
            )
            
            # Create monitoring recommendations
            monitoring_recommendations = await self._create_monitoring_recommendations(
                hazard_zones, probabilistic_mapping
            )
            
            hazard_mapping_results = {
                'hazard_zones': [
                    {
                        'zone_id': hz.zone_id,
                        'geometry': hz.geometry.__geo_interface__,
                        'hazard_level': hz.hazard_level,
                        'hazard_type': hz.hazard_type,
                        'probability': hz.probability,
                        'impact_radius': hz.impact_radius,
                        'population_at_risk': hz.population_at_risk,
                        'infrastructure_at_risk': hz.infrastructure_at_risk,
                        'evacuation_time': hz.evacuation_time,
                        'monitoring_priority': hz.monitoring_priority
                    } for hz in hazard_zones
                ],
                'composite_hazard_map': composite_hazard_map,
                'risk_metrics': risk_metrics,
                'evacuation_plans': evacuation_plans,
                'monitoring_recommendations': monitoring_recommendations,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Store results in database
            await self._store_hazard_mapping(hazard_mapping_results)
            
            logger.info(f"Created {len(hazard_zones)} hazard zones")
            return hazard_mapping_results
            
        except Exception as e:
            logger.error(f"Error in hazard zone mapping: {e}")
            raise

    async def validate_against_historical_eruptions(
        self,
        prediction_results: Dict[str, Any],
        historical_data_source: str = "global_volcanism_program"
    ) -> Dict[str, Any]:
        """
        Validate eruption location predictions against historical sill-related eruptions
        """
        try:
            # Load historical eruption data
            historical_eruptions = await self._load_historical_eruption_data(
                historical_data_source
            )
            
            # Filter for sill-related eruptions
            sill_related_eruptions = await self._filter_sill_related_eruptions(
                historical_eruptions
            )
            
            # Extract prediction locations and probabilities
            predicted_locations = prediction_results['surface_coordinates']
            predicted_probabilities = prediction_results['eruption_probability']
            
            # Initialize validation metrics
            validation_metrics = {
                'spatial_accuracy': {},
                'probability_calibration': {},
                'temporal_accuracy': {},
                'style_prediction_accuracy': {},
                'overall_performance': {}
            }
            
            # Spatial accuracy validation
            spatial_accuracy = await self._validate_spatial_accuracy(
                predicted_locations, predicted_probabilities, sill_related_eruptions
            )
            validation_metrics['spatial_accuracy'] = spatial_accuracy
            
            # Probability calibration validation
            prob_calibration = await self._validate_probability_calibration(
                predicted_probabilities, sill_related_eruptions
            )
            validation_metrics['probability_calibration'] = prob_calibration
            
            # Temporal accuracy validation (if temporal predictions available)
            if 'eruption_forecasts' in prediction_results:
                temporal_accuracy = await self._validate_temporal_accuracy(
                    prediction_results['eruption_forecasts'], sill_related_eruptions
                )
                validation_metrics['temporal_accuracy'] = temporal_accuracy
            
            # Eruption style prediction accuracy
            if 'eruption_style_predictions' in prediction_results:
                style_accuracy = await self._validate_style_predictions(
                    prediction_results['eruption_style_predictions'], sill_related_eruptions
                )
                validation_metrics['style_prediction_accuracy'] = style_accuracy
            
            # Calculate overall performance metrics
            overall_performance = await self._calculate_overall_performance(
                validation_metrics, sill_related_eruptions
            )
            validation_metrics['overall_performance'] = overall_performance
            
            # Generate validation report
            validation_report = await self._generate_validation_report(
                validation_metrics, sill_related_eruptions, prediction_results
            )
            
            validation_results = {
                'validation_metrics': validation_metrics,
                'historical_eruptions_used': len(sill_related_eruptions),
                'validation_report': validation_report,
                'model_performance_score': overall_performance['composite_score'],
                'recommendations': await self._generate_model_improvement_recommendations(
                    validation_metrics
                ),
                'validation_timestamp': datetime.now().isoformat()
            }
            
            # Store validation results in database
            await self._store_validation_results(validation_results)
            
            logger.info(f"Completed validation against {len(sill_related_eruptions)} historical eruptions")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in historical validation: {e}")
            raise

    # Helper methods for database operations and calculations
    async def _load_historical_eruptions(self):
        """Load historical eruption data from database"""
        async with self.engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT eruption_id, volcano_name, latitude, longitude, 
                       eruption_date, vei, eruption_style, sill_related
                FROM historical_eruptions 
                WHERE sill_related = true
                ORDER BY eruption_date DESC
            """))
            return [dict(row) for row in result]

    async def _initialize_prediction_models(self):
        """Initialize machine learning models for prediction enhancement"""
        # Random Forest for eruption probability
        self.prediction_models['rf_probability'] = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        
        # Gaussian Process for uncertainty quantification
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.prediction_models['gp_uncertainty'] = GaussianProcessRegressor(
            kernel=kernel, random_state=42
        )
        
        # Gradient Boosting for temporal evolution
        self.prediction_models['gb_temporal'] = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )

    async def _store_stress_field_analysis(self, results: Dict[str, Any]):
        """Store stress field analysis results in database"""
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO stress_field_analyses 
                (analysis_id, analysis_timestamp, results_data)
                VALUES (:analysis_id, :timestamp, :results)
            """), {
                'analysis_id': f"stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now(),
                'results': str(results)
            })

    async def _store_eruption_predictions(self, predictions: Dict[str, Any]):
        """Store eruption location predictions in database"""
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO eruption_predictions 
                (prediction_id, prediction_timestamp, prediction_data, model_version)
                VALUES (:prediction_id, :timestamp, :data, :version)
            """), {
                'prediction_id': f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now(),
                'data': str(predictions),
                'version': "1.0.0"
            })

# Example usage and testing
async def main():
    """Example usage of the Eruption Location Prediction System"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'prometheus_volcanic',
        'user': 'prometheus',
        'password': 'prometheus123',
        'redis_host': 'localhost',
        'redis_port': 6379
    }
    
    # Initialize system
    prediction_system = EruptionLocationPredictionSystem(db_config)
    await prediction_system.initialize()
    
    # Example sill geometries
    sill_geometries = [
        SillGeometry(
            sill_id="sill_001",
            center_coordinates=(0, 0, -1500),  # 1.5 km depth
            length=3000, width=2000, thickness=50,
            dip=15, strike=45, volume=3e8,
            aspect_ratio=1.5, shape_factor=0.8, connectivity_index=0.6
        ),
        SillGeometry(
            sill_id="sill_002", 
            center_coordinates=(1000, 500, -2000),  # 2 km depth
            length=2500, width=1800, thickness=40,
            dip=20, strike=60, volume=1.8e8,
            aspect_ratio=1.4, shape_factor=0.7, connectivity_index=0.4
        )
    ]
    
    logger.info("Eruption Location Prediction System example completed")

if __name__ == "__main__":
    asyncio.run(main())