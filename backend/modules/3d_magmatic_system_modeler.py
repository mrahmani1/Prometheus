"""
Advanced 3D Magmatic System Modeling from Seismic Tomography
Integrates velocity models, attenuation, anisotropy, and melt fraction estimation
for comprehensive magmatic reservoir characterization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial import cKDTree, Voronoi
from scipy.interpolate import RegularGridInterpolator, RBFInterpolator
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class MagmaChamberType(Enum):
    MAIN_CHAMBER = "main_chamber"
    SECONDARY_CHAMBER = "secondary_chamber"
    SILL_COMPLEX = "sill_complex"
    DYKE_SYSTEM = "dyke_system"
    MUSH_ZONE = "mush_zone"
    CRYSTAL_MUSH = "crystal_mush"

class MeltState(Enum):
    SOLID = "solid"
    PARTIAL_MELT = "partial_melt"
    HIGH_MELT_FRACTION = "high_melt_fraction"
    LIQUID_DOMINATED = "liquid_dominated"
    VOLATILE_RICH = "volatile_rich"

class MagmaComposition(Enum):
    BASALTIC = "basaltic"
    ANDESITIC = "andesitic" 
    DACITIC = "dacitic"
    RHYOLITIC = "rhyolitic"
    MIXED = "mixed"

@dataclass
class MagmaChamberGeometry:
    """3D geometry of magma chamber"""
    center_coordinates: Tuple[float, float, float]  # lon, lat, depth_km
    volume_km3: float
    aspect_ratio: float  # width/height
    orientation: Dict[str, float]  # strike, dip, plunge
    shape_type: str  # ellipsoid, cylinder, complex
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    boundary_surface: np.ndarray

@dataclass
class SeismicProperties:
    """Seismic properties of magmatic system"""
    vp_field: np.ndarray
    vs_field: np.ndarray
    vp_vs_ratio: np.ndarray
    attenuation_qp: np.ndarray
    attenuation_qs: np.ndarray
    anisotropy_strength: np.ndarray
    anisotropy_orientation: np.ndarray
    scattering_strength: np.ndarray
    velocity_gradients: Dict[str, np.ndarray]

@dataclass
class MeltFractionModel:
    """3D melt fraction distribution"""
    melt_fraction_3d: np.ndarray
    crystal_fraction_3d: np.ndarray
    volatile_fraction_3d: np.ndarray
    temperature_3d: np.ndarray
    pressure_3d: np.ndarray
    composition_3d: Dict[str, np.ndarray]
    phase_assemblage: np.ndarray
    melt_connectivity: np.ndarray
    viscosity_3d: np.ndarray

@dataclass
class MagmaticSystem:
    """Complete 3D magmatic system model"""
    system_id: str
    chamber_geometries: List[MagmaChamberGeometry]
    seismic_properties: SeismicProperties
    melt_model: MeltFractionModel
    thermal_structure: Dict[str, np.ndarray]
    fluid_pathways: List[Dict]
    eruption_potential_3d: np.ndarray
    stress_perturbations: np.ndarray
    coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]
    uncertainty_3d: np.ndarray
    validation_metrics: Dict[str, float]

class VelocityToMeltFractionNN(nn.Module):
    """Neural network to predict melt fraction from seismic velocities"""
    
    def __init__(self, input_features: int = 10, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads
        self.melt_fraction_head = nn.Linear(prev_dim, 1)
        self.temperature_head = nn.Linear(prev_dim, 1)
        self.pressure_head = nn.Linear(prev_dim, 1)
        self.composition_head = nn.Linear(prev_dim, 4)  # SiO2, MgO, H2O, CO2
        self.uncertainty_head = nn.Linear(prev_dim, 1)
        
        # Physics-informed constraints
        self.physics_layer = nn.Linear(prev_dim, prev_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        
        # Apply physics constraints
        physics_features = torch.tanh(self.physics_layer(features)) * features
        
        outputs = {
            'melt_fraction': torch.sigmoid(self.melt_fraction_head(physics_features)),
            'temperature': 600 + 600 * torch.sigmoid(self.temperature_head(physics_features)),  # 600-1200°C
            'pressure': 50 + 450 * torch.sigmoid(self.pressure_head(physics_features)),  # 50-500 MPa
            'composition': torch.sigmoid(self.composition_head(physics_features)),
            'uncertainty': torch.sigmoid(self.uncertainty_head(physics_features))
        }
        
        return outputs

class ThreeDMagmaticSystemModeler:
    """Advanced 3D magmatic system modeling from seismic tomography"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # ML models
        self.velocity_to_melt_model = None
        self.geometry_estimator = None
        self.thermal_model = None
        
        # Physical constants and relationships
        self.physical_constants = self._initialize_physical_constants()
        
        # Calibration data
        self.calibration_data = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'spatial_resolution_km': 0.5,
            'melt_detection_threshold': 0.05,  # 5% melt fraction
            'velocity_anomaly_threshold': -10,  # % deviation
            'chamber_min_volume_km3': 1.0,
            'connectivity_threshold': 0.3,
            'temperature_constraints': {
                'solidus_c': 650,
                'liquidus_c': 1200,
                'ambient_gradient_c_km': 25
            },
            'pressure_constraints': {
                'lithostatic_gradient_mpa_km': 27,
                'minimum_overpressure_mpa': 5
            },
            'composition_models': {
                'basalt': {'vp_dry': 6.8, 'vs_dry': 3.9, 'density': 2900},
                'andesite': {'vp_dry': 6.2, 'vs_dry': 3.5, 'density': 2600},
                'dacite': {'vp_dry': 5.8, 'vs_dry': 3.2, 'density': 2400},
                'rhyolite': {'vp_dry': 5.4, 'vs_dry': 2.9, 'density': 2300}
            },
            'ml_training': {
                'batch_size': 128,
                'learning_rate': 0.001,
                'epochs': 300,
                'validation_split': 0.15
            },
            'uncertainty_quantification': True,
            'monte_carlo_samples': 1000
        }
    
    def _initialize_physical_constants(self) -> Dict[str, Any]:
        """Initialize physical constants and empirical relationships"""
        
        return {
            # Velocity-melt fraction relationships (Takei 2002, Hammond & Humphreys 2000)
            'velocity_melt_relations': {
                'vp_reduction_per_percent_melt': 0.01,  # 1% per 1% melt
                'vs_reduction_per_percent_melt': 0.015,  # 1.5% per 1% melt
                'critical_melt_fraction': 0.07,  # Critical connectivity threshold
                'percolation_threshold': 0.03   # Onset of interconnected melt
            },
            # Attenuation relationships
            'attenuation_melt_relations': {
                'q_background': 300,
                'q_melt_coefficient': -50,  # Q decreases with melt
                'frequency_dependence': 0.3
            },
            # Temperature-velocity relations
            'thermal_velocity_relations': {
                'vp_temp_gradient_ms_k': -0.0004,  # m/s per K
                'vs_temp_gradient_ms_k': -0.0002
            },
            # Pressure effects
            'pressure_velocity_relations': {
                'vp_pressure_gradient_ms_mpa': 0.001,
                'vs_pressure_gradient_ms_mpa': 0.0005
            }
        }
    
    async def create_magmatic_system_model(self, tomography_data: Dict, 
                                         additional_constraints: Dict = None) -> MagmaticSystem:
        """Create comprehensive 3D magmatic system model"""
        
        self.logger.info("Creating 3D magmatic system model from tomography")
        
        # Extract tomography data
        coordinates = tomography_data['coordinates']
        vp_3d = tomography_data['vp_3d']
        vs_3d = tomography_data['vs_3d']
        
        if 'attenuation_3d' in tomography_data:
            q_3d = tomography_data['attenuation_3d']
        else:
            q_3d = np.ones_like(vp_3d) * 200  # Default Q
        
        # Create seismic properties structure
        seismic_properties = await self._analyze_seismic_properties(
            vp_3d, vs_3d, q_3d, coordinates
        )
        
        # Estimate melt fraction distribution
        melt_model = await self._estimate_melt_fraction_3d(
            seismic_properties, coordinates, additional_constraints
        )
        
        # Identify and model magma chamber geometries
        chamber_geometries = await self._identify_chamber_geometries(
            melt_model, seismic_properties, coordinates
        )
        
        # Model thermal structure
        thermal_structure = await self._model_thermal_structure(
            melt_model, chamber_geometries, coordinates
        )
        
        # Identify fluid pathways
        fluid_pathways = await self._identify_fluid_pathways(
            melt_model, seismic_properties, coordinates
        )
        
        # Calculate eruption potential
        eruption_potential_3d = await self._calculate_eruption_potential_3d(
            melt_model, thermal_structure, chamber_geometries
        )
        
        # Estimate stress perturbations from magma chambers
        stress_perturbations = await self._calculate_stress_perturbations(
            chamber_geometries, coordinates
        )
        
        # Uncertainty quantification
        uncertainty_3d = await self._quantify_model_uncertainty(
            seismic_properties, melt_model, coordinates
        )
        
        # Validation metrics
        validation_metrics = await self._calculate_validation_metrics(
            seismic_properties, melt_model, chamber_geometries
        )
        
        magmatic_system = MagmaticSystem(
            system_id=f"magmatic_system_{int(time.time())}",
            chamber_geometries=chamber_geometries,
            seismic_properties=seismic_properties,
            melt_model=melt_model,
            thermal_structure=thermal_structure,
            fluid_pathways=fluid_pathways,
            eruption_potential_3d=eruption_potential_3d,
            stress_perturbations=stress_perturbations,
            coordinates=coordinates,
            uncertainty_3d=uncertainty_3d,
            validation_metrics=validation_metrics
        )
        
        return magmatic_system
    
    async def _analyze_seismic_properties(self, vp_3d: np.ndarray, vs_3d: np.ndarray,
                                        q_3d: np.ndarray, coordinates: Tuple) -> SeismicProperties:
        """Analyze comprehensive seismic properties"""
        
        # Calculate Vp/Vs ratio
        vp_vs_ratio = vp_3d / (vs_3d + 1e-8)
        
        # Calculate velocity gradients
        X, Y, Z = coordinates
        dx = X[1, 0, 0] - X[0, 0, 0]
        dy = Y[0, 1, 0] - Y[0, 0, 0]
        dz = Z[0, 0, 1] - Z[0, 0, 0]
        
        velocity_gradients = {
            'vp_dx': np.gradient(vp_3d, dx, axis=0),
            'vp_dy': np.gradient(vp_3d, dy, axis=1),
            'vp_dz': np.gradient(vp_3d, dz, axis=2),
            'vs_dx': np.gradient(vs_3d, dx, axis=0),
            'vs_dy': np.gradient(vs_3d, dy, axis=1),
            'vs_dz': np.gradient(vs_3d, dz, axis=2)
        }
        
        # Estimate anisotropy (simplified - would need full tensor analysis)
        anisotropy_strength = np.sqrt(
            velocity_gradients['vp_dx']**2 + 
            velocity_gradients['vp_dy']**2 + 
            velocity_gradients['vp_dz']**2
        ) / (vp_3d + 1e-8)
        
        # Simple anisotropy orientation (fast axis direction)
        anisotropy_orientation = np.arctan2(velocity_gradients['vp_dy'], 
                                          velocity_gradients['vp_dx'])
        
        # Scattering strength from velocity heterogeneity
        scattering_strength = self._calculate_scattering_strength(vp_3d, vs_3d)
        
        # Split Q into Qp and Qs
        qp = q_3d
        qs = q_3d * 0.7  # Typical Qs/Qp ratio
        
        return SeismicProperties(
            vp_field=vp_3d,
            vs_field=vs_3d,
            vp_vs_ratio=vp_vs_ratio,
            attenuation_qp=qp,
            attenuation_qs=qs,
            anisotropy_strength=anisotropy_strength,
            anisotropy_orientation=anisotropy_orientation,
            scattering_strength=scattering_strength,
            velocity_gradients=velocity_gradients
        )
    
    def _calculate_scattering_strength(self, vp_3d: np.ndarray, vs_3d: np.ndarray) -> np.ndarray:
        """Calculate seismic scattering strength from heterogeneity"""
        
        # Local velocity variance as proxy for scattering
        vp_variance = gaussian_filter(vp_3d, sigma=2) - gaussian_filter(vp_3d, sigma=0.5)
        vs_variance = gaussian_filter(vs_3d, sigma=2) - gaussian_filter(vs_3d, sigma=0.5)
        
        scattering = np.sqrt(vp_variance**2 + vs_variance**2) / (vp_3d + 1e-8)
        
        return scattering
    
    async def _estimate_melt_fraction_3d(self, seismic_props: SeismicProperties,
                                       coordinates: Tuple, 
                                       constraints: Dict = None) -> MeltFractionModel:
        """Estimate 3D melt fraction from seismic properties"""
        
        vp = seismic_props.vp_field
        vs = seismic_props.vs_field
        vp_vs_ratio = seismic_props.vp_vs_ratio
        qp = seismic_props.attenuation_qp
        
        X, Y, Z = coordinates
        
        # Initialize melt fraction estimation
        if self.velocity_to_melt_model is None:
            # Use empirical relationships if ML model not trained
            melt_fraction = self._empirical_melt_estimation(vp, vs, vp_vs_ratio, qp)
        else:
            # Use trained ML model
            melt_fraction = await self._ml_melt_estimation(seismic_props)
        
        # Estimate temperature from velocities and melt fraction
        temperature_3d = self._estimate_temperature_3d(vp, vs, melt_fraction, Z)
        
        # Estimate pressure (lithostatic + overpressure)
        pressure_3d = self._estimate_pressure_3d(melt_fraction, Z)
        
        # Calculate crystal fraction
        crystal_fraction = 1.0 - melt_fraction
        
        # Estimate volatile content
        volatile_fraction = self._estimate_volatile_content(
            melt_fraction, temperature_3d, pressure_3d
        )
        
        # Determine composition distribution
        composition_3d = self._estimate_composition_3d(vp, vs, temperature_3d)
        
        # Calculate melt connectivity
        melt_connectivity = self._calculate_melt_connectivity(melt_fraction)
        
        # Estimate viscosity
        viscosity_3d = self._calculate_viscosity_3d(
            melt_fraction, temperature_3d, composition_3d, volatile_fraction
        )
        
        # Determine phase assemblage
        phase_assemblage = self._determine_phase_assemblage(
            temperature_3d, pressure_3d, composition_3d
        )
        
        return MeltFractionModel(
            melt_fraction_3d=melt_fraction,
            crystal_fraction_3d=crystal_fraction,
            volatile_fraction_3d=volatile_fraction,
            temperature_3d=temperature_3d,
            pressure_3d=pressure_3d,
            composition_3d=composition_3d,
            phase_assemblage=phase_assemblage,
            melt_connectivity=melt_connectivity,
            viscosity_3d=viscosity_3d
        )
    
    def _empirical_melt_estimation(self, vp: np.ndarray, vs: np.ndarray, 
                                 vp_vs_ratio: np.ndarray, qp: np.ndarray) -> np.ndarray:
        """Empirical melt fraction estimation from seismic properties"""
        
        # Reference velocities for solid rock
        vp_solid = 6.5  # km/s
        vs_solid = 3.7  # km/s
        vp_vs_solid = vp_solid / vs_solid
        
        # Velocity reduction approach (Takei 2002)
        vp_reduction = (vp_solid - vp) / vp_solid
        vs_reduction = (vs_solid - vs) / vs_solid
        
        # Convert velocity reduction to melt fraction
        vp_melt = vp_reduction / self.physical_constants['velocity_melt_relations']['vp_reduction_per_percent_melt']
        vs_melt = vs_reduction / self.physical_constants['velocity_melt_relations']['vs_reduction_per_percent_melt']
        
        # Average estimates, but weight Vs more heavily
        melt_from_velocity = 0.3 * vp_melt + 0.7 * vs_melt
        
        # Vp/Vs ratio approach - high ratios indicate melt
        vp_vs_anomaly = vp_vs_ratio - vp_vs_solid
        melt_from_ratio = vp_vs_anomaly * 0.5  # Empirical scaling
        
        # Attenuation approach - low Q indicates melt
        q_background = self.physical_constants['attenuation_melt_relations']['q_background']
        q_anomaly = (q_background - qp) / q_background
        melt_from_attenuation = q_anomaly * 0.3  # Empirical scaling
        
        # Combine estimates
        melt_fraction = 0.4 * melt_from_velocity + 0.4 * melt_from_ratio + 0.2 * melt_from_attenuation
        
        # Apply constraints
        melt_fraction = np.clip(melt_fraction, 0, 0.8)  # Max 80% melt
        
        # Apply connectivity threshold
        connectivity_threshold = self.physical_constants['velocity_melt_relations']['percolation_threshold']
        melt_fraction[melt_fraction < connectivity_threshold] *= 0.1  # Isolated melt pockets
        
        return melt_fraction
    
    async def _ml_melt_estimation(self, seismic_props: SeismicProperties) -> np.ndarray:
        """ML-based melt fraction estimation"""
        
        # Prepare features
        features = self._prepare_ml_features(seismic_props)
        
        # Run ML model
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            outputs = self.velocity_to_melt_model(features_tensor)
            melt_fraction = outputs['melt_fraction'].numpy()
        
        return melt_fraction.reshape(seismic_props.vp_field.shape)
    
    def _prepare_ml_features(self, seismic_props: SeismicProperties) -> np.ndarray:
        """Prepare features for ML model"""
        
        # Flatten all arrays
        vp_flat = seismic_props.vp_field.flatten()
        vs_flat = seismic_props.vs_field.flatten()
        vp_vs_flat = seismic_props.vp_vs_ratio.flatten()
        qp_flat = seismic_props.attenuation_qp.flatten()
        qs_flat = seismic_props.attenuation_qs.flatten()
        
        # Gradients
        dvp_dx_flat = seismic_props.velocity_gradients['vp_dx'].flatten()
        dvp_dy_flat = seismic_props.velocity_gradients['vp_dy'].flatten()
        dvp_dz_flat = seismic_props.velocity_gradients['vp_dz'].flatten()
        
        aniso_flat = seismic_props.anisotropy_strength.flatten()
        scatter_flat = seismic_props.scattering_strength.flatten()
        
        # Stack features
        features = np.column_stack([
            vp_flat, vs_flat, vp_vs_flat, qp_flat, qs_flat,
            dvp_dx_flat, dvp_dy_flat, dvp_dz_flat,
            aniso_flat, scatter_flat
        ])
        
        return features
    
    def _estimate_temperature_3d(self, vp: np.ndarray, vs: np.ndarray,
                               melt_fraction: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Estimate 3D temperature field"""
        
        # Ambient geothermal gradient
        surface_temp = 20  # °C
        gradient = self.config['temperature_constraints']['ambient_gradient_c_km']
        
        ambient_temp = surface_temp - z * gradient  # z is negative depth
        
        # Temperature increase from melt presence
        solidus = self.config['temperature_constraints']['solidus_c']
        liquidus = self.config['temperature_constraints']['liquidus_c']
        
        # Linear relationship between melt fraction and temperature
        melt_temp_increase = melt_fraction * (liquidus - solidus)
        
        # Velocity-based temperature estimate
        temp_from_velocity = self._temperature_from_velocity_reduction(vp, vs, ambient_temp)
        
        # Combine estimates
        temperature = ambient_temp + melt_temp_increase + temp_from_velocity
        
        # Apply constraints
        temperature = np.clip(temperature, surface_temp, liquidus + 200)
        
        return temperature
    
    def _temperature_from_velocity_reduction(self, vp: np.ndarray, vs: np.ndarray,
                                           ambient_temp: np.ndarray) -> np.ndarray:
        """Estimate temperature anomaly from velocity reduction"""
        
        # Reference velocities at ambient temperature
        vp_ref = 6.5  # km/s
        vs_ref = 3.7  # km/s
        
        # Velocity reduction
        vp_reduction = vp_ref - vp
        vs_reduction = vs_ref - vs
        
        # Convert to temperature (empirical relationship)
        temp_from_vp = vp_reduction * 1000 / abs(self.physical_constants['thermal_velocity_relations']['vp_temp_gradient_ms_k'])
        temp_from_vs = vs_reduction * 1000 / abs(self.physical_constants['thermal_velocity_relations']['vs_temp_gradient_ms_k'])
        
        # Average
        temp_anomaly = 0.5 * (temp_from_vp + temp_from_vs)
        
        return np.clip(temp_anomaly, 0, 500)  # Max 500°C anomaly
    
    def _estimate_pressure_3d(self, melt_fraction: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Estimate 3D pressure field"""
        
        # Lithostatic pressure
        gradient = self.config['pressure_constraints']['lithostatic_gradient_mpa_km']
        lithostatic = -z * gradient  # z is negative depth
        
        # Overpressure from melt buoyancy
        min_overpressure = self.config['pressure_constraints']['minimum_overpressure_mpa']
        melt_overpressure = melt_fraction * min_overpressure * 5  # Up to 5x minimum
        
        total_pressure = lithostatic + melt_overpressure
        
        return np.maximum(total_pressure, 0.1)  # Minimum 0.1 MPa
    
    def _estimate_volatile_content(self, melt_fraction: np.ndarray,
                                 temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Estimate volatile content in melt"""
        
        # H2O solubility increases with pressure, decreases with temperature
        # Simplified relationship
        h2o_solubility = 0.01 + 0.0001 * pressure - 0.00001 * (temperature - 800)
        h2o_content = melt_fraction * np.clip(h2o_solubility, 0, 0.08)  # Max 8 wt%
        
        return h2o_content
    
    def _estimate_composition_3d(self, vp: np.ndarray, vs: np.ndarray,
                               temperature: np.ndarray) -> Dict[str, np.ndarray]:
        """Estimate 3D composition distribution"""
        
        # Simple composition estimate based on velocities
        # Lower velocities = more evolved (silicic) composition
        
        composition_models = self.config['composition_models']
        
        # Initialize with intermediate composition
        sio2 = np.ones_like(vp) * 60.0  # Andesitic baseline
        
        # Adjust based on velocities
        low_velocity_mask = (vp < 5.8) & (vs < 3.2)
        high_velocity_mask = (vp > 6.5) & (vs > 3.8)
        
        sio2[low_velocity_mask] = 68.0  # Dacitic
        sio2[high_velocity_mask] = 52.0  # Basaltic
        
        # Temperature effect - hotter = less evolved
        temp_effect = (temperature - 800) * 0.01
        sio2 -= temp_effect
        
        # Other components (simplified)
        mgo = 100 - sio2 - 15  # Simplified
        al2o3 = np.ones_like(vp) * 15.0
        
        composition = {
            'SiO2': np.clip(sio2, 45, 75),
            'MgO': np.clip(mgo, 0.5, 15),
            'Al2O3': al2o3
        }
        
        return composition
    
    def _calculate_melt_connectivity(self, melt_fraction: np.ndarray) -> np.ndarray:
        """Calculate melt connectivity using percolation theory"""
        
        # Critical melt fraction for connectivity
        critical_mf = self.physical_constants['velocity_melt_relations']['critical_melt_fraction']
        
        # Connectivity function
        connectivity = np.zeros_like(melt_fraction)
        
        # Above critical threshold - exponential increase
        above_critical = melt_fraction >= critical_mf
        connectivity[above_critical] = 1 - np.exp(-10 * (melt_fraction[above_critical] - critical_mf))
        
        # Below critical - minimal connectivity
        below_critical = melt_fraction < critical_mf
        connectivity[below_critical] = (melt_fraction[below_critical] / critical_mf) * 0.1
        
        return connectivity
    
    def _calculate_viscosity_3d(self, melt_fraction: np.ndarray, temperature: np.ndarray,
                              composition: Dict[str, np.ndarray], 
                              volatile_content: np.ndarray) -> np.ndarray:
        """Calculate melt viscosity from composition, temperature, volatiles"""
        
        # Simplified viscosity model (Shaw 1972, Giordano et al. 2008)
        
        # Base viscosity from silica content
        sio2 = composition['SiO2']
        log_visc_base = -4.55 + 0.11 * sio2  # log Pa·s
        
        # Temperature effect
        temp_effect = 8000 / (temperature + 273.15)  # Arrhenius relationship
        
        # Volatile effect (reduces viscosity)
        volatile_effect = -2.0 * volatile_content  # log Pa·s reduction
        
        # Combine effects
        log_viscosity = log_visc_base + temp_effect + volatile_effect
        
        # Convert to Pa·s
        viscosity = 10**log_viscosity
        
        # Modify by melt fraction (crystal suspension effect)
        # Einstein-Roscoe equation for suspension viscosity
        crystal_fraction = 1 - melt_fraction
        suspension_factor = (1 - crystal_fraction / 0.6)**(-2.5)  # φmax = 0.6
        suspension_factor = np.clip(suspension_factor, 1, 1e6)
        
        total_viscosity = viscosity * suspension_factor
        
        return np.clip(total_viscosity, 1e2, 1e15)  # Physical bounds
    
    def _determine_phase_assemblage(self, temperature: np.ndarray, pressure: np.ndarray,
                                  composition: Dict[str, np.ndarray]) -> np.ndarray:
        """Determine mineral phase assemblage"""
        
        # Simplified phase relations
        sio2 = composition['SiO2']
        
        # Phase codes: 0=solid, 1=partial melt, 2=liquid, 3=gas phase
        phases = np.zeros_like(temperature, dtype=int)
        
        # Solidus temperature (composition dependent)
        solidus = 650 + (sio2 - 50) * 3  # Higher SiO2 = lower solidus
        
        # Liquidus temperature
        liquidus = solidus + 200 + (60 - sio2) * 2  # More mafic = higher liquidus
        
        # Assign phases
        phases[temperature < solidus] = 0  # Solid
        phases[(temperature >= solidus) & (temperature < liquidus)] = 1  # Partial melt
        phases[temperature >= liquidus] = 2  # Liquid
        
        # Gas phase at low pressure and high temperature
        gas_conditions = (pressure < 50) & (temperature > liquidus)
        phases[gas_conditions] = 3
        
        return phases
    
    async def _identify_chamber_geometries(self, melt_model: MeltFractionModel,
                                         seismic_props: SeismicProperties,
                                         coordinates: Tuple) -> List[MagmaChamberGeometry]:
        """Identify and characterize magma chamber geometries"""
        
        melt_fraction = melt_model.melt_fraction_3d
        connectivity = melt_model.melt_connectivity
        X, Y, Z = coordinates
        
        # Threshold for significant melt presence
        melt_threshold = self.config['melt_detection_threshold']
        chamber_mask = (melt_fraction > melt_threshold) & (connectivity > 0.3)
        
        # Connected component analysis
        from scipy.ndimage import label, center_of_mass
        labeled_chambers, num_chambers = label(chamber_mask)
        
        chamber_geometries = []
        
        for chamber_id in range(1, num_chambers + 1):
            chamber_points = labeled_chambers == chamber_id
            
            # Skip small chambers
            chamber_volume = np.sum(chamber_points) * (self.config['spatial_resolution_km'])**3
            if chamber_volume < self.config['chamber_min_volume_km3']:
                continue
            
            # Calculate center of mass
            center_indices = center_of_mass(chamber_points)
            center_coords = (
                X[int(center_indices[0]), int(center_indices[1]), int(center_indices[2])],
                Y[int(center_indices[0]), int(center_indices[1]), int(center_indices[2])],
                Z[int(center_indices[0]), int(center_indices[1]), int(center_indices[2])]
            )
            
            # Calculate chamber properties
            chamber_props = self._analyze_chamber_properties(
                chamber_points, X, Y, Z, melt_fraction
            )
            
            # Generate mesh for chamber boundary
            mesh_vertices, mesh_faces = self._generate_chamber_mesh(chamber_points, X, Y, Z)
            
            # Determine chamber type
            chamber_type = self._classify_chamber_type(chamber_props)
            
            geometry = MagmaChamberGeometry(
                center_coordinates=center_coords,
                volume_km3=chamber_volume,
                aspect_ratio=chamber_props['aspect_ratio'],
                orientation=chamber_props['orientation'],
                shape_type=chamber_props['shape_type'],
                mesh_vertices=mesh_vertices,
                mesh_faces=mesh_faces,
                boundary_surface=chamber_props['boundary_surface']
            )
            
            chamber_geometries.append(geometry)
        
        return chamber_geometries
    
    def _analyze_chamber_properties(self, chamber_mask: np.ndarray,
                                  X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                  melt_fraction: np.ndarray) -> Dict:
        """Analyze geometric properties of magma chamber"""
        
        # Get chamber points
        chamber_indices = np.where(chamber_mask)
        chamber_coords = np.column_stack([
            X[chamber_indices], Y[chamber_indices], Z[chamber_indices]
        ])
        
        # Principal component analysis for orientation
        centered_coords = chamber_coords - np.mean(chamber_coords, axis=0)
        covariance = np.cov(centered_coords.T)
        eigenvals, eigenvecs = np.linalg.eigh(covariance)
        
        # Sort by eigenvalue (largest first)
        sorted_idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sorted_idx]
        eigenvecs = eigenvecs[:, sorted_idx]
        
        # Aspect ratio
        aspect_ratio = eigenvals[0] / eigenvals[2]
        
        # Orientation (strike, dip of major axis)
        major_axis = eigenvecs[:, 0]
        strike = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
        dip = np.degrees(np.arcsin(-major_axis[2]))
        
        # Shape characterization
        if aspect_ratio < 2:
            shape_type = "spherical"
        elif aspect_ratio < 5:
            shape_type = "ellipsoidal"
        else:
            shape_type = "elongated"
        
        # Boundary surface (simplified - just convex hull of points)
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(chamber_coords)
            boundary_surface = hull.vertices
        except:
            boundary_surface = np.array([])
        
        return {
            'aspect_ratio': aspect_ratio,
            'orientation': {'strike': strike, 'dip': dip, 'plunge': 0},
            'shape_type': shape_type,
            'boundary_surface': boundary_surface,
            'eigenvals': eigenvals,
            'eigenvecs': eigenvecs
        }
    
    def _generate_chamber_mesh(self, chamber_mask: np.ndarray,
                             X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh for chamber boundary visualization"""
        
        # Find boundary using marching cubes (simplified approach)
        from skimage import measure
        
        try:
            # Create isosurface at melt fraction threshold
            vertices, faces, _, _ = measure.marching_cubes(
                chamber_mask.astype(float), level=0.5
            )
            
            # Convert indices to real coordinates
            dx = X[1, 0, 0] - X[0, 0, 0]
            dy = Y[0, 1, 0] - Y[0, 0, 0] 
            dz = Z[0, 0, 1] - Z[0, 0, 0]
            
            # Scale vertices to real coordinates
            vertices[:, 0] = X[0, 0, 0] + vertices[:, 0] * dx
            vertices[:, 1] = Y[0, 0, 0] + vertices[:, 1] * dy
            vertices[:, 2] = Z[0, 0, 0] + vertices[:, 2] * dz
            
            return vertices, faces
            
        except Exception as e:
            self.logger.warning(f"Failed to generate chamber mesh: {e}")
            return np.array([]), np.array([])
    
    def _classify_chamber_type(self, chamber_props: Dict) -> MagmaChamberType:
        """Classify type of magma chamber"""
        
        aspect_ratio = chamber_props['aspect_ratio']
        shape_type = chamber_props['shape_type']
        
        if aspect_ratio < 3 and shape_type in ['spherical', 'ellipsoidal']:
            return MagmaChamberType.MAIN_CHAMBER
        elif aspect_ratio >= 3 and aspect_ratio < 8:
            return MagmaChamberType.SILL_COMPLEX
        elif aspect_ratio >= 8:
            return MagmaChamberType.DYKE_SYSTEM
        else:
            return MagmaChamberType.SECONDARY_CHAMBER
    
    async def _model_thermal_structure(self, melt_model: MeltFractionModel,
                                     chamber_geometries: List[MagmaChamberGeometry],
                                     coordinates: Tuple) -> Dict[str, np.ndarray]:
        """Model 3D thermal structure"""
        
        X, Y, Z = coordinates
        
        # Start with temperature from melt model
        temperature_base = melt_model.temperature_3d.copy()
        
        # Add thermal halos around magma chambers
        thermal_perturbation = np.zeros_like(temperature_base)
        
        for chamber in chamber_geometries:
            center_x, center_y, center_z = chamber.center_coordinates
            
            # Distance from chamber center
            distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2)
            
            # Thermal halo (exponential decay)
            chamber_radius = (chamber.volume_km3 * 3/(4*np.pi))**(1/3)  # Equivalent sphere radius
            halo_extent = chamber_radius * 3  # Halo extends 3x chamber radius
            
            thermal_halo = 200 * np.exp(-distance / halo_extent)  # °C temperature increase
            thermal_perturbation += thermal_halo
        
        temperature_total = temperature_base + thermal_perturbation
        
        # Calculate heat flow
        thermal_diffusivity = 1e-6  # m²/s
        heat_capacity = 1000  # J/kg/K
        density = 2600  # kg/m³
        thermal_conductivity = thermal_diffusivity * heat_capacity * density
        
        # Heat flow = -k * ∇T
        dx = X[1, 0, 0] - X[0, 0, 0]
        dy = Y[0, 1, 0] - Y[0, 0, 0]
        dz = Z[0, 0, 1] - Z[0, 0, 0]
        
        temp_grad_x = np.gradient(temperature_total, dx * 1000, axis=0)
        temp_grad_y = np.gradient(temperature_total, dy * 1000, axis=1)
        temp_grad_z = np.gradient(temperature_total, dz * 1000, axis=2)
        
        heat_flow_magnitude = thermal_conductivity * np.sqrt(
            temp_grad_x**2 + temp_grad_y**2 + temp_grad_z**2
        )
        
        return {
            'temperature': temperature_total,
            'thermal_gradient': np.sqrt(temp_grad_x**2 + temp_grad_y**2 + temp_grad_z**2),
            'heat_flow': heat_flow_magnitude,
            'thermal_perturbation': thermal_perturbation
        }
    
    async def _identify_fluid_pathways(self, melt_model: MeltFractionModel,
                                     seismic_props: SeismicProperties,
                                     coordinates: Tuple) -> List[Dict]:
        """Identify potential fluid migration pathways"""
        
        connectivity = melt_model.melt_connectivity
        X, Y, Z = coordinates
        
        # High connectivity regions are potential pathways
        pathway_threshold = 0.5
        pathway_mask = connectivity > pathway_threshold
        
        # Find connected pathways
        from scipy.ndimage import label
        labeled_pathways, num_pathways = label(pathway_mask)
        
        pathways = []
        
        for pathway_id in range(1, num_pathways + 1):
            pathway_points = labeled_pathways == pathway_id
            
            # Skip small pathways
            pathway_volume = np.sum(pathway_points) * (self.config['spatial_resolution_km'])**3
            if pathway_volume < 0.1:  # 0.1 km³ minimum
                continue
            
            # Analyze pathway geometry
            pathway_indices = np.where(pathway_points)
            pathway_coords = np.column_stack([
                X[pathway_indices], Y[pathway_indices], Z[pathway_indices]
            ])
            
            # Find pathway orientation (PCA)
            centered_coords = pathway_coords - np.mean(pathway_coords, axis=0)
            covariance = np.cov(centered_coords.T)
            eigenvals, eigenvecs = np.linalg.eigh(covariance)
            
            # Primary flow direction
            flow_direction = eigenvecs[:, np.argmax(eigenvals)]
            
            # Pathway properties
            pathways.append({
                'pathway_id': pathway_id,
                'volume_km3': pathway_volume,
                'center_coordinates': tuple(np.mean(pathway_coords, axis=0)),
                'flow_direction': flow_direction,
                'connectivity_strength': float(np.mean(connectivity[pathway_points])),
                'pathway_type': 'conduit' if eigenvals.max() / eigenvals.min() > 10 else 'distributed'
            })
        
        return pathways
    
    async def _calculate_eruption_potential_3d(self, melt_model: MeltFractionModel,
                                             thermal_structure: Dict[str, np.ndarray],
                                             chamber_geometries: List) -> np.ndarray:
        """Calculate 3D eruption potential"""
        
        melt_fraction = melt_model.melt_fraction_3d
        connectivity = melt_model.melt_connectivity
        temperature = thermal_structure['temperature']
        pressure = melt_model.pressure_3d
        volatile_content = melt_model.volatile_fraction_3d
        
        # Multiple factors contribute to eruption potential
        
        # 1. Melt fraction - higher = more likely to erupt
        melt_factor = np.clip(melt_fraction / 0.7, 0, 1)
        
        # 2. Connectivity - need connected melt network
        connectivity_factor = np.clip(connectivity, 0, 1)
        
        # 3. Temperature - hotter = more mobile
        temp_factor = np.clip((temperature - 800) / 400, 0, 1)
        
        # 4. Volatile content - drives eruption
        volatile_factor = np.clip(volatile_content / 0.06, 0, 1)
        
        # 5. Proximity to surface (depth effect)
        depth_coords = melt_model.pressure_3d / 27  # Approximate depth from pressure
        depth_factor = np.exp(-depth_coords / 5)  # Exponential decay with depth
        
        # Combine factors (weighted)
        eruption_potential = (
            0.25 * melt_factor +
            0.25 * connectivity_factor +
            0.20 * temp_factor +
            0.20 * volatile_factor +
            0.10 * depth_factor
        )
        
        return eruption_potential
    
    async def _calculate_stress_perturbations(self, chamber_geometries: List[MagmaChamberGeometry],
                                            coordinates: Tuple) -> np.ndarray:
        """Calculate stress perturbations from magma chamber pressure"""
        
        X, Y, Z = coordinates
        stress_perturbation = np.zeros_like(X)
        
        for chamber in chamber_geometries:
            center_x, center_y, center_z = chamber.center_coordinates
            
            # Distance from chamber center
            r = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2)
            
            # Chamber radius
            chamber_radius = (chamber.volume_km3 * 3/(4*np.pi))**(1/3)
            
            # Pressure source (Mogi model)
            # Stress ∝ ΔV/r³ for r > R
            volume_change = chamber.volume_km3 * 0.1  # 10% volume change
            
            # Avoid singularity at chamber center
            r_safe = np.maximum(r, chamber_radius * 0.5)
            
            stress_from_chamber = volume_change / r_safe**3
            stress_perturbation += stress_from_chamber
        
        return stress_perturbation
    
    async def _quantify_model_uncertainty(self, seismic_props: SeismicProperties,
                                        melt_model: MeltFractionModel,
                                        coordinates: Tuple) -> np.ndarray:
        """Quantify uncertainty in magmatic system model"""
        
        # Sources of uncertainty:
        # 1. Velocity model uncertainty
        # 2. Empirical relationship uncertainty
        # 3. Resolution limitations
        
        X, Y, Z = coordinates
        
        # Velocity uncertainty (from tomography)
        velocity_uncertainty = 0.1  # 10% typical uncertainty
        
        # Resolution-based uncertainty (higher uncertainty away from stations/rays)
        # Simplified: uncertainty increases with depth
        depth = -Z  # Positive depth
        resolution_uncertainty = 0.05 + 0.01 * depth  # 5% + 1% per km depth
        
        # Empirical relationship uncertainty
        empirical_uncertainty = 0.2  # 20% uncertainty in melt fraction relationships
        
        # Combined uncertainty
        total_uncertainty = np.sqrt(
            velocity_uncertainty**2 + 
            resolution_uncertainty**2 + 
            empirical_uncertainty**2
        )
        
        return total_uncertainty
    
    async def _calculate_validation_metrics(self, seismic_props: SeismicProperties,
                                          melt_model: MeltFractionModel,
                                          chamber_geometries: List) -> Dict[str, float]:
        """Calculate model validation metrics"""
        
        metrics = {}
        
        # Velocity-melt fraction consistency
        vp_anomaly = (6.5 - seismic_props.vp_field) / 6.5
        expected_anomaly = melt_model.melt_fraction_3d * 0.15  # 15% reduction per unit melt fraction
        
        velocity_consistency = np.corrcoef(vp_anomaly.flatten(), expected_anomaly.flatten())[0, 1]
        metrics['velocity_melt_consistency'] = float(velocity_consistency)
        
        # Physical realism checks
        temp_reasonable = np.sum((melt_model.temperature_3d > 600) & 
                                (melt_model.temperature_3d < 1200)) / melt_model.temperature_3d.size
        metrics['temperature_realism'] = float(temp_reasonable)
        
        # Melt fraction distribution
        melt_distribution = np.mean(melt_model.melt_fraction_3d)
        metrics['mean_melt_fraction'] = float(melt_distribution)
        
        # Chamber count and total volume
        metrics['num_chambers'] = len(chamber_geometries)
        metrics['total_chamber_volume_km3'] = sum(c.volume_km3 for c in chamber_geometries)
        
        # Overall model quality score
        quality_score = (
            0.3 * velocity_consistency +
            0.3 * temp_reasonable +
            0.2 * min(1.0, len(chamber_geometries) / 3) +  # Prefer 1-3 chambers
            0.2 * (1.0 if 10 <= metrics['total_chamber_volume_km3'] <= 10000 else 0.5)
        )
        metrics['overall_quality_score'] = float(quality_score)
        
        return metrics
    
    def create_3d_visualization(self, magmatic_system: MagmaticSystem,
                              visualization_type: str = 'comprehensive') -> go.Figure:
        """Create 3D visualization of magmatic system"""
        
        if visualization_type == 'comprehensive':
            return self._create_comprehensive_magma_plot(magmatic_system)
        elif visualization_type == 'melt_fraction':
            return self._create_melt_fraction_plot(magmatic_system)
        elif visualization_type == 'temperature':
            return self._create_temperature_plot(magmatic_system)
        elif visualization_type == 'chambers':
            return self._create_chamber_geometry_plot(magmatic_system)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")
    
    def _create_comprehensive_magma_plot(self, magmatic_system: MagmaticSystem) -> go.Figure:
        """Create comprehensive 3D magmatic system visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=['Melt Fraction', 'Temperature', 'Chamber Geometry', 'Eruption Potential']
        )
        
        X, Y, Z = magmatic_system.coordinates
        
        # Subsample for visualization
        step = max(1, min(X.shape) // 50)
        x_sub = X[::step, ::step, ::step]
        y_sub = Y[::step, ::step, ::step]
        z_sub = Z[::step, ::step, ::step]
        
        # Melt fraction isosurface
        melt_sub = magmatic_system.melt_model.melt_fraction_3d[::step, ::step, ::step]
        fig.add_trace(
            go.Isosurface(
                x=x_sub.flatten(),
                y=y_sub.flatten(),
                z=z_sub.flatten(),
                value=melt_sub.flatten(),
                isomin=0.1,
                isomax=0.6,
                colorscale='Reds',
                name='Melt Zones'
            ),
            row=1, col=1
        )
        
        # Temperature isosurface
        temp_sub = magmatic_system.thermal_structure['temperature'][::step, ::step, ::step]
        fig.add_trace(
            go.Isosurface(
                x=x_sub.flatten(),
                y=y_sub.flatten(), 
                z=z_sub.flatten(),
                value=temp_sub.flatten(),
                isomin=800,
                isomax=1100,
                colorscale='Hot',
                name='Hot Zones'
            ),
            row=1, col=2
        )
        
        # Chamber geometries as mesh
        for i, chamber in enumerate(magmatic_system.chamber_geometries[:3]):  # Max 3 chambers
            if len(chamber.mesh_vertices) > 0:
                fig.add_trace(
                    go.Mesh3d(
                        x=chamber.mesh_vertices[:, 0],
                        y=chamber.mesh_vertices[:, 1],
                        z=chamber.mesh_vertices[:, 2],
                        i=chamber.mesh_faces[:, 0],
                        j=chamber.mesh_faces[:, 1],
                        k=chamber.mesh_faces[:, 2],
                        opacity=0.7,
                        name=f'Chamber {i+1}'
                    ),
                    row=2, col=1
                )
        
        # Eruption potential
        erupt_sub = magmatic_system.eruption_potential_3d[::step, ::step, ::step]
        fig.add_trace(
            go.Isosurface(
                x=x_sub.flatten(),
                y=y_sub.flatten(),
                z=z_sub.flatten(),
                value=erupt_sub.flatten(),
                isomin=0.3,
                isomax=0.8,
                colorscale='Plasma',
                name='Eruption Potential'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Comprehensive 3D Magmatic System Model",
            scene=dict(
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                zaxis_title="Depth (km)"
            ),
            showlegend=True
        )
        
        return fig

# Example usage
async def example_3d_magmatic_modeling():
    """Example of 3D magmatic system modeling"""
    
    print("3D Magmatic System Modeling Example")
    
    # Initialize modeler
    modeler = ThreeDMagmaticSystemModeler()
    
    # Create synthetic tomography data
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    z = np.linspace(-15, 0, 30)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Synthetic velocity model with low-velocity zone (magma chamber)
    vp_background = 6.5 + 0.1 * (-Z)  # Increase with depth
    vs_background = vp_background / 1.73
    
    # Add magma chamber (low velocities)
    chamber_center = (0, 0, -8)
    distance = np.sqrt((X - chamber_center[0])**2 + 
                      (Y - chamber_center[1])**2 + 
                      (Z - chamber_center[2])**2)
    
    chamber_mask = distance < 3  # 3 km radius
    vp_model = vp_background.copy()
    vs_model = vs_background.copy()
    
    vp_model[chamber_mask] *= 0.7  # 30% velocity reduction
    vs_model[chamber_mask] *= 0.6  # 40% velocity reduction
    
    # Attenuation
    q_model = np.ones_like(vp_model) * 200
    q_model[chamber_mask] = 50  # Low Q in magma chamber
    
    tomography_data = {
        'coordinates': (X, Y, Z),
        'vp_3d': vp_model,
        'vs_3d': vs_model,
        'attenuation_3d': q_model
    }
    
    # Create magmatic system model
    print("Creating magmatic system model...")
    magmatic_system = await modeler.create_magmatic_system_model(tomography_data)
    
    print(f"Model created successfully!")
    print(f"Number of chambers identified: {len(magmatic_system.chamber_geometries)}")
    print(f"Total chamber volume: {sum(c.volume_km3 for c in magmatic_system.chamber_geometries):.1f} km³")
    print(f"Mean melt fraction: {np.mean(magmatic_system.melt_model.melt_fraction_3d):.3f}")
    print(f"Max temperature: {np.max(magmatic_system.thermal_structure['temperature']):.0f}°C")
    print(f"Model quality score: {magmatic_system.validation_metrics['overall_quality_score']:.2f}")
    
    # Create visualization
    print("Creating visualization...")
    fig = modeler.create_3d_visualization(magmatic_system, 'comprehensive')
    
    # Save visualization
    output_path = "3d_magmatic_system_model.html"
    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")
    
    print("3D magmatic system modeling completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_3d_magmatic_modeling())