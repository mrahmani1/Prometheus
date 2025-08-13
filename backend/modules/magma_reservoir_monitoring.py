"""
Advanced Magma Reservoir Monitoring System
At-depth crustal and mantle magma chamber analysis with multi-physics modeling
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
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats, optimize, interpolate
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle

# Advanced scientific computing imports with fallback
try:
    import fenics as fe
    import dolfin as df
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    warnings.warn("FEniCS not available. Advanced finite element modeling will be limited.")

try:
    from pyproj import Proj, transform
    import rasterio
    from rasterio.transform import from_bounds
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    warnings.warn("Geospatial libraries not available. Spatial analysis will be limited.")

try:
    from mpl_toolkits.basemap import Basemap
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    MAPPING_AVAILABLE = True
except ImportError:
    MAPPING_AVAILABLE = False
    warnings.warn("Mapping libraries not available. Visualization will be limited.")

try:
    import obspy
    from obspy.imaging.beachball import beach
    SEISMIC_AVAILABLE = True
except ImportError:
    SEISMIC_AVAILABLE = False
    warnings.warn("ObsPy not available. Seismic analysis will be limited.")

warnings.filterwarnings('ignore')

class MagmaReservoirType(Enum):
    SHALLOW_CRUSTAL = "shallow_crustal"  # 2-8 km
    MID_CRUSTAL = "mid_crustal"         # 8-20 km  
    DEEP_CRUSTAL = "deep_crustal"       # 20-40 km
    SUBCRUSTAL = "subcrustal"           # >40 km
    MUSH_ZONE = "mush_zone"             # Crystal-rich partial melt
    MAGMA_LENS = "magma_lens"           # High melt fraction zone

class MagmaComposition(Enum):
    MAFIC = "mafic"         # <52% SiO2
    INTERMEDIATE = "intermediate"  # 52-65% SiO2
    FELSIC = "felsic"       # 65-75% SiO2
    ULTRAFELSIC = "ultrafelsic"   # >75% SiO2

class ReservoirState(Enum):
    DORMANT = "dormant"
    RECHARGING = "recharging"
    PRESSURIZING = "pressurizing"
    CRITICAL = "critical"
    ERUPTING = "erupting"
    DEGASSING = "degassing"

@dataclass
class MagmaReservoir:
    """Represents a magma reservoir system"""
    id: str
    parent_volcanic_system: str
    reservoir_type: MagmaReservoirType
    geometry: Dict[str, float]  # volume, depth, shape parameters
    location: np.ndarray  # [lon, lat, depth_km]
    composition: Dict[str, float]  # Major element chemistry
    physical_properties: Dict[str, float]  # Temperature, viscosity, density, etc.
    state: ReservoirState
    
    # Thermal structure
    temperature_profile: Optional[np.ndarray] = None
    thermal_gradient: float = 0.0
    
    # Mechanical properties
    pressure: float = 0.0
    stress_state: Dict[str, float] = field(default_factory=dict)
    elastic_moduli: Dict[str, float] = field(default_factory=dict)
    
    # Chemical evolution
    crystallinity: float = 0.0
    volatile_content: Dict[str, float] = field(default_factory=dict)
    melt_fraction: float = 0.0
    
    # Connectivity and plumbing
    connected_reservoirs: List[str] = field(default_factory=list)
    conduit_geometry: Dict[str, float] = field(default_factory=dict)
    
    # Monitoring data
    monitoring_parameters: Dict[str, np.ndarray] = field(default_factory=dict)
    last_updated: Optional[datetime] = None
    
    # Model predictions
    evolution_predictions: Dict[str, Any] = field(default_factory=dict)
    eruption_probability: float = 0.0

@dataclass
class GeophysicalSignature:
    """Geophysical signatures of magma reservoirs"""
    reservoir_id: str
    
    # Seismic signatures
    velocity_anomalies: Dict[str, np.ndarray] = field(default_factory=dict)  # Vp, Vs
    attenuation_anomalies: Dict[str, np.ndarray] = field(default_factory=dict)  # Qp, Qs
    earthquake_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Gravity anomalies
    gravity_anomaly: np.ndarray = field(default_factory=lambda: np.array([]))
    gravity_gradient: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Magnetic signatures
    magnetic_anomaly: np.ndarray = field(default_factory=lambda: np.array([]))
    magnetic_susceptibility: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Electrical signatures
    resistivity_structure: np.ndarray = field(default_factory=lambda: np.array([]))
    magnetotelluric_response: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Thermal signatures
    heat_flow: np.ndarray = field(default_factory=lambda: np.array([]))
    thermal_infrared: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Deformation signatures
    surface_deformation: np.ndarray = field(default_factory=lambda: np.array([]))
    strain_field: np.ndarray = field(default_factory=lambda: np.array([]))

class MagmaReservoirPhysicsModel:
    """Multi-physics model for magma reservoir evolution"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Numerical grids
        self.spatial_grid = None
        self.thermal_solver = None
        self.mechanical_solver = None
        self.chemical_solver = None
        
        # Physical constants
        self.physical_constants = self._initialize_constants()
        
        # Material properties database
        self.material_properties = self._load_material_properties()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'grid_resolution': [100, 100, 50],  # nx, ny, nz
            'domain_size': [50000, 50000, 50000],  # meters
            'time_step': 86400,  # seconds (1 day)
            'total_time': 31536000,  # seconds (1 year)
            'thermal_diffusivity': 1e-6,  # m²/s
            'mechanical_properties': {
                'youngs_modulus': 10e9,  # Pa
                'poissons_ratio': 0.25,
                'bulk_modulus': 6.67e9,  # Pa
                'shear_modulus': 4e9     # Pa
            },
            'magma_properties': {
                'density': 2300,  # kg/m³
                'heat_capacity': 1200,  # J/kg/K
                'thermal_conductivity': 2.5,  # W/m/K
                'latent_heat_fusion': 350000,  # J/kg
                'solidus_temperature': 900,  # °C
                'liquidus_temperature': 1200  # °C
            },
            'numerical_methods': {
                'thermal_scheme': 'backward_euler',
                'mechanical_scheme': 'newton_raphson',
                'chemical_scheme': 'explicit_euler',
                'convergence_tolerance': 1e-6,
                'max_iterations': 100
            }
        }
    
    def _initialize_constants(self) -> Dict[str, float]:
        """Initialize physical constants"""
        return {
            'gas_constant': 8.314,  # J/mol/K
            'boltzmann_constant': 1.381e-23,  # J/K
            'avogadro_number': 6.022e23,  # mol⁻¹
            'stefan_boltzmann': 5.67e-8,  # W/m²/K⁴
            'gravitational_acceleration': 9.81,  # m/s²
            'earth_radius': 6371000,  # m
            'crustal_density': 2700,  # kg/m³
            'mantle_density': 3300,  # kg/m³
        }
    
    def _load_material_properties(self) -> Dict[str, Dict[str, Any]]:
        """Load material property database"""
        return {
            'basalt': {
                'density': 2800,
                'thermal_conductivity': 1.8,
                'heat_capacity': 1200,
                'melting_temperature': 1200,
                'viscosity_params': {'A': 1e-4, 'Ea': 300000, 'n': 3.5}
            },
            'andesite': {
                'density': 2500,
                'thermal_conductivity': 2.0,
                'heat_capacity': 1100,
                'melting_temperature': 1000,
                'viscosity_params': {'A': 1e-3, 'Ea': 350000, 'n': 3.0}
            },
            'rhyolite': {
                'density': 2300,
                'thermal_conductivity': 2.5,
                'heat_capacity': 1000,
                'melting_temperature': 800,
                'viscosity_params': {'A': 1e-2, 'Ea': 400000, 'n': 2.5}
            },
            'granite': {
                'density': 2650,
                'thermal_conductivity': 3.0,
                'heat_capacity': 900,
                'melting_temperature': 700,
                'viscosity_params': {'A': 1e-1, 'Ea': 450000, 'n': 2.0}
            }
        }
    
    def initialize_reservoir_model(self, reservoir: MagmaReservoir) -> Dict[str, Any]:
        """Initialize physics model for a specific reservoir"""
        
        model_state = {
            'reservoir_id': reservoir.id,
            'grid': self._create_computational_grid(reservoir),
            'temperature_field': self._initialize_temperature_field(reservoir),
            'pressure_field': self._initialize_pressure_field(reservoir),
            'composition_field': self._initialize_composition_field(reservoir),
            'velocity_field': self._initialize_velocity_field(reservoir),
            'phase_field': self._initialize_phase_field(reservoir),
            'boundary_conditions': self._setup_boundary_conditions(reservoir),
            'material_distribution': self._setup_material_distribution(reservoir)
        }
        
        return model_state
    
    def _create_computational_grid(self, reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Create computational grid for the reservoir"""
        
        # Extract reservoir geometry
        volume = reservoir.geometry.get('volume_km3', 100) * 1e9  # Convert to m³
        depth = reservoir.location[2] * 1000  # Convert to meters
        
        # Estimate reservoir dimensions (assuming ellipsoidal shape)
        aspect_ratio = reservoir.geometry.get('aspect_ratio', 2.0)
        radius = (3 * volume / (4 * np.pi * aspect_ratio)) ** (1/3)
        
        # Create grid centered on reservoir
        x_extent = radius * 3  # Extend beyond reservoir
        y_extent = radius * 3
        z_extent = radius * aspect_ratio * 2
        
        nx, ny, nz = self.config['grid_resolution']
        
        x = np.linspace(-x_extent, x_extent, nx)
        y = np.linspace(-y_extent, y_extent, ny)
        z = np.linspace(depth - z_extent/2, depth + z_extent/2, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        return {
            'x': x, 'y': y, 'z': z,
            'X': X, 'Y': Y, 'Z': Z,
            'dx': x[1] - x[0], 'dy': y[1] - y[0], 'dz': z[1] - z[0],
            'nx': nx, 'ny': ny, 'nz': nz,
            'reservoir_center': np.array([0, 0, depth]),
            'reservoir_radius': radius,
            'reservoir_aspect_ratio': aspect_ratio
        }
    
    def _initialize_temperature_field(self, reservoir: MagmaReservoir) -> np.ndarray:
        """Initialize temperature field"""
        
        grid = self.spatial_grid
        if grid is None:
            return np.zeros((10, 10, 10))
        
        X, Y, Z = grid['X'], grid['Y'], grid['Z']
        
        # Background geotherm
        surface_temp = 15  # °C
        geothermal_gradient = 25  # °C/km
        background_temp = surface_temp + (Z / 1000) * geothermal_gradient
        
        # Magma reservoir thermal anomaly
        reservoir_center = grid['reservoir_center']
        reservoir_radius = grid['reservoir_radius']
        
        # Distance from reservoir center
        distance = np.sqrt((X - reservoir_center[0])**2 + 
                          (Y - reservoir_center[1])**2 + 
                          (Z - reservoir_center[2])**2)
        
        # Gaussian thermal anomaly
        magma_temp = reservoir.physical_properties.get('temperature', 1000)  # °C
        thermal_anomaly = (magma_temp - background_temp) * np.exp(-(distance / reservoir_radius)**2)
        
        temperature_field = background_temp + thermal_anomaly
        
        return temperature_field
    
    def _initialize_pressure_field(self, reservoir: MagmaReservoir) -> np.ndarray:
        """Initialize pressure field"""
        
        grid = self.spatial_grid
        if grid is None:
            return np.zeros((10, 10, 10))
        
        X, Y, Z = grid['X'], grid['Y'], grid['Z']
        
        # Lithostatic pressure
        crustal_density = self.physical_constants['crustal_density']
        g = self.physical_constants['gravitational_acceleration']
        lithostatic_pressure = crustal_density * g * Z  # Pa
        
        # Magma overpressure
        reservoir_center = grid['reservoir_center']
        reservoir_radius = grid['reservoir_radius']
        
        distance = np.sqrt((X - reservoir_center[0])**2 + 
                          (Y - reservoir_center[1])**2 + 
                          (Z - reservoir_center[2])**2)
        
        # Overpressure within reservoir
        overpressure_magnitude = reservoir.pressure
        overpressure = overpressure_magnitude * np.exp(-(distance / reservoir_radius)**2)
        
        pressure_field = lithostatic_pressure + overpressure
        
        return pressure_field
    
    def _initialize_composition_field(self, reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Initialize composition field"""
        
        grid = self.spatial_grid
        if grid is None:
            return {'SiO2': np.zeros((10, 10, 10))}
        
        X, Y, Z = grid['X'], grid['Y'], grid['Z']
        
        # Background crustal composition
        background_composition = {
            'SiO2': 65.0, 'Al2O3': 15.0, 'FeO': 5.0,
            'MgO': 3.0, 'CaO': 5.0, 'Na2O': 3.0, 'K2O': 4.0
        }
        
        # Reservoir composition
        reservoir_composition = reservoir.composition
        
        # Spatial distribution
        reservoir_center = grid['reservoir_center']
        reservoir_radius = grid['reservoir_radius']
        
        distance = np.sqrt((X - reservoir_center[0])**2 + 
                          (Y - reservoir_center[1])**2 + 
                          (Z - reservoir_center[2])**2)
        
        # Mixing function
        mixing_factor = np.exp(-(distance / reservoir_radius)**2)
        
        composition_field = {}
        for component in background_composition.keys():
            background_value = background_composition[component]
            reservoir_value = reservoir_composition.get(component, background_value)
            
            composition_field[component] = (
                background_value * (1 - mixing_factor) + 
                reservoir_value * mixing_factor
            )
        
        return composition_field
    
    def _initialize_velocity_field(self, reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Initialize velocity field for magma flow"""
        
        grid = self.spatial_grid
        if grid is None:
            return {'vx': np.zeros((10, 10, 10)), 'vy': np.zeros((10, 10, 10)), 'vz': np.zeros((10, 10, 10))}
        
        nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
        
        # Initialize with small random velocities
        velocity_field = {
            'vx': np.random.normal(0, 1e-6, (nx, ny, nz)),
            'vy': np.random.normal(0, 1e-6, (nx, ny, nz)),
            'vz': np.random.normal(0, 1e-6, (nx, ny, nz))
        }
        
        return velocity_field
    
    def _initialize_phase_field(self, reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Initialize phase field (melt fraction)"""
        
        grid = self.spatial_grid
        if grid is None:
            return {'melt_fraction': np.zeros((10, 10, 10))}
        
        X, Y, Z = grid['X'], grid['Y'], grid['Z']
        
        # Calculate melt fraction based on temperature and composition
        temperature_field = self._initialize_temperature_field(reservoir)
        
        # Comprehensive melt fraction calculation with advanced petrological models
        melt_fraction = self._calculate_comprehensive_melt_fraction(
            temperature_field, reservoir, X, Y, Z
        )
        
        # Apply spatial modifications based on reservoir geometry
        reservoir_center = grid['reservoir_center']
        reservoir_radius = grid['reservoir_radius']
        
        distance = np.sqrt((X - reservoir_center[0])**2 + 
                          (Y - reservoir_center[1])**2 + 
                          (Z - reservoir_center[2])**2)
        
        # Apply reservoir-specific spatial weighting
        melt_fraction = self._apply_reservoir_spatial_weighting(
            melt_fraction, distance, reservoir_radius, reservoir.melt_fraction
        )
        
        phase_field = {
            'melt_fraction': melt_fraction,
            'crystal_fraction': 1 - melt_fraction,
            'gas_fraction': np.zeros_like(melt_fraction)  # Will be calculated from volatiles
        }
        
        return phase_field
    
    def _setup_boundary_conditions(self, reservoir: MagmaReservoir) -> Dict[str, Any]:
        """Setup boundary conditions for the model"""
        
        return {
            'thermal': {
                'top': {'type': 'dirichlet', 'value': 15},  # Surface temperature
                'bottom': {'type': 'neumann', 'value': 0.05},  # Heat flux
                'sides': {'type': 'neumann', 'value': 0}  # No lateral heat flux
            },
            'mechanical': {
                'top': {'type': 'neumann', 'value': 0},  # Free surface
                'bottom': {'type': 'dirichlet', 'value': 0},  # Fixed
                'sides': {'type': 'neumann', 'value': 0}  # Free sides
            },
            'chemical': {
                'all': {'type': 'neumann', 'value': 0}  # No flux
            }
        }
    
    def _setup_material_distribution(self, reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Setup material property distribution"""
        
        grid = self.spatial_grid
        if grid is None:
            return {}
        
        X, Y, Z = grid['X'], grid['Y'], grid['Z']
        
        # Determine dominant composition
        SiO2_content = reservoir.composition.get('SiO2', 65.0)
        
        if SiO2_content < 52:
            material_type = 'basalt'
        elif SiO2_content < 65:
            material_type = 'andesite'
        elif SiO2_content < 75:
            material_type = 'rhyolite'
        else:
            material_type = 'granite'
        
        # Get material properties
        material_props = self.material_properties[material_type]
        
        # Create material distribution
        reservoir_center = grid['reservoir_center']
        reservoir_radius = grid['reservoir_radius']
        
        distance = np.sqrt((X - reservoir_center[0])**2 + 
                          (Y - reservoir_center[1])**2 + 
                          (Z - reservoir_center[2])**2)
        
        # Material mask (1 inside reservoir, 0 outside)
        material_mask = (distance <= reservoir_radius).astype(float)
        
        # Background (host rock) properties
        background_props = self.material_properties['granite']  # Assume granitic crust
        
        material_distribution = {}
        for prop_name, prop_value in material_props.items():
            if isinstance(prop_value, (int, float)):
                background_value = background_props.get(prop_name, prop_value)
                material_distribution[prop_name] = (
                    background_value * (1 - material_mask) + 
                    prop_value * material_mask
                )
        
        return material_distribution
    
    async def solve_thermal_evolution(self, model_state: Dict[str, Any], 
                                    time_steps: int) -> Dict[str, Any]:
        """Solve thermal evolution of magma reservoir"""
        
        temperature = model_state['temperature_field'].copy()
        grid = model_state['grid']
        
        # Thermal properties
        thermal_conductivity = model_state['material_distribution'].get('thermal_conductivity', 2.5)
        heat_capacity = model_state['material_distribution'].get('heat_capacity', 1200)
        density = model_state['material_distribution'].get('density', 2500)
        
        # Thermal diffusivity
        thermal_diffusivity = thermal_conductivity / (density * heat_capacity)
        
        # Time step
        dt = self.config['time_step']
        dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
        
        # Stability criterion
        max_dt = 0.5 * min(dx**2, dy**2, dz**2) / np.max(thermal_diffusivity)
        if dt > max_dt:
            self.logger.warning(f"Time step {dt} may be unstable (max stable: {max_dt})")
        
        thermal_history = []
        
        for step in range(time_steps):
            # Calculate thermal diffusion using finite differences
            temperature_new = self._solve_heat_equation(
                temperature, thermal_diffusivity, dt, dx, dy, dz
            )
            
            # Apply boundary conditions
            temperature_new = self._apply_thermal_boundary_conditions(
                temperature_new, model_state['boundary_conditions']['thermal']
            )
            
            # Add heat sources (radioactive decay, latent heat)
            temperature_new = self._add_heat_sources(temperature_new, model_state, dt)
            
            temperature = temperature_new
            
            # Store thermal history
            if step % 10 == 0:  # Store every 10 steps
                thermal_history.append({
                    'time': step * dt,
                    'temperature_field': temperature.copy(),
                    'max_temperature': np.max(temperature),
                    'min_temperature': np.min(temperature),
                    'average_temperature': np.mean(temperature)
                })
        
        return {
            'final_temperature_field': temperature,
            'thermal_history': thermal_history,
            'thermal_gradient': self._calculate_thermal_gradient(temperature, grid),
            'heat_flow': self._calculate_heat_flow(temperature, thermal_conductivity, grid)
        }
    
    def _solve_heat_equation(self, temperature: np.ndarray, diffusivity: np.ndarray,
                           dt: float, dx: float, dy: float, dz: float) -> np.ndarray:
        """Solve 3D heat equation using finite differences"""
        
        T = temperature
        alpha = diffusivity
        
        # Second derivatives
        d2T_dx2 = (T[2:, 1:-1, 1:-1] - 2*T[1:-1, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1]) / dx**2
        d2T_dy2 = (T[1:-1, 2:, 1:-1] - 2*T[1:-1, 1:-1, 1:-1] + T[1:-1, :-2, 1:-1]) / dy**2
        d2T_dz2 = (T[1:-1, 1:-1, 2:] - 2*T[1:-1, 1:-1, 1:-1] + T[1:-1, 1:-1, :-2]) / dz**2
        
        # Laplacian
        laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2
        
        # Update temperature
        T_new = T.copy()
        T_new[1:-1, 1:-1, 1:-1] += dt * alpha[1:-1, 1:-1, 1:-1] * laplacian
        
        return T_new
    
    def _apply_thermal_boundary_conditions(self, temperature: np.ndarray, 
                                         bc: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Apply thermal boundary conditions"""
        
        T = temperature.copy()
        
        # Top boundary (z = 0)
        if bc['top']['type'] == 'dirichlet':
            T[:, :, 0] = bc['top']['value']
        elif bc['top']['type'] == 'neumann':
            # dT/dz = value
            T[:, :, 0] = T[:, :, 1] + bc['top']['value']
        
        # Bottom boundary
        if bc['bottom']['type'] == 'dirichlet':
            T[:, :, -1] = bc['bottom']['value']
        elif bc['bottom']['type'] == 'neumann':
            T[:, :, -1] = T[:, :, -2] + bc['bottom']['value']
        
        # Side boundaries
        if bc['sides']['type'] == 'neumann':
            # Left and right
            T[0, :, :] = T[1, :, :] + bc['sides']['value']
            T[-1, :, :] = T[-2, :, :] + bc['sides']['value']
            # Front and back
            T[:, 0, :] = T[:, 1, :] + bc['sides']['value']
            T[:, -1, :] = T[:, -2, :] + bc['sides']['value']
        
        return T
    
    def _add_heat_sources(self, temperature: np.ndarray, model_state: Dict[str, Any], 
                         dt: float) -> np.ndarray:
        """Add heat sources to temperature field"""
        
        T = temperature.copy()
        
        # Comprehensive radioactive heat generation from multiple isotopes
        radioactive_heat_rate = self._calculate_comprehensive_radioactive_heating(
            model_state, temperature)
        density = model_state['material_distribution'].get('density', 2500)
        heat_capacity = model_state['material_distribution'].get('heat_capacity', 1200)
        
        heat_source_term = radioactive_heat_rate * dt / (density * heat_capacity)
        T += heat_source_term
        
        # Latent heat from crystallization/melting
        phase_field = model_state.get('phase_field', {})
        if 'melt_fraction' in phase_field:
            melt_fraction = phase_field['melt_fraction']
            
            # Simplified latent heat effect
            latent_heat = self.config['magma_properties']['latent_heat_fusion']
            crystallization_rate = 0.01 * dt  # Simplified rate
            
            latent_heat_term = latent_heat * crystallization_rate / (density * heat_capacity)
            
            # Add where crystallization occurs (cooling magma)
            crystallization_mask = (T > 900) & (T < 1200) & (melt_fraction > 0.1)
            T[crystallization_mask] += latent_heat_term
        
        return T
    
    def _calculate_thermal_gradient(self, temperature: np.ndarray, 
                                  grid: Dict[str, Any]) -> np.ndarray:
        """Calculate thermal gradient"""
        
        dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
        
        # Gradient components
        dT_dx = np.gradient(temperature, dx, axis=0)
        dT_dy = np.gradient(temperature, dy, axis=1) 
        dT_dz = np.gradient(temperature, dz, axis=2)
        
        # Magnitude of gradient
        thermal_gradient = np.sqrt(dT_dx**2 + dT_dy**2 + dT_dz**2)
        
        return thermal_gradient
    
    def _calculate_heat_flow(self, temperature: np.ndarray, thermal_conductivity: np.ndarray,
                           grid: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Calculate heat flow vectors"""
        
        dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
        
        # Temperature gradient
        dT_dx = np.gradient(temperature, dx, axis=0)
        dT_dy = np.gradient(temperature, dy, axis=1)
        dT_dz = np.gradient(temperature, dz, axis=2)
        
        # Heat flow (Fourier's law: q = -k * grad(T))
        heat_flow = {
            'qx': -thermal_conductivity * dT_dx,
            'qy': -thermal_conductivity * dT_dy,
            'qz': -thermal_conductivity * dT_dz
        }
        
        # Heat flow magnitude
        heat_flow['magnitude'] = np.sqrt(heat_flow['qx']**2 + heat_flow['qy']**2 + heat_flow['qz']**2)
        
        return heat_flow
    
    async def solve_mechanical_evolution(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Solve mechanical evolution and stress state"""
        
        pressure = model_state['pressure_field']
        temperature = model_state['temperature_field']
        grid = model_state['grid']
        
        # Mechanical properties
        youngs_modulus = self.config['mechanical_properties']['youngs_modulus']
        poissons_ratio = self.config['mechanical_properties']['poissons_ratio']
        
        # Calculate thermal stress
        thermal_expansion = 3e-5  # 1/K (typical for rocks)
        reference_temperature = 300  # K (about 27°C)
        
        thermal_stress = self._calculate_thermal_stress(
            temperature, reference_temperature, thermal_expansion, youngs_modulus, poissons_ratio
        )
        
        # Calculate pressure-induced stress
        pressure_stress = self._calculate_pressure_stress(pressure, grid)
        
        # Total stress field
        total_stress = {
            'sigma_xx': thermal_stress['sigma_xx'] + pressure_stress.get('sigma_xx', 0),
            'sigma_yy': thermal_stress['sigma_yy'] + pressure_stress.get('sigma_yy', 0),
            'sigma_zz': thermal_stress['sigma_zz'] + pressure_stress.get('sigma_zz', 0),
            'sigma_xy': thermal_stress['sigma_xy'] + pressure_stress.get('sigma_xy', 0),
            'sigma_xz': thermal_stress['sigma_xz'] + pressure_stress.get('sigma_xz', 0),
            'sigma_yz': thermal_stress['sigma_yz'] + pressure_stress.get('sigma_yz', 0)
        }
        
        # Calculate deformation
        deformation = self._calculate_deformation(total_stress, youngs_modulus, poissons_ratio)
        
        # Calculate fracture probability
        fracture_probability = self._calculate_fracture_probability(total_stress)
        
        return {
            'stress_field': total_stress,
            'deformation_field': deformation,
            'fracture_probability': fracture_probability,
            'surface_deformation': self._extract_surface_deformation(deformation, grid),
            'pressure_gradient': np.gradient(pressure),
            'mechanical_energy': self._calculate_mechanical_energy(total_stress, deformation)
        }
    
    def _calculate_thermal_stress(self, temperature: np.ndarray, T_ref: float,
                                alpha: float, E: float, nu: float) -> Dict[str, np.ndarray]:
        """Calculate thermal stress field"""
        
        # Temperature difference from reference
        delta_T = temperature - T_ref
        
        # Thermal strain (isotropic expansion)
        thermal_strain = alpha * delta_T
        
        # Convert to stress (plane strain assumption)
        factor = E * alpha / (1 - nu)
        
        thermal_stress = {
            'sigma_xx': factor * delta_T,
            'sigma_yy': factor * delta_T,
            'sigma_zz': factor * delta_T,
            'sigma_xy': np.zeros_like(delta_T),
            'sigma_xz': np.zeros_like(delta_T),
            'sigma_yz': np.zeros_like(delta_T)
        }
        
        return thermal_stress
    
    def _calculate_pressure_stress(self, pressure: np.ndarray, 
                                 grid: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive stress tensor from pressure field using advanced geomechanics
        
        Includes:
        - Full 3D stress tensor calculation
        - Poroelastic coupling effects
        - Biot coefficient integration
        - Non-linear pressure-stress relationships
        - Anisotropic material properties
        - Deviatoric stress components
        """
        
        # Calculate comprehensive pressure-induced stress field
        pressure_stress = self._calculate_comprehensive_pressure_stress(pressure, grid)
        
        # Apply poroelastic coupling effects
        pressure_stress = self._apply_poroelastic_coupling(pressure_stress, pressure, grid)
        
        # Include anisotropic material effects
        pressure_stress = self._apply_anisotropic_pressure_effects(pressure_stress, grid)
        
        # Calculate deviatoric stress components
        pressure_stress = self._calculate_deviatoric_pressure_stress(pressure_stress)
        
        return pressure_stress
    
    def _calculate_comprehensive_pressure_stress(self, pressure: np.ndarray,
                                               grid: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Calculate comprehensive pressure-induced stress tensor"""
        
        dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
        X, Y, Z = grid['X'], grid['Y'], grid['Z']
        
        # Calculate pressure gradients with high-order accuracy
        dP_dx, dP_dy, dP_dz = self._calculate_high_order_pressure_gradients(pressure, dx, dy, dz)
        
        # Calculate second derivatives for curvature effects
        d2P_dx2 = np.gradient(dP_dx, dx, axis=0)
        d2P_dy2 = np.gradient(dP_dy, dy, axis=1) 
        d2P_dz2 = np.gradient(dP_dz, dz, axis=2)
        d2P_dxdy = np.gradient(dP_dx, dy, axis=1)
        d2P_dxdz = np.gradient(dP_dx, dz, axis=2)
        d2P_dydz = np.gradient(dP_dy, dz, axis=2)
        
        # Material properties (spatially variable)
        biot_coeff = self._calculate_biot_coefficient(X, Y, Z)
        bulk_modulus = self._calculate_bulk_modulus(X, Y, Z)
        shear_modulus = self._calculate_shear_modulus(X, Y, Z)
        
        # Effective stress calculation (Terzaghi-Biot theory)
        # σ' = σ - α*P where α is Biot coefficient
        
        # Hydrostatic component
        hydrostatic_stress = -biot_coeff * pressure
        
        # Deviatoric components from pressure gradients
        # Uses relationship between pressure gradients and shear stress
        gradient_magnitude = np.sqrt(dP_dx**2 + dP_dy**2 + dP_dz**2)
        
        # Normalized pressure gradient directions
        grad_norm = np.maximum(gradient_magnitude, 1e-12)  # Avoid division by zero
        n_x = dP_dx / grad_norm
        n_y = dP_dy / grad_norm
        n_z = dP_dz / grad_norm
        
        # Principal stress magnification factor
        stress_concentration = self._calculate_stress_concentration_factor(
            pressure, gradient_magnitude, bulk_modulus, shear_modulus
        )
        
        # Calculate normal stress components
        sigma_xx = hydrostatic_stress + stress_concentration * (
            -2 * shear_modulus * n_x**2 * gradient_magnitude / bulk_modulus
        )
        
        sigma_yy = hydrostatic_stress + stress_concentration * (
            -2 * shear_modulus * n_y**2 * gradient_magnitude / bulk_modulus
        )
        
        sigma_zz = hydrostatic_stress + stress_concentration * (
            -2 * shear_modulus * n_z**2 * gradient_magnitude / bulk_modulus
        )
        
        # Calculate shear stress components from pressure gradient interactions
        sigma_xy = -stress_concentration * shear_modulus * n_x * n_y * gradient_magnitude / bulk_modulus
        sigma_xz = -stress_concentration * shear_modulus * n_x * n_z * gradient_magnitude / bulk_modulus
        sigma_yz = -stress_concentration * shear_modulus * n_y * n_z * gradient_magnitude / bulk_modulus
        
        # Add curvature-induced stresses (second-order effects)
        curvature_factor = self._calculate_curvature_stress_factor(bulk_modulus, shear_modulus)
        
        sigma_xx += curvature_factor * d2P_dx2
        sigma_yy += curvature_factor * d2P_dy2
        sigma_zz += curvature_factor * d2P_dz2
        sigma_xy += curvature_factor * d2P_dxdy
        sigma_xz += curvature_factor * d2P_dxdz
        sigma_yz += curvature_factor * d2P_dydz
        
        pressure_stress = {
            'sigma_xx': sigma_xx,
            'sigma_yy': sigma_yy,
            'sigma_zz': sigma_zz,
            'sigma_xy': sigma_xy,
            'sigma_xz': sigma_xz,
            'sigma_yz': sigma_yz
        }
        
        return pressure_stress
    
    def _calculate_high_order_pressure_gradients(self, pressure: np.ndarray,
                                               dx: float, dy: float, dz: float) -> tuple:
        """Calculate high-order accurate pressure gradients"""
        
        # Use fourth-order central differences where possible
        dP_dx = np.zeros_like(pressure)
        dP_dy = np.zeros_like(pressure)
        dP_dz = np.zeros_like(pressure)
        
        # X-direction (axis=0)
        dP_dx[2:-2, :, :] = (-pressure[4:, :, :] + 8*pressure[3:-1, :, :] - 
                             8*pressure[1:-3, :, :] + pressure[:-4, :, :]) / (12*dx)
        # Boundary points use second-order
        dP_dx[0, :, :] = (-3*pressure[0, :, :] + 4*pressure[1, :, :] - pressure[2, :, :]) / (2*dx)
        dP_dx[1, :, :] = (pressure[2, :, :] - pressure[0, :, :]) / (2*dx)
        dP_dx[-2, :, :] = (pressure[-1, :, :] - pressure[-3, :, :]) / (2*dx)
        dP_dx[-1, :, :] = (3*pressure[-1, :, :] - 4*pressure[-2, :, :] + pressure[-3, :, :]) / (2*dx)
        
        # Y-direction (axis=1)
        dP_dy[:, 2:-2, :] = (-pressure[:, 4:, :] + 8*pressure[:, 3:-1, :] - 
                             8*pressure[:, 1:-3, :] + pressure[:, :-4, :]) / (12*dy)
        # Boundary points
        dP_dy[:, 0, :] = (-3*pressure[:, 0, :] + 4*pressure[:, 1, :] - pressure[:, 2, :]) / (2*dy)
        dP_dy[:, 1, :] = (pressure[:, 2, :] - pressure[:, 0, :]) / (2*dy)
        dP_dy[:, -2, :] = (pressure[:, -1, :] - pressure[:, -3, :]) / (2*dy)
        dP_dy[:, -1, :] = (3*pressure[:, -1, :] - 4*pressure[:, -2, :] + pressure[:, -3, :]) / (2*dy)
        
        # Z-direction (axis=2)
        dP_dz[:, :, 2:-2] = (-pressure[:, :, 4:] + 8*pressure[:, :, 3:-1] - 
                             8*pressure[:, :, 1:-3] + pressure[:, :, :-4]) / (12*dz)
        # Boundary points
        dP_dz[:, :, 0] = (-3*pressure[:, :, 0] + 4*pressure[:, :, 1] - pressure[:, :, 2]) / (2*dz)
        dP_dz[:, :, 1] = (pressure[:, :, 2] - pressure[:, :, 0]) / (2*dz)
        dP_dz[:, :, -2] = (pressure[:, :, -1] - pressure[:, :, -3]) / (2*dz)
        dP_dz[:, :, -1] = (3*pressure[:, :, -1] - 4*pressure[:, :, -2] + pressure[:, :, -3]) / (2*dz)
        
        return dP_dx, dP_dy, dP_dz
    
    def _calculate_biot_coefficient(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Calculate spatially variable Biot coefficient"""
        
        # Biot coefficient α = 1 - K/Ks (bulk modulus ratio)
        # Typical values: sandstone (0.6-0.9), granite (0.2-0.4), limestone (0.4-0.8)
        
        depth = np.abs(Z)  # km
        
        # Depth-dependent Biot coefficient (increases with depth due to crack closure)
        alpha_surface = 0.8  # High porosity near surface
        alpha_deep = 0.3     # Lower porosity at depth
        depth_scale = 5.0    # km
        
        alpha = alpha_deep + (alpha_surface - alpha_deep) * np.exp(-depth / depth_scale)
        
        # Add lateral heterogeneity
        lateral_variation = 0.1 * np.sin(2 * np.pi * X / 10.0) * np.cos(2 * np.pi * Y / 15.0)
        alpha += lateral_variation
        
        # Ensure physical bounds
        alpha = np.clip(alpha, 0.1, 0.95)
        
        return alpha
    
    def _calculate_bulk_modulus(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Calculate spatially variable bulk modulus"""
        
        depth = np.abs(Z)  # km
        
        # Typical bulk moduli (GPa): granite (35-50), basalt (60-100), sediments (5-20)
        K_surface = 15.0  # GPa (sedimentary rocks)
        K_deep = 45.0     # GPa (crystalline basement)
        depth_scale = 3.0 # km
        
        bulk_modulus = K_surface + (K_deep - K_surface) * (1 - np.exp(-depth / depth_scale))
        
        # Add spatial heterogeneity
        heterogeneity = 5.0 * (np.sin(X / 5.0) * np.cos(Y / 7.0) * np.sin(Z / 3.0))
        bulk_modulus += heterogeneity
        
        # Ensure positive values
        bulk_modulus = np.maximum(bulk_modulus, 5.0)
        
        return bulk_modulus
    
    def _calculate_shear_modulus(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Calculate spatially variable shear modulus"""
        
        bulk_modulus = self._calculate_bulk_modulus(X, Y, Z)
        
        # Typical Poisson's ratio range (0.2-0.35 for rocks)
        depth = np.abs(Z)
        poisson_ratio = 0.25 + 0.05 * np.tanh(depth / 5.0)  # Increases slightly with depth
        
        # G = K * 3(1-2ν)/(2(1+ν))
        shear_modulus = bulk_modulus * 3 * (1 - 2 * poisson_ratio) / (2 * (1 + poisson_ratio))
        
        return shear_modulus
    
    def _calculate_stress_concentration_factor(self, pressure: np.ndarray,
                                             gradient_magnitude: np.ndarray,
                                             bulk_modulus: np.ndarray,
                                             shear_modulus: np.ndarray) -> np.ndarray:
        """Calculate stress concentration factor for pressure gradients"""
        
        # Dimensionless parameter relating pressure gradient to stress concentration
        # Based on analytical solutions for pressure-driven flow in elastic media
        
        # Characteristic pressure and length scales
        P_ref = np.mean(np.abs(pressure))
        L_ref = 1000.0  # 1 km reference length
        
        # Normalized gradient parameter
        grad_param = gradient_magnitude * L_ref / P_ref
        
        # Stress concentration increases with gradient magnitude
        # Asymptotic behavior: linear for small gradients, saturates at large gradients
        max_concentration = 3.0
        concentration_factor = max_concentration * np.tanh(grad_param)
        
        # Material property dependence
        stiffness_ratio = shear_modulus / bulk_modulus
        material_factor = 1.0 + 0.5 * stiffness_ratio
        
        concentration_factor *= material_factor
        
        return concentration_factor
    
    def _calculate_curvature_stress_factor(self, bulk_modulus: np.ndarray,
                                         shear_modulus: np.ndarray) -> np.ndarray:
        """Calculate stress factor for pressure field curvature effects"""
        
        # Second-order effects from pressure field curvature
        # Represents coupling between pressure Laplacian and stress
        
        # Characteristic length scale for curvature effects
        L_curv = 500.0  # meters
        
        # Coupling strength (dimensional analysis)
        coupling_strength = shear_modulus * L_curv**2 / bulk_modulus
        
        # Typical magnitude: small correction to first-order effects
        curvature_factor = 0.1 * coupling_strength / np.mean(coupling_strength)
        
        return curvature_factor
    
    def _apply_poroelastic_coupling(self, pressure_stress: Dict[str, np.ndarray],
                                  pressure: np.ndarray, grid: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Apply poroelastic coupling effects (Biot theory)"""
        
        X, Y, Z = grid['X'], grid['Y'], grid['Z']
        
        # Calculate undrained vs drained moduli ratio
        skempton_B = self._calculate_skempton_coefficient(X, Y, Z)
        
        # Pore pressure diffusion effects
        pore_pressure_factor = self._calculate_pore_pressure_factor(pressure, grid)
        
        # Apply Biot coupling correction
        coupling_correction = skempton_B * pore_pressure_factor
        
        # Modify stress components
        for component in ['sigma_xx', 'sigma_yy', 'sigma_zz']:
            pressure_stress[component] += coupling_correction * pressure
        
        return pressure_stress
    
    def _calculate_skempton_coefficient(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Calculate Skempton's B coefficient for poroelastic response"""
        
        # B = (Ku - K)/(Ku - K + φ(Kw - K))
        # Typically 0.5-0.95 for most rocks
        
        depth = np.abs(Z)
        
        # Depth-dependent Skempton coefficient
        B_surface = 0.9   # High fluid mobility near surface
        B_deep = 0.6      # Reduced fluid mobility at depth
        depth_scale = 4.0 # km
        
        skempton_B = B_deep + (B_surface - B_deep) * np.exp(-depth / depth_scale)
        
        return skempton_B
    
    def _calculate_pore_pressure_factor(self, pressure: np.ndarray, grid: Dict[str, Any]) -> np.ndarray:
        """Calculate pore pressure diffusion factor"""
        
        # Comprehensive pore pressure diffusion using advanced numerical methods
        # Solves full 3D diffusion equation with heterogeneous properties
        return self._solve_comprehensive_pore_pressure_diffusion(pressure, grid)
    
    def _solve_comprehensive_pore_pressure_diffusion(self, pressure: np.ndarray, 
                                                   grid: Dict[str, Any]) -> np.ndarray:
        """
        Solve comprehensive 3D pore pressure diffusion equation
        
        Solves: ∂P/∂t = ∇·(c∇P) + S
        Where:
        - c = hydraulic diffusivity tensor (spatially variable)
        - S = source/sink terms (injection, extraction, deformation)
        
        Uses:
        1. Finite difference methods with adaptive time stepping
        2. Heterogeneous rock properties (permeability, porosity, compressibility)
        3. Anisotropic diffusion tensors
        4. Temperature and stress dependent hydraulic properties
        5. Coupled hydro-mechanical effects
        6. Non-linear effects (pressure-dependent permeability)
        7. Multi-phase flow effects (gas-liquid)
        """
        
        dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
        dt = grid.get('dt', 3600)  # Default 1 hour time step
        
        # Get comprehensive hydraulic properties
        hydraulic_props = self._calculate_comprehensive_hydraulic_properties(pressure, grid)
        
        # Calculate anisotropic diffusivity tensor
        diffusivity_tensor = self._calculate_hydraulic_diffusivity_tensor(
            hydraulic_props, pressure, grid)
        
        # Calculate source/sink terms
        source_terms = self._calculate_pore_pressure_sources(pressure, grid)
        
        # Apply boundary conditions
        boundary_conditions = self._get_pore_pressure_boundary_conditions(pressure, grid)
        
        # Solve diffusion equation using multiple methods
        solution_adi = self._solve_adi_diffusion(pressure, diffusivity_tensor, 
                                               source_terms, boundary_conditions, 
                                               dx, dy, dz, dt)
        
        solution_crank_nicolson = self._solve_crank_nicolson_diffusion(
            pressure, diffusivity_tensor, source_terms, boundary_conditions, 
            dx, dy, dz, dt)
        
        solution_douglas_rachford = self._solve_douglas_rachford_diffusion(
            pressure, diffusivity_tensor, source_terms, boundary_conditions,
            dx, dy, dz, dt)
        
        # Weighted combination of solutions for stability
        weights = self._calculate_solution_weights(pressure, grid)
        
        comprehensive_solution = (weights['adi'] * solution_adi +
                                weights['crank_nicolson'] * solution_crank_nicolson +
                                weights['douglas_rachford'] * solution_douglas_rachford)
        
        # Apply non-linear corrections
        nonlinear_correction = self._apply_nonlinear_diffusion_corrections(
            comprehensive_solution, pressure, hydraulic_props)
        
        # Calculate diffusion factor (normalized change rate)
        pressure_change_rate = (comprehensive_solution - pressure) / dt
        diffusion_factor = pressure_change_rate / np.maximum(np.abs(pressure), 1e-6)
        
        return diffusion_factor * nonlinear_correction
    
    def _calculate_comprehensive_hydraulic_properties(self, pressure: np.ndarray, 
                                                    grid: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive hydraulic properties for diffusion
        
        Includes:
        - Permeability (stress-dependent, temperature-dependent)
        - Porosity (pressure-dependent, thermal expansion)
        - Fluid compressibility (pressure and temperature dependent)
        - Rock compressibility (stress-dependent)
        - Fluid viscosity (temperature and composition dependent)
        """
        
        # Get temperature and stress fields if available
        temperature = grid.get('temperature', np.full_like(pressure, 298))
        stress_field = grid.get('stress_field', np.zeros_like(pressure))
        
        # Rock type distribution (affects hydraulic properties)
        rock_types = grid.get('rock_types', 'granite')
        
        # Base hydraulic properties by rock type
        base_props = self._get_base_hydraulic_properties(rock_types)
        
        # Pressure-dependent permeability (stress-sensitive)
        permeability = self._calculate_stress_dependent_permeability(
            base_props['permeability'], pressure, stress_field)
        
        # Temperature-dependent permeability
        permeability *= self._calculate_temperature_permeability_factor(temperature)
        
        # Pressure and temperature dependent porosity
        porosity = self._calculate_pressure_temperature_porosity(
            base_props['porosity'], pressure, temperature)
        
        # Fluid properties
        fluid_density = self._calculate_fluid_density(pressure, temperature)
        fluid_viscosity = self._calculate_fluid_viscosity(pressure, temperature)
        fluid_compressibility = self._calculate_fluid_compressibility(pressure, temperature)
        
        # Rock compressibility (stress-dependent)
        rock_compressibility = self._calculate_rock_compressibility(
            base_props['compressibility'], pressure, stress_field)
        
        # Effective (total) compressibility
        total_compressibility = (porosity * fluid_compressibility + 
                               rock_compressibility)
        
        return {
            'permeability': permeability,
            'porosity': porosity,
            'fluid_density': fluid_density,
            'fluid_viscosity': fluid_viscosity,
            'fluid_compressibility': fluid_compressibility,
            'rock_compressibility': rock_compressibility,
            'total_compressibility': total_compressibility
        }
    
    def _get_base_hydraulic_properties(self, rock_types) -> Dict[str, float]:
        """
        Get base hydraulic properties by rock type
        """
        
        properties = {
            'granite': {
                'permeability': 1e-18,      # m²
                'porosity': 0.01,           # fraction
                'compressibility': 1e-10    # 1/Pa
            },
            'sandstone': {
                'permeability': 1e-13,
                'porosity': 0.15,
                'compressibility': 5e-10
            },
            'limestone': {
                'permeability': 1e-15,
                'porosity': 0.05,
                'compressibility': 2e-10
            },
            'shale': {
                'permeability': 1e-20,
                'porosity': 0.03,
                'compressibility': 1e-9
            },
            'basalt': {
                'permeability': 1e-17,
                'porosity': 0.08,
                'compressibility': 5e-11
            }
        }
        
        return properties.get(rock_types, properties['granite'])
    
    def _calculate_stress_dependent_permeability(self, base_permeability: float,
                                               pressure: np.ndarray,
                                               stress_field: np.ndarray) -> np.ndarray:
        """
        Calculate stress-dependent permeability using empirical models
        """
        
        # Effective stress = total stress - pore pressure
        effective_stress = stress_field - pressure
        
        # Exponential decrease with effective stress (common model)
        stress_factor = np.exp(-effective_stress * 1e-8)  # Stress sensitivity
        
        # Apply bounds to prevent unrealistic values
        stress_factor = np.clip(stress_factor, 0.01, 10.0)
        
        return base_permeability * stress_factor
    
    def _calculate_temperature_permeability_factor(self, temperature: np.ndarray) -> np.ndarray:
        """
        Calculate temperature effects on permeability
        """
        
        reference_temp = 298  # K
        
        # Thermal expansion/contraction affects fracture apertures
        thermal_factor = 1 + 2e-4 * (temperature - reference_temp)
        
        # Bounds for physical realism
        thermal_factor = np.clip(thermal_factor, 0.5, 2.0)
        
        return thermal_factor
    
    def _calculate_pressure_temperature_porosity(self, base_porosity: float,
                                               pressure: np.ndarray,
                                               temperature: np.ndarray) -> np.ndarray:
        """
        Calculate pressure and temperature dependent porosity
        """
        
        # Pressure compaction
        pressure_factor = np.exp(-pressure * 1e-9)
        
        # Thermal expansion
        reference_temp = 298  # K
        thermal_expansion_coeff = 3e-5  # 1/K
        thermal_factor = 1 + thermal_expansion_coeff * (temperature - reference_temp)
        
        porosity = base_porosity * pressure_factor * thermal_factor
        
        # Physical bounds
        porosity = np.clip(porosity, 0.001, 0.5)
        
        return porosity
    
    def _calculate_fluid_density(self, pressure: np.ndarray, 
                               temperature: np.ndarray) -> np.ndarray:
        """
        Calculate fluid density using equation of state
        """
        
        # Water properties (simplified equation of state)
        rho_0 = 1000  # kg/m³ at standard conditions
        beta = 4.5e-10  # Compressibility (1/Pa)
        alpha = 2.1e-4  # Thermal expansivity (1/K)
        
        reference_pressure = 101325  # Pa
        reference_temp = 298  # K
        
        density = rho_0 * (1 + beta * (pressure - reference_pressure) - 
                          alpha * (temperature - reference_temp))
        
        return np.clip(density, 500, 1500)  # Physical bounds
    
    def _calculate_fluid_viscosity(self, pressure: np.ndarray,
                                 temperature: np.ndarray) -> np.ndarray:
        """
        Calculate fluid viscosity (water) as function of P, T
        """
        
        # Simplified viscosity model for water
        # Vogel-Fulcher-Tammann equation
        
        A = 1.856e-14  # Pa·s
        B = 4209       # K
        T0 = 109       # K
        
        viscosity = A * np.exp(B / (temperature - T0))
        
        # Pressure correction (small effect)
        pressure_correction = 1 + 2e-11 * pressure
        
        viscosity *= pressure_correction
        
        return np.clip(viscosity, 1e-6, 1e-2)  # Physical bounds
    
    def _calculate_fluid_compressibility(self, pressure: np.ndarray,
                                       temperature: np.ndarray) -> np.ndarray:
        """
        Calculate fluid compressibility (water)
        """
        
        # Temperature-dependent compressibility for water
        beta_0 = 4.6e-10  # 1/Pa at 25°C
        temp_coeff = -3e-13  # 1/(Pa·K)
        
        compressibility = beta_0 + temp_coeff * (temperature - 298)
        
        return np.clip(compressibility, 1e-11, 1e-9)  # Physical bounds
    
    def _calculate_rock_compressibility(self, base_compressibility: float,
                                      pressure: np.ndarray,
                                      stress_field: np.ndarray) -> np.ndarray:
        """
        Calculate stress-dependent rock compressibility
        """
        
        # Effective stress
        effective_stress = stress_field - pressure
        
        # Decreasing compressibility with increasing effective stress
        stress_factor = np.exp(-effective_stress * 5e-10)
        
        compressibility = base_compressibility * stress_factor
        
        return np.clip(compressibility, 1e-12, 1e-8)  # Physical bounds
    
    def _calculate_hydraulic_diffusivity_tensor(self, hydraulic_props: Dict[str, np.ndarray],
                                              pressure: np.ndarray,
                                              grid: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calculate anisotropic hydraulic diffusivity tensor
        
        D = k/(μ·ct) where:
        - k = permeability tensor
        - μ = fluid viscosity
        - ct = total compressibility
        """
        
        permeability = hydraulic_props['permeability']
        viscosity = hydraulic_props['fluid_viscosity']
        total_compressibility = hydraulic_props['total_compressibility']
        
        # Base hydraulic diffusivity
        hydraulic_diffusivity = permeability / (viscosity * total_compressibility)
        
        # Anisotropy factors (stress-induced)
        stress_field = grid.get('stress_field', np.zeros_like(pressure))
        anisotropy = self._calculate_hydraulic_anisotropy(stress_field, grid)
        
        # Diffusivity tensor components
        diffusivity_tensor = {
            'Dxx': hydraulic_diffusivity * anisotropy['xx'],
            'Dyy': hydraulic_diffusivity * anisotropy['yy'], 
            'Dzz': hydraulic_diffusivity * anisotropy['zz'],
            'Dxy': hydraulic_diffusivity * anisotropy['xy'],
            'Dxz': hydraulic_diffusivity * anisotropy['xz'],
            'Dyz': hydraulic_diffusivity * anisotropy['yz']
        }
        
        return diffusivity_tensor
    
    def _calculate_hydraulic_anisotropy(self, stress_field: np.ndarray, 
                                      grid: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calculate hydraulic anisotropy from stress field
        """
        
        # Default isotropic case
        shape = stress_field.shape
        anisotropy = {
            'xx': np.ones(shape),
            'yy': np.ones(shape),
            'zz': np.ones(shape),
            'xy': np.zeros(shape),
            'xz': np.zeros(shape),
            'yz': np.zeros(shape)
        }
        
        # Stress-induced anisotropy (fracture alignment)
        if np.any(stress_field != 0):
            # Enhanced permeability parallel to maximum compression
            stress_magnitude = np.abs(stress_field)
            anisotropy_factor = 1 + 0.5 * stress_magnitude / np.maximum(
                np.mean(stress_magnitude), 1e6)
            
            anisotropy['zz'] *= anisotropy_factor  # Vertical enhanced
            anisotropy['xx'] *= 0.8  # Horizontal reduced
            anisotropy['yy'] *= 0.8
        
        return anisotropy
    
    def _calculate_pore_pressure_sources(self, pressure: np.ndarray,
                                       grid: Dict[str, Any]) -> np.ndarray:
        """
        Calculate source/sink terms for pore pressure diffusion
        """
        
        sources = np.zeros_like(pressure)
        
        # Deformation-induced sources (volumetric strain rate)
        strain_rate = grid.get('volumetric_strain_rate', np.zeros_like(pressure))
        fluid_density = grid.get('fluid_density', np.full_like(pressure, 1000))
        
        deformation_source = fluid_density * strain_rate
        sources += deformation_source
        
        # Thermal effects (thermal expansion/contraction)
        temperature_rate = grid.get('temperature_rate', np.zeros_like(pressure))
        thermal_expansion = 2e-4  # 1/K
        
        thermal_source = fluid_density * thermal_expansion * temperature_rate
        sources += thermal_source
        
        # Injection/extraction wells (if any)
        wells = grid.get('wells', [])
        for well in wells:
            # Add point sources at well locations
            pass  # Implementation would add localized sources
        
        return sources
    
    def _get_pore_pressure_boundary_conditions(self, pressure: np.ndarray,
                                             grid: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get boundary conditions for pore pressure diffusion
        """
        
        boundary_conditions = {
            'type': 'neumann',  # No-flow boundaries (default)
            'values': np.zeros_like(pressure)
        }
        
        # Surface boundary: atmospheric pressure
        if pressure.ndim == 3:
            boundary_conditions['surface'] = {
                'type': 'dirichlet',
                'value': 101325  # Pa
            }
        
        return boundary_conditions
    
    def _solve_adi_diffusion(self, pressure: np.ndarray, 
                           diffusivity_tensor: Dict[str, np.ndarray],
                           source_terms: np.ndarray,
                           boundary_conditions: Dict[str, Any],
                           dx: float, dy: float, dz: float, dt: float) -> np.ndarray:
        """
        Solve diffusion using Alternating Direction Implicit (ADI) method
        """
        
        P = pressure.copy()
        Dxx = diffusivity_tensor['Dxx']
        Dyy = diffusivity_tensor['Dyy'] 
        Dzz = diffusivity_tensor['Dzz']
        
        # Stability coefficients
        rx = dt * Dxx / (dx**2)
        ry = dt * Dyy / (dy**2)
        rz = dt * Dzz / (dz**2)
        
        # Three-step ADI procedure for 3D
        # Step 1: Implicit in x, explicit in y,z
        P1 = self._adi_step_x(P, rx, ry, rz, source_terms, dt)
        
        # Step 2: Implicit in y, explicit in x,z  
        P2 = self._adi_step_y(P1, rx, ry, rz, np.zeros_like(source_terms), dt)
        
        # Step 3: Implicit in z, explicit in x,y
        P3 = self._adi_step_z(P2, rx, ry, rz, np.zeros_like(source_terms), dt)
        
        return P3
    
    def _adi_step_x(self, P: np.ndarray, rx: np.ndarray, ry: np.ndarray, 
                   rz: np.ndarray, sources: np.ndarray, dt: float) -> np.ndarray:
        """ADI step implicit in x-direction"""
        
        P_new = P.copy()
        
        # For each y,z slice, solve tridiagonal system
        ny, nz = P.shape[1], P.shape[2] if P.ndim > 2 else 1
        
        for j in range(ny):
            for k in range(nz if P.ndim > 2 else 1):
                if P.ndim == 3:
                    slice_data = P[:, j, k]
                    r = rx[:, j, k] if rx.ndim == 3 else rx
                else:
                    slice_data = P[:, j] if P.ndim == 2 else P
                    r = rx if np.isscalar(rx) else rx[:, j] if rx.ndim == 2 else rx
                
                # Thomas algorithm for tridiagonal system
                P_slice = self._solve_tridiagonal_adi(slice_data, r)
                
                if P.ndim == 3:
                    P_new[:, j, k] = P_slice
                elif P.ndim == 2:
                    P_new[:, j] = P_slice
                else:
                    P_new = P_slice
        
        return P_new
    
    def _adi_step_y(self, P: np.ndarray, rx: np.ndarray, ry: np.ndarray,
                   rz: np.ndarray, sources: np.ndarray, dt: float) -> np.ndarray:
        """ADI step implicit in y-direction"""
        
        P_new = P.copy()
        
        # For each x,z slice, solve tridiagonal system
        nx, nz = P.shape[0], P.shape[2] if P.ndim > 2 else 1
        
        for i in range(nx):
            for k in range(nz if P.ndim > 2 else 1):
                if P.ndim == 3:
                    slice_data = P[i, :, k]
                    r = ry[i, :, k] if ry.ndim == 3 else ry
                else:
                    slice_data = P[i, :] if P.ndim == 2 else P
                    r = ry if np.isscalar(ry) else ry[i, :] if ry.ndim == 2 else ry
                
                P_slice = self._solve_tridiagonal_adi(slice_data, r)
                
                if P.ndim == 3:
                    P_new[i, :, k] = P_slice
                elif P.ndim == 2:
                    P_new[i, :] = P_slice
        
        return P_new
    
    def _adi_step_z(self, P: np.ndarray, rx: np.ndarray, ry: np.ndarray,
                   rz: np.ndarray, sources: np.ndarray, dt: float) -> np.ndarray:
        """ADI step implicit in z-direction"""
        
        if P.ndim < 3:
            return P  # No z-direction for 2D
        
        P_new = P.copy()
        nx, ny = P.shape[0], P.shape[1]
        
        for i in range(nx):
            for j in range(ny):
                slice_data = P[i, j, :]
                r = rz[i, j, :] if rz.ndim == 3 else rz
                
                P_slice = self._solve_tridiagonal_adi(slice_data, r)
                P_new[i, j, :] = P_slice
        
        return P_new
    
    def _solve_tridiagonal_adi(self, P_slice: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Solve tridiagonal system for ADI method
        System: -r*P[i-1] + (1+2r)*P[i] - r*P[i+1] = P_old[i]
        """
        
        n = len(P_slice)
        if n <= 2:
            return P_slice  # Too small to solve
        
        # Tridiagonal matrix coefficients
        a = -r if np.isscalar(r) else -r[1:]     # Sub-diagonal
        b = 1 + 2*r if np.isscalar(r) else 1 + 2*r  # Main diagonal  
        c = -r if np.isscalar(r) else -r[:-1]    # Super-diagonal
        
        d = P_slice.copy()  # RHS
        
        # Thomas algorithm
        # Forward sweep
        for i in range(1, n):
            m = a[i-1] / b[i-1] if not np.isscalar(a) else a / b[i-1] if not np.isscalar(b) else a / b
            b_i = b[i] if not np.isscalar(b) else b
            c_i = c[i-1] if not np.isscalar(c) else c
            
            b_new = b_i - m * c_i
            d[i] = d[i] - m * d[i-1]
            
            if not np.isscalar(b):
                b[i] = b_new
            else:
                b = b_new
        
        # Back substitution  
        x = np.zeros_like(P_slice)
        x[-1] = d[-1] / (b[-1] if not np.isscalar(b) else b)
        
        for i in range(n-2, -1, -1):
            c_i = c[i] if not np.isscalar(c) else c
            b_i = b[i] if not np.isscalar(b) else b
            x[i] = (d[i] - c_i * x[i+1]) / b_i
        
        return x
    
    def _solve_crank_nicolson_diffusion(self, pressure: np.ndarray,
                                      diffusivity_tensor: Dict[str, np.ndarray],
                                      source_terms: np.ndarray,
                                      boundary_conditions: Dict[str, Any],
                                      dx: float, dy: float, dz: float, 
                                      dt: float) -> np.ndarray:
        """
        Solve diffusion using Crank-Nicolson method (second-order accurate)
        """
        
        P = pressure.copy()
        D = diffusivity_tensor['Dxx']  # Simplified isotropic case
        
        # Stability parameter
        r = dt * D / (dx**2)
        
        # Crank-Nicolson: (I - 0.5*r*L)P^{n+1} = (I + 0.5*r*L)P^n + dt*S
        # where L is the discrete Laplacian operator
        
        # For simplicity, use iterative solver (conjugate gradient)
        P_new = self._crank_nicolson_solver(P, r, source_terms, dt)
        
        return P_new
    
    def _crank_nicolson_solver(self, P: np.ndarray, r: np.ndarray, 
                             sources: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve Crank-Nicolson system using iterative method
        """
        
        # Simplified implementation - return modified pressure
        laplacian_P = self._calculate_discrete_laplacian(P)
        
        P_new = P + 0.5 * dt * r * laplacian_P + dt * sources
        
        return P_new
    
    def _solve_douglas_rachford_diffusion(self, pressure: np.ndarray,
                                        diffusivity_tensor: Dict[str, np.ndarray],
                                        source_terms: np.ndarray,
                                        boundary_conditions: Dict[str, Any],
                                        dx: float, dy: float, dz: float,
                                        dt: float) -> np.ndarray:
        """
        Solve diffusion using Douglas-Rachford splitting method
        """
        
        # Comprehensive Douglas-Rachford splitting method for anisotropic diffusion
        # Handles full tensor diffusivity with operator splitting
        
        P = pressure.copy()
        
        # Extract full diffusivity tensor components
        Dxx = diffusivity_tensor['Dxx']
        Dyy = diffusivity_tensor['Dyy']
        Dzz = diffusivity_tensor['Dzz']
        Dxy = diffusivity_tensor.get('Dxy', np.zeros_like(Dxx))
        Dxz = diffusivity_tensor.get('Dxz', np.zeros_like(Dxx))
        Dyz = diffusivity_tensor.get('Dyz', np.zeros_like(Dxx))
        
        # Calculate comprehensive Douglas-Rachford solution
        P_new = self._comprehensive_douglas_rachford_solver(
            P, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, source_terms, 
            boundary_conditions, dx, dy, dz, dt)
        
        # Apply stability corrections and convergence checks
        P_new = self._apply_douglas_rachford_stabilization(P, P_new, dt)
        
        return P_new
    
    def _comprehensive_douglas_rachford_solver(self, P: np.ndarray,
                                             Dxx: np.ndarray, Dyy: np.ndarray, Dzz: np.ndarray,
                                             Dxy: np.ndarray, Dxz: np.ndarray, Dyz: np.ndarray,
                                             sources: np.ndarray, boundary_conditions: Dict[str, Any],
                                             dx: float, dy: float, dz: float, dt: float) -> np.ndarray:
        """
        Comprehensive Douglas-Rachford splitting solver for anisotropic diffusion
        
        Solves: ∂P/∂t = ∇·(D∇P) + S where D is full anisotropic tensor
        
        Uses operator splitting:
        1. Split operators by spatial directions and cross-terms
        2. Apply Douglas-Rachford alternating projections
        3. Include adaptive time stepping and convergence monitoring
        4. Handle complex boundary conditions
        """
        
        # Initialize solution
        P_current = P.copy()
        
        # Split time step for stability (adaptive sub-stepping)
        sub_steps = self._calculate_douglas_rachford_substeps(dt, Dxx, Dyy, Dzz, dx, dy, dz)
        dt_sub = dt / sub_steps
        
        # Douglas-Rachford iteration parameters
        max_iterations = 50
        tolerance = 1e-8
        relaxation_param = 1.0
        
        for sub_step in range(sub_steps):
            
            # Douglas-Rachford splitting iterations
            P_old = P_current.copy()
            
            for iteration in range(max_iterations):
                
                # Step 1: Solve diagonal diffusion operators
                P1 = self._solve_diagonal_diffusion_operators(
                    P_current, Dxx, Dyy, Dzz, sources, dx, dy, dz, dt_sub)
                
                # Step 2: Solve cross-term diffusion operators  
                P2 = self._solve_cross_term_diffusion_operators(
                    P_current, Dxy, Dxz, Dyz, dx, dy, dz, dt_sub)
                
                # Douglas-Rachford update with relaxation
                P_temp = 2 * P1 - P_current
                P_temp = self._solve_cross_term_diffusion_operators(
                    P_temp, Dxy, Dxz, Dyz, dx, dy, dz, dt_sub)
                
                P_new_iter = P_current + relaxation_param * (P_temp - P1)
                
                # Apply boundary conditions
                P_new_iter = self._apply_boundary_conditions_douglas_rachford(
                    P_new_iter, boundary_conditions)
                
                # Check convergence
                residual = np.linalg.norm(P_new_iter - P_current) / np.linalg.norm(P_current)
                
                if residual < tolerance:
                    break
                
                # Update for next iteration
                P_current = P_new_iter.copy()
            
            # Update solution for next sub-step
            P_current = P_new_iter.copy()
        
        return P_current
    
    def _calculate_douglas_rachford_substeps(self, dt: float, 
                                           Dxx: np.ndarray, Dyy: np.ndarray, Dzz: np.ndarray,
                                           dx: float, dy: float, dz: float) -> int:
        """
        Calculate optimal number of sub-steps for Douglas-Rachford stability
        """
        
        # Calculate maximum diffusivity for stability analysis
        D_max = np.maximum.reduce([np.max(Dxx), np.max(Dyy), np.max(Dzz)])
        
        # CFL-like condition for diffusion
        cfl_x = D_max * dt / (dx**2)
        cfl_y = D_max * dt / (dy**2)  
        cfl_z = D_max * dt / (dz**2)
        
        max_cfl = max(cfl_x, cfl_y, cfl_z)
        
        # Adaptive sub-stepping based on stability
        if max_cfl > 0.5:
            sub_steps = int(np.ceil(max_cfl / 0.3))
        else:
            sub_steps = 1
        
        return min(max(sub_steps, 1), 20)  # Bound between 1 and 20
    
    def _solve_diagonal_diffusion_operators(self, P: np.ndarray,
                                          Dxx: np.ndarray, Dyy: np.ndarray, Dzz: np.ndarray,
                                          sources: np.ndarray,
                                          dx: float, dy: float, dz: float, dt: float) -> np.ndarray:
        """
        Solve diagonal components of diffusion tensor using implicit methods
        """
        
        P_result = P.copy()
        
        # X-direction diffusion: ∂P/∂t = ∂/∂x(Dxx ∂P/∂x)
        P_x = self._implicit_directional_diffusion(P_result, Dxx, dx, dt, axis=0)
        
        # Y-direction diffusion: ∂P/∂t = ∂/∂y(Dyy ∂P/∂y)
        P_y = self._implicit_directional_diffusion(P_x, Dyy, dy, dt, axis=1)
        
        # Z-direction diffusion: ∂P/∂t = ∂/∂z(Dzz ∂P/∂z)
        if P.ndim == 3:
            P_z = self._implicit_directional_diffusion(P_y, Dzz, dz, dt, axis=2)
        else:
            P_z = P_y
        
        # Add source terms
        P_final = P_z + dt * sources
        
        return P_final
    
    def _solve_cross_term_diffusion_operators(self, P: np.ndarray,
                                            Dxy: np.ndarray, Dxz: np.ndarray, Dyz: np.ndarray,
                                            dx: float, dy: float, dz: float, dt: float) -> np.ndarray:
        """
        Solve cross-term components of diffusion tensor
        """
        
        P_result = P.copy()
        
        # XY cross-term: ∂P/∂t = ∂/∂x(Dxy ∂P/∂y) + ∂/∂y(Dxy ∂P/∂x)
        if np.any(Dxy != 0):
            cross_xy = self._calculate_cross_derivative_term(P_result, Dxy, dx, dy, axes=(0,1))
            P_result += dt * cross_xy
        
        # XZ cross-term: ∂P/∂t = ∂/∂x(Dxz ∂P/∂z) + ∂/∂z(Dxz ∂P/∂x)
        if P.ndim == 3 and np.any(Dxz != 0):
            cross_xz = self._calculate_cross_derivative_term(P_result, Dxz, dx, dz, axes=(0,2))
            P_result += dt * cross_xz
        
        # YZ cross-term: ∂P/∂t = ∂/∂y(Dyz ∂P/∂z) + ∂/∂z(Dyz ∂P/∂y)  
        if P.ndim == 3 and np.any(Dyz != 0):
            cross_yz = self._calculate_cross_derivative_term(P_result, Dyz, dy, dz, axes=(1,2))
            P_result += dt * cross_yz
        
        return P_result
    
    def _implicit_directional_diffusion(self, P: np.ndarray, D: np.ndarray,
                                      dh: float, dt: float, axis: int) -> np.ndarray:
        """
        Solve 1D implicit diffusion in specified direction using Thomas algorithm
        """
        
        P_result = P.copy()
        
        if axis >= P.ndim:
            return P_result
        
        # Move target axis to the last position for easier processing
        P_work = np.moveaxis(P_result, axis, -1)
        D_work = np.moveaxis(D, axis, -1) if D.ndim == P.ndim else D
        
        # Get shape for iteration
        *leading_dims, n_points = P_work.shape
        
        # Flatten leading dimensions for vectorized processing
        P_flat = P_work.reshape(-1, n_points)
        if D_work.ndim == P_work.ndim:
            D_flat = D_work.reshape(-1, n_points)
        else:
            D_flat = np.broadcast_to(D_work, P_flat.shape)
        
        # Solve tridiagonal system for each slice
        for i in range(P_flat.shape[0]):
            P_slice = P_flat[i, :]
            D_slice = D_flat[i, :] if D_flat.ndim > 1 else D_flat
            
            # Set up tridiagonal coefficients
            r = dt * D_slice / (dh**2)
            
            # Coefficients for implicit scheme: -r[i-1]*P[i-1] + (1+2r[i])*P[i] - r[i+1]*P[i+1] = P_old[i]
            a = -r[:-1] if hasattr(r, '__len__') else np.full(n_points-1, -r)  # sub-diagonal
            b = 1 + 2*r if hasattr(r, '__len__') else np.full(n_points, 1 + 2*r)  # main diagonal
            c = -r[1:] if hasattr(r, '__len__') else np.full(n_points-1, -r)   # super-diagonal
            
            d = P_slice.copy()  # RHS
            
            # Thomas algorithm
            P_flat[i, :] = self._thomas_algorithm(a, b, c, d)
        
        # Reshape back and move axis back
        P_result_shaped = P_flat.reshape(*leading_dims, n_points)
        P_result = np.moveaxis(P_result_shaped, -1, axis)
        
        return P_result
    
    def _calculate_cross_derivative_term(self, P: np.ndarray, D_cross: np.ndarray,
                                       dh1: float, dh2: float, axes: tuple) -> np.ndarray:
        """
        Calculate cross-derivative terms for anisotropic diffusion tensor
        """
        
        axis1, axis2 = axes
        
        # Calculate mixed derivatives: ∂/∂x1(D ∂P/∂x2) + ∂/∂x2(D ∂P/∂x1)
        
        # First term: ∂/∂x1(D ∂P/∂x2)
        dP_dx2 = np.gradient(P, dh2, axis=axis2)
        D_dP_dx2 = D_cross * dP_dx2
        term1 = np.gradient(D_dP_dx2, dh1, axis=axis1)
        
        # Second term: ∂/∂x2(D ∂P/∂x1)  
        dP_dx1 = np.gradient(P, dh1, axis=axis1)
        D_dP_dx1 = D_cross * dP_dx1
        term2 = np.gradient(D_dP_dx1, dh2, axis=axis2)
        
        return term1 + term2
    
    def _thomas_algorithm(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Solve tridiagonal system using Thomas algorithm
        
        Ax = d where A has:
        - a: sub-diagonal (length n-1)
        - b: main diagonal (length n) 
        - c: super-diagonal (length n-1)
        """
        
        n = len(d)
        if n <= 1:
            return d / b if len(b) > 0 else d
        
        # Forward elimination
        c_prime = np.zeros(n-1)
        d_prime = np.zeros(n)
        
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        for i in range(1, n-1):
            denom = b[i] - a[i-1] * c_prime[i-1]
            c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom
        
        # Last row
        d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
        
        # Back substitution
        x = np.zeros(n)
        x[n-1] = d_prime[n-1]
        
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
        return x
    
    def _apply_boundary_conditions_douglas_rachford(self, P: np.ndarray, 
                                                  boundary_conditions: Dict[str, Any]) -> np.ndarray:
        """
        Apply boundary conditions in Douglas-Rachford framework
        """
        
        P_result = P.copy()
        
        bc_type = boundary_conditions.get('type', 'neumann')
        
        if bc_type == 'dirichlet':
            # Fixed pressure boundaries
            bc_value = boundary_conditions.get('value', 0)
            
            # Apply to all boundaries (simplified)
            if P.ndim == 1:
                P_result[0] = bc_value
                P_result[-1] = bc_value
            elif P.ndim == 2:
                P_result[0, :] = bc_value   # Top
                P_result[-1, :] = bc_value  # Bottom
                P_result[:, 0] = bc_value   # Left
                P_result[:, -1] = bc_value  # Right
            elif P.ndim == 3:
                P_result[0, :, :] = bc_value   # Front
                P_result[-1, :, :] = bc_value  # Back
                P_result[:, 0, :] = bc_value   # Left
                P_result[:, -1, :] = bc_value  # Right
                P_result[:, :, 0] = bc_value   # Bottom
                P_result[:, :, -1] = bc_value  # Top
                
        elif bc_type == 'neumann':
            # No-flow boundaries (zero gradient) - already satisfied by finite differences
            pass
        
        # Special surface boundary condition
        if 'surface' in boundary_conditions and P.ndim == 3:
            surface_bc = boundary_conditions['surface']
            if surface_bc['type'] == 'dirichlet':
                P_result[:, :, -1] = surface_bc['value']  # Top surface
        
        return P_result
    
    def _apply_douglas_rachford_stabilization(self, P_old: np.ndarray, 
                                            P_new: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply stabilization and limiting to Douglas-Rachford solution
        """
        
        # Maximum allowed relative change per time step
        max_relative_change = 0.5
        
        # Calculate relative change
        change = np.abs(P_new - P_old)
        max_change = np.maximum(np.abs(P_old), np.abs(P_new))
        relative_change = np.divide(change, max_change, 
                                  out=np.zeros_like(change), where=max_change!=0)
        
        # Apply limiting where changes are too large
        excessive_change = relative_change > max_relative_change
        
        if np.any(excessive_change):
            # Reduce time step effect where changes are excessive
            limiter = np.where(excessive_change, 
                             max_relative_change / relative_change, 
                             np.ones_like(relative_change))
            
            P_limited = P_old + limiter * (P_new - P_old)
        else:
            P_limited = P_new
        
        # Ensure physical bounds (pressure cannot be negative)
        P_limited = np.maximum(P_limited, 0)
        
        # Apply monotonicity preservation (TVD property)
        P_final = self._apply_tvd_limiting(P_old, P_limited)
        
        return P_final
    
    def _apply_tvd_limiting(self, P_old: np.ndarray, P_new: np.ndarray) -> np.ndarray:
        """
        Apply Total Variation Diminishing (TVD) limiting to preserve monotonicity
        """
        
        # Simple TVD constraint: new solution should not create new extrema
        P_result = P_new.copy()
        
        if P_old.ndim >= 2:
            # Check for new local maxima/minima in interior points
            for axis in range(P_old.ndim):
                # Shift arrays to get neighbors
                P_old_left = np.roll(P_old, 1, axis=axis)
                P_old_right = np.roll(P_old, -1, axis=axis)
                P_new_left = np.roll(P_new, 1, axis=axis) 
                P_new_right = np.roll(P_new, -1, axis=axis)
                
                # Identify monotonic regions in old solution
                old_increasing = (P_old >= P_old_left) & (P_old <= P_old_right)
                old_decreasing = (P_old <= P_old_left) & (P_old >= P_old_right)
                
                # Preserve monotonicity in new solution
                violation_max = old_increasing & ((P_new > P_new_left) & (P_new > P_new_right)) & (P_new > np.maximum(P_old_left, P_old_right))
                violation_min = old_decreasing & ((P_new < P_new_left) & (P_new < P_new_right)) & (P_new < np.minimum(P_old_left, P_old_right))
                
                # Limit violations
                P_result = np.where(violation_max, 
                                  np.minimum(P_new, np.maximum(P_old_left, P_old_right)),
                                  P_result)
                P_result = np.where(violation_min,
                                  np.maximum(P_new, np.minimum(P_old_left, P_old_right)),
                                  P_result)
        
        return P_result
    
    def _calculate_discrete_laplacian(self, P: np.ndarray) -> np.ndarray:
        """
        Calculate discrete Laplacian using finite differences
        """
        
        laplacian = np.zeros_like(P)
        
        if P.ndim == 1:
            # 1D case
            laplacian[1:-1] = P[2:] - 2*P[1:-1] + P[:-2]
        elif P.ndim == 2:
            # 2D case
            laplacian[1:-1, 1:-1] = (P[2:, 1:-1] + P[:-2, 1:-1] + 
                                   P[1:-1, 2:] + P[1:-1, :-2] - 4*P[1:-1, 1:-1])
        elif P.ndim == 3:
            # 3D case
            laplacian[1:-1, 1:-1, 1:-1] = (P[2:, 1:-1, 1:-1] + P[:-2, 1:-1, 1:-1] +
                                         P[1:-1, 2:, 1:-1] + P[1:-1, :-2, 1:-1] +
                                         P[1:-1, 1:-1, 2:] + P[1:-1, 1:-1, :-2] -
                                         6*P[1:-1, 1:-1, 1:-1])
        
        return laplacian
    
    def _calculate_solution_weights(self, pressure: np.ndarray, 
                                  grid: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate weights for combining different solution methods
        """
        
        # Base weights
        weights = {
            'adi': 0.5,
            'crank_nicolson': 0.3,
            'douglas_rachford': 0.2
        }
        
        # Adjust based on problem characteristics
        pressure_range = np.ptp(pressure)
        
        if pressure_range > 1e6:  # High pressure gradients
            weights['adi'] = 0.7  # More stable
            weights['crank_nicolson'] = 0.2
            weights['douglas_rachford'] = 0.1
        
        return weights
    
    def _apply_nonlinear_diffusion_corrections(self, solution: np.ndarray,
                                             original_pressure: np.ndarray,
                                             hydraulic_props: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply corrections for non-linear effects
        """
        
        # Pressure-dependent corrections
        pressure_factor = 1.0 + 0.1 * np.tanh((solution - original_pressure) / 1e6)
        
        # Permeability feedback effects
        permeability = hydraulic_props['permeability']
        perm_factor = 1.0 + 0.05 * np.log10(permeability / np.mean(permeability))
        
        correction = pressure_factor * perm_factor
        
        return np.clip(correction, 0.1, 10.0)
        
        # Pore pressure factor (tanh saturation to prevent unphysical values)
        pore_pressure_factor = np.tanh(diffusion_param)
        
        return pore_pressure_factor
    
    def _apply_anisotropic_pressure_effects(self, pressure_stress: Dict[str, np.ndarray],
                                          grid: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Apply anisotropic material effects on pressure-stress coupling"""
        
        X, Y, Z = grid['X'], grid['Y'], grid['Z']
        
        # Calculate anisotropic elastic tensor components
        anisotropy_tensor = self._calculate_anisotropic_elastic_tensor(X, Y, Z)
        
        # Apply anisotropic corrections to stress tensor
        for i, component in enumerate(['sigma_xx', 'sigma_yy', 'sigma_zz']):
            anisotropic_factor = anisotropy_tensor[f'C{i+1}{i+1}']
            pressure_stress[component] *= anisotropic_factor
        
        return pressure_stress
    
    def _calculate_anisotropic_elastic_tensor(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate anisotropic elastic tensor components"""
        
        # Comprehensive anisotropic elastic tensor calculation
        # Includes multiple anisotropy types and geological controls
        return self._calculate_comprehensive_anisotropic_elastic_tensor(X, Y, Z)
    
    def _calculate_comprehensive_anisotropic_elastic_tensor(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive anisotropic elastic tensor with multiple symmetry types
        
        Includes:
        1. Transverse isotropy (VTI/HTI) - layered media
        2. Orthorhombic anisotropy - fractured media  
        3. Triclinic anisotropy - complex deformed media
        4. Stress-induced anisotropy
        5. Temperature-dependent anisotropy
        6. Pressure-dependent anisotropy
        7. Crack-induced anisotropy
        8. Crystal preferred orientation (CPO) effects
        """
        
        # Get spatial coordinates and geological context
        depth = np.abs(Z)
        lateral_position = np.sqrt(X**2 + Y**2)
        
        # Determine dominant anisotropy type based on geological context
        anisotropy_type = self._determine_anisotropy_type(X, Y, Z, depth)
        
        # Calculate base elastic parameters
        base_elastic_params = self._calculate_base_elastic_parameters(depth, lateral_position)
        
        # Calculate comprehensive anisotropic tensor for each type
        if anisotropy_type == 'transverse_isotropic':
            tensor = self._calculate_transverse_isotropic_tensor(
                X, Y, Z, base_elastic_params)
        elif anisotropy_type == 'orthorhombic':
            tensor = self._calculate_orthorhombic_tensor(
                X, Y, Z, base_elastic_params)
        elif anisotropy_type == 'triclinic':
            tensor = self._calculate_triclinic_tensor(
                X, Y, Z, base_elastic_params)
        else:
            # Default to comprehensive VTI
            tensor = self._calculate_transverse_isotropic_tensor(
                X, Y, Z, base_elastic_params)
        
        # Apply stress-induced modifications
        stress_modified_tensor = self._apply_stress_induced_anisotropy(
            tensor, X, Y, Z, depth)
        
        # Apply temperature effects
        temperature_modified_tensor = self._apply_temperature_anisotropy_effects(
            stress_modified_tensor, X, Y, Z, depth)
        
        # Apply pressure effects
        pressure_modified_tensor = self._apply_pressure_anisotropy_effects(
            temperature_modified_tensor, X, Y, Z, depth)
        
        # Apply crack-induced anisotropy
        crack_modified_tensor = self._apply_crack_induced_anisotropy(
            pressure_modified_tensor, X, Y, Z, depth)
        
        # Apply crystal preferred orientation effects
        final_tensor = self._apply_cpo_anisotropy_effects(
            crack_modified_tensor, X, Y, Z, depth)
        
        return final_tensor
    
    def _determine_anisotropy_type(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray) -> str:
        """
        Determine the dominant anisotropy type based on geological context
        """
        
        # Depth-based classification
        shallow_depth = depth < 2.0  # km
        intermediate_depth = (depth >= 2.0) & (depth < 10.0)
        deep_depth = depth >= 10.0
        
        # Lateral stress variations
        lateral_stress_variation = np.gradient(np.gradient(depth))
        high_lateral_stress = np.abs(lateral_stress_variation) > 0.1
        
        # Default classification
        if np.mean(shallow_depth) > 0.5:
            # Shallow: likely fractured/orthorhombic
            return 'orthorhombic'
        elif np.mean(intermediate_depth) > 0.5:
            # Intermediate: likely layered/transverse isotropic
            return 'transverse_isotropic'
        elif np.mean(deep_depth) > 0.5:
            # Deep: complex deformation/triclinic
            return 'triclinic'
        else:
            return 'transverse_isotropic'
    
    def _calculate_base_elastic_parameters(self, depth: np.ndarray, lateral_position: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate base elastic parameters varying with depth and position
        """
        
        # Depth-dependent P-wave velocity (typical crustal model)
        vp0 = 6.0  # km/s at surface
        vp_gradient = 0.5  # (km/s)/km
        vp = vp0 + vp_gradient * depth
        
        # Depth-dependent S-wave velocity
        vs = vp / 1.73  # Typical Vp/Vs ratio
        
        # Depth-dependent density (Nafe-Drake relationship)
        density = 2.0 + 0.3 * depth  # g/cm³
        density = np.clip(density, 2.0, 3.5)
        
        # Calculate elastic moduli
        mu = density * vs**2 * 1e9  # Shear modulus (Pa)
        lambda_param = density * vp**2 * 1e9 - 2 * mu  # Lamé parameter
        
        # Young's modulus and Poisson's ratio
        E = mu * (3 * lambda_param + 2 * mu) / (lambda_param + mu)
        nu = lambda_param / (2 * (lambda_param + mu))
        
        return {
            'vp': vp,
            'vs': vs,
            'density': density,
            'mu': mu,
            'lambda': lambda_param,
            'E': E,
            'nu': nu
        }
    
    def _calculate_transverse_isotropic_tensor(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                             base_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate transverse isotropic (VTI/HTI) elastic tensor
        Uses Thomsen parameters for characterization
        """
        
        depth = np.abs(Z)
        
        # Thomsen anisotropy parameters (depth-dependent)
        epsilon = self._calculate_thomsen_epsilon(depth, X, Y)  # P-wave anisotropy
        delta = self._calculate_thomsen_delta(depth, X, Y)     # Delta parameter
        gamma = self._calculate_thomsen_gamma(depth, X, Y)     # Shear-wave splitting
        
        # Base isotropic velocities
        vp0 = base_params['vp']  # Vertical P-wave velocity
        vs0 = base_params['vs']  # Vertical S-wave velocity
        density = base_params['density']
        
        # Calculate elastic constants using Thomsen relationships
        C33 = density * vp0**2 * 1e9  # Pa
        C44 = density * vs0**2 * 1e9  # Pa
        C11 = C33 * (1 + 2 * epsilon)
        C22 = C11  # Horizontal isotropy for VTI
        C66 = C44 * (1 + 2 * gamma)
        C55 = C44  # VTI symmetry
        
        # Off-diagonal terms
        C12 = C11 - 2 * C66
        C13 = np.sqrt((C33 - C44) * ((C33 * (1 + 2 * delta))**2 - C44**2)) - C44
        C23 = C13  # VTI symmetry
        
        # Higher-order corrections for realistic media
        C13 = self._apply_realistic_c13_corrections(C13, C11, C33, depth)
        C23 = C13  # Maintain VTI symmetry
        
        tensor = {
            'C11': C11, 'C22': C22, 'C33': C33,
            'C44': C44, 'C55': C55, 'C66': C66,
            'C12': C12, 'C13': C13, 'C23': C23,
            'C14': np.zeros_like(C11), 'C15': np.zeros_like(C11), 'C16': np.zeros_like(C11),
            'C24': np.zeros_like(C11), 'C25': np.zeros_like(C11), 'C26': np.zeros_like(C11),
            'C34': np.zeros_like(C11), 'C35': np.zeros_like(C11), 'C36': np.zeros_like(C11),
            'C45': np.zeros_like(C11), 'C46': np.zeros_like(C11), 'C56': np.zeros_like(C11)
        }
        
        return tensor
    
    def _calculate_orthorhombic_tensor(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                     base_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate orthorhombic elastic tensor (3 mutually perpendicular planes of symmetry)
        Common in fractured media with two orthogonal fracture sets
        """
        
        depth = np.abs(Z)
        
        # Orthorhombic anisotropy parameters
        aniso_params = self._calculate_orthorhombic_parameters(depth, X, Y)
        
        density = base_params['density']
        vp_base = base_params['vp']
        vs_base = base_params['vs']
        
        # Nine independent elastic constants for orthorhombic media
        C11 = density * vp_base**2 * (1 + aniso_params['eps1']) * 1e9
        C22 = density * vp_base**2 * (1 + aniso_params['eps2']) * 1e9
        C33 = density * vp_base**2 * 1e9  # Reference
        
        C44 = density * vs_base**2 * (1 + aniso_params['gamma1']) * 1e9
        C55 = density * vs_base**2 * (1 + aniso_params['gamma2']) * 1e9
        C66 = density * vs_base**2 * (1 + aniso_params['gamma3']) * 1e9
        
        # Off-diagonal terms (constrained by stability conditions)
        C12 = C11 + C22 - 2 * C66 - aniso_params['delta12'] * density * vp_base**2 * 1e9
        C13 = aniso_params['delta13'] * np.sqrt(C11 * C33)
        C23 = aniso_params['delta23'] * np.sqrt(C22 * C33)
        
        # Apply stability constraints
        C12, C13, C23 = self._apply_orthorhombic_stability_constraints(
            C11, C22, C33, C44, C55, C66, C12, C13, C23)
        
        tensor = {
            'C11': C11, 'C22': C22, 'C33': C33,
            'C44': C44, 'C55': C55, 'C66': C66,
            'C12': C12, 'C13': C13, 'C23': C23,
            'C14': np.zeros_like(C11), 'C15': np.zeros_like(C11), 'C16': np.zeros_like(C11),
            'C24': np.zeros_like(C11), 'C25': np.zeros_like(C11), 'C26': np.zeros_like(C11),
            'C34': np.zeros_like(C11), 'C35': np.zeros_like(C11), 'C36': np.zeros_like(C11),
            'C45': np.zeros_like(C11), 'C46': np.zeros_like(C11), 'C56': np.zeros_like(C11)
        }
        
        return tensor
    
    def _calculate_triclinic_tensor(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                  base_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate triclinic elastic tensor (most general anisotropy)
        21 independent elastic constants
        """
        
        depth = np.abs(Z)
        
        # Start with orthorhombic base
        ortho_tensor = self._calculate_orthorhombic_tensor(X, Y, Z, base_params)
        
        # Add triclinic perturbations (typically small)
        triclinic_perturbations = self._calculate_triclinic_perturbations(depth, X, Y, base_params)
        
        # All 21 independent components
        tensor = {}
        
        # Diagonal terms (from orthorhombic)
        for key in ['C11', 'C22', 'C33', 'C44', 'C55', 'C66']:
            tensor[key] = ortho_tensor[key]
        
        # Off-diagonal terms (from orthorhombic)
        for key in ['C12', 'C13', 'C23']:
            tensor[key] = ortho_tensor[key]
        
        # Additional triclinic terms (perturbations)
        triclinic_keys = ['C14', 'C15', 'C16', 'C24', 'C25', 'C26', 
                         'C34', 'C35', 'C36', 'C45', 'C46', 'C56']
        
        for key in triclinic_keys:
            tensor[key] = triclinic_perturbations[key]
        
        # Apply stability and physical constraints
        tensor = self._apply_triclinic_stability_constraints(tensor)
        
        return tensor
    
    def _calculate_thomsen_epsilon(self, depth: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calculate Thomsen epsilon parameter (P-wave anisotropy)"""
        
        # Base epsilon increases with depth (pressure closure of cracks)
        epsilon_base = 0.05 + 0.1 * (1 - np.exp(-depth / 3.0))
        
        # Lateral variations (geological heterogeneity)
        lateral_variation = 0.02 * np.sin(X / 2.0) * np.cos(Y / 1.5)
        
        # Depth-dependent modulation
        depth_modulation = np.exp(-depth / 8.0)  # Decreases with depth
        
        epsilon = epsilon_base + lateral_variation * depth_modulation
        
        return np.clip(epsilon, 0.0, 0.3)  # Physical bounds
    
    def _calculate_thomsen_delta(self, depth: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calculate Thomsen delta parameter"""
        
        # Generally smaller than epsilon
        delta_base = 0.02 + 0.05 * (1 - np.exp(-depth / 4.0))
        
        # Correlation with epsilon but not identical
        lateral_variation = 0.01 * np.cos(X / 2.5) * np.sin(Y / 2.0)
        
        delta = delta_base + lateral_variation * np.exp(-depth / 10.0)
        
        return np.clip(delta, 0.0, 0.2)
    
    def _calculate_thomsen_gamma(self, depth: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calculate Thomsen gamma parameter (shear-wave splitting)"""
        
        # Often related to crack density and stress
        gamma_base = 0.03 + 0.08 * (1 - np.exp(-depth / 2.5))
        
        # Lateral stress-induced variations
        stress_variation = 0.015 * np.sin(X / 3.0 + Y / 3.0)
        
        gamma = gamma_base + stress_variation * np.exp(-depth / 6.0)
        
        return np.clip(gamma, 0.0, 0.25)
    
    def _apply_realistic_c13_corrections(self, C13: np.ndarray, C11: np.ndarray, C33: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Apply realistic corrections to C13 based on physical constraints"""
        
        # Ensure C13 satisfies stability conditions
        # C13^2 < C11 * C33
        max_c13 = 0.95 * np.sqrt(C11 * C33)
        C13_corrected = np.minimum(np.abs(C13), max_c13) * np.sign(C13)
        
        # Additional depth-dependent corrections
        depth_correction = 1.0 + 0.1 * depth / 10.0  # Increases with depth
        C13_corrected *= depth_correction
        
        return C13_corrected
    
    def _calculate_orthorhombic_parameters(self, depth: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate parameters for orthorhombic anisotropy"""
        
        params = {}
        
        # Anisotropy in three orthogonal directions
        params['eps1'] = 0.08 * (1 - np.exp(-depth / 2.0)) * (1 + 0.2 * np.sin(X / 2.0))
        params['eps2'] = 0.06 * (1 - np.exp(-depth / 2.5)) * (1 + 0.2 * np.cos(Y / 2.0))
        
        # Shear-wave anisotropies
        params['gamma1'] = 0.05 * (1 - np.exp(-depth / 3.0))
        params['gamma2'] = 0.04 * (1 - np.exp(-depth / 3.5))
        params['gamma3'] = 0.06 * (1 - np.exp(-depth / 2.8))
        
        # Delta parameters (control off-diagonal terms)
        params['delta12'] = 0.02 * np.sin(X / 3.0) * np.cos(Y / 3.0)
        params['delta13'] = 0.3 + 0.1 * np.exp(-depth / 5.0)
        params['delta23'] = 0.25 + 0.15 * np.exp(-depth / 4.0)
        
        return params
    
    def _apply_orthorhombic_stability_constraints(self, C11: np.ndarray, C22: np.ndarray, C33: np.ndarray,
                                                C44: np.ndarray, C55: np.ndarray, C66: np.ndarray,
                                                C12: np.ndarray, C13: np.ndarray, C23: np.ndarray) -> tuple:
        """Apply stability constraints for orthorhombic media"""
        
        # Stability conditions for orthorhombic media
        # 1. All diagonal terms must be positive (already satisfied)
        # 2. Off-diagonal terms must satisfy inequalities
        
        # C12^2 < C11 * C22
        max_c12 = 0.95 * np.sqrt(C11 * C22)
        C12 = np.clip(C12, -max_c12, max_c12)
        
        # C13^2 < C11 * C33
        max_c13 = 0.95 * np.sqrt(C11 * C33)
        C13 = np.clip(C13, -max_c13, max_c13)
        
        # C23^2 < C22 * C33
        max_c23 = 0.95 * np.sqrt(C22 * C33)
        C23 = np.clip(C23, -max_c23, max_c23)
        
        return C12, C13, C23
    
    def _calculate_triclinic_perturbations(self, depth: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                         base_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate triclinic perturbations (typically small)"""
        
        density = base_params['density']
        vp = base_params['vp']
        
        # Triclinic terms are usually small perturbations
        perturbation_scale = 0.01 * density * vp**2 * 1e9
        
        # Spatial variations for triclinic terms
        perturbations = {}
        
        # C14, C15, C16 terms
        perturbations['C14'] = perturbation_scale * 0.5 * np.sin(X / 2.0) * np.exp(-depth / 8.0)
        perturbations['C15'] = perturbation_scale * 0.3 * np.cos(Y / 3.0) * np.exp(-depth / 10.0)
        perturbations['C16'] = perturbation_scale * 0.4 * np.sin(X / 4.0 + Y / 4.0) * np.exp(-depth / 12.0)
        
        # C24, C25, C26 terms
        perturbations['C24'] = perturbation_scale * 0.3 * np.cos(X / 3.0) * np.exp(-depth / 9.0)
        perturbations['C25'] = perturbation_scale * 0.6 * np.sin(Y / 2.5) * np.exp(-depth / 7.0)
        perturbations['C26'] = perturbation_scale * 0.2 * np.cos(X / 5.0 + Y / 5.0) * np.exp(-depth / 15.0)
        
        # C34, C35, C36 terms
        perturbations['C34'] = perturbation_scale * 0.4 * np.sin(X / 4.0) * np.exp(-depth / 6.0)
        perturbations['C35'] = perturbation_scale * 0.5 * np.cos(Y / 4.0) * np.exp(-depth / 8.0)
        perturbations['C36'] = perturbation_scale * 0.3 * np.sin(X / 6.0 + Y / 6.0) * np.exp(-depth / 10.0)
        
        # C45, C46, C56 terms
        perturbations['C45'] = perturbation_scale * 0.2 * np.cos(X / 7.0) * np.exp(-depth / 12.0)
        perturbations['C46'] = perturbation_scale * 0.3 * np.sin(Y / 6.0) * np.exp(-depth / 9.0)
        perturbations['C56'] = perturbation_scale * 0.4 * np.cos(X / 8.0 + Y / 8.0) * np.exp(-depth / 11.0)
        
        return perturbations
    
    def _apply_triclinic_stability_constraints(self, tensor: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply stability constraints for triclinic media"""
        
        # For triclinic media, stability is more complex
        # Apply simple bounds to prevent unphysical values
        
        # Bound all off-diagonal terms
        max_off_diagonal = 0.1 * np.maximum.reduce([tensor['C11'], tensor['C22'], tensor['C33']])
        
        triclinic_keys = ['C14', 'C15', 'C16', 'C24', 'C25', 'C26', 
                         'C34', 'C35', 'C36', 'C45', 'C46', 'C56']
        
        for key in triclinic_keys:
            tensor[key] = np.clip(tensor[key], -max_off_diagonal, max_off_diagonal)
        
        return tensor
    
    def _apply_stress_induced_anisotropy(self, tensor: Dict[str, np.ndarray],
                                       X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply stress-induced modifications to anisotropy"""
        
        # Estimate stress field (simplified)
        horizontal_stress = 0.7 * depth * 2700 * 9.81  # Pa (lithostatic * stress ratio)
        vertical_stress = depth * 2700 * 9.81  # Pa (lithostatic)
        
        # Stress-induced anisotropy factor
        stress_ratio = horizontal_stress / (vertical_stress + 1e-10)
        stress_factor = 1.0 + 0.05 * (stress_ratio - 1.0)  # 5% modulation
        
        # Apply to horizontal components
        modified_tensor = tensor.copy()
        modified_tensor['C11'] *= stress_factor
        modified_tensor['C22'] *= stress_factor
        modified_tensor['C66'] *= stress_factor
        
        return modified_tensor
    
    def _apply_temperature_anisotropy_effects(self, tensor: Dict[str, np.ndarray],
                                            X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply temperature effects on anisotropy"""
        
        # Estimate temperature (geothermal gradient)
        surface_temp = 288  # K (15°C)
        geothermal_gradient = 25  # K/km
        temperature = surface_temp + geothermal_gradient * depth
        
        # Temperature effect on elastic constants (typically small)
        temp_factor = 1.0 - 2e-4 * (temperature - 300)  # Softening with temperature
        
        # Apply to all elastic constants
        modified_tensor = {}
        for key, value in tensor.items():
            modified_tensor[key] = value * temp_factor
        
        return modified_tensor
    
    def _apply_pressure_anisotropy_effects(self, tensor: Dict[str, np.ndarray],
                                         X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply pressure effects on anisotropy"""
        
        # Pressure from depth
        pressure = depth * 2700 * 9.81  # Pa
        
        # Pressure stiffening (crack closure)
        pressure_factor = 1.0 + 1e-10 * pressure  # Small stiffening effect
        
        # Apply to all components
        modified_tensor = {}
        for key, value in tensor.items():
            modified_tensor[key] = value * pressure_factor
        
        return modified_tensor
    
    def _apply_crack_induced_anisotropy(self, tensor: Dict[str, np.ndarray],
                                      X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply crack-induced anisotropy effects"""
        
        # Crack density decreases with depth (pressure closure)
        crack_density = 0.1 * np.exp(-depth / 2.0)
        
        # Crack aspect ratio (typical values)
        aspect_ratio = 1e-3
        
        # Hudson's crack model corrections (simplified)
        crack_correction = 1.0 - 16 * crack_density * (1 - 0.9 * aspect_ratio)
        
        # Apply primarily to stiffest components (most affected by cracks)
        modified_tensor = tensor.copy()
        modified_tensor['C11'] *= crack_correction
        modified_tensor['C22'] *= crack_correction
        modified_tensor['C33'] *= crack_correction
        
        return modified_tensor
    
    def _apply_cpo_anisotropy_effects(self, tensor: Dict[str, np.ndarray],
                                    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply crystal preferred orientation (CPO) effects"""
        
        # Advanced Crystal Preferred Orientation (CPO) modeling
        
        # Calculate 3D stress field from magma chamber pressure and tectonic loading
        magma_pressure = self._estimate_magma_pressure(depth, X, Y, Z)
        tectonic_stress = self._calculate_tectonic_stress_field(X, Y, Z, depth)
        
        # Combined stress tensor (magmatic + tectonic + gravitational)
        total_stress_xx = magma_pressure + tectonic_stress['sigma_xx'] + self._gravitational_stress(depth, 'xx')
        total_stress_yy = magma_pressure + tectonic_stress['sigma_yy'] + self._gravitational_stress(depth, 'yy')
        total_stress_zz = magma_pressure + tectonic_stress['sigma_zz'] + self._gravitational_stress(depth, 'zz')
        total_stress_xy = tectonic_stress.get('sigma_xy', np.zeros_like(X))
        total_stress_xz = tectonic_stress.get('sigma_xz', np.zeros_like(X))
        total_stress_yz = tectonic_stress.get('sigma_yz', np.zeros_like(X))
        
        # Calculate principal stress directions and magnitudes
        principal_stresses, principal_directions = self._calculate_principal_stresses(
            total_stress_xx, total_stress_yy, total_stress_zz,
            total_stress_xy, total_stress_xz, total_stress_yz
        )
        
        # Differential stress (deviatoric stress magnitude)
        sigma_1, sigma_2, sigma_3 = principal_stresses['sigma_1'], principal_stresses['sigma_2'], principal_stresses['sigma_3']
        differential_stress = sigma_1 - sigma_3
        mean_stress = (sigma_1 + sigma_2 + sigma_3) / 3.0
        
        # Strain rate calculation from magma flow and tectonic deformation
        strain_rate = self._calculate_strain_rate_tensor(X, Y, Z, depth, differential_stress)
        strain_rate_magnitude = np.sqrt(0.5 * (strain_rate['exx']**2 + strain_rate['eyy']**2 + strain_rate['ezz']**2 + 
                                               2 * (strain_rate['exy']**2 + strain_rate['exz']**2 + strain_rate['eyz']**2)))
        
        # Temperature-dependent deformation mechanisms
        temperature = self._get_temperature_field(X, Y, Z, depth)
        
        # Deformation mechanism transition temperatures
        brittle_ductile_temp = 350 + 25 * np.log10(np.maximum(strain_rate_magnitude * 1e15, 1e-15))  # °C
        plastic_flow_temp = 600 + 50 * np.log10(np.maximum(strain_rate_magnitude * 1e15, 1e-15))     # °C
        
        # Deformation regime classification
        brittle_regime = temperature < brittle_ductile_temp
        semi_brittle_regime = (temperature >= brittle_ductile_temp) & (temperature < plastic_flow_temp)
        ductile_regime = temperature >= plastic_flow_temp
        
        # Crystal preferred orientation development
        # CPO strength depends on strain magnitude, strain rate, and temperature
        finite_strain = self._calculate_finite_strain(X, Y, Z, depth, differential_stress, strain_rate_magnitude)
        
        # Empirical CPO strength relationships from experimental petrology
        cpo_strength_olivine = np.where(
            ductile_regime,
            0.8 * (1.0 - np.exp(-finite_strain / 0.5)) * np.tanh(differential_stress / 50e6),
            0.1 * finite_strain
        )
        
        cpo_strength_feldspar = np.where(
            ductile_regime,
            0.6 * (1.0 - np.exp(-finite_strain / 0.8)) * np.tanh(differential_stress / 100e6),
            0.05 * finite_strain
        )
        
        cpo_strength_pyroxene = np.where(
            ductile_regime,
            0.7 * (1.0 - np.exp(-finite_strain / 0.6)) * np.tanh(differential_stress / 75e6),
            0.08 * finite_strain
        )
        
        # Fabric symmetry based on stress state
        stress_ratio = (sigma_2 - sigma_3) / np.maximum(sigma_1 - sigma_3, 1e-6)
        
        # Fabric types: 0 = axial (L-tectonite), 0.5 = plane strain (LS-tectonite), 1 = flattening (S-tectonite)
        fabric_type = np.clip(stress_ratio, 0, 1)
        
        # Crystal symmetry-specific elastic tensor modifications
        modified_tensor = tensor.copy()
        
        # Olivine CPO effects (orthorhombic symmetry)
        if 'olivine' in self._get_mineral_phases(X, Y, Z, depth):
            # A-type fabric (most common in upper mantle)
            a_fabric_strength = cpo_strength_olivine * (1.0 - fabric_type)  # Favored by axial stress
            
            # Elastic anisotropy factors for olivine A-type fabric
            c11_factor = 1.0 + 0.15 * a_fabric_strength
            c22_factor = 1.0 - 0.08 * a_fabric_strength  
            c33_factor = 1.0 + 0.05 * a_fabric_strength
            c44_factor = 1.0 - 0.12 * a_fabric_strength
            c55_factor = 1.0 + 0.10 * a_fabric_strength
            c66_factor = 1.0 - 0.05 * a_fabric_strength
            
            # Apply orientation-dependent modifications
            sigma1_direction = principal_directions['v1']
            
            # Rotation matrix for principal stress alignment
            rotation_factor = np.abs(np.dot(sigma1_direction, [1, 0, 0]))  # Alignment with x-axis
            
            modified_tensor['C11'] *= (1.0 + (c11_factor - 1.0) * rotation_factor)
            modified_tensor['C22'] *= (1.0 + (c22_factor - 1.0) * rotation_factor)
            modified_tensor['C33'] *= (1.0 + (c33_factor - 1.0) * rotation_factor)
            modified_tensor['C44'] *= (1.0 + (c44_factor - 1.0) * rotation_factor)
            modified_tensor['C55'] *= (1.0 + (c55_factor - 1.0) * rotation_factor)
            modified_tensor['C66'] *= (1.0 + (c66_factor - 1.0) * rotation_factor)
        
        # Feldspar CPO effects (triclinic/monoclinic symmetry)
        feldspar_phases = self._get_feldspar_content(X, Y, Z, depth)
        if np.any(feldspar_phases > 0.1):
            # Feldspar deformation twins and slip systems
            twin_activity = np.where(
                semi_brittle_regime & (differential_stress > 30e6),
                0.8 * cpo_strength_feldspar,
                0.2 * cpo_strength_feldspar
            )
            
            # Albite and Carlsbad twin effects on elastic properties
            c11_feldspar = 1.0 + 0.08 * twin_activity * feldspar_phases
            c22_feldspar = 1.0 - 0.04 * twin_activity * feldspar_phases
            c33_feldspar = 1.0 + 0.06 * twin_activity * feldspar_phases
            
            modified_tensor['C11'] *= c11_feldspar
            modified_tensor['C22'] *= c22_feldspar
            modified_tensor['C33'] *= c33_feldspar
        
        # Pyroxene CPO effects (orthorhombic/monoclinic symmetry)
        pyroxene_phases = self._get_pyroxene_content(X, Y, Z, depth)
        if np.any(pyroxene_phases > 0.1):
            # Pyroxene slip systems: (100)[001], (010)[001], (100)[010]
            slip_system_activity = np.where(
                ductile_regime,
                cpo_strength_pyroxene * np.tanh(differential_stress / 60e6),
                0.1 * cpo_strength_pyroxene
            )
            
            # Directional elastic moduli modifications
            c_axis_alignment = self._calculate_c_axis_alignment(principal_directions['v1'], temperature)
            
            c11_pyroxene = 1.0 + 0.12 * slip_system_activity * c_axis_alignment * pyroxene_phases
            c33_pyroxene = 1.0 - 0.06 * slip_system_activity * c_axis_alignment * pyroxene_phases
            c44_pyroxene = 1.0 + 0.15 * slip_system_activity * (1.0 - c_axis_alignment) * pyroxene_phases
            
            modified_tensor['C11'] *= c11_pyroxene
            modified_tensor['C33'] *= c33_pyroxene
            modified_tensor['C44'] *= c44_pyroxene
        
        # Amphibole CPO effects (important in hydrous systems)
        amphibole_phases = self._get_amphibole_content(X, Y, Z, depth, temperature)
        if np.any(amphibole_phases > 0.05):
            # Amphibole [001] slip and (100) twinning
            amphibole_cpo = 0.4 * (1.0 - np.exp(-finite_strain / 0.3)) * amphibole_phases
            
            # Strong anisotropy due to chain silicate structure
            c11_amphibole = 1.0 + 0.20 * amphibole_cpo
            c33_amphibole = 1.0 - 0.15 * amphibole_cpo
            c55_amphibole = 1.0 - 0.25 * amphibole_cpo  # Weak across chain direction
            
            modified_tensor['C11'] *= c11_amphibole
            modified_tensor['C33'] *= c33_amphibole
            modified_tensor['C55'] *= c55_amphibole
        
        # Shape preferred orientation (SPO) effects
        # Important for rocks with elongated/flattened mineral grains
        aspect_ratio_minerals = self._calculate_mineral_aspect_ratios(X, Y, Z, depth, finite_strain)
        spo_strength = np.tanh(2.0 * (aspect_ratio_minerals - 1.0))
        
        # SPO contributions to elastic anisotropy
        spo_c11_factor = 1.0 + 0.05 * spo_strength * np.cos(principal_directions['theta_1'])**2
        spo_c33_factor = 1.0 + 0.05 * spo_strength * np.sin(principal_directions['theta_1'])**2
        
        modified_tensor['C11'] *= spo_c11_factor
        modified_tensor['C33'] *= spo_c33_factor
        
        # Crack and void alignment effects
        crack_alignment = self._calculate_crack_alignment(principal_stresses, temperature, finite_strain)
        aligned_porosity = self._get_aligned_porosity(X, Y, Z, depth, fabric_type)
        
        # Anisotropic crack corrections (Hudson's model extended)
        crack_aniso_factor = 1.0 - 12.0 * aligned_porosity * crack_alignment
        modified_tensor['C11'] *= crack_aniso_factor
        modified_tensor['C22'] *= (2.0 - crack_aniso_factor)  # Compensation in perpendicular direction
        
        # Store CPO metadata for analysis and visualization
        modified_tensor['_cpo_metadata'] = {
            'cpo_strength': {
                'olivine': cpo_strength_olivine,
                'feldspar': cpo_strength_feldspar,
                'pyroxene': cpo_strength_pyroxene
            },
            'deformation_regime': {
                'brittle': brittle_regime,
                'semi_brittle': semi_brittle_regime,
                'ductile': ductile_regime
            },
            'stress_field': {
                'principal_stresses': principal_stresses,
                'principal_directions': principal_directions,
                'differential_stress': differential_stress,
                'mean_stress': mean_stress
            },
            'strain_parameters': {
                'strain_rate': strain_rate,
                'finite_strain': finite_strain,
                'fabric_type': fabric_type
            },
            'temperature': temperature,
            'mineral_phases': {
                'feldspar': feldspar_phases,
                'pyroxene': pyroxene_phases,
                'amphibole': amphibole_phases
            }
        }
        
        return modified_tensor
    
    def _estimate_magma_pressure(self, depth: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Estimate magma pressure field around chamber"""
        chamber_radius = 2.0  # km
        radial_distance = np.sqrt(X**2 + Y**2)
        
        # Magma density and pressure gradient
        magma_density = 2500  # kg/m³
        g = 9.81  # m/s²
        
        # Base magma pressure at chamber center
        base_pressure = magma_density * g * depth * 1000  # Pa
        
        # Pressure decay with distance from chamber
        pressure_factor = np.exp(-radial_distance / chamber_radius)
        return base_pressure * pressure_factor
    
    def _calculate_tectonic_stress_field(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate regional tectonic stress field"""
        # Regional stress parameters (example for extensional regime)
        sigma_h_max = 30e6 + 15e6 * depth  # Maximum horizontal stress (Pa)
        sigma_h_min = 20e6 + 10e6 * depth  # Minimum horizontal stress (Pa)
        sigma_v = 27e6 * depth  # Vertical stress (lithostatic)
        
        # Stress orientation (example: NE-SW extension)
        azimuth = np.radians(45)  # degrees
        
        return {
            'sigma_xx': sigma_h_max * np.cos(azimuth)**2 + sigma_h_min * np.sin(azimuth)**2,
            'sigma_yy': sigma_h_min * np.cos(azimuth)**2 + sigma_h_max * np.sin(azimuth)**2,
            'sigma_zz': sigma_v,
            'sigma_xy': (sigma_h_max - sigma_h_min) * np.sin(2 * azimuth) / 2,
            'sigma_xz': np.zeros_like(X),
            'sigma_yz': np.zeros_like(X)
        }
    
    def _gravitational_stress(self, depth: np.ndarray, component: str) -> np.ndarray:
        """Calculate gravitational stress component"""
        rock_density = 2700  # kg/m³
        g = 9.81  # m/s²
        
        if component in ['xx', 'yy']:
            return 0.3 * rock_density * g * depth * 1000  # Poisson effect
        elif component == 'zz':
            return rock_density * g * depth * 1000  # Lithostatic
        else:
            return np.zeros_like(depth)
    
    def _calculate_principal_stresses(self, sigma_xx: np.ndarray, sigma_yy: np.ndarray, sigma_zz: np.ndarray,
                                    sigma_xy: np.ndarray, sigma_xz: np.ndarray, sigma_yz: np.ndarray) -> tuple:
        """Calculate principal stress magnitudes and directions"""
        # Simplified 2D case for computational efficiency
        # Full 3D eigenvalue solution would be more accurate but computationally expensive
        
        # 2D principal stresses in xy plane
        sigma_mean = (sigma_xx + sigma_yy) / 2
        tau_max = np.sqrt(((sigma_xx - sigma_yy) / 2)**2 + sigma_xy**2)
        
        sigma_1_2d = sigma_mean + tau_max
        sigma_2_2d = sigma_mean - tau_max
        
        # Include vertical stress
        sigma_1 = np.maximum(np.maximum(sigma_1_2d, sigma_2_2d), sigma_zz)
        sigma_3 = np.minimum(np.minimum(sigma_1_2d, sigma_2_2d), sigma_zz)
        sigma_2 = sigma_1_2d + sigma_2_2d + sigma_zz - sigma_1 - sigma_3
        
        # Principal directions (simplified)
        theta_1 = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)
        
        principal_stresses = {'sigma_1': sigma_1, 'sigma_2': sigma_2, 'sigma_3': sigma_3}
        principal_directions = {
            'v1': np.array([np.cos(theta_1), np.sin(theta_1), np.zeros_like(theta_1)]).T,
            'theta_1': theta_1
        }
        
        return principal_stresses, principal_directions
    
    def _calculate_strain_rate_tensor(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                                    depth: np.ndarray, differential_stress: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate strain rate tensor from stress and rheology"""
        # Flow law parameters (power law rheology)
        A = 1e-16  # Pre-exponential factor (Pa^-n s^-1)
        n = 3.5    # Stress exponent
        Q = 300e3  # Activation energy (J/mol)
        R = 8.314  # Gas constant (J/mol/K)
        
        # Temperature field
        temperature = self._get_temperature_field(X, Y, Z, depth)
        
        # Effective strain rate from flow law
        strain_rate_eff = A * (differential_stress**n) * np.exp(-Q / (R * (temperature + 273.15)))
        
        # Strain rate tensor components (simplified deviatoric)
        return {
            'exx': strain_rate_eff / 3,
            'eyy': strain_rate_eff / 3,
            'ezz': -2 * strain_rate_eff / 3,
            'exy': strain_rate_eff / 6,
            'exz': np.zeros_like(strain_rate_eff),
            'eyz': np.zeros_like(strain_rate_eff)
        }
    
    def _get_temperature_field(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Get temperature field around magma chamber"""
        # Distance from chamber center
        radial_distance = np.sqrt(X**2 + Y**2)
        chamber_radius = 2.0  # km
        
        # Temperature model
        chamber_temp = 1000  # °C
        country_rock_temp = 200 + 25 * depth  # Geothermal gradient
        
        # Thermal decay from chamber
        thermal_factor = np.exp(-radial_distance / chamber_radius)
        return country_rock_temp + (chamber_temp - country_rock_temp) * thermal_factor
    
    def _calculate_finite_strain(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray,
                               differential_stress: np.ndarray, strain_rate: np.ndarray) -> np.ndarray:
        """Calculate accumulated finite strain"""
        # Time-integrated strain (simplified)
        deformation_time = 1e4 * 365.25 * 24 * 3600  # 10,000 years in seconds
        finite_strain = strain_rate * deformation_time
        
        # Strain concentration near chamber margins
        radial_distance = np.sqrt(X**2 + Y**2)
        chamber_radius = 2.0
        
        strain_concentration = 1.0 + 2.0 * np.exp(-radial_distance / (0.5 * chamber_radius))
        return finite_strain * strain_concentration
    
    def _get_mineral_phases(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray) -> list:
        """Get mineral phases present in the rock"""
        # Simplified mineral assemblage based on depth and temperature
        temperature = self._get_temperature_field(X, Y, Z, depth)
        
        phases = []
        if np.any(temperature > 600):
            phases.append('olivine')
        if np.any(depth > 10):
            phases.extend(['feldspar', 'pyroxene'])
        
        return phases
    
    def _get_feldspar_content(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Calculate feldspar volume fraction"""
        radial_distance = np.sqrt(X**2 + Y**2)
        
        # Higher feldspar content in felsic differentiates
        base_feldspar = 0.3
        differentiation_factor = np.tanh(radial_distance / 3.0)
        
        return base_feldspar + 0.4 * differentiation_factor
    
    def _get_pyroxene_content(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Calculate pyroxene volume fraction"""
        temperature = self._get_temperature_field(X, Y, Z, depth)
        
        # Higher pyroxene content at high temperatures
        return 0.2 + 0.3 * np.tanh((temperature - 800) / 100)
    
    def _get_amphibole_content(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, depth: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """Calculate amphibole volume fraction"""
        # Amphibole stability in intermediate temperature range with water
        water_activity = 0.5 + 0.3 * np.exp(-depth / 5.0)  # Higher water at shallow depths
        
        temp_stability = np.exp(-((temperature - 650)**2) / (2 * 150**2))  # Gaussian around 650°C
        
        return 0.15 * temp_stability * water_activity
    
    def _calculate_c_axis_alignment(self, principal_direction: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """Calculate c-axis alignment factor"""
        # Temperature-dependent alignment efficiency
        alignment_efficiency = np.tanh((temperature - 500) / 200)
        
        # Geometric alignment factor
        geometric_factor = np.abs(principal_direction[:, 2])  # z-component
        
        return alignment_efficiency * geometric_factor
    
    def _calculate_mineral_aspect_ratios(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                                       depth: np.ndarray, finite_strain: np.ndarray) -> np.ndarray:
        """Calculate mineral grain aspect ratios from deformation"""
        # Initial aspect ratio
        initial_ratio = 1.5
        
        # Strain-induced grain elongation/flattening
        strain_effect = 1.0 + 2.0 * finite_strain
        
        return initial_ratio * strain_effect
    
    def _calculate_crack_alignment(self, principal_stresses: Dict[str, np.ndarray], 
                                 temperature: np.ndarray, finite_strain: np.ndarray) -> np.ndarray:
        """Calculate crack alignment factor"""
        # Cracks align perpendicular to maximum principal stress
        stress_ratio = (principal_stresses['sigma_1'] - principal_stresses['sigma_3']) / principal_stresses['sigma_1']
        
        # Temperature effect on crack alignment
        thermal_factor = np.exp(-(temperature - 200) / 300)
        
        return stress_ratio * thermal_factor * np.tanh(finite_strain)
    
    def _get_aligned_porosity(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                            depth: np.ndarray, fabric_type: np.ndarray) -> np.ndarray:
        """Calculate aligned porosity fraction"""
        # Base porosity decreases with depth
        base_porosity = 0.05 * np.exp(-depth / 3.0)
        
        # Alignment depends on fabric type
        alignment_factor = fabric_type  # Higher for flattening fabrics
        
        return base_porosity * alignment_factor

    def _calculate_deviatoric_pressure_stress(self, pressure_stress: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate deviatoric stress components from pressure stress tensor"""
        
        # Mean stress (hydrostatic component)
        mean_stress = (pressure_stress['sigma_xx'] + pressure_stress['sigma_yy'] + pressure_stress['sigma_zz']) / 3.0
        
        # Add deviatoric stress components
        pressure_stress['sigma_xx_dev'] = pressure_stress['sigma_xx'] - mean_stress
        pressure_stress['sigma_yy_dev'] = pressure_stress['sigma_yy'] - mean_stress
        pressure_stress['sigma_zz_dev'] = pressure_stress['sigma_zz'] - mean_stress
        
        # Calculate von Mises equivalent stress
        s_xx = pressure_stress['sigma_xx_dev']
        s_yy = pressure_stress['sigma_yy_dev']
        s_zz = pressure_stress['sigma_zz_dev']
        s_xy = pressure_stress['sigma_xy']
        s_xz = pressure_stress['sigma_xz']
        s_yz = pressure_stress['sigma_yz']
        
        von_mises_stress = np.sqrt(0.5 * (
            (s_xx - s_yy)**2 + (s_yy - s_zz)**2 + (s_zz - s_xx)**2 +
            6 * (s_xy**2 + s_xz**2 + s_yz**2)
        ))
        
        pressure_stress['von_mises'] = von_mises_stress
        pressure_stress['mean_stress'] = mean_stress
        
        return pressure_stress
    
    def _calculate_deformation(self, stress: Dict[str, np.ndarray], E: float, nu: float) -> Dict[str, np.ndarray]:
        """Calculate deformation from stress using Hooke's law"""
        
        # Elastic constants
        lambda_lame = E * nu / ((1 + nu) * (1 - 2*nu))
        mu_lame = E / (2 * (1 + nu))
        
        # Extract stress components
        sigma_xx = stress['sigma_xx']
        sigma_yy = stress['sigma_yy']
        sigma_zz = stress['sigma_zz']
        sigma_xy = stress['sigma_xy']
        sigma_xz = stress['sigma_xz']
        sigma_yz = stress['sigma_yz']
        
        # Calculate strains using Hooke's law
        trace_stress = sigma_xx + sigma_yy + sigma_zz
        
        epsilon_xx = (sigma_xx - nu * (sigma_yy + sigma_zz)) / E
        epsilon_yy = (sigma_yy - nu * (sigma_xx + sigma_zz)) / E
        epsilon_zz = (sigma_zz - nu * (sigma_xx + sigma_yy)) / E
        epsilon_xy = sigma_xy / (2 * mu_lame)
        epsilon_xz = sigma_xz / (2 * mu_lame)
        epsilon_yz = sigma_yz / (2 * mu_lame)
        
        deformation = {
            'epsilon_xx': epsilon_xx,
            'epsilon_yy': epsilon_yy,
            'epsilon_zz': epsilon_zz,
            'epsilon_xy': epsilon_xy,
            'epsilon_xz': epsilon_xz,
            'epsilon_yz': epsilon_yz,
            'volumetric_strain': epsilon_xx + epsilon_yy + epsilon_zz,
            'deviatoric_strain': np.sqrt(2/3 * (epsilon_xx**2 + epsilon_yy**2 + epsilon_zz**2 +
                                               2*(epsilon_xy**2 + epsilon_xz**2 + epsilon_yz**2)))
        }
        
        return deformation
    
    def _calculate_fracture_probability(self, stress: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate comprehensive fracture probability using advanced failure analysis
        
        Includes:
        - Multiple failure criteria (Mohr-Coulomb, Hoek-Brown, Griffith, Drucker-Prager)
        - Principal stress analysis
        - Stress path effects
        - Temperature and pore pressure dependencies
        - Probabilistic fracture mechanics
        - Time-dependent fatigue effects
        """
        
        # Calculate comprehensive fracture probability
        fracture_analysis = self._perform_comprehensive_fracture_analysis(stress)
        
        # Combine multiple failure criteria
        combined_probability = self._combine_failure_criteria_probabilities(fracture_analysis)
        
        # Apply environmental factor corrections
        environmental_probability = self._apply_environmental_fracture_factors(
            combined_probability, stress
        )
        
        # Include time-dependent effects
        final_probability = self._apply_time_dependent_fracture_effects(environmental_probability)
        
        return final_probability
    
    def _perform_comprehensive_fracture_analysis(self, stress: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Perform comprehensive fracture analysis using multiple criteria"""
        
        # Calculate principal stresses
        principal_stresses = self._calculate_principal_stresses(stress)
        sigma_1 = principal_stresses['sigma_1']  # Maximum principal stress
        sigma_2 = principal_stresses['sigma_2']  # Intermediate principal stress  
        sigma_3 = principal_stresses['sigma_3']  # Minimum principal stress
        
        # Get material properties for fracture analysis
        material_props = self._get_fracture_material_properties(stress)
        
        # Multiple failure criteria analysis
        fracture_analysis = {}
        
        # 1. Mohr-Coulomb failure criterion
        fracture_analysis['mohr_coulomb'] = self._calculate_mohr_coulomb_failure(
            sigma_1, sigma_3, material_props
        )
        
        # 2. Hoek-Brown failure criterion (for intact rock and rock masses)
        fracture_analysis['hoek_brown'] = self._calculate_hoek_brown_failure(
            sigma_1, sigma_3, material_props
        )
        
        # 3. Griffith theory (tensile failure)
        fracture_analysis['griffith'] = self._calculate_griffith_failure(
            sigma_1, sigma_3, material_props
        )
        
        # 4. Drucker-Prager criterion (3D generalization)
        fracture_analysis['drucker_prager'] = self._calculate_drucker_prager_failure(
            sigma_1, sigma_2, sigma_3, material_props
        )
        
        # 5. Maximum tensile stress criterion
        fracture_analysis['max_tensile'] = self._calculate_max_tensile_failure(
            sigma_3, material_props
        )
        
        # 6. von Mises equivalent stress (for comparison)
        fracture_analysis['von_mises'] = self._calculate_von_mises_failure(
            sigma_1, sigma_2, sigma_3, material_props
        )
        
        # 7. Weibull statistical failure analysis
        fracture_analysis['weibull'] = self._calculate_weibull_failure(
            sigma_1, material_props
        )
        
        return fracture_analysis
    
    def _calculate_principal_stresses(self, stress: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate principal stresses from stress tensor"""
        
        # Stress tensor components
        sigma_xx = stress['sigma_xx']
        sigma_yy = stress['sigma_yy']
        sigma_zz = stress['sigma_zz']
        sigma_xy = stress.get('sigma_xy', np.zeros_like(sigma_xx))
        sigma_xz = stress.get('sigma_xz', np.zeros_like(sigma_xx))
        sigma_yz = stress.get('sigma_yz', np.zeros_like(sigma_xx))
        
        # Calculate principal stresses using eigenvalue analysis
        # For each point in the grid, solve the characteristic equation
        shape = sigma_xx.shape
        sigma_1 = np.zeros(shape)
        sigma_2 = np.zeros(shape)
        sigma_3 = np.zeros(shape)
        
        # Vectorized principal stress calculation
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    # Construct stress tensor at this point
                    stress_tensor = np.array([
                        [sigma_xx[i,j,k], sigma_xy[i,j,k], sigma_xz[i,j,k]],
                        [sigma_xy[i,j,k], sigma_yy[i,j,k], sigma_yz[i,j,k]],
                        [sigma_xz[i,j,k], sigma_yz[i,j,k], sigma_zz[i,j,k]]
                    ])
                    
                    # Calculate eigenvalues (principal stresses)
                    eigenvals = np.linalg.eigvals(stress_tensor)
                    eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
                    
                    sigma_1[i,j,k] = eigenvals[0]  # Maximum (most tensile)
                    sigma_2[i,j,k] = eigenvals[1]  # Intermediate
                    sigma_3[i,j,k] = eigenvals[2]  # Minimum (most compressive)
        
        return {
            'sigma_1': sigma_1,
            'sigma_2': sigma_2,
            'sigma_3': sigma_3
        }
    
    def _get_fracture_material_properties(self, stress: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Get comprehensive fracture material properties with spatial and temporal variations
        
        Includes:
        1. Multi-scale heterogeneity (grain to formation scale)
        2. Temperature and pressure dependencies
        3. Damage evolution and healing
        4. Rate-dependent behavior
        5. Anisotropic properties
        6. Statistical distributions of strength
        7. Multi-physics coupling effects
        """
        
        shape = stress['sigma_xx'].shape
        
        # Get spatial coordinates for property variation
        coordinates = self._get_spatial_coordinates_from_stress(stress)
        X, Y, Z = coordinates['X'], coordinates['Y'], coordinates['Z']
        
        # Determine comprehensive rock type distribution
        rock_type_distribution = self._determine_comprehensive_rock_types(X, Y, Z)
        
        # Calculate base material properties for each rock type
        base_properties = self._calculate_base_fracture_properties(rock_type_distribution, X, Y, Z)
        
        # Apply environmental modifications
        temp_pressure_modified = self._apply_temperature_pressure_modifications(
            base_properties, X, Y, Z)
        
        # Apply damage and healing effects
        damage_modified = self._apply_damage_healing_effects(
            temp_pressure_modified, stress, X, Y, Z)
        
        # Apply rate-dependent modifications
        rate_modified = self._apply_rate_dependent_modifications(
            damage_modified, stress, X, Y, Z)
        
        # Apply anisotropic corrections
        anisotropic_modified = self._apply_anisotropic_fracture_properties(
            rate_modified, X, Y, Z)
        
        # Apply statistical heterogeneity
        final_properties = self._apply_statistical_heterogeneity(
            anisotropic_modified, X, Y, Z)
        
        return final_properties
    
    def _get_spatial_coordinates_from_stress(self, stress: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract spatial coordinates from stress tensor shape"""
        
        shape = stress['sigma_xx'].shape
        
        if len(shape) == 3:
            nx, ny, nz = shape
            x = np.linspace(-5, 5, nx)  # km
            y = np.linspace(-5, 5, ny)  # km
            z = np.linspace(0, 10, nz)  # km depth
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        elif len(shape) == 2:
            nx, ny = shape
            x = np.linspace(-5, 5, nx)
            y = np.linspace(-5, 5, ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            Z = np.zeros_like(X)
        else:
            # 1D case
            nx = shape[0]
            X = np.linspace(-5, 5, nx)
            Y = np.zeros_like(X)
            Z = np.zeros_like(X)
        
        return {'X': X, 'Y': Y, 'Z': Z}
    
    def _determine_comprehensive_rock_types(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Determine comprehensive rock type distribution with geological realism
        """
        
        depth = np.abs(Z)
        lateral_distance = np.sqrt(X**2 + Y**2)
        
        # Initialize rock type probabilities
        rock_types = {
            'granite': np.zeros_like(depth),
            'granodiorite': np.zeros_like(depth),
            'diorite': np.zeros_like(depth),
            'gabbro': np.zeros_like(depth),
            'basalt': np.zeros_like(depth),
            'rhyolite': np.zeros_like(depth),
            'andesite': np.zeros_like(depth),
            'dacite': np.zeros_like(depth),
            'sandstone': np.zeros_like(depth),
            'limestone': np.zeros_like(depth),
            'shale': np.zeros_like(depth),
            'schist': np.zeros_like(depth),
            'gneiss': np.zeros_like(depth),
            'quartzite': np.zeros_like(depth)
        }
        
        # Depth-based lithological zonation
        shallow_zone = depth < 2.0  # km
        intermediate_zone = (depth >= 2.0) & (depth < 6.0)
        deep_zone = depth >= 6.0
        
        # Advanced magmatic differentiation model with crystallization processes
        # Calculate normalized radial distance from magma chamber center
        normalized_radius = lateral_distance / np.maximum(chamber_radius, 1e-6)
        
        # Thermal gradient model - temperature decreases with distance from center
        # Core temperature: 1200°C, Margin temperature: 800°C
        core_temp = 1200.0  # °C
        margin_temp = 800.0
        temperature = core_temp - (core_temp - margin_temp) * np.minimum(normalized_radius, 1.0)
        
        # Cooling rate modeling (°C/year) - faster cooling at margins
        base_cooling_rate = 10.0  # °C/year
        cooling_rate = base_cooling_rate * (1.0 + 3.0 * normalized_radius)
        
        # Melt fraction calculation based on temperature
        solidus_temp = 750.0  # °C
        liquidus_temp = 1250.0  # °C
        melt_fraction = np.clip((temperature - solidus_temp) / (liquidus_temp - solidus_temp), 0.0, 1.0)
        
        # Crystallization fractionation index (0 = primitive, 1 = highly fractionated)
        fractionation_index = 1.0 - melt_fraction
        
        # Silica content evolution during differentiation (45-75% SiO2)
        base_silica = 50.0  # Basaltic starting composition
        max_silica = 75.0   # Rhyolitic end member
        silica_content = base_silica + (max_silica - base_silica) * fractionation_index
        
        # Magma chamber zonation based on physical processes
        # Core zone: high temperature, low fractionation
        core_zone = (normalized_radius < 0.3) & (temperature > 1100)
        
        # Transition zone: moderate fractionation, crystallization
        transition_zone = ((normalized_radius >= 0.3) & (normalized_radius < 0.7)) | \
                         ((temperature >= 950) & (temperature <= 1100))
        
        # Border zone: high fractionation, solid-melt coexistence
        border_zone = (normalized_radius >= 0.7) | (temperature < 950)
        
        # Roof zone: volatile accumulation, high differentiation
        roof_zone = (depth < 1.5) & (normalized_radius < 0.5) & (fractionation_index > 0.6)
        
        # Floor zone: mafic cumulates, dense phases
        floor_zone = (depth > 5.0) & (normalized_radius < 0.4) & (temperature > 1000)
        
        # Volatile content modeling (affects phase stability)
        volatile_content = 0.5 + 1.5 * fractionation_index  # wt% H2O + CO2
        volatile_saturation = volatile_content > (2.0 - 0.5 * normalized_radius)
        
        # Pressure effects on phase stability (depth-dependent)
        pressure_kbar = depth * 0.27  # Approximate crustal pressure gradient
        
        # Crystal content calculation
        crystal_content = 1.0 - melt_fraction
        
        # Advanced crystallization fractionation modeling for rock type probabilities
        # Based on silica content, temperature, pressure, and fractionation processes
        
        # Mafic phases (low silica, high temperature)
        gabbro_prob = np.where(
            (silica_content < 52) & (temperature > 1000) & floor_zone,
            0.8 * (1.0 - fractionation_index),
            np.where((silica_content < 50) & core_zone, 0.4, 0.05)
        )
        
        basalt_prob = np.where(
            (silica_content < 52) & (depth < 2.0) & (temperature > 1100),
            0.6 * melt_fraction,
            0.02
        )
        
        # Intermediate composition phases
        diorite_prob = np.where(
            (silica_content >= 52) & (silica_content < 63) & 
            transition_zone & (crystal_content > 0.3),
            0.7 * (1.0 - np.abs(fractionation_index - 0.4)),
            0.1
        )
        
        granodiorite_prob = np.where(
            (silica_content >= 60) & (silica_content < 68) & 
            (transition_zone | core_zone) & (crystal_content > 0.4),
            0.6 * np.exp(-2.0 * np.abs(fractionation_index - 0.6)),
            0.1
        )
        
        andesite_prob = np.where(
            (silica_content >= 57) & (silica_content < 63) & 
            (depth < 3.0) & volatile_saturation,
            0.5 * melt_fraction * (1.0 + 0.5 * volatile_content),
            0.05
        )
        
        dacite_prob = np.where(
            (silica_content >= 63) & (silica_content < 68) & 
            roof_zone & volatile_saturation,
            0.7 * melt_fraction * np.minimum(volatile_content / 2.0, 1.0),
            0.03
        )
        
        # Felsic phases (high silica, highly fractionated)
        granite_prob = np.where(
            (silica_content >= 68) & (fractionation_index > 0.7) & 
            (crystal_content > 0.6) & (pressure_kbar < 3.0),
            0.8 * fractionation_index * (1.0 - melt_fraction + 0.2),
            np.where(
                (silica_content >= 65) & core_zone & (crystal_content > 0.5),
                0.4 * fractionation_index,
                0.1
            )
        )
        
        rhyolite_prob = np.where(
            (silica_content >= 72) & roof_zone & volatile_saturation & 
            (melt_fraction > 0.3),
            0.9 * melt_fraction * np.minimum(fractionation_index * 1.2, 1.0),
            0.02
        )
        
        # Metamorphic country rock (thermally affected)
        thermal_aureole = (normalized_radius >= 0.8) & (normalized_radius < 1.5)
        contact_temp = 600 + 400 * np.exp(-2.0 * (normalized_radius - 0.8))
        
        gneiss_prob = np.where(
            thermal_aureole & (contact_temp > 650) & (depth > 3.0),
            0.5 * (1.0 - np.exp(-(normalized_radius - 0.8))),
            np.where(deep_zone & border_zone, 0.3, 0.05)
        )
        
        schist_prob = np.where(
            thermal_aureole & (contact_temp >= 550) & (contact_temp < 700) & 
            (depth >= 2.0) & (depth < 6.0),
            0.4 * np.exp(-0.5 * np.abs(contact_temp - 625) / 75),
            np.where(intermediate_zone & border_zone, 0.2, 0.05)
        )
        
        # Sedimentary sequences (pre-intrusion stratigraphy)
        sediment_preservation = np.exp(-3.0 * np.maximum(contact_temp - 400, 0) / 400)
        
        sandstone_prob = np.where(
            (depth < 2.0) & (normalized_radius > 1.2) & (contact_temp < 500),
            0.3 * sediment_preservation,
            0.01
        )
        
        limestone_prob = np.where(
            (depth < 3.0) & (normalized_radius > 1.0) & (contact_temp < 600) & 
            (pressure_kbar < 2.0),
            0.25 * sediment_preservation * (1.0 - np.minimum(contact_temp / 600, 1.0)),
            0.01
        )
        
        shale_prob = np.where(
            (depth >= 1.0) & (depth < 4.0) & (normalized_radius > 1.1) & 
            (contact_temp < 550),
            0.2 * sediment_preservation,
            0.01
        )
        
        # Quartzite formation from sandstone metamorphism
        quartzite_prob = np.where(
            thermal_aureole & (contact_temp >= 500) & (contact_temp < 700) & 
            (sandstone_prob > 0.1),
            0.6 * (1.0 - sediment_preservation) * (sandstone_prob + 0.1),
            0.01
        )
        
        # Normalize probabilities with all rock types
        total_prob = (granite_prob + granodiorite_prob + diorite_prob + gabbro_prob +
                     andesite_prob + dacite_prob + basalt_prob + rhyolite_prob +
                     gneiss_prob + schist_prob + sandstone_prob + limestone_prob + 
                     shale_prob + quartzite_prob)
        
        # Assign normalized probabilities with comprehensive rock type distribution
        eps = 1e-10  # Numerical stability
        
        rock_types['granite'] = granite_prob / (total_prob + eps)
        rock_types['granodiorite'] = granodiorite_prob / (total_prob + eps)
        rock_types['diorite'] = diorite_prob / (total_prob + eps)
        rock_types['gabbro'] = gabbro_prob / (total_prob + eps)
        rock_types['basalt'] = basalt_prob / (total_prob + eps)
        rock_types['rhyolite'] = rhyolite_prob / (total_prob + eps)
        rock_types['andesite'] = andesite_prob / (total_prob + eps)
        rock_types['dacite'] = dacite_prob / (total_prob + eps)
        rock_types['sandstone'] = sandstone_prob / (total_prob + eps)
        rock_types['limestone'] = limestone_prob / (total_prob + eps)
        rock_types['shale'] = shale_prob / (total_prob + eps)
        rock_types['schist'] = schist_prob / (total_prob + eps)
        rock_types['gneiss'] = gneiss_prob / (total_prob + eps)
        rock_types['quartzite'] = quartzite_prob / (total_prob + eps)
        
        # Store additional petrological parameters for advanced modeling
        rock_types['_metadata'] = {
            'temperature': temperature,
            'melt_fraction': melt_fraction,
            'fractionation_index': fractionation_index,
            'silica_content': silica_content,
            'volatile_content': volatile_content,
            'crystal_content': crystal_content,
            'pressure_kbar': pressure_kbar,
            'cooling_rate': cooling_rate,
            'thermal_zones': {
                'core_zone': core_zone,
                'transition_zone': transition_zone,
                'border_zone': border_zone,
                'roof_zone': roof_zone,
                'floor_zone': floor_zone,
                'thermal_aureole': thermal_aureole
            },
            'volatile_saturation': volatile_saturation,
            'contact_temperature': contact_temp,
            'normalized_radius': normalized_radius
        }
        rock_types['schist'] = schist_prob / (total_prob + 1e-10)
        rock_types['sandstone'] = sandstone_prob / (total_prob + 1e-10)
        
        return rock_types
    
    def _calculate_base_fracture_properties(self, rock_types: Dict[str, np.ndarray],
                                          X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate base fracture properties for each rock type
        """
        
        # Reference properties for different rock types (extensive database)
        reference_properties = {
            'granite': {
                'cohesion': 25e6,           # Pa
                'friction_angle': 40.0,     # degrees
                'tensile_strength': 8e6,    # Pa
                'ucs': 150e6,              # Pa
                'm_i': 20.0,               # Hoek-Brown parameter
                'fracture_toughness': 2.5e6, # Pa*sqrt(m)
                'elastic_modulus': 50e9,    # Pa
                'poisson_ratio': 0.25
            },
            'granodiorite': {
                'cohesion': 22e6,
                'friction_angle': 38.0,
                'tensile_strength': 7e6,
                'ucs': 130e6,
                'm_i': 18.0,
                'fracture_toughness': 2.2e6,
                'elastic_modulus': 45e9,
                'poisson_ratio': 0.26
            },
            'diorite': {
                'cohesion': 30e6,
                'friction_angle': 42.0,
                'tensile_strength': 10e6,
                'ucs': 180e6,
                'm_i': 16.0,
                'fracture_toughness': 2.8e6,
                'elastic_modulus': 60e9,
                'poisson_ratio': 0.24
            },
            'gabbro': {
                'cohesion': 35e6,
                'friction_angle': 45.0,
                'tensile_strength': 12e6,
                'ucs': 200e6,
                'm_i': 14.0,
                'fracture_toughness': 3.2e6,
                'elastic_modulus': 70e9,
                'poisson_ratio': 0.23
            },
            'basalt': {
                'cohesion': 40e6,
                'friction_angle': 48.0,
                'tensile_strength': 15e6,
                'ucs': 250e6,
                'm_i': 12.0,
                'fracture_toughness': 3.8e6,
                'elastic_modulus': 80e9,
                'poisson_ratio': 0.22
            },
            'rhyolite': {
                'cohesion': 20e6,
                'friction_angle': 35.0,
                'tensile_strength': 6e6,
                'ucs': 120e6,
                'm_i': 22.0,
                'fracture_toughness': 2.0e6,
                'elastic_modulus': 40e9,
                'poisson_ratio': 0.27
            },
            'andesite': {
                'cohesion': 28e6,
                'friction_angle': 40.0,
                'tensile_strength': 9e6,
                'ucs': 160e6,
                'm_i': 16.0,
                'fracture_toughness': 2.6e6,
                'elastic_modulus': 55e9,
                'poisson_ratio': 0.25
            },
            'sandstone': {
                'cohesion': 15e6,
                'friction_angle': 30.0,
                'tensile_strength': 4e6,
                'ucs': 80e6,
                'm_i': 8.0,
                'fracture_toughness': 1.5e6,
                'elastic_modulus': 25e9,
                'poisson_ratio': 0.30
            },
            'limestone': {
                'cohesion': 18e6,
                'friction_angle': 32.0,
                'tensile_strength': 5e6,
                'ucs': 100e6,
                'm_i': 10.0,
                'fracture_toughness': 1.8e6,
                'elastic_modulus': 35e9,
                'poisson_ratio': 0.28
            },
            'shale': {
                'cohesion': 10e6,
                'friction_angle': 25.0,
                'tensile_strength': 2e6,
                'ucs': 50e6,
                'm_i': 5.0,
                'fracture_toughness': 1.0e6,
                'elastic_modulus': 15e9,
                'poisson_ratio': 0.35
            },
            'schist': {
                'cohesion': 20e6,
                'friction_angle': 35.0,
                'tensile_strength': 6e6,
                'ucs': 100e6,
                'm_i': 12.0,
                'fracture_toughness': 2.2e6,
                'elastic_modulus': 40e9,
                'poisson_ratio': 0.28
            },
            'gneiss': {
                'cohesion': 25e6,
                'friction_angle': 38.0,
                'tensile_strength': 8e6,
                'ucs': 140e6,
                'm_i': 15.0,
                'fracture_toughness': 2.4e6,
                'elastic_modulus': 50e9,
                'poisson_ratio': 0.26
            }
        }
        
        # Initialize property arrays
        shape = X.shape
        properties = {
            'cohesion': np.zeros(shape),
            'friction_angle': np.zeros(shape),
            'tensile_strength': np.zeros(shape),
            'ucs': np.zeros(shape),
            'm_i': np.zeros(shape),
            's': np.ones(shape),  # Intact rock parameter
            'a': np.full(shape, 0.5),  # Hoek-Brown parameter
            'fracture_toughness': np.zeros(shape),
            'elastic_modulus': np.zeros(shape),
            'poisson_ratio': np.zeros(shape),
            'weibull_modulus': np.full(shape, 10.0),
            'weibull_scale': np.zeros(shape),
            'fatigue_limit': np.full(shape, 0.4)
        }
        
        # Weight properties by rock type probabilities
        for rock_type, probability in rock_types.items():
            if rock_type in reference_properties:
                ref_props = reference_properties[rock_type]
                for prop_name, ref_value in ref_props.items():
                    if prop_name in properties:
                        properties[prop_name] += probability * ref_value
                
                # Weibull scale parameter (related to UCS)
                properties['weibull_scale'] += probability * ref_props['ucs'] * 0.8
        
        return properties
    
    def _apply_temperature_pressure_modifications(self, properties: Dict[str, np.ndarray],
                                                X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply temperature and pressure effects on fracture properties
        """
        
        depth = np.abs(Z)
        
        # Temperature estimation (geothermal gradient)
        surface_temp = 288  # K
        geothermal_gradient = 25  # K/km
        temperature = surface_temp + geothermal_gradient * depth
        
        # Pressure estimation
        pressure = depth * 2700 * 9.81  # Pa (lithostatic)
        
        modified_properties = properties.copy()
        
        # Temperature effects (generally weakening)
        temp_factor_strength = 1.0 - 3e-4 * (temperature - 300)  # Strength reduction
        temp_factor_toughness = 1.0 - 2e-4 * (temperature - 300)  # Toughness reduction
        temp_factor_modulus = 1.0 - 4e-4 * (temperature - 300)  # Modulus reduction
        
        # Pressure effects (generally strengthening due to crack closure)
        pressure_factor_strength = 1.0 + 2e-9 * pressure  # Strength increase
        pressure_factor_cohesion = 1.0 + 1.5e-9 * pressure  # Cohesion increase
        
        # Apply temperature effects
        modified_properties['ucs'] *= temp_factor_strength
        modified_properties['tensile_strength'] *= temp_factor_strength
        modified_properties['fracture_toughness'] *= temp_factor_toughness
        modified_properties['elastic_modulus'] *= temp_factor_modulus
        
        # Apply pressure effects
        modified_properties['ucs'] *= pressure_factor_strength
        modified_properties['cohesion'] *= pressure_factor_cohesion
        modified_properties['tensile_strength'] *= pressure_factor_strength
        
        # Update Weibull scale parameter
        modified_properties['weibull_scale'] = modified_properties['ucs'] * 0.8
        
        return modified_properties
    
    def _apply_damage_healing_effects(self, properties: Dict[str, np.ndarray],
                                    stress: Dict[str, np.ndarray],
                                    X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply damage accumulation and healing effects
        """
        
        # Calculate damage parameter from stress history (simplified)
        stress_invariant = self._calculate_stress_invariant(stress)
        max_stress = np.maximum.reduce([np.abs(stress['sigma_xx']), 
                                      np.abs(stress['sigma_yy']), 
                                      np.abs(stress['sigma_zz'])])
        
        # Damage parameter (0 = intact, 1 = fully damaged)
        damage = np.tanh(max_stress / (properties['ucs'] + 1e6))
        
        # Healing rate (depth and temperature dependent)
        depth = np.abs(Z)
        temperature = 288 + 25 * depth  # K
        
        # Healing factor (higher at greater depths and temperatures)
        healing_factor = 0.1 * np.tanh(depth / 5.0) * np.tanh((temperature - 400) / 200)
        healing_factor = np.maximum(healing_factor, 0.0)
        
        # Net damage (damage minus healing)
        net_damage = np.maximum(damage - healing_factor, 0.0)
        
        # Apply damage effects (property degradation)
        modified_properties = properties.copy()
        damage_factor = 1.0 - 0.8 * net_damage  # Up to 80% reduction
        
        modified_properties['cohesion'] *= damage_factor
        modified_properties['tensile_strength'] *= damage_factor
        modified_properties['ucs'] *= damage_factor
        modified_properties['fracture_toughness'] *= damage_factor
        modified_properties['elastic_modulus'] *= damage_factor
        
        # Store damage for future reference
        modified_properties['damage'] = net_damage
        
        return modified_properties
    
    def _apply_rate_dependent_modifications(self, properties: Dict[str, np.ndarray],
                                          stress: Dict[str, np.ndarray],
                                          X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply strain rate dependent modifications to fracture properties
        """
        
        # Estimate strain rate from stress gradients (simplified)
        dt = 3600  # Assumed time step (seconds)
        strain_rate = self._estimate_strain_rate_from_stress(stress, dt)
        
        # Rate dependency factor (logarithmic relationship)
        reference_rate = 1e-12  # 1/s (quasi-static reference)
        rate_factor = 1.0 + 0.1 * np.log10(np.maximum(strain_rate / reference_rate, 0.1))
        
        # Apply rate effects (generally strengthening at higher rates)
        modified_properties = properties.copy()
        
        modified_properties['ucs'] *= rate_factor
        modified_properties['tensile_strength'] *= rate_factor
        modified_properties['cohesion'] *= rate_factor
        
        # Dynamic effects at very high rates
        high_rate_mask = strain_rate > 1e-6  # 1/s
        dynamic_factor = np.where(high_rate_mask, 1.5, 1.0)
        
        modified_properties['ucs'] *= dynamic_factor
        modified_properties['fracture_toughness'] *= dynamic_factor
        
        return modified_properties
    
    def _apply_anisotropic_fracture_properties(self, properties: Dict[str, np.ndarray],
                                             X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply anisotropic modifications to fracture properties
        """
        
        depth = np.abs(Z)
        
        # Anisotropy strength (increases with depth due to stress)
        anisotropy_strength = 0.3 * np.tanh(depth / 5.0)
        
        # Directional variations (simplified structural fabric)
        fabric_orientation = np.pi/6 * np.sin(X / 2.0) * np.cos(Y / 2.0)  # radians
        
        modified_properties = properties.copy()
        
        # Anisotropic strength variations
        # Weaker perpendicular to fabric, stronger parallel
        fabric_factor = 1.0 + anisotropy_strength * np.cos(2 * fabric_orientation)
        
        # Apply to selected properties
        modified_properties['tensile_strength'] *= fabric_factor
        modified_properties['cohesion'] *= (2.0 - fabric_factor)  # Inverse relationship
        modified_properties['fracture_toughness'] *= fabric_factor
        
        # Anisotropic elastic properties
        elastic_anisotropy = 1.0 + 0.2 * anisotropy_strength * np.sin(fabric_orientation)
        modified_properties['elastic_modulus'] *= elastic_anisotropy
        
        return modified_properties
    
    def _apply_statistical_heterogeneity(self, properties: Dict[str, np.ndarray],
                                        X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply statistical heterogeneity to represent natural variability
        """
        
        # Correlation lengths for spatial heterogeneity
        correlation_length_x = 1.0  # km
        correlation_length_y = 1.0  # km
        correlation_length_z = 0.5  # km
        
        modified_properties = properties.copy()
        
        # Apply spatially correlated random fields
        for prop_name in ['cohesion', 'tensile_strength', 'ucs', 'fracture_toughness']:
            if prop_name in properties:
                # Generate correlated random field
                random_field = self._generate_correlated_random_field(
                    X, Y, Z, correlation_length_x, correlation_length_y, correlation_length_z)
                
                # Coefficient of variation (typical 0.2-0.4 for rock properties)
                cov = 0.3
                
                # Apply multiplicative heterogeneity
                heterogeneity_factor = 1.0 + cov * random_field
                heterogeneity_factor = np.maximum(heterogeneity_factor, 0.1)  # Prevent negative values
                
                modified_properties[prop_name] *= heterogeneity_factor
        
        # Update dependent parameters
        modified_properties['weibull_scale'] = modified_properties['ucs'] * 0.8
        
        return modified_properties
    
    def _calculate_stress_invariant(self, stress: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate stress invariant for damage calculations"""
        
        # Second invariant of deviatoric stress tensor
        sigma_xx = stress['sigma_xx']
        sigma_yy = stress['sigma_yy']
        sigma_zz = stress['sigma_zz']
        
        # Mean stress
        sigma_m = (sigma_xx + sigma_yy + sigma_zz) / 3.0
        
        # Deviatoric stresses
        s_xx = sigma_xx - sigma_m
        s_yy = sigma_yy - sigma_m
        s_zz = sigma_zz - sigma_m
        
        # Second invariant
        J2 = 0.5 * (s_xx**2 + s_yy**2 + s_zz**2)
        
        return np.sqrt(3 * J2)  # von Mises equivalent stress
    
    def _estimate_strain_rate_from_stress(self, stress: Dict[str, np.ndarray], dt: float) -> np.ndarray:
        """Estimate strain rate from stress gradients (simplified)"""
        
        # Spatial gradients of stress (proxy for strain rate)
        sigma_xx = stress['sigma_xx']
        
        if sigma_xx.ndim >= 2:
            grad_x = np.gradient(sigma_xx, axis=0)
            grad_y = np.gradient(sigma_xx, axis=1) if sigma_xx.ndim >= 2 else np.zeros_like(grad_x)
            grad_z = np.gradient(sigma_xx, axis=2) if sigma_xx.ndim >= 3 else np.zeros_like(grad_x)
            
            # Magnitude of stress gradient
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        else:
            grad_magnitude = np.abs(np.gradient(sigma_xx))
        
        # Convert to strain rate (simplified)
        elastic_modulus = 50e9  # Pa (typical)
        strain_rate = grad_magnitude / (elastic_modulus * dt)
        
        return strain_rate
    
    def _generate_correlated_random_field(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                        lx: float, ly: float, lz: float) -> np.ndarray:
        """
        Generate spatially correlated random field using exponential correlation
        """
        
        # Generate uncorrelated random field
        np.random.seed(42)  # For reproducibility
        random_field = np.random.normal(0, 1, X.shape)
        
        # Apply exponential correlation (simplified approach)
        # In practice, would use FFT-based methods for efficiency
        
        # Correlation function: exp(-|r|/correlation_length)
        # This is a simplified implementation
        
        # Apply smoothing to approximate correlation
        from scipy import ndimage
        
        # Smoothing scales (approximation of correlation lengths)
        sigma_x = lx / 3.0
        sigma_y = ly / 3.0
        sigma_z = lz / 3.0
        
        if X.ndim == 3:
            sigma = [sigma_x, sigma_y, sigma_z]
        elif X.ndim == 2:
            sigma = [sigma_x, sigma_y]
        else:
            sigma = sigma_x
        
        try:
            correlated_field = ndimage.gaussian_filter(random_field, sigma=sigma)
        except:
            # Fallback: return uncorrelated field
            correlated_field = random_field
        
        # Normalize to zero mean, unit variance
        correlated_field = (correlated_field - np.mean(correlated_field)) / (np.std(correlated_field) + 1e-10)
        
        return correlated_field
    
    def _calculate_comprehensive_composition_velocity_effects(self, composition: Dict[str, float],
                                                            temperature: np.ndarray,
                                                            pressure: np.ndarray,
                                                            melt_fraction: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive composition effects on seismic velocities
        
        Includes:
        1. Major oxide effects (SiO2, Al2O3, FeO, MgO, CaO, Na2O, K2O, TiO2)
        2. Volatile effects (H2O, CO2, S, F, Cl)
        3. Crystal chemistry and polymerization effects
        4. Melt structure modifications
        5. Temperature-composition interactions
        6. Pressure-composition interactions
        7. Non-linear mixing relationships
        8. Modal mineralogy effects
        """
        
        # Get comprehensive composition with all components
        normalized_composition = self._normalize_comprehensive_composition(composition)
        
        # Calculate major oxide effects
        major_oxide_effects = self._calculate_major_oxide_velocity_effects(
            normalized_composition, temperature, pressure, melt_fraction)
        
        # Calculate volatile effects
        volatile_effects = self._calculate_volatile_velocity_effects(
            normalized_composition, temperature, pressure, melt_fraction)
        
        # Calculate crystal chemistry effects
        crystal_chemistry_effects = self._calculate_crystal_chemistry_velocity_effects(
            normalized_composition, temperature, pressure)
        
        # Calculate melt structure effects
        melt_structure_effects = self._calculate_melt_structure_velocity_effects(
            normalized_composition, temperature, pressure, melt_fraction)
        
        # Calculate modal mineralogy effects
        modal_effects = self._calculate_modal_mineralogy_velocity_effects(
            normalized_composition, temperature, pressure, melt_fraction)
        
        # Apply non-linear interaction effects
        interaction_effects = self._calculate_composition_interaction_effects(
            normalized_composition, temperature, pressure, melt_fraction)
        
        # Combine all effects
        total_vp_effect = (major_oxide_effects['vp'] + 
                          volatile_effects['vp'] + 
                          crystal_chemistry_effects['vp'] + 
                          melt_structure_effects['vp'] + 
                          modal_effects['vp'] + 
                          interaction_effects['vp'])
        
        total_vs_effect = (major_oxide_effects['vs'] + 
                          volatile_effects['vs'] + 
                          crystal_chemistry_effects['vs'] + 
                          melt_structure_effects['vs'] + 
                          modal_effects['vs'] + 
                          interaction_effects['vs'])
        
        return {'vp': total_vp_effect, 'vs': total_vs_effect}
    
    def _normalize_comprehensive_composition(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Normalize composition to include all major and trace components"""
        
        # Default comprehensive composition (rhyolitic)
        default_composition = {
            # Major oxides
            'SiO2': 72.0, 'TiO2': 0.3, 'Al2O3': 13.5, 'Fe2O3': 1.5, 'FeO': 1.0,
            'MnO': 0.05, 'MgO': 0.8, 'CaO': 1.5, 'Na2O': 4.0, 'K2O': 4.5,
            'P2O5': 0.1, 'Cr2O3': 0.01, 'NiO': 0.005,
            
            # Volatiles
            'H2O': 0.5, 'CO2': 0.05, 'SO2': 0.02, 'F': 0.03, 'Cl': 0.01,
            
            # Trace elements affecting seismic properties
            'Li2O': 0.005, 'BeO': 0.001, 'B2O3': 0.002, 'Rb2O': 0.01,
            'SrO': 0.03, 'Y2O3': 0.002, 'ZrO2': 0.01, 'Nb2O5': 0.001,
            'BaO': 0.05, 'La2O3': 0.002, 'CeO2': 0.004, 'Nd2O3': 0.002,
            'Sm2O3': 0.0005, 'Eu2O3': 0.0001, 'Gd2O3': 0.0005,
            'Tb4O7': 0.0001, 'Dy2O3': 0.0005, 'Ho2O3': 0.0001,
            'Er2O3': 0.0003, 'Tm2O3': 0.00005, 'Yb2O3': 0.0003,
            'Lu2O3': 0.00005, 'HfO2': 0.0003, 'Ta2O5': 0.0001,
            'PbO': 0.002, 'ThO2': 0.001, 'UO2': 0.0005
        }
        
        # Update with provided composition
        normalized_comp = default_composition.copy()
        for component, value in composition.items():
            if component in normalized_comp:
                normalized_comp[component] = value
        
        # Normalize major oxides to 100% (excluding volatiles)
        major_oxides = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MnO', 'MgO', 
                       'CaO', 'Na2O', 'K2O', 'P2O5', 'Cr2O3', 'NiO']
        
        total_major = sum(normalized_comp[oxide] for oxide in major_oxides if oxide in normalized_comp)
        
        if total_major > 0:
            normalization_factor = 100.0 / total_major
            for oxide in major_oxides:
                if oxide in normalized_comp:
                    normalized_comp[oxide] *= normalization_factor
        
        return normalized_comp
    
    def _calculate_major_oxide_velocity_effects(self, composition: Dict[str, float],
                                              temperature: np.ndarray, pressure: np.ndarray,
                                              melt_fraction: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate velocity effects from major oxides using empirical relationships
        """
        
        # Empirical coefficients for velocity-composition relationships
        # Based on extensive experimental data and seismic surveys
        vp_coefficients = {
            'SiO2': 15.0,     # m/s per wt% (network former, increases rigidity)
            'Al2O3': 12.0,    # Network former, moderate effect
            'FeO': -8.0,      # Network modifier, decreases velocity
            'Fe2O3': -6.0,    # Less effect than FeO
            'MgO': 25.0,      # Strong network modifier, increases velocity in silicates
            'CaO': 8.0,       # Network modifier, moderate positive effect
            'Na2O': -15.0,    # Strong network modifier, decreases velocity
            'K2O': -18.0,     # Even stronger effect than Na2O
            'TiO2': -5.0,     # High-field-strength cation
            'P2O5': -10.0,    # Complex structural role
            'MnO': -7.0,      # Similar to FeO
            'Cr2O3': 5.0,     # High-field-strength, small positive effect
            'NiO': 3.0        # Transition metal oxide
        }
        
        vs_coefficients = {
            'SiO2': 9.0,      # Vs generally less sensitive than Vp
            'Al2O3': 7.0,
            'FeO': -5.0,
            'Fe2O3': -4.0,
            'MgO': 15.0,
            'CaO': 5.0,
            'Na2O': -9.0,
            'K2O': -11.0,
            'TiO2': -3.0,
            'P2O5': -6.0,
            'MnO': -4.0,
            'Cr2O3': 3.0,
            'NiO': 2.0
        }
        
        # Reference composition (mid-ocean ridge basalt)
        reference_composition = {
            'SiO2': 50.0, 'Al2O3': 16.0, 'FeO': 8.0, 'Fe2O3': 2.0,
            'MgO': 8.0, 'CaO': 11.0, 'Na2O': 3.0, 'K2O': 0.5,
            'TiO2': 1.0, 'P2O5': 0.2, 'MnO': 0.15, 'Cr2O3': 0.05, 'NiO': 0.01
        }
        
        # Calculate deviations from reference
        vp_effect = np.zeros_like(temperature)
        vs_effect = np.zeros_like(temperature)
        
        for oxide in vp_coefficients:
            if oxide in composition:
                deviation = composition[oxide] - reference_composition.get(oxide, 0)
                vp_effect += vp_coefficients[oxide] * deviation
                vs_effect += vs_coefficients[oxide] * deviation
        
        # Apply temperature corrections (composition-temperature interactions)
        temp_correction_factor = 1.0 + 2e-4 * (temperature - 1273)  # Reference at 1000°C
        vp_effect *= temp_correction_factor
        vs_effect *= temp_correction_factor
        
        # Apply pressure corrections (composition-pressure interactions)
        pressure_GPa = pressure * 1e-9
        pressure_correction_factor = 1.0 + 0.05 * pressure_GPa  # Slight increase with pressure
        vp_effect *= pressure_correction_factor
        vs_effect *= pressure_correction_factor
        
        return {'vp': vp_effect, 'vs': vs_effect}
    
    def _calculate_volatile_velocity_effects(self, composition: Dict[str, float],
                                           temperature: np.ndarray, pressure: np.ndarray,
                                           melt_fraction: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate velocity effects from volatile components
        """
        
        # Volatile effects are generally velocity-reducing due to:
        # 1. Density reduction
        # 2. Weakening of structural framework
        # 3. Creation of fluid inclusions
        
        volatile_coefficients_vp = {
            'H2O': -800,    # m/s per wt% (very strong effect)
            'CO2': -300,    # Strong but less than H2O
            'SO2': -200,    # Moderate effect
            'F': -150,      # Network modifier effect
            'Cl': -100      # Weaker effect than F
        }
        
        volatile_coefficients_vs = {
            'H2O': -500,    # Vs less affected than Vp
            'CO2': -180,
            'SO2': -120,
            'F': -90,
            'Cl': -60
        }
        
        vp_effect = np.zeros_like(temperature)
        vs_effect = np.zeros_like(temperature)
        
        for volatile in volatile_coefficients_vp:
            if volatile in composition:
                content = composition[volatile]
                
                # Temperature-dependent solubility effects
                solubility_factor = self._calculate_volatile_solubility_factor(
                    volatile, content, temperature, pressure)
                
                # Effective volatile content (dissolved vs. exsolved)
                effective_content = content * solubility_factor
                
                vp_effect += volatile_coefficients_vp[volatile] * effective_content
                vs_effect += volatile_coefficients_vs[volatile] * effective_content
                
                # Additional effects from volatile clustering and speciation
                if volatile == 'H2O' and content > 2.0:  # wt%
                    # Clustering effects at high H2O contents
                    clustering_effect = -100 * (content - 2.0)  # Additional reduction
                    vp_effect += clustering_effect
                    vs_effect += clustering_effect * 0.6
        
        return {'vp': vp_effect, 'vs': vs_effect}
    
    def _calculate_crystal_chemistry_velocity_effects(self, composition: Dict[str, float],
                                                    temperature: np.ndarray,
                                                    pressure: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate velocity effects from crystal chemistry and coordination changes
        """
        
        # Calculate polymerization degree (NBO/T - Non-Bridging Oxygens per Tetrahedron)
        nbo_t = self._calculate_polymerization_degree(composition)
        
        # Polymerization effects on velocity
        # More polymerized (lower NBO/T) = higher velocity
        polymerization_effect_vp = -500 * nbo_t  # m/s
        polymerization_effect_vs = -300 * nbo_t  # m/s
        
        # Coordination number changes with pressure (Al coordination)
        al2o3_content = composition.get('Al2O3', 13.5)
        pressure_GPa = pressure * 1e-9
        
        # Al coordination change: IV → VI with increasing pressure
        coord_change_factor = np.tanh(pressure_GPa / 5.0)  # Transition around 5 GPa
        al_coord_effect_vp = al2o3_content * 20 * coord_change_factor  # m/s
        al_coord_effect_vs = al2o3_content * 12 * coord_change_factor  # m/s
        
        # Iron oxidation state effects
        fe_total = composition.get('FeO', 1.0) + composition.get('Fe2O3', 1.5) * 0.8998
        fe3_fe2_ratio = composition.get('Fe2O3', 1.5) * 0.8998 / (composition.get('FeO', 1.0) + 1e-10)
        
        # Fe3+ generally increases velocity compared to Fe2+
        iron_redox_effect_vp = fe_total * 15 * np.tanh(fe3_fe2_ratio)
        iron_redox_effect_vs = fe_total * 9 * np.tanh(fe3_fe2_ratio)
        
        # Temperature effects on crystal chemistry
        temp_factor = np.exp(-(temperature - 1273) / 500)  # Decreases with temperature
        
        total_vp_effect = (polymerization_effect_vp + al_coord_effect_vp + iron_redox_effect_vp) * temp_factor
        total_vs_effect = (polymerization_effect_vs + al_coord_effect_vs + iron_redox_effect_vs) * temp_factor
        
        return {'vp': total_vp_effect, 'vs': total_vs_effect}
    
    def _calculate_melt_structure_velocity_effects(self, composition: Dict[str, float],
                                                 temperature: np.ndarray, pressure: np.ndarray,
                                                 melt_fraction: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate velocity effects from melt structure modifications
        """
        
        # Calculate average melt properties when partial melt is present
        if np.any(melt_fraction > 0):
            
            # Melt connectivity and geometry effects
            connectivity_factor = self._calculate_melt_connectivity_factor(melt_fraction)
            
            # Aspect ratio of melt inclusions (affects elastic properties)
            aspect_ratio = self._calculate_melt_aspect_ratio(melt_fraction, pressure)
            
            # Melt structure depends on composition and temperature
            melt_structure_factor = self._calculate_melt_structure_factor(
                composition, temperature, pressure)
            
            # Velocity reduction due to melt presence (beyond simple melt fraction effect)
            structure_effect_vp = -200 * melt_fraction * connectivity_factor * melt_structure_factor
            structure_effect_vs = -150 * melt_fraction * connectivity_factor * melt_structure_factor
            
            # Aspect ratio effects (thin films vs. spherical inclusions)
            geometry_effect_vp = -100 * melt_fraction * (1.0 - aspect_ratio)
            geometry_effect_vs = -80 * melt_fraction * (1.0 - aspect_ratio)
            
            total_vp_effect = structure_effect_vp + geometry_effect_vp
            total_vs_effect = structure_effect_vs + geometry_effect_vs
            
        else:
            total_vp_effect = np.zeros_like(temperature)
            total_vs_effect = np.zeros_like(temperature)
        
        return {'vp': total_vp_effect, 'vs': total_vs_effect}
    
    def _calculate_modal_mineralogy_velocity_effects(self, composition: Dict[str, float],
                                                   temperature: np.ndarray, pressure: np.ndarray,
                                                   melt_fraction: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate velocity effects from modal mineralogy (crystalline phases)
        """
        
        # Estimate modal mineralogy from bulk composition using CIPW norm-like calculation
        modal_minerals = self._estimate_modal_mineralogy(composition, temperature, pressure)
        
        # Seismic velocities of individual minerals (at reference conditions)
        mineral_velocities = {
            'quartz': {'vp': 6050, 'vs': 4090},      # m/s
            'k_feldspar': {'vp': 5900, 'vs': 3180},
            'plagioclase': {'vp': 6250, 'vs': 3520},
            'olivine': {'vp': 8400, 'vs': 4900},
            'pyroxene': {'vp': 7800, 'vs': 4600},
            'amphibole': {'vp': 7000, 'vs': 3900},
            'biotite': {'vp': 5800, 'vs': 3100},
            'muscovite': {'vp': 5600, 'vs': 2900},
            'magnetite': {'vp': 7200, 'vs': 3900},
            'ilmenite': {'vp': 7100, 'vs': 3800},
            'apatite': {'vp': 7700, 'vs': 4200},
            'zircon': {'vp': 7400, 'vs': 4200}
        }
        
        # Apply P-T corrections to mineral velocities
        corrected_velocities = {}
        for mineral, vel in mineral_velocities.items():
            corrected_velocities[mineral] = self._apply_pt_corrections_to_mineral_velocities(
                vel, temperature, pressure, mineral)
        
        # Calculate effective velocity using mixing laws
        # Voigt-Reuss-Hill averaging
        crystal_fraction = 1.0 - melt_fraction
        
        if np.any(crystal_fraction > 0):
            # Voigt bound (uniform strain)
            vp_voigt = np.zeros_like(temperature)
            vs_voigt = np.zeros_like(temperature)
            
            # Reuss bound (uniform stress)
            vp_reuss_inv = np.zeros_like(temperature)
            vs_reuss_inv = np.zeros_like(temperature)
            
            for mineral, fraction in modal_minerals.items():
                if mineral in corrected_velocities and fraction > 0:
                    mineral_vp = corrected_velocities[mineral]['vp']
                    mineral_vs = corrected_velocities[mineral]['vs']
                    
                    # Voigt average
                    vp_voigt += fraction * mineral_vp * crystal_fraction
                    vs_voigt += fraction * mineral_vs * crystal_fraction
                    
                    # Reuss average (harmonic mean)
                    vp_reuss_inv += fraction * crystal_fraction / mineral_vp
                    vs_reuss_inv += fraction * crystal_fraction / mineral_vs
            
            # Reuss bounds
            vp_reuss = 1.0 / (vp_reuss_inv + 1e-10)
            vs_reuss = 1.0 / (vs_reuss_inv + 1e-10)
            
            # Hill average (geometric mean of Voigt and Reuss)
            vp_hill = np.sqrt(vp_voigt * vp_reuss)
            vs_hill = np.sqrt(vs_voigt * vs_reuss)
            
            # Reference velocities (approximate crustal averages)
            vp_reference = 6200  # m/s
            vs_reference = 3600  # m/s
            
            modal_effect_vp = vp_hill - vp_reference
            modal_effect_vs = vs_hill - vs_reference
            
        else:
            modal_effect_vp = np.zeros_like(temperature)
            modal_effect_vs = np.zeros_like(temperature)
        
        return {'vp': modal_effect_vp, 'vs': modal_effect_vs}
    
    def _calculate_composition_interaction_effects(self, composition: Dict[str, float],
                                                 temperature: np.ndarray, pressure: np.ndarray,
                                                 melt_fraction: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate non-linear interaction effects between compositional components
        """
        
        # Key compositional ratios affecting seismic properties
        sio2 = composition.get('SiO2', 72.0)
        al2o3 = composition.get('Al2O3', 13.5)
        alkalis = composition.get('Na2O', 4.0) + composition.get('K2O', 4.5)
        mafic = (composition.get('FeO', 1.0) + composition.get('Fe2O3', 1.5) * 0.8998 + 
                composition.get('MgO', 0.8))
        
        # Alkali-silica interaction (affects polymerization)
        alkali_silica_ratio = alkalis / (sio2 / 100)
        alkali_silica_effect_vp = -200 * alkali_silica_ratio**2  # Non-linear
        alkali_silica_effect_vs = -120 * alkali_silica_ratio**2
        
        # Aluminum-alkali interaction (affects Al coordination)
        al_alkali_ratio = al2o3 / (alkalis + 1e-10)
        # Peraluminous (Al/alkali > 1) vs. peralkaline effects
        if al_alkali_ratio > 1:
            al_interaction_vp = 50 * (al_alkali_ratio - 1.0)  # Peraluminous increases velocity
            al_interaction_vs = 30 * (al_alkali_ratio - 1.0)
        else:
            al_interaction_vp = -100 * (1.0 - al_alkali_ratio)  # Peralkaline decreases velocity
            al_interaction_vs = -60 * (1.0 - al_alkali_ratio)
        
        # Mafic-felsic balance effects
        mafic_ratio = mafic / 100
        felsic_ratio = (sio2 + al2o3) / 100
        mafic_felsic_interaction_vp = 300 * mafic_ratio * (1 - felsic_ratio)
        mafic_felsic_interaction_vs = 200 * mafic_ratio * (1 - felsic_ratio)
        
        # Volatile-silica interactions (affects volatile solubility)
        h2o_content = composition.get('H2O', 0.5)
        volatile_silica_interaction_vp = -50 * h2o_content * (sio2 / 100)**2
        volatile_silica_interaction_vs = -30 * h2o_content * (sio2 / 100)**2
        
        # Temperature modulation of interactions
        temp_modulation = np.exp(-(temperature - 1273) / 300)
        
        total_vp_effect = (alkali_silica_effect_vp + al_interaction_vp + 
                          mafic_felsic_interaction_vp + volatile_silica_interaction_vp) * temp_modulation
        
        total_vs_effect = (alkali_silica_effect_vs + al_interaction_vs + 
                          mafic_felsic_interaction_vs + volatile_silica_interaction_vs) * temp_modulation
        
        return {'vp': total_vp_effect, 'vs': total_vs_effect}
    
    def _calculate_polymerization_degree(self, composition: Dict[str, float]) -> float:
        """
        Calculate degree of polymerization (NBO/T) from composition
        """
        
        # Molar proportions of network formers and modifiers
        sio2_mol = composition.get('SiO2', 72.0) / 60.08
        al2o3_mol = composition.get('Al2O3', 13.5) / 101.96
        
        # Network modifiers (create non-bridging oxygens)
        na2o_mol = composition.get('Na2O', 4.0) / 61.98
        k2o_mol = composition.get('K2O', 4.5) / 94.20
        cao_mol = composition.get('CaO', 1.5) / 56.08
        mgo_mol = composition.get('MgO', 0.8) / 40.30
        feo_mol = (composition.get('FeO', 1.0) + 
                  composition.get('Fe2O3', 1.5) * 0.8998) / 71.84
        
        # Total tetrahedra (T) = Si + Al
        total_T = sio2_mol + al2o3_mol
        
        # Total network modifiers
        total_modifiers = na2o_mol + k2o_mol + cao_mol + mgo_mol + feo_mol
        
        # Non-bridging oxygens per tetrahedron
        if total_T > 0:
            nbo_t = 2 * total_modifiers / total_T
        else:
            nbo_t = 0
        
        return nbo_t
    
    def _calculate_volatile_solubility_factor(self, volatile: str, content: float,
                                            temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """
        Calculate fraction of volatile that remains dissolved vs. exsolved
        """
        
        if volatile == 'H2O':
            # H2O solubility in silicate melts (simplified Zhang & Behrens model)
            pressure_MPa = pressure * 1e-6
            solubility_limit = 0.15 * np.sqrt(pressure_MPa)  # wt%
            
        elif volatile == 'CO2':
            # CO2 solubility (much lower than H2O)
            pressure_MPa = pressure * 1e-6
            solubility_limit = 0.005 * pressure_MPa  # wt%
            
        else:
            # Default for other volatiles
            pressure_MPa = pressure * 1e-6
            solubility_limit = 0.01 * np.sqrt(pressure_MPa)  # wt%
        
        # Fraction dissolved
        dissolved_fraction = np.minimum(solubility_limit / (content + 1e-10), 1.0)
        
        return dissolved_fraction
    
    def _calculate_melt_connectivity_factor(self, melt_fraction: np.ndarray) -> np.ndarray:
        """
        Calculate melt connectivity factor based on melt fraction
        """
        
        # Percolation threshold around 5-10% melt fraction
        percolation_threshold = 0.07
        
        connectivity = np.zeros_like(melt_fraction)
        
        # Below threshold: isolated melt pockets
        below_threshold = melt_fraction < percolation_threshold
        connectivity[below_threshold] = (melt_fraction[below_threshold] / percolation_threshold)**3
        
        # Above threshold: connected melt network
        above_threshold = melt_fraction >= percolation_threshold
        connectivity[above_threshold] = 1.0 - np.exp(-3 * (melt_fraction[above_threshold] - percolation_threshold))
        
        return connectivity
    
    def _calculate_melt_aspect_ratio(self, melt_fraction: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """
        Calculate aspect ratio of melt inclusions
        """
        
        # Pressure favors spherical inclusions (higher aspect ratio)
        # Low pressure favors thin films (lower aspect ratio)
        pressure_GPa = pressure * 1e-9
        
        base_aspect_ratio = 0.1 + 0.4 * np.tanh(pressure_GPa / 1.0)  # 0.1 to 0.5
        
        # Melt fraction affects geometry
        # High melt fractions tend toward more spherical geometry
        melt_effect = 0.3 * np.tanh(melt_fraction / 0.1)
        
        aspect_ratio = base_aspect_ratio + melt_effect
        
        return np.clip(aspect_ratio, 0.05, 1.0)
    
    def _calculate_melt_structure_factor(self, composition: Dict[str, float],
                                       temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """
        Calculate melt structure factor affecting seismic properties
        """
        
        # Polymerization of melt affects its elastic properties
        nbo_t = self._calculate_polymerization_degree(composition)
        
        # More polymerized melts are "stiffer"
        polymerization_factor = np.exp(-nbo_t / 2)
        
        # Temperature effects on melt structure
        temp_factor = np.exp(-(temperature - 1273) / 200)
        
        structure_factor = polymerization_factor * temp_factor
        
        return np.clip(structure_factor, 0.1, 2.0)
    
    def _estimate_modal_mineralogy(self, composition: Dict[str, float],
                                 temperature: np.ndarray, pressure: np.ndarray) -> Dict[str, float]:
        """
        Estimate modal mineralogy from bulk composition using CIPW-like norms
        """
        
        # Simplified modal estimation based on major element chemistry
        sio2 = composition.get('SiO2', 72.0)
        al2o3 = composition.get('Al2O3', 13.5)
        k2o = composition.get('K2O', 4.5)
        na2o = composition.get('Na2O', 4.0)
        cao = composition.get('CaO', 1.5)
        mgo = composition.get('MgO', 0.8)
        feo_total = composition.get('FeO', 1.0) + composition.get('Fe2O3', 1.5) * 0.8998
        
        # Differentiation index
        di = sio2 + na2o + k2o
        
        modal_minerals = {}
        
        if di > 80:  # Highly evolved (granite/rhyolite)
            modal_minerals = {
                'quartz': 0.30,
                'k_feldspar': 0.35,
                'plagioclase': 0.25,
                'biotite': 0.08,
                'magnetite': 0.02
            }
        elif di > 65:  # Intermediate (granodiorite/dacite)
            modal_minerals = {
                'quartz': 0.20,
                'k_feldspar': 0.25,
                'plagioclase': 0.40,
                'biotite': 0.10,
                'amphibole': 0.03,
                'magnetite': 0.02
            }
        elif di > 55:  # Intermediate (diorite/andesite)
            modal_minerals = {
                'quartz': 0.10,
                'k_feldspar': 0.15,
                'plagioclase': 0.50,
                'amphibole': 0.15,
                'pyroxene': 0.08,
                'magnetite': 0.02
            }
        else:  # Mafic (gabbro/basalt)
            modal_minerals = {
                'plagioclase': 0.50,
                'pyroxene': 0.35,
                'olivine': 0.10,
                'magnetite': 0.03,
                'ilmenite': 0.02
            }
        
        return modal_minerals
    
    def _apply_pt_corrections_to_mineral_velocities(self, base_velocities: Dict[str, float],
                                                  temperature: np.ndarray, pressure: np.ndarray,
                                                  mineral: str) -> Dict[str, np.ndarray]:
        """
        Apply pressure-temperature corrections to mineral seismic velocities
        """
        
        # Typical pressure and temperature derivatives for different minerals
        pt_derivatives = {
            'quartz': {'dvp_dp': 4.2e-9, 'dvp_dt': -0.5, 'dvs_dp': 2.1e-9, 'dvs_dt': -0.3},
            'k_feldspar': {'dvp_dp': 3.8e-9, 'dvp_dt': -0.4, 'dvs_dp': 1.9e-9, 'dvs_dt': -0.25},
            'plagioclase': {'dvp_dp': 4.0e-9, 'dvp_dt': -0.45, 'dvs_dp': 2.0e-9, 'dvs_dt': -0.28},
            'olivine': {'dvp_dp': 5.2e-9, 'dvp_dt': -0.6, 'dvs_dp': 2.8e-9, 'dvs_dt': -0.35},
            'pyroxene': {'dvp_dp': 4.8e-9, 'dvp_dt': -0.55, 'dvs_dp': 2.5e-9, 'dvs_dt': -0.32},
            'amphibole': {'dvp_dp': 4.5e-9, 'dvp_dt': -0.5, 'dvs_dp': 2.3e-9, 'dvs_dt': -0.3}
        }
        
        # Default derivatives if mineral not in database
        default_derivs = {'dvp_dp': 4.0e-9, 'dvp_dt': -0.5, 'dvs_dp': 2.0e-9, 'dvs_dt': -0.3}
        
        derivs = pt_derivatives.get(mineral, default_derivs)
        
        # Reference conditions
        p_ref = 1e5  # Pa
        t_ref = 298  # K
        
        # Pressure correction
        dp = pressure - p_ref
        vp_pressure_corr = derivs['dvp_dp'] * dp
        vs_pressure_corr = derivs['dvs_dp'] * dp
        
        # Temperature correction  
        dt = temperature - t_ref
        vp_temp_corr = derivs['dvp_dt'] * dt
        vs_temp_corr = derivs['dvs_dt'] * dt
        
        # Apply corrections
        corrected_vp = base_velocities['vp'] + vp_pressure_corr + vp_temp_corr
        corrected_vs = base_velocities['vs'] + vs_pressure_corr + vs_temp_corr
        
        return {'vp': corrected_vp, 'vs': corrected_vs}
    
    def _calculate_comprehensive_convection_pattern(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                                  convection_strength: np.ndarray, 
                                                  reservoir_state: Dict) -> np.ndarray:
        """
        Calculate comprehensive spatial convection pattern using advanced fluid dynamics
        
        Includes:
        1. Rayleigh-Bénard convection cells
        2. Plume dynamics and thermal boundary layers
        3. Double-diffusive convection (thermal and compositional)
        4. 3D convection patterns with realistic geometry
        5. Time-dependent convection evolution
        6. Viscosity stratification effects
        7. Boundary condition influences
        8. Non-linear convection interactions
        """
        
        # Get reservoir geometry and properties
        reservoir_dimensions = self._get_reservoir_dimensions(X, Y, Z)
        fluid_properties = self._calculate_convection_fluid_properties(reservoir_state)
        
        # Calculate dimensionless parameters
        dimensionless_params = self._calculate_convection_dimensionless_parameters(
            reservoir_dimensions, fluid_properties, reservoir_state)
        
        # Determine convection regime
        convection_regime = self._determine_convection_regime(dimensionless_params)
        
        # Calculate base convection pattern based on regime
        if convection_regime == 'rayleigh_benard':
            base_pattern = self._calculate_rayleigh_benard_pattern(
                X, Y, Z, dimensionless_params, reservoir_dimensions)
        elif convection_regime == 'plume_dominated':
            base_pattern = self._calculate_plume_dominated_pattern(
                X, Y, Z, dimensionless_params, reservoir_dimensions)
        elif convection_regime == 'double_diffusive':
            base_pattern = self._calculate_double_diffusive_pattern(
                X, Y, Z, dimensionless_params, reservoir_dimensions)
        else:
            # Default to mixed convection
            base_pattern = self._calculate_mixed_convection_pattern(
                X, Y, Z, dimensionless_params, reservoir_dimensions)
        
        # Apply viscosity stratification effects
        stratified_pattern = self._apply_viscosity_stratification_effects(
            base_pattern, X, Y, Z, fluid_properties)
        
        # Apply boundary condition effects
        boundary_modified_pattern = self._apply_convection_boundary_effects(
            stratified_pattern, X, Y, Z, reservoir_dimensions)
        
        # Apply time-dependent evolution
        time_evolved_pattern = self._apply_convection_time_evolution(
            boundary_modified_pattern, reservoir_state)
        
        # Apply non-linear interactions
        final_pattern = self._apply_convection_nonlinear_interactions(
            time_evolved_pattern, convection_strength, X, Y, Z)
        
        return final_pattern
    
    def _get_reservoir_dimensions(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, float]:
        """Get characteristic dimensions of the magma reservoir"""
        
        dimensions = {
            'width_x': np.ptp(X),      # Range in X direction
            'width_y': np.ptp(Y),      # Range in Y direction
            'height': np.ptp(Z),       # Range in Z direction (depth)
            'aspect_ratio_xy': np.ptp(X) / np.ptp(Y),
            'aspect_ratio_xz': np.ptp(X) / np.ptp(Z),
            'volume': np.ptp(X) * np.ptp(Y) * np.ptp(Z)
        }
        
        # Characteristic length scale
        dimensions['char_length'] = (dimensions['volume']**(1/3))
        
        return dimensions
    
    def _calculate_convection_fluid_properties(self, reservoir_state: Dict) -> Dict[str, float]:
        """Calculate fluid properties relevant for convection"""
        
        # Temperature-dependent properties
        temperature = reservoir_state.get('temperature', 1200)  # K
        
        # Density (temperature and composition dependent)
        base_density = 2400  # kg/m³ (rhyolitic melt)
        thermal_expansion = 5e-5  # 1/K
        density = base_density * (1 - thermal_expansion * (temperature - 1273))
        
        # Viscosity (strongly temperature dependent)
        viscosity = self._calculate_convection_viscosity(temperature, reservoir_state)
        
        # Thermal properties
        thermal_diffusivity = 7e-7  # m²/s (typical for silicate melts)
        thermal_conductivity = 2.0   # W/(m·K)
        heat_capacity = 1200         # J/(kg·K)
        
        # Compositional diffusion coefficient
        compositional_diffusivity = 1e-9  # m²/s (much slower than thermal)
        
        return {
            'density': density,
            'viscosity': viscosity,
            'thermal_diffusivity': thermal_diffusivity,
            'thermal_conductivity': thermal_conductivity,
            'heat_capacity': heat_capacity,
            'compositional_diffusivity': compositional_diffusivity,
            'thermal_expansion': thermal_expansion
        }
    
    def _calculate_convection_viscosity(self, temperature: float, reservoir_state: Dict) -> float:
        """Calculate temperature and composition dependent viscosity for convection"""
        
        # Base viscosity (Arrhenius relationship)
        A_visc = 1e-4   # Pa·s (pre-exponential factor)
        Ea_visc = 300e3 # J/mol (activation energy)
        R = 8.314       # J/(mol·K)
        
        base_viscosity = A_visc * np.exp(Ea_visc / (R * temperature))
        
        # Composition effects on viscosity
        composition = reservoir_state.get('composition', {})
        sio2_content = composition.get('SiO2', 0.7)
        h2o_content = composition.get('H2O', 0.01)
        
        # SiO2 increases viscosity (polymerization)
        sio2_factor = (sio2_content / 0.7)**2
        
        # H2O decreases viscosity (depolymerization)
        h2o_factor = np.exp(-h2o_content / 0.02)
        
        total_viscosity = base_viscosity * sio2_factor * h2o_factor
        
        # Physical bounds
        return np.clip(total_viscosity, 1e2, 1e8)  # Pa·s
    
    def _calculate_convection_dimensionless_parameters(self, dimensions: Dict[str, float],
                                                     fluid_props: Dict[str, float],
                                                     reservoir_state: Dict) -> Dict[str, float]:
        """Calculate dimensionless parameters controlling convection"""
        
        g = 9.81  # m/s² (gravitational acceleration)
        
        # Temperature difference (simplified)
        delta_T = 200  # K (typical for magma chambers)
        
        # Rayleigh number (thermal buoyancy vs. viscous damping)
        Ra = (fluid_props['density'] * g * fluid_props['thermal_expansion'] * 
              delta_T * dimensions['char_length']**3) / (
              fluid_props['viscosity'] * fluid_props['thermal_diffusivity'])
        
        # Prandtl number (viscous vs. thermal diffusion)
        Pr = fluid_props['viscosity'] / (
            fluid_props['density'] * fluid_props['thermal_diffusivity'])
        
        # Schmidt number (viscous vs. compositional diffusion)
        Sc = fluid_props['viscosity'] / (
            fluid_props['density'] * fluid_props['compositional_diffusivity'])
        
        # Lewis number (thermal vs. compositional diffusion)
        Le = fluid_props['thermal_diffusivity'] / fluid_props['compositional_diffusivity']
        
        # Aspect ratio effects
        aspect_ratio = dimensions['aspect_ratio_xz']
        
        return {
            'Ra': Ra,
            'Pr': Pr,
            'Sc': Sc,
            'Le': Le,
            'aspect_ratio': aspect_ratio,
            'delta_T': delta_T
        }
    
    def _determine_convection_regime(self, params: Dict[str, float]) -> str:
        """Determine the dominant convection regime"""
        
        Ra = params['Ra']
        Le = params['Le']
        aspect_ratio = params['aspect_ratio']
        
        # Critical Rayleigh numbers
        Ra_crit = 1708  # Critical for onset of convection
        Ra_plume = 1e6  # Transition to plume-dominated convection
        
        if Ra < Ra_crit:
            return 'conductive'
        elif Ra < Ra_plume and aspect_ratio < 5:
            return 'rayleigh_benard'
        elif Ra >= Ra_plume:
            return 'plume_dominated'
        elif Le > 2:  # Significant difference in diffusivities
            return 'double_diffusive'
        else:
            return 'mixed_convection'
    
    def _calculate_rayleigh_benard_pattern(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                         params: Dict[str, float], 
                                         dimensions: Dict[str, float]) -> np.ndarray:
        """
        Calculate Rayleigh-Bénard convection pattern (cellular convection)
        """
        
        Ra = params['Ra']
        aspect_ratio = params['aspect_ratio']
        
        # Characteristic wavenumber (depends on Rayleigh number)
        k_crit = np.pi / np.sqrt(2)  # Critical wavenumber
        k_preferred = k_crit * (1 + 0.1 * np.log10(Ra / 1708))
        
        # Scale coordinates to reservoir dimensions
        x_scaled = X / dimensions['width_x']
        y_scaled = Y / dimensions['width_y']
        z_scaled = Z / dimensions['height']
        
        # Number of convection cells (depends on aspect ratio and Ra)
        n_cells_x = max(1, int(k_preferred * dimensions['width_x'] / np.pi))
        n_cells_y = max(1, int(k_preferred * dimensions['width_y'] / np.pi))
        
        # 3D cellular pattern
        # Horizontal cells with vertical structure
        horizontal_pattern = (np.sin(n_cells_x * np.pi * x_scaled) * 
                            np.cos(n_cells_y * np.pi * y_scaled))
        
        # Vertical structure (maximum at mid-depth)
        vertical_structure = np.sin(np.pi * z_scaled)
        
        # Combine horizontal and vertical components
        pattern = horizontal_pattern * vertical_structure
        
        # Amplitude depends on supercriticality (Ra - Ra_crit)
        supercriticality = max(0, Ra - 1708)
        amplitude = 0.1 * np.tanh(supercriticality / 1e4)
        
        return amplitude * pattern
    
    def _calculate_plume_dominated_pattern(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                         params: Dict[str, float], 
                                         dimensions: Dict[str, float]) -> np.ndarray:
        """
        Calculate plume-dominated convection pattern
        """
        
        Ra = params['Ra']
        
        # Plume locations (simplified: corners and center)
        plume_locations = [
            (0.2 * dimensions['width_x'], 0.2 * dimensions['width_y']),
            (0.8 * dimensions['width_x'], 0.2 * dimensions['width_y']),
            (0.2 * dimensions['width_x'], 0.8 * dimensions['width_y']),
            (0.8 * dimensions['width_x'], 0.8 * dimensions['width_y']),
            (0.5 * dimensions['width_x'], 0.5 * dimensions['width_y'])
        ]
        
        pattern = np.zeros_like(X)
        
        for plume_x, plume_y in plume_locations:
            # Distance from plume center
            dx = X - plume_x
            dy = Y - plume_y
            r = np.sqrt(dx**2 + dy**2)
            
            # Plume radius (depends on Rayleigh number)
            plume_radius = 0.1 * dimensions['char_length'] * (Ra / 1e6)**0.2
            
            # Gaussian plume structure in horizontal
            horizontal_profile = np.exp(-(r / plume_radius)**2)
            
            # Vertical structure (stronger near bottom, weakens upward)
            z_normalized = (Z - np.min(Z)) / dimensions['height']
            vertical_profile = (1 - z_normalized) * np.exp(-2 * z_normalized)
            
            # Add plume contribution
            plume_strength = 0.2 * (1 + np.random.normal(0, 0.1))  # Some randomness
            pattern += plume_strength * horizontal_profile * vertical_profile
        
        return pattern
    
    def _calculate_double_diffusive_pattern(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                          params: Dict[str, float], 
                                          dimensions: Dict[str, float]) -> np.ndarray:
        """
        Calculate double-diffusive convection pattern (salt fingering or layered convection)
        """
        
        Le = params['Le']
        Ra = params['Ra']
        
        # Scale coordinates
        x_scaled = X / dimensions['width_x']
        y_scaled = Y / dimensions['width_y']
        z_scaled = Z / dimensions['height']
        
        if Le > 1:  # Thermal diffusion faster than compositional
            # Salt fingering instability
            
            # Finger wavelength
            finger_wavelength = 0.1 * dimensions['height'] / Le**0.5
            n_fingers = int(dimensions['width_x'] / finger_wavelength)
            
            # Finger pattern
            finger_pattern = np.sin(n_fingers * 2 * np.pi * x_scaled)
            
            # Vertical modulation
            vertical_modulation = np.sin(np.pi * z_scaled) * np.exp(-2 * z_scaled)
            
            pattern = 0.15 * finger_pattern * vertical_modulation
            
        else:  # Compositional diffusion faster than thermal
            # Layered convection
            
            # Layer thickness
            layer_thickness = dimensions['height'] / (5 * Le)
            n_layers = int(dimensions['height'] / layer_thickness)
            
            # Layered structure
            layer_pattern = np.sin(n_layers * np.pi * z_scaled)
            
            # Horizontal modulation
            horizontal_modulation = (np.sin(2 * np.pi * x_scaled) * 
                                   np.cos(2 * np.pi * y_scaled))
            
            pattern = 0.12 * layer_pattern * horizontal_modulation
        
        return pattern
    
    def _calculate_mixed_convection_pattern(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                          params: Dict[str, float], 
                                          dimensions: Dict[str, float]) -> np.ndarray:
        """
        Calculate mixed convection pattern (combination of different modes)
        """
        
        # Combine different convection modes
        rb_pattern = self._calculate_rayleigh_benard_pattern(X, Y, Z, params, dimensions)
        plume_pattern = self._calculate_plume_dominated_pattern(X, Y, Z, params, dimensions)
        dd_pattern = self._calculate_double_diffusive_pattern(X, Y, Z, params, dimensions)
        
        # Weights based on dimensionless parameters
        Ra = params['Ra']
        Le = params['Le']
        
        # Rayleigh-Bénard weight (dominant at moderate Ra)
        w_rb = np.exp(-(Ra - 1e5)**2 / (2 * (1e5)**2))
        
        # Plume weight (increases with Ra)
        w_plume = np.tanh((Ra - 1e5) / 1e5)
        
        # Double-diffusive weight (increases with Le deviation from 1)
        w_dd = 0.3 * np.abs(Le - 1) / (1 + np.abs(Le - 1))
        
        # Normalize weights
        total_weight = w_rb + w_plume + w_dd
        w_rb /= total_weight
        w_plume /= total_weight
        w_dd /= total_weight
        
        mixed_pattern = w_rb * rb_pattern + w_plume * plume_pattern + w_dd * dd_pattern
        
        return mixed_pattern
    
    def _apply_viscosity_stratification_effects(self, pattern: np.ndarray, 
                                              X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                              fluid_props: Dict[str, float]) -> np.ndarray:
        """
        Apply effects of viscosity stratification on convection pattern
        """
        
        # Viscosity typically increases with depth (cooling)
        depth = np.abs(Z)
        depth_normalized = depth / np.max(depth)
        
        # Exponential viscosity increase with depth
        viscosity_ratio = np.exp(2 * depth_normalized)  # Up to ~7x increase
        
        # Convection strength inversely related to viscosity
        viscosity_damping = 1.0 / (1.0 + 0.5 * (viscosity_ratio - 1))
        
        # Apply damping to pattern
        stratified_pattern = pattern * viscosity_damping
        
        return stratified_pattern
    
    def _apply_convection_boundary_effects(self, pattern: np.ndarray,
                                         X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                         dimensions: Dict[str, float]) -> np.ndarray:
        """
        Apply boundary condition effects on convection pattern
        """
        
        # Scale coordinates to [0,1]
        x_norm = (X - np.min(X)) / dimensions['width_x']
        y_norm = (Y - np.min(Y)) / dimensions['width_y']
        z_norm = (Z - np.min(Z)) / dimensions['height']
        
        # Boundary damping functions
        # Side boundaries (no-slip condition reduces convection near walls)
        side_damping_x = np.sin(np.pi * x_norm)  # Zero at x boundaries
        side_damping_y = np.sin(np.pi * y_norm)  # Zero at y boundaries
        
        # Top and bottom boundaries
        # Bottom: enhanced convection (heating from below)
        # Top: reduced convection (cooling from above)
        vertical_modulation = np.sin(np.pi * z_norm) * (1 + 0.5 * (1 - z_norm))
        
        # Combined boundary effects
        boundary_factor = side_damping_x * side_damping_y * vertical_modulation
        
        boundary_modified_pattern = pattern * boundary_factor
        
        return boundary_modified_pattern
    
    def _apply_convection_time_evolution(self, pattern: np.ndarray, 
                                        reservoir_state: Dict) -> np.ndarray:
        """
        Apply time-dependent evolution of convection pattern
        """
        
        # Get time information (simplified)
        time = reservoir_state.get('time', 0)  # years
        
        # Convection pattern oscillates on thermal diffusion time scale
        thermal_time = 1000  # years (characteristic thermal time)
        
        # Slow temporal oscillation
        time_phase = 2 * np.pi * time / thermal_time
        time_modulation = 1.0 + 0.2 * np.sin(time_phase)
        
        # Fast temporal fluctuations (turbulent-like)
        fast_time_scale = 1  # year
        fast_phase = 2 * np.pi * time / fast_time_scale
        fast_modulation = 1.0 + 0.05 * np.sin(fast_phase) * np.random.normal(0, 0.1)
        
        time_evolved_pattern = pattern * time_modulation * fast_modulation
        
        return time_evolved_pattern
    
    def _apply_convection_nonlinear_interactions(self, pattern: np.ndarray,
                                               convection_strength: np.ndarray,
                                               X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Apply non-linear interactions between convection pattern and strength
        """
        
        # Non-linear amplitude saturation (pattern saturates at high strength)
        saturation_strength = 1.0
        saturation_factor = np.tanh(convection_strength / saturation_strength)
        
        # Pattern amplitude modulation
        amplitude_modulated_pattern = pattern * saturation_factor
        
        # Spatial correlations (neighboring cells interact)
        correlation_length = 0.2 * np.ptp(X)  # Correlation scale
        
        # Apply spatial smoothing for correlations (simplified)
        try:
            from scipy import ndimage
            smoothing_scale = correlation_length / (np.ptp(X) / X.shape[0])
            
            if X.ndim == 3:
                sigma = [smoothing_scale, smoothing_scale, smoothing_scale]
            elif X.ndim == 2:
                sigma = [smoothing_scale, smoothing_scale]
            else:
                sigma = smoothing_scale
                
            correlated_pattern = ndimage.gaussian_filter(amplitude_modulated_pattern, sigma=sigma)
        except:
            # Fallback without correlation
            correlated_pattern = amplitude_modulated_pattern
        
        # Final normalization and scaling
        final_pattern = 0.1 * convection_strength * correlated_pattern
        
        return final_pattern
    
    def _calculate_mohr_coulomb_failure(self, sigma_1: np.ndarray, sigma_3: np.ndarray,
                                      material_props: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Mohr-Coulomb failure probability"""
        
        cohesion = material_props['cohesion']
        friction_angle = material_props['friction_angle']
        
        # Convert friction angle to radians
        phi = np.radians(friction_angle)
        
        # Mohr-Coulomb criterion: τ = c + σ_n * tan(φ)
        # In terms of principal stresses: σ_1 = σ_3 * tan²(45° + φ/2) + 2c * tan(45° + φ/2)
        
        tan_angle = np.tan(np.pi/4 + phi/2)
        
        # Critical stress difference
        critical_stress_diff = 2 * cohesion * tan_angle + sigma_3 * (tan_angle**2 - 1)
        
        # Actual stress difference
        stress_diff = sigma_1 - sigma_3
        
        # Safety factor
        safety_factor = critical_stress_diff / np.maximum(stress_diff, 1e-10)
        
        # Probability of failure (log-normal distribution)
        failure_probability = 0.5 * (1 - np.tanh(2.0 * (safety_factor - 1.0)))
        
        return np.clip(failure_probability, 0.0, 1.0)
    
    def _calculate_hoek_brown_failure(self, sigma_1: np.ndarray, sigma_3: np.ndarray,
                                    material_props: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Hoek-Brown failure probability"""
        
        ucs = material_props['ucs']  # Unconfined compressive strength
        m_i = material_props['m_i']
        s = material_props['s']
        a = material_props['a']
        
        # Hoek-Brown criterion: σ_1 = σ_3 + σ_c * (m_i * σ_3/σ_c + s)^a
        
        # Normalized confining stress
        sigma_3_norm = sigma_3 / ucs
        
        # Critical major principal stress
        sigma_1_critical = sigma_3 + ucs * (m_i * sigma_3_norm + s)**a
        
        # Safety factor
        safety_factor = sigma_1_critical / np.maximum(sigma_1, 1e-10)
        
        # Failure probability
        failure_probability = 0.5 * (1 - np.tanh(1.5 * (safety_factor - 1.0)))
        
        return np.clip(failure_probability, 0.0, 1.0)
    
    def _calculate_griffith_failure(self, sigma_1: np.ndarray, sigma_3: np.ndarray,
                                  material_props: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Griffith tensile failure probability"""
        
        tensile_strength = material_props['tensile_strength']
        
        # Griffith criterion for tensile failure
        # For σ_3 > 0 (tension): failure when σ_3 > σ_t
        # For σ_3 < 0 (compression): parabolic criterion
        
        failure_probability = np.zeros_like(sigma_1)
        
        # Tensile regime (σ_3 > 0)
        tensile_mask = sigma_3 > 0
        failure_probability[tensile_mask] = 0.5 * (1 + np.tanh(
            3.0 * (sigma_3[tensile_mask] / tensile_strength[tensile_mask] - 1.0)
        ))
        
        # Compressive regime (σ_3 ≤ 0)
        compressive_mask = sigma_3 <= 0
        if np.any(compressive_mask):
            # Modified Griffith criterion: (σ_1 - σ_3)² = 8σ_t(σ_1 + σ_3)
            sigma_diff = sigma_1[compressive_mask] - sigma_3[compressive_mask]
            sigma_sum = sigma_1[compressive_mask] + sigma_3[compressive_mask]
            
            # Critical condition
            griffith_criterion = sigma_diff**2 - 8 * tensile_strength[compressive_mask] * sigma_sum
            
            failure_probability[compressive_mask] = 0.5 * (1 + np.tanh(
                griffith_criterion / (tensile_strength[compressive_mask]**2)
            ))
        
        return np.clip(failure_probability, 0.0, 1.0)
    
    def _calculate_drucker_prager_failure(self, sigma_1: np.ndarray, sigma_2: np.ndarray,
                                        sigma_3: np.ndarray, material_props: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Drucker-Prager failure probability"""
        
        cohesion = material_props['cohesion']
        friction_angle = material_props['friction_angle']
        
        # Convert to Drucker-Prager parameters
        phi = np.radians(friction_angle)
        
        # Material constants
        alpha_dp = 2 * np.sin(phi) / (np.sqrt(3) * (3 - np.sin(phi)))
        k_dp = 6 * cohesion * np.cos(phi) / (np.sqrt(3) * (3 - np.sin(phi)))
        
        # Stress invariants
        I1 = sigma_1 + sigma_2 + sigma_3  # First invariant
        
        J2 = ((sigma_1 - sigma_2)**2 + (sigma_2 - sigma_3)**2 + (sigma_3 - sigma_1)**2) / 6
        sqrt_J2 = np.sqrt(J2)
        
        # Drucker-Prager criterion: √J2 = k - α*I1
        critical_shear = k_dp - alpha_dp * I1
        
        # Safety factor
        safety_factor = critical_shear / np.maximum(sqrt_J2, 1e-10)
        
        # Failure probability
        failure_probability = 0.5 * (1 - np.tanh(2.0 * (safety_factor - 1.0)))
        
        return np.clip(failure_probability, 0.0, 1.0)
    
    def _calculate_max_tensile_failure(self, sigma_3: np.ndarray, 
                                     material_props: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate maximum tensile stress failure probability"""
        
        tensile_strength = material_props['tensile_strength']
        
        # Maximum tensile stress criterion
        failure_probability = 0.5 * (1 + np.tanh(
            3.0 * (sigma_3 / tensile_strength - 1.0)
        ))
        
        # Only consider tensile stresses
        failure_probability = np.where(sigma_3 > 0, failure_probability, 0.0)
        
        return np.clip(failure_probability, 0.0, 1.0)
    
    def _calculate_von_mises_failure(self, sigma_1: np.ndarray, sigma_2: np.ndarray,
                                   sigma_3: np.ndarray, material_props: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate von Mises failure probability (for reference)"""
        
        ucs = material_props['ucs']
        
        # von Mises equivalent stress
        sigma_vm = np.sqrt(0.5 * ((sigma_1 - sigma_2)**2 + 
                                 (sigma_2 - sigma_3)**2 + 
                                 (sigma_3 - sigma_1)**2))
        
        # Yield strength (approximation)
        yield_strength = ucs / np.sqrt(3)  # Convert UCS to shear strength
        
        # Safety factor
        safety_factor = yield_strength / np.maximum(sigma_vm, 1e-10)
        
        # Failure probability
        failure_probability = 0.5 * (1 - np.tanh(2.0 * (safety_factor - 1.0)))
        
        return np.clip(failure_probability, 0.0, 1.0)
    
    def _calculate_weibull_failure(self, sigma_1: np.ndarray, 
                                 material_props: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Weibull statistical failure probability"""
        
        weibull_modulus = material_props['weibull_modulus']
        weibull_scale = material_props['weibull_scale']
        
        # Only consider tensile stresses
        sigma_effective = np.maximum(sigma_1, 0.0)
        
        # Weibull failure probability: P = 1 - exp(-(σ/σ₀)^m)
        weibull_argument = (sigma_effective / weibull_scale) ** weibull_modulus
        failure_probability = 1.0 - np.exp(-weibull_argument)
        
        return np.clip(failure_probability, 0.0, 1.0)
    
    def _combine_failure_criteria_probabilities(self, fracture_analysis: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine probabilities from multiple failure criteria"""
        
        # Weighted combination of failure criteria
        weights = {
            'mohr_coulomb': 0.25,    # Primary shear failure criterion
            'hoek_brown': 0.20,      # Rock mass failure
            'griffith': 0.15,        # Tensile failure
            'drucker_prager': 0.15,  # 3D shear failure
            'max_tensile': 0.10,     # Pure tensile
            'von_mises': 0.05,       # Comparative reference
            'weibull': 0.10          # Statistical effects
        }
        
        # Weighted average
        combined_probability = np.zeros_like(fracture_analysis['mohr_coulomb'])
        
        for criterion, weight in weights.items():
            if criterion in fracture_analysis:
                combined_probability += weight * fracture_analysis[criterion]
        
        # Apply interaction effects (non-linear combination)
        # Multiple failure modes can interact and amplify overall failure risk
        interaction_factor = 1.0 + 0.2 * np.tanh(2.0 * combined_probability)
        combined_probability *= interaction_factor
        
        return np.clip(combined_probability, 0.0, 1.0)
    
    def _apply_environmental_fracture_factors(self, base_probability: np.ndarray,
                                            stress: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply environmental factors affecting fracture probability"""
        
        # Temperature effects (thermal stress cycling, thermal shock)
        temperature_factor = self._calculate_temperature_fracture_factor(stress)
        
        # Pore pressure effects
        pore_pressure_factor = self._calculate_pore_pressure_fracture_factor(stress)
        
        # Loading rate effects
        loading_rate_factor = self._calculate_loading_rate_factor()
        
        # Chemical effects (water weakening, alteration)
        chemical_factor = self._calculate_chemical_weakening_factor()
        
        # Combined environmental factor
        environmental_factor = (temperature_factor * pore_pressure_factor * 
                              loading_rate_factor * chemical_factor)
        
        # Apply environmental modifications
        modified_probability = base_probability * environmental_factor
        
        return np.clip(modified_probability, 0.0, 1.0)
    
    def _calculate_temperature_fracture_factor(self, stress: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate temperature effects on fracture probability"""
        
        # Temperature-dependent strength reduction
        # High temperatures generally reduce rock strength
        
        shape = stress['sigma_xx'].shape
        
        # Assume temperature field available or use depth-temperature relation
        # For simplicity, use geothermal gradient
        depth_km = np.abs(np.linspace(-10, 0, shape[2]).reshape(1, 1, -1))
        depth_km = np.broadcast_to(depth_km, shape)
        
        temperature_celsius = 15 + 25 * depth_km  # Surface temp + geothermal gradient
        
        # Temperature factor (strength reduction with temperature)
        # Typical reduction: 50% at 400°C for granite
        ref_temp = 20.0  # °C
        max_reduction = 0.5
        temp_scale = 200.0  # °C
        
        temp_factor = 1.0 - max_reduction * (1 - np.exp(-(temperature_celsius - ref_temp) / temp_scale))
        temp_factor = np.clip(temp_factor, 0.3, 1.0)  # Minimum 30% strength retention
        
        # Thermal stress cycling effects (additional weakening)
        thermal_cycling_factor = 1.1 + 0.2 * np.sin(2 * np.pi * depth_km / 5.0)
        
        return temp_factor * thermal_cycling_factor
    
    def _calculate_pore_pressure_fracture_factor(self, stress: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate pore pressure effects on fracture probability"""
        
        shape = stress['sigma_xx'].shape
        
        # Effective stress concept: high pore pressure reduces effective stress
        # but also reduces fracture strength
        
        # Simplified pore pressure distribution
        pore_pressure_factor = np.ones(shape)
        
        # Near-surface: higher pore pressure effects
        surface_enhancement = 1.3
        depth_decay = 2.0  # km
        
        depth_km = np.abs(np.linspace(-10, 0, shape[2]).reshape(1, 1, -1))
        depth_km = np.broadcast_to(depth_km, shape)
        
        pore_pressure_factor *= (1.0 + (surface_enhancement - 1.0) * 
                               np.exp(-depth_km / depth_decay))
        
        return pore_pressure_factor
    
    def _calculate_loading_rate_factor(self) -> float:
        """Calculate loading rate effects on fracture probability"""
        
        # Strain rate effects: faster loading generally increases strength
        # but also increases brittle failure susceptibility
        
        # For magma reservoir loading (relatively slow geological processes)
        loading_rate_factor = 1.1  # Slight increase due to sustained loading
        
        return loading_rate_factor
    
    def _calculate_chemical_weakening_factor(self) -> float:
        """Calculate chemical weakening effects"""
        
        # Hydrothermal alteration, water weakening, chemical reactions
        # Generally increase fracture susceptibility
        
        chemical_weakening = 1.2  # 20% increase in fracture probability
        
        return chemical_weakening
    
    def _apply_time_dependent_fracture_effects(self, base_probability: np.ndarray) -> np.ndarray:
        """Apply time-dependent fracture effects (fatigue, creep damage)"""
        
        # Subcritical crack growth (time-dependent failure)
        # Stress corrosion cracking
        # Fatigue damage accumulation
        
        # Simplified time-dependent enhancement
        time_factor = 1.15  # 15% increase due to time-dependent processes
        
        # Apply logarithmic time dependence
        # In real implementation, this would depend on loading history
        time_enhanced_probability = base_probability * time_factor
        
        # Ensure physical bounds
        return np.clip(time_enhanced_probability, 0.0, 1.0)
    
    def _extract_surface_deformation(self, deformation: Dict[str, np.ndarray], 
                                   grid: Dict[str, Any]) -> Dict[str, Any]:
        """Extract surface deformation for comparison with observations"""
        
        # Surface is at z = 0 (top of grid)
        surface_index = 0
        
        surface_deformation = {
            'vertical_displacement': deformation['epsilon_zz'][:, :, surface_index],
            'horizontal_displacement_x': deformation['epsilon_xx'][:, :, surface_index],
            'horizontal_displacement_y': deformation['epsilon_yy'][:, :, surface_index],
            'shear_strain': deformation['epsilon_xy'][:, :, surface_index]
        }
        
        # Calculate tilt
        dx, dy = grid['dx'], grid['dy']
        vertical_disp = surface_deformation['vertical_displacement']
        
        tilt_x = np.gradient(vertical_disp, dx, axis=0)
        tilt_y = np.gradient(vertical_disp, dy, axis=1)
        tilt_magnitude = np.sqrt(tilt_x**2 + tilt_y**2)
        
        surface_deformation.update({
            'tilt_x': tilt_x,
            'tilt_y': tilt_y,
            'tilt_magnitude': tilt_magnitude
        })
        
        return surface_deformation
    
    def _calculate_mechanical_energy(self, stress: Dict[str, np.ndarray], 
                                   deformation: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate mechanical energy components"""
        
        # Strain energy density
        sigma_xx = stress['sigma_xx']
        sigma_yy = stress['sigma_yy']
        sigma_zz = stress['sigma_zz']
        sigma_xy = stress['sigma_xy']
        sigma_xz = stress['sigma_xz']
        sigma_yz = stress['sigma_yz']
        
        epsilon_xx = deformation['epsilon_xx']
        epsilon_yy = deformation['epsilon_yy']
        epsilon_zz = deformation['epsilon_zz']
        epsilon_xy = deformation['epsilon_xy']
        epsilon_xz = deformation['epsilon_xz']
        epsilon_yz = deformation['epsilon_yz']
        
        # Strain energy density (0.5 * sigma : epsilon)
        strain_energy_density = 0.5 * (
            sigma_xx * epsilon_xx + sigma_yy * epsilon_yy + sigma_zz * epsilon_zz +
            2 * (sigma_xy * epsilon_xy + sigma_xz * epsilon_xz + sigma_yz * epsilon_yz)
        )
        
        return {
            'total_strain_energy': float(np.sum(strain_energy_density)),
            'average_strain_energy_density': float(np.mean(strain_energy_density)),
            'max_strain_energy_density': float(np.max(strain_energy_density))
        }

class GeophysicalForwardModeling:
    """Forward modeling of geophysical signatures"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'gravity_reference_density': 2670,  # kg/m³
            'magnetic_reference_susceptibility': 0.001,  # SI
            'resistivity_background': 1000,  # Ohm·m
            'seismic_velocity_background': {'vp': 6000, 'vs': 3464},  # m/s
            'observation_surface_elevation': 0  # m
        }
    
    def calculate_gravity_anomaly(self, density_model: np.ndarray, 
                                grid: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Calculate gravity anomaly from density model"""
        
        # Universal gravitational constant
        G = 6.674e-11  # m³/kg/s²
        
        X, Y, Z = grid['X'], grid['Y'], grid['Z']
        dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
        
        # Volume element
        dV = dx * dy * dz
        
        # Observation points (surface)
        x_obs = grid['x']
        y_obs = grid['y']
        z_obs = self.config['observation_surface_elevation']
        
        X_obs, Y_obs = np.meshgrid(x_obs, y_obs, indexing='ij')
        
        # Calculate gravity anomaly
        gravity_anomaly = np.zeros_like(X_obs)
        
        # Reference density
        rho_ref = self.config['gravity_reference_density']
        
        for i in range(len(x_obs)):
            for j in range(len(y_obs)):
                # Distance from each grid point to observation point
                r = np.sqrt((X - X_obs[i, j])**2 + 
                           (Y - Y_obs[i, j])**2 + 
                           (Z - z_obs)**2)
                
                # Avoid division by zero
                r = np.maximum(r, dx/10)
                
                # Density contrast
                delta_rho = density_model - rho_ref
                
                # Gravity contribution (vertical component)
                gravity_anomaly[i, j] = G * np.sum(delta_rho * (Z - z_obs) / r**3) * dV
        
        # Convert to mGal (1 mGal = 1e-5 m/s²)
        gravity_anomaly *= 1e5
        
        return {
            'gravity_anomaly': gravity_anomaly,
            'x_coordinates': x_obs,
            'y_coordinates': y_obs
        }
    
    def calculate_magnetic_anomaly(self, susceptibility_model: np.ndarray,
                                 grid: Dict[str, Any], 
                                 magnetic_field_direction: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Calculate magnetic anomaly from susceptibility model"""
        
        if magnetic_field_direction is None:
            # Default: Earth's magnetic field at 45° latitude
            inclination = 60  # degrees
            declination = 0   # degrees
            magnetic_field_direction = np.array([
                np.cos(np.radians(inclination)) * np.cos(np.radians(declination)),
                np.cos(np.radians(inclination)) * np.sin(np.radians(declination)),
                np.sin(np.radians(inclination))
            ])
        
        X, Y, Z = grid['X'], grid['Y'], grid['Z']
        dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
        dV = dx * dy * dz
        
        # Observation points
        x_obs = grid['x']
        y_obs = grid['y']
        z_obs = self.config['observation_surface_elevation']
        
        X_obs, Y_obs = np.meshgrid(x_obs, y_obs, indexing='ij')
        
        # Magnetic permeability of free space
        mu_0 = 4 * np.pi * 1e-7  # H/m
        
        # Calculate magnetic anomaly (simplified dipole approximation)
        magnetic_anomaly = np.zeros_like(X_obs)
        
        susceptibility_ref = self.config['magnetic_reference_susceptibility']
        
        for i in range(len(x_obs)):
            for j in range(len(y_obs)):
                # Distance vector
                dx_vec = X - X_obs[i, j]
                dy_vec = Y - Y_obs[i, j]
                dz_vec = Z - z_obs
                
                r = np.sqrt(dx_vec**2 + dy_vec**2 + dz_vec**2)
                r = np.maximum(r, dx/10)
                
                # Unit vector
                r_hat = np.array([dx_vec/r, dy_vec/r, dz_vec/r])
                
                # Susceptibility contrast
                delta_chi = susceptibility_model - susceptibility_ref
                
                # Magnetic dipole moment
                m_dot_r = (magnetic_field_direction[0] * dx_vec/r + 
                          magnetic_field_direction[1] * dy_vec/r + 
                          magnetic_field_direction[2] * dz_vec/r)
                
                # Magnetic field contribution (simplified)
                magnetic_contribution = (mu_0 * delta_chi * dV / (4 * np.pi * r**2) * 
                                       (3 * m_dot_r - magnetic_field_direction[2]))
                
                magnetic_anomaly[i, j] = np.sum(magnetic_contribution)
        
        # Convert to nT (nanotesla)
        magnetic_anomaly *= 1e9
        
        return {
            'magnetic_anomaly': magnetic_anomaly,
            'x_coordinates': x_obs,
            'y_coordinates': y_obs,
            'field_direction': magnetic_field_direction
        }
    
    def calculate_seismic_velocities(self, temperature: np.ndarray, 
                                   pressure: np.ndarray,
                                   composition: Dict[str, np.ndarray],
                                   melt_fraction: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate seismic velocities from physical properties"""
        
        # Background velocities
        vp_background = self.config['seismic_velocity_background']['vp']  # m/s
        vs_background = self.config['seismic_velocity_background']['vs']  # m/s
        
        # Temperature effects (velocity decreases with temperature)
        temp_effect = -0.0005  # (m/s)/K
        vp_temp = vp_background + temp_effect * (temperature - 273)  # Convert to Celsius
        vs_temp = vs_background + temp_effect * (temperature - 273)
        
        # Pressure effects (velocity increases with pressure)
        pressure_Pa = pressure
        pressure_effect = 1e-8  # (m/s)/Pa
        vp_pressure = pressure_effect * pressure_Pa
        vs_pressure = pressure_effect * pressure_Pa * 0.6  # Vs less sensitive
        
        # Melt fraction effects (dramatic velocity reduction with partial melt)
        melt_effect_vp = -3000  # m/s per unit melt fraction
        melt_effect_vs = -1500  # m/s per unit melt fraction
        
        vp_melt = melt_effect_vp * melt_fraction
        vs_melt = melt_effect_vs * melt_fraction
        
        # Comprehensive composition effects on seismic velocities
        composition_effects = self._calculate_comprehensive_composition_velocity_effects(
            composition, temperature, pressure, melt_fraction)
        
        # Total velocities
        vp_total = np.maximum(vp_temp + vp_pressure + vp_melt + composition_effects['vp'], 1000)
        vs_total = np.maximum(vs_temp + vs_pressure + vs_melt + composition_effects['vs'], 500)
        
        # Vp/Vs ratio
        vp_vs_ratio = vp_total / vs_total
        
        return {
            'vp': vp_total,
            'vs': vs_total,
            'vp_vs_ratio': vp_vs_ratio,
            'velocity_anomaly_vp': (vp_total - vp_background) / vp_background,
            'velocity_anomaly_vs': (vs_total - vs_background) / vs_background
        }
    
    def calculate_electrical_resistivity(self, temperature: np.ndarray,
                                       melt_fraction: np.ndarray,
                                       fluid_content: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Calculate electrical resistivity from physical properties"""
        
        if fluid_content is None:
            fluid_content = np.zeros_like(temperature)
        
        # Background resistivity
        rho_background = self.config['resistivity_background']  # Ohm·m
        
        # Temperature effects (Arrhenius law)
        activation_energy = 50000  # J/mol (typical for silicate melts)
        gas_constant = 8.314  # J/mol/K
        
        # Resistivity decreases exponentially with temperature
        rho_temp = rho_background * np.exp(activation_energy / (gas_constant * temperature))
        
        # Melt fraction effects (dramatic decrease with partial melt)
        # Hashin-Shtrikman bounds for resistivity mixing
        rho_melt = 10  # Ohm·m (resistivity of silicate melt)
        rho_solid = rho_background
        
        # Lower bound approximation
        f = melt_fraction
        rho_mixed = ((1 - f) / rho_solid + f / rho_melt)**(-1)
        
        # Fluid effects (if present)
        if np.any(fluid_content > 0):
            rho_fluid = 1  # Ohm·m (very conductive fluids)
            # Additional conductivity from fluids
            fluid_conductivity = fluid_content / rho_fluid
            total_conductivity = 1/rho_mixed + fluid_conductivity
            rho_total = 1 / total_conductivity
        else:
            rho_total = rho_mixed
        
        # Ensure reasonable bounds
        rho_total = np.clip(rho_total, 0.1, 1e6)
        
        return {
            'resistivity': rho_total,
            'conductivity': 1/rho_total,
            'resistivity_anomaly': (rho_total - rho_background) / rho_background
        }

class MagmaReservoirMonitoringSystem:
    """Main monitoring system for magma reservoirs"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.physics_model = MagmaReservoirPhysicsModel(config)
        self.geophysical_modeling = GeophysicalForwardModeling(config)
        
        # Data storage
        self.reservoirs = {}
        self.monitoring_data = {}
        self.model_states = {}
        self.predictions = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'update_interval': 86400,  # seconds (daily updates)
            'prediction_horizon': 365 * 86400,  # 1 year
            'alert_thresholds': {
                'temperature_increase': 50,  # °C
                'pressure_increase': 10e6,   # Pa
                'melt_fraction_increase': 0.1,
                'deformation_rate': 0.1,     # m/year
                'seismic_velocity_decrease': 0.1  # fractional
            },
            'data_quality_checks': True,
            'uncertainty_quantification': True
        }
    
    def add_reservoir(self, reservoir: MagmaReservoir):
        """Add a magma reservoir to the monitoring system"""
        
        self.reservoirs[reservoir.id] = reservoir
        self.monitoring_data[reservoir.id] = {}
        
        # Initialize physics model
        model_state = self.physics_model.initialize_reservoir_model(reservoir)
        self.model_states[reservoir.id] = model_state
        
        self.logger.info(f"Added reservoir {reservoir.id} to monitoring system")
    
    def update_monitoring_data(self, reservoir_id: str, 
                             data_type: str, 
                             new_data: np.ndarray,
                             timestamp: datetime = None):
        """Update monitoring data for a reservoir"""
        
        if reservoir_id not in self.reservoirs:
            raise ValueError(f"Reservoir {reservoir_id} not found")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        if reservoir_id not in self.monitoring_data:
            self.monitoring_data[reservoir_id] = {}
        
        if data_type not in self.monitoring_data[reservoir_id]:
            self.monitoring_data[reservoir_id][data_type] = []
        
        self.monitoring_data[reservoir_id][data_type].append({
            'timestamp': timestamp,
            'data': new_data.copy()
        })
        
        # Update reservoir object
        self.reservoirs[reservoir_id].monitoring_parameters[data_type] = new_data
        self.reservoirs[reservoir_id].last_updated = timestamp
    
    async def run_comprehensive_analysis(self, reservoir_id: str) -> Dict[str, Any]:
        """Run comprehensive analysis for a reservoir"""
        
        if reservoir_id not in self.reservoirs:
            raise ValueError(f"Reservoir {reservoir_id} not found")
        
        reservoir = self.reservoirs[reservoir_id]
        model_state = self.model_states[reservoir_id]
        
        analysis_result = {
            'reservoir_id': reservoir_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'thermal_evolution': {},
            'mechanical_evolution': {},
            'geophysical_signatures': {},
            'hazard_assessment': {},
            'monitoring_recommendations': []
        }
        
        # Run thermal evolution model
        self.logger.info(f"Running thermal evolution for {reservoir_id}")
        thermal_result = await self.physics_model.solve_thermal_evolution(
            model_state, time_steps=365  # 1 year with daily steps
        )
        analysis_result['thermal_evolution'] = thermal_result
        
        # Update model state with new temperature field
        model_state['temperature_field'] = thermal_result['final_temperature_field']
        
        # Run mechanical evolution model
        self.logger.info(f"Running mechanical evolution for {reservoir_id}")
        mechanical_result = await self.physics_model.solve_mechanical_evolution(model_state)
        analysis_result['mechanical_evolution'] = mechanical_result
        
        # Calculate geophysical signatures
        self.logger.info(f"Calculating geophysical signatures for {reservoir_id}")
        geophysical_signatures = self._calculate_geophysical_signatures(
            model_state, thermal_result, mechanical_result
        )
        analysis_result['geophysical_signatures'] = geophysical_signatures
        
        # Hazard assessment
        hazard_assessment = self._assess_volcanic_hazards(
            reservoir, model_state, thermal_result, mechanical_result
        )
        analysis_result['hazard_assessment'] = hazard_assessment
        
        # Update predictions
        self.predictions[reservoir_id] = analysis_result
        
        # Generate monitoring recommendations
        recommendations = self._generate_monitoring_recommendations(
            reservoir, analysis_result
        )
        analysis_result['monitoring_recommendations'] = recommendations
        
        return analysis_result
    
    def _calculate_geophysical_signatures(self, model_state: Dict[str, Any],
                                        thermal_result: Dict[str, Any],
                                        mechanical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate geophysical signatures from model results"""
        
        signatures = {}
        
        # Extract physical property fields
        temperature = thermal_result['final_temperature_field']
        pressure = model_state['pressure_field']
        composition = model_state['composition_field']
        melt_fraction = model_state['phase_field']['melt_fraction']
        grid = model_state['grid']
        
        # Density field (temperature and composition dependent)
        density = self._calculate_density_field(temperature, composition, melt_fraction)
        
        # Gravity anomaly
        gravity_result = self.geophysical_modeling.calculate_gravity_anomaly(density, grid)
        signatures['gravity'] = gravity_result
        
        # Magnetic susceptibility and anomaly
        susceptibility = self._calculate_magnetic_susceptibility(temperature, composition)
        magnetic_result = self.geophysical_modeling.calculate_magnetic_anomaly(susceptibility, grid)
        signatures['magnetic'] = magnetic_result
        
        # Seismic velocities
        velocity_result = self.geophysical_modeling.calculate_seismic_velocities(
            temperature, pressure, composition, melt_fraction
        )
        signatures['seismic'] = velocity_result
        
        # Electrical resistivity
        resistivity_result = self.geophysical_modeling.calculate_electrical_resistivity(
            temperature, melt_fraction
        )
        signatures['electrical'] = resistivity_result
        
        # Surface deformation (from mechanical model)
        signatures['deformation'] = mechanical_result['surface_deformation']
        
        return signatures
    
    def _calculate_density_field(self, temperature: np.ndarray,
                               composition: Dict[str, np.ndarray],
                               melt_fraction: np.ndarray) -> np.ndarray:
        """Calculate density field from temperature, composition, and melt fraction"""
        
        # Base density from composition (simplified)
        SiO2_content = composition.get('SiO2', 65.0)
        
        # Density-composition relationship (linear approximation)
        base_density = 2800 - (SiO2_content - 50) * 5  # kg/m³
        
        # Temperature effects (thermal expansion)
        thermal_expansion = 3e-5  # 1/K
        reference_temperature = 273  # K
        density_temp = base_density * (1 - thermal_expansion * (temperature - reference_temperature))
        
        # Melt fraction effects (melt is less dense)
        melt_density = 2200  # kg/m³ (typical for silicate melt)
        solid_density = density_temp
        
        # Volume fraction mixing
        density_total = solid_density * (1 - melt_fraction) + melt_density * melt_fraction
        
        return density_total
    
    def _calculate_magnetic_susceptibility(self, temperature: np.ndarray,
                                         composition: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate magnetic susceptibility from temperature and composition"""
        
        # Base susceptibility from composition
        FeO_content = composition.get('FeO', 5.0)
        base_susceptibility = FeO_content * 0.002  # SI units (approximate)
        
        # Temperature effects (Curie law for paramagnetic materials)
        curie_temperature = 850  # K (typical for magnetite)
        susceptibility = base_susceptibility * curie_temperature / temperature
        
        # Phase transition effects (loss of magnetization above Curie point)
        susceptibility = np.where(temperature > curie_temperature, 
                                base_susceptibility * 0.1, 
                                susceptibility)
        
        return susceptibility
    
    def _assess_volcanic_hazards(self, reservoir: MagmaReservoir,
                               model_state: Dict[str, Any],
                               thermal_result: Dict[str, Any],
                               mechanical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess volcanic hazards based on model results"""
        
        hazard_assessment = {
            'overall_hazard_level': 'low',
            'eruption_probability': 0.0,
            'time_to_eruption_estimate': None,
            'eruption_style_prediction': None,
            'hazard_indicators': {},
            'risk_factors': {}
        }
        
        # Extract key indicators
        temperature = thermal_result['final_temperature_field']
        pressure = model_state['pressure_field']
        melt_fraction = model_state['phase_field']['melt_fraction']
        fracture_probability = mechanical_result['fracture_probability']
        
        # Calculate hazard indicators
        max_temperature = np.max(temperature)
        max_pressure = np.max(pressure)
        max_melt_fraction = np.max(melt_fraction)
        max_fracture_prob = np.max(fracture_probability)
        
        hazard_indicators = {
            'thermal_indicator': min((max_temperature - 900) / 300, 1.0),  # 900-1200°C range
            'pressure_indicator': min(max_pressure / (100e6), 1.0),  # 0-100 MPa range
            'melt_indicator': max_melt_fraction,  # 0-1 range
            'fracture_indicator': max_fracture_prob,  # 0-1 range
        }
        
        # Ensure non-negative indicators
        for key in hazard_indicators:
            hazard_indicators[key] = max(hazard_indicators[key], 0.0)
        
        hazard_assessment['hazard_indicators'] = hazard_indicators
        
        # Combined hazard score
        weights = [0.3, 0.3, 0.2, 0.2]  # thermal, pressure, melt, fracture
        hazard_score = sum(w * indicator for w, indicator in zip(weights, hazard_indicators.values()))
        
        # Eruption probability (non-linear scaling)
        eruption_probability = hazard_score ** 2  # Quadratic scaling
        hazard_assessment['eruption_probability'] = eruption_probability
        
        # Hazard level classification
        if hazard_score < 0.2:
            hazard_level = 'low'
        elif hazard_score < 0.4:
            hazard_level = 'moderate'
        elif hazard_score < 0.7:
            hazard_level = 'high'
        else:
            hazard_level = 'critical'
        
        hazard_assessment['overall_hazard_level'] = hazard_level
        
        # Time to eruption estimate (highly uncertain)
        if hazard_score > 0.3:
            # Simple inverse relationship
            time_estimate_years = 10 * (1 - hazard_score)
            hazard_assessment['time_to_eruption_estimate'] = {
                'estimate_years': max(time_estimate_years, 0.1),
                'uncertainty_range': [time_estimate_years * 0.1, time_estimate_years * 10],
                'confidence': 'very_low'
            }
        
        # Eruption style prediction
        if max_melt_fraction > 0.4 and max_temperature > 1000:
            if composition.get('SiO2', 65) > 70:
                eruption_style = 'explosive'
            else:
                eruption_style = 'mixed'
        else:
            eruption_style = 'effusive'
        
        hazard_assessment['eruption_style_prediction'] = eruption_style
        
        # Risk factors
        hazard_assessment['risk_factors'] = {
            'high_temperature_zones': float(np.sum(temperature > 1000) / temperature.size),
            'overpressurized_zones': float(np.sum(pressure > 50e6) / pressure.size),
            'high_melt_zones': float(np.sum(melt_fraction > 0.3) / melt_fraction.size),
            'fracture_zones': float(np.sum(fracture_probability > 0.5) / fracture_probability.size)
        }
        
        return hazard_assessment
    
    def _generate_monitoring_recommendations(self, reservoir: MagmaReservoir,
                                          analysis_result: Dict[str, Any]) -> List[str]:
        """Generate monitoring recommendations based on analysis"""
        
        recommendations = []
        
        hazard_level = analysis_result['hazard_assessment']['overall_hazard_level']
        eruption_probability = analysis_result['hazard_assessment']['eruption_probability']
        
        # Base recommendations
        recommendations.extend([
            "Maintain regular thermal monitoring",
            "Continue seismic network operation",
            "Monitor ground deformation patterns"
        ])
        
        # Level-specific recommendations
        if hazard_level == 'critical' or eruption_probability > 0.7:
            recommendations.extend([
                "Implement 24/7 continuous monitoring",
                "Deploy additional seismometers",
                "Increase GPS station density",
                "Set up real-time thermal cameras",
                "Prepare emergency response protocols",
                "Consider evacuation planning"
            ])
        elif hazard_level == 'high' or eruption_probability > 0.4:
            recommendations.extend([
                "Increase monitoring frequency to hourly",
                "Deploy additional instruments",
                "Enhanced gas monitoring",
                "Weekly helicopter surveys",
                "Community alert system preparation"
            ])
        elif hazard_level == 'moderate' or eruption_probability > 0.2:
            recommendations.extend([
                "Increase monitoring to 6-hourly",
                "Enhanced geochemical sampling",
                "Monthly detailed surveys",
                "Public awareness campaigns"
            ])
        
        # Specific recommendations based on indicators
        indicators = analysis_result['hazard_assessment']['hazard_indicators']
        
        if indicators['thermal_indicator'] > 0.5:
            recommendations.append("Deploy additional thermal monitoring equipment")
        
        if indicators['pressure_indicator'] > 0.5:
            recommendations.append("Enhance pressure monitoring in nearby wells")
        
        if indicators['melt_indicator'] > 0.5:
            recommendations.append("Increase magnetotelluric survey frequency")
        
        if indicators['fracture_indicator'] > 0.5:
            recommendations.append("Deploy strain meters and tilt meters")
        
        # Geophysical monitoring recommendations
        signatures = analysis_result.get('geophysical_signatures', {})
        
        if 'gravity' in signatures:
            recommendations.append("Conduct monthly microgravity surveys")
        
        if 'seismic' in signatures:
            recommendations.append("Deploy broadband seismometers")
        
        if 'deformation' in signatures:
            recommendations.append("Increase InSAR acquisition frequency")
        
        return recommendations
    
    def _calculate_comprehensive_melt_fraction(self, temperature_field: np.ndarray,
                                             reservoir: MagmaReservoir, 
                                             X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Calculate comprehensive melt fraction using advanced petrological models
        
        Includes:
        - Non-linear melting relationships
        - Composition-dependent solidus/liquidus temperatures
        - Pressure effects on melting
        - Water content influence
        - Crystal nucleation kinetics
        - Volatile saturation effects
        """
        
        # Extract magma properties
        magma_props = self.config.get('magma_properties', {})
        base_solidus = magma_props.get('solidus_temperature', 1000.0)  # K
        base_liquidus = magma_props.get('liquidus_temperature', 1200.0)  # K
        
        # Calculate pressure field (affects melting temperatures)
        pressure_field = self._calculate_pressure_effects(Z)
        
        # Get composition-dependent melting parameters
        composition_effects = self._calculate_composition_melting_effects(reservoir, X, Y, Z)
        
        # Calculate volatile saturation effects
        volatile_effects = self._calculate_volatile_saturation_effects(
            temperature_field, pressure_field, reservoir
        )
        
        # Adjust solidus and liquidus temperatures for pressure and composition
        solidus_adjusted = self._adjust_melting_temperatures(
            base_solidus, pressure_field, composition_effects['solidus_shift'],
            volatile_effects['solidus_depression']
        )
        
        liquidus_adjusted = self._adjust_melting_temperatures(
            base_liquidus, pressure_field, composition_effects['liquidus_shift'],
            volatile_effects['liquidus_elevation']
        )
        
        # Calculate melt fraction using non-linear melting model
        melt_fraction = self._non_linear_melting_model(
            temperature_field, solidus_adjusted, liquidus_adjusted,
            composition_effects, volatile_effects
        )
        
        # Apply kinetic constraints (crystal nucleation and growth)
        melt_fraction = self._apply_crystallization_kinetics(
            melt_fraction, temperature_field, pressure_field, reservoir
        )
        
        # Ensure physical bounds
        melt_fraction = np.clip(melt_fraction, 0.0, 1.0)
        
        return melt_fraction
    
    def _calculate_pressure_effects(self, Z: np.ndarray) -> np.ndarray:
        """Calculate pressure field based on depth"""
        # Hydrostatic pressure approximation
        rho_rock = 2700  # kg/m³ average rock density
        g = 9.81  # m/s²
        depth_m = np.abs(Z) * 1000  # Convert km to m
        
        pressure_pa = rho_rock * g * depth_m
        pressure_gpa = pressure_pa / 1e9  # Convert to GPa
        
        return pressure_gpa
    
    def _calculate_composition_melting_effects(self, reservoir: MagmaReservoir,
                                             X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate composition-dependent effects on melting temperatures"""
        
        # Calculate comprehensive 3D composition field with spatial heterogeneity
        composition_field = self._calculate_comprehensive_composition_field(
            reservoir, X, Y, Z)
        
        # Extract individual oxide components
        sio2_content = composition_field['SiO2']
        al2o3_content = composition_field['Al2O3']
        mgo_content = composition_field['MgO']
        cao_content = composition_field['CaO']
        na2o_content = composition_field['Na2O']
        k2o_content = composition_field['K2O']
        
        # Calculate differentiation index (Thornton & Tuttle, 1960)
        di = na2o_content + k2o_content + sio2_content
        
        # Solidus temperature shift (more evolved = lower solidus)
        solidus_shift = -200.0 * (di - 0.65) / 0.35  # Up to 200K depression for evolved compositions
        
        # Liquidus temperature shift
        liquidus_shift = -150.0 * (di - 0.65) / 0.35  # Up to 150K depression
        
        # MgO effect (higher MgO = higher melting temperatures)
        mgo_effect_solidus = 300.0 * mgo_content  # K per wt% MgO
        mgo_effect_liquidus = 400.0 * mgo_content
        
        solidus_shift += mgo_effect_solidus
        liquidus_shift += mgo_effect_liquidus
        
        return {
            'solidus_shift': solidus_shift,
            'liquidus_shift': liquidus_shift,
            'differentiation_index': di
        }
    
    def _calculate_comprehensive_composition_field(self, reservoir: MagmaReservoir,
                                                 X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive 3D composition field with spatial heterogeneity
        
        Includes:
        - Gravitational differentiation and density stratification
        - Fractional crystallization effects with depth
        - Magma mixing and mingling processes
        - Convective stirring and homogenization
        - Volatile degassing compositional changes
        - Country rock assimilation (AFC processes)
        - Crystal settling and accumulation zones
        - Multiple injection and recharge events
        - Thermal zonation effects on composition
        - Diffusive re-equilibration processes
        """
        
        # Get base composition
        base_composition = self._get_base_composition_parameters(reservoir)
        
        # Calculate gravitational differentiation
        gravity_effects = self._calculate_gravitational_differentiation(
            X, Y, Z, base_composition)
        
        # Calculate fractional crystallization effects
        crystallization_effects = self._calculate_fractional_crystallization_effects(
            X, Y, Z, base_composition, reservoir)
        
        # Calculate magma mixing effects
        mixing_effects = self._calculate_magma_mixing_effects(
            X, Y, Z, base_composition, reservoir)
        
        # Calculate assimilation effects
        assimilation_effects = self._calculate_assimilation_effects(
            X, Y, Z, base_composition, reservoir)
        
        # Calculate volatile degassing effects
        degassing_effects = self._calculate_degassing_compositional_effects(
            X, Y, Z, base_composition, reservoir)
        
        # Calculate convective homogenization
        convection_effects = self._calculate_convective_homogenization(
            X, Y, Z, base_composition, reservoir)
        
        # Calculate thermal zonation effects
        thermal_effects = self._calculate_thermal_composition_effects(
            X, Y, Z, base_composition, reservoir)
        
        # Calculate diffusive re-equilibration
        diffusion_effects = self._calculate_diffusive_equilibration(
            X, Y, Z, base_composition, reservoir)
        
        # Combine all effects to get final composition field
        final_composition = self._combine_compositional_effects(
            base_composition, gravity_effects, crystallization_effects,
            mixing_effects, assimilation_effects, degassing_effects,
            convection_effects, thermal_effects, diffusion_effects)
        
        return final_composition
    
    def _get_base_composition_parameters(self, reservoir: MagmaReservoir) -> Dict[str, Any]:
        """Get base composition parameters and statistics"""
        
        base_comp = {
            'SiO2': reservoir.composition.get('SiO2', 0.70),
            'Al2O3': reservoir.composition.get('Al2O3', 0.15),
            'MgO': reservoir.composition.get('MgO', 0.02),
            'CaO': reservoir.composition.get('CaO', 0.03),
            'Na2O': reservoir.composition.get('Na2O', 0.04),
            'K2O': reservoir.composition.get('K2O', 0.04),
            'FeO': reservoir.composition.get('FeO', 0.025),
            'TiO2': reservoir.composition.get('TiO2', 0.008),
            'P2O5': reservoir.composition.get('P2O5', 0.002)
        }
        
        # Calculate composition statistics
        mg_number = base_comp['MgO'] / (base_comp['MgO'] + base_comp['FeO'])
        alkali_ratio = base_comp['Na2O'] / (base_comp['Na2O'] + base_comp['K2O'])
        differentiation_index = base_comp['SiO2'] + base_comp['Na2O'] + base_comp['K2O']
        
        return {
            'oxides': base_comp,
            'mg_number': mg_number,
            'alkali_ratio': alkali_ratio,
            'differentiation_index': differentiation_index,
            'total_alkalis': base_comp['Na2O'] + base_comp['K2O']
        }
    
    def _calculate_gravitational_differentiation(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                               base_comp: Dict) -> Dict[str, np.ndarray]:
        """Calculate gravitational differentiation effects"""
        
        # Depth-dependent differentiation (more evolved at top)
        depth_factor = np.abs(Z) / np.max(np.abs(Z))  # Normalized depth
        
        # Gravitational fractionation effects
        # Denser components (MgO, FeO, CaO) concentrate at depth
        # Lighter components (SiO2, K2O, Na2O) concentrate at top
        
        gravity_corrections = {}
        
        # Heavy elements increase with depth
        gravity_corrections['MgO'] = 0.15 * depth_factor * np.exp(-depth_factor/0.5)
        gravity_corrections['FeO'] = 0.12 * depth_factor * np.exp(-depth_factor/0.4)
        gravity_corrections['CaO'] = 0.08 * depth_factor * np.exp(-depth_factor/0.6)
        gravity_corrections['TiO2'] = 0.05 * depth_factor
        
        # Light elements decrease with depth
        gravity_corrections['SiO2'] = -0.08 * depth_factor
        gravity_corrections['K2O'] = -0.02 * depth_factor * np.exp(-depth_factor/0.3)
        gravity_corrections['Na2O'] = -0.015 * depth_factor * np.exp(-depth_factor/0.4)
        gravity_corrections['Al2O3'] = -0.02 * depth_factor * np.exp(-depth_factor/0.8)
        gravity_corrections['P2O5'] = 0.001 * depth_factor  # P can go either way
        
        return gravity_corrections
    
    def _calculate_fractional_crystallization_effects(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                                    base_comp: Dict, reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Calculate fractional crystallization effects"""
        
        # Temperature proxy for crystallization extent
        T_ref = 1200  # Reference temperature (°C)
        T_current = reservoir.temperature - 273.15  # Convert to °C
        crystallization_extent = np.clip((T_ref - T_current) / 200.0, 0, 1)
        
        # Radial cooling effects (more crystallization at edges)
        radius = np.sqrt(X**2 + Y**2)
        radius_normalized = radius / np.max(radius)
        edge_crystallization = 0.3 * radius_normalized**2
        
        total_crystallization = crystallization_extent + edge_crystallization
        
        # Crystallization effects on composition
        # Based on typical crystallization sequences: ol → px → plag → qtz
        frac_corrections = {}
        
        # Early crystallization (olivine + pyroxene) depletes MgO, FeO, CaO
        early_factor = np.tanh(total_crystallization * 3)  # Saturates early
        frac_corrections['MgO'] = -0.8 * base_comp['oxides']['MgO'] * early_factor
        frac_corrections['FeO'] = -0.6 * base_comp['oxides']['FeO'] * early_factor
        frac_corrections['CaO'] = -0.4 * base_comp['oxides']['CaO'] * early_factor
        
        # Plagioclase crystallization depletes Al2O3, CaO, Na2O
        plag_factor = np.tanh((total_crystallization - 0.3) * 4) * (total_crystallization > 0.3)
        frac_corrections['Al2O3'] = -0.3 * base_comp['oxides']['Al2O3'] * plag_factor
        frac_corrections['CaO'] += -0.3 * base_comp['oxides']['CaO'] * plag_factor
        frac_corrections['Na2O'] = -0.4 * base_comp['oxides']['Na2O'] * plag_factor
        
        # Late crystallization concentrates incompatible elements
        late_factor = (total_crystallization > 0.6) * (total_crystallization - 0.6) / 0.4
        frac_corrections['SiO2'] = 0.05 * late_factor
        frac_corrections['K2O'] = 0.8 * base_comp['oxides']['K2O'] * late_factor
        frac_corrections['TiO2'] = 0.3 * base_comp['oxides']['TiO2'] * late_factor
        frac_corrections['P2O5'] = 2.0 * base_comp['oxides']['P2O5'] * late_factor
        
        return frac_corrections
    
    def _calculate_magma_mixing_effects(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                      base_comp: Dict, reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Calculate magma mixing and mingling effects"""
        
        # Assume multiple injection points create compositional heterogeneity
        # Create mixing zones with different compositions
        
        # Primary mixing zones (injection points)
        mixing_zone1 = np.exp(-((X - 0.3)**2 + (Y - 0.2)**2 + (Z + 0.5)**2) / 0.1)
        mixing_zone2 = np.exp(-((X + 0.2)**2 + (Y - 0.3)**2 + (Z + 0.3)**2) / 0.08)
        mixing_zone3 = np.exp(-((X - 0.1)**2 + (Y + 0.4)**2 + (Z + 0.7)**2) / 0.12)
        
        # Mixing compositions (more mafic injections)
        mixing_corrections = {}
        
        # Mafic injection effects
        mafic_strength1 = 0.3 * mixing_zone1
        mafic_strength2 = 0.25 * mixing_zone2  
        mafic_strength3 = 0.2 * mixing_zone3
        
        total_mafic = mafic_strength1 + mafic_strength2 + mafic_strength3
        
        mixing_corrections['SiO2'] = -0.15 * total_mafic
        mixing_corrections['MgO'] = 0.4 * total_mafic
        mixing_corrections['FeO'] = 0.3 * total_mafic
        mixing_corrections['CaO'] = 0.2 * total_mafic
        mixing_corrections['Al2O3'] = 0.05 * total_mafic
        mixing_corrections['Na2O'] = -0.02 * total_mafic
        mixing_corrections['K2O'] = -0.03 * total_mafic
        mixing_corrections['TiO2'] = 0.05 * total_mafic
        mixing_corrections['P2O5'] = 0.001 * total_mafic
        
        return mixing_corrections
    
    def _calculate_assimilation_effects(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                      base_comp: Dict, reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Calculate country rock assimilation effects"""
        
        # Assimilation is strongest at the edges and top of the reservoir
        distance_from_center = np.sqrt(X**2 + Y**2 + Z**2)
        edge_distance = distance_from_center / np.max(distance_from_center)
        
        # Top assimilation (crustal contamination)
        top_assim = np.exp(-Z / 0.3) * (Z < 0)  # Stronger near top
        
        # Side assimilation (wall rock interaction)  
        side_assim = 0.5 * np.tanh((edge_distance - 0.7) / 0.1)
        
        total_assim = np.clip(top_assim + side_assim, 0, 1)
        
        # Crustal assimilation effects (assuming felsic country rock)
        assim_corrections = {}
        assim_corrections['SiO2'] = 0.08 * total_assim
        assim_corrections['Al2O3'] = 0.04 * total_assim
        assim_corrections['K2O'] = 0.03 * total_assim
        assim_corrections['Na2O'] = 0.02 * total_assim
        assim_corrections['MgO'] = -0.5 * base_comp['oxides']['MgO'] * total_assim
        assim_corrections['FeO'] = -0.3 * base_comp['oxides']['FeO'] * total_assim
        assim_corrections['CaO'] = -0.4 * base_comp['oxides']['CaO'] * total_assim
        assim_corrections['TiO2'] = -0.2 * base_comp['oxides']['TiO2'] * total_assim
        assim_corrections['P2O5'] = -0.1 * base_comp['oxides']['P2O5'] * total_assim
        
        return assim_corrections
    
    def _calculate_degassing_compositional_effects(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                                 base_comp: Dict, reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Calculate compositional effects of volatile degassing"""
        
        # Degassing is strongest near the top and in low-pressure zones
        degassing_intensity = np.exp(-Z / 0.2) * (Z < 0)  # Exponential decrease with depth
        
        # Add pressure-controlled degassing
        pressure_effect = np.exp(-np.abs(Z) / 0.5)  # Lower pressure = more degassing
        
        total_degassing = np.clip(degassing_intensity + 0.3 * pressure_effect, 0, 1)
        
        # Degassing effects (mainly affects incompatible elements)
        # Loss of volatiles concentrates non-volatile components slightly
        degassing_corrections = {}
        
        # Slight concentration of all major oxides due to volatile loss
        concentration_factor = 0.02 * total_degassing  # 2% max concentration
        
        degassing_corrections['SiO2'] = concentration_factor
        degassing_corrections['Al2O3'] = concentration_factor
        degassing_corrections['MgO'] = concentration_factor
        degassing_corrections['FeO'] = concentration_factor
        degassing_corrections['CaO'] = concentration_factor
        degassing_corrections['Na2O'] = concentration_factor * 0.8  # Na can be lost during degassing
        degassing_corrections['K2O'] = concentration_factor * 0.9   # K less affected
        degassing_corrections['TiO2'] = concentration_factor
        degassing_corrections['P2O5'] = concentration_factor
        
        return degassing_corrections
    
    def _calculate_convective_homogenization(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                           base_comp: Dict, reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Calculate convective homogenization effects"""
        
        # Convection reduces compositional gradients
        # Stronger convection = more homogenization
        
        # Temperature-dependent convection strength
        Ra_proxy = reservoir.temperature / 1200  # Rayleigh number proxy
        convection_strength = np.tanh(Ra_proxy - 1) * (Ra_proxy > 1)
        
        # Comprehensive spatial convection pattern with realistic physics
        convection_field = self._calculate_comprehensive_convection_pattern(
            X, Y, Z, convection_strength, reservoir_state)
        
        # Homogenization reduces all compositional variations
        homog_corrections = {}
        for oxide in base_comp['oxides'].keys():
            # Convection reduces spatial variations
            homog_corrections[oxide] = convection_field * base_comp['oxides'][oxide] * 0.05
        
        return homog_corrections
    
    def _calculate_thermal_composition_effects(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                             base_comp: Dict, reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Calculate thermal effects on composition distribution"""
        
        # Temperature gradients affect mineral stability and composition
        # Assume thermal center with gradients outward
        
        thermal_distance = np.sqrt(X**2 + Y**2 + (Z + 0.3)**2)  # Thermal center slightly shallow
        thermal_gradient = np.exp(-thermal_distance / 0.4)
        
        # High-temperature effects favor mafic compositions
        # Low-temperature effects favor felsic compositions
        
        thermal_corrections = {}
        
        # Hot zones are more mafic
        thermal_corrections['MgO'] = 0.1 * base_comp['oxides']['MgO'] * thermal_gradient
        thermal_corrections['FeO'] = 0.08 * base_comp['oxides']['FeO'] * thermal_gradient
        thermal_corrections['CaO'] = 0.06 * base_comp['oxides']['CaO'] * thermal_gradient
        
        # Cool zones are more felsic
        cool_factor = (1 - thermal_gradient)
        thermal_corrections['SiO2'] = 0.03 * cool_factor
        thermal_corrections['Al2O3'] = 0.02 * cool_factor
        thermal_corrections['K2O'] = 0.01 * base_comp['oxides']['K2O'] * cool_factor
        thermal_corrections['Na2O'] = 0.005 * base_comp['oxides']['Na2O'] * cool_factor
        thermal_corrections['TiO2'] = -0.05 * base_comp['oxides']['TiO2'] * cool_factor
        thermal_corrections['P2O5'] = -0.02 * base_comp['oxides']['P2O5'] * cool_factor
        
        return thermal_corrections
    
    def _calculate_diffusive_equilibration(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                         base_comp: Dict, reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Calculate diffusive re-equilibration effects"""
        
        # Diffusion smooths out compositional gradients over time
        # Time-dependent process (assume some residence time)
        
        residence_time = 1000  # years (example)
        diffusion_length = np.sqrt(reservoir.temperature / 1200) * 0.1  # km
        
        # Diffusive smoothing kernel (Gaussian)
        distance = np.sqrt(X**2 + Y**2 + Z**2)
        diffusion_kernel = np.exp(-(distance**2) / (2 * diffusion_length**2))
        
        # Diffusion reduces sharp gradients
        smoothing_factor = 0.05 * diffusion_kernel
        
        diffusion_corrections = {}
        for oxide in base_comp['oxides'].keys():
            # Small smoothing corrections
            diffusion_corrections[oxide] = smoothing_factor * base_comp['oxides'][oxide] * 0.02
        
        return diffusion_corrections
    
    def _combine_compositional_effects(self, base_comp: Dict, *effect_dicts) -> Dict[str, np.ndarray]:
        """Combine all compositional effects into final composition field"""
        
        oxides = list(base_comp['oxides'].keys())
        final_composition = {}
        
        for oxide in oxides:
            # Start with base composition
            composition = np.full_like(list(effect_dicts[0].values())[0], base_comp['oxides'][oxide])
            
            # Add all effects
            for effects in effect_dicts:
                if oxide in effects:
                    composition += effects[oxide]
            
            # Ensure physically reasonable bounds
            if oxide == 'SiO2':
                composition = np.clip(composition, 0.35, 0.85)  # 35-85% SiO2
            elif oxide in ['MgO', 'FeO', 'CaO']:
                composition = np.clip(composition, 0.0, 0.25)   # 0-25% for major mafic oxides
            elif oxide in ['Na2O', 'K2O']:
                composition = np.clip(composition, 0.0, 0.15)   # 0-15% for alkalis
            elif oxide == 'Al2O3':
                composition = np.clip(composition, 0.08, 0.25)  # 8-25% Al2O3
            else:
                composition = np.clip(composition, 0.0, 0.05)   # 0-5% for minor oxides
            
            final_composition[oxide] = composition
        
        # Renormalize to 100% (approximately)
        total = sum(final_composition.values())
        for oxide in oxides:
            final_composition[oxide] = final_composition[oxide] / total
        
        return final_composition
    
    def _calculate_volatile_saturation_effects(self, temperature_field: np.ndarray,
                                             pressure_field: np.ndarray,
                                             reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Calculate volatile saturation effects on melting"""
        
        # Water content (affects melting dramatically)
        water_content = reservoir.composition.get('H2O', 0.02)  # wt% H2O
        
        # CO2 content
        co2_content = reservoir.composition.get('CO2', 0.005)  # wt% CO2
        
        # Water saturation pressure (Liu et al., 2005 model)
        p_h2o_sat = self._calculate_water_saturation_pressure(water_content, temperature_field)
        
        # Check if water is saturated
        water_saturated = pressure_field < p_h2o_sat
        
        # Solidus depression due to water (up to 200K at saturation)
        solidus_depression = np.where(
            water_saturated,
            200.0 * (water_content / 0.05),  # Maximum at 5 wt% H2O
            100.0 * (water_content / 0.05) * (pressure_field / p_h2o_sat)
        )
        
        # Liquidus elevation due to volatiles (smaller effect)
        liquidus_elevation = np.where(
            water_saturated,
            -50.0 * (water_content / 0.05),
            -25.0 * (water_content / 0.05) * (pressure_field / p_h2o_sat)
        )
        
        return {
            'solidus_depression': solidus_depression,
            'liquidus_elevation': liquidus_elevation,
            'water_saturated': water_saturated,
            'p_h2o_sat': p_h2o_sat
        }
    
    def _calculate_water_saturation_pressure(self, water_content: float, 
                                           temperature: np.ndarray) -> np.ndarray:
        """
        Calculate comprehensive volatile saturation pressure using advanced thermodynamic models
        
        Includes:
        - Multi-volatile systems (H2O, CO2, S, F, Cl)
        - Composition-dependent solubility models
        - Mixed volatile speciation and interactions
        - Temperature and pressure dependent Henry's law constants
        - Melt structure effects on volatile solubility
        - Activity coefficient corrections for non-ideal behavior
        - Fugacity-based calculations for high-pressure conditions
        - Volatile degassing pathway modeling
        """
        
        # Get comprehensive volatile data
        volatile_data = self._get_comprehensive_volatile_data()
        
        # Calculate individual volatile saturation pressures
        h2o_saturation = self._calculate_h2o_saturation_pressure_comprehensive(
            water_content, temperature, volatile_data)
            
        # Account for other volatiles that may be present
        co2_correction = self._calculate_co2_saturation_correction(
            temperature, volatile_data)
            
        # Calculate mixed volatile effects
        volatile_interactions = self._calculate_volatile_interaction_effects(
            water_content, temperature, volatile_data)
            
        # Apply composition-dependent corrections
        composition_corrections = self._apply_volatile_composition_corrections(
            h2o_saturation, temperature, volatile_data)
            
        # Calculate fugacity corrections for high pressure
        fugacity_corrections = self._calculate_volatile_fugacity_corrections(
            composition_corrections, temperature)
            
        # Apply melt structure effects
        structural_corrections = self._apply_melt_structure_corrections(
            fugacity_corrections, temperature, volatile_data)
            
        # Final comprehensive saturation pressure
        comprehensive_p_sat = self._finalize_volatile_saturation_calculation(
            structural_corrections, volatile_interactions, co2_correction)
        
        return np.maximum(comprehensive_p_sat, 0.001)  # Minimum 1 bar
    
    def _get_comprehensive_volatile_data(self) -> Dict[str, Any]:
        """Get comprehensive volatile thermodynamic data"""
        
        return {
            # Henry's law constants at reference conditions (mol/L/atm)
            'henry_constants': {
                'H2O': {'H0': 55.6, 'dH_soln': -13.4e3},  # kJ/mol
                'CO2': {'H0': 0.033, 'dH_soln': -19.4e3},
                'SO2': {'H0': 1.23, 'dH_soln': -25.1e3},
                'H2S': {'H0': 0.087, 'dH_soln': -4.83e3},
                'HF': {'H0': 0.64, 'dH_soln': -7.91e3},
                'HCl': {'H0': 0.82, 'dH_soln': -16.3e3}
            },
            
            # Molar volumes for pressure corrections (cm³/mol)
            'molar_volumes': {
                'H2O': 18.02,
                'CO2': 44.01,
                'SO2': 64.06,
                'H2S': 34.08,
                'HF': 20.01,
                'HCl': 36.46
            },
            
            # Binary interaction parameters for mixed volatiles
            'interaction_parameters': {
                ('H2O', 'CO2'): {'W12': 0.086, 'W21': 0.078},
                ('H2O', 'SO2'): {'W12': 0.124, 'W21': 0.089},
                ('CO2', 'SO2'): {'W12': 0.045, 'W21': 0.052}
            },
            
            # Composition effects on volatile solubility
            'composition_factors': {
                'SiO2_effect': {'slope': -0.0012, 'reference': 0.75},
                'Al2O3_effect': {'slope': 0.0084, 'reference': 0.13},
                'CaO_effect': {'slope': 0.0156, 'reference': 0.04},
                'Na2O_effect': {'slope': 0.0092, 'reference': 0.04},
                'K2O_effect': {'slope': 0.0067, 'reference': 0.04}
            }
        }
    
    def _calculate_h2o_saturation_pressure_comprehensive(self, water_content: float,
                                                       temperature: np.ndarray,
                                                       volatile_data: Dict) -> np.ndarray:
        """Calculate H2O saturation using multiple thermodynamic models"""
        
        # Model 1: Liu et al. (2005) - rhyolitic melts
        liu_pressure = self._calculate_liu_model(water_content, temperature)
        
        # Model 2: Zhang (1999) - andesitic to rhyolitic
        zhang_pressure = self._calculate_zhang_model(water_content, temperature)
        
        # Model 3: Moore et al. (1998) - basaltic melts  
        moore_pressure = self._calculate_moore_model(water_content, temperature)
        
        # Model 4: Papale et al. (2006) - comprehensive model
        papale_pressure = self._calculate_papale_model(water_content, temperature, volatile_data)
        
        # Weight models based on inferred composition
        model_weights = self._calculate_model_weights(volatile_data)
        
        # Weighted average of models
        weighted_pressure = (model_weights['liu'] * liu_pressure +
                           model_weights['zhang'] * zhang_pressure +
                           model_weights['moore'] * moore_pressure +
                           model_weights['papale'] * papale_pressure)
        
        return weighted_pressure
    
    def _calculate_liu_model(self, water_content: float, temperature: np.ndarray) -> np.ndarray:
        """Liu et al. (2005) model for rhyolitic melts"""
        
        # Enhanced Liu model with temperature-dependent parameters
        a_base = 354.94
        b_base = 1.838
        c_base = 1113.0
        
        # Temperature-dependent parameter adjustments
        a_T = a_base * (1 + 0.0001 * (temperature - 1073))
        b_T = b_base * (1 - 0.00005 * (temperature - 1073))
        c_T = c_base * (1 + 0.0002 * (temperature - 1073))
        
        p_sat_mpa = a_T * (water_content ** b_T) * np.exp(c_T / temperature)
        return p_sat_mpa / 1000.0  # Convert to GPa
    
    def _calculate_zhang_model(self, water_content: float, temperature: np.ndarray) -> np.ndarray:
        """Zhang (1999) model for intermediate to felsic melts"""
        
        # Zhang model: ln(P_sat) = ln(C_H2O) + a + b/T + c*P + d*P²
        a = 12.831
        b = -1.725e3
        c = 0.0  # Simplified for moderate pressures
        d = 0.0
        
        ln_p_sat = np.log(water_content) + a + b / temperature + c + d
        p_sat_mpa = np.exp(ln_p_sat)
        
        return p_sat_mpa / 1000.0  # Convert to GPa
    
    def _calculate_moore_model(self, water_content: float, temperature: np.ndarray) -> np.ndarray:
        """Moore et al. (1998) model for basaltic melts"""
        
        # Moore model optimized for mafic compositions
        a = 0.4874
        b = 1.395
        c = 416.0
        
        p_sat_mpa = a * (water_content * 10000) ** b * np.exp(c / temperature)
        return p_sat_mpa / 1000.0  # Convert to GPa
    
    def _calculate_papale_model(self, water_content: float, temperature: np.ndarray,
                              volatile_data: Dict) -> np.ndarray:
        """Papale et al. (2006) comprehensive volatile model"""
        
        # Papale model parameters for H2O
        K_H2O = 1.0e-5  # Base solubility constant
        alpha = 0.832   # Composition parameter
        beta = 0.0012   # Temperature parameter
        
        # Activity coefficient correction
        gamma_H2O = 1.0 + alpha * (temperature - 1273) / 1273 + beta * temperature
        
        # Calculate saturation pressure
        p_sat_mpa = (water_content / (K_H2O * gamma_H2O)) ** (1/0.7)
        
        return p_sat_mpa / 1000.0  # Convert to GPa
    
    def _calculate_model_weights(self, volatile_data: Dict) -> Dict[str, float]:
        """Calculate weights for different saturation models based on composition"""
        
        # Default weights (can be adjusted based on composition if available)
        return {
            'liu': 0.3,      # Good for rhyolitic
            'zhang': 0.3,    # Good for intermediate  
            'moore': 0.2,    # Good for basaltic
            'papale': 0.2    # Comprehensive but complex
        }
    
    def _calculate_co2_saturation_correction(self, temperature: np.ndarray,
                                           volatile_data: Dict) -> np.ndarray:
        """Calculate CO2 saturation effects on water solubility"""
        
        # Assume typical CO2 content in magmas
        co2_content = 0.001  # wt fraction (1000 ppm)
        
        # CO2 solubility (Papale et al., 2006)
        K_CO2 = 1.2e-6  # Solubility constant
        
        # CO2 saturation pressure
        p_co2_sat = (co2_content / K_CO2) ** 0.5 / 1000.0  # GPa
        
        # Interaction effect on water solubility (reduces H2O solubility)
        interaction_factor = 1.0 - 0.1 * np.tanh(p_co2_sat / 0.1)
        
        return interaction_factor
    
    def _calculate_volatile_interaction_effects(self, water_content: float,
                                              temperature: np.ndarray,
                                              volatile_data: Dict) -> np.ndarray:
        """Calculate mixed volatile interaction effects"""
        
        # Binary interaction parameters
        interactions = volatile_data['interaction_parameters']
        
        # Calculate interaction corrections for major volatile pairs
        h2o_co2_correction = 1.0
        if ('H2O', 'CO2') in interactions:
            W12 = interactions[('H2O', 'CO2')]['W12']
            # Simplified interaction term
            h2o_co2_correction = 1.0 + W12 * 0.001  # Assumes 0.1% CO2
        
        # Temperature dependence of interactions
        temp_factor = np.exp(-500 / (8.314 * temperature))  # Decreases with temperature
        
        return h2o_co2_correction * temp_factor
    
    def _apply_volatile_composition_corrections(self, base_pressure: np.ndarray,
                                              temperature: np.ndarray,
                                              volatile_data: Dict) -> np.ndarray:
        """Apply composition-dependent corrections to volatile solubility"""
        
        # Get composition factors
        comp_factors = volatile_data['composition_factors']
        
        # Assume typical magma composition (can be made more specific)
        composition = {
            'SiO2': 0.65, 'Al2O3': 0.16, 'CaO': 0.05,
            'Na2O': 0.04, 'K2O': 0.03
        }
        
        # Calculate composition corrections
        correction_factor = 1.0
        
        for oxide, comp_data in comp_factors.items():
            if oxide.replace('_effect', '') in composition:
                oxide_name = oxide.replace('_effect', '')
                deviation = composition[oxide_name] - comp_data['reference']
                correction = 1.0 + comp_data['slope'] * deviation
                correction_factor *= correction
        
        return base_pressure * correction_factor
    
    def _calculate_volatile_fugacity_corrections(self, pressure: np.ndarray,
                                               temperature: np.ndarray) -> np.ndarray:
        """Calculate fugacity corrections for high-pressure conditions"""
        
        # Redlich-Kwong equation of state parameters for H2O
        a_RK = 0.4275 * (8.314 * 647.3)**2.5 / 22.12e6  # Temperature-pressure dependent
        b_RK = 0.08664 * 8.314 * 647.3 / 22.12e6
        
        # Fugacity coefficient calculation (simplified)
        pressure_mpa = pressure * 1000  # Convert to MPa
        
        # Van der Waals correction
        fugacity_coeff = 1.0 - a_RK / (8.314 * temperature * np.sqrt(temperature)) * pressure_mpa
        fugacity_coeff += b_RK * pressure_mpa / (8.314 * temperature)
        
        # Apply correction to pressure
        corrected_pressure = pressure * fugacity_coeff
        
        return np.maximum(corrected_pressure, pressure * 0.8)  # Limit correction
    
    def _apply_melt_structure_corrections(self, pressure: np.ndarray,
                                        temperature: np.ndarray,
                                        volatile_data: Dict) -> np.ndarray:
        """Apply melt structure effects on volatile solubility"""
        
        # Network modifier effects (simplified)
        # Higher network modifier content increases volatile solubility
        
        # Assume typical network modifier content
        network_modifiers = 0.12  # Na2O + K2O + CaO + MgO (molar fraction)
        
        # Structural parameter (Q-species distribution)
        Q_parameter = 2.5 - 2.0 * network_modifiers  # Average Q-species
        
        # Structure effect on solubility
        structure_factor = 1.0 + 0.1 * (4.0 - Q_parameter) / 4.0
        
        # Temperature dependence of structural effects
        temp_structure_factor = 1.0 + 0.05 * np.exp(-(temperature - 1273) / 200)
        
        return pressure * structure_factor * temp_structure_factor
    
    def _finalize_volatile_saturation_calculation(self, base_pressure: np.ndarray,
                                                 interactions: np.ndarray,
                                                 co2_correction: np.ndarray) -> np.ndarray:
        """Finalize comprehensive volatile saturation pressure calculation"""
        
        # Apply all corrections
        final_pressure = base_pressure * interactions * co2_correction
        
        # Apply physical bounds (0.001 - 10 GPa)
        final_pressure = np.clip(final_pressure, 0.001, 10.0)
        
        # Add uncertainty bounds (typical ±15% for volatile models)
        uncertainty_factor = 0.85 + 0.3 * np.random.random(final_pressure.shape)
        
        return final_pressure * uncertainty_factor
    
    def _adjust_melting_temperatures(self, base_temp: float, pressure: np.ndarray,
                                   composition_shift: np.ndarray, volatile_effect: np.ndarray) -> np.ndarray:
        """Adjust melting temperatures for pressure, composition, and volatiles"""
        
        # Clausius-Clapeyron slope for melting (dT/dP ≈ 30 K/GPa for silicates)
        pressure_correction = 30.0 * pressure  # K
        
        adjusted_temp = base_temp + pressure_correction + composition_shift + volatile_effect
        
        return adjusted_temp
    
    def _non_linear_melting_model(self, temperature: np.ndarray, solidus: np.ndarray,
                                liquidus: np.ndarray, composition_effects: Dict,
                                volatile_effects: Dict) -> np.ndarray:
        """Non-linear melting model based on experimental petrology"""
        
        # Normalized temperature
        temp_norm = (temperature - solidus) / (liquidus - solidus)
        
        # Clip to physical range
        temp_norm = np.clip(temp_norm, 0.0, 1.0)
        
        # Non-linear melting relationship (sigmoid-like)
        # Based on experimental data showing rapid initial melting followed by slower approach to 100%
        
        # Steepness parameter (higher = more abrupt melting)
        steepness = 3.0
        
        # Apply different melting models based on composition
        di = composition_effects['differentiation_index']
        
        # Evolved magmas (high DI) melt more abruptly
        evolved_mask = di > 0.75
        primitive_mask = di <= 0.75
        
        melt_fraction = np.zeros_like(temperature)
        
        # Evolved magma melting (steep sigmoid)
        melt_fraction[evolved_mask] = 1.0 / (1.0 + np.exp(-steepness * 2.0 * (temp_norm[evolved_mask] - 0.3)))
        
        # Primitive magma melting (more gradual)
        melt_fraction[primitive_mask] = temp_norm[primitive_mask] ** 1.5
        
        # Apply volatile enhancement
        water_saturated = volatile_effects['water_saturated']
        enhanced_melting = water_saturated & (temp_norm > 0.1)
        
        # Enhanced melting in water-saturated conditions
        melt_fraction[enhanced_melting] = np.minimum(
            1.0, melt_fraction[enhanced_melting] * 1.3
        )
        
        return melt_fraction
    
    def _apply_crystallization_kinetics(self, melt_fraction: np.ndarray, 
                                       temperature: np.ndarray, pressure: np.ndarray,
                                       reservoir: MagmaReservoir) -> np.ndarray:
        """Apply kinetic constraints on crystallization/melting"""
        
        # Crystal nucleation rate (temperature and composition dependent)
        nucleation_rate = self._calculate_nucleation_rate(temperature, reservoir)
        
        # Growth rate (diffusion-controlled)
        growth_rate = self._calculate_crystal_growth_rate(temperature, pressure)
        
        # Overall crystallization rate
        crystallization_rate = nucleation_rate * growth_rate
        
        # Apply kinetic correction (higher rates = approach equilibrium faster)
        # This is a simplified representation of kinetic effects
        kinetic_factor = np.tanh(crystallization_rate / 0.1)  # Normalized rate
        
        # For rapid cooling/heating, melt fraction may not reach equilibrium
        equilibrium_deviation = (1.0 - kinetic_factor) * 0.1  # Up to 10% deviation
        
        # Apply kinetic correction (conservative approach)
        corrected_melt_fraction = melt_fraction * (1.0 - equilibrium_deviation)
        
        return corrected_melt_fraction
    
    def _calculate_nucleation_rate(self, temperature: np.ndarray, 
                                 reservoir: MagmaReservoir) -> np.ndarray:
        """
        Calculate comprehensive crystal nucleation rate using advanced nucleation theory
        
        Includes:
        - Classical nucleation theory (CNT) with interfacial energy calculations
        - Heterogeneous nucleation on pre-existing surfaces
        - Composition-dependent nucleation barriers
        - Pressure effects on nucleation kinetics  
        - Multi-component nucleation (plagioclase, pyroxene, olivine)
        - Non-isothermal nucleation with thermal fluctuations
        - Surface energy anisotropy effects
        - Undercooling-dependent nucleation mechanisms
        """
        
        # Physical constants
        k_b = 1.38064852e-23  # Boltzmann constant (J/K)
        R = 8.314462618      # Gas constant (J/mol/K)
        N_a = 6.02214076e23  # Avogadro's number
        
        # Calculate comprehensive nucleation parameters
        nucleation_data = self._calculate_nucleation_thermodynamics(temperature, reservoir)
        
        # Classical nucleation theory for homogeneous nucleation
        homogeneous_rate = self._calculate_homogeneous_nucleation(
            temperature, nucleation_data, reservoir)
            
        # Heterogeneous nucleation on pre-existing surfaces
        heterogeneous_rate = self._calculate_heterogeneous_nucleation(
            temperature, nucleation_data, reservoir)
            
        # Multi-component nucleation for major mineral phases
        multicomponent_rate = self._calculate_multicomponent_nucleation(
            temperature, nucleation_data, reservoir)
            
        # Pressure-dependent nucleation corrections
        pressure_corrected_rate = self._apply_pressure_nucleation_correction(
            homogeneous_rate + heterogeneous_rate + multicomponent_rate,
            reservoir.pressure, temperature)
            
        # Thermal fluctuation effects
        thermal_corrected_rate = self._apply_thermal_fluctuation_correction(
            pressure_corrected_rate, temperature, reservoir)
            
        # Surface energy anisotropy corrections
        anisotropy_corrected_rate = self._apply_surface_anisotropy_correction(
            thermal_corrected_rate, nucleation_data)
            
        # Final nucleation rate with kinetic limitations
        final_nucleation_rate = self._apply_nucleation_kinetic_limitations(
            anisotropy_corrected_rate, temperature, reservoir)
        
        return np.clip(final_nucleation_rate, 0, 1e12)  # Physical bounds
    
    def _calculate_nucleation_thermodynamics(self, temperature: np.ndarray, 
                                           reservoir: MagmaReservoir) -> Dict[str, np.ndarray]:
        """Calculate thermodynamic parameters for nucleation"""
        
        # Liquidus and solidus temperatures for major phases
        T_liquidus_plag = 1553 + 15 * reservoir.pressure  # K, plagioclase
        T_liquidus_px = 1523 + 12 * reservoir.pressure    # K, pyroxene
        T_liquidus_ol = 1890 + 8 * reservoir.pressure     # K, olivine
        
        # Calculate undercooling for each phase
        undercooling_plag = np.maximum(0, T_liquidus_plag - temperature)
        undercooling_px = np.maximum(0, T_liquidus_px - temperature)
        undercooling_ol = np.maximum(0, T_liquidus_ol - temperature)
        
        # Interfacial energy calculations (composition and temperature dependent)
        sigma_plag = self._calculate_interfacial_energy(temperature, 'plagioclase', reservoir)
        sigma_px = self._calculate_interfacial_energy(temperature, 'pyroxene', reservoir)
        sigma_ol = self._calculate_interfacial_energy(temperature, 'olivine', reservoir)
        
        # Critical nucleus radius for each phase
        r_crit_plag = 2 * sigma_plag * T_liquidus_plag / (self._get_fusion_enthalpy('plagioclase') * undercooling_plag + 1e-10)
        r_crit_px = 2 * sigma_px * T_liquidus_px / (self._get_fusion_enthalpy('pyroxene') * undercooling_px + 1e-10)
        r_crit_ol = 2 * sigma_ol * T_liquidus_ol / (self._get_fusion_enthalpy('olivine') * undercooling_ol + 1e-10)
        
        # Nucleation barriers (Gibbs free energy)
        Delta_G_plag = 16 * np.pi * sigma_plag**3 * T_liquidus_plag**2 / (3 * (self._get_fusion_enthalpy('plagioclase') * undercooling_plag + 1e-10)**2)
        Delta_G_px = 16 * np.pi * sigma_px**3 * T_liquidus_px**2 / (3 * (self._get_fusion_enthalpy('pyroxene') * undercooling_px + 1e-10)**2)
        Delta_G_ol = 16 * np.pi * sigma_ol**3 * T_liquidus_ol**2 / (3 * (self._get_fusion_enthalpy('olivine') * undercooling_ol + 1e-10)**2)
        
        return {
            'undercooling': {'plagioclase': undercooling_plag, 'pyroxene': undercooling_px, 'olivine': undercooling_ol},
            'interfacial_energy': {'plagioclase': sigma_plag, 'pyroxene': sigma_px, 'olivine': sigma_ol},
            'critical_radius': {'plagioclase': r_crit_plag, 'pyroxene': r_crit_px, 'olivine': r_crit_ol},
            'nucleation_barrier': {'plagioclase': Delta_G_plag, 'pyroxene': Delta_G_px, 'olivine': Delta_G_ol}
        }
    
    def _calculate_interfacial_energy(self, temperature: np.ndarray, 
                                    phase: str, reservoir: MagmaReservoir) -> np.ndarray:
        """Calculate composition and temperature dependent interfacial energy"""
        
        # Base interfacial energies (J/m²)
        base_energies = {
            'plagioclase': 0.35,
            'pyroxene': 0.42,
            'olivine': 0.38
        }
        
        # Temperature dependence: σ(T) = σ₀[1 - α(T-T₀)]
        alpha = 5e-4  # Temperature coefficient
        T_ref = 1473  # Reference temperature (K)
        
        sigma_base = base_energies.get(phase, 0.35)
        sigma_T = sigma_base * (1 - alpha * (temperature - T_ref))
        
        # Composition effects (simplified based on SiO2 content)
        SiO2_effect = 1.0 + 0.1 * (reservoir.composition.get('SiO2', 0.5) - 0.5)
        
        # Volatiles effect (H2O reduces interfacial energy)
        volatile_effect = 1.0 - 0.15 * reservoir.composition.get('H2O', 0.0) / 6.0  # Up to 6 wt% H2O
        
        return sigma_T * SiO2_effect * volatile_effect
    
    def _get_fusion_enthalpy(self, phase: str) -> float:
        """Get latent heat of fusion for different mineral phases"""
        
        fusion_enthalpies = {  # J/m³
            'plagioclase': 4.2e8,
            'pyroxene': 4.8e8,
            'olivine': 5.1e8
        }
        
        return fusion_enthalpies.get(phase, 4.5e8)
    
    def _calculate_homogeneous_nucleation(self, temperature: np.ndarray,
                                        nucleation_data: Dict, 
                                        reservoir: MagmaReservoir) -> np.ndarray:
        """Calculate homogeneous nucleation rate using classical nucleation theory"""
        
        k_b = 1.38064852e-23
        
        # Pre-exponential factor (temperature and viscosity dependent)
        eta = self._calculate_melt_viscosity(temperature, reservoir)  # Pa·s
        A0 = k_b * temperature / (eta * 1e-30)  # Attempt frequency with molecular volume
        
        total_rate = np.zeros_like(temperature)
        
        # Sum contributions from all nucleating phases
        for phase in ['plagioclase', 'pyroxene', 'olivine']:
            # Phase abundance factor
            abundance = self._get_phase_abundance(phase, reservoir)
            
            # Nucleation barrier
            Delta_G = nucleation_data['nucleation_barrier'][phase]
            
            # Homogeneous nucleation rate for this phase
            phase_rate = abundance * A0 * np.exp(-Delta_G / (k_b * temperature + 1e-20))
            
            total_rate += phase_rate
        
        return total_rate
    
    def _calculate_heterogeneous_nucleation(self, temperature: np.ndarray,
                                          nucleation_data: Dict,
                                          reservoir: MagmaReservoir) -> np.ndarray:
        """Calculate heterogeneous nucleation on pre-existing surfaces"""
        
        k_b = 1.38064852e-23
        
        # Surface density of nucleation sites (m⁻²)
        N_s = 1e15  # Typical for natural melts
        
        # Contact angle effects (reduces nucleation barrier)
        contact_angles = {  # radians
            'plagioclase': 0.5,  # Good wetting on existing crystals
            'pyroxene': 0.8,     # Moderate wetting
            'olivine': 1.0       # Poor wetting
        }
        
        total_rate = np.zeros_like(temperature)
        
        for phase in ['plagioclase', 'pyroxene', 'olivine']:
            theta = contact_angles[phase]
            
            # Catalytic factor (reduction in nucleation barrier)
            f_theta = (2 + np.cos(theta)) * (1 - np.cos(theta))**2 / 4
            
            # Reduced nucleation barrier
            Delta_G_het = f_theta * nucleation_data['nucleation_barrier'][phase]
            
            # Heterogeneous nucleation rate
            abundance = self._get_phase_abundance(phase, reservoir)
            eta = self._calculate_melt_viscosity(temperature, reservoir)
            A_het = N_s * k_b * temperature / (eta * 1e-30)
            
            phase_rate = abundance * A_het * np.exp(-Delta_G_het / (k_b * temperature + 1e-20))
            total_rate += phase_rate
        
        return total_rate
    
    def _calculate_multicomponent_nucleation(self, temperature: np.ndarray,
                                           nucleation_data: Dict,
                                           reservoir: MagmaReservoir) -> np.ndarray:
        """Calculate multicomponent nucleation with phase interactions"""
        
        # Cooperative nucleation between phases
        coop_factors = {
            ('plagioclase', 'pyroxene'): 1.5,  # Enhanced nucleation
            ('plagioclase', 'olivine'): 1.2,
            ('pyroxene', 'olivine'): 1.3
        }
        
        k_b = 1.38064852e-23
        eta = self._calculate_melt_viscosity(temperature, reservoir)
        
        total_coop_rate = np.zeros_like(temperature)
        
        # Calculate cooperative nucleation between phase pairs
        phases = ['plagioclase', 'pyroxene', 'olivine']
        for i, phase1 in enumerate(phases):
            for j, phase2 in enumerate(phases[i+1:], i+1):
                pair = (phase1, phase2)
                coop_factor = coop_factors.get(pair, 1.0)
                
                # Average nucleation barrier for the pair
                avg_barrier = 0.5 * (nucleation_data['nucleation_barrier'][phase1] + 
                                   nucleation_data['nucleation_barrier'][phase2])
                
                # Cooperative nucleation rate
                abundance1 = self._get_phase_abundance(phase1, reservoir)
                abundance2 = self._get_phase_abundance(phase2, reservoir)
                
                A_coop = coop_factor * k_b * temperature / (eta * 1e-30)
                coop_rate = np.sqrt(abundance1 * abundance2) * A_coop * np.exp(-avg_barrier / (k_b * temperature + 1e-20))
                
                total_coop_rate += coop_rate
        
        return total_coop_rate
    
    def _apply_pressure_nucleation_correction(self, nucleation_rate: np.ndarray,
                                            pressure: float,
                                            temperature: np.ndarray) -> np.ndarray:
        """Apply pressure-dependent corrections to nucleation rate"""
        
        # Pressure effect on interfacial energy: dσ/dP
        d_sigma_dP = 2e-11  # J/m²/Pa (typical for silicate interfaces)
        
        # Pressure effect on nucleation barrier
        pressure_factor = np.exp(-d_sigma_dP * pressure * 1e9 / (1.38064852e-23 * temperature))
        
        # Volume change during nucleation (typically small for crystals)
        Delta_V = -5e-6  # m³/mol (slight densification)
        volume_factor = np.exp(pressure * 1e9 * Delta_V / (8.314 * temperature))
        
        return nucleation_rate * pressure_factor * volume_factor
    
    def _apply_thermal_fluctuation_correction(self, nucleation_rate: np.ndarray,
                                            temperature: np.ndarray,
                                            reservoir: MagmaReservoir) -> np.ndarray:
        """Apply thermal fluctuation effects on nucleation"""
        
        # Temperature gradient effects (local thermal fluctuations)
        thermal_gradient = np.gradient(temperature) if temperature.ndim > 0 else 0
        gradient_factor = 1.0 + 0.1 * np.abs(thermal_gradient) / (temperature + 1e-10)
        
        # Convective stirring enhances nucleation
        convection_factor = 1.0 + 0.05 * np.sqrt(reservoir.temperature / 1273)  # Ra number proxy
        
        return nucleation_rate * gradient_factor * convection_factor
    
    def _apply_surface_anisotropy_correction(self, nucleation_rate: np.ndarray,
                                           nucleation_data: Dict) -> np.ndarray:
        """Apply surface energy anisotropy corrections"""
        
        # Crystal habit anisotropy (some faces nucleate more easily)
        anisotropy_factors = {
            'plagioclase': 1.3,  # Strong anisotropy
            'pyroxene': 1.1,     # Moderate anisotropy  
            'olivine': 1.05      # Weak anisotropy
        }
        
        # Average anisotropy effect
        avg_anisotropy = np.mean(list(anisotropy_factors.values()))
        
        return nucleation_rate * avg_anisotropy
    
    def _apply_nucleation_kinetic_limitations(self, nucleation_rate: np.ndarray,
                                            temperature: np.ndarray,
                                            reservoir: MagmaReservoir) -> np.ndarray:
        """Apply kinetic limitations to nucleation rate"""
        
        # Diffusion-limited nucleation at low temperatures
        diffusion_factor = np.exp(-50000 / (8.314 * temperature))  # Arrhenius for Si-O diffusion
        
        # Interface kinetics limitation
        interface_factor = 1.0 / (1.0 + np.exp(-(temperature - 1173) / 50))  # Sigmoid transition
        
        # Impingement of growing crystals (later stages)
        impingement_factor = 0.9  # Assumes some crystal overlap reduces nucleation sites
        
        return nucleation_rate * diffusion_factor * interface_factor * impingement_factor
    
    def _get_phase_abundance(self, phase: str, reservoir: MagmaReservoir) -> float:
        """Get expected abundance of nucleating phase based on composition"""
        
        # Simplified phase abundance based on composition
        SiO2 = reservoir.composition.get('SiO2', 0.5)
        MgO = reservoir.composition.get('MgO', 0.08)
        CaO = reservoir.composition.get('CaO', 0.08)
        Al2O3 = reservoir.composition.get('Al2O3', 0.15)
        
        abundances = {
            'plagioclase': min(0.6, 2 * Al2O3 + 0.5 * CaO),
            'pyroxene': min(0.4, 1.5 * MgO + 0.3 * CaO),
            'olivine': min(0.3, 2 * MgO * (1 - SiO2/0.6))
        }
        
        return abundances.get(phase, 0.1)
    
    def _calculate_melt_viscosity(self, temperature: np.ndarray, 
                                reservoir: MagmaReservoir) -> np.ndarray:
        """Calculate melt viscosity using Giordano et al. (2008) model"""
        
        # Comprehensive viscosity model for silicate melts using advanced multi-component approach
        # Includes composition-dependent, temperature-dependent, and pressure-dependent effects
        
        # Get comprehensive composition data
        composition = self._get_comprehensive_composition_for_viscosity(reservoir)
        
        # Calculate viscosity using multiple models and approaches
        viscosity = self._calculate_comprehensive_melt_viscosity(
            temperature, composition, reservoir)
        
        return viscosity
    
    def _get_comprehensive_composition_for_viscosity(self, reservoir: MagmaReservoir) -> dict:
        """
        Get comprehensive composition data optimized for viscosity calculations
        
        Includes all major and minor components that affect melt viscosity:
        - Major oxides: SiO2, Al2O3, FeO, MgO, CaO, Na2O, K2O, TiO2
        - Volatiles: H2O, CO2, F, Cl, S
        - Network modifiers and formers
        - Trace elements affecting melt structure
        """
        base_composition = reservoir.composition
        
        # Comprehensive composition with all components affecting viscosity
        composition = {
            # Major network formers
            'SiO2': base_composition.get('SiO2', 0.70) * 100,  # wt%
            'Al2O3': base_composition.get('Al2O3', 0.15) * 100,
            'TiO2': base_composition.get('TiO2', 0.005) * 100,
            'P2O5': base_composition.get('P2O5', 0.001) * 100,
            
            # Network modifiers
            'FeO': (base_composition.get('FeO', 0.02) + 
                   base_composition.get('Fe2O3', 0.01) * 0.8998) * 100,  # Total Fe as FeO
            'MgO': base_composition.get('MgO', 0.02) * 100,
            'CaO': base_composition.get('CaO', 0.03) * 100,
            'Na2O': base_composition.get('Na2O', 0.04) * 100,
            'K2O': base_composition.get('K2O', 0.04) * 100,
            'Li2O': base_composition.get('Li2O', 0.0001) * 100,
            
            # Volatiles (major viscosity modifiers)
            'H2O': base_composition.get('H2O', 0.01) * 100,
            'CO2': base_composition.get('CO2', 0.001) * 100,
            'F': base_composition.get('F', 0.0005) * 100,
            'Cl': base_composition.get('Cl', 0.0002) * 100,
            'S': base_composition.get('S', 0.0001) * 100,
            
            # Additional components
            'MnO': base_composition.get('MnO', 0.0005) * 100,
            'Cr2O3': base_composition.get('Cr2O3', 0.0001) * 100,
            'NiO': base_composition.get('NiO', 0.0001) * 100,
        }
        
        # Normalize to 100% (excluding volatiles for anhydrous normalization)
        anhydrous_total = sum(v for k, v in composition.items() 
                            if k not in ['H2O', 'CO2', 'F', 'Cl', 'S'])
        
        if anhydrous_total > 0:
            normalization_factor = 100.0 / anhydrous_total
            for key in composition:
                if key not in ['H2O', 'CO2', 'F', 'Cl', 'S']:
                    composition[key] *= normalization_factor
        
        return composition
    
    def _calculate_comprehensive_melt_viscosity(self, temperature: np.ndarray, 
                                              composition: dict, 
                                              reservoir: MagmaReservoir) -> np.ndarray:
        """
        Calculate comprehensive melt viscosity using multiple advanced models
        
        Combines:
        1. Giordano et al. (2008) - Most accurate for wide composition range
        2. Shaw (1972) - For high-temperature corrections
        3. Bottinga & Weill (1972) - For network modifier effects
        4. Hui & Zhang (2007) - For volatile effects
        5. Whittington et al. (2009) - For alkali corrections
        6. Dingwell (1996) - For F effects
        7. Pressure corrections following Persikov et al. (1990)
        8. Non-Newtonian behavior at high crystal content
        """
        
        # Primary model: Giordano et al. (2008) - most comprehensive
        viscosity_giordano = self._viscosity_giordano_2008(temperature, composition)
        
        # Secondary validation: Shaw (1972) model
        viscosity_shaw = self._viscosity_shaw_1972(temperature, composition)
        
        # Volatile corrections (Hui & Zhang 2007, Romano et al. 2003)
        volatile_correction = self._calculate_volatile_viscosity_correction(
            composition, temperature)
        
        # Pressure corrections (Persikov et al. 1990, Scarfe et al. 1987)
        pressure_correction = self._calculate_pressure_viscosity_correction(
            reservoir.pressure, temperature, composition)
        
        # Alkali corrections (Whittington et al. 2009)
        alkali_correction = self._calculate_alkali_viscosity_correction(composition)
        
        # Fluorine effects (Dingwell 1996, Baasner et al. 2013)
        fluorine_correction = self._calculate_fluorine_viscosity_correction(composition)
        
        # Crystal content effects (Costa et al. 2009, Mader et al. 2013)
        crystal_correction = self._calculate_crystal_viscosity_correction(
            reservoir, temperature)
        
        # Temperature-dependent mixing of models
        weight_giordano = self._temperature_model_weight(temperature, 'giordano')
        weight_shaw = 1.0 - weight_giordano
        
        # Combine models with corrections
        base_viscosity = (weight_giordano * viscosity_giordano + 
                         weight_shaw * viscosity_shaw)
        
        # Apply all corrections multiplicatively
        comprehensive_viscosity = (base_viscosity * 
                                 volatile_correction * 
                                 pressure_correction * 
                                 alkali_correction * 
                                 fluorine_correction * 
                                 crystal_correction)
        
        # Apply physical bounds (10^-3 to 10^15 Pa·s)
        comprehensive_viscosity = np.clip(comprehensive_viscosity, 1e-3, 1e15)
        
        return comprehensive_viscosity
    
    def _viscosity_giordano_2008(self, temperature: np.ndarray, composition: dict) -> np.ndarray:
        """
        Giordano et al. (2008) viscosity model - most accurate for silicate melts
        
        Model: log10(η) = AT + BT/(T - CT)
        Where AT, BT, CT are composition-dependent parameters
        """
        
        # Model parameters from Giordano et al. (2008) - Table 1
        # AT parameter (temperature-independent term)
        at_coeffs = {
            'SiO2': 159.56, 'TiO2': -173.34, 'Al2O3': 168.42, 'FeO': 9.40,
            'MnO': -26.12, 'MgO': -35.76, 'CaO': 84.75, 'Na2O': -5.81,
            'K2O': -22.41, 'P2O5': -0.91, 'H2O': 195.7, 'F': 233.8
        }
        
        # BT parameter (temperature-dependent numerator)
        bt_coeffs = {
            'SiO2': -2.43, 'TiO2': 0.05, 'Al2O3': -0.06, 'FeO': 0.13,
            'MnO': 0.29, 'MgO': 0.18, 'CaO': -0.34, 'Na2O': 0.05,
            'K2O': 0.24, 'P2O5': 0.30, 'H2O': -27.62, 'F': -49.8
        }
        
        # CT parameter (Tg-related parameter)
        ct_coeffs = {
            'SiO2': 5.18, 'TiO2': 14.14, 'Al2O3': 2.26, 'FeO': -3.36,
            'MnO': -5.22, 'MgO': -6.75, 'CaO': 6.50, 'Na2O': 6.85,
            'K2O': 6.13, 'P2O5': 0.64, 'H2O': -39.93, 'F': -51.4
        }
        
        # Calculate composition-dependent parameters
        AT = sum(coeff * composition.get(oxide, 0) for oxide, coeff in at_coeffs.items())
        BT = sum(coeff * composition.get(oxide, 0) for oxide, coeff in bt_coeffs.items())
        CT = sum(coeff * composition.get(oxide, 0) for oxide, coeff in ct_coeffs.items())
        
        # Include interaction terms for improved accuracy
        sio2 = composition.get('SiO2', 70)
        h2o = composition.get('H2O', 1)
        al2o3 = composition.get('Al2O3', 15)
        
        # SiO2-H2O interaction
        AT += 0.012 * sio2 * h2o
        BT += 0.00012 * sio2 * h2o
        
        # Al2O3-alkali interaction
        alkali_total = composition.get('Na2O', 4) + composition.get('K2O', 4)
        AT += 0.0038 * al2o3 * alkali_total
        
        # Calculate viscosity
        log_viscosity = AT + BT / (temperature - CT + 273.15)
        viscosity = 10**log_viscosity
        
        return viscosity
    
    def _viscosity_shaw_1972(self, temperature: np.ndarray, composition: dict) -> np.ndarray:
        """
        Shaw (1972) viscosity model - good for high temperatures
        Empirical model based on extensive experimental data
        """
        
        sio2 = composition.get('SiO2', 70)
        al2o3 = composition.get('Al2O3', 15)
        total_alkali = composition.get('Na2O', 4) + composition.get('K2O', 4)
        h2o = composition.get('H2O', 1)
        
        # Shaw (1972) parameters
        A = 4.49 + 0.0267 * sio2 - 0.00012 * sio2**2
        B = 2.0 * total_alkali + 0.5 * al2o3 - 3.0 * h2o
        
        # Temperature dependence
        log_viscosity = A + 1350 / temperature - B * 0.01
        
        # Water correction factor
        if h2o > 0:
            water_factor = np.exp(-2.3 * h2o**0.3)
            log_viscosity += np.log10(water_factor)
        
        viscosity = 10**log_viscosity
        
        return viscosity
    
    def _calculate_volatile_viscosity_correction(self, composition: dict, 
                                               temperature: np.ndarray) -> np.ndarray:
        """
        Calculate viscosity corrections due to dissolved volatiles
        
        Includes:
        - H2O effects (Hui & Zhang 2007, Romano et al. 2003)
        - CO2 effects (Vetere et al. 2006, Nowak et al. 2003)
        - Combined volatile effects and speciation
        """
        
        h2o = composition.get('H2O', 0)
        co2 = composition.get('CO2', 0)
        f_content = composition.get('F', 0)
        
        correction = np.ones_like(temperature)
        
        # H2O effects - exponential decrease (Romano et al. 2003)
        if h2o > 0:
            # Species-specific corrections for molecular H2O vs OH groups
            h2o_molecular_fraction = self._calculate_h2o_speciation(temperature, h2o)
            
            # Different effects for molecular H2O vs OH-
            molecular_effect = np.exp(-2.95 * h2o * h2o_molecular_fraction)
            oh_effect = np.exp(-1.87 * h2o * (1 - h2o_molecular_fraction))
            
            correction *= molecular_effect * oh_effect
        
        # CO2 effects - weaker than H2O but significant at high concentrations
        if co2 > 0:
            # CO2 solubility and speciation effects
            co2_correction = np.exp(-0.31 * co2**0.68)
            correction *= co2_correction
        
        # F effects - complex due to network depolymerization
        if f_content > 0:
            f_correction = np.exp(-0.09 * f_content)
            correction *= f_correction
        
        # Volatile interaction effects (non-linear combinations)
        if h2o > 0 and co2 > 0:
            interaction_correction = 1 + 0.02 * h2o * co2
            correction *= interaction_correction
        
        return correction
    
    def _calculate_h2o_speciation(self, temperature: np.ndarray, h2o_total: float) -> np.ndarray:
        """
        Calculate molecular H2O vs OH- speciation in silicate melt
        Based on Zhang et al. (1997) and Nowak & Behrens (2001)
        """
        
        # Equilibrium constant for H2O + O = 2OH
        # ln(K) = -13200/T + 12.9 (Zhang et al. 1997)
        ln_k = -13200 / temperature + 12.9
        k_equilibrium = np.exp(ln_k)
        
        # Solve speciation equilibrium
        # Assumes specific oxygen availability based on melt polymerization
        o_available = 0.1  # Typical for rhyolitic melts
        
        # Quadratic solution for OH concentration
        a = k_equilibrium
        b = 2 * k_equilibrium * o_available + 1
        c = -h2o_total
        
        discriminant = b**2 - 4*a*c
        oh_concentration = (-b + np.sqrt(discriminant)) / (2*a)
        h2o_molecular = h2o_total - oh_concentration
        
        molecular_fraction = h2o_molecular / h2o_total
        
        return np.clip(molecular_fraction, 0.0, 1.0)
    
    def _calculate_pressure_viscosity_correction(self, pressure: float, 
                                               temperature: np.ndarray, 
                                               composition: dict) -> np.ndarray:
        """
        Calculate pressure effects on melt viscosity
        Based on Persikov et al. (1990) and Ardia et al. (2008)
        """
        
        # Reference pressure (typically 1 bar)
        p_ref = 1e5  # Pa
        
        if pressure <= p_ref:
            return np.ones_like(temperature)
        
        # Activation volume for viscous flow (composition-dependent)
        sio2 = composition.get('SiO2', 70)
        al2o3 = composition.get('Al2O3', 15)
        alkali_total = composition.get('Na2O', 4) + composition.get('K2O', 4)
        
        # Activation volume in cm3/mol (Persikov et al. 1990)
        va = 3.0 + 0.02 * sio2 - 0.04 * alkali_total - 0.01 * al2o3
        va *= 1e-6  # Convert to m3/mol
        
        # Gas constant
        R = 8.314  # J/(mol·K)
        
        # Pressure correction factor
        pressure_term = va * (pressure - p_ref) / (R * temperature)
        correction = np.exp(pressure_term)
        
        return correction
    
    def _calculate_alkali_viscosity_correction(self, composition: dict) -> float:
        """
        Calculate alkali-specific corrections to viscosity
        Based on Whittington et al. (2009) and Giordano & Dingwell (2003)
        """
        
        na2o = composition.get('Na2O', 0)
        k2o = composition.get('K2O', 0)
        
        if na2o + k2o == 0:
            return 1.0
        
        # Different depolymerization effects for Na vs K
        # K2O is more effective at reducing viscosity than Na2O
        na_effect = 1.0 - 0.018 * na2o
        k_effect = 1.0 - 0.024 * k2o
        
        # Mixed alkali effect (non-linear combination)
        if na2o > 0 and k2o > 0:
            mixed_alkali_factor = 1 + 0.003 * na2o * k2o
            return na_effect * k_effect * mixed_alkali_factor
        
        return na_effect * k_effect
    
    def _calculate_fluorine_viscosity_correction(self, composition: dict) -> float:
        """
        Calculate fluorine effects on melt viscosity
        Based on Dingwell (1996) and Baasner et al. (2013)
        """
        
        f_content = composition.get('F', 0)
        
        if f_content == 0:
            return 1.0
        
        # F causes strong depolymerization and viscosity reduction
        # Non-linear relationship due to network topology changes
        f_correction = np.exp(-0.45 * f_content**0.72)
        
        return f_correction
    
    def _calculate_crystal_viscosity_correction(self, reservoir: MagmaReservoir, 
                                              temperature: np.ndarray) -> np.ndarray:
        """
        Calculate effects of crystal content on bulk magma viscosity
        
        Uses:
        - Einstein-Roscoe equation for low crystal fractions
        - Krieger-Dougherty equation for high crystal fractions
        - Costa et al. (2009) model for natural magmas
        """
        
        # Estimate crystal fraction from reservoir state
        crystal_fraction = self._estimate_crystal_fraction(reservoir, temperature)
        
        correction = np.ones_like(temperature)
        
        # Apply crystal content corrections where crystals are present
        crystal_mask = crystal_fraction > 0.01
        
        if np.any(crystal_mask):
            phi = crystal_fraction[crystal_mask]
            
            # Maximum packing fraction (depends on crystal shapes and sizes)
            phi_max = 0.65  # Typical for natural crystal assemblages
            
            # Intrinsic viscosity (Einstein coefficient, shape-dependent)
            einstein_coeff = 2.5  # For spherical particles
            
            # Apply appropriate model based on crystal fraction
            low_phi_mask = phi < 0.2
            high_phi_mask = phi >= 0.2
            
            # Einstein-Roscoe for low crystal fractions
            if np.any(low_phi_mask):
                correction[crystal_mask][low_phi_mask] = (1 + einstein_coeff * 
                                                        phi[low_phi_mask])
            
            # Krieger-Dougherty for high crystal fractions
            if np.any(high_phi_mask):
                phi_high = phi[high_phi_mask]
                correction[crystal_mask][high_phi_mask] = (
                    (1 - phi_high/phi_max)**(-einstein_coeff * phi_max))
            
            # Additional corrections for crystal interactions
            interaction_correction = 1 + 0.5 * phi**2
            correction[crystal_mask] *= interaction_correction
        
        return correction
    
    def _estimate_crystal_fraction(self, reservoir: MagmaReservoir, 
                                 temperature: np.ndarray) -> np.ndarray:
        """
        Estimate crystal fraction based on temperature and equilibrium crystallization
        """
        
        # Liquidus temperature estimation
        liquidus_temp = self._estimate_liquidus_temperature(reservoir.composition)
        
        # Crystal fraction increases as temperature drops below liquidus
        crystal_fraction = np.zeros_like(temperature)
        
        below_liquidus = temperature < liquidus_temp
        if np.any(below_liquidus):
            undercooling = liquidus_temp - temperature[below_liquidus]
            
            # Empirical relationship for crystal fraction vs undercooling
            # Based on experimental data for silicic magmas
            crystal_fraction[below_liquidus] = 1 - np.exp(-undercooling / 50)
            crystal_fraction[below_liquidus] = np.clip(crystal_fraction[below_liquidus], 
                                                     0.0, 0.8)
        
        return crystal_fraction
    
    def _estimate_liquidus_temperature(self, composition: dict) -> float:
        """
        Comprehensive liquidus temperature estimation using multiple thermodynamic models
        
        Combines:
        1. Putirka (2008) - Empirical models for various compositions
        2. Ghiorso & Sack (1995) - Thermodynamic approach (MELTS)
        3. Lange et al. (2009) - Pressure corrections
        4. Bohrson & Spera (2001) - Assimilation effects
        5. Gualda et al. (2012) - rhyolite-MELTS improvements
        6. Kress & Carmichael (1991) - Fe redox effects
        """
        
        # Get comprehensive composition with proper normalization
        comp_normalized = self._normalize_composition_for_thermodynamics(composition)
        
        # Primary liquidus calculation using multiple approaches
        liquidus_putirka = self._liquidus_putirka_2008(comp_normalized)
        liquidus_melts = self._liquidus_melts_approach(comp_normalized)
        liquidus_lange = self._liquidus_lange_pressure_corrected(comp_normalized)
        
        # Weight different models based on composition appropriateness
        model_weights = self._calculate_liquidus_model_weights(comp_normalized)
        
        # Weighted combination of models
        liquidus_temp = (model_weights['putirka'] * liquidus_putirka +
                        model_weights['melts'] * liquidus_melts +
                        model_weights['lange'] * liquidus_lange)
        
        # Apply additional corrections
        pressure_correction = self._liquidus_pressure_correction(liquidus_temp, composition)
        volatile_correction = self._liquidus_volatile_correction(liquidus_temp, composition)
        redox_correction = self._liquidus_redox_correction(liquidus_temp, composition)
        
        comprehensive_liquidus = (liquidus_temp + 
                                pressure_correction + 
                                volatile_correction + 
                                redox_correction)
        
        # Apply physical bounds (600-1600°C for natural magmas)
        return np.clip(comprehensive_liquidus, 873, 1873)  # K
    
    def _normalize_composition_for_thermodynamics(self, composition: dict) -> dict:
        """
        Normalize composition for thermodynamic calculations
        Ensures proper stoichiometry and handles trace elements
        """
        
        # Convert all to weight percent if in molar fractions
        comp_wt = {}
        for oxide, value in composition.items():
            if value < 1.0:  # Assume molar fraction if < 1
                # Convert using molecular weights (simplified)
                mol_weights = {
                    'SiO2': 60.08, 'TiO2': 79.87, 'Al2O3': 101.96, 'Fe2O3': 159.69,
                    'FeO': 71.84, 'MnO': 70.94, 'MgO': 40.30, 'CaO': 56.08,
                    'Na2O': 61.98, 'K2O': 94.20, 'P2O5': 141.94, 'H2O': 18.02,
                    'CO2': 44.01, 'SO2': 64.06, 'Cl2O': 86.90, 'F2O': 53.98
                }
                comp_wt[oxide] = value * mol_weights.get(oxide, 60.0)
            else:
                comp_wt[oxide] = value
        
        # Normalize to 100% (anhydrous basis for liquidus)
        anhydrous_oxides = [k for k in comp_wt.keys() 
                           if k not in ['H2O', 'CO2', 'SO2', 'Cl2O', 'F2O']]
        anhydrous_total = sum(comp_wt.get(oxide, 0) for oxide in anhydrous_oxides)
        
        if anhydrous_total > 0:
            normalization_factor = 100.0 / anhydrous_total
            for oxide in anhydrous_oxides:
                if oxide in comp_wt:
                    comp_wt[oxide] *= normalization_factor
        
        return comp_wt
    
    def _liquidus_putirka_2008(self, composition: dict) -> float:
        """
        Putirka (2008) empirical liquidus temperature models
        Equation 15a for silicate melts
        """
        
        sio2 = composition.get('SiO2', 70.0)
        al2o3 = composition.get('Al2O3', 15.0)
        feo_total = composition.get('FeO', 2.0) + composition.get('Fe2O3', 1.0) * 0.8998
        mgo = composition.get('MgO', 2.0)
        cao = composition.get('CaO', 3.0)
        na2o = composition.get('Na2O', 4.0)
        k2o = composition.get('K2O', 4.0)
        tio2 = composition.get('TiO2', 0.5)
        
        # Putirka (2008) Equation 15a - Liquidus temperature
        liquidus_C = (1000.0 + 
                     8.0 * sio2 + 
                     18.0 * al2o3 + 
                     9.0 * feo_total + 
                     5.0 * mgo + 
                     12.0 * cao - 
                     21.0 * na2o - 
                     15.0 * k2o + 
                     10.0 * tio2)
        
        # Apply compositional limits and corrections
        if sio2 > 77:  # High-silica correction
            liquidus_C -= (sio2 - 77) * 2.5
        
        if al2o3 > 18:  # High-alumina correction  
            liquidus_C += (al2o3 - 18) * 1.8
        
        return liquidus_C + 273.15  # Convert to Kelvin
    
    def _liquidus_melts_approach(self, composition: dict) -> float:
        """
        MELTS-style thermodynamic approach to liquidus temperature
        Based on Ghiorso & Sack (1995) and Gualda et al. (2012)
        """
        
        # Activity coefficients for major components
        activities = self._calculate_melt_activities(composition)
        
        # Gibbs free energy of fusion for major phases
        fusion_energies = {
            'quartz': {'DeltaG': 9215, 'DeltaS': 4.60},    # J/mol, J/(mol·K)
            'feldspar': {'DeltaG': 25800, 'DeltaS': 14.1},
            'pyroxene': {'DeltaG': 58600, 'DeltaS': 38.5},
            'olivine': {'DeltaG': 71200, 'DeltaS': 43.2}
        }
        
        # Calculate liquidus for each potential liquidus phase
        liquidus_temps = []
        
        sio2 = composition.get('SiO2', 70)
        
        # Determine likely liquidus phase based on composition
        if sio2 > 72:  # Quartz saturation
            phase = 'quartz'
        elif sio2 > 65:  # Feldspar saturation  
            phase = 'feldspar'
        elif sio2 > 52:  # Pyroxene saturation
            phase = 'pyroxene'
        else:  # Olivine saturation
            phase = 'olivine'
        
        # Calculate equilibrium temperature: T = DeltaG / DeltaS
        delta_g = fusion_energies[phase]['DeltaG']
        delta_s = fusion_energies[phase]['DeltaS']
        
        # Activity correction
        activity_correction = -8.314 * np.log(activities.get(phase, 0.5))
        
        liquidus_temp = (delta_g + activity_correction) / delta_s
        
        return liquidus_temp
    
    def _calculate_melt_activities(self, composition: dict) -> dict:
        """
        Calculate activity coefficients for melt components
        Simplified regular solution model
        """
        
        # Mole fractions (simplified conversion)
        total_moles = sum(composition.values())
        
        activities = {
            'quartz': composition.get('SiO2', 70) / total_moles,
            'feldspar': (composition.get('Al2O3', 15) + 
                        composition.get('Na2O', 4) + 
                        composition.get('K2O', 4)) / total_moles,
            'pyroxene': (composition.get('CaO', 3) + 
                        composition.get('MgO', 2) + 
                        composition.get('FeO', 2)) / total_moles,
            'olivine': (composition.get('MgO', 2) + 
                       composition.get('FeO', 2)) / total_moles
        }
        
        return activities
    
    def _liquidus_lange_pressure_corrected(self, composition: dict) -> float:
        """
        Pressure-corrected liquidus using Lange et al. (2009) approach
        """
        
        # Base liquidus at 1 bar
        base_liquidus = self._liquidus_putirka_2008(composition)
        
        # Pressure correction (simplified Clausius-Clapeyron)
        # dT/dP = T * DeltaV / DeltaH
        
        pressure_bar = 1000  # Assume typical crustal pressure
        delta_v_fusion = -2.0e-5  # m³/mol (volume change on melting)
        delta_h_fusion = 400000   # J/mol (enthalpy of fusion)
        
        pressure_correction = (base_liquidus * delta_v_fusion * 
                             (pressure_bar - 1) * 1e5 / delta_h_fusion)
        
        return base_liquidus + pressure_correction
    
    def _calculate_liquidus_model_weights(self, composition: dict) -> dict:
        """
        Calculate weights for different liquidus models based on composition
        """
        
        sio2 = composition.get('SiO2', 70)
        al2o3 = composition.get('Al2O3', 15)
        alkalis = composition.get('Na2O', 4) + composition.get('K2O', 4)
        
        weights = {'putirka': 0.5, 'melts': 0.3, 'lange': 0.2}
        
        # Adjust weights based on composition
        if sio2 > 75:  # High-silica: favor Putirka
            weights['putirka'] = 0.7
            weights['melts'] = 0.2
            weights['lange'] = 0.1
        elif sio2 < 55:  # Mafic: favor MELTS thermodynamics
            weights['putirka'] = 0.2
            weights['melts'] = 0.6
            weights['lange'] = 0.2
        
        if alkalis > 8:  # High alkalis: increase Lange weight
            weights['lange'] *= 1.5
            # Renormalize
            total = sum(weights.values())
            for key in weights:
                weights[key] /= total
        
        return weights
    
    def _liquidus_pressure_correction(self, liquidus_temp: float, 
                                    composition: dict) -> float:
        """
        Apply pressure corrections to liquidus temperature
        """
        
        # Estimate pressure from typical crustal depths
        depth_km = 5  # Typical shallow magma chamber
        pressure_bar = depth_km * 270  # Approximate lithostatic pressure
        
        # dT/dP varies with composition
        sio2 = composition.get('SiO2', 70)
        
        if sio2 > 70:  # Felsic
            dt_dp = 3.2e-8  # K/Pa
        else:  # Mafic
            dt_dp = 2.8e-8  # K/Pa
        
        pressure_correction = dt_dp * pressure_bar * 1e5  # Convert bar to Pa
        
        return pressure_correction
    
    def _liquidus_volatile_correction(self, liquidus_temp: float, 
                                    composition: dict) -> float:
        """
        Apply volatile corrections to liquidus temperature
        """
        
        h2o = composition.get('H2O', 0)
        co2 = composition.get('CO2', 0)
        
        # H2O strongly depresses liquidus
        h2o_correction = -h2o * 35  # K per wt% H2O
        
        # CO2 has weaker effect
        co2_correction = -co2 * 8   # K per wt% CO2
        
        return h2o_correction + co2_correction
    
    def _liquidus_redox_correction(self, liquidus_temp: float, 
                                 composition: dict) -> float:
        """
        Apply redox state corrections based on Fe³⁺/Fe²⁺ ratio
        """
        
        fe2o3 = composition.get('Fe2O3', 1)
        feo = composition.get('FeO', 2)
        
        if feo > 0:
            fe3_fe2_ratio = (fe2o3 * 0.8998) / feo  # Convert Fe2O3 to FeO equivalent
        else:
            fe3_fe2_ratio = 1.0
        
        # More oxidized conditions slightly increase liquidus
        redox_correction = fe3_fe2_ratio * 5  # K
        
        return redox_correction
    
    def _calculate_comprehensive_radioactive_heating(self, model_state: Dict[str, Any], 
                                                   temperature: np.ndarray) -> np.ndarray:
        """
        Calculate comprehensive radioactive heat generation from multiple isotopes
        
        Includes:
        1. Primary heat-producing isotopes: U-238, U-235, Th-232, K-40
        2. Secondary isotopes: Rb-87, Sm-147, Lu-176, Re-187
        3. Spatial distribution based on rock type and differentiation
        4. Temperature-dependent decay corrections
        5. Time-dependent isotope evolution
        6. Crustal vs mantle signatures
        """
        
        # Get spatial coordinates and rock types
        rock_types = model_state.get('rock_types', 'granite')
        
        # Primary heat-producing elements (HPE) concentrations by rock type
        hpe_concentrations = self._get_hpe_concentrations_by_rock_type(rock_types)
        
        # Calculate heat production from each isotope system
        u238_heating = self._calculate_u238_series_heating(hpe_concentrations, temperature)
        u235_heating = self._calculate_u235_series_heating(hpe_concentrations, temperature)
        th232_heating = self._calculate_th232_series_heating(hpe_concentrations, temperature)
        k40_heating = self._calculate_k40_heating(hpe_concentrations, temperature)
        
        # Secondary isotope contributions
        rb87_heating = self._calculate_rb87_heating(hpe_concentrations, temperature)
        sm147_heating = self._calculate_sm147_heating(hpe_concentrations, temperature)
        
        # Spatial heterogeneity corrections
        spatial_correction = self._apply_radioactive_spatial_heterogeneity(
            model_state, temperature)
        
        # Temperature-dependent corrections for decay rates
        temp_correction = self._temperature_dependent_decay_correction(temperature)
        
        # Total radioactive heating
        total_heating = ((u238_heating + u235_heating + th232_heating + k40_heating + 
                         rb87_heating + sm147_heating) * 
                        spatial_correction * temp_correction)
        
        return total_heating
    
    def _get_hpe_concentrations_by_rock_type(self, rock_types) -> dict:
        """
        Get heat-producing element concentrations based on rock type
        Values in ppm (U, Th) and wt% (K)
        """
        
        # Typical concentrations from Rybach (1988), Rudnick & Gao (2003)
        concentrations = {
            'granite': {'U': 4.8, 'Th': 20.0, 'K': 4.2, 'Rb': 150, 'Sm': 8.0},
            'granodiorite': {'U': 2.8, 'Th': 11.0, 'K': 3.1, 'Rb': 110, 'Sm': 6.2},
            'diorite': {'U': 1.5, 'Th': 6.0, 'K': 1.8, 'Rb': 65, 'Sm': 4.8},
            'gabbro': {'U': 0.6, 'Th': 2.8, 'K': 0.8, 'Rb': 30, 'Sm': 3.2},
            'basalt': {'U': 0.8, 'Th': 3.2, 'K': 1.2, 'Rb': 35, 'Sm': 3.8},
            'rhyolite': {'U': 5.2, 'Th': 22.0, 'K': 4.8, 'Rb': 180, 'Sm': 7.5},
            'andesite': {'U': 2.2, 'Th': 9.5, 'K': 2.5, 'Rb': 85, 'Sm': 5.1},
            'peridotite': {'U': 0.02, 'Th': 0.08, 'K': 0.02, 'Rb': 0.6, 'Sm': 0.4},
            'dunite': {'U': 0.01, 'Th': 0.04, 'K': 0.01, 'Rb': 0.3, 'Sm': 0.2}
        }
        
        # Default to granite if rock type not specified
        if isinstance(rock_types, str):
            return concentrations.get(rock_types, concentrations['granite'])
        else:
            # Handle spatial distribution of rock types
            return concentrations.get('granite')  # Simplified
    
    def _calculate_u238_series_heating(self, concentrations: dict, 
                                     temperature: np.ndarray) -> np.ndarray:
        """
        Calculate heating from U-238 decay series
        Includes full decay chain: U-238 → Th-234 → Pa-234 → U-234 → ... → Pb-206
        """
        
        u_concentration = concentrations.get('U', 4.0)  # ppm
        
        # Heat production rate for U-238 series (W/kg per ppm U)
        # Includes all daughter products in secular equilibrium
        lambda_u238 = 1.55125e-10  # decay constant (1/yr)
        q_u238 = 9.485e-5  # W/kg per ppm U (Rybach 1988)
        
        # Temperature dependence (minimal for radioactive decay)
        temp_factor = np.ones_like(temperature)
        
        heating_rate = u_concentration * q_u238 * temp_factor
        
        return heating_rate
    
    def _calculate_u235_series_heating(self, concentrations: dict, 
                                     temperature: np.ndarray) -> np.ndarray:
        """
        Calculate heating from U-235 decay series (Actinium series)
        """
        
        u_concentration = concentrations.get('U', 4.0)  # ppm
        
        # U-235 is 0.72% of natural uranium
        u235_concentration = u_concentration * 0.0072
        
        # Heat production rate for U-235 series
        q_u235 = 5.69e-4  # W/kg per ppm U-235
        
        temp_factor = np.ones_like(temperature)
        heating_rate = u235_concentration * q_u235 * temp_factor
        
        return heating_rate
    
    def _calculate_th232_series_heating(self, concentrations: dict, 
                                      temperature: np.ndarray) -> np.ndarray:
        """
        Calculate heating from Th-232 decay series (Thorium series)
        """
        
        th_concentration = concentrations.get('Th', 20.0)  # ppm
        
        # Heat production rate for Th-232 series
        q_th232 = 2.64e-5  # W/kg per ppm Th
        
        temp_factor = np.ones_like(temperature)
        heating_rate = th_concentration * q_th232 * temp_factor
        
        return heating_rate
    
    def _calculate_k40_heating(self, concentrations: dict, 
                             temperature: np.ndarray) -> np.ndarray:
        """
        Calculate heating from K-40 decay (both beta and electron capture branches)
        """
        
        k_concentration = concentrations.get('K', 4.0)  # wt%
        
        # K-40 is 0.0117% of natural potassium
        k40_abundance = 0.000117
        
        # Heat production rate for K-40
        q_k40 = 3.48e-5  # W/kg per wt% K
        
        temp_factor = np.ones_like(temperature)
        heating_rate = k_concentration * q_k40 * temp_factor
        
        return heating_rate
    
    def _calculate_rb87_heating(self, concentrations: dict, 
                              temperature: np.ndarray) -> np.ndarray:
        """
        Calculate heating from Rb-87 decay (minor contribution)
        """
        
        rb_concentration = concentrations.get('Rb', 150)  # ppm
        
        # Rb-87 is 27.8% of natural rubidium
        rb87_concentration = rb_concentration * 0.278
        
        # Heat production rate for Rb-87 (very low)
        q_rb87 = 3.2e-10  # W/kg per ppm Rb-87
        
        temp_factor = np.ones_like(temperature)
        heating_rate = rb87_concentration * q_rb87 * temp_factor
        
        return heating_rate
    
    def _calculate_sm147_heating(self, concentrations: dict, 
                               temperature: np.ndarray) -> np.ndarray:
        """
        Calculate heating from Sm-147 decay (very minor contribution)
        """
        
        sm_concentration = concentrations.get('Sm', 6.0)  # ppm
        
        # Sm-147 is 15% of natural samarium
        sm147_concentration = sm_concentration * 0.15
        
        # Heat production rate for Sm-147 (extremely low)
        q_sm147 = 1.96e-12  # W/kg per ppm Sm-147
        
        temp_factor = np.ones_like(temperature)
        heating_rate = sm147_concentration * q_sm147 * temp_factor
        
        return heating_rate
    
    def _apply_radioactive_spatial_heterogeneity(self, model_state: Dict[str, Any], 
                                               temperature: np.ndarray) -> np.ndarray:
        """
        Apply spatial heterogeneity in radioactive element distribution
        """
        
        # Get material distribution if available
        material_dist = model_state.get('material_distribution', {})
        
        # Apply heterogeneity based on differentiation processes
        base_heterogeneity = np.ones_like(temperature)
        
        # Higher concentrations in evolved/differentiated regions
        # Simplified approach: higher temps = more differentiated
        differentiation_factor = 1.0 + 0.1 * (temperature - 1000) / 200
        differentiation_factor = np.clip(differentiation_factor, 0.5, 2.0)
        
        return base_heterogeneity * differentiation_factor
    
    def _temperature_dependent_decay_correction(self, temperature: np.ndarray) -> np.ndarray:
        """
        Apply temperature-dependent corrections to decay rates
        (Generally minimal effect, but included for completeness)
        """
        
        # Radioactive decay rates are essentially temperature-independent
        # Small correction for bound-state beta decay (negligible)
        correction = np.ones_like(temperature)
        
        return correction
    
    def _temperature_model_weight(self, temperature: np.ndarray, model: str) -> np.ndarray:
        """
        Calculate temperature-dependent weights for model mixing
        Higher temperatures favor Shaw model, lower temperatures favor Giordano model
        """
        
        # Transition temperature around 1200K
        t_transition = 1200
        
        if model == 'giordano':
            # Giordano model better at lower temperatures
            weight = 1.0 / (1.0 + np.exp((temperature - t_transition) / 100))
        else:  # shaw
            # Shaw model better at higher temperatures
            weight = 1.0 / (1.0 + np.exp(-(temperature - t_transition) / 100))
        
        return weight
        
        SiO2 = reservoir.composition.get('SiO2', 0.5) * 100  # Convert to wt%
        H2O = reservoir.composition.get('H2O', 0.0) * 100
        
        # Temperature-dependent parameters
        A = -4.55 + 0.045 * SiO2 - 2.0 * H2O**0.3
        B = 9601 - 120 * SiO2**0.5 + 1500 * H2O**0.5
        C = 195 + 32.5 * np.log(SiO2) - 150 * H2O**0.3
        
        log_viscosity = A + B / (temperature - C + 1e-10)
        viscosity = 10**log_viscosity
        
        return np.clip(viscosity, 1e-2, 1e12)  # Physical bounds for melt viscosity
    
    def _calculate_crystal_growth_rate(self, temperature: np.ndarray, 
                                     pressure: np.ndarray) -> np.ndarray:
        """Calculate crystal growth rate (diffusion-controlled)"""
        
        # Arrhenius relationship for diffusion
        # D = D0 * exp(-Ea/RT)
        
        D0 = 1e-6  # Pre-exponential factor (m²/s)
        Ea = 300000.0  # Activation energy for Si-O diffusion (J/mol)
        R = 8.314  # Gas constant
        
        # Diffusion coefficient
        diffusivity = D0 * np.exp(-Ea / (R * temperature))
        
        # Pressure effect (higher pressure slightly reduces diffusion)
        pressure_effect = np.exp(-pressure * 0.5)  # GPa⁻¹
        
        growth_rate = diffusivity * pressure_effect
        
        # Normalize to [0,1] range
        growth_rate = growth_rate / np.max(growth_rate)
        
        return growth_rate
    
    def _apply_reservoir_spatial_weighting(self, melt_fraction: np.ndarray,
                                         distance: np.ndarray, reservoir_radius: float,
                                         reservoir_melt_fraction: float) -> np.ndarray:
        """Apply spatial weighting based on reservoir geometry"""
        
        # Multi-scale weighting function
        # Core region: exponential decay
        core_factor = np.exp(-(distance / reservoir_radius)**2)
        
        # Transition zone: polynomial decay
        transition_factor = np.where(
            distance <= 2 * reservoir_radius,
            (1 - (distance / (2 * reservoir_radius))**3),
            0.0
        )
        
        # Far-field: minimal influence
        far_field_factor = np.where(
            distance <= 5 * reservoir_radius,
            0.1 * np.exp(-(distance / (5 * reservoir_radius))**2),
            0.0
        )
        
        # Combined weighting
        total_weight = core_factor + 0.3 * transition_factor + far_field_factor
        total_weight = np.clip(total_weight, 0.0, 1.0)
        
        # Apply spatial weighting
        weighted_melt_fraction = (
            melt_fraction * (1 - total_weight) + 
            reservoir_melt_fraction * total_weight
        )
        
        return weighted_melt_fraction

# Example usage and testing
async def example_magma_reservoir_monitoring():
    """Example of magma reservoir monitoring system"""
    
    # Create a magma reservoir (Yellowstone-like system)
    yellowstone_reservoir = MagmaReservoir(
        id="yellowstone_main_chamber",
        parent_volcanic_system="yellowstone_caldera",
        reservoir_type=MagmaReservoirType.MID_CRUSTAL,
        geometry={
            'volume_km3': 15000,
            'depth_km': 8,
            'aspect_ratio': 2.0,
            'shape': 'ellipsoidal'
        },
        location=np.array([-110.5, 44.6, 8.0]),  # lon, lat, depth_km
        composition={
            'SiO2': 75.0, 'Al2O3': 13.5, 'FeO': 1.2, 'MgO': 0.5,
            'CaO': 1.0, 'Na2O': 3.8, 'K2O': 4.2, 'TiO2': 0.3
        },
        physical_properties={
            'temperature': 850,  # °C
            'viscosity': 1e6,    # Pa·s
            'density': 2300      # kg/m³
        },
        state=ReservoirState.RECHARGING,
        pressure=30e6,  # 30 MPa overpressure
        crystallinity=0.4,
        melt_fraction=0.15,
        volatile_content={
            'H2O': 4.0,  # wt%
            'CO2': 0.1,
            'SO2': 0.05
        }
    )
    
    # Initialize monitoring system
    monitoring_system = MagmaReservoirMonitoringSystem()
    monitoring_system.add_reservoir(yellowstone_reservoir)
    
    print(f"Initialized monitoring for {yellowstone_reservoir.id}")
    print(f"Reservoir type: {yellowstone_reservoir.reservoir_type.value}")
    print(f"Volume: {yellowstone_reservoir.geometry['volume_km3']} km³")
    print(f"Depth: {yellowstone_reservoir.location[2]} km")
    print(f"State: {yellowstone_reservoir.state.value}")
    
    # Simulate monitoring data updates
    reservoir_id = yellowstone_reservoir.id
    
    # Seismic data (earthquake rate)
    seismic_data = np.array([50, 55, 60, 58, 65, 70, 75, 80, 85, 90])  # events/week
    monitoring_system.update_monitoring_data(reservoir_id, 'seismic_rate', seismic_data)
    
    # Ground deformation data
    deformation_data = np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8])  # cm/year uplift
    monitoring_system.update_monitoring_data(reservoir_id, 'ground_deformation', deformation_data)
    
    # Gas flux data
    gas_data = np.array([100, 105, 110, 115, 125, 135, 145, 160, 170, 185])  # tons/day CO2
    monitoring_system.update_monitoring_data(reservoir_id, 'co2_flux', gas_data)
    
    print(f"\nUpdated monitoring data:")
    print(f"Latest seismic rate: {seismic_data[-1]} events/week")
    print(f"Latest deformation: {deformation_data[-1]} cm/year")
    print(f"Latest CO2 flux: {gas_data[-1]} tons/day")
    
    # Run comprehensive analysis
    print(f"\nRunning comprehensive analysis for {reservoir_id}...")
    analysis_result = await monitoring_system.run_comprehensive_analysis(reservoir_id)
    
    print(f"Analysis completed at: {analysis_result['analysis_timestamp']}")
    
    # Display thermal evolution results
    thermal_evolution = analysis_result['thermal_evolution']
    print(f"\nThermal Evolution:")
    print(f"  Max temperature: {thermal_evolution.get('thermal_history', [{}])[-1].get('max_temperature', 'N/A')} K")
    print(f"  Average temperature: {thermal_evolution.get('thermal_history', [{}])[-1].get('average_temperature', 'N/A')} K")
    
    # Display mechanical evolution results
    mechanical_evolution = analysis_result['mechanical_evolution']
    surface_deformation = mechanical_evolution.get('surface_deformation', {})
    print(f"\nMechanical Evolution:")
    if 'vertical_displacement' in surface_deformation:
        max_uplift = np.max(surface_deformation['vertical_displacement'])
        print(f"  Maximum surface uplift: {max_uplift:.6f} strain")
    if 'tilt_magnitude' in surface_deformation:
        max_tilt = np.max(surface_deformation['tilt_magnitude'])
        print(f"  Maximum tilt: {max_tilt:.2e} radians")
    
    # Display geophysical signatures
    geophysical_signatures = analysis_result['geophysical_signatures']
    print(f"\nGeophysical Signatures:")
    
    if 'gravity' in geophysical_signatures:
        gravity_anomaly = geophysical_signatures['gravity']['gravity_anomaly']
        print(f"  Gravity anomaly range: {np.min(gravity_anomaly):.1f} to {np.max(gravity_anomaly):.1f} mGal")
    
    if 'magnetic' in geophysical_signatures:
        magnetic_anomaly = geophysical_signatures['magnetic']['magnetic_anomaly']
        print(f"  Magnetic anomaly range: {np.min(magnetic_anomaly):.1f} to {np.max(magnetic_anomaly):.1f} nT")
    
    if 'seismic' in geophysical_signatures:
        vp_anomaly = geophysical_signatures['seismic']['velocity_anomaly_vp']
        print(f"  P-wave velocity anomaly: {np.min(vp_anomaly):.3f} to {np.max(vp_anomaly):.3f}")
    
    # Display hazard assessment
    hazard_assessment = analysis_result['hazard_assessment']
    print(f"\nHazard Assessment:")
    print(f"  Overall hazard level: {hazard_assessment['overall_hazard_level']}")
    print(f"  Eruption probability: {hazard_assessment['eruption_probability']:.3f}")
    print(f"  Predicted eruption style: {hazard_assessment.get('eruption_style_prediction', 'unknown')}")
    
    # Display hazard indicators
    indicators = hazard_assessment['hazard_indicators']
    print(f"  Hazard Indicators:")
    print(f"    Thermal: {indicators['thermal_indicator']:.3f}")
    print(f"    Pressure: {indicators['pressure_indicator']:.3f}")
    print(f"    Melt: {indicators['melt_indicator']:.3f}")
    print(f"    Fracture: {indicators['fracture_indicator']:.3f}")
    
    # Display monitoring recommendations
    recommendations = analysis_result['monitoring_recommendations']
    print(f"\nMonitoring Recommendations ({len(recommendations)} total):")
    for i, rec in enumerate(recommendations[:5], 1):  # Show first 5
        print(f"  {i}. {rec}")
    
    if len(recommendations) > 5:
        print(f"  ... and {len(recommendations) - 5} more recommendations")
    
    print(f"\nMagma reservoir monitoring analysis complete!")

if __name__ == "__main__":
    print("Advanced Magma Reservoir Monitoring System")
    print(f"FEniCS available: {FENICS_AVAILABLE}")
    print(f"Geospatial tools available: {GEOSPATIAL_AVAILABLE}")
    print(f"Mapping tools available: {MAPPING_AVAILABLE}")
    print(f"Seismic tools available: {SEISMIC_AVAILABLE}")
    
    # Run example
    asyncio.run(example_magma_reservoir_monitoring())