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
        
        # Simplified melt fraction calculation
        solidus_temp = self.config['magma_properties']['solidus_temperature']
        liquidus_temp = self.config['magma_properties']['liquidus_temperature']
        
        # Linear interpolation between solidus and liquidus
        melt_fraction = np.clip(
            (temperature_field - solidus_temp) / (liquidus_temp - solidus_temp),
            0, 1
        )
        
        # Modify based on reservoir crystallinity
        reservoir_center = grid['reservoir_center']
        reservoir_radius = grid['reservoir_radius']
        
        distance = np.sqrt((X - reservoir_center[0])**2 + 
                          (Y - reservoir_center[1])**2 + 
                          (Z - reservoir_center[2])**2)
        
        # Higher melt fraction in reservoir core
        core_factor = np.exp(-(distance / reservoir_radius)**2)
        reservoir_melt_fraction = reservoir.melt_fraction
        
        melt_fraction = melt_fraction * (1 - core_factor) + reservoir_melt_fraction * core_factor
        
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
        
        # Radioactive heat generation (simplified)
        radioactive_heat_rate = 1e-6  # W/m³
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
        """Calculate stress from pressure gradients"""
        
        dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
        
        # Pressure gradients
        dP_dx = np.gradient(pressure, dx, axis=0)
        dP_dy = np.gradient(pressure, dy, axis=1)
        dP_dz = np.gradient(pressure, dz, axis=2)
        
        # Convert pressure gradients to stress (simplified)
        pressure_stress = {
            'sigma_xx': -pressure,  # Compressive pressure
            'sigma_yy': -pressure,
            'sigma_zz': -pressure,
            'sigma_xy': np.zeros_like(pressure),
            'sigma_xz': np.zeros_like(pressure),
            'sigma_yz': np.zeros_like(pressure)
        }
        
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
        """Calculate probability of fracture formation"""
        
        # Extract principal stresses (simplified)
        sigma_xx = stress['sigma_xx']
        sigma_yy = stress['sigma_yy'] 
        sigma_zz = stress['sigma_zz']
        
        # von Mises stress
        sigma_vm = np.sqrt(0.5 * ((sigma_xx - sigma_yy)**2 + 
                                 (sigma_yy - sigma_zz)**2 + 
                                 (sigma_zz - sigma_xx)**2))
        
        # Fracture criterion (simplified)
        fracture_strength = 50e6  # Pa (typical rock strength)
        fracture_probability = 1 / (1 + np.exp(-(sigma_vm - fracture_strength) / (0.1 * fracture_strength)))
        
        return fracture_probability
    
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
        
        # Composition effects (simplified)
        SiO2_content = composition.get('SiO2', 65.0)
        composition_effect = (SiO2_content - 65.0) * 10  # Higher SiO2 = higher velocity
        
        # Total velocities
        vp_total = np.maximum(vp_temp + vp_pressure + vp_melt + composition_effect, 1000)
        vs_total = np.maximum(vs_temp + vs_pressure + vs_melt + composition_effect * 0.6, 500)
        
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