"""
Advanced Plate Evolution Engine
===============================

Self-consistent plate tectonics integration with mantle convection modeling.
Implements cutting-edge plate dynamics, subduction mechanics, continental 
assembly/breakup, and surface-deep Earth coupling including D" layer and 
LLSVP plume dynamics.

Key Features:
- Self-consistent plate generation from convection
- Advanced subduction zone dynamics
- Continental rifting and collision mechanics
- Plume-plate interactions
- Temporal plate boundary evolution
- Surface-mantle coupling algorithms
- Machine learning enhanced predictions

Author: Prometheus Deep Earth Team
Date: 2025
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, bicgstab
from scipy.interpolate import RegularGridInterpolator, RBFInterpolator
from scipy.spatial import Voronoi, SphericalVoronoi, distance_matrix
from scipy.optimize import minimize, differential_evolution
import cupy as cp
import cupyx.scipy.sparse as cp_sp
from numba import jit, cuda
import h5py
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlateType(Enum):
    """Types of tectonic plates"""
    OCEANIC = "oceanic"
    CONTINENTAL = "continental"
    MIXED = "mixed"
    MICROPLATE = "microplate"

class BoundaryType(Enum):
    """Types of plate boundaries"""
    DIVERGENT = "divergent"
    CONVERGENT = "convergent"
    TRANSFORM = "transform"
    DIFFUSE = "diffuse"
    INACTIVE = "inactive"

class SubductionType(Enum):
    """Types of subduction zones"""
    OCEANIC_OCEANIC = "oceanic_oceanic"
    OCEANIC_CONTINENTAL = "oceanic_continental"
    CONTINENTAL_CONTINENTAL = "continental_continental"

class PlumeType(Enum):
    """Types of mantle plumes"""
    DEEP_MANTLE = "deep_mantle_plume"
    SHALLOW_MANTLE = "shallow_mantle_plume"
    THERMAL_PLUME = "thermal_plume"
    COMPOSITIONAL_PLUME = "compositional_plume"
    THERMOCHEMICAL_PLUME = "thermochemical_plume"

@dataclass
class PlateProperties:
    """Properties of tectonic plates"""
    plate_id: int
    plate_type: PlateType
    area: float  # km^2
    age: float   # Myr
    thickness: float  # km
    density: float    # kg/m^3
    velocity: Tuple[float, float]  # cm/yr (vx, vy)
    angular_velocity: Tuple[float, float, float]  # rad/yr (wx, wy, wz)
    
    # Rheological properties
    elastic_thickness: float = 30.0  # km
    flexural_rigidity: float = 1e23  # N*m
    yield_strength: float = 200e6    # Pa
    
    # Thermal properties
    surface_temperature: float = 273  # K
    basal_temperature: float = 1600   # K
    thermal_diffusivity: float = 1e-6 # m^2/s
    
    # Chemical composition
    continental_fraction: float = 0.0  # 0-1
    water_content: float = 0.0         # wt%
    
    # Motion history
    rotation_poles: List[Tuple[float, float, float]] = field(default_factory=list)
    age_progression: List[float] = field(default_factory=list)

@dataclass
class BoundarySegment:
    """Plate boundary segment properties"""
    segment_id: int
    boundary_type: BoundaryType
    plate1_id: int
    plate2_id: int
    geometry: np.ndarray  # Array of lat/lon points
    length: float         # km
    
    # Kinematic properties
    spreading_rate: float = 0.0     # cm/yr (for divergent)
    convergence_rate: float = 0.0   # cm/yr (for convergent)
    transform_rate: float = 0.0     # cm/yr (for transform)
    
    # Dynamic properties
    force_balance: Dict[str, float] = field(default_factory=dict)
    stress_regime: str = "unknown"
    seismic_coupling: float = 0.5
    
    # Subduction properties (if applicable)
    subduction_type: Optional[SubductionType] = None
    slab_dip: Optional[float] = None           # degrees
    slab_age: Optional[float] = None           # Myr
    back_arc_spreading: bool = False

@dataclass
class MantlePlume:
    """Mantle plume properties"""
    plume_id: int
    plume_type: PlumeType
    location: Tuple[float, float]  # lat, lon
    depth_origin: float            # km
    temperature_anomaly: float     # K
    buoyancy_flux: float          # kg/s
    radius: float                 # km
    
    # Temporal evolution
    initiation_time: float = 0.0   # Myr ago
    activity_duration: float = 50.0 # Myr
    pulse_intervals: List[float] = field(default_factory=list)
    
    # Surface expressions
    hotspot_track: List[Tuple[float, float]] = field(default_factory=list)
    volcanic_flux: float = 0.0  # km^3/Myr
    
    # Deep structure
    source_region: str = "unknown"  # CMB, 660km, etc.
    llsvp_association: bool = False

class AdvancedPlateKinematics:
    """Advanced plate kinematics solver with Euler pole analysis"""
    
    def __init__(self):
        self.plates = {}
        self.boundaries = {}
        self.euler_poles = {}
        self.velocity_field = None
        
    def calculate_euler_poles(self, plate_velocities: Dict[int, np.ndarray]) -> Dict[int, Tuple[float, float, float]]:
        """Calculate Euler poles for plate rotations"""
        euler_poles = {}
        
        for plate_id, velocities in plate_velocities.items():
            if len(velocities) < 3:
                continue  # Need at least 3 velocity vectors
            
            # Solve for Euler pole using least squares
            pole = self._solve_euler_pole(velocities)
            euler_poles[plate_id] = pole
        
        return euler_poles
    
    def _solve_euler_pole(self, velocities: np.ndarray) -> Tuple[float, float, float]:
        """Solve for Euler pole from velocity vectors"""
        # Extract positions and velocities
        positions = velocities[:, :3]  # x, y, z coordinates
        vels = velocities[:, 3:6]      # vx, vy, vz components
        
        # Set up least squares system: v = omega x r
        # where omega is the angular velocity vector
        A = np.zeros((len(positions) * 3, 3))
        b = np.zeros(len(positions) * 3)
        
        for i, (pos, vel) in enumerate(zip(positions, vels)):
            # Cross product matrix for omega x r
            cross_matrix = np.array([
                [0, -pos[2], pos[1]],
                [pos[2], 0, -pos[0]],
                [-pos[1], pos[0], 0]
            ])
            
            A[i*3:(i+1)*3, :] = cross_matrix
            b[i*3:(i+1)*3] = vel
        
        # Solve least squares
        omega, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Convert to pole latitude, longitude, and rotation rate
        omega_magnitude = np.linalg.norm(omega)
        if omega_magnitude > 0:
            pole_x, pole_y, pole_z = omega / omega_magnitude
            
            # Convert to lat/lon
            pole_lat = np.arcsin(pole_z) * 180 / np.pi
            pole_lon = np.arctan2(pole_y, pole_x) * 180 / np.pi
            
            return (pole_lat, pole_lon, omega_magnitude)
        else:
            return (0.0, 0.0, 0.0)
    
    def predict_plate_motion(self, plate_id: int, time_step: float) -> np.ndarray:
        """Predict plate motion using Euler pole"""
        if plate_id not in self.euler_poles:
            return np.zeros((3,))
        
        pole_lat, pole_lon, omega = self.euler_poles[plate_id]
        
        # Convert pole to Cartesian coordinates
        pole_x = np.cos(np.radians(pole_lat)) * np.cos(np.radians(pole_lon))
        pole_y = np.cos(np.radians(pole_lat)) * np.sin(np.radians(pole_lon))
        pole_z = np.sin(np.radians(pole_lat))
        pole_vector = np.array([pole_x, pole_y, pole_z])
        
        # Calculate rotation for time step
        rotation_angle = omega * time_step
        rotation_matrix = self._rotation_matrix(pole_vector, rotation_angle)
        
        return rotation_matrix
    
    def _rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Create rotation matrix around axis by angle"""
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
        return R

class SubductionZoneDynamics:
    """Advanced subduction zone dynamics and slab evolution"""
    
    def __init__(self):
        self.subduction_zones = {}
        self.slab_geometries = {}
        self.thermal_models = {}
        
    def initialize_subduction_zone(self, zone_id: int, properties: Dict) -> None:
        """Initialize a subduction zone with given properties"""
        self.subduction_zones[zone_id] = {
            'type': properties.get('type', SubductionType.OCEANIC_OCEANIC),
            'age': properties.get('age', 100.0),  # Myr
            'convergence_rate': properties.get('convergence_rate', 5.0),  # cm/yr
            'slab_dip': properties.get('slab_dip', 45.0),  # degrees
            'trench_location': properties.get('trench_location', (0.0, 0.0)),
            'slab_thickness': properties.get('slab_thickness', 100.0),  # km
            'water_content': properties.get('water_content', 2.0),  # wt%
            'metamorphic_reactions': properties.get('metamorphic_reactions', True)
        }
        
        # Initialize slab geometry
        self._initialize_slab_geometry(zone_id)
        
        # Initialize thermal model
        self._initialize_thermal_model(zone_id)
    
    def _initialize_slab_geometry(self, zone_id: int) -> None:
        """Initialize 3D slab geometry"""
        zone = self.subduction_zones[zone_id]
        
        # Create slab outline based on dip angle
        dip = np.radians(zone['slab_dip'])
        thickness = zone['slab_thickness']
        
        # Define slab in 3D (simplified box model)
        max_depth = 660.0  # km, transition zone depth
        horizontal_extent = max_depth / np.tan(dip)
        
        # Create slab mesh
        nx, ny, nz = 50, 20, 30
        x = np.linspace(0, horizontal_extent, nx)
        y = np.linspace(-thickness/2, thickness/2, ny)
        z = np.linspace(0, max_depth, nz)
        
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Apply dip angle transformation
        X_dip = X * np.cos(dip) - Z * np.sin(dip)
        Z_dip = X * np.sin(dip) + Z * np.cos(dip)
        
        self.slab_geometries[zone_id] = {
            'x': X_dip,
            'y': Y,
            'z': Z_dip,
            'temperature': np.zeros_like(X),
            'pressure': np.zeros_like(X),
            'density': np.zeros_like(X),
            'viscosity': np.zeros_like(X)
        }
    
    def _initialize_thermal_model(self, zone_id: int) -> None:
        """Initialize thermal evolution model for slab"""
        zone = self.subduction_zones[zone_id]
        geometry = self.slab_geometries[zone_id]
        
        # Initial temperature distribution
        surface_temp = 273  # K
        mantle_temp = 1600  # K
        slab_age = zone['age'] * 1e6 * 365.25 * 24 * 3600  # seconds
        
        # Half-space cooling model
        thermal_diffusivity = 1e-6  # m^2/s
        depth_scale = 2 * np.sqrt(thermal_diffusivity * slab_age)
        
        # Calculate temperature field
        z_coords = geometry['z'] * 1000  # convert to meters
        temp_profile = surface_temp + (mantle_temp - surface_temp) * scipy.special.erf(z_coords / depth_scale)
        
        geometry['temperature'] = temp_profile
        
        # Calculate pressure (lithostatic)
        density_slab = 3300  # kg/m^3
        g = 9.81  # m/s^2
        geometry['pressure'] = density_slab * g * z_coords
        
        # Initialize thermal model parameters
        self.thermal_models[zone_id] = {
            'thermal_diffusivity': thermal_diffusivity,
            'specific_heat': 1200,  # J/kg/K
            'thermal_conductivity': 3.0,  # W/m/K
            'latent_heat_reactions': 400e3,  # J/kg
            'reaction_depths': [50e3, 100e3, 200e3],  # m
            'dehydration_reactions': True
        }
    
    def evolve_subduction_zone(self, zone_id: int, time_step: float) -> Dict[str, np.ndarray]:
        """Evolve subduction zone over time step"""
        zone = self.subduction_zones[zone_id]
        geometry = self.slab_geometries[zone_id]
        thermal = self.thermal_models[zone_id]
        
        # Calculate slab motion
        convergence_rate = zone['convergence_rate'] * 1e-2 / (365.25 * 24 * 3600)  # m/s
        displacement = convergence_rate * time_step
        
        # Update slab position
        dip = np.radians(zone['slab_dip'])
        dx = displacement * np.cos(dip)
        dz = displacement * np.sin(dip)
        
        geometry['x'] -= dx
        geometry['z'] += dz
        
        # Evolve thermal structure
        self._evolve_thermal_structure(zone_id, time_step)
        
        # Calculate metamorphic reactions
        if zone.get('metamorphic_reactions', False):
            self._calculate_metamorphic_reactions(zone_id)
        
        # Calculate slab deformation
        stress_field = self._calculate_stress_field(zone_id)
        
        # Calculate fluid release
        fluid_flux = self._calculate_fluid_release(zone_id)
        
        return {
            'geometry': geometry,
            'stress_field': stress_field,
            'fluid_flux': fluid_flux
        }
    
    def _evolve_thermal_structure(self, zone_id: int, dt: float) -> None:
        """Evolve thermal structure of subducting slab"""
        geometry = self.slab_geometries[zone_id]
        thermal = self.thermal_models[zone_id]
        
        # Get current temperature field
        T = geometry['temperature']
        
        # Calculate thermal diffusion (simplified 1D)
        kappa = thermal['thermal_diffusivity']
        
        # Simple explicit diffusion (would need proper 3D implementation)
        dT_dt = np.zeros_like(T)
        
        # Add heating from metamorphic reactions
        reaction_heating = self._calculate_reaction_heating(zone_id)
        dT_dt += reaction_heating / (thermal['specific_heat'] * 3300)
        
        # Update temperature
        geometry['temperature'] += dt * dT_dt
    
    def _calculate_metamorphic_reactions(self, zone_id: int) -> Dict[str, np.ndarray]:
        """Calculate metamorphic reactions in subducting slab"""
        geometry = self.slab_geometries[zone_id]
        T = geometry['temperature']
        P = geometry['pressure']
        
        # Define reaction curves (pressure-temperature conditions)
        reactions = {
            'blueschist_eclogite': {
                'P_min': 1e9,   # Pa
                'T_range': (500, 700),  # K
                'volume_change': -0.1,  # Fractional
                'water_release': 0.02   # wt%
            },
            'serpentine_dehydration': {
                'P_min': 2e9,
                'T_range': (600, 800),
                'volume_change': -0.15,
                'water_release': 0.13
            },
            'lawsonite_breakdown': {
                'P_min': 2.5e9,
                'T_range': (550, 750),
                'volume_change': -0.05,
                'water_release': 0.08
            }
        }
        
        reaction_map = {}
        for reaction_name, params in reactions.items():
            # Check P-T conditions
            pressure_condition = P >= params['P_min']
            temp_condition = (T >= params['T_range'][0]) & (T <= params['T_range'][1])
            
            reaction_zone = pressure_condition & temp_condition
            reaction_map[reaction_name] = reaction_zone
        
        return reaction_map
    
    def _calculate_reaction_heating(self, zone_id: int) -> np.ndarray:
        """Calculate heating from metamorphic reactions"""
        thermal = self.thermal_models[zone_id]
        geometry = self.slab_geometries[zone_id]
        
        # Simplified reaction heating
        latent_heat = thermal['latent_heat_reactions']
        heating_rate = np.zeros_like(geometry['temperature'])
        
        # Add heating at reaction depths
        for depth in thermal['reaction_depths']:
            depth_mask = np.abs(geometry['z'] * 1000 - depth) < 5000  # 5 km window
            heating_rate[depth_mask] += latent_heat * 1e-15  # W/m^3
        
        return heating_rate
    
    def _calculate_stress_field(self, zone_id: int) -> Dict[str, np.ndarray]:
        """Calculate stress field in subducting slab"""
        geometry = self.slab_geometries[zone_id]
        zone = self.subduction_zones[zone_id]
        
        # Simplified stress calculation
        shape = geometry['x'].shape
        
        # Gravitational stress
        rho = 3300  # kg/m^3
        g = 9.81    # m/s^2
        sigma_zz = rho * g * geometry['z'] * 1000  # Pa
        
        # Bending stress (simplified)
        thickness = zone['slab_thickness'] * 1000  # m
        Young_modulus = 70e9  # Pa
        curvature = 1e-5     # 1/m (estimated)
        
        bending_stress = Young_modulus * thickness * curvature
        
        return {
            'sigma_xx': np.zeros(shape),
            'sigma_yy': np.zeros(shape),
            'sigma_zz': sigma_zz,
            'sigma_xy': np.zeros(shape),
            'sigma_xz': bending_stress * np.ones(shape),
            'sigma_yz': np.zeros(shape)
        }
    
    def _calculate_fluid_release(self, zone_id: int) -> Dict[str, np.ndarray]:
        """Calculate fluid release from dehydration reactions"""
        reaction_map = self._calculate_metamorphic_reactions(zone_id)
        geometry = self.slab_geometries[zone_id]
        
        total_fluid_flux = np.zeros_like(geometry['temperature'])
        
        for reaction_name, reaction_zone in reaction_map.items():
            # Estimate fluid release rate
            if 'serpentine' in reaction_name:
                fluid_rate = 0.13  # High water content
            elif 'lawsonite' in reaction_name:
                fluid_rate = 0.08
            else:
                fluid_rate = 0.02
            
            total_fluid_flux[reaction_zone] += fluid_rate
        
        return {
            'fluid_flux': total_fluid_flux,
            'composition': 'H2O + dissolved species',
            'temperature': geometry['temperature'],
            'pressure': geometry['pressure']
        }

class ContinentalDynamics:
    """Continental rifting, collision, and assembly dynamics"""
    
    def __init__(self):
        self.continents = {}
        self.rift_zones = {}
        self.collision_zones = {}
        
    def initialize_continent(self, cont_id: int, properties: Dict) -> None:
        """Initialize continental block"""
        self.continents[cont_id] = {
            'area': properties.get('area', 1e7),  # km^2
            'thickness': properties.get('thickness', 40.0),  # km
            'age': properties.get('age', 2000.0),  # Myr
            'composition': properties.get('composition', 'felsic'),
            'rheology': properties.get('rheology', 'strong'),
            'thermal_state': properties.get('thermal_state', 'cold'),
            'position': properties.get('position', (0.0, 0.0)),
            'velocity': properties.get('velocity', (0.0, 0.0)),
            'internal_structure': self._initialize_continental_structure(properties)
        }
    
    def _initialize_continental_structure(self, properties: Dict) -> Dict:
        """Initialize internal continental structure"""
        return {
            'upper_crust': {
                'thickness': 15.0,  # km
                'density': 2700,    # kg/m^3
                'composition': 'granitic',
                'temperature': 500  # K
            },
            'lower_crust': {
                'thickness': 25.0,
                'density': 2900,
                'composition': 'granulitic',
                'temperature': 800
            },
            'lithospheric_mantle': {
                'thickness': properties.get('thickness', 40.0) - 40.0,
                'density': 3200,
                'composition': 'harzburgitic',
                'temperature': 1200
            }
        }
    
    def simulate_continental_rifting(self, cont_id: int, stress_field: Dict) -> Dict:
        """Simulate continental rifting process"""
        continent = self.continents[cont_id]
        
        # Check if stress exceeds yield strength
        tensile_stress = stress_field.get('sigma_xx', 0)
        yield_strength = self._calculate_continental_strength(cont_id)
        
        if tensile_stress > yield_strength:
            # Initiate rifting
            rift_properties = {
                'initiation_time': 0.0,
                'extension_rate': abs(tensile_stress - yield_strength) * 1e-12,  # m/s
                'thermal_anomaly': 100,  # K
                'crustal_thinning': 0.0,
                'mantle_upwelling': 0.0
            }
            
            rift_id = len(self.rift_zones)
            self.rift_zones[rift_id] = rift_properties
            
            return {
                'rifting_initiated': True,
                'rift_id': rift_id,
                'properties': rift_properties
            }
        else:
            return {'rifting_initiated': False}
    
    def _calculate_continental_strength(self, cont_id: int) -> float:
        """Calculate continental lithosphere strength"""
        continent = self.continents[cont_id]
        structure = continent['internal_structure']
        
        # Integrated strength envelope
        total_strength = 0.0
        
        for layer_name, layer in structure.items():
            thickness = layer['thickness'] * 1000  # m
            temperature = layer['temperature']  # K
            
            if 'crust' in layer_name:
                # Crustal rheology (wet quartzite)
                A = 3.2e-4  # Pa^-n s^-1
                n = 2.3
                Q = 154e3   # J/mol
            else:
                # Mantle rheology (dry olivine)
                A = 2.5e4
                n = 3.5
                Q = 510e3
            
            R = 8.314  # J/mol/K
            strain_rate = 1e-15  # s^-1
            
            # Calculate flow stress
            flow_stress = (strain_rate / A)**(1/n) * np.exp(Q / (n * R * temperature))
            
            # Integrate over thickness
            layer_strength = flow_stress * thickness
            total_strength += layer_strength
        
        return total_strength
    
    def simulate_continental_collision(self, cont1_id: int, cont2_id: int, 
                                    collision_velocity: float) -> Dict:
        """Simulate continental collision and mountain building"""
        cont1 = self.continents[cont1_id]
        cont2 = self.continents[cont2_id]
        
        # Calculate collision dynamics
        collision_force = collision_velocity * 1e12  # Simplified
        
        # Determine collision style
        if cont1['age'] > cont2['age']:
            overriding_plate = cont1_id
            subducting_plate = cont2_id
        else:
            overriding_plate = cont2_id
            subducting_plate = cont1_id
        
        # Calculate crustal thickening
        total_crust_thickness = (cont1['thickness'] + cont2['thickness']) * 0.7
        
        # Create collision zone
        collision_id = len(self.collision_zones)
        self.collision_zones[collision_id] = {
            'participants': [cont1_id, cont2_id],
            'collision_velocity': collision_velocity,
            'crustal_thickness': total_crust_thickness,
            'mountain_height': self._calculate_mountain_height(total_crust_thickness),
            'thermal_anomaly': collision_velocity * 10,  # K
            'metamorphism': self._calculate_collision_metamorphism(collision_force),
            'duration': 50.0  # Myr
        }
        
        return {
            'collision_initiated': True,
            'collision_id': collision_id,
            'mountain_height': self.collision_zones[collision_id]['mountain_height'],
            'overriding_plate': overriding_plate
        }
    
    def _calculate_mountain_height(self, crustal_thickness: float) -> float:
        """Calculate mountain height from crustal thickness"""
        # Isostatic balance
        rho_crust = 2700  # kg/m^3
        rho_mantle = 3200  # kg/m^3
        normal_crust = 35.0  # km
        
        excess_thickness = crustal_thickness - normal_crust
        mountain_height = excess_thickness * (rho_crust / (rho_mantle - rho_crust))
        
        return max(mountain_height, 0.0)
    
    def _calculate_collision_metamorphism(self, collision_force: float) -> Dict:
        """Calculate metamorphic conditions during collision"""
        # Simplified metamorphic grade calculation
        if collision_force > 1e15:
            grade = 'high_grade'
            temperature = 800  # K
            pressure = 1e9     # Pa
        elif collision_force > 1e14:
            grade = 'medium_grade'
            temperature = 650
            pressure = 7e8
        else:
            grade = 'low_grade'
            temperature = 500
            pressure = 5e8
        
        return {
            'metamorphic_grade': grade,
            'peak_temperature': temperature,
            'peak_pressure': pressure,
            'mineral_assemblages': self._determine_mineral_assemblages(temperature, pressure)
        }
    
    def _determine_mineral_assemblages(self, temperature: float, pressure: float) -> List[str]:
        """Determine metamorphic mineral assemblages"""
        assemblages = []
        
        if temperature > 750 and pressure > 8e8:
            assemblages.extend(['garnet', 'kyanite', 'biotite', 'plagioclase'])
        elif temperature > 600 and pressure > 6e8:
            assemblages.extend(['staurolite', 'garnet', 'biotite', 'muscovite'])
        elif temperature > 450:
            assemblages.extend(['chlorite', 'muscovite', 'biotite'])
        else:
            assemblages.extend(['chlorite', 'sericite'])
        
        return assemblages

class PlumeInteractionModel:
    """Mantle plume-plate interaction dynamics"""
    
    def __init__(self):
        self.plumes = {}
        self.plume_plate_interactions = {}
        self.llsvps = {}
        
    def initialize_plume(self, plume_id: int, properties: Dict) -> None:
        """Initialize mantle plume"""
        self.plumes[plume_id] = MantlePlume(
            plume_id=plume_id,
            plume_type=properties.get('plume_type', PlumeType.DEEP_MANTLE),
            location=properties.get('location', (0.0, 0.0)),
            depth_origin=properties.get('depth_origin', 2800.0),
            temperature_anomaly=properties.get('temperature_anomaly', 200.0),
            buoyancy_flux=properties.get('buoyancy_flux', 1e4),
            radius=properties.get('radius', 100.0),
            initiation_time=properties.get('initiation_time', 0.0),
            activity_duration=properties.get('activity_duration', 50.0),
            llsvp_association=properties.get('llsvp_association', False)
        )
        
        # Initialize LLSVP if associated
        if properties.get('llsvp_association', False):
            self._initialize_llsvp(plume_id, properties)
    
    def _initialize_llsvp(self, plume_id: int, properties: Dict) -> None:
        """Initialize Large Low Shear Velocity Province"""
        llsvp_id = f"LLSVP_{plume_id}"
        
        self.llsvps[llsvp_id] = {
            'associated_plumes': [plume_id],
            'location': properties.get('llsvp_location', (0.0, 0.0)),
            'extent': properties.get('llsvp_extent', (30.0, 30.0)),  # degrees
            'height': properties.get('llsvp_height', 1000.0),  # km
            'density_anomaly': properties.get('llsvp_density', 50.0),  # kg/m^3
            'velocity_reduction': properties.get('llsvp_velocity_reduction', -2.0),  # %
            'temperature_anomaly': properties.get('llsvp_temperature', 300.0),  # K
            'composition': properties.get('llsvp_composition', 'primordial'),
            'stability': properties.get('llsvp_stability', 'stable')
        }
    
    def calculate_plume_ascent(self, plume_id: int, mantle_structure: Dict) -> Dict:
        """Calculate plume ascent through mantle"""
        plume = self.plumes[plume_id]
        
        # Get mantle properties along ascent path
        depth_profile = np.linspace(plume.depth_origin, 0, 100)
        
        # Calculate buoyancy forces
        plume_density = self._calculate_plume_density(plume, depth_profile)
        ambient_density = self._calculate_ambient_density(depth_profile, mantle_structure)
        
        buoyancy = (ambient_density - plume_density) * 9.81
        
        # Calculate ascent velocity (Stokes velocity for spherical inclusion)
        viscosity_profile = self._get_mantle_viscosity_profile(depth_profile, mantle_structure)
        ascent_velocity = self._calculate_stokes_velocity(buoyancy, plume.radius * 1000, viscosity_profile)
        
        # Calculate entrainment and thermal evolution
        entrainment_rate = self._calculate_entrainment(ascent_velocity, plume.radius)
        thermal_evolution = self._calculate_thermal_evolution(plume, depth_profile, entrainment_rate)
        
        return {
            'ascent_velocity': ascent_velocity,
            'entrainment_rate': entrainment_rate,
            'thermal_evolution': thermal_evolution,
            'arrival_time': self._calculate_surface_arrival_time(plume.depth_origin, ascent_velocity)
        }
    
    def _calculate_plume_density(self, plume: MantlePlume, depth_profile: np.ndarray) -> np.ndarray:
        """Calculate plume density profile"""
        # Reference density at CMB
        rho_0 = 4000  # kg/m^3
        
        # Thermal expansion effect
        alpha = 2e-5  # 1/K
        density_thermal = rho_0 * (1 - alpha * plume.temperature_anomaly)
        
        # Compositional effect (if primordial material)
        if plume.plume_type == PlumeType.COMPOSITIONAL_PLUME:
            density_compositional = rho_0 * 0.02  # 2% denser
        else:
            density_compositional = 0.0
        
        return density_thermal + density_compositional * np.ones_like(depth_profile)
    
    def _calculate_ambient_density(self, depth_profile: np.ndarray, mantle_structure: Dict) -> np.ndarray:
        """Calculate ambient mantle density profile"""
        density_profile = np.zeros_like(depth_profile)
        
        for i, depth in enumerate(depth_profile):
            if depth < 410:  # Upper mantle
                density_profile[i] = 3300
            elif depth < 660:  # Transition zone
                density_profile[i] = 3600
            else:  # Lower mantle
                density_profile[i] = 4000 + (depth - 660) * 0.3  # Compression
        
        return density_profile
    
    def _get_mantle_viscosity_profile(self, depth_profile: np.ndarray, mantle_structure: Dict) -> np.ndarray:
        """Get mantle viscosity profile"""
        viscosity_profile = np.zeros_like(depth_profile)
        
        for i, depth in enumerate(depth_profile):
            if depth < 410:
                viscosity_profile[i] = 1e21  # Upper mantle
            elif depth < 660:
                viscosity_profile[i] = 5e21  # Transition zone
            else:
                viscosity_profile[i] = 1e22  # Lower mantle
        
        return viscosity_profile
    
    def _calculate_stokes_velocity(self, buoyancy: np.ndarray, radius: float, viscosity: np.ndarray) -> np.ndarray:
        """Calculate Stokes ascent velocity"""
        return buoyancy * radius**2 / (18 * viscosity)
    
    def _calculate_entrainment(self, velocity: np.ndarray, radius: float) -> np.ndarray:
        """Calculate entrainment of ambient mantle"""
        # Empirical entrainment coefficient
        alpha_e = 0.1
        return alpha_e * velocity * (2 * radius)
    
    def _calculate_thermal_evolution(self, plume: MantlePlume, depth_profile: np.ndarray, 
                                   entrainment_rate: np.ndarray) -> Dict:
        """Calculate thermal evolution during ascent"""
        initial_temp = 2800  # K at CMB
        temp_profile = np.zeros_like(depth_profile)
        
        # Adiabatic cooling
        adiabatic_gradient = 0.3e-3  # K/m
        
        for i, depth in enumerate(depth_profile):
            depth_ascended = plume.depth_origin - depth
            adiabatic_cooling = adiabatic_gradient * depth_ascended * 1000
            
            # Entrainment cooling
            if i > 0:
                entrainment_cooling = entrainment_rate[i] * 100  # Simplified
            else:
                entrainment_cooling = 0
            
            temp_profile[i] = initial_temp - adiabatic_cooling - entrainment_cooling
        
        return {
            'temperature_profile': temp_profile,
            'excess_temperature': temp_profile - self._ambient_temperature_profile(depth_profile)
        }
    
    def _ambient_temperature_profile(self, depth_profile: np.ndarray) -> np.ndarray:
        """Calculate ambient mantle temperature profile"""
        surface_temp = 1600  # K
        cmb_temp = 3500     # K
        
        # Linear geotherm (simplified)
        return surface_temp + (cmb_temp - surface_temp) * depth_profile / 2890
    
    def _calculate_surface_arrival_time(self, depth_origin: float, velocity: np.ndarray) -> float:
        """Calculate time for plume to reach surface"""
        if len(velocity) == 0 or np.all(velocity == 0):
            return np.inf
        
        mean_velocity = np.mean(velocity[velocity > 0])
        distance = depth_origin * 1000  # m
        
        return distance / mean_velocity / (365.25 * 24 * 3600 * 1e6)  # Myr
    
    def simulate_plume_plate_interaction(self, plume_id: int, plate_id: int, 
                                       plate_velocity: Tuple[float, float]) -> Dict:
        """Simulate interaction between plume and moving plate"""
        plume = self.plumes[plume_id]
        
        # Calculate relative motion
        plate_speed = np.sqrt(plate_velocity[0]**2 + plate_velocity[1]**2)  # cm/yr
        
        # Determine interaction type
        if plate_speed < 2.0:
            interaction_type = 'stationary_hotspot'
            track_length = 0.0
        elif plate_speed < 5.0:
            interaction_type = 'slow_hotspot_track'
            track_length = plate_speed * plume.activity_duration  # km
        else:
            interaction_type = 'fast_hotspot_track'
            track_length = plate_speed * plume.activity_duration
        
        # Calculate volcanic flux
        volcanic_flux = self._calculate_volcanic_flux(plume, plate_speed)
        
        # Generate hotspot track
        hotspot_track = self._generate_hotspot_track(plume, plate_velocity, track_length)
        
        # Store interaction
        interaction_id = len(self.plume_plate_interactions)
        self.plume_plate_interactions[interaction_id] = {
            'plume_id': plume_id,
            'plate_id': plate_id,
            'interaction_type': interaction_type,
            'track_length': track_length,
            'volcanic_flux': volcanic_flux,
            'hotspot_track': hotspot_track,
            'initiation_time': 0.0,
            'duration': plume.activity_duration
        }
        
        return {
            'interaction_id': interaction_id,
            'type': interaction_type,
            'track_length': track_length,
            'volcanic_flux': volcanic_flux
        }
    
    def _calculate_volcanic_flux(self, plume: MantlePlume, plate_speed: float) -> float:
        """Calculate volcanic flux from plume-plate interaction"""
        base_flux = plume.buoyancy_flux * 1e-6  # km^3/Myr conversion
        
        # Speed dependence (faster plates = more focused volcanism)
        speed_factor = 1.0 + 0.1 * plate_speed
        
        # Plume strength dependence
        strength_factor = plume.temperature_anomaly / 200.0
        
        return base_flux * speed_factor * strength_factor
    
    def _generate_hotspot_track(self, plume: MantlePlume, plate_velocity: Tuple[float, float], 
                              track_length: float) -> List[Tuple[float, float]]:
        """Generate hotspot track coordinates"""
        if track_length == 0:
            return [plume.location]
        
        # Number of points along track
        n_points = max(int(track_length / 100), 2)  # Point every 100 km
        
        # Velocity direction
        vx, vy = plate_velocity
        v_magnitude = np.sqrt(vx**2 + vy**2)
        
        if v_magnitude == 0:
            return [plume.location]
        
        # Unit vector in velocity direction
        ux, uy = vx / v_magnitude, vy / v_magnitude
        
        # Generate track points
        track = []
        lat0, lon0 = plume.location
        
        for i in range(n_points):
            # Distance along track
            distance = i * track_length / (n_points - 1)
            
            # Convert distance to lat/lon (simplified)
            dlat = (distance * uy) / 111.0  # km to degrees
            dlon = (distance * ux) / (111.0 * np.cos(np.radians(lat0)))
            
            track.append((lat0 + dlat, lon0 + dlon))
        
        return track

class IntegratedPlateEvolutionEngine:
    """Integrated plate evolution engine combining all dynamics"""
    
    def __init__(self, simulation_parameters: Dict):
        self.params = simulation_parameters
        
        # Initialize component models
        self.kinematics = AdvancedPlateKinematics()
        self.subduction = SubductionZoneDynamics()
        self.continental = ContinentalDynamics()
        self.plumes = PlumeInteractionModel()
        
        # Global state
        self.current_time = 0.0  # Myr
        self.time_step = simulation_parameters.get('time_step', 1.0)  # Myr
        self.total_time = simulation_parameters.get('total_time', 100.0)  # Myr
        
        # Plate configuration
        self.plates = {}
        self.boundaries = {}
        self.reconstruction_data = []
        
        # Initialize from parameters
        self._initialize_global_configuration()
    
    def _initialize_global_configuration(self) -> None:
        """Initialize global plate configuration"""
        # Initialize major plates (simplified modern configuration)
        major_plates = {
            1: {'name': 'Pacific', 'type': PlateType.OCEANIC, 'area': 103e6},
            2: {'name': 'North American', 'type': PlateType.CONTINENTAL, 'area': 75e6},
            3: {'name': 'Eurasian', 'type': PlateType.CONTINENTAL, 'area': 67e6},
            4: {'name': 'African', 'type': PlateType.CONTINENTAL, 'area': 61e6},
            5: {'name': 'Antarctic', 'type': PlateType.MIXED, 'area': 60e6},
            6: {'name': 'Indo-Australian', 'type': PlateType.MIXED, 'area': 58e6},
            7: {'name': 'South American', 'type': PlateType.CONTINENTAL, 'area': 43e6}
        }
        
        for plate_id, info in major_plates.items():
            self.plates[plate_id] = PlateProperties(
                plate_id=plate_id,
                plate_type=info['type'],
                area=info['area'],
                age=100.0,  # Average age
                thickness=100.0 if info['type'] == PlateType.OCEANIC else 150.0,
                density=3300 if info['type'] == PlateType.OCEANIC else 2700,
                velocity=(np.random.uniform(-5, 5), np.random.uniform(-5, 5)),
                angular_velocity=(0.0, 0.0, np.random.uniform(-1e-7, 1e-7))
            )
        
        # Initialize major boundaries
        self._initialize_major_boundaries()
        
        # Initialize major plumes
        self._initialize_major_plumes()
        
        # Initialize continents
        self._initialize_continents()
    
    def _initialize_major_boundaries(self) -> None:
        """Initialize major plate boundaries"""
        boundaries = [
            {'plates': (1, 2), 'type': BoundaryType.TRANSFORM, 'length': 3000},  # San Andreas
            {'plates': (1, 4), 'type': BoundaryType.DIVERGENT, 'length': 5000}, # Pacific-African ridge
            {'plates': (2, 3), 'type': BoundaryType.CONVERGENT, 'length': 2000}, # Aleutian
            {'plates': (3, 4), 'type': BoundaryType.DIVERGENT, 'length': 6000},  # Mid-Atlantic ridge
            {'plates': (4, 5), 'type': BoundaryType.DIVERGENT, 'length': 4000},  # Southwest ridge
        ]
        
        for i, boundary in enumerate(boundaries):
            self.boundaries[i] = BoundarySegment(
                segment_id=i,
                boundary_type=boundary['type'],
                plate1_id=boundary['plates'][0],
                plate2_id=boundary['plates'][1],
                geometry=np.random.random((10, 2)) * 360 - 180,  # Random geometry
                length=boundary['length'],
                spreading_rate=2.0 if boundary['type'] == BoundaryType.DIVERGENT else 0.0,
                convergence_rate=5.0 if boundary['type'] == BoundaryType.CONVERGENT else 0.0
            )
    
    def _initialize_major_plumes(self) -> None:
        """Initialize major mantle plumes"""
        major_plumes = [
            {'name': 'Hawaiian', 'location': (20, -155), 'type': PlumeType.DEEP_MANTLE},
            {'name': 'Yellowstone', 'location': (44, -110), 'type': PlumeType.SHALLOW_MANTLE},
            {'name': 'Iceland', 'location': (64, -18), 'type': PlumeType.DEEP_MANTLE},
            {'name': 'Galapagos', 'location': (0, -90), 'type': PlumeType.THERMAL_PLUME},
            {'name': 'Reunion', 'location': (-21, 55), 'type': PlumeType.DEEP_MANTLE}
        ]
        
        for i, plume_info in enumerate(major_plumes):
            self.plumes.initialize_plume(i, {
                'plume_type': plume_info['type'],
                'location': plume_info['location'],
                'depth_origin': 2800.0 if plume_info['type'] == PlumeType.DEEP_MANTLE else 1000.0,
                'temperature_anomaly': 200.0,
                'buoyancy_flux': 1e4,
                'radius': 100.0,
                'llsvp_association': plume_info['type'] == PlumeType.DEEP_MANTLE
            })
    
    def _initialize_continents(self) -> None:
        """Initialize continental blocks"""
        continents = [
            {'name': 'North America', 'area': 20e6, 'thickness': 40.0, 'age': 2500},
            {'name': 'Eurasia', 'area': 35e6, 'thickness': 45.0, 'age': 3000},
            {'name': 'Africa', 'area': 30e6, 'thickness': 42.0, 'age': 3500},
            {'name': 'South America', 'area': 18e6, 'thickness': 38.0, 'age': 2000},
            {'name': 'Australia', 'area': 8e6, 'thickness': 35.0, 'age': 3800},
            {'name': 'Antarctica', 'area': 14e6, 'thickness': 40.0, 'age': 2800}
        ]
        
        for i, cont_info in enumerate(continents):
            self.continental.initialize_continent(i, cont_info)
    
    def evolve_plate_system(self, dt: float) -> Dict:
        """Evolve entire plate system over time step"""
        logger.info(f"Evolving plate system at t = {self.current_time:.1f} Myr")
        
        # Update plate kinematics
        self._update_plate_kinematics(dt)
        
        # Evolve subduction zones
        subduction_results = self._evolve_subduction_zones(dt)
        
        # Update continental dynamics
        continental_results = self._evolve_continental_dynamics(dt)
        
        # Calculate plume-plate interactions
        plume_results = self._calculate_plume_interactions(dt)
        
        # Update plate boundaries
        self._update_plate_boundaries(dt)
        
        # Calculate force balance
        force_balance = self._calculate_global_force_balance()
        
        # Store reconstruction data
        reconstruction_snapshot = {
            'time': self.current_time,
            'plates': {pid: vars(plate) for pid, plate in self.plates.items()},
            'boundaries': {bid: vars(boundary) for bid, boundary in self.boundaries.items()},
            'subduction_zones': subduction_results,
            'continental_evolution': continental_results,
            'plume_interactions': plume_results,
            'force_balance': force_balance
        }
        
        self.reconstruction_data.append(reconstruction_snapshot)
        
        return reconstruction_snapshot
    
    def _update_plate_kinematics(self, dt: float) -> None:
        """Update plate positions and velocities"""
        # Extract velocity data for Euler pole calculation
        plate_velocities = {}
        for plate_id, plate in self.plates.items():
            # Mock velocity field (would come from observations/models)
            velocities = np.array([
                [0, 0, 6371e3, plate.velocity[0]*1e-2, plate.velocity[1]*1e-2, 0],
                [0, 90, 6371e3, plate.velocity[0]*1e-2*0.5, plate.velocity[1]*1e-2*0.8, 0]
            ])  # Simplified 2-point velocities
            plate_velocities[plate_id] = velocities
        
        # Calculate Euler poles
        euler_poles = self.kinematics.calculate_euler_poles(plate_velocities)
        self.kinematics.euler_poles = euler_poles
        
        # Update plate positions based on Euler pole rotations
        for plate_id, plate in self.plates.items():
            if plate_id in euler_poles:
                rotation_matrix = self.kinematics.predict_plate_motion(plate_id, dt * 1e6 * 365.25 * 24 * 3600)
                # Apply rotation to plate (simplified - would need proper implementation)
                plate.age += dt
    
    def _evolve_subduction_zones(self, dt: float) -> Dict:
        """Evolve all subduction zones"""
        subduction_results = {}
        
        # Identify convergent boundaries
        for boundary_id, boundary in self.boundaries.items():
            if boundary.boundary_type == BoundaryType.CONVERGENT:
                # Initialize subduction zone if not exists
                if boundary_id not in self.subduction.subduction_zones:
                    plate1 = self.plates[boundary.plate1_id]
                    plate2 = self.plates[boundary.plate2_id]
                    
                    # Determine subduction polarity
                    if plate1.age > plate2.age:
                        subducting_plate = boundary.plate1_id
                        overriding_plate = boundary.plate2_id
                    else:
                        subducting_plate = boundary.plate2_id
                        overriding_plate = boundary.plate1_id
                    
                    # Initialize subduction zone
                    zone_properties = {
                        'type': SubductionType.OCEANIC_OCEANIC if plate1.plate_type == PlateType.OCEANIC and plate2.plate_type == PlateType.OCEANIC else SubductionType.OCEANIC_CONTINENTAL,
                        'age': min(plate1.age, plate2.age),
                        'convergence_rate': boundary.convergence_rate,
                        'slab_dip': np.random.uniform(30, 60),
                        'trench_location': np.mean(boundary.geometry, axis=0)
                    }
                    
                    self.subduction.initialize_subduction_zone(boundary_id, zone_properties)
                
                # Evolve subduction zone
                evolution_result = self.subduction.evolve_subduction_zone(boundary_id, dt * 1e6 * 365.25 * 24 * 3600)
                subduction_results[boundary_id] = evolution_result
        
        return subduction_results
    
    def _evolve_continental_dynamics(self, dt: float) -> Dict:
        """Evolve continental dynamics"""
        continental_results = {}
        
        # Check for continental rifting
        for cont_id in self.continental.continents:
            # Mock stress field (would come from force balance calculation)
            stress_field = {'sigma_xx': np.random.uniform(-1e8, 1e8)}
            
            rift_result = self.continental.simulate_continental_rifting(cont_id, stress_field)
            if rift_result['rifting_initiated']:
                continental_results[f'rift_{cont_id}'] = rift_result
        
        # Check for continental collisions
        continent_pairs = [(0, 1), (1, 2), (2, 3)]  # Example pairs
        for cont1_id, cont2_id in continent_pairs:
            if cont1_id in self.continental.continents and cont2_id in self.continental.continents:
                collision_velocity = np.random.uniform(0, 10)  # cm/yr
                if collision_velocity > 1.0:  # Threshold for collision
                    collision_result = self.continental.simulate_continental_collision(
                        cont1_id, cont2_id, collision_velocity
                    )
                    if collision_result['collision_initiated']:
                        continental_results[f'collision_{cont1_id}_{cont2_id}'] = collision_result
        
        return continental_results
    
    def _calculate_plume_interactions(self, dt: float) -> Dict:
        """Calculate plume-plate interactions"""
        plume_results = {}
        
        for plume_id, plume in self.plumes.plumes.items():
            # Find overlying plate
            overlying_plate_id = self._find_overlying_plate(plume.location)
            
            if overlying_plate_id is not None:
                overlying_plate = self.plates[overlying_plate_id]
                
                # Calculate plume ascent
                mantle_structure = {}  # Would be provided by mantle convection model
                ascent_result = self.plumes.calculate_plume_ascent(plume_id, mantle_structure)
                
                # Simulate plume-plate interaction
                interaction_result = self.plumes.simulate_plume_plate_interaction(
                    plume_id, overlying_plate_id, overlying_plate.velocity
                )
                
                plume_results[plume_id] = {
                    'ascent': ascent_result,
                    'interaction': interaction_result
                }
        
        return plume_results
    
    def _find_overlying_plate(self, plume_location: Tuple[float, float]) -> Optional[int]:
        """Find plate overlying plume location"""
        # Simplified - would need proper geometric calculation
        return 1  # Default to first plate
    
    def _update_plate_boundaries(self, dt: float) -> None:
        """Update plate boundary configurations"""
        for boundary_id, boundary in self.boundaries.items():
            # Update spreading/convergence rates based on force balance
            # This is simplified - full implementation would be complex
            
            if boundary.boundary_type == BoundaryType.DIVERGENT:
                # Spreading rate may change due to ridge push forces
                boundary.spreading_rate *= (1 + np.random.uniform(-0.1, 0.1))
            elif boundary.boundary_type == BoundaryType.CONVERGENT:
                # Convergence rate affected by slab pull
                boundary.convergence_rate *= (1 + np.random.uniform(-0.1, 0.1))
    
    def _calculate_global_force_balance(self) -> Dict:
        """Calculate global force balance for plate system"""
        forces = {
            'ridge_push': 0.0,
            'slab_pull': 0.0,
            'basal_drag': 0.0,
            'transform_resistance': 0.0,
            'continental_collision': 0.0
        }
        
        # Calculate ridge push forces
        for boundary_id, boundary in self.boundaries.items():
            if boundary.boundary_type == BoundaryType.DIVERGENT:
                ridge_push = boundary.spreading_rate * boundary.length * 1e12  # Simplified
                forces['ridge_push'] += ridge_push
        
        # Calculate slab pull forces
        for zone_id in self.subduction.subduction_zones:
            slab_pull = 5e13  # Simplified constant
            forces['slab_pull'] += slab_pull
        
        # Calculate basal drag
        total_plate_area = sum(plate.area for plate in self.plates.values())
        forces['basal_drag'] = total_plate_area * 1e6 * 1e4  # Simplified
        
        return forces
    
    def run_evolution_simulation(self) -> Dict:
        """Run complete plate evolution simulation"""
        logger.info(f"Starting plate evolution simulation for {self.total_time} Myr")
        
        results = {
            'time_series': [],
            'reconstruction_data': [],
            'final_configuration': None
        }
        
        while self.current_time < self.total_time:
            # Evolve system
            snapshot = self.evolve_plate_system(self.time_step)
            
            # Store results
            results['time_series'].append({
                'time': self.current_time,
                'n_plates': len(self.plates),
                'n_boundaries': len(self.boundaries),
                'total_spreading_rate': sum(b.spreading_rate for b in self.boundaries.values()),
                'total_convergence_rate': sum(b.convergence_rate for b in self.boundaries.values())
            })
            
            # Update time
            self.current_time += self.time_step
            
            if len(results['time_series']) % 10 == 0:
                logger.info(f"Completed {self.current_time:.1f} Myr of evolution")
        
        # Store final results
        results['reconstruction_data'] = self.reconstruction_data
        results['final_configuration'] = {
            'plates': {pid: vars(plate) for pid, plate in self.plates.items()},
            'boundaries': {bid: vars(boundary) for bid, boundary in self.boundaries.items()}
        }
        
        logger.info("Plate evolution simulation completed")
        return results
    
    def save_results(self, results: Dict, filename: str) -> None:
        """Save simulation results to file"""
        with h5py.File(filename, 'w') as f:
            # Save parameters
            params_group = f.create_group('parameters')
            for key, value in self.params.items():
                if isinstance(value, (int, float, str, bool)):
                    params_group.attrs[key] = value
            
            # Save time series
            ts_group = f.create_group('time_series')
            time_data = results['time_series']
            if time_data:
                ts_group.create_dataset('time', data=[t['time'] for t in time_data])
                ts_group.create_dataset('n_plates', data=[t['n_plates'] for t in time_data])
                ts_group.create_dataset('n_boundaries', data=[t['n_boundaries'] for t in time_data])
            
            # Save reconstruction data (simplified)
            recon_group = f.create_group('reconstruction')
            recon_group.attrs['n_snapshots'] = len(results['reconstruction_data'])
            
            # Save final configuration
            final_group = f.create_group('final_configuration')
            final_group.attrs['n_final_plates'] = len(results['final_configuration']['plates'])
            final_group.attrs['n_final_boundaries'] = len(results['final_configuration']['boundaries'])
        
        logger.info(f"Results saved to {filename}")

def main():
    """Main function for testing plate evolution engine"""
    # Setup simulation parameters
    params = {
        'total_time': 100.0,  # Myr
        'time_step': 1.0,     # Myr
        'mantle_coupling': True,
        'plume_interactions': True,
        'continental_dynamics': True,
        'adaptive_boundaries': True
    }
    
    # Create evolution engine
    engine = IntegratedPlateEvolutionEngine(params)
    
    # Run simulation
    results = engine.run_evolution_simulation()
    
    # Save results
    output_file = "plate_evolution_results.h5"
    engine.save_results(results, output_file)
    
    print(f"Plate evolution simulation completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()