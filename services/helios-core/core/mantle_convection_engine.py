#!/usr/bin/env python3
"""
Advanced Mantle Convection and Plume Modeling Engine
Comprehensive 3D simulation of mantle dynamics and plume behavior
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)

@dataclass
class MantleLayer:
    """Mantle layer specification"""
    layer_id: str
    name: str
    depth_range: Tuple[float, float]  # km
    temperature_range: Tuple[float, float]  # K
    viscosity: float  # Pa·s
    density: float  # kg/m³
    thermal_conductivity: float  # W/(m·K)
    thermal_expansion: float  # K⁻¹
    heat_capacity: float  # J/(kg·K)
    phase_transitions: List[Dict[str, Any]]
    composition: Dict[str, float]  # major element composition

@dataclass
class MantlePlume:
    """Comprehensive mantle plume specification"""
    plume_id: str
    name: str
    source_location: Tuple[float, float, float]  # lat, lon, depth
    current_head_location: Tuple[float, float, float]
    tail_geometry: np.ndarray  # 3D coordinates of plume tail
    temperature_anomaly: float  # K above ambient
    buoyancy_flux: float  # m⁴/s²
    ascent_velocity: float  # m/year
    radius: float  # km
    age: float  # Ma
    geochemical_signature: Dict[str, float]
    surface_expression: Optional[str]  # hotspot name if any
    interaction_history: List[Dict[str, Any]]  # interactions with other plumes/plates

@dataclass
class ConvectionCell:
    """Mantle convection cell specification"""
    cell_id: str
    geometry: np.ndarray  # 3D cell boundaries
    flow_velocity: np.ndarray  # 3D velocity field
    temperature_field: np.ndarray  # 3D temperature distribution
    pressure_field: np.ndarray  # 3D pressure distribution
    upwelling_zones: List[Tuple[float, float, float]]
    downwelling_zones: List[Tuple[float, float, float]]
    circulation_pattern: str  # whole_mantle, layered, hybrid
    rayleigh_number: float
    nusselt_number: float

class MantleConvectionEngine:
    """Advanced mantle convection and plume modeling engine"""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        self.earth_radius = 6371.0  # km
        self.core_radius = 3485.0  # km
        self.mantle_layers = {}
        self.active_plumes = {}
        self.convection_cells = {}
        self.temperature_field = None
        self.velocity_field = None
        self.viscosity_field = None
        
        # Physical constants
        self.gravitational_acceleration = 9.81  # m/s²
        self.gas_constant = 8.314  # J/(mol·K)
        self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        
        # Initialize mantle structure
        self._initialize_mantle_structure()
        self._initialize_reference_plumes()
        self._initialize_convection_parameters()
    
    def _initialize_mantle_structure(self):
        """Initialize detailed mantle layer structure"""
        
        self.mantle_layers = {
            'upper_mantle': MantleLayer(
                layer_id='upper_mantle',
                name='Upper Mantle',
                depth_range=(0, 410),  # km
                temperature_range=(1600, 1900),  # K
                viscosity=1e19,  # Pa·s
                density=3300,  # kg/m³
                thermal_conductivity=4.0,  # W/(m·K)
                thermal_expansion=3e-5,  # K⁻¹
                heat_capacity=1200,  # J/(kg·K)
                phase_transitions=[
                    {'depth': 410, 'name': 'olivine_wadsleyite', 'clapeyron_slope': 2.5}
                ],
                composition={
                    'SiO2': 45.0, 'MgO': 37.8, 'FeO': 8.1, 'Al2O3': 4.5,
                    'CaO': 3.6, 'Na2O': 0.4, 'Cr2O3': 0.4, 'NiO': 0.2
                }
            ),
            'transition_zone': MantleLayer(
                layer_id='transition_zone',
                name='Transition Zone',
                depth_range=(410, 660),
                temperature_range=(1900, 2100),
                viscosity=1e20,
                density=3600,
                thermal_conductivity=4.5,
                thermal_expansion=2.5e-5,
                heat_capacity=1250,
                phase_transitions=[
                    {'depth': 520, 'name': 'wadsleyite_ringwoodite', 'clapeyron_slope': 1.5},
                    {'depth': 660, 'name': 'ringwoodite_perovskite', 'clapeyron_slope': -2.8}
                ],
                composition={
                    'SiO2': 45.0, 'MgO': 37.8, 'FeO': 8.1, 'Al2O3': 4.5,
                    'CaO': 3.6, 'Na2O': 0.4, 'Cr2O3': 0.4, 'NiO': 0.2
                }
            ),
            'lower_mantle': MantleLayer(
                layer_id='lower_mantle',
                name='Lower Mantle',
                depth_range=(660, 2890),
                temperature_range=(2100, 3000),
                viscosity=1e22,
                density=4500,
                thermal_conductivity=6.0,
                thermal_expansion=2e-5,
                heat_capacity=1300,
                phase_transitions=[
                    {'depth': 2700, 'name': 'post_perovskite', 'clapeyron_slope': 7.5}
                ],
                composition={
                    'SiO2': 45.0, 'MgO': 37.8, 'FeO': 8.1, 'Al2O3': 4.5,
                    'CaO': 3.6, 'Na2O': 0.4, 'Cr2O3': 0.4, 'NiO': 0.2
                }
            ),
            'd_double_prime': MantleLayer(
                layer_id='d_double_prime',
                name='D" Layer',
                depth_range=(2890, 2890),  # Interface layer
                temperature_range=(3000, 4000),
                viscosity=1e23,
                density=5500,
                thermal_conductivity=8.0,
                thermal_expansion=1.5e-5,
                heat_capacity=1400,
                phase_transitions=[],
                composition={
                    'SiO2': 45.0, 'MgO': 37.8, 'FeO': 8.1, 'Al2O3': 4.5,
                    'CaO': 3.6, 'Na2O': 0.4, 'Cr2O3': 0.4, 'NiO': 0.2
                }
            )
        }
    
    def _initialize_reference_plumes(self):
        """Initialize reference mantle plumes based on observations"""
        
        self.reference_plumes = {
            'hawaii': {
                'source_location': (19.5, -155.5, 2890),
                'surface_location': (19.5, -155.5, 0),
                'temperature_anomaly': 200,  # K
                'buoyancy_flux': 8500,  # kg/s
                'ascent_velocity': 20,  # m/year
                'radius': 100,  # km
                'age': 85,  # Ma
                'geochemistry': {
                    '3He/4He': 32.0,  # Ra
                    '87Sr/86Sr': 0.7035,
                    '143Nd/144Nd': 0.51315,
                    '206Pb/204Pb': 18.2,
                    'Nb/U': 47,
                    'Ce/Pb': 25
                }
            },
            'iceland': {
                'source_location': (64.1, -21.9, 2890),
                'surface_location': (64.1, -21.9, 0),
                'temperature_anomaly': 180,
                'buoyancy_flux': 12000,
                'ascent_velocity': 15,
                'radius': 150,
                'age': 62,
                'geochemistry': {
                    '3He/4He': 18.5,
                    '87Sr/86Sr': 0.7031,
                    '143Nd/144Nd': 0.51308,
                    '206Pb/204Pb': 18.8,
                    'Nb/U': 52,
                    'Ce/Pb': 28
                }
            },
            'yellowstone': {
                'source_location': (44.4, -110.6, 2890),
                'surface_location': (44.4, -110.6, 0),
                'temperature_anomaly': 150,
                'buoyancy_flux': 4500,
                'ascent_velocity': 25,
                'radius': 80,
                'age': 17,
                'geochemistry': {
                    '3He/4He': 12.0,
                    '87Sr/86Sr': 0.7065,
                    '143Nd/144Nd': 0.51285,
                    '206Pb/204Pb': 17.8,
                    'Nb/U': 35,
                    'Ce/Pb': 20
                }
            },
            'reunion': {
                'source_location': (-21.1, 55.5, 2890),
                'surface_location': (-21.1, 55.5, 0),
                'temperature_anomaly': 175,
                'buoyancy_flux': 6800,
                'ascent_velocity': 18,
                'radius': 120,
                'age': 68,
                'geochemistry': {
                    '3He/4He': 13.8,
                    '87Sr/86Sr': 0.7045,
                    '143Nd/144Nd': 0.51295,
                    '206Pb/204Pb': 18.5,
                    'Nb/U': 42,
                    'Ce/Pb': 23
                }
            }
        }
    
    async def simulate_mantle_convection(self, 
                                       simulation_time: float,
                                       resolution: Tuple[int, int, int] = (64, 64, 32),
                                       boundary_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Simulate 3D mantle convection with plume dynamics
        
        Args:
            simulation_time: Simulation duration in Ma
            resolution: Grid resolution (nr, ntheta, nphi)
            boundary_conditions: Optional boundary conditions
            
        Returns:
            Complete simulation results
        """
        
        logger.info(f"Starting mantle convection simulation for {simulation_time} Ma")
        
        # Initialize 3D spherical grid
        grid = await self._initialize_spherical_grid(resolution)
        
        # Initialize temperature and velocity fields
        initial_state = await self._initialize_convection_state(grid, boundary_conditions)
        
        # Set up governing equations
        equations = await self._setup_convection_equations(grid)
        
        # Time integration
        time_steps = int(simulation_time * 100)  # 0.01 Ma time steps
        dt = simulation_time / time_steps
        
        results = {
            'time_evolution': [],
            'plume_tracks': {},
            'convection_patterns': [],
            'heat_flow_evolution': [],
            'temperature_profiles': [],
            'velocity_profiles': []
        }
        
        current_state = initial_state
        
        for step in range(time_steps):
            time = step * dt
            
            # Solve convection equations
            new_state = await self._solve_convection_step(
                current_state, equations, dt, grid
            )
            
            # Update plume dynamics
            plume_state = await self._update_plume_dynamics(
                new_state, time, dt
            )
            
            # Calculate derived quantities
            derived_quantities = await self._calculate_derived_quantities(
                new_state, plume_state, grid
            )
            
            # Store results (every 10 steps to save memory)
            if step % 10 == 0:
                results['time_evolution'].append({
                    'time': time,
                    'temperature_field': new_state['temperature'].copy(),
                    'velocity_field': new_state['velocity'].copy(),
                    'pressure_field': new_state['pressure'].copy()
                })
                
                results['heat_flow_evolution'].append(derived_quantities['surface_heat_flow'])
                results['temperature_profiles'].append(derived_quantities['radial_temperature'])
                results['velocity_profiles'].append(derived_quantities['radial_velocity'])
            
            # Update plume tracks
            for plume_id, plume_data in plume_state.items():
                if plume_id not in results['plume_tracks']:
                    results['plume_tracks'][plume_id] = []
                results['plume_tracks'][plume_id].append({
                    'time': time,
                    'head_location': plume_data['head_location'],
                    'temperature_anomaly': plume_data['temperature_anomaly'],
                    'radius': plume_data['radius']
                })
            
            current_state = new_state
            
            if step % 100 == 0:
                logger.info(f"Completed convection step {step}/{time_steps} (t={time:.2f} Ma)")
        
        # Post-process results
        processed_results = await self._post_process_convection_results(results, grid)
        
        return processed_results
    
    async def _initialize_spherical_grid(self, resolution: Tuple[int, int, int]) -> Dict[str, Any]:
        """Initialize 3D spherical coordinate grid"""
        
        nr, ntheta, nphi = resolution
        
        # Radial coordinates (core-mantle boundary to surface)
        r_min = self.core_radius
        r_max = self.earth_radius
        r = np.linspace(r_min, r_max, nr)
        
        # Colatitude coordinates (0 to π)
        theta = np.linspace(0, np.pi, ntheta)
        
        # Azimuthal coordinates (0 to 2π)
        phi = np.linspace(0, 2*np.pi, nphi)
        
        # Create 3D meshgrid
        R, THETA, PHI = np.meshgrid(r, theta, phi, indexing='ij')
        
        # Convert to Cartesian coordinates for some calculations
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        
        return {
            'r': r, 'theta': theta, 'phi': phi,
            'R': R, 'THETA': THETA, 'PHI': PHI,
            'X': X, 'Y': Y, 'Z': Z,
            'resolution': resolution,
            'dr': r[1] - r[0],
            'dtheta': theta[1] - theta[0],
            'dphi': phi[1] - phi[0]
        }
    
    async def _initialize_convection_state(self, grid: Dict[str, Any], 
                                         boundary_conditions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize temperature and velocity fields"""
        
        R, THETA, PHI = grid['R'], grid['THETA'], grid['PHI']
        
        # Initialize temperature field with adiabatic profile + perturbations
        T_surface = 300  # K
        T_cmb = 4000     # K
        
        # Adiabatic temperature profile
        adiabatic_gradient = 0.5e-3  # K/km
        T_adiabatic = T_surface + adiabatic_gradient * (self.earth_radius - R)
        
        # Add thermal boundary layers
        T_field = T_adiabatic.copy()
        
        # Surface thermal boundary layer
        surface_bl_thickness = 100  # km
        surface_mask = (self.earth_radius - R) < surface_bl_thickness
        T_field[surface_mask] = T_surface + (T_adiabatic[surface_mask] - T_surface) * \
                               np.exp(-(self.earth_radius - R[surface_mask]) / 30)
        
        # Core-mantle boundary thermal boundary layer
        cmb_bl_thickness = 200  # km
        cmb_mask = (R - self.core_radius) < cmb_bl_thickness
        T_field[cmb_mask] = T_cmb - (T_cmb - T_adiabatic[cmb_mask]) * \
                           np.exp(-(R[cmb_mask] - self.core_radius) / 50)
        
        # Add random perturbations to seed convection
        perturbation_amplitude = 50  # K
        perturbations = perturbation_amplitude * np.random.normal(0, 1, T_field.shape)
        T_field += perturbations
        
        # Initialize velocity field (initially zero)
        v_r = np.zeros_like(R)
        v_theta = np.zeros_like(R)
        v_phi = np.zeros_like(R)
        
        # Initialize pressure field (hydrostatic)
        pressure = await self._calculate_hydrostatic_pressure(R, T_field)
        
        return {
            'temperature': T_field,
            'velocity': np.stack([v_r, v_theta, v_phi], axis=-1),
            'pressure': pressure
        }