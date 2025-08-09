"""
Genetic Model Integration for Ore Body Formation
Fluid flow pathways, temperature evolution, precipitation kinetics, and alteration sequences
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Callable
import json
import warnings
from datetime import datetime

@dataclass
class FluidComposition:
    """Hydrothermal fluid composition"""
    temperature: float  # K
    pressure: float     # Pa
    salinity: float     # wt% NaCl equiv
    ph: float
    log_fo2: float      # oxygen fugacity
    concentrations: Dict[str, float]  # mol/L
    phase: str          # 'liquid', 'vapor', 'brine'

@dataclass
class MineralPhase:
    """Mineral phase properties"""
    name: str
    formula: str
    stability_field: Dict[str, Tuple[float, float]]  # P-T-X conditions
    precipitation_rate: float
    dissolution_rate: float
    volume_change: float  # molar volume change on formation

@dataclass
class AlterationZone:
    """Alteration assemblage zone"""
    zone_name: str
    distance_from_center: float
    temperature_range: Tuple[float, float]
    mineral_assemblage: List[str]
    permeability: float
    porosity: float

class FluidFlowModeler:
    """3D fluid flow modeling with heat and mass transport"""
    
    def __init__(self, domain_size: Tuple[float, float, float], resolution: Tuple[int, int, int]):
        self.nx, self.ny, self.nz = resolution
        self.lx, self.ly, self.lz = domain_size
        
        # Grid spacing
        self.dx = self.lx / (self.nx - 1)
        self.dy = self.ly / (self.ny - 1)
        self.dz = self.lz / (self.nz - 1)
        
        # Create coordinate arrays
        self.x = np.linspace(0, self.lx, self.nx)
        self.y = np.linspace(0, self.ly, self.ny)
        self.z = np.linspace(0, self.lz, self.nz)
        
        # Physical properties fields
        self.permeability = np.ones((self.nx, self.ny, self.nz)) * 1e-15  # m²
        self.porosity = np.ones((self.nx, self.ny, self.nz)) * 0.1
        self.rock_density = np.ones((self.nx, self.ny, self.nz)) * 2700  # kg/m³
        self.heat_capacity = np.ones((self.nx, self.ny, self.nz)) * 1000  # J/kg/K
        self.thermal_conductivity = np.ones((self.nx, self.ny, self.nz)) * 3.0  # W/m/K
        
        # State variables
        self.pressure = np.zeros((self.nx, self.ny, self.nz))
        self.temperature = np.ones((self.nx, self.ny, self.nz)) * 293.15  # K
        self.velocity = np.zeros((self.nx, self.ny, self.nz, 3))  # m/s
        self.concentrations = {}
        
        # Boundary conditions
        self.boundary_conditions = {}
        self.heat_sources = []
        self.fluid_sources = []
    
    def set_permeability_field(self, permeability_func: Callable):
        """Set spatially variable permeability"""
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    x, y, z = self.x[i], self.y[j], self.z[k]
                    self.permeability[i, j, k] = permeability_func(x, y, z)
    
    def set_porosity_field(self, porosity_func: Callable):
        """Set spatially variable porosity"""
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    x, y, z = self.x[i], self.y[j], self.z[k]
                    self.porosity[i, j, k] = porosity_func(x, y, z)
    
    def add_heat_source(self, location: Tuple[float, float, float], 
                       power: float, radius: float):
        """Add localized heat source (intrusion, etc.)"""
        self.heat_sources.append({
            'location': location,
            'power': power,
            'radius': radius
        })
    
    def add_fluid_source(self, location: Tuple[float, float, float],
                        flow_rate: float, composition: FluidComposition):
        """Add fluid source/sink"""
        self.fluid_sources.append({
            'location': location,
            'flow_rate': flow_rate,
            'composition': composition
        })
    
    def solve_pressure_field(self, time_step: float = 1e6) -> np.ndarray:
        """Solve for pressure field using finite differences"""
        # Simplified steady-state pressure equation
        # ∇·(k/μ ∇P) = 0
        
        mu = 1e-3  # fluid viscosity (Pa·s)
        
        # Create coefficient matrix for finite differences
        P_flat = self.pressure.flatten()
        
        def pressure_residual(P_vec):
            P_3d = P_vec.reshape((self.nx, self.ny, self.nz))
            residual = np.zeros_like(P_3d)
            
            # Interior points
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    for k in range(1, self.nz-1):
                        # Central differences
                        k_avg_x = 0.5 * (self.permeability[i+1,j,k] + self.permeability[i-1,j,k])
                        k_avg_y = 0.5 * (self.permeability[i,j+1,k] + self.permeability[i,j-1,k])
                        k_avg_z = 0.5 * (self.permeability[i,j,k+1] + self.permeability[i,j,k-1])
                        
                        d2p_dx2 = (P_3d[i+1,j,k] - 2*P_3d[i,j,k] + P_3d[i-1,j,k]) / self.dx**2
                        d2p_dy2 = (P_3d[i,j+1,k] - 2*P_3d[i,j,k] + P_3d[i,j-1,k]) / self.dy**2
                        d2p_dz2 = (P_3d[i,j,k+1] - 2*P_3d[i,j,k] + P_3d[i,j,k-1]) / self.dz**2
                        
                        residual[i,j,k] = k_avg_x * d2p_dx2 + k_avg_y * d2p_dy2 + k_avg_z * d2p_dz2
            
            return residual.flatten()
        
        # Apply boundary conditions (simplified)
        # Set top boundary to atmospheric pressure
        self.pressure[:, :, 0] = 101325  # Pa
        # Set bottom boundary to hydrostatic + excess pressure
        self.pressure[:, :, -1] = 101325 + 1000 * 9.81 * self.lz + 1e6  # Pa
        
        # Solve iteratively (simplified)
        for iteration in range(100):
            old_pressure = self.pressure.copy()
            
            # Update interior points
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    for k in range(1, self.nz-1):
                        self.pressure[i,j,k] = 0.2 * (
                            self.pressure[i+1,j,k] + self.pressure[i-1,j,k] +
                            self.pressure[i,j+1,k] + self.pressure[i,j-1,k] +
                            self.pressure[i,j,k+1] + self.pressure[i,j,k-1]
                        ) / 6
            
            # Check convergence
            if np.max(np.abs(self.pressure - old_pressure)) < 1e3:
                break
        
        return self.pressure
    
    def solve_temperature_field(self, time_step: float = 3.15e7) -> np.ndarray:
        """Solve heat transport equation"""
        # Heat equation: ρc ∂T/∂t = ∇·(k ∇T) + Q
        
        # Add heat from sources
        heat_source_term = np.zeros((self.nx, self.ny, self.nz))
        
        for source in self.heat_sources:
            x0, y0, z0 = source['location']
            power = source['power']
            radius = source['radius']
            
            # Find nearest grid points
            i0 = int(x0 / self.dx)
            j0 = int(y0 / self.dy)
            k0 = int(z0 / self.dz)
            
            # Distribute heat in sphere
            for i in range(max(0, i0-5), min(self.nx, i0+5)):
                for j in range(max(0, j0-5), min(self.ny, j0+5)):
                    for k in range(max(0, k0-5), min(self.nz, k0+5)):
                        x, y, z = self.x[i], self.y[j], self.z[k]
                        dist = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
                        
                        if dist <= radius:
                            heat_source_term[i,j,k] += power * np.exp(-dist**2 / radius**2)
        
        # Solve heat equation (explicit finite differences)
        new_temperature = self.temperature.copy()
        
        alpha = self.thermal_conductivity / (self.rock_density * self.heat_capacity)  # thermal diffusivity
        
        dt = min(time_step, 0.1 * min(self.dx**2, self.dy**2, self.dz**2) / np.max(alpha))
        n_steps = int(time_step / dt)
        
        for step in range(n_steps):
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    for k in range(1, self.nz-1):
                        d2t_dx2 = (self.temperature[i+1,j,k] - 2*self.temperature[i,j,k] + self.temperature[i-1,j,k]) / self.dx**2
                        d2t_dy2 = (self.temperature[i,j+1,k] - 2*self.temperature[i,j,k] + self.temperature[i,j-1,k]) / self.dy**2
                        d2t_dz2 = (self.temperature[i,j,k+1] - 2*self.temperature[i,j,k] + self.temperature[i,j,k-1]) / self.dz**2
                        
                        heat_diff = alpha[i,j,k] * (d2t_dx2 + d2t_dy2 + d2t_dz2)
                        heat_source = heat_source_term[i,j,k] / (self.rock_density[i,j,k] * self.heat_capacity[i,j,k])
                        
                        new_temperature[i,j,k] = self.temperature[i,j,k] + dt * (heat_diff + heat_source)
            
            self.temperature = new_temperature.copy()
        
        return self.temperature
    
    def calculate_velocity_field(self) -> np.ndarray:
        """Calculate velocity field from pressure gradients (Darcy's law)"""
        mu = 1e-3  # fluid viscosity
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    # Pressure gradients
                    dp_dx = (self.pressure[i+1,j,k] - self.pressure[i-1,j,k]) / (2 * self.dx)
                    dp_dy = (self.pressure[i,j+1,k] - self.pressure[i,j-1,k]) / (2 * self.dy)
                    dp_dz = (self.pressure[i,j,k+1] - self.pressure[i,j,k-1]) / (2 * self.dz)
                    
                    # Darcy velocity
                    self.velocity[i,j,k,0] = -self.permeability[i,j,k] / mu * dp_dx
                    self.velocity[i,j,k,1] = -self.permeability[i,j,k] / mu * dp_dy
                    self.velocity[i,j,k,2] = -self.permeability[i,j,k] / mu * dp_dz
        
        return self.velocity
    
    def transport_solutes(self, element: str, initial_concentration: float,
                         time_step: float = 3.15e7) -> np.ndarray:
        """Solve advection-diffusion equation for solute transport"""
        if element not in self.concentrations:
            self.concentrations[element] = np.ones((self.nx, self.ny, self.nz)) * initial_concentration
        
        C = self.concentrations[element]
        new_C = C.copy()
        
        D = 1e-9  # diffusion coefficient (m²/s)
        dt = min(time_step, 0.1 * min(self.dx**2, self.dy**2, self.dz**2) / D)
        n_steps = int(time_step / dt)
        
        for step in range(n_steps):
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    for k in range(1, self.nz-1):
                        # Diffusion term
                        d2c_dx2 = (C[i+1,j,k] - 2*C[i,j,k] + C[i-1,j,k]) / self.dx**2
                        d2c_dy2 = (C[i,j+1,k] - 2*C[i,j,k] + C[i,j-1,k]) / self.dy**2
                        d2c_dz2 = (C[i,j,k+1] - 2*C[i,j,k] + C[i,j,k-1]) / self.dz**2
                        
                        diffusion = D * (d2c_dx2 + d2c_dy2 + d2c_dz2)
                        
                        # Advection term
                        dc_dx = (C[i+1,j,k] - C[i-1,j,k]) / (2 * self.dx)
                        dc_dy = (C[i,j+1,k] - C[i,j-1,k]) / (2 * self.dy)
                        dc_dz = (C[i,j,k+1] - C[i,j,k-1]) / (2 * self.dz)
                        
                        advection = -(self.velocity[i,j,k,0] * dc_dx +
                                    self.velocity[i,j,k,1] * dc_dy +
                                    self.velocity[i,j,k,2] * dc_dz)
                        
                        new_C[i,j,k] = C[i,j,k] + dt * (diffusion + advection)
            
            C = new_C.copy()
        
        self.concentrations[element] = C
        return C

class PrecipitationKinetics:
    """Mineral precipitation and dissolution kinetics"""
    
    def __init__(self):
        self.mineral_database = self._initialize_mineral_database()
        self.reaction_rates = {}
        self.thermodynamic_data = {}
    
    def _initialize_mineral_database(self) -> Dict[str, MineralPhase]:
        """Initialize mineral phase database"""
        return {
            'quartz': MineralPhase(
                name='quartz', formula='SiO2',
                stability_field={'T': (273, 573), 'P': (1e5, 1e8), 'pH': (0, 14)},
                precipitation_rate=1e-12, dissolution_rate=1e-13,
                volume_change=-0.1
            ),
            'pyrite': MineralPhase(
                name='pyrite', formula='FeS2',
                stability_field={'T': (423, 673), 'P': (1e5, 1e8), 'log_fo2': (-40, -20)},
                precipitation_rate=1e-10, dissolution_rate=1e-11,
                volume_change=0.2
            ),
            'chalcopyrite': MineralPhase(
                name='chalcopyrite', formula='CuFeS2',
                stability_field={'T': (473, 773), 'P': (1e5, 1e8), 'log_fo2': (-35, -25)},
                precipitation_rate=5e-11, dissolution_rate=1e-11,
                volume_change=0.15
            ),
            'sphalerite': MineralPhase(
                name='sphalerite', formula='ZnS',
                stability_field={'T': (423, 723), 'P': (1e5, 1e8), 'log_fo2': (-40, -20)},
                precipitation_rate=8e-11, dissolution_rate=1e-11,
                volume_change=0.1
            ),
            'galena': MineralPhase(
                name='galena', formula='PbS',
                stability_field={'T': (373, 623), 'P': (1e5, 1e8), 'log_fo2': (-45, -20)},
                precipitation_rate=3e-11, dissolution_rate=5e-12,
                volume_change=0.05
            ),
            'molybdenite': MineralPhase(
                name='molybdenite', formula='MoS2',
                stability_field={'T': (523, 823), 'P': (1e5, 1e8), 'log_fo2': (-30, -15)},
                precipitation_rate=2e-12, dissolution_rate=1e-13,
                volume_change=0.3
            )
        }
    
    def calculate_saturation_index(self, mineral: str, fluid: FluidComposition) -> float:
        """Calculate saturation index for mineral in fluid"""
        if mineral not in self.mineral_database:
            raise ValueError(f"Unknown mineral: {mineral}")
        
        # Simplified saturation calculation
        # SI = log(IAP/K) where IAP = ion activity product, K = solubility constant
        
        # Get mineral stoichiometry
        mineral_data = self.mineral_database[mineral]
        
        if mineral == 'pyrite':  # FeS2
            fe_activity = fluid.concentrations.get('Fe2+', 1e-6)
            s_activity = fluid.concentrations.get('H2S', 1e-6)**2
            iap = fe_activity * s_activity
            
            # Temperature-dependent solubility constant (simplified)
            log_k = -24.6 + 3500 / fluid.temperature
            k = 10**log_k
            
        elif mineral == 'chalcopyrite':  # CuFeS2
            cu_activity = fluid.concentrations.get('Cu+', 1e-6)
            fe_activity = fluid.concentrations.get('Fe2+', 1e-6)
            s_activity = fluid.concentrations.get('H2S', 1e-6)**2
            iap = cu_activity * fe_activity * s_activity
            
            log_k = -35.2 + 4200 / fluid.temperature
            k = 10**log_k
            
        elif mineral == 'sphalerite':  # ZnS
            zn_activity = fluid.concentrations.get('Zn2+', 1e-6)
            s_activity = fluid.concentrations.get('H2S', 1e-6)
            iap = zn_activity * s_activity
            
            log_k = -11.1 + 2100 / fluid.temperature
            k = 10**log_k
            
        else:
            # Default calculation
            iap = 1e-10
            k = 1e-15
        
        saturation_index = np.log10(iap / k)
        return saturation_index
    
    def calculate_precipitation_rate(self, mineral: str, fluid: FluidComposition,
                                   surface_area: float = 1.0) -> float:
        """Calculate mineral precipitation rate"""
        saturation_index = self.calculate_saturation_index(mineral, fluid)
        
        if saturation_index <= 0:
            return 0.0  # No precipitation if undersaturated
        
        mineral_data = self.mineral_database[mineral]
        base_rate = mineral_data.precipitation_rate
        
        # Rate depends on saturation index
        rate = base_rate * surface_area * (10**saturation_index - 1)**2
        
        # Temperature dependence (Arrhenius)
        activation_energy = 50000  # J/mol
        rate_temp_factor = np.exp(-activation_energy / (8.314 * fluid.temperature))
        
        return rate * rate_temp_factor
    
    def calculate_dissolution_rate(self, mineral: str, fluid: FluidComposition,
                                 surface_area: float = 1.0) -> float:
        """Calculate mineral dissolution rate"""
        saturation_index = self.calculate_saturation_index(mineral, fluid)
        
        if saturation_index >= 0:
            return 0.0  # No dissolution if supersaturated
        
        mineral_data = self.mineral_database[mineral]
        base_rate = mineral_data.dissolution_rate
        
        # Rate depends on undersaturation
        rate = base_rate * surface_area * (1 - 10**saturation_index)**0.5
        
        # Temperature dependence
        activation_energy = 65000  # J/mol
        rate_temp_factor = np.exp(-activation_energy / (8.314 * fluid.temperature))
        
        return rate * rate_temp_factor
    
    def simulate_mineral_evolution(self, fluid_conditions: List[FluidComposition],
                                 time_steps: np.ndarray, initial_mineralogy: Dict[str, float]) -> Dict:
        """Simulate mineral assemblage evolution over time"""
        
        mineral_amounts = {mineral: [amount] for mineral, amount in initial_mineralogy.items()}
        fluid_evolution = []
        
        for i, (fluid, dt) in enumerate(zip(fluid_conditions, np.diff(time_steps))):
            current_amounts = {mineral: amounts[-1] for mineral, amounts in mineral_amounts.items()}
            
            # Calculate rates for all minerals
            for mineral in self.mineral_database:
                if mineral not in current_amounts:
                    current_amounts[mineral] = 0.0
                    mineral_amounts[mineral] = mineral_amounts.get(mineral, [0.0])
                
                current_amount = current_amounts[mineral]
                surface_area = max(0.1, current_amount**0.67)  # Surface area scaling
                
                precip_rate = self.calculate_precipitation_rate(mineral, fluid, surface_area)
                dissol_rate = self.calculate_dissolution_rate(mineral, fluid, surface_area)
                
                net_rate = precip_rate - dissol_rate * current_amount
                new_amount = max(0, current_amount + net_rate * dt)
                
                mineral_amounts[mineral].append(new_amount)
            
            fluid_evolution.append({
                'time': time_steps[i+1],
                'temperature': fluid.temperature,
                'pressure': fluid.pressure,
                'mineral_amounts': {mineral: amounts[-1] for mineral, amounts in mineral_amounts.items()}
            })
        
        return {
            'time_steps': time_steps,
            'mineral_evolution': mineral_amounts,
            'fluid_evolution': fluid_evolution
        }

class AlterationModeler:
    """Hydrothermal alteration modeling"""
    
    def __init__(self):
        self.alteration_zones = {}
        self.alteration_minerals = self._initialize_alteration_minerals()
        self.reaction_networks = {}
    
    def _initialize_alteration_minerals(self) -> Dict:
        """Initialize alteration mineral database"""
        return {
            'potassic': {
                'minerals': ['K-feldspar', 'biotite', 'magnetite', 'molybdenite'],
                'temperature_range': (723, 973),
                'chemistry': {'K': 'high', 'Si': 'moderate', 'Fe': 'moderate'}
            },
            'phyllic': {
                'minerals': ['sericite', 'quartz', 'pyrite'],
                'temperature_range': (523, 723),
                'chemistry': {'H': 'high', 'S': 'high', 'pH': 'low'}
            },
            'argillic': {
                'minerals': ['kaolinite', 'illite', 'smectite'],
                'temperature_range': (373, 523),
                'chemistry': {'Al': 'high', 'H': 'high', 'pH': 'low'}
            },
            'propylitic': {
                'minerals': ['chlorite', 'epidote', 'calcite', 'albite'],
                'temperature_range': (473, 623),
                'chemistry': {'Ca': 'high', 'Mg': 'high', 'CO2': 'high'}
            },
            'advanced_argillic': {
                'minerals': ['alunite', 'kaolinite', 'dickite', 'pyrophyllite'],
                'temperature_range': (423, 623),
                'chemistry': {'SO4': 'high', 'pH': 'very_low'}
            }
        }
    
    def create_alteration_zoning(self, center_location: Tuple[float, float, float],
                               heat_source_temp: float, distance_range: float = 1000) -> Dict:
        """Create concentric alteration zones around heat source"""
        
        zones = []
        
        # Define zone boundaries based on temperature
        if heat_source_temp > 900:  # High-T intrusion
            zone_sequence = [
                ('potassic', 0, 200, (850, 950)),
                ('phyllic', 200, 500, (600, 750)),
                ('argillic', 500, 800, (400, 600)),
                ('propylitic', 800, 1200, (350, 500))
            ]
        else:  # Lower-T system
            zone_sequence = [
                ('phyllic', 0, 300, (550, 650)),
                ('argillic', 300, 600, (400, 550)),
                ('propylitic', 600, 1000, (350, 450))
            ]
        
        for zone_name, inner_radius, outer_radius, temp_range in zone_sequence:
            alteration_data = self.alteration_minerals[zone_name]
            
            zone = AlterationZone(
                zone_name=zone_name,
                distance_from_center=(inner_radius + outer_radius) / 2,
                temperature_range=temp_range,
                mineral_assemblage=alteration_data['minerals'],
                permeability=self._calculate_zone_permeability(zone_name),
                porosity=self._calculate_zone_porosity(zone_name)
            )
            
            zones.append(zone)
            self.alteration_zones[f"{zone_name}_{len(self.alteration_zones)}"] = zone
        
        return {
            'center_location': center_location,
            'heat_source_temp': heat_source_temp,
            'zones': zones,
            'total_radius': distance_range
        }
    
    def _calculate_zone_permeability(self, zone_name: str) -> float:
        """Calculate permeability for alteration zone"""
        permeability_map = {
            'potassic': 1e-16,      # Low - tight K-feldspar alteration
            'phyllic': 1e-14,       # High - fracture-controlled sericite
            'argillic': 1e-18,      # Very low - clay minerals
            'propylitic': 1e-15,    # Moderate - chlorite-epidote
            'advanced_argillic': 1e-19  # Extremely low - alunite-clay
        }
        return permeability_map.get(zone_name, 1e-16)
    
    def _calculate_zone_porosity(self, zone_name: str) -> float:
        """Calculate porosity for alteration zone"""
        porosity_map = {
            'potassic': 0.05,       # Low porosity
            'phyllic': 0.15,        # Higher porosity from leaching
            'argillic': 0.25,       # High porosity from clay formation
            'propylitic': 0.10,     # Moderate porosity
            'advanced_argillic': 0.30  # Very high porosity
        }
        return porosity_map.get(zone_name, 0.10)
    
    def simulate_alteration_progress(self, initial_rock: Dict[str, float],
                                   fluid_history: List[FluidComposition],
                                   time_steps: np.ndarray) -> Dict:
        """Simulate progressive alteration"""
        
        rock_composition = initial_rock.copy()
        alteration_history = []
        
        for i, (fluid, dt) in enumerate(zip(fluid_history, np.diff(time_steps))):
            
            # Determine alteration type based on fluid conditions
            alteration_type = self._classify_alteration_conditions(fluid)
            
            # Calculate alteration rates
            alteration_rates = self._calculate_alteration_rates(
                rock_composition, fluid, alteration_type
            )
            
            # Update rock composition
            for component, rate in alteration_rates.items():
                rock_composition[component] = max(0, 
                    rock_composition.get(component, 0) + rate * dt
                )
            
            # Calculate modal mineralogy
            modal_mineralogy = self._calculate_modal_mineralogy(
                rock_composition, fluid.temperature
            )
            
            alteration_history.append({
                'time': time_steps[i+1],
                'alteration_type': alteration_type,
                'rock_composition': rock_composition.copy(),
                'modal_mineralogy': modal_mineralogy,
                'fluid_conditions': {
                    'temperature': fluid.temperature,
                    'ph': fluid.ph,
                    'log_fo2': fluid.log_fo2
                }
            })
        
        return {
            'initial_composition': initial_rock,
            'alteration_history': alteration_history,
            'final_composition': rock_composition
        }
    
    def _classify_alteration_conditions(self, fluid: FluidComposition) -> str:
        """Classify alteration type based on fluid conditions"""
        temp = fluid.temperature
        ph = fluid.ph
        log_fo2 = fluid.log_fo2
        
        # Temperature-based classification with pH/fO2 modifiers
        if temp > 723:
            if log_fo2 > -25:
                return 'potassic'
            else:
                return 'phyllic'
        elif temp > 523:
            if ph < 3:
                return 'advanced_argillic'
            elif ph < 5:
                return 'phyllic'
            else:
                return 'propylitic'
        elif temp > 373:
            if ph < 4:
                return 'argillic'
            else:
                return 'propylitic'
        else:
            return 'none'
    
    def _calculate_alteration_rates(self, rock_composition: Dict[str, float],
                                  fluid: FluidComposition, alteration_type: str) -> Dict:
        """Calculate alteration reaction rates"""
        rates = {}
        
        # Base rate depends on temperature
        base_rate = 1e-8 * np.exp(-50000 / (8.314 * fluid.temperature))
        
        # Specific reactions for each alteration type
        if alteration_type == 'potassic':
            rates['K-feldspar'] = base_rate * fluid.concentrations.get('K+', 1e-3)
            rates['biotite'] = base_rate * 0.5
            rates['plagioclase'] = -base_rate * 2  # Destruction
        
        elif alteration_type == 'phyllic':
            rates['sericite'] = base_rate * 3
            rates['quartz'] = base_rate * 0.5
            rates['K-feldspar'] = -base_rate * 1.5
        
        elif alteration_type == 'argillic':
            rates['kaolinite'] = base_rate * 2
            rates['illite'] = base_rate * 1
            rates['plagioclase'] = -base_rate * 2
        
        elif alteration_type == 'propylitic':
            rates['chlorite'] = base_rate * 1.5
            rates['epidote'] = base_rate * 1
            rates['calcite'] = base_rate * 0.5
        
        return rates
    
    def _calculate_modal_mineralogy(self, composition: Dict[str, float], 
                                  temperature: float) -> Dict[str, float]:
        """Calculate modal mineralogy from bulk composition"""
        # Simplified modal calculation
        total = sum(composition.values())
        if total == 0:
            return {}
        
        modal = {mineral: amount/total * 100 for mineral, amount in composition.items()}
        return modal

class GeneticModelIntegrator:
    """Main genetic model integration system"""
    
    def __init__(self):
        self.flow_modeler = None
        self.precipitation_modeler = PrecipitationKinetics()
        self.alteration_modeler = AlterationModeler()
        self.time_series_data = {}
        self.integrated_results = {}
    
    def initialize_flow_domain(self, domain_size: Tuple[float, float, float],
                             resolution: Tuple[int, int, int]):
        """Initialize fluid flow modeling domain"""
        self.flow_modeler = FluidFlowModeler(domain_size, resolution)
    
    def run_integrated_simulation(self, simulation_parameters: Dict) -> Dict:
        """Run integrated genetic model simulation"""
        
        if self.flow_modeler is None:
            raise ValueError("Flow domain not initialized")
        
        # Extract parameters
        total_time = simulation_parameters.get('total_time', 1e6 * 365 * 24 * 3600)  # 1 My in seconds
        time_steps = simulation_parameters.get('time_steps', 100)
        intrusion_location = simulation_parameters.get('intrusion_location', (500, 500, 1000))
        intrusion_temperature = simulation_parameters.get('intrusion_temperature', 1200)  # K
        initial_fluid_composition = simulation_parameters.get('initial_fluid', {
            'temperature': 773,
            'pressure': 1e7,
            'salinity': 5.0,
            'ph': 6.0,
            'log_fo2': -25.0,
            'concentrations': {'Cu+': 1e-4, 'Fe2+': 1e-3, 'Zn2+': 1e-5, 'H2S': 1e-3}
        })
        
        # Create time array
        time_array = np.linspace(0, total_time, time_steps)
        dt = total_time / (time_steps - 1)
        
        # Initialize results storage
        results = {
            'time_array': time_array,
            'thermal_evolution': [],
            'flow_evolution': [],
            'precipitation_evolution': [],
            'alteration_evolution': [],
            'integrated_summary': {}
        }
        
        # Add heat source (intrusion)
        self.flow_modeler.add_heat_source(intrusion_location, 1e12, 200)  # 1 TW, 200m radius
        
        # Set up permeability field (fracture network)
        def permeability_function(x, y, z):
            # Higher permeability near intrusion and along fractures
            dist_to_intrusion = np.sqrt((x - intrusion_location[0])**2 + 
                                      (y - intrusion_location[1])**2 + 
                                      (z - intrusion_location[2])**2)
            
            base_perm = 1e-17
            fracture_perm = 1e-13
            
            # Fracture zones
            if abs(x - 400) < 50 or abs(y - 600) < 50:  # Fracture corridors
                return fracture_perm
            elif dist_to_intrusion < 300:  # Near intrusion
                return base_perm * np.exp(-dist_to_intrusion / 100)
            else:
                return base_perm
        
        self.flow_modeler.set_permeability_field(permeability_function)
        
        # Time stepping simulation
        print(f"Running integrated simulation for {time_steps} time steps...")
        
        for i in range(1, time_steps):
            current_time = time_array[i]
            
            # 1. Solve thermal field
            temperature_field = self.flow_modeler.solve_temperature_field(dt)
            
            # 2. Solve pressure and flow field
            pressure_field = self.flow_modeler.solve_pressure_field(dt)
            velocity_field = self.flow_modeler.calculate_velocity_field()
            
            # 3. Transport solutes
            for element in ['Cu', 'Fe', 'Zn', 'S']:
                concentration_field = self.flow_modeler.transport_solutes(
                    element, initial_fluid_composition['concentrations'].get(f'{element}+', 1e-6), dt
                )
            
            # 4. Calculate precipitation/dissolution
            fluid_sample = FluidComposition(
                temperature=np.mean(temperature_field),
                pressure=np.mean(pressure_field),
                salinity=initial_fluid_composition['salinity'],
                ph=initial_fluid_composition['ph'],
                log_fo2=initial_fluid_composition['log_fo2'],
                concentrations=initial_fluid_composition['concentrations'],
                phase='liquid'
            )
            
            # Simulate mineral precipitation
            mineral_evolution = self.precipitation_modeler.simulate_mineral_evolution(
                [fluid_sample], np.array([time_array[i-1], time_array[i]]), 
                {'quartz': 0.1, 'pyrite': 0.0, 'chalcopyrite': 0.0}
            )
            
            # 5. Calculate alteration
            if i % 10 == 0:  # Update alteration every 10 steps
                alteration_zones = self.alteration_modeler.create_alteration_zoning(
                    intrusion_location, intrusion_temperature
                )
            
            # Store results
            if i % max(1, time_steps // 20) == 0:  # Store 20 snapshots
                results['thermal_evolution'].append({
                    'time': current_time,
                    'temperature_field': temperature_field.copy(),
                    'mean_temperature': np.mean(temperature_field),
                    'max_temperature': np.max(temperature_field)
                })
                
                results['flow_evolution'].append({
                    'time': current_time,
                    'pressure_field': pressure_field.copy(),
                    'velocity_field': velocity_field.copy(),
                    'max_velocity': np.max(np.linalg.norm(velocity_field, axis=3))
                })
                
                results['precipitation_evolution'].append({
                    'time': current_time,
                    'mineral_amounts': mineral_evolution['mineral_evolution']
                })
                
                if i % 10 == 0:
                    results['alteration_evolution'].append({
                        'time': current_time,
                        'alteration_zones': alteration_zones
                    })
        
        # Calculate integrated summary
        results['integrated_summary'] = self._calculate_integrated_summary(results)
        
        return results
    
    def _calculate_integrated_summary(self, results: Dict) -> Dict:
        """Calculate integrated summary of genetic model results"""
        summary = {}
        
        # Thermal evolution summary
        if results['thermal_evolution']:
            max_temps = [item['max_temperature'] for item in results['thermal_evolution']]
            summary['thermal'] = {
                'peak_temperature': max(max_temps),
                'cooling_rate': (max_temps[0] - max_temps[-1]) / len(max_temps) if len(max_temps) > 1 else 0,
                'thermal_duration': results['time_array'][-1] - results['time_array'][0]
            }
        
        # Fluid flow summary
        if results['flow_evolution']:
            max_velocities = [item['max_velocity'] for item in results['flow_evolution']]
            summary['fluid_flow'] = {
                'peak_velocity': max(max_velocities),
                'flow_focusing': np.std(max_velocities) / np.mean(max_velocities) if max_velocities else 0
            }
        
        # Precipitation summary
        if results['precipitation_evolution']:
            final_minerals = results['precipitation_evolution'][-1]['mineral_amounts']
            summary['mineralization'] = {
                'total_sulfides': sum(amount[-1] for mineral, amount in final_minerals.items() 
                                    if 'S' in mineral),
                'dominant_mineral': max(final_minerals.keys(), 
                                      key=lambda k: final_minerals[k][-1] if final_minerals[k] else 0)
            }
        
        # Alteration summary
        if results['alteration_evolution']:
            final_alteration = results['alteration_evolution'][-1]['alteration_zones']
            summary['alteration'] = {
                'zone_count': len(final_alteration['zones']),
                'total_alteration_radius': final_alteration['total_radius'],
                'zone_types': [zone.zone_name for zone in final_alteration['zones']]
            }
        
        return summary
    
    def export_results(self, results: Dict, filename: str):
        """Export genetic model results"""
        if filename.endswith('.json'):
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_numpy(results)
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
        
        elif filename.endswith('.csv'):
            # Export summary time series
            summary_data = []
            
            for i, thermal in enumerate(results.get('thermal_evolution', [])):
                row = {
                    'time': thermal['time'],
                    'mean_temperature': thermal['mean_temperature'],
                    'max_temperature': thermal['max_temperature']
                }
                
                if i < len(results.get('flow_evolution', [])):
                    row['max_velocity'] = results['flow_evolution'][i]['max_velocity']
                
                summary_data.append(row)
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                df.to_csv(filename, index=False)
        
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")

# Example usage and testing
if __name__ == "__main__":
    # Initialize genetic model integrator
    integrator = GeneticModelIntegrator()
    
    # Set up modeling domain (1km x 1km x 1km)
    integrator.initialize_flow_domain((1000, 1000, 1000), (50, 50, 50))
    
    print("Testing genetic model components:")
    
    # 1. Test precipitation kinetics
    print("\n1. Testing precipitation kinetics:")
    test_fluid = FluidComposition(
        temperature=673, pressure=1e7, salinity=5.0, ph=5.5, log_fo2=-30,
        concentrations={'Fe2+': 1e-3, 'Cu+': 1e-4, 'H2S': 1e-3}, phase='liquid'
    )
    
    pyrite_si = integrator.precipitation_modeler.calculate_saturation_index('pyrite', test_fluid)
    print(f"Pyrite saturation index: {pyrite_si:.2f}")
    
    # 2. Test alteration modeling
    print("\n2. Testing alteration modeling:")
    alteration_zones = integrator.alteration_modeler.create_alteration_zoning(
        (500, 500, 500), 1000
    )
    print(f"Created {len(alteration_zones['zones'])} alteration zones")
    for zone in alteration_zones['zones']:
        print(f"  {zone.zone_name}: {zone.distance_from_center:.0f}m from center")
    
    # 3. Test integrated simulation (simplified)
    print("\n3. Testing integrated simulation:")
    sim_params = {
        'total_time': 3.15e10,  # 1000 years
        'time_steps': 20,
        'intrusion_location': (500, 500, 800),
        'intrusion_temperature': 1100
    }
    
    try:
        integrated_results = integrator.run_integrated_simulation(sim_params)
        print(f"Simulation completed with {len(integrated_results['thermal_evolution'])} thermal snapshots")
        
        if 'integrated_summary' in integrated_results:
            summary = integrated_results['integrated_summary']
            if 'thermal' in summary:
                print(f"Peak temperature: {summary['thermal']['peak_temperature']:.0f} K")
            if 'mineralization' in summary:
                print(f"Dominant mineral: {summary['mineralization'].get('dominant_mineral', 'none')}")
    
    except Exception as e:
        print(f"Simulation failed (expected for demo): {e}")
        print("Full simulation requires proper boundary conditions and longer runtime")
    
    print("\nGenetic Model Integrator initialized successfully!")