"""
Pressure Coupling and Magma Transfer Simulation Engine
Advanced real-time simulation of pressure coupling and magma transfer between interconnected chambers

This module implements sophisticated algorithms for:
- Real-time pressure coupling simulation between interconnected chambers
- Magma transfer rate calculations based on pressure gradients
- Viscosity-dependent flow modeling between chambers
- Volatile transport and degassing during magma transfer
- Thermal evolution during inter-chamber magma migration
- Crystallization effects on magma transfer dynamics
- Feedback mechanisms between chamber pressure and transfer rates
- Validation against laboratory experiments and natural systems
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path
import json
import pickle
from scipy import optimize, integrate, interpolate, sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Advanced scientific computing
try:
    import xarray as xr
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. Some computations will be slower.")

# Machine learning for pattern recognition
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("Machine learning libraries not available.")

# Database integration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.database_integration import DatabaseIntegration
    from core.gis_postgresql_operations import GISPostgreSQLOperations
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    warnings.warn("Database integration not available.")

# ============================================================================
# Data Models and Structures
# ============================================================================

@dataclass
class MagmaProperties:
    """Comprehensive magma properties for transfer calculations"""
    composition: Dict[str, float]  # Oxide weight percentages
    temperature: float  # Temperature (K)
    pressure: float  # Pressure (Pa)
    density: float  # Density (kg/m³)
    viscosity: float  # Dynamic viscosity (Pa·s)
    crystallinity: float  # Crystal fraction (0-1)
    volatile_content: Dict[str, float]  # Volatile concentrations (wt%)
    gas_fraction: float  # Gas bubble fraction (0-1)
    surface_tension: float  # Surface tension (N/m)
    compressibility: float  # Compressibility (1/Pa)
    thermal_expansion: float  # Thermal expansion coefficient (1/K)
    heat_capacity: float  # Specific heat capacity (J/kg·K)
    thermal_conductivity: float  # Thermal conductivity (W/m·K)
    electrical_conductivity: float  # Electrical conductivity (S/m)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ChamberGeometry3D:
    """3D chamber geometry representation"""
    chamber_id: str
    center_coordinates: Tuple[float, float, float]  # x, y, z (km)
    volume: float  # Chamber volume (km³)
    shape_parameters: Dict[str, float]  # Shape-specific parameters
    shape_type: str  # 'ellipsoid', 'prolate', 'oblate', 'irregular'
    orientation: Dict[str, float]  # Strike, dip, plunge angles
    aspect_ratios: Tuple[float, float, float]  # a/b, b/c ratios
    surface_mesh: Optional[np.ndarray] = None  # 3D surface mesh
    uncertainty_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    confidence_level: float = 0.95
    reconstruction_method: str = 'seismic_tomography'
    validation_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ChamberState:
    """Real-time state of a magma chamber"""
    chamber_id: str
    geometry: ChamberGeometry3D
    magma_properties: MagmaProperties
    pressure: float  # Chamber pressure (Pa)
    volume: float  # Current volume (m³)
    mass: float  # Magma mass (kg)
    temperature_distribution: np.ndarray  # 3D temperature field (K)
    composition_distribution: Dict[str, np.ndarray]  # 3D composition fields
    crystallinity_distribution: np.ndarray  # 3D crystallinity field
    volatile_distribution: Dict[str, np.ndarray]  # 3D volatile fields
    gas_distribution: np.ndarray  # 3D gas fraction field
    velocity_field: np.ndarray  # 3D velocity field (m/s)
    stress_tensor: np.ndarray  # 3D stress tensor (Pa)
    deformation_rate: np.ndarray  # Deformation rate tensor (1/s)
    heat_flux: np.ndarray  # Heat flux vectors (W/m²)
    mass_flux_boundaries: Dict[str, float]  # Boundary mass fluxes (kg/s)
    energy_balance: Dict[str, float]  # Energy balance components (W)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ConnectionProperties:
    """Properties of connections between chambers"""
    connection_id: str
    from_chamber_id: str
    to_chamber_id: str
    geometry: Dict[str, Any]  # Connection geometry parameters
    hydraulic_diameter: float  # Hydraulic diameter (m)
    length: float  # Connection length (m)
    cross_sectional_area: float  # Cross-sectional area (m²)
    roughness: float  # Surface roughness (m)
    tortuosity: float  # Tortuosity factor
    permeability: float  # Permeability (m²)
    thermal_conductivity: float  # Wall thermal conductivity (W/m·K)
    heat_transfer_coefficient: float  # Heat transfer coefficient (W/m²·K)
    wall_temperature: float  # Wall temperature (K)
    pressure_loss_coefficient: float  # Pressure loss coefficient
    flow_regime: str  # 'laminar', 'turbulent', 'transitional'
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TransferFlowState:
    """State of magma transfer flow between chambers"""
    connection_id: str
    flow_rate: float  # Mass flow rate (kg/s)
    velocity: float  # Average velocity (m/s)
    pressure_gradient: float  # Pressure gradient (Pa/m)
    temperature_profile: np.ndarray  # Temperature along flow path (K)
    composition_evolution: Dict[str, np.ndarray]  # Composition evolution
    crystallization_rate: float  # Crystallization rate (1/s)
    degassing_rate: Dict[str, float]  # Degassing rates by species (kg/s)
    heat_transfer_rate: float  # Heat transfer rate (W)
    friction_factor: float  # Friction factor
    reynolds_number: float  # Reynolds number
    prandtl_number: float  # Prandtl number
    nusselt_number: float  # Nusselt number
    flow_stability: float  # Flow stability indicator (0-1)
    residence_time: float  # Residence time in connection (s)
    mixing_efficiency: float  # Mixing efficiency (0-1)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SystemCouplingState:
    """Overall system coupling state"""
    system_id: str
    chambers: Dict[str, ChamberState]
    connections: Dict[str, ConnectionProperties]
    transfer_flows: Dict[str, TransferFlowState]
    pressure_coupling_matrix: np.ndarray  # Pressure coupling coefficients
    mass_transfer_matrix: np.ndarray  # Mass transfer rates (kg/s)
    energy_transfer_matrix: np.ndarray  # Energy transfer rates (W)
    system_pressure: float  # System-wide pressure (Pa)
    total_mass: float  # Total system mass (kg)
    total_energy: float  # Total system energy (J)
    entropy_production: float  # Entropy production rate (W/K)
    stability_metrics: Dict[str, float]  # System stability indicators
    feedback_strengths: Dict[str, float]  # Feedback mechanism strengths
    critical_thresholds: Dict[str, float]  # Critical system thresholds
    eruption_potential: Dict[str, float]  # Eruption potential by chamber
    timestamp: datetime = field(default_factory=datetime.now)

# ============================================================================
# Pressure Coupling and Magma Transfer Engine
# ============================================================================

class PressureCouplingMagmaTransferEngine:
    """
    Advanced pressure coupling and magma transfer simulation engine with
    comprehensive multi-physics modeling and real-time capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the pressure coupling and magma transfer engine"""
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Initialize database connection if available
        if DATABASE_AVAILABLE:
            self.db_integration = DatabaseIntegration()
            self.gis_ops = GISPostgreSQLOperations()
        else:
            self.db_integration = None
            self.gis_ops = None
        
        # Initialize physical models
        self._initialize_physical_models()
        
        # Initialize numerical solvers
        self._initialize_numerical_solvers()
        
        # Cache for computed results
        self.flow_cache = {}
        self.thermodynamic_cache = {}
        
        # Real-time simulation state
        self.current_system_state = None
        self.simulation_time = 0.0
        self.time_step = self.config['simulation']['default_time_step']
        
        self.logger.info("Pressure Coupling and Magma Transfer Engine initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters"""
        return {
            'simulation': {
                'default_time_step': 1.0,  # seconds
                'max_time_step': 3600.0,  # seconds
                'min_time_step': 0.001,  # seconds
                'adaptive_time_stepping': True,
                'convergence_tolerance': 1e-6,
                'max_iterations': 1000,
                'stability_threshold': 0.95
            },
            'physics': {
                'include_viscosity_effects': True,
                'include_thermal_effects': True,
                'include_crystallization': True,
                'include_degassing': True,
                'include_compressibility': True,
                'include_turbulence': True,
                'gravity_acceleration': 9.81  # m/s²
            },
            'numerical': {
                'spatial_discretization': 'finite_volume',
                'temporal_discretization': 'implicit_euler',
                'pressure_solver': 'conjugate_gradient',
                'momentum_solver': 'gmres',
                'energy_solver': 'bicgstab',
                'species_solver': 'upwind'
            },
            'validation': {
                'enable_validation': True,
                'validation_frequency': 100,  # time steps
                'experimental_weight': 0.7,
                'natural_system_weight': 0.3
            },
            'database': {
                'cache_results': True,
                'auto_save_frequency': 3600,  # seconds
                'compression': 'gzip'
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__ + ".PressureCouplingMagmaTransferEngine")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_physical_models(self):
        """Initialize physical models for magma transfer simulation"""
        
        # Viscosity models
        self.viscosity_models = {
            'arrhenius': self._arrhenius_viscosity,
            'vogel_fulcher_tammann': self._vft_viscosity,
            'giordano_2008': self._giordano_viscosity,
            'crystal_bearing': self._crystal_bearing_viscosity
        }
        
        # Equation of state models
        self.eos_models = {
            'ideal_gas': self._ideal_gas_eos,
            'redlich_kwong': self._redlich_kwong_eos,
            'peng_robinson': self._peng_robinson_eos,
            'magma_eos': self._magma_equation_of_state
        }
        
        # Crystallization models
        self.crystallization_models = {
            'avrami': self._avrami_crystallization,
            'johnson_mehl': self._johnson_mehl_crystallization,
            'nucleation_growth': self._nucleation_growth_model,
            'equilibrium': self._equilibrium_crystallization
        }
        
        # Degassing models
        self.degassing_models = {
            'henry_law': self._henry_law_degassing,
            'bubble_growth': self._bubble_growth_degassing,
            'diffusive': self._diffusive_degassing,
            'explosive': self._explosive_degassing
        }
        
        self.logger.info("Physical models initialized")
    
    def _initialize_numerical_solvers(self):
        """Initialize numerical solvers for coupled equations"""
        
        # Pressure-velocity coupling solvers
        self.pressure_velocity_solvers = {
            'simple': self._simple_algorithm,
            'piso': self._piso_algorithm,
            'coupled': self._coupled_solver
        }
        
        # Linear system solvers
        self.linear_solvers = {
            'direct': self._direct_solver,
            'iterative': self._iterative_solver,
            'multigrid': self._multigrid_solver
        }
        
        # Time integration schemes
        self.time_integrators = {
            'explicit_euler': self._explicit_euler,
            'implicit_euler': self._implicit_euler,
            'runge_kutta': self._runge_kutta_4,
            'adams_bashforth': self._adams_bashforth
        }
        
        self.logger.info("Numerical solvers initialized")
    
    # ========================================================================
    # Core Simulation Methods
    # ========================================================================
    
    async def simulate_real_time_pressure_coupling(
        self,
        chamber_states: Dict[str, ChamberState],
        connections: Dict[str, ConnectionProperties],
        time_step: float,
        simulation_duration: float
    ) -> SystemCouplingState:
        """
        Simulate real-time pressure coupling between interconnected chambers
        
        Args:
            chamber_states: Current states of all chambers
            connections: Connection properties between chambers
            time_step: Simulation time step (s)
            simulation_duration: Total simulation duration (s)
            
        Returns:
            Final system coupling state
        """
        try:
            self.logger.info(f"Starting real-time pressure coupling simulation for {simulation_duration}s")
            
            # Initialize system state
            system_state = await self._initialize_system_state(chamber_states, connections)
            
            # Time integration loop
            current_time = 0.0
            time_steps = int(simulation_duration / time_step)
            
            for step in range(time_steps):
                # Update chamber pressures based on coupling
                await self._update_pressure_coupling(system_state, time_step)
                
                # Calculate magma transfer rates
                await self._calculate_transfer_rates(system_state, time_step)
                
                # Update chamber states
                await self._update_chamber_states(system_state, time_step)
                
                # Check system stability
                stability = await self._check_system_stability(system_state)
                if stability < self.config['simulation']['stability_threshold']:
                    self.logger.warning(f"System instability detected at t={current_time:.2f}s")
                    time_step = await self._adapt_time_step(system_state, time_step)
                
                # Update simulation time
                current_time += time_step
                system_state.timestamp = datetime.now()
                
                # Periodic validation and saving
                if step % self.config['validation']['validation_frequency'] == 0:
                    await self._validate_simulation_state(system_state)
                    if self.db_integration:
                        await self._save_system_state(system_state)
                
                # Progress logging
                if step % (time_steps // 10) == 0:
                    progress = (step / time_steps) * 100
                    self.logger.info(f"Simulation progress: {progress:.1f}%")
            
            self.logger.info("Real-time pressure coupling simulation completed successfully")
            return system_state
            
        except Exception as e:
            self.logger.error(f"Pressure coupling simulation failed: {str(e)}")
            raise
    
    async def calculate_magma_transfer_rates(
        self,
        from_chamber: ChamberState,
        to_chamber: ChamberState,
        connection: ConnectionProperties
    ) -> TransferFlowState:
        """
        Calculate magma transfer rates based on pressure gradients
        
        Args:
            from_chamber: Source chamber state
            to_chamber: Destination chamber state
            connection: Connection properties
            
        Returns:
            Transfer flow state
        """
        try:
            # Calculate pressure gradient
            pressure_gradient = (from_chamber.pressure - to_chamber.pressure) / connection.length
            
            # Determine flow regime
            flow_regime = await self._determine_flow_regime(
                from_chamber.magma_properties, connection, pressure_gradient
            )
            
            # Calculate flow rate based on regime
            if flow_regime == 'laminar':
                flow_rate = await self._calculate_laminar_flow_rate(
                    from_chamber, to_chamber, connection, pressure_gradient
                )
            elif flow_regime == 'turbulent':
                flow_rate = await self._calculate_turbulent_flow_rate(
                    from_chamber, to_chamber, connection, pressure_gradient
                )
            else:  # transitional
                flow_rate = await self._calculate_transitional_flow_rate(
                    from_chamber, to_chamber, connection, pressure_gradient
                )
            
            # Calculate velocity
            velocity = flow_rate / (from_chamber.magma_properties.density * connection.cross_sectional_area)
            
            # Calculate Reynolds number
            reynolds_number = await self._calculate_reynolds_number(
                from_chamber.magma_properties, velocity, connection.hydraulic_diameter
            )
            
            # Calculate heat transfer coefficients
            nusselt_number = await self._calculate_nusselt_number(reynolds_number, from_chamber.magma_properties)
            
            # Calculate crystallization during transfer
            crystallization_rate = await self._calculate_transfer_crystallization(
                from_chamber.magma_properties, velocity, connection.length
            )
            
            # Calculate degassing during transfer
            degassing_rates = await self._calculate_transfer_degassing(
                from_chamber.magma_properties, pressure_gradient, velocity
            )
            
            # Calculate residence time
            residence_time = connection.length / velocity if velocity > 0 else float('inf')
            
            # Create transfer flow state
            transfer_state = TransferFlowState(
                connection_id=connection.connection_id,
                flow_rate=flow_rate,
                velocity=velocity,
                pressure_gradient=pressure_gradient,
                temperature_profile=await self._calculate_temperature_profile(
                    from_chamber, to_chamber, connection, velocity
                ),
                composition_evolution=await self._calculate_composition_evolution(
                    from_chamber.magma_properties, connection, velocity
                ),
                crystallization_rate=crystallization_rate,
                degassing_rate=degassing_rates,
                heat_transfer_rate=await self._calculate_heat_transfer_rate(
                    from_chamber, to_chamber, connection, velocity
                ),
                friction_factor=await self._calculate_friction_factor(reynolds_number, connection.roughness),
                reynolds_number=reynolds_number,
                prandtl_number=await self._calculate_prandtl_number(from_chamber.magma_properties),
                nusselt_number=nusselt_number,
                flow_stability=await self._assess_flow_stability(reynolds_number, pressure_gradient),
                residence_time=residence_time,
                mixing_efficiency=await self._calculate_mixing_efficiency(reynolds_number, connection.tortuosity)
            )
            
            return transfer_state
            
        except Exception as e:
            self.logger.error(f"Transfer rate calculation failed: {str(e)}")
            raise
    
    async def model_viscosity_dependent_flow(
        self,
        magma_properties: MagmaProperties,
        connection: ConnectionProperties,
        pressure_gradient: float
    ) -> Dict[str, float]:
        """
        Model viscosity-dependent flow between chambers
        
        Args:
            magma_properties: Magma properties
            connection: Connection properties
            pressure_gradient: Pressure gradient (Pa/m)
            
        Returns:
            Flow characteristics dictionary
        """
        try:
            # Calculate effective viscosity including crystal and bubble effects
            effective_viscosity = await self._calculate_effective_viscosity(magma_properties)
            
            # Calculate flow rate using Hagen-Poiseuille equation for circular pipes
            if connection.geometry.get('shape') == 'circular':
                radius = connection.hydraulic_diameter / 2
                flow_rate = (np.pi * radius**4 * abs(pressure_gradient)) / (8 * effective_viscosity * connection.length)
            else:
                # Use hydraulic diameter approximation for non-circular cross-sections
                flow_rate = (connection.cross_sectional_area**2 * abs(pressure_gradient)) / (
                    12 * effective_viscosity * connection.length * connection.tortuosity
                )
            
            # Account for pressure gradient direction
            if pressure_gradient < 0:
                flow_rate = -flow_rate
            
            # Calculate velocity
            velocity = flow_rate / (magma_properties.density * connection.cross_sectional_area)
            
            # Calculate wall shear stress
            wall_shear_stress = (4 * effective_viscosity * velocity) / connection.hydraulic_diameter
            
            # Calculate pressure drop due to viscous effects
            viscous_pressure_drop = (32 * effective_viscosity * velocity * connection.length) / connection.hydraulic_diameter**2
            
            # Calculate energy dissipation rate
            energy_dissipation = wall_shear_stress * velocity * connection.cross_sectional_area * connection.length
            
            return {
                'flow_rate': flow_rate,
                'velocity': velocity,
                'effective_viscosity': effective_viscosity,
                'wall_shear_stress': wall_shear_stress,
                'viscous_pressure_drop': viscous_pressure_drop,
                'energy_dissipation': energy_dissipation,
                'reynolds_number': (magma_properties.density * velocity * connection.hydraulic_diameter) / effective_viscosity
            }
            
        except Exception as e:
            self.logger.error(f"Viscosity-dependent flow modeling failed: {str(e)}")
            raise
    
    async def simulate_volatile_transport_degassing(
        self,
        magma_properties: MagmaProperties,
        connection: ConnectionProperties,
        flow_velocity: float,
        pressure_profile: np.ndarray
    ) -> Dict[str, Any]:
        """
        Simulate volatile transport and degassing during magma transfer
        
        Args:
            magma_properties: Initial magma properties
            connection: Connection properties
            flow_velocity: Flow velocity (m/s)
            pressure_profile: Pressure profile along connection (Pa)
            
        Returns:
            Volatile transport and degassing results
        """
        try:
            # Initialize spatial discretization along connection
            n_points = 100
            distance = np.linspace(0, connection.length, n_points)
            
            # Initialize volatile concentrations
            volatile_species = list(magma_properties.volatile_content.keys())
            volatile_profiles = {}
            
            for species in volatile_species:
                volatile_profiles[species] = np.full(n_points, magma_properties.volatile_content[species])
            
            # Calculate solubility along pressure profile
            solubility_profiles = {}
            for species in volatile_species:
                solubility_profiles[species] = await self._calculate_volatile_solubility(
                    species, pressure_profile, magma_properties.temperature
                )
            
            # Simulate degassing along flow path
            gas_fraction_profile = np.zeros(n_points)
            degassing_rates = {species: np.zeros(n_points) for species in volatile_species}
            
            for i in range(1, n_points):
                dt = (distance[i] - distance[i-1]) / flow_velocity
                
                for species in volatile_species:
                    # Check for oversaturation
                    if volatile_profiles[species][i-1] > solubility_profiles[species][i]:
                        # Calculate degassing rate
                        oversaturation = volatile_profiles[species][i-1] - solubility_profiles[species][i]
                        degassing_rate = await self._calculate_degassing_rate(
                            species, oversaturation, magma_properties, pressure_profile[i]
                        )
                        
                        # Update volatile concentration
                        volatile_loss = degassing_rate * dt
                        volatile_profiles[species][i] = max(
                            volatile_profiles[species][i-1] - volatile_loss,
                            solubility_profiles[species][i]
                        )
                        
                        degassing_rates[species][i] = degassing_rate
                        
                        # Update gas fraction
                        gas_fraction_profile[i] += volatile_loss / magma_properties.density
                    else:
                        volatile_profiles[species][i] = volatile_profiles[species][i-1]
            
            # Calculate bubble nucleation and growth
            bubble_dynamics = await self._simulate_bubble_dynamics(
                volatile_profiles, pressure_profile, magma_properties, flow_velocity
            )
            
            # Calculate effects on flow properties
            flow_effects = await self._calculate_degassing_flow_effects(
                gas_fraction_profile, bubble_dynamics, magma_properties
            )
            
            return {
                'distance_profile': distance,
                'volatile_profiles': volatile_profiles,
                'solubility_profiles': solubility_profiles,
                'gas_fraction_profile': gas_fraction_profile,
                'degassing_rates': degassing_rates,
                'bubble_dynamics': bubble_dynamics,
                'flow_effects': flow_effects,
                'total_gas_released': np.sum([np.trapz(rates, distance) for rates in degassing_rates.values()]),
                'final_volatile_content': {species: profiles[-1] for species, profiles in volatile_profiles.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Volatile transport and degassing simulation failed: {str(e)}")
            raise    
   
    async def simulate_thermal_evolution_during_transfer(
        self,
        initial_temperature: float,
        magma_properties: MagmaProperties,
        connection: ConnectionProperties,
        flow_velocity: float,
        ambient_temperature: float
    ) -> Dict[str, Any]:
        """
        Simulate thermal evolution during inter-chamber magma migration
        
        Args:
            initial_temperature: Initial magma temperature (K)
            magma_properties: Magma properties
            connection: Connection properties
            flow_velocity: Flow velocity (m/s)
            ambient_temperature: Ambient rock temperature (K)
            
        Returns:
            Thermal evolution results
        """
        try:
            # Initialize spatial discretization
            n_points = 100
            distance = np.linspace(0, connection.length, n_points)
            temperature_profile = np.full(n_points, initial_temperature)
            
            # Calculate heat transfer parameters
            reynolds_number = (magma_properties.density * flow_velocity * connection.hydraulic_diameter) / magma_properties.viscosity
            prandtl_number = (magma_properties.viscosity * magma_properties.heat_capacity) / magma_properties.thermal_conductivity
            
            # Calculate Nusselt number for heat transfer
            if reynolds_number < 2300:  # Laminar flow
                nusselt_number = 3.66  # Constant for fully developed laminar flow
            else:  # Turbulent flow
                nusselt_number = 0.023 * (reynolds_number**0.8) * (prandtl_number**0.4)
            
            # Calculate heat transfer coefficient
            heat_transfer_coefficient = (nusselt_number * magma_properties.thermal_conductivity) / connection.hydraulic_diameter
            
            # Calculate thermal time constant
            thermal_mass = magma_properties.density * magma_properties.heat_capacity * connection.cross_sectional_area
            thermal_resistance = 1 / (heat_transfer_coefficient * np.pi * connection.hydraulic_diameter)
            thermal_time_constant = thermal_mass * thermal_resistance
            
            # Simulate temperature evolution along flow path
            heat_flux_profile = np.zeros(n_points)
            
            for i in range(1, n_points):
                dt = (distance[i] - distance[i-1]) / flow_velocity
                
                # Calculate heat loss to walls
                temperature_difference = temperature_profile[i-1] - ambient_temperature
                heat_flux = heat_transfer_coefficient * temperature_difference
                heat_flux_profile[i] = heat_flux
                
                # Calculate temperature change
                heat_loss_rate = heat_flux * np.pi * connection.hydraulic_diameter * (distance[i] - distance[i-1])
                temperature_change = -heat_loss_rate / (magma_properties.density * magma_properties.heat_capacity * 
                                                      connection.cross_sectional_area * flow_velocity)
                
                temperature_profile[i] = temperature_profile[i-1] + temperature_change
            
            # Calculate crystallization effects on thermal evolution
            crystallization_effects = await self._calculate_crystallization_thermal_effects(
                temperature_profile, magma_properties, distance, flow_velocity
            )
            
            # Calculate viscosity evolution due to temperature changes
            viscosity_profile = np.zeros(n_points)
            for i, temp in enumerate(temperature_profile):
                temp_magma_props = MagmaProperties(
                    composition=magma_properties.composition,
                    temperature=temp,
                    pressure=magma_properties.pressure,
                    density=magma_properties.density,
                    viscosity=magma_properties.viscosity,
                    crystallinity=magma_properties.crystallinity,
                    volatile_content=magma_properties.volatile_content,
                    gas_fraction=magma_properties.gas_fraction,
                    surface_tension=magma_properties.surface_tension,
                    compressibility=magma_properties.compressibility,
                    thermal_expansion=magma_properties.thermal_expansion,
                    heat_capacity=magma_properties.heat_capacity,
                    thermal_conductivity=magma_properties.thermal_conductivity,
                    electrical_conductivity=magma_properties.electrical_conductivity
                )
                viscosity_profile[i] = await self._calculate_temperature_dependent_viscosity(temp_magma_props)
            
            return {
                'distance_profile': distance,
                'temperature_profile': temperature_profile,
                'heat_flux_profile': heat_flux_profile,
                'viscosity_profile': viscosity_profile,
                'crystallization_effects': crystallization_effects,
                'total_heat_loss': np.trapz(heat_flux_profile, distance) * np.pi * connection.hydraulic_diameter,
                'final_temperature': temperature_profile[-1],
                'temperature_drop': initial_temperature - temperature_profile[-1],
                'thermal_time_constant': thermal_time_constant,
                'heat_transfer_coefficient': heat_transfer_coefficient,
                'nusselt_number': nusselt_number
            }
            
        except Exception as e:
            self.logger.error(f"Thermal evolution simulation failed: {str(e)}")
            raise
    
    async def model_crystallization_effects_on_transfer(
        self,
        magma_properties: MagmaProperties,
        temperature_profile: np.ndarray,
        pressure_profile: np.ndarray,
        flow_velocity: float,
        distance_profile: np.ndarray
    ) -> Dict[str, Any]:
        """
        Model crystallization effects on magma transfer dynamics
        
        Args:
            magma_properties: Initial magma properties
            temperature_profile: Temperature along flow path (K)
            pressure_profile: Pressure along flow path (Pa)
            flow_velocity: Flow velocity (m/s)
            distance_profile: Distance along flow path (m)
            
        Returns:
            Crystallization effects on transfer
        """
        try:
            n_points = len(temperature_profile)
            
            # Initialize crystallization profiles
            crystallinity_profile = np.full(n_points, magma_properties.crystallinity)
            crystal_size_profile = np.zeros(n_points)
            nucleation_rate_profile = np.zeros(n_points)
            growth_rate_profile = np.zeros(n_points)
            
            # Calculate liquidus temperature
            liquidus_temperature = await self._calculate_liquidus_temperature(
                magma_properties.composition, pressure_profile[0]
            )
            
            # Simulate crystallization along flow path
            for i in range(1, n_points):
                dt = (distance_profile[i] - distance_profile[i-1]) / flow_velocity
                
                # Calculate undercooling
                undercooling = liquidus_temperature - temperature_profile[i]
                
                if undercooling > 0:
                    # Calculate nucleation rate
                    nucleation_rate = await self._calculate_nucleation_rate(
                        undercooling, magma_properties, crystallinity_profile[i-1]
                    )
                    nucleation_rate_profile[i] = nucleation_rate
                    
                    # Calculate growth rate
                    growth_rate = await self._calculate_crystal_growth_rate(
                        undercooling, magma_properties, temperature_profile[i]
                    )
                    growth_rate_profile[i] = growth_rate
                    
                    # Update crystallinity using Avrami equation
                    crystallinity_change = await self._calculate_crystallinity_change(
                        nucleation_rate, growth_rate, dt, crystallinity_profile[i-1]
                    )
                    
                    crystallinity_profile[i] = min(
                        crystallinity_profile[i-1] + crystallinity_change, 0.95
                    )
                    
                    # Update crystal size
                    if crystallinity_profile[i] > crystallinity_profile[i-1]:
                        crystal_size_profile[i] = crystal_size_profile[i-1] + growth_rate * dt
                    else:
                        crystal_size_profile[i] = crystal_size_profile[i-1]
                else:
                    crystallinity_profile[i] = crystallinity_profile[i-1]
                    crystal_size_profile[i] = crystal_size_profile[i-1]
            
            # Calculate effects on flow properties
            viscosity_effects = await self._calculate_crystallization_viscosity_effects(
                crystallinity_profile, crystal_size_profile, magma_properties
            )
            
            # Calculate effects on density
            density_effects = await self._calculate_crystallization_density_effects(
                crystallinity_profile, magma_properties
            )
            
            # Calculate latent heat effects
            latent_heat_effects = await self._calculate_crystallization_latent_heat(
                crystallinity_profile, magma_properties, distance_profile
            )
            
            return {
                'distance_profile': distance_profile,
                'crystallinity_profile': crystallinity_profile,
                'crystal_size_profile': crystal_size_profile,
                'nucleation_rate_profile': nucleation_rate_profile,
                'growth_rate_profile': growth_rate_profile,
                'viscosity_effects': viscosity_effects,
                'density_effects': density_effects,
                'latent_heat_effects': latent_heat_effects,
                'final_crystallinity': crystallinity_profile[-1],
                'crystallinity_increase': crystallinity_profile[-1] - crystallinity_profile[0],
                'average_crystal_size': np.mean(crystal_size_profile[crystal_size_profile > 0])
            }
            
        except Exception as e:
            self.logger.error(f"Crystallization effects modeling failed: {str(e)}")
            raise
    
    async def implement_feedback_mechanisms(
        self,
        system_state: SystemCouplingState,
        time_step: float
    ) -> SystemCouplingState:
        """
        Implement feedback mechanisms between chamber pressure and transfer rates
        
        Args:
            system_state: Current system coupling state
            time_step: Time step for feedback calculation (s)
            
        Returns:
            Updated system state with feedback effects
        """
        try:
            # Calculate pressure-flow feedback
            pressure_flow_feedback = await self._calculate_pressure_flow_feedback(system_state)
            
            # Calculate thermal feedback effects
            thermal_feedback = await self._calculate_thermal_feedback(system_state)
            
            # Calculate compositional feedback
            compositional_feedback = await self._calculate_compositional_feedback(system_state)
            
            # Calculate crystallization feedback
            crystallization_feedback = await self._calculate_crystallization_feedback(system_state)
            
            # Calculate degassing feedback
            degassing_feedback = await self._calculate_degassing_feedback(system_state)
            
            # Update chamber pressures based on feedback
            for chamber_id, chamber_state in system_state.chambers.items():
                # Pressure feedback from flow changes
                pressure_change_flow = pressure_flow_feedback.get(chamber_id, 0.0) * time_step
                
                # Pressure feedback from thermal effects
                pressure_change_thermal = thermal_feedback.get(chamber_id, 0.0) * time_step
                
                # Pressure feedback from compositional changes
                pressure_change_composition = compositional_feedback.get(chamber_id, 0.0) * time_step
                
                # Pressure feedback from crystallization
                pressure_change_crystallization = crystallization_feedback.get(chamber_id, 0.0) * time_step
                
                # Pressure feedback from degassing
                pressure_change_degassing = degassing_feedback.get(chamber_id, 0.0) * time_step
                
                # Total pressure change
                total_pressure_change = (
                    pressure_change_flow + pressure_change_thermal + 
                    pressure_change_composition + pressure_change_crystallization + 
                    pressure_change_degassing
                )
                
                # Update chamber pressure
                chamber_state.pressure += total_pressure_change
                
                # Update feedback strengths in system state
                system_state.feedback_strengths[f"{chamber_id}_pressure_flow"] = abs(pressure_change_flow)
                system_state.feedback_strengths[f"{chamber_id}_thermal"] = abs(pressure_change_thermal)
                system_state.feedback_strengths[f"{chamber_id}_compositional"] = abs(pressure_change_composition)
                system_state.feedback_strengths[f"{chamber_id}_crystallization"] = abs(pressure_change_crystallization)
                system_state.feedback_strengths[f"{chamber_id}_degassing"] = abs(pressure_change_degassing)
            
            # Update transfer flow rates based on new pressure distribution
            for connection_id, transfer_flow in system_state.transfer_flows.items():
                connection = system_state.connections[connection_id]
                from_chamber = system_state.chambers[connection.from_chamber_id]
                to_chamber = system_state.chambers[connection.to_chamber_id]
                
                # Recalculate transfer flow with updated pressures
                updated_transfer = await self.calculate_magma_transfer_rates(
                    from_chamber, to_chamber, connection
                )
                
                system_state.transfer_flows[connection_id] = updated_transfer
            
            # Update system-wide metrics
            system_state.system_pressure = np.mean([
                chamber.pressure for chamber in system_state.chambers.values()
            ])
            
            # Calculate system stability after feedback
            system_state.stability_metrics = await self._calculate_system_stability_metrics(system_state)
            
            return system_state
            
        except Exception as e:
            self.logger.error(f"Feedback mechanism implementation failed: {str(e)}")
            raise    

    async def validate_against_laboratory_experiments(
        self,
        simulation_results: Dict[str, Any],
        experimental_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Validate simulation results against laboratory experiments
        
        Args:
            simulation_results: Simulation results to validate
            experimental_data: Laboratory experimental data
            
        Returns:
            Validation metrics
        """
        try:
            validation_metrics = {}
            
            # Validate flow rates
            if 'flow_rates' in experimental_data and 'flow_rates' in simulation_results:
                flow_rate_error = await self._calculate_relative_error(
                    simulation_results['flow_rates'], experimental_data['flow_rates']
                )
                validation_metrics['flow_rate_accuracy'] = 1.0 - flow_rate_error
            
            # Validate pressure drops
            if 'pressure_drops' in experimental_data and 'pressure_drops' in simulation_results:
                pressure_error = await self._calculate_relative_error(
                    simulation_results['pressure_drops'], experimental_data['pressure_drops']
                )
                validation_metrics['pressure_accuracy'] = 1.0 - pressure_error
            
            # Validate temperature profiles
            if 'temperature_profiles' in experimental_data and 'temperature_profiles' in simulation_results:
                temp_correlation = await self._calculate_correlation(
                    simulation_results['temperature_profiles'], experimental_data['temperature_profiles']
                )
                validation_metrics['temperature_correlation'] = temp_correlation
            
            # Validate crystallization behavior
            if 'crystallinity' in experimental_data and 'crystallinity' in simulation_results:
                crystal_error = await self._calculate_relative_error(
                    simulation_results['crystallinity'], experimental_data['crystallinity']
                )
                validation_metrics['crystallization_accuracy'] = 1.0 - crystal_error
            
            # Validate degassing rates
            if 'degassing_rates' in experimental_data and 'degassing_rates' in simulation_results:
                degassing_correlation = await self._calculate_correlation(
                    simulation_results['degassing_rates'], experimental_data['degassing_rates']
                )
                validation_metrics['degassing_correlation'] = degassing_correlation
            
            # Calculate overall validation score
            validation_metrics['overall_score'] = np.mean(list(validation_metrics.values()))
            
            self.logger.info(f"Laboratory validation completed with overall score: {validation_metrics['overall_score']:.3f}")
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Laboratory validation failed: {str(e)}")
            raise
    
    async def validate_against_natural_systems(
        self,
        simulation_results: Dict[str, Any],
        natural_system_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Validate simulation results against natural volcanic systems
        
        Args:
            simulation_results: Simulation results to validate
            natural_system_data: Natural system observational data
            
        Returns:
            Validation metrics
        """
        try:
            validation_metrics = {}
            
            # Validate eruption timing predictions
            if 'eruption_timing' in natural_system_data and 'eruption_predictions' in simulation_results:
                timing_accuracy = await self._validate_eruption_timing(
                    simulation_results['eruption_predictions'], natural_system_data['eruption_timing']
                )
                validation_metrics['eruption_timing_accuracy'] = timing_accuracy
            
            # Validate magma composition evolution
            if 'composition_evolution' in natural_system_data and 'composition_evolution' in simulation_results:
                composition_correlation = await self._calculate_correlation(
                    simulation_results['composition_evolution'], natural_system_data['composition_evolution']
                )
                validation_metrics['composition_correlation'] = composition_correlation
            
            # Validate gas emission patterns
            if 'gas_emissions' in natural_system_data and 'gas_emissions' in simulation_results:
                gas_correlation = await self._calculate_correlation(
                    simulation_results['gas_emissions'], natural_system_data['gas_emissions']
                )
                validation_metrics['gas_emission_correlation'] = gas_correlation
            
            # Validate deformation patterns
            if 'deformation' in natural_system_data and 'deformation_predictions' in simulation_results:
                deformation_accuracy = await self._validate_deformation_patterns(
                    simulation_results['deformation_predictions'], natural_system_data['deformation']
                )
                validation_metrics['deformation_accuracy'] = deformation_accuracy
            
            # Validate seismic activity patterns
            if 'seismic_activity' in natural_system_data and 'seismic_predictions' in simulation_results:
                seismic_correlation = await self._calculate_correlation(
                    simulation_results['seismic_predictions'], natural_system_data['seismic_activity']
                )
                validation_metrics['seismic_correlation'] = seismic_correlation
            
            # Calculate overall validation score
            validation_metrics['overall_score'] = np.mean(list(validation_metrics.values()))
            
            self.logger.info(f"Natural system validation completed with overall score: {validation_metrics['overall_score']:.3f}")
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Natural system validation failed: {str(e)}")
            raise
    
    # ========================================================================
    # Helper Methods for Physical Calculations
    # ========================================================================
    
    async def _calculate_effective_viscosity(self, magma_properties: MagmaProperties) -> float:
        """Calculate effective viscosity including crystal and bubble effects"""
        base_viscosity = magma_properties.viscosity
        
        # Einstein-Roscoe equation for crystal effects
        if magma_properties.crystallinity > 0:
            crystal_factor = (1 - magma_properties.crystallinity / 0.6) ** (-2.5)
            base_viscosity *= crystal_factor
        
        # Bubble effects on viscosity
        if magma_properties.gas_fraction > 0:
            bubble_factor = (1 + 2.5 * magma_properties.gas_fraction)
            base_viscosity *= bubble_factor
        
        return base_viscosity
    
    async def _calculate_temperature_dependent_viscosity(self, magma_properties: MagmaProperties) -> float:
        """
        Advanced temperature-dependent viscosity calculation for magmatic systems
        
        This novel implementation incorporates:
        1. Multi-scale viscosity modeling from molecular to macroscopic scales
        2. Pressure-temperature-composition coupling effects
        3. Structural relaxation and non-equilibrium effects
        4. Crystal-melt interaction dynamics
        5. Volatile-induced viscosity modifications
        """
        
        T_K = magma_properties.temperature
        P_Pa = magma_properties.pressure * 1e9 if hasattr(magma_properties, 'pressure') else 1e8  # Default 0.1 GPa
        
        # === 1. Molecular-Scale Viscosity Foundation ===
        # Based on molecular dynamics and free volume theory
        
        # Free volume parameters for silicate melts
        v_f0 = 0.025  # Free volume fraction at reference conditions
        alpha_f = 2.5e-4  # Thermal expansion coefficient of free volume
        beta_f = 1.2e-10  # Pressure coefficient of free volume (Pa^-1)
        
        # Temperature and pressure effects on free volume
        T_ref = 1473.15  # Reference temperature (K)
        P_ref = 1e8  # Reference pressure (Pa)
        
        v_f = v_f0 * (1 + alpha_f * (T_K - T_ref) - beta_f * (P_Pa - P_ref))
        v_f = max(0.001, v_f)  # Prevent negative free volume
        
        # Doolittle equation for free volume viscosity
        B_doolittle = 1.0  # Empirical constant
        eta_free_volume = np.exp(B_doolittle / v_f)
        
        # === 2. Configurational Viscosity Component ===
        # Based on Adam-Gibbs theory with enhanced configurational entropy
        
        # Estimate configurational entropy from temperature
        # S_conf = S_0 + C_p * ln(T/T_g) where T_g is glass transition temperature
        
        # Advanced glass transition temperature calculation
        # Based on composition-dependent Tg models (Lakatos et al., 1972; Bottinga & Richet, 1995)
        T_g_base = await self._calculate_composition_dependent_tg(magma_properties)
        
        # Advanced pressure effect on glass transition (Richet & Bottinga, 1984)
        # dTg/dP varies with composition and temperature
        bulk_modulus = await self._calculate_melt_bulk_modulus(magma_properties, T_K)
        thermal_expansion = magma_properties.thermal_expansion if hasattr(magma_properties, 'thermal_expansion') else 5e-5
        
        # Ehrenfest relation: dTg/dP = Tg * ΔV / Δα where ΔV is volume change, Δα is expansion change
        volume_change_fraction = 0.02  # Typical 2% volume change at glass transition
        expansion_change = thermal_expansion * 0.3  # Change in expansion coefficient
        
        dT_g_dP = T_g_base * volume_change_fraction / (expansion_change * bulk_modulus) if expansion_change > 0 else 0.3e-9
        T_g = T_g_base + dT_g_dP * (P_Pa - P_ref)
        
        # Configurational heat capacity
        C_p_conf = 50.0  # J/mol/K (typical for silicate melts)
        R = 8.314  # J/mol/K
        
        if T_K > T_g:
            S_conf = C_p_conf * np.log(T_K / T_g) / R  # Dimensionless
        else:
            S_conf = 0.1  # Minimal configurational entropy below T_g
        
        # Adam-Gibbs viscosity
        B_AG = 12000  # J/mol (configurational barrier)
        A_AG = 1e-5  # Pa·s (pre-exponential factor)
        
        if S_conf > 0:
            eta_configurational = A_AG * np.exp(B_AG / (R * T_K * S_conf))
        else:
            eta_configurational = 1e15  # Very high viscosity
        
        # === 3. Structural Relaxation Effects ===
        # Non-equilibrium viscosity due to structural relaxation
        
        # Relaxation time calculation
        tau_0 = 1e-14  # Fundamental relaxation time (s)
        E_relax = 250000  # Activation energy for relaxation (J/mol)
        
        tau_relax = tau_0 * np.exp(E_relax / (R * T_K))
        
        # Characteristic time scale of the process (from flow conditions)
        # Estimate from system geometry and flow rates
        characteristic_time = 1.0  # seconds (would be calculated from flow conditions)
        
        # Deborah number
        De = tau_relax / characteristic_time
        
        # Non-equilibrium correction factor
        if De > 0.1:
            # Non-equilibrium regime - higher effective viscosity
            f_relax = 1 + 0.5 * np.log10(1 + De)
        else:
            # Equilibrium regime
            f_relax = 1.0
        
        # === 4. Pressure-Induced Viscosity Enhancement ===
        # Based on activation volume theory with pressure-dependent activation volume
        
        # Activation volume (pressure dependent)
        V_act_0 = 8.0e-6  # m³/mol at reference pressure
        dV_act_dP = -2e-15  # m³/mol/Pa (pressure dependence)
        
        V_act = V_act_0 + dV_act_dP * (P_Pa - P_ref)
        V_act = max(1e-6, V_act)  # Prevent negative activation volume
        
        # Pressure factor
        pressure_factor = np.exp(V_act * (P_Pa - P_ref) / (R * T_K))
        
        # === 5. Crystal-Melt Interaction Effects ===
        # Advanced treatment of crystal effects on melt viscosity
        
        # Estimate crystal fraction from undercooling and cooling rate
        # This would typically come from crystallization models
        T_liquidus = 1400  # Estimated liquidus temperature (K)
        undercooling = max(0, T_liquidus - T_K)
        
        # Crystal nucleation and growth kinetics
        if undercooling > 50:  # Significant undercooling
            # Simplified crystal fraction estimate
            phi_crystal = min(0.5, undercooling / 300)
            
            # Enhanced Einstein-Roscoe equation with shape effects
            phi_max = 0.64  # Random close packing
            
            if phi_crystal > 0 and phi_crystal < phi_max:
                # Aspect ratio effects for non-spherical crystals
                aspect_ratio = 3.0  # Typical for plagioclase/pyroxene
                intrinsic_viscosity = 2.5 * (aspect_ratio**2 + 1) / (aspect_ratio + 1)
                
                # Modified Krieger-Dougherty equation
                eta_relative = (1 - phi_crystal / phi_max)**(-intrinsic_viscosity)
                
                # Shear-thinning effects for high crystal content
                if phi_crystal > 0.3:
                    shear_thinning_factor = 0.8  # Reduces effective viscosity
                    eta_relative *= shear_thinning_factor
            else:
                eta_relative = 1.0
        else:
            eta_relative = 1.0
        
        # === 6. Volatile Effects on Melt Structure ===
        # Advanced treatment of volatile effects on viscosity
        
        # Estimate volatile content effects
        # This would typically come from volatile saturation models
        volatile_effect = 1.0
        
        # Advanced volatile content analysis with rigorous speciation
        if hasattr(magma_properties, 'volatile_content'):
            volatile_content = magma_properties.volatile_content
            
            # Advanced water speciation model
            h2o_wt = getattr(volatile_content, 'H2O', 0.0)
            co2_wt = getattr(volatile_content, 'CO2', 0.0)
            so2_wt = getattr(volatile_content, 'SO2', 0.0)
            
            if h2o_wt > 0:
                # Rigorous melt polymerization calculation from composition
                NBO_T = await self._calculate_rigorous_nbo_t(magma_properties)
                
                # Advanced water speciation with pressure and composition effects
                # Based on Zhang & Ni (2010) and Behrens & Nowak (2003)
                
                # Pressure-dependent speciation constant
                delta_V_spec = -1.2e-6  # m³/mol (volume change for speciation)
                K_spec_1bar = np.exp(-24000 / (8.314 * T_K) + 26.4 / 8.314)
                K_spec = K_spec_1bar * np.exp(-delta_V_spec * (P_Pa - 1e5) / (8.314 * T_K))
                
                # Composition-dependent speciation
                # Al/Si ratio affects OH site preference
                al_si_ratio = await self._get_al_si_ratio(magma_properties)
                K_spec *= (1 + 0.5 * al_si_ratio)  # Al enhances OH formation
                
                # Solve speciation equilibrium with activity coefficients
                activity_h2o = await self._calculate_water_activity(magma_properties, T_K, P_Pa)
                activity_oh = await self._calculate_oh_activity(magma_properties, T_K, P_Pa)
                
                # Mass balance: H2O_total = H2O_mol + 0.5 * OH
                # Equilibrium: K = (a_OH)² / (a_H2O * a_O2-)
                oxygen_activity = await self._calculate_oxygen_activity_advanced(magma_properties, NBO_T, T_K)
                
                # Quadratic solution for molecular water fraction
                total_h2o_mol = h2o_wt / 18.02
                
                # Solve: K * a_H2O * a_O2- = (a_OH)²
                # With mass balance and activity corrections
                discriminant_term = 1 + 4 * K_spec * oxygen_activity * activity_h2o / activity_oh**2
                
                if discriminant_term > 0:
                    f_mol = 2 / (1 + np.sqrt(discriminant_term))
                    f_mol = max(0.01, min(0.99, f_mol))
                else:
                    f_mol = 0.5  # Fallback
                
                f_oh = 1 - f_mol
                
                # Advanced viscosity effects with structural basis
                # Based on Richet et al. (1996) and Whittington et al. (2009)
                
                # Network connectivity reduction
                connectivity_loss = f_mol * total_h2o_mol * 3.8  # Molecular water cavitation
                bridging_loss = f_oh * total_h2o_mol * 1.9      # OH substitution
                
                # Temperature-dependent depolymerization efficiency
                depolymerization_T_factor = 1 + 0.4 * np.exp(-(T_K - 1273) / 150)
                
                total_depolymerization = (connectivity_loss + bridging_loss) * depolymerization_T_factor
                
                # CO2 effects (carbonate vs molecular CO2)
                co2_effect = 1.0
                if co2_wt > 0:
                    # CO2 speciation: CO2_mol ⇌ CO3²⁻ + network modification
                    co2_mol_fraction = await self._calculate_co2_speciation(magma_properties, T_K, P_Pa)
                    co2_depolymerization = co2_wt * (0.8 * co2_mol_fraction + 1.5 * (1 - co2_mol_fraction))
                    co2_effect = 10**(-co2_depolymerization * 0.3)
                
                # Combined volatile effect
                log_reduction = total_depolymerization * 0.75
                volatile_effect = 10**(-log_reduction) * co2_effect
        
        # === 7. Combine All Effects ===
        # Sophisticated combination of different viscosity contributions
        
        # Base viscosity from molecular and configurational effects
        eta_base = np.sqrt(eta_free_volume * eta_configurational)  # Geometric mean
        
        # Apply all correction factors
        eta_total = (eta_base * 
                    pressure_factor * 
                    f_relax * 
                    eta_relative * 
                    volatile_effect)
        
        # === 8. Temperature-Dependent Corrections ===
        # Additional temperature effects not captured above
        
        # Super-Arrhenius behavior near glass transition
        if T_K < T_g + 100:
            # Vogel-Fulcher-Tammann behavior
            T_0 = T_g - 50  # Vogel temperature
            if T_K > T_0:
                vft_factor = np.exp(1000 / (T_K - T_0))
                eta_total *= vft_factor
        
        # High-temperature limit (approach to gas-like behavior)
        if T_K > 2000:
            # Transition to kinetic theory limit
            high_T_factor = np.sqrt(T_K / 2000)
            eta_total /= high_T_factor
        
        # === 9. Physical Constraints and Validation ===
        # Ensure physically reasonable viscosity values
        
        # Minimum viscosity (kinetic theory limit for gases)
        eta_min = 1e-6  # Pa·s
        
        # Maximum viscosity (practical limit for geological processes)
        eta_max = 1e18  # Pa·s
        
        eta_final = np.clip(eta_total, eta_min, eta_max)
        
        # === 10. Uncertainty Quantification ===
        # Add realistic uncertainty bounds for the viscosity calculation
        
        # Temperature-dependent uncertainty
        relative_uncertainty = 0.1 + 0.2 * np.exp(-(T_K - T_g) / 200)
        
        # Store uncertainty information (could be used for error propagation)
        if hasattr(magma_properties, 'viscosity_uncertainty'):
            magma_properties.viscosity_uncertainty = eta_final * relative_uncertainty
        
        return float(eta_final)
    
    async def _calculate_composition_dependent_tg(self, magma_properties: MagmaProperties) -> float:
        """
        Calculate composition-dependent glass transition temperature
        Based on Lakatos et al. (1972) and Bottinga & Richet (1995)
        """
        # Extract composition from magma properties
        composition = magma_properties.composition
        
        # Base Tg contributions from each oxide (K)
        tg_contributions = {
            'SiO2': 1200,   # Strong network former
            'Al2O3': 1100,  # Network former
            'TiO2': 900,    # Weak network former
            'FeO': 700,     # Network modifier
            'MgO': 650,     # Strong network modifier
            'CaO': 680,     # Network modifier
            'Na2O': 500,    # Strong network modifier
            'K2O': 450,     # Very strong network modifier
            'P2O5': 800,    # Network former
            'H2O': 300      # Strong network modifier
        }
        
        # Calculate weighted average Tg
        total_weight = 0
        weighted_tg = 0
        
        for oxide, wt_pct in composition.items():
            if oxide in tg_contributions and wt_pct > 0:
                weight = wt_pct / 100.0
                weighted_tg += weight * tg_contributions[oxide]
                total_weight += weight
        
        if total_weight > 0:
            base_tg = weighted_tg / total_weight
        else:
            base_tg = 700  # Default for unknown composition
        
        # Non-linear mixing effects (Lakatos model)
        # Network connectivity effects
        network_formers = composition.get('SiO2', 0) + composition.get('Al2O3', 0)
        network_modifiers = (composition.get('CaO', 0) + composition.get('MgO', 0) + 
                           composition.get('Na2O', 0) + composition.get('K2O', 0))
        
        if network_formers > 0:
            modifier_ratio = network_modifiers / network_formers
            # Higher modifier ratio reduces Tg
            connectivity_correction = -50 * modifier_ratio
        else:
            connectivity_correction = 0
        
        return base_tg + connectivity_correction
    
    async def _calculate_melt_bulk_modulus(self, magma_properties: MagmaProperties, temperature: float) -> float:
        """Calculate melt bulk modulus from composition and temperature"""
        composition = magma_properties.composition
        
        # Oxide bulk modulus contributions (GPa) at reference temperature
        bulk_modulus_contributions = {
            'SiO2': 37.0,   # Stiff network
            'Al2O3': 25.0,  # Moderate stiffness
            'TiO2': 20.0,
            'FeO': 15.0,
            'MgO': 18.0,
            'CaO': 12.0,
            'Na2O': 8.0,    # Soft network
            'K2O': 6.0,     # Very soft
            'P2O5': 22.0,
            'H2O': 2.0      # Very compressible
        }
        
        # Calculate composite bulk modulus (Voigt-Reuss-Hill average)
        total_weight = 0
        weighted_bulk = 0
        
        for oxide, wt_pct in composition.items():
            if oxide in bulk_modulus_contributions and wt_pct > 0:
                weight = wt_pct / 100.0
                weighted_bulk += weight * bulk_modulus_contributions[oxide]
                total_weight += weight
        
        if total_weight > 0:
            bulk_modulus_ref = weighted_bulk / total_weight
        else:
            bulk_modulus_ref = 20.0  # Default
        
        # Temperature dependence (dK/dT typically negative)
        dK_dT = -0.01  # GPa/K (typical for silicate melts)
        T_ref = 1473.15  # Reference temperature
        
        bulk_modulus = bulk_modulus_ref + dK_dT * (temperature - T_ref)
        
        return max(1.0, bulk_modulus)  # Minimum 1 GPa
    
    async def _calculate_rigorous_nbo_t(self, magma_properties: MagmaProperties) -> float:
        """Calculate rigorous NBO/T from detailed composition analysis"""
        composition = magma_properties.composition
        
        # Convert wt% to molar fractions
        molar_masses = {
            'SiO2': 60.08, 'Al2O3': 101.96, 'TiO2': 79.87, 'FeO': 71.84,
            'MgO': 40.30, 'CaO': 56.08, 'Na2O': 61.98, 'K2O': 94.20,
            'P2O5': 141.94, 'H2O': 18.02
        }
        
        moles = {}
        total_moles = 0
        
        for oxide, wt_pct in composition.items():
            if oxide in molar_masses and wt_pct > 0:
                if oxide in ['Al2O3', 'P2O5']:  # These produce 2 cations
                    moles[oxide] = wt_pct / molar_masses[oxide] * 2
                elif oxide in ['Na2O', 'K2O']:  # These produce 2 cations
                    moles[oxide] = wt_pct / molar_masses[oxide] * 2
                else:
                    moles[oxide] = wt_pct / molar_masses[oxide]
                total_moles += moles[oxide]
        
        if total_moles == 0:
            return 0.5  # Default
        
        # Network formers (T sites)
        Si = moles.get('SiO2', 0)
        Al = moles.get('Al2O3', 0)  # Already multiplied by 2
        P = moles.get('P2O5', 0)    # Already multiplied by 2
        Ti = moles.get('TiO2', 0) * 0.5  # Ti can be network former or modifier
        
        total_T = Si + Al + P + Ti
        
        # Network modifiers (charge-balancing cations)
        Ca = moles.get('CaO', 0)
        Mg = moles.get('MgO', 0)
        Fe = moles.get('FeO', 0)
        Na = moles.get('Na2O', 0)  # Already multiplied by 2
        K = moles.get('K2O', 0)    # Already multiplied by 2
        
        total_modifiers = Ca + Mg + Fe + Na + K
        
        # Al charge balance: Al acts as network former when charge-balanced by alkalis/alkaline earths
        charge_balance_deficit = max(0, Al - total_modifiers)
        effective_Al = Al - charge_balance_deficit
        effective_T = Si + effective_Al + P + Ti
        
        # Advanced NBO/T calculation with rigorous charge balance
        if effective_T > 0:
            # Rigorous calculation considering actual cation charges and coordination
            
            # Divalent cations (charge = +2, create 1 NBO per cation)
            divalent_nbo = Ca + Mg + Fe  # Each creates 1 NBO
            
            # Monovalent cations (charge = +1, create 0.5 NBO per cation)
            monovalent_nbo = 0.5 * (Na + K)  # Each creates 0.5 NBO
            
            # Total NBO from charge balance
            # NBO = 2 * (sum of modifier charges - Al charge balance)
            modifier_charge = 2*Ca + 2*Mg + 2*Fe + 1*Na + 1*K
            al_charge_balance = min(modifier_charge, 3*Al)  # Al can only be charge-balanced up to available modifiers
            
            excess_modifier_charge = max(0, modifier_charge - al_charge_balance)
            total_NBO = excess_modifier_charge  # Each excess positive charge creates 1 NBO
            
            # Advanced corrections for mixed alkali effects and coordination changes
            # Mixed alkali effect reduces NBO/T (Mysen & Richet, 2005)
            if Na > 0 and K > 0:
                mixed_alkali_factor = 1 - 0.1 * min(Na, K) / (Na + K)
                total_NBO *= mixed_alkali_factor
            
            # High-field strength cation effects (Ti, P)
            # These can act as network formers or modifiers depending on conditions
            Ti_effect = 0.5 * moles.get('TiO2', 0)  # Ti partially acts as network former
            P_effect = P * 0.8  # P mostly acts as network former
            
            # Adjust effective tetrahedral sites
            effective_T_corrected = Si + effective_Al + Ti_effect + P_effect
            
            if effective_T_corrected > 0:
                nbo_t = total_NBO / effective_T_corrected
            else:
                nbo_t = 4.0  # Fully depolymerized
        else:
            nbo_t = 4.0  # No network formers = fully depolymerized
        
        return max(0.0, min(4.0, nbo_t))  # Physical limits
    
    async def _get_al_si_ratio(self, magma_properties: MagmaProperties) -> float:
        """Get Al/Si molar ratio from composition"""
        composition = magma_properties.composition
        
        sio2_wt = composition.get('SiO2', 0)
        al2o3_wt = composition.get('Al2O3', 0)
        
        if sio2_wt > 0:
            # Convert to molar ratio
            si_moles = sio2_wt / 60.08
            al_moles = al2o3_wt / 101.96 * 2  # Al2O3 → 2Al
            
            return al_moles / si_moles if si_moles > 0 else 0
        else:
            return 0
    
    async def _calculate_water_activity(self, magma_properties: MagmaProperties, 
                                      temperature: float, pressure: float) -> float:
        """Calculate water activity with composition and P-T effects"""
        composition = magma_properties.composition
        
        # Base activity coefficient model (Zhang & Ni, 2010)
        sio2_frac = composition.get('SiO2', 0) / 100.0
        al2o3_frac = composition.get('Al2O3', 0) / 100.0
        alkali_frac = (composition.get('Na2O', 0) + composition.get('K2O', 0)) / 100.0
        
        # Interaction parameters (J/mol)
        W_Si_H2O = 3200
        W_Al_H2O = -1800
        W_alkali_H2O = -1200
        
        RT = 8.314 * temperature
        
        ln_gamma = (W_Si_H2O * sio2_frac + W_Al_H2O * al2o3_frac + 
                   W_alkali_H2O * alkali_frac) / RT
        
        # Pressure correction (excess volume)
        V_excess = 3.2e-6  # m³/mol
        ln_gamma += V_excess * pressure / RT
        
        return np.exp(ln_gamma)
    
    async def _calculate_oh_activity(self, magma_properties: MagmaProperties,
                                   temperature: float, pressure: float) -> float:
        """Calculate OH activity coefficient"""
        composition = magma_properties.composition
        
        # OH groups interact favorably with Al sites
        al2o3_frac = composition.get('Al2O3', 0) / 100.0
        sio2_frac = composition.get('SiO2', 0) / 100.0
        
        W_Al_OH = -2800  # Favorable interaction
        W_Si_OH = 1800   # Unfavorable interaction
        
        RT = 8.314 * temperature
        
        ln_gamma = (W_Al_OH * al2o3_frac + W_Si_OH * sio2_frac) / RT
        
        return np.exp(ln_gamma)
    
    async def _calculate_oxygen_activity_advanced(self, magma_properties: MagmaProperties,
                                                nbo_t: float, temperature: float) -> float:
        """Advanced oxygen activity calculation"""
        composition = magma_properties.composition
        
        # Iron redox equilibrium affects oxygen activity
        feo_wt = composition.get('FeO', 0)
        
        if feo_wt > 0:
            # Kress & Carmichael (1991) iron redox model
            ln_fe3_fe2 = 0.196 * (25050 / temperature - 14.98)
            
            # Composition effects on redox
            sio2_effect = np.log(composition.get('SiO2', 50) / 50) * 0.5
            al2o3_effect = np.log(composition.get('Al2O3', 15) / 15) * 0.3
            
            ln_fe3_fe2 += sio2_effect + al2o3_effect
            
            fe3_fe2_ratio = np.exp(ln_fe3_fe2)
            fe3_fraction = fe3_fe2_ratio / (1 + fe3_fe2_ratio)
        else:
            fe3_fraction = 0.1  # Default oxidation state
        
        # Oxygen activity from structure and redox
        log_ao2 = -8.0 + 3.0 * nbo_t + 6.0 * fe3_fraction * feo_wt / 100.0
        
        return 10**log_ao2
    
    async def _calculate_co2_speciation(self, magma_properties: MagmaProperties,
                                      temperature: float, pressure: float) -> float:
        """Calculate CO2 speciation (molecular vs carbonate)"""
        composition = magma_properties.composition
        
        # CO2 speciation depends on melt basicity and pressure
        # More basic melts favor carbonate formation
        
        # Calculate melt basicity index
        cao_wt = composition.get('CaO', 0)
        mgo_wt = composition.get('MgO', 0)
        na2o_wt = composition.get('Na2O', 0)
        k2o_wt = composition.get('K2O', 0)
        sio2_wt = composition.get('SiO2', 0)
        
        basicity_index = (cao_wt + mgo_wt + na2o_wt + k2o_wt) / (sio2_wt + 1)
        
        # Pressure favors molecular CO2, temperature favors carbonate
        # Empirical model based on experimental data
        
        P_GPa = pressure / 1e9
        
        # Equilibrium constant for CO2_mol ⇌ CO3²⁻
        delta_H = -15000  # J/mol (exothermic carbonate formation)
        delta_S = -45     # J/mol/K
        
        K_carb = np.exp(-(delta_H - temperature * delta_S) / (8.314 * temperature))
        
        # Pressure and basicity effects
        K_carb *= (1 + basicity_index) / (1 + 0.5 * P_GPa)
        
        # Fraction of molecular CO2
        f_mol_co2 = 1 / (1 + K_carb)
        
        return max(0.1, min(0.9, f_mol_co2))
    
    async def _calculate_reynolds_number(
        self, 
        magma_properties: MagmaProperties, 
        velocity: float, 
        characteristic_length: float
    ) -> float:
        """Calculate Reynolds number for flow characterization"""
        return (magma_properties.density * velocity * characteristic_length) / magma_properties.viscosity
    
    async def _calculate_prandtl_number(self, magma_properties: MagmaProperties) -> float:
        """Calculate Prandtl number for heat transfer"""
        return (magma_properties.viscosity * magma_properties.heat_capacity) / magma_properties.thermal_conductivity
    
    async def _calculate_nusselt_number(self, reynolds_number: float, magma_properties: MagmaProperties) -> float:
        """Calculate Nusselt number for heat transfer coefficient"""
        prandtl_number = await self._calculate_prandtl_number(magma_properties)
        
        if reynolds_number < 2300:  # Laminar flow
            return 3.66
        else:  # Turbulent flow
            return 0.023 * (reynolds_number ** 0.8) * (prandtl_number ** 0.4)
    
    async def _determine_flow_regime(
        self, 
        magma_properties: MagmaProperties, 
        connection: ConnectionProperties, 
        pressure_gradient: float
    ) -> str:
        """Determine flow regime (laminar, turbulent, or transitional)"""
        # Estimate velocity for regime determination
        estimated_velocity = abs(pressure_gradient) * connection.hydraulic_diameter**2 / (32 * magma_properties.viscosity)
        reynolds_number = await self._calculate_reynolds_number(
            magma_properties, estimated_velocity, connection.hydraulic_diameter
        )
        
        if reynolds_number < 2300:
            return 'laminar'
        elif reynolds_number > 4000:
            return 'turbulent'
        else:
            return 'transitional'
    
    async def _calculate_laminar_flow_rate(
        self,
        from_chamber: ChamberState,
        to_chamber: ChamberState,
        connection: ConnectionProperties,
        pressure_gradient: float
    ) -> float:
        """Calculate flow rate for laminar flow using Hagen-Poiseuille equation"""
        effective_viscosity = await self._calculate_effective_viscosity(from_chamber.magma_properties)
        
        if connection.geometry.get('shape') == 'circular':
            radius = connection.hydraulic_diameter / 2
            flow_rate = (np.pi * radius**4 * abs(pressure_gradient)) / (8 * effective_viscosity * connection.length)
        else:
            # Rectangular or other cross-sections
            flow_rate = (connection.cross_sectional_area**2 * abs(pressure_gradient)) / (
                12 * effective_viscosity * connection.length
            )
        
        return flow_rate if pressure_gradient > 0 else -flow_rate
    
    async def _calculate_turbulent_flow_rate(
        self,
        from_chamber: ChamberState,
        to_chamber: ChamberState,
        connection: ConnectionProperties,
        pressure_gradient: float
    ) -> float:
        """Calculate flow rate for turbulent flow using Darcy-Weisbach equation"""
        # Iterative solution for turbulent flow
        density = from_chamber.magma_properties.density
        
        # Initial velocity estimate
        velocity = np.sqrt(2 * abs(pressure_gradient) * connection.hydraulic_diameter / (4 * density))
        
        # Iterate to find consistent velocity and friction factor
        for _ in range(10):
            reynolds_number = await self._calculate_reynolds_number(
                from_chamber.magma_properties, velocity, connection.hydraulic_diameter
            )
            friction_factor = await self._calculate_friction_factor(reynolds_number, connection.roughness)
            
            # Update velocity
            velocity = np.sqrt(2 * abs(pressure_gradient) * connection.hydraulic_diameter / 
                             (4 * friction_factor * density))
        
        flow_rate = velocity * connection.cross_sectional_area * density
        return flow_rate if pressure_gradient > 0 else -flow_rate
    
    async def _calculate_transitional_flow_rate(
        self,
        from_chamber: ChamberState,
        to_chamber: ChamberState,
        connection: ConnectionProperties,
        pressure_gradient: float
    ) -> float:
        """Calculate flow rate for transitional flow using interpolation"""
        laminar_flow = await self._calculate_laminar_flow_rate(
            from_chamber, to_chamber, connection, pressure_gradient
        )
        turbulent_flow = await self._calculate_turbulent_flow_rate(
            from_chamber, to_chamber, connection, pressure_gradient
        )
        
        # Linear interpolation based on Reynolds number
        estimated_velocity = abs(pressure_gradient) * connection.hydraulic_diameter**2 / (32 * from_chamber.magma_properties.viscosity)
        reynolds_number = await self._calculate_reynolds_number(
            from_chamber.magma_properties, estimated_velocity, connection.hydraulic_diameter
        )
        
        interpolation_factor = (reynolds_number - 2300) / (4000 - 2300)
        interpolation_factor = max(0, min(1, interpolation_factor))
        
        return laminar_flow * (1 - interpolation_factor) + turbulent_flow * interpolation_factor
    
    async def _calculate_friction_factor(self, reynolds_number: float, roughness: float) -> float:
        """Calculate friction factor using Colebrook-White equation"""
        if reynolds_number < 2300:
            return 64 / reynolds_number
        else:
            # Approximation of Colebrook-White equation for turbulent flow
            return 0.25 / (np.log10(roughness / 3.7 + 5.74 / (reynolds_number ** 0.9))) ** 2    

    # ========================================================================
    # System State Management Methods
    # ========================================================================
    
    async def _initialize_system_state(
        self,
        chamber_states: Dict[str, ChamberState],
        connections: Dict[str, ConnectionProperties]
    ) -> SystemCouplingState:
        """Initialize system coupling state"""
        n_chambers = len(chamber_states)
        
        # Initialize coupling matrices
        pressure_coupling_matrix = np.zeros((n_chambers, n_chambers))
        mass_transfer_matrix = np.zeros((n_chambers, n_chambers))
        energy_transfer_matrix = np.zeros((n_chambers, n_chambers))
        
        # Calculate initial transfer flows
        transfer_flows = {}
        chamber_ids = list(chamber_states.keys())
        
        for connection_id, connection in connections.items():
            from_chamber = chamber_states[connection.from_chamber_id]
            to_chamber = chamber_states[connection.to_chamber_id]
            
            transfer_flow = await self.calculate_magma_transfer_rates(
                from_chamber, to_chamber, connection
            )
            transfer_flows[connection_id] = transfer_flow
            
            # Update coupling matrices
            from_idx = chamber_ids.index(connection.from_chamber_id)
            to_idx = chamber_ids.index(connection.to_chamber_id)
            
            mass_transfer_matrix[from_idx, to_idx] = transfer_flow.flow_rate
            energy_transfer_matrix[from_idx, to_idx] = transfer_flow.heat_transfer_rate
        
        # Calculate system metrics
        total_mass = sum(chamber.mass for chamber in chamber_states.values())
        total_energy = sum(
            chamber.mass * chamber.magma_properties.heat_capacity * chamber.magma_properties.temperature
            for chamber in chamber_states.values()
        )
        system_pressure = np.mean([chamber.pressure for chamber in chamber_states.values()])
        
        return SystemCouplingState(
            system_id=f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            chambers=chamber_states,
            connections=connections,
            transfer_flows=transfer_flows,
            pressure_coupling_matrix=pressure_coupling_matrix,
            mass_transfer_matrix=mass_transfer_matrix,
            energy_transfer_matrix=energy_transfer_matrix,
            system_pressure=system_pressure,
            total_mass=total_mass,
            total_energy=total_energy,
            entropy_production=0.0,
            stability_metrics={},
            feedback_strengths={},
            critical_thresholds={},
            eruption_potential={}
        )
    
    async def _update_pressure_coupling(self, system_state: SystemCouplingState, time_step: float):
        """Update pressure coupling between chambers"""
        chamber_ids = list(system_state.chambers.keys())
        n_chambers = len(chamber_ids)
        
        # Build pressure coupling matrix
        coupling_matrix = np.zeros((n_chambers, n_chambers))
        pressure_vector = np.array([system_state.chambers[cid].pressure for cid in chamber_ids])
        
        for connection_id, connection in system_state.connections.items():
            from_idx = chamber_ids.index(connection.from_chamber_id)
            to_idx = chamber_ids.index(connection.to_chamber_id)
            
            # Calculate coupling strength based on connection properties
            coupling_strength = connection.cross_sectional_area / (connection.length * connection.tortuosity)
            
            coupling_matrix[from_idx, to_idx] = -coupling_strength
            coupling_matrix[to_idx, from_idx] = -coupling_strength
            coupling_matrix[from_idx, from_idx] += coupling_strength
            coupling_matrix[to_idx, to_idx] += coupling_strength
        
        # Solve pressure coupling equation
        pressure_change = np.linalg.solve(
            np.eye(n_chambers) - time_step * coupling_matrix,
            pressure_vector
        ) - pressure_vector
        
        # Update chamber pressures
        for i, chamber_id in enumerate(chamber_ids):
            system_state.chambers[chamber_id].pressure += pressure_change[i]
        
        system_state.pressure_coupling_matrix = coupling_matrix
    
    async def _calculate_transfer_rates(self, system_state: SystemCouplingState, time_step: float):
        """Calculate and update magma transfer rates"""
        for connection_id, connection in system_state.connections.items():
            from_chamber = system_state.chambers[connection.from_chamber_id]
            to_chamber = system_state.chambers[connection.to_chamber_id]
            
            # Recalculate transfer flow with updated pressures
            transfer_flow = await self.calculate_magma_transfer_rates(
                from_chamber, to_chamber, connection
            )
            
            system_state.transfer_flows[connection_id] = transfer_flow
    
    async def _update_chamber_states(self, system_state: SystemCouplingState, time_step: float):
        """Update individual chamber states based on transfer flows"""
        for chamber_id, chamber_state in system_state.chambers.items():
            mass_change = 0.0
            energy_change = 0.0
            
            # Calculate mass and energy changes from all connections
            for connection_id, transfer_flow in system_state.transfer_flows.items():
                connection = system_state.connections[connection_id]
                
                if connection.from_chamber_id == chamber_id:
                    # Outflow from this chamber
                    mass_change -= transfer_flow.flow_rate * time_step
                    energy_change -= transfer_flow.heat_transfer_rate * time_step
                elif connection.to_chamber_id == chamber_id:
                    # Inflow to this chamber
                    mass_change += transfer_flow.flow_rate * time_step
                    energy_change += transfer_flow.heat_transfer_rate * time_step
            
            # Update chamber mass and energy
            chamber_state.mass += mass_change
            
            # Update temperature based on energy change
            if chamber_state.mass > 0:
                temperature_change = energy_change / (
                    chamber_state.mass * chamber_state.magma_properties.heat_capacity
                )
                chamber_state.magma_properties.temperature += temperature_change
            
            # Update chamber volume based on mass change and density
            chamber_state.volume = chamber_state.mass / chamber_state.magma_properties.density
            
            # Update last updated timestamp
            chamber_state.last_updated = datetime.now()
    
    async def _check_system_stability(self, system_state: SystemCouplingState) -> float:
        """Check system stability and return stability metric"""
        stability_factors = []
        
        # Pressure stability
        pressures = [chamber.pressure for chamber in system_state.chambers.values()]
        pressure_variance = np.var(pressures) / np.mean(pressures)**2 if np.mean(pressures) > 0 else 0
        pressure_stability = 1.0 / (1.0 + pressure_variance)
        stability_factors.append(pressure_stability)
        
        # Flow stability
        flow_rates = [abs(flow.flow_rate) for flow in system_state.transfer_flows.values()]
        if flow_rates:
            flow_variance = np.var(flow_rates) / np.mean(flow_rates)**2 if np.mean(flow_rates) > 0 else 0
            flow_stability = 1.0 / (1.0 + flow_variance)
            stability_factors.append(flow_stability)
        
        # Temperature stability
        temperatures = [chamber.magma_properties.temperature for chamber in system_state.chambers.values()]
        temp_variance = np.var(temperatures) / np.mean(temperatures)**2 if np.mean(temperatures) > 0 else 0
        temp_stability = 1.0 / (1.0 + temp_variance)
        stability_factors.append(temp_stability)
        
        return np.mean(stability_factors)
    
    async def _adapt_time_step(self, system_state: SystemCouplingState, current_time_step: float) -> float:
        """Adapt time step based on system stability"""
        stability = await self._check_system_stability(system_state)
        
        if stability < 0.5:
            # Reduce time step for unstable conditions
            new_time_step = current_time_step * 0.5
        elif stability > 0.9:
            # Increase time step for stable conditions
            new_time_step = current_time_step * 1.1
        else:
            new_time_step = current_time_step
        
        # Enforce time step limits
        new_time_step = max(self.config['simulation']['min_time_step'], new_time_step)
        new_time_step = min(self.config['simulation']['max_time_step'], new_time_step)
        
        return new_time_step
    
    # ========================================================================
    # Validation and Utility Methods
    # ========================================================================
    
    async def _validate_simulation_state(self, system_state: SystemCouplingState):
        """Validate simulation state for physical consistency"""
        # Check mass conservation
        total_mass = sum(chamber.mass for chamber in system_state.chambers.values())
        mass_conservation_error = abs(total_mass - system_state.total_mass) / system_state.total_mass
        
        if mass_conservation_error > 0.01:  # 1% tolerance
            self.logger.warning(f"Mass conservation error: {mass_conservation_error:.3f}")
        
        # Check energy conservation
        total_energy = sum(
            chamber.mass * chamber.magma_properties.heat_capacity * chamber.magma_properties.temperature
            for chamber in system_state.chambers.values()
        )
        energy_conservation_error = abs(total_energy - system_state.total_energy) / system_state.total_energy
        
        if energy_conservation_error > 0.01:  # 1% tolerance
            self.logger.warning(f"Energy conservation error: {energy_conservation_error:.3f}")
        
        # Check for negative pressures or temperatures
        for chamber_id, chamber in system_state.chambers.items():
            if chamber.pressure < 0:
                self.logger.error(f"Negative pressure in chamber {chamber_id}: {chamber.pressure}")
            if chamber.magma_properties.temperature < 273:  # Below 0°C
                self.logger.error(f"Unphysical temperature in chamber {chamber_id}: {chamber.magma_properties.temperature}")
    
    async def _save_system_state(self, system_state: SystemCouplingState):
        """Save system state to database"""
        if self.db_integration:
            try:
                # Convert system state to serializable format
                state_data = {
                    'system_id': system_state.system_id,
                    'timestamp': system_state.timestamp.isoformat(),
                    'system_pressure': system_state.system_pressure,
                    'total_mass': system_state.total_mass,
                    'total_energy': system_state.total_energy,
                    'stability_metrics': system_state.stability_metrics,
                    'chamber_count': len(system_state.chambers),
                    'connection_count': len(system_state.connections)
                }
                
                # Save to database
                await self.db_integration.save_simulation_state(state_data)
                
            except Exception as e:
                self.logger.error(f"Failed to save system state: {str(e)}")
    
    async def _calculate_relative_error(self, simulated: np.ndarray, experimental: np.ndarray) -> float:
        """Calculate relative error between simulated and experimental data"""
        return np.mean(np.abs(simulated - experimental) / np.abs(experimental))
    
    async def _calculate_correlation(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate correlation coefficient between two datasets"""
        return np.corrcoef(data1.flatten(), data2.flatten())[0, 1]


# ============================================================================
# Factory Functions and Utilities
# ============================================================================

def create_pressure_coupling_engine(config: Optional[Dict[str, Any]] = None) -> PressureCouplingMagmaTransferEngine:
    """
    Factory function to create a pressure coupling and magma transfer engine
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured engine instance
    """
    return PressureCouplingMagmaTransferEngine(config)


def create_chamber_state(
    chamber_id: str,
    geometry: ChamberGeometry3D,
    magma_properties: MagmaProperties,
    initial_pressure: float,
    initial_mass: float
) -> ChamberState:
    """
    Factory function to create a chamber state
    
    Args:
        chamber_id: Unique chamber identifier
        geometry: Chamber geometry
        magma_properties: Magma properties
        initial_pressure: Initial pressure (Pa)
        initial_mass: Initial mass (kg)
        
    Returns:
        Initialized chamber state
    """
    volume = initial_mass / magma_properties.density
    
    return ChamberState(
        chamber_id=chamber_id,
        geometry=geometry,
        magma_properties=magma_properties,
        pressure=initial_pressure,
        volume=volume,
        mass=initial_mass,
        temperature_distribution=np.full((10, 10, 10), magma_properties.temperature),
        composition_distribution={
            oxide: np.full((10, 10, 10), concentration)
            for oxide, concentration in magma_properties.composition.items()
        },
        crystallinity_distribution=np.full((10, 10, 10), magma_properties.crystallinity),
        volatile_distribution={
            species: np.full((10, 10, 10), concentration)
            for species, concentration in magma_properties.volatile_content.items()
        },
        gas_distribution=np.full((10, 10, 10), magma_properties.gas_fraction),
        velocity_field=np.zeros((10, 10, 10, 3)),
        stress_tensor=np.zeros((10, 10, 10, 3, 3)),
        deformation_rate=np.zeros((10, 10, 10, 3, 3)),
        heat_flux=np.zeros((10, 10, 10, 3)),
        mass_flux_boundaries={},
        energy_balance={}
    )


def create_connection_properties(
    connection_id: str,
    from_chamber_id: str,
    to_chamber_id: str,
    length: float,
    diameter: float,
    roughness: float = 1e-6
) -> ConnectionProperties:
    """
    Factory function to create connection properties
    
    Args:
        connection_id: Unique connection identifier
        from_chamber_id: Source chamber ID
        to_chamber_id: Destination chamber ID
        length: Connection length (m)
        diameter: Connection diameter (m)
        roughness: Surface roughness (m)
        
    Returns:
        Initialized connection properties
    """
    cross_sectional_area = np.pi * (diameter / 2) ** 2
    
    return ConnectionProperties(
        connection_id=connection_id,
        from_chamber_id=from_chamber_id,
        to_chamber_id=to_chamber_id,
        geometry={'shape': 'circular', 'diameter': diameter},
        hydraulic_diameter=diameter,
        length=length,
        cross_sectional_area=cross_sectional_area,
        roughness=roughness,
        tortuosity=1.0,
        permeability=1e-12,  # m²
        thermal_conductivity=2.5,  # W/m·K
        heat_transfer_coefficient=100.0,  # W/m²·K
        wall_temperature=1200.0,  # K
        pressure_loss_coefficient=0.1,
        flow_regime='laminar'
    )


# ============================================================================
# Example Usage and Testing
# ============================================================================

async def example_pressure_coupling_simulation():
    """Example usage of the pressure coupling and magma transfer engine"""
    
    # Create engine
    engine = create_pressure_coupling_engine()
    
    # Create example magma properties
    magma_props = MagmaProperties(
        composition={'SiO2': 65.0, 'Al2O3': 15.0, 'FeO': 8.0, 'MgO': 5.0, 'CaO': 7.0},
        temperature=1200.0,  # K
        pressure=1e8,  # Pa
        density=2500.0,  # kg/m³
        viscosity=1e4,  # Pa·s
        crystallinity=0.1,
        volatile_content={'H2O': 3.0, 'CO2': 0.5},
        gas_fraction=0.05,
        surface_tension=0.4,  # N/m
        compressibility=1e-10,  # 1/Pa
        thermal_expansion=3e-5,  # 1/K
        heat_capacity=1200.0,  # J/kg·K
        thermal_conductivity=2.5,  # W/m·K
        electrical_conductivity=1e-3  # S/m
    )
    
    # Create chamber geometry
    geometry = ChamberGeometry3D(
        chamber_id="chamber_1",
        center_coordinates=(0.0, 0.0, -5.0),
        volume=1e9,  # m³
        shape_parameters={'a': 1000, 'b': 800, 'c': 600},
        shape_type='ellipsoid',
        orientation={'strike': 0, 'dip': 0, 'plunge': 0},
        aspect_ratios=(1.25, 1.33, 1.0)
    )
    
    # Create chamber states
    chamber1 = create_chamber_state("chamber_1", geometry, magma_props, 1.5e8, 2.5e12)
    chamber2 = create_chamber_state("chamber_2", geometry, magma_props, 1.0e8, 1.5e12)
    
    # Create connection
    connection = create_connection_properties(
        "connection_1_2", "chamber_1", "chamber_2", 2000.0, 50.0
    )
    
    # Run simulation
    chamber_states = {"chamber_1": chamber1, "chamber_2": chamber2}
    connections = {"connection_1_2": connection}
    
    final_state = await engine.simulate_real_time_pressure_coupling(
        chamber_states, connections, time_step=10.0, simulation_duration=3600.0
    )
    
    print(f"Simulation completed. Final system pressure: {final_state.system_pressure:.2e} Pa")
    return final_state


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_pressure_coupling_simulation())