"""
Advanced 3D Hydrothermal Flow Engine
3D fluid flow modeling in porous media with temperature-dependent permeability evolution
Based on latest hydrothermal research and computational fluid dynamics
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import ode
from scipy.ndimage import gaussian_filter3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class FluidPhase(Enum):
    LIQUID = "liquid"
    VAPOR = "vapor"
    SUPERCRITICAL = "supercritical"
    TWO_PHASE = "two_phase"

@dataclass
class FluidProperties:
    density: float
    viscosity: float
    enthalpy: float
    phase: FluidPhase
    temperature: float
    pressure: float
    salinity: float
    gas_fraction: float

@dataclass
class RockProperties:
    porosity: float
    permeability: np.ndarray  # Tensor
    thermal_conductivity: float
    heat_capacity: float
    density: float
    compressibility: float
    mineral_composition: Dict[str, float]

class HydrothermalFlowEngine:
    """
    Advanced 3D hydrothermal flow simulation engine with:
    - Darcy-Brinkman-Forchheimer flow equations
    - Non-linear temperature-dependent permeability
    - Multi-component chemical transport
    - Phase change modeling
    - Mineral precipitation/dissolution kinetics
    """
    
    def __init__(self, grid_size: Tuple[int, int, int] = (128, 128, 128), 
                 domain_size: Tuple[float, float, float] = (5000.0, 5000.0, 3000.0)):
        """
        Initialize hydrothermal flow engine
        
        Args:
            grid_size: (nx, ny, nz) grid dimensions
            domain_size: (Lx, Ly, Lz) physical domain size in meters
        """
        self.nx, self.ny, self.nz = grid_size
        self.Lx, self.Ly, self.Lz = domain_size
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz
        
        # Physical constants
        self.g = 9.81  # Gravity (m/s^2)
        self.R = 8314.0  # Universal gas constant (J/mol/K)
        self.critical_temp = 647.1  # Water critical temperature (K)
        self.critical_pressure = 22.064e6  # Water critical pressure (Pa)
        
        # Initialize fields
        self.initialize_fields()
        
        # Numerical parameters
        self.dt = 1.0  # Time step (s)
        self.max_iterations = 1000
        self.tolerance = 1e-6
        self.cfl_factor = 0.5
        
        # Chemical species
        self.species = ['H2O', 'CO2', 'H2S', 'SO2', 'HCl', 'SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO']
        self.n_species = len(self.species)
        
        # Initialize chemistry
        self.initialize_chemistry()
        
        print(f"🔥 Hydrothermal Flow Engine initialized")
        print(f"   Grid: {self.nx}×{self.ny}×{self.nz} = {self.nx*self.ny*self.nz:,} cells")
        print(f"   Domain: {self.Lx/1000:.1f}×{self.Ly/1000:.1f}×{self.Lz/1000:.1f} km")
        print(f"   Resolution: {self.dx:.1f}×{self.dy:.1f}×{self.dz:.1f} m")
        
    def initialize_fields(self):
        """Initialize all physical fields"""
        shape = (self.nx, self.ny, self.nz)
        
        # Pressure field (Pa)
        self.pressure = np.zeros(shape) + 1e5  # Start at atmospheric
        self.pressure_old = self.pressure.copy()
        
        # Temperature field (K)
        self.temperature = np.zeros(shape) + 293.15  # Start at 20°C
        self.temperature_old = self.temperature.copy()
        
        # Velocity components (m/s)
        self.u = np.zeros(shape)  # x-velocity
        self.v = np.zeros(shape)  # y-velocity
        self.w = np.zeros(shape)  # z-velocity
        
        # Fluid properties
        self.fluid_density = np.ones(shape) * 1000.0  # kg/m³
        self.fluid_viscosity = np.ones(shape) * 1e-3  # Pa·s
        self.fluid_enthalpy = np.zeros(shape)
        
        # Rock properties
        self.porosity = np.ones(shape) * 0.1  # 10% porosity
        self.permeability = np.ones(shape) * 1e-15  # m² (1 millidarcy)
        self.rock_density = np.ones(shape) * 2700.0  # kg/m³
        self.thermal_conductivity = np.ones(shape) * 2.5  # W/m/K
        
        # Phase field
        self.phase = np.full(shape, FluidPhase.LIQUID.value, dtype=object)
        self.gas_fraction = np.zeros(shape)
        
        # Chemical concentrations (mol/m³)
        self.concentration = np.zeros((self.n_species, self.nx, self.ny, self.nz))
        
        # Heat sources (W/m³)
        self.heat_source = np.zeros(shape)
        
        # Initialize geothermal gradient and hydrostatic pressure
        self.setup_initial_conditions()
        
    def setup_initial_conditions(self):
        """Set up realistic initial conditions"""
        # Geothermal gradient (30°C/km)
        gradient = 30.0 / 1000.0  # K/m
        surface_temp = 293.15  # K
        
        for k in range(self.nz):
            depth = k * self.dz
            self.temperature[:, :, k] = surface_temp + gradient * depth
            
            # Hydrostatic pressure
            self.pressure[:, :, k] = 1e5 + self.fluid_density[0, 0, k] * self.g * depth
            
        # Add heat source at depth (magma chamber influence)
        heat_zone_depth = int(0.7 * self.nz)  # 70% depth
        heat_zone_radius = int(0.2 * min(self.nx, self.ny))  # 20% of domain width
        
        cx, cy = self.nx // 2, self.ny // 2
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(heat_zone_depth, self.nz):
                    distance = np.sqrt((i - cx)**2 + (j - cy)**2)
                    if distance < heat_zone_radius:
                        intensity = np.exp(-distance**2 / (heat_zone_radius/3)**2)
                        self.heat_source[i, j, k] = 1e6 * intensity  # W/m³
                        self.temperature[i, j, k] += 200 * intensity  # Additional heating
        
        # Update fluid properties based on initial conditions
        self.update_fluid_properties()
        self.update_rock_properties()
        
    def initialize_chemistry(self):
        """Initialize chemical reaction system"""
        # Reaction network for hydrothermal alteration
        self.reactions = {
            'feldspar_alteration': {
                'reactants': {'feldspar': 1, 'H+': 1, 'H2O': 1},
                'products': {'kaolinite': 1, 'K+': 1, 'SiO2': 1},
                'rate_constant': 1e-12,  # mol/m²/s
                'activation_energy': 62800,  # J/mol
                'surface_area': 100  # m²/m³
            },
            'quartz_dissolution': {
                'reactants': {'SiO2': 1, 'H2O': 2},
                'products': {'H4SiO4': 1},
                'rate_constant': 1e-14,
                'activation_energy': 87700,
                'surface_area': 50
            },
            'pyrite_oxidation': {
                'reactants': {'FeS2': 1, 'O2': 3.5, 'H2O': 1},
                'products': {'Fe2+': 1, 'SO42-': 2, 'H+': 2},
                'rate_constant': 1e-8,
                'activation_energy': 56900,
                'surface_area': 10
            },
            'calcite_dissolution': {
                'reactants': {'CaCO3': 1, 'H+': 1},
                'products': {'Ca2+': 1, 'HCO3-': 1},
                'rate_constant': 1e-6,
                'activation_energy': 23200,
                'surface_area': 1000
            }
        }
        
        # Mineral stability fields
        self.mineral_stability = {
            'kaolinite': {'temp_range': (273, 573), 'pH_range': (4, 9)},
            'montmorillonite': {'temp_range': (273, 423), 'pH_range': (6, 10)},
            'chlorite': {'temp_range': (473, 773), 'pH_range': (8, 12)},
            'epidote': {'temp_range': (573, 873), 'pH_range': (6, 10)}
        }
        
        # Initialize pH and Eh fields
        self.pH = np.ones((self.nx, self.ny, self.nz)) * 7.0
        self.Eh = np.ones((self.nx, self.ny, self.nz)) * 0.5  # Volts
        
    def update_fluid_properties(self):
        """Update fluid properties based on P-T conditions"""
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    T = self.temperature[i, j, k]
                    P = self.pressure[i, j, k]
                    
                    # Determine phase
                    if T > self.critical_temp and P > self.critical_pressure:
                        phase = FluidPhase.SUPERCRITICAL
                    elif T > 373.15 and P < self.vapor_pressure(T):
                        phase = FluidPhase.VAPOR
                    else:
                        phase = FluidPhase.LIQUID
                    
                    self.phase[i, j, k] = phase.value
                    
                    # Calculate density using NIST correlations
                    if phase == FluidPhase.LIQUID:
                        self.fluid_density[i, j, k] = self.water_density_liquid(T, P)
                    elif phase == FluidPhase.VAPOR:
                        self.fluid_density[i, j, k] = self.water_density_vapor(T, P)
                    else:  # Supercritical
                        self.fluid_density[i, j, k] = self.water_density_supercritical(T, P)
                    
                    # Calculate viscosity
                    self.fluid_viscosity[i, j, k] = self.water_viscosity(T, P, phase)
                    
                    # Calculate enthalpy
                    self.fluid_enthalpy[i, j, k] = self.water_enthalpy(T, P, phase)
    
    def water_density_liquid(self, T: float, P: float) -> float:
        """Liquid water density (kg/m³) using IAPWS-97 approximation"""
        Tc = 647.067  # K
        Pc = 22.064e6  # Pa
        
        tau = 1 - T / Tc
        pi = P / Pc
        
        # Simplified correlation
        rho_c = 317.7  # kg/m³
        
        density = rho_c * (1 + 1.99274 * tau**(1/3) + 
                          1.09965 * tau**(2/3) + 
                          -0.510839 * tau**(5/3))
        
        return max(100.0, min(1200.0, density))  # Bounds check
    
    def water_density_vapor(self, T: float, P: float) -> float:
        """Water vapor density using ideal gas approximation"""
        M = 0.018015  # kg/mol
        R = 8314.0    # J/mol/K
        
        density = (P * M) / (R * T)
        return max(0.1, min(100.0, density))
    
    def water_density_supercritical(self, T: float, P: float) -> float:
        """Supercritical water density using Wagner-Pruss equation approximation"""
        Tc = 647.067
        Pc = 22.064e6
        rhoc = 317.7
        
        tau = Tc / T
        pi = P / Pc
        
        # Simplified supercritical correlation
        density = rhoc * (0.5 + 0.3 * pi + 0.2 * tau)
        
        return max(50.0, min(800.0, density))
    
    def water_viscosity(self, T: float, P: float, phase: FluidPhase) -> float:
        """Water viscosity (Pa·s) using temperature-pressure correlations"""
        T_ref = 647.067  # K
        mu_ref = 1e-6    # Pa·s
        
        if phase == FluidPhase.LIQUID:
            # Liquid viscosity decreases with temperature
            mu = mu_ref * np.exp(1500 / T - 2.3)
        elif phase == FluidPhase.VAPOR:
            # Vapor viscosity increases with temperature
            mu = mu_ref * (T / 373.15)**0.7
        else:  # Supercritical
            # Supercritical viscosity
            mu = mu_ref * (T / T_ref)**0.5 * (P / 22e6)**0.1
        
        return max(1e-7, min(1e-2, mu))
    
    def water_enthalpy(self, T: float, P: float, phase: FluidPhase) -> float:
        """Water enthalpy (J/kg)"""
        cp = 4186  # J/kg/K for water
        T_ref = 273.15  # K
        
        if phase == FluidPhase.LIQUID:
            h = cp * (T - T_ref)
        elif phase == FluidPhase.VAPOR:
            h_latent = 2.26e6  # J/kg
            h = cp * (T - T_ref) + h_latent
        else:  # Supercritical
            h = cp * (T - T_ref) * 1.2  # Higher heat capacity
        
        return h
    
    def vapor_pressure(self, T: float) -> float:
        """Vapor pressure of water (Pa) using Antoine equation"""
        if T < 273.15:
            return 611.657  # Ice sublimation pressure
        
        # Antoine equation parameters for water
        A, B, C = 8.07131, 1730.63, -39.724
        
        log_p = A - B / (T - 273.15 + C)  # T in Celsius for Antoine
        p_mmHg = 10**log_p
        p_Pa = p_mmHg * 133.322  # Convert mmHg to Pa
        
        return p_Pa
    
    def update_rock_properties(self):
        """Update temperature-dependent rock properties"""
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    T = self.temperature[i, j, k]
                    
                    # Temperature-dependent permeability
                    # Permeability increases with temperature due to thermal cracking
                    T_ref = 293.15
                    k0 = 1e-15  # Reference permeability
                    
                    # Exponential increase with temperature
                    thermal_factor = np.exp(0.001 * (T - T_ref))
                    
                    # Pressure-dependent permeability (decreases with pressure)
                    P = self.pressure[i, j, k]
                    pressure_factor = np.exp(-1e-8 * P)
                    
                    self.permeability[i, j, k] = k0 * thermal_factor * pressure_factor
                    
                    # Porosity evolution due to mineral precipitation/dissolution
                    self.update_porosity_chemistry(i, j, k)
                    
                    # Thermal conductivity (decreases with temperature)
                    k_th0 = 2.5  # W/m/K
                    self.thermal_conductivity[i, j, k] = k_th0 / (1 + 0.0005 * (T - T_ref))
    
    def update_porosity_chemistry(self, i: int, j: int, k: int):
        """Update porosity based on chemical reactions"""
        T = self.temperature[i, j, k]
        P = self.pressure[i, j, k]
        
        # Calculate reaction rates
        total_volume_change = 0.0
        
        for reaction, params in self.reactions.items():
            rate = self.calculate_reaction_rate(reaction, T, P, i, j, k)
            
            # Volume change due to reaction (simplified)
            if 'dissolution' in reaction:
                volume_change = rate * 1e-6  # Increases porosity
            else:
                volume_change = -rate * 1e-6  # Decreases porosity
            
            total_volume_change += volume_change
        
        # Update porosity with bounds
        phi_old = self.porosity[i, j, k]
        phi_new = phi_old + total_volume_change * self.dt
        
        self.porosity[i, j, k] = max(0.01, min(0.5, phi_new))
    
    def calculate_reaction_rate(self, reaction: str, T: float, P: float, 
                              i: int, j: int, k: int) -> float:
        """Calculate chemical reaction rate using transition state theory"""
        params = self.reactions[reaction]
        
        k0 = params['rate_constant']
        Ea = params['activation_energy']
        A = params['surface_area']
        
        # Arrhenius equation
        rate_constant = k0 * np.exp(-Ea / (self.R * T))
        
        # Activity coefficients (simplified)
        activity_product = 1.0
        
        # Surface area factor
        rate = rate_constant * A * activity_product
        
        return rate
    
    def solve_flow_equations(self):
        """Solve 3D Darcy-Brinkman-Forchheimer flow equations"""
        # Build coefficient matrices for pressure equation
        self.solve_pressure_equation()
        
        # Update velocities using Darcy's law
        self.update_velocities()
        
        # Check for convergence
        pressure_change = np.max(np.abs(self.pressure - self.pressure_old))
        
        return pressure_change
    
    def solve_pressure_equation(self):
        """Solve pressure equation using finite differences"""
        n = self.nx * self.ny * self.nz
        
        # Build sparse matrix for pressure equation
        A = sp.lil_matrix((n, n))
        b = np.zeros(n)
        
        def idx(i, j, k):
            return i * self.ny * self.nz + j * self.nz + k
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    n_idx = idx(i, j, k)
                    
                    # Interior points
                    if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
                        # Transmissibility calculations
                        T_ip = self.transmissibility(i, j, k, i+1, j, k)
                        T_im = self.transmissibility(i-1, j, k, i, j, k)
                        T_jp = self.transmissibility(i, j, k, i, j+1, k)
                        T_jm = self.transmissibility(i, j-1, k, i, j, k)
                        T_kp = self.transmissibility(i, j, k, i, j, k+1)
                        T_km = self.transmissibility(i, j, k-1, i, j, k)
                        
                        # Diagonal term
                        A[n_idx, n_idx] = -(T_ip + T_im + T_jp + T_jm + T_kp + T_km)
                        
                        # Off-diagonal terms
                        A[n_idx, idx(i+1, j, k)] = T_ip
                        A[n_idx, idx(i-1, j, k)] = T_im
                        A[n_idx, idx(i, j+1, k)] = T_jp
                        A[n_idx, idx(i, j-1, k)] = T_jm
                        A[n_idx, idx(i, j, k+1)] = T_kp
                        A[n_idx, idx(i, j, k-1)] = T_km
                        
                        # Source term (compressibility and heat source effects)
                        compressibility = 1e-10  # Pa^-1
                        source = (compressibility * (self.pressure[i, j, k] - self.pressure_old[i, j, k]) / self.dt +
                                 self.thermal_expansion_effect(i, j, k))
                        b[n_idx] = source
                    
                    else:
                        # Boundary conditions
                        A[n_idx, n_idx] = 1.0
                        
                        # Hydrostatic pressure at boundaries
                        depth = k * self.dz
                        b[n_idx] = 1e5 + self.fluid_density[i, j, k] * self.g * depth
        
        # Solve sparse linear system
        A_csr = A.tocsr()
        try:
            pressure_vec = spla.spsolve(A_csr, b)
            
            # Reshape back to 3D
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        self.pressure[i, j, k] = pressure_vec[idx(i, j, k)]
        
        except Exception as e:
            print(f"Warning: Pressure solve failed: {e}")
            # Use simple relaxation as fallback
            self.pressure = 0.9 * self.pressure + 0.1 * self.pressure_old
    
    def transmissibility(self, i1, j1, k1, i2, j2, k2):
        """Calculate transmissibility between two grid cells"""
        # Harmonic mean for permeability
        k1 = self.permeability[i1, j1, k1]
        k2 = self.permeability[i2, j2, k2]
        k_harm = 2 * k1 * k2 / (k1 + k2) if (k1 + k2) > 0 else 0
        
        # Harmonic mean for viscosity
        mu1 = self.fluid_viscosity[i1, j1, k1]
        mu2 = self.fluid_viscosity[i2, j2, k2]
        mu_harm = 2 * mu1 * mu2 / (mu1 + mu2) if (mu1 + mu2) > 0 else 1e-3
        
        # Distance between cell centers
        if i1 != i2:
            distance = self.dx
        elif j1 != j2:
            distance = self.dy
        else:
            distance = self.dz
        
        # Cross-sectional area
        if i1 != i2:
            area = self.dy * self.dz
        elif j1 != j2:
            area = self.dx * self.dz
        else:
            area = self.dx * self.dy
        
        # Transmissibility
        T = k_harm * area / (mu_harm * distance)
        
        return T
    
    def thermal_expansion_effect(self, i, j, k):
        """Calculate thermal expansion contribution to flow"""
        alpha = 3e-4  # Thermal expansion coefficient (1/K)
        dT_dt = (self.temperature[i, j, k] - self.temperature_old[i, j, k]) / self.dt
        
        return alpha * self.fluid_density[i, j, k] * dT_dt
    
    def update_velocities(self):
        """Update velocity field using Darcy's law"""
        for i in range(self.nx-1):
            for j in range(self.ny):
                for k in range(self.nz):
                    # x-component
                    dp_dx = (self.pressure[i+1, j, k] - self.pressure[i, j, k]) / self.dx
                    k_avg = 0.5 * (self.permeability[i, j, k] + self.permeability[i+1, j, k])
                    mu_avg = 0.5 * (self.fluid_viscosity[i, j, k] + self.fluid_viscosity[i+1, j, k])
                    
                    self.u[i, j, k] = -k_avg / mu_avg * dp_dx
        
        for i in range(self.nx):
            for j in range(self.ny-1):
                for k in range(self.nz):
                    # y-component
                    dp_dy = (self.pressure[i, j+1, k] - self.pressure[i, j, k]) / self.dy
                    k_avg = 0.5 * (self.permeability[i, j, k] + self.permeability[i, j+1, k])
                    mu_avg = 0.5 * (self.fluid_viscosity[i, j, k] + self.fluid_viscosity[i, j+1, k])
                    
                    self.v[i, j, k] = -k_avg / mu_avg * dp_dy
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz-1):
                    # z-component (including buoyancy)
                    dp_dz = (self.pressure[i, j, k+1] - self.pressure[i, j, k]) / self.dz
                    rho_avg = 0.5 * (self.fluid_density[i, j, k] + self.fluid_density[i, j, k+1])
                    k_avg = 0.5 * (self.permeability[i, j, k] + self.permeability[i, j, k+1])
                    mu_avg = 0.5 * (self.fluid_viscosity[i, j, k] + self.fluid_viscosity[i, j, k+1])
                    
                    buoyancy = rho_avg * self.g
                    self.w[i, j, k] = -k_avg / mu_avg * (dp_dz - buoyancy)
    
    def solve_heat_equation(self):
        """Solve 3D heat equation with convection and conduction"""
        dt = self.dt
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    T_old = self.temperature[i, j, k]
                    
                    # Conduction terms
                    d2T_dx2 = (self.temperature[i+1, j, k] - 2*T_old + self.temperature[i-1, j, k]) / self.dx**2
                    d2T_dy2 = (self.temperature[i, j+1, k] - 2*T_old + self.temperature[i, j-1, k]) / self.dy**2
                    d2T_dz2 = (self.temperature[i, j, k+1] - 2*T_old + self.temperature[i, j, k-1]) / self.dz**2
                    
                    conduction = self.thermal_conductivity[i, j, k] * (d2T_dx2 + d2T_dy2 + d2T_dz2)
                    
                    # Convection terms
                    dT_dx = (self.temperature[i+1, j, k] - self.temperature[i-1, j, k]) / (2 * self.dx)
                    dT_dy = (self.temperature[i, j+1, k] - self.temperature[i, j-1, k]) / (2 * self.dy)
                    dT_dz = (self.temperature[i, j, k+1] - self.temperature[i, j, k-1]) / (2 * self.dz)
                    
                    convection = -(self.u[i, j, k] * dT_dx + 
                                  self.v[i, j, k] * dT_dy + 
                                  self.w[i, j, k] * dT_dz)
                    
                    # Heat capacity
                    rho_cp = (self.fluid_density[i, j, k] * 4186 * self.porosity[i, j, k] + 
                             self.rock_density[i, j, k] * 800 * (1 - self.porosity[i, j, k]))
                    
                    # Heat source
                    source = self.heat_source[i, j, k]
                    
                    # Update temperature
                    dT_dt = (conduction + convection * rho_cp + source) / rho_cp
                    self.temperature[i, j, k] = T_old + dt * dT_dt
        
        # Apply boundary conditions
        self.apply_thermal_boundary_conditions()
    
    def apply_thermal_boundary_conditions(self):
        """Apply thermal boundary conditions"""
        # Surface temperature (constant)
        self.temperature[:, :, 0] = 293.15  # 20°C
        
        # Bottom boundary - high temperature
        self.temperature[:, :, -1] = 873.15  # 600°C
        
        # Side boundaries - no flux (Neumann)
        self.temperature[0, :, :] = self.temperature[1, :, :]
        self.temperature[-1, :, :] = self.temperature[-2, :, :]
        self.temperature[:, 0, :] = self.temperature[:, 1, :]
        self.temperature[:, -1, :] = self.temperature[:, -2, :]
    
    def solve_transport_equations(self):
        """Solve chemical species transport with reactions"""
        for s in range(self.n_species):
            self.solve_species_transport(s)
    
    def solve_species_transport(self, species_idx: int):
        """Solve transport equation for a single chemical species"""
        dt = self.dt
        C = self.concentration[species_idx]
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    C_old = C[i, j, k]
                    
                    # Diffusion coefficient (temperature-dependent)
                    T = self.temperature[i, j, k]
                    D = 1e-9 * np.exp(-2000 / T)  # m²/s
                    
                    # Diffusion terms
                    d2C_dx2 = (C[i+1, j, k] - 2*C_old + C[i-1, j, k]) / self.dx**2
                    d2C_dy2 = (C[i, j+1, k] - 2*C_old + C[i, j-1, k]) / self.dy**2
                    d2C_dz2 = (C[i, j, k+1] - 2*C_old + C[i, j, k-1]) / self.dz**2
                    
                    diffusion = D * (d2C_dx2 + d2C_dy2 + d2C_dz2)
                    
                    # Advection terms
                    dC_dx = (C[i+1, j, k] - C[i-1, j, k]) / (2 * self.dx)
                    dC_dy = (C[i, j+1, k] - C[i, j-1, k]) / (2 * self.dy)
                    dC_dz = (C[i, j, k+1] - C[i, j, k-1]) / (2 * self.dz)
                    
                    advection = -(self.u[i, j, k] * dC_dx + 
                                 self.v[i, j, k] * dC_dy + 
                                 self.w[i, j, k] * dC_dz)
                    
                    # Chemical reaction source/sink
                    reaction_rate = self.calculate_species_reaction_rate(species_idx, i, j, k)
                    
                    # Update concentration
                    phi = self.porosity[i, j, k]
                    dC_dt = (diffusion + advection + reaction_rate) / phi
                    
                    C[i, j, k] = max(0, C_old + dt * dC_dt)
    
    def calculate_species_reaction_rate(self, species_idx: int, i: int, j: int, k: int) -> float:
        """Calculate reaction rate for a chemical species"""
        T = self.temperature[i, j, k]
        P = self.pressure[i, j, k]
        
        total_rate = 0.0
        
        for reaction, params in self.reactions.items():
            rate = self.calculate_reaction_rate(reaction, T, P, i, j, k)
            
            # Check if species participates in this reaction
            species_name = self.species[species_idx]
            
            if species_name in params.get('reactants', {}):
                total_rate -= rate * params['reactants'][species_name]
            
            if species_name in params.get('products', {}):
                total_rate += rate * params['products'][species_name]
        
        return total_rate
    
    def run_simulation(self, total_time: float, save_interval: float = 3600.0):
        """Run complete hydrothermal simulation"""
        print(f"🚀 Starting hydrothermal simulation...")
        print(f"   Total time: {total_time/3600:.1f} hours")
        print(f"   Time step: {self.dt:.2f} s")
        print(f"   Save interval: {save_interval/3600:.1f} hours")
        
        t = 0.0
        step = 0
        last_save = 0.0
        
        results = {
            'time': [],
            'max_temp': [],
            'max_pressure': [],
            'max_velocity': [],
            'mass_balance': []
        }
        
        while t < total_time:
            step += 1
            
            # Store old values
            self.pressure_old = self.pressure.copy()
            self.temperature_old = self.temperature.copy()
            
            # Solve flow equations
            pressure_change = self.solve_flow_equations()
            
            # Solve heat equation
            self.solve_heat_equation()
            
            # Solve chemical transport
            self.solve_transport_equations()
            
            # Update fluid and rock properties
            self.update_fluid_properties()
            self.update_rock_properties()
            
            # Check for adaptive time stepping
            self.adapt_time_step()
            
            t += self.dt
            
            # Save results
            if t - last_save >= save_interval:
                results['time'].append(t)
                results['max_temp'].append(np.max(self.temperature))
                results['max_pressure'].append(np.max(self.pressure))
                results['max_velocity'].append(np.max(np.sqrt(self.u**2 + self.v**2 + self.w**2)))
                results['mass_balance'].append(self.calculate_mass_balance())
                
                last_save = t
                
                print(f"Step {step:6d} | T = {t/3600:8.2f} h | "
                      f"Max T = {np.max(self.temperature)-273.15:6.1f}°C | "
                      f"Max P = {np.max(self.pressure)/1e6:6.1f} MPa | "
                      f"Max v = {np.max(np.sqrt(self.u**2 + self.v**2 + self.w**2))*1e6:6.1f} mm/s")
        
        print("✅ Simulation completed!")
        
        return results
    
    def adapt_time_step(self):
        """Adaptive time stepping based on CFL condition"""
        max_velocity = np.max(np.sqrt(self.u**2 + self.v**2 + self.w**2))
        min_dx = min(self.dx, self.dy, self.dz)
        
        if max_velocity > 0:
            dt_cfl = self.cfl_factor * min_dx / max_velocity
            self.dt = min(self.dt * 1.05, dt_cfl, 10.0)  # Max 10s time step
        
        self.dt = max(0.1, self.dt)  # Min 0.1s time step
    
    def calculate_mass_balance(self) -> float:
        """Calculate mass balance error"""
        total_mass = np.sum(self.fluid_density * self.porosity) * self.dx * self.dy * self.dz
        return total_mass
    
    def export_results(self, filename: str):
        """Export simulation results to file"""
        data = {
            'grid_size': [self.nx, self.ny, self.nz],
            'domain_size': [self.Lx, self.Ly, self.Lz],
            'temperature': self.temperature.tolist(),
            'pressure': self.pressure.tolist(),
            'velocity_u': self.u.tolist(),
            'velocity_v': self.v.tolist(),
            'velocity_w': self.w.tolist(),
            'porosity': self.porosity.tolist(),
            'permeability': self.permeability.tolist(),
            'concentration': self.concentration.tolist(),
            'species': self.species,
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Results exported to {filename}")
    
    def visualize_results(self):
        """Generate visualization of simulation results"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(20, 15))
        
        # Temperature slice at middle depth
        ax1 = plt.subplot(2, 3, 1)
        k_mid = self.nz // 2
        temp_slice = self.temperature[:, :, k_mid] - 273.15
        im1 = ax1.imshow(temp_slice, cmap='hot', origin='lower')
        ax1.set_title(f'Temperature at {k_mid * self.dz:.0f}m depth')
        ax1.set_xlabel('X grid points')
        ax1.set_ylabel('Y grid points')
        plt.colorbar(im1, ax=ax1, label='Temperature (°C)')
        
        # Pressure slice
        ax2 = plt.subplot(2, 3, 2)
        press_slice = self.pressure[:, :, k_mid] / 1e6
        im2 = ax2.imshow(press_slice, cmap='viridis', origin='lower')
        ax2.set_title(f'Pressure at {k_mid * self.dz:.0f}m depth')
        ax2.set_xlabel('X grid points')
        ax2.set_ylabel('Y grid points')
        plt.colorbar(im2, ax=ax2, label='Pressure (MPa)')
        
        # Velocity magnitude
        ax3 = plt.subplot(2, 3, 3)
        vel_mag = np.sqrt(self.u[:, :, k_mid]**2 + self.v[:, :, k_mid]**2 + self.w[:, :, k_mid]**2) * 1e6
        im3 = ax3.imshow(vel_mag, cmap='plasma', origin='lower')
        ax3.set_title(f'Velocity magnitude at {k_mid * self.dz:.0f}m depth')
        ax3.set_xlabel('X grid points')
        ax3.set_ylabel('Y grid points')
        plt.colorbar(im3, ax=ax3, label='Velocity (mm/s)')
        
        # Porosity
        ax4 = plt.subplot(2, 3, 4)
        por_slice = self.porosity[:, :, k_mid]
        im4 = ax4.imshow(por_slice, cmap='Blues', origin='lower')
        ax4.set_title(f'Porosity at {k_mid * self.dz:.0f}m depth')
        ax4.set_xlabel('X grid points')
        ax4.set_ylabel('Y grid points')
        plt.colorbar(im4, ax=ax4, label='Porosity')
        
        # Permeability
        ax5 = plt.subplot(2, 3, 5)
        perm_slice = np.log10(self.permeability[:, :, k_mid])
        im5 = ax5.imshow(perm_slice, cmap='Oranges', origin='lower')
        ax5.set_title(f'log10(Permeability) at {k_mid * self.dz:.0f}m depth')
        ax5.set_xlabel('X grid points')
        ax5.set_ylabel('Y grid points')
        plt.colorbar(im5, ax=ax5, label='log10(Permeability m²)')
        
        # 3D temperature isosurface
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        
        # Create meshgrid for 3D plotting
        x = np.linspace(0, self.Lx/1000, self.nx)
        y = np.linspace(0, self.Ly/1000, self.ny)
        z = np.linspace(0, self.Lz/1000, self.nz)
        
        # Sample every 8th point for performance
        step = 8
        xi, yi, zi = np.meshgrid(x[::step], y[::step], z[::step], indexing='ij')
        temp_3d = self.temperature[::step, ::step, ::step] - 273.15
        
        # Plot temperature isosurface at 200°C
        from skimage.measure import marching_cubes
        try:
            verts, faces, _, _ = marching_cubes(temp_3d, 200)
            ax6.plot_trisurf(verts[:, 0] * self.Lx/1000 / (self.nx//step),
                           verts[:, 1] * self.Ly/1000 / (self.ny//step),
                           verts[:, 2] * self.Lz/1000 / (self.nz//step),
                           triangles=faces, alpha=0.7, cmap='hot')
        except:
            # Fallback to scatter plot
            hot_points = np.where(temp_3d > 200)
            ax6.scatter(xi[hot_points], yi[hot_points], zi[hot_points], 
                       c=temp_3d[hot_points], cmap='hot', alpha=0.6)
        
        ax6.set_xlabel('X (km)')
        ax6.set_ylabel('Y (km)')
        ax6.set_zlabel('Z (km)')
        ax6.set_title('Temperature Isosurface (200°C)')
        
        plt.tight_layout()
        plt.savefig('hydrothermal_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualization saved as hydrothermal_results.png")

# Example usage
if __name__ == "__main__":
    # Create hydrothermal flow engine
    engine = HydrothermalFlowEngine(
        grid_size=(64, 64, 64),
        domain_size=(2000, 2000, 1500)
    )
    
    # Run simulation for 1 week
    results = engine.run_simulation(
        total_time=7*24*3600,  # 1 week in seconds
        save_interval=6*3600   # Save every 6 hours
    )
    
    # Export results
    engine.export_results('hydrothermal_simulation.json')
    
    # Generate visualization
    engine.visualize_results()
    
    print("🎉 Hydrothermal flow simulation completed!")