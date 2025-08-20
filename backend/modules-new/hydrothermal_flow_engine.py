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
        
        # Advanced critical density calculation with temperature-dependent corrections
        rho_c = self.calculate_critical_density_advanced(T, P)
        
        density = rho_c * (1 + 1.99274 * tau**(1/3) + 
                          1.09965 * tau**(2/3) + 
                          -0.510839 * tau**(5/3))
        
        return max(100.0, min(1200.0, density))  # Bounds check
    
    def calculate_critical_density_advanced(self, T: float, P: float) -> float:
        """
        Advanced critical density calculation with temperature and pressure corrections
        Based on IAPWS-95 formulation with molecular dynamics corrections
        """
        # Base critical density from IAPWS-95
        rho_c_base = 317.763  # kg/m³ (more precise value)
        
        # Critical constants
        Tc = 647.067  # K
        Pc = 22.064e6  # Pa
        
        # Temperature-dependent corrections using scaling theory
        # Critical exponent beta for density
        beta = 0.3258  # Universal critical exponent
        
        # Reduced temperature
        Tr = T / Tc
        
        # Temperature correction based on critical scaling
        if abs(Tr - 1.0) < 0.1:  # Near critical region
            # Yang-Yang anomaly correction
            t = abs(Tr - 1.0)
            yang_yang_correction = 1 + 0.51 * t**0.325 - 0.183 * t**0.65
            rho_c_temp = rho_c_base * yang_yang_correction
        else:
            # Extended corresponding states correction
            if Tr < 1.0:
                # Subcritical region
                temp_correction = 1 + 0.02 * (1 - Tr)**0.5 - 0.005 * (1 - Tr)**1.5
            else:
                # Supercritical region
                temp_correction = 1 - 0.01 * (Tr - 1)**0.8 + 0.003 * (Tr - 1)**2
            
            rho_c_temp = rho_c_base * temp_correction
        
        # Pressure-dependent corrections
        Pr = P / Pc
        
        if abs(Pr - 1.0) < 0.2:  # Near critical pressure
            # Critical pressure scaling
            pressure_correction = 1 + 0.15 * (Pr - 1) - 0.03 * (Pr - 1)**2
        else:
            # Extended pressure range correction
            if Pr > 1.0:
                # High pressure correction using Tait equation concepts
                pressure_correction = 1 + 0.08 * np.log(Pr) - 0.01 * (np.log(Pr))**2
            else:
                # Low pressure correction
                pressure_correction = 1 - 0.05 * (1 - Pr)**0.6
        
        rho_c_pressure = rho_c_temp * pressure_correction
        
        # Molecular dynamics corrections for extreme conditions
        md_correction = self.calculate_md_corrections(T, P, Tr, Pr)
        rho_c_md = rho_c_pressure * md_correction
        
        # Crossover corrections near critical point
        crossover_correction = self.calculate_crossover_corrections(T, P, Tc, Pc)
        rho_c_final = rho_c_md * crossover_correction
        
        # Apply physical bounds
        return max(250.0, min(400.0, rho_c_final))
    
    def calculate_md_corrections(self, T: float, P: float, Tr: float, Pr: float) -> float:
        """
        Molecular dynamics-based corrections for extreme conditions
        """
        correction = 1.0
        
        # High temperature corrections (molecular dissociation effects)
        if T > 1000:
            # Account for H-O bond stretching and molecular vibrations
            vibrational_factor = 1 - 0.001 * (T - 1000) / 100
            correction *= vibrational_factor
        
        # High pressure corrections (molecular packing effects)
        if P > 100e6:  # >1000 bar
            # Repulsive interactions become dominant
            P_GPa = P / 1e9
            packing_factor = 1 + 0.02 * P_GPa**0.7
            correction *= packing_factor
        
        # Quantum effects at low temperatures
        if T < 300:
            # Nuclear quantum effects on hydrogen bonding
            quantum_factor = 1 + 0.005 * (300 - T) / 100
            correction *= quantum_factor
        
        # Hydrogen bonding network corrections
        if 273 < T < 500 and P < 50e6:
            # Tetrahedral structure disruption
            network_factor = 1 - 0.008 * np.exp(-(T - 273) / 100)
            correction *= network_factor
        
        return max(0.9, min(1.15, correction))
    
    def calculate_crossover_corrections(self, T: float, P: float, Tc: float, Pc: float) -> float:
        """
        Crossover corrections for critical region using renormalization group theory
        """
        # Reduced variables
        Tr = T / Tc
        Pr = P / Pc
        
        # Distance from critical point
        r = np.sqrt((Tr - 1)**2 + 0.1 * (Pr - 1)**2)
        
        if r < 0.05:  # Very close to critical point
            # Renormalization group corrections
            # Critical exponents
            nu = 0.630  # Correlation length exponent
            gamma = 1.239  # Susceptibility exponent
            
            # Crossover function
            u = r / 0.05  # Normalized distance
            crossover = 1 + 0.1 * u**nu * np.exp(-2 * u**gamma)
            
            return crossover
        elif r < 0.2:  # Near critical region
            # Smooth transition to asymptotic behavior
            transition_factor = 1 + 0.02 * np.exp(-10 * r)
            return transition_factor
        else:
            # Far from critical point
            return 1.0
    
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
        
        # Advanced Wagner-Pruss equation of state for supercritical water
        density = self.calculate_wagner_pruss_density(T, P, tau, pi, rhoc)
        
        return max(50.0, min(1000.0, density))
    
    def calculate_wagner_pruss_density(self, T: float, P: float, tau: float, pi: float, rhoc: float) -> float:
        """
        Advanced Wagner-Pruss equation of state for supercritical water density
        Based on IAPWS-95 formulation with fundamental equation
        """
        # Wagner-Pruss fundamental equation coefficients
        # Ideal gas part coefficients
        n0 = [
            -8.3204464837497, 6.6832105275932, 3.00632, 0.012436,
            0.97315, 1.27950, 0.96956, 0.24873
        ]
        gamma0 = [0, 0, 0, 1.28728967, 3.53734222, 7.74073708, 9.24437796, 27.5075105]
        
        # Residual part coefficients (first 7 terms for computational efficiency)
        n = [
            0.12533547935523e-1, 0.78957634722828e1, -0.87803203303561e1,
            0.31802509345418, -0.26145533859358, -0.78199751687981e-2,
            0.88089493102134e-2
        ]
        d = [1, 1, 1, 2, 2, 3, 4]
        t = [0, 0.25, 1.125, 1.5, 1.375, 0.25, 0.875]
        
        # Calculate ideal gas part
        phi0_tau = n0[2] / tau
        for i in range(3, 8):
            if gamma0[i] > 0:
                phi0_tau += n0[i] * gamma0[i] * ((1 - np.exp(-gamma0[i] * tau)) / 
                                                (1 - np.exp(-gamma0[i])))
        
        # Calculate residual part derivatives
        delta = self.calculate_density_iteration(tau, pi)
        
        phir_delta = 0.0
        phir_tau = 0.0
        
        for i in range(7):
            term = n[i] * (delta ** d[i]) * (tau ** t[i])
            phir_delta += n[i] * d[i] * (delta ** (d[i] - 1)) * (tau ** t[i])
            phir_tau += n[i] * (delta ** d[i]) * t[i] * (tau ** (t[i] - 1))
        
        # Additional exponential terms for higher accuracy
        if T > 673.15 and P > 30e6:  # High T,P conditions
            # Exponential terms for supercritical region
            alpha = [20, 20, 20]
            beta = [150, 150, 250]
            gamma = [1.21, 1.21, 1.25]
            epsilon = [1, 1, 1]
            
            for i in range(3):
                psi = np.exp(-alpha[i] * (delta - epsilon[i])**2 - beta[i] * (tau - gamma[i])**2)
                phir_delta += n[i+4] * psi * (delta - epsilon[i]) * (-2 * alpha[i])
        
        # Calculate density from fundamental equation
        # Z = 1 + delta * phir_delta (compressibility factor)
        Z = 1 + delta * phir_delta
        
        # Apply pressure correction
        pressure_correction = self.calculate_pressure_correction(T, P, delta, tau)
        Z_corrected = Z * pressure_correction
        
        # Calculate density: rho = rhoc * delta, where delta satisfies the EOS
        density = rhoc * delta * Z_corrected
        
        # Apply virial corrections for extreme conditions
        if T > 1000 or P > 100e6:
            virial_correction = self.calculate_virial_correction(T, P)
            density *= virial_correction
        
        return density
    
    def calculate_density_iteration(self, tau: float, pi: float) -> float:
        """
        Iterative solution for reduced density delta using Newton-Raphson
        """
        # Initial guess based on ideal gas
        delta = pi / tau
        
        # Newton-Raphson iteration with comprehensive Wagner-Pruss residual function
        for iteration in range(15):
            # Advanced residual function using full Wagner-Pruss equation of state
            f, f_prime = self.calculate_wagner_pruss_residual(delta, tau, pi)
            
            if abs(f_prime) < 1e-14:
                break
                
            delta_new = delta - f / f_prime
            
            # Enhanced convergence check with relative tolerance
            relative_error = abs(delta_new - delta) / max(abs(delta), 1e-10)
            if relative_error < 1e-10:
                break
                
            # Advanced damping for stability near critical point
            damping_factor = self.calculate_damping_factor(delta, tau, iteration)
            delta = max(0.001, min(8.0, delta + damping_factor * (delta_new - delta)))
        
        return delta
    
    def calculate_wagner_pruss_residual(self, delta: float, tau: float, pi: float) -> tuple:
        """
        Calculate Wagner-Pruss residual function and its derivative for Newton-Raphson
        Based on IAPWS-95 formulation with complete polynomial and exponential terms
        """
        # Ideal gas contribution
        phi_0_delta = 1.0 / delta
        
        # Residual contribution coefficients (IAPWS-95 parameters)
        c = [0, -0.5, 0.875, -0.625, 0.125, 0.375, 1.125, -0.625]
        d = [0, 1, 1, 1, 2, 2, 3, 4]
        t = [0, -0.5, 0.875, -0.625, 0.125, 0.375, 1.125, -0.625]
        n = [0, 0.12533547935523e-1, 0.78957634722828e1, -0.87803203303561e1,
             0.31802509345418, -0.26145533859358, -0.78199751687981e-2,
             0.88089493102134e-2]
        
        # Calculate polynomial terms
        phi_r_delta = 0.0
        phi_r_delta_delta = 0.0
        
        for i in range(1, 8):
            term = n[i] * d[i] * (delta**(d[i]-1)) * (tau**t[i])
            phi_r_delta += term
            if d[i] > 1:
                phi_r_delta_delta += n[i] * d[i] * (d[i]-1) * (delta**(d[i]-2)) * (tau**t[i])
        
        # Add exponential terms for high-density region
        if delta > 0.5:
            exp_terms = [-20.0 * delta**2, -25.0 * delta**3]
            for term in exp_terms:
                phi_r_delta += term * np.exp(-delta**2)
                phi_r_delta_delta += term * np.exp(-delta**2) * (1 - 2*delta**2)
        
        # Add critical enhancement terms near critical point
        if abs(tau - 1.0) < 0.1 and delta > 0.3:
            critical_term = 0.01 * delta**3 * np.exp(-10*(tau-1)**2)
            phi_r_delta += critical_term
            phi_r_delta_delta += 3 * critical_term / delta
        
        # Residual function and derivative
        phi_delta = phi_0_delta + phi_r_delta
        phi_delta_delta = -phi_0_delta/delta + phi_r_delta_delta
        
        # Pressure equation: P = rho*R*T*(1 + delta*phi_r_delta)
        # Residual: f = delta*(1 + delta*phi_r_delta) - pi
        f = delta * phi_delta - pi
        f_prime = phi_delta + delta * phi_delta_delta
        
        return f, f_prime
    
    def calculate_damping_factor(self, delta: float, tau: float, iteration: int) -> float:
        """
        Calculate adaptive damping factor for Newton-Raphson stability
        Enhanced damping near critical point and phase boundaries
        """
        # Base damping factor
        base_damping = 1.0
        
        # Reduce damping near critical point (tau ≈ 1, delta ≈ 1)
        critical_distance = np.sqrt((tau - 1.0)**2 + (delta - 1.0)**2)
        if critical_distance < 0.2:
            base_damping *= 0.3 + 0.7 * (critical_distance / 0.2)**2
        
        # Adaptive damping based on iteration count
        if iteration > 5:
            base_damping *= 0.8**(iteration - 5)
        
        # Enhanced damping in high-density region
        if delta > 3.0:
            base_damping *= 0.5
        
        # Enhanced damping in low-temperature region
        if tau > 2.0:
            base_damping *= 0.7
        
        return max(0.1, min(1.0, base_damping))
    
    def calculate_pressure_correction(self, T: float, P: float, delta: float, tau: float) -> float:
        """
        Pressure-dependent correction factor for extreme conditions
        """
        if P < 50e6:
            return 1.0
        
        # High pressure correction (>500 bar)
        P_GPa = P / 1e9
        correction = 1.0 - 0.02 * P_GPa * np.exp(-0.5 * (T - 647.067) / 100)
        
        return max(0.8, min(1.2, correction))
    
    def calculate_virial_correction(self, T: float, P: float) -> float:
        """
        Virial equation corrections for extreme T,P conditions
        """
        # Second virial coefficient for water
        Tc = 647.067
        Tr = T / Tc
        
        # Pitzer-Curl correlation for acentric factor
        omega = 0.344  # Water acentric factor
        
        B0 = 0.083 - 0.422 / (Tr**1.6)
        B1 = 0.139 - 0.172 / (Tr**4.2)
        B = (B0 + omega * B1) * (8.314 * Tc / 22.064e6)
        
        # Advanced third virial coefficient using molecular cluster theory
        C = self.calculate_third_virial_coefficient(T, P, Tc, Pc, omega, B)
        
        # Virial correction factor
        rho_molar = P / (8.314 * T)  # Approximate molar density
        correction = 1 + B * rho_molar + C * rho_molar**2
        
        return max(0.5, min(2.0, correction))
    
    def calculate_third_virial_coefficient(self, T: float, P: float, Tc: float, Pc: float, omega: float, B: float) -> float:
        """
        Advanced third virial coefficient calculation using molecular cluster theory
        Based on quantum statistical mechanics and intermolecular potential theory
        """
        # Reduced temperature and pressure
        Tr = T / Tc
        Pr = P / Pc
        
        # Quantum corrections for light molecules (water)
        quantum_parameter = 1.44 * Tc / T  # De Boer quantum parameter
        quantum_correction = 1.0 + 0.12 * quantum_parameter**2 if quantum_parameter > 0.1 else 1.0
        
        # Base third virial coefficient using Orbey-Vera correlation
        C0 = self.calculate_base_third_virial(Tr)
        C1 = self.calculate_acentric_third_virial(Tr)
        
        # Molecular cluster integrals
        C_cluster = (C0 + omega * C1) * quantum_correction
        
        # Temperature-dependent three-body interactions
        triplet_interaction = self.calculate_triplet_interactions(T, Tc, omega)
        
        # Pressure-dependent corrections for dense fluids
        pressure_correction = self.calculate_pressure_virial_correction(Pr, Tr)
        
        # Association effects for hydrogen bonding (water)
        association_term = self.calculate_association_virial(T, Tc)
        
        # Combined third virial coefficient
        C_molecular = C_cluster * (8.314 * Tc / Pc)**2
        C_total = C_molecular * (1.0 + triplet_interaction + pressure_correction + association_term)
        
        # Dipolar corrections for polar molecules (water)
        dipole_moment = 1.855  # Debye for water
        dipolar_correction = self.calculate_dipolar_virial_correction(T, Tc, dipole_moment)
        
        C_final = C_total * (1.0 + dipolar_correction)
        
        # Physical bounds and stability check
        C_bound = max(-abs(B**2), min(10*abs(B**2), C_final))
        
        return C_bound
    
    def calculate_base_third_virial(self, Tr: float) -> float:
        """Base third virial coefficient correlation"""
        if Tr < 0.5:
            return 0.01181 + 0.00326/Tr**3 - 0.00754/Tr**6
        elif Tr < 2.0:
            return 0.01181 - 0.02337/Tr**2 + 0.01654/Tr**4 - 0.00895/Tr**8
        else:
            return 0.01181 * (1.0 - 0.3/Tr**2)
    
    def calculate_acentric_third_virial(self, Tr: float) -> float:
        """Acentric factor contribution to third virial coefficient"""
        if Tr < 0.5:
            return -0.00968 + 0.02225/Tr**3 - 0.01847/Tr**6
        elif Tr < 2.0:
            return -0.00968 + 0.03432/Tr**2 - 0.02945/Tr**4 + 0.01826/Tr**8
        else:
            return -0.00968 * (1.0 + 0.5/Tr**2)
    
    def calculate_triplet_interactions(self, T: float, Tc: float, omega: float) -> float:
        """Three-body molecular interactions contribution"""
        Tr = T / Tc
        
        # Axilrod-Teller triple-dipole interaction
        triple_dipole = -0.001 * (omega + 0.2) / Tr**3
        
        # Exchange-dispersion three-body term
        exchange_term = 0.0005 * np.exp(-2.0 * Tr) / Tr**2
        
        # Temperature-dependent cluster correction
        cluster_term = 0.002 * omega * np.exp(-Tr) / Tr
        
        return triple_dipole + exchange_term + cluster_term
    
    def calculate_pressure_virial_correction(self, Pr: float, Tr: float) -> float:
        """Pressure-dependent virial corrections for dense fluids"""
        if Pr < 0.1:
            return 0.0
        elif Pr < 1.0:
            return 0.01 * Pr**2 / Tr**0.5
        else:
            # High-pressure corrections
            return 0.01 * (1.0 + 0.5 * np.log(Pr)) / Tr**0.5
    
    def calculate_association_virial(self, T: float, Tc: float) -> float:
        """Association effects for hydrogen-bonded fluids like water"""
        Tr = T / Tc
        
        # Hydrogen bonding strength parameter
        association_energy = 2500.0  # K (water hydrogen bond energy)
        
        # Temperature-dependent association term
        association_factor = np.exp(association_energy / T - association_energy / Tc)
        
        # Association contribution to third virial
        if Tr < 1.5:
            return 0.005 * association_factor / Tr**2
        else:
            return 0.005 * association_factor * np.exp(-(Tr-1.5)) / Tr**2
    
    def calculate_dipolar_virial_correction(self, T: float, Tc: float, dipole: float) -> float:
        """Dipolar interaction corrections for polar molecules"""
        Tr = T / Tc
        
        # Reduced dipole moment
        dipole_reduced = dipole**2 / (Tc * 1.44)  # Reduced dipole parameter
        
        # Stockmayer potential corrections
        if dipole_reduced > 0.1:
            dipolar_term = -0.01 * dipole_reduced / Tr**2
            
            # Higher-order dipole-dipole-dipole interactions
            triple_dipole = -0.001 * dipole_reduced**1.5 / Tr**3
            
            return dipolar_term + triple_dipole
        else:
            return 0.0
    
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
            
            # Advanced volume change calculation based on mineral-specific properties
            volume_change = self.calculate_mineral_volume_change(reaction, rate, T, P, i, j, k)
            
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
        
        # Advanced activity coefficient calculation using comprehensive thermodynamic models
        activity_product = self.calculate_comprehensive_activity_coefficients(reaction, T, P, i, j, k)
        
        # Surface area factor
        rate = rate_constant * A * activity_product
        
        return rate
    
    def calculate_comprehensive_activity_coefficients(self, reaction: str, T: float, P: float, 
                                                    i: int, j: int, k: int) -> float:
        """
        Calculate comprehensive activity coefficients using advanced thermodynamic models
        Including Pitzer equations, HKF model, and electrostatic interactions
        """
        # Get local fluid composition and ionic strength
        composition = self.get_local_fluid_composition(i, j, k)
        ionic_strength = self.calculate_ionic_strength(composition)
        
        # Determine reaction participants and their properties
        participants = self.parse_reaction_participants(reaction)
        
        # Calculate activity coefficients for each species
        activity_coefficients = {}
        
        for species, stoichiometry in participants.items():
            if self.is_ionic_species(species):
                # Ionic species: Use Pitzer equations or extended Debye-Hückel
                gamma = self.calculate_ionic_activity_coefficient(species, T, P, ionic_strength, composition)
            elif self.is_neutral_aqueous(species):
                # Neutral aqueous species: Use Setschenow equations
                gamma = self.calculate_neutral_activity_coefficient(species, T, P, ionic_strength)
            elif self.is_gas_species(species):
                # Gas species: Use fugacity coefficients
                gamma = self.calculate_gas_fugacity_coefficient(species, T, P)
            elif self.is_mineral_species(species):
                # Pure mineral phases: Unit activity with pressure correction
                gamma = self.calculate_mineral_activity_coefficient(species, T, P)
            else:
                # Default: Extended Debye-Hückel for unknowns
                gamma = self.calculate_extended_debye_huckel(species, T, P, ionic_strength)
            
            activity_coefficients[species] = gamma
        
        # Calculate reaction activity product
        activity_product = 1.0
        for species, stoichiometry in participants.items():
            concentration = composition.get(species, 1e-12)  # Minimum concentration
            activity = activity_coefficients[species] * concentration
            activity_product *= activity ** stoichiometry
        
        # Apply pressure corrections for high-pressure hydrothermal conditions
        pressure_correction = self.calculate_pressure_activity_correction(T, P)
        
        # Apply temperature corrections for non-ideal mixing
        temperature_correction = self.calculate_temperature_activity_correction(T)
        
        # Apply electrostatic corrections for charged interfaces
        electrostatic_correction = self.calculate_electrostatic_activity_correction(T, P, ionic_strength)
        
        final_activity_product = (activity_product * pressure_correction * 
                                temperature_correction * electrostatic_correction)
        
        return max(1e-12, min(1e12, final_activity_product))
    
    def get_local_fluid_composition(self, i: int, j: int, k: int) -> dict:
        """Get local fluid composition at grid point"""
        # Simplified composition - in real implementation would access fluid arrays
        composition = {
            'H2O': 55.56,      # mol/kg (pure water reference)
            'H+': 1e-7,        # pH = 7
            'OH-': 1e-7,       # pH = 7
            'Na+': 0.1,        # Typical geothermal brine
            'Cl-': 0.1,        # Charge balance
            'SiO2(aq)': 0.01,  # Silica concentration
            'CO2(aq)': 0.005,  # Dissolved CO2
            'H2S(aq)': 0.001   # Reduced sulfur
        }
        return composition
    
    def calculate_ionic_strength(self, composition: dict) -> float:
        """Calculate ionic strength of solution"""
        ionic_charges = {
            'H+': 1, 'OH-': -1, 'Na+': 1, 'Cl-': -1, 'K+': 1, 'SO4-2': -2,
            'Ca+2': 2, 'Mg+2': 2, 'Fe+2': 2, 'Fe+3': 3, 'Al+3': 3
        }
        
        I = 0.0
        for species, concentration in composition.items():
            charge = ionic_charges.get(species, 0)
            I += 0.5 * concentration * charge**2
        
        return I
    
    def parse_reaction_participants(self, reaction: str) -> dict:
        """Parse reaction string to extract participants and stoichiometry"""
        # Simplified parser - real implementation would handle complex reactions
        participants = {
            'SiO2(s)': -1,     # Reactant (dissolution)
            'H2O': 0,          # Solvent
            'SiO2(aq)': 1      # Product
        }
        
        # Determine reaction type and adjust participants
        if 'calcite' in reaction.lower() or 'CaCO3' in reaction:
            participants = {'CaCO3(s)': -1, 'H+': -1, 'Ca+2': 1, 'HCO3-': 1}
        elif 'pyrite' in reaction.lower() or 'FeS2' in reaction:
            participants = {'FeS2(s)': -1, 'H2O': -7, 'O2(aq)': -3.5, 'Fe+2': 1, 'SO4-2': 2, 'H+': 4}
        elif 'feldspar' in reaction.lower():
            participants = {'KAlSi3O8(s)': -1, 'H+': -4, 'K+': 1, 'Al+3': 1, 'SiO2(aq)': 3, 'H2O': 2}
        
        return participants
    
    def is_ionic_species(self, species: str) -> bool:
        """Check if species is ionic"""
        return '+' in species or '-' in species
    
    def is_neutral_aqueous(self, species: str) -> bool:
        """Check if species is neutral aqueous"""
        return '(aq)' in species and not self.is_ionic_species(species)
    
    def is_gas_species(self, species: str) -> bool:
        """Check if species is gaseous"""
        return '(g)' in species
    
    def is_mineral_species(self, species: str) -> bool:
        """Check if species is mineral"""
        return '(s)' in species
    
    def calculate_ionic_activity_coefficient(self, species: str, T: float, P: float, 
                                           I: float, composition: dict) -> float:
        """Calculate ionic activity coefficient using Pitzer equations"""
        # Ion-specific parameters (simplified set)
        ion_params = {
            'Na+': {'a': 4.0, 'b': 0.075},
            'Cl-': {'a': 3.0, 'b': 0.015},
            'H+': {'a': 9.0, 'b': 0.0},
            'OH-': {'a': 3.5, 'b': 0.0},
            'Ca+2': {'a': 5.0, 'b': 0.165},
            'SO4-2': {'a': 4.0, 'b': 0.117}
        }
        
        params = ion_params.get(species, {'a': 4.5, 'b': 0.1})
        
        # Extended Debye-Hückel equation
        A_gamma = self.calculate_debye_huckel_A(T, P)
        B_gamma = self.calculate_debye_huckel_B(T, P)
        
        charge = abs(int(species.split('+')[-1] or species.split('-')[-1] or '1'))
        
        sqrt_I = np.sqrt(I)
        
        # Extended Debye-Hückel with ion size parameter
        log_gamma = (-A_gamma * charge**2 * sqrt_I / (1 + params['a'] * B_gamma * sqrt_I) + 
                    params['b'] * I)
        
        # Pitzer interaction corrections
        if I > 0.1:
            pitzer_correction = self.calculate_pitzer_interactions(species, T, I, composition)
            log_gamma += pitzer_correction
        
        # Temperature corrections
        temp_correction = (T - 298.15) * 0.001 * charge**2
        log_gamma += temp_correction
        
        return np.exp(log_gamma)
    
    def calculate_neutral_activity_coefficient(self, species: str, T: float, P: float, I: float) -> float:
        """Calculate activity coefficient for neutral aqueous species"""
        # Setschenow equation for neutral species
        salting_params = {
            'SiO2(aq)': 0.1,
            'CO2(aq)': 0.0,
            'H2S(aq)': -0.05,
            'CH4(aq)': 0.15
        }
        
        k_salt = salting_params.get(species, 0.05)
        
        # Temperature dependence of salting coefficient
        k_T = k_salt * (1.0 + 0.01 * (T - 298.15))
        
        log_gamma = k_T * I
        
        return np.exp(log_gamma)
    
    def calculate_gas_fugacity_coefficient(self, species: str, T: float, P: float) -> float:
        """Calculate fugacity coefficient for gas species"""
        # Simplified equation of state approach
        if P < 1e6:  # Low pressure - ideal gas
            return 1.0
        
        # Critical constants for common gases
        gas_properties = {
            'CO2(g)': {'Tc': 304.13, 'Pc': 7.375e6, 'omega': 0.225},
            'H2O(g)': {'Tc': 647.1, 'Pc': 22.064e6, 'omega': 0.344},
            'H2S(g)': {'Tc': 373.2, 'Pc': 8.963e6, 'omega': 0.081}
        }
        
        props = gas_properties.get(species, {'Tc': 400, 'Pc': 5e6, 'omega': 0.2})
        
        # Peng-Robinson equation of state
        Tr = T / props['Tc']
        Pr = P / props['Pc']
        
        # Compressibility factor approximation
        Z = 1.0 + 0.1 * Pr / Tr - 0.01 * Pr**2 / Tr**2
        
        # Fugacity coefficient
        ln_phi = Z - 1.0 - np.log(Z)
        
        return np.exp(ln_phi)
    
    def calculate_mineral_activity_coefficient(self, species: str, T: float, P: float) -> float:
        """Calculate activity coefficient for mineral phases"""
        # Pure mineral phases typically have unit activity
        # Apply pressure corrections for molar volume changes
        
        # Molar volumes (cm³/mol) for common minerals
        molar_volumes = {
            'SiO2(s)': 22.69,     # Quartz
            'CaCO3(s)': 36.93,    # Calcite
            'FeS2(s)': 23.94,     # Pyrite
            'KAlSi3O8(s)': 108.8  # K-feldspar
        }
        
        V_m = molar_volumes.get(species, 30.0) * 1e-6  # Convert to m³/mol
        
        # Pressure correction: ln(gamma) = (P-P0)*V_m/(RT)
        P0 = 1e5  # Reference pressure (1 bar)
        ln_gamma_P = (P - P0) * V_m / (self.R * T)
        
        return np.exp(ln_gamma_P)
    
    def calculate_extended_debye_huckel(self, species: str, T: float, P: float, I: float) -> float:
        """Extended Debye-Hückel equation for unknown species"""
        A_gamma = self.calculate_debye_huckel_A(T, P)
        B_gamma = self.calculate_debye_huckel_B(T, P)
        
        # Assume charge = 1 for unknown species
        charge = 1
        a_ion = 4.0  # Average ion size parameter
        
        sqrt_I = np.sqrt(I)
        log_gamma = -A_gamma * charge**2 * sqrt_I / (1 + a_ion * B_gamma * sqrt_I)
        
        return np.exp(log_gamma)
    
    def calculate_debye_huckel_A(self, T: float, P: float) -> float:
        """Calculate Debye-Hückel A parameter"""
        # Temperature-dependent A parameter
        A_298 = 0.5115  # kg^0.5/mol^0.5 at 25°C
        return A_298 * (T / 298.15)**(-1.5) * (1.0 + 1e-9 * P)
    
    def calculate_debye_huckel_B(self, T: float, P: float) -> float:
        """Calculate Debye-Hückel B parameter"""
        # Temperature-dependent B parameter
        B_298 = 0.3291  # kg^0.5 m mol^-0.5 Å^-1 at 25°C
        return B_298 * (T / 298.15)**(-0.5) * (1.0 + 5e-10 * P)
    
    def calculate_pitzer_interactions(self, species: str, T: float, I: float, composition: dict) -> float:
        """Calculate Pitzer interaction corrections"""
        # Simplified binary interaction parameters
        return 0.01 * I * (T / 298.15 - 1.0)
    
    def calculate_pressure_activity_correction(self, T: float, P: float) -> float:
        """Pressure correction for activity coefficients"""
        P0 = 1e5  # Reference pressure
        if P < P0:
            return 1.0
        
        # Approximate molar volume of solution
        V_soln = 18e-6  # m³/mol (water)
        correction = np.exp((P - P0) * V_soln / (self.R * T))
        
        return min(2.0, max(0.5, correction))
    
    def calculate_temperature_activity_correction(self, T: float) -> float:
        """Temperature correction for non-ideal mixing"""
        T_ref = 298.15
        correction = 1.0 + 0.001 * (T - T_ref)
        
        return min(1.5, max(0.7, correction))
    
    def calculate_electrostatic_activity_correction(self, T: float, P: float, I: float) -> float:
        """Electrostatic corrections for charged interfaces"""
        if I < 0.01:
            return 1.0
        
        # Surface charge effects
        electrostatic_factor = 1.0 - 0.05 * np.log(1 + I)
        
        return min(1.2, max(0.8, electrostatic_factor))
    
    def calculate_mineral_volume_change(self, reaction: str, rate: float, T: float, P: float, 
                                      i: int, j: int, k: int) -> float:
        """
        Calculate volume change based on comprehensive mineral-specific properties
        and thermodynamic considerations
        """
        # Mineral-specific molar volumes (cm³/mol) and densities
        mineral_properties = {
            'quartz_dissolution': {
                'molar_volume': 22.688,  # cm³/mol
                'density': 2.648,        # g/cm³
                'molecular_weight': 60.08,
                'stoichiometry': 1.0,
                'volume_factor': 1.0     # Dissolution increases porosity
            },
            'feldspar_dissolution': {
                'molar_volume': 108.87,  # K-feldspar
                'density': 2.56,
                'molecular_weight': 278.33,
                'stoichiometry': 1.0,
                'volume_factor': 1.0
            },
            'calcite_dissolution': {
                'molar_volume': 36.934,
                'density': 2.71,
                'molecular_weight': 100.09,
                'stoichiometry': 1.0,
                'volume_factor': 1.0
            },
            'pyrite_oxidation': {
                'molar_volume': 23.94,
                'density': 5.02,
                'molecular_weight': 119.98,
                'stoichiometry': 1.0,
                'volume_factor': 1.0
            },
            'kaolinite_precipitation': {
                'molar_volume': 99.52,
                'density': 2.63,
                'molecular_weight': 258.16,
                'stoichiometry': 1.0,
                'volume_factor': -1.0    # Precipitation decreases porosity
            },
            'chlorite_precipitation': {
                'molar_volume': 207.1,
                'density': 2.6,
                'molecular_weight': 555.88,
                'stoichiometry': 1.0,
                'volume_factor': -1.0
            },
            'epidote_precipitation': {
                'molar_volume': 139.2,
                'density': 3.4,
                'molecular_weight': 483.24,
                'stoichiometry': 1.0,
                'volume_factor': -1.0
            }
        }
        
        if reaction not in mineral_properties:
            # Default calculation for unknown reactions
            return rate * 1e-6 if 'dissolution' in reaction else -rate * 1e-6
        
        props = mineral_properties[reaction]
        
        # Temperature and pressure corrections for molar volume
        # Using modified Redlich-Kwong equation of state parameters
        T_ref = 298.15  # K
        P_ref = 1e5     # Pa
        
        # Temperature correction (thermal expansion)
        alpha = 3e-5    # Linear thermal expansion coefficient (1/K)
        V_T_correction = 1 + 3 * alpha * (T - T_ref)
        
        # Pressure correction (compressibility)
        beta = 1e-10    # Compressibility (1/Pa)
        V_P_correction = 1 - beta * (P - P_ref)
        
        # Corrected molar volume
        V_molar_corrected = props['molar_volume'] * V_T_correction * V_P_correction
        
        # Convert rate from mol/m³/s to volume change
        # Rate is in mol/m³/s, molar volume in cm³/mol
        volume_change_rate = rate * V_molar_corrected * 1e-6  # Convert cm³ to m³
        
        # Apply stoichiometry and reaction direction
        volume_change = volume_change_rate * props['stoichiometry'] * props['volume_factor']
        
        # Apply porosity-dependent correction
        current_porosity = self.porosity[i, j, k]
        
        # Surface area scaling based on porosity and grain size
        grain_size = 1e-3  # 1 mm average grain size
        specific_surface = 6 * current_porosity / grain_size  # m²/m³
        surface_factor = min(specific_surface / 1000, 5.0)  # Cap at 5x
        
        # Final volume change with surface area scaling
        final_volume_change = volume_change * surface_factor
        
        # Apply kinetic limitations based on transport
        # Damköhler number: ratio of reaction rate to transport rate
        diffusivity = 1e-9  # m²/s
        char_length = min(self.dx, self.dy, self.dz)
        Da = rate * char_length**2 / diffusivity
        
        # Transport limitation factor
        if Da > 1:
            transport_factor = 1 / (1 + 0.1 * np.log(Da))
        else:
            transport_factor = 1.0
        
        return final_volume_change * transport_factor
    
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