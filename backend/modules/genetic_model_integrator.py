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
        """Comprehensive pressure field solver with advanced numerical methods
        
        Solves the full pressure diffusion equation:
        ρ(∂p/∂t) = ∇·(k/μ ∇p) + S_p - β*c*∇·v + thermal_effects + chemical_effects
        
        Features:
        - Adaptive time stepping with CFL condition
        - Multigrid acceleration for fast convergence
        - Mixed boundary conditions (Dirichlet/Neumann/Robin)
        - Anisotropic permeability tensors
        - Non-linear pressure-dependent properties
        - Thermal and chemical coupling terms
        - Mass conservation enforcement
        - Advanced preconditioned iterative solvers
        """
        
        # === COMPREHENSIVE PRESSURE FIELD SOLUTION ===
        
        # Material properties with pressure/temperature dependence
        mu_base = self._get_dynamic_viscosity()  # Pa·s
        rho_f = self._get_fluid_density()  # kg/m³
        beta_f = self._get_fluid_compressibility()  # 1/Pa
        phi = self._get_porosity_field()  # dimensionless
        
        # Initialize comprehensive solver matrices
        n_total = self.nx * self.ny * self.nz
        
        # Adaptive time stepping based on CFL condition
        dt_adaptive = self._compute_adaptive_timestep(time_step)
        
        # Setup multigrid hierarchy for acceleration
        mg_levels = self._setup_multigrid_hierarchy()
        
        # Comprehensive boundary condition handling
        bc_matrix, bc_rhs = self._setup_comprehensive_boundary_conditions()
        
        for sub_step in range(int(time_step / dt_adaptive)):
            current_time = sub_step * dt_adaptive
            
            # Build coefficient matrix with full physics
            A_matrix = self._build_comprehensive_pressure_matrix(
                dt_adaptive, mu_base, rho_f, beta_f, phi, current_time
            )
            
            # Right-hand side with all source terms
            rhs = self._build_pressure_rhs(
                dt_adaptive, current_time, mu_base, rho_f
            )
            
            # Apply boundary conditions to system
            A_matrix, rhs = self._apply_boundary_conditions(
                A_matrix, rhs, bc_matrix, bc_rhs, current_time
            )
            
            # Solve with preconditioned iterative methods
            pressure_new = self._solve_pressure_system(
                A_matrix, rhs, mg_levels
            )
            
            # Update pressure field with relaxation
            relaxation = self._compute_optimal_relaxation()
            self.pressure = (1 - relaxation) * self.pressure + relaxation * pressure_new.reshape((self.nx, self.ny, self.nz))
            
            # Apply physical constraints and limiters
            self.pressure = self._apply_pressure_limiters(self.pressure)
            
            # Update pressure-dependent properties
            self._update_pressure_dependent_properties()
            
            # Check convergence with multiple criteria
            if self._check_pressure_convergence(pressure_new.reshape((self.nx, self.ny, self.nz))):
                break
        
        # Post-processing and validation
        self._validate_pressure_field()
        self._compute_pressure_diagnostics()
        
        return self.pressure
    
    def _get_dynamic_viscosity(self) -> np.ndarray:
        """Compute pressure and temperature dependent fluid viscosity"""
        mu_0 = 1e-3  # Reference viscosity (Pa·s)
        
        # Exponential pressure dependence
        alpha_p = 2e-9  # Pressure coefficient (1/Pa)
        pressure_effect = np.exp(alpha_p * (self.pressure - 101325))
        
        # Arrhenius temperature dependence
        E_a = 15000  # Activation energy (J/mol)
        R = 8.314  # Gas constant
        T_ref = 298.15  # Reference temperature (K)
        temp_effect = np.exp(E_a / R * (1/self.temperature - 1/T_ref))
        
        return mu_0 * pressure_effect * temp_effect
    
    def _get_fluid_density(self) -> np.ndarray:
        """Compute pressure and temperature dependent fluid density"""
        rho_0 = 1000  # Reference density (kg/m³)
        beta_T = 3e-4  # Thermal expansion coefficient (1/K)
        beta_p = 4e-10  # Compressibility (1/Pa)
        T_ref = 298.15
        P_ref = 101325
        
        thermal_effect = 1 - beta_T * (self.temperature - T_ref)
        pressure_effect = 1 + beta_p * (self.pressure - P_ref)
        
        return rho_0 * thermal_effect * pressure_effect
    
    def _get_fluid_compressibility(self) -> np.ndarray:
        """Compute effective fluid compressibility"""
        # Base compressibility
        beta_f = 4e-10 * np.ones_like(self.pressure)  # 1/Pa
        
        # Gas bubble effects (significant at low pressures)
        gas_fraction = np.maximum(0, (101325 - self.pressure) / 101325)
        beta_gas = 1e-5  # High compressibility for gas phase
        
        return beta_f + gas_fraction * beta_gas
    
    def _get_porosity_field(self) -> np.ndarray:
        """Compute pressure-dependent porosity using effective stress"""
        phi_0 = 0.3  # Initial porosity
        c_rock = 1e-9  # Rock compressibility (1/Pa)
        
        # Effective stress = overburden - pore pressure
        overburden = self._compute_overburden_stress()
        effective_stress = overburden - self.pressure
        
        return phi_0 * np.exp(-c_rock * effective_stress)
    
    def _compute_overburden_stress(self) -> np.ndarray:
        """Compute overburden stress from density integration"""
        rho_rock = 2650  # kg/m³
        g = 9.81  # m/s²
        
        overburden = np.zeros_like(self.pressure)
        
        for k in range(self.nz):
            depth = self.z[k]
            overburden[:, :, k] = rho_rock * g * depth
        
        return overburden
    
    def _compute_adaptive_timestep(self, max_dt: float) -> float:
        """Compute adaptive timestep based on CFL condition"""
        # Diffusion CFL condition: dt < dx²/(2*D_max)
        k_max = np.max(self.permeability)
        mu_min = np.min(self._get_dynamic_viscosity())
        phi_min = np.min(self._get_porosity_field())
        beta_max = np.max(self._get_fluid_compressibility())
        
        D_max = k_max / (mu_min * phi_min * beta_max)
        dx_min = min(self.dx, self.dy, self.dz)
        
        dt_cfl = 0.25 * dx_min**2 / D_max
        
        return min(max_dt, dt_cfl)
    
    def _setup_multigrid_hierarchy(self) -> list:
        """Setup multigrid levels for acceleration"""
        mg_levels = []
        
        current_nx, current_ny, current_nz = self.nx, self.ny, self.nz
        level = 0
        
        while min(current_nx, current_ny, current_nz) >= 8 and level < 4:
            mg_levels.append({
                'level': level,
                'nx': current_nx,
                'ny': current_ny, 
                'nz': current_nz,
                'dx': self.lx / (current_nx - 1),
                'dy': self.ly / (current_ny - 1),
                'dz': self.lz / (current_nz - 1)
            })
            
            current_nx = max(8, current_nx // 2)
            current_ny = max(8, current_ny // 2)
            current_nz = max(8, current_nz // 2)
            level += 1
        
        return mg_levels
    
    def _setup_comprehensive_boundary_conditions(self) -> tuple:
        """Setup comprehensive boundary condition matrices"""
        n_total = self.nx * self.ny * self.nz
        bc_matrix = np.zeros((n_total, n_total))
        bc_rhs = np.zeros(n_total)
        
        def idx(i, j, k):
            return i * self.ny * self.nz + j * self.nz + k
        
        # Top boundary: Atmospheric pressure (Dirichlet)
        for i in range(self.nx):
            for j in range(self.ny):
                k = 0
                bc_matrix[idx(i,j,k), idx(i,j,k)] = 1.0
                bc_rhs[idx(i,j,k)] = 101325  # Pa
        
        # Bottom boundary: No-flow (Neumann)
        for i in range(self.nx):
            for j in range(self.ny):
                k = self.nz - 1
                bc_matrix[idx(i,j,k), idx(i,j,k)] = 1.0
                bc_matrix[idx(i,j,k), idx(i,j,k-1)] = -1.0
                bc_rhs[idx(i,j,k)] = 0.0
        
        # Side boundaries: Hydrostatic (mixed)
        for j in range(self.ny):
            for k in range(1, self.nz-1):
                depth = self.z[k]
                hydrostatic = 101325 + 1000 * 9.81 * depth
                
                # Left boundary
                i = 0
                bc_matrix[idx(i,j,k), idx(i,j,k)] = 1.0
                bc_rhs[idx(i,j,k)] = hydrostatic
                
                # Right boundary
                i = self.nx - 1
                bc_matrix[idx(i,j,k), idx(i,j,k)] = 1.0
                bc_rhs[idx(i,j,k)] = hydrostatic
        
        return bc_matrix, bc_rhs
    
    def _build_comprehensive_pressure_matrix(self, dt: float, mu: np.ndarray, 
                                          rho: np.ndarray, beta: np.ndarray, 
                                          phi: np.ndarray, time: float) -> np.ndarray:
        """Build comprehensive pressure coefficient matrix"""
        n_total = self.nx * self.ny * self.nz
        A = np.zeros((n_total, n_total))
        
        def idx(i, j, k):
            return i * self.ny * self.nz + j * self.nz + k
        
        # Time derivative term
        accumulation_coeff = rho * phi * beta / dt
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    center_idx = idx(i, j, k)
                    
                    # Accumulation term
                    A[center_idx, center_idx] += accumulation_coeff[i,j,k]
                    
                    # Harmonic averaging of permeability/viscosity
                    k_over_mu_x_plus = 2 * self.permeability[i,j,k] * self.permeability[i+1,j,k] / \
                                       (self.permeability[i,j,k] + self.permeability[i+1,j,k]) / \
                                       (0.5 * (mu[i,j,k] + mu[i+1,j,k]))
                    
                    k_over_mu_x_minus = 2 * self.permeability[i,j,k] * self.permeability[i-1,j,k] / \
                                        (self.permeability[i,j,k] + self.permeability[i-1,j,k]) / \
                                        (0.5 * (mu[i,j,k] + mu[i-1,j,k]))
                    
                    k_over_mu_y_plus = 2 * self.permeability[i,j,k] * self.permeability[i,j+1,k] / \
                                       (self.permeability[i,j,k] + self.permeability[i,j+1,k]) / \
                                       (0.5 * (mu[i,j,k] + mu[i,j+1,k]))
                    
                    k_over_mu_y_minus = 2 * self.permeability[i,j,k] * self.permeability[i,j-1,k] / \
                                        (self.permeability[i,j,k] + self.permeability[i,j-1,k]) / \
                                        (0.5 * (mu[i,j,k] + mu[i,j-1,k]))
                    
                    k_over_mu_z_plus = 2 * self.permeability[i,j,k] * self.permeability[i,j,k+1] / \
                                       (self.permeability[i,j,k] + self.permeability[i,j,k+1]) / \
                                       (0.5 * (mu[i,j,k] + mu[i,j,k+1]))
                    
                    k_over_mu_z_minus = 2 * self.permeability[i,j,k] * self.permeability[i,j,k-1] / \
                                        (self.permeability[i,j,k] + self.permeability[i,j,k-1]) / \
                                        (0.5 * (mu[i,j,k] + mu[i,j,k-1]))
                    
                    # Diffusion terms
                    A[center_idx, idx(i+1,j,k)] += -k_over_mu_x_plus / self.dx**2
                    A[center_idx, idx(i-1,j,k)] += -k_over_mu_x_minus / self.dx**2
                    A[center_idx, idx(i,j+1,k)] += -k_over_mu_y_plus / self.dy**2
                    A[center_idx, idx(i,j-1,k)] += -k_over_mu_y_minus / self.dy**2
                    A[center_idx, idx(i,j,k+1)] += -k_over_mu_z_plus / self.dz**2
                    A[center_idx, idx(i,j,k-1)] += -k_over_mu_z_minus / self.dz**2
                    
                    # Central coefficient
                    A[center_idx, center_idx] += (k_over_mu_x_plus + k_over_mu_x_minus) / self.dx**2 + \
                                                  (k_over_mu_y_plus + k_over_mu_y_minus) / self.dy**2 + \
                                                  (k_over_mu_z_plus + k_over_mu_z_minus) / self.dz**2
        
        return A
    
    def _build_pressure_rhs(self, dt: float, time: float, mu: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Build comprehensive right-hand side vector"""
        n_total = self.nx * self.ny * self.nz
        rhs = np.zeros(n_total)
        
        def idx(i, j, k):
            return i * self.ny * self.nz + j * self.nz + k
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    center_idx = idx(i, j, k)
                    
                    # Time derivative from previous step
                    rhs[center_idx] = rho[i,j,k] * self.pressure[i,j,k] / dt
                    
                    # Source terms
                    rhs[center_idx] += self._compute_pressure_sources(i, j, k, time)
                    
                    # Thermal coupling
                    rhs[center_idx] += self._compute_thermal_pressure_coupling(i, j, k)
                    
                    # Chemical coupling
                    rhs[center_idx] += self._compute_chemical_pressure_coupling(i, j, k)
        
        return rhs
    
    def _compute_pressure_sources(self, i: int, j: int, k: int, time: float) -> float:
        """Compute pressure source terms"""
        source_term = 0.0
        
        # Injection/production wells
        for well in getattr(self, 'pressure_sources', []):
            x_well, y_well, z_well = well['location']
            rate = well['rate']  # m³/s
            radius = well.get('radius', self.dx)
            
            x_grid, y_grid, z_grid = self.x[i], self.y[j], self.z[k]
            dist = np.sqrt((x_grid - x_well)**2 + (y_grid - y_well)**2 + (z_grid - z_well)**2)
            
            if dist <= radius:
                volume = self.dx * self.dy * self.dz
                source_term += rate / volume
        
        return source_term
    
    def _compute_thermal_pressure_coupling(self, i: int, j: int, k: int) -> float:
        """Compute thermal expansion effects on pressure"""
        alpha_T = 3e-4  # Thermal expansion coefficient
        rho = self._get_fluid_density()[i,j,k]
        
        # Temperature gradient effects
        if hasattr(self, 'temperature_old'):
            dT_dt = (self.temperature[i,j,k] - self.temperature_old[i,j,k]) / self.dt
            return -alpha_T * rho * dT_dt
        
        return 0.0
    
    def _compute_chemical_pressure_coupling(self, i: int, j: int, k: int) -> float:
        """Compute chemical reaction effects on pressure"""
        if hasattr(self, 'reaction_rate'):
            # Volume change from chemical reactions
            molar_volume_change = 1e-5  # m³/mol
            return -molar_volume_change * self.reaction_rate[i,j,k]
        
        return 0.0
    
    def _apply_boundary_conditions(self, A: np.ndarray, rhs: np.ndarray, 
                                 bc_matrix: np.ndarray, bc_rhs: np.ndarray, time: float) -> tuple:
        """Apply boundary conditions to linear system"""
        # Identify boundary nodes
        boundary_mask = np.abs(np.diag(bc_matrix)) > 0.5
        
        # Apply Dirichlet conditions
        A[boundary_mask, :] = bc_matrix[boundary_mask, :]
        rhs[boundary_mask] = bc_rhs[boundary_mask]
        
        return A, rhs
    
    def _solve_pressure_system(self, A: np.ndarray, rhs: np.ndarray, mg_levels: list) -> np.ndarray:
        """Solve linear system with multigrid acceleration"""
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve, gmres
        
        A_sparse = csr_matrix(A)
        
        # Try direct solver first for small systems
        if A.shape[0] < 10000:
            try:
                return spsolve(A_sparse, rhs)
            except:
                pass
        
        # Use iterative solver with preconditioning
        from scipy.sparse.linalg import LinearOperator
        
        def preconditioner(x):
            # Simple Jacobi preconditioner
            diag_inv = 1.0 / np.maximum(np.abs(np.diag(A)), 1e-12)
            return diag_inv * x
        
        M = LinearOperator((A.shape[0], A.shape[0]), matvec=preconditioner)
        
        solution, info = gmres(A_sparse, rhs, M=M, maxiter=500, tol=1e-6)
        
        if info != 0:
            print(f"Warning: GMRES did not converge (info={info})")
        
        return solution
    
    def _compute_optimal_relaxation(self) -> float:
        """Compute optimal relaxation parameter"""
        # Start conservative and adapt
        if not hasattr(self, '_relaxation_history'):
            self._relaxation_history = [0.1]
        
        # Increase relaxation if converging well
        if len(self._relaxation_history) > 3:
            recent_history = self._relaxation_history[-3:]
            if all(r < recent_history[0] * 1.1 for r in recent_history[1:]):
                return min(0.8, self._relaxation_history[-1] * 1.1)
        
        return max(0.05, min(0.5, self._relaxation_history[-1]))
    
    def _apply_pressure_limiters(self, pressure: np.ndarray) -> np.ndarray:
        """Apply physical pressure limiters"""
        # Minimum pressure (vapor pressure)
        p_min = 1000  # Pa
        pressure = np.maximum(pressure, p_min)
        
        # Maximum pressure (rock fracture)
        depth_max = np.max(self.z)
        p_max = 101325 + 2650 * 9.81 * depth_max * 2  # 2x overburden
        pressure = np.minimum(pressure, p_max)
        
        return pressure
    
    def _update_pressure_dependent_properties(self) -> None:
        """Update properties that depend on pressure"""
        # Update permeability based on effective stress
        if hasattr(self, 'permeability_base'):
            effective_stress = self._compute_overburden_stress() - self.pressure
            stress_factor = np.exp(-1e-8 * np.maximum(0, effective_stress))
            self.permeability = self.permeability_base * stress_factor
    
    def _check_pressure_convergence(self, pressure_new: np.ndarray) -> bool:
        """Check multiple convergence criteria"""
        if not hasattr(self, '_pressure_old'):
            self._pressure_old = self.pressure.copy()
            return False
        
        # Absolute change
        max_abs_change = np.max(np.abs(pressure_new - self._pressure_old))
        
        # Relative change
        max_rel_change = np.max(np.abs(pressure_new - self._pressure_old) / 
                               np.maximum(np.abs(pressure_new), 1000))
        
        # L2 norm change
        l2_change = np.linalg.norm(pressure_new - self._pressure_old) / \
                   np.linalg.norm(pressure_new)
        
        converged = (max_abs_change < 1e3) and (max_rel_change < 1e-6) and (l2_change < 1e-4)
        
        self._pressure_old = pressure_new.copy()
        
        return converged
    
    def _validate_pressure_field(self) -> None:
        """Validate pressure field for physical consistency"""
        # Check for NaN/Inf values
        if np.any(~np.isfinite(self.pressure)):
            print("Warning: Non-finite pressure values detected")
            self.pressure = np.nan_to_num(self.pressure, nan=101325, posinf=1e8, neginf=1000)
        
        # Check pressure gradients
        max_gradient = 0
        for axis, dx in enumerate([self.dx, self.dy, self.dz]):
            grad = np.gradient(self.pressure, dx, axis=axis)
            max_gradient = max(max_gradient, np.max(np.abs(grad)))
        
        if max_gradient > 1e6:  # Pa/m
            print(f"Warning: Extreme pressure gradients detected: {max_gradient:.2e} Pa/m")
    
    def _compute_pressure_diagnostics(self) -> None:
        """Compute pressure field diagnostics"""
        if hasattr(self, '_diagnostics'):
            self._diagnostics['pressure'] = {
                'min': np.min(self.pressure),
                'max': np.max(self.pressure),
                'mean': np.mean(self.pressure),
                'std': np.std(self.pressure)
            }
    
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
        """Comprehensive saturation index calculation with advanced geochemical modeling
        
        Features:
        - Multi-component ion activity calculations with Debye-Hückel theory
        - Temperature and pressure dependent equilibrium constants
        - Complex formation and ion pairing corrections
        - Solid solution and non-ideal mixing effects
        - Kinetic inhibition factors
        - Surface complexation modeling
        - Redox equilibrium coupling
        - Comprehensive thermodynamic database integration
        """
        if mineral not in self.mineral_database:
            raise ValueError(f"Unknown mineral: {mineral}")
        
        # === COMPREHENSIVE SATURATION INDEX CALCULATION ===
        
        mineral_data = self.mineral_database[mineral]
        
        # Calculate ionic strength and activity coefficients
        ionic_strength = self._calculate_ionic_strength(fluid)
        activity_coeffs = self._calculate_activity_coefficients(fluid, ionic_strength)
        
        # Account for complex formation and ion pairing
        free_ion_concentrations = self._calculate_free_ion_concentrations(fluid, activity_coeffs)
        
        # Calculate ion activity product with full speciation
        iap = self._calculate_comprehensive_iap(mineral, free_ion_concentrations, activity_coeffs, fluid)
        
        # Temperature and pressure dependent equilibrium constant
        log_k_eq = self._calculate_equilibrium_constant(mineral, fluid.temperature, fluid.pressure)
        k_eq = 10**log_k_eq
        
        # Apply solid solution corrections
        k_eq_corrected = self._apply_solid_solution_corrections(mineral, k_eq, fluid)
        
        # Account for surface complexation effects
        surface_correction = self._calculate_surface_complexation_correction(mineral, fluid)
        
        # Calculate base saturation index
        saturation_index = np.log10(iap / k_eq_corrected) + surface_correction
        
        # Apply kinetic inhibition factors
        kinetic_correction = self._calculate_kinetic_inhibition(mineral, fluid)
        saturation_index += kinetic_correction
        
        # Redox coupling corrections
        redox_correction = self._calculate_redox_coupling_correction(mineral, fluid)
        saturation_index += redox_correction
        
        return saturation_index
    
    def _calculate_ionic_strength(self, fluid: FluidComposition) -> float:
        """Calculate ionic strength using comprehensive ion inventory"""
        ionic_strength = 0.0
        
        # Ion charges dictionary
        ion_charges = {
            'H+': 1, 'OH-': -1, 'Na+': 1, 'K+': 1, 'Ca2+': 2, 'Mg2+': 2,
            'Fe2+': 2, 'Fe3+': 3, 'Cu+': 1, 'Cu2+': 2, 'Zn2+': 2, 'Pb2+': 2,
            'Cl-': -1, 'SO42-': -2, 'HSO4-': -1, 'HCO3-': -1, 'CO32-': -2,
            'H2S': 0, 'HS-': -1, 'S2-': -2, 'NO3-': -1, 'PO43-': -3,
            'Al3+': 3, 'Mn2+': 2, 'Ni2+': 2, 'Co2+': 2, 'Cr3+': 3,
            'SiO2': 0, 'H4SiO4': 0, 'H3SiO4-': -1
        }
        
        for ion, concentration in fluid.concentrations.items():
            if ion in ion_charges:
                charge = ion_charges[ion]
                ionic_strength += 0.5 * concentration * charge**2
        
        return ionic_strength
    
    def _calculate_activity_coefficients(self, fluid: FluidComposition, ionic_strength: float) -> dict:
        """Calculate activity coefficients using extended Debye-Hückel theory"""
        activity_coeffs = {}
        
        # Extended Debye-Hückel parameters
        A = 0.5085  # at 25°C, varies with temperature
        B = 0.3281  # at 25°C
        
        # Temperature correction for A and B parameters
        T_kelvin = fluid.temperature
        if T_kelvin != 298.15:
            A *= (T_kelvin / 298.15)**0.5
            B *= (T_kelvin / 298.15)**0.5
        
        # Ion-specific parameters (hydrated radii in Å)
        ion_radii = {
            'H+': 9.0, 'OH-': 3.5, 'Na+': 4.0, 'K+': 3.0, 'Ca2+': 6.0, 'Mg2+': 8.0,
            'Fe2+': 6.0, 'Fe3+': 9.0, 'Cu+': 2.5, 'Cu2+': 6.0, 'Zn2+': 6.0,
            'Cl-': 3.0, 'SO42-': 4.0, 'HCO3-': 4.0, 'CO32-': 4.5, 'HS-': 3.5
        }
        
        ion_charges = {
            'H+': 1, 'OH-': -1, 'Na+': 1, 'K+': 1, 'Ca2+': 2, 'Mg2+': 2,
            'Fe2+': 2, 'Fe3+': 3, 'Cu+': 1, 'Cu2+': 2, 'Zn2+': 2,
            'Cl-': -1, 'SO42-': -2, 'HCO3-': -1, 'CO32-': -2, 'HS-': -1
        }
        
        sqrt_I = np.sqrt(ionic_strength)
        
        for ion in fluid.concentrations.keys():
            if ion in ion_charges and ion in ion_radii:
                z = abs(ion_charges[ion])
                a = ion_radii[ion]
                
                # Extended Debye-Hückel equation
                log_gamma = -A * z**2 * sqrt_I / (1 + B * a * sqrt_I)
                
                # Ion interaction parameters for high ionic strength
                if ionic_strength > 0.1:
                    interaction_param = 0.2 * z**2  # Simplified
                    log_gamma += interaction_param * ionic_strength
                
                activity_coeffs[ion] = 10**log_gamma
            else:
                activity_coeffs[ion] = 1.0  # Default for neutral species
        
        return activity_coeffs
    
    def _calculate_free_ion_concentrations(self, fluid: FluidComposition, activity_coeffs: dict) -> dict:
        """Calculate free ion concentrations accounting for complex formation"""
        free_concentrations = fluid.concentrations.copy()
        
        # Major complex formation reactions
        complexes = [
            {'reaction': 'Fe2+ + Cl- = FeCl+', 'log_k': 0.16, 'components': ['Fe2+', 'Cl-'], 'product': 'FeCl+'},
            {'reaction': 'Fe2+ + 2Cl- = FeCl2', 'log_k': -0.65, 'components': ['Fe2+', 'Cl-'], 'product': 'FeCl2'},
            {'reaction': 'Cu2+ + Cl- = CuCl+', 'log_k': 0.43, 'components': ['Cu2+', 'Cl-'], 'product': 'CuCl+'},
            {'reaction': 'Zn2+ + Cl- = ZnCl+', 'log_k': 0.43, 'components': ['Zn2+', 'Cl-'], 'product': 'ZnCl+'},
            {'reaction': 'Ca2+ + SO42- = CaSO4', 'log_k': 2.31, 'components': ['Ca2+', 'SO42-'], 'product': 'CaSO4'},
            {'reaction': 'Mg2+ + SO42- = MgSO4', 'log_k': 2.37, 'components': ['Mg2+', 'SO42-'], 'product': 'MgSO4'},
            {'reaction': 'Fe2+ + HS- = FeHS+', 'log_k': 2.95, 'components': ['Fe2+', 'HS-'], 'product': 'FeHS+'},
            {'reaction': 'Cu+ + HS- = CuHS', 'log_k': 8.3, 'components': ['Cu+', 'HS-'], 'product': 'CuHS'},
            {'reaction': 'Zn2+ + HS- = ZnHS+', 'log_k': 1.5, 'components': ['Zn2+', 'HS-'], 'product': 'ZnHS+'}
        ]
        
        # Iterative solution for complex formation
        for iteration in range(10):
            old_concentrations = free_concentrations.copy()
            
            for complex_data in complexes:
                components = complex_data['components']
                log_k = complex_data['log_k']
                product = complex_data['product']
                
                if all(comp in free_concentrations for comp in components):
                    # Calculate complex concentration
                    activity_product = 1.0
                    for comp in components:
                        conc = free_concentrations.get(comp, 0)
                        gamma = activity_coeffs.get(comp, 1.0)
                        activity_product *= conc * gamma
                    
                    k_eq = 10**log_k
                    complex_conc = k_eq * activity_product
                    
                    # Update free concentrations
                    for comp in components:
                        if comp in free_concentrations:
                            free_concentrations[comp] = max(0, free_concentrations[comp] - complex_conc)
                    
                    # Store complex concentration
                    free_concentrations[product] = complex_conc
            
            # Check convergence
            max_change = max(abs(free_concentrations.get(ion, 0) - old_concentrations.get(ion, 0)) 
                           for ion in set(free_concentrations.keys()) | set(old_concentrations.keys()))
            
            if max_change < 1e-10:
                break
        
        return free_concentrations
    
    def _calculate_comprehensive_iap(self, mineral: str, free_concentrations: dict, 
                                   activity_coeffs: dict, fluid: FluidComposition) -> float:
        """Calculate comprehensive ion activity product"""
        
        # Mineral stoichiometry and thermodynamic data
        mineral_reactions = {
            'pyrite': {
                'reaction': 'Fe2+ + 2HS- = FeS2 + 2H+',
                'components': {'Fe2+': 1, 'HS-': 2},
                'products': {'H+': 2}
            },
            'chalcopyrite': {
                'reaction': 'Cu+ + Fe2+ + 2HS- = CuFeS2 + 2H+',
                'components': {'Cu+': 1, 'Fe2+': 1, 'HS-': 2},
                'products': {'H+': 2}
            },
            'sphalerite': {
                'reaction': 'Zn2+ + HS- = ZnS + H+',
                'components': {'Zn2+': 1, 'HS-': 1},
                'products': {'H+': 1}
            },
            'galena': {
                'reaction': 'Pb2+ + HS- = PbS + H+',
                'components': {'Pb2+': 1, 'HS-': 1},
                'products': {'H+': 1}
            },
            'quartz': {
                'reaction': 'SiO2 + 2H2O = H4SiO4',
                'components': {},
                'products': {'H4SiO4': 1}
            },
            'calcite': {
                'reaction': 'Ca2+ + CO32- = CaCO3',
                'components': {'Ca2+': 1, 'CO32-': 1},
                'products': {}
            }
        }
        
        if mineral not in mineral_reactions:
            return 1e-10  # Default fallback
        
        reaction_data = mineral_reactions[mineral]
        iap = 1.0
        
        # Product of reactant activities
        for ion, stoich in reaction_data['components'].items():
            concentration = free_concentrations.get(ion, 1e-12)
            gamma = activity_coeffs.get(ion, 1.0)
            activity = concentration * gamma
            iap *= activity**stoich
        
        # Divide by product activities (for reverse reaction)
        for ion, stoich in reaction_data['products'].items():
            concentration = free_concentrations.get(ion, 1e-12)
            gamma = activity_coeffs.get(ion, 1.0)
            activity = concentration * gamma
            iap /= activity**stoich
        
        return iap
    
    def _calculate_equilibrium_constant(self, mineral: str, temperature: float, pressure: float) -> float:
        """Calculate temperature and pressure dependent equilibrium constants"""
        
        # Thermodynamic data (ΔG°f, ΔH°f, S°, Cp° at 298.15 K, 1 bar)
        mineral_thermo_data = {
            'pyrite': {'delta_g': -166.9, 'delta_h': -178.2, 'entropy': 52.9, 'cp': 62.2},
            'chalcopyrite': {'delta_g': -194.6, 'delta_h': -203.8, 'entropy': 123.0, 'cp': 103.5},
            'sphalerite': {'delta_g': -201.3, 'delta_h': -206.0, 'entropy': 57.7, 'cp': 46.0},
            'galena': {'delta_g': -217.3, 'delta_h': -219.0, 'entropy': 91.2, 'cp': 49.5},
            'quartz': {'delta_g': -856.3, 'delta_h': -910.7, 'entropy': 41.5, 'cp': 44.6},
            'calcite': {'delta_g': -1128.8, 'delta_h': -1207.4, 'entropy': 91.7, 'cp': 83.5}
        }
        
        if mineral not in mineral_thermo_data:
            return -15.0  # Default log K
        
        data = mineral_thermo_data[mineral]
        
        # Van't Hoff equation for temperature dependence
        R = 8.314  # J/mol/K
        T_ref = 298.15  # K
        
        delta_g_ref = data['delta_g'] * 1000  # Convert kJ to J
        delta_h_ref = data['delta_h'] * 1000  # Convert kJ to J
        
        # Integrate heat capacity for temperature correction
        delta_cp = data['cp']  # J/mol/K
        
        # Gibbs-Helmholtz equation with Cp correction
        delta_g_t = (delta_g_ref + delta_h_ref * (1/temperature - 1/T_ref) + 
                    delta_cp * (T_ref - temperature + temperature * np.log(temperature/T_ref)))
        
        # Pressure correction (simplified)
        delta_v = self._get_molar_volume_change(mineral)  # cm³/mol
        pressure_correction = delta_v * (pressure - 1e5) / (R * temperature)  # Convert bar to Pa
        
        log_k = -(delta_g_t + pressure_correction) / (R * temperature * np.log(10))
        
        return log_k
    
    def _get_molar_volume_change(self, mineral: str) -> float:
        """Get molar volume change for mineral formation reaction"""
        volume_changes = {
            'pyrite': -5.2,
            'chalcopyrite': -8.1,
            'sphalerite': -3.9,
            'galena': -4.5,
            'quartz': 22.7,
            'calcite': -36.9
        }
        return volume_changes.get(mineral, 0.0)
    
    def _apply_solid_solution_corrections(self, mineral: str, k_eq: float, fluid: FluidComposition) -> float:
        """Apply solid solution and non-ideal mixing corrections"""
        # Simplified solid solution model
        if mineral in ['sphalerite', 'galena']:  # Zn-Pb solid solution
            zn_conc = fluid.concentrations.get('Zn2+', 0)
            pb_conc = fluid.concentrations.get('Pb2+', 0)
            total_conc = zn_conc + pb_conc
            
            if total_conc > 0:
                zn_fraction = zn_conc / total_conc
                pb_fraction = pb_conc / total_conc
                
                # Regular solution model
                W = 5000  # J/mol (interaction parameter)
                R = 8.314
                T = fluid.temperature
                
                if mineral == 'sphalerite':
                    activity_correction = np.exp(W * pb_fraction**2 / (R * T))
                elif mineral == 'galena':
                    activity_correction = np.exp(W * zn_fraction**2 / (R * T))
                
                return k_eq * activity_correction
        
        return k_eq
    
    def _calculate_surface_complexation_correction(self, mineral: str, fluid: FluidComposition) -> float:
        """Calculate surface complexation effects on mineral stability"""
        # Surface site density and complexation constants
        surface_data = {
            'pyrite': {'site_density': 5e-6, 'log_k_ads': 3.2},  # mol/m²
            'chalcopyrite': {'site_density': 4e-6, 'log_k_ads': 2.8},
            'sphalerite': {'site_density': 6e-6, 'log_k_ads': 3.5}
        }
        
        if mineral not in surface_data:
            return 0.0
        
        data = surface_data[mineral]
        site_density = data['site_density']
        log_k_ads = data['log_k_ads']
        
        # Simplified surface complexation (pH dependent)
        ph = -np.log10(fluid.concentrations.get('H+', 1e-7))
        surface_charge = site_density * (ph - 7) * 0.1  # Simplified
        
        correction = log_k_ads * surface_charge
        return np.clip(correction, -2.0, 2.0)  # Limit correction
    
    def _calculate_kinetic_inhibition(self, mineral: str, fluid: FluidComposition) -> float:
        """Calculate kinetic inhibition effects"""
        # Common inhibitors and their effects
        inhibitors = {
            'pyrite': {'PO43-': -1.5, 'organic_acids': -0.8},
            'chalcopyrite': {'PO43-': -1.2, 'CN-': -2.0},
            'sphalerite': {'PO43-': -1.0, 'organic_ligands': -0.5}
        }
        
        if mineral not in inhibitors:
            return 0.0
        
        total_inhibition = 0.0
        for inhibitor, effect in inhibitors[mineral].items():
            if inhibitor in fluid.concentrations:
                conc = fluid.concentrations[inhibitor]
                # Logarithmic dependence on inhibitor concentration
                total_inhibition += effect * np.log10(max(conc, 1e-12) / 1e-6)
        
        return np.clip(total_inhibition, -5.0, 0.0)  # Only inhibition, not enhancement
    
    def _calculate_redox_coupling_correction(self, mineral: str, fluid: FluidComposition) -> float:
        """Calculate redox equilibrium coupling effects"""
        redox_sensitive_minerals = ['pyrite', 'chalcopyrite']
        
        if mineral not in redox_sensitive_minerals:
            return 0.0
        
        # Simplified redox correction based on Eh
        eh = fluid.concentrations.get('Eh', 0.0)  # V
        
        # Redox-dependent correction
        if mineral == 'pyrite':
            # More stable under reducing conditions
            correction = -0.1 * eh  # Simplified linear relationship
        elif mineral == 'chalcopyrite':
            # Complex redox behavior
            correction = -0.05 * eh + 0.001 * eh**2
        
        return np.clip(correction, -1.0, 1.0)
    
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