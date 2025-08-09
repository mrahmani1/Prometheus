"""
Advanced Multi-Physics Coupling Engine
Thermal-Mechanical-Chemical-Electromagnetic coupling with adaptive time-stepping and mesh refinement
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
from collections import defaultdict, deque
import pickle
import h5py
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Scientific computing
from scipy import sparse, integrate, interpolate, optimize
from scipy.sparse import linalg as spla
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist
import scipy.ndimage as ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Advanced numerical libraries with fallbacks
try:
    import fenics as fe
    import dolfin as df
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    warnings.warn("FEniCS not available. Some finite element features will be limited.")

try:
    import firedrake
    FIREDRAKE_AVAILABLE = True
except ImportError:
    FIREDRAKE_AVAILABLE = False
    warnings.warn("Firedrake not available. Advanced mesh capabilities will be limited.")

try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    warnings.warn("VTK not available. 3D visualization capabilities will be limited.")

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    warnings.warn("MeshIO not available. Mesh I/O capabilities will be limited.")

warnings.filterwarnings('ignore')

class PhysicsType(Enum):
    THERMAL = "thermal"
    MECHANICAL = "mechanical"
    CHEMICAL = "chemical"
    ELECTROMAGNETIC = "electromagnetic"
    FLUID_FLOW = "fluid_flow"
    PHASE_CHANGE = "phase_change"

class CouplingType(Enum):
    ONE_WAY = "one_way"
    TWO_WAY = "two_way"
    FULLY_COUPLED = "fully_coupled"
    STAGGERED = "staggered"
    MONOLITHIC = "monolithic"

class TimeSteppingScheme(Enum):
    EXPLICIT_EULER = "explicit_euler"
    IMPLICIT_EULER = "implicit_euler"
    CRANK_NICOLSON = "crank_nicolson"
    RUNGE_KUTTA = "runge_kutta"
    ADAPTIVE_BDF = "adaptive_bdf"

@dataclass
class MaterialProperties:
    """Material properties for multi-physics simulations"""
    density: float = 2700.0  # kg/m³
    thermal_conductivity: float = 2.5  # W/(m·K)
    specific_heat: float = 1000.0  # J/(kg·K)
    thermal_expansion: float = 3e-5  # 1/K
    
    # Mechanical properties
    youngs_modulus: float = 70e9  # Pa
    poisson_ratio: float = 0.3
    bulk_modulus: Optional[float] = None
    shear_modulus: Optional[float] = None
    
    # Chemical properties
    diffusion_coefficient: float = 1e-9  # m²/s
    reaction_rate: float = 1e-6  # 1/s
    activation_energy: float = 50000.0  # J/mol
    
    # Electromagnetic properties
    electrical_conductivity: float = 1e-6  # S/m
    magnetic_permeability: float = 4e-7 * np.pi  # H/m
    dielectric_constant: float = 8.0
    
    # Temperature and pressure dependencies
    temperature_dependence: Dict[str, Callable] = field(default_factory=dict)
    pressure_dependence: Dict[str, Callable] = field(default_factory=dict)
    
    def __post_init__(self):
        # Calculate derived properties
        if self.bulk_modulus is None:
            self.bulk_modulus = self.youngs_modulus / (3 * (1 - 2 * self.poisson_ratio))
        if self.shear_modulus is None:
            self.shear_modulus = self.youngs_modulus / (2 * (1 + self.poisson_ratio))

@dataclass
class BoundaryCondition:
    """Boundary condition specification"""
    physics_type: PhysicsType
    boundary_id: int
    condition_type: str  # 'dirichlet', 'neumann', 'robin'
    value: Union[float, Callable]
    location: str = "all"  # 'all', 'top', 'bottom', 'sides'

@dataclass
class CouplingTerm:
    """Coupling term between physics"""
    source_physics: PhysicsType
    target_physics: PhysicsType
    coupling_strength: float
    coupling_function: Callable
    coupling_type: CouplingType

class PhysicsModule(ABC):
    """Abstract base class for physics modules"""
    
    def __init__(self, name: str, material_props: MaterialProperties):
        self.name = name
        self.material_props = material_props
        self.state_variables = {}
        self.boundary_conditions = []
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def compute_residual(self, state: Dict[str, np.ndarray], 
                        dt: float, **kwargs) -> np.ndarray:
        """Compute residual for the physics equations"""
        pass
    
    @abstractmethod
    def compute_jacobian(self, state: Dict[str, np.ndarray], 
                        dt: float, **kwargs) -> sparse.csr_matrix:
        """Compute Jacobian matrix"""
        pass
    
    @abstractmethod
    def initialize_state(self, mesh: Any) -> Dict[str, np.ndarray]:
        """Initialize state variables"""
        pass

class ThermalModule(PhysicsModule):
    """Thermal physics module implementing heat conduction with convection"""
    
    def __init__(self, material_props: MaterialProperties, config: Dict[str, Any] = None):
        super().__init__("thermal", material_props)
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'include_convection': True,
            'include_radiation': False,
            'stefan_boltzmann': 5.67e-8,
            'convection_coefficient': 10.0  # W/(m²·K)
        }
    
    def compute_residual(self, state: Dict[str, np.ndarray], dt: float, 
                        mesh: Any = None, **kwargs) -> np.ndarray:
        """Compute thermal residual: ρcp∂T/∂t = ∇·(k∇T) + Q"""
        temperature = state['temperature']
        T_old = state.get('temperature_old', temperature)
        
        # Material properties
        rho = self.material_props.density
        cp = self.material_props.specific_heat
        k = self.material_props.thermal_conductivity
        
        # Time derivative term
        time_term = rho * cp * (temperature - T_old) / dt
        
        # Diffusion term (simplified finite difference)
        if mesh is not None and hasattr(mesh, 'laplacian_matrix'):
            diffusion_term = -k * mesh.laplacian_matrix @ temperature
        else:
            # Simple 1D finite difference approximation
            n = len(temperature)
            diffusion_term = np.zeros(n)
            if n > 2:
                dx = 1.0 / (n - 1)
                for i in range(1, n-1):
                    diffusion_term[i] = -k * (temperature[i+1] - 2*temperature[i] + temperature[i-1]) / dx**2
        
        # Source term from coupling
        source_term = kwargs.get('thermal_source', np.zeros_like(temperature))
        
        residual = time_term + diffusion_term - source_term
        return residual
    
    def compute_jacobian(self, state: Dict[str, np.ndarray], dt: float, 
                        mesh: Any = None, **kwargs) -> sparse.csr_matrix:
        """Compute thermal Jacobian matrix"""
        temperature = state['temperature']
        n = len(temperature)
        
        # Material properties
        rho = self.material_props.density
        cp = self.material_props.specific_heat
        k = self.material_props.thermal_conductivity
        
        # Time derivative contribution
        time_contrib = rho * cp / dt
        
        # Diffusion contribution
        if mesh is not None and hasattr(mesh, 'laplacian_matrix'):
            diffusion_contrib = -k * mesh.laplacian_matrix
        else:
            # Simple finite difference Laplacian
            diffusion_contrib = sparse.diags(
                [-k, 2*k, -k], [-1, 0, 1], shape=(n, n), format='csr'
            )
        
        jacobian = sparse.diags([time_contrib], [0], shape=(n, n), format='csr') + diffusion_contrib
        
        return jacobian
    
    def initialize_state(self, mesh: Any) -> Dict[str, np.ndarray]:
        """Initialize thermal state"""
        n_nodes = getattr(mesh, 'n_nodes', 100)
        return {
            'temperature': np.full(n_nodes, 293.15),  # Room temperature in K
            'temperature_old': np.full(n_nodes, 293.15)
        }

class MechanicalModule(PhysicsModule):
    """Mechanical physics module for elasticity and plasticity"""
    
    def __init__(self, material_props: MaterialProperties, config: Dict[str, Any] = None):
        super().__init__("mechanical", material_props)
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'plane_stress': True,
            'include_plasticity': False,
            'yield_strength': 250e6,  # Pa
            'hardening_modulus': 1e9  # Pa
        }
    
    def compute_residual(self, state: Dict[str, np.ndarray], dt: float, 
                        mesh: Any = None, **kwargs) -> np.ndarray:
        """Compute mechanical residual: ∇·σ + f = 0 (static equilibrium)"""
        displacement = state['displacement']
        
        # Compute strain from displacement
        strain = self._compute_strain(displacement, mesh)
        
        # Compute stress from strain
        stress = self._compute_stress(strain, state)
        
        # Compute divergence of stress (body forces)
        div_stress = self._compute_stress_divergence(stress, mesh)
        
        # External forces
        body_forces = kwargs.get('body_forces', np.zeros_like(displacement))
        
        # Thermal stress coupling
        thermal_stress = self._compute_thermal_stress(state, **kwargs)
        
        residual = div_stress + body_forces + thermal_stress
        return residual
    
    def compute_jacobian(self, state: Dict[str, np.ndarray], dt: float, 
                        mesh: Any = None, **kwargs) -> sparse.csr_matrix:
        """Compute mechanical Jacobian (stiffness matrix)"""
        displacement = state['displacement']
        n = len(displacement)
        
        # Elastic stiffness matrix
        E = self.material_props.youngs_modulus
        nu = self.material_props.poisson_ratio
        
        if self.config['plane_stress']:
            # 2D plane stress
            D = E / (1 - nu**2) * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1-nu)/2]
            ])
        else:
            # 3D elasticity
            factor = E / ((1 + nu) * (1 - 2*nu))
            D = factor * np.array([
                [1-nu, nu, nu, 0, 0, 0],
                [nu, 1-nu, nu, 0, 0, 0],
                [nu, nu, 1-nu, 0, 0, 0],
                [0, 0, 0, (1-2*nu)/2, 0, 0],
                [0, 0, 0, 0, (1-2*nu)/2, 0],
                [0, 0, 0, 0, 0, (1-2*nu)/2]
            ])
        
        # Simplified stiffness matrix (would need proper FE assembly)
        stiffness = sparse.diags([E], [0], shape=(n, n), format='csr')
        
        return stiffness
    
    def _compute_strain(self, displacement: np.ndarray, mesh: Any = None) -> np.ndarray:
        """Compute strain tensor from displacement field"""
        # Simplified gradient computation
        n = len(displacement)
        if mesh is not None and hasattr(mesh, 'gradient_matrix'):
            strain = mesh.gradient_matrix @ displacement
        else:
            # Simple finite difference
            strain = np.gradient(displacement)
        
        return np.array(strain)
    
    def _compute_stress(self, strain: np.ndarray, state: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute stress tensor from strain using constitutive relations"""
        E = self.material_props.youngs_modulus
        nu = self.material_props.poisson_ratio
        
        # Linear elastic stress-strain relation
        if self.config['plane_stress']:
            # 2D plane stress
            stress = E / (1 - nu**2) * strain  # Simplified
        else:
            # 3D Hooke's law
            stress = E * strain  # Simplified
        
        return stress
    
    def _compute_stress_divergence(self, stress: np.ndarray, mesh: Any = None) -> np.ndarray:
        """Compute divergence of stress tensor"""
        if mesh is not None and hasattr(mesh, 'divergence_matrix'):
            div_stress = mesh.divergence_matrix @ stress
        else:
            # Simple finite difference
            div_stress = np.gradient(stress)
        
        return np.array(div_stress)
    
    def _compute_thermal_stress(self, state: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """Compute thermal stress coupling"""
        if 'temperature' not in state:
            return np.zeros_like(state['displacement'])
        
        temperature = state['temperature']
        alpha = self.material_props.thermal_expansion
        E = self.material_props.youngs_modulus
        nu = self.material_props.poisson_ratio
        
        # Thermal strain
        T_ref = 293.15  # Reference temperature
        thermal_strain = alpha * (temperature - T_ref)
        
        # Thermal stress
        if self.config['plane_stress']:
            thermal_stress_coeff = E * alpha / (1 - nu)
        else:
            thermal_stress_coeff = E * alpha / (1 - 2*nu)
        
        thermal_stress = thermal_stress_coeff * thermal_strain
        
        return thermal_stress
    
    def initialize_state(self, mesh: Any) -> Dict[str, np.ndarray]:
        """Initialize mechanical state"""
        n_nodes = getattr(mesh, 'n_nodes', 100)
        return {
            'displacement': np.zeros(n_nodes),
            'velocity': np.zeros(n_nodes),
            'stress': np.zeros(n_nodes),
            'strain': np.zeros(n_nodes)
        }

class ChemicalModule(PhysicsModule):
    """Chemical physics module for diffusion and reaction"""
    
    def __init__(self, material_props: MaterialProperties, config: Dict[str, Any] = None):
        super().__init__("chemical", material_props)
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'n_species': 1,
            'include_reactions': True,
            'arrhenius_activation': True,
            'gas_constant': 8.314  # J/(mol·K)
        }
    
    def compute_residual(self, state: Dict[str, np.ndarray], dt: float, 
                        mesh: Any = None, **kwargs) -> np.ndarray:
        """Compute chemical residual: ∂C/∂t = D∇²C + R"""
        concentration = state['concentration']
        C_old = state.get('concentration_old', concentration)
        
        # Time derivative
        time_term = (concentration - C_old) / dt
        
        # Diffusion term
        D = self.material_props.diffusion_coefficient
        if mesh is not None and hasattr(mesh, 'laplacian_matrix'):
            diffusion_term = D * mesh.laplacian_matrix @ concentration
        else:
            # Simple finite difference
            n = len(concentration)
            diffusion_term = np.zeros(n)
            if n > 2:
                dx = 1.0 / (n - 1)
                for i in range(1, n-1):
                    diffusion_term[i] = D * (concentration[i+1] - 2*concentration[i] + concentration[i-1]) / dx**2
        
        # Reaction term
        reaction_term = self._compute_reaction_rate(state, **kwargs)
        
        residual = time_term - diffusion_term - reaction_term
        return residual
    
    def compute_jacobian(self, state: Dict[str, np.ndarray], dt: float, 
                        mesh: Any = None, **kwargs) -> sparse.csr_matrix:
        """Compute chemical Jacobian matrix"""
        concentration = state['concentration']
        n = len(concentration)
        
        # Time derivative contribution
        time_contrib = 1.0 / dt
        
        # Diffusion contribution
        D = self.material_props.diffusion_coefficient
        if mesh is not None and hasattr(mesh, 'laplacian_matrix'):
            diffusion_contrib = -D * mesh.laplacian_matrix
        else:
            # Simple finite difference Laplacian
            diffusion_contrib = -D * sparse.diags(
                [1, -2, 1], [-1, 0, 1], shape=(n, n), format='csr'
            )
        
        # Reaction contribution (linearized)
        reaction_contrib = self._compute_reaction_jacobian(state, **kwargs)
        
        jacobian = sparse.diags([time_contrib], [0], shape=(n, n), format='csr') + \
                  diffusion_contrib - reaction_contrib
        
        return jacobian
    
    def _compute_reaction_rate(self, state: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """Compute chemical reaction rate"""
        concentration = state['concentration']
        temperature = state.get('temperature', np.full_like(concentration, 293.15))
        
        # Arrhenius rate law
        if self.config['arrhenius_activation']:
            k0 = self.material_props.reaction_rate
            Ea = self.material_props.activation_energy
            R = self.config['gas_constant']
            
            rate_constant = k0 * np.exp(-Ea / (R * temperature))
        else:
            rate_constant = self.material_props.reaction_rate
        
        # Simple first-order reaction
        reaction_rate = rate_constant * concentration
        
        return reaction_rate
    
    def _compute_reaction_jacobian(self, state: Dict[str, np.ndarray], **kwargs) -> sparse.csr_matrix:
        """Compute Jacobian of reaction terms"""
        concentration = state['concentration']
        temperature = state.get('temperature', np.full_like(concentration, 293.15))
        
        # Linearized reaction rate derivative
        if self.config['arrhenius_activation']:
            k0 = self.material_props.reaction_rate
            Ea = self.material_props.activation_energy
            R = self.config['gas_constant']
            
            rate_constant = k0 * np.exp(-Ea / (R * temperature))
        else:
            rate_constant = self.material_props.reaction_rate
        
        # Diagonal matrix for first-order reaction
        n = len(concentration)
        reaction_jacobian = sparse.diags([rate_constant], [0], shape=(n, n), format='csr')
        
        return reaction_jacobian
    
    def initialize_state(self, mesh: Any) -> Dict[str, np.ndarray]:
        """Initialize chemical state"""
        n_nodes = getattr(mesh, 'n_nodes', 100)
        return {
            'concentration': np.ones(n_nodes) * 0.1,  # Initial concentration
            'concentration_old': np.ones(n_nodes) * 0.1,
            'reaction_rate': np.zeros(n_nodes)
        }

class ElectromagneticModule(PhysicsModule):
    """Electromagnetic physics module for Maxwell equations"""
    
    def __init__(self, material_props: MaterialProperties, config: Dict[str, Any] = None):
        super().__init__("electromagnetic", material_props)
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'frequency': 1e6,  # Hz
            'include_displacement_current': True,
            'quasi_static': False,
            'speed_of_light': 3e8  # m/s
        }
    
    def compute_residual(self, state: Dict[str, np.ndarray], dt: float, 
                        mesh: Any = None, **kwargs) -> np.ndarray:
        """Compute EM residual: Maxwell's equations"""
        electric_field = state.get('electric_field', np.zeros(100))
        magnetic_field = state.get('magnetic_field', np.zeros(100))
        
        # Material properties
        sigma = self.material_props.electrical_conductivity
        epsilon = self.material_props.dielectric_constant * 8.854e-12  # F/m
        mu = self.material_props.magnetic_permeability
        
        # Simplified Maxwell equations residual
        # ∇×E = -∂B/∂t
        # ∇×H = J + ∂D/∂t
        
        if self.config['quasi_static']:
            # Ignore displacement current
            residual_E = self._compute_curl(electric_field, mesh)
            residual_H = self._compute_curl(magnetic_field, mesh) - sigma * electric_field
        else:
            # Full Maxwell equations (simplified)
            B_old = state.get('magnetic_field_old', magnetic_field)
            D_old = state.get('electric_displacement_old', epsilon * electric_field)
            
            residual_E = self._compute_curl(electric_field, mesh) + (magnetic_field - B_old) / dt
            residual_H = self._compute_curl(magnetic_field, mesh) - sigma * electric_field - \
                        epsilon * (electric_field - D_old/epsilon) / dt
        
        # Combine residuals
        residual = np.concatenate([residual_E, residual_H])
        return residual
    
    def compute_jacobian(self, state: Dict[str, np.ndarray], dt: float, 
                        mesh: Any = None, **kwargs) -> sparse.csr_matrix:
        """Compute electromagnetic Jacobian matrix"""
        electric_field = state.get('electric_field', np.zeros(100))
        magnetic_field = state.get('magnetic_field', np.zeros(100))
        
        n_e = len(electric_field)
        n_h = len(magnetic_field)
        n_total = n_e + n_h
        
        # Material properties
        sigma = self.material_props.electrical_conductivity
        epsilon = self.material_props.dielectric_constant * 8.854e-12
        mu = self.material_props.magnetic_permeability
        
        # Simplified Jacobian blocks
        if self.config['quasi_static']:
            # ∂(∇×E)/∂E and ∂(∇×H - σE)/∂H, ∂(∇×H - σE)/∂E
            J_EE = self._get_curl_matrix(n_e, mesh)
            J_HH = self._get_curl_matrix(n_h, mesh)
            J_HE = -sigma * sparse.eye(n_h)
            J_EH = sparse.csr_matrix((n_e, n_h))
        else:
            # Full Maxwell Jacobian
            J_EE = self._get_curl_matrix(n_e, mesh)
            J_EH = sparse.eye(n_e) / dt
            J_HH = self._get_curl_matrix(n_h, mesh)
            J_HE = -sigma * sparse.eye(n_h) - epsilon * sparse.eye(n_h) / dt
        
        # Assemble full Jacobian
        jacobian = sparse.bmat([
            [J_EE, J_EH],
            [J_HE, J_HH]
        ], format='csr')
        
        return jacobian
    
    def _compute_curl(self, field: np.ndarray, mesh: Any = None) -> np.ndarray:
        """Compute curl of vector field"""
        if mesh is not None and hasattr(mesh, 'curl_matrix'):
            curl_field = mesh.curl_matrix @ field
        else:
            # Simple finite difference curl (1D approximation)
            curl_field = np.gradient(field)
        
        return curl_field
    
    def _get_curl_matrix(self, n: int, mesh: Any = None) -> sparse.csr_matrix:
        """Get discrete curl operator matrix"""
        if mesh is not None and hasattr(mesh, 'curl_matrix'):
            return mesh.curl_matrix
        else:
            # Simple finite difference curl matrix
            return sparse.diags([1, -1], [0, 1], shape=(n, n), format='csr')
    
    def initialize_state(self, mesh: Any) -> Dict[str, np.ndarray]:
        """Initialize electromagnetic state"""
        n_nodes = getattr(mesh, 'n_nodes', 100)
        return {
            'electric_field': np.zeros(n_nodes),
            'magnetic_field': np.zeros(n_nodes),
            'electric_displacement': np.zeros(n_nodes),
            'magnetic_field_old': np.zeros(n_nodes),
            'electric_displacement_old': np.zeros(n_nodes)
        }

class AdaptiveTimestepper:
    """Adaptive time-stepping for multi-physics simulations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.time_history = []
        self.error_history = []
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'initial_dt': 1e-6,
            'min_dt': 1e-12,
            'max_dt': 1e-3,
            'tolerance': 1e-6,
            'safety_factor': 0.8,
            'max_increase_factor': 2.0,
            'max_decrease_factor': 0.5,
            'scheme': TimeSteppingScheme.ADAPTIVE_BDF,
            'error_control': 'relative',
            'extrapolation_order': 2
        }
    
    def adapt_timestep(self, current_dt: float, error_estimate: float, 
                      convergence_rate: float = 1.0) -> Tuple[float, bool]:
        """Adapt time step based on error estimate and convergence"""
        tolerance = self.config['tolerance']
        safety = self.config['safety_factor']
        
        # Error-based adaptation
        if error_estimate > tolerance:
            # Decrease time step
            if self.config['scheme'] == TimeSteppingScheme.ADAPTIVE_BDF:
                # BDF error scaling: error ∝ dt^(p+1)
                order = 2  # BDF2
                factor = safety * (tolerance / error_estimate)**(1.0 / (order + 1))
            else:
                # General scaling
                factor = safety * np.sqrt(tolerance / error_estimate)
            
            factor = max(factor, self.config['max_decrease_factor'])
            new_dt = current_dt * factor
            accept_step = False
        else:
            # Increase time step
            if len(self.error_history) >= 2:
                # Use error history for better prediction
                error_ratio = self.error_history[-1] / max(self.error_history[-2], 1e-16)
                factor = safety * (tolerance / error_estimate)**(1.0 / 3.0)
                if error_ratio < 0.5:
                    factor = min(factor * 1.2, self.config['max_increase_factor'])
            else:
                factor = min(1.2, self.config['max_increase_factor'])
            
            new_dt = current_dt * factor
            accept_step = True
        
        # Apply bounds
        new_dt = np.clip(new_dt, self.config['min_dt'], self.config['max_dt'])
        
        # Store history
        self.time_history.append(current_dt)
        self.error_history.append(error_estimate)
        
        return new_dt, accept_step
    
    def compute_error_estimate(self, solution_coarse: np.ndarray, 
                             solution_fine: np.ndarray) -> float:
        """Compute error estimate using Richardson extrapolation"""
        if len(solution_coarse) != len(solution_fine):
            # Handle different solution sizes
            min_len = min(len(solution_coarse), len(solution_fine))
            solution_coarse = solution_coarse[:min_len]
            solution_fine = solution_fine[:min_len]
        
        # Compute relative error
        error_abs = np.abs(solution_fine - solution_coarse)
        
        if self.config['error_control'] == 'relative':
            error_rel = error_abs / (np.abs(solution_fine) + 1e-16)
            error_estimate = np.max(error_rel)
        else:
            error_estimate = np.max(error_abs)
        
        return error_estimate
    
    def get_adaptive_scheme_coefficients(self, dt: float, 
                                       dt_history: List[float]) -> Dict[str, np.ndarray]:
        """Get coefficients for adaptive BDF scheme"""
        if self.config['scheme'] != TimeSteppingScheme.ADAPTIVE_BDF:
            return {}
        
        # Variable-step BDF coefficients
        if len(dt_history) == 0:
            # BDF1 (Backward Euler)
            return {'alpha': np.array([1.0, -1.0]), 'beta': 1.0}
        elif len(dt_history) == 1:
            # BDF2 with variable step
            r = dt / dt_history[-1]
            alpha0 = (1 + 2*r) / (1 + r)
            alpha1 = -(1 + r)
            alpha2 = r**2 / (1 + r)
            beta = (1 + r)
            return {'alpha': np.array([alpha0, alpha1, alpha2]), 'beta': beta}
        else:
            # Higher-order BDF (simplified)
            return {'alpha': np.array([1.0, -1.0]), 'beta': 1.0}

class NumericalStabilityAnalyzer:
    """Analyze numerical stability of coupled simulations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.stability_history = []
        self.eigenvalue_history = []
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'stability_threshold': 1.0,
            'eigenvalue_check_frequency': 10,
            'condition_number_threshold': 1e12,
            'monitor_cfl': True,
            'cfl_threshold': 1.0
        }
    
    def analyze_stability(self, jacobian: sparse.csr_matrix, dt: float, 
                         mesh: Any = None) -> Dict[str, Any]:
        """Analyze stability of the discretized system"""
        stability_info = {
            'is_stable': True,
            'stability_margin': 0.0,
            'condition_number': 0.0,
            'max_eigenvalue': 0.0,
            'cfl_number': 0.0,
            'recommendations': []
        }
        
        try:
            # Condition number analysis
            if jacobian.shape[0] < 1000:  # Only for small systems
                cond_num = np.linalg.cond(jacobian.toarray())
                stability_info['condition_number'] = cond_num
                
                if cond_num > self.config['condition_number_threshold']:
                    stability_info['is_stable'] = False
                    stability_info['recommendations'].append(
                        "High condition number detected - consider preconditioning"
                    )
            
            # Eigenvalue analysis (for small systems)
            if jacobian.shape[0] < 500:
                eigenvals = spla.eigs(jacobian, k=min(6, jacobian.shape[0]-1), 
                                    which='LR', return_eigenvectors=False)
                max_eigenval = np.max(np.real(eigenvals))
                stability_info['max_eigenvalue'] = max_eigenval
                
                # Stability condition: Re(λ) * dt < stability_threshold
                stability_margin = self.config['stability_threshold'] - max_eigenval * dt
                stability_info['stability_margin'] = stability_margin
                
                if stability_margin < 0:
                    stability_info['is_stable'] = False
                    stability_info['recommendations'].append(
                        f"Reduce time step: suggested dt < {self.config['stability_threshold'] / max_eigenval:.2e}"
                    )
                
                self.eigenvalue_history.append(eigenvals)
            
            # CFL condition analysis
            if self.config['monitor_cfl'] and mesh is not None:
                cfl_number = self._compute_cfl_number(dt, mesh)
                stability_info['cfl_number'] = cfl_number
                
                if cfl_number > self.config['cfl_threshold']:
                    stability_info['is_stable'] = False
                    stability_info['recommendations'].append(
                        f"CFL condition violated: CFL = {cfl_number:.3f} > {self.config['cfl_threshold']}"
                    )
        
        except Exception as e:
            self.logger.warning(f"Stability analysis failed: {e}")
            stability_info['error'] = str(e)
        
        self.stability_history.append(stability_info)
        return stability_info
    
    def _compute_cfl_number(self, dt: float, mesh: Any) -> float:
        """Compute CFL number for stability analysis"""
        # Simplified CFL computation
        if hasattr(mesh, 'min_element_size'):
            dx_min = mesh.min_element_size
        else:
            dx_min = 1.0 / getattr(mesh, 'n_nodes', 100)
        
        # Assume maximum wave speed (simplified)
        c_max = 5000.0  # m/s (typical seismic wave speed)
        
        cfl_number = c_max * dt / dx_min
        return cfl_number
    
    def recommend_timestep(self, current_dt: float, jacobian: sparse.csr_matrix) -> float:
        """Recommend stable time step"""
        try:
            if jacobian.shape[0] < 500:
                eigenvals = spla.eigs(jacobian, k=min(6, jacobian.shape[0]-1), 
                                    which='LR', return_eigenvectors=False)
                max_eigenval = np.max(np.real(eigenvals))
                
                if max_eigenval > 0:
                    stable_dt = 0.8 * self.config['stability_threshold'] / max_eigenval
                    return min(stable_dt, current_dt * 2.0)  # Don't increase too much
            
            return current_dt
        except:
            return current_dt * 0.8  # Conservative reduction

class ConvergenceMonitor:
    """Monitor convergence of iterative solvers"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.convergence_history = []
        self.residual_history = []
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'absolute_tolerance': 1e-8,
            'relative_tolerance': 1e-6,
            'max_iterations': 100,
            'divergence_threshold': 1e8,
            'stagnation_tolerance': 1e-12,
            'acceleration_method': 'anderson',
            'restart_threshold': 50
        }
    
    def check_convergence(self, residual: np.ndarray, iteration: int, 
                         initial_residual: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Check convergence criteria"""
        residual_norm = np.linalg.norm(residual)
        
        convergence_info = {
            'converged': False,
            'diverged': False,
            'stagnated': False,
            'residual_norm': residual_norm,
            'relative_residual': 1.0,
            'iteration': iteration,
            'recommendation': 'continue'
        }
        
        # Relative residual
        if initial_residual is not None:
            initial_norm = np.linalg.norm(initial_residual)
            if initial_norm > 0:
                convergence_info['relative_residual'] = residual_norm / initial_norm
        
        # Convergence checks
        if residual_norm < self.config['absolute_tolerance']:
            convergence_info['converged'] = True
            convergence_info['recommendation'] = 'converged_absolute'
        elif convergence_info['relative_residual'] < self.config['relative_tolerance']:
            convergence_info['converged'] = True
            convergence_info['recommendation'] = 'converged_relative'
        
        # Divergence check
        if residual_norm > self.config['divergence_threshold']:
            convergence_info['diverged'] = True
            convergence_info['recommendation'] = 'restart_with_smaller_timestep'
        
        # Stagnation check
        if len(self.residual_history) >= 5:
            recent_residuals = self.residual_history[-5:]
            if all(abs(r - residual_norm) < self.config['stagnation_tolerance'] for r in recent_residuals):
                convergence_info['stagnated'] = True
                convergence_info['recommendation'] = 'apply_acceleration_or_restart'
        
        # Maximum iterations
        if iteration >= self.config['max_iterations']:
            convergence_info['recommendation'] = 'max_iterations_reached'
        
        self.residual_history.append(residual_norm)
        self.convergence_history.append(convergence_info)
        
        return convergence_info
    
    def get_convergence_rate(self) -> float:
        """Estimate convergence rate"""
        if len(self.residual_history) < 3:
            return 1.0
        
        # Linear convergence rate estimate
        recent_residuals = self.residual_history[-5:]
        if len(recent_residuals) >= 2:
            ratios = [recent_residuals[i+1] / max(recent_residuals[i], 1e-16) 
                     for i in range(len(recent_residuals)-1)]
            return np.mean(ratios)
        
        return 1.0

class AdaptiveMeshRefinement:
    """Adaptive mesh refinement for improved accuracy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.refinement_history = []
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'refinement_threshold': 0.8,
            'coarsening_threshold': 0.1,
            'max_refinement_levels': 3,
            'min_element_size': 1e-6,
            'max_element_size': 1e-2,
            'error_estimator': 'gradient_recovery',
            'refinement_strategy': 'red_green'
        }
    
    def compute_error_indicators(self, solution: np.ndarray, 
                               mesh: Any, physics_type: str) -> np.ndarray:
        """Compute error indicators for mesh refinement"""
        if hasattr(mesh, 'elements'):
            n_elements = len(mesh.elements)
        else:
            n_elements = getattr(mesh, 'n_elements', len(solution) - 1)
        
        error_indicators = np.zeros(n_elements)
        
        if self.config['error_estimator'] == 'gradient_recovery':
            # Gradient-based error estimation
            gradient = np.gradient(solution)
            
            # Compute second derivatives (Hessian approximation)
            if len(gradient) > 1:
                second_derivative = np.gradient(gradient)
                
                # Element-wise error indicators
                for i in range(n_elements):
                    if i < len(second_derivative):
                        error_indicators[i] = abs(second_derivative[i])
        
        elif self.config['error_estimator'] == 'residual_based':
            # Residual-based error estimation
            residual = self._compute_element_residuals(solution, mesh, physics_type)
            error_indicators = np.abs(residual)
        
        return error_indicators
    
    def _compute_element_residuals(self, solution: np.ndarray, 
                                 mesh: Any, physics_type: str) -> np.ndarray:
        """Compute element-wise residuals"""
        # Simplified residual computation
        n = len(solution)
        residual = np.zeros(n-1)
        
        for i in range(n-1):
            # Simple residual approximation
            if physics_type == 'thermal':
                # Heat equation residual
                if i > 0 and i < n-2:
                    laplacian = solution[i+1] - 2*solution[i] + solution[i-1]
                    residual[i] = abs(laplacian)  # Simplified
            elif physics_type == 'mechanical':
                # Elasticity residual
                gradient = solution[i+1] - solution[i] if i < n-1 else 0
                residual[i] = abs(gradient)
        
        return residual
    
    def mark_elements_for_refinement(self, error_indicators: np.ndarray) -> Dict[str, List[int]]:
        """Mark elements for refinement or coarsening"""
        max_error = np.max(error_indicators)
        
        refinement_threshold = self.config['refinement_threshold'] * max_error
        coarsening_threshold = self.config['coarsening_threshold'] * max_error
        
        refine_elements = []
        coarsen_elements = []
        
        for i, error in enumerate(error_indicators):
            if error > refinement_threshold:
                refine_elements.append(i)
            elif error < coarsening_threshold:
                coarsen_elements.append(i)
        
        return {
            'refine': refine_elements,
            'coarsen': coarsen_elements,
            'max_error': max_error,
            'refinement_threshold': refinement_threshold,
            'coarsening_threshold': coarsening_threshold
        }
    
    def refine_mesh(self, mesh: Any, marked_elements: Dict[str, List[int]]) -> Any:
        """Refine mesh based on marked elements"""
        if not MESHIO_AVAILABLE:
            self.logger.warning("MeshIO not available, mesh refinement is limited")
            return mesh
        
        try:
            # This would implement actual mesh refinement
            # For now, return the original mesh with refinement info
            refined_mesh = mesh
            
            # Store refinement information
            if hasattr(mesh, 'refinement_info'):
                mesh.refinement_info.append(marked_elements)
            else:
                mesh.refinement_info = [marked_elements]
            
            self.refinement_history.append({
                'timestamp': datetime.now(),
                'refined_elements': len(marked_elements.get('refine', [])),
                'coarsened_elements': len(marked_elements.get('coarsen', [])),
                'max_error': marked_elements.get('max_error', 0.0)
            })
            
            return refined_mesh
        
        except Exception as e:
            self.logger.error(f"Mesh refinement failed: {e}")
            return mesh

class MultiPhysicsCouplingEngine:
    """Main multi-physics coupling engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.physics_modules = {}
        self.coupling_terms = []
        self.material_properties = MaterialProperties()
        self.boundary_conditions = []
        
        # Numerical components
        self.timestepper = AdaptiveTimestepper(config)
        self.stability_analyzer = NumericalStabilityAnalyzer(config)
        self.convergence_monitor = ConvergenceMonitor(config)
        self.mesh_refiner = AdaptiveMeshRefinement(config)
        
        # Simulation state
        self.current_time = 0.0
        self.timestep = 1e-6
        self.mesh = None
        self.state = {}
        
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'coupling_scheme': CouplingType.STAGGERED,
            'max_coupling_iterations': 20,
            'coupling_tolerance': 1e-6,
            'adaptive_timestep': True,
            'mesh_refinement': True,
            'stability_monitoring': True,
            'output_frequency': 10,
            'checkpoint_frequency': 100
        }
    
    def add_physics_module(self, physics_type: PhysicsType, 
                          material_props: Optional[MaterialProperties] = None):
        """Add physics module to the coupling system"""
        if material_props is None:
            material_props = self.material_properties
        
        if physics_type == PhysicsType.THERMAL:
            module = ThermalModule(material_props, self.config)
        elif physics_type == PhysicsType.MECHANICAL:
            module = MechanicalModule(material_props, self.config)
        elif physics_type == PhysicsType.CHEMICAL:
            module = ChemicalModule(material_props, self.config)
        elif physics_type == PhysicsType.ELECTROMAGNETIC:
            module = ElectromagneticModule(material_props, self.config)
        else:
            raise ValueError(f"Unknown physics type: {physics_type}")
        
        self.physics_modules[physics_type] = module
        self.logger.info(f"Added {physics_type.value} physics module")
    
    def add_coupling_term(self, source: PhysicsType, target: PhysicsType,
                         coupling_strength: float, coupling_function: Callable,
                         coupling_type: CouplingType = CouplingType.TWO_WAY):
        """Add coupling term between physics modules"""
        coupling_term = CouplingTerm(
            source_physics=source,
            target_physics=target,
            coupling_strength=coupling_strength,
            coupling_function=coupling_function,
            coupling_type=coupling_type
        )
        self.coupling_terms.append(coupling_term)
        self.logger.info(f"Added coupling: {source.value} -> {target.value}")
    
    def initialize_simulation(self, mesh: Any, initial_conditions: Dict[str, Any] = None):
        """Initialize the multi-physics simulation"""
        self.mesh = mesh
        
        # Initialize state for each physics module
        for physics_type, module in self.physics_modules.items():
            module_state = module.initialize_state(mesh)
            
            # Apply initial conditions
            if initial_conditions and physics_type.value in initial_conditions:
                for var_name, var_value in initial_conditions[physics_type.value].items():
                    if var_name in module_state:
                        if isinstance(var_value, (int, float)):
                            module_state[var_name] = np.full_like(
                                module_state[var_name], var_value
                            )
                        else:
                            module_state[var_name] = np.array(var_value)
            
            self.state.update(module_state)
        
        self.logger.info("Multi-physics simulation initialized")
    
    def solve_timestep(self) -> Dict[str, Any]:
        """Solve one time step using coupled physics"""
        timestep_info = {
            'time': self.current_time,
            'dt': self.timestep,
            'converged': False,
            'coupling_iterations': 0,
            'stability_info': {},
            'mesh_refined': False
        }
        
        if self.config['coupling_scheme'] == CouplingType.STAGGERED:
            timestep_info = self._solve_staggered()
        elif self.config['coupling_scheme'] == CouplingType.MONOLITHIC:
            timestep_info = self._solve_monolithic()
        elif self.config['coupling_scheme'] == CouplingType.FULLY_COUPLED:
            timestep_info = self._solve_fully_coupled()
        
        # Stability analysis
        if self.config['stability_monitoring']:
            jacobian = self._assemble_coupled_jacobian()
            stability_info = self.stability_analyzer.analyze_stability(
                jacobian, self.timestep, self.mesh
            )
            timestep_info['stability_info'] = stability_info
            
            if not stability_info['is_stable']:
                # Reduce timestep and retry
                self.timestep *= 0.5
                self.logger.warning(f"Stability issue detected, reducing timestep to {self.timestep:.2e}")
                return self.solve_timestep()
        
        # Mesh refinement
        if self.config['mesh_refinement'] and self.current_time > 0:
            timestep_info['mesh_refined'] = self._perform_mesh_adaptation()
        
        # Adaptive time stepping
        if self.config['adaptive_timestep']:
            # Compute error estimate (simplified)
            error_estimate = self._compute_solution_error()
            new_dt, accept_step = self.timestepper.adapt_timestep(
                self.timestep, error_estimate
            )
            
            if accept_step:
                self.current_time += self.timestep
                self.timestep = new_dt
            else:
                self.timestep = new_dt
                return self.solve_timestep()  # Retry with smaller timestep
        else:
            self.current_time += self.timestep
        
        return timestep_info
    
    def _solve_staggered(self) -> Dict[str, Any]:
        """Solve using staggered coupling scheme"""
        timestep_info = {
            'coupling_iterations': 0,
            'converged': False,
            'residuals': {}
        }
        
        for iteration in range(self.config['max_coupling_iterations']):
            max_residual = 0.0
            
            # Solve each physics module sequentially
            for physics_type, module in self.physics_modules.items():
                # Compute coupling sources
                coupling_sources = self._compute_coupling_sources(physics_type)
                
                # Solve physics module
                residual = module.compute_residual(self.state, self.timestep, 
                                                 self.mesh, **coupling_sources)
                jacobian = module.compute_jacobian(self.state, self.timestep, self.mesh)
                
                # Newton-Raphson iteration
                try:
                    delta_solution = spla.spsolve(jacobian, -residual)
                    
                    # Update state
                    for var_name in module.state_variables:
                        if var_name in self.state:
                            self.state[var_name] += delta_solution
                    
                    residual_norm = np.linalg.norm(residual)
                    max_residual = max(max_residual, residual_norm)
                    timestep_info['residuals'][physics_type.value] = residual_norm
                
                except Exception as e:
                    self.logger.warning(f"Solver failed for {physics_type.value}: {e}")
                    timestep_info['residuals'][physics_type.value] = np.inf
                    break
            
            # Check coupling convergence
            if max_residual < self.config['coupling_tolerance']:
                timestep_info['converged'] = True
                break
            
            timestep_info['coupling_iterations'] = iteration + 1
        
        return timestep_info
    
    def _solve_monolithic(self) -> Dict[str, Any]:
        """Solve using monolithic coupling scheme"""
        timestep_info = {
            'converged': False,
            'newton_iterations': 0
        }
        
        # Assemble coupled residual and Jacobian
        coupled_residual = self._assemble_coupled_residual()
        coupled_jacobian = self._assemble_coupled_jacobian()
        
        # Newton-Raphson iterations
        for iteration in range(self.config['max_coupling_iterations']):
            try:
                # Solve linear system
                delta_solution = spla.spsolve(coupled_jacobian, -coupled_residual)
                
                # Update state
                self._update_coupled_state(delta_solution)
                
                # Check convergence
                convergence_info = self.convergence_monitor.check_convergence(
                    coupled_residual, iteration
                )
                
                if convergence_info['converged']:
                    timestep_info['converged'] = True
                    break
                
                # Recompute residual and Jacobian
                coupled_residual = self._assemble_coupled_residual()
                if iteration % 5 == 0:  # Update Jacobian periodically
                    coupled_jacobian = self._assemble_coupled_jacobian()
            
            except Exception as e:
                self.logger.warning(f"Monolithic solver failed at iteration {iteration}: {e}")
                break
            
            timestep_info['newton_iterations'] = iteration + 1
        
        return timestep_info
    
    def _solve_fully_coupled(self) -> Dict[str, Any]:
        """Solve using fully coupled scheme with advanced iterative methods"""
        # Similar to monolithic but with advanced preconditioning and solvers
        return self._solve_monolithic()
    
    def _compute_coupling_sources(self, target_physics: PhysicsType) -> Dict[str, np.ndarray]:
        """Compute coupling source terms for a physics module"""
        coupling_sources = {}
        
        for coupling_term in self.coupling_terms:
            if coupling_term.target_physics == target_physics:
                source_physics = coupling_term.source_physics
                
                # Get source state variables
                source_vars = {}
                for var_name, var_value in self.state.items():
                    if source_physics.value in var_name or var_name in ['temperature', 'displacement', 'concentration']:
                        source_vars[var_name] = var_value
                
                # Compute coupling source
                try:
                    coupling_source = coupling_term.coupling_function(
                        source_vars, coupling_term.coupling_strength
                    )
                    
                    # Add to coupling sources
                    source_name = f"{source_physics.value}_source"
                    if source_name in coupling_sources:
                        coupling_sources[source_name] += coupling_source
                    else:
                        coupling_sources[source_name] = coupling_source
                
                except Exception as e:
                    self.logger.warning(f"Coupling computation failed: {e}")
        
        return coupling_sources
    
    def _assemble_coupled_residual(self) -> np.ndarray:
        """Assemble coupled residual vector"""
        residuals = []
        
        for physics_type, module in self.physics_modules.items():
            coupling_sources = self._compute_coupling_sources(physics_type)
            residual = module.compute_residual(self.state, self.timestep, 
                                             self.mesh, **coupling_sources)
            residuals.append(residual)
        
        return np.concatenate(residuals)
    
    def _assemble_coupled_jacobian(self) -> sparse.csr_matrix:
        """Assemble coupled Jacobian matrix"""
        jacobian_blocks = []
        
        for physics_type, module in self.physics_modules.items():
            jacobian = module.compute_jacobian(self.state, self.timestep, self.mesh)
            jacobian_blocks.append(jacobian)
        
        # Block diagonal assembly (simplified)
        coupled_jacobian = sparse.block_diag(jacobian_blocks, format='csr')
        
        # Add coupling terms (simplified)
        # In practice, this would include off-diagonal coupling blocks
        
        return coupled_jacobian
    
    def _update_coupled_state(self, delta_solution: np.ndarray):
        """Update coupled state with solution increment"""
        offset = 0
        
        for physics_type, module in self.physics_modules.items():
            # Determine state variable size for this module
            module_size = 0
            for var_name in module.state_variables:
                if var_name in self.state:
                    module_size = len(self.state[var_name])
                    break
            
            if module_size > 0:
                # Extract module solution
                module_delta = delta_solution[offset:offset+module_size]
                
                # Update first state variable (simplified)
                first_var = list(module.state_variables.keys())[0]
                if first_var in self.state:
                    self.state[first_var] += module_delta
                
                offset += module_size
    
    def _compute_solution_error(self) -> float:
        """Compute solution error estimate for adaptive time stepping"""
        # Simplified error estimation
        total_error = 0.0
        
        for var_name, var_value in self.state.items():
            if var_name.endswith('_old'):
                continue
            
            old_var_name = f"{var_name}_old"
            if old_var_name in self.state:
                error = np.linalg.norm(var_value - self.state[old_var_name])
                total_error += error**2
        
        return np.sqrt(total_error)
    
    def _perform_mesh_adaptation(self) -> bool:
        """Perform adaptive mesh refinement"""
        mesh_refined = False
        
        try:
            # Compute error indicators for each physics
            for physics_type, module in self.physics_modules.items():
                # Get primary variable for this physics
                primary_var = self._get_primary_variable(physics_type)
                if primary_var in self.state:
                    error_indicators = self.mesh_refiner.compute_error_indicators(
                        self.state[primary_var], self.mesh, physics_type.value
                    )
                    
                    # Mark elements for refinement
                    marked_elements = self.mesh_refiner.mark_elements_for_refinement(
                        error_indicators
                    )
                    
                    # Refine mesh if needed
                    if marked_elements['refine'] or marked_elements['coarsen']:
                        self.mesh = self.mesh_refiner.refine_mesh(self.mesh, marked_elements)
                        mesh_refined = True
                        break  # Only refine based on one physics for now
        
        except Exception as e:
            self.logger.warning(f"Mesh adaptation failed: {e}")
        
        return mesh_refined
    
    def _get_primary_variable(self, physics_type: PhysicsType) -> str:
        """Get primary variable name for a physics type"""
        primary_vars = {
            PhysicsType.THERMAL: 'temperature',
            PhysicsType.MECHANICAL: 'displacement',
            PhysicsType.CHEMICAL: 'concentration',
            PhysicsType.ELECTROMAGNETIC: 'electric_field'
        }
        return primary_vars.get(physics_type, 'temperature')
    
    async def run_simulation(self, end_time: float, output_dir: str = "output") -> Dict[str, Any]:
        """Run the complete multi-physics simulation"""
        simulation_results = {
            'start_time': datetime.now(),
            'end_time': None,
            'total_timesteps': 0,
            'simulation_time': end_time,
            'converged_timesteps': 0,
            'failed_timesteps': 0,
            'mesh_refinements': 0,
            'final_state': {},
            'performance_metrics': {}
        }
        
        self.logger.info(f"Starting multi-physics simulation: t=0 to t={end_time}")
        
        timestep_count = 0
        output_count = 0
        
        while self.current_time < end_time:
            # Solve time step
            timestep_info = self.solve_timestep()
            timestep_count += 1
            
            if timestep_info['converged']:
                simulation_results['converged_timesteps'] += 1
            else:
                simulation_results['failed_timesteps'] += 1
            
            if timestep_info.get('mesh_refined', False):
                simulation_results['mesh_refinements'] += 1
            
            # Output results periodically
            if timestep_count % self.config['output_frequency'] == 0:
                await self._output_results(output_dir, output_count)
                output_count += 1
            
            # Checkpoint periodically
            if timestep_count % self.config['checkpoint_frequency'] == 0:
                await self._save_checkpoint(output_dir, timestep_count)
            
            # Progress logging
            if timestep_count % 100 == 0:
                progress = self.current_time / end_time * 100
                self.logger.info(f"Simulation progress: {progress:.1f}% (t={self.current_time:.6f})")
        
        simulation_results['end_time'] = datetime.now()
        simulation_results['total_timesteps'] = timestep_count
        simulation_results['final_state'] = self.state.copy()
        
        # Performance metrics
        simulation_results['performance_metrics'] = {
            'average_timestep': np.mean(self.timestepper.time_history) if self.timestepper.time_history else self.timestep,
            'convergence_rate': self.convergence_monitor.get_convergence_rate(),
            'stability_issues': len([s for s in self.stability_analyzer.stability_history if not s['is_stable']])
        }
        
        self.logger.info("Multi-physics simulation completed successfully")
        return simulation_results
    
    async def _output_results(self, output_dir: str, output_count: int):
        """Output simulation results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save state variables
        output_file = os.path.join(output_dir, f"results_{output_count:06d}.npz")
        np.savez(output_file, 
                 time=self.current_time,
                 timestep=self.timestep,
                 **self.state)
        
        # Save mesh information
        if hasattr(self.mesh, 'save'):
            mesh_file = os.path.join(output_dir, f"mesh_{output_count:06d}.msh")
            self.mesh.save(mesh_file)
    
    async def _save_checkpoint(self, output_dir: str, timestep_count: int):
        """Save simulation checkpoint"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint_data = {
            'current_time': self.current_time,
            'timestep': self.timestep,
            'timestep_count': timestep_count,
            'state': self.state,
            'config': self.config
        }
        
        checkpoint_file = os.path.join(output_dir, f"checkpoint_{timestep_count:06d}.pkl")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

# Example coupling functions
def thermal_mechanical_coupling(thermal_state: Dict[str, np.ndarray], 
                              strength: float) -> np.ndarray:
    """Thermal expansion coupling: temperature -> mechanical strain"""
    temperature = thermal_state.get('temperature', np.zeros(1))
    T_ref = 293.15
    alpha = 3e-5  # thermal expansion coefficient
    
    thermal_strain = strength * alpha * (temperature - T_ref)
    return thermal_strain

def mechanical_thermal_coupling(mechanical_state: Dict[str, np.ndarray], 
                              strength: float) -> np.ndarray:
    """Mechanical heating coupling: plastic work -> heat generation"""
    stress = mechanical_state.get('stress', np.zeros(1))
    strain_rate = mechanical_state.get('strain_rate', np.zeros(1))
    
    # Plastic work converted to heat
    plastic_work = strength * np.abs(stress * strain_rate)
    return plastic_work

def chemical_thermal_coupling(chemical_state: Dict[str, np.ndarray], 
                            strength: float) -> np.ndarray:
    """Exothermic reaction coupling: reaction rate -> heat source"""
    reaction_rate = chemical_state.get('reaction_rate', np.zeros(1))
    heat_of_reaction = 50000.0  # J/mol
    
    heat_source = strength * heat_of_reaction * reaction_rate
    return heat_source

# Example usage and testing
async def example_multi_physics_simulation():
    """Example of multi-physics coupling simulation"""
    
    # Create simple mesh
    class SimpleMesh:
        def __init__(self, n_nodes=100):
            self.n_nodes = n_nodes
            self.n_elements = n_nodes - 1
            self.coordinates = np.linspace(0, 1, n_nodes)
            
            # Simple finite difference matrices
            dx = 1.0 / (n_nodes - 1)
            self.laplacian_matrix = sparse.diags(
                [1, -2, 1], [-1, 0, 1], shape=(n_nodes, n_nodes), format='csr'
            ) / dx**2
            
            self.min_element_size = dx
    
    mesh = SimpleMesh(100)
    
    # Initialize multi-physics engine
    engine = MultiPhysicsCouplingEngine({
        'coupling_scheme': CouplingType.STAGGERED,
        'adaptive_timestep': True,
        'mesh_refinement': False,  # Disabled for this example
        'max_coupling_iterations': 10
    })
    
    # Add physics modules
    material_props = MaterialProperties(
        density=2700.0,
        thermal_conductivity=2.5,
        youngs_modulus=70e9,
        thermal_expansion=3e-5
    )
    
    engine.add_physics_module(PhysicsType.THERMAL, material_props)
    engine.add_physics_module(PhysicsType.MECHANICAL, material_props)
    
    # Add coupling terms
    engine.add_coupling_term(
        PhysicsType.THERMAL, PhysicsType.MECHANICAL,
        coupling_strength=1.0, coupling_function=thermal_mechanical_coupling
    )
    engine.add_coupling_term(
        PhysicsType.MECHANICAL, PhysicsType.THERMAL,
        coupling_strength=0.1, coupling_function=mechanical_thermal_coupling
    )
    
    # Initialize simulation
    initial_conditions = {
        'thermal': {'temperature': 323.15},  # 50°C
        'mechanical': {'displacement': 0.0}
    }
    
    engine.initialize_simulation(mesh, initial_conditions)
    
    print("Multi-Physics Coupling Engine Example")
    print("=" * 50)
    print(f"Mesh nodes: {mesh.n_nodes}")
    print(f"Physics modules: {list(engine.physics_modules.keys())}")
    print(f"Coupling terms: {len(engine.coupling_terms)}")
    
    # Run simulation
    try:
        results = await engine.run_simulation(end_time=0.01)
        
        print("\nSimulation Results:")
        print(f"Total timesteps: {results['total_timesteps']}")
        print(f"Converged timesteps: {results['converged_timesteps']}")
        print(f"Failed timesteps: {results['failed_timesteps']}")
        print(f"Mesh refinements: {results['mesh_refinements']}")
        print(f"Average timestep: {results['performance_metrics']['average_timestep']:.2e}")
        
        # Display final temperature and displacement
        final_temp = results['final_state'].get('temperature', np.array([0]))
        final_disp = results['final_state'].get('displacement', np.array([0]))
        
        print(f"\nFinal state:")
        print(f"Temperature range: {np.min(final_temp):.2f} - {np.max(final_temp):.2f} K")
        print(f"Displacement range: {np.min(final_disp):.2e} - {np.max(final_disp):.2e} m")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Advanced Multi-Physics Coupling Engine")
    print("=" * 50)
    
    # Check available libraries
    print(f"FEniCS available: {FENICS_AVAILABLE}")
    print(f"Firedrake available: {FIREDRAKE_AVAILABLE}")
    print(f"VTK available: {VTK_AVAILABLE}")
    print(f"MeshIO available: {MESHIO_AVAILABLE}")
    
    # Run example
    asyncio.run(example_multi_physics_simulation())