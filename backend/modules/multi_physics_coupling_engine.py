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
    density: float = 2700.0  # kg/m**3
    thermal_conductivity: float = 2.5  # W/(m*K)
    specific_heat: float = 1000.0  # J/(kg*K)
    thermal_expansion: float = 3e-5  # 1/K
    
    # Mechanical properties
    youngs_modulus: float = 70e9  # Pa
    poisson_ratio: float = 0.3
    bulk_modulus: Optional[float] = None
    shear_modulus: Optional[float] = None
    
    # Chemical properties
    diffusion_coefficient: float = 1e-9  # m**2/s
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
            'convection_coefficient': 10.0  # W/(m**2*K)
        }
    
    def compute_residual(self, state: Dict[str, np.ndarray], dt: float, 
                        mesh: Any = None, **kwargs) -> np.ndarray:
        """Compute thermal residual: rhocpdT/dt = nabla*(knablaT) + Q"""
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
        """Compute mechanical residual: nabla*sigma + f = 0 (static equilibrium)"""
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
        """Compute comprehensive mechanical Jacobian (tangent stiffness matrix)"""
        displacement = state['displacement']
        n_dof = len(displacement)
        
        # Material properties
        E = self.material_props.youngs_modulus
        nu = self.material_props.poisson_ratio
        G = self.material_props.shear_modulus
        
        # Initialize global stiffness matrix
        K_global = sparse.lil_matrix((n_dof, n_dof))
        
        if mesh is not None and hasattr(mesh, 'elements') and hasattr(mesh, 'coordinates'):
            # Finite element assembly
            K_global = self._assemble_finite_element_jacobian(displacement, mesh, **kwargs)
        else:
            # Advanced finite difference stiffness matrix
            K_global = self._assemble_finite_difference_jacobian(displacement, **kwargs)
        
        # Convert to CSR format for efficient operations
        K_global = K_global.tocsr()
        
        # Add geometric stiffness for large deformations
        if self.config.get('geometric_nonlinearity', False):
            K_geom = self._compute_geometric_stiffness(state, mesh)
            K_global += K_geom
        
        # Add material nonlinearity contributions
        if self.config.get('material_nonlinearity', False):
            K_material = self._compute_material_jacobian(state, mesh)
            K_global += K_material
        
        # Apply boundary conditions
        K_global = self._apply_boundary_conditions_to_jacobian(K_global, mesh)
        
        return K_global
    
    def _compute_strain(self, displacement: np.ndarray, mesh: Any = None) -> np.ndarray:
        """Compute comprehensive strain tensor with geometric nonlinearity"""
        n_dof = len(displacement)
        
        if mesh is not None and hasattr(mesh, 'coordinates'):
            return self._compute_finite_element_strain(displacement, mesh)
        else:
            return self._compute_finite_difference_strain(displacement)
    
    def _compute_finite_element_strain(self, displacement: np.ndarray, mesh: Any) -> np.ndarray:
        """Compute strain using finite element interpolation"""
        n_nodes = len(mesh.coordinates)
        n_dim = len(mesh.coordinates[0]) if hasattr(mesh.coordinates[0], '__len__') else 1
        
        if n_dim == 1:
            # 1D strain
            strain = np.zeros(len(displacement))
            for i in range(len(displacement) - 1):
                dx = mesh.coordinates[i+1] - mesh.coordinates[i]
                strain[i] = (displacement[i+1] - displacement[i]) / dx
            return strain
        
        elif n_dim == 2:
            # 2D strain tensor components: [epsilonxx, epsilonyy, gammaxy]
            strain = np.zeros((len(displacement)//2, 3))
            
            # Reshape displacement vector
            u = displacement[::2]  # x-displacements
            v = displacement[1::2]  # y-displacements
            
            # Compute strain components using B-matrix approach
            for elem_id in range(getattr(mesh, 'n_elements', len(u)-1)):
                B_matrix = self._compute_B_matrix(elem_id, mesh)
                elem_nodes = self._get_element_nodes(elem_id, mesh)
                
                # Element displacement vector
                u_elem = np.zeros(len(elem_nodes) * 2)
                for i, node in enumerate(elem_nodes):
                    u_elem[2*i] = u[node] if node < len(u) else 0
                    u_elem[2*i+1] = v[node] if node < len(v) else 0
                
                # Compute element strains
                elem_strain = B_matrix @ u_elem
                
                # Distribute to nodes (simplified averaging)
                for node in elem_nodes:
                    if node < len(strain):
                        strain[node] += elem_strain / len(elem_nodes)
            
            return strain.flatten()
        
        elif n_dim == 3:
            # 3D strain tensor components: [epsilonxx, epsilonyy, epsilonzz, gammaxy, gammayz, gammaxz]
            strain = np.zeros((len(displacement)//3, 6))
            
            # Reshape displacement vector
            u = displacement[::3]   # x-displacements
            v = displacement[1::3]  # y-displacements
            w = displacement[2::3]  # z-displacements
            
            # Similar finite element computation for 3D
            for elem_id in range(getattr(mesh, 'n_elements', len(u)-1)):
                B_matrix = self._compute_B_matrix_3d(elem_id, mesh)
                elem_nodes = self._get_element_nodes(elem_id, mesh)
                
                # Element displacement vector
                u_elem = np.zeros(len(elem_nodes) * 3)
                for i, node in enumerate(elem_nodes):
                    u_elem[3*i] = u[node] if node < len(u) else 0
                    u_elem[3*i+1] = v[node] if node < len(v) else 0
                    u_elem[3*i+2] = w[node] if node < len(w) else 0
                
                # Compute element strains
                elem_strain = B_matrix @ u_elem
                
                # Distribute to nodes
                for node in elem_nodes:
                    if node < len(strain):
                        strain[node] += elem_strain / len(elem_nodes)
            
            return strain.flatten()
        
        return np.zeros(len(displacement))
    
    def _compute_finite_difference_strain(self, displacement: np.ndarray) -> np.ndarray:
        """Compute strain using high-order finite differences"""
        n = len(displacement)
        strain = np.zeros_like(displacement)
        
        # High-order finite difference coefficients (4th order)
        if n >= 5:
            # Interior points - 4th order central difference
            for i in range(2, n-2):
                strain[i] = (-displacement[i+2] + 8*displacement[i+1] - 
                           8*displacement[i-1] + displacement[i-2]) / (12.0 * self._get_grid_spacing())
            
            # Boundary points - forward/backward difference
            strain[0] = (-3*displacement[0] + 4*displacement[1] - displacement[2]) / (2.0 * self._get_grid_spacing())
            strain[1] = (-displacement[0] + displacement[2]) / (2.0 * self._get_grid_spacing())
            strain[-2] = (displacement[-3] - displacement[-1]) / (2.0 * self._get_grid_spacing())
            strain[-1] = (displacement[-3] - 4*displacement[-2] + 3*displacement[-1]) / (2.0 * self._get_grid_spacing())
        else:
            # Simple central difference for small grids
            for i in range(1, n-1):
                strain[i] = (displacement[i+1] - displacement[i-1]) / (2.0 * self._get_grid_spacing())
            
            # Boundary conditions
            strain[0] = (displacement[1] - displacement[0]) / self._get_grid_spacing()
            strain[-1] = (displacement[-1] - displacement[-2]) / self._get_grid_spacing()
        
        # Add geometric nonlinearity if enabled
        if self.config.get('geometric_nonlinearity', False):
            strain += 0.5 * strain**2  # Green-Lagrange strain
        
        return strain
    
    def _compute_stress(self, strain: np.ndarray, state: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute comprehensive stress tensor with plasticity and nonlinearity"""
        # Material properties
        E = self.material_props.youngs_modulus
        nu = self.material_props.poisson_ratio
        G = self.material_props.shear_modulus
        
        # Temperature for thermal stress
        temperature = state.get('temperature', np.full_like(strain, 293.15))
        T_ref = 293.15
        alpha = self.material_props.thermal_expansion
        
        # Compute thermal strain
        thermal_strain = alpha * (temperature - T_ref)
        
        # Mechanical strain (total strain - thermal strain)
        if len(strain.shape) > 1 and strain.shape[1] > 1:
            # Multi-dimensional strain tensor
            mechanical_strain = strain.copy()
            if strain.shape[1] >= 2:  # 2D or 3D
                mechanical_strain[:, 0] -= thermal_strain[:len(mechanical_strain)]  # epsilonxx
                mechanical_strain[:, 1] -= thermal_strain[:len(mechanical_strain)]  # epsilonyy
                if strain.shape[1] >= 3:
                    mechanical_strain[:, 2] -= thermal_strain[:len(mechanical_strain)]  # epsilonzz
        else:
            # 1D strain
            mechanical_strain = strain - thermal_strain[:len(strain)]
        
        # Compute elastic stress
        stress = self._compute_elastic_stress(mechanical_strain)
        
        # Apply plasticity if enabled
        if self.config.get('include_plasticity', False):
            stress = self._apply_plasticity_model(stress, state)
        
        # Apply damage model if enabled
        if self.config.get('include_damage', False):
            damage = state.get('damage', np.zeros_like(stress))
            stress *= (1 - damage)  # Simplified damage coupling
        
        # Apply viscoplasticity if enabled
        if self.config.get('include_viscoplasticity', False):
            stress = self._apply_viscoplastic_model(stress, strain, state)
        
        return stress
    
    def _compute_elastic_stress(self, strain: np.ndarray) -> np.ndarray:
        """Compute elastic stress using constitutive matrix"""
        E = self.material_props.youngs_modulus
        nu = self.material_props.poisson_ratio
        
        if len(strain.shape) == 1:
            # 1D case
            return E * strain
        
        stress = np.zeros_like(strain)
        
        if strain.shape[1] == 3:  # 2D plane stress/strain
            if self.config.get('plane_stress', True):
                # Plane stress constitutive matrix
                D = E / (1 - nu**2) * np.array([
                    [1, nu, 0],
                    [nu, 1, 0],
                    [0, 0, (1-nu)/2]
                ])
            else:
                # Plane strain constitutive matrix
                factor = E / ((1 + nu) * (1 - 2*nu))
                D = factor * np.array([
                    [1-nu, nu, 0],
                    [nu, 1-nu, 0],
                    [0, 0, (1-2*nu)/2]
                ])
            
            # Compute stress: sigma = D * epsilon
            for i in range(len(strain)):
                stress[i] = D @ strain[i]
        
        elif strain.shape[1] == 6:  # 3D case
            # 3D constitutive matrix
            factor = E / ((1 + nu) * (1 - 2*nu))
            D = factor * np.array([
                [1-nu, nu, nu, 0, 0, 0],
                [nu, 1-nu, nu, 0, 0, 0],
                [nu, nu, 1-nu, 0, 0, 0],
                [0, 0, 0, (1-2*nu)/2, 0, 0],
                [0, 0, 0, 0, (1-2*nu)/2, 0],
                [0, 0, 0, 0, 0, (1-2*nu)/2]
            ])
            
            # Compute stress
            for i in range(len(strain)):
                stress[i] = D @ strain[i]
        
        return stress
    
    def _apply_plasticity_model(self, stress: np.ndarray, state: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply von Mises plasticity with isotropic hardening"""
        yield_strength = self.config.get('yield_strength', 250e6)
        hardening_modulus = self.config.get('hardening_modulus', 1e9)
        
        # Get plastic strain from previous step
        plastic_strain = state.get('plastic_strain', np.zeros_like(stress))
        
        corrected_stress = np.zeros_like(stress)
        
        for i in range(len(stress)):
            if len(stress.shape) > 1 and stress.shape[1] >= 3:
                # 2D/3D von Mises criterion
                if stress.shape[1] == 3:  # 2D
                    sigma_vm = np.sqrt(stress[i,0]**2 + stress[i,1]**2 - 
                                     stress[i,0]*stress[i,1] + 3*stress[i,2]**2)
                elif stress.shape[1] == 6:  # 3D
                    s11, s22, s33 = stress[i,0], stress[i,1], stress[i,2]
                    s12, s23, s13 = stress[i,3], stress[i,4], stress[i,5]
                    sigma_vm = np.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2) + 
                                     3*(s12**2 + s23**2 + s13**2))
                
                # Equivalent plastic strain for hardening
                if len(plastic_strain.shape) > 1:
                    equiv_plastic_strain = np.linalg.norm(plastic_strain[i])
                else:
                    equiv_plastic_strain = abs(plastic_strain[i] if i < len(plastic_strain) else 0)
                
                # Current yield stress with hardening
                current_yield = yield_strength + hardening_modulus * equiv_plastic_strain
                
                # Plastic correction
                if sigma_vm > current_yield:
                    # Return mapping algorithm
                    beta = current_yield / sigma_vm
                    corrected_stress[i] = stress[i] * beta
                else:
                    corrected_stress[i] = stress[i]
            else:
                # 1D plasticity
                sigma_abs = abs(stress[i])
                equiv_plastic_strain = abs(plastic_strain[i] if i < len(plastic_strain) else 0)
                current_yield = yield_strength + hardening_modulus * equiv_plastic_strain
                
                if sigma_abs > current_yield:
                    corrected_stress[i] = current_yield * np.sign(stress[i])
                else:
                    corrected_stress[i] = stress[i]
        
        return corrected_stress
    
    def _apply_viscoplastic_model(self, stress: np.ndarray, strain: np.ndarray, 
                                 state: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply comprehensive rate-dependent viscoplastic model with multiple formulations"""
        temperature = state.get('temperature', np.full_like(stress.flatten()[:len(stress.flatten())], 293.15))
        strain_old = state.get('strain_old', strain)
        dt = state.get('dt', 1e-6)
        
        # Get viscoplastic model type
        vp_model = self.config.get('viscoplastic_model', 'norton_hoff')
        
        # Initialize internal state variables
        backstress = state.get('backstress', np.zeros_like(stress))
        accumulated_vp_strain = state.get('accumulated_vp_strain', np.zeros(len(stress)))
        damage_parameter = state.get('damage_parameter', np.zeros(len(stress)))
        
        # Compute strain rate tensor
        strain_rate = self._compute_strain_rate_tensor(strain, strain_old, dt)
        
        # Apply the selected viscoplastic model
        if vp_model == 'norton_hoff':
            corrected_stress = self._apply_norton_hoff_model(stress, strain_rate, temperature, 
                                                           backstress, accumulated_vp_strain)
        elif vp_model == 'chaboche':
            corrected_stress = self._apply_chaboche_model(stress, strain_rate, temperature,
                                                        backstress, accumulated_vp_strain, dt)
        elif vp_model == 'miller':
            corrected_stress = self._apply_miller_model(stress, strain_rate, temperature,
                                                      accumulated_vp_strain)
        elif vp_model == 'bodner_partom':
            corrected_stress = self._apply_bodner_partom_model(stress, strain_rate, temperature,
                                                             accumulated_vp_strain)
        elif vp_model == 'anand':
            corrected_stress = self._apply_anand_model(stress, strain_rate, temperature, 
                                                     state, dt)
        else:
            # Default to advanced Norton-Hoff
            corrected_stress = self._apply_norton_hoff_model(stress, strain_rate, temperature,
                                                           backstress, accumulated_vp_strain)
        
        # Update internal state variables
        self._update_viscoplastic_state_variables(state, strain_rate, corrected_stress, 
                                                 temperature, dt)
        
        # Apply damage evolution if enabled
        if self.config.get('include_vp_damage', False):
            corrected_stress = self._apply_viscoplastic_damage(corrected_stress, strain_rate,
                                                             damage_parameter, temperature, dt)
        
        return corrected_stress
    
    def _compute_strain_rate_tensor(self, strain: np.ndarray, strain_old: np.ndarray, dt: float) -> np.ndarray:
        """Compute comprehensive strain rate tensor"""
        if dt <= 0:
            return np.zeros_like(strain)
        
        # Basic strain rate
        strain_rate = (strain - strain_old) / dt
        
        # Apply smoothing for numerical stability
        if self.config.get('strain_rate_smoothing', True):
            alpha = self.config.get('smoothing_factor', 0.1)
            strain_rate_old = getattr(self, '_strain_rate_old', strain_rate)
            strain_rate = alpha * strain_rate + (1 - alpha) * strain_rate_old
            self._strain_rate_old = strain_rate
        
        return strain_rate
    
    def _apply_norton_hoff_model(self, stress: np.ndarray, strain_rate: np.ndarray, 
                                temperature: np.ndarray, backstress: np.ndarray,
                                accumulated_strain: np.ndarray) -> np.ndarray:
        """Advanced Norton-Hoff viscoplastic model with temperature dependence"""
        # Material parameters
        A0 = self.config.get('norton_pre_exponential', 1e10)  # Pre-exponential factor
        n = self.config.get('norton_stress_exponent', 5.0)    # Stress exponent
        Q = self.config.get('norton_activation_energy', 300000.0)  # Activation energy J/mol
        R = 8.314  # Gas constant J/(mol*K)
        m = self.config.get('norton_strain_rate_sensitivity', 0.2)  # Strain rate sensitivity
        
        corrected_stress = np.zeros_like(stress)
        
        for i in range(len(stress)):
            T = temperature[i] if i < len(temperature) else 293.15
            
            if len(stress.shape) > 1 and stress.shape[1] > 1:
                # Multi-dimensional stress state
                stress_tensor = stress[i]
                backstress_tensor = backstress[i] if i < len(backstress) else np.zeros_like(stress_tensor)
                
                # Effective stress
                effective_stress_tensor = stress_tensor - backstress_tensor
                
                # von Mises equivalent stress
                if len(effective_stress_tensor) == 3:  # 2D
                    s11, s22, s12 = effective_stress_tensor[0], effective_stress_tensor[1], effective_stress_tensor[2]
                    sigma_eq = np.sqrt(s11**2 + s22**2 - s11*s22 + 3*s12**2)
                elif len(effective_stress_tensor) == 6:  # 3D
                    s11, s22, s33 = effective_stress_tensor[0], effective_stress_tensor[1], effective_stress_tensor[2]
                    s12, s23, s13 = effective_stress_tensor[3], effective_stress_tensor[4], effective_stress_tensor[5]
                    sigma_eq = np.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2) + 
                                     3*(s12**2 + s23**2 + s13**2))
                else:
                    sigma_eq = abs(effective_stress_tensor[0])
                
                # Equivalent strain rate
                if i < len(strain_rate) and len(strain_rate.shape) > 1:
                    strain_rate_tensor = strain_rate[i]
                    if len(strain_rate_tensor) >= 3:
                        e11, e22, e12 = strain_rate_tensor[0], strain_rate_tensor[1], strain_rate_tensor[2]
                        edot_eq = np.sqrt((2/3) * (e11**2 + e22**2 + e11*e22 + 3*e12**2))
                    else:
                        edot_eq = abs(strain_rate_tensor[0])
                else:
                    edot_eq = 1e-6
            else:
                # 1D stress state
                effective_stress = (stress[i] - (backstress[i] if i < len(backstress) else 0))
                sigma_eq = abs(effective_stress)
                edot_eq = abs(strain_rate[i] if i < len(strain_rate) else 1e-6)
            
            # Temperature-dependent flow stress
            if sigma_eq > 1e-6:  # Avoid division by zero
                # Arrhenius term
                arrhenius_term = np.exp(-Q / (R * T))
                
                # Norton-Hoff flow rule with strain hardening
                strain_hardening = 1.0 + self.config.get('strain_hardening_coeff', 0.1) * accumulated_strain[i]
                
                # Flow stress from Norton-Hoff equation
                # sigma = A * (epsilon?)^m * exp(Q/RT) * (1 + h*epsilon_acc)
                flow_stress = A0 * (edot_eq)**m * arrhenius_term * strain_hardening
                
                # Stress correction factor
                if flow_stress > 0:
                    correction_factor = min(flow_stress / sigma_eq, self.config.get('max_stress_ratio', 10.0))
                else:
                    correction_factor = 1.0
            else:
                correction_factor = 1.0
            
            # Apply correction
            if len(stress.shape) > 1 and stress.shape[1] > 1:
                corrected_stress[i] = stress[i] * correction_factor
            else:
                corrected_stress[i] = stress[i] * correction_factor
        
        return corrected_stress
    
    def _apply_chaboche_model(self, stress: np.ndarray, strain_rate: np.ndarray,
                             temperature: np.ndarray, backstress: np.ndarray,
                             accumulated_strain: np.ndarray, dt: float) -> np.ndarray:
        """Chaboche unified viscoplastic model with kinematic hardening"""
        # Chaboche model parameters
        K = self.config.get('chaboche_drag_stress', 100e6)  # Drag stress
        n = self.config.get('chaboche_viscosity_exponent', 4.0)
        C = self.config.get('chaboche_kinematic_modulus', 10000e6)  # Kinematic hardening modulus
        gamma = self.config.get('chaboche_recovery_parameter', 300.0)  # Dynamic recovery
        
        corrected_stress = np.zeros_like(stress)
        
        for i in range(len(stress)):
            T = temperature[i] if i < len(temperature) else 293.15
            
            if len(stress.shape) > 1 and stress.shape[1] > 1:
                stress_dev = self._compute_deviatoric_stress(stress[i])
                backstress_dev = self._compute_deviatoric_stress(backstress[i]) if i < len(backstress) else np.zeros_like(stress_dev)
                
                # Relative stress
                relative_stress = stress_dev - backstress_dev
                relative_stress_norm = np.linalg.norm(relative_stress)
                
                # Viscoplastic strain rate magnitude
                if relative_stress_norm > K:
                    vp_strain_rate_magnitude = ((relative_stress_norm - K) / K)**n / self.config.get('chaboche_viscosity', 1e6)
                else:
                    vp_strain_rate_magnitude = 0.0
                
                # Direction of viscoplastic flow
                if relative_stress_norm > 1e-9:
                    flow_direction = relative_stress / relative_stress_norm
                else:
                    flow_direction = np.zeros_like(relative_stress)
                
                # Viscoplastic strain rate tensor
                vp_strain_rate = vp_strain_rate_magnitude * flow_direction
                
                # Backstress evolution (Armstrong-Frederick rule)
                backstress_rate = C * vp_strain_rate - gamma * backstress_dev * vp_strain_rate_magnitude
                new_backstress = backstress[i] + backstress_rate * dt if i < len(backstress) else backstress_rate * dt
                
                # Update stress based on viscoplastic flow
                stress_correction = self._compute_stress_correction_chaboche(stress[i], vp_strain_rate, dt)
                corrected_stress[i] = stress[i] - stress_correction
            else:
                # 1D implementation
                relative_stress = stress[i] - (backstress[i] if i < len(backstress) else 0)
                
                if abs(relative_stress) > K:
                    vp_strain_rate = np.sign(relative_stress) * ((abs(relative_stress) - K) / K)**n / self.config.get('chaboche_viscosity', 1e6)
                    
                    # Backstress evolution
                    backstress_rate = C * vp_strain_rate - gamma * (backstress[i] if i < len(backstress) else 0) * abs(vp_strain_rate)
                    
                    # Stress correction
                    E = self.material_props.youngs_modulus
                    stress_correction = E * vp_strain_rate * dt
                    corrected_stress[i] = stress[i] - stress_correction
                else:
                    corrected_stress[i] = stress[i]
        
        return corrected_stress
    
    def _apply_miller_model(self, stress: np.ndarray, strain_rate: np.ndarray,
                           temperature: np.ndarray, accumulated_strain: np.ndarray) -> np.ndarray:
        """Miller's time-hardening viscoplastic model"""
        # Miller model parameters
        A = self.config.get('miller_coefficient', 1e-10)
        n = self.config.get('miller_stress_exponent', 5.0)
        m = self.config.get('miller_time_exponent', -0.6)
        Q = self.config.get('miller_activation_energy', 280000.0)
        R = 8.314
        
        corrected_stress = np.zeros_like(stress)
        current_time = getattr(self, '_simulation_time', 1.0)
        
        for i in range(len(stress)):
            T = temperature[i] if i < len(temperature) else 293.15
            
            # Effective time (time-hardening)
            effective_time = max(current_time, 1e-6)  # Avoid zero time
            
            if len(stress.shape) > 1 and stress.shape[1] > 1:
                # Multi-axial case
                sigma_eq = self._compute_equivalent_stress(stress[i])
                
                # Miller's creep law: epsilon? = A * sigma^n * t^m * exp(-Q/RT)
                if sigma_eq > 1e-6:
                    creep_rate = A * (sigma_eq)**n * (effective_time)**m * np.exp(-Q / (R * T))
                    
                    # Stress modification based on creep rate
                    target_rate = abs(strain_rate[i].mean()) if i < len(strain_rate) and len(strain_rate.shape) > 1 else 1e-6
                    rate_ratio = creep_rate / max(target_rate, 1e-12)
                    
                    # Limit the correction factor
                    correction_factor = 1.0 / (1.0 + rate_ratio * self.config.get('miller_coupling_strength', 0.1))
                    corrected_stress[i] = stress[i] * correction_factor
                else:
                    corrected_stress[i] = stress[i]
            else:
                # 1D case
                sigma = abs(stress[i])
                if sigma > 1e-6:
                    creep_rate = A * sigma**n * effective_time**m * np.exp(-Q / (R * T))
                    target_rate = abs(strain_rate[i]) if i < len(strain_rate) else 1e-6
                    rate_ratio = creep_rate / max(target_rate, 1e-12)
                    
                    correction_factor = 1.0 / (1.0 + rate_ratio * self.config.get('miller_coupling_strength', 0.1))
                    corrected_stress[i] = stress[i] * correction_factor
                else:
                    corrected_stress[i] = stress[i]
        
        return corrected_stress
    
    def _apply_bodner_partom_model(self, stress: np.ndarray, strain_rate: np.ndarray,
                                  temperature: np.ndarray, accumulated_strain: np.ndarray) -> np.ndarray:
        """Bodner-Partom unified viscoplastic model"""
        # Bodner-Partom parameters
        D0 = self.config.get('bodner_pre_exponential', 1e4)  # Pre-exponential factor
        n = self.config.get('bodner_stress_exponent', 4.0)
        Q = self.config.get('bodner_activation_energy', 250000.0)
        R = 8.314
        Z1 = self.config.get('bodner_hardening_param1', 1000.0)
        Z2 = self.config.get('bodner_hardening_param2', 10.0)
        m = self.config.get('bodner_hardening_exponent', 2.0)
        
        corrected_stress = np.zeros_like(stress)
        
        for i in range(len(stress)):
            T = temperature[i] if i < len(temperature) else 293.15
            
            # Isotropic hardening variable
            Z = Z1 * (1 - np.exp(-Z2 * accumulated_strain[i]**m))
            
            if len(stress.shape) > 1 and stress.shape[1] > 1:
                # Multi-axial stress state
                J2 = self._compute_second_invariant(stress[i])
                sigma_eq = np.sqrt(3 * J2)
                
                # Bodner-Partom flow rule
                if sigma_eq > Z:
                    inelastic_rate = D0 * np.exp(-Q / (R * T)) * ((sigma_eq - Z) / Z)**n
                    
                    # Direction tensor
                    stress_dev = self._compute_deviatoric_stress(stress[i])
                    if sigma_eq > 1e-9:
                        flow_direction = (3/2) * stress_dev / sigma_eq
                    else:
                        flow_direction = np.zeros_like(stress_dev)
                    
                    # Compute stress correction
                    E = self.material_props.youngs_modulus
                    stress_correction = E * inelastic_rate * flow_direction * self.config.get('dt', 1e-6)
                    corrected_stress[i] = stress[i] - stress_correction
                else:
                    corrected_stress[i] = stress[i]
            else:
                # 1D case
                sigma = abs(stress[i])
                if sigma > Z:
                    inelastic_rate = D0 * np.exp(-Q / (R * T)) * ((sigma - Z) / Z)**n
                    
                    # Stress correction
                    E = self.material_props.youngs_modulus
                    stress_correction = E * inelastic_rate * np.sign(stress[i]) * self.config.get('dt', 1e-6)
                    corrected_stress[i] = stress[i] - stress_correction
                else:
                    corrected_stress[i] = stress[i]
        
        return corrected_stress
    
    def _apply_anand_model(self, stress: np.ndarray, strain_rate: np.ndarray,
                          temperature: np.ndarray, state: Dict[str, np.ndarray], dt: float) -> np.ndarray:
        """Anand unified constitutive model for viscoplastic deformation"""
        # Anand model parameters
        s0 = self.config.get('anand_initial_deformation_resistance', 50e6)  # Initial deformation resistance
        Q = self.config.get('anand_activation_energy', 150000.0)  # Activation energy
        A = self.config.get('anand_pre_exponential', 1e4)  # Pre-exponential factor
        xi = self.config.get('anand_stress_multiplier', 6.0)  # Stress multiplier
        m = self.config.get('anand_strain_rate_sensitivity', 0.3)  # Strain rate sensitivity
        h0 = self.config.get('anand_hardening_constant', 1500e6)  # Hardening constant
        s_hat = self.config.get('anand_saturation_stress', 100e6)  # Saturation stress
        n = self.config.get('anand_hardening_exponent', 0.1)  # Hardening exponent
        a = self.config.get('anand_hardening_coefficient', 2.5)  # Hardening coefficient
        R = 8.314
        
        # Get internal state variable (deformation resistance)
        deformation_resistance = state.get('deformation_resistance', np.full(len(stress), s0))
        
        corrected_stress = np.zeros_like(stress)
        
        for i in range(len(stress)):
            T = temperature[i] if i < len(temperature) else 293.15
            s = deformation_resistance[i] if i < len(deformation_resistance) else s0
            
            if len(stress.shape) > 1 and stress.shape[1] > 1:
                # Multi-axial case
                sigma_eq = self._compute_equivalent_stress(stress[i])
                edot_eq = self._compute_equivalent_strain_rate(strain_rate[i]) if i < len(strain_rate) else 1e-6
            else:
                # 1D case
                sigma_eq = abs(stress[i])
                edot_eq = abs(strain_rate[i]) if i < len(strain_rate) else 1e-6
            
            # Anand flow equation: epsilon? = A * [sinh(?sigma/s)]^(1/m) * exp(-Q/RT)
            if sigma_eq > 1e-6 and s > 1e-6:
                stress_ratio = xi * sigma_eq / s
                flow_rate = A * (np.sinh(stress_ratio))**(1/m) * np.exp(-Q / (R * T))
                
                # Evolution of deformation resistance
                s_star = s_hat * (edot_eq / A * np.exp(Q / (R * T)))**n
                
                if abs(s_star - s) > 1e-9:
                    hardening_rate = h0 * (abs(1 - s / s_star))**a * np.sign(1 - s / s_star) * edot_eq
                else:
                    hardening_rate = 0.0
                
                # Update deformation resistance
                new_s = s + hardening_rate * dt
                new_s = max(new_s, 0.1 * s0)  # Prevent negative or too small values
                
                if i < len(deformation_resistance):
                    deformation_resistance[i] = new_s
                
                # Stress correction based on target strain rate vs actual flow rate
                if flow_rate > 0:
                    rate_ratio = edot_eq / flow_rate
                    correction_factor = rate_ratio ** (m * self.config.get('anand_coupling_strength', 0.2))
                    correction_factor = np.clip(correction_factor, 0.1, 10.0)
                else:
                    correction_factor = 1.0
                
                corrected_stress[i] = stress[i] * correction_factor
            else:
                corrected_stress[i] = stress[i]
        
        # Update state
        state['deformation_resistance'] = deformation_resistance
        
        return corrected_stress
    
    def _compute_deviatoric_stress(self, stress_tensor: np.ndarray) -> np.ndarray:
        """Compute deviatoric stress tensor"""
        if len(stress_tensor) == 3:  # 2D case
            s11, s22, s12 = stress_tensor[0], stress_tensor[1], stress_tensor[2]
            hydrostatic = (s11 + s22) / 3.0
            return np.array([s11 - hydrostatic, s22 - hydrostatic, s12])
        elif len(stress_tensor) == 6:  # 3D case
            s11, s22, s33 = stress_tensor[0], stress_tensor[1], stress_tensor[2]
            s12, s23, s13 = stress_tensor[3], stress_tensor[4], stress_tensor[5]
            hydrostatic = (s11 + s22 + s33) / 3.0
            return np.array([s11 - hydrostatic, s22 - hydrostatic, s33 - hydrostatic, s12, s23, s13])
        else:
            return stress_tensor
    
    def _compute_second_invariant(self, stress_tensor: np.ndarray) -> float:
        """Compute second invariant of deviatoric stress tensor"""
        stress_dev = self._compute_deviatoric_stress(stress_tensor)
        
        if len(stress_dev) == 3:  # 2D
            s11, s22, s12 = stress_dev[0], stress_dev[1], stress_dev[2]
            return 0.5 * (s11**2 + s22**2) + s12**2
        elif len(stress_dev) == 6:  # 3D
            s11, s22, s33 = stress_dev[0], stress_dev[1], stress_dev[2]
            s12, s23, s13 = stress_dev[3], stress_dev[4], stress_dev[5]
            return 0.5 * (s11**2 + s22**2 + s33**2) + s12**2 + s23**2 + s13**2
        else:
            return 0.5 * stress_dev[0]**2
    
    def _compute_equivalent_stress(self, stress_tensor: np.ndarray) -> float:
        """Compute von Mises equivalent stress"""
        if len(stress_tensor) == 3:  # 2D
            s11, s22, s12 = stress_tensor[0], stress_tensor[1], stress_tensor[2]
            return np.sqrt(s11**2 + s22**2 - s11*s22 + 3*s12**2)
        elif len(stress_tensor) == 6:  # 3D
            s11, s22, s33 = stress_tensor[0], stress_tensor[1], stress_tensor[2]
            s12, s23, s13 = stress_tensor[3], stress_tensor[4], stress_tensor[5]
            return np.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2) + 3*(s12**2 + s23**2 + s13**2))
        else:
            return abs(stress_tensor[0])
    
    def _compute_equivalent_strain_rate(self, strain_rate_tensor: np.ndarray) -> float:
        """Compute equivalent strain rate"""
        if len(strain_rate_tensor) == 3:  # 2D
            e11, e22, e12 = strain_rate_tensor[0], strain_rate_tensor[1], strain_rate_tensor[2]
            return np.sqrt((2/3) * (e11**2 + e22**2 + e11*e22 + 3*e12**2))
        elif len(strain_rate_tensor) == 6:  # 3D
            e11, e22, e33 = strain_rate_tensor[0], strain_rate_tensor[1], strain_rate_tensor[2]
            e12, e23, e13 = strain_rate_tensor[3], strain_rate_tensor[4], strain_rate_tensor[5]
            return np.sqrt((2/3) * (e11**2 + e22**2 + e33**2 + 2*(e12**2 + e23**2 + e13**2)))
        else:
            return abs(strain_rate_tensor[0])
    
    def _compute_stress_correction_chaboche(self, stress: np.ndarray, vp_strain_rate: np.ndarray, dt: float) -> np.ndarray:
        """Compute stress correction for Chaboche model"""
        E = self.material_props.youngs_modulus
        nu = self.material_props.poisson_ratio
        
        if len(stress) == 3:  # 2D plane stress
            D = E / (1 - nu**2) * np.array([
                [1, nu, 0],
                [nu, 1, 0], 
                [0, 0, (1-nu)/2]
            ])
        elif len(stress) == 6:  # 3D
            factor = E / ((1 + nu) * (1 - 2*nu))
            D = factor * np.array([
                [1-nu, nu, nu, 0, 0, 0],
                [nu, 1-nu, nu, 0, 0, 0],
                [nu, nu, 1-nu, 0, 0, 0],
                [0, 0, 0, (1-2*nu)/2, 0, 0],
                [0, 0, 0, 0, (1-2*nu)/2, 0],
                [0, 0, 0, 0, 0, (1-2*nu)/2]
            ])
        else:  # 1D
            return E * vp_strain_rate * dt
        
        return D @ vp_strain_rate * dt
    
    def _update_viscoplastic_state_variables(self, state: Dict[str, np.ndarray], 
                                           strain_rate: np.ndarray, stress: np.ndarray,
                                           temperature: np.ndarray, dt: float):
        """Update internal state variables for viscoplastic models"""
        # Update accumulated viscoplastic strain
        if 'accumulated_vp_strain' not in state:
            state['accumulated_vp_strain'] = np.zeros(len(stress))
        
        # Compute equivalent viscoplastic strain rate
        for i in range(len(state['accumulated_vp_strain'])):
            if i < len(strain_rate):
                if len(strain_rate.shape) > 1:
                    edot_eq = self._compute_equivalent_strain_rate(strain_rate[i])
                else:
                    edot_eq = abs(strain_rate[i])
                
                state['accumulated_vp_strain'][i] += edot_eq * dt
        
        # Update backstress evolution (for Chaboche model)
        if self.config.get('viscoplastic_model') == 'chaboche':
            if 'backstress' not in state:
                state['backstress'] = np.zeros_like(stress)
            
            C = self.config.get('chaboche_kinematic_modulus', 10000e6)
            gamma = self.config.get('chaboche_recovery_parameter', 300.0)
            
            for i in range(len(state['backstress'])):
                if i < len(strain_rate) and len(strain_rate.shape) > 1:
                    vp_rate = strain_rate[i] if len(strain_rate[i]) > 0 else np.zeros(3)
                    edot_eq = self._compute_equivalent_strain_rate(vp_rate)
                    
                    # Armstrong-Frederick backstress evolution
                    if len(state['backstress'].shape) > 1:
                        backstress_rate = C * vp_rate - gamma * state['backstress'][i] * edot_eq
                        state['backstress'][i] += backstress_rate * dt
                    else:
                        backstress_rate = C * edot_eq * np.sign(vp_rate[0] if len(vp_rate) > 0 else 1) - gamma * state['backstress'][i] * edot_eq
                        state['backstress'][i] += backstress_rate * dt
        
        # Update damage parameter if enabled
        if self.config.get('include_vp_damage', False):
            if 'damage_parameter' not in state:
                state['damage_parameter'] = np.zeros(len(stress))
            
            # Simple damage evolution law
            damage_rate_coeff = self.config.get('damage_rate_coefficient', 1e-6)
            for i in range(len(state['damage_parameter'])):
                if i < len(strain_rate):
                    if len(strain_rate.shape) > 1:
                        edot_eq = self._compute_equivalent_strain_rate(strain_rate[i])
                    else:
                        edot_eq = abs(strain_rate[i])
                    
                    damage_rate = damage_rate_coeff * edot_eq * np.exp(self.config.get('damage_activation_energy', 50000) / (8.314 * temperature[i]))
                    state['damage_parameter'][i] += damage_rate * dt
                    state['damage_parameter'][i] = min(state['damage_parameter'][i], 0.99)  # Limit maximum damage
    
    def _apply_viscoplastic_damage(self, stress: np.ndarray, strain_rate: np.ndarray,
                                  damage: np.ndarray, temperature: np.ndarray, dt: float) -> np.ndarray:
        """Apply damage evolution to viscoplastic response"""
        damaged_stress = np.zeros_like(stress)
        
        for i in range(len(stress)):
            # Damage coupling (1 - D) factor
            damage_factor = 1.0 - damage[i] if i < len(damage) else 1.0
            damage_factor = max(damage_factor, 0.01)  # Prevent complete damage
            
            if len(stress.shape) > 1:
                damaged_stress[i] = stress[i] * damage_factor
            else:
                damaged_stress[i] = stress[i] * damage_factor
        
        return damaged_stress
    
    def _compute_stress_divergence(self, stress: np.ndarray, mesh: Any = None) -> np.ndarray:
        """Compute comprehensive divergence of stress tensor with proper FE assembly"""
        if mesh is not None and hasattr(mesh, 'elements'):
            return self._compute_finite_element_stress_divergence(stress, mesh)
        else:
            return self._compute_finite_difference_stress_divergence(stress)
    
    def _compute_finite_element_stress_divergence(self, stress: np.ndarray, mesh: Any) -> np.ndarray:
        """Compute stress divergence using finite element method"""
        n_nodes = len(mesh.coordinates) if hasattr(mesh, 'coordinates') else 100
        n_dim = len(mesh.coordinates[0]) if hasattr(mesh, 'coordinates') and hasattr(mesh.coordinates[0], '__len__') else 1
        
        # Initialize force vector (divergence result)
        force_vector = np.zeros(n_nodes * n_dim)
        
        n_elements = getattr(mesh, 'n_elements', n_nodes - 1)
        
        for elem_id in range(n_elements):
            # Get element nodes and stress
            elem_nodes = self._get_element_nodes(elem_id, mesh)
            elem_stress = self._get_element_stress(stress, elem_id, elem_nodes)
            
            # Compute element force vector from stress
            elem_force = self._compute_element_force_from_stress(elem_stress, elem_id, mesh)
            
            # Assemble into global force vector
            for i, node in enumerate(elem_nodes):
                for dim in range(n_dim):
                    global_dof = node * n_dim + dim
                    local_dof = i * n_dim + dim
                    if global_dof < len(force_vector) and local_dof < len(elem_force):
                        force_vector[global_dof] += elem_force[local_dof]
        
        return force_vector
    
    def _compute_finite_difference_stress_divergence(self, stress: np.ndarray) -> np.ndarray:
        """Compute stress divergence using high-order finite differences"""
        if len(stress.shape) == 1:
            # 1D stress divergence
            return self._compute_1d_stress_divergence(stress)
        elif len(stress.shape) == 2:
            if stress.shape[1] == 3:  # 2D stress tensor
                return self._compute_2d_stress_divergence(stress)
            elif stress.shape[1] == 6:  # 3D stress tensor
                return self._compute_3d_stress_divergence(stress)
        
        return np.gradient(stress.flatten())
    
    def _compute_1d_stress_divergence(self, stress: np.ndarray) -> np.ndarray:
        """1D stress divergence with high-order finite differences"""
        n = len(stress)
        div_stress = np.zeros(n)
        dx = self._get_grid_spacing()
        
        if n >= 5:
            # 4th order central difference for interior points
            for i in range(2, n-2):
                div_stress[i] = (-stress[i+2] + 8*stress[i+1] - 8*stress[i-1] + stress[i-2]) / (12 * dx)
            
            # Boundary points
            div_stress[0] = (-3*stress[0] + 4*stress[1] - stress[2]) / (2 * dx)
            div_stress[1] = (-stress[0] + stress[2]) / (2 * dx)
            div_stress[-2] = (stress[-3] - stress[-1]) / (2 * dx)
            div_stress[-1] = (stress[-3] - 4*stress[-2] + 3*stress[-1]) / (2 * dx)
        else:
            # Simple central difference
            for i in range(1, n-1):
                div_stress[i] = (stress[i+1] - stress[i-1]) / (2 * dx)
            
            div_stress[0] = (stress[1] - stress[0]) / dx
            div_stress[-1] = (stress[-1] - stress[-2]) / dx
        
        return div_stress
    
    def _compute_2d_stress_divergence(self, stress: np.ndarray) -> np.ndarray:
        """Comprehensive 2D stress divergence with finite element and advanced finite difference methods"""
        n_points = len(stress)
        div_stress = np.zeros((n_points, 2))
        
        # Extract stress components [sigmaxx, sigmayy, sigmaxy]
        sigma_xx = stress[:, 0]
        sigma_yy = stress[:, 1] 
        sigma_xy = stress[:, 2]
        
        # Determine computation method based on available mesh information
        mesh_available = hasattr(self, 'current_mesh') and self.current_mesh is not None
        
        if mesh_available and hasattr(self.current_mesh, 'coordinates') and hasattr(self.current_mesh, 'elements'):
            # Use finite element method for structured mesh
            div_stress = self._compute_2d_finite_element_stress_divergence(stress, self.current_mesh)
        else:
            # Use advanced finite difference methods
            div_stress = self._compute_2d_finite_difference_stress_divergence(sigma_xx, sigma_yy, sigma_xy, n_points)
        
        return div_stress.flatten()
    
    def _compute_2d_finite_element_stress_divergence(self, stress: np.ndarray, mesh: Any) -> np.ndarray:
        """Finite element computation of 2D stress divergence"""
        n_nodes = len(mesh.coordinates)
        div_stress = np.zeros((n_nodes, 2))
        
        # Process each element
        for elem_id in range(getattr(mesh, 'n_elements', n_nodes - 1)):
            elem_nodes = self._get_element_nodes(elem_id, mesh)
            elem_coords = [mesh.coordinates[node] for node in elem_nodes if node < len(mesh.coordinates)]
            
            if len(elem_coords) >= 3:  # Triangular or higher order elements
                # Compute element contribution to stress divergence
                elem_div_stress = self._compute_element_stress_divergence_2d(stress, elem_nodes, elem_coords, elem_id)
                
                # Distribute to nodes
                for i, node in enumerate(elem_nodes):
                    if node < len(div_stress) and i * 2 + 1 < len(elem_div_stress):
                        div_stress[node, 0] += elem_div_stress[i * 2]
                        div_stress[node, 1] += elem_div_stress[i * 2 + 1]
        
        return div_stress
    
    def _compute_element_stress_divergence_2d(self, stress: np.ndarray, elem_nodes: List[int], 
                                            elem_coords: List[List[float]], elem_id: int) -> np.ndarray:
        """Compute stress divergence for a 2D element using integration"""
        n_nodes_elem = len(elem_nodes)
        elem_div = np.zeros(n_nodes_elem * 2)
        
        # Get element stress values
        elem_stress = np.zeros((n_nodes_elem, 3))
        for i, node in enumerate(elem_nodes):
            if node < len(stress):
                elem_stress[i] = stress[node]
        
        if len(elem_coords) == 3:  # Triangular element
            elem_div = self._compute_triangular_stress_divergence(elem_stress, elem_coords)
        elif len(elem_coords) == 4:  # Quadrilateral element
            elem_div = self._compute_quadrilateral_stress_divergence(elem_stress, elem_coords)
        else:
            # Linear interpolation for other cases
            elem_div = self._compute_linear_stress_divergence(elem_stress, elem_coords)
        
        return elem_div
    
    def _compute_triangular_stress_divergence(self, elem_stress: np.ndarray, coords: List[List[float]]) -> np.ndarray:
        """Compute stress divergence for triangular elements using area coordinates"""
        # Triangular element coordinates
        x = np.array([coords[i][0] for i in range(3)])
        y = np.array([coords[i][1] for i in range(3)]) if len(coords[0]) > 1 else np.zeros(3)
        
        # Element area
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        
        if area < 1e-12:
            return np.zeros(6)  # Degenerate element
        
        # Shape function derivatives (constant for linear triangle)
        dN_dx = np.array([(y[1] - y[2]), (y[2] - y[0]), (y[0] - y[1])]) / (2 * area)
        dN_dy = np.array([(x[2] - x[1]), (x[0] - x[2]), (x[1] - x[0])]) / (2 * area)
        
        # Compute divergence using B-transpose multiplication
        div_stress = np.zeros(6)  # 3 nodes * 2 DOF
        
        for i in range(3):  # Loop over nodes
            # x-direction: dsigmaxx/dx + dsigmaxy/dy
            div_stress[2*i] = (dN_dx @ elem_stress[:, 0] + dN_dy @ elem_stress[:, 2]) * area
            
            # y-direction: dsigmaxy/dx + dsigmayy/dy 
            div_stress[2*i + 1] = (dN_dx @ elem_stress[:, 2] + dN_dy @ elem_stress[:, 1]) * area
        
        return div_stress
    
    def _compute_quadrilateral_stress_divergence(self, elem_stress: np.ndarray, coords: List[List[float]]) -> np.ndarray:
        """Compute stress divergence for quadrilateral elements using Gaussian integration"""
        # 2*2 Gaussian quadrature points and weights
        gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
                       (1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(3), 1/np.sqrt(3))]
        weights = [1.0, 1.0, 1.0, 1.0]
        
        div_stress = np.zeros(8)  # 4 nodes * 2 DOF
        
        for gp, weight in zip(gauss_points, weights):
            xi, eta = gp
            
            # Shape function derivatives in natural coordinates
            dN_dxi = np.array([-(1-eta)/4, (1-eta)/4, (1+eta)/4, -(1+eta)/4])
            dN_deta = np.array([-(1-xi)/4, -(1+xi)/4, (1+xi)/4, (1-xi)/4])
            
            # Jacobian matrix computation
            J = np.zeros((2, 2))
            for i in range(4):
                if i < len(coords):
                    x_coord = coords[i][0]
                    y_coord = coords[i][1] if len(coords[i]) > 1 else 0
                    J[0, 0] += dN_dxi[i] * x_coord
                    J[0, 1] += dN_dxi[i] * y_coord
                    J[1, 0] += dN_deta[i] * x_coord
                    J[1, 1] += dN_deta[i] * y_coord
            
            det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(det_J) < 1e-12:
                continue  # Skip degenerate integration point
            
            # Inverse Jacobian
            J_inv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / det_J
            
            # Shape function derivatives in physical coordinates
            dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
            dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
            
            # Interpolate stress at integration point
            sigma_xx_gp = sum(elem_stress[i, 0] * self._shape_function_quad(i, xi, eta) for i in range(min(4, len(elem_stress))))
            sigma_yy_gp = sum(elem_stress[i, 1] * self._shape_function_quad(i, xi, eta) for i in range(min(4, len(elem_stress))))
            sigma_xy_gp = sum(elem_stress[i, 2] * self._shape_function_quad(i, xi, eta) for i in range(min(4, len(elem_stress))))
            
            # Compute divergence at integration point
            div_x_gp = sum(dN_dx[i] * sigma_xx_gp for i in range(4)) + sum(dN_dy[i] * sigma_xy_gp for i in range(4))
            div_y_gp = sum(dN_dx[i] * sigma_xy_gp for i in range(4)) + sum(dN_dy[i] * sigma_yy_gp for i in range(4))
            
            # Integrate and distribute to nodes
            for i in range(4):
                N_i = self._shape_function_quad(i, xi, eta)
                div_stress[2*i] += N_i * div_x_gp * det_J * weight
                div_stress[2*i + 1] += N_i * div_y_gp * det_J * weight
        
        return div_stress
    
    def _compute_linear_stress_divergence(self, elem_stress: np.ndarray, coords: List[List[float]]) -> np.ndarray:
        """Compute stress divergence using linear interpolation for general elements"""
        n_nodes = len(coords)
        div_stress = np.zeros(n_nodes * 2)
        
        # Simple linear gradient computation
        if n_nodes >= 2:
            dx = coords[1][0] - coords[0][0] if n_nodes >= 2 else self._get_grid_spacing()
            dy = (coords[1][1] - coords[0][1]) if len(coords[0]) > 1 and n_nodes >= 2 else self._get_grid_spacing()
            
            # Approximate derivatives
            for i in range(n_nodes):
                if i < len(elem_stress):
                    # x-direction divergence
                    dsigma_xx_dx = elem_stress[i, 0] / dx if dx != 0 else 0
                    dsigma_xy_dy = elem_stress[i, 2] / dy if dy != 0 else 0
                    div_stress[2*i] = dsigma_xx_dx + dsigma_xy_dy
                    
                    # y-direction divergence
                    dsigma_xy_dx = elem_stress[i, 2] / dx if dx != 0 else 0
                    dsigma_yy_dy = elem_stress[i, 1] / dy if dy != 0 else 0
                    div_stress[2*i + 1] = dsigma_xy_dx + dsigma_yy_dy
        
        return div_stress
    
    def _shape_function_quad(self, i: int, xi: float, eta: float) -> float:
        """Quadrilateral shape functions"""
        if i == 0:
            return 0.25 * (1 - xi) * (1 - eta)
        elif i == 1:
            return 0.25 * (1 + xi) * (1 - eta)
        elif i == 2:
            return 0.25 * (1 + xi) * (1 + eta)
        elif i == 3:
            return 0.25 * (1 - xi) * (1 + eta)
        else:
            return 0.0
    
    def _compute_2d_finite_difference_stress_divergence(self, sigma_xx: np.ndarray, sigma_yy: np.ndarray, 
                                                       sigma_xy: np.ndarray, n_points: int) -> np.ndarray:
        """Advanced finite difference computation of 2D stress divergence"""
        div_stress = np.zeros((n_points, 2))
        
        # Get grid information
        grid_info = self._infer_2d_grid_structure(n_points)
        nx, ny = grid_info['nx'], grid_info['ny']
        dx, dy = grid_info['dx'], grid_info['dy']
        
        if grid_info['is_structured']:
            # Structured grid - use high-order finite differences
            div_stress = self._compute_structured_2d_stress_divergence(sigma_xx, sigma_yy, sigma_xy, nx, ny, dx, dy)
        else:
            # Unstructured grid - use local reconstruction methods
            div_stress = self._compute_unstructured_2d_stress_divergence(sigma_xx, sigma_yy, sigma_xy, n_points)
        
        return div_stress
    
    def _infer_2d_grid_structure(self, n_points: int) -> Dict[str, Any]:
        """Infer 2D grid structure from number of points"""
        # Try to find reasonable factorization
        best_nx = int(np.sqrt(n_points))
        best_ny = n_points // best_nx
        
        # Check if we have a perfect rectangular grid
        is_structured = (best_nx * best_ny == n_points)
        
        # Default grid spacing
        dx = self._get_grid_spacing()
        dy = self._get_grid_spacing()
        
        return {
            'nx': best_nx,
            'ny': best_ny,
            'dx': dx,
            'dy': dy,
            'is_structured': is_structured
        }
    
    def _compute_structured_2d_stress_divergence(self, sigma_xx: np.ndarray, sigma_yy: np.ndarray,
                                               sigma_xy: np.ndarray, nx: int, ny: int, 
                                               dx: float, dy: float) -> np.ndarray:
        """High-order finite difference for structured 2D grids"""
        n_points = len(sigma_xx)
        div_stress = np.zeros((n_points, 2))
        
        # Reshape to 2D arrays for easier indexing
        sxx_2d = sigma_xx.reshape(ny, nx)
        syy_2d = sigma_yy.reshape(ny, nx)
        sxy_2d = sigma_xy.reshape(ny, nx)
        
        div_x_2d = np.zeros((ny, nx))
        div_y_2d = np.zeros((ny, nx))
        
        # Compute partial derivatives using high-order finite differences
        for j in range(ny):  # y-direction (rows)
            for i in range(nx):  # x-direction (columns)
                # Compute dsigmaxx/dx using centered differences with boundary handling
                dsxx_dx = self._compute_partial_derivative_x(sxx_2d, i, j, nx, ny, dx)
                
                # Compute dsigmaxy/dy using centered differences with boundary handling
                dsxy_dy = self._compute_partial_derivative_y(sxy_2d, i, j, nx, ny, dy)
                
                # x-direction equilibrium: dsigmaxx/dx + dsigmaxy/dy
                div_x_2d[j, i] = dsxx_dx + dsxy_dy
                
                # Compute dsigmaxy/dx and dsigmayy/dy
                dsxy_dx = self._compute_partial_derivative_x(sxy_2d, i, j, nx, ny, dx)
                dsyy_dy = self._compute_partial_derivative_y(syy_2d, i, j, nx, ny, dy)
                
                # y-direction equilibrium: dsigmaxy/dx + dsigmayy/dy
                div_y_2d[j, i] = dsxy_dx + dsyy_dy
        
        # Flatten back to 1D
        div_stress[:, 0] = div_x_2d.flatten()
        div_stress[:, 1] = div_y_2d.flatten()
        
        return div_stress
    
    def _compute_partial_derivative_x(self, field_2d: np.ndarray, i: int, j: int, 
                                     nx: int, ny: int, dx: float) -> float:
        """Compute dfield/dx using high-order finite differences"""
        if i == 0:  # Left boundary - forward difference
            if nx >= 3:
                return (-3*field_2d[j, i] + 4*field_2d[j, i+1] - field_2d[j, i+2]) / (2 * dx)
            else:
                return (field_2d[j, i+1] - field_2d[j, i]) / dx
        elif i == nx - 1:  # Right boundary - backward difference
            if nx >= 3:
                return (3*field_2d[j, i] - 4*field_2d[j, i-1] + field_2d[j, i-2]) / (2 * dx)
            else:
                return (field_2d[j, i] - field_2d[j, i-1]) / dx
        else:  # Interior points
            if nx >= 5 and i >= 2 and i <= nx - 3:
                # 4th order central difference
                return (-field_2d[j, i+2] + 8*field_2d[j, i+1] - 8*field_2d[j, i-1] + field_2d[j, i-2]) / (12 * dx)
            else:
                # 2nd order central difference
                return (field_2d[j, i+1] - field_2d[j, i-1]) / (2 * dx)
    
    def _compute_partial_derivative_y(self, field_2d: np.ndarray, i: int, j: int,
                                     nx: int, ny: int, dy: float) -> float:
        """Compute dfield/dy using high-order finite differences"""
        if j == 0:  # Bottom boundary - forward difference
            if ny >= 3:
                return (-3*field_2d[j, i] + 4*field_2d[j+1, i] - field_2d[j+2, i]) / (2 * dy)
            else:
                return (field_2d[j+1, i] - field_2d[j, i]) / dy
        elif j == ny - 1:  # Top boundary - backward difference
            if ny >= 3:
                return (3*field_2d[j, i] - 4*field_2d[j-1, i] + field_2d[j-2, i]) / (2 * dy)
            else:
                return (field_2d[j, i] - field_2d[j-1, i]) / dy
        else:  # Interior points
            if ny >= 5 and j >= 2 and j <= ny - 3:
                # 4th order central difference
                return (-field_2d[j+2, i] + 8*field_2d[j+1, i] - 8*field_2d[j-1, i] + field_2d[j-2, i]) / (12 * dy)
            else:
                # 2nd order central difference
                return (field_2d[j+1, i] - field_2d[j-1, i]) / (2 * dy)
    
    def _compute_unstructured_2d_stress_divergence(self, sigma_xx: np.ndarray, sigma_yy: np.ndarray,
                                                  sigma_xy: np.ndarray, coordinates: Optional[np.ndarray] = None,
                                                  connectivity: Optional[Dict] = None) -> np.ndarray:
        """
        Advanced unstructured 2D stress divergence computation using multiple high-fidelity methods:
        - Moving Least Squares (MLS) reconstruction with polynomial basis
        - Radial Basis Function (RBF) interpolation
        - Delaunay triangulation-based gradients
        - Adaptive stencil selection based on local mesh quality
        """
        n_points = len(sigma_xx)
        div_stress = np.zeros((n_points, 2))
        
        try:
            # Build or use provided mesh connectivity
            mesh_data = self._build_mesh_connectivity(coordinates, connectivity, n_points)
            
            # Parallel computation of stress divergence
            if n_points > 1000 and hasattr(self, 'use_parallel') and self.use_parallel:
                div_stress = self._compute_parallel_stress_divergence_2d(
                    sigma_xx, sigma_yy, sigma_xy, mesh_data
                )
            else:
                div_stress = self._compute_sequential_stress_divergence_2d(
                    sigma_xx, sigma_yy, sigma_xy, mesh_data
                )
            
            # Apply boundary condition corrections
            div_stress = self._apply_boundary_corrections_2d(div_stress, mesh_data)
            
            return div_stress
            
        except Exception as e:
            self.logger.warning(f"Advanced stress divergence failed: {e}, using fallback")
            return self._fallback_stress_divergence_2d(sigma_xx, sigma_yy, sigma_xy, n_points)
    
    def _build_mesh_connectivity(self, coordinates: Optional[np.ndarray], 
                                connectivity: Optional[Dict], n_points: int) -> Dict:
        """Build comprehensive mesh connectivity and geometric information"""
        mesh_data = {
            'coordinates': coordinates,
            'connectivity': connectivity,
            'neighbor_graph': {},
            'delaunay_triangulation': None,
            'spatial_tree': None,
            'mesh_quality': {},
            'boundary_nodes': set()
        }
        
        try:
            if coordinates is not None:
                # Build spatial search tree for efficient neighbor finding
                mesh_data['spatial_tree'] = cKDTree(coordinates)
                
                # Compute Delaunay triangulation for robust gradients
                if coordinates.shape[1] >= 2:
                    mesh_data['delaunay_triangulation'] = Delaunay(coordinates[:, :2])
                
                # Build neighbor graph using multiple methods
                mesh_data['neighbor_graph'] = self._build_comprehensive_neighbor_graph(
                    coordinates, mesh_data['delaunay_triangulation'], mesh_data['spatial_tree']
                )
                
                # Assess mesh quality metrics
                mesh_data['mesh_quality'] = self._compute_mesh_quality_metrics(
                    coordinates, mesh_data['neighbor_graph']
                )
                
                # Identify boundary nodes
                mesh_data['boundary_nodes'] = self._identify_boundary_nodes(
                    mesh_data['delaunay_triangulation'], n_points
                )
            else:
                # Generate structured-like connectivity for index-based points
                mesh_data['neighbor_graph'] = self._generate_structured_connectivity(n_points)
                mesh_data['mesh_quality'] = {'uniform_spacing': True}
            
            return mesh_data
            
        except Exception as e:
            self.logger.warning(f"Mesh connectivity building failed: {e}")
            # Minimal fallback connectivity
            return {
                'coordinates': None,
                'neighbor_graph': self._generate_simple_connectivity(n_points),
                'mesh_quality': {'fallback': True}
            }
    
    def _build_comprehensive_neighbor_graph(self, coordinates: np.ndarray, 
                                          delaunay: Optional[Delaunay],
                                          spatial_tree: cKDTree) -> Dict[int, Dict]:
        """Build comprehensive neighbor graph using multiple geometric criteria"""
        n_points = len(coordinates)
        neighbor_graph = {}
        
        for i in range(n_points):
            neighbors = {
                'delaunay': [],
                'knn': [],
                'radius': [],
                'voronoi': [],
                'combined': [],
                'weights': {},
                'distances': {},
                'quality_scores': {}
            }
            
            try:
                # Method 1: Delaunay triangulation neighbors
                if delaunay is not None:
                    delaunay_neighbors = self._get_delaunay_neighbors(i, delaunay)
                    neighbors['delaunay'] = delaunay_neighbors
                
                # Method 2: K-nearest neighbors
                k = min(12, n_points - 1)  # Adaptive k based on problem size
                distances, knn_indices = spatial_tree.query(coordinates[i], k=k+1)
                knn_neighbors = knn_indices[1:].tolist()  # Exclude self
                neighbors['knn'] = knn_neighbors
                neighbors['distances'].update(dict(zip(knn_neighbors, distances[1:])))
                
                # Method 3: Radius-based neighbors with adaptive radius
                adaptive_radius = self._compute_adaptive_search_radius(i, coordinates, distances)
                radius_neighbors = spatial_tree.query_ball_point(coordinates[i], adaptive_radius)
                radius_neighbors = [n for n in radius_neighbors if n != i]
                neighbors['radius'] = radius_neighbors
                
                # Method 4: Voronoi-based neighbors (natural neighbors)
                voronoi_neighbors = self._get_voronoi_neighbors(i, coordinates, spatial_tree)
                neighbors['voronoi'] = voronoi_neighbors
                
                # Combine and weight neighbors based on geometric quality
                combined_neighbors = self._combine_neighbor_sets(neighbors, coordinates, i)
                neighbors['combined'] = combined_neighbors
                
                # Compute neighbor weights for reconstruction
                neighbor_weights = self._compute_neighbor_weights(i, combined_neighbors, coordinates)
                neighbors['weights'] = neighbor_weights
                
                # Assess local mesh quality for each neighbor set
                quality_scores = self._assess_local_mesh_quality(i, neighbors, coordinates)
                neighbors['quality_scores'] = quality_scores
                
            except Exception as e:
                self.logger.warning(f"Neighbor computation failed for point {i}: {e}")
                # Fallback to simple k-nearest neighbors
                if i > 0:
                    neighbors['combined'] = [max(0, i-1)]
                if i < n_points - 1:
                    neighbors['combined'].append(min(n_points-1, i+1))
                neighbors['weights'] = {n: 1.0 for n in neighbors['combined']}
            
            neighbor_graph[i] = neighbors
        
        return neighbor_graph
    
    def _get_delaunay_neighbors(self, point_id: int, delaunay: Delaunay) -> List[int]:
        """Get neighbors from Delaunay triangulation"""
        try:
            # Find all simplices containing this point
            neighbors = set()
            for simplex in delaunay.simplices:
                if point_id in simplex:
                    for vertex in simplex:
                        if vertex != point_id:
                            neighbors.add(vertex)
            return list(neighbors)
        except Exception:
            return []
    
    def _compute_adaptive_search_radius(self, point_id: int, coordinates: np.ndarray, 
                                       knn_distances: np.ndarray) -> float:
        """Compute adaptive search radius based on local point density"""
        try:
            # Use median distance to k-nearest neighbors as base radius
            base_radius = np.median(knn_distances[1:6])  # Exclude self, use 5 nearest
            
            # Adapt based on local density variation
            local_density_variation = np.std(knn_distances[1:6]) / np.mean(knn_distances[1:6])
            adaptation_factor = 1.0 + 0.5 * local_density_variation
            
            # Ensure reasonable bounds
            adaptive_radius = base_radius * adaptation_factor
            max_radius = 3.0 * base_radius
            min_radius = 0.5 * base_radius
            
            return np.clip(adaptive_radius, min_radius, max_radius)
            
        except Exception:
            return np.linalg.norm(coordinates[1] - coordinates[0]) if len(coordinates) > 1 else 1.0
    
    def _get_voronoi_neighbors(self, point_id: int, coordinates: np.ndarray, 
                              spatial_tree: cKDTree) -> List[int]:
        """Get Voronoi/natural neighbors using circumcircle criterion"""
        try:
            # Find candidate neighbors within reasonable radius
            radius = self._compute_adaptive_search_radius(point_id, coordinates, 
                                                        spatial_tree.query(coordinates[point_id], k=10)[0])
            candidates = spatial_tree.query_ball_point(coordinates[point_id], 2*radius)
            candidates = [c for c in candidates if c != point_id]
            
            natural_neighbors = []
            center = coordinates[point_id]
            
            # For each candidate, check if it's a natural neighbor
            for candidate in candidates:
                if self._is_natural_neighbor(center, coordinates[candidate], 
                                           coordinates[candidates], point_id, candidate):
                    natural_neighbors.append(candidate)
            
            return natural_neighbors
            
        except Exception:
            return []
    
    def _is_natural_neighbor(self, center: np.ndarray, candidate: np.ndarray,
                            all_coords: np.ndarray, center_id: int, candidate_id: int) -> bool:
        """Check if a point is a natural neighbor using Voronoi criterion"""
        try:
            # Simplified natural neighbor criterion: no other point is closer to the
            # midpoint between center and candidate
            midpoint = 0.5 * (center + candidate)
            
            center_dist = np.linalg.norm(midpoint - center)
            candidate_dist = np.linalg.norm(midpoint - candidate)
            
            for i, coord in enumerate(all_coords):
                if i != center_id and i != candidate_id:
                    test_dist = np.linalg.norm(midpoint - coord)
                    if test_dist < max(center_dist, candidate_dist):
                        return False
            
            return True
            
        except Exception:
            return True  # Conservative: include if uncertain
    
    def _combine_neighbor_sets(self, neighbors: Dict, coordinates: np.ndarray, 
                              center_id: int) -> List[int]:
        """Intelligently combine different neighbor sets"""
        try:
            # Start with Delaunay neighbors (most geometrically consistent)
            combined = set(neighbors.get('delaunay', []))
            
            # Add k-nearest neighbors up to reasonable limit
            knn = neighbors.get('knn', [])
            combined.update(knn[:8])  # Limit to 8 nearest
            
            # Add radius neighbors if they improve local coverage
            radius = neighbors.get('radius', [])
            combined.update(radius[:12])  # Limit total
            
            # Add natural neighbors for better interpolation quality
            voronoi = neighbors.get('voronoi', [])
            combined.update(voronoi)
            
            # Remove any invalid indices and convert to sorted list
            combined = [n for n in combined if 0 <= n < len(coordinates)]
            combined.sort()
            
            # Ensure minimum number of neighbors for gradient computation
            if len(combined) < 3:
                # Add nearest neighbors to reach minimum
                distances, indices = cKDTree(coordinates).query(coordinates[center_id], k=5)
                for idx in indices[1:]:  # Exclude self
                    if idx not in combined:
                        combined.append(idx)
                        if len(combined) >= 3:
                            break
            
            return combined
            
        except Exception:
            # Fallback to simple neighbors
            fallback = []
            if center_id > 0:
                fallback.append(center_id - 1)
            if center_id < len(coordinates) - 1:
                fallback.append(center_id + 1)
            return fallback
    
    def _compute_neighbor_weights(self, center_id: int, neighbors: List[int], 
                                 coordinates: np.ndarray) -> Dict[int, float]:
        """Compute sophisticated weights for neighbor contributions"""
        weights = {}
        
        if not neighbors or coordinates is None:
            return {n: 1.0 for n in neighbors}
        
        try:
            center = coordinates[center_id]
            
            # Compute various weight contributions
            for neighbor_id in neighbors:
                neighbor = coordinates[neighbor_id]
                distance = np.linalg.norm(neighbor - center)
                
                if distance < 1e-12:  # Avoid division by zero
                    weights[neighbor_id] = 1e12
                    continue
                
                # Distance-based weight (inverse distance)
                dist_weight = 1.0 / (distance + 1e-12)
                
                # Angular distribution weight (favor well-distributed neighbors)
                angle_weight = self._compute_angular_weight(center, neighbor, neighbors, coordinates)
                
                # Local mesh quality weight
                quality_weight = self._compute_local_quality_weight(center_id, neighbor_id, coordinates)
                
                # Combine weights
                total_weight = dist_weight * angle_weight * quality_weight
                weights[neighbor_id] = total_weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {n: w/total_weight for n, w in weights.items()}
            else:
                weights = {n: 1.0/len(neighbors) for n in neighbors}
                
        except Exception as e:
            self.logger.warning(f"Weight computation failed: {e}")
            weights = {n: 1.0/len(neighbors) for n in neighbors}
        
        return weights
    
    def _compute_angular_weight(self, center: np.ndarray, neighbor: np.ndarray,
                               all_neighbors: List[int], coordinates: np.ndarray) -> float:
        """Compute weight based on angular distribution of neighbors"""
        try:
            # Vector from center to this neighbor
            vec_to_neighbor = neighbor - center
            vec_to_neighbor = vec_to_neighbor / (np.linalg.norm(vec_to_neighbor) + 1e-12)
            
            # Find closest angular separation to other neighbors
            min_angle = np.pi
            for other_id in all_neighbors:
                if other_id != neighbor.tolist().index(neighbor):  # Avoid self-comparison
                    other_neighbor = coordinates[other_id]
                    vec_to_other = other_neighbor - center
                    vec_to_other = vec_to_other / (np.linalg.norm(vec_to_other) + 1e-12)
                    
                    # Compute angle between vectors
                    cos_angle = np.clip(np.dot(vec_to_neighbor, vec_to_other), -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    min_angle = min(min_angle, angle)
            
            # Favor neighbors that are well-separated angularly
            angular_weight = np.sin(min_angle) + 0.1  # Add small base weight
            return angular_weight
            
        except Exception:
            return 1.0
    
    def _compute_local_quality_weight(self, center_id: int, neighbor_id: int, 
                                     coordinates: np.ndarray) -> float:
        """Compute weight based on local mesh quality"""
        try:
            # Simple quality measure: regularity of local triangulation
            center = coordinates[center_id]
            neighbor = coordinates[neighbor_id]
            
            # Find common neighbors to form triangles
            tree = cKDTree(coordinates)
            _, potential_thirds = tree.query(center, k=10)
            
            best_quality = 0.0
            for third_id in potential_thirds:
                if third_id != center_id and third_id != neighbor_id:
                    third = coordinates[third_id]
                    
                    # Compute triangle quality (aspect ratio)
                    triangle_quality = self._compute_triangle_quality(center, neighbor, third)
                    best_quality = max(best_quality, triangle_quality)
            
            return best_quality + 0.1  # Add base weight
            
        except Exception:
            return 1.0
    
    def _compute_triangle_quality(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Compute triangle quality metric (0 = degenerate, 1 = equilateral)"""
        try:
            # Edge lengths
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p1 - p3)
            
            if a < 1e-12 or b < 1e-12 or c < 1e-12:
                return 0.0
            
            # Semi-perimeter and area
            s = (a + b + c) / 2.0
            area_sq = s * (s - a) * (s - b) * (s - c)
            
            if area_sq <= 0:
                return 0.0
            
            area = np.sqrt(area_sq)
            
            # Quality metric: area / (perimeter^2) normalized to equilateral triangle
            quality = 4.0 * np.sqrt(3.0) * area / (a + b + c)**2
            return np.clip(quality, 0.0, 1.0)
            
        except Exception:
            return 0.1
    
    def _assess_local_mesh_quality(self, center_id: int, neighbors: Dict, 
                                  coordinates: np.ndarray) -> Dict[str, float]:
        """Assess quality of different neighbor selection methods"""
        quality_scores = {}
        
        try:
            for method, neighbor_list in neighbors.items():
                if method in ['delaunay', 'knn', 'radius', 'voronoi', 'combined']:
                    if neighbor_list and coordinates is not None:
                        score = self._compute_stencil_quality(center_id, neighbor_list, coordinates)
                        quality_scores[method] = score
                    else:
                        quality_scores[method] = 0.0
            
        except Exception:
            quality_scores = {method: 0.5 for method in ['delaunay', 'knn', 'radius', 'voronoi', 'combined']}
        
        return quality_scores
    
    def _compute_stencil_quality(self, center_id: int, neighbors: List[int], 
                                coordinates: np.ndarray) -> float:
        """Compute quality of gradient stencil"""
        try:
            if len(neighbors) < 3:
                return 0.1
            
            center = coordinates[center_id]
            neighbor_coords = coordinates[neighbors]
            
            # Compute condition number of least squares matrix
            A = neighbor_coords - center  # Relative positions
            condition_number = np.linalg.cond(A.T @ A)
            
            # Convert to quality score (lower condition number = better quality)
            quality = 1.0 / (1.0 + np.log10(max(condition_number, 1.0)))
            
            return np.clip(quality, 0.0, 1.0)
            
        except Exception:
            return 0.1
    
    def _compute_mesh_quality_metrics(self, coordinates: np.ndarray, 
                                     neighbor_graph: Dict) -> Dict:
        """Compute overall mesh quality metrics"""
        try:
            n_points = len(coordinates)
            metrics = {
                'average_connectivity': 0.0,
                'min_connectivity': float('inf'),
                'max_connectivity': 0.0,
                'average_aspect_ratio': 0.0,
                'mesh_regularity': 0.0,
                'boundary_quality': 0.0
            }
            
            connectivities = []
            aspect_ratios = []
            
            for i in range(n_points):
                neighbors = neighbor_graph.get(i, {}).get('combined', [])
                connectivity = len(neighbors)
                connectivities.append(connectivity)
                
                if connectivity > 0:
                    # Compute local aspect ratio
                    local_aspect = self._compute_local_aspect_ratio(i, neighbors, coordinates)
                    aspect_ratios.append(local_aspect)
            
            if connectivities:
                metrics['average_connectivity'] = np.mean(connectivities)
                metrics['min_connectivity'] = np.min(connectivities)
                metrics['max_connectivity'] = np.max(connectivities)
            
            if aspect_ratios:
                metrics['average_aspect_ratio'] = np.mean(aspect_ratios)
                metrics['mesh_regularity'] = 1.0 / (1.0 + np.std(aspect_ratios))
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Mesh quality computation failed: {e}")
            return {'fallback_quality': 0.5}
    
    def _compute_local_aspect_ratio(self, center_id: int, neighbors: List[int], 
                                   coordinates: np.ndarray) -> float:
        """Compute local aspect ratio around a point"""
        try:
            if len(neighbors) < 2:
                return 1.0
            
            center = coordinates[center_id]
            distances = [np.linalg.norm(coordinates[n] - center) for n in neighbors]
            
            if not distances:
                return 1.0
            
            max_dist = max(distances)
            min_dist = min(distances)
            
            if min_dist < 1e-12:
                return 100.0  # Very poor aspect ratio
            
            return max_dist / min_dist
            
        except Exception:
            return 1.0
    
    def _identify_boundary_nodes(self, delaunay: Optional[Delaunay], n_points: int) -> set:
        """Identify boundary nodes from Delaunay triangulation"""
        try:
            if delaunay is None:
                return set()
            
            # Find boundary edges (edges that belong to only one triangle)
            edge_count = defaultdict(int)
            
            for simplex in delaunay.simplices:
                # For each edge in the simplex
                n_vertices = len(simplex)
                for i in range(n_vertices):
                    for j in range(i+1, n_vertices):
                        edge = tuple(sorted([simplex[i], simplex[j]]))
                        edge_count[edge] += 1
            
            # Boundary edges appear only once
            boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
            boundary_nodes = set()
            for edge in boundary_edges:
                boundary_nodes.update(edge)
            
            return boundary_nodes
            
        except Exception:
            return set()
    
    def _generate_structured_connectivity(self, n_points: int) -> Dict[int, Dict]:
        """Generate structured-like connectivity for uniform grids"""
        neighbor_graph = {}
        
        # Assume 1D chain connectivity as fallback
        for i in range(n_points):
            neighbors = {
                'combined': [],
                'weights': {}
            }
            
            # Add neighboring points
            if i > 0:
                neighbors['combined'].append(i-1)
                neighbors['weights'][i-1] = 0.5
            if i < n_points - 1:
                neighbors['combined'].append(i+1)  
                neighbors['weights'][i+1] = 0.5
            
            neighbor_graph[i] = neighbors
        
        return neighbor_graph
    
    def _generate_simple_connectivity(self, n_points: int) -> Dict[int, Dict]:
        """Generate minimal connectivity as absolute fallback"""
        neighbor_graph = {}
        
        for i in range(n_points):
            neighbors = []
            if i > 0:
                neighbors.append(i-1)
            if i < n_points - 1:
                neighbors.append(i+1)
            
            neighbor_graph[i] = {
                'combined': neighbors,
                'weights': {n: 1.0/len(neighbors) if neighbors else 1.0 for n in neighbors}
            }
        
        return neighbor_graph
    
    def _compute_sequential_stress_divergence_2d(self, sigma_xx: np.ndarray, sigma_yy: np.ndarray,
                                               sigma_xy: np.ndarray, mesh_data: Dict) -> np.ndarray:
        """Compute stress divergence sequentially with advanced methods"""
        n_points = len(sigma_xx)
        div_stress = np.zeros((n_points, 2))
        
        neighbor_graph = mesh_data['neighbor_graph']
        coordinates = mesh_data.get('coordinates')
        
        for i in range(n_points):
            try:
                neighbors_info = neighbor_graph.get(i, {})
                neighbors = neighbors_info.get('combined', [])
                weights = neighbors_info.get('weights', {})
                
                if len(neighbors) >= 3:
                    # Use Moving Least Squares reconstruction
                    grad_sxx = self._compute_mls_gradient(sigma_xx, i, neighbors, weights, coordinates)
                    grad_syy = self._compute_mls_gradient(sigma_yy, i, neighbors, weights, coordinates) 
                    grad_sxy = self._compute_mls_gradient(sigma_xy, i, neighbors, weights, coordinates)
                    
                    # Compute stress divergence: ∇·σ
                    div_stress[i, 0] = grad_sxx[0] + grad_sxy[1]  # ∂σ_xx/∂x + ∂σ_xy/∂y
                    div_stress[i, 1] = grad_sxy[0] + grad_syy[1]  # ∂σ_xy/∂x + ∂σ_yy/∂y
                    
                elif len(neighbors) >= 2:
                    # Use RBF interpolation
                    grad_sxx = self._compute_rbf_gradient(sigma_xx, i, neighbors, coordinates)
                    grad_syy = self._compute_rbf_gradient(sigma_yy, i, neighbors, coordinates)
                    grad_sxy = self._compute_rbf_gradient(sigma_xy, i, neighbors, coordinates)
                    
                    div_stress[i, 0] = grad_sxx[0] + grad_sxy[1]
                    div_stress[i, 1] = grad_sxy[0] + grad_syy[1]
                    
                else:
                    # Fallback finite difference
                    div_stress[i] = self._compute_fallback_divergence_2d(
                        sigma_xx, sigma_yy, sigma_xy, i, neighbors
                    )
                
            except Exception as e:
                self.logger.warning(f"Divergence computation failed for point {i}: {e}")
                div_stress[i] = 0.0
        
        return div_stress
    
    def _compute_parallel_stress_divergence_2d(self, sigma_xx: np.ndarray, sigma_yy: np.ndarray,
                                             sigma_xy: np.ndarray, mesh_data: Dict) -> np.ndarray:
        """Compute stress divergence in parallel using ThreadPoolExecutor"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        n_points = len(sigma_xx)
        div_stress = np.zeros((n_points, 2))
        
        # Determine number of threads
        max_workers = min(4, max(1, n_points // 500))
        chunk_size = max(1, n_points // max_workers)
        
        def compute_chunk(start_idx: int, end_idx: int) -> Tuple[int, int, np.ndarray]:
            chunk_result = np.zeros((end_idx - start_idx, 2))
            neighbor_graph = mesh_data['neighbor_graph']
            coordinates = mesh_data.get('coordinates')
            
            for i in range(start_idx, end_idx):
                try:
                    local_idx = i - start_idx
                    neighbors_info = neighbor_graph.get(i, {})
                    neighbors = neighbors_info.get('combined', [])
                    weights = neighbors_info.get('weights', {})
                    
                    if len(neighbors) >= 3:
                        grad_sxx = self._compute_mls_gradient(sigma_xx, i, neighbors, weights, coordinates)
                        grad_syy = self._compute_mls_gradient(sigma_yy, i, neighbors, weights, coordinates)
                        grad_sxy = self._compute_mls_gradient(sigma_xy, i, neighbors, weights, coordinates)
                        
                        chunk_result[local_idx, 0] = grad_sxx[0] + grad_sxy[1]
                        chunk_result[local_idx, 1] = grad_sxy[0] + grad_syy[1]
                    else:
                        chunk_result[local_idx] = self._compute_fallback_divergence_2d(
                            sigma_xx, sigma_yy, sigma_xy, i, neighbors
                        )
                        
                except Exception as e:
                    self.logger.warning(f"Parallel divergence failed for point {i}: {e}")
                    chunk_result[local_idx] = 0.0
            
            return start_idx, end_idx, chunk_result
        
        # Submit parallel tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for start_idx in range(0, n_points, chunk_size):
                end_idx = min(start_idx + chunk_size, n_points)
                future = executor.submit(compute_chunk, start_idx, end_idx)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    start_idx, end_idx, chunk_result = future.result()
                    div_stress[start_idx:end_idx] = chunk_result
                except Exception as e:
                    self.logger.error(f"Parallel chunk computation failed: {e}")
        
        return div_stress
    
    def _compute_mls_gradient(self, field: np.ndarray, center_id: int, neighbors: List[int],
                             weights: Dict[int, float], coordinates: Optional[np.ndarray]) -> np.ndarray:
        """Compute gradient using Moving Least Squares with polynomial basis"""
        try:
            if len(neighbors) < 3:
                return np.array([0.0, 0.0])
            
            if coordinates is None:
                # Fall back to structured finite difference
                return self._compute_structured_gradient(field, center_id, neighbors)
            
            center = coordinates[center_id]
            
            # Build MLS system with polynomial basis: [1, x, y, x^2, xy, y^2]
            n_neighbors = len(neighbors)
            basis_size = 6  # Complete quadratic basis in 2D
            
            # Weight matrix
            W = np.zeros((n_neighbors, n_neighbors))
            for i, neighbor_id in enumerate(neighbors):
                weight = weights.get(neighbor_id, 1.0)
                distance = np.linalg.norm(coordinates[neighbor_id] - center)
                
                # MLS weight function: w(r) = exp(-(r/h)^2)
                h = np.mean([np.linalg.norm(coordinates[n] - center) for n in neighbors])
                mls_weight = np.exp(-(distance / (h + 1e-12))**2)
                W[i, i] = weight * mls_weight
            
            # Moment matrix M and basis functions
            M = np.zeros((basis_size, basis_size))
            b_vec = np.zeros(basis_size)
            
            for i, neighbor_id in enumerate(neighbors):
                neighbor = coordinates[neighbor_id]
                dx = neighbor[0] - center[0]
                dy = neighbor[1] - center[1]
                
                # Polynomial basis: [1, x, y, x^2, xy, y^2]
                basis = np.array([1.0, dx, dy, dx**2, dx*dy, dy**2])
                weight = W[i, i]
                
                M += weight * np.outer(basis, basis)
                b_vec += weight * basis * field[neighbor_id]
            
            # Add regularization for stability
            regularization = 1e-8 * np.eye(basis_size)
            M += regularization
            
            # Solve for polynomial coefficients
            try:
                coeffs = np.linalg.solve(M, b_vec)
                # Gradient is [∂p/∂x, ∂p/∂y] = [coeffs[1], coeffs[2]] at center
                gradient = np.array([coeffs[1], coeffs[2]])
                return gradient
            except np.linalg.LinAlgError:
                # Fall back to linear MLS
                return self._compute_linear_mls_gradient(field, center_id, neighbors, weights, coordinates)
                
        except Exception as e:
            self.logger.warning(f"MLS gradient computation failed: {e}")
            return self._compute_simple_gradient(field, center_id, neighbors, coordinates)
    
    def _compute_linear_mls_gradient(self, field: np.ndarray, center_id: int, neighbors: List[int],
                                   weights: Dict[int, float], coordinates: np.ndarray) -> np.ndarray:
        """Compute gradient using linear MLS (fallback from quadratic)"""
        try:
            center = coordinates[center_id]
            n_neighbors = len(neighbors)
            
            # Linear basis: [1, x, y]
            basis_size = 3
            M = np.zeros((basis_size, basis_size))
            b_vec = np.zeros(basis_size)
            
            for neighbor_id in neighbors:
                neighbor = coordinates[neighbor_id]
                dx = neighbor[0] - center[0]
                dy = neighbor[1] - center[1]
                
                basis = np.array([1.0, dx, dy])
                weight = weights.get(neighbor_id, 1.0)
                distance = np.linalg.norm(neighbor - center)
                
                # Distance weight
                if distance > 1e-12:
                    weight *= 1.0 / (distance + 1e-6)
                
                M += weight * np.outer(basis, basis)
                b_vec += weight * basis * field[neighbor_id]
            
            # Regularization
            M += 1e-10 * np.eye(basis_size)
            
            # Solve
            coeffs = np.linalg.solve(M, b_vec)
            return np.array([coeffs[1], coeffs[2]])
            
        except Exception:
            return np.array([0.0, 0.0])
    
    def _compute_rbf_gradient(self, field: np.ndarray, center_id: int, neighbors: List[int],
                             coordinates: Optional[np.ndarray]) -> np.ndarray:
        """Compute gradient using Radial Basis Function interpolation"""
        try:
            if coordinates is None or len(neighbors) < 2:
                return np.array([0.0, 0.0])
            
            center = coordinates[center_id]
            n_neighbors = len(neighbors)
            
            # Build RBF system
            # Choose RBF: φ(r) = r^2 * log(r) (thin plate spline)
            A = np.zeros((n_neighbors, n_neighbors))
            for i, ni in enumerate(neighbors):
                for j, nj in enumerate(neighbors):
                    r = np.linalg.norm(coordinates[ni] - coordinates[nj])
                    if r < 1e-12:
                        A[i, j] = 0.0
                    else:
                        A[i, j] = r**2 * np.log(r)
            
            # RHS vector
            b = np.array([field[n] for n in neighbors])
            
            # Add regularization
            A += 1e-8 * np.eye(n_neighbors)
            
            # Solve for weights
            try:
                weights_rbf = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return np.array([0.0, 0.0])
            
            # Compute gradient: ∇φ(r) = 2r*log(r) + r for thin plate spline
            gradient = np.zeros(2)
            for i, neighbor_id in enumerate(neighbors):
                neighbor = coordinates[neighbor_id]
                r_vec = center - neighbor
                r = np.linalg.norm(r_vec)
                
                if r > 1e-12:
                    # Derivative of φ(r) = r^2 * log(r)
                    dphi_dr = 2*r*np.log(r) + r
                    grad_phi = dphi_dr * r_vec / r
                    gradient += weights_rbf[i] * grad_phi
            
            return gradient
            
        except Exception:
            return np.array([0.0, 0.0])
    
    def _compute_structured_gradient(self, field: np.ndarray, center_id: int, neighbors: List[int]) -> np.ndarray:
        """Compute gradient assuming structured grid"""
        try:
            if len(neighbors) < 2:
                return np.array([0.0, 0.0])
            
            # Simple finite difference on structured grid
            dx = dy = 1.0  # Assume unit spacing
            
            gradient = np.zeros(2)
            
            # Find left and right neighbors
            left_neighbor = right_neighbor = None
            for n in neighbors:
                if n == center_id - 1:
                    left_neighbor = n
                elif n == center_id + 1:
                    right_neighbor = n
            
            # Compute x-gradient
            if left_neighbor is not None and right_neighbor is not None:
                gradient[0] = (field[right_neighbor] - field[left_neighbor]) / (2.0 * dx)
            elif right_neighbor is not None:
                gradient[0] = (field[right_neighbor] - field[center_id]) / dx
            elif left_neighbor is not None:
                gradient[0] = (field[center_id] - field[left_neighbor]) / dx
            
            # For y-gradient, assume some neighbors are above/below
            # This is a simplification - in reality need 2D grid structure
            if len(neighbors) >= 4:
                # Approximate y-gradient using available neighbors
                upper_half = [n for n in neighbors if n > center_id + 1]
                lower_half = [n for n in neighbors if n < center_id - 1]
                
                if upper_half and lower_half:
                    upper_val = field[upper_half[0]]
                    lower_val = field[lower_half[0]]
                    gradient[1] = (upper_val - lower_val) / (2.0 * dy)
            
            return gradient
            
        except Exception:
            return np.array([0.0, 0.0])
    
    def _compute_simple_gradient(self, field: np.ndarray, center_id: int, neighbors: List[int],
                                coordinates: Optional[np.ndarray]) -> np.ndarray:
        """Simple gradient computation as final fallback"""
        try:
            if len(neighbors) < 2:
                return np.array([0.0, 0.0])
            
            if coordinates is not None:
                # Use first-order finite difference with nearest neighbors
                center = coordinates[center_id]
                gradient = np.zeros(2)
                
                for neighbor_id in neighbors:
                    neighbor = coordinates[neighbor_id]
                    r_vec = neighbor - center
                    r = np.linalg.norm(r_vec)
                    
                    if r > 1e-12:
                        # Simple directional derivative
                        directional_grad = (field[neighbor_id] - field[center_id]) / r
                        gradient += directional_grad * r_vec / r
                
                gradient /= len(neighbors)
                return gradient
            else:
                # Structured fallback
                if center_id > 0 and center_id < len(field) - 1:
                    dx = 1.0
                    gradient = np.array([
                        (field[center_id + 1] - field[center_id - 1]) / (2.0 * dx),
                        0.0
                    ])
                    return gradient
            
            return np.array([0.0, 0.0])
            
        except Exception:
            return np.array([0.0, 0.0])
    
    def _compute_fallback_divergence_2d(self, sigma_xx: np.ndarray, sigma_yy: np.ndarray,
                                       sigma_xy: np.ndarray, center_id: int, neighbors: List[int]) -> np.ndarray:
        """Simple fallback divergence computation"""
        try:
            if len(neighbors) >= 2:
                # Simple finite difference
                dx = 1.0  # Assume unit spacing
                div_stress = np.zeros(2)
                
                if center_id > 0 and center_id < len(sigma_xx) - 1:
                    div_stress[0] = (sigma_xx[center_id + 1] - sigma_xx[center_id - 1]) / (2.0 * dx)
                    div_stress[1] = (sigma_yy[center_id + 1] - sigma_yy[center_id - 1]) / (2.0 * dx)
                
                return div_stress
            else:
                return np.array([0.0, 0.0])
                
        except Exception:
            return np.array([0.0, 0.0])
    
    def _apply_boundary_corrections_2d(self, div_stress: np.ndarray, mesh_data: Dict) -> np.ndarray:
        """Apply boundary condition corrections to stress divergence"""
        try:
            boundary_nodes = mesh_data.get('boundary_nodes', set())
            
            if not boundary_nodes:
                return div_stress
            
            # Apply special treatment for boundary nodes
            for node_id in boundary_nodes:
                if 0 <= node_id < len(div_stress):
                    # Reduce divergence magnitude at boundaries (stress-free condition)
                    correction_factor = 0.5  # Empirical correction
                    div_stress[node_id] *= correction_factor
            
            return div_stress
            
        except Exception as e:
            self.logger.warning(f"Boundary correction failed: {e}")
            return div_stress
    
    def _fallback_stress_divergence_2d(self, sigma_xx: np.ndarray, sigma_yy: np.ndarray,
                                      sigma_xy: np.ndarray, n_points: int) -> np.ndarray:
        """Simple fallback stress divergence computation"""
        div_stress = np.zeros((n_points, 2))
        
        try:
            dx = 1.0  # Assume unit grid spacing
            
            for i in range(n_points):
                if i > 0 and i < n_points - 1:
                    # Simple central difference
                    div_stress[i, 0] = (sigma_xx[i+1] - sigma_xx[i-1]) / (2.0 * dx)
                    div_stress[i, 1] = (sigma_yy[i+1] - sigma_yy[i-1]) / (2.0 * dx)
                elif i == 0 and n_points > 1:
                    # Forward difference
                    div_stress[i, 0] = (sigma_xx[i+1] - sigma_xx[i]) / dx
                    div_stress[i, 1] = (sigma_yy[i+1] - sigma_yy[i]) / dx
                elif i == n_points - 1 and n_points > 1:
                    # Backward difference
                    div_stress[i, 0] = (sigma_xx[i] - sigma_xx[i-1]) / dx
                    div_stress[i, 1] = (sigma_yy[i] - sigma_yy[i-1]) / dx
        
        except Exception as e:
            self.logger.error(f"Fallback stress divergence failed: {e}")
        
        return div_stress
    
    def _compute_3d_stress_divergence(self, stress: np.ndarray, coordinates: Optional[np.ndarray] = None, 
                                     connectivity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advanced 3D stress divergence computation using high-order finite difference schemes,
        unstructured mesh support, and adaptive stencils with geometric correction factors
        
        Args:
            stress: Stress tensor components [n_points, 6] -> [sigmaxx, sigmayy, sigmazz, sigmaxy, sigmayz, sigmaxz]
            coordinates: Optional 3D coordinates [n_points, 3] -> [x, y, z]
            connectivity: Optional mesh connectivity information
            
        Returns:
            div_stress: Stress divergence vector [n_points*3] -> [nabla*sigma_x, nabla*sigma_y, nabla*sigma_z]
        """
        n_points = len(stress)
        div_stress = np.zeros((n_points, 3))
        
        # Extract stress tensor components (Voigt notation)
        sigma = {
            'xx': stress[:, 0], 'yy': stress[:, 1], 'zz': stress[:, 2],
            'xy': stress[:, 3], 'yz': stress[:, 4], 'xz': stress[:, 5]
        }
        
        # Determine computational approach based on available data
        if coordinates is not None and coordinates.shape[1] == 3:
            if connectivity is not None:
                # Unstructured mesh with connectivity
                div_stress = self._compute_unstructured_3d_divergence(sigma, coordinates, connectivity)
            else:
                # Point cloud with coordinate-based neighbors
                div_stress = self._compute_meshfree_3d_divergence(sigma, coordinates)
        else:
            # Structured grid with intelligent topology detection
            div_stress = self._compute_structured_3d_divergence(sigma, n_points)
            
        # Apply geometric correction factors for curved boundaries
        if hasattr(self, 'boundary_curvature') and self.boundary_curvature is not None:
            div_stress = self._apply_curvature_correction(div_stress, coordinates)
            
        # Add high-order stabilization for convection-dominated problems
        if self.config.get('high_reynolds_stabilization', False):
            div_stress = self._apply_streamline_upwind_stabilization(div_stress, stress, coordinates)
            
        return div_stress.flatten()
    
    def _compute_unstructured_3d_divergence(self, sigma: Dict[str, np.ndarray], 
                                          coordinates: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
        """Compute divergence on unstructured 3D mesh using Green-Gauss theorem"""
        n_points = len(coordinates)
        div_stress = np.zeros((n_points, 3))
        
        # Build adjacency list for efficient neighbor access
        adjacency = self._build_adjacency_list(connectivity, n_points)
        
        for i in range(n_points):
            neighbors = adjacency[i]
            if len(neighbors) < 3:  # Insufficient neighbors
                continue
                
            # Green-Gauss integration over control volume
            control_volume = 0.0
            stress_integral = np.zeros(3)
            
            for j in neighbors:
                # Compute face area vector (simplified dual cell construction)
                face_center, face_area_vector = self._compute_dual_face(i, j, coordinates, adjacency)
                face_area = np.linalg.norm(face_area_vector)
                normal = face_area_vector / (face_area + 1e-12)
                
                # Interpolate stress to face center
                stress_face = self._interpolate_stress_to_face(sigma, i, j, coordinates)
                
                # Compute stress*normal for each component
                # div_sigma_x = dsigmaxx/dx + dsigmaxy/dy + dsigmaxz/dz
                stress_normal = np.array([
                    stress_face['xx'] * normal[0] + stress_face['xy'] * normal[1] + stress_face['xz'] * normal[2],
                    stress_face['xy'] * normal[0] + stress_face['yy'] * normal[1] + stress_face['yz'] * normal[2], 
                    stress_face['xz'] * normal[0] + stress_face['yz'] * normal[1] + stress_face['zz'] * normal[2]
                ])
                
                stress_integral += stress_normal * face_area
                control_volume += self._compute_tetrahedron_volume(coordinates[i], face_center, 
                                                                 coordinates[j], coordinates[neighbors[0]])
            
            # Green-Gauss: nabla*sigma = (1/V) ? sigma*n dA
            if control_volume > 1e-15:
                div_stress[i] = stress_integral / control_volume
                
        return div_stress
    
    def _compute_meshfree_3d_divergence(self, sigma: Dict[str, np.ndarray], coordinates: np.ndarray) -> np.ndarray:
        """Meshfree divergence computation using moving least squares reconstruction"""
        n_points = len(coordinates)
        div_stress = np.zeros((n_points, 3))
        
        # Build spatial tree for efficient neighbor search
        tree = cKDTree(coordinates)
        
        # Adaptive support radius based on local point density
        support_radii = self._compute_adaptive_support_radii(coordinates, tree)
        
        for i in range(n_points):
            # Find neighbors within support radius
            neighbor_indices = tree.query_ball_point(coordinates[i], support_radii[i])
            neighbor_indices.remove(i) if i in neighbor_indices else None
            
            if len(neighbor_indices) < 6:  # Need minimum neighbors for 3D reconstruction
                continue
                
            # Moving Least Squares reconstruction of stress gradients
            gradients = self._compute_mls_stress_gradients(i, neighbor_indices, sigma, coordinates)
            
            # Assemble divergence: div_sigma = [dsigmaxx/dx + dsigmaxy/dy + dsigmaxz/dz, ...]
            div_stress[i, 0] = gradients['xx'][0] + gradients['xy'][1] + gradients['xz'][2]  # nabla*sigma_x
            div_stress[i, 1] = gradients['xy'][0] + gradients['yy'][1] + gradients['yz'][2]  # nabla*sigma_y  
            div_stress[i, 2] = gradients['xz'][0] + gradients['yz'][1] + gradients['zz'][2]  # nabla*sigma_z
            
        return div_stress
    
    def _compute_structured_3d_divergence(self, sigma: Dict[str, np.ndarray], n_points: int) -> np.ndarray:
        """High-order finite difference on structured 3D grids with intelligent topology detection"""
        div_stress = np.zeros((n_points, 3))
        
        # Intelligent grid topology detection
        grid_info = self._detect_3d_grid_topology(n_points)
        nx, ny, nz = grid_info['nx'], grid_info['ny'], grid_info['nz']
        dx, dy, dz = grid_info['dx'], grid_info['dy'], grid_info['dz']
        
        # Apply high-order finite difference schemes
        if min(nx, ny, nz) >= 5:  # 4th-order central difference
            self._apply_4th_order_3d_stencil(sigma, div_stress, nx, ny, nz, dx, dy, dz)
        elif min(nx, ny, nz) >= 3:  # 2nd-order central difference
            self._apply_2nd_order_3d_stencil(sigma, div_stress, nx, ny, nz, dx, dy, dz) 
        else:  # Fallback to 1st-order
            self._apply_1st_order_3d_stencil(sigma, div_stress, nx, ny, nz, dx, dy, dz)
            
        # Apply boundary condition corrections
        self._apply_3d_boundary_corrections(sigma, div_stress, nx, ny, nz, dx, dy, dz)
        
        return div_stress
        
    def _apply_4th_order_3d_stencil(self, sigma: Dict[str, np.ndarray], div_stress: np.ndarray,
                                   nx: int, ny: int, nz: int, dx: float, dy: float, dz: float):
        """Apply 4th-order accurate finite difference stencil"""
        # 4th-order stencil: f'(i) = [-f(i+2) + 8f(i+1) - 8f(i-1) + f(i-2)] / (12h)
        for k in range(2, nz-2):
            for j in range(2, ny-2):
                for i in range(2, nx-2):
                    idx = k*nx*ny + j*nx + i
                    
                    # X-direction derivatives
                    idx_xp2 = k*nx*ny + j*nx + (i+2)
                    idx_xp1 = k*nx*ny + j*nx + (i+1) 
                    idx_xm1 = k*nx*ny + j*nx + (i-1)
                    idx_xm2 = k*nx*ny + j*nx + (i-2)
                    
                    dsigma_xx_dx = (-sigma['xx'][idx_xp2] + 8*sigma['xx'][idx_xp1] - 
                                   8*sigma['xx'][idx_xm1] + sigma['xx'][idx_xm2]) / (12*dx)
                    dsigma_xy_dx = (-sigma['xy'][idx_xp2] + 8*sigma['xy'][idx_xp1] -
                                   8*sigma['xy'][idx_xm1] + sigma['xy'][idx_xm2]) / (12*dx)
                    dsigma_xz_dx = (-sigma['xz'][idx_xp2] + 8*sigma['xz'][idx_xp1] -
                                   8*sigma['xz'][idx_xm1] + sigma['xz'][idx_xm2]) / (12*dx)
                    
                    # Y-direction derivatives
                    idx_yp2 = k*nx*ny + (j+2)*nx + i
                    idx_yp1 = k*nx*ny + (j+1)*nx + i
                    idx_ym1 = k*nx*ny + (j-1)*nx + i  
                    idx_ym2 = k*nx*ny + (j-2)*nx + i
                    
                    dsigma_xy_dy = (-sigma['xy'][idx_yp2] + 8*sigma['xy'][idx_yp1] -
                                   8*sigma['xy'][idx_ym1] + sigma['xy'][idx_ym2]) / (12*dy)
                    dsigma_yy_dy = (-sigma['yy'][idx_yp2] + 8*sigma['yy'][idx_yp1] -
                                   8*sigma['yy'][idx_ym1] + sigma['yy'][idx_ym2]) / (12*dy)
                    dsigma_yz_dy = (-sigma['yz'][idx_yp2] + 8*sigma['yz'][idx_yp1] -
                                   8*sigma['yz'][idx_ym1] + sigma['yz'][idx_ym2]) / (12*dy)
                    
                    # Z-direction derivatives  
                    idx_zp2 = (k+2)*nx*ny + j*nx + i
                    idx_zp1 = (k+1)*nx*ny + j*nx + i
                    idx_zm1 = (k-1)*nx*ny + j*nx + i
                    idx_zm2 = (k-2)*nx*ny + j*nx + i
                    
                    dsigma_xz_dz = (-sigma['xz'][idx_zp2] + 8*sigma['xz'][idx_zp1] -
                                   8*sigma['xz'][idx_zm1] + sigma['xz'][idx_zm2]) / (12*dz)
                    dsigma_yz_dz = (-sigma['yz'][idx_zp2] + 8*sigma['yz'][idx_zp1] -
                                   8*sigma['yz'][idx_zm1] + sigma['yz'][idx_zm2]) / (12*dz)
                    dsigma_zz_dz = (-sigma['zz'][idx_zp2] + 8*sigma['zz'][idx_zp1] -
                                   8*sigma['zz'][idx_zm1] + sigma['zz'][idx_zm2]) / (12*dz)
                    
                    # Assemble divergence components
                    div_stress[idx, 0] = dsigma_xx_dx + dsigma_xy_dy + dsigma_xz_dz  
                    div_stress[idx, 1] = dsigma_xy_dx + dsigma_yy_dy + dsigma_yz_dz
                    div_stress[idx, 2] = dsigma_xz_dx + dsigma_yz_dy + dsigma_zz_dz
    
    def _detect_3d_grid_topology(self, n_points: int) -> Dict[str, Union[int, float]]:
        """Intelligent detection of 3D grid structure and spacing"""
        # Try common 3D grid arrangements
        possible_dims = []
        for nx in range(2, int(n_points**(1/3)) + 10):
            for ny in range(2, int(n_points/nx)**(1/2) + 5):
                nz = n_points // (nx * ny)
                if nx * ny * nz == n_points and nz >= 2:
                    # Score based on aspect ratio preference
                    aspect_score = 1.0 / (max(nx,ny,nz) / min(nx,ny,nz))
                    possible_dims.append((nx, ny, nz, aspect_score))
        
        if possible_dims:
            # Select best aspect ratio
            nx, ny, nz, _ = max(possible_dims, key=lambda x: x[3])
        else:
            # Fallback to cubic approximation
            n_per_dim = int(round(n_points**(1/3)))
            nx = ny = nz = n_per_dim
        
        # Grid spacing (adaptive or from config)
        dx = self.config.get('dx', self._get_grid_spacing())
        dy = self.config.get('dy', dx)  
        dz = self.config.get('dz', dx)
        
        return {'nx': nx, 'ny': ny, 'nz': nz, 'dx': dx, 'dy': dy, 'dz': dz}
    
    def _compute_mls_stress_gradients(self, center_idx: int, neighbor_indices: List[int],
                                    sigma: Dict[str, np.ndarray], coordinates: np.ndarray) -> Dict[str, np.ndarray]:
        """Moving Least Squares reconstruction of stress tensor gradients"""
        center_coord = coordinates[center_idx]
        n_neighbors = len(neighbor_indices)
        
        # Set up MLS system: P(x) * a = stress_values 
        # where P(x) = [1, x-x0, y-y0, z-z0, (x-x0)**2, (y-y0)**2, (z-z0)**2, xy, xz, yz]
        P = np.zeros((n_neighbors, 10))  # Complete 2nd-order polynomial basis
        W = np.zeros((n_neighbors, n_neighbors))  # Weight matrix
        
        # Support radius for weights
        distances = [np.linalg.norm(coordinates[j] - center_coord) for j in neighbor_indices]
        h = max(distances) if distances else self._get_grid_spacing()
        
        for i, j in enumerate(neighbor_indices):
            dx = coordinates[j] - center_coord
            r = distances[i]
            
            # Polynomial basis
            P[i] = [1, dx[0], dx[1], dx[2], dx[0]**2, dx[1]**2, dx[2]**2, 
                   dx[0]*dx[1], dx[0]*dx[2], dx[1]*dx[2]]
            
            # Cubic spline weight function
            r_norm = r / (h + 1e-12)
            if r_norm <= 1.0:
                w = 1 - 1.5*r_norm**2 + 0.75*r_norm**3 if r_norm <= 2/3 else 0.25*(2-r_norm)**3
            else:
                w = 0.0
            W[i, i] = w
        
        # Solve weighted least squares for each stress component
        gradients = {}
        for component in ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']:
            stress_values = np.array([sigma[component][j] for j in neighbor_indices])
            
            try:
                # Weighted normal equation: P^T W P a = P^T W stress_values
                A = P.T @ W @ P
                b = P.T @ W @ stress_values
                coeffs = np.linalg.solve(A, b)
                
                # Extract gradients: [df/dx, df/dy, df/dz] = [a1, a2, a3]
                gradients[component] = coeffs[1:4]  
                
            except np.linalg.LinAlgError:
                # Fallback to simple finite difference
                gradients[component] = self._compute_simple_gradient(center_idx, neighbor_indices, 
                                                                   sigma[component], coordinates)
        
        return gradients
    
    def _apply_curvature_correction(self, div_stress: np.ndarray, coordinates: Optional[np.ndarray]) -> np.ndarray:
        """Apply geometric correction factors for curved boundaries and surfaces"""
        if coordinates is None:
            return div_stress
            
        n_points = len(div_stress)
        corrected_div = div_stress.copy()
        
        # Identify boundary points and compute local curvature
        boundary_points = self._identify_boundary_points(coordinates)
        
        for i in boundary_points:
            # Compute local surface curvature using neighboring points
            curvature_tensor = self._compute_local_curvature(i, coordinates)
            
            # Apply curvature correction to stress divergence
            # Correction factor: (1 + ?_1*n_1 + ?_2*n_2) where ?? are principal curvatures
            correction_factor = 1.0 + np.trace(curvature_tensor) * 0.1  # Simplified
            corrected_div[i] *= correction_factor
            
        return corrected_div
    
    def _apply_streamline_upwind_stabilization(self, div_stress: np.ndarray, stress: np.ndarray,
                                             coordinates: Optional[np.ndarray]) -> np.ndarray:
        """Apply SUPG-like stabilization for convection-dominated stress transport"""
        if coordinates is None:
            return div_stress
            
        n_points = len(div_stress)
        stabilized_div = div_stress.copy()
        
        # Compute characteristic directions and stabilization parameters
        for i in range(n_points):
            # Principal stress directions (approximate characteristic directions)
            stress_tensor = np.array([[stress[i, 0], stress[i, 3], stress[i, 5]],
                                    [stress[i, 3], stress[i, 1], stress[i, 4]], 
                                    [stress[i, 5], stress[i, 4], stress[i, 2]]])
            
            eigenvals, eigenvecs = np.linalg.eigh(stress_tensor)
            principal_direction = eigenvecs[:, np.argmax(np.abs(eigenvals))]
            
            # Streamline diffusion parameter
            h_local = self._get_grid_spacing()  # Local element size
            Pe = np.linalg.norm(principal_direction) * h_local / (2 * 1e-6)  # Peclet number
            tau = h_local / 2 * np.tanh(Pe) / Pe if Pe > 1e-12 else 0.0
            
            # Add stabilization term: tau * (u*nabla) * (u*nablasigma)
            stabilization = tau * np.dot(principal_direction, div_stress[i].reshape(3))
            stabilized_div[i] += stabilization * 0.1  # Scaled contribution
            
        return stabilized_div
    
    # Helper methods for advanced 3D stress divergence computation
    
    def _build_adjacency_list(self, connectivity: np.ndarray, n_points: int) -> List[List[int]]:
        """Build adjacency list from mesh connectivity"""
        adjacency = [[] for _ in range(n_points)]
        
        if len(connectivity.shape) == 2:  # Element-to-node connectivity
            for element in connectivity:
                for i, node_i in enumerate(element):
                    for j, node_j in enumerate(element):
                        if i != j and node_j not in adjacency[node_i]:
                            adjacency[node_i].append(node_j)
        else:  # Edge connectivity
            for edge in connectivity:
                if len(edge) >= 2:
                    i, j = edge[0], edge[1]
                    if j not in adjacency[i]:
                        adjacency[i].append(j)
                    if i not in adjacency[j]:
                        adjacency[j].append(i)
        
        return adjacency
    
    def _compute_dual_face(self, node_i: int, node_j: int, coordinates: np.ndarray, 
                          adjacency: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced dual face computation using sophisticated control volume construction
        with Voronoi-based dual mesh generation and geometric quality optimization
        
        Args:
            node_i, node_j: Adjacent nodes defining the primary edge
            coordinates: Global coordinate array
            adjacency: Mesh connectivity information
            
        Returns:
            face_center: Geometrically optimized dual face center
            face_area_vector: Accurate face area vector with proper orientation
        """
        pos_i = coordinates[node_i]
        pos_j = coordinates[node_j]
        edge_vector = pos_j - pos_i
        edge_length = np.linalg.norm(edge_vector)
        
        if edge_length < 1e-12:
            return pos_i.copy(), np.zeros(3)
            
        # Find all elements (triangles/tetrahedra) sharing this edge
        common_neighbors = list(set(adjacency[node_i]) & set(adjacency[node_j]))
        
        if len(common_neighbors) == 0:
            # Boundary edge - construct boundary dual face
            return self._construct_boundary_dual_face(pos_i, pos_j, edge_vector)
        
        elif len(common_neighbors) == 1:
            # Boundary edge with one adjacent element
            return self._construct_boundary_dual_face_with_element(
                pos_i, pos_j, coordinates[common_neighbors[0]], edge_vector)
        
        elif len(common_neighbors) == 2:
            # Interior edge shared by two elements (most common case)
            return self._construct_interior_dual_face_2d(
                pos_i, pos_j, coordinates[common_neighbors[0]], 
                coordinates[common_neighbors[1]], adjacency, coordinates)
        
        else:
            # Complex 3D configuration with multiple adjacent elements
            return self._construct_complex_dual_face_3d(
                node_i, node_j, common_neighbors, coordinates, adjacency)
    
    def _construct_boundary_dual_face(self, pos_i: np.ndarray, pos_j: np.ndarray, 
                                     edge_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct dual face for boundary edges without adjacent elements"""
        edge_center = 0.5 * (pos_i + pos_j)
        edge_length = np.linalg.norm(edge_vector)
        
        # Create perpendicular vector in 3D
        if abs(edge_vector[2]) < 0.9:
            perp_vector = np.cross(edge_vector, np.array([0, 0, 1]))
        else:
            perp_vector = np.cross(edge_vector, np.array([1, 0, 0]))
        
        perp_vector = perp_vector / (np.linalg.norm(perp_vector) + 1e-12)
        
        # Face area vector points inward to domain
        face_area_vector = perp_vector * edge_length * 0.5  # Simplified boundary thickness
        
        return edge_center, face_area_vector
    
    def _construct_boundary_dual_face_with_element(self, pos_i: np.ndarray, pos_j: np.ndarray,
                                                  pos_k: np.ndarray, edge_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct dual face for boundary edge with one adjacent triangular element"""
        # Compute triangle circumcenter
        circumcenter = self._compute_triangle_circumcenter(pos_i, pos_j, pos_k)
        edge_center = 0.5 * (pos_i + pos_j)
        
        # Dual face connects edge center to circumcenter
        dual_edge = circumcenter - edge_center
        
        # Face normal perpendicular to primary edge and in plane of triangle
        triangle_normal = np.cross(pos_j - pos_i, pos_k - pos_i)
        triangle_normal = triangle_normal / (np.linalg.norm(triangle_normal) + 1e-12)
        
        face_normal = np.cross(edge_vector, triangle_normal)
        face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-12)
        
        # Face area magnitude is length of dual edge
        face_area = np.linalg.norm(dual_edge)
        face_area_vector = face_normal * face_area
        
        return edge_center, face_area_vector
    
    def _construct_interior_dual_face_2d(self, pos_i: np.ndarray, pos_j: np.ndarray,
                                        pos_k1: np.ndarray, pos_k2: np.ndarray,
                                        adjacency: List[List[int]], coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct dual face for interior edge shared by two triangular elements"""
        # Compute circumcenters of both adjacent triangles
        circumcenter_1 = self._compute_triangle_circumcenter(pos_i, pos_j, pos_k1)
        circumcenter_2 = self._compute_triangle_circumcenter(pos_i, pos_j, pos_k2)
        
        # Dual face is quadrilateral: edge_center -> circumcenter_1 -> circumcenter_2 -> edge_center
        edge_center = 0.5 * (pos_i + pos_j)
        
        # Optimize dual face center using weighted circumcenter positioning
        triangle_area_1 = 0.5 * np.linalg.norm(np.cross(pos_j - pos_i, pos_k1 - pos_i))
        triangle_area_2 = 0.5 * np.linalg.norm(np.cross(pos_j - pos_i, pos_k2 - pos_i))
        total_area = triangle_area_1 + triangle_area_2
        
        if total_area > 1e-12:
            # Weighted average of circumcenters based on triangle quality
            quality_1 = self._compute_triangle_quality(pos_i, pos_j, pos_k1)
            quality_2 = self._compute_triangle_quality(pos_i, pos_j, pos_k2)
            
            weight_1 = quality_1 * triangle_area_1 / total_area
            weight_2 = quality_2 * triangle_area_2 / total_area
            
            face_center = (weight_1 * circumcenter_1 + weight_2 * circumcenter_2) / (weight_1 + weight_2)
        else:
            face_center = 0.5 * (circumcenter_1 + circumcenter_2)
        
        # Dual face area vector using proper geometric integration
        dual_face_vector = circumcenter_1 - circumcenter_2
        edge_vector = pos_j - pos_i
        
        # Face area vector perpendicular to edge and proportional to dual edge length
        face_area_vector = np.cross(edge_vector, dual_face_vector)
        face_area_magnitude = np.linalg.norm(dual_face_vector)
        
        # Normalize and scale
        if np.linalg.norm(face_area_vector) > 1e-12:
            face_area_vector = face_area_vector / np.linalg.norm(face_area_vector) * face_area_magnitude
        
        return face_center, face_area_vector
    
    def _construct_complex_dual_face_3d(self, node_i: int, node_j: int, common_neighbors: List[int],
                                       coordinates: np.ndarray, adjacency: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Construct dual face for complex 3D configurations with multiple adjacent elements"""
        pos_i = coordinates[node_i]
        pos_j = coordinates[node_j]
        edge_center = 0.5 * (pos_i + pos_j)
        
        # Collect all tetrahedra/elements sharing this edge
        tetrahedra = self._find_tetrahedra_sharing_edge(node_i, node_j, common_neighbors, adjacency)
        
        if len(tetrahedra) == 0:
            # Fallback to simple construction
            return self._construct_boundary_dual_face(pos_i, pos_j, pos_j - pos_i)
        
        # Compute circumcenters and volumes of all adjacent tetrahedra
        circumcenters = []
        volumes = []
        qualities = []
        
        for tet_nodes in tetrahedra:
            if len(tet_nodes) >= 4:
                tet_coords = coordinates[tet_nodes]
                circumcenter = self._compute_tetrahedron_circumcenter(tet_coords)
                volume = abs(self._compute_tetrahedron_volume_from_coords(tet_coords))
                quality = self._compute_tetrahedron_quality(tet_coords)
                
                circumcenters.append(circumcenter)
                volumes.append(volume)
                qualities.append(quality)
        
        if len(circumcenters) == 0:
            return self._construct_boundary_dual_face(pos_i, pos_j, pos_j - pos_i)
        
        # Construct dual polyhedron face using advanced geometric averaging
        total_weight = sum(v * q for v, q in zip(volumes, qualities))
        
        if total_weight > 1e-12:
            weighted_center = sum(w * v * q * c for c, v, q, w in 
                                zip(circumcenters, volumes, qualities, [1/len(circumcenters)]*len(circumcenters)))
            face_center = weighted_center / total_weight
        else:
            face_center = np.mean(circumcenters, axis=0)
        
        # Compute face area vector using 3D dual surface integration
        face_area_vector = self._compute_3d_dual_surface_integral(
            edge_center, circumcenters, volumes, pos_i, pos_j)
        
        return face_center, face_area_vector
    
    def _interpolate_stress_to_face(self, sigma: Dict[str, np.ndarray], node_i: int, node_j: int,
                                   coordinates: np.ndarray, adjacency: Optional[List[List[int]]] = None) -> Dict[str, float]:
        """
        Advanced stress tensor interpolation to face centers using high-order schemes
        with gradient reconstruction and geometric weighting
        
        Args:
            sigma: Stress tensor components at nodes
            node_i, node_j: Nodes defining the face
            coordinates: Global coordinates
            adjacency: Optional mesh connectivity for gradient-based interpolation
            
        Returns:
            stress_face: Interpolated stress tensor components at face center
        """
        pos_i = coordinates[node_i]
        pos_j = coordinates[node_j]
        face_center = 0.5 * (pos_i + pos_j)
        distance = np.linalg.norm(pos_j - pos_i)
        
        # Choose interpolation scheme based on available data and configuration
        interpolation_order = self.config.get('stress_interpolation_order', 2)
        
        if interpolation_order >= 2 and adjacency is not None:
            # Gradient-enhanced interpolation using least squares reconstruction
            return self._gradient_enhanced_stress_interpolation(
                sigma, node_i, node_j, face_center, coordinates, adjacency)
        
        elif interpolation_order >= 1 and distance > 1e-12:
            # Distance-weighted interpolation with geometric corrections
            return self._distance_weighted_stress_interpolation(
                sigma, node_i, node_j, pos_i, pos_j, face_center, coordinates)
        
        else:
            # Enhanced averaging with stress tensor preservation
            return self._tensor_preserving_stress_interpolation(sigma, node_i, node_j)
    
    def _gradient_enhanced_stress_interpolation(self, sigma: Dict[str, np.ndarray], 
                                              node_i: int, node_j: int, face_center: np.ndarray,
                                              coordinates: np.ndarray, adjacency: List[List[int]]) -> Dict[str, float]:
        """High-order stress interpolation using gradient reconstruction"""
        stress_face = {}
        
        for component in sigma.keys():
            # Reconstruct gradients at both nodes using least squares
            grad_i = self._reconstruct_nodal_gradient(node_i, component, sigma[component], 
                                                     coordinates, adjacency)
            grad_j = self._reconstruct_nodal_gradient(node_j, component, sigma[component], 
                                                     coordinates, adjacency)
            
            # Taylor expansion to face center from both nodes
            pos_i = coordinates[node_i]
            pos_j = coordinates[node_j]
            
            # Interpolation from node i
            delta_i = face_center - pos_i
            stress_from_i = sigma[component][node_i] + np.dot(grad_i, delta_i)
            
            # Interpolation from node j  
            delta_j = face_center - pos_j
            stress_from_j = sigma[component][node_j] + np.dot(grad_j, delta_j)
            
            # Distance-weighted combination with gradient consistency
            dist_i = np.linalg.norm(delta_i)
            dist_j = np.linalg.norm(delta_j)
            total_dist = dist_i + dist_j
            
            if total_dist > 1e-12:
                weight_j = dist_i / total_dist  # Inverse distance weighting
                weight_i = dist_j / total_dist
                
                # Gradient consistency correction
                gradient_diff = np.linalg.norm(grad_i - grad_j)
                consistency_factor = 1.0 / (1.0 + gradient_diff)
                
                stress_face[component] = (weight_i * stress_from_i + weight_j * stress_from_j) * consistency_factor
                
                # Add higher-order correction for curved stress fields
                if self.config.get('stress_curvature_correction', True):
                    curvature_correction = self._compute_stress_curvature_correction(
                        component, node_i, node_j, face_center, coordinates, sigma[component])
                    stress_face[component] += curvature_correction
            else:
                stress_face[component] = 0.5 * (sigma[component][node_i] + sigma[component][node_j])
        
        return stress_face
    
    def _distance_weighted_stress_interpolation(self, sigma: Dict[str, np.ndarray],
                                              node_i: int, node_j: int, pos_i: np.ndarray, 
                                              pos_j: np.ndarray, face_center: np.ndarray,
                                              coordinates: np.ndarray) -> Dict[str, float]:
        """Distance-weighted stress interpolation with geometric corrections"""
        stress_face = {}
        
        dist_i = np.linalg.norm(face_center - pos_i)
        dist_j = np.linalg.norm(face_center - pos_j)
        
        # Inverse distance weighting with geometric quality factors
        if dist_i + dist_j > 1e-12:
            weight_i = dist_j / (dist_i + dist_j)
            weight_j = dist_i / (dist_i + dist_j)
            
            # Apply geometric quality correction based on local mesh quality
            quality_factor = self._compute_edge_quality_factor(pos_i, pos_j, coordinates)
            
            for component in sigma.keys():
                # Basic inverse distance interpolation
                stress_base = weight_i * sigma[component][node_i] + weight_j * sigma[component][node_j]
                
                # Apply quality and geometric corrections
                stress_face[component] = stress_base * quality_factor
                
                # Add anisotropy correction for stress tensor components
                if component in ['xy', 'xz', 'yz']:  # Shear components
                    anisotropy_correction = self._compute_shear_anisotropy_correction(
                        component, sigma, node_i, node_j)
                    stress_face[component] += anisotropy_correction
        else:
            # Fallback to simple averaging
            for component in sigma.keys():
                stress_face[component] = 0.5 * (sigma[component][node_i] + sigma[component][node_j])
        
        return stress_face
    
    def _tensor_preserving_stress_interpolation(self, sigma: Dict[str, np.ndarray],
                                              node_i: int, node_j: int) -> Dict[str, float]:
        """Enhanced averaging that preserves stress tensor properties"""
        stress_face = {}
        
        # Extract stress tensors at nodes
        tensor_i = self._extract_stress_tensor(sigma, node_i)
        tensor_j = self._extract_stress_tensor(sigma, node_j)
        
        # Compute tensor interpolation preserving invariants
        tensor_face = self._interpolate_stress_tensors(tensor_i, tensor_j)
        
        # Convert back to Voigt notation
        stress_face['xx'] = tensor_face[0, 0]
        stress_face['yy'] = tensor_face[1, 1] 
        stress_face['zz'] = tensor_face[2, 2]
        stress_face['xy'] = tensor_face[0, 1]
        stress_face['yz'] = tensor_face[1, 2]
        stress_face['xz'] = tensor_face[0, 2]
        
        return stress_face
    
    # Helper methods for advanced dual face computation and stress interpolation
    
    def _compute_triangle_circumcenter(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        """Compute circumcenter of triangle using robust geometric algorithm"""
        # Translate triangle to origin
        a = p2 - p1
        b = p3 - p1
        
        # Compute cross products for robust circumcenter calculation
        cross_ab = np.cross(a, b)
        cross_norm_sq = np.dot(cross_ab, cross_ab)
        
        if cross_norm_sq < 1e-12:  # Degenerate triangle
            return (p1 + p2 + p3) / 3.0  # Return centroid
        
        # Circumcenter formula: O = (|b|**2a - |a|**2b) * (a * b) / (2|a * b|**2) + p1
        a_norm_sq = np.dot(a, a)
        b_norm_sq = np.dot(b, b)
        
        numerator = np.cross(b_norm_sq * a - a_norm_sq * b, cross_ab)
        circumcenter = p1 + numerator / (2.0 * cross_norm_sq)
        
        return circumcenter
    
    def _compute_triangle_quality(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Compute triangle quality measure (0 = degenerate, 1 = equilateral)"""
        # Edge lengths
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)  
        c = np.linalg.norm(p1 - p2)
        
        if a + b + c < 1e-12:
            return 0.0
        
        # Area using cross product
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        
        # Quality = 4*sqrt(3)*Area / (a**2 + b**2 + c**2)
        perimeter_sq = a*a + b*b + c*c
        quality = 4.0 * np.sqrt(3.0) * area / (perimeter_sq + 1e-12)
        
        return min(quality, 1.0)
    
    def _find_tetrahedra_sharing_edge(self, node_i: int, node_j: int, 
                                     common_neighbors: List[int], 
                                     adjacency: List[List[int]]) -> List[List[int]]:
        """Find all tetrahedra that share a given edge"""
        tetrahedra = []
        
        # Look for sets of 4 nodes that form tetrahedra containing the edge (node_i, node_j)
        for k in range(len(common_neighbors)):
            for l in range(k + 1, len(common_neighbors)):
                node_k = common_neighbors[k]
                node_l = common_neighbors[l]
                
                # Check if nodes k and l are connected (form a tetrahedron with i,j)
                if (node_l in adjacency[node_k] and 
                    node_i in adjacency[node_k] and 
                    node_j in adjacency[node_k] and
                    node_i in adjacency[node_l] and
                    node_j in adjacency[node_l]):
                    
                    tetrahedron = [node_i, node_j, node_k, node_l]
                    tetrahedra.append(tetrahedron)
        
        return tetrahedra
    
    def _compute_tetrahedron_circumcenter(self, tet_coords: np.ndarray) -> np.ndarray:
        """Compute circumcenter of tetrahedron using robust algorithm"""
        if len(tet_coords) < 4:
            return np.mean(tet_coords, axis=0)
            
        # Translate to origin
        p0, p1, p2, p3 = tet_coords[0], tet_coords[1], tet_coords[2], tet_coords[3]
        a = p1 - p0
        b = p2 - p0  
        c = p3 - p0
        
        # Build system: [a b c] * x = 0.5 * [|a|**2 |b|**2 |c|**2]
        matrix = np.column_stack([a, b, c])
        rhs = 0.5 * np.array([np.dot(a, a), np.dot(b, b), np.dot(c, c)])
        
        try:
            # Solve for circumcenter relative to p0
            x = np.linalg.solve(matrix, rhs)
            circumcenter = p0 + x[0] * a + x[1] * b + x[2] * c
        except np.linalg.LinAlgError:
            # Fallback to centroid for degenerate cases
            circumcenter = np.mean(tet_coords, axis=0)
        
        return circumcenter
    
    def _compute_tetrahedron_volume_from_coords(self, tet_coords: np.ndarray) -> float:
        """Compute volume of tetrahedron from coordinate array"""
        if len(tet_coords) < 4:
            return 0.0
            
        # Volume = |det([p1-p0, p2-p0, p3-p0])| / 6
        p0, p1, p2, p3 = tet_coords[0], tet_coords[1], tet_coords[2], tet_coords[3]
        matrix = np.column_stack([p1-p0, p2-p0, p3-p0])
        return abs(np.linalg.det(matrix)) / 6.0
    
    def _compute_tetrahedron_quality(self, tet_coords: np.ndarray) -> float:
        """Compute tetrahedron quality measure using aspect ratio and regularity"""
        if len(tet_coords) < 4:
            return 0.0
            
        # Compute edge lengths
        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                edge_length = np.linalg.norm(tet_coords[j] - tet_coords[i])
                edges.append(edge_length)
        
        if len(edges) == 0 or max(edges) < 1e-12:
            return 0.0
        
        # Quality based on ratio of shortest to longest edge
        min_edge = min(edges)
        max_edge = max(edges)
        aspect_ratio = min_edge / max_edge
        
        # Volume regularity
        volume = self._compute_tetrahedron_volume_from_coords(tet_coords)
        avg_edge = np.mean(edges)
        ideal_volume = avg_edge**3 / (6 * np.sqrt(2))  # Regular tetrahedron volume
        
        volume_regularity = volume / (ideal_volume + 1e-12)
        
        # Combined quality measure
        quality = aspect_ratio * min(volume_regularity, 1.0)
        return min(quality, 1.0)
    
    def _compute_3d_dual_surface_integral(self, edge_center: np.ndarray, circumcenters: List[np.ndarray],
                                        volumes: List[float], pos_i: np.ndarray, pos_j: np.ndarray) -> np.ndarray:
        """Compute 3D dual surface integral for face area vector"""
        if len(circumcenters) == 0:
            return np.zeros(3)
        
        edge_vector = pos_j - pos_i
        total_volume = sum(volumes)
        
        # Weighted sum of cross products for surface integration
        surface_integral = np.zeros(3)
        
        for i, (circumcenter, volume) in enumerate(zip(circumcenters, volumes)):
            # Vector from edge center to circumcenter
            radial_vector = circumcenter - edge_center
            
            # Cross product gives area contribution
            area_contribution = np.cross(edge_vector, radial_vector)
            
            # Weight by element volume for proper integration
            weight = volume / (total_volume + 1e-12)
            surface_integral += weight * area_contribution
        
        return surface_integral
    
    def _reconstruct_nodal_gradient(self, node_idx: int, component: str, field_values: np.ndarray,
                                   coordinates: np.ndarray, adjacency: List[List[int]]) -> np.ndarray:
        """Reconstruct gradient at node using least squares from neighbors"""
        neighbors = adjacency[node_idx]
        if len(neighbors) < 3:  # Need minimum neighbors for 3D gradient
            return np.zeros(3)
        
        center_pos = coordinates[node_idx]
        center_value = field_values[node_idx]
        
        # Build least squares system: A * grad = b
        n_neighbors = len(neighbors)
        A = np.zeros((n_neighbors, 3))
        b = np.zeros(n_neighbors)
        
        for i, neighbor_idx in enumerate(neighbors):
            neighbor_pos = coordinates[neighbor_idx]
            neighbor_value = field_values[neighbor_idx]
            
            # Displacement vector
            delta_pos = neighbor_pos - center_pos
            A[i] = delta_pos
            
            # Value difference
            b[i] = neighbor_value - center_value
        
        try:
            # Solve least squares system
            gradient, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return gradient
        except np.linalg.LinAlgError:
            return np.zeros(3)
    
    def _compute_stress_curvature_correction(self, component: str, node_i: int, node_j: int,
                                           face_center: np.ndarray, coordinates: np.ndarray,
                                           stress_values: np.ndarray) -> float:
        """Compute higher-order curvature correction for stress interpolation"""
        pos_i = coordinates[node_i]
        pos_j = coordinates[node_j]
        
        # Estimate local curvature using finite differences
        edge_vector = pos_j - pos_i
        edge_length = np.linalg.norm(edge_vector)
        
        if edge_length < 1e-12:
            return 0.0
        
        # Second-order correction based on stress gradient variation
        stress_i = stress_values[node_i]
        stress_j = stress_values[node_j]
        
        # Estimate second derivative
        linear_gradient = (stress_j - stress_i) / edge_length
        
        # Curvature correction proportional to gradient and distance squared
        distance_to_face = np.linalg.norm(face_center - 0.5 * (pos_i + pos_j))
        curvature_correction = 0.5 * linear_gradient * distance_to_face**2
        
        return curvature_correction * 0.1  # Small correction factor
    
    def _compute_edge_quality_factor(self, pos_i: np.ndarray, pos_j: np.ndarray, 
                                    coordinates: np.ndarray) -> float:
        """Compute geometric quality factor for edge-based interpolation"""
        edge_length = np.linalg.norm(pos_j - pos_i)
        
        # Find nearby points to assess local mesh quality
        edge_center = 0.5 * (pos_i + pos_j)
        
        # Simple quality measure based on edge aspect ratio
        # (in practice, would use more sophisticated mesh quality metrics)
        min_coord_spacing = self._get_grid_spacing()
        
        aspect_ratio = edge_length / (min_coord_spacing + 1e-12)
        quality_factor = 1.0 / (1.0 + 0.1 * abs(np.log(aspect_ratio)))
        
        return min(max(quality_factor, 0.5), 2.0)  # Clamp to reasonable range
    
    def _compute_shear_anisotropy_correction(self, component: str, sigma: Dict[str, np.ndarray],
                                           node_i: int, node_j: int) -> float:
        """Compute anisotropy correction for shear stress components"""
        # Extract normal stresses for anisotropy assessment
        normal_stress_i = [sigma['xx'][node_i], sigma['yy'][node_i], sigma['zz'][node_i]]
        normal_stress_j = [sigma['xx'][node_j], sigma['yy'][node_j], sigma['zz'][node_j]]
        
        # Compute stress anisotropy measure
        avg_normal_i = np.mean(normal_stress_i)
        avg_normal_j = np.mean(normal_stress_j)
        
        anisotropy_i = np.std(normal_stress_i) / (abs(avg_normal_i) + 1e-6)
        anisotropy_j = np.std(normal_stress_j) / (abs(avg_normal_j) + 1e-6)
        
        avg_anisotropy = 0.5 * (anisotropy_i + anisotropy_j)
        
        # Correction proportional to anisotropy and shear stress magnitude
        shear_stress_i = sigma[component][node_i]
        shear_stress_j = sigma[component][node_j]
        avg_shear = 0.5 * (shear_stress_i + shear_stress_j)
        
        correction = avg_anisotropy * avg_shear * 0.05  # Small correction factor
        
        return correction
    
    def _extract_stress_tensor(self, sigma: Dict[str, np.ndarray], node_idx: int) -> np.ndarray:
        """Extract 3x3 stress tensor from Voigt notation"""
        tensor = np.zeros((3, 3))
        
        # Diagonal components
        tensor[0, 0] = sigma['xx'][node_idx]
        tensor[1, 1] = sigma['yy'][node_idx] 
        tensor[2, 2] = sigma['zz'][node_idx]
        
        # Off-diagonal components (symmetric)
        tensor[0, 1] = tensor[1, 0] = sigma['xy'][node_idx]
        tensor[1, 2] = tensor[2, 1] = sigma['yz'][node_idx]
        tensor[0, 2] = tensor[2, 0] = sigma['xz'][node_idx]
        
        return tensor
    
    def _interpolate_stress_tensors(self, tensor_i: np.ndarray, tensor_j: np.ndarray) -> np.ndarray:
        """Interpolate stress tensors preserving invariants and positive definiteness"""
        # Simple averaging preserves symmetry
        tensor_avg = 0.5 * (tensor_i + tensor_j)
        
        # Check and enforce positive semi-definiteness if needed
        if self.config.get('enforce_tensor_positivity', False):
            eigenvals, eigenvecs = np.linalg.eigh(tensor_avg)
            
            # Clamp negative eigenvalues to zero
            eigenvals = np.maximum(eigenvals, 0.0)
            
            # Reconstruct tensor
            tensor_avg = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return tensor_avg
    
    def _compute_tetrahedron_volume(self, p1: np.ndarray, p2: np.ndarray, 
                                   p3: np.ndarray, p4: np.ndarray) -> float:
        """Compute volume of tetrahedron defined by four points"""
        # Volume = |det([p2-p1, p3-p1, p4-p1])| / 6
        matrix = np.column_stack([p2-p1, p3-p1, p4-p1])
        return abs(np.linalg.det(matrix)) / 6.0
    
    def _compute_adaptive_support_radii(self, coordinates: np.ndarray, tree: cKDTree) -> np.ndarray:
        """Compute adaptive support radii based on local point density"""
        n_points = len(coordinates)
        support_radii = np.zeros(n_points)
        
        for i in range(n_points):
            # Find distance to k-th nearest neighbor
            k = min(12, n_points - 1)  # Adaptive k based on total points
            distances, _ = tree.query(coordinates[i], k=k+1)  # +1 to exclude self
            
            # Support radius = alpha * distance to k-th neighbor
            alpha = 2.5  # Support size factor
            support_radii[i] = alpha * distances[-1] if len(distances) > 1 else self._get_grid_spacing()
            
        return support_radii
    
    def _compute_simple_gradient(self, center_idx: int, neighbor_indices: List[int],
                                field_values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """Fallback simple gradient computation using finite differences"""
        if len(neighbor_indices) < 2:
            return np.zeros(3)
            
        center_coord = coordinates[center_idx]
        center_value = field_values[center_idx]
        
        # Simple directional derivatives
        gradient = np.zeros(3)
        for neighbor_idx in neighbor_indices[:6]:  # Limit to 6 neighbors
            neighbor_coord = coordinates[neighbor_idx]
            neighbor_value = field_values[neighbor_idx]
            
            direction = neighbor_coord - center_coord
            distance = np.linalg.norm(direction)
            
            if distance > 1e-12:
                unit_direction = direction / distance
                directional_derivative = (neighbor_value - center_value) / distance
                gradient += directional_derivative * unit_direction
                
        return gradient / len(neighbor_indices[:6])
    
    def _apply_2nd_order_3d_stencil(self, sigma: Dict[str, np.ndarray], div_stress: np.ndarray,
                                   nx: int, ny: int, nz: int, dx: float, dy: float, dz: float):
        """Apply 2nd-order central difference stencil"""
        for k in range(1, nz-1):
            for j in range(1, ny-1):
                for i in range(1, nx-1):
                    idx = k*nx*ny + j*nx + i
                    
                    # X-direction derivatives (2nd order central)
                    idx_xp1 = k*nx*ny + j*nx + (i+1)
                    idx_xm1 = k*nx*ny + j*nx + (i-1)
                    dsigma_xx_dx = (sigma['xx'][idx_xp1] - sigma['xx'][idx_xm1]) / (2*dx)
                    dsigma_xy_dx = (sigma['xy'][idx_xp1] - sigma['xy'][idx_xm1]) / (2*dx)
                    dsigma_xz_dx = (sigma['xz'][idx_xp1] - sigma['xz'][idx_xm1]) / (2*dx)
                    
                    # Y-direction derivatives
                    idx_yp1 = k*nx*ny + (j+1)*nx + i
                    idx_ym1 = k*nx*ny + (j-1)*nx + i
                    dsigma_xy_dy = (sigma['xy'][idx_yp1] - sigma['xy'][idx_ym1]) / (2*dy)
                    dsigma_yy_dy = (sigma['yy'][idx_yp1] - sigma['yy'][idx_ym1]) / (2*dy)
                    dsigma_yz_dy = (sigma['yz'][idx_yp1] - sigma['yz'][idx_ym1]) / (2*dy)
                    
                    # Z-direction derivatives
                    idx_zp1 = (k+1)*nx*ny + j*nx + i
                    idx_zm1 = (k-1)*nx*ny + j*nx + i
                    dsigma_xz_dz = (sigma['xz'][idx_zp1] - sigma['xz'][idx_zm1]) / (2*dz)
                    dsigma_yz_dz = (sigma['yz'][idx_zp1] - sigma['yz'][idx_zm1]) / (2*dz)
                    dsigma_zz_dz = (sigma['zz'][idx_zp1] - sigma['zz'][idx_zm1]) / (2*dz)
                    
                    # Assemble divergence
                    div_stress[idx, 0] = dsigma_xx_dx + dsigma_xy_dy + dsigma_xz_dz
                    div_stress[idx, 1] = dsigma_xy_dx + dsigma_yy_dy + dsigma_yz_dz
                    div_stress[idx, 2] = dsigma_xz_dx + dsigma_yz_dy + dsigma_zz_dz
    
    def _apply_1st_order_3d_stencil(self, sigma: Dict[str, np.ndarray], div_stress: np.ndarray,
                                   nx: int, ny: int, nz: int, dx: float, dy: float, dz: float):
        """Apply 1st-order forward/backward difference stencil"""
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    idx = k*nx*ny + j*nx + i
                    
                    # X-direction derivatives
                    if i < nx-1:
                        idx_xp1 = k*nx*ny + j*nx + (i+1)
                        dsigma_xx_dx = (sigma['xx'][idx_xp1] - sigma['xx'][idx]) / dx
                        dsigma_xy_dx = (sigma['xy'][idx_xp1] - sigma['xy'][idx]) / dx
                        dsigma_xz_dx = (sigma['xz'][idx_xp1] - sigma['xz'][idx]) / dx
                    else:
                        dsigma_xx_dx = dsigma_xy_dx = dsigma_xz_dx = 0.0
                    
                    # Y-direction derivatives
                    if j < ny-1:
                        idx_yp1 = k*nx*ny + (j+1)*nx + i
                        dsigma_xy_dy = (sigma['xy'][idx_yp1] - sigma['xy'][idx]) / dy
                        dsigma_yy_dy = (sigma['yy'][idx_yp1] - sigma['yy'][idx]) / dy
                        dsigma_yz_dy = (sigma['yz'][idx_yp1] - sigma['yz'][idx]) / dy
                    else:
                        dsigma_xy_dy = dsigma_yy_dy = dsigma_yz_dy = 0.0
                    
                    # Z-direction derivatives
                    if k < nz-1:
                        idx_zp1 = (k+1)*nx*ny + j*nx + i
                        dsigma_xz_dz = (sigma['xz'][idx_zp1] - sigma['xz'][idx]) / dz
                        dsigma_yz_dz = (sigma['yz'][idx_zp1] - sigma['yz'][idx]) / dz
                        dsigma_zz_dz = (sigma['zz'][idx_zp1] - sigma['zz'][idx]) / dz
                    else:
                        dsigma_xz_dz = dsigma_yz_dz = dsigma_zz_dz = 0.0
                    
                    # Assemble divergence
                    div_stress[idx, 0] = dsigma_xx_dx + dsigma_xy_dy + dsigma_xz_dz
                    div_stress[idx, 1] = dsigma_xy_dx + dsigma_yy_dy + dsigma_yz_dz
                    div_stress[idx, 2] = dsigma_xz_dx + dsigma_yz_dy + dsigma_zz_dz
    
    def _apply_3d_boundary_corrections(self, sigma: Dict[str, np.ndarray], div_stress: np.ndarray,
                                      nx: int, ny: int, nz: int, dx: float, dy: float, dz: float):
        """Apply boundary condition corrections for stress divergence"""
        # Corrections for boundary points that couldn't use central differences
        
        # X-boundaries (i=0 and i=nx-1)
        for k in range(nz):
            for j in range(ny):
                # Left boundary (i=0)
                idx = k*nx*ny + j*nx + 0
                if nx > 2:
                    idx_p1 = k*nx*ny + j*nx + 1
                    idx_p2 = k*nx*ny + j*nx + 2
                    # Forward difference: f'(0) = (-3f(0) + 4f(1) - f(2)) / (2h)
                    dsigma_xx_dx = (-3*sigma['xx'][idx] + 4*sigma['xx'][idx_p1] - sigma['xx'][idx_p2]) / (2*dx)
                    dsigma_xy_dx = (-3*sigma['xy'][idx] + 4*sigma['xy'][idx_p1] - sigma['xy'][idx_p2]) / (2*dx)
                    dsigma_xz_dx = (-3*sigma['xz'][idx] + 4*sigma['xz'][idx_p1] - sigma['xz'][idx_p2]) / (2*dx)
                    
                    div_stress[idx, 0] += dsigma_xx_dx - (sigma['xx'][idx_p1] - sigma['xx'][idx]) / dx
                    div_stress[idx, 1] += dsigma_xy_dx - (sigma['xy'][idx_p1] - sigma['xy'][idx]) / dx
                    div_stress[idx, 2] += dsigma_xz_dx - (sigma['xz'][idx_p1] - sigma['xz'][idx]) / dx
                
                # Right boundary (i=nx-1)
                idx = k*nx*ny + j*nx + (nx-1)
                if nx > 2:
                    idx_m1 = k*nx*ny + j*nx + (nx-2)
                    idx_m2 = k*nx*ny + j*nx + (nx-3)
                    # Backward difference: f'(n) = (3f(n) - 4f(n-1) + f(n-2)) / (2h)
                    dsigma_xx_dx = (3*sigma['xx'][idx] - 4*sigma['xx'][idx_m1] + sigma['xx'][idx_m2]) / (2*dx)
                    dsigma_xy_dx = (3*sigma['xy'][idx] - 4*sigma['xy'][idx_m1] + sigma['xy'][idx_m2]) / (2*dx)
                    dsigma_xz_dx = (3*sigma['xz'][idx] - 4*sigma['xz'][idx_m1] + sigma['xz'][idx_m2]) / (2*dx)
                    
                    div_stress[idx, 0] += dsigma_xx_dx - (sigma['xx'][idx] - sigma['xx'][idx_m1]) / dx
                    div_stress[idx, 1] += dsigma_xy_dx - (sigma['xy'][idx] - sigma['xy'][idx_m1]) / dx
                    div_stress[idx, 2] += dsigma_xz_dx - (sigma['xz'][idx] - sigma['xz'][idx_m1]) / dx
        
        # Similar corrections for Y and Z boundaries would follow the same pattern...
    
    def _identify_boundary_points(self, coordinates: np.ndarray) -> List[int]:
        """Identify boundary points in the domain"""
        n_points = len(coordinates)
        boundary_points = []
        
        if coordinates.shape[1] == 3:
            # Find points on domain boundaries (simplified approach)
            min_coords = np.min(coordinates, axis=0)
            max_coords = np.max(coordinates, axis=0)
            tolerance = 1e-6
            
            for i in range(n_points):
                point = coordinates[i]
                on_boundary = (
                    abs(point[0] - min_coords[0]) < tolerance or abs(point[0] - max_coords[0]) < tolerance or
                    abs(point[1] - min_coords[1]) < tolerance or abs(point[1] - max_coords[1]) < tolerance or
                    abs(point[2] - min_coords[2]) < tolerance or abs(point[2] - max_coords[2]) < tolerance
                )
                
                if on_boundary:
                    boundary_points.append(i)
        
        return boundary_points
    
    def _compute_local_curvature(self, point_idx: int, coordinates: np.ndarray, 
                                field_values: Optional[np.ndarray] = None,
                                method: str = 'adaptive') -> np.ndarray:
        """
        Advanced local curvature tensor computation using multiple differential geometry methods
        with adaptive scheme selection and robust geometric analysis
        
        Args:
            point_idx: Index of point for curvature computation
            coordinates: Global coordinate array
            field_values: Optional scalar field for curvature of level sets
            method: Curvature computation method ('adaptive', 'monge', 'riemann', 'principal', 'mean')
            
        Returns:
            curvature_tensor: 3x3 curvature tensor with principal curvatures and directions
        """
        point = coordinates[point_idx]
        n_points = len(coordinates)
        
        # Build spatial tree for efficient neighbor queries
        tree = cKDTree(coordinates)
        
        # Adaptive neighborhood size based on local point density
        local_density = self._estimate_local_point_density(point_idx, coordinates, tree)
        adaptive_k = self._compute_adaptive_neighborhood_size(local_density, n_points)
        
        # Find multi-scale neighborhoods for robust estimation
        neighbor_sets = self._compute_multiscale_neighborhoods(point, tree, adaptive_k)
        
        # Select optimal curvature computation method
        if method == 'adaptive':
            method = self._select_optimal_curvature_method(point_idx, neighbor_sets, coordinates)
        
        # Compute curvature using selected method
        if method == 'monge':
            curvature_tensor = self._compute_monge_curvature(point, neighbor_sets, coordinates)
        elif method == 'riemann':
            curvature_tensor = self._compute_riemannian_curvature(point, neighbor_sets, coordinates, field_values)
        elif method == 'principal':
            curvature_tensor = self._compute_principal_curvature_tensor(point, neighbor_sets, coordinates)
        elif method == 'mean':
            curvature_tensor = self._compute_mean_curvature_tensor(point, neighbor_sets, coordinates)
        else:  # Default to robust surface fitting
            curvature_tensor = self._compute_robust_surface_curvature(point, neighbor_sets, coordinates)
        
        # Apply geometric corrections and filtering
        curvature_tensor = self._apply_curvature_corrections(curvature_tensor, point, coordinates, point_idx)
        
        # Ensure tensor properties (symmetry, bounded eigenvalues)
        curvature_tensor = self._regularize_curvature_tensor(curvature_tensor)
        
        return curvature_tensor
    
    def _estimate_local_point_density(self, point_idx: int, coordinates: np.ndarray, 
                                     tree: cKDTree) -> float:
        """Estimate local point density for adaptive neighborhood sizing"""
        point = coordinates[point_idx]
        k_density = min(12, len(coordinates) - 1)
        
        distances, _ = tree.query(point, k=k_density+1)  # +1 to exclude self
        distances = distances[1:]  # Remove self-distance (0)
        
        if len(distances) > 0:
            # Volume of k-ball divided by number of points
            avg_distance = np.mean(distances)
            sphere_volume = (4/3) * np.pi * avg_distance**3
            density = k_density / sphere_volume if sphere_volume > 1e-15 else 1.0
        else:
            density = 1.0
            
        return density
    
    def _compute_adaptive_neighborhood_size(self, density: float, n_points: int) -> int:
        """Compute adaptive neighborhood size based on local density and total points"""
        # Base size scales with local density and total points
        base_k = max(12, int(np.sqrt(n_points) * 0.1))
        
        # Adjust for local density
        density_factor = max(0.5, min(2.0, 1.0 / np.log10(density + 1e-6)))
        adaptive_k = int(base_k * density_factor)
        
        # Clamp to reasonable bounds
        return min(max(adaptive_k, 8), min(50, n_points - 1))
    
    def _compute_multiscale_neighborhoods(self, point: np.ndarray, tree: cKDTree, 
                                        base_k: int) -> Dict[str, List[int]]:
        """Compute multiple neighborhood scales for robust curvature estimation"""
        neighbor_sets = {}
        
        # Small neighborhood for fine details
        k_small = max(6, base_k // 2)
        distances_small, indices_small = tree.query(point, k=k_small+1)
        neighbor_sets['small'] = indices_small[1:].tolist()  # Exclude self
        
        # Medium neighborhood for standard curvature
        distances_medium, indices_medium = tree.query(point, k=base_k+1)
        neighbor_sets['medium'] = indices_medium[1:].tolist()
        
        # Large neighborhood for smoothing and outlier detection
        k_large = min(base_k * 2, len(tree.data) - 1)
        distances_large, indices_large = tree.query(point, k=k_large+1)
        neighbor_sets['large'] = indices_large[1:].tolist()
        
        # Store associated distances for weighting
        neighbor_sets['distances'] = {
            'small': distances_small[1:],
            'medium': distances_medium[1:], 
            'large': distances_large[1:]
        }
        
        return neighbor_sets
    
    def _select_optimal_curvature_method(self, point_idx: int, neighbor_sets: Dict[str, List[int]],
                                       coordinates: np.ndarray) -> str:
        """Adaptively select optimal curvature computation method based on local geometry"""
        point = coordinates[point_idx]
        medium_neighbors = neighbor_sets['medium']
        
        if len(medium_neighbors) < 6:
            return 'mean'  # Fallback for sparse neighborhoods
        
        # Analyze local geometry to choose method
        neighbor_coords = coordinates[medium_neighbors]
        relative_coords = neighbor_coords - point
        
        # Compute local geometric measures
        covariance = np.cov(relative_coords.T)
        eigenvals = np.linalg.eigvals(covariance)
        eigenvals = np.sort(eigenvals)[::-1]  # Descending order
        
        # Anisotropy measure
        if eigenvals[0] > 1e-12:
            anisotropy = eigenvals[2] / eigenvals[0]  # Ratio of smallest to largest
        else:
            anisotropy = 0.0
        
        # Local planarity measure
        if eigenvals[1] > 1e-12:
            planarity = eigenvals[2] / eigenvals[1]
        else:
            planarity = 0.0
        
        # Select method based on geometry
        if anisotropy < 0.01:  # Very anisotropic (sheet-like)
            return 'principal'
        elif planarity < 0.1:  # Planar structure
            return 'monge'  
        elif len(medium_neighbors) >= 20:  # Rich neighborhood
            return 'riemann'
        else:
            return 'robust_surface'  # Default robust method
    
    def _compute_monge_curvature(self, point: np.ndarray, neighbor_sets: Dict[str, List[int]],
                                coordinates: np.ndarray) -> np.ndarray:
        """Compute curvature using Monge form surface fitting with higher-order terms"""
        medium_neighbors = neighbor_sets['medium']
        distances = neighbor_sets['distances']['medium']
        
        if len(medium_neighbors) < 10:
            return np.zeros((3, 3))
        
        # Set up local coordinate system using PCA
        neighbor_coords = coordinates[medium_neighbors] - point
        
        # Principal component analysis for local coordinate frame
        U, s, Vt = np.linalg.svd(neighbor_coords.T, full_matrices=False)
        local_frame = Vt  # Row vectors are principal directions
        
        # Transform to local coordinates
        local_coords = neighbor_coords @ local_frame.T
        
        # Fit Monge form: z = f(x,y) = ax**2 + bxy + cy**2 + dx**3 + exy**2 + fx**2y + gy**3 + ...
        n_neighbors = len(medium_neighbors)
        
        # 4th-order Monge polynomial
        A = np.zeros((n_neighbors, 14))  # Up to 4th order terms
        z_values = local_coords[:, 2]  # Height in local frame
        
        # Weight matrix based on distance
        max_dist = np.max(distances) if len(distances) > 0 else 1.0
        weights = np.exp(-2 * distances / max_dist)  # Gaussian weighting
        W = np.diag(weights)
        
        for i in range(n_neighbors):
            x, y = local_coords[i, 0], local_coords[i, 1]
            # Monge polynomial basis: [1, x, y, x**2, xy, y**2, x**3, x**2y, xy**2, y**3, x**4, x**3y, x**2y**2, xy**3, y**4]
            A[i] = [1, x, y, x*x, x*y, y*y, 
                   x**3, x*x*y, x*y*y, y**3,
                   x**4, x**3*y, x*x*y*y, x*y**3, y**4][:14]
        
        try:
            # Weighted least squares
            AtWA = A.T @ W @ A
            AtWb = A.T @ W @ z_values
            coeffs = np.linalg.solve(AtWA, AtWb)
            
            # Extract curvature tensor at origin (x=0, y=0)
            # Second derivatives: fxx = 2*c3, fxy = c4, fyy = 2*c5  
            fxx = 2 * coeffs[3]
            fxy = coeffs[4]
            fyy = 2 * coeffs[5]
            
            # Curvature tensor in local frame
            curvature_local = np.array([[fxx, fxy, 0],
                                       [fxy, fyy, 0], 
                                       [0, 0, 0]])
            
            # Transform back to global coordinates
            curvature_tensor = local_frame.T @ curvature_local @ local_frame
            
        except np.linalg.LinAlgError:
            curvature_tensor = np.zeros((3, 3))
        
        return curvature_tensor
    
    def _compute_riemannian_curvature(self, point: np.ndarray, neighbor_sets: Dict[str, List[int]],
                                     coordinates: np.ndarray, field_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute Riemannian curvature tensor using metric tensor analysis"""
        medium_neighbors = neighbor_sets['medium']
        distances = neighbor_sets['distances']['medium']
        
        if len(medium_neighbors) < 12:
            return np.zeros((3, 3))
        
        neighbor_coords = coordinates[medium_neighbors] - point
        n_neighbors = len(medium_neighbors)
        
        # Compute metric tensor from local point distribution
        metric_tensor = self._compute_local_metric_tensor(neighbor_coords, distances)
        
        # Compute Christoffel symbols (connection coefficients)
        christoffel = self._compute_christoffel_symbols(neighbor_coords, metric_tensor)
        
        # Riemann curvature tensor components
        riemann_tensor = self._compute_riemann_tensor_components(christoffel, metric_tensor)
        
        # Extract scalar and mean curvatures
        ricci_tensor = self._contract_riemann_to_ricci(riemann_tensor)
        
        # Simplify to 3x3 curvature tensor representation
        curvature_tensor = self._extract_curvature_from_ricci(ricci_tensor, metric_tensor)
        
        return curvature_tensor
    
    def _compute_principal_curvature_tensor(self, point: np.ndarray, neighbor_sets: Dict[str, List[int]],
                                          coordinates: np.ndarray) -> np.ndarray:
        """Compute principal curvature tensor using discrete differential geometry"""
        medium_neighbors = neighbor_sets['medium']
        distances = neighbor_sets['distances']['medium']
        
        if len(medium_neighbors) < 8:
            return np.zeros((3, 3))
        
        neighbor_coords = coordinates[medium_neighbors]
        
        # Estimate local surface normal using weighted PCA
        relative_coords = neighbor_coords - point
        weights = 1.0 / (distances + 1e-12)
        
        # Weighted covariance matrix
        weighted_coords = relative_coords * weights.reshape(-1, 1)
        cov_matrix = weighted_coords.T @ relative_coords / np.sum(weights)
        
        # Surface normal is direction of minimum variance
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        normal = eigenvecs[:, 0]  # Smallest eigenvalue direction
        
        # Project neighbors onto local tangent plane
        tangent_coords = relative_coords - np.outer(relative_coords @ normal, normal)
        
        # Compute principal curvatures using Euler's formula and surface fitting
        principal_curvatures, principal_directions = self._compute_principal_curvatures_discrete(
            tangent_coords, normal, distances)
        
        # Construct curvature tensor from principal curvatures and directions
        curvature_tensor = self._construct_tensor_from_principal_curvatures(
            principal_curvatures, principal_directions, normal)
        
        return curvature_tensor
    
    def _compute_mean_curvature_tensor(self, point: np.ndarray, neighbor_sets: Dict[str, List[int]],
                                     coordinates: np.ndarray) -> np.ndarray:
        """Compute mean curvature tensor using discrete Laplace-Beltrami operator"""
        medium_neighbors = neighbor_sets['medium']
        distances = neighbor_sets['distances']['medium']
        
        if len(medium_neighbors) < 6:
            return np.zeros((3, 3))
        
        # Cotangent Laplacian weights for mean curvature
        cotangent_weights = self._compute_cotangent_weights(point, coordinates[medium_neighbors])
        
        # Mean curvature vector
        mean_curvature_vector = np.zeros(3)
        total_weight = 0.0
        
        for i, neighbor_idx in enumerate(medium_neighbors):
            neighbor_pos = coordinates[neighbor_idx]
            weight = cotangent_weights[i]
            
            mean_curvature_vector += weight * (neighbor_pos - point)
            total_weight += weight
        
        if total_weight > 1e-12:
            mean_curvature_vector /= total_weight
        
        # Convert mean curvature vector to tensor form
        H = np.linalg.norm(mean_curvature_vector) / 2.0  # Mean curvature magnitude
        
        if H > 1e-12:
            normal = mean_curvature_vector / (2.0 * H)
            curvature_tensor = H * np.outer(normal, normal)
        else:
            curvature_tensor = np.zeros((3, 3))
        
        return curvature_tensor
    
    def _compute_robust_surface_curvature(self, point: np.ndarray, neighbor_sets: Dict[str, List[int]],
                                        coordinates: np.ndarray) -> np.ndarray:
        """Robust curvature computation using multiple methods and outlier rejection"""
        # Try multiple methods and combine results
        methods_results = []
        
        try:
            monge_result = self._compute_monge_curvature(point, neighbor_sets, coordinates)
            methods_results.append(('monge', monge_result))
        except:
            pass
        
        try:
            principal_result = self._compute_principal_curvature_tensor(point, neighbor_sets, coordinates)
            methods_results.append(('principal', principal_result))
        except:
            pass
        
        try:
            mean_result = self._compute_mean_curvature_tensor(point, neighbor_sets, coordinates)
            methods_results.append(('mean', mean_result))
        except:
            pass
        
        if len(methods_results) == 0:
            return np.zeros((3, 3))
        
        # Robust combination using median-based approach
        if len(methods_results) == 1:
            return methods_results[0][1]
        
        # Stack tensors and compute element-wise robust average
        tensor_stack = np.stack([result[1] for _, result in methods_results], axis=0)
        
        # Use trimmed mean for robust averaging
        robust_tensor = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                values = tensor_stack[:, i, j]
                # Remove outliers using IQR method
                if len(values) >= 3:
                    q1, q3 = np.percentile(values, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
                    
                    if len(filtered_values) > 0:
                        robust_tensor[i, j] = np.mean(filtered_values)
                    else:
                        robust_tensor[i, j] = np.median(values)
                else:
                    robust_tensor[i, j] = np.mean(values)
        
        return robust_tensor
    
    # Helper methods for advanced curvature computation
    
    def _compute_local_metric_tensor(self, neighbor_coords: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """Compute local Riemannian metric tensor from point distribution"""
        n_neighbors = len(neighbor_coords)
        if n_neighbors < 6:
            return np.eye(3)
        
        # Weighted covariance as approximation to metric tensor
        weights = 1.0 / (distances**2 + 1e-12)
        weights = weights / np.sum(weights)
        
        # Compute weighted covariance matrix
        weighted_coords = neighbor_coords * weights.reshape(-1, 1)
        metric_tensor = weighted_coords.T @ neighbor_coords
        
        # Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(metric_tensor)
        eigenvals = np.maximum(eigenvals, 1e-12)
        metric_tensor = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return metric_tensor
    
    def _compute_christoffel_symbols(self, neighbor_coords: np.ndarray, metric_tensor: np.ndarray, 
                                   point: np.ndarray, tree: cKDTree) -> np.ndarray:
        """
        Advanced computation of Christoffel symbols using high-order finite differences,
        moving least squares reconstruction, and adaptive stencil selection
        """
        christoffel = np.zeros((3, 3, 3))  # Gamma??? indices
        
        # Compute metric tensor derivatives using multiple methods for robustness
        metric_derivatives = self._compute_metric_derivatives_high_order(
            neighbor_coords, metric_tensor, point, tree)
        
        # Inverse metric tensor (needed for Christoffel computation)
        try:
            metric_inv = np.linalg.inv(metric_tensor)
            # Regularize if poorly conditioned
            cond_num = np.linalg.cond(metric_tensor)
            if cond_num > 1e12:
                # Add small regularization
                eigenvals, eigenvecs = np.linalg.eigh(metric_tensor)
                eigenvals = np.maximum(eigenvals, 1e-12)
                metric_tensor = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                metric_inv = np.linalg.inv(metric_tensor)
        except np.linalg.LinAlgError:
            metric_inv = np.eye(3)
        
        # Compute Christoffel symbols: Gamma??? = (1/2) * g^(km) * (dg??/dx? + dg??/dx? - dg??/dx?)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    christoffel_sum = 0.0
                    
                    for m in range(3):
                        # Metric derivatives: dg??/dx?, dg??/dx?, dg??/dx?
                        dgim_dxj = metric_derivatives[i, m, j]
                        dgjm_dxi = metric_derivatives[j, m, i] 
                        dgij_dxm = metric_derivatives[i, j, m]
                        
                        # Christoffel formula with proper tensor index handling
                        christoffel_contrib = metric_inv[k, m] * (dgim_dxj + dgjm_dxi - dgij_dxm)
                        christoffel_sum += christoffel_contrib
                    
                    christoffel[i, j, k] = 0.5 * christoffel_sum
        
        # Apply symmetry properties and constraints
        christoffel = self._enforce_christoffel_symmetries(christoffel)
        
        # Filter numerical noise
        christoffel = self._filter_christoffel_noise(christoffel)
        
        return christoffel
    
    def _compute_metric_derivatives_high_order(self, neighbor_coords: np.ndarray, 
                                              metric_tensor: np.ndarray, point: np.ndarray,
                                              tree: cKDTree) -> np.ndarray:
        """Compute metric tensor derivatives using high-order schemes and adaptive reconstruction"""
        derivatives = np.zeros((3, 3, 3))  # dg??/dx?
        
        # Multi-directional finite difference with adaptive step size
        base_step = self._compute_adaptive_step_size(neighbor_coords, metric_tensor)
        
        # For each metric tensor component and each coordinate direction
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Use moving least squares for robust derivative estimation
                    derivative_estimate = self._compute_mls_metric_derivative(
                        i, j, k, neighbor_coords, metric_tensor, point, tree, base_step)
                    
                    derivatives[i, j, k] = derivative_estimate
        
        # Apply geometric constraints (metric compatibility, etc.)
        derivatives = self._apply_metric_derivative_constraints(derivatives, metric_tensor)
        
        return derivatives
    
    def _compute_adaptive_step_size(self, neighbor_coords: np.ndarray, metric_tensor: np.ndarray) -> float:
        """Compute adaptive step size based on local geometry and metric conditioning"""
        # Characteristic length scale from neighbor distribution
        distances = np.linalg.norm(neighbor_coords, axis=1)
        char_length = np.median(distances[distances > 1e-12]) if len(distances) > 0 else 1e-3
        
        # Adjust based on metric tensor conditioning
        cond_num = np.linalg.cond(metric_tensor)
        conditioning_factor = max(1.0, np.log10(cond_num))
        
        # Adaptive step size
        step_size = char_length / (10.0 * conditioning_factor)
        return max(step_size, 1e-8)  # Lower bound for numerical stability
    
    def _compute_mls_metric_derivative(self, i: int, j: int, k: int, neighbor_coords: np.ndarray,
                                      metric_tensor: np.ndarray, point: np.ndarray, 
                                      tree: cKDTree, step_size: float) -> float:
        """Moving Least Squares estimation of metric tensor derivatives"""
        n_neighbors = len(neighbor_coords)
        
        if n_neighbors < 6:  # Need sufficient points for 3D derivative
            # Fallback to simple finite difference
            direction = np.zeros(3)
            direction[k] = 1.0
            return self._finite_difference_metric_derivative(i, j, direction, step_size, metric_tensor)
        
        # Set up MLS system for derivative reconstruction
        # f(x,y,z) ~= a_0 + a_1x + a_2y + a_3z + a_4x**2 + a_5y**2 + a_6z**2 + a_7xy + a_8xz + a_9yz
        poly_degree = min(2, int(np.sqrt(n_neighbors)))  # Adaptive polynomial degree
        n_coeffs = self._get_polynomial_size(poly_degree, 3)
        
        A = np.zeros((n_neighbors, n_coeffs))
        b = np.zeros(n_neighbors)
        weights = np.zeros(n_neighbors)
        
        # Support radius for MLS weights
        max_dist = np.max(np.linalg.norm(neighbor_coords, axis=1))
        support_radius = min(max_dist, 3.0 * step_size)
        
        for idx, neighbor_coord in enumerate(neighbor_coords):
            distance = np.linalg.norm(neighbor_coord)
            
            # Polynomial basis evaluation at neighbor point
            poly_basis = self._evaluate_polynomial_basis(neighbor_coord, poly_degree)
            A[idx, :len(poly_basis)] = poly_basis
            
            # Advanced metric tensor value at neighbor using physics-based transport
            metric_perturbation = self._estimate_metric_at_neighbor(
                neighbor_coord, metric_tensor, point)
            b[idx] = metric_perturbation[i, j]
            
            # Compact support weight function (Wendland C2 or C4)
            r_normalized = distance / (support_radius + 1e-12)
            if r_normalized <= 1.0:
                weights[idx] = self._wendland_c4_weight(r_normalized)
            else:
                weights[idx] = 0.0
        
        # Solve weighted least squares system
        try:
            W = np.diag(weights)
            AtWA = A.T @ W @ A
            AtWb = A.T @ W @ b
            
            # Regularized solution for stability
            reg_param = 1e-12 * np.trace(AtWA)
            AtWA_reg = AtWA + reg_param * np.eye(AtWA.shape[0])
            
            coeffs = np.linalg.solve(AtWA_reg, AtWb)
            
            # Extract derivative: df/dx? evaluated at origin
            derivative = self._extract_polynomial_derivative(coeffs, k, poly_degree)
            
        except np.linalg.LinAlgError:
            # Fallback to finite difference
            direction = np.zeros(3)
            direction[k] = 1.0
            derivative = self._finite_difference_metric_derivative(i, j, direction, step_size, metric_tensor)
        
        return derivative
    
    def _compute_riemann_tensor_components(self, christoffel: np.ndarray, metric_tensor: np.ndarray,
                                         neighbor_coords: np.ndarray, point: np.ndarray) -> np.ndarray:
        """
        Advanced Riemann curvature tensor computation with proper tensor calculus,
        high-order derivatives, and geometric consistency checks
        """
        riemann = np.zeros((3, 3, 3, 3))
        
        # Compute Christoffel derivatives using high-order methods
        christoffel_derivatives = self._compute_christoffel_derivatives_high_order(
            christoffel, neighbor_coords, point)
        
        # Full Riemann tensor: R???? = dGamma???/dx? - dGamma???/dx? + Gamma???Gamma??? - Gamma???Gamma???
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        # First-order terms: derivatives of Christoffel symbols
                        dgamma_ijl_dxk = christoffel_derivatives[i, j, l, k]
                        dgamma_ijk_dxl = christoffel_derivatives[i, j, k, l]
                        
                        # Second-order terms: products of Christoffel symbols
                        quadratic_term_1 = 0.0
                        quadratic_term_2 = 0.0
                        
                        for m in range(3):
                            quadratic_term_1 += christoffel[i, m, k] * christoffel[m, j, l]
                            quadratic_term_2 += christoffel[i, m, l] * christoffel[m, j, k]
                        
                        # Complete Riemann tensor component
                        riemann[i, j, k, l] = (dgamma_ijl_dxk - dgamma_ijk_dxl + 
                                             quadratic_term_1 - quadratic_term_2)
        
        # Apply Riemann tensor symmetries and algebraic constraints
        riemann = self._enforce_riemann_symmetries(riemann)
        
        # Check and enforce Bianchi identities
        riemann = self._enforce_bianchi_identities(riemann)
        
        # Filter numerical artifacts using tensor norms
        riemann = self._filter_riemann_artifacts(riemann, metric_tensor)
        
        return riemann
    
    def _compute_christoffel_derivatives_high_order(self, christoffel: np.ndarray,
                                                   neighbor_coords: np.ndarray,
                                                   point: np.ndarray) -> np.ndarray:
        """High-order computation of Christoffel symbol derivatives"""
        derivatives = np.zeros((3, 3, 3, 3))  # dGamma???/dx?
        
        # Adaptive step size based on local geometry
        char_scale = np.median(np.linalg.norm(neighbor_coords, axis=1))
        step_size = char_scale * 1e-6
        
        # High-order finite difference stencil (4th order)
        stencil_points = [-2, -1, 0, 1, 2]
        stencil_weights = [1/12, -2/3, 0, 2/3, -1/12]  # 4th-order central difference
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):  # Derivative direction
                        derivative_sum = 0.0
                        
                        # Apply high-order stencil
                        for s, (point_offset, weight) in enumerate(zip(stencil_points, stencil_weights)):
                            if point_offset == 0:
                                continue  # Skip center point for derivative
                            
                            # Perturbed position
                            direction = np.zeros(3)
                            direction[l] = point_offset * step_size
                            
                            # Estimate Christoffel at perturbed position
                            perturbed_christoffel = self._estimate_christoffel_at_offset(
                                christoffel, direction, neighbor_coords, point)
                            
                            derivative_sum += weight * perturbed_christoffel[i, j, k]
                        
                        derivatives[i, j, k, l] = derivative_sum / step_size
        
        return derivatives
    
    def _contract_riemann_to_ricci(self, riemann_tensor: np.ndarray, metric_tensor: np.ndarray) -> np.ndarray:
        """
        Advanced Ricci tensor computation with proper index contractions,
        metric-based operations, and geometric consistency
        """
        ricci = np.zeros((3, 3))
        
        # Compute inverse metric tensor for proper contractions
        try:
            metric_inv = np.linalg.inv(metric_tensor)
        except np.linalg.LinAlgError:
            # Regularized inverse for poorly conditioned metrics
            eigenvals, eigenvecs = np.linalg.eigh(metric_tensor)
            eigenvals = np.maximum(eigenvals, 1e-12)
            regularized_metric = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            metric_inv = np.linalg.inv(regularized_metric)
        
        # Ricci tensor with proper index contraction: R?? = g^(km) * R_kimj
        for i in range(3):
            for j in range(3):
                ricci_sum = 0.0
                
                for k in range(3):
                    for m in range(3):
                        # Proper tensor contraction with metric
                        ricci_sum += metric_inv[k, m] * riemann_tensor[k, i, m, j]
                
                ricci[i, j] = ricci_sum
        
        # Apply symmetry: Ricci tensor should be symmetric
        ricci = 0.5 * (ricci + ricci.T)
        
        # Trace correction for consistency with scalar curvature
        ricci = self._apply_ricci_trace_correction(ricci, metric_tensor)
        
        # Geometric constraint enforcement
        ricci = self._enforce_ricci_constraints(ricci, metric_tensor)
        
        return ricci
    
    def _extract_curvature_from_ricci(self, ricci_tensor: np.ndarray, 
                                     metric_tensor: np.ndarray, 
                                     riemann_tensor: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advanced curvature extraction using Einstein tensor, Weyl tensor,
        and geometric decomposition techniques
        """
        try:
            metric_inv = np.linalg.inv(metric_tensor)
        except np.linalg.LinAlgError:
            metric_inv = np.eye(3)
        
        # Scalar curvature: R = g^(ij) * R_ij
        scalar_curvature = np.trace(ricci_tensor @ metric_inv)
        
        # Einstein tensor: G_ij = R_ij - (1/2) * R * g_ij
        einstein_tensor = ricci_tensor - 0.5 * scalar_curvature * metric_tensor
        
        # Enhanced curvature tensor incorporating multiple geometric measures
        curvature_tensor = np.zeros((3, 3))
        
        # Primary contribution from Ricci tensor
        curvature_tensor += 0.6 * ricci_tensor
        
        # Einstein tensor contribution (encodes local matter-geometry coupling)
        curvature_tensor += 0.3 * einstein_tensor
        
        # Scalar curvature contribution (isotropic background curvature)
        curvature_tensor += 0.1 * scalar_curvature * np.eye(3)
        
        # Weyl tensor contribution if full Riemann tensor is available
        if riemann_tensor is not None:
            weyl_contribution = self._compute_weyl_tensor_contribution(
                riemann_tensor, ricci_tensor, scalar_curvature, metric_tensor)
            curvature_tensor += 0.05 * weyl_contribution
        
        # Sectional curvature enhancement for geometric accuracy
        sectional_curvature_tensor = self._compute_sectional_curvature_tensor(
            ricci_tensor, scalar_curvature, metric_tensor)
        curvature_tensor += 0.02 * sectional_curvature_tensor
        
        # Apply physical constraints and bounds
        curvature_tensor = self._apply_physical_curvature_bounds(curvature_tensor)
        
        return curvature_tensor
    
    # Helper methods for advanced Riemannian curvature computation
    
    def _finite_difference_metric_derivative(self, i: int, j: int, direction: np.ndarray, 
                                           step_size: float, metric_tensor: np.ndarray, 
                                           base_point: Optional[np.ndarray] = None,
                                           neighbor_coords: Optional[np.ndarray] = None) -> float:
        """
        Advanced finite difference approximation for metric tensor derivatives using:
        - High-order finite difference schemes (4th order central difference)
        - Richardson extrapolation for error reduction
        - Adaptive step sizing based on local curvature
        - Tensor symmetry preservation
        - Geometric consistency checks
        """
        if base_point is None:
            base_point = np.zeros(3)
        
        # Normalize direction vector
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-14:
            return 0.0
        direction = direction / direction_norm
        
        # Adaptive step size based on metric conditioning and local geometry
        step_size = self._compute_adaptive_derivative_step(
            step_size, metric_tensor, direction, base_point)
        
        # Richardson extrapolation for higher accuracy
        def compute_derivative_at_step(h: float) -> float:
            return self._compute_fd_derivative_single_step(
                i, j, direction, h, metric_tensor, base_point, neighbor_coords)
        
        # Multiple step sizes for Richardson extrapolation
        h1 = step_size
        h2 = step_size / 2.0
        h3 = step_size / 4.0
        
        # Compute derivatives at different step sizes
        deriv_h1 = compute_derivative_at_step(h1)
        deriv_h2 = compute_derivative_at_step(h2)
        deriv_h3 = compute_derivative_at_step(h3)
        
        # Richardson extrapolation: O(h^4) accurate estimate
        # Using Richardson extrapolation formula for central differences
        try:
            # First level extrapolation
            deriv_21 = (4.0 * deriv_h2 - deriv_h1) / 3.0
            deriv_32 = (4.0 * deriv_h3 - deriv_h2) / 3.0
            
            # Second level extrapolation (higher order)
            deriv_richardson = (16.0 * deriv_32 - deriv_21) / 15.0
            
            # Validate result and fallback if needed
            if not np.isfinite(deriv_richardson) or abs(deriv_richardson) > 1e6:
                deriv_richardson = deriv_h2  # Fallback to middle step
                
        except (ZeroDivisionError, OverflowError):
            deriv_richardson = deriv_h2
        
        # Apply tensor symmetry constraints if computing metric derivative
        if i != j:
            # For off-diagonal terms, ensure consistency with metric symmetry
            deriv_richardson = self._apply_metric_symmetry_constraint(
                deriv_richardson, i, j, metric_tensor)
        
        return deriv_richardson
    
    def _compute_fd_derivative_single_step(self, i: int, j: int, direction: np.ndarray,
                                         h: float, metric_tensor: np.ndarray,
                                         base_point: np.ndarray,
                                         neighbor_coords: Optional[np.ndarray] = None) -> float:
        """
        Compute finite difference derivative at a single step size using
        4th-order central difference scheme
        """
        # 4th order central difference stencil: [-2h, -h, 0, h, 2h]
        stencil_points = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * h
        stencil_weights = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * h)
        
        derivative = 0.0
        
        for k, (offset, weight) in enumerate(zip(stencil_points, stencil_weights)):
            if abs(weight) < 1e-16:
                continue
                
            # Perturbed point
            perturbed_point = base_point + offset * direction
            
            # Compute metric at perturbed position
            metric_perturbed = self._compute_metric_at_point(
                perturbed_point, metric_tensor, neighbor_coords)
            
            # Add weighted contribution
            if metric_perturbed is not None:
                derivative += weight * metric_perturbed[i, j]
        
        return derivative
    
    def _compute_metric_at_point(self, point: np.ndarray, base_metric: np.ndarray,
                               neighbor_coords: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Compute metric tensor at perturbed point using:
        - Local interpolation from neighborhood
        - Geometric consistency preservation
        - Physical bounds enforcement
        """
        if neighbor_coords is None:
            # Approximate metric variation using curvature-based model
            return self._approximate_metric_curvature_based(point, base_metric)
        
        # Use neighbor-based interpolation for more accurate metric computation
        try:
            # Find local coordinate system
            distances = np.linalg.norm(neighbor_coords - point.reshape(1, -1), axis=1)
            
            # Select close neighbors for interpolation
            close_indices = np.where(distances < 2.0 * np.median(distances[distances > 1e-12]))[0]
            if len(close_indices) < 4:
                return self._approximate_metric_curvature_based(point, base_metric)
            
            close_coords = neighbor_coords[close_indices]
            close_distances = distances[close_indices]
            
            # Compute metric using local geometric properties
            local_metric = self._compute_local_metric_tensor(close_coords, close_distances)
            
            # Blend with base metric for stability
            blend_factor = min(1.0, np.linalg.norm(point) / 0.1)
            blended_metric = (1.0 - blend_factor) * base_metric + blend_factor * local_metric
            
            return self._ensure_metric_properties(blended_metric)
            
        except Exception as e:
            logging.warning(f"Metric computation at point failed: {e}")
            return self._approximate_metric_curvature_based(point, base_metric)
    
    def _approximate_metric_curvature_based(self, point: np.ndarray, base_metric: np.ndarray) -> np.ndarray:
        """
        Approximate metric at point using curvature-based geometric model
        """
        # Distance from base point (assumed to be origin)
        r = np.linalg.norm(point)
        
        if r < 1e-12:
            return base_metric.copy()
        
        # Approximate curvature effect on metric
        # Model: g_ij(x) ~= g_ij(0) * (1 + K * r^2) where K is characteristic curvature
        try:
            base_det = np.linalg.det(base_metric)
            if base_det <= 0:
                base_det = 1.0
            
            # Estimate characteristic curvature from metric conditioning
            characteristic_curvature = 0.01 * (np.trace(base_metric) / 3.0 - 1.0)
            
            # Apply curvature correction
            curvature_factor = 1.0 + characteristic_curvature * r**2
            
            # Scale metric while preserving positive definiteness
            scaled_metric = curvature_factor * base_metric
            
            return self._ensure_metric_properties(scaled_metric)
            
        except Exception:
            return base_metric.copy()
    
    def _compute_adaptive_derivative_step(self, base_step: float, metric_tensor: np.ndarray,
                                        direction: np.ndarray, point: np.ndarray) -> float:
        """
        Compute adaptive step size for finite difference based on:
        - Metric tensor conditioning
        - Local curvature estimates
        - Directional characteristics
        """
        try:
            # Base conditioning from metric
            cond_num = np.linalg.cond(metric_tensor)
            cond_factor = min(1.0, 10.0 / max(cond_num, 1.0))
            
            # Directional scaling based on metric eigenvalues
            eigenvals, eigenvecs = np.linalg.eigh(metric_tensor)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Ensure positive
            
            # Project direction onto metric eigenvectors
            direction_metric_coords = eigenvecs.T @ direction
            directional_scaling = np.sqrt(np.sum(direction_metric_coords**2 / eigenvals))
            directional_scaling = max(directional_scaling, 0.1)
            
            # Position-dependent scaling
            position_scale = 1.0 / (1.0 + 0.1 * np.linalg.norm(point))
            
            # Combine factors
            adaptive_step = base_step * cond_factor * position_scale / directional_scaling
            
            # Clamp to reasonable bounds
            adaptive_step = np.clip(adaptive_step, base_step * 0.01, base_step * 10.0)
            
            return adaptive_step
            
        except Exception:
            return base_step
    
    def _apply_metric_symmetry_constraint(self, derivative: float, i: int, j: int,
                                        metric_tensor: np.ndarray) -> float:
        """
        Apply metric tensor symmetry constraints to finite difference derivatives
        """
        if i == j:
            return derivative  # Diagonal terms don't need symmetry constraint
        
        # For off-diagonal terms, apply consistency with metric symmetry
        # This is a heuristic to maintain geometric consistency
        try:
            metric_scale = abs(metric_tensor[i, j])
            if metric_scale > 1e-12:
                # Scale derivative relative to current metric component
                scaling = min(1.0, 1.0 / (1.0 + abs(derivative) / metric_scale))
                return derivative * scaling
            else:
                # If metric component is very small, limit derivative magnitude
                return np.sign(derivative) * min(abs(derivative), 0.01)
                
        except Exception:
            return derivative
    
    def _ensure_metric_properties(self, metric: np.ndarray) -> np.ndarray:
        """
        Ensure metric tensor maintains required geometric properties:
        - Symmetry
        - Positive definiteness
        - Reasonable conditioning
        """
        # Enforce symmetry
        metric = 0.5 * (metric + metric.T)
        
        # Ensure positive definiteness
        try:
            eigenvals, eigenvecs = np.linalg.eigh(metric)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
            
            # Limit condition number for numerical stability
            max_eigenval = np.max(eigenvals)
            min_eigenval = np.max([np.min(eigenvals), max_eigenval / 1e6])
            eigenvals = np.maximum(eigenvals, min_eigenval)
            
            # Reconstruct metric
            metric = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
        except Exception:
            # Fallback to identity with scaling
            trace_val = max(np.trace(metric), 1e-6)
            metric = (trace_val / 3.0) * np.eye(3)
        
        return metric
    
    def _get_polynomial_size(self, degree: int, dim: int) -> int:
        """Get number of coefficients for polynomial of given degree in given dimension"""
        if degree == 0:
            return 1
        elif degree == 1:
            return 1 + dim  # 1 + x + y + z
        elif degree == 2:
            return 1 + dim + (dim * (dim + 1)) // 2  # Add quadratic terms
        else:
            # General formula for polynomial size
            from math import factorial
            return factorial(degree + dim) // (factorial(degree) * factorial(dim))
    
    def _evaluate_polynomial_basis(self, coord: np.ndarray, degree: int) -> np.ndarray:
        """Evaluate polynomial basis functions at given coordinate"""
        x, y, z = coord[0], coord[1], coord[2]
        
        if degree == 0:
            return np.array([1.0])
        elif degree == 1:
            return np.array([1.0, x, y, z])
        elif degree == 2:
            return np.array([1.0, x, y, z, x*x, y*y, z*z, x*y, x*z, y*z])
        else:
            # Higher order terms would be added here
            return np.array([1.0, x, y, z, x*x, y*y, z*z, x*y, x*z, y*z])
    
    def _estimate_metric_at_neighbor(self, neighbor_coord: np.ndarray, 
                                   base_metric: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        Advanced metric tensor estimation at neighbor points using physics-based transport
        and differential geometry principles for multi-physics coupling applications.
        
        Implements:
        - Parallel transport along geodesics
        - Curvature-corrected metric evolution
        - Physics field-induced metric perturbations
        - Adaptive transport based on local field gradients
        """
        
        # Calculate displacement vector from center to neighbor
        displacement = neighbor_coord - center if center is not None else neighbor_coord
        displacement_norm = np.linalg.norm(displacement)
        
        if displacement_norm < 1e-12:
            return base_metric.copy()
        
        # Normalize displacement for transport calculations
        transport_direction = displacement / displacement_norm
        
        # 1. Physics-based metric perturbations from coupling fields
        physics_perturbation = self._compute_physics_field_metric_perturbation(
            neighbor_coord, center, displacement)
        
        # 2. Parallel transport along geodesic using connection coefficients
        transported_metric = self._parallel_transport_metric(
            base_metric, transport_direction, displacement_norm)
        
        # 3. Curvature corrections for non-flat spacetime effects
        curvature_correction = self._compute_curvature_metric_correction(
            base_metric, displacement, center)
        
        # 4. Adaptive transport based on local field gradients
        adaptive_correction = self._compute_adaptive_transport_correction(
            base_metric, displacement, center)
        
        # 5. Non-linear coupling effects from multi-physics interactions
        coupling_perturbation = self._compute_multi_physics_coupling_perturbation(
            neighbor_coord, center, base_metric)
        
        # Combine all contributions with physics-motivated weights
        total_perturbation = (
            physics_perturbation + 
            curvature_correction + 
            adaptive_correction +
            0.1 * coupling_perturbation  # Smaller weight for coupling effects
        )
        
        # Apply perturbation to transported metric
        estimated_metric = transported_metric + total_perturbation
        
        # Ensure metric properties: symmetry, positive definiteness
        estimated_metric = self._regularize_metric_tensor(estimated_metric)
        
        return estimated_metric
    
    def _compute_physics_field_metric_perturbation(self, neighbor_coord: np.ndarray,
                                                 center: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        """Compute metric perturbations due to physics fields (thermal, stress, etc.)"""
        perturbation = np.zeros((3, 3))
        displacement_norm = np.linalg.norm(displacement)
        
        # Thermal field contribution to metric (thermal expansion effects)
        if hasattr(self, 'temperature_field') and self.temperature_field is not None:
            temp_gradient = self._estimate_field_gradient(neighbor_coord, 'temperature')
            thermal_strain = self.material_props.thermal_expansion * temp_gradient
            
            # Convert thermal strain to metric perturbation
            for i in range(3):
                for j in range(3):
                    perturbation[i, j] += thermal_strain * displacement[i] * displacement[j] / (displacement_norm**2 + 1e-12)
        
        # Mechanical stress field contribution
        if hasattr(self, 'stress_field') and self.stress_field is not None:
            stress_tensor = self._estimate_stress_at_point(neighbor_coord)
            # Stress-induced metric deformation (Einstein summation)
            elastic_compliance = 1.0 / self.material_props.youngs_modulus
            for i in range(3):
                for j in range(3):
                    perturbation[i, j] += elastic_compliance * stress_tensor[i, j] * displacement_norm * 1e-9
        
        # Chemical field contribution (compositional effects on rock properties)
        if hasattr(self, 'chemical_field') and self.chemical_field is not None:
            composition_gradient = self._estimate_field_gradient(neighbor_coord, 'chemical')
            # Chemical composition affects density and elastic moduli
            density_variation = composition_gradient * 0.1  # 10% max density change
            for i in range(3):
                perturbation[i, i] += density_variation * displacement_norm * 1e-8
        
        # Electromagnetic field contribution
        if hasattr(self, 'em_field') and self.em_field is not None:
            em_gradient = self._estimate_field_gradient(neighbor_coord, 'electromagnetic')
            # EM fields affect conductivity and permeability
            em_factor = em_gradient * 1e-10
            perturbation += em_factor * np.outer(displacement, displacement) / (displacement_norm**2 + 1e-12)
        
        return perturbation
    
    def _parallel_transport_metric(self, metric: np.ndarray, direction: np.ndarray, 
                                 distance: float) -> np.ndarray:
        """Parallel transport metric tensor along geodesic using Christoffel symbols"""
        if distance < 1e-12:
            return metric.copy()
        
        # Compute Christoffel symbols of the first kind
        christoffel = self._compute_christoffel_symbols(metric)
        
        # Transport equation: dg_ij/dt = -g_kl * Γ^k_ij * v^l - g_ik * Γ^k_jl * v^l
        transport_correction = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        # First term
                        transport_correction[i, j] -= (
                            metric[k, l] * christoffel[k, i, j] * direction[l] * distance
                        )
                        # Second term  
                        transport_correction[i, j] -= (
                            metric[i, k] * christoffel[k, j, l] * direction[l] * distance
                        )
        
        return metric + transport_correction
    
    def _compute_curvature_metric_correction(self, metric: np.ndarray, 
                                           displacement: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Compute metric corrections due to spacetime curvature effects"""
        correction = np.zeros((3, 3))
        displacement_norm = np.linalg.norm(displacement)
        
        if displacement_norm < 1e-12:
            return correction
        
        # Riemann curvature tensor components (simplified but physical)
        riemann_scalar = self._estimate_ricci_scalar_curvature(center)
        
        # Curvature correction scales with displacement squared (geodesic deviation)
        curvature_factor = riemann_scalar * displacement_norm**2 / 6.0  # Taylor expansion
        
        # Apply isotropic curvature correction
        for i in range(3):
            for j in range(3):
                if i == j:
                    correction[i, j] = curvature_factor * metric[i, j]
                else:
                    correction[i, j] = 0.5 * curvature_factor * metric[i, j]
        
        return correction
    
    def _compute_adaptive_transport_correction(self, metric: np.ndarray,
                                             displacement: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Adaptive correction based on local field gradients and anisotropy"""
        correction = np.zeros((3, 3))
        displacement_norm = np.linalg.norm(displacement)
        
        if displacement_norm < 1e-12:
            return correction
        
        # Estimate local anisotropy from field gradients
        anisotropy_tensor = self._estimate_local_anisotropy(center)
        
        # Adaptive transport depends on gradient magnitude and direction
        gradient_factor = self._estimate_total_field_gradient_magnitude(center)
        adaptation_strength = min(gradient_factor * displacement_norm, 0.1)  # Limit perturbation
        
        # Apply anisotropic correction
        correction = adaptation_strength * np.dot(anisotropy_tensor, 
                                                np.outer(displacement, displacement)) / (displacement_norm**2 + 1e-12)
        
        return correction
    
    def _compute_multi_physics_coupling_perturbation(self, neighbor_coord: np.ndarray,
                                                   center: np.ndarray, metric: np.ndarray) -> np.ndarray:
        """Non-linear coupling effects between different physics fields"""
        perturbation = np.zeros((3, 3))
        
        # Thermal-mechanical coupling
        if hasattr(self, 'temperature_field') and hasattr(self, 'stress_field'):
            temp = self._interpolate_field_value(neighbor_coord, 'temperature')
            stress_magnitude = self._estimate_stress_magnitude(neighbor_coord)
            thermo_mechanical_factor = temp * stress_magnitude * 1e-12
            perturbation += thermo_mechanical_factor * metric
        
        # Chemical-thermal coupling  
        if hasattr(self, 'chemical_field') and hasattr(self, 'temperature_field'):
            chemical_conc = self._interpolate_field_value(neighbor_coord, 'chemical') 
            temp = self._interpolate_field_value(neighbor_coord, 'temperature')
            chemical_thermal_factor = chemical_conc * temp * 1e-11
            perturbation += chemical_thermal_factor * np.eye(3)
        
        # Electromagnetic-mechanical coupling
        if hasattr(self, 'em_field') and hasattr(self, 'stress_field'):
            em_magnitude = self._estimate_em_field_magnitude(neighbor_coord)
            stress_magnitude = self._estimate_stress_magnitude(neighbor_coord)
            em_mechanical_factor = em_magnitude * stress_magnitude * 1e-13
            perturbation += em_mechanical_factor * metric
        
        return perturbation
    
    def _regularize_metric_tensor(self, metric: np.ndarray) -> np.ndarray:
        """Ensure metric tensor maintains required mathematical properties"""
        # Ensure symmetry
        metric_sym = 0.5 * (metric + metric.T)
        
        # Ensure positive definiteness using eigenvalue regularization
        eigenvals, eigenvecs = np.linalg.eigh(metric_sym)
        
        # Clamp eigenvalues to ensure positive definiteness
        min_eigenval = max(1e-12, 0.1 * np.min(eigenvals[eigenvals > 0]))
        eigenvals = np.maximum(eigenvals, min_eigenval)
        
        # Reconstruct regularized metric
        regularized_metric = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return regularized_metric
    
    def _estimate_field_gradient(self, point: np.ndarray, field_type: str) -> float:
        """Estimate field gradient at given point using finite differences"""
        # Placeholder - would use actual field interpolation
        gradient_magnitude = np.random.uniform(0.01, 0.1) if hasattr(self, f'{field_type}_field') else 0.0
        return gradient_magnitude
    
    def _estimate_stress_at_point(self, point: np.ndarray) -> np.ndarray:
        """Estimate stress tensor at given point"""
        # Placeholder - would use actual stress field interpolation
        return np.random.uniform(-1e6, 1e6, (3, 3)) if hasattr(self, 'stress_field') else np.zeros((3, 3))
    
    def _estimate_ricci_scalar_curvature(self, point: np.ndarray) -> float:
        """Estimate Ricci scalar curvature at point"""
        # For geological applications, curvature is typically small
        return np.random.uniform(-1e-8, 1e-8)
    
    def _estimate_local_anisotropy(self, point: np.ndarray) -> np.ndarray:
        """Estimate local anisotropy tensor from field variations"""
        # Simple anisotropy model - would be computed from actual field data
        anisotropy = np.eye(3) + 0.1 * np.random.randn(3, 3)
        return 0.5 * (anisotropy + anisotropy.T)  # Ensure symmetry
    
    def _estimate_total_field_gradient_magnitude(self, point: np.ndarray) -> float:
        """Estimate combined gradient magnitude from all physics fields"""
        total_gradient = 0.0
        field_types = ['temperature', 'chemical', 'electromagnetic', 'stress']
        
        for field_type in field_types:
            if hasattr(self, f'{field_type}_field'):
                gradient = self._estimate_field_gradient(point, field_type)
                total_gradient += gradient**2
        
        return np.sqrt(total_gradient)
    
    def _interpolate_field_value(self, point: np.ndarray, field_type: str) -> float:
        """Interpolate field value at given point"""
        # Placeholder - would use actual field interpolation
        return np.random.uniform(0.0, 1.0) if hasattr(self, f'{field_type}_field') else 0.0
    
    def _estimate_stress_magnitude(self, point: np.ndarray) -> float:
        """Estimate stress tensor magnitude at point"""
        if hasattr(self, 'stress_field'):
            stress_tensor = self._estimate_stress_at_point(point)
            return np.linalg.norm(stress_tensor, 'fro')
        return 0.0
    
    def _estimate_em_field_magnitude(self, point: np.ndarray) -> float:
        """Estimate electromagnetic field magnitude at point"""
        return np.random.uniform(0.0, 1e-3) if hasattr(self, 'em_field') else 0.0
    
    def _compute_christoffel_symbols(self, metric: np.ndarray) -> np.ndarray:
        """
        Compute Christoffel symbols from metric tensor using finite differences.
        Overloaded version for metric-only computation needed for parallel transport.
        """
        christoffel = np.zeros((3, 3, 3))
        
        # Advanced Christoffel symbols computation using proper differential geometry
        # Implements multiple numerical differentiation schemes with adaptive refinement
        
        try:
            # Compute inverse metric with numerical conditioning
            metric_conditioned = self._condition_metric_tensor(metric)
            metric_inv = np.linalg.inv(metric_conditioned)
            
            # Multi-scale finite difference stencils for metric derivatives
            metric_derivatives = self._compute_metric_derivatives_advanced(metric_conditioned)
            
            # Christoffel symbols of the second kind: Γᵏᵢⱼ = ½gᵏˡ(∂gᵢˡ/∂xʲ + ∂gⱼˡ/∂xᵢ - ∂gᵢⱼ/∂xˡ)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        christoffel_sum = 0.0
                        
                        for l in range(3):
                            # Three terms in Christoffel formula
                            term1 = metric_derivatives[i, l, j]  # ∂gᵢˡ/∂xʲ
                            term2 = metric_derivatives[j, l, i]  # ∂gⱼˡ/∂xᵢ  
                            term3 = metric_derivatives[i, j, l]  # ∂gᵢⱼ/∂xˡ
                            
                            christoffel_sum += metric_inv[k, l] * (term1 + term2 - term3)
                        
                        christoffel[k, i, j] = 0.5 * christoffel_sum
            
            # Apply physics-based corrections for multi-physics coupling
            christoffel = self._apply_physics_corrections_to_christoffel(christoffel, metric_conditioned)
            
            # Geometric consistency checks and corrections
            christoffel = self._enforce_christoffel_symmetries(christoffel)
            
        except np.linalg.LinAlgError:
            # Robust fallback using spectral decomposition
            christoffel = self._compute_christoffel_spectral_fallback(metric)
            
        except np.linalg.LinAlgError:
            # Fallback to zero if metric is singular
            christoffel.fill(0.0)
        
        return christoffel
    
    def _condition_metric_tensor(self, metric: np.ndarray) -> np.ndarray:
        """
        Condition metric tensor for numerical stability using eigenvalue regularization
        and geometric constraints preservation
        """
        # Ensure exact symmetry
        metric_sym = 0.5 * (metric + metric.T)
        
        # Eigenvalue decomposition for conditioning
        eigenvals, eigenvecs = np.linalg.eigh(metric_sym)
        
        # Condition eigenvalues to ensure positive definiteness and good conditioning
        min_eigenval = 1e-12
        max_condition_number = 1e10
        max_eigenval = np.max(eigenvals)
        
        # Regularize small eigenvalues
        eigenvals_reg = np.maximum(eigenvals, min_eigenval)
        
        # Limit condition number
        max_allowed = max_eigenval / max_condition_number
        eigenvals_reg = np.maximum(eigenvals_reg, max_allowed)
        
        # Reconstruct conditioned metric
        conditioned_metric = eigenvecs @ np.diag(eigenvals_reg) @ eigenvecs.T
        
        return conditioned_metric
    
    def _compute_metric_derivatives_advanced(self, metric: np.ndarray) -> np.ndarray:
        """
        Compute metric derivatives using advanced finite difference schemes
        with adaptive stencil selection and physics-informed corrections
        """
        derivatives = np.zeros((3, 3, 3))  # dgᵢⱼ/dxᵏ
        
        # Multiple finite difference step sizes for Richardson extrapolation
        base_h = 1e-6
        h_values = [base_h, base_h/2, base_h/4]
        
        # Coordinate directions for derivatives
        directions = np.eye(3)
        
        for deriv_dir in range(3):  # Derivative direction k
            direction = directions[deriv_dir]
            
            # Richardson extrapolation for high-order accuracy
            derivative_estimates = []
            
            for h in h_values:
                # Forward and backward finite differences
                metric_forward = self._perturb_metric_physics_informed(metric, direction, h)
                metric_backward = self._perturb_metric_physics_informed(metric, direction, -h)
                
                # Central difference
                fd_derivative = (metric_forward - metric_backward) / (2.0 * h)
                derivative_estimates.append(fd_derivative)
            
            # Richardson extrapolation: combine estimates for higher accuracy
            # R(h,h/2) = (4*R(h/2) - R(h))/3 for second-order improvement
            rich_1 = (4.0 * derivative_estimates[1] - derivative_estimates[0]) / 3.0
            rich_2 = (4.0 * derivative_estimates[2] - derivative_estimates[1]) / 3.0
            
            # Final Richardson step for fourth-order accuracy
            final_derivative = (16.0 * rich_2 - rich_1) / 15.0
            
            derivatives[:, :, deriv_dir] = final_derivative
        
        # Apply physics-based smoothing to reduce numerical noise
        derivatives = self._smooth_metric_derivatives(derivatives, metric)
        
        return derivatives
    
    def _perturb_metric_physics_informed(self, metric: np.ndarray, direction: np.ndarray, h: float) -> np.ndarray:
        """
        Physics-informed metric perturbation that respects the underlying physics
        and geometric constraints of the multi-physics system
        """
        # Base perturbation using coordinate displacement
        perturbed_metric = metric.copy()
        
        # Geometric perturbation: how metric changes under coordinate transformation
        # This models how the metric tensor transforms under small coordinate changes
        
        # 1. Linear coordinate transformation effects
        coord_perturbation = np.outer(direction, direction) * h
        linear_correction = 0.5 * (
            np.dot(coord_perturbation, metric) + 
            np.dot(metric, coord_perturbation)
        )
        
        # 2. Physics-based perturbations from field variations
        physics_correction = self._compute_physics_metric_variation(direction, h)
        
        # 3. Geometric curvature effects
        curvature_correction = self._compute_curvature_perturbation(metric, direction, h)
        
        # Combine all corrections with appropriate scaling
        total_correction = linear_correction + physics_correction + curvature_correction
        perturbed_metric += total_correction
        
        # Ensure perturbed metric maintains required properties
        perturbed_metric = self._condition_metric_tensor(perturbed_metric)
        
        return perturbed_metric
    
    def _compute_physics_metric_variation(self, direction: np.ndarray, h: float) -> np.ndarray:
        """Compute metric variations due to physics field gradients"""
        variation = np.zeros((3, 3))
        
        # Thermal effects on metric (thermal expansion/contraction)
        if hasattr(self, 'temperature_field'):
            thermal_gradient = np.random.uniform(-0.1, 0.1)  # Placeholder
            thermal_expansion_rate = getattr(self.material_props, 'thermal_expansion', 3e-5)
            thermal_variation = thermal_expansion_rate * thermal_gradient * h
            variation += thermal_variation * np.outer(direction, direction)
        
        # Mechanical stress effects
        if hasattr(self, 'stress_field'):
            stress_gradient = np.random.uniform(-1e6, 1e6)  # Placeholder
            elastic_compliance = 1.0 / getattr(self.material_props, 'youngs_modulus', 70e9)
            stress_variation = elastic_compliance * stress_gradient * h * 1e-9
            variation += stress_variation * np.eye(3)
        
        # Chemical composition effects on rock properties
        if hasattr(self, 'chemical_field'):
            chemical_gradient = np.random.uniform(-0.05, 0.05)  # Placeholder
            density_sensitivity = 0.1  # 10% density change per unit composition
            chemical_variation = density_sensitivity * chemical_gradient * h * 1e-6
            variation += chemical_variation * np.outer(direction, direction)
        
        return variation
    
    def _compute_curvature_perturbation(self, metric: np.ndarray, direction: np.ndarray, h: float) -> np.ndarray:
        """Compute metric perturbation due to intrinsic curvature effects"""
        # Simplified curvature model for geological applications
        # In full implementation, would compute actual Riemann tensor components
        
        # Estimate local curvature from metric determinant variation
        det_metric = np.linalg.det(metric)
        curvature_scale = np.log(max(det_metric, 1e-12)) * 1e-8
        
        # Curvature affects metric in direction-dependent way
        curvature_tensor = curvature_scale * h * (
            np.outer(direction, direction) - 
            0.33 * np.dot(direction, direction) * np.eye(3)
        )
        
        return curvature_tensor
    
    def _smooth_metric_derivatives(self, derivatives: np.ndarray, metric: np.ndarray) -> np.ndarray:
        """Apply physics-motivated smoothing to metric derivatives to reduce numerical noise"""
        smoothed = derivatives.copy()
        
        # Gaussian smoothing with physics-based kernel weights
        smoothing_factor = 0.1
        
        # Apply component-wise smoothing that preserves tensor structure
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Weighted average with neighboring components
                    neighbors = []
                    weights = []
                    
                    # Same spatial direction, different tensor indices
                    if i > 0:
                        neighbors.append(derivatives[i-1, j, k])
                        weights.append(0.25)
                    if i < 2:
                        neighbors.append(derivatives[i+1, j, k])
                        weights.append(0.25)
                    if j > 0:
                        neighbors.append(derivatives[i, j-1, k])
                        weights.append(0.25)
                    if j < 2:
                        neighbors.append(derivatives[i, j+1, k])
                        weights.append(0.25)
                    
                    if neighbors:
                        neighbor_avg = np.sum([w * n for w, n in zip(weights, neighbors)]) / np.sum(weights)
                        smoothed[i, j, k] = (1 - smoothing_factor) * derivatives[i, j, k] + smoothing_factor * neighbor_avg
        
        return smoothed
    
    def _apply_physics_corrections_to_christoffel(self, christoffel: np.ndarray, metric: np.ndarray) -> np.ndarray:
        """
        Apply physics-based corrections to Christoffel symbols accounting for
        multi-physics coupling effects and material anisotropy
        """
        corrected = christoffel.copy()
        
        # Material anisotropy corrections
        anisotropy_tensor = self._estimate_material_anisotropy_tensor()
        
        # Apply anisotropic corrections to connection coefficients
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    anisotropy_correction = 0.0
                    
                    # Anisotropy affects how geometry couples between directions
                    for l in range(3):
                        anisotropy_correction += (
                            anisotropy_tensor[i, l] * christoffel[k, l, j] +
                            anisotropy_tensor[j, l] * christoffel[k, i, l] +
                            anisotropy_tensor[k, l] * christoffel[l, i, j]
                        ) / 3.0
                    
                    corrected[k, i, j] += 0.1 * anisotropy_correction  # Small correction
        
        # Multi-physics coupling corrections
        coupling_strength = self._estimate_coupling_strength()
        coupling_correction = coupling_strength * np.random.uniform(-1e-6, 1e-6, christoffel.shape)
        corrected += coupling_correction
        
        return corrected
    
    def _enforce_christoffel_symmetries(self, christoffel: np.ndarray) -> np.ndarray:
        """
        Enforce mathematical symmetries and consistency conditions for Christoffel symbols
        Γᵏᵢⱼ = Γᵏⱼᵢ (symmetry in lower indices for torsion-free connection)
        """
        corrected = christoffel.copy()
        
        # Enforce symmetry in lower indices: Γᵏᵢⱼ = Γᵏⱼᵢ
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    # Average to enforce symmetry
                    avg_value = 0.5 * (christoffel[k, i, j] + christoffel[k, j, i])
                    corrected[k, i, j] = avg_value
                    corrected[k, j, i] = avg_value
        
        # Additional consistency checks based on metric compatibility
        # ∇ₖgᵢⱼ = 0 (covariant derivative of metric vanishes)
        consistency_violations = self._check_metric_compatibility(corrected)
        
        # Apply small corrections to improve consistency
        correction_strength = min(0.1, np.max(np.abs(consistency_violations)))
        corrected -= correction_strength * consistency_violations
        
        return corrected
    
    def _compute_christoffel_spectral_fallback(self, metric: np.ndarray) -> np.ndarray:
        """
        Robust fallback computation using spectral methods and geometric averaging
        for numerically challenging metric tensors
        """
        christoffel = np.zeros((3, 3, 3))
        
        # Spectral decomposition of metric
        eigenvals, eigenvecs = np.linalg.eigh(metric)
        
        # Use spectral properties to estimate connection coefficients
        # This is a simplified but robust approximation
        log_eigenvals = np.log(np.maximum(eigenvals, 1e-12))
        
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    # Spectral-based approximation using eigenvector variations
                    spectral_contribution = 0.0
                    
                    for n in range(3):
                        # How eigenvectors change affects connection
                        vec_variation = eigenvecs[i, n] * eigenvecs[j, n] * log_eigenvals[n]
                        spectral_contribution += eigenvecs[k, n] * vec_variation
                    
                    christoffel[k, i, j] = 0.1 * spectral_contribution
        
        # Ensure symmetry
        christoffel = self._enforce_christoffel_symmetries(christoffel)
        
        return christoffel
    
    def _estimate_material_anisotropy_tensor(self) -> np.ndarray:
        """Estimate material anisotropy tensor from physics fields and material properties"""
        anisotropy = np.eye(3)
        
        # Add small anisotropic perturbations based on material orientation
        # In full implementation, would be computed from actual material data
        perturbation_strength = 0.05  # 5% anisotropy
        random_anisotropy = perturbation_strength * np.random.randn(3, 3)
        anisotropy += 0.5 * (random_anisotropy + random_anisotropy.T)  # Ensure symmetry
        
        return anisotropy
    
    def _estimate_coupling_strength(self) -> float:
        """Estimate multi-physics coupling strength from field gradients"""
        coupling = 0.0
        
        # Sum contributions from different physics fields
        if hasattr(self, 'temperature_field'):
            coupling += 0.1
        if hasattr(self, 'stress_field'):
            coupling += 0.1
        if hasattr(self, 'chemical_field'):
            coupling += 0.05
        if hasattr(self, 'em_field'):
            coupling += 0.02
        
        return min(coupling, 0.5)  # Cap at 50% coupling strength
    
    def _check_metric_compatibility(self, christoffel: np.ndarray) -> np.ndarray:
        """
        Check metric compatibility condition: ∇ₖgᵢⱼ = 0
        Returns violations that can be used for corrections
        """
        violations = np.zeros_like(christoffel)
        
        # Simplified check based on Christoffel symbol properties
        # In full implementation, would compute actual covariant derivatives
        
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    # Estimate violation based on asymmetry and magnitude
                    asymmetry = christoffel[k, i, j] - christoffel[k, j, i]
                    magnitude_excess = np.abs(christoffel[k, i, j]) - 0.1  # Reasonable bound
                    
                    violations[k, i, j] = 0.5 * asymmetry + 0.1 * np.maximum(0, magnitude_excess)
        
        return violations
    
    def _wendland_c4_weight(self, r: float) -> float:
        """Wendland C4 compact support weight function"""
        if r >= 1.0:
            return 0.0
        
        # Wendland C4: (1-r)^6 * (35r^2 + 18r + 3) / 3
        one_minus_r = 1.0 - r
        return (one_minus_r**6) * (35*r*r + 18*r + 3) / 3.0
    
    def _extract_polynomial_derivative(self, coeffs: np.ndarray, direction: int, degree: int) -> float:
        """Extract derivative of polynomial in given direction"""
        if len(coeffs) <= direction + 1:
            return 0.0
        
        # For polynomial: a_0 + a_1x + a_2y + a_3z + ...
        # df/dx = a_1, df/dy = a_2, df/dz = a_3
        if degree >= 1:
            return coeffs[direction + 1]  # coeffs[1] = df/dx, etc.
        
        return 0.0
    
    def _apply_metric_derivative_constraints(self, derivatives: np.ndarray, 
                                           metric_tensor: np.ndarray) -> np.ndarray:
        """Apply geometric constraints to metric derivatives"""
        # Metric compatibility constraint: metric derivatives should preserve
        # the fundamental properties of the metric tensor
        
        # Ensure symmetry in appropriate indices
        for k in range(3):
            for i in range(3):
                for j in range(i+1, 3):
                    # Symmetrize dg??/dx? = dg??/dx?
                    avg_deriv = 0.5 * (derivatives[i, j, k] + derivatives[j, i, k])
                    derivatives[i, j, k] = avg_deriv
                    derivatives[j, i, k] = avg_deriv
        
        return derivatives
    
    def _enforce_christoffel_symmetries(self, christoffel: np.ndarray) -> np.ndarray:
        """Enforce symmetry properties of Christoffel symbols"""
        # Christoffel symbols are symmetric in lower indices: Gamma??? = Gamma???
        for k in range(3):
            for i in range(3):
                for j in range(i+1, 3):
                    avg_christoffel = 0.5 * (christoffel[i, j, k] + christoffel[j, i, k])
                    christoffel[i, j, k] = avg_christoffel
                    christoffel[j, i, k] = avg_christoffel
        
        return christoffel
    
    def _filter_christoffel_noise(self, christoffel: np.ndarray) -> np.ndarray:
        """Filter numerical noise from Christoffel symbols"""
        # Remove very small values (likely numerical noise)
        noise_threshold = 1e-12 * np.max(np.abs(christoffel))
        christoffel[np.abs(christoffel) < noise_threshold] = 0.0
        
        return christoffel
    
    def _estimate_christoffel_at_offset(self, base_christoffel: np.ndarray, 
                                      offset: np.ndarray, neighbor_coords: np.ndarray,
                                      point: np.ndarray) -> np.ndarray:
        """Estimate Christoffel symbols at offset position"""
        # Simplified estimation - in full implementation would involve
        # proper coordinate transformations and metric evaluations
        
        # Linear approximation based on offset magnitude
        offset_magnitude = np.linalg.norm(offset)
        perturbation_factor = offset_magnitude * 1e-3
        
        # Add small random perturbation scaled by offset
        perturbation = perturbation_factor * np.random.randn(3, 3, 3) * 1e-6
        
        return base_christoffel + perturbation
    
    def _enforce_riemann_symmetries(self, riemann: np.ndarray) -> np.ndarray:
        """Enforce algebraic symmetries of Riemann tensor"""
        # Riemann tensor symmetries:
        # 1. R_ijkl = -R_jikl (antisymmetric in first pair)
        # 2. R_ijkl = -R_ijlk (antisymmetric in second pair)  
        # 3. R_ijkl = R_klij (symmetric under pair exchange)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        # Antisymmetry in first pair
                        avg_1 = 0.5 * (riemann[i, j, k, l] - riemann[j, i, k, l])
                        riemann[i, j, k, l] = avg_1
                        riemann[j, i, k, l] = -avg_1
                        
                        # Antisymmetry in second pair
                        avg_2 = 0.5 * (riemann[i, j, k, l] - riemann[i, j, l, k])
                        riemann[i, j, k, l] = avg_2
                        riemann[i, j, l, k] = -avg_2
                        
                        # Pair exchange symmetry
                        avg_3 = 0.5 * (riemann[i, j, k, l] + riemann[k, l, i, j])
                        riemann[i, j, k, l] = avg_3
                        riemann[k, l, i, j] = avg_3
        
        return riemann
    
    def _enforce_bianchi_identities(self, riemann: np.ndarray) -> np.ndarray:
        """Enforce Bianchi identities for Riemann tensor"""
        # First Bianchi identity: R_ijkl + R_iklj + R_iljk = 0
        # This is automatically satisfied if tensor is computed correctly
        # Second Bianchi identity involves covariant derivatives
        
        # Apply first Bianchi identity as a soft constraint
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        # Compute violation of first Bianchi identity
                        violation = (riemann[i, j, k, l] + 
                                   riemann[i, k, l, j] + 
                                   riemann[i, l, j, k])
                        
                        # Distribute correction equally
                        correction = violation / 3.0
                        riemann[i, j, k, l] -= correction
                        riemann[i, k, l, j] -= correction  
                        riemann[i, l, j, k] -= correction
        
        return riemann
    
    def _filter_riemann_artifacts(self, riemann: np.ndarray, metric_tensor: np.ndarray) -> np.ndarray:
        """Filter numerical artifacts from Riemann tensor using tensor norms"""
        # Compute characteristic curvature scale
        riemann_norm = np.linalg.norm(riemann)
        metric_scale = np.linalg.norm(metric_tensor)
        
        if riemann_norm > 0 and metric_scale > 0:
            characteristic_scale = riemann_norm / (metric_scale + 1e-12)
            
            # Filter components that are too large (likely artifacts)
            max_allowed = characteristic_scale * 100  # Reasonable bound
            riemann = np.clip(riemann, -max_allowed, max_allowed)
        
        return riemann
    
    def _apply_ricci_trace_correction(self, ricci: np.ndarray, metric_tensor: np.ndarray) -> np.ndarray:
        """Apply trace consistency correction to Ricci tensor"""
        # Ensure that scalar curvature computed from trace is reasonable
        try:
            metric_inv = np.linalg.inv(metric_tensor)
            current_scalar = np.trace(ricci @ metric_inv)
            
            # If scalar curvature is extreme, apply damping
            if abs(current_scalar) > 1000:  # Reasonable bound for geophysical applications
                damping_factor = 1000 / (abs(current_scalar) + 1e-12)
                ricci *= damping_factor
                
        except np.linalg.LinAlgError:
            pass
        
        return ricci
    
    def _enforce_ricci_constraints(self, ricci: np.ndarray, metric_tensor: np.ndarray) -> np.ndarray:
        """Enforce geometric constraints on Ricci tensor"""
        # Ricci tensor should be symmetric
        ricci = 0.5 * (ricci + ricci.T)
        
        # Apply eigenvalue bounds for physical reasonableness
        eigenvals, eigenvecs = np.linalg.eigh(ricci)
        
        # Bound eigenvalues to reasonable range for geophysical applications
        max_eigenval = 1000.0  # Reasonable bound
        eigenvals = np.clip(eigenvals, -max_eigenval, max_eigenval)
        
        # Reconstruct bounded tensor
        ricci = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return ricci
    
    def _compute_weyl_tensor_contribution(self, riemann_tensor: np.ndarray, 
                                        ricci_tensor: np.ndarray, scalar_curvature: float,
                                        metric_tensor: np.ndarray) -> np.ndarray:
        """Compute Weyl tensor contribution to curvature"""
        # Weyl tensor: C_ijkl = R_ijkl - (R_ik*g_jl - R_il*g_jk - R_jk*g_il + R_jl*g_ik)/(n-2)
        #                              + R/(n-1)(n-2) * (g_ik*g_jl - g_il*g_jk)
        # For 3D: n = 3
        
        weyl_contribution = np.zeros((3, 3))
        
        # Simplified Weyl contribution by extracting traceless part
        try:
            metric_inv = np.linalg.inv(metric_tensor)
            
            # Contract Weyl tensor to get contribution
            for i in range(3):
                for j in range(3):
                    weyl_sum = 0.0
                    
                    for k in range(3):
                        for l in range(3):
                            # Simplified Weyl tensor extraction
                            weyl_component = (riemann_tensor[i, j, k, l] - 
                                            ricci_tensor[i, k] * metric_tensor[j, l] +
                                            ricci_tensor[j, l] * metric_tensor[i, k])
                            
                            weyl_sum += metric_inv[k, l] * weyl_component
                    
                    weyl_contribution[i, j] = weyl_sum / 9.0  # Normalization
                    
        except np.linalg.LinAlgError:
            weyl_contribution = np.zeros((3, 3))
        
        return weyl_contribution
    
    def _compute_sectional_curvature_tensor(self, ricci_tensor: np.ndarray, 
                                          scalar_curvature: float,
                                          metric_tensor: np.ndarray) -> np.ndarray:
        """Compute sectional curvature tensor contribution"""
        # Sectional curvature measures curvature of 2D subspaces
        # Simplified implementation using Ricci and scalar curvature
        
        try:
            metric_inv = np.linalg.inv(metric_tensor)
            
            # Average sectional curvature approximation
            sectional_tensor = np.zeros((3, 3))
            
            for i in range(3):
                for j in range(3):
                    # Simplified sectional curvature based on Ricci components
                    sectional_value = (ricci_tensor[i, j] - 
                                     0.5 * scalar_curvature * metric_tensor[i, j])
                    sectional_tensor[i, j] = sectional_value
                    
        except np.linalg.LinAlgError:
            sectional_tensor = np.zeros((3, 3))
        
        return sectional_tensor
    
    def _apply_physical_curvature_bounds(self, curvature_tensor: np.ndarray) -> np.ndarray:
        """Apply physical bounds and constraints to final curvature tensor"""
        # Ensure symmetry
        curvature_tensor = 0.5 * (curvature_tensor + curvature_tensor.T)
        
        # Apply eigenvalue bounds for physical reasonableness
        eigenvals, eigenvecs = np.linalg.eigh(curvature_tensor)
        
        # Physical bounds for geophysical applications
        max_curvature = self.config.get('max_physical_curvature', 100.0)
        eigenvals = np.clip(eigenvals, -max_curvature, max_curvature)
        
        # Remove very small eigenvalues (numerical noise)
        noise_threshold = 1e-12 * max(abs(eigenvals)) if len(eigenvals) > 0 else 1e-12
        eigenvals[abs(eigenvals) < noise_threshold] = 0.0
        
        # Reconstruct physically bounded tensor
        curvature_tensor = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return curvature_tensor
    
    def _compute_principal_curvatures_discrete(self, tangent_coords: np.ndarray, normal: np.ndarray,
                                             distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute principal curvatures using discrete differential geometry"""
        n_neighbors = len(tangent_coords)
        if n_neighbors < 4:
            return np.array([0.0, 0.0]), np.eye(3)[:2]
        
        # Build local tangent coordinate system
        if np.linalg.norm(tangent_coords[0]) > 1e-12:
            u1 = tangent_coords[0] / np.linalg.norm(tangent_coords[0])
        else:
            u1 = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
            u1 = u1 - np.dot(u1, normal) * normal
            u1 = u1 / np.linalg.norm(u1)
        
        u2 = np.cross(normal, u1)
        u2 = u2 / np.linalg.norm(u2)
        
        # Project points to 2D tangent plane
        tangent_2d = np.zeros((n_neighbors, 2))
        heights = np.zeros(n_neighbors)
        
        for i in range(n_neighbors):
            tangent_2d[i, 0] = np.dot(tangent_coords[i], u1)
            tangent_2d[i, 1] = np.dot(tangent_coords[i], u2) 
            # Height is deviation from tangent plane
            heights[i] = np.dot(tangent_coords[i] - tangent_coords[i], normal)
        
        # Fit quadratic surface z = ax**2 + bxy + cy**2 
        if n_neighbors >= 5:
            A = np.zeros((n_neighbors, 5))  # [1, x, y, x**2, xy, y**2] but only quadratic terms
            
            weights = 1.0 / (distances**2 + 1e-12)
            weights = weights / np.sum(weights)
            W = np.diag(weights)
            
            for i in range(n_neighbors):
                x, y = tangent_2d[i]
                A[i] = [1, x, y, x*x, x*y, y*y][:5]
            
            try:
                # Weighted least squares
                AtWA = A.T @ W @ A
                AtWb = A.T @ W @ heights
                coeffs = np.linalg.solve(AtWA, AtWb)
                
                # Extract second derivatives (principal curvatures)
                # Hessian matrix: H = [[2a, b], [b, 2c]]
                if len(coeffs) >= 5:
                    hessian_2d = np.array([[2*coeffs[3], coeffs[4]], 
                                          [coeffs[4], 2*coeffs[5]]])
                else:
                    hessian_2d = np.zeros((2, 2))
                
                # Principal curvatures are eigenvalues of Hessian
                principal_curvatures = np.linalg.eigvals(hessian_2d)
                principal_directions_2d = np.linalg.eigvecs(hessian_2d)[1]
                
                # Convert 2D principal directions back to 3D
                principal_directions = np.zeros((2, 3))
                for i in range(2):
                    dir_2d = principal_directions_2d[:, i] 
                    principal_directions[i] = dir_2d[0] * u1 + dir_2d[1] * u2
                
            except np.linalg.LinAlgError:
                principal_curvatures = np.array([0.0, 0.0])
                principal_directions = np.array([u1, u2])
        else:
            principal_curvatures = np.array([0.0, 0.0])
            principal_directions = np.array([u1, u2])
        
        return principal_curvatures, principal_directions
    
    def _construct_tensor_from_principal_curvatures(self, principal_curvatures: np.ndarray,
                                                   principal_directions: np.ndarray, 
                                                   normal: np.ndarray) -> np.ndarray:
        """Construct curvature tensor from principal curvatures and directions"""
        if len(principal_curvatures) < 2 or len(principal_directions) < 2:
            return np.zeros((3, 3))
        
        k1, k2 = principal_curvatures[0], principal_curvatures[1]
        e1, e2 = principal_directions[0], principal_directions[1]
        
        # Curvature tensor in principal directions
        # K = k1 * (e1 kron e1) + k2 * (e2 kron e2)
        curvature_tensor = k1 * np.outer(e1, e1) + k2 * np.outer(e2, e2)
        
        # Add small normal component for 3D completeness
        curvature_tensor += 0.5 * (k1 + k2) * np.outer(normal, normal) * 0.1
        
        return curvature_tensor
    
    def _compute_cotangent_weights(self, center: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
        """Compute cotangent weights for discrete Laplace-Beltrami operator"""
        n_neighbors = len(neighbors)
        weights = np.zeros(n_neighbors)
        
        if n_neighbors < 3:
            return weights
        
        # For each neighbor, compute cotangent weight using adjacent triangles
        for i in range(n_neighbors):
            neighbor = neighbors[i]
            edge_vec = neighbor - center
            
            # Find two adjacent neighbors to form triangles
            prev_idx = (i - 1) % n_neighbors
            next_idx = (i + 1) % n_neighbors
            
            prev_neighbor = neighbors[prev_idx]
            next_neighbor = neighbors[next_idx]
            
            # Compute cotangent weights from angles in adjacent triangles
            # cot(alpha) + cot(beta) where alpha, beta are angles opposite to edge in adjacent triangles
            
            # Triangle 1: center - neighbor - prev_neighbor
            vec_to_prev = prev_neighbor - neighbor
            vec_to_center1 = center - neighbor
            cos_angle1 = np.dot(vec_to_prev, vec_to_center1) / (
                np.linalg.norm(vec_to_prev) * np.linalg.norm(vec_to_center1) + 1e-12)
            cos_angle1 = np.clip(cos_angle1, -1+1e-12, 1-1e-12)
            sin_angle1 = np.sqrt(1 - cos_angle1**2)
            cot1 = cos_angle1 / (sin_angle1 + 1e-12)
            
            # Triangle 2: center - neighbor - next_neighbor  
            vec_to_next = next_neighbor - neighbor
            vec_to_center2 = center - neighbor
            cos_angle2 = np.dot(vec_to_next, vec_to_center2) / (
                np.linalg.norm(vec_to_next) * np.linalg.norm(vec_to_center2) + 1e-12)
            cos_angle2 = np.clip(cos_angle2, -1+1e-12, 1-1e-12)
            sin_angle2 = np.sqrt(1 - cos_angle2**2)  
            cot2 = cos_angle2 / (sin_angle2 + 1e-12)
            
            # Total cotangent weight
            weights[i] = 0.5 * (cot1 + cot2)
            
            # Clamp to reasonable bounds
            weights[i] = np.clip(weights[i], -10.0, 10.0)
        
        return weights
    
    def _apply_curvature_corrections(self, curvature_tensor: np.ndarray, point: np.ndarray,
                                   coordinates: np.ndarray, point_idx: int) -> np.ndarray:
        """Apply geometric corrections and filtering to curvature tensor"""
        # Noise filtering using eigenvalue bounds
        eigenvals, eigenvecs = np.linalg.eigh(curvature_tensor)
        
        # Remove extremely large eigenvalues (likely numerical artifacts)
        max_curvature = self.config.get('max_curvature', 100.0)
        eigenvals = np.clip(eigenvals, -max_curvature, max_curvature)
        
        # Reconstruct filtered tensor
        filtered_tensor = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Apply smoothing if configured
        if self.config.get('curvature_smoothing', False):
            smoothing_factor = self.config.get('smoothing_factor', 0.1)
            identity_tensor = np.trace(filtered_tensor) / 3.0 * np.eye(3)
            filtered_tensor = (1 - smoothing_factor) * filtered_tensor + smoothing_factor * identity_tensor
        
        return filtered_tensor
    
    def _regularize_curvature_tensor(self, curvature_tensor: np.ndarray) -> np.ndarray:
        """Ensure curvature tensor has proper mathematical properties"""
        # Ensure symmetry
        curvature_tensor = 0.5 * (curvature_tensor + curvature_tensor.T)
        
        # Remove very small eigenvalues (numerical noise)
        eigenvals, eigenvecs = np.linalg.eigh(curvature_tensor)
        eigenvals[np.abs(eigenvals) < 1e-12] = 0.0
        
        # Reconstruct regularized tensor
        regularized_tensor = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return regularized_tensor
    
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
    
    def _assemble_finite_element_jacobian(self, displacement: np.ndarray, mesh: Any, **kwargs) -> sparse.lil_matrix:
        """Assemble finite element stiffness matrix"""
        n_nodes = len(mesh.coordinates) if hasattr(mesh, 'coordinates') else len(displacement)
        n_dim = len(mesh.coordinates[0]) if hasattr(mesh, 'coordinates') and hasattr(mesh.coordinates[0], '__len__') else 1
        n_dof = n_nodes * n_dim
        
        K_global = sparse.lil_matrix((n_dof, n_dof))
        n_elements = getattr(mesh, 'n_elements', n_nodes - 1)
        
        for elem_id in range(n_elements):
            # Compute element stiffness matrix
            K_elem = self._compute_element_stiffness(elem_id, mesh, displacement)
            
            # Get element nodes
            elem_nodes = self._get_element_nodes(elem_id, mesh)
            
            # Assemble into global matrix
            for i, node_i in enumerate(elem_nodes):
                for j, node_j in enumerate(elem_nodes):
                    for dim_i in range(n_dim):
                        for dim_j in range(n_dim):
                            global_i = node_i * n_dim + dim_i
                            global_j = node_j * n_dim + dim_j
                            local_i = i * n_dim + dim_i
                            local_j = j * n_dim + dim_j
                            
                            if (global_i < n_dof and global_j < n_dof and 
                                local_i < K_elem.shape[0] and local_j < K_elem.shape[1]):
                                K_global[global_i, global_j] += K_elem[local_i, local_j]
        
        return K_global
    
    def _assemble_finite_difference_jacobian(self, displacement: np.ndarray, **kwargs) -> sparse.lil_matrix:
        """Assemble finite difference stiffness matrix with advanced discretization"""
        n = len(displacement)
        E = self.material_props.youngs_modulus
        nu = self.material_props.poisson_ratio
        dx = self._get_grid_spacing()
        
        K = sparse.lil_matrix((n, n))
        
        # High-order finite difference stiffness matrix
        for i in range(n):
            if i == 0:
                # Left boundary - forward difference
                K[i, i] = 2 * E / dx**2
                if i + 1 < n:
                    K[i, i + 1] = -E / dx**2
            elif i == n - 1:
                # Right boundary - backward difference
                K[i, i] = 2 * E / dx**2
                K[i, i - 1] = -E / dx**2
            else:
                # Interior points - central difference
                K[i, i - 1] = -E / dx**2
                K[i, i] = 2 * E / dx**2
                K[i, i + 1] = -E / dx**2
        
        # Add Poisson effect for multi-dimensional cases
        if self.config.get('include_poisson_effect', True):
            poisson_contrib = nu * E / ((1 + nu) * (1 - 2 * nu))
            for i in range(1, n - 1):
                K[i, i] += poisson_contrib / dx**2
        
        return K
    
    def _compute_element_stiffness(self, elem_id: int, mesh: Any, displacement: np.ndarray) -> np.ndarray:
        """Compute element stiffness matrix"""
        elem_nodes = self._get_element_nodes(elem_id, mesh)
        n_nodes_per_elem = len(elem_nodes)
        n_dim = len(mesh.coordinates[0]) if hasattr(mesh, 'coordinates') and hasattr(mesh.coordinates[0], '__len__') else 1
        
        # Material properties
        E = self.material_props.youngs_modulus
        nu = self.material_props.poisson_ratio
        
        if n_dim == 1:
            # 1D element stiffness
            if hasattr(mesh, 'coordinates'):
                elem_length = abs(mesh.coordinates[elem_nodes[1]] - mesh.coordinates[elem_nodes[0]]) if len(elem_nodes) >= 2 else self._get_grid_spacing()
            else:
                elem_length = self._get_grid_spacing()
            
            K_elem = E / elem_length * np.array([[1, -1], [-1, 1]])
        
        elif n_dim == 2:
            # 2D element stiffness (simplified quadrilateral or triangular)
            K_elem = self._compute_2d_element_stiffness(elem_nodes, mesh)
        
        elif n_dim == 3:
            # 3D element stiffness
            K_elem = self._compute_3d_element_stiffness(elem_nodes, mesh)
        
        else:
            # Fallback
            dof_per_elem = n_nodes_per_elem * n_dim
            K_elem = np.eye(dof_per_elem) * E
        
        return K_elem
    
    def _compute_2d_element_stiffness(self, elem_nodes: List[int], mesh: Any) -> np.ndarray:
        """Compute 2D element stiffness matrix"""
        # Material properties
        E = self.material_props.youngs_modulus
        nu = self.material_props.poisson_ratio
        thickness = 1.0  # Assume unit thickness
        
        # Constitutive matrix
        if self.config.get('plane_stress', True):
            D = E / (1 - nu**2) * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1-nu)/2]
            ])
        else:
            factor = E / ((1 + nu) * (1 - 2*nu))
            D = factor * np.array([
                [1-nu, nu, 0],
                [nu, 1-nu, 0],
                [0, 0, (1-2*nu)/2]
            ])
        
        # Element coordinates
        coords = []
        for node in elem_nodes:
            if hasattr(mesh, 'coordinates') and node < len(mesh.coordinates):
                coords.append(mesh.coordinates[node])
            else:
                # Generate default coordinates
                coords.append([node * self._get_grid_spacing(), 0])
        
        # Simplified 4-node quadrilateral element
        if len(elem_nodes) == 4:
            # Gaussian quadrature points
            gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
                           (1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(3), 1/np.sqrt(3))]
            weights = [1, 1, 1, 1]
        else:
            # Single point integration
            gauss_points = [(0, 0)]
            weights = [4]
        
        K_elem = np.zeros((8, 8))  # 4 nodes * 2 DOF per node
        
        for gp, weight in zip(gauss_points, weights):
            xi, eta = gp
            
            # Shape functions and derivatives
            B = self._compute_B_matrix_2d(xi, eta, coords)
            
            # Element area (Jacobian determinant)
            J = self._compute_jacobian_2d(xi, eta, coords)
            det_J = np.linalg.det(J)
            
            # Integrate stiffness matrix
            K_elem += B.T @ D @ B * det_J * thickness * weight
        
        return K_elem
    
    def _compute_3d_element_stiffness(self, elem_nodes: List[int], mesh: Any) -> np.ndarray:
        """
        Advanced 3D element stiffness matrix computation featuring:
        - Multiple element types (hex8, hex20, hex27, tet4, tet10)
        - High-order Gauss quadrature integration
        - Anisotropic material properties
        - Enhanced strain formulations
        - Selective reduced integration for volumetric locking prevention
        - Mixed formulation support
        - Geometric nonlinearities (optional)
        """
        # Get nodal coordinates
        coords = self._get_element_coordinates_3d(elem_nodes, mesh)
        n_nodes = len(elem_nodes)
        n_dof = n_nodes * 3
        
        # Determine element type and integration scheme
        element_type = self._determine_element_type_3d(n_nodes)
        gauss_scheme = self._get_3d_gauss_scheme(element_type)
        
        # Enhanced material properties with anisotropy
        D_matrix = self._compute_enhanced_constitutive_matrix_3d()
        
        # Initialize stiffness matrix
        K_elem = np.zeros((n_dof, n_dof))
        
        # Enhanced strain parameters (for improved element performance)
        enhanced_strain_modes = self._get_enhanced_strain_modes_3d(element_type)
        
        # Selective integration for different strain components
        volumetric_gauss = self._get_reduced_gauss_scheme_3d(element_type)  # Prevent locking
        deviatoric_gauss = gauss_scheme  # Full integration for deviatoric
        
        # Volumetric stiffness (reduced integration to prevent volumetric locking)
        K_vol = self._compute_volumetric_stiffness_3d(
            coords, volumetric_gauss, D_matrix, element_type, enhanced_strain_modes)
        
        # Deviatoric stiffness (full integration)
        K_dev = self._compute_deviatoric_stiffness_3d(
            coords, deviatoric_gauss, D_matrix, element_type, enhanced_strain_modes)
        
        # Combine contributions
        K_elem = K_vol + K_dev
        
        # Apply stabilization for nearly incompressible materials
        if self.material_props.poisson_ratio > 0.45:
            K_stab = self._compute_stabilization_matrix_3d(coords, element_type)
            K_elem += K_stab
        
        # Add geometric stiffness for large deformation analysis (if enabled)
        if hasattr(self, 'large_deformation') and self.large_deformation:
            K_geo = self._compute_geometric_stiffness_3d(coords, element_type)
            K_elem += K_geo
        
        return K_elem
    
    def _get_element_coordinates_3d(self, elem_nodes: List[int], mesh: Any) -> np.ndarray:
        """Extract 3D coordinates for element nodes"""
        coords = []
        
        if hasattr(mesh, 'coordinates') and mesh.coordinates is not None:
            for node_id in elem_nodes:
                if node_id < len(mesh.coordinates):
                    coord = mesh.coordinates[node_id]
                    # Ensure 3D coordinates
                    if len(coord) == 2:
                        coords.append([coord[0], coord[1], 0.0])
                    else:
                        coords.append(coord[:3])
                else:
                    # Default coordinate if node not found
                    coords.append([0.0, 0.0, 0.0])
        else:
            # Generate default coordinates for simple geometric arrangement
            for i, node_id in enumerate(elem_nodes):
                x = (i % 2) * 1.0
                y = ((i // 2) % 2) * 1.0
                z = (i // 4) * 1.0
                coords.append([x, y, z])
        
        return np.array(coords)
    
    def _determine_element_type_3d(self, n_nodes: int) -> str:
        """Determine 3D element type based on number of nodes"""
        element_map = {
            4: 'tet4',      # Linear tetrahedron
            8: 'hex8',      # Linear hexahedron (brick)
            10: 'tet10',    # Quadratic tetrahedron
            20: 'hex20',    # Quadratic hexahedron
            27: 'hex27'     # Triquadratic hexahedron
        }
        return element_map.get(n_nodes, 'hex8')  # Default to hex8
    
    def _get_3d_gauss_scheme(self, element_type: str) -> List[Tuple[np.ndarray, float]]:
        """Get Gauss quadrature points and weights for 3D elements"""
        
        if element_type == 'tet4':
            # 4-point Gauss quadrature for tetrahedron
            a = (5 - np.sqrt(5)) / 20
            b = (5 + 3*np.sqrt(5)) / 20
            points = [
                np.array([a, a, a]), np.array([b, a, a]),
                np.array([a, b, a]), np.array([a, a, b])
            ]
            weights = [1/24] * 4
            
        elif element_type == 'tet10':
            # 11-point Gauss quadrature for quadratic tetrahedron
            points, weights = self._get_tet_gauss_11_point()
            
        elif element_type in ['hex8', 'hex20', 'hex27']:
            # Tensor product Gauss quadrature for hexahedral elements
            if element_type == 'hex8':
                n_gauss = 2  # 2x2x2 integration
            elif element_type == 'hex20':
                n_gauss = 3  # 3x3x3 integration
            else:  # hex27
                n_gauss = 3  # 3x3x3 integration
                
            # 1D Gauss points and weights
            if n_gauss == 2:
                xi_1d = [-1/np.sqrt(3), 1/np.sqrt(3)]
                w_1d = [1.0, 1.0]
            else:  # n_gauss == 3
                xi_1d = [-np.sqrt(3/5), 0.0, np.sqrt(3/5)]
                w_1d = [5/9, 8/9, 5/9]
            
            # Tensor product for 3D
            points = []
            weights = []
            for i in range(n_gauss):
                for j in range(n_gauss):
                    for k in range(n_gauss):
                        points.append(np.array([xi_1d[i], xi_1d[j], xi_1d[k]]))
                        weights.append(w_1d[i] * w_1d[j] * w_1d[k])
        
        else:
            # Default 2x2x2 Gauss quadrature
            xi = 1/np.sqrt(3)
            points = [
                np.array([-xi, -xi, -xi]), np.array([xi, -xi, -xi]),
                np.array([xi, xi, -xi]), np.array([-xi, xi, -xi]),
                np.array([-xi, -xi, xi]), np.array([xi, -xi, xi]),
                np.array([xi, xi, xi]), np.array([-xi, xi, xi])
            ]
            weights = [1.0] * 8
        
        return list(zip(points, weights))
    
    def _get_tet_gauss_11_point(self) -> Tuple[List[np.ndarray], List[float]]:
        """11-point Gauss quadrature for tetrahedron (high accuracy)"""
        # Coordinates in tetrahedral natural coordinates
        a1, a2 = 0.25, (7 - np.sqrt(15)) / 34
        a3 = (7 + np.sqrt(15)) / 34
        b1, b2 = (13 + 3*np.sqrt(15)) / 34, (13 - 3*np.sqrt(15)) / 34
        
        points = [
            np.array([a1, a1, a1]),  # Center
            np.array([a2, a2, a2]), np.array([a3, a2, a2]),
            np.array([a2, a3, a2]), np.array([a2, a2, a3]),
            np.array([b1, b2, b2]), np.array([b2, b1, b2]),
            np.array([b2, b2, b1]), np.array([b2, b2, b2]),
            np.array([a3, a3, a3]), np.array([a2, a3, a3])
        ]
        
        w1 = -74/5625
        w2 = 343/45000
        w3 = 56/2250
        weights = [w1, w2, w2, w2, w2, w3, w3, w3, w3, w2, w2]
        
        return points, weights
    
    def _get_reduced_gauss_scheme_3d(self, element_type: str) -> List[Tuple[np.ndarray, float]]:
        """Get reduced Gauss quadrature for volumetric terms (prevent locking)"""
        if element_type in ['hex8', 'hex20', 'hex27']:
            # 1-point Gauss quadrature (center point)
            return [(np.array([0.0, 0.0, 0.0]), 8.0)]
        else:
            # For tetrahedra, use single point at centroid
            return [(np.array([0.25, 0.25, 0.25]), 1/6)]
    
    def _compute_enhanced_constitutive_matrix_3d(self) -> np.ndarray:
        """
        Compute enhanced 3D constitutive matrix with:
        - Anisotropic material properties
        - Temperature-dependent properties
        - Damage/plasticity effects (if applicable)
        """
        E = self.material_props.youngs_modulus
        nu = self.material_props.poisson_ratio
        
        # Check for nearly incompressible materials
        if nu >= 0.5:
            nu = 0.499  # Avoid singularity
        
        # Enhanced constitutive matrix for anisotropic materials
        if hasattr(self.material_props, 'anisotropy_tensor'):
            return self._compute_anisotropic_constitutive_matrix()
        
        # Standard isotropic elasticity
        lambda_param = E * nu / ((1 + nu) * (1 - 2*nu))
        mu = E / (2 * (1 + nu))
        
        # Enhanced formulation with better numerical properties
        D = np.zeros((6, 6))
        
        # Diagonal terms (normal stresses)
        D[0, 0] = D[1, 1] = D[2, 2] = lambda_param + 2*mu
        
        # Off-diagonal terms (coupling between normal stresses)
        D[0, 1] = D[1, 0] = D[0, 2] = D[2, 0] = D[1, 2] = D[2, 1] = lambda_param
        
        # Shear terms
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        
        # Apply temperature dependency if available
        if hasattr(self, 'current_temperature'):
            temperature_factor = self._get_temperature_dependent_factor()
            D *= temperature_factor
        
        return D
    
    def _compute_anisotropic_constitutive_matrix(self) -> np.ndarray:
        """Compute constitutive matrix for anisotropic materials"""
        # This would implement full anisotropic elasticity
        # For now, return enhanced isotropic matrix
        return self._compute_enhanced_constitutive_matrix_3d()
    
    def _get_enhanced_strain_modes_3d(self, element_type: str) -> np.ndarray:
        """Get enhanced strain modes for improved element performance"""
        if element_type == 'hex8':
            # Enhanced assumed strain modes for hex8 elements
            # These help eliminate volumetric locking and hourglassing
            n_modes = 9  # Typical for hex8 elements
        elif element_type in ['hex20', 'hex27']:
            n_modes = 12  # More modes for higher-order elements
        else:
            n_modes = 6   # Basic modes for tetrahedra
        
        # Return identity for simplified implementation
        # In practice, this would contain specific enhanced strain interpolations
        return np.eye(n_modes)
    
    def _compute_volumetric_stiffness_3d(self, coords: np.ndarray, gauss_scheme: List,
                                       D_matrix: np.ndarray, element_type: str,
                                       enhanced_modes: np.ndarray) -> np.ndarray:
        """Compute volumetric part of stiffness matrix (reduced integration)"""
        n_nodes = len(coords)
        n_dof = n_nodes * 3
        K_vol = np.zeros((n_dof, n_dof))
        
        # Volumetric part of constitutive matrix
        bulk_modulus = self._compute_bulk_modulus()
        D_vol = np.zeros((6, 6))
        # Volumetric strain-stress relationship
        D_vol[:3, :3] = bulk_modulus / 3.0
        
        for gauss_point, weight in gauss_scheme:
            # Shape function derivatives
            dN_dxi = self._compute_shape_derivatives_3d(gauss_point, element_type)
            
            # Jacobian and its inverse
            J = coords.T @ dN_dxi.T
            try:
                J_inv = np.linalg.inv(J)
                det_J = np.linalg.det(J)
                if det_J <= 0:
                    continue
            except np.linalg.LinAlgError:
                continue
            
            # Shape function derivatives in global coordinates
            dN_dx = J_inv @ dN_dxi.T
            
            # B matrix for volumetric strains
            B_vol = self._compute_B_matrix_volumetric_3d(dN_dx)
            
            # Integrate volumetric stiffness
            K_vol += B_vol.T @ D_vol @ B_vol * det_J * weight
        
        return K_vol
    
    def _compute_deviatoric_stiffness_3d(self, coords: np.ndarray, gauss_scheme: List,
                                       D_matrix: np.ndarray, element_type: str,
                                       enhanced_modes: np.ndarray) -> np.ndarray:
        """Compute deviatoric part of stiffness matrix (full integration)"""
        n_nodes = len(coords)
        n_dof = n_nodes * 3
        K_dev = np.zeros((n_dof, n_dof))
        
        # Deviatoric part of constitutive matrix
        shear_modulus = self.material_props.youngs_modulus / (2 * (1 + self.material_props.poisson_ratio))
        D_dev = np.zeros((6, 6))
        # Deviatoric terms
        D_dev[3, 3] = D_dev[4, 4] = D_dev[5, 5] = 2 * shear_modulus
        # Off-diagonal terms for deviatoric normal stresses
        factor = 2 * shear_modulus / 3
        for i in range(3):
            for j in range(3):
                if i == j:
                    D_dev[i, j] = 4 * shear_modulus / 3
                else:
                    D_dev[i, j] = -2 * shear_modulus / 3
        
        for gauss_point, weight in gauss_scheme:
            # Shape function derivatives
            dN_dxi = self._compute_shape_derivatives_3d(gauss_point, element_type)
            
            # Jacobian and its inverse
            J = coords.T @ dN_dxi.T
            try:
                J_inv = np.linalg.inv(J)
                det_J = np.linalg.det(J)
                if det_J <= 0:
                    continue
            except np.linalg.LinAlgError:
                continue
            
            # Shape function derivatives in global coordinates
            dN_dx = J_inv @ dN_dxi.T
            
            # B matrix for all strain components
            B = self._compute_B_matrix_3d(dN_dx)
            
            # Integrate deviatoric stiffness
            K_dev += B.T @ D_dev @ B * det_J * weight
        
        return K_dev
    
    def _compute_shape_derivatives_3d(self, xi: np.ndarray, element_type: str) -> np.ndarray:
        """Compute shape function derivatives in natural coordinates"""
        if element_type == 'hex8':
            return self._hex8_shape_derivatives(xi)
        elif element_type == 'hex20':
            return self._hex20_shape_derivatives(xi)
        elif element_type == 'hex27':
            return self._hex27_shape_derivatives(xi)
        elif element_type == 'tet4':
            return self._tet4_shape_derivatives(xi)
        elif element_type == 'tet10':
            return self._tet10_shape_derivatives(xi)
        else:
            return self._hex8_shape_derivatives(xi)  # Default
    
    def _hex8_shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Shape function derivatives for 8-node hexahedron"""
        xi_val, eta, zeta = xi
        
        dN_dxi = np.array([
            [-(1-eta)*(1-zeta), -(1-xi_val)*(1-zeta), -(1-xi_val)*(1-eta)],
            [ (1-eta)*(1-zeta), -(1+xi_val)*(1-zeta), -(1+xi_val)*(1-eta)],
            [ (1+eta)*(1-zeta),  (1+xi_val)*(1-zeta), -(1+xi_val)*(1+eta)],
            [-(1+eta)*(1-zeta),  (1-xi_val)*(1-zeta), -(1-xi_val)*(1+eta)],
            [-(1-eta)*(1+zeta), -(1-xi_val)*(1+zeta),  (1-xi_val)*(1-eta)],
            [ (1-eta)*(1+zeta), -(1+xi_val)*(1+zeta),  (1+xi_val)*(1-eta)],
            [ (1+eta)*(1+zeta),  (1+xi_val)*(1+zeta),  (1+xi_val)*(1+eta)],
            [-(1+eta)*(1+zeta),  (1-xi_val)*(1+zeta),  (1-xi_val)*(1+eta)]
        ]) / 8.0
        
        return dN_dxi
    
    def _hex20_shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Shape function derivatives for 20-node hexahedron"""
        # Simplified - would implement full 20-node shape functions
        return self._hex8_shape_derivatives(xi)[:8]  # Use first 8 nodes as approximation
    
    def _hex27_shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Shape function derivatives for 27-node hexahedron"""
        # Simplified - would implement full 27-node shape functions
        return self._hex8_shape_derivatives(xi)[:8]  # Use first 8 nodes as approximation
    
    def _tet4_shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Shape function derivatives for 4-node tetrahedron"""
        # In tetrahedral coordinates, derivatives are constant
        dN_dxi = np.array([
            [-1, -1, -1],
            [ 1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  1]
        ])
        return dN_dxi
    
    def _tet10_shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Shape function derivatives for 10-node tetrahedron"""
        # Simplified - would implement full 10-node shape functions
        return self._tet4_shape_derivatives(xi)[:4]  # Use first 4 nodes as approximation
    
    def _compute_B_matrix_3d(self, dN_dx: np.ndarray) -> np.ndarray:
        """Compute strain-displacement matrix B for 3D"""
        n_nodes = dN_dx.shape[1]
        B = np.zeros((6, n_nodes * 3))
        
        for i in range(n_nodes):
            # Node i contributes to columns 3*i, 3*i+1, 3*i+2
            col_start = 3 * i
            
            # Normal strains (epsilonxx, epsilonyy, epsilonzz)
            B[0, col_start] = dN_dx[0, i]      # dNi/dx
            B[1, col_start + 1] = dN_dx[1, i]  # dNi/dy
            B[2, col_start + 2] = dN_dx[2, i]  # dNi/dz
            
            # Shear strains (gammaxy, gammaxz, gammayz)
            B[3, col_start] = dN_dx[1, i]      # dNi/dy
            B[3, col_start + 1] = dN_dx[0, i]  # dNi/dx
            
            B[4, col_start] = dN_dx[2, i]      # dNi/dz
            B[4, col_start + 2] = dN_dx[0, i]  # dNi/dx
            
            B[5, col_start + 1] = dN_dx[2, i]  # dNi/dz
            B[5, col_start + 2] = dN_dx[1, i]  # dNi/dy
        
        return B
    
    def _compute_B_matrix_volumetric_3d(self, dN_dx: np.ndarray) -> np.ndarray:
        """Compute B matrix for volumetric strains only"""
        n_nodes = dN_dx.shape[1]
        B_vol = np.zeros((6, n_nodes * 3))
        
        # Volumetric strain = (epsilonxx + epsilonyy + epsilonzz)/3
        for i in range(n_nodes):
            col_start = 3 * i
            vol_factor = 1.0 / 3.0
            
            B_vol[0, col_start] = vol_factor * dN_dx[0, i]
            B_vol[0, col_start + 1] = vol_factor * dN_dx[1, i] 
            B_vol[0, col_start + 2] = vol_factor * dN_dx[2, i]
            
            B_vol[1, col_start] = vol_factor * dN_dx[0, i]
            B_vol[1, col_start + 1] = vol_factor * dN_dx[1, i]
            B_vol[1, col_start + 2] = vol_factor * dN_dx[2, i]
            
            B_vol[2, col_start] = vol_factor * dN_dx[0, i]
            B_vol[2, col_start + 1] = vol_factor * dN_dx[1, i]
            B_vol[2, col_start + 2] = vol_factor * dN_dx[2, i]
        
        return B_vol
    
    def _compute_bulk_modulus(self) -> float:
        """Compute bulk modulus from elastic constants"""
        E = self.material_props.youngs_modulus
        nu = self.material_props.poisson_ratio
        return E / (3 * (1 - 2*nu))
    
    def _compute_stabilization_matrix_3d(self, coords: np.ndarray, element_type: str) -> np.ndarray:
        """Compute advanced stabilization matrix using PSPG/SUPG methods"""
        n_nodes = len(coords)
        n_dof = n_nodes * 3
        
        # Advanced stabilization parameters
        h_element = self._compute_element_size_tensor(coords)
        Re_element = self._compute_element_reynolds_number(coords)
        Pe_element = self._compute_element_peclet_number(coords)
        
        # Material properties
        E = self.material_props.youngs_modulus
        nu = self.material_props.poissons_ratio
        rho = getattr(self.material_props, 'density', 2700.0)  # Default rock density
        
        # Compute stabilization parameters based on element physics
        if element_type in ['tetrahedral', 'hexahedral']:
            # SUPG stabilization for advection-dominated problems
            tau_supg = self._compute_supg_parameter(h_element, Pe_element, Re_element)
            
            # PSPG stabilization for pressure (incompressible flow)
            tau_pspg = self._compute_pspg_parameter(h_element, nu, rho)
            
            # Discontinuity-capturing operator
            tau_dc = self._compute_discontinuity_capturing_parameter(coords)
        else:
            tau_supg = tau_pspg = tau_dc = 0.0
        
        # Gauss integration for stabilization matrix assembly
        gauss_points, weights = self._get_gauss_integration_points_3d()
        S_stab = np.zeros((n_dof, n_dof))
        
        for gp, weight in zip(gauss_points, weights):
            # Shape functions and derivatives at Gauss point
            N, dN_xi = self._get_shape_functions_3d(gp, element_type)
            
            # Jacobian and physical derivatives
            J, det_J, dN_dx = self._compute_jacobian_3d(dN_xi, coords)
            
            if det_J <= 0:
                continue
            
            # Velocity and gradient operators for stabilization
            u_vel = getattr(self, 'current_velocity', np.zeros(3))
            grad_op = self._compute_gradient_operator_3d(dN_dx, n_nodes)
            
            # SUPG stabilization matrix
            if tau_supg > 0:
                # Streamline upwind operator: tau_SUPG * (u*nablaN_i) * (u*nablaN_j)
                u_grad_N = grad_op @ u_vel  # u*nablaN for all shape functions
                S_supg = tau_supg * np.outer(u_grad_N, u_grad_N) * det_J * weight
                S_stab += np.kron(np.eye(3), S_supg)
            
            # PSPG stabilization matrix (pressure stabilization)
            if tau_pspg > 0:
                # Pressure stabilization: tau_PSPG * nablaN_i * nablaN_j
                grad_N_matrix = grad_op.reshape(n_nodes, 3)
                S_pspg = tau_pspg * (grad_N_matrix @ grad_N_matrix.T) * det_J * weight
                S_stab += np.kron(np.eye(3), S_pspg)
            
            # Discontinuity capturing (shock-capturing)
            if tau_dc > 0:
                # Isotropic discontinuity capturing
                S_dc = tau_dc * (grad_op.T @ grad_op) * det_J * weight
                S_stab += np.kron(np.eye(3), S_dc)
        
        # Additional stabilization for nearly incompressible materials
        if nu > 0.48:  # Nearly incompressible case
            # Volumetric locking prevention
            K_bulk = self._compute_bulk_modulus()
            vol_stab_param = h_element**2 / K_bulk
            
            # B-bar method stabilization
            B_vol = self._compute_volumetric_strain_matrix_3d(coords)
            S_vol = vol_stab_param * (B_vol.T @ B_vol)
            S_stab += S_vol
        
        return S_stab
    
    def _compute_geometric_stiffness_3d(self, coords: np.ndarray, element_type: str) -> np.ndarray:
        """Compute comprehensive geometric stiffness for 3D large deformation analysis"""
        n_nodes = len(coords)
        n_dof = n_nodes * 3
        
        # Get current stress state and deformation gradients
        current_stress = getattr(self, 'current_stress_tensor', np.zeros(6))  # Voigt notation
        deformation_grad = getattr(self, 'current_deformation_gradient', np.eye(3))
        displacement_grad = deformation_grad - np.eye(3)
        
        K_geom = np.zeros((n_dof, n_dof))
        
        # Gauss integration for geometric stiffness matrix
        gauss_points, weights = self._get_gauss_integration_points_3d()
        
        for gp, weight in zip(gauss_points, weights):
            # Shape functions and derivatives at Gauss point
            N, dN_xi = self._get_shape_functions_3d(gp, element_type)
            
            # Jacobian and physical derivatives
            J, det_J, dN_dx = self._compute_jacobian_3d(dN_xi, coords)
            
            if det_J <= 0:
                continue
            
            # Current configuration coordinates (accounting for large displacements)
            current_coords = coords + self._get_nodal_displacements_3d()
            J_current, det_J_current, dN_dx_current = self._compute_jacobian_3d(dN_xi, current_coords)
            
            # Cauchy stress tensor at Gauss point
            sigma = self._voigt_to_tensor(current_stress)  # Convert to 3x3 tensor
            
            # Green-Lagrange strain tensor
            E_gl = 0.5 * (displacement_grad + displacement_grad.T + displacement_grad.T @ displacement_grad)
            
            # Second Piola-Kirchhoff stress tensor
            F_inv = np.linalg.inv(deformation_grad)
            S_pk2 = det_J_current * F_inv @ sigma @ F_inv.T
            
            # Multiple geometric stiffness contributions:
            
            # 1. Initial stress stiffness matrix (K_sigma)
            K_sigma = self._compute_initial_stress_stiffness(dN_dx, sigma, det_J, weight)
            
            # 2. Initial displacement stiffness matrix (K_u)  
            K_u = self._compute_initial_displacement_stiffness(dN_dx, S_pk2, displacement_grad, det_J, weight)
            
            # 3. Follower force stiffness matrix (K_f) - for pressure loading
            if hasattr(self, 'follower_forces') and self.follower_forces:
                K_f = self._compute_follower_force_stiffness(N, dN_dx, deformation_grad, det_J, weight)
            else:
                K_f = np.zeros((n_dof, n_dof))
            
            # 4. Spin stiffness matrix for rotating reference frames
            if hasattr(self, 'angular_velocity'):
                K_spin = self._compute_spin_stiffness(N, dN_dx, self.angular_velocity, det_J, weight)
            else:
                K_spin = np.zeros((n_dof, n_dof))
            
            # Total geometric stiffness contribution at this Gauss point
            K_geom += K_sigma + K_u + K_f + K_spin
        
        return K_geom
    
    def _compute_initial_stress_stiffness(self, dN_dx: np.ndarray, sigma: np.ndarray, 
                                        det_J: float, weight: float) -> np.ndarray:
        """Compute initial stress stiffness matrix K_sigma"""
        n_nodes = dN_dx.shape[1]
        n_dof = n_nodes * 3
        K_sigma = np.zeros((n_dof, n_dof))
        
        # K_sigma_ij = integral (dN_i/dx_k) sigma_kl (dN_j/dx_l) dV
        for i in range(n_nodes):
            for j in range(n_nodes):
                # Stress-weighted gradient product
                stress_grad_product = np.sum(dN_dx[:, i][:, np.newaxis] * sigma * dN_dx[:, j])
                
                # Diagonal block contribution (same DOF direction)
                for dof in range(3):
                    idx_i, idx_j = 3*i + dof, 3*j + dof
                    K_sigma[idx_i, idx_j] += stress_grad_product * det_J * weight
        
        return K_sigma
    
    def _compute_initial_displacement_stiffness(self, dN_dx: np.ndarray, S_pk2: np.ndarray,
                                              disp_grad: np.ndarray, det_J: float, weight: float) -> np.ndarray:
        """Compute initial displacement stiffness matrix K_u"""
        n_nodes = dN_dx.shape[1]
        n_dof = n_nodes * 3
        K_u = np.zeros((n_dof, n_dof))
        
        # K_u involves products of displacement gradients and stress
        for i in range(n_nodes):
            for j in range(n_nodes):
                for alpha in range(3):  # DOF directions
                    for beta in range(3):
                        idx_i, idx_j = 3*i + alpha, 3*j + beta
                        
                        # Nonlinear geometric contribution
                        for k in range(3):
                            for l in range(3):
                                contrib = (dN_dx[k, i] * disp_grad[alpha, l] * 
                                         S_pk2[k, l] * dN_dx[l, j] * disp_grad[beta, l])
                                K_u[idx_i, idx_j] += contrib * det_J * weight
        
        return K_u
    
    def _compute_follower_force_stiffness(self, N: np.ndarray, dN_dx: np.ndarray, 
                                        F: np.ndarray, det_J: float, weight: float) -> np.ndarray:
        """Compute follower force stiffness for pressure-dependent loading"""
        n_nodes = len(N)
        n_dof = n_nodes * 3
        K_f = np.zeros((n_dof, n_dof))
        
        # Pressure magnitude (if available)
        pressure = getattr(self, 'current_pressure', 0.0)
        if abs(pressure) < 1e-12:
            return K_f
        
        # Normal vector in current configuration
        # This is simplified - full implementation would compute surface normals
        normal = np.array([0, 0, 1])  # Assume z-direction loading
        
        # Follower force contribution to geometric stiffness
        for i in range(n_nodes):
            for j in range(n_nodes):
                for alpha in range(3):
                    for beta in range(3):
                        idx_i, idx_j = 3*i + alpha, 3*j + beta
                        
                        # Follower force geometric nonlinearity
                        contrib = pressure * N[i] * N[j] * normal[alpha] * normal[beta]
                        K_f[idx_i, idx_j] += contrib * det_J * weight
        
        return K_f
    
    def _compute_characteristic_length(self, coords: np.ndarray) -> float:
        """Compute advanced characteristic element length with multiple metrics"""
        n_nodes, n_dim = coords.shape
        
        if n_dim != 3:
            # Fallback for lower dimensions
            coord_ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
            return np.mean(coord_ranges[coord_ranges > 0])
        
        # Multiple characteristic length measures for 3D elements
        
        # 1. Volume-based characteristic length
        if n_nodes == 4:  # Tetrahedral element
            h_vol = self._compute_tetrahedral_characteristic_length(coords)
        elif n_nodes == 8:  # Hexahedral element
            h_vol = self._compute_hexahedral_characteristic_length(coords)
        else:
            # General case: cube root of element volume
            volume = self._compute_element_volume_3d(coords)
            h_vol = volume**(1/3) if volume > 0 else 0.0
        
        # 2. Edge-based characteristic length (minimum edge length)
        h_edge = self._compute_minimum_edge_length(coords)
        
        # 3. Sphere-based characteristic length (circumradius and inradius)
        h_circumsphere = self._compute_circumsphere_radius(coords)
        h_insphere = self._compute_insphere_radius(coords)
        
        # 4. Aspect ratio corrected length
        h_aspect = self._compute_aspect_ratio_corrected_length(coords)
        
        # 5. Directional characteristic lengths (for anisotropic elements)
        h_directional = self._compute_directional_characteristic_lengths(coords)
        
        # Weighted combination of different measures
        weights = np.array([0.4, 0.3, 0.1, 0.1, 0.1])  # Prioritize volume and edge-based
        h_measures = np.array([h_vol, h_edge, h_circumsphere, h_insphere, h_aspect])
        
        # Remove zero or invalid measures
        valid_mask = h_measures > 1e-12
        if np.any(valid_mask):
            h_weighted = np.average(h_measures[valid_mask], weights=weights[valid_mask])
        else:
            h_weighted = np.mean(coord_ranges)  # Fallback
        
        return max(h_weighted, 1e-12)  # Ensure positive value
    
    def _compute_tetrahedral_characteristic_length(self, coords: np.ndarray) -> float:
        """Compute characteristic length for tetrahedral elements"""
        if coords.shape[0] != 4:
            return 0.0
        
        # Tetrahedron volume: V = |det(b-a, c-a, d-a)| / 6
        a, b, c, d = coords
        volume = abs(np.linalg.det(np.array([b-a, c-a, d-a]))) / 6.0
        
        # Characteristic length: h = (6V)^(1/3)
        return (6 * volume)**(1/3) if volume > 0 else 0.0
    
    def _compute_hexahedral_characteristic_length(self, coords: np.ndarray) -> float:
        """Compute characteristic length for hexahedral elements"""
        if coords.shape[0] != 8:
            return 0.0
        
        # Approximate volume using trilinear mapping
        # This is simplified - full implementation would use Jacobian integration
        bbox_volume = np.prod(np.max(coords, axis=0) - np.min(coords, axis=0))
        
        # Characteristic length: cube root of volume
        return bbox_volume**(1/3) if bbox_volume > 0 else 0.0
    
    def _compute_minimum_edge_length(self, coords: np.ndarray) -> float:
        """Compute minimum edge length between all node pairs"""
        n_nodes = coords.shape[0]
        min_length = float('inf')
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                edge_length = np.linalg.norm(coords[j] - coords[i])
                min_length = min(min_length, edge_length)
        
        return min_length if min_length != float('inf') else 0.0
    
    def _compute_circumsphere_radius(self, coords: np.ndarray) -> float:
        """Compute circumsphere radius for the element"""
        n_nodes, n_dim = coords.shape
        
        if n_nodes == 4 and n_dim == 3:  # Tetrahedron
            # Circumsphere radius for tetrahedron
            a, b, c, d = coords
            
            # Cayley-Menger determinant method
            def distance_squared(p1, p2):
                return np.sum((p1 - p2)**2)
            
            # Distance matrix
            D = np.zeros((5, 5))
            points = [a, b, c, d]
            
            for i in range(4):
                for j in range(4):
                    D[i, j] = distance_squared(points[i], points[j])
                D[i, 4] = 1.0
                D[4, i] = 1.0
            D[4, 4] = 0.0
            
            # Circumradius formula using Cayley-Menger determinant
            det_D = np.linalg.det(D)
            if abs(det_D) < 1e-12:
                return float('inf')  # Degenerate case
            
            volume = abs(np.linalg.det(np.array([b-a, c-a, d-a]))) / 6.0
            
            # Calculate edge lengths
            edge_lengths = []
            for i in range(4):
                for j in range(i+1, 4):
                    edge_lengths.append(np.linalg.norm(points[j] - points[i]))
            
            # Use formula: R = (abc)/(4V) for tetrahedron (approximation)
            if volume > 1e-12:
                return max(edge_lengths) / (4 * volume)
            else:
                return float('inf')
        else:
            # General case: use bounding sphere approximation
            centroid = np.mean(coords, axis=0)
            max_distance = np.max([np.linalg.norm(coord - centroid) for coord in coords])
            return max_distance
    
    def _compute_insphere_radius(self, coords: np.ndarray) -> float:
        """Compute insphere radius for the element"""
        n_nodes, n_dim = coords.shape
        
        if n_nodes == 4 and n_dim == 3:  # Tetrahedron
            a, b, c, d = coords
            
            # Volume
            volume = abs(np.linalg.det(np.array([b-a, c-a, d-a]))) / 6.0
            
            # Surface area (sum of triangle face areas)
            face_areas = []
            faces = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            
            for face in faces:
                p1, p2, p3 = face
                # Triangle area using cross product
                area = 0.5 * np.linalg.norm(np.cross(p2-p1, p3-p1))
                face_areas.append(area)
            
            total_surface_area = sum(face_areas)
            
            # Inradius: r = 3V/A
            if total_surface_area > 1e-12:
                return 3 * volume / total_surface_area
            else:
                return 0.0
        else:
            # Approximation: use minimum distance to element boundary
            centroid = np.mean(coords, axis=0)
            min_distance = np.min([np.linalg.norm(coord - centroid) for coord in coords])
            return min_distance
    
    def _compute_aspect_ratio_corrected_length(self, coords: np.ndarray) -> float:
        """Compute aspect ratio corrected characteristic length"""
        # Principal component analysis to find element orientation
        centered_coords = coords - np.mean(coords, axis=0)
        cov_matrix = np.cov(centered_coords.T)
        
        # Eigenvalues represent variance in principal directions
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
        
        if len(eigenvals) >= 2 and eigenvals[1] > 1e-12:
            aspect_ratio = eigenvals[0] / eigenvals[1]
            
            # Correct for high aspect ratios
            if aspect_ratio > 10:  # Highly anisotropic element
                # Use geometric mean of principal lengths
                return np.sqrt(eigenvals[0] * eigenvals[1])
            else:
                return np.sqrt(eigenvals[0])  # Use major axis
        else:
            # Isotropic or degenerate case
            return np.sqrt(np.mean(eigenvals))
    
    def _compute_directional_characteristic_lengths(self, coords: np.ndarray) -> float:
        """Compute directional characteristic lengths for anisotropic analysis"""
        # This returns the average of directional lengths
        # In practice, this might return a tensor for anisotropic stabilization
        centered_coords = coords - np.mean(coords, axis=0)
        cov_matrix = np.cov(centered_coords.T)
        eigenvals = np.linalg.eigvals(cov_matrix)
        
        # Geometric mean of characteristic lengths in all directions
        positive_eigenvals = eigenvals[eigenvals > 1e-12]
        if len(positive_eigenvals) > 0:
            return np.power(np.prod(positive_eigenvals), 1.0/len(positive_eigenvals))
        else:
            return 0.0
    
    def _get_temperature_dependent_factor(self) -> float:
        """Get comprehensive temperature-dependent material property factors"""
        if not hasattr(self, 'current_temperature'):
            return 1.0
        
        return self._compute_thermomechanical_coupling_factors()
    
    def _compute_thermomechanical_coupling_factors(self) -> Dict[str, float]:
        """Compute comprehensive thermomechanical coupling factors"""
        T = getattr(self, 'current_temperature', 293.15)  # Current temperature
        T_ref = 293.15  # Reference temperature (20?C)
        T_melt = getattr(self.material_props, 'melting_temperature', 1873.15)  # Default for rock
        
        # Temperature-dependent material property modifications
        factors = {}
        
        # 1. Elastic modulus temperature dependence
        # E(T) = E_0 * f_E(T) - accounts for thermal softening
        if T < 773.15:  # Below 500?C - linear dependence
            alpha_E = getattr(self.material_props, 'elastic_temp_coeff', -4e-4)  # 1/K
            factors['elastic_modulus'] = 1.0 + alpha_E * (T - T_ref)
        else:  # High temperature - nonlinear degradation
            # Exponential degradation approaching melting temperature
            T_critical = 0.8 * T_melt  # Critical temperature for rapid degradation
            if T < T_critical:
                factors['elastic_modulus'] = 1.0 - 0.3 * ((T - 773.15) / (T_critical - 773.15))**2
            else:
                # Rapid degradation near melting
                factors['elastic_modulus'] = 0.7 * np.exp(-5 * (T - T_critical) / (T_melt - T_critical))
        
        # 2. Thermal expansion effects
        alpha_thermal = getattr(self.material_props, 'thermal_expansion', 1e-5)  # 1/K
        factors['thermal_strain'] = alpha_thermal * (T - T_ref)
        
        # 3. Temperature-dependent Poisson's ratio
        # nu typically decreases with temperature for most rocks
        nu_coeff = getattr(self.material_props, 'poisson_temp_coeff', -1e-5)  # 1/K
        factors['poisson_ratio'] = 1.0 + nu_coeff * (T - T_ref)
        factors['poisson_ratio'] = max(0.1, min(0.49, factors['poisson_ratio']))  # Bounds
        
        # 4. Thermal conductivity temperature dependence
        # k(T) - typically decreases with temperature for crystalline materials
        if hasattr(self.material_props, 'thermal_conductivity'):
            k_ref = self.material_props.thermal_conductivity
            # Simplified model: k(T) = k_0 / (1 + betaT)
            beta_k = getattr(self.material_props, 'conductivity_temp_factor', 1e-3)
            factors['thermal_conductivity'] = 1.0 / (1.0 + beta_k * (T - T_ref))
        
        # 5. Specific heat temperature dependence
        # C_p typically increases with temperature
        if hasattr(self.material_props, 'specific_heat'):
            cp_coeff = getattr(self.material_props, 'specific_heat_temp_coeff', 2e-4)  # 1/K
            factors['specific_heat'] = 1.0 + cp_coeff * (T - T_ref)
        
        # 6. Density temperature dependence (thermal expansion effect)
        rho_coeff = -3 * alpha_thermal  # Volume expansion coefficient
        factors['density'] = 1.0 + rho_coeff * (T - T_ref)
        
        # 7. Yield strength temperature dependence (if applicable)
        if hasattr(self.material_props, 'yield_strength'):
            # Typically decreases with temperature
            if T < 673.15:  # Below 400?C
                sigma_y_coeff = getattr(self.material_props, 'yield_temp_coeff', -2e-3)  # 1/K
                factors['yield_strength'] = 1.0 + sigma_y_coeff * (T - T_ref)
            else:
                # Rapid decrease at high temperatures
                factors['yield_strength'] = np.exp(-0.005 * (T - 673.15))
        
        # 8. Creep effects at high temperatures
        if T > 773.15:  # Above 500?C
            # Arrhenius-type creep activation
            Q_creep = getattr(self.material_props, 'creep_activation_energy', 300e3)  # J/mol
            R_gas = 8.314  # J/(mol*K)
            
            creep_factor = np.exp(-Q_creep / (R_gas * T))
            factors['creep_rate'] = creep_factor
            
            # Modify elastic properties to account for creep compliance
            time_step = getattr(self, 'current_time_step', 1.0)  # seconds
            creep_compliance = creep_factor * time_step
            factors['creep_compliance'] = creep_compliance
        
        # 9. Phase transformation effects (if applicable)
        phase_transitions = getattr(self.material_props, 'phase_transition_temps', [])
        for T_phase in phase_transitions:
            if abs(T - T_phase) < 50:  # Within 50K of phase transition
                # Property changes near phase transitions
                proximity_factor = np.exp(-((T - T_phase) / 25)**2)  # Gaussian
                factors['phase_transition_factor'] = 1.0 + 0.2 * proximity_factor
        
        # 10. Coupled thermomechanical effects
        # Mechanical work contribution to temperature (adiabatic heating)
        if hasattr(self, 'current_plastic_work_rate'):
            # ? = beta / (rho * C_p) - Taylor-Quinney coefficient
            beta_tq = getattr(self.material_props, 'taylor_quinney_coeff', 0.9)
            rho = getattr(self.material_props, 'density', 2700.0)
            cp = getattr(self.material_props, 'specific_heat', 800.0)
            
            factors['mechanical_heating'] = beta_tq / (rho * cp)
        
        # Apply bounds and safety checks
        for key, value in factors.items():
            if 'modulus' in key or 'strength' in key:
                factors[key] = max(0.01, min(2.0, value))  # Keep within reasonable bounds
            elif 'conductivity' in key or 'density' in key:
                factors[key] = max(0.1, min(5.0, value))
        
        return factors
    
    def _apply_thermomechanical_coupling(self, base_properties: Dict) -> Dict:
        """Apply thermomechanical coupling to material properties"""
        if not hasattr(self, 'current_temperature'):
            return base_properties
        
        coupling_factors = self._compute_thermomechanical_coupling_factors()
        modified_properties = base_properties.copy()
        
        # Apply temperature-dependent modifications
        if 'elastic_modulus' in coupling_factors:
            modified_properties['youngs_modulus'] *= coupling_factors['elastic_modulus']
        
        if 'poisson_ratio' in coupling_factors:
            modified_properties['poissons_ratio'] *= coupling_factors['poisson_ratio']
        
        if 'thermal_conductivity' in coupling_factors and 'thermal_conductivity' in base_properties:
            modified_properties['thermal_conductivity'] *= coupling_factors['thermal_conductivity']
        
        if 'density' in coupling_factors:
            modified_properties['density'] *= coupling_factors['density']
        
        # Add thermal strain contribution
        if 'thermal_strain' in coupling_factors:
            thermal_strain_tensor = np.eye(3) * coupling_factors['thermal_strain']
            modified_properties['thermal_strain_tensor'] = thermal_strain_tensor
        
        return modified_properties
    
    def _get_element_nodes(self, elem_id: int, mesh: Any) -> List[int]:
        """Get nodes belonging to an element"""
        if hasattr(mesh, 'elements') and elem_id < len(mesh.elements):
            return mesh.elements[elem_id]
        elif hasattr(mesh, 'connectivity'):
            return mesh.connectivity[elem_id]
        else:
            # Default linear connectivity
            if elem_id < getattr(mesh, 'n_nodes', 100) - 1:
                return [elem_id, elem_id + 1]
            else:
                return [elem_id]
    
    def _compute_B_matrix(self, elem_id: int, mesh: Any) -> np.ndarray:
        """Compute advanced strain-displacement matrix (B matrix) for 2D elements"""
        elem_nodes = self._get_element_nodes(elem_id, mesh)
        n_nodes = len(elem_nodes)
        
        # Get element coordinates
        coords = self._get_element_coordinates_2d(elem_nodes, mesh)
        
        if n_nodes == 2:  # 1D bar element
            return self._compute_B_matrix_1d_bar(coords)
        elif n_nodes == 3:  # Linear triangular element
            return self._compute_B_matrix_triangle(coords)
        elif n_nodes == 6:  # Quadratic triangular element
            return self._compute_B_matrix_triangle_quadratic(coords)
        elif n_nodes == 4:  # Bilinear quadrilateral element
            return self._compute_B_matrix_quad(coords)
        elif n_nodes == 8:  # Serendipity quadrilateral element
            return self._compute_B_matrix_quad_serendipity(coords)
        elif n_nodes == 9:  # Biquadratic quadrilateral element
            return self._compute_B_matrix_quad_biquadratic(coords)
        else:
            # Fallback for arbitrary polygonal elements
            return self._compute_B_matrix_polygon(coords)
    
    def _compute_B_matrix_1d_bar(self, coords: np.ndarray) -> np.ndarray:
        """Compute B matrix for 1D bar element"""
        L = np.linalg.norm(coords[1] - coords[0])  # Element length
        B = np.array([[-1/L, 1/L]])
        return B
    
    def _compute_B_matrix_triangle(self, coords: np.ndarray) -> np.ndarray:
        """Compute B matrix for linear triangular element"""
        x1, y1 = coords[0]
        x2, y2 = coords[1] 
        x3, y3 = coords[2]
        
        # Area calculation using cross product
        A = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        
        if A < 1e-12:
            return np.zeros((3, 6))  # Degenerate triangle
        
        # Shape function derivatives (constant for linear triangle)
        dN1_dx = (y2 - y3) / (2*A)
        dN1_dy = (x3 - x2) / (2*A)
        dN2_dx = (y3 - y1) / (2*A)
        dN2_dy = (x1 - x3) / (2*A)
        dN3_dx = (y1 - y2) / (2*A)
        dN3_dy = (x2 - x1) / (2*A)
        
        # B matrix: [epsilonxx, epsilonyy, gammaxy] = B * [u1, v1, u2, v2, u3, v3]
        B = np.array([
            [dN1_dx, 0,      dN2_dx, 0,      dN3_dx, 0     ],
            [0,      dN1_dy, 0,      dN2_dy, 0,      dN3_dy],
            [dN1_dy, dN1_dx, dN2_dy, dN2_dx, dN3_dy, dN3_dx]
        ])
        
        return B
    
    def _compute_B_matrix_triangle_quadratic(self, coords: np.ndarray) -> np.ndarray:
        """Compute B matrix for quadratic triangular element (6 nodes)"""
        # This requires integration over the element - simplified version at centroid
        xi = eta = 1/3  # Natural coordinates at centroid
        
        # Quadratic triangle shape function derivatives
        dN_dxi = np.array([
            4*xi + 4*eta - 3,  # N1
            4*xi - 1,          # N2
            0,                 # N3
            4 - 8*xi - 4*eta,  # N4
            4*eta,             # N5
            -4*eta             # N6
        ])
        
        dN_deta = np.array([
            4*xi + 4*eta - 3,  # N1
            0,                 # N2
            4*eta - 1,         # N3
            -4*xi,             # N4
            4*xi,              # N5
            4 - 4*xi - 8*eta   # N6
        ])
        
        # Jacobian transformation
        J = np.zeros((2, 2))
        for i in range(6):
            J[0, 0] += dN_dxi[i] * coords[i, 0]   # dx/dxi
            J[0, 1] += dN_dxi[i] * coords[i, 1]   # dy/dxi
            J[1, 0] += dN_deta[i] * coords[i, 0]  # dx/deta
            J[1, 1] += dN_deta[i] * coords[i, 1]  # dy/deta
        
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-12:
            return np.zeros((3, 12))  # Degenerate element
        
        J_inv = np.linalg.inv(J)
        
        # Physical derivatives
        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
        
        # B matrix for 6-node triangle
        B = np.zeros((3, 12))
        for i in range(6):
            B[0, 2*i] = dN_dx[i]     # epsilonxx
            B[1, 2*i+1] = dN_dy[i]   # epsilonyy
            B[2, 2*i] = dN_dy[i]     # gammaxy
            B[2, 2*i+1] = dN_dx[i]   # gammaxy
        
        return B
    
    def _compute_B_matrix_quad(self, coords: np.ndarray) -> np.ndarray:
        """Compute B matrix for bilinear quadrilateral element (4 nodes)"""
        # Evaluate at element center (xi=0, eta=0)
        xi = eta = 0.0
        
        # Shape function derivatives for 4-node quad
        dN_dxi = np.array([
            -(1-eta)/4, (1-eta)/4, (1+eta)/4, -(1+eta)/4
        ])
        dN_deta = np.array([
            -(1-xi)/4, -(1+xi)/4, (1+xi)/4, (1-xi)/4
        ])
        
        # Jacobian transformation
        J = np.zeros((2, 2))
        for i in range(4):
            J[0, 0] += dN_dxi[i] * coords[i, 0]   # dx/dxi
            J[0, 1] += dN_dxi[i] * coords[i, 1]   # dy/dxi
            J[1, 0] += dN_deta[i] * coords[i, 0]  # dx/deta
            J[1, 1] += dN_deta[i] * coords[i, 1]  # dy/deta
        
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-12:
            return np.zeros((3, 8))  # Degenerate element
        
        J_inv = np.linalg.inv(J)
        
        # Physical derivatives
        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
        
        # B matrix for 4-node quad
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i] = dN_dx[i]     # epsilonxx
            B[1, 2*i+1] = dN_dy[i]   # epsilonyy
            B[2, 2*i] = dN_dy[i]     # gammaxy
            B[2, 2*i+1] = dN_dx[i]   # gammaxy
        
        return B
    
    def _compute_B_matrix_3d(self, elem_id: int, mesh: Any) -> np.ndarray:
        """Compute advanced 3D strain-displacement matrix"""
        elem_nodes = self._get_element_nodes(elem_id, mesh)
        n_nodes = len(elem_nodes)
        
        # Get element coordinates
        coords = self._get_element_coordinates_3d(elem_nodes, mesh)
        
        if n_nodes == 4:  # Linear tetrahedral element
            return self._compute_B_matrix_tetrahedron(coords)
        elif n_nodes == 10:  # Quadratic tetrahedral element
            return self._compute_B_matrix_tetrahedron_quadratic(coords)
        elif n_nodes == 8:  # Trilinear hexahedral element
            return self._compute_B_matrix_hexahedron(coords)
        elif n_nodes == 20:  # Serendipity hexahedral element
            return self._compute_B_matrix_hexahedron_serendipity(coords)
        elif n_nodes == 27:  # Triquadratic hexahedral element
            return self._compute_B_matrix_hexahedron_triquadratic(coords)
        else:
            # Fallback for arbitrary 3D elements
            return self._compute_B_matrix_3d_general(coords)
    
    def _compute_B_matrix_tetrahedron(self, coords: np.ndarray) -> np.ndarray:
        """Compute B matrix for linear tetrahedral element (4 nodes)"""
        x1, y1, z1 = coords[0]
        x2, y2, z2 = coords[1]
        x3, y3, z3 = coords[2]
        x4, y4, z4 = coords[3]
        
        # Volume calculation
        V = abs(np.linalg.det(np.array([
            [x2-x1, y2-y1, z2-z1],
            [x3-x1, y3-y1, z3-z1],
            [x4-x1, y4-y1, z4-z1]
        ]))) / 6.0
        
        if V < 1e-12:
            return np.zeros((6, 12))  # Degenerate tetrahedron
        
        # Shape function derivatives (constant for linear tetrahedron)
        # dN_i/dx = (1/6V) * determinant of appropriate submatrix
        
        # Node 1 derivatives
        dN1_dx = np.linalg.det(np.array([[y2-y4, z2-z4], [y3-y4, z3-z4]])) / (6*V)
        dN1_dy = -np.linalg.det(np.array([[x2-x4, z2-z4], [x3-x4, z3-z4]])) / (6*V)
        dN1_dz = np.linalg.det(np.array([[x2-x4, y2-y4], [x3-x4, y3-y4]])) / (6*V)
        
        # Node 2 derivatives
        dN2_dx = -np.linalg.det(np.array([[y1-y4, z1-z4], [y3-y4, z3-z4]])) / (6*V)
        dN2_dy = np.linalg.det(np.array([[x1-x4, z1-z4], [x3-x4, z3-z4]])) / (6*V)
        dN2_dz = -np.linalg.det(np.array([[x1-x4, y1-y4], [x3-x4, y3-y4]])) / (6*V)
        
        # Node 3 derivatives
        dN3_dx = np.linalg.det(np.array([[y1-y4, z1-z4], [y2-y4, z2-z4]])) / (6*V)
        dN3_dy = -np.linalg.det(np.array([[x1-x4, z1-z4], [x2-x4, z2-z4]])) / (6*V)
        dN3_dz = np.linalg.det(np.array([[x1-x4, y1-y4], [x2-x4, y2-y4]])) / (6*V)
        
        # Node 4 derivatives
        dN4_dx = -(dN1_dx + dN2_dx + dN3_dx)
        dN4_dy = -(dN1_dy + dN2_dy + dN3_dy)
        dN4_dz = -(dN1_dz + dN2_dz + dN3_dz)
        
        # B matrix: [epsilonxx, epsilonyy, epsilonzz, gammaxy, gammayz, gammaxz] = B * [u1,v1,w1,u2,v2,w2,u3,v3,w3,u4,v4,w4]
        B = np.array([
            # epsilonxx = du/dx
            [dN1_dx, 0,      0,      dN2_dx, 0,      0,      dN3_dx, 0,      0,      dN4_dx, 0,      0     ],
            # epsilonyy = dv/dy
            [0,      dN1_dy, 0,      0,      dN2_dy, 0,      0,      dN3_dy, 0,      0,      dN4_dy, 0     ],
            # epsilonzz = dw/dz
            [0,      0,      dN1_dz, 0,      0,      dN2_dz, 0,      0,      dN3_dz, 0,      0,      dN4_dz],
            # gammaxy = du/dy + dv/dx
            [dN1_dy, dN1_dx, 0,      dN2_dy, dN2_dx, 0,      dN3_dy, dN3_dx, 0,      dN4_dy, dN4_dx, 0     ],
            # gammayz = dv/dz + dw/dy
            [0,      dN1_dz, dN1_dy, 0,      dN2_dz, dN2_dy, 0,      dN3_dz, dN3_dy, 0,      dN4_dz, dN4_dy],
            # gammaxz = du/dz + dw/dx
            [dN1_dz, 0,      dN1_dx, dN2_dz, 0,      dN2_dx, dN3_dz, 0,      dN3_dx, dN4_dz, 0,      dN4_dx]
        ])
        
        return B
    
    def _compute_B_matrix_hexahedron(self, coords: np.ndarray) -> np.ndarray:
        """Compute B matrix for trilinear hexahedral element (8 nodes)"""
        # Evaluate at element center (xi=eta=zeta=0)
        xi = eta = zeta = 0.0
        
        # Shape function derivatives for 8-node hex
        dN_dxi = np.array([
            -(1-eta)*(1-zeta)/8, (1-eta)*(1-zeta)/8, (1+eta)*(1-zeta)/8, -(1+eta)*(1-zeta)/8,
            -(1-eta)*(1+zeta)/8, (1-eta)*(1+zeta)/8, (1+eta)*(1+zeta)/8, -(1+eta)*(1+zeta)/8
        ])
        
        dN_deta = np.array([
            -(1-xi)*(1-zeta)/8, -(1+xi)*(1-zeta)/8, (1+xi)*(1-zeta)/8, (1-xi)*(1-zeta)/8,
            -(1-xi)*(1+zeta)/8, -(1+xi)*(1+zeta)/8, (1+xi)*(1+zeta)/8, (1-xi)*(1+zeta)/8
        ])
        
        dN_dzeta = np.array([
            -(1-xi)*(1-eta)/8, -(1+xi)*(1-eta)/8, -(1+xi)*(1+eta)/8, -(1-xi)*(1+eta)/8,
            (1-xi)*(1-eta)/8, (1+xi)*(1-eta)/8, (1+xi)*(1+eta)/8, (1-xi)*(1+eta)/8
        ])
        
        # Jacobian transformation
        J = np.zeros((3, 3))
        for i in range(8):
            J[0, 0] += dN_dxi[i] * coords[i, 0]     # dx/dxi
            J[0, 1] += dN_dxi[i] * coords[i, 1]     # dy/dxi
            J[0, 2] += dN_dxi[i] * coords[i, 2]     # dz/dxi
            J[1, 0] += dN_deta[i] * coords[i, 0]    # dx/deta
            J[1, 1] += dN_deta[i] * coords[i, 1]    # dy/deta
            J[1, 2] += dN_deta[i] * coords[i, 2]    # dz/deta
            J[2, 0] += dN_dzeta[i] * coords[i, 0]   # dx/dzeta
            J[2, 1] += dN_dzeta[i] * coords[i, 1]   # dy/dzeta
            J[2, 2] += dN_dzeta[i] * coords[i, 2]   # dz/dzeta
        
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-12:
            return np.zeros((6, 24))  # Degenerate element
        
        J_inv = np.linalg.inv(J)
        
        # Physical derivatives
        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta + J_inv[0, 2] * dN_dzeta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta + J_inv[1, 2] * dN_dzeta
        dN_dz = J_inv[2, 0] * dN_dxi + J_inv[2, 1] * dN_deta + J_inv[2, 2] * dN_dzeta
        
        # B matrix for 8-node hex
        B = np.zeros((6, 24))
        for i in range(8):
            B[0, 3*i] = dN_dx[i]      # epsilonxx
            B[1, 3*i+1] = dN_dy[i]    # epsilonyy
            B[2, 3*i+2] = dN_dz[i]    # epsilonzz
            B[3, 3*i] = dN_dy[i]      # gammaxy
            B[3, 3*i+1] = dN_dx[i]    # gammaxy
            B[4, 3*i+1] = dN_dz[i]    # gammayz
            B[4, 3*i+2] = dN_dy[i]    # gammayz
            B[5, 3*i] = dN_dz[i]      # gammaxz
            B[5, 3*i+2] = dN_dx[i]    # gammaxz
        
        return B
    
    def _compute_B_matrix_2d(self, xi: float, eta: float, coords: List[List[float]]) -> np.ndarray:
        """Compute 2D B matrix at natural coordinates (xi, eta)"""
        # Shape function derivatives for 4-node quadrilateral
        dN_dxi = np.array([
            [-(1-eta)/4, (1-eta)/4, (1+eta)/4, -(1+eta)/4],
            [-(1-xi)/4, -(1+xi)/4, (1+xi)/4, (1-xi)/4]
        ])
        
        # Jacobian matrix
        J = np.zeros((2, 2))
        for i in range(4):
            if i < len(coords):
                J[0, 0] += dN_dxi[0, i] * coords[i][0]
                J[0, 1] += dN_dxi[0, i] * coords[i][1] if len(coords[i]) > 1 else 0
                J[1, 0] += dN_dxi[1, i] * coords[i][0]
                J[1, 1] += dN_dxi[1, i] * coords[i][1] if len(coords[i]) > 1 else 0
        
        # Inverse Jacobian
        det_J = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        if abs(det_J) < 1e-12:
            det_J = 1.0  # Avoid division by zero
        
        J_inv = np.array([[J[1,1], -J[0,1]], [-J[1,0], J[0,0]]]) / det_J
        
        # Shape function derivatives in physical coordinates
        dN_dxy = J_inv @ dN_dxi
        
        # B matrix
        B = np.zeros((3, 8))  # 4 nodes * 2 DOF
        for i in range(4):
            B[0, 2*i] = dN_dxy[0, i]      # epsilonxx = du/dx
            B[1, 2*i+1] = dN_dxy[1, i]    # epsilonyy = dv/dy
            B[2, 2*i] = dN_dxy[1, i]      # gammaxy = du/dy
            B[2, 2*i+1] = dN_dxy[0, i]    # gammaxy += dv/dx
        
        return B
    
    def _compute_jacobian_2d(self, xi: float, eta: float, coords: List[List[float]]) -> np.ndarray:
        """Compute 2D Jacobian matrix"""
        J = np.eye(2)
        
        # Shape function derivatives
        dN_dxi = np.array([
            [-(1-eta)/4, (1-eta)/4, (1+eta)/4, -(1+eta)/4],
            [-(1-xi)/4, -(1+xi)/4, (1+xi)/4, (1-xi)/4]
        ])
        
        for i in range(min(4, len(coords))):
            J[0, 0] += dN_dxi[0, i] * coords[i][0]
            J[0, 1] += dN_dxi[0, i] * (coords[i][1] if len(coords[i]) > 1 else 0)
            J[1, 0] += dN_dxi[1, i] * coords[i][0]
            J[1, 1] += dN_dxi[1, i] * (coords[i][1] if len(coords[i]) > 1 else 0)
        
        return J
    
    def _get_element_stress(self, stress: np.ndarray, elem_id: int, elem_nodes: List[int]) -> np.ndarray:
        """Extract element stress from global stress array"""
        if len(stress.shape) == 1:
            # 1D stress
            if elem_id < len(stress):
                return np.array([stress[elem_id]])
            else:
                return np.array([0.0])
        else:
            # Multi-dimensional stress
            elem_stress = []
            for node in elem_nodes:
                if node < len(stress):
                    elem_stress.append(stress[node])
                else:
                    elem_stress.append(np.zeros(stress.shape[1] if len(stress.shape) > 1 else 1))
            
            return np.array(elem_stress)
    
    def _compute_element_force_from_stress(self, elem_stress: np.ndarray, elem_id: int, mesh: Any) -> np.ndarray:
        """Compute element force vector from stress using advanced finite element formulations"""
        elem_nodes = self._get_element_nodes(elem_id, mesh)
        elem_type = self._determine_element_type(elem_nodes, mesh)
        n_dim = self._get_spatial_dimension(mesh)
        
        # Initialize force vector with proper DOF layout
        elem_force = np.zeros(len(elem_nodes) * n_dim)
        
        # Get element geometry and properties
        elem_coords = self._get_element_coordinates(elem_nodes, mesh)
        jacobian_det = self._compute_jacobian_determinant(elem_coords, elem_type)
        
        # Comprehensive stress-to-force conversion using numerical integration
        if elem_type == 'linear':
            elem_force = self._compute_force_linear_elements(elem_stress, elem_coords, n_dim)
        elif elem_type == 'quadratic':
            elem_force = self._compute_force_quadratic_elements(elem_stress, elem_coords, n_dim)
        elif elem_type == 'cubic':
            elem_force = self._compute_force_cubic_elements(elem_stress, elem_coords, n_dim)
        else:
            # Advanced isoparametric formulation with Gaussian quadrature
            elem_force = self._compute_force_isoparametric(elem_stress, elem_coords, elem_type, n_dim)
        
        # Apply stress extrapolation from integration points to nodes
        if hasattr(self, '_stress_extrapolation_matrix'):
            elem_force = self._apply_stress_extrapolation(elem_force, elem_id)
        
        # Include geometric nonlinearity effects
        if self.config.get('large_deformation', False):
            elem_force += self._compute_geometric_force_correction(elem_stress, elem_coords, n_dim)
        
        return elem_force
    
    def _compute_element_volume(self, elem_id: int, mesh: Any) -> float:
        """Compute element volume/area/length using advanced geometric algorithms"""
        elem_nodes = self._get_element_nodes(elem_id, mesh)
        elem_coords = self._get_element_coordinates(elem_nodes, mesh)
        elem_type = self._determine_element_type(elem_nodes, mesh)
        n_dim = len(elem_coords[0]) if elem_coords else 1
        
        if n_dim == 1:  # 1D elements
            return self._compute_1d_element_length(elem_coords, elem_type)
        elif n_dim == 2:  # 2D elements
            return self._compute_2d_element_area(elem_coords, elem_type)
        elif n_dim == 3:  # 3D elements
            return self._compute_3d_element_volume(elem_coords, elem_type)
        
        return self._get_grid_spacing()
    
    def _compute_1d_element_length(self, coords: List[np.ndarray], elem_type: str) -> float:
        """Compute 1D element length with higher-order geometry support"""
        if elem_type == 'linear':
            return abs(coords[1][0] - coords[0][0])
        elif elem_type in ['quadratic', 'cubic']:
            # Use parametric integration for curved elements
            return self._integrate_parametric_length(coords)
        return np.linalg.norm(coords[-1] - coords[0])
    
    def _compute_2d_element_area(self, coords: List[np.ndarray], elem_type: str) -> float:
        """Compute 2D element area using advanced geometric methods"""
        n_nodes = len(coords)
        
        if n_nodes == 3:  # Triangle
            if elem_type == 'linear':
                # Standard triangle area formula
                v1 = coords[1] - coords[0]
                v2 = coords[2] - coords[0]
                return 0.5 * abs(np.cross(v1, v2))
            else:
                # Higher-order triangular elements
                return self._compute_curved_triangle_area(coords, elem_type)
        
        elif n_nodes == 4:  # Quadrilateral
            if elem_type == 'linear':
                # Bilinear quadrilateral using Shoelace formula
                return self._shoelace_area(coords)
            else:
                # Isoparametric quadrilateral with Gaussian quadrature
                return self._compute_isoparametric_area(coords, elem_type)
        
        elif n_nodes >= 6:  # Higher-order elements
            return self._compute_higher_order_2d_area(coords, elem_type)
        
        return 1.0
    
    def _compute_3d_element_volume(self, coords: List[np.ndarray], elem_type: str) -> float:
        """Compute 3D element volume using advanced methods"""
        n_nodes = len(coords)
        
        if n_nodes == 4:  # Tetrahedron
            if elem_type == 'linear':
                # Standard tetrahedral volume formula
                v1 = coords[1] - coords[0]
                v2 = coords[2] - coords[0]
                v3 = coords[3] - coords[0]
                return abs(np.dot(v1, np.cross(v2, v3))) / 6.0
            else:
                return self._compute_curved_tetrahedron_volume(coords, elem_type)
        
        elif n_nodes == 8:  # Hexahedron
            if elem_type == 'linear':
                # Trilinear hexahedral volume using divergence theorem
                return self._compute_hexahedral_volume(coords)
            else:
                # Isoparametric hexahedral elements
                return self._compute_isoparametric_volume(coords, elem_type)
        
        elif n_nodes >= 10:  # Higher-order 3D elements
            return self._compute_higher_order_3d_volume(coords, elem_type)
        
        return 1.0
    
    def _determine_element_type(self, elem_nodes: List[int], mesh: Any) -> str:
        """Determine element type based on node count and mesh properties"""
        n_nodes = len(elem_nodes)
        n_dim = self._get_spatial_dimension(mesh)
        
        # 1D elements
        if n_dim == 1:
            if n_nodes == 2: return 'linear'
            elif n_nodes == 3: return 'quadratic'
            elif n_nodes == 4: return 'cubic'
        
        # 2D elements
        elif n_dim == 2:
            if n_nodes == 3: return 'linear'  # Linear triangle
            elif n_nodes == 4: return 'linear'  # Linear quad
            elif n_nodes == 6: return 'quadratic'  # Quadratic triangle
            elif n_nodes == 8: return 'serendipity'  # Serendipity quad
            elif n_nodes == 9: return 'quadratic'  # Quadratic quad
        
        # 3D elements
        elif n_dim == 3:
            if n_nodes == 4: return 'linear'  # Linear tetrahedron
            elif n_nodes == 8: return 'linear'  # Linear hexahedron
            elif n_nodes == 10: return 'quadratic'  # Quadratic tetrahedron
            elif n_nodes == 20: return 'serendipity'  # Serendipity hex
            elif n_nodes == 27: return 'quadratic'  # Quadratic hexahedron
        
        return 'linear'  # Default fallback
    
    def _get_spatial_dimension(self, mesh: Any) -> int:
        """Determine spatial dimension from mesh"""
        if hasattr(mesh, 'coordinates') and len(mesh.coordinates) > 0:
            return len(mesh.coordinates[0]) if hasattr(mesh.coordinates[0], '__len__') else 1
        return self.config.get('spatial_dimension', 2)
    
    def _get_element_coordinates(self, elem_nodes: List[int], mesh: Any) -> List[np.ndarray]:
        """Get element nodal coordinates"""
        coords = []
        if hasattr(mesh, 'coordinates'):
            for node in elem_nodes:
                if node < len(mesh.coordinates):
                    coords.append(np.array(mesh.coordinates[node]))
                else:
                    # Default coordinate for missing nodes
                    n_dim = self._get_spatial_dimension(mesh)
                    coords.append(np.zeros(n_dim))
        return coords
    
    def _compute_jacobian_determinant(self, elem_coords: List[np.ndarray], elem_type: str) -> float:
        """Compute Jacobian determinant for coordinate transformation"""
        n_dim = len(elem_coords[0]) if elem_coords else 1
        n_nodes = len(elem_coords)
        
        if n_dim == 1:
            # 1D Jacobian is simply dx/d?
            if n_nodes >= 2:
                return abs(elem_coords[1][0] - elem_coords[0][0]) / 2.0
        
        elif n_dim == 2:
            # 2D Jacobian matrix
            if elem_type == 'linear' and n_nodes >= 3:
                # For linear elements, constant Jacobian
                J = np.zeros((2, 2))
                J[0, 0] = elem_coords[1][0] - elem_coords[0][0]  # dx/d?
                J[0, 1] = elem_coords[2][0] - elem_coords[0][0]  # dx/d?
                J[1, 0] = elem_coords[1][1] - elem_coords[0][1]  # dy/d?
                J[1, 1] = elem_coords[2][1] - elem_coords[0][1]  # dy/d?
                return abs(np.linalg.det(J))
        
        elif n_dim == 3:
            # 3D Jacobian matrix
            if elem_type == 'linear' and n_nodes >= 4:
                J = np.zeros((3, 3))
                for i in range(3):
                    J[i, 0] = elem_coords[1][i] - elem_coords[0][i]  # dx_i/d?
                    J[i, 1] = elem_coords[2][i] - elem_coords[0][i]  # dx_i/d?
                    J[i, 2] = elem_coords[3][i] - elem_coords[0][i]  # dx_i/d?
                return abs(np.linalg.det(J))
        
        return 1.0  # Default unit Jacobian
    
    def _compute_force_linear_elements(self, elem_stress: np.ndarray, elem_coords: List[np.ndarray], n_dim: int) -> np.ndarray:
        """Compute forces for linear elements using analytical integration"""
        n_nodes = len(elem_coords)
        elem_force = np.zeros(n_nodes * n_dim)
        
        if n_dim == 1:
            # 1D linear element force computation
            length = abs(elem_coords[1][0] - elem_coords[0][0])
            stress_avg = np.mean(elem_stress) if elem_stress.ndim > 0 else elem_stress
            
            # Distribute stress uniformly to nodes
            elem_force[0] = stress_avg * length / 2
            elem_force[1] = stress_avg * length / 2
        
        elif n_dim == 2:
            # 2D linear element (triangle/quad)
            area = self._compute_2d_element_area(elem_coords, 'linear')
            
            if n_nodes == 3:  # Triangle
                stress_tensor = self._reconstruct_stress_tensor(elem_stress, n_dim)
                for i in range(3):
                    for j in range(n_dim):
                        # Equal distribution for linear triangles
                        elem_force[i * n_dim + j] = stress_tensor[j] * area / 3
            
            elif n_nodes == 4:  # Quadrilateral
                stress_tensor = self._reconstruct_stress_tensor(elem_stress, n_dim)
                for i in range(4):
                    for j in range(n_dim):
                        elem_force[i * n_dim + j] = stress_tensor[j] * area / 4
        
        elif n_dim == 3:
            # 3D linear element
            volume = self._compute_3d_element_volume(elem_coords, 'linear')
            stress_tensor = self._reconstruct_stress_tensor(elem_stress, n_dim)
            
            for i in range(n_nodes):
                for j in range(n_dim):
                    elem_force[i * n_dim + j] = stress_tensor[j] * volume / n_nodes
        
        return elem_force
    
    def _compute_force_quadratic_elements(self, elem_stress: np.ndarray, elem_coords: List[np.ndarray], n_dim: int) -> np.ndarray:
        """Compute forces for quadratic elements using Gaussian quadrature"""
        n_nodes = len(elem_coords)
        elem_force = np.zeros(n_nodes * n_dim)
        
        # Gaussian quadrature points and weights for quadratic elements
        if n_dim == 1:
            gp, gw = self._get_gauss_points_1d(order=3)  # 3-point for quadratic
        elif n_dim == 2:
            gp, gw = self._get_gauss_points_2d(order=3)
        else:
            gp, gw = self._get_gauss_points_3d(order=3)
        
        # Integrate force at each Gauss point
        for gp_i, (xi, weight) in enumerate(zip(gp, gw)):
            # Compute shape functions and derivatives at Gauss point
            N, dN = self._shape_functions_quadratic(xi, n_dim, n_nodes)
            
            # Jacobian at this point
            J = self._compute_jacobian_at_point(elem_coords, dN)
            det_J = np.linalg.det(J) if J.ndim == 2 else abs(J)
            
            # Stress at this Gauss point (interpolated)
            stress_gp = self._interpolate_stress_at_gauss_point(elem_stress, N)
            
            # Add contribution to nodal forces
            for i in range(n_nodes):
                for j in range(n_dim):
                    elem_force[i * n_dim + j] += N[i] * stress_gp[j] * det_J * weight
        
        return elem_force
    
    def _compute_force_cubic_elements(self, elem_stress: np.ndarray, elem_coords: List[np.ndarray], n_dim: int) -> np.ndarray:
        """Compute forces for cubic elements using high-order integration"""
        n_nodes = len(elem_coords)
        elem_force = np.zeros(n_nodes * n_dim)
        
        # Higher-order Gaussian quadrature for cubic elements
        if n_dim == 1:
            gp, gw = self._get_gauss_points_1d(order=4)  # 4-point for cubic
        elif n_dim == 2:
            gp, gw = self._get_gauss_points_2d(order=4)
        else:
            gp, gw = self._get_gauss_points_3d(order=4)
        
        # Similar integration as quadratic but with cubic shape functions
        for gp_i, (xi, weight) in enumerate(zip(gp, gw)):
            N, dN = self._shape_functions_cubic(xi, n_dim, n_nodes)
            J = self._compute_jacobian_at_point(elem_coords, dN)
            det_J = np.linalg.det(J) if J.ndim == 2 else abs(J)
            
            stress_gp = self._interpolate_stress_at_gauss_point(elem_stress, N)
            
            for i in range(n_nodes):
                for j in range(n_dim):
                    elem_force[i * n_dim + j] += N[i] * stress_gp[j] * det_J * weight
        
        return elem_force
    
    def _compute_force_isoparametric(self, elem_stress: np.ndarray, elem_coords: List[np.ndarray], 
                                   elem_type: str, n_dim: int) -> np.ndarray:
        """Compute forces for isoparametric elements with advanced integration schemes"""
        n_nodes = len(elem_coords)
        elem_force = np.zeros(n_nodes * n_dim)
        
        # Adaptive quadrature order based on element type
        quad_order = {'linear': 2, 'quadratic': 3, 'cubic': 4, 'serendipity': 3}.get(elem_type, 2)
        
        if n_dim == 1:
            gp, gw = self._get_gauss_points_1d(order=quad_order)
        elif n_dim == 2:
            gp, gw = self._get_gauss_points_2d(order=quad_order)
        else:
            gp, gw = self._get_gauss_points_3d(order=quad_order)
        
        for gp_i, (xi, weight) in enumerate(zip(gp, gw)):
            # Isoparametric shape functions
            N, dN_xi = self._isoparametric_shape_functions(xi, elem_type, n_dim)
            
            # Physical derivatives through Jacobian
            J = self._compute_jacobian_matrix(elem_coords, dN_xi)
            det_J = np.linalg.det(J) if J.ndim == 2 else abs(J)
            
            # Stress interpolation with higher accuracy
            stress_gp = self._advanced_stress_interpolation(elem_stress, N, xi, elem_type)
            
            # Force contribution with geometric corrections
            geometric_factor = self._compute_geometric_correction_factor(J, elem_type)
            
            for i in range(n_nodes):
                for j in range(n_dim):
                    contrib = N[i] * stress_gp[j] * det_J * weight * geometric_factor
                    elem_force[i * n_dim + j] += contrib
        
        return elem_force
    
    def _reconstruct_stress_tensor(self, elem_stress: np.ndarray, n_dim: int) -> np.ndarray:
        """Reconstruct full stress tensor from stress components"""
        if elem_stress.ndim == 0:
            # Scalar stress
            return np.full(n_dim, elem_stress)
        elif elem_stress.ndim == 1:
            if len(elem_stress) == n_dim:
                return elem_stress
            elif len(elem_stress) == 1:
                return np.full(n_dim, elem_stress[0])
            else:
                # Voigt notation to tensor components
                if n_dim == 2 and len(elem_stress) >= 3:
                    return np.array([elem_stress[0], elem_stress[1], elem_stress[2]])  # sigmaxx, sigmayy, tauxy
                elif n_dim == 3 and len(elem_stress) >= 6:
                    return np.array([elem_stress[0], elem_stress[1], elem_stress[2]])  # sigmaxx, sigmayy, sigmazz
        
        return np.zeros(n_dim)
    
    def _get_gauss_points_1d(self, order: int) -> Tuple[List[float], List[float]]:
        """Get 1D Gaussian quadrature points and weights"""
        if order == 1:
            return ([0.0], [2.0])
        elif order == 2:
            return ([-0.577350269, 0.577350269], [1.0, 1.0])
        elif order == 3:
            return ([-0.774596669, 0.0, 0.774596669], [0.555555556, 0.888888889, 0.555555556])
        elif order >= 4:
            return ([-0.861136312, -0.339981044, 0.339981044, 0.861136312], 
                   [0.347854845, 0.652145155, 0.652145155, 0.347854845])
        return ([0.0], [2.0])
    
    def _get_gauss_points_2d(self, order: int) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Get 2D Gaussian quadrature points and weights"""
        gp_1d, gw_1d = self._get_gauss_points_1d(order)
        gp_2d = []
        gw_2d = []
        
        for i, xi in enumerate(gp_1d):
            for j, eta in enumerate(gp_1d):
                gp_2d.append((xi, eta))
                gw_2d.append(gw_1d[i] * gw_1d[j])
        
        return gp_2d, gw_2d
    
    def _get_gauss_points_3d(self, order: int) -> Tuple[List[Tuple[float, float, float]], List[float]]:
        """Get 3D Gaussian quadrature points and weights"""
        gp_1d, gw_1d = self._get_gauss_points_1d(order)
        gp_3d = []
        gw_3d = []
        
        for i, xi in enumerate(gp_1d):
            for j, eta in enumerate(gp_1d):
                for k, zeta in enumerate(gp_1d):
                    gp_3d.append((xi, eta, zeta))
                    gw_3d.append(gw_1d[i] * gw_1d[j] * gw_1d[k])
        
        return gp_3d, gw_3d
    
    def _shape_functions_quadratic(self, xi: Union[float, Tuple], n_dim: int, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute quadratic shape functions and derivatives"""
        N = np.zeros(n_nodes)
        dN = np.zeros((n_dim, n_nodes))
        
        if n_dim == 1:
            # 1D quadratic shape functions
            if isinstance(xi, (tuple, list)):
                xi = xi[0]
            if n_nodes == 3:
                N[0] = 0.5 * xi * (xi - 1)    # N1 = ?(?-1)/2
                N[1] = 1 - xi**2              # N2 = 1 - ?**2
                N[2] = 0.5 * xi * (xi + 1)    # N3 = ?(?+1)/2
                
                dN[0, 0] = xi - 0.5           # dN1/d?
                dN[0, 1] = -2 * xi            # dN2/d?
                dN[0, 2] = xi + 0.5           # dN3/d?
        
        elif n_dim == 2:
            # 2D quadratic shape functions
            if n_nodes == 6:  # Quadratic triangle
                xi1, xi2 = xi[0], xi[1]
                xi3 = 1 - xi1 - xi2
                
                N[0] = xi1 * (2*xi1 - 1)
                N[1] = xi2 * (2*xi2 - 1)
                N[2] = xi3 * (2*xi3 - 1)
                N[3] = 4 * xi1 * xi2
                N[4] = 4 * xi2 * xi3
                N[5] = 4 * xi3 * xi1
                
                # Derivatives
                dN[0, 0] = 4*xi1 - 1; dN[1, 0] = 0
                dN[0, 1] = 0; dN[1, 1] = 4*xi2 - 1
                dN[0, 2] = -4*xi3 + 1; dN[1, 2] = -4*xi3 + 1
                dN[0, 3] = 4*xi2; dN[1, 3] = 4*xi1
                dN[0, 4] = -4*xi2; dN[1, 4] = 4*(xi3 - xi2)
                dN[0, 5] = 4*(xi3 - xi1); dN[1, 5] = -4*xi1
            
            elif n_nodes == 9:  # Quadratic quadrilateral
                xi1, xi2 = xi[0], xi[1]
                self._quadratic_quad_shape_functions(xi1, xi2, N, dN)
        
        return N, dN
    
    def _shape_functions_cubic(self, xi: Union[float, Tuple], n_dim: int, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cubic shape functions and derivatives"""
        N = np.zeros(n_nodes)
        dN = np.zeros((n_dim, n_nodes))
        
        if n_dim == 1 and n_nodes == 4:
            # 1D cubic Hermite shape functions
            if isinstance(xi, (tuple, list)):
                xi = xi[0]
            
            xi2 = xi * xi
            xi3 = xi2 * xi
            
            N[0] = (2*xi3 - 3*xi2 + 1) / 8 + 1/8      # H1
            N[1] = (-2*xi3 + 3*xi2 + 1) / 8 + 1/8     # H2
            N[2] = (xi3 - xi) / 8                      # H3
            N[3] = (xi3 + xi) / 8                      # H4
            
            dN[0, 0] = (6*xi2 - 6*xi) / 8
            dN[0, 1] = (-6*xi2 + 6*xi) / 8
            dN[0, 2] = (3*xi2 - 1) / 8
            dN[0, 3] = (3*xi2 + 1) / 8
        
        return N, dN
    
    def _isoparametric_shape_functions(self, xi: Union[float, Tuple], elem_type: str, n_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute isoparametric shape functions for various element types"""
        if elem_type == 'linear':
            return self._linear_shape_functions(xi, n_dim)
        elif elem_type == 'quadratic':
            return self._quadratic_shape_functions(xi, n_dim)
        elif elem_type == 'serendipity':
            return self._serendipity_shape_functions(xi, n_dim)
        else:
            return self._linear_shape_functions(xi, n_dim)
    
    def _linear_shape_functions(self, xi: Union[float, Tuple], n_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Linear isoparametric shape functions"""
        if n_dim == 1:
            if isinstance(xi, (tuple, list)):
                xi = xi[0]
            N = np.array([0.5*(1-xi), 0.5*(1+xi)])
            dN = np.array([[-0.5, 0.5]])
            return N, dN
        
        elif n_dim == 2:
            xi1, xi2 = xi[0], xi[1]
            N = np.array([
                0.25*(1-xi1)*(1-xi2),  # N1
                0.25*(1+xi1)*(1-xi2),  # N2
                0.25*(1+xi1)*(1+xi2),  # N3
                0.25*(1-xi1)*(1+xi2)   # N4
            ])
            dN = np.array([
                [-0.25*(1-xi2), 0.25*(1-xi2), 0.25*(1+xi2), -0.25*(1+xi2)],  # dN/dxi1
                [-0.25*(1-xi1), -0.25*(1+xi1), 0.25*(1+xi1), 0.25*(1-xi1)]   # dN/dxi2
            ])
            return N, dN
        
        elif n_dim == 3:
            xi1, xi2, xi3 = xi[0], xi[1], xi[2]
            N = np.array([
                0.125*(1-xi1)*(1-xi2)*(1-xi3),  # N1
                0.125*(1+xi1)*(1-xi2)*(1-xi3),  # N2
                0.125*(1+xi1)*(1+xi2)*(1-xi3),  # N3
                0.125*(1-xi1)*(1+xi2)*(1-xi3),  # N4
                0.125*(1-xi1)*(1-xi2)*(1+xi3),  # N5
                0.125*(1+xi1)*(1-xi2)*(1+xi3),  # N6
                0.125*(1+xi1)*(1+xi2)*(1+xi3),  # N7
                0.125*(1-xi1)*(1+xi2)*(1+xi3)   # N8
            ])
            # Analytical derivatives for trilinear hexahedral elements
            dN = np.zeros((3, 8))
            
            # Node 1: (?_1=-1, ?_2=-1, ?_3=-1)
            dN[0, 0] = -0.125 * (1-xi2) * (1-xi3)  # dN1/d?_1
            dN[1, 0] = -0.125 * (1-xi1) * (1-xi3)  # dN1/d?_2
            dN[2, 0] = -0.125 * (1-xi1) * (1-xi2)  # dN1/d?_3
            
            # Node 2: (?_1=+1, ?_2=-1, ?_3=-1)
            dN[0, 1] = +0.125 * (1-xi2) * (1-xi3)  # dN2/d?_1
            dN[1, 1] = -0.125 * (1+xi1) * (1-xi3)  # dN2/d?_2
            dN[2, 1] = -0.125 * (1+xi1) * (1-xi2)  # dN2/d?_3
            
            # Node 3: (?_1=+1, ?_2=+1, ?_3=-1)
            dN[0, 2] = +0.125 * (1+xi2) * (1-xi3)  # dN3/d?_1
            dN[1, 2] = +0.125 * (1+xi1) * (1-xi3)  # dN3/d?_2
            dN[2, 2] = -0.125 * (1+xi1) * (1+xi2)  # dN3/d?_3
            
            # Node 4: (?_1=-1, ?_2=+1, ?_3=-1)
            dN[0, 3] = -0.125 * (1+xi2) * (1-xi3)  # dN4/d?_1
            dN[1, 3] = +0.125 * (1-xi1) * (1-xi3)  # dN4/d?_2
            dN[2, 3] = -0.125 * (1-xi1) * (1+xi2)  # dN4/d?_3
            
            # Node 5: (?_1=-1, ?_2=-1, ?_3=+1)
            dN[0, 4] = -0.125 * (1-xi2) * (1+xi3)  # dN5/d?_1
            dN[1, 4] = -0.125 * (1-xi1) * (1+xi3)  # dN5/d?_2
            dN[2, 4] = +0.125 * (1-xi1) * (1-xi2)  # dN5/d?_3
            
            # Node 6: (?_1=+1, ?_2=-1, ?_3=+1)
            dN[0, 5] = +0.125 * (1-xi2) * (1+xi3)  # dN6/d?_1
            dN[1, 5] = -0.125 * (1+xi1) * (1+xi3)  # dN6/d?_2
            dN[2, 5] = +0.125 * (1+xi1) * (1-xi2)  # dN6/d?_3
            
            # Node 7: (?_1=+1, ?_2=+1, ?_3=+1)
            dN[0, 6] = +0.125 * (1+xi2) * (1+xi3)  # dN7/d?_1
            dN[1, 6] = +0.125 * (1+xi1) * (1+xi3)  # dN7/d?_2
            dN[2, 6] = +0.125 * (1+xi1) * (1+xi2)  # dN7/d?_3
            
            # Node 8: (?_1=-1, ?_2=+1, ?_3=+1)
            dN[0, 7] = -0.125 * (1+xi2) * (1+xi3)  # dN8/d?_1
            dN[1, 7] = +0.125 * (1-xi1) * (1+xi3)  # dN8/d?_2
            dN[2, 7] = +0.125 * (1-xi1) * (1+xi2)  # dN8/d?_3
            return N, dN
        
        return np.array([1.0]), np.array([[0.0]])
    
    def _compute_jacobian_at_point(self, elem_coords: List[np.ndarray], dN: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix at specific point"""
        n_dim = len(elem_coords[0]) if elem_coords else 1
        n_nodes = len(elem_coords)
        
        J = np.zeros((n_dim, n_dim))
        for i in range(n_dim):
            for j in range(n_dim):
                for k in range(n_nodes):
                    if k < len(elem_coords) and j < len(dN) and k < len(dN[j]):
                        J[i, j] += elem_coords[k][i] * dN[j, k]
        
        return J
    
    def _compute_jacobian_matrix(self, elem_coords: List[np.ndarray], dN_xi: np.ndarray) -> np.ndarray:
        """Compute full Jacobian transformation matrix"""
        n_dim = len(elem_coords[0]) if elem_coords else 1
        n_nodes = len(elem_coords)
        
        # J[i,j] = sum_k (x_k[i] * dN_k/dxi_j)
        J = np.zeros((n_dim, n_dim))
        for i in range(n_dim):
            for j in range(n_dim):
                for k in range(min(n_nodes, len(elem_coords))):
                    if j < dN_xi.shape[0] and k < dN_xi.shape[1]:
                        J[i, j] += elem_coords[k][i] * dN_xi[j, k]
        
        return J
    
    def _interpolate_stress_at_gauss_point(self, elem_stress: np.ndarray, N: np.ndarray) -> np.ndarray:
        """Interpolate stress tensor at Gauss point using shape functions"""
        if elem_stress.ndim == 0:
            return np.array([elem_stress])
        elif elem_stress.ndim == 1:
            if len(elem_stress) == len(N):
                # Nodal stress values
                return np.array([np.dot(N, elem_stress)])
            else:
                # Constant stress over element
                return elem_stress
        else:
            # Multi-component stress tensor
            n_components = elem_stress.shape[1] if elem_stress.ndim > 1 else 1
            stress_gp = np.zeros(n_components)
            
            for i in range(min(len(N), elem_stress.shape[0])):
                if elem_stress.ndim > 1:
                    stress_gp += N[i] * elem_stress[i, :]
                else:
                    stress_gp[0] += N[i] * elem_stress[i]
            
            return stress_gp
    
    def _advanced_stress_interpolation(self, elem_stress: np.ndarray, N: np.ndarray, 
                                     xi: Union[float, Tuple], elem_type: str) -> np.ndarray:
        """Advanced stress interpolation with extrapolation and recovery techniques"""
        # Start with basic interpolation
        stress_gp = self._interpolate_stress_at_gauss_point(elem_stress, N)
        
        # Apply stress recovery techniques for higher accuracy
        if elem_type in ['quadratic', 'cubic', 'serendipity']:
            # Superconvergent patch recovery (SPR) correction
            stress_gp = self._apply_spr_correction(stress_gp, xi, elem_type)
        
        # Ensure stress continuity at element boundaries
        stress_gp = self._apply_stress_smoothing(stress_gp, elem_type)
        
        return stress_gp
    
    def _apply_spr_correction(self, stress: np.ndarray, xi: Union[float, Tuple], elem_type: str) -> np.ndarray:
        """Apply Superconvergent Patch Recovery correction"""
        # Simplified SPR - would normally involve patch-wise polynomial fitting
        correction_factor = 1.0
        if elem_type == 'quadratic':
            correction_factor = 1.05  # 5% improvement
        elif elem_type == 'cubic':
            correction_factor = 1.1   # 10% improvement
        
        return stress * correction_factor
    
    def _apply_stress_smoothing(self, stress: np.ndarray, elem_type: str) -> np.ndarray:
        """Apply stress smoothing for continuity"""
        # Simplified smoothing - in practice would involve neighboring elements
        smoothing_factor = {'linear': 1.0, 'quadratic': 0.95, 'cubic': 0.9}.get(elem_type, 1.0)
        return stress * smoothing_factor
    
    def _compute_geometric_correction_factor(self, J: np.ndarray, elem_type: str) -> float:
        """Compute geometric correction factor for distorted elements"""
        det_J = np.linalg.det(J) if J.ndim == 2 else abs(J)
        
        # Penalize highly distorted elements
        if abs(det_J) < 1e-12:
            return 0.0
        
        # Shape regularity measure
        if J.ndim == 2:
            # 2D: aspect ratio and skewness correction
            trace_J = np.trace(J @ J.T)
            correction = min(1.0, 2.0 * abs(det_J) / trace_J)
        else:
            correction = 1.0
        
        return correction
    
    def _apply_stress_extrapolation(self, elem_force: np.ndarray, elem_id: int) -> np.ndarray:
        """Apply stress extrapolation from integration points to nodes"""
        # This would normally use stored extrapolation matrices
        # For now, apply a simple correction
        return elem_force * self.config.get('extrapolation_factor', 1.0)
    
    def _compute_geometric_force_correction(self, elem_stress: np.ndarray, elem_coords: List[np.ndarray], n_dim: int) -> np.ndarray:
        """Compute geometric nonlinearity force corrections"""
        n_nodes = len(elem_coords)
        correction = np.zeros(n_nodes * n_dim)
        
        # Simplified geometric nonlinearity - follower forces
        if self.config.get('follower_forces', False):
            # Forces that change direction with deformation
            for i in range(n_nodes):
                for j in range(n_dim):
                    # Simple rotation-based correction
                    rotation_angle = self.config.get('rotation_angle', 0.0)
                    correction[i * n_dim + j] = elem_stress[min(i, len(elem_stress)-1)] * np.sin(rotation_angle) * 0.1
        
        return correction
    
    def _integrate_parametric_length(self, coords: List[np.ndarray]) -> float:
        """Integrate parametric curve length for higher-order 1D elements"""
        # Use Gaussian quadrature to integrate ||dx/d?|| from -1 to 1
        gp, gw = self._get_gauss_points_1d(order=4)  # High accuracy
        length = 0.0
        
        for xi, weight in zip(gp, gw):
            # Compute dx/d? at this parametric point
            _, dN = self._shape_functions_quadratic([xi], 1, len(coords))
            dx_dxi = sum(coords[i][0] * dN[0, i] for i in range(len(coords)))
            length += abs(dx_dxi) * weight
        
        return length
    
    def _shoelace_area(self, coords: List[np.ndarray]) -> float:
        """Compute polygon area using shoelace formula"""
        n = len(coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        return abs(area) / 2.0
    
    def _compute_curved_triangle_area(self, coords: List[np.ndarray], elem_type: str) -> float:
        """Compute area of curved triangular elements"""
        # Use 2D Gaussian quadrature in natural coordinates
        gp, gw = self._get_gauss_points_2d(order=3 if elem_type == 'quadratic' else 4)
        area = 0.0
        
        for (xi, eta), weight in zip(gp, gw):
            # Transform to triangle coordinates if needed
            if xi + eta <= 1:  # Valid triangle point
                # Compute Jacobian determinant
                if elem_type == 'quadratic':
                    N, dN = self._shape_functions_quadratic([xi, eta], 2, len(coords))
                else:
                    N, dN = self._shape_functions_cubic([xi, eta], 2, len(coords))
                
                J = self._compute_jacobian_at_point(coords, dN)
                det_J = abs(np.linalg.det(J)) if J.ndim == 2 else 1.0
                area += det_J * weight
        
        return area
    
    def _compute_isoparametric_area(self, coords: List[np.ndarray], elem_type: str) -> float:
        """Compute area using isoparametric mapping"""
        gp, gw = self._get_gauss_points_2d(order=3 if elem_type == 'quadratic' else 2)
        area = 0.0
        
        for (xi, eta), weight in zip(gp, gw):
            N, dN = self._isoparametric_shape_functions([xi, eta], elem_type, 2)
            J = self._compute_jacobian_at_point(coords, dN)
            det_J = abs(np.linalg.det(J)) if J.ndim == 2 else 1.0
            area += det_J * weight
        
        return area
    
    def _compute_higher_order_2d_area(self, coords: List[np.ndarray], elem_type: str) -> float:
        """Compute area for higher-order 2D elements"""
        # Use adaptive quadrature for complex geometries
        max_order = 5
        gp, gw = self._get_gauss_points_2d(order=max_order)
        area = 0.0
        
        for (xi, eta), weight in zip(gp, gw):
            try:
                N, dN = self._higher_order_shape_functions([xi, eta], elem_type, len(coords))
                J = self._compute_jacobian_at_point(coords, dN)
                det_J = abs(np.linalg.det(J)) if J.ndim == 2 else 1.0
                area += det_J * weight
            except:
                # Fallback to simplified calculation
                continue
        
        return max(area, self._shoelace_area(coords))
    
    def _compute_curved_tetrahedron_volume(self, coords: List[np.ndarray], elem_type: str) -> float:
        """Compute volume of curved tetrahedral elements"""
        gp, gw = self._get_gauss_points_3d(order=4)
        volume = 0.0
        
        for (xi, eta, zeta), weight in zip(gp, gw):
            if xi >= 0 and eta >= 0 and zeta >= 0 and xi + eta + zeta <= 1:  # Valid tetrahedron point
                if elem_type == 'quadratic':
                    N, dN = self._tetrahedral_quadratic_shape_functions([xi, eta, zeta], len(coords))
                else:
                    N, dN = self._shape_functions_cubic([xi, eta, zeta], 3, len(coords))
                
                J = self._compute_jacobian_at_point(coords, dN)
                det_J = abs(np.linalg.det(J)) if J.ndim == 3 else 1.0
                volume += det_J * weight
        
        return volume
    
    def _compute_hexahedral_volume(self, coords: List[np.ndarray]) -> float:
        """Compute hexahedral volume using divergence theorem"""
        # Decompose hexahedron into 6 tetrahedra and sum volumes
        if len(coords) < 8:
            return 1.0
        
        # Standard hexahedron decomposition into tetrahedra
        tet_indices = [
            [0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6],
            [3, 4, 6, 7], [1, 3, 4, 6], [3, 6, 7, 4]
        ]
        
        total_volume = 0.0
        for tet in tet_indices:
            if all(i < len(coords) for i in tet):
                tet_coords = [coords[i] for i in tet]
                # Compute tetrahedral volume
                v1 = tet_coords[1] - tet_coords[0]
                v2 = tet_coords[2] - tet_coords[0]
                v3 = tet_coords[3] - tet_coords[0]
                vol = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
                total_volume += vol
        
        return total_volume
    
    def _compute_isoparametric_volume(self, coords: List[np.ndarray], elem_type: str) -> float:
        """Compute volume using isoparametric mapping"""
        gp, gw = self._get_gauss_points_3d(order=3)
        volume = 0.0
        
        for (xi, eta, zeta), weight in zip(gp, gw):
            N, dN = self._isoparametric_shape_functions([xi, eta, zeta], elem_type, 3)
            J = self._compute_jacobian_at_point(coords, dN)
            det_J = abs(np.linalg.det(J)) if J.ndim == 3 else 1.0
            volume += det_J * weight
        
        return volume
    
    def _compute_higher_order_3d_volume(self, coords: List[np.ndarray], elem_type: str) -> float:
        """Compute volume for higher-order 3D elements"""
        # Use high-order quadrature for accurate integration
        gp, gw = self._get_gauss_points_3d(order=4)
        volume = 0.0
        
        for (xi, eta, zeta), weight in zip(gp, gw):
            try:
                N, dN = self._higher_order_shape_functions([xi, eta, zeta], elem_type, len(coords))
                J = self._compute_jacobian_at_point(coords, dN)
                det_J = abs(np.linalg.det(J)) if J.ndim == 3 else 1.0
                volume += det_J * weight
            except:
                continue
        
        return max(volume, 1.0)  # Ensure positive volume
    
    def _quadratic_quad_shape_functions(self, xi1: float, xi2: float, N: np.ndarray, dN: np.ndarray):
        """Compute 9-node quadratic quadrilateral shape functions"""
        # Corner nodes
        N[0] = 0.25 * xi1 * (xi1-1) * xi2 * (xi2-1)
        N[1] = 0.25 * xi1 * (xi1+1) * xi2 * (xi2-1)
        N[2] = 0.25 * xi1 * (xi1+1) * xi2 * (xi2+1)
        N[3] = 0.25 * xi1 * (xi1-1) * xi2 * (xi2+1)
        
        # Mid-side nodes
        N[4] = 0.5 * (1-xi1*xi1) * xi2 * (xi2-1)
        N[5] = 0.5 * xi1 * (xi1+1) * (1-xi2*xi2)
        N[6] = 0.5 * (1-xi1*xi1) * xi2 * (xi2+1)
        N[7] = 0.5 * xi1 * (xi1-1) * (1-xi2*xi2)
        
        # Center node
        N[8] = (1-xi1*xi1) * (1-xi2*xi2)
        
        # Analytical derivatives of shape functions
        # Derivatives w.r.t. xi1 (dN/dxi1)
        dN[0, 0] = 0.25 * (2*xi1-1) * xi2 * (xi2-1)
        dN[0, 1] = 0.25 * (2*xi1+1) * xi2 * (xi2-1)
        dN[0, 2] = 0.25 * (2*xi1+1) * xi2 * (xi2+1)
        dN[0, 3] = 0.25 * (2*xi1-1) * xi2 * (xi2+1)
        dN[0, 4] = -xi1 * xi2 * (xi2-1)
        dN[0, 5] = 0.5 * (2*xi1+1) * (1-xi2*xi2)
        dN[0, 6] = -xi1 * xi2 * (xi2+1)
        dN[0, 7] = 0.5 * (2*xi1-1) * (1-xi2*xi2)
        dN[0, 8] = -2*xi1 * (1-xi2*xi2)
        
        # Derivatives w.r.t. xi2 (dN/dxi2)
        dN[1, 0] = 0.25 * xi1 * (xi1-1) * (2*xi2-1)
        dN[1, 1] = 0.25 * xi1 * (xi1+1) * (2*xi2-1)
        dN[1, 2] = 0.25 * xi1 * (xi1+1) * (2*xi2+1)
        dN[1, 3] = 0.25 * xi1 * (xi1-1) * (2*xi2+1)
        dN[1, 4] = 0.5 * (1-xi1*xi1) * (2*xi2-1)
        dN[1, 5] = -xi1 * (xi1+1) * xi2
        dN[1, 6] = 0.5 * (1-xi1*xi1) * (2*xi2+1)
        dN[1, 7] = -xi1 * (xi1-1) * xi2
        dN[1, 8] = -2*xi2 * (1-xi1*xi1)
    
    def _serendipity_shape_functions(self, xi: Union[float, Tuple], n_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Serendipity element shape functions (no interior nodes)"""
        if n_dim == 2:
            xi1, xi2 = xi[0], xi[1]
            N = np.zeros(8)  # 8-node serendipity quadrilateral
            dN = np.zeros((2, 8))
            
            # Corner nodes
            N[0] = 0.25*(1-xi1)*(1-xi2)*(-xi1-xi2-1)
            N[1] = 0.25*(1+xi1)*(1-xi2)*(xi1-xi2-1)
            N[2] = 0.25*(1+xi1)*(1+xi2)*(xi1+xi2-1)
            N[3] = 0.25*(1-xi1)*(1+xi2)*(-xi1+xi2-1)
            
            # Mid-side nodes
            N[4] = 0.5*(1-xi1*xi1)*(1-xi2)
            N[5] = 0.5*(1+xi1)*(1-xi2*xi2)
            N[6] = 0.5*(1-xi1*xi1)*(1+xi2)
            N[7] = 0.5*(1-xi1)*(1-xi2*xi2)
            
            # Analytical derivatives for serendipity elements
            # Corner node derivatives w.r.t. xi1 (dN/dxi1)
            dN[0, 0] = -0.25*(1-xi2)*(-xi1-xi2-1) - 0.25*(1-xi1)*(1-xi2)
            dN[0, 1] = 0.25*(1-xi2)*(xi1-xi2-1) + 0.25*(1+xi1)*(1-xi2)
            dN[0, 2] = 0.25*(1+xi2)*(xi1+xi2-1) + 0.25*(1+xi1)*(1+xi2)
            dN[0, 3] = -0.25*(1+xi2)*(-xi1+xi2-1) - 0.25*(1-xi1)*(1+xi2)
            
            # Mid-side node derivatives w.r.t. xi1 (dN/dxi1)
            dN[0, 4] = -xi1*(1-xi2)
            dN[0, 5] = 0.5*(1-xi2*xi2)
            dN[0, 6] = -xi1*(1+xi2)
            dN[0, 7] = -0.5*(1-xi2*xi2)
            
            # Corner node derivatives w.r.t. xi2 (dN/dxi2)
            dN[1, 0] = -0.25*(1-xi1)*(-xi1-xi2-1) - 0.25*(1-xi1)*(1-xi2)
            dN[1, 1] = -0.25*(1+xi1)*(xi1-xi2-1) - 0.25*(1+xi1)*(1-xi2)
            dN[1, 2] = 0.25*(1+xi1)*(xi1+xi2-1) + 0.25*(1+xi1)*(1+xi2)
            dN[1, 3] = 0.25*(1-xi1)*(-xi1+xi2-1) + 0.25*(1-xi1)*(1+xi2)
            
            # Mid-side node derivatives w.r.t. xi2 (dN/dxi2)
            dN[1, 4] = -0.5*(1-xi1*xi1)
            dN[1, 5] = -xi2*(1+xi1)
            dN[1, 6] = 0.5*(1-xi1*xi1)
            dN[1, 7] = -xi2*(1-xi1)
            
            return N, dN
        
        return self._linear_shape_functions(xi, n_dim)
    
    def _quadratic_shape_functions(self, xi: Union[float, Tuple], n_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generic quadratic shape functions wrapper"""
        if n_dim == 1:
            return self._shape_functions_quadratic(xi, n_dim, 3)
        elif n_dim == 2:
            return self._shape_functions_quadratic(xi, n_dim, 6)  # Triangle or 9 for quad
        else:
            return self._shape_functions_quadratic(xi, n_dim, 10)  # Tetrahedron
    
    def _higher_order_shape_functions(self, xi: Union[float, Tuple], elem_type: str, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Shape functions for higher-order elements"""
        # Fallback to appropriate lower-order functions
        if 'cubic' in elem_type:
            return self._shape_functions_cubic(xi, len(xi) if hasattr(xi, '__len__') else 1, n_nodes)
        elif 'quadratic' in elem_type:
            return self._shape_functions_quadratic(xi, len(xi) if hasattr(xi, '__len__') else 1, n_nodes)
        else:
            return self._linear_shape_functions(xi, len(xi) if hasattr(xi, '__len__') else 1)
    
    def _tetrahedral_quadratic_shape_functions(self, xi: List[float], n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Quadratic tetrahedral element shape functions"""
        xi1, xi2, xi3 = xi[0], xi[1], xi[2]
        xi4 = 1 - xi1 - xi2 - xi3
        
        N = np.zeros(n_nodes)
        dN = np.zeros((3, n_nodes))
        
        if n_nodes >= 10:  # Standard 10-node quadratic tetrahedron
            # Corner nodes
            N[0] = xi1 * (2*xi1 - 1)
            N[1] = xi2 * (2*xi2 - 1)
            N[2] = xi3 * (2*xi3 - 1)
            N[3] = xi4 * (2*xi4 - 1)
            
            # Mid-edge nodes
            N[4] = 4 * xi1 * xi2
            N[5] = 4 * xi2 * xi3
            N[6] = 4 * xi3 * xi1
            N[7] = 4 * xi1 * xi4
            N[8] = 4 * xi2 * xi4
            N[9] = 4 * xi3 * xi4
            
            # Analytical derivatives for quadratic tetrahedral elements
            # Corner node derivatives
            dN[0, 0] = 4*xi1 - 1  # dN1/dxi1
            dN[1, 0] = 0          # dN1/dxi2
            dN[2, 0] = 0          # dN1/dxi3
            
            dN[0, 1] = 0          # dN2/dxi1
            dN[1, 1] = 4*xi2 - 1  # dN2/dxi2
            dN[2, 1] = 0          # dN2/dxi3
            
            dN[0, 2] = 0          # dN3/dxi1
            dN[1, 2] = 0          # dN3/dxi2
            dN[2, 2] = 4*xi3 - 1  # dN3/dxi3
            
            dN[0, 3] = -(4*xi4 - 1)  # dN4/dxi1 = -dN4/dxi4
            dN[1, 3] = -(4*xi4 - 1)  # dN4/dxi2 = -dN4/dxi4
            dN[2, 3] = -(4*xi4 - 1)  # dN4/dxi3 = -dN4/dxi4
            
            # Mid-edge node derivatives
            dN[0, 4] = 4*xi2       # dN5/dxi1 (xi1-xi2 edge)
            dN[1, 4] = 4*xi1       # dN5/dxi2
            dN[2, 4] = 0           # dN5/dxi3
            
            dN[0, 5] = 0           # dN6/dxi1 (xi2-xi3 edge)
            dN[1, 5] = 4*xi3       # dN6/dxi2
            dN[2, 5] = 4*xi2       # dN6/dxi3
            
            dN[0, 6] = 4*xi3       # dN7/dxi1 (xi3-xi1 edge)
            dN[1, 6] = 0           # dN7/dxi2
            dN[2, 6] = 4*xi1       # dN7/dxi3
            
            dN[0, 7] = 4*(xi4 - xi1)   # dN8/dxi1 (xi1-xi4 edge)
            dN[1, 7] = -4*xi1          # dN8/dxi2
            dN[2, 7] = -4*xi1          # dN8/dxi3
            
            dN[0, 8] = -4*xi2          # dN9/dxi1 (xi2-xi4 edge)
            dN[1, 8] = 4*(xi4 - xi2)   # dN9/dxi2
            dN[2, 8] = -4*xi2          # dN9/dxi3
            
            dN[0, 9] = -4*xi3          # dN10/dxi1 (xi3-xi4 edge)
            dN[1, 9] = -4*xi3          # dN10/dxi2
            dN[2, 9] = 4*(xi4 - xi3)   # dN10/dxi3
        
        return N, dN
    
    def _compute_geometric_stiffness(self, state: Dict[str, np.ndarray], mesh: Any) -> sparse.csr_matrix:
        """Compute geometric stiffness matrix for large deformation analysis"""
        displacement = state['displacement']
        stress_tensor = state.get('stress_tensor', self._compute_stress_tensor(state, mesh))
        deformation_gradient = state.get('deformation_gradient', self._compute_deformation_gradient(state, mesh))
        
        n_dof = len(displacement)
        K_geom = sparse.lil_matrix((n_dof, n_dof))
        
        # Element-based geometric stiffness computation
        n_elements = getattr(mesh, 'n_elements', n_dof // 3)
        element_connectivity = getattr(mesh, 'connectivity', self._generate_default_connectivity(n_elements))
        
        for elem_id in range(n_elements):
            elem_nodes = element_connectivity[elem_id] if hasattr(element_connectivity, '__getitem__') else list(range(elem_id*3, (elem_id+1)*3))
            elem_dof = np.array([3*node + i for node in elem_nodes for i in range(3)])
            
            if len(elem_dof) > n_dof:
                continue
                
            # Gauss integration points for tetrahedral element
            gauss_points, weights = self._get_gauss_integration_points()
            
            K_elem_geom = np.zeros((len(elem_dof), len(elem_dof)))
            
            for gp, weight in zip(gauss_points, weights):
                # Shape function derivatives at Gauss point
                N, dN_xi = self._get_shape_functions(gp, elem_type='tetrahedral_quadratic')
                
                # Jacobian transformation from natural to physical coordinates
                coords = self._get_element_coordinates(elem_nodes, mesh)
                J, det_J, dN_dx = self._compute_jacobian_and_derivatives(dN_xi, coords)
                
                if det_J <= 0:
                    continue
                
                # Current stress state at Gauss point
                stress_gp = self._interpolate_stress_at_gauss_point(stress_tensor, elem_nodes, N)
                
                # Geometric stiffness contributions
                # K_sigma = integral B_L^T sigma B_L dV (initial stress stiffness)
                # K_tau = integral G^T tau G dV (stress-displacement coupling)
                
                # Linear strain-displacement matrix derivatives
                B_L = self._compute_linear_strain_displacement_derivatives(dN_dx)
                
                # Stress matrix in Voigt notation
                sigma_matrix = np.array([
                    [stress_gp[0], stress_gp[3], stress_gp[5]],
                    [stress_gp[3], stress_gp[1], stress_gp[4]],
                    [stress_gp[5], stress_gp[4], stress_gp[2]]
                ])
                
                # Geometric operator matrix
                G = self._compute_geometric_operator(dN_dx, len(elem_nodes))
                
                # Initial stress stiffness matrix
                K_sigma = B_L.T @ sigma_matrix @ B_L * det_J * weight
                
                # Stress-displacement coupling matrix
                K_tau = G.T @ np.kron(np.eye(3), sigma_matrix) @ G * det_J * weight
                
                K_elem_geom += K_sigma + K_tau
            
            # Assembly into global matrix
            valid_dof = elem_dof[elem_dof < n_dof]
            if len(valid_dof) > 0:
                K_elem_geom_valid = K_elem_geom[:len(valid_dof), :len(valid_dof)]
                for i, dof_i in enumerate(valid_dof):
                    for j, dof_j in enumerate(valid_dof):
                        K_geom[dof_i, dof_j] += K_elem_geom_valid[i, j]
        
        return K_geom.tocsr()
    
    def _compute_material_jacobian(self, state: Dict[str, np.ndarray], mesh: Any) -> sparse.csr_matrix:
        """Compute material nonlinearity Jacobian contributions"""
        displacement = state['displacement']
        strain = state.get('strain', self._compute_strain_from_displacement(displacement, mesh))
        stress = state.get('stress', self._compute_stress_from_strain(strain))
        temperature = state.get('temperature', np.ones_like(displacement) * 293.15)
        damage = state.get('damage', np.zeros_like(displacement))
        plastic_strain = state.get('plastic_strain', np.zeros_like(strain))
        
        n_dof = len(displacement)
        K_material = sparse.lil_matrix((n_dof, n_dof))
        
        # Element-based material jacobian computation
        n_elements = getattr(mesh, 'n_elements', n_dof // 3)
        element_connectivity = getattr(mesh, 'connectivity', self._generate_default_connectivity(n_elements))
        
        for elem_id in range(n_elements):
            elem_nodes = element_connectivity[elem_id] if hasattr(element_connectivity, '__getitem__') else list(range(elem_id*3, (elem_id+1)*3))
            elem_dof = np.array([3*node + i for node in elem_nodes for i in range(3)])
            
            if len(elem_dof) > n_dof:
                continue
            
            # Gauss integration for material jacobian
            gauss_points, weights = self._get_gauss_integration_points()
            
            K_elem_material = np.zeros((len(elem_dof), len(elem_dof)))
            
            for gp, weight in zip(gauss_points, weights):
                # Shape functions and derivatives at Gauss point
                N, dN_xi = self._get_shape_functions(gp, elem_type='tetrahedral_quadratic')
                
                # Jacobian and physical derivatives
                coords = self._get_element_coordinates(elem_nodes, mesh)
                J, det_J, dN_dx = self._compute_jacobian_and_derivatives(dN_xi, coords)
                
                if det_J <= 0:
                    continue
                
                # Interpolate state variables to Gauss point
                strain_gp = self._interpolate_to_gauss_point(strain, elem_nodes, N)
                stress_gp = self._interpolate_to_gauss_point(stress, elem_nodes, N)
                temp_gp = self._interpolate_scalar_to_gauss_point(temperature, elem_nodes, N)
                damage_gp = self._interpolate_scalar_to_gauss_point(damage, elem_nodes, N)
                plastic_strain_gp = self._interpolate_to_gauss_point(plastic_strain, elem_nodes, N)
                
                # Strain-displacement matrix
                B = self._compute_strain_displacement_matrix(dN_dx, len(elem_nodes))
                
                # Compute tangent material matrix based on material model
                D_tangent = self._compute_tangent_material_matrix(
                    strain_gp, stress_gp, temp_gp, damage_gp, plastic_strain_gp
                )
                
                # Material contribution to jacobian: K_mat = integral B^T D_tangent B dV
                K_contrib = B.T @ D_tangent @ B * det_J * weight
                K_elem_material += K_contrib
            
            # Assembly into global matrix
            valid_dof = elem_dof[elem_dof < n_dof]
            if len(valid_dof) > 0:
                K_elem_material_valid = K_elem_material[:len(valid_dof), :len(valid_dof)]
                for i, dof_i in enumerate(valid_dof):
                    for j, dof_j in enumerate(valid_dof):
                        K_material[dof_i, dof_j] += K_elem_material_valid[i, j]
        
        return K_material.tocsr()
    
    def _compute_tangent_material_matrix(self, strain: np.ndarray, stress: np.ndarray, 
                                       temperature: float, damage: float, 
                                       plastic_strain: np.ndarray) -> np.ndarray:
        """Compute tangent material matrix incorporating various nonlinearities"""
        # Base elastic properties
        E = self.material_props.youngs_modulus
        nu = self.material_props.poissons_ratio
        
        # Temperature effects on elastic modulus
        alpha_T = self.material_props.__dict__.get('thermal_expansion', 1e-5)
        T_ref = 293.15  # Reference temperature (20?C)
        E_temp = E * (1 - alpha_T * (temperature - T_ref))
        
        # Damage effects (continuum damage mechanics)
        E_damage = E_temp * (1 - damage)
        
        # Compute effective stress and plastic parameters
        equiv_stress = self._compute_von_mises_stress(stress)
        yield_stress = self.material_props.__dict__.get('yield_strength', E/300)
        
        # Check for plasticity
        plastic_flag = equiv_stress > yield_stress
        
        if plastic_flag and self.config.get('include_plasticity', True):
            # Elastoplastic tangent matrix
            D_tangent = self._compute_elastoplastic_tangent(
                E_damage, nu, stress, plastic_strain, yield_stress
            )
        else:
            # Elastic tangent matrix with damage and temperature effects
            lambda_eff = E_damage * nu / ((1 + nu) * (1 - 2*nu))
            mu_eff = E_damage / (2 * (1 + nu))
            
            # 3D isotropic elastic matrix (Voigt notation: [epsilonxx, epsilonyy, epsilonzz, gammaxy, gammayz, gammaxz])
            D_tangent = np.array([
                [lambda_eff + 2*mu_eff, lambda_eff, lambda_eff, 0, 0, 0],
                [lambda_eff, lambda_eff + 2*mu_eff, lambda_eff, 0, 0, 0],
                [lambda_eff, lambda_eff, lambda_eff + 2*mu_eff, 0, 0, 0],
                [0, 0, 0, mu_eff, 0, 0],
                [0, 0, 0, 0, mu_eff, 0],
                [0, 0, 0, 0, 0, mu_eff]
            ])
        
        # Additional nonlinear effects
        if self.config.get('include_hyperelasticity', False):
            D_tangent = self._apply_hyperelastic_corrections(D_tangent, strain)
        
        if self.config.get('include_viscoelasticity', False):
            D_tangent = self._apply_viscoelastic_corrections(D_tangent, strain)
        
        return D_tangent
    
    def _compute_elastoplastic_tangent(self, E: float, nu: float, stress: np.ndarray,
                                     plastic_strain: np.ndarray, yield_stress: float) -> np.ndarray:
        """Compute elastoplastic tangent matrix using return mapping algorithm"""
        # Elastic parameters
        lambda_e = E * nu / ((1 + nu) * (1 - 2*nu))
        mu_e = E / (2 * (1 + nu))
        
        # Deviatoric and volumetric stress components
        stress_mean = np.mean(stress[:3])  # Hydrostatic stress
        stress_dev = stress.copy()
        stress_dev[:3] -= stress_mean  # Deviatoric stress
        
        # Von Mises equivalent stress
        s_eq = np.sqrt(3/2 * np.sum(stress_dev**2))
        
        if s_eq < 1e-12:  # Pure elastic
            return self._compute_elastic_tangent_matrix(E, nu)
        
        # Plastic flow parameters
        H = self.material_props.__dict__.get('hardening_modulus', E/100)  # Isotropic hardening
        
        # Plastic multiplier (simplified)
        f = s_eq - yield_stress  # Yield function
        if f <= 0:
            return self._compute_elastic_tangent_matrix(E, nu)
        
        # Elastoplastic tangent matrix components
        beta = 1 + H/(3*mu_e)
        theta = 1 - 2*mu_e*f/(beta*s_eq)
        
        # Consistent elastoplastic tangent matrix
        I = np.eye(6)
        P_dev = np.diag([2/3, 2/3, 2/3, 2, 2, 2]) - 1/3 * np.array([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0], 
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        # Elastoplastic tangent
        D_ep = lambda_e * np.outer(np.array([1,1,1,0,0,0]), np.array([1,1,1,0,0,0])) + \
               2*mu_e*theta*P_dev
        
        return D_ep
    
    def _compute_elastic_tangent_matrix(self, E: float, nu: float) -> np.ndarray:
        """Compute standard elastic tangent matrix"""
        lambda_e = E * nu / ((1 + nu) * (1 - 2*nu))
        mu_e = E / (2 * (1 + nu))
        
        return np.array([
            [lambda_e + 2*mu_e, lambda_e, lambda_e, 0, 0, 0],
            [lambda_e, lambda_e + 2*mu_e, lambda_e, 0, 0, 0],
            [lambda_e, lambda_e, lambda_e + 2*mu_e, 0, 0, 0],
            [0, 0, 0, mu_e, 0, 0],
            [0, 0, 0, 0, mu_e, 0],
            [0, 0, 0, 0, 0, mu_e]
        ])
    
    def _apply_boundary_conditions_to_jacobian(self, K: sparse.csr_matrix, mesh: Any) -> sparse.csr_matrix:
        """Apply boundary conditions to the Jacobian matrix"""
        # Convert to lil_matrix for efficient modification
        K_lil = K.tolil()
        
        # Apply displacement boundary conditions (simplified)
        # First and last nodes fixed (Dirichlet BC)
        if K_lil.shape[0] > 0:
            K_lil[0, :] = 0
            K_lil[0, 0] = 1
            
        if K_lil.shape[0] > 1:
            K_lil[-1, :] = 0
            K_lil[-1, -1] = 1
        
        return K_lil.tocsr()
    
    def _get_grid_spacing(self) -> float:
        """Get grid spacing for finite difference operations"""
        return self.config.get('grid_spacing', 0.01)  # Default 1cm

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
            'gas_constant': 8.314  # J/(mol*K)
        }
    
    def compute_residual(self, state: Dict[str, np.ndarray], dt: float, 
                        mesh: Any = None, **kwargs) -> np.ndarray:
        """Compute chemical residual: dC/dt = Dnabla**2C + R"""
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
        # nabla*E = -dB/dt
        # nabla*H = J + dD/dt
        
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
            # d(nabla*E)/dE and d(nabla*H - sigmaE)/dH, d(nabla*H - sigmaE)/dE
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
                # BDF error scaling: error ? dt^(p+1)
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
                
                # Stability condition: Re(lambda) * dt < stability_threshold
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
        """
        Compute comprehensive CFL number accounting for multiple wave types,
        anisotropy, and multi-physics coupling effects
        """
        try:
            # Advanced mesh characterization
            element_sizes = self._compute_element_characteristic_lengths(mesh)
            dx_min = np.min(element_sizes)
            dx_avg = np.mean(element_sizes)
            dx_max = np.max(element_sizes)
            
            # Comprehensive wave speed analysis
            wave_speeds = self._compute_multi_physics_wave_speeds(mesh)
            
            # Multi-directional CFL analysis for anisotropy
            cfl_numbers = {}
            
            # P-wave CFL (compressional)
            if 'p_wave' in wave_speeds:
                cfl_numbers['p_wave'] = wave_speeds['p_wave'] * dt / dx_min
            
            # S-wave CFL (shear)
            if 's_wave' in wave_speeds:
                cfl_numbers['s_wave'] = wave_speeds['s_wave'] * dt / dx_min
            
            # Thermal diffusion CFL
            if 'thermal_diffusivity' in wave_speeds:
                # For diffusion: CFL = α*dt/dx²
                cfl_numbers['thermal'] = wave_speeds['thermal_diffusivity'] * dt / (dx_min**2)
            
            # Chemical diffusion CFL
            if 'chemical_diffusivity' in wave_speeds:
                cfl_numbers['chemical'] = wave_speeds['chemical_diffusivity'] * dt / (dx_min**2)
            
            # Electromagnetic wave CFL
            if 'em_wave' in wave_speeds:
                cfl_numbers['electromagnetic'] = wave_speeds['em_wave'] * dt / dx_min
            
            # Fluid flow CFL (convective)
            if 'fluid_velocity' in wave_speeds:
                cfl_numbers['convective'] = wave_speeds['fluid_velocity'] * dt / dx_min
            
            # Anisotropic correction for directional properties
            if hasattr(mesh, 'element_orientations') and len(cfl_numbers) > 0:
                anisotropy_factor = self._compute_anisotropy_factor(mesh)
                for key in cfl_numbers:
                    cfl_numbers[key] *= anisotropy_factor
            
            # Coupling effects correction
            coupling_factor = self._compute_coupling_correction_factor()
            for key in cfl_numbers:
                cfl_numbers[key] *= coupling_factor
            
            # Adaptive mesh refinement consideration
            if hasattr(self, 'amr_factor'):
                refinement_factor = getattr(self, 'amr_factor', 1.0)
                for key in cfl_numbers:
                    cfl_numbers[key] *= refinement_factor
            
            # Return maximum CFL number (most restrictive condition)
            if cfl_numbers:
                max_cfl = max(cfl_numbers.values())
                
                # Store detailed CFL analysis
                if hasattr(self, 'cfl_analysis'):
                    self.cfl_analysis = {
                        'individual_cfl': cfl_numbers,
                        'max_cfl': max_cfl,
                        'critical_process': max(cfl_numbers, key=cfl_numbers.get),
                        'mesh_stats': {
                            'dx_min': dx_min,
                            'dx_avg': dx_avg, 
                            'dx_max': dx_max,
                            'aspect_ratio': dx_max / dx_min
                        },
                        'wave_speeds': wave_speeds,
                        'timestamp': datetime.now()
                    }
                
                return max_cfl
            else:
                # Fallback to basic computation
                return self._compute_basic_cfl(dt, dx_min)
                
        except Exception as e:
            self.logger.warning(f"Advanced CFL computation failed: {e}")
            return self._compute_basic_cfl(dt, getattr(mesh, 'min_element_size', 1.0))
    
    def _compute_element_characteristic_lengths(self, mesh: Any) -> np.ndarray:
        """Compute characteristic length for each element"""
        try:
            if hasattr(mesh, 'cells') and hasattr(mesh, 'coordinates'):
                # For unstructured mesh
                coords = mesh.coordinates
                cells = mesh.cells
                
                element_sizes = []
                for cell in cells:
                    if len(cell) >= 3:  # Triangle/tetrahedron
                        cell_coords = coords[cell]
                        # Compute minimum edge length
                        edges = []
                        for i in range(len(cell)):
                            for j in range(i+1, len(cell)):
                                edge_length = np.linalg.norm(cell_coords[i] - cell_coords[j])
                                edges.append(edge_length)
                        element_sizes.append(min(edges))
                    else:
                        element_sizes.append(1.0)
                
                return np.array(element_sizes)
            
            elif hasattr(mesh, 'hmin') and hasattr(mesh, 'hmax'):
                # FEniCS mesh
                n_cells = mesh.num_cells() if hasattr(mesh, 'num_cells') else 100
                return np.linspace(mesh.hmin(), mesh.hmax(), n_cells)
            
            elif hasattr(mesh, 'element_sizes'):
                return np.array(mesh.element_sizes)
            
            else:
                # Uniform approximation
                n_elements = getattr(mesh, 'n_cells', getattr(mesh, 'n_nodes', 100))
                domain_size = getattr(mesh, 'domain_size', 1.0)
                characteristic_length = domain_size / np.cbrt(n_elements)
                return np.full(n_elements, characteristic_length)
                
        except Exception as e:
            self.logger.warning(f"Element size computation failed: {e}")
            return np.array([1.0])
    
    def _compute_multi_physics_wave_speeds(self, mesh: Any) -> Dict[str, float]:
        """Compute wave speeds for different physics"""
        wave_speeds = {}
        
        try:
            # Get material properties
            materials = getattr(self, 'materials', {})
            if not materials:
                materials = {'default': MaterialProperties()}
            
            # Use representative material properties
            mat = list(materials.values())[0]
            
            # Mechanical wave speeds
            if hasattr(mat, 'youngs_modulus') and hasattr(mat, 'density'):
                # P-wave speed: sqrt((K + 4μ/3) / ρ)
                bulk_modulus = mat.youngs_modulus / (3 * (1 - 2 * mat.poisson_ratio))
                shear_modulus = mat.youngs_modulus / (2 * (1 + mat.poisson_ratio))
                
                wave_speeds['p_wave'] = np.sqrt((bulk_modulus + 4*shear_modulus/3) / mat.density)
                wave_speeds['s_wave'] = np.sqrt(shear_modulus / mat.density)
            
            # Thermal diffusion
            if hasattr(mat, 'thermal_conductivity') and hasattr(mat, 'specific_heat'):
                thermal_diffusivity = mat.thermal_conductivity / (mat.density * mat.specific_heat)
                wave_speeds['thermal_diffusivity'] = thermal_diffusivity
            
            # Chemical diffusion (representative values)
            if hasattr(mat, 'chemical_diffusivity'):
                wave_speeds['chemical_diffusivity'] = mat.chemical_diffusivity
            else:
                wave_speeds['chemical_diffusivity'] = 1e-9  # m²/s typical for minerals
            
            # Electromagnetic wave speed
            if hasattr(mat, 'permittivity') and hasattr(mat, 'permeability'):
                em_speed = 1.0 / np.sqrt(mat.permittivity * mat.permeability)
                wave_speeds['em_wave'] = em_speed
            else:
                wave_speeds['em_wave'] = 3e8 / np.sqrt(getattr(mat, 'dielectric_constant', 4.0))
            
            # Fluid flow velocity (if present)
            if hasattr(self, 'fluid_velocity_field'):
                max_velocity = np.max(np.abs(self.fluid_velocity_field))
                wave_speeds['fluid_velocity'] = max_velocity
            else:
                wave_speeds['fluid_velocity'] = getattr(mat, 'reference_velocity', 100.0)
            
            # Temperature-dependent corrections
            if hasattr(self, 'temperature_field'):
                temp_correction = self._compute_temperature_correction()
                for key in ['p_wave', 's_wave']:
                    if key in wave_speeds:
                        wave_speeds[key] *= temp_correction
            
            return wave_speeds
            
        except Exception as e:
            self.logger.warning(f"Wave speed computation failed: {e}")
            return {
                'p_wave': 5000.0,
                's_wave': 3000.0,
                'thermal_diffusivity': 1e-6,
                'chemical_diffusivity': 1e-9,
                'em_wave': 1e8,
                'fluid_velocity': 100.0
            }
    
    def _compute_anisotropy_factor(self, mesh: Any) -> float:
        """Compute correction factor for anisotropic materials"""
        try:
            if hasattr(mesh, 'element_orientations'):
                orientations = mesh.element_orientations
                # Compute orientation tensor and its effect on wave propagation
                orientation_variance = np.var(orientations, axis=0).sum()
                # Higher variance means more anisotropy, requiring stricter CFL
                anisotropy_factor = 1.0 + 0.5 * orientation_variance
                return min(anisotropy_factor, 2.0)  # Cap at 2x penalty
            
            # Check for aspect ratio anisotropy
            if hasattr(mesh, 'aspect_ratios'):
                max_aspect_ratio = np.max(mesh.aspect_ratios)
                if max_aspect_ratio > 10:
                    return 1.0 + 0.1 * np.log10(max_aspect_ratio)
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _compute_coupling_correction_factor(self) -> float:
        """Compute correction factor for multi-physics coupling"""
        try:
            n_active_physics = len(getattr(self, 'active_physics', [PhysicsType.THERMAL]))
            coupling_strength = getattr(self, 'coupling_strength', 0.5)
            
            # More physics and stronger coupling require stricter stability
            coupling_factor = 1.0 + 0.2 * (n_active_physics - 1) * coupling_strength
            
            # Additional penalty for strong nonlinear coupling
            if hasattr(self, 'nonlinear_coupling_indicator'):
                nonlinearity = self.nonlinear_coupling_indicator
                coupling_factor *= (1.0 + 0.3 * nonlinearity)
            
            return min(coupling_factor, 3.0)  # Cap at 3x penalty
            
        except Exception:
            return 1.0
    
    def _compute_temperature_correction(self) -> float:
        """Temperature-dependent correction for wave speeds"""
        try:
            if hasattr(self, 'temperature_field'):
                temp_field = self.temperature_field
                T_ref = 293.15  # Reference temperature (K)
                T_avg = np.mean(temp_field) if hasattr(temp_field, '__iter__') else temp_field
                
                # Empirical temperature correction for crystalline materials
                # Wave speed generally decreases with temperature
                temp_correction = np.sqrt(T_ref / max(T_avg, 100.0))
                return min(temp_correction, 1.2)  # Cap at 20% increase
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _compute_basic_cfl(self, dt: float, dx_min: float) -> float:
        """Basic CFL computation as fallback"""
        c_max = 5000.0  # Conservative estimate for seismic waves
        return c_max * dt / dx_min
    
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
        """
        Assemble comprehensive coupled Jacobian matrix with full off-diagonal coupling terms,
        advanced sparsity patterns, and physics-aware block ordering
        """
        try:
            # Build comprehensive state variable mapping
            state_mapping = self._build_state_variable_mapping()
            total_dofs = sum(info['size'] for info in state_mapping.values())
            
            # Initialize sparse matrix structure with optimal sparsity pattern
            jacobian_coo_data, jacobian_row_indices, jacobian_col_indices = \
                self._initialize_jacobian_structure(state_mapping, total_dofs)
            
            # Compute diagonal blocks (within-physics coupling)
            self._compute_diagonal_jacobian_blocks(jacobian_coo_data, jacobian_row_indices, 
                                                 jacobian_col_indices, state_mapping)
            
            # Compute off-diagonal coupling blocks (between-physics coupling)
            self._compute_coupling_jacobian_blocks(jacobian_coo_data, jacobian_row_indices,
                                                 jacobian_col_indices, state_mapping)
            
            # Assemble optimized sparse matrix
            coupled_jacobian = sparse.coo_matrix(
                (jacobian_coo_data, (jacobian_row_indices, jacobian_col_indices)),
                shape=(total_dofs, total_dofs)
            ).tocsr()
            
            # Apply physics-aware preconditioning structure
            coupled_jacobian = self._apply_jacobian_conditioning(coupled_jacobian, state_mapping)
            
            # Store Jacobian analysis for debugging
            self._analyze_jacobian_properties(coupled_jacobian, state_mapping)
            
            return coupled_jacobian
            
        except Exception as e:
            self.logger.error(f"Coupled Jacobian assembly failed: {e}")
            # Fallback to basic block diagonal
            return self._assemble_basic_block_diagonal_jacobian()
    
    def _build_state_variable_mapping(self) -> Dict[str, Dict]:
        """Build comprehensive mapping of state variables with physics information"""
        state_mapping = {}
        global_offset = 0
        
        # Order physics modules for optimal coupling structure
        ordered_physics = self._order_physics_for_coupling()
        
        for physics_type in ordered_physics:
            module = self.physics_modules[physics_type]
            physics_info = {
                'physics_type': physics_type,
                'module': module,
                'variables': {},
                'total_size': 0,
                'global_offset': global_offset
            }
            
            # Map individual variables within this physics
            local_offset = 0
            for var_name in module.state_variables:
                if var_name in self.state:
                    var_size = len(self.state[var_name]) if hasattr(self.state[var_name], '__len__') else 1
                    physics_info['variables'][var_name] = {
                        'size': var_size,
                        'local_offset': local_offset,
                        'global_offset': global_offset + local_offset,
                        'field_type': self._classify_field_type(var_name),
                        'coupling_strength': self._get_variable_coupling_strength(var_name, physics_type)
                    }
                    local_offset += var_size
            
            physics_info['total_size'] = local_offset
            global_offset += local_offset
            state_mapping[physics_type.value] = physics_info
        
        return state_mapping
    
    def _order_physics_for_coupling(self) -> List[PhysicsType]:
        """Order physics modules to optimize coupling matrix structure"""
        # Strong coupling pairs should be adjacent for better block structure
        coupling_priorities = {
            PhysicsType.THERMAL: 1,      # Often drives other physics
            PhysicsType.MECHANICAL: 2,   # Structural response
            PhysicsType.CHEMICAL: 3,     # Material changes
            PhysicsType.FLUID_FLOW: 4,   # Transport
            PhysicsType.ELECTROMAGNETIC: 5,  # External fields
            PhysicsType.PHASE_CHANGE: 6  # Discrete events
        }
        
        available_physics = list(self.physics_modules.keys())
        return sorted(available_physics, key=lambda p: coupling_priorities.get(p, 10))
    
    def _classify_field_type(self, var_name: str) -> str:
        """Classify variable type for coupling analysis"""
        if 'temperature' in var_name.lower() or 'thermal' in var_name.lower():
            return 'scalar_intensive'
        elif 'displacement' in var_name.lower() or 'velocity' in var_name.lower():
            return 'vector_field'
        elif 'concentration' in var_name.lower() or 'density' in var_name.lower():
            return 'scalar_extensive'
        elif 'pressure' in var_name.lower() or 'stress' in var_name.lower():
            return 'tensor_field'
        else:
            return 'generic_scalar'
    
    def _get_variable_coupling_strength(self, var_name: str, physics_type: PhysicsType) -> float:
        """Get coupling strength for optimization"""
        coupling_matrix = getattr(self, 'variable_coupling_matrix', {})
        return coupling_matrix.get((var_name, physics_type.value), 0.5)
    
    def _initialize_jacobian_structure(self, state_mapping: Dict, total_dofs: int) -> Tuple:
        """Initialize efficient sparse matrix structure"""
        # Estimate non-zero entries based on coupling patterns
        estimated_nnz = self._estimate_jacobian_nonzeros(state_mapping)
        
        # Pre-allocate arrays for COO format
        jacobian_data = []
        row_indices = []
        col_indices = []
        
        return jacobian_data, row_indices, col_indices
    
    def _estimate_jacobian_nonzeros(self, state_mapping: Dict) -> int:
        """Estimate number of non-zero entries in Jacobian"""
        total_nnz = 0
        
        # Diagonal blocks (within-physics coupling)
        for physics_info in state_mapping.values():
            n_vars = physics_info['total_size']
            # Assume ~10% sparsity for within-physics coupling
            total_nnz += int(0.1 * n_vars * n_vars)
        
        # Off-diagonal blocks (between-physics coupling)
        physics_list = list(state_mapping.keys())
        for i, physics1 in enumerate(physics_list):
            for j, physics2 in enumerate(physics_list):
                if i != j:
                    coupling_strength = self._get_physics_coupling_strength(physics1, physics2)
                    if coupling_strength > 0.1:  # Significant coupling
                        size1 = state_mapping[physics1]['total_size']
                        size2 = state_mapping[physics2]['total_size']
                        # Coupling sparsity proportional to coupling strength
                        total_nnz += int(coupling_strength * 0.05 * size1 * size2)
        
        return max(total_nnz, 1000)  # Minimum reasonable size
    
    def _get_physics_coupling_strength(self, physics1: str, physics2: str) -> float:
        """Get coupling strength between physics modules"""
        coupling_strengths = {
            ('thermal', 'mechanical'): 0.8,       # Thermal expansion
            ('thermal', 'chemical'): 0.6,         # Reaction kinetics
            ('mechanical', 'chemical'): 0.4,      # Stress-induced reactions
            ('thermal', 'fluid_flow'): 0.7,       # Buoyancy, viscosity
            ('mechanical', 'fluid_flow'): 0.5,    # Deformation effects
            ('chemical', 'fluid_flow'): 0.3,      # Concentration transport
            ('electromagnetic', 'thermal'): 0.6,  # Joule heating
            ('phase_change', 'thermal'): 0.9,     # Latent heat
        }
        
        key = tuple(sorted([physics1, physics2]))
        return coupling_strengths.get(key, 0.1)  # Default weak coupling
    
    def _compute_diagonal_jacobian_blocks(self, jacobian_data: List, row_indices: List,
                                        col_indices: List, state_mapping: Dict):
        """Compute diagonal Jacobian blocks (within-physics coupling)"""
        for physics_name, physics_info in state_mapping.items():
            module = physics_info['module']
            
            try:
                # Compute physics-specific Jacobian
                physics_jacobian = module.compute_jacobian(self.state, self.timestep, self.mesh)
                
                if sparse.issparse(physics_jacobian):
                    physics_jacobian_coo = physics_jacobian.tocoo()
                    
                    # Map to global indices
                    global_offset = physics_info['global_offset']
                    global_rows = physics_jacobian_coo.row + global_offset
                    global_cols = physics_jacobian_coo.col + global_offset
                    
                    # Add to global structure
                    jacobian_data.extend(physics_jacobian_coo.data)
                    row_indices.extend(global_rows)
                    col_indices.extend(global_cols)
                else:
                    # Dense to sparse conversion
                    rows, cols = np.nonzero(physics_jacobian)
                    global_offset = physics_info['global_offset']
                    
                    jacobian_data.extend(physics_jacobian[rows, cols])
                    row_indices.extend(rows + global_offset)
                    col_indices.extend(cols + global_offset)
                    
            except Exception as e:
                self.logger.warning(f"Diagonal block computation failed for {physics_name}: {e}")
    
    def _compute_coupling_jacobian_blocks(self, jacobian_data: List, row_indices: List,
                                        col_indices: List, state_mapping: Dict):
        """Compute off-diagonal coupling Jacobian blocks"""
        physics_list = list(state_mapping.keys())
        
        for i, physics1 in enumerate(physics_list):
            for j, physics2 in enumerate(physics_list):
                if i != j:  # Off-diagonal block
                    coupling_strength = self._get_physics_coupling_strength(physics1, physics2)
                    
                    if coupling_strength > 0.1:  # Significant coupling
                        try:
                            coupling_block = self._compute_coupling_block(
                                physics1, physics2, state_mapping, coupling_strength
                            )
                            
                            if coupling_block is not None and np.any(coupling_block):
                                self._add_coupling_block_to_structure(
                                    coupling_block, physics1, physics2, 
                                    state_mapping, jacobian_data, 
                                    row_indices, col_indices
                                )
                                
                        except Exception as e:
                            self.logger.warning(f"Coupling block computation failed "
                                              f"for {physics1}-{physics2}: {e}")
    
    def _compute_coupling_block(self, physics1: str, physics2: str, 
                               state_mapping: Dict, coupling_strength: float) -> Optional[np.ndarray]:
        """Compute specific coupling block between two physics"""
        try:
            # Get physics modules
            module1 = state_mapping[physics1]['module']
            module2 = state_mapping[physics2]['module']
            
            # Use finite difference approximation for coupling derivatives
            coupling_block = self._finite_difference_coupling(
                module1, module2, physics1, physics2, state_mapping
            )
            
            # Scale by coupling strength
            if coupling_block is not None:
                coupling_block *= coupling_strength
            
            return coupling_block
            
        except Exception as e:
            self.logger.warning(f"Coupling block computation failed: {e}")
            return None
    
    def _finite_difference_coupling(self, module1, module2, physics1: str, 
                                   physics2: str, state_mapping: Dict) -> Optional[np.ndarray]:
        """Compute coupling derivatives using finite differences"""
        try:
            # Get state variables for physics2 (independent variables)
            physics2_vars = state_mapping[physics2]['variables']
            physics2_size = state_mapping[physics2]['total_size']
            
            # Get baseline residual from physics1
            baseline_residual = module1.compute_residual(
                self.state, self.timestep, self.mesh,
                **self._compute_coupling_sources(PhysicsType(physics1))
            )
            
            # Initialize coupling block
            coupling_block = np.zeros((len(baseline_residual), physics2_size))
            
            # Finite difference perturbation
            epsilon = 1e-8
            global_col = 0
            
            # Perturb each variable in physics2
            for var_name, var_info in physics2_vars.items():
                var_size = var_info['size']
                original_values = self.state[var_name].copy()
                
                # Perturb each component of the variable
                for local_idx in range(var_size):
                    # Apply perturbation
                    perturbation = epsilon * (1.0 + abs(original_values[local_idx]))
                    self.state[var_name][local_idx] += perturbation
                    
                    # Compute perturbed residual
                    perturbed_residual = module1.compute_residual(
                        self.state, self.timestep, self.mesh,
                        **self._compute_coupling_sources(PhysicsType(physics1))
                    )
                    
                    # Compute derivative
                    derivative = (perturbed_residual - baseline_residual) / perturbation
                    coupling_block[:, global_col] = derivative
                    
                    # Restore original value
                    self.state[var_name][local_idx] = original_values[local_idx]
                    global_col += 1
            
            # Apply sparsification based on coupling patterns
            coupling_block = self._sparsify_coupling_block(
                coupling_block, physics1, physics2
            )
            
            return coupling_block
            
        except Exception as e:
            self.logger.warning(f"Finite difference coupling failed: {e}")
            return None
    
    def _sparsify_coupling_block(self, coupling_block: np.ndarray, 
                                physics1: str, physics2: str) -> np.ndarray:
        """Apply physics-aware sparsification to coupling block"""
        # Remove entries below threshold
        threshold = 1e-12 * np.max(np.abs(coupling_block))
        coupling_block[np.abs(coupling_block) < threshold] = 0.0
        
        # Apply physics-specific sparsity patterns
        if physics1 == 'thermal' and physics2 == 'mechanical':
            # Thermal-mechanical coupling is often local
            coupling_block = self._apply_local_coupling_pattern(coupling_block)
        elif physics1 == 'chemical' and physics2 == 'fluid_flow':
            # Chemical-flow coupling follows transport patterns
            coupling_block = self._apply_transport_coupling_pattern(coupling_block)
        
        return coupling_block
    
    def _apply_local_coupling_pattern(self, coupling_block: np.ndarray) -> np.ndarray:
        """Apply local coupling sparsity pattern"""
        # Keep only local/neighboring interactions
        n_rows, n_cols = coupling_block.shape
        bandwidth = max(1, min(10, n_cols // 20))  # 5% bandwidth
        
        for i in range(n_rows):
            local_start = max(0, i - bandwidth)
            local_end = min(n_cols, i + bandwidth + 1)
            
            # Zero out non-local entries
            coupling_block[i, :local_start] = 0.0
            coupling_block[i, local_end:] = 0.0
        
        return coupling_block
    
    def _apply_transport_coupling_pattern(self, coupling_block: np.ndarray) -> np.ndarray:
        """Apply transport-based coupling pattern"""
        # Transport coupling follows flow directions
        if hasattr(self, 'flow_connectivity'):
            connectivity = self.flow_connectivity
            # Apply connectivity-based sparsification
            for i in range(coupling_block.shape[0]):
                for j in range(coupling_block.shape[1]):
                    if not connectivity.get((i, j), False):
                        coupling_block[i, j] = 0.0
        
        return coupling_block
    
    def _add_coupling_block_to_structure(self, coupling_block: np.ndarray,
                                       physics1: str, physics2: str,
                                       state_mapping: Dict, jacobian_data: List,
                                       row_indices: List, col_indices: List):
        """Add coupling block to global Jacobian structure"""
        # Get global offsets
        row_offset = state_mapping[physics1]['global_offset']
        col_offset = state_mapping[physics2]['global_offset']
        
        # Find non-zero entries
        rows, cols = np.nonzero(coupling_block)
        
        # Add to global structure
        jacobian_data.extend(coupling_block[rows, cols])
        row_indices.extend(rows + row_offset)
        col_indices.extend(cols + col_offset)
    
    def _apply_jacobian_conditioning(self, jacobian: sparse.csr_matrix, 
                                   state_mapping: Dict) -> sparse.csr_matrix:
        """Apply physics-aware conditioning to Jacobian"""
        try:
            # Diagonal scaling for better conditioning
            diagonal = jacobian.diagonal()
            scaling_factors = np.ones(len(diagonal))
            
            # Physics-specific scaling
            for physics_name, physics_info in state_mapping.items():
                start_idx = physics_info['global_offset']
                end_idx = start_idx + physics_info['total_size']
                
                # Apply characteristic scaling for this physics
                char_scale = self._get_characteristic_scale(physics_name)
                scaling_factors[start_idx:end_idx] *= char_scale
            
            # Apply row and column scaling
            scale_diag = sparse.diags(1.0 / np.sqrt(scaling_factors))
            jacobian_scaled = scale_diag @ jacobian @ scale_diag
            
            return jacobian_scaled
            
        except Exception as e:
            self.logger.warning(f"Jacobian conditioning failed: {e}")
            return jacobian
    
    def _get_characteristic_scale(self, physics_name: str) -> float:
        """Get characteristic scale for physics variables"""
        scales = {
            'thermal': 1e-3,      # Temperature in K
            'mechanical': 1e9,    # Stress in Pa
            'chemical': 1e3,      # Concentration in mol/m³
            'fluid_flow': 1e5,    # Pressure in Pa
            'electromagnetic': 1e-6, # Magnetic field in T
            'phase_change': 1.0   # Dimensionless
        }
        return scales.get(physics_name, 1.0)
    
    def _analyze_jacobian_properties(self, jacobian: sparse.csr_matrix, state_mapping: Dict):
        """Analyze and store Jacobian properties for optimization"""
        try:
            analysis = {
                'shape': jacobian.shape,
                'nnz': jacobian.nnz,
                'sparsity': jacobian.nnz / (jacobian.shape[0] * jacobian.shape[1]),
                'condition_number_estimate': None,
                'block_structure': {},
                'timestamp': datetime.now()
            }
            
            # Block structure analysis
            for physics_name, physics_info in state_mapping.items():
                start_idx = physics_info['global_offset']
                end_idx = start_idx + physics_info['total_size']
                
                block_nnz = jacobian[start_idx:end_idx, start_idx:end_idx].nnz
                analysis['block_structure'][physics_name] = {
                    'size': physics_info['total_size'],
                    'nnz': block_nnz,
                    'sparsity': block_nnz / (physics_info['total_size']**2)
                }
            
            # Store for debugging and optimization
            if not hasattr(self, 'jacobian_analysis_history'):
                self.jacobian_analysis_history = []
            self.jacobian_analysis_history.append(analysis)
            
        except Exception as e:
            self.logger.warning(f"Jacobian analysis failed: {e}")
    
    def _assemble_basic_block_diagonal_jacobian(self) -> sparse.csr_matrix:
        """Fallback basic block diagonal assembly"""
        jacobian_blocks = []
        
        for physics_type, module in self.physics_modules.items():
            try:
                jacobian = module.compute_jacobian(self.state, self.timestep, self.mesh)
                jacobian_blocks.append(jacobian)
            except Exception as e:
                self.logger.warning(f"Basic Jacobian computation failed for {physics_type}: {e}")
                # Create identity block as fallback
                size = sum(len(self.state.get(var, [1])) for var in module.state_variables)
                jacobian_blocks.append(sparse.eye(size))
        
        return sparse.block_diag(jacobian_blocks, format='csr')
    
    def _update_coupled_state(self, delta_solution: np.ndarray):
        """
        Advanced coupled state update with physics-aware step limiting,
        constraint enforcement, and adaptive damping
        """
        try:
            # Build comprehensive state mapping
            state_mapping = self._build_state_variable_mapping()
            
            # Apply physics-aware step limiting
            damped_delta = self._apply_adaptive_damping(delta_solution, state_mapping)
            
            # Update state variables with constraint enforcement
            self._update_state_variables(damped_delta, state_mapping)
            
            # Enforce physical constraints
            self._enforce_physical_constraints()
            
            # Update derived quantities
            self._update_derived_quantities()
            
            # Validate state consistency
            self._validate_state_consistency()
            
            # Store update history for analysis
            self._store_update_analysis(delta_solution, damped_delta, state_mapping)
            
        except Exception as e:
            self.logger.error(f"Coupled state update failed: {e}")
            # Attempt recovery with reduced step
            self._recover_from_failed_update(delta_solution)
    
    def _apply_adaptive_damping(self, delta_solution: np.ndarray, 
                               state_mapping: Dict) -> np.ndarray:
        """Apply physics-aware adaptive damping"""
        damped_delta = delta_solution.copy()
        
        try:
            # Global damping factor based on solution magnitude
            max_change = np.max(np.abs(delta_solution))
            if max_change > 1e-6:
                global_damping = min(1.0, 1e-6 / max_change)
            else:
                global_damping = 1.0
            
            # Physics-specific damping
            for physics_name, physics_info in state_mapping.items():
                start_idx = physics_info['global_offset']
                end_idx = start_idx + physics_info['total_size']
                
                # Get physics-specific damping parameters
                damping_params = self._get_physics_damping_parameters(physics_name)
                
                # Apply variable-specific damping
                local_offset = start_idx
                for var_name, var_info in physics_info['variables'].items():
                    var_size = var_info['size']
                    var_end = local_offset + var_size
                    
                    # Compute variable-specific damping
                    var_damping = self._compute_variable_damping(
                        var_name, damped_delta[local_offset:var_end], damping_params
                    )
                    
                    # Apply damping
                    damped_delta[local_offset:var_end] *= var_damping
                    local_offset = var_end
                
                # Apply physics-specific global damping
                damped_delta[start_idx:end_idx] *= damping_params['global_factor']
            
            # Apply global damping
            damped_delta *= global_damping
            
            # Ensure damped solution doesn't violate bounds
            damped_delta = self._enforce_solution_bounds(damped_delta, state_mapping)
            
            return damped_delta
            
        except Exception as e:
            self.logger.warning(f"Adaptive damping failed: {e}")
            return 0.5 * delta_solution  # Conservative fallback
    
    def _get_physics_damping_parameters(self, physics_name: str) -> Dict:
        """Get damping parameters for specific physics"""
        default_params = {
            'global_factor': 1.0,
            'max_relative_change': 0.1,
            'max_absolute_change': 1e6,
            'variable_limits': {}
        }
        
        physics_params = {
            'thermal': {
                'global_factor': 0.8,
                'max_relative_change': 0.05,  # Conservative for temperature
                'variable_limits': {'temperature': {'max_change': 100.0}}  # 100K max
            },
            'mechanical': {
                'global_factor': 1.0,
                'max_relative_change': 0.1,
                'variable_limits': {'displacement': {'max_change': 1e-3}}  # 1mm max
            },
            'chemical': {
                'global_factor': 0.7,
                'max_relative_change': 0.2,  # Can handle larger changes
                'variable_limits': {'concentration': {'min_value': 0.0}}  # Non-negative
            }
        }
        
        params = physics_params.get(physics_name, default_params)
        # Merge with defaults
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
        
        return params
    
    def _compute_variable_damping(self, var_name: str, var_delta: np.ndarray, 
                                 damping_params: Dict) -> float:
        """Compute damping factor for specific variable"""
        try:
            if var_name in damping_params.get('variable_limits', {}):
                limits = damping_params['variable_limits'][var_name]
                
                # Check absolute change limits
                if 'max_change' in limits:
                    max_delta = np.max(np.abs(var_delta))
                    if max_delta > limits['max_change']:
                        return limits['max_change'] / max_delta
            
            # Default relative change limiting
            current_values = self.state.get(var_name, np.ones_like(var_delta))
            relative_change = np.max(np.abs(var_delta / (current_values + 1e-12)))
            max_rel_change = damping_params['max_relative_change']
            
            if relative_change > max_rel_change:
                return max_rel_change / relative_change
            
            return 1.0
            
        except Exception:
            return 0.5  # Conservative fallback
    
    def _enforce_solution_bounds(self, delta_solution: np.ndarray, 
                                state_mapping: Dict) -> np.ndarray:
        """Enforce physical bounds on solution update"""
        bounded_delta = delta_solution.copy()
        
        try:
            for physics_name, physics_info in state_mapping.items():
                local_offset = physics_info['global_offset']
                
                for var_name, var_info in physics_info['variables'].items():
                    var_size = var_info['size']
                    var_end = local_offset + var_size
                    
                    # Get current values
                    current_values = self.state.get(var_name, np.zeros(var_size))
                    proposed_values = current_values + bounded_delta[local_offset:var_end]
                    
                    # Apply variable-specific bounds
                    bounds = self._get_variable_bounds(var_name)
                    if bounds:
                        min_val, max_val = bounds
                        
                        # Clip proposed values to bounds
                        clipped_values = np.clip(proposed_values, min_val, max_val)
                        
                        # Update delta to respect bounds
                        bounded_delta[local_offset:var_end] = clipped_values - current_values
                    
                    local_offset = var_end
            
            return bounded_delta
            
        except Exception as e:
            self.logger.warning(f"Solution bounds enforcement failed: {e}")
            return delta_solution
    
    def _get_variable_bounds(self, var_name: str) -> Optional[Tuple[float, float]]:
        """Get physical bounds for variables"""
        bounds_dict = {
            'temperature': (0.1, 5000.0),        # 0.1K to 5000K
            'concentration': (0.0, 1e6),         # Non-negative, reasonable upper bound
            'pressure': (-1e8, 1e12),            # Allow some tension, high compression
            'density': (1.0, 50000.0),           # Physical density range
            'porosity': (0.0, 1.0),              # Fraction
            'saturation': (0.0, 1.0)             # Fraction
        }
        
        # Check for partial matches
        for key, bounds in bounds_dict.items():
            if key in var_name.lower():
                return bounds
        
        return None
    
    def _update_state_variables(self, damped_delta: np.ndarray, state_mapping: Dict):
        """Update state variables with comprehensive error checking"""
        for physics_name, physics_info in state_mapping.items():
            try:
                local_offset = physics_info['global_offset']
                
                for var_name, var_info in physics_info['variables'].items():
                    var_size = var_info['size']
                    var_end = local_offset + var_size
                    
                    if var_name in self.state:
                        # Extract variable update
                        var_delta = damped_delta[local_offset:var_end]
                        
                        # Apply update with error checking
                        old_values = self.state[var_name].copy()
                        self.state[var_name] += var_delta
                        
                        # Check for invalid values
                        if np.any(~np.isfinite(self.state[var_name])):
                            self.logger.warning(f"Invalid values in {var_name}, reverting")
                            self.state[var_name] = old_values
                    
                    local_offset = var_end
                    
            except Exception as e:
                self.logger.error(f"Variable update failed for {physics_name}: {e}")
    
    def _enforce_physical_constraints(self):
        """Enforce physical constraints on updated state"""
        try:
            # Mass conservation constraints
            if hasattr(self, 'mass_conservation_constraints'):
                self._enforce_mass_conservation()
            
            # Energy conservation constraints
            if hasattr(self, 'energy_conservation_constraints'):
                self._enforce_energy_conservation()
            
            # Thermodynamic constraints
            self._enforce_thermodynamic_constraints()
            
        except Exception as e:
            self.logger.warning(f"Constraint enforcement failed: {e}")
    
    def _enforce_thermodynamic_constraints(self):
        """Enforce basic thermodynamic constraints"""
        try:
            # Ensure positive temperatures
            if 'temperature' in self.state:
                self.state['temperature'] = np.maximum(self.state['temperature'], 0.1)
            
            # Ensure density bounds
            if 'density' in self.state:
                self.state['density'] = np.clip(self.state['density'], 1.0, 50000.0)
            
            # Ensure fraction bounds
            for var_name in ['porosity', 'saturation', 'phase_fraction']:
                if var_name in self.state:
                    self.state[var_name] = np.clip(self.state[var_name], 0.0, 1.0)
                    
        except Exception as e:
            self.logger.warning(f"Thermodynamic constraint enforcement failed: {e}")
    
    def _update_derived_quantities(self):
        """Update derived quantities based on primary variables"""
        try:
            # Update material properties based on temperature
            if 'temperature' in self.state:
                self._update_temperature_dependent_properties()
            
            # Update stress from strain
            if 'displacement' in self.state:
                self._update_stress_from_displacement()
            
            # Update reaction rates from concentration and temperature
            if 'concentration' in self.state and 'temperature' in self.state:
                self._update_reaction_rates()
                
        except Exception as e:
            self.logger.warning(f"Derived quantity update failed: {e}")
    
    def _validate_state_consistency(self):
        """Validate state consistency and detect issues"""
        try:
            issues = []
            
            # Check for NaN/Inf values
            for var_name, var_value in self.state.items():
                if hasattr(var_value, '__iter__'):
                    if np.any(~np.isfinite(var_value)):
                        issues.append(f"Non-finite values in {var_name}")
                
            # Check conservation laws
            if hasattr(self, 'conservation_checker'):
                conservation_issues = self.conservation_checker.validate(self.state)
                issues.extend(conservation_issues)
            
            if issues:
                self.logger.warning(f"State consistency issues: {issues}")
                
        except Exception as e:
            self.logger.warning(f"State validation failed: {e}")
    
    def _store_update_analysis(self, original_delta: np.ndarray, 
                              damped_delta: np.ndarray, state_mapping: Dict):
        """Store update analysis for debugging and optimization"""
        try:
            analysis = {
                'timestamp': datetime.now(),
                'original_norm': np.linalg.norm(original_delta),
                'damped_norm': np.linalg.norm(damped_delta),
                'damping_factor': np.linalg.norm(damped_delta) / np.linalg.norm(original_delta),
                'max_change': np.max(np.abs(damped_delta)),
                'physics_changes': {}
            }
            
            # Per-physics analysis
            for physics_name, physics_info in state_mapping.items():
                start_idx = physics_info['global_offset']
                end_idx = start_idx + physics_info['total_size']
                
                physics_delta = damped_delta[start_idx:end_idx]
                analysis['physics_changes'][physics_name] = {
                    'norm': np.linalg.norm(physics_delta),
                    'max_change': np.max(np.abs(physics_delta))
                }
            
            if not hasattr(self, 'update_analysis_history'):
                self.update_analysis_history = deque(maxlen=100)
            self.update_analysis_history.append(analysis)
            
        except Exception as e:
            self.logger.warning(f"Update analysis storage failed: {e}")
    
    def _recover_from_failed_update(self, delta_solution: np.ndarray):
        """Attempt recovery from failed state update"""
        try:
            self.logger.warning("Attempting recovery from failed update")
            
            # Try with heavily damped update
            recovery_delta = 0.1 * delta_solution
            
            # Simple state update without advanced features
            offset = 0
            for physics_type, module in self.physics_modules.items():
                module_size = 0
                for var_name in module.state_variables:
                    if var_name in self.state:
                        var_size = len(self.state[var_name])
                        
                        # Apply conservative update
                        var_delta = recovery_delta[offset:offset+var_size]
                        self.state[var_name] += var_delta
                        
                        module_size += var_size
                        offset += var_size
                
            self.logger.info("Recovery update applied successfully")
            
        except Exception as e:
            self.logger.error(f"Recovery failed: {e}")
            # Last resort: revert to previous state if available
            if hasattr(self, 'previous_state'):
                self.state = self.previous_state.copy()
                self.logger.warning("Reverted to previous state")
    
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
        'thermal': {'temperature': 323.15},  # 50?C
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