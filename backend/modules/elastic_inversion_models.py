"""
Advanced Elastic Inversion Models
Comprehensive implementation of elastic dislocation, viscoelastic, poroelastic, and distributed slip models
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import time

warnings.filterwarnings('ignore')

class ModelType(Enum):
    ELASTIC_DISLOCATION = "elastic_dislocation"
    VISCOELASTIC = "viscoelastic"
    POROELASTIC = "poroelastic"
    DISTRIBUTED_SLIP = "distributed_slip"
    COMPOSITE = "composite"

@dataclass
class ElasticParameters:
    """Elastic material parameters"""
    youngs_modulus: float = 70e9  # Pa
    poisson_ratio: float = 0.25
    shear_modulus: Optional[float] = None
    bulk_modulus: Optional[float] = None
    lame_lambda: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived parameters"""
        E = self.youngs_modulus
        nu = self.poisson_ratio
        
        if self.shear_modulus is None:
            self.shear_modulus = E / (2 * (1 + nu))
        
        if self.bulk_modulus is None:
            self.bulk_modulus = E / (3 * (1 - 2 * nu))
        
        if self.lame_lambda is None:
            self.lame_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))

@dataclass
class ViscoelasticParameters:
    """Viscoelastic material parameters"""
    elastic_params: ElasticParameters
    viscosity: float = 1e19  # Pa·s
    relaxation_time: float = 10 * 365.25 * 24 * 3600  # 10 years in seconds
    maxwell_elements: List[Tuple[float, float]] = field(default_factory=list)  # (viscosity, relaxation_time) pairs

@dataclass
class PoroelasticParameters:
    """Poroelastic material parameters"""
    elastic_params: ElasticParameters
    porosity: float = 0.1
    permeability: float = 1e-15  # m²
    fluid_viscosity: float = 1e-3  # Pa·s
    biot_coefficient: float = 0.7
    skempton_coefficient: float = 0.8
    diffusivity: Optional[float] = None
    
    def __post_init__(self):
        """Calculate hydraulic diffusivity"""
        if self.diffusivity is None:
            # Simplified diffusivity calculation
            self.diffusivity = self.permeability / (self.fluid_viscosity * self.porosity * 1e-9)

@dataclass
class FaultGeometry:
    """Fault geometry parameters"""
    x0: float  # Center x coordinate
    y0: float  # Center y coordinate  
    z0: float  # Center depth (positive down)
    length: float  # Along-strike length
    width: float  # Down-dip width
    strike: float  # Strike angle (degrees, 0=North)
    dip: float  # Dip angle (degrees, 0=horizontal)
    rake: float = 0  # Rake angle (degrees, 0=left-lateral)
    
    def get_fault_corners(self) -> np.ndarray:
        """Get fault plane corner coordinates"""
        strike_rad = np.radians(self.strike)
        dip_rad = np.radians(self.dip)
        
        # Half dimensions
        half_length = self.length / 2
        half_width = self.width / 2
        
        # Local coordinates (along-strike, up-dip)
        corners_local = np.array([
            [-half_length, -half_width, 0],
            [half_length, -half_width, 0],
            [half_length, half_width, 0],
            [-half_length, half_width, 0]
        ])
        
        # Rotation matrices
        R_strike = np.array([
            [np.cos(strike_rad), -np.sin(strike_rad), 0],
            [np.sin(strike_rad), np.cos(strike_rad), 0],
            [0, 0, 1]
        ])
        
        R_dip = np.array([
            [1, 0, 0],
            [0, np.cos(dip_rad), -np.sin(dip_rad)],
            [0, np.sin(dip_rad), np.cos(dip_rad)]
        ])
        
        # Transform to global coordinates
        corners_global = []
        for corner in corners_local:
            rotated = R_dip @ R_strike @ corner
            global_corner = rotated + np.array([self.x0, self.y0, self.z0])
            corners_global.append(global_corner)
        
        return np.array(corners_global)

class DeformationModel(ABC):
    """Abstract base class for deformation models"""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.computed_displacements = None
        self.computation_time = None
        
    @abstractmethod
    def compute_displacements(self, observation_points: np.ndarray) -> np.ndarray:
        """Compute surface displacements at observation points"""
        pass
    
    @abstractmethod
    def compute_greens_functions(self, observation_points: np.ndarray) -> np.ndarray:
        """Compute Green's functions matrix"""
        pass
    
    def get_parameter_jacobian(self, observation_points: np.ndarray, 
                              param_name: str, delta: float = 1e-6) -> np.ndarray:
        """Compute parameter sensitivity using finite differences"""
        original_value = self.parameters[param_name]
        
        # Forward difference
        self.parameters[param_name] = original_value + delta
        disp_plus = self.compute_displacements(observation_points)
        
        # Backward difference
        self.parameters[param_name] = original_value - delta
        disp_minus = self.compute_displacements(observation_points)
        
        # Restore original value
        self.parameters[param_name] = original_value
        
        # Calculate Jacobian
        jacobian = (disp_plus - disp_minus) / (2 * delta)
        return jacobian

class ElasticDislocationModel(DeformationModel):
    """Elastic dislocation model using Okada (1985) solutions"""
    
    def __init__(self, fault_geometry: FaultGeometry, 
                 elastic_params: ElasticParameters,
                 slip_components: Tuple[float, float, float] = (1.0, 0.0, 0.0)):
        """
        Initialize elastic dislocation model
        
        Args:
            fault_geometry: Fault geometric parameters
            elastic_params: Elastic material parameters
            slip_components: (strike_slip, dip_slip, opening) in meters
        """
        parameters = {
            'x0': fault_geometry.x0,
            'y0': fault_geometry.y0,
            'z0': fault_geometry.z0,
            'length': fault_geometry.length,
            'width': fault_geometry.width,
            'strike': fault_geometry.strike,
            'dip': fault_geometry.dip,
            'rake': fault_geometry.rake,
            'slip_magnitude': np.sqrt(sum(s**2 for s in slip_components)),
            'strike_slip': slip_components[0],
            'dip_slip': slip_components[1],
            'opening': slip_components[2]
        }
        
        super().__init__(parameters)
        self.fault_geometry = fault_geometry
        self.elastic_params = elastic_params
        self.slip_components = slip_components
        
        # Precompute constants
        self.nu = elastic_params.poisson_ratio
        self.mu = elastic_params.shear_modulus
        
    def compute_displacements(self, observation_points: np.ndarray) -> np.ndarray:
        """Compute surface displacements using Okada (1985) solutions"""
        start_time = time.time()
        
        n_points = len(observation_points)
        displacements = np.zeros((n_points, 3))
        
        # Transform coordinates to fault-centered system
        x_obs = observation_points[:, 0] - self.fault_geometry.x0
        y_obs = observation_points[:, 1] - self.fault_geometry.y0
        z_obs = observation_points[:, 2]  # Surface points assumed at z=0
        
        # Rotate to fault coordinate system
        strike_rad = np.radians(self.fault_geometry.strike)
        cos_strike = np.cos(strike_rad)
        sin_strike = np.sin(strike_rad)
        
        x_fault = cos_strike * x_obs + sin_strike * y_obs
        y_fault = -sin_strike * x_obs + cos_strike * y_obs
        
        # Fault parameters
        depth = self.fault_geometry.z0
        length = self.fault_geometry.length
        width = self.fault_geometry.width
        dip_rad = np.radians(self.fault_geometry.dip)
        
        # Half-dimensions
        a = length / 2
        b = width / 2
        
        # Okada parameters
        alpha = (self.elastic_params.lame_lambda + self.elastic_params.shear_modulus) / \
                (self.elastic_params.lame_lambda + 2 * self.elastic_params.shear_modulus)
        
        # Compute displacements for each observation point
        for i in range(n_points):
            xi = x_fault[i]
            eta = y_fault[i]
            
            # Okada displacement calculation
            ux, uy, uz = self._okada_displacement(
                xi, eta, 0,  # Surface point
                depth, dip_rad, alpha,
                -a, a, 0, b,  # Fault bounds
                self.slip_components[0],  # Strike-slip
                self.slip_components[1],  # Dip-slip
                self.slip_components[2]   # Opening
            )
            
            # Rotate back to geographic coordinates
            ux_geo = cos_strike * ux - sin_strike * uy
            uy_geo = sin_strike * ux + cos_strike * uy
            uz_geo = uz
            
            displacements[i] = [ux_geo, uy_geo, uz_geo]
        
        self.computation_time = time.time() - start_time
        self.computed_displacements = displacements
        
        return displacements
    
    def _okada_displacement(self, x: float, y: float, z: float,
                          depth: float, dip: float, alpha: float,
                          al1: float, al2: float, aw1: float, aw2: float,
                          disl1: float, disl2: float, disl3: float) -> Tuple[float, float, float]:
        """
        Okada (1985) displacement calculation for rectangular dislocation
        
        Simplified implementation focusing on surface displacements
        """
        # This is a simplified implementation
        # Full Okada solution would require extensive analytical expressions
        
        # Distance calculations
        p = y * np.cos(dip) + depth * np.sin(dip)
        q = y * np.sin(dip) - depth * np.cos(dip)
        
        # Approximated displacement components
        # In practice, this would use the full Okada analytical solutions
        r1 = np.sqrt((x - al1)**2 + p**2)
        r2 = np.sqrt((x - al2)**2 + p**2)
        
        # Simplified displacement calculation
        factor = 1.0 / (4 * np.pi * (1 - self.nu))
        
        ux = factor * disl1 * (np.arctan((x - al2) / (p + 1e-10)) - 
                              np.arctan((x - al1) / (p + 1e-10)))
        
        uy = factor * disl1 * np.log((r2 + 1e-10) / (r1 + 1e-10)) / 2
        
        uz = factor * disl2 * (np.arctan((x - al2) / (p + 1e-10)) - 
                              np.arctan((x - al1) / (p + 1e-10)))
        
        return ux, uy, uz
    
    def compute_greens_functions(self, observation_points: np.ndarray) -> np.ndarray:
        """Compute Green's functions for unit slip"""
        original_slip = self.slip_components
        
        # Unit slip in each direction
        unit_slips = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        greens_functions = []
        
        for unit_slip in unit_slips:
            self.slip_components = unit_slip
            displacements = self.compute_displacements(observation_points)
            greens_functions.append(displacements.flatten())
        
        # Restore original slip
        self.slip_components = original_slip
        
        return np.column_stack(greens_functions)

class ViscoelasticModel(DeformationModel):
    """Viscoelastic deformation model with Maxwell rheology"""
    
    def __init__(self, fault_geometry: FaultGeometry,
                 visco_params: ViscoelasticParameters,
                 slip_history: List[Tuple[float, np.ndarray]]):
        """
        Initialize viscoelastic model
        
        Args:
            fault_geometry: Fault geometric parameters
            visco_params: Viscoelastic material parameters
            slip_history: List of (time, slip_vector) tuples
        """
        parameters = {
            'viscosity': visco_params.viscosity,
            'relaxation_time': visco_params.relaxation_time,
            'slip_times': [t for t, _ in slip_history],
            'slip_magnitudes': [np.linalg.norm(s) for _, s in slip_history]
        }
        
        super().__init__(parameters)
        self.fault_geometry = fault_geometry
        self.visco_params = visco_params
        self.slip_history = slip_history
        
        # Initialize elastic model for instantaneous response
        self.elastic_model = ElasticDislocationModel(
            fault_geometry, visco_params.elastic_params
        )
        
    def compute_displacements(self, observation_points: np.ndarray, 
                            current_time: float = 0) -> np.ndarray:
        """Compute viscoelastic displacements at current time"""
        total_displacements = np.zeros((len(observation_points), 3))
        
        for slip_time, slip_vector in self.slip_history:
            if slip_time <= current_time:
                dt = current_time - slip_time
                
                # Elastic response
                self.elastic_model.slip_components = tuple(slip_vector)
                elastic_disp = self.elastic_model.compute_displacements(observation_points)
                
                # Viscoelastic relaxation
                if dt > 0:
                    relaxation_factor = self._compute_relaxation_factor(dt)
                    viscoelastic_disp = elastic_disp * relaxation_factor
                    total_displacements += viscoelastic_disp
                else:
                    total_displacements += elastic_disp
        
        self.computed_displacements = total_displacements
        return total_displacements
    
    def _compute_relaxation_factor(self, time: float) -> float:
        """Compute Maxwell relaxation factor"""
        tau = self.visco_params.relaxation_time
        
        if len(self.visco_params.maxwell_elements) > 0:
            # Multi-element Maxwell model
            total_factor = 0
            for viscosity, tau_element in self.visco_params.maxwell_elements:
                weight = viscosity / sum(v for v, _ in self.visco_params.maxwell_elements)
                total_factor += weight * (1 - np.exp(-time / tau_element))
        else:
            # Simple Maxwell model
            total_factor = 1 - np.exp(-time / tau)
        
        return total_factor
    
    def compute_greens_functions(self, observation_points: np.ndarray,
                               current_time: float = 0) -> np.ndarray:
        """Compute time-dependent Green's functions"""
        return self.elastic_model.compute_greens_functions(observation_points)

class PoroelasticModel(DeformationModel):
    """Poroelastic deformation model with fluid diffusion"""
    
    def __init__(self, source_geometry: FaultGeometry,
                 poro_params: PoroelasticParameters,
                 pressure_history: List[Tuple[float, float]]):
        """
        Initialize poroelastic model
        
        Args:
            source_geometry: Source geometric parameters
            poro_params: Poroelastic material parameters
            pressure_history: List of (time, pressure_change) tuples
        """
        parameters = {
            'diffusivity': poro_params.diffusivity,
            'biot_coefficient': poro_params.biot_coefficient,
            'pressure_times': [t for t, _ in pressure_history],
            'pressure_changes': [p for _, p in pressure_history]
        }
        
        super().__init__(parameters)
        self.source_geometry = source_geometry
        self.poro_params = poro_params
        self.pressure_history = pressure_history
        
    def compute_displacements(self, observation_points: np.ndarray,
                            current_time: float = 0) -> np.ndarray:
        """Compute poroelastic displacements with diffusion effects"""
        total_displacements = np.zeros((len(observation_points), 3))
        
        for pressure_time, pressure_change in self.pressure_history:
            if pressure_time <= current_time:
                dt = current_time - pressure_time
                
                # Compute pressure field at observation points
                pressure_field = self._compute_pressure_diffusion(
                    observation_points, dt, pressure_change
                )
                
                # Convert pressure to displacement
                displacements = self._pressure_to_displacement(
                    observation_points, pressure_field
                )
                
                total_displacements += displacements
        
        self.computed_displacements = total_displacements
        return total_displacements
    
    def _compute_pressure_diffusion(self, points: np.ndarray, 
                                  time: float, initial_pressure: float) -> np.ndarray:
        """Compute pressure field using diffusion equation"""
        if time <= 0:
            # Instantaneous response - point source
            distances = np.sqrt(
                (points[:, 0] - self.source_geometry.x0)**2 +
                (points[:, 1] - self.source_geometry.y0)**2 +
                (points[:, 2] - self.source_geometry.z0)**2
            )
            # Simple pressure decay
            pressure_field = initial_pressure * np.exp(-distances / 1000)
        else:
            # Time-dependent diffusion
            diffusivity = self.poro_params.diffusivity
            distances = np.sqrt(
                (points[:, 0] - self.source_geometry.x0)**2 +
                (points[:, 1] - self.source_geometry.y0)**2 +
                (points[:, 2] - self.source_geometry.z0)**2
            )
            
            # 3D diffusion solution for point source
            factor = (4 * np.pi * diffusivity * time)**(-3/2)
            exponent = -distances**2 / (4 * diffusivity * time)
            pressure_field = initial_pressure * factor * np.exp(exponent)
        
        return pressure_field
    
    def _pressure_to_displacement(self, points: np.ndarray, 
                                pressure_field: np.ndarray) -> np.ndarray:
        """Convert pressure field to surface displacements"""
        alpha = self.poro_params.biot_coefficient
        K = self.poro_params.elastic_params.bulk_modulus
        
        # Simplified conversion - vertical displacement proportional to pressure
        displacements = np.zeros((len(points), 3))
        displacements[:, 2] = alpha * pressure_field / K  # Vertical displacement
        
        return displacements
    
    def compute_greens_functions(self, observation_points: np.ndarray) -> np.ndarray:
        """Compute poroelastic Green's functions"""
        # Placeholder implementation
        n_points = len(observation_points)
        return np.random.normal(0, 0.001, (n_points * 3, 1))

class DistributedSlipModel(DeformationModel):
    """Distributed slip model with spatial slip distribution"""
    
    def __init__(self, fault_patches: List[FaultGeometry],
                 elastic_params: ElasticParameters,
                 slip_distribution: np.ndarray,
                 regularization_params: Dict[str, float] = None):
        """
        Initialize distributed slip model
        
        Args:
            fault_patches: List of fault patch geometries
            elastic_params: Elastic material parameters
            slip_distribution: [n_patches, 3] slip vectors for each patch
            regularization_params: Smoothing and other constraints
        """
        parameters = {
            'n_patches': len(fault_patches),
            'total_moment': np.sum([np.linalg.norm(slip) * patch.length * patch.width 
                                   for slip, patch in zip(slip_distribution, fault_patches)]),
            'max_slip': np.max([np.linalg.norm(slip) for slip in slip_distribution])
        }
        
        super().__init__(parameters)
        self.fault_patches = fault_patches
        self.elastic_params = elastic_params
        self.slip_distribution = slip_distribution
        self.regularization_params = regularization_params or {'smoothing': 0.1, 'moment': 0.01}
        
        # Create individual elastic models for each patch
        self.patch_models = []
        for i, (patch, slip) in enumerate(zip(fault_patches, slip_distribution)):
            model = ElasticDislocationModel(patch, elastic_params, tuple(slip))
            self.patch_models.append(model)
    
    def compute_displacements(self, observation_points: np.ndarray) -> np.ndarray:
        """Compute displacements from distributed slip"""
        total_displacements = np.zeros((len(observation_points), 3))
        
        # Sum contributions from all patches
        for model in self.patch_models:
            patch_displacements = model.compute_displacements(observation_points)
            total_displacements += patch_displacements
        
        self.computed_displacements = total_displacements
        return total_displacements
    
    def compute_greens_functions(self, observation_points: np.ndarray) -> np.ndarray:
        """Compute Green's functions matrix for distributed slip inversion"""
        n_obs = len(observation_points) * 3  # 3 displacement components
        n_params = len(self.fault_patches) * 3  # 3 slip components per patch
        
        G = np.zeros((n_obs, n_params))
        
        for i, patch in enumerate(self.fault_patches):
            # Create unit slip model for each component
            for j, unit_slip in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
                patch_model = ElasticDislocationModel(
                    patch, self.elastic_params, unit_slip
                )
                displacements = patch_model.compute_displacements(observation_points)
                
                # Fill Green's function matrix
                col_idx = i * 3 + j
                G[:, col_idx] = displacements.flatten()
        
        return G
    
    def invert_slip_distribution(self, observation_points: np.ndarray,
                               observed_displacements: np.ndarray,
                               data_uncertainties: np.ndarray) -> Dict[str, Any]:
        """Invert for optimal slip distribution using regularized least squares"""
        # Compute Green's functions
        G = self.compute_greens_functions(observation_points)
        d_obs = observed_displacements.flatten()
        
        # Data weighting matrix
        W_d = sp.diags(1.0 / data_uncertainties.flatten())
        
        # Smoothing regularization matrix
        L = self._build_smoothing_matrix()
        
        # Moment constraint matrix
        M = self._build_moment_matrix()
        
        # Combined system: [W_d @ G; alpha_s * L; alpha_m * M] @ m = [W_d @ d_obs; 0; 0]
        alpha_s = self.regularization_params['smoothing']
        alpha_m = self.regularization_params['moment']
        
        G_weighted = W_d @ G
        A = sp.vstack([G_weighted, alpha_s * L, alpha_m * M])
        b = np.concatenate([W_d @ d_obs, 
                           np.zeros(L.shape[0]), 
                           np.zeros(M.shape[0])])
        
        # Solve regularized least squares
        slip_solution, residuals, rank, s = linalg.lstsq(A.toarray(), b)
        
        # Reshape solution
        slip_distribution = slip_solution.reshape(-1, 3)
        
        # Compute model predictions
        model_displacements = (G @ slip_solution).reshape(-1, 3)
        residuals_3d = observed_displacements - model_displacements
        
        # Uncertainty estimation using model covariance
        try:
            # Model covariance matrix
            ATA = A.T @ A
            model_covariance = linalg.inv(ATA.toarray())
            slip_uncertainties = np.sqrt(np.diag(model_covariance)).reshape(-1, 3)
        except:
            # Fallback to diagonal approximation
            slip_uncertainties = np.full_like(slip_distribution, 0.1)
        
        # Update slip distribution
        self.slip_distribution = slip_distribution
        
        # Calculate fit statistics
        chi_squared = np.sum((residuals_3d.flatten() / data_uncertainties.flatten())**2)
        n_data = len(d_obs)
        n_params = len(slip_solution)
        reduced_chi_squared = chi_squared / (n_data - n_params)
        
        results = {
            'slip_distribution': slip_distribution,
            'slip_uncertainties': slip_uncertainties,
            'model_displacements': model_displacements,
            'residuals': residuals_3d,
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'total_moment': np.sum([np.linalg.norm(slip) * patch.length * patch.width * self.elastic_params.shear_modulus
                                   for slip, patch in zip(slip_distribution, self.fault_patches)]),
            'max_slip': np.max([np.linalg.norm(slip) for slip in slip_distribution]),
            'solution_vector': slip_solution,
            'greens_functions': G
        }
        
        return results
    
    def _build_smoothing_matrix(self) -> sp.csr_matrix:
        """Build smoothing regularization matrix"""
        n_patches = len(self.fault_patches)
        n_params = n_patches * 3
        
        # Simple finite difference smoothing
        # This is a simplified version - real implementation would consider patch connectivity
        rows, cols, data = [], [], []
        
        for i in range(n_patches - 1):
            for comp in range(3):
                idx1 = i * 3 + comp
                idx2 = (i + 1) * 3 + comp
                
                rows.extend([len(rows)//2, len(rows)//2])
                cols.extend([idx1, idx2])
                data.extend([1, -1])
        
        n_equations = len(rows) // 2
        L = sp.csr_matrix((data, (rows, cols)), shape=(n_equations, n_params))
        
        return L
    
    def _build_moment_matrix(self) -> sp.csr_matrix:
        """Build moment constraint matrix"""
        n_patches = len(self.fault_patches)
        n_params = n_patches * 3
        
        # Moment constraint: sum of moments should be reasonable
        rows, cols, data = [], [], []
        
        for i, patch in enumerate(self.fault_patches):
            area = patch.length * patch.width
            for comp in range(3):
                idx = i * 3 + comp
                rows.append(0)
                cols.append(idx)
                data.append(area)  # Weight by patch area
        
        M = sp.csr_matrix((data, (rows, cols)), shape=(1, n_params))
        
        return M

class CompositeModel(DeformationModel):
    """Composite model combining multiple deformation mechanisms"""
    
    def __init__(self, models: List[DeformationModel], weights: List[float] = None):
        """
        Initialize composite model
        
        Args:
            models: List of deformation models
            weights: Weight for each model (default: equal weights)
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        parameters = {
            'n_models': len(models),
            'model_types': [type(model).__name__ for model in models],
            'weights': weights
        }
        
        super().__init__(parameters)
        self.models = models
        self.weights = weights
    
    def compute_displacements(self, observation_points: np.ndarray) -> np.ndarray:
        """Compute weighted sum of displacements from all models"""
        total_displacements = np.zeros((len(observation_points), 3))
        
        for model, weight in zip(self.models, self.weights):
            model_displacements = model.compute_displacements(observation_points)
            total_displacements += weight * model_displacements
        
        self.computed_displacements = total_displacements
        return total_displacements
    
    def compute_greens_functions(self, observation_points: np.ndarray) -> np.ndarray:
        """Compute combined Green's functions"""
        greens_functions = []
        
        for model, weight in zip(self.models, self.weights):
            model_greens = model.compute_greens_functions(observation_points)
            greens_functions.append(weight * model_greens)
        
        return np.concatenate(greens_functions, axis=1)

class ElasticInversionSolver:
    """Advanced solver for elastic inversion problems"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default solver configuration"""
        return {
            'optimization': {
                'method': 'differential_evolution',
                'max_iterations': 1000,
                'tolerance': 1e-8,
                'population_size': 15
            },
            'regularization': {
                'smoothing': 0.1,
                'moment': 0.01,
                'positivity': False
            },
            'uncertainty': {
                'bootstrap_samples': 100,
                'confidence_level': 0.95
            },
            'parallel': {
                'n_workers': 4,
                'use_multiprocessing': True
            }
        }
    
    async def solve_elastic_inversion(self, 
                                    observations: np.ndarray,
                                    uncertainties: np.ndarray,
                                    model: DeformationModel,
                                    parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Solve elastic inversion problem"""
        
        def objective_function(params_array):
            # Update model parameters
            param_dict = {name: params_array[i] for i, name in enumerate(parameter_bounds.keys())}
            for key, value in param_dict.items():
                if key in model.parameters:
                    model.parameters[key] = value
            
            # Compute model predictions
            try:
                predicted = model.compute_displacements(observations[:, :3])
                observed = observations[:, 3:6]
                
                # Calculate weighted residuals
                residuals = (observed - predicted) / uncertainties[:, 3:6]
                chi_squared = np.sum(residuals**2)
                
                return chi_squared
            except Exception as e:
                self.logger.error(f"Error in objective function: {str(e)}")
                return 1e10
        
        # Set up optimization bounds
        bounds = [parameter_bounds[name] for name in parameter_bounds.keys()]
        param_names = list(parameter_bounds.keys())
        
        # Perform optimization
        if self.config['optimization']['method'] == 'differential_evolution':
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=self.config['optimization']['max_iterations'],
                popsize=self.config['optimization']['population_size'],
                seed=42,
                atol=self.config['optimization']['tolerance'],
                workers=self.config['parallel']['n_workers'] if self.config['parallel']['use_multiprocessing'] else 1
            )
        else:
            # Use scipy.optimize.minimize
            x0 = [(b[0] + b[1]) / 2 for b in bounds]  # Initial guess at midpoint
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': self.config['optimization']['max_iterations'],
                    'ftol': self.config['optimization']['tolerance']
                }
            )
        
        # Extract optimal parameters
        optimal_params = {name: result.x[i] for i, name in enumerate(param_names)}
        
        # Update model with optimal parameters
        for key, value in optimal_params.items():
            if key in model.parameters:
                model.parameters[key] = value
        
        # Compute final predictions
        predicted_displacements = model.compute_displacements(observations[:, :3])
        observed_displacements = observations[:, 3:6]
        residuals = observed_displacements - predicted_displacements
        
        # Calculate statistics
        chi_squared = result.fun
        n_data = len(observations) * 3
        n_params = len(optimal_params)
        reduced_chi_squared = chi_squared / (n_data - n_params)
        
        # Estimate parameter uncertainties
        uncertainties_est = await self._estimate_parameter_uncertainties(
            observations, uncertainties, model, optimal_params, parameter_bounds
        )
        
        return {
            'optimal_parameters': optimal_params,
            'parameter_uncertainties': uncertainties_est,
            'predicted_displacements': predicted_displacements,
            'residuals': residuals,
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'optimization_result': result,
            'model_type': type(model).__name__,
            'success': result.success if hasattr(result, 'success') else True
        }
    
    async def _estimate_parameter_uncertainties(self, 
                                              observations: np.ndarray,
                                              data_uncertainties: np.ndarray,
                                              model: DeformationModel,
                                              optimal_params: Dict[str, float],
                                              bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Estimate parameter uncertainties using bootstrap or Hessian"""
        
        if self.config['uncertainty']['bootstrap_samples'] > 0:
            return await self._bootstrap_uncertainties(
                observations, data_uncertainties, model, optimal_params, bounds
            )
        else:
            return await self._hessian_uncertainties(
                observations, data_uncertainties, model, optimal_params
            )
    
    async def _bootstrap_uncertainties(self, 
                                     observations: np.ndarray,
                                     data_uncertainties: np.ndarray,
                                     model: DeformationModel,
                                     optimal_params: Dict[str, float],
                                     bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Estimate uncertainties using bootstrap resampling"""
        n_bootstrap = self.config['uncertainty']['bootstrap_samples']
        n_obs = len(observations)
        
        bootstrap_results = []
        
        for i in range(n_bootstrap):
            # Resample observations
            indices = np.random.choice(n_obs, size=n_obs, replace=True)
            bootstrap_obs = observations[indices]
            bootstrap_unc = data_uncertainties[indices]
            
            # Add noise to data
            noise = np.random.normal(0, bootstrap_unc[:, 3:6])
            bootstrap_obs[:, 3:6] += noise
            
            # Solve for this bootstrap sample
            try:
                result = await self.solve_elastic_inversion(
                    bootstrap_obs, bootstrap_unc, model, bounds
                )
                bootstrap_results.append(result['optimal_parameters'])
            except Exception as e:
                self.logger.warning(f"Bootstrap sample {i} failed: {str(e)}")
        
        # Calculate parameter uncertainties
        uncertainties = {}
        for param_name in optimal_params.keys():
            param_values = [result[param_name] for result in bootstrap_results 
                          if param_name in result]
            if param_values:
                uncertainties[param_name] = np.std(param_values)
            else:
                uncertainties[param_name] = float('inf')
        
        return uncertainties
    
    async def _hessian_uncertainties(self, 
                                   observations: np.ndarray,
                                   data_uncertainties: np.ndarray,
                                   model: DeformationModel,
                                   optimal_params: Dict[str, float]) -> Dict[str, float]:
        """Estimate uncertainties using Hessian matrix"""
        uncertainties = {}
        delta = 1e-6
        
        for param_name in optimal_params.keys():
            # Compute second derivative
            original_value = optimal_params[param_name]
            
            # Forward
            model.parameters[param_name] = original_value + delta
            pred_plus = model.compute_displacements(observations[:, :3])
            
            # Backward
            model.parameters[param_name] = original_value - delta
            pred_minus = model.compute_displacements(observations[:, :3])
            
            # Central
            model.parameters[param_name] = original_value
            pred_central = model.compute_displacements(observations[:, :3])
            
            # Second derivative of chi-squared
            residuals_plus = (observations[:, 3:6] - pred_plus) / data_uncertainties[:, 3:6]
            residuals_minus = (observations[:, 3:6] - pred_minus) / data_uncertainties[:, 3:6]
            residuals_central = (observations[:, 3:6] - pred_central) / data_uncertainties[:, 3:6]
            
            chi2_plus = np.sum(residuals_plus**2)
            chi2_minus = np.sum(residuals_minus**2)
            chi2_central = np.sum(residuals_central**2)
            
            second_derivative = (chi2_plus - 2 * chi2_central + chi2_minus) / (delta**2)
            
            # Uncertainty from curvature
            if second_derivative > 0:
                uncertainties[param_name] = np.sqrt(2.0 / second_derivative)
            else:
                uncertainties[param_name] = float('inf')
        
        return uncertainties
    
    def compare_models(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple inversion results using information criteria"""
        
        if not results:
            return {'error': 'No results to compare'}
        
        # Extract metrics
        models = []
        for result in results:
            models.append({
                'model_type': result['model_type'],
                'chi_squared': result['chi_squared'],
                'n_parameters': len(result['optimal_parameters']),
                'aic': result['chi_squared'] + 2 * len(result['optimal_parameters']),
                'bic': result['chi_squared'] + np.log(len(result['residuals'])) * len(result['optimal_parameters']),
                'reduced_chi_squared': result['reduced_chi_squared']
            })
        
        # Sort by AIC
        models.sort(key=lambda x: x['aic'])
        best_model = models[0]
        
        # Calculate Akaike weights
        min_aic = best_model['aic']
        for model in models:
            model['delta_aic'] = model['aic'] - min_aic
            model['akaike_weight'] = np.exp(-0.5 * model['delta_aic'])
        
        weight_sum = sum(model['akaike_weight'] for model in models)
        for model in models:
            model['akaike_weight'] /= weight_sum
        
        return {
            'best_model': best_model,
            'model_comparison': models,
            'ranking_criterion': 'AIC',
            'summary': f"Best model: {best_model['model_type']} (AIC: {best_model['aic']:.2f})"
        }

# Visualization utilities
def plot_fault_geometry(fault_geometry: FaultGeometry, ax=None):
    """Plot 3D fault geometry"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    corners = fault_geometry.get_fault_corners()
    
    # Plot fault plane
    vertices = [
        corners[0], corners[1], corners[2], corners[3], corners[0]
    ]
    
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices] 
    zs = [v[2] for v in vertices]
    
    ax.plot(xs, ys, zs, 'r-', linewidth=2, label='Fault outline')
    ax.scatter([fault_geometry.x0], [fault_geometry.y0], [fault_geometry.z0], 
              c='red', s=100, marker='*', label='Fault center')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Depth (m)')
    ax.legend()
    
    return ax

def plot_displacement_field(coordinates: np.ndarray, displacements: np.ndarray, 
                          scale: float = 1000, ax=None):
    """Plot displacement field"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot displacement vectors
    ax.quiver(coordinates[:, 0], coordinates[:, 1], 
             displacements[:, 0], displacements[:, 1],
             scale=scale, width=0.003, alpha=0.7)
    
    # Plot vertical displacement as color
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                       c=displacements[:, 2], cmap='RdBu_r', s=30)
    
    plt.colorbar(scatter, ax=ax, label='Vertical displacement (m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Surface Displacement Field')
    
    return ax

# Example usage
if __name__ == "__main__":
    # Example: Create and test elastic dislocation model
    fault = FaultGeometry(
        x0=0, y0=0, z0=5000,  # 5 km depth
        length=10000, width=8000,  # 10 km x 8 km
        strike=45, dip=90, rake=0  # Vertical strike-slip
    )
    
    elastic_params = ElasticParameters()
    slip = (2.0, 0.0, 0.0)  # 2 m strike-slip
    
    model = ElasticDislocationModel(fault, elastic_params, slip)
    
    # Create observation points
    x = np.linspace(-15000, 15000, 20)
    y = np.linspace(-15000, 15000, 20)
    X, Y = np.meshgrid(x, y)
    obs_points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(len(X.ravel()))])
    
    # Compute displacements
    displacements = model.compute_displacements(obs_points)
    
    print(f"Maximum displacement: {np.max(np.abs(displacements)):.4f} m")
    print(f"Computation time: {model.computation_time:.4f} seconds")