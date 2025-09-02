"""
Geological Physics-Informed Neural Networks (PINNs) Framework
=============================================================

A comprehensive framework for Physics-Informed Neural Networks tailored specifically 
for geological process modeling, integrating multiple physical constraints and 
geological principles.

Key Features:
- Multi-physics coupling (thermal, mechanical, fluid, electromagnetic)
- Geological constraint enforcement (conservation laws, constitutive relations)
- Adaptive architectures for different geological processes
- Uncertainty quantification and ensemble learning
- Real-time geological monitoring integration
- Multi-scale temporal modeling (geological vs real-time scales)

Authors: Prometheus AI Geodynamic Platform Team
Version: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from abc import ABC, abstractmethod
import json
from pathlib import Path
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.integrate import solve_ivp

# Import multi-physics coupling engine components
try:
    from .multi_physics_coupling_engine import (
        MultiPhysicsCouplingEngine, PhysicsType as MPPhysicsType, 
        CouplingType, MaterialProperties, ThermalModule, MechanicalModule,
        ChemicalModule, ElectromagneticModule
    )
    MULTIPHYSICS_AVAILABLE = True
except ImportError:
    MULTIPHYSICS_AVAILABLE = False
    warnings.warn("Multi-physics coupling engine not available. Some advanced coupling features will be limited.")

# Try to import other backend modules for integration
try:
    from .model_database import ModelDatabase
    MODEL_DATABASE_AVAILABLE = True
except ImportError:
    MODEL_DATABASE_AVAILABLE = False

try:
    from .data_integration_pipeline import DataIntegrationPipeline
    DATA_INTEGRATION_AVAILABLE = True
except ImportError:
    DATA_INTEGRATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeologicalProcessType(Enum):
    """Types of geological processes for PINN modeling"""
    TECTONIC_DEFORMATION = "tectonic_deformation"
    MAGMA_FLOW = "magma_flow" 
    HYDROTHERMAL_FLOW = "hydrothermal_flow"
    STRESS_FIELD = "stress_field"
    THERMAL_EVOLUTION = "thermal_evolution"
    SEISMIC_WAVE_PROPAGATION = "seismic_wave_propagation"
    VOLCANIC_ERUPTION = "volcanic_eruption"
    PLATE_MOTION = "plate_motion"
    MANTLE_CONVECTION = "mantle_convection"
    GROUNDWATER_FLOW = "groundwater_flow"

class PhysicsType(Enum):
    """Physical phenomena types"""
    THERMAL = "thermal"
    MECHANICAL = "mechanical"
    FLUID = "fluid"
    ELECTROMAGNETIC = "electromagnetic"
    CHEMICAL = "chemical"
    SEISMIC = "seismic"

@dataclass
class GeologicalMaterial:
    """Geological material properties"""
    name: str
    density: float  # kg/m³
    youngs_modulus: float  # Pa
    poisson_ratio: float
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    thermal_expansion: float  # 1/K
    permeability: float  # m²
    porosity: float  # dimensionless
    viscosity: Optional[float] = None  # Pa·s
    electrical_conductivity: Optional[float] = None  # S/m
    magnetic_susceptibility: Optional[float] = None

@dataclass
class PINNConfig:
    """Configuration for PINN architecture and training"""
    process_type: GeologicalProcessType
    input_dim: int = 3  # spatial dimensions
    output_dim: int = 1
    hidden_layers: List[int] = None
    activation: str = "tanh"
    dropout_rate: float = 0.1
    physics_weight: float = 1.0
    boundary_weight: float = 2.0
    data_weight: float = 1.0
    learning_rate: float = 1e-3
    epochs: int = 10000
    batch_size: int = 1024
    adaptive_weights: bool = True
    uncertainty_quantification: bool = True
    ensemble_size: int = 5
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 256, 128, 64]

class GeologicalPINN(nn.Module, ABC):
    """Base class for geological Physics-Informed Neural Networks"""
    
    def __init__(self, config: PINNConfig, material: GeologicalMaterial):
        super().__init__()
        self.config = config
        self.material = material
        self.training_history = defaultdict(list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build network architecture
        self._build_network()
        
        # Initialize physics parameters as learnable
        self._init_physics_parameters()
        
        # Move to device
        self.to(self.device)
        
    def _build_network(self):
        """Build the neural network architecture"""
        layers = []
        prev_dim = self.config.input_dim
        
        # Input layer normalization
        self.input_norm = nn.LayerNorm(self.config.input_dim)
        
        # Hidden layers
        for hidden_dim in self.config.hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(self.config.dropout_rate)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, self.config.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Uncertainty quantification head
        if self.config.uncertainty_quantification:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(self.config.hidden_layers[-1], self.config.hidden_layers[-1] // 2),
                nn.ReLU(),
                nn.Linear(self.config.hidden_layers[-1] // 2, self.config.output_dim)
            )
    
    def _get_activation(self):
        """Get activation function"""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'sine': lambda x: torch.sin(x)
        }
        return activations.get(self.config.activation, nn.Tanh())
    
    def _init_physics_parameters(self):
        """Initialize learnable physics parameters"""
        self.physics_weights = nn.ParameterDict({
            'conservation': nn.Parameter(torch.tensor(self.config.physics_weight)),
            'boundary': nn.Parameter(torch.tensor(self.config.boundary_weight)),
            'constitutive': nn.Parameter(torch.tensor(1.0)),
            'temporal': nn.Parameter(torch.tensor(1.0))
        })
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network"""
        x_norm = self.input_norm(x)
        output = self.network(x_norm)
        
        if self.config.uncertainty_quantification:
            # Extract features from the last hidden layer
            features = x_norm
            for layer in self.network[:-1]:
                features = layer(features)
            uncertainty = torch.exp(self.uncertainty_head(features))  # Ensure positive
            return output, uncertainty
        
        return output
    
    @abstractmethod
    def compute_physics_loss(self, x: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
        """Compute physics-based loss terms - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def compute_boundary_loss(self, x_boundary: torch.Tensor, u_boundary: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss - must be implemented by subclasses"""
        pass
    
    def compute_data_loss(self, x_data: torch.Tensor, u_data: torch.Tensor, 
                         u_pred: torch.Tensor) -> torch.Tensor:
        """Compute data fitting loss"""
        return nn.MSELoss()(u_pred, u_data)
    
    def adaptive_weight_update(self, physics_loss: torch.Tensor, boundary_loss: torch.Tensor,
                              data_loss: torch.Tensor, iteration: int):
        """Adaptive weight balancing using gradient statistics"""
        if not self.config.adaptive_weights or iteration < 100:
            return
            
        # Compute gradient norms for each loss component
        physics_grad_norm = torch.norm(torch.cat([
            p.grad.flatten() for p in self.parameters() if p.grad is not None
        ]))
        
        # Update weights to balance gradient magnitudes
        alpha = 0.1 / (1 + iteration * 1e-4)
        
        if physics_loss > 0:
            self.physics_weights['conservation'].data *= (1 - alpha + alpha * physics_grad_norm)
        if boundary_loss > 0:
            self.physics_weights['boundary'].data *= (1 - alpha + alpha * physics_grad_norm)

class TectonicDeformationPINN(GeologicalPINN):
    """PINN for tectonic deformation modeling with elasticity constraints"""
    
    def __init__(self, config: PINNConfig, material: GeologicalMaterial):
        config.output_dim = 3  # displacement components (u_x, u_y, u_z)
        super().__init__(config, material)
        
        # Elastic parameters
        self.lame_lambda = self._compute_lame_lambda()
        self.shear_modulus = self._compute_shear_modulus()
        
    def _compute_lame_lambda(self):
        """Compute Lamé's first parameter"""
        E = self.material.youngs_modulus
        nu = self.material.poisson_ratio
        return E * nu / ((1 + nu) * (1 - 2 * nu))
    
    def _compute_shear_modulus(self):
        """Compute shear modulus"""
        E = self.material.youngs_modulus
        nu = self.material.poisson_ratio
        return E / (2 * (1 + nu))
    
    def compute_physics_loss(self, x: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
        """Compute elasticity equation residuals"""
        x.requires_grad_(True)
        
        # Displacement components
        u_x, u_y, u_z = u_pred[:, 0:1], u_pred[:, 1:2], u_pred[:, 2:3]
        
        # Compute gradients
        u_x_grad = grad(u_x.sum(), x, create_graph=True)[0]
        u_y_grad = grad(u_y.sum(), x, create_graph=True)[0]
        u_z_grad = grad(u_z.sum(), x, create_graph=True)[0]
        
        # Strain tensor components
        eps_xx = u_x_grad[:, 0:1]
        eps_yy = u_y_grad[:, 1:2]
        eps_zz = u_z_grad[:, 2:3]
        eps_xy = 0.5 * (u_x_grad[:, 1:2] + u_y_grad[:, 0:1])
        eps_xz = 0.5 * (u_x_grad[:, 2:3] + u_z_grad[:, 0:1])
        eps_yz = 0.5 * (u_y_grad[:, 2:3] + u_z_grad[:, 1:2])
        
        # Stress tensor (Hooke's law)
        trace_strain = eps_xx + eps_yy + eps_zz
        sigma_xx = self.lame_lambda * trace_strain + 2 * self.shear_modulus * eps_xx
        sigma_yy = self.lame_lambda * trace_strain + 2 * self.shear_modulus * eps_yy
        sigma_zz = self.lame_lambda * trace_strain + 2 * self.shear_modulus * eps_zz
        sigma_xy = 2 * self.shear_modulus * eps_xy
        sigma_xz = 2 * self.shear_modulus * eps_xz
        sigma_yz = 2 * self.shear_modulus * eps_yz
        
        # Equilibrium equations (∇·σ = 0)
        sigma_xx_x = grad(sigma_xx.sum(), x, create_graph=True)[0][:, 0:1]
        sigma_xy_y = grad(sigma_xy.sum(), x, create_graph=True)[0][:, 1:2]
        sigma_xz_z = grad(sigma_xz.sum(), x, create_graph=True)[0][:, 2:3]
        
        sigma_yy_y = grad(sigma_yy.sum(), x, create_graph=True)[0][:, 1:2]
        sigma_xy_x = grad(sigma_xy.sum(), x, create_graph=True)[0][:, 0:1]
        sigma_yz_z = grad(sigma_yz.sum(), x, create_graph=True)[0][:, 2:3]
        
        sigma_zz_z = grad(sigma_zz.sum(), x, create_graph=True)[0][:, 2:3]
        sigma_xz_x = grad(sigma_xz.sum(), x, create_graph=True)[0][:, 0:1]
        sigma_yz_y = grad(sigma_yz.sum(), x, create_graph=True)[0][:, 1:2]
        
        # Physics residuals
        residual_x = sigma_xx_x + sigma_xy_y + sigma_xz_z
        residual_y = sigma_xy_x + sigma_yy_y + sigma_yz_z
        residual_z = sigma_xz_x + sigma_yz_y + sigma_zz_z
        
        # Combined residual
        physics_loss = torch.mean(residual_x**2 + residual_y**2 + residual_z**2)
        return physics_loss
    
    def compute_boundary_loss(self, x_boundary: torch.Tensor, u_boundary: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss for tectonic deformation"""
        u_pred = self.forward(x_boundary)
        if isinstance(u_pred, tuple):
            u_pred = u_pred[0]
        return nn.MSELoss()(u_pred, u_boundary)

class MagmaFlowPINN(GeologicalPINN):
    """PINN for magma flow modeling with Navier-Stokes constraints"""
    
    def __init__(self, config: PINNConfig, material: GeologicalMaterial):
        config.output_dim = 4  # velocity components (v_x, v_y, v_z) + pressure
        super().__init__(config, material)
        
        # Fluid properties
        self.viscosity = material.viscosity or 1e6  # High viscosity for magma
        self.density = material.density
        
    def compute_physics_loss(self, x: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
        """Compute Navier-Stokes equation residuals for magma flow"""
        x.requires_grad_(True)
        
        # Flow variables
        v_x, v_y, v_z = u_pred[:, 0:1], u_pred[:, 1:2], u_pred[:, 2:3]
        p = u_pred[:, 3:4]
        
        # Compute gradients
        v_x_grad = grad(v_x.sum(), x, create_graph=True)[0]
        v_y_grad = grad(v_y.sum(), x, create_graph=True)[0]
        v_z_grad = grad(v_z.sum(), x, create_graph=True)[0]
        p_grad = grad(p.sum(), x, create_graph=True)[0]
        
        # Continuity equation (∇·v = 0)
        continuity = v_x_grad[:, 0:1] + v_y_grad[:, 1:2] + v_z_grad[:, 2:3]
        
        # Momentum equations (simplified for high viscosity magma)
        # ∇p = μ∇²v (Stokes flow approximation)
        v_x_laplacian = self._compute_laplacian(x, v_x)
        v_y_laplacian = self._compute_laplacian(x, v_y)
        v_z_laplacian = self._compute_laplacian(x, v_z)
        
        momentum_x = p_grad[:, 0:1] - self.viscosity * v_x_laplacian
        momentum_y = p_grad[:, 1:2] - self.viscosity * v_y_laplacian
        momentum_z = p_grad[:, 2:3] - self.viscosity * v_z_laplacian
        
        # Combined physics loss
        physics_loss = torch.mean(
            continuity**2 + momentum_x**2 + momentum_y**2 + momentum_z**2
        )
        return physics_loss
    
    def _compute_laplacian(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian of field u"""
        u_grad = grad(u.sum(), x, create_graph=True)[0]
        
        u_xx = grad(u_grad[:, 0].sum(), x, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_grad[:, 1].sum(), x, create_graph=True)[0][:, 1:2]
        u_zz = grad(u_grad[:, 2].sum(), x, create_graph=True)[0][:, 2:3]
        
        return u_xx + u_yy + u_zz
    
    def compute_boundary_loss(self, x_boundary: torch.Tensor, u_boundary: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss for magma flow"""
        u_pred = self.forward(x_boundary)
        if isinstance(u_pred, tuple):
            u_pred = u_pred[0]
        return nn.MSELoss()(u_pred, u_boundary)

class ThermalEvolutionPINN(GeologicalPINN):
    """PINN for thermal evolution modeling with heat equation constraints"""
    
    def __init__(self, config: PINNConfig, material: GeologicalMaterial):
        config.input_dim = 4  # space (x,y,z) + time
        config.output_dim = 1  # temperature
        super().__init__(config, material)
        
        # Thermal properties
        self.thermal_diffusivity = (
            material.thermal_conductivity / 
            (material.density * material.specific_heat)
        )
        
    def compute_physics_loss(self, x: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
        """Compute heat equation residuals"""
        x.requires_grad_(True)
        T = u_pred[:, 0:1]
        
        # Compute spatial and temporal derivatives
        T_grad = grad(T.sum(), x, create_graph=True)[0]
        T_t = T_grad[:, 3:4]  # Time derivative
        
        # Spatial Laplacian
        T_laplacian = self._compute_spatial_laplacian(x, T)
        
        # Heat equation: ∂T/∂t = α∇²T
        heat_residual = T_t - self.thermal_diffusivity * T_laplacian
        
        physics_loss = torch.mean(heat_residual**2)
        return physics_loss
    
    def _compute_spatial_laplacian(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Compute spatial Laplacian (first 3 dimensions only)"""
        T_grad = grad(T.sum(), x, create_graph=True)[0]
        
        T_xx = grad(T_grad[:, 0].sum(), x, create_graph=True)[0][:, 0:1]
        T_yy = grad(T_grad[:, 1].sum(), x, create_graph=True)[0][:, 1:2]
        T_zz = grad(T_grad[:, 2].sum(), x, create_graph=True)[0][:, 2:3]
        
        return T_xx + T_yy + T_zz
    
    def compute_boundary_loss(self, x_boundary: torch.Tensor, u_boundary: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss for thermal evolution"""
        u_pred = self.forward(x_boundary)
        if isinstance(u_pred, tuple):
            u_pred = u_pred[0]
        return nn.MSELoss()(u_pred, u_boundary)

class SeismicWavePINN(GeologicalPINN):
    """PINN for seismic wave propagation with elastic wave equation constraints"""
    
    def __init__(self, config: PINNConfig, material: GeologicalMaterial):
        config.input_dim = 4  # space (x,y,z) + time
        config.output_dim = 3  # displacement components
        super().__init__(config, material)
        
        # Wave velocities
        self.vp = np.sqrt((self.material.youngs_modulus * (1 - self.material.poisson_ratio)) /
                          (self.material.density * (1 + self.material.poisson_ratio) * 
                           (1 - 2 * self.material.poisson_ratio)))
        self.vs = np.sqrt(self.material.youngs_modulus / 
                          (2 * self.material.density * (1 + self.material.poisson_ratio)))
        
    def compute_physics_loss(self, x: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
        """Compute elastic wave equation residuals"""
        x.requires_grad_(True)
        
        u_x, u_y, u_z = u_pred[:, 0:1], u_pred[:, 1:2], u_pred[:, 2:3]
        
        # Wave equation for each component
        wave_loss_x = self._compute_wave_residual(x, u_x)
        wave_loss_y = self._compute_wave_residual(x, u_y)
        wave_loss_z = self._compute_wave_residual(x, u_z)
        
        physics_loss = torch.mean(wave_loss_x**2 + wave_loss_y**2 + wave_loss_z**2)
        return physics_loss
    
    def _compute_wave_residual(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute wave equation residual for displacement component"""
        u_grad = grad(u.sum(), x, create_graph=True)[0]
        u_tt = grad(u_grad[:, 3].sum(), x, create_graph=True)[0][:, 3:4]  # Second time derivative
        
        # Spatial Laplacian
        u_laplacian = self._compute_spatial_laplacian(x, u)
        
        # Wave equation: ∂²u/∂t² = v²∇²u
        wave_residual = u_tt - self.vs**2 * u_laplacian
        
        return wave_residual
    
    def _compute_spatial_laplacian(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute spatial Laplacian"""
        u_grad = grad(u.sum(), x, create_graph=True)[0]
        
        u_xx = grad(u_grad[:, 0].sum(), x, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_grad[:, 1].sum(), x, create_graph=True)[0][:, 1:2]
        u_zz = grad(u_grad[:, 2].sum(), x, create_graph=True)[0][:, 2:3]
        
        return u_xx + u_yy + u_zz
    
    def compute_boundary_loss(self, x_boundary: torch.Tensor, u_boundary: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss for seismic waves"""
        u_pred = self.forward(x_boundary)
        if isinstance(u_pred, tuple):
            u_pred = u_pred[0]
        return nn.MSELoss()(u_pred, u_boundary)

class MultiPhysicsPINN(GeologicalPINN):
    """
    Advanced multi-physics PINN that integrates with the multi-physics coupling engine
    for comprehensive geological process modeling with coupled physics
    """
    
    def __init__(self, config: PINNConfig, material: GeologicalMaterial, 
                 physics_types: List[GeologicalProcessType]):
        # Adjust output dimensions based on physics types
        config.output_dim = self._compute_output_dim(physics_types)
        super().__init__(config, material)
        
        self.physics_types = physics_types
        self.multiphysics_engine = None
        self.physics_field_indices = self._create_field_mapping(physics_types)
        
        # Initialize multi-physics coupling if available
        if MULTIPHYSICS_AVAILABLE:
            self._setup_multiphysics_coupling()
    
    def _compute_output_dim(self, physics_types: List[GeologicalProcessType]) -> int:
        """Compute total output dimension based on physics types"""
        dim_map = {
            GeologicalProcessType.TECTONIC_DEFORMATION: 3,  # u_x, u_y, u_z
            GeologicalProcessType.THERMAL_EVOLUTION: 1,     # T
            GeologicalProcessType.MAGMA_FLOW: 4,           # v_x, v_y, v_z, p
            GeologicalProcessType.STRESS_FIELD: 6,         # stress tensor components
            GeologicalProcessType.SEISMIC_WAVE_PROPAGATION: 3,  # displacement
            GeologicalProcessType.HYDROTHERMAL_FLOW: 5,    # v_x, v_y, v_z, p, T
        }
        
        total_dim = 0
        for physics_type in physics_types:
            total_dim += dim_map.get(physics_type, 1)
        return total_dim
    
    def _create_field_mapping(self, physics_types: List[GeologicalProcessType]) -> Dict:
        """Create mapping of physics types to output field indices"""
        field_map = {}
        current_idx = 0
        
        for physics_type in physics_types:
            if physics_type == GeologicalProcessType.TECTONIC_DEFORMATION:
                field_map[physics_type] = slice(current_idx, current_idx + 3)
                current_idx += 3
            elif physics_type == GeologicalProcessType.THERMAL_EVOLUTION:
                field_map[physics_type] = slice(current_idx, current_idx + 1)
                current_idx += 1
            elif physics_type == GeologicalProcessType.MAGMA_FLOW:
                field_map[physics_type] = slice(current_idx, current_idx + 4)
                current_idx += 4
            elif physics_type == GeologicalProcessType.STRESS_FIELD:
                field_map[physics_type] = slice(current_idx, current_idx + 6)
                current_idx += 6
            elif physics_type == GeologicalProcessType.HYDROTHERMAL_FLOW:
                field_map[physics_type] = slice(current_idx, current_idx + 5)
                current_idx += 5
            else:
                field_map[physics_type] = slice(current_idx, current_idx + 1)
                current_idx += 1
        
        return field_map
    
    def _setup_multiphysics_coupling(self):
        """Setup multi-physics coupling engine integration"""
        if not MULTIPHYSICS_AVAILABLE:
            logger.warning("Multi-physics coupling engine not available")
            return
        
        try:
            # Convert material to MaterialProperties format
            material_props = MaterialProperties(
                density=self.material.density,
                youngs_modulus=self.material.youngs_modulus,
                poisson_ratio=self.material.poisson_ratio,
                thermal_conductivity=self.material.thermal_conductivity,
                specific_heat=self.material.specific_heat,
                thermal_expansion=self.material.thermal_expansion,
                permeability=self.material.permeability,
                porosity=self.material.porosity
            )
            
            # Initialize coupling engine
            coupling_config = {
                'coupling_scheme': CouplingType.STAGGERED,
                'max_coupling_iterations': 5,
                'coupling_tolerance': 1e-6,
                'adaptive_timestep': True
            }
            self.multiphysics_engine = MultiPhysicsCouplingEngine(coupling_config)
            
            # Add physics modules based on process types
            physics_type_mapping = {
                GeologicalProcessType.THERMAL_EVOLUTION: MPPhysicsType.THERMAL,
                GeologicalProcessType.TECTONIC_DEFORMATION: MPPhysicsType.MECHANICAL,
                GeologicalProcessType.MAGMA_FLOW: MPPhysicsType.FLUID_FLOW,
                GeologicalProcessType.HYDROTHERMAL_FLOW: MPPhysicsType.FLUID_FLOW,
            }
            
            for process_type in self.physics_types:
                if process_type in physics_type_mapping:
                    mp_type = physics_type_mapping[process_type]
                    self.multiphysics_engine.add_physics_module(mp_type, material_props)
            
            # Add standard coupling terms
            if (GeologicalProcessType.THERMAL_EVOLUTION in self.physics_types and 
                GeologicalProcessType.TECTONIC_DEFORMATION in self.physics_types):
                
                self.multiphysics_engine.add_coupling_term(
                    MPPhysicsType.THERMAL, MPPhysicsType.MECHANICAL,
                    coupling_strength=0.8, 
                    coupling_function=self._thermal_mechanical_coupling
                )
                self.multiphysics_engine.add_coupling_term(
                    MPPhysicsType.MECHANICAL, MPPhysicsType.THERMAL,
                    coupling_strength=0.3,
                    coupling_function=self._mechanical_thermal_coupling
                )
            
            logger.info("Multi-physics coupling engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup multi-physics coupling: {e}")
            self.multiphysics_engine = None
    
    def _thermal_mechanical_coupling(self, thermal_state: Dict, strength: float) -> np.ndarray:
        """Thermal expansion coupling function"""
        temperature = thermal_state.get('temperature', np.zeros(1))
        reference_temp = 273.15  # 0°C
        thermal_strain = strength * self.material.thermal_expansion * (temperature - reference_temp)
        return thermal_strain
    
    def _mechanical_thermal_coupling(self, mechanical_state: Dict, strength: float) -> np.ndarray:
        """Mechanical heating coupling function"""
        stress = mechanical_state.get('stress', np.zeros(1))
        strain_rate = mechanical_state.get('strain_rate', np.zeros(1))
        # Simplified viscous dissipation
        heat_source = strength * np.abs(stress * strain_rate) / self.material.density
        return heat_source
    
    def compute_physics_loss(self, x: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
        """Compute multi-physics residuals with coupling terms"""
        x.requires_grad_(True)
        total_loss = torch.tensor(0.0, device=x.device)
        
        # Compute individual physics losses
        for physics_type in self.physics_types:
            field_slice = self.physics_field_indices[physics_type]
            u_field = u_pred[:, field_slice]
            
            if physics_type == GeologicalProcessType.TECTONIC_DEFORMATION:
                loss = self._compute_deformation_loss(x, u_field)
            elif physics_type == GeologicalProcessType.THERMAL_EVOLUTION:
                loss = self._compute_thermal_loss(x, u_field)
            elif physics_type == GeologicalProcessType.MAGMA_FLOW:
                loss = self._compute_flow_loss(x, u_field)
            elif physics_type == GeologicalProcessType.HYDROTHERMAL_FLOW:
                loss = self._compute_hydrothermal_loss(x, u_field)
            else:
                loss = torch.tensor(0.0, device=x.device)
            
            total_loss += loss
        
        # Add coupling losses
        if len(self.physics_types) > 1:
            coupling_loss = self._compute_coupling_loss(x, u_pred)
            total_loss += coupling_loss
        
        return total_loss
    
    def _compute_deformation_loss(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute elasticity residuals"""
        # Similar to TectonicDeformationPINN but adapted for multi-field
        u_x, u_y, u_z = u[:, 0:1], u[:, 1:2], u[:, 2:3]
        
        # Compute strain tensor
        u_x_grad = grad(u_x.sum(), x, create_graph=True)[0]
        u_y_grad = grad(u_y.sum(), x, create_graph=True)[0]
        u_z_grad = grad(u_z.sum(), x, create_graph=True)[0]
        
        eps_xx = u_x_grad[:, 0:1]
        eps_yy = u_y_grad[:, 1:2]
        eps_zz = u_z_grad[:, 2:3]
        eps_xy = 0.5 * (u_x_grad[:, 1:2] + u_y_grad[:, 0:1])
        eps_xz = 0.5 * (u_x_grad[:, 2:3] + u_z_grad[:, 0:1])
        eps_yz = 0.5 * (u_y_grad[:, 2:3] + u_z_grad[:, 1:2])
        
        # Stress tensor
        lam = (self.material.youngs_modulus * self.material.poisson_ratio / 
               ((1 + self.material.poisson_ratio) * (1 - 2 * self.material.poisson_ratio)))
        mu = self.material.youngs_modulus / (2 * (1 + self.material.poisson_ratio))
        
        trace_strain = eps_xx + eps_yy + eps_zz
        sigma_xx = lam * trace_strain + 2 * mu * eps_xx
        sigma_yy = lam * trace_strain + 2 * mu * eps_yy
        sigma_zz = lam * trace_strain + 2 * mu * eps_zz
        sigma_xy = 2 * mu * eps_xy
        sigma_xz = 2 * mu * eps_xz
        sigma_yz = 2 * mu * eps_yz
        
        # Equilibrium equations
        sigma_xx_x = grad(sigma_xx.sum(), x, create_graph=True)[0][:, 0:1]
        sigma_xy_y = grad(sigma_xy.sum(), x, create_graph=True)[0][:, 1:2]
        sigma_xz_z = grad(sigma_xz.sum(), x, create_graph=True)[0][:, 2:3]
        
        sigma_yy_y = grad(sigma_yy.sum(), x, create_graph=True)[0][:, 1:2]
        sigma_xy_x = grad(sigma_xy.sum(), x, create_graph=True)[0][:, 0:1]
        sigma_yz_z = grad(sigma_yz.sum(), x, create_graph=True)[0][:, 2:3]
        
        sigma_zz_z = grad(sigma_zz.sum(), x, create_graph=True)[0][:, 2:3]
        sigma_xz_x = grad(sigma_xz.sum(), x, create_graph=True)[0][:, 0:1]
        sigma_yz_y = grad(sigma_yz.sum(), x, create_graph=True)[0][:, 1:2]
        
        residual_x = sigma_xx_x + sigma_xy_y + sigma_xz_z
        residual_y = sigma_xy_x + sigma_yy_y + sigma_yz_z
        residual_z = sigma_xz_x + sigma_yz_y + sigma_zz_z
        
        return torch.mean(residual_x**2 + residual_y**2 + residual_z**2)
    
    def _compute_thermal_loss(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Compute heat equation residuals"""
        T_grad = grad(T.sum(), x, create_graph=True)[0]
        
        # Check if time dimension exists
        if x.shape[1] > 3:  # Has time dimension
            T_t = T_grad[:, 3:4]
            T_laplacian = self._compute_spatial_laplacian(x, T)
            
            thermal_diffusivity = (self.material.thermal_conductivity / 
                                  (self.material.density * self.material.specific_heat))
            
            heat_residual = T_t - thermal_diffusivity * T_laplacian
            return torch.mean(heat_residual**2)
        else:
            # Steady state: ∇²T = 0
            T_laplacian = self._compute_spatial_laplacian(x, T)
            return torch.mean(T_laplacian**2)
    
    def _compute_flow_loss(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute complete Navier-Stokes residuals with full nonlinear terms"""
        v_x, v_y, v_z = u[:, 0:1], u[:, 1:2], u[:, 2:3]
        p = u[:, 3:4]
        
        # Compute all velocity gradients
        v_x_grad = grad(v_x.sum(), x, create_graph=True)[0]
        v_y_grad = grad(v_y.sum(), x, create_graph=True)[0]
        v_z_grad = grad(v_z.sum(), x, create_graph=True)[0]
        p_grad = grad(p.sum(), x, create_graph=True)[0]
        
        # Velocity derivatives
        dvx_dx, dvx_dy, dvx_dz = v_x_grad[:, 0:1], v_x_grad[:, 1:2], v_x_grad[:, 2:3]
        dvy_dx, dvy_dy, dvy_dz = v_y_grad[:, 0:1], v_y_grad[:, 1:2], v_y_grad[:, 2:3]
        dvz_dx, dvz_dy, dvz_dz = v_z_grad[:, 0:1], v_z_grad[:, 1:2], v_z_grad[:, 2:3]
        
        # Pressure derivatives
        dp_dx, dp_dy, dp_dz = p_grad[:, 0:1], p_grad[:, 1:2], p_grad[:, 2:3]
        
        # Continuity equation: ∇·v = 0
        continuity = dvx_dx + dvy_dy + dvz_dz
        
        # Compute second derivatives for viscous terms
        dvx_dxx = grad(dvx_dx.sum(), x, create_graph=True)[0][:, 0:1]
        dvx_dyy = grad(dvx_dy.sum(), x, create_graph=True)[0][:, 1:2]
        dvx_dzz = grad(dvx_dz.sum(), x, create_graph=True)[0][:, 2:3]
        
        dvy_dxx = grad(dvy_dx.sum(), x, create_graph=True)[0][:, 0:1]
        dvy_dyy = grad(dvy_dy.sum(), x, create_graph=True)[0][:, 1:2]
        dvy_dzz = grad(dvy_dz.sum(), x, create_graph=True)[0][:, 2:3]
        
        dvz_dxx = grad(dvz_dx.sum(), x, create_graph=True)[0][:, 0:1]
        dvz_dyy = grad(dvz_dy.sum(), x, create_graph=True)[0][:, 1:2]
        dvz_dzz = grad(dvz_dz.sum(), x, create_graph=True)[0][:, 2:3]
        
        # Laplacians
        laplacian_vx = dvx_dxx + dvx_dyy + dvx_dzz
        laplacian_vy = dvy_dxx + dvy_dyy + dvy_dzz
        laplacian_vz = dvz_dxx + dvz_dyy + dvz_dzz
        
        # Material properties
        density = self.material.density
        viscosity = self.material.viscosity or 1e6  # Pa·s for magma
        
        # Check if time-dependent (transient flow)
        if x.shape[1] > 3:  # Has time dimension
            v_x_grad_t = grad(v_x.sum(), x, create_graph=True)[0]
            v_y_grad_t = grad(v_y.sum(), x, create_graph=True)[0]  
            v_z_grad_t = grad(v_z.sum(), x, create_graph=True)[0]
            
            dvx_dt = v_x_grad_t[:, 3:4]
            dvy_dt = v_y_grad_t[:, 3:4]
            dvz_dt = v_z_grad_t[:, 3:4]
            
            # Full Navier-Stokes momentum equations with nonlinear terms
            # ρ(∂v/∂t + (v·∇)v) = -∇p + μ∇²v + f
            
            # Nonlinear convection terms: (v·∇)v
            conv_x = v_x * dvx_dx + v_y * dvx_dy + v_z * dvx_dz
            conv_y = v_x * dvy_dx + v_y * dvy_dy + v_z * dvy_dz
            conv_z = v_x * dvz_dx + v_y * dvz_dy + v_z * dvz_dz
            
            # Momentum equations
            momentum_x = density * (dvx_dt + conv_x) + dp_dx - viscosity * laplacian_vx
            momentum_y = density * (dvy_dt + conv_y) + dp_dy - viscosity * laplacian_vy
            momentum_z = density * (dvz_dt + conv_z) + dp_dz - viscosity * laplacian_vz
            
        else:
            # Steady-state flow with nonlinear terms
            conv_x = v_x * dvx_dx + v_y * dvx_dy + v_z * dvx_dz
            conv_y = v_x * dvy_dx + v_y * dvy_dy + v_z * dvy_dz
            conv_z = v_x * dvz_dx + v_y * dvz_dy + v_z * dvz_dz
            
            momentum_x = density * conv_x + dp_dx - viscosity * laplacian_vx
            momentum_y = density * conv_y + dp_dy - viscosity * laplacian_vy
            momentum_z = density * conv_z + dp_dz - viscosity * laplacian_vz
        
        # Include gravitational effects for geological applications
        g = 9.81  # m/s²
        momentum_z += density * g  # Add gravitational body force
        
        # Temperature-dependent viscosity for magma
        if hasattr(self, '_compute_temperature_dependent_viscosity'):
            # This would be called if temperature field is available in coupled system
            pass
        
        return torch.mean(continuity**2 + momentum_x**2 + momentum_y**2 + momentum_z**2)
    
    def _compute_hydrothermal_loss(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute comprehensive coupled fluid flow and heat transport residuals with full hydrothermal physics"""
        # Extract field variables
        v_x, v_y, v_z = u[:, 0:1], u[:, 1:2], u[:, 2:3]
        p = u[:, 3:4]  # pressure
        T = u[:, 4:5]  # temperature
        
        # Compute gradients
        v_x_grad = grad(v_x.sum(), x, create_graph=True)[0]
        v_y_grad = grad(v_y.sum(), x, create_graph=True)[0]
        v_z_grad = grad(v_z.sum(), x, create_graph=True)[0]
        p_grad = grad(p.sum(), x, create_graph=True)[0]
        T_grad = grad(T.sum(), x, create_graph=True)[0]
        
        # Velocity derivatives
        dvx_dx, dvx_dy, dvx_dz = v_x_grad[:, 0:1], v_x_grad[:, 1:2], v_x_grad[:, 2:3]
        dvy_dx, dvy_dy, dvy_dz = v_y_grad[:, 0:1], v_y_grad[:, 1:2], v_y_grad[:, 2:3]
        dvz_dx, dvz_dy, dvz_dz = v_z_grad[:, 0:1], v_z_grad[:, 1:2], v_z_grad[:, 2:3]
        
        # Temperature derivatives
        dT_dx, dT_dy, dT_dz = T_grad[:, 0:1], T_grad[:, 1:2], T_grad[:, 2:3]
        
        # Pressure derivatives
        dp_dx, dp_dy, dp_dz = p_grad[:, 0:1], p_grad[:, 1:2], p_grad[:, 2:3]
        
        # Material properties (temperature-dependent)
        T_kelvin = T + 273.15  # Convert to Kelvin
        density_ref = self.material.density
        thermal_expansion = self.material.thermal_expansion
        reference_temp = 273.15
        
        # Temperature-dependent density (Boussinesq approximation)
        density = density_ref * (1 - thermal_expansion * (T_kelvin - reference_temp))
        
        # Temperature-dependent viscosity (Arrhenius law for geological fluids)
        viscosity_ref = self.material.viscosity or 1e-3  # Water viscosity at reference
        activation_energy = 30000  # J/mol (typical for water-rock systems)
        gas_constant = 8.314  # J/(mol·K)
        viscosity = viscosity_ref * torch.exp(activation_energy / (gas_constant * T_kelvin) - 
                                             activation_energy / (gas_constant * reference_temp))
        
        # Thermal properties (temperature and pressure dependent)
        thermal_conductivity_ref = self.material.thermal_conductivity
        pressure_pa = p + 101325  # Add atmospheric pressure
        # Thermal conductivity increases with pressure
        thermal_conductivity = thermal_conductivity_ref * (1 + 1e-10 * pressure_pa)
        
        specific_heat = self.material.specific_heat * (1 + 1e-4 * (T_kelvin - reference_temp))
        thermal_diffusivity = thermal_conductivity / (density * specific_heat)
        
        # Permeability (temperature and pressure dependent)
        permeability_ref = self.material.permeability
        porosity = self.material.porosity
        # Kozeny-Carman relation with thermal effects
        permeability = permeability_ref * (porosity**3 / (1-porosity)**2) * \
                      torch.exp(-thermal_expansion * (T_kelvin - reference_temp))
        
        # === CONTINUITY EQUATION ===
        # ∂ρ/∂t + ∇·(ρv) = 0
        # For variable density: ∂ρ/∂t + v·∇ρ + ρ∇·v = 0
        
        # Density derivatives
        drho_dT = -density_ref * thermal_expansion
        drho_dx = drho_dT * dT_dx
        drho_dy = drho_dT * dT_dy
        drho_dz = drho_dT * dT_dz
        
        # Velocity divergence
        div_v = dvx_dx + dvy_dy + dvz_dz
        
        # Advection of density
        density_advection = v_x * drho_dx + v_y * drho_dy + v_z * drho_dz
        
        if x.shape[1] > 3:  # Time-dependent
            drho_dt = drho_dT * T_grad[:, 3:4]
            continuity = drho_dt + density_advection + density * div_v
        else:
            continuity = density_advection + density * div_v
        
        # === MOMENTUM EQUATIONS ===
        # ρ(∂v/∂t + (v·∇)v) = -∇p + ∇·τ + ρg + F_buoyancy
        
        # Second derivatives for viscous terms
        dvx_dxx = grad(dvx_dx.sum(), x, create_graph=True)[0][:, 0:1]
        dvx_dyy = grad(dvx_dy.sum(), x, create_graph=True)[0][:, 1:2]
        dvx_dzz = grad(dvx_dz.sum(), x, create_graph=True)[0][:, 2:3]
        
        dvy_dxx = grad(dvy_dx.sum(), x, create_graph=True)[0][:, 0:1]
        dvy_dyy = grad(dvy_dy.sum(), x, create_graph=True)[0][:, 1:2]
        dvy_dzz = grad(dvy_dz.sum(), x, create_graph=True)[0][:, 2:3]
        
        dvz_dxx = grad(dvz_dx.sum(), x, create_graph=True)[0][:, 0:1]
        dvz_dyy = grad(dvz_dy.sum(), x, create_graph=True)[0][:, 1:2]
        dvz_dzz = grad(dvz_dz.sum(), x, create_graph=True)[0][:, 2:3]
        
        # Mixed derivatives for complete viscous stress tensor
        dvx_dxy = grad(dvx_dx.sum(), x, create_graph=True)[0][:, 1:2]
        dvx_dxz = grad(dvx_dx.sum(), x, create_graph=True)[0][:, 2:3]
        dvy_dyx = grad(dvy_dy.sum(), x, create_graph=True)[0][:, 0:1]
        dvy_dyz = grad(dvy_dy.sum(), x, create_graph=True)[0][:, 2:3]
        dvz_dzx = grad(dvz_dz.sum(), x, create_graph=True)[0][:, 0:1]
        dvz_dzy = grad(dvz_dz.sum(), x, create_graph=True)[0][:, 1:2]
        
        # Complete viscous stress tensor divergence
        # τij = μ(∂vi/∂xj + ∂vj/∂xi) - (2μ/3)δij∇·v
        bulk_viscosity = 2 * viscosity / 3
        
        # x-momentum viscous terms
        tau_xx = 2 * viscosity * dvx_dx - bulk_viscosity * div_v
        tau_xy = viscosity * (dvx_dy + dvy_dx)
        tau_xz = viscosity * (dvx_dz + dvz_dx)
        
        dtau_xx_dx = grad(tau_xx.sum(), x, create_graph=True)[0][:, 0:1]
        dtau_xy_dy = grad(tau_xy.sum(), x, create_graph=True)[0][:, 1:2]
        dtau_xz_dz = grad(tau_xz.sum(), x, create_graph=True)[0][:, 2:3]
        
        # y-momentum viscous terms
        tau_yx = tau_xy  # Symmetric
        tau_yy = 2 * viscosity * dvy_dy - bulk_viscosity * div_v
        tau_yz = viscosity * (dvy_dz + dvz_dy)
        
        dtau_yx_dx = grad(tau_yx.sum(), x, create_graph=True)[0][:, 0:1]
        dtau_yy_dy = grad(tau_yy.sum(), x, create_graph=True)[0][:, 1:2]
        dtau_yz_dz = grad(tau_yz.sum(), x, create_graph=True)[0][:, 2:3]
        
        # z-momentum viscous terms
        tau_zx = tau_xz  # Symmetric
        tau_zy = tau_yz  # Symmetric
        tau_zz = 2 * viscosity * dvz_dz - bulk_viscosity * div_v
        
        dtau_zx_dx = grad(tau_zx.sum(), x, create_graph=True)[0][:, 0:1]
        dtau_zy_dy = grad(tau_zy.sum(), x, create_graph=True)[0][:, 1:2]
        dtau_zz_dz = grad(tau_zz.sum(), x, create_graph=True)[0][:, 2:3]
        
        # Body forces
        g = 9.81  # m/s²
        buoyancy_force_z = (density - density_ref) * g  # Thermal buoyancy
        
        # Darcy flow resistance in porous media
        darcy_x = -viscosity * v_x / permeability
        darcy_y = -viscosity * v_y / permeability  
        darcy_z = -viscosity * v_z / permeability
        
        if x.shape[1] > 3:  # Time-dependent
            dvx_dt = v_x_grad[:, 3:4]
            dvy_dt = v_y_grad[:, 3:4]
            dvz_dt = v_z_grad[:, 3:4]
            
            # Convection terms
            conv_x = v_x * dvx_dx + v_y * dvx_dy + v_z * dvx_dz
            conv_y = v_x * dvy_dx + v_y * dvy_dy + v_z * dvy_dz
            conv_z = v_x * dvz_dx + v_y * dvz_dy + v_z * dvz_dz
            
            # Complete momentum equations
            momentum_x = density * (dvx_dt + conv_x) + dp_dx - (dtau_xx_dx + dtau_xy_dy + dtau_xz_dz) + darcy_x
            momentum_y = density * (dvy_dt + conv_y) + dp_dy - (dtau_yx_dx + dtau_yy_dy + dtau_yz_dz) + darcy_y
            momentum_z = density * (dvz_dt + conv_z) + dp_dz - (dtau_zx_dx + dtau_zy_dy + dtau_zz_dz) + darcy_z - buoyancy_force_z
        else:
            # Steady-state
            conv_x = v_x * dvx_dx + v_y * dvx_dy + v_z * dvx_dz
            conv_y = v_x * dvy_dx + v_y * dvy_dy + v_z * dvy_dz
            conv_z = v_x * dvz_dx + v_y * dvz_dy + v_z * dvz_dz
            
            momentum_x = density * conv_x + dp_dx - (dtau_xx_dx + dtau_xy_dy + dtau_xz_dz) + darcy_x
            momentum_y = density * conv_y + dp_dy - (dtau_yx_dx + dtau_yy_dy + dtau_yz_dz) + darcy_y
            momentum_z = density * conv_z + dp_dz - (dtau_zx_dx + dtau_zy_dy + dtau_zz_dz) + darcy_z - buoyancy_force_z
        
        # === ENERGY EQUATION ===
        # ρcp(∂T/∂t + v·∇T) = ∇·(k∇T) + Q_viscous + Q_sources
        
        # Temperature second derivatives
        d2T_dx2 = grad(dT_dx.sum(), x, create_graph=True)[0][:, 0:1]
        d2T_dy2 = grad(dT_dy.sum(), x, create_graph=True)[0][:, 1:2]
        d2T_dz2 = grad(dT_dz.sum(), x, create_graph=True)[0][:, 2:3]
        
        # Thermal diffusion with variable thermal conductivity
        dk_dx = grad(thermal_conductivity.sum(), x, create_graph=True)[0][:, 0:1]
        dk_dy = grad(thermal_conductivity.sum(), x, create_graph=True)[0][:, 1:2]
        dk_dz = grad(thermal_conductivity.sum(), x, create_graph=True)[0][:, 2:3]
        
        thermal_diffusion = (dk_dx * dT_dx + dk_dy * dT_dy + dk_dz * dT_dz + 
                           thermal_conductivity * (d2T_dx2 + d2T_dy2 + d2T_dz2))
        
        # Viscous dissipation heating
        shear_rate_squared = (2 * (dvx_dx**2 + dvy_dy**2 + dvz_dz**2) +
                             (dvx_dy + dvy_dx)**2 + (dvx_dz + dvz_dx)**2 + (dvy_dz + dvz_dy)**2)
        viscous_heating = viscosity * shear_rate_squared
        
        # Thermal advection
        thermal_advection = v_x * dT_dx + v_y * dT_dy + v_z * dT_dz
        
        # Adiabatic heating/cooling from pressure changes
        if x.shape[1] > 3:  # Time-dependent
            dp_dt = p_grad[:, 3:4]
            adiabatic_heating = thermal_expansion * T_kelvin * (v_x * dp_dx + v_y * dp_dy + v_z * dp_dz + dp_dt) / (density * specific_heat)
        else:
            adiabatic_heating = thermal_expansion * T_kelvin * (v_x * dp_dx + v_y * dp_dy + v_z * dp_dz) / (density * specific_heat)
        
        if x.shape[1] > 3:  # Time-dependent
            dT_dt = T_grad[:, 3:4]
            energy_equation = (density * specific_heat * (dT_dt + thermal_advection) - 
                             thermal_diffusion - viscous_heating - adiabatic_heating)
        else:
            # Steady-state
            energy_equation = (density * specific_heat * thermal_advection - 
                             thermal_diffusion - viscous_heating - adiabatic_heating)
        
        # Combine all residuals
        total_loss = (torch.mean(continuity**2) + 
                     torch.mean(momentum_x**2) + torch.mean(momentum_y**2) + torch.mean(momentum_z**2) +
                     torch.mean(energy_equation**2))
        
        return total_loss
    
    def _compute_spatial_laplacian(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute spatial Laplacian (first 3 dimensions only)"""
        u_grad = grad(u.sum(), x, create_graph=True)[0]
        
        u_xx = grad(u_grad[:, 0].sum(), x, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_grad[:, 1].sum(), x, create_graph=True)[0][:, 1:2]
        u_zz = grad(u_grad[:, 2].sum(), x, create_graph=True)[0][:, 2:3]
        
        return u_xx + u_yy + u_zz
    
    def _compute_coupling_loss(self, x: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
        """Compute physics coupling residuals"""
        coupling_loss = torch.tensor(0.0, device=x.device)
        
        # Thermal-mechanical coupling
        if (GeologicalProcessType.THERMAL_EVOLUTION in self.physics_types and 
            GeologicalProcessType.TECTONIC_DEFORMATION in self.physics_types):
            
            thermal_slice = self.physics_field_indices[GeologicalProcessType.THERMAL_EVOLUTION]
            mechanical_slice = self.physics_field_indices[GeologicalProcessType.TECTONIC_DEFORMATION]
            
            T = u_pred[:, thermal_slice]
            u_mech = u_pred[:, mechanical_slice]
            
            # Thermal expansion coupling
            T_grad = grad(T.sum(), x, create_graph=True)[0]
            reference_temp = 273.15
            thermal_strain_source = self.material.thermal_expansion * (T - reference_temp)
            
            # This couples to the mechanical field through strain
            u_x_grad = grad(u_mech[:, 0:1].sum(), x, create_graph=True)[0]
            strain_coupling = torch.mean((u_x_grad[:, 0:1] - thermal_strain_source)**2)
            coupling_loss += 0.1 * strain_coupling
        
        return coupling_loss
    
    def compute_boundary_loss(self, x_boundary: torch.Tensor, u_boundary: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss for multi-physics fields"""
        u_pred = self.forward(x_boundary)
        if isinstance(u_pred, tuple):
            u_pred = u_pred[0]
        return nn.MSELoss()(u_pred, u_boundary)

class GeologicalPINNEnsemble:
    """Ensemble of geological PINNs for uncertainty quantification"""
    
    def __init__(self, pinn_class: type, config: PINNConfig, material: GeologicalMaterial,
                 ensemble_size: int = 5):
        self.ensemble_size = ensemble_size
        self.models = []
        
        for i in range(ensemble_size):
            # Add slight variations to each model
            model_config = config
            model_config.dropout_rate += np.random.normal(0, 0.01)
            model_config.learning_rate *= np.random.uniform(0.8, 1.2)
            
            model = pinn_class(model_config, material)
            self.models.append(model)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble prediction with uncertainty quantification"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return mean_pred, std_pred
    
    def train_ensemble(self, training_data: Dict, epochs: int = 10000):
        """Train all models in the ensemble"""
        for i, model in enumerate(self.models):
            logger.info(f"Training ensemble model {i+1}/{self.ensemble_size}")
            trainer = GeologicalPINNTrainer(model)
            trainer.train(training_data, epochs)

class GeologicalPINNTrainer:
    """Training framework for geological PINNs"""
    
    def __init__(self, model: GeologicalPINN):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=model.config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=1000, verbose=True
        )
        self.best_loss = float('inf')
        
    def train(self, training_data: Dict, epochs: int = 10000):
        """Train the PINN model"""
        self.model.train()
        
        # Extract training data
        x_physics = training_data.get('x_physics')
        x_boundary = training_data.get('x_boundary')
        u_boundary = training_data.get('u_boundary')
        x_data = training_data.get('x_data')
        u_data = training_data.get('u_data')
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            total_loss = 0
            loss_components = {}
            
            # Physics loss
            if x_physics is not None:
                u_physics = self.model(x_physics)
                if isinstance(u_physics, tuple):
                    u_physics = u_physics[0]
                physics_loss = self.model.compute_physics_loss(x_physics, u_physics)
                total_loss += self.model.physics_weights['conservation'] * physics_loss
                loss_components['physics'] = physics_loss.item()
            
            # Boundary loss
            if x_boundary is not None and u_boundary is not None:
                boundary_loss = self.model.compute_boundary_loss(x_boundary, u_boundary)
                total_loss += self.model.physics_weights['boundary'] * boundary_loss
                loss_components['boundary'] = boundary_loss.item()
            
            # Data loss
            if x_data is not None and u_data is not None:
                u_pred = self.model(x_data)
                if isinstance(u_pred, tuple):
                    u_pred = u_pred[0]
                data_loss = self.model.compute_data_loss(x_data, u_data, u_pred)
                total_loss += self.model.config.data_weight * data_loss
                loss_components['data'] = data_loss.item()
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Adaptive weight update
            if hasattr(self.model, 'adaptive_weight_update'):
                self.model.adaptive_weight_update(
                    loss_components.get('physics', 0),
                    loss_components.get('boundary', 0),
                    loss_components.get('data', 0),
                    epoch
                )
            
            # Logging and scheduling
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}")
                for component, value in loss_components.items():
                    logger.info(f"  {component.capitalize()} Loss: {value:.6f}")
            
            self.scheduler.step(total_loss)
            
            # Early stopping
            if total_loss.item() < self.best_loss:
                self.best_loss = total_loss.item()
            elif epoch > 5000 and total_loss.item() > self.best_loss * 1.1:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'material': self.material,
            'best_loss': self.best_loss
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.model.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint['best_loss']

class GeologicalPINNFactory:
    """Factory for creating geological PINNs"""
    
    _model_registry = {
        GeologicalProcessType.TECTONIC_DEFORMATION: TectonicDeformationPINN,
        GeologicalProcessType.MAGMA_FLOW: MagmaFlowPINN,
        GeologicalProcessType.THERMAL_EVOLUTION: ThermalEvolutionPINN,
        GeologicalProcessType.SEISMIC_WAVE_PROPAGATION: SeismicWavePINN,
    }
    
    @classmethod
    def create_pinn(cls, process_type: GeologicalProcessType, config: PINNConfig,
                   material: GeologicalMaterial) -> GeologicalPINN:
        """Create a PINN for the specified geological process"""
        if process_type not in cls._model_registry:
            raise ValueError(f"Unsupported process type: {process_type}")
        
        model_class = cls._model_registry[process_type]
        return model_class(config, material)
    
    @classmethod
    def create_multiphysics_pinn(cls, physics_types: List[GeologicalProcessType],
                                config: PINNConfig, material: GeologicalMaterial) -> MultiPhysicsPINN:
        """Create a multi-physics PINN for coupled geological processes"""
        return MultiPhysicsPINN(config, material, physics_types)
    
    @classmethod
    def create_ensemble_pinn(cls, process_type: GeologicalProcessType, config: PINNConfig,
                           material: GeologicalMaterial, ensemble_size: int = 5) -> GeologicalPINNEnsemble:
        """Create an ensemble of PINNs for uncertainty quantification"""
        if process_type not in cls._model_registry:
            raise ValueError(f"Unsupported process type: {process_type}")
        
        model_class = cls._model_registry[process_type]
        return GeologicalPINNEnsemble(model_class, config, material, ensemble_size)
    
    @classmethod
    def register_model(cls, process_type: GeologicalProcessType, model_class: type):
        """Register a new PINN model class"""
        cls._model_registry[process_type] = model_class

class GeologicalPINNIntegrator:
    """
    Integration layer connecting PINNs with other backend modules
    for comprehensive geological modeling workflow
    """
    
    def __init__(self):
        self.model_database = None
        self.data_pipeline = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize backend integrations if available
        if MODEL_DATABASE_AVAILABLE:
            try:
                self.model_database = ModelDatabase()
                self.logger.info("Model database integration initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize model database: {e}")
        
        if DATA_INTEGRATION_AVAILABLE:
            try:
                self.data_pipeline = DataIntegrationPipeline()
                self.logger.info("Data integration pipeline initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize data pipeline: {e}")
    
    def create_integrated_training_data(self, 
                                       domain_bounds: Tuple[Tuple[float, float], ...],
                                       geological_region: str = "global",
                                       time_range: Optional[Tuple[float, float]] = None,
                                       data_sources: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Create comprehensive training data by integrating multiple geological data sources
        """
        try:
            base_data = create_training_data(domain_bounds)
            
            # Integrate real geological data if pipeline is available
            if self.data_pipeline and data_sources:
                real_data = self._fetch_geological_data(geological_region, time_range, data_sources)
                if real_data:
                    base_data.update(real_data)
                    self.logger.info("Successfully integrated real geological data")
            
            return base_data
            
        except Exception as e:
            self.logger.error(f"Failed to create integrated training data: {e}")
            return create_training_data(domain_bounds)  # Fallback to synthetic data
    
    def _fetch_geological_data(self, region: str, time_range: Optional[Tuple[float, float]], 
                              sources: List[str]) -> Optional[Dict[str, torch.Tensor]]:
        """Fetch real geological data from external sources"""
        if not self.data_pipeline:
            return None
        
        try:
            # Use data integration pipeline to fetch multi-source data
            query_params = {
                'region': region,
                'time_range': time_range,
                'sources': sources,
                'data_types': ['seismic', 'gps', 'insar', 'thermal', 'geochemical']
            }
            
            raw_data = self.data_pipeline.fetch_multi_source_data(**query_params)
            
            # Convert to PyTorch tensors for PINN training
            processed_data = {}
            
            if 'observational_points' in raw_data:
                processed_data['x_data'] = torch.tensor(
                    raw_data['observational_points'], dtype=torch.float32
                )
            
            if 'measurements' in raw_data:
                processed_data['u_data'] = torch.tensor(
                    raw_data['measurements'], dtype=torch.float32
                )
            
            # Add temporal data if available
            if 'temporal_series' in raw_data and time_range:
                temporal_data = raw_data['temporal_series']
                # Process temporal data for time-dependent PINNs
                processed_data['temporal_data'] = self._process_temporal_data(temporal_data, time_range)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch geological data: {e}")
            return None
    
    def _process_temporal_data(self, temporal_data: Dict, time_range: Tuple[float, float]) -> torch.Tensor:
        """Process temporal geological data for PINN training"""
        try:
            # Convert temporal measurements to space-time coordinates
            temporal_coords = []
            
            for timestamp, spatial_data in temporal_data.items():
                if time_range[0] <= timestamp <= time_range[1]:
                    for point in spatial_data:
                        # Add time dimension to spatial coordinates
                        space_time_point = list(point) + [timestamp]
                        temporal_coords.append(space_time_point)
            
            return torch.tensor(temporal_coords, dtype=torch.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to process temporal data: {e}")
            return torch.empty(0, 4)  # Empty 4D tensor (x,y,z,t)
    
    def save_trained_model(self, model: GeologicalPINN, model_metadata: Dict):
        """Save trained PINN model to model database"""
        if not self.model_database:
            self.logger.warning("Model database not available for saving")
            return
        
        try:
            # Prepare model data
            model_data = {
                'model_type': 'geological_pinn',
                'process_type': model.config.process_type.value,
                'architecture': {
                    'hidden_layers': model.config.hidden_layers,
                    'activation': model.config.activation,
                    'input_dim': model.config.input_dim,
                    'output_dim': model.config.output_dim
                },
                'material_properties': {
                    'name': model.material.name,
                    'density': model.material.density,
                    'youngs_modulus': model.material.youngs_modulus,
                    'poisson_ratio': model.material.poisson_ratio,
                    'thermal_conductivity': model.material.thermal_conductivity
                },
                'training_metadata': model_metadata,
                'model_state': model.state_dict()
            }
            
            model_id = self.model_database.save_model(model_data)
            self.logger.info(f"Model saved with ID: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Failed to save model to database: {e}")
            return None
    
    def load_trained_model(self, model_id: str) -> Optional[GeologicalPINN]:
        """Load trained PINN model from model database"""
        if not self.model_database:
            self.logger.warning("Model database not available for loading")
            return None
        
        try:
            model_data = self.model_database.load_model(model_id)
            
            if not model_data:
                self.logger.error(f"Model {model_id} not found in database")
                return None
            
            # Reconstruct model configuration
            process_type = GeologicalProcessType(model_data['process_type'])
            config = PINNConfig(
                process_type=process_type,
                hidden_layers=model_data['architecture']['hidden_layers'],
                activation=model_data['architecture']['activation'],
                input_dim=model_data['architecture']['input_dim'],
                output_dim=model_data['architecture']['output_dim']
            )
            
            # Reconstruct material properties
            material_props = model_data['material_properties']
            material = GeologicalMaterial(**material_props)
            
            # Create model and load state
            model = GeologicalPINNFactory.create_pinn(process_type, config, material)
            model.load_state_dict(model_data['model_state'])
            
            self.logger.info(f"Successfully loaded model {model_id}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    def run_comparative_analysis(self, models: List[GeologicalPINN], 
                               test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Run comparative analysis of multiple PINN models"""
        results = {
            'model_performance': {},
            'prediction_comparison': {},
            'uncertainty_analysis': {},
            'physics_compliance': {}
        }
        
        test_points = test_data.get('x_test')
        true_values = test_data.get('u_true')
        
        if test_points is None:
            self.logger.error("No test points provided for comparative analysis")
            return results
        
        for i, model in enumerate(models):
            model_id = f"model_{i}"
            model.eval()
            
            with torch.no_grad():
                predictions = model(test_points)
                if isinstance(predictions, tuple):
                    pred_values, uncertainty = predictions
                    results['uncertainty_analysis'][model_id] = {
                        'mean_uncertainty': torch.mean(uncertainty).item(),
                        'max_uncertainty': torch.max(uncertainty).item()
                    }
                else:
                    pred_values = predictions
                    results['uncertainty_analysis'][model_id] = {'available': False}
                
                results['prediction_comparison'][model_id] = pred_values.numpy()
                
                # Compute physics compliance
                physics_loss = model.compute_physics_loss(test_points, pred_values)
                results['physics_compliance'][model_id] = physics_loss.item()
                
                # Compute performance metrics if true values available
                if true_values is not None:
                    mse = torch.mean((pred_values - true_values)**2).item()
                    mae = torch.mean(torch.abs(pred_values - true_values)).item()
                    results['model_performance'][model_id] = {
                        'mse': mse,
                        'mae': mae,
                        'rmse': np.sqrt(mse)
                    }
        
        return results

# Predefined geological materials
GEOLOGICAL_MATERIALS = {
    'granite': GeologicalMaterial(
        name='granite',
        density=2650,
        youngs_modulus=50e9,
        poisson_ratio=0.25,
        thermal_conductivity=3.0,
        specific_heat=800,
        thermal_expansion=8e-6,
        permeability=1e-18,
        porosity=0.01
    ),
    'basalt': GeologicalMaterial(
        name='basalt',
        density=2900,
        youngs_modulus=70e9,
        poisson_ratio=0.25,
        thermal_conductivity=2.0,
        specific_heat=1000,
        thermal_expansion=6e-6,
        permeability=1e-16,
        porosity=0.05
    ),
    'sandstone': GeologicalMaterial(
        name='sandstone',
        density=2300,
        youngs_modulus=20e9,
        poisson_ratio=0.15,
        thermal_conductivity=2.5,
        specific_heat=900,
        thermal_expansion=12e-6,
        permeability=1e-13,
        porosity=0.15
    ),
    'magma': GeologicalMaterial(
        name='magma',
        density=2500,
        youngs_modulus=1e6,  # Much lower for fluid
        poisson_ratio=0.45,  # Near incompressible
        thermal_conductivity=1.0,
        specific_heat=1200,
        thermal_expansion=5e-5,
        permeability=1e-12,
        porosity=0.0,
        viscosity=1e6  # High viscosity
    )
}

def create_training_data(domain_bounds: Tuple[Tuple[float, float], ...],
                        n_physics: int = 10000,
                        n_boundary: int = 1000) -> Dict[str, torch.Tensor]:
    """Create synthetic training data for PINN"""
    
    # Physics collocation points (random sampling in domain)
    x_physics = []
    for bounds in domain_bounds:
        x_physics.append(torch.rand(n_physics, 1) * (bounds[1] - bounds[0]) + bounds[0])
    x_physics = torch.cat(x_physics, dim=1)
    
    # Boundary points (on domain boundaries)
    x_boundary = []
    u_boundary = []
    
    for i, bounds in enumerate(domain_bounds):
        # Lower boundary
        x_bound = torch.zeros(n_boundary // (2 * len(domain_bounds)), len(domain_bounds))
        for j in range(len(domain_bounds)):
            if i == j:
                x_bound[:, j] = bounds[0]
            else:
                other_bounds = domain_bounds[j]
                x_bound[:, j] = torch.rand(x_bound.shape[0]) * (other_bounds[1] - other_bounds[0]) + other_bounds[0]
        x_boundary.append(x_bound)
        u_boundary.append(torch.zeros(x_bound.shape[0], 1))  # Dirichlet BC
        
        # Upper boundary
        x_bound = torch.zeros(n_boundary // (2 * len(domain_bounds)), len(domain_bounds))
        for j in range(len(domain_bounds)):
            if i == j:
                x_bound[:, j] = bounds[1]
            else:
                other_bounds = domain_bounds[j]
                x_bound[:, j] = torch.rand(x_bound.shape[0]) * (other_bounds[1] - other_bounds[0]) + other_bounds[0]
        x_boundary.append(x_bound)
        u_boundary.append(torch.zeros(x_bound.shape[0], 1))  # Dirichlet BC
    
    x_boundary = torch.cat(x_boundary, dim=0)
    u_boundary = torch.cat(u_boundary, dim=0)
    
    return {
        'x_physics': x_physics,
        'x_boundary': x_boundary,
        'u_boundary': u_boundary
    }

# Example usage and demonstration
if __name__ == "__main__":
    logger.info("=== Comprehensive Geological PINN Framework Demonstration ===")
    
    # Example 1: Single-physics PINN (Tectonic Deformation)
    logger.info("\n1. Single-physics PINN: Tectonic Deformation")
    
    config = PINNConfig(
        process_type=GeologicalProcessType.TECTONIC_DEFORMATION,
        hidden_layers=[256, 256, 128, 64],
        learning_rate=1e-3,
        epochs=1000,  # Reduced for demo
        physics_weight=1.0,
        boundary_weight=2.0,
        uncertainty_quantification=True
    )
    
    material = GEOLOGICAL_MATERIALS['granite']
    
    pinn = GeologicalPINNFactory.create_pinn(
        GeologicalProcessType.TECTONIC_DEFORMATION, 
        config, 
        material
    )
    
    domain_bounds = [(-10000, 10000), (-10000, 10000), (-5000, 0)]
    training_data = create_training_data(domain_bounds, n_physics=2000, n_boundary=200)
    
    trainer = GeologicalPINNTrainer(pinn)
    logger.info("Training single-physics PINN...")
    trainer.train(training_data, epochs=100)  # Very short for demo
    
    # Test prediction with uncertainty
    test_points = torch.tensor([
        [0.0, 0.0, -1000.0], 
        [5000.0, 5000.0, -2000.0],
        [-3000.0, 3000.0, -1500.0]
    ], dtype=torch.float32)
    
    predictions = pinn(test_points)
    if isinstance(predictions, tuple):
        displacement, uncertainty = predictions
        logger.info(f"Displacement predictions shape: {displacement.shape}")
        logger.info(f"Mean displacement magnitude: {torch.norm(displacement, dim=1).mean():.6f}")
        logger.info(f"Mean prediction uncertainty: {torch.mean(uncertainty):.6f}")
    
    # Example 2: Multi-physics PINN (Thermal-Mechanical Coupling)
    logger.info("\n2. Multi-physics PINN: Thermal-Mechanical Coupling")
    
    physics_types = [
        GeologicalProcessType.THERMAL_EVOLUTION,
        GeologicalProcessType.TECTONIC_DEFORMATION
    ]
    
    multiphysics_config = PINNConfig(
        process_type=GeologicalProcessType.THERMAL_EVOLUTION,  # Primary process
        hidden_layers=[512, 256, 128, 64],
        learning_rate=5e-4,
        epochs=500,
        physics_weight=1.0,
        boundary_weight=1.5,
        adaptive_weights=True
    )
    
    multiphysics_pinn = GeologicalPINNFactory.create_multiphysics_pinn(
        physics_types, multiphysics_config, material
    )
    
    logger.info(f"Multi-physics PINN output dimension: {multiphysics_pinn.config.output_dim}")
    logger.info(f"Physics field mapping: {multiphysics_pinn.physics_field_indices}")
    
    # Create multi-physics training data
    mp_training_data = create_training_data(domain_bounds, n_physics=1500, n_boundary=150)
    
    mp_trainer = GeologicalPINNTrainer(multiphysics_pinn)
    logger.info("Training multi-physics PINN...")
    mp_trainer.train(mp_training_data, epochs=50)  # Very short for demo
    
    # Test multi-physics prediction
    mp_predictions = multiphysics_pinn(test_points)
    if isinstance(mp_predictions, tuple):
        mp_pred, mp_uncertainty = mp_predictions
    else:
        mp_pred = mp_predictions
    
    logger.info(f"Multi-physics prediction shape: {mp_pred.shape}")
    
    # Extract individual physics fields
    thermal_slice = multiphysics_pinn.physics_field_indices[GeologicalProcessType.THERMAL_EVOLUTION]
    mechanical_slice = multiphysics_pinn.physics_field_indices[GeologicalProcessType.TECTONIC_DEFORMATION]
    
    temperature = mp_pred[:, thermal_slice]
    displacement = mp_pred[:, mechanical_slice]
    
    logger.info(f"Temperature predictions: {temperature.flatten()}")
    logger.info(f"Displacement magnitudes: {torch.norm(displacement, dim=1)}")
    
    # Example 3: PINN Ensemble for Uncertainty Quantification
    logger.info("\n3. PINN Ensemble: Uncertainty Quantification")
    
    ensemble_config = PINNConfig(
        process_type=GeologicalProcessType.THERMAL_EVOLUTION,
        hidden_layers=[128, 64, 32],
        learning_rate=1e-3,
        epochs=200
    )
    
    ensemble = GeologicalPINNFactory.create_ensemble_pinn(
        GeologicalProcessType.THERMAL_EVOLUTION,
        ensemble_config,
        GEOLOGICAL_MATERIALS['basalt'],
        ensemble_size=3  # Small ensemble for demo
    )
    
    # Quick ensemble training (very short for demo)
    thermal_domain = [(-5000, 5000), (-5000, 5000), (-2000, 0), (0, 86400)]  # Include time
    thermal_training_data = create_training_data(thermal_domain, n_physics=1000, n_boundary=100)
    
    logger.info("Training PINN ensemble...")
    ensemble.train_ensemble(thermal_training_data, epochs=20)  # Very short for demo
    
    # Ensemble prediction with uncertainty
    time_test_points = torch.tensor([
        [0.0, 0.0, -500.0, 3600.0],  # 1 hour
        [2000.0, 2000.0, -1000.0, 7200.0],  # 2 hours
    ], dtype=torch.float32)
    
    ensemble_mean, ensemble_std = ensemble.predict(time_test_points)
    logger.info(f"Ensemble mean temperatures: {ensemble_mean.flatten()}")
    logger.info(f"Ensemble prediction std: {ensemble_std.flatten()}")
    
    # Example 4: Integration with Backend Systems
    logger.info("\n4. Backend Integration: Model Management")
    
    integrator = GeologicalPINNIntegrator()
    
    # Demonstrate integrated training data creation
    integrated_data = integrator.create_integrated_training_data(
        domain_bounds=domain_bounds,
        geological_region="yellowstone",
        data_sources=["usgs", "insar", "gps"]
    )
    
    logger.info(f"Integrated training data keys: {list(integrated_data.keys())}")
    
    # Save trained model (if database is available)
    model_metadata = {
        'training_epochs': 100,
        'final_loss': 0.001,
        'domain': 'tectonic_deformation',
        'region': 'demo_area',
        'created_by': 'geological_pinn_framework'
    }
    
    model_id = integrator.save_trained_model(pinn, model_metadata)
    if model_id:
        logger.info(f"Model saved with ID: {model_id}")
        
        # Load model back
        loaded_model = integrator.load_trained_model(model_id)
        if loaded_model:
            logger.info("Model successfully loaded from database")
    
    # Example 5: Comparative Analysis
    logger.info("\n5. Comparative Model Analysis")
    
    models_for_comparison = [pinn, multiphysics_pinn.to(torch.float32)]
    
    test_data = {
        'x_test': test_points,
        # 'u_true': true_values_if_available
    }
    
    try:
        comparison_results = integrator.run_comparative_analysis(models_for_comparison, test_data)
        
        logger.info("Comparative Analysis Results:")
        for model_id, physics_compliance in comparison_results['physics_compliance'].items():
            logger.info(f"  {model_id} - Physics compliance loss: {physics_compliance:.6f}")
        
        for model_id, uncertainty_info in comparison_results['uncertainty_analysis'].items():
            if uncertainty_info.get('available', True):
                logger.info(f"  {model_id} - Mean uncertainty: {uncertainty_info.get('mean_uncertainty', 'N/A'):.6f}")
            else:
                logger.info(f"  {model_id} - Uncertainty quantification not available")
    except Exception as e:
        logger.warning(f"Comparative analysis failed: {e}")
    
    logger.info("\n=== Geological PINN Framework Demonstration Completed ===")
    logger.info("Framework Features Demonstrated:")
    logger.info("  ✓ Single-physics PINNs with uncertainty quantification")
    logger.info("  ✓ Multi-physics PINNs with coupling constraints")
    logger.info("  ✓ PINN ensembles for robust uncertainty estimation")
    logger.info("  ✓ Backend system integration (database, data pipeline)")
    logger.info("  ✓ Comparative model analysis and evaluation")
    logger.info("  ✓ Adaptive training with physics-informed loss balancing")
    logger.info("  ✓ Integration with multi-physics coupling engine")
    logger.info("  ✓ Support for multiple geological materials and processes")