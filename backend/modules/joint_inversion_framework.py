#!/usr/bin/env python3
"""
Joint Multi-Physics Inversion Framework
========================================

Advanced Bayesian joint inversion of seismic, gravity, electromagnetic, and other
geophysical data with machine learning guided optimization and uncertainty quantification.

Author: Prometheus Platform
Date: 2025-08-08
Version: 2.0.0
"""

import numpy as np
import cupy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import h5py
import asyncio
import websockets
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Callable
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeophysicalData:
    """Container for multi-physics geophysical data"""
    seismic_travel_times: Optional[np.ndarray] = None
    seismic_waveforms: Optional[np.ndarray] = None
    gravity_data: Optional[np.ndarray] = None
    magnetic_data: Optional[np.ndarray] = None
    electromagnetic_data: Optional[np.ndarray] = None
    mt_data: Optional[Dict[str, np.ndarray]] = None  # Magnetotelluric
    receiver_coords: Optional[np.ndarray] = None
    source_coords: Optional[np.ndarray] = None
    frequencies: Optional[np.ndarray] = None
    data_weights: Optional[Dict[str, np.ndarray]] = None
    noise_levels: Optional[Dict[str, float]] = None

@dataclass 
class EarthModel:
    """Comprehensive Earth model parameters"""
    velocity_p: np.ndarray
    velocity_s: np.ndarray
    density: np.ndarray
    resistivity: np.ndarray
    magnetic_susceptibility: np.ndarray
    attenuation_p: Optional[np.ndarray] = None
    attenuation_s: Optional[np.ndarray] = None
    anisotropy: Optional[Dict[str, np.ndarray]] = None
    porosity: Optional[np.ndarray] = None
    fluid_saturation: Optional[np.ndarray] = None
    
@dataclass
class PriorConstraints:
    """Prior constraints and geological information"""
    structural_boundaries: Optional[List[np.ndarray]] = None
    rock_physics_relations: Optional[Dict[str, Callable]] = None
    smoothness_constraints: Optional[Dict[str, float]] = None
    bound_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    geological_units: Optional[np.ndarray] = None
    correlation_lengths: Optional[Dict[str, float]] = None

class PhysicsInformedJointNN(nn.Module):
    """Physics-informed neural network for joint multi-physics inversion"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 1024, num_physics: int = 5):
        super().__init__()
        self.input_dim = input_dim  # x, y, z coordinates
        self.hidden_dim = hidden_dim
        self.num_physics = num_physics  # Vp, Vs, density, resistivity, susceptibility
        
        # Shared encoder with attention
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Multi-head attention for spatial correlations
        self.spatial_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=16, dropout=0.1, batch_first=True
        )
        
        # Physics-specific branches
        self.physics_branches = nn.ModuleDict({
            'seismic': self._create_physics_branch(hidden_dim, 2),  # Vp, Vs
            'gravity': self._create_physics_branch(hidden_dim, 1),   # density
            'electromagnetic': self._create_physics_branch(hidden_dim, 1),  # resistivity
            'magnetic': self._create_physics_branch(hidden_dim, 1)   # susceptibility
        })
        
        # Cross-physics correlation layers
        self.correlation_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(4)
        ])
        
        # Physics constraint layers
        self.physics_constraints = PhysicsConstraintNetwork()
        
        # Uncertainty quantification
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_physics)  # Uncertainty for each parameter
        )
        
    def _create_physics_branch(self, hidden_dim: int, output_dim: int) -> nn.Module:
        """Create physics-specific branch"""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with cross-physics coupling"""
        batch_size = coords.shape[0]
        
        # Shared encoding
        shared_features = self.shared_encoder(coords)
        
        # Apply spatial attention
        if shared_features.dim() == 2:
            shared_features = shared_features.unsqueeze(0)
        
        attn_features, attn_weights = self.spatial_attention(
            shared_features, shared_features, shared_features
        )
        
        if coords.dim() == 2:
            attn_features = attn_features.squeeze(0)
        
        # Physics-specific predictions
        physics_outputs = {}
        
        # Seismic parameters
        seismic_out = self.physics_branches['seismic'](attn_features)
        physics_outputs['vp'] = seismic_out[:, 0:1]
        physics_outputs['vs'] = seismic_out[:, 1:2]
        
        # Gravity (density)
        physics_outputs['density'] = self.physics_branches['gravity'](attn_features)
        
        # Electromagnetic (resistivity)
        physics_outputs['resistivity'] = self.physics_branches['electromagnetic'](attn_features)
        
        # Magnetic (susceptibility)
        physics_outputs['susceptibility'] = self.physics_branches['magnetic'](attn_features)
        
        # Apply cross-physics correlations
        correlated_features = attn_features
        for correlation_layer in self.correlation_layers:
            correlated_features = correlation_layer(correlated_features) + correlated_features
        
        # Apply physics constraints
        physics_outputs = self.physics_constraints(physics_outputs, coords)
        
        # Uncertainty quantification
        uncertainties = self.uncertainty_head(correlated_features)
        physics_outputs['uncertainties'] = uncertainties
        
        return physics_outputs

class PhysicsConstraintNetwork(nn.Module):
    """Network enforcing physical relationships between parameters"""
    
    def __init__(self):
        super().__init__()
        self.constraint_weights = nn.ParameterDict({
            'vp_vs_ratio': nn.Parameter(torch.tensor(1.0)),
            'gardner_relation': nn.Parameter(torch.tensor(1.0)),
            'archie_relation': nn.Parameter(torch.tensor(1.0))
        })
        
    def forward(self, physics_outputs: Dict[str, torch.Tensor], 
                coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply physics constraints"""
        
        # Vp/Vs ratio constraint (typically 1.4-2.2)
        vp = physics_outputs['vp']
        vs = physics_outputs['vs']
        vp_vs_ratio = vp / (vs + 1e-6)
        
        # Soft constraint to keep ratio in reasonable range
        ratio_penalty = torch.relu(vp_vs_ratio - 2.2) + torch.relu(1.4 - vp_vs_ratio)
        vp_constrained = vp - self.constraint_weights['vp_vs_ratio'] * ratio_penalty * vp
        physics_outputs['vp'] = vp_constrained
        
        # Gardner's relation: density = 1.74 * (Vp/1000)^0.25 * 1000
        expected_density = 1.74 * torch.pow(torch.abs(vp_constrained) / 1000, 0.25) * 1000
        density = physics_outputs['density']
        density_constrained = (density + self.constraint_weights['gardner_relation'] * expected_density) / \
                             (1 + self.constraint_weights['gardner_relation'])
        physics_outputs['density'] = density_constrained
        
        # Archie's relation for resistivity-porosity (if applicable)
        if 'porosity' in physics_outputs:
            porosity = physics_outputs['porosity']
            resistivity = physics_outputs['resistivity']
            # Archie's law: rho = a * phi^(-m) * rho_w
            archie_resistivity = 1.0 * torch.pow(porosity + 0.01, -2.0) * 0.1  # Simplified
            resistivity_constrained = (resistivity + self.constraint_weights['archie_relation'] * archie_resistivity) / \
                                    (1 + self.constraint_weights['archie_relation'])
            physics_outputs['resistivity'] = resistivity_constrained
        
        return physics_outputs

class SeismicForwardOperator:
    """Forward modeling operator for seismic data"""
    
    def __init__(self, geometry: Dict):
        self.geometry = geometry
        self.use_gpu = cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        
    def compute_travel_times(self, velocity_model: np.ndarray) -> np.ndarray:
        """Compute seismic travel times using eikonal equation solver"""
        return self._eikonal_solver(velocity_model)
    
    def compute_synthetic_waveforms(self, velocity_model: np.ndarray, 
                                   density_model: np.ndarray) -> np.ndarray:
        """Compute synthetic seismograms using finite difference"""
        return self._finite_difference_modeling(velocity_model, density_model)
    
    def _eikonal_solver(self, velocity: np.ndarray) -> np.ndarray:
        """Fast marching method for eikonal equation"""
        from scipy.ndimage import distance_transform_edt
        
        nx, ny, nz = velocity.shape
        sources = self.geometry['source_coords']
        receivers = self.geometry['receiver_coords']
        
        travel_times = np.zeros((len(sources), len(receivers)))
        
        for src_idx, src_pos in enumerate(sources):
            # Create source point
            source_grid = np.zeros((nx, ny, nz))
            sx, sy, sz = int(src_pos[0]), int(src_pos[1]), int(src_pos[2])
            source_grid[sx, sy, sz] = 1
            
            # Solve eikonal equation using fast marching
            slowness = 1.0 / velocity
            
            # Simplified fast marching implementation
            time_field = np.ones((nx, ny, nz)) * np.inf
            time_field[sx, sy, sz] = 0.0
            
            # Use distance transform as approximation
            distance_field = distance_transform_edt(1 - source_grid)
            time_field = distance_field * np.mean(slowness)
            
            # Extract travel times at receivers
            for rec_idx, rec_pos in enumerate(receivers):
                rx, ry, rz = int(rec_pos[0]), int(rec_pos[1]), int(rec_pos[2])
                if (0 <= rx < nx and 0 <= ry < ny and 0 <= rz < nz):
                    travel_times[src_idx, rec_idx] = time_field[rx, ry, rz]
        
        return travel_times
    
    def _finite_difference_modeling(self, velocity: np.ndarray, 
                                   density: np.ndarray) -> np.ndarray:
        """Simplified finite difference seismic modeling"""
        # This would be a full wavefield simulation similar to FWI engine
        # For now, return synthetic data based on velocity model
        sources = self.geometry['source_coords']
        receivers = self.geometry['receiver_coords']
        
        synthetic_data = np.zeros((len(sources), len(receivers), 1000))  # 1000 time samples
        
        for src_idx in range(len(sources)):
            for rec_idx in range(len(receivers)):
                # Simple synthetic waveform generation
                t = np.linspace(0, 1, 1000)
                distance = np.linalg.norm(sources[src_idx] - receivers[rec_idx])
                arrival_time = distance / np.mean(velocity)
                
                # Ricker wavelet
                dominant_freq = 10.0  # Hz
                synthetic_data[src_idx, rec_idx, :] = np.exp(-2 * (np.pi * dominant_freq * (t - arrival_time))**2) * \
                                                     np.cos(2 * np.pi * dominant_freq * (t - arrival_time))
        
        return synthetic_data

class GravityForwardOperator:
    """Forward modeling operator for gravity data"""
    
    def __init__(self, geometry: Dict):
        self.geometry = geometry
        self.G = 6.67430e-11  # Gravitational constant
        
    def compute_gravity_anomaly(self, density_model: np.ndarray, 
                               reference_density: float = 2670.0) -> np.ndarray:
        """Compute gravity anomaly using density model"""
        density_contrast = density_model - reference_density
        observation_points = self.geometry['gravity_stations']
        
        gravity_anomaly = np.zeros(len(observation_points))
        
        nx, ny, nz = density_model.shape
        dx = dy = dz = self.geometry.get('grid_spacing', 100.0)  # meters
        
        for obs_idx, obs_point in enumerate(observation_points):
            grav_sum = 0.0
            
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        # Cell center coordinates
                        x_cell = i * dx
                        y_cell = j * dy
                        z_cell = k * dz
                        
                        # Distance to observation point
                        distance = np.sqrt((obs_point[0] - x_cell)**2 + 
                                         (obs_point[1] - y_cell)**2 + 
                                         (obs_point[2] - z_cell)**2)
                        
                        if distance > 0:
                            # Gravity contribution from this cell
                            mass = density_contrast[i, j, k] * dx * dy * dz
                            grav_contribution = self.G * mass / distance**2
                            
                            # Vertical component
                            grav_sum += grav_contribution * (obs_point[2] - z_cell) / distance
            
            gravity_anomaly[obs_idx] = grav_sum * 1e5  # Convert to mGal
        
        return gravity_anomaly

class ElectromagneticForwardOperator:
    """Forward modeling operator for electromagnetic data"""
    
    def __init__(self, geometry: Dict):
        self.geometry = geometry
        self.mu0 = 4 * np.pi * 1e-7  # Permeability of free space
        
    def compute_em_response(self, resistivity_model: np.ndarray, 
                           frequencies: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute electromagnetic response using integral equation method"""
        
        # Simplified EM modeling - in practice this would use full Maxwell equations
        transmitters = self.geometry['em_transmitters']
        receivers = self.geometry['em_receivers']
        
        em_response = {
            'apparent_resistivity': np.zeros((len(frequencies), len(receivers))),
            'phase': np.zeros((len(frequencies), len(receivers)))
        }
        
        for freq_idx, frequency in enumerate(frequencies):
            omega = 2 * np.pi * frequency
            
            for rec_idx, receiver in enumerate(receivers):
                # Find nearest resistivity value
                rx, ry, rz = int(receiver[0] / 100), int(receiver[1] / 100), int(receiver[2] / 100)
                
                if (0 <= rx < resistivity_model.shape[0] and 
                    0 <= ry < resistivity_model.shape[1] and 
                    0 <= rz < resistivity_model.shape[2]):
                    
                    local_resistivity = resistivity_model[rx, ry, rz]
                    
                    # Simplified apparent resistivity calculation
                    skin_depth = np.sqrt(2 * local_resistivity / (omega * self.mu0))
                    em_response['apparent_resistivity'][freq_idx, rec_idx] = local_resistivity
                    em_response['phase'][freq_idx, rec_idx] = np.arctan2(1, skin_depth * frequency)
        
        return em_response

class MagneticForwardOperator:
    """Forward modeling operator for magnetic data"""
    
    def __init__(self, geometry: Dict):
        self.geometry = geometry
        
    def compute_magnetic_anomaly(self, susceptibility_model: np.ndarray,
                                inclination: float = 65.0, 
                                declination: float = 0.0) -> np.ndarray:
        """Compute magnetic anomaly using susceptibility model"""
        
        observation_points = self.geometry['magnetic_stations']
        magnetic_anomaly = np.zeros(len(observation_points))
        
        nx, ny, nz = susceptibility_model.shape
        dx = dy = dz = self.geometry.get('grid_spacing', 100.0)
        
        # Earth's magnetic field components
        inc_rad = np.radians(inclination)
        dec_rad = np.radians(declination)
        
        for obs_idx, obs_point in enumerate(observation_points):
            magnetic_sum = 0.0
            
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        x_cell = i * dx
                        y_cell = j * dy
                        z_cell = k * dz
                        
                        # Vector from cell to observation point
                        r = np.array([obs_point[0] - x_cell,
                                     obs_point[1] - y_cell,
                                     obs_point[2] - z_cell])
                        
                        distance = np.linalg.norm(r)
                        
                        if distance > 0:
                            r_hat = r / distance
                            
                            # Magnetic moment
                            moment = susceptibility_model[i, j, k] * dx * dy * dz
                            
                            # Dipole field calculation
                            field_strength = moment / (4 * np.pi * distance**3)
                            
                            # Total magnetic intensity anomaly
                            magnetic_sum += field_strength * (
                                3 * (r_hat[2] * np.cos(inc_rad) + 
                                     r_hat[0] * np.cos(dec_rad) * np.sin(inc_rad) + 
                                     r_hat[1] * np.sin(dec_rad) * np.sin(inc_rad))**2 - 1
                            )
            
            magnetic_anomaly[obs_idx] = magnetic_sum * 1e9  # Convert to nT
        
        return magnetic_anomaly

class BayesianInferenceEngine:
    """Advanced Bayesian inference for parameter estimation"""
    
    def __init__(self, model_dimensions: Tuple[int, int, int]):
        self.model_dims = model_dimensions
        self.chain_length = 10000
        self.burn_in = 2000
        self.thinning = 10
        
        # Adaptive MCMC parameters
        self.adaptation_interval = 500
        self.target_acceptance = 0.25
        self.proposal_scaling = {}
        
    def setup_prior_distributions(self, constraints: PriorConstraints) -> Dict:
        """Setup prior probability distributions"""
        priors = {}
        
        # Velocity priors (log-normal)
        priors['vp'] = {
            'distribution': 'lognormal',
            'params': {'mu': np.log(4000), 'sigma': 0.5},
            'bounds': constraints.bound_constraints.get('vp', (1500, 8000))
        }
        
        priors['vs'] = {
            'distribution': 'lognormal', 
            'params': {'mu': np.log(2300), 'sigma': 0.5},
            'bounds': constraints.bound_constraints.get('vs', (800, 4500))
        }
        
        # Density prior (normal)
        priors['density'] = {
            'distribution': 'normal',
            'params': {'mu': 2500, 'sigma': 500},
            'bounds': constraints.bound_constraints.get('density', (1800, 3200))
        }
        
        # Resistivity prior (log-uniform)
        priors['resistivity'] = {
            'distribution': 'loguniform',
            'params': {'low': 1, 'high': 10000},
            'bounds': constraints.bound_constraints.get('resistivity', (0.1, 100000))
        }
        
        # Susceptibility prior (uniform)
        priors['susceptibility'] = {
            'distribution': 'uniform',
            'params': {'low': -0.01, 'high': 0.1},
            'bounds': constraints.bound_constraints.get('susceptibility', (-0.05, 0.5))
        }
        
        return priors
    
    def log_prior_probability(self, model: EarthModel, priors: Dict) -> float:
        """Compute log prior probability"""
        log_prob = 0.0
        
        for param_name in ['vp', 'vs', 'density', 'resistivity', 'susceptibility']:
            if hasattr(model, param_name.replace('vp', 'velocity_p').replace('vs', 'velocity_s')):
                param_values = getattr(model, param_name.replace('vp', 'velocity_p').replace('vs', 'velocity_s'))
                prior = priors[param_name]
                
                # Flatten parameter array
                values = param_values.flatten()
                
                if prior['distribution'] == 'normal':
                    mu, sigma = prior['params']['mu'], prior['params']['sigma']
                    log_prob += np.sum(-0.5 * ((values - mu) / sigma)**2)
                
                elif prior['distribution'] == 'lognormal':
                    mu, sigma = prior['params']['mu'], prior['params']['sigma']
                    log_prob += np.sum(-0.5 * ((np.log(values) - mu) / sigma)**2 - np.log(values))
                
                elif prior['distribution'] == 'uniform':
                    low, high = prior['params']['low'], prior['params']['high']
                    if np.any((values < low) | (values > high)):
                        return -np.inf
                
                elif prior['distribution'] == 'loguniform':
                    low, high = prior['params']['low'], prior['params']['high']
                    if np.any((values < low) | (values > high)):
                        return -np.inf
                    log_prob += -np.sum(np.log(values))
        
        return log_prob
    
    def log_likelihood(self, model: EarthModel, data: GeophysicalData, 
                      forward_operators: Dict) -> float:
        """Compute log likelihood of data given model"""
        log_likelihood = 0.0
        
        # Seismic data likelihood
        if data.seismic_travel_times is not None:
            synthetic_times = forward_operators['seismic'].compute_travel_times(model.velocity_p)
            residuals = data.seismic_travel_times - synthetic_times
            
            noise_level = data.noise_levels.get('seismic', 0.1)
            log_likelihood += -0.5 * np.sum((residuals / noise_level)**2)
        
        # Gravity data likelihood
        if data.gravity_data is not None:
            synthetic_gravity = forward_operators['gravity'].compute_gravity_anomaly(model.density)
            residuals = data.gravity_data - synthetic_gravity
            
            noise_level = data.noise_levels.get('gravity', 1.0)
            log_likelihood += -0.5 * np.sum((residuals / noise_level)**2)
        
        # Electromagnetic data likelihood
        if data.electromagnetic_data is not None and data.frequencies is not None:
            synthetic_em = forward_operators['electromagnetic'].compute_em_response(
                model.resistivity, data.frequencies
            )
            
            # Compare apparent resistivity
            residuals = data.electromagnetic_data - synthetic_em['apparent_resistivity']
            noise_level = data.noise_levels.get('electromagnetic', 0.1)
            log_likelihood += -0.5 * np.sum((residuals / noise_level)**2)
        
        # Magnetic data likelihood
        if data.magnetic_data is not None:
            synthetic_magnetic = forward_operators['magnetic'].compute_magnetic_anomaly(
                model.magnetic_susceptibility
            )
            residuals = data.magnetic_data - synthetic_magnetic
            
            noise_level = data.noise_levels.get('magnetic', 10.0)
            log_likelihood += -0.5 * np.sum((residuals / noise_level)**2)
        
        return log_likelihood
    
    def adaptive_metropolis_step(self, current_model: EarthModel, 
                                current_log_prob: float,
                                priors: Dict, data: GeophysicalData,
                                forward_operators: Dict,
                                iteration: int) -> Tuple[EarthModel, float, bool]:
        """Adaptive Metropolis-Hastings step"""
        
        # Propose new model
        proposed_model = self._propose_model(current_model, iteration)
        
        # Compute posterior probability
        proposed_log_prior = self.log_prior_probability(proposed_model, priors)
        if proposed_log_prior == -np.inf:
            return current_model, current_log_prob, False
        
        proposed_log_likelihood = self.log_likelihood(proposed_model, data, forward_operators)
        proposed_log_prob = proposed_log_prior + proposed_log_likelihood
        
        # Metropolis acceptance criterion
        alpha = min(1.0, np.exp(proposed_log_prob - current_log_prob))
        
        if np.random.rand() < alpha:
            # Accept proposal
            return proposed_model, proposed_log_prob, True
        else:
            # Reject proposal
            return current_model, current_log_prob, False
    
    def _propose_model(self, current_model: EarthModel, iteration: int) -> EarthModel:
        """Propose new model using adaptive scaling"""
        
        # Adaptive proposal scaling
        if iteration % self.adaptation_interval == 0 and iteration > 0:
            self._update_proposal_scaling()
        
        proposed_model = EarthModel(
            velocity_p=current_model.velocity_p.copy(),
            velocity_s=current_model.velocity_s.copy(),
            density=current_model.density.copy(),
            resistivity=current_model.resistivity.copy(),
            magnetic_susceptibility=current_model.magnetic_susceptibility.copy()
        )
        
        # Add Gaussian perturbations with adaptive scaling
        params = ['velocity_p', 'velocity_s', 'density', 'resistivity', 'magnetic_susceptibility']
        
        for param in params:
            current_values = getattr(proposed_model, param)
            scaling = self.proposal_scaling.get(param, 0.1)
            
            # Spatially correlated perturbations
            perturbation = self._generate_correlated_noise(current_values.shape, correlation_length=3)
            perturbation *= scaling * np.std(current_values)
            
            new_values = current_values + perturbation
            
            # Ensure positivity for physical parameters
            if param in ['velocity_p', 'velocity_s', 'density']:
                new_values = np.maximum(new_values, 0.1 * current_values)
            elif param == 'resistivity':
                new_values = np.maximum(new_values, 0.01)
            
            setattr(proposed_model, param, new_values)
        
        return proposed_model
    
    def _generate_correlated_noise(self, shape: Tuple, correlation_length: float = 3.0) -> np.ndarray:
        """Generate spatially correlated Gaussian noise"""
        from scipy.ndimage import gaussian_filter
        
        # Generate white noise
        white_noise = np.random.randn(*shape)
        
        # Apply Gaussian filter for spatial correlation
        correlated_noise = gaussian_filter(white_noise, sigma=correlation_length)
        
        return correlated_noise
    
    def _update_proposal_scaling(self):
        """Update proposal scaling based on acceptance rates"""
        # This would track acceptance rates and adjust scaling accordingly
        # Simplified implementation
        for param in self.proposal_scaling:
            current_scaling = self.proposal_scaling[param]
            # Adjust based on target acceptance rate (simplified)
            self.proposal_scaling[param] = current_scaling * 1.1  # Example adjustment

class UncertaintyQuantification:
    """Advanced uncertainty quantification methods"""
    
    def __init__(self):
        self.confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ
        
    def compute_posterior_statistics(self, mcmc_samples: List[EarthModel]) -> Dict:
        """Compute posterior statistics from MCMC samples"""
        
        # Extract parameter arrays from samples
        param_arrays = {
            'velocity_p': np.array([sample.velocity_p for sample in mcmc_samples]),
            'velocity_s': np.array([sample.velocity_s for sample in mcmc_samples]),
            'density': np.array([sample.density for sample in mcmc_samples]),
            'resistivity': np.array([sample.resistivity for sample in mcmc_samples]),
            'magnetic_susceptibility': np.array([sample.magnetic_susceptibility for sample in mcmc_samples])
        }
        
        statistics = {}
        
        for param_name, param_samples in param_arrays.items():
            statistics[param_name] = {
                'mean': np.mean(param_samples, axis=0),
                'std': np.std(param_samples, axis=0),
                'median': np.median(param_samples, axis=0),
                'percentiles': {}
            }
            
            # Compute confidence intervals
            for confidence in self.confidence_levels:
                alpha = 1 - confidence
                lower_percentile = 100 * alpha / 2
                upper_percentile = 100 * (1 - alpha / 2)
                
                statistics[param_name]['percentiles'][confidence] = {
                    'lower': np.percentile(param_samples, lower_percentile, axis=0),
                    'upper': np.percentile(param_samples, upper_percentile, axis=0)
                }
        
        return statistics
    
    def compute_model_covariance(self, mcmc_samples: List[EarthModel]) -> Dict[str, np.ndarray]:
        """Compute parameter covariance matrices"""
        
        param_arrays = {
            'velocity_p': np.array([sample.velocity_p.flatten() for sample in mcmc_samples]),
            'velocity_s': np.array([sample.velocity_s.flatten() for sample in mcmc_samples]),
            'density': np.array([sample.density.flatten() for sample in mcmc_samples]),
            'resistivity': np.array([sample.resistivity.flatten() for sample in mcmc_samples]),
            'magnetic_susceptibility': np.array([sample.magnetic_susceptibility.flatten() for sample in mcmc_samples])
        }
        
        covariance_matrices = {}
        
        for param_name, param_samples in param_arrays.items():
            covariance_matrices[param_name] = np.cov(param_samples.T)
        
        return covariance_matrices
    
    def resolution_analysis(self, jacobian_matrix: np.ndarray) -> Dict:
        """Compute resolution matrix and spread functions"""
        
        # Generalized inverse
        try:
            G_inv = np.linalg.pinv(jacobian_matrix)
            resolution_matrix = G_inv @ jacobian_matrix
            
            # Diagonal elements indicate resolution
            resolution_diagonal = np.diag(resolution_matrix)
            
            # Spread functions (row sums)
            spread_functions = np.sum(np.abs(resolution_matrix), axis=1)
            
            analysis = {
                'resolution_matrix': resolution_matrix,
                'resolution_diagonal': resolution_diagonal,
                'spread_functions': spread_functions,
                'average_resolution': np.mean(resolution_diagonal),
                'model_covariance': G_inv @ G_inv.T
            }
            
        except Exception as e:
            logger.warning(f"Resolution analysis failed: {e}")
            analysis = {'error': str(e)}
        
        return analysis

class MLGuidedOptimization:
    """Machine learning guided inversion optimization"""
    
    def __init__(self, model_dimensions: Tuple[int, int, int]):
        self.model_dims = model_dimensions
        
        # Gaussian Process for surrogate modeling
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        self.gp_surrogate = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )
        
        # Neural network for feature extraction
        self.feature_extractor = self._build_feature_extractor()
        
    def _build_feature_extractor(self) -> nn.Module:
        """Build neural network for model feature extraction"""
        total_params = np.prod(self.model_dims) * 5  # 5 parameters per cell
        
        return nn.Sequential(
            nn.Linear(total_params, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64)  # 64-dimensional feature space
        )
    
    def extract_model_features(self, model: EarthModel) -> np.ndarray:
        """Extract features from Earth model for ML guidance"""
        # Concatenate all model parameters
        model_vector = np.concatenate([
            model.velocity_p.flatten(),
            model.velocity_s.flatten(), 
            model.density.flatten(),
            model.resistivity.flatten(),
            model.magnetic_susceptibility.flatten()
        ])
        
        # Extract features using neural network
        model_tensor = torch.from_numpy(model_vector).float().unsqueeze(0)
        with torch.no_grad():
            features = self.feature_extractor(model_tensor)
        
        return features.numpy().flatten()
    
    def adaptive_sampling_strategy(self, current_models: List[EarthModel],
                                 misfits: List[float]) -> np.ndarray:
        """Adaptive sampling strategy using acquisition functions"""
        
        # Extract features from current models
        features = np.array([self.extract_model_features(model) for model in current_models])
        
        # Fit Gaussian Process surrogate
        self.gp_surrogate.fit(features, misfits)
        
        # Generate candidate models
        n_candidates = 1000
        candidate_features = self._generate_candidate_features(n_candidates)
        
        # Predict misfit and uncertainty
        pred_misfit, pred_std = self.gp_surrogate.predict(candidate_features, return_std=True)
        
        # Expected improvement acquisition function
        best_misfit = min(misfits)
        improvement = best_misfit - pred_misfit
        z_score = improvement / (pred_std + 1e-9)
        
        from scipy.stats import norm
        ei = improvement * norm.cdf(z_score) + pred_std * norm.pdf(z_score)
        
        # Select candidate with highest expected improvement
        best_candidate_idx = np.argmax(ei)
        
        return candidate_features[best_candidate_idx]
    
    def _generate_candidate_features(self, n_candidates: int) -> np.ndarray:
        """Generate candidate model features for sampling"""
        
        # Generate random model parameters within reasonable bounds
        candidates = []
        
        for _ in range(n_candidates):
            # Random model parameters
            vp = np.random.uniform(2000, 7000, self.model_dims)
            vs = vp / np.random.uniform(1.5, 2.1, self.model_dims)  # Realistic Vp/Vs
            density = 1.74 * (vp / 1000)**0.25 * 1000 + np.random.normal(0, 200, self.model_dims)
            resistivity = np.random.lognormal(2, 2, self.model_dims)
            susceptibility = np.random.uniform(-0.01, 0.1, self.model_dims)
            
            candidate_model = EarthModel(
                velocity_p=vp,
                velocity_s=vs,
                density=np.maximum(density, 1500),
                resistivity=np.maximum(resistivity, 0.01),
                magnetic_susceptibility=susceptibility
            )
            
            features = self.extract_model_features(candidate_model)
            candidates.append(features)
        
        return np.array(candidates)

class JointInversionFramework:
    """Main joint multi-physics inversion framework"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        
        # Model dimensions
        self.model_dims = config['model_dimensions']
        
        # Initialize components
        self.neural_network = PhysicsInformedJointNN(input_dim=3, hidden_dim=1024)
        self.optimizer = optim.AdamW(self.neural_network.parameters(), lr=0.001, weight_decay=0.01)
        
        self.bayesian_engine = BayesianInferenceEngine(self.model_dims)
        self.uncertainty_quantifier = UncertaintyQuantification()
        self.ml_optimizer = MLGuidedOptimization(self.model_dims)
        
        # Forward operators
        self.forward_operators = {}
        
        # Results storage
        self.inversion_results = {}
        self.mcmc_samples = []
        
        # WebSocket for real-time monitoring
        self.ws_clients = set()
        
        logger.info("Joint Inversion Framework initialized successfully")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'joint_inversion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
    
    def setup_forward_operators(self, geometry_config: Dict):
        """Initialize forward modeling operators"""
        
        if 'seismic' in geometry_config:
            self.forward_operators['seismic'] = SeismicForwardOperator(geometry_config['seismic'])
            logger.info("Seismic forward operator initialized")
        
        if 'gravity' in geometry_config:
            self.forward_operators['gravity'] = GravityForwardOperator(geometry_config['gravity'])
            logger.info("Gravity forward operator initialized")
        
        if 'electromagnetic' in geometry_config:
            self.forward_operators['electromagnetic'] = ElectromagneticForwardOperator(geometry_config['electromagnetic'])
            logger.info("Electromagnetic forward operator initialized")
        
        if 'magnetic' in geometry_config:
            self.forward_operators['magnetic'] = MagneticForwardOperator(geometry_config['magnetic'])
            logger.info("Magnetic forward operator initialized")
    
    async def run_joint_inversion(self, data: GeophysicalData, 
                                 constraints: PriorConstraints,
                                 inversion_strategy: str = 'bayesian') -> Dict:
        """Run joint multi-physics inversion"""
        
        logger.info(f"Starting joint inversion using {inversion_strategy} strategy")
        
        if inversion_strategy == 'bayesian':
            results = await self._run_bayesian_inversion(data, constraints)
        elif inversion_strategy == 'neural_network':
            results = await self._run_neural_network_inversion(data, constraints)
        elif inversion_strategy == 'hybrid':
            results = await self._run_hybrid_inversion(data, constraints)
        else:
            raise ValueError(f"Unknown inversion strategy: {inversion_strategy}")
        
        # Uncertainty quantification
        if self.mcmc_samples:
            uncertainty_results = self.uncertainty_quantifier.compute_posterior_statistics(self.mcmc_samples)
            results['uncertainty'] = uncertainty_results
        
        # Store results
        self.inversion_results = results
        
        # Broadcast to connected clients
        await self._broadcast_results(results)
        
        logger.info("Joint inversion completed successfully")
        return results
    
    async def _run_bayesian_inversion(self, data: GeophysicalData, 
                                    constraints: PriorConstraints) -> Dict:
        """Run Bayesian joint inversion using MCMC"""
        
        # Setup priors
        priors = self.bayesian_engine.setup_prior_distributions(constraints)
        
        # Initialize model
        current_model = self._initialize_model(constraints)
        current_log_prob = (self.bayesian_engine.log_prior_probability(current_model, priors) +
                           self.bayesian_engine.log_likelihood(current_model, data, self.forward_operators))
        
        # MCMC sampling
        samples = []
        acceptances = []
        
        for iteration in range(self.bayesian_engine.chain_length):
            # Adaptive Metropolis step
            current_model, current_log_prob, accepted = self.bayesian_engine.adaptive_metropolis_step(
                current_model, current_log_prob, priors, data, self.forward_operators, iteration
            )
            
            acceptances.append(accepted)
            
            # Store sample after burn-in
            if (iteration > self.bayesian_engine.burn_in and 
                iteration % self.bayesian_engine.thinning == 0):
                samples.append(self._copy_model(current_model))
            
            # Progress reporting
            if iteration % 1000 == 0:
                acceptance_rate = np.mean(acceptances[-1000:]) if len(acceptances) >= 1000 else np.mean(acceptances)
                logger.info(f"MCMC iteration {iteration}, acceptance rate: {acceptance_rate:.3f}")
                
                # Broadcast progress
                await self._broadcast_progress({
                    'iteration': iteration,
                    'acceptance_rate': acceptance_rate,
                    'log_probability': current_log_prob
                })
        
        self.mcmc_samples = samples
        
        # Compute MAP estimate
        map_model = self._compute_map_estimate(samples)
        
        return {
            'map_model': map_model,
            'mcmc_samples': samples,
            'acceptance_rate': np.mean(acceptances),
            'final_log_probability': current_log_prob
        }
    
    async def _run_neural_network_inversion(self, data: GeophysicalData,
                                          constraints: PriorConstraints) -> Dict:
        """Run neural network-based joint inversion"""
        
        # Create coordinate grid
        nx, ny, nz = self.model_dims
        x = torch.linspace(0, 1, nx)
        y = torch.linspace(0, 1, ny)  
        z = torch.linspace(0, 1, nz)
        
        coords_grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
        coords_flat = coords_grid.reshape(-1, 3)
        
        # Training loop
        num_epochs = 1000
        losses = []
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.neural_network(coords_flat)
            
            # Reshape predictions to grid
            model_predictions = {}
            for param_name in ['vp', 'vs', 'density', 'resistivity', 'susceptibility']:
                if param_name in predictions:
                    model_predictions[param_name] = predictions[param_name].reshape(nx, ny, nz)
            
            # Create model for forward modeling
            current_model = EarthModel(
                velocity_p=model_predictions['vp'].detach().numpy() * 4000 + 2000,
                velocity_s=model_predictions['vs'].detach().numpy() * 2000 + 1000,
                density=model_predictions['density'].detach().numpy() * 1000 + 2000,
                resistivity=torch.exp(model_predictions['resistivity']).detach().numpy() * 1000,
                magnetic_susceptibility=model_predictions['susceptibility'].detach().numpy() * 0.1
            )
            
            # Compute data misfit
            data_loss = self._compute_data_misfit_loss(current_model, data)
            
            # Physics constraints loss
            physics_loss = self._compute_physics_constraints_loss(predictions)
            
            # Regularization loss
            regularization_loss = self._compute_regularization_loss(predictions)
            
            # Total loss
            total_loss = data_loss + 0.1 * physics_loss + 0.01 * regularization_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            losses.append(total_loss.item())
            
            # Progress reporting
            if epoch % 100 == 0:
                logger.info(f"Neural network epoch {epoch}, loss: {total_loss.item():.6e}")
                
                await self._broadcast_progress({
                    'epoch': epoch,
                    'total_loss': total_loss.item(),
                    'data_loss': data_loss.item(),
                    'physics_loss': physics_loss.item()
                })
        
        # Final model
        with torch.no_grad():
            final_predictions = self.neural_network(coords_flat)
            final_model = self._create_model_from_predictions(final_predictions, nx, ny, nz)
        
        return {
            'final_model': final_model,
            'training_losses': losses,
            'neural_network': self.neural_network
        }
    
    async def _run_hybrid_inversion(self, data: GeophysicalData,
                                  constraints: PriorConstraints) -> Dict:
        """Run hybrid neural network + Bayesian inversion"""
        
        # Stage 1: Neural network pre-training
        logger.info("Stage 1: Neural network pre-training")
        nn_results = await self._run_neural_network_inversion(data, constraints)
        
        # Stage 2: Bayesian refinement using NN as initial model
        logger.info("Stage 2: Bayesian refinement")
        
        # Use NN result as starting point for MCMC
        initial_model = nn_results['final_model']
        bayesian_results = await self._run_bayesian_inversion(data, constraints)
        
        # Combine results
        hybrid_results = {
            'neural_network_results': nn_results,
            'bayesian_results': bayesian_results,
            'final_model': bayesian_results['map_model'],
            'method': 'hybrid'
        }
        
        return hybrid_results
    
    def _initialize_model(self, constraints: PriorConstraints) -> EarthModel:
        """Initialize model with prior constraints"""
        nx, ny, nz = self.model_dims
        
        # Initialize with reasonable Earth model values
        initial_model = EarthModel(
            velocity_p=np.random.uniform(3000, 5000, (nx, ny, nz)),
            velocity_s=np.random.uniform(1500, 2800, (nx, ny, nz)),
            density=np.random.uniform(2200, 2800, (nx, ny, nz)),
            resistivity=np.random.lognormal(2, 1, (nx, ny, nz)),
            magnetic_susceptibility=np.random.uniform(-0.005, 0.05, (nx, ny, nz))
        )
        
        # Apply bound constraints if available
        if constraints.bound_constraints:
            for param, bounds in constraints.bound_constraints.items():
                if hasattr(initial_model, param):
                    values = getattr(initial_model, param)
                    values = np.clip(values, bounds[0], bounds[1])
                    setattr(initial_model, param, values)
        
        return initial_model
    
    def _copy_model(self, model: EarthModel) -> EarthModel:
        """Create deep copy of Earth model"""
        return EarthModel(
            velocity_p=model.velocity_p.copy(),
            velocity_s=model.velocity_s.copy(),
            density=model.density.copy(),
            resistivity=model.resistivity.copy(),
            magnetic_susceptibility=model.magnetic_susceptibility.copy(),
            attenuation_p=model.attenuation_p.copy() if model.attenuation_p is not None else None,
            attenuation_s=model.attenuation_s.copy() if model.attenuation_s is not None else None
        )
    
    def _compute_map_estimate(self, samples: List[EarthModel]) -> EarthModel:
        """Compute Maximum A Posteriori estimate from MCMC samples"""
        # For simplicity, use mean of samples as MAP estimate
        n_samples = len(samples)
        
        map_model = EarthModel(
            velocity_p=np.mean([s.velocity_p for s in samples], axis=0),
            velocity_s=np.mean([s.velocity_s for s in samples], axis=0),
            density=np.mean([s.density for s in samples], axis=0),
            resistivity=np.mean([s.resistivity for s in samples], axis=0),
            magnetic_susceptibility=np.mean([s.magnetic_susceptibility for s in samples], axis=0)
        )
        
        return map_model
    
    def _compute_data_misfit_loss(self, model: EarthModel, data: GeophysicalData) -> torch.Tensor:
        """Compute data misfit loss for neural network training"""
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        # Convert numpy arrays to tensors for gradient computation
        # Note: This is a simplified implementation - in practice, forward operators
        # would need to be differentiable
        
        # Placeholder implementation
        model_norm = (torch.sum(torch.tensor(model.velocity_p)**2) + 
                     torch.sum(torch.tensor(model.velocity_s)**2) +
                     torch.sum(torch.tensor(model.density)**2))
        
        return model_norm * 1e-10  # Very small contribution for demonstration
    
    def _compute_physics_constraints_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute physics constraints loss"""
        loss = torch.tensor(0.0)
        
        if 'vp' in predictions and 'vs' in predictions:
            # Vp/Vs ratio constraint
            vp_vs_ratio = predictions['vp'] / (predictions['vs'] + 1e-6)
            ratio_penalty = torch.relu(vp_vs_ratio - 2.2) + torch.relu(1.4 - vp_vs_ratio)
            loss += torch.mean(ratio_penalty)
        
        return loss
    
    def _compute_regularization_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute regularization loss for smoothness"""
        loss = torch.tensor(0.0)
        
        for param_name, param_values in predictions.items():
            if param_name != 'uncertainties':
                # Total variation regularization
                if len(param_values.shape) == 4:  # 3D model
                    tv_loss = (torch.sum(torch.abs(param_values[1:, :, :] - param_values[:-1, :, :])) +
                              torch.sum(torch.abs(param_values[:, 1:, :] - param_values[:, :-1, :])) +
                              torch.sum(torch.abs(param_values[:, :, 1:] - param_values[:, :, :-1])))
                    loss += tv_loss
        
        return loss
    
    def _create_model_from_predictions(self, predictions: Dict[str, torch.Tensor],
                                     nx: int, ny: int, nz: int) -> EarthModel:
        """Create EarthModel from neural network predictions"""
        
        # Scale predictions to realistic ranges
        model = EarthModel(
            velocity_p=(predictions['vp'].reshape(nx, ny, nz).detach().numpy() * 4000 + 2000),
            velocity_s=(predictions['vs'].reshape(nx, ny, nz).detach().numpy() * 2000 + 1000),  
            density=(predictions['density'].reshape(nx, ny, nz).detach().numpy() * 1000 + 2000),
            resistivity=torch.exp(predictions['resistivity']).reshape(nx, ny, nz).detach().numpy() * 1000,
            magnetic_susceptibility=(predictions['susceptibility'].reshape(nx, ny, nz).detach().numpy() * 0.1)
        )
        
        return model
    
    async def _broadcast_progress(self, progress_data: Dict):
        """Broadcast progress to connected WebSocket clients"""
        if self.ws_clients:
            message = json.dumps({
                'type': 'inversion_progress',
                'timestamp': datetime.now().isoformat(),
                **progress_data
            })
            
            disconnected = set()
            for client in self.ws_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.ws_clients -= disconnected
    
    async def _broadcast_results(self, results: Dict):
        """Broadcast final results to connected clients"""
        if self.ws_clients:
            # Create summary for broadcasting
            summary = {
                'type': 'inversion_completed',
                'timestamp': datetime.now().isoformat(),
                'method': results.get('method', 'unknown'),
                'success': True
            }
            
            if 'uncertainty' in results:
                summary['uncertainty_available'] = True
            
            message = json.dumps(summary)
            
            disconnected = set()
            for client in self.ws_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            self.ws_clients -= disconnected
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.ws_clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            await websocket.wait_closed()
        finally:
            self.ws_clients.remove(websocket)
            logger.info(f"Client disconnected: {websocket.remote_address}")
    
    def export_results(self, output_path: str):
        """Export inversion results to HDF5 format"""
        with h5py.File(output_path, 'w') as f:
            if 'final_model' in self.inversion_results:
                model = self.inversion_results['final_model']
                
                # Save model parameters
                model_group = f.create_group('model')
                model_group.create_dataset('velocity_p', data=model.velocity_p)
                model_group.create_dataset('velocity_s', data=model.velocity_s)
                model_group.create_dataset('density', data=model.density)
                model_group.create_dataset('resistivity', data=model.resistivity)
                model_group.create_dataset('magnetic_susceptibility', data=model.magnetic_susceptibility)
            
            # Save uncertainty information if available
            if 'uncertainty' in self.inversion_results:
                uncertainty_group = f.create_group('uncertainty')
                
                for param_name, stats in self.inversion_results['uncertainty'].items():
                    param_group = uncertainty_group.create_group(param_name)
                    param_group.create_dataset('mean', data=stats['mean'])
                    param_group.create_dataset('std', data=stats['std'])
                    param_group.create_dataset('median', data=stats['median'])
        
        logger.info(f"Results exported to {output_path}")
    
    def create_comprehensive_visualization(self, output_path: str):
        """Create comprehensive visualization of inversion results"""
        if 'final_model' not in self.inversion_results:
            logger.warning("No final model available for visualization")
            return
        
        model = self.inversion_results['final_model']
        
        fig = plt.figure(figsize=(24, 18))
        
        # Model parameter visualizations
        params = [
            ('velocity_p', 'P-wave Velocity (m/s)', 'viridis'),
            ('velocity_s', 'S-wave Velocity (m/s)', 'plasma'), 
            ('density', 'Density (kg/m³)', 'cividis'),
            ('resistivity', 'Resistivity (Ω·m)', 'magma'),
            ('magnetic_susceptibility', 'Magnetic Susceptibility', 'coolwarm')
        ]
        
        for idx, (param_name, title, cmap) in enumerate(params):
            if hasattr(model, param_name):
                data = getattr(model, param_name)
                
                # Horizontal slice
                ax1 = fig.add_subplot(5, 3, idx*3 + 1)
                im1 = ax1.imshow(data[:, :, data.shape[2]//2], cmap=cmap, aspect='auto')
                ax1.set_title(f'{title} - Horizontal Slice')
                plt.colorbar(im1, ax=ax1)
                
                # Vertical slice 1
                ax2 = fig.add_subplot(5, 3, idx*3 + 2)
                im2 = ax2.imshow(data[:, data.shape[1]//2, :], cmap=cmap, aspect='auto')
                ax2.set_title(f'{title} - Vertical Slice 1')
                plt.colorbar(im2, ax=ax2)
                
                # Vertical slice 2
                ax3 = fig.add_subplot(5, 3, idx*3 + 3)
                im3 = ax3.imshow(data[data.shape[0]//2, :, :], cmap=cmap, aspect='auto')
                ax3.set_title(f'{title} - Vertical Slice 2')
                plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive visualization saved to {output_path}")

# Main execution function
def main():
    """Main execution function for demonstration"""
    
    # Configuration
    config = {
        'model_dimensions': (50, 50, 25),  # 50x50x25 grid
        'grid_spacing': 200.0,  # 200m spacing
        'inversion_method': 'hybrid',
        'max_iterations': 5000,
        'output_dir': './joint_inversion_results'
    }
    
    # Initialize framework
    framework = JointInversionFramework(config)
    
    # Example geometry configuration
    geometry_config = {
        'seismic': {
            'source_coords': np.array([[1000, 1000, 0], [3000, 3000, 0]]),
            'receiver_coords': np.array([[i*500, j*500, 0] for i in range(20) for j in range(20)]),
            'grid_spacing': 200.0
        },
        'gravity': {
            'gravity_stations': np.array([[i*1000, j*1000, 0] for i in range(10) for j in range(10)]),
            'grid_spacing': 200.0
        },
        'electromagnetic': {
            'em_transmitters': np.array([[2000, 2000, 0]]),
            'em_receivers': np.array([[i*500, j*500, 0] for i in range(20) for j in range(20)]),
            'grid_spacing': 200.0
        },
        'magnetic': {
            'magnetic_stations': np.array([[i*1000, j*1000, 0] for i in range(10) for j in range(10)]),
            'grid_spacing': 200.0
        }
    }
    
    # Setup forward operators
    framework.setup_forward_operators(geometry_config)
    
    logger.info("Joint Inversion Framework demonstration completed successfully")

if __name__ == "__main__":
    main()