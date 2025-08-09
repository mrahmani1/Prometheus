"""
Advanced Deformation Analysis with Source Modeling

This module provides comprehensive deformation analysis capabilities including:
- Advanced source modeling (Mogi, Sill, Dike, Yang models)
- Bayesian parameter estimation with uncertainty quantification
- Time-dependent deformation analysis
- Multi-source inversion with regularization
- Volcanic and tectonic deformation modeling
- Real-time deformation monitoring and alerting

Key Features:
- Multiple deformation source models
- MCMC Bayesian parameter estimation
- Uncertainty quantification and model selection
- Time series analysis with trend detection
- Automated source characterization
- Integration with InSAR processing pipeline
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from scipy import optimize, interpolate, stats
from scipy.spatial.distance import cdist
from scipy.linalg import inv, pinv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy.random as npr
from typing import Dict, List, Tuple, Optional, Union, Callable
import datetime
import warnings
import logging
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import h5py
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SourceType(Enum):
    MOGI = "mogi"
    SILL = "sill"
    DIKE = "dike"
    YANG = "yang"
    FAULT = "fault"
    COMPOUND = "compound"

class DeformationRegime(Enum):
    VOLCANIC = "volcanic"
    TECTONIC = "tectonic"
    ANTHROPOGENIC = "anthropogenic"
    COMBINED = "combined"

@dataclass
class DeformationSource:
    source_type: SourceType
    parameters: Dict[str, float]
    parameter_bounds: Dict[str, Tuple[float, float]]
    uncertainty: Optional[Dict[str, float]] = None
    confidence_interval: Optional[Dict[str, Tuple[float, float]]] = None
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None

@dataclass
class DeformationObservation:
    location: Tuple[float, float]  # (lon, lat)
    displacement: Tuple[float, float, float]  # (east, north, up)
    uncertainty: Tuple[float, float, float]  # (sigma_e, sigma_n, sigma_u)
    timestamp: datetime.datetime
    data_type: str = "insar"  # insar, gps, leveling, etc.

@dataclass
class AnalysisConfig:
    max_sources: int = 5
    mcmc_samples: int = 10000
    burnin_samples: int = 2000
    parallel_chains: int = 4
    uncertainty_quantification: bool = True
    model_selection: bool = True
    regularization_strength: float = 0.01
    temporal_smoothing: bool = True
    real_time_monitoring: bool = False
    alert_threshold: float = 0.05  # m/day
    use_gpu: bool = True

class DeformationModel(ABC):
    """Abstract base class for deformation source models"""
    
    @abstractmethod
    def forward_model(self, parameters: Dict[str, float], coordinates: np.ndarray) -> np.ndarray:
        """Calculate surface deformation for given source parameters"""
        pass
    
    @abstractmethod
    def jacobian(self, parameters: Dict[str, float], coordinates: np.ndarray) -> np.ndarray:
        """Calculate Jacobian matrix for parameter estimation"""
        pass
    
    @abstractmethod
    def get_parameter_names(self) -> List[str]:
        """Get list of model parameter names"""
        pass
    
    @abstractmethod
    def get_default_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get default parameter bounds"""
        pass

class MogiModel(DeformationModel):
    """Mogi point pressure source model"""
    
    def forward_model(self, parameters: Dict[str, float], coordinates: np.ndarray) -> np.ndarray:
        """
        Calculate Mogi deformation
        
        Parameters:
        - x0, y0: source location (m)
        - depth: source depth (m, positive downward)
        - volume_change: volume change (m³)
        - poisson_ratio: Poisson's ratio
        """
        
        x0 = parameters['x0']
        y0 = parameters['y0']
        depth = parameters['depth']
        dV = parameters['volume_change']
        nu = parameters.get('poisson_ratio', 0.25)
        
        # Extract coordinates
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        
        # Calculate distances
        r = np.sqrt((x - x0)**2 + (y - y0)**2 + depth**2)
        
        # Mogi equations
        C = (1 - nu) * dV / (np.pi * depth)
        
        # Surface displacements
        ur = C * np.sqrt((x - x0)**2 + (y - y0)**2) / (r**3)
        uz = C * depth / (r**3)
        
        # Convert radial to east-north components
        distance_horizontal = np.sqrt((x - x0)**2 + (y - y0)**2)
        ue = np.where(distance_horizontal > 0, ur * (x - x0) / distance_horizontal, 0)
        un = np.where(distance_horizontal > 0, ur * (y - y0) / distance_horizontal, 0)
        
        return np.column_stack([ue, un, uz])
    
    def jacobian(self, parameters: Dict[str, float], coordinates: np.ndarray) -> np.ndarray:
        """Calculate Jacobian matrix for Mogi model"""
        
        x0 = parameters['x0']
        y0 = parameters['y0']
        depth = parameters['depth']
        dV = parameters['volume_change']
        nu = parameters.get('poisson_ratio', 0.25)
        
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        n_obs = len(x)
        
        # Initialize Jacobian
        n_params = 4  # x0, y0, depth, volume_change
        jacobian = np.zeros((n_obs * 3, n_params))
        
        # Calculate partial derivatives using finite differences
        eps = 1e-6
        base_deformation = self.forward_model(parameters, coordinates).flatten()
        
        param_list = ['x0', 'y0', 'depth', 'volume_change']
        
        for i, param_name in enumerate(param_list):
            params_perturbed = parameters.copy()
            params_perturbed[param_name] += eps
            
            perturbed_deformation = self.forward_model(params_perturbed, coordinates).flatten()
            jacobian[:, i] = (perturbed_deformation - base_deformation) / eps
        
        return jacobian
    
    def get_parameter_names(self) -> List[str]:
        return ['x0', 'y0', 'depth', 'volume_change']
    
    def get_default_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            'x0': (-10000, 10000),        # ±10 km
            'y0': (-10000, 10000),        # ±10 km
            'depth': (100, 20000),        # 0.1-20 km
            'volume_change': (-1e8, 1e8)  # ±100 million m³
        }

class SillModel(DeformationModel):
    """Horizontal sill intrusion model"""
    
    def forward_model(self, parameters: Dict[str, float], coordinates: np.ndarray) -> np.ndarray:
        """Calculate sill deformation using Okada formulation"""
        
        x0 = parameters['x0']
        y0 = parameters['y0']
        depth = parameters['depth']
        length = parameters['length']
        width = parameters['width']
        opening = parameters['opening']
        strike = parameters.get('strike', 0)  # degrees
        
        # Convert strike to radians
        strike_rad = np.radians(strike)
        
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        n_obs = len(x)
        
        # Rotate coordinates to sill coordinate system
        cos_strike = np.cos(strike_rad)
        sin_strike = np.sin(strike_rad)
        
        x_rot = cos_strike * (x - x0) + sin_strike * (y - y0)
        y_rot = -sin_strike * (x - x0) + cos_strike * (y - y0)
        
        # Sill dimensions
        a = length / 2
        b = width / 2
        
        # Initialize displacements
        ue = np.zeros(n_obs)
        un = np.zeros(n_obs)
        uz = np.zeros(n_obs)
        
        # Okada solutions for rectangular dislocation
        mu = 3e10  # Shear modulus (Pa)
        nu = 0.25  # Poisson's ratio
        
        for i in range(n_obs):
            xi = x_rot[i]
            eta = y_rot[i]
            
            # Integration over sill surface
            u_total = self._okada_sill_integration(xi, eta, depth, a, b, opening, mu, nu)
            
            ue[i] = u_total[0]
            un[i] = u_total[1]  
            uz[i] = u_total[2]
        
        # Rotate back to geographic coordinates
        ue_geo = cos_strike * ue - sin_strike * un
        un_geo = sin_strike * ue + cos_strike * un
        
        return np.column_stack([ue_geo, un_geo, uz])
    
    def _okada_sill_integration(self, xi: float, eta: float, depth: float,
                               a: float, b: float, opening: float,
                               mu: float, nu: float) -> np.ndarray:
        """Integrate Okada solution over sill surface"""
        
        # Simplified analytical solution for opening mode
        # Full implementation would use numerical integration
        
        R1 = np.sqrt(xi**2 + eta**2 + depth**2)
        R2 = np.sqrt((xi-a)**2 + eta**2 + depth**2)
        R3 = np.sqrt(xi**2 + (eta-b)**2 + depth**2)
        R4 = np.sqrt((xi-a)**2 + (eta-b)**2 + depth**2)
        
        # Opening mode displacement (simplified)
        factor = opening * (1 - nu) / (2 * np.pi * mu)
        
        ux = factor * depth * (np.arctan(eta/R1) - np.arctan(eta/R2) + 
                              np.arctan(eta/R3) - np.arctan(eta/R4))
        
        uy = factor * depth * (np.arctan(xi/R1) + np.arctan(xi/R2) - 
                              np.arctan(xi/R3) - np.arctan(xi/R4))
        
        uz = factor * (R1 + R2 - R3 - R4)
        
        return np.array([ux, uy, uz])
    
    def jacobian(self, parameters: Dict[str, float], coordinates: np.ndarray) -> np.ndarray:
        """Calculate Jacobian for sill model"""
        
        n_obs = len(coordinates)
        n_params = 6  # x0, y0, depth, length, width, opening
        jacobian = np.zeros((n_obs * 3, n_params))
        
        # Numerical derivatives
        eps = 1e-6
        base_deformation = self.forward_model(parameters, coordinates).flatten()
        
        param_list = ['x0', 'y0', 'depth', 'length', 'width', 'opening']
        
        for i, param_name in enumerate(param_list):
            params_perturbed = parameters.copy()
            params_perturbed[param_name] += eps
            
            perturbed_deformation = self.forward_model(params_perturbed, coordinates).flatten()
            jacobian[:, i] = (perturbed_deformation - base_deformation) / eps
        
        return jacobian
    
    def get_parameter_names(self) -> List[str]:
        return ['x0', 'y0', 'depth', 'length', 'width', 'opening']
    
    def get_default_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            'x0': (-10000, 10000),      # ±10 km
            'y0': (-10000, 10000),      # ±10 km
            'depth': (100, 10000),      # 0.1-10 km
            'length': (100, 10000),     # 0.1-10 km
            'width': (100, 10000),      # 0.1-10 km
            'opening': (-10, 10)        # ±10 m
        }

class DikeModel(DeformationModel):
    """Vertical dike intrusion model"""
    
    def forward_model(self, parameters: Dict[str, float], coordinates: np.ndarray) -> np.ndarray:
        """Calculate dike deformation using Okada formulation"""
        
        x0 = parameters['x0']
        y0 = parameters['y0']
        depth_top = parameters['depth_top']
        depth_bottom = parameters['depth_bottom']
        length = parameters['length']
        opening = parameters['opening']
        strike = parameters.get('strike', 0)  # degrees
        
        strike_rad = np.radians(strike)
        
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        n_obs = len(x)
        
        # Rotate coordinates
        cos_strike = np.cos(strike_rad)
        sin_strike = np.sin(strike_rad)
        
        x_rot = cos_strike * (x - x0) + sin_strike * (y - y0)
        y_rot = -sin_strike * (x - x0) + cos_strike * (y - y0)
        
        # Initialize displacements
        ue = np.zeros(n_obs)
        un = np.zeros(n_obs)
        uz = np.zeros(n_obs)
        
        # Okada parameters
        mu = 3e10  # Shear modulus (Pa)
        nu = 0.25  # Poisson's ratio
        
        for i in range(n_obs):
            xi = x_rot[i]
            eta = y_rot[i]
            
            # Dike deformation calculation
            u_total = self._okada_dike_integration(xi, eta, depth_top, depth_bottom, 
                                                 length, opening, mu, nu)
            
            ue[i] = u_total[0]
            un[i] = u_total[1]
            uz[i] = u_total[2]
        
        # Rotate back to geographic coordinates
        ue_geo = cos_strike * ue - sin_strike * un
        un_geo = sin_strike * ue + cos_strike * un
        
        return np.column_stack([ue_geo, un_geo, uz])
    
    def _okada_dike_integration(self, xi: float, eta: float, 
                               depth_top: float, depth_bottom: float,
                               length: float, opening: float,
                               mu: float, nu: float) -> np.ndarray:
        """Calculate dike displacement using Okada solution"""
        
        # Simplified analytical solution for tensile dislocation
        depth = (depth_top + depth_bottom) / 2
        height = depth_bottom - depth_top
        
        R = np.sqrt(xi**2 + eta**2 + depth**2)
        
        # Opening mode displacement (simplified)
        factor = opening * (1 - nu) / (np.pi * mu)
        
        ux = factor * xi * depth / (R**3)
        uy = factor * eta * depth / (R**3)
        uz = factor * height * depth / (R**3)
        
        return np.array([ux, uy, uz])
    
    def jacobian(self, parameters: Dict[str, float], coordinates: np.ndarray) -> np.ndarray:
        """Calculate Jacobian for dike model"""
        
        n_obs = len(coordinates)
        n_params = 6  # x0, y0, depth_top, depth_bottom, length, opening
        jacobian = np.zeros((n_obs * 3, n_params))
        
        # Numerical derivatives
        eps = 1e-6
        base_deformation = self.forward_model(parameters, coordinates).flatten()
        
        param_list = ['x0', 'y0', 'depth_top', 'depth_bottom', 'length', 'opening']
        
        for i, param_name in enumerate(param_list):
            params_perturbed = parameters.copy()
            params_perturbed[param_name] += eps
            
            perturbed_deformation = self.forward_model(params_perturbed, coordinates).flatten()
            jacobian[:, i] = (perturbed_deformation - base_deformation) / eps
        
        return jacobian
    
    def get_parameter_names(self) -> List[str]:
        return ['x0', 'y0', 'depth_top', 'depth_bottom', 'length', 'opening']
    
    def get_default_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            'x0': (-10000, 10000),         # ±10 km
            'y0': (-10000, 10000),         # ±10 km
            'depth_top': (100, 5000),      # 0.1-5 km
            'depth_bottom': (1000, 15000), # 1-15 km
            'length': (500, 20000),        # 0.5-20 km
            'opening': (-5, 5)             # ±5 m
        }

class YangModel(DeformationModel):
    """Yang finite spherical source model"""
    
    def forward_model(self, parameters: Dict[str, float], coordinates: np.ndarray) -> np.ndarray:
        """Calculate Yang model deformation"""
        
        x0 = parameters['x0']
        y0 = parameters['y0']
        depth = parameters['depth']
        radius = parameters['radius']
        pressure_change = parameters['pressure_change']
        
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        n_obs = len(x)
        
        # Material properties
        mu = 3e10  # Shear modulus (Pa)
        nu = 0.25  # Poisson's ratio
        
        # Calculate distances
        r_horizontal = np.sqrt((x - x0)**2 + (y - y0)**2)
        r_total = np.sqrt((x - x0)**2 + (y - y0)**2 + depth**2)
        
        # Yang model equations
        a = radius
        c = depth
        
        # Displacement calculations
        factor = pressure_change * a**3 / (mu * (1 - nu))
        
        # Radial displacement
        f1 = (3 - 4*nu) / (8*r_total**3)
        f2 = a**2 / (8*r_total**5)
        
        ur = factor * r_horizontal * (f1 + f2 * (3*r_horizontal**2 + 6*c**2))
        
        # Vertical displacement  
        uz = factor * c * (f1 + f2 * (3*c**2 + 6*r_horizontal**2))
        
        # Convert to east-north components
        ue = np.where(r_horizontal > 0, ur * (x - x0) / r_horizontal, 0)
        un = np.where(r_horizontal > 0, ur * (y - y0) / r_horizontal, 0)
        
        return np.column_stack([ue, un, uz])
    
    def jacobian(self, parameters: Dict[str, float], coordinates: np.ndarray) -> np.ndarray:
        """Calculate Jacobian for Yang model"""
        
        n_obs = len(coordinates)
        n_params = 5  # x0, y0, depth, radius, pressure_change
        jacobian = np.zeros((n_obs * 3, n_params))
        
        # Numerical derivatives
        eps = 1e-6
        base_deformation = self.forward_model(parameters, coordinates).flatten()
        
        param_list = ['x0', 'y0', 'depth', 'radius', 'pressure_change']
        
        for i, param_name in enumerate(param_list):
            params_perturbed = parameters.copy()
            params_perturbed[param_name] += eps
            
            perturbed_deformation = self.forward_model(params_perturbed, coordinates).flatten()
            jacobian[:, i] = (perturbed_deformation - base_deformation) / eps
        
        return jacobian
    
    def get_parameter_names(self) -> List[str]:
        return ['x0', 'y0', 'depth', 'radius', 'pressure_change']
    
    def get_default_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            'x0': (-10000, 10000),           # ±10 km
            'y0': (-10000, 10000),           # ±10 km
            'depth': (100, 20000),           # 0.1-20 km
            'radius': (10, 5000),            # 10 m - 5 km
            'pressure_change': (-1e8, 1e8)   # ±100 MPa
        }

class BayesianInversion:
    """Bayesian parameter estimation with MCMC sampling"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
    
    def invert(self, 
               observations: List[DeformationObservation],
               model: DeformationModel,
               prior_bounds: Dict[str, Tuple[float, float]]) -> Dict:
        """Perform Bayesian inversion using MCMC"""
        
        logger.info(f"Starting Bayesian inversion with {len(observations)} observations")
        
        # Prepare data
        coordinates, displacements, uncertainties = self._prepare_data(observations)
        
        # Initialize MCMC
        n_params = len(model.get_parameter_names())
        initial_params = self._initialize_parameters(model, prior_bounds)
        
        # Run MCMC chains
        chains = []
        log_likelihoods = []
        
        if self.config.parallel_chains > 1:
            # Parallel MCMC
            with ProcessPoolExecutor(max_workers=self.config.parallel_chains) as executor:
                futures = []
                
                for chain_id in range(self.config.parallel_chains):
                    future = executor.submit(
                        self._run_mcmc_chain,
                        chain_id, model, coordinates, displacements, 
                        uncertainties, initial_params, prior_bounds
                    )
                    futures.append(future)
                
                for future in futures:
                    chain_samples, chain_log_likes = future.result()
                    chains.append(chain_samples)
                    log_likelihoods.append(chain_log_likes)
        else:
            # Single chain
            chain_samples, chain_log_likes = self._run_mcmc_chain(
                0, model, coordinates, displacements, 
                uncertainties, initial_params, prior_bounds
            )
            chains.append(chain_samples)
            log_likelihoods.append(chain_log_likes)
        
        # Combine chains
        all_samples = np.concatenate(chains, axis=0)
        all_log_likes = np.concatenate(log_likelihoods)
        
        # Calculate statistics
        results = self._calculate_posterior_statistics(all_samples, all_log_likes, model)
        
        # Model selection criteria
        results['aic'] = self._calculate_aic(all_log_likes, n_params)
        results['bic'] = self._calculate_bic(all_log_likes, n_params, len(observations))
        
        return results
    
    def _prepare_data(self, observations: List[DeformationObservation]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare observation data for inversion"""
        
        n_obs = len(observations)
        coordinates = np.zeros((n_obs, 2))
        displacements = np.zeros((n_obs, 3))
        uncertainties = np.zeros((n_obs, 3))
        
        for i, obs in enumerate(observations):
            coordinates[i] = [obs.location[0], obs.location[1]]
            displacements[i] = obs.displacement
            uncertainties[i] = obs.uncertainty
        
        return coordinates, displacements, uncertainties
    
    def _initialize_parameters(self, 
                              model: DeformationModel,
                              prior_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Initialize parameters for MCMC"""
        
        initial_params = {}
        param_names = model.get_parameter_names()
        
        for param_name in param_names:
            if param_name in prior_bounds:
                low, high = prior_bounds[param_name]
                initial_params[param_name] = np.random.uniform(low, high)
            else:
                # Use default bounds
                default_bounds = model.get_default_bounds()
                low, high = default_bounds[param_name]
                initial_params[param_name] = np.random.uniform(low, high)
        
        return initial_params
    
    def _run_mcmc_chain(self,
                       chain_id: int,
                       model: DeformationModel,
                       coordinates: np.ndarray,
                       observations: np.ndarray,
                       uncertainties: np.ndarray,
                       initial_params: Dict[str, float],
                       prior_bounds: Dict[str, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Run single MCMC chain"""
        
        n_samples = self.config.mcmc_samples + self.config.burnin_samples
        n_params = len(initial_params)
        param_names = list(initial_params.keys())
        
        # Initialize chain
        samples = np.zeros((n_samples, n_params))
        log_likelihoods = np.zeros(n_samples)
        
        # Current state
        current_params = initial_params.copy()
        current_log_like = self._log_likelihood(current_params, model, coordinates, 
                                               observations, uncertainties)
        
        # Adaptive step sizes
        step_sizes = {name: (bounds[1] - bounds[0]) * 0.01 
                     for name, bounds in prior_bounds.items()}
        
        n_accepted = 0
        
        logger.info(f"Starting MCMC chain {chain_id}")
        
        for i in range(n_samples):
            # Propose new parameters
            proposed_params = current_params.copy()
            
            # Random walk proposal
            for param_name in param_names:
                proposed_params[param_name] += np.random.normal(0, step_sizes[param_name])
            
            # Check bounds
            if not self._check_bounds(proposed_params, prior_bounds):
                # Reject if out of bounds
                samples[i] = [current_params[name] for name in param_names]
                log_likelihoods[i] = current_log_like
                continue
            
            # Calculate likelihood
            proposed_log_like = self._log_likelihood(proposed_params, model, coordinates,
                                                   observations, uncertainties)
            
            # Metropolis acceptance
            log_alpha = proposed_log_like - current_log_like
            
            if log_alpha > 0 or np.random.random() < np.exp(log_alpha):
                # Accept
                current_params = proposed_params
                current_log_like = proposed_log_like
                n_accepted += 1
            
            # Store sample
            samples[i] = [current_params[name] for name in param_names]
            log_likelihoods[i] = current_log_like
            
            # Adapt step sizes
            if i > 0 and i % 1000 == 0:
                acceptance_rate = n_accepted / (i + 1)
                if acceptance_rate < 0.2:
                    # Decrease step sizes
                    for name in step_sizes:
                        step_sizes[name] *= 0.9
                elif acceptance_rate > 0.5:
                    # Increase step sizes
                    for name in step_sizes:
                        step_sizes[name] *= 1.1
        
        # Remove burn-in
        samples = samples[self.config.burnin_samples:]
        log_likelihoods = log_likelihoods[self.config.burnin_samples:]
        
        acceptance_rate = n_accepted / n_samples
        logger.info(f"Chain {chain_id} completed with acceptance rate: {acceptance_rate:.3f}")
        
        return samples, log_likelihoods
    
    def _log_likelihood(self,
                       parameters: Dict[str, float],
                       model: DeformationModel,
                       coordinates: np.ndarray,
                       observations: np.ndarray,
                       uncertainties: np.ndarray) -> float:
        """Calculate log-likelihood"""
        
        try:
            # Forward model
            predicted = model.forward_model(parameters, coordinates)
            
            # Calculate residuals
            residuals = observations - predicted
            
            # Log-likelihood calculation
            log_like = 0.0
            
            for i in range(len(observations)):
                for j in range(3):  # East, North, Up
                    sigma = uncertainties[i, j]
                    if sigma > 0:
                        log_like -= 0.5 * (residuals[i, j] / sigma)**2
                        log_like -= 0.5 * np.log(2 * np.pi * sigma**2)
            
            return log_like
            
        except Exception as e:
            # Return very low likelihood for invalid parameters
            return -1e10
    
    def _check_bounds(self, 
                     parameters: Dict[str, float], 
                     bounds: Dict[str, Tuple[float, float]]) -> bool:
        """Check if parameters are within bounds"""
        
        for param_name, value in parameters.items():
            if param_name in bounds:
                low, high = bounds[param_name]
                if value < low or value > high:
                    return False
        return True
    
    def _calculate_posterior_statistics(self,
                                      samples: np.ndarray,
                                      log_likelihoods: np.ndarray,
                                      model: DeformationModel) -> Dict:
        """Calculate posterior statistics"""
        
        param_names = model.get_parameter_names()
        n_params = len(param_names)
        
        # Best fit parameters (MAP estimate)
        best_idx = np.argmax(log_likelihoods)
        best_params = {param_names[i]: samples[best_idx, i] for i in range(n_params)}
        
        # Posterior means
        posterior_means = {param_names[i]: np.mean(samples[:, i]) for i in range(n_params)}
        
        # Posterior standard deviations
        posterior_stds = {param_names[i]: np.std(samples[:, i]) for i in range(n_params)}
        
        # Confidence intervals
        confidence_intervals = {}
        for i, param_name in enumerate(param_names):
            ci_lower = np.percentile(samples[:, i], 2.5)
            ci_upper = np.percentile(samples[:, i], 97.5)
            confidence_intervals[param_name] = (ci_lower, ci_upper)
        
        # Correlation matrix
        correlation_matrix = np.corrcoef(samples.T)
        
        return {
            'best_parameters': best_params,
            'posterior_means': posterior_means,
            'posterior_stds': posterior_stds,
            'confidence_intervals': confidence_intervals,
            'correlation_matrix': correlation_matrix,
            'samples': samples,
            'log_likelihoods': log_likelihoods,
            'max_log_likelihood': log_likelihoods[best_idx]
        }
    
    def _calculate_aic(self, log_likelihoods: np.ndarray, n_params: int) -> float:
        """Calculate Akaike Information Criterion"""
        max_log_like = np.max(log_likelihoods)
        return 2 * n_params - 2 * max_log_like
    
    def _calculate_bic(self, log_likelihoods: np.ndarray, n_params: int, n_obs: int) -> float:
        """Calculate Bayesian Information Criterion"""
        max_log_like = np.max(log_likelihoods)
        return n_params * np.log(n_obs) - 2 * max_log_like

class TimeSeriesAnalyzer:
    """Time series analysis for temporal deformation"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def analyze_temporal_evolution(self, 
                                 time_series_data: Dict[str, List[DeformationObservation]]) -> Dict:
        """Analyze temporal evolution of deformation"""
        
        results = {}
        
        for location_id, observations in time_series_data.items():
            # Sort observations by time
            observations.sort(key=lambda x: x.timestamp)
            
            # Extract time series
            times = np.array([obs.timestamp.timestamp() for obs in observations])
            displacements = np.array([obs.displacement for obs in observations])
            uncertainties = np.array([obs.uncertainty for obs in observations])
            
            # Analyze each component
            components = ['east', 'north', 'up']
            location_results = {}
            
            for i, component in enumerate(components):
                comp_data = displacements[:, i]
                comp_uncertainties = uncertainties[:, i]
                
                # Trend analysis
                trend_results = self._analyze_trend(times, comp_data, comp_uncertainties)
                
                # Seasonal analysis
                seasonal_results = self._analyze_seasonal(times, comp_data, comp_uncertainties)
                
                # Change point detection
                change_points = self._detect_change_points(times, comp_data, comp_uncertainties)
                
                # Acceleration detection
                acceleration = self._detect_acceleration(times, comp_data, comp_uncertainties)
                
                location_results[component] = {
                    'trend': trend_results,
                    'seasonal': seasonal_results,
                    'change_points': change_points,
                    'acceleration': acceleration
                }
            
            results[location_id] = location_results
        
        return results
    
    def _analyze_trend(self, times: np.ndarray, data: np.ndarray, uncertainties: np.ndarray) -> Dict:
        """Analyze linear and non-linear trends"""
        
        # Convert times to relative days
        times_rel = (times - times[0]) / (24 * 3600)  # Days since first observation
        
        # Linear trend
        weights = 1.0 / (uncertainties + 1e-6)
        linear_fit = np.polyfit(times_rel, data, 1, w=weights)
        linear_velocity = linear_fit[0]  # m/day
        
        # Quadratic trend
        quadratic_fit = np.polyfit(times_rel, data, 2, w=weights)
        acceleration = 2 * quadratic_fit[0]  # m/day²
        
        # R-squared values
        linear_pred = np.polyval(linear_fit, times_rel)
        quadratic_pred = np.polyval(quadratic_fit, times_rel)
        
        ss_tot = np.sum((data - np.mean(data))**2)
        ss_res_linear = np.sum((data - linear_pred)**2)
        ss_res_quadratic = np.sum((data - quadratic_pred)**2)
        
        r2_linear = 1 - ss_res_linear / ss_tot if ss_tot > 0 else 0
        r2_quadratic = 1 - ss_res_quadratic / ss_tot if ss_tot > 0 else 0
        
        # Statistical significance
        n = len(data)
        linear_std_error = np.sqrt(ss_res_linear / (n - 2)) / np.sqrt(np.sum((times_rel - np.mean(times_rel))**2))
        t_stat = linear_velocity / linear_std_error if linear_std_error > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
        
        return {
            'linear_velocity': linear_velocity,
            'acceleration': acceleration,
            'r2_linear': r2_linear,
            'r2_quadratic': r2_quadratic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def _analyze_seasonal(self, times: np.ndarray, data: np.ndarray, uncertainties: np.ndarray) -> Dict:
        """Analyze seasonal components"""
        
        # Convert to decimal years
        times_years = np.array([datetime.datetime.fromtimestamp(t).year + 
                               (datetime.datetime.fromtimestamp(t).timetuple().tm_yday - 1) / 365.25
                               for t in times])
        
        # Remove linear trend first
        detrended_data = data - np.polyval(np.polyfit(times_years, data, 1), times_years)
        
        # Fit seasonal model: A*sin(2πt) + B*cos(2πt) + C*sin(4πt) + D*cos(4πt)
        A = np.column_stack([
            np.sin(2 * np.pi * times_years),
            np.cos(2 * np.pi * times_years),
            np.sin(4 * np.pi * times_years),
            np.cos(4 * np.pi * times_years),
            np.ones(len(times_years))
        ])
        
        # Weighted least squares
        weights = 1.0 / (uncertainties + 1e-6)
        W = np.diag(weights)
        
        try:
            coeffs = inv(A.T @ W @ A) @ A.T @ W @ detrended_data
        except:
            # Fallback to unweighted
            coeffs = np.linalg.lstsq(A, detrended_data, rcond=None)[0]
        
        # Calculate seasonal amplitude and phase
        annual_amplitude = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
        annual_phase = np.arctan2(coeffs[1], coeffs[0])
        
        semi_annual_amplitude = np.sqrt(coeffs[2]**2 + coeffs[3]**2)
        semi_annual_phase = np.arctan2(coeffs[3], coeffs[2])
        
        # Seasonal prediction
        seasonal_pred = A @ coeffs
        
        # R-squared for seasonal model
        ss_tot = np.sum(detrended_data**2)
        ss_res = np.sum((detrended_data - seasonal_pred)**2)
        r2_seasonal = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'annual_amplitude': annual_amplitude,
            'annual_phase': annual_phase,
            'semi_annual_amplitude': semi_annual_amplitude,
            'semi_annual_phase': semi_annual_phase,
            'r2_seasonal': r2_seasonal,
            'coefficients': coeffs
        }
    
    def _detect_change_points(self, times: np.ndarray, data: np.ndarray, uncertainties: np.ndarray) -> List[Dict]:
        """Detect change points in time series"""
        
        if len(data) < 10:  # Need minimum data points
            return []
        
        change_points = []
        
        # Simple change point detection using cumulative sum
        residuals = data - np.mean(data)
        cumsum = np.cumsum(residuals)
        
        # Find potential change points
        threshold = 2 * np.std(cumsum)
        
        for i in range(3, len(data) - 3):  # Avoid edges
            if abs(cumsum[i]) > threshold:
                # Test significance of change point
                before = data[:i]
                after = data[i:]
                
                if len(before) > 2 and len(after) > 2:
                    # t-test for difference in means
                    t_stat, p_value = stats.ttest_ind(before, after)
                    
                    if p_value < 0.05:
                        change_points.append({
                            'time': times[i],
                            'index': i,
                            'magnitude': np.mean(after) - np.mean(before),
                            'p_value': p_value
                        })
        
        return change_points
    
    def _detect_acceleration(self, times: np.ndarray, data: np.ndarray, uncertainties: np.ndarray) -> Dict:
        """Detect acceleration in deformation"""
        
        # Convert times to relative days
        times_rel = (times - times[0]) / (24 * 3600)
        
        # Fit cubic polynomial to detect acceleration changes
        if len(data) >= 4:
            cubic_fit = np.polyfit(times_rel, data, 3)
            
            # Second derivative gives acceleration
            acceleration_func = lambda t: 6 * cubic_fit[0] * t + 2 * cubic_fit[1]
            
            # Calculate acceleration at different times
            accelerations = [acceleration_func(t) for t in times_rel]
            
            # Detect significant acceleration changes
            acceleration_threshold = 2 * np.std(accelerations)
            significant_acceleration = np.max(np.abs(accelerations)) > acceleration_threshold
            
            return {
                'has_acceleration': significant_acceleration,
                'max_acceleration': np.max(accelerations),
                'min_acceleration': np.min(accelerations),
                'acceleration_series': accelerations,
                'cubic_coefficients': cubic_fit
            }
        
        return {'has_acceleration': False}

class DeformationAnalyzer:
    """Main deformation analysis class"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.models = {
            SourceType.MOGI: MogiModel(),
            SourceType.SILL: SillModel(),
            SourceType.DIKE: DikeModel(),
            SourceType.YANG: YangModel()
        }
        self.bayesian_inverter = BayesianInversion(config)
        self.time_series_analyzer = TimeSeriesAnalyzer(config)
        
        # Model selection history
        self.inversion_history = []
    
    def analyze_deformation(self, 
                          observations: List[DeformationObservation],
                          source_types: List[SourceType] = None) -> Dict:
        """Complete deformation analysis"""
        
        logger.info(f"Analyzing {len(observations)} deformation observations")
        
        if source_types is None:
            source_types = [SourceType.MOGI, SourceType.SILL, SourceType.DIKE, SourceType.YANG]
        
        results = {}
        
        # Single source inversions
        single_source_results = {}
        
        for source_type in source_types:
            logger.info(f"Testing {source_type.value} model")
            
            model = self.models[source_type]
            prior_bounds = model.get_default_bounds()
            
            # Bayesian inversion
            inversion_results = self.bayesian_inverter.invert(observations, model, prior_bounds)
            
            # Create deformation source object
            source = DeformationSource(
                source_type=source_type,
                parameters=inversion_results['best_parameters'],
                parameter_bounds=prior_bounds,
                uncertainty=inversion_results['posterior_stds'],
                confidence_interval=inversion_results['confidence_intervals'],
                log_likelihood=inversion_results['max_log_likelihood'],
                aic=inversion_results['aic'],
                bic=inversion_results['bic']
            )
            
            single_source_results[source_type] = {
                'source': source,
                'inversion_results': inversion_results
            }
        
        # Model selection
        best_single_source = self._select_best_model(single_source_results)
        
        # Multi-source inversion if requested
        multi_source_results = None
        if self.config.max_sources > 1:
            multi_source_results = self._multi_source_inversion(observations, source_types)
        
        # Residual analysis
        residual_analysis = self._analyze_residuals(observations, best_single_source)
        
        # Compile results
        results = {
            'single_source_results': single_source_results,
            'best_single_source': best_single_source,
            'multi_source_results': multi_source_results,
            'residual_analysis': residual_analysis,
            'model_comparison': self._compare_models(single_source_results)
        }
        
        return results
    
    def analyze_temporal_deformation(self, 
                                   time_series_data: Dict[str, List[DeformationObservation]]) -> Dict:
        """Analyze temporal evolution of deformation"""
        
        logger.info("Analyzing temporal deformation evolution")
        
        # Time series analysis
        temporal_analysis = self.time_series_analyzer.analyze_temporal_evolution(time_series_data)
        
        # Detect significant changes
        alerts = self._generate_alerts(temporal_analysis)
        
        return {
            'temporal_analysis': temporal_analysis,
            'alerts': alerts
        }
    
    def _select_best_model(self, results: Dict[SourceType, Dict]) -> Dict:
        """Select best model using information criteria"""
        
        if not results:
            return None
        
        # Compare AIC values
        aic_values = {source_type: result['source'].aic for source_type, result in results.items()}
        best_aic = min(aic_values, key=aic_values.get)
        
        # Compare BIC values
        bic_values = {source_type: result['source'].bic for source_type, result in results.items()}
        best_bic = min(bic_values, key=bic_values.get)
        
        # Use BIC for final selection (more conservative)
        best_source_type = best_bic
        
        logger.info(f"Best model selected: {best_source_type.value} (BIC: {bic_values[best_source_type]:.2f})")
        
        return results[best_source_type]
    
    def _multi_source_inversion(self, 
                               observations: List[DeformationObservation],
                               source_types: List[SourceType]) -> Dict:
        """Multi-source deformation inversion"""
        
        logger.info("Performing multi-source inversion")
        
        # This is a simplified implementation
        # Full implementation would use more sophisticated algorithms
        
        results = {}
        
        # Try combinations of 2 sources
        from itertools import combinations
        
        for source_combination in combinations(source_types, 2):
            if len(source_combination) <= self.config.max_sources:
                try:
                    combination_result = self._invert_combined_sources(observations, source_combination)
                    results[source_combination] = combination_result
                except Exception as e:
                    logger.warning(f"Failed to invert {source_combination}: {e}")
        
        return results
    
    def _invert_combined_sources(self, 
                               observations: List[DeformationObservation],
                               source_types: Tuple[SourceType, ...]) -> Dict:
        """Invert for combined sources"""
        
        # This would implement multi-source inversion
        # For now, return placeholder
        return {
            'source_types': source_types,
            'message': 'Multi-source inversion not fully implemented'
        }
    
    def _analyze_residuals(self, 
                          observations: List[DeformationObservation],
                          best_model_result: Dict) -> Dict:
        """Analyze residuals from best-fit model"""
        
        if not best_model_result:
            return {}
        
        source = best_model_result['source']
        model = self.models[source.source_type]
        
        # Prepare data
        coordinates, observed_displacements, uncertainties = self.bayesian_inverter._prepare_data(observations)
        
        # Predict displacements
        predicted_displacements = model.forward_model(source.parameters, coordinates)
        
        # Calculate residuals
        residuals = observed_displacements - predicted_displacements
        
        # Residual statistics
        residual_stats = {}
        components = ['east', 'north', 'up']
        
        for i, component in enumerate(components):
            comp_residuals = residuals[:, i]
            comp_uncertainties = uncertainties[:, i]
            
            # Normalize by uncertainty
            normalized_residuals = comp_residuals / (comp_uncertainties + 1e-6)
            
            residual_stats[component] = {
                'mean': np.mean(comp_residuals),
                'std': np.std(comp_residuals),
                'rms': np.sqrt(np.mean(comp_residuals**2)),
                'max_abs': np.max(np.abs(comp_residuals)),
                'normalized_rms': np.sqrt(np.mean(normalized_residuals**2))
            }
        
        # Spatial correlation of residuals
        spatial_correlation = self._analyze_spatial_correlation(coordinates, residuals)
        
        return {
            'residuals': residuals,
            'statistics': residual_stats,
            'spatial_correlation': spatial_correlation
        }
    
    def _analyze_spatial_correlation(self, coordinates: np.ndarray, residuals: np.ndarray) -> Dict:
        """Analyze spatial correlation of residuals"""
        
        # Calculate distances between observation points
        distances = cdist(coordinates, coordinates)
        
        correlation_results = {}
        components = ['east', 'north', 'up']
        
        for i, component in enumerate(components):
            comp_residuals = residuals[:, i]
            
            # Calculate correlation as function of distance
            distance_bins = np.linspace(0, np.max(distances), 20)
            correlations = []
            
            for j in range(len(distance_bins) - 1):
                bin_mask = (distances >= distance_bins[j]) & (distances < distance_bins[j+1])
                
                if np.sum(bin_mask) > 10:  # Need sufficient pairs
                    bin_distances = distances[bin_mask]
                    bin_correlations = []
                    
                    for k in range(len(comp_residuals)):
                        for l in range(k+1, len(comp_residuals)):
                            if bin_mask[k, l]:
                                corr = comp_residuals[k] * comp_residuals[l]
                                bin_correlations.append(corr)
                    
                    correlations.append(np.mean(bin_correlations) if bin_correlations else 0)
                else:
                    correlations.append(0)
            
            correlation_results[component] = {
                'distance_bins': distance_bins[:-1],
                'correlations': correlations
            }
        
        return correlation_results
    
    def _compare_models(self, results: Dict[SourceType, Dict]) -> Dict:
        """Compare different models"""
        
        comparison = {}
        
        # Extract metrics
        for source_type, result in results.items():
            source = result['source']
            
            comparison[source_type.value] = {
                'aic': source.aic,
                'bic': source.bic,
                'log_likelihood': source.log_likelihood,
                'n_parameters': len(source.parameters)
            }
        
        # Calculate weights (Akaike weights)
        aic_values = np.array([comp['aic'] for comp in comparison.values()])
        min_aic = np.min(aic_values)
        
        weights = np.exp(-(aic_values - min_aic) / 2)
        weights = weights / np.sum(weights)
        
        source_names = list(comparison.keys())
        for i, source_name in enumerate(source_names):
            comparison[source_name]['weight'] = weights[i]
        
        return comparison
    
    def _generate_alerts(self, temporal_analysis: Dict) -> List[Dict]:
        """Generate alerts based on temporal analysis"""
        
        alerts = []
        
        for location_id, analysis in temporal_analysis.items():
            for component, comp_analysis in analysis.items():
                trend = comp_analysis['trend']
                change_points = comp_analysis['change_points']
                acceleration = comp_analysis['acceleration']
                
                # Velocity alert
                if abs(trend['linear_velocity']) > self.config.alert_threshold:
                    alerts.append({
                        'type': 'high_velocity',
                        'location': location_id,
                        'component': component,
                        'velocity': trend['linear_velocity'],
                        'significance': trend['p_value'],
                        'timestamp': datetime.datetime.now()
                    })
                
                # Change point alerts
                for cp in change_points:
                    if cp['p_value'] < 0.01:  # Highly significant
                        alerts.append({
                            'type': 'change_point',
                            'location': location_id,
                            'component': component,
                            'time': datetime.datetime.fromtimestamp(cp['time']),
                            'magnitude': cp['magnitude'],
                            'significance': cp['p_value'],
                            'timestamp': datetime.datetime.now()
                        })
                
                # Acceleration alert
                if acceleration.get('has_acceleration', False):
                    max_accel = abs(acceleration['max_acceleration'])
                    if max_accel > self.config.alert_threshold * 10:  # Acceleration threshold
                        alerts.append({
                            'type': 'acceleration',
                            'location': location_id,
                            'component': component,
                            'acceleration': max_accel,
                            'timestamp': datetime.datetime.now()
                        })
        
        return alerts
    
    def save_results(self, results: Dict, output_path: str):
        """Save analysis results"""
        
        # Convert to JSON-serializable format
        serializable_results = self._make_json_serializable(results)
        
        # Save to JSON
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {json_path}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj

def create_default_config() -> AnalysisConfig:
    """Create default analysis configuration"""
    return AnalysisConfig(
        max_sources=3,
        mcmc_samples=5000,
        burnin_samples=1000,
        parallel_chains=2,
        uncertainty_quantification=True,
        model_selection=True,
        regularization_strength=0.01,
        temporal_smoothing=True,
        real_time_monitoring=False,
        alert_threshold=0.01,  # 1 cm/day
        use_gpu=True
    )

async def main():
    """Example usage of deformation analyzer"""
    
    # Create configuration
    config = create_default_config()
    
    # Initialize analyzer
    analyzer = DeformationAnalyzer(config)
    
    # Create synthetic observations
    observations = []
    
    # Synthetic GPS network around a volcanic center
    center_lon, center_lat = 0.0, 0.0
    
    for i in range(20):
        # Random locations around center
        lon = center_lon + np.random.uniform(-0.1, 0.1)
        lat = center_lat + np.random.uniform(-0.1, 0.1)
        
        # Synthetic deformation (Mogi source)
        r = np.sqrt((lon * 111000)**2 + (lat * 111000)**2)  # Distance in meters
        depth = 3000  # 3 km depth
        volume_change = 1e6  # 1 million m³
        
        # Mogi equations
        R = np.sqrt(r**2 + depth**2)
        ur = 0.75 * volume_change * r / (np.pi * R**3)
        uz = 0.75 * volume_change * depth / (np.pi * R**3)
        
        # Add noise
        noise_level = 0.005  # 5 mm
        ue = ur * (lon * 111000) / r + np.random.normal(0, noise_level)
        un = ur * (lat * 111000) / r + np.random.normal(0, noise_level)
        uz += np.random.normal(0, noise_level)
        
        # Create observation
        obs = DeformationObservation(
            location=(lon, lat),
            displacement=(ue, un, uz),
            uncertainty=(noise_level, noise_level, noise_level),
            timestamp=datetime.datetime(2023, 1, 1),
            data_type="gps"
        )
        
        observations.append(obs)
    
    # Analyze deformation
    results = analyzer.analyze_deformation(observations)
    
    # Save results
    analyzer.save_results(results, "deformation_analysis_results.json")
    
    # Print best model
    best_source = results['best_single_source']['source']
    logger.info(f"Best model: {best_source.source_type.value}")
    logger.info(f"Parameters: {best_source.parameters}")
    logger.info(f"AIC: {best_source.aic:.2f}, BIC: {best_source.bic:.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())