"""
Advanced Hazard Interactions and Probabilistic Risk Assessment System
Comprehensive implementation for volcanic-seismic interactions and cascading hazard modeling
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.special import logsumexp, gamma, digamma
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import warnings
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import networkx as nx
import itertools
from collections import defaultdict, deque
import pickle
import h5py

# Advanced statistical libraries with fallbacks
try:
    import pymc3 as pm
    PYMC3_AVAILABLE = True
except ImportError:
    PYMC3_AVAILABLE = False
    warnings.warn("PyMC3 not available. Some Bayesian features will be limited.")

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from scipy.spatial.distance import cdist
    from scipy.cluster.hierarchy import linkage, fcluster
    SPATIAL_ANALYSIS_AVAILABLE = True
except ImportError:
    SPATIAL_ANALYSIS_AVAILABLE = False

warnings.filterwarnings('ignore')

class HazardType(Enum):
    VOLCANIC_ERUPTION = "volcanic_eruption"
    EARTHQUAKE = "earthquake"
    LANDSLIDE = "landslide"
    TSUNAMI = "tsunami"
    PYROCLASTIC_FLOW = "pyroclastic_flow"
    LAHAR = "lahar"
    GROUND_DEFORMATION = "ground_deformation"
    GAS_EMISSION = "gas_emission"

class InteractionType(Enum):
    TRIGGER = "trigger"
    AMPLIFY = "amplify"
    SUPPRESS = "suppress"
    SYNCHRONOUS = "synchronous"
    CASCADING = "cascading"

class ProbabilityDistributionType(Enum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    WEIBULL = "weibull"
    GAMMA = "gamma"
    BETA = "beta"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    PARETO = "pareto"

@dataclass
class HazardEvent:
    """Individual hazard event with spatio-temporal characteristics"""
    id: str
    hazard_type: HazardType
    location: np.ndarray  # [lat, lon, depth/elevation]
    magnitude: float
    timestamp: float
    duration: float
    intensity: float
    uncertainty: float = 0.1
    properties: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def spatial_distance(self, other: 'HazardEvent') -> float:
        """Calculate spatial distance to another event"""
        return np.linalg.norm(self.location - other.location)
    
    def temporal_distance(self, other: 'HazardEvent') -> float:
        """Calculate temporal distance to another event"""
        return abs(self.timestamp - other.timestamp)
    
    def magnitude_ratio(self, other: 'HazardEvent') -> float:
        """Calculate magnitude ratio with another event"""
        return self.magnitude / max(other.magnitude, 1e-6)

@dataclass
class HazardInteraction:
    """Interaction relationship between hazard types"""
    primary_hazard: HazardType
    secondary_hazard: HazardType
    interaction_type: InteractionType
    strength: float  # 0-1 interaction strength
    spatial_threshold: float  # km
    temporal_threshold: float  # hours
    magnitude_threshold: float
    probability_model: Optional[Callable] = None
    parameters: Dict[str, float] = field(default_factory=dict)

class ProbabilisticHazardModel:
    """Advanced probabilistic model for hazard occurrence and interactions"""
    
    def __init__(self, hazard_type: HazardType, config: Dict[str, Any] = None):
        self.hazard_type = hazard_type
        self.config = config or self._get_default_config()
        
        # Model parameters
        self.base_rate = self.config.get('base_rate', 0.1)  # events per year
        self.magnitude_distribution = None
        self.spatial_model = None
        self.temporal_model = None
        
        # Fitted parameters
        self.is_fitted = False
        self.parameters = {}
        
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for hazard model"""
        return {
            'base_rate': 0.1,
            'magnitude_dist_type': ProbabilityDistributionType.LOGNORMAL,
            'spatial_kernel': 'gaussian',
            'temporal_kernel': 'exponential',
            'max_magnitude': 10.0,
            'min_magnitude': 1.0,
            'spatial_correlation_length': 50.0,  # km
            'temporal_correlation_length': 365.25,  # days
            'uncertainty_quantification': True
        }
    
    def fit_magnitude_distribution(self, magnitudes: np.ndarray) -> Dict[str, float]:
        """Fit magnitude distribution to historical data"""
        try:
            dist_type = self.config['magnitude_dist_type']
            
            if dist_type == ProbabilityDistributionType.LOGNORMAL:
                # Fit log-normal distribution
                shape, loc, scale = stats.lognorm.fit(magnitudes, floc=0)
                self.magnitude_distribution = stats.lognorm(shape, loc, scale)
                params = {'shape': shape, 'loc': loc, 'scale': scale}
                
            elif dist_type == ProbabilityDistributionType.GAMMA:
                # Fit gamma distribution
                a, loc, scale = stats.gamma.fit(magnitudes, floc=0)
                self.magnitude_distribution = stats.gamma(a, loc, scale)
                params = {'a': a, 'loc': loc, 'scale': scale}
                
            elif dist_type == ProbabilityDistributionType.WEIBULL:
                # Fit Weibull distribution
                c, loc, scale = stats.weibull_min.fit(magnitudes, floc=0)
                self.magnitude_distribution = stats.weibull_min(c, loc, scale)
                params = {'c': c, 'loc': loc, 'scale': scale}
                
            else:
                # Default to normal distribution
                loc, scale = stats.norm.fit(magnitudes)
                self.magnitude_distribution = stats.norm(loc, scale)
                params = {'loc': loc, 'scale': scale}
            
            self.parameters['magnitude'] = params
            return params
            
        except Exception as e:
            self.logger.warning(f"Failed to fit magnitude distribution: {e}")
            # Fallback to normal distribution
            loc, scale = stats.norm.fit(magnitudes)
            self.magnitude_distribution = stats.norm(loc, scale)
            return {'loc': loc, 'scale': scale}
    
    def fit_spatial_model(self, locations: np.ndarray) -> Dict[str, Any]:
        """Fit spatial intensity model"""
        try:
            if len(locations) < 3:
                self.logger.warning("Insufficient data for spatial modeling")
                return {}
            
            # Gaussian Mixture Model for spatial clustering
            n_components = min(5, max(1, len(locations) // 10))
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(locations[:, :2])  # Use only lat/lon
            
            self.spatial_model = gmm
            
            # Extract parameters
            spatial_params = {
                'n_components': n_components,
                'means': gmm.means_.tolist(),
                'covariances': gmm.covariances_.tolist(),
                'weights': gmm.weights_.tolist()
            }
            
            self.parameters['spatial'] = spatial_params
            return spatial_params
            
        except Exception as e:
            self.logger.warning(f"Failed to fit spatial model: {e}")
            return {}
    
    def fit_temporal_model(self, timestamps: np.ndarray) -> Dict[str, float]:
        """Fit temporal occurrence model"""
        try:
            if len(timestamps) < 2:
                return {'rate': self.base_rate}
            
            # Calculate inter-event times
            sorted_times = np.sort(timestamps)
            inter_event_times = np.diff(sorted_times)
            
            # Fit exponential distribution to inter-event times
            rate = 1.0 / np.mean(inter_event_times)
            
            # Test for temporal clustering (Hawkes process parameters)
            # Simple estimation using maximum likelihood
            hawkes_params = self._estimate_hawkes_parameters(sorted_times)
            
            temporal_params = {
                'base_rate': rate,
                'hawkes_mu': hawkes_params.get('mu', rate),
                'hawkes_alpha': hawkes_params.get('alpha', 0.1),
                'hawkes_beta': hawkes_params.get('beta', 1.0)
            }
            
            self.parameters['temporal'] = temporal_params
            return temporal_params
            
        except Exception as e:
            self.logger.warning(f"Failed to fit temporal model: {e}")
            return {'rate': self.base_rate}
    
    def _estimate_hawkes_parameters(self, timestamps: np.ndarray) -> Dict[str, float]:
        """Estimate Hawkes process parameters"""
        try:
            # Simple MLE estimation for Hawkes process
            def neg_log_likelihood(params):
                mu, alpha, beta = params
                if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
                    return np.inf
                
                T = timestamps[-1] - timestamps[0]
                n = len(timestamps)
                
                # Compute intensity at each event time
                intensities = []
                for i, t in enumerate(timestamps):
                    base_intensity = mu
                    triggered_intensity = 0
                    
                    for j in range(i):
                        dt = t - timestamps[j]
                        if dt > 0:
                            triggered_intensity += alpha * np.exp(-beta * dt)
                    
                    intensities.append(base_intensity + triggered_intensity)
                
                # Log-likelihood calculation
                log_likelihood = np.sum(np.log(np.maximum(intensities, 1e-10)))
                log_likelihood -= mu * T
                log_likelihood -= alpha / beta * np.sum(1 - np.exp(-beta * (T - timestamps)))
                
                return -log_likelihood
            
            # Initial guess
            base_rate = len(timestamps) / (timestamps[-1] - timestamps[0])
            initial_params = [base_rate, 0.1, 1.0]
            
            # Optimize
            result = optimize.minimize(
                neg_log_likelihood, 
                initial_params,
                method='L-BFGS-B',
                bounds=[(1e-6, None), (0, 0.99), (1e-6, None)]
            )
            
            if result.success:
                mu, alpha, beta = result.x
                return {'mu': mu, 'alpha': alpha, 'beta': beta}
            else:
                return {'mu': base_rate, 'alpha': 0.1, 'beta': 1.0}
                
        except Exception as e:
            self.logger.warning(f"Failed to estimate Hawkes parameters: {e}")
            base_rate = len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else self.base_rate
            return {'mu': base_rate, 'alpha': 0.1, 'beta': 1.0}
    
    def predict_probability(self, location: np.ndarray, time: float, 
                          magnitude_range: Tuple[float, float] = None) -> Dict[str, float]:
        """Predict occurrence probability at given location and time"""
        if not self.is_fitted:
            return {'probability': self.base_rate, 'uncertainty': 1.0}
        
        try:
            # Temporal probability
            temporal_prob = self._calculate_temporal_probability(time)
            
            # Spatial probability
            spatial_prob = self._calculate_spatial_probability(location)
            
            # Magnitude probability
            if magnitude_range:
                magnitude_prob = self._calculate_magnitude_probability(magnitude_range)
            else:
                magnitude_prob = 1.0
            
            # Combined probability
            combined_prob = temporal_prob * spatial_prob * magnitude_prob
            
            # Uncertainty estimation
            uncertainty = self._estimate_prediction_uncertainty(location, time)
            
            return {
                'probability': combined_prob,
                'temporal_component': temporal_prob,
                'spatial_component': spatial_prob,
                'magnitude_component': magnitude_prob,
                'uncertainty': uncertainty
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {'probability': self.base_rate, 'uncertainty': 1.0}
    
    def _calculate_temporal_probability(self, time: float) -> float:
        """Calculate temporal probability component"""
        try:
            temporal_params = self.parameters.get('temporal', {})
            base_rate = temporal_params.get('base_rate', self.base_rate)
            
            # Simple exponential model
            return min(1.0, base_rate)
            
        except Exception:
            return self.base_rate
    
    def _calculate_spatial_probability(self, location: np.ndarray) -> float:
        """Calculate spatial probability component"""
        try:
            if self.spatial_model is None:
                return 1.0
            
            # Evaluate probability density
            log_prob = self.spatial_model.score_samples(location[:2].reshape(1, -1))
            prob = np.exp(log_prob[0])
            
            return min(1.0, prob)
            
        except Exception:
            return 1.0
    
    def _calculate_magnitude_probability(self, magnitude_range: Tuple[float, float]) -> float:
        """Calculate magnitude probability component"""
        try:
            if self.magnitude_distribution is None:
                return 1.0
            
            min_mag, max_mag = magnitude_range
            prob = self.magnitude_distribution.cdf(max_mag) - self.magnitude_distribution.cdf(min_mag)
            
            return max(0.0, min(1.0, prob))
            
        except Exception:
            return 1.0
    
    def _estimate_prediction_uncertainty(self, location: np.ndarray, time: float) -> float:
        """Estimate uncertainty in probability prediction"""
        # Simple uncertainty model based on data density
        base_uncertainty = 0.1
        
        # Add spatial uncertainty
        if hasattr(self, 'training_locations'):
            min_distance = np.min(np.linalg.norm(self.training_locations - location[:2], axis=1))
            spatial_uncertainty = min(1.0, min_distance / 100.0)  # Increase uncertainty with distance
        else:
            spatial_uncertainty = 0.5
        
        total_uncertainty = min(1.0, base_uncertainty + spatial_uncertainty)
        return total_uncertainty
    
    def fit(self, events: List[HazardEvent]) -> Dict[str, Any]:
        """Fit model to historical hazard events"""
        if not events:
            self.logger.warning("No events provided for model fitting")
            return {}
        
        # Extract data
        magnitudes = np.array([event.magnitude for event in events])
        locations = np.array([event.location for event in events])
        timestamps = np.array([event.timestamp for event in events])
        
        # Store for uncertainty estimation
        self.training_locations = locations[:, :2]
        
        # Fit components
        magnitude_params = self.fit_magnitude_distribution(magnitudes)
        spatial_params = self.fit_spatial_model(locations)
        temporal_params = self.fit_temporal_model(timestamps)
        
        self.is_fitted = True
        
        return {
            'magnitude_params': magnitude_params,
            'spatial_params': spatial_params,
            'temporal_params': temporal_params
        }

class VolcanicSeismicInteractionModel:
    """Specialized model for volcanic-seismic hazard interactions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Interaction parameters
        self.trigger_probability_model = None
        self.amplification_model = None
        self.spatial_coupling_model = None
        
        # Physical parameters
        self.stress_transfer_model = None
        self.magma_chamber_model = None
        
        self.is_fitted = False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'trigger_threshold': 0.1,  # Minimum probability for triggering
            'spatial_coupling_distance': 100.0,  # km
            'temporal_coupling_window': 72.0,  # hours
            'stress_threshold': 1e5,  # Pa
            'magnitude_amplification_factor': 1.2,
            'cascade_probability_decay': 0.8,
            'max_cascade_depth': 5
        }
    
    def calculate_stress_transfer(self, earthquake: HazardEvent, volcano_location: np.ndarray) -> Dict[str, float]:
        """Calculate Coulomb stress transfer from earthquake to volcano"""
        try:
            # Distance calculation
            distance = np.linalg.norm(earthquake.location - volcano_location)
            
            if distance == 0:
                distance = 1.0  # Avoid division by zero
            
            # Simplified stress calculation (Okada model approximation)
            magnitude = earthquake.magnitude
            
            # Empirical relationships
            rupture_length = 10**(magnitude - 5)  # km
            rupture_width = rupture_length / 2  # km
            slip = 10**(0.5 * magnitude - 2)  # m
            
            # Stress components
            shear_modulus = 30e9  # Pa
            
            # Normal stress (simplified)
            normal_stress = (shear_modulus * slip) / (2 * np.pi * distance * 1000)
            
            # Shear stress
            shear_stress = normal_stress * 0.6  # Empirical factor
            
            # Coulomb stress change
            friction_coefficient = 0.6
            coulomb_stress = shear_stress - friction_coefficient * normal_stress
            
            # Volumetric stress (relevant for magma chambers)
            volumetric_stress = -normal_stress / 3
            
            return {
                'coulomb_stress': coulomb_stress,
                'normal_stress': normal_stress,
                'shear_stress': shear_stress,
                'volumetric_stress': volumetric_stress,
                'distance': distance
            }
            
        except Exception as e:
            self.logger.error(f"Stress transfer calculation failed: {e}")
            return {'coulomb_stress': 0.0, 'normal_stress': 0.0, 'shear_stress': 0.0, 'volumetric_stress': 0.0, 'distance': float('inf')}
    
    def calculate_trigger_probability(self, earthquake: HazardEvent, volcano_location: np.ndarray,
                                   volcano_state: Dict[str, float] = None) -> Dict[str, float]:
        """Calculate probability of earthquake triggering volcanic eruption"""
        try:
            # Stress transfer
            stress_dict = self.calculate_stress_transfer(earthquake, volcano_location)
            coulomb_stress = stress_dict['coulomb_stress']
            distance = stress_dict['distance']
            
            # Base probability from stress
            stress_threshold = self.config['stress_threshold']
            stress_probability = 1.0 / (1.0 + np.exp(-(coulomb_stress - stress_threshold) / (stress_threshold * 0.1)))
            
            # Distance decay
            coupling_distance = self.config['spatial_coupling_distance']
            distance_factor = np.exp(-distance / coupling_distance)
            
            # Magnitude factor
            magnitude_factor = min(1.0, earthquake.magnitude / 7.0)
            
            # Volcano state factor
            if volcano_state:
                unrest_level = volcano_state.get('unrest_level', 0.5)
                magma_volume = volcano_state.get('magma_volume', 0.5)
                gas_pressure = volcano_state.get('gas_pressure', 0.5)
                
                volcano_readiness = (unrest_level + magma_volume + gas_pressure) / 3.0
            else:
                volcano_readiness = 0.5
            
            # Combined probability
            base_probability = stress_probability * distance_factor * magnitude_factor * volcano_readiness
            
            # Time-dependent factors
            temporal_factor = self._calculate_temporal_coupling_factor(earthquake.timestamp)
            
            final_probability = min(1.0, base_probability * temporal_factor)
            
            return {
                'trigger_probability': final_probability,
                'stress_component': stress_probability,
                'distance_component': distance_factor,
                'magnitude_component': magnitude_factor,
                'volcano_state_component': volcano_readiness,
                'temporal_component': temporal_factor,
                'stress_values': stress_dict
            }
            
        except Exception as e:
            self.logger.error(f"Trigger probability calculation failed: {e}")
            return {'trigger_probability': 0.0}
    
    def _calculate_temporal_coupling_factor(self, earthquake_time: float) -> float:
        """Calculate temporal coupling factor"""
        # Simple exponential decay model
        current_time = time.time()
        time_since_earthquake = current_time - earthquake_time
        
        coupling_window = self.config['temporal_coupling_window'] * 3600  # Convert to seconds
        
        if time_since_earthquake < 0:
            return 0.0  # Future earthquake
        
        temporal_factor = np.exp(-time_since_earthquake / coupling_window)
        return temporal_factor
    
    def predict_eruption_characteristics(self, trigger_earthquake: HazardEvent,
                                       volcano_properties: Dict[str, float]) -> Dict[str, float]:
        """Predict characteristics of triggered eruption"""
        try:
            # Base eruption magnitude from volcano properties
            base_vei = volcano_properties.get('historical_vei', 2.0)
            
            # Earthquake influence on eruption magnitude
            earthquake_magnitude_factor = min(2.0, trigger_earthquake.magnitude / 7.0)
            stress_influence = self.calculate_stress_transfer(trigger_earthquake, 
                                                           volcano_properties.get('location', np.zeros(3)))
            stress_magnitude_factor = min(1.5, abs(stress_influence['coulomb_stress']) / self.config['stress_threshold'])
            
            # Predicted VEI
            predicted_vei = base_vei * (1.0 + 0.3 * earthquake_magnitude_factor + 0.2 * stress_magnitude_factor)
            predicted_vei = min(8.0, max(1.0, predicted_vei))
            
            # Duration estimation
            base_duration = volcano_properties.get('historical_duration', 24.0)  # hours
            duration_factor = 1.0 + 0.5 * earthquake_magnitude_factor
            predicted_duration = base_duration * duration_factor
            
            # Eruption style probability
            explosive_probability = min(1.0, 0.3 + 0.4 * earthquake_magnitude_factor)
            
            return {
                'predicted_vei': predicted_vei,
                'predicted_duration': predicted_duration,
                'explosive_probability': explosive_probability,
                'effusive_probability': 1.0 - explosive_probability,
                'uncertainty': 0.3  # Base uncertainty
            }
            
        except Exception as e:
            self.logger.error(f"Eruption characteristics prediction failed: {e}")
            return {
                'predicted_vei': 2.0,
                'predicted_duration': 24.0,
                'explosive_probability': 0.5,
                'effusive_probability': 0.5,
                'uncertainty': 0.8
            }

class BayesianHazardNetwork:
    """Bayesian network for modeling hazard dependencies"""
    
    def __init__(self, hazard_types: List[HazardType], config: Dict[str, Any] = None):
        self.hazard_types = hazard_types
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Network structure
        self.network = nx.DiGraph()
        self.conditional_probabilities = {}
        self.marginal_probabilities = {}
        
        # Evidence and inference
        self.evidence = {}
        self.inference_engine = None
        
        self._initialize_network()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'learning_method': 'maximum_likelihood',
            'structure_learning': True,
            'smoothing_parameter': 1.0,
            'discretization_bins': 5,
            'temporal_window': 24.0,  # hours
            'spatial_resolution': 10.0,  # km
        }
    
    def _initialize_network(self):
        """Initialize Bayesian network structure"""
        # Add nodes for each hazard type
        for hazard_type in self.hazard_types:
            self.network.add_node(hazard_type.value)
        
        # Initialize with expert knowledge edges
        expert_edges = self._get_expert_knowledge_edges()
        for parent, child, weight in expert_edges:
            if parent in [h.value for h in self.hazard_types] and child in [h.value for h in self.hazard_types]:
                self.network.add_edge(parent, child, weight=weight)
    
    def _get_expert_knowledge_edges(self) -> List[Tuple[str, str, float]]:
        """Define expert knowledge dependencies"""
        return [
            (HazardType.EARTHQUAKE.value, HazardType.VOLCANIC_ERUPTION.value, 0.7),
            (HazardType.EARTHQUAKE.value, HazardType.LANDSLIDE.value, 0.8),
            (HazardType.EARTHQUAKE.value, HazardType.TSUNAMI.value, 0.9),
            (HazardType.VOLCANIC_ERUPTION.value, HazardType.PYROCLASTIC_FLOW.value, 0.95),
            (HazardType.VOLCANIC_ERUPTION.value, HazardType.LAHAR.value, 0.6),
            (HazardType.VOLCANIC_ERUPTION.value, HazardType.GAS_EMISSION.value, 0.9),
            (HazardType.PYROCLASTIC_FLOW.value, HazardType.LAHAR.value, 0.4),
            (HazardType.GROUND_DEFORMATION.value, HazardType.VOLCANIC_ERUPTION.value, 0.5),
            (HazardType.GAS_EMISSION.value, HazardType.VOLCANIC_ERUPTION.value, 0.6),
        ]
    
    def learn_structure(self, events: List[HazardEvent]) -> nx.DiGraph:
        """Learn Bayesian network structure from data"""
        try:
            if not self.config['structure_learning'] or len(events) < 10:
                return self.network
            
            # Create event correlation matrix
            correlation_matrix = self._calculate_event_correlations(events)
            
            # Structure learning using correlation-based approach
            threshold = 0.3  # Correlation threshold for edge creation
            
            for i, hazard1 in enumerate(self.hazard_types):
                for j, hazard2 in enumerate(self.hazard_types):
                    if i != j and correlation_matrix[i, j] > threshold:
                        # Add edge if correlation is significant
                        self.network.add_edge(hazard1.value, hazard2.value, 
                                            weight=correlation_matrix[i, j])
            
            self.logger.info(f"Learned network structure with {self.network.number_of_edges()} edges")
            return self.network
            
        except Exception as e:
            self.logger.error(f"Structure learning failed: {e}")
            return self.network
    
    def _calculate_event_correlations(self, events: List[HazardEvent]) -> np.ndarray:
        """Calculate pairwise correlations between hazard types"""
        n_hazards = len(self.hazard_types)
        correlation_matrix = np.zeros((n_hazards, n_hazards))
        
        try:
            # Create time series for each hazard type
            time_resolution = 24.0  # hours
            min_time = min(event.timestamp for event in events)
            max_time = max(event.timestamp for event in events)
            
            n_time_steps = int((max_time - min_time) / (time_resolution * 3600)) + 1
            time_series = np.zeros((n_hazards, n_time_steps))
            
            # Fill time series
            for event in events:
                hazard_idx = self.hazard_types.index(event.hazard_type)
                time_idx = int((event.timestamp - min_time) / (time_resolution * 3600))
                if 0 <= time_idx < n_time_steps:
                    time_series[hazard_idx, time_idx] += event.magnitude
            
            # Calculate correlations
            for i in range(n_hazards):
                for j in range(n_hazards):
                    if i != j:
                        correlation = np.corrcoef(time_series[i], time_series[j])[0, 1]
                        if not np.isnan(correlation):
                            correlation_matrix[i, j] = max(0, correlation)
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Correlation calculation failed: {e}")
            return np.zeros((n_hazards, n_hazards))
    
    def learn_parameters(self, events: List[HazardEvent]) -> Dict[str, Dict[str, float]]:
        """Learn conditional probability parameters"""
        try:
            # Group events by hazard type and discretize
            hazard_events = defaultdict(list)
            for event in events:
                hazard_events[event.hazard_type.value].append(event)
            
            # Learn marginal probabilities
            total_events = len(events)
            for hazard_type in self.hazard_types:
                count = len(hazard_events[hazard_type.value])
                self.marginal_probabilities[hazard_type.value] = count / total_events
            
            # Learn conditional probabilities for each edge
            for parent, child in self.network.edges():
                cpt = self._learn_conditional_probability_table(events, parent, child)
                self.conditional_probabilities[(parent, child)] = cpt
            
            return self.conditional_probabilities
            
        except Exception as e:
            self.logger.error(f"Parameter learning failed: {e}")
            return {}
    
    def _learn_conditional_probability_table(self, events: List[HazardEvent], 
                                           parent: str, child: str) -> Dict[str, float]:
        """Learn conditional probability table for parent->child relationship"""
        try:
            # Find parent and child events within temporal window
            temporal_window = self.config['temporal_window'] * 3600  # Convert to seconds
            spatial_threshold = self.config['spatial_resolution'] * 1000  # Convert to meters
            
            parent_events = [e for e in events if e.hazard_type.value == parent]
            child_events = [e for e in events if e.hazard_type.value == child]
            
            # Count conditional occurrences
            conditional_counts = defaultdict(int)
            parent_counts = defaultdict(int)
            
            for parent_event in parent_events:
                # Discretize parent event
                parent_state = self._discretize_event(parent_event)
                parent_counts[parent_state] += 1
                
                # Look for child events within window
                triggered_child = False
                for child_event in child_events:
                    temporal_diff = child_event.timestamp - parent_event.timestamp
                    spatial_diff = parent_event.spatial_distance(child_event)
                    
                    if 0 <= temporal_diff <= temporal_window and spatial_diff <= spatial_threshold:
                        triggered_child = True
                        break
                
                if triggered_child:
                    conditional_counts[parent_state] += 1
            
            # Calculate conditional probabilities with smoothing
            smoothing = self.config['smoothing_parameter']
            cpt = {}
            
            for state in parent_counts:
                count_triggered = conditional_counts[state]
                count_parent = parent_counts[state]
                
                # Laplace smoothing
                prob = (count_triggered + smoothing) / (count_parent + 2 * smoothing)
                cpt[state] = prob
            
            return cpt
            
        except Exception as e:
            self.logger.error(f"CPT learning failed for {parent}->{child}: {e}")
            return {'default': 0.1}
    
    def _discretize_event(self, event: HazardEvent) -> str:
        """Discretize event into categorical state"""
        # Simple discretization based on magnitude
        magnitude = event.magnitude
        
        if magnitude < 2.0:
            mag_category = 'low'
        elif magnitude < 5.0:
            mag_category = 'medium'
        else:
            mag_category = 'high'
        
        return f"{event.hazard_type.value}_{mag_category}"
    
    def infer_probabilities(self, evidence: Dict[str, str]) -> Dict[str, float]:
        """Perform Bayesian inference given evidence"""
        try:
            self.evidence = evidence
            
            # Simple approximate inference using belief propagation
            marginals = {}
            
            for hazard_type in self.hazard_types:
                hazard_name = hazard_type.value
                
                if hazard_name in evidence:
                    # Evidence node
                    marginals[hazard_name] = 1.0
                else:
                    # Calculate marginal probability
                    prob = self._calculate_marginal_probability(hazard_name, evidence)
                    marginals[hazard_name] = prob
            
            return marginals
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return {h.value: 0.1 for h in self.hazard_types}
    
    def _calculate_marginal_probability(self, target: str, evidence: Dict[str, str]) -> float:
        """Calculate marginal probability using approximate inference"""
        try:
            # Start with prior probability
            prior = self.marginal_probabilities.get(target, 0.1)
            
            # Adjust based on parent nodes with evidence
            parents = list(self.network.predecessors(target))
            
            if not parents:
                return prior
            
            # Simple approach: multiply by conditional probabilities
            posterior = prior
            
            for parent in parents:
                if parent in evidence:
                    parent_state = evidence[parent]
                    cpt_key = (parent, target)
                    
                    if cpt_key in self.conditional_probabilities:
                        cpt = self.conditional_probabilities[cpt_key]
                        conditional_prob = cpt.get(parent_state, cpt.get('default', 0.1))
                        posterior *= conditional_prob
            
            return min(1.0, posterior)
            
        except Exception as e:
            self.logger.error(f"Marginal probability calculation failed: {e}")
            return 0.1

class MonteCarloHazardSimulator:
    """Monte Carlo simulation system for hazard scenarios"""
    
    def __init__(self, hazard_models: Dict[HazardType, ProbabilisticHazardModel],
                 interaction_models: Dict[Tuple[HazardType, HazardType], Any],
                 config: Dict[str, Any] = None):
        self.hazard_models = hazard_models
        self.interaction_models = interaction_models
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Simulation state
        self.simulation_results = []
        self.scenario_cache = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default simulation configuration"""
        return {
            'n_simulations': 10000,
            'time_horizon': 365.25,  # days
            'time_step': 1.0,  # days
            'spatial_grid_size': 10.0,  # km
            'max_events_per_simulation': 1000,
            'parallel_processing': True,
            'random_seed': 42,
            'output_percentiles': [5, 25, 50, 75, 95],
            'save_individual_scenarios': False
        }
    
    async def run_monte_carlo_simulation(self, region_bounds: Dict[str, float],
                                       initial_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run Monte Carlo hazard simulation"""
        try:
            n_simulations = self.config['n_simulations']
            
            if self.config['parallel_processing']:
                # Parallel execution
                with ProcessPoolExecutor(max_workers=4) as executor:
                    futures = []
                    
                    for sim_id in range(n_simulations):
                        future = executor.submit(
                            self._run_single_simulation,
                            sim_id, region_bounds, initial_conditions
                        )
                        futures.append(future)
                    
                    # Collect results
                    simulation_results = []
                    for future in futures:
                        try:
                            result = future.result(timeout=60)
                            if result:
                                simulation_results.append(result)
                        except Exception as e:
                            self.logger.warning(f"Simulation failed: {e}")
            else:
                # Sequential execution
                simulation_results = []
                for sim_id in range(n_simulations):
                    result = self._run_single_simulation(sim_id, region_bounds, initial_conditions)
                    if result:
                        simulation_results.append(result)
            
            self.simulation_results = simulation_results
            
            # Analyze results
            analysis = self._analyze_simulation_results(simulation_results)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simulation failed: {e}")
            return {}
    
    def _run_single_simulation(self, sim_id: int, region_bounds: Dict[str, float],
                              initial_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run single Monte Carlo simulation"""
        try:
            # Set random seed for reproducibility
            np.random.seed(self.config['random_seed'] + sim_id)
            
            # Initialize simulation
            current_time = 0.0
            max_time = self.config['time_horizon'] * 24 * 3600  # Convert to seconds
            time_step = self.config['time_step'] * 24 * 3600  # Convert to seconds
            
            events = []
            cascade_events = []
            
            # Main simulation loop
            while current_time < max_time and len(events) < self.config['max_events_per_simulation']:
                # Generate primary events
                for hazard_type, model in self.hazard_models.items():
                    if model.is_fitted:
                        # Sample event occurrence
                        occurrence_prob = self._sample_event_occurrence(model, current_time, region_bounds)
                        
                        if np.random.random() < occurrence_prob:
                            # Generate event
                            event = self._generate_event(hazard_type, model, current_time, region_bounds)
                            events.append(event)
                            
                            # Check for cascading events
                            triggered_events = self._simulate_cascading_events(event, current_time)
                            cascade_events.extend(triggered_events)
                
                current_time += time_step
            
            # Combine all events
            all_events = events + cascade_events
            
            # Calculate scenario metrics
            scenario_metrics = self._calculate_scenario_metrics(all_events, region_bounds)
            
            return {
                'simulation_id': sim_id,
                'events': all_events,
                'metrics': scenario_metrics,
                'total_events': len(all_events),
                'primary_events': len(events),
                'cascading_events': len(cascade_events)
            }
            
        except Exception as e:
            self.logger.error(f"Single simulation {sim_id} failed: {e}")
            return None
    
    def _sample_event_occurrence(self, model: ProbabilisticHazardModel, 
                               current_time: float, region_bounds: Dict[str, float]) -> float:
        """Sample event occurrence probability"""
        try:
            # Simple Poisson process sampling
            temporal_params = model.parameters.get('temporal', {})
            base_rate = temporal_params.get('base_rate', 0.1)  # events per year
            
            # Convert to probability for time step
            time_step_years = self.config['time_step'] / 365.25
            occurrence_prob = base_rate * time_step_years
            
            return min(1.0, occurrence_prob)
            
        except Exception:
            return 0.001  # Fallback probability
    
    def _generate_event(self, hazard_type: HazardType, model: ProbabilisticHazardModel,
                       current_time: float, region_bounds: Dict[str, float]) -> HazardEvent:
        """Generate synthetic hazard event"""
        try:
            # Sample location
            lat_min, lat_max = region_bounds.get('lat_min', 0), region_bounds.get('lat_max', 1)
            lon_min, lon_max = region_bounds.get('lon_min', 0), region_bounds.get('lon_max', 1)
            depth_min, depth_max = region_bounds.get('depth_min', 0), region_bounds.get('depth_max', 10)
            
            if model.spatial_model:
                # Sample from spatial model
                location_sample = model.spatial_model.sample(1)[0]
                location = np.array([
                    np.clip(location_sample[0], lat_min, lat_max),
                    np.clip(location_sample[1], lon_min, lon_max),
                    np.random.uniform(depth_min, depth_max)
                ])
            else:
                # Uniform sampling
                location = np.array([
                    np.random.uniform(lat_min, lat_max),
                    np.random.uniform(lon_min, lon_max),
                    np.random.uniform(depth_min, depth_max)
                ])
            
            # Sample magnitude
            if model.magnitude_distribution:
                magnitude = model.magnitude_distribution.rvs(1)[0]
                magnitude = max(model.config['min_magnitude'], 
                              min(model.config['max_magnitude'], magnitude))
            else:
                magnitude = np.random.uniform(model.config['min_magnitude'], 
                                           model.config['max_magnitude'])
            
            # Generate other properties
            intensity = magnitude * np.random.uniform(0.8, 1.2)
            duration = self._estimate_event_duration(hazard_type, magnitude)
            
            event = HazardEvent(
                id=f"{hazard_type.value}_{int(current_time)}_{np.random.randint(1000)}",
                hazard_type=hazard_type,
                location=location,
                magnitude=magnitude,
                timestamp=current_time,
                duration=duration,
                intensity=intensity,
                uncertainty=0.2
            )
            
            return event
            
        except Exception as e:
            self.logger.error(f"Event generation failed: {e}")
            # Return default event
            return HazardEvent(
                id=f"default_{int(current_time)}",
                hazard_type=hazard_type,
                location=np.array([0, 0, 0]),
                magnitude=2.0,
                timestamp=current_time,
                duration=3600,
                intensity=2.0,
                uncertainty=0.5
            )
    
    def _estimate_event_duration(self, hazard_type: HazardType, magnitude: float) -> float:
        """Estimate event duration based on type and magnitude"""
        duration_models = {
            HazardType.EARTHQUAKE: lambda m: 10 + m * 20,  # seconds
            HazardType.VOLCANIC_ERUPTION: lambda m: 3600 * (m ** 2),  # hours to seconds
            HazardType.LANDSLIDE: lambda m: 60 + m * 300,  # seconds
            HazardType.TSUNAMI: lambda m: 3600 + m * 1800,  # seconds
            HazardType.PYROCLASTIC_FLOW: lambda m: 300 + m * 600,  # seconds
        }
        
        duration_func = duration_models.get(hazard_type, lambda m: 3600)
        return duration_func(magnitude)
    
    def _simulate_cascading_events(self, trigger_event: HazardEvent, 
                                 current_time: float) -> List[HazardEvent]:
        """Simulate cascading events triggered by primary event"""
        cascading_events = []
        
        try:
            # Look for applicable interaction models
            for (primary_hazard, secondary_hazard), interaction_model in self.interaction_models.items():
                if primary_hazard == trigger_event.hazard_type:
                    # Check if cascading event should occur
                    if hasattr(interaction_model, 'calculate_trigger_probability'):
                        # Volcanic-seismic interaction
                        volcano_location = trigger_event.location + np.random.normal(0, 10, 3)  # Nearby volcano
                        trigger_result = interaction_model.calculate_trigger_probability(
                            trigger_event, volcano_location
                        )
                        trigger_prob = trigger_result.get('trigger_probability', 0.0)
                        
                        if np.random.random() < trigger_prob:
                            # Generate cascading event
                            cascading_event = self._generate_cascading_event(
                                secondary_hazard, trigger_event, trigger_result, current_time
                            )
                            cascading_events.append(cascading_event)
                    else:
                        # Generic interaction model
                        trigger_prob = 0.1  # Default probability
                        if np.random.random() < trigger_prob:
                            cascading_event = self._generate_cascading_event(
                                secondary_hazard, trigger_event, {}, current_time
                            )
                            cascading_events.append(cascading_event)
            
            return cascading_events
            
        except Exception as e:
            self.logger.error(f"Cascading event simulation failed: {e}")
            return []
    
    def _generate_cascading_event(self, hazard_type: HazardType, trigger_event: HazardEvent,
                                trigger_result: Dict[str, float], current_time: float) -> HazardEvent:
        """Generate cascading event from trigger"""
        try:
            # Location near trigger event
            location_offset = np.random.normal(0, 5, 3)  # 5 km standard deviation
            location = trigger_event.location + location_offset
            
            # Magnitude influenced by trigger
            if 'predicted_vei' in trigger_result:
                magnitude = trigger_result['predicted_vei']
            else:
                magnitude_factor = np.random.uniform(0.5, 1.2)
                magnitude = trigger_event.magnitude * magnitude_factor
            
            # Timing - small delay after trigger
            delay = np.random.exponential(3600)  # Exponential delay, mean 1 hour
            timestamp = current_time + delay
            
            # Duration
            duration = self._estimate_event_duration(hazard_type, magnitude)
            
            cascading_event = HazardEvent(
                id=f"cascade_{hazard_type.value}_{int(timestamp)}",
                hazard_type=hazard_type,
                location=location,
                magnitude=magnitude,
                timestamp=timestamp,
                duration=duration,
                intensity=magnitude,
                uncertainty=0.3,
                metadata={'trigger_event_id': trigger_event.id, 'is_cascading': True}
            )
            
            return cascading_event
            
        except Exception as e:
            self.logger.error(f"Cascading event generation failed: {e}")
            return HazardEvent(
                id=f"cascade_default_{int(current_time)}",
                hazard_type=hazard_type,
                location=trigger_event.location,
                magnitude=trigger_event.magnitude * 0.8,
                timestamp=current_time + 3600,
                duration=3600,
                intensity=1.0,
                uncertainty=0.5
            )
    
    def _calculate_scenario_metrics(self, events: List[HazardEvent], 
                                  region_bounds: Dict[str, float]) -> Dict[str, float]:
        """Calculate metrics for scenario"""
        if not events:
            return {}
        
        try:
            metrics = {}
            
            # Basic statistics
            metrics['total_events'] = len(events)
            metrics['total_energy'] = sum(event.magnitude ** 2 for event in events)
            metrics['max_magnitude'] = max(event.magnitude for event in events)
            metrics['avg_magnitude'] = np.mean([event.magnitude for event in events])
            
            # Temporal metrics
            event_times = [event.timestamp for event in events]
            metrics['time_span'] = max(event_times) - min(event_times)
            metrics['event_rate'] = len(events) / (metrics['time_span'] / (24 * 3600)) if metrics['time_span'] > 0 else 0
            
            # Spatial metrics
            locations = np.array([event.location[:2] for event in events])
            if len(locations) > 1:
                distances = cdist(locations, locations)
                metrics['spatial_clustering'] = np.mean(np.min(distances + np.eye(len(distances)) * 1000, axis=1))
            else:
                metrics['spatial_clustering'] = 0
            
            # Hazard-specific metrics
            hazard_counts = defaultdict(int)
            for event in events:
                hazard_counts[event.hazard_type.value] += 1
            
            for hazard_type, count in hazard_counts.items():
                metrics[f'{hazard_type}_count'] = count
            
            # Risk metrics
            metrics['high_magnitude_events'] = sum(1 for event in events if event.magnitude >= 6.0)
            metrics['cascading_events'] = sum(1 for event in events if event.metadata.get('is_cascading', False))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Scenario metrics calculation failed: {e}")
            return {'total_events': len(events)}
    
    def _analyze_simulation_results(self, simulation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results"""
        try:
            if not simulation_results:
                return {}
            
            # Extract metrics from all simulations
            all_metrics = []
            for result in simulation_results:
                all_metrics.append(result['metrics'])
            
            # Statistical analysis
            analysis = {}
            percentiles = self.config['output_percentiles']
            
            # Aggregate metrics
            metric_names = set()
            for metrics in all_metrics:
                metric_names.update(metrics.keys())
            
            for metric_name in metric_names:
                values = [metrics.get(metric_name, 0) for metrics in all_metrics]
                
                analysis[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'percentiles': {p: np.percentile(values, p) for p in percentiles}
                }
            
            # Event type analysis
            hazard_type_stats = {}
            for hazard_type in HazardType:
                hazard_key = f'{hazard_type.value}_count'
                if hazard_key in metric_names:
                    values = [metrics.get(hazard_key, 0) for metrics in all_metrics]
                    hazard_type_stats[hazard_type.value] = {
                        'occurrence_probability': sum(1 for v in values if v > 0) / len(values),
                        'mean_count': np.mean(values),
                        'max_count': max(values)
                    }
            
            analysis['hazard_type_statistics'] = hazard_type_stats
            
            # Extreme scenarios
            extreme_scenarios = []
            total_events_values = [metrics.get('total_events', 0) for metrics in all_metrics]
            extreme_threshold = np.percentile(total_events_values, 95)
            
            for i, result in enumerate(simulation_results):
                if result['metrics'].get('total_events', 0) >= extreme_threshold:
                    extreme_scenarios.append({
                        'simulation_id': result['simulation_id'],
                        'total_events': result['total_events'],
                        'metrics': result['metrics']
                    })
            
            analysis['extreme_scenarios'] = extreme_scenarios
            analysis['simulation_summary'] = {
                'total_simulations': len(simulation_results),
                'successful_simulations': len([r for r in simulation_results if r]),
                'extreme_scenario_count': len(extreme_scenarios)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Result analysis failed: {e}")
            return {'error': str(e)}

class ScenarioTreeBuilder:
    """Build and analyze scenario trees for risk assessment"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Tree structure
        self.scenario_tree = nx.DiGraph()
        self.node_probabilities = {}
        self.edge_probabilities = {}
        self.scenario_outcomes = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'max_tree_depth': 5,
            'min_probability_threshold': 0.001,
            'branching_factor': 3,
            'time_step': 24.0,  # hours
            'probability_decay': 0.8,
            'outcome_metrics': ['economic_loss', 'casualties', 'infrastructure_damage']
        }
    
    def build_scenario_tree(self, initial_conditions: Dict[str, Any],
                           hazard_models: Dict[HazardType, Any],
                           time_horizon: float) -> nx.DiGraph:
        """Build scenario tree from initial conditions"""
        try:
            # Initialize root node
            root_node = "root"
            self.scenario_tree.add_node(root_node, 
                                      time=0.0, 
                                      conditions=initial_conditions,
                                      probability=1.0)
            
            # Build tree recursively
            self._build_tree_recursive(root_node, initial_conditions, hazard_models, 
                                     0.0, time_horizon, 0)
            
            # Calculate cumulative probabilities
            self._calculate_cumulative_probabilities()
            
            # Calculate outcome metrics
            self._calculate_scenario_outcomes()
            
            return self.scenario_tree
            
        except Exception as e:
            self.logger.error(f"Scenario tree building failed: {e}")
            return nx.DiGraph()
    
    def _build_tree_recursive(self, parent_node: str, current_conditions: Dict[str, Any],
                            hazard_models: Dict[HazardType, Any], current_time: float,
                            max_time: float, depth: int):
        """Recursively build scenario tree"""
        if depth >= self.config['max_tree_depth'] or current_time >= max_time:
            return
        
        try:
            next_time = current_time + self.config['time_step'] * 3600  # Convert to seconds
            
            # Generate possible events at next time step
            possible_events = self._generate_possible_events(current_conditions, hazard_models, next_time)
            
            # Create branches for each possible event (including no event)
            branches = self._create_branches(possible_events)
            
            for branch_id, branch_info in enumerate(branches):
                if branch_info['probability'] < self.config['min_probability_threshold']:
                    continue
                
                # Create child node
                child_node = f"{parent_node}_t{int(next_time)}_b{branch_id}"
                new_conditions = self._update_conditions(current_conditions, branch_info['events'])
                
                self.scenario_tree.add_node(child_node,
                                          time=next_time,
                                          conditions=new_conditions,
                                          events=branch_info['events'],
                                          probability=branch_info['probability'])
                
                # Add edge
                self.scenario_tree.add_edge(parent_node, child_node,
                                          probability=branch_info['probability'],
                                          events=branch_info['events'])
                
                # Continue recursion
                self._build_tree_recursive(child_node, new_conditions, hazard_models,
                                         next_time, max_time, depth + 1)
                
        except Exception as e:
            self.logger.error(f"Recursive tree building failed at depth {depth}: {e}")
    
    def _generate_possible_events(self, conditions: Dict[str, Any],
                                hazard_models: Dict[HazardType, Any],
                                time: float) -> List[Dict[str, Any]]:
        """Generate possible events for next time step"""
        possible_events = []
        
        try:
            # No event scenario
            possible_events.append({
                'type': 'no_event',
                'probability': 0.8,  # Base probability of no event
                'events': []
            })
            
            # Single hazard events
            for hazard_type, model in hazard_models.items():
                if hasattr(model, 'predict_probability'):
                    # Use model to predict probability
                    location = conditions.get('location', np.array([0, 0, 0]))
                    pred_result = model.predict_probability(location, time)
                    prob = pred_result.get('probability', 0.1)
                else:
                    # Use default probability
                    prob = 0.1
                
                # Adjust probability based on current conditions
                adjusted_prob = self._adjust_probability_for_conditions(prob, conditions, hazard_type)
                
                if adjusted_prob > self.config['min_probability_threshold']:
                    event = {
                        'hazard_type': hazard_type,
                        'magnitude': self._sample_magnitude(hazard_type),
                        'location': conditions.get('location', np.array([0, 0, 0])),
                        'time': time
                    }
                    
                    possible_events.append({
                        'type': 'single_hazard',
                        'hazard_type': hazard_type,
                        'probability': adjusted_prob,
                        'events': [event]
                    })
            
            return possible_events
            
        except Exception as e:
            self.logger.error(f"Event generation failed: {e}")
            return [{'type': 'no_event', 'probability': 1.0, 'events': []}]
    
    def _adjust_probability_for_conditions(self, base_prob: float, 
                                         conditions: Dict[str, Any],
                                         hazard_type: HazardType) -> float:
        """Adjust probability based on current conditions"""
        try:
            adjusted_prob = base_prob
            
            # Stress level influence
            stress_level = conditions.get('stress_level', 0.5)
            if hazard_type in [HazardType.EARTHQUAKE, HazardType.VOLCANIC_ERUPTION]:
                stress_factor = 1.0 + stress_level
                adjusted_prob *= stress_factor
            
            # Recent activity influence
            recent_activity = conditions.get('recent_activity', 0.0)
            if recent_activity > 0:
                activity_factor = 1.0 + recent_activity * 0.5
                adjusted_prob *= activity_factor
            
            # Environmental conditions
            if hazard_type == HazardType.LANDSLIDE:
                precipitation = conditions.get('precipitation', 0.0)
                slope_stability = conditions.get('slope_stability', 1.0)
                adjusted_prob *= (1.0 + precipitation) / slope_stability
            
            return min(1.0, adjusted_prob)
            
        except Exception as e:
            self.logger.error(f"Probability adjustment failed: {e}")
            return base_prob
    
    def _sample_magnitude(self, hazard_type: HazardType) -> float:
        """Sample magnitude for hazard type"""
        magnitude_ranges = {
            HazardType.EARTHQUAKE: (1.0, 9.0),
            HazardType.VOLCANIC_ERUPTION: (1.0, 8.0),
            HazardType.LANDSLIDE: (1.0, 5.0),
            HazardType.TSUNAMI: (1.0, 10.0),
        }
        
        min_mag, max_mag = magnitude_ranges.get(hazard_type, (1.0, 5.0))
        
        # Log-normal distribution for most hazards
        log_mean = np.log(min_mag + (max_mag - min_mag) * 0.3)
        log_std = 0.5
        
        magnitude = np.random.lognormal(log_mean, log_std)
        return np.clip(magnitude, min_mag, max_mag)
    
    def _create_branches(self, possible_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create branches from possible events"""
        try:
            branches = []
            
            # Normalize probabilities
            total_prob = sum(event['probability'] for event in possible_events)
            if total_prob > 0:
                for event in possible_events:
                    event['probability'] /= total_prob
            
            # Create branches (for now, each event becomes a separate branch)
            for event in possible_events:
                if event['probability'] >= self.config['min_probability_threshold']:
                    branches.append(event)
            
            # Limit branching factor
            if len(branches) > self.config['branching_factor']:
                # Sort by probability and take top branches
                branches.sort(key=lambda x: x['probability'], reverse=True)
                branches = branches[:self.config['branching_factor']]
                
                # Renormalize probabilities
                total_prob = sum(branch['probability'] for branch in branches)
                for branch in branches:
                    branch['probability'] /= total_prob
            
            return branches
            
        except Exception as e:
            self.logger.error(f"Branch creation failed: {e}")
            return [{'type': 'no_event', 'probability': 1.0, 'events': []}]
    
    def _update_conditions(self, current_conditions: Dict[str, Any],
                          events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update conditions based on events"""
        try:
            new_conditions = current_conditions.copy()
            
            if not events:
                # No events - gradual relaxation
                new_conditions['stress_level'] = max(0.0, 
                    current_conditions.get('stress_level', 0.5) * 0.95)
                new_conditions['recent_activity'] = max(0.0,
                    current_conditions.get('recent_activity', 0.0) * 0.9)
                return new_conditions
            
            # Update based on events
            for event in events:
                hazard_type = event['hazard_type']
                magnitude = event['magnitude']
                
                # Increase stress level
                stress_increase = magnitude / 10.0
                new_conditions['stress_level'] = min(1.0,
                    current_conditions.get('stress_level', 0.5) + stress_increase)
                
                # Set recent activity
                new_conditions['recent_activity'] = magnitude / 10.0
                
                # Hazard-specific updates
                if hazard_type == HazardType.EARTHQUAKE:
                    # Earthquakes can trigger other events
                    new_conditions['seismic_activity'] = magnitude
                
                elif hazard_type == HazardType.VOLCANIC_ERUPTION:
                    # Volcanic activity affects local conditions
                    new_conditions['volcanic_activity'] = magnitude
                    new_conditions['gas_emissions'] = magnitude * 0.5
                
                elif hazard_type == HazardType.LANDSLIDE:
                    # Landslides affect slope stability
                    new_conditions['slope_stability'] = max(0.1,
                        current_conditions.get('slope_stability', 1.0) - magnitude * 0.1)
            
            return new_conditions
            
        except Exception as e:
            self.logger.error(f"Condition update failed: {e}")
            return current_conditions
    
    def _calculate_cumulative_probabilities(self):
        """Calculate cumulative probabilities for all paths"""
        try:
            # Use topological sorting to ensure proper order
            for node in nx.topological_sort(self.scenario_tree):
                if node == "root":
                    self.node_probabilities[node] = 1.0
                else:
                    # Calculate probability as product of path probabilities
                    cum_prob = 0.0
                    for parent in self.scenario_tree.predecessors(node):
                        edge_prob = self.scenario_tree[parent][node]['probability']
                        parent_prob = self.node_probabilities.get(parent, 0.0)
                        cum_prob += parent_prob * edge_prob
                    
                    self.node_probabilities[node] = cum_prob
                    
        except Exception as e:
            self.logger.error(f"Cumulative probability calculation failed: {e}")
    
    def _calculate_scenario_outcomes(self):
        """Calculate outcome metrics for each scenario"""
        try:
            # Find leaf nodes (end scenarios)
            leaf_nodes = [node for node in self.scenario_tree.nodes() 
                         if self.scenario_tree.out_degree(node) == 0]
            
            for leaf_node in leaf_nodes:
                # Trace path from root to leaf
                path_events = []
                current = leaf_node
                
                while current != "root":
                    predecessors = list(self.scenario_tree.predecessors(current))
                    if predecessors:
                        parent = predecessors[0]
                        edge_data = self.scenario_tree[parent][current]
                        if 'events' in edge_data:
                            path_events.extend(edge_data['events'])
                        current = parent
                    else:
                        break
                
                # Calculate outcome metrics
                outcomes = self._calculate_outcome_metrics(path_events)
                outcomes['probability'] = self.node_probabilities.get(leaf_node, 0.0)
                
                self.scenario_outcomes[leaf_node] = outcomes
                
        except Exception as e:
            self.logger.error(f"Scenario outcome calculation failed: {e}")
    
    def _calculate_outcome_metrics(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate outcome metrics for a scenario"""
        try:
            metrics = {
                'economic_loss': 0.0,
                'casualties': 0.0,
                'infrastructure_damage': 0.0,
                'total_events': len(events),
                'max_magnitude': 0.0
            }
            
            if not events:
                return metrics
            
            # Calculate metrics based on events
            total_magnitude = 0.0
            
            for event in events:
                magnitude = event.get('magnitude', 0.0)
                hazard_type = event.get('hazard_type')
                
                total_magnitude += magnitude
                metrics['max_magnitude'] = max(metrics['max_magnitude'], magnitude)
                
                # Simple impact models
                if hazard_type == HazardType.EARTHQUAKE:
                    # Earthquake impacts
                    metrics['economic_loss'] += magnitude ** 2 * 1e6  # USD
                    metrics['casualties'] += max(0, magnitude - 5) ** 2 * 100
                    metrics['infrastructure_damage'] += magnitude / 10.0
                
                elif hazard_type == HazardType.VOLCANIC_ERUPTION:
                    # Volcanic impacts
                    metrics['economic_loss'] += magnitude ** 1.5 * 5e5  # USD
                    metrics['casualties'] += max(0, magnitude - 3) * 50
                    metrics['infrastructure_damage'] += magnitude / 8.0
                
                elif hazard_type == HazardType.TSUNAMI:
                    # Tsunami impacts
                    metrics['economic_loss'] += magnitude ** 2.5 * 2e6  # USD
                    metrics['casualties'] += magnitude ** 2 * 200
                    metrics['infrastructure_damage'] += magnitude / 5.0
            
            metrics['avg_magnitude'] = total_magnitude / len(events) if events else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Outcome metrics calculation failed: {e}")
            return {
                'economic_loss': 0.0,
                'casualties': 0.0,
                'infrastructure_damage': 0.0,
                'total_events': len(events),
                'max_magnitude': 0.0
            }
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Calculate overall risk metrics from scenario tree"""
        try:
            if not self.scenario_outcomes:
                return {}
            
            risk_metrics = {}
            
            # Expected values
            for metric in self.config['outcome_metrics']:
                expected_value = 0.0
                for node, outcomes in self.scenario_outcomes.items():
                    prob = outcomes.get('probability', 0.0)
                    value = outcomes.get(metric, 0.0)
                    expected_value += prob * value
                
                risk_metrics[f'expected_{metric}'] = expected_value
            
            # Probability of extreme scenarios
            high_impact_prob = 0.0
            extreme_loss_threshold = 1e8  # $100M
            
            for outcomes in self.scenario_outcomes.values():
                if outcomes.get('economic_loss', 0.0) > extreme_loss_threshold:
                    high_impact_prob += outcomes.get('probability', 0.0)
            
            risk_metrics['extreme_scenario_probability'] = high_impact_prob
            
            # Value at Risk (VaR) estimation
            loss_distribution = []
            for outcomes in self.scenario_outcomes.values():
                prob = outcomes.get('probability', 0.0)
                loss = outcomes.get('economic_loss', 0.0)
                # Approximate discrete distribution
                n_samples = max(1, int(prob * 10000))
                loss_distribution.extend([loss] * n_samples)
            
            if loss_distribution:
                risk_metrics['var_95'] = np.percentile(loss_distribution, 95)
                risk_metrics['var_99'] = np.percentile(loss_distribution, 99)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return {}

# Integration and main system class
class IntegratedHazardInteractionSystem:
    """Main system integrating all hazard interaction components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Component systems
        self.hazard_models = {}
        self.interaction_models = {}
        self.bayesian_network = None
        self.monte_carlo_simulator = None
        self.scenario_tree_builder = None
        
        # System state
        self.is_initialized = False
        self.historical_events = []
        
        self._initialize_system()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
            'hazard_types': [
                HazardType.EARTHQUAKE,
                HazardType.VOLCANIC_ERUPTION,
                HazardType.LANDSLIDE,
                HazardType.TSUNAMI,
                HazardType.PYROCLASTIC_FLOW
            ],
            'enable_monte_carlo': True,
            'enable_scenario_trees': True,
            'enable_bayesian_network': True,
            'parallel_processing': True,
            'cache_results': True,
            'output_directory': './hazard_outputs',
            'save_intermediate_results': True
        }
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            # Initialize hazard models
            for hazard_type in self.config['hazard_types']:
                self.hazard_models[hazard_type] = ProbabilisticHazardModel(hazard_type)
            
            # Initialize specialized interaction models
            self.interaction_models[(HazardType.EARTHQUAKE, HazardType.VOLCANIC_ERUPTION)] = \
                VolcanicSeismicInteractionModel()
            
            # Initialize Bayesian network
            if self.config['enable_bayesian_network']:
                self.bayesian_network = BayesianHazardNetwork(self.config['hazard_types'])
            
            # Initialize scenario tree builder
            if self.config['enable_scenario_trees']:
                self.scenario_tree_builder = ScenarioTreeBuilder()
            
            self.is_initialized = True
            self.logger.info("Hazard interaction system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            self.is_initialized = False
    
    async def train_system(self, historical_events: List[HazardEvent]) -> Dict[str, Any]:
        """Train all models with historical data"""
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            self.historical_events = historical_events
            training_results = {}
            
            # Train individual hazard models
            hazard_events_by_type = defaultdict(list)
            for event in historical_events:
                hazard_events_by_type[event.hazard_type].append(event)
            
            for hazard_type, events in hazard_events_by_type.items():
                if hazard_type in self.hazard_models and len(events) > 5:
                    model_results = self.hazard_models[hazard_type].fit(events)
                    training_results[f'{hazard_type.value}_model'] = model_results
            
            # Train Bayesian network
            if self.bayesian_network:
                self.bayesian_network.learn_structure(historical_events)
                network_results = self.bayesian_network.learn_parameters(historical_events)
                training_results['bayesian_network'] = network_results
            
            # Initialize Monte Carlo simulator
            if self.config['enable_monte_carlo']:
                self.monte_carlo_simulator = MonteCarloHazardSimulator(
                    self.hazard_models, self.interaction_models
                )
            
            self.logger.info("System training completed successfully")
            return training_results
            
        except Exception as e:
            self.logger.error(f"System training failed: {e}")
            return {'error': str(e)}
    
    async def assess_hazard_interactions(self, region_bounds: Dict[str, float],
                                       time_horizon: float = 365.25,
                                       analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """Comprehensive hazard interaction assessment"""
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            assessment_results = {
                'region_bounds': region_bounds,
                'time_horizon': time_horizon,
                'analysis_type': analysis_type,
                'timestamp': time.time()
            }
            
            # Monte Carlo simulation
            if analysis_type in ['comprehensive', 'monte_carlo'] and self.monte_carlo_simulator:
                self.logger.info("Running Monte Carlo simulation...")
                mc_results = await self.monte_carlo_simulator.run_monte_carlo_simulation(
                    region_bounds, {}
                )
                assessment_results['monte_carlo'] = mc_results
            
            # Scenario tree analysis
            if analysis_type in ['comprehensive', 'scenario_trees'] and self.scenario_tree_builder:
                self.logger.info("Building scenario trees...")
                initial_conditions = {
                    'location': np.array([
                        (region_bounds['lat_min'] + region_bounds['lat_max']) / 2,
                        (region_bounds['lon_min'] + region_bounds['lon_max']) / 2,
                        0
                    ]),
                    'stress_level': 0.5,
                    'recent_activity': 0.0
                }
                
                scenario_tree = self.scenario_tree_builder.build_scenario_tree(
                    initial_conditions, self.hazard_models, time_horizon * 24 * 3600
                )
                
                risk_metrics = self.scenario_tree_builder.get_risk_metrics()
                
                assessment_results['scenario_trees'] = {
                    'tree_size': len(scenario_tree.nodes()),
                    'risk_metrics': risk_metrics
                }
            
            # Bayesian network inference
            if analysis_type in ['comprehensive', 'bayesian'] and self.bayesian_network:
                self.logger.info("Performing Bayesian inference...")
                # Example evidence
                evidence = {}
                probabilities = self.bayesian_network.infer_probabilities(evidence)
                assessment_results['bayesian_inference'] = probabilities
            
            # Interaction analysis
            interaction_analysis = await self._analyze_specific_interactions(region_bounds)
            assessment_results['interaction_analysis'] = interaction_analysis
            
            return assessment_results
            
        except Exception as e:
            self.logger.error(f"Hazard assessment failed: {e}")
            return {'error': str(e)}
    
    async def _analyze_specific_interactions(self, region_bounds: Dict[str, float]) -> Dict[str, Any]:
        """Analyze specific hazard interactions"""
        try:
            interaction_results = {}
            
            # Volcanic-seismic interactions
            vs_model = self.interaction_models.get((HazardType.EARTHQUAKE, HazardType.VOLCANIC_ERUPTION))
            if vs_model:
                # Create synthetic earthquake for analysis
                synthetic_eq = HazardEvent(
                    id="synthetic_earthquake",
                    hazard_type=HazardType.EARTHQUAKE,
                    location=np.array([
                        (region_bounds['lat_min'] + region_bounds['lat_max']) / 2,
                        (region_bounds['lon_min'] + region_bounds['lon_max']) / 2,
                        10  # 10 km depth
                    ]),
                    magnitude=6.0,
                    timestamp=time.time(),
                    duration=60,
                    intensity=6.0
                )
                
                # Analyze potential volcanic triggering
                volcano_location = synthetic_eq.location + np.array([20, 20, -5])  # 20 km offset
                trigger_analysis = vs_model.calculate_trigger_probability(
                    synthetic_eq, volcano_location
                )
                
                interaction_results['volcanic_seismic'] = trigger_analysis
            
            return interaction_results
            
        except Exception as e:
            self.logger.error(f"Interaction analysis failed: {e}")
            return {}
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save analysis results"""
        try:
            if not filename:
                timestamp = int(time.time())
                filename = f"hazard_assessment_{timestamp}.json"
            
            output_dir = self.config.get('output_directory', './hazard_outputs')
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            filepath = os.path.join(output_dir, filename)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"Results saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return None
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, HazardType):
            return obj.value
        else:
            return obj

# Example usage and testing
async def example_hazard_interaction_system():
    """Example of using the integrated hazard interaction system"""
    
    # Create system
    system = IntegratedHazardInteractionSystem()
    
    # Generate synthetic historical events
    historical_events = []
    np.random.seed(42)
    
    for i in range(200):
        hazard_type = np.random.choice([
            HazardType.EARTHQUAKE,
            HazardType.VOLCANIC_ERUPTION,
            HazardType.LANDSLIDE
        ])
        
        event = HazardEvent(
            id=f"historical_{i}",
            hazard_type=hazard_type,
            location=np.random.uniform([35, 135, 0], [40, 140, 20], 3),
            magnitude=np.random.lognormal(1.0, 0.5),
            timestamp=time.time() - np.random.uniform(0, 10 * 365.25 * 24 * 3600),
            duration=np.random.exponential(3600),
            intensity=np.random.uniform(1, 8)
        )
        historical_events.append(event)
    
    # Train system
    print("Training hazard interaction system...")
    training_results = await system.train_system(historical_events)
    print(f"Training completed. Results: {len(training_results)} models trained")
    
    # Run comprehensive assessment
    region_bounds = {
        'lat_min': 35.0, 'lat_max': 40.0,
        'lon_min': 135.0, 'lon_max': 140.0,
        'depth_min': 0.0, 'depth_max': 30.0
    }
    
    print("Running comprehensive hazard assessment...")
    assessment_results = await system.assess_hazard_interactions(
        region_bounds, time_horizon=365.25, analysis_type='comprehensive'
    )
    
    # Display results
    print("\nAssessment Results Summary:")
    print(f"Monte Carlo simulations: {assessment_results.get('monte_carlo', {}).get('simulation_summary', {})}")
    print(f"Scenario trees: {assessment_results.get('scenario_trees', {})}")
    print(f"Bayesian inference: {len(assessment_results.get('bayesian_inference', {}))}")
    
    # Save results
    output_file = system.save_results(assessment_results)
    print(f"Results saved to: {output_file}")
    
    return assessment_results

if __name__ == "__main__":
    print("Testing Hazard Interactions and Probabilistic Framework...")
    
    # Run example
    results = asyncio.run(example_hazard_interaction_system())
    
    print("\nSystem test completed successfully!")