"""
Advanced ML-Guided Source Characterization System
Automated source characterization with comprehensive inversion algorithms and active learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.optimize as opt
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class SourceType(Enum):
    DIKE = "dike"
    SILL = "sill"
    FAULT = "fault"
    POINT_SOURCE = "point_source"
    DISTRIBUTED_SLIP = "distributed_slip"
    VISCOELASTIC = "viscoelastic"
    POROELASTIC = "poroelastic"

@dataclass
class ModelParameters:
    source_type: SourceType
    parameters: Dict[str, float]
    uncertainty: Dict[str, float]
    likelihood: float
    aic: float
    bic: float
    posterior_samples: Optional[np.ndarray] = None

@dataclass
class ObservationData:
    coordinates: np.ndarray  # [N, 3] - x, y, z coordinates
    displacements: np.ndarray  # [N, 3] - ux, uy, uz displacements
    uncertainties: np.ndarray  # [N, 3] - displacement uncertainties
    data_type: str  # 'insar', 'gps', 'leveling', etc.
    acquisition_time: np.ndarray  # Time stamps
    metadata: Dict[str, Any]

class SourceTypeClassifier(nn.Module):
    """Advanced neural network for source type classification with attention mechanism"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, n_classes: int = 7):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=8,
            dropout=0.1
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, n_classes),
            nn.Softmax(dim=1)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Apply attention
        attended_features, _ = self.attention(
            features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        classification = self.classifier(attended_features)
        uncertainty = self.uncertainty_head(attended_features)
        
        return classification, uncertainty

class ParameterSpaceExplorer:
    """Advanced parameter space exploration with Bayesian optimization"""
    
    def __init__(self, bounds: Dict[str, Tuple[float, float]]):
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.gp_models = {}
        self.acquisition_history = []
        self.evaluation_history = []
        
        # Initialize Gaussian Process for each parameter
        for param in self.param_names:
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
            self.gp_models[param] = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10
            )
    
    def expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function"""
        if not self.evaluation_history:
            return np.ones(X.shape[0])
        
        mu, sigma = self.gp_models[self.param_names[0]].predict(X, return_std=True)
        best_y = np.max(self.evaluation_history)
        
        with np.errstate(divide='warn'):
            imp = mu - best_y - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def upper_confidence_bound(self, X: np.ndarray, kappa: float = 2.576) -> np.ndarray:
        """Upper Confidence Bound acquisition function"""
        if not self.evaluation_history:
            return np.ones(X.shape[0])
        
        mu, sigma = self.gp_models[self.param_names[0]].predict(X, return_std=True)
        return mu + kappa * sigma
    
    def suggest_next_points(self, n_points: int = 5, 
                          acquisition_func: str = 'ei') -> List[Dict[str, float]]:
        """Suggest next parameter combinations to evaluate"""
        if len(self.evaluation_history) < 3:
            # Random sampling for initial points
            suggestions = []
            for _ in range(n_points):
                point = {}
                for param, (low, high) in self.bounds.items():
                    point[param] = np.random.uniform(low, high)
                suggestions.append(point)
            return suggestions
        
        # Generate candidate points
        n_candidates = 1000
        candidates = np.random.rand(n_candidates, len(self.param_names))
        
        for i, (param, (low, high)) in enumerate(self.bounds.items()):
            candidates[:, i] = candidates[:, i] * (high - low) + low
        
        # Evaluate acquisition function
        if acquisition_func == 'ei':
            acquisition_values = self.expected_improvement(candidates)
        else:  # ucb
            acquisition_values = self.upper_confidence_bound(candidates)
        
        # Select top candidates
        top_indices = np.argsort(acquisition_values)[-n_points:]
        suggestions = []
        
        for idx in top_indices:
            point = {}
            for i, param in enumerate(self.param_names):
                point[param] = candidates[idx, i]
            suggestions.append(point)
        
        return suggestions
    
    def update_with_evaluation(self, parameters: Dict[str, float], 
                              objective_value: float):
        """Update GP models with new evaluation"""
        self.acquisition_history.append(parameters)
        self.evaluation_history.append(objective_value)
        
        if len(self.evaluation_history) >= 2:
            X = np.array([[p[param] for param in self.param_names] 
                         for p in self.acquisition_history])
            y = np.array(self.evaluation_history)
            
            for param in self.param_names:
                self.gp_models[param].fit(X, y)

class ActiveLearningEngine:
    """Active learning system for hypothesis formation and adaptive sampling"""
    
    def __init__(self, model_pool: List[Any]):
        self.model_pool = model_pool
        self.uncertainty_threshold = 0.8
        self.diversity_weight = 0.3
        self.disagreement_weight = 0.4
        self.uncertainty_weight = 0.3
        
        self.hypothesis_history = []
        self.confidence_tracker = []
        self.exploration_regions = []
    
    def calculate_model_disagreement(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Calculate disagreement between ensemble models"""
        if len(predictions) < 2:
            return np.zeros(predictions[0].shape[0])
        
        predictions_array = np.stack(predictions, axis=0)
        disagreement = np.var(predictions_array, axis=0)
        return np.mean(disagreement, axis=1) if len(disagreement.shape) > 1 else disagreement
    
    def calculate_prediction_uncertainty(self, predictions: List[np.ndarray], 
                                       uncertainties: List[np.ndarray]) -> np.ndarray:
        """Calculate prediction uncertainty from ensemble"""
        if not uncertainties:
            return np.ones(predictions[0].shape[0]) * 0.5
        
        mean_uncertainty = np.mean(uncertainties, axis=0)
        return np.mean(mean_uncertainty, axis=1) if len(mean_uncertainty.shape) > 1 else mean_uncertainty
    
    def sample_diversity(self, candidates: np.ndarray, n_samples: int) -> np.ndarray:
        """Select diverse samples from candidate pool"""
        if len(candidates) <= n_samples:
            return candidates
        
        # Use k-medoids for diverse sampling
        distances = squareform(pdist(candidates))
        selected_indices = [0]  # Start with first candidate
        
        for _ in range(n_samples - 1):
            remaining_indices = [i for i in range(len(candidates)) if i not in selected_indices]
            min_distances = []
            
            for idx in remaining_indices:
                min_dist = min([distances[idx][selected_idx] for selected_idx in selected_indices])
                min_distances.append(min_dist)
            
            # Select candidate with maximum minimum distance
            best_idx = remaining_indices[np.argmax(min_distances)]
            selected_indices.append(best_idx)
        
        return candidates[selected_indices]
    
    def form_hypothesis(self, observations: np.ndarray, 
                       current_models: Dict[str, Any]) -> Dict[str, Any]:
        """Form new hypothesis based on model performance and data patterns"""
        hypothesis = {
            'timestamp': pd.Timestamp.now(),
            'data_characteristics': self._analyze_data_characteristics(observations),
            'model_performance': self._evaluate_model_ensemble(current_models, observations),
            'suggested_modifications': [],
            'confidence_score': 0.0,
            'exploration_priority': []
        }
        
        # Analyze data patterns
        data_stats = hypothesis['data_characteristics']
        
        # Suggest model modifications based on patterns
        if data_stats['temporal_correlation'] > 0.7:
            hypothesis['suggested_modifications'].append({
                'type': 'temporal_modeling',
                'description': 'Strong temporal correlation detected - consider time-dependent models',
                'priority': 0.8
            })
        
        if data_stats['spatial_clustering'] > 0.6:
            hypothesis['suggested_modifications'].append({
                'type': 'spatial_clustering',
                'description': 'Spatial clustering detected - consider distributed source models',
                'priority': 0.7
            })
        
        if data_stats['nonlinearity_score'] > 0.5:
            hypothesis['suggested_modifications'].append({
                'type': 'nonlinear_effects',
                'description': 'Nonlinear behavior detected - consider viscoelastic/poroelastic models',
                'priority': 0.9
            })
        
        # Calculate overall confidence
        performance_scores = [perf['accuracy'] for perf in hypothesis['model_performance'].values()]
        hypothesis['confidence_score'] = np.mean(performance_scores) if performance_scores else 0.0
        
        self.hypothesis_history.append(hypothesis)
        return hypothesis
    
    def _analyze_data_characteristics(self, observations: np.ndarray) -> Dict[str, float]:
        """Analyze key characteristics of observation data"""
        characteristics = {}
        
        # Temporal correlation (assuming last column is time)
        if observations.shape[1] > 3:
            time_series = observations[:, -1]
            characteristics['temporal_correlation'] = np.corrcoef(
                time_series[:-1], time_series[1:]
            )[0, 1] if len(time_series) > 1 else 0.0
        else:
            characteristics['temporal_correlation'] = 0.0
        
        # Spatial clustering
        spatial_coords = observations[:, :2]  # Assuming first two columns are spatial
        if len(spatial_coords) > 1:
            distances = pdist(spatial_coords)
            characteristics['spatial_clustering'] = 1.0 - (np.std(distances) / np.mean(distances))
        else:
            characteristics['spatial_clustering'] = 0.0
        
        # Nonlinearity score
        if observations.shape[1] > 2:
            displacements = observations[:, 2]  # Assuming third column is displacement
            linear_trend = np.polyfit(range(len(displacements)), displacements, 1)
            linear_pred = np.polyval(linear_trend, range(len(displacements)))
            residuals = displacements - linear_pred
            characteristics['nonlinearity_score'] = np.std(residuals) / (np.std(displacements) + 1e-10)
        else:
            characteristics['nonlinearity_score'] = 0.0
        
        # Data quality metrics
        characteristics['completeness'] = 1.0 - (np.sum(np.isnan(observations)) / observations.size)
        characteristics['signal_to_noise'] = np.mean(observations) / (np.std(observations) + 1e-10)
        
        return characteristics
    
    def _evaluate_model_ensemble(self, models: Dict[str, Any], 
                                observations: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate performance of model ensemble"""
        performance = {}
        
        for model_name, model in models.items():
            try:
                # Generate predictions (placeholder - actual implementation depends on model type)
                predictions = np.random.rand(len(observations))  # Placeholder
                residuals = observations[:, 2] - predictions  # Assuming displacement in column 2
                
                performance[model_name] = {
                    'rmse': np.sqrt(np.mean(residuals**2)),
                    'mae': np.mean(np.abs(residuals)),
                    'r2': 1 - np.var(residuals) / np.var(observations[:, 2]),
                    'accuracy': max(0, 1 - np.std(residuals) / (np.std(observations[:, 2]) + 1e-10))
                }
            except Exception as e:
                performance[model_name] = {
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'r2': -float('inf'),
                    'accuracy': 0.0
                }
        
        return performance
    
    def select_next_samples(self, candidate_pool: np.ndarray, 
                          model_predictions: List[np.ndarray],
                          model_uncertainties: List[np.ndarray],
                          n_samples: int = 10) -> np.ndarray:
        """Select next samples for active learning"""
        if len(candidate_pool) <= n_samples:
            return candidate_pool
        
        # Calculate acquisition scores
        disagreement = self.calculate_model_disagreement(model_predictions)
        uncertainty = self.calculate_prediction_uncertainty(model_predictions, model_uncertainties)
        
        # Normalize scores
        disagreement_norm = (disagreement - np.min(disagreement)) / (np.max(disagreement) - np.min(disagreement) + 1e-10)
        uncertainty_norm = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty) + 1e-10)
        
        # Combined acquisition score
        acquisition_scores = (
            self.disagreement_weight * disagreement_norm +
            self.uncertainty_weight * uncertainty_norm
        )
        
        # Select top candidates
        top_indices = np.argsort(acquisition_scores)[-n_samples*2:]  # Get more candidates for diversity
        top_candidates = candidate_pool[top_indices]
        
        # Apply diversity sampling
        selected_samples = self.sample_diversity(top_candidates, n_samples)
        
        return selected_samples

class SourceCharacterizationML:
    """Main ML-guided source characterization system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.classifier = SourceTypeClassifier(
            input_dim=self.config['classifier']['input_dim'],
            hidden_dim=self.config['classifier']['hidden_dim'],
            n_classes=len(SourceType)
        )
        
        self.parameter_explorer = None
        self.active_learner = None
        
        # Model storage
        self.fitted_models = {}
        self.model_performance = {}
        self.inversion_history = []
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.feature_extractor = None
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'classifier': {
                'input_dim': 512,
                'hidden_dim': 256,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'bayesian_optimization': {
                'n_initial_points': 10,
                'n_iterations': 50,
                'acquisition_function': 'ei'
            },
            'active_learning': {
                'uncertainty_threshold': 0.8,
                'batch_size': 10,
                'diversity_weight': 0.3
            },
            'inversion': {
                'max_iterations': 1000,
                'tolerance': 1e-6,
                'regularization': 1e-4
            }
        }
    
    def extract_features(self, observations: ObservationData) -> np.ndarray:
        """Extract comprehensive features from observation data"""
        features = []
        
        # Spatial features
        coords = observations.coordinates
        features.extend([
            np.mean(coords, axis=0),  # Mean coordinates
            np.std(coords, axis=0),   # Coordinate spread
            np.max(coords, axis=0) - np.min(coords, axis=0),  # Bounding box
        ])
        
        # Displacement features
        displacements = observations.displacements
        features.extend([
            np.mean(displacements, axis=0),  # Mean displacements
            np.std(displacements, axis=0),   # Displacement variability
            np.max(np.abs(displacements), axis=0),  # Maximum displacements
        ])
        
        # Geometric features
        if len(coords) > 3:
            # Principal component analysis of coordinates
            coords_centered = coords - np.mean(coords, axis=0)
            _, _, V = np.linalg.svd(coords_centered, full_matrices=False)
            features.extend(V.flatten())
        
        # Temporal features
        if hasattr(observations, 'acquisition_time') and observations.acquisition_time is not None:
            time_diffs = np.diff(observations.acquisition_time)
            features.extend([
                np.mean(time_diffs) if len(time_diffs) > 0 else 0,
                np.std(time_diffs) if len(time_diffs) > 0 else 0,
            ])
        else:
            features.extend([0, 0])
        
        # Uncertainty features
        uncertainties = observations.uncertainties
        features.extend([
            np.mean(uncertainties, axis=0),  # Mean uncertainties
            np.std(uncertainties, axis=0),   # Uncertainty variability
        ])
        
        # Pad or truncate to fixed size
        features_array = np.concatenate(features) if isinstance(features[0], np.ndarray) else np.array(features)
        target_size = self.config['classifier']['input_dim']
        
        if len(features_array) > target_size:
            features_array = features_array[:target_size]
        elif len(features_array) < target_size:
            padding = np.zeros(target_size - len(features_array))
            features_array = np.concatenate([features_array, padding])
        
        return features_array
    
    async def classify_source_type(self, observations: ObservationData) -> Tuple[SourceType, Dict[str, float]]:
        """Classify source type using ML classifier"""
        features = self.extract_features(observations)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        self.classifier.eval()
        with torch.no_grad():
            classification_probs, uncertainty = self.classifier(features_tensor)
            
        # Get most likely source type
        class_idx = torch.argmax(classification_probs, dim=1).item()
        source_types = list(SourceType)
        predicted_type = source_types[class_idx]
        
        # Extract confidence scores
        confidence_scores = {
            source_type.value: prob.item() 
            for source_type, prob in zip(source_types, classification_probs[0])
        }
        confidence_scores['overall_confidence'] = 1.0 - uncertainty.item()
        
        return predicted_type, confidence_scores
    
    async def optimize_parameters(self, observations: ObservationData,
                                source_type: SourceType,
                                bounds: Dict[str, Tuple[float, float]]) -> ModelParameters:
        """Optimize source parameters using Bayesian optimization"""
        if self.parameter_explorer is None:
            self.parameter_explorer = ParameterSpaceExplorer(bounds)
        
        best_params = None
        best_likelihood = -np.inf
        
        for iteration in range(self.config['bayesian_optimization']['n_iterations']):
            # Get parameter suggestions
            suggestions = self.parameter_explorer.suggest_next_points(
                n_points=5,
                acquisition_func=self.config['bayesian_optimization']['acquisition_function']
            )
            
            # Evaluate each suggestion
            for params in suggestions:
                likelihood = await self._evaluate_model_likelihood(
                    observations, source_type, params
                )
                
                # Update parameter explorer
                self.parameter_explorer.update_with_evaluation(params, likelihood)
                
                # Track best parameters
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
                    best_params = params
                    
                self.logger.info(f"Iteration {iteration}: Likelihood = {likelihood:.6f}")
        
        # Calculate uncertainties using Hessian approximation
        uncertainties = await self._estimate_parameter_uncertainties(
            observations, source_type, best_params
        )
        
        # Calculate information criteria
        n_params = len(best_params)
        n_data = len(observations.displacements.flatten())
        aic = 2 * n_params - 2 * best_likelihood
        bic = n_params * np.log(n_data) - 2 * best_likelihood
        
        return ModelParameters(
            source_type=source_type,
            parameters=best_params,
            uncertainty=uncertainties,
            likelihood=best_likelihood,
            aic=aic,
            bic=bic
        )
    
    async def _evaluate_model_likelihood(self, observations: ObservationData,
                                       source_type: SourceType,
                                       parameters: Dict[str, float]) -> float:
        """Evaluate likelihood of model given parameters"""
        try:
            # Generate forward model predictions
            predicted_displacements = await self._forward_model(
                observations.coordinates, source_type, parameters
            )
            
            # Calculate likelihood assuming Gaussian errors
            residuals = observations.displacements.flatten() - predicted_displacements.flatten()
            uncertainties = observations.uncertainties.flatten()
            
            # Avoid division by zero
            uncertainties = np.maximum(uncertainties, 1e-10)
            
            # Log-likelihood calculation
            log_likelihood = -0.5 * np.sum(
                (residuals / uncertainties) ** 2 + np.log(2 * np.pi * uncertainties ** 2)
            )
            
            return log_likelihood
            
        except Exception as e:
            self.logger.error(f"Error evaluating likelihood: {str(e)}")
            return -np.inf
    
    async def _forward_model(self, coordinates: np.ndarray,
                           source_type: SourceType,
                           parameters: Dict[str, float]) -> np.ndarray:
        """Generate forward model predictions"""
        # Placeholder implementation - would integrate with elastic_inversion_models.py
        n_points = len(coordinates)
        
        if source_type == SourceType.POINT_SOURCE:
            # Simple point source (Mogi model)
            x0, y0, z0 = parameters.get('x0', 0), parameters.get('y0', 0), parameters.get('z0', 1000)
            volume_change = parameters.get('volume_change', 1e6)
            
            dx = coordinates[:, 0] - x0
            dy = coordinates[:, 1] - y0
            dz = coordinates[:, 2] - z0
            
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            r = np.maximum(r, 1e-10)  # Avoid division by zero
            
            C = 3 * volume_change / (4 * np.pi)
            
            ux = C * dx / r**3
            uy = C * dy / r**3
            uz = C * dz / r**3
            
            return np.column_stack([ux, uy, uz])
        
        elif source_type == SourceType.DIKE:
            # Simple dike model
            length = parameters.get('length', 1000)
            width = parameters.get('width', 500)
            opening = parameters.get('opening', 1.0)
            
            # Simplified dike calculation
            displacements = np.zeros((n_points, 3))
            displacements[:, 2] = opening * np.exp(-coordinates[:, 2] / 1000)  # Exponential decay with depth
            
            return displacements
        
        else:
            # Default: random small displacements
            return np.random.normal(0, 0.01, (n_points, 3))
    
    async def _estimate_parameter_uncertainties(self, observations: ObservationData,
                                              source_type: SourceType,
                                              parameters: Dict[str, float]) -> Dict[str, float]:
        """Estimate parameter uncertainties using finite differences"""
        uncertainties = {}
        delta = 1e-4
        
        for param_name, param_value in parameters.items():
            # Forward difference
            params_plus = parameters.copy()
            params_plus[param_name] = param_value + delta
            likelihood_plus = await self._evaluate_model_likelihood(
                observations, source_type, params_plus
            )
            
            # Backward difference  
            params_minus = parameters.copy()
            params_minus[param_name] = param_value - delta
            likelihood_minus = await self._evaluate_model_likelihood(
                observations, source_type, params_minus
            )
            
            # Second derivative approximation
            second_derivative = (likelihood_plus - 2 * await self._evaluate_model_likelihood(
                observations, source_type, parameters
            ) + likelihood_minus) / (delta ** 2)
            
            # Uncertainty from inverse Hessian
            if second_derivative < 0:
                uncertainties[param_name] = np.sqrt(-1.0 / second_derivative)
            else:
                uncertainties[param_name] = float('inf')  # Invalid curvature
        
        return uncertainties
    
    def model_selection(self, candidate_models: List[ModelParameters]) -> ModelParameters:
        """Select best model using information criteria"""
        if not candidate_models:
            raise ValueError("No candidate models provided")
        
        # Sort by AIC (lower is better)
        sorted_models = sorted(candidate_models, key=lambda x: x.aic)
        best_model = sorted_models[0]
        
        # Calculate Akaike weights for model averaging
        aics = [model.aic for model in candidate_models]
        min_aic = min(aics)
        delta_aics = [aic - min_aic for aic in aics]
        weights = [np.exp(-0.5 * delta) for delta in delta_aics]
        weight_sum = sum(weights)
        akaike_weights = [w / weight_sum for w in weights]
        
        self.logger.info(f"Best model: {best_model.source_type.value} (AIC: {best_model.aic:.2f})")
        for model, weight in zip(candidate_models, akaike_weights):
            self.logger.info(f"  {model.source_type.value}: weight = {weight:.3f}")
        
        return best_model
    
    async def comprehensive_characterization(self, observations: ObservationData) -> Dict[str, Any]:
        """Perform comprehensive source characterization"""
        results = {
            'timestamp': pd.Timestamp.now(),
            'data_summary': {
                'n_observations': len(observations.displacements),
                'data_type': observations.data_type,
                'spatial_extent': {
                    'x_range': [np.min(observations.coordinates[:, 0]), np.max(observations.coordinates[:, 0])],
                    'y_range': [np.min(observations.coordinates[:, 1]), np.max(observations.coordinates[:, 1])],
                    'z_range': [np.min(observations.coordinates[:, 2]), np.max(observations.coordinates[:, 2])]
                },
                'displacement_stats': {
                    'mean': np.mean(observations.displacements, axis=0).tolist(),
                    'std': np.std(observations.displacements, axis=0).tolist(),
                    'max_magnitude': np.max(np.linalg.norm(observations.displacements, axis=1))
                }
            },
            'classification': {},
            'models': [],
            'best_model': None,
            'active_learning_insights': {},
            'recommendations': []
        }
        
        # Step 1: Source type classification
        self.logger.info("Performing source type classification...")
        predicted_type, confidence_scores = await self.classify_source_type(observations)
        results['classification'] = {
            'predicted_type': predicted_type.value,
            'confidence_scores': confidence_scores
        }
        
        # Step 2: Parameter optimization for multiple source types
        self.logger.info("Optimizing parameters for candidate source types...")
        candidate_types = [
            SourceType.POINT_SOURCE,
            SourceType.DIKE,
            SourceType.SILL,
            SourceType.FAULT
        ]
        
        candidate_models = []
        for source_type in candidate_types:
            bounds = self._get_parameter_bounds(source_type)
            try:
                model_params = await self.optimize_parameters(observations, source_type, bounds)
                candidate_models.append(model_params)
                self.logger.info(f"Optimized {source_type.value}: AIC = {model_params.aic:.2f}")
            except Exception as e:
                self.logger.error(f"Failed to optimize {source_type.value}: {str(e)}")
        
        results['models'] = [
            {
                'source_type': model.source_type.value,
                'parameters': model.parameters,
                'uncertainty': model.uncertainty,
                'likelihood': model.likelihood,
                'aic': model.aic,
                'bic': model.bic
            }
            for model in candidate_models
        ]
        
        # Step 3: Model selection
        if candidate_models:
            best_model = self.model_selection(candidate_models)
            results['best_model'] = {
                'source_type': best_model.source_type.value,
                'parameters': best_model.parameters,
                'uncertainty': best_model.uncertainty,
                'likelihood': best_model.likelihood,
                'aic': best_model.aic,
                'bic': best_model.bic
            }
        
        # Step 4: Active learning insights
        if self.active_learner is None:
            self.active_learner = ActiveLearningEngine(candidate_models)
        
        hypothesis = self.active_learner.form_hypothesis(
            observations.displacements, 
            {model.source_type.value: model for model in candidate_models}
        )
        results['active_learning_insights'] = hypothesis
        
        # Step 5: Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Store results
        self.inversion_history.append(results)
        
        return results
    
    def _get_parameter_bounds(self, source_type: SourceType) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for different source types"""
        bounds = {
            SourceType.POINT_SOURCE: {
                'x0': (-10000, 10000),
                'y0': (-10000, 10000),
                'z0': (100, 10000),
                'volume_change': (1e3, 1e9)
            },
            SourceType.DIKE: {
                'x0': (-10000, 10000),
                'y0': (-10000, 10000),
                'z0': (100, 5000),
                'length': (100, 10000),
                'width': (10, 5000),
                'opening': (0.01, 10.0),
                'strike': (0, 360),
                'dip': (0, 90)
            },
            SourceType.SILL: {
                'x0': (-10000, 10000),
                'y0': (-10000, 10000),
                'z0': (100, 10000),
                'length': (100, 20000),
                'width': (100, 20000),
                'opening': (0.01, 10.0)
            },
            SourceType.FAULT: {
                'x0': (-10000, 10000),
                'y0': (-10000, 10000),
                'z0': (100, 20000),
                'length': (100, 50000),
                'width': (100, 20000),
                'slip': (0.01, 20.0),
                'strike': (0, 360),
                'dip': (0, 90),
                'rake': (-180, 180)
            }
        }
        
        return bounds.get(source_type, {})
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Check classification confidence
        max_confidence = max(results['classification']['confidence_scores'].values())
        if max_confidence < 0.7:
            recommendations.append({
                'type': 'data_collection',
                'priority': 'high',
                'description': 'Low classification confidence suggests need for additional data or different observation geometry'
            })
        
        # Check parameter uncertainties
        if results['best_model']:
            uncertainties = results['best_model']['uncertainty']
            high_uncertainty_params = [
                param for param, unc in uncertainties.items() 
                if unc == float('inf') or unc > 0.5 * abs(results['best_model']['parameters'].get(param, 1))
            ]
            
            if high_uncertainty_params:
                recommendations.append({
                    'type': 'parameter_uncertainty',
                    'priority': 'medium',
                    'description': f'High uncertainty in parameters: {", ".join(high_uncertainty_params)}. Consider constraining priors or collecting more data.'
                })
        
        # Check active learning insights
        if 'active_learning_insights' in results:
            insights = results['active_learning_insights']
            if insights['confidence_score'] < 0.6:
                recommendations.append({
                    'type': 'model_improvement',
                    'priority': 'medium',
                    'description': 'Low model confidence suggests need for more complex models or additional physics'
                })
            
            for modification in insights.get('suggested_modifications', []):
                if modification['priority'] > 0.7:
                    recommendations.append({
                        'type': 'model_modification',
                        'priority': 'high',
                        'description': modification['description']
                    })
        
        return recommendations
    
    def export_results(self, results: Dict[str, Any], filepath: str):
        """Export characterization results to JSON file"""
        # Convert numpy arrays and other non-serializable objects
        serializable_results = json.loads(json.dumps(results, default=str))
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results exported to {filepath}")

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create synthetic observation data
        n_points = 100
        coordinates = np.random.rand(n_points, 3) * 1000
        displacements = np.random.normal(0, 0.01, (n_points, 3))
        uncertainties = np.full_like(displacements, 0.005)
        
        observations = ObservationData(
            coordinates=coordinates,
            displacements=displacements,
            uncertainties=uncertainties,
            data_type='synthetic',
            acquisition_time=np.arange(n_points),
            metadata={'test': True}
        )
        
        # Initialize characterization system
        characterizer = SourceCharacterizationML()
        
        # Perform comprehensive analysis
        results = await characterizer.comprehensive_characterization(observations)
        
        # Export results
        characterizer.export_results(results, 'characterization_results.json')
        
        print("Source characterization completed successfully!")
        print(f"Best model: {results['best_model']['source_type']}")
        print(f"Classification confidence: {results['classification']['confidence_scores']}")
    
    # Run example
    asyncio.run(main())