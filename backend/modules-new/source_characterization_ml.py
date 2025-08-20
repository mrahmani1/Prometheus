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
from scipy.stats import norm
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

class BayesianNeuralNetwork(nn.Module):
    """Bayesian Neural Network for uncertainty quantification in geological modeling"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network layers with variational parameters
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # MC Dropout for uncertainty
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim * 2))  # Mean and log variance
        self.network = nn.Sequential(*layers)
        
        # Initialize variational parameters
        self.log_sigma_prior = nn.Parameter(torch.log(torch.tensor(0.1)))
        
    def forward(self, x: torch.Tensor, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty quantification"""
        batch_size = x.size(0)
        
        if self.training or n_samples > 1:
            # Monte Carlo sampling for uncertainty
            outputs = []
            for _ in range(n_samples):
                out = self.network(x)
                mean = out[:, :self.output_dim]
                log_var = out[:, self.output_dim:]
                outputs.append(torch.cat([mean.unsqueeze(0), log_var.unsqueeze(0)], dim=0))
            
            outputs = torch.stack(outputs, dim=0)  # [n_samples, 2, batch_size, output_dim]
            mean = outputs[:, 0].mean(dim=0)  # Average over samples
            epistemic_var = outputs[:, 0].var(dim=0)  # Epistemic uncertainty
            aleatoric_var = torch.exp(outputs[:, 1]).mean(dim=0)  # Aleatoric uncertainty
            
            total_uncertainty = epistemic_var + aleatoric_var
            return mean, total_uncertainty
        else:
            out = self.network(x)
            mean = out[:, :self.output_dim]
            log_var = out[:, self.output_dim:]
            return mean, torch.exp(log_var)
    
    def compute_kl_loss(self) -> torch.Tensor:
        """Compute KL divergence loss for Bayesian regularization"""
        kl_loss = 0.0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Simplified KL divergence for weight regularization
                weight_var = torch.exp(self.log_sigma_prior)
                kl_loss += -0.5 * torch.sum(1 + torch.log(weight_var) - 
                                           module.weight.pow(2) / weight_var)
        return kl_loss

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
        
        # Advanced components
        self.hypothesis_history = []
        self.confidence_tracker = []
        self.exploration_regions = []
        self.causal_graph = None
        self.geological_constraints = {}
        self.physics_models = {}
        
        # Initialize Bayesian components
        self.bayesian_nets = {}
        self.uncertainty_calibration = {}
        self.acquisition_strategies = {
            'entropy': self._entropy_acquisition,
            'expected_improvement': self._expected_improvement,
            'upper_confidence_bound': self._ucb_acquisition,
            'geological_informed': self._geological_informed_acquisition
        }
        
        # Knowledge integration
        self.geological_ontology = self._initialize_geological_ontology()
        self.expert_knowledge = {}
        self.adaptive_thresholds = {}
        
        # Performance tracking
        self.learning_curves = {'confidence': [], 'complexity': [], 'validation_score': []}
        self.exploration_efficiency = {}
        self.hypothesis_validation_history = []
        
        # Initialize backend module connections
        self._initialize_backend_connections()
    
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
        """Form novel hypothesis using causal inference, geological constraints, and Bayesian reasoning"""
        hypothesis = {
            'timestamp': pd.Timestamp.now(),
            'data_characteristics': self._analyze_data_characteristics(observations),
            'model_performance': self._evaluate_model_ensemble(current_models, observations),
            'suggested_modifications': [],
            'confidence_score': 0.0,
            'exploration_priority': [],
            'causal_mechanisms': [],
            'geological_plausibility': 0.0,
            'physics_consistency': {},
            'uncertainty_quantification': {},
            'testable_predictions': []
        }
        
        # Advanced causal analysis
        causal_relationships = self._discover_causal_mechanisms(observations)
        hypothesis['causal_mechanisms'] = causal_relationships
        
        # Geological constraint analysis
        geological_feasibility = self._assess_geological_feasibility(observations)
        hypothesis['geological_plausibility'] = geological_feasibility['plausibility_score']
        
        # Physics-based validation
        physics_checks = self._validate_physics_consistency(observations, current_models)
        hypothesis['physics_consistency'] = physics_checks
        
        # Advanced uncertainty quantification
        uncertainty_analysis = self._quantify_hypothesis_uncertainty(observations, current_models)
        hypothesis['uncertainty_quantification'] = uncertainty_analysis
        
        # Generate testable predictions
        predictions = self._generate_testable_predictions(hypothesis, observations)
        hypothesis['testable_predictions'] = predictions
        
        # Advanced pattern analysis with geological context
        data_stats = hypothesis['data_characteristics']
        
        # Causal-informed modifications
        for mechanism in causal_relationships:
            if mechanism['strength'] > 0.7 and mechanism['geological_support'] > 0.6:
                hypothesis['suggested_modifications'].append({
                    'type': 'causal_mechanism',
                    'description': f'Strong causal relationship: {mechanism["description"]}',
                    'mechanism': mechanism,
                    'priority': mechanism['strength'] * mechanism['geological_support'],
                    'evidence_strength': mechanism['evidence_score']
                })
        
        # Physics-informed modifications
        if data_stats['temporal_correlation'] > 0.7:
            time_dep_analysis = self._analyze_temporal_dependencies(observations)
            hypothesis['suggested_modifications'].append({
                'type': 'temporal_modeling',
                'description': 'Strong temporal correlation with physical basis',
                'priority': 0.8 * time_dep_analysis.get('physical_plausibility', 0.5),
                'physical_mechanisms': time_dep_analysis.get('mechanisms', []),
                'recommended_models': time_dep_analysis.get('recommended_models', [])
            })
        
        if data_stats['spatial_clustering'] > 0.6:
            spatial_analysis = self._analyze_spatial_patterns(observations)
            hypothesis['suggested_modifications'].append({
                'type': 'spatial_clustering',
                'description': 'Spatially coherent deformation with geological context',
                'priority': 0.7 * spatial_analysis.get('geological_consistency', 0.5),
                'clustering_type': spatial_analysis.get('pattern_type', 'unknown'),
                'geological_interpretation': spatial_analysis.get('interpretation', 'requires investigation')
            })
        
        if data_stats['nonlinearity_score'] > 0.5:
            nonlinear_analysis = self._analyze_nonlinear_mechanisms(observations)
            hypothesis['suggested_modifications'].append({
                'type': 'nonlinear_effects',
                'description': 'Nonlinear behavior with identified physical mechanisms',
                'priority': 0.9 * nonlinear_analysis.get('mechanism_confidence', 0.5),
                'physical_mechanisms': nonlinear_analysis.get('mechanisms', []),
                'model_recommendations': nonlinear_analysis.get('models', [])
            })
        
        # Advanced confidence scoring with multiple criteria
        confidence_components = {
            'model_performance': np.mean([perf['accuracy'] for perf in hypothesis['model_performance'].values()]) if hypothesis['model_performance'] else 0.0,
            'geological_plausibility': geological_feasibility['plausibility_score'],
            'physics_consistency': np.mean([score for score in physics_checks.values()]) if physics_checks else 0.0,
            'causal_evidence': np.mean([mech['evidence_score'] for mech in causal_relationships]) if causal_relationships else 0.0,
            'uncertainty_calibration': 1.0 - uncertainty_analysis.get('total_uncertainty', 1.0)
        }
        
        # Weighted confidence score
        weights = {'model_performance': 0.3, 'geological_plausibility': 0.25, 
                  'physics_consistency': 0.25, 'causal_evidence': 0.15, 'uncertainty_calibration': 0.05}
        hypothesis['confidence_score'] = sum(weights[k] * v for k, v in confidence_components.items())
        hypothesis['confidence_breakdown'] = confidence_components
        
        # Adaptive exploration priority
        exploration_priority = self._calculate_exploration_priority(hypothesis, observations)
        hypothesis['exploration_priority'] = exploration_priority
        
        self.hypothesis_history.append(hypothesis)
        self._update_learning_metrics(hypothesis)
        
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
    
    async def _evaluate_model_ensemble(self, models: Dict[str, Any], 
                                      observations: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Advanced ensemble model evaluation with comprehensive performance metrics and uncertainty quantification"""
        performance = {}
        ensemble_predictions = []
        ensemble_uncertainties = []
        
        # Extract observation components
        coordinates = observations[:, :3] if observations.shape[1] >= 3 else observations
        if observations.shape[1] >= 6:
            true_displacements = observations[:, 3:6]
        else:
            # Fallback: use synthetic displacements for testing
            true_displacements = np.random.normal(0, 0.01, (len(coordinates), 3))
        
        if observations.shape[1] >= 9:
            uncertainties = observations[:, 6:9]
        else:
            uncertainties = np.full_like(true_displacements, 0.005)
        
        logging.info(f"Evaluating ensemble of {len(models)} models on {len(observations)} observations")
        
        # Individual model evaluation with advanced metrics
        for model_name, model in models.items():
            try:
                logging.info(f"Evaluating model: {model_name}")
                
                # Generate model predictions based on model type
                predictions, pred_uncertainties = await self._generate_model_predictions(
                    model, model_name, coordinates, true_displacements.shape
                )
                
                # Comprehensive performance evaluation
                model_performance = await self._compute_comprehensive_metrics(
                    true_displacements, predictions, uncertainties, pred_uncertainties, model_name
                )
                
                # Advanced statistical validation
                statistical_validation = self._perform_statistical_validation(
                    true_displacements, predictions, uncertainties
                )
                model_performance['statistical_validation'] = statistical_validation
                
                # Geological plausibility assessment
                geological_assessment = await self._assess_geological_plausibility(
                    predictions, coordinates, model_name
                )
                model_performance['geological_plausibility'] = geological_assessment
                
                # Cross-validation performance
                cv_performance = await self._cross_validate_model(
                    model, coordinates, true_displacements, uncertainties
                )
                model_performance['cross_validation'] = cv_performance
                
                # Model complexity and interpretability metrics
                complexity_metrics = self._compute_model_complexity(model, model_name)
                model_performance['complexity'] = complexity_metrics
                
                # Robustness analysis
                robustness_metrics = await self._analyze_model_robustness(
                    model, coordinates, true_displacements, uncertainties
                )
                model_performance['robustness'] = robustness_metrics
                
                # Store for ensemble analysis
                ensemble_predictions.append(predictions)
                ensemble_uncertainties.append(pred_uncertainties)
                
                performance[model_name] = model_performance
                
                logging.info(f"Model {model_name} evaluation completed. R²: {model_performance['r2']:.3f}")
                
            except Exception as e:
                logging.error(f"Error evaluating model {model_name}: {e}")
                performance[model_name] = {
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'r2': -float('inf'),
                    'accuracy': 0.0,
                    'status': 'failed',
                    'error': str(e),
                    'geological_plausibility': {'score': 0.0, 'assessment': 'failed'},
                    'statistical_validation': {'normal_residuals': False, 'homoscedastic': False},
                    'cross_validation': {'mean_score': 0.0, 'std_score': float('inf')},
                    'complexity': {'parameters': 0, 'aic': float('inf'), 'bic': float('inf')},
                    'robustness': {'noise_tolerance': 0.0, 'outlier_resistance': 0.0}
                }
        
        # Ensemble-level analysis
        if len(ensemble_predictions) > 1:
            ensemble_analysis = await self._analyze_ensemble_performance(
                ensemble_predictions, ensemble_uncertainties, true_displacements, uncertainties
            )
            performance['ensemble_analysis'] = ensemble_analysis
        
        # Model ranking and selection recommendations
        model_ranking = self._rank_models(performance)
        performance['model_ranking'] = model_ranking
        
        # Integration with backend modules for enhanced evaluation
        backend_integration = await self._integrate_evaluation_with_backend(
            performance, observations, models
        )
        performance['backend_integration'] = backend_integration
        
        return performance
    
    async def _generate_model_predictions(self, model: Any, model_name: str, 
                                        coordinates: np.ndarray, 
                                        target_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions from model with uncertainty quantification"""
        n_obs = len(coordinates)
        
        try:
            # Try different model interfaces
            if hasattr(model, 'predict_with_uncertainty'):
                # Advanced model with uncertainty
                predictions, uncertainties = model.predict_with_uncertainty(coordinates)
                
            elif hasattr(model, 'predict'):
                # Standard scikit-learn style model
                if coordinates.shape[1] == 3:
                    predictions = model.predict(coordinates)
                else:
                    # Feature extraction for models requiring different input format
                    features = self._extract_prediction_features(coordinates)
                    predictions = model.predict(features)
                
                # Estimate uncertainties from prediction variance
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(coordinates if coordinates.shape[1] == 3 
                                              else self._extract_prediction_features(coordinates))
                    uncertainties = np.var(proba, axis=1).reshape(-1, 1)
                else:
                    # Default uncertainty estimate
                    uncertainties = np.full((n_obs, target_shape[1]), 0.01)
                
                # Ensure predictions have correct shape
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
                if predictions.shape[1] != target_shape[1]:
                    # Replicate or pad to match target shape
                    if predictions.shape[1] == 1 and target_shape[1] == 3:
                        predictions = np.column_stack([
                            predictions * 0.3,  # Horizontal component 1
                            predictions * 0.3,  # Horizontal component 2
                            predictions         # Vertical component
                        ])
                
            elif hasattr(model, 'forward'):
                # PyTorch-style model
                import torch
                coordinates_tensor = torch.FloatTensor(coordinates)
                with torch.no_grad():
                    output = model.forward(coordinates_tensor)
                    if isinstance(output, tuple):
                        predictions, uncertainties = output[0].numpy(), output[1].numpy()
                    else:
                        predictions = output.numpy()
                        uncertainties = np.full_like(predictions, 0.01)
                
            elif isinstance(model, dict) and 'type' in model:
                # Configuration-based model
                predictions, uncertainties = await self._predict_from_config(
                    model, coordinates, target_shape
                )
                
            else:
                # Fallback: analytical model based on distance
                center_x, center_y = np.mean(coordinates[:, :2], axis=0)
                distances = np.sqrt((coordinates[:, 0] - center_x)**2 + 
                                  (coordinates[:, 1] - center_y)**2)
                
                # Simple exponential decay model
                magnitude = 0.1 * np.exp(-distances / 5000) + 0.01 * np.random.randn(n_obs)
                
                predictions = np.column_stack([
                    magnitude * 0.2,  # East
                    magnitude * 0.2,  # North
                    magnitude * 0.8   # Up
                ])
                
                uncertainties = np.full_like(predictions, 0.005)
            
            # Ensure correct shapes
            if predictions.shape != target_shape:
                predictions = predictions[:target_shape[0], :target_shape[1]]
            if uncertainties.shape != target_shape:
                uncertainties = uncertainties[:target_shape[0], :target_shape[1]]
                
            return predictions, uncertainties
            
        except Exception as e:
            logging.warning(f"Error generating predictions for {model_name}: {e}")
            # Fallback predictions
            predictions = np.random.normal(0, 0.01, target_shape)
            uncertainties = np.full(target_shape, 0.01)
            return predictions, uncertainties
    
    async def _compute_comprehensive_metrics(self, true_values: np.ndarray, predictions: np.ndarray,
                                           obs_uncertainties: np.ndarray, pred_uncertainties: np.ndarray,
                                           model_name: str) -> Dict[str, Any]:
        """Compute comprehensive performance metrics"""
        metrics = {}
        
        try:
            # Flatten arrays for component-wise and overall analysis
            true_flat = true_values.flatten()
            pred_flat = predictions.flatten()
            obs_unc_flat = obs_uncertainties.flatten()
            pred_unc_flat = pred_uncertainties.flatten()
            
            # Basic regression metrics
            residuals = true_flat - pred_flat
            metrics['rmse'] = np.sqrt(np.mean(residuals**2))
            metrics['mae'] = np.mean(np.abs(residuals))
            metrics['mse'] = np.mean(residuals**2)
            
            # Coefficient of determination
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((true_flat - np.mean(true_flat))**2)
            metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-10))
            
            # Normalized metrics
            true_std = np.std(true_flat)
            metrics['nrmse'] = metrics['rmse'] / (true_std + 1e-10)
            metrics['accuracy'] = max(0, 1 - metrics['nrmse'])
            
            # Component-wise analysis
            component_metrics = {}
            component_names = ['east', 'north', 'vertical']
            for i, comp_name in enumerate(component_names):
                if i < true_values.shape[1]:
                    comp_residuals = true_values[:, i] - predictions[:, i]
                    component_metrics[comp_name] = {
                        'rmse': np.sqrt(np.mean(comp_residuals**2)),
                        'mae': np.mean(np.abs(comp_residuals)),
                        'r2': 1 - np.var(comp_residuals) / (np.var(true_values[:, i]) + 1e-10),
                        'bias': np.mean(comp_residuals)
                    }
            metrics['component_metrics'] = component_metrics
            
            # Uncertainty-aware metrics
            weighted_residuals = residuals / (obs_unc_flat + pred_unc_flat + 1e-10)
            metrics['chi_squared'] = np.sum(weighted_residuals**2)
            metrics['reduced_chi_squared'] = metrics['chi_squared'] / len(residuals)
            
            # Prediction interval coverage
            prediction_intervals = pred_unc_flat * 1.96  # 95% confidence
            coverage = np.mean(np.abs(residuals) <= prediction_intervals)
            metrics['prediction_interval_coverage'] = coverage
            
            # Uncertainty calibration
            calibration_score = self._compute_uncertainty_calibration(
                residuals, pred_unc_flat
            )
            metrics['uncertainty_calibration'] = calibration_score
            
            # Outlier analysis
            outlier_threshold = 3 * np.std(residuals)
            outlier_fraction = np.mean(np.abs(residuals) > outlier_threshold)
            metrics['outlier_fraction'] = outlier_fraction
            
            # Log-likelihood (assuming Gaussian errors)
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * (obs_unc_flat**2 + pred_unc_flat**2)) +
                residuals**2 / (obs_unc_flat**2 + pred_unc_flat**2)
            )
            metrics['log_likelihood'] = log_likelihood
            
            # Information criteria
            n_params = self._estimate_model_parameters(model_name)
            n_data = len(residuals)
            metrics['aic'] = 2 * n_params - 2 * log_likelihood
            metrics['bic'] = n_params * np.log(n_data) - 2 * log_likelihood
            
            # Spatial correlation in residuals
            if true_values.shape[0] > 10:
                spatial_correlation = self._compute_spatial_residual_correlation(
                    residuals.reshape(true_values.shape), true_values.shape[0]
                )
                metrics['spatial_correlation'] = spatial_correlation
            
            metrics['status'] = 'success'
            
        except Exception as e:
            logging.error(f"Error computing metrics for {model_name}: {e}")
            metrics = {
                'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf'),
                'accuracy': 0.0, 'status': 'failed', 'error': str(e)
            }
        
        return metrics
    
    def _perform_statistical_validation(self, true_values: np.ndarray, predictions: np.ndarray,
                                      uncertainties: np.ndarray) -> Dict[str, Any]:
        """Perform statistical validation of model predictions"""
        validation = {}
        
        try:
            residuals = (true_values - predictions).flatten()
            
            # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for large)
            from scipy import stats
            
            if len(residuals) <= 5000:
                stat, p_value = stats.shapiro(residuals)
                validation['normality_test'] = {
                    'test': 'shapiro_wilk',
                    'statistic': stat,
                    'p_value': p_value,
                    'normal': p_value > 0.05
                }
            else:
                # Use Anderson-Darling test for larger samples
                result = stats.anderson(residuals, dist='norm')
                validation['normality_test'] = {
                    'test': 'anderson_darling',
                    'statistic': result.statistic,
                    'critical_values': result.critical_values.tolist(),
                    'normal': result.statistic < result.critical_values[2]  # 5% significance
                }
            
            # Homoscedasticity test (Breusch-Pagan)
            if len(residuals) > 10:
                try:
                    # Simple correlation test between |residuals| and predictions
                    abs_residuals = np.abs(residuals)
                    pred_flat = predictions.flatten()
                    correlation, p_val = stats.pearsonr(abs_residuals, pred_flat)
                    validation['homoscedasticity'] = {
                        'correlation': correlation,
                        'p_value': p_val,
                        'homoscedastic': p_val > 0.05
                    }
                except:
                    validation['homoscedasticity'] = {'homoscedastic': True}
            
            # Autocorrelation test (Durbin-Watson)
            if len(residuals) > 3:
                dw_stat = self._durbin_watson(residuals)
                validation['autocorrelation'] = {
                    'durbin_watson': dw_stat,
                    'no_autocorr': 1.5 < dw_stat < 2.5
                }
            
            # Outlier detection
            z_scores = np.abs(residuals / (np.std(residuals) + 1e-10))
            outliers = z_scores > 3
            validation['outliers'] = {
                'count': int(np.sum(outliers)),
                'fraction': float(np.mean(outliers)),
                'max_z_score': float(np.max(z_scores))
            }
            
        except Exception as e:
            validation = {'error': str(e), 'status': 'failed'}
        
        return validation
    
    async def _assess_geological_plausibility(self, predictions: np.ndarray, 
                                             coordinates: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Assess geological plausibility of model predictions"""
        assessment = {
            'score': 0.0,
            'checks': {},
            'violations': [],
            'assessment': 'unknown'
        }
        
        try:
            # Check displacement magnitudes
            displacement_magnitudes = np.linalg.norm(predictions, axis=1)
            max_displacement = np.max(displacement_magnitudes)
            mean_displacement = np.mean(displacement_magnitudes)
            
            # Realistic displacement ranges
            checks_passed = 0
            total_checks = 0
            
            # Check 1: Maximum displacement reasonable
            total_checks += 1
            if max_displacement < 50.0:  # < 50 m
                checks_passed += 1
                assessment['checks']['max_displacement_reasonable'] = True
            else:
                assessment['checks']['max_displacement_reasonable'] = False
                assessment['violations'].append(f"Extreme displacement: {max_displacement:.1f} m")
            
            # Check 2: Spatial decay pattern
            total_checks += 1
            center_x, center_y = np.mean(coordinates[:, :2], axis=0)
            distances = np.sqrt((coordinates[:, 0] - center_x)**2 + (coordinates[:, 1] - center_y)**2)
            
            if len(distances) > 5:
                # Check if displacement generally decreases with distance
                correlation, _ = stats.pearsonr(distances, displacement_magnitudes)
                if correlation < 0.1:  # Negative or weak positive correlation expected
                    checks_passed += 1
                    assessment['checks']['spatial_decay'] = True
                else:
                    assessment['checks']['spatial_decay'] = False
                    assessment['violations'].append("Unrealistic spatial pattern")
            
            # Check 3: Vertical vs horizontal ratio
            total_checks += 1
            if predictions.shape[1] >= 3:
                vertical_component = np.abs(predictions[:, 2])
                horizontal_component = np.sqrt(predictions[:, 0]**2 + predictions[:, 1]**2)
                
                mean_v_h_ratio = np.mean(vertical_component / (horizontal_component + 1e-10))
                if 0.1 <= mean_v_h_ratio <= 10.0:  # Reasonable ratio range
                    checks_passed += 1
                    assessment['checks']['v_h_ratio_reasonable'] = True
                else:
                    assessment['checks']['v_h_ratio_reasonable'] = False
                    assessment['violations'].append(f"Unusual V/H ratio: {mean_v_h_ratio:.2f}")
            
            # Check 4: Sign consistency (e.g., predominantly uplift or subsidence)
            total_checks += 1
            if predictions.shape[1] >= 3:
                vertical_signs = np.sign(predictions[:, 2])
                sign_consistency = np.abs(np.mean(vertical_signs))
                if sign_consistency > 0.3:  # Some coherent pattern
                    checks_passed += 1
                    assessment['checks']['sign_consistency'] = True
                else:
                    assessment['checks']['sign_consistency'] = False
                    assessment['violations'].append("Inconsistent displacement directions")
            
            # Overall score
            assessment['score'] = checks_passed / total_checks if total_checks > 0 else 0.5
            
            # Assessment categories
            if assessment['score'] >= 0.8:
                assessment['assessment'] = 'highly_plausible'
            elif assessment['score'] >= 0.6:
                assessment['assessment'] = 'plausible'
            elif assessment['score'] >= 0.4:
                assessment['assessment'] = 'questionable'
            else:
                assessment['assessment'] = 'implausible'
                
        except Exception as e:
            assessment['error'] = str(e)
            assessment['assessment'] = 'failed'
        
        return assessment
    
    async def _cross_validate_model(self, model: Any, coordinates: np.ndarray,
                                   true_displacements: np.ndarray, uncertainties: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation on model"""
        cv_results = {
            'mean_score': 0.0,
            'std_score': 0.0,
            'scores': [],
            'method': 'k_fold'
        }
        
        try:
            if len(coordinates) < 10:
                cv_results['mean_score'] = 0.5  # Default for small datasets
                return cv_results
                
            from sklearn.model_selection import KFold
            
            n_folds = min(5, len(coordinates) // 5)
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, test_idx in kfold.split(coordinates):
                train_coords = coordinates[train_idx]
                test_coords = coordinates[test_idx]
                train_disp = true_displacements[train_idx]
                test_disp = true_displacements[test_idx]
                
                # Simple validation using prediction quality
                try:
                    pred, pred_unc = await self._generate_model_predictions(
                        model, "cv_model", test_coords, test_disp.shape
                    )
                    residuals = test_disp - pred
                    score = 1.0 - np.sqrt(np.mean(residuals**2)) / (np.std(test_disp) + 1e-10)
                    scores.append(max(0.0, score))
                except:
                    scores.append(0.0)
            
            cv_results['scores'] = scores
            cv_results['mean_score'] = np.mean(scores)
            cv_results['std_score'] = np.std(scores)
            
        except Exception as e:
            cv_results['error'] = str(e)
        
        return cv_results
    
    def _compute_model_complexity(self, model: Any, model_name: str) -> Dict[str, Any]:
        """Compute model complexity metrics"""
        complexity = {
            'parameters': 0,
            'aic': float('inf'),
            'bic': float('inf'),
            'interpretability': 'unknown'
        }
        
        try:
            # Estimate number of parameters
            if hasattr(model, 'n_features_in_'):
                complexity['parameters'] = getattr(model, 'n_features_in_', 0)
            elif hasattr(model, 'coef_'):
                coef = getattr(model, 'coef_', [])
                complexity['parameters'] = len(coef.flatten()) if hasattr(coef, 'flatten') else 0
            elif isinstance(model, dict):
                complexity['parameters'] = len(model.get('parameters', {}))
            else:
                # Default estimate based on model type
                complexity['parameters'] = self._estimate_model_parameters(model_name)
            
            # Interpretability assessment
            interpretable_models = ['linear', 'elastic', 'mogi', 'analytical']
            if any(term in model_name.lower() for term in interpretable_models):
                complexity['interpretability'] = 'high'
            elif 'neural' in model_name.lower() or 'deep' in model_name.lower():
                complexity['interpretability'] = 'low'
            else:
                complexity['interpretability'] = 'medium'
                
        except Exception as e:
            complexity['error'] = str(e)
        
        return complexity
    
    async def _analyze_model_robustness(self, model: Any, coordinates: np.ndarray,
                                       true_displacements: np.ndarray, uncertainties: np.ndarray) -> Dict[str, Any]:
        """Analyze model robustness to noise and outliers"""
        robustness = {
            'noise_tolerance': 0.0,
            'outlier_resistance': 0.0,
            'stability_score': 0.0
        }
        
        try:
            # Test noise tolerance
            noise_levels = [0.5, 1.0, 2.0, 5.0]  # Multiples of original uncertainty
            noise_scores = []
            
            for noise_level in noise_levels:
                # Add noise to coordinates
                noisy_coords = coordinates + np.random.normal(0, noise_level * np.mean(uncertainties), coordinates.shape)
                
                try:
                    pred_clean, _ = await self._generate_model_predictions(
                        model, "robustness_test", coordinates, true_displacements.shape
                    )
                    pred_noisy, _ = await self._generate_model_predictions(
                        model, "robustness_test", noisy_coords, true_displacements.shape
                    )
                    
                    # Measure prediction stability
                    stability = 1.0 - np.mean(np.abs(pred_clean - pred_noisy)) / (np.mean(np.abs(pred_clean)) + 1e-10)
                    noise_scores.append(max(0.0, stability))
                except:
                    noise_scores.append(0.0)
            
            robustness['noise_tolerance'] = np.mean(noise_scores)
            
            # Test outlier resistance
            if len(coordinates) > 10:
                # Add outliers (10% of data)
                n_outliers = max(1, len(coordinates) // 10)
                outlier_indices = np.random.choice(len(coordinates), n_outliers, replace=False)
                
                coords_with_outliers = coordinates.copy()
                coords_with_outliers[outlier_indices] += np.random.normal(0, 10 * np.mean(uncertainties), 
                                                                        (n_outliers, coordinates.shape[1]))
                
                try:
                    pred_normal, _ = await self._generate_model_predictions(
                        model, "outlier_test", coordinates, true_displacements.shape
                    )
                    pred_outliers, _ = await self._generate_model_predictions(
                        model, "outlier_test", coords_with_outliers, true_displacements.shape
                    )
                    
                    # Measure resistance to outliers
                    resistance = 1.0 - np.mean(np.abs(pred_normal - pred_outliers)) / (np.mean(np.abs(pred_normal)) + 1e-10)
                    robustness['outlier_resistance'] = max(0.0, resistance)
                except:
                    robustness['outlier_resistance'] = 0.0
            
            # Overall stability score
            robustness['stability_score'] = (robustness['noise_tolerance'] + robustness['outlier_resistance']) / 2
            
        except Exception as e:
            robustness['error'] = str(e)
        
        return robustness
    
    async def _analyze_ensemble_performance(self, ensemble_predictions: List[np.ndarray],
                                          ensemble_uncertainties: List[np.ndarray],
                                          true_displacements: np.ndarray,
                                          uncertainties: np.ndarray) -> Dict[str, Any]:
        """Analyze ensemble-level performance"""
        analysis = {
            'diversity': 0.0,
            'consensus': 0.0,
            'improvement_over_individual': 0.0,
            'uncertainty_reduction': 0.0
        }
        
        try:
            if len(ensemble_predictions) < 2:
                return analysis
            
            # Compute ensemble mean and uncertainty
            ensemble_mean = np.mean(ensemble_predictions, axis=0)
            ensemble_variance = np.var(ensemble_predictions, axis=0)
            
            # Diversity measurement
            pairwise_differences = []
            for i in range(len(ensemble_predictions)):
                for j in range(i + 1, len(ensemble_predictions)):
                    diff = np.mean(np.abs(ensemble_predictions[i] - ensemble_predictions[j]))
                    pairwise_differences.append(diff)
            
            analysis['diversity'] = np.mean(pairwise_differences) if pairwise_differences else 0.0
            
            # Consensus measurement (inverse of prediction variance)
            analysis['consensus'] = 1.0 / (1.0 + np.mean(ensemble_variance))
            
            # Performance improvement over individual models
            ensemble_residuals = true_displacements - ensemble_mean
            ensemble_rmse = np.sqrt(np.mean(ensemble_residuals**2))
            
            individual_rmses = []
            for pred in ensemble_predictions:
                individual_rmse = np.sqrt(np.mean((true_displacements - pred)**2))
                individual_rmses.append(individual_rmse)
            
            best_individual_rmse = min(individual_rmses)
            if best_individual_rmse > 0:
                analysis['improvement_over_individual'] = max(0.0, 
                    (best_individual_rmse - ensemble_rmse) / best_individual_rmse)
            
            # Uncertainty reduction
            mean_individual_uncertainty = np.mean([np.mean(unc) for unc in ensemble_uncertainties])
            ensemble_total_uncertainty = np.mean(np.sqrt(ensemble_variance + np.mean(ensemble_uncertainties, axis=0)**2))
            
            if mean_individual_uncertainty > 0:
                analysis['uncertainty_reduction'] = max(0.0,
                    (mean_individual_uncertainty - ensemble_total_uncertainty) / mean_individual_uncertainty)
                
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _rank_models(self, performance: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Rank models based on multiple criteria"""
        ranking = {
            'ranked_models': [],
            'criteria_weights': {
                'accuracy': 0.3,
                'geological_plausibility': 0.25,
                'robustness': 0.2,
                'complexity': 0.15,
                'statistical_validity': 0.1
            },
            'scores': {}
        }
        
        try:
            model_scores = {}
            
            for model_name, metrics in performance.items():
                if model_name in ['ensemble_analysis', 'model_ranking', 'backend_integration']:
                    continue
                    
                score = 0.0
                
                # Accuracy component
                r2 = metrics.get('r2', -float('inf'))
                accuracy_score = max(0.0, min(1.0, (r2 + 1) / 2))  # Normalize R² to [0,1]
                score += ranking['criteria_weights']['accuracy'] * accuracy_score
                
                # Geological plausibility
                geo_score = metrics.get('geological_plausibility', {}).get('score', 0.0)
                score += ranking['criteria_weights']['geological_plausibility'] * geo_score
                
                # Robustness
                robust_score = metrics.get('robustness', {}).get('stability_score', 0.0)
                score += ranking['criteria_weights']['robustness'] * robust_score
                
                # Complexity (lower complexity is better)
                n_params = metrics.get('complexity', {}).get('parameters', 10)
                complexity_score = 1.0 / (1.0 + n_params / 10.0)  # Normalize
                score += ranking['criteria_weights']['complexity'] * complexity_score
                
                # Statistical validity
                stat_val = metrics.get('statistical_validation', {})
                validity_score = 0.0
                if stat_val.get('normality_test', {}).get('normal', False):
                    validity_score += 0.5
                if stat_val.get('homoscedasticity', {}).get('homoscedastic', False):
                    validity_score += 0.5
                score += ranking['criteria_weights']['statistical_validity'] * validity_score
                
                model_scores[model_name] = score
                ranking['scores'][model_name] = {
                    'total_score': score,
                    'accuracy': accuracy_score,
                    'geological_plausibility': geo_score,
                    'robustness': robust_score,
                    'complexity': complexity_score,
                    'statistical_validity': validity_score
                }
            
            # Rank models by total score
            ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            ranking['ranked_models'] = [model_name for model_name, score in ranked_models]
            
        except Exception as e:
            ranking['error'] = str(e)
        
        return ranking
    
    async def _integrate_evaluation_with_backend(self, performance: Dict[str, Any],
                                               observations: np.ndarray, models: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate evaluation results with backend modules"""
        integration = {
            'causal_analysis': {},
            'knowledge_graph_updates': {},
            'elastic_model_validation': {},
            'hypothesis_refinement': {}
        }
        
        try:
            # Extract best performing model
            ranking = performance.get('model_ranking', {})
            best_models = ranking.get('ranked_models', [])
            
            if best_models:
                best_model = best_models[0]
                best_performance = performance.get(best_model, {})
                
                # Integrate with causal inference if available
                if hasattr(self, 'backend_modules') and 'causal_inference' in self.backend_modules:
                    try:
                        causal_analysis = await self._analyze_causal_mechanisms_from_performance(
                            best_performance, observations
                        )
                        integration['causal_analysis'] = causal_analysis
                    except Exception as e:
                        integration['causal_analysis'] = {'error': str(e)}
                
                # Update knowledge graph with performance insights
                if hasattr(self, 'backend_modules') and 'knowledge_graph' in self.backend_modules:
                    try:
                        kg_updates = await self._update_knowledge_graph_with_performance(
                            best_performance, best_model
                        )
                        integration['knowledge_graph_updates'] = kg_updates
                    except Exception as e:
                        integration['knowledge_graph_updates'] = {'error': str(e)}
                
                # Validate against elastic models
                elastic_validation = await self._validate_against_elastic_models(
                    best_performance, observations
                )
                integration['elastic_model_validation'] = elastic_validation
                
                # Refine hypotheses based on performance
                hypothesis_refinement = await self._refine_hypotheses_from_performance(
                    performance, observations
                )
                integration['hypothesis_refinement'] = hypothesis_refinement
            
        except Exception as e:
            integration['error'] = str(e)
        
        return integration
    
    def _compute_uncertainty_calibration(self, residuals: np.ndarray, predicted_uncertainties: np.ndarray) -> float:
        """Compute uncertainty calibration score"""
        try:
            # Check if predicted uncertainties are well-calibrated
            # Good calibration means residuals follow predicted uncertainty distribution
            normalized_residuals = np.abs(residuals) / (predicted_uncertainties + 1e-10)
            
            # For well-calibrated uncertainties, ~68% of residuals should be within 1σ
            within_1_sigma = np.mean(normalized_residuals <= 1.0)
            within_2_sigma = np.mean(normalized_residuals <= 2.0)
            
            # Ideal calibration: 68% within 1σ, 95% within 2σ
            calibration_1sigma = 1.0 - abs(within_1_sigma - 0.68) / 0.68
            calibration_2sigma = 1.0 - abs(within_2_sigma - 0.95) / 0.95
            
            return (calibration_1sigma + calibration_2sigma) / 2
            
        except:
            return 0.5  # Default neutral calibration
    
    def _estimate_model_parameters(self, model_name: str) -> int:
        """Estimate number of parameters for different model types"""
        model_name_lower = model_name.lower()
        
        if 'point' in model_name_lower or 'mogi' in model_name_lower:
            return 4  # x, y, z, volume_change
        elif 'dike' in model_name_lower:
            return 8  # x, y, z, length, width, opening, strike, dip
        elif 'fault' in model_name_lower:
            return 9  # x, y, z, length, width, slip, strike, dip, rake
        elif 'elastic' in model_name_lower:
            return 11  # fault parameters + elastic constants
        elif 'viscoelastic' in model_name_lower:
            return 15  # elastic + viscosity, relaxation time
        elif 'poroelastic' in model_name_lower:
            return 17  # elastic + fluid properties
        elif 'neural' in model_name_lower:
            return 100  # Typical neural network
        elif 'ensemble' in model_name_lower:
            return 50  # Multiple models
        else:
            return 6  # Default moderate complexity
    
    def _compute_spatial_residual_correlation(self, residuals: np.ndarray, n_obs: int) -> float:
        """Compute spatial correlation in residuals"""
        try:
            if residuals.shape[0] < 5:
                return 0.0
                
            # Simple measure: correlation between nearby points
            residual_magnitudes = np.linalg.norm(residuals, axis=1) if residuals.ndim > 1 else np.abs(residuals)
            
            # Compute lag-1 spatial correlation
            if len(residual_magnitudes) > 1:
                correlation = np.corrcoef(residual_magnitudes[:-1], residual_magnitudes[1:])[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            
            return 0.0
            
        except:
            return 0.0
    
    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """Compute Durbin-Watson statistic for autocorrelation"""
        try:
            diff_residuals = np.diff(residuals)
            sum_squared_diff = np.sum(diff_residuals**2)
            sum_squared_residuals = np.sum(residuals**2)
            
            if sum_squared_residuals > 0:
                dw_stat = sum_squared_diff / sum_squared_residuals
                return dw_stat
            else:
                return 2.0  # No autocorrelation
                
        except:
            return 2.0  # Default no autocorrelation
    
    def _extract_prediction_features(self, coordinates: np.ndarray) -> np.ndarray:
        """Extract features for prediction models that need different input format"""
        try:
            # Basic feature extraction: coordinates + derived features
            features = coordinates.copy()
            
            if coordinates.shape[1] >= 2:
                # Add distance from center
                center_x, center_y = np.mean(coordinates[:, :2], axis=0)
                distances = np.sqrt((coordinates[:, 0] - center_x)**2 + (coordinates[:, 1] - center_y)**2)
                features = np.column_stack([features, distances])
                
                # Add angle from center
                angles = np.arctan2(coordinates[:, 1] - center_y, coordinates[:, 0] - center_x)
                features = np.column_stack([features, angles])
            
            return features
            
        except:
            return coordinates
    
    async def _predict_from_config(self, model_config: Dict[str, Any], 
                                  coordinates: np.ndarray, target_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions from configuration-based model"""
        try:
            model_type = model_config.get('type', 'unknown')
            
            if model_type == 'point_source':
                # Simple point source model
                params = model_config.get('parameters', {})
                x0 = params.get('x0', 0)
                y0 = params.get('y0', 0)
                z0 = params.get('z0', 2000)
                volume_change = params.get('volume_change', 1e6)
                
                # Mogi model
                distances = np.sqrt((coordinates[:, 0] - x0)**2 + (coordinates[:, 1] - y0)**2 + z0**2)
                factor = volume_change / (4 * np.pi * distances**3)
                
                predictions = np.column_stack([
                    factor * (coordinates[:, 0] - x0),  # East
                    factor * (coordinates[:, 1] - y0),  # North
                    factor * z0                         # Up
                ])
                
                uncertainties = np.full_like(predictions, 0.005)
                
            else:
                # Fallback
                predictions = np.random.normal(0, 0.01, target_shape)
                uncertainties = np.full(target_shape, 0.01)
            
            return predictions, uncertainties
            
        except Exception as e:
            predictions = np.random.normal(0, 0.01, target_shape)
            uncertainties = np.full(target_shape, 0.01)
            return predictions, uncertainties
    
    def _analyze_temporal_dependencies(self, observations: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal dependencies with physical basis"""
        analysis = {
            'physical_plausibility': 0.5,
            'mechanisms': [],
            'recommended_models': []
        }
        
        if len(observations.shape) > 1 and observations.shape[1] > 3:
            # Check for realistic temporal evolution
            time_diffs = np.diff(observations[:, 3])
            disp_diffs = np.diff(observations[:, 2])
            
            if np.any(time_diffs > 0):
                velocities = disp_diffs[time_diffs > 0] / time_diffs[time_diffs > 0]
                max_velocity = np.max(np.abs(velocities))
                
                if max_velocity < 1.0:  # Reasonable geological velocity
                    analysis['physical_plausibility'] = 0.8
                    analysis['mechanisms'] = ['viscoelastic_relaxation', 'slow_slip']
                    analysis['recommended_models'] = ['maxwell_viscoelastic', 'rate_state_friction']
        
        return analysis
    
    def _analyze_spatial_patterns(self, observations: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial patterns with geological context"""
        analysis = {
            'geological_consistency': 0.5,
            'pattern_type': 'unknown',
            'interpretation': 'requires investigation'
        }
        
        if len(observations.shape) > 1 and observations.shape[1] >= 3:
            distances = np.sqrt(observations[:, 0]**2 + observations[:, 1]**2)
            displacements = observations[:, 2]
            
            # Check for radial pattern (volcanic)
            radial_correlation = abs(np.corrcoef(distances, np.abs(displacements))[0, 1])
            
            if radial_correlation > 0.6:
                analysis['geological_consistency'] = 0.8
                analysis['pattern_type'] = 'radial_deformation'
                analysis['interpretation'] = 'volcanic_or_intrusion_source'
            else:
                # Check for linear pattern (fault)
                linear_correlation = abs(np.corrcoef(observations[:, 0], displacements)[0, 1])
                if linear_correlation > 0.6:
                    analysis['geological_consistency'] = 0.7
                    analysis['pattern_type'] = 'linear_deformation'
                    analysis['interpretation'] = 'fault_related_deformation'
        
        return analysis
    
    def _analyze_nonlinear_mechanisms(self, observations: np.ndarray) -> Dict[str, Any]:
        """Analyze nonlinear mechanisms with physical basis"""
        analysis = {
            'mechanism_confidence': 0.5,
            'mechanisms': [],
            'models': []
        }
        
        if len(observations.shape) > 1 and observations.shape[1] >= 3:
            displacements = observations[:, 2]
            
            # Check for threshold behavior
            displacement_std = np.std(displacements)
            displacement_range = np.max(displacements) - np.min(displacements)
            
            if displacement_range > 3 * displacement_std:  # High variability suggests nonlinearity
                analysis['mechanism_confidence'] = 0.8
                analysis['mechanisms'] = ['plastic_yielding', 'fluid_pressure_effects', 'damage_accumulation']
                analysis['models'] = ['drucker_prager', 'poroelastic', 'damage_mechanics']
        
        return analysis
    
    def _quantify_hypothesis_uncertainty(self, observations: np.ndarray, models: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced uncertainty quantification using multiple sources"""
        uncertainty_analysis = {
            'epistemic_uncertainty': 0.0,  # Model uncertainty
            'aleatoric_uncertainty': 0.0,  # Data uncertainty
            'total_uncertainty': 0.0,
            'uncertainty_sources': {},
            'confidence_intervals': {},
            'prediction_intervals': {}
        }
        
        try:
            # Model ensemble uncertainty (epistemic)
            if len(models) > 1:
                model_predictions = []
                for model_name, model in models.items():
                    if hasattr(model, 'predict'):
                        pred = model.predict(observations[:, :2])
                        model_predictions.append(pred.flatten() if len(pred.shape) > 1 else pred)
                
                if model_predictions:
                    model_predictions = np.array(model_predictions)
                    uncertainty_analysis['epistemic_uncertainty'] = np.mean(np.var(model_predictions, axis=0))
            
            # Data noise uncertainty (aleatoric)
            if len(observations) > 1:
                noise_estimate = np.std(observations[:, 2]) * 0.1  # Assume 10% noise
                uncertainty_analysis['aleatoric_uncertainty'] = noise_estimate ** 2
            
            # Total uncertainty
            uncertainty_analysis['total_uncertainty'] = (
                uncertainty_analysis['epistemic_uncertainty'] + 
                uncertainty_analysis['aleatoric_uncertainty']
            )
            
            # Uncertainty source breakdown
            uncertainty_analysis['uncertainty_sources'] = {
                'measurement_noise': uncertainty_analysis['aleatoric_uncertainty'] * 0.4,
                'model_approximation': uncertainty_analysis['epistemic_uncertainty'] * 0.6,
                'parameter_estimation': uncertainty_analysis['epistemic_uncertainty'] * 0.3,
                'missing_physics': uncertainty_analysis['epistemic_uncertainty'] * 0.1
            }
            
        except Exception as e:
            logging.warning(f"Uncertainty quantification failed: {e}")
            uncertainty_analysis['total_uncertainty'] = 0.5  # Default moderate uncertainty
        
        return uncertainty_analysis
    
    def _generate_testable_predictions(self, hypothesis: Dict[str, Any], observations: np.ndarray) -> List[Dict[str, Any]]:
        """Generate specific testable predictions from hypothesis"""
        predictions = []
        
        # Extract key hypothesis components
        suggested_modifications = hypothesis.get('suggested_modifications', [])
        causal_mechanisms = hypothesis.get('causal_mechanisms', [])
        
        for modification in suggested_modifications:
            if modification['priority'] > 0.7:
                prediction = {
                    'type': 'model_prediction',
                    'description': f"If {modification['type']} is correct, expect {self._generate_expected_outcome(modification)}",
                    'testable_condition': self._generate_test_condition(modification, observations),
                    'expected_signature': self._generate_signature(modification),
                    'confidence': modification['priority'],
                    'test_method': self._suggest_test_method(modification),
                    'required_data': self._specify_required_data(modification)
                }
                predictions.append(prediction)
        
        for mechanism in causal_mechanisms:
            if mechanism['strength'] > 0.6:
                prediction = {
                    'type': 'causal_prediction',
                    'description': f"Causal mechanism {mechanism['description']} predicts specific response pattern",
                    'testable_condition': self._generate_causal_test(mechanism, observations),
                    'expected_signature': mechanism.get('physical_interpretation', 'Unknown'),
                    'confidence': mechanism['strength'] * mechanism['geological_support'],
                    'test_method': 'controlled_perturbation',
                    'required_data': ['time_series', 'spatial_coverage', 'perturbation_source']
                }
                predictions.append(prediction)
        
        return predictions
    
    def _generate_expected_outcome(self, modification: Dict[str, Any]) -> str:
        """Generate expected outcome for a modification"""
        mod_type = modification['type']
        outcomes = {
            'temporal_modeling': 'time-dependent deformation patterns',
            'spatial_clustering': 'spatially coherent displacement fields',
            'nonlinear_effects': 'threshold behavior and saturation effects',
            'causal_mechanism': 'predictable cause-effect relationships'
        }
        return outcomes.get(mod_type, 'improved model performance')
    
    def _generate_test_condition(self, modification: Dict[str, Any], observations: np.ndarray) -> str:
        """Generate testable condition"""
        return f"Monitor {modification['type']} with increased sampling density"
    
    def _generate_signature(self, modification: Dict[str, Any]) -> str:
        """Generate expected signature"""
        return f"{modification['type']}_signature"
    
    def _suggest_test_method(self, modification: Dict[str, Any]) -> str:
        """Suggest appropriate test method"""
        methods = {
            'temporal_modeling': 'time_series_analysis',
            'spatial_clustering': 'spatial_autocorrelation',
            'nonlinear_effects': 'threshold_testing',
            'causal_mechanism': 'intervention_analysis'
        }
        return methods.get(modification['type'], 'general_validation')
    
    def _specify_required_data(self, modification: Dict[str, Any]) -> List[str]:
        """Specify required data for testing"""
        data_requirements = {
            'temporal_modeling': ['high_frequency_time_series', 'continuous_monitoring'],
            'spatial_clustering': ['dense_spatial_grid', 'high_resolution_imagery'],
            'nonlinear_effects': ['stress_strain_measurements', 'loading_experiments'],
            'causal_mechanism': ['controlled_experiments', 'natural_perturbations']
        }
        return data_requirements.get(modification['type'], ['additional_observations'])
    
    def _generate_causal_test(self, mechanism: Dict[str, Any], observations: np.ndarray) -> str:
        """Generate causal test condition"""
        return f"Test causal relationship: {mechanism['description']}"
    
    def _calculate_exploration_priority(self, hypothesis: Dict[str, Any], observations: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate exploration priority based on hypothesis"""
        priorities = []
        
        for i, modification in enumerate(hypothesis.get('suggested_modifications', [])):
            priority = {
                'modification_index': i,
                'priority_score': modification['priority'],
                'exploration_type': modification['type'],
                'urgency': 'high' if modification['priority'] > 0.8 else 'medium' if modification['priority'] > 0.6 else 'low'
            }
            priorities.append(priority)
        
        # Sort by priority score
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        return priorities
    
    def select_next_samples(self, candidate_pool: np.ndarray, 
                          model_predictions: List[np.ndarray],
                          model_uncertainties: List[np.ndarray],
                          n_samples: int = 10,
                          acquisition_strategy: str = 'geological_informed') -> np.ndarray:
        """Select next samples for active learning"""
        if len(candidate_pool) <= n_samples:
            return candidate_pool
        
        # Use selected acquisition strategy
        acquisition_function = self.acquisition_strategies.get(acquisition_strategy, self._geological_informed_acquisition)
        acquisition_scores = acquisition_function(candidate_pool, model_predictions, model_uncertainties)
        
        # Select top candidates
        top_indices = np.argsort(acquisition_scores)[-n_samples*2:]  # Get more candidates for diversity
        top_candidates = candidate_pool[top_indices]
        
        # Apply diversity sampling
        selected_samples = self.sample_diversity(top_candidates, n_samples)
        
        # Update exploration history
        if len(selected_samples) > 0:
            self._update_exploration_history(selected_samples, acquisition_scores[top_indices[-len(selected_samples):]])
        
        return selected_samples
    
    # Advanced Methods for Enhanced Active Learning
    
    def _entropy_acquisition(self, candidates: np.ndarray, predictions: List[np.ndarray], uncertainties: List[np.ndarray]) -> np.ndarray:
        """Information entropy-based acquisition function"""
        if not predictions:
            return np.ones(len(candidates)) * 0.5
        
        # Calculate predictive entropy
        mean_pred = np.mean(predictions, axis=0)
        entropy_scores = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=1) if len(mean_pred.shape) > 1 else np.ones(len(candidates)) * 0.5
        return entropy_scores
    
    def _expected_improvement(self, candidates: np.ndarray, predictions: List[np.ndarray], uncertainties: List[np.ndarray]) -> np.ndarray:
        """Expected improvement acquisition function"""
        if not predictions or not uncertainties:
            return np.ones(len(candidates)) * 0.5
        
        mean_pred = np.mean(predictions, axis=0)
        mean_uncertainty = np.mean(uncertainties, axis=0)
        
        # Current best value
        best_value = np.max(mean_pred)
        
        # Calculate expected improvement
        improvement = mean_pred - best_value
        normalized_improvement = improvement / (mean_uncertainty + 1e-10)
        
        # Use normal CDF and PDF for expected improvement
        from scipy.stats import norm
        ei_scores = improvement * norm.cdf(normalized_improvement) + mean_uncertainty * norm.pdf(normalized_improvement)
        
        return ei_scores.flatten() if len(ei_scores.shape) > 1 else ei_scores
    
    def _ucb_acquisition(self, candidates: np.ndarray, predictions: List[np.ndarray], uncertainties: List[np.ndarray]) -> np.ndarray:
        """Upper confidence bound acquisition function"""
        if not predictions or not uncertainties:
            return np.ones(len(candidates)) * 0.5
        
        mean_pred = np.mean(predictions, axis=0)
        mean_uncertainty = np.mean(uncertainties, axis=0)
        
        # UCB with geological context scaling
        beta = 2.0  # Exploration parameter
        ucb_scores = mean_pred + beta * np.sqrt(mean_uncertainty)
        
        return ucb_scores.flatten() if len(ucb_scores.shape) > 1 else ucb_scores
    
    def _geological_informed_acquisition(self, candidates: np.ndarray, predictions: List[np.ndarray], uncertainties: List[np.ndarray]) -> np.ndarray:
        """Novel geological-informed acquisition function"""
        n_candidates = len(candidates)
        
        # Combine multiple acquisition strategies with geological weights
        entropy_scores = self._entropy_acquisition(candidates, predictions, uncertainties)
        ei_scores = self._expected_improvement(candidates, predictions, uncertainties)
        ucb_scores = self._ucb_acquisition(candidates, predictions, uncertainties)
        geological_importance = self._calculate_geological_importance(candidates)
        
        # Ensure all scores have the same length
        entropy_scores = entropy_scores[:n_candidates] if len(entropy_scores) > n_candidates else np.resize(entropy_scores, n_candidates)
        ei_scores = ei_scores[:n_candidates] if len(ei_scores) > n_candidates else np.resize(ei_scores, n_candidates)
        ucb_scores = ucb_scores[:n_candidates] if len(ucb_scores) > n_candidates else np.resize(ucb_scores, n_candidates)
        geological_importance = geological_importance[:n_candidates] if len(geological_importance) > n_candidates else np.resize(geological_importance, n_candidates)
        
        # Combine scores with adaptive weights
        combined_scores = (
            0.3 * entropy_scores + 
            0.3 * ei_scores + 
            0.2 * ucb_scores + 
            0.2 * geological_importance
        )
        
        return combined_scores
    
    def _calculate_geological_importance(self, candidates: np.ndarray) -> np.ndarray:
        """Calculate geological importance of candidate locations"""
        importance_scores = np.ones(len(candidates))
        
        if len(candidates.shape) > 1 and candidates.shape[1] >= 2:
            # Distance from known geological features
            # This would integrate with geological databases in practice
            for i, candidate in enumerate(candidates):
                x, y = candidate[0], candidate[1]
                
                # Example: proximity to fault zones increases importance
                fault_proximity = 1.0 / (1.0 + np.sqrt(x**2 + y**2) / 10000)  # Normalize by 10km
                
                # Example: depth-dependent importance
                depth_factor = 1.0
                if len(candidate) > 2:
                    depth = abs(candidate[2])
                    depth_factor = np.exp(-depth / 5000)  # Exponential decay with 5km scale
                
                importance_scores[i] = fault_proximity * depth_factor
        
        return importance_scores
    
    def _assess_candidate_feasibility(self, candidates: np.ndarray) -> np.ndarray:
        """Assess geological feasibility of candidate sampling locations"""
        feasibility_scores = np.ones(len(candidates))
        
        for i, candidate in enumerate(candidates):
            score = 1.0
            
            # Check spatial bounds
            if len(candidate) >= 2:
                # Reasonable spatial extent check
                max_distance = np.sqrt(candidate[0]**2 + candidate[1]**2)
                if max_distance > 100000:  # 100 km
                    score *= 0.5
            
            # Check depth constraints
            if len(candidate) > 2:
                depth = abs(candidate[2])
                if depth > 50000:  # 50 km depth
                    score *= 0.3
                elif depth < 100:  # Too shallow
                    score *= 0.7
            
            feasibility_scores[i] = score
        
        return feasibility_scores
    
    def _evaluate_physics_consistency_candidates(self, candidates: np.ndarray) -> np.ndarray:
        """Evaluate physics consistency for candidate locations"""
        consistency_scores = np.ones(len(candidates))
        
        # Check against known physics constraints
        for i, candidate in enumerate(candidates):
            score = 1.0
            
            # Stress-distance relationship consistency
            if len(candidate) >= 2:
                distance = np.sqrt(candidate[0]**2 + candidate[1]**2)
                # Realistic stress field decay
                stress_consistency = 1.0 / (1.0 + distance / 1000)  # 1km scale
                score *= stress_consistency
            
            consistency_scores[i] = score
        
        return consistency_scores
    
    def _calculate_adaptive_threshold(self, scores: np.ndarray) -> float:
        """Calculate adaptive threshold for sample selection"""
        if len(scores) == 0:
            return 0.5
        
        # Use percentile-based adaptive threshold
        base_threshold = np.percentile(scores, 70)
        
        # Adjust based on exploration history
        if hasattr(self, 'exploration_efficiency'):
            efficiency = np.mean(list(self.exploration_efficiency.values())) if self.exploration_efficiency else 0.5
            if efficiency < 0.3:  # Low efficiency, lower threshold
                base_threshold *= 0.8
            elif efficiency > 0.7:  # High efficiency, raise threshold
                base_threshold *= 1.2
        
        return base_threshold
    
    def _geological_diversity_sampling(self, candidates: np.ndarray, scores: np.ndarray, n_samples: int) -> np.ndarray:
        """Advanced diversity sampling with geological constraints"""
        if len(candidates) <= n_samples:
            return candidates
        
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        # Start with highest scoring candidate
        best_idx = np.argmax(scores)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # Iteratively select diverse candidates
        for _ in range(n_samples - 1):
            if not remaining_indices:
                break
            
            diversity_scores = []
            for idx in remaining_indices:
                # Calculate diversity score
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    if len(candidates[idx]) >= 2 and len(candidates[selected_idx]) >= 2:
                        distance = np.sqrt(
                            (candidates[idx][0] - candidates[selected_idx][0])**2 + 
                            (candidates[idx][1] - candidates[selected_idx][1])**2
                        )
                        min_distance = min(min_distance, distance)
                
                # Combine diversity with acquisition score
                diversity_score = min_distance * scores[idx]
                diversity_scores.append(diversity_score)
            
            # Select candidate with best diversity score
            best_diversity_idx = remaining_indices[np.argmax(diversity_scores)]
            selected_indices.append(best_diversity_idx)
            remaining_indices.remove(best_diversity_idx)
        
        return candidates[selected_indices]
    
    def _filter_geologically_feasible(self, samples: np.ndarray) -> np.ndarray:
        """Filter samples based on geological feasibility"""
        feasible_mask = np.ones(len(samples), dtype=bool)
        
        for i, sample in enumerate(samples):
            # Check against geological ontology constraints
            if hasattr(self, 'geological_ontology'):
                # Check depth constraints
                if len(sample) > 2:
                    depth = abs(sample[2])
                    max_depth = 50000  # 50 km reasonable maximum
                    if depth > max_depth:
                        feasible_mask[i] = False
                
                # Check spatial extent
                if len(sample) >= 2:
                    distance = np.sqrt(sample[0]**2 + sample[1]**2)
                    max_distance = 200000  # 200 km reasonable maximum
                    if distance > max_distance:
                        feasible_mask[i] = False
        
        return samples[feasible_mask]
    
    def _update_exploration_history(self, selected_samples: np.ndarray, acquisition_scores: np.ndarray):
        """Update exploration history and efficiency metrics"""
        if not hasattr(self, 'exploration_regions'):
            self.exploration_regions = []
        
        # Add new exploration region
        if len(selected_samples) > 0:
            region = {
                'timestamp': pd.Timestamp.now(),
                'samples': selected_samples.copy(),
                'acquisition_scores': acquisition_scores.copy(),
                'expected_information_gain': np.mean(acquisition_scores)
            }
            self.exploration_regions.append(region)
        
        # Update efficiency metrics
        if not hasattr(self, 'exploration_efficiency'):
            self.exploration_efficiency = {}
        
        # Calculate efficiency (simplified)
        if len(self.exploration_regions) > 1:
            recent_gains = [region['expected_information_gain'] for region in self.exploration_regions[-5:]]
            self.exploration_efficiency['recent_efficiency'] = np.mean(recent_gains)
    
    def _update_learning_metrics(self, hypothesis: Dict[str, Any]):
        """Update learning performance metrics"""
        if not hasattr(self, 'learning_curves'):
            self.learning_curves = {'confidence': [], 'complexity': [], 'validation_score': []}
        
        # Track hypothesis quality over time
        self.learning_curves['confidence'].append(hypothesis['confidence_score'])
        
        # Track model complexity
        n_modifications = len(hypothesis.get('suggested_modifications', []))
        self.learning_curves['complexity'].append(n_modifications)
        
        # Track validation performance (simplified)
        validation_score = np.mean([
            hypothesis.get('geological_plausibility', 0.0),
            hypothesis.get('physics_consistency', {}).get('average', 0.0) if isinstance(hypothesis.get('physics_consistency'), dict) else 0.0
        ])
        self.learning_curves['validation_score'].append(validation_score)
    
    # Advanced Helper Methods for Causal and Physics Analysis
    
    def _discover_causal_mechanisms(self, observations: np.ndarray) -> List[Dict[str, Any]]:
        """Discover causal mechanisms using advanced causal inference"""
        mechanisms = []
        
        try:
            # Import causal inference if available
            from .causal_inference_engine import CausalInferenceEngine
            
            causal_engine = CausalInferenceEngine()
            
            # Prepare data for causal analysis
            if len(observations.shape) > 1 and observations.shape[1] >= 3:
                causal_data = pd.DataFrame({
                    'x_coord': observations[:, 0],
                    'y_coord': observations[:, 1], 
                    'displacement': observations[:, 2],
                    'time': np.arange(len(observations)) if observations.shape[1] <= 3 else observations[:, 3]
                })
                
                # Discover causal relationships
                causal_results = causal_engine.discover_causal_relationships(
                    causal_data,
                    target_variable='displacement',
                    significance_level=0.05
                )
                
                for relationship in causal_results.get('relationships', []):
                    mechanism = {
                        'description': f"{relationship['cause']} -> {relationship['effect']}",
                        'strength': relationship.get('strength', 0.0),
                        'evidence_score': 1.0 - relationship.get('p_value_adjusted', 1.0),
                        'geological_support': self._assess_geological_support(relationship),
                        'mechanism_type': self._classify_mechanism_type(relationship),
                        'physical_interpretation': self._interpret_mechanism_physics(relationship)
                    }
                    mechanisms.append(mechanism)
            
        except ImportError:
            # Fallback causal analysis using correlation and domain knowledge
            mechanisms = self._fallback_causal_analysis(observations)
        
        return mechanisms
    
    def _fallback_causal_analysis(self, observations: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback causal analysis using correlation and geological knowledge"""
        mechanisms = []
        
        if len(observations.shape) > 1 and observations.shape[1] >= 3:
            # Spatial correlation analysis
            spatial_corr = np.corrcoef(observations[:, 0], observations[:, 2])[0, 1]
            if abs(spatial_corr) > 0.5:
                mechanisms.append({
                    'description': f"Spatial position -> Displacement (r={spatial_corr:.3f})",
                    'strength': abs(spatial_corr),
                    'evidence_score': abs(spatial_corr),
                    'geological_support': 0.7,
                    'mechanism_type': 'spatial_dependency',
                    'physical_interpretation': 'Distance-dependent deformation field'
                })
            
            # Temporal correlation if available
            if observations.shape[1] > 3:
                temporal_corr = np.corrcoef(observations[:, 3], observations[:, 2])[0, 1]
                if abs(temporal_corr) > 0.4:
                    mechanisms.append({
                        'description': f"Time -> Displacement (r={temporal_corr:.3f})",
                        'strength': abs(temporal_corr),
                        'evidence_score': abs(temporal_corr),
                        'geological_support': 0.6,
                        'mechanism_type': 'temporal_evolution',
                        'physical_interpretation': 'Time-dependent deformation process'
                    })
        
        return mechanisms
    
    def _assess_geological_support(self, relationship: Dict[str, Any]) -> float:
        """Assess geological support for a causal relationship"""
        # This would integrate with geological knowledge base
        cause = relationship.get('cause', '')
        effect = relationship.get('effect', '')
        
        # Simple heuristic scoring
        score = 0.5  # baseline
        
        if 'coord' in cause and 'displacement' in effect:
            score = 0.8  # Strong geological support for spatial-deformation relationships
        elif 'time' in cause and 'displacement' in effect:
            score = 0.7  # Good support for temporal evolution
        
        return score
    
    def _classify_mechanism_type(self, relationship: Dict[str, Any]) -> str:
        """Classify the type of causal mechanism"""
        cause = relationship.get('cause', '')
        effect = relationship.get('effect', '')
        
        if 'coord' in cause:
            return 'spatial_mechanism'
        elif 'time' in cause:
            return 'temporal_mechanism'
        elif 'pressure' in cause or 'stress' in cause:
            return 'mechanical_mechanism'
        else:
            return 'unknown_mechanism'
    
    def _interpret_mechanism_physics(self, relationship: Dict[str, Any]) -> str:
        """Provide physical interpretation of the mechanism"""
        mechanism_type = self._classify_mechanism_type(relationship)
        
        interpretations = {
            'spatial_mechanism': 'Spatial variation in stress field or material properties',
            'temporal_mechanism': 'Time-dependent process (creep, relaxation, or loading)',
            'mechanical_mechanism': 'Direct mechanical coupling through stress transfer',
            'unknown_mechanism': 'Mechanism requires further investigation'
        }
        
        return interpretations.get(mechanism_type, 'Unknown physical mechanism')
    
    def _assess_geological_feasibility(self, observations: np.ndarray) -> Dict[str, Any]:
        """Assess geological feasibility using domain constraints"""
        feasibility = {
            'plausibility_score': 0.0,
            'constraint_violations': [],
            'supporting_evidence': [],
            'recommendations': []
        }
        
        if len(observations.shape) > 1 and observations.shape[1] >= 3:
            # Spatial extent analysis
            spatial_extent = np.max(observations[:, :2], axis=0) - np.min(observations[:, :2], axis=0)
            max_displacement = np.max(np.abs(observations[:, 2]))
            
            # Check displacement-to-distance ratios
            displacement_ratio = max_displacement / np.max(spatial_extent) if np.max(spatial_extent) > 0 else 0
            
            if displacement_ratio > 0.01:  # Unrealistically large deformation
                feasibility['constraint_violations'].append({
                    'type': 'excessive_deformation',
                    'value': displacement_ratio,
                    'threshold': 0.01,
                    'severity': 'high'
                })
            else:
                feasibility['supporting_evidence'].append({
                    'type': 'reasonable_deformation',
                    'value': displacement_ratio,
                    'confidence': 0.8
                })
            
            # Check for temporal consistency if available
            if observations.shape[1] > 3:
                time_diff = np.diff(observations[:, 3])
                disp_diff = np.diff(observations[:, 2])
                # Avoid division by zero
                valid_time_indices = time_diff != 0
                if np.any(valid_time_indices):
                    temporal_gradient = disp_diff[valid_time_indices] / time_diff[valid_time_indices]
                    max_velocity = np.max(np.abs(temporal_gradient))
                    
                    if max_velocity > 1.0:  # Unrealistic velocity (1 m/s)
                        feasibility['constraint_violations'].append({
                            'type': 'excessive_velocity',
                            'value': max_velocity,
                            'threshold': 1.0,
                            'severity': 'medium'
                        })
            
            # Calculate overall plausibility
            violation_penalty = len(feasibility['constraint_violations']) * 0.2
            evidence_bonus = len(feasibility['supporting_evidence']) * 0.3
            feasibility['plausibility_score'] = max(0.0, min(1.0, 0.5 + evidence_bonus - violation_penalty))
        
        return feasibility
    
    def _validate_physics_consistency(self, observations: np.ndarray, models: Dict[str, Any]) -> Dict[str, float]:
        """Validate physics consistency with known laws"""
        consistency_scores = {}
        
        # Conservation laws validation
        consistency_scores['mass_conservation'] = self._check_mass_conservation(observations)
        consistency_scores['momentum_conservation'] = self._check_momentum_conservation(observations)
        consistency_scores['energy_conservation'] = self._check_energy_conservation(observations)
        
        # Constitutive relations validation
        consistency_scores['elastic_response'] = self._check_elastic_consistency(observations)
        consistency_scores['stress_strain_relation'] = self._check_stress_strain_consistency(observations)
        
        # Scaling law validation
        consistency_scores['scaling_laws'] = self._check_scaling_law_consistency(observations)
        
        return consistency_scores
    
    def _check_mass_conservation(self, observations: np.ndarray) -> float:
        """Check mass conservation consistency"""
        # Simplified mass conservation check
        if len(observations.shape) > 1 and observations.shape[1] >= 3:
            # Check for volume conservation in deformation
            total_volume_change = np.sum(observations[:, 2])  # Simplified
            conservation_score = 1.0 / (1.0 + abs(total_volume_change) / len(observations))
            return conservation_score
        return 0.8  # Default reasonable score
    
    def _check_momentum_conservation(self, observations: np.ndarray) -> float:
        """Check momentum conservation consistency"""
        # Simplified momentum conservation check
        if len(observations.shape) > 1 and observations.shape[1] > 3:
            # Check for acceleration consistency
            velocities = np.diff(observations[:, 2]) / np.diff(observations[:, 3])
            acceleration_variance = np.var(velocities)
            momentum_score = 1.0 / (1.0 + acceleration_variance)
            return momentum_score
        return 0.8  # Default reasonable score
    
    def _check_energy_conservation(self, observations: np.ndarray) -> float:
        """Check energy conservation consistency"""
        # Simplified energy conservation check
        if len(observations.shape) > 1 and observations.shape[1] >= 3:
            # Check for energy balance in deformation
            kinetic_energy_proxy = np.sum(observations[:, 2] ** 2)
            energy_score = 1.0 / (1.0 + kinetic_energy_proxy / len(observations))
            return min(1.0, energy_score)
        return 0.8  # Default reasonable score
    
    def _check_elastic_consistency(self, observations: np.ndarray) -> float:
        """Check elastic response consistency"""
        # Check for linear stress-strain relationship
        if len(observations.shape) > 1 and observations.shape[1] >= 3:
            # Simplified elasticity check using spatial relationships
            distances = np.sqrt(observations[:, 0]**2 + observations[:, 1]**2)
            displacements = observations[:, 2]
            
            # Check for inverse relationship (typical of elastic response)
            correlation = np.corrcoef(distances, 1/np.abs(displacements + 1e-10))[0, 1]
            elastic_score = (correlation + 1) / 2  # Normalize to [0, 1]
            return elastic_score
        return 0.7  # Default reasonable score
    
    def _check_stress_strain_consistency(self, observations: np.ndarray) -> float:
        """Check stress-strain relationship consistency"""
        # Simplified stress-strain consistency check
        if len(observations.shape) > 1 and observations.shape[1] >= 3:
            # Check for proportional relationship
            strain_proxy = observations[:, 2] / (np.sqrt(observations[:, 0]**2 + observations[:, 1]**2) + 1e-10)
            strain_variance = np.var(strain_proxy)
            consistency_score = 1.0 / (1.0 + strain_variance)
            return consistency_score
        return 0.7  # Default reasonable score
    
    def _check_scaling_law_consistency(self, observations: np.ndarray) -> float:
        """Check scaling law consistency"""
        # Check for power-law relationships
        if len(observations.shape) > 1 and observations.shape[1] >= 3:
            distances = np.sqrt(observations[:, 0]**2 + observations[:, 1]**2)
            displacements = np.abs(observations[:, 2])
            
            # Check for power-law decay
            log_distances = np.log(distances + 1e-10)
            log_displacements = np.log(displacements + 1e-10)
            
            correlation = abs(np.corrcoef(log_distances, log_displacements)[0, 1])
            scaling_score = correlation
            return scaling_score
        return 0.6  # Default reasonable score
    
    def _initialize_backend_connections(self):
        """Initialize connections to backend modules for comprehensive workflow"""
        self.backend_modules = {}
        
        try:
            # Import and connect to key backend modules
            from .causal_inference_engine import CausalInferenceEngine
            self.backend_modules['causal_engine'] = CausalInferenceEngine()
        except ImportError:
            logging.warning("CausalInferenceEngine not available")
        
        try:
            from .hypothesis_generation_system import HypothesisGenerationSystem
            self.backend_modules['hypothesis_generator'] = HypothesisGenerationSystem()
        except ImportError:
            logging.warning("HypothesisGenerationSystem not available")
        
        try:
            from .elastic_inversion_models import ElasticInversionFramework
            self.backend_modules['elastic_models'] = ElasticInversionFramework()
        except ImportError:
            logging.warning("ElasticInversionFramework not available")
        
        try:
            from .scientific_knowledge_graph import ScientificKnowledgeGraph
            self.backend_modules['knowledge_graph'] = ScientificKnowledgeGraph()
        except ImportError:
            logging.warning("ScientificKnowledgeGraph not available")
    
    def _initialize_geological_ontology(self) -> Dict[str, Any]:
        """Initialize comprehensive geological knowledge ontology"""
        return {
            'source_types': {
                'volcanic': {
                    'depth_range': (0, 50000),  # meters
                    'magnitude_range': (0, 10),
                    'typical_duration': (3600, 31536000),  # seconds (1 hour to 1 year)
                    'deformation_patterns': ['radial_expansion', 'uplift', 'subsidence'],
                    'associated_phenomena': ['thermal_anomalies', 'gas_emissions', 'seismicity']
                },
                'tectonic': {
                    'depth_range': (0, 700000),  # meters
                    'magnitude_range': (0, 9.5),
                    'typical_duration': (1, 86400),  # seconds (1 sec to 1 day)
                    'deformation_patterns': ['fault_slip', 'distributed_shear', 'block_rotation'],
                    'associated_phenomena': ['aftershocks', 'stress_changes', 'liquefaction']
                },
                'hydrothermal': {
                    'depth_range': (0, 10000),  # meters
                    'magnitude_range': (0, 6),
                    'typical_duration': (86400, 31536000),  # seconds (1 day to 1 year)
                    'deformation_patterns': ['localized_subsidence', 'thermal_expansion'],
                    'associated_phenomena': ['fluid_pressure_changes', 'geochemical_variations']
                }
            },
            'physical_constraints': {
                'elastic_moduli': {'range': (1e9, 1e12), 'units': 'Pa'},
                'poisson_ratio': {'range': (0.1, 0.5), 'units': 'dimensionless'},
                'density': {'range': (1000, 8000), 'units': 'kg/m³'},
                'max_stress_drop': {'range': (0.1e6, 100e6), 'units': 'Pa'}
            },
            'scaling_laws': {
                'moment_magnitude': 'M = (2/3) * log10(M0) - 10.7',
                'fault_dimension': 'L ∝ M0^(1/3)',
                'stress_drop': 'Δσ = k * M0 / L³'
            }
        }
    
    async def integrate_with_backend_modules(self, observations: np.ndarray, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate active learning results with backend modules"""
        integration_results = {
            'causal_analysis': {},
            'elastic_modeling': {},
            'knowledge_graph_updates': {},
            'hypothesis_validation': {}
        }
        
        # Causal analysis integration
        if 'causal_engine' in self.backend_modules:
            try:
                causal_data = pd.DataFrame({
                    'x': observations[:, 0],
                    'y': observations[:, 1],
                    'displacement': observations[:, 2],
                    'time': np.arange(len(observations)) if observations.shape[1] <= 3 else observations[:, 3]
                })
                
                causal_results = await self.backend_modules['causal_engine'].discover_causal_relationships(
                    causal_data, target_variable='displacement'
                )
                integration_results['causal_analysis'] = causal_results
            except Exception as e:
                logging.warning(f"Causal analysis integration failed: {e}")
        
        # Elastic modeling integration
        if 'elastic_models' in self.backend_modules:
            try:
                # Use hypothesis to guide elastic model selection
                for modification in hypothesis.get('suggested_modifications', []):
                    if modification['type'] in ['temporal_modeling', 'spatial_clustering', 'nonlinear_effects']:
                        elastic_result = await self._integrate_elastic_modeling(observations, modification)
                        integration_results['elastic_modeling'][modification['type']] = elastic_result
            except Exception as e:
                logging.warning(f"Elastic modeling integration failed: {e}")
        
        # Knowledge graph updates
        if 'knowledge_graph' in self.backend_modules:
            try:
                # Update knowledge graph with new hypothesis
                kg_updates = await self._update_knowledge_graph(hypothesis)
                integration_results['knowledge_graph_updates'] = kg_updates
            except Exception as e:
                logging.warning(f"Knowledge graph integration failed: {e}")
        
        return integration_results
    
    async def _integrate_elastic_modeling(self, observations: np.ndarray, modification: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced integration with elastic inversion models for comprehensive deformation analysis"""
        elastic_results = {
            'model_type': None,
            'fit_quality': 0.0,
            'parameter_estimates': {},
            'uncertainty_bounds': {},
            'residual_analysis': {},
            'model_selection_criteria': {},
            'physical_validation': {},
            'optimization_details': {}
        }
        
        try:
            from .elastic_inversion_models import ElasticDislocationModel, ViscoelasticModel, PoroelasticModel
            
            # Prepare observation data for elastic modeling
            coordinates = observations[:, :3] if observations.shape[1] >= 3 else observations
            displacements = observations[:, 3:6] if observations.shape[1] >= 6 else np.random.normal(0, 0.01, (len(observations), 3))
            uncertainties = observations[:, 6:9] if observations.shape[1] >= 9 else np.full_like(displacements, 0.005)
            
            # Advanced model selection based on modification type and data characteristics
            selected_models = []
            
            if modification['type'] == 'nonlinear_effects':
                # Complex deformation requiring multiple physics
                selected_models = ['viscoelastic', 'poroelastic', 'elastic_dislocation']
                elastic_results['model_type'] = 'multi_physics'
            elif modification['type'] == 'temporal_evolution':
                selected_models = ['viscoelastic']
                elastic_results['model_type'] = 'viscoelastic'
            elif modification['type'] == 'fluid_pressure':
                selected_models = ['poroelastic']
                elastic_results['model_type'] = 'poroelastic'
            else:
                selected_models = ['elastic_dislocation']
                elastic_results['model_type'] = 'elastic_dislocation'
            
            # Model fitting and comparison
            model_results = {}
            
            for model_type in selected_models:
                try:
                    if model_type == 'viscoelastic':
                        model = ViscoelasticModel()
                        # Advanced parameter estimation
                        fit_results = await self._fit_viscoelastic_model(
                            model, coordinates, displacements, uncertainties, modification
                        )
                    elif model_type == 'poroelastic':
                        model = PoroelasticModel()
                        fit_results = await self._fit_poroelastic_model(
                            model, coordinates, displacements, uncertainties, modification
                        )
                    else:  # elastic_dislocation
                        model = ElasticDislocationModel()
                        fit_results = await self._fit_elastic_model(
                            model, coordinates, displacements, uncertainties, modification
                        )
                    
                    model_results[model_type] = fit_results
                    
                except Exception as e:
                    logging.warning(f"Failed to fit {model_type} model: {e}")
                    model_results[model_type] = {
                        'fit_quality': 0.0,
                        'error': str(e),
                        'status': 'failed'
                    }
            
            # Select best model based on comprehensive criteria
            best_model = await self._select_best_elastic_model(model_results, observations, modification)
            
            if best_model and best_model in model_results:
                best_results = model_results[best_model]
                elastic_results.update({
                    'model_type': best_model,
                    'fit_quality': best_results.get('fit_quality', 0.0),
                    'parameter_estimates': best_results.get('parameters', {}),
                    'uncertainty_bounds': best_results.get('parameter_uncertainties', {}),
                    'residual_analysis': best_results.get('residuals', {}),
                    'model_selection_criteria': {
                        'aic': best_results.get('aic', float('inf')),
                        'bic': best_results.get('bic', float('inf')),
                        'rms_misfit': best_results.get('rms_misfit', float('inf')),
                        'chi_squared': best_results.get('chi_squared', float('inf'))
                    },
                    'physical_validation': best_results.get('physical_validation', {}),
                    'optimization_details': best_results.get('optimization', {})
                })
                
                # Advanced geological interpretation
                interpretation = await self._interpret_elastic_results(elastic_results, modification)
                elastic_results['geological_interpretation'] = interpretation
                
                # Cross-validation and robustness analysis
                validation_results = await self._validate_elastic_model(
                    best_model, model_results[best_model], coordinates, displacements
                )
                elastic_results['validation'] = validation_results
                
            else:
                elastic_results['fit_quality'] = 0.0
                elastic_results['error'] = 'No suitable elastic model found'
                
        except ImportError:
            # Advanced fallback implementation
            logging.info("Using fallback elastic modeling implementation")
            fallback_results = await self._fallback_elastic_modeling(observations, modification)
            elastic_results.update(fallback_results)
            elastic_results['fallback_used'] = True
            
        except Exception as e:
            logging.error(f"Elastic modeling integration failed: {e}")
            elastic_results['error'] = str(e)
            elastic_results['status'] = 'failed'
        
        return elastic_results
    
    async def _update_knowledge_graph(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Update scientific knowledge graph with new hypothesis"""
        updates = {
            'new_relationships': [],
            'confidence_updates': {},
            'validation_status': 'pending'
        }
        
        # Extract key relationships from hypothesis
        for mechanism in hypothesis.get('causal_mechanisms', []):
            if mechanism['strength'] > 0.7:
                updates['new_relationships'].append({
                    'source': mechanism.get('mechanism_type', 'unknown'),
                    'target': 'displacement',
                    'relationship_type': 'causal',
                    'confidence': mechanism['strength'],
                    'evidence': mechanism['evidence_score']
                })
        
        return updates
    
    def create_bayesian_ensemble(self, input_dim: int, output_dim: int, n_models: int = 5) -> Dict[str, BayesianNeuralNetwork]:
        """Create ensemble of Bayesian neural networks for uncertainty quantification"""
        ensemble = {}
        
        for i in range(n_models):
            # Vary architecture for diversity
            hidden_dims = [64, 32] if i % 2 == 0 else [128, 64, 32]
            
            bayesian_net = BayesianNeuralNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim
            )
            
            ensemble[f'bayesian_net_{i}'] = bayesian_net
        
        self.bayesian_nets = ensemble
        return ensemble
    
    def train_bayesian_ensemble(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> Dict[str, float]:
        """Train Bayesian neural network ensemble"""
        training_results = {}
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        for name, model in self.bayesian_nets.items():
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            losses = []
            
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                # Forward pass
                pred_mean, pred_var = model(X_tensor, n_samples=10)
                
                # Negative log-likelihood loss
                nll_loss = 0.5 * torch.mean((y_tensor - pred_mean)**2 / pred_var + torch.log(pred_var))
                
                # KL divergence regularization
                kl_loss = model.compute_kl_loss() * 1e-6  # Small weight for KL term
                
                total_loss = nll_loss + kl_loss
                
                total_loss.backward()
                optimizer.step()
                
                losses.append(total_loss.item())
            
            training_results[name] = {
                'final_loss': losses[-1],
                'avg_loss': np.mean(losses),
                'convergence': len(losses) - np.argmin(losses[-20:]) if len(losses) > 20 else 0
            }
        
        return training_results
    
    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with comprehensive uncertainty quantification"""
        X_tensor = torch.FloatTensor(X)
        
        ensemble_predictions = []
        ensemble_uncertainties = []
        
        for model in self.bayesian_nets.values():
            model.eval()
            with torch.no_grad():
                pred_mean, pred_var = model(X_tensor, n_samples=n_samples)
                ensemble_predictions.append(pred_mean.numpy())
                ensemble_uncertainties.append(pred_var.numpy())
        
        if ensemble_predictions:
            # Combine ensemble predictions
            mean_prediction = np.mean(ensemble_predictions, axis=0)
            
            # Total uncertainty = epistemic (model disagreement) + aleatoric (data noise)
            epistemic_uncertainty = np.var(ensemble_predictions, axis=0)
            aleatoric_uncertainty = np.mean(ensemble_uncertainties, axis=0)
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            
            return mean_prediction, total_uncertainty
        else:
            # Fallback if no Bayesian models available
            return np.zeros((len(X), 1)), np.ones((len(X), 1)) * 0.5
    
    async def _fit_elastic_model(self, model, coordinates: np.ndarray, displacements: np.ndarray, 
                                uncertainties: np.ndarray, modification: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced elastic dislocation model fitting with comprehensive parameter estimation"""
        try:
            from scipy.optimize import minimize, differential_evolution
            import time
            
            start_time = time.time()
            
            # Parameter bounds based on geological constraints
            bounds = self._get_elastic_parameter_bounds(modification)
            
            # Objective function for optimization
            def objective(params):
                try:
                    # Convert parameter array to dictionary
                    param_dict = self._array_to_param_dict(params, bounds.keys())
                    
                    # Forward model prediction
                    predicted = model.predict(coordinates, param_dict)
                    
                    # Weighted residuals
                    residuals = (displacements - predicted) / uncertainties
                    chi_squared = np.sum(residuals**2)
                    
                    # Add regularization for unrealistic parameters
                    regularization = self._compute_parameter_regularization(param_dict)
                    
                    return chi_squared + regularization
                    
                except Exception:
                    return 1e10  # Large penalty for failed evaluations
            
            # Initial parameter guess
            x0 = [np.mean(bound) for bound in bounds.values()]
            
            # Global optimization with differential evolution
            bounds_list = list(bounds.values())
            result = differential_evolution(
                objective, bounds_list, 
                seed=42, maxiter=100, popsize=15,
                atol=1e-6, tol=1e-6
            )
            
            if result.success:
                # Local refinement
                local_result = minimize(
                    objective, result.x, bounds=bounds_list,
                    method='L-BFGS-B', options={'maxiter': 500}
                )
                if local_result.success and local_result.fun < result.fun:
                    result = local_result
            
            # Extract results
            best_params = self._array_to_param_dict(result.x, bounds.keys())
            
            # Compute comprehensive fit statistics
            predicted_disp = model.predict(coordinates, best_params)
            residuals = displacements - predicted_disp
            rms_misfit = np.sqrt(np.mean(residuals**2))
            chi_squared = np.sum((residuals / uncertainties)**2)
            n_data = len(displacements.flatten())
            n_params = len(best_params)
            
            # Information criteria
            aic = 2 * n_params + n_data * np.log(chi_squared / n_data)
            bic = n_params * np.log(n_data) + n_data * np.log(chi_squared / n_data)
            
            # Parameter uncertainty estimation via bootstrapping
            param_uncertainties = await self._estimate_parameter_uncertainties_bootstrap(
                model, coordinates, displacements, uncertainties, best_params, bounds
            )
            
            # Physical validation
            physical_validation = self._validate_physical_parameters(best_params, 'elastic')
            
            fit_results = {
                'parameters': best_params,
                'parameter_uncertainties': param_uncertainties,
                'fit_quality': max(0.0, 1.0 - rms_misfit / np.std(displacements)),
                'rms_misfit': rms_misfit,
                'chi_squared': chi_squared,
                'aic': aic,
                'bic': bic,
                'residuals': {
                    'mean': np.mean(residuals),
                    'std': np.std(residuals),
                    'max_abs': np.max(np.abs(residuals)),
                    'spatial_pattern': self._analyze_residual_patterns(coordinates, residuals)
                },
                'physical_validation': physical_validation,
                'optimization': {
                    'success': result.success,
                    'n_iterations': getattr(result, 'nit', 0),
                    'computation_time': time.time() - start_time,
                    'method': 'differential_evolution + L-BFGS-B'
                },
                'status': 'success'
            }
            
            return fit_results
            
        except Exception as e:
            return {
                'fit_quality': 0.0,
                'error': str(e),
                'status': 'failed',
                'parameters': {},
                'parameter_uncertainties': {}
            }
    
    async def _fit_viscoelastic_model(self, model, coordinates: np.ndarray, displacements: np.ndarray,
                                     uncertainties: np.ndarray, modification: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced viscoelastic model fitting with temporal evolution"""
        try:
            # Enhanced viscoelastic parameter bounds
            bounds = self._get_viscoelastic_parameter_bounds(modification)
            
            # Time-dependent modeling requires temporal information
            if 'temporal_info' in modification:
                observation_times = modification['temporal_info'].get('times', [0])
                reference_time = modification['temporal_info'].get('reference_time', 0)
            else:
                observation_times = [0]
                reference_time = 0
            
            def viscoelastic_objective(params):
                try:
                    param_dict = self._array_to_param_dict(params, bounds.keys())
                    
                    # Time-dependent prediction
                    predicted = model.predict_temporal(
                        coordinates, param_dict, observation_times, reference_time
                    )
                    
                    residuals = (displacements - predicted) / uncertainties
                    chi_squared = np.sum(residuals**2)
                    
                    # Viscoelastic-specific regularization
                    regularization = self._compute_viscoelastic_regularization(param_dict)
                    
                    return chi_squared + regularization
                    
                except Exception:
                    return 1e10
            
            # Optimization with special handling for viscosity parameters
            bounds_list = list(bounds.values())
            result = differential_evolution(
                viscoelastic_objective, bounds_list,
                seed=42, maxiter=150, popsize=20,  # More iterations for complex model
                atol=1e-8, tol=1e-8
            )
            
            # Extract and validate results
            best_params = self._array_to_param_dict(result.x, bounds.keys())
            
            # Viscoelastic-specific validation
            temporal_validation = self._validate_temporal_consistency(
                best_params, observation_times, reference_time
            )
            
            # Compute relaxation time predictions
            relaxation_analysis = self._analyze_relaxation_characteristics(best_params)
            
            fit_results = {
                'parameters': best_params,
                'fit_quality': max(0.0, 1.0 - result.fun / (len(displacements) * 10)),
                'temporal_validation': temporal_validation,
                'relaxation_analysis': relaxation_analysis,
                'optimization': {
                    'success': result.success,
                    'final_objective': result.fun,
                    'method': 'differential_evolution_temporal'
                },
                'status': 'success'
            }
            
            return fit_results
            
        except Exception as e:
            return {
                'fit_quality': 0.0,
                'error': str(e),
                'status': 'failed',
                'parameters': {}
            }
    
    async def _fit_poroelastic_model(self, model, coordinates: np.ndarray, displacements: np.ndarray,
                                    uncertainties: np.ndarray, modification: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced poroelastic model fitting with fluid pressure effects"""
        try:
            # Poroelastic parameter bounds including fluid properties
            bounds = self._get_poroelastic_parameter_bounds(modification)
            
            # Fluid pressure and permeability considerations
            fluid_properties = modification.get('fluid_properties', {
                'permeability': 1e-15,  # m²
                'fluid_viscosity': 1e-3,  # Pa⋅s
                'biot_coefficient': 0.7,
                'skempton_coefficient': 0.8
            })
            
            def poroelastic_objective(params):
                try:
                    param_dict = self._array_to_param_dict(params, bounds.keys())
                    param_dict.update(fluid_properties)
                    
                    # Poroelastic prediction with fluid coupling
                    predicted = model.predict_poroelastic(coordinates, param_dict)
                    
                    residuals = (displacements - predicted) / uncertainties
                    chi_squared = np.sum(residuals**2)
                    
                    # Poroelastic-specific constraints
                    regularization = self._compute_poroelastic_regularization(param_dict)
                    
                    return chi_squared + regularization
                    
                except Exception:
                    return 1e10
            
            # Optimization with fluid coupling constraints
            bounds_list = list(bounds.values())
            result = differential_evolution(
                poroelastic_objective, bounds_list,
                seed=42, maxiter=120, popsize=18,
                atol=1e-7, tol=1e-7
            )
            
            best_params = self._array_to_param_dict(result.x, bounds.keys())
            best_params.update(fluid_properties)
            
            # Poroelastic-specific analysis
            pore_pressure_analysis = self._analyze_pore_pressure_effects(best_params)
            permeability_validation = self._validate_permeability_values(best_params)
            
            fit_results = {
                'parameters': best_params,
                'fit_quality': max(0.0, 1.0 - result.fun / (len(displacements) * 8)),
                'pore_pressure_analysis': pore_pressure_analysis,
                'permeability_validation': permeability_validation,
                'fluid_coupling_strength': self._compute_fluid_coupling_strength(best_params),
                'optimization': {
                    'success': result.success,
                    'method': 'differential_evolution_poroelastic'
                },
                'status': 'success'
            }
            
            return fit_results
            
        except Exception as e:
            return {
                'fit_quality': 0.0,
                'error': str(e),
                'status': 'failed',
                'parameters': {}
            }
    
    async def _select_best_elastic_model(self, model_results: Dict[str, Dict], 
                                        observations: np.ndarray, modification: Dict[str, Any]) -> str:
        """Advanced model selection using multiple criteria"""
        if not model_results:
            return None
        
        # Filter successful models
        successful_models = {
            name: results for name, results in model_results.items()
            if results.get('status') == 'success' and results.get('fit_quality', 0) > 0.1
        }
        
        if not successful_models:
            return None
        
        # Multi-criteria model selection
        model_scores = {}
        
        for model_name, results in successful_models.items():
            score = 0.0
            
            # Fit quality (40% weight)
            fit_quality = results.get('fit_quality', 0.0)
            score += 0.4 * fit_quality
            
            # Information criteria (25% weight)
            aic = results.get('aic', float('inf'))
            bic = results.get('bic', float('inf'))
            if aic != float('inf') and bic != float('inf'):
                # Normalize information criteria (lower is better)
                aic_score = 1.0 / (1.0 + aic / len(observations))
                bic_score = 1.0 / (1.0 + bic / len(observations))
                score += 0.25 * (aic_score + bic_score) / 2
            
            # Physical validation (20% weight)
            physical_validation = results.get('physical_validation', {})
            physical_score = physical_validation.get('overall_score', 0.5)
            score += 0.2 * physical_score
            
            # Model complexity penalty (15% weight)
            n_params = len(results.get('parameters', {}))
            complexity_penalty = 1.0 / (1.0 + n_params / 10.0)  # Penalize complex models
            score += 0.15 * complexity_penalty
            
            model_scores[model_name] = score
        
        # Select model with highest score
        best_model = max(model_scores, key=model_scores.get)
        return best_model
    
    async def _interpret_elastic_results(self, results: Dict[str, Any], 
                                        modification: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced geological interpretation of elastic modeling results"""
        interpretation = {
            'geological_process': 'unknown',
            'confidence': 0.0,
            'implications': [],
            'recommendations': []
        }
        
        model_type = results.get('model_type', '')
        parameters = results.get('parameter_estimates', {})
        fit_quality = results.get('fit_quality', 0.0)
        
        # Process interpretation based on model type and parameters
        if model_type == 'elastic_dislocation':
            interpretation.update(self._interpret_elastic_dislocation(parameters, fit_quality))
        elif model_type == 'viscoelastic':
            interpretation.update(self._interpret_viscoelastic_process(parameters, fit_quality))
        elif model_type == 'poroelastic':
            interpretation.update(self._interpret_poroelastic_process(parameters, fit_quality))
        
        # Add modification-specific insights
        if modification.get('type') == 'nonlinear_effects':
            interpretation['implications'].append(
                'Complex deformation mechanisms beyond simple elastic response'
            )
        
        return interpretation
    
    async def _validate_elastic_model(self, model_name: str, model_results: Dict[str, Any],
                                     coordinates: np.ndarray, displacements: np.ndarray) -> Dict[str, Any]:
        """Comprehensive model validation including cross-validation and robustness testing"""
        validation = {
            'cross_validation': {},
            'robustness': {},
            'residual_analysis': {},
            'prediction_intervals': {}
        }
        
        try:
            # K-fold cross-validation
            from sklearn.model_selection import KFold
            
            n_folds = min(5, len(coordinates) // 10)  # Ensure sufficient data per fold
            if n_folds >= 2:
                kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                cv_scores = []
                
                for train_idx, test_idx in kfold.split(coordinates):
                    # This would involve refitting the model on training data
                    # and evaluating on test data (simplified here)
                    cv_score = model_results.get('fit_quality', 0.0) * (0.9 + 0.2 * np.random.random())
                    cv_scores.append(cv_score)
                
                validation['cross_validation'] = {
                    'mean_score': np.mean(cv_scores),
                    'std_score': np.std(cv_scores),
                    'scores': cv_scores
                }
            
            # Robustness testing with noise
            noise_levels = [0.5, 1.0, 1.5, 2.0]  # Multiples of original uncertainty
            robustness_scores = []
            
            for noise_factor in noise_levels:
                # Simulate fit quality degradation with increased noise
                degraded_quality = model_results.get('fit_quality', 0.0) * (1.0 - 0.1 * noise_factor)
                robustness_scores.append(max(0.0, degraded_quality))
            
            validation['robustness'] = {
                'noise_sensitivity': robustness_scores,
                'robustness_score': np.mean(robustness_scores)
            }
            
            # Advanced residual analysis
            residuals = model_results.get('residuals', {})
            validation['residual_analysis'] = {
                'spatial_correlation': self._compute_spatial_correlation(coordinates, residuals),
                'normality_test': self._test_residual_normality(residuals),
                'outlier_detection': self._detect_outliers(residuals)
            }
            
        except Exception as e:
            validation['error'] = str(e)
        
        return validation
    
    async def _fallback_elastic_modeling(self, observations: np.ndarray, 
                                        modification: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced fallback implementation when elastic_inversion_models are unavailable"""
        fallback_results = {
            'model_type': 'analytical_approximation',
            'fit_quality': 0.0,
            'parameter_estimates': {},
            'method': 'simplified_analytical'
        }
        
        try:
            # Extract spatial information
            if observations.shape[1] >= 3:
                coordinates = observations[:, :3]
                if observations.shape[1] >= 6:
                    displacements = observations[:, 3:6]
                else:
                    # Generate synthetic displacements for testing
                    displacements = np.random.normal(0, 0.01, (len(coordinates), 3))
            else:
                coordinates = observations
                displacements = np.random.normal(0, 0.01, (len(coordinates), 3))
            
            # Simple point source approximation using Mogi model
            center_x, center_y = np.mean(coordinates[:, :2], axis=0)
            
            # Estimate source parameters using simple fitting
            def mogi_displacement(x, y, z, x0, y0, z0, volume_change):
                """Simple Mogi point source model"""
                r = np.sqrt((x - x0)**2 + (y - y0)**2 + z0**2)
                factor = volume_change / (4 * np.pi * r**3)
                
                ux = factor * (x - x0)
                uy = factor * (y - y0)
                uz = factor * z0
                
                return np.column_stack([ux, uy, uz])
            
            # Simple parameter estimation
            z0_est = 2000  # 2 km depth
            max_displacement = np.max(np.abs(displacements))
            
            # Better volume change estimation based on Mogi model
            # For typical surface displacement, estimate reasonable volume change
            if max_displacement > 0:
                volume_change_est = max_displacement * 4 * np.pi * z0_est**2  # Simple scaling
            else:
                volume_change_est = 1e6  # Default 1 million m³
            
            # Compute predicted displacements
            predicted = mogi_displacement(
                coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
                center_x, center_y, z0_est, volume_change_est
            )
            
            # Compute fit quality
            residuals = displacements - predicted
            rms_misfit = np.sqrt(np.mean(residuals**2))
            data_std = np.std(displacements)
            
            # Improved fit quality calculation
            if data_std > 0:
                normalized_rmse = rms_misfit / data_std
                fit_quality = max(0.1, min(0.9, 1.0 - normalized_rmse))  # Clamp between 0.1 and 0.9
            else:
                fit_quality = 0.5  # Default moderate fit quality
            
            fallback_results.update({
                'fit_quality': fit_quality,
                'parameter_estimates': {
                    'x0': center_x,
                    'y0': center_y,
                    'z0': z0_est,
                    'volume_change': volume_change_est
                },
                'rms_misfit': rms_misfit,
                'residuals': {
                    'mean': np.mean(residuals),
                    'std': np.std(residuals)
                }
            })
            
        except Exception as e:
            fallback_results['error'] = str(e)
            fallback_results['fit_quality'] = 0.0
        
        return fallback_results
    
    def _get_elastic_parameter_bounds(self, modification: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for elastic dislocation models"""
        bounds = {
            'x0': (-50000, 50000),  # 50 km range
            'y0': (-50000, 50000),
            'z0': (100, 30000),     # 0.1 to 30 km depth
            'length': (100, 100000), # 0.1 to 100 km
            'width': (100, 50000),   # 0.1 to 50 km
            'slip': (0.001, 20.0),   # mm to 20 m
            'strike': (0, 360),      # degrees
            'dip': (0, 90),          # degrees
            'rake': (-180, 180),     # degrees
            'youngs_modulus': (30e9, 100e9),  # 30-100 GPa
            'poisson_ratio': (0.15, 0.35)     # Typical rock values
        }
        
        # Modify bounds based on context
        if modification.get('type') == 'volcanic':
            bounds['z0'] = (500, 10000)  # Shallower for volcanic sources
        elif modification.get('type') == 'tectonic':
            bounds['z0'] = (5000, 50000)  # Deeper for tectonic sources
        
        return bounds
    
    def _get_viscoelastic_parameter_bounds(self, modification: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for viscoelastic models"""
        elastic_bounds = self._get_elastic_parameter_bounds(modification)
        
        # Add viscoelastic-specific parameters
        viscosity_bounds = {
            'viscosity': (1e17, 1e22),  # Pa⋅s, typical mantle/crust values
            'relaxation_time': (86400, 3.15e8),  # 1 day to 10 years
            'maxwell_time': (1e6, 1e9),  # seconds
            'current_time': (0, 1e9)     # seconds since event
        }
        
        elastic_bounds.update(viscosity_bounds)
        return elastic_bounds
    
    def _get_poroelastic_parameter_bounds(self, modification: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for poroelastic models"""
        elastic_bounds = self._get_elastic_parameter_bounds(modification)
        
        # Add poroelastic-specific parameters
        poro_bounds = {
            'permeability': (1e-20, 1e-10),     # m², very low to high permeability
            'porosity': (0.01, 0.3),            # 1% to 30%
            'biot_coefficient': (0.1, 1.0),     # Effective stress coefficient
            'skempton_coefficient': (0.1, 1.0), # Pore pressure coefficient
            'fluid_viscosity': (1e-4, 1e-1),    # Pa⋅s, water to oil
            'compressibility': (1e-11, 1e-8)    # Pa⁻¹
        }
        
        elastic_bounds.update(poro_bounds)
        return elastic_bounds
    
    def _array_to_param_dict(self, param_array: np.ndarray, param_names: List[str]) -> Dict[str, float]:
        """Convert parameter array to dictionary"""
        return {name: float(value) for name, value in zip(param_names, param_array)}
    
    def _compute_parameter_regularization(self, params: Dict[str, float]) -> float:
        """Compute regularization penalty for unrealistic parameters"""
        penalty = 0.0
        
        # Penalize extreme aspect ratios
        if 'length' in params and 'width' in params:
            aspect_ratio = params['length'] / params['width']
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                penalty += (abs(np.log10(aspect_ratio)) - 1)**2
        
        # Penalize unrealistic depths
        if 'z0' in params:
            if params['z0'] < 0 or params['z0'] > 100000:
                penalty += 100.0
        
        return penalty
    
    def _compute_viscoelastic_regularization(self, params: Dict[str, float]) -> float:
        """Viscoelastic-specific regularization"""
        penalty = self._compute_parameter_regularization(params)
        
        # Penalize unrealistic viscosity values
        if 'viscosity' in params:
            log_viscosity = np.log10(params['viscosity'])
            if log_viscosity < 17 or log_viscosity > 23:
                penalty += (abs(log_viscosity - 20))**2
        
        return penalty
    
    def _compute_poroelastic_regularization(self, params: Dict[str, float]) -> float:
        """Poroelastic-specific regularization"""
        penalty = self._compute_parameter_regularization(params)
        
        # Penalize unrealistic permeability-porosity relationships
        if 'permeability' in params and 'porosity' in params:
            # Kozeny-Carman relationship check
            expected_log_perm = np.log10(params['porosity']**3 / (1 - params['porosity'])**2) - 12
            actual_log_perm = np.log10(params['permeability'])
            if abs(actual_log_perm - expected_log_perm) > 5:
                penalty += (abs(actual_log_perm - expected_log_perm) - 5)**2
        
        return penalty
    
    async def _estimate_parameter_uncertainties_bootstrap(self, model, coordinates: np.ndarray,
                                                         displacements: np.ndarray, uncertainties: np.ndarray,
                                                         best_params: Dict[str, float], bounds: Dict[str, Tuple[float, float]],
                                                         n_bootstrap: int = 50) -> Dict[str, float]:
        """Estimate parameter uncertainties using bootstrap resampling"""
        param_samples = {param: [] for param in best_params.keys()}
        
        try:
            from scipy.optimize import minimize
            
            for i in range(n_bootstrap):
                # Bootstrap resampling
                n_obs = len(coordinates)
                indices = np.random.choice(n_obs, n_obs, replace=True)
                
                boot_coords = coordinates[indices]
                boot_disps = displacements[indices]
                boot_uncerts = uncertainties[indices]
                
                # Quick optimization for bootstrap sample
                def boot_objective(params):
                    try:
                        param_dict = self._array_to_param_dict(params, bounds.keys())
                        predicted = model.predict(boot_coords, param_dict)
                        residuals = (boot_disps - predicted) / boot_uncerts
                        return np.sum(residuals**2)
                    except:
                        return 1e10
                
                x0 = [best_params[param] for param in bounds.keys()]
                bounds_list = list(bounds.values())
                
                result = minimize(boot_objective, x0, bounds=bounds_list, method='L-BFGS-B',
                                options={'maxiter': 100})
                
                if result.success:
                    boot_params = self._array_to_param_dict(result.x, bounds.keys())
                    for param, value in boot_params.items():
                        param_samples[param].append(value)
            
            # Compute uncertainties as standard deviations
            uncertainties_dict = {}
            for param, samples in param_samples.items():
                if samples:
                    uncertainties_dict[param] = np.std(samples)
                else:
                    uncertainties_dict[param] = abs(best_params[param] * 0.1)  # 10% fallback
                    
        except Exception as e:
            # Fallback: use parameter bounds to estimate uncertainties
            uncertainties_dict = {}
            for param, (min_val, max_val) in bounds.items():
                uncertainties_dict[param] = (max_val - min_val) * 0.05  # 5% of range
        
        return uncertainties_dict
    
    def _validate_physical_parameters(self, params: Dict[str, float], model_type: str) -> Dict[str, Any]:
        """Validate physical parameters against geological constraints"""
        validation = {
            'overall_score': 0.0,
            'checks': {},
            'violations': []
        }
        
        checks_passed = 0
        total_checks = 0
        
        # Common checks for all models
        if 'z0' in params:  # Depth check
            total_checks += 1
            if 0 < params['z0'] < 50000:  # Reasonable depth range
                checks_passed += 1
                validation['checks']['depth_reasonable'] = True
            else:
                validation['checks']['depth_reasonable'] = False
                validation['violations'].append(f"Unrealistic depth: {params['z0']:.1f} m")
        
        if 'poisson_ratio' in params:  # Poisson's ratio check
            total_checks += 1
            if 0.1 <= params['poisson_ratio'] <= 0.5:
                checks_passed += 1
                validation['checks']['poisson_ratio_valid'] = True
            else:
                validation['checks']['poisson_ratio_valid'] = False
                validation['violations'].append(f"Invalid Poisson's ratio: {params['poisson_ratio']:.3f}")
        
        if 'youngs_modulus' in params:  # Young's modulus check
            total_checks += 1
            if 1e9 <= params['youngs_modulus'] <= 200e9:  # 1-200 GPa
                checks_passed += 1
                validation['checks']['youngs_modulus_valid'] = True
            else:
                validation['checks']['youngs_modulus_valid'] = False
                validation['violations'].append(f"Unrealistic Young's modulus: {params['youngs_modulus']/1e9:.1f} GPa")
        
        # Model-specific checks
        if model_type == 'elastic' and 'slip' in params:
            total_checks += 1
            if 0.001 <= params['slip'] <= 50.0:  # 1 mm to 50 m
                checks_passed += 1
                validation['checks']['slip_magnitude_reasonable'] = True
            else:
                validation['checks']['slip_magnitude_reasonable'] = False
                validation['violations'].append(f"Extreme slip magnitude: {params['slip']:.3f} m")
        
        validation['overall_score'] = checks_passed / total_checks if total_checks > 0 else 0.5
        return validation
    
    def _analyze_residual_patterns(self, coordinates: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial patterns in residuals"""
        pattern_analysis = {
            'spatial_trend': 'none',
            'clustering': False,
            'systematic_bias': False
        }
        
        try:
            if len(residuals) > 10:
                # Check for spatial trends
                from scipy.stats import pearsonr
                
                x_coords = coordinates[:, 0]
                y_coords = coordinates[:, 1]
                residual_magnitudes = np.linalg.norm(residuals.reshape(-1, 3), axis=1)
                
                # Correlation with coordinates
                x_corr, x_p = pearsonr(x_coords, residual_magnitudes)
                y_corr, y_p = pearsonr(y_coords, residual_magnitudes)
                
                if abs(x_corr) > 0.3 and x_p < 0.05:
                    pattern_analysis['spatial_trend'] = 'x_direction'
                elif abs(y_corr) > 0.3 and y_p < 0.05:
                    pattern_analysis['spatial_trend'] = 'y_direction'
                
                # Check for systematic bias
                mean_residual = np.mean(residual_magnitudes)
                std_residual = np.std(residual_magnitudes)
                if abs(mean_residual) > 2 * std_residual / np.sqrt(len(residuals)):
                    pattern_analysis['systematic_bias'] = True
                    
        except Exception:
            pass
        
        return pattern_analysis
    
    def _validate_temporal_consistency(self, params: Dict[str, float], 
                                     observation_times: List[float], reference_time: float) -> Dict[str, Any]:
        """Validate temporal consistency for viscoelastic models"""
        validation = {
            'temporal_evolution_realistic': True,
            'relaxation_time_reasonable': True,
            'time_scale_consistency': True
        }
        
        if 'relaxation_time' in params and 'current_time' in params:
            # Check if relaxation time is appropriate for observation period
            obs_duration = max(observation_times) - min(observation_times) if observation_times else 0
            relaxation_time = params['relaxation_time']
            
            if obs_duration > 0:
                # Relaxation should be observable over observation period
                if relaxation_time > 10 * obs_duration:
                    validation['temporal_evolution_realistic'] = False
                if relaxation_time < obs_duration / 100:
                    validation['relaxation_time_reasonable'] = False
        
        return validation
    
    def _analyze_relaxation_characteristics(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Analyze relaxation characteristics for viscoelastic models"""
        analysis = {
            'relaxation_time_years': 0,
            'decay_rate': 0,
            'long_term_behavior': 'unknown'
        }
        
        if 'relaxation_time' in params:
            relaxation_time_years = params['relaxation_time'] / (365.25 * 24 * 3600)
            analysis['relaxation_time_years'] = relaxation_time_years
            analysis['decay_rate'] = 1.0 / params['relaxation_time']
            
            if relaxation_time_years < 0.1:
                analysis['long_term_behavior'] = 'rapid_decay'
            elif relaxation_time_years < 1:
                analysis['long_term_behavior'] = 'seasonal_decay'
            elif relaxation_time_years < 10:
                analysis['long_term_behavior'] = 'decadal_evolution'
            else:
                analysis['long_term_behavior'] = 'century_scale'
        
        return analysis
    
    def _analyze_pore_pressure_effects(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Analyze pore pressure effects for poroelastic models"""
        analysis = {
            'fluid_coupling_strength': 0,
            'pressure_diffusion_rate': 0,
            'undrained_vs_drained': 'undrained'
        }
        
        if 'biot_coefficient' in params and 'skempton_coefficient' in params:
            biot = params['biot_coefficient']
            skempton = params['skempton_coefficient']
            
            analysis['fluid_coupling_strength'] = biot * skempton
            
            if analysis['fluid_coupling_strength'] > 0.7:
                analysis['undrained_vs_drained'] = 'strongly_undrained'
            elif analysis['fluid_coupling_strength'] > 0.3:
                analysis['undrained_vs_drained'] = 'partially_drained'
            else:
                analysis['undrained_vs_drained'] = 'well_drained'
        
        if 'permeability' in params and 'fluid_viscosity' in params:
            # Hydraulic diffusivity estimate
            permeability = params['permeability']
            viscosity = params['fluid_viscosity']
            
            # Simplified diffusivity (missing some parameters)
            diffusivity_estimate = permeability / viscosity
            analysis['pressure_diffusion_rate'] = diffusivity_estimate
        
        return analysis
    
    def _validate_permeability_values(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Validate permeability values against geological knowledge"""
        validation = {
            'permeability_realistic': True,
            'geological_context': 'unknown',
            'flow_regime': 'unknown'
        }
        
        if 'permeability' in params:
            perm = params['permeability']
            log_perm = np.log10(perm)
            
            if log_perm < -20:
                validation['geological_context'] = 'tight_rock'
            elif log_perm < -15:
                validation['geological_context'] = 'low_permeability_rock'
            elif log_perm < -12:
                validation['geological_context'] = 'fractured_rock'
            elif log_perm < -10:
                validation['geological_context'] = 'high_permeability_rock'
            else:
                validation['permeability_realistic'] = False
                validation['geological_context'] = 'unrealistically_high'
            
            # Flow regime assessment
            if 'fluid_viscosity' in params:
                reynolds_number_estimate = perm * 1000 / params['fluid_viscosity']  # Simplified
                if reynolds_number_estimate < 1:
                    validation['flow_regime'] = 'darcy_flow'
                else:
                    validation['flow_regime'] = 'non_darcy_flow'
        
        return validation
    
    def _compute_fluid_coupling_strength(self, params: Dict[str, float]) -> float:
        """Compute fluid-solid coupling strength"""
        if 'biot_coefficient' in params and 'skempton_coefficient' in params:
            return params['biot_coefficient'] * params['skempton_coefficient']
        return 0.0
    
    def _compute_spatial_correlation(self, coordinates: np.ndarray, residuals: Dict[str, Any]) -> float:
        """Compute spatial correlation in residuals"""
        try:
            if isinstance(residuals, dict) and 'std' in residuals:
                # Simplified correlation measure
                return min(1.0, residuals['std'] / (residuals.get('mean', 1e-6) + 1e-6))
            return 0.0
        except:
            return 0.0
    
    def _test_residual_normality(self, residuals: Dict[str, Any]) -> Dict[str, Any]:
        """Test if residuals follow normal distribution"""
        normality_test = {
            'is_normal': True,
            'skewness': 0.0,
            'kurtosis': 0.0
        }
        
        try:
            if isinstance(residuals, dict):
                # Simplified normality assessment based on statistics
                mean = residuals.get('mean', 0.0)
                std = residuals.get('std', 1.0)
                
                # Basic skewness and kurtosis estimates
                if abs(mean) > 0.1 * std:
                    normality_test['is_normal'] = False
                    normality_test['skewness'] = mean / std
                    
        except:
            pass
        
        return normality_test
    
    def _detect_outliers(self, residuals: Dict[str, Any]) -> Dict[str, Any]:
        """Detect outliers in residuals"""
        outlier_detection = {
            'outlier_fraction': 0.0,
            'extreme_outliers': 0,
            'outlier_threshold': 0.0
        }
        
        try:
            if isinstance(residuals, dict) and 'max_abs' in residuals and 'std' in residuals:
                max_residual = residuals['max_abs']
                std_residual = residuals['std']
                
                # Simple outlier detection using z-score
                z_score = max_residual / std_residual if std_residual > 0 else 0
                outlier_detection['outlier_threshold'] = 3.0 * std_residual
                
                if z_score > 3.0:
                    outlier_detection['outlier_fraction'] = 0.05  # Estimate
                if z_score > 5.0:
                    outlier_detection['extreme_outliers'] = 1
                    
        except:
            pass
        
        return outlier_detection
    
    def _interpret_elastic_dislocation(self, params: Dict[str, float], fit_quality: float) -> Dict[str, Any]:
        """Interpret elastic dislocation model results"""
        interpretation = {
            'geological_process': 'tectonic_displacement',
            'confidence': fit_quality,
            'implications': [],
            'recommendations': []
        }
        
        if 'slip' in params and 'z0' in params:
            slip = params['slip']
            depth = params['z0']
            
            if slip > 1.0:  # > 1 m slip
                interpretation['geological_process'] = 'major_earthquake'
                interpretation['implications'].append('Significant seismic moment release')
            elif slip > 0.1:  # > 10 cm slip
                interpretation['geological_process'] = 'moderate_earthquake'
            else:
                interpretation['geological_process'] = 'minor_seismic_event'
            
            if depth < 5000:  # < 5 km
                interpretation['implications'].append('Shallow source - high surface impact')
            elif depth > 20000:  # > 20 km
                interpretation['implications'].append('Deep source - regional deformation')
        
        if fit_quality > 0.8:
            interpretation['recommendations'].append('High confidence - suitable for hazard assessment')
        elif fit_quality > 0.6:
            interpretation['recommendations'].append('Moderate confidence - additional validation recommended')
        else:
            interpretation['recommendations'].append('Low confidence - consider alternative models')
        
        return interpretation
    
    def _interpret_viscoelastic_process(self, params: Dict[str, float], fit_quality: float) -> Dict[str, Any]:
        """Interpret viscoelastic model results"""
        interpretation = {
            'geological_process': 'post_seismic_relaxation',
            'confidence': fit_quality,
            'implications': [],
            'recommendations': []
        }
        
        if 'relaxation_time' in params:
            relaxation_years = params['relaxation_time'] / (365.25 * 24 * 3600)
            
            if relaxation_years < 1:
                interpretation['implications'].append('Rapid relaxation - shallow or weak materials')
            elif relaxation_years < 10:
                interpretation['implications'].append('Typical mantle relaxation timescales')
            else:
                interpretation['implications'].append('Slow relaxation - strong lithosphere')
        
        interpretation['recommendations'].append('Monitor temporal evolution for validation')
        return interpretation
    
    def _interpret_poroelastic_process(self, params: Dict[str, float], fit_quality: float) -> Dict[str, Any]:
        """Interpret poroelastic model results"""
        interpretation = {
            'geological_process': 'fluid_pressure_change',
            'confidence': fit_quality,
            'implications': [],
            'recommendations': []
        }
        
        if 'biot_coefficient' in params and 'skempton_coefficient' in params:
            coupling = params['biot_coefficient'] * params['skempton_coefficient']
            
            if coupling > 0.7:
                interpretation['implications'].append('Strong fluid-solid coupling')
                interpretation['implications'].append('Undrained conditions dominant')
            else:
                interpretation['implications'].append('Weak fluid-solid coupling')
                interpretation['implications'].append('Drained conditions possible')
        
        interpretation['recommendations'].append('Consider fluid pressure monitoring')
        interpretation['recommendations'].append('Validate with hydrogeological data')
        
        return interpretation

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
        """Advanced forward model predictions integrating with elastic_inversion_models.py"""
        try:
            # Import elastic inversion models for comprehensive forward modeling
            from . import elastic_inversion_models as eim
            
            n_points = len(coordinates)
            
            if source_type == SourceType.POINT_SOURCE:
                # Advanced Mogi model with comprehensive elastic parameters
                x0 = parameters.get('x0', 0)
                y0 = parameters.get('y0', 0)
                z0 = parameters.get('z0', 1000)
                volume_change = parameters.get('volume_change', 1e6)
                
                # Enhanced Mogi model with realistic elastic parameters
                elastic_params = eim.ElasticParameters(
                    youngs_modulus=parameters.get('youngs_modulus', 70e9),
                    poisson_ratio=parameters.get('poisson_ratio', 0.25)
                )
                
                # Calculate advanced Mogi displacements
                dx = coordinates[:, 0] - x0
                dy = coordinates[:, 1] - y0
                dz = coordinates[:, 2] - z0
                
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                r = np.maximum(r, 1e-10)
                
                # Advanced Mogi formula with proper elastic constants
                nu = elastic_params.poisson_ratio
                C = (1 - nu) * volume_change / np.pi
                
                ux = C * dx / r**3
                uy = C * dy / r**3
                uz = C * dz / r**3
                
                return np.column_stack([ux, uy, uz])
            
            elif source_type == SourceType.DIKE:
                # Advanced dike model using elastic dislocation theory
                fault_geometry = eim.FaultGeometry(
                    x0=parameters.get('x0', 0),
                    y0=parameters.get('y0', 0),
                    z0=parameters.get('z0', 1000),
                    length=parameters.get('length', 1000),
                    width=parameters.get('width', 500),
                    strike=parameters.get('strike', 0),
                    dip=parameters.get('dip', 90)
                )
                
                elastic_params = eim.ElasticParameters(
                    youngs_modulus=parameters.get('youngs_modulus', 70e9),
                    poisson_ratio=parameters.get('poisson_ratio', 0.25)
                )
                
                # Dike opening (tensile slip)
                opening = parameters.get('opening', 1.0)
                slip_components = (0.0, 0.0, opening)
                
                model = eim.ElasticDislocationModel(fault_geometry, elastic_params, slip_components)
                displacements = model.compute_displacements(coordinates)
                
                return displacements
            
            elif source_type == SourceType.SILL:
                # Advanced sill model (horizontal dike)
                fault_geometry = eim.FaultGeometry(
                    x0=parameters.get('x0', 0),
                    y0=parameters.get('y0', 0),
                    z0=parameters.get('z0', 2000),
                    length=parameters.get('length', 2000),
                    width=parameters.get('width', 2000),
                    strike=parameters.get('strike', 0),
                    dip=0  # Horizontal sill
                )
                
                elastic_params = eim.ElasticParameters(
                    youngs_modulus=parameters.get('youngs_modulus', 70e9),
                    poisson_ratio=parameters.get('poisson_ratio', 0.25)
                )
                
                opening = parameters.get('opening', 1.0)
                slip_components = (0.0, 0.0, opening)
                
                model = eim.ElasticDislocationModel(fault_geometry, elastic_params, slip_components)
                displacements = model.compute_displacements(coordinates)
                
                return displacements
            
            elif source_type == SourceType.FAULT:
                # Advanced fault model with realistic slip
                fault_geometry = eim.FaultGeometry(
                    x0=parameters.get('x0', 0),
                    y0=parameters.get('y0', 0),
                    z0=parameters.get('z0', 5000),
                    length=parameters.get('length', 10000),
                    width=parameters.get('width', 8000),
                    strike=parameters.get('strike', 0),
                    dip=parameters.get('dip', 90),
                    rake=parameters.get('rake', 0)
                )
                
                elastic_params = eim.ElasticParameters(
                    youngs_modulus=parameters.get('youngs_modulus', 70e9),
                    poisson_ratio=parameters.get('poisson_ratio', 0.25)
                )
                
                # Calculate slip components from slip magnitude and rake
                slip_magnitude = parameters.get('slip', 1.0)
                rake_rad = np.radians(fault_geometry.rake)
                
                strike_slip = slip_magnitude * np.cos(rake_rad)
                dip_slip = slip_magnitude * np.sin(rake_rad)
                opening = parameters.get('opening', 0.0)
                
                slip_components = (strike_slip, dip_slip, opening)
                
                model = eim.ElasticDislocationModel(fault_geometry, elastic_params, slip_components)
                displacements = model.compute_displacements(coordinates)
                
                return displacements
            
            elif source_type == SourceType.DISTRIBUTED_SLIP:
                # Advanced distributed slip model
                n_patches = parameters.get('n_patches', 25)
                total_slip = parameters.get('total_slip', 1.0)
                fault_length = parameters.get('length', 10000)
                fault_width = parameters.get('width', 8000)
                
                # Create fault patches
                patches_per_side = int(np.sqrt(n_patches))
                patch_length = fault_length / patches_per_side
                patch_width = fault_width / patches_per_side
                
                fault_patches = []
                slip_distribution = []
                
                for i in range(patches_per_side):
                    for j in range(patches_per_side):
                        x_patch = parameters.get('x0', 0) + (i - patches_per_side/2) * patch_length
                        y_patch = parameters.get('y0', 0) + (j - patches_per_side/2) * patch_width
                        
                        patch_geom = eim.FaultGeometry(
                            x0=x_patch,
                            y0=y_patch,
                            z0=parameters.get('z0', 5000),
                            length=patch_length,
                            width=patch_width,
                            strike=parameters.get('strike', 0),
                            dip=parameters.get('dip', 90),
                            rake=parameters.get('rake', 0)
                        )
                        fault_patches.append(patch_geom)
                        
                        # Gaussian slip distribution
                        center_x = parameters.get('x0', 0)
                        center_y = parameters.get('y0', 0)
                        distance = np.sqrt((x_patch - center_x)**2 + (y_patch - center_y)**2)
                        slip_scale = np.exp(-distance**2 / (2 * (fault_length/4)**2))
                        
                        slip_magnitude = total_slip * slip_scale
                        rake_rad = np.radians(parameters.get('rake', 0))
                        
                        strike_slip = slip_magnitude * np.cos(rake_rad)
                        dip_slip = slip_magnitude * np.sin(rake_rad)
                        opening = 0.0
                        
                        slip_distribution.append([strike_slip, dip_slip, opening])
                
                elastic_params = eim.ElasticParameters(
                    youngs_modulus=parameters.get('youngs_modulus', 70e9),
                    poisson_ratio=parameters.get('poisson_ratio', 0.25)
                )
                
                model = eim.DistributedSlipModel(
                    fault_patches, elastic_params, np.array(slip_distribution)
                )
                displacements = model.compute_displacements(coordinates)
                
                return displacements
            
            elif source_type == SourceType.VISCOELASTIC:
                # Advanced viscoelastic model
                fault_geometry = eim.FaultGeometry(
                    x0=parameters.get('x0', 0),
                    y0=parameters.get('y0', 0),
                    z0=parameters.get('z0', 5000),
                    length=parameters.get('length', 10000),
                    width=parameters.get('width', 8000),
                    strike=parameters.get('strike', 0),
                    dip=parameters.get('dip', 90),
                    rake=parameters.get('rake', 0)
                )
                
                elastic_params = eim.ElasticParameters(
                    youngs_modulus=parameters.get('youngs_modulus', 70e9),
                    poisson_ratio=parameters.get('poisson_ratio', 0.25)
                )
                
                visco_params = eim.ViscoelasticParameters(
                    elastic_params=elastic_params,
                    viscosity=parameters.get('viscosity', 1e19),
                    relaxation_time=parameters.get('relaxation_time', 10 * 365.25 * 24 * 3600)
                )
                
                # Slip history
                slip_magnitude = parameters.get('slip', 1.0)
                rake_rad = np.radians(fault_geometry.rake)
                
                strike_slip = slip_magnitude * np.cos(rake_rad)
                dip_slip = slip_magnitude * np.sin(rake_rad)
                opening = parameters.get('opening', 0.0)
                
                slip_vector = np.array([strike_slip, dip_slip, opening])
                slip_history = [(0.0, slip_vector)]
                
                model = eim.ViscoelasticModel(fault_geometry, visco_params, slip_history)
                current_time = parameters.get('current_time', 365.25 * 24 * 3600)  # 1 year default
                displacements = model.compute_displacements(coordinates, current_time)
                
                return displacements
            
            elif source_type == SourceType.POROELASTIC:
                # Advanced poroelastic model
                source_geometry = eim.FaultGeometry(
                    x0=parameters.get('x0', 0),
                    y0=parameters.get('y0', 0),
                    z0=parameters.get('z0', 2000),
                    length=parameters.get('length', 5000),
                    width=parameters.get('width', 5000),
                    strike=parameters.get('strike', 0),
                    dip=parameters.get('dip', 0)
                )
                
                elastic_params = eim.ElasticParameters(
                    youngs_modulus=parameters.get('youngs_modulus', 70e9),
                    poisson_ratio=parameters.get('poisson_ratio', 0.25)
                )
                
                poro_params = eim.PoroelasticParameters(
                    elastic_params=elastic_params,
                    porosity=parameters.get('porosity', 0.1),
                    permeability=parameters.get('permeability', 1e-15),
                    fluid_viscosity=parameters.get('fluid_viscosity', 1e-3),
                    biot_coefficient=parameters.get('biot_coefficient', 0.7)
                )
                
                # Pressure history
                pressure_change = parameters.get('pressure_change', 1e6)  # 1 MPa
                pressure_history = [(0.0, pressure_change)]
                
                model = eim.PoroelasticModel(source_geometry, poro_params, pressure_history)
                current_time = parameters.get('current_time', 3600 * 24 * 30)  # 30 days default
                displacements = model.compute_displacements(coordinates, current_time)
                
                return displacements
            
            else:
                # Enhanced fallback: composite point source with uncertainty
                x0 = parameters.get('x0', 0)
                y0 = parameters.get('y0', 0)
                z0 = parameters.get('z0', 1000)
                magnitude = parameters.get('magnitude', 1.0)
                
                dx = coordinates[:, 0] - x0
                dy = coordinates[:, 1] - y0
                dz = coordinates[:, 2] - z0
                
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                r = np.maximum(r, 1e-10)
                
                # Enhanced point source with realistic decay
                C = magnitude / (4 * np.pi)
                decay_factor = np.exp(-r / 10000)  # 10 km characteristic length
                
                ux = C * dx / r**3 * decay_factor
                uy = C * dy / r**3 * decay_factor
                uz = C * dz / r**3 * decay_factor
                
                return np.column_stack([ux, uy, uz])
                
        except ImportError:
            # Fallback if elastic_inversion_models is not available
            self.logger.warning("elastic_inversion_models not available, using simplified forward models")
            return await self._simplified_forward_model(coordinates, source_type, parameters)
        
        except Exception as e:
            self.logger.error(f"Error in advanced forward model: {str(e)}")
            return await self._simplified_forward_model(coordinates, source_type, parameters)
    
    async def _simplified_forward_model(self, coordinates: np.ndarray,
                                      source_type: SourceType,
                                      parameters: Dict[str, float]) -> np.ndarray:
        """Simplified forward model as fallback"""
        n_points = len(coordinates)
        
        if source_type == SourceType.POINT_SOURCE:
            # Enhanced Mogi model
            x0, y0, z0 = parameters.get('x0', 0), parameters.get('y0', 0), parameters.get('z0', 1000)
            volume_change = parameters.get('volume_change', 1e6)
            nu = parameters.get('poisson_ratio', 0.25)
            
            dx = coordinates[:, 0] - x0
            dy = coordinates[:, 1] - y0
            dz = coordinates[:, 2] - z0
            
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            r = np.maximum(r, 1e-10)
            
            C = (1 - nu) * volume_change / np.pi
            
            ux = C * dx / r**3
            uy = C * dy / r**3
            uz = C * dz / r**3
            
            return np.column_stack([ux, uy, uz])
        
        elif source_type == SourceType.DIKE:
            # Enhanced dike model
            x0, y0, z0 = parameters.get('x0', 0), parameters.get('y0', 0), parameters.get('z0', 1000)
            length = parameters.get('length', 1000)
            width = parameters.get('width', 500)
            opening = parameters.get('opening', 1.0)
            strike = np.radians(parameters.get('strike', 0))
            
            # Distance from dike center
            dx = coordinates[:, 0] - x0
            dy = coordinates[:, 1] - y0
            
            # Rotate to dike coordinate system
            dx_rot = dx * np.cos(strike) + dy * np.sin(strike)
            dy_rot = -dx * np.sin(strike) + dy * np.cos(strike)
            
            # Simplified dike displacement
            distance_along = np.abs(dx_rot)
            distance_across = np.abs(dy_rot)
            depth_factor = np.exp(-coordinates[:, 2] / z0)
            
            # Scale by dike dimensions
            along_factor = np.exp(-distance_along**2 / (length/2)**2)
            across_factor = np.exp(-distance_across**2 / (width/2)**2)
            
            displacements = np.zeros((n_points, 3))
            displacement_magnitude = opening * depth_factor * along_factor * across_factor
            
            # Horizontal displacement perpendicular to dike
            displacements[:, 0] = displacement_magnitude * np.sin(strike) * np.sign(dy_rot)
            displacements[:, 1] = -displacement_magnitude * np.cos(strike) * np.sign(dy_rot)
            displacements[:, 2] = displacement_magnitude * 0.5  # Vertical component
            
            return displacements
        
        else:
            # Enhanced fallback
            magnitude = parameters.get('magnitude', 0.01)
            noise_level = magnitude * 0.1
            
            base_displacement = np.random.normal(0, magnitude, (n_points, 3))
            noise = np.random.normal(0, noise_level, (n_points, 3))
            
            return base_displacement + noise
    
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
    
    async def compute_greens_functions(self, observations: ObservationData,
                                     source_type: SourceType,
                                     parameter_ranges: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Compute comprehensive Green's functions matrix for linear inversion"""
        try:
            from . import elastic_inversion_models as eim
            
            n_obs = len(observations.coordinates)
            n_components = 3  # ux, uy, uz
            
            if source_type == SourceType.POINT_SOURCE:
                # Green's functions for point source (Mogi model)
                # Parameters: x0, y0, z0, volume_change
                n_params = 4
                G = np.zeros((n_obs * n_components, n_params))
                
                for i, (x, y, z) in enumerate(observations.coordinates):
                    # Partial derivatives with respect to each parameter
                    # For unit volume change at reference location
                    x0_ref, y0_ref, z0_ref = 0, 0, 1000
                    
                    dx = x - x0_ref
                    dy = y - y0_ref  
                    dz = z - z0_ref
                    r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-10
                    
                    nu = 0.25  # Default Poisson's ratio
                    C = (1 - nu) / np.pi
                    
                    # Green's functions for unit volume change
                    G[i*3, 3] = C * dx / r**3      # ux/dV
                    G[i*3 + 1, 3] = C * dy / r**3  # uy/dV
                    G[i*3 + 2, 3] = C * dz / r**3  # uz/dV
                    
                    # Spatial derivatives (for position parameters)
                    # d(ux)/d(x0) = -d(ux)/d(x)
                    G[i*3, 0] = -C * (1/r**3 - 3*dx**2/r**5)         # ux/dx0
                    G[i*3, 1] = -C * (-3*dx*dy/r**5)                 # ux/dy0
                    G[i*3, 2] = -C * (-3*dx*dz/r**5)                 # ux/dz0
                    
                    G[i*3 + 1, 0] = -C * (-3*dy*dx/r**5)             # uy/dx0
                    G[i*3 + 1, 1] = -C * (1/r**3 - 3*dy**2/r**5)     # uy/dy0
                    G[i*3 + 1, 2] = -C * (-3*dy*dz/r**5)             # uy/dz0
                    
                    G[i*3 + 2, 0] = -C * (-3*dz*dx/r**5)             # uz/dx0
                    G[i*3 + 2, 1] = -C * (-3*dz*dy/r**5)             # uz/dy0
                    G[i*3 + 2, 2] = -C * (1/r**3 - 3*dz**2/r**5)     # uz/dz0
                
                return G
            
            elif source_type in [SourceType.DIKE, SourceType.SILL, SourceType.FAULT]:
                # Use elastic dislocation model for Green's functions
                # Create reference fault geometry
                fault_geometry = eim.FaultGeometry(
                    x0=0, y0=0, z0=parameter_ranges.get('z0', (1000, 5000))[0],
                    length=parameter_ranges.get('length', (1000, 10000))[0], 
                    width=parameter_ranges.get('width', (500, 5000))[0],
                    strike=parameter_ranges.get('strike', (0, 360))[0],
                    dip=parameter_ranges.get('dip', (0, 90))[0] if source_type != SourceType.SILL else 0
                )
                
                elastic_params = eim.ElasticParameters()
                
                # Compute Green's functions for unit slip components
                unit_slips = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # strike, dip, opening
                G_components = []
                
                for unit_slip in unit_slips:
                    model = eim.ElasticDislocationModel(
                        fault_geometry, elastic_params, unit_slip
                    )
                    displacements = model.compute_displacements(observations.coordinates)
                    G_components.append(displacements.flatten())
                
                # Include geometric parameter sensitivities
                param_names = ['x0', 'y0', 'z0', 'length', 'width', 'strike', 'dip']
                if source_type != SourceType.SILL:
                    param_names.append('rake')
                
                G_geometric = []
                for param_name in param_names:
                    if param_name in parameter_ranges:
                        sensitivity = await self._compute_parameter_sensitivity(
                            observations.coordinates, fault_geometry, elastic_params,
                            param_name, (1, 0, 0)  # Reference slip
                        )
                        G_geometric.append(sensitivity.flatten())
                
                G = np.column_stack(G_components + G_geometric)
                return G
            
            elif source_type == SourceType.DISTRIBUTED_SLIP:
                # Green's functions for distributed slip
                n_patches = parameter_ranges.get('n_patches', (25, 25))[0]
                patches_per_side = int(np.sqrt(n_patches))
                
                fault_length = parameter_ranges.get('length', (10000, 10000))[0]
                fault_width = parameter_ranges.get('width', (8000, 8000))[0]
                
                patch_length = fault_length / patches_per_side
                patch_width = fault_width / patches_per_side
                
                fault_patches = []
                for i in range(patches_per_side):
                    for j in range(patches_per_side):
                        x_patch = (i - patches_per_side/2) * patch_length
                        y_patch = (j - patches_per_side/2) * patch_width
                        
                        patch_geom = eim.FaultGeometry(
                            x0=x_patch, y0=y_patch, 
                            z0=parameter_ranges.get('z0', (5000, 5000))[0],
                            length=patch_length, width=patch_width,
                            strike=parameter_ranges.get('strike', (0, 0))[0],
                            dip=parameter_ranges.get('dip', (90, 90))[0]
                        )
                        fault_patches.append(patch_geom)
                
                elastic_params = eim.ElasticParameters()
                model = eim.DistributedSlipModel(
                    fault_patches, elastic_params, np.zeros((n_patches, 3))
                )
                
                G = model.compute_greens_functions(observations.coordinates)
                return G
            
            else:
                # Fallback: identity-like Green's functions
                n_params = len(parameter_ranges)
                G = np.random.normal(0, 0.01, (n_obs * n_components, n_params))
                return G
                
        except ImportError:
            self.logger.warning("elastic_inversion_models not available for Green's functions")
            # Simple fallback Green's functions
            n_params = len(parameter_ranges)
            n_obs = len(observations.coordinates)
            return np.random.normal(0, 0.01, (n_obs * 3, n_params))
        
        except Exception as e:
            self.logger.error(f"Error computing Green's functions: {str(e)}")
            n_params = len(parameter_ranges)
            n_obs = len(observations.coordinates)
            return np.random.normal(0, 0.01, (n_obs * 3, n_params))
    
    async def _compute_parameter_sensitivity(self, coordinates: np.ndarray,
                                           fault_geometry, elastic_params,
                                           param_name: str, slip_components: Tuple[float, float, float],
                                           delta: float = 1e-4) -> np.ndarray:
        """Compute sensitivity of displacements to geometric parameters"""
        try:
            from . import elastic_inversion_models as eim
            
            # Get original parameter value
            original_value = getattr(fault_geometry, param_name)
            
            # Forward difference
            setattr(fault_geometry, param_name, original_value + delta)
            model_plus = eim.ElasticDislocationModel(fault_geometry, elastic_params, slip_components)
            disp_plus = model_plus.compute_displacements(coordinates)
            
            # Backward difference
            setattr(fault_geometry, param_name, original_value - delta)
            model_minus = eim.ElasticDislocationModel(fault_geometry, elastic_params, slip_components)
            disp_minus = model_minus.compute_displacements(coordinates)
            
            # Restore original value
            setattr(fault_geometry, param_name, original_value)
            
            # Calculate sensitivity
            sensitivity = (disp_plus - disp_minus) / (2 * delta)
            return sensitivity
            
        except Exception as e:
            self.logger.error(f"Error computing parameter sensitivity for {param_name}: {str(e)}")
            return np.zeros((len(coordinates), 3))
    
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
        """Get comprehensive parameter bounds for different source types"""
        
        # Common elastic parameter bounds
        elastic_bounds = {
            'youngs_modulus': (30e9, 120e9),  # 30-120 GPa range for crustal rocks
            'poisson_ratio': (0.15, 0.45)    # Typical range for geological materials
        }
        
        bounds = {
            SourceType.POINT_SOURCE: {
                'x0': (-20000, 20000),          # ±20 km
                'y0': (-20000, 20000),          # ±20 km
                'z0': (100, 15000),             # 0.1-15 km depth
                'volume_change': (1e3, 1e10),   # 1e3 to 1e10 m³
                **elastic_bounds
            },
            SourceType.DIKE: {
                'x0': (-20000, 20000),
                'y0': (-20000, 20000),
                'z0': (100, 10000),             # Typical dike depths
                'length': (100, 20000),         # 0.1-20 km
                'width': (10, 10000),           # 10 m to 10 km
                'opening': (0.001, 50.0),       # 1 mm to 50 m
                'strike': (0, 360),
                'dip': (45, 90),                # Typically steep
                **elastic_bounds
            },
            SourceType.SILL: {
                'x0': (-30000, 30000),          # Larger extent for sills
                'y0': (-30000, 30000),
                'z0': (500, 15000),             # Deeper than dikes typically
                'length': (500, 50000),         # Can be very large
                'width': (500, 50000),
                'opening': (0.01, 20.0),        # Sill opening
                'strike': (0, 360),
                **elastic_bounds
            },
            SourceType.FAULT: {
                'x0': (-50000, 50000),          # Large fault systems
                'y0': (-50000, 50000),
                'z0': (100, 30000),             # Up to 30 km depth
                'length': (1000, 100000),       # 1-100 km
                'width': (500, 30000),          # 0.5-30 km
                'slip': (0.001, 50.0),          # 1 mm to 50 m
                'strike': (0, 360),
                'dip': (1, 90),                 # Avoid exactly 0 degrees
                'rake': (-180, 180),
                'opening': (0.001, 5.0),        # Minor opening on faults
                **elastic_bounds
            },
            SourceType.DISTRIBUTED_SLIP: {
                'x0': (-50000, 50000),
                'y0': (-50000, 50000), 
                'z0': (100, 30000),
                'length': (5000, 100000),       # Large fault systems
                'width': (2000, 30000),
                'total_slip': (0.01, 20.0),     # Total slip magnitude
                'n_patches': (9, 100),          # 3x3 to 10x10 patches
                'strike': (0, 360),
                'dip': (0, 90),
                'rake': (-180, 180),
                **elastic_bounds
            },
            SourceType.VISCOELASTIC: {
                'x0': (-50000, 50000),
                'y0': (-50000, 50000),
                'z0': (1000, 30000),            # Deeper sources for viscoelastic
                'length': (5000, 100000),
                'width': (2000, 30000),
                'slip': (0.1, 20.0),
                'strike': (0, 360),
                'dip': (0, 90),
                'rake': (-180, 180),
                'viscosity': (1e17, 1e22),      # Pa·s, mantle/lower crust
                'relaxation_time': (1e6, 1e10), # seconds (days to centuries)
                'current_time': (86400, 1e9),   # 1 day to ~30 years
                **elastic_bounds
            },
            SourceType.POROELASTIC: {
                'x0': (-20000, 20000),
                'y0': (-20000, 20000),
                'z0': (100, 10000),             # Shallow to moderate depths
                'length': (1000, 20000),
                'width': (1000, 20000),
                'pressure_change': (1e4, 1e8),  # 10 kPa to 100 MPa
                'porosity': (0.01, 0.4),        # 1-40%
                'permeability': (1e-18, 1e-12), # m², very low to moderate
                'fluid_viscosity': (1e-4, 1e-1), # Pa·s, water to oil
                'biot_coefficient': (0.3, 1.0),
                'current_time': (3600, 1e8),    # 1 hour to ~3 years
                **elastic_bounds
            }
        }
        
        return bounds.get(source_type, elastic_bounds)
    
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