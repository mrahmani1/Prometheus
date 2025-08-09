"""
Adaptive Model Selection System for Geological Simulations
Intelligent model selection based on data availability, complexity, and computational constraints
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Protocol
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from abc import ABC, abstractmethod
import networkx as nx
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import scipy.stats as stats
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns

# Import our complexity assessment system
try:
    from .geological_complexity_assessment import (
        GeologicalComplexityAssessmentSystem, 
        ComplexityMetrics, 
        SimulationData,
        ComplexityType
    )
    COMPLEXITY_ASSESSMENT_AVAILABLE = True
except ImportError:
    COMPLEXITY_ASSESSMENT_AVAILABLE = False
    warnings.warn("Complexity assessment system not available. Model selection will be limited.")

# Advanced ML imports with fallbacks
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingRegressor
    HIST_GB_AVAILABLE = True
except ImportError:
    HIST_GB_AVAILABLE = False

warnings.filterwarnings('ignore')

class ModelType(Enum):
    """Available model types for geological simulation"""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    SVM = "svm"
    GAUSSIAN_PROCESS = "gaussian_process"
    NEURAL_NETWORK = "neural_network"
    DEEP_NEURAL_NETWORK = "deep_neural_network"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    PHYSICS_INFORMED_NN = "physics_informed_nn"
    ENSEMBLE = "ensemble"
    MULTI_PHYSICS = "multi_physics"
    ADAPTIVE_MESH = "adaptive_mesh"
    SPECTRAL_METHODS = "spectral_methods"
    FINITE_ELEMENT = "finite_element"
    FINITE_DIFFERENCE = "finite_difference"
    MONTE_CARLO = "monte_carlo"
    HYBRID = "hybrid"

class DataAvailability(Enum):
    """Data availability levels"""
    MINIMAL = auto()    # < 100 samples
    LIMITED = auto()    # 100-1000 samples
    MODERATE = auto()   # 1000-10000 samples
    ABUNDANT = auto()   # 10000-100000 samples
    MASSIVE = auto()    # > 100000 samples

class ComputationalConstraint(Enum):
    """Computational constraint levels"""
    VERY_LOW = auto()   # Single CPU core, < 1 GB RAM
    LOW = auto()        # Few CPU cores, < 8 GB RAM
    MODERATE = auto()   # Multi-core CPU, 8-32 GB RAM
    HIGH = auto()       # High-end CPU/GPU, 32-128 GB RAM
    UNLIMITED = auto()  # HPC cluster, > 128 GB RAM

@dataclass
class ModelCharacteristics:
    """Characteristics of a model type"""
    model_type: ModelType
    complexity_tolerance: Tuple[float, float]  # (min, max) complexity scores
    data_requirements: Tuple[int, int]  # (min_samples, optimal_samples)
    computational_cost: float  # Relative computational cost (1.0 = baseline)
    memory_cost: float  # Relative memory cost (1.0 = baseline)
    interpretability: float  # 0.0 (black box) to 1.0 (fully interpretable)
    accuracy_potential: float  # Expected accuracy potential (0.0 to 1.0)
    handles_nonlinearity: bool
    handles_high_dimensions: bool
    handles_temporal: bool
    handles_spatial: bool
    handles_uncertainty: bool
    training_stability: float  # Training stability (0.0 to 1.0)
    hyperparameter_sensitivity: float  # 0.0 (insensitive) to 1.0 (very sensitive)

@dataclass
class ModelRecommendation:
    """Model recommendation with justification"""
    model_type: ModelType
    confidence: float
    expected_performance: float
    computational_feasibility: float
    justification: str
    hyperparameters: Dict[str, Any]
    preprocessing_steps: List[str]
    alternative_models: List[ModelType]
    risk_factors: List[str]
    estimated_training_time: float
    estimated_memory_usage: float

@dataclass
class DataCharacteristics:
    """Characteristics of available data"""
    sample_count: int
    feature_count: int
    temporal_length: Optional[int]
    spatial_dimensions: Optional[Tuple[int, ...]]
    missing_data_ratio: float
    noise_level: float
    data_availability: DataAvailability
    complexity_metrics: Optional[ComplexityMetrics]
    data_types: List[str]  # ['spatial', 'temporal', 'structural', etc.]
    correlation_structure: Optional[np.ndarray]
    outlier_ratio: float
    class_imbalance: Optional[float]

class ModelSelector(Protocol):
    """Protocol for model selectors"""
    
    def select_model(self, 
                    data_chars: DataCharacteristics,
                    comp_constraints: ComputationalConstraint,
                    objectives: List[str]) -> ModelRecommendation:
        """Select optimal model based on data and constraints"""
        ...

class GeologicalModelDatabase:
    """Database of geological model characteristics"""
    
    def __init__(self):
        self.models = self._initialize_model_database()
    
    def _initialize_model_database(self) -> Dict[ModelType, ModelCharacteristics]:
        """Initialize database of model characteristics"""
        return {
            ModelType.LINEAR: ModelCharacteristics(
                model_type=ModelType.LINEAR,
                complexity_tolerance=(0.0, 0.3),
                data_requirements=(50, 500),
                computational_cost=1.0,
                memory_cost=1.0,
                interpretability=1.0,
                accuracy_potential=0.6,
                handles_nonlinearity=False,
                handles_high_dimensions=False,
                handles_temporal=False,
                handles_spatial=False,
                handles_uncertainty=False,
                training_stability=0.9,
                hyperparameter_sensitivity=0.2
            ),
            
            ModelType.POLYNOMIAL: ModelCharacteristics(
                model_type=ModelType.POLYNOMIAL,
                complexity_tolerance=(0.2, 0.6),
                data_requirements=(100, 1000),
                computational_cost=1.5,
                memory_cost=1.2,
                interpretability=0.8,
                accuracy_potential=0.7,
                handles_nonlinearity=True,
                handles_high_dimensions=False,
                handles_temporal=False,
                handles_spatial=False,
                handles_uncertainty=False,
                training_stability=0.7,
                hyperparameter_sensitivity=0.4
            ),
            
            ModelType.RANDOM_FOREST: ModelCharacteristics(
                model_type=ModelType.RANDOM_FOREST,
                complexity_tolerance=(0.3, 0.8),
                data_requirements=(200, 5000),
                computational_cost=2.0,
                memory_cost=1.5,
                interpretability=0.6,
                accuracy_potential=0.8,
                handles_nonlinearity=True,
                handles_high_dimensions=True,
                handles_temporal=False,
                handles_spatial=True,
                handles_uncertainty=True,
                training_stability=0.8,
                hyperparameter_sensitivity=0.3
            ),
            
            ModelType.GRADIENT_BOOSTING: ModelCharacteristics(
                model_type=ModelType.GRADIENT_BOOSTING,
                complexity_tolerance=(0.4, 0.9),
                data_requirements=(500, 10000),
                computational_cost=3.0,
                memory_cost=2.0,
                interpretability=0.4,
                accuracy_potential=0.85,
                handles_nonlinearity=True,
                handles_high_dimensions=True,
                handles_temporal=False,
                handles_spatial=True,
                handles_uncertainty=True,
                training_stability=0.7,
                hyperparameter_sensitivity=0.6
            ),
            
            ModelType.XGBOOST: ModelCharacteristics(
                model_type=ModelType.XGBOOST,
                complexity_tolerance=(0.5, 1.0),
                data_requirements=(1000, 50000),
                computational_cost=4.0,
                memory_cost=2.5,
                interpretability=0.3,
                accuracy_potential=0.9,
                handles_nonlinearity=True,
                handles_high_dimensions=True,
                handles_temporal=False,
                handles_spatial=True,
                handles_uncertainty=True,
                training_stability=0.6,
                hyperparameter_sensitivity=0.7
            ),
            
            ModelType.NEURAL_NETWORK: ModelCharacteristics(
                model_type=ModelType.NEURAL_NETWORK,
                complexity_tolerance=(0.6, 1.0),
                data_requirements=(2000, 20000),
                computational_cost=5.0,
                memory_cost=3.0,
                interpretability=0.2,
                accuracy_potential=0.85,
                handles_nonlinearity=True,
                handles_high_dimensions=True,
                handles_temporal=True,
                handles_spatial=True,
                handles_uncertainty=False,
                training_stability=0.5,
                hyperparameter_sensitivity=0.8
            ),
            
            ModelType.DEEP_NEURAL_NETWORK: ModelCharacteristics(
                model_type=ModelType.DEEP_NEURAL_NETWORK,
                complexity_tolerance=(0.7, 1.0),
                data_requirements=(10000, 100000),
                computational_cost=10.0,
                memory_cost=8.0,
                interpretability=0.1,
                accuracy_potential=0.95,
                handles_nonlinearity=True,
                handles_high_dimensions=True,
                handles_temporal=True,
                handles_spatial=True,
                handles_uncertainty=False,
                training_stability=0.4,
                hyperparameter_sensitivity=0.9
            ),
            
            ModelType.GRAPH_NEURAL_NETWORK: ModelCharacteristics(
                model_type=ModelType.GRAPH_NEURAL_NETWORK,
                complexity_tolerance=(0.8, 1.0),
                data_requirements=(5000, 50000),
                computational_cost=12.0,
                memory_cost=10.0,
                interpretability=0.15,
                accuracy_potential=0.92,
                handles_nonlinearity=True,
                handles_high_dimensions=True,
                handles_temporal=True,
                handles_spatial=True,
                handles_uncertainty=True,
                training_stability=0.3,
                hyperparameter_sensitivity=0.85
            ),
            
            ModelType.PHYSICS_INFORMED_NN: ModelCharacteristics(
                model_type=ModelType.PHYSICS_INFORMED_NN,
                complexity_tolerance=(0.5, 1.0),
                data_requirements=(1000, 10000),
                computational_cost=15.0,
                memory_cost=12.0,
                interpretability=0.7,
                accuracy_potential=0.95,
                handles_nonlinearity=True,
                handles_high_dimensions=True,
                handles_temporal=True,
                handles_spatial=True,
                handles_uncertainty=True,
                training_stability=0.6,
                hyperparameter_sensitivity=0.7
            ),
            
            ModelType.GAUSSIAN_PROCESS: ModelCharacteristics(
                model_type=ModelType.GAUSSIAN_PROCESS,
                complexity_tolerance=(0.3, 0.7),
                data_requirements=(100, 5000),
                computational_cost=6.0,
                memory_cost=4.0,
                interpretability=0.5,
                accuracy_potential=0.8,
                handles_nonlinearity=True,
                handles_high_dimensions=False,
                handles_temporal=True,
                handles_spatial=True,
                handles_uncertainty=True,
                training_stability=0.8,
                hyperparameter_sensitivity=0.5
            ),
            
            ModelType.SVM: ModelCharacteristics(
                model_type=ModelType.SVM,
                complexity_tolerance=(0.4, 0.8),
                data_requirements=(500, 10000),
                computational_cost=3.0,
                memory_cost=2.0,
                interpretability=0.3,
                accuracy_potential=0.75,
                handles_nonlinearity=True,
                handles_high_dimensions=True,
                handles_temporal=False,
                handles_spatial=False,
                handles_uncertainty=False,
                training_stability=0.7,
                hyperparameter_sensitivity=0.6
            ),
            
            ModelType.ENSEMBLE: ModelCharacteristics(
                model_type=ModelType.ENSEMBLE,
                complexity_tolerance=(0.5, 1.0),
                data_requirements=(2000, 20000),
                computational_cost=8.0,
                memory_cost=6.0,
                interpretability=0.4,
                accuracy_potential=0.92,
                handles_nonlinearity=True,
                handles_high_dimensions=True,
                handles_temporal=True,
                handles_spatial=True,
                handles_uncertainty=True,
                training_stability=0.8,
                hyperparameter_sensitivity=0.5
            )
        }
    
    def get_model_characteristics(self, model_type: ModelType) -> ModelCharacteristics:
        """Get characteristics for a specific model type"""
        return self.models.get(model_type, self.models[ModelType.LINEAR])
    
    def get_suitable_models(self, 
                           data_chars: DataCharacteristics,
                           comp_constraints: ComputationalConstraint) -> List[ModelType]:
        """Get list of suitable models for given data and constraints"""
        suitable_models = []
        
        complexity_score = (data_chars.complexity_metrics.overall_complexity 
                          if data_chars.complexity_metrics else 0.5)
        
        for model_type, characteristics in self.models.items():
            # Check data requirements
            if data_chars.sample_count < characteristics.data_requirements[0]:
                continue
            
            # Check complexity tolerance
            if not (characteristics.complexity_tolerance[0] <= complexity_score <= 
                   characteristics.complexity_tolerance[1]):
                continue
            
            # Check computational constraints
            if not self._check_computational_feasibility(characteristics, comp_constraints):
                continue
            
            # Check data type compatibility
            if not self._check_data_compatibility(characteristics, data_chars):
                continue
            
            suitable_models.append(model_type)
        
        return suitable_models
    
    def _check_computational_feasibility(self, 
                                       characteristics: ModelCharacteristics,
                                       constraints: ComputationalConstraint) -> bool:
        """Check if model is computationally feasible"""
        constraint_limits = {
            ComputationalConstraint.VERY_LOW: 2.0,
            ComputationalConstraint.LOW: 5.0,
            ComputationalConstraint.MODERATE: 10.0,
            ComputationalConstraint.HIGH: 20.0,
            ComputationalConstraint.UNLIMITED: float('inf')
        }
        
        return characteristics.computational_cost <= constraint_limits[constraints]
    
    def _check_data_compatibility(self, 
                                characteristics: ModelCharacteristics,
                                data_chars: DataCharacteristics) -> bool:
        """Check if model is compatible with available data types"""
        # Check spatial data compatibility
        if 'spatial' in data_chars.data_types and not characteristics.handles_spatial:
            return False
        
        # Check temporal data compatibility
        if 'temporal' in data_chars.data_types and not characteristics.handles_temporal:
            return False
        
        # Check high-dimensional data compatibility
        if data_chars.feature_count > 1000 and not characteristics.handles_high_dimensions:
            return False
        
        return True

class AdaptiveModelSelector:
    """Adaptive model selection system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_db = GeologicalModelDatabase()
        self.complexity_assessor = (GeologicalComplexityAssessmentSystem() 
                                  if COMPLEXITY_ASSESSMENT_AVAILABLE else None)
        
        # Selection history for adaptive learning
        self.selection_history = []
        self.performance_history = {}
        
        # Model builders
        self.model_builders = self._initialize_model_builders()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'optimization_budget': 100,  # Optuna trials
            'cross_validation_folds': 5,
            'performance_weight': 0.4,
            'efficiency_weight': 0.3,
            'interpretability_weight': 0.2,
            'stability_weight': 0.1,
            'enable_ensemble': True,
            'enable_hyperparameter_optimization': True,
            'max_evaluation_time': 3600,  # seconds
            'parallel_evaluation': True,
            'risk_tolerance': 0.2,
            'performance_threshold': 0.8
        }
    
    def _initialize_model_builders(self) -> Dict[ModelType, Callable]:
        """Initialize model builders for different model types"""
        return {
            ModelType.LINEAR: self._build_linear_model,
            ModelType.POLYNOMIAL: self._build_polynomial_model,
            ModelType.RANDOM_FOREST: self._build_random_forest_model,
            ModelType.GRADIENT_BOOSTING: self._build_gradient_boosting_model,
            ModelType.XGBOOST: self._build_xgboost_model,
            ModelType.NEURAL_NETWORK: self._build_neural_network_model,
            ModelType.GAUSSIAN_PROCESS: self._build_gaussian_process_model,
            ModelType.SVM: self._build_svm_model,
            ModelType.ENSEMBLE: self._build_ensemble_model
        }
    
    async def select_optimal_model(self, 
                                 simulation_data: SimulationData,
                                 target_data: np.ndarray,
                                 computational_constraints: ComputationalConstraint = ComputationalConstraint.MODERATE,
                                 objectives: List[str] = None) -> ModelRecommendation:
        """Select optimal model for given data and constraints"""
        try:
            objectives = objectives or ['accuracy', 'efficiency']
            
            # Analyze data characteristics
            self.logger.info("Analyzing data characteristics...")
            data_chars = await self._analyze_data_characteristics(simulation_data, target_data)
            
            # Get candidate models
            self.logger.info("Identifying candidate models...")
            candidate_models = self.model_db.get_suitable_models(data_chars, computational_constraints)
            
            if not candidate_models:
                self.logger.warning("No suitable models found, falling back to linear model")
                candidate_models = [ModelType.LINEAR]
            
            # Evaluate candidate models
            self.logger.info(f"Evaluating {len(candidate_models)} candidate models...")
            evaluations = await self._evaluate_candidate_models(
                candidate_models, simulation_data, target_data, data_chars, computational_constraints
            )
            
            # Select optimal model based on multi-criteria optimization
            self.logger.info("Selecting optimal model...")
            recommendation = self._select_optimal_model_from_evaluations(
                evaluations, objectives, computational_constraints
            )
            
            # Update selection history
            self._update_selection_history(data_chars, recommendation)
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error in model selection: {e}")
            return self._get_fallback_recommendation()
    
    async def _analyze_data_characteristics(self, 
                                          simulation_data: SimulationData,
                                          target_data: np.ndarray) -> DataCharacteristics:
        """Analyze characteristics of available data"""
        try:
            # Basic statistics
            sample_count = len(target_data)
            all_arrays = simulation_data.get_all_data_arrays()
            feature_count = sum(arr.size // sample_count for arr in all_arrays if arr.size >= sample_count)
            
            # Determine data availability level
            if sample_count < 100:
                data_availability = DataAvailability.MINIMAL
            elif sample_count < 1000:
                data_availability = DataAvailability.LIMITED
            elif sample_count < 10000:
                data_availability = DataAvailability.MODERATE
            elif sample_count < 100000:
                data_availability = DataAvailability.ABUNDANT
            else:
                data_availability = DataAvailability.MASSIVE
            
            # Analyze data quality
            missing_ratio = self._calculate_missing_data_ratio(simulation_data)
            noise_level = self._estimate_noise_level(target_data)
            outlier_ratio = self._calculate_outlier_ratio(target_data)
            
            # Determine data types
            data_types = []
            if simulation_data.spatial_data is not None:
                data_types.append('spatial')
            if simulation_data.temporal_data is not None:
                data_types.append('temporal')
            if simulation_data.structural_data is not None:
                data_types.append('structural')
            
            # Get complexity assessment if available
            complexity_metrics = None
            if self.complexity_assessor:
                complexity_metrics = await self.complexity_assessor.assess_simulation_complexity(simulation_data)
            
            # Spatial dimensions
            spatial_dimensions = (simulation_data.spatial_data.shape 
                                if simulation_data.spatial_data is not None else None)
            
            # Temporal length
            temporal_length = (simulation_data.temporal_data.shape[0] 
                             if simulation_data.temporal_data is not None else None)
            
            return DataCharacteristics(
                sample_count=sample_count,
                feature_count=feature_count,
                temporal_length=temporal_length,
                spatial_dimensions=spatial_dimensions,
                missing_data_ratio=missing_ratio,
                noise_level=noise_level,
                data_availability=data_availability,
                complexity_metrics=complexity_metrics,
                data_types=data_types,
                correlation_structure=None,  # Could be computed if needed
                outlier_ratio=outlier_ratio,
                class_imbalance=None  # For regression tasks
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing data characteristics: {e}")
            # Return minimal characteristics
            return DataCharacteristics(
                sample_count=len(target_data),
                feature_count=100,
                temporal_length=None,
                spatial_dimensions=None,
                missing_data_ratio=0.0,
                noise_level=0.1,
                data_availability=DataAvailability.LIMITED,
                complexity_metrics=None,
                data_types=['spatial'],
                correlation_structure=None,
                outlier_ratio=0.05,
                class_imbalance=None
            )
    
    def _calculate_missing_data_ratio(self, simulation_data: SimulationData) -> float:
        """Calculate ratio of missing data"""
        try:
            total_elements = 0
            missing_elements = 0
            
            for array in simulation_data.get_all_data_arrays():
                total_elements += array.size
                missing_elements += np.sum(np.isnan(array)) + np.sum(np.isinf(array))
            
            return missing_elements / total_elements if total_elements > 0 else 0.0
        except Exception:
            return 0.0
    
    def _estimate_noise_level(self, target_data: np.ndarray) -> float:
        """Estimate noise level in target data"""
        try:
            # Use detrended fluctuation analysis or simple approach
            if len(target_data) < 10:
                return 0.1
            
            # Simple approach: ratio of high-frequency to total variance
            if len(target_data) > 50:
                # Apply smoothing filter
                from scipy import ndimage
                smoothed = ndimage.gaussian_filter1d(target_data, sigma=2)
                noise = target_data - smoothed
                
                noise_std = np.std(noise)
                signal_std = np.std(target_data)
                
                return noise_std / signal_std if signal_std > 0 else 0.1
            else:
                # For small datasets, estimate from local variations
                diffs = np.diff(target_data)
                return np.std(diffs) / (np.std(target_data) * np.sqrt(2)) if np.std(target_data) > 0 else 0.1
                
        except Exception:
            return 0.1
    
    def _calculate_outlier_ratio(self, target_data: np.ndarray) -> float:
        """Calculate ratio of outliers in target data"""
        try:
            if len(target_data) < 10:
                return 0.0
            
            # Use IQR method
            q1, q3 = np.percentile(target_data, [25, 75])
            iqr = q3 - q1
            
            if iqr == 0:
                return 0.0
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = np.sum((target_data < lower_bound) | (target_data > upper_bound))
            
            return outliers / len(target_data)
            
        except Exception:
            return 0.0
    
    async def _evaluate_candidate_models(self, 
                                       candidate_models: List[ModelType],
                                       simulation_data: SimulationData,
                                       target_data: np.ndarray,
                                       data_chars: DataCharacteristics,
                                       comp_constraints: ComputationalConstraint) -> Dict[ModelType, Dict[str, float]]:
        """Evaluate candidate models"""
        evaluations = {}
        
        # Prepare data for evaluation
        X, y = self._prepare_data_for_evaluation(simulation_data, target_data)
        
        if self.config['parallel_evaluation']:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=min(len(candidate_models), mp.cpu_count())) as executor:
                future_to_model = {
                    executor.submit(self._evaluate_single_model, model_type, X, y, data_chars): model_type
                    for model_type in candidate_models
                }
                
                for future in future_to_model:
                    model_type = future_to_model[future]
                    try:
                        evaluation = future.result(timeout=self.config['max_evaluation_time'] // len(candidate_models))
                        evaluations[model_type] = evaluation
                    except Exception as e:
                        self.logger.error(f"Error evaluating {model_type}: {e}")
                        evaluations[model_type] = self._get_default_evaluation()
        else:
            # Sequential evaluation
            for model_type in candidate_models:
                try:
                    evaluation = self._evaluate_single_model(model_type, X, y, data_chars)
                    evaluations[model_type] = evaluation
                except Exception as e:
                    self.logger.error(f"Error evaluating {model_type}: {e}")
                    evaluations[model_type] = self._get_default_evaluation()
        
        return evaluations
    
    def _prepare_data_for_evaluation(self, 
                                   simulation_data: SimulationData,
                                   target_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model evaluation"""
        try:
            # Flatten and combine all input data
            feature_arrays = []
            
            if simulation_data.spatial_data is not None:
                spatial_flat = simulation_data.spatial_data.reshape(len(target_data), -1)
                feature_arrays.append(spatial_flat)
            
            if simulation_data.temporal_data is not None:
                if simulation_data.temporal_data.shape[0] == len(target_data):
                    temporal_flat = simulation_data.temporal_data.reshape(len(target_data), -1)
                    feature_arrays.append(temporal_flat)
            
            # Add other data types as needed
            
            if feature_arrays:
                X = np.hstack(feature_arrays)
            else:
                # Create dummy features if no input data
                X = np.random.rand(len(target_data), 10)
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Limit feature count for evaluation efficiency
            if X.shape[1] > 1000:
                # Use PCA or feature selection to reduce dimensionality
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(500, X.shape[0] // 2))
                X = pca.fit_transform(X)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            # Return dummy data
            return np.random.rand(len(target_data), 10), target_data
    
    def _evaluate_single_model(self, 
                             model_type: ModelType,
                             X: np.ndarray,
                             y: np.ndarray,
                             data_chars: DataCharacteristics) -> Dict[str, float]:
        """Evaluate a single model type"""
        try:
            start_time = time.time()
            
            # Build model
            model = self._build_model(model_type, X, y)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X, y, 
                cv=min(self.config['cross_validation_folds'], len(y) // 10),
                scoring='neg_mean_squared_error',
                n_jobs=1  # Avoid nested parallelism
            )
            
            training_time = time.time() - start_time
            
            # Calculate metrics
            performance = -np.mean(cv_scores)  # MSE (lower is better)
            performance_std = np.std(cv_scores)
            stability = 1.0 / (1.0 + performance_std)  # Higher stability for lower std
            
            # Estimate memory usage (rough approximation)
            model_chars = self.model_db.get_model_characteristics(model_type)
            memory_usage = model_chars.memory_cost * X.nbytes / (1024**3)  # GB
            
            # Calculate efficiency score
            efficiency = 1.0 / (1.0 + training_time / 60.0)  # Normalize by minutes
            
            # Interpretability score
            interpretability = model_chars.interpretability
            
            # Risk assessment
            risk_score = self._calculate_risk_score(model_type, data_chars, performance_std)
            
            return {
                'performance': max(0.0, 1.0 - performance),  # Convert to score (higher is better)
                'stability': stability,
                'efficiency': efficiency,
                'interpretability': interpretability,
                'risk_score': risk_score,
                'training_time': training_time,
                'memory_usage': memory_usage,
                'cross_val_std': performance_std
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_type}: {e}")
            return self._get_default_evaluation()
    
    def _build_model(self, model_type: ModelType, X: np.ndarray, y: np.ndarray):
        """Build a model of specified type"""
        if model_type in self.model_builders:
            return self.model_builders[model_type](X, y)
        else:
            # Fallback to linear model
            return self.model_builders[ModelType.LINEAR](X, y)
    
    def _build_linear_model(self, X: np.ndarray, y: np.ndarray):
        """Build linear regression model"""
        return Ridge(alpha=1.0)
    
    def _build_polynomial_model(self, X: np.ndarray, y: np.ndarray):
        """Build polynomial regression model"""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        
        degree = min(3, X.shape[0] // 20)  # Conservative degree selection
        return Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', Ridge(alpha=1.0))
        ])
    
    def _build_random_forest_model(self, X: np.ndarray, y: np.ndarray):
        """Build random forest model"""
        n_estimators = min(100, max(10, X.shape[0] // 10))
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=42,
            n_jobs=1
        )
    
    def _build_gradient_boosting_model(self, X: np.ndarray, y: np.ndarray):
        """Build gradient boosting model"""
        n_estimators = min(100, max(50, X.shape[0] // 20))
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            random_state=42
        )
    
    def _build_xgboost_model(self, X: np.ndarray, y: np.ndarray):
        """Build XGBoost model"""
        if not XGBOOST_AVAILABLE:
            return self._build_gradient_boosting_model(X, y)
        
        return xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=1
        )
    
    def _build_neural_network_model(self, X: np.ndarray, y: np.ndarray):
        """Build neural network model"""
        hidden_layer_size = min(100, max(10, X.shape[1] * 2))
        return MLPRegressor(
            hidden_layer_sizes=(hidden_layer_size, hidden_layer_size // 2),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
    
    def _build_gaussian_process_model(self, X: np.ndarray, y: np.ndarray):
        """Build Gaussian process model"""
        # Limit to reasonable sample size for GP
        if X.shape[0] > 2000:
            indices = np.random.choice(X.shape[0], 2000, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample, y_sample = X, y
        
        return GaussianProcessRegressor(random_state=42)
    
    def _build_svm_model(self, X: np.ndarray, y: np.ndarray):
        """Build SVM model"""
        return SVR(kernel='rbf', gamma='scale')
    
    def _build_ensemble_model(self, X: np.ndarray, y: np.ndarray):
        """Build ensemble model"""
        from sklearn.ensemble import VotingRegressor
        
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
        ]
        
        if XGBOOST_AVAILABLE:
            estimators.append(('xgb', xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=1)))
        
        return VotingRegressor(estimators=estimators)
    
    def _calculate_risk_score(self, 
                            model_type: ModelType,
                            data_chars: DataCharacteristics,
                            performance_std: float) -> float:
        """Calculate risk score for model selection"""
        try:
            risk_factors = []
            
            # Model complexity vs data size risk
            model_chars = self.model_db.get_model_characteristics(model_type)
            if data_chars.sample_count < model_chars.data_requirements[1]:
                risk_factors.append(0.3)  # Overfitting risk
            
            # Performance instability risk
            if performance_std > 0.1:
                risk_factors.append(performance_std)
            
            # High-dimensional data risk
            if data_chars.feature_count > data_chars.sample_count:
                risk_factors.append(0.4)  # Curse of dimensionality
            
            # Missing data risk
            if data_chars.missing_data_ratio > 0.1:
                risk_factors.append(data_chars.missing_data_ratio)
            
            # Noise level risk
            if data_chars.noise_level > 0.3:
                risk_factors.append(data_chars.noise_level * 0.5)
            
            return np.mean(risk_factors) if risk_factors else 0.1
            
        except Exception:
            return 0.2  # Default moderate risk
    
    def _get_default_evaluation(self) -> Dict[str, float]:
        """Get default evaluation scores for failed evaluations"""
        return {
            'performance': 0.5,
            'stability': 0.5,
            'efficiency': 0.5,
            'interpretability': 0.5,
            'risk_score': 0.5,
            'training_time': 60.0,
            'memory_usage': 1.0,
            'cross_val_std': 0.1
        }
    
    def _select_optimal_model_from_evaluations(self, 
                                             evaluations: Dict[ModelType, Dict[str, float]],
                                             objectives: List[str],
                                             comp_constraints: ComputationalConstraint) -> ModelRecommendation:
        """Select optimal model from evaluations using multi-criteria optimization"""
        try:
            if not evaluations:
                return self._get_fallback_recommendation()
            
            # Calculate composite scores
            composite_scores = {}
            
            for model_type, eval_scores in evaluations.items():
                score = 0.0
                
                # Weight different criteria based on objectives
                if 'accuracy' in objectives:
                    score += self.config['performance_weight'] * eval_scores['performance']
                
                if 'efficiency' in objectives:
                    score += self.config['efficiency_weight'] * eval_scores['efficiency']
                
                if 'interpretability' in objectives:
                    score += self.config['interpretability_weight'] * eval_scores['interpretability']
                
                if 'stability' in objectives:
                    score += self.config['stability_weight'] * eval_scores['stability']
                
                # Penalty for high risk
                score -= self.config['risk_tolerance'] * eval_scores['risk_score']
                
                composite_scores[model_type] = score
            
            # Select best model
            best_model = max(composite_scores.keys(), key=lambda k: composite_scores[k])
            best_evaluation = evaluations[best_model]
            
            # Generate recommendation
            model_chars = self.model_db.get_model_characteristics(best_model)
            
            # Get alternative models
            sorted_models = sorted(composite_scores.keys(), 
                                 key=lambda k: composite_scores[k], reverse=True)
            alternatives = sorted_models[1:4]  # Top 3 alternatives
            
            # Generate hyperparameters and preprocessing steps
            hyperparameters = self._generate_hyperparameters(best_model)
            preprocessing_steps = self._generate_preprocessing_steps(best_model)
            
            # Generate justification
            justification = self._generate_justification(
                best_model, best_evaluation, model_chars, composite_scores[best_model]
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(best_model, best_evaluation)
            
            return ModelRecommendation(
                model_type=best_model,
                confidence=min(composite_scores[best_model], 1.0),
                expected_performance=best_evaluation['performance'],
                computational_feasibility=best_evaluation['efficiency'],
                justification=justification,
                hyperparameters=hyperparameters,
                preprocessing_steps=preprocessing_steps,
                alternative_models=alternatives,
                risk_factors=risk_factors,
                estimated_training_time=best_evaluation['training_time'],
                estimated_memory_usage=best_evaluation['memory_usage']
            )
            
        except Exception as e:
            self.logger.error(f"Error selecting optimal model: {e}")
            return self._get_fallback_recommendation()
    
    def _generate_hyperparameters(self, model_type: ModelType) -> Dict[str, Any]:
        """Generate recommended hyperparameters for model type"""
        hyperparameter_templates = {
            ModelType.LINEAR: {'alpha': 1.0, 'fit_intercept': True},
            ModelType.RANDOM_FOREST: {
                'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2,
                'min_samples_leaf': 1, 'random_state': 42
            },
            ModelType.GRADIENT_BOOSTING: {
                'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3,
                'random_state': 42
            },
            ModelType.NEURAL_NETWORK: {
                'hidden_layer_sizes': (100, 50), 'activation': 'relu',
                'learning_rate': 'adaptive', 'max_iter': 500,
                'early_stopping': True, 'random_state': 42
            },
            ModelType.SVM: {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
            ModelType.GAUSSIAN_PROCESS: {'alpha': 1e-10, 'random_state': 42}
        }
        
        return hyperparameter_templates.get(model_type, {})
    
    def _generate_preprocessing_steps(self, model_type: ModelType) -> List[str]:
        """Generate recommended preprocessing steps"""
        preprocessing_templates = {
            ModelType.LINEAR: ['standardization', 'feature_selection'],
            ModelType.POLYNOMIAL: ['standardization', 'feature_selection', 'regularization'],
            ModelType.RANDOM_FOREST: ['missing_value_imputation', 'outlier_removal'],
            ModelType.GRADIENT_BOOSTING: ['missing_value_imputation', 'outlier_removal'],
            ModelType.NEURAL_NETWORK: ['standardization', 'missing_value_imputation', 'feature_scaling'],
            ModelType.SVM: ['standardization', 'feature_scaling', 'feature_selection'],
            ModelType.GAUSSIAN_PROCESS: ['standardization', 'feature_selection', 'dimensionality_reduction']
        }
        
        return preprocessing_templates.get(model_type, ['standardization'])
    
    def _generate_justification(self, 
                              model_type: ModelType,
                              evaluation: Dict[str, float],
                              characteristics: ModelCharacteristics,
                              composite_score: float) -> str:
        """Generate justification for model selection"""
        justification_parts = []
        
        justification_parts.append(f"Selected {model_type.value} model based on:")
        justification_parts.append(f"• Composite score: {composite_score:.3f}")
        justification_parts.append(f"• Expected performance: {evaluation['performance']:.3f}")
        justification_parts.append(f"• Computational efficiency: {evaluation['efficiency']:.3f}")
        justification_parts.append(f"• Model stability: {evaluation['stability']:.3f}")
        
        if characteristics.handles_nonlinearity:
            justification_parts.append("• Handles nonlinear relationships effectively")
        
        if characteristics.handles_uncertainty:
            justification_parts.append("• Provides uncertainty quantification")
        
        if evaluation['risk_score'] < 0.3:
            justification_parts.append("• Low risk of overfitting/poor generalization")
        
        return " ".join(justification_parts)
    
    def _identify_risk_factors(self, 
                             model_type: ModelType,
                             evaluation: Dict[str, float]) -> List[str]:
        """Identify risk factors for selected model"""
        risk_factors = []
        
        if evaluation['performance'] < self.config['performance_threshold']:
            risk_factors.append("Below-threshold expected performance")
        
        if evaluation['cross_val_std'] > 0.2:
            risk_factors.append("High performance variability across cross-validation folds")
        
        if evaluation['training_time'] > 3600:  # 1 hour
            risk_factors.append("Long training time may impact iteration speed")
        
        if evaluation['memory_usage'] > 32:  # 32 GB
            risk_factors.append("High memory requirements may limit deployment options")
        
        model_chars = self.model_db.get_model_characteristics(model_type)
        if model_chars.hyperparameter_sensitivity > 0.7:
            risk_factors.append("High sensitivity to hyperparameter tuning")
        
        if model_chars.training_stability < 0.6:
            risk_factors.append("Potential training instability")
        
        return risk_factors
    
    def _get_fallback_recommendation(self) -> ModelRecommendation:
        """Get fallback recommendation when selection fails"""
        return ModelRecommendation(
            model_type=ModelType.LINEAR,
            confidence=0.5,
            expected_performance=0.6,
            computational_feasibility=0.9,
            justification="Fallback to linear model due to selection failure",
            hyperparameters={'alpha': 1.0, 'fit_intercept': True},
            preprocessing_steps=['standardization'],
            alternative_models=[ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING],
            risk_factors=['Selection process failed, using conservative fallback'],
            estimated_training_time=10.0,
            estimated_memory_usage=0.5
        )
    
    def _update_selection_history(self, 
                                data_chars: DataCharacteristics,
                                recommendation: ModelRecommendation):
        """Update selection history for adaptive learning"""
        history_entry = {
            'timestamp': time.time(),
            'data_characteristics': data_chars,
            'selected_model': recommendation.model_type,
            'confidence': recommendation.confidence,
            'expected_performance': recommendation.expected_performance
        }
        
        self.selection_history.append(history_entry)
        
        # Limit history size
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]

# Advanced optimization functions
class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {'n_trials': 100, 'timeout': 3600}
        self.logger = logging.getLogger(__name__)
    
    async def optimize_hyperparameters(self, 
                                     model_type: ModelType,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     optimization_budget: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters for selected model"""
        try:
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(),
                pruner=MedianPruner(n_startup_trials=10)
            )
            
            # Define objective function based on model type
            objective_func = self._create_objective_function(model_type, X, y)
            
            # Run optimization
            study.optimize(objective_func, n_trials=optimization_budget, timeout=3600)
            
            return study.best_params
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {e}")
            return {}
    
    def _create_objective_function(self, model_type: ModelType, X: np.ndarray, y: np.ndarray):
        """Create objective function for hyperparameter optimization"""
        def objective(trial):
            try:
                # Define hyperparameter search space based on model type
                if model_type == ModelType.RANDOM_FOREST:
                    n_estimators = trial.suggest_int('n_estimators', 50, 300)
                    max_depth = trial.suggest_int('max_depth', 3, 20)
                    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                    
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42,
                        n_jobs=1
                    )
                
                elif model_type == ModelType.GRADIENT_BOOSTING:
                    n_estimators = trial.suggest_int('n_estimators', 50, 300)
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                    max_depth = trial.suggest_int('max_depth', 2, 10)
                    
                    model = GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=42
                    )
                
                else:
                    # Fallback to simple model
                    alpha = trial.suggest_float('alpha', 0.01, 10.0)
                    model = Ridge(alpha=alpha)
                
                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X, y, cv=5, scoring='neg_mean_squared_error'
                )
                
                return -np.mean(cv_scores)  # Minimize MSE
                
            except Exception as e:
                return float('inf')  # Return worst score on error
        
        return objective

# Example usage and testing
async def example_adaptive_model_selection():
    """Example of using the adaptive model selection system"""
    
    # Create synthetic geological data
    np.random.seed(42)
    
    # Create simulation data
    spatial_data = np.random.rand(1000, 50, 50)  # 1000 samples, 50x50 spatial grid
    temporal_data = np.random.rand(1000, 20)     # 1000 samples, 20 time points
    
    # Add some structure
    for i in range(1000):
        spatial_data[i] += 0.3 * np.sin(np.arange(50)[:, None] * 0.1)
        temporal_data[i] += 0.2 * np.sin(np.arange(20) * 0.3)
    
    # Create target variable (e.g., deformation magnitude)
    target_data = (np.mean(spatial_data.reshape(1000, -1), axis=1) + 
                   np.mean(temporal_data, axis=1) + 
                   0.1 * np.random.randn(1000))
    
    # Create simulation data object
    sim_data = SimulationData(
        spatial_data=spatial_data,
        temporal_data=temporal_data,
        metadata={
            'simulation_type': 'geological_deformation',
            'resolution': 'medium',
            'time_span': '20_years'
        }
    )
    
    # Initialize adaptive model selector
    model_selector = AdaptiveModelSelector()
    
    # Select optimal model
    print("Selecting optimal model for geological simulation...")
    recommendation = await model_selector.select_optimal_model(
        simulation_data=sim_data,
        target_data=target_data,
        computational_constraints=ComputationalConstraint.MODERATE,
        objectives=['accuracy', 'efficiency', 'interpretability']
    )
    
    # Display results
    print("\n=== Model Selection Results ===")
    print(f"Recommended Model: {recommendation.model_type.value}")
    print(f"Confidence: {recommendation.confidence:.3f}")
    print(f"Expected Performance: {recommendation.expected_performance:.3f}")
    print(f"Computational Feasibility: {recommendation.computational_feasibility:.3f}")
    print(f"Estimated Training Time: {recommendation.estimated_training_time:.1f} seconds")
    print(f"Estimated Memory Usage: {recommendation.estimated_memory_usage:.1f} GB")
    
    print(f"\nJustification: {recommendation.justification}")
    
    print(f"\nRecommended Hyperparameters:")
    for param, value in recommendation.hyperparameters.items():
        print(f"  {param}: {value}")
    
    print(f"\nPreprocessing Steps:")
    for step in recommendation.preprocessing_steps:
        print(f"  • {step}")
    
    print(f"\nAlternative Models:")
    for alt_model in recommendation.alternative_models:
        print(f"  • {alt_model.value}")
    
    if recommendation.risk_factors:
        print(f"\nRisk Factors:")
        for risk in recommendation.risk_factors:
            print(f"  ⚠ {risk}")
    
    return recommendation

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    asyncio.run(example_adaptive_model_selection())