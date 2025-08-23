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
            'optimization_budget': 300,  # Optuna trials
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
        """Calculate advanced missing data ratio with pattern analysis and spatial coherence"""
        try:
            total_elements = 0
            missing_elements = 0
            spatial_missing_patterns = []
            temporal_missing_patterns = []
            data_quality_metrics = {}
            
            # Comprehensive missing data analysis
            # Get arrays with names (create compatibility wrapper)
            arrays_with_names = []
            if hasattr(simulation_data, 'get_all_data_arrays_with_names'):
                arrays_with_names = simulation_data.get_all_data_arrays_with_names()
            else:
                # Fallback: create names from array indices and available attributes
                arrays = simulation_data.get_all_data_arrays()
                for i, array in enumerate(arrays):
                    # Try to infer names from common geological data attributes
                    name = f"array_{i}"
                    if hasattr(simulation_data, 'temperature_3d') and array is simulation_data.temperature_3d:
                        name = "temperature_3d"
                    elif hasattr(simulation_data, 'pressure_3d') and array is simulation_data.pressure_3d:
                        name = "pressure_3d"
                    elif hasattr(simulation_data, 'melt_fraction_3d') and array is simulation_data.melt_fraction_3d:
                        name = "melt_fraction_3d"
                    elif hasattr(simulation_data, 'density_3d') and array is simulation_data.density_3d:
                        name = "density_3d"
                    arrays_with_names.append((name, array))
                    
            for array_name, array in arrays_with_names:
                if array is None or array.size == 0:
                    continue
                    
                array_flat = array.flatten()
                total_elements += array.size
                
                # Detect different types of missing/invalid data
                nan_mask = np.isnan(array_flat)
                inf_mask = np.isinf(array_flat)
                zero_mask = (array_flat == 0) if array.dtype != bool else np.zeros_like(array_flat, dtype=bool)
                outlier_mask = self._detect_statistical_outliers(array_flat)
                
                # Count missing elements with weights based on missingness type
                nan_count = np.sum(nan_mask)
                inf_count = np.sum(inf_mask)
                suspicious_zero_count = self._count_suspicious_zeros(array_flat, zero_mask)
                outlier_count = np.sum(outlier_mask)
                
                # Weight different types of missing data
                weighted_missing = (
                    nan_count * 1.0 +  # Full weight for NaN
                    inf_count * 1.0 +  # Full weight for infinite values
                    suspicious_zero_count * 0.7 +  # Partial weight for suspicious zeros
                    outlier_count * 0.3  # Light weight for statistical outliers
                )
                
                missing_elements += weighted_missing
                
                # Analyze spatial patterns of missingness (if data has spatial structure)
                if len(array.shape) >= 2:
                    spatial_pattern_score = self._analyze_spatial_missing_patterns(array, nan_mask.reshape(array.shape))
                    spatial_missing_patterns.append(spatial_pattern_score)
                
                # Analyze temporal patterns (if data has time dimension)
                if len(array.shape) >= 3 or 'time' in array_name.lower() or 't_' in array_name:
                    temporal_pattern_score = self._analyze_temporal_missing_patterns(array, nan_mask.reshape(array.shape))
                    temporal_missing_patterns.append(temporal_pattern_score)
                
                # Store detailed quality metrics for this array
                data_quality_metrics[array_name] = {
                    'missing_ratio': weighted_missing / array.size,
                    'nan_ratio': nan_count / array.size,
                    'inf_ratio': inf_count / array.size,
                    'outlier_ratio': outlier_count / array.size,
                    'data_completeness': 1.0 - (weighted_missing / array.size)
                }
            
            # Calculate base missing ratio
            base_ratio = missing_elements / total_elements if total_elements > 0 else 0.0
            
            # Apply pattern-based adjustments
            pattern_penalty = self._calculate_pattern_penalty(spatial_missing_patterns, temporal_missing_patterns)
            
            # Calculate information content loss due to missing data
            information_loss = self._estimate_information_loss(data_quality_metrics, base_ratio)
            
            # Final adjusted missing data ratio
            adjusted_ratio = min(1.0, base_ratio + pattern_penalty + information_loss * 0.1)
            
            # Store detailed analysis for model selection decisions
            if hasattr(simulation_data, 'quality_assessment'):
                simulation_data.quality_assessment['missing_data_analysis'] = {
                    'base_ratio': base_ratio,
                    'pattern_penalty': pattern_penalty,
                    'information_loss': information_loss,
                    'adjusted_ratio': adjusted_ratio,
                    'array_metrics': data_quality_metrics
                }
            
            return adjusted_ratio
            
        except Exception as e:
            logging.warning(f"Advanced missing data calculation failed: {e}")
            # Fallback to basic calculation
            try:
                total_elements = sum(array.size for array in simulation_data.get_all_data_arrays() if array is not None)
                missing_elements = sum(
                    np.sum(np.isnan(array)) + np.sum(np.isinf(array)) 
                    for array in simulation_data.get_all_data_arrays() 
                    if array is not None
                )
                return missing_elements / total_elements if total_elements > 0 else 0.0
            except Exception:
                return 0.5  # Conservative estimate when calculation fails
    
    def _detect_statistical_outliers(self, data: np.ndarray, method: str = 'iqr') -> np.ndarray:
        """Detect statistical outliers in data array"""
        try:
            valid_data = data[~np.isnan(data) & ~np.isinf(data)]
            if len(valid_data) < 4:
                return np.zeros_like(data, dtype=bool)
            
            if method == 'iqr':
                # Interquartile Range method
                q1, q3 = np.percentile(valid_data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_mask = (data < lower_bound) | (data > upper_bound)
                
            elif method == 'zscore':
                # Z-score method (modified for robustness)
                median = np.median(valid_data)
                mad = np.median(np.abs(valid_data - median))
                modified_z_scores = 0.6745 * (data - median) / mad if mad > 0 else np.zeros_like(data)
                outlier_mask = np.abs(modified_z_scores) > 3.5
                
            elif method == 'isolation':
                # Isolation Forest approach (simplified)
                try:
                    from sklearn.ensemble import IsolationForest
                    if len(valid_data.reshape(-1, 1)) >= 10:
                        clf = IsolationForest(contamination=0.1, random_state=42)
                        outlier_scores = clf.fit_predict(data.reshape(-1, 1))
                        outlier_mask = outlier_scores == -1
                    else:
                        outlier_mask = np.zeros_like(data, dtype=bool)
                except ImportError:
                    # Fallback to IQR method
                    return self._detect_statistical_outliers(data, 'iqr')
            else:
                outlier_mask = np.zeros_like(data, dtype=bool)
                
            return outlier_mask
            
        except Exception:
            return np.zeros_like(data, dtype=bool)
    
    def _count_suspicious_zeros(self, data: np.ndarray, zero_mask: np.ndarray) -> int:
        """Count zeros that are likely missing data rather than true zeros"""
        try:
            if not np.any(zero_mask):
                return 0
                
            # Remove NaN and inf values for analysis
            valid_data = data[~np.isnan(data) & ~np.isinf(data)]
            if len(valid_data) < 10:
                return int(np.sum(zero_mask) * 0.5)  # Conservative estimate
                
            # Analyze distribution characteristics
            non_zero_data = valid_data[valid_data != 0]
            if len(non_zero_data) < 5:
                return int(np.sum(zero_mask))
                
            # Statistical tests for suspicious zeros
            suspicious_count = 0
            zero_count = np.sum(zero_mask)
            
            # Test 1: Unusual clustering of zeros
            if zero_count / len(valid_data) > 0.1:  # More than 10% zeros
                # Check if zeros appear in clusters (spatial or temporal)
                zero_positions = np.where(zero_mask)[0]
                if len(zero_positions) > 1:
                    gaps = np.diff(zero_positions)
                    clustered_zeros = np.sum(gaps == 1)  # Adjacent zeros
                    if clustered_zeros / zero_count > 0.3:  # 30% are clustered
                        suspicious_count += int(clustered_zeros * 0.8)
                        
            # Test 2: Distribution analysis
            if len(non_zero_data) >= 5:
                # Check if the non-zero distribution suggests zeros are outliers
                mean_nonzero = np.mean(non_zero_data)
                std_nonzero = np.std(non_zero_data)
                
                # If distribution is far from zero, zeros might be missing data
                if mean_nonzero > 2 * std_nonzero:
                    suspicious_count += int(zero_count * 0.6)
                elif mean_nonzero > std_nonzero:
                    suspicious_count += int(zero_count * 0.3)
                    
            return min(suspicious_count, zero_count)
            
        except Exception:
            return int(np.sum(zero_mask) * 0.2)  # Conservative default
    
    def _analyze_spatial_missing_patterns(self, array: np.ndarray, missing_mask: np.ndarray) -> float:
        """Analyze spatial patterns in missing data"""
        try:
            if len(array.shape) < 2:
                return 0.0
                
            # Reshape missing mask to match spatial dimensions
            spatial_shape = array.shape[:2]  # Take first two dimensions as spatial
            if missing_mask.size != np.prod(spatial_shape):
                # Try to reshape or sample appropriately
                if len(array.shape) > 2:
                    missing_2d = missing_mask.reshape(array.shape).mean(axis=tuple(range(2, len(array.shape))))
                else:
                    missing_2d = missing_mask.reshape(spatial_shape)
            else:
                missing_2d = missing_mask.reshape(spatial_shape)
                
            # Convert to binary missing pattern
            missing_binary = (missing_2d > 0.5).astype(int)
            
            # Calculate spatial autocorrelation of missing data
            try:
                from scipy import ndimage
                
                # Moran's I approximation for missing data clustering
                kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # 8-connectivity
                neighbor_sum = ndimage.convolve(missing_binary.astype(float), kernel, mode='constant')
                
                # Calculate local spatial correlation
                n_missing = np.sum(missing_binary)
                if n_missing == 0:
                    return 0.0
                    
                total_neighbors = ndimage.convolve(np.ones_like(missing_binary, dtype=float), kernel, mode='constant')
                local_correlation = np.where(total_neighbors > 0, neighbor_sum / total_neighbors, 0)
                
                # Pattern score: higher values indicate more problematic spatial clustering
                clustering_score = np.mean(local_correlation[missing_binary == 1]) if n_missing > 0 else 0.0
                
                # Additional penalty for large connected components
                try:
                    labeled_array, num_features = ndimage.label(missing_binary)
                    if num_features > 0:
                        component_sizes = ndimage.sum(missing_binary, labeled_array, range(1, num_features + 1))
                        max_component_size = np.max(component_sizes)
                        large_component_penalty = min(0.3, max_component_size / missing_binary.size)
                    else:
                        large_component_penalty = 0.0
                except Exception:
                    large_component_penalty = 0.0
                    
                return min(1.0, clustering_score + large_component_penalty)
                
            except ImportError:
                # Fallback: simple edge detection approach
                edges_x = np.abs(np.diff(missing_binary, axis=1))
                edges_y = np.abs(np.diff(missing_binary, axis=0))
                edge_density = (np.sum(edges_x) + np.sum(edges_y)) / missing_binary.size
                return min(1.0, 1.0 - edge_density)  # Lower edge density = more clustering
                
        except Exception:
            return 0.1  # Conservative default
    
    def _analyze_temporal_missing_patterns(self, array: np.ndarray, missing_mask: np.ndarray) -> float:
        """Analyze temporal patterns in missing data"""
        try:
            # Assume last dimension or longest dimension is temporal
            if len(array.shape) == 1:
                temporal_missing = missing_mask.astype(float)
            elif len(array.shape) == 3:
                # For 3D arrays, assume last dimension is temporal
                temporal_missing = np.mean(missing_mask, axis=(0, 1))
            else:
                # For 2D arrays, use the longer dimension as temporal
                if array.shape[0] > array.shape[1]:
                    temporal_missing = np.mean(missing_mask, axis=1)
                else:
                    temporal_missing = np.mean(missing_mask, axis=0)
                    
            if len(temporal_missing) < 3:
                return 0.0
                
            # Detect temporal clustering of missing data
            binary_missing = (temporal_missing > 0.5).astype(int)
            
            # Calculate run lengths of missing data
            diff = np.diff(np.concatenate(([0], binary_missing, [0])))
            run_starts = np.where(diff == 1)[0]
            run_ends = np.where(diff == -1)[0]
            
            if len(run_starts) == 0:
                return 0.0
                
            run_lengths = run_ends - run_starts
            
            # Pattern analysis
            total_missing = np.sum(binary_missing)
            if total_missing == 0:
                return 0.0
                
            # Clustering score based on run length distribution
            mean_run_length = np.mean(run_lengths)
            max_run_length = np.max(run_lengths)
            
            # Higher penalty for fewer, longer runs (indicates systematic missing periods)
            clustering_score = min(1.0, (mean_run_length - 1) / len(temporal_missing))
            systematic_penalty = min(0.3, max_run_length / len(temporal_missing))
            
            return min(1.0, clustering_score + systematic_penalty)
            
        except Exception:
            return 0.1  # Conservative default
    
    def _calculate_pattern_penalty(self, spatial_patterns: List[float], temporal_patterns: List[float]) -> float:
        """Calculate penalty based on problematic missing data patterns"""
        try:
            penalty = 0.0
            
            # Spatial pattern penalties
            if spatial_patterns:
                mean_spatial_clustering = np.mean(spatial_patterns)
                max_spatial_clustering = np.max(spatial_patterns)
                
                # Penalty increases with spatial clustering
                penalty += min(0.15, mean_spatial_clustering * 0.1)
                penalty += min(0.1, (max_spatial_clustering - mean_spatial_clustering) * 0.05)
            
            # Temporal pattern penalties
            if temporal_patterns:
                mean_temporal_clustering = np.mean(temporal_patterns)
                max_temporal_clustering = np.max(temporal_patterns)
                
                # Penalty increases with temporal clustering (systematic gaps)
                penalty += min(0.1, mean_temporal_clustering * 0.08)
                penalty += min(0.05, (max_temporal_clustering - mean_temporal_clustering) * 0.03)
            
            # Cross-pattern penalties (spatial and temporal clustering together)
            if spatial_patterns and temporal_patterns:
                cross_correlation = min(1.0, np.mean(spatial_patterns) * np.mean(temporal_patterns))
                penalty += min(0.05, cross_correlation * 0.02)
            
            return min(0.3, penalty)  # Cap total pattern penalty
            
        except Exception:
            return 0.05  # Conservative default penalty
    
    def _estimate_information_loss(self, quality_metrics: Dict[str, Dict], base_ratio: float) -> float:
        """Estimate information content loss due to missing data patterns"""
        try:
            if not quality_metrics:
                return base_ratio
                
            # Weight arrays by importance (geological significance)
            importance_weights = {
                'temperature': 1.0, 'pressure': 1.0, 'melt_fraction': 1.0,
                'seismic': 0.9, 'velocity': 0.9, 'density': 0.8,
                'composition': 0.8, 'viscosity': 0.7, 'porosity': 0.6,
                'permeability': 0.6, 'stress': 0.7, 'strain': 0.6
            }
            
            weighted_information_loss = 0.0
            total_weight = 0.0
            
            for array_name, metrics in quality_metrics.items():
                # Determine importance weight
                weight = 0.5  # default weight
                for key, w in importance_weights.items():
                    if key.lower() in array_name.lower():
                        weight = w
                        break
                
                # Calculate information loss for this array
                missing_ratio = metrics['missing_ratio']
                completeness = metrics['data_completeness']
                
                # Non-linear information loss (missing 50% data loses more than 50% information)
                if missing_ratio > 0:
                    info_loss = 1.0 - (completeness ** 0.7)  # Power law information loss
                else:
                    info_loss = 0.0
                
                # Additional penalties for specific missing data types
                nan_penalty = metrics['nan_ratio'] * 0.1  # NaN values are worse
                inf_penalty = metrics['inf_ratio'] * 0.15  # Infinite values are worst
                outlier_penalty = metrics['outlier_ratio'] * 0.05  # Outliers less critical
                
                total_info_loss = info_loss + nan_penalty + inf_penalty + outlier_penalty
                
                weighted_information_loss += weight * total_info_loss
                total_weight += weight
            
            if total_weight > 0:
                return min(1.0, weighted_information_loss / total_weight)
            else:
                return base_ratio
                
        except Exception:
            return base_ratio * 1.2  # Conservative estimate with small penalty

    def _estimate_noise_level(self, target_data: np.ndarray) -> float:
        """Advanced noise level estimation using multiple sophisticated methods"""
        try:
            # Input validation and preprocessing
            if target_data is None or target_data.size == 0:
                return 0.1
                
            # Remove NaN and infinite values
            valid_data = target_data[np.isfinite(target_data)]
            if len(valid_data) < 5:
                return 0.5  # High noise estimate for very sparse data
                
            if len(valid_data) < 10:
                return self._estimate_noise_minimal_data(valid_data)
                
            # Determine optimal analysis method based on data characteristics
            analysis_method = self._select_noise_analysis_method(valid_data)
            
            # Apply multiple noise estimation techniques
            noise_estimates = {}
            
            # Method 1: Detrended Fluctuation Analysis (DFA)
            if len(valid_data) >= 20:
                try:
                    dfa_noise = self._detrended_fluctuation_analysis(valid_data)
                    noise_estimates['dfa'] = {'value': dfa_noise, 'weight': 0.3, 'confidence': 0.8}
                except Exception:
                    pass
                    
            # Method 2: Wavelet-based noise estimation
            if len(valid_data) >= 16:  # Minimum for wavelet analysis
                try:
                    wavelet_noise = self._wavelet_noise_estimation(valid_data)
                    noise_estimates['wavelet'] = {'value': wavelet_noise, 'weight': 0.25, 'confidence': 0.9}
                except Exception:
                    pass
                    
            # Method 3: Local polynomial detrending with robust statistics
            if len(valid_data) >= 15:
                try:
                    robust_noise = self._robust_local_noise_estimation(valid_data)
                    noise_estimates['robust'] = {'value': robust_noise, 'weight': 0.2, 'confidence': 0.85}
                except Exception:
                    pass
                    
            # Method 4: Spectral analysis approach
            if len(valid_data) >= 32:  # Minimum for meaningful FFT
                try:
                    spectral_noise = self._spectral_noise_estimation(valid_data)
                    noise_estimates['spectral'] = {'value': spectral_noise, 'weight': 0.15, 'confidence': 0.7}
                except Exception:
                    pass
                    
            # Method 5: Empirical Mode Decomposition (EMD) based
            if len(valid_data) >= 50:
                try:
                    emd_noise = self._emd_noise_estimation(valid_data)
                    noise_estimates['emd'] = {'value': emd_noise, 'weight': 0.1, 'confidence': 0.6}
                except Exception:
                    pass
                    
            # Fallback methods for edge cases
            if not noise_estimates:
                # Basic methods as fallback
                if len(valid_data) > 10:
                    basic_noise = self._basic_noise_estimation(valid_data)
                    noise_estimates['basic'] = {'value': basic_noise, 'weight': 1.0, 'confidence': 0.5}
                else:
                    return 0.2  # Conservative estimate
                    
            # Combine estimates using weighted confidence intervals
            final_noise_estimate = self._combine_noise_estimates(noise_estimates, valid_data)
            
            # Apply geological context adjustments
            contextual_noise = self._apply_geological_noise_context(final_noise_estimate, valid_data, analysis_method)
            
            # Validate and bound the result
            bounded_noise = np.clip(contextual_noise, 0.001, 0.95)  # Reasonable bounds
            
            return float(bounded_noise)
            
        except Exception as e:
            logging.warning(f"Advanced noise estimation failed: {e}")
            # Fallback to basic robust estimation
            try:
                return self._basic_noise_estimation(valid_data if 'valid_data' in locals() else target_data)
            except Exception:
                return 0.1
    
    def _estimate_noise_minimal_data(self, data: np.ndarray) -> float:
        """Noise estimation for very small datasets"""
        try:
            if len(data) < 3:
                return 0.3
            
            # Use median absolute deviation for robustness
            median_val = np.median(data)
            mad = np.median(np.abs(data - median_val))
            
            # Estimate noise as ratio of MAD to median (if non-zero)
            if abs(median_val) > 1e-10:
                noise_ratio = mad / abs(median_val)
            else:
                # For data centered near zero, use MAD relative to data range
                data_range = np.ptp(data)  # peak-to-peak
                noise_ratio = mad / data_range if data_range > 0 else 0.3
                
            return min(0.8, max(0.05, noise_ratio))
            
        except Exception:
            return 0.3
    
    def _select_noise_analysis_method(self, data: np.ndarray) -> str:
        """Select optimal noise analysis method based on data characteristics"""
        try:
            n = len(data)
            
            # Check for trend characteristics
            has_trend = abs(stats.pearsonr(np.arange(n), data)[0]) > 0.3
            
            # Check for periodicity using autocorrelation
            if n >= 20:
                autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]
                has_periodicity = np.max(autocorr[1:min(n//4, 50)]) > 0.4
            else:
                has_periodicity = False
                
            # Check for non-stationarity
            if n >= 30:
                # Simple Augmented Dickey-Fuller test approximation
                # Split data in half and compare statistics
                mid = n // 2
                var1 = np.var(data[:mid])
                var2 = np.var(data[mid:])
                is_stationary = abs(var1 - var2) / (var1 + var2) < 0.5
            else:
                is_stationary = True
                
            # Select method based on characteristics
            if has_periodicity and n >= 50:
                return 'spectral_primary'
            elif not is_stationary and n >= 32:
                return 'wavelet_primary'
            elif has_trend and n >= 20:
                return 'dfa_primary'
            else:
                return 'robust_primary'
                
        except Exception:
            return 'robust_primary'
    
    def _detrended_fluctuation_analysis(self, data: np.ndarray) -> float:
        """Detrended Fluctuation Analysis for noise estimation"""
        try:
            # Integrate the data
            y = np.cumsum(data - np.mean(data))
            n = len(y)
            
            # Define box sizes (scales)
            scales = np.unique(np.logspace(0.5, np.log10(n//4), 10).astype(int))
            scales = scales[scales >= 4]
            
            if len(scales) < 3:
                # Fallback for short series
                return self._basic_noise_estimation(data)
                
            fluctuations = []
            
            for scale in scales:
                # Divide series into boxes
                n_boxes = n // scale
                if n_boxes < 2:
                    continue
                    
                box_fluctuations = []
                
                for i in range(n_boxes):
                    start_idx = i * scale
                    end_idx = (i + 1) * scale
                    box_data = y[start_idx:end_idx]
                    
                    # Fit polynomial trend (linear for simplicity)
                    x_box = np.arange(len(box_data))
                    if len(box_data) > 1:
                        poly_coeff = np.polyfit(x_box, box_data, 1)
                        trend = np.polyval(poly_coeff, x_box)
                        fluctuation = np.sqrt(np.mean((box_data - trend) ** 2))
                        box_fluctuations.append(fluctuation)
                
                if box_fluctuations:
                    fluctuations.append(np.mean(box_fluctuations))
            
            if len(fluctuations) < 2:
                return self._basic_noise_estimation(data)
                
            # Estimate scaling exponent
            log_scales = np.log10(scales[:len(fluctuations)])
            log_fluctuations = np.log10(fluctuations)
            
            # Remove invalid values
            valid_idx = np.isfinite(log_fluctuations) & np.isfinite(log_scales)
            if np.sum(valid_idx) < 2:
                return self._basic_noise_estimation(data)
                
            slope, _ = np.polyfit(log_scales[valid_idx], log_fluctuations[valid_idx], 1)
            
            # Convert scaling exponent to noise level
            # Higher scaling exponent indicates more correlation (less noise)
            # Scaling exponent: ~0.5 for white noise, ~1.0 for random walk, ~1.5 for very smooth
            noise_level = max(0.01, min(0.9, 1.5 - slope))
            
            return noise_level
            
        except Exception:
            return self._basic_noise_estimation(data)
    
    def _wavelet_noise_estimation(self, data: np.ndarray) -> float:
        """Wavelet-based noise estimation using detail coefficients"""
        try:
            # Use scipy wavelet transform if available, otherwise approximate
            try:
                from scipy import signal
                
                # Use Daubechies wavelets for noise estimation
                # Approximate with simple high-pass filtering
                nyquist = 0.5
                high_freq = 0.3  # High frequency cutoff
                b, a = signal.butter(4, high_freq/nyquist, btype='high')
                
                # Apply filter to get high-frequency components (noise)
                high_freq_component = signal.filtfilt(b, a, data)
                
                # Estimate noise from high-frequency energy
                noise_energy = np.var(high_freq_component)
                signal_energy = np.var(data)
                
                noise_ratio = np.sqrt(noise_energy / signal_energy) if signal_energy > 0 else 0.5
                
            except ImportError:
                # Fallback: simple wavelet approximation using differences
                # Approximate detail coefficients using differences at multiple scales
                noise_estimates = []
                
                for scale in [1, 2, 4]:
                    if len(data) > 2 * scale:
                        # Downsample and compute differences
                        downsampled = data[::scale]
                        if len(downsampled) > 1:
                            details = np.diff(downsampled)
                            noise_level = np.std(details) / (np.std(downsampled) * np.sqrt(2))
                            noise_estimates.append(noise_level)
                
                noise_ratio = np.median(noise_estimates) if noise_estimates else 0.2
                
            return min(0.9, max(0.01, noise_ratio))
            
        except Exception:
            return self._basic_noise_estimation(data)
    
    def _robust_local_noise_estimation(self, data: np.ndarray) -> float:
        """Robust local polynomial detrending with noise estimation"""
        try:
            n = len(data)
            window_sizes = [max(5, n//10), max(7, n//8), max(9, n//6)]
            noise_estimates = []
            
            for window_size in window_sizes:
                if window_size >= n:
                    continue
                    
                local_noise_levels = []
                
                # Sliding window analysis
                step = max(1, window_size // 3)
                for i in range(0, n - window_size + 1, step):
                    window_data = data[i:i + window_size]
                    
                    # Fit robust polynomial (quadratic)
                    x_window = np.arange(len(window_data))
                    
                    try:
                        # Use RANSAC-like approach for robust fitting
                        best_residuals = None
                        best_noise = float('inf')
                        
                        for poly_degree in [1, 2]:  # Linear and quadratic
                            if len(window_data) > poly_degree + 1:
                                poly_coeff = np.polyfit(x_window, window_data, poly_degree)
                                trend = np.polyval(poly_coeff, x_window)
                                residuals = window_data - trend
                                
                                # Use median absolute deviation for robustness
                                mad = np.median(np.abs(residuals - np.median(residuals)))
                                noise_level = mad * 1.4826  # MAD to std conversion
                                
                                if noise_level < best_noise:
                                    best_noise = noise_level
                                    best_residuals = residuals
                        
                        if best_residuals is not None:
                            # Normalize by local signal strength
                            local_signal_strength = np.std(window_data)
                            if local_signal_strength > 0:
                                relative_noise = best_noise / local_signal_strength
                                local_noise_levels.append(relative_noise)
                                
                    except Exception:
                        continue
                
                if local_noise_levels:
                    # Use robust statistics to combine local estimates
                    median_noise = np.median(local_noise_levels)
                    noise_estimates.append(median_noise)
            
            if noise_estimates:
                # Weight different window sizes
                weights = np.array([0.5, 0.3, 0.2])[:len(noise_estimates)]
                weights = weights / np.sum(weights)
                final_noise = np.average(noise_estimates, weights=weights)
                return min(0.9, max(0.01, final_noise))
            else:
                return self._basic_noise_estimation(data)
                
        except Exception:
            return self._basic_noise_estimation(data)
    
    def _spectral_noise_estimation(self, data: np.ndarray) -> float:
        """Spectral analysis approach for noise estimation"""
        try:
            # Remove DC component and apply window
            windowed_data = data - np.mean(data)
            
            # Apply Hanning window to reduce spectral leakage
            window = np.hanning(len(windowed_data))
            windowed_data = windowed_data * window
            
            # Compute power spectral density
            fft_data = np.fft.fft(windowed_data)
            power_spectrum = np.abs(fft_data) ** 2
            frequencies = np.fft.fftfreq(len(windowed_data))
            
            # Consider only positive frequencies
            positive_freq_idx = frequencies > 0
            pos_freqs = frequencies[positive_freq_idx]
            pos_power = power_spectrum[positive_freq_idx]
            
            if len(pos_power) < 4:
                return self._basic_noise_estimation(data)
                
            # Divide spectrum into low, mid, and high frequency bands
            n_bands = min(3, len(pos_power) // 3)
            if n_bands < 2:
                return self._basic_noise_estimation(data)
                
            band_size = len(pos_power) // n_bands
            band_powers = []
            
            for i in range(n_bands):
                start_idx = i * band_size
                end_idx = (i + 1) * band_size if i < n_bands - 1 else len(pos_power)
                band_power = np.mean(pos_power[start_idx:end_idx])
                band_powers.append(band_power)
            
            # Estimate noise level from spectral characteristics
            total_power = np.sum(band_powers)
            if total_power > 0:
                # High frequency band typically contains more noise
                high_freq_power = band_powers[-1]
                low_freq_power = band_powers[0]
                
                # Noise ratio based on high-to-low frequency power ratio
                if low_freq_power > 0:
                    hf_lf_ratio = high_freq_power / low_freq_power
                    # Calibrate: higher ratio suggests more noise
                    noise_level = min(0.8, max(0.05, np.log10(1 + hf_lf_ratio) * 0.3))
                else:
                    noise_level = 0.3
                    
                # Additional check: spectral flatness (measure of noisiness)
                geometric_mean = np.exp(np.mean(np.log(pos_power + 1e-10)))
                arithmetic_mean = np.mean(pos_power)
                
                if arithmetic_mean > 0:
                    spectral_flatness = geometric_mean / arithmetic_mean
                    # Higher flatness indicates more noise-like signal
                    noise_level = max(noise_level, spectral_flatness * 0.5)
                    
                return min(0.9, max(0.01, noise_level))
            else:
                return 0.3
                
        except Exception:
            return self._basic_noise_estimation(data)
    
    def _emd_noise_estimation(self, data: np.ndarray) -> float:
        """Advanced Empirical Mode Decomposition with ensemble and bidimensional extensions"""
        try:
            # Validate input and preprocess
            if len(data) < 10:
                return self._basic_noise_estimation(data)
            
            # Detect signal characteristics for adaptive EMD parameters
            signal_stats = self._analyze_emd_signal_characteristics(data)
            
            # Ensemble EMD for robustness against mode mixing
            if signal_stats['complexity_score'] > 0.7 and len(data) > 100:
                noise_estimate = self._ensemble_emd_noise_estimation(data, signal_stats)
            else:
                noise_estimate = self._adaptive_emd_noise_estimation(data, signal_stats)
            
            # Multi-scale noise validation using Hilbert-Huang spectrum
            validated_noise = self._validate_emd_noise_with_hht(data, noise_estimate, signal_stats)
            
            # Geological context adjustment for volcanic/seismic data patterns
            geological_adjusted = self._adjust_emd_noise_for_geological_context(validated_noise, signal_stats)
            
            return min(0.95, max(0.005, geological_adjusted))
            
        except Exception as e:
            self.logger.warning(f"Advanced EMD noise estimation failed: {e}")
            return self._basic_noise_estimation(data)
    
    def _analyze_emd_signal_characteristics(self, data: np.ndarray) -> dict:
        """Comprehensive signal analysis for adaptive EMD parameter selection"""
        try:
            # Basic statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            skewness = self._calculate_skewness(data)
            kurtosis = self._calculate_kurtosis(data)
            
            # Stationarity tests
            stationarity_score = self._assess_stationarity(data)
            
            # Frequency content analysis
            dominant_frequencies = self._identify_dominant_frequencies(data)
            bandwidth_ratio = self._calculate_bandwidth_ratio(data)
            
            # Non-linearity and complexity measures
            correlation_dimension = self._estimate_correlation_dimension(data)
            lyapunov_exponent = self._estimate_largest_lyapunov_exponent(data)
            
            # Mode mixing potential assessment
            mode_mixing_risk = self._assess_mode_mixing_risk(data, dominant_frequencies)
            
            # Signal-to-noise ratio preliminary estimate
            snr_estimate = self._preliminary_snr_estimation(data)
            
            # Composite complexity score
            complexity_components = [
                abs(skewness) / 3.0,  # Normalized skewness contribution
                min(abs(kurtosis - 3) / 10.0, 1.0),  # Excess kurtosis contribution
                1.0 - stationarity_score,  # Non-stationarity contribution
                min(correlation_dimension / 5.0, 1.0),  # Dimensional complexity
                min(abs(lyapunov_exponent), 1.0),  # Chaotic behavior
                mode_mixing_risk
            ]
            
            complexity_score = np.mean(complexity_components)
            
            return {
                'mean': mean_val,
                'std': std_val,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'stationarity_score': stationarity_score,
                'dominant_frequencies': dominant_frequencies,
                'bandwidth_ratio': bandwidth_ratio,
                'correlation_dimension': correlation_dimension,
                'lyapunov_exponent': lyapunov_exponent,
                'mode_mixing_risk': mode_mixing_risk,
                'snr_estimate': snr_estimate,
                'complexity_score': complexity_score,
                'signal_length': len(data)
            }
            
        except Exception:
            # Fallback to basic characteristics
            return {
                'mean': np.mean(data),
                'std': np.std(data),
                'complexity_score': 0.5,
                'stationarity_score': 0.5,
                'snr_estimate': 1.0,
                'signal_length': len(data),
                'dominant_frequencies': [0.1],
                'mode_mixing_risk': 0.5
            }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate sample skewness with bias correction"""
        try:
            n = len(data)
            if n < 3:
                return 0.0
            
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            
            if std == 0:
                return 0.0
            
            # Sample skewness
            skew_sum = np.sum(((data - mean) / std) ** 3)
            sample_skew = skew_sum / n
            
            # Bias correction for finite samples
            correction = np.sqrt(n * (n - 1)) / (n - 2)
            corrected_skew = sample_skew * correction
            
            return corrected_skew
            
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate sample kurtosis with bias correction"""
        try:
            n = len(data)
            if n < 4:
                return 3.0  # Normal distribution kurtosis
            
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            
            if std == 0:
                return 3.0
            
            # Sample kurtosis
            kurt_sum = np.sum(((data - mean) / std) ** 4)
            sample_kurt = kurt_sum / n
            
            # Bias correction for finite samples
            correction1 = (n - 1) / ((n - 2) * (n - 3))
            correction2 = (n + 1) * sample_kurt - 3 * (n - 1)
            corrected_kurt = correction1 * correction2 + 3
            
            return corrected_kurt
            
        except Exception:
            return 3.0
    
    def _assess_stationarity(self, data: np.ndarray) -> float:
        """Assess signal stationarity using multiple statistical tests"""
        try:
            if len(data) < 20:
                return 0.5
            
            stationarity_scores = []
            
            # Method 1: Augmented Dickey-Fuller test approximation
            # Test for unit root (non-stationarity)
            try:
                # Simple differencing test
                diff_data = np.diff(data)
                variance_ratio = np.var(diff_data) / max(np.var(data), 1e-10)
                adf_score = min(variance_ratio, 2.0) / 2.0
                stationarity_scores.append(adf_score)
            except:
                pass
            
            # Method 2: KPSS test approximation
            # Test for trend stationarity
            try:
                # Detrend using linear regression
                x = np.arange(len(data))
                coeffs = np.polyfit(x, data, 1)
                trend = np.polyval(coeffs, x)
                detrended = data - trend
                
                # Calculate cumulative sum of squared residuals
                cumsum = np.cumsum(detrended)
                partial_sums = cumsum ** 2
                
                # KPSS statistic approximation
                kpss_stat = np.sum(partial_sums) / (len(data) ** 2 * np.var(detrended))
                kpss_score = max(0, 1.0 - kpss_stat / 0.5)  # Normalize
                stationarity_scores.append(kpss_score)
            except:
                pass
            
            # Method 3: Rolling statistics test
            try:
                window_size = max(10, len(data) // 5)
                n_windows = len(data) // window_size
                
                if n_windows >= 2:
                    rolling_means = []
                    rolling_stds = []
                    
                    for i in range(n_windows):
                        start = i * window_size
                        end = start + window_size
                        window_data = data[start:end]
                        rolling_means.append(np.mean(window_data))
                        rolling_stds.append(np.std(window_data))
                    
                    # Test consistency of means and variances
                    mean_consistency = 1.0 / (1.0 + np.std(rolling_means) / max(np.mean(rolling_means), 1e-10))
                    var_consistency = 1.0 / (1.0 + np.std(rolling_stds) / max(np.mean(rolling_stds), 1e-10))
                    
                    rolling_score = (mean_consistency + var_consistency) / 2.0
                    stationarity_scores.append(rolling_score)
            except:
                pass
            
            # Return combined stationarity score
            if stationarity_scores:
                return np.mean(stationarity_scores)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _identify_dominant_frequencies(self, data: np.ndarray) -> list:
        """Identify dominant frequencies using advanced spectral analysis"""
        try:
            if len(data) < 8:
                return [0.1]
            
            # Zero-padding for better frequency resolution
            n_fft = max(len(data), 2 ** int(np.ceil(np.log2(len(data)) + 2)))
            
            # Apply windowing to reduce spectral leakage
            window = np.hanning(len(data))
            windowed_data = data * window
            
            # Compute power spectral density
            fft_data = np.fft.fft(windowed_data, n=n_fft)
            psd = np.abs(fft_data) ** 2
            
            # Frequency bins (normalize to [0, 0.5])
            freqs = np.fft.fftfreq(n_fft)[:n_fft//2]
            psd = psd[:n_fft//2]
            
            # Find peaks in PSD
            peak_indices = []
            for i in range(1, len(psd) - 1):
                if (psd[i] > psd[i-1] and psd[i] > psd[i+1] and 
                    psd[i] > 0.1 * np.max(psd)):
                    peak_indices.append(i)
            
            # Extract corresponding frequencies
            dominant_freqs = []
            for idx in peak_indices[:5]:  # Top 5 dominant frequencies
                freq = abs(freqs[idx])
                if freq > 0:  # Exclude DC component
                    dominant_freqs.append(freq)
            
            if not dominant_freqs:
                dominant_freqs = [0.1]
            
            return sorted(dominant_freqs)
            
        except Exception:
            return [0.1]
    
    def _calculate_bandwidth_ratio(self, data: np.ndarray) -> float:
        """Calculate effective bandwidth ratio of the signal"""
        try:
            if len(data) < 8:
                return 0.5
            
            # Compute power spectral density
            n_fft = max(len(data), 256)
            fft_data = np.fft.fft(data, n=n_fft)
            psd = np.abs(fft_data) ** 2
            psd = psd[:n_fft//2]  # One-sided spectrum
            
            # Frequency vector
            freqs = np.linspace(0, 0.5, len(psd))
            
            # Calculate spectral centroid (mean frequency)
            total_power = np.sum(psd)
            if total_power == 0:
                return 0.5
            
            spectral_centroid = np.sum(freqs * psd) / total_power
            
            # Calculate spectral spread (bandwidth)
            spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power)
            
            # Bandwidth ratio: normalized spread relative to centroid
            if spectral_centroid > 0:
                bandwidth_ratio = min(spectral_spread / spectral_centroid, 2.0)
            else:
                bandwidth_ratio = 0.5
            
            return bandwidth_ratio
            
        except Exception:
            return 0.5
    
    def _estimate_correlation_dimension(self, data: np.ndarray) -> float:
        """Estimate correlation dimension using Grassberger-Procaccia algorithm"""
        try:
            if len(data) < 50:
                return 1.0
            
            # Parameters for correlation dimension estimation
            max_embedding_dim = min(5, len(data) // 10)
            n_points = min(len(data), 200)  # Subsample for computational efficiency
            
            # Subsample data uniformly
            if len(data) > n_points:
                indices = np.linspace(0, len(data) - 1, n_points).astype(int)
                data_subset = data[indices]
            else:
                data_subset = data.copy()
            
            correlation_dims = []
            
            for embedding_dim in range(1, max_embedding_dim + 1):
                # Create time-delay embedding
                if len(data_subset) <= embedding_dim:
                    break
                
                # Embedding with unit delay
                embedded_vectors = []
                for i in range(len(data_subset) - embedding_dim + 1):
                    vector = data_subset[i:i + embedding_dim]
                    embedded_vectors.append(vector)
                
                if len(embedded_vectors) < 10:
                    break
                
                embedded_vectors = np.array(embedded_vectors)
                
                # Calculate correlation sum for different radii
                distances = []
                n_vectors = len(embedded_vectors)
                
                # Compute pairwise distances (sample subset for efficiency)
                max_pairs = min(1000, n_vectors * (n_vectors - 1) // 2)
                pair_count = 0
                
                for i in range(n_vectors):
                    for j in range(i + 1, n_vectors):
                        if pair_count >= max_pairs:
                            break
                        
                        dist = np.linalg.norm(embedded_vectors[i] - embedded_vectors[j])
                        distances.append(dist)
                        pair_count += 1
                    
                    if pair_count >= max_pairs:
                        break
                
                if not distances:
                    continue
                
                distances = np.array(distances)
                
                # Calculate correlation dimension
                radii = np.percentile(distances, [10, 20, 30, 40, 50])
                log_radii = []
                log_corr_sums = []
                
                for radius in radii:
                    if radius > 0:
                        corr_sum = np.sum(distances <= radius) / len(distances)
                        if corr_sum > 0:
                            log_radii.append(np.log(radius))
                            log_corr_sums.append(np.log(corr_sum))
                
                # Linear regression to estimate slope (correlation dimension)
                if len(log_radii) >= 3:
                    coeffs = np.polyfit(log_radii, log_corr_sums, 1)
                    correlation_dim = coeffs[0]  # Slope
                    correlation_dims.append(max(0, correlation_dim))
            
            if correlation_dims:
                return np.mean(correlation_dims)
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _estimate_largest_lyapunov_exponent(self, data: np.ndarray) -> float:
        """Estimate largest Lyapunov exponent using Wolf's algorithm approximation"""
        try:
            if len(data) < 100:
                return 0.0
            
            # Parameters
            embedding_dim = 3
            delay = 1
            min_separation = 10
            
            # Create time-delay embedding
            if len(data) <= embedding_dim:
                return 0.0
            
            embedded_vectors = []
            for i in range(len(data) - (embedding_dim - 1) * delay):
                vector = []
                for j in range(embedding_dim):
                    vector.append(data[i + j * delay])
                embedded_vectors.append(vector)
            
            embedded_vectors = np.array(embedded_vectors)
            n_vectors = len(embedded_vectors)
            
            if n_vectors < 50:
                return 0.0
            
            # Track evolution of nearby trajectories
            lyapunov_sums = []
            
            # Sample reference points
            n_refs = min(20, n_vectors // 10)
            ref_indices = np.linspace(min_separation, n_vectors - min_separation - 1, n_refs).astype(int)
            
            for ref_idx in ref_indices:
                ref_vector = embedded_vectors[ref_idx]
                
                # Find nearest neighbor
                distances = []
                for i in range(n_vectors):
                    if abs(i - ref_idx) >= min_separation:  # Temporal separation
                        dist = np.linalg.norm(embedded_vectors[i] - ref_vector)
                        distances.append((dist, i))
                
                if not distances:
                    continue
                
                # Sort by distance and select nearest
                distances.sort()
                nearest_dist, nearest_idx = distances[0]
                
                if nearest_dist == 0:
                    continue
                
                # Track trajectory evolution
                max_steps = min(20, n_vectors - max(ref_idx, nearest_idx) - 1)
                
                log_separations = []
                for step in range(1, max_steps):
                    if (ref_idx + step < n_vectors and 
                        nearest_idx + step < n_vectors):
                        
                        evolved_ref = embedded_vectors[ref_idx + step]
                        evolved_nearest = embedded_vectors[nearest_idx + step]
                        
                        separation = np.linalg.norm(evolved_ref - evolved_nearest)
                        
                        if separation > 0:
                            log_separations.append(np.log(separation / nearest_dist))
                
                if len(log_separations) >= 3:
                    # Linear regression to estimate divergence rate
                    time_steps = np.arange(1, len(log_separations) + 1)
                    coeffs = np.polyfit(time_steps, log_separations, 1)
                    lyapunov_sums.append(coeffs[0])  # Slope
            
            if lyapunov_sums:
                return np.mean(lyapunov_sums)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _assess_mode_mixing_risk(self, data: np.ndarray, dominant_frequencies: list) -> float:
        """Assess the risk of mode mixing in EMD decomposition"""
        try:
            if len(data) < 20:
                return 0.5
            
            risk_factors = []
            
            # Factor 1: Frequency proximity
            if len(dominant_frequencies) > 1:
                freq_ratios = []
                sorted_freqs = sorted(dominant_frequencies)
                for i in range(len(sorted_freqs) - 1):
                    if sorted_freqs[i+1] > 0:
                        ratio = sorted_freqs[i] / sorted_freqs[i+1]
                        freq_ratios.append(ratio)
                
                if freq_ratios:
                    # Higher risk when frequencies are close (ratio near 1)
                    proximity_risk = np.mean([abs(r - 0.5) / 0.5 for r in freq_ratios])
                    risk_factors.append(min(proximity_risk, 1.0))
            
            # Factor 2: Signal intermittency
            try:
                # Measure signal intermittency using Hilbert transform
                analytic_signal = np.abs(data + 1j * np.imag(np.fft.ifft(np.fft.fft(data) * 1j * np.sign(np.fft.fftfreq(len(data))))))
                instantaneous_amplitude = np.abs(analytic_signal)
                
                # Calculate amplitude variability
                amp_cv = np.std(instantaneous_amplitude) / max(np.mean(instantaneous_amplitude), 1e-10)
                intermittency_risk = min(amp_cv / 2.0, 1.0)
                risk_factors.append(intermittency_risk)
            except:
                pass
            
            # Factor 3: Non-linearity assessment
            try:
                # Surrogate data test for non-linearity
                original_spectrum = np.abs(np.fft.fft(data))**2
                
                # Create surrogate with same spectrum but randomized phases
                phases = np.random.uniform(0, 2*np.pi, len(data)//2)
                surrogate_fft = np.sqrt(original_spectrum[:len(data)//2]) * np.exp(1j * phases)
                
                # Make it symmetric for real signal
                full_surrogate_fft = np.concatenate([surrogate_fft, np.conj(surrogate_fft[::-1])])
                if len(full_surrogate_fft) > len(data):
                    full_surrogate_fft = full_surrogate_fft[:len(data)]
                
                surrogate = np.real(np.fft.ifft(full_surrogate_fft))
                
                # Compare non-linear measures
                orig_acf = np.correlate(data, data, mode='full')
                surr_acf = np.correlate(surrogate, surrogate, mode='full')
                
                acf_diff = np.std(orig_acf - surr_acf) / max(np.std(orig_acf), 1e-10)
                nonlinearity_risk = min(acf_diff / 5.0, 1.0)
                risk_factors.append(nonlinearity_risk)
            except:
                pass
            
            # Factor 4: Signal length adequacy
            # Shorter signals have higher mode mixing risk
            length_risk = max(0, 1.0 - len(data) / 200.0)
            risk_factors.append(length_risk)
            
            # Combine risk factors
            if risk_factors:
                return np.mean(risk_factors)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _preliminary_snr_estimation(self, data: np.ndarray) -> float:
        """Preliminary signal-to-noise ratio estimation"""
        try:
            if len(data) < 10:
                return 1.0
            
            # Method 1: Spectral approach
            try:
                # Compute power spectral density
                n_fft = max(len(data), 64)
                fft_data = np.fft.fft(data, n=n_fft)
                psd = np.abs(fft_data) ** 2
                psd = psd[:n_fft//2]
                
                # Estimate signal power (dominant peaks)
                sorted_psd = np.sort(psd)[::-1]
                signal_power = np.mean(sorted_psd[:max(1, len(sorted_psd)//10)])
                
                # Estimate noise power (background level)
                noise_power = np.median(psd)
                
                if noise_power > 0:
                    spectral_snr = signal_power / noise_power
                else:
                    spectral_snr = 1.0
                    
            except:
                spectral_snr = 1.0
            
            # Method 2: Time domain approach
            try:
                # Smooth signal using moving average
                window_size = max(3, len(data) // 20)
                kernel = np.ones(window_size) / window_size
                
                # Pad for convolution
                padded_data = np.pad(data, (window_size//2, window_size//2), mode='edge')
                smoothed = np.convolve(padded_data, kernel, mode='valid')
                
                # Estimate signal and noise powers
                signal_power_td = np.var(smoothed)
                noise_power_td = np.var(data - smoothed)
                
                if noise_power_td > 0:
                    temporal_snr = signal_power_td / noise_power_td
                else:
                    temporal_snr = 1.0
                    
            except:
                temporal_snr = 1.0
            
            # Combine estimates
            combined_snr = np.sqrt(spectral_snr * temporal_snr)
            return min(max(combined_snr, 0.1), 100.0)
            
        except Exception:
            return 1.0
    
    def _ensemble_emd_noise_estimation(self, data: np.ndarray, signal_stats: dict) -> float:
        """Ensemble Empirical Mode Decomposition for robust noise estimation"""
        try:
            # EEMD parameters based on signal characteristics
            noise_std = 0.2 * signal_stats['std']  # Standard deviation of added white noise
            n_ensembles = min(max(50, int(100 / signal_stats['complexity_score'])), 200)
            n_modes = min(int(np.log2(len(data))), 12)
            
            ensemble_imfs = []
            ensemble_residues = []
            
            for ensemble_idx in range(n_ensembles):
                # Add white noise realization
                noise_realization = np.random.normal(0, noise_std, len(data))
                noisy_signal = data + noise_realization
                
                # Extract IMFs for this ensemble member
                imfs, residue = self._adaptive_sifting_process(
                    noisy_signal, n_modes, signal_stats
                )
                
                ensemble_imfs.append(imfs)
                ensemble_residues.append(residue)
            
            # Average the ensemble
            if ensemble_imfs:
                # Find minimum number of IMFs across all ensemble members
                min_imfs = min(len(imfs) for imfs in ensemble_imfs)
                
                if min_imfs > 0:
                    # Average corresponding IMFs across ensemble
                    averaged_imfs = []
                    for mode_idx in range(min_imfs):
                        mode_ensemble = np.array([imfs[mode_idx] for imfs in ensemble_imfs])
                        averaged_imf = np.mean(mode_ensemble, axis=0)
                        averaged_imfs.append(averaged_imf)
                    
                    # Noise estimation from ensemble IMFs
                    noise_estimate = self._estimate_noise_from_ensemble_imfs(
                        averaged_imfs, data, signal_stats
                    )
                    
                    return noise_estimate
            
            # Fallback to adaptive EMD
            return self._adaptive_emd_noise_estimation(data, signal_stats)
            
        except Exception:
            return self._adaptive_emd_noise_estimation(data, signal_stats)
    
    def _adaptive_emd_noise_estimation(self, data: np.ndarray, signal_stats: dict) -> float:
        """Adaptive EMD with signal-specific parameter optimization"""
        try:
            # Determine optimal number of modes based on signal characteristics
            max_modes = min(
                int(np.log2(len(data))) + 2,
                int(10 + 5 * signal_stats['complexity_score'])
            )
            
            # Perform adaptive sifting
            imfs, residue = self._adaptive_sifting_process(data, max_modes, signal_stats)
            
            if not imfs:
                return self._basic_noise_estimation(data)
            
            # Multi-criteria noise estimation
            noise_estimates = []
            
            # Method 1: High-frequency IMF analysis
            if len(imfs) > 0:
                hf_noise = self._analyze_high_frequency_imf_noise(
                    imfs[0], data, signal_stats
                )
                noise_estimates.append(('hf_imf', hf_noise, 0.4))
            
            # Method 2: Energy distribution analysis
            energy_noise = self._analyze_imf_energy_distribution(
                imfs, data, signal_stats
            )
            noise_estimates.append(('energy', energy_noise, 0.3))
            
            # Method 3: Instantaneous frequency analysis
            if len(imfs) >= 2:
                if_noise = self._analyze_instantaneous_frequency_noise(
                    imfs[:2], signal_stats
                )
                noise_estimates.append(('inst_freq', if_noise, 0.2))
            
            # Method 4: Cross-correlation analysis between adjacent IMFs
            if len(imfs) >= 3:
                corr_noise = self._analyze_imf_cross_correlation_noise(
                    imfs[:3], signal_stats
                )
                noise_estimates.append(('correlation', corr_noise, 0.1))
            
            # Weighted combination of estimates
            if noise_estimates:
                total_weight = sum(weight for _, _, weight in noise_estimates)
                weighted_noise = sum(
                    estimate * weight for _, estimate, weight in noise_estimates
                ) / total_weight
                
                return weighted_noise
            
            return self._basic_noise_estimation(data)
            
        except Exception:
            return self._basic_noise_estimation(data)
    
    def _adaptive_sifting_process(self, signal: np.ndarray, max_modes: int, signal_stats: dict) -> tuple:
        """Advanced sifting process with adaptive stopping criteria"""
        try:
            imfs = []
            residue = signal.copy()
            
            # Adaptive stopping criteria based on signal characteristics
            energy_threshold = 0.001 * signal_stats.get('std', 1.0) ** 2
            correlation_threshold = 0.1 * signal_stats.get('complexity_score', 0.5)
            
            for mode_idx in range(max_modes):
                if len(residue) < 10:
                    break
                
                # Extract single IMF with adaptive sifting
                imf = self._extract_single_imf_adaptive(
                    residue, signal_stats, mode_idx
                )
                
                if imf is None:
                    break
                
                # Validate IMF quality
                if self._validate_imf_quality(imf, residue, signal_stats):
                    imfs.append(imf)
                    residue = residue - imf
                    
                    # Check stopping criteria
                    residue_energy = np.var(residue)
                    if residue_energy < energy_threshold:
                        break
                    
                    # Check for mode mixing or over-decomposition
                    if len(imfs) > 1:
                        correlation = np.corrcoef(imfs[-1], imfs[-2])[0, 1]
                        if abs(correlation) > (1.0 - correlation_threshold):
                            break
                else:
                    break
            
            return imfs, residue
            
        except Exception:
            return [], signal
    
    def _extract_single_imf_adaptive(self, signal: np.ndarray, signal_stats: dict, mode_idx: int) -> np.ndarray:
        """Extract single IMF with adaptive parameters and advanced envelope construction"""
        try:
            if len(signal) < 6:
                return None
            
            current_signal = signal.copy()
            
            # Adaptive sifting parameters
            max_iterations = min(50, 10 + mode_idx * 5)
            convergence_threshold = 0.001 * signal_stats.get('std', 1.0)
            
            # Enhanced extrema detection parameters
            min_extrema_separation = max(1, int(len(signal) / 100))
            
            for iteration in range(max_iterations):
                # Advanced extrema detection with noise robustness
                maxima_indices, minima_indices = self._detect_robust_extrema(
                    current_signal, min_extrema_separation
                )
                
                if len(maxima_indices) < 2 or len(minima_indices) < 2:
                    break
                
                # Advanced envelope construction using spline interpolation
                try:
                    upper_envelope, lower_envelope = self._construct_advanced_envelopes(
                        current_signal, maxima_indices, minima_indices
                    )
                    
                    # Mean envelope
                    mean_envelope = (upper_envelope + lower_envelope) / 2.0
                    
                    # Update current signal
                    previous_signal = current_signal.copy()
                    current_signal = current_signal - mean_envelope
                    
                    # Check IMF convergence criteria
                    if self._check_imf_convergence(
                        current_signal, previous_signal, convergence_threshold
                    ):
                        break
                        
                except Exception:
                    break
            
            # Validate extracted IMF
            if self._is_valid_imf(current_signal):
                return current_signal
            else:
                return None
                
        except Exception:
            return None
    
    def _detect_robust_extrema(self, signal: np.ndarray, min_separation: int) -> tuple:
        """Robust extrema detection with noise filtering"""
        try:
            # Pre-filtering for noise reduction
            if len(signal) > 10:
                # Apply light smoothing to reduce noise impact
                kernel_size = min(5, len(signal) // 20)
                if kernel_size >= 3:
                    kernel = np.ones(kernel_size) / kernel_size
                    padded = np.pad(signal, (kernel_size//2, kernel_size//2), mode='edge')
                    filtered_signal = np.convolve(padded, kernel, mode='valid')
                else:
                    filtered_signal = signal.copy()
            else:
                filtered_signal = signal.copy()
            
            maxima_indices = []
            minima_indices = []
            
            # Enhanced extrema detection
            for i in range(1, len(filtered_signal) - 1):
                # Local maximum detection
                if (filtered_signal[i] > filtered_signal[i-1] and 
                    filtered_signal[i] > filtered_signal[i+1]):
                    
                    # Check separation from previous extrema
                    if (not maxima_indices or 
                        i - maxima_indices[-1] >= min_separation):
                        maxima_indices.append(i)
                
                # Local minimum detection
                elif (filtered_signal[i] < filtered_signal[i-1] and 
                      filtered_signal[i] < filtered_signal[i+1]):
                    
                    # Check separation from previous extrema
                    if (not minima_indices or 
                        i - minima_indices[-1] >= min_separation):
                        minima_indices.append(i)
            
            # Ensure boundary extrema are included if appropriate
            if len(maxima_indices) > 0 or len(minima_indices) > 0:
                # Add boundary points if they are extremal
                if (len(maxima_indices) == 0 or 
                    (len(maxima_indices) > 0 and 0 not in maxima_indices and
                     signal[0] > signal[1])):
                    maxima_indices = [0] + maxima_indices
                
                if (len(maxima_indices) == 0 or 
                    (len(maxima_indices) > 0 and len(signal)-1 not in maxima_indices and
                     signal[-1] > signal[-2])):
                    maxima_indices = maxima_indices + [len(signal)-1]
                
                if (len(minima_indices) == 0 or 
                    (len(minima_indices) > 0 and 0 not in minima_indices and
                     signal[0] < signal[1])):
                    minima_indices = [0] + minima_indices
                
                if (len(minima_indices) == 0 or 
                    (len(minima_indices) > 0 and len(signal)-1 not in minima_indices and
                     signal[-1] < signal[-2])):
                    minima_indices = minima_indices + [len(signal)-1]
            
            return maxima_indices, minima_indices
            
        except Exception:
            return [], []
    
    def _construct_advanced_envelopes(self, signal: np.ndarray, maxima_indices: list, minima_indices: list) -> tuple:
        """Construct advanced envelopes using cubic spline interpolation with extrapolation"""
        try:
            x = np.arange(len(signal))
            
            # Upper envelope (maxima)
            if len(maxima_indices) >= 2:
                # Use cubic spline interpolation
                max_values = signal[maxima_indices]
                upper_envelope = self._cubic_spline_interpolation(
                    maxima_indices, max_values, x
                )
            else:
                # Fallback to linear interpolation or constant
                if len(maxima_indices) == 1:
                    upper_envelope = np.full(len(signal), signal[maxima_indices[0]])
                else:
                    upper_envelope = np.zeros(len(signal))
            
            # Lower envelope (minima)
            if len(minima_indices) >= 2:
                # Use cubic spline interpolation
                min_values = signal[minima_indices]
                lower_envelope = self._cubic_spline_interpolation(
                    minima_indices, min_values, x
                )
            else:
                # Fallback to linear interpolation or constant
                if len(minima_indices) == 1:
                    lower_envelope = np.full(len(signal), signal[minima_indices[0]])
                else:
                    lower_envelope = np.zeros(len(signal))
            
            return upper_envelope, lower_envelope
            
        except Exception:
            # Fallback to simple interpolation
            if len(maxima_indices) > 0 and len(minima_indices) > 0:
                x = np.arange(len(signal))
                upper_envelope = np.interp(x, maxima_indices, signal[maxima_indices])
                lower_envelope = np.interp(x, minima_indices, signal[minima_indices])
                return upper_envelope, lower_envelope
            else:
                return np.zeros(len(signal)), np.zeros(len(signal))
    
    def _cubic_spline_interpolation(self, x_points: list, y_points: list, x_interp: np.ndarray) -> np.ndarray:
        """Custom cubic spline interpolation with natural boundary conditions"""
        try:
            n = len(x_points)
            if n < 2:
                return np.zeros(len(x_interp))
            
            x_points = np.array(x_points, dtype=float)
            y_points = np.array(y_points, dtype=float)
            
            if n == 2:
                # Linear interpolation for 2 points
                return np.interp(x_interp, x_points, y_points)
            
            # Calculate differences
            h = np.diff(x_points)
            b = np.diff(y_points) / h
            
            # Set up tridiagonal system for natural spline
            A = np.zeros((n, n))
            rhs = np.zeros(n)
            
            # Natural boundary conditions (second derivative = 0)
            A[0, 0] = 1
            A[n-1, n-1] = 1
            rhs[0] = 0
            rhs[n-1] = 0
            
            # Interior points
            for i in range(1, n-1):
                A[i, i-1] = h[i-1]
                A[i, i] = 2 * (h[i-1] + h[i])
                A[i, i+1] = h[i]
                rhs[i] = 6 * (b[i] - b[i-1])
            
            # Solve for second derivatives
            try:
                c = np.linalg.solve(A, rhs)
            except:
                # Fallback to linear interpolation
                return np.interp(x_interp, x_points, y_points)
            
            # Evaluate spline at interpolation points
            result = np.zeros(len(x_interp))
            
            for i, x_val in enumerate(x_interp):
                # Find interval
                if x_val <= x_points[0]:
                    # Extrapolation below
                    j = 0
                elif x_val >= x_points[-1]:
                    # Extrapolation above
                    j = n - 2
                else:
                    # Interpolation
                    j = np.searchsorted(x_points[1:], x_val)
                
                # Evaluate cubic polynomial
                dx = x_val - x_points[j]
                dx1 = x_points[j+1] - x_val
                
                result[i] = (c[j] * dx1**3 + c[j+1] * dx**3) / (6 * h[j]) + \
                           (y_points[j] - c[j] * h[j]**2 / 6) * dx1 / h[j] + \
                           (y_points[j+1] - c[j+1] * h[j]**2 / 6) * dx / h[j]
            
            return result
            
        except Exception:
            # Final fallback
            return np.interp(x_interp, x_points, y_points)
    
    def _basic_noise_estimation(self, data: np.ndarray) -> float:
        """Fallback basic noise estimation method"""
        try:
            if len(data) < 3:
                return 0.2
                
            # Method 1: Difference-based approach
            diffs = np.diff(data)
            diff_std = np.std(diffs)
            data_std = np.std(data)
            
            if data_std > 0:
                diff_noise = diff_std / (data_std * np.sqrt(2))
            else:
                diff_noise = 0.2
                
            # Method 2: Local variability
            if len(data) >= 5:
                # Moving standard deviation
                window_size = min(5, len(data) // 2)
                local_stds = []
                
                for i in range(len(data) - window_size + 1):
                    window = data[i:i + window_size]
                    local_stds.append(np.std(window))
                    
                if local_stds:
                    local_variability = np.mean(local_stds)
                    if data_std > 0:
                        variability_noise = local_variability / data_std
                    else:
                        variability_noise = 0.2
                else:
                    variability_noise = diff_noise
            else:
                variability_noise = diff_noise
                
            # Combine estimates
            combined_noise = (diff_noise + variability_noise) / 2
            
            return min(0.8, max(0.05, combined_noise))
            
        except Exception:
            return 0.2
    
    def _combine_noise_estimates(self, estimates: Dict, data: np.ndarray) -> float:
        """Combine multiple noise estimates using weighted confidence intervals"""
        try:
            if not estimates:
                return 0.2
                
            # Calculate weighted average based on confidence and data characteristics
            total_weighted_value = 0.0
            total_weight = 0.0
            
            for method, info in estimates.items():
                value = info['value']
                base_weight = info['weight']
                confidence = info['confidence']
                
                # Adjust weight based on data characteristics
                n = len(data)
                
                # Bonus weight for methods better suited to data size
                if method == 'dfa' and n >= 50:
                    size_bonus = 1.2
                elif method == 'wavelet' and 16 <= n <= 200:
                    size_bonus = 1.1
                elif method == 'spectral' and n >= 64:
                    size_bonus = 1.15
                elif method == 'robust' and n >= 20:
                    size_bonus = 1.05
                else:
                    size_bonus = 1.0
                    
                # Final weight incorporating confidence and size bonus
                final_weight = base_weight * confidence * size_bonus
                
                total_weighted_value += value * final_weight
                total_weight += final_weight
            
            if total_weight > 0:
                weighted_average = total_weighted_value / total_weight
                
                # Apply consistency check: if estimates vary widely, increase uncertainty
                values = [info['value'] for info in estimates.values()]
                if len(values) > 1:
                    value_std = np.std(values)
                    value_mean = np.mean(values)
                    
                    # If coefficient of variation is high, add uncertainty
                    if value_mean > 0 and value_std / value_mean > 0.5:
                        uncertainty_penalty = min(0.1, (value_std / value_mean - 0.5) * 0.2)
                        weighted_average = min(0.9, weighted_average + uncertainty_penalty)
                
                return weighted_average
            else:
                # Fallback: simple average
                values = [info['value'] for info in estimates.values()]
                return np.mean(values)
                
        except Exception:
            return 0.2
    
    def _apply_geological_noise_context(self, base_noise: float, data: np.ndarray, analysis_method: str) -> float:
        """Apply geological context to adjust noise estimate"""
        try:
            # Geological data types typically have different noise characteristics
            adjusted_noise = base_noise
            
            # Check data characteristics that suggest geological signal types
            data_range = np.ptp(data)  # peak-to-peak
            data_mean = np.abs(np.mean(data))
            
            # Temperature data (typically smooth, low noise expected)
            if data_range > 100 and data_mean > 200:  # Likely temperature in Kelvin
                adjusted_noise *= 0.8  # Reduce noise estimate
                
            # Pressure data (can have measurement noise)
            elif data_range > 1e6 and data_mean > 1e5:  # Likely pressure in Pa
                adjusted_noise *= 1.1  # Slightly increase noise estimate
                
            # Seismic velocity data (typically well-measured)
            elif 1000 <= data_mean <= 10000 and data_range < data_mean * 2:  # Likely velocity in m/s
                adjusted_noise *= 0.9  # Reduce noise estimate
                
            # Composition data (can have significant measurement uncertainty)
            elif 0 <= data_mean <= 1 and data_range <= 1:  # Likely fractional composition
                adjusted_noise *= 1.2  # Increase noise estimate
                
            # Very small values (likely normalized or fractional data)
            elif data_mean < 0.1 and data_range < 1:
                adjusted_noise *= 1.1  # Slightly increase noise estimate
                
            # Apply method-specific adjustments
            if analysis_method == 'spectral_primary':
                # Spectral analysis may overestimate noise in geological signals
                adjusted_noise *= 0.95
            elif analysis_method == 'dfa_primary':
                # DFA tends to be reliable for geological time series
                adjusted_noise *= 1.0  # No adjustment
            elif analysis_method == 'wavelet_primary':
                # Wavelets are good for non-stationary geological signals
                adjusted_noise *= 0.98
                
            return min(0.9, max(0.01, adjusted_noise))
            
        except Exception:
            return base_noise

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