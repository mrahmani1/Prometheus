"""
OSS Vizier Integration for Hyperparameter Optimization and Model Selection
Advanced Bayesian optimization and automated machine learning pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import asyncio
import json
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback

# Vizier imports
try:
    from vizier import pyvizier as vz
    from vizier import algorithms as vza
    from vizier import service
    from vizier.service import clients
    VIZIER_AVAILABLE = True
except ImportError:
    VIZIER_AVAILABLE = False
    warnings.warn("Google Vizier not available. Using fallback optimization.")

# ML and optimization imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')

class OptimizationObjective(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

class ParameterType(Enum):
    DOUBLE = "double"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    DISCRETE = "discrete"

@dataclass
class OptimizationParameter:
    """Parameter specification for optimization"""
    name: str
    type: ParameterType
    bounds: Optional[Tuple[float, float]] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    description: str = ""
    
    def __post_init__(self):
        if self.type in [ParameterType.DOUBLE, ParameterType.INTEGER] and self.bounds is None:
            raise ValueError(f"Bounds required for {self.type.value} parameter {self.name}")
        if self.type == ParameterType.CATEGORICAL and self.choices is None:
            raise ValueError(f"Choices required for categorical parameter {self.name}")

@dataclass
class OptimizationResult:
    """Result from optimization run"""
    best_parameters: Dict[str, Any]
    best_objective_value: float
    n_trials: int
    optimization_history: List[Tuple[Dict[str, Any], float]]
    convergence_metrics: Dict[str, float]
    elapsed_time: float
    algorithm_info: Dict[str, Any]

class ModelPerformanceTracker:
    """Track model performance across different parameter configurations"""
    
    def __init__(self):
        self.performance_history = []
        self.parameter_importance = {}
        self.convergence_data = []
        
    def add_evaluation(self, parameters: Dict[str, Any], 
                      performance_metrics: Dict[str, float],
                      metadata: Dict[str, Any] = None):
        """Add performance evaluation"""
        evaluation = {
            'parameters': parameters.copy(),
            'metrics': performance_metrics.copy(),
            'timestamp': time.time(),
            'trial_id': len(self.performance_history),
            'metadata': metadata or {}
        }
        self.performance_history.append(evaluation)
        
        # Update convergence tracking
        if 'primary_metric' in performance_metrics:
            self.convergence_data.append(performance_metrics['primary_metric'])
    
    def get_best_configuration(self, metric: str = 'primary_metric', 
                             minimize: bool = True) -> Dict[str, Any]:
        """Get best parameter configuration"""
        if not self.performance_history:
            return {}
        
        best_eval = min(self.performance_history, 
                       key=lambda x: x['metrics'].get(metric, float('inf')) if minimize 
                                   else -x['metrics'].get(metric, -float('inf')))
        return best_eval
    
    def analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze parameter importance using correlation"""
        if len(self.performance_history) < 10:
            return {}
        
        # Convert to DataFrame for analysis
        param_data = []
        performance_data = []
        
        for eval_data in self.performance_history:
            param_data.append(eval_data['parameters'])
            performance_data.append(eval_data['metrics'].get('primary_metric', 0))
        
        df_params = pd.DataFrame(param_data)
        df_performance = pd.Series(performance_data)
        
        # Calculate correlations for numerical parameters
        importance = {}
        for col in df_params.select_dtypes(include=[np.number]).columns:
            try:
                correlation = abs(df_params[col].corr(df_performance))
                importance[col] = correlation if not np.isnan(correlation) else 0.0
            except:
                importance[col] = 0.0
        
        self.parameter_importance = importance
        return importance
    
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """Analyze optimization convergence"""
        if len(self.convergence_data) < 5:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        values = np.array(self.convergence_data)
        
        # Simple convergence check: improvement rate in last 20% of trials
        last_portion = max(5, len(values) // 5)
        recent_values = values[-last_portion:]
        early_values = values[:-last_portion] if len(values) > last_portion else values[:last_portion]
        
        recent_improvement = abs(recent_values[-1] - recent_values[0]) / len(recent_values)
        early_improvement = abs(early_values[-1] - early_values[0]) / len(early_values) if len(early_values) > 1 else recent_improvement
        
        convergence_ratio = recent_improvement / (early_improvement + 1e-10)
        
        return {
            'converged': convergence_ratio < 0.1,
            'convergence_ratio': convergence_ratio,
            'best_value': float(np.min(values)),
            'improvement_trend': float(np.mean(np.diff(values[-10:]))),
            'stability': float(np.std(recent_values)),
            'total_trials': len(values)
        }

class VizierOptimizer:
    """OSS Vizier-based hyperparameter optimizer"""
    
    def __init__(self, study_config: Dict[str, Any] = None):
        self.study_config = study_config or self._get_default_config()
        self.performance_tracker = ModelPerformanceTracker()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Vizier components if available
        if VIZIER_AVAILABLE:
            self._setup_vizier()
        else:
            self._setup_fallback()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default optimization configuration"""
        return {
            'algorithm': 'GAUSSIAN_PROCESS_BANDIT',
            'n_trials': 100,
            'n_parallel': 4,
            'acquisition_function': 'EXPECTED_IMPROVEMENT',
            'random_seed': 42,
            'early_stopping': {
                'enabled': True,
                'patience': 20,
                'min_improvement': 1e-4
            },
            'fallback_algorithm': 'differential_evolution'  # 'optuna', 'differential_evolution', 'random'
        }
    
    def _setup_vizier(self):
        """Setup Vizier optimization components"""
        if not VIZIER_AVAILABLE:
            return
        
        # Create study configuration
        self.problem = vz.ProblemStatement()
        self.study_owner = 'prometheus_optimizer'
        self.study_name = f'geological_optimization_{int(time.time())}'
        
        # Will be configured when parameters are added
        self.vizier_client = None
        self.vizier_study = None
    
    def _setup_fallback(self):
        """Setup fallback optimization when Vizier is not available"""
        self.fallback_optimizer = None
        self.optuna_study = None
        
        if self.study_config['fallback_algorithm'] == 'optuna':
            try:
                import optuna
                self.optuna_study = optuna.create_study(
                    direction='minimize',
                    sampler=optuna.samplers.TPESampler(seed=self.study_config['random_seed'])
                )
            except ImportError:
                self.logger.warning("Optuna not available, using differential evolution")
                self.study_config['fallback_algorithm'] = 'differential_evolution'
    
    def add_parameter(self, parameter: OptimizationParameter):
        """Add optimization parameter"""
        if VIZIER_AVAILABLE and self.problem is not None:
            self._add_vizier_parameter(parameter)
        else:
            self._add_fallback_parameter(parameter)
    
    def _add_vizier_parameter(self, parameter: OptimizationParameter):
        """Add parameter to Vizier problem"""
        if parameter.type == ParameterType.DOUBLE:
            if parameter.log_scale:
                self.problem.search_space.root.add_float_param(
                    parameter.name,
                    min_value=parameter.bounds[0],
                    max_value=parameter.bounds[1],
                    scale_type=vz.ScaleType.LOG
                )
            else:
                self.problem.search_space.root.add_float_param(
                    parameter.name,
                    min_value=parameter.bounds[0],
                    max_value=parameter.bounds[1]
                )
        elif parameter.type == ParameterType.INTEGER:
            self.problem.search_space.root.add_int_param(
                parameter.name,
                min_value=int(parameter.bounds[0]),
                max_value=int(parameter.bounds[1])
            )
        elif parameter.type == ParameterType.CATEGORICAL:
            self.problem.search_space.root.add_categorical_param(
                parameter.name,
                feasible_values=parameter.choices
            )
        elif parameter.type == ParameterType.DISCRETE:
            self.problem.search_space.root.add_discrete_param(
                parameter.name,
                feasible_values=parameter.choices
            )
    
    def _add_fallback_parameter(self, parameter: OptimizationParameter):
        """Add parameter for fallback optimization"""
        # Store parameter info for later use
        if not hasattr(self, 'fallback_parameters'):
            self.fallback_parameters = []
        self.fallback_parameters.append(parameter)
    
    def set_objective(self, objective: OptimizationObjective, 
                     metric_name: str = 'objective'):
        """Set optimization objective"""
        if VIZIER_AVAILABLE and self.problem is not None:
            if objective == OptimizationObjective.MINIMIZE:
                self.problem.metric_information.append(
                    vz.MetricInformation(name=metric_name, goal=vz.ObjectiveMetricGoal.MINIMIZE)
                )
            else:
                self.problem.metric_information.append(
                    vz.MetricInformation(name=metric_name, goal=vz.ObjectiveMetricGoal.MAXIMIZE)
                )
        
        self.objective_type = objective
        self.metric_name = metric_name
    
    async def optimize(self, 
                      objective_function: Callable[[Dict[str, Any]], float],
                      n_trials: Optional[int] = None,
                      timeout: Optional[float] = None) -> OptimizationResult:
        """Run optimization"""
        n_trials = n_trials or self.study_config['n_trials']
        start_time = time.time()
        
        if VIZIER_AVAILABLE:
            result = await self._optimize_with_vizier(objective_function, n_trials, timeout)
        else:
            result = await self._optimize_with_fallback(objective_function, n_trials, timeout)
        
        result.elapsed_time = time.time() - start_time
        return result
    
    async def _optimize_with_vizier(self, 
                                  objective_function: Callable[[Dict[str, Any]], float],
                                  n_trials: int,
                                  timeout: Optional[float]) -> OptimizationResult:
        """Optimize using Vizier"""
        # Initialize Vizier study
        self.vizier_client = clients.create_or_load_study(
            study_config=vz.StudyConfig.from_problem(self.problem),
            study_id=self.study_name,
            owner=self.study_owner
        )
        
        algorithm_name = self.study_config['algorithm']
        algorithm = vza.create_designer(algorithm_name=algorithm_name)
        
        best_value = float('inf') if self.objective_type == OptimizationObjective.MINIMIZE else float('-inf')
        best_parameters = {}
        optimization_history = []
        
        early_stopping_counter = 0
        patience = self.study_config['early_stopping']['patience']
        min_improvement = self.study_config['early_stopping']['min_improvement']
        
        for trial_id in range(n_trials):
            if timeout and time.time() - start_time > timeout:
                break
            
            # Get suggestions from algorithm
            suggestions = algorithm.suggest(count=1)
            
            if not suggestions:
                break
            
            suggestion = suggestions[0]
            
            # Extract parameters
            parameters = {}
            for param_name, param_value in suggestion.parameters.items():
                parameters[param_name] = param_value.value
            
            try:
                # Evaluate objective function
                if asyncio.iscoroutinefunction(objective_function):
                    objective_value = await objective_function(parameters)
                else:
                    objective_value = objective_function(parameters)
                
                # Create measurement
                measurement = vz.Measurement(
                    metrics={self.metric_name: objective_value}
                )
                
                # Complete trial
                trial = suggestion.to_trial()
                trial.complete(measurement)
                
                # Update best value
                is_better = (
                    (self.objective_type == OptimizationObjective.MINIMIZE and objective_value < best_value) or
                    (self.objective_type == OptimizationObjective.MAXIMIZE and objective_value > best_value)
                )
                
                if is_better:
                    improvement = abs(objective_value - best_value)
                    if improvement > min_improvement:
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                    
                    best_value = objective_value
                    best_parameters = parameters.copy()
                else:
                    early_stopping_counter += 1
                
                # Track performance
                self.performance_tracker.add_evaluation(
                    parameters,
                    {'primary_metric': objective_value, self.metric_name: objective_value},
                    {'trial_id': trial_id, 'algorithm': algorithm_name}
                )
                
                optimization_history.append((parameters.copy(), objective_value))
                
                self.logger.info(f"Trial {trial_id}: {objective_value:.6f} (best: {best_value:.6f})")
                
                # Early stopping
                if (self.study_config['early_stopping']['enabled'] and 
                    early_stopping_counter >= patience and trial_id > patience):
                    self.logger.info(f"Early stopping at trial {trial_id}")
                    break
                
            except Exception as e:
                self.logger.error(f"Trial {trial_id} failed: {str(e)}")
                # Handle failed trials
                trial = suggestion.to_trial()
                trial.complete(vz.Measurement(
                    metrics={self.metric_name: float('inf') if self.objective_type == OptimizationObjective.MINIMIZE else float('-inf')}
                ))
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_objective_value=best_value,
            n_trials=len(optimization_history),
            optimization_history=optimization_history,
            convergence_metrics=self.performance_tracker.get_convergence_analysis(),
            elapsed_time=0,  # Will be set by calling function
            algorithm_info={'algorithm': algorithm_name, 'backend': 'vizier'}
        )
    
    async def _optimize_with_fallback(self, 
                                    objective_function: Callable[[Dict[str, Any]], float],
                                    n_trials: int,
                                    timeout: Optional[float]) -> OptimizationResult:
        """Optimize using fallback methods"""
        algorithm = self.study_config['fallback_algorithm']
        
        if algorithm == 'optuna' and self.optuna_study is not None:
            return await self._optimize_with_optuna(objective_function, n_trials)
        elif algorithm == 'differential_evolution':
            return await self._optimize_with_differential_evolution(objective_function, n_trials)
        else:
            return await self._optimize_with_random_search(objective_function, n_trials)
    
    async def _optimize_with_optuna(self, 
                                  objective_function: Callable[[Dict[str, Any]], float],
                                  n_trials: int) -> OptimizationResult:
        """Optimize using Optuna"""
        def optuna_objective(trial):
            parameters = {}
            
            for param in self.fallback_parameters:
                if param.type == ParameterType.DOUBLE:
                    if param.log_scale:
                        parameters[param.name] = trial.suggest_loguniform(
                            param.name, param.bounds[0], param.bounds[1]
                        )
                    else:
                        parameters[param.name] = trial.suggest_uniform(
                            param.name, param.bounds[0], param.bounds[1]
                        )
                elif param.type == ParameterType.INTEGER:
                    parameters[param.name] = trial.suggest_int(
                        param.name, int(param.bounds[0]), int(param.bounds[1])
                    )
                elif param.type == ParameterType.CATEGORICAL:
                    parameters[param.name] = trial.suggest_categorical(
                        param.name, param.choices
                    )
            
            # Track performance
            if asyncio.iscoroutinefunction(objective_function):
                # For async functions, we need to handle this differently
                return asyncio.create_task(objective_function(parameters))
            else:
                objective_value = objective_function(parameters)
                
                self.performance_tracker.add_evaluation(
                    parameters,
                    {'primary_metric': objective_value},
                    {'trial_id': trial.number, 'algorithm': 'optuna'}
                )
                
                return objective_value
        
        # Run optimization
        self.optuna_study.optimize(optuna_objective, n_trials=n_trials)
        
        # Extract results
        best_trial = self.optuna_study.best_trial
        
        optimization_history = []
        for trial in self.optuna_study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                optimization_history.append((trial.params, trial.value))
        
        return OptimizationResult(
            best_parameters=best_trial.params,
            best_objective_value=best_trial.value,
            n_trials=len(optimization_history),
            optimization_history=optimization_history,
            convergence_metrics=self.performance_tracker.get_convergence_analysis(),
            elapsed_time=0,
            algorithm_info={'algorithm': 'optuna', 'backend': 'optuna'}
        )
    
    async def _optimize_with_differential_evolution(self, 
                                                  objective_function: Callable[[Dict[str, Any]], float],
                                                  n_trials: int) -> OptimizationResult:
        """Optimize using differential evolution"""
        # Prepare bounds for numerical parameters only
        bounds = []
        param_names = []
        
        for param in self.fallback_parameters:
            if param.type in [ParameterType.DOUBLE, ParameterType.INTEGER]:
                bounds.append(param.bounds)
                param_names.append(param.name)
        
        if not bounds:
            raise ValueError("Differential evolution requires at least one numerical parameter")
        
        optimization_history = []
        
        def de_objective(x):
            parameters = {}
            for i, (name, param) in enumerate(zip(param_names, self.fallback_parameters)):
                if param.type == ParameterType.INTEGER:
                    parameters[name] = int(round(x[i]))
                else:
                    parameters[name] = x[i]
            
            # Add categorical parameters with default values
            for param in self.fallback_parameters:
                if param.type == ParameterType.CATEGORICAL and param.name not in parameters:
                    parameters[param.name] = param.choices[0]
            
            objective_value = objective_function(parameters)
            optimization_history.append((parameters.copy(), objective_value))
            
            self.performance_tracker.add_evaluation(
                parameters,
                {'primary_metric': objective_value},
                {'algorithm': 'differential_evolution'}
            )
            
            return objective_value
        
        # Run differential evolution
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=n_trials // 15,  # Approximate trials per generation
            popsize=15,
            seed=self.study_config['random_seed']
        )
        
        # Reconstruct best parameters
        best_parameters = {}
        for i, (name, param) in enumerate(zip(param_names, self.fallback_parameters)):
            if param.type == ParameterType.INTEGER:
                best_parameters[name] = int(round(result.x[i]))
            else:
                best_parameters[name] = result.x[i]
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_objective_value=result.fun,
            n_trials=len(optimization_history),
            optimization_history=optimization_history,
            convergence_metrics=self.performance_tracker.get_convergence_analysis(),
            elapsed_time=0,
            algorithm_info={'algorithm': 'differential_evolution', 'backend': 'scipy'}
        )
    
    async def _optimize_with_random_search(self, 
                                         objective_function: Callable[[Dict[str, Any]], float],
                                         n_trials: int) -> OptimizationResult:
        """Optimize using random search"""
        best_value = float('inf') if self.objective_type == OptimizationObjective.MINIMIZE else float('-inf')
        best_parameters = {}
        optimization_history = []
        
        for trial_id in range(n_trials):
            # Generate random parameters
            parameters = {}
            for param in self.fallback_parameters:
                if param.type == ParameterType.DOUBLE:
                    if param.log_scale:
                        log_min, log_max = np.log(param.bounds[0]), np.log(param.bounds[1])
                        parameters[param.name] = np.exp(np.random.uniform(log_min, log_max))
                    else:
                        parameters[param.name] = np.random.uniform(param.bounds[0], param.bounds[1])
                elif param.type == ParameterType.INTEGER:
                    parameters[param.name] = np.random.randint(param.bounds[0], param.bounds[1] + 1)
                elif param.type == ParameterType.CATEGORICAL:
                    parameters[param.name] = np.random.choice(param.choices)
            
            # Evaluate
            objective_value = objective_function(parameters)
            optimization_history.append((parameters.copy(), objective_value))
            
            # Update best
            is_better = (
                (self.objective_type == OptimizationObjective.MINIMIZE and objective_value < best_value) or
                (self.objective_type == OptimizationObjective.MAXIMIZE and objective_value > best_value)
            )
            
            if is_better:
                best_value = objective_value
                best_parameters = parameters.copy()
            
            self.performance_tracker.add_evaluation(
                parameters,
                {'primary_metric': objective_value},
                {'trial_id': trial_id, 'algorithm': 'random_search'}
            )
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_objective_value=best_value,
            n_trials=n_trials,
            optimization_history=optimization_history,
            convergence_metrics=self.performance_tracker.get_convergence_analysis(),
            elapsed_time=0,
            algorithm_info={'algorithm': 'random_search', 'backend': 'numpy'}
        )

class AutoMLModelSelector:
    """Automated model selection using Vizier optimization"""
    
    def __init__(self, models: Dict[str, Any], metrics: List[str] = None):
        self.models = models
        self.metrics = metrics or ['accuracy', 'f1_score', 'roc_auc']
        self.optimization_results = {}
        self.best_models = {}
        self.logger = logging.getLogger(__name__)
    
    def _create_model_objective(self, model_name: str, model_class: Any, 
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Callable:
        """Create objective function for model hyperparameter optimization"""
        
        def objective(params: Dict[str, Any]) -> float:
            try:
                # Create model with parameters
                model = model_class(**params)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                y_pred = model.predict(X_val)
                
                # Calculate primary metric (assumes regression for simplicity)
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                # Return negative R² (to minimize)
                return -r2
                
            except Exception as e:
                self.logger.error(f"Model evaluation failed: {str(e)}")
                return float('inf')
        
        return objective
    
    async def optimize_model(self, model_name: str, model_class: Any,
                           parameter_space: List[OptimizationParameter],
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           n_trials: int = 50) -> OptimizationResult:
        """Optimize hyperparameters for a specific model"""
        
        # Create optimizer
        optimizer = VizierOptimizer()
        
        # Add parameters
        for param in parameter_space:
            optimizer.add_parameter(param)
        
        # Set objective
        optimizer.set_objective(OptimizationObjective.MINIMIZE, 'validation_loss')
        
        # Create objective function
        objective_fn = self._create_model_objective(
            model_name, model_class, X_train, y_train, X_val, y_val
        )
        
        # Run optimization
        result = await optimizer.optimize(objective_fn, n_trials=n_trials)
        
        # Store results
        self.optimization_results[model_name] = result
        
        # Train best model
        best_model = model_class(**result.best_parameters)
        best_model.fit(X_train, y_train)
        self.best_models[model_name] = best_model
        
        return result
    
    async def optimize_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                n_trials_per_model: int = 50) -> Dict[str, OptimizationResult]:
        """Optimize all models concurrently"""
        
        tasks = []
        for model_name, model_info in self.models.items():
            if isinstance(model_info, dict):
                model_class = model_info['class']
                parameter_space = model_info['parameter_space']
            else:
                # Default parameter spaces for common models
                model_class = model_info
                parameter_space = self._get_default_parameter_space(model_name)
            
            task = self.optimize_model(
                model_name, model_class, parameter_space,
                X_train, y_train, X_val, y_val, n_trials_per_model
            )
            tasks.append(task)
        
        # Run all optimizations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        optimized_results = {}
        for i, (model_name, result) in enumerate(zip(self.models.keys(), results)):
            if isinstance(result, Exception):
                self.logger.error(f"Model {model_name} optimization failed: {str(result)}")
            else:
                optimized_results[model_name] = result
        
        return optimized_results
    
    def _get_default_parameter_space(self, model_name: str) -> List[OptimizationParameter]:
        """Get default parameter space for common models"""
        
        if 'randomforest' in model_name.lower():
            return [
                OptimizationParameter('n_estimators', ParameterType.INTEGER, (10, 200)),
                OptimizationParameter('max_depth', ParameterType.INTEGER, (3, 20)),
                OptimizationParameter('min_samples_split', ParameterType.INTEGER, (2, 20)),
                OptimizationParameter('min_samples_leaf', ParameterType.INTEGER, (1, 10))
            ]
        elif 'gradientboosting' in model_name.lower():
            return [
                OptimizationParameter('n_estimators', ParameterType.INTEGER, (50, 300)),
                OptimizationParameter('learning_rate', ParameterType.DOUBLE, (0.01, 0.3), log_scale=True),
                OptimizationParameter('max_depth', ParameterType.INTEGER, (3, 10)),
                OptimizationParameter('subsample', ParameterType.DOUBLE, (0.5, 1.0))
            ]
        elif 'gaussianprocess' in model_name.lower():
            return [
                OptimizationParameter('alpha', ParameterType.DOUBLE, (1e-12, 1e-2), log_scale=True),
                OptimizationParameter('kernel_length_scale', ParameterType.DOUBLE, (0.1, 10.0), log_scale=True)
            ]
        else:
            # Generic parameter space
            return [
                OptimizationParameter('regularization', ParameterType.DOUBLE, (1e-6, 1e-1), log_scale=True)
            ]
    
    def select_best_model(self, metric: str = 'validation_loss') -> Tuple[str, Any, Dict[str, Any]]:
        """Select the best model across all optimized models"""
        
        if not self.optimization_results:
            raise ValueError("No models have been optimized yet")
        
        best_model_name = None
        best_value = float('inf')
        best_result = None
        
        for model_name, result in self.optimization_results.items():
            if result.best_objective_value < best_value:
                best_value = result.best_objective_value
                best_model_name = model_name
                best_result = result
        
        return best_model_name, self.best_models[best_model_name], best_result
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all optimized models"""
        
        comparison_data = []
        for model_name, result in self.optimization_results.items():
            comparison_data.append({
                'model_name': model_name,
                'best_objective': result.best_objective_value,
                'n_trials': result.n_trials,
                'optimization_time': result.elapsed_time,
                'converged': result.convergence_metrics.get('converged', False),
                'algorithm': result.algorithm_info.get('algorithm', 'unknown')
            })
        
        return pd.DataFrame(comparison_data)

class GeologicalParameterOptimizer:
    """Specialized optimizer for geological model parameters"""
    
    def __init__(self, geological_priors: Dict[str, Dict[str, Any]] = None):
        self.geological_priors = geological_priors or self._get_default_geological_priors()
        self.optimizer = VizierOptimizer()
        self.logger = logging.getLogger(__name__)
    
    def _get_default_geological_priors(self) -> Dict[str, Dict[str, Any]]:
        """Get default geological parameter priors"""
        return {
            'fault_parameters': {
                'strike': {'type': 'uniform', 'bounds': (0, 360), 'unit': 'degrees'},
                'dip': {'type': 'uniform', 'bounds': (0, 90), 'unit': 'degrees'},
                'rake': {'type': 'uniform', 'bounds': (-180, 180), 'unit': 'degrees'},
                'length': {'type': 'lognormal', 'bounds': (100, 50000), 'unit': 'meters'},
                'width': {'type': 'lognormal', 'bounds': (100, 20000), 'unit': 'meters'},
                'depth': {'type': 'lognormal', 'bounds': (100, 30000), 'unit': 'meters'},
                'slip': {'type': 'lognormal', 'bounds': (0.001, 50), 'unit': 'meters'}
            },
            'material_properties': {
                'youngs_modulus': {'type': 'lognormal', 'bounds': (10e9, 200e9), 'unit': 'Pa'},
                'poisson_ratio': {'type': 'beta', 'bounds': (0.15, 0.45), 'unit': 'dimensionless'},
                'density': {'type': 'normal', 'bounds': (2000, 4000), 'unit': 'kg/m³'},
                'viscosity': {'type': 'lognormal', 'bounds': (1e17, 1e22), 'unit': 'Pa·s'}
            },
            'source_geometry': {
                'volume_change': {'type': 'lognormal', 'bounds': (1e3, 1e9), 'unit': 'm³'},
                'pressure_change': {'type': 'normal', 'bounds': (-100e6, 100e6), 'unit': 'Pa'},
                'temperature': {'type': 'uniform', 'bounds': (800, 1500), 'unit': 'K'}
            }
        }
    
    def create_geological_parameter_space(self, parameter_categories: List[str]) -> List[OptimizationParameter]:
        """Create parameter space for geological optimization"""
        parameters = []
        
        for category in parameter_categories:
            if category in self.geological_priors:
                category_params = self.geological_priors[category]
                
                for param_name, param_info in category_params.items():
                    bounds = param_info['bounds']
                    param_type = ParameterType.DOUBLE
                    log_scale = param_info['type'] == 'lognormal'
                    
                    # Special handling for angles
                    if param_name in ['strike', 'dip', 'rake']:
                        param_type = ParameterType.DOUBLE
                        log_scale = False
                    
                    parameter = OptimizationParameter(
                        name=f"{category}_{param_name}",
                        type=param_type,
                        bounds=bounds,
                        log_scale=log_scale,
                        description=f"{param_name} ({param_info['unit']})"
                    )
                    parameters.append(parameter)
        
        return parameters
    
    async def optimize_geological_model(self, 
                                      forward_model: Callable[[Dict[str, Any]], np.ndarray],
                                      observations: np.ndarray,
                                      uncertainties: np.ndarray,
                                      parameter_categories: List[str],
                                      n_trials: int = 100) -> OptimizationResult:
        """Optimize geological model parameters"""
        
        # Create parameter space
        parameter_space = self.create_geological_parameter_space(parameter_categories)
        
        # Setup optimizer
        for param in parameter_space:
            self.optimizer.add_parameter(param)
        
        self.optimizer.set_objective(OptimizationObjective.MINIMIZE, 'chi_squared')
        
        def geological_objective(params: Dict[str, Any]) -> float:
            try:
                # Run forward model
                predicted = forward_model(params)
                
                # Calculate chi-squared
                residuals = (observations - predicted) / uncertainties
                chi_squared = np.sum(residuals**2)
                
                # Add geological constraints
                constraint_penalty = self._evaluate_geological_constraints(params)
                
                return chi_squared + constraint_penalty
                
            except Exception as e:
                self.logger.error(f"Forward model evaluation failed: {str(e)}")
                return 1e10
        
        # Run optimization
        result = await self.optimizer.optimize(geological_objective, n_trials=n_trials)
        
        return result
    
    def _evaluate_geological_constraints(self, params: Dict[str, Any]) -> float:
        """Evaluate geological constraints and return penalty"""
        penalty = 0.0
        
        # Example constraints
        for param_name, value in params.items():
            if 'depth' in param_name:
                # Depth should be positive
                if value <= 0:
                    penalty += 1e6
            
            if 'slip' in param_name:
                # Slip should be positive
                if value <= 0:
                    penalty += 1e6
            
            if 'poisson_ratio' in param_name:
                # Poisson ratio physical bounds
                if value <= 0 or value >= 0.5:
                    penalty += 1e6
        
        return penalty

# Example usage and testing
async def example_geological_optimization():
    """Example of using Vizier for geological parameter optimization"""
    
    # Create synthetic data
    np.random.seed(42)
    n_points = 100
    true_params = {'depth': 5000, 'slip': 2.0, 'strike': 45}
    
    # Synthetic forward model
    def forward_model(params):
        depth = params.get('fault_parameters_depth', 5000)
        slip = params.get('fault_parameters_slip', 1.0)
        
        # Simple synthetic model
        x = np.linspace(-10000, 10000, n_points)
        displacement = slip * np.exp(-np.abs(x) / depth) * 0.001
        return displacement
    
    # Synthetic observations
    true_displacement = forward_model({'fault_parameters_depth': 5000, 'fault_parameters_slip': 2.0})
    observations = true_displacement + np.random.normal(0, 0.0001, len(true_displacement))
    uncertainties = np.full_like(observations, 0.0001)
    
    # Create optimizer
    optimizer = GeologicalParameterOptimizer()
    
    # Run optimization
    result = await optimizer.optimize_geological_model(
        forward_model=forward_model,
        observations=observations,
        uncertainties=uncertainties,
        parameter_categories=['fault_parameters'],
        n_trials=50
    )
    
    print("Optimization completed!")
    print(f"Best parameters: {result.best_parameters}")
    print(f"Best objective value: {result.best_objective_value:.6f}")
    print(f"Number of trials: {result.n_trials}")
    print(f"Algorithm: {result.algorithm_info}")

if __name__ == "__main__":
    # Test Vizier integration
    print(f"Vizier available: {VIZIER_AVAILABLE}")
    
    # Run example
    asyncio.run(example_geological_optimization())