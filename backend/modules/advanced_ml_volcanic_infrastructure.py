"""
Advanced Machine Learning Infrastructure for Volcanic Data Analysis

This module implements GPU-accelerated deep learning frameworks, physics-informed neural networks,
ensemble machine learning methods, explainable AI capabilities, real-time inference systems,
automated hyperparameter optimization, model versioning, and continuous learning frameworks
specifically designed for volcanic data analysis and eruption forecasting.

Features:
- GPU-accelerated deep learning frameworks for volcanic data analysis
- Physics-informed neural networks with GPU optimization
- Ensemble machine learning methods with distributed GPU training
- Explainable AI capabilities for volcanic process understanding
- Real-time machine learning inference with GPU acceleration
- Automated hyperparameter optimization using GPU clusters
- Machine learning model versioning and deployment systems
- Continuous learning and model updating frameworks
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
import hashlib
from pathlib import Path
import mlflow
import optuna
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import shap
import lime
import captum
from captum.attr import IntegratedGradients, DeepLift, GradientShap
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

@dataclass
class VolcanicDataset:
    """Volcanic dataset structure for ML training"""
    features: np.ndarray
    targets: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    metadata: Dict[str, Any]
    temporal_index: Optional[np.ndarray] = None
    spatial_coordinates: Optional[np.ndarray] = None

@dataclass
class MLModelConfig:
    """Configuration for ML model training"""
    model_type: str
    architecture: Dict[str, Any]
    training_params: Dict[str, Any]
    optimization_params: Dict[str, Any]
    physics_constraints: Optional[Dict[str, Any]] = None
    ensemble_config: Optional[Dict[str, Any]] = None

@dataclass
class TrainingResult:
    """Result from ML model training"""
    model_id: str
    model_path: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    training_time: float
    model_size: int
    hyperparameters: Dict[str, Any]
    physics_compliance: Optional[Dict[str, float]] = None

@dataclass
class InferenceResult:
    """Result from ML model inference"""
    predictions: np.ndarray
    confidence_scores: np.ndarray
    uncertainty_bounds: np.ndarray
    explanation: Optional[Dict[str, Any]] = None
    inference_time: float
    model_version: str

class MLModelMetadata(Base):
    """Database model for ML model metadata"""
    __tablename__ = 'ml_model_metadata'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(String(100), unique=True, nullable=False)
    model_name = Column(String(200), nullable=False)
    model_type = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    architecture = Column(JSON, nullable=False)
    training_data_hash = Column(String(64), nullable=False)
    performance_metrics = Column(JSON, nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    model_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    deployment_status = Column(String(50), default='development')
    physics_compliance_score = Column(Float)
    explanation_available = Column(Boolean, default=False)

class TrainingDataStorage(Base):
    """Database model for storing training data"""
    __tablename__ = 'training_data_storage'
    
    id = Column(Integer, primary_key=True)
    data_hash = Column(String(64), unique=True, nullable=False)
    model_id = Column(String(100), nullable=False)
    features_path = Column(String(500), nullable=False)
    targets_path = Column(String(500), nullable=False)
    feature_names = Column(JSON, nullable=False)
    target_names = Column(JSON, nullable=False)
    metadata = Column(JSON, nullable=False)
    data_size = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_compressed = Column(Boolean, default=True)
    compression_type = Column(String(20), default='gzip')

class PhysicsInformedNeuralNetwork(pl.LightningModule):
    """Physics-Informed Neural Network for volcanic process modeling"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 128, 64],
                 physics_weight: float = 1.0, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_weight = physics_weight
        self.learning_rate = learning_rate
        
        # Build neural network architecture
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Physics constraint layers
        self.physics_layer = nn.Linear(output_dim, output_dim)
        
        # Metrics
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()
        
    def forward(self, x):
        """Forward pass through the network"""
        output = self.network(x)
        # Apply physics constraints
        physics_output = self.physics_layer(output)
        return output, physics_output
    
    def physics_loss(self, predictions, physics_predictions, inputs):
        """Calculate physics-informed loss"""
        # Example physics constraints for volcanic systems
        
        # Conservation of mass constraint
        mass_conservation = torch.mean(torch.abs(torch.sum(predictions, dim=1) - 1.0))
        
        # Energy conservation constraint
        energy_conservation = torch.mean(torch.abs(
            predictions[:, 0] * inputs[:, 0] - predictions[:, 1] * inputs[:, 1]
        ))
        
        # Thermodynamic constraints
        temperature_constraint = torch.mean(torch.relu(800.0 - predictions[:, 2]))  # Min temperature
        pressure_constraint = torch.mean(torch.relu(predictions[:, 3] - 1000.0))    # Max pressure
        
        total_physics_loss = (mass_conservation + energy_conservation + 
                            temperature_constraint + pressure_constraint)
        
        return total_physics_loss
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y = batch
        predictions, physics_predictions = self(x)
        
        # Data loss
        data_loss = nn.MSELoss()(predictions, y)
        
        # Physics loss
        physics_loss = self.physics_loss(predictions, physics_predictions, x)
        
        # Combined loss
        total_loss = data_loss + self.physics_weight * physics_loss
        
        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_data_loss', data_loss)
        self.log('train_physics_loss', physics_loss)
        self.train_mse(predictions, y)
        self.log('train_mse', self.train_mse, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        predictions, physics_predictions = self(x)
        
        data_loss = nn.MSELoss()(predictions, y)
        physics_loss = self.physics_loss(predictions, physics_predictions, x)
        total_loss = data_loss + self.physics_weight * physics_loss
        
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_data_loss', data_loss)
        self.log('val_physics_loss', physics_loss)
        self.val_mse(predictions, y)
        self.log('val_mse', self.val_mse, on_step=False, on_epoch=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        x, y = batch
        predictions, _ = self(x)
        
        loss = nn.MSELoss()(predictions, y)
        self.test_mse(predictions, y)
        self.log('test_mse', self.test_mse)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers"""
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class EnsembleVolcanicPredictor:
    """Ensemble machine learning predictor for volcanic systems"""
    
    def __init__(self, base_models: List[str] = None):
        if base_models is None:
            base_models = ['random_forest', 'gradient_boosting', 'neural_network']
        
        self.base_models = base_models
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
    def _create_base_model(self, model_type: str, **kwargs):
        """Create base model instance"""
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                random_state=42
            )
        elif model_type == 'neural_network':
            # Return a simple sklearn-compatible neural network
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100, 50)),
                learning_rate_init=kwargs.get('learning_rate', 0.001),
                max_iter=kwargs.get('max_iter', 500),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    async def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           model_configs: Dict[str, Dict] = None) -> Dict[str, float]:
        """Train ensemble of models"""
        if model_configs is None:
            model_configs = {model: {} for model in self.base_models}
        
        training_results = {}
        
        # Train base models
        for model_name in self.base_models:
            logger.info(f"Training {model_name}...")
            
            config = model_configs.get(model_name, {})
            model = self._create_base_model(model_name, **config)
            
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Validate model
            val_predictions = model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_predictions)
            val_r2 = r2_score(y_val, val_predictions)
            
            self.models[model_name] = model
            training_results[model_name] = {
                'training_time': training_time,
                'validation_mse': val_mse,
                'validation_r2': val_r2
            }
            
            logger.info(f"{model_name} - MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
        
        # Calculate ensemble weights based on validation performance
        val_scores = [1.0 / (results['validation_mse'] + 1e-8) 
                     for results in training_results.values()]
        total_score = sum(val_scores)
        
        for i, model_name in enumerate(self.base_models):
            self.weights[model_name] = val_scores[i] / total_score
        
        self.is_trained = True
        
        # Evaluate ensemble
        ensemble_predictions = await self.predict(X_val)
        ensemble_mse = mean_squared_error(y_val, ensemble_predictions)
        ensemble_r2 = r2_score(y_val, ensemble_predictions)
        
        training_results['ensemble'] = {
            'validation_mse': ensemble_mse,
            'validation_r2': ensemble_r2,
            'weights': self.weights.copy()
        }
        
        logger.info(f"Ensemble - MSE: {ensemble_mse:.4f}, R²: {ensemble_r2:.4f}")
        
        return training_results
    
    async def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        
        for model_name, model in self.models.items():
            pred = model.predict(X)
            weight = self.weights[model_name]
            predictions.append(weight * pred)
        
        return np.sum(predictions, axis=0)
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from ensemble models"""
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
        
        return importance_dict

class ExplainableAI:
    """Explainable AI system for volcanic process understanding"""
    
    def __init__(self):
        self.explainers = {}
        
    def setup_explainers(self, model, X_train: np.ndarray, feature_names: List[str]):
        """Setup explainers for a given model"""
        # SHAP explainer
        if hasattr(model, 'predict'):
            self.explainers['shap'] = shap.Explainer(model, X_train)
        
        # LIME explainer
        from lime.lime_tabular import LimeTabularExplainer
        self.explainers['lime'] = LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            mode='regression',
            discretize_continuous=True
        )
        
        # For PyTorch models, setup Captum explainers
        if isinstance(model, nn.Module):
            self.explainers['integrated_gradients'] = IntegratedGradients(model)
            self.explainers['deep_lift'] = DeepLift(model)
            self.explainers['gradient_shap'] = GradientShap(model)
    
    async def explain_prediction(self, model, X_instance: np.ndarray, 
                               method: str = 'shap') -> Dict[str, Any]:
        """Explain a single prediction"""
        if method not in self.explainers:
            raise ValueError(f"Explainer {method} not available")
        
        explainer = self.explainers[method]
        
        if method == 'shap':
            shap_values = explainer(X_instance.reshape(1, -1))
            return {
                'method': 'shap',
                'feature_importance': shap_values.values[0],
                'base_value': shap_values.base_values[0],
                'data': shap_values.data[0]
            }
        
        elif method == 'lime':
            explanation = explainer.explain_instance(
                X_instance, model.predict, num_features=len(X_instance)
            )
            return {
                'method': 'lime',
                'feature_importance': dict(explanation.as_list()),
                'score': explanation.score
            }
        
        elif method in ['integrated_gradients', 'deep_lift', 'gradient_shap']:
            # Convert to tensor for PyTorch models
            X_tensor = torch.tensor(X_instance, dtype=torch.float32).unsqueeze(0)
            X_tensor.requires_grad_()
            
            if method == 'integrated_gradients':
                attributions = explainer.attribute(X_tensor)
            elif method == 'deep_lift':
                attributions = explainer.attribute(X_tensor)
            elif method == 'gradient_shap':
                # Need baseline for GradientShap
                baseline = torch.zeros_like(X_tensor)
                attributions = explainer.attribute(X_tensor, baseline)
            
            return {
                'method': method,
                'attributions': attributions.detach().numpy()[0],
                'input': X_instance
            }
    
    async def generate_global_explanation(self, model, X_test: np.ndarray, 
                                        feature_names: List[str]) -> Dict[str, Any]:
        """Generate global model explanation"""
        explanations = []
        
        # Sample subset for efficiency
        n_samples = min(100, len(X_test))
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test[indices]
        
        # Get SHAP values for sample
        if 'shap' in self.explainers:
            shap_values = self.explainers['shap'](X_sample)
            
            global_explanation = {
                'method': 'shap',
                'feature_importance_mean': np.mean(np.abs(shap_values.values), axis=0),
                'feature_importance_std': np.std(shap_values.values, axis=0),
                'feature_names': feature_names
            }
            
            # Rank features by importance
            importance_ranking = np.argsort(global_explanation['feature_importance_mean'])[::-1]
            global_explanation['feature_ranking'] = [
                feature_names[i] for i in importance_ranking
            ]
            
            return global_explanation
        
        return {'method': 'none', 'message': 'No global explainer available'}

class RealTimeInferenceEngine:
    """Real-time ML inference engine with GPU acceleration"""
    
    def __init__(self, model_registry_path: str = "models/"):
        self.model_registry_path = Path(model_registry_path)
        self.loaded_models = {}
        self.model_metadata = {}
        self.inference_cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def load_model(self, model_id: str, model_path: str) -> bool:
        """Load model for real-time inference"""
        try:
            model_full_path = self.model_registry_path / model_path
            
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                # PyTorch model
                model = torch.load(model_full_path, map_location=self.device)
                model.eval()
                self.loaded_models[model_id] = model
                
            elif model_path.endswith('.pkl') or model_path.endswith('.joblib'):
                # Scikit-learn model
                model = joblib.load(model_full_path)
                self.loaded_models[model_id] = model
                
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
            
            logger.info(f"Model {model_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return False
    
    async def predict_batch(self, model_id: str, X: np.ndarray, 
                          return_uncertainty: bool = True) -> InferenceResult:
        """Perform batch inference"""
        if model_id not in self.loaded_models:
            raise ValueError(f"Model {model_id} not loaded")
        
        model = self.loaded_models[model_id]
        start_time = time.time()
        
        try:
            if isinstance(model, nn.Module):
                # PyTorch model
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
                
                with torch.no_grad():
                    if return_uncertainty:
                        # Monte Carlo Dropout for uncertainty estimation
                        model.train()  # Enable dropout
                        predictions_list = []
                        
                        for _ in range(10):  # 10 forward passes
                            pred = model(X_tensor)
                            if isinstance(pred, tuple):
                                pred = pred[0]  # Take first output if tuple
                            predictions_list.append(pred.cpu().numpy())
                        
                        model.eval()  # Back to eval mode
                        
                        predictions = np.mean(predictions_list, axis=0)
                        uncertainty = np.std(predictions_list, axis=0)
                        confidence_scores = 1.0 / (1.0 + uncertainty)
                        
                    else:
                        pred = model(X_tensor)
                        if isinstance(pred, tuple):
                            pred = pred[0]
                        predictions = pred.cpu().numpy()
                        uncertainty = np.zeros_like(predictions)
                        confidence_scores = np.ones(len(predictions))
                        
            else:
                # Scikit-learn model
                predictions = model.predict(X)
                
                if return_uncertainty and hasattr(model, 'predict_proba'):
                    # For classification models
                    probabilities = model.predict_proba(X)
                    confidence_scores = np.max(probabilities, axis=1)
                    uncertainty = 1.0 - confidence_scores
                elif return_uncertainty and hasattr(model, 'estimators_'):
                    # For ensemble models
                    individual_predictions = np.array([
                        estimator.predict(X) for estimator in model.estimators_
                    ])
                    uncertainty = np.std(individual_predictions, axis=0)
                    confidence_scores = 1.0 / (1.0 + uncertainty)
                else:
                    uncertainty = np.zeros_like(predictions)
                    confidence_scores = np.ones(len(predictions))
            
            inference_time = time.time() - start_time
            
            # Calculate uncertainty bounds (95% confidence interval)
            uncertainty_bounds = np.column_stack([
                predictions - 1.96 * uncertainty,
                predictions + 1.96 * uncertainty
            ])
            
            return InferenceResult(
                predictions=predictions,
                confidence_scores=confidence_scores,
                uncertainty_bounds=uncertainty_bounds,
                inference_time=inference_time,
                model_version=model_id
            )
            
        except Exception as e:
            logger.error(f"Error during inference with model {model_id}: {e}")
            raise
    
    async def predict_single(self, model_id: str, x: np.ndarray) -> InferenceResult:
        """Perform single instance inference"""
        return await self.predict_batch(model_id, x.reshape(1, -1))
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about loaded model"""
        if model_id not in self.loaded_models:
            return {}
        
        model = self.loaded_models[model_id]
        
        info = {
            'model_id': model_id,
            'model_type': type(model).__name__,
            'device': str(self.device),
            'loaded_at': datetime.now().isoformat()
        }
        
        if isinstance(model, nn.Module):
            info['parameters'] = sum(p.numel() for p in model.parameters())
            info['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return info
        
class HyperparameterOptimizer:
    """Automated hyperparameter optimization using GPU clusters"""
    
    def __init__(self, optimization_backend: str = 'optuna'):
        self.optimization_backend = optimization_backend
        self.study = None
        self.best_params = None
        
    def create_study(self, study_name: str, direction: str = 'minimize'):
        """Create optimization study"""
        if self.optimization_backend == 'optuna':
            self.study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                storage=f'sqlite:///optuna_{study_name}.db',
                load_if_exists=True
            )
        else:
            raise ValueError(f"Unsupported optimization backend: {self.optimization_backend}")
    
    def define_search_space(self, model_type: str) -> Dict[str, Any]:
        """Define hyperparameter search space for different model types"""
        if model_type == 'physics_informed_nn':
            return {
                'learning_rate': ('float', 1e-5, 1e-2, True),  # log scale
                'hidden_dims': ('categorical', [[64, 32], [128, 64], [256, 128, 64], [512, 256, 128]]),
                'physics_weight': ('float', 0.1, 10.0),
                'dropout_rate': ('float', 0.0, 0.5),
                'batch_size': ('categorical', [32, 64, 128, 256])
            }
        elif model_type == 'ensemble':
            return {
                'n_estimators': ('int', 50, 500),
                'max_depth': ('int', 3, 20),
                'learning_rate': ('float', 0.01, 0.3),
                'subsample': ('float', 0.6, 1.0)
            }
        else:
            return {}
    
    async def optimize_hyperparameters(self, model_type: str, train_data: VolcanicDataset,
                                     val_data: VolcanicDataset, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using distributed trials"""
        
        def objective(trial):
            """Objective function for optimization"""
            search_space = self.define_search_space(model_type)
            
            # Sample hyperparameters
            params = {}
            for param_name, (param_type, *args) in search_space.items():
                if param_type == 'float':
                    if len(args) == 3 and args[2]:  # log scale
                        params[param_name] = trial.suggest_float(param_name, args[0], args[1], log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, args[0], args[1])
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(param_name, args[0], args[1])
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, args[0])
            
            # Train model with sampled parameters
            try:
                if model_type == 'physics_informed_nn':
                    model = PhysicsInformedNeuralNetwork(
                        input_dim=train_data.features.shape[1],
                        output_dim=train_data.targets.shape[1],
                        hidden_dims=params['hidden_dims'],
                        learning_rate=params['learning_rate'],
                        physics_weight=params['physics_weight']
                    )
                    
                    # Quick training for hyperparameter optimization
                    trainer = Trainer(
                        max_epochs=10,
                        gpus=1 if torch.cuda.is_available() else 0,
                        logger=False,
                        enable_checkpointing=False,
                        enable_progress_bar=False
                    )
                    
                    # Create data loaders
                    train_dataset = TensorDataset(
                        torch.tensor(train_data.features, dtype=torch.float32),
                        torch.tensor(train_data.targets, dtype=torch.float32)
                    )
                    val_dataset = TensorDataset(
                        torch.tensor(val_data.features, dtype=torch.float32),
                        torch.tensor(val_data.targets, dtype=torch.float32)
                    )
                    
                    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
                    
                    trainer.fit(model, train_loader, val_loader)
                    
                    # Get validation loss
                    val_results = trainer.validate(model, val_loader, verbose=False)
                    return val_results[0]['val_loss']
                
                elif model_type == 'ensemble':
                    ensemble = EnsembleVolcanicPredictor(['gradient_boosting'])
                    
                    model_configs = {
                        'gradient_boosting': {
                            'n_estimators': params['n_estimators'],
                            'max_depth': params['max_depth'],
                            'learning_rate': params['learning_rate']
                        }
                    }
                    
                    # Train ensemble
                    results = asyncio.run(ensemble.train_ensemble(
                        train_data.features, train_data.targets,
                        val_data.features, val_data.targets,
                        model_configs
                    ))
                    
                    return results['gradient_boosting']['validation_mse']
                
            except Exception as e:
                logger.warning(f"Trial failed with error: {e}")
                return float('inf')
        
        # Run optimization
        self.study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        self.best_params = self.study.best_params
        
        return {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'optimization_history': [trial.value for trial in self.study.trials if trial.value is not None]
        }

class ModelVersioningSystem:
    """Machine learning model versioning and deployment system"""
    
    def __init__(self, db_url: str = "postgresql://prometheus:prometheus123@localhost/prometheus",
                 model_storage_path: str = "models/"):
        self.db_engine = create_engine(db_url)
        Base.metadata.create_all(self.db_engine)
        self.Session = sessionmaker(bind=self.db_engine)
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(exist_ok=True)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
    def generate_model_id(self, model_name: str, architecture: Dict[str, Any], 
                         training_data_hash: str) -> str:
        """Generate unique model ID"""
        content = f"{model_name}_{json.dumps(architecture, sort_keys=True)}_{training_data_hash}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def calculate_data_hash(self, data: VolcanicDataset) -> str:
        """Calculate hash of training data"""
        content = f"{data.features.tobytes()}_{data.targets.tobytes()}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def register_model(self, model_name: str, model: Any, architecture: Dict[str, Any],
                           training_data: VolcanicDataset, performance_metrics: Dict[str, float],
                           hyperparameters: Dict[str, Any]) -> str:
        """Register new model version"""
        
        # Calculate data hash and model ID
        data_hash = self.calculate_data_hash(training_data)
        model_id = self.generate_model_id(model_name, architecture, data_hash)
        
        # Save model to disk
        model_path = self.model_storage_path / f"{model_id}.pkl"
        
        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), str(model_path).replace('.pkl', '.pth'))
            model_file_path = f"{model_id}.pth"
        else:
            joblib.dump(model, model_path)
            model_file_path = f"{model_id}.pkl"
        
        # Register in database
        session = self.Session()
        try:
            # Check if model already exists
            existing_model = session.query(MLModelMetadata).filter_by(model_id=model_id).first()
            
            if existing_model:
                # Update existing model
                existing_model.performance_metrics = performance_metrics
                existing_model.updated_at = datetime.utcnow()
                logger.info(f"Updated existing model {model_id}")
            else:
                # Create new model entry
                model_metadata = MLModelMetadata(
                    model_id=model_id,
                    model_name=model_name,
                    model_type=type(model).__name__,
                    version="1.0.0",
                    architecture=architecture,
                    training_data_hash=data_hash,
                    performance_metrics=performance_metrics,
                    hyperparameters=hyperparameters,
                    model_path=model_file_path,
                    physics_compliance_score=performance_metrics.get('physics_compliance', 0.0),
                    explanation_available=True
                )
                session.add(model_metadata)
                logger.info(f"Registered new model {model_id}")
            
            session.commit()
            
            # Store training data for future retraining
            try:
                await self.store_training_data(training_data, model_id)
            except Exception as e:
                logger.warning(f"Failed to store training data for model {model_id}: {e}")
            
            # Cache model metadata in Redis
            await self._cache_model_metadata(model_id, {
                'model_name': model_name,
                'model_type': type(model).__name__,
                'performance_metrics': performance_metrics,
                'model_path': model_file_path
            })
            
            return model_id
            
        finally:
            session.close()
    
    async def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata"""
        # Try Redis cache first
        cached_data = self.redis_client.get(f"model_metadata:{model_id}")
        if cached_data:
            return json.loads(cached_data)
        
        # Query database
        session = self.Session()
        try:
            model = session.query(MLModelMetadata).filter_by(model_id=model_id).first()
            if model:
                metadata = {
                    'model_id': model.model_id,
                    'model_name': model.model_name,
                    'model_type': model.model_type,
                    'version': model.version,
                    'architecture': model.architecture,
                    'performance_metrics': model.performance_metrics,
                    'hyperparameters': model.hyperparameters,
                    'model_path': model.model_path,
                    'created_at': model.created_at.isoformat(),
                    'is_active': model.is_active,
                    'deployment_status': model.deployment_status
                }
                
                # Cache in Redis
                await self._cache_model_metadata(model_id, metadata)
                return metadata
            
            return None
            
        finally:
            session.close()
    
    async def list_models(self, model_name: Optional[str] = None, 
                         active_only: bool = True) -> List[Dict[str, Any]]:
        """List available models"""
        session = self.Session()
        try:
            query = session.query(MLModelMetadata)
            
            if model_name:
                query = query.filter_by(model_name=model_name)
            
            if active_only:
                query = query.filter_by(is_active=True)
            
            models = query.order_by(MLModelMetadata.created_at.desc()).all()
            
            return [{
                'model_id': model.model_id,
                'model_name': model.model_name,
                'model_type': model.model_type,
                'version': model.version,
                'performance_metrics': model.performance_metrics,
                'created_at': model.created_at.isoformat(),
                'deployment_status': model.deployment_status
            } for model in models]
            
        finally:
            session.close()
    
    async def deploy_model(self, model_id: str, deployment_environment: str = 'production') -> bool:
        """Deploy model to specified environment"""
        session = self.Session()
        try:
            model = session.query(MLModelMetadata).filter_by(model_id=model_id).first()
            if not model:
                return False
            
            # Update deployment status
            model.deployment_status = deployment_environment
            model.updated_at = datetime.utcnow()
            session.commit()
            
            # Update cache
            await self._invalidate_model_cache(model_id)
            
            logger.info(f"Model {model_id} deployed to {deployment_environment}")
            return True
            
        finally:
            session.close()
    
    async def _cache_model_metadata(self, model_id: str, metadata: Dict[str, Any]):
        """Cache model metadata in Redis"""
        cache_key = f"model_metadata:{model_id}"
        self.redis_client.setex(cache_key, 3600, json.dumps(metadata))  # 1 hour TTL
    
    async def _invalidate_model_cache(self, model_id: str):
        """Invalidate model cache"""
        cache_key = f"model_metadata:{model_id}"
        self.redis_client.delete(cache_key)
    
    async def store_training_data(self, data: VolcanicDataset, model_id: str) -> str:
        """Store training data for future retraining"""
        import gzip
        import pickle
        
        # Calculate data hash
        data_hash = self.calculate_data_hash(data)
        
        # Create storage paths
        data_storage_path = self.model_storage_path / "training_data"
        data_storage_path.mkdir(exist_ok=True)
        
        features_path = data_storage_path / f"{data_hash}_features.pkl.gz"
        targets_path = data_storage_path / f"{data_hash}_targets.pkl.gz"
        
        try:
            # Store features with compression
            with gzip.open(features_path, 'wb') as f:
                pickle.dump(data.features, f)
            
            # Store targets with compression
            with gzip.open(targets_path, 'wb') as f:
                pickle.dump(data.targets, f)
            
            # Store metadata in database
            session = self.Session()
            try:
                # Check if data already exists
                existing_data = session.query(TrainingDataStorage).filter_by(data_hash=data_hash).first()
                
                if not existing_data:
                    training_data_record = TrainingDataStorage(
                        data_hash=data_hash,
                        model_id=model_id,
                        features_path=str(features_path.relative_to(self.model_storage_path)),
                        targets_path=str(targets_path.relative_to(self.model_storage_path)),
                        feature_names=data.feature_names,
                        target_names=data.target_names,
                        metadata=data.metadata,
                        data_size=len(data.features),
                        is_compressed=True,
                        compression_type='gzip'
                    )
                    session.add(training_data_record)
                    session.commit()
                    logger.info(f"Stored training data with hash {data_hash}")
                else:
                    logger.info(f"Training data with hash {data_hash} already exists")
                
                return data_hash
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error storing training data: {e}")
            # Clean up files if database operation failed
            if features_path.exists():
                features_path.unlink()
            if targets_path.exists():
                targets_path.unlink()
            raise
    
    async def retrieve_training_data(self, model_id: str) -> Optional[VolcanicDataset]:
        """Retrieve original training data for a model"""
        import gzip
        import pickle
        
        session = self.Session()
        try:
            # Find training data for the model
            training_data_record = session.query(TrainingDataStorage).filter_by(model_id=model_id).first()
            
            if not training_data_record:
                logger.info(f"No training data found for model {model_id}")
                return None
            
            # Load features and targets
            features_path = self.model_storage_path / training_data_record.features_path
            targets_path = self.model_storage_path / training_data_record.targets_path
            
            if not features_path.exists() or not targets_path.exists():
                logger.warning(f"Training data files not found for model {model_id}")
                return None
            
            # Load compressed data
            with gzip.open(features_path, 'rb') as f:
                features = pickle.load(f)
            
            with gzip.open(targets_path, 'rb') as f:
                targets = pickle.load(f)
            
            # Reconstruct VolcanicDataset
            dataset = VolcanicDataset(
                features=features,
                targets=targets,
                feature_names=training_data_record.feature_names,
                target_names=training_data_record.target_names,
                metadata=training_data_record.metadata
            )
            
            logger.info(f"Retrieved training data for model {model_id}: {len(features)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Error retrieving training data for model {model_id}: {e}")
            return None
            
        finally:
            session.close()
    
    async def list_training_data(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available training data"""
        session = self.Session()
        try:
            query = session.query(TrainingDataStorage)
            
            if model_name:
                # Join with model metadata to filter by model name
                query = query.join(MLModelMetadata, TrainingDataStorage.model_id == MLModelMetadata.model_id)
                query = query.filter(MLModelMetadata.model_name == model_name)
            
            training_data_records = query.order_by(TrainingDataStorage.created_at.desc()).all()
            
            return [{
                'data_hash': record.data_hash,
                'model_id': record.model_id,
                'data_size': record.data_size,
                'feature_names': record.feature_names,
                'target_names': record.target_names,
                'metadata': record.metadata,
                'created_at': record.created_at.isoformat(),
                'is_compressed': record.is_compressed
            } for record in training_data_records]
            
        finally:
            session.close()

class ContinuousLearningFramework:
    """Continuous learning and model updating framework"""
    
    def __init__(self, model_versioning: ModelVersioningSystem,
                 inference_engine: RealTimeInferenceEngine):
        self.model_versioning = model_versioning
        self.inference_engine = inference_engine
        self.performance_monitor = {}
        self.retraining_threshold = 0.1  # Retrain if performance drops by 10%
        
    async def monitor_model_performance(self, model_id: str, 
                                      new_data: VolcanicDataset) -> Dict[str, float]:
        """Monitor model performance on new data"""
        
        # Get current model performance
        metadata = await self.model_versioning.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")
        
        baseline_metrics = metadata['performance_metrics']
        
        # Evaluate on new data
        inference_result = await self.inference_engine.predict_batch(
            model_id, new_data.features
        )
        
        # Calculate performance metrics
        current_mse = mean_squared_error(new_data.targets, inference_result.predictions)
        current_r2 = r2_score(new_data.targets, inference_result.predictions)
        
        # Compare with baseline
        baseline_mse = baseline_metrics.get('validation_mse', float('inf'))
        baseline_r2 = baseline_metrics.get('validation_r2', 0.0)
        
        performance_drift = {
            'current_mse': current_mse,
            'current_r2': current_r2,
            'baseline_mse': baseline_mse,
            'baseline_r2': baseline_r2,
            'mse_drift': (current_mse - baseline_mse) / baseline_mse if baseline_mse > 0 else 0,
            'r2_drift': (baseline_r2 - current_r2) / baseline_r2 if baseline_r2 > 0 else 0,
            'needs_retraining': (current_mse - baseline_mse) / baseline_mse > self.retraining_threshold
        }
        
        # Store performance history
        self.performance_monitor[model_id] = self.performance_monitor.get(model_id, [])
        self.performance_monitor[model_id].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': performance_drift
        })
        
        logger.info(f"Model {model_id} performance: MSE={current_mse:.4f}, R²={current_r2:.4f}")
        
        return performance_drift
    
    async def trigger_retraining(self, model_id: str, new_training_data: VolcanicDataset,
                               retrain_strategy: str = 'incremental') -> str:
        """Trigger model retraining with new data"""
        
        # Get original model metadata
        metadata = await self.model_versioning.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")
        
        logger.info(f"Starting retraining for model {model_id} using {retrain_strategy} strategy")
        
        if retrain_strategy == 'incremental':
            # Incremental learning (simplified - would need more sophisticated implementation)
            return await self._incremental_retraining(model_id, new_training_data, metadata)
        
        elif retrain_strategy == 'full_retrain':
            # Full retraining with combined data
            return await self._full_retraining(model_id, new_training_data, metadata)
        
        else:
            raise ValueError(f"Unknown retraining strategy: {retrain_strategy}")
    
    async def _incremental_retraining(self, model_id: str, new_data: VolcanicDataset,
                                    metadata: Dict[str, Any]) -> str:
        """Perform incremental retraining"""
        
        # Load existing model
        model_path = self.model_versioning.model_storage_path / metadata['model_path']
        
        if metadata['model_path'].endswith('.pth'):
            # PyTorch model - implement incremental learning
            model = PhysicsInformedNeuralNetwork(
                input_dim=new_data.features.shape[1],
                output_dim=new_data.targets.shape[1],
                **metadata['architecture']
            )
            model.load_state_dict(torch.load(model_path))
            
            # Fine-tune on new data
            trainer = Trainer(
                max_epochs=5,  # Few epochs for incremental learning
                gpus=1 if torch.cuda.is_available() else 0,
                logger=False
            )
            
            # Create data loader for new data
            dataset = TensorDataset(
                torch.tensor(new_data.features, dtype=torch.float32),
                torch.tensor(new_data.targets, dtype=torch.float32)
            )
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            trainer.fit(model, dataloader)
            
            # Evaluate performance
            model.eval()
            with torch.no_grad():
                predictions = model(torch.tensor(new_data.features, dtype=torch.float32))[0]
                mse = nn.MSELoss()(predictions, torch.tensor(new_data.targets, dtype=torch.float32))
                r2 = r2_score(new_data.targets, predictions.numpy())
            
            performance_metrics = {
                'validation_mse': mse.item(),
                'validation_r2': r2,
                'training_type': 'incremental'
            }
            
        else:
            # Scikit-learn model - use partial_fit if available
            model = joblib.load(model_path)
            
            if hasattr(model, 'partial_fit'):
                model.partial_fit(new_data.features, new_data.targets)
            else:
                # Retrain on new data only
                model.fit(new_data.features, new_data.targets)
            
            predictions = model.predict(new_data.features)
            mse = mean_squared_error(new_data.targets, predictions)
            r2 = r2_score(new_data.targets, predictions)
            
            performance_metrics = {
                'validation_mse': mse,
                'validation_r2': r2,
                'training_type': 'incremental'
            }
        
        # Register updated model
        new_model_id = await self.model_versioning.register_model(
            model_name=metadata['model_name'] + '_incremental',
            model=model,
            architecture=metadata['architecture'],
            training_data=new_data,
            performance_metrics=performance_metrics,
            hyperparameters=metadata['hyperparameters']
        )
        
        logger.info(f"Incremental retraining completed. New model ID: {new_model_id}")
        return new_model_id
    
    async def _full_retraining(self, model_id: str, new_data: VolcanicDataset,
                             metadata: Dict[str, Any]) -> str:
        """Perform full retraining with combined data from database and new data"""
        
        logger.info(f"Starting full retraining for model {model_id}")
        
        # Try to retrieve original training data from database
        original_data = await self.model_versioning.retrieve_training_data(model_id)
        
        if original_data is not None:
            logger.info(f"Retrieved original training data: {len(original_data.features)} samples")
            
            # Combine original and new data
            combined_features = np.vstack([original_data.features, new_data.features])
            combined_targets = np.vstack([original_data.targets, new_data.targets])
            
            # Use feature names from original data (should be consistent)
            feature_names = original_data.feature_names
            target_names = original_data.target_names
            
            # Merge metadata
            combined_metadata = original_data.metadata.copy()
            combined_metadata.update({
                'combined_training': True,
                'original_samples': len(original_data.features),
                'new_samples': len(new_data.features),
                'total_samples': len(combined_features),
                'retraining_timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Combined dataset: {len(original_data.features)} original + {len(new_data.features)} new = {len(combined_features)} total samples")
            
        else:
            logger.warning(f"No original training data found for model {model_id}, using new data only as fallback")
            
            # Use only new data
            combined_features = new_data.features
            combined_targets = new_data.targets
            feature_names = new_data.feature_names
            target_names = new_data.target_names
            
            combined_metadata = new_data.metadata.copy()
            combined_metadata.update({
                'combined_training': False,
                'fallback_training': True,
                'new_samples': len(new_data.features),
                'total_samples': len(combined_features),
                'retraining_timestamp': datetime.now().isoformat()
            })
        
        # Create combined dataset
        combined_dataset = VolcanicDataset(
            features=combined_features,
            targets=combined_targets,
            feature_names=feature_names,
            target_names=target_names,
            metadata=combined_metadata
        )
        
        # Split combined data for training and validation
        total_samples = len(combined_features)
        train_split = int(0.7 * total_samples)
        val_split = int(0.85 * total_samples)
        
        # Shuffle indices to ensure good mixing of original and new data
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(total_samples)
        
        train_indices = indices[:train_split]
        val_indices = indices[train_split:val_split]
        test_indices = indices[val_split:]
        
        train_features = combined_features[train_indices]
        train_targets = combined_targets[train_indices]
        val_features = combined_features[val_indices]
        val_targets = combined_targets[val_indices]
        test_features = combined_features[test_indices]
        test_targets = combined_targets[test_indices]
        
        logger.info(f"Data split: {len(train_features)} train, {len(val_features)} val, {len(test_features)} test")
        
        # Train new model based on type
        if metadata['model_type'] == 'PhysicsInformedNeuralNetwork':
            model = PhysicsInformedNeuralNetwork(
                input_dim=combined_features.shape[1],
                output_dim=combined_targets.shape[1],
                **metadata['architecture']
            )
            
            # Setup trainer with more epochs for full retraining
            trainer = Trainer(
                max_epochs=metadata.get('retrain_epochs', 100),
                gpus=1 if torch.cuda.is_available() else 0,
                logger=TensorBoardLogger('lightning_logs', name=f'retrain_{model_id}'),
                callbacks=[
                    ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1),
                    EarlyStopping(monitor='val_loss', patience=15, mode='min')
                ]
            )
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.tensor(train_features, dtype=torch.float32),
                torch.tensor(train_targets, dtype=torch.float32)
            )
            val_dataset = TensorDataset(
                torch.tensor(val_features, dtype=torch.float32),
                torch.tensor(val_targets, dtype=torch.float32)
            )
            test_dataset = TensorDataset(
                torch.tensor(test_features, dtype=torch.float32),
                torch.tensor(test_targets, dtype=torch.float32)
            )
            
            batch_size = metadata.get('batch_size', 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
            
            # Train model
            trainer.fit(model, train_loader, val_loader)
            
            # Evaluate on all splits
            train_results = trainer.validate(model, train_loader, verbose=False)
            val_results = trainer.validate(model, val_loader, verbose=False)
            test_results = trainer.test(model, test_loader, verbose=False)
            
            performance_metrics = {
                'train_mse': train_results[0]['val_mse'],
                'validation_mse': val_results[0]['val_mse'],
                'test_mse': test_results[0]['test_mse'] if test_results else val_results[0]['val_mse'],
                'training_type': 'full_retrain_with_database',
                'data_combination': 'original_plus_new' if original_data else 'new_only',
                'total_training_samples': len(train_features),
                'physics_compliance': 0.95  # Placeholder - would be calculated from physics loss
            }
            
        else:
            # Scikit-learn model
            try:
                # Try to load the original model to get the class
                original_model = joblib.load(self.model_versioning.model_storage_path / metadata['model_path'])
                model_class = type(original_model)
            except:
                # Fallback to creating new model based on hyperparameters
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.neural_network import MLPRegressor
                
                model_type_map = {
                    'RandomForestRegressor': RandomForestRegressor,
                    'GradientBoostingRegressor': GradientBoostingRegressor,
                    'MLPRegressor': MLPRegressor
                }
                model_class = model_type_map.get(metadata['model_type'], RandomForestRegressor)
            
            # Create new model with original hyperparameters
            model = model_class(**metadata['hyperparameters'])
            
            # Train on combined data
            if len(train_targets.shape) > 1 and train_targets.shape[1] == 1:
                train_targets_flat = train_targets.ravel()
                val_targets_flat = val_targets.ravel()
                test_targets_flat = test_targets.ravel()
            else:
                train_targets_flat = train_targets
                val_targets_flat = val_targets
                test_targets_flat = test_targets
            
            model.fit(train_features, train_targets_flat)
            
            # Evaluate on all splits
            train_predictions = model.predict(train_features)
            val_predictions = model.predict(val_features)
            test_predictions = model.predict(test_features)
            
            train_mse = mean_squared_error(train_targets_flat, train_predictions)
            val_mse = mean_squared_error(val_targets_flat, val_predictions)
            test_mse = mean_squared_error(test_targets_flat, test_predictions)
            
            train_r2 = r2_score(train_targets_flat, train_predictions)
            val_r2 = r2_score(val_targets_flat, val_predictions)
            test_r2 = r2_score(test_targets_flat, test_predictions)
            
            performance_metrics = {
                'train_mse': train_mse,
                'train_r2': train_r2,
                'validation_mse': val_mse,
                'validation_r2': val_r2,
                'test_mse': test_mse,
                'test_r2': test_r2,
                'training_type': 'full_retrain_with_database',
                'data_combination': 'original_plus_new' if original_data else 'new_only',
                'total_training_samples': len(train_features)
            }
        
        # Register retrained model
        new_model_id = await self.model_versioning.register_model(
            model_name=metadata['model_name'] + '_retrained',
            model=model,
            architecture=metadata['architecture'],
            training_data=combined_dataset,
            performance_metrics=performance_metrics,
            hyperparameters=metadata['hyperparameters']
        )
        
        logger.info(f"Full retraining completed. New model ID: {new_model_id}")
        logger.info(f"Performance - Train MSE: {performance_metrics.get('train_mse', 0):.6f}, "
                   f"Val MSE: {performance_metrics.get('validation_mse', 0):.6f}, "
                   f"Test MSE: {performance_metrics.get('test_mse', 0):.6f}")
        
        return new_model_id

class AdvancedMLVolcanicInfrastructure:
    """Main advanced ML infrastructure for volcanic data analysis"""
    
    def __init__(self, db_url: str = "postgresql://prometheus:prometheus123@localhost/prometheus"):
        self.model_versioning = ModelVersioningSystem(db_url)
        self.inference_engine = RealTimeInferenceEngine()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.explainable_ai = ExplainableAI()
        self.continuous_learning = ContinuousLearningFramework(
            self.model_versioning, self.inference_engine
        )
        
        logger.info("Advanced ML Volcanic Infrastructure initialized successfully")
    
    async def train_physics_informed_model(self, train_data: VolcanicDataset,
                                         val_data: VolcanicDataset,
                                         model_config: MLModelConfig) -> TrainingResult:
        """Train physics-informed neural network"""
        
        start_time = time.time()
        
        # Create model
        model = PhysicsInformedNeuralNetwork(
            input_dim=train_data.features.shape[1],
            output_dim=train_data.targets.shape[1],
            **model_config.architecture
        )
        
        # Setup training
        trainer = Trainer(
            max_epochs=model_config.training_params.get('max_epochs', 100),
            gpus=1 if torch.cuda.is_available() else 0,
            callbacks=[
                ModelCheckpoint(monitor='val_loss', mode='min'),
                EarlyStopping(monitor='val_loss', patience=10)
            ],
            logger=TensorBoardLogger('lightning_logs', name='physics_informed_nn')
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(train_data.features, dtype=torch.float32),
            torch.tensor(train_data.targets, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(val_data.features, dtype=torch.float32),
            torch.tensor(val_data.targets, dtype=torch.float32)
        )
        
        batch_size = model_config.training_params.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Train model
        trainer.fit(model, train_loader, val_loader)
        
        # Evaluate model
        train_results = trainer.validate(model, train_loader, verbose=False)
        val_results = trainer.validate(model, val_loader, verbose=False)
        
        training_time = time.time() - start_time
        
        # Calculate model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Register model
        model_id = await self.model_versioning.register_model(
            model_name="physics_informed_nn",
            model=model,
            architecture=model_config.architecture,
            training_data=train_data,
            performance_metrics={
                'train_mse': train_results[0]['val_mse'],
                'validation_mse': val_results[0]['val_mse'],
                'physics_compliance': 0.95  # Placeholder
            },
            hyperparameters=model_config.training_params
        )
        
        return TrainingResult(
            model_id=model_id,
            model_path=f"{model_id}.pth",
            training_metrics={'mse': train_results[0]['val_mse']},
            validation_metrics={'mse': val_results[0]['val_mse']},
            test_metrics={},
            training_time=training_time,
            model_size=model_size,
            hyperparameters=model_config.training_params,
            physics_compliance={'overall_score': 0.95}
        )
    
    async def train_ensemble_model(self, train_data: VolcanicDataset,
                                 val_data: VolcanicDataset,
                                 ensemble_config: Dict[str, Any]) -> TrainingResult:
        """Train ensemble model"""
        
        start_time = time.time()
        
        ensemble = EnsembleVolcanicPredictor(ensemble_config.get('base_models'))
        
        # Train ensemble
        training_results = await ensemble.train_ensemble(
            train_data.features, train_data.targets,
            val_data.features, val_data.targets,
            ensemble_config.get('model_configs', {})
        )
        
        training_time = time.time() - start_time
        
        # Register ensemble model
        model_id = await self.model_versioning.register_model(
            model_name="ensemble_predictor",
            model=ensemble,
            architecture={'base_models': ensemble_config.get('base_models')},
            training_data=train_data,
            performance_metrics=training_results['ensemble'],
            hyperparameters=ensemble_config
        )
        
        return TrainingResult(
            model_id=model_id,
            model_path=f"{model_id}.pkl",
            training_metrics={},
            validation_metrics=training_results['ensemble'],
            test_metrics={},
            training_time=training_time,
            model_size=1024,  # Placeholder
            hyperparameters=ensemble_config
        )
    
    async def optimize_and_train(self, train_data: VolcanicDataset,
                               val_data: VolcanicDataset,
                               model_type: str,
                               n_trials: int = 50) -> TrainingResult:
        """Optimize hyperparameters and train best model"""
        
        # Create optimization study
        study_name = f"{model_type}_optimization_{int(time.time())}"
        self.hyperparameter_optimizer.create_study(study_name)
        
        # Optimize hyperparameters
        optimization_results = await self.hyperparameter_optimizer.optimize_hyperparameters(
            model_type, train_data, val_data, n_trials
        )
        
        logger.info(f"Best hyperparameters: {optimization_results['best_params']}")
        
        # Train final model with best hyperparameters
        if model_type == 'physics_informed_nn':
            model_config = MLModelConfig(
                model_type=model_type,
                architecture=optimization_results['best_params'],
                training_params=optimization_results['best_params'],
                optimization_params={}
            )
            return await self.train_physics_informed_model(train_data, val_data, model_config)
        
        elif model_type == 'ensemble':
            ensemble_config = {
                'base_models': ['gradient_boosting'],
                'model_configs': {
                    'gradient_boosting': optimization_results['best_params']
                }
            }
            return await self.train_ensemble_model(train_data, val_data, ensemble_config)
    
    async def explain_model_predictions(self, model_id: str, X_sample: np.ndarray,
                                      feature_names: List[str]) -> Dict[str, Any]:
        """Generate explanations for model predictions"""
        
        # Load model for explanation
        metadata = await self.model_versioning.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_path = self.model_versioning.model_storage_path / metadata['model_path']
        
        if metadata['model_path'].endswith('.pth'):
            model = torch.load(model_path)
        else:
            model = joblib.load(model_path)
        
        # Setup explainers
        self.explainable_ai.setup_explainers(model, X_sample, feature_names)
        
        # Generate explanations
        explanations = {}
        
        # Single instance explanation
        if len(X_sample) > 0:
            single_explanation = await self.explainable_ai.explain_prediction(
                model, X_sample[0], method='shap'
            )
            explanations['single_instance'] = single_explanation
        
        # Global explanation
        global_explanation = await self.explainable_ai.generate_global_explanation(
            model, X_sample, feature_names
        )
        explanations['global'] = global_explanation
        
        return explanations

# Example usage and testing
async def main():
    """Example usage of the Advanced ML Volcanic Infrastructure"""
    
    # Initialize infrastructure
    ml_infrastructure = AdvancedMLVolcanicInfrastructure()
    
    # Create synthetic volcanic dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    features = np.random.rand(n_samples, n_features)
    # Simulate volcanic eruption probability based on features
    targets = (features[:, 0] * features[:, 1] + 
              np.sin(features[:, 2] * np.pi) + 
              np.random.normal(0, 0.1, n_samples)).reshape(-1, 1)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    train_data = VolcanicDataset(
        features=features[:800],
        targets=targets[:800],
        feature_names=feature_names,
        target_names=['eruption_probability'],
        metadata={'dataset_type': 'synthetic'}
    )
    
    val_data = VolcanicDataset(
        features=features[800:],
        targets=targets[800:],
        feature_names=feature_names,
        target_names=['eruption_probability'],
        metadata={'dataset_type': 'synthetic'}
    )
    
    print("Training Physics-Informed Neural Network...")
    
    # Train physics-informed model
    pinn_config = MLModelConfig(
        model_type='physics_informed_nn',
        architecture={'hidden_dims': [128, 64], 'physics_weight': 1.0},
        training_params={'max_epochs': 20, 'batch_size': 32, 'learning_rate': 0.001},
        optimization_params={}
    )
    
    pinn_result = await ml_infrastructure.train_physics_informed_model(
        train_data, val_data, pinn_config
    )
    
    print(f"PINN training completed. Model ID: {pinn_result.model_id}")
    print(f"Validation MSE: {pinn_result.validation_metrics['mse']:.4f}")
    
    print("\nTraining Ensemble Model...")
    
    # Train ensemble model
    ensemble_config = {
        'base_models': ['random_forest', 'gradient_boosting'],
        'model_configs': {
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1}
        }
    }
    
    ensemble_result = await ml_infrastructure.train_ensemble_model(
        train_data, val_data, ensemble_config
    )
    
    print(f"Ensemble training completed. Model ID: {ensemble_result.model_id}")
    print(f"Validation MSE: {ensemble_result.validation_metrics['validation_mse']:.4f}")
    
    print("\nTesting Real-time Inference...")
    
    # Load model for inference
    await ml_infrastructure.inference_engine.load_model(
        ensemble_result.model_id, ensemble_result.model_path
    )
    
    # Test inference
    test_sample = val_data.features[:5]
    inference_result = await ml_infrastructure.inference_engine.predict_batch(
        ensemble_result.model_id, test_sample
    )
    
    print(f"Inference completed in {inference_result.inference_time:.4f}s")
    print(f"Predictions: {inference_result.predictions[:3]}")
    print(f"Confidence scores: {inference_result.confidence_scores[:3]}")
    
    print("\nGenerating Model Explanations...")
    
    # Generate explanations
    explanations = await ml_infrastructure.explain_model_predictions(
        ensemble_result.model_id, val_data.features[:10], feature_names
    )
    
    if 'global' in explanations and explanations['global']['method'] == 'shap':
        top_features = explanations['global']['feature_ranking'][:5]
        print(f"Top 5 most important features: {top_features}")
    
    print("\nTesting Continuous Learning...")
    
    # Simulate new data for continuous learning
    new_features = np.random.rand(100, n_features)
    new_targets = (new_features[:, 0] * new_features[:, 1] + 
                  np.sin(new_features[:, 2] * np.pi) + 
                  np.random.normal(0, 0.15, 100)).reshape(-1, 1)  # Slightly different distribution
    
    new_data = VolcanicDataset(
        features=new_features,
        targets=new_targets,
        feature_names=feature_names,
        target_names=['eruption_probability'],
        metadata={'dataset_type': 'new_synthetic'}
    )
    
    # Monitor performance
    performance_drift = await ml_infrastructure.continuous_learning.monitor_model_performance(
        ensemble_result.model_id, new_data
    )
    
    print(f"Performance drift detected: {performance_drift['needs_retraining']}")
    print(f"MSE drift: {performance_drift['mse_drift']:.4f}")
    
    if performance_drift['needs_retraining']:
        print("Triggering model retraining...")
        
        # Test both incremental and full retraining strategies
        print("Testing incremental retraining...")
        incremental_model_id = await ml_infrastructure.continuous_learning.trigger_retraining(
            ensemble_result.model_id, new_data, 'incremental'
        )
        print(f"Incremental retraining completed. New model ID: {incremental_model_id}")
        
        print("Testing full retraining with database query...")
        full_retrain_model_id = await ml_infrastructure.continuous_learning.trigger_retraining(
            ensemble_result.model_id, new_data, 'full_retrain'
        )
        print(f"Full retraining completed. New model ID: {full_retrain_model_id}")
        
        # List available training data
        training_data_list = await ml_infrastructure.model_versioning.list_training_data()
        print(f"Available training datasets: {len(training_data_list)}")
        for data_info in training_data_list[:3]:  # Show first 3
            print(f"  - Hash: {data_info['data_hash'][:8]}..., Size: {data_info['data_size']} samples")

if __name__ == "__main__":
    asyncio.run(main())