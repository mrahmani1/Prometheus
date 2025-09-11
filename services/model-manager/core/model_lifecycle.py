"""
Model Lifecycle Management - Core business logic for ML model operations
Integrates with existing backend modules for comprehensive model management
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import uuid
import json
import numpy as np
import torch

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend'))

from modules.model_manager import ModelManager, ModelConfig
from modules.mlops_pipeline import MLOpsPipeline, PipelineConfig, ModelStage, DeploymentStrategy
from modules.model_database import ModelDatabase, ModelMetadata
from modules.advanced_ml_model_management import AdvancedMLModelManager

logger = logging.getLogger(__name__)

class ModelLifecycleManager:
    """
    Comprehensive model lifecycle management integrating all existing modules
    """
    
    def __init__(self):
        self.model_manager: Optional[ModelManager] = None
        self.mlops_pipeline: Optional[MLOpsPipeline] = None
        self.model_database: Optional[ModelDatabase] = None
        self.advanced_manager: Optional[AdvancedMLModelManager] = None
        
        # State tracking
        self.active_training_jobs: Dict[str, Dict[str, Any]] = {}
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.drift_detection_results: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self, config: Dict[str, Any]):
        """Initialize all model management components"""
        try:
            # Initialize model database
            db_config = {
                'database_url': config.get('database_url', 'sqlite:///model_lifecycle.db'),
                'storage_backend': 'multi_cloud',
                'storage_path': config.get('model_storage_path', './models'),
                'cloud_providers': config.get('cloud_providers', ['local'])
            }
            self.model_database = ModelDatabase(db_config)
            await self.model_database.initialize()
            
            # Initialize MLOps pipeline
            pipeline_config = PipelineConfig(
                model_registry_url=config.get('mlflow_url', 'http://localhost:5000'),
                experiment_tracking_url=config.get('mlflow_url', 'http://localhost:5000'),
                data_drift_threshold=config.get('drift_threshold', 0.1),
                model_performance_threshold=config.get('performance_threshold', 0.75),
                retraining_frequency=config.get('retrain_frequency', 24),
                validation_split=0.2,
                monitoring_interval=300,
                multi_cloud_enabled=config.get('multi_cloud_enabled', True),
                auto_deploy=config.get('auto_deploy', True),
                enable_a_b_testing=True,
                enable_canary_deployment=True
            )
            self.mlops_pipeline = MLOpsPipeline(pipeline_config)
            await self.mlops_pipeline.initialize()
            
            # Initialize advanced model manager
            advanced_config = {
                'enable_multi_cloud': config.get('multi_cloud_enabled', True),
                'enable_automl': config.get('enable_automl', True),
                'enable_federated_learning': config.get('enable_federated_learning', False),
                'performance_monitoring': True,
                'automated_retraining': config.get('automated_retraining', True)
            }
            self.advanced_manager = AdvancedMLModelManager(advanced_config)
            
            # Initialize basic model manager
            model_config = ModelConfig(
                model_type="volcanic_prediction",
                hyperparameters=config.get('default_hyperparameters', {}),
                azure_vm_endpoint=config.get('azure_vm_endpoint', ''),
                google_cloud_project=config.get('google_cloud_project', ''),
                aws_region=config.get('aws_region', 'us-west-2'),
                dataset_path=config.get('dataset_path', './data'),
                model_save_path=config.get('model_save_path', './models'),
                multi_cloud_training=config.get('multi_cloud_enabled', True),
                auto_deploy=config.get('auto_deploy', True)
            )
            self.model_manager = ModelManager(model_config)
            
            # Load existing models
            await self._load_model_registry()
            
            logger.info("Model lifecycle manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Model lifecycle manager initialization failed: {e}")
            raise
    
    async def _load_model_registry(self):
        """Load existing models into registry"""
        try:
            if self.model_database:
                models = await self.model_database.list_models()
                for model in models:
                    self.model_registry[model.model_id] = {
                        'metadata': model,
                        'status': 'registered',
                        'last_updated': model.created_at
                    }
            
            logger.info(f"Loaded {len(self.model_registry)} models into registry")
            
        except Exception as e:
            logger.error(f"Model registry loading failed: {e}")
    
    async def train_model(self, model_type: str, config: Dict[str, Any], 
                         force_retrain: bool = False) -> Dict[str, Any]:
        """Train a new model with comprehensive lifecycle management"""
        try:
            job_id = f"train_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create training job record
            training_job = {
                'job_id': job_id,
                'model_type': model_type,
                'status': 'pending',
                'start_time': datetime.now(),
                'config': config,
                'cloud_providers': config.get('cloud_providers', ['local']),
                'force_retrain': force_retrain
            }
            
            self.active_training_jobs[job_id] = training_job
            
            # Start training process
            training_job['status'] = 'running'
            
            # Use advanced manager for comprehensive training
            if self.advanced_manager:
                result = await self.advanced_manager.train_model(
                    model_id=model_type,
                    force_retrain=force_retrain
                )
                
                # Update job with results
                training_job.update({
                    'status': 'completed' if result.get('success') else 'failed',
                    'end_time': datetime.now(),
                    'metrics': result.get('metrics', {}),
                    'artifacts': result.get('artifacts', {}),
                    'model_version': result.get('version')
                })
                
                # Register model if successful
                if result.get('success'):
                    await self._register_trained_model(model_type, result, training_job)
                
            else:
                # Fallback to basic model manager
                result = await self.model_manager.train_multi_cloud_models(
                    force_retrain=force_retrain
                )
                
                training_job.update({
                    'status': 'completed' if result.get('status') == 'success' else 'failed',
                    'end_time': datetime.now(),
                    'metrics': result.get('metrics', {}),
                    'artifacts': result.get('artifacts', {})
                })
            
            return {
                'job_id': job_id,
                'status': training_job['status'],
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            if job_id in self.active_training_jobs:
                self.active_training_jobs[job_id].update({
                    'status': 'failed',
                    'end_time': datetime.now(),
                    'error': str(e)
                })
            raise
    
    async def _register_trained_model(self, model_type: str, training_result: Dict[str, Any], 
                                    training_job: Dict[str, Any]):
        """Register a newly trained model"""
        try:
            if not self.model_database:
                return
            
            # Create model metadata
            metadata = ModelMetadata(
                model_id=f"{model_type}_{training_result.get('version', 'v1')}",
                model_type=model_type,
                version=training_result.get('version', 'v1'),
                framework='pytorch',
                created_at=datetime.now(),
                metrics=training_result.get('metrics', {}),
                hyperparameters=training_job.get('config', {}).get('hyperparameters', {}),
                training_data_hash=training_result.get('data_hash'),
                model_size_bytes=training_result.get('model_size', 0),
                tags=['trained', 'volcanic_prediction'],
                description=f"Model trained on {datetime.now().isoformat()}"
            )
            
            # Store model in database
            await self.model_database.store_model(
                metadata=metadata,
                model_data=training_result.get('model_data'),
                artifacts=training_result.get('artifacts', {})
            )
            
            # Update registry
            self.model_registry[metadata.model_id] = {
                'metadata': metadata,
                'status': 'trained',
                'last_updated': datetime.now(),
                'training_job_id': training_job['job_id']
            }
            
            logger.info(f"Model {metadata.model_id} registered successfully")
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
    
    async def deploy_model(self, model_id: str, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a model using MLOps pipeline"""
        try:
            deployment_id = f"deploy_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create deployment record
            deployment = {
                'deployment_id': deployment_id,
                'model_id': model_id,
                'status': 'pending',
                'start_time': datetime.now(),
                'config': deployment_config,
                'strategy': deployment_config.get('strategy', 'rolling'),
                'target_environment': deployment_config.get('target_environment', 'staging'),
                'cloud_providers': deployment_config.get('cloud_providers', ['local'])
            }
            
            self.active_deployments[deployment_id] = deployment
            
            # Use MLOps pipeline for deployment
            if self.mlops_pipeline:
                deployment['status'] = 'deploying'
                
                result = await self.mlops_pipeline.deploy_model(
                    model_id=model_id,
                    strategy=DeploymentStrategy(deployment_config.get('strategy', 'rolling')),
                    target_stage=ModelStage(deployment_config.get('target_environment', 'staging')),
                    traffic_percentage=deployment_config.get('traffic_percentage', 100.0)
                )
                
                deployment.update({
                    'status': 'completed' if result.get('success') else 'failed',
                    'end_time': datetime.now(),
                    'endpoint_urls': result.get('endpoints', {}),
                    'deployment_result': result
                })
                
                # Update model registry
                if model_id in self.model_registry:
                    self.model_registry[model_id]['deployment_status'] = deployment['status']
                    self.model_registry[model_id]['deployment_id'] = deployment_id
            
            return {
                'deployment_id': deployment_id,
                'status': deployment['status'],
                'result': deployment.get('deployment_result', {})
            }
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            if deployment_id in self.active_deployments:
                self.active_deployments[deployment_id].update({
                    'status': 'failed',
                    'end_time': datetime.now(),
                    'error': str(e)
                })
            raise
    
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive model performance metrics"""
        try:
            performance = {}
            
            # Get from advanced manager if available
            if self.advanced_manager:
                performance = await self.advanced_manager.get_model_performance(model_id)
            
            # Get from model registry
            if model_id in self.model_registry:
                registry_data = self.model_registry[model_id]
                performance.update({
                    'model_id': model_id,
                    'model_type': registry_data.get('metadata', {}).get('model_type'),
                    'version': registry_data.get('metadata', {}).get('version'),
                    'deployment_status': registry_data.get('deployment_status', 'not_deployed'),
                    'last_updated': registry_data.get('last_updated')
                })
            
            # Get performance history
            if model_id in self.performance_history:
                recent_performance = self.performance_history[model_id][-1] if self.performance_history[model_id] else {}
                performance.update(recent_performance)
            
            # Get drift detection results
            if model_id in self.drift_detection_results:
                performance['drift_score'] = self.drift_detection_results[model_id].get('drift_score')
                performance['is_drifting'] = self.drift_detection_results[model_id].get('is_drifting', False)
            
            return performance
            
        except Exception as e:
            logger.error(f"Performance retrieval failed for model {model_id}: {e}")
            raise
    
    async def predict(self, model_id: str, features: List[float], 
                     uncertainty_estimation: bool = True, n_samples: int = 100) -> Dict[str, Any]:
        """Make predictions using specified model"""
        try:
            start_time = datetime.now()
            
            # Get model from manager
            if not self.model_manager or model_id not in self.model_manager.models:
                raise ValueError(f"Model {model_id} not found or not loaded")
            
            model = self.model_manager.models[model_id]
            scaler = self.model_manager.scalers.get(model_id)
            
            # Prepare features
            features_array = np.array(features).reshape(1, -1)
            if scaler:
                features_array = scaler.transform(features_array)
            
            # Make prediction
            prediction_result = {}
            
            if hasattr(model, 'predict_with_uncertainty') and uncertainty_estimation:
                prediction, uncertainty = model.predict_with_uncertainty(
                    torch.FloatTensor(features_array), 
                    n_samples=n_samples
                )
                prediction_result = {
                    'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                    'uncertainty': uncertainty.tolist() if hasattr(uncertainty, 'tolist') else uncertainty,
                    'confidence': float(1.0 - uncertainty.mean()) if hasattr(uncertainty, 'mean') else None
                }
            else:
                prediction = model.predict(features_array)
                prediction_result = {
                    'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction
                }
            
            # Calculate prediction time
            prediction_time = (datetime.now() - start_time).total_seconds() * 1000
            
            prediction_result.update({
                'model_id': model_id,
                'model_type': self.model_registry.get(model_id, {}).get('metadata', {}).get('model_type'),
                'model_version': self.model_registry.get(model_id, {}).get('metadata', {}).get('version'),
                'prediction_time_ms': prediction_time,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update performance metrics
            await self._update_prediction_metrics(model_id, prediction_time)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Prediction failed for model {model_id}: {e}")
            raise
    
    async def _update_prediction_metrics(self, model_id: str, prediction_time_ms: float):
        """Update prediction performance metrics"""
        try:
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            
            # Add current prediction metrics
            metrics = {
                'timestamp': datetime.now(),
                'prediction_time_ms': prediction_time_ms,
                'throughput_rps': 1000.0 / prediction_time_ms if prediction_time_ms > 0 else 0
            }
            
            self.performance_history[model_id].append(metrics)
            
            # Keep only recent history (last 1000 predictions)
            if len(self.performance_history[model_id]) > 1000:
                self.performance_history[model_id] = self.performance_history[model_id][-1000:]
            
        except Exception as e:
            logger.error(f"Metrics update failed for model {model_id}: {e}")
    
    async def detect_drift(self, model_id: str, reference_data: np.ndarray, 
                          current_data: np.ndarray) -> Dict[str, Any]:
        """Detect data drift for a model"""
        try:
            if not self.mlops_pipeline:
                raise ValueError("MLOps pipeline not available for drift detection")
            
            # Use MLOps pipeline for drift detection
            drift_result = await self.mlops_pipeline.detect_drift(
                model_id=model_id,
                reference_data=reference_data,
                current_data=current_data
            )
            
            # Store drift detection results
            self.drift_detection_results[model_id] = {
                'drift_score': drift_result.get('drift_score', 0.0),
                'is_drifting': drift_result.get('is_drifting', False),
                'drift_type': drift_result.get('drift_type', 'unknown'),
                'detection_time': datetime.now(),
                'affected_features': drift_result.get('affected_features', []),
                'confidence': drift_result.get('confidence', 0.0)
            }
            
            # Trigger retraining if significant drift detected
            if drift_result.get('is_drifting') and drift_result.get('drift_score', 0) > 0.5:
                logger.warning(f"Significant drift detected for model {model_id}, triggering retraining")
                await self._trigger_automatic_retraining(model_id, "drift_detected")
            
            return self.drift_detection_results[model_id]
            
        except Exception as e:
            logger.error(f"Drift detection failed for model {model_id}: {e}")
            raise
    
    async def _trigger_automatic_retraining(self, model_id: str, reason: str):
        """Trigger automatic model retraining"""
        try:
            if not self.advanced_manager:
                return
            
            logger.info(f"Triggering automatic retraining for model {model_id}, reason: {reason}")
            
            # Use advanced manager for retraining
            retraining_result = await self.advanced_manager.trigger_retraining(
                model_id=model_id,
                reason=reason
            )
            
            logger.info(f"Automatic retraining initiated for model {model_id}: {retraining_result}")
            
        except Exception as e:
            logger.error(f"Automatic retraining failed for model {model_id}: {e}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all models in registry"""
        try:
            models = []
            
            for model_id, registry_data in self.model_registry.items():
                metadata = registry_data.get('metadata', {})
                models.append({
                    'model_id': model_id,
                    'model_type': metadata.get('model_type'),
                    'version': metadata.get('version'),
                    'status': registry_data.get('status'),
                    'deployment_status': registry_data.get('deployment_status', 'not_deployed'),
                    'created_at': metadata.get('created_at'),
                    'last_updated': registry_data.get('last_updated'),
                    'metrics': metadata.get('metrics', {}),
                    'size_mb': metadata.get('model_size_bytes', 0) / (1024 * 1024) if metadata.get('model_size_bytes') else None
                })
            
            return models
            
        except Exception as e:
            logger.error(f"Model listing failed: {e}")
            raise
    
    async def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Delete a model from all systems"""
        try:
            # Remove from model manager
            if self.model_manager and model_id in self.model_manager.models:
                del self.model_manager.models[model_id]
                if model_id in self.model_manager.scalers:
                    del self.model_manager.scalers[model_id]
                if model_id in self.model_manager.model_metadata:
                    del self.model_manager.model_metadata[model_id]
            
            # Remove from database
            if self.model_database:
                await self.model_database.delete_model(model_id)
            
            # Remove from registry
            if model_id in self.model_registry:
                del self.model_registry[model_id]
            
            # Clean up performance history
            if model_id in self.performance_history:
                del self.performance_history[model_id]
            
            # Clean up drift detection results
            if model_id in self.drift_detection_results:
                del self.drift_detection_results[model_id]
            
            return {
                'model_id': model_id,
                'status': 'deleted',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Model deletion failed for {model_id}: {e}")
            raise
    
    async def get_training_jobs(self) -> List[Dict[str, Any]]:
        """Get all training jobs"""
        return list(self.active_training_jobs.values())
    
    async def get_deployments(self) -> List[Dict[str, Any]]:
        """Get all deployments"""
        return list(self.active_deployments.values())
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'model_count': len(self.model_registry),
                'active_training_jobs': len([j for j in self.active_training_jobs.values() if j['status'] == 'running']),
                'active_deployments': len([d for d in self.active_deployments.values() if d['status'] in ['deploying', 'completed']]),
                'models_with_drift': len([m for m in self.drift_detection_results.values() if m.get('is_drifting', False)])
            }
            
            # Add advanced metrics if available
            if self.advanced_manager:
                advanced_metrics = await self.advanced_manager.get_system_metrics()
                metrics.update(advanced_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"System metrics retrieval failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown all components"""
        try:
            if self.mlops_pipeline:
                await self.mlops_pipeline.shutdown()
            if self.model_database:
                await self.model_database.close()
            
            logger.info("Model lifecycle manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")