"""
Unit tests for Model Lifecycle Management
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add the service to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.model_lifecycle import ModelLifecycleManager
from api.models import ModelType, ModelStage, DeploymentStrategy

@pytest.fixture
async def lifecycle_manager():
    """Create a test lifecycle manager"""
    manager = ModelLifecycleManager()
    
    # Mock the initialization to avoid external dependencies
    with patch.object(manager, 'initialize') as mock_init:
        mock_init.return_value = None
        await manager.initialize({})
    
    return manager

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'model_type': 'eruption',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        },
        'cloud_providers': ['local'],
        'enable_uncertainty_quantification': True,
        'enable_physics_constraints': True
    }

@pytest.fixture
def sample_features():
    """Sample features for prediction testing"""
    return [1.2, 3.4, 5.6, 7.8, 9.0, 2.1, 4.3, 6.5, 8.7, 0.9]

class TestModelLifecycleManager:
    """Test cases for ModelLifecycleManager"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test manager initialization"""
        manager = ModelLifecycleManager()
        
        config = {
            'database_url': 'sqlite:///test.db',
            'model_storage_path': './test_models',
            'multi_cloud_enabled': False,
            'auto_deploy': False
        }
        
        # Mock all external dependencies
        with patch('core.model_lifecycle.ModelDatabase') as mock_db, \
             patch('core.model_lifecycle.MLOpsPipeline') as mock_pipeline, \
             patch('core.model_lifecycle.AdvancedMLModelManager') as mock_advanced, \
             patch('core.model_lifecycle.ModelManager') as mock_manager:
            
            # Setup mocks
            mock_db.return_value.initialize = AsyncMock()
            mock_pipeline.return_value.initialize = AsyncMock()
            
            await manager.initialize(config)
            
            assert manager.model_database is not None
            assert manager.mlops_pipeline is not None
            assert manager.advanced_manager is not None
            assert manager.model_manager is not None
    
    @pytest.mark.asyncio
    async def test_train_model_success(self, lifecycle_manager, sample_config):
        """Test successful model training"""
        # Mock advanced manager
        lifecycle_manager.advanced_manager = Mock()
        lifecycle_manager.advanced_manager.train_model = AsyncMock(return_value={
            'success': True,
            'version': 'v1.0',
            'metrics': {'accuracy': 0.85, 'loss': 0.15},
            'artifacts': {'model_path': '/path/to/model'},
            'model_data': b'mock_model_data'
        })
        
        # Mock model database
        lifecycle_manager.model_database = Mock()
        lifecycle_manager.model_database.store_model = AsyncMock()
        
        result = await lifecycle_manager.train_model('eruption', sample_config)
        
        assert 'job_id' in result
        assert result['status'] == 'completed'
        assert 'result' in result
        
        # Check that training job was recorded
        job_id = result['job_id']
        assert job_id in lifecycle_manager.active_training_jobs
        assert lifecycle_manager.active_training_jobs[job_id]['status'] == 'completed'
    
    @pytest.mark.asyncio
    async def test_train_model_failure(self, lifecycle_manager, sample_config):
        """Test model training failure"""
        # Mock advanced manager to raise exception
        lifecycle_manager.advanced_manager = Mock()
        lifecycle_manager.advanced_manager.train_model = AsyncMock(
            side_effect=Exception("Training failed")
        )
        
        with pytest.raises(Exception, match="Training failed"):
            await lifecycle_manager.train_model('eruption', sample_config)
        
        # Check that failure was recorded
        failed_jobs = [job for job in lifecycle_manager.active_training_jobs.values() 
                      if job['status'] == 'failed']
        assert len(failed_jobs) > 0
    
    @pytest.mark.asyncio
    async def test_deploy_model_success(self, lifecycle_manager):
        """Test successful model deployment"""
        # Setup model in registry
        model_id = 'eruption_v1'
        lifecycle_manager.model_registry[model_id] = {
            'metadata': {'model_type': 'eruption', 'version': 'v1'},
            'status': 'trained'
        }
        
        # Mock MLOps pipeline
        lifecycle_manager.mlops_pipeline = Mock()
        lifecycle_manager.mlops_pipeline.deploy_model = AsyncMock(return_value={
            'success': True,
            'endpoints': {'staging': 'http://staging.example.com/predict'}
        })
        
        deployment_config = {
            'strategy': 'rolling',
            'target_environment': 'staging',
            'traffic_percentage': 100.0
        }
        
        result = await lifecycle_manager.deploy_model(model_id, deployment_config)
        
        assert 'deployment_id' in result
        assert result['status'] == 'completed'
        
        # Check deployment was recorded
        deployment_id = result['deployment_id']
        assert deployment_id in lifecycle_manager.active_deployments
    
    @pytest.mark.asyncio
    async def test_get_model_performance(self, lifecycle_manager):
        """Test model performance retrieval"""
        model_id = 'eruption_v1'
        
        # Setup model in registry
        lifecycle_manager.model_registry[model_id] = {
            'metadata': {
                'model_type': 'eruption',
                'version': 'v1'
            },
            'deployment_status': 'deployed',
            'last_updated': datetime.now()
        }
        
        # Setup performance history
        lifecycle_manager.performance_history[model_id] = [{
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85,
            'latency_ms': 45.2
        }]
        
        # Mock advanced manager
        lifecycle_manager.advanced_manager = Mock()
        lifecycle_manager.advanced_manager.get_model_performance = AsyncMock(return_value={
            'throughput_rps': 22.1,
            'memory_usage_mb': 128.5
        })
        
        performance = await lifecycle_manager.get_model_performance(model_id)
        
        assert performance['model_id'] == model_id
        assert performance['model_type'] == 'eruption'
        assert performance['accuracy'] == 0.85
        assert performance['throughput_rps'] == 22.1
    
    @pytest.mark.asyncio
    async def test_predict_with_uncertainty(self, lifecycle_manager, sample_features):
        """Test prediction with uncertainty estimation"""
        model_id = 'eruption_v1'
        
        # Mock model manager
        mock_model = Mock()
        mock_model.predict_with_uncertainty = Mock(return_value=(
            np.array([0.75]),  # prediction
            np.array([0.1])    # uncertainty
        ))
        
        mock_scaler = Mock()
        mock_scaler.transform = Mock(return_value=np.array([sample_features]))
        
        lifecycle_manager.model_manager = Mock()
        lifecycle_manager.model_manager.models = {model_id: mock_model}
        lifecycle_manager.model_manager.scalers = {model_id: mock_scaler}
        
        # Setup model in registry
        lifecycle_manager.model_registry[model_id] = {
            'metadata': {
                'model_type': 'eruption',
                'version': 'v1'
            }
        }
        
        result = await lifecycle_manager.predict(
            model_id, sample_features, uncertainty_estimation=True
        )
        
        assert 'prediction' in result
        assert 'uncertainty' in result
        assert 'confidence' in result
        assert result['model_id'] == model_id
        assert 'prediction_time_ms' in result
    
    @pytest.mark.asyncio
    async def test_predict_without_uncertainty(self, lifecycle_manager, sample_features):
        """Test prediction without uncertainty estimation"""
        model_id = 'eruption_v1'
        
        # Mock model manager
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.75]))
        
        lifecycle_manager.model_manager = Mock()
        lifecycle_manager.model_manager.models = {model_id: mock_model}
        lifecycle_manager.model_manager.scalers = {}
        
        # Setup model in registry
        lifecycle_manager.model_registry[model_id] = {
            'metadata': {
                'model_type': 'eruption',
                'version': 'v1'
            }
        }
        
        result = await lifecycle_manager.predict(
            model_id, sample_features, uncertainty_estimation=False
        )
        
        assert 'prediction' in result
        assert 'uncertainty' not in result
        assert result['model_id'] == model_id
    
    @pytest.mark.asyncio
    async def test_detect_drift(self, lifecycle_manager):
        """Test drift detection"""
        model_id = 'eruption_v1'
        reference_data = np.random.randn(100, 10)
        current_data = np.random.randn(100, 10) + 0.5  # Shifted data
        
        # Mock MLOps pipeline
        lifecycle_manager.mlops_pipeline = Mock()
        lifecycle_manager.mlops_pipeline.detect_drift = AsyncMock(return_value={
            'drift_score': 0.3,
            'is_drifting': True,
            'drift_type': 'data',
            'affected_features': ['feature_1', 'feature_3'],
            'confidence': 0.85
        })
        
        # Mock advanced manager for retraining
        lifecycle_manager.advanced_manager = Mock()
        lifecycle_manager.advanced_manager.trigger_retraining = AsyncMock()
        
        result = await lifecycle_manager.detect_drift(model_id, reference_data, current_data)
        
        assert result['drift_score'] == 0.3
        assert result['is_drifting'] is True
        assert result['drift_type'] == 'data'
        assert 'detection_time' in result
        
        # Check that drift was stored
        assert model_id in lifecycle_manager.drift_detection_results
    
    @pytest.mark.asyncio
    async def test_list_models(self, lifecycle_manager):
        """Test model listing"""
        # Setup models in registry
        lifecycle_manager.model_registry = {
            'eruption_v1': {
                'metadata': {
                    'model_type': 'eruption',
                    'version': 'v1',
                    'created_at': datetime.now(),
                    'metrics': {'accuracy': 0.85},
                    'model_size_bytes': 1024 * 1024  # 1MB
                },
                'status': 'trained',
                'deployment_status': 'deployed',
                'last_updated': datetime.now()
            },
            'vei_v2': {
                'metadata': {
                    'model_type': 'vei',
                    'version': 'v2',
                    'created_at': datetime.now(),
                    'metrics': {'accuracy': 0.78}
                },
                'status': 'training',
                'deployment_status': 'not_deployed',
                'last_updated': datetime.now()
            }
        }
        
        models = await lifecycle_manager.list_models()
        
        assert len(models) == 2
        
        eruption_model = next(m for m in models if m['model_id'] == 'eruption_v1')
        assert eruption_model['model_type'] == 'eruption'
        assert eruption_model['deployment_status'] == 'deployed'
        assert eruption_model['size_mb'] == 1.0
        
        vei_model = next(m for m in models if m['model_id'] == 'vei_v2')
        assert vei_model['model_type'] == 'vei'
        assert vei_model['deployment_status'] == 'not_deployed'
    
    @pytest.mark.asyncio
    async def test_delete_model(self, lifecycle_manager):
        """Test model deletion"""
        model_id = 'eruption_v1'
        
        # Setup model in all systems
        lifecycle_manager.model_registry[model_id] = {'metadata': {}}
        lifecycle_manager.performance_history[model_id] = []
        lifecycle_manager.drift_detection_results[model_id] = {}
        
        # Mock model manager
        lifecycle_manager.model_manager = Mock()
        lifecycle_manager.model_manager.models = {model_id: Mock()}
        lifecycle_manager.model_manager.scalers = {model_id: Mock()}
        lifecycle_manager.model_manager.model_metadata = {model_id: {}}
        
        # Mock model database
        lifecycle_manager.model_database = Mock()
        lifecycle_manager.model_database.delete_model = AsyncMock()
        
        result = await lifecycle_manager.delete_model(model_id)
        
        assert result['model_id'] == model_id
        assert result['status'] == 'deleted'
        
        # Check that model was removed from all systems
        assert model_id not in lifecycle_manager.model_registry
        assert model_id not in lifecycle_manager.performance_history
        assert model_id not in lifecycle_manager.drift_detection_results
    
    @pytest.mark.asyncio
    async def test_get_system_metrics(self, lifecycle_manager):
        """Test system metrics retrieval"""
        # Setup test data
        lifecycle_manager.model_registry = {'model1': {}, 'model2': {}}
        lifecycle_manager.active_training_jobs = {
            'job1': {'status': 'running'},
            'job2': {'status': 'completed'}
        }
        lifecycle_manager.active_deployments = {
            'deploy1': {'status': 'completed'},
            'deploy2': {'status': 'deploying'}
        }
        lifecycle_manager.drift_detection_results = {
            'model1': {'is_drifting': True},
            'model2': {'is_drifting': False}
        }
        
        # Mock advanced manager
        lifecycle_manager.advanced_manager = Mock()
        lifecycle_manager.advanced_manager.get_system_metrics = AsyncMock(return_value={
            'cpu_usage': 45.2,
            'memory_usage': 67.8,
            'gpu_usage': 23.1
        })
        
        metrics = await lifecycle_manager.get_system_metrics()
        
        assert metrics['model_count'] == 2
        assert metrics['active_training_jobs'] == 1
        assert metrics['active_deployments'] == 2
        assert metrics['models_with_drift'] == 1
        assert metrics['cpu_usage'] == 45.2

if __name__ == '__main__':
    pytest.main([__file__])