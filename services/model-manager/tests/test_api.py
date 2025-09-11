"""
Integration tests for Model Management Service API
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
import sys
import os

# Add the service to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import app
from api.models import ModelType, ModelStage, DeploymentStrategy

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def sample_training_request():
    """Sample training request"""
    return {
        "model_type": "eruption",
        "force_retrain": False,
        "cloud_providers": ["local"],
        "experiment_name": "test_experiment",
        "hyperparameter_tuning": True,
        "cross_validation_folds": 5
    }

@pytest.fixture
def sample_deployment_request():
    """Sample deployment request"""
    return {
        "model_id": "eruption_v1",
        "deployment_strategy": "rolling",
        "target_environment": "staging",
        "cloud_providers": ["local"],
        "traffic_percentage": 100.0
    }

@pytest.fixture
def sample_prediction_request():
    """Sample prediction request"""
    return {
        "model_type": "eruption",
        "features": [1.2, 3.4, 5.6, 7.8, 9.0, 2.1, 4.3, 6.5, 8.7, 0.9],
        "uncertainty_estimation": True,
        "n_samples": 100
    }

class TestModelManagementAPI:
    """Test cases for Model Management API"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        with patch('main.model_manager', Mock()), \
             patch('main.mlops_pipeline', Mock()), \
             patch('main.model_database', Mock()), \
             patch('main.advanced_manager', Mock()):
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "version" in data
            assert "components" in data
    
    def test_readiness_check_ready(self, client):
        """Test readiness check when service is ready"""
        with patch('main.model_manager', Mock()), \
             patch('main.mlops_pipeline', Mock()), \
             patch('main.model_database', Mock()):
            
            response = client.get("/ready")
            assert response.status_code == 200
            assert response.json()["status"] == "ready"
    
    def test_readiness_check_not_ready(self, client):
        """Test readiness check when service is not ready"""
        with patch('main.model_manager', None), \
             patch('main.mlops_pipeline', None), \
             patch('main.model_database', None):
            
            response = client.get("/ready")
            assert response.status_code == 503
    
    def test_train_model_success(self, client, sample_training_request):
        """Test successful model training request"""
        mock_manager = Mock()
        
        with patch('main.model_manager', mock_manager):
            response = client.post("/models/train", json=sample_training_request)
            assert response.status_code == 200
            
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "started"
            assert "message" in data
    
    def test_train_model_service_unavailable(self, client, sample_training_request):
        """Test training request when service is unavailable"""
        with patch('main.model_manager', None):
            response = client.post("/models/train", json=sample_training_request)
            assert response.status_code == 503
    
    def test_deploy_model_success(self, client, sample_deployment_request):
        """Test successful model deployment"""
        mock_manager = Mock()
        mock_pipeline = Mock()
        mock_pipeline.deploy_model = AsyncMock(return_value={
            "deployment_id": "deploy_123",
            "status": "completed"
        })
        
        with patch('main.model_manager', mock_manager), \
             patch('main.mlops_pipeline', mock_pipeline):
            
            response = client.post("/models/deploy", json=sample_deployment_request)
            assert response.status_code == 200
            
            data = response.json()
            assert "deployment_id" in data
            assert "status" in data
    
    def test_get_model_performance_success(self, client):
        """Test model performance retrieval"""
        mock_advanced = Mock()
        mock_advanced.get_model_performance = AsyncMock(return_value={
            "model_type": "eruption",
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "latency_ms": 45.2,
            "throughput_rps": 22.1,
            "last_updated": datetime.now(),
            "deployment_status": "deployed"
        })
        
        with patch('main.advanced_manager', mock_advanced):
            response = client.get("/models/eruption_v1/performance")
            assert response.status_code == 200
            
            data = response.json()
            assert data["model_id"] == "eruption_v1"
            assert data["accuracy"] == 0.85
            assert data["deployment_status"] == "deployed"
    
    def test_predict_with_uncertainty(self, client, sample_prediction_request):
        """Test prediction with uncertainty estimation"""
        mock_model = Mock()
        mock_model.predict_with_uncertainty = Mock(return_value=(
            Mock(tolist=Mock(return_value=[0.75])),
            Mock(tolist=Mock(return_value=[0.1]), mean=Mock(return_value=0.1))
        ))
        
        mock_manager = Mock()
        mock_manager.models = {"eruption": mock_model}
        mock_manager.scalers = {}
        
        with patch('main.model_manager', mock_manager):
            response = client.post("/models/predict", json=sample_prediction_request)
            assert response.status_code == 200
            
            data = response.json()
            assert "prediction" in data
            assert "uncertainty" in data
            assert data["model_type"] == "eruption"
    
    def test_predict_without_uncertainty(self, client):
        """Test prediction without uncertainty estimation"""
        request_data = {
            "model_type": "eruption",
            "features": [1.2, 3.4, 5.6, 7.8, 9.0],
            "uncertainty_estimation": False
        }
        
        mock_model = Mock()
        mock_model.predict = Mock(return_value=Mock(tolist=Mock(return_value=[0.75])))
        
        mock_manager = Mock()
        mock_manager.models = {"eruption": mock_model}
        mock_manager.scalers = {}
        
        with patch('main.model_manager', mock_manager):
            response = client.post("/models/predict", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "prediction" in data
            assert "uncertainty" not in data
    
    def test_predict_model_not_found(self, client, sample_prediction_request):
        """Test prediction with non-existent model"""
        mock_manager = Mock()
        mock_manager.models = {}  # No models available
        
        with patch('main.model_manager', mock_manager):
            response = client.post("/models/predict", json=sample_prediction_request)
            assert response.status_code == 404
    
    def test_list_models_success(self, client):
        """Test model listing"""
        mock_manager = Mock()
        mock_manager.models = {
            "eruption": Mock(__class__=Mock(__name__="VolcanicPredictionNet")),
            "vei": Mock(__class__=Mock(__name__="MultiTaskVolcanicNet"))
        }
        mock_manager.model_metadata = {
            "eruption": {
                "last_trained": "2024-01-01T00:00:00",
                "accuracy": 0.85,
                "version": "v1"
            },
            "vei": {
                "last_trained": "2024-01-02T00:00:00",
                "accuracy": 0.78,
                "version": "v2"
            }
        }
        
        with patch('main.model_manager', mock_manager):
            response = client.get("/models")
            assert response.status_code == 200
            
            data = response.json()
            assert "models" in data
            assert len(data["models"]) == 2
            
            eruption_model = next(m for m in data["models"] if m["model_type"] == "eruption")
            assert eruption_model["accuracy"] == 0.85
    
    def test_list_training_jobs(self, client):
        """Test training job listing"""
        mock_manager = Mock()
        mock_manager.training_jobs = {
            "job_1": Mock(
                job_id="job_1",
                model_type="eruption",
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                cloud_provider="local",
                metrics={"accuracy": 0.85},
                artifacts={"model_path": "/path/to/model"}
            )
        }
        
        with patch('main.model_manager', mock_manager):
            response = client.get("/training-jobs")
            assert response.status_code == 200
            
            data = response.json()
            assert "training_jobs" in data
            assert len(data["training_jobs"]) == 1
    
    def test_delete_model_success(self, client):
        """Test successful model deletion"""
        mock_manager = Mock()
        mock_manager.models = {"eruption_v1": Mock()}
        mock_manager.scalers = {"eruption_v1": Mock()}
        mock_manager.model_metadata = {"eruption_v1": {}}
        
        with patch('main.model_manager', mock_manager):
            response = client.delete("/models/eruption_v1")
            assert response.status_code == 200
            
            data = response.json()
            assert "message" in data
            assert "eruption_v1" in data["message"]
    
    def test_delete_model_not_found(self, client):
        """Test deletion of non-existent model"""
        mock_manager = Mock()
        mock_manager.models = {}  # No models
        
        with patch('main.model_manager', mock_manager):
            response = client.delete("/models/nonexistent")
            assert response.status_code == 404
    
    def test_get_metrics_success(self, client):
        """Test metrics retrieval"""
        mock_advanced = Mock()
        mock_advanced.get_system_metrics = AsyncMock(return_value={
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "model_count": 5,
            "active_deployments": 2
        })
        
        with patch('main.advanced_manager', mock_advanced):
            response = client.get("/metrics")
            assert response.status_code == 200
            
            data = response.json()
            assert "cpu_usage" in data
            assert "model_count" in data
    
    def test_invalid_model_type(self, client):
        """Test request with invalid model type"""
        invalid_request = {
            "model_type": "invalid_type",
            "features": [1.0, 2.0, 3.0]
        }
        
        response = client.post("/models/predict", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_missing_required_fields(self, client):
        """Test request with missing required fields"""
        incomplete_request = {
            "model_type": "eruption"
            # Missing features field
        }
        
        response = client.post("/models/predict", json=incomplete_request)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_deployment_strategy(self, client):
        """Test deployment with invalid strategy"""
        invalid_deployment = {
            "model_id": "eruption_v1",
            "deployment_strategy": "invalid_strategy",
            "target_environment": "staging"
        }
        
        response = client.post("/models/deploy", json=invalid_deployment)
        assert response.status_code == 422  # Validation error

if __name__ == '__main__':
    pytest.main([__file__])