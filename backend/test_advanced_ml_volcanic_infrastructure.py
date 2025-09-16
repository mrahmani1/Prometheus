"""
Test suite for Advanced ML Volcanic Infrastructure

This test suite validates the functionality of the advanced machine learning infrastructure
for volcanic data analysis, including physics-informed neural networks, ensemble methods,
explainable AI, real-time inference, hyperparameter optimization, model versioning,
and continuous learning frameworks.
"""

import pytest
import asyncio
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.advanced_ml_volcanic_infrastructure import (
    AdvancedMLVolcanicInfrastructure,
    PhysicsInformedNeuralNetwork,
    EnsembleVolcanicPredictor,
    ExplainableAI,
    RealTimeInferenceEngine,
    HyperparameterOptimizer,
    ModelVersioningSystem,
    ContinuousLearningFramework,
    VolcanicDataset,
    MLModelConfig,
    TrainingResult,
    InferenceResult
)

class TestVolcanicDataset:
    """Test VolcanicDataset dataclass"""
    
    def test_volcanic_dataset_creation(self):
        """Test VolcanicDataset creation"""
        features = np.random.rand(100, 10)
        targets = np.random.rand(100, 1)
        feature_names = [f'feature_{i}' for i in range(10)]
        target_names = ['eruption_probability']
        metadata = {'dataset_type': 'test'}
        
        dataset = VolcanicDataset(
            features=features,
            targets=targets,
            feature_names=feature_names,
            target_names=target_names,
            metadata=metadata
        )
        
        assert dataset.features.shape == (100, 10)
        assert dataset.targets.shape == (100, 1)
        assert len(dataset.feature_names) == 10
        assert dataset.metadata['dataset_type'] == 'test'

class TestPhysicsInformedNeuralNetwork:
    """Test Physics-Informed Neural Network"""
    
    def setup_method(self):
        """Setup test environment"""
        self.input_dim = 10
        self.output_dim = 1
        self.model = PhysicsInformedNeuralNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=[32, 16],
            physics_weight=1.0,
            learning_rate=0.001
        )
    
    def test_model_initialization(self):
        """Test model initialization"""
        assert self.model.input_dim == self.input_dim
        assert self.model.output_dim == self.output_dim
        assert self.model.physics_weight == 1.0
        assert self.model.learning_rate == 0.001
    
    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 5
        x = torch.randn(batch_size, self.input_dim)
        
        output, physics_output = self.model(x)
        
        assert output.shape == (batch_size, self.output_dim)
        assert physics_output.shape == (batch_size, self.output_dim)
    
    def test_physics_loss_calculation(self):
        """Test physics loss calculation"""
        batch_size = 5
        predictions = torch.randn(batch_size, self.output_dim)
        physics_predictions = torch.randn(batch_size, self.output_dim)
        inputs = torch.randn(batch_size, self.input_dim)
        
        physics_loss = self.model.physics_loss(predictions, physics_predictions, inputs)
        
        assert isinstance(physics_loss, torch.Tensor)
        assert physics_loss.dim() == 0  # Scalar
        assert physics_loss >= 0  # Loss should be non-negative
    
    def test_training_step(self):
        """Test training step"""
        batch_size = 5
        x = torch.randn(batch_size, self.input_dim)
        y = torch.randn(batch_size, self.output_dim)
        batch = (x, y)
        
        loss = self.model.training_step(batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss >= 0

class TestEnsembleVolcanicPredictor:
    """Test Ensemble Volcanic Predictor"""
    
    def setup_method(self):
        """Setup test environment"""
        self.ensemble = EnsembleVolcanicPredictor(['random_forest', 'gradient_boosting'])
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization"""
        assert 'random_forest' in self.ensemble.base_models
        assert 'gradient_boosting' in self.ensemble.base_models
        assert not self.ensemble.is_trained
    
    def test_create_base_model(self):
        """Test base model creation"""
        rf_model = self.ensemble._create_base_model('random_forest', n_estimators=50)
        gb_model = self.ensemble._create_base_model('gradient_boosting', n_estimators=50)
        
        assert rf_model is not None
        assert gb_model is not None
        assert hasattr(rf_model, 'fit')
        assert hasattr(gb_model, 'fit')
    
    @pytest.mark.asyncio
    async def test_ensemble_training(self):
        """Test ensemble training"""
        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        y_train = np.sum(X_train, axis=1) + np.random.normal(0, 0.1, 100)
        X_val = np.random.rand(20, 5)
        y_val = np.sum(X_val, axis=1) + np.random.normal(0, 0.1, 20)
        
        # Train ensemble
        results = await self.ensemble.train_ensemble(X_train, y_train, X_val, y_val)
        
        assert self.ensemble.is_trained
        assert 'random_forest' in results
        assert 'gradient_boosting' in results
        assert 'ensemble' in results
        assert 'validation_mse' in results['ensemble']
        assert results['ensemble']['validation_mse'] >= 0
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction(self):
        """Test ensemble prediction"""
        # Create and train ensemble
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        y_train = np.sum(X_train, axis=1)
        X_val = np.random.rand(20, 5)
        y_val = np.sum(X_val, axis=1)
        
        await self.ensemble.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Test prediction
        X_test = np.random.rand(10, 5)
        predictions = await self.ensemble.predict(X_test)
        
        assert predictions.shape == (10,)
        assert not np.isnan(predictions).any()
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        # Train a simple ensemble first
        np.random.seed(42)
        X_train = np.random.rand(50, 3)
        y_train = np.sum(X_train, axis=1)
        
        # Create and fit models manually for testing
        rf_model = self.ensemble._create_base_model('random_forest', n_estimators=10)
        rf_model.fit(X_train, y_train)
        self.ensemble.models['random_forest'] = rf_model
        self.ensemble.is_trained = True
        
        importance_dict = self.ensemble.get_feature_importance()
        
        assert 'random_forest' in importance_dict
        assert len(importance_dict['random_forest']) == 3

class TestExplainableAI:
    """Test Explainable AI functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.explainer = ExplainableAI()
    
    def test_explainer_initialization(self):
        """Test explainer initialization"""
        assert hasattr(self.explainer, 'explainers')
        assert isinstance(self.explainer.explainers, dict)
    
    @patch('shap.Explainer')
    @patch('lime.lime_tabular.LimeTabularExplainer')
    def test_setup_explainers(self, mock_lime, mock_shap):
        """Test explainer setup"""
        # Create mock model
        model = Mock()
        model.predict = Mock(return_value=np.array([1.0, 2.0, 3.0]))
        
        X_train = np.random.rand(10, 5)
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        self.explainer.setup_explainers(model, X_train, feature_names)
        
        mock_shap.assert_called_once_with(model, X_train)
        mock_lime.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('shap.Explainer')
    async def test_explain_prediction_shap(self, mock_shap_class):
        """Test SHAP explanation"""
        # Setup mock SHAP explainer
        mock_explainer = Mock()
        mock_shap_values = Mock()
        mock_shap_values.values = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_shap_values.base_values = np.array([0.0])
        mock_shap_values.data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        mock_explainer.return_value = mock_shap_values
        
        self.explainer.explainers['shap'] = mock_explainer
        
        # Test explanation
        model = Mock()
        X_instance = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        explanation = await self.explainer.explain_prediction(model, X_instance, method='shap')
        
        assert explanation['method'] == 'shap'
        assert 'feature_importance' in explanation
        assert 'base_value' in explanation
        assert 'data' in explanation

class TestRealTimeInferenceEngine:
    """Test Real-Time Inference Engine"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.inference_engine = RealTimeInferenceEngine(model_registry_path=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_inference_engine_initialization(self):
        """Test inference engine initialization"""
        assert hasattr(self.inference_engine, 'model_registry_path')
        assert hasattr(self.inference_engine, 'loaded_models')
        assert hasattr(self.inference_engine, 'device')
    
    @pytest.mark.asyncio
    async def test_load_pytorch_model(self):
        """Test PyTorch model loading"""
        # Create a simple PyTorch model
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
        # Save model
        model_path = Path(self.temp_dir) / "test_model.pth"
        torch.save(model, model_path)
        
        # Test loading
        success = await self.inference_engine.load_model("test_model", "test_model.pth")
        
        assert success
        assert "test_model" in self.inference_engine.loaded_models
    
    @pytest.mark.asyncio
    async def test_load_sklearn_model(self):
        """Test scikit-learn model loading"""
        from sklearn.linear_model import LinearRegression
        import joblib
        
        # Create and save sklearn model
        model = LinearRegression()
        X_dummy = np.random.rand(10, 3)
        y_dummy = np.random.rand(10)
        model.fit(X_dummy, y_dummy)
        
        model_path = Path(self.temp_dir) / "test_sklearn_model.pkl"
        joblib.dump(model, model_path)
        
        # Test loading
        success = await self.inference_engine.load_model("test_sklearn", "test_sklearn_model.pkl")
        
        assert success
        assert "test_sklearn" in self.inference_engine.loaded_models
    
    @pytest.mark.asyncio
    async def test_batch_inference_sklearn(self):
        """Test batch inference with sklearn model"""
        from sklearn.linear_model import LinearRegression
        import joblib
        
        # Create, train, and save model
        model = LinearRegression()
        X_train = np.random.rand(50, 3)
        y_train = np.sum(X_train, axis=1) + np.random.normal(0, 0.1, 50)
        model.fit(X_train, y_train)
        
        model_path = Path(self.temp_dir) / "inference_test_model.pkl"
        joblib.dump(model, model_path)
        
        # Load model
        await self.inference_engine.load_model("inference_test", "inference_test_model.pkl")
        
        # Test inference
        X_test = np.random.rand(5, 3)
        result = await self.inference_engine.predict_batch("inference_test", X_test)
        
        assert isinstance(result, InferenceResult)
        assert result.predictions.shape == (5,)
        assert result.confidence_scores.shape == (5,)
        assert result.uncertainty_bounds.shape == (5, 2)
        assert result.inference_time > 0
    
    def test_get_model_info(self):
        """Test model info retrieval"""
        # Create a simple model and add to loaded models
        model = nn.Linear(5, 1)
        self.inference_engine.loaded_models["test_info"] = model
        
        info = self.inference_engine.get_model_info("test_info")
        
        assert info['model_id'] == "test_info"
        assert info['model_type'] == "Linear"
        assert 'parameters' in info
        assert 'trainable_parameters' in info

class TestHyperparameterOptimizer:
    """Test Hyperparameter Optimizer"""
    
    def setup_method(self):
        """Setup test environment"""
        self.optimizer = HyperparameterOptimizer()
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        assert self.optimizer.optimization_backend == 'optuna'
        assert self.optimizer.study is None
    
    def test_create_study(self):
        """Test study creation"""
        self.optimizer.create_study("test_study", direction="minimize")
        
        assert self.optimizer.study is not None
        assert self.optimizer.study.study_name == "test_study"
    
    def test_define_search_space(self):
        """Test search space definition"""
        pinn_space = self.optimizer.define_search_space('physics_informed_nn')
        ensemble_space = self.optimizer.define_search_space('ensemble')
        
        assert 'learning_rate' in pinn_space
        assert 'hidden_dims' in pinn_space
        assert 'physics_weight' in pinn_space
        
        assert 'n_estimators' in ensemble_space
        assert 'max_depth' in ensemble_space

class TestModelVersioningSystem:
    """Test Model Versioning System"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock database and Redis
        with patch('sqlalchemy.create_engine'), \
             patch('redis.Redis'):
            self.versioning = ModelVersioningSystem(
                db_url="sqlite:///test.db",
                model_storage_path=self.temp_dir
            )
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_versioning_initialization(self):
        """Test versioning system initialization"""
        assert hasattr(self.versioning, 'model_storage_path')
        assert self.versioning.model_storage_path.exists()
    
    def test_generate_model_id(self):
        """Test model ID generation"""
        model_name = "test_model"
        architecture = {"layers": [10, 5, 1]}
        data_hash = "abc123"
        
        model_id = self.versioning.generate_model_id(model_name, architecture, data_hash)
        
        assert isinstance(model_id, str)
        assert len(model_id) == 16  # SHA256 truncated to 16 chars
    
    def test_calculate_data_hash(self):
        """Test data hash calculation"""
        features = np.random.rand(10, 5)
        targets = np.random.rand(10, 1)
        
        dataset = VolcanicDataset(
            features=features,
            targets=targets,
            feature_names=['f1', 'f2', 'f3', 'f4', 'f5'],
            target_names=['target'],
            metadata={}
        )
        
        data_hash = self.versioning.calculate_data_hash(dataset)
        
        assert isinstance(data_hash, str)
        assert len(data_hash) == 64  # SHA256 hash
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_training_data(self):
        """Test storing and retrieving training data"""
        # Create test dataset
        features = np.random.rand(50, 8)
        targets = np.random.rand(50, 1)
        
        dataset = VolcanicDataset(
            features=features,
            targets=targets,
            feature_names=[f'feature_{i}' for i in range(8)],
            target_names=['target'],
            metadata={'test': True, 'source': 'unit_test'}
        )
        
        model_id = "test_model_123"
        
        # Mock database operations
        with patch.object(self.versioning, 'Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_session.query.return_value.filter_by.return_value.first.return_value = None
            
            # Test storing training data
            data_hash = await self.versioning.store_training_data(dataset, model_id)
            
            assert isinstance(data_hash, str)
            assert len(data_hash) == 64
            
            # Verify files were created
            data_storage_path = self.versioning.model_storage_path / "training_data"
            features_path = data_storage_path / f"{data_hash}_features.pkl.gz"
            targets_path = data_storage_path / f"{data_hash}_targets.pkl.gz"
            
            assert features_path.exists()
            assert targets_path.exists()
            
            # Mock retrieval
            mock_record = Mock()
            mock_record.features_path = f"training_data/{data_hash}_features.pkl.gz"
            mock_record.targets_path = f"training_data/{data_hash}_targets.pkl.gz"
            mock_record.feature_names = dataset.feature_names
            mock_record.target_names = dataset.target_names
            mock_record.metadata = dataset.metadata
            
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_record
            
            # Test retrieving training data
            retrieved_dataset = await self.versioning.retrieve_training_data(model_id)
            
            assert retrieved_dataset is not None
            assert retrieved_dataset.features.shape == features.shape
            assert retrieved_dataset.targets.shape == targets.shape
            assert retrieved_dataset.feature_names == dataset.feature_names
            assert retrieved_dataset.target_names == dataset.target_names
            assert retrieved_dataset.metadata == dataset.metadata

class TestContinuousLearningFramework:
    """Test Continuous Learning Framework"""
    
    def setup_method(self):
        """Setup test environment"""
        # Mock dependencies
        mock_versioning = Mock()
        mock_inference = Mock()
        
        self.continuous_learning = ContinuousLearningFramework(
            mock_versioning, mock_inference
        )
    
    def test_continuous_learning_initialization(self):
        """Test continuous learning initialization"""
        assert hasattr(self.continuous_learning, 'model_versioning')
        assert hasattr(self.continuous_learning, 'inference_engine')
        assert hasattr(self.continuous_learning, 'retraining_threshold')
        assert self.continuous_learning.retraining_threshold == 0.1
    
    @pytest.mark.asyncio
    async def test_full_retraining_with_database_data(self):
        """Test full retraining using database-stored training data"""
        # Create mock original training data
        original_features = np.random.rand(100, 5)
        original_targets = np.random.rand(100, 1)
        original_dataset = VolcanicDataset(
            features=original_features,
            targets=original_targets,
            feature_names=['f1', 'f2', 'f3', 'f4', 'f5'],
            target_names=['target'],
            metadata={'original': True}
        )
        
        # Create new training data
        new_features = np.random.rand(50, 5)
        new_targets = np.random.rand(50, 1)
        new_dataset = VolcanicDataset(
            features=new_features,
            targets=new_targets,
            feature_names=['f1', 'f2', 'f3', 'f4', 'f5'],
            target_names=['target'],
            metadata={'new': True}
        )
        
        # Mock model metadata
        metadata = {
            'model_name': 'test_model',
            'model_type': 'RandomForestRegressor',
            'architecture': {'n_estimators': 100},
            'hyperparameters': {'n_estimators': 100, 'random_state': 42},
            'model_path': 'test_model.pkl'
        }
        
        # Mock model versioning methods
        self.continuous_learning.model_versioning.retrieve_training_data = Mock(
            return_value=original_dataset
        )
        self.continuous_learning.model_versioning.register_model = Mock(
            return_value="new_model_123"
        )
        
        # Mock model storage path
        temp_dir = tempfile.mkdtemp()
        try:
            self.continuous_learning.model_versioning.model_storage_path = Path(temp_dir)
            
            # Create a dummy model file
            from sklearn.ensemble import RandomForestRegressor
            dummy_model = RandomForestRegressor(n_estimators=100, random_state=42)
            dummy_model.fit(np.random.rand(10, 5), np.random.rand(10))
            
            model_path = Path(temp_dir) / 'test_model.pkl'
            joblib.dump(dummy_model, model_path)
            
            # Test full retraining
            new_model_id = await self.continuous_learning._full_retraining(
                "original_model_123", new_dataset, metadata
            )
            
            assert new_model_id == "new_model_123"
            
            # Verify that retrieve_training_data was called
            self.continuous_learning.model_versioning.retrieve_training_data.assert_called_once_with(
                "original_model_123"
            )
            
            # Verify that register_model was called with combined data
            register_call = self.continuous_learning.model_versioning.register_model.call_args
            assert register_call is not None
            
            # Check that the training data passed to register_model has combined samples
            training_data_arg = register_call[1]['training_data']
            expected_total_samples = len(original_features) + len(new_features)
            assert len(training_data_arg.features) == expected_total_samples
            
        finally:
            shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_full_retraining_fallback_to_new_data(self):
        """Test full retraining fallback when no original data is available"""
        # Create new training data
        new_features = np.random.rand(50, 5)
        new_targets = np.random.rand(50, 1)
        new_dataset = VolcanicDataset(
            features=new_features,
            targets=new_targets,
            feature_names=['f1', 'f2', 'f3', 'f4', 'f5'],
            target_names=['target'],
            metadata={'new': True}
        )
        
        # Mock model metadata
        metadata = {
            'model_name': 'test_model',
            'model_type': 'RandomForestRegressor',
            'architecture': {'n_estimators': 100},
            'hyperparameters': {'n_estimators': 100, 'random_state': 42},
            'model_path': 'test_model.pkl'
        }
        
        # Mock model versioning methods - no original data available
        self.continuous_learning.model_versioning.retrieve_training_data = Mock(
            return_value=None
        )
        self.continuous_learning.model_versioning.register_model = Mock(
            return_value="fallback_model_123"
        )
        
        # Mock model storage path
        temp_dir = tempfile.mkdtemp()
        try:
            self.continuous_learning.model_versioning.model_storage_path = Path(temp_dir)
            
            # Create a dummy model file
            from sklearn.ensemble import RandomForestRegressor
            dummy_model = RandomForestRegressor(n_estimators=100, random_state=42)
            dummy_model.fit(np.random.rand(10, 5), np.random.rand(10))
            
            model_path = Path(temp_dir) / 'test_model.pkl'
            joblib.dump(dummy_model, model_path)
            
            # Test full retraining with fallback
            new_model_id = await self.continuous_learning._full_retraining(
                "original_model_123", new_dataset, metadata
            )
            
            assert new_model_id == "fallback_model_123"
            
            # Verify that retrieve_training_data was called
            self.continuous_learning.model_versioning.retrieve_training_data.assert_called_once_with(
                "original_model_123"
            )
            
            # Verify that register_model was called with only new data
            register_call = self.continuous_learning.model_versioning.register_model.call_args
            assert register_call is not None
            
            # Check that the training data passed to register_model has only new samples
            training_data_arg = register_call[1]['training_data']
            assert len(training_data_arg.features) == len(new_features)
            assert training_data_arg.metadata['fallback_training'] == True
            
        finally:
            shutil.rmtree(temp_dir)

class TestAdvancedMLVolcanicInfrastructure:
    """Test main Advanced ML Volcanic Infrastructure"""
    
    def setup_method(self):
        """Setup test environment"""
        with patch('modules.advanced_ml_volcanic_infrastructure.ModelVersioningSystem'), \
             patch('modules.advanced_ml_volcanic_infrastructure.RealTimeInferenceEngine'), \
             patch('modules.advanced_ml_volcanic_infrastructure.HyperparameterOptimizer'), \
             patch('modules.advanced_ml_volcanic_infrastructure.ExplainableAI'), \
             patch('modules.advanced_ml_volcanic_infrastructure.ContinuousLearningFramework'):
            
            self.infrastructure = AdvancedMLVolcanicInfrastructure()
    
    def test_infrastructure_initialization(self):
        """Test infrastructure initialization"""
        assert hasattr(self.infrastructure, 'model_versioning')
        assert hasattr(self.infrastructure, 'inference_engine')
        assert hasattr(self.infrastructure, 'hyperparameter_optimizer')
        assert hasattr(self.infrastructure, 'explainable_ai')
        assert hasattr(self.infrastructure, 'continuous_learning')

class TestDataStructures:
    """Test data structure classes"""
    
    def test_ml_model_config_creation(self):
        """Test MLModelConfig dataclass"""
        config = MLModelConfig(
            model_type="physics_informed_nn",
            architecture={"hidden_dims": [64, 32]},
            training_params={"epochs": 100, "lr": 0.001},
            optimization_params={"optimizer": "adam"}
        )
        
        assert config.model_type == "physics_informed_nn"
        assert config.architecture["hidden_dims"] == [64, 32]
        assert config.training_params["epochs"] == 100
    
    def test_training_result_creation(self):
        """Test TrainingResult dataclass"""
        result = TrainingResult(
            model_id="test_model_123",
            model_path="models/test_model_123.pth",
            training_metrics={"loss": 0.1},
            validation_metrics={"loss": 0.15},
            test_metrics={"loss": 0.12},
            training_time=120.5,
            model_size=1024000,
            hyperparameters={"lr": 0.001}
        )
        
        assert result.model_id == "test_model_123"
        assert result.training_time == 120.5
        assert result.model_size == 1024000
    
    def test_inference_result_creation(self):
        """Test InferenceResult dataclass"""
        predictions = np.array([0.1, 0.2, 0.3])
        confidence_scores = np.array([0.9, 0.8, 0.7])
        uncertainty_bounds = np.array([[0.05, 0.15], [0.15, 0.25], [0.25, 0.35]])
        
        result = InferenceResult(
            predictions=predictions,
            confidence_scores=confidence_scores,
            uncertainty_bounds=uncertainty_bounds,
            inference_time=0.05,
            model_version="v1.0"
        )
        
        assert np.array_equal(result.predictions, predictions)
        assert np.array_equal(result.confidence_scores, confidence_scores)
        assert result.inference_time == 0.05
        assert result.model_version == "v1.0"

# Integration tests
class TestIntegration:
    """Integration tests for the complete ML infrastructure"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_ml_workflow(self):
        """Test complete end-to-end ML workflow"""
        
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        features = np.random.rand(n_samples, n_features)
        targets = (np.sum(features[:, :3], axis=1) + 
                  np.random.normal(0, 0.1, n_samples)).reshape(-1, 1)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        train_data = VolcanicDataset(
            features=features[:150],
            targets=targets[:150],
            feature_names=feature_names,
            target_names=['target'],
            metadata={'type': 'synthetic'}
        )
        
        val_data = VolcanicDataset(
            features=features[150:],
            targets=targets[150:],
            feature_names=feature_names,
            target_names=['target'],
            metadata={'type': 'synthetic'}
        )
        
        # Test ensemble training (simpler than PINN for testing)
        ensemble = EnsembleVolcanicPredictor(['random_forest'])
        
        # Train ensemble
        results = await ensemble.train_ensemble(
            train_data.features, train_data.targets.ravel(),
            val_data.features, val_data.targets.ravel()
        )
        
        assert 'ensemble' in results
        assert results['ensemble']['validation_mse'] >= 0
        
        # Test predictions
        predictions = await ensemble.predict(val_data.features)
        assert predictions.shape == (len(val_data.features),)
        
        # Test feature importance
        importance = ensemble.get_feature_importance()
        assert len(importance) > 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])