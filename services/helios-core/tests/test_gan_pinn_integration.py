"""
Tests for GAN-PINN Integration

This module contains tests for the integration between Generative Adversarial Networks
and Physics-Informed Neural Networks for probabilistic geological modeling.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile

from core.gan_pinn_integration import (
    GANPINNConfig, ProbabilisticPINNPredictor, AdversarialPINNTrainer,
    EnsemblePredictionSystem, GANPINNIntegrationEngine
)
from core.gan_engine import GANConfig, GeologicalGAN, GANEnsemble, GANEngine
from core.pinn_engine import PINNEngine
from models.ai_models import UncertaintyQuantification, UncertaintyMethod


class TestGANPINNConfig:
    """Test GAN-PINN configuration"""
    
    def test_valid_config(self):
        """Test valid GAN-PINN configuration"""
        config = GANPINNConfig(
            integration_id="test_integration",
            gan_model_id="test_gan",
            pinn_model_id="test_pinn"
        )
        
        assert config.integration_id == "test_integration"
        assert config.gan_model_id == "test_gan"
        assert config.pinn_model_id == "test_pinn"
        assert config.coupling_method == "ensemble_sampling"
        assert config.num_ensemble_samples == 100
        assert config.confidence_level == 0.95
    
    def test_empty_integration_id(self):
        """Test empty integration ID"""
        with pytest.raises(ValueError, match="Integration ID cannot be empty"):
            GANPINNConfig(
                integration_id="",
                gan_model_id="test_gan",
                pinn_model_id="test_pinn"
            )
    
    def test_empty_gan_model_id(self):
        """Test empty GAN model ID"""
        with pytest.raises(ValueError, match="GAN model ID cannot be empty"):
            GANPINNConfig(
                integration_id="test_integration",
                gan_model_id="",
                pinn_model_id="test_pinn"
            )
    
    def test_empty_pinn_model_id(self):
        """Test empty PINN model ID"""
        with pytest.raises(ValueError, match="PINN model ID cannot be empty"):
            GANPINNConfig(
                integration_id="test_integration",
                gan_model_id="test_gan",
                pinn_model_id=""
            )
    
    def test_invalid_confidence_level(self):
        """Test invalid confidence level"""
        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            GANPINNConfig(
                integration_id="test_integration",
                gan_model_id="test_gan",
                pinn_model_id="test_pinn",
                confidence_level=1.5
            )


class TestProbabilisticPINNPredictor:
    """Test probabilistic PINN predictor"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cpu')
    
    @pytest.fixture
    def mock_pinn_engine(self):
        mock_engine = Mock(spec=PINNEngine)
        mock_engine.predict.return_value = torch.randn(16, 5)  # Mock prediction
        return mock_engine
    
    @pytest.fixture
    def mock_gan_engine(self):
        mock_engine = Mock(spec=GANEngine)
        
        # Mock ensemble
        mock_ensemble = Mock(spec=GANEnsemble)
        mock_scenarios = [torch.randn(10, 32) for _ in range(3)]  # 3 GAN models, 10 scenarios each
        mock_ensemble.generate_ensemble_scenarios.return_value = mock_scenarios
        
        mock_engine.get_ensemble.return_value = mock_ensemble
        return mock_engine
    
    @pytest.fixture
    def predictor(self, mock_pinn_engine, mock_gan_engine, device):
        return ProbabilisticPINNPredictor(mock_pinn_engine, mock_gan_engine, device)
    
    def test_predictor_initialization(self, predictor, mock_pinn_engine, mock_gan_engine):
        """Test predictor initialization"""
        assert predictor.pinn_engine == mock_pinn_engine
        assert predictor.gan_engine == mock_gan_engine
        assert predictor.device == torch.device('cpu')
    
    def test_predict_with_uncertainty(self, predictor):
        """Test probabilistic prediction with uncertainty"""
        inputs = torch.randn(16, 10)  # Batch of 16 samples with 10 features
        
        result = predictor.predict_with_uncertainty(
            pinn_model_id="test_pinn",
            gan_ensemble_id="test_ensemble",
            inputs=inputs,
            num_scenarios=30
        )
        
        assert "mean_prediction" in result
        assert "std_prediction" in result
        assert "confidence_intervals" in result
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        assert "total_uncertainty" in result
        assert result["num_scenarios"] == 30
        assert result["num_gan_models"] == 3
    
    def test_combine_input_with_scenario(self, predictor):
        """Test combining input with scenario"""
        inputs = torch.randn(16, 10)
        scenario = torch.randn(32)
        
        combined = predictor._combine_input_with_scenario(inputs, scenario)
        
        assert combined.shape == (16, 42)  # 10 + 32
        assert torch.allclose(combined[:, :10], inputs)
        assert torch.allclose(combined[:, 10:], scenario.unsqueeze(0).expand(16, -1))
    
    def test_apply_physics_constraints(self, predictor):
        """Test physics constraints application"""
        predictions = torch.tensor([
            [200.0, -100.0, 0.3, 0.4, 0.3],  # Low temp, negative pressure
            [1500.0, 1e6, 0.2, 0.5, 0.3]     # Normal values
        ])
        
        constraints = ["temperature_positive", "pressure_positive", "mass_conservation"]
        constrained = predictor._apply_physics_constraints(predictions, constraints)
        
        # Check temperature constraint
        assert constrained[0, 0] >= 273.15  # Above absolute zero
        
        # Check pressure constraint
        assert constrained[0, 1] >= 0.0  # Non-negative pressure
        
        # Check mass conservation (sum of components 2:5 should be normalized)
        mass_sum = torch.sum(constrained[:, 2:5], dim=1)
        assert torch.allclose(mass_sum, torch.ones(2), atol=1e-6)
    
    def test_calculate_confidence_intervals(self, predictor):
        """Test confidence interval calculation"""
        # Create mock predictions with known distribution
        predictions = torch.randn(100, 16, 5)  # 100 scenarios, 16 batch, 5 outputs
        
        intervals = predictor._calculate_confidence_intervals(predictions, confidence_level=0.95)
        
        assert len(intervals) == 5  # One interval per output dimension
        for i in range(5):
            assert f"output_{i}" in intervals
            lower, upper = intervals[f"output_{i}"]
            assert lower < upper  # Lower bound should be less than upper bound
    
    def test_decompose_uncertainty(self, predictor):
        """Test uncertainty decomposition"""
        # Create mock scenario list (3 models, 10 scenarios each)
        scenario_list = [torch.randn(10, 32) for _ in range(3)]
        
        # Create mock predictions
        all_predictions = torch.randn(30, 16, 5)  # 30 total scenarios, 16 batch, 5 outputs
        
        epistemic, aleatoric = predictor._decompose_uncertainty(all_predictions, scenario_list)
        
        assert epistemic.shape == (16, 5)
        assert aleatoric.shape == (16, 5)
        assert np.all(epistemic >= 0)  # Uncertainty should be non-negative
        assert np.all(aleatoric >= 0)


class TestAdversarialPINNTrainer:
    """Test adversarial PINN trainer"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cpu')
    
    @pytest.fixture
    def mock_pinn_engine(self):
        mock_engine = Mock(spec=PINNEngine)
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.model = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 5)
        )
        mock_trainer._compute_physics_loss.return_value = torch.tensor(0.1)
        
        mock_engine.get_trainer.return_value = mock_trainer
        return mock_engine
    
    @pytest.fixture
    def mock_gan_engine(self):
        mock_engine = Mock(spec=GANEngine)
        
        # Mock GAN
        mock_gan = Mock(spec=GeologicalGAN)
        mock_discriminator = Mock()
        mock_discriminator.return_value = torch.rand(16, 1)  # Random plausibility scores
        mock_gan.discriminator = mock_discriminator
        
        mock_engine.get_gan.return_value = mock_gan
        return mock_engine
    
    @pytest.fixture
    def trainer(self, mock_pinn_engine, mock_gan_engine, device):
        return AdversarialPINNTrainer(mock_pinn_engine, mock_gan_engine, device)
    
    def test_trainer_initialization(self, trainer, mock_pinn_engine, mock_gan_engine):
        """Test trainer initialization"""
        assert trainer.pinn_engine == mock_pinn_engine
        assert trainer.gan_engine == mock_gan_engine
        assert trainer.device == torch.device('cpu')
    
    def test_compute_adversarial_loss(self, trainer):
        """Test adversarial loss computation"""
        # Mock PINN predictions
        pinn_predictions = torch.randn(16, 5)
        
        # Mock GAN
        mock_gan = Mock()
        mock_discriminator = Mock()
        mock_discriminator.return_value = torch.rand(16, 1)
        mock_gan.discriminator = mock_discriminator
        
        loss = trainer._compute_adversarial_loss(pinn_predictions, mock_gan)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative
        mock_discriminator.assert_called_once_with(pinn_predictions)
    
    @patch('torch.utils.data.DataLoader')
    def test_train_pinn_with_gan_feedback(self, mock_dataloader, trainer):
        """Test PINN training with GAN feedback"""
        # Create mock training data
        mock_batch = (torch.randn(16, 10), torch.randn(16, 5))
        mock_dataloader.__iter__.return_value = [mock_batch] * 5  # 5 batches
        
        history = trainer.train_pinn_with_gan_feedback(
            pinn_model_id="test_pinn",
            gan_model_id="test_gan",
            training_data=mock_dataloader,
            num_epochs=3,
            adversarial_weight=0.1
        )
        
        assert "pinn_loss" in history
        assert "physics_loss" in history
        assert "adversarial_loss" in history
        assert "total_loss" in history
        assert "epoch" in history
        assert len(history["epoch"]) == 3


class TestEnsemblePredictionSystem:
    """Test ensemble prediction system"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cpu')
    
    @pytest.fixture
    def mock_pinn_engine(self):
        mock_engine = Mock(spec=PINNEngine)
        mock_engine.models = {"test_pinn": Mock()}
        mock_engine.predict.return_value = torch.randn(16, 5)
        return mock_engine
    
    @pytest.fixture
    def mock_gan_engine(self):
        mock_engine = Mock(spec=GANEngine)
        mock_engine.gans = {}
        mock_engine.ensembles = {"test_ensemble": Mock()}
        
        # Mock ensemble
        mock_ensemble = Mock(spec=GANEnsemble)
        mock_scenarios = [torch.randn(10, 32) for _ in range(3)]
        mock_ensemble.generate_ensemble_scenarios.return_value = mock_scenarios
        mock_ensemble.gans = [Mock() for _ in range(3)]
        for gan in mock_ensemble.gans:
            gan.assess_plausibility.return_value = torch.rand(16, 1)
        
        mock_engine.get_ensemble.return_value = mock_ensemble
        return mock_engine
    
    @pytest.fixture
    def system(self, mock_pinn_engine, mock_gan_engine, device):
        return EnsemblePredictionSystem(mock_pinn_engine, mock_gan_engine, device)
    
    @pytest.fixture
    def config(self):
        return GANPINNConfig(
            integration_id="test_integration",
            gan_model_id="test_ensemble",
            pinn_model_id="test_pinn"
        )
    
    def test_system_initialization(self, system, mock_pinn_engine, mock_gan_engine):
        """Test system initialization"""
        assert system.pinn_engine == mock_pinn_engine
        assert system.gan_engine == mock_gan_engine
        assert system.device == torch.device('cpu')
        assert len(system.ensemble_configs) == 0
    
    def test_create_ensemble_predictor(self, system, config):
        """Test ensemble predictor creation"""
        integration_id = system.create_ensemble_predictor(config)
        
        assert integration_id == config.integration_id
        assert config.integration_id in system.ensemble_configs
        assert system.ensemble_configs[config.integration_id] == config
    
    def test_create_duplicate_ensemble_predictor(self, system, config):
        """Test creating duplicate ensemble predictor"""
        system.create_ensemble_predictor(config)
        
        with pytest.raises(ValueError, match="already exists"):
            system.create_ensemble_predictor(config)
    
    def test_create_ensemble_predictor_invalid_gan(self, system):
        """Test creating ensemble predictor with invalid GAN model"""
        config = GANPINNConfig(
            integration_id="test_integration",
            gan_model_id="nonexistent_gan",
            pinn_model_id="test_pinn"
        )
        
        with pytest.raises(ValueError, match="GAN model nonexistent_gan not found"):
            system.create_ensemble_predictor(config)
    
    def test_create_ensemble_predictor_invalid_pinn(self, system):
        """Test creating ensemble predictor with invalid PINN model"""
        config = GANPINNConfig(
            integration_id="test_integration",
            gan_model_id="test_ensemble",
            pinn_model_id="nonexistent_pinn"
        )
        
        with pytest.raises(ValueError, match="PINN model nonexistent_pinn not found"):
            system.create_ensemble_predictor(config)
    
    def test_predict_ensemble(self, system, config):
        """Test ensemble prediction"""
        system.create_ensemble_predictor(config)
        
        inputs = torch.randn(16, 10)
        geological_context = {
            'location': {'latitude': 45.0, 'longitude': -120.0, 'depth': 5000.0},
            'time': 1234567890,
            'setting': 'volcanic'
        }
        
        result = system.predict_ensemble(
            config.integration_id,
            inputs,
            geological_context
        )
        
        assert "integration_id" in result
        assert "mean_prediction" in result
        assert "std_prediction" in result
        assert "confidence_intervals" in result
        assert "mean_plausibility" in result
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        assert "total_uncertainty" in result
        assert result["integration_id"] == config.integration_id
    
    def test_extract_context_features(self, system):
        """Test geological context feature extraction"""
        geological_context = {
            'location': {'latitude': 45.0, 'longitude': -120.0, 'depth': 5000.0},
            'time': 1234567890,
            'setting': 'volcanic'
        }
        
        features = system._extract_context_features(geological_context, batch_size=16)
        
        assert features.shape[0] == 16  # Batch size
        assert features.shape[1] > 0  # Should have some features
        
        # Check location normalization
        assert -1 <= features[0, 0] <= 1  # Normalized latitude
        assert -1 <= features[0, 1] <= 1  # Normalized longitude
        assert 0 <= features[0, 2] <= 1   # Normalized depth
    
    def test_extract_context_features_empty(self, system):
        """Test context feature extraction with empty context"""
        features = system._extract_context_features({}, batch_size=16)
        
        assert features.shape == (16, 0)  # Empty features
    
    def test_calculate_epistemic_uncertainty(self, system):
        """Test epistemic uncertainty calculation"""
        # Create mock ensemble predictions (3 models, 10 scenarios each)
        ensemble_predictions = [
            torch.randn(10, 16, 5) for _ in range(3)
        ]
        
        epistemic = system._calculate_epistemic_uncertainty(ensemble_predictions)
        
        assert epistemic.shape == (16, 5)
        assert np.all(epistemic >= 0)  # Should be non-negative
    
    def test_calculate_epistemic_uncertainty_single_model(self, system):
        """Test epistemic uncertainty with single model"""
        ensemble_predictions = [torch.randn(10, 16, 5)]
        
        epistemic = system._calculate_epistemic_uncertainty(ensemble_predictions)
        
        assert epistemic.shape == (16, 5)
        assert np.allclose(epistemic, 0)  # Should be zero for single model
    
    def test_calculate_aleatoric_uncertainty(self, system):
        """Test aleatoric uncertainty calculation"""
        ensemble_predictions = [
            torch.randn(10, 16, 5) for _ in range(3)
        ]
        
        aleatoric = system._calculate_aleatoric_uncertainty(ensemble_predictions)
        
        assert aleatoric.shape == (16, 5)
        assert np.all(aleatoric >= 0)  # Should be non-negative
    
    def test_get_ensemble_info(self, system, config):
        """Test getting ensemble information"""
        system.create_ensemble_predictor(config)
        
        info = system.get_ensemble_info(config.integration_id)
        
        assert info["integration_id"] == config.integration_id
        assert info["gan_model_id"] == config.gan_model_id
        assert info["pinn_model_id"] == config.pinn_model_id
        assert info["coupling_method"] == config.coupling_method
    
    def test_get_ensemble_info_nonexistent(self, system):
        """Test getting info for nonexistent ensemble"""
        with pytest.raises(ValueError, match="not found"):
            system.get_ensemble_info("nonexistent")
    
    def test_list_ensemble_predictors(self, system, config):
        """Test listing ensemble predictors"""
        assert system.list_ensemble_predictors() == []
        
        system.create_ensemble_predictor(config)
        assert system.list_ensemble_predictors() == [config.integration_id]


class TestGANPINNIntegrationEngine:
    """Test GAN-PINN integration engine"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cpu')
    
    @pytest.fixture
    def mock_pinn_engine(self):
        mock_engine = Mock(spec=PINNEngine)
        mock_engine.models = {"test_pinn": Mock()}
        mock_engine.predict.return_value = torch.randn(16, 5)
        return mock_engine
    
    @pytest.fixture
    def mock_gan_engine(self):
        mock_engine = Mock(spec=GANEngine)
        mock_engine.ensembles = {"test_ensemble": Mock()}
        return mock_engine
    
    @pytest.fixture
    def engine(self, mock_pinn_engine, mock_gan_engine, device):
        return GANPINNIntegrationEngine(mock_pinn_engine, mock_gan_engine, device)
    
    @pytest.fixture
    def config(self):
        return GANPINNConfig(
            integration_id="test_integration",
            gan_model_id="test_ensemble",
            pinn_model_id="test_pinn"
        )
    
    def test_engine_initialization(self, engine, mock_pinn_engine, mock_gan_engine):
        """Test engine initialization"""
        assert engine.pinn_engine == mock_pinn_engine
        assert engine.gan_engine == mock_gan_engine
        assert engine.device == torch.device('cpu')
        assert hasattr(engine, 'probabilistic_predictor')
        assert hasattr(engine, 'adversarial_trainer')
        assert hasattr(engine, 'ensemble_system')
    
    def test_create_probabilistic_model(self, engine, config):
        """Test probabilistic model creation"""
        integration_id = engine.create_probabilistic_model(config)
        
        assert integration_id == config.integration_id
    
    def test_predict_probabilistic(self, engine, config):
        """Test probabilistic prediction"""
        # Create model first
        engine.create_probabilistic_model(config)
        
        # Mock the ensemble system prediction
        mock_result = {
            "mean_prediction": [[1.0, 2.0, 3.0]],
            "std_prediction": [[0.1, 0.2, 0.3]],
            "confidence_intervals": {"output_0": (0.8, 1.2)},
            "epistemic_uncertainty": [[0.05, 0.1, 0.15]],
            "aleatoric_uncertainty": [[0.05, 0.1, 0.15]],
            "total_uncertainty": [[0.1, 0.2, 0.3]]
        }
        engine.ensemble_system.predict_ensemble = Mock(return_value=mock_result)
        
        inputs = np.random.randn(16, 10)
        result = engine.predict_probabilistic(config.integration_id, inputs)
        
        assert "mean_prediction" in result
        assert "std_prediction" in result
        assert "confidence_intervals" in result
    
    def test_get_integration_info(self, engine, config):
        """Test getting integration information"""
        engine.create_probabilistic_model(config)
        
        info = engine.get_integration_info(config.integration_id)
        
        assert "integration_id" in info
        assert info["integration_id"] == config.integration_id
    
    def test_list_integrations(self, engine, config):
        """Test listing integrations"""
        assert engine.list_integrations() == []
        
        engine.create_probabilistic_model(config)
        assert engine.list_integrations() == [config.integration_id]


@pytest.mark.integration
class TestGANPINNIntegrationIntegration:
    """Integration tests for GAN-PINN system"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cpu')
    
    def test_end_to_end_integration(self, device):
        """Test end-to-end GAN-PINN integration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create engines
            gan_engine = GANEngine(device, temp_dir)
            pinn_engine = Mock(spec=PINNEngine)
            pinn_engine.models = {"test_pinn": Mock()}
            pinn_engine.predict.return_value = torch.randn(16, 5)
            
            # Create GAN ensemble
            ensemble_configs = [
                GANConfig(
                    model_id=f"integration_gan_{i}",
                    latent_dim=50,
                    output_dim=20,
                    generator_layers=[64, 128, 64],
                    discriminator_layers=[64, 32, 1]
                )
                for i in range(2)
            ]
            
            gan_ensemble = gan_engine.create_ensemble("test_ensemble", ensemble_configs)
            
            # Create integration engine
            integration_engine = GANPINNIntegrationEngine(pinn_engine, gan_engine, device)
            
            # Create probabilistic model
            config = GANPINNConfig(
                integration_id="test_integration",
                gan_model_id="test_ensemble",
                pinn_model_id="test_pinn"
            )
            
            integration_id = integration_engine.create_probabilistic_model(config)
            assert integration_id == config.integration_id
            
            # Test prediction
            inputs = np.random.randn(16, 10)
            result = integration_engine.predict_probabilistic(integration_id, inputs)
            
            assert "mean_prediction" in result
            assert "std_prediction" in result
            assert "confidence_intervals" in result
            assert "epistemic_uncertainty" in result
            assert "aleatoric_uncertainty" in result
            
            # Test info retrieval
            info = integration_engine.get_integration_info(integration_id)
            assert info["integration_id"] == integration_id
            
            # Test listing
            integrations = integration_engine.list_integrations()
            assert integration_id in integrations