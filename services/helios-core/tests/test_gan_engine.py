"""
Tests for GAN Engine and Geological Scenario Generation

This module contains comprehensive tests for the GAN engine, including
generator and discriminator networks, training procedures, and uncertainty
quantification methods.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from core.gan_engine import (
    GANConfig,
    GANEngine,
    GANEnsemble,
    GeologicalDiscriminator,
    GeologicalGAN,
    GeologicalGenerator,
    SpectralNorm,
)
from core.gan_training import (
    GANTrainingManager,
    GeologicalConstraintLoss,
    TrainingConfig,
)
from core.uncertainty_quantification import (
    GeologicalUncertaintyQuantifier,
    MonteCarloDropout,
    UncertaintyConfig,
)
from models.ai_models import UncertaintyMethod, UncertaintyQuantification


class TestGANConfig:
    """Test GAN configuration"""

    def test_valid_config(self):
        """Test valid GAN configuration"""
        config = GANConfig(
            model_id="test_gan",
            latent_dim=100,
            output_dim=64,
            generator_layers=[256, 512, 256],
            discriminator_layers=[256, 128, 1],
        )

        assert config.model_id == "test_gan"
        assert config.latent_dim == 100
        assert config.output_dim == 64
        assert config.learning_rate == 0.0002
        assert config.spectral_normalization is True

    def test_invalid_latent_dim(self):
        """Test invalid latent dimension"""
        with pytest.raises(ValueError, match="Latent dimension must be positive"):
            GANConfig(
                model_id="test_gan",
                latent_dim=0,
                output_dim=64,
                generator_layers=[256],
                discriminator_layers=[256, 1],
            )

    def test_invalid_output_dim(self):
        """Test invalid output dimension"""
        with pytest.raises(ValueError, match="Output dimension must be positive"):
            GANConfig(
                model_id="test_gan",
                latent_dim=100,
                output_dim=-1,
                generator_layers=[256],
                discriminator_layers=[256, 1],
            )

    def test_empty_generator_layers(self):
        """Test empty generator layers"""
        with pytest.raises(
            ValueError, match="Generator must have at least one hidden layer"
        ):
            GANConfig(
                model_id="test_gan",
                latent_dim=100,
                output_dim=64,
                generator_layers=[],
                discriminator_layers=[256, 1],
            )


class TestSpectralNorm:
    """Test spectral normalization"""

    def test_spectral_norm_initialization(self):
        """Test spectral normalization initialization"""
        linear = torch.nn.Linear(10, 5)
        spectral_linear = SpectralNorm(linear)

        # Check that spectral norm parameters are created
        assert hasattr(spectral_linear.module, "weight_u")
        assert hasattr(spectral_linear.module, "weight_v")
        assert hasattr(spectral_linear.module, "weight_bar")

    def test_spectral_norm_forward(self):
        """Test spectral normalization forward pass"""
        linear = torch.nn.Linear(10, 5)
        spectral_linear = SpectralNorm(linear)

        input_tensor = torch.randn(32, 10)
        output = spectral_linear(input_tensor)

        assert output.shape == (32, 5)


class TestGeologicalGenerator:
    """Test geological generator network"""

    @pytest.fixture
    def config(self):
        return GANConfig(
            model_id="test_generator",
            latent_dim=100,
            output_dim=64,
            generator_layers=[256, 512, 256],
            discriminator_layers=[256, 128, 1],
        )

    @pytest.fixture
    def generator(self, config):
        return GeologicalGenerator(config)

    def test_generator_initialization(self, generator, config):
        """Test generator initialization"""
        assert generator.latent_dim == config.latent_dim
        assert generator.output_dim == config.output_dim
        assert isinstance(generator.network, torch.nn.Sequential)

    def test_generator_forward(self, generator):
        """Test generator forward pass"""
        batch_size = 32
        noise = torch.randn(batch_size, generator.latent_dim)

        output = generator(noise)

        assert output.shape == (batch_size, generator.output_dim)
        assert torch.all(output >= -1) and torch.all(output <= 1)  # Tanh output

    def test_generate_scenarios(self, generator):
        """Test scenario generation"""
        num_scenarios = 50
        device = torch.device("cpu")

        scenarios = generator.generate_scenarios(num_scenarios, device)

        assert scenarios.shape == (num_scenarios, generator.output_dim)
        assert torch.all(scenarios >= -1) and torch.all(scenarios <= 1)

    def test_generate_scenarios_uniform_noise(self, config):
        """Test scenario generation with uniform noise"""
        config.noise_type = "uniform"
        generator = GeologicalGenerator(config)

        num_scenarios = 50
        device = torch.device("cpu")

        scenarios = generator.generate_scenarios(num_scenarios, device)

        assert scenarios.shape == (num_scenarios, generator.output_dim)

    def test_invalid_noise_type(self, config):
        """Test invalid noise type"""
        config.noise_type = "invalid"
        generator = GeologicalGenerator(config)

        with pytest.raises(ValueError, match="Unknown noise type"):
            generator.generate_scenarios(10, torch.device("cpu"))


class TestGeologicalDiscriminator:
    """Test geological discriminator network"""

    @pytest.fixture
    def config(self):
        return GANConfig(
            model_id="test_discriminator",
            latent_dim=100,
            output_dim=64,
            generator_layers=[256, 512, 256],
            discriminator_layers=[256, 128, 1],
        )

    @pytest.fixture
    def discriminator(self, config):
        return GeologicalDiscriminator(config)

    def test_discriminator_initialization(self, discriminator, config):
        """Test discriminator initialization"""
        assert discriminator.input_dim == config.output_dim
        assert isinstance(discriminator.network, torch.nn.Sequential)

    def test_discriminator_forward(self, discriminator, config):
        """Test discriminator forward pass"""
        batch_size = 32
        scenarios = torch.randn(batch_size, config.output_dim)

        output = discriminator(scenarios)

        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output

    def test_discriminator_with_spectral_norm(self, config):
        """Test discriminator with spectral normalization"""
        config.spectral_normalization = True
        discriminator = GeologicalDiscriminator(config)

        batch_size = 32
        scenarios = torch.randn(batch_size, config.output_dim)

        output = discriminator(scenarios)

        assert output.shape == (batch_size, 1)


class TestGeologicalGAN:
    """Test complete GAN system"""

    @pytest.fixture
    def config(self):
        return GANConfig(
            model_id="test_gan",
            latent_dim=100,
            output_dim=64,
            generator_layers=[256, 512, 256],
            discriminator_layers=[256, 128, 1],
        )

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def gan(self, config, device):
        return GeologicalGAN(config, device)

    def test_gan_initialization(self, gan, config):
        """Test GAN initialization"""
        assert gan.config == config
        assert isinstance(gan.generator, GeologicalGenerator)
        assert isinstance(gan.discriminator, GeologicalDiscriminator)
        assert gan.generator.latent_dim == config.latent_dim
        assert gan.discriminator.input_dim == config.output_dim

    def test_gradient_penalty(self, gan):
        """Test gradient penalty calculation"""
        batch_size = 16
        real_samples = torch.randn(batch_size, gan.config.output_dim)
        fake_samples = torch.randn(batch_size, gan.config.output_dim)

        penalty = gan.gradient_penalty(real_samples, fake_samples)

        assert isinstance(penalty, torch.Tensor)
        assert penalty.item() >= 0

    def test_train_step(self, gan):
        """Test single training step"""
        batch_size = 16
        real_data = torch.randn(batch_size, gan.config.output_dim)

        metrics = gan.train_step(real_data)

        assert "g_loss" in metrics
        assert "d_loss" in metrics
        assert "gradient_penalty" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_generate_scenarios(self, gan):
        """Test scenario generation"""
        num_scenarios = 25
        scenarios = gan.generate_scenarios(num_scenarios)

        assert scenarios.shape == (num_scenarios, gan.config.output_dim)

    def test_assess_plausibility(self, gan):
        """Test plausibility assessment"""
        num_scenarios = 25
        scenarios = torch.randn(num_scenarios, gan.config.output_dim)

        plausibility = gan.assess_plausibility(scenarios)

        assert plausibility.shape == (num_scenarios, 1)
        assert torch.all(plausibility >= 0) and torch.all(plausibility <= 1)

    def test_save_and_load_model(self, gan):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_gan.pth"

            # Save model
            gan.save_model(filepath)
            assert filepath.exists()

            # Create new GAN and load model
            new_gan = GeologicalGAN(gan.config, gan.device)
            new_gan.load_model(filepath)

            # Check that models are equivalent
            assert new_gan.config.model_id == gan.config.model_id
            assert new_gan.metadata.model_id == gan.metadata.model_id


class TestGANEnsemble:
    """Test GAN ensemble for uncertainty quantification"""

    @pytest.fixture
    def configs(self):
        return [
            GANConfig(
                model_id=f"ensemble_gan_{i}",
                latent_dim=100,
                output_dim=32,
                generator_layers=[128, 256, 128],
                discriminator_layers=[128, 64, 1],
                learning_rate=0.0002 + i * 0.0001,  # Slightly different learning rates
            )
            for i in range(3)
        ]

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def ensemble(self, configs, device):
        return GANEnsemble(configs, device)

    def test_ensemble_initialization(self, ensemble, configs):
        """Test ensemble initialization"""
        assert len(ensemble.gans) == len(configs)
        assert all(isinstance(gan, GeologicalGAN) for gan in ensemble.gans)

    def test_generate_ensemble_scenarios(self, ensemble):
        """Test ensemble scenario generation"""
        num_scenarios = 20
        scenarios_list = ensemble.generate_ensemble_scenarios(num_scenarios)

        assert len(scenarios_list) == len(ensemble.gans)
        assert all(
            scenarios.shape == (num_scenarios, 32) for scenarios in scenarios_list
        )

    def test_quantify_uncertainty(self, ensemble):
        """Test uncertainty quantification"""
        num_scenarios = 50
        uncertainty_result = ensemble.quantify_uncertainty(num_scenarios)

        assert isinstance(uncertainty_result, UncertaintyQuantification)
        assert uncertainty_result.method == UncertaintyMethod.ENSEMBLE_VARIANCE
        assert uncertainty_result.epistemic_uncertainty.shape[0] == num_scenarios
        assert uncertainty_result.aleatoric_uncertainty.shape[0] == num_scenarios
        assert uncertainty_result.total_uncertainty.shape[0] == num_scenarios
        assert len(uncertainty_result.confidence_intervals) > 0


class TestGANEngine:
    """Test GAN engine management"""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def engine(self, device):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield GANEngine(device, temp_dir)

    @pytest.fixture
    def config(self):
        return GANConfig(
            model_id="engine_test_gan",
            latent_dim=50,
            output_dim=32,
            generator_layers=[128, 256, 128],
            discriminator_layers=[128, 64, 1],
        )

    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert len(engine.gans) == 0
        assert len(engine.ensembles) == 0
        assert engine.model_cache_dir.exists()

    def test_create_gan(self, engine, config):
        """Test GAN creation"""
        gan = engine.create_gan(config)

        assert isinstance(gan, GeologicalGAN)
        assert config.model_id in engine.gans
        assert engine.get_gan(config.model_id) == gan

    def test_create_duplicate_gan(self, engine, config):
        """Test creating duplicate GAN"""
        engine.create_gan(config)

        with pytest.raises(ValueError, match="already exists"):
            engine.create_gan(config)

    def test_create_ensemble(self, engine):
        """Test ensemble creation"""
        configs = [
            GANConfig(
                model_id=f"ensemble_gan_{i}",
                latent_dim=50,
                output_dim=32,
                generator_layers=[128, 256, 128],
                discriminator_layers=[128, 64, 1],
            )
            for i in range(2)
        ]

        ensemble = engine.create_ensemble("test_ensemble", configs)

        assert isinstance(ensemble, GANEnsemble)
        assert "test_ensemble" in engine.ensembles
        assert engine.get_ensemble("test_ensemble") == ensemble

    def test_get_nonexistent_gan(self, engine):
        """Test getting non-existent GAN"""
        with pytest.raises(ValueError, match="not found"):
            engine.get_gan("nonexistent")

    def test_list_gans(self, engine, config):
        """Test listing GANs"""
        assert engine.list_gans() == []

        engine.create_gan(config)
        assert engine.list_gans() == [config.model_id]

    def test_save_and_load_gan(self, engine, config):
        """Test GAN saving and loading"""
        # Create and save GAN
        engine.create_gan(config)
        engine.save_gan(config.model_id)

        # Remove from engine
        del engine.gans[config.model_id]
        assert config.model_id not in engine.gans

        # Load GAN
        engine.load_gan(config.model_id)
        assert config.model_id in engine.gans

        loaded_gan = engine.get_gan(config.model_id)
        assert loaded_gan.config.model_id == config.model_id


class TestGeologicalConstraintLoss:
    """Test geological constraint loss"""

    @pytest.fixture
    def constraint_loss(self):
        return GeologicalConstraintLoss(constraint_weight=1.0)

    def test_constraint_loss_calculation(self, constraint_loss):
        """Test constraint loss calculation"""
        # Create scenarios with some constraint violations
        batch_size = 16
        output_dim = 10
        scenarios = torch.randn(batch_size, output_dim)

        # Set some extreme values to trigger constraints
        scenarios[:, 0] = 3000.0  # Extreme temperature
        scenarios[:, 1] = -1000.0  # Negative pressure
        scenarios[:, 2] = -5.0  # Negative volume

        loss = constraint_loss(scenarios)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0  # Should have positive loss due to violations

    def test_constraint_loss_valid_scenarios(self, constraint_loss):
        """Test constraint loss with valid scenarios"""
        batch_size = 16
        output_dim = 10

        # Create scenarios within valid ranges
        scenarios = torch.zeros(batch_size, output_dim)
        scenarios[:, 0] = 1000.0  # Valid temperature
        scenarios[:, 1] = 1e6  # Valid pressure
        scenarios[:, 2] = 1000.0  # Valid volume

        loss = constraint_loss(scenarios)

        assert isinstance(loss, torch.Tensor)
        # Loss should be lower for valid scenarios
        assert loss.item() >= 0


class TestMonteCarloDropout:
    """Test Monte Carlo Dropout uncertainty quantification"""

    @pytest.fixture
    def model(self):
        return torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 5),
        )

    @pytest.fixture
    def mc_dropout(self, model):
        return MonteCarloDropout(model, dropout_rate=0.1)

    def test_mc_dropout_initialization(self, mc_dropout, model):
        """Test Monte Carlo Dropout initialization"""
        assert mc_dropout.model == model
        assert mc_dropout.dropout_rate == 0.1

    def test_predict_with_uncertainty(self, mc_dropout):
        """Test prediction with uncertainty"""
        batch_size = 16
        input_dim = 10
        inputs = torch.randn(batch_size, input_dim)

        mean_pred, uncertainty = mc_dropout.predict_with_uncertainty(
            inputs, num_samples=50
        )

        assert mean_pred.shape == (batch_size, 5)
        assert uncertainty.shape == (batch_size, 5)
        assert torch.all(uncertainty >= 0)  # Uncertainty should be non-negative


class TestUncertaintyConfig:
    """Test uncertainty configuration"""

    def test_valid_config(self):
        """Test valid uncertainty configuration"""
        config = UncertaintyConfig(
            method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
            num_samples=100,
            confidence_level=0.95,
        )

        assert config.method == UncertaintyMethod.MONTE_CARLO_DROPOUT
        assert config.num_samples == 100
        assert config.confidence_level == 0.95

    def test_invalid_confidence_level(self):
        """Test invalid confidence level"""
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            UncertaintyConfig(
                method=UncertaintyMethod.MONTE_CARLO_DROPOUT, confidence_level=1.5
            )

    def test_invalid_num_samples(self):
        """Test invalid number of samples"""
        with pytest.raises(ValueError, match="Number of samples must be positive"):
            UncertaintyConfig(
                method=UncertaintyMethod.MONTE_CARLO_DROPOUT, num_samples=0
            )


class TestTrainingConfig:
    """Test training configuration"""

    def test_valid_config(self):
        """Test valid training configuration"""
        config = TrainingConfig(
            num_epochs=100,
            batch_size=32,
            learning_rate_g=0.0002,
            learning_rate_d=0.0002,
        )

        assert config.num_epochs == 100
        assert config.batch_size == 32
        assert config.use_progressive_training is True
        assert config.use_geological_constraints is True

    def test_invalid_epochs(self):
        """Test invalid number of epochs"""
        with pytest.raises(ValueError, match="Number of epochs must be positive"):
            TrainingConfig(num_epochs=0)

    def test_invalid_batch_size(self):
        """Test invalid batch size"""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            TrainingConfig(batch_size=-1)

    def test_invalid_learning_rate(self):
        """Test invalid learning rate"""
        with pytest.raises(
            ValueError, match="Generator learning rate must be between 0 and 1"
        ):
            TrainingConfig(learning_rate_g=1.5)


@pytest.mark.integration
class TestGANIntegration:
    """Integration tests for GAN system"""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def gan_config(self):
        return GANConfig(
            model_id="integration_test_gan",
            latent_dim=50,
            output_dim=20,
            generator_layers=[128, 256, 128],
            discriminator_layers=[128, 64, 1],
            batch_size=16,
        )

    @pytest.fixture
    def training_config(self):
        return TrainingConfig(
            num_epochs=5,  # Short training for testing
            batch_size=16,
            log_interval=2,
            save_interval=10,
            generate_samples_interval=3,
        )

    def test_end_to_end_training(self, device, gan_config, training_config):
        """Test end-to-end GAN training"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create training manager
            training_manager = GANTrainingManager(device)

            # Create trainer
            trainer = training_manager.create_trainer(
                "test_trainer", gan_config, training_config
            )

            # Create synthetic geological data
            num_samples = 100
            synthetic_data = []
            for _ in range(num_samples):
                sample = {
                    "temperature": np.random.uniform(500, 1500),
                    "pressure": np.random.uniform(1e5, 1e8),
                    "volume": np.random.uniform(1e3, 1e6),
                    "composition": {"SiO2": np.random.uniform(45, 75)},
                    "location": {
                        "latitude": np.random.uniform(-90, 90),
                        "longitude": np.random.uniform(-180, 180),
                        "depth": np.random.uniform(0, 50000),
                    },
                }
                synthetic_data.append(sample)

            # Create data loader
            dataloader = training_manager.create_geological_dataset(
                synthetic_data, batch_size=training_config.batch_size
            )

            # Train GAN
            history = training_manager.train_gan("test_trainer", dataloader)

            # Check training history
            assert "g_loss" in history
            assert "d_loss" in history
            assert len(history["g_loss"]) == training_config.num_epochs
            assert len(history["d_loss"]) == training_config.num_epochs

            # Test scenario generation
            gan = trainer.gan
            scenarios = gan.generate_scenarios(50)
            assert scenarios.shape == (50, gan_config.output_dim)

            # Test plausibility assessment
            plausibility = gan.assess_plausibility(scenarios)
            assert plausibility.shape == (50, 1)
            assert torch.all(plausibility >= 0) and torch.all(plausibility <= 1)

    def test_uncertainty_quantification_integration(self, device):
        """Test integration of uncertainty quantification with GANs"""
        # Create GAN engine
        with tempfile.TemporaryDirectory() as temp_dir:
            gan_engine = GANEngine(device, temp_dir)

            # Create ensemble configs
            ensemble_configs = [
                GANConfig(
                    model_id=f"uncertainty_gan_{i}",
                    latent_dim=50,
                    output_dim=20,
                    generator_layers=[64, 128, 64],
                    discriminator_layers=[64, 32, 1],
                )
                for i in range(3)
            ]

            # Create ensemble
            ensemble = gan_engine.create_ensemble(
                "uncertainty_ensemble", ensemble_configs
            )

            # Create uncertainty quantifier
            uncertainty_quantifier = GeologicalUncertaintyQuantifier(gan_engine, device)

            # Test uncertainty quantification
            config = UncertaintyConfig(
                method=UncertaintyMethod.ENSEMBLE_VARIANCE, num_samples=30
            )

            uncertainty_result = uncertainty_quantifier.quantify_uncertainty(
                "uncertainty_ensemble", torch.randn(10, 20), config
            )

            assert isinstance(uncertainty_result, UncertaintyQuantification)
            assert uncertainty_result.method == UncertaintyMethod.ENSEMBLE_VARIANCE
            assert uncertainty_result.epistemic_uncertainty.shape[0] == 30
            assert uncertainty_result.total_uncertainty.shape[0] == 30

            # Test geological scenario generation with uncertainty
            scenario_results = (
                uncertainty_quantifier.generate_geological_scenarios_with_uncertainty(
                    "uncertainty_ensemble", num_scenarios=25
                )
            )

            assert "scenarios" in scenario_results
            assert "uncertainty" in scenario_results
            assert "plausibility_scores" in scenario_results
            assert len(scenario_results["scenarios"]) == 25
            assert len(scenario_results["uncertainty"]) == 25
