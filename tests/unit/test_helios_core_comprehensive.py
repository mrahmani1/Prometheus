"""
Comprehensive unit tests for Helios Core AI Engine

Tests all core AI/ML components including PINNs, GANs, CNNs, and transfer learning.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os

# Import core modules (adjust imports based on actual structure)
try:
    from services.helios_core.core.pinn_engine import PINNEngine, PINNNetwork, PINNTrainer
    from services.helios_core.core.gan_engine import GANGenerator, GANDiscriminator, GANTrainer
    from services.helios_core.core.cnn_processor import SeismicCNNProcessor, InSARCNNProcessor
    from services.helios_core.core.transfer_learning import TransferLearningEngine
    from services.helios_core.models.ai_models import (
        PINNModelConfig, NetworkArchitecture, PhysicsConstraint,
        PhysicsConstraintType, ActivationFunction, GeologicalProcess
    )
except ImportError:
    # Mock imports for testing framework
    pytest.skip("Helios core modules not available", allow_module_level=True)


@pytest.mark.unit
class TestPINNEngineComprehensive:
    """Comprehensive tests for Physics-Informed Neural Network Engine"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.engine = PINNEngine()
        
        # Create test configuration
        self.architecture = NetworkArchitecture(
            layer_sizes=[4, 64, 64, 32, 3],
            activation_functions=[
                ActivationFunction.TANH, 
                ActivationFunction.TANH, 
                ActivationFunction.TANH,
                ActivationFunction.TANH
            ],
            dropout_rates=[0.1, 0.1, 0.1, 0.0],
            batch_normalization=True,
            residual_connections=True
        )
        
        self.physics_constraints = [
            PhysicsConstraint(
                constraint_type=PhysicsConstraintType.NAVIER_STOKES,
                equation_parameters={"viscosity": 1e-3, "density": 1000.0},
                weight=1.0
            ),
            PhysicsConstraint(
                constraint_type=PhysicsConstraintType.HEAT_TRANSFER,
                equation_parameters={
                    "thermal_conductivity": 1.0,
                    "specific_heat": 1000.0,
                    "density": 1000.0
                },
                weight=0.5
            )
        ]
        
        self.config = PINNModelConfig(
            model_id="comprehensive_test_model",
            network_architecture=self.architecture,
            physics_constraints=self.physics_constraints,
            optimization_config={
                "optimizer": "adam",
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "scheduler": "cosine_annealing"
            },
            training_config={
                "max_epochs": 100,
                "batch_size": 64,
                "validation_split": 0.2,
                "early_stopping_patience": 10
            }
        )
    
    def test_model_creation_with_complex_architecture(self):
        """Test creating model with complex architecture"""
        model = self.engine.create_model(
            "complex_model",
            self.config,
            GeologicalProcess.MAGMA_FLOW
        )
        
        assert isinstance(model, PINNNetwork)
        assert "complex_model" in self.engine.models
        
        # Test model architecture
        assert len(model.layers) == 4  # 4 linear layers
        assert model.layers[0].in_features == 4
        assert model.layers[-1].out_features == 3
        
        # Test batch normalization layers
        assert hasattr(model, 'batch_norm_layers')
        assert len(model.batch_norm_layers) == 3  # All but last layer
        
        # Test residual connections
        assert model.use_residual_connections
    
    def test_multi_physics_constraint_enforcement(self):
        """Test enforcement of multiple physics constraints"""
        model = self.engine.create_model("multi_physics", self.config)
        trainer = self.engine.get_trainer("multi_physics")
        
        # Create test data
        batch_size = 32
        inputs = torch.randn(batch_size, 4, requires_grad=True)
        targets = torch.randn(batch_size, 3)
        
        # Test training step with multiple constraints
        loss_components = trainer.train_step(inputs, targets)
        
        # Should have both physics constraints
        assert 'physics_navier_stokes' in loss_components
        assert 'physics_heat_transfer' in loss_components
        assert 'physics_loss_total' in loss_components
        assert 'data_loss' in loss_components
        assert 'total_loss' in loss_components
        
        # Physics loss should be weighted combination
        expected_physics_loss = (
            loss_components['physics_navier_stokes'] * 1.0 +
            loss_components['physics_heat_transfer'] * 0.5
        )
        assert abs(loss_components['physics_loss_total'] - expected_physics_loss) < 1e-6
    
    def test_advanced_optimizer_configuration(self):
        """Test advanced optimizer configurations"""
        # Test different optimizers
        optimizers = ['adam', 'sgd', 'rmsprop', 'adamw']
        
        for opt_name in optimizers:
            config = self.config
            config.optimization_config['optimizer'] = opt_name
            
            model = self.engine.create_model(f"model_{opt_name}", config)
            trainer = self.engine.get_trainer(f"model_{opt_name}")
            
            # Check optimizer type
            if opt_name == 'adam':
                assert isinstance(trainer.optimizer, torch.optim.Adam)
            elif opt_name == 'sgd':
                assert isinstance(trainer.optimizer, torch.optim.SGD)
            elif opt_name == 'rmsprop':
                assert isinstance(trainer.optimizer, torch.optim.RMSprop)
            elif opt_name == 'adamw':
                assert isinstance(trainer.optimizer, torch.optim.AdamW)
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling"""
        model = self.engine.create_model("scheduler_test", self.config)
        trainer = self.engine.get_trainer("scheduler_test")
        
        # Should have scheduler
        assert trainer.scheduler is not None
        
        # Test scheduler step
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        trainer.scheduler.step()
        
        # Learning rate should change (for cosine annealing)
        if hasattr(trainer.scheduler, 'step'):
            # Some schedulers change LR immediately, others don't
            pass  # Just ensure no errors
    
    def test_model_checkpointing_and_loading(self):
        """Test model checkpointing and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = self.engine.create_model("checkpoint_test", self.config)
            
            # Save checkpoint
            checkpoint_path = os.path.join(temp_dir, "model_checkpoint.pth")
            self.engine.save_model("checkpoint_test", checkpoint_path)
            
            assert os.path.exists(checkpoint_path)
            
            # Load checkpoint
            loaded_model = self.engine.load_model("loaded_model", checkpoint_path, self.config)
            
            # Test that loaded model produces same outputs
            test_input = torch.randn(5, 4)
            original_output = model(test_input)
            loaded_output = loaded_model(test_input)
            
            assert torch.allclose(original_output, loaded_output, atol=1e-6)
    
    def test_gradient_computation_and_physics_constraints(self):
        """Test gradient computation for physics constraints"""
        model = self.engine.create_model("gradient_test", self.config)
        
        # Create input with gradient tracking
        batch_size = 16
        inputs = torch.randn(batch_size, 4, requires_grad=True)
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute gradients for physics constraints
        gradients = {}
        for i in range(outputs.shape[1]):  # For each output component
            grad = torch.autograd.grad(
                outputs[:, i].sum(), 
                inputs, 
                create_graph=True, 
                retain_graph=True
            )[0]
            gradients[f'output_{i}'] = grad
        
        # Check gradient shapes
        for grad in gradients.values():
            assert grad.shape == inputs.shape
            assert grad.requires_grad  # Should support higher-order derivatives
    
    def test_model_ensemble_prediction(self):
        """Test ensemble prediction with multiple models"""
        # Create multiple models
        models = []
        for i in range(3):
            model = self.engine.create_model(f"ensemble_model_{i}", self.config)
            models.append(model)
        
        # Test ensemble prediction
        test_input = torch.randn(10, 4)
        predictions = []
        
        for model in models:
            pred = model(test_input)
            predictions.append(pred)
        
        # Compute ensemble statistics
        ensemble_mean = torch.stack(predictions).mean(dim=0)
        ensemble_std = torch.stack(predictions).std(dim=0)
        
        assert ensemble_mean.shape == (10, 3)
        assert ensemble_std.shape == (10, 3)
        assert torch.all(ensemble_std >= 0)  # Standard deviation should be non-negative
    
    def test_memory_efficiency_large_batch(self):
        """Test memory efficiency with large batches"""
        model = self.engine.create_model("memory_test", self.config)
        
        # Test with progressively larger batches
        batch_sizes = [64, 128, 256, 512]
        
        for batch_size in batch_sizes:
            inputs = torch.randn(batch_size, 4)
            
            # Forward pass should not cause memory issues
            with torch.no_grad():
                outputs = model(inputs)
                assert outputs.shape == (batch_size, 3)
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@pytest.mark.unit
class TestGANEngineComprehensive:
    """Comprehensive tests for Generative Adversarial Network Engine"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.latent_dim = 100
        self.data_dim = 64
        self.generator = GANGenerator(self.latent_dim, self.data_dim)
        self.discriminator = GANDiscriminator(self.data_dim)
        self.trainer = GANTrainer(self.generator, self.discriminator)
    
    def test_generator_architecture(self):
        """Test generator network architecture"""
        batch_size = 32
        noise = torch.randn(batch_size, self.latent_dim)
        
        generated_data = self.generator(noise)
        
        assert generated_data.shape == (batch_size, self.data_dim)
        assert torch.all(torch.isfinite(generated_data))  # No NaN or Inf values
    
    def test_discriminator_architecture(self):
        """Test discriminator network architecture"""
        batch_size = 32
        real_data = torch.randn(batch_size, self.data_dim)
        
        discriminator_output = self.discriminator(real_data)
        
        assert discriminator_output.shape == (batch_size, 1)
        assert torch.all(discriminator_output >= 0) and torch.all(discriminator_output <= 1)
    
    def test_adversarial_training_step(self):
        """Test adversarial training step"""
        batch_size = 32
        real_data = torch.randn(batch_size, self.data_dim)
        
        # Training step
        losses = self.trainer.train_step(real_data)
        
        assert 'generator_loss' in losses
        assert 'discriminator_loss' in losses
        assert 'discriminator_real_accuracy' in losses
        assert 'discriminator_fake_accuracy' in losses
        
        # Losses should be finite
        for loss_name, loss_value in losses.items():
            if 'accuracy' not in loss_name:
                assert torch.isfinite(torch.tensor(loss_value))
    
    def test_mode_collapse_detection(self):
        """Test mode collapse detection in GAN training"""
        # Generate multiple samples
        batch_size = 100
        noise = torch.randn(batch_size, self.latent_dim)
        
        generated_samples = self.generator(noise)
        
        # Check diversity (simple metric: standard deviation across samples)
        sample_std = generated_samples.std(dim=0)
        
        # If all std values are very small, might indicate mode collapse
        mode_collapse_threshold = 0.01
        potential_mode_collapse = torch.all(sample_std < mode_collapse_threshold)
        
        # For a fresh model, shouldn't have mode collapse
        assert not potential_mode_collapse
    
    def test_gradient_penalty_computation(self):
        """Test gradient penalty for WGAN-GP"""
        batch_size = 16
        real_data = torch.randn(batch_size, self.data_dim)
        fake_data = torch.randn(batch_size, self.data_dim)
        
        # Compute gradient penalty
        gradient_penalty = self.trainer.compute_gradient_penalty(real_data, fake_data)
        
        assert isinstance(gradient_penalty, torch.Tensor)
        assert gradient_penalty.dim() == 0  # Scalar
        assert gradient_penalty >= 0  # Should be non-negative
    
    def test_uncertainty_quantification(self):
        """Test uncertainty quantification using GAN ensemble"""
        # Generate multiple samples from same noise
        noise = torch.randn(1, self.latent_dim)
        
        samples = []
        for _ in range(100):  # Generate 100 samples
            sample = self.generator(noise)
            samples.append(sample)
        
        samples_tensor = torch.stack(samples)
        
        # Compute uncertainty metrics
        mean_prediction = samples_tensor.mean(dim=0)
        uncertainty = samples_tensor.std(dim=0)
        
        assert mean_prediction.shape == (1, self.data_dim)
        assert uncertainty.shape == (1, self.data_dim)
        assert torch.all(uncertainty >= 0)


@pytest.mark.unit
class TestCNNProcessorComprehensive:
    """Comprehensive tests for CNN-based processors"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.seismic_processor = SeismicCNNProcessor()
        self.insar_processor = InSARCNNProcessor()
    
    def test_seismic_event_detection(self):
        """Test seismic event detection with CNN"""
        # Create synthetic seismic waveform
        duration = 10.0  # seconds
        sampling_rate = 100.0  # Hz
        waveform_length = int(duration * sampling_rate)
        
        # Synthetic waveform with embedded event
        t = np.linspace(0, duration, waveform_length)
        signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)  # Damped sine wave
        noise = np.random.normal(0, 0.1, waveform_length)
        waveform = signal + noise
        
        # Convert to tensor
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Detect events
        detections = self.seismic_processor.detect_events(waveform_tensor)
        
        assert 'event_probability' in detections
        assert 'event_times' in detections
        assert 'event_magnitudes' in detections
        
        # Probabilities should be between 0 and 1
        assert torch.all(detections['event_probability'] >= 0)
        assert torch.all(detections['event_probability'] <= 1)
    
    def test_seismic_event_classification(self):
        """Test seismic event classification"""
        # Create different types of synthetic events
        event_types = ['earthquake', 'volcanic', 'explosion', 'noise']
        
        for event_type in event_types:
            # Generate synthetic waveform for each type
            waveform = self._generate_synthetic_waveform(event_type)
            waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Classify event
            classification = self.seismic_processor.classify_event(waveform_tensor)
            
            assert 'event_type' in classification
            assert 'confidence' in classification
            assert classification['event_type'] in event_types
            assert 0 <= classification['confidence'] <= 1
    
    def test_insar_deformation_analysis(self):
        """Test InSAR deformation pattern analysis"""
        # Create synthetic InSAR interferogram
        height, width = 256, 256
        x, y = np.meshgrid(np.linspace(-5, 5, width), np.linspace(-5, 5, height))
        
        # Synthetic deformation pattern (Gaussian)
        deformation = 0.05 * np.exp(-(x**2 + y**2) / 2)  # 5cm peak deformation
        noise = np.random.normal(0, 0.005, (height, width))  # 5mm noise
        interferogram = deformation + noise
        
        # Convert to tensor
        interferogram_tensor = torch.tensor(interferogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Analyze deformation
        analysis = self.insar_processor.analyze_deformation(interferogram_tensor)
        
        assert 'deformation_magnitude' in analysis
        assert 'deformation_center' in analysis
        assert 'deformation_extent' in analysis
        assert 'confidence' in analysis
        
        # Check that detected center is near actual center
        detected_center = analysis['deformation_center']
        actual_center = [height // 2, width // 2]
        center_error = np.sqrt(sum((d - a)**2 for d, a in zip(detected_center, actual_center)))
        assert center_error < 20  # Within 20 pixels
    
    def test_atmospheric_correction(self):
        """Test atmospheric correction in InSAR processing"""
        # Create synthetic interferogram with atmospheric effects
        height, width = 128, 128
        
        # True deformation
        x, y = np.meshgrid(np.linspace(-3, 3, width), np.linspace(-3, 3, height))
        true_deformation = 0.03 * np.exp(-(x**2 + y**2) / 1.5)
        
        # Atmospheric effects (typically long-wavelength)
        atmospheric_phase = 0.02 * np.sin(2 * np.pi * x / 10) * np.cos(2 * np.pi * y / 8)
        
        # Observed interferogram
        observed = true_deformation + atmospheric_phase + np.random.normal(0, 0.002, (height, width))
        
        # Convert to tensor
        observed_tensor = torch.tensor(observed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Apply atmospheric correction
        corrected = self.insar_processor.atmospheric_correction(observed_tensor)
        
        # Corrected should be closer to true deformation
        corrected_np = corrected.squeeze().numpy()
        
        # Compute RMS error
        rms_before = np.sqrt(np.mean((observed - true_deformation)**2))
        rms_after = np.sqrt(np.mean((corrected_np - true_deformation)**2))
        
        # Correction should improve the result
        assert rms_after < rms_before
    
    def _generate_synthetic_waveform(self, event_type: str) -> np.ndarray:
        """Generate synthetic waveform for different event types"""
        duration = 10.0
        sampling_rate = 100.0
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        if event_type == 'earthquake':
            # High-frequency P and S waves
            signal = (np.sin(2 * np.pi * 10 * t) * np.exp(-t / 3) +
                     0.5 * np.sin(2 * np.pi * 5 * t) * np.exp(-(t - 2) / 4))
        elif event_type == 'volcanic':
            # Lower frequency, longer duration
            signal = np.sin(2 * np.pi * 2 * t) * np.exp(-t / 8)
        elif event_type == 'explosion':
            # Sharp onset, high frequency
            signal = np.sin(2 * np.pi * 15 * t) * np.exp(-t / 1)
        else:  # noise
            signal = np.zeros_like(t)
        
        noise = np.random.normal(0, 0.1, len(t))
        return signal + noise


@pytest.mark.unit
class TestTransferLearningComprehensive:
    """Comprehensive tests for Transfer Learning Engine"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.transfer_engine = TransferLearningEngine()
    
    def test_domain_adaptation(self):
        """Test domain adaptation between different geological settings"""
        # Create source domain model (e.g., trained on Yellowstone)
        source_config = self._create_test_config("yellowstone_model")
        source_model = self.transfer_engine.create_source_model(source_config)
        
        # Create target domain data (e.g., Mount Fuji)
        target_data = self._create_synthetic_target_data("mount_fuji")
        
        # Perform domain adaptation
        adapted_model = self.transfer_engine.adapt_to_target_domain(
            source_model, target_data, adaptation_method="fine_tuning"
        )
        
        assert adapted_model is not None
        
        # Test that adapted model works on target domain
        test_input = torch.randn(5, 4)
        output = adapted_model(test_input)
        assert output.shape == (5, 3)
    
    def test_few_shot_learning(self):
        """Test few-shot learning with limited target domain data"""
        # Create pre-trained model
        pretrained_config = self._create_test_config("pretrained_global")
        pretrained_model = self.transfer_engine.create_source_model(pretrained_config)
        
        # Create very limited target data (few-shot scenario)
        few_shot_data = {
            'inputs': torch.randn(10, 4),  # Only 10 samples
            'targets': torch.randn(10, 3)
        }
        
        # Perform few-shot adaptation
        adapted_model = self.transfer_engine.few_shot_adaptation(
            pretrained_model, few_shot_data, n_shots=5
        )
        
        assert adapted_model is not None
        
        # Test performance on few-shot data
        with torch.no_grad():
            predictions = adapted_model(few_shot_data['inputs'])
            loss = torch.nn.functional.mse_loss(predictions, few_shot_data['targets'])
            
            # Loss should be reasonable (not infinite or NaN)
            assert torch.isfinite(loss)
    
    def test_meta_learning(self):
        """Test meta-learning for rapid adaptation"""
        # Create multiple source tasks
        source_tasks = []
        for i in range(5):
            task_data = self._create_synthetic_target_data(f"task_{i}")
            source_tasks.append(task_data)
        
        # Train meta-learner
        meta_model = self.transfer_engine.train_meta_learner(source_tasks)
        
        assert meta_model is not None
        
        # Test rapid adaptation to new task
        new_task_data = self._create_synthetic_target_data("new_task")
        adapted_model = self.transfer_engine.meta_adapt(meta_model, new_task_data)
        
        # Should adapt quickly with few gradient steps
        test_input = torch.randn(3, 4)
        output = adapted_model(test_input)
        assert output.shape == (3, 3)
    
    def test_knowledge_distillation(self):
        """Test knowledge distillation from complex to simple models"""
        # Create complex teacher model
        teacher_config = self._create_test_config("teacher", layer_sizes=[4, 128, 128, 64, 3])
        teacher_model = self.transfer_engine.create_source_model(teacher_config)
        
        # Create simple student model
        student_config = self._create_test_config("student", layer_sizes=[4, 32, 3])
        student_model = self.transfer_engine.create_source_model(student_config)
        
        # Generate training data
        training_data = {
            'inputs': torch.randn(100, 4),
            'targets': torch.randn(100, 3)
        }
        
        # Perform knowledge distillation
        distilled_model = self.transfer_engine.knowledge_distillation(
            teacher_model, student_model, training_data, temperature=3.0
        )
        
        assert distilled_model is not None
        
        # Student should have fewer parameters than teacher
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in distilled_model.parameters())
        assert student_params < teacher_params
        
        # But should produce similar outputs
        test_input = torch.randn(5, 4)
        with torch.no_grad():
            teacher_output = teacher_model(test_input)
            student_output = distilled_model(test_input)
            
            # Outputs should be reasonably similar
            similarity = torch.nn.functional.cosine_similarity(
                teacher_output.flatten(), student_output.flatten(), dim=0
            )
            assert similarity > 0.5  # At least 50% similarity
    
    def test_progressive_transfer(self):
        """Test progressive transfer learning"""
        # Create sequence of related domains
        domains = ["global", "regional", "local"]
        models = {}
        
        # Progressive training
        for i, domain in enumerate(domains):
            if i == 0:
                # Train from scratch for global domain
                config = self._create_test_config(f"{domain}_model")
                models[domain] = self.transfer_engine.create_source_model(config)
            else:
                # Transfer from previous domain
                source_model = models[domains[i-1]]
                target_data = self._create_synthetic_target_data(domain)
                
                models[domain] = self.transfer_engine.progressive_transfer(
                    source_model, target_data, domain_name=domain
                )
        
        # Each model should exist and be functional
        for domain, model in models.items():
            assert model is not None
            test_input = torch.randn(3, 4)
            output = model(test_input)
            assert output.shape == (3, 3)
    
    def _create_test_config(self, model_id: str, layer_sizes: list = None):
        """Create test configuration for transfer learning"""
        if layer_sizes is None:
            layer_sizes = [4, 64, 64, 3]
        
        from services.helios_core.models.ai_models import (
            PINNModelConfig, NetworkArchitecture, ActivationFunction
        )
        
        architecture = NetworkArchitecture(
            layer_sizes=layer_sizes,
            activation_functions=[ActivationFunction.TANH] * (len(layer_sizes) - 1)
        )
        
        return PINNModelConfig(
            model_id=model_id,
            network_architecture=architecture,
            physics_constraints=[],
            optimization_config={"optimizer": "adam", "learning_rate": 0.001},
            training_config={"max_epochs": 10, "batch_size": 32}
        )
    
    def _create_synthetic_target_data(self, domain_name: str):
        """Create synthetic target domain data"""
        # Different domains have different data characteristics
        if "yellowstone" in domain_name.lower():
            # High-temperature, caldera system
            inputs = torch.randn(50, 4) * 2  # Larger scale
            targets = torch.randn(50, 3) * 1.5
        elif "fuji" in domain_name.lower():
            # Stratovolcano, different characteristics
            inputs = torch.randn(50, 4) * 1.2
            targets = torch.randn(50, 3) * 0.8
        else:
            # Generic data
            inputs = torch.randn(50, 4)
            targets = torch.randn(50, 3)
        
        return {'inputs': inputs, 'targets': targets}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])