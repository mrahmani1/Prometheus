"""
Unit tests for Physics-Informed Neural Network (PINN) Engine
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.pinn_engine import (
    ActivationRegistry,
    ElasticDeformationConstraint,
    GeologicalProcess,
    HeatTransferConstraint,
    MassConservationConstraint,
    NavierStokesConstraint,
    PhysicsConstraintRegistry,
    PINNEngine,
    PINNLoss,
    PINNNetwork,
    PINNTrainer,
)
from models.ai_models import (
    ActivationFunction,
    NetworkArchitecture,
    PhysicsConstraint,
    PhysicsConstraintType,
    PINNModelConfig,
)
from models.ai_models import numpy as np
from models.ai_models import torch.nn as nn
from models.ai_models import torch.nn.functional as F

class TestActivationRegistry:
    """Test activation function registry"""

    def test_get_activation_relu(self):
        activation = ActivationRegistry.get_activation(ActivationFunction.RELU)
        assert isinstance(activation, torch.nn.ReLU)

    def test_get_activation_tanh(self):
        activation = ActivationRegistry.get_activation(ActivationFunction.TANH)
        assert isinstance(activation, torch.nn.Tanh)

    def test_get_activation_sigmoid(self):
        activation = ActivationRegistry.get_activation(ActivationFunction.SIGMOID)
        assert isinstance(activation, torch.nn.Sigmoid)

class TestPhysicsConstraints:
    """Test physics constraint solvers"""

    def setup_method(self):
        """Setup test data"""
        self.batch_size = 10
        self.input_dim = 4  # x, y, z, t
        self.output_dim = 4  # u, v, w, p or T

        # Create sample inputs (x, y, z, t)
        self.inputs = torch.randn(self.batch_size, self.input_dim, requires_grad=True)

        # Create sample outputs
        self.outputs = torch.randn(self.batch_size, self.output_dim, requires_grad=True)

    def test_navier_stokes_constraint(self):
        """Test Navier-Stokes constraint computation"""
        constraint = PhysicsConstraint(
            constraint_type=PhysicsConstraintType.NAVIER_STOKES,
            equation_parameters={"viscosity": 1e-3, "density": 1000.0},
            weight=1.0,
        )

        solver = NavierStokesConstraint()
        residual = solver.compute_residual(self.inputs, self.outputs, constraint)

        # Check residual shape (should have 4 components: 3 momentum + 1 continuity)
        assert residual.shape == (self.batch_size, 4)

        # Check that residual requires gradient
        assert residual.requires_grad

    def test_heat_transfer_constraint(self):
        """Test heat transfer constraint computation"""
        constraint = PhysicsConstraint(
            constraint_type=PhysicsConstraintType.HEAT_TRANSFER,
            equation_parameters={
                "thermal_conductivity": 1.0,
                "specific_heat": 1000.0,
                "density": 1000.0,
            },
            weight=1.0,
        )

        solver = HeatTransferConstraint()
        residual = solver.compute_residual(self.inputs, self.outputs, constraint)

        # Check residual shape (should be single component for temperature)
        assert residual.shape == (self.batch_size, 1)
        assert residual.requires_grad

    def test_elastic_deformation_constraint(self):
        """Test elastic deformation constraint computation"""
        constraint = PhysicsConstraint(
            constraint_type=PhysicsConstraintType.ELASTIC_DEFORMATION,
            equation_parameters={"youngs_modulus": 2e11, "poisson_ratio": 0.25},
            weight=1.0,
        )

        solver = ElasticDeformationConstraint()
        residual = solver.compute_residual(self.inputs, self.outputs[:, :3], constraint)

        # Check residual shape (should have 3 components for u, v, w)
        assert residual.shape == (self.batch_size, 3)
        assert residual.requires_grad

    def test_mass_conservation_constraint(self):
        """Test mass conservation constraint computation"""
        constraint = PhysicsConstraint(
            constraint_type=PhysicsConstraintType.MASS_CONSERVATION,
            equation_parameters={"diffusion_coefficient": 1e-9},
            weight=1.0,
        )

        solver = MassConservationConstraint()
        residual = solver.compute_residual(self.inputs, self.outputs, constraint)

        # Check residual shape (should be single component for concentration)
        assert residual.shape == (self.batch_size, 1)
        assert residual.requires_grad

class TestPhysicsConstraintRegistry:
    """Test physics constraint registry"""

    def test_get_solver_navier_stokes(self):
        registry = PhysicsConstraintRegistry()
        solver = registry.get_solver(PhysicsConstraintType.NAVIER_STOKES)
        assert isinstance(solver, NavierStokesConstraint)

    def test_get_solver_heat_transfer(self):
        registry = PhysicsConstraintRegistry()
        solver = registry.get_solver(PhysicsConstraintType.HEAT_TRANSFER)
        assert isinstance(solver, HeatTransferConstraint)

    def test_get_solver_unsupported(self):
        registry = PhysicsConstraintRegistry()
        with pytest.raises(ValueError, match="Unsupported physics constraint type"):
            registry.get_solver("unsupported_type")

class TestPINNNetwork:
    """Test PINN network architecture"""

    def setup_method(self):
        """Setup test network"""
        self.architecture = NetworkArchitecture(
            layer_sizes=[4, 64, 64, 3],
            activation_functions=[
                ActivationFunction.TANH,
                ActivationFunction.TANH,
                ActivationFunction.TANH,
            ],
            dropout_rates=[0.1, 0.1, 0.0],
            batch_normalization=False,
            residual_connections=False,
            attention_mechanism=False,
        )

        self.network = PINNNetwork(
            self.architecture, GeologicalProcess.SEISMIC_WAVE_PROPAGATION
        )

    def test_network_creation(self):
        """Test network creation"""
        assert len(self.network.layers) == 3  # 3 linear layers
        assert len(self.network.activations) == 3
        assert len(self.network.dropout_layers) == 3

    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 10
        input_tensor = torch.randn(batch_size, 4, requires_grad=True)

        output = self.network(input_tensor)

        assert output.shape == (batch_size, 3)
        assert output.requires_grad

    def test_network_with_residual_connections(self):
        """Test network with residual connections"""
        architecture = NetworkArchitecture(
            layer_sizes=[4, 64, 64, 3],
            activation_functions=[
                ActivationFunction.RELU,
                ActivationFunction.RELU,
                ActivationFunction.RELU,
            ],
            residual_connections=True,
        )

        network = PINNNetwork(architecture, GeologicalProcess.MAGMA_FLOW)

        input_tensor = torch.randn(5, 4, requires_grad=True)
        output = network(input_tensor)

        assert output.shape == (5, 3)
        assert output.requires_grad

class TestPINNLoss:
    """Test PINN loss function"""

    def setup_method(self):
        """Setup test loss function"""
        self.physics_constraints = [
            PhysicsConstraint(
                constraint_type=PhysicsConstraintType.HEAT_TRANSFER,
                equation_parameters={
                    "thermal_conductivity": 1.0,
                    "specific_heat": 1000.0,
                    "density": 1000.0,
                },
                weight=1.0,
            )
        ]

        self.loss_fn = PINNLoss(self.physics_constraints)

        # Create mock model
        self.mock_model = Mock()

    def test_data_loss_computation(self):
        """Test data loss computation"""
        batch_size = 10
        output_dim = 3

        predictions = torch.randn(batch_size, output_dim)
        targets = torch.randn(batch_size, output_dim)
        inputs = torch.randn(batch_size, 4, requires_grad=True)

        total_loss, loss_components = self.loss_fn(
            predictions, targets, inputs, self.mock_model
        )

        assert "data_loss" in loss_components
        assert "total_loss" in loss_components
        assert total_loss.requires_grad

    def test_physics_loss_computation(self):
        """Test physics loss computation"""
        batch_size = 10
        output_dim = 3

        predictions = torch.randn(batch_size, output_dim, requires_grad=True)
        inputs = torch.randn(batch_size, 4, requires_grad=True)

        total_loss, loss_components = self.loss_fn(
            predictions, None, inputs, self.mock_model
        )

        assert "physics_heat_transfer" in loss_components
        assert "physics_loss_total" in loss_components
        assert total_loss.requires_grad

    def test_combined_loss(self):
        """Test combined data and physics loss"""
        batch_size = 10
        output_dim = 3

        predictions = torch.randn(batch_size, output_dim, requires_grad=True)
        targets = torch.randn(batch_size, output_dim)
        inputs = torch.randn(batch_size, 4, requires_grad=True)

        total_loss, loss_components = self.loss_fn(
            predictions, targets, inputs, self.mock_model
        )

        assert "data_loss" in loss_components
        assert "physics_heat_transfer" in loss_components
        assert "physics_loss_total" in loss_components
        assert "total_loss" in loss_components
        assert total_loss.requires_grad

class TestPINNTrainer:
    """Test PINN trainer"""

    def setup_method(self):
        """Setup test trainer"""
        # Create network architecture
        architecture = NetworkArchitecture(
            layer_sizes=[4, 32, 32, 3],
            activation_functions=[
                ActivationFunction.TANH,
                ActivationFunction.TANH,
                ActivationFunction.TANH,
            ],
        )

        # Create physics constraints
        physics_constraints = [
            PhysicsConstraint(
                constraint_type=PhysicsConstraintType.HEAT_TRANSFER,
                equation_parameters={
                    "thermal_conductivity": 1.0,
                    "specific_heat": 1000.0,
                    "density": 1000.0,
                },
                weight=1.0,
            )
        ]

        # Create PINN config
        self.config = PINNModelConfig(
            model_id="test_model",
            network_architecture=architecture,
            physics_constraints=physics_constraints,
            optimization_config={"optimizer": "adam", "learning_rate": 0.001},
            training_config={"max_epochs": 10, "batch_size": 32},
        )

        # Create network and trainer
        self.network = PINNNetwork(architecture, GeologicalProcess.HEAT_TRANSFER)
        self.trainer = PINNTrainer(self.network, self.config)

    def test_trainer_creation(self):
        """Test trainer creation"""
        assert self.trainer.model is not None
        assert self.trainer.optimizer is not None
        assert self.trainer.loss_fn is not None

    def test_train_step(self):
        """Test single training step"""
        batch_size = 16
        inputs = torch.randn(batch_size, 4, requires_grad=True)
        targets = torch.randn(batch_size, 3)

        loss_components = self.trainer.train_step(inputs, targets)

        assert "data_loss" in loss_components
        assert "total_loss" in loss_components
        assert isinstance(loss_components["total_loss"], float)

    def test_optimizer_creation_adam(self):
        """Test Adam optimizer creation"""
        assert isinstance(self.trainer.optimizer, torch.optim.Adam)
        assert self.trainer.optimizer.param_groups[0]["lr"] == 0.001

    def test_optimizer_creation_sgd(self):
        """Test SGD optimizer creation"""
        config = self.config
        config.optimization_config["optimizer"] = "sgd"

        trainer = PINNTrainer(self.network, config)
        assert isinstance(trainer.optimizer, torch.optim.SGD)

class TestPINNEngine:
    """Test PINN engine"""

    def setup_method(self):
        """Setup test engine"""
        self.engine = PINNEngine()

        # Create test configuration
        architecture = NetworkArchitecture(
            layer_sizes=[4, 32, 32, 3],
            activation_functions=[
                ActivationFunction.TANH,
                ActivationFunction.TANH,
                ActivationFunction.TANH,
            ],
        )

        physics_constraints = [
            PhysicsConstraint(
                constraint_type=PhysicsConstraintType.HEAT_TRANSFER,
                equation_parameters={
                    "thermal_conductivity": 1.0,
                    "specific_heat": 1000.0,
                    "density": 1000.0,
                },
                weight=1.0,
            )
        ]

        self.config = PINNModelConfig(
            model_id="test_model",
            network_architecture=architecture,
            physics_constraints=physics_constraints,
            optimization_config={"optimizer": "adam", "learning_rate": 0.001},
            training_config={"max_epochs": 10, "batch_size": 32},
        )

    def test_engine_creation(self):
        """Test engine creation"""
        assert len(self.engine.models) == 0
        assert len(self.engine.trainers) == 0
        assert len(self.engine.configs) == 0

    def test_create_model(self):
        """Test model creation"""
        model = self.engine.create_model(
            "test_model", self.config, GeologicalProcess.HEAT_TRANSFER
        )

        assert isinstance(model, PINNNetwork)
        assert "test_model" in self.engine.models
        assert "test_model" in self.engine.trainers
        assert "test_model" in self.engine.configs

    def test_get_model(self):
        """Test getting model by ID"""
        self.engine.create_model("test_model", self.config)

        model = self.engine.get_model("test_model")
        assert model is not None
        assert isinstance(model, PINNNetwork)

        # Test non-existent model
        assert self.engine.get_model("non_existent") is None

    def test_list_models(self):
        """Test listing models"""
        assert len(self.engine.list_models()) == 0

        self.engine.create_model("model1", self.config)
        self.engine.create_model("model2", self.config)

        models = self.engine.list_models()
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models

    def test_remove_model(self):
        """Test removing model"""
        self.engine.create_model("test_model", self.config)
        assert "test_model" in self.engine.models

        self.engine.remove_model("test_model")
        assert "test_model" not in self.engine.models
        assert "test_model" not in self.engine.trainers
        assert "test_model" not in self.engine.configs

    def test_predict(self):
        """Test model prediction"""
        self.engine.create_model("test_model", self.config)

        batch_size = 5
        inputs = torch.randn(batch_size, 4)

        predictions = self.engine.predict("test_model", inputs)

        assert predictions.shape == (batch_size, 3)
        assert not predictions.requires_grad  # Should be in eval mode

    def test_predict_nonexistent_model(self):
        """Test prediction with non-existent model"""
        inputs = torch.randn(5, 4)

        with pytest.raises(ValueError, match="Model non_existent not found"):
            self.engine.predict("non_existent", inputs)

    def test_get_model_info(self):
        """Test getting model information"""
        self.engine.create_model(
            "test_model", self.config, GeologicalProcess.MAGMA_FLOW
        )

        info = self.engine.get_model_info("test_model")

        assert info["model_id"] == "test_model"
        assert info["geological_process"] == GeologicalProcess.MAGMA_FLOW.value
        assert "architecture" in info
        assert "physics_constraints" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0

    def test_get_model_info_nonexistent(self):
        """Test getting info for non-existent model"""
        with pytest.raises(ValueError, match="Model non_existent not found"):
            self.engine.get_model_info("non_existent")

class TestIntegration:
    """Integration tests for PINN engine components"""

    def test_end_to_end_training(self):
        """Test end-to-end training process"""
        # Create engine
        engine = PINNEngine()

        # Create configuration
        architecture = NetworkArchitecture(
            layer_sizes=[4, 16, 16, 1],  # Small network for fast testing
            activation_functions=[
                ActivationFunction.TANH,
                ActivationFunction.TANH,
                ActivationFunction.TANH,
            ],
        )

        physics_constraints = [
            PhysicsConstraint(
                constraint_type=PhysicsConstraintType.HEAT_TRANSFER,
                equation_parameters={
                    "thermal_conductivity": 1.0,
                    "specific_heat": 1000.0,
                    "density": 1000.0,
                },
                weight=1.0,
            )
        ]

        config = PINNModelConfig(
            model_id="integration_test",
            network_architecture=architecture,
            physics_constraints=physics_constraints,
            optimization_config={"optimizer": "adam", "learning_rate": 0.01},
            training_config={
                "max_epochs": 5,  # Small number for testing
                "batch_size": 16,
            },
        )

        # Create model
        model = engine.create_model(
            "integration_test", config, GeologicalProcess.HEAT_TRANSFER
        )
        trainer = engine.get_trainer("integration_test")

        # Create synthetic training data
        batch_size = 32
        inputs = torch.randn(batch_size, 4, requires_grad=True)
        targets = torch.randn(batch_size, 1)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        # Train model
        history = trainer.train(train_loader)

        # Verify training history
        assert len(history["epoch"]) == 5
        assert len(history["total_loss"]) == 5
        assert all(isinstance(loss, float) for loss in history["total_loss"])

        # Test prediction
        test_inputs = torch.randn(10, 4)
        predictions = engine.predict("integration_test", test_inputs)

        assert predictions.shape == (10, 1)
        assert not predictions.requires_grad

    def test_multiple_physics_constraints(self):
        """Test model with multiple physics constraints"""
        engine = PINNEngine()

        architecture = NetworkArchitecture(
            layer_sizes=[4, 32, 32, 4],  # Output: u, v, w, T
            activation_functions=[
                ActivationFunction.TANH,
                ActivationFunction.TANH,
                ActivationFunction.TANH,
            ],
        )

        physics_constraints = [
            PhysicsConstraint(
                constraint_type=PhysicsConstraintType.NAVIER_STOKES,
                equation_parameters={"viscosity": 1e-3, "density": 1000.0},
                weight=1.0,
            ),
            PhysicsConstraint(
                constraint_type=PhysicsConstraintType.HEAT_TRANSFER,
                equation_parameters={
                    "thermal_conductivity": 1.0,
                    "specific_heat": 1000.0,
                    "density": 1000.0,
                },
                weight=0.5,
            ),
        ]

        config = PINNModelConfig(
            model_id="multi_physics",
            network_architecture=architecture,
            physics_constraints=physics_constraints,
            optimization_config={"optimizer": "adam", "learning_rate": 0.001},
            training_config={"max_epochs": 3, "batch_size": 16},
        )

        # Create and test model
        model = engine.create_model(
            "multi_physics", config, GeologicalProcess.MAGMA_FLOW
        )

        # Test forward pass
        inputs = torch.randn(10, 4, requires_grad=True)
        outputs = model(inputs)

        assert outputs.shape == (10, 4)
        assert outputs.requires_grad

        # Test loss computation
        trainer = engine.get_trainer("multi_physics")
        loss_components = trainer.train_step(inputs)

        assert "physics_navier_stokes" in loss_components
        assert "physics_heat_transfer" in loss_components
        assert "physics_loss_total" in loss_components

if __name__ == "__main__":
    pytest.main([__file__])
