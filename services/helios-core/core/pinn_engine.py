"""
Physics-Informed Neural Network (PINN) Engine

Core implementation of Physics-Informed Neural Networks for geological processes.
Implements automatic differentiation for physics constraint enforcement,
neural network architectures for different geological processes,
loss functions combining data fitting and physics constraints,
and training loops with physics-informed regularization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.ai_models import (
    ActivationFunction,
    NetworkArchitecture,
    OptimizationType,
    PhysicsConstraint,
    PINNModelConfig,
)

from .physics_constraints import create_physics_constraint_enforcer
from .temporal_processing import GeologicalProcess as TemporalGeologicalProcess
from .temporal_processing import (
    MultiScaleTemporalPINN,
    TemporalScale,
    TimeScale,
    create_default_temporal_scales,
)

logger = structlog.get_logger()

class GeologicalProcess(str, Enum):
    """Types of geological processes for specialized PINN architectures"""

    SEISMIC_WAVE_PROPAGATION = "seismic_wave_propagation"
    MAGMA_FLOW = "magma_flow"
    CRUSTAL_DEFORMATION = "crustal_deformation"
    HEAT_TRANSFER = "heat_transfer"
    FLUID_FLOW = "fluid_flow"
    GEOCHEMICAL_TRANSPORT = "geochemical_transport"
    MANTLE_CONVECTION = "mantle_convection"

@dataclass
class PINNTrainingState:
    """Training state for PINN models"""

    epoch: int
    data_loss: float
    physics_loss: float
    total_loss: float
    learning_rate: float
    gradient_norm: float
    physics_residual_max: float
    convergence_metric: float

class ActivationRegistry:
    """Registry of activation functions"""

    @staticmethod
    def get_activation(activation_type: ActivationFunction) -> nn.Module:
        """Get activation function by type"""
        activations = {
            ActivationFunction.RELU: nn.ReLU(),
            ActivationFunction.TANH: nn.Tanh(),
            ActivationFunction.SIGMOID: nn.Sigmoid(),
            ActivationFunction.SWISH: nn.SiLU(),  # SiLU is Swish in PyTorch
            ActivationFunction.GELU: nn.GELU(),
            ActivationFunction.LEAKY_RELU: nn.LeakyReLU(0.01),
            ActivationFunction.ELU: nn.ELU(),
            ActivationFunction.SELU: nn.SELU(),
        }
        return activations[activation_type]

# Physics constraint enforcement is now handled by the dedicated physics_constraints module

class PINNNetwork(nn.Module):
    """
    Physics-Informed Neural Network with configurable architecture
    """

    def __init__(
        self, architecture: NetworkArchitecture, geological_process: GeologicalProcess
    ):
        super().__init__()

        self.architecture = architecture
        self.geological_process = geological_process

        # Build network layers
        self.layers = nn.ModuleList()
        self.activations = []
        self.dropout_layers = nn.ModuleList()

        for i in range(len(architecture.layer_sizes) - 1):
            # Linear layer
            layer = nn.Linear(
                architecture.layer_sizes[i], architecture.layer_sizes[i + 1]
            )
            self.layers.append(layer)

            # Activation function (except for output layer)
            if i < len(architecture.activation_functions):
                activation = ActivationRegistry.get_activation(
                    architecture.activation_functions[i]
                )
                self.activations.append(activation)

            # Dropout layer
            if (
                i < len(architecture.dropout_rates)
                and architecture.dropout_rates[i] > 0
            ):
                dropout = nn.Dropout(architecture.dropout_rates[i])
                self.dropout_layers.append(dropout)
            else:
                self.dropout_layers.append(nn.Identity())

            # Batch normalization
            if (
                architecture.batch_normalization
                and i < len(architecture.layer_sizes) - 2
            ):
                self.layers.append(nn.BatchNorm1d(architecture.layer_sizes[i + 1]))

        # Initialize weights using Xavier initialization
        self._initialize_weights()

        # Residual connections
        self.use_residual = architecture.residual_connections

        # Attention mechanism
        if architecture.attention_mechanism:
            self.attention = nn.MultiheadAttention(
                embed_dim=architecture.layer_sizes[-2], num_heads=8, batch_first=True
            )

    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Ensure input requires gradient for automatic differentiation
        if not x.requires_grad:
            x = x.requires_grad_(True)

        residual = None

        for i, (layer, activation, dropout) in enumerate(
            zip(self.layers, self.activations, self.dropout_layers)
        ):
            if isinstance(layer, nn.Linear):
                # Store input for residual connection
                if self.use_residual and i > 0 and x.shape[-1] == layer.out_features:
                    residual = x

                x = layer(x)

                # Apply batch normalization if present
                if i + 1 < len(self.layers) and isinstance(
                    self.layers[i + 1], nn.BatchNorm1d
                ):
                    x = self.layers[i + 1](x)

                # Apply activation (except for output layer)
                if i < len(self.activations):
                    x = activation(x)

                # Apply dropout
                x = dropout(x)

                # Add residual connection
                if (
                    self.use_residual
                    and residual is not None
                    and x.shape == residual.shape
                ):
                    x = x + residual

        return x

class PINNLoss(nn.Module):
    """
    Combined loss function for Physics-Informed Neural Networks
    """

    def __init__(self, physics_constraints: List[PhysicsConstraint]):
        super().__init__()

        self.physics_constraints = physics_constraints
        self.constraint_enforcer = create_physics_constraint_enforcer()

        # Loss weights
        self.data_weight = 1.0
        self.physics_weights = {
            constraint.constraint_type: constraint.weight
            for constraint in physics_constraints
        }

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined data and physics loss

        Args:
            predictions: Model predictions
            targets: Ground truth targets (can be None for physics-only regions)
            inputs: Input coordinates and parameters
            model: PINN model for computing physics residuals

        Returns:
            total_loss: Combined loss
            loss_components: Dictionary of individual loss components
        """
        loss_components = {}
        total_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Data loss (only where targets are available)
        if targets is not None:
            # Mask for available data points
            data_mask = ~torch.isnan(targets).any(dim=1)

            if data_mask.sum() > 0:
                data_predictions = predictions[data_mask]
                data_targets = targets[data_mask]

                data_loss = self.mse_loss(data_predictions, data_targets)
                total_loss = total_loss + self.data_weight * data_loss
                loss_components["data_loss"] = data_loss.item()
            else:
                loss_components["data_loss"] = 0.0
        else:
            loss_components["data_loss"] = 0.0

        # Physics losses using the comprehensive constraint enforcer
        try:
            physics_loss_total, physics_loss_components = (
                self.constraint_enforcer.compute_physics_loss(
                    inputs, predictions, self.physics_constraints
                )
            )

            total_loss = total_loss + physics_loss_total
            loss_components.update(physics_loss_components)

        except Exception as e:
            logger.warning(f"Failed to compute physics losses: {e}")
            physics_loss_total = torch.tensor(
                0.0, device=predictions.device, requires_grad=True
            )
            loss_components["total_physics_loss"] = 0.0

        loss_components["total_loss"] = total_loss.item()

        return total_loss, loss_components

class PINNTrainer:
    """
    Training engine for Physics-Informed Neural Networks
    """

    def __init__(
        self,
        model: PINNNetwork,
        config: PINNModelConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Loss function
        self.loss_fn = PINNLoss(config.physics_constraints)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.training_state = PINNTrainingState(
            epoch=0,
            data_loss=float("in"),
            physics_loss=float("in"),
            total_loss=float("in"),
            learning_rate=config.optimization_config["learning_rate"],
            gradient_norm=0.0,
            physics_residual_max=float("in"),
            convergence_metric=float("inf"),
        )

        # Training history
        self.training_history = {
            "epoch": [],
            "data_loss": [],
            "physics_loss": [],
            "total_loss": [],
            "learning_rate": [],
            "gradient_norm": [],
        }

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        opt_config = self.config.optimization_config
        optimizer_type = OptimizationType(opt_config["optimizer"])

        if optimizer_type == OptimizationType.ADAM:
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config["learning_rate"],
                betas=opt_config.get("betas", (0.9, 0.999)),
                weight_decay=opt_config.get("weight_decay", 0.0),
            )
        elif optimizer_type == OptimizationType.ADAMW:
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config["learning_rate"],
                betas=opt_config.get("betas", (0.9, 0.999)),
                weight_decay=opt_config.get("weight_decay", 0.01),
            )
        elif optimizer_type == OptimizationType.SGD:
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config["learning_rate"],
                momentum=opt_config.get("momentum", 0.9),
                weight_decay=opt_config.get("weight_decay", 0.0),
            )
        elif optimizer_type == OptimizationType.LBFGS:
            return optim.LBFGS(
                self.model.parameters(),
                lr=opt_config["learning_rate"],
                max_iter=opt_config.get("max_iter", 20),
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        train_config = self.config.training_config

        if "scheduler" not in train_config:
            return None

        scheduler_type = train_config["scheduler"]

        if scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config.get("step_size", 1000),
                gamma=train_config.get("gamma", 0.9),
            )
        elif scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=train_config.get("gamma", 0.99)
            )
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=train_config.get("T_max", 1000)
            )
        else:
            return None

    def train_step(
        self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one training step

        Args:
            inputs: Input coordinates and parameters
            targets: Ground truth targets (optional)

        Returns:
            Dictionary of loss components
        """
        self.model.train()

        # Move data to device
        inputs = inputs.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)

        # Forward pass
        def closure():
            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            loss, loss_components = self.loss_fn(
                predictions, targets, inputs, self.model
            )
            loss.backward()
            return loss, loss_components

        # Optimization step
        if isinstance(self.optimizer, optim.LBFGS):
            loss, loss_components = self.optimizer.step(closure)
        else:
            loss, loss_components = closure()
            self.optimizer.step()

        # Compute gradient norm
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        gradient_norm = total_norm ** (1.0 / 2)

        # Update training state
        self.training_state.data_loss = loss_components.get("data_loss", 0.0)
        self.training_state.physics_loss = loss_components.get(
            "physics_loss_total", 0.0
        )
        self.training_state.total_loss = loss_components.get("total_loss", 0.0)
        self.training_state.gradient_norm = gradient_norm
        self.training_state.learning_rate = self.optimizer.param_groups[0]["lr"]

        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        return loss_components

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: Optional[torch.utils.data.DataLoader] = None,
        physics_points: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the PINN model

        Args:
            train_loader: Training data loader
            validation_loader: Validation data loader (optional)
            physics_points: Additional physics constraint points

        Returns:
            Training history dictionary
        """
        max_epochs = self.config.training_config["max_epochs"]

        logger.info(f"Starting PINN training for {max_epochs} epochs")

        for epoch in range(max_epochs):
            self.training_state.epoch = epoch

            # Training phase
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch[0]
                    targets = None

                loss_components = self.train_step(inputs, targets)
                epoch_losses.append(loss_components)

            # Physics constraint training (if physics points provided)
            if physics_points is not None:
                physics_loss_components = self.train_step(physics_points, None)
                epoch_losses.append(physics_loss_components)

            # Average losses for the epoch
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])

            # Update training history
            self.training_history["epoch"].append(epoch)
            self.training_history["data_loss"].append(avg_losses.get("data_loss", 0.0))
            self.training_history["physics_loss"].append(
                avg_losses.get("physics_loss_total", 0.0)
            )
            self.training_history["total_loss"].append(
                avg_losses.get("total_loss", 0.0)
            )
            self.training_history["learning_rate"].append(
                self.training_state.learning_rate
            )
            self.training_history["gradient_norm"].append(
                self.training_state.gradient_norm
            )

            # Validation phase
            if validation_loader is not None and epoch % 10 == 0:
                val_loss = self.validate(validation_loader)
                logger.info(
                    f"Epoch {epoch}: Train Loss = {avg_losses['total_loss']:.6f}, "
                    f"Val Loss = {val_loss:.6f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch}: Train Loss = {avg_losses['total_loss']:.6f}"
                )

            # Early stopping check
            if self._should_stop_early():
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info("PINN training completed")
        return self.training_history

    def validate(self, validation_loader: torch.utils.data.DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in validation_loader:
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch[0]
                    targets = None

                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)

                predictions = self.model(inputs)
                loss, _ = self.loss_fn(predictions, targets, inputs, self.model)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _should_stop_early(self) -> bool:
        """Check if training should stop early"""
        # Simple convergence check based on loss improvement
        if len(self.training_history["total_loss"]) < 100:
            return False

        recent_losses = self.training_history["total_loss"][-50:]
        older_losses = self.training_history["total_loss"][-100:-50]

        recent_avg = np.mean(recent_losses)
        older_avg = np.mean(older_losses)

        improvement = (older_avg - recent_avg) / older_avg

        return improvement < 1e-6  # Stop if less than 0.0001% improvement

    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "training_history": self.training_history,
                "training_state": self.training_state,
            },
            filepath,
        )

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history = checkpoint["training_history"]
        self.training_state = checkpoint["training_state"]

        logger.info(f"Model loaded from {filepath}")

class PINNEngine:
    """
    Main PINN Engine for managing multiple PINN models
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.models: Dict[str, PINNNetwork] = {}
        self.trainers: Dict[str, PINNTrainer] = {}
        self.temporal_models: Dict[str, MultiScaleTemporalPINN] = {}
        self.temporal_scales = create_default_temporal_scales()

        logger.info("PINN Engine initialized", device=str(device))

    def create_model(
        self,
        model_id: str,
        config: PINNModelConfig,
        geological_process: GeologicalProcess,
    ) -> PINNNetwork:
        """Create a new PINN model"""
        try:
            # Create standard PINN model
            model = PINNNetwork(config.network_architecture, geological_process)
            model.to(self.device)

            # Create trainer
            trainer = PINNTrainer(model, config, self.device)

            # Store model and trainer
            self.models[model_id] = model
            self.trainers[model_id] = trainer

            logger.info(f"Created PINN model: {model_id}")
            return model

        except Exception as e:
            logger.error(f"Failed to create PINN model {model_id}", error=str(e))
            raise

    def create_temporal_model(
        self,
        model_id: str,
        input_dim: int,
        output_dim: int,
        temporal_scales: Optional[List[TemporalScale]] = None,
        branch_hidden_dims: List[int] = [64, 128, 64],
        attention_heads: int = 8,
    ) -> MultiScaleTemporalPINN:
        """
        Create a multi-scale temporal PINN model

        Args:
            model_id: Unique identifier for the model
            input_dim: Input dimension
            output_dim: Output dimension
            temporal_scales: List of temporal scales (uses default if None)
            branch_hidden_dims: Hidden layer dimensions for each branch
            attention_heads: Number of attention heads

        Returns:
            Created multi-scale temporal PINN model
        """
        try:
            if temporal_scales is None:
                temporal_scales = self.temporal_scales

            # Create multi-scale temporal PINN
            temporal_model = MultiScaleTemporalPINN(
                input_dim=input_dim,
                output_dim=output_dim,
                temporal_scales=temporal_scales,
                branch_hidden_dims=branch_hidden_dims,
                attention_heads=attention_heads,
            )

            temporal_model.to(self.device)

            # Store model
            self.temporal_models[model_id] = temporal_model

            logger.info(f"Created multi-scale temporal PINN model: {model_id}")
            return temporal_model

        except Exception as e:
            logger.error(
                f"Failed to create temporal PINN model {model_id}", error=str(e)
            )
            raise

    def get_model(self, model_id: str) -> Optional[PINNNetwork]:
        """Get a PINN model by ID"""
        return self.models.get(model_id)

    def get_temporal_model(self, model_id: str) -> Optional[MultiScaleTemporalPINN]:
        """Get a temporal PINN model by ID"""
        return self.temporal_models.get(model_id)

    def get_trainer(self, model_id: str) -> Optional[PINNTrainer]:
        """Get a PINN trainer by ID"""
        return self.trainers.get(model_id)

    def list_models(self) -> List[str]:
        """List all available PINN models"""
        return list(self.models.keys())

    def list_temporal_models(self) -> List[str]:
        """List all available temporal PINN models"""
        return list(self.temporal_models.keys())

    def predict(self, model_id: str, inputs: torch.Tensor) -> torch.Tensor:
        """Make predictions using a PINN model"""
        model = self.models.get(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            predictions = model(inputs)

        return predictions

    def predict_temporal(
        self,
        model_id: str,
        inputs: torch.Tensor,
        time_coords: torch.Tensor,
        target_scale: Optional[TimeScale] = None,
    ) -> Dict[str, torch.Tensor]:
        """Make predictions using a temporal PINN model"""
        model = self.temporal_models.get(model_id)
        if model is None:
            raise ValueError(f"Temporal model {model_id} not found")

        model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            time_coords = time_coords.to(self.device)
            predictions = model(inputs, time_coords, target_scale)

        return predictions

    def predict_with_adaptive_stepping(
        self,
        model_id: str,
        inputs: torch.Tensor,
        time_start: float,
        time_end: float,
        process_type: TemporalGeologicalProcess,
        initial_solution: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Make predictions with adaptive time stepping"""
        model = self.temporal_models.get(model_id)
        if model is None:
            raise ValueError(f"Temporal model {model_id} not found")

        model.eval()
        return model.predict_with_adaptive_stepping(
            inputs, time_start, time_end, process_type, initial_solution
        )

    def interpolate_temporal_solution(
        self,
        model_id: str,
        known_times: torch.Tensor,
        known_solutions: torch.Tensor,
        target_times: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate temporal solution"""
        model = self.temporal_models.get(model_id)
        if model is None:
            raise ValueError(f"Temporal model {model_id} not found")

        return model.interpolate_temporal_solution(
            known_times, known_solutions, target_times
        )

    def extrapolate_temporal_solution(
        self,
        model_id: str,
        known_times: torch.Tensor,
        known_solutions: torch.Tensor,
        target_times: torch.Tensor,
        method: str = "polynomial",
    ) -> torch.Tensor:
        """Extrapolate temporal solution"""
        model = self.temporal_models.get(model_id)
        if model is None:
            raise ValueError(f"Temporal model {model_id} not found")

        return model.extrapolate_temporal_solution(
            known_times, known_solutions, target_times, method
        )

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a PINN model"""
        info = {}

        if model_id in self.models:
            model = self.models[model_id]
            trainer = self.trainers.get(model_id)

            info.update(
                {
                    "model_id": model_id,
                    "model_type": "standard_pinn",
                    "geological_process": model.geological_process.value,
                    "architecture": model.architecture.to_dict(),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "device": str(next(model.parameters()).device),
                    "training_state": trainer.training_state if trainer else None,
                }
            )

        if model_id in self.temporal_models:
            temporal_model = self.temporal_models[model_id]

            info.update(
                {
                    "model_id": model_id,
                    "model_type": "temporal_pinn",
                    "input_dim": temporal_model.input_dim,
                    "output_dim": temporal_model.output_dim,
                    "num_temporal_scales": temporal_model.num_scales,
                    "temporal_scales": [
                        scale.name.value for scale in temporal_model.temporal_scales
                    ],
                    "parameter_count": sum(
                        p.numel() for p in temporal_model.parameters()
                    ),
                    "device": str(next(temporal_model.parameters()).device),
                }
            )

        if not info:
            raise ValueError(f"Model {model_id} not found")

        return info

    def create_model(
        self,
        model_id: str,
        config: PINNModelConfig,
        geological_process: GeologicalProcess = GeologicalProcess.SEISMIC_WAVE_PROPAGATION,
    ) -> PINNNetwork:
        """
        Create a new PINN model

        Args:
            model_id: Unique identifier for the model
            config: PINN model configuration
            geological_process: Type of geological process

        Returns:
            Created PINN model
        """
        # Create network
        model = PINNNetwork(config.network_architecture, geological_process)

        # Create trainer
        trainer = PINNTrainer(model, config, self.device)

        # Store model and trainer
        self.models[model_id] = model
        self.trainers[model_id] = trainer
        self.configs[model_id] = config

        logger.info(f"Created PINN model: {model_id} for process: {geological_process}")

        return model

    def get_model(self, model_id: str) -> Optional[PINNNetwork]:
        """Get a model by ID"""
        return self.models.get(model_id)

    def get_trainer(self, model_id: str) -> Optional[PINNTrainer]:
        """Get a trainer by model ID"""
        return self.trainers.get(model_id)

    def list_models(self) -> List[str]:
        """List all available model IDs"""
        return list(self.models.keys())

    def remove_model(self, model_id: str):
        """Remove a model and its trainer"""
        if model_id in self.models:
            del self.models[model_id]
        if model_id in self.trainers:
            del self.trainers[model_id]
        if model_id in self.configs:
            del self.configs[model_id]

        logger.info(f"Removed PINN model: {model_id}")

    def predict(self, model_id: str, inputs: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using a trained model

        Args:
            model_id: Model identifier
            inputs: Input tensor

        Returns:
            Model predictions
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]
        model.eval()

        inputs = inputs.to(self.device)

        with torch.no_grad():
            predictions = model(inputs)

        return predictions

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]
        config = self.configs[model_id]
        trainer = self.trainers[model_id]

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "model_id": model_id,
            "geological_process": model.geological_process.value,
            "architecture": config.network_architecture.to_dict(),
            "physics_constraints": [
                pc.constraint_type.value for pc in config.physics_constraints
            ],
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "training_state": trainer.training_state,
            "device": str(self.device),
        }
