"""
Example demonstrating Physics-Informed Neural Network (PINN) Engine usage

This example shows how to:
    1. Create PINN models with different geological processes
2. Configure physics constraints
3. Train models with physics-informed loss
4. Make predictions with trained models
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from core.pinn_engine import GeologicalProcess, PINNEngine
from models.ai_models import (
    ActivationFunction,
    NetworkArchitecture,
    PhysicsConstraint,
    PhysicsConstraintType,
    PINNModelConfig,
)

def create_synthetic_heat_transfer_data(num_points: int = 1000):
    """
    Create synthetic data for heat transfer problem

    Args:
        num_points: Number of data points to generate

    Returns:
        inputs: [x, y, z, t] coordinates
        targets: Temperature values
    """
    # Generate random spatial and temporal coordinates
    x = torch.rand(num_points, 1) * 2 - 1  # [-1, 1]
    y = torch.rand(num_points, 1) * 2 - 1  # [-1, 1]
    z = torch.rand(num_points, 1) * 2 - 1  # [-1, 1]
    t = torch.rand(num_points, 1) * 1.0  # [0, 1]

    # Combine coordinates
    inputs = torch.cat([x, y, z, t], dim=1)
    inputs.requires_grad_(True)

    # Generate synthetic temperature field
    # Simple analytical solution: T = exp(-t) * sin(π*x) * sin(π*y) * sin(π*z)
    temperature = (
        torch.exp(-t)
        * torch.sin(np.pi * x)
        * torch.sin(np.pi * y)
        * torch.sin(np.pi * z)
    )

    return inputs, temperature

def create_heat_transfer_pinn_config():
    """Create PINN configuration for heat transfer problem"""

    # Network architecture
    architecture = NetworkArchitecture(
        layer_sizes=[4, 64, 64, 64, 1],  # Input: [x, y, z, t], Output: [T]
        activation_functions=[
            ActivationFunction.TANH,
            ActivationFunction.TANH,
            ActivationFunction.TANH,
            ActivationFunction.TANH,
        ],
        dropout_rates=[0.0, 0.1, 0.1, 0.0],
        batch_normalization=False,
        residual_connections=True,
    )

    # Physics constraints
    physics_constraints = [
        PhysicsConstraint(
            constraint_type=PhysicsConstraintType.HEAT_TRANSFER,
            equation_parameters={
                "thermal_conductivity": 1.0,
                "specific_heat": 1000.0,
                "density": 1000.0,
                "heat_source": 0.0,
            },
            weight=1.0,
            tolerance=1e-6,
        )
    ]

    # PINN configuration
    config = PINNModelConfig(
        model_id="heat_transfer_pinn",
        network_architecture=architecture,
        physics_constraints=physics_constraints,
        optimization_config={
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 1e-6,
        },
        training_config={
            "max_epochs": 100,
            "batch_size": 64,
            "scheduler": "step",
            "step_size": 50,
            "gamma": 0.5,
        },
        metadata={
            "description": "Heat transfer PINN for geological thermal processes",
            "geological_process": "thermal_diffusion",
            "spatial_dimensions": 3,
            "temporal": True,
        },
    )

    return config

def create_fluid_flow_pinn_config():
    """Create PINN configuration for fluid flow problem"""

    # Network architecture for fluid flow (outputs: u, v, w, p)
    architecture = NetworkArchitecture(
        layer_sizes=[4, 128, 128, 128, 4],  # Input: [x, y, z, t], Output: [u, v, w, p]
        activation_functions=[
            ActivationFunction.TANH,
            ActivationFunction.TANH,
            ActivationFunction.TANH,
            ActivationFunction.TANH,
        ],
        dropout_rates=[0.0, 0.1, 0.1, 0.0],
        batch_normalization=True,
        residual_connections=True,
    )

    # Physics constraints
    physics_constraints = [
        PhysicsConstraint(
            constraint_type=PhysicsConstraintType.NAVIER_STOKES,
            equation_parameters={"viscosity": 1e-3, "density": 1000.0},
            weight=1.0,
            tolerance=1e-6,
        )
    ]

    # PINN configuration
    config = PINNModelConfig(
        model_id="fluid_flow_pinn",
        network_architecture=architecture,
        physics_constraints=physics_constraints,
        optimization_config={
            "optimizer": "adamw",
            "learning_rate": 0.0005,
            "weight_decay": 1e-4,
        },
        training_config={
            "max_epochs": 200,
            "batch_size": 32,
            "scheduler": "cosine",
            "T_max": 200,
        },
        metadata={
            "description": "Navier-Stokes PINN for magma flow simulation",
            "geological_process": "magma_flow",
            "spatial_dimensions": 3,
            "temporal": True,
        },
    )

    return config

def demonstrate_pinn_training():
    """Demonstrate PINN model training"""

    print("=== Physics-Informed Neural Network (PINN) Engine Demo ===\n")

    # Initialize PINN engine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    engine = PINNEngine(device=device)

    # Create heat transfer model
    print("\n1. Creating Heat Transfer PINN Model...")
    heat_config = create_heat_transfer_pinn_config()
    heat_model = engine.create_model(
        "heat_transfer", heat_config, GeologicalProcess.HEAT_TRANSFER
    )

    # Create fluid flow model
    print("2. Creating Fluid Flow PINN Model...")
    fluid_config = create_fluid_flow_pinn_config()
    fluid_model = engine.create_model(
        "fluid_flow", fluid_config, GeologicalProcess.MAGMA_FLOW
    )

    # List created models
    print(f"\n3. Created models: {engine.list_models()}")

    # Get model information
    print("\n4. Model Information:")
    for model_id in engine.list_models():
        info = engine.get_model_info(model_id)
        print(f"   Model: {model_id}")
        print(f"   Process: {info['geological_process']}")
        print(f"   Parameters: {info['total_parameters']:,}")
        print(f"   Physics Constraints: {info['physics_constraints']}")
        print()

    # Generate training data for heat transfer
    print("5. Generating synthetic training data...")
    train_inputs, train_targets = create_synthetic_heat_transfer_data(1000)
    val_inputs, val_targets = create_synthetic_heat_transfer_data(200)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )

    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Train heat transfer model
    print("6. Training Heat Transfer PINN...")
    heat_trainer = engine.get_trainer("heat_transfer")

    # Reduce epochs for demo
    heat_trainer.config.training_config["max_epochs"] = 20

    training_history = heat_trainer.train(train_loader, val_loader)

    print("   Training completed!")
    print(f"   Final loss: {training_history['total_loss'][-1]:.6f}")

    # Make predictions
    print("\n7. Making predictions...")
    test_inputs = torch.randn(10, 4)  # Random test points

    heat_predictions = engine.predict("heat_transfer", test_inputs)
    print(f"   Heat transfer predictions shape: {heat_predictions.shape}")

    # Test physics constraint evaluation
    print("\n8. Testing physics constraints...")
    heat_trainer = engine.get_trainer("heat_transfer")

    # Create physics points (no data, only physics constraints)
    physics_points = torch.randn(100, 4, requires_grad=True)
    physics_loss = heat_trainer.train_step(physics_points, None)

    print(f"   Physics loss components: {list(physics_loss.keys())}")
    print(f"   Physics loss value: {physics_loss.get('physics_loss_total', 0.0):.6f}")

    # Demonstrate model saving/loading
    print("\n9. Saving and loading models...")
    model_path = "heat_transfer_model.pth"
    heat_trainer.save_model(model_path)
    print(f"   Model saved to {model_path}")

    # Create new trainer and load model
    new_trainer = engine.get_trainer("heat_transfer")
    new_trainer.load_model(model_path)
    print("   Model loaded successfully")

    # Verify loaded model works
    loaded_predictions = engine.predict("heat_transfer", test_inputs)
    prediction_diff = torch.abs(heat_predictions - loaded_predictions).max()
    print(f"   Prediction difference after loading: {prediction_diff:.8f}")

    print("\n=== Demo completed successfully! ===")

def plot_training_history(history):
    """Plot training history"""
    try:
        import matplotlib.pyplot as plt

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Total loss
        axes[0, 0].plot(history["epoch"], history["total_loss"])
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)

        # Data loss
        axes[0, 1].plot(history["epoch"], history["data_loss"])
        axes[0, 1].set_title("Data Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True)

        # Physics loss
        axes[1, 0].plot(history["epoch"], history["physics_loss"])
        axes[1, 0].set_title("Physics Loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True)

        # Learning rate
        axes[1, 1].plot(history["epoch"], history["learning_rate"])
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig("pinn_training_history.png", dpi=300, bbox_inches="tight")
        print("Training history plot saved as 'pinn_training_history.png'")

    except ImportError:
        print("Matplotlib not available, skipping plot generation")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run demonstration
    demonstrate_pinn_training()
