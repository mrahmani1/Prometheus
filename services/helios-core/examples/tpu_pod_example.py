"""
Example demonstrating TPU Pod integration for massive parallel geological modeling.

This example shows how to use TPU Pods for distributed training and inference
of Physics-Informed Neural Networks for geological simulations.
"""

import asyncio
import numpy as np
import logging
from typing import Dict, Any

from ..core.tpu_pod_manager import (
    TPUPodManager, TPUPodConfig, create_tpu_pod_manager
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_geological_data(num_samples: int = 10000) -> tuple:
    """
    Generate synthetic geological data for TPU training demonstration.
    
    Returns:
        tuple: (input_data, target_data) for training
    """
    # Spatial coordinates (x, y, z) + time
    spatial_coords = np.random.uniform(-1000, 1000, (num_samples, 3))  # km
    time_coords = np.random.uniform(0, 100, (num_samples, 1))  # years
    
    # Additional physical parameters (pressure, temperature gradients, etc.)
    physical_params = np.random.uniform(0, 1, (num_samples, 2))
    
    # Combine all inputs
    input_data = np.concatenate([
        spatial_coords, time_coords, physical_params
    ], axis=1).astype(np.float32)
    
    # Generate synthetic targets based on simplified geological relationships
    x, y, z, t = spatial_coords[:, 0], spatial_coords[:, 1], spatial_coords[:, 2], time_coords[:, 0]
    
    # Synthetic temperature field (simplified thermal model)
    temperature = 300 + 0.03 * np.abs(z) + 50 * np.sin(0.001 * x) * np.cos(0.001 * y)
    
    # Synthetic pressure field
    pressure = 1e5 + 2700 * 9.81 * np.abs(z)  # Hydrostatic pressure
    
    # Synthetic displacement field (simplified deformation)
    displacement_x = 0.001 * np.sin(0.0001 * x * t)
    displacement_y = 0.001 * np.cos(0.0001 * y * t)
    
    # Combine targets
    target_data = np.column_stack([
        temperature, pressure, displacement_x, displacement_y
    ]).astype(np.float32)
    
    return input_data, target_data


async def demonstrate_tpu_pod_initialization():
    """Demonstrate TPU Pod initialization and configuration."""
    print("\n=== TPU Pod Initialization Demo ===")
    
    # Configure TPU Pod
    config = TPUPodConfig(
        pod_size=8,  # 8 TPU cores
        mesh_shape=(2, 4),  # 2x4 mesh topology
        model_parallel_submesh=(1, 4),  # Model parallelism across 4 cores
        data_parallel_submesh=(2, 1),  # Data parallelism across 2 cores
        precision="bfloat16",  # TPU-optimized precision
        batch_size_per_device=64
    )
    
    try:
        # Initialize TPU Pod manager
        tpu_manager = create_tpu_pod_manager(config)
        
        print(f"✓ TPU Pod initialized successfully")
        print(f"  - Device count: {tpu_manager.mesh.size if tpu_manager.mesh else 'N/A'}")
        print(f"  - Mesh shape: {config.mesh_shape}")
        print(f"  - Status: {tpu_manager.status.value}")
        
        return tpu_manager
        
    except Exception as e:
        print(f"✗ TPU Pod initialization failed: {e}")
        print("  Note: This requires actual TPU hardware or TPU emulation")
        return None


async def demonstrate_model_creation(tpu_manager: TPUPodManager):
    """Demonstrate geological PINN model creation for TPU."""
    print("\n=== Geological PINN Model Creation Demo ===")
    
    if tpu_manager is None:
        print("✗ Skipping model creation (no TPU manager)")
        return
    
    try:
        # Model configuration for geological simulation
        model_config = {
            'features': [512, 512, 512, 256, 4],  # Large network for complex geology
            'physics_constraints': [
                'navier_stokes',      # Fluid flow in porous media
                'heat_transfer',      # Thermal processes
                'elastic_deformation' # Crustal deformation
            ],
            'activation': 'tanh',
            'input_dim': 6,  # x, y, z, t, pressure_grad, temp_grad
            'learning_rate': 1e-3,
            'weight_decay': 1e-4
        }
        
        # Create model
        model = tpu_manager.create_geological_model(model_config)
        
        print(f"✓ Geological PINN model created successfully")
        print(f"  - Network architecture: {model_config['features']}")
        print(f"  - Physics constraints: {model_config['physics_constraints']}")
        print(f"  - Input dimension: {model_config['input_dim']}")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")


async def demonstrate_dataset_sharding(tpu_manager: TPUPodManager):
    """Demonstrate automatic dataset sharding across TPU cores."""
    print("\n=== Dataset Sharding Demo ===")
    
    if tpu_manager is None:
        print("✗ Skipping dataset sharding (no TPU manager)")
        return None, None
    
    try:
        # Generate synthetic geological data
        print("Generating synthetic geological dataset...")
        input_data, target_data = generate_synthetic_geological_data(num_samples=50000)
        
        print(f"✓ Generated dataset:")
        print(f"  - Input shape: {input_data.shape}")
        print(f"  - Target shape: {target_data.shape}")
        print(f"  - Data type: {input_data.dtype}")
        
        # Setup dataset sharding
        tpu_manager.setup_dataset_sharding(input_data, target_data)
        
        print(f"✓ Dataset sharding configured")
        print(f"  - Batch size per device: {tpu_manager.config.batch_size_per_device}")
        print(f"  - Total batch size: {tpu_manager.config.batch_size_per_device * (tpu_manager.mesh.size if tpu_manager.mesh else 1)}")
        
        return input_data, target_data
        
    except Exception as e:
        print(f"✗ Dataset sharding failed: {e}")
        return None, None


async def demonstrate_distributed_training(tpu_manager: TPUPodManager):
    """Demonstrate distributed training across TPU Pod."""
    print("\n=== Distributed Training Demo ===")
    
    if tpu_manager is None:
        print("✗ Skipping distributed training (no TPU manager)")
        return
    
    try:
        # Run distributed training
        print("Starting distributed training across TPU Pod...")
        
        training_metrics = tpu_manager.train_distributed(
            num_steps=500,
            log_interval=50
        )
        
        print(f"✓ Distributed training completed")
        print(f"  - Training steps: 500")
        print(f"  - Metrics recorded: {len(training_metrics)}")
        
        if training_metrics:
            final_metrics = training_metrics[-1]
            print(f"  - Final loss: {final_metrics.loss:.6f}")
            print(f"  - Final physics loss: {final_metrics.physics_loss:.6f}")
            print(f"  - Final data loss: {final_metrics.data_loss:.6f}")
            print(f"  - Throughput: {final_metrics.throughput_samples_per_sec:.1f} samples/sec")
            print(f"  - Memory usage: {final_metrics.memory_usage_gb:.1f} GB")
            print(f"  - FLOPS utilization: {final_metrics.flops_utilization:.1f}%")
        
    except Exception as e:
        print(f"✗ Distributed training failed: {e}")


async def demonstrate_distributed_inference(tpu_manager: TPUPodManager, test_data: np.ndarray):
    """Demonstrate distributed inference across TPU Pod."""
    print("\n=== Distributed Inference Demo ===")
    
    if tpu_manager is None or test_data is None:
        print("✗ Skipping distributed inference (no TPU manager or test data)")
        return
    
    try:
        # Use subset of data for inference
        inference_data = test_data[:1000]  # First 1000 samples
        
        print(f"Running distributed inference on {inference_data.shape[0]} samples...")
        
        # Run distributed inference
        predictions = tpu_manager.inference_distributed(inference_data)
        
        print(f"✓ Distributed inference completed")
        print(f"  - Input shape: {inference_data.shape}")
        print(f"  - Output shape: {predictions.shape}")
        print(f"  - Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        # Analyze predictions
        if predictions.shape[1] >= 4:
            print(f"  - Temperature predictions: [{predictions[:, 0].min():.1f}, {predictions[:, 0].max():.1f}] K")
            print(f"  - Pressure predictions: [{predictions[:, 1].min():.0f}, {predictions[:, 1].max():.0f}] Pa")
            print(f"  - Displacement X: [{predictions[:, 2].min():.6f}, {predictions[:, 2].max():.6f}] m")
            print(f"  - Displacement Y: [{predictions[:, 3].min():.6f}, {predictions[:, 3].max():.6f}] m")
        
    except Exception as e:
        print(f"✗ Distributed inference failed: {e}")


async def demonstrate_model_checkpointing(tpu_manager: TPUPodManager):
    """Demonstrate model checkpointing for TPU."""
    print("\n=== Model Checkpointing Demo ===")
    
    if tpu_manager is None:
        print("✗ Skipping model checkpointing (no TPU manager)")
        return
    
    try:
        checkpoint_path = "tpu_geological_model_checkpoint.pkl"
        
        # Save model checkpoint
        tpu_manager.save_model(checkpoint_path)
        print(f"✓ Model checkpoint saved to {checkpoint_path}")
        
        # Load model checkpoint (simulate restart)
        tpu_manager.load_model(checkpoint_path)
        print(f"✓ Model checkpoint loaded from {checkpoint_path}")
        
    except Exception as e:
        print(f"✗ Model checkpointing failed: {e}")


async def demonstrate_performance_monitoring(tpu_manager: TPUPodManager):
    """Demonstrate TPU performance monitoring."""
    print("\n=== Performance Monitoring Demo ===")
    
    if tpu_manager is None:
        print("✗ Skipping performance monitoring (no TPU manager)")
        return
    
    try:
        # Get performance metrics
        metrics = tpu_manager.get_performance_metrics()
        
        print(f"✓ TPU Pod Performance Metrics:")
        print(f"  - Status: {metrics['status']}")
        print(f"  - Device count: {metrics['device_count']}")
        print(f"  - Mesh shape: {metrics['mesh_shape']}")
        print(f"  - Memory usage: {metrics['memory_usage_gb']:.1f} GB")
        print(f"  - FLOPS utilization: {metrics['flops_utilization']:.1f}%")
        print(f"  - Recent training metrics: {len(metrics['training_metrics'])} entries")
        
    except Exception as e:
        print(f"✗ Performance monitoring failed: {e}")


async def main():
    """Main demonstration function."""
    print("🚀 TPU Pod Integration for Geological Modeling Demo")
    print("=" * 60)
    
    # Initialize TPU Pod
    tpu_manager = await demonstrate_tpu_pod_initialization()
    
    if tpu_manager is not None:
        # Create geological model
        await demonstrate_model_creation(tpu_manager)
        
        # Setup dataset sharding
        input_data, target_data = await demonstrate_dataset_sharding(tpu_manager)
        
        # Run distributed training
        await demonstrate_distributed_training(tpu_manager)
        
        # Run distributed inference
        await demonstrate_distributed_inference(tpu_manager, input_data)
        
        # Demonstrate checkpointing
        await demonstrate_model_checkpointing(tpu_manager)
        
        # Monitor performance
        await demonstrate_performance_monitoring(tpu_manager)
        
        # Cleanup
        print("\n=== Cleanup ===")
        tpu_manager.shutdown()
        print("✓ TPU Pod shutdown completed")
    
    else:
        print("\n⚠️  TPU Pod demo completed with limitations")
        print("   This demo requires actual TPU hardware or TPU emulation")
        print("   The implementation is ready for deployment on Google Cloud TPU Pods")
    
    print("\n🎉 TPU Pod Integration Demo Complete!")


if __name__ == "__main__":
    asyncio.run(main())