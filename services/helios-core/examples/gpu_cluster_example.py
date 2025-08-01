"""
Example demonstrating GPU cluster management for distributed geological modeling.

This example shows how to use GPU clusters for distributed training
of geological AI models with automatic memory optimization and scaling.
"""

import asyncio
import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..core.gpu_cluster_manager import (
    ClusterConfig,
    GPUClusterManager,
    create_gpu_cluster_manager,
)
from ..core.gpu_cluster_manager import numpy as np
from ..core.gpu_cluster_manager import torch.nn as nn
from ..core.gpu_cluster_manager import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_training_data(num_samples: int = 50000) -> tuple:
    """
    Generate synthetic geological training data for GPU cluster demonstration.

    Returns:
        tuple: (input_data, target_data) for training
    """
    # Multi-dimensional geological features
    # Spatial coordinates (x, y, z)
    spatial_coords = np.random.uniform(-2000, 2000, (num_samples, 3))  # km

    # Temporal coordinates
    time_coords = np.random.uniform(0, 1000, (num_samples, 1))  # years

    # Physical parameters (pressure, temperature, density, etc.)
    physical_params = np.random.uniform(0, 1, (num_samples, 8))

    # Combine all inputs
    input_data = np.concatenate(
        [spatial_coords, time_coords, physical_params], axis=1
    ).astype(np.float32)

    # Generate complex synthetic targets based on geological relationships
    x, y, z, t = (
        spatial_coords[:, 0],
        spatial_coords[:, 1],
        spatial_coords[:, 2],
        time_coords[:, 0],
    )

    # Multi-physics targets
    # Temperature field (3D thermal model)
    temperature = (
        300
        + 0.025 * np.abs(z)
        + 100 * np.sin(0.0005 * x) * np.cos(0.0005 * y) * np.exp(-0.001 * t)
    )

    # Pressure field (hydrostatic + tectonic)
    pressure = 1e5 + 2700 * 9.81 * np.abs(z) + 1e6 * np.sin(0.0001 * x * y)

    # Velocity field (fluid flow)
    velocity_x = 0.01 * np.sin(0.0002 * x * t) * np.cos(0.0001 * z)
    velocity_y = 0.01 * np.cos(0.0002 * y * t) * np.sin(0.0001 * z)
    velocity_z = 0.005 * np.sin(0.0001 * (x + y) * t)

    # Stress field (tectonic stress)
    stress_xx = 1e7 * np.sin(0.0001 * x) * np.cos(0.0001 * y)
    stress_yy = 1e7 * np.cos(0.0001 * x) * np.sin(0.0001 * y)
    stress_zz = 1e7 * np.sin(0.0001 * z)

    # Combine targets
    target_data = np.column_stack(
        [
            temperature,
            pressure,
            velocity_x,
            velocity_y,
            velocity_z,
            stress_xx,
            stress_yy,
            stress_zz,
        ]
    ).astype(np.float32)

    return input_data, target_data

async def demonstrate_cluster_initialization():
    """Demonstrate GPU cluster initialization and configuration."""
    print("\n=== GPU Cluster Initialization Demo ===")

    # Configure GPU cluster
    config = ClusterConfig(
        cluster_name="geological-modeling-cluster",
        max_nodes=4,
        gpus_per_node=8,
        memory_threshold_gb=4.0,
        utilization_threshold=85.0,
        temperature_threshold=80.0,
        auto_scaling=True,
        mixed_precision=True,
        batch_size_per_gpu=64,
        distributed_backend="nccl",
    )

    try:
        # Initialize GPU cluster manager
        cluster_manager = create_gpu_cluster_manager(config)

        print(f"✓ GPU cluster '{config.cluster_name}' initialized successfully")
        print(f"  - Total GPUs: {len(cluster_manager.gpu_devices)}")
        print(f"  - Cluster status: {cluster_manager.status.value}")
        print(f"  - Auto-scaling: {'Enabled' if config.auto_scaling else 'Disabled'}")
        print(
            f"  - Mixed precision: {'Enabled' if config.mixed_precision else 'Disabled'}"
        )

        return cluster_manager

    except Exception as e:
        print(f"✗ GPU cluster initialization failed: {e}")
        print("  Note: This requires CUDA-capable GPUs")
        return None

async def demonstrate_cluster_status(cluster_manager: GPUClusterManager):
    """Demonstrate cluster status monitoring."""
    print("\n=== Cluster Status Monitoring Demo ===")

    if cluster_manager is None:
        print("✗ Skipping cluster status (no cluster manager)")
        return

    try:
        # Get cluster status
        status_info = cluster_manager.get_cluster_status()

        print(f"✓ Cluster Status: {status_info['status']}")
        print(f"  - Total GPUs: {status_info['total_gpus']}")
        print(f"  - Active GPUs: {status_info['active_gpus']}")
        print(
            f"  - Total Memory: {status_info['cluster_metrics']['total_memory_gb']:.1f} GB"
        )
        print(
            f"  - Free Memory: {status_info['cluster_metrics']['free_memory_gb']:.1f} GB"
        )
        print(
            f"  - Average Utilization: {status_info['cluster_metrics']['average_gpu_utilization']:.1f}%"
        )
        print(
            f"  - Average Temperature: {status_info['cluster_metrics']['average_temperature']:.1f}°C"
        )

        print("\n  GPU Devices:")
        for gpu in status_info["gpu_devices"]:
            print(f"    GPU {gpu['device_id']}: {gpu['name']} ({gpu['gpu_type']})")
            print(
                f"      Memory: {gpu['memory_used_gb']:.1f}/{gpu['memory_total_gb']:.1f} GB"
            )
            print(f"      Utilization: {gpu['utilization_percent']:.1f}%")
            print(f"      Temperature: {gpu['temperature_celsius']:.1f}°C")
            print(f"      Available: {'Yes' if gpu['is_available'] else 'No'}")

    except Exception as e:
        print(f"✗ Cluster status monitoring failed: {e}")

async def demonstrate_job_submission(cluster_manager: GPUClusterManager):
    """Demonstrate training job submission and management."""
    print("\n=== Training Job Submission Demo ===")

    if cluster_manager is None:
        print("✗ Skipping job submission (no cluster manager)")
        return []

    try:
        # Submit multiple training jobs with different configurations
        job_configs = [
            {
                "model_name": "geological_pinn_large",
                "dataset_path": "/data/geological/large_dataset",
                "num_epochs": 200,
                "learning_rate": 1e-3,
                "batch_size": 128,
                "num_gpus": 4,
                "priority": 1,
            },
            {
                "model_name": "seismic_cnn_detector",
                "dataset_path": "/data/seismic/waveforms",
                "num_epochs": 100,
                "learning_rate": 5e-4,
                "batch_size": 64,
                "num_gpus": 2,
                "priority": 2,
            },
            {
                "model_name": "volcanic_risk_predictor",
                "dataset_path": "/data/volcanic/monitoring",
                "num_epochs": 150,
                "learning_rate": 2e-3,
                "batch_size": 32,
                "num_gpus": 1,
                "priority": 1,
            },
        ]

        submitted_jobs = []

        for i, job_config in enumerate(job_configs):
            job_id = cluster_manager.submit_training_job(job_config)
            submitted_jobs.append(job_id)

            print(f"✓ Submitted job {i+1}: {job_id}")
            print(f"  - Model: {job_config['model_name']}")
            print(f"  - GPUs required: {job_config['num_gpus']}")
            print(f"  - Priority: {job_config['priority']}")
            print(f"  - Epochs: {job_config['num_epochs']}")

        print(f"\n  Total jobs submitted: {len(submitted_jobs)}")
        print(f"  Jobs in queue: {len(cluster_manager.job_queue)}")

        return submitted_jobs

    except Exception as e:
        print(f"✗ Job submission failed: {e}")
        return []

async def demonstrate_distributed_training(
    cluster_manager: GPUClusterManager, job_ids: list
):
    """Demonstrate distributed training execution."""
    print("\n=== Distributed Training Demo ===")

    if cluster_manager is None or not job_ids:
        print("✗ Skipping distributed training (no cluster manager or jobs)")
        return

    try:
        # Start training for the first job
        job_id = job_ids[0]

        print(f"Starting distributed training for job: {job_id}")

        success = cluster_manager.start_distributed_training(job_id)

        if success:
            print(f"✓ Distributed training started for job {job_id}")

            # Get job details
            if job_id in cluster_manager.active_jobs:
                job_info = cluster_manager.active_jobs[job_id]
                allocated_gpus = job_info["allocated_gpus"]
                print(f"  - Allocated GPUs: {allocated_gpus}")
                print(f"  - Training status: {job_info['job'].status}")

            # Simulate monitoring training progress
            print("\n  Monitoring training progress...")
            for i in range(5):  # Monitor for 5 iterations
                await asyncio.sleep(1)  # Wait 1 second

                progress_info = cluster_manager.monitor_training_progress(job_id)
                if progress_info["status"] != "not_found":
                    print(
                        f"    Step {i+1}: Progress={progress_info['progress']:.1%}, "
                        f"Loss={progress_info.get('current_loss', 'N/A')}"
                    )

                    # Show GPU metrics
                    if "gpu_metrics" in progress_info:
                        for gpu_id, metrics in progress_info["gpu_metrics"].items():
                            print(
                                f"      GPU {gpu_id}: "
                                f"Memory={metrics['memory_used_gb']:.1f}GB, "
                                f"Util={metrics['utilization_percent']:.1f}%"
                            )
        else:
            print(f"✗ Failed to start distributed training for job {job_id}")

    except Exception as e:
        print(f"✗ Distributed training failed: {e}")

async def demonstrate_batch_size_optimization(
    cluster_manager: GPUClusterManager, job_ids: list
):
    """Demonstrate automatic batch size optimization."""
    print("\n=== Batch Size Optimization Demo ===")

    if cluster_manager is None or not job_ids:
        print("✗ Skipping batch size optimization (no cluster manager or jobs)")
        return

    try:
        # Optimize batch sizes for active jobs
        for job_id in job_ids:
            if job_id in cluster_manager.active_jobs:
                print(f"Optimizing batch sizes for job: {job_id}")

                optimized_batch_sizes = cluster_manager.optimize_batch_sizes(job_id)

                if optimized_batch_sizes:
                    print("✓ Batch size optimization completed:")
                    total_batch_size = 0
                    for gpu_id, batch_size in optimized_batch_sizes.items():
                        print(f"  - GPU {gpu_id}: {batch_size} samples/batch")
                        total_batch_size += batch_size

                    print(f"  - Total effective batch size: {total_batch_size}")
                    print("  - Memory optimization applied")
                else:
                    print(f"✗ Batch size optimization failed for job {job_id}")

    except Exception as e:
        print(f"✗ Batch size optimization failed: {e}")

async def demonstrate_performance_monitoring(cluster_manager: GPUClusterManager):
    """Demonstrate cluster performance monitoring."""
    print("\n=== Performance Monitoring Demo ===")

    if cluster_manager is None:
        print("✗ Skipping performance monitoring (no cluster manager)")
        return

    try:
        # Get performance metrics
        metrics = cluster_manager.get_performance_metrics()

        print("✓ Cluster Performance Metrics:")
        print(f"  - Timestamp: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - Total GPUs: {metrics.total_gpus}")
        print(f"  - Active GPUs: {metrics.active_gpus}")
        print(
            f"  - Memory Usage: {metrics.used_memory_gb:.1f}/{metrics.total_memory_gb:.1f} GB "
            f"({(metrics.used_memory_gb/metrics.total_memory_gb)*100:.1f}%)"
        )
        print(f"  - Average Utilization: {metrics.average_utilization:.1f}%")
        print(f"  - Average Temperature: {metrics.average_temperature:.1f}°C")
        print(f"  - Total Power Usage: {metrics.total_power_usage_watts:.0f} W")
        print(f"  - Throughput: {metrics.throughput_samples_per_sec:.1f} samples/sec")
        print(f"  - Active Jobs: {metrics.active_jobs}")
        print(f"  - Queued Jobs: {metrics.queued_jobs}")

        # Show metrics history
        if len(cluster_manager.metrics_history) > 1:
            print(
                f"\n  Metrics History ({len(cluster_manager.metrics_history)} entries):"
            )
            recent_metrics = cluster_manager.metrics_history[-3:]  # Last 3 entries
            for i, hist_metrics in enumerate(recent_metrics):
                print(
                    f"    {i+1}. {hist_metrics.timestamp.strftime('%H:%M:%S')}: "
                    f"Util={hist_metrics.average_utilization:.1f}%, "
                    f"Mem={hist_metrics.used_memory_gb:.1f}GB"
                )

    except Exception as e:
        print(f"✗ Performance monitoring failed: {e}")

async def demonstrate_auto_scaling(cluster_manager: GPUClusterManager):
    """Demonstrate automatic cluster scaling."""
    print("\n=== Auto-Scaling Demo ===")

    if cluster_manager is None:
        print("✗ Skipping auto-scaling (no cluster manager)")
        return

    try:
        print("Evaluating auto-scaling conditions...")

        # Get current metrics
        metrics = cluster_manager.get_performance_metrics()

        print(f"  - Current utilization: {metrics.average_utilization:.1f}%")
        print(f"  - Active jobs: {metrics.active_jobs}")
        print(f"  - Queued jobs: {metrics.queued_jobs}")

        # Trigger auto-scaling evaluation
        scaling_performed = cluster_manager.auto_scale_cluster()

        if scaling_performed:
            print("✓ Auto-scaling triggered")
            if metrics.average_utilization > 80 and metrics.queued_jobs > 2:
                print("  - Recommendation: Scale up (high utilization + queue backlog)")
            elif metrics.average_utilization < 20 and metrics.queued_jobs == 0:
                print("  - Recommendation: Scale down (low utilization + no queue)")
        else:
            print("✓ Auto-scaling evaluated - no scaling needed")
            print("  - Current resource allocation is optimal")

    except Exception as e:
        print(f"✗ Auto-scaling failed: {e}")

async def demonstrate_job_management(cluster_manager: GPUClusterManager):
    """Demonstrate comprehensive job management."""
    print("\n=== Job Management Demo ===")

    if cluster_manager is None:
        print("✗ Skipping job management (no cluster manager)")
        return

    try:
        # List all jobs
        print("Current training jobs:")

        if not cluster_manager.training_jobs:
            print("  - No jobs found")
            return

        for job_id, job in cluster_manager.training_jobs.items():
            status_icon = (
                "🟢"
                if job.status == "running"
                else "🟡" if job.status == "queued" else "🔴"
            )
            print(f"  {status_icon} {job_id}:")
            print(f"    - Model: {job.model_name}")
            print(f"    - Status: {job.status}")
            print(f"    - Progress: {job.progress:.1%}")
            print(f"    - Loss: {job.current_loss:.6f}")
            print(f"    - GPUs: {job.num_gpus}")
            print(f"    - Priority: {job.priority}")
            print(f"    - Created: {job.created_at.strftime('%H:%M:%S')}")
            if job.started_at:
                print(f"    - Started: {job.started_at.strftime('%H:%M:%S')}")
            if job.completed_at:
                print(f"    - Completed: {job.completed_at.strftime('%H:%M:%S')}")

        # Show queue status
        print("\n  Queue Status:")
        print(f"    - Jobs in queue: {len(cluster_manager.job_queue)}")
        print(f"    - Active jobs: {len(cluster_manager.active_jobs)}")
        print(f"    - Total jobs: {len(cluster_manager.training_jobs)}")

    except Exception as e:
        print(f"✗ Job management failed: {e}")

async def main():
    """Main demonstration function."""
    print("🚀 GPU Cluster Management for Geological Modeling Demo")
    print("=" * 65)

    # Initialize GPU cluster
    cluster_manager = await demonstrate_cluster_initialization()

    if cluster_manager is not None:
        # Monitor cluster status
        await demonstrate_cluster_status(cluster_manager)

        # Submit training jobs
        job_ids = await demonstrate_job_submission(cluster_manager)

        # Start distributed training
        await demonstrate_distributed_training(cluster_manager, job_ids)

        # Optimize batch sizes
        await demonstrate_batch_size_optimization(cluster_manager, job_ids)

        # Monitor performance
        await demonstrate_performance_monitoring(cluster_manager)

        # Demonstrate auto-scaling
        await demonstrate_auto_scaling(cluster_manager)

        # Manage jobs
        await demonstrate_job_management(cluster_manager)

        # Cleanup
        print("\n=== Cleanup ===")
        cluster_manager.shutdown_cluster()
        print("✓ GPU cluster shutdown completed")

    else:
        print("\n⚠️  GPU cluster demo completed with limitations")
        print("   This demo requires CUDA-capable GPUs")
        print("   The implementation is ready for deployment on GPU clusters")

    print("\n🎉 GPU Cluster Management Demo Complete!")

if __name__ == "__main__":
    asyncio.run(main())
