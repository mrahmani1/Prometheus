"""
Edge AI Optimization Example

This example demonstrates how to use the edge AI optimization pipeline
to create optimized models for different edge deployment scenarios.

Requirements addressed: 10.1, 10.5
"""

import asyncio
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from core.automated_edge_generation import AutomatedEdgeGenerator

# Import edge optimization modules
from core.edge_ai_optimizer import (
    EdgeAIOptimizer,
    HardwareConstraints,
    OptimizationConfig,
    create_edge_server_constraints,
    create_embedded_constraints,
    create_mobile_constraints,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_geological_model():
    """
    Create a sample geological neural network model
    This represents a simplified version of a geological process model
    """

    class GeologicalProcessModel(nn.Module):
        def __init__(self, input_features=20, hidden_size=128, num_classes=5):
            super().__init__()

            # Feature extraction layers
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_features, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size // 2),
                nn.Dropout(0.3),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size // 4),
            )

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size // 4, num_classes), nn.Softmax(dim=1)
            )

        def forward(self, x):
            features = self.feature_extractor(x)
            return self.classifier(features)

    return GeologicalProcessModel()

def create_sample_seismic_cnn():
    """
    Create a sample seismic CNN model
    This represents a simplified seismic event detection model
    """

    class SeismicCNN(nn.Module):
        def __init__(self):
            super().__init__()

            # Convolutional layers for seismic waveform processing
            self.conv_layers = nn.Sequential(
                nn.Conv1d(3, 64, kernel_size=7, padding=3),  # 3 channels (X, Y, Z)
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(2),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.AdaptiveAvgPool1d(1),
            )

            # Classification layers
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 4),  # 4 classes: noise, earthquake, explosion, volcanic
            )

        def forward(self, x):
            features = self.conv_layers(x)
            return self.classifier(features)

    return SeismicCNN()

def generate_synthetic_geological_data(num_samples=1000, num_features=20):
    """Generate synthetic geological data for testing"""
    # Simulate geological measurements (temperature, pressure, chemical composition, etc.)
    data = torch.randn(num_samples, num_features)

    # Add some geological patterns
    # Temperature correlation
    data[:, 0] = torch.abs(data[:, 0]) * 100 + 273  # Temperature in Kelvin

    # Pressure correlation with depth
    data[:, 1] = torch.abs(data[:, 1]) * 1000 + 101325  # Pressure in Pa

    # Chemical composition (normalized)
    data[:, 2:10] = torch.softmax(data[:, 2:10], dim=1)

    # Generate labels based on patterns
    labels = torch.zeros(num_samples, dtype=torch.long)

    # Simple classification based on temperature and pressure
    high_temp = data[:, 0] > 400
    high_pressure = data[:, 1] > 500000

    labels[high_temp & high_pressure] = 0  # Volcanic activity
    labels[high_temp & ~high_pressure] = 1  # Hydrothermal activity
    labels[~high_temp & high_pressure] = 2  # Deep crustal process
    labels[~high_temp & ~high_pressure] = 3  # Surface process

    # Add some random volcanic events
    volcanic_indices = torch.randperm(num_samples)[: num_samples // 10]
    labels[volcanic_indices] = 4  # Eruption precursor

    return data, labels

def generate_synthetic_seismic_data(num_samples=500, sequence_length=1000):
    """Generate synthetic seismic waveform data"""
    # Create synthetic seismic waveforms
    data = torch.randn(num_samples, 3, sequence_length)  # 3 components (X, Y, Z)

    # Add seismic patterns
    for i in range(num_samples):
        # Add P-wave and S-wave arrivals for earthquakes
        if i % 4 == 0:  # Earthquake
            p_arrival = np.random.randint(100, 300)
            s_arrival = p_arrival + np.random.randint(50, 150)

            # P-wave
            data[i, :, p_arrival : p_arrival + 50] += torch.randn(3, 50) * 2
            # S-wave
            data[i, :, s_arrival : s_arrival + 100] += torch.randn(3, 100) * 3

        elif i % 4 == 1:  # Explosion
            explosion_time = np.random.randint(200, 400)
            data[i, :, explosion_time : explosion_time + 30] += torch.randn(3, 30) * 5

        elif i % 4 == 2:  # Volcanic tremor
            tremor_freq = np.random.uniform(1, 10)  # Hz
            t = torch.linspace(
                0, sequence_length / 100, sequence_length
            )  # 100 Hz sampling
            tremor = torch.sin(2 * np.pi * tremor_freq * t) * 0.5
            data[i, 0, :] += tremor

    # Generate labels
    labels = torch.tensor([i % 4 for i in range(num_samples)])

    return data, labels

def demonstrate_basic_optimization():
    """Demonstrate basic edge optimization functionality"""
    logger.info("=== Basic Edge Optimization Demo ===")

    # Create sample model and data
    model = create_sample_geological_model()
    test_data, test_labels = generate_synthetic_geological_data(200, 20)

    # Create optimizer
    optimizer = EdgeAIOptimizer()

    # Define different hardware scenarios
    scenarios = {
        "Mobile Device": create_mobile_constraints(),
        "Embedded Sensor": create_embedded_constraints(),
        "Edge Server": create_edge_server_constraints(),
    }

    results = {}

    for scenario_name, constraints in scenarios.items():
        logger.info(f"\nOptimizing for {scenario_name}:")
        logger.info(f"  Max Memory: {constraints.max_memory_mb}MB")
        logger.info(f"  Max Inference Time: {constraints.max_inference_time_ms}ms")
        logger.info(f"  CPU Cores: {constraints.cpu_cores}")

        # Create optimization config based on constraints
        if constraints.max_memory_mb < 100:
            # Aggressive optimization for very constrained devices
            config = OptimizationConfig(
                compression_ratio=0.3,
                quantization_bits=8,
                pruning_sparsity=0.5,
                validation_threshold=0.8,
            )
        else:
            # Moderate optimization
            config = OptimizationConfig(
                compression_ratio=0.5,
                quantization_bits=8,
                pruning_sparsity=0.3,
                validation_threshold=0.9,
            )

        # Optimize model
        optimized_model, optimization_results = optimizer.optimize_for_edge(
            model, constraints, config, test_data, test_labels
        )

        results[scenario_name] = optimization_results

        # Print results
        logger.info(f"  Compression: {optimization_results.compression_ratio:.2f}x")
        logger.info(f"  Speedup: {optimization_results.speedup_factor:.2f}x")
        logger.info(
            f"  Accuracy Retention: {optimization_results.accuracy_retention:.1%}"
        )
        logger.info(
            f"  Size: {optimization_results.original_size_mb:.1f}MB → {optimization_results.optimized_size_mb:.1f}MB"
        )
        logger.info(
            f"  Inference Time: {optimization_results.original_inference_time_ms:.1f}ms → {optimization_results.optimized_inference_time_ms:.1f}ms"
        )
        logger.info(f"  Meets Constraints: {optimization_results.meets_constraints}")

    return results

async def demonstrate_automated_generation():
    """Demonstrate automated edge model generation"""
    logger.info("\n=== Automated Edge Model Generation Demo ===")

    # Create mock Helios core (in practice, this would be the real Helios instance)
    class MockHeliosCore:
        def __init__(self):
            self.pinn_engine = type(
                "obj", (object,), {"model": create_sample_geological_model()}
            )()
            self.gan_generator = create_sample_geological_model()

    helios_core = MockHeliosCore()

    # Create storage directory
    storage_path = Path("model_storage/edge_models_demo")
    storage_path.mkdir(parents=True, exist_ok=True)

    # Create automated generator
    generator = AutomatedEdgeGenerator(helios_core, storage_path)

    # Generate edge models for all profiles
    source_model = create_sample_geological_model()
    generated_models = await generator.generate_all_edge_models(
        source_model, "demo_geological_model_v1"
    )

    logger.info(f"Generated {len(generated_models)} edge model variants:")

    for profile_name, metadata in generated_models.items():
        logger.info(f"\n  Profile: {profile_name}")
        logger.info(f"    Model ID: {metadata.model_id}")
        logger.info(f"    Deployment Ready: {metadata.deployment_ready}")
        logger.info(f"    Validation Status: {metadata.validation_status}")

        if metadata.deployment_ready:
            results = metadata.optimization_results
            logger.info(f"    Compression: {results.compression_ratio:.2f}x")
            logger.info(f"    Speedup: {results.speedup_factor:.2f}x")
            logger.info(f"    Accuracy Retention: {results.accuracy_retention:.1%}")

    # Demonstrate model recommendation
    logger.info("\n=== Model Recommendation Demo ===")

    use_cases = ["field_monitoring", "sensor_nodes", "real_time_analysis"]

    for use_case in use_cases:
        recommended = generator.get_recommended_model(use_case)
        if recommended:
            logger.info(f"\nRecommended model for '{use_case}':")
            logger.info(f"  Model ID: {recommended.model_id}")
            logger.info(f"  Profile: {recommended.deployment_profile}")
            logger.info(
                f"  Accuracy Retention: {recommended.optimization_results.accuracy_retention:.1%}"
            )
        else:
            logger.info(f"\nNo suitable model found for '{use_case}'")

    return generated_models

def demonstrate_seismic_cnn_optimization():
    """Demonstrate optimization of seismic CNN model"""
    logger.info("\n=== Seismic CNN Optimization Demo ===")

    # Create seismic CNN model
    seismic_model = create_sample_seismic_cnn()
    test_data, test_labels = generate_synthetic_seismic_data(100, 1000)

    # Create optimizer
    optimizer = EdgeAIOptimizer()

    # Optimize for field station deployment
    field_constraints = HardwareConstraints(
        max_memory_mb=200,
        max_inference_time_ms=500,  # Real-time processing requirement
        cpu_cores=2,
        has_gpu=False,
        target_platform="field_station",
    )

    config = OptimizationConfig(
        compression_ratio=0.4,
        quantization_bits=8,
        pruning_sparsity=0.3,
        validation_threshold=0.85,
    )

    logger.info("Optimizing Seismic CNN for field station deployment...")

    optimized_model, results = optimizer.optimize_for_edge(
        seismic_model, field_constraints, config, test_data, test_labels
    )

    logger.info("Seismic CNN Optimization Results:")
    logger.info(f"  Original Size: {results.original_size_mb:.1f}MB")
    logger.info(f"  Optimized Size: {results.optimized_size_mb:.1f}MB")
    logger.info(f"  Compression: {results.compression_ratio:.2f}x")
    logger.info(
        f"  Original Inference Time: {results.original_inference_time_ms:.1f}ms"
    )
    logger.info(
        f"  Optimized Inference Time: {results.optimized_inference_time_ms:.1f}ms"
    )
    logger.info(f"  Speedup: {results.speedup_factor:.2f}x")
    logger.info(f"  Accuracy Retention: {results.accuracy_retention:.1%}")
    logger.info(f"  Meets Field Station Constraints: {results.meets_constraints}")

    return optimized_model, results

def visualize_optimization_results(results_dict):
    """Create visualizations of optimization results"""
    logger.info("\n=== Creating Optimization Visualizations ===")

    scenarios = list(results_dict.keys())
    compression_ratios = [results_dict[s].compression_ratio for s in scenarios]
    speedup_factors = [results_dict[s].speedup_factor for s in scenarios]
    accuracy_retentions = [results_dict[s].accuracy_retention for s in scenarios]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Compression ratios
    ax1.bar(scenarios, compression_ratios, color="skyblue")
    ax1.set_title("Model Compression Ratios")
    ax1.set_ylabel("Compression Ratio (x)")
    ax1.tick_params(axis="x", rotation=45)

    # Speedup factors
    ax2.bar(scenarios, speedup_factors, color="lightgreen")
    ax2.set_title("Inference Speedup Factors")
    ax2.set_ylabel("Speedup Factor (x)")
    ax2.tick_params(axis="x", rotation=45)

    # Accuracy retention
    ax3.bar(scenarios, [a * 100 for a in accuracy_retentions], color="lightcoral")
    ax3.set_title("Accuracy Retention")
    ax3.set_ylabel("Accuracy Retention (%)")
    ax3.tick_params(axis="x", rotation=45)
    ax3.set_ylim(0, 100)

    # Size comparison
    original_sizes = [results_dict[s].original_size_mb for s in scenarios]
    optimized_sizes = [results_dict[s].optimized_size_mb for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    ax4.bar(x - width / 2, original_sizes, width, label="Original", color="orange")
    ax4.bar(x + width / 2, optimized_sizes, width, label="Optimized", color="purple")
    ax4.set_title("Model Size Comparison")
    ax4.set_ylabel("Model Size (MB)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenarios, rotation=45)
    ax4.legend()

    plt.tight_layout()
    plt.savefig("edge_optimization_results.png", dpi=300, bbox_inches="tight")
    logger.info(
        "Saved optimization results visualization to 'edge_optimization_results.png'"
    )

    return fig

def demonstrate_custom_deployment_profile():
    """Demonstrate creating custom deployment profiles"""
    logger.info("\n=== Custom Deployment Profile Demo ===")

    # Create mock Helios core
    class MockHeliosCore:
        def __init__(self):
            self.pinn_engine = type(
                "obj", (object,), {"model": create_sample_geological_model()}
            )()

    helios_core = MockHeliosCore()
    storage_path = Path("model_storage/custom_profile_demo")

    # Create generator
    generator = AutomatedEdgeGenerator(helios_core, storage_path)

    # Define custom deployment profile for underwater sensors
    from ..core.automated_edge_generation import EdgeDeploymentProfile

    underwater_profile = EdgeDeploymentProfile(
        name="underwater_sensor",
        description="Underwater geological sensors with extreme power constraints",
        hardware_constraints=HardwareConstraints(
            max_memory_mb=20,  # Very limited memory
            max_inference_time_ms=1000,  # Can tolerate longer inference for power savings
            cpu_cores=1,
            has_gpu=False,
            target_platform="underwater",
        ),
        optimization_config=OptimizationConfig(
            compression_ratio=0.1,  # Extreme compression
            quantization_bits=4,  # Very low precision
            pruning_sparsity=0.7,  # Aggressive pruning
            validation_threshold=0.7,  # Accept lower accuracy for extreme constraints
        ),
        target_use_cases=[
            "underwater_monitoring",
            "tsunami_detection",
            "seafloor_analysis",
        ],
        priority=1,
    )

    # Add custom profile
    generator.update_deployment_profile("underwater_sensor", underwater_profile)

    logger.info("Created custom underwater sensor deployment profile:")
    logger.info(
        f"  Max Memory: {underwater_profile.hardware_constraints.max_memory_mb}MB"
    )
    logger.info(
        f"  Max Inference Time: {underwater_profile.hardware_constraints.max_inference_time_ms}ms"
    )
    logger.info(
        f"  Compression Ratio: {underwater_profile.optimization_config.compression_ratio}"
    )
    logger.info(
        f"  Quantization Bits: {underwater_profile.optimization_config.quantization_bits}"
    )
    logger.info(
        f"  Pruning Sparsity: {underwater_profile.optimization_config.pruning_sparsity}"
    )
    logger.info(f"  Target Use Cases: {underwater_profile.target_use_cases}")

    return generator

async def main():
    """Main demonstration function"""
    logger.info("Starting Edge AI Optimization Demonstration")
    logger.info("=" * 50)

    try:
        # Demonstrate basic optimization
        basic_results = demonstrate_basic_optimization()

        # Demonstrate automated generation
        await demonstrate_automated_generation()

        # Demonstrate seismic CNN optimization
        seismic_model, seismic_results = demonstrate_seismic_cnn_optimization()

        # Demonstrate custom deployment profile
        demonstrate_custom_deployment_profile()

        # Create visualizations
        try:
            visualize_optimization_results(basic_results)
        except ImportError:
            logger.warning("Matplotlib not available, skipping visualizations")

        logger.info("\n" + "=" * 50)
        logger.info("Edge AI Optimization Demonstration Complete!")
        logger.info("\nKey Takeaways:")
        logger.info(
            "1. Edge optimization can achieve 2-5x compression with minimal accuracy loss"
        )
        logger.info(
            "2. Different deployment scenarios require different optimization strategies"
        )
        logger.info(
            "3. Automated generation creates optimized variants for all target platforms"
        )
        logger.info(
            "4. Custom deployment profiles can be created for specialized use cases"
        )
        logger.info(
            "5. Real-time seismic processing can be optimized for field deployment"
        )

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
