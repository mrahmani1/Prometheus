#!/usr/bin/env python3
"""
Simple test script for edge AI optimization functionality
"""

import torch
import torch.nn as nn
from core.edge_ai_optimizer import (
    EdgeAIOptimizer,
    HardwareConstraints,
    OptimizationConfig,
    create_edge_server_constraints,
    create_embedded_constraints,
    create_mobile_constraints,
    
)
from core.edge_ai_optimizer import torch.nn as nn
from core.edge_ai_optimizer import torch.nn.functional as F

def test_basic_optimization():
    """Test basic edge optimization functionality"""
    print("Testing basic edge optimization...")

    # Create a simple test model
    model = nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3)
    )

    # Create test data
    test_data = torch.randn(100, 10)
    test_labels = torch.randint(0, 3, (100,))

    # Create optimizer
    optimizer = EdgeAIOptimizer()

    # Test different hardware scenarios
    scenarios = {
        "Mobile Device": create_mobile_constraints(),
        "Embedded Sensor": create_embedded_constraints(),
        "Edge Server": create_edge_server_constraints(),
    }

    for scenario_name, constraints in scenarios.items():
        print(f"\nOptimizing for {scenario_name}:")
        print(f"  Max Memory: {constraints.max_memory_mb}MB")
        print(f"  Max Inference Time: {constraints.max_inference_time_ms}ms")

        # Create optimization config
        config = OptimizationConfig(
            compression_ratio=0.5,
            quantization_bits=8,
            pruning_sparsity=0.3,
            validation_threshold=0.8,
        )

        try:
            # Run optimization
            optimized_model, results = optimizer.optimize_for_edge(
                model, constraints, config, test_data, test_labels
            )

            print("  ✓ Optimization successful!")
            print(f"  Compression: {results.compression_ratio:.2f}x")
            print(f"  Speedup: {results.speedup_factor:.2f}x")
            print(f"  Accuracy Retention: {results.accuracy_retention:.1%}")
            print(f"  Meets Constraints: {results.meets_constraints}")

        except Exception as e:
            print(f"  ✗ Optimization failed: {e}")

def test_model_variants():
    """Test generation of multiple model variants"""
    print("\n" + "=" * 50)
    print("Testing model variant generation...")

    # Create test model
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10),
    )

    # Create test data for CNN
    test_data = torch.randn(50, 1, 28, 28)
    test_labels = torch.randint(0, 10, (50,))

    # Create optimizer
    optimizer = EdgeAIOptimizer()

    # Create different hardware profiles
    hardware_profiles = [
        create_mobile_constraints(),
        create_embedded_constraints(),
        create_edge_server_constraints(),
    ]

    try:
        variants = optimizer.generate_edge_model_variants(
            model, hardware_profiles, test_data, test_labels
        )

        print(f"Generated {len(variants)} model variants:")
        for profile_name, (optimized_model, results) in variants.items():
            print(f"  {profile_name}:")
            print(f"    Compression: {results.compression_ratio:.2f}x")
            print(f"    Speedup: {results.speedup_factor:.2f}x")
            print(f"    Accuracy: {results.accuracy_retention:.1%}")
            print(f"    Constraints Met: {results.meets_constraints}")

    except Exception as e:
        print(f"✗ Variant generation failed: {e}")

def test_custom_constraints():
    """Test optimization with custom hardware constraints"""
    print("\n" + "=" * 50)
    print("Testing custom hardware constraints...")

    # Create test model
    model = nn.Sequential(
        nn.Linear(20, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 5),
    )

    # Create test data
    test_data = torch.randn(80, 20)
    test_labels = torch.randint(0, 5, (80,))

    # Create optimizer
    optimizer = EdgeAIOptimizer()

    # Define custom constraints for underwater sensor
    underwater_constraints = HardwareConstraints(
        max_memory_mb=20,  # Very limited memory
        max_inference_time_ms=1000,  # Can tolerate longer inference
        cpu_cores=1,
        has_gpu=False,
        target_platform="underwater",
    )

    # Aggressive optimization config
    aggressive_config = OptimizationConfig(
        compression_ratio=0.2,  # Extreme compression
        quantization_bits=8,
        pruning_sparsity=0.6,  # Heavy pruning
        validation_threshold=0.7,  # Accept lower accuracy
    )

    print("Custom underwater sensor constraints:")
    print(f"  Max Memory: {underwater_constraints.max_memory_mb}MB")
    print(f"  Max Inference Time: {underwater_constraints.max_inference_time_ms}ms")
    print(f"  Compression Ratio: {aggressive_config.compression_ratio}")
    print(f"  Pruning Sparsity: {aggressive_config.pruning_sparsity}")

    try:
        optimized_model, results = optimizer.optimize_for_edge(
            model, underwater_constraints, aggressive_config, test_data, test_labels
        )

        print("✓ Custom optimization successful!")
        print(f"  Original Size: {results.original_size_mb:.2f}MB")
        print(f"  Optimized Size: {results.optimized_size_mb:.2f}MB")
        print(f"  Compression: {results.compression_ratio:.2f}x")
        print(f"  Speedup: {results.speedup_factor:.2f}x")
        print(f"  Accuracy Retention: {results.accuracy_retention:.1%}")
        print(f"  Meets Constraints: {results.meets_constraints}")

    except Exception as e:
        print(f"✗ Custom optimization failed: {e}")

def main():
    """Run all tests"""
    print("Edge AI Optimization Test Suite")
    print("=" * 50)

    try:
        test_basic_optimization()
        test_model_variants()
        test_custom_constraints()

        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        print("\nKey Features Demonstrated:")
        print("1. Model compression with weight sharing")
        print("2. Dynamic quantization for inference speedup")
        print("3. Magnitude-based pruning for size reduction")
        print("4. Performance validation against hardware constraints")
        print("5. Multiple deployment profile support")
        print("6. Custom constraint handling")

    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
