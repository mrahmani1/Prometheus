"""
Model Complexity Management System Example

Demonstrates the usage of the model complexity management system
for geological simulations.
"""

import json
import os
import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_complexity_manager import ModelComplexityManager
from models.ai_models import GeologicalProcess

def create_synthetic_geological_data() -> Dict[str, torch.Tensor]:
    """Create synthetic geological data for demonstration"""

    print("Creating synthetic geological data...")

    # Seismic data - high frequency, continuous monitoring
    seismic_data = torch.randn(10000, 12)  # 10k measurements, 12 features
    seismic_data += 0.1 * torch.randn_like(seismic_data)  # Add noise

    # GPS/InSAR deformation data - lower frequency but high precision
    deformation_data = torch.randn(3000, 8)  # 3k measurements, 8 features
    deformation_data *= 0.001  # Small displacements (mm scale)

    # Thermal data - satellite and ground-based
    thermal_data = torch.randn(1500, 6) * 10 + 300  # Temperature in Kelvin

    # Geochemical data - sparse but informative
    geochemical_data = torch.randn(500, 15)  # 500 samples, 15 chemical components
    geochemical_data = torch.abs(geochemical_data)  # Concentrations are positive

    # Gas emission data
    gas_data = torch.randn(2000, 10)  # 2k measurements, 10 gas species
    gas_data = torch.abs(gas_data)  # Concentrations are positive

    # Add some missing data to simulate real-world conditions
    for data_name, data_tensor in [
        ("seismic", seismic_data),
        ("deformation", deformation_data),
        ("thermal", thermal_data),
        ("geochemical", geochemical_data),
        ("gas", gas_data),
    ]:
        # Randomly set 3-8% of data to NaN
        missing_fraction = np.random.uniform(0.03, 0.08)
        mask = torch.rand(data_tensor.shape) < missing_fraction
        data_tensor[mask] = float("nan")

    data = {
        "seismic_data": seismic_data,
        "deformation_data": deformation_data,
        "thermal_data": thermal_data,
        "geochemical_data": geochemical_data,
        "gas_emission_data": gas_data,
    }

    print(
        f"Created data with {sum(d.shape[0] for d in data.values())} total measurements"
    )
    return data

def demonstrate_complexity_assessment():
    """Demonstrate comprehensive complexity assessment"""

    print("\n" + "=" * 60)
    print("COMPLEXITY ASSESSMENT DEMONSTRATION")
    print("=" * 60)

    # Initialize the complexity manager
    manager = ModelComplexityManager()

    # Create synthetic data
    data = create_synthetic_geological_data()

    # Assess complexity for volcanic eruption simulation
    geological_process = GeologicalProcess.VOLCANIC_ERUPTION
    target_accuracy = 0.85

    print(f"\nAssessing complexity for {geological_process} simulation...")
    print(f"Target accuracy: {target_accuracy}")

    assessment_result = manager.assess_simulation_complexity(
        geological_process=geological_process,
        data=data,
        target_accuracy=target_accuracy,
    )

    # Display results
    print("\n--- DATA ASSESSMENT ---")
    data_assessment = assessment_result["data_assessment"]
    print(f"Total data points: {data_assessment.total_data_points:,}")
    print(f"Data quality score: {data_assessment.data_quality_score:.3f}")
    print(f"Temporal coverage: {data_assessment.temporal_coverage:.3f}")
    print(f"Spatial coverage: {data_assessment.spatial_coverage:.3f}")
    print(f"Missing data fraction: {data_assessment.missing_data_fraction:.3f}")
    print(f"Noise level: {data_assessment.noise_level:.3f}")
    print(f"Data availability level: {data_assessment.availability_level()}")

    print("\n--- RESOURCE ASSESSMENT ---")
    resource_assessment = assessment_result["resource_assessment"]
    print(f"CPU cores: {resource_assessment.cpu_cores}")
    print(f"Memory: {resource_assessment.memory_gb:.1f} GB")
    print(f"GPU count: {resource_assessment.gpu_count}")
    print(f"GPU memory: {resource_assessment.gpu_memory_gb:.1f} GB")
    print(f"Resource level: {resource_assessment.resource_level()}")

    print("\n--- OPTIMAL CONFIGURATION ---")
    optimal_config = assessment_result["optimal_configuration"]
    print(f"Recommended complexity: {optimal_config.complexity_level}")
    print(f"Network layers: {optimal_config.network_layers}")
    print(f"Physics constraints: {len(optimal_config.physics_constraints)}")
    print(f"Coupling terms: {len(optimal_config.coupling_terms)}")
    print(f"Batch size: {optimal_config.batch_size}")
    print(f"Training epochs: {optimal_config.training_epochs}")

    print("\n--- COST BREAKDOWN ---")
    cost_breakdown = assessment_result["cost_breakdown"]
    for cost_type, cost_value in cost_breakdown.items():
        print(f"{cost_type.replace('_', ' ').title()}: {cost_value:.3f}")

    print("\n--- RECOMMENDATIONS ---")
    for i, recommendation in enumerate(assessment_result["recommendations"], 1):
        print(f"{i}. {recommendation}")

    return assessment_result

def demonstrate_model_hierarchy():
    """Demonstrate the model complexity hierarchy"""

    print("\n" + "=" * 60)
    print("MODEL HIERARCHY DEMONSTRATION")
    print("=" * 60)

    manager = ModelComplexityManager()
    hierarchy = manager.get_complexity_hierarchy()

    print("\nModel Complexity Hierarchy:")
    print("-" * 40)

    for level, info in hierarchy.items():
        config = info["configuration"]
        metrics = info["metrics"]

        print(f"\n{level.value.upper()} COMPLEXITY:")
        print(f"  Network layers: {config['network_layers']}")
        print(f"  Parameters: {metrics['parameter_count']:,}")
        print(f"  Memory requirement: {metrics['memory_requirement']:.2f} GB")
        print(
            f"  Training time estimate: {metrics['training_time_estimate']:.2f} hours"
        )
        print(f"  Inference time: {metrics['inference_time_estimate']:.4f} seconds")
        print(f"  Complexity score: {metrics['complexity_score']:.3f}")
        print(f"  Physics constraints: {len(config['physics_constraints'])}")
        print(f"  Coupling terms: {len(config['coupling_terms'])}")

def demonstrate_cost_optimization():
    """Demonstrate cost optimization for different scenarios"""

    print("\n" + "=" * 60)
    print("COST OPTIMIZATION DEMONSTRATION")
    print("=" * 60)

    manager = ModelComplexityManager()

    # Define different scenarios
    scenarios = [
        {
            "name": "Limited Resources",
            "data_size": 1000,
            "data_quality": 0.6,
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 8.0,
                "gpu_count": 0,
                "gpu_memory_gb": 0.0,
                "storage_gb": 100.0,
                "network_bandwidth_mbps": 100.0,
            },
        },
        {
            "name": "Moderate Resources",
            "data_size": 10000,
            "data_quality": 0.8,
            "resources": {
                "cpu_cores": 16,
                "memory_gb": 64.0,
                "gpu_count": 1,
                "gpu_memory_gb": 16.0,
                "storage_gb": 1000.0,
                "network_bandwidth_mbps": 1000.0,
            },
        },
        {
            "name": "High-End Resources",
            "data_size": 100000,
            "data_quality": 0.95,
            "resources": {
                "cpu_cores": 64,
                "memory_gb": 512.0,
                "gpu_count": 8,
                "gpu_memory_gb": 320.0,
                "storage_gb": 10000.0,
                "network_bandwidth_mbps": 10000.0,
            },
        },
    ]

    results = []

    for scenario in scenarios:
        print(f"\n--- {scenario['name'].upper()} SCENARIO ---")

        # Create data assessment
        from core.model_complexity_manager import DataAssessment, ResourceAssessment

        data_assessment = DataAssessment(
            total_data_points=scenario["data_size"],
            data_quality_score=scenario["data_quality"],
            temporal_coverage=0.7,
            spatial_coverage=0.7,
            data_types_available=["seismic", "deformation", "thermal"],
            missing_data_fraction=0.1,
            noise_level=0.2,
        )

        resource_assessment = ResourceAssessment(**scenario["resources"])

        # Select optimal model
        optimal_config = manager.select_optimal_model(
            data_assessment, resource_assessment, target_accuracy=0.8
        )

        # Estimate costs
        cost_breakdown = manager.cost_estimator.estimate_cost(
            optimal_config, data_assessment.total_data_points, resource_assessment
        )

        print(f"Data size: {scenario['data_size']:,} points")
        print(f"Data quality: {scenario['data_quality']:.2f}")
        print(f"Resource level: {resource_assessment.resource_level().value}")
        print(f"Recommended complexity: {optimal_config.complexity_level.value}")
        print(f"Total cost: {cost_breakdown['total_cost']:.3f}")
        print(f"Training cost: {cost_breakdown['training_cost']:.3f}")
        print(f"Memory cost: {cost_breakdown['memory_cost']:.3f}")

        results.append(
            {
                "scenario": scenario["name"],
                "complexity": optimal_config.complexity_level.value,
                "total_cost": cost_breakdown["total_cost"],
                "data_size": scenario["data_size"],
            }
        )

    return results

def visualize_complexity_tradeoffs(results):
    """Visualize complexity vs cost tradeoffs"""

    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    try:
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Cost vs Data Size
        scenarios = [r["scenario"] for r in results]
        costs = [r["total_cost"] for r in results]
        data_sizes = [r["data_size"] for r in results]

        ax1.scatter(data_sizes, costs, s=100, alpha=0.7)
        for i, scenario in enumerate(scenarios):
            ax1.annotate(
                scenario,
                (data_sizes[i], costs[i]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax1.set_xlabel("Data Size (points)")
        ax1.set_ylabel("Total Cost")
        ax1.set_title("Cost vs Data Size")
        ax1.set_xscale("log")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Complexity Distribution
        complexity_levels = [r["complexity"] for r in results]
        complexity_counts = {}
        for level in complexity_levels:
            complexity_counts[level] = complexity_counts.get(level, 0) + 1

        ax2.bar(complexity_counts.keys(), complexity_counts.values())
        ax2.set_xlabel("Complexity Level")
        ax2.set_ylabel("Count")
        ax2.set_title("Recommended Complexity Distribution")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("model_complexity_analysis.png", dpi=300, bbox_inches="tight")
        print("Visualization saved as 'model_complexity_analysis.png'")

    except ImportError:
        print("Matplotlib not available - skipping visualization")

def save_assessment_results(assessment_result: Dict[str, Any]):
    """Save assessment results to file"""

    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    manager = ModelComplexityManager()

    # Save to JSON file
    filename = "complexity_assessment_example.json"
    manager.save_assessment(assessment_result, filename)

    print(f"Assessment results saved to {filename}")

    # Also save a summary
    summary = {
        "optimal_complexity": assessment_result[
            "optimal_configuration"
        ].complexity_level.value,
        "total_cost": assessment_result["cost_breakdown"]["total_cost"],
        "data_points": assessment_result["data_assessment"].total_data_points,
        "data_quality": assessment_result["data_assessment"].data_quality_score,
        "recommendations_count": len(assessment_result["recommendations"]),
    }

    with open("complexity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Summary saved to complexity_summary.json")

def main():
    """Main demonstration function"""

    print("Model Complexity Management System Demonstration")
    print("=" * 60)

    try:
        # 1. Demonstrate complexity assessment
        assessment_result = demonstrate_complexity_assessment()

        # 2. Demonstrate model hierarchy
        demonstrate_model_hierarchy()

        # 3. Demonstrate cost optimization
        optimization_results = demonstrate_cost_optimization()

        # 4. Visualize results
        visualize_complexity_tradeoffs(optimization_results)

        # 5. Save results
        save_assessment_results(assessment_result)

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey takeaways:")
        print("1. The system automatically assesses data quality and availability")
        print("2. It evaluates computational resources and constraints")
        print(
            "3. It recommends optimal model complexity based on cost-accuracy tradeoffs"
        )
        print("4. It provides actionable recommendations for improvement")
        print("5. All assessments are saved for future reference and analysis")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

        traceback.print_exc()

if __name__ == "__main__":
    main()
