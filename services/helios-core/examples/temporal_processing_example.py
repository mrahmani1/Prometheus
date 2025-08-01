"""
Multi-Scale Temporal Processing Example

Demonstrates the implementation of separate model branches for geological vs. real-time timescales,
temporal attention mechanisms, adaptive time-stepping, and temporal interpolation/extrapolation.

This example shows how to:
    1. Create a multi-scale temporal PINN model
2. Make predictions across different temporal scales
3. Use adaptive time stepping for geological processes
4. Interpolate and extrapolate temporal solutions
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.temporal_processing import (
    GeologicalProcess,
    MultiScaleTemporalPINN,
    TimeScale,
    create_default_temporal_scales,
)

def create_sample_geological_data():
    """Create sample geological data for different temporal scales"""

    # Spatial coordinates (x, y, z, additional parameters)
    spatial_coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1000.0],  # Surface location with temperature
            [10.0, 10.0, -5.0, 1200.0],  # Shallow depth
            [0.0, 0.0, -20.0, 1500.0],  # Deep location
        ],
        dtype=torch.float32,
    )

    # Time coordinates for different scales
    time_scales = {
        "real_time": torch.tensor([[3600.0], [7200.0], [10800.0]]),  # Hours
        "short_term": torch.tensor([[86400.0], [172800.0], [259200.0]]),  # Days
        "geological": torch.tensor(
            [[3.15e10], [6.3e10], [9.45e10]]
        ),  # Thousands of years
    }

    return spatial_coords, time_scales

def demonstrate_multi_scale_prediction():
    """Demonstrate multi-scale temporal predictions"""
    print("=== Multi-Scale Temporal Prediction Demo ===")

    # Create temporal scales
    temporal_scales = create_default_temporal_scales()[:3]  # Use first 3 scales

    # Create multi-scale temporal PINN
    model = MultiScaleTemporalPINN(
        input_dim=4,  # x, y, z, temperature
        output_dim=8,  # velocity components, pressure, temperature evolution, etc.
        temporal_scales=temporal_scales,
        branch_hidden_dims=[64, 128, 64],
        attention_heads=8,
    )

    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Temporal scales: {[scale.name.value for scale in temporal_scales]}")

    # Create sample data
    spatial_coords, time_scales = create_sample_geological_data()

    # Make predictions for different temporal scales
    results = {}

    for scale_name, time_coords in time_scales.items():
        if scale_name in [
            "real_time",
            "short_term",
        ]:  # Only use scales we have in our model
            print(f"\nPredicting for {scale_name} scale...")

            # Make prediction
            with torch.no_grad():
                prediction = model(spatial_coords, time_coords)

            results[scale_name] = prediction

            print(f"Fused prediction shape: {prediction['fused_prediction'].shape}")
            print(
                f"Branch predictions: {list(prediction['branch_predictions'].keys())}"
            )

    return model, results

def demonstrate_adaptive_time_stepping():
    """Demonstrate adaptive time stepping for geological processes"""
    print("\n=== Adaptive Time Stepping Demo ===")

    # Create temporal scales
    temporal_scales = create_default_temporal_scales()[:3]

    # Create model
    model = MultiScaleTemporalPINN(
        input_dim=4,
        output_dim=8,
        temporal_scales=temporal_scales,
        branch_hidden_dims=[64, 128, 64],
        attention_heads=8,
    )

    # Single spatial location
    spatial_coords = torch.tensor([[0.0, 0.0, -10.0, 1300.0]], dtype=torch.float32)

    # Different geological processes with different time scales
    processes = [
        (GeologicalProcess.SEISMIC_WAVE_PROPAGATION, 0.0, 100.0, "Seismic Wave"),
        (GeologicalProcess.CRUSTAL_DEFORMATION, 0.0, 1000.0, "Crustal Deformation"),
        (GeologicalProcess.MAGMA_CHAMBER_EVOLUTION, 0.0, 10000.0, "Magma Evolution"),
    ]

    for process, t_start, t_end, name in processes:
        print(f"\nSimulating {name}...")

        # Run adaptive time stepping
        results = model.predict_with_adaptive_stepping(
            spatial_coords, time_start=t_start, time_end=t_end, process_type=process
        )

        print(f"Total time steps: {results['total_steps']}")
        print(f"Time range: {results['times'][0]:.2f} to {results['times'][-1]:.2f}")
        print(f"Average time step: {np.mean(results['time_steps']):.2f}")
        print(
            f"Min/Max time step: {np.min(results['time_steps']):.2f} / {np.max(results['time_steps']):.2f}"
        )

def demonstrate_temporal_interpolation_extrapolation():
    """Demonstrate temporal interpolation and extrapolation"""
    print("\n=== Temporal Interpolation/Extrapolation Demo ===")

    # Create temporal scales
    temporal_scales = create_default_temporal_scales()[:3]

    # Create model
    model = MultiScaleTemporalPINN(
        input_dim=4,
        output_dim=8,
        temporal_scales=temporal_scales,
        branch_hidden_dims=[64, 128, 64],
        attention_heads=8,
    )

    # Create known data points (simulating historical geological data)
    known_times = torch.tensor([0.0, 1000.0, 2000.0, 3000.0, 4000.0])
    known_solutions = torch.randn(5, 8)  # Simulated geological measurements

    print("Known data points:")
    for i, (time, solution) in enumerate(zip(known_times, known_solutions)):
        print(f"  Time {time:.0f}: Solution norm = {torch.norm(solution):.3f}")

    # Interpolation: Fill in missing data points
    interp_times = torch.tensor([500.0, 1500.0, 2500.0, 3500.0])
    interpolated = model.interpolate_temporal_solution(
        known_times, known_solutions, interp_times
    )

    print("\nInterpolated data points:")
    for i, (time, solution) in enumerate(zip(interp_times, interpolated)):
        print(f"  Time {time:.0f}: Solution norm = {torch.norm(solution):.3f}")

    # Extrapolation: Predict future evolution
    future_times = torch.tensor([5000.0, 6000.0, 7000.0])

    # Try different extrapolation methods
    methods = ["linear", "polynomial", "exponential"]

    for method in methods:
        extrapolated = model.extrapolate_temporal_solution(
            known_times, known_solutions, future_times, method=method
        )

        print(f"\nExtrapolated data ({method} method):")
        for i, (time, solution) in enumerate(zip(future_times, extrapolated)):
            print(f"  Time {time:.0f}: Solution norm = {torch.norm(solution):.3f}")

def demonstrate_geological_timescale_modeling():
    """Demonstrate modeling across geological timescales"""
    print("\n=== Geological Timescale Modeling Demo ===")

    # Create all temporal scales
    temporal_scales = create_default_temporal_scales()

    # Create comprehensive model
    model = MultiScaleTemporalPINN(
        input_dim=4,
        output_dim=8,
        temporal_scales=temporal_scales,
        branch_hidden_dims=[64, 128, 64],
        attention_heads=8,
    )

    print(f"Model covers {len(temporal_scales)} temporal scales:")
    for scale in temporal_scales:
        min_years = scale.min_time / (365.25 * 24 * 3600)
        max_years = scale.max_time / (365.25 * 24 * 3600)
        char_years = scale.characteristic_time / (365.25 * 24 * 3600)

        print(
            f"  {scale.name.value}: {min_years:.0e} to {max_years:.0e} years (char: {char_years:.0e})"
        )

    # Simulate tectonic plate movement over deep time
    print("\nSimulating tectonic plate movement...")

    # Plate boundary location
    plate_coords = torch.tensor([[0.0, 0.0, 0.0, 1000.0]], dtype=torch.float32)

    # Deep time coordinates (millions of years) - single time point to match single spatial location
    deep_time_coords = torch.tensor([[3.15e13]])  # 1 million years

    with torch.no_grad():
        deep_time_prediction = model(
            plate_coords, deep_time_coords, target_scale=TimeScale.DEEP_TIME
        )

    print("Deep time prediction completed")
    print(f"Prediction shape: {deep_time_prediction['fused_prediction'].shape}")

    if "target_scale_prediction" in deep_time_prediction:
        print("Deep time scale prediction available")

    # Simulate mantle convection
    print("\nSimulating mantle convection...")

    # Mantle location
    mantle_coords = torch.tensor([[0.0, 0.0, -100.0, 2000.0]], dtype=torch.float32)

    # Geological timescale - single time point to match single spatial location
    geological_time_coords = torch.tensor([[3.15e11]])  # 10k years

    with torch.no_grad():
        geological_prediction = model(
            mantle_coords, geological_time_coords, target_scale=TimeScale.GEOLOGICAL
        )

    print("Geological timescale prediction completed")
    print(f"Prediction shape: {geological_prediction['fused_prediction'].shape}")

def plot_temporal_scales():
    """Create a visualization of temporal scales"""
    print("\n=== Creating Temporal Scale Visualization ===")

    temporal_scales = create_default_temporal_scales()

    # Convert to years for plotting
    scale_data = []
    for scale in temporal_scales:
        min_years = scale.min_time / (365.25 * 24 * 3600)
        max_years = scale.max_time / (365.25 * 24 * 3600)
        char_years = scale.characteristic_time / (365.25 * 24 * 3600)

        scale_data.append(
            {
                "name": scale.name.value,
                "min": min_years,
                "max": max_years,
                "char": char_years,
            }
        )

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    y_positions = range(len(scale_data))
    colors = ["blue", "green", "orange", "red", "purple"]

    for i, (scale, color) in enumerate(zip(scale_data, colors)):
        # Plot range as horizontal bar
        ax.barh(
            i,
            np.log10(scale["max"]) - np.log10(scale["min"]),
            left=np.log10(scale["min"]),
            alpha=0.6,
            color=color,
            label=scale["name"],
        )

        # Mark characteristic time
        ax.axvline(np.log10(scale["char"]), color=color, linestyle="--", alpha=0.8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([scale["name"] for scale in scale_data])
    ax.set_xlabel("Log10(Time in Years)")
    ax.set_title("Temporal Scales in Multi-Scale Geological Modeling")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("temporal_scales.png", dpi=300, bbox_inches="tight")
    print("Temporal scale visualization saved as 'temporal_scales.png'")

def main():
    """Run all demonstrations"""
    print("Multi-Scale Temporal Processing for Geological Systems")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Run demonstrations
        model, results = demonstrate_multi_scale_prediction()
        demonstrate_adaptive_time_stepping()
        demonstrate_temporal_interpolation_extrapolation()
        demonstrate_geological_timescale_modeling()

        # Create visualization
        try:
            plot_temporal_scales()
        except ImportError:
            print("\nMatplotlib not available, skipping visualization")

        print("\n" + "=" * 60)
        print("Multi-Scale Temporal Processing Demo Completed Successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Separate model branches for different temporal scales")
        print("✓ Temporal attention mechanisms for multi-scale integration")
        print("✓ Adaptive time-stepping for different geological processes")
        print("✓ Temporal interpolation and extrapolation methods")
        print("✓ Deep time geological modeling (millions to billions of years)")
        print("✓ Real-time to geological timescale integration")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
