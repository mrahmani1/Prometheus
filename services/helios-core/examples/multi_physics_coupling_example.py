"""
Multi-Physics Coupling Engine Example

Demonstrates the usage of the advanced multi-physics coupling engine for geological processes.
Shows thermal-mechanical-chemical-electromagnetic coupling with adaptive time stepping,
stability analysis, and convergence monitoring.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add the core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.multi_physics_coupling import (
    AdaptiveTimeStepConfig,
    ConvergenceConfig,
    MechanicalSolver,
    MeshRefinementConfig,
    MultiPhysicsCouplingEngine,
    PhysicsDomain,
    ThermalSolver,
    create_multi_physics_engine,
    create_thermal_mechanical_coupling,
)
from core.multi_physics_integration import create_magma_flow_system
from core.pinn_engine import GeologicalProcess

def example_thermal_mechanical_coupling():
    """
    Example 1: Thermal-Mechanical Coupling

    Demonstrates coupling between thermal diffusion and elastic deformation,
    showing thermal expansion effects and mechanical heating.
    """
    print("=" * 60)
    print("Example 1: Thermal-Mechanical Coupling")
    print("=" * 60)

    # Create domain solvers
    domains = {
        PhysicsDomain.THERMAL: ThermalSolver(
            thermal_diffusivity=1e-6, heat_capacity=1000.0  # m²/s  # J/(kg·K)
        ),
        PhysicsDomain.MECHANICAL: MechanicalSolver(
            youngs_modulus=70e9,  # Pa
            poisson_ratio=0.3,  # dimensionless
            density=2700.0,  # kg/m³
        ),
    }

    # Create coupling terms
    coupling_terms = create_thermal_mechanical_coupling(
        thermal_expansion_coefficient=1e-5, reference_temperature=293.15  # 1/K  # K
    )

    # Configuration
    time_step_config = AdaptiveTimeStepConfig(
        initial_dt=1e-3,
        min_dt=1e-6,
        max_dt=1e-2,
        safety_factor=0.8,
        error_tolerance=1e-6,
    )

    convergence_config = ConvergenceConfig(
        max_iterations=50, absolute_tolerance=1e-6, relative_tolerance=1e-6
    )

    mesh_refinement_config = MeshRefinementConfig(
        max_refinement_levels=3, refinement_threshold=1e-3
    )

    # Create multi-physics engine
    engine = MultiPhysicsCouplingEngine(
        domains=domains,
        coupling_terms=coupling_terms,
        time_step_config=time_step_config,
        convergence_config=convergence_config,
        mesh_refinement_config=mesh_refinement_config,
    )

    # Initialize states
    grid_size = 20
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Initial temperature field with hot spot in center
    temperature_initial = 293.15 + 100.0 * torch.exp(
        -((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.1
    )

    # Initial displacement field (zero)
    displacement_initial = torch.zeros(grid_size, grid_size, 3)

    initial_states = {
        PhysicsDomain.THERMAL: temperature_initial,
        PhysicsDomain.MECHANICAL: displacement_initial,
    }

    engine.initialize_states(initial_states)

    print(
        f"Initial temperature range: {temperature_initial.min():.1f} - {temperature_initial.max():.1f} K"
    )
    print(f"Grid size: {grid_size} x {grid_size}")

    # Run simulation
    end_time = 0.1  # seconds
    print(f"\nRunning thermal-mechanical simulation for {end_time} seconds...")

    results = engine.simulate(end_time, output_frequency=0.02)

    print("Simulation completed:")
    print(f"  Total steps: {results['total_steps']}")
    print(f"  Final time: {results['final_time']:.4f} s")
    print(f"  Refinement level: {results['refinement_level']}")
    print(
        f"  Average convergence iterations: {np.mean([c.iterations for c in results['convergence_history']]):.1f}"
    )

    # Analyze results
    final_temperature = results["states"][PhysicsDomain.THERMAL][-1]
    final_displacement = results["states"][PhysicsDomain.MECHANICAL][-1]

    print("\nFinal state:")
    print(
        f"  Temperature range: {final_temperature.min():.1f} - {final_temperature.max():.1f} K"
    )
    print(f"  Max displacement: {torch.norm(final_displacement, dim=-1).max():.2e} m")

    # Plot results if matplotlib is available
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Initial temperature
        im1 = axes[0, 0].imshow(temperature_initial.numpy(), cmap="hot", origin="lower")
        axes[0, 0].set_title("Initial Temperature (K)")
        plt.colorbar(im1, ax=axes[0, 0])

        # Final temperature
        im2 = axes[0, 1].imshow(final_temperature.numpy(), cmap="hot", origin="lower")
        axes[0, 1].set_title("Final Temperature (K)")
        plt.colorbar(im2, ax=axes[0, 1])

        # Displacement magnitude
        displacement_mag = torch.norm(final_displacement, dim=-1)
        im3 = axes[1, 0].imshow(
            displacement_mag.numpy(), cmap="viridis", origin="lower"
        )
        axes[1, 0].set_title("Displacement Magnitude (m)")
        plt.colorbar(im3, ax=axes[1, 0])

        # Time step history
        axes[1, 1].plot(results["time_history"], results["dt_history"])
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Time Step (s)")
        axes[1, 1].set_title("Adaptive Time Stepping")
        axes[1, 1].set_yscale("log")

        plt.tight_layout()
        plt.savefig(
            "thermal_mechanical_coupling_results.png", dpi=150, bbox_inches="tight"
        )
        print("\nResults saved to: thermal_mechanical_coupling_results.png")

    except ImportError:
        print("\nMatplotlib not available - skipping plots")

    return results

def example_multi_domain_coupling():
    """
    Example 2: Multi-Domain Coupling

    Demonstrates coupling between thermal, chemical, and electromagnetic domains
    for a comprehensive multi-physics simulation.
    """
    print("\n" + "=" * 60)
    print("Example 2: Multi-Domain Coupling")
    print("=" * 60)

    # Create comprehensive multi-physics engine
    engine = create_multi_physics_engine(
        geological_process=GeologicalProcess.MAGMA_FLOW,
        domain_configs={
            PhysicsDomain.THERMAL: {
                "thermal_diffusivity": 1e-6,
                "heat_capacity": 1200.0,
            },
            PhysicsDomain.CHEMICAL: {
                "diffusion_coefficient": 1e-9,
                "reaction_rate": 1e-3,
            },
            PhysicsDomain.ELECTROMAGNETIC: {
                "permittivity": 8.854e-12,
                "permeability": 4e-7 * np.pi,
            },
        },
    )

    # Initialize states
    grid_size = 15  # Smaller grid for faster computation

    # Temperature field with gradient
    temperature = 1000.0 + 500.0 * torch.rand(grid_size, grid_size)

    # Chemical concentration field
    concentration = 0.1 * torch.rand(grid_size, grid_size)

    # Electromagnetic field (simplified)
    em_field = torch.zeros(grid_size, grid_size, 6)  # [Ex, Ey, Ez, Bx, By, Bz]
    em_field[:, :, 0] = 1e3 * torch.rand(grid_size, grid_size)  # Electric field

    initial_states = {
        PhysicsDomain.THERMAL: temperature,
        PhysicsDomain.CHEMICAL: concentration,
        PhysicsDomain.ELECTROMAGNETIC: em_field,
    }

    engine.initialize_states(initial_states)

    print(f"Initialized {len(initial_states)} physics domains")
    print(f"Grid size: {grid_size} x {grid_size}")
    print(f"Coupling terms: {len(engine.coupling_terms)}")

    # Run simulation
    end_time = 0.05
    print(f"\nRunning multi-domain simulation for {end_time} seconds...")

    results = engine.simulate(end_time, output_frequency=0.01)

    print("Simulation completed:")
    print(f"  Total steps: {results['total_steps']}")
    print(f"  Final time: {results['final_time']:.4f} s")
    print(
        f"  Stability violations: {sum(1 for s in results['stability_history'] if not s.is_stable)}"
    )

    # Get diagnostics
    diagnostics = engine.get_diagnostics()
    print("\nDiagnostics:")
    print(
        f"  Average convergence iterations: {diagnostics['average_convergence_iterations']:.1f}"
    )
    print(f"  Time step adaptations: {diagnostics['time_step_adaptations']}")
    print(f"  Active coupling terms: {diagnostics['coupling_terms_active']}")

    return results

def example_pinn_integration():
    """
    Example 3: PINN-MultiPhysics Integration

    Demonstrates integration between Physics-Informed Neural Networks
    and multi-physics coupling for enhanced geological modeling.
    """
    print("\n" + "=" * 60)
    print("Example 3: PINN-MultiPhysics Integration")
    print("=" * 60)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create integrated system for magma flow
    try:
        integrated_system = create_magma_flow_system(device=device)

        print("Created integrated system:")
        print(f"  Geological process: {integrated_system.geological_process.value}")
        print(f"  Integration mode: {integrated_system.integration_config.mode.value}")
        print(f"  PINN models: {len(integrated_system.pinn_engine.list_models())}")
        print(
            f"  Physics domains: {len(integrated_system.multi_physics_engine.domains)}"
        )

        # Create sample coordinates for training
        n_points = 100
        x = torch.rand(n_points, 1) * 2 - 1  # [-1, 1]
        y = torch.rand(n_points, 1) * 2 - 1  # [-1, 1]
        z = torch.rand(n_points, 1) * 2 - 1  # [-1, 1]
        t = torch.rand(n_points, 1) * 0.1  # [0, 0.1]

        coordinates = torch.cat([x, y, z, t], dim=1)

        print(f"\nTraining coordinates shape: {coordinates.shape}")

        # Train coupled system
        model_id = integrated_system.pinn_engine.list_models()[0]
        print(f"Training model: {model_id}")

        training_results = integrated_system.train_coupled_system(
            model_id=model_id,
            coordinates=coordinates,
            end_time=0.02,  # Short training for example
            output_frequency=0.005,
        )

        print("\nTraining completed:")
        print(f"  Total steps: {training_results['total_steps']}")
        print(
            f"  Final coupling residual: {training_results['training_history']['coupling_residuals'][-1]:.2e}"
        )

        # Validate system
        validation_coords = torch.rand(50, 4) * torch.tensor(
            [2, 2, 2, 0.1]
        ) - torch.tensor([1, 1, 1, 0])
        validation_metrics = integrated_system.validate_coupling(
            model_id, validation_coords
        )

        print("\nValidation metrics:")
        for metric, value in validation_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2e}")
            else:
                print(f"  {metric}: {value}")

        # Get integration diagnostics
        diagnostics = integrated_system.get_integration_diagnostics()
        print("\nIntegration diagnostics:")
        print(f"  Convergence rate: {diagnostics.get('convergence_rate', 0):.2%}")
        print(
            f"  Average coupling residual: {diagnostics.get('average_coupling_residual', 0):.2e}"
        )

        return training_results

    except Exception as e:
        print(f"Integration example failed: {e}")
        print("This may be due to missing dependencies or model complexity")
        return None

def example_stability_analysis():
    """
    Example 4: Stability Analysis and Adaptive Time Stepping

    Demonstrates the stability analysis capabilities and adaptive time stepping
    for maintaining numerical stability in coupled simulations.
    """
    print("\n" + "=" * 60)
    print("Example 4: Stability Analysis and Adaptive Time Stepping")
    print("=" * 60)

    # Create a simple thermal system for stability analysis
    domains = {
        PhysicsDomain.THERMAL: ThermalSolver(
            thermal_diffusivity=1e-5,  # Higher diffusivity for faster dynamics
            heat_capacity=1000.0,
        )
    }

    # No coupling terms for this example
    coupling_terms = []

    # Aggressive time stepping configuration to test stability
    time_step_config = AdaptiveTimeStepConfig(
        initial_dt=1e-2,  # Start with large time step
        min_dt=1e-8,
        max_dt=1e-1,
        safety_factor=0.5,  # Conservative safety factor
        error_tolerance=1e-5,
    )

    convergence_config = ConvergenceConfig(max_iterations=20, absolute_tolerance=1e-7)

    engine = MultiPhysicsCouplingEngine(
        domains=domains,
        coupling_terms=coupling_terms,
        time_step_config=time_step_config,
        convergence_config=convergence_config,
    )

    # Initialize with sharp temperature gradient (challenging for stability)
    grid_size = 10
    x = torch.linspace(0, 1, grid_size)
    temperature = 300.0 + 1000.0 * (x > 0.5).float().unsqueeze(0).expand(grid_size, -1)

    initial_states = {PhysicsDomain.THERMAL: temperature}

    engine.initialize_states(initial_states)

    print(
        f"Initial temperature profile: step function with {temperature.max() - temperature.min():.0f} K jump"
    )

    # Run simulation with stability monitoring
    stability_history = []
    time_steps = []

    current_time = 0.0
    max_time = 0.1
    step = 0

    print("\nRunning stability analysis simulation...")

    while current_time < max_time and step < 100:  # Limit steps for example
        # Perform stability analysis
        stability_result = engine.analyze_stability()
        stability_history.append(stability_result)
        time_steps.append(engine.current_dt)

        # Solve one step
        convergence_result = engine.solve_coupled_step(engine.current_dt)

        # Adapt time step based on stability and convergence
        engine.adapt_time_step(convergence_result, stability_result)

        current_time += engine.current_dt
        step += 1

        # Log every 10 steps
        if step % 10 == 0:
            print(
                f"  Step {step}: t={current_time:.4f}, dt={engine.current_dt:.2e}, "
                f"stable={stability_result.is_stable}, "
                f"margin={stability_result.stability_margin:.2e}"
            )

    print("\nStability analysis completed:")
    print(f"  Total steps: {step}")
    print(f"  Final time: {current_time:.4f} s")
    print(
        f"  Stability violations: {sum(1 for s in stability_history if not s.is_stable)}"
    )
    print(f"  Time step range: {min(time_steps):.2e} - {max(time_steps):.2e} s")
    print(
        f"  Average stability margin: {np.mean([s.stability_margin for s in stability_history]):.2e}"
    )

    # Plot stability analysis if matplotlib available
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Time step evolution
        axes[0, 0].plot(range(len(time_steps)), time_steps)
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Time Step (s)")
        axes[0, 0].set_title("Adaptive Time Step Evolution")
        axes[0, 0].set_yscale("log")

        # Stability margin
        stability_margins = [s.stability_margin for s in stability_history]
        axes[0, 1].plot(range(len(stability_margins)), stability_margins)
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Stability Margin")
        axes[0, 1].set_title("Stability Margin Evolution")
        axes[0, 1].set_yscale("log")

        # Critical time step
        critical_time_steps = [s.critical_time_step for s in stability_history]
        axes[1, 0].plot(range(len(critical_time_steps)), critical_time_steps)
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Critical Time Step (s)")
        axes[1, 0].set_title("Critical Time Step Evolution")
        axes[1, 0].set_yscale("log")

        # Stability status
        stability_status = [1 if s.is_stable else 0 for s in stability_history]
        axes[1, 1].plot(range(len(stability_status)), stability_status, "o-")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Stable (1) / Unstable (0)")
        axes[1, 1].set_title("Stability Status")
        axes[1, 1].set_ylim(-0.1, 1.1)

        plt.tight_layout()
        plt.savefig("stability_analysis_results.png", dpi=150, bbox_inches="tight")
        print("\nStability analysis plots saved to: stability_analysis_results.png")

    except ImportError:
        print("\nMatplotlib not available - skipping plots")

    return {
        "stability_history": stability_history,
        "time_steps": time_steps,
        "final_time": current_time,
        "total_steps": step,
    }

def main():
    """
    Run all multi-physics coupling examples
    """
    print("Multi-Physics Coupling Engine Examples")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Example 1: Thermal-Mechanical Coupling
        results1 = example_thermal_mechanical_coupling()

        # Example 2: Multi-Domain Coupling
        results2 = example_multi_domain_coupling()

        # Example 3: PINN Integration (may fail if dependencies missing)
        results3 = example_pinn_integration()

        # Example 4: Stability Analysis
        results4 = example_stability_analysis()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

        return {
            "thermal_mechanical": results1,
            "multi_domain": results2,
            "pinn_integration": results3,
            "stability_analysis": results4,
        }

    except Exception as e:
        print(f"\nExample execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
