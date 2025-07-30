"""
Simple Multi-Physics Coupling Demo

A standalone demonstration of the multi-physics coupling engine
without external dependencies.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add the core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core.multi_physics_coupling import (
    MultiPhysicsCouplingEngine,
    PhysicsDomain,
    ThermalSolver,
    MechanicalSolver,
    AdaptiveTimeStepConfig,
    ConvergenceConfig,
    MeshRefinementConfig,
    create_thermal_mechanical_coupling,
    create_multi_physics_engine
)
from core.pinn_engine import GeologicalProcess


def simple_thermal_mechanical_demo():
    """
    Simple demonstration of thermal-mechanical coupling
    """
    print("=" * 60)
    print("Simple Multi-Physics Coupling Demo")
    print("=" * 60)
    
    # Create domain solvers
    domains = {
        PhysicsDomain.THERMAL: ThermalSolver(
            thermal_diffusivity=1e-6,  # m²/s
            heat_capacity=1000.0       # J/(kg·K)
        ),
        PhysicsDomain.MECHANICAL: MechanicalSolver(
            youngs_modulus=70e9,       # Pa
            poisson_ratio=0.3,         # dimensionless
            density=2700.0             # kg/m³
        )
    }
    
    # Create coupling terms
    coupling_terms = create_thermal_mechanical_coupling(
        thermal_expansion_coefficient=1e-5,  # 1/K
        reference_temperature=293.15         # K
    )
    
    # Configuration
    time_step_config = AdaptiveTimeStepConfig(
        initial_dt=1e-3,
        min_dt=1e-6,
        max_dt=1e-2,
        safety_factor=0.8
    )
    
    convergence_config = ConvergenceConfig(
        max_iterations=20,  # Reduced for faster demo
        absolute_tolerance=1e-5
    )
    
    # Create multi-physics engine (disable mesh refinement for demo)
    mesh_refinement_config = MeshRefinementConfig(
        max_refinement_levels=0  # Disable mesh refinement
    )
    
    engine = MultiPhysicsCouplingEngine(
        domains=domains,
        coupling_terms=coupling_terms,
        time_step_config=time_step_config,
        convergence_config=convergence_config,
        mesh_refinement_config=mesh_refinement_config
    )
    
    # Initialize states with simple test case
    grid_size = 5  # Small grid for fast demo
    
    # Initial temperature field with hot spot
    temperature_initial = torch.ones(grid_size, grid_size) * 300.0
    temperature_initial[2, 2] = 400.0  # Hot spot in center
    
    # Initial displacement field (zero)
    displacement_initial = torch.zeros(grid_size, grid_size, 3)
    
    initial_states = {
        PhysicsDomain.THERMAL: temperature_initial,
        PhysicsDomain.MECHANICAL: displacement_initial
    }
    
    engine.initialize_states(initial_states)
    
    print(f"Initial setup:")
    print(f"  Grid size: {grid_size} x {grid_size}")
    print(f"  Temperature range: {temperature_initial.min():.1f} - {temperature_initial.max():.1f} K")
    print(f"  Physics domains: {len(domains)}")
    print(f"  Coupling terms: {len(coupling_terms)}")
    
    # Run simulation
    end_time = 0.01  # 10 milliseconds
    print(f"\nRunning simulation for {end_time} seconds...")
    
    results = engine.simulate(end_time, output_frequency=0.002)
    
    print(f"\nSimulation completed:")
    print(f"  Total steps: {results['total_steps']}")
    print(f"  Final time: {results['final_time']:.6f} s")
    print(f"  Average convergence iterations: {np.mean([c.iterations for c in results['convergence_history']]):.1f}")
    
    # Analyze final state
    final_temperature = results['states'][PhysicsDomain.THERMAL][-1]
    final_displacement = results['states'][PhysicsDomain.MECHANICAL][-1]
    
    print(f"\nFinal state analysis:")
    print(f"  Temperature range: {final_temperature.min():.1f} - {final_temperature.max():.1f} K")
    print(f"  Temperature change: {(final_temperature - temperature_initial).abs().max():.2f} K")
    print(f"  Max displacement: {torch.norm(final_displacement, dim=-1).max():.2e} m")
    
    # Get diagnostics
    diagnostics = engine.get_diagnostics()
    print(f"\nDiagnostics:")
    print(f"  Time step adaptations: {diagnostics['time_step_adaptations']}")
    print(f"  Stability violations: {diagnostics['stability_violations']}")
    print(f"  Domain state norms: {diagnostics['domain_states_norm']}")
    
    return results


def stability_analysis_demo():
    """
    Demonstration of stability analysis capabilities
    """
    print("\n" + "=" * 60)
    print("Stability Analysis Demo")
    print("=" * 60)
    
    # Create a simple thermal system
    domains = {
        PhysicsDomain.THERMAL: ThermalSolver(
            thermal_diffusivity=1e-5,  # Higher diffusivity for faster dynamics
            heat_capacity=1000.0
        )
    }
    
    # No coupling for this demo
    coupling_terms = []
    
    # Aggressive time stepping to test stability
    time_step_config = AdaptiveTimeStepConfig(
        initial_dt=1e-2,      # Large initial step
        min_dt=1e-8,
        max_dt=1e-1,
        safety_factor=0.5
    )
    
    convergence_config = ConvergenceConfig(
        max_iterations=10,
        absolute_tolerance=1e-6
    )
    
    mesh_refinement_config = MeshRefinementConfig(
        max_refinement_levels=0  # Disable mesh refinement
    )
    
    engine = MultiPhysicsCouplingEngine(
        domains=domains,
        coupling_terms=coupling_terms,
        time_step_config=time_step_config,
        convergence_config=convergence_config,
        mesh_refinement_config=mesh_refinement_config
    )
    
    # Initialize with sharp temperature gradient
    grid_size = 8
    temperature = torch.ones(grid_size, grid_size) * 300.0
    temperature[:, grid_size//2:] = 500.0  # Sharp step
    
    initial_states = {
        PhysicsDomain.THERMAL: temperature
    }
    
    engine.initialize_states(initial_states)
    
    print(f"Initial temperature profile: step function with {temperature.max() - temperature.min():.0f} K jump")
    
    # Run with stability monitoring
    stability_results = []
    time_steps = []
    
    current_time = 0.0
    max_time = 0.05
    step = 0
    
    print(f"\nRunning stability analysis...")
    
    while current_time < max_time and step < 50:  # Limit for demo
        # Analyze stability
        stability_result = engine.analyze_stability()
        stability_results.append(stability_result)
        time_steps.append(engine.current_dt)
        
        # Solve one step
        convergence_result = engine.solve_coupled_step(engine.current_dt)
        
        # Adapt time step
        engine.adapt_time_step(convergence_result, stability_result)
        
        current_time += engine.current_dt
        step += 1
        
        # Log every 10 steps
        if step % 10 == 0:
            print(f"  Step {step}: t={current_time:.4f}, dt={engine.current_dt:.2e}, "
                  f"stable={stability_result.is_stable}")
    
    print(f"\nStability analysis completed:")
    print(f"  Total steps: {step}")
    print(f"  Final time: {current_time:.4f} s")
    print(f"  Stability violations: {sum(1 for s in stability_results if not s.is_stable)}")
    print(f"  Time step range: {min(time_steps):.2e} - {max(time_steps):.2e} s")
    
    if stability_results:
        avg_margin = np.mean([s.stability_margin for s in stability_results])
        print(f"  Average stability margin: {avg_margin:.2e}")
    
    return {
        'stability_results': stability_results,
        'time_steps': time_steps,
        'final_time': current_time
    }


def multi_domain_demo():
    """
    Demonstration of multi-domain coupling
    """
    print("\n" + "=" * 60)
    print("Multi-Domain Coupling Demo")
    print("=" * 60)
    
    try:
        # Create multi-physics engine for magma flow
        engine = create_multi_physics_engine(
            geological_process=GeologicalProcess.MAGMA_FLOW,
            domain_configs={
                PhysicsDomain.THERMAL: {
                    'thermal_diffusivity': 1e-6,
                    'heat_capacity': 1200.0
                },
                PhysicsDomain.CHEMICAL: {
                    'diffusion_coefficient': 1e-9,
                    'reaction_rate': 1e-3
                }
            }
        )
        
        # Initialize states
        grid_size = 6  # Small for fast demo
        
        # Temperature field
        temperature = 1000.0 + 200.0 * torch.rand(grid_size, grid_size)
        
        # Chemical concentration
        concentration = 0.1 * torch.rand(grid_size, grid_size)
        
        initial_states = {
            PhysicsDomain.THERMAL: temperature,
            PhysicsDomain.CHEMICAL: concentration
        }
        
        engine.initialize_states(initial_states)
        
        print(f"Initialized multi-domain system:")
        print(f"  Domains: {len(engine.domains)}")
        print(f"  Coupling terms: {len(engine.coupling_terms)}")
        print(f"  Grid size: {grid_size} x {grid_size}")
        
        # Run simulation
        end_time = 0.02
        results = engine.simulate(end_time, output_frequency=0.005)
        
        print(f"\nMulti-domain simulation completed:")
        print(f"  Total steps: {results['total_steps']}")
        print(f"  Final time: {results['final_time']:.4f} s")
        
        # Get diagnostics
        diagnostics = engine.get_diagnostics()
        print(f"  Average convergence iterations: {diagnostics['average_convergence_iterations']:.1f}")
        print(f"  Active coupling terms: {diagnostics['coupling_terms_active']}")
        
        return results
        
    except Exception as e:
        print(f"Multi-domain demo failed: {e}")
        return None


def main():
    """
    Run all demonstrations
    """
    print("Multi-Physics Coupling Engine Demonstrations")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Demo 1: Simple thermal-mechanical coupling
        results1 = simple_thermal_mechanical_demo()
        
        # Demo 2: Stability analysis
        results2 = stability_analysis_demo()
        
        # Demo 3: Multi-domain coupling
        results3 = multi_domain_demo()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("=" * 60)
        
        return {
            'thermal_mechanical': results1,
            'stability_analysis': results2,
            'multi_domain': results3
        }
        
    except Exception as e:
        print(f"\nDemo execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()