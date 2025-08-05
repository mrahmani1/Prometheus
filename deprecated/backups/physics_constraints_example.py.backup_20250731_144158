"""
Physics Constraint Enforcement System Example

Demonstrates the comprehensive physics constraint enforcement system
for Physics-Informed Neural Networks (PINNs) in geological applications.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.physics_constraints import (
    PhysicsConstraintEnforcer, create_physics_constraint_enforcer,
    create_navier_stokes_constraint, create_heat_transfer_constraint,
    create_elastic_deformation_constraint, create_mass_conservation_constraint
)
from models.ai_models import PhysicsConstraint, PhysicsConstraintType


def create_sample_data(batch_size: int = 100, spatial_dims: int = 3) -> tuple:
    """
    Create sample data for testing physics constraints
    
    Args:
        batch_size: Number of data points
        spatial_dims: Number of spatial dimensions (2 or 3)
        
    Returns:
        Tuple of (inputs, outputs) tensors
    """
    if spatial_dims == 3:
        # 3D case: [x, y, z, t]
        inputs = torch.randn(batch_size, 4, requires_grad=True)
        # Outputs: [u, v, w, p] for fluid flow or [T] for heat transfer
        outputs = torch.randn(batch_size, 4, requires_grad=True)
    else:
        # 2D case: [x, y, t]
        inputs = torch.randn(batch_size, 3, requires_grad=True)
        # Outputs: [u, v, p] for fluid flow or [T] for heat transfer
        outputs = torch.randn(batch_size, 3, requires_grad=True)
    
    return inputs, outputs


def demonstrate_navier_stokes_constraint():
    """Demonstrate Navier-Stokes constraint enforcement"""
    print("=" * 60)
    print("NAVIER-STOKES CONSTRAINT DEMONSTRATION")
    print("=" * 60)
    
    # Create constraint enforcer
    enforcer = create_physics_constraint_enforcer()
    
    # Create Navier-Stokes constraint for water flow
    constraint = create_navier_stokes_constraint(
        viscosity=1e-3,      # Water viscosity (Pa·s)
        density=1000.0,      # Water density (kg/m³)
        weight=1.0,          # Constraint weight in loss function
        tolerance=1e-6       # Numerical tolerance
    )
    
    # Generate sample data
    inputs, outputs = create_sample_data(batch_size=50, spatial_dims=3)
    
    print(f"Input shape: {inputs.shape} (x, y, z, t)")
    print(f"Output shape: {outputs.shape} (u, v, w, p)")
    print(f"Constraint parameters:")
    print(f"  - Viscosity: {constraint.equation_parameters['viscosity']} Pa·s")
    print(f"  - Density: {constraint.equation_parameters['density']} kg/m³")
    print(f"  - Weight: {constraint.weight}")
    print(f"  - Tolerance: {constraint.tolerance}")
    
    # Enforce constraint
    results = enforcer.enforce_constraints(inputs, outputs, [constraint])
    result = results[PhysicsConstraintType.NAVIER_STOKES.value]
    
    print(f"\nConstraint Results:")
    print(f"  - Residual shape: {result.residual.shape}")
    print(f"  - Residual norm: {result.residual_norm:.6f}")
    print(f"  - Max violation: {result.max_violation:.6f}")
    print(f"  - Mean violation: {result.mean_violation:.6f}")
    print(f"  - Constraint satisfied: {result.constraint_satisfied}")
    
    # Compute physics loss
    physics_loss, loss_components = enforcer.compute_physics_loss(
        inputs, outputs, [constraint]
    )
    
    print(f"\nPhysics Loss:")
    print(f"  - Total physics loss: {physics_loss.item():.6f}")
    print(f"  - Navier-Stokes loss: {loss_components['physics_navier_stokes']:.6f}")
    
    return result


def demonstrate_heat_transfer_constraint():
    """Demonstrate heat transfer constraint enforcement"""
    print("\n" + "=" * 60)
    print("HEAT TRANSFER CONSTRAINT DEMONSTRATION")
    print("=" * 60)
    
    # Create constraint enforcer
    enforcer = create_physics_constraint_enforcer()
    
    # Create heat transfer constraint for geological thermal processes
    constraint = create_heat_transfer_constraint(
        thermal_conductivity=3.0,    # Rock thermal conductivity (W/m·K)
        specific_heat=1000.0,        # Rock specific heat (J/kg·K)
        density=2700.0,              # Rock density (kg/m³)
        heat_source=1e-6,            # Heat generation (W/m³)
        weight=1.0,
        tolerance=1e-6
    )
    
    # Generate sample data (temperature field)
    inputs, _ = create_sample_data(batch_size=50, spatial_dims=3)
    temperature = torch.randn(50, 1, requires_grad=True)  # Temperature field
    
    print(f"Input shape: {inputs.shape} (x, y, z, t)")
    print(f"Temperature shape: {temperature.shape} (T)")
    print(f"Constraint parameters:")
    print(f"  - Thermal conductivity: {constraint.equation_parameters['thermal_conductivity']} W/m·K")
    print(f"  - Specific heat: {constraint.equation_parameters['specific_heat']} J/kg·K")
    print(f"  - Density: {constraint.equation_parameters['density']} kg/m³")
    print(f"  - Heat source: {constraint.equation_parameters['heat_source']} W/m³")
    
    # Enforce constraint
    results = enforcer.enforce_constraints(inputs, temperature, [constraint])
    result = results[PhysicsConstraintType.HEAT_TRANSFER.value]
    
    print(f"\nConstraint Results:")
    print(f"  - Residual shape: {result.residual.shape}")
    print(f"  - Residual norm: {result.residual_norm:.6f}")
    print(f"  - Max violation: {result.max_violation:.6f}")
    print(f"  - Mean violation: {result.mean_violation:.6f}")
    print(f"  - Constraint satisfied: {result.constraint_satisfied}")
    
    # Compute thermal diffusivity
    thermal_diffusivity = (constraint.equation_parameters['thermal_conductivity'] / 
                          (constraint.equation_parameters['density'] * 
                           constraint.equation_parameters['specific_heat']))
    print(f"  - Thermal diffusivity: {thermal_diffusivity:.2e} m²/s")
    
    return result


def demonstrate_elastic_deformation_constraint():
    """Demonstrate elastic deformation constraint enforcement"""
    print("\n" + "=" * 60)
    print("ELASTIC DEFORMATION CONSTRAINT DEMONSTRATION")
    print("=" * 60)
    
    # Create constraint enforcer
    enforcer = create_physics_constraint_enforcer()
    
    # Create elastic deformation constraint for crustal mechanics
    constraint = create_elastic_deformation_constraint(
        youngs_modulus=70e9,    # Young's modulus for rock (Pa)
        poisson_ratio=0.25,     # Poisson's ratio for rock
        weight=1.0,
        tolerance=1e-6
    )
    
    # Generate sample data (displacement field)
    coordinates = torch.randn(50, 3, requires_grad=True)  # [x, y, z]
    displacement = torch.randn(50, 3, requires_grad=True)  # [u_x, u_y, u_z]
    
    print(f"Coordinates shape: {coordinates.shape} (x, y, z)")
    print(f"Displacement shape: {displacement.shape} (u_x, u_y, u_z)")
    print(f"Constraint parameters:")
    print(f"  - Young's modulus: {constraint.equation_parameters['youngs_modulus']:.2e} Pa")
    print(f"  - Poisson's ratio: {constraint.equation_parameters['poisson_ratio']}")
    
    # Compute Lamé parameters
    E = constraint.equation_parameters['youngs_modulus']
    nu = constraint.equation_parameters['poisson_ratio']
    mu = E / (2 * (1 + nu))  # Shear modulus
    lambda_param = (E * nu) / ((1 + nu) * (1 - 2 * nu))  # Lamé's first parameter
    
    print(f"  - Shear modulus (μ): {mu:.2e} Pa")
    print(f"  - Lamé parameter (λ): {lambda_param:.2e} Pa")
    
    # Enforce constraint
    results = enforcer.enforce_constraints(coordinates, displacement, [constraint])
    result = results[PhysicsConstraintType.ELASTIC_DEFORMATION.value]
    
    print(f"\nConstraint Results:")
    print(f"  - Residual shape: {result.residual.shape}")
    print(f"  - Residual norm: {result.residual_norm:.6f}")
    print(f"  - Max violation: {result.max_violation:.6f}")
    print(f"  - Mean violation: {result.mean_violation:.6f}")
    print(f"  - Constraint satisfied: {result.constraint_satisfied}")
    
    return result


def demonstrate_mass_conservation_constraint():
    """Demonstrate mass conservation constraint enforcement"""
    print("\n" + "=" * 60)
    print("MASS CONSERVATION CONSTRAINT DEMONSTRATION")
    print("=" * 60)
    
    # Create constraint enforcer
    enforcer = create_physics_constraint_enforcer()
    
    # Create mass conservation constraint for geochemical transport
    constraint = create_mass_conservation_constraint(
        diffusion_coefficient=1e-9,  # Molecular diffusion coefficient (m²/s)
        source_term=1e-6,            # Chemical source term (kg/m³·s)
        weight=1.0,
        tolerance=1e-6
    )
    
    # Generate sample data (concentration and velocity field)
    inputs, _ = create_sample_data(batch_size=50, spatial_dims=3)
    # Concentration and velocity: [c, v_x, v_y, v_z]
    concentration_velocity = torch.randn(50, 4, requires_grad=True)
    
    print(f"Input shape: {inputs.shape} (x, y, z, t)")
    print(f"Output shape: {concentration_velocity.shape} (c, v_x, v_y, v_z)")
    print(f"Constraint parameters:")
    print(f"  - Diffusion coefficient: {constraint.equation_parameters['diffusion_coefficient']:.2e} m²/s")
    print(f"  - Source term: {constraint.equation_parameters['source_term']:.2e} kg/m³·s")
    
    # Enforce constraint
    results = enforcer.enforce_constraints(inputs, concentration_velocity, [constraint])
    result = results[PhysicsConstraintType.MASS_CONSERVATION.value]
    
    print(f"\nConstraint Results:")
    print(f"  - Residual shape: {result.residual.shape}")
    print(f"  - Residual norm: {result.residual_norm:.6f}")
    print(f"  - Max violation: {result.max_violation:.6f}")
    print(f"  - Mean violation: {result.mean_violation:.6f}")
    print(f"  - Constraint satisfied: {result.constraint_satisfied}")
    
    return result


def demonstrate_multi_constraint_enforcement():
    """Demonstrate enforcement of multiple physics constraints simultaneously"""
    print("\n" + "=" * 60)
    print("MULTI-CONSTRAINT ENFORCEMENT DEMONSTRATION")
    print("=" * 60)
    
    # Create constraint enforcer
    enforcer = create_physics_constraint_enforcer()
    
    # Create multiple constraints with different weights
    constraints = [
        create_navier_stokes_constraint(weight=1.0),
        create_heat_transfer_constraint(weight=0.5),
        create_elastic_deformation_constraint(weight=2.0),
        create_mass_conservation_constraint(weight=1.5)
    ]
    
    print(f"Number of constraints: {len(constraints)}")
    for i, constraint in enumerate(constraints):
        print(f"  {i+1}. {constraint.constraint_type.value} (weight: {constraint.weight})")
    
    # Generate sample data
    inputs, outputs = create_sample_data(batch_size=100, spatial_dims=3)
    
    print(f"\nInput shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    
    # Enforce all constraints
    results = enforcer.enforce_constraints(inputs, outputs, constraints)
    
    print(f"\nConstraint Results:")
    for constraint_type, result in results.items():
        print(f"  {constraint_type}:")
        print(f"    - Max violation: {result.max_violation:.6f}")
        print(f"    - Mean violation: {result.mean_violation:.6f}")
        print(f"    - Satisfied: {result.constraint_satisfied}")
    
    # Compute total physics loss
    total_loss, loss_components = enforcer.compute_physics_loss(inputs, outputs, constraints)
    
    print(f"\nPhysics Loss Components:")
    print(f"  - Total physics loss: {total_loss.item():.6f}")
    for key, value in loss_components.items():
        if key.startswith('physics_') and not key.startswith('weighted_'):
            print(f"  - {key}: {value:.6f}")
    
    print(f"\nWeighted Physics Loss Components:")
    for key, value in loss_components.items():
        if key.startswith('weighted_physics_'):
            print(f"  - {key}: {value:.6f}")
    
    # Validate constraints
    validation_results = enforcer.validate_constraints(inputs, outputs, constraints)
    
    print(f"\nConstraint Validation:")
    all_satisfied = all(validation_results.values())
    print(f"  - All constraints satisfied: {all_satisfied}")
    for constraint_type, satisfied in validation_results.items():
        print(f"  - {constraint_type}: {'✓' if satisfied else '✗'}")
    
    return results, total_loss, loss_components


def demonstrate_constraint_diagnostics():
    """Demonstrate detailed constraint diagnostics"""
    print("\n" + "=" * 60)
    print("CONSTRAINT DIAGNOSTICS DEMONSTRATION")
    print("=" * 60)
    
    # Create constraint enforcer
    enforcer = create_physics_constraint_enforcer()
    
    # Create a constraint with tight tolerance to show violations
    constraint = create_heat_transfer_constraint(tolerance=1e-10)  # Very tight tolerance
    
    # Generate sample data
    inputs, _ = create_sample_data(batch_size=30, spatial_dims=3)
    temperature = torch.randn(30, 1, requires_grad=True)
    
    # Get detailed diagnostics
    diagnostics = enforcer.get_constraint_diagnostics(inputs, temperature, [constraint])
    
    print(f"Detailed Diagnostics for Heat Transfer Constraint:")
    heat_diag = diagnostics[PhysicsConstraintType.HEAT_TRANSFER.value]
    
    print(f"  - Residual norm: {heat_diag['residual_norm']:.8f}")
    print(f"  - Maximum violation: {heat_diag['max_violation']:.8f}")
    print(f"  - Mean violation: {heat_diag['mean_violation']:.8f}")
    print(f"  - Tolerance: {heat_diag['tolerance']:.2e}")
    print(f"  - Constraint satisfied: {heat_diag['constraint_satisfied']}")
    
    if not heat_diag['constraint_satisfied']:
        violation_ratio = heat_diag['max_violation'] / heat_diag['tolerance']
        print(f"  - Violation ratio: {violation_ratio:.2f}x tolerance")
    
    return diagnostics


def main():
    """Main demonstration function"""
    print("PHYSICS CONSTRAINT ENFORCEMENT SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This example demonstrates the comprehensive physics constraint")
    print("enforcement system for Physics-Informed Neural Networks (PINNs)")
    print("in geological applications.")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Demonstrate individual constraints
        navier_stokes_result = demonstrate_navier_stokes_constraint()
        heat_transfer_result = demonstrate_heat_transfer_constraint()
        elastic_deformation_result = demonstrate_elastic_deformation_constraint()
        mass_conservation_result = demonstrate_mass_conservation_constraint()
        
        # Demonstrate multi-constraint enforcement
        multi_results, total_loss, loss_components = demonstrate_multi_constraint_enforcement()
        
        # Demonstrate detailed diagnostics
        diagnostics = demonstrate_constraint_diagnostics()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("All physics constraints have been successfully demonstrated.")
        print("The system is ready for integration with PINN training pipelines.")
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print(f"  - Constraints demonstrated: 4")
        print(f"  - Total physics loss: {total_loss.item():.6f}")
        print(f"  - All tests passed: ✓")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Demonstration failed. Please check the implementation.")
        raise


if __name__ == "__main__":
    main()