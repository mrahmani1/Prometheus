# Physics Constraint Enforcement System Documentation

## Overview

The Physics Constraint Enforcement System is a comprehensive implementation of physics-based constraints for Physics-Informed Neural Networks (PINNs) in the Prometheus AI geodynamic platform. This system enforces fundamental physical laws during neural network training to ensure that predictions respect the underlying physics of geological processes.

## Architecture

### Core Components

1. **PhysicsConstraintEnforcer**: Main orchestrator for constraint enforcement
2. **PhysicsConstraintSolver**: Abstract base class for individual constraint implementations
3. **Constraint Implementations**: Specific solvers for different physics equations
4. **Factory Functions**: Convenient constructors for common constraint configurations

### Supported Physics Constraints

#### 1. Navier-Stokes Equations (Fluid Dynamics)

**Mathematical Formulation:**
```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
∇·u = 0 (continuity equation)
```

**Parameters:**
- `viscosity`: Dynamic viscosity μ (Pa·s)
- `density`: Fluid density ρ (kg/m³)

**Applications:**
- Magma flow modeling
- Hydrothermal fluid circulation
- Groundwater flow

#### 2. Heat Transfer Equation

**Mathematical Formulation:**
```
∂T/∂t = α∇²T + Q/(ρc)
```
where α = k/(ρc) is thermal diffusivity

**Parameters:**
- `thermal_conductivity`: Thermal conductivity k (W/m·K)
- `specific_heat`: Specific heat capacity c (J/kg·K)
- `density`: Material density ρ (kg/m³)
- `heat_source`: Heat generation Q (W/m³)

**Applications:**
- Geothermal modeling
- Volcanic thermal processes
- Crustal heat flow

#### 3. Elastic Deformation (Crustal Mechanics)

**Mathematical Formulation:**
```
μ∇²u + (λ + μ)∇(∇·u) + f = 0
```
where μ and λ are Lamé parameters

**Parameters:**
- `youngs_modulus`: Young's modulus E (Pa)
- `poisson_ratio`: Poisson's ratio ν

**Applications:**
- Crustal deformation modeling
- Earthquake stress analysis
- Tectonic plate mechanics

#### 4. Mass Conservation (Geochemical Transport)

**Mathematical Formulation:**
```
∂c/∂t + ∇·(cv) = D∇²c + S
```

**Parameters:**
- `diffusion_coefficient`: Diffusion coefficient D (m²/s)
- `source_term`: Source/sink term S (kg/m³·s)

**Applications:**
- Geochemical transport
- Contaminant migration
- Mineral dissolution/precipitation

## Usage Guide

### Basic Usage

```python
from core.physics_constraints import (
    create_physics_constraint_enforcer,
    create_navier_stokes_constraint,
    create_heat_transfer_constraint
)

# Create constraint enforcer
enforcer = create_physics_constraint_enforcer()

# Create constraints
navier_stokes = create_navier_stokes_constraint(
    viscosity=1e-3,
    density=1000.0,
    weight=1.0
)

heat_transfer = create_heat_transfer_constraint(
    thermal_conductivity=3.0,
    specific_heat=1000.0,
    density=2700.0
)

constraints = [navier_stokes, heat_transfer]

# Enforce constraints
results = enforcer.enforce_constraints(inputs, outputs, constraints)

# Compute physics loss for training
physics_loss, loss_components = enforcer.compute_physics_loss(
    inputs, outputs, constraints
)
```

### Integration with PINN Training

```python
from core.pinn_engine import PINNLoss

# Create PINN loss function with physics constraints
loss_fn = PINNLoss(constraints)

# During training
predictions = model(inputs)
total_loss, loss_components = loss_fn(predictions, targets, inputs, model)

# Backpropagate
total_loss.backward()
optimizer.step()
```

### Advanced Configuration

#### Custom Constraint Parameters

```python
# Geological-specific parameters
magma_flow_constraint = create_navier_stokes_constraint(
    viscosity=1e6,      # High viscosity for magma (Pa·s)
    density=2500.0,     # Magma density (kg/m³)
    weight=2.0,         # Higher weight for critical constraint
    tolerance=1e-5      # Relaxed tolerance for complex flow
)

# Rock thermal properties
crustal_heat_constraint = create_heat_transfer_constraint(
    thermal_conductivity=2.5,    # Typical crustal rock (W/m·K)
    specific_heat=1000.0,        # Rock specific heat (J/kg·K)
    density=2700.0,              # Crustal density (kg/m³)
    heat_source=1e-6,            # Radiogenic heating (W/m³)
    weight=1.0
)
```

#### Multi-Scale Constraints

```python
# Different constraints for different scales
constraints = [
    # Regional scale: crustal deformation
    create_elastic_deformation_constraint(
        youngs_modulus=70e9,
        poisson_ratio=0.25,
        weight=1.0
    ),
    
    # Local scale: fluid flow
    create_navier_stokes_constraint(
        viscosity=1e-3,
        density=1000.0,
        weight=0.5  # Lower weight for local effects
    ),
    
    # Chemical transport
    create_mass_conservation_constraint(
        diffusion_coefficient=1e-9,
        source_term=0.0,
        weight=1.5
    )
]
```

## API Reference

### PhysicsConstraintEnforcer

#### Methods

- `enforce_constraints(inputs, outputs, constraints)`: Enforce multiple constraints
- `compute_physics_loss(inputs, outputs, constraints)`: Compute physics-based loss
- `validate_constraints(inputs, outputs, constraints)`: Check constraint satisfaction
- `get_constraint_diagnostics(inputs, outputs, constraints)`: Detailed diagnostics
- `register_solver(constraint_type, solver)`: Register custom constraint solver

#### Returns

- `PhysicsConstraintResult`: Detailed constraint evaluation results
- `Dict[str, float]`: Loss components for training
- `Dict[str, bool]`: Constraint validation status
- `Dict[str, Dict[str, float]]`: Detailed diagnostics

### Factory Functions

#### create_navier_stokes_constraint
```python
create_navier_stokes_constraint(
    viscosity: float = 1e-3,
    density: float = 1000.0,
    weight: float = 1.0,
    tolerance: float = 1e-6
) -> PhysicsConstraint
```

#### create_heat_transfer_constraint
```python
create_heat_transfer_constraint(
    thermal_conductivity: float = 1.0,
    specific_heat: float = 1000.0,
    density: float = 1000.0,
    heat_source: float = 0.0,
    weight: float = 1.0,
    tolerance: float = 1e-6
) -> PhysicsConstraint
```

#### create_elastic_deformation_constraint
```python
create_elastic_deformation_constraint(
    youngs_modulus: float = 70e9,
    poisson_ratio: float = 0.3,
    weight: float = 1.0,
    tolerance: float = 1e-6
) -> PhysicsConstraint
```

#### create_mass_conservation_constraint
```python
create_mass_conservation_constraint(
    diffusion_coefficient: float = 1e-9,
    source_term: float = 0.0,
    weight: float = 1.0,
    tolerance: float = 1e-6
) -> PhysicsConstraint
```

## Implementation Details

### Automatic Differentiation

The system uses PyTorch's automatic differentiation to compute:
- First-order derivatives: ∂u/∂x, ∂u/∂t
- Second-order derivatives: ∂²u/∂x²
- Divergence: ∇·u
- Laplacian: ∇²u

### Gradient Computation

```python
def _compute_gradient(self, y: torch.Tensor, x: torch.Tensor, order: int = 1):
    """Compute gradient using automatic differentiation"""
    grad_result = grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    return grad_result
```

### Error Handling

The system includes robust error handling:
- Graceful degradation when gradient computation fails
- Fallback to zero residuals for failed constraints
- Detailed logging of constraint violations
- Validation of constraint parameters

### Performance Considerations

- Efficient batch processing of constraints
- Minimal memory overhead for gradient computation
- Optimized tensor operations for large-scale problems
- Support for both 2D and 3D spatial domains

## Testing

### Unit Tests

The system includes comprehensive unit tests:
- Individual constraint solver tests
- Multi-constraint enforcement tests
- Gradient computation validation
- Error handling verification

### Running Tests

```bash
cd services/helios-core
python -m pytest tests/test_physics_constraints.py -v
```

### Example Demonstrations

Run the comprehensive example:
```bash
python examples/physics_constraints_example.py
```

## Geological Applications

### Volcanic Systems

```python
# Magma chamber modeling
magma_constraints = [
    create_navier_stokes_constraint(
        viscosity=1e6,      # Magma viscosity
        density=2500.0,     # Magma density
        weight=2.0
    ),
    create_heat_transfer_constraint(
        thermal_conductivity=1.5,
        specific_heat=1200.0,
        density=2500.0,
        heat_source=1e-3,   # Magma cooling
        weight=1.5
    )
]
```

### Earthquake Modeling

```python
# Crustal stress analysis
earthquake_constraints = [
    create_elastic_deformation_constraint(
        youngs_modulus=80e9,    # Crustal rock
        poisson_ratio=0.25,
        weight=2.0
    ),
    create_mass_conservation_constraint(
        diffusion_coefficient=1e-12,  # Pore fluid
        source_term=0.0,
        weight=1.0
    )
]
```

### Geothermal Systems

```python
# Hydrothermal circulation
geothermal_constraints = [
    create_heat_transfer_constraint(
        thermal_conductivity=2.5,
        specific_heat=1000.0,
        density=2700.0,
        heat_source=5e-6,   # Geothermal gradient
        weight=1.0
    ),
    create_navier_stokes_constraint(
        viscosity=1e-3,     # Water viscosity
        density=1000.0,
        weight=1.5
    )
]
```

## Best Practices

### Constraint Weighting

1. **Critical constraints**: Use higher weights (2.0-5.0) for fundamental physics
2. **Secondary effects**: Use lower weights (0.1-0.5) for minor corrections
3. **Scale-dependent**: Adjust weights based on spatial/temporal scales

### Tolerance Settings

1. **Tight tolerance** (1e-8): For well-conditioned problems
2. **Standard tolerance** (1e-6): For most geological applications
3. **Relaxed tolerance** (1e-4): For complex, multi-scale problems

### Performance Optimization

1. **Batch processing**: Process multiple constraints simultaneously
2. **Gradient caching**: Reuse computed gradients when possible
3. **Selective enforcement**: Apply constraints only where needed
4. **Adaptive weighting**: Adjust weights during training

## Troubleshooting

### Common Issues

1. **Gradient computation failures**: Check tensor requires_grad settings
2. **Constraint violations**: Adjust tolerance or constraint weights
3. **Training instability**: Balance physics and data loss weights
4. **Memory issues**: Reduce batch size or use gradient checkpointing

### Debugging Tools

```python
# Get detailed diagnostics
diagnostics = enforcer.get_constraint_diagnostics(inputs, outputs, constraints)

# Check constraint satisfaction
validation = enforcer.validate_constraints(inputs, outputs, constraints)

# Monitor loss components
_, loss_components = enforcer.compute_physics_loss(inputs, outputs, constraints)
```

## Future Extensions

### Planned Features

1. **Additional constraints**: Wave equations, Maxwell's equations
2. **Adaptive constraints**: Dynamic constraint weighting
3. **Multi-physics coupling**: Coupled constraint systems
4. **Uncertainty quantification**: Constraint uncertainty propagation

### Custom Constraint Development

```python
class CustomConstraint(PhysicsConstraintSolver):
    def compute_residual(self, inputs, outputs, constraint):
        # Implement custom physics equation
        return residual
    
    def get_required_derivatives(self):
        return ["first_order", "second_order"]

# Register custom constraint
enforcer.register_solver(custom_type, CustomConstraint())
```

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440.

3. Cuomo, S., Di Cola, V. S., Giampaolo, F., Rozza, G., Raissi, M., & Piccialli, F. (2022). Scientific machine learning through physics–informed neural networks: Where we are and what's next. Journal of Scientific Computing, 92(3), 1-62.