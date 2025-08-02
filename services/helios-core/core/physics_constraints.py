"""
Physics Constraint Enforcement System

Comprehensive implementation of physics constraints for Physics-Informed Neural Networks (PINNs).
Implements Navier-Stokes equations, heat transfer equations, elastic deformation constraints,
and mass conservation laws with proper mathematical formulations and automatic differentiation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from models.ai_models import PhysicsConstraint, PhysicsConstraintType
from torch.autograd import grad

logger = structlog.get_logger()


@dataclass
class PhysicsConstraintResult:
    """Result of physics constraint evaluation"""

    constraint_type: PhysicsConstraintType
    residual: torch.Tensor
    residual_norm: float
    max_violation: float
    mean_violation: float
    constraint_satisfied: bool
    tolerance: float


class PhysicsConstraintSolver(ABC):
    """Abstract base class for physics constraint solvers"""

    @abstractmethod
    def compute_residual(
        self, inputs: torch.Tensor, outputs: torch.Tensor, constraint: PhysicsConstraint
    ) -> torch.Tensor:
        """Compute physics constraint residual"""

    @abstractmethod
    def get_required_derivatives(self) -> List[str]:
        """Get list of required derivatives for this constraint"""

    def evaluate_constraint(
        self, inputs: torch.Tensor, outputs: torch.Tensor, constraint: PhysicsConstraint
    ) -> PhysicsConstraintResult:
        """Evaluate constraint and return detailed results"""
        residual = self.compute_residual(inputs, outputs, constraint)

        # Compute constraint metrics
        residual_norm = torch.norm(residual).item()
        max_violation = torch.max(torch.abs(residual)).item()
        mean_violation = torch.mean(torch.abs(residual)).item()
        constraint_satisfied = max_violation <= constraint.tolerance

        return PhysicsConstraintResult(
            constraint_type=constraint.constraint_type,
            residual=residual,
            residual_norm=residual_norm,
            max_violation=max_violation,
            mean_violation=mean_violation,
            constraint_satisfied=constraint_satisfied,
            tolerance=constraint.tolerance,
        )

    def _compute_gradient(
        self, y: torch.Tensor, x: torch.Tensor, order: int = 1
    ) -> torch.Tensor:
        """
        Compute gradient dy/dx using automatic differentiation

        Args:
            y: Output tensor
            x: Input tensor
            order: Order of derivative (1 for first order, 2 for second order)

        Returns:
            Gradient tensor
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)

        try:
            # First order derivative
            grad_result = grad(
                outputs=y,
                inputs=x,
                grad_outputs=torch.ones_like(y),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]

            if grad_result is None:
                return torch.zeros_like(x)

            # Second order derivative if requested
            if order == 2:
                # Compute second derivative for each component
                second_derivatives = []
                for i in range(grad_result.shape[-1]):
                    grad_component = grad_result[..., i : i + 1]
                    second_grad = grad(
                        outputs=grad_component,
                        inputs=x,
                        grad_outputs=torch.ones_like(grad_component),
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]

                    if second_grad is None:
                        second_grad = torch.zeros_like(x)

                    second_derivatives.append(second_grad)

                return torch.stack(second_derivatives, dim=-1)

            return grad_result

        except RuntimeError as e:
            logger.warning(f"Gradient computation failed: {e}")
            return torch.zeros_like(x)

    def _compute_divergence(
        self, vector_field: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute divergence of a vector field

        Args:
            vector_field: Vector field tensor [batch_size, spatial_dims]
            coordinates: Coordinate tensor [batch_size, spatial_dims]

        Returns:
            Divergence tensor [batch_size, 1]
        """
        batch_size = vector_field.shape[0]
        spatial_dims = vector_field.shape[1]

        divergence = torch.zeros(
            batch_size, 1, device=vector_field.device, requires_grad=True
        )

        for i in range(spatial_dims):
            # Compute ∂u_i/∂x_i
            component = vector_field[:, i : i + 1]
            coord = coordinates[:, i : i + 1]

            partial_derivative = self._compute_gradient(component, coord)
            if partial_derivative.shape[-1] > i:
                divergence = divergence + partial_derivative[:, i : i + 1]

        return divergence

    def _compute_laplacian(
        self, scalar_field: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Laplacian (∇²) of a scalar field

        Args:
            scalar_field: Scalar field tensor [batch_size, 1]
            coordinates: Coordinate tensor [batch_size, spatial_dims]

        Returns:
            Laplacian tensor [batch_size, 1]
        """
        batch_size = scalar_field.shape[0]
        spatial_dims = coordinates.shape[1]

        laplacian = torch.zeros(
            batch_size, 1, device=scalar_field.device, requires_grad=True
        )

        for i in range(spatial_dims):
            coord = coordinates[:, i : i + 1]

            # First derivative
            first_grad = self._compute_gradient(scalar_field, coord)
            if first_grad.shape[-1] > i:
                # Second derivative
                second_grad = self._compute_gradient(first_grad[:, i : i + 1], coord)
                if second_grad.shape[-1] > i:
                    laplacian = laplacian + second_grad[:, i : i + 1]

        return laplacian


class NavierStokesConstraint(PhysicsConstraintSolver):
    """
    Navier-Stokes equations for fluid dynamics

    Incompressible Navier-Stokes equations:
    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
    ∇·u = 0 (continuity equation)

    Where:
    u = velocity vector [u, v, w]
    p = pressure
    ρ = density
    ν = kinematic viscosity (μ/ρ)
    f = body forces
    """

    def compute_residual(
        self, inputs: torch.Tensor, outputs: torch.Tensor, constraint: PhysicsConstraint
    ) -> torch.Tensor:
        """
        Compute Navier-Stokes residual

        Args:
            inputs: [batch_size, input_dim] - (x, y, z, t, ...)
            outputs: [batch_size, output_dim] - (u, v, w, p, ...)
            constraint: Physics constraint parameters

        Returns:
            residual: Physics constraint residual [batch_size, 4] (3 momentum + 1 continuity)
        """
        inputs.shape[0]
        inputs.device

        # Extract parameters
        params = constraint.equation_parameters
        viscosity = params.get("viscosity", 1e-3)  # Dynamic viscosity μ
        density = params.get("density", 1000.0)  # Density ρ
        kinematic_viscosity = viscosity / density  # ν = μ/ρ

        # Extract coordinates and time
        if inputs.shape[1] >= 4:
            x, y, z, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4]
            coordinates = inputs[:, :3]  # [x, y, z]
        else:
            # 2D case
            x, y, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
            z = torch.zeros_like(x)
            coordinates = inputs[:, :2]  # [x, y]

        # Extract velocity components and pressure
        if outputs.shape[1] >= 4:
            u, v, w, p = (
                outputs[:, 0:1],
                outputs[:, 1:2],
                outputs[:, 2:3],
                outputs[:, 3:4],
            )
            velocity = outputs[:, :3]  # [u, v, w]
        else:
            # 2D case
            u, v, p = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
            w = torch.zeros_like(u)
            velocity = outputs[:, :2]  # [u, v]

        # Initialize residuals
        momentum_residuals = []

        # Momentum equations for each component
        velocity_components = [u, v, w] if outputs.shape[1] >= 4 else [u, v]
        coord_components = [x, y, z] if inputs.shape[1] >= 4 else [x, y]

        for i, (vel_comp, coord_comp) in enumerate(
            zip(velocity_components, coord_components)
        ):
            # Time derivative: ∂u_i/∂t
            du_dt = self._compute_gradient(vel_comp, t)

            # Convective term: (u·∇)u_i
            convective_term = torch.zeros_like(vel_comp)
            for j, coord_j in enumerate(coord_components):
                du_dx = self._compute_gradient(vel_comp, coord_j)
                if j < len(velocity_components):
                    convective_term = convective_term + velocity_components[j] * du_dx

            # Pressure gradient: ∇p/ρ
            dp_dx = self._compute_gradient(p, coord_comp) / density

            # Viscous term: ν∇²u_i
            laplacian_u = self._compute_laplacian(vel_comp, coordinates)
            viscous_term = kinematic_viscosity * laplacian_u

            # Body forces (assume zero for now)
            body_force = torch.zeros_like(vel_comp)

            # Momentum equation residual: ∂u_i/∂t + (u·∇)u_i + ∇p/ρ - ν∇²u_i - f_i = 0
            momentum_residual = (
                du_dt + convective_term + dp_dx - viscous_term - body_force
            )
            momentum_residuals.append(momentum_residual)

        # Continuity equation: ∇·u = 0
        continuity_residual = self._compute_divergence(velocity, coordinates)

        # Combine all residuals
        all_residuals = momentum_residuals + [continuity_residual]
        residual = torch.cat(all_residuals, dim=1)

        return residual

    def get_required_derivatives(self) -> List[str]:
        """Get required derivatives for Navier-Stokes"""
        return ["first_order", "second_order"]


class HeatTransferConstraint(PhysicsConstraintSolver):
    """
    Heat transfer equation constraint

    Heat equation: ∂T/∂t = α∇²T + Q/(ρc)

    Where:
    T = temperature
    α = thermal diffusivity = k/(ρc)
    k = thermal conductivity
    ρ = density
    c = specific heat capacity
    Q = heat source term
    """

    def compute_residual(
        self, inputs: torch.Tensor, outputs: torch.Tensor, constraint: PhysicsConstraint
    ) -> torch.Tensor:
        """
        Compute heat transfer equation residual

        Args:
            inputs: [batch_size, input_dim] - (x, y, z, t, ...)
            outputs: [batch_size, output_dim] - (T, ...)
            constraint: Physics constraint parameters

        Returns:
            residual: Heat equation residual [batch_size, 1]
        """
        inputs.shape[0]
        inputs.device

        # Extract parameters
        params = constraint.equation_parameters
        thermal_conductivity = params.get("thermal_conductivity", 1.0)  # k
        specific_heat = params.get("specific_heat", 1000.0)  # c
        density = params.get("density", 1000.0)  # ρ
        heat_source = params.get("heat_source", 0.0)  # Q

        # Thermal diffusivity
        thermal_diffusivity = thermal_conductivity / (density * specific_heat)

        # Extract coordinates and time
        if inputs.shape[1] >= 4:
            coordinates = inputs[:, :3]  # [x, y, z]
            t = inputs[:, 3:4]
        else:
            coordinates = inputs[:, :2]  # [x, y]
            t = inputs[:, 2:3]

        # Extract temperature
        T = outputs[:, 0:1]

        # Time derivative: ∂T/∂t
        dT_dt = self._compute_gradient(T, t)

        # Laplacian: ∇²T
        laplacian_T = self._compute_laplacian(T, coordinates)

        # Heat source term
        source_term = torch.full_like(T, heat_source / (density * specific_heat))

        # Heat equation residual: ∂T/∂t - α∇²T - Q/(ρc) = 0
        residual = dT_dt - thermal_diffusivity * laplacian_T - source_term

        return residual

    def get_required_derivatives(self) -> List[str]:
        """Get required derivatives for heat transfer"""
        return ["first_order", "second_order"]


class ElasticDeformationConstraint(PhysicsConstraintSolver):
    """
    Elastic deformation constraint for crustal mechanics

    Navier's equations for linear elasticity:
    μ∇²u + (λ + μ)∇(∇·u) + f = 0

    Where:
    u = displacement vector [u_x, u_y, u_z]
    μ = shear modulus (Lamé's second parameter)
    λ = Lamé's first parameter
    f = body forces

    Lamé parameters in terms of Young's modulus E and Poisson's ratio ν:
    μ = E / (2(1 + ν))
    λ = Eν / ((1 + ν)(1 - 2ν))
    """

    def compute_residual(
        self, inputs: torch.Tensor, outputs: torch.Tensor, constraint: PhysicsConstraint
    ) -> torch.Tensor:
        """
        Compute elastic deformation residual using Navier's equations

        Args:
            inputs: [batch_size, input_dim] - (x, y, z, ...)
            outputs: [batch_size, output_dim] - (u_x, u_y, u_z, ...)
            constraint: Physics constraint parameters

        Returns:
            residual: Elastic equilibrium residual [batch_size, 3]
        """
        inputs.shape[0]
        inputs.device

        # Extract parameters
        params = constraint.equation_parameters
        youngs_modulus = params.get("youngs_modulus", 70e9)  # E (Pa)
        poisson_ratio = params.get("poisson_ratio", 0.3)  # ν

        # Compute Lamé parameters
        mu = youngs_modulus / (2 * (1 + poisson_ratio))  # Shear modulus
        lambda_param = (youngs_modulus * poisson_ratio) / (
            (1 + poisson_ratio) * (1 - 2 * poisson_ratio)
        )

        # Extract coordinates
        if inputs.shape[1] >= 3:
            coordinates = inputs[:, :3]  # [x, y, z]
        else:
            coordinates = inputs[:, :2]  # [x, y] for 2D case

        # Extract displacement components
        if outputs.shape[1] >= 3:
            u_x, u_y, u_z = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
            displacement = outputs[:, :3]
        else:
            # 2D case
            u_x, u_y = outputs[:, 0:1], outputs[:, 1:2]
            u_z = torch.zeros_like(u_x)
            displacement = outputs[:, :2]

        # Initialize residuals for each component
        residuals = []

        # Displacement components
        displacement_components = (
            [u_x, u_y, u_z] if outputs.shape[1] >= 3 else [u_x, u_y]
        )
        coord_components = coordinates.split(1, dim=1)

        for i, (u_i, x_i) in enumerate(zip(displacement_components, coord_components)):
            # Laplacian term: μ∇²u_i
            laplacian_u_i = self._compute_laplacian(u_i, coordinates)
            laplacian_term = mu * laplacian_u_i

            # Divergence term: (λ + μ)∂(∇·u)/∂x_i
            divergence_u = self._compute_divergence(displacement, coordinates)
            div_gradient = self._compute_gradient(divergence_u, x_i)
            divergence_term = (lambda_param + mu) * div_gradient

            # Body forces (assume zero for now, could be gravity or tectonic forces)
            body_force = torch.zeros_like(u_i)

            # Navier's equation residual: μ∇²u_i + (λ + μ)∂(∇·u)/∂x_i + f_i = 0
            residual_i = laplacian_term + divergence_term + body_force
            residuals.append(residual_i)

        # Combine residuals
        residual = torch.cat(residuals, dim=1)

        return residual

    def get_required_derivatives(self) -> List[str]:
        """Get required derivatives for elastic deformation"""
        return ["first_order", "second_order"]


class MassConservationConstraint(PhysicsConstraintSolver):
    """
    Mass conservation constraint for geochemical processes

    Advection-diffusion equation:
    ∂c/∂t + ∇·(cv) = D∇²c + S

    Where:
    c = concentration
    v = velocity field
    D = diffusion coefficient
    S = source/sink term
    """

    def compute_residual(
        self, inputs: torch.Tensor, outputs: torch.Tensor, constraint: PhysicsConstraint
    ) -> torch.Tensor:
        """
        Compute mass conservation residual

        Args:
            inputs: [batch_size, input_dim] - (x, y, z, t, ...)
            outputs: [batch_size, output_dim] - (c, v_x, v_y, v_z, ...)
            constraint: Physics constraint parameters

        Returns:
            residual: Mass conservation residual [batch_size, 1]
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        # Extract parameters
        params = constraint.equation_parameters
        diffusion_coefficient = params.get("diffusion_coefficient", 1e-9)  # D
        source_term = params.get("source_term", 0.0)  # S

        # Extract coordinates and time
        if inputs.shape[1] >= 4:
            coordinates = inputs[:, :3]  # [x, y, z]
            t = inputs[:, 3:4]
        else:
            coordinates = inputs[:, :2]  # [x, y]
            t = inputs[:, 2:3]

        # Extract concentration and velocity
        c = outputs[:, 0:1]  # Concentration

        # Velocity field (if provided in outputs, otherwise assume zero)
        if outputs.shape[1] >= 4:
            velocity = outputs[:, 1:4]  # [v_x, v_y, v_z]
        elif outputs.shape[1] >= 3:
            velocity = outputs[:, 1:3]  # [v_x, v_y] for 2D
        else:
            # No velocity field provided, assume zero velocity
            spatial_dims = coordinates.shape[1]
            velocity = torch.zeros(batch_size, spatial_dims, device=device)

        # Time derivative: ∂c/∂t
        dc_dt = self._compute_gradient(c, t)

        # Advection term: ∇·(cv)
        # First compute cv for each component
        concentration_velocity = c * velocity
        advection_term = self._compute_divergence(concentration_velocity, coordinates)

        # Diffusion term: D∇²c
        laplacian_c = self._compute_laplacian(c, coordinates)
        diffusion_term = diffusion_coefficient * laplacian_c

        # Source term
        source = torch.full_like(c, source_term)

        # Mass conservation residual: ∂c/∂t + ∇·(cv) - D∇²c - S = 0
        residual = dc_dt + advection_term - diffusion_term - source

        return residual

    def get_required_derivatives(self) -> List[str]:
        """Get required derivatives for mass conservation"""
        return ["first_order", "second_order"]


class PhysicsConstraintEnforcer:
    """
    Main physics constraint enforcement system

    Manages multiple physics constraints and provides unified interface
    for constraint evaluation and enforcement in PINN training.
    """

    def __init__(self):
        """Initialize the physics constraint enforcer"""
        self.solvers = {
            PhysicsConstraintType.NAVIER_STOKES: NavierStokesConstraint(),
            PhysicsConstraintType.HEAT_TRANSFER: HeatTransferConstraint(),
            PhysicsConstraintType.ELASTIC_DEFORMATION: ElasticDeformationConstraint(),
            PhysicsConstraintType.MASS_CONSERVATION: MassConservationConstraint(),
        }

        self.logger = structlog.get_logger()

    def register_solver(
        self, constraint_type: PhysicsConstraintType, solver: PhysicsConstraintSolver
    ):
        """Register a custom physics constraint solver"""
        self.solvers[constraint_type] = solver
        self.logger.info(f"Registered custom solver for {constraint_type}")

    def get_solver(
        self, constraint_type: PhysicsConstraintType
    ) -> PhysicsConstraintSolver:
        """Get physics constraint solver by type"""
        if constraint_type not in self.solvers:
            raise ValueError(f"Unsupported physics constraint type: {constraint_type}")
        return self.solvers[constraint_type]

    def enforce_constraints(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        constraints: List[PhysicsConstraint],
    ) -> Dict[str, PhysicsConstraintResult]:
        """
        Enforce multiple physics constraints

        Args:
            inputs: Input tensor (coordinates, time, etc.)
            outputs: Output tensor (predicted values)
            constraints: List of physics constraints to enforce

        Returns:
            Dictionary of constraint results keyed by constraint type
        """
        results = {}

        for constraint in constraints:
            try:
                solver = self.get_solver(constraint.constraint_type)
                result = solver.evaluate_constraint(inputs, outputs, constraint)
                results[constraint.constraint_type.value] = result

                # Log constraint satisfaction
                if result.constraint_satisfied:
                    self.logger.debug(
                        f"Constraint {constraint.constraint_type} satisfied "
                        f"(max violation: {result.max_violation:.2e})"
                    )
                else:
                    self.logger.warning(
                        f"Constraint {constraint.constraint_type} violated "
                        f"(max violation: {result.max_violation:.2e}, "
                        f"tolerance: {result.tolerance:.2e})"
                    )

            except Exception as e:
                self.logger.error(
                    f"Failed to enforce constraint {constraint.constraint_type}: {e}"
                )
                # Create a dummy result for failed constraints
                results[constraint.constraint_type.value] = PhysicsConstraintResult(
                    constraint_type=constraint.constraint_type,
                    residual=torch.zeros(inputs.shape[0], 1),
                    residual_norm=float("in"),
                    max_violation=float("in"),
                    mean_violation=float("in"),
                    constraint_satisfied=False,
                    tolerance=constraint.tolerance,
                )

        return results

    def compute_physics_loss(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        constraints: List[PhysicsConstraint],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-based loss for PINN training

        Args:
            inputs: Input tensor
            outputs: Output tensor
            constraints: List of physics constraints

        Returns:
            Total physics loss and individual constraint losses
        """
        constraint_results = self.enforce_constraints(inputs, outputs, constraints)

        total_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
        loss_components = {}

        for constraint in constraints:
            constraint_key = constraint.constraint_type.value

            if constraint_key in constraint_results:
                result = constraint_results[constraint_key]

                # Compute loss as mean squared residual
                constraint_loss = torch.mean(result.residual**2)

                # Weight the constraint loss
                weighted_loss = constraint.weight * constraint_loss
                total_loss = total_loss + weighted_loss

                loss_components[f"physics_{constraint_key}"] = constraint_loss.item()
                loss_components[f"weighted_physics_{constraint_key}"] = (
                    weighted_loss.item()
                )
            else:
                loss_components[f"physics_{constraint_key}"] = float("in")
                loss_components[f"weighted_physics_{constraint_key}"] = float("in")

        loss_components["total_physics_loss"] = total_loss.item()

        return total_loss, loss_components

    def validate_constraints(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        constraints: List[PhysicsConstraint],
    ) -> Dict[str, bool]:
        """
        Validate that all constraints are satisfied within tolerance

        Args:
            inputs: Input tensor
            outputs: Output tensor
            constraints: List of physics constraints

        Returns:
            Dictionary of constraint satisfaction status
        """
        constraint_results = self.enforce_constraints(inputs, outputs, constraints)

        validation_results = {}
        for constraint in constraints:
            constraint_key = constraint.constraint_type.value
            if constraint_key in constraint_results:
                result = constraint_results[constraint_key]
                validation_results[constraint_key] = result.constraint_satisfied
            else:
                validation_results[constraint_key] = False

        return validation_results

    def get_constraint_diagnostics(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        constraints: List[PhysicsConstraint],
    ) -> Dict[str, Dict[str, float]]:
        """
        Get detailed diagnostics for all constraints

        Args:
            inputs: Input tensor
            outputs: Output tensor
            constraints: List of physics constraints

        Returns:
            Dictionary of constraint diagnostics
        """
        constraint_results = self.enforce_constraints(inputs, outputs, constraints)

        diagnostics = {}
        for constraint in constraints:
            constraint_key = constraint.constraint_type.value
            if constraint_key in constraint_results:
                result = constraint_results[constraint_key]
                diagnostics[constraint_key] = {
                    "residual_norm": result.residual_norm,
                    "max_violation": result.max_violation,
                    "mean_violation": result.mean_violation,
                    "constraint_satisfied": result.constraint_satisfied,
                    "tolerance": result.tolerance,
                }
            else:
                diagnostics[constraint_key] = {
                    "residual_norm": float("in"),
                    "max_violation": float("in"),
                    "mean_violation": float("in"),
                    "constraint_satisfied": False,
                    "tolerance": constraint.tolerance,
                }

        return diagnostics


# Factory function for creating constraint enforcer
def create_physics_constraint_enforcer() -> PhysicsConstraintEnforcer:
    """Create and return a physics constraint enforcer instance"""
    return PhysicsConstraintEnforcer()


# Utility functions for creating common constraints
def create_navier_stokes_constraint(
    viscosity: float = 1e-3,
    density: float = 1000.0,
    weight: float = 1.0,
    tolerance: float = 1e-6,
) -> PhysicsConstraint:
    """Create a Navier-Stokes constraint with specified parameters"""
    return PhysicsConstraint(
        constraint_type=PhysicsConstraintType.NAVIER_STOKES,
        equation_parameters={"viscosity": viscosity, "density": density},
        weight=weight,
        tolerance=tolerance,
    )


def create_heat_transfer_constraint(
    thermal_conductivity: float = 1.0,
    specific_heat: float = 1000.0,
    density: float = 1000.0,
    heat_source: float = 0.0,
    weight: float = 1.0,
    tolerance: float = 1e-6,
) -> PhysicsConstraint:
    """Create a heat transfer constraint with specified parameters"""
    return PhysicsConstraint(
        constraint_type=PhysicsConstraintType.HEAT_TRANSFER,
        equation_parameters={
            "thermal_conductivity": thermal_conductivity,
            "specific_heat": specific_heat,
            "density": density,
            "heat_source": heat_source,
        },
        weight=weight,
        tolerance=tolerance,
    )


def create_elastic_deformation_constraint(
    youngs_modulus: float = 70e9,
    poisson_ratio: float = 0.3,
    weight: float = 1.0,
    tolerance: float = 1e-6,
) -> PhysicsConstraint:
    """Create an elastic deformation constraint with specified parameters"""
    return PhysicsConstraint(
        constraint_type=PhysicsConstraintType.ELASTIC_DEFORMATION,
        equation_parameters={
            "youngs_modulus": youngs_modulus,
            "poisson_ratio": poisson_ratio,
        },
        weight=weight,
        tolerance=tolerance,
    )


def create_mass_conservation_constraint(
    diffusion_coefficient: float = 1e-9,
    source_term: float = 0.0,
    weight: float = 1.0,
    tolerance: float = 1e-6,
) -> PhysicsConstraint:
    """Create a mass conservation constraint with specified parameters"""
    return PhysicsConstraint(
        constraint_type=PhysicsConstraintType.MASS_CONSERVATION,
        equation_parameters={
            "diffusion_coefficient": diffusion_coefficient,
            "source_term": source_term,
        },
        weight=weight,
        tolerance=tolerance,
    )
