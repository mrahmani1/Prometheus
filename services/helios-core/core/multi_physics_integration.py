"""
Multi-Physics Integration Module

Integrates the advanced multi-physics coupling engine with the existing Helios AI system,
providing seamless integration between Physics-Informed Neural Networks (PINNs) and
multi-physics simulations for geological processes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import structlog
import torch
from models.ai_models import PINNModelConfig

from .multi_physics_coupling import (
    MultiPhysicsCouplingEngine,
    PhysicsDomain,
    create_multi_physics_engine,
)
from .pinn_engine import GeologicalProcess, PINNEngine

logger = structlog.get_logger()

class IntegrationMode(str, Enum):
    """Integration modes between PINN and multi-physics"""

    PINN_DRIVEN = "pinn_driven"  # PINN provides initial conditions and constraints
    PHYSICS_DRIVEN = "physics_driven"  # Multi-physics provides training data for PINN
    COUPLED = "coupled"  # Bidirectional coupling during simulation
    HYBRID = "hybrid"  # Switch between modes based on conditions

@dataclass
class IntegrationConfig:
    """Configuration for PINN-MultiPhysics integration"""

    mode: IntegrationMode = IntegrationMode.COUPLED
    pinn_update_frequency: int = 10  # Update PINN every N physics steps
    physics_update_frequency: int = 1  # Update physics every N PINN steps
    coupling_strength: float = 1.0  # Strength of bidirectional coupling
    convergence_threshold: float = 1e-6  # Convergence threshold for coupled system
    max_coupling_iterations: int = 100  # Maximum coupling iterations
    use_adaptive_coupling: bool = True  # Use adaptive coupling strength
    physics_weight: float = 0.5  # Weight of physics vs PINN in hybrid mode
    pinn_weight: float = 0.5  # Weight of PINN vs physics in hybrid mode

class PINNMultiPhysicsIntegrator:
    """
    Integrates PINN models with multi-physics coupling engine
    """

    def __init__(
        self,
        pinn_engine: PINNEngine,
        multi_physics_engine: MultiPhysicsCouplingEngine,
        integration_config: IntegrationConfig,
        geological_process: GeologicalProcess,
    ):
        self.pinn_engine = pinn_engine
        self.multi_physics_engine = multi_physics_engine
        self.integration_config = integration_config
        self.geological_process = geological_process

        # Integration state
        self.coupling_history = []
        self.convergence_history = []
        self.current_coupling_strength = integration_config.coupling_strength

        # Model mappings between PINN outputs and physics domains
        self.domain_mappings = self._create_domain_mappings()

        logger.info(
            f"PINN-MultiPhysics integrator initialized for {geological_process.value}"
        )

    def _create_domain_mappings(self) -> Dict[PhysicsDomain, Dict[str, Any]]:
        """Create mappings between PINN outputs and physics domains"""
        mappings = {}

        # Default mappings based on geological process
        if self.geological_process == GeologicalProcess.MAGMA_FLOW:
            mappings = {
                PhysicsDomain.THERMAL: {
                    "pinn_output_indices": [0],  # Temperature
                    "physics_state_shape": (50, 50),
                    "scaling_factor": 1.0,
                    "offset": 0.0,
                },
                PhysicsDomain.FLUID_DYNAMICS: {
                    "pinn_output_indices": [1, 2, 3],  # Velocity components
                    "physics_state_shape": (50, 50, 3),
                    "scaling_factor": 1.0,
                    "offset": 0.0,
                },
                PhysicsDomain.CHEMICAL: {
                    "pinn_output_indices": [4],  # Composition
                    "physics_state_shape": (50, 50),
                    "scaling_factor": 1.0,
                    "offset": 0.0,
                },
            }
        elif self.geological_process == GeologicalProcess.CRUSTAL_DEFORMATION:
            mappings = {
                PhysicsDomain.MECHANICAL: {
                    "pinn_output_indices": [0, 1, 2],  # Displacement components
                    "physics_state_shape": (50, 50, 3),
                    "scaling_factor": 1.0,
                    "offset": 0.0,
                },
                PhysicsDomain.THERMAL: {
                    "pinn_output_indices": [3],  # Temperature
                    "physics_state_shape": (50, 50),
                    "scaling_factor": 1.0,
                    "offset": 0.0,
                },
            }
        elif self.geological_process == GeologicalProcess.HEAT_TRANSFER:
            mappings = {
                PhysicsDomain.THERMAL: {
                    "pinn_output_indices": [0],  # Temperature
                    "physics_state_shape": (50, 50),
                    "scaling_factor": 1.0,
                    "offset": 0.0,
                }
            }
        else:
            # Generic mapping for other processes
            mappings = {
                PhysicsDomain.THERMAL: {
                    "pinn_output_indices": [0],
                    "physics_state_shape": (50, 50),
                    "scaling_factor": 1.0,
                    "offset": 0.0,
                }
            }

        return mappings

    def pinn_to_physics_state(
        self, pinn_output: torch.Tensor, domain: PhysicsDomain
    ) -> torch.Tensor:
        """Convert PINN output to physics domain state"""
        if domain not in self.domain_mappings:
            raise ValueError(f"No mapping defined for domain {domain}")

        mapping = self.domain_mappings[domain]
        indices = mapping["pinn_output_indices"]
        target_shape = mapping["physics_state_shape"]
        scaling = mapping["scaling_factor"]
        offset = mapping["offset"]

        # Extract relevant outputs
        relevant_output = pinn_output[:, indices]

        # Apply scaling and offset
        scaled_output = relevant_output * scaling + offset

        # Reshape to physics domain shape
        batch_size = scaled_output.shape[0]
        if len(target_shape) == 2:
            # 2D field
            physics_state = scaled_output.view(batch_size, *target_shape)
        elif len(target_shape) == 3:
            # 3D field or 2D vector field
            physics_state = scaled_output.view(batch_size, *target_shape)
        else:
            # Keep original shape
            physics_state = scaled_output

        return physics_state

    def physics_to_pinn_data(
        self,
        physics_states: Dict[PhysicsDomain, torch.Tensor],
        coordinates: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert physics states to PINN training data"""
        # Flatten physics states and combine
        combined_outputs = []

        for domain in sorted(physics_states.keys(), key=lambda x: x.value):
            if domain in self.domain_mappings:
                state = physics_states[domain]
                # Flatten spatial dimensions
                flattened_state = state.view(state.shape[0], -1)
                combined_outputs.append(flattened_state)

        if combined_outputs:
            targets = torch.cat(combined_outputs, dim=1)
        else:
            targets = torch.zeros(coordinates.shape[0], 1)

        return coordinates, targets

    def solve_coupled_step(
        self, model_id: str, coordinates: torch.Tensor, dt: float
    ) -> Dict[str, Any]:
        """
        Solve one coupled PINN-MultiPhysics step
        """
        results = {
            "pinn_predictions": None,
            "physics_states": None,
            "coupling_residual": 0.0,
            "converged": False,
            "iterations": 0,
        }

        iteration = 0
        converged = False
        coupling_residual = float("in")

        while (
            iteration < self.integration_config.max_coupling_iterations
            and not converged
        ):
            # PINN prediction step
            if self.integration_config.mode in [
                IntegrationMode.PINN_DRIVEN,
                IntegrationMode.COUPLED,
            ]:
                pinn_predictions = self.pinn_engine.predict(model_id, coordinates)
                results["pinn_predictions"] = pinn_predictions

                # Convert PINN predictions to physics initial conditions
                physics_initial_states = {}
                for domain in self.multi_physics_engine.domains:
                    if domain in self.domain_mappings:
                        physics_state = self.pinn_to_physics_state(
                            pinn_predictions, domain
                        )
                        physics_initial_states[domain] = physics_state.mean(
                            dim=0
                        )  # Average over batch

                # Update physics engine states
                if physics_initial_states:
                    self.multi_physics_engine.initialize_states(physics_initial_states)

            # Multi-physics step
            if self.integration_config.mode in [
                IntegrationMode.PHYSICS_DRIVEN,
                IntegrationMode.COUPLED,
            ]:
                self.multi_physics_engine.solve_coupled_step(dt)
                results["physics_states"] = dict(
                    self.multi_physics_engine.domain_states
                )

                # Update PINN training data if in coupled mode
                if self.integration_config.mode == IntegrationMode.COUPLED:
                    physics_coords, physics_targets = self.physics_to_pinn_data(
                        self.multi_physics_engine.domain_states, coordinates
                    )

                    # Compute coupling residual
                    if results["pinn_predictions"] is not None:
                        pinn_flat = results["pinn_predictions"].view(-1)
                        physics_flat = physics_targets.view(-1)

                        # Ensure same size for comparison
                        min_size = min(pinn_flat.shape[0], physics_flat.shape[0])
                        coupling_residual = torch.norm(
                            pinn_flat[:min_size] - physics_flat[:min_size]
                        ).item()

                        results["coupling_residual"] = coupling_residual

            # Check convergence
            if coupling_residual < self.integration_config.convergence_threshold:
                converged = True
                results["converged"] = True

            iteration += 1
            results["iterations"] = iteration

            # Adaptive coupling strength
            if self.integration_config.use_adaptive_coupling:
                self._adapt_coupling_strength(coupling_residual)

        # Store coupling history
        self.coupling_history.append(
            {
                "iteration": len(self.coupling_history),
                "coupling_residual": coupling_residual,
                "converged": converged,
                "iterations": iteration,
                "coupling_strength": self.current_coupling_strength,
            }
        )

        return results

    def _adapt_coupling_strength(self, coupling_residual: float):
        """Adapt coupling strength based on residual"""
        if coupling_residual > 1e-3:
            # Increase coupling for large residuals
            self.current_coupling_strength = min(
                2.0, self.current_coupling_strength * 1.1
            )
        elif coupling_residual < 1e-6:
            # Decrease coupling for small residuals
            self.current_coupling_strength = max(
                0.1, self.current_coupling_strength * 0.9
            )

    def train_coupled_system(
        self,
        model_id: str,
        coordinates: torch.Tensor,
        end_time: float,
        output_frequency: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Train the coupled PINN-MultiPhysics system
        """
        logger.info(f"Starting coupled system training to t={end_time}")

        # Initialize physics states from PINN if needed
        if self.integration_config.mode in [
            IntegrationMode.PINN_DRIVEN,
            IntegrationMode.COUPLED,
        ]:
            initial_predictions = self.pinn_engine.predict(model_id, coordinates)
            physics_initial_states = {}

            for domain in self.multi_physics_engine.domains:
                if domain in self.domain_mappings:
                    physics_state = self.pinn_to_physics_state(
                        initial_predictions, domain
                    )
                    physics_initial_states[domain] = physics_state.mean(dim=0)

            if physics_initial_states:
                self.multi_physics_engine.initialize_states(physics_initial_states)

        # Training loop
        training_history = {
            "times": [],
            "coupling_residuals": [],
            "pinn_losses": [],
            "physics_convergence": [],
            "coupling_iterations": [],
        }

        current_time = 0.0
        step = 0

        while current_time < end_time:
            # Determine time step
            dt = min(self.multi_physics_engine.current_dt, end_time - current_time)

            # Solve coupled step
            step_results = self.solve_coupled_step(model_id, coordinates, dt)

            # Update PINN if physics-driven or coupled
            if self.integration_config.mode in [
                IntegrationMode.PHYSICS_DRIVEN,
                IntegrationMode.COUPLED,
            ]:
                if step % self.integration_config.pinn_update_frequency == 0:
                    # Create training data from physics states
                    physics_coords, physics_targets = self.physics_to_pinn_data(
                        step_results["physics_states"], coordinates
                    )

                    # Train PINN on physics data
                    trainer = self.pinn_engine.get_trainer(model_id)
                    if trainer:
                        # Create simple dataset
                        dataset = torch.utils.data.TensorDataset(
                            physics_coords, physics_targets
                        )
                        dataloader = torch.utils.data.DataLoader(
                            dataset, batch_size=32, shuffle=True
                        )

                        # Single training step
                        for batch in dataloader:
                            loss_components = trainer.train_step(batch[0], batch[1])
                            training_history["pinn_losses"].append(
                                loss_components.get("total_loss", 0.0)
                            )
                            break  # Only one batch per update

            # Update time
            current_time += dt
            step += 1

            # Record training history
            if (
                output_frequency is None
                or len(training_history["times"]) == 0
                or current_time - training_history["times"][-1] >= output_frequency
            ):
                training_history["times"].append(current_time)
                training_history["coupling_residuals"].append(
                    step_results["coupling_residual"]
                )
                training_history["coupling_iterations"].append(
                    step_results["iterations"]
                )

                # Get physics convergence info
                if self.multi_physics_engine.convergence_history:
                    last_convergence = self.multi_physics_engine.convergence_history[-1]
                    training_history["physics_convergence"].append(
                        last_convergence.final_residual
                    )
                else:
                    training_history["physics_convergence"].append(0.0)

            # Log progress
            if step % 100 == 0:
                logger.info(
                    f"Coupled training step {step}: t={current_time:.3f}/{end_time:.3f}, "
                    f"coupling_residual={step_results['coupling_residual']:.2e}"
                )

        logger.info(f"Coupled system training completed: {step} steps")

        return {
            "training_history": training_history,
            "final_states": step_results["physics_states"],
            "final_predictions": step_results["pinn_predictions"],
            "coupling_history": self.coupling_history,
            "total_steps": step,
        }

    def validate_coupling(
        self,
        model_id: str,
        validation_coordinates: torch.Tensor,
        validation_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Validate the coupled system performance
        """
        # Get PINN predictions
        pinn_predictions = self.pinn_engine.predict(model_id, validation_coordinates)

        # Convert to physics states and run physics simulation
        physics_initial_states = {}
        for domain in self.multi_physics_engine.domains:
            if domain in self.domain_mappings:
                physics_state = self.pinn_to_physics_state(pinn_predictions, domain)
                physics_initial_states[domain] = physics_state.mean(dim=0)

        if physics_initial_states:
            self.multi_physics_engine.initialize_states(physics_initial_states)

        # Run short physics simulation
        dt = 1e-3
        convergence_result = self.multi_physics_engine.solve_coupled_step(dt)

        # Convert physics results back to PINN format
        physics_coords, physics_targets_pred = self.physics_to_pinn_data(
            self.multi_physics_engine.domain_states, validation_coordinates
        )

        # Compute validation metrics
        metrics = {}

        # PINN-Physics consistency
        pinn_flat = pinn_predictions.view(-1)
        physics_flat = physics_targets_pred.view(-1)
        min_size = min(pinn_flat.shape[0], physics_flat.shape[0])

        consistency_error = torch.norm(pinn_flat[:min_size] - physics_flat[:min_size])
        metrics["pinn_physics_consistency"] = consistency_error.item()

        # Physics convergence
        metrics["physics_convergence_residual"] = convergence_result.final_residual
        metrics["physics_convergence_iterations"] = convergence_result.iterations
        metrics["physics_converged"] = convergence_result.status.value

        # PINN accuracy (if validation targets provided)
        if validation_targets is not None:
            pinn_error = torch.norm(pinn_predictions - validation_targets)
            metrics["pinn_validation_error"] = pinn_error.item()

        return metrics

    def get_integration_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive integration diagnostics"""
        diagnostics = {
            "integration_mode": self.integration_config.mode.value,
            "geological_process": self.geological_process.value,
            "current_coupling_strength": self.current_coupling_strength,
            "total_coupling_steps": len(self.coupling_history),
            "domain_mappings": {
                domain.value: mapping
                for domain, mapping in self.domain_mappings.items()
            },
            "pinn_models": self.pinn_engine.list_models(),
            "physics_domains": list(self.multi_physics_engine.domains.keys()),
            "coupling_terms": len(self.multi_physics_engine.coupling_terms),
        }

        # Coupling performance statistics
        if self.coupling_history:
            residuals = [h["coupling_residual"] for h in self.coupling_history]
            iterations = [h["iterations"] for h in self.coupling_history]

            diagnostics.update(
                {
                    "average_coupling_residual": np.mean(residuals),
                    "min_coupling_residual": np.min(residuals),
                    "max_coupling_residual": np.max(residuals),
                    "average_coupling_iterations": np.mean(iterations),
                    "convergence_rate": sum(
                        1 for h in self.coupling_history if h["converged"]
                    )
                    / len(self.coupling_history),
                }
            )

        return diagnostics

# Factory function for creating integrated systems
def create_integrated_system(
    geological_process: GeologicalProcess,
    pinn_config: PINNModelConfig,
    integration_config: Optional[IntegrationConfig] = None,
    domain_configs: Optional[Dict[PhysicsDomain, Dict[str, Any]]] = None,
    device: torch.device = torch.device("cpu"),
) -> PINNMultiPhysicsIntegrator:
    """
    Create a complete integrated PINN-MultiPhysics system

    Args:
        geological_process: Type of geological process
        pinn_config: Configuration for PINN model
        integration_config: Configuration for integration
        domain_configs: Configuration for physics domains
        device: Computation device

    Returns:
        Configured integrated system
    """
    # Create PINN engine
    pinn_engine = PINNEngine(device=device)

    # Create PINN model
    model_id = f"{geological_process.value}_integrated_model"
    pinn_engine.create_model(model_id, pinn_config, geological_process)

    # Create multi-physics engine
    multi_physics_engine = create_multi_physics_engine(
        geological_process=geological_process, domain_configs=domain_configs
    )

    # Default integration config
    if integration_config is None:
        integration_config = IntegrationConfig()

    # Create integrator
    integrator = PINNMultiPhysicsIntegrator(
        pinn_engine=pinn_engine,
        multi_physics_engine=multi_physics_engine,
        integration_config=integration_config,
        geological_process=geological_process,
    )

    logger.info(
        f"Created integrated PINN-MultiPhysics system for {geological_process.value}"
    )

    return integrator

# Utility functions for common integration scenarios
def create_magma_flow_system(
    device: torch.device = torch.device("cpu"),
) -> PINNMultiPhysicsIntegrator:
    """Create integrated system for magma flow modeling"""
    from models.ai_models import ActivationFunction, NetworkArchitecture

    # PINN configuration for magma flow
    network_architecture = NetworkArchitecture(
        layer_sizes=[4, 64, 128, 128, 64, 5],  # [x,y,z,t] -> [T,vx,vy,vz,c]
        activation_functions=[ActivationFunction.TANH] * 4,
        dropout_rates=[0.0] * 5,
        batch_normalization=True,
        residual_connections=True,
    )

    pinn_config = PINNModelConfig(
        network_architecture=network_architecture,
        physics_constraints=[],  # Will be set by integration
        optimization_config={"optimizer": "adam", "learning_rate": 1e-3},
        training_config={"max_epochs": 1000},
    )

    # Integration configuration
    integration_config = IntegrationConfig(
        mode=IntegrationMode.COUPLED,
        pinn_update_frequency=5,
        physics_update_frequency=1,
        coupling_strength=1.0,
    )

    # Domain configurations for magma flow
    domain_configs = {
        PhysicsDomain.THERMAL: {"thermal_diffusivity": 1e-6, "heat_capacity": 1200.0},
        PhysicsDomain.FLUID_DYNAMICS: {"viscosity": 1e3, "density": 2500.0},
        PhysicsDomain.CHEMICAL: {"diffusion_coefficient": 1e-10, "reaction_rate": 1e-4},
    }

    return create_integrated_system(
        geological_process=GeologicalProcess.MAGMA_FLOW,
        pinn_config=pinn_config,
        integration_config=integration_config,
        domain_configs=domain_configs,
        device=device,
    )

def create_crustal_deformation_system(
    device: torch.device = torch.device("cpu"),
) -> PINNMultiPhysicsIntegrator:
    """Create integrated system for crustal deformation modeling"""
    from models.ai_models import ActivationFunction, NetworkArchitecture

from typing import Tuple

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

    # PINN configuration for crustal deformation
    network_architecture = NetworkArchitecture(
        layer_sizes=[4, 64, 128, 64, 4],  # [x,y,z,t] -> [ux,uy,uz,T]
        activation_functions=[ActivationFunction.TANH] * 3,
        dropout_rates=[0.0] * 4,
        batch_normalization=True,
    )

    pinn_config = PINNModelConfig(
        network_architecture=network_architecture,
        physics_constraints=[],
        optimization_config={"optimizer": "adam", "learning_rate": 1e-3},
        training_config={"max_epochs": 1000},
    )

    # Integration configuration
    integration_config = IntegrationConfig(
        mode=IntegrationMode.COUPLED,
        pinn_update_frequency=10,
        physics_update_frequency=1,
    )

    # Domain configurations for crustal deformation
    domain_configs = {
        PhysicsDomain.MECHANICAL: {
            "youngs_modulus": 70e9,
            "poisson_ratio": 0.25,
            "density": 2700.0,
        },
        PhysicsDomain.THERMAL: {"thermal_diffusivity": 1e-6, "heat_capacity": 1000.0},
    }

    return create_integrated_system(
        geological_process=GeologicalProcess.CRUSTAL_DEFORMATION,
        pinn_config=pinn_config,
        integration_config=integration_config,
        domain_configs=domain_configs,
        device=device,
    )
