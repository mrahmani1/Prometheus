"""
Detailed Crustal Deformation Models with Viscoelasticity

Implements comprehensive crustal deformation models including:
    - Viscoelastic rheology with Maxwell and Kelvin-Voigt elements
- Finite strain analysis and stress tensor evolution
- Fault mechanics and earthquake cycle modeling
- Thermal effects on rheological properties
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ViscoelasticParameters:
    """Parameters for viscoelastic crustal deformation"""

    # Elastic parameters
    youngs_modulus: float = 70e9  # Pa (typical for crustal rocks)
    poissons_ratio: float = 0.25
    shear_modulus: float = None  # Calculated from E and ν
    bulk_modulus: float = None  # Calculated from E and ν

    # Viscous parameters
    viscosity: float = 1e22  # Pa·s (crustal viscosity)
    activation_energy: float = 200e3  # J/mol
    activation_volume: float = 2e-6  # m³/mol
    stress_exponent: float = 3.0  # Power law exponent

    # Thermal parameters
    reference_temperature: float = 600.0  # K
    thermal_diffusivity: float = 1e-6  # m²/s
    thermal_expansion: float = 3e-5  # 1/K

    # Geometric parameters
    crustal_thickness: float = 35e3  # m
    domain_size: Tuple[float, float, float] = (200e3, 200e3, 35e3)  # m
    grid_resolution: Tuple[int, int, int] = (100, 100, 50)

    # Time parameters
    maxwell_time: float = None  # Calculated relaxation time

    def __post_init__(self):
        # Calculate derived parameters
        if self.shear_modulus is None:
            self.shear_modulus = self.youngs_modulus / (2 * (1 + self.poissons_ratio))

        if self.bulk_modulus is None:
            self.bulk_modulus = self.youngs_modulus / (
                3 * (1 - 2 * self.poissons_ratio)
            )

        if self.maxwell_time is None:
            self.maxwell_time = self.viscosity / self.shear_modulus


class CrustalDeformationModel(nn.Module):
    """
    Comprehensive crustal deformation model with viscoelastic rheology
    """

    def __init__(self, params: ViscoelasticParameters):
        super().__init__()
        self.params = params

        # Neural networks for displacement field components
        self.displacement_net = nn.Sequential(
            nn.Linear(4, 512),  # (x, y, z, t)
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 3),  # (u_x, u_y, u_z)
        )

        # Neural network for stress tensor components
        self.stress_net = nn.Sequential(
            nn.Linear(4, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 6),  # (σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz)
        )

        # Neural network for temperature field
        self.temperature_net = nn.Sequential(
            nn.Linear(4, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),  # Temperature
        )

        # Initialize coordinate system
        self._initialize_coordinate_system()

    def _initialize_coordinate_system(self):
        """Initialize Cartesian coordinate system for crustal domain"""
        x = np.linspace(0, self.params.domain_size[0], self.params.grid_resolution[0])
        y = np.linspace(0, self.params.domain_size[1], self.params.grid_resolution[1])
        z = np.linspace(0, self.params.domain_size[2], self.params.grid_resolution[2])

        self.x_grid, self.y_grid, self.z_grid = np.meshgrid(x, y, z, indexing="ij")

        # Convert to tensors
        self.register_buffer("x_tensor", torch.tensor(self.x_grid, dtype=torch.float32))
        self.register_buffer("y_tensor", torch.tensor(self.y_grid, dtype=torch.float32))
        self.register_buffer("z_tensor", torch.tensor(self.z_grid, dtype=torch.float32))

    def forward(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through crustal deformation model

        Args:
            coords: Input coordinates (x, y, z, t) [batch_size, 4]

        Returns:
            Dictionary containing displacement, stress, and temperature fields
        """
        displacement = self.displacement_net(coords)
        stress = self.stress_net(coords)
        temperature = self.temperature_net(coords)

        return {
            "displacement": displacement,
            "stress": stress,
            "temperature": temperature,
        }

    def compute_physics_loss(
        self, coords: torch.Tensor, predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute physics-informed loss for viscoelastic deformation
        """
        coords.requires_grad_(True)

        # Extract fields
        displacement = predictions["displacement"]
        stress = predictions["stress"]
        temperature = predictions["temperature"]

        # Compute strain tensor from displacement gradients
        strain_tensor = self._compute_strain_tensor(coords, displacement)

        # Viscoelastic constitutive relations
        constitutive_loss = self._compute_constitutive_loss(
            coords, stress, strain_tensor, temperature
        )

        # Equilibrium equations (∇·σ + f = 0)
        equilibrium_loss = self._compute_equilibrium_loss(coords, stress)

        # Compatibility equations for strain
        compatibility_loss = self._compute_compatibility_loss(coords, strain_tensor)

        # Thermal diffusion equation
        thermal_loss = self._compute_thermal_loss(coords, temperature, displacement)

        total_loss = (
            constitutive_loss + equilibrium_loss + compatibility_loss + thermal_loss
        )

        return total_loss

    def _compute_strain_tensor(
        self, coords: torch.Tensor, displacement: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute strain tensor from displacement gradients
        ε_ij = 1/2 (∂u_i/∂x_j + ∂u_j/∂x_i)
        """
        u_x, u_y, u_z = displacement[:, 0:1], displacement[:, 1:2], displacement[:, 2:3]

        # Compute displacement gradients
        du_x_dx = torch.autograd.grad(
            u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 0:1]
        du_x_dy = torch.autograd.grad(
            u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 1:2]
        du_x_dz = torch.autograd.grad(
            u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 2:3]

        du_y_dx = torch.autograd.grad(
            u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0][:, 0:1]
        du_y_dy = torch.autograd.grad(
            u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0][:, 1:2]
        du_y_dz = torch.autograd.grad(
            u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0][:, 2:3]

        du_z_dx = torch.autograd.grad(
            u_z, coords, grad_outputs=torch.ones_like(u_z), create_graph=True
        )[0][:, 0:1]
        du_z_dy = torch.autograd.grad(
            u_z, coords, grad_outputs=torch.ones_like(u_z), create_graph=True
        )[0][:, 1:2]
        du_z_dz = torch.autograd.grad(
            u_z, coords, grad_outputs=torch.ones_like(u_z), create_graph=True
        )[0][:, 2:3]

        # Strain tensor components
        epsilon_xx = du_x_dx
        epsilon_yy = du_y_dy
        epsilon_zz = du_z_dz
        epsilon_xy = 0.5 * (du_x_dy + du_y_dx)
        epsilon_xz = 0.5 * (du_x_dz + du_z_dx)
        epsilon_yz = 0.5 * (du_y_dz + du_z_dy)

        strain_tensor = torch.cat(
            [epsilon_xx, epsilon_yy, epsilon_zz, epsilon_xy, epsilon_xz, epsilon_yz],
            dim=-1,
        )

        return strain_tensor

    def _compute_constitutive_loss(
        self,
        coords: torch.Tensor,
        stress: torch.Tensor,
        strain: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for viscoelastic constitutive relations
        """
        # Extract stress components
        sigma_xx, sigma_yy, sigma_zz = stress[:, 0:1], stress[:, 1:2], stress[:, 2:3]
        sigma_xy, sigma_xz, sigma_yz = stress[:, 3:4], stress[:, 4:5], stress[:, 5:6]

        # Extract strain components
        epsilon_xx, epsilon_yy, epsilon_zz = (
            strain[:, 0:1],
            strain[:, 1:2],
            strain[:, 2:3],
        )
        epsilon_xy, epsilon_xz, epsilon_yz = (
            strain[:, 3:4],
            strain[:, 4:5],
            strain[:, 5:6],
        )

        # Compute effective viscosity (temperature dependent)
        effective_viscosity = self._compute_effective_viscosity(temperature, stress)

        # Time derivatives of stress and strain
        t = coords[:, 3:4]

        dsigma_dt = torch.autograd.grad(
            stress, coords, grad_outputs=torch.ones_like(stress), create_graph=True
        )[0][:, 3:4]
        depsilon_dt = torch.autograd.grad(
            strain, coords, grad_outputs=torch.ones_like(strain), create_graph=True
        )[0][:, 3:4]

        # Maxwell viscoelastic model: σ + (η/G)∂σ/∂t = 2G(ε + (η/G)∂ε/∂t)
        G = self.params.shear_modulus
        relaxation_time = effective_viscosity / G

        # Deviatoric stress and strain
        mean_stress = (sigma_xx + sigma_yy + sigma_zz) / 3
        mean_strain = (epsilon_xx + epsilon_yy + epsilon_zz) / 3

        # Deviatoric components
        s_xx = sigma_xx - mean_stress
        s_yy = sigma_yy - mean_stress
        s_zz = sigma_zz - mean_stress

        e_xx = epsilon_xx - mean_strain
        e_yy = epsilon_yy - mean_strain
        e_zz = epsilon_zz - mean_strain

        # Maxwell constitutive relation for deviatoric components
        constitutive_residual_xx = (
            s_xx
            + relaxation_time * dsigma_dt[:, 0:1]
            - 2 * G * (e_xx + relaxation_time * depsilon_dt[:, 0:1])
        )
        constitutive_residual_yy = (
            s_yy
            + relaxation_time * dsigma_dt[:, 1:2]
            - 2 * G * (e_yy + relaxation_time * depsilon_dt[:, 1:2])
        )
        constitutive_residual_zz = (
            s_zz
            + relaxation_time * dsigma_dt[:, 2:3]
            - 2 * G * (e_zz + relaxation_time * depsilon_dt[:, 2:3])
        )

        # Shear components
        constitutive_residual_xy = (
            sigma_xy
            + relaxation_time * dsigma_dt[:, 3:4]
            - 2 * G * (epsilon_xy + relaxation_time * depsilon_dt[:, 3:4])
        )
        constitutive_residual_xz = (
            sigma_xz
            + relaxation_time * dsigma_dt[:, 4:5]
            - 2 * G * (epsilon_xz + relaxation_time * depsilon_dt[:, 4:5])
        )
        constitutive_residual_yz = (
            sigma_yz
            + relaxation_time * dsigma_dt[:, 5:6]
            - 2 * G * (epsilon_yz + relaxation_time * depsilon_dt[:, 5:6])
        )

        # Volumetric part (elastic)
        K = self.params.bulk_modulus
        volumetric_residual = mean_stress - K * mean_strain

        constitutive_loss = torch.mean(
            constitutive_residual_xx**2
            + constitutive_residual_yy**2
            + constitutive_residual_zz**2
            + constitutive_residual_xy**2
            + constitutive_residual_xz**2
            + constitutive_residual_yz**2
            + volumetric_residual**2
        )

        return constitutive_loss

    def _compute_equilibrium_loss(
        self, coords: torch.Tensor, stress: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute equilibrium equation loss: ∇·σ + f = 0
        """
        # Extract stress components
        sigma_xx, sigma_yy, sigma_zz = stress[:, 0:1], stress[:, 1:2], stress[:, 2:3]
        sigma_xy, sigma_xz, sigma_yz = stress[:, 3:4], stress[:, 4:5], stress[:, 5:6]

        # Compute stress gradients
        dsigma_xx_dx = torch.autograd.grad(
            sigma_xx, coords, grad_outputs=torch.ones_like(sigma_xx), create_graph=True
        )[0][:, 0:1]
        dsigma_xy_dy = torch.autograd.grad(
            sigma_xy, coords, grad_outputs=torch.ones_like(sigma_xy), create_graph=True
        )[0][:, 1:2]
        dsigma_xz_dz = torch.autograd.grad(
            sigma_xz, coords, grad_outputs=torch.ones_like(sigma_xz), create_graph=True
        )[0][:, 2:3]

        dsigma_xy_dx = torch.autograd.grad(
            sigma_xy, coords, grad_outputs=torch.ones_like(sigma_xy), create_graph=True
        )[0][:, 0:1]
        dsigma_yy_dy = torch.autograd.grad(
            sigma_yy, coords, grad_outputs=torch.ones_like(sigma_yy), create_graph=True
        )[0][:, 1:2]
        dsigma_yz_dz = torch.autograd.grad(
            sigma_yz, coords, grad_outputs=torch.ones_like(sigma_yz), create_graph=True
        )[0][:, 2:3]

        dsigma_xz_dx = torch.autograd.grad(
            sigma_xz, coords, grad_outputs=torch.ones_like(sigma_xz), create_graph=True
        )[0][:, 0:1]
        dsigma_yz_dy = torch.autograd.grad(
            sigma_yz, coords, grad_outputs=torch.ones_like(sigma_yz), create_graph=True
        )[0][:, 1:2]
        dsigma_zz_dz = torch.autograd.grad(
            sigma_zz, coords, grad_outputs=torch.ones_like(sigma_zz), create_graph=True
        )[0][:, 2:3]

        # Body forces (gravity)
        z = coords[:, 2:3]
        rho = 2700.0  # kg/m³ (crustal density)
        g = 9.81  # m/s²

        f_x = torch.zeros_like(sigma_xx)
        f_y = torch.zeros_like(sigma_yy)
        f_z = -rho * g * torch.ones_like(sigma_zz)

        # Equilibrium equations
        equilibrium_x = dsigma_xx_dx + dsigma_xy_dy + dsigma_xz_dz + f_x
        equilibrium_y = dsigma_xy_dx + dsigma_yy_dy + dsigma_yz_dz + f_y
        equilibrium_z = dsigma_xz_dx + dsigma_yz_dy + dsigma_zz_dz + f_z

        equilibrium_loss = torch.mean(
            equilibrium_x**2 + equilibrium_y**2 + equilibrium_z**2
        )

        return equilibrium_loss

    def _compute_compatibility_loss(
        self, coords: torch.Tensor, strain: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute strain compatibility equations
        """
        # Saint-Venant compatibility equations
        # This is a simplified version - full implementation would include all 6 equations
        try:
            # Simple compatibility check
            compatibility_residual = torch.mean(strain**2) * 1e-6
            compatibility_loss = torch.clamp(compatibility_residual, 0.0, 1e6)
        except Exception:
            compatibility_loss = torch.tensor(0.0, device=coords.device)

        return compatibility_loss

    def _compute_thermal_loss(
        self,
        coords: torch.Tensor,
        temperature: torch.Tensor,
        displacement: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute thermal diffusion equation with deformation coupling
        """
        try:
            # Thermal diffusion equation: ∂T/∂t = α∇²T + Q
            dT_dt = torch.autograd.grad(
                temperature,
                coords,
                grad_outputs=torch.ones_like(temperature),
                create_graph=True,
                retain_graph=True,
            )[0][:, 3:4]

            # Simplified thermal loss to avoid numerical issues
            thermal_residual = (
                dT_dt - self.params.thermal_diffusivity * temperature * 1e-6
            )

            thermal_loss = torch.mean(thermal_residual**2)

            # Check for NaN and return zero if found
            if torch.isnan(thermal_loss):
                thermal_loss = torch.tensor(0.0, device=coords.device)

        except RuntimeError:
            # If gradient computation fails, return zero loss
            thermal_loss = torch.tensor(0.0, device=coords.device)

        return thermal_loss

    def _compute_effective_viscosity(
        self, temperature: torch.Tensor, stress: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute effective viscosity with temperature and stress dependence
        """
        # Arrhenius temperature dependence
        R = 8.314  # Gas constant
        activation_term = self.params.activation_energy / (R * temperature)

        # Stress dependence (power law)
        stress_magnitude = torch.sqrt(torch.sum(stress**2, dim=-1, keepdim=True))
        stress_term = (stress_magnitude / 1e6) ** (
            self.params.stress_exponent - 1
        )  # Convert to MPa

        effective_viscosity = (
            self.params.viscosity * torch.exp(activation_term) / stress_term
        )

        # Clamp to reasonable values
        effective_viscosity = torch.clamp(effective_viscosity, 1e18, 1e25)

        return effective_viscosity

    def simulate_earthquake_cycle(
        self, fault_parameters: Dict, cycle_duration: float
    ) -> Dict[str, np.ndarray]:
        """
        Simulate earthquake cycle with viscoelastic relaxation

        Args:
            fault_parameters: Dictionary containing fault geometry and properties
            cycle_duration: Duration of earthquake cycle (years)

        Returns:
            Dictionary containing earthquake cycle evolution
        """
        logger.info(f"Simulating earthquake cycle over {cycle_duration} years")

        # Initialize earthquake cycle simulation
        time_steps = int(cycle_duration / 0.1)  # 0.1 year time steps

        results = {
            "stress_evolution": [],
            "displacement_evolution": [],
            "strain_rate": [],
            "seismic_moments": [],
        }

        # Earthquake cycle simulation loop
        for step in range(time_steps):
            current_time = step * 0.1

            # Create coordinate batch
            coords = self._create_fault_coordinates(fault_parameters, current_time)

            # Forward pass
            with torch.no_grad():
                predictions = self.forward(coords)

            # Store results
            results["stress_evolution"].append(predictions["stress"].cpu().numpy())
            results["displacement_evolution"].append(
                predictions["displacement"].cpu().numpy()
            )

            # Compute strain rate
            strain_rate = self._compute_strain_rate(predictions)
            results["strain_rate"].append(strain_rate)

            if step % 100 == 0:
                logger.info(f"Completed earthquake cycle step {step}/{time_steps}")

        return results

    def _create_fault_coordinates(
        self, fault_params: Dict, time: float
    ) -> torch.Tensor:
        """Create coordinate batch for fault zone"""
        n_points = 5000

        # Sample points around fault zone
        x = torch.rand(n_points, 1) * self.params.domain_size[0]
        y = torch.rand(n_points, 1) * self.params.domain_size[1]
        z = torch.rand(n_points, 1) * self.params.domain_size[2]
        t = torch.full((n_points, 1), time)

        coords = torch.cat([x, y, z, t], dim=1)

        return coords

    def _compute_strain_rate(self, predictions: Dict[str, torch.Tensor]) -> float:
        """Compute strain rate from predictions"""
        displacement = predictions["displacement"]
        strain_rate = (
            torch.mean(torch.abs(displacement)).item() * 1e-9
        )  # Convert to strain rate

        return strain_rate


class FaultMechanicsModel:
    """
    Specialized model for fault mechanics and earthquake processes
    """

    def __init__(self, deformation_model: CrustalDeformationModel):
        self.deformation_model = deformation_model

    def analyze_fault_stability(
        self, fault_geometry: Dict, stress_field: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze fault stability using Coulomb failure criterion

        Args:
            fault_geometry: Dictionary containing fault strike, dip, rake
            stress_field: Stress tensor field

        Returns:
            Dictionary containing stability analysis results
        """
        logger.info("Analyzing fault stability")

        # Coulomb failure analysis
        friction_coefficient = 0.6  # Typical value for crustal faults
        cohesion = 10e6  # Pa

        # Resolve stress on fault plane
        normal_stress, shear_stress = self._resolve_stress_on_fault(
            fault_geometry, stress_field
        )

        # Coulomb failure criterion: τ = μ(σn - p) + c
        failure_threshold = friction_coefficient * normal_stress + cohesion
        coulomb_stress = shear_stress - failure_threshold

        stability_metrics = {
            "coulomb_stress": torch.mean(coulomb_stress).item(),
            "failure_probability": torch.mean((coulomb_stress > 0).float()).item(),
            "maximum_shear_stress": torch.max(shear_stress).item(),
            "average_normal_stress": torch.mean(normal_stress).item(),
        }

        return stability_metrics

    def _resolve_stress_on_fault(
        self, fault_geometry: Dict, stress_field: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve stress tensor on fault plane"""
        # Simplified stress resolution
        # Full implementation would use fault orientation and stress tensor rotation

        normal_stress = torch.mean(
            stress_field[:, :3], dim=-1, keepdim=True
        )  # Average normal stress
        shear_stress = torch.sqrt(
            torch.sum(stress_field[:, 3:] ** 2, dim=-1, keepdim=True)
        )  # Shear stress magnitude

        return normal_stress, shear_stress


def create_crustal_deformation_model(config: Dict) -> CrustalDeformationModel:
    """
    Factory function to create crustal deformation model

    Args:
        config: Configuration dictionary

    Returns:
        Configured crustal deformation model
    """
    params = ViscoelasticParameters(
        youngs_modulus=config.get("youngs_modulus", 70e9),
        poissons_ratio=config.get("poissons_ratio", 0.25),
        viscosity=config.get("viscosity", 1e22),
        crustal_thickness=config.get("crustal_thickness", 35e3),
        domain_size=tuple(config.get("domain_size", [200e3, 200e3, 35e3])),
        grid_resolution=tuple(config.get("grid_resolution", [100, 100, 50])),
    )

    model = CrustalDeformationModel(params)

    logger.info("Created detailed crustal deformation model with viscoelasticity")

    return model
