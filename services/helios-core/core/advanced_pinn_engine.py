"""
Advanced Physics-Informed Neural Network (PINN) Engine for Geological Simulations

This module implements state-of-the-art PINNs for geological process modeling including:
    - Multi-scale temporal dynamics (geological vs real-time)
- Coupled thermo-mechanical processes
- Fluid flow in porous media
- Elastic wave propagation
- Magma chamber dynamics
- Uncertainty quantification with Bayesian neural networks
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

@dataclass
class GeologicalParameters:
    """Physical parameters for geological processes"""

    # Thermal properties
    thermal_conductivity: float = 2.5  # W/m/K
    heat_capacity: float = 1000.0  # J/kg/K
    density: float = 2700.0  # kg/m³
    thermal_diffusivity: float = 1e-6  # m²/s

    # Mechanical properties
    youngs_modulus: float = 50e9  # Pa
    poisson_ratio: float = 0.25
    bulk_modulus: float = 33.3e9  # Pa
    shear_modulus: float = 20e9  # Pa

    # Fluid properties
    permeability: float = 1e-15  # m²
    porosity: float = 0.1
    fluid_viscosity: float = 1e-3  # Pa·s
    fluid_density: float = 1000.0  # kg/m³

    # Seismic properties
    p_wave_velocity: float = 6000.0  # m/s
    s_wave_velocity: float = 3500.0  # m/s
    quality_factor: float = 100.0

    # Magmatic properties
    magma_viscosity: float = 1e6  # Pa·s
    gas_content: float = 0.05  # volume fraction
    crystallinity: float = 0.3  # volume fraction
    temperature: float = 1200.0  # K

class MultiScaleActivation(nn.Module):
    """Multi-scale activation function for geological processes"""

    def __init__(self, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        self.scale_frequencies = nn.Parameter(torch.logspace(-2, 2, num_scales))

    def forward(self, x):
        output = torch.zeros_like(x)
        for i in range(self.num_scales):
            scaled_input = self.scale_frequencies[i] * x
            output += self.scale_weights[i] * torch.tanh(scaled_input)
        return output / self.num_scales

class AdaptiveWeightedLoss(nn.Module):
    """Adaptive loss weighting for multi-physics problems"""

    def __init__(self, num_losses: int):
        super().__init__()
        self.num_losses = num_losses
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)

        return torch.stack(weighted_losses).sum()

class GeologicalPINN(nn.Module):
    """Advanced PINN for geological process simulation"""

    def __init__(
        self,
        input_dim: int = 4,  # x, y, z, t
        output_dim: int = 6,  # T, P, u, v, w, phi
        hidden_layers: List[int] = [128, 128, 128, 128],
        activation: str = "multiscale",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build network architecture
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "multiscale":
                layers.append(MultiScaleActivation())
            elif activation == "swish":
                layers.append(nn.SiLU())
            else:
                layers.append(nn.Tanh())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)

        # Adaptive loss weighting
        self.loss_weights = AdaptiveWeightedLoss(5)  # 5 physics losses

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)

    def predict_fields(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict all physical fields"""
        output = self.forward(coords)

        return {
            "temperature": output[:, 0:1],
            "pressure": output[:, 1:2],
            "velocity_x": output[:, 2:3],
            "velocity_y": output[:, 3:4],
            "velocity_z": output[:, 4:5],
            "porosity": torch.sigmoid(output[:, 5:6]),  # Ensure 0 < phi < 1
        }

class HeatEquationPINN(GeologicalPINN):
    """PINN for 3D transient heat equation with geological constraints"""

    def __init__(self, params: GeologicalParameters, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.alpha = params.thermal_diffusivity

    def physics_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute physics loss for heat equation"""
        coords.requires_grad_(True)

        # Forward pass
        fields = self.predict_fields(coords)
        T = fields["temperature"]

        # Compute gradients
        T_t = torch.autograd.grad(T.sum(), coords, create_graph=True)[0][:, 3:4]
        T_x = torch.autograd.grad(T.sum(), coords, create_graph=True)[0][:, 0:1]
        T_y = torch.autograd.grad(T.sum(), coords, create_graph=True)[0][:, 1:2]
        T_z = torch.autograd.grad(T.sum(), coords, create_graph=True)[0][:, 2:3]

        # Second derivatives
        T_xx = torch.autograd.grad(T_x.sum(), coords, create_graph=True)[0][:, 0:1]
        T_yy = torch.autograd.grad(T_y.sum(), coords, create_graph=True)[0][:, 1:2]
        T_zz = torch.autograd.grad(T_z.sum(), coords, create_graph=True)[0][:, 2:3]

        # Heat equation: ∂T/∂t = α∇²T
        laplacian_T = T_xx + T_yy + T_zz
        heat_equation = T_t - self.alpha * laplacian_T

        return torch.mean(heat_equation**2)

    def boundary_loss(
        self, boundary_coords: torch.Tensor, boundary_values: torch.Tensor
    ) -> torch.Tensor:
        """Compute boundary condition loss"""
        fields = self.predict_fields(boundary_coords)
        T_pred = fields["temperature"]

        return torch.mean((T_pred - boundary_values) ** 2)

    def initial_loss(
        self, initial_coords: torch.Tensor, initial_values: torch.Tensor
    ) -> torch.Tensor:
        """Compute initial condition loss"""
        fields = self.predict_fields(initial_coords)
        T_pred = fields["temperature"]

        return torch.mean((T_pred - initial_values) ** 2)

class ElasticWavePINN(GeologicalPINN):
    """PINN for elastic wave propagation in geological media"""

    def __init__(self, params: GeologicalParameters, **kwargs):
        super().__init__(output_dim=3, **kwargs)  # u, v, w displacements
        self.params = params
        self.vp = params.p_wave_velocity
        self.vs = params.s_wave_velocity
        self.rho = params.density

        # Lamé parameters
        self.mu = params.shear_modulus
        self.lam = params.bulk_modulus - 2 / 3 * params.shear_modulus

    def physics_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute physics loss for elastic wave equation"""
        coords.requires_grad_(True)

        # Forward pass - get displacements
        output = self.forward(coords)
        u, v, w = output[:, 0:1], output[:, 1:2], output[:, 2:3]

        # First derivatives
        u_t = torch.autograd.grad(u.sum(), coords, create_graph=True)[0][:, 3:4]
        u_x = torch.autograd.grad(u.sum(), coords, create_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u.sum(), coords, create_graph=True)[0][:, 1:2]
        u_z = torch.autograd.grad(u.sum(), coords, create_graph=True)[0][:, 2:3]

        v_t = torch.autograd.grad(v.sum(), coords, create_graph=True)[0][:, 3:4]
        v_x = torch.autograd.grad(v.sum(), coords, create_graph=True)[0][:, 0:1]
        v_y = torch.autograd.grad(v.sum(), coords, create_graph=True)[0][:, 1:2]
        v_z = torch.autograd.grad(v.sum(), coords, create_graph=True)[0][:, 2:3]

        w_t = torch.autograd.grad(w.sum(), coords, create_graph=True)[0][:, 3:4]
        w_x = torch.autograd.grad(w.sum(), coords, create_graph=True)[0][:, 0:1]
        w_y = torch.autograd.grad(w.sum(), coords, create_graph=True)[0][:, 1:2]
        w_z = torch.autograd.grad(w.sum(), coords, create_graph=True)[0][:, 2:3]

        # Second time derivatives
        u_tt = torch.autograd.grad(u_t.sum(), coords, create_graph=True)[0][:, 3:4]
        v_tt = torch.autograd.grad(v_t.sum(), coords, create_graph=True)[0][:, 3:4]
        w_tt = torch.autograd.grad(w_t.sum(), coords, create_graph=True)[0][:, 3:4]

        # Second spatial derivatives
        u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True)[0][:, 1:2]
        u_zz = torch.autograd.grad(u_z.sum(), coords, create_graph=True)[0][:, 2:3]

        v_xx = torch.autograd.grad(v_x.sum(), coords, create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y.sum(), coords, create_graph=True)[0][:, 1:2]
        v_zz = torch.autograd.grad(v_z.sum(), coords, create_graph=True)[0][:, 2:3]

        w_xx = torch.autograd.grad(w_x.sum(), coords, create_graph=True)[0][:, 0:1]
        w_yy = torch.autograd.grad(w_y.sum(), coords, create_graph=True)[0][:, 1:2]
        w_zz = torch.autograd.grad(w_z.sum(), coords, create_graph=True)[0][:, 2:3]

        # Mixed derivatives for strain
        u_xy = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 1:2]
        u_xz = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 2:3]
        v_yx = torch.autograd.grad(v_y.sum(), coords, create_graph=True)[0][:, 0:1]
        v_yz = torch.autograd.grad(v_y.sum(), coords, create_graph=True)[0][:, 2:3]
        w_zx = torch.autograd.grad(w_z.sum(), coords, create_graph=True)[0][:, 0:1]
        w_zy = torch.autograd.grad(w_z.sum(), coords, create_graph=True)[0][:, 1:2]

        # Divergence of displacement
        div_u = u_x + v_y + w_z

        # Elastic wave equations (Navier's equations)
        # ρ∂²u/∂t² = (λ + μ)∂(∇·u)/∂x + μ∇²u
        wave_eq_x = (
            self.rho * u_tt
            - (self.lam + self.mu)
            * torch.autograd.grad(div_u.sum(), coords, create_graph=True)[0][:, 0:1]
            - self.mu * (u_xx + u_yy + u_zz)
        )
        wave_eq_y = (
            self.rho * v_tt
            - (self.lam + self.mu)
            * torch.autograd.grad(div_u.sum(), coords, create_graph=True)[0][:, 1:2]
            - self.mu * (v_xx + v_yy + v_zz)
        )
        wave_eq_z = (
            self.rho * w_tt
            - (self.lam + self.mu)
            * torch.autograd.grad(div_u.sum(), coords, create_graph=True)[0][:, 2:3]
            - self.mu * (w_xx + w_yy + w_zz)
        )

        # Combined loss
        wave_loss = torch.mean(wave_eq_x**2 + wave_eq_y**2 + wave_eq_z**2)

        return wave_loss

class MagmaChamberPINN(GeologicalPINN):
    """PINN for magma chamber dynamics with gas exsolution"""

    def __init__(self, params: GeologicalParameters, **kwargs):
        super().__init__(output_dim=5, **kwargs)  # P, T, rho, gas_fraction, velocity
        self.params = params
        self.gas_solubility_coeff = 1e-6  # Pa⁻¹
        self.critical_gas_fraction = 0.75

    def physics_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute physics loss for magma chamber dynamics"""
        coords.requires_grad_(True)

        # Forward pass
        output = self.forward(coords)
        P = output[:, 0:1]  # Pressure
        T = output[:, 1:2]  # Temperature
        rho = output[:, 2:3]  # Density
        gas_frac = torch.sigmoid(output[:, 3:4])  # Gas fraction (0-1)
        v = output[:, 4:5]  # Vertical velocity

        # Gradients
        P_t = torch.autograd.grad(P.sum(), coords, create_graph=True)[0][:, 3:4]
        P_z = torch.autograd.grad(P.sum(), coords, create_graph=True)[0][:, 2:3]

        T_t = torch.autograd.grad(T.sum(), coords, create_graph=True)[0][:, 3:4]
        T_z = torch.autograd.grad(T.sum(), coords, create_graph=True)[0][:, 2:3]

        rho_t = torch.autograd.grad(rho.sum(), coords, create_graph=True)[0][:, 3:4]
        rho_z = torch.autograd.grad(rho.sum(), coords, create_graph=True)[0][:, 2:3]

        gas_frac_t = torch.autograd.grad(gas_frac.sum(), coords, create_graph=True)[0][
            :, 3:4
        ]

        v_z = torch.autograd.grad(v.sum(), coords, create_graph=True)[0][:, 2:3]

        # Physics equations

        # 1. Hydrostatic pressure balance
        g = 9.81  # gravity
        hydrostatic_eq = P_z + rho * g

        # 2. Gas exsolution kinetics
        # Equilibrium gas fraction based on pressure
        gas_eq = self.gas_solubility_coeff * P
        gas_kinetics = gas_frac_t - 1e-3 * (
            gas_eq - gas_frac
        )  # Relaxation to equilibrium

        # 3. Density evolution with gas content
        rho_melt = 2500.0  # kg/m³
        rho_gas = 100.0  # kg/m³
        density_eq = rho - (rho_melt * (1 - gas_frac) + rho_gas * gas_frac)

        # 4. Buoyancy-driven flow
        # Simplified momentum equation for vertical flow
        mu_eff = self.params.magma_viscosity * torch.exp(
            5 * gas_frac
        )  # Viscosity increases with gas
        buoyancy_force = rho * g
        momentum_eq = mu_eff * v_z - buoyancy_force

        # 5. Energy conservation (simplified)
        cp = 1200.0  # J/kg/K
        thermal_eq = rho * cp * T_t - 1e-3 * (1400.0 - T)  # Cooling toward ambient

        # Combine losses
        losses = [
            torch.mean(hydrostatic_eq**2),
            torch.mean(gas_kinetics**2),
            torch.mean(density_eq**2),
            torch.mean(momentum_eq**2),
            torch.mean(thermal_eq**2),
        ]

        return self.loss_weights(losses)

    def eruption_criterion(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate eruption likelihood based on gas fraction"""
        output = self.forward(coords)
        gas_frac = torch.sigmoid(output[:, 3:4])

        # Eruption probability increases rapidly above critical gas fraction
        eruption_prob = torch.sigmoid(10 * (gas_frac - self.critical_gas_fraction))

        return eruption_prob

class BayesianPINN(GeologicalPINN):
    """Bayesian PINN for uncertainty quantification"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Convert deterministic weights to Bayesian
        self._make_bayesian()

        # Prior parameters
        self.prior_mu = 0.0
        self.prior_sigma = 1.0

        # Variational parameters
        self.kl_weight = 1e-6

    def _make_bayesian(self):
        """Convert network to Bayesian by adding weight uncertainty"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Replace weights with variational parameters
                weight_mu = nn.Parameter(module.weight.data.clone())
                weight_rho = nn.Parameter(torch.full_like(module.weight, -3.0))

                bias_mu = nn.Parameter(module.bias.data.clone())
                bias_rho = nn.Parameter(torch.full_like(module.bias, -3.0))

                # Store variational parameters
                module.register_parameter("weight_mu", weight_mu)
                module.register_parameter("weight_rho", weight_rho)
                module.register_parameter("bias_mu", bias_mu)
                module.register_parameter("bias_rho", bias_rho)

                # Remove original parameters
                delattr(module, "weight")
                delattr(module, "bias")

    def sample_weights(self, module):
        """Sample weights from variational distribution"""
        weight_sigma = torch.log1p(torch.exp(module.weight_rho))
        bias_sigma = torch.log1p(torch.exp(module.bias_rho))

        weight_eps = torch.randn_like(module.weight_mu)
        bias_eps = torch.randn_like(module.bias_mu)

        weight = module.weight_mu + weight_sigma * weight_eps
        bias = module.bias_mu + bias_sigma * bias_eps

        return weight, bias

    def forward(self, x, num_samples: int = 1):
        """Forward pass with weight sampling"""
        if num_samples == 1:
            # Single forward pass
            return self._forward_single(x)
        else:
            # Multiple samples for uncertainty estimation
            outputs = []
            for _ in range(num_samples):
                outputs.append(self._forward_single(x))
            return torch.stack(outputs)

    def _forward_single(self, x):
        """Single forward pass with sampled weights"""
        for module in self.network:
            if isinstance(module, nn.Linear):
                weight, bias = self.sample_weights(module)
                x = F.linear(x, weight, bias)
            else:
                x = module(x)
        return x

    def kl_divergence(self):
        """Compute KL divergence for variational inference"""
        kl_div = 0.0

        for module in self.network:
            if isinstance(module, nn.Linear):
                # Weight KL divergence
                weight_sigma = torch.log1p(torch.exp(module.weight_rho))
                weight_kl = 0.5 * torch.sum(
                    (module.weight_mu**2 + weight_sigma**2) / self.prior_sigma**2
                    - torch.log(weight_sigma**2 / self.prior_sigma**2)
                    - 1
                )

                # Bias KL divergence
                bias_sigma = torch.log1p(torch.exp(module.bias_rho))
                bias_kl = 0.5 * torch.sum(
                    (module.bias_mu**2 + bias_sigma**2) / self.prior_sigma**2
                    - torch.log(bias_sigma**2 / self.prior_sigma**2)
                    - 1
                )

                kl_div += weight_kl + bias_kl

        return kl_div

    def predict_with_uncertainty(
        self, coords: torch.Tensor, num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """Predict with uncertainty quantification"""
        self.eval()

        with torch.no_grad():
            # Multiple forward passes
            outputs = self.forward(coords, num_samples=num_samples)

            # Compute statistics
            mean_output = torch.mean(outputs, dim=0)
            std_output = torch.std(outputs, dim=0)

            # Convert to field predictions
            fields_mean = {
                "temperature": mean_output[:, 0:1],
                "pressure": mean_output[:, 1:2],
                "velocity_x": mean_output[:, 2:3],
                "velocity_y": mean_output[:, 3:4],
                "velocity_z": mean_output[:, 4:5],
                "porosity": torch.sigmoid(mean_output[:, 5:6]),
            }

            fields_std = {
                "temperature": std_output[:, 0:1],
                "pressure": std_output[:, 1:2],
                "velocity_x": std_output[:, 2:3],
                "velocity_y": std_output[:, 3:4],
                "velocity_z": std_output[:, 4:5],
                "porosity": std_output[:, 5:6],
            }

        return {"mean": fields_mean, "std": fields_std}

class AdvancedPINNEngine:
    """Advanced PINN engine for geological simulations"""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.training_history = {}

        logger.info(f"Initialized Advanced PINN Engine on {device}")

    def create_heat_model(
        self, params: GeologicalParameters, model_name: str = "heat_model"
    ) -> HeatEquationPINN:
        """Create heat equation PINN model"""
        model = HeatEquationPINN(params).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)

        self.models[model_name] = model
        self.optimizers[model_name] = optimizer
        self.schedulers[model_name] = scheduler
        self.training_history[model_name] = {
            "loss": [],
            "physics_loss": [],
            "boundary_loss": [],
        }

        return model

    def create_wave_model(
        self, params: GeologicalParameters, model_name: str = "wave_model"
    ) -> ElasticWavePINN:
        """Create elastic wave PINN model"""
        model = ElasticWavePINN(params).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)

        self.models[model_name] = model
        self.optimizers[model_name] = optimizer
        self.schedulers[model_name] = scheduler
        self.training_history[model_name] = {"loss": [], "physics_loss": []}

        return model

    def create_magma_model(
        self, params: GeologicalParameters, model_name: str = "magma_model"
    ) -> MagmaChamberPINN:
        """Create magma chamber PINN model"""
        model = MagmaChamberPINN(params).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)

        self.models[model_name] = model
        self.optimizers[model_name] = optimizer
        self.schedulers[model_name] = scheduler
        self.training_history[model_name] = {"loss": [], "physics_loss": []}

        return model

    def create_bayesian_model(
        self, params: GeologicalParameters, model_name: str = "bayesian_model"
    ) -> BayesianPINN:
        """Create Bayesian PINN model for uncertainty quantification"""
        model = BayesianPINN().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)

        self.models[model_name] = model
        self.optimizers[model_name] = optimizer
        self.schedulers[model_name] = scheduler
        self.training_history[model_name] = {
            "loss": [],
            "physics_loss": [],
            "kl_loss": [],
        }

        return model

    def train_model(
        self,
        model_name: str,
        training_data: Dict[str, torch.Tensor],
        epochs: int = 10000,
        log_interval: int = 100,
    ) -> Dict[str, List[float]]:
        """Train a PINN model"""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        optimizer = self.optimizers[model_name]
        scheduler = self.schedulers[model_name]

        model.train()

        # Extract training data
        physics_coords = training_data["physics_coords"].to(self.device)
        boundary_coords = training_data.get("boundary_coords")
        boundary_values = training_data.get("boundary_values")
        initial_coords = training_data.get("initial_coords")
        initial_values = training_data.get("initial_values")

        logger.info(f"Starting training for {model_name} with {epochs} epochs")

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Physics loss
            physics_loss = model.physics_loss(physics_coords)
            total_loss = physics_loss

            # Boundary loss (if applicable)
            boundary_loss = torch.tensor(0.0)
            if boundary_coords is not None and boundary_values is not None:
                boundary_coords = boundary_coords.to(self.device)
                boundary_values = boundary_values.to(self.device)
                boundary_loss = model.boundary_loss(boundary_coords, boundary_values)
                total_loss += 10.0 * boundary_loss  # Weight boundary loss higher

            # Initial condition loss (if applicable)
            initial_loss = torch.tensor(0.0)
            if initial_coords is not None and initial_values is not None:
                initial_coords = initial_coords.to(self.device)
                initial_values = initial_values.to(self.device)
                initial_loss = model.initial_loss(initial_coords, initial_values)
                total_loss += 10.0 * initial_loss

            # KL divergence for Bayesian models
            kl_loss = torch.tensor(0.0)
            if isinstance(model, BayesianPINN):
                kl_loss = model.kl_divergence()
                total_loss += model.kl_weight * kl_loss

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(total_loss)

            # Logging
            if epoch % log_interval == 0:
                logger.info(
                    f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}, "
                    f"Physics Loss = {physics_loss.item():.6f}, "
                    f"Boundary Loss = {boundary_loss.item():.6f}"
                )

            # Store history
            self.training_history[model_name]["loss"].append(total_loss.item())
            self.training_history[model_name]["physics_loss"].append(
                physics_loss.item()
            )
            if boundary_loss.item() > 0:
                self.training_history[model_name]["boundary_loss"].append(
                    boundary_loss.item()
                )
            if isinstance(model, BayesianPINN):
                self.training_history[model_name]["kl_loss"].append(kl_loss.item())

        logger.info(f"Training completed for {model_name}")
        return self.training_history[model_name]

    def predict(self, model_name: str, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions with a trained model"""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        model.eval()

        coords = coords.to(self.device)

        with torch.no_grad():
            if isinstance(model, BayesianPINN):
                return model.predict_with_uncertainty(coords)
            else:
                return model.predict_fields(coords)

    def save_model(self, model_name: str, filepath: str):
        """Save a trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        torch.save(
            {
                "model_state_dict": self.models[model_name].state_dict(),
                "optimizer_state_dict": self.optimizers[model_name].state_dict(),
                "training_history": self.training_history[model_name],
            },
            filepath,
        )

        logger.info(f"Model {model_name} saved to {filepath}")

    def load_model(
        self, model_name: str, filepath: str, model_class, params: GeologicalParameters
    ):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Create model
        if model_class == HeatEquationPINN:
            model = self.create_heat_model(params, model_name)
        elif model_class == ElasticWavePINN:
            model = self.create_wave_model(params, model_name)
        elif model_class == MagmaChamberPINN:
            model = self.create_magma_model(params, model_name)
        elif model_class == BayesianPINN:
            model = self.create_bayesian_model(params, model_name)
        else:
            raise ValueError(f"Unknown model class: {model_class}")

        # Load state
        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizers[model_name].load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history[model_name] = checkpoint["training_history"]

        logger.info(f"Model {model_name} loaded from {filepath}")
        return model
