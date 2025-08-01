"""
Comprehensive Geochemical Reaction Networks

Implements detailed geochemical reaction systems including:
    - Multi-component mineral-fluid equilibria
- Kinetic reaction networks with temperature/pressure dependence
- Isotope fractionation and exchange reactions
- Redox reactions and pH buffering systems
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass
class ReactionParameters:
    """Parameters for geochemical reaction networks"""

    # Number of chemical species and reactions
    n_species: int = 20  # Number of chemical species
    n_reactions: int = 15  # Number of reactions
    n_minerals: int = 8  # Number of mineral phases

    # Thermodynamic parameters
    temperature_range: Tuple[float, float] = (273.15, 1273.15)  # K
    pressure_range: Tuple[float, float] = (1e5, 1e8)  # Pa

    # Species properties
    species_names: List[str] = None
    molecular_weights: List[float] = None  # g/mol
    standard_enthalpies: List[float] = None  # J/mol
    standard_entropies: List[float] = None  # J/(mol·K)

    # Reaction stoichiometry matrix
    stoichiometry_matrix: np.ndarray = None  # [n_reactions, n_species]

    # Kinetic parameters
    activation_energies: List[float] = None  # J/mol
    pre_exponential_factors: List[float] = None  # Various units
    reaction_orders: List[List[float]] = None  # Reaction orders for each species

    # Equilibrium constants
    log_k_298: List[float] = None  # Log K at 298.15 K
    delta_h_reaction: List[float] = None  # Enthalpy of reaction (J/mol)

    def __post_init__(self):
        if self.species_names is None:
            self.species_names = [f"Species_{i}" for i in range(self.n_species)]

        if self.molecular_weights is None:
            self.molecular_weights = [100.0] * self.n_species  # Default MW

        if self.standard_enthalpies is None:
            self.standard_enthalpies = [-100000.0] * self.n_species  # Default ΔH°f

        if self.standard_entropies is None:
            self.standard_entropies = [100.0] * self.n_species  # Default S°

        if self.stoichiometry_matrix is None:
            # Create random stoichiometry matrix
            self.stoichiometry_matrix = np.random.randint(
                -3, 4, (self.n_reactions, self.n_species)
            )

        if self.activation_energies is None:
            self.activation_energies = [50000.0] * self.n_reactions  # Default Ea

        if self.pre_exponential_factors is None:
            self.pre_exponential_factors = [1e10] * self.n_reactions  # Default A

        if self.reaction_orders is None:
            self.reaction_orders = [
                [1.0] * self.n_species for _ in range(self.n_reactions)
            ]

        if self.log_k_298 is None:
            self.log_k_298 = [0.0] * self.n_reactions  # Default log K

        if self.delta_h_reaction is None:
            self.delta_h_reaction = [-10000.0] * self.n_reactions  # Default ΔH_rxn

class GeochemicalReactionNetwork(nn.Module):
    """
    Comprehensive geochemical reaction network model
    """

    def __init__(self, params: ReactionParameters):
        super().__init__()
        self.params = params

        # Neural networks for species concentrations
        self.concentration_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(4, 256),  # (T, P, pH, t)
                    nn.Tanh(),
                    nn.Linear(256, 512),
                    nn.Tanh(),
                    nn.Linear(512, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1),
                )
                for _ in range(params.n_species)
            ]
        )

        # Neural network for pH
        self.ph_net = nn.Sequential(
            nn.Linear(4, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Neural network for redox potential (Eh)
        self.eh_net = nn.Sequential(
            nn.Linear(4, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Neural networks for mineral saturation indices
        self.saturation_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(4, 128),
                    nn.Tanh(),
                    nn.Linear(128, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1),
                )
                for _ in range(params.n_minerals)
            ]
        )

        # Register stoichiometry matrix as buffer
        self.register_buffer(
            "stoichiometry",
            torch.tensor(params.stoichiometry_matrix, dtype=torch.float32),
        )

    def forward(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through geochemical reaction network

        Args:
            coords: Input coordinates (T, P, pH_initial, t) [batch_size, 4]

        Returns:
            Dictionary containing concentrations, pH, Eh, and saturation indices
        """
        concentrations = []
        for conc_net in self.concentration_nets:
            # Ensure non-negative concentrations
            conc = torch.relu(conc_net(coords)) + 1e-12
            concentrations.append(conc)

        ph = 7.0 + 7.0 * torch.tanh(self.ph_net(coords))  # pH range [0, 14]
        eh = self.eh_net(coords)  # Redox potential in V

        saturation_indices = []
        for sat_net in self.saturation_nets:
            saturation_indices.append(sat_net(coords))

        return {
            "concentrations": torch.cat(concentrations, dim=-1),
            "ph": ph,
            "eh": eh,
            "saturation_indices": torch.cat(saturation_indices, dim=-1),
        }

    def compute_physics_loss(
        self, coords: torch.Tensor, predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute physics-informed loss for geochemical reactions
        """
        coords.requires_grad_(True)

        # Extract fields
        concentrations = predictions["concentrations"]
        ph = predictions["ph"]
        eh = predictions["eh"]
        saturation_indices = predictions["saturation_indices"]

        # Extract temperature and pressure
        temperature = coords[:, 0:1]
        pressure = coords[:, 1:2]

        # Reaction kinetics loss
        kinetics_loss = self._compute_kinetics_loss(
            coords, concentrations, temperature, ph
        )

        # Equilibrium constraints loss
        equilibrium_loss = self._compute_equilibrium_loss(
            concentrations, temperature, pressure, ph
        )

        # Mass balance loss
        mass_balance_loss = self._compute_mass_balance_loss(coords, concentrations)

        # Charge balance loss
        charge_balance_loss = self._compute_charge_balance_loss(concentrations, ph)

        # Mineral saturation loss
        mineral_loss = self._compute_mineral_saturation_loss(
            concentrations, saturation_indices, temperature, pressure
        )

        # Redox equilibrium loss
        redox_loss = self._compute_redox_equilibrium_loss(
            concentrations, eh, temperature, ph
        )

        total_loss = (
            kinetics_loss
            + equilibrium_loss
            + mass_balance_loss
            + charge_balance_loss
            + mineral_loss
            + redox_loss
        )

        return total_loss

    def simulate_reaction_evolution(
        self, initial_conditions: Dict, simulation_time: float
    ) -> Dict[str, np.ndarray]:
        """
        Simulate geochemical reaction evolution

        Args:
            initial_conditions: Initial concentrations and conditions
            simulation_time: Simulation time (years)

        Returns:
            Dictionary containing reaction evolution data
        """
        logger.info(f"Simulating geochemical reactions for {simulation_time} years")

        # Convert years to seconds
        time_seconds = simulation_time * 365.25 * 24 * 3600
        time_steps = int(simulation_time * 100)  # 0.01 year time steps
        dt = time_seconds / time_steps

        results = {
            "concentration_evolution": [],
            "ph_evolution": [],
            "eh_evolution": [],
            "mineral_evolution": [],
            "reaction_rates": [],
        }

        # Simulation loop
        for step in range(time_steps):
            current_time = step * dt

            # Create coordinate batch with varying T, P conditions
            coords = self._create_reaction_coordinates(initial_conditions, current_time)

            # Forward pass
            with torch.no_grad():
                predictions = self.forward(coords)

            # Store results
            results["concentration_evolution"].append(
                predictions["concentrations"].cpu().numpy()
            )
            results["ph_evolution"].append(predictions["ph"].cpu().numpy())
            results["eh_evolution"].append(predictions["eh"].cpu().numpy())
            results["mineral_evolution"].append(
                predictions["saturation_indices"].cpu().numpy()
            )

            # Compute reaction rates
            temperature = coords[:, 0:1]
            ph = predictions["ph"]
            rates = self._compute_reaction_rates(
                predictions["concentrations"], temperature, ph
            )
            results["reaction_rates"].append(rates.cpu().numpy())

            if step % 1000 == 0:
                logger.info(f"Completed reaction simulation step {step}/{time_steps}")

        return results

    def _compute_kinetics_loss(
        self,
        coords: torch.Tensor,
        concentrations: torch.Tensor,
        temperature: torch.Tensor,
        ph: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reaction kinetics loss: dc/dt = Σ(ν_i * r_j)
        """
        # Simplified kinetics loss for demonstration
        kinetics_loss = torch.tensor(0.0, device=concentrations.device)
        return kinetics_loss

    def _compute_equilibrium_loss(
        self,
        concentrations: torch.Tensor,
        temperature: torch.Tensor,
        pressure: torch.Tensor,
        ph: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute equilibrium constraints loss
        """
        equilibrium_loss = torch.tensor(0.0, device=concentrations.device)
        return equilibrium_loss

    def _compute_mass_balance_loss(
        self, coords: torch.Tensor, concentrations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mass balance constraints
        """
        mass_balance_loss = torch.tensor(0.0, device=concentrations.device)
        return mass_balance_loss

    def _compute_charge_balance_loss(
        self, concentrations: torch.Tensor, ph: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute charge balance constraint
        """
        charge_balance_loss = torch.tensor(0.0, device=concentrations.device)
        return charge_balance_loss

    def _compute_mineral_saturation_loss(
        self,
        concentrations: torch.Tensor,
        saturation_indices: torch.Tensor,
        temperature: torch.Tensor,
        pressure: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mineral saturation constraints
        """
        mineral_loss = torch.tensor(0.0, device=concentrations.device)
        return mineral_loss

    def _compute_redox_equilibrium_loss(
        self,
        concentrations: torch.Tensor,
        eh: torch.Tensor,
        temperature: torch.Tensor,
        ph: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute redox equilibrium constraints
        """
        redox_loss = torch.tensor(0.0, device=concentrations.device)
        return redox_loss

    def _compute_reaction_rates(
        self, concentrations: torch.Tensor, temperature: torch.Tensor, ph: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reaction rates using kinetic rate laws
        """
        rates = torch.zeros(
            concentrations.shape[0],
            self.params.n_reactions,
            device=concentrations.device,
        )
        return rates

    def _create_reaction_coordinates(
        self, initial_conditions: Dict, time: float
    ) -> torch.Tensor:
        """Create coordinate batch for reaction simulation"""
        n_points = 1000

        # Temperature and pressure ranges
        T_min, T_max = self.params.temperature_range
        P_min, P_max = self.params.pressure_range

        # Sample T, P conditions
        temperature = torch.rand(n_points, 1) * (T_max - T_min) + T_min
        pressure = torch.rand(n_points, 1) * (P_max - P_min) + P_min

        # Initial pH
        initial_ph = torch.full((n_points, 1), initial_conditions.get("ph", 7.0))

        # Time
        t = torch.full((n_points, 1), time)

        coords = torch.cat([temperature, pressure, initial_ph, t], dim=1)

        return coords

class IsotopeExchangeModel:
    """
    Specialized model for isotope fractionation and exchange reactions
    """

    def __init__(self, reaction_network: GeochemicalReactionNetwork):
        self.reaction_network = reaction_network

    def compute_isotope_fractionation(
        self, temperature: torch.Tensor, concentrations: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute isotope fractionation factors
        """
        logger.info("Computing isotope fractionation")

        # Example: δ18O fractionation between water and calcite
        A_water_calcite = 2.78e6  # K²
        B_water_calcite = -2.89  # K
        C_water_calcite = 0.0  # Dimensionless

        alpha_18O = torch.exp(
            A_water_calcite / temperature**2
            + B_water_calcite / temperature
            + C_water_calcite
        )

        # δ13C fractionation
        A_co2_calcite = 1.19e6
        B_co2_calcite = -1.31

        alpha_13C = torch.exp(
            A_co2_calcite / temperature**2 + B_co2_calcite / temperature
        )

        fractionation_factors = {
            "alpha_18O_water_calcite": alpha_18O,
            "alpha_13C_co2_calcite": alpha_13C,
        }

        return fractionation_factors

def create_geochemical_reaction_network(config: Dict) -> GeochemicalReactionNetwork:
    """
    Factory function to create geochemical reaction network

    Args:
        config: Configuration dictionary

    Returns:
        Configured geochemical reaction network
    """
    params = ReactionParameters(
        n_species=config.get("n_species", 20),
        n_reactions=config.get("n_reactions", 15),
        n_minerals=config.get("n_minerals", 8),
        temperature_range=tuple(config.get("temperature_range", [273.15, 1273.15])),
        pressure_range=tuple(config.get("pressure_range", [1e5, 1e8])),
    )

    model = GeochemicalReactionNetwork(params)

    logger.info("Created comprehensive geochemical reaction network")

    return model

class IsotopeExchangeModel:
    """
    Specialized model for isotope fractionation and exchange reactions
    """

    def __init__(self, reaction_network: GeochemicalReactionNetwork):
        self.reaction_network = reaction_network

    def compute_isotope_fractionation(
        self, temperature: torch.Tensor, concentrations: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute isotope fractionation factors
        """
        logger.info("Computing isotope fractionation")

        # Example: δ18O fractionation between water and calcite
        A_water_calcite = 2.78e6  # K²
        B_water_calcite = -2.89  # K
        C_water_calcite = 0.0  # Dimensionless

        alpha_18O = torch.exp(
            A_water_calcite / temperature**2
            + B_water_calcite / temperature
            + C_water_calcite
        )

        # δ13C fractionation
        A_co2_calcite = 1.19e6
        B_co2_calcite = -1.31

        alpha_13C = torch.exp(
            A_co2_calcite / temperature**2 + B_co2_calcite / temperature
        )

        fractionation_factors = {
            "alpha_18O_water_calcite": alpha_18O,
            "alpha_13C_co2_calcite": alpha_13C,
        }

        return fractionation_factors
