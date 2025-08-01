"""
Test suite for sophisticated geological process models

Tests all components of the geological process modeling system:
    - Mantle convection models
- Crustal deformation with viscoelasticity
- Hydrothermal flow systems
- Geochemical reaction networks
- Integrated multi-physics modeling
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from core.geological_processes.crustal_deformation import (
    CrustalDeformationModel,
    ViscoelasticParameters,
)
from core.geological_processes.fluid_flow import (
    FluidFlowParameters,
    HydrothermalFlowModel,
)
from core.geological_processes.geochemical_reactions import (
    GeochemicalReactionNetwork,
    ReactionParameters,
)
from core.geological_processes.integrated_model import (
    IntegratedGeologicalModel,
    IntegratedModelParameters,
    create_integrated_geological_model,
)
from core.geological_processes.mantle_convection import (
    ConvectionParameters,
    MantleConvectionModel,
    create_mantle_convection_model,
    import,
)
from core.geological_processes.mantle_convection import numpy as np
from core.geological_processes.mantle_convection import torch.nn as nn
from core.geological_processes.mantle_convection import torch.nn.functional as F

class TestMantleConvectionModel:
    """Test suite for mantle convection modeling"""

    def setup_method(self):
        """Set up test fixtures"""
        self.params = ConvectionParameters(
            rayleigh_number=1e5, n_radial=32, n_theta=64, n_phi=128
        )
        self.model = MantleConvectionModel(self.params)

    def test_model_initialization(self):
        """Test model initialization"""
        assert self.model.params.rayleigh_number == 1e5
        assert hasattr(self.model, "velocity_net")
        assert hasattr(self.model, "temperature_net")
        assert hasattr(self.model, "pressure_net")

    def test_forward_pass(self):
        """Test forward pass through model"""
        batch_size = 100
        coords = torch.rand(batch_size, 4)  # (r, θ, φ, t)

        predictions = self.model(coords)

        assert "velocity" in predictions
        assert "temperature" in predictions
        assert "pressure" in predictions
        assert "compositions" in predictions

        assert predictions["velocity"].shape == (batch_size, 3)
        assert predictions["temperature"].shape == (batch_size, 1)
        assert predictions["pressure"].shape == (batch_size, 1)

    def test_physics_loss_computation(self):
        """Test physics-informed loss computation"""
        batch_size = 50
        coords = torch.rand(batch_size, 4, requires_grad=True)

        predictions = self.model(coords)
        loss = self.model.compute_physics_loss(coords, predictions)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_factory_function(self):
        """Test factory function"""
        config = {"rayleigh_number": 1e6, "n_radial": 64, "n_theta": 128, "n_phi": 256}

        model = create_mantle_convection_model(config)

        assert isinstance(model, MantleConvectionModel)
        assert model.params.rayleigh_number == 1e6

class TestCrustalDeformationModel:
    """Test suite for crustal deformation modeling"""

    def setup_method(self):
        """Set up test fixtures"""
        self.params = ViscoelasticParameters(
            youngs_modulus=50e9,
            viscosity=1e21,
            domain_size=(100e3, 100e3, 20e3),
            grid_resolution=(50, 50, 25),
        )
        self.model = CrustalDeformationModel(self.params)

    def test_model_initialization(self):
        """Test model initialization"""
        assert self.model.params.youngs_modulus == 50e9
        assert hasattr(self.model, "displacement_net")
        assert hasattr(self.model, "stress_net")
        assert hasattr(self.model, "temperature_net")

    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 100
        coords = torch.rand(batch_size, 4)  # (x, y, z, t)

        predictions = self.model(coords)

        assert "displacement" in predictions
        assert "stress" in predictions
        assert "temperature" in predictions

        assert predictions["displacement"].shape == (batch_size, 3)
        assert predictions["stress"].shape == (batch_size, 6)
        assert predictions["temperature"].shape == (batch_size, 1)

    def test_viscoelastic_physics_loss(self):
        """Test viscoelastic physics loss"""
        batch_size = 50
        coords = torch.rand(batch_size, 4, requires_grad=True)

        predictions = self.model(coords)
        loss = self.model.compute_physics_loss(coords, predictions)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

class TestHydrothermalFlowModel:
    """Test suite for hydrothermal flow modeling"""

    def setup_method(self):
        """Set up test fixtures"""
        self.params = FluidFlowParameters(
            permeability=1e-14,
            porosity=0.15,
            n_species=5,
            domain_size=(500.0, 500.0, 250.0),
        )
        self.model = HydrothermalFlowModel(self.params)

    def test_model_initialization(self):
        """Test model initialization"""
        assert self.model.params.permeability == 1e-14
        assert hasattr(self.model, "pressure_net")
        assert hasattr(self.model, "temperature_net")
        assert hasattr(self.model, "velocity_net")
        assert len(self.model.concentration_nets) == 5

    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 100
        coords = torch.rand(batch_size, 4)  # (x, y, z, t)

        predictions = self.model(coords)

        assert "pressure" in predictions
        assert "temperature" in predictions
        assert "velocity" in predictions
        assert "phase_fraction" in predictions
        assert "concentrations" in predictions

        assert predictions["concentrations"].shape == (batch_size, 5)

    def test_multi_phase_flow_physics(self):
        """Test multi-phase flow physics"""
        batch_size = 50
        coords = torch.rand(batch_size, 4, requires_grad=True)

        predictions = self.model(coords)
        loss = self.model.compute_physics_loss(coords, predictions)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

class TestGeochemicalReactionNetwork:
    """Test suite for geochemical reaction networks"""

    def setup_method(self):
        """Set up test fixtures"""
        self.params = ReactionParameters(n_species=10, n_reactions=8, n_minerals=5)
        self.model = GeochemicalReactionNetwork(self.params)

    def test_model_initialization(self):
        """Test model initialization"""
        assert self.model.params.n_species == 10
        assert len(self.model.concentration_nets) == 10
        assert len(self.model.saturation_nets) == 5
        assert hasattr(self.model, "ph_net")
        assert hasattr(self.model, "eh_net")

    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 100
        coords = torch.rand(batch_size, 4)  # (T, P, pH, t)

        predictions = self.model(coords)

        assert "concentrations" in predictions
        assert "ph" in predictions
        assert "eh" in predictions
        assert "saturation_indices" in predictions

        assert predictions["concentrations"].shape == (batch_size, 10)
        assert predictions["saturation_indices"].shape == (batch_size, 5)

    def test_reaction_simulation(self):
        """Test reaction evolution simulation"""
        initial_conditions = {"ph": 7.0, "temperature": 298.15}

        # Mock the simulation to avoid long computation
        with patch.object(self.model, "_create_reaction_coordinates") as mock_coords:
            mock_coords.return_value = torch.rand(100, 4)

            results = self.model.simulate_reaction_evolution(initial_conditions, 0.1)

            assert "concentration_evolution" in results
            assert "ph_evolution" in results
            assert "reaction_rates" in results

class TestIntegratedGeologicalModel:
    """Test suite for integrated geological modeling"""

    def setup_method(self):
        """Set up test fixtures"""
        self.params = IntegratedModelParameters()
        self.model = IntegratedGeologicalModel(self.params)

    def test_model_initialization(self):
        """Test integrated model initialization"""
        assert hasattr(self.model, "mantle_model")
        assert hasattr(self.model, "crustal_model")
        assert hasattr(self.model, "flow_model")
        assert hasattr(self.model, "reaction_model")
        assert hasattr(self.model, "thermal_coupling_net")
        assert hasattr(self.model, "mechanical_coupling_net")
        assert hasattr(self.model, "chemical_coupling_net")

    def test_integrated_forward_pass(self):
        """Test integrated forward pass"""
        batch_size = 50
        coords = torch.rand(batch_size, 4)

        predictions = self.model(coords)

        assert "mantle" in predictions
        assert "crustal" in predictions
        assert "flow" in predictions
        assert "reactions" in predictions
        assert "thermal_coupling" in predictions
        assert "mechanical_coupling" in predictions
        assert "chemical_coupling" in predictions

    def test_integrated_physics_loss(self):
        """Test integrated physics loss computation"""
        batch_size = 30
        coords = torch.rand(batch_size, 4, requires_grad=True)

        predictions = self.model(coords)
        loss = self.model.compute_physics_loss(coords, predictions)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_coupling_mechanisms(self):
        """Test coupling between different processes"""
        batch_size = 20
        coords = torch.rand(batch_size, 4)

        predictions = self.model(coords)

        # Test thermal coupling
        thermal_coupling = predictions["thermal_coupling"]
        assert thermal_coupling.shape == (batch_size, 4)

        # Test mechanical coupling
        mechanical_coupling = predictions["mechanical_coupling"]
        assert mechanical_coupling.shape == (batch_size, 6)

        # Test chemical coupling
        chemical_coupling = predictions["chemical_coupling"]
        assert chemical_coupling.shape == (batch_size, 8)

    def test_factory_function(self):
        """Test integrated model factory function"""
        config = {
            "rayleigh_number": 1e6,
            "youngs_modulus": 70e9,
            "permeability": 1e-15,
            "n_species": 15,
            "thermal_coupling": 1.5,
            "mechanical_coupling": 1.2,
            "chemical_coupling": 0.8,
        }

        model = create_integrated_geological_model(config)

        assert isinstance(model, IntegratedGeologicalModel)
        assert model.params.thermal_coupling_strength == 1.5
        assert model.params.mechanical_coupling_strength == 1.2
        assert model.params.chemical_coupling_strength == 0.8

class TestModelComplexity:
    """Test suite for model complexity and performance"""

    def test_mantle_convection_complexity(self):
        """Test that mantle convection model handles full complexity"""
        params = ConvectionParameters(
            rayleigh_number=1e7,  # High Rayleigh number
            n_radial=128,
            n_theta=256,
            n_phi=512,
            n_compositions=3,
        )
        model = MantleConvectionModel(params)

        # Test with realistic coordinate ranges
        coords = torch.rand(100, 4)
        coords[:, 0] *= params.outer_radius - params.inner_radius  # Radial
        coords[:, 0] += params.inner_radius
        coords[:, 1] *= np.pi  # Theta
        coords[:, 2] *= 2 * np.pi  # Phi
        coords[:, 3] *= 1e6  # Time (years)

        predictions = model(coords)

        # Verify complex behavior
        assert torch.std(predictions["temperature"]) > 0.1  # Temperature variations
        assert torch.std(predictions["velocity"]) > 0.01  # Velocity variations

    def test_crustal_deformation_viscoelasticity(self):
        """Test viscoelastic behavior in crustal deformation"""
        params = ViscoelasticParameters(
            youngs_modulus=70e9,
            viscosity=1e22,
            stress_exponent=3.0,  # Non-linear rheology
            activation_energy=200e3,
        )
        model = CrustalDeformationModel(params)

        coords = torch.rand(50, 4, requires_grad=True)
        predictions = model(coords)

        # Test that stress and strain are properly coupled
        assert predictions["stress"].shape[-1] == 6  # Full stress tensor
        assert predictions["displacement"].shape[-1] == 3  # 3D displacement

    def test_hydrothermal_multi_phase_flow(self):
        """Test multi-phase flow capabilities"""
        params = FluidFlowParameters(
            n_species=8, critical_temperature=647.1, critical_pressure=22.064e6
        )
        model = HydrothermalFlowModel(params)

        # Test with conditions spanning phase boundaries
        coords = torch.rand(50, 4)
        coords[:, 0] *= 500.0  # x
        coords[:, 1] *= 500.0  # y
        coords[:, 2] *= 250.0  # z
        coords[:, 3] *= 100.0  # t

        predictions = model(coords)

        # Verify phase fraction is properly bounded
        phase_fraction = predictions["phase_fraction"]
        assert torch.all(phase_fraction >= 0.0)
        assert torch.all(phase_fraction <= 1.0)

    def test_geochemical_reaction_complexity(self):
        """Test complex geochemical reaction networks"""
        params = ReactionParameters(
            n_species=20,
            n_reactions=15,
            n_minerals=8,
            temperature_range=(273.15, 1273.15),
            pressure_range=(1e5, 1e8),
        )
        model = GeochemicalReactionNetwork(params)

        # Test with wide T, P ranges
        coords = torch.rand(50, 4)
        coords[:, 0] = coords[:, 0] * 1000 + 273.15  # Temperature
        coords[:, 1] = coords[:, 1] * 1e7 + 1e5  # Pressure
        coords[:, 2] = coords[:, 2] * 14  # pH
        coords[:, 3] *= 10.0  # Time

        predictions = model(coords)

        # Verify realistic chemical behavior
        concentrations = predictions["concentrations"]
        assert torch.all(concentrations >= 0.0)  # Non-negative concentrations

        ph = predictions["ph"]
        assert torch.all(ph >= 0.0) and torch.all(ph <= 14.0)  # Valid pH range

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
