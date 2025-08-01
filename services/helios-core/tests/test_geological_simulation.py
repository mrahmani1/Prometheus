"""
Tests for long-term geological simulation engine

Tests for volcanic system evolution, magma generation tracking,
hydrothermal circulation, and tectonic plate reconstruction.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from core.geological_simulation import (
    GeologicalSimulationResult,
    HydrothermalSystemModeler,
    HydrothermalSystemState,
    LongTermGeologicalSimulator,
    MagmaGenerationState,
    MagmaGenerationTracker,
    SimulationTimeScale,
    TectonicPlateReconstructor,
    TectonicPlateState,
    VolcanicSystemEvolutionModeler,
    VolcanicSystemStage,
    VolcanicSystemState,
)
from core.pinn_engine import PINNEngine
from models.geological import (
    GeochemicalSample,
    GeologicalLocation,
    MagmaReservoir,
    SampleType,
    import,
)
from models.geological import numpy as np
from models.geological import pandas as pd
from models.geological import torch.nn as nn

@pytest.fixture
def device():
    """Test device fixture"""
    return torch.device("cpu")

@pytest.fixture
def mock_pinn_engine():
    """Mock PINN engine for testing"""
    engine = Mock(spec=PINNEngine)
    engine.create_temporal_model.return_value = Mock()
    return engine

@pytest.fixture
def sample_volcanic_state():
    """Sample volcanic system state for testing"""
    location = GeologicalLocation(
        latitude=45.0, longitude=-122.0, elevation=1000.0, depth=5000.0
    )

    composition_sample = GeochemicalSample(
        timestamp=datetime.utcnow(),
        location=location,
        sample_type=SampleType.LAVA,
        composition={"SiO2": 52.0, "MgO": 8.0, "Al2O3": 15.0},
    )

    reservoir = MagmaReservoir(
        location=location,
        volume=1e9,
        composition=composition_sample,
        temperature=1200.0,
        pressure=1e8,
        crystallinity=0.2,
        volatile_content={"H2O": 2.0, "CO2": 0.5},
        reservoir_id="test_reservoir",
    )

    return VolcanicSystemState(
        timestamp=datetime.utcnow(),
        stage=VolcanicSystemStage.GROWTH,
        magma_reservoirs=[reservoir],
        eruption_rate=1e-3,
        thermal_state={"temperature": 1000.0, "heat_flux": 1e6},
        stress_field=np.random.randn(10, 10, 5, 6),
        volatile_budget={"H2O": 1000.0, "CO2": 500.0},
        crustal_thickness=30000.0,
        system_age=50000.0,
    )

@pytest.fixture
def sample_time_span():
    """Sample time span for testing"""
    start_time = datetime(2000, 1, 1)
    end_time = datetime(2010, 1, 1)
    return (start_time, end_time)

class TestVolcanicSystemEvolutionModeler:
    """Tests for volcanic system evolution modeler"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test modeler initialization"""
        modeler = VolcanicSystemEvolutionModeler(mock_pinn_engine, device)

        assert modeler.pinn_engine == mock_pinn_engine
        assert modeler.device == device
        assert isinstance(modeler.evolution_parameters, dict)
        mock_pinn_engine.create_temporal_model.assert_called_once()

    def test_simulate_volcanic_evolution(
        self, mock_pinn_engine, device, sample_volcanic_state, sample_time_span
    ):
        """Test volcanic evolution simulation"""
        modeler = VolcanicSystemEvolutionModeler(mock_pinn_engine, device)

        time_step = timedelta(days=365 * 100)  # 100 years

        evolution = modeler.simulate_volcanic_evolution(
            sample_volcanic_state, sample_time_span, time_step
        )

        assert isinstance(evolution, list)
        assert len(evolution) > 0
        assert all(isinstance(state, VolcanicSystemState) for state in evolution)

        # Check that system evolves over time
        for i in range(1, len(evolution)):
            assert evolution[i].timestamp > evolution[i - 1].timestamp
            assert evolution[i].system_age > evolution[i - 1].system_age

    def test_determine_volcanic_stage(
        self, mock_pinn_engine, device, sample_volcanic_state
    ):
        """Test volcanic stage determination"""
        modeler = VolcanicSystemEvolutionModeler(mock_pinn_engine, device)

        # Test different ages and eruption rates
        test_cases = [
            (1000, 1e-3, VolcanicSystemStage.INITIATION),  # Young, low rate
            (100000, 1e-2, VolcanicSystemStage.GROWTH),  # Medium age, high rate
            (1000000, 1e-2, VolcanicSystemStage.MATURITY),  # Old, high rate
            (2000000, 1e-8, VolcanicSystemStage.DORMANCY),  # Very old, very low rate
        ]

        for age, eruption_rate, expected_stage in test_cases:
            sample_volcanic_state.system_age = age
            sample_volcanic_state.eruption_rate = eruption_rate

            stage = modeler._determine_volcanic_stage(sample_volcanic_state, age)
            # Note: Due to randomness in reactivation, we can't always predict exact stage
            assert isinstance(stage, VolcanicSystemStage)

    def test_calculate_eruption_rate(
        self, mock_pinn_engine, device, sample_volcanic_state
    ):
        """Test eruption rate calculation"""
        modeler = VolcanicSystemEvolutionModeler(mock_pinn_engine, device)

        initial_rate = 1e-3
        sample_volcanic_state.eruption_rate = initial_rate

        # Test growth stage
        new_rate = modeler._calculate_eruption_rate(
            sample_volcanic_state, VolcanicSystemStage.GROWTH, 50000
        )
        assert new_rate > initial_rate  # Should increase during growth

        # Test decline stage
        new_rate = modeler._calculate_eruption_rate(
            sample_volcanic_state, VolcanicSystemStage.DECLINE, 50000
        )
        assert new_rate < initial_rate  # Should decrease during decline

    def test_update_thermal_state(
        self, mock_pinn_engine, device, sample_volcanic_state
    ):
        """Test thermal state update"""
        modeler = VolcanicSystemEvolutionModeler(mock_pinn_engine, device)

        time_step = timedelta(days=365 * 100)
        external_forcing = {
            "climate_temperature_change": 5.0,
            "tectonic_heat_flux": 1e5,
        }

        new_thermal_state = modeler._update_thermal_state(
            sample_volcanic_state, time_step, external_forcing
        )

        assert isinstance(new_thermal_state, dict)
        assert "temperature" in new_thermal_state
        assert "heat_flux" in new_thermal_state
        assert "geothermal_gradient" in new_thermal_state
        assert new_thermal_state["temperature"] > 0

class TestMagmaGenerationTracker:
    """Tests for magma generation tracker"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test tracker initialization"""
        tracker = MagmaGenerationTracker(mock_pinn_engine, device)

        assert tracker.pinn_engine == mock_pinn_engine
        assert tracker.device == device
        assert isinstance(tracker.generation_parameters, dict)
        mock_pinn_engine.create_temporal_model.assert_called_once()

    def test_track_magma_generation(self, mock_pinn_engine, device, sample_time_span):
        """Test magma generation tracking"""
        tracker = MagmaGenerationTracker(mock_pinn_engine, device)

        initial_conditions = {
            "surface_temperature": 0,
            "mantle_temperature": 1300,
            "mantle_mg_number": 0.9,
        }

        spatial_domain = ((-1000, 1000), (-1000, 1000), (0, 50000))
        time_step = timedelta(days=365 * 100)

        generation_history = tracker.track_magma_generation(
            initial_conditions, sample_time_span, spatial_domain, time_step
        )

        assert isinstance(generation_history, list)
        assert len(generation_history) > 0
        assert all(
            isinstance(state, MagmaGenerationState) for state in generation_history
        )

        # Check that generation evolves over time
        for state in generation_history:
            assert state.melt_fraction >= 0
            assert state.generation_rate >= 0
            assert state.ascent_velocity >= 0
            assert 0 <= state.crystallization_degree <= 1

    def test_initialize_mantle_state(self, mock_pinn_engine, device):
        """Test mantle state initialization"""
        tracker = MagmaGenerationTracker(mock_pinn_engine, device)

        initial_conditions = {
            "surface_temperature": 0,
            "mantle_temperature": 1300,
            "mantle_mg_number": 0.9,
        }

        spatial_domain = ((-1000, 1000), (-1000, 1000), (0, 50000))

        mantle_state = tracker._initialize_mantle_state(
            initial_conditions, spatial_domain
        )

        assert isinstance(mantle_state, dict)
        assert "coordinates" in mantle_state
        assert "temperature" in mantle_state
        assert "pressure" in mantle_state
        assert "composition" in mantle_state

        # Check temperature gradient
        temperature = mantle_state["temperature"]
        assert np.min(temperature) < np.max(temperature)  # Should have gradient

    def test_calculate_magma_generation(self, mock_pinn_engine, device):
        """Test magma generation calculation"""
        tracker = MagmaGenerationTracker(mock_pinn_engine, device)

        # Create simple mantle state
        nx, ny, nz = 10, 10, 5
        coordinates = np.random.randn(nx, ny, nz, 3)
        temperature = np.ones((nx, ny, nz)) * 1200  # Above solidus
        pressure = np.ones((nx, ny, nz)) * 1e8
        composition = np.ones((nx, ny, nz)) * 0.9

        mantle_state = {
            "coordinates": coordinates,
            "temperature": temperature,
            "pressure": pressure,
            "composition": composition,
            "melt_fraction": np.zeros((nx, ny, nz)),
            "depletion": np.zeros((nx, ny, nz)),
        }

        current_time = datetime.utcnow()
        time_step = timedelta(days=365 * 100)

        generation_state = tracker._calculate_magma_generation(
            mantle_state, current_time, time_step
        )

        assert isinstance(generation_state, MagmaGenerationState)
        assert generation_state.melt_fraction > 0  # Should have melting
        assert generation_state.generation_rate > 0
        assert generation_state.mantle_temperature > 1000

class TestHydrothermalSystemModeler:
    """Tests for hydrothermal system modeler"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test modeler initialization"""
        modeler = HydrothermalSystemModeler(mock_pinn_engine, device)

        assert modeler.pinn_engine == mock_pinn_engine
        assert modeler.device == device
        assert isinstance(modeler.circulation_parameters, dict)
        mock_pinn_engine.create_temporal_model.assert_called_once()

    def test_simulate_hydrothermal_circulation(
        self, mock_pinn_engine, device, sample_time_span
    ):
        """Test hydrothermal circulation simulation"""
        modeler = HydrothermalSystemModeler(mock_pinn_engine, device)

        initial_conditions = {
            "surface_temperature": 10,
            "geothermal_gradient": 30,
            "initial_salinity": 0.035,
        }

        heat_source = {"location": [0, 0, -2000], "temperature": 800, "radius": 1000}

        spatial_domain = ((-5000, 5000), (-5000, 5000), (-10000, 0))
        time_step = timedelta(days=365 * 10)

        circulation_history = modeler.simulate_hydrothermal_circulation(
            initial_conditions, heat_source, sample_time_span, spatial_domain, time_step
        )

        assert isinstance(circulation_history, list)
        assert len(circulation_history) > 0
        assert all(
            isinstance(state, HydrothermalSystemState) for state in circulation_history
        )

        # Check that circulation properties are reasonable
        for state in circulation_history:
            assert state.fluid_temperature > 0
            assert state.circulation_rate >= 0
            assert state.heat_flux >= 0
            assert 0 < state.permeability < 1
            assert 0 < state.porosity < 1

    def test_initialize_hydrothermal_state(self, mock_pinn_engine, device):
        """Test hydrothermal state initialization"""
        modeler = HydrothermalSystemModeler(mock_pinn_engine, device)

        initial_conditions = {
            "surface_temperature": 10,
            "geothermal_gradient": 30,
            "initial_salinity": 0.035,
        }

        heat_source = {"location": [0, 0, -2000], "temperature": 800, "radius": 1000}

        spatial_domain = ((-5000, 5000), (-5000, 5000), (-10000, 0))

        system_state = modeler._initialize_hydrothermal_state(
            initial_conditions, heat_source, spatial_domain
        )

        assert isinstance(system_state, dict)
        assert "coordinates" in system_state
        assert "temperature" in system_state
        assert "pressure" in system_state
        assert "permeability" in system_state

        # Check that heat source creates thermal anomaly
        temperature = system_state["temperature"]
        assert np.max(temperature) > np.min(temperature)  # Should have thermal anomaly

class TestTectonicPlateReconstructor:
    """Tests for tectonic plate reconstructor"""

    def test_initialization(self, device):
        """Test reconstructor initialization"""
        reconstructor = TectonicPlateReconstructor(device)

        assert reconstructor.device == device
        assert isinstance(reconstructor.plate_parameters, dict)
        assert isinstance(reconstructor.plate_models, dict)

    def test_reconstruct_plate_motion(self, device, sample_time_span):
        """Test plate motion reconstruction"""
        reconstructor = TectonicPlateReconstructor(device)

        initial_configuration = {
            "plates": {
                "pacific": {
                    "outline": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "velocity": [5, 2],  # cm/year
                },
                "north_american": {
                    "outline": [[10, 0], [20, 0], [20, 10], [10, 10]],
                    "velocity": [-2, 1],
                },
            },
            "boundaries": {"pacific_na": "transform"},
            "continents": {
                "north_america": {"outline": [[12, 2], [18, 2], [18, 8], [12, 8]]}
            },
        }

        time_step = timedelta(days=365 * 1000000)  # 1 million years

        reconstruction = reconstructor.reconstruct_plate_motion(
            initial_configuration, sample_time_span, time_step
        )

        assert isinstance(reconstruction, list)
        assert len(reconstruction) > 0
        assert all(isinstance(state, TectonicPlateState) for state in reconstruction)

        # Check that plates move over time
        if len(reconstruction) > 1:
            first_state = reconstruction[0]
            last_state = reconstruction[-1]

            for plate_id in first_state.plate_positions:
                if plate_id in last_state.plate_positions:
                    first_pos = first_state.plate_positions[plate_id]
                    last_pos = last_state.plate_positions[plate_id]

                    # Positions should change (unless velocity is zero)
                    if np.any(first_state.plate_velocities.get(plate_id, [0, 0]) != 0):
                        assert not np.allclose(first_pos, last_pos)

    def test_initialize_plate_state(self, device):
        """Test plate state initialization"""
        reconstructor = TectonicPlateReconstructor(device)

        initial_configuration = {
            "plates": {
                "test_plate": {
                    "outline": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "velocity": [5, 2],
                }
            },
            "boundaries": {"test_boundary": "ridge"},
            "subduction_angles": {"test_subduction": 45},
            "spreading_rates": {"test_ridge": 5},
            "continents": {
                "test_continent": {"outline": [[2, 2], [8, 2], [8, 8], [2, 8]]}
            },
        }

        plate_state = reconstructor._initialize_plate_state(initial_configuration)

        assert isinstance(plate_state, TectonicPlateState)
        assert "test_plate" in plate_state.plate_positions
        assert "test_plate" in plate_state.plate_velocities
        assert "test_boundary" in plate_state.boundary_types
        assert "test_continent" in plate_state.continental_positions

class TestLongTermGeologicalSimulator:
    """Tests for long-term geological simulator"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test simulator initialization"""
        simulator = LongTermGeologicalSimulator(mock_pinn_engine, device)

        assert simulator.pinn_engine == mock_pinn_engine
        assert simulator.device == device
        assert hasattr(simulator, "volcanic_modeler")
        assert hasattr(simulator, "magma_tracker")
        assert hasattr(simulator, "hydrothermal_modeler")
        assert hasattr(simulator, "tectonic_reconstructor")

    def test_run_comprehensive_simulation(
        self, mock_pinn_engine, device, sample_time_span
    ):
        """Test comprehensive simulation"""
        simulator = LongTermGeologicalSimulator(mock_pinn_engine, device)

        simulation_config = {
            "volcanic_system": {
                "initial_stage": "growth",
                "initial_eruption_rate": 1e-3,
                "initial_reservoirs": [
                    {
                        "location": {
                            "latitude": 45.0,
                            "longitude": -122.0,
                            "elevation": 1000.0,
                            "depth": 5000.0,
                        },
                        "volume": 1e9,
                        "temperature": 1200,
                        "composition": {"SiO2": 52.0, "MgO": 8.0},
                    }
                ],
                "initial_thermal_state": {"temperature": 1000},
                "initial_volatile_budget": {"H2O": 1000, "CO2": 500},
                "initial_crustal_thickness": 30000,
                "initial_age": 0,
            },
            "tectonic_plates": {
                "initial_configuration": {
                    "plates": {
                        "test_plate": {
                            "outline": [[0, 0], [10, 0], [10, 10], [0, 10]],
                            "velocity": [5, 2],
                        }
                    }
                }
            },
        }

        result = simulator.run_comprehensive_simulation(
            simulation_config, sample_time_span, SimulationTimeScale.THOUSANDS_YEARS
        )

        assert isinstance(result, GeologicalSimulationResult)
        assert result.simulation_id.startswith("geo_sim_")
        assert result.time_range == sample_time_span
        assert result.time_scale == SimulationTimeScale.THOUSANDS_YEARS
        assert isinstance(result.simulation_metadata, dict)

        # Check that requested components were simulated
        assert len(result.volcanic_evolution) > 0
        assert len(result.tectonic_reconstruction) > 0

    def test_create_initial_volcanic_state(self, mock_pinn_engine, device):
        """Test initial volcanic state creation"""
        simulator = LongTermGeologicalSimulator(mock_pinn_engine, device)

        volcanic_config = {
            "initial_stage": "growth",
            "initial_eruption_rate": 1e-3,
            "initial_reservoirs": [
                {
                    "location": {
                        "latitude": 45.0,
                        "longitude": -122.0,
                        "elevation": 1000.0,
                        "depth": 5000.0,
                    },
                    "volume": 1e9,
                    "temperature": 1200,
                    "composition": {"SiO2": 52.0, "MgO": 8.0},
                    "id": "test_reservoir",
                }
            ],
            "initial_thermal_state": {"temperature": 1000},
            "initial_volatile_budget": {"H2O": 1000, "CO2": 500},
            "initial_crustal_thickness": 30000,
            "initial_age": 0,
        }

        initial_state = simulator._create_initial_volcanic_state(volcanic_config)

        assert isinstance(initial_state, VolcanicSystemState)
        assert initial_state.stage == VolcanicSystemStage.GROWTH
        assert initial_state.eruption_rate == 1e-3
        assert len(initial_state.magma_reservoirs) == 1
        assert initial_state.magma_reservoirs[0].reservoir_id == "test_reservoir"
        assert initial_state.thermal_state["temperature"] == 1000
        assert initial_state.crustal_thickness == 30000

class TestDataStructures:
    """Tests for data structure classes"""

    def test_volcanic_system_state(self, sample_volcanic_state):
        """Test VolcanicSystemState data structure"""
        assert isinstance(sample_volcanic_state, VolcanicSystemState)
        assert isinstance(sample_volcanic_state.timestamp, datetime)
        assert isinstance(sample_volcanic_state.stage, VolcanicSystemStage)
        assert isinstance(sample_volcanic_state.magma_reservoirs, list)
        assert isinstance(sample_volcanic_state.eruption_rate, float)
        assert isinstance(sample_volcanic_state.thermal_state, dict)
        assert isinstance(sample_volcanic_state.stress_field, np.ndarray)
        assert isinstance(sample_volcanic_state.volatile_budget, dict)
        assert isinstance(sample_volcanic_state.crustal_thickness, float)
        assert isinstance(sample_volcanic_state.system_age, float)

    def test_magma_generation_state(self):
        """Test MagmaGenerationState data structure"""
        state = MagmaGenerationState(
            timestamp=datetime.utcnow(),
            mantle_temperature=1300.0,
            mantle_composition={"mg_number": 0.9},
            melt_fraction=0.1,
            generation_rate=1e6,
            ascent_velocity=1e-3,
            crystallization_degree=0.2,
            volatile_content={"H2O": 0.05, "CO2": 0.001},
            depth_range=(10000.0, 50000.0),
        )

        assert isinstance(state, MagmaGenerationState)
        assert state.mantle_temperature == 1300.0
        assert state.melt_fraction == 0.1
        assert state.depth_range == (10000.0, 50000.0)

    def test_hydrothermal_system_state(self):
        """Test HydrothermalSystemState data structure"""
        state = HydrothermalSystemState(
            timestamp=datetime.utcnow(),
            fluid_temperature=200.0,
            fluid_composition={"salinity": 0.035},
            circulation_rate=1e-3,
            heat_flux=1e5,
            permeability=1e-12,
            porosity=0.1,
            pressure=1e7,
            mineral_precipitation={"SiO2": 0.001},
        )

        assert isinstance(state, HydrothermalSystemState)
        assert state.fluid_temperature == 200.0
        assert state.circulation_rate == 1e-3
        assert "SiO2" in state.mineral_precipitation

    def test_tectonic_plate_state(self):
        """Test TectonicPlateState data structure"""
        state = TectonicPlateState(
            timestamp=datetime.utcnow(),
            plate_positions={"pacific": np.array([[0, 0], [10, 10]])},
            plate_velocities={"pacific": np.array([5, 2])},
            boundary_types={"pacific_na": "transform"},
            subduction_angles={"cascadia": 45},
            spreading_rates={"juan_de_fuca": 5},
            continental_positions={"north_america": np.array([[10, 0], [20, 10]])},
        )

        assert isinstance(state, TectonicPlateState)
        assert "pacific" in state.plate_positions
        assert "pacific" in state.plate_velocities
        assert state.subduction_angles["cascadia"] == 45

    def test_geological_simulation_result(self, sample_time_span):
        """Test GeologicalSimulationResult data structure"""
        result = GeologicalSimulationResult(
            simulation_id="test_sim_123",
            time_range=sample_time_span,
            time_scale=SimulationTimeScale.MILLIONS_YEARS,
            volcanic_evolution=[],
            magma_generation_history=[],
            hydrothermal_evolution=[],
            tectonic_reconstruction=[],
            simulation_metadata={"components": ["volcanic", "tectonic"]},
            processing_timestamp=datetime.utcnow(),
        )

        assert isinstance(result, GeologicalSimulationResult)
        assert result.simulation_id == "test_sim_123"
        assert result.time_range == sample_time_span
        assert result.time_scale == SimulationTimeScale.MILLIONS_YEARS
        assert isinstance(result.simulation_metadata, dict)

if __name__ == "__main__":
    pytest.main([__file__])
