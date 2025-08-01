"""
Tests for deep Earth and mantle dynamics modeling

Tests for core-mantle boundary, mantle plumes, convection,
hotspot volcanism, and large igneous provinces.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.mantle_dynamics import (
    ConvectionRegime,
    CoreMantleBoundaryModeler,
    CoreMantleBoundaryState,
    DeepEarthMantleDynamicsSimulator,
    HotspotType,
    HotspotVolcanismModeler,
    HotspotVolcanismState,
    LargeIgneousProvinceModeler,
    LargeIgneousProvinceState,
    MantleConvectionModeler,
    MantleConvectionState,
    MantleLayer,
    MantlePlumeModeler,
    MantlePlumeState,
    PlumeStage,
)
from core.pinn_engine import PINNEngine
from models.geological import GeologicalLocation

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
def sample_time_span():
    """Sample time span for testing"""
    start_time = datetime(1000, 1, 1)  # 1000 years ago
    end_time = datetime(2000, 1, 1)  # 2000 years ago
    return (start_time, end_time)

@pytest.fixture
def sample_location():
    """Sample geological location"""
    return GeologicalLocation(latitude=19.5, longitude=-155.5, elevation=0.0)  # Hawaii

class TestCoreMantleBoundaryModeler:
    """Tests for core-mantle boundary modeler"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test CMB modeler initialization"""
        modeler = CoreMantleBoundaryModeler(mock_pinn_engine, device)

        assert modeler.pinn_engine == mock_pinn_engine
        assert modeler.device == device
        assert isinstance(modeler.cmb_parameters, dict)
        mock_pinn_engine.create_temporal_model.assert_called_once()

    def test_simulate_cmb_dynamics(self, mock_pinn_engine, device, sample_time_span):
        """Test CMB dynamics simulation"""
        modeler = CoreMantleBoundaryModeler(mock_pinn_engine, device)

        initial_conditions = {
            "base_temperature": 2500,
            "temperature_variations": 0.1,
            "chemical_composition": {"Fe": 0.8, "Ni": 0.1, "Si": 0.05, "O": 0.05},
            "d_prime_thickness": 200e3,
        }

        time_step = timedelta(days=365 * 1000000)  # 1 Myr

        cmb_evolution = modeler.simulate_cmb_dynamics(
            initial_conditions, sample_time_span, time_step
        )

        assert isinstance(cmb_evolution, list)
        assert len(cmb_evolution) > 0
        assert all(
            isinstance(state, CoreMantleBoundaryState) for state in cmb_evolution
        )

        # Check that properties are reasonable
        for state in cmb_evolution:
            assert state.temperature_cmb > 2000  # Reasonable CMB temperature
            assert state.heat_flux_cmb > 0
            assert state.d_prime_thickness > 0
            assert isinstance(state.post_perovskite_stability, bool)

    def test_initialize_cmb_state(self, mock_pinn_engine, device):
        """Test CMB state initialization"""
        modeler = CoreMantleBoundaryModeler(mock_pinn_engine, device)

        initial_conditions = {
            "base_temperature": 2500,
            "temperature_variations": 0.1,
            "chemical_composition": {"Fe": 0.8, "Ni": 0.1},
        }

        cmb_state = modeler._initialize_cmb_state(initial_conditions)

        assert isinstance(cmb_state, dict)
        assert "coordinates" in cmb_state
        assert "temperature_field" in cmb_state
        assert "heat_flux" in cmb_state
        assert "chemical_composition" in cmb_state

        # Check temperature field has variations
        temp_field = cmb_state["temperature_field"]
        assert np.min(temp_field) < np.max(temp_field)

class TestMantlePlumeModeler:
    """Tests for mantle plume modeler"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test plume modeler initialization"""
        modeler = MantlePlumeModeler(mock_pinn_engine, device)

        assert modeler.pinn_engine == mock_pinn_engine
        assert modeler.device == device
        assert isinstance(modeler.plume_parameters, dict)
        mock_pinn_engine.create_temporal_model.assert_called_once()

    def test_simulate_plume_evolution(self, mock_pinn_engine, device, sample_time_span):
        """Test plume evolution simulation"""
        modeler = MantlePlumeModeler(mock_pinn_engine, device)

        initiation_location = np.array([0, 0, -2800e3])  # Near CMB
        initial_conditions = {
            "temperature_anomaly": 200,
            "initial_radius": 50e3,
            "composition": {"mg_number": 0.92},
        }

        time_step = timedelta(days=365 * 10000)  # 10 kyr

        plume_evolution = modeler.simulate_plume_evolution(
            initiation_location, initial_conditions, sample_time_span, time_step
        )

        assert isinstance(plume_evolution, list)
        assert len(plume_evolution) > 0
        assert all(isinstance(state, MantlePlumeState) for state in plume_evolution)

        # Check plume evolution
        first_state = plume_evolution[0]
        assert first_state.stage in [PlumeStage.INITIATION, PlumeStage.HEAD_FORMATION]
        assert first_state.temperature_anomaly > 0
        assert first_state.head_radius > 0

    def test_determine_plume_stage(self, mock_pinn_engine, device):
        """Test plume stage determination"""
        modeler = MantlePlumeModeler(mock_pinn_engine, device)

        # Test different ages
        test_state = {
            "stage": PlumeStage.INITIATION,
            "head_location": np.array([0, 0, -1000e3]),  # 1000 km depth
            "temperature_anomaly": 200,
        }

        # Young plume should be in head formation
        stage = modeler._determine_plume_stage(test_state, 1e6)  # 1 Myr
        assert stage == PlumeStage.HEAD_FORMATION

        # Shallow plume should reach surface expression
        test_state["head_location"][2] = -50e3  # 50 km depth
        stage = modeler._determine_plume_stage(test_state, 20e6)  # 20 Myr
        assert stage == PlumeStage.SURFACE_EXPRESSION

class TestMantleConvectionModeler:
    """Tests for mantle convection modeler"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test convection modeler initialization"""
        modeler = MantleConvectionModeler(mock_pinn_engine, device)

        assert modeler.pinn_engine == mock_pinn_engine
        assert modeler.device == device
        assert isinstance(modeler.convection_parameters, dict)
        mock_pinn_engine.create_temporal_model.assert_called_once()

    def test_simulate_mantle_convection(
        self, mock_pinn_engine, device, sample_time_span
    ):
        """Test mantle convection simulation"""
        modeler = MantleConvectionModeler(mock_pinn_engine, device)

        initial_conditions = {
            "perturbation_amplitude": 100,
            "reference_composition": 0.9,
            "compositional_perturbation": 0.05,
        }

        time_step = timedelta(days=365 * 100000)  # 100 kyr

        convection_evolution = modeler.simulate_mantle_convection(
            initial_conditions, sample_time_span, time_step
        )

        assert isinstance(convection_evolution, list)
        assert len(convection_evolution) > 0
        assert all(
            isinstance(state, MantleConvectionState) for state in convection_evolution
        )

        # Check convection properties
        for state in convection_evolution:
            assert isinstance(state.convection_regime, ConvectionRegime)
            assert state.rayleigh_number > 0
            assert state.nusselt_number > 0
            assert 0 <= state.heat_transport_efficiency <= 1

    def test_initialize_convection_state(self, mock_pinn_engine, device):
        """Test convection state initialization"""
        modeler = MantleConvectionModeler(mock_pinn_engine, device)

        initial_conditions = {
            "perturbation_amplitude": 100,
            "reference_composition": 0.9,
        }

        convection_state = modeler._initialize_convection_state(initial_conditions)

        assert isinstance(convection_state, dict)
        assert "coordinates" in convection_state
        assert "temperature_field" in convection_state
        assert "velocity_field" in convection_state
        assert "composition_field" in convection_state

        # Check field shapes
        temp_field = convection_state["temperature_field"]
        vel_field = convection_state["velocity_field"]
        assert temp_field.ndim == 3
        assert vel_field.ndim == 4  # 3D + 3 velocity components
        assert vel_field.shape[-1] == 3

class TestHotspotVolcanismModeler:
    """Tests for hotspot volcanism modeler"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test hotspot modeler initialization"""
        modeler = HotspotVolcanismModeler(mock_pinn_engine, device)

        assert modeler.pinn_engine == mock_pinn_engine
        assert modeler.device == device
        assert isinstance(modeler.hotspot_parameters, dict)
        mock_pinn_engine.create_temporal_model.assert_called_once()

    def test_simulate_hotspot_volcanism(
        self, mock_pinn_engine, device, sample_location, sample_time_span
    ):
        """Test hotspot volcanism simulation"""
        modeler = HotspotVolcanismModeler(mock_pinn_engine, device)

        plume_properties = {
            "composition": {"SiO2": 45.0, "MgO": 15.0},
            "temperature": 1600,
            "source_depth": 2800e3,
        }

        plate_motion = {"velocity": [0.08, 0.02]}  # m/year

        time_step = timedelta(days=365 * 10000)  # 10 kyr

        hotspot_evolution = modeler.simulate_hotspot_volcanism(
            sample_location,
            HotspotType.DEEP_PLUME,
            plume_properties,
            plate_motion,
            sample_time_span,
            time_step,
        )

        assert isinstance(hotspot_evolution, list)
        assert len(hotspot_evolution) > 0
        assert all(
            isinstance(state, HotspotVolcanismState) for state in hotspot_evolution
        )

        # Check hotspot properties
        for state in hotspot_evolution:
            assert state.hotspot_type == HotspotType.DEEP_PLUME
            assert state.volcanic_flux > 0
            assert state.temperature > 1000
            assert len(state.age_progression) >= 0

    def test_initialize_hotspot_state(self, mock_pinn_engine, device, sample_location):
        """Test hotspot state initialization"""
        modeler = HotspotVolcanismModeler(mock_pinn_engine, device)

        plume_properties = {
            "composition": {"SiO2": 45.0, "MgO": 15.0},
            "temperature": 1600,
        }

        plate_motion = {"velocity": [0.05, 0.0]}

        hotspot_state = modeler._initialize_hotspot_state(
            "test_hotspot",
            sample_location,
            HotspotType.DEEP_PLUME,
            plume_properties,
            plate_motion,
        )

        assert isinstance(hotspot_state, dict)
        assert hotspot_state["hotspot_id"] == "test_hotspot"
        assert hotspot_state["hotspot_type"] == HotspotType.DEEP_PLUME
        assert hotspot_state["volcanic_flux"] > 0
        assert hotspot_state["temperature"] == 1600

class TestLargeIgneousProvinceModeler:
    """Tests for large igneous province modeler"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test LIP modeler initialization"""
        modeler = LargeIgneousProvinceModeler(mock_pinn_engine, device)

        assert modeler.pinn_engine == mock_pinn_engine
        assert modeler.device == device
        assert isinstance(modeler.lip_parameters, dict)
        mock_pinn_engine.create_temporal_model.assert_called_once()

    def test_simulate_lip_formation(self, mock_pinn_engine, device, sample_time_span):
        """Test LIP formation simulation"""
        modeler = LargeIgneousProvinceModeler(mock_pinn_engine, device)

        plume_head_arrival = {
            "diameter": 2000e3,
            "temperature_anomaly": 250,
            "composition": {"SiO2": 48.0, "MgO": 12.0},
        }

        crustal_properties = {
            "thickness": 35e3,
            "composition": "continental",
            "age": 2e9,
        }

        time_step = timedelta(days=365 * 1000)  # 1 kyr

        lip_evolution = modeler.simulate_lip_formation(
            plume_head_arrival, crustal_properties, sample_time_span, time_step
        )

        assert isinstance(lip_evolution, list)
        assert len(lip_evolution) > 0
        assert all(
            isinstance(state, LargeIgneousProvinceState) for state in lip_evolution
        )

        # Check LIP evolution
        for state in lip_evolution:
            assert state.formation_stage in [
                "initiation",
                "main_phase",
                "waning",
                "complete",
            ]
            assert state.total_volume >= 0
            assert state.eruption_rate >= 0
            assert state.areal_extent >= 0

    def test_calculate_environmental_impact(self, mock_pinn_engine, device):
        """Test environmental impact calculation"""
        modeler = LargeIgneousProvinceModeler(mock_pinn_engine, device)

        volume_increment = 1e9  # m³
        eruption_rate = 1e6  # m³/year
        composition = {"SO2": 0.1, "CO2": 0.05}

        impact = modeler._calculate_environmental_impact(
            volume_increment, eruption_rate, composition
        )

        assert isinstance(impact, dict)
        assert "SO2_release" in impact
        assert "CO2_release" in impact
        assert "temperature_change" in impact
        assert all(isinstance(v, float) for v in impact.values())

class TestDeepEarthMantleDynamicsSimulator:
    """Tests for deep Earth mantle dynamics simulator"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test simulator initialization"""
        simulator = DeepEarthMantleDynamicsSimulator(mock_pinn_engine, device)

        assert simulator.pinn_engine == mock_pinn_engine
        assert simulator.device == device
        assert hasattr(simulator, "cmb_modeler")
        assert hasattr(simulator, "plume_modeler")
        assert hasattr(simulator, "convection_modeler")
        assert hasattr(simulator, "hotspot_modeler")
        assert hasattr(simulator, "lip_modeler")

    def test_run_comprehensive_mantle_simulation(
        self, mock_pinn_engine, device, sample_time_span
    ):
        """Test comprehensive mantle simulation"""
        simulator = DeepEarthMantleDynamicsSimulator(mock_pinn_engine, device)

        simulation_config = {
            "cmb_dynamics": {
                "initial_conditions": {
                    "base_temperature": 2500,
                    "temperature_variations": 0.1,
                }
            },
            "mantle_plumes": {
                "plumes": [
                    {
                        "id": "test_plume",
                        "initiation_location": [0, 0, -2800e3],
                        "initial_conditions": {
                            "temperature_anomaly": 200,
                            "composition": {"mg_number": 0.92},
                        },
                    }
                ]
            },
            "hotspot_volcanism": {
                "hotspots": [
                    {
                        "id": "test_hotspot",
                        "location": {
                            "latitude": 19.5,
                            "longitude": -155.5,
                            "elevation": 0.0,
                        },
                        "type": "deep_plume",
                        "plume_properties": {
                            "temperature": 1600,
                            "composition": {"SiO2": 45.0},
                        },
                        "plate_motion": {"velocity": [0.08, 0.02]},
                    }
                ]
            },
        }

        result = simulator.run_comprehensive_mantle_simulation(
            simulation_config, sample_time_span
        )

        assert isinstance(result, dict)
        assert "simulation_id" in result
        assert "time_range" in result
        assert "components_simulated" in result
        assert "results" in result

        # Check that requested components were simulated
        results = result["results"]
        assert "cmb_evolution" in results
        assert "plume_evolution" in results
        assert "hotspot_evolution" in results

        # Check plume results structure
        plume_results = results["plume_evolution"]
        assert isinstance(plume_results, list)
        assert len(plume_results) == 1
        assert plume_results[0]["plume_id"] == "test_plume"

        # Check hotspot results structure
        hotspot_results = results["hotspot_evolution"]
        assert isinstance(hotspot_results, list)
        assert len(hotspot_results) == 1
        assert hotspot_results[0]["hotspot_id"] == "test_hotspot"

class TestDataStructures:
    """Tests for data structure classes"""

    def test_core_mantle_boundary_state(self):
        """Test CoreMantleBoundaryState data structure"""
        state = CoreMantleBoundaryState(
            timestamp=datetime.utcnow(),
            temperature_cmb=2500.0,
            heat_flux_cmb=0.1,
            chemical_composition={"Fe": 0.8, "Ni": 0.1},
            d_prime_thickness=200e3,
            seismic_velocity_anomaly={"vs_anomaly": -0.05, "vp_anomaly": -0.02},
            thermal_boundary_layer_thickness=50e3,
            ultra_low_velocity_zones=[],
            post_perovskite_stability=True,
        )

        assert isinstance(state, CoreMantleBoundaryState)
        assert state.temperature_cmb == 2500.0
        assert state.post_perovskite_stability is True
        assert "Fe" in state.chemical_composition

    def test_mantle_plume_state(self):
        """Test MantlePlumeState data structure"""
        state = MantlePlumeState(
            timestamp=datetime.utcnow(),
            plume_id="test_plume",
            stage=PlumeStage.ASCENT,
            head_location=np.array([0, 0, -1000e3]),
            tail_location=np.array([0, 0, -2800e3]),
            head_radius=500e3,
            tail_radius=100e3,
            temperature_anomaly=200.0,
            ascent_velocity=0.1,
            buoyancy_flux=1e4,
            surface_expression=None,
            volume_flux=1e3,
            composition={"mg_number": 0.92},
        )

        assert isinstance(state, MantlePlumeState)
        assert state.stage == PlumeStage.ASCENT
        assert state.head_radius == 500e3
        assert state.temperature_anomaly == 200.0

    def test_mantle_convection_state(self):
        """Test MantleConvectionState data structure"""
        state = MantleConvectionState(
            timestamp=datetime.utcnow(),
            convection_regime=ConvectionRegime.WHOLE_MANTLE,
            temperature_field=np.random.randn(10, 10, 10),
            velocity_field=np.random.randn(10, 10, 10, 3),
            composition_field=np.random.randn(10, 10, 10),
            rayleigh_number=1e6,
            nusselt_number=100,
            heat_transport_efficiency=0.8,
            upwelling_locations=[],
            downwelling_locations=[],
            slab_penetration_depth={"pacific": 1500e3},
        )

        assert isinstance(state, MantleConvectionState)
        assert state.convection_regime == ConvectionRegime.WHOLE_MANTLE
        assert state.rayleigh_number == 1e6
        assert state.temperature_field.shape == (10, 10, 10)
        assert state.velocity_field.shape == (10, 10, 10, 3)

    def test_hotspot_volcanism_state(self, sample_location):
        """Test HotspotVolcanismState data structure"""
        state = HotspotVolcanismState(
            timestamp=datetime.utcnow(),
            hotspot_id="hawaii",
            hotspot_type=HotspotType.DEEP_PLUME,
            location=sample_location,
            volcanic_flux=1e6,
            temperature=1600.0,
            composition={"SiO2": 45.0, "MgO": 15.0},
            age_progression=[(0, 0), (100e3, 1e6)],
            track_direction=np.array([1, 0]),
            plume_source_depth=2800e3,
            crustal_thickness_anomaly=15e3,
        )

        assert isinstance(state, HotspotVolcanismState)
        assert state.hotspot_type == HotspotType.DEEP_PLUME
        assert state.volcanic_flux == 1e6
        assert len(state.age_progression) == 2

    def test_large_igneous_province_state(self):
        """Test LargeIgneousProvinceState data structure"""
        state = LargeIgneousProvinceState(
            timestamp=datetime.utcnow(),
            lip_id="siberian_traps",
            formation_stage="main_phase",
            total_volume=1e15,
            eruption_rate=1e6,
            areal_extent=5e12,
            duration=2e6,
            plume_head_diameter=2000e3,
            environmental_impact={"SO2_release": 1e12, "CO2_release": 1e13},
            geochemical_signature={"SiO2": 48.0, "TiO2": 2.0},
        )

        assert isinstance(state, LargeIgneousProvinceState)
        assert state.formation_stage == "main_phase"
        assert state.total_volume == 1e15
        assert "SO2_release" in state.environmental_impact

class TestEnums:
    """Tests for enum classes"""

    def test_mantle_layer_enum(self):
        """Test MantleLayer enum"""
        assert MantleLayer.UPPER_MANTLE == "upper_mantle"
        assert MantleLayer.TRANSITION_ZONE == "transition_zone"
        assert MantleLayer.LOWER_MANTLE == "lower_mantle"
        assert MantleLayer.D_DOUBLE_PRIME == "d_double_prime"

    def test_plume_stage_enum(self):
        """Test PlumeStage enum"""
        assert PlumeStage.INITIATION == "initiation"
        assert PlumeStage.HEAD_FORMATION == "head_formation"
        assert PlumeStage.ASCENT == "ascent"
        assert PlumeStage.SURFACE_EXPRESSION == "surface_expression"
        assert PlumeStage.TAIL_PHASE == "tail_phase"
        assert PlumeStage.DECAY == "decay"

    def test_convection_regime_enum(self):
        """Test ConvectionRegime enum"""
        assert ConvectionRegime.LAYERED == "layered"
        assert ConvectionRegime.WHOLE_MANTLE == "whole_mantle"
        assert ConvectionRegime.MIXED == "mixed"
        assert ConvectionRegime.STAGNANT_LID == "stagnant_lid"

    def test_hotspot_type_enum(self):
        """Test HotspotType enum"""
        assert HotspotType.DEEP_PLUME == "deep_plume"
        assert HotspotType.SHALLOW_PLUME == "shallow_plume"
        assert HotspotType.EDGE_DRIVEN == "edge_driven"
        assert HotspotType.LITHOSPHERIC == "lithospheric"

if __name__ == "__main__":
    pytest.main([__file__])
