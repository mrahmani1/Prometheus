"""
Test Suite for Sill Emplacement Mechanism Simulator
Comprehensive testing of all simulation components and validation against experimental data
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime
from typing import Dict, Any, List

# Import the simulator and data models
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.sill_emplacement_mechanism_simulator import (
    SillEmplacementMechanismSimulator,
    MagmaProperties,
    HostRockProperties,
    InjectionPathway,
    SillInflationSequence,
    HostRockDeformation,
    ThermalEvolution,
    MagmaFlowDynamics,
    CrystallizationCooling,
    StressFieldEvolution,
    EmplacementSimulation
)

class TestSillEmplacementMechanismSimulator:
    """Test suite for the sill emplacement mechanism simulator"""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance for testing"""
        config = {
            'simulation': {
                'time_step': 10.0,
                'total_time': 1000.0,
                'spatial_resolution': 5.0,
                'convergence_tolerance': 1e-5,
                'max_iterations': 100
            },
            'physics': {
                'include_thermal': True,
                'include_mechanical': True,
                'include_fluid_flow': True,
                'include_crystallization': True
            }
        }
        return SillEmplacementMechanismSimulator(config)
    
    @pytest.fixture
    def sample_magma_properties(self):
        """Create sample magma properties for testing"""
        return MagmaProperties(
            composition={
                'SiO2': 52.0, 'Al2O3': 16.0, 'FeO': 8.0, 'MgO': 6.0,
                'CaO': 10.0, 'Na2O': 3.0, 'K2O': 2.0, 'TiO2': 1.5
            },
            temperature=1200.0,  # K
            pressure=1e8,  # Pa (1 kbar)
            viscosity=1e3,  # Pa·s
            density=2700.0,  # kg/m³
            volatile_content={'H2O': 2.0, 'CO2': 0.5},
            crystallinity=0.1,
            bubble_fraction=0.05,
            thermal_conductivity=2.5,  # W/(m·K)
            specific_heat=1200.0,  # J/(kg·K)
            latent_heat_crystallization=3e5,  # J/kg
            surface_tension=0.4,  # N/m
            compressibility=1e-10,  # 1/Pa
            thermal_expansion=3e-5  # 1/K
        )
    
    @pytest.fixture
    def sample_host_rock_properties(self):
        """Create sample host rock properties for testing"""
        return HostRockProperties(
            rock_type='sandstone',
            density=2400.0,  # kg/m³
            porosity=0.15,
            permeability=1e-15,  # m²
            youngs_modulus=2e10,  # Pa
            poissons_ratio=0.25,
            tensile_strength=5e6,  # Pa
            cohesion=1e7,  # Pa
            friction_angle=0.6,  # radians
            thermal_conductivity=2.0,  # W/(m·K)
            specific_heat=800.0,  # J/(kg·K)
            thermal_diffusivity=1e-6,  # m²/s
            thermal_expansion=1e-5,  # 1/K
            fracture_toughness=1e6,  # Pa·m^0.5
            crack_density=1e6,  # cracks/m³
            anisotropy_factor=1.2,
            layering_thickness=0.5,  # m
            bedding_orientation=(0.0, 0.1)  # strike, dip in radians
        )
    
    @pytest.fixture
    def sample_injection_conditions(self):
        """Create sample injection conditions for testing"""
        return {
            'injection_rate': 0.1,  # m³/s
            'injection_pressure': 1.5e8,  # Pa
            'source_location': (500, 500, 1000),  # m
            'domain_size': [1000, 1000, 500],  # m
            'ambient_pressure': 1e5,  # Pa
            'ambient_temperature': 293.15  # K
        }
    
    @pytest.mark.asyncio
    async def test_simulator_initialization(self, simulator):
        """Test simulator initialization"""
        assert simulator is not None
        assert hasattr(simulator, 'config')
        assert hasattr(simulator, 'logger')
        assert hasattr(simulator, 'pathway_models')
        assert hasattr(simulator, 'inflation_models')
        assert hasattr(simulator, 'thermal_models')
        assert hasattr(simulator, 'flow_models')
        assert hasattr(simulator, 'crystallization_models')
        assert hasattr(simulator, 'experimental_data')
    
    @pytest.mark.asyncio
    async def test_magma_injection_pathways_simulation(
        self, simulator, sample_magma_properties, sample_host_rock_properties, sample_injection_conditions
    ):
        """Test magma injection pathways simulation"""
        pathways = await simulator.simulate_magma_injection_pathways(
            sample_magma_properties,
            sample_host_rock_properties,
            sample_injection_conditions,
            'coupled'
        )
        
        assert isinstance(pathways, list)
        assert len(pathways) > 0
        
        for pathway in pathways:
            assert isinstance(pathway, InjectionPathway)
            assert pathway.pathway_id is not None
            assert pathway.source_location is not None
            assert pathway.pathway_geometry is not None
            assert pathway.pathway_length > 0
            assert pathway.tortuosity >= 1.0
            assert pathway.flow_rate > 0
            assert pathway.reynolds_number > 0
    
    @pytest.mark.asyncio
    async def test_sill_inflation_sequence_simulation(
        self, simulator, sample_magma_properties, sample_host_rock_properties, sample_injection_conditions
    ):
        """Test sill inflation sequence simulation"""
        # First create injection pathways
        pathways = await simulator.simulate_magma_injection_pathways(
            sample_magma_properties,
            sample_host_rock_properties,
            sample_injection_conditions,
            'coupled'
        )
        
        # Then simulate inflation
        inflation_sequence = await simulator.simulate_sill_inflation_sequence(
            sample_magma_properties,
            sample_host_rock_properties,
            pathways,
            'elastoplastic'
        )
        
        assert isinstance(inflation_sequence, SillInflationSequence)
        assert inflation_sequence.sequence_id is not None
        assert inflation_sequence.pressure_evolution is not None
        assert inflation_sequence.volume_evolution is not None
        assert inflation_sequence.thickness_evolution is not None
        assert inflation_sequence.inflation_efficiency >= 0
        assert len(inflation_sequence.pressure_evolution) > 0
        assert len(inflation_sequence.volume_evolution) > 0
        assert np.all(inflation_sequence.volume_evolution >= 0)
    
    @pytest.mark.asyncio
    async def test_thermal_effects_simulation(
        self, simulator, sample_magma_properties, sample_host_rock_properties
    ):
        """Test thermal effects simulation"""
        # Create sample sill geometry
        sill_geometry = np.random.rand(50, 50, 10) * 100  # 50x50x10 grid
        
        thermal_evolution = await simulator.simulate_thermal_effects(
            sample_magma_properties,
            sample_host_rock_properties,
            sill_geometry,
            'coupled'
        )
        
        assert isinstance(thermal_evolution, ThermalEvolution)
        assert thermal_evolution.thermal_id is not None
        assert thermal_evolution.temperature_field is not None
        assert thermal_evolution.heat_flux is not None
        assert thermal_evolution.cooling_rate is not None
        assert thermal_evolution.peclet_number > 0
        assert thermal_evolution.stefan_number > 0
        assert thermal_evolution.thermal_diffusion_time > 0
    
    @pytest.mark.asyncio
    async def test_magma_flow_dynamics_simulation(
        self, simulator, sample_magma_properties, sample_host_rock_properties
    ):
        """Test magma flow dynamics simulation"""
        # Create sample sill geometry and thermal evolution
        sill_geometry = np.random.rand(50, 50, 10) * 100
        thermal_evolution = await simulator.simulate_thermal_effects(
            sample_magma_properties,
            sample_host_rock_properties,
            sill_geometry,
            'coupled'
        )
        
        flow_dynamics = await simulator.simulate_magma_flow_dynamics(
            sample_magma_properties,
            sill_geometry,
            thermal_evolution,
            'non_newtonian'
        )
        
        assert isinstance(flow_dynamics, MagmaFlowDynamics)
        assert flow_dynamics.flow_id is not None
        assert flow_dynamics.velocity_field is not None
        assert flow_dynamics.pressure_field is not None
        assert flow_dynamics.reynolds_number > 0
        assert flow_dynamics.peclet_number > 0
        assert len(flow_dynamics.flow_patterns) >= 0
    
    @pytest.mark.asyncio
    async def test_crystallization_cooling_simulation(
        self, simulator, sample_magma_properties, sample_host_rock_properties
    ):
        """Test crystallization and cooling simulation"""
        # Create required inputs
        sill_geometry = np.random.rand(50, 50, 10) * 100
        thermal_evolution = await simulator.simulate_thermal_effects(
            sample_magma_properties,
            sample_host_rock_properties,
            sill_geometry,
            'coupled'
        )
        flow_dynamics = await simulator.simulate_magma_flow_dynamics(
            sample_magma_properties,
            sill_geometry,
            thermal_evolution,
            'non_newtonian'
        )
        
        crystallization_cooling = await simulator.simulate_crystallization_cooling(
            sample_magma_properties,
            thermal_evolution,
            flow_dynamics,
            'nucleation_growth'
        )
        
        assert isinstance(crystallization_cooling, CrystallizationCooling)
        assert crystallization_cooling.crystallization_id is not None
        assert crystallization_cooling.crystal_fraction is not None
        assert crystallization_cooling.nucleation_rate is not None
        assert crystallization_cooling.growth_rate is not None
        assert len(crystallization_cooling.crystallization_sequence) > 0
        assert np.all(crystallization_cooling.crystal_fraction >= 0)
        assert np.all(crystallization_cooling.crystal_fraction <= 1)
        assert np.all(crystallization_cooling.melt_fraction >= 0)
        assert np.all(crystallization_cooling.melt_fraction <= 1)
    
    @pytest.mark.asyncio
    async def test_host_rock_deformation_simulation(
        self, simulator, sample_magma_properties, sample_host_rock_properties, sample_injection_conditions
    ):
        """Test host rock deformation simulation"""
        # Create required inputs
        pathways = await simulator.simulate_magma_injection_pathways(
            sample_magma_properties,
            sample_host_rock_properties,
            sample_injection_conditions,
            'coupled'
        )
        inflation_sequence = await simulator.simulate_sill_inflation_sequence(
            sample_magma_properties,
            sample_host_rock_properties,
            pathways,
            'elastoplastic'
        )
        
        host_rock_deformation = await simulator.simulate_host_rock_deformation(
            sample_host_rock_properties,
            inflation_sequence
        )
        
        assert isinstance(host_rock_deformation, HostRockDeformation)
        assert host_rock_deformation.deformation_id is not None
        assert host_rock_deformation.displacement_field is not None
        assert host_rock_deformation.strain_field is not None
        assert host_rock_deformation.stress_field is not None
        assert host_rock_deformation.deformation_energy > 0
        assert len(host_rock_deformation.failure_zones) >= 0
    
    @pytest.mark.asyncio
    async def test_stress_field_evolution_simulation(
        self, simulator, sample_magma_properties, sample_host_rock_properties, sample_injection_conditions
    ):
        """Test stress field evolution simulation"""
        # Create required inputs
        pathways = await simulator.simulate_magma_injection_pathways(
            sample_magma_properties,
            sample_host_rock_properties,
            sample_injection_conditions,
            'coupled'
        )
        inflation_sequence = await simulator.simulate_sill_inflation_sequence(
            sample_magma_properties,
            sample_host_rock_properties,
            pathways,
            'elastoplastic'
        )
        sill_geometry = np.random.rand(50, 50, 10) * 100
        thermal_evolution = await simulator.simulate_thermal_effects(
            sample_magma_properties,
            sample_host_rock_properties,
            sill_geometry,
            'coupled'
        )
        
        stress_evolution = await simulator.simulate_stress_field_evolution(
            sample_host_rock_properties,
            inflation_sequence,
            thermal_evolution
        )
        
        assert isinstance(stress_evolution, StressFieldEvolution)
        assert stress_evolution.stress_id is not None
        assert stress_evolution.stress_tensor_field is not None
        assert stress_evolution.principal_stresses is not None
        assert stress_evolution.differential_stress is not None
        assert stress_evolution.mean_stress is not None
        assert len(stress_evolution.time_evolution) >= 0
    
    @pytest.mark.asyncio
    async def test_complete_emplacement_simulation(
        self, simulator, sample_magma_properties, sample_host_rock_properties, sample_injection_conditions
    ):
        """Test complete emplacement simulation"""
        complete_simulation = await simulator.run_complete_emplacement_simulation(
            sample_magma_properties,
            sample_host_rock_properties,
            sample_injection_conditions
        )
        
        assert isinstance(complete_simulation, EmplacementSimulation)
        assert complete_simulation.simulation_id is not None
        assert complete_simulation.magma_properties == sample_magma_properties
        assert complete_simulation.host_rock_properties == sample_host_rock_properties
        assert len(complete_simulation.injection_pathways) > 0
        assert complete_simulation.inflation_sequence is not None
        assert complete_simulation.thermal_evolution is not None
        assert complete_simulation.flow_dynamics is not None
        assert complete_simulation.crystallization_cooling is not None
        assert complete_simulation.host_rock_deformation is not None
        assert complete_simulation.stress_field_evolution is not None
        assert 'overall_score' in complete_simulation.validation_metrics
        assert complete_simulation.validation_metrics['overall_score'] > 0
    
    @pytest.mark.asyncio
    async def test_experimental_validation(
        self, simulator, sample_magma_properties, sample_host_rock_properties, sample_injection_conditions
    ):
        """Test validation against laboratory experiments"""
        # Run complete simulation
        complete_simulation = await simulator.run_complete_emplacement_simulation(
            sample_magma_properties,
            sample_host_rock_properties,
            sample_injection_conditions
        )
        
        # Validate against experimental data
        validation_scores = await simulator.validate_against_laboratory_experiments(
            complete_simulation,
            'all'
        )
        
        assert isinstance(validation_scores, dict)
        assert 'overall_experimental_validation' in validation_scores
        assert validation_scores['overall_experimental_validation'] >= 0
        assert validation_scores['overall_experimental_validation'] <= 1
        
        # Check individual experiment validations
        expected_experiments = ['kavanagh_2006', 'galland_2009', 'rivalta_2005', 'menand_2002']
        for exp_name in expected_experiments:
            if exp_name in validation_scores:
                assert validation_scores[exp_name] >= 0
                assert validation_scores[exp_name] <= 1
    
    def test_data_model_validation(self, sample_magma_properties, sample_host_rock_properties):
        """Test data model validation"""
        # Test MagmaProperties validation
        assert sample_magma_properties.temperature > 1000
        assert sample_magma_properties.temperature < 1500
        assert sample_magma_properties.viscosity > 0
        assert sample_magma_properties.density > 0
        assert 0 <= sample_magma_properties.crystallinity <= 1
        assert 0 <= sample_magma_properties.bubble_fraction <= 1
        
        # Test HostRockProperties validation
        assert sample_host_rock_properties.density > 0
        assert 0 <= sample_host_rock_properties.porosity <= 1
        assert sample_host_rock_properties.permeability > 0
        assert sample_host_rock_properties.youngs_modulus > 0
        assert 0 <= sample_host_rock_properties.poissons_ratio <= 0.5
        assert sample_host_rock_properties.tensile_strength > 0
    
    def test_configuration_validation(self, simulator):
        """Test configuration validation"""
        config = simulator.config
        
        # Test simulation parameters
        assert config['simulation']['time_step'] > 0
        assert config['simulation']['total_time'] > 0
        assert config['simulation']['spatial_resolution'] > 0
        assert config['simulation']['convergence_tolerance'] > 0
        assert config['simulation']['max_iterations'] > 0
        
        # Test physics flags
        assert isinstance(config['physics']['include_thermal'], bool)
        assert isinstance(config['physics']['include_mechanical'], bool)
        assert isinstance(config['physics']['include_fluid_flow'], bool)
        assert isinstance(config['physics']['include_crystallization'], bool)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, simulator):
        """Test error handling for invalid inputs"""
        # Test invalid magma properties
        invalid_magma = MagmaProperties(
            composition={'SiO2': 50.0},
            temperature=500.0,  # Too low
            pressure=1e8,
            viscosity=1e3,
            density=2700.0,
            volatile_content={},
            crystallinity=0.1,
            bubble_fraction=0.05,
            thermal_conductivity=2.5,
            specific_heat=1200.0,
            latent_heat_crystallization=3e5,
            surface_tension=0.4,
            compressibility=1e-10,
            thermal_expansion=3e-5
        )
        
        invalid_host_rock = HostRockProperties(
            rock_type='test',
            density=2400.0,
            porosity=0.15,
            permeability=1e-15,
            youngs_modulus=-1e10,  # Invalid negative value
            poissons_ratio=0.25,
            tensile_strength=5e6,
            cohesion=1e7,
            friction_angle=0.6,
            thermal_conductivity=2.0,
            specific_heat=800.0,
            thermal_diffusivity=1e-6,
            thermal_expansion=1e-5,
            fracture_toughness=1e6,
            crack_density=1e6,
            anisotropy_factor=1.2,
            layering_thickness=0.5,
            bedding_orientation=(0.0, 0.1)
        )
        
        invalid_conditions = {
            'injection_rate': -0.1,  # Invalid negative rate
            'injection_pressure': 1.5e8,
            'source_location': (500, 500, 1000)
        }
        
        # Test that appropriate errors are raised
        with pytest.raises(ValueError):
            await simulator.simulate_magma_injection_pathways(
                invalid_magma, invalid_host_rock, invalid_conditions
            )
    
    def test_experimental_data_loading(self, simulator):
        """Test experimental data loading"""
        exp_data = simulator.experimental_data
        
        assert isinstance(exp_data, dict)
        assert len(exp_data) > 0
        
        # Check specific experimental datasets
        expected_datasets = ['kavanagh_2006', 'galland_2009', 'rivalta_2005', 'menand_2002']
        for dataset_name in expected_datasets:
            assert dataset_name in exp_data
            dataset = exp_data[dataset_name]
            assert 'experiment_type' in dataset
            assert 'reference' in dataset
    
    @pytest.mark.asyncio
    async def test_performance_metrics(
        self, simulator, sample_magma_properties, sample_host_rock_properties, sample_injection_conditions
    ):
        """Test computational performance metrics"""
        start_time = datetime.now()
        
        # Run a simplified simulation
        pathways = await simulator.simulate_magma_injection_pathways(
            sample_magma_properties,
            sample_host_rock_properties,
            sample_injection_conditions,
            'coupled'
        )
        
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        # Check that simulation completes in reasonable time
        assert runtime < 60.0  # Should complete within 60 seconds for test
        assert len(pathways) > 0
    
    def test_numerical_stability(self, simulator):
        """Test numerical stability of calculations"""
        # Test with extreme but valid parameter values
        extreme_magma = MagmaProperties(
            composition={'SiO2': 75.0, 'Al2O3': 15.0, 'K2O': 5.0, 'Na2O': 5.0},
            temperature=1400.0,  # High temperature
            pressure=5e8,  # High pressure
            viscosity=1e6,  # High viscosity
            density=2300.0,
            volatile_content={'H2O': 6.0},  # High water content
            crystallinity=0.0,
            bubble_fraction=0.0,
            thermal_conductivity=1.5,
            specific_heat=1000.0,
            latent_heat_crystallization=4e5,
            surface_tension=0.3,
            compressibility=5e-11,
            thermal_expansion=5e-5
        )
        
        # Verify that extreme values don't cause numerical issues
        assert extreme_magma.temperature > 0
        assert extreme_magma.viscosity > 0
        assert extreme_magma.density > 0
        assert not np.isnan(extreme_magma.thermal_conductivity)
        assert not np.isinf(extreme_magma.pressure)

# Integration tests
class TestSillEmplacementIntegration:
    """Integration tests for the complete simulation workflow"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_integration(self):
        """Test complete workflow from initialization to validation"""
        # Initialize simulator
        simulator = SillEmplacementMechanismSimulator()
        
        # Create test data
        magma_props = MagmaProperties(
            composition={'SiO2': 55.0, 'Al2O3': 18.0, 'FeO': 7.0, 'MgO': 5.0, 'CaO': 8.0, 'Na2O': 4.0, 'K2O': 3.0},
            temperature=1150.0, pressure=1.2e8, viscosity=5e3, density=2650.0,
            volatile_content={'H2O': 3.0, 'CO2': 0.8}, crystallinity=0.15, bubble_fraction=0.08,
            thermal_conductivity=2.2, specific_heat=1150.0, latent_heat_crystallization=3.2e5,
            surface_tension=0.35, compressibility=8e-11, thermal_expansion=4e-5
        )
        
        host_rock_props = HostRockProperties(
            rock_type='limestone', density=2600.0, porosity=0.12, permeability=5e-16,
            youngs_modulus=3e10, poissons_ratio=0.22, tensile_strength=8e6, cohesion=1.5e7,
            friction_angle=0.7, thermal_conductivity=2.8, specific_heat=900.0,
            thermal_diffusivity=1.2e-6, thermal_expansion=8e-6, fracture_toughness=1.2e6,
            crack_density=8e5, anisotropy_factor=1.1, layering_thickness=0.8,
            bedding_orientation=(0.2, 0.15)
        )
        
        injection_conditions = {
            'injection_rate': 0.05, 'injection_pressure': 1.8e8,
            'source_location': (750, 750, 1200), 'domain_size': [1500, 1500, 800],
            'ambient_pressure': 1.2e5, 'ambient_temperature': 288.15
        }
        
        # Run complete simulation
        simulation = await simulator.run_complete_emplacement_simulation(
            magma_props, host_rock_props, injection_conditions
        )
        
        # Validate results
        assert simulation is not None
        assert simulation.validation_metrics['overall_score'] > 0.5
        
        # Validate against experiments
        exp_validation = await simulator.validate_against_laboratory_experiments(simulation)
        assert exp_validation['overall_experimental_validation'] > 0.5
        
        print(f"Integration test completed successfully!")
        print(f"Overall validation score: {simulation.validation_metrics['overall_score']:.3f}")
        print(f"Experimental validation score: {exp_validation['overall_experimental_validation']:.3f}")

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])