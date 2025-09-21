"""
Comprehensive test suite for the Geochemical Evolution Tracking System

This test suite validates all components of the geochemical evolution tracking
system including major element evolution, trace element modeling, isotopic
evolution, mineral crystallization, volatile degassing, thermodynamic
equilibrium calculations, temporal prediction, and validation against
natural compositions.
"""

import pytest
import numpy as np
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geochemical_evolution_tracking_system import (
    GeochemicalEvolutionTrackingSystem,
    MagmaComposition,
    ThermalProperties,
    RheologicalProperties,
    TraceElementPartitionCoefficients,
    IsotopicComposition,
    MineralPhase,
    VolatileSpecies,
    ThermodynamicState,
    GeochemicalEvolutionResult
)

class TestGeochemicalEvolutionTrackingSystem:
    """Test suite for GeochemicalEvolutionTrackingSystem"""
    
    @pytest.fixture
    async def system(self):
        """Create a test system instance"""
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'test_user',
            'password': 'test_password',
            'database': 'test_db'
        }
        
        system = GeochemicalEvolutionTrackingSystem(db_config)
        
        # Mock database connection
        system.db_pool = AsyncMock()
        
        yield system
        
        await system.close()
    
    @pytest.fixture
    def sample_magma_compositions(self):
        """Create sample magma compositions for testing"""
        mafic_composition = MagmaComposition(
            SiO2=47.2, TiO2=1.8, Al2O3=17.1, FeO=9.8, MnO=0.18,
            MgO=6.2, CaO=11.5, Na2O=3.1, K2O=2.8, P2O5=0.32,
            trace_elements={
                'Rb': 45, 'Sr': 680, 'Y': 22, 'Zr': 120, 'Nb': 8,
                'Ba': 450, 'La': 35, 'Ce': 68, 'Nd': 32, 'Sm': 6.8
            },
            isotope_ratios={
                '87Sr/86Sr': 0.70485, '143Nd/144Nd': 0.51268,
                '206Pb/204Pb': 18.92, '18O/16O': 0.002045
            },
            volatile_content={'H2O': 3.5, 'CO2': 0.8, 'SO2': 0.15, 'H2S': 0.05}
        )
        
        felsic_composition = MagmaComposition(
            SiO2=68.5, TiO2=0.4, Al2O3=15.8, FeO=3.2, MnO=0.08,
            MgO=1.2, CaO=3.8, Na2O=4.5, K2O=2.4, P2O5=0.12,
            trace_elements={
                'Rb': 120, 'Sr': 280, 'Y': 45, 'Zr': 280, 'Nb': 18,
                'Ba': 350, 'La': 52, 'Ce': 105, 'Nd': 48, 'Sm': 9.8
            },
            isotope_ratios={
                '87Sr/86Sr': 0.71025, '143Nd/144Nd': 0.51218,
                '206Pb/204Pb': 17.25, '18O/16O': 0.002055
            },
            volatile_content={'H2O': 5.2, 'CO2': 0.3, 'SO2': 0.08, 'H2S': 0.02}
        )
        
        return [mafic_composition, felsic_composition]
    
    @pytest.fixture
    def sample_mineral_assemblage(self):
        """Create sample mineral assemblage for testing"""
        olivine = MineralPhase(
            mineral_name='olivine',
            formula='(Mg,Fe)2SiO4',
            crystallization_temperature=1350,
            crystallization_pressure=2e8,
            modal_abundance=0.15,
            composition={'MgO': 45, 'FeO': 15, 'SiO2': 40},
            trace_element_content={'Ni': 2500, 'Co': 150, 'Cr': 300},
            nucleation_rate=1e6,
            growth_rate=1e-8
        )
        
        plagioclase = MineralPhase(
            mineral_name='plagioclase',
            formula='(Na,Ca)(Al,Si)4O8',
            crystallization_temperature=1200,
            crystallization_pressure=2e8,
            modal_abundance=0.25,
            composition={'SiO2': 55, 'Al2O3': 28, 'CaO': 12, 'Na2O': 5},
            trace_element_content={'Sr': 1200, 'Ba': 150, 'Eu': 2.5},
            nucleation_rate=5e5,
            growth_rate=5e-9
        )
        
        return [olivine, plagioclase]
    
    @pytest.fixture
    def mixing_parameters(self):
        """Create sample mixing parameters"""
        return {
            'temperature': 1150,  # °C
            'pressure': 2e8,      # Pa (200 MPa)
            'mixing_efficiency': 0.1,
            'convection_velocity': 1e-3,  # m/s
            'chamber_geometry': {'radius': 2000, 'height': 1000},
            'crystallinity': 0.15,
            'mixing_rate': 1e-6,  # s-1
            'chamber_volume': 1e9  # m3
        }
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, system):
        """Test system initialization"""
        assert system is not None
        assert system.mineral_database is not None
        assert system.partition_coefficient_database is not None
        assert system.isotope_database is not None
        assert system.volatile_database is not None
        assert system.natural_composition_database is not None
        
        # Test database loading
        assert 'olivine' in system.mineral_database
        assert 'plagioclase' in system.mineral_database
        assert 'olivine_melt' in system.partition_coefficient_database
        assert 'Sr' in system.isotope_database
        assert 'H2O' in system.volatile_database
        assert len(system.natural_composition_database) > 0
    
    @pytest.mark.asyncio
    async def test_major_element_evolution_tracking(
        self, system, sample_magma_compositions, mixing_parameters
    ):
        """Test major element evolution tracking"""
        chamber_id = "test_chamber_001"
        time_steps = np.linspace(0, 86400, 24)  # 24 hours with hourly steps
        
        # Mock database storage
        system._store_major_element_evolution = AsyncMock()
        
        evolution_data = await system.track_major_element_evolution(
            chamber_id, sample_magma_compositions, mixing_parameters, time_steps
        )
        
        # Verify results
        assert evolution_data is not None
        assert isinstance(evolution_data, dict)
        
        major_elements = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']
        for element in major_elements:
            assert element in evolution_data
            assert evolution_data[element].shape == (len(time_steps), len(sample_magma_compositions))
            
            # Check that evolution is reasonable (no negative values, bounded)
            assert np.all(evolution_data[element] >= 0)
            assert np.all(evolution_data[element] <= 100)  # Weight percent
        
        # Verify database storage was called
        system._store_major_element_evolution.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trace_element_evolution_tracking(
        self, system, sample_magma_compositions, sample_mineral_assemblage, mixing_parameters
    ):
        """Test trace element evolution tracking"""
        chamber_id = "test_chamber_002"
        time_steps = np.linspace(0, 86400, 24)
        
        # Mock database storage
        system._store_trace_element_evolution = AsyncMock()
        
        trace_evolution = await system.track_trace_element_evolution(
            chamber_id, sample_magma_compositions, sample_mineral_assemblage, 
            mixing_parameters, time_steps
        )
        
        # Verify results
        assert trace_evolution is not None
        assert isinstance(trace_evolution, dict)
        
        expected_traces = ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Nd', 'Sm']
        for element in expected_traces:
            if element in trace_evolution:
                assert trace_evolution[element].shape == (len(time_steps), len(sample_magma_compositions))
                assert np.all(trace_evolution[element] >= 0)  # No negative concentrations
        
        # Verify database storage was called
        system._store_trace_element_evolution.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_isotopic_evolution_tracking(
        self, system, sample_magma_compositions, sample_mineral_assemblage, mixing_parameters
    ):
        """Test isotopic evolution tracking"""
        chamber_id = "test_chamber_003"
        time_steps = np.linspace(0, 86400, 24)
        
        # Create initial isotopic compositions
        initial_isotopic = []
        for comp in sample_magma_compositions:
            isotopic_comp = {}
            for element, ratio_dict in comp.isotope_ratios.items():
                element_name = element.split('/')[0][2:]  # Extract element from ratio
                if element_name not in isotopic_comp:
                    isotopic_comp[element_name] = IsotopicComposition(
                        element=element_name,
                        isotope_ratios={element: ratio_dict},
                        delta_values={},
                        fractionation_factors={}
                    )
            initial_isotopic.append(isotopic_comp)
        
        # Mock database storage
        system._store_isotopic_evolution = AsyncMock()
        
        isotope_evolution = await system.track_isotopic_evolution(
            chamber_id, initial_isotopic, sample_mineral_assemblage, 
            mixing_parameters, time_steps
        )
        
        # Verify results
        assert isotope_evolution is not None
        assert isinstance(isotope_evolution, dict)
        
        expected_systems = ['Sr', 'Nd', 'Pb', 'O']
        for system_name in expected_systems:
            if system_name in isotope_evolution:
                assert isinstance(isotope_evolution[system_name], dict)
                for ratio in isotope_evolution[system_name]:
                    data = isotope_evolution[system_name][ratio]
                    assert data.shape == (len(time_steps), len(sample_magma_compositions))
                    assert np.all(data > 0)  # Isotope ratios should be positive
        
        # Verify database storage was called
        system._store_isotopic_evolution.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mineral_crystallization_sequence(
        self, system, sample_magma_compositions, mixing_parameters
    ):
        """Test mineral crystallization sequence tracking"""
        chamber_id = "test_chamber_004"
        time_steps = np.linspace(0, 86400, 24)
        
        # Create thermal and pressure evolution
        thermal_evolution = {
            'temperature': np.linspace(1200, 1000, len(time_steps))  # Cooling
        }
        pressure_evolution = {
            'pressure': np.full(len(time_steps), mixing_parameters['pressure'])
        }
        
        # Mock database storage
        system._store_crystallization_sequence = AsyncMock()
        
        crystallization_sequence = await system.track_mineral_crystallization_sequence(
            chamber_id, sample_magma_compositions[0], thermal_evolution, 
            pressure_evolution, time_steps
        )
        
        # Verify results
        assert crystallization_sequence is not None
        assert isinstance(crystallization_sequence, list)
        assert len(crystallization_sequence) == len(time_steps)
        
        # Check that crystallization sequence makes sense
        for t_idx, minerals in enumerate(crystallization_sequence):
            assert isinstance(minerals, list)
            for mineral in minerals:
                assert isinstance(mineral, MineralPhase)
                assert mineral.modal_abundance >= 0
                assert mineral.modal_abundance <= 1
                assert mineral.nucleation_rate >= 0
                assert mineral.growth_rate >= 0
        
        # Verify database storage was called
        system._store_crystallization_sequence.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_volatile_evolution_and_degassing(
        self, system, mixing_parameters
    ):
        """Test volatile evolution and degassing tracking"""
        chamber_id = "test_chamber_005"
        time_steps = np.linspace(0, 86400, 24)
        
        initial_volatile_content = {'H2O': 4.0, 'CO2': 0.5, 'SO2': 0.1, 'H2S': 0.03}
        
        # Create pressure and temperature evolution (decompression)
        pressure_evolution = {
            'pressure': np.linspace(2e8, 1e5, len(time_steps))  # Decompression
        }
        temperature_evolution = {
            'temperature': np.linspace(1150, 1100, len(time_steps))  # Slight cooling
        }
        
        # Mock database storage
        system._store_volatile_evolution = AsyncMock()
        
        volatile_evolution = await system.track_volatile_evolution_and_degassing(
            chamber_id, initial_volatile_content, pressure_evolution, 
            temperature_evolution, mixing_parameters, time_steps
        )
        
        # Verify results
        assert volatile_evolution is not None
        assert isinstance(volatile_evolution, dict)
        
        volatile_species = ['H2O', 'CO2', 'SO2', 'H2S']
        for species in volatile_species:
            if species in volatile_evolution:
                assert 'dissolved' in volatile_evolution[species]
                assert 'gas_phase' in volatile_evolution[species]
                assert 'saturation_pressure' in volatile_evolution[species]
                assert 'degassing_rate' in volatile_evolution[species]
                assert 'bubble_volume_fraction' in volatile_evolution[species]
                
                # Check array shapes
                for key in volatile_evolution[species]:
                    assert len(volatile_evolution[species][key]) == len(time_steps)
                    assert np.all(volatile_evolution[species][key] >= 0)
        
        # Verify database storage was called
        system._store_volatile_evolution.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_thermodynamic_equilibrium_calculation(
        self, system, sample_magma_compositions
    ):
        """Test thermodynamic equilibrium calculations"""
        chamber_id = "test_chamber_006"
        composition = sample_magma_compositions[0]
        temperature = 1150  # °C
        pressure = 2e8     # Pa
        volatile_content = {'H2O': 4.0, 'CO2': 0.5}
        
        # Mock database storage
        system._store_thermodynamic_state = AsyncMock()
        
        thermo_state = await system.calculate_thermodynamic_equilibrium(
            chamber_id, composition, temperature, pressure, volatile_content
        )
        
        # Verify results
        assert thermo_state is not None
        assert isinstance(thermo_state, ThermodynamicState)
        assert thermo_state.temperature == temperature
        assert thermo_state.pressure == pressure
        assert thermo_state.oxygen_fugacity > 0
        assert 0 <= thermo_state.water_activity <= 1
        assert 0 <= thermo_state.silica_activity <= 1
        assert thermo_state.heat_capacity > 0
        assert thermo_state.compressibility > 0
        assert thermo_state.thermal_expansion > 0
        
        # Verify database storage was called
        system._store_thermodynamic_state.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_temporal_geochemical_evolution_prediction(
        self, system, sample_magma_compositions
    ):
        """Test temporal geochemical evolution prediction"""
        chamber_id = "test_chamber_007"
        
        # Create a sample current state
        current_state = GeochemicalEvolutionResult(
            timestamp=datetime.now(),
            chamber_id=chamber_id,
            major_elements=sample_magma_compositions[0],
            trace_elements=sample_magma_compositions[0].trace_elements,
            isotopic_ratios={},
            mineral_assemblage=[],
            volatile_content={},
            thermodynamic_state=ThermodynamicState(
                temperature=1150, pressure=2e8, oxygen_fugacity=1e-10,
                water_activity=0.5, silica_activity=0.8, gibbs_free_energy=-1e6,
                enthalpy=-1e6, entropy=100, heat_capacity=80,
                compressibility=1e-11, thermal_expansion=5e-5
            ),
            mixing_degree=0.5,
            crystallinity=0.15,
            melt_fraction=0.85,
            uncertainty_bounds={},
            validation_metrics={}
        )
        
        prediction_horizon = 86400  # 1 day
        mixing_scenario = {
            'mixing_rate': 1e-6,
            'temperature_difference': 100,
            'composition_contrast': 0.3,
            'volatile_content_difference': 1.5
        }
        
        # Mock ML models
        system.major_element_predictor = Mock()
        system.major_element_predictor.predict.return_value = np.array([[50, 1, 16, 8, 0.1, 5, 10, 4, 2, 0.2]])
        
        # Mock database storage
        system._store_geochemical_predictions = AsyncMock()
        
        predictions = await system.predict_temporal_geochemical_evolution(
            chamber_id, current_state, prediction_horizon, mixing_scenario
        )
        
        # Verify results
        assert predictions is not None
        assert isinstance(predictions, dict)
        assert 'major_elements' in predictions
        assert 'uncertainty_bounds' in predictions
        assert 'confidence_intervals' in predictions
        
        # Verify database storage was called
        system._store_geochemical_predictions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validation_against_natural_compositions(
        self, system, sample_magma_compositions
    ):
        """Test validation against natural compositions"""
        chamber_id = "test_chamber_008"
        
        # Create a sample predicted composition
        predicted_composition = GeochemicalEvolutionResult(
            timestamp=datetime.now(),
            chamber_id=chamber_id,
            major_elements=sample_magma_compositions[0],
            trace_elements=sample_magma_compositions[0].trace_elements,
            isotopic_ratios={
                'Sr': IsotopicComposition(
                    element='Sr',
                    isotope_ratios={'87Sr/86Sr': 0.70485}
                )
            },
            mineral_assemblage=[],
            volatile_content={},
            thermodynamic_state=ThermodynamicState(
                temperature=1150, pressure=2e8, oxygen_fugacity=1e-10,
                water_activity=0.5, silica_activity=0.8, gibbs_free_energy=-1e6,
                enthalpy=-1e6, entropy=100, heat_capacity=80,
                compressibility=1e-11, thermal_expansion=5e-5
            ),
            mixing_degree=0.5,
            crystallinity=0.15,
            melt_fraction=0.85,
            uncertainty_bounds={},
            validation_metrics={}
        )
        
        mixing_parameters = {'temperature': 1150, 'pressure': 2e8}
        
        # Mock database storage
        system._store_validation_results = AsyncMock()
        
        validation_metrics = await system.validate_against_natural_compositions(
            chamber_id, predicted_composition, mixing_parameters
        )
        
        # Verify results
        assert validation_metrics is not None
        assert isinstance(validation_metrics, dict)
        assert 'major_element_rmse' in validation_metrics
        assert 'trace_element_rmse' in validation_metrics
        assert 'overall_similarity_score' in validation_metrics
        assert 'best_match_volcano' in validation_metrics
        assert 'best_match_score' in validation_metrics
        assert 'statistical_significance' in validation_metrics
        
        # Check that metrics are reasonable
        assert 0 <= validation_metrics['overall_similarity_score'] <= 1
        assert 0 <= validation_metrics['best_match_score'] <= 1
        assert 0 <= validation_metrics['statistical_significance'] <= 1
        
        # Verify database storage was called
        system._store_validation_results.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_partition_coefficient_calculation(self, system, sample_magma_compositions):
        """Test partition coefficient calculations"""
        mineral_name = 'olivine'
        temperature = 1200  # °C
        pressure = 1e8     # Pa
        composition = sample_magma_compositions[0]
        
        partition_coeffs = await system._calculate_partition_coefficients(
            mineral_name, temperature, pressure, composition
        )
        
        # Verify results
        assert partition_coeffs is not None
        assert isinstance(partition_coeffs, dict)
        
        # Check that expected elements are present
        expected_elements = ['Ni', 'Co', 'Cr', 'V', 'Sc', 'Mn', 'Zn']
        for element in expected_elements:
            if element in partition_coeffs:
                assert partition_coeffs[element] > 0
                assert partition_coeffs[element] < 100  # Reasonable range
    
    @pytest.mark.asyncio
    async def test_volatile_solubility_calculation(self, system):
        """Test volatile solubility calculations"""
        species = 'H2O'
        temperature = 1200  # °C
        pressure = 2e8     # Pa
        composition = {'H2O': 4.0, 'CO2': 0.5}
        
        solubility = await system._calculate_volatile_solubility(
            species, temperature, pressure, composition
        )
        
        # Verify results
        assert solubility is not None
        assert isinstance(solubility, dict)
        assert 'max_dissolved' in solubility
        assert 'saturation_pressure' in solubility
        assert solubility['max_dissolved'] >= 0
        assert solubility['saturation_pressure'] >= 0
    
    @pytest.mark.asyncio
    async def test_thermodynamic_calculations(self, system, sample_magma_compositions):
        """Test individual thermodynamic calculations"""
        composition = sample_magma_compositions[0]
        temperature = 1200  # °C
        pressure = 2e8     # Pa
        
        # Test FMQ buffer calculation
        log_fO2 = await system._calculate_fmq_buffer(temperature, pressure)
        assert isinstance(log_fO2, float)
        assert -20 < log_fO2 < 0  # Reasonable range for FMQ
        
        # Test water activity calculation
        water_activity = await system._calculate_water_activity(4.0, temperature, pressure, composition)
        assert isinstance(water_activity, float)
        assert 0 <= water_activity <= 1
        
        # Test silica activity calculation
        silica_activity = await system._calculate_silica_activity(composition, temperature, pressure)
        assert isinstance(silica_activity, float)
        assert 0 <= silica_activity <= 1
    
    def test_database_schema_validation(self):
        """Test that database schema is properly defined"""
        # This would typically involve checking the SQL schema file
        schema_file = os.path.join(os.path.dirname(__file__), 'database_schema.sql')
        assert os.path.exists(schema_file)
        
        with open(schema_file, 'r') as f:
            schema_content = f.read()
        
        # Check for required tables
        required_tables = [
            'geochemical_predictions',
            'geochemical_validation_results',
            'thermodynamic_equilibrium',
            'partition_coefficients',
            'isotopic_fractionation_factors',
            'volatile_solubility_data',
            'natural_magma_compositions',
            'ml_model_performance',
            'geochemical_evolution_timeseries',
            'mineral_crystallization_kinetics',
            'volatile_degassing_kinetics'
        ]
        
        for table in required_tables:
            assert f"CREATE TABLE IF NOT EXISTS {table}" in schema_content
    
    @pytest.mark.asyncio
    async def test_error_handling(self, system):
        """Test error handling in various scenarios"""
        chamber_id = "test_chamber_error"
        
        # Test with invalid inputs
        with pytest.raises(Exception):
            await system.track_major_element_evolution(
                chamber_id, [], {}, np.array([])  # Empty inputs
            )
        
        # Test with None inputs
        with pytest.raises(Exception):
            await system.calculate_thermodynamic_equilibrium(
                chamber_id, None, 1200, 2e8, {}
            )
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, system, sample_magma_compositions, mixing_parameters):
        """Test performance benchmarks for key operations"""
        import time
        
        chamber_id = "test_chamber_performance"
        time_steps = np.linspace(0, 86400, 100)  # 100 time steps
        
        # Mock database operations to focus on computation time
        system._store_major_element_evolution = AsyncMock()
        
        # Benchmark major element evolution
        start_time = time.time()
        await system.track_major_element_evolution(
            chamber_id, sample_magma_compositions, mixing_parameters, time_steps
        )
        major_element_time = time.time() - start_time
        
        # Performance should be reasonable (< 10 seconds for 100 time steps)
        assert major_element_time < 10.0
        
        print(f"Major element evolution time: {major_element_time:.2f} seconds")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])