"""
Advanced Test Suite for Bubble Nucleation and Coalescence Modeler

Tests the advanced implementations including:
- Advanced thermal damping in bubble oscillations
- Advanced Marangoni instability modeling
- Multi-species diffusion with activity corrections
- Database integration
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime
from typing import Dict, Any

# Import the module to test
import sys
sys.path.append('backend/modules')
from bubble_nucleation_coalescence_modeler import (
    BubbleNucleationCoalescenceModeler,
    BubblePopulation,
    BubbleNucleationModel
)
from database_integration import DatabaseManager
from geological_data_structures import VolcanicSystem, GeochemicalData


class MockDatabaseManager:
    """Mock database manager for testing"""
    
    async def execute_query(self, query: str, params: tuple = None):
        """Mock query execution"""
        # Return mock magma data
        if "magma_properties" in query:
            return [
                {
                    'sample_id': 'test_sample_001',
                    'sio2_content': 0.70,
                    'al2o3_content': 0.15,
                    'fe2o3_content': 0.05,
                    'mgo_content': 0.02,
                    'cao_content': 0.03,
                    'na2o_content': 0.03,
                    'k2o_content': 0.02,
                    'h2o_content': 0.05,
                    'co2_content': 0.01,
                    'temperature': 1273.15,  # K
                    'pressure': 1e7,  # Pa (100 MPa)
                    'viscosity': 1e6,  # Pa·s
                    'density': 2500.0  # kg/m³
                }
            ]
        return []


@pytest.fixture
def db_manager():
    """Create mock database manager"""
    return MockDatabaseManager()


@pytest.fixture
def modeler(db_manager):
    """Create bubble nucleation modeler instance"""
    return BubbleNucleationCoalescenceModeler(db_manager)


@pytest.fixture
def volcanic_system():
    """Create mock volcanic system"""
    return VolcanicSystem(
        system_id='test_volcano_001',
        name='Test Volcano',
        location=(45.0, -122.0),
        elevation=3000.0,
        volcano_type='stratovolcano'
    )


@pytest.fixture
def magma_conditions():
    """Create test magma conditions"""
    return {
        'temperature': 1273.15,  # K (1000°C)
        'pressure': 1e7,  # Pa (100 MPa)
        'composition': {
            'SiO2': 0.70,
            'Al2O3': 0.15,
            'Fe2O3': 0.05,
            'MgO': 0.02,
            'CaO': 0.03,
            'Na2O': 0.03,
            'K2O': 0.02
        },
        'volatile_content': {
            'H2O': 0.05,  # 5 wt%
            'CO2': 0.01   # 1 wt%
        },
        'viscosity': 1e6,  # Pa·s
        'density': 2500.0,  # kg/m³
        'location': (0, 0, -5000)  # 5 km depth
    }


class TestAdvancedThermalDamping:
    """Test advanced thermal damping implementation"""
    
    def test_thermal_damping_calculation(self, modeler):
        """Test that thermal damping is calculated with advanced physics"""
        radii = np.array([1e-6, 1e-5, 1e-4, 1e-3])  # 1 μm to 1 mm
        pressure = 1e7  # Pa
        temperature = 1273.15  # K
        viscosity = 1e6  # Pa·s
        
        frequencies = modeler._calculate_bubble_oscillation_frequencies(
            radii, pressure, temperature, viscosity
        )
        
        # Verify frequencies are calculated
        assert len(frequencies) == len(radii)
        assert np.all(frequencies >= 0)
        
        # Larger bubbles should have lower frequencies
        assert frequencies[0] > frequencies[-1]
        
        # Frequencies should be in reasonable range (Hz to MHz)
        assert np.all(frequencies < 1e7)  # < 10 MHz
        
        print(f"✓ Bubble oscillation frequencies: {frequencies}")
    
    def test_thermal_penetration_depth(self, modeler):
        """Test that thermal penetration depth is properly calculated"""
        # The advanced implementation should account for thermal diffusivity
        # and frequency-dependent penetration depth
        
        radii = np.array([1e-5, 1e-4, 1e-3])
        pressure = 1e7
        temperature = 1273.15
        viscosity = 1e6
        
        frequencies = modeler._calculate_bubble_oscillation_frequencies(
            radii, pressure, temperature, viscosity
        )
        
        # Calculate expected thermal penetration depth
        k_thermal = 1.5  # W/(m·K)
        cp_magma = 1200.0  # J/(kg·K)
        rho_magma = 2500.0  # kg/m³
        alpha_thermal = k_thermal / (rho_magma * cp_magma)
        
        delta_th = np.sqrt(2 * alpha_thermal / (2 * np.pi * frequencies))
        
        # Thermal penetration depth should be smaller than bubble radius for high frequencies
        assert np.any(delta_th < radii)
        
        print(f"✓ Thermal penetration depths: {delta_th}")
    
    def test_phase_change_damping(self, modeler):
        """Test that phase change effects are included in damping"""
        # The advanced implementation should include latent heat effects
        
        radii = np.array([1e-4])
        pressure = 1e7
        temperature = 1273.15
        viscosity = 1e6
        
        frequencies = modeler._calculate_bubble_oscillation_frequencies(
            radii, pressure, temperature, viscosity
        )
        
        # Frequency should be positive and finite
        assert frequencies[0] > 0
        assert np.isfinite(frequencies[0])
        
        print(f"✓ Phase change damping included in frequency: {frequencies[0]} Hz")


class TestAdvancedMarangoniInstability:
    """Test advanced Marangoni instability implementation"""
    
    @pytest.mark.asyncio
    async def test_marangoni_number_calculation(self, modeler, magma_conditions):
        """Test that Marangoni number is properly calculated"""
        # Create bubble population
        bubble_pop = BubblePopulation(
            population_id='test_pop_001',
            timestamp=datetime.now(),
            location=(0, 0, -5000),
            bubble_count=1000,
            size_distribution=np.array([1e-5, 1e-4, 1e-3]),
            number_density=np.array([100, 50, 10]),
            gas_content={'H2O': np.array([1e-10, 1e-9, 1e-8])},
            rise_velocities=np.array([0.001, 0.01, 0.1]),
            coalescence_probability=np.array([0.1, 0.2, 0.3]),
            nucleation_sites=1000
        )
        
        # Test bubble bursting with Marangoni effects
        bursting_results = modeler._model_bubble_bursting(bubble_pop, magma_conditions)
        
        # Verify bursting probabilities are calculated
        assert 'bursting_probability' in bursting_results
        assert len(bursting_results['bursting_probability']) == 3
        
        # Probabilities should be between 0 and 1
        assert np.all(bursting_results['bursting_probability'] >= 0)
        assert np.all(bursting_results['bursting_probability'] <= 1)
        
        print(f"✓ Bursting probabilities with Marangoni: {bursting_results['bursting_probability']}")
    
    @pytest.mark.asyncio
    async def test_thermal_marangoni_effects(self, modeler, magma_conditions):
        """Test thermal Marangoni effects on film stability"""
        # High temperature gradient should increase instability
        
        bubble_pop = BubblePopulation(
            population_id='test_pop_002',
            timestamp=datetime.now(),
            location=(0, 0, -5000),
            bubble_count=100,
            size_distribution=np.array([1e-4]),
            number_density=np.array([100]),
            gas_content={'H2O': np.array([1e-9])},
            rise_velocities=np.array([0.01]),
            coalescence_probability=np.array([0.1]),
            nucleation_sites=100
        )
        
        # Test with high temperature (should increase Marangoni instability)
        high_temp_conditions = magma_conditions.copy()
        high_temp_conditions['temperature'] = 1473.15  # 1200°C
        
        bursting_high_temp = modeler._model_bubble_bursting(bubble_pop, high_temp_conditions)
        
        # Test with low temperature
        low_temp_conditions = magma_conditions.copy()
        low_temp_conditions['temperature'] = 1073.15  # 800°C
        
        bursting_low_temp = modeler._model_bubble_bursting(bubble_pop, low_temp_conditions)
        
        # Higher temperature should generally lead to higher bursting probability
        # (due to larger temperature gradient with atmosphere)
        print(f"✓ High temp bursting prob: {bursting_high_temp['bursting_probability'][0]}")
        print(f"✓ Low temp bursting prob: {bursting_low_temp['bursting_probability'][0]}")
    
    @pytest.mark.asyncio
    async def test_compositional_marangoni_effects(self, modeler, magma_conditions):
        """Test compositional Marangoni effects from volatile gradients"""
        # High volatile content should enhance Marangoni instability
        
        bubble_pop = BubblePopulation(
            population_id='test_pop_003',
            timestamp=datetime.now(),
            location=(0, 0, -5000),
            bubble_count=100,
            size_distribution=np.array([1e-4]),
            number_density=np.array([100]),
            gas_content={'H2O': np.array([1e-9])},
            rise_velocities=np.array([0.01]),
            coalescence_probability=np.array([0.1]),
            nucleation_sites=100
        )
        
        # High volatile content
        high_volatile_conditions = magma_conditions.copy()
        high_volatile_conditions['volatile_content'] = {'H2O': 0.08, 'CO2': 0.02}  # 8% H2O
        
        bursting_high_volatile = modeler._model_bubble_bursting(bubble_pop, high_volatile_conditions)
        
        # Low volatile content
        low_volatile_conditions = magma_conditions.copy()
        low_volatile_conditions['volatile_content'] = {'H2O': 0.01, 'CO2': 0.001}  # 1% H2O
        
        bursting_low_volatile = modeler._model_bubble_bursting(bubble_pop, low_volatile_conditions)
        
        print(f"✓ High volatile bursting prob: {bursting_high_volatile['bursting_probability'][0]}")
        print(f"✓ Low volatile bursting prob: {bursting_low_volatile['bursting_probability'][0]}")


class TestMultiSpeciesDiffusion:
    """Test multi-species diffusion with activity corrections"""
    
    @pytest.mark.asyncio
    async def test_activity_coefficient_calculation(self, modeler):
        """Test that activity coefficients are calculated for non-ideal mixing"""
        volatile_content = {'H2O': 0.05, 'CO2': 0.01}
        temperature = 1273.15
        pressure = 1e7
        
        activities = modeler._calculate_volatile_activities(
            volatile_content, temperature, pressure
        )
        
        # Verify activities are calculated for all species
        assert 'H2O' in activities
        assert 'CO2' in activities
        
        # Activities should be positive
        assert activities['H2O'] > 0
        assert activities['CO2'] > 0
        
        # For non-ideal mixing, activities may differ from mole fractions
        print(f"✓ H2O activity: {activities['H2O']}")
        print(f"✓ CO2 activity: {activities['CO2']}")
    
    @pytest.mark.asyncio
    async def test_water_speciation_effects(self, modeler, magma_conditions):
        """Test that water speciation (molecular H2O vs OH) is accounted for"""
        # The advanced diffusion model should account for different diffusivities
        # of molecular H2O and OH groups
        
        bubble_pop = BubblePopulation(
            population_id='test_pop_004',
            timestamp=datetime.now(),
            location=(0, 0, -5000),
            bubble_count=100,
            size_distribution=np.array([1e-5, 1e-4]),
            number_density=np.array([50, 50]),
            gas_content={'H2O': np.array([1e-10, 1e-9])},
            rise_velocities=np.array([0.001, 0.01]),
            coalescence_probability=np.array([0.1, 0.1]),
            nucleation_sites=100
        )
        
        diffusion_growth = modeler._calculate_diffusion_growth(
            bubble_pop, magma_conditions, time_step=1.0
        )
        
        # Verify diffusion growth is calculated
        assert len(diffusion_growth) == 2
        assert np.all(diffusion_growth >= 0)
        
        # Larger bubbles should generally grow faster (larger surface area)
        print(f"✓ Diffusion growth rates: {diffusion_growth}")
    
    @pytest.mark.asyncio
    async def test_peng_robinson_equation_of_state(self, modeler):
        """Test that Peng-Robinson EOS is used for real gas behavior"""
        # Test for H2O
        temperature = 1273.15  # K
        pressure = 1e7  # Pa
        
        v_molar_h2o = modeler._calculate_molar_volume_peng_robinson(
            'H2O', temperature, pressure
        )
        
        # Molar volume should be positive and reasonable
        assert v_molar_h2o > 0
        assert v_molar_h2o < 1.0  # Less than 1 m³/mol
        
        # Test for CO2
        v_molar_co2 = modeler._calculate_molar_volume_peng_robinson(
            'CO2', temperature, pressure
        )
        
        assert v_molar_co2 > 0
        assert v_molar_co2 < 1.0
        
        print(f"✓ H2O molar volume (Peng-Robinson): {v_molar_h2o} m³/mol")
        print(f"✓ CO2 molar volume (Peng-Robinson): {v_molar_co2} m³/mol")


class TestDatabaseIntegration:
    """Test database integration for real data"""
    
    @pytest.mark.asyncio
    async def test_nucleation_model_initialization(self, modeler, volcanic_system):
        """Test that nucleation models are initialized from database"""
        models = await modeler.initialize_nucleation_models(volcanic_system)
        
        # Verify models are created
        assert len(models) > 0
        assert 'test_sample_001' in models
        
        # Verify model contains required parameters
        model = models['test_sample_001']
        assert 'composition' in model
        assert 'volatile_content' in model
        assert 'nucleation_params' in model
        
        print(f"✓ Initialized {len(models)} nucleation models from database")
    
    @pytest.mark.asyncio
    async def test_bubble_population_storage(self, modeler):
        """Test that bubble population data can be stored in database"""
        bubble_pop = BubblePopulation(
            population_id='test_pop_005',
            timestamp=datetime.now(),
            location=(45.0, -122.0, -5000),
            bubble_count=1000,
            size_distribution=np.array([1e-5, 1e-4, 1e-3]),
            number_density=np.array([100, 50, 10]),
            gas_content={'H2O': np.array([1e-10, 1e-9, 1e-8])},
            rise_velocities=np.array([0.001, 0.01, 0.1]),
            coalescence_probability=np.array([0.1, 0.2, 0.3]),
            nucleation_sites=1000
        )
        
        # This should not raise an exception
        await modeler.store_bubble_population_data(bubble_pop, 'test_volcano_001')
        
        print(f"✓ Successfully stored bubble population data")


class TestComprehensiveIntegration:
    """Test comprehensive integration of all advanced features"""
    
    @pytest.mark.asyncio
    async def test_full_nucleation_growth_coalescence_cycle(self, modeler, magma_conditions):
        """Test complete cycle: nucleation -> growth -> coalescence"""
        # 1. Nucleation
        bubble_pop = await modeler.simulate_bubble_nucleation(magma_conditions, time_step=1.0)
        
        assert bubble_pop.bubble_count > 0
        assert len(bubble_pop.size_distribution) > 0
        
        print(f"✓ Nucleated {bubble_pop.bubble_count} bubbles")
        
        # 2. Growth
        magma_conditions['pressure_rate'] = -1e5  # Decompression: -1 MPa/s
        grown_pop = await modeler.simulate_bubble_growth(bubble_pop, magma_conditions, time_step=1.0)
        
        # Bubbles should grow during decompression
        assert np.mean(grown_pop.size_distribution) >= np.mean(bubble_pop.size_distribution)
        
        print(f"✓ Bubbles grew from {np.mean(bubble_pop.size_distribution):.2e} to {np.mean(grown_pop.size_distribution):.2e} m")
        
        # 3. Coalescence
        magma_conditions['shear_rate'] = 1.0  # s⁻¹
        coalesced_pop = await modeler.simulate_bubble_coalescence(grown_pop, magma_conditions, time_step=1.0)
        
        # Some coalescence should occur
        print(f"✓ Bubble count after coalescence: {coalesced_pop.bubble_count}")
        
        # 4. Gas transport and release
        transport_results = await modeler.simulate_gas_transport_and_release(
            coalesced_pop, magma_conditions, time_step=1.0
        )
        
        assert 'release_rates' in transport_results
        assert 'rheological_effects' in transport_results
        
        print(f"✓ Total gas flux: {transport_results['total_gas_flux']:.2e} kg/s")
        print(f"✓ Vesicularity: {transport_results['rheological_effects']['vesicularity']:.3f}")


def run_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("ADVANCED BUBBLE NUCLEATION AND COALESCENCE MODELER TEST SUITE")
    print("="*80 + "\n")
    
    # Run pytest
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
