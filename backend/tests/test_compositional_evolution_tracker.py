"""
Comprehensive test suite for Compositional Evolution Tracking System

Tests cover:
- Advanced geochemical modeling for magma mixing
- Trace element and isotopic evolution
- Mineral crystallization and dissolution modeling
- Volatile saturation and degassing calculations
- Thermodynamic equilibrium calculations
- Compositional gradient modeling
- Temporal evolution tracking and prediction
- Validation against natural volcanic rock compositions
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

# Import the modules we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from compositional_evolution_tracker import (
    MagmaComposition, MineralPhase, PartitionCoefficient,
    CompositionEvolutionDatabase, ThermodynamicCalculator,
    MagmaMixingCalculator, CrystallizationModeler,
    CompositionGradientModeler, CompositionEvolutionTracker
)

class TestMagmaComposition:
    """Test MagmaComposition data structure and methods"""
    
    def test_magma_composition_creation(self):
        """Test creation of MagmaComposition with all parameters"""
        composition = MagmaComposition(
            SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
            MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
            H2O=2.0, CO2=0.1, SO2=0.05, Cl=0.02, F=0.01,
            temperature=1200.0, pressure=0.2,
            trace_elements={'Ni': 150, 'Cr': 300, 'Sr': 400},
            isotopic_ratios={'Sr87_Sr86': 0.7035, 'Nd143_Nd144': 0.5131}
        )
        
        assert composition.SiO2 == 50.0
        assert composition.H2O == 2.0
        assert composition.trace_elements['Ni'] == 150
        assert composition.isotopic_ratios['Sr87_Sr86'] == 0.7035
    
    def test_normalize_major_elements(self):
        """Test major element normalization to 100%"""
        composition = MagmaComposition(
            SiO2=45.0, TiO2=1.35, Al2O3=13.5, FeO=9.0, MnO=0.18,
            MgO=7.2, CaO=9.9, Na2O=2.25, K2O=0.45, P2O5=0.27,
            temperature=1200.0, pressure=0.2
        )
        
        normalized = composition.normalize_major_elements()
        
        total = (normalized.SiO2 + normalized.TiO2 + normalized.Al2O3 + 
                normalized.FeO + normalized.MnO + normalized.MgO + 
                normalized.CaO + normalized.Na2O + normalized.K2O + normalized.P2O5)
        
        assert abs(total - 100.0) < 0.001

class TestMineralPhase:
    """Test MineralPhase thermodynamic calculations"""
    
    def test_mineral_phase_creation(self):
        """Test creation of MineralPhase with thermodynamic properties"""
        olivine = MineralPhase(
            name='olivine',
            formula='(Mg,Fe)2SiO4',
            composition={'Mg': 0.4, 'Fe': 0.1, 'Si': 0.2, 'O': 0.3},
            molar_volume=43.6,
            enthalpy_formation=-2174.0,
            entropy=95.1,
            heat_capacity_coefficients=[118.1, 0.0133, -2.26e6, -0.0026],
            stability_field={'temperature': (1000, 2000), 'pressure': (0.001, 15)}
        )
        
        assert olivine.name == 'olivine'
        assert olivine.molar_volume == 43.6
        assert len(olivine.heat_capacity_coefficients) == 4
    
    def test_gibbs_energy_calculation(self):
        """Test Gibbs free energy calculation at different P-T conditions"""
        olivine = MineralPhase(
            name='olivine',
            formula='(Mg,Fe)2SiO4',
            composition={'Mg': 0.4, 'Fe': 0.1, 'Si': 0.2, 'O': 0.3},
            molar_volume=43.6,
            enthalpy_formation=-2174.0,
            entropy=95.1,
            heat_capacity_coefficients=[118.1, 0.0133, -2.26e6, -0.0026],
            stability_field={'temperature': (1000, 2000), 'pressure': (0.001, 15)}
        )
        
        # Test at different conditions
        gibbs_1200_01 = olivine.calculate_gibbs_energy(1200, 0.1)
        gibbs_1400_05 = olivine.calculate_gibbs_energy(1400, 0.5)
        
        # Higher temperature should generally increase Gibbs energy
        # Higher pressure should increase Gibbs energy
        assert isinstance(gibbs_1200_01, float)
        assert isinstance(gibbs_1400_05, float)

class TestPartitionCoefficient:
    """Test partition coefficient calculations"""
    
    def test_partition_coefficient_creation(self):
        """Test creation of PartitionCoefficient"""
        kd_ni_ol = PartitionCoefficient(
            element='Ni',
            mineral_phase='olivine',
            melt_phase='basaltic_melt',
            coefficient=14.0,
            temperature_dependence=-3000,
            composition_dependence={'MgO': 0.5}
        )
        
        assert kd_ni_ol.element == 'Ni'
        assert kd_ni_ol.coefficient == 14.0
        assert kd_ni_ol.temperature_dependence == -3000
    
    def test_kd_calculation(self):
        """Test partition coefficient calculation with P-T-X dependence"""
        kd_ni_ol = PartitionCoefficient(
            element='Ni',
            mineral_phase='olivine',
            melt_phase='basaltic_melt',
            coefficient=14.0,
            temperature_dependence=-3000,
            composition_dependence={'MgO': 0.5}
        )
        
        composition = MagmaComposition(
            SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
            MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
            temperature=1200.0, pressure=0.2
        )
        
        kd_calculated = kd_ni_ol.calculate_kd(1200, 0.2, composition)
        
        assert isinstance(kd_calculated, float)
        assert kd_calculated > 0

class TestThermodynamicCalculator:
    """Test thermodynamic calculations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.thermo_calc = ThermodynamicCalculator()
    
    def test_mineral_stability_calculation(self):
        """Test mineral stability calculations"""
        composition = MagmaComposition(
            SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
            MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
            temperature=1200.0, pressure=0.2
        )
        
        stability = self.thermo_calc.calculate_mineral_stability(composition, 1200, 0.2)
        
        assert isinstance(stability, dict)
        assert len(stability) > 0
        
        # Check that stabilities sum to approximately 1
        total_stability = sum(stability.values())
        assert abs(total_stability - 1.0) < 0.001
    
    def test_volatile_saturation_calculation(self):
        """Test volatile saturation calculations"""
        composition = MagmaComposition(
            SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
            MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
            temperature=1200.0, pressure=0.2
        )
        
        saturation = self.thermo_calc.calculate_volatile_saturation(composition, 1200, 0.2)
        
        assert isinstance(saturation, dict)
        assert 'H2O' in saturation
        assert 'CO2' in saturation
        assert 'SO2' in saturation
        
        # All saturation values should be positive
        for volatile, sat_value in saturation.items():
            assert sat_value >= 0
    
    def test_degassing_rate_calculation(self):
        """Test volatile degassing rate calculations"""
        composition = MagmaComposition(
            SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
            MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
            H2O=3.0, CO2=0.5, SO2=0.1,  # Oversaturated
            temperature=1200.0, pressure=0.2
        )
        
        saturation = {'H2O': 2.0, 'CO2': 0.3, 'SO2': 0.05, 'Cl': 0.02, 'F': 0.01}
        degassing_rates = self.thermo_calc.calculate_degassing_rate(composition, saturation, 0.1)
        
        assert isinstance(degassing_rates, dict)
        
        # Should have positive degassing rates for oversaturated volatiles
        assert degassing_rates['H2O'] > 0  # 3.0 > 2.0
        assert degassing_rates['CO2'] > 0  # 0.5 > 0.3
        assert degassing_rates['SO2'] > 0  # 0.1 > 0.05

class TestMagmaMixingCalculator:
    """Test magma mixing calculations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.thermo_calc = ThermodynamicCalculator()
        self.mixing_calc = MagmaMixingCalculator(self.thermo_calc)
    
    def test_binary_mixing(self):
        """Test binary magma mixing calculations"""
        # Create two different compositions
        basalt = MagmaComposition(
            SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
            MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
            temperature=1200.0, pressure=0.2,
            trace_elements={'Ni': 150, 'Cr': 300},
            isotopic_ratios={'Sr87_Sr86': 0.7035}
        )
        
        dacite = MagmaComposition(
            SiO2=68.0, TiO2=0.8, Al2O3=16.0, FeO=4.0, MnO=0.1,
            MgO=2.0, CaO=4.0, Na2O=4.0, K2O=3.0, P2O5=0.1,
            temperature=900.0, pressure=0.2,
            trace_elements={'Ni': 20, 'Cr': 50},
            isotopic_ratios={'Sr87_Sr86': 0.7080}
        )
        
        # Test 50-50 mixing
        mixed = self.mixing_calc.calculate_binary_mixing(basalt, dacite, 0.5)
        
        # Check that mixed composition is intermediate
        assert basalt.SiO2 < mixed.SiO2 < dacite.SiO2
        assert dacite.MgO < mixed.MgO < basalt.MgO
        assert abs(mixed.SiO2 - 59.0) < 1.0  # Should be approximately average
        
        # Check trace elements
        assert mixed.trace_elements['Ni'] == 85.0  # (150 + 20) / 2
        
        # Check isotopes
        assert abs(mixed.isotopic_ratios['Sr87_Sr86'] - 0.70575) < 0.0001
    
    def test_mixing_entropy_calculation(self):
        """Test mixing entropy calculations"""
        compositions = [
            MagmaComposition(SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
                           MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
                           temperature=1200.0, pressure=0.2),
            MagmaComposition(SiO2=68.0, TiO2=0.8, Al2O3=16.0, FeO=4.0, MnO=0.1,
                           MgO=2.0, CaO=4.0, Na2O=4.0, K2O=3.0, P2O5=0.1,
                           temperature=900.0, pressure=0.2)
        ]
        
        proportions = [0.5, 0.5]
        
        mixing_entropy = self.mixing_calc.calculate_mixing_entropy(compositions, proportions)
        
        assert isinstance(mixing_entropy, float)
        assert mixing_entropy > 0  # Mixing should increase entropy
    
    def test_thermal_equilibration(self):
        """Test thermal equilibration during mixing"""
        compositions = [
            MagmaComposition(SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
                           MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
                           temperature=1200.0, pressure=0.2),
            MagmaComposition(SiO2=68.0, TiO2=0.8, Al2O3=16.0, FeO=4.0, MnO=0.1,
                           MgO=2.0, CaO=4.0, Na2O=4.0, K2O=3.0, P2O5=0.1,
                           temperature=800.0, pressure=0.2)
        ]
        
        proportions = [0.5, 0.5]
        heat_capacities = [1200.0, 1200.0]  # J/kg/K
        
        eq_temp = self.mixing_calc.calculate_thermal_equilibration(compositions, proportions, heat_capacities)
        
        assert isinstance(eq_temp, float)
        assert 800 < eq_temp < 1200  # Should be between the two temperatures

class TestCrystallizationModeler:
    """Test crystallization and dissolution modeling"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.thermo_calc = ThermodynamicCalculator()
        self.cryst_modeler = CrystallizationModeler(self.thermo_calc)
    
    def test_liquidus_temperature_calculation(self):
        """Test liquidus temperature calculations"""
        composition = MagmaComposition(
            SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
            MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
            temperature=1200.0, pressure=0.2
        )
        
        liquidus_temps = self.cryst_modeler.calculate_liquidus_temperature(composition, 0.2)
        
        assert isinstance(liquidus_temps, dict)
        assert 'olivine' in liquidus_temps
        assert 'plagioclase' in liquidus_temps
        
        # Liquidus temperatures should be reasonable
        for mineral, temp in liquidus_temps.items():
            assert 1000 < temp < 2000
    
    def test_nucleation_rate_calculation(self):
        """Test nucleation rate calculations"""
        nucleation_rate = self.cryst_modeler.calculate_nucleation_rate('olivine', 1150, 100)
        
        assert isinstance(nucleation_rate, float)
        assert nucleation_rate >= 0
        
        # Higher undercooling should increase nucleation rate
        higher_undercooling_rate = self.cryst_modeler.calculate_nucleation_rate('olivine', 1100, 150)
        assert higher_undercooling_rate > nucleation_rate
    
    def test_crystal_growth_rate_calculation(self):
        """Test crystal growth rate calculations"""
        composition = MagmaComposition(
            SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
            MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
            temperature=1150.0, pressure=0.2
        )
        
        growth_rate = self.cryst_modeler.calculate_crystal_growth_rate('olivine', composition, 1150, 100)
        
        assert isinstance(growth_rate, float)
        assert growth_rate >= 0

class TestCompositionGradientModeler:
    """Test compositional gradient modeling"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.gradient_modeler = CompositionGradientModeler()
    
    def test_diffusion_coefficient_calculation(self):
        """Test diffusion coefficient calculations"""
        D_sio2 = self.gradient_modeler.calculate_diffusion_coefficient('SiO2', 'basaltic_melt', 1200)
        D_h2o = self.gradient_modeler.calculate_diffusion_coefficient('H2O', 'basaltic_melt', 1200)
        
        assert isinstance(D_sio2, float)
        assert isinstance(D_h2o, float)
        assert D_sio2 > 0
        assert D_h2o > 0
        
        # H2O should diffuse faster than SiO2
        assert D_h2o > D_sio2
    
    def test_diffusion_1d_solver(self):
        """Test 1D diffusion equation solver"""
        # Create initial concentration profile
        initial_profile = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        
        # Solve diffusion for one time step
        new_profile = self.gradient_modeler.solve_diffusion_1d(
            initial_profile, 1e-12, 3600, 0.1, 'closed'
        )
        
        assert len(new_profile) == len(initial_profile)
        assert isinstance(new_profile, np.ndarray)
        
        # Profile should be smoothed (less steep gradients)
        initial_gradient = np.abs(np.diff(initial_profile)).max()
        new_gradient = np.abs(np.diff(new_profile)).max()
        assert new_gradient <= initial_gradient
    
    def test_chamber_stratification_modeling(self):
        """Test chamber stratification modeling"""
        compositions = [
            MagmaComposition(SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
                           MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
                           temperature=1200.0, pressure=0.2),
            MagmaComposition(SiO2=55.0, TiO2=1.2, Al2O3=16.0, FeO=8.0, MnO=0.15,
                           MgO=6.0, CaO=9.0, Na2O=3.0, K2O=1.0, P2O5=0.25,
                           temperature=1100.0, pressure=0.2)
        ]
        
        densities = [2800, 2600]  # kg/m³
        
        stratification = self.gradient_modeler.model_chamber_stratification(
            1000.0, compositions, densities, 0.5
        )
        
        assert isinstance(stratification, dict)
        assert 'heights' in stratification
        assert 'densities' in stratification
        assert 'temperatures' in stratification
        assert len(stratification['heights']) == len(compositions)

@pytest.mark.asyncio
class TestCompositionEvolutionTracker:
    """Test the main compositional evolution tracking system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Mock database URL for testing
        self.database_url = "postgresql://test:test@localhost:5432/test_db"
        
        # Create tracker with mocked database
        with patch('compositional_evolution_tracker.CompositionEvolutionDatabase'):
            self.tracker = CompositionEvolutionTracker(self.database_url)
            
            # Mock the database methods
            self.tracker.database.store_composition_evolution = AsyncMock(return_value="test-evolution-id")
            self.tracker.database.retrieve_composition_history = AsyncMock(return_value=[])
    
    async def test_temporal_evolution_tracking(self):
        """Test comprehensive temporal evolution tracking"""
        # Create initial chamber compositions
        initial_compositions = {
            'chamber_1': MagmaComposition(
                SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
                MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
                H2O=3.0, CO2=0.5, SO2=0.1,
                temperature=1200.0, pressure=0.2,
                trace_elements={'Ni': 150, 'Cr': 300},
                isotopic_ratios={'Sr87_Sr86': 0.7035}
            ),
            'chamber_2': MagmaComposition(
                SiO2=65.0, TiO2=0.8, Al2O3=16.0, FeO=4.0, MnO=0.1,
                MgO=2.0, CaO=4.0, Na2O=4.0, K2O=3.0, P2O5=0.1,
                H2O=4.0, CO2=0.2, SO2=0.05,
                temperature=900.0, pressure=0.15,
                trace_elements={'Ni': 20, 'Cr': 50},
                isotopic_ratios={'Sr87_Sr86': 0.7080}
            )
        }
        
        # Define time steps (in years)
        time_steps = np.array([0, 1, 2, 5, 10])
        
        # Define mixing and thermal parameters
        mixing_parameters = {
            'enable_mixing': True,
            'connections': [
                {'chamber1': 'chamber_1', 'chamber2': 'chamber_2', 'mixing_rate': 0.01}
            ]
        }
        
        thermal_parameters = {
            'cooling_rate': 2.0,  # °C/year
            'ascent_rate': 0.1    # m/s
        }
        
        # Run evolution tracking
        results = await self.tracker.track_temporal_evolution(
            'test-chamber-system',
            initial_compositions,
            time_steps,
            mixing_parameters,
            thermal_parameters
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'chamber_system_id' in results
        assert 'compositions' in results
        assert 'crystallization_data' in results
        assert 'volatile_evolution' in results
        assert 'validation_metrics' in results
        
        # Check that we have results for each time step
        assert len(results['compositions']) == len(time_steps)
        
        # Verify database storage was called
        self.tracker.database.store_composition_evolution.assert_called_once()
    
    async def test_evolution_step_processing(self):
        """Test individual evolution step processing"""
        compositions = {
            'chamber_1': MagmaComposition(
                SiO2=50.0, TiO2=1.5, Al2O3=15.0, FeO=10.0, MnO=0.2,
                MgO=8.0, CaO=11.0, Na2O=2.5, K2O=0.5, P2O5=0.3,
                H2O=3.0, CO2=0.5, SO2=0.1,
                temperature=1200.0, pressure=0.2
            )
        }
        
        mixing_parameters = {'enable_mixing': False}
        thermal_parameters = {'cooling_rate': 1.0, 'ascent_rate': 0.1}
        
        step_results = await self.tracker._process_evolution_step(
            compositions, 1.0, mixing_parameters, thermal_parameters
        )
        
        assert isinstance(step_results, dict)
        assert 'updated_compositions' in step_results
        assert 'crystallization_data' in step_results
        assert 'volatile_evolution' in step_results
        
        # Check that temperature decreased due to cooling
        updated_comp = step_results['updated_compositions']['chamber_1']
        assert updated_comp.temperature < compositions['chamber_1'].temperature

if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])