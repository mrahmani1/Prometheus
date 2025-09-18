"""
Simple test for Sill Emplacement Mechanism Simulator
"""

import asyncio
import numpy as np
from modules.sill_emplacement_mechanism_simulator import (
    SillEmplacementMechanismSimulator,
    MagmaProperties,
    HostRockProperties
)

async def test_basic_functionality():
    """Test basic functionality of the simulator"""
    print("Testing Sill Emplacement Mechanism Simulator...")
    
    # Initialize simulator
    simulator = SillEmplacementMechanismSimulator()
    print("✓ Simulator initialized successfully")
    
    # Create sample magma properties
    magma_props = MagmaProperties(
        composition={'SiO2': 52.0, 'Al2O3': 16.0, 'FeO': 8.0, 'MgO': 6.0, 'CaO': 10.0, 'Na2O': 3.0, 'K2O': 2.0},
        temperature=1200.0, pressure=1e8, viscosity=1e3, density=2700.0,
        volatile_content={'H2O': 2.0, 'CO2': 0.5}, crystallinity=0.1, bubble_fraction=0.05,
        thermal_conductivity=2.5, specific_heat=1200.0, latent_heat_crystallization=3e5,
        surface_tension=0.4, compressibility=1e-10, thermal_expansion=3e-5
    )
    print("✓ Magma properties created")
    
    # Create sample host rock properties
    host_rock_props = HostRockProperties(
        rock_type='sandstone', density=2400.0, porosity=0.15, permeability=1e-15,
        youngs_modulus=2e10, poissons_ratio=0.25, tensile_strength=5e6, cohesion=1e7,
        friction_angle=0.6, thermal_conductivity=2.0, specific_heat=800.0,
        thermal_diffusivity=1e-6, thermal_expansion=1e-5, fracture_toughness=1e6,
        crack_density=1e6, anisotropy_factor=1.2, layering_thickness=0.5,
        bedding_orientation=(0.0, 0.1)
    )
    print("✓ Host rock properties created")
    
    # Create injection conditions (smaller domain for testing)
    injection_conditions = {
        'injection_rate': 0.1, 'injection_pressure': 1.5e8,
        'source_location': (50, 50, 100), 'domain_size': [100, 100, 50],
        'ambient_pressure': 1e5, 'ambient_temperature': 293.15
    }
    print("✓ Injection conditions defined")
    
    # Test magma injection pathways simulation
    try:
        pathways = await simulator.simulate_magma_injection_pathways(
            magma_props, host_rock_props, injection_conditions, 'coupled'
        )
        print(f"✓ Magma injection pathways simulated: {len(pathways)} pathways")
        
        # Validate pathway results
        for i, pathway in enumerate(pathways):
            assert pathway.pathway_length > 0, f"Pathway {i} has invalid length"
            assert pathway.tortuosity >= 1.0, f"Pathway {i} has invalid tortuosity"
            assert pathway.flow_rate > 0, f"Pathway {i} has invalid flow rate"
        print("✓ Pathway validation passed")
        
    except Exception as e:
        print(f"✗ Pathway simulation failed: {e}")
        return False
    
    # Test sill inflation sequence simulation
    try:
        inflation_sequence = await simulator.simulate_sill_inflation_sequence(
            magma_props, host_rock_props, pathways, 'elastoplastic'
        )
        print("✓ Sill inflation sequence simulated")
        
        # Validate inflation results
        assert len(inflation_sequence.pressure_evolution) > 0, "No pressure evolution data"
        assert len(inflation_sequence.volume_evolution) > 0, "No volume evolution data"
        assert inflation_sequence.inflation_efficiency >= 0, "Invalid inflation efficiency"
        print("✓ Inflation validation passed")
        
    except Exception as e:
        print(f"✗ Inflation simulation failed: {e}")
        return False
    
    # Test thermal effects simulation
    try:
        sill_geometry = np.random.rand(20, 20, 5) * 100
        thermal_evolution = await simulator.simulate_thermal_effects(
            magma_props, host_rock_props, sill_geometry, 'coupled'
        )
        print("✓ Thermal effects simulated")
        
        # Validate thermal results
        assert thermal_evolution.peclet_number > 0, "Invalid Peclet number"
        assert thermal_evolution.stefan_number > 0, "Invalid Stefan number"
        assert thermal_evolution.thermal_diffusion_time > 0, "Invalid diffusion time"
        print("✓ Thermal validation passed")
        
    except Exception as e:
        print(f"✗ Thermal simulation failed: {e}")
        return False
    
    # Test complete simulation
    try:
        complete_simulation = await simulator.run_complete_emplacement_simulation(
            magma_props, host_rock_props, injection_conditions
        )
        print("✓ Complete emplacement simulation finished")
        
        # Validate complete simulation
        assert complete_simulation.simulation_id is not None, "No simulation ID"
        assert len(complete_simulation.injection_pathways) > 0, "No pathways in complete simulation"
        assert complete_simulation.validation_metrics['overall_score'] > 0, "Invalid validation score"
        print(f"✓ Complete simulation validation passed (score: {complete_simulation.validation_metrics['overall_score']:.3f})")
        
    except Exception as e:
        print(f"✗ Complete simulation failed: {e}")
        return False
    
    # Test experimental validation
    try:
        exp_validation = await simulator.validate_against_laboratory_experiments(complete_simulation)
        print("✓ Experimental validation completed")
        
        # Validate experimental results
        assert 'overall_experimental_validation' in exp_validation, "No overall validation score"
        assert exp_validation['overall_experimental_validation'] >= 0, "Invalid experimental validation"
        print(f"✓ Experimental validation passed (score: {exp_validation['overall_experimental_validation']:.3f})")
        
    except Exception as e:
        print(f"✗ Experimental validation failed: {e}")
        return False
    
    print("\n🎉 All tests passed successfully!")
    print(f"Final Results:")
    print(f"  - Simulation ID: {complete_simulation.simulation_id}")
    print(f"  - Number of pathways: {len(complete_simulation.injection_pathways)}")
    print(f"  - Validation score: {complete_simulation.validation_metrics['overall_score']:.3f}")
    print(f"  - Experimental validation: {exp_validation['overall_experimental_validation']:.3f}")
    print(f"  - Inflation efficiency: {complete_simulation.inflation_sequence.inflation_efficiency:.3f}")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    if success:
        print("\n✅ Sill Emplacement Mechanism Simulator is working correctly!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")