"""
Simple test for Eruption Characteristic Prediction System

This script performs basic functionality tests without relative imports.
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from backend.modules.magma_mixing.eruption_characteristic_prediction_system import (
        EruptionCharacteristicPredictionSystem
    )
    from backend.modules.magma_mixing.data_models import RheologicalProperties
    print("✓ Successfully imported eruption prediction system")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


async def test_basic_functionality():
    """Test basic functionality of the eruption prediction system"""
    
    print("Testing Eruption Characteristic Prediction System")
    print("=" * 50)
    
    # Initialize system with mock database config
    database_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'test_user',
        'password': 'test_password',
        'database': 'test_db'
    }
    
    system = EruptionCharacteristicPredictionSystem(database_config)
    
    try:
        print("1. Testing system initialization...")
        assert system is not None
        assert hasattr(system, 'empirical_relationships')
        assert hasattr(system, 'historical_eruption_database')
        assert hasattr(system, 'hazard_models')
        print("✓ System initialization successful")
        
        print("\n2. Testing empirical relationships loading...")
        assert 'intensity_mixing_relationship' in system.empirical_relationships
        assert 'explosivity_volatile_relationship' in system.empirical_relationships
        assert 'discharge_rate_rheology_relationship' in system.empirical_relationships
        print("✓ Empirical relationships loaded successfully")
        
        print("\n3. Testing historical database loading...")
        assert 'mount_vesuvius_1944' in system.historical_eruption_database
        assert 'santorini_minoan' in system.historical_eruption_database
        assert 'pinatubo_1991' in system.historical_eruption_database
        print("✓ Historical database loaded successfully")
        
        print("\n4. Testing hazard models loading...")
        assert 'lava_flow_model' in system.hazard_models
        assert 'pyroclastic_flow_model' in system.hazard_models
        assert 'ash_fall_model' in system.hazard_models
        print("✓ Hazard models loaded successfully")
        
        print("\n5. Testing intensity prediction...")
        mixing_degree = 0.6
        magma_properties = {
            'volatile_content': 0.04,
            'temperature': 1250,
            'viscosity': 2e6,
            'crystallinity': 0.3
        }
        chamber_conditions = {'pressure': 1.2e8}
        
        intensity_result = await system.predict_eruption_intensity(
            mixing_degree, magma_properties, chamber_conditions
        )
        
        assert hasattr(intensity_result, 'predicted_intensity')
        assert hasattr(intensity_result, 'intensity_confidence')
        assert 0 <= intensity_result.predicted_intensity <= 8
        assert 0 <= intensity_result.intensity_confidence <= 1
        print(f"✓ Intensity prediction: {intensity_result.predicted_intensity:.2f} (confidence: {intensity_result.intensity_confidence:.2f})")
        
        print("\n6. Testing explosivity forecasting...")
        volatile_evolution = {
            'H2O': [0.02, 0.03, 0.04],
            'CO2': [0.005, 0.008, 0.01],
            'SO2': [0.001, 0.002, 0.003]
        }
        pressure_evolution = [8e7, 1e8, 1.2e8]
        temperature_evolution = [1200, 1230, 1250]
        time_steps = [
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=1),
            datetime.now()
        ]
        
        explosivity_result = await system.forecast_explosivity(
            volatile_evolution, pressure_evolution, temperature_evolution, time_steps
        )
        
        assert hasattr(explosivity_result, 'predicted_explosivity')
        assert hasattr(explosivity_result, 'explosivity_index')
        assert 0 <= explosivity_result.predicted_explosivity <= 1
        assert 0 <= explosivity_result.explosivity_index <= 8
        print(f"✓ Explosivity forecast: {explosivity_result.predicted_explosivity:.3f} (VEI: {explosivity_result.explosivity_index})")
        
        print("\n7. Testing discharge rate prediction...")
        rheological_props = RheologicalProperties(
            viscosity=2e6,
            yield_strength=1.5e4,
            shear_rate=0.1,
            strain_rate=0.01,
            non_newtonian_index=0.8,
            thixotropy_parameter=0.1
        )
        
        conduit_geometry = {
            'diameter': 120,
            'length': 5500,
            'roughness': 0.015,
            'chamber_volume': 1.5e8
        }
        
        discharge_result = await system.predict_discharge_rate(
            rheological_props, conduit_geometry, 1.5e4, 0.04
        )
        
        assert hasattr(discharge_result, 'predicted_discharge_rate')
        assert discharge_result.predicted_discharge_rate >= 0
        print(f"✓ Discharge rate: {discharge_result.predicted_discharge_rate:.1f} m³/s")
        
        print("\n8. Testing style prediction...")
        mixed_magma_props = {
            'gas_content': 0.05,
            'viscosity': 3e6,
            'temperature': 1200,
            'ascent_rate': 0.3,
            'crystallinity': 0.3
        }
        
        env_conditions = {
            'water_interaction': 0.1,
            'depth_to_water': 400,
            'external_water': False
        }
        
        style_result = await system.predict_eruption_style(
            mixed_magma_props, env_conditions
        )
        
        assert hasattr(style_result, 'predicted_style')
        assert hasattr(style_result, 'style_probability')
        valid_styles = ['effusive', 'explosive', 'mixed', 'phreatomagmatic']
        assert style_result.predicted_style in valid_styles
        print(f"✓ Style prediction: {style_result.predicted_style} (confidence: {style_result.style_confidence:.2f})")
        
        print("\n9. Testing gas volume fraction calculation...")
        volatile_content = {'H2O': 0.04, 'CO2': 0.01, 'SO2': 0.002}
        gas_fraction = system._calculate_gas_volume_fraction(volatile_content, 1e8, 1200)
        assert 0 <= gas_fraction <= 1
        print(f"✓ Gas volume fraction: {gas_fraction:.4f}")
        
        print("\n10. Testing prediction confidence calculation...")
        confidence = system._calculate_prediction_confidence(
            3.5, {'param1': 1.0, 'param2': 2.0}, 'intensity'
        )
        assert 0 <= confidence <= 1
        print(f"✓ Prediction confidence: {confidence:.3f}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("The Eruption Characteristic Prediction System is working correctly.")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await system.close()


def test_data_structures():
    """Test data structure creation and validation"""
    
    print("\nTesting Data Structures...")
    print("-" * 30)
    
    try:
        # Test RheologicalProperties
        rheological_props = RheologicalProperties(
            viscosity=1e6,
            yield_strength=1e4,
            shear_rate=0.1,
            strain_rate=0.01,
            non_newtonian_index=0.8,
            thixotropy_parameter=0.1
        )
        
        assert rheological_props.viscosity == 1e6
        assert rheological_props.yield_strength == 1e4
        print("✓ RheologicalProperties creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False


if __name__ == '__main__':
    print("Starting Eruption Characteristic Prediction System Tests")
    print("=" * 60)
    
    # Test data structures first
    data_success = test_data_structures()
    
    if data_success:
        # Run async tests
        async_success = asyncio.run(test_basic_functionality())
        
        if async_success:
            print("\n🎉 ALL VALIDATION COMPLETED SUCCESSFULLY!")
            sys.exit(0)
        else:
            print("\n❌ Async tests failed")
            sys.exit(1)
    else:
        print("\n❌ Data structure tests failed")
        sys.exit(1)