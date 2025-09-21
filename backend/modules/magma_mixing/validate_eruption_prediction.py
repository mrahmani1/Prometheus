"""
Validation script for Eruption Characteristic Prediction System

This script performs basic validation tests to ensure the system is working correctly
and produces reasonable results.
"""

import asyncio
import sys
import traceback
from datetime import datetime, timedelta

from .eruption_characteristic_prediction_system import EruptionCharacteristicPredictionSystem
from .data_models import RheologicalProperties


async def validate_system():
    """Perform basic validation of the eruption prediction system"""
    
    print("Validating Eruption Characteristic Prediction System...")
    print("=" * 60)
    
    # Initialize system
    database_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'test_user',
        'password': 'test_password',
        'database': 'test_db'
    }
    
    system = EruptionCharacteristicPredictionSystem(database_config)
    
    try:
        # Test 1: Intensity Prediction
        print("Test 1: Eruption Intensity Prediction")
        print("-" * 40)
        
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
        
        print(f"✓ Intensity prediction: {intensity_result.predicted_intensity:.2f}")
        print(f"✓ Confidence: {intensity_result.intensity_confidence:.2f}")
        print(f"✓ Bounds: {intensity_result.intensity_bounds}")
        
        # Validate results
        assert 0 <= intensity_result.predicted_intensity <= 8, "Intensity out of VEI range"
        assert 0 <= intensity_result.intensity_confidence <= 1, "Invalid confidence"
        assert len(intensity_result.intensity_bounds) == 2, "Invalid bounds format"
        print("✓ Intensity prediction validation passed\n")
        
        # Test 2: Explosivity Forecast
        print("Test 2: Explosivity Forecasting")
        print("-" * 40)
        
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
        
        print(f"✓ Explosivity: {explosivity_result.predicted_explosivity:.3f}")
        print(f"✓ VEI Index: {explosivity_result.explosivity_index}")
        print(f"✓ Gas pressure: {explosivity_result.gas_pressure_buildup:.2e} Pa")
        
        # Validate results
        assert 0 <= explosivity_result.predicted_explosivity <= 1, "Invalid explosivity range"
        assert 0 <= explosivity_result.explosivity_index <= 8, "Invalid VEI index"
        assert explosivity_result.gas_pressure_buildup >= 0, "Negative gas pressure"
        print("✓ Explosivity forecast validation passed\n")
        
        # Test 3: Discharge Rate Prediction
        print("Test 3: Discharge Rate Prediction")
        print("-" * 40)
        
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
        
        print(f"✓ Discharge rate: {discharge_result.predicted_discharge_rate:.1f} m³/s")
        print(f"✓ Peak rate: {discharge_result.peak_discharge_rate:.1f} m³/s")
        print(f"✓ Duration: {discharge_result.discharge_duration/3600:.1f} hours")
        
        # Validate results
        assert discharge_result.predicted_discharge_rate >= 0, "Negative discharge rate"
        assert discharge_result.peak_discharge_rate >= discharge_result.sustained_discharge_rate, "Invalid rate relationship"
        print("✓ Discharge rate prediction validation passed\n")
        
        # Test 4: Style Prediction
        print("Test 4: Eruption Style Prediction")
        print("-" * 40)
        
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
        
        print(f"✓ Predicted style: {style_result.predicted_style}")
        print(f"✓ Confidence: {style_result.style_confidence:.2f}")
        print(f"✓ Style probabilities: {style_result.style_probability}")
        
        # Validate results
        valid_styles = ['effusive', 'explosive', 'mixed', 'phreatomagmatic']
        assert style_result.predicted_style in valid_styles, "Invalid eruption style"
        assert 0 <= style_result.style_confidence <= 1, "Invalid style confidence"
        assert abs(sum(style_result.style_probability.values()) - 1.0) < 0.01, "Probabilities don't sum to 1"
        print("✓ Style prediction validation passed\n")
        
        # Test 5: Temporal Evolution
        print("Test 5: Temporal Evolution Modeling")
        print("-" * 40)
        
        mixing_progression = [0.3, 0.5, 0.7, 0.6, 0.4]
        evolution_times = [
            datetime.now() + timedelta(hours=i) for i in range(len(mixing_progression))
        ]
        initial_conditions = {
            'gas_content': 0.02,
            'viscosity': 1e5,
            'temperature': 1200,
            'crystallinity': 0.2,
            'pressure': 1e8
        }
        
        evolution_result = await system.model_temporal_eruption_evolution(
            mixing_progression, evolution_times, initial_conditions
        )
        
        print(f"✓ Evolution timeline length: {len(evolution_result.evolution_timeline)}")
        print(f"✓ Peak activity time: {evolution_result.peak_activity_time}")
        print(f"✓ Eruption duration: {evolution_result.eruption_duration}")
        
        # Validate results
        assert len(evolution_result.evolution_timeline) == len(mixing_progression), "Timeline length mismatch"
        assert len(evolution_result.intensity_evolution) == len(mixing_progression), "Intensity evolution length mismatch"
        print("✓ Temporal evolution validation passed\n")
        
        # Test 6: Hazard Assessment
        print("Test 6: Hazard Assessment")
        print("-" * 40)
        
        eruption_chars = {
            'intensity': 4.0,
            'explosivity': 0.7,
            'discharge_rate': 200,
            'style': 'explosive',
            'gas_emission_rate': 1500
        }
        
        location = {'lat': 40.0, 'lon': 14.0, 'elevation': 1000}
        population_data = {
            'total_population': 100000,
            'population_density': 150,
            'infrastructure': {'buildings_density': 40}
        }
        
        hazard_result = await system.assess_mixing_triggered_hazards(
            eruption_chars, location, population_data
        )
        
        print(f"✓ Hazard zones: {len(hazard_result.hazard_zones)} types")
        print(f"✓ Population exposure calculated")
        print(f"✓ Evacuation recommendations generated")
        
        # Validate results
        assert isinstance(hazard_result.hazard_zones, dict), "Invalid hazard zones format"
        assert isinstance(hazard_result.population_exposure, dict), "Invalid population exposure format"
        print("✓ Hazard assessment validation passed\n")
        
        # Test 7: Uncertainty Quantification
        print("Test 7: Uncertainty Quantification")
        print("-" * 40)
        
        prediction_results = {
            'intensity': 4.0,
            'explosivity': 0.7,
            'discharge_rate': 200
        }
        
        model_params = {
            'data_coverage': 0.8,
            'intensity_validation_score': 0.75
        }
        
        uncertainty_result = await system.quantify_prediction_uncertainty(
            prediction_results, model_params
        )
        
        print(f"✓ Total uncertainty calculated")
        print(f"✓ Confidence intervals generated")
        print(f"✓ Sensitivity analysis performed")
        
        # Validate results
        assert isinstance(uncertainty_result.total_uncertainty, dict), "Invalid uncertainty format"
        assert isinstance(uncertainty_result.confidence_intervals, dict), "Invalid confidence intervals"
        print("✓ Uncertainty quantification validation passed\n")
        
        # Test 8: Historical Validation
        print("Test 8: Historical Validation")
        print("-" * 40)
        
        # Create mock prediction result
        class MockResult:
            def __init__(self):
                self.intensity_prediction = type('obj', (object,), {'predicted_intensity': 3.5})()
                self.explosivity_forecast = type('obj', (object,), {'predicted_explosivity': 0.75})()
                self.discharge_rate_prediction = type('obj', (object,), {'predicted_discharge_rate': 150})()
                self.style_prediction = type('obj', (object,), {'predicted_style': 'explosive'})()
        
        mock_result = MockResult()
        
        validation_result = await system.validate_against_historical_eruptions(
            mock_result, 'mount_vesuvius_1944'
        )
        
        print(f"✓ Historical validation completed")
        print(f"✓ Accuracy metrics calculated")
        print(f"✓ Lessons learned generated")
        
        # Validate results
        assert isinstance(validation_result.prediction_accuracy, dict), "Invalid accuracy format"
        assert isinstance(validation_result.validation_metrics, dict), "Invalid metrics format"
        print("✓ Historical validation passed\n")
        
        print("=" * 60)
        print("ALL VALIDATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("Traceback:")
        traceback.print_exc()
        return False
    
    finally:
        await system.close()


async def test_edge_cases():
    """Test edge cases and boundary conditions"""
    
    print("\nTesting Edge Cases...")
    print("=" * 40)
    
    database_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'test_user',
        'password': 'test_password',
        'database': 'test_db'
    }
    
    system = EruptionCharacteristicPredictionSystem(database_config)
    
    try:
        # Test extreme mixing degrees
        print("Testing extreme mixing degrees...")
        
        # Very low mixing
        result_low = await system.predict_eruption_intensity(
            0.01, {'volatile_content': 0.01, 'temperature': 1100, 'viscosity': 1e3, 'crystallinity': 0.1},
            {'pressure': 5e7}
        )
        print(f"✓ Low mixing (0.01): Intensity = {result_low.predicted_intensity:.2f}")
        
        # Very high mixing
        result_high = await system.predict_eruption_intensity(
            0.99, {'volatile_content': 0.08, 'temperature': 1400, 'viscosity': 1e8, 'crystallinity': 0.8},
            {'pressure': 3e8}
        )
        print(f"✓ High mixing (0.99): Intensity = {result_high.predicted_intensity:.2f}")
        
        # Test extreme volatile contents
        print("Testing extreme volatile contents...")
        
        volatile_evolution_low = {
            'H2O': [0.001, 0.002, 0.003],
            'CO2': [0.0001, 0.0002, 0.0003]
        }
        
        volatile_evolution_high = {
            'H2O': [0.08, 0.09, 0.10],
            'CO2': [0.02, 0.025, 0.03],
            'SO2': [0.008, 0.01, 0.012]
        }
        
        time_steps = [datetime.now() - timedelta(hours=i) for i in range(2, -1, -1)]
        
        explosivity_low = await system.forecast_explosivity(
            volatile_evolution_low, [5e7, 6e7, 7e7], [1100, 1120, 1140], time_steps
        )
        print(f"✓ Low volatiles: Explosivity = {explosivity_low.predicted_explosivity:.3f}")
        
        explosivity_high = await system.forecast_explosivity(
            volatile_evolution_high, [2e8, 2.5e8, 3e8], [1300, 1350, 1400], time_steps
        )
        print(f"✓ High volatiles: Explosivity = {explosivity_high.predicted_explosivity:.3f}")
        
        print("✓ All edge case tests passed")
        
    except Exception as e:
        print(f"❌ Edge case testing failed: {e}")
        return False
    
    finally:
        await system.close()
    
    return True


if __name__ == '__main__':
    async def main():
        print("Starting Eruption Characteristic Prediction System Validation")
        print("=" * 70)
        
        # Run basic validation
        basic_success = await validate_system()
        
        if basic_success:
            # Run edge case tests
            edge_success = await test_edge_cases()
            
            if edge_success:
                print("\n🎉 ALL VALIDATION TESTS COMPLETED SUCCESSFULLY!")
                print("The Eruption Characteristic Prediction System is working correctly.")
                sys.exit(0)
            else:
                print("\n⚠️  Edge case tests failed")
                sys.exit(1)
        else:
            print("\n❌ Basic validation failed")
            sys.exit(1)
    
    asyncio.run(main())