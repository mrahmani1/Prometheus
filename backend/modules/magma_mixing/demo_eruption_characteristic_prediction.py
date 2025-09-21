"""
Demonstration of Eruption Characteristic Prediction System

This script demonstrates the comprehensive eruption characteristic prediction
capabilities based on magma mixing processes, including all major prediction
components and their integration.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging

from .eruption_characteristic_prediction_system import EruptionCharacteristicPredictionSystem
from .data_models import RheologicalProperties

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_eruption_prediction():
    """Demonstrate comprehensive eruption characteristic prediction"""
    
    print("=" * 80)
    print("ERUPTION CHARACTERISTIC PREDICTION SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Initialize the prediction system
    database_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'volcanic_user',
        'password': 'volcanic_password',
        'database': 'volcanic_monitoring'
    }
    
    prediction_system = EruptionCharacteristicPredictionSystem(database_config)
    
    try:
        print("\n1. ERUPTION INTENSITY PREDICTION")
        print("-" * 50)
        
        # Scenario: High mixing degree with volatile-rich magma
        mixing_degree = 0.75
        magma_properties = {
            'volatile_content': 0.055,  # High volatile content
            'temperature': 1250,
            'viscosity': 3e6,
            'crystallinity': 0.35
        }
        chamber_conditions = {
            'pressure': 1.5e8
        }
        
        intensity_prediction = await prediction_system.predict_eruption_intensity(
            mixing_degree, magma_properties, chamber_conditions
        )
        
        print(f"Mixing Degree: {mixing_degree}")
        print(f"Predicted Intensity: {intensity_prediction.predicted_intensity:.2f} (VEI scale)")
        print(f"Confidence: {intensity_prediction.intensity_confidence:.2f}")
        print(f"Intensity Bounds: {intensity_prediction.intensity_bounds[0]:.2f} - {intensity_prediction.intensity_bounds[1]:.2f}")
        print("Contributing Factors:")
        for factor, contribution in intensity_prediction.contributing_factors.items():
            print(f"  {factor}: {contribution:.3f}")
        
        print("\n2. EXPLOSIVITY FORECASTING")
        print("-" * 50)
        
        # Volatile evolution scenario
        volatile_evolution = {
            'H2O': [0.02, 0.035, 0.045, 0.055, 0.06],
            'CO2': [0.005, 0.008, 0.012, 0.015, 0.018],
            'SO2': [0.001, 0.002, 0.003, 0.004, 0.005],
            'H2S': [0.0005, 0.001, 0.0015, 0.002, 0.0025]
        }
        pressure_evolution = [8e7, 1e8, 1.2e8, 1.5e8, 1.8e8]
        temperature_evolution = [1200, 1220, 1240, 1250, 1260]
        time_steps = [
            datetime.now() - timedelta(hours=4),
            datetime.now() - timedelta(hours=3),
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=1),
            datetime.now()
        ]
        
        explosivity_forecast = await prediction_system.forecast_explosivity(
            volatile_evolution, pressure_evolution, temperature_evolution, time_steps
        )
        
        print(f"Current Volatile Content:")
        for species, values in volatile_evolution.items():
            print(f"  {species}: {values[-1]:.4f}")
        print(f"Predicted Explosivity: {explosivity_forecast.predicted_explosivity:.3f}")
        print(f"Explosivity Index (VEI): {explosivity_forecast.explosivity_index}")
        print(f"Gas Pressure Buildup: {explosivity_forecast.gas_pressure_buildup:.2e} Pa")
        print(f"Forecast Horizon: {explosivity_forecast.forecast_horizon}")
        print(f"Confidence: {explosivity_forecast.explosivity_confidence:.2f}")
        
        print("\n3. DISCHARGE RATE PREDICTION")
        print("-" * 50)
        
        # Rheological properties for mixed magma
        rheological_properties = RheologicalProperties(
            viscosity=3e6,
            yield_strength=2e4,
            shear_rate=0.05,
            strain_rate=0.01,
            non_newtonian_index=0.7,
            thixotropy_parameter=0.15
        )
        
        conduit_geometry = {
            'diameter': 150,  # meters
            'length': 6000,   # meters
            'roughness': 0.02,
            'chamber_volume': 2e8  # m³
        }
        
        pressure_gradient = 2e4  # Pa/m
        gas_content = 0.055
        
        discharge_prediction = await prediction_system.predict_discharge_rate(
            rheological_properties, conduit_geometry, pressure_gradient, gas_content
        )
        
        print(f"Magma Viscosity: {rheological_properties.viscosity:.2e} Pa·s")
        print(f"Yield Strength: {rheological_properties.yield_strength:.2e} Pa")
        print(f"Gas Content: {gas_content:.3f}")
        print(f"Predicted Discharge Rate: {discharge_prediction.predicted_discharge_rate:.1f} m³/s")
        print(f"Peak Discharge Rate: {discharge_prediction.peak_discharge_rate:.1f} m³/s")
        print(f"Sustained Discharge Rate: {discharge_prediction.sustained_discharge_rate:.1f} m³/s")
        print(f"Estimated Duration: {discharge_prediction.discharge_duration/3600:.1f} hours")
        
        print("\n4. ERUPTION STYLE PREDICTION")
        print("-" * 50)
        
        # Mixed magma properties for style prediction
        mixed_magma_properties = {
            'gas_content': 0.055,
            'viscosity': 3e6,
            'temperature': 1250,
            'ascent_rate': 0.4,
            'crystallinity': 0.35,
            'temporal_variability': 0.25,
            'conduit_complexity': 0.3,
            'supply_variability': 0.2
        }
        
        environmental_conditions = {
            'water_interaction': 0.15,
            'depth_to_water': 300,
            'external_water': True
        }
        
        style_prediction = await prediction_system.predict_eruption_style(
            mixed_magma_properties, environmental_conditions
        )
        
        print(f"Predicted Style: {style_prediction.predicted_style}")
        print(f"Style Confidence: {style_prediction.style_confidence:.2f}")
        print("Style Probabilities:")
        for style, probability in style_prediction.style_probability.items():
            print(f"  {style}: {probability:.3f}")
        print("Controlling Parameters:")
        for param, value in style_prediction.controlling_parameters.items():
            print(f"  {param}: {value:.3f}")
        
        print("\n5. TEMPORAL ERUPTION EVOLUTION")
        print("-" * 50)
        
        # Mixing progression over time
        mixing_progression = [0.3, 0.5, 0.7, 0.85, 0.9, 0.8, 0.6, 0.4, 0.2]
        evolution_time_steps = [
            datetime.now() + timedelta(hours=i*2) for i in range(len(mixing_progression))
        ]
        
        initial_conditions = {
            'gas_content': 0.02,
            'viscosity': 1e5,
            'temperature': 1200,
            'crystallinity': 0.2,
            'pressure': 1e8
        }
        
        temporal_evolution = await prediction_system.model_temporal_eruption_evolution(
            mixing_progression, evolution_time_steps, initial_conditions
        )
        
        print(f"Peak Activity Time: {temporal_evolution.peak_activity_time}")
        print(f"Eruption Duration: {temporal_evolution.eruption_duration}")
        print(f"Number of Style Transitions: {len(temporal_evolution.style_transitions)}")
        print("Intensity Evolution:")
        for i, intensity in enumerate(temporal_evolution.intensity_evolution):
            print(f"  Step {i+1}: {intensity:.2f}")
        
        if temporal_evolution.style_transitions:
            print("Style Transitions:")
            for transition in temporal_evolution.style_transitions:
                print(f"  {transition[0]}: {transition[1]} → {transition[2]}")
        
        print("\n6. HAZARD ASSESSMENT")
        print("-" * 50)
        
        # Eruption characteristics for hazard assessment
        eruption_characteristics = {
            'intensity': intensity_prediction.predicted_intensity,
            'explosivity': explosivity_forecast.predicted_explosivity,
            'discharge_rate': discharge_prediction.predicted_discharge_rate,
            'style': style_prediction.predicted_style,
            'gas_emission_rate': 3000,  # kg/s
            'water_interaction': environmental_conditions['water_interaction']
        }
        
        location = {
            'lat': 40.821,
            'lon': 14.426,
            'elevation': 1281
        }
        
        population_data = {
            'total_population': 600000,
            'population_density': 250,  # people/km²
            'infrastructure': {
                'buildings_density': 80,
                'roads_density': 3,
                'utilities_density': 8,
                'airports_density': 0.1,
                'hospitals_density': 0.5
            }
        }
        
        hazard_assessment = await prediction_system.assess_mixing_triggered_hazards(
            eruption_characteristics, location, population_data
        )
        
        print("Hazard Zones:")
        for hazard_type, zones in hazard_assessment.hazard_zones.items():
            print(f"  {hazard_type}:")
            for zone_name, zone_data in zones.items():
                print(f"    {zone_name}: {zone_data.get('distance', 0)/1000:.1f} km, "
                      f"P={zone_data.get('probability', 0):.2f}")
        
        print("\nPopulation Exposure:")
        total_exposed = 0
        for hazard_type, exposure in hazard_assessment.population_exposure.items():
            hazard_total = sum(exposure.values()) if isinstance(exposure, dict) else exposure
            total_exposed += hazard_total
            print(f"  {hazard_type}: {hazard_total:,} people")
        
        print(f"\nTotal Population at Risk: {total_exposed:,}")
        
        print("\nEvacuation Recommendations:")
        evac = hazard_assessment.evacuation_recommendations
        print(f"  Immediate Evacuation: {evac.get('immediate_evacuation_population', 0):,} people")
        print(f"  Precautionary Evacuation: {evac.get('precautionary_evacuation_population', 0):,} people")
        print(f"  Evacuation Radius: {evac.get('evacuation_radius', 0)/1000:.1f} km")
        print(f"  Timeline: {evac.get('evacuation_timeline', 0)} hours")
        
        print("\n7. UNCERTAINTY QUANTIFICATION")
        print("-" * 50)
        
        prediction_results = {
            'intensity': intensity_prediction.predicted_intensity,
            'explosivity': explosivity_forecast.predicted_explosivity,
            'discharge_rate': discharge_prediction.predicted_discharge_rate,
            'style_probability': style_prediction.style_confidence
        }
        
        model_parameters = {
            'data_coverage': 0.82,
            'intensity_validation_score': 0.78,
            'explosivity_validation_score': 0.74,
            'discharge_rate_validation_score': 0.71,
            'style_prediction_validation_score': 0.69,
            'mixing_degree': mixing_degree,
            'volatile_content': magma_properties['volatile_content']
        }
        
        uncertainty_quantification = await prediction_system.quantify_prediction_uncertainty(
            prediction_results, model_parameters
        )
        
        print("Total Uncertainty:")
        for param, uncertainty in uncertainty_quantification.total_uncertainty.items():
            print(f"  {param}: ±{uncertainty:.3f}")
        
        print("\nConfidence Intervals (95%):")
        for param, (lower, upper) in uncertainty_quantification.confidence_intervals.items():
            print(f"  {param}: [{lower:.3f}, {upper:.3f}]")
        
        print("\nSensitivity Analysis:")
        for param, sensitivity in uncertainty_quantification.sensitivity_analysis.items():
            print(f"  {param}: {sensitivity:.3f}")
        
        print("\n8. HISTORICAL VALIDATION")
        print("-" * 50)
        
        # Create a mock prediction result for validation
        class MockPredictionResult:
            def __init__(self):
                self.intensity_prediction = type('obj', (object,), {
                    'predicted_intensity': intensity_prediction.predicted_intensity
                })()
                self.explosivity_forecast = type('obj', (object,), {
                    'predicted_explosivity': explosivity_forecast.predicted_explosivity
                })()
                self.discharge_rate_prediction = type('obj', (object,), {
                    'predicted_discharge_rate': discharge_prediction.predicted_discharge_rate
                })()
                self.style_prediction = type('obj', (object,), {
                    'predicted_style': style_prediction.predicted_style
                })()
        
        mock_result = MockPredictionResult()
        
        validation_result = await prediction_system.validate_against_historical_eruptions(
            mock_result, 'mount_vesuvius_1944'
        )
        
        print(f"Historical Eruption: {validation_result.eruption_name}")
        print(f"Eruption Date: {validation_result.eruption_date}")
        print("\nPrediction Accuracy:")
        for characteristic, accuracy in validation_result.prediction_accuracy.items():
            print(f"  {characteristic}: {accuracy:.3f}")
        
        print(f"\nOverall Validation Score: {validation_result.validation_metrics['overall_accuracy']:.3f}")
        print(f"RMSE: {validation_result.validation_metrics['rmse']:.3f}")
        
        print("\nLessons Learned:")
        for lesson in validation_result.lessons_learned[:3]:  # Show first 3 lessons
            print(f"  • {lesson}")
        
        print("\n9. PREDICTION SUMMARY")
        print("-" * 50)
        
        print(f"Scenario: High-degree magma mixing (mixing degree = {mixing_degree})")
        print(f"Predicted Eruption Characteristics:")
        print(f"  • Intensity: {intensity_prediction.predicted_intensity:.1f} VEI")
        print(f"  • Style: {style_prediction.predicted_style}")
        print(f"  • Explosivity: {explosivity_forecast.predicted_explosivity:.2f}")
        print(f"  • Discharge Rate: {discharge_prediction.predicted_discharge_rate:.0f} m³/s")
        print(f"  • Duration: {discharge_prediction.discharge_duration/3600:.1f} hours")
        
        print(f"\nHazard Impact:")
        print(f"  • Population at Risk: {total_exposed:,}")
        print(f"  • Evacuation Needed: {evac.get('immediate_evacuation_population', 0) + evac.get('precautionary_evacuation_population', 0):,}")
        print(f"  • Economic Risk: ${sum(hazard_assessment.infrastructure_risk.values())/1e6:.1f}M")
        
        print(f"\nPrediction Confidence:")
        print(f"  • Intensity: {intensity_prediction.intensity_confidence:.2f}")
        print(f"  • Style: {style_prediction.style_confidence:.2f}")
        print(f"  • Explosivity: {explosivity_forecast.explosivity_confidence:.2f}")
        print(f"  • Overall Uncertainty: ±{np.mean(list(uncertainty_quantification.total_uncertainty.values())):.2f}")
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        raise
    
    finally:
        await prediction_system.close()


async def demonstrate_comparative_scenarios():
    """Demonstrate predictions for different mixing scenarios"""
    
    print("\n" + "=" * 80)
    print("COMPARATIVE SCENARIO ANALYSIS")
    print("=" * 80)
    
    database_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'volcanic_user',
        'password': 'volcanic_password',
        'database': 'volcanic_monitoring'
    }
    
    prediction_system = EruptionCharacteristicPredictionSystem(database_config)
    
    scenarios = [
        {
            'name': 'Low Mixing - Effusive',
            'mixing_degree': 0.2,
            'volatile_content': 0.015,
            'viscosity': 1e4,
            'gas_content': 0.015
        },
        {
            'name': 'Moderate Mixing - Mixed Style',
            'mixing_degree': 0.5,
            'volatile_content': 0.035,
            'viscosity': 1e6,
            'gas_content': 0.035
        },
        {
            'name': 'High Mixing - Explosive',
            'mixing_degree': 0.8,
            'volatile_content': 0.065,
            'viscosity': 5e6,
            'gas_content': 0.065
        }
    ]
    
    try:
        results = []
        
        for scenario in scenarios:
            print(f"\nScenario: {scenario['name']}")
            print("-" * 40)
            
            magma_properties = {
                'volatile_content': scenario['volatile_content'],
                'temperature': 1200,
                'viscosity': scenario['viscosity'],
                'crystallinity': 0.25,
                'gas_content': scenario['gas_content'],
                'ascent_rate': 0.1
            }
            
            chamber_conditions = {'pressure': 1e8}
            environmental_conditions = {
                'water_interaction': 0.05,
                'depth_to_water': 500,
                'external_water': False
            }
            
            # Predict characteristics
            intensity = await prediction_system.predict_eruption_intensity(
                scenario['mixing_degree'], magma_properties, chamber_conditions
            )
            
            style = await prediction_system.predict_eruption_style(
                magma_properties, environmental_conditions
            )
            
            result = {
                'scenario': scenario['name'],
                'mixing_degree': scenario['mixing_degree'],
                'intensity': intensity.predicted_intensity,
                'style': style.predicted_style,
                'style_confidence': style.style_confidence,
                'intensity_confidence': intensity.intensity_confidence
            }
            results.append(result)
            
            print(f"Mixing Degree: {scenario['mixing_degree']}")
            print(f"Predicted Intensity: {intensity.predicted_intensity:.2f}")
            print(f"Predicted Style: {style.predicted_style}")
            print(f"Style Confidence: {style.style_confidence:.2f}")
        
        # Summary comparison
        print("\n" + "=" * 60)
        print("SCENARIO COMPARISON SUMMARY")
        print("=" * 60)
        
        print(f"{'Scenario':<25} {'Mixing':<8} {'Intensity':<10} {'Style':<12} {'Confidence':<10}")
        print("-" * 65)
        
        for result in results:
            print(f"{result['scenario']:<25} {result['mixing_degree']:<8.1f} "
                  f"{result['intensity']:<10.2f} {result['style']:<12} "
                  f"{result['style_confidence']:<10.2f}")
        
        print("\nKey Observations:")
        print("• Higher mixing degrees correlate with increased eruption intensity")
        print("• Volatile content strongly influences eruption style")
        print("• Style prediction confidence varies with parameter certainty")
        
    finally:
        await prediction_system.close()


if __name__ == '__main__':
    # Run the demonstrations
    asyncio.run(demonstrate_eruption_prediction())
    asyncio.run(demonstrate_comparative_scenarios())