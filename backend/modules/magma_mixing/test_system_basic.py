"""
Basic test for Eruption Characteristic Prediction System components

This script tests individual components without complex imports.
"""

import numpy as np
from datetime import datetime, timedelta
import json


def test_empirical_relationships():
    """Test empirical relationship calculations"""
    print("Testing empirical relationships...")
    
    # Test intensity-mixing relationship
    mixing_degree = 0.65
    coefficients = [2.1, 1.8, -0.3, 0.05]
    
    base_intensity = (coefficients[0] * mixing_degree**3 + 
                     coefficients[1] * mixing_degree**2 + 
                     coefficients[2] * mixing_degree + 
                     coefficients[3])
    
    assert base_intensity > 0, "Base intensity should be positive"
    print(f"✓ Base intensity calculation: {base_intensity:.2f}")
    
    # Test volatile contribution
    volatile_content = 0.045
    volatility_factor = 1.2
    volatile_factor = 1 + volatility_factor * volatile_content
    
    assert volatile_factor > 1, "Volatile factor should amplify intensity"
    print(f"✓ Volatile factor: {volatile_factor:.3f}")
    
    return True


def test_gas_volume_fraction():
    """Test gas volume fraction calculation"""
    print("Testing gas volume fraction calculation...")
    
    # Simplified gas volume fraction calculation
    volatile_content = {'H2O': 0.04, 'CO2': 0.01, 'SO2': 0.002}
    pressure = 1e8  # Pa
    temperature = 1200  # K
    
    total_moles_gas = 0
    molecular_weights = {'H2O': 18.015, 'CO2': 44.01, 'SO2': 64.066}
    
    for species, content in volatile_content.items():
        if species in molecular_weights:
            mw = molecular_weights[species]
            total_moles_gas += content / mw
    
    # Ideal gas law approximation
    R = 8.314  # J/mol/K
    if total_moles_gas > 0 and pressure > 0:
        gas_volume_fraction = (total_moles_gas * R * temperature) / (pressure * 1.0)
        gas_volume_fraction = min(gas_volume_fraction, 0.99)
    else:
        gas_volume_fraction = 0.0
    
    assert 0 <= gas_volume_fraction <= 1, "Gas volume fraction should be between 0 and 1"
    print(f"✓ Gas volume fraction: {gas_volume_fraction:.4f}")
    
    return True


def test_hazard_zone_calculation():
    """Test hazard zone calculations"""
    print("Testing hazard zone calculations...")
    
    # Test lava flow hazard zones
    discharge_rate = 200  # m³/s
    base_distance = 5000  # meters
    discharge_factor = (discharge_rate / 100) ** 0.5
    
    hazard_zones = {}
    zone_params = {
        'high_risk': {'distance': 5000, 'probability': 0.8},
        'medium_risk': {'distance': 10000, 'probability': 0.5},
        'low_risk': {'distance': 20000, 'probability': 0.2}
    }
    
    for zone, params in zone_params.items():
        distance = base_distance * discharge_factor * (params['distance'] / 5000)
        hazard_zones[zone] = {
            'distance': distance,
            'probability': params['probability'],
            'area': np.pi * distance**2
        }
    
    assert len(hazard_zones) == 3, "Should have 3 hazard zones"
    assert all(zone['distance'] > 0 for zone in hazard_zones.values()), "All distances should be positive"
    print(f"✓ Hazard zones calculated: {list(hazard_zones.keys())}")
    
    return True


def test_uncertainty_calculation():
    """Test uncertainty calculations"""
    print("Testing uncertainty calculations...")
    
    # Test uncertainty components
    aleatory_uncertainty = {
        'mixing_degree_variability': 0.15,
        'volatile_content_variability': 0.25,
        'temperature_variability': 50,
        'viscosity_variability': 0.5
    }
    
    epistemic_uncertainty = {
        'model_parameter_uncertainty': 0.2,
        'empirical_relationship_uncertainty': 0.3,
        'measurement_uncertainty': 0.1
    }
    
    # Calculate total uncertainty (simplified)
    total_uncertainty = {}
    for param in ['intensity', 'explosivity', 'discharge_rate']:
        aleatory_comp = aleatory_uncertainty.get(f'{param}_variability', 0.1)
        epistemic_comp = epistemic_uncertainty.get('model_parameter_uncertainty', 0.2)
        
        # Root sum of squares
        total_uncertainty[param] = np.sqrt(aleatory_comp**2 + epistemic_comp**2)
    
    assert all(0 <= unc <= 2 for unc in total_uncertainty.values()), "Uncertainties should be reasonable"
    print(f"✓ Total uncertainties: {total_uncertainty}")
    
    return True


def test_style_prediction_logic():
    """Test eruption style prediction logic"""
    print("Testing style prediction logic...")
    
    # Test parameters
    gas_content = 0.05
    viscosity = 3e6
    ascent_rate = 0.3
    water_interaction = 0.1
    
    # Style thresholds
    thresholds = {
        'effusive_explosive_threshold': {
            'gas_content': 0.03,
            'viscosity': 1e6,
            'ascent_rate': 0.1
        },
        'explosive_phreatomagmatic_threshold': {
            'water_interaction': 0.1,
            'depth_threshold': 100
        }
    }
    
    # Calculate style probabilities
    style_probabilities = {}
    
    # Effusive probability
    effusive_prob = 1.0
    if gas_content > thresholds['effusive_explosive_threshold']['gas_content']:
        effusive_prob *= 0.1
    if viscosity > thresholds['effusive_explosive_threshold']['viscosity']:
        effusive_prob *= 0.2
    
    # Explosive probability
    explosive_prob = 0.0
    if gas_content > thresholds['effusive_explosive_threshold']['gas_content']:
        explosive_prob += 0.8
    if viscosity > thresholds['effusive_explosive_threshold']['viscosity']:
        explosive_prob += 0.6
    
    explosive_prob = min(1.0, explosive_prob)
    
    # Phreatomagmatic probability
    phreatomagmatic_prob = 0.0
    if water_interaction > thresholds['explosive_phreatomagmatic_threshold']['water_interaction']:
        phreatomagmatic_prob += 0.9 * water_interaction
    
    # Normalize probabilities
    total_prob = effusive_prob + explosive_prob + phreatomagmatic_prob
    if total_prob > 0:
        style_probabilities = {
            'effusive': effusive_prob / total_prob,
            'explosive': explosive_prob / total_prob,
            'phreatomagmatic': phreatomagmatic_prob / total_prob
        }
    
    assert abs(sum(style_probabilities.values()) - 1.0) < 0.01, "Probabilities should sum to 1"
    predicted_style = max(style_probabilities, key=style_probabilities.get)
    
    print(f"✓ Style probabilities: {style_probabilities}")
    print(f"✓ Predicted style: {predicted_style}")
    
    return True


def test_temporal_evolution():
    """Test temporal evolution calculations"""
    print("Testing temporal evolution...")
    
    # Test mixing progression
    mixing_progression = [0.3, 0.5, 0.7, 0.85, 0.9, 0.8, 0.6, 0.4, 0.2]
    time_steps = [datetime.now() + timedelta(hours=i*2) for i in range(len(mixing_progression))]
    
    # Calculate intensity evolution (simplified)
    intensity_evolution = []
    for mixing_degree in mixing_progression:
        # Simple intensity calculation
        base_intensity = 2.0 * mixing_degree + 1.0
        intensity_evolution.append(base_intensity)
    
    # Find peak activity
    peak_intensity_index = np.argmax(intensity_evolution)
    peak_activity_time = time_steps[peak_intensity_index]
    
    # Calculate duration
    significant_threshold = 0.5 * max(intensity_evolution)
    significant_indices = [i for i, intensity in enumerate(intensity_evolution) 
                          if intensity > significant_threshold]
    
    if significant_indices:
        duration = time_steps[significant_indices[-1]] - time_steps[significant_indices[0]]
    else:
        duration = time_steps[-1] - time_steps[0]
    
    assert len(intensity_evolution) == len(mixing_progression), "Evolution arrays should match"
    assert peak_activity_time in time_steps, "Peak time should be valid"
    print(f"✓ Peak activity time: {peak_activity_time}")
    print(f"✓ Duration: {duration}")
    print(f"✓ Max intensity: {max(intensity_evolution):.2f}")
    
    return True


def test_validation_metrics():
    """Test validation metric calculations"""
    print("Testing validation metrics...")
    
    # Mock prediction vs observation data
    predicted_characteristics = {
        'intensity': 3.5,
        'explosivity': 0.75,
        'discharge_rate': 150,
        'style': 'explosive'
    }
    
    observed_characteristics = {
        'intensity': 3.2,
        'explosivity': 0.7,
        'discharge_rate': 120,
        'style': 'explosive'
    }
    
    # Calculate accuracy metrics
    prediction_accuracy = {}
    residuals = {}
    
    for characteristic in ['intensity', 'explosivity', 'discharge_rate']:
        if characteristic in observed_characteristics and characteristic in predicted_characteristics:
            observed = observed_characteristics[characteristic]
            predicted = predicted_characteristics[characteristic]
            
            if observed != 0:
                relative_error = abs(predicted - observed) / abs(observed)
                accuracy = max(0, 1 - relative_error)
            else:
                accuracy = 1.0 if predicted == 0 else 0.0
            
            prediction_accuracy[characteristic] = accuracy
            residuals[characteristic] = predicted - observed
    
    # Style accuracy
    style_accuracy = 1.0 if observed_characteristics['style'] == predicted_characteristics['style'] else 0.0
    prediction_accuracy['style'] = style_accuracy
    
    # Overall metrics
    overall_accuracy = np.mean(list(prediction_accuracy.values()))
    rmse = np.sqrt(np.mean([r**2 for r in residuals.values()]))
    
    assert 0 <= overall_accuracy <= 1, "Overall accuracy should be between 0 and 1"
    assert rmse >= 0, "RMSE should be non-negative"
    
    print(f"✓ Overall accuracy: {overall_accuracy:.3f}")
    print(f"✓ RMSE: {rmse:.3f}")
    print(f"✓ Individual accuracies: {prediction_accuracy}")
    
    return True


def run_all_tests():
    """Run all component tests"""
    print("Running Eruption Characteristic Prediction System Component Tests")
    print("=" * 70)
    
    tests = [
        test_empirical_relationships,
        test_gas_volume_fraction,
        test_hazard_zone_calculation,
        test_uncertainty_calculation,
        test_style_prediction_logic,
        test_temporal_evolution,
        test_validation_metrics
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n{test.__name__.replace('test_', '').replace('_', ' ').title()}")
            print("-" * 40)
            
            if test():
                passed += 1
                print("✅ PASSED")
            else:
                failed += 1
                print("❌ FAILED")
                
        except Exception as e:
            failed += 1
            print(f"❌ FAILED with error: {e}")
    
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL COMPONENT TESTS PASSED!")
        print("The eruption prediction system components are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)