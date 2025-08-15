#!/usr/bin/env python3
"""
Debug script to check individual bilinear DCM method outputs
"""

import sys
import os
import numpy as np

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_individual_methods():
    """Test individual bilinear DCM methods"""
    
    try:
        from supereruption_analysis import SupereruptionBackendIntegrator
        analyzer = SupereruptionBackendIntegrator()
        
        # Generate test signals
        np.random.seed(42)
        n_samples = 1000
        t = np.linspace(0, 100, n_samples)
        signal1 = np.sin(0.5 * t) + 0.3 * np.sin(1.2 * t) + 0.2 * np.random.randn(n_samples)
        signal2 = 0.8 * np.sin(2.1 * t) + 0.4 * np.sin(4.5 * t) + 0.1 * signal1 + 0.15 * np.random.randn(n_samples)
        
        print("Testing individual methods...")
        
        # Test each method individually
        methods_to_test = [
            '_state_space_bilinear_dcm',
            '_neural_dynamics_bilinear',
            '_multi_input_bilinear_model',
            '_nonlinear_interaction_decomposition',
            '_spectral_bilinear_analysis'
        ]
        
        for method_name in methods_to_test:
            try:
                print(f"\nTesting {method_name}:")
                if hasattr(analyzer, method_name):
                    method = getattr(analyzer, method_name)
                    result = method(signal1, signal2)
                    print(f"  Result type: {type(result)}")
                    print(f"  Number of keys: {len(result) if isinstance(result, dict) else 'N/A'}")
                    if isinstance(result, dict):
                        # Show first few keys to understand structure
                        keys = list(result.keys())[:5]
                        print(f"  Sample keys: {keys}")
                        # Check for fallback indicators
                        if any('fallback' in str(k).lower() or 'error' in str(k).lower() for k in result.keys()):
                            print(f"  WARNING: Fallback or error detected in results")
                else:
                    print(f"  Method {method_name} not found!")
            except Exception as e:
                print(f"  ERROR testing {method_name}: {e}")
        
        # Test main bilinear DCM method
        print(f"\nTesting main _bilinear_dcm_model:")
        try:
            result = analyzer._bilinear_dcm_model(signal1, signal2)
            print(f"  Result type: {type(result)}")
            print(f"  Number of keys: {len(result) if isinstance(result, dict) else 'N/A'}")
            if isinstance(result, dict):
                # Count keys by prefix
                prefixes = {}
                for key in result.keys():
                    prefix = key.split('_')[0] if '_' in key else 'other'
                    prefixes[prefix] = prefixes.get(prefix, 0) + 1
                print(f"  Key prefixes: {prefixes}")
        except Exception as e:
            print(f"  ERROR testing main method: {e}")
    
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_individual_methods()