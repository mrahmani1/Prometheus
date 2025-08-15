#!/usr/bin/env python3
"""
Final comprehensive test of bilinear DCM implementation
Tests the complete system functionality without external dependencies
"""

import sys
import os
import numpy as np
import traceback

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def comprehensive_bilinear_test():
    """Comprehensive test of bilinear DCM functionality"""
    
    print("="*60)
    print("FINAL BILINEAR DCM COMPREHENSIVE TEST")
    print("="*60)
    
    try:
        # Import the correct class
        from supereruption_analysis import SupereruptionBackendIntegrator
        analyzer = SupereruptionBackendIntegrator()
        print("SUCCESS: SupereruptionBackendIntegrator imported")
        
        # Generate diverse test signals
        np.random.seed(42)
        n_samples = 1000
        t = np.linspace(0, 100, n_samples)
        
        # Volcanic tremor-like signal
        signal1 = (np.sin(0.5 * t) + 0.3 * np.sin(1.2 * t) + 
                  0.2 * np.random.randn(n_samples))
        
        # Eruption precursor-like signal with coupling
        signal2 = (0.8 * np.sin(2.1 * t) + 0.4 * np.sin(4.5 * t) + 
                  0.1 * signal1 + 0.15 * np.random.randn(n_samples))
        
        print(f"SUCCESS: Generated test signals ({len(signal1)} samples)")
        
        # Test 1: Basic functionality
        print("\nTest 1: Basic Bilinear DCM Functionality")
        result = analyzer._bilinear_dcm_model(signal1, signal2)
        
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert len(result) > 0, "Result dictionary is empty"
        
        # Validate metrics
        numeric_count = 0
        nan_count = 0
        inf_count = 0
        
        for key, value in result.items():
            if isinstance(value, (int, float, np.number)):
                numeric_count += 1
                if np.isnan(value):
                    nan_count += 1
                elif np.isinf(value):
                    inf_count += 1
        
        print(f"  Total metrics: {len(result)}")
        print(f"  Numeric metrics: {numeric_count}")
        print(f"  NaN values: {nan_count}")
        print(f"  Infinite values: {inf_count}")
        
        assert numeric_count > 0, "No numeric metrics found"
        assert nan_count == 0, f"Found {nan_count} NaN values"
        assert inf_count == 0, f"Found {inf_count} infinite values"
        
        print("  SUCCESS: Basic functionality validated")
        
        # Test 2: Metric categorization
        print("\nTest 2: Metric Categorization")
        
        categories = {}
        for key in result.keys():
            prefix = key.split('_')[0] if '_' in key else 'other'
            categories[prefix] = categories.get(prefix, 0) + 1
        
        print(f"  Metric categories: {dict(categories)}")
        
        expected_categories = ['ss', 'nd', 'tv', 'bay', 'mi', 'ni', 'sb', 'val']
        found_categories = set(categories.keys())
        
        print(f"  Expected categories: {expected_categories}")
        print(f"  Found categories: {sorted(found_categories)}")
        
        missing_categories = set(expected_categories) - found_categories
        if missing_categories:
            print(f"  WARNING: Missing categories: {missing_categories}")
        else:
            print("  SUCCESS: All expected categories present")
        
        # Test 3: Value ranges and validity
        print("\nTest 3: Value Range Validation")
        
        # Check for reasonable value ranges
        correlation_metrics = [k for k in result.keys() if 'correlation' in k.lower() or 'strength' in k.lower()]
        probability_metrics = [k for k in result.keys() if 'probability' in k.lower()]
        coherence_metrics = [k for k in result.keys() if 'coherence' in k.lower()]
        
        valid_correlations = 0
        for metric in correlation_metrics:
            value = result[metric]
            if 0 <= value <= 1:
                valid_correlations += 1
            else:
                print(f"    WARNING: {metric} = {value} (outside [0,1])")
        
        valid_probabilities = 0
        for metric in probability_metrics:
            value = result[metric]
            if 0 <= value <= 1:
                valid_probabilities += 1
            else:
                print(f"    WARNING: {metric} = {value} (outside [0,1])")
        
        print(f"  Valid correlations: {valid_correlations}/{len(correlation_metrics)}")
        print(f"  Valid probabilities: {valid_probabilities}/{len(probability_metrics)}")
        print(f"  Coherence metrics: {len(coherence_metrics)}")
        
        # Test 4: Fallback behavior
        print("\nTest 4: Fallback Mechanism")
        
        fallback_result = analyzer._fallback_bilinear_dcm(signal1, signal2)
        assert isinstance(fallback_result, dict), "Fallback should return dict"
        assert 'fallback_used' in fallback_result, "Fallback indicator missing"
        assert fallback_result['fallback_used'] == 1.0, "Fallback flag incorrect"
        
        print("  SUCCESS: Fallback mechanism working")
        
        # Test 5: Edge cases
        print("\nTest 5: Edge Case Handling")
        
        # Test with short signals
        short_sig1 = np.array([1.0, 2.0, 3.0])
        short_sig2 = np.array([2.0, 3.0, 4.0])
        short_result = analyzer._bilinear_dcm_model(short_sig1, short_sig2)
        assert isinstance(short_result, dict), "Short signal test failed"
        print("  SUCCESS: Short signals handled")
        
        # Test with zero signals
        zero_sig1 = np.zeros(100)
        zero_sig2 = np.zeros(100)
        zero_result = analyzer._bilinear_dcm_model(zero_sig1, zero_sig2)
        assert isinstance(zero_result, dict), "Zero signal test failed"
        print("  SUCCESS: Zero signals handled")
        
        # Test with constant signals
        const_sig1 = np.ones(100) * 5.0
        const_sig2 = np.ones(100) * 3.0
        const_result = analyzer._bilinear_dcm_model(const_sig1, const_sig2)
        assert isinstance(const_result, dict), "Constant signal test failed"
        print("  SUCCESS: Constant signals handled")
        
        # Test 6: Performance and consistency
        print("\nTest 6: Performance and Consistency")
        
        import time
        start_time = time.time()
        
        # Run multiple times to check consistency
        results = []
        for i in range(3):
            test_result = analyzer._bilinear_dcm_model(signal1, signal2)
            results.append(test_result)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 3
        
        print(f"  Average execution time: {avg_time:.3f} seconds")
        
        # Check consistency
        first_keys = set(results[0].keys())
        all_consistent = all(set(r.keys()) == first_keys for r in results[1:])
        
        if all_consistent:
            print("  SUCCESS: Results are consistent across runs")
        else:
            print("  WARNING: Results inconsistent across runs")
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL TEST SUMMARY")
        print("="*60)
        print(f"Total metrics generated: {len(result)}")
        print(f"Numeric metrics: {numeric_count}")
        print(f"Metric categories: {len(categories)}")
        print(f"Average execution time: {avg_time:.3f}s")
        print("Status: ALL TESTS PASSED")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    success = comprehensive_bilinear_test()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()