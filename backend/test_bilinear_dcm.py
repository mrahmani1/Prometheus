#!/usr/bin/env python3
"""
Comprehensive testing campaign for bilinear DCM implementation
Tests without requiring external imports beyond basic Python and numpy
"""

import sys
import os
import numpy as np
import traceback
from typing import Dict, Any

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

class BilinearDCMTestSuite:
    """Test suite for advanced bilinear DCM implementation"""
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        
    def generate_test_signals(self, n_samples: int = 1000) -> tuple:
        """Generate synthetic volcanic signals for testing"""
        np.random.seed(42)  # For reproducible results
        
        # Create volcanic-like signals with different characteristics
        t = np.linspace(0, 100, n_samples)
        
        # Signal 1: Volcanic tremor with low-frequency oscillations
        signal1 = (np.sin(0.5 * t) + 0.3 * np.sin(1.2 * t) + 
                  0.2 * np.random.randn(n_samples))
        
        # Signal 2: Eruption precursor with higher frequency components
        signal2 = (0.8 * np.sin(2.1 * t) + 0.4 * np.sin(4.5 * t) + 
                  0.1 * signal1 + 0.15 * np.random.randn(n_samples))
        
        return signal1, signal2
    
    def run_test(self, test_name: str, test_func, *args, **kwargs):
        """Run a single test and record results"""
        self.total_tests += 1
        print(f"\nRunning test: {test_name}")
        
        try:
            result = test_func(*args, **kwargs)
            if result:
                print(f"PASSED: {test_name}")
                self.passed_tests += 1
                self.test_results[test_name] = {"status": "PASSED", "details": result}
            else:
                print(f"FAILED: {test_name}")
                self.failed_tests += 1
                self.test_results[test_name] = {"status": "FAILED", "details": "Test returned False"}
        except Exception as e:
            print(f"ERROR: {test_name} - {str(e)}")
            self.failed_tests += 1
            self.test_results[test_name] = {"status": "ERROR", "details": str(e), "traceback": traceback.format_exc()}
    
    def test_bilinear_dcm_basic(self, analyzer, signal1, signal2) -> bool:
        """Test basic bilinear DCM functionality"""
        try:
            result = analyzer._bilinear_dcm_model(signal1, signal2)
            
            # Check if result is a dictionary
            if not isinstance(result, dict):
                print(f"Expected dict, got {type(result)}")
                return False
            
            # Check for required keys
            required_keys = ['ss_bilinear_strength', 'nd_bilinear_correlation', 'tv_coupling_variability']
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                print(f"Missing required keys: {missing_keys}")
                return False
            
            # Check if values are numeric
            for key, value in result.items():
                if not isinstance(value, (int, float, np.number)):
                    print(f"Non-numeric value for {key}: {value} ({type(value)})")
                    return False
                if np.isnan(value) or np.isinf(value):
                    print(f"Invalid value for {key}: {value}")
                    return False
            
            print(f"Bilinear DCM returned {len(result)} metrics")
            return True
            
        except Exception as e:
            print(f"Bilinear DCM test failed: {e}")
            return False
    
    def test_state_space_bilinear(self, analyzer, signal1, signal2) -> bool:
        """Test state-space bilinear DCM component"""
        try:
            result = analyzer._state_space_bilinear_dcm(signal1, signal2)
            
            # Check for state-space specific metrics
            ss_keys = [k for k in result.keys() if k.startswith('ss_')]
            if len(ss_keys) < 5:
                print(f"Expected at least 5 state-space metrics, got {len(ss_keys)}")
                return False
            
            # Validate specific metrics
            if 'ss_bilinear_strength' in result:
                strength = result['ss_bilinear_strength']
                if not (0 <= strength <= 1):
                    print(f"Bilinear strength out of range: {strength}")
                    return False
            
            print(f"State-space bilinear DCM returned {len(ss_keys)} SS metrics")
            return True
            
        except Exception as e:
            print(f"State-space bilinear test failed: {e}")
            return False
    
    def test_neural_dynamics_bilinear(self, analyzer, signal1, signal2) -> bool:
        """Test neural dynamics bilinear component"""
        try:
            result = analyzer._neural_dynamics_bilinear(signal1, signal2)
            
            # Check for neural dynamics specific metrics
            nd_keys = [k for k in result.keys() if k.startswith('nd_')]
            if len(nd_keys) < 8:
                print(f"Expected at least 8 neural dynamics metrics, got {len(nd_keys)}")
                return False
            
            # Validate coupling parameters
            if 'nd_bilinear_coupling_b1' in result and 'nd_bilinear_coupling_b2' in result:
                b1 = result['nd_bilinear_coupling_b1']
                b2 = result['nd_bilinear_coupling_b2']
                if b1 != 0.1 or b2 != 0.15:  # These are hardcoded values
                    print(f"Unexpected coupling parameters: B1={b1}, B2={b2}")
                    return False
            
            print(f"Neural dynamics bilinear returned {len(nd_keys)} ND metrics")
            return True
            
        except Exception as e:
            print(f"Neural dynamics bilinear test failed: {e}")
            return False
    
    def test_time_varying_bilinear(self, analyzer, signal1, signal2) -> bool:
        """Test time-varying bilinear coupling component"""
        try:
            result = analyzer._time_varying_bilinear_coupling(signal1, signal2)
            
            # Check for time-varying specific metrics
            tv_keys = [k for k in result.keys() if k.startswith('tv_')]
            if len(tv_keys) < 10:
                print(f"Expected at least 10 time-varying metrics, got {len(tv_keys)}")
                return False
            
            # Validate temporal metrics
            if 'tv_coupling_variability' in result:
                variability = result['tv_coupling_variability']
                if variability < 0:
                    print(f"Negative coupling variability: {variability}")
                    return False
            
            print(f"Time-varying bilinear returned {len(tv_keys)} TV metrics")
            return True
            
        except Exception as e:
            print(f"Time-varying bilinear test failed: {e}")
            return False
    
    def test_bayesian_bilinear(self, analyzer, signal1, signal2) -> bool:
        """Test Bayesian bilinear estimation component"""
        try:
            result = analyzer._bayesian_bilinear_estimation(signal1, signal2)
            
            # Check for Bayesian specific metrics
            bay_keys = [k for k in result.keys() if k.startswith('bay_')]
            if len(bay_keys) < 10:
                print(f"Expected at least 10 Bayesian metrics, got {len(bay_keys)}")
                return False
            
            # Validate probability metrics
            if 'bay_linear_probability' in result and 'bay_bilinear_probability' in result:
                p_linear = result['bay_linear_probability']
                p_bilinear = result['bay_bilinear_probability']
                total_prob = p_linear + p_bilinear
                if not (0.95 <= total_prob <= 1.05):  # Allow for small numerical errors
                    print(f"Probabilities don't sum to 1: {total_prob}")
                    return False
            
            print(f"Bayesian bilinear returned {len(bay_keys)} Bayesian metrics")
            return True
            
        except Exception as e:
            print(f"Bayesian bilinear test failed: {e}")
            return False
    
    def test_multi_input_bilinear(self, analyzer, signal1, signal2) -> bool:
        """Test multi-input bilinear model component"""
        try:
            result = analyzer._multi_input_bilinear_model(signal1, signal2)
            
            # Check for multi-input specific metrics
            mi_keys = [k for k in result.keys() if k.startswith('mi_')]
            if len(mi_keys) < 10:
                print(f"Expected at least 10 multi-input metrics, got {len(mi_keys)}")
                return False
            
            # Validate input influence metrics
            if 'mi_input_influence1' in result and 'mi_input_influence2' in result:
                inf1 = result['mi_input_influence1']
                inf2 = result['mi_input_influence2']
                if not (0 <= inf1 <= 1) or not (0 <= inf2 <= 1):
                    print(f"Input influences out of range: {inf1}, {inf2}")
                    return False
            
            print(f"Multi-input bilinear returned {len(mi_keys)} MI metrics")
            return True
            
        except Exception as e:
            print(f"Multi-input bilinear test failed: {e}")
            return False
    
    def test_nonlinear_interaction(self, analyzer, signal1, signal2) -> bool:
        """Test nonlinear interaction decomposition component"""
        try:
            result = analyzer._nonlinear_interaction_decomposition(signal1, signal2)
            
            # Check for nonlinear interaction specific metrics
            ni_keys = [k for k in result.keys() if k.startswith('ni_')]
            if len(ni_keys) < 12:
                print(f"Expected at least 12 nonlinear interaction metrics, got {len(ni_keys)}")
                return False
            
            # Validate contribution metrics sum to approximately 1
            contribution_keys = ['ni_linear_contribution', 'ni_bilinear_contribution', 
                               'ni_quadratic_contribution', 'ni_cubic_contribution']
            if all(key in result for key in contribution_keys):
                total_contribution = sum(result[key] for key in contribution_keys)
                if not (0.95 <= total_contribution <= 1.05):
                    print(f"Contributions don't sum to 1: {total_contribution}")
                    return False
            
            print(f"Nonlinear interaction returned {len(ni_keys)} NI metrics")
            return True
            
        except Exception as e:
            print(f"Nonlinear interaction test failed: {e}")
            return False
    
    def test_spectral_bilinear(self, analyzer, signal1, signal2) -> bool:
        """Test spectral bilinear analysis component"""
        try:
            result = analyzer._spectral_bilinear_analysis(signal1, signal2)
            
            # Check for spectral bilinear specific metrics
            sb_keys = [k for k in result.keys() if k.startswith('sb_')]
            if len(sb_keys) < 12:
                print(f"Expected at least 12 spectral bilinear metrics, got {len(sb_keys)}")
                return False
            
            # Validate volcanic frequency band powers
            volcanic_bands = ['sb_tremor_power_bil', 'sb_precursor_power_bil', 'sb_eruption_power_bil']
            for band in volcanic_bands:
                if band in result and result[band] < 0:
                    print(f"Negative power in {band}: {result[band]}")
                    return False
            
            print(f"Spectral bilinear returned {len(sb_keys)} SB metrics")
            return True
            
        except Exception as e:
            print(f"Spectral bilinear test failed: {e}")
            return False
    
    def test_validation_component(self, analyzer, signal1, signal2) -> bool:
        """Test statistical validation component"""
        try:
            # First get some results to validate
            main_results = analyzer._bilinear_dcm_model(signal1, signal2)
            result = analyzer._bilinear_dcm_validation(signal1, signal2, main_results)
            
            # Check for validation specific metrics
            val_keys = [k for k in result.keys() if k.startswith('val_')]
            if len(val_keys) < 10:
                print(f"Expected at least 10 validation metrics, got {len(val_keys)}")
                return False
            
            # Validate confidence intervals
            if 'val_ci_lower' in result and 'val_ci_upper' in result:
                ci_lower = result['val_ci_lower']
                ci_upper = result['val_ci_upper']
                if ci_lower > ci_upper:
                    print(f"Invalid confidence interval: [{ci_lower}, {ci_upper}]")
                    return False
            
            print(f"Statistical validation returned {len(val_keys)} validation metrics")
            return True
            
        except Exception as e:
            print(f"Statistical validation test failed: {e}")
            return False
    
    def test_fallback_mechanism(self, analyzer, signal1, signal2) -> bool:
        """Test fallback mechanism"""
        try:
            result = analyzer._fallback_bilinear_dcm(signal1, signal2)
            
            # Check basic fallback functionality
            if not isinstance(result, dict):
                print(f"Fallback should return dict, got {type(result)}")
                return False
            
            required_fallback_keys = ['bilinear_dcm_strength', 'bilinear_interaction_term', 
                                    'bilinear_nonlinearity_index', 'fallback_used']
            missing_keys = [key for key in required_fallback_keys if key not in result]
            if missing_keys:
                print(f"Missing fallback keys: {missing_keys}")
                return False
            
            if result['fallback_used'] != 1.0:
                print(f"Fallback flag should be 1.0, got {result['fallback_used']}")
                return False
            
            print(f"Fallback mechanism works correctly")
            return True
            
        except Exception as e:
            print(f"Fallback mechanism test failed: {e}")
            return False
    
    def test_edge_cases(self, analyzer) -> bool:
        """Test edge cases and error handling"""
        try:
            # Test with very short signals
            short_signal1 = np.array([1.0, 2.0, 3.0])
            short_signal2 = np.array([2.0, 3.0, 4.0])
            result = analyzer._bilinear_dcm_model(short_signal1, short_signal2)
            
            if not isinstance(result, dict):
                print("Short signal test failed")
                return False
            
            # Test with zero signals
            zero_signal1 = np.zeros(100)
            zero_signal2 = np.zeros(100)
            result = analyzer._bilinear_dcm_model(zero_signal1, zero_signal2)
            
            if not isinstance(result, dict):
                print("Zero signal test failed")
                return False
            
            # Test with constant signals
            const_signal1 = np.ones(100) * 5.0
            const_signal2 = np.ones(100) * 3.0
            result = analyzer._bilinear_dcm_model(const_signal1, const_signal2)
            
            if not isinstance(result, dict):
                print("Constant signal test failed")
                return False
            
            print("Edge cases handled correctly")
            return True
            
        except Exception as e:
            print(f"Edge cases test failed: {e}")
            return False
    
    def run_comprehensive_tests(self):
        """Run the complete test suite"""
        print("="*60)
        print("BILINEAR DCM COMPREHENSIVE TESTING CAMPAIGN")
        print("="*60)
        
        try:
            # Import the analyzer
            from supereruption_analysis import SupereruptionBackendIntegrator
            analyzer = SupereruptionBackendIntegrator()
            
            # Generate test signals
            signal1, signal2 = self.generate_test_signals()
            print(f"Generated test signals: {len(signal1)} samples")
            
            # Run all tests
            self.run_test("Basic Bilinear DCM", self.test_bilinear_dcm_basic, analyzer, signal1, signal2)
            self.run_test("State-Space Bilinear", self.test_state_space_bilinear, analyzer, signal1, signal2)
            self.run_test("Neural Dynamics Bilinear", self.test_neural_dynamics_bilinear, analyzer, signal1, signal2)
            self.run_test("Time-Varying Bilinear", self.test_time_varying_bilinear, analyzer, signal1, signal2)
            self.run_test("Bayesian Bilinear", self.test_bayesian_bilinear, analyzer, signal1, signal2)
            self.run_test("Multi-Input Bilinear", self.test_multi_input_bilinear, analyzer, signal1, signal2)
            self.run_test("Nonlinear Interaction", self.test_nonlinear_interaction, analyzer, signal1, signal2)
            self.run_test("Spectral Bilinear", self.test_spectral_bilinear, analyzer, signal1, signal2)
            self.run_test("Statistical Validation", self.test_validation_component, analyzer, signal1, signal2)
            self.run_test("Fallback Mechanism", self.test_fallback_mechanism, analyzer, signal1, signal2)
            self.run_test("Edge Cases", self.test_edge_cases, analyzer)
            
        except ImportError as e:
            print(f"CRITICAL ERROR: Could not import SupereruptionBackendIntegrator: {e}")
            self.failed_tests = self.total_tests = 1
            return
        except Exception as e:
            print(f"CRITICAL ERROR during testing: {e}")
            traceback.print_exc()
            return
        
        # Print summary
        print("\n" + "="*60)
        print("TEST CAMPAIGN SUMMARY")
        print("="*60)
        print(f"Total Tests Run: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%" if self.total_tests > 0 else "N/A")
        
        if self.failed_tests > 0:
            print("\nFAILED TESTS:")
            for test_name, result in self.test_results.items():
                if result["status"] != "PASSED":
                    print(f"  {test_name}: {result['details']}")
        else:
            print("\nALL TESTS PASSED!")
        
        return self.failed_tests == 0

def main():
    """Main test execution function"""
    test_suite = BilinearDCMTestSuite()
    success = test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()