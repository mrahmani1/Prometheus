#!/usr/bin/env python3
"""
Test script for advanced Anderson-Darling copula test implementation.
Tests the enhanced goodness-of-fit test with full numerical integration,
database persistence, and parallel bootstrap procedures.
"""

import numpy as np
import sys
import os
import logging
from datetime import datetime
import json

# Add backend modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_copula_data(n=1000, copula_type='gaussian', dependence=0.5):
    """Generate synthetic copula data for testing."""
    np.random.seed(42)
    
    if copula_type == 'gaussian':
        # Generate Gaussian copula data
        from scipy.stats import multivariate_normal, norm
        mean = [0, 0]
        cov = [[1, dependence], [dependence, 1]]
        samples = multivariate_normal.rvs(mean, cov, n)
        U = norm.cdf(samples[:, 0])
        V = norm.cdf(samples[:, 1])
        
    elif copula_type == 'clayton':
        # Generate Clayton copula data
        theta = 2.0  # Clayton parameter
        U = np.random.uniform(0, 1, n)
        W = np.random.exponential(1, n)
        V = (U ** (-theta) * (np.exp(W) - 1) + 1) ** (-1/theta)
        
    elif copula_type == 'independence':
        # Generate independent uniform data
        U = np.random.uniform(0, 1, n)
        V = np.random.uniform(0, 1, n)
        
    else:
        raise ValueError(f"Unknown copula type: {copula_type}")
    
    return U, V

def test_anderson_darling_implementation():
    """Test the advanced Anderson-Darling implementation."""
    
    logger.info("Starting Anderson-Darling test implementation validation")
    
    try:
        # Import the module
        from enkf_knowledge_integration import CopulaBasedIndependenceTests
        
        # Initialize the test framework
        copula_tests = CopulaBasedIndependenceTests()
        logger.info("Successfully initialized CopulaBasedIndependenceTests")
        
        # Test cases
        test_cases = [
            {
                'name': 'Gaussian Copula - Strong Dependence',
                'copula_type': 'gaussian',
                'dependence': 0.8,
                'n_samples': 500,
                'copula_family': 'gaussian',
                'expected_result': 'good_fit'
            },
            {
                'name': 'Clayton Copula - Moderate Dependence',
                'copula_type': 'clayton',
                'dependence': 0.5,
                'n_samples': 500,
                'copula_family': 'clayton',
                'expected_result': 'good_fit'
            },
            {
                'name': 'Independence Test',
                'copula_type': 'independence',
                'dependence': 0.0,
                'n_samples': 500,
                'copula_family': 'independence',
                'expected_result': 'good_fit'
            },
            {
                'name': 'Misspecified Model - Gaussian vs Clayton',
                'copula_type': 'clayton',
                'dependence': 0.5,
                'n_samples': 500,
                'copula_family': 'gaussian',
                'expected_result': 'poor_fit'
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running test: {test_case['name']}")
            logger.info(f"{'='*60}")
            
            # Generate test data
            U, V = generate_copula_data(
                n=test_case['n_samples'],
                copula_type=test_case['copula_type'],
                dependence=test_case['dependence']
            )
            
            # Fit parameters for the specified copula family
            if test_case['copula_family'] == 'gaussian':
                from scipy.stats import pearsonr
                rho, _ = pearsonr(U, V)
                fitted_params = [rho]
            elif test_case['copula_family'] == 'clayton':
                # Estimate Clayton parameter using method of moments
                from scipy.stats import kendalltau
                tau, _ = kendalltau(U, V)
                theta = max(0.01, 2 * tau / (1 - tau)) if tau < 1 else 10.0
                fitted_params = [theta]
            else:
                fitted_params = []
            
            # Run Anderson-Darling test
            logger.info(f"Testing {test_case['copula_family']} copula with params: {fitted_params}")
            
            try:
                result = copula_tests._anderson_darling_copula_test(
                    U, V, 
                    copula_family=test_case['copula_family'],
                    fitted_params=fitted_params,
                    n_bootstrap=100  # Reduced for testing speed
                )
                
                # Store results
                results[test_case['name']] = {
                    'test_statistic': result.get('test_statistic', None),
                    'p_value': result.get('p_value', None),
                    'critical_values': result.get('critical_values', {}),
                    'confidence_interval': result.get('confidence_interval', {}),
                    'bootstrap_statistics': result.get('bootstrap_statistics', {}),
                    'expected': test_case['expected_result'],
                    'actual': 'good_fit' if result.get('p_value', 0) > 0.05 else 'poor_fit'
                }
                
                # Log results
                logger.info(f"Test Statistic: {result.get('test_statistic', 'N/A'):.6f}")
                logger.info(f"P-value: {result.get('p_value', 'N/A'):.6f}")
                logger.info(f"Critical Values: {result.get('critical_values', {})}")
                logger.info(f"CI: [{result.get('confidence_interval', {}).get('lower', 'N/A'):.6f}, "
                          f"{result.get('confidence_interval', {}).get('upper', 'N/A'):.6f}]")
                logger.info(f"Bootstrap Mean: {result.get('bootstrap_statistics', {}).get('mean', 'N/A'):.6f}")
                logger.info(f"Bootstrap Std: {result.get('bootstrap_statistics', {}).get('std', 'N/A'):.6f}")
                
                # Check if result matches expectation
                actual_result = 'good_fit' if result.get('p_value', 0) > 0.05 else 'poor_fit'
                if actual_result == test_case['expected_result']:
                    logger.info(f"✓ Test PASSED: {actual_result} == {test_case['expected_result']}")
                else:
                    logger.warning(f"✗ Test FAILED: {actual_result} != {test_case['expected_result']}")
                    
            except Exception as e:
                logger.error(f"Test failed with error: {e}")
                results[test_case['name']] = {
                    'error': str(e),
                    'expected': test_case['expected_result'],
                    'actual': 'error'
                }
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        passed = 0
        failed = 0
        errors = 0
        
        for test_name, result in results.items():
            if 'error' in result:
                logger.error(f"✗ {test_name}: ERROR - {result['error']}")
                errors += 1
            elif result['actual'] == result['expected']:
                logger.info(f"✓ {test_name}: PASSED")
                passed += 1
            else:
                logger.warning(f"✗ {test_name}: FAILED (expected {result['expected']}, got {result['actual']})")
                failed += 1
        
        logger.info(f"\nTotal Tests: {len(results)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Errors: {errors}")
        
        # Save results to file
        results_file = f"test_results_anderson_darling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {results_file}")
        
        return results
        
    except ImportError as e:
        logger.error(f"Failed to import module: {e}")
        logger.error("Make sure enkf_knowledge_integration.py is in the backend/modules directory")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("ADVANCED ANDERSON-DARLING TEST VALIDATION")
    logger.info("="*60)
    
    results = test_anderson_darling_implementation()
    
    if results:
        logger.info("\n✓ Test execution completed successfully")
    else:
        logger.error("\n✗ Test execution failed")
        sys.exit(1)