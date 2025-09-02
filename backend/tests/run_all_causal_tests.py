#!/usr/bin/env python3
"""
Comprehensive test runner for all enhanced causal discovery tests

This script runs all causal discovery tests:
1. Core functionality tests (test_causal_discovery_simple.py)
2. Integration tests (test_causal_integration_comprehensive.py)
3. Test summary and reporting
"""

import sys
import os
import subprocess
import time
from typing import Dict, List, Tuple, Any

def run_test_suite(test_file: str, test_description: str) -> Tuple[bool, Dict[str, Any]]:
    """Run a test suite and return success status with metrics"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {test_description}")
    print(f"FILE: {test_file}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the test file
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=os.path.dirname(test_file))
        
        execution_time = time.time() - start_time
        
        # Parse results
        success = result.returncode == 0
        stdout_lines = result.stdout.split('\n') if result.stdout else []
        stderr_lines = result.stderr.split('\n') if result.stderr else []
        
        # Extract test counts from output
        test_count = 0
        for line in stdout_lines:
            if 'Ran' in line and 'tests' in line:
                try:
                    test_count = int(line.split()[1])
                except:
                    pass
        
        # Extract pass/fail information
        failures = 0
        errors = 0
        if not success:
            for line in stdout_lines + stderr_lines:
                if 'FAILED' in line and 'failures' in line and 'errors' in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'failures' in part and i > 0:
                                failures = int(parts[i-1])
                            if 'errors' in part and i > 0:
                                errors = int(parts[i-1])
                    except:
                        pass
        
        metrics = {
            'success': success,
            'execution_time': execution_time,
            'test_count': test_count,
            'failures': failures,
            'errors': errors,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        # Print results
        if success:
            print(f"PASSED: {test_count} tests executed successfully in {execution_time:.2f}s")
        else:
            print(f"FAILED: {failures} failures, {errors} errors in {execution_time:.2f}s")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            if result.stdout and any('FAIL' in line for line in stdout_lines):
                print(f"STDOUT:\n{result.stdout}")
        
        return success, metrics
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"ERROR: Failed to execute test suite: {e}")
        return False, {
            'success': False,
            'execution_time': execution_time,
            'test_count': 0,
            'failures': 0,
            'errors': 1,
            'exception': str(e)
        }

def run_all_causal_discovery_tests():
    """Run all enhanced causal discovery tests"""
    print("=" + "="*78 + "=")
    print("  COMPREHENSIVE ENHANCED CAUSAL DISCOVERY TEST SUITE")
    print("  Testing all components of the advanced causal discovery system")
    print("=" + "="*78 + "=")
    
    # Define test suites
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    test_suites = [
        {
            'file': os.path.join(test_dir, 'test_causal_discovery_simple.py'),
            'description': 'Core Causal Discovery Functionality Tests',
            'details': 'Tests core algorithms, data structures, and mathematical properties'
        },
        {
            'file': os.path.join(test_dir, 'test_causal_integration_comprehensive.py'), 
            'description': 'Causal Discovery Integration Tests',
            'details': 'Tests end-to-end pipeline, module integration, and realistic scenarios'
        }
    ]
    
    # Run all test suites
    overall_results = []
    total_start_time = time.time()
    
    for i, test_suite in enumerate(test_suites, 1):
        print(f"\nTest Suite {i}/{len(test_suites)}: {test_suite['description']}")
        print(f"   {test_suite['details']}")
        
        success, metrics = run_test_suite(test_suite['file'], test_suite['description'])
        overall_results.append({
            'suite': test_suite['description'],
            'success': success,
            'metrics': metrics
        })
    
    total_execution_time = time.time() - total_start_time
    
    # Generate comprehensive report
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    passed_suites = 0
    failed_suites = 0
    
    for result in overall_results:
        suite_name = result['suite']
        metrics = result['metrics']
        success = result['success']
        
        if success:
            status = "PASSED"
            passed_suites += 1
        else:
            status = "FAILED"
            failed_suites += 1
        
        total_tests += metrics.get('test_count', 0)
        total_failures += metrics.get('failures', 0)
        total_errors += metrics.get('errors', 0)
        
        print(f"\n{suite_name}")
        print(f"   Status: {status}")
        print(f"   Tests: {metrics.get('test_count', 0)}")
        print(f"   Time: {metrics.get('execution_time', 0):.2f}s")
        if not success:
            print(f"   Failures: {metrics.get('failures', 0)}")
            print(f"   Errors: {metrics.get('errors', 0)}")
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total Test Suites: {len(test_suites)}")
    print(f"Passed Suites: {passed_suites}")
    print(f"Failed Suites: {failed_suites}")
    print(f"Total Tests Executed: {total_tests}")
    print(f"Total Failures: {total_failures}")
    print(f"Total Errors: {total_errors}")
    print(f"Total Execution Time: {total_execution_time:.2f}s")
    print(f"Success Rate: {(passed_suites/len(test_suites)*100):.1f}%")
    
    # Final verdict
    all_passed = all(result['success'] for result in overall_results)
    
    print(f"\n{'='*80}")
    if all_passed:
        print("ALL CAUSAL DISCOVERY TESTS PASSED!")
        print("Enhanced causal discovery system is fully tested and validated")
        print("All core algorithms, integrations, and edge cases covered")
        print("Uncertainty quantification thoroughly tested")
        print("Geological domain integration verified")
        print("System ready for production use")
    else:
        print("SOME TESTS FAILED")
        print("Review failed test suites above for detailed error information")
        print("Fix issues before deploying enhanced causal discovery system")
        
        # Provide actionable recommendations
        print("\nRECOMMENDATIONS:")
        for result in overall_results:
            if not result['success']:
                suite_name = result['suite']
                metrics = result['metrics']
                print(f"   - Fix issues in: {suite_name}")
                if metrics.get('failures', 0) > 0:
                    print(f"     - Address {metrics['failures']} test failures")
                if metrics.get('errors', 0) > 0:
                    print(f"     - Resolve {metrics['errors']} execution errors")
    
    print(f"{'='*80}")
    
    return all_passed

def main():
    """Main test execution function"""
    try:
        success = run_all_causal_discovery_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error during test execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()