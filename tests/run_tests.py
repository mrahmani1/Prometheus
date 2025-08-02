#!/usr/bin/env python3
"""
Comprehensive test runner for Prometheus AI platform

Provides different test execution modes and reporting capabilities.
"""

from pathlib import Path
from typing import List, Dict, Any
import argparse
import json
import os
import subprocess
import sys
import time


class TestRunner:
    """Test runner with comprehensive reporting and execution modes"""
    
    def __init__(self):
        self.test_root = Path(__file__).parent
        self.project_root = self.test_root.parent
        
    def run_unit_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run unit tests"""
        print("🧪 Running Unit Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_root / "unit"),
            "-m", "unit",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        result = self._execute_pytest(cmd)
        return result
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests"""
        print("🔗 Running Integration Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_root / "integration"),
            "-m", "integration",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        result = self._execute_pytest(cmd)
        return result
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests"""
        print("⚡ Running Performance Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_root / "performance"),
            "-m", "performance",
            "--tb=short",
            "--timeout=600"  # 10 minute timeout for performance tests
        ]
        
        if verbose:
            cmd.append("-v")
        
        result = self._execute_pytest(cmd)
        return result
    
    def run_benchmark_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run benchmark validation tests"""
        print("📊 Running Benchmark Validation Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_root / "benchmark"),
            "-m", "benchmark",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        result = self._execute_pytest(cmd)
        return result
    
    def run_all_tests(self, verbose: bool = False, parallel: bool = False) -> Dict[str, Any]:
        """Run all test suites"""
        print("🚀 Running All Test Suites...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_root),
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        result = self._execute_pytest(cmd)
        return result
    
    def run_coverage_report(self) -> Dict[str, Any]:
        """Generate comprehensive coverage report"""
        print("📈 Generating Coverage Report...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_root),
            "--cov=services",
            "--cov-report=html:tests/coverage_html",
            "--cov-report=term-missing",
            "--cov-report=json:tests/coverage.json"
        ]
        
        result = self._execute_pytest(cmd)
        
        # Parse coverage data
        coverage_file = self.test_root / "coverage.json"
        if coverage_file.exists():
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
            
            total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
            print(f"📊 Total Coverage: {total_coverage:.1f}%")
            
            # Coverage by service
            print("\n📋 Coverage by Service:")
            for filename, file_data in coverage_data.get('files', {}).items():
                if 'services/' in filename:
                    service_name = filename.split('services/')[1].split('/')[0]
                    coverage_percent = file_data.get('summary', {}).get('percent_covered', 0)
                    print(f"  {service_name}: {coverage_percent:.1f}%")
        
        return result
    
    def run_specific_test(self, test_path: str, verbose: bool = False) -> Dict[str, Any]:
        """Run a specific test file or test function"""
        print(f"🎯 Running Specific Test: {test_path}")
        
        cmd = [
            "python", "-m", "pytest",
            test_path,
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        result = self._execute_pytest(cmd)
        return result
    
    def run_failed_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Re-run only failed tests from last run"""
        print("🔄 Re-running Failed Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "--lf",  # Last failed
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        result = self._execute_pytest(cmd)
        return result
    
    def run_smoke_tests(self) -> Dict[str, Any]:
        """Run quick smoke tests to verify basic functionality"""
        print("💨 Running Smoke Tests...")
        
        # Run a subset of critical tests
        smoke_tests = [
            "tests/unit/test_helios_core_comprehensive.py::TestPINNEngineComprehensive::test_model_creation_with_complex_architecture",
            "tests/integration/test_end_to_end_workflows.py::TestDataIngestionToProcessingWorkflow::test_seismic_data_workflow",
            "tests/benchmark/test_geological_model_validation.py::TestSeismicModelValidation::test_yellowstone_seismic_pattern_recognition"
        ]
        
        cmd = [
            "python", "-m", "pytest",
            "--tb=short",
            "-x"  # Stop on first failure
        ] + smoke_tests
        
        result = self._execute_pytest(cmd)
        return result
    
    def _execute_pytest(self, cmd: List[str]) -> Dict[str, Any]:
        """Execute pytest command and capture results"""
        start_time = time.time()
        
        try:
            # Change to project root directory
            original_cwd = os.getcwd()
            os.chdir(self.project_root)
            
            # Execute pytest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Test execution timed out',
                'execution_time': time.time() - start_time,
                'command': ' '.join(cmd)
            }
        
        finally:
            os.chdir(original_cwd)
    
    def generate_test_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("=" * 80)
        report.append("PROMETHEUS AI PLATFORM - TEST EXECUTION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        total_time = 0
        
        for test_type, result in results.items():
            report.append(f"{test_type.upper()} TESTS:")
            report.append("-" * 40)
            
            if result['success']:
                status = "✅ PASSED"
                passed_tests += 1
            else:
                status = "❌ FAILED"
                failed_tests += 1
            
            total_tests += 1
            total_time += result['execution_time']
            
            report.append(f"Status: {status}")
            report.append(f"Execution Time: {result['execution_time']:.2f} seconds")
            report.append(f"Return Code: {result['return_code']}")
            
            if result['stderr']:
                report.append("Errors:")
                report.append(result['stderr'][:500] + "..." if len(result['stderr']) > 500 else result['stderr'])
            
            report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 40)
        report.append(f"Total Test Suites: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {failed_tests}")
        report.append(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        report.append(f"Total Execution Time: {total_time:.2f} seconds")
        report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description="Prometheus AI Platform Test Runner")
    
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "performance", "benchmark", "all", "coverage", "smoke", "failed"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel (for 'all' test type)"
    )
    
    parser.add_argument(
        "--specific", "-s",
        type=str,
        help="Run specific test file or function"
    )
    
    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate detailed test report"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    results = {}
    
    try:
        if args.specific:
            results['specific'] = runner.run_specific_test(args.specific, args.verbose)
        elif args.test_type == "unit":
            results['unit'] = runner.run_unit_tests(args.verbose)
        elif args.test_type == "integration":
            results['integration'] = runner.run_integration_tests(args.verbose)
        elif args.test_type == "performance":
            results['performance'] = runner.run_performance_tests(args.verbose)
        elif args.test_type == "benchmark":
            results['benchmark'] = runner.run_benchmark_tests(args.verbose)
        elif args.test_type == "coverage":
            results['coverage'] = runner.run_coverage_report()
        elif args.test_type == "smoke":
            results['smoke'] = runner.run_smoke_tests()
        elif args.test_type == "failed":
            results['failed'] = runner.run_failed_tests(args.verbose)
        elif args.test_type == "all":
            print("🚀 Running Complete Test Suite...")
            results['unit'] = runner.run_unit_tests(args.verbose)
            results['integration'] = runner.run_integration_tests(args.verbose)
            results['performance'] = runner.run_performance_tests(args.verbose)
            results['benchmark'] = runner.run_benchmark_tests(args.verbose)
        
        # Print results
        success_count = sum(1 for r in results.values() if r['success'])
        total_count = len(results)
        
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Passed: {success_count}/{total_count}")
        print(f"Success Rate: {(success_count/total_count)*100:.1f}%")
        
        for test_type, result in results.items():
            status = "✅" if result['success'] else "❌"
            print(f"{status} {test_type.capitalize()}: {result['execution_time']:.2f}s")
        
        # Generate detailed report if requested
        if args.report:
            report = runner.generate_test_report(results)
            report_file = runner.test_root / "test_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        if success_count == total_count:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print(f"\n❌ {total_count - success_count} test suite(s) failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test execution failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()