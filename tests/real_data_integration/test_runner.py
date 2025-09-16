"""
Comprehensive Test Runner for Real Data Integration

Orchestrates all real data integration tests including backend API tests,
frontend Playwright tests, performance tests, and generates unified reports.
"""

import asyncio
import json
import logging
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import concurrent.futures
from dataclasses import dataclass, field

# Import test suites
from .test_suite import RealDataIntegrationTestSuite
from .playwright_tests import PlaywrightRealDataTests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestSuiteResult:
    """Result from a test suite execution"""
    suite_name: str
    success: bool
    start_time: datetime
    end_time: datetime
    duration_ms: float
    test_count: int
    passed_count: int
    failed_count: int
    error_message: Optional[str] = None
    detailed_results: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveTestRunner:
    """
    Comprehensive test runner for all real data integration tests
    """
    
    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_results: List[TestSuiteResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Test configuration
        self.config = {
            'backend_tests': {
                'enabled': True,
                'timeout_seconds': 300,
                'concurrent_requests': 10
            },
            'frontend_tests': {
                'enabled': True,
                'timeout_seconds': 600,
                'headless': True
            },
            'performance_tests': {
                'enabled': True,
                'timeout_seconds': 900,
                'load_test_duration': 60
            },
            'integration_tests': {
                'enabled': True,
                'timeout_seconds': 1200
            }
        }
    
    async def run_backend_api_tests(self) -> TestSuiteResult:
        """Run backend API and data integration tests"""
        logger.info("Starting backend API tests")
        start_time = datetime.now()
        
        try:
            # Initialize test suite
            test_suite = RealDataIntegrationTestSuite()
            
            # Run comprehensive test suite
            report = await asyncio.wait_for(
                test_suite.run_comprehensive_test_suite(),
                timeout=self.config['backend_tests']['timeout_seconds']
            )
            
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Extract results
            total_tests = report['summary']['total_tests']
            successful_tests = report['summary']['successful_tests']
            failed_tests = report['summary']['failed_tests']
            
            result = TestSuiteResult(
                suite_name='backend_api_tests',
                success=report['summary']['success_rate'] > 0.9,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                test_count=total_tests,
                passed_count=successful_tests,
                failed_count=failed_tests,
                detailed_results=report
            )
            
            # Save detailed report
            report_file = self.output_dir / f"backend_api_test_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Backend API tests completed: {successful_tests}/{total_tests} passed")
            
        except asyncio.TimeoutError:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            result = TestSuiteResult(
                suite_name='backend_api_tests',
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                test_count=0,
                passed_count=0,
                failed_count=0,
                error_message="Backend API tests timed out"
            )
            
            logger.error("Backend API tests timed out")
            
        except Exception as e:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            result = TestSuiteResult(
                suite_name='backend_api_tests',
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                test_count=0,
                passed_count=0,
                failed_count=0,
                error_message=str(e)
            )
            
            logger.error(f"Backend API tests failed: {e}")
        
        return result
    
    async def run_frontend_playwright_tests(self) -> TestSuiteResult:
        """Run frontend Playwright tests"""
        logger.info("Starting frontend Playwright tests")
        start_time = datetime.now()
        
        try:
            # Initialize Playwright test runner
            playwright_tests = PlaywrightRealDataTests()
            
            # Run Playwright tests
            report = await asyncio.wait_for(
                playwright_tests.run_all_playwright_tests(
                    headless=self.config['frontend_tests']['headless']
                ),
                timeout=self.config['frontend_tests']['timeout_seconds']
            )
            
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Extract results
            total_tests = report['summary']['total_tests']
            successful_tests = report['summary']['successful_tests']
            failed_tests = report['summary']['failed_tests']
            
            result = TestSuiteResult(
                suite_name='frontend_playwright_tests',
                success=report['summary']['success_rate'] > 0.8,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                test_count=total_tests,
                passed_count=successful_tests,
                failed_count=failed_tests,
                detailed_results=report
            )
            
            # Save detailed report
            report_file = self.output_dir / f"playwright_test_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Playwright tests completed: {successful_tests}/{total_tests} passed")
            
        except asyncio.TimeoutError:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            result = TestSuiteResult(
                suite_name='frontend_playwright_tests',
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                test_count=0,
                passed_count=0,
                failed_count=0,
                error_message="Playwright tests timed out"
            )
            
            logger.error("Playwright tests timed out")
            
        except Exception as e:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            result = TestSuiteResult(
                suite_name='frontend_playwright_tests',
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                test_count=0,
                passed_count=0,
                failed_count=0,
                error_message=str(e)
            )
            
            logger.error(f"Playwright tests failed: {e}")
        
        return result
    
    async def run_performance_tests(self) -> TestSuiteResult:
        """Run performance and load tests"""
        logger.info("Starting performance tests")
        start_time = datetime.now()
        
        try:
            # Run performance tests using existing test suite
            test_suite = RealDataIntegrationTestSuite()
            await test_suite.setup_session()
            
            performance_results = []
            
            # Test each data source under load
            for source_id in test_suite.data_source_configs.keys():
                logger.info(f"Running performance test for {source_id}")
                
                result = await test_suite.test_performance_under_load(
                    source_id, 
                    concurrent_requests=self.config['performance_tests'].get('concurrent_requests', 20)
                )
                performance_results.append(result)
            
            await test_suite.cleanup_session()
            
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Analyze performance results
            total_tests = len(performance_results)
            successful_tests = sum(1 for r in performance_results if r.success)
            failed_tests = total_tests - successful_tests
            
            # Create performance report
            performance_report = {
                'timestamp': start_time.isoformat(),
                'summary': {
                    'total_tests': total_tests,
                    'successful_tests': successful_tests,
                    'failed_tests': failed_tests,
                    'success_rate': successful_tests / total_tests if total_tests > 0 else 0
                },
                'performance_results': [
                    {
                        'test_name': r.test_name,
                        'success': r.success,
                        'duration_ms': r.duration_ms,
                        'performance_metrics': r.performance_metrics,
                        'error_message': r.error_message
                    }
                    for r in performance_results
                ]
            }
            
            result = TestSuiteResult(
                suite_name='performance_tests',
                success=successful_tests / total_tests > 0.8 if total_tests > 0 else False,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                test_count=total_tests,
                passed_count=successful_tests,
                failed_count=failed_tests,
                detailed_results=performance_report
            )
            
            # Save performance report
            report_file = self.output_dir / f"performance_test_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(performance_report, f, indent=2, default=str)
            
            logger.info(f"Performance tests completed: {successful_tests}/{total_tests} passed")
            
        except Exception as e:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            result = TestSuiteResult(
                suite_name='performance_tests',
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                test_count=0,
                passed_count=0,
                failed_count=0,
                error_message=str(e)
            )
            
            logger.error(f"Performance tests failed: {e}")
        
        return result
    
    async def run_integration_tests(self) -> TestSuiteResult:
        """Run end-to-end integration tests"""
        logger.info("Starting integration tests")
        start_time = datetime.now()
        
        try:
            # Run existing integration tests using pytest
            integration_result = await self._run_pytest_integration_tests()
            
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            result = TestSuiteResult(
                suite_name='integration_tests',
                success=integration_result['success'],
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                test_count=integration_result['total'],
                passed_count=integration_result['passed'],
                failed_count=integration_result['failed'],
                detailed_results=integration_result
            )
            
            logger.info(f"Integration tests completed: {integration_result['passed']}/{integration_result['total']} passed")
            
        except Exception as e:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            result = TestSuiteResult(
                suite_name='integration_tests',
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                test_count=0,
                passed_count=0,
                failed_count=0,
                error_message=str(e)
            )
            
            logger.error(f"Integration tests failed: {e}")
        
        return result
    
    async def _run_pytest_integration_tests(self) -> Dict[str, Any]:
        """Run pytest integration tests"""
        try:
            # Run pytest on backend integration tests
            cmd = [
                sys.executable, '-m', 'pytest',
                'backend/tests/',
                '-v',
                '--tb=short',
                '--json-report',
                '--json-report-file=integration_test_results.json'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            stdout, stderr = await process.communicate()
            
            # Try to read JSON report
            try:
                with open('integration_test_results.json', 'r') as f:
                    pytest_report = json.load(f)
                
                return {
                    'success': pytest_report['summary']['failed'] == 0,
                    'total': pytest_report['summary']['total'],
                    'passed': pytest_report['summary']['passed'],
                    'failed': pytest_report['summary']['failed'],
                    'skipped': pytest_report['summary'].get('skipped', 0),
                    'stdout': stdout.decode(),
                    'stderr': stderr.decode(),
                    'detailed_report': pytest_report
                }
                
            except FileNotFoundError:
                # Fallback to parsing stdout
                output = stdout.decode()
                
                # Simple parsing of pytest output
                if 'failed' in output.lower():
                    return {
                        'success': False,
                        'total': 0,
                        'passed': 0,
                        'failed': 1,
                        'stdout': output,
                        'stderr': stderr.decode()
                    }
                else:
                    return {
                        'success': True,
                        'total': 1,
                        'passed': 1,
                        'failed': 0,
                        'stdout': output,
                        'stderr': stderr.decode()
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'total': 0,
                'passed': 0,
                'failed': 1,
                'error': str(e)
            }
    
    def run_syntax_checks(self) -> TestSuiteResult:
        """Run syntax checks on all code"""
        logger.info("Starting syntax checks")
        start_time = datetime.now()
        
        try:
            syntax_results = []
            
            # Python syntax checks
            python_result = self._run_python_syntax_check()
            syntax_results.append(python_result)
            
            # TypeScript/JavaScript syntax checks
            ts_result = self._run_typescript_syntax_check()
            syntax_results.append(ts_result)
            
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            total_checks = len(syntax_results)
            passed_checks = sum(1 for r in syntax_results if r['success'])
            failed_checks = total_checks - passed_checks
            
            result = TestSuiteResult(
                suite_name='syntax_checks',
                success=failed_checks == 0,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                test_count=total_checks,
                passed_count=passed_checks,
                failed_count=failed_checks,
                detailed_results={'syntax_results': syntax_results}
            )
            
            logger.info(f"Syntax checks completed: {passed_checks}/{total_checks} passed")
            
        except Exception as e:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            result = TestSuiteResult(
                suite_name='syntax_checks',
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                test_count=0,
                passed_count=0,
                failed_count=0,
                error_message=str(e)
            )
            
            logger.error(f"Syntax checks failed: {e}")
        
        return result
    
    def _run_python_syntax_check(self) -> Dict[str, Any]:
        """Run Python syntax checks"""
        try:
            # Use py_compile to check Python files
            import py_compile
            import glob
            
            python_files = []
            for pattern in ['backend/**/*.py', 'tests/**/*.py', 'scripts/**/*.py']:
                python_files.extend(glob.glob(pattern, recursive=True))
            
            errors = []
            for file_path in python_files:
                try:
                    py_compile.compile(file_path, doraise=True)
                except py_compile.PyCompileError as e:
                    errors.append(f"{file_path}: {str(e)}")
            
            return {
                'check_type': 'python_syntax',
                'success': len(errors) == 0,
                'files_checked': len(python_files),
                'errors': errors
            }
            
        except Exception as e:
            return {
                'check_type': 'python_syntax',
                'success': False,
                'error': str(e)
            }
    
    def _run_typescript_syntax_check(self) -> Dict[str, Any]:
        """Run TypeScript/JavaScript syntax checks"""
        try:
            # Check if TypeScript compiler is available
            result = subprocess.run(
                ['npx', 'tsc', '--version'],
                capture_output=True,
                text=True,
                cwd='frontend-react'
            )
            
            if result.returncode != 0:
                return {
                    'check_type': 'typescript_syntax',
                    'success': False,
                    'error': 'TypeScript compiler not available'
                }
            
            # Run TypeScript compilation check
            result = subprocess.run(
                ['npx', 'tsc', '--noEmit'],
                capture_output=True,
                text=True,
                cwd='frontend-react'
            )
            
            return {
                'check_type': 'typescript_syntax',
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except Exception as e:
            return {
                'check_type': 'typescript_syntax',
                'success': False,
                'error': str(e)
            }
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run all test suites comprehensively"""
        logger.info("Starting comprehensive test suite execution")
        self.start_time = datetime.now()
        
        # Run syntax checks first (synchronous)
        if self.config.get('syntax_checks', {}).get('enabled', True):
            syntax_result = self.run_syntax_checks()
            self.test_results.append(syntax_result)
        
        # Run async test suites
        async_tests = []
        
        if self.config['backend_tests']['enabled']:
            async_tests.append(self.run_backend_api_tests())
        
        if self.config['frontend_tests']['enabled']:
            async_tests.append(self.run_frontend_playwright_tests())
        
        if self.config['performance_tests']['enabled']:
            async_tests.append(self.run_performance_tests())
        
        if self.config['integration_tests']['enabled']:
            async_tests.append(self.run_integration_tests())
        
        # Run async tests concurrently
        if async_tests:
            async_results = await asyncio.gather(*async_tests, return_exceptions=True)
            
            for result in async_results:
                if isinstance(result, Exception):
                    logger.error(f"Test suite failed with exception: {result}")
                    # Create error result
                    error_result = TestSuiteResult(
                        suite_name='unknown_suite',
                        success=False,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        duration_ms=0,
                        test_count=0,
                        passed_count=0,
                        failed_count=1,
                        error_message=str(result)
                    )
                    self.test_results.append(error_result)
                else:
                    self.test_results.append(result)
        
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        # Save comprehensive report
        report_file = self.output_dir / f"comprehensive_test_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive test suite completed. Overall success rate: {report['summary']['overall_success_rate']:.2%}")
        
        return report
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.start_time or not self.end_time:
            raise ValueError("Test execution not completed")
        
        total_duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        # Calculate overall statistics
        total_tests = sum(r.test_count for r in self.test_results)
        total_passed = sum(r.passed_count for r in self.test_results)
        total_failed = sum(r.failed_count for r in self.test_results)
        
        successful_suites = sum(1 for r in self.test_results if r.success)
        total_suites = len(self.test_results)
        
        report = {
            'timestamp': self.start_time.isoformat(),
            'execution_time': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'total_duration_ms': total_duration_ms,
                'total_duration_minutes': total_duration_ms / 60000
            },
            'summary': {
                'total_test_suites': total_suites,
                'successful_test_suites': successful_suites,
                'failed_test_suites': total_suites - successful_suites,
                'suite_success_rate': successful_suites / total_suites if total_suites > 0 else 0,
                'total_individual_tests': total_tests,
                'total_passed_tests': total_passed,
                'total_failed_tests': total_failed,
                'overall_success_rate': total_passed / total_tests if total_tests > 0 else 0
            },
            'test_suite_results': [],
            'performance_summary': {},
            'recommendations': [],
            'configuration': self.config
        }
        
        # Add individual test suite results
        for result in self.test_results:
            suite_result = {
                'suite_name': result.suite_name,
                'success': result.success,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat(),
                'duration_ms': result.duration_ms,
                'test_count': result.test_count,
                'passed_count': result.passed_count,
                'failed_count': result.failed_count,
                'success_rate': result.passed_count / result.test_count if result.test_count > 0 else 0,
                'error_message': result.error_message
            }
            report['test_suite_results'].append(suite_result)
        
        # Generate performance summary
        performance_data = {}
        for result in self.test_results:
            if 'performance' in result.suite_name and result.detailed_results:
                performance_data[result.suite_name] = {
                    'duration_ms': result.duration_ms,
                    'success_rate': result.passed_count / result.test_count if result.test_count > 0 else 0
                }
        
        report['performance_summary'] = performance_data
        
        # Generate recommendations
        recommendations = []
        
        # Check for failed test suites
        for result in self.test_results:
            if not result.success:
                recommendations.append(
                    f"Fix issues in {result.suite_name}: {result.error_message or 'Multiple test failures'}"
                )
        
        # Check for performance issues
        slow_suites = [r for r in self.test_results if r.duration_ms > 300000]  # > 5 minutes
        for suite in slow_suites:
            recommendations.append(
                f"Optimize performance for {suite.suite_name} - execution time: {suite.duration_ms/1000:.1f}s"
            )
        
        # Check overall success rate
        if report['summary']['overall_success_rate'] < 0.9:
            recommendations.append(
                f"Overall test success rate {report['summary']['overall_success_rate']:.2%} is below 90% - investigate failing tests"
            )
        
        report['recommendations'] = recommendations
        
        return report


# CLI interface
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive real data integration tests')
    parser.add_argument('--backend', action='store_true', help='Run backend API tests only')
    parser.add_argument('--frontend', action='store_true', help='Run frontend Playwright tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--syntax', action='store_true', help='Run syntax checks only')
    parser.add_argument('--output-dir', default='test_reports', help='Output directory for reports')
    parser.add_argument('--headless', action='store_true', default=True, help='Run browser tests in headless mode')
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = ComprehensiveTestRunner(output_dir=args.output_dir)
    
    # Configure based on arguments
    if args.backend or args.frontend or args.performance or args.integration or args.syntax:
        # Disable all by default if specific tests are requested
        for key in runner.config:
            if isinstance(runner.config[key], dict) and 'enabled' in runner.config[key]:
                runner.config[key]['enabled'] = False
        
        # Enable requested tests
        if args.backend:
            runner.config['backend_tests']['enabled'] = True
        if args.frontend:
            runner.config['frontend_tests']['enabled'] = True
            runner.config['frontend_tests']['headless'] = args.headless
        if args.performance:
            runner.config['performance_tests']['enabled'] = True
        if args.integration:
            runner.config['integration_tests']['enabled'] = True
        if args.syntax:
            runner.config['syntax_checks'] = {'enabled': True}
    
    # Run comprehensive test suite
    try:
        report = await runner.run_comprehensive_test_suite()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST SUITE RESULTS")
        print("="*80)
        print(f"Total Duration: {report['execution_time']['total_duration_minutes']:.1f} minutes")
        print(f"Test Suites: {report['summary']['successful_test_suites']}/{report['summary']['total_test_suites']} passed")
        print(f"Individual Tests: {report['summary']['total_passed_tests']}/{report['summary']['total_individual_tests']} passed")
        print(f"Overall Success Rate: {report['summary']['overall_success_rate']:.2%}")
        
        if report['recommendations']:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print(f"\nDetailed report saved to: {runner.output_dir}")
        
        # Exit with appropriate code
        if report['summary']['overall_success_rate'] < 1.0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nTest execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())