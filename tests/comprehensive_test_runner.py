"""
Comprehensive Test Runner

Orchestrates all testing suites including real data integration,
causal analysis validation, dashboard performance, accessibility,
and security compliance testing.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Import all test suites
from real_data_integration.test_runner import ComprehensiveTestRunner as RealDataTestRunner
from causal_analysis_validation.test_runner import CausalAnalysisTestRunner
from dashboard_performance.performance_test_suite import DashboardPerformanceTestSuite
from dashboard_performance.accessibility_test_suite import AccessibilityTestSuite
from security_compliance.security_test_suite import SecurityComplianceTestSuite

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MasterTestRunner:
    """
    Master test runner that orchestrates all testing suites
    """
    
    def __init__(self, output_dir: str = "comprehensive_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_results = {}
        
    async def run_all_tests(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run all test suites"""
        logger.info("Starting comprehensive test execution")
        
        # Real data integration tests
        if config.get('real_data_tests', True):
            logger.info("Running real data integration tests...")
            real_data_runner = RealDataTestRunner(str(self.output_dir / "real_data"))
            self.test_results['real_data'] = await real_data_runner.run_comprehensive_test_suite()
        
        # Causal analysis validation tests
        if config.get('causal_tests', True):
            logger.info("Running causal analysis validation tests...")
            causal_runner = CausalAnalysisTestRunner(str(self.output_dir / "causal"))
            self.test_results['causal'] = await causal_runner.run_comprehensive_algorithm_comparison()
        
        # Dashboard performance tests
        if config.get('performance_tests', True):
            logger.info("Running dashboard performance tests...")
            perf_runner = DashboardPerformanceTestSuite(
                config.get('dashboard_url', 'http://localhost:3000'),
                str(self.output_dir / "performance")
            )
            self.test_results['performance'] = await perf_runner.run_comprehensive_performance_tests()
        
        # Accessibility tests
        if config.get('accessibility_tests', True):
            logger.info("Running accessibility tests...")
            access_runner = AccessibilityTestSuite(
                config.get('dashboard_url', 'http://localhost:3000'),
                str(self.output_dir / "accessibility")
            )
            self.test_results['accessibility'] = await access_runner.run_comprehensive_accessibility_tests()
        
        # Security tests
        if config.get('security_tests', True):
            logger.info("Running security tests...")
            security_runner = SecurityComplianceTestSuite(
                config.get('api_url', 'http://localhost:8000'),
                str(self.output_dir / "security")
            )
            self.test_results['security'] = await security_runner.run_comprehensive_security_tests()
        
        # Generate master report
        master_report = self._generate_master_report()
        
        # Save master report
        report_file = self.output_dir / f"master_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(master_report, f, indent=2, default=str)
        
        logger.info(f"All tests completed. Master report saved to {report_file}")
        
        return master_report
    
    def _generate_master_report(self) -> Dict[str, Any]:
        """Generate master test report"""
        
        # Calculate overall statistics
        total_success_rate = 0
        test_suite_count = 0
        
        for suite_name, results in self.test_results.items():
            if 'summary' in results and 'success_rate' in results['summary']:
                total_success_rate += results['summary']['success_rate']
                test_suite_count += 1
        
        overall_success_rate = total_success_rate / test_suite_count if test_suite_count > 0 else 0
        
        # Determine overall status
        if overall_success_rate >= 0.95:
            overall_status = 'EXCELLENT'
        elif overall_success_rate >= 0.85:
            overall_status = 'GOOD'
        elif overall_success_rate >= 0.70:
            overall_status = 'FAIR'
        else:
            overall_status = 'POOR'
        
        master_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_summary': {
                'status': overall_status,
                'success_rate': overall_success_rate,
                'test_suites_executed': test_suite_count,
                'total_duration_minutes': 0  # Would need to track actual duration
            },
            'test_suite_results': self.test_results,
            'recommendations': self._generate_master_recommendations()
        }
        
        return master_report
    
    def _generate_master_recommendations(self) -> List[str]:
        """Generate master recommendations"""
        recommendations = []
        
        # Check each test suite for critical issues
        for suite_name, results in self.test_results.items():
            if 'summary' in results:
                success_rate = results['summary'].get('success_rate', 0)
                
                if success_rate < 0.8:
                    recommendations.append(f"Address critical issues in {suite_name} testing (success rate: {success_rate:.1%})")
        
        # Security-specific recommendations
        if 'security' in self.test_results:
            security_results = self.test_results['security']
            if security_results.get('vulnerability_breakdown', {}).get('critical', 0) > 0:
                recommendations.append("URGENT: Fix critical security vulnerabilities immediately")
        
        # Performance-specific recommendations
        if 'performance' in self.test_results:
            perf_results = self.test_results['performance']
            if perf_results.get('summary', {}).get('success_rate', 1) < 0.8:
                recommendations.append("Optimize dashboard performance for better user experience")
        
        return recommendations


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Run comprehensive test suite')
    parser.add_argument('--output-dir', default='comprehensive_test_results', help='Output directory')
    parser.add_argument('--dashboard-url', default='http://localhost:3000', help='Dashboard URL')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API URL')
    parser.add_argument('--skip-real-data', action='store_true', help='Skip real data tests')
    parser.add_argument('--skip-causal', action='store_true', help='Skip causal analysis tests')
    parser.add_argument('--skip-performance', action='store_true', help='Skip performance tests')
    parser.add_argument('--skip-accessibility', action='store_true', help='Skip accessibility tests')
    parser.add_argument('--skip-security', action='store_true', help='Skip security tests')
    
    args = parser.parse_args()
    
    config = {
        'dashboard_url': args.dashboard_url,
        'api_url': args.api_url,
        'real_data_tests': not args.skip_real_data,
        'causal_tests': not args.skip_causal,
        'performance_tests': not args.skip_performance,
        'accessibility_tests': not args.skip_accessibility,
        'security_tests': not args.skip_security
    }
    
    runner = MasterTestRunner(args.output_dir)
    
    try:
        report = await runner.run_all_tests(config)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST SUITE RESULTS")
        print("="*80)
        print(f"Overall Status: {report['overall_summary']['status']}")
        print(f"Overall Success Rate: {report['overall_summary']['success_rate']:.2%}")
        print(f"Test Suites Executed: {report['overall_summary']['test_suites_executed']}")
        
        if report.get('recommendations'):
            print("\nMASTER RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print(f"\nDetailed reports saved to: {runner.output_dir}")
        
        # Exit with appropriate code
        if report['overall_summary']['success_rate'] < 0.8:
            return 1
        else:
            return 0
        
    except Exception as e:
        print(f"Comprehensive test execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))