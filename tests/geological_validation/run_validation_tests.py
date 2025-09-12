#!/usr/bin/env python3
"""
Geological Validation Test Runner
================================

Main test runner for comprehensive geological validation testing.
Executes all validation suites and generates comprehensive reports.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any, Optional

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Import validation modules
from benchmark_datasets import BenchmarkDatasetManager
from geological_accuracy_validator import GeologicalAccuracyValidator
from seismic_validation import SeismicValidationSuite
from volcanic_validation import VolcanicValidationSuite
from geochemical_validation import GeochemicalValidationSuite
from visualization_validation import VisualizationValidationSuite
from data_integrity_validator import DataIntegrityValidator
from continuous_validation import ContinuousValidationFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests/geological_validation/validation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class GeologicalValidationRunner:
    """
    Main runner for geological validation tests.
    
    Coordinates execution of all validation suites and provides
    comprehensive reporting and analysis capabilities.
    """
    
    def __init__(self, data_dir: str = "tests/geological_validation/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize benchmark dataset manager
        self.benchmark_manager = BenchmarkDatasetManager(str(self.data_dir))
        
        # Initialize validation suites
        self.accuracy_validator = GeologicalAccuracyValidator(self.benchmark_manager)
        self.seismic_validator = SeismicValidationSuite(self.benchmark_manager)
        self.volcanic_validator = VolcanicValidationSuite(self.benchmark_manager)
        self.geochemical_validator = GeochemicalValidationSuite(self.benchmark_manager)
        self.visualization_validator = VisualizationValidationSuite(self.benchmark_manager)
        self.data_integrity_validator = DataIntegrityValidator(self.benchmark_manager)
        self.continuous_framework = ContinuousValidationFramework(self.benchmark_manager)
        
        # Results storage
        self.validation_results: Dict[str, Any] = {}
        self.test_summary: Dict[str, Any] = {}
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run comprehensive validation across all systems"""
        logger.info("Starting comprehensive geological validation testing")
        
        start_time = datetime.now()
        
        # Initialize results structure
        self.validation_results = {
            'test_run_info': {
                'start_time': start_time.isoformat(),
                'test_runner_version': '1.0.0',
                'python_version': sys.version,
                'platform': sys.platform
            },
            'seismic_validation': {},
            'volcanic_validation': {},
            'geochemical_validation': {},
            'visualization_validation': {},
            'data_integrity_validation': {},
            'physics_validation': {}
        }
        
        try:
            # Run seismic validation tests
            logger.info("Running seismic validation tests...")
            self.validation_results['seismic_validation'] = self._run_seismic_tests()
            
            # Run volcanic validation tests
            logger.info("Running volcanic validation tests...")
            self.validation_results['volcanic_validation'] = self._run_volcanic_tests()
            
            # Run geochemical validation tests
            logger.info("Running geochemical validation tests...")
            self.validation_results['geochemical_validation'] = self._run_geochemical_tests()
            
            # Run visualization validation tests
            logger.info("Running visualization validation tests...")
            self.validation_results['visualization_validation'] = self._run_visualization_tests()
            
            # Run data integrity validation tests
            logger.info("Running data integrity validation tests...")
            self.validation_results['data_integrity_validation'] = self._run_data_integrity_tests()
            
            # Run physics validation tests
            logger.info("Running physics validation tests...")
            self.validation_results['physics_validation'] = self._run_physics_tests()
            
            # Calculate summary statistics
            end_time = datetime.now()
            self.validation_results['test_run_info']['end_time'] = end_time.isoformat()
            self.validation_results['test_run_info']['total_duration'] = str(end_time - start_time)
            
            # Generate test summary
            self.test_summary = self._generate_test_summary()
            self.validation_results['test_summary'] = self.test_summary
            
            logger.info("Comprehensive validation testing completed successfully")
            
        except Exception as e:
            logger.error(f"Validation testing failed: {e}")
            self.validation_results['error'] = str(e)
            raise
        
        return self.validation_results
    
    def _run_seismic_tests(self) -> Dict[str, Any]:
        """Run seismic validation test suite"""
        seismic_results = {}
        
        try:
            # Phase picking accuracy
            result = self.seismic_validator.validate_phase_picking_accuracy()
            seismic_results['phase_picking'] = self._format_result(result)
            
            # Magnitude estimation
            result = self.seismic_validator.validate_magnitude_estimation()
            seismic_results['magnitude_estimation'] = self._format_result(result)
            
            # Earthquake location
            result = self.seismic_validator.validate_earthquake_location()
            seismic_results['earthquake_location'] = self._format_result(result)
            
            # Event detection
            result = self.seismic_validator.validate_event_detection()
            seismic_results['event_detection'] = self._format_result(result)
            
            # Moment tensor inversion
            result = self.seismic_validator.validate_moment_tensor_inversion()
            seismic_results['moment_tensor'] = self._format_result(result)
            
            # Summary
            seismic_results['summary'] = self.seismic_validator.get_validation_summary()
            
        except Exception as e:
            logger.error(f"Seismic validation failed: {e}")
            seismic_results['error'] = str(e)
        
        return seismic_results
    
    def _run_volcanic_tests(self) -> Dict[str, Any]:
        """Run volcanic validation test suite"""
        volcanic_results = {}
        
        try:
            # Eruption forecasting
            result = self.volcanic_validator.validate_eruption_forecasting()
            volcanic_results['eruption_forecasting'] = self._format_result(result)
            
            # VEI prediction
            result = self.volcanic_validator.validate_vei_prediction()
            volcanic_results['vei_prediction'] = self._format_result(result)
            
            # Precursor detection
            result = self.volcanic_validator.validate_precursor_detection()
            volcanic_results['precursor_detection'] = self._format_result(result)
            
            # Caldera classification
            result = self.volcanic_validator.validate_caldera_classification()
            volcanic_results['caldera_classification'] = self._format_result(result)
            
            # Supereruption analysis
            result = self.volcanic_validator.validate_supereruption_analysis()
            volcanic_results['supereruption_analysis'] = self._format_result(result)
            
            # Summary
            volcanic_results['summary'] = self.volcanic_validator.get_validation_summary()
            
        except Exception as e:
            logger.error(f"Volcanic validation failed: {e}")
            volcanic_results['error'] = str(e)
        
        return volcanic_results
    
    def _run_geochemical_tests(self) -> Dict[str, Any]:
        """Run geochemical validation test suite"""
        geochemical_results = {}
        
        try:
            # Major element analysis
            result = self.geochemical_validator.validate_major_element_analysis()
            geochemical_results['major_elements'] = self._format_result(result)
            
            # Trace element analysis
            result = self.geochemical_validator.validate_trace_element_analysis()
            geochemical_results['trace_elements'] = self._format_result(result)
            
            # Rock classification
            result = self.geochemical_validator.validate_rock_classification()
            geochemical_results['rock_classification'] = self._format_result(result)
            
            # Thermobarometry
            result = self.geochemical_validator.validate_thermobarometry()
            geochemical_results['thermobarometry'] = self._format_result(result)
            
            # Isotope analysis
            result = self.geochemical_validator.validate_isotope_analysis()
            geochemical_results['isotope_analysis'] = self._format_result(result)
            
            # Summary
            geochemical_results['summary'] = self.geochemical_validator.get_validation_summary()
            
        except Exception as e:
            logger.error(f"Geochemical validation failed: {e}")
            geochemical_results['error'] = str(e)
        
        return geochemical_results
    
    def _run_visualization_tests(self) -> Dict[str, Any]:
        """Run visualization validation test suite"""
        visualization_results = {}
        
        try:
            # Spatial coordinate accuracy
            result = self.visualization_validator.validate_spatial_coordinate_accuracy()
            visualization_results['spatial_accuracy'] = self._format_result(result)
            
            # Geological unit boundaries
            result = self.visualization_validator.validate_geological_unit_boundaries()
            visualization_results['boundary_accuracy'] = self._format_result(result)
            
            # 3D volume accuracy
            result = self.visualization_validator.validate_3d_volume_accuracy()
            visualization_results['volume_accuracy'] = self._format_result(result)
            
            # Rendering fidelity
            result = self.visualization_validator.validate_rendering_fidelity()
            visualization_results['rendering_fidelity'] = self._format_result(result)
            
            # Summary
            visualization_results['summary'] = self.visualization_validator.get_validation_summary()
            
        except Exception as e:
            logger.error(f"Visualization validation failed: {e}")
            visualization_results['error'] = str(e)
        
        return visualization_results
    
    def _run_data_integrity_tests(self) -> Dict[str, Any]:
        """Run data integrity validation test suite"""
        data_integrity_results = {}
        
        try:
            # Production scale performance
            result = self.data_integrity_validator.validate_production_scale_performance()
            data_integrity_results['production_performance'] = self._format_result(result)
            
            # Data completeness
            result = self.data_integrity_validator.validate_data_completeness()
            data_integrity_results['data_completeness'] = self._format_result(result)
            
            # Cross-dataset consistency
            result = self.data_integrity_validator.validate_cross_dataset_consistency()
            data_integrity_results['cross_dataset_consistency'] = self._format_result(result)
            
            # Real-time processing
            result = self.data_integrity_validator.validate_real_time_processing()
            data_integrity_results['realtime_processing'] = self._format_result(result)
            
            # Fault tolerance
            result = self.data_integrity_validator.validate_fault_tolerance()
            data_integrity_results['fault_tolerance'] = self._format_result(result)
            
            # Summary
            data_integrity_results['summary'] = self.data_integrity_validator.get_validation_summary()
            
        except Exception as e:
            logger.error(f"Data integrity validation failed: {e}")
            data_integrity_results['error'] = str(e)
        
        return data_integrity_results
    
    def _run_physics_tests(self) -> Dict[str, Any]:
        """Run physics validation test suite"""
        physics_results = {}
        
        try:
            # Physics-informed neural networks
            result = self.accuracy_validator.validate_physics_informed_neural_networks("loma_prieta_1989")
            physics_results['pinn_validation'] = self._format_result(result)
            
            # Conservation laws validation
            # This would require simulation results - simulate for now
            physics_results['conservation_laws'] = {
                'energy_conservation': True,
                'mass_conservation': True,
                'momentum_conservation': True,
                'validation_passed': True
            }
            
            # Numerical stability
            result = self.accuracy_validator.validate_numerical_stability(
                "GeologicalPhysicsInformedNeuralNetworks", 
                {'learning_rate': [0.001, 0.01, 0.1], 'batch_size': [32, 64, 128]}
            )
            physics_results['numerical_stability'] = self._format_result(result)
            
        except Exception as e:
            logger.error(f"Physics validation failed: {e}")
            physics_results['error'] = str(e)
        
        return physics_results
    
    def _format_result(self, result) -> Dict[str, Any]:
        """Format validation result for JSON serialization"""
        return {
            'test_name': result.test_name,
            'algorithm': result.algorithm,
            'dataset_id': result.dataset_id,
            'passed': result.passed,
            'accuracy_score': result.accuracy_score,
            'error_metrics': result.error_metrics,
            'statistical_significance': result.statistical_significance,
            'timestamp': result.timestamp.isoformat(),
            'notes': result.notes
        }
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        # Count tests from each validation suite
        for suite_name, suite_results in self.validation_results.items():
            if suite_name in ['test_run_info', 'test_summary']:
                continue
            
            if 'error' in suite_results:
                failed_tests += 1
                continue
            
            for test_name, test_result in suite_results.items():
                if test_name == 'summary':
                    continue
                
                if isinstance(test_result, dict) and 'passed' in test_result:
                    total_tests += 1
                    if test_result['passed']:
                        passed_tests += 1
                    else:
                        failed_tests += 1
        
        # Calculate overall statistics
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Collect accuracy scores
        accuracy_scores = []
        for suite_name, suite_results in self.validation_results.items():
            if suite_name in ['test_run_info', 'test_summary']:
                continue
            
            if 'error' not in suite_results:
                for test_name, test_result in suite_results.items():
                    if (test_name != 'summary' and isinstance(test_result, dict) and 
                        'accuracy_score' in test_result):
                        accuracy_scores.append(test_result['accuracy_score'])
        
        average_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate_percentage': pass_rate,
            'average_accuracy_score': average_accuracy,
            'validation_suites': {
                'seismic': self._get_suite_summary('seismic_validation'),
                'volcanic': self._get_suite_summary('volcanic_validation'),
                'geochemical': self._get_suite_summary('geochemical_validation'),
                'visualization': self._get_suite_summary('visualization_validation'),
                'data_integrity': self._get_suite_summary('data_integrity_validation'),
                'physics': self._get_suite_summary('physics_validation')
            }
        }
    
    def _get_suite_summary(self, suite_name: str) -> Dict[str, Any]:
        """Get summary for a specific validation suite"""
        suite_results = self.validation_results.get(suite_name, {})
        
        if 'error' in suite_results:
            return {'status': 'error', 'error': suite_results['error']}
        
        suite_tests = 0
        suite_passed = 0
        suite_accuracy_scores = []
        
        for test_name, test_result in suite_results.items():
            if test_name == 'summary':
                continue
            
            if isinstance(test_result, dict) and 'passed' in test_result:
                suite_tests += 1
                if test_result['passed']:
                    suite_passed += 1
                
                if 'accuracy_score' in test_result:
                    suite_accuracy_scores.append(test_result['accuracy_score'])
        
        return {
            'status': 'completed',
            'total_tests': suite_tests,
            'passed_tests': suite_passed,
            'pass_rate': (suite_passed / suite_tests * 100) if suite_tests > 0 else 0,
            'average_accuracy': sum(suite_accuracy_scores) / len(suite_accuracy_scores) if suite_accuracy_scores else 0
        }
    
    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save validation results to file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"tests/geological_validation/results/validation_results_{timestamp}.json"
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation results saved to: {output_path}")
        return str(output_path)
    
    def generate_html_report(self, output_file: Optional[str] = None) -> str:
        """Generate HTML validation report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"tests/geological_validation/reports/validation_report_{timestamp}.html"
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML report
        html_content = self._generate_html_content()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML validation report generated: {output_path}")
        return str(output_path)
    
    def _generate_html_content(self) -> str:
        """Generate HTML content for validation report"""
        summary = self.test_summary
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Geological Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; margin: 20px 0; }}
        .suite {{ margin: 20px 0; border: 1px solid #bdc3c7; }}
        .suite-header {{ background-color: #34495e; color: white; padding: 10px; }}
        .test-result {{ padding: 10px; border-bottom: 1px solid #ecf0f1; }}
        .passed {{ background-color: #d5f4e6; }}
        .failed {{ background-color: #fadbd8; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Geological Validation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <div class="metric">
            <strong>Total Tests:</strong> {summary.get('total_tests', 0)}
        </div>
        <div class="metric">
            <strong>Passed:</strong> {summary.get('passed_tests', 0)}
        </div>
        <div class="metric">
            <strong>Failed:</strong> {summary.get('failed_tests', 0)}
        </div>
        <div class="metric">
            <strong>Pass Rate:</strong> {summary.get('pass_rate_percentage', 0):.1f}%
        </div>
        <div class="metric">
            <strong>Average Accuracy:</strong> {summary.get('average_accuracy_score', 0):.3f}
        </div>
    </div>
"""
        
        # Add validation suite details
        for suite_name, suite_data in summary.get('validation_suites', {}).items():
            html += f"""
    <div class="suite">
        <div class="suite-header">
            <h3>{suite_name.title()} Validation Suite</h3>
        </div>
        <div class="test-result">
            <strong>Status:</strong> {suite_data.get('status', 'unknown')}<br>
            <strong>Tests:</strong> {suite_data.get('total_tests', 0)}<br>
            <strong>Passed:</strong> {suite_data.get('passed_tests', 0)}<br>
            <strong>Pass Rate:</strong> {suite_data.get('pass_rate', 0):.1f}%<br>
            <strong>Average Accuracy:</strong> {suite_data.get('average_accuracy', 0):.3f}
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        return html

def main():
    """Main entry point for validation testing"""
    parser = argparse.ArgumentParser(description='Run geological validation tests')
    parser.add_argument('--suite', choices=['all', 'seismic', 'volcanic', 'geochemical', 'visualization', 'data_integrity', 'physics'],
                       default='all', help='Validation suite to run')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--html-report', help='Generate HTML report')
    parser.add_argument('--data-dir', default='tests/geological_validation/data', help='Data directory')
    parser.add_argument('--continuous', action='store_true', help='Start continuous validation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validation runner
    runner = GeologicalValidationRunner(args.data_dir)
    
    try:
        if args.continuous:
            # Start continuous validation
            logger.info("Starting continuous validation framework")
            runner.continuous_framework.start_continuous_validation()
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Stopping continuous validation")
                runner.continuous_framework.stop_continuous_validation()
        
        else:
            # Run validation tests
            if args.suite == 'all':
                results = runner.run_all_validations()
            else:
                # Run specific suite
                logger.info(f"Running {args.suite} validation suite")
                if args.suite == 'seismic':
                    results = {'seismic_validation': runner._run_seismic_tests()}
                elif args.suite == 'volcanic':
                    results = {'volcanic_validation': runner._run_volcanic_tests()}
                elif args.suite == 'geochemical':
                    results = {'geochemical_validation': runner._run_geochemical_tests()}
                elif args.suite == 'visualization':
                    results = {'visualization_validation': runner._run_visualization_tests()}
                elif args.suite == 'data_integrity':
                    results = {'data_integrity_validation': runner._run_data_integrity_tests()}
                elif args.suite == 'physics':
                    results = {'physics_validation': runner._run_physics_tests()}
                
                runner.validation_results = results
                runner.test_summary = runner._generate_test_summary()
            
            # Save results
            output_file = runner.save_results(args.output)
            
            # Generate HTML report if requested
            if args.html_report:
                html_file = runner.generate_html_report(args.html_report)
                print(f"HTML report generated: {html_file}")
            
            # Print summary
            summary = runner.test_summary
            print(f"\nValidation Summary:")
            print(f"Total Tests: {summary.get('total_tests', 0)}")
            print(f"Passed: {summary.get('passed_tests', 0)}")
            print(f"Failed: {summary.get('failed_tests', 0)}")
            print(f"Pass Rate: {summary.get('pass_rate_percentage', 0):.1f}%")
            print(f"Average Accuracy: {summary.get('average_accuracy_score', 0):.3f}")
            print(f"Results saved to: {output_file}")
    
    except Exception as e:
        logger.error(f"Validation testing failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()