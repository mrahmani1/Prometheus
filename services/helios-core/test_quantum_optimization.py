#!/usr/bin/env python3
"""
Test suite for quantum optimization algorithms for geological problems

This script demonstrates and validates the quantum computing integration
for geological parameter optimization, system analysis, pattern recognition,
and quantum-classical hybrid algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import asyncio
from datetime import datetime
import logging
from typing import Dict, List, Any

# Import quantum optimization components
from core.quantum_optimization import (
    QuantumGeologicalOptimizationManager,
    GeologicalQuantumProblem,
    create_geological_quantum_problem,
    analyze_quantum_advantage
)

from core.quantum_helios_integration import (
    QuantumHeliosIntegrator,
    create_quantum_enhanced_helios_workflow,
    run_quantum_enhanced_geological_analysis
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumGeologicalTestSuite:
    """Comprehensive test suite for quantum geological optimization"""
    
    def __init__(self):
        self.quantum_manager = QuantumGeologicalOptimizationManager()
        self.helios_integrator = QuantumHeliosIntegrator()
        self.test_results = {}
        
    def generate_synthetic_geological_data(self, data_type: str, size: int = 50) -> np.ndarray:
        """Generate synthetic geological data for testing"""
        np.random.seed(42)  # For reproducible results
        
        if data_type == 'seismic_velocities':
            # Realistic seismic velocity data (km/s)
            base_velocity = 6.0
            depth_gradient = np.linspace(0, 2, size)
            noise = np.random.normal(0, 0.2, size)
            return base_velocity + depth_gradient + noise
            
        elif data_type == 'magnetic_anomalies':
            # Magnetic anomaly data (nT)
            x = np.linspace(0, 10, size)
            anomalies = 100 * np.sin(x) + 50 * np.cos(2*x) + np.random.normal(0, 10, size)
            return anomalies
            
        elif data_type == 'gravity_data':
            # Gravity anomaly data (mGal)
            x = np.linspace(0, 20, size)
            gravity = 50 * np.exp(-x/5) + np.random.normal(0, 5, size)
            return gravity
            
        elif data_type == 'geochemical_concentrations':
            # Geochemical element concentrations (ppm)
            concentrations = np.random.lognormal(2, 1, size)
            return concentrations
            
        elif data_type == 'fault_stress_data':
            # Fault stress measurements (MPa)
            stress_levels = np.random.gamma(2, 10, size)
            return stress_levels
            
        else:
            # Generic geological measurements
            return np.random.normal(10, 3, size)
    
    async def test_quantum_annealing_optimization(self) -> Dict[str, Any]:
        """Test quantum annealing for geological parameter optimization"""
        logger.info("Testing Quantum Annealing for Geological Parameter Optimization")
        
        # Generate test data
        seismic_data = self.generate_synthetic_geological_data('seismic_velocities', 20)
        
        # Create quantum problem
        problem = create_geological_quantum_problem(
            seismic_data,
            problem_type='parameter_optimization',
            constraints={'velocity_range': (4.0, 8.0)},
            objective_function='minimize_velocity_misfit',
            quantum_encoding='amplitude'
        )
        
        # Run quantum annealing
        result = await self.quantum_manager.optimize_geological_problem(
            problem, algorithm='quantum_annealing'
        )
        
        test_result = {
            'algorithm': 'quantum_annealing',
            'problem_type': 'parameter_optimization',
            'data_size': len(seismic_data),
            'optimal_energy': result.optimal_energy,
            'quantum_advantage': result.quantum_advantage_metric,
            'confidence_score': result.confidence_score,
            'execution_time': result.execution_time,
            'geological_insights': result.geological_interpretation,
            'success': result.confidence_score > 0.5
        }
        
        logger.info(f"Quantum Annealing Results: Advantage={result.quantum_advantage_metric:.3f}, "
                   f"Confidence={result.confidence_score:.3f}")
        
        return test_result
    
    async def test_variational_quantum_eigensolver(self) -> Dict[str, Any]:
        """Test VQE for geological system analysis"""
        logger.info("Testing Variational Quantum Eigensolver for System Analysis")
        
        # Generate fault network data
        fault_data = self.generate_synthetic_geological_data('fault_stress_data', 15)
        
        # Create quantum problem for system analysis
        problem = create_geological_quantum_problem(
            fault_data,
            problem_type='system_analysis',
            constraints={'stress_threshold': 50.0},
            objective_function='analyze_fault_stability',
            quantum_encoding='angle'
        )
        
        # Run VQE
        result = await self.quantum_manager.optimize_geological_problem(
            problem, algorithm='vqe'
        )
        
        test_result = {
            'algorithm': 'vqe',
            'problem_type': 'system_analysis',
            'data_size': len(fault_data),
            'ground_state_energy': result.optimal_energy,
            'quantum_advantage': result.quantum_advantage_metric,
            'confidence_score': result.confidence_score,
            'execution_time': result.execution_time,
            'system_stability': result.geological_interpretation.get('stability', 'unknown'),
            'success': result.confidence_score > 0.4
        }
        
        logger.info(f"VQE Results: Ground State Energy={result.optimal_energy:.3f}, "
                   f"System Stability={test_result['system_stability']}")
        
        return test_result
    
    async def test_quantum_machine_learning(self) -> Dict[str, Any]:
        """Test quantum machine learning for pattern recognition"""
        logger.info("Testing Quantum Machine Learning for Pattern Recognition")
        
        # Generate geochemical pattern data
        geochemical_data = self.generate_synthetic_geological_data('geochemical_concentrations', 30)
        
        # Create quantum ML problem
        problem = create_geological_quantum_problem(
            geochemical_data,
            problem_type='pattern_recognition',
            constraints={'concentration_threshold': 100.0},
            objective_function='maximize_classification_accuracy',
            quantum_encoding='basis'
        )
        
        # Run quantum ML
        result = await self.quantum_manager.optimize_geological_problem(
            problem, algorithm='quantum_ml'
        )
        
        test_result = {
            'algorithm': 'quantum_ml',
            'problem_type': 'pattern_recognition',
            'data_size': len(geochemical_data),
            'classification_loss': result.optimal_energy,
            'quantum_advantage': result.quantum_advantage_metric,
            'accuracy': result.confidence_score,
            'execution_time': result.execution_time,
            'pattern_insights': result.geological_interpretation.get('geological_insights', []),
            'success': result.confidence_score > 0.6
        }
        
        logger.info(f"Quantum ML Results: Accuracy={result.confidence_score:.3f}, "
                   f"Quantum Advantage={result.quantum_advantage_metric:.3f}")
        
        return test_result
    
    async def test_quantum_classical_hybrid(self) -> Dict[str, Any]:
        """Test quantum-classical hybrid algorithms for large-scale problems"""
        logger.info("Testing Quantum-Classical Hybrid for Large-Scale Problems")
        
        # Generate large-scale geological dataset
        magnetic_data = self.generate_synthetic_geological_data('magnetic_anomalies', 40)
        gravity_data = self.generate_synthetic_geological_data('gravity_data', 40)
        combined_data = np.concatenate([magnetic_data, gravity_data])
        
        # Create hybrid quantum problem
        problem = create_geological_quantum_problem(
            combined_data,
            problem_type='parameter_optimization',
            constraints={'data_range': (-200, 200)},
            objective_function='optimize_geophysical_model',
            quantum_encoding='amplitude'
        )
        
        # Run hybrid optimization
        result = await self.quantum_manager.optimize_geological_problem(
            problem, algorithm='hybrid'
        )
        
        test_result = {
            'algorithm': 'quantum_classical_hybrid',
            'problem_type': 'large_scale_optimization',
            'data_size': len(combined_data),
            'optimal_loss': result.optimal_energy,
            'quantum_advantage': result.quantum_advantage_metric,
            'confidence_score': result.confidence_score,
            'execution_time': result.execution_time,
            'hybrid_performance': result.geological_interpretation.get('hybrid_performance', 'unknown'),
            'success': result.confidence_score > 0.5
        }
        
        logger.info(f"Hybrid Results: Loss={result.optimal_energy:.3f}, "
                   f"Performance={test_result['hybrid_performance']}")
        
        return test_result
    
    async def test_helios_integration(self) -> Dict[str, Any]:
        """Test integration with Helios AI system"""
        logger.info("Testing Quantum-Helios Integration")
        
        # Create multi-component geological dataset
        geological_data = {
            'seismic_data': self.generate_synthetic_geological_data('seismic_velocities', 25),
            'magnetic_data': self.generate_synthetic_geological_data('magnetic_anomalies', 25),
            'gravity_data': self.generate_synthetic_geological_data('gravity_data', 25)
        }
        
        # Test parameter optimization integration
        param_result = await self.helios_integrator.quantum_enhanced_parameter_optimization(
            geological_data['seismic_data'],
            {'velocity_model': 6.0, 'density': 2.5, 'attenuation': 0.1}
        )
        
        # Test system analysis integration
        system_result = await self.helios_integrator.quantum_enhanced_system_analysis(
            geological_data,
            'stability_analysis'
        )
        
        test_result = {
            'integration_type': 'quantum_helios',
            'parameter_optimization': {
                'classical_misfit': param_result.classical_result.get('final_misfit', 0),
                'quantum_advantage': param_result.quantum_optimization_result.quantum_advantage_metric,
                'enhancement_achieved': param_result.enhancement_metrics['overall_enhancement'],
                'processing_time': param_result.processing_time
            },
            'system_analysis': {
                'classical_stability': system_result.classical_result.get('overall_stability', 0),
                'quantum_stability': system_result.quantum_optimization_result.confidence_score,
                'enhanced_classification': system_result.combined_interpretation.get('enhanced_system_classification', 'unknown'),
                'processing_time': system_result.processing_time
            },
            'overall_success': (
                param_result.enhancement_metrics['overall_enhancement'] > 0.5 and
                system_result.enhancement_metrics['overall_enhancement'] > 0.5
            )
        }
        
        logger.info(f"Helios Integration: Parameter Enhancement={test_result['parameter_optimization']['enhancement_achieved']:.3f}, "
                   f"System Classification={test_result['system_analysis']['enhanced_classification']}")
        
        return test_result
    
    async def test_algorithm_benchmarking(self) -> Dict[str, Any]:
        """Benchmark all quantum algorithms on the same problem"""
        logger.info("Benchmarking All Quantum Algorithms")
        
        # Create standardized test problem
        test_data = self.generate_synthetic_geological_data('seismic_velocities', 20)
        problem = create_geological_quantum_problem(
            test_data,
            problem_type='parameter_optimization',
            objective_function='minimize_misfit'
        )
        
        # Benchmark all algorithms
        benchmark_results = self.quantum_manager.benchmark_algorithms(problem)
        
        # Analyze results
        algorithm_performance = {}
        for alg_name, result in benchmark_results.items():
            algorithm_performance[alg_name] = {
                'quantum_advantage': result.quantum_advantage_metric,
                'confidence_score': result.confidence_score,
                'execution_time': result.execution_time,
                'optimal_energy': result.optimal_energy
            }
        
        # Find best performing algorithm
        best_algorithm = max(benchmark_results.keys(), 
                           key=lambda k: benchmark_results[k].quantum_advantage_metric)
        
        test_result = {
            'benchmark_type': 'algorithm_comparison',
            'test_problem_size': len(test_data),
            'algorithm_performance': algorithm_performance,
            'best_algorithm': best_algorithm,
            'best_advantage': benchmark_results[best_algorithm].quantum_advantage_metric,
            'performance_analysis': analyze_quantum_advantage(list(benchmark_results.values()))
        }
        
        logger.info(f"Best Algorithm: {best_algorithm} with advantage {test_result['best_advantage']:.3f}")
        
        return test_result
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("Starting Comprehensive Quantum Geological Optimization Test Suite")
        start_time = datetime.now()
        
        # Run all tests
        test_results = {}
        
        try:
            test_results['quantum_annealing'] = await self.test_quantum_annealing_optimization()
        except Exception as e:
            logger.error(f"Quantum annealing test failed: {e}")
            test_results['quantum_annealing'] = {'success': False, 'error': str(e)}
        
        try:
            test_results['vqe'] = await self.test_variational_quantum_eigensolver()
        except Exception as e:
            logger.error(f"VQE test failed: {e}")
            test_results['vqe'] = {'success': False, 'error': str(e)}
        
        try:
            test_results['quantum_ml'] = await self.test_quantum_machine_learning()
        except Exception as e:
            logger.error(f"Quantum ML test failed: {e}")
            test_results['quantum_ml'] = {'success': False, 'error': str(e)}
        
        try:
            test_results['hybrid'] = await self.test_quantum_classical_hybrid()
        except Exception as e:
            logger.error(f"Hybrid test failed: {e}")
            test_results['hybrid'] = {'success': False, 'error': str(e)}
        
        try:
            test_results['helios_integration'] = await self.test_helios_integration()
        except Exception as e:
            logger.error(f"Helios integration test failed: {e}")
            test_results['helios_integration'] = {'success': False, 'error': str(e)}
        
        try:
            test_results['benchmarking'] = await self.test_algorithm_benchmarking()
        except Exception as e:
            logger.error(f"Benchmarking test failed: {e}")
            test_results['benchmarking'] = {'success': False, 'error': str(e)}
        
        # Calculate overall results
        total_time = (datetime.now() - start_time).total_seconds()
        successful_tests = sum(1 for result in test_results.values() 
                             if result.get('success', False))
        total_tests = len(test_results)
        
        summary = {
            'test_suite': 'quantum_geological_optimization',
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests,
            'total_execution_time': total_time,
            'test_results': test_results,
            'overall_success': successful_tests >= total_tests * 0.8  # 80% success threshold
        }
        
        logger.info(f"Test Suite Complete: {successful_tests}/{total_tests} tests passed "
                   f"({summary['success_rate']:.1%} success rate)")
        
        return summary
    
    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report"""
        report = []
        report.append("=" * 80)
        report.append("QUANTUM GEOLOGICAL OPTIMIZATION TEST REPORT")
        report.append("=" * 80)
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tests: {test_results['total_tests']}")
        report.append(f"Successful Tests: {test_results['successful_tests']}")
        report.append(f"Success Rate: {test_results['success_rate']:.1%}")
        report.append(f"Total Execution Time: {test_results['total_execution_time']:.2f} seconds")
        report.append("")
        
        # Individual test results
        for test_name, result in test_results['test_results'].items():
            report.append(f"Test: {test_name.upper()}")
            report.append("-" * 40)
            
            if result.get('success', False):
                report.append("Status: PASSED ✓")
                
                if 'quantum_advantage' in result:
                    report.append(f"Quantum Advantage: {result['quantum_advantage']:.3f}")
                if 'confidence_score' in result:
                    report.append(f"Confidence Score: {result['confidence_score']:.3f}")
                if 'execution_time' in result:
                    report.append(f"Execution Time: {result['execution_time']:.3f}s")
                    
                # Algorithm-specific metrics
                if test_name == 'quantum_annealing':
                    report.append(f"Optimal Energy: {result.get('optimal_energy', 'N/A')}")
                elif test_name == 'vqe':
                    report.append(f"System Stability: {result.get('system_stability', 'N/A')}")
                elif test_name == 'quantum_ml':
                    report.append(f"Classification Accuracy: {result.get('accuracy', 'N/A'):.3f}")
                elif test_name == 'hybrid':
                    report.append(f"Hybrid Performance: {result.get('hybrid_performance', 'N/A')}")
                elif test_name == 'helios_integration':
                    param_enh = result.get('parameter_optimization', {}).get('enhancement_achieved', 'N/A')
                    report.append(f"Parameter Enhancement: {param_enh}")
                elif test_name == 'benchmarking':
                    report.append(f"Best Algorithm: {result.get('best_algorithm', 'N/A')}")
                    report.append(f"Best Advantage: {result.get('best_advantage', 'N/A'):.3f}")
            else:
                report.append("Status: FAILED ✗")
                if 'error' in result:
                    report.append(f"Error: {result['error']}")
            
            report.append("")
        
        # Overall assessment
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 40)
        if test_results['overall_success']:
            report.append("Overall Status: PASSED ✓")
            report.append("Quantum geological optimization algorithms are functioning correctly.")
        else:
            report.append("Overall Status: FAILED ✗")
            report.append("Some quantum algorithms require attention.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_test_results(self, test_results: Dict[str, Any], filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantum_optimization_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {filename}")
    
    def create_visualization(self, test_results: Dict[str, Any]):
        """Create visualization of test results"""
        try:
            # Extract performance metrics
            algorithms = []
            advantages = []
            confidences = []
            times = []
            
            for test_name, result in test_results['test_results'].items():
                if result.get('success', False) and 'quantum_advantage' in result:
                    algorithms.append(test_name)
                    advantages.append(result['quantum_advantage'])
                    confidences.append(result.get('confidence_score', 0))
                    times.append(result.get('execution_time', 0))
            
            if not algorithms:
                logger.warning("No successful test results to visualize")
                return
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Quantum Geological Optimization Test Results', fontsize=16)
            
            # Quantum advantage comparison
            ax1.bar(algorithms, advantages, color='skyblue', alpha=0.7)
            ax1.set_title('Quantum Advantage by Algorithm')
            ax1.set_ylabel('Quantum Advantage Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # Confidence scores
            ax2.bar(algorithms, confidences, color='lightgreen', alpha=0.7)
            ax2.set_title('Confidence Scores')
            ax2.set_ylabel('Confidence Score')
            ax2.tick_params(axis='x', rotation=45)
            
            # Execution times
            ax3.bar(algorithms, times, color='salmon', alpha=0.7)
            ax3.set_title('Execution Times')
            ax3.set_ylabel('Time (seconds)')
            ax3.tick_params(axis='x', rotation=45)
            
            # Success rate pie chart
            success_count = test_results['successful_tests']
            failure_count = test_results['total_tests'] - success_count
            ax4.pie([success_count, failure_count], 
                   labels=['Passed', 'Failed'],
                   colors=['lightgreen', 'lightcoral'],
                   autopct='%1.1f%%')
            ax4.set_title('Overall Test Success Rate')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantum_optimization_test_visualization_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")


async def main():
    """Main function to run the quantum optimization test suite"""
    print("Quantum Geological Optimization Test Suite")
    print("=" * 50)
    
    # Initialize test suite
    test_suite = QuantumGeologicalTestSuite()
    
    # Run comprehensive tests
    results = await test_suite.run_comprehensive_test_suite()
    
    # Generate and display report
    report = test_suite.generate_test_report(results)
    print(report)
    
    # Save results
    test_suite.save_test_results(results)
    
    # Create visualization
    test_suite.create_visualization(results)
    
    # Return results for further analysis
    return results


if __name__ == "__main__":
    # Run the test suite
    test_results = asyncio.run(main())
    
    # Print summary
    print(f"\nTest Suite Summary:")
    print(f"Overall Success: {'✓' if test_results['overall_success'] else '✗'}")
    print(f"Success Rate: {test_results['success_rate']:.1%}")
    print(f"Total Time: {test_results['total_execution_time']:.2f} seconds")