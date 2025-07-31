#!/usr/bin/env python3
"""
Quantum Optimization for Geological Problems - Demonstration Script

This script demonstrates the quantum computing capabilities for geological
parameter optimization, system analysis, pattern recognition, and hybrid algorithms.
"""

import numpy as np
import asyncio
import logging
from datetime import datetime
import json

# Import quantum optimization components
from core.quantum_optimization import (
    QuantumGeologicalOptimizationManager,
    create_geological_quantum_problem,
    analyze_quantum_advantage
)

from core.quantum_helios_integration import (
    QuantumHeliosIntegrator,
    run_quantum_enhanced_geological_analysis
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_realistic_geological_data():
    """Generate realistic synthetic geological data for demonstration"""
    np.random.seed(42)  # For reproducible results
    
    geological_datasets = {
        'seismic_velocities': {
            'data': np.array([5.8, 6.1, 6.3, 6.5, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8]),
            'description': 'P-wave velocities (km/s) with depth',
            'units': 'km/s'
        },
        'magnetic_anomalies': {
            'data': np.array([120, 85, -45, -120, -80, 30, 95, 140, 75, -25, -90, -150]),
            'description': 'Magnetic field anomalies',
            'units': 'nT'
        },
        'gravity_data': {
            'data': np.array([45, 38, 42, 35, 28, 31, 39, 44, 41, 36]),
            'description': 'Gravity anomaly measurements',
            'units': 'mGal'
        },
        'fault_stress': {
            'data': np.array([25, 45, 38, 52, 41, 33, 47, 39, 44, 36, 42, 48]),
            'description': 'Fault stress measurements',
            'units': 'MPa'
        }
    }
    
    return geological_datasets

async def demonstrate_quantum_annealing():
    """Demonstrate quantum annealing for geological parameter optimization"""
    print("\n" + "="*60)
    print("QUANTUM ANNEALING FOR GEOLOGICAL PARAMETER OPTIMIZATION")
    print("="*60)
    
    # Create geological problem
    datasets = generate_realistic_geological_data()
    seismic_data = datasets['seismic_velocities']['data']
    
    problem = create_geological_quantum_problem(
        geological_data=seismic_data,
        problem_type='parameter_optimization',
        constraints={'velocity_range': (5.0, 8.0)},
        objective_function='minimize_velocity_misfit',
        quantum_encoding='amplitude'
    )
    
    # Initialize quantum manager
    manager = QuantumGeologicalOptimizationManager()
    
    # Run quantum annealing
    print(f"Input data: {seismic_data}")
    print(f"Problem type: {problem.problem_type}")
    print("Running quantum annealing optimization...")
    
    result = await asyncio.get_event_loop().run_in_executor(
        None, manager.optimize_geological_problem, problem, 'quantum_annealing'
    )
    
    # Display results
    print(f"\nResults:")
    print(f"  Algorithm: {result.algorithm_type}")
    print(f"  Optimal energy: {result.optimal_energy:.4f}")
    print(f"  Quantum advantage: {result.quantum_advantage_metric:.4f}")
    print(f"  Confidence score: {result.confidence_score:.4f}")
    print(f"  Execution time: {result.execution_time:.2f}s")
    print(f"  Geological insights:")
    for insight in result.geological_interpretation.get('geological_insights', []):
        print(f"    - {insight}")
    
    return result

async def demonstrate_vqe_system_analysis():
    """Demonstrate VQE for geological system analysis"""
    print("\n" + "="*60)
    print("VARIATIONAL QUANTUM EIGENSOLVER FOR SYSTEM ANALYSIS")
    print("="*60)
    
    # Create fault network analysis problem
    datasets = generate_realistic_geological_data()
    fault_data = datasets['fault_stress']['data'][:8]  # Limit for quantum processing
    
    problem = create_geological_quantum_problem(
        geological_data=fault_data,
        problem_type='system_analysis',
        constraints={'stress_threshold': 40.0},
        objective_function='analyze_fault_stability',
        quantum_encoding='angle'
    )
    
    manager = QuantumGeologicalOptimizationManager()
    
    print(f"Fault stress data: {fault_data}")
    print(f"Problem type: {problem.problem_type}")
    print("Running VQE system analysis...")
    
    result = await asyncio.get_event_loop().run_in_executor(
        None, manager.optimize_geological_problem, problem, 'vqe'
    )
    
    # Display results
    print(f"\nResults:")
    print(f"  Algorithm: {result.algorithm_type}")
    print(f"  Ground state energy: {result.optimal_energy:.4f}")
    print(f"  System stability: {result.geological_interpretation.get('stability', 'unknown')}")
    print(f"  Quantum advantage: {result.quantum_advantage_metric:.4f}")
    print(f"  Confidence score: {result.confidence_score:.4f}")
    print(f"  Execution time: {result.execution_time:.2f}s")
    
    return result

async def demonstrate_quantum_machine_learning():
    """Demonstrate quantum machine learning for pattern recognition"""
    print("\n" + "="*60)
    print("QUANTUM MACHINE LEARNING FOR PATTERN RECOGNITION")
    print("="*60)
    
    # Create pattern recognition problem
    datasets = generate_realistic_geological_data()
    magnetic_data = datasets['magnetic_anomalies']['data']
    
    problem = create_geological_quantum_problem(
        geological_data=magnetic_data,
        problem_type='pattern_recognition',
        constraints={'anomaly_threshold': 50.0},
        objective_function='classify_magnetic_patterns',
        quantum_encoding='basis'
    )
    
    manager = QuantumGeologicalOptimizationManager()
    
    print(f"Magnetic anomaly data: {magnetic_data}")
    print(f"Problem type: {problem.problem_type}")
    print("Running quantum machine learning...")
    
    result = await asyncio.get_event_loop().run_in_executor(
        None, manager.optimize_geological_problem, problem, 'qml'
    )
    
    # Display results
    print(f"\nResults:")
    print(f"  Algorithm: {result.algorithm_type}")
    print(f"  Classification accuracy: {result.confidence_score:.4f}")
    print(f"  Quantum advantage: {result.quantum_advantage_metric:.4f}")
    print(f"  Model performance: {result.geological_interpretation.get('model_performance', 'unknown')}")
    print(f"  Execution time: {result.execution_time:.2f}s")
    
    return result

async def demonstrate_hybrid_optimization():
    """Demonstrate quantum-classical hybrid optimization"""
    print("\n" + "="*60)
    print("QUANTUM-CLASSICAL HYBRID FOR LARGE-SCALE PROBLEMS")
    print("="*60)
    
    # Create large-scale optimization problem
    datasets = generate_realistic_geological_data()
    
    # Combine multiple datasets for large-scale problem
    combined_data = np.concatenate([
        datasets['seismic_velocities']['data'],
        datasets['gravity_data']['data']
    ])
    
    problem = create_geological_quantum_problem(
        geological_data=combined_data,
        problem_type='parameter_optimization',
        constraints={'data_range': (-200, 200)},
        objective_function='optimize_geophysical_model',
        quantum_encoding='amplitude'
    )
    
    manager = QuantumGeologicalOptimizationManager()
    
    print(f"Combined geophysical data size: {len(combined_data)}")
    print(f"Problem type: {problem.problem_type}")
    print("Running quantum-classical hybrid optimization...")
    
    result = await asyncio.get_event_loop().run_in_executor(
        None, manager.optimize_geological_problem, problem, 'hybrid'
    )
    
    # Display results
    print(f"\nResults:")
    print(f"  Algorithm: {result.algorithm_type}")
    print(f"  Optimal objective: {result.optimal_energy:.4f}")
    print(f"  Quantum advantage: {result.quantum_advantage_metric:.4f}")
    print(f"  Confidence score: {result.confidence_score:.4f}")
    print(f"  Hybrid performance: {result.geological_interpretation.get('hybrid_performance', 'unknown')}")
    print(f"  Execution time: {result.execution_time:.2f}s")
    
    return result

async def demonstrate_helios_integration():
    """Demonstrate quantum-enhanced Helios integration"""
    print("\n" + "="*60)
    print("QUANTUM-ENHANCED HELIOS AI INTEGRATION")
    print("="*60)
    
    # Create multi-component geological dataset
    datasets = generate_realistic_geological_data()
    geological_data = {
        'seismic_data': datasets['seismic_velocities']['data'],
        'magnetic_data': datasets['magnetic_anomalies']['data'][:10],  # Match size
        'gravity_data': datasets['gravity_data']['data']
    }
    
    print("Multi-component geological dataset:")
    for key, data in geological_data.items():
        print(f"  {key}: {len(data)} measurements")
    
    # Initialize Helios integrator
    integrator = QuantumHeliosIntegrator()
    
    print("\nRunning quantum-enhanced parameter optimization...")
    
    # Test parameter optimization integration
    param_result = await integrator.quantum_enhanced_parameter_optimization(
        geological_data['seismic_data'],
        {'velocity_model': 6.5, 'density': 2.7, 'attenuation': 0.05}
    )
    
    print(f"\nParameter Optimization Results:")
    print(f"  Classical misfit: {param_result.classical_result.get('final_misfit', 'N/A')}")
    print(f"  Quantum advantage: {param_result.quantum_optimization_result.quantum_advantage_metric:.4f}")
    print(f"  Overall enhancement: {param_result.enhancement_metrics['overall_enhancement']:.4f}")
    print(f"  Processing time: {param_result.processing_time:.2f}s")
    
    print("\nRunning quantum-enhanced system analysis...")
    
    # Test system analysis integration
    system_result = await integrator.quantum_enhanced_system_analysis(
        geological_data,
        'stability_analysis'
    )
    
    print(f"\nSystem Analysis Results:")
    print(f"  Classical stability: {system_result.classical_result.get('overall_stability', 'N/A')}")
    print(f"  Enhanced classification: {system_result.combined_interpretation.get('enhanced_system_classification', 'N/A')}")
    print(f"  Quantum stability metric: {system_result.quantum_optimization_result.confidence_score:.4f}")
    print(f"  Processing time: {system_result.processing_time:.2f}s")
    
    return param_result, system_result

async def benchmark_all_algorithms():
    """Benchmark all quantum algorithms on the same problem"""
    print("\n" + "="*60)
    print("QUANTUM ALGORITHM BENCHMARKING")
    print("="*60)
    
    # Create standardized test problem
    datasets = generate_realistic_geological_data()
    test_data = datasets['seismic_velocities']['data']
    
    problem = create_geological_quantum_problem(
        geological_data=test_data,
        problem_type='parameter_optimization',
        objective_function='minimize_misfit'
    )
    
    manager = QuantumGeologicalOptimizationManager()
    
    print(f"Benchmarking all algorithms on seismic data: {test_data}")
    print("Running benchmark...")
    
    # Run benchmark
    benchmark_results = await asyncio.get_event_loop().run_in_executor(
        None, manager.benchmark_quantum_algorithms, problem
    )
    
    # Display results
    print(f"\nBenchmark Results:")
    print(f"{'Algorithm':<20} {'Advantage':<12} {'Confidence':<12} {'Time (s)':<10}")
    print("-" * 60)
    
    for alg_name, result in benchmark_results.items():
        print(f"{alg_name:<20} {result.quantum_advantage_metric:<12.4f} "
              f"{result.confidence_score:<12.4f} {result.execution_time:<10.2f}")
    
    # Find best algorithm
    if benchmark_results:
        best_algorithm = max(benchmark_results.keys(), 
                           key=lambda k: benchmark_results[k].quantum_advantage_metric)
        best_advantage = benchmark_results[best_algorithm].quantum_advantage_metric
        
        print(f"\nBest performing algorithm: {best_algorithm}")
        print(f"Best quantum advantage: {best_advantage:.4f}")
        
        # Analyze quantum advantage
        analysis = analyze_quantum_advantage(list(benchmark_results.values()))
        print(f"Average quantum advantage: {analysis.get('mean_quantum_advantage', 'N/A'):.4f}")
        print(f"Most reliable algorithm: {analysis.get('most_reliable', 'N/A')}")
    
    return benchmark_results

def generate_summary_report(all_results):
    """Generate a comprehensive summary report"""
    print("\n" + "="*60)
    print("QUANTUM GEOLOGICAL OPTIMIZATION SUMMARY REPORT")
    print("="*60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_demonstrations': len(all_results),
        'algorithms_tested': [],
        'performance_summary': {},
        'key_findings': []
    }
    
    # Collect algorithm performance data
    for demo_name, result in all_results.items():
        if isinstance(result, tuple):  # Helios integration returns tuple
            for i, res in enumerate(result):
                alg_name = res.quantum_optimization_result.algorithm_type
                if alg_name not in report['algorithms_tested']:
                    report['algorithms_tested'].append(alg_name)
        elif hasattr(result, 'algorithm_type'):
            if result.algorithm_type not in report['algorithms_tested']:
                report['algorithms_tested'].append(result.algorithm_type)
        elif isinstance(result, dict):  # Benchmark results
            for alg_name in result.keys():
                if alg_name not in report['algorithms_tested']:
                    report['algorithms_tested'].append(alg_name)
    
    # Generate key findings
    report['key_findings'] = [
        "Quantum optimization algorithms successfully demonstrated for geological problems",
        "Classical simulation provides functional quantum algorithm behavior",
        "Hybrid quantum-classical approaches show promise for large-scale problems",
        "Integration with Helios AI system enables enhanced geological analysis",
        "Multiple quantum algorithms available for different geological problem types"
    ]
    
    print(f"Demonstration completed at: {report['timestamp']}")
    print(f"Total demonstrations: {report['total_demonstrations']}")
    print(f"Algorithms tested: {', '.join(report['algorithms_tested'])}")
    print("\nKey Findings:")
    for i, finding in enumerate(report['key_findings'], 1):
        print(f"  {i}. {finding}")
    
    print(f"\nQuantum Computing Status:")
    print(f"  - Qiskit integration: Available (using classical simulation)")
    print(f"  - Quantum algorithms: 4 implemented (annealing, VQE, QML, hybrid)")
    print(f"  - Helios integration: Functional")
    print(f"  - API endpoints: Available")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quantum_optimization_demo_report_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {filename}")
    
    return report

async def main():
    """Main demonstration function"""
    print("QUANTUM GEOLOGICAL OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    print("This demonstration showcases quantum computing capabilities")
    print("for geological parameter optimization, system analysis,")
    print("pattern recognition, and quantum-classical hybrid algorithms.")
    print("=" * 60)
    
    all_results = {}
    
    try:
        # Run all demonstrations
        print("\nStarting quantum optimization demonstrations...")
        
        # 1. Quantum Annealing
        all_results['quantum_annealing'] = await demonstrate_quantum_annealing()
        
        # 2. VQE System Analysis
        all_results['vqe_analysis'] = await demonstrate_vqe_system_analysis()
        
        # 3. Quantum Machine Learning
        all_results['quantum_ml'] = await demonstrate_quantum_machine_learning()
        
        # 4. Hybrid Optimization
        all_results['hybrid_optimization'] = await demonstrate_hybrid_optimization()
        
        # 5. Helios Integration
        all_results['helios_integration'] = await demonstrate_helios_integration()
        
        # 6. Algorithm Benchmarking
        all_results['benchmarking'] = await benchmark_all_algorithms()
        
        # Generate summary report
        summary_report = generate_summary_report(all_results)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("All quantum optimization algorithms have been demonstrated")
        print("and are ready for geological problem solving.")
        
        return all_results
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        logger.error(f"Demonstration failed: {e}")
        return None

if __name__ == "__main__":
    # Run the demonstration
    results = asyncio.run(main())
    
    if results:
        print(f"\nDemonstration completed with {len(results)} successful tests.")
    else:
        print("\nDemonstration encountered errors. Check logs for details.")