#!/usr/bin/env python3
"""
Quantum Computing Integration Examples

This script demonstrates the usage of quantum computing capabilities
for geological applications including simulation, machine learning,
optimization, and cryptography.
"""

import asyncio
import numpy as np
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from quantum_computing_integration import (
    QuantumComputingIntegration,
    GeologicalDataset,
    QuantumCircuitConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_quantum_config():
    """Load quantum computing configuration"""
    config_path = Path(__file__).parent.parent / 'config' / 'quantum_config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['quantum_computing']
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return get_default_config()

def get_default_config():
    """Get default quantum computing configuration"""
    return {
        'circuit': {
            'num_qubits': 6,
            'circuit_depth': 4
        },
        'machine_learning': {
            'num_qubits': 4,
            'num_layers': 3
        },
        'optimization': {
            'num_qubits': 6
        },
        'cryptography': {
            'key_length': 256
        },
        'performance': {
            'max_qubits': 8
        }
    }

def create_sample_geological_data():
    """Create sample geological datasets for testing"""
    datasets = []
    
    # Volcanic monitoring dataset
    volcanic_data = GeologicalDataset(
        id="mount_vesuvius_2024",
        timestamp=datetime.now(),
        data_type="volcanic_monitoring",
        spatial_extent=None,
        temporal_range=None,
        quality_metrics=None,
        metadata={
            'volcano_name': 'Mount Vesuvius',
            'volcano_type': 'stratovolcano',
            'location': {'lat': 40.8214, 'lon': 14.4262},
            'elevation': 1281,
            'last_eruption': '1944-03-29',
            'magma_composition': 'intermediate',
            'monitoring_stations': 15,
            'alert_level': 'green'
        }
    )
    # Add synthetic stress and thermal data
    volcanic_data.stress_tensor = np.random.uniform(-10, 10, (3, 3))
    volcanic_data.temperature_field = np.random.uniform(200, 800, (10, 10))
    datasets.append(volcanic_data)
    
    # Seismic monitoring dataset
    seismic_data = GeologicalDataset(
        id="san_andreas_fault_2024",
        timestamp=datetime.now(),
        data_type="seismic_monitoring",
        spatial_extent=None,
        temporal_range=None,
        quality_metrics=None,
        metadata={
            'fault_name': 'San Andreas Fault',
            'fault_type': 'strike-slip',
            'location': {'lat': 37.0, 'lon': -122.0},
            'depth_range': {'min': 0, 'max': 15},
            'recent_activity': 'moderate',
            'monitoring_network': 'USGS',
            'stations_count': 50
        }
    )
    # Add synthetic seismic data
    seismic_data.stress_tensor = np.random.uniform(-5, 15, (3, 3))
    seismic_data.temperature_field = np.random.uniform(100, 300, (20, 20))
    datasets.append(seismic_data)
    
    # Geothermal exploration dataset
    geothermal_data = GeologicalDataset(
        id="yellowstone_geothermal_2024",
        timestamp=datetime.now(),
        data_type="geothermal_exploration",
        spatial_extent=None,
        temporal_range=None,
        quality_metrics=None,
        metadata={
            'field_name': 'Yellowstone Geothermal Field',
            'field_type': 'high_temperature',
            'location': {'lat': 44.4280, 'lon': -110.5885},
            'estimated_temperature': 350,
            'reservoir_depth': 2000,
            'exploration_phase': 'advanced',
            'resource_potential': 'high'
        }
    )
    # Add synthetic geothermal data
    geothermal_data.stress_tensor = np.random.uniform(-2, 8, (3, 3))
    geothermal_data.temperature_field = np.random.uniform(300, 400, (15, 15))
    datasets.append(geothermal_data)
    
    return datasets

async def example_quantum_geological_simulation():
    """Example: Quantum geological process simulation"""
    print("\n" + "="*60)
    print("QUANTUM GEOLOGICAL SIMULATION EXAMPLE")
    print("="*60)
    
    # Load configuration
    config = load_quantum_config()
    
    # Initialize quantum computing integration
    quantum_config = {
        'num_qubits': config['circuit']['num_qubits'],
        'circuit_depth': config['circuit']['circuit_depth'],
        'ml_qubits': config['machine_learning']['num_qubits'],
        'ml_layers': config['machine_learning']['num_layers'],
        'opt_qubits': config['optimization']['num_qubits'],
        'key_length': config['cryptography']['key_length'],
        'max_qubits': config['performance']['max_qubits'],
        'cloud_services': {}  # Disable cloud services for example
    }
    
    quantum_system = QuantumComputingIntegration(quantum_config)
    await quantum_system.initialize()
    
    # Create sample geological datasets
    datasets = create_sample_geological_data()
    
    for dataset in datasets:
        print(f"\nSimulating geological processes for: {dataset.metadata.get('volcano_name', dataset.metadata.get('fault_name', dataset.metadata.get('field_name', dataset.id)))}")
        print(f"Dataset type: {dataset.data_type}")
        
        try:
            # Run quantum simulation
            result = await quantum_system.simulate_geological_process(dataset)
            
            if 'error' in result:
                print(f"Simulation error: {result['error']}")
                continue
                
            # Display results
            if 'geological_analysis' in result:
                analysis = result['geological_analysis']
                print(f"  Dominant geological state: {analysis.get('dominant_state', 'N/A')}")
                print(f"  Entanglement entropy: {analysis.get('entanglement_entropy', 0):.3f}")
                print(f"  Number of stable states: {len(analysis.get('stable_geological_states', {}))}")
                
                # Process correlations
                correlations = analysis.get('process_correlations', {})
                if correlations:
                    print("  Process correlations:")
                    for process_pair, correlation in correlations.items():
                        print(f"    {process_pair}: {correlation:.3f}")
                        
            if 'geological_insights' in result:
                insights = result['geological_insights']
                print(f"  System stability: {insights.get('system_stability', 'Unknown')}")
                print(f"  Quantum advantage detected: {insights.get('quantum_advantage', False)}")
                
            print(f"  Quantum volume: {result.get('quantum_volume', 0)}")
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            
    print("\nQuantum geological simulation completed!")

async def example_quantum_machine_learning():
    """Example: Quantum machine learning for seismic classification"""
    print("\n" + "="*60)
    print("QUANTUM MACHINE LEARNING EXAMPLE")
    print("="*60)
    
    # Load configuration
    config = load_quantum_config()
    
    # Initialize quantum system
    quantum_config = {
        'ml_qubits': config['machine_learning']['num_qubits'],
        'ml_layers': config['machine_learning']['num_layers'],
        'cloud_services': {}
    }
    
    quantum_system = QuantumComputingIntegration(quantum_config)
    await quantum_system.initialize()
    
    # Generate synthetic seismic data
    print("Generating synthetic seismic event data...")
    np.random.seed(42)  # For reproducible results
    
    # Features: magnitude, depth, duration, frequency_content
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    
    # Add some structure to the data
    # Class 0: shallow earthquakes (depth < 0, higher frequency)
    # Class 1: deep earthquakes (depth > 0, lower frequency)
    y = np.zeros(n_samples)
    for i in range(n_samples):
        if X[i, 1] < 0:  # Shallow (negative depth feature)
            y[i] = -1
            X[i, 3] += 0.5  # Higher frequency content
        else:  # Deep
            y[i] = 1
            X[i, 3] -= 0.5  # Lower frequency content
            
    print(f"Dataset: {n_samples} seismic events")
    print(f"Features: magnitude, depth, duration, frequency_content")
    print(f"Classes: {np.sum(y == -1)} shallow events, {np.sum(y == 1)} deep events")
    
    try:
        # Train quantum classifier
        print("\nTraining quantum seismic event classifier...")
        result = await quantum_system.train_quantum_geological_classifier(X, y)
        
        # Display results
        print(f"Training completed!")
        print(f"Quantum predictions shape: {result.predictions.shape}")
        print(f"Confidence scores shape: {result.confidence_scores.shape}")
        
        # Calculate accuracy
        predicted_classes = np.sign(result.predictions)
        accuracy = np.mean(predicted_classes == y)
        print(f"Training accuracy: {accuracy:.3f}")
        
        # Quantum advantage analysis
        quantum_advantage = result.classical_comparison.get('accuracy_difference', 0)
        print(f"Quantum advantage over classical: {quantum_advantage:.3f}")
        
        # Entanglement analysis
        avg_entanglement = result.entanglement_measures.get('average_entanglement', 0)
        print(f"Average quantum entanglement: {avg_entanglement:.3f}")
        
        # Show some predictions
        print("\nSample predictions:")
        for i in range(min(10, len(result.predictions))):
            actual = "shallow" if y[i] == -1 else "deep"
            predicted = "shallow" if result.predictions[i] < 0 else "deep"
            confidence = result.confidence_scores[i]
            print(f"  Event {i+1}: Actual={actual}, Predicted={predicted}, Confidence={confidence:.3f}")
            
    except Exception as e:
        print(f"Error in quantum machine learning: {e}")
        
    print("\nQuantum machine learning example completed!")

async def example_quantum_optimization():
    """Example: Quantum optimization for geological resource allocation"""
    print("\n" + "="*60)
    print("QUANTUM OPTIMIZATION EXAMPLE")
    print("="*60)
    
    # Load configuration
    config = load_quantum_config()
    
    # Initialize quantum system
    quantum_config = {
        'opt_qubits': config['optimization']['num_qubits'],
        'cloud_services': {}
    }
    
    quantum_system = QuantumComputingIntegration(quantum_config)
    await quantum_system.initialize()
    
    # Define geological resource allocation problem
    print("Setting up geological resource allocation problem...")
    
    # Cost matrix for drilling locations
    # Rows/columns represent different geological sites
    locations = [
        "Site A - Volcanic prospect",
        "Site B - Geothermal field", 
        "Site C - Mineral deposit",
        "Site D - Oil/gas reservoir"
    ]
    
    # Cost matrix (drilling costs, environmental impact, accessibility)
    cost_matrix = np.array([
        [10, 15, 20, 25],  # Site A costs
        [15, 8, 18, 22],   # Site B costs  
        [20, 18, 12, 16],  # Site C costs
        [25, 22, 16, 14]   # Site D costs
    ])
    
    # Constraints
    constraints = {
        'penalty_weight': 5.0,
        'max_resources': 2,  # Can only drill at 2 locations
        'capacity_constraints': ['budget_limit', 'equipment_availability']
    }
    
    print("Optimization problem:")
    print(f"  Locations: {len(locations)}")
    for i, location in enumerate(locations):
        print(f"    {i}: {location}")
    print(f"  Cost matrix shape: {cost_matrix.shape}")
    print(f"  Maximum resources: {constraints['max_resources']}")
    
    try:
        # Solve optimization problem
        print("\nSolving with quantum optimization (QAOA)...")
        result = await quantum_system.optimize_resource_allocation(cost_matrix, constraints)
        
        # Display results
        print(f"Optimization completed!")
        print(f"Optimal value: {result.optimal_value:.3f}")
        print(f"Execution time: {result.execution_time:.3f} seconds")
        print(f"Quantum advantage: {result.quantum_advantage}")
        print(f"Error mitigation applied: {result.error_mitigation_applied}")
        
        # Interpret solution
        print("\nOptimal resource allocation:")
        selected_sites = []
        for i, param in enumerate(result.optimal_parameters):
            if i < len(locations):
                selection_prob = abs(param)
                selected = selection_prob > 0.5
                print(f"  {locations[i]}: {'SELECTED' if selected else 'not selected'} (prob: {selection_prob:.3f})")
                if selected:
                    selected_sites.append(locations[i])
                    
        print(f"\nRecommended drilling sites: {len(selected_sites)}")
        for site in selected_sites:
            print(f"  - {site}")
            
        # Calculate total cost
        binary_solution = (np.abs(result.optimal_parameters[:len(locations)]) > 0.5).astype(int)
        total_cost = binary_solution.T @ cost_matrix @ binary_solution
        print(f"\nEstimated total cost: {total_cost:.2f}")
        
    except Exception as e:
        print(f"Error in quantum optimization: {e}")
        
    print("\nQuantum optimization example completed!")

async def example_quantum_cryptography():
    """Example: Quantum cryptography for geological data security"""
    print("\n" + "="*60)
    print("QUANTUM CRYPTOGRAPHY EXAMPLE")
    print("="*60)
    
    # Load configuration
    config = load_quantum_config()
    
    # Initialize quantum system
    quantum_config = {
        'key_length': config['cryptography']['key_length'],
        'cloud_services': {}
    }
    
    quantum_system = QuantumComputingIntegration(quantum_config)
    await quantum_system.initialize()
    
    # Sample geological survey data (sensitive information)
    survey_reports = [
        {
            'title': 'Confidential Mineral Survey Report',
            'data': b"""
CONFIDENTIAL GEOLOGICAL SURVEY REPORT
=====================================

Survey ID: GEO-2024-001
Location: 40.7128° N, 74.0060° W
Date: 2024-01-15

FINDINGS:
- High-grade copper ore deposits identified
- Estimated reserves: 2.5 million tons
- Ore grade: 1.2% Cu average
- Economic viability: Highly profitable
- Environmental impact: Moderate

RECOMMENDATIONS:
- Proceed with detailed exploration
- Conduct environmental impact assessment
- Secure mining permits
- Estimated project value: $500M

CONFIDENTIAL - DO NOT DISTRIBUTE
            """.strip()
        },
        {
            'title': 'Volcanic Hazard Assessment',
            'data': b"""
VOLCANIC HAZARD ASSESSMENT - RESTRICTED
======================================

Volcano: Mount Vesuvius
Assessment Date: 2024-01-20
Classification: CONFIDENTIAL

CURRENT STATUS:
- Alert Level: Yellow (Advisory)
- Seismic Activity: Elevated
- Gas Emissions: Increasing CO2/SO2
- Ground Deformation: 2cm uplift detected

RISK ANALYSIS:
- Eruption probability (6 months): 15%
- Population at risk: 600,000
- Economic impact estimate: $2.1B
- Evacuation time required: 72 hours

MONITORING RECOMMENDATIONS:
- Increase seismic station density
- Deploy additional gas sensors
- Enhance satellite monitoring
- Update evacuation plans

RESTRICTED ACCESS - EMERGENCY MANAGEMENT ONLY
            """.strip()
        }
    ]
    
    print("Securing geological survey data with quantum cryptography...")
    
    encrypted_reports = []
    
    for i, report in enumerate(survey_reports):
        print(f"\nProcessing report {i+1}: {report['title']}")
        
        try:
            # Encrypt data using quantum cryptography
            encrypted_data, key_id = await quantum_system.secure_geological_data(
                report['data'], 
                security_level='high'
            )
            
            print(f"  Original size: {len(report['data'])} bytes")
            print(f"  Encrypted size: {len(encrypted_data)} bytes")
            print(f"  Quantum key ID: {key_id}")
            
            # Verify encryption worked
            if encrypted_data != report['data']:
                print(f"  ✓ Data successfully encrypted")
            else:
                print(f"  ✗ Encryption may have failed")
                
            # Store encrypted report info
            encrypted_reports.append({
                'title': report['title'],
                'original_data': report['data'],
                'encrypted_data': encrypted_data,
                'key_id': key_id
            })
            
            # Get key information
            key_info = quantum_system.quantum_crypto.quantum_keys[key_id]
            print(f"  Key security level: {key_info.security_level}")
            print(f"  Key generation time: {key_info.generation_timestamp}")
            print(f"  Key expiry time: {key_info.expiry_timestamp}")
            
        except Exception as e:
            print(f"  Error encrypting report: {e}")
            
    # Demonstrate decryption
    print(f"\nDemonstrating secure decryption...")
    
    for i, encrypted_report in enumerate(encrypted_reports):
        print(f"\nDecrypting report {i+1}: {encrypted_report['title']}")
        
        try:
            # Get key for decryption
            key_id = encrypted_report['key_id']
            key_info = quantum_system.quantum_crypto.quantum_keys[key_id]
            
            # Decrypt data
            decrypted_data = quantum_system.quantum_crypto.decrypt_geological_data(
                encrypted_report['encrypted_data'],
                key_id,
                key_info.classical_verification
            )
            
            # Verify decryption
            if decrypted_data == encrypted_report['original_data']:
                print(f"  ✓ Data successfully decrypted and verified")
                print(f"  Decrypted size: {len(decrypted_data)} bytes")
                
                # Show first few lines of decrypted data
                preview = decrypted_data.decode('utf-8').split('\n')[:3]
                print(f"  Preview:")
                for line in preview:
                    print(f"    {line}")
                print(f"    ... (truncated)")
                
            else:
                print(f"  ✗ Decryption verification failed")
                
        except Exception as e:
            print(f"  Error decrypting report: {e}")
            
    # Security analysis
    print(f"\nQuantum cryptography security analysis:")
    total_keys = len(quantum_system.quantum_crypto.quantum_keys)
    print(f"  Total quantum keys generated: {total_keys}")
    print(f"  Key length: {quantum_system.quantum_crypto.key_length} bits")
    print(f"  Security level: High (quantum-resistant)")
    print(f"  Key distribution protocol: BB84 (simulated)")
    
    print("\nQuantum cryptography example completed!")

async def example_performance_benchmarking():
    """Example: Quantum computing performance benchmarking"""
    print("\n" + "="*60)
    print("QUANTUM PERFORMANCE BENCHMARKING EXAMPLE")
    print("="*60)
    
    # Load configuration
    config = load_quantum_config()
    
    # Initialize quantum system with full configuration
    quantum_config = {
        'num_qubits': config['circuit']['num_qubits'],
        'circuit_depth': config['circuit']['circuit_depth'],
        'ml_qubits': config['machine_learning']['num_qubits'],
        'ml_layers': config['machine_learning']['num_layers'],
        'opt_qubits': config['optimization']['num_qubits'],
        'key_length': config['cryptography']['key_length'],
        'max_qubits': config['performance']['max_qubits'],
        'cloud_services': {}
    }
    
    quantum_system = QuantumComputingIntegration(quantum_config)
    await quantum_system.initialize()
    
    print("Running comprehensive quantum performance benchmarks...")
    
    try:
        # Run benchmarks
        benchmarks = await quantum_system.benchmark_quantum_performance()
        
        print(f"\nBenchmark Results:")
        print(f"{'='*50}")
        
        # Circuit building benchmarks
        circuit_benchmarks = {k: v for k, v in benchmarks.items() if 'circuit_build' in k}
        if circuit_benchmarks:
            print(f"\nQuantum Circuit Building Performance:")
            for test_name, metrics in circuit_benchmarks.items():
                qubits = test_name.split('_')[-1]
                print(f"  {qubits} qubits:")
                print(f"    Build time: {metrics.get('time_seconds', 0):.4f} seconds")
                print(f"    Circuit depth: {metrics.get('circuit_depth', 0)}")
                print(f"    Parameters: {metrics.get('num_parameters', 0)}")
                
        # Machine learning benchmarks
        if 'quantum_ml' in benchmarks:
            ml_metrics = benchmarks['quantum_ml']
            print(f"\nQuantum Machine Learning Performance:")
            if 'error' in ml_metrics:
                print(f"  Error: {ml_metrics['error']}")
            else:
                print(f"  Training time: {ml_metrics.get('training_time', 0):.4f} seconds")
                print(f"  Training accuracy: {ml_metrics.get('accuracy', 0):.3f}")
                print(f"  Quantum advantage: {ml_metrics.get('quantum_advantage', 0):.3f}")
                
        # Optimization benchmarks
        if 'quantum_optimization' in benchmarks:
            opt_metrics = benchmarks['quantum_optimization']
            print(f"\nQuantum Optimization Performance:")
            if 'error' in opt_metrics:
                print(f"  Error: {opt_metrics['error']}")
            else:
                print(f"  Optimization time: {opt_metrics.get('optimization_time', 0):.4f} seconds")
                print(f"  Optimal value: {opt_metrics.get('optimal_value', 0):.3f}")
                print(f"  Quantum advantage: {opt_metrics.get('quantum_advantage', False)}")
                
        # System status
        print(f"\nSystem Status:")
        status = await quantum_system.get_quantum_status()
        print(f"  Initialized: {status['initialized']}")
        print(f"  Quantum libraries available: {status['quantum_libraries_available']}")
        print(f"  Cloud services available: {status['cloud_services_available']}")
        print(f"  Active cloud services: {status['active_cloud_services']}")
        
        # Configuration summary
        print(f"\nConfiguration Summary:")
        config_summary = status['configuration']
        print(f"  Qubits: {config_summary['num_qubits']}")
        print(f"  Circuit depth: {config_summary['circuit_depth']}")
        print(f"  ML qubits: {config_summary['ml_qubits']}")
        print(f"  Key length: {config_summary['key_length']}")
        
    except Exception as e:
        print(f"Error in performance benchmarking: {e}")
        
    print("\nPerformance benchmarking completed!")

async def main():
    """Main function to run all quantum computing examples"""
    print("QUANTUM COMPUTING INTEGRATION EXAMPLES")
    print("=" * 80)
    print("This script demonstrates quantum computing capabilities for geological applications")
    print("including simulation, machine learning, optimization, and cryptography.")
    print("=" * 80)
    
    try:
        # Run all examples
        await example_quantum_geological_simulation()
        await example_quantum_machine_learning()
        await example_quantum_optimization()
        await example_quantum_cryptography()
        await example_performance_benchmarking()
        
        print("\n" + "="*80)
        print("ALL QUANTUM COMPUTING EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")
        logger.exception("Error in main execution")

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())