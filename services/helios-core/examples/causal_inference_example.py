"""
Example usage of the Causal Inference Engine for Geological Data

This example demonstrates how to use the causal inference engine to discover
causal relationships in geological data, perform intervention analysis,
and generate testable hypotheses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Import the causal inference engine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.causal_inference_engine import (
    CausalInferenceEngine,
    CausalAlgorithm,
    GeologicalDomainKnowledge
)


def create_synthetic_volcanic_data(n_samples=500):
    """
    Create synthetic volcanic monitoring data with known causal relationships
    
    This simulates a volcanic system with the following causal structure:
    - Magma intrusion -> Ground deformation + Gas emissions
    - Temperature increase -> Gas emissions + Thermal anomalies
    - Stress changes -> Seismic activity
    - Gas emissions -> Atmospheric changes
    """
    np.random.seed(42)
    
    # Primary causal variables
    magma_intrusion_rate = np.random.gamma(2, 2, n_samples)  # Rate of magma intrusion
    deep_temperature = np.random.normal(1200, 150, n_samples)  # Deep temperature in Celsius
    tectonic_stress = np.random.normal(0, 1, n_samples)  # Normalized stress changes
    
    # Intermediate variables (caused by primary variables)
    ground_deformation = (
        0.8 * magma_intrusion_rate + 
        0.1 * deep_temperature / 100 + 
        np.random.normal(0, 0.5, n_samples)
    )
    
    gas_emissions = (
        0.6 * magma_intrusion_rate + 
        0.4 * (deep_temperature - 1000) / 100 + 
        np.random.normal(0, 1, n_samples)
    )
    
    seismic_activity = (
        0.7 * tectonic_stress + 
        0.3 * magma_intrusion_rate + 
        np.random.normal(0, 0.3, n_samples)
    )
    
    thermal_anomalies = (
        0.5 * deep_temperature / 100 + 
        0.3 * magma_intrusion_rate + 
        np.random.normal(0, 0.4, n_samples)
    )
    
    # Secondary effects
    atmospheric_changes = (
        0.8 * gas_emissions + 
        np.random.normal(0, 0.2, n_samples)
    )
    
    # Confounding variables
    elevation = np.random.uniform(1000, 4000, n_samples)
    weather_conditions = np.random.normal(0, 1, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'magma_intrusion_rate': magma_intrusion_rate,
        'deep_temperature': deep_temperature,
        'tectonic_stress': tectonic_stress,
        'ground_deformation': ground_deformation,
        'gas_emissions': gas_emissions,
        'seismic_activity': seismic_activity,
        'thermal_anomalies': thermal_anomalies,
        'atmospheric_changes': atmospheric_changes,
        'elevation': elevation,
        'weather_conditions': weather_conditions
    })
    
    return data


def create_domain_knowledge():
    """Create geological domain knowledge for volcanic systems"""
    return GeologicalDomainKnowledge(
        known_causal_relationships={
            "magma_intrusion_rate": ["ground_deformation", "gas_emissions", "thermal_anomalies"],
            "deep_temperature": ["gas_emissions", "thermal_anomalies"],
            "tectonic_stress": ["seismic_activity"],
            "gas_emissions": ["atmospheric_changes"]
        },
        impossible_relationships={
            "ground_deformation": ["magma_intrusion_rate"],
            "gas_emissions": ["magma_intrusion_rate", "deep_temperature"],
            "seismic_activity": ["tectonic_stress"],
            "atmospheric_changes": ["gas_emissions"]
        },
        temporal_constraints={
            "magma_intrusion_rate": {"before": "ground_deformation"},
            "deep_temperature": {"before": "thermal_anomalies"},
            "tectonic_stress": {"before": "seismic_activity"},
            "gas_emissions": {"before": "atmospheric_changes"}
        },
        physical_constraints={
            "temperature": {"min": 273, "max": 2000, "unit": "K"},
            "pressure": {"min": 0, "max": 1e9, "unit": "Pa"},
            "deformation": {"min": -10, "max": 10, "unit": "cm"}
        },
        geological_process_hierarchies={
            "volcanic_processes": [
                "magma_generation", "magma_transport", "degassing", "eruption"
            ],
            "monitoring_signals": [
                "seismic", "deformation", "geochemical", "thermal"
            ]
        }
    )


def demonstrate_causal_discovery():
    """Demonstrate causal discovery with different algorithms"""
    print("=== Causal Discovery Demonstration ===\n")
    
    # Create synthetic data
    print("Creating synthetic volcanic monitoring data...")
    data = create_synthetic_volcanic_data(n_samples=300)
    print(f"Created dataset with {len(data)} samples and {len(data.columns)} variables")
    print(f"Variables: {list(data.columns)}\n")
    
    # Create domain knowledge
    domain_knowledge = create_domain_knowledge()
    
    # Initialize causal inference engine
    engine = CausalInferenceEngine(domain_knowledge=domain_knowledge)
    
    # Test different algorithms
    algorithms = [CausalAlgorithm.PC, CausalAlgorithm.FCI, CausalAlgorithm.GES]
    results = {}
    
    for algorithm in algorithms:
        print(f"Running {algorithm.value.upper()} algorithm...")
        
        try:
            causal_graph = engine.discover_causal_relationships(
                data=data,
                algorithm=algorithm,
                confidence_threshold=0.05
            )
            
            results[algorithm.value] = causal_graph
            
            print(f"  Found {len(causal_graph.edges)} causal relationships")
            print(f"  Nodes: {len(causal_graph.nodes)}")
            
            # Display discovered relationships
            if causal_graph.edges:
                print("  Top relationships:")
                sorted_edges = sorted(causal_graph.edges, key=lambda x: x.strength, reverse=True)
                for edge in sorted_edges[:5]:  # Top 5
                    print(f"    {edge.source} -> {edge.target} "
                          f"(strength: {edge.strength:.3f}, confidence: {edge.confidence:.3f})")
            print()
            
        except Exception as e:
            print(f"  Error with {algorithm.value}: {str(e)}\n")
    
    return results, data, engine


def demonstrate_intervention_analysis(causal_graph, data, engine):
    """Demonstrate causal intervention analysis"""
    print("=== Intervention Analysis Demonstration ===\n")
    
    if not causal_graph.edges:
        print("No causal relationships found for intervention analysis.\n")
        return
    
    # Find variables with outgoing edges for intervention
    intervention_candidates = list(set([edge.source for edge in causal_graph.edges]))
    
    if not intervention_candidates:
        print("No suitable intervention variables found.\n")
        return
    
    # Perform intervention on the first candidate
    intervention_var = intervention_candidates[0]
    print(f"Performing intervention analysis on: {intervention_var}")
    
    # Calculate intervention value (e.g., increase by 1 standard deviation)
    current_mean = data[intervention_var].mean()
    current_std = data[intervention_var].std()
    intervention_value = current_mean + current_std
    
    print(f"Setting {intervention_var} from {current_mean:.3f} to {intervention_value:.3f}")
    print(f"(increase of {current_std:.3f} units)\n")
    
    try:
        intervention_result = engine.perform_intervention_analysis(
            causal_graph=causal_graph,
            data=data,
            intervention_variable=intervention_var,
            intervention_value=intervention_value
        )
        
        print("Intervention Results:")
        print(f"Target variables: {intervention_result.target_variables}")
        print("\nCausal Effects:")
        for target, effect in intervention_result.causal_effects.items():
            ci = intervention_result.confidence_intervals[target]
            p_val = intervention_result.statistical_significance[target]
            print(f"  {target}: {effect:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}], p={p_val:.3f})")
        
        print(f"\nGeological Interpretation:")
        print(intervention_result.geological_interpretation)
        
    except Exception as e:
        print(f"Error in intervention analysis: {str(e)}")
    
    print()


def demonstrate_hypothesis_generation(causal_graph, engine):
    """Demonstrate hypothesis generation from causal graph"""
    print("=== Hypothesis Generation Demonstration ===\n")
    
    if not causal_graph.edges:
        print("No causal relationships found for hypothesis generation.\n")
        return
    
    try:
        hypotheses = engine.generate_hypothesis_from_graph(causal_graph)
        
        print(f"Generated {len(hypotheses)} testable hypotheses:\n")
        
        for i, hypothesis in enumerate(hypotheses, 1):
            print(f"Hypothesis {i}:")
            print(f"{hypothesis}\n")
            print("-" * 80 + "\n")
    
    except Exception as e:
        print(f"Error in hypothesis generation: {str(e)}\n")


def demonstrate_validation(causal_graph, original_data, engine):
    """Demonstrate causal graph validation with independent data"""
    print("=== Causal Graph Validation Demonstration ===\n")
    
    if not causal_graph.edges:
        print("No causal relationships to validate.\n")
        return
    
    # Create validation data (similar structure but different samples)
    print("Creating independent validation dataset...")
    validation_data = create_synthetic_volcanic_data(n_samples=200)
    
    try:
        validation_results = engine.validate_causal_relationships(
            causal_graph=causal_graph,
            validation_data=validation_data
        )
        
        print("Validation Results:")
        print(f"  Total edges: {validation_results['total_edges']}")
        print(f"  Validated edges: {validation_results['validated_edges']}")
        print(f"  Failed edges: {validation_results['failed_edges']}")
        print(f"  Overall accuracy: {validation_results['overall_accuracy']:.3f}")
        print(f"  Domain knowledge consistency: {validation_results['domain_knowledge_consistency']:.3f}")
        
        print("\nDetailed Edge Validation:")
        for edge_name, validation in validation_results['edge_validations'].items():
            status = "✓ Valid" if validation['valid'] else "✗ Invalid"
            print(f"  {edge_name}: {status} "
                  f"(strength: {validation['original_strength']:.3f}, "
                  f"confidence: {validation['original_confidence']:.3f})")
    
    except Exception as e:
        print(f"Error in validation: {str(e)}")
    
    print()


def visualize_causal_graph(causal_graph, title="Causal Graph"):
    """Visualize the discovered causal graph"""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        if not causal_graph.edges:
            print("No edges to visualize.\n")
            return
        
        # Convert to NetworkX graph
        G = causal_graph.to_networkx()
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=2000, alpha=0.7)
        
        # Draw edges with varying thickness based on strength
        edges = G.edges(data=True)
        for edge in edges:
            strength = edge[2].get('strength', 0.5)
            nx.draw_networkx_edges(G, pos, [(edge[0], edge[1])], 
                                  width=strength * 5, alpha=0.6, 
                                  edge_color='red', arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title(title, size=16, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save the plot
        filename = f"causal_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Causal graph visualization saved as: {filename}")
        plt.show()
        
    except ImportError:
        print("NetworkX and matplotlib required for visualization")
    except Exception as e:
        print(f"Error in visualization: {str(e)}")


def export_results(results, filename_prefix="causal_analysis"):
    """Export analysis results to JSON files"""
    import numpy as np
    
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for algorithm, causal_graph in results.items():
        filename = f"{filename_prefix}_{algorithm}_{timestamp}.json"
        
        try:
            # Create export data
            export_data = {
                "algorithm": algorithm,
                "timestamp": causal_graph.creation_timestamp.isoformat(),
                "nodes": causal_graph.nodes,
                "edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "edge_type": edge.edge_type.value,
                        "strength": float(edge.strength),
                        "confidence": float(edge.confidence),
                        "p_value": float(edge.p_value),
                        "geological_interpretation": edge.geological_interpretation
                    }
                    for edge in causal_graph.edges
                ],
                "confidence_threshold": float(causal_graph.confidence_threshold),
                "domain_knowledge": convert_numpy_types(causal_graph.geological_domain_knowledge)
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Results exported to: {filename}")
            
        except Exception as e:
            print(f"Error exporting {algorithm} results: {str(e)}")


def main():
    """Main demonstration function"""
    print("Geological Causal Inference Engine Demonstration")
    print("=" * 60 + "\n")
    
    # Demonstrate causal discovery
    results, data, engine = demonstrate_causal_discovery()
    
    if not results:
        print("No successful causal discovery results to demonstrate further analysis.")
        return
    
    # Use the best result (PC algorithm typically works well)
    best_algorithm = 'pc' if 'pc' in results else list(results.keys())[0]
    best_graph = results[best_algorithm]
    
    print(f"Using {best_algorithm.upper()} results for further analysis...\n")
    
    # Demonstrate intervention analysis
    demonstrate_intervention_analysis(best_graph, data, engine)
    
    # Demonstrate hypothesis generation
    demonstrate_hypothesis_generation(best_graph, engine)
    
    # Demonstrate validation
    demonstrate_validation(best_graph, data, engine)
    
    # Visualize results
    visualize_causal_graph(best_graph, f"Causal Graph ({best_algorithm.upper()} Algorithm)")
    
    # Export results
    export_results(results)
    
    print("\nDemonstration completed successfully!")
    print("Check the generated files for detailed results and visualizations.")


if __name__ == "__main__":
    main()