"""
Demonstration of supereruption analysis and large caldera system modeling.

This example shows how the geological network representation learning system
can identify and analyze supereruption potential in large caldera systems.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from datetime import datetime
from core.geological_network_representation import (
    GeologicalNetworkRepresentationEngine,
    SupereruptionAnalyzer,
    SupereruptionSignature,
    GeologicalGraphData
)

def create_yellowstone_like_system():
    """Create a Yellowstone-like caldera system for analysis."""
    print("🌋 Creating Yellowstone-like Caldera System...")
    
    # Simulate a large caldera system with multiple components
    num_nodes = 25  # Caldera, magma chambers, monitoring stations, etc.
    num_edges = 40  # Relationships between components
    
    # Node features representing different geological entities
    node_features = torch.randn(num_nodes, 64)
    
    # Edge connectivity (relationships between entities)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Node types: 0=caldera, 1=upper_chamber, 2=lower_chamber, 3=monitoring_station, etc.
    node_types = torch.tensor([
        0,  # Main caldera
        1, 1, 1,  # Upper magma chambers
        2, 2,  # Lower magma chambers
        3, 3, 3, 3, 3,  # Monitoring stations
        4, 4, 4,  # Thermal features
        5, 5,  # Gas emission sites
        6, 6, 6,  # Deformation sources
        7, 7, 7, 7, 7, 7  # Seismic sources
    ])
    
    # Edge types: 0=spatial, 1=causal, 2=temporal, 3=structural
    edge_types = torch.randint(0, 4, (num_edges,))
    
    # Spatial coordinates (lat, lon, depth in km)
    # Centered around Yellowstone coordinates
    base_lat, base_lon = 44.4, -110.6
    spatial_coords = torch.tensor([
        [base_lat, base_lon, 0.0],  # Main caldera at surface
        [base_lat + 0.1, base_lon + 0.1, -8.0],  # Upper chambers
        [base_lat - 0.1, base_lon - 0.1, -8.5],
        [base_lat + 0.05, base_lon - 0.05, -9.0],
        [base_lat, base_lon, -15.0],  # Lower chambers
        [base_lat + 0.2, base_lon - 0.2, -16.0],
        # Monitoring stations around the caldera
        [base_lat + 0.3, base_lon + 0.3, 2.0],
        [base_lat - 0.3, base_lon - 0.3, 2.1],
        [base_lat + 0.3, base_lon - 0.3, 2.2],
        [base_lat - 0.3, base_lon + 0.3, 1.9],
        [base_lat, base_lon + 0.4, 2.0],
        # Thermal features
        [base_lat + 0.1, base_lon, 0.1],
        [base_lat - 0.1, base_lon, 0.1],
        [base_lat, base_lon + 0.1, 0.1],
        # Gas emission sites
        [base_lat + 0.05, base_lon + 0.05, 0.0],
        [base_lat - 0.05, base_lon - 0.05, 0.0],
        # Deformation sources
        [base_lat + 0.2, base_lon, -5.0],
        [base_lat - 0.2, base_lon, -5.0],
        [base_lat, base_lon + 0.2, -6.0],
        # Seismic sources
        [base_lat + 0.15, base_lon + 0.15, -3.0],
        [base_lat - 0.15, base_lon - 0.15, -3.0],
        [base_lat + 0.15, base_lon - 0.15, -4.0],
        [base_lat - 0.15, base_lon + 0.15, -4.0],
        [base_lat, base_lon, -7.0],
        [base_lat + 0.1, base_lon - 0.1, -8.0]
    ])
    
    return GeologicalGraphData(
        node_features=node_features,
        edge_index=edge_index,
        node_types=node_types,
        edge_types=edge_types,
        spatial_coords=spatial_coords
    )

def create_toba_like_system():
    """Create a Toba-like supereruption system for comparison."""
    print("🌋 Creating Toba-like Supereruption System...")
    
    num_nodes = 20
    num_edges = 35
    
    node_features = torch.randn(num_nodes, 64)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Toba system with massive magma chamber
    node_types = torch.tensor([
        0,  # Main caldera (much larger than Yellowstone)
        1, 1,  # Massive upper chambers
        2, 2, 2,  # Multiple lower chambers
        3, 3, 3, 3,  # Monitoring stations
        4, 4,  # Thermal features
        5, 5,  # Gas emission sites
        6, 6, 6,  # Deformation sources
        7, 7, 7  # Seismic sources
    ])
    
    edge_types = torch.randint(0, 4, (num_edges,))
    
    # Toba coordinates
    base_lat, base_lon = 2.7, 98.8
    spatial_coords = torch.tensor([
        [base_lat, base_lon, 0.0],  # Main caldera
        [base_lat + 0.2, base_lon + 0.2, -10.0],  # Massive upper chambers
        [base_lat - 0.2, base_lon - 0.2, -12.0],
        [base_lat, base_lon, -20.0],  # Deep lower chambers
        [base_lat + 0.3, base_lon, -22.0],
        [base_lat - 0.3, base_lon, -25.0],
        # Monitoring and other features
        [base_lat + 0.5, base_lon + 0.5, 1.0],
        [base_lat - 0.5, base_lon - 0.5, 1.0],
        [base_lat + 0.5, base_lon - 0.5, 1.0],
        [base_lat - 0.5, base_lon + 0.5, 1.0],
        [base_lat + 0.1, base_lon, 0.0],
        [base_lat - 0.1, base_lon, 0.0],
        [base_lat + 0.05, base_lon + 0.05, 0.0],
        [base_lat - 0.05, base_lon - 0.05, 0.0],
        [base_lat + 0.3, base_lon, -8.0],
        [base_lat - 0.3, base_lon, -8.0],
        [base_lat, base_lon + 0.3, -9.0],
        [base_lat + 0.2, base_lon + 0.2, -5.0],
        [base_lat - 0.2, base_lon - 0.2, -5.0],
        [base_lat, base_lon, -15.0]
    ])
    
    return GeologicalGraphData(
        node_features=node_features,
        edge_index=edge_index,
        node_types=node_types,
        edge_types=edge_types,
        spatial_coords=spatial_coords
    )

def analyze_supereruption_systems():
    """Analyze multiple supereruption systems and compare them."""
    print("\n" + "="*60)
    print("🔬 SUPERERUPTION ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Initialize the representation engine
    engine = GeologicalNetworkRepresentationEngine(
        node_embedding_dim=256,
        edge_embedding_dim=128,
        graph_embedding_dim=512
    )
    
    # Create different caldera systems
    yellowstone_system = create_yellowstone_like_system()
    toba_system = create_toba_like_system()
    
    print("\n📊 ANALYZING YELLOWSTONE-LIKE SYSTEM")
    print("-" * 40)
    
    # Process Yellowstone-like system
    yellowstone_results = engine.process_geological_network(
        yellowstone_system, include_supereruption_analysis=True
    )
    
    print(f"System Classification: {yellowstone_results['system_classification']['system_type']}")
    print(f"Classification Confidence: {yellowstone_results['system_classification']['confidence']:.3f}")
    
    if 'supereruption_analysis' in yellowstone_results:
        analysis = yellowstone_results['supereruption_analysis']
        print(f"Number of Caldera Systems: {analysis['num_caldera_systems']}")
        print(f"Overall Supereruption Risk: {analysis['overall_supereruption_risk']:.3f}")
        
        for i, caldera_analysis in enumerate(analysis['caldera_analyses']):
            signature = caldera_analysis['supereruption_signature']
            chamber_state = caldera_analysis['chamber_state']
            
            print(f"\n  Caldera Component {i+1}:")
            print(f"    Eruption Probability: {signature.eruption_probability:.3f}")
            print(f"    Magma Chamber Volume: {signature.magma_chamber_volume:.1f} km³")
            print(f"    Chamber State: {chamber_state['chamber_state']}")
            print(f"    Explosive Potential: {chamber_state['explosive_potential']}")
            print(f"    Predicted Eruption Style: {chamber_state['eruption_style_prediction']}")
    
    print("\n📊 ANALYZING TOBA-LIKE SYSTEM")
    print("-" * 40)
    
    # Process Toba-like system
    toba_results = engine.process_geological_network(
        toba_system, include_supereruption_analysis=True
    )
    
    print(f"System Classification: {toba_results['system_classification']['system_type']}")
    print(f"Classification Confidence: {toba_results['system_classification']['confidence']:.3f}")
    
    if 'supereruption_analysis' in toba_results:
        analysis = toba_results['supereruption_analysis']
        print(f"Number of Caldera Systems: {analysis['num_caldera_systems']}")
        print(f"Overall Supereruption Risk: {analysis['overall_supereruption_risk']:.3f}")
    
    print("\n🔍 DETAILED SUPERERUPTION ANALYSIS")
    print("-" * 40)
    
    # Detailed analysis using SupereruptionAnalyzer
    analyzer = SupereruptionAnalyzer()
    
    # Simulate current monitoring data for Yellowstone-like system
    yellowstone_monitoring = {
        'deformation_rate': 15.0,  # mm/year - moderate uplift
        'seismic_swarm_intensity': 80.0,  # events/day - elevated
        'gas_emission_anomaly': 1.8,  # relative to baseline - slightly elevated
        'thermal_anomaly_extent': 12.0,  # km² - significant thermal activity
        'ground_uplift_pattern': [2, 4, 6, 8, 10, 8, 6, 4, 2, 1],
        'geochemical_indicators': {
            'he3_he4_ratio': 0.08,  # Mantle signature
            'co2_so2_ratio': 0.4    # Magma degassing
        }
    }
    
    yellowstone_caldera = {
        'system_id': 'yellowstone_caldera',
        'magma_chamber_volume': 1000.0,  # km³ - supereruption threshold
        'caldera_diameter': 45.0,  # km
    }
    
    signature = analyzer.analyze_supereruption_potential(
        yellowstone_caldera, yellowstone_monitoring
    )
    
    print(f"Yellowstone-like System Analysis:")
    print(f"  Caldera ID: {signature.caldera_id}")
    print(f"  Magma Chamber Volume: {signature.magma_chamber_volume:.1f} km³")
    print(f"  Eruption Probability: {signature.eruption_probability:.3f}")
    print(f"  Deformation Rate: {signature.crustal_deformation_rate:.1f} mm/year")
    print(f"  Seismic Intensity: {signature.seismic_swarm_intensity:.1f} events/day")
    print(f"  Gas Emission Anomaly: {signature.gas_emission_anomaly:.1f}x baseline")
    print(f"  Thermal Anomaly Extent: {signature.thermal_anomaly_extent:.1f} km²")
    
    if signature.time_to_eruption_estimate:
        print(f"  Estimated Time to Eruption: {signature.time_to_eruption_estimate:.1f} years")
    
    # Compare with historical supereruptions
    print(f"\n📚 COMPARISON WITH HISTORICAL SUPERERUPTIONS")
    print("-" * 50)
    
    comparisons = analyzer.compare_with_historical_supereruptions(signature)
    
    for i, comparison in enumerate(comparisons[:3]):  # Top 3 most similar
        print(f"\n{i+1}. {comparison['historical_eruption']}")
        print(f"   Similarity Score: {comparison['similarity']:.3f}")
        print(f"   Volume Ratio: {comparison['volume_comparison']['ratio']:.2f}")
        print(f"   Historical Volume: {comparison['volume_comparison']['historical']:.0f} km³")
        print(f"   Key Implications:")
        for implication in comparison['implications'][:2]:  # Top 2 implications
            print(f"     • {implication}")
    
    print("\n🎯 MAGMA CHAMBER DETAILED ANALYSIS")
    print("-" * 40)
    
    # Analyze individual magma chambers
    chamber_embeddings = yellowstone_results['node_embeddings'][:3]  # First 3 nodes (chambers)
    
    for i, chamber_embedding in enumerate(chamber_embeddings):
        chamber_monitoring = {
            'temperature': 800 + i * 50,  # Varying temperatures
            'pressure': 150 + i * 25,     # Varying pressures
            'crystallinity': 0.3 + i * 0.1,  # Varying crystallinity
            'volatile_content': {
                'H2O': 3.0 + i * 0.5,  # Water content
                'CO2': 800 + i * 100   # CO2 content
            }
        }
        
        chamber_state = analyzer.identify_magma_chamber_state(
            chamber_embedding, chamber_monitoring
        )
        
        print(f"\nChamber {i+1} Analysis:")
        print(f"  State: {chamber_state['chamber_state']}")
        print(f"  Magma Mobility: {chamber_state['magma_mobility']}")
        print(f"  Connectivity Score: {chamber_state['connectivity_score']:.3f}")
        print(f"  Explosive Potential: {chamber_state['explosive_potential']}")
        print(f"  Temperature: {chamber_state['temperature']:.0f}°C")
        print(f"  Crystallinity: {chamber_state['crystallinity']:.1f}")
        print(f"  Predicted Eruption Style: {chamber_state['eruption_style_prediction']}")
    
    print("\n🌐 SYSTEM SIMILARITY ANALYSIS")
    print("-" * 40)
    
    # Add systems to similarity database
    yellowstone_embedding = yellowstone_results['graph_embedding'].detach().cpu().numpy()[0]
    toba_embedding = toba_results['graph_embedding'].detach().cpu().numpy()[0]
    
    engine.similarity_search.add_geological_system(
        'yellowstone_type',
        yellowstone_embedding,
        {
            'system_type': 'large_caldera_complex',
            'location': [44.4, -110.6],
            'magma_chamber_volume': 1000.0,
            'caldera_diameter': 45.0,
            'last_major_eruption': 640000,  # years ago
            'eruption_frequency': 0.000002,  # very low frequency
            'max_vei': 8
        }
    )
    
    engine.similarity_search.add_geological_system(
        'toba_type',
        toba_embedding,
        {
            'system_type': 'supereruption_caldera',
            'location': [2.7, 98.8],
            'magma_chamber_volume': 2800.0,
            'caldera_diameter': 100.0,
            'last_major_eruption': 74000,  # years ago
            'eruption_frequency': 0.000001,  # extremely low frequency
            'max_vei': 8
        }
    )
    
    # Find similar systems
    similar_to_yellowstone = engine.find_similar_geological_systems(yellowstone_system, k=2)
    
    print("Systems similar to Yellowstone-like caldera:")
    for system in similar_to_yellowstone:
        similarity_score = float(system['similarity']) if hasattr(system['similarity'], 'item') else system['similarity']
        print(f"  • {system['system_id']}: Similarity = {similarity_score:.3f}")
        metadata = system['metadata']
        print(f"    Type: {metadata['system_type']}")
        print(f"    Chamber Volume: {metadata.get('magma_chamber_volume', 'N/A')} km³")
        print(f"    Max VEI: {metadata.get('max_vei', 'N/A')}")
    
    print("\n" + "="*60)
    print("✅ SUPERERUPTION ANALYSIS COMPLETE")
    print("="*60)
    
    print("\n🔑 KEY INSIGHTS:")
    print("• Advanced neural network embeddings capture complex geological relationships")
    print("• Supereruption detection integrates multiple monitoring signals")
    print("• Magma chamber state analysis provides detailed eruption style predictions")
    print("• Historical comparisons enable risk assessment and timeline estimation")
    print("• Similarity search identifies analogous systems for comparative analysis")
    print("• Large caldera systems require specialized monitoring and analysis approaches")

if __name__ == "__main__":
    analyze_supereruption_systems()