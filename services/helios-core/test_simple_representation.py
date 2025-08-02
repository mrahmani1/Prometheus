"""
Simple test script for geological network representation learning.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "core"))

import numpy as np
import torch
import torch.nn.functional as F

from core.geological_network_representation import (GeologicalEdgeEmbedding,
                                                    GeologicalGraphEmbedding,
                                                    GeologicalNodeEmbedding,
                                                    GeologicalSimilaritySearch,
                                                    SupereruptionAnalyzer)


def test_basic_functionality():
    """Test basic functionality without torch-geometric dependencies."""
    print("Testing Geological Network Representation Learning...")

    # Test node embedding
    print("1. Testing Node Embedding...")
    node_embedder = GeologicalNodeEmbedding(num_nodes=100, embedding_dim=256)

    batch_size = 10
    node_ids = torch.randint(0, 100, (batch_size,))
    entity_types = torch.randint(0, 10, (batch_size,))
    process_types = torch.randint(0, 15, (batch_size,))
    spatial_coords = torch.randn(batch_size, 3)

    node_results = node_embedder(
        node_ids=node_ids,
        entity_types=entity_types,
        process_types=process_types,
        spatial_coords=spatial_coords,
    )

    print(f"   Node embeddings shape: {node_results['node_embeddings'].shape}")
    print(
        f"   Supereruption scores shape: {node_results['supereruption_scores'].shape}"
    )
    print("   ✓ Node embedding test passed")

    # Test edge embedding
    print("2. Testing Edge Embedding...")
    edge_embedder = GeologicalEdgeEmbedding(embedding_dim=128)

    edge_batch_size = 20
    edge_types = torch.randint(0, 5, (edge_batch_size,))
    edge_distances = torch.randn(edge_batch_size)
    edge_strengths = torch.rand(edge_batch_size)

    edge_results = edge_embedder(
        edge_types=edge_types,
        edge_distances=edge_distances,
        edge_strengths=edge_strengths,
    )

    print(f"   Edge embeddings shape: {edge_results['edge_embeddings'].shape}")
    print(f"   Causal scores shape: {edge_results['causal_scores'].shape}")
    print("   ✓ Edge embedding test passed")

    # Test graph embedding
    print("3. Testing Graph Embedding...")
    graph_embedder = GeologicalGraphEmbedding(
        node_embedding_dim=256, edge_embedding_dim=128, graph_embedding_dim=512
    )

    num_nodes = 50
    num_edges = 100
    node_embeddings = torch.randn(num_nodes, 256)
    edge_embeddings = torch.randn(num_edges, 128)

    graph_results = graph_embedder(
        node_embeddings=node_embeddings, edge_embeddings=edge_embeddings
    )

    print(f"   Graph embedding shape: {graph_results['graph_embedding'].shape}")
    print(f"   System type logits shape: {graph_results['system_type_logits'].shape}")
    print("   ✓ Graph embedding test passed")

    # Test similarity search
    print("4. Testing Similarity Search...")
    similarity_search = GeologicalSimilaritySearch(
        embedding_dim=512, use_faiss=False  # Disable FAISS for this test
    )

    # Add some systems
    for i in range(5):
        embedding = np.random.randn(512)
        metadata = {"system_type": "volcanic_arc", "location": [45.0 + i, -122.0 + i]}
        similarity_search.add_geological_system(f"volcano_{i}", embedding, metadata)

    # Search for similar systems
    query_embedding = np.random.randn(512)
    similar_systems = similarity_search.find_similar_systems(
        query_embedding, k=3, min_similarity=0.0
    )

    print(f"   Found {len(similar_systems)} similar systems")
    print("   ✓ Similarity search test passed")

    # Test supereruption analyzer
    print("5. Testing Supereruption Analyzer...")
    analyzer = SupereruptionAnalyzer()

    caldera_system = {
        "system_id": "test_caldera",
        "magma_chamber_volume": 600.0,
        "caldera_diameter": 35.0,
    }

    monitoring_data = {
        "deformation_rate": 20.0,
        "seismic_swarm_intensity": 150.0,
        "gas_emission_anomaly": 2.5,
        "thermal_anomaly_extent": 8.0,
        "geochemical_indicators": {"he3_he4_ratio": 0.12, "co2_so2_ratio": 0.6},
    }

    signature = analyzer.analyze_supereruption_potential(
        caldera_system, monitoring_data
    )
    print(f"   Eruption probability: {signature.eruption_probability:.3f}")
    print(f"   Magma chamber volume: {signature.magma_chamber_volume} km³")
    print("   ✓ Supereruption analyzer test passed")

    # Test chamber state analysis
    print("6. Testing Magma Chamber State Analysis...")
    chamber_embedding = torch.randn(256)
    chamber_monitoring = {
        "temperature": 850.0,
        "pressure": 200.0,
        "crystallinity": 0.4,
        "volatile_content": {"H2O": 3.5, "CO2": 1000.0},
    }

    chamber_state = analyzer.identify_magma_chamber_state(
        chamber_embedding, chamber_monitoring
    )

    print(f"   Chamber state: {chamber_state['chamber_state']}")
    print(f"   Explosive potential: {chamber_state['explosive_potential']}")
    print(f"   Eruption style prediction: {chamber_state['eruption_style_prediction']}")
    print("   ✓ Chamber state analysis test passed")

    print("\n✅ All basic tests passed successfully!")
    print("\nKey Features Implemented:")
    print("- Node embeddings for geological entities and processes")
    print("- Edge embeddings for geological relationships")
    print("- Graph-level embeddings for system classification")
    print("- Similarity search for analogous geological systems")
    print("- Supereruption detection and analysis")
    print("- Magma chamber state identification")
    print("- Large caldera system analysis")

    return True

if __name__ == "__main__":
    test_basic_functionality()
