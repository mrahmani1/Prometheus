"""
Test suite for geological network representation learning.

Tests all components of the geological network representation system including
node embeddings, edge embeddings, graph-level embeddings, similarity search,
and supereruption analysis.
"""

import os
import sys

import numpy as np
import torch

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))

from geological_network_representation import (
    GeologicalEdgeEmbedding,
    GeologicalGraphEmbedding,
    GeologicalNetworkRepresentationEngine,
    GeologicalNodeEmbedding,
    GeologicalSimilaritySearch,
    SupereruptionAnalyzer,
    SupereruptionSignature,
)
from graph_neural_networks import GeologicalGraphData

class TestGeologicalNodeEmbedding:
    """Test geological node embedding functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.embedding_dim = 256
        self.num_nodes = 100
        self.node_embedder = GeologicalNodeEmbedding(
            num_nodes=self.num_nodes, embedding_dim=self.embedding_dim
        )

    def test_initialization(self):
        """Test proper initialization of node embedder."""
        assert self.node_embedder.embedding_dim == self.embedding_dim
        assert self.node_embedder.num_nodes == self.num_nodes
        assert self.node_embedder.use_physics_constraints  is True
        assert self.node_embedder.include_temporal_dynamics  is True

    def test_forward_pass(self):
        """Test forward pass through node embedder."""
        batch_size = 10

        # Create sample inputs
        node_ids = torch.randint(0, self.num_nodes, (batch_size,))
        entity_types = torch.randint(0, 10, (batch_size,))
        process_types = torch.randint(0, 15, (batch_size,))
        spatial_coords = torch.randn(batch_size, 3)
        temporal_info = torch.randn(batch_size)

        # Forward pass
        results = self.node_embedder(
            node_ids=node_ids,
            entity_types=entity_types,
            process_types=process_types,
            spatial_coords=spatial_coords,
            temporal_info=temporal_info,
        )

        # Check outputs
        assert "node_embeddings" in results
        assert "supereruption_scores" in results
        assert "magma_chamber_states" in results

        assert results["node_embeddings"].shape == (batch_size, self.embedding_dim)
        assert results["supereruption_scores"].shape == (batch_size, 1)
        assert results["magma_chamber_states"].shape == (batch_size, 10)

    def test_caldera_embeddings(self):
        """Test specialized caldera embedding generation."""
        batch_size = 5

        caldera_nodes = torch.randint(0, self.num_nodes, (batch_size,))
        caldera_properties = torch.randn(batch_size, 20)

        results = self.node_embedder.get_caldera_embeddings(
            caldera_nodes, caldera_properties
        )

        assert "caldera_embeddings" in results
        assert "supereruption_risk" in results
        assert "chamber_analysis" in results

        assert results["caldera_embeddings"].shape == (batch_size, self.embedding_dim)

class TestGeologicalEdgeEmbedding:
    """Test geological edge embedding functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.embedding_dim = 128
        self.edge_embedder = GeologicalEdgeEmbedding(embedding_dim=self.embedding_dim)

    def test_forward_pass(self):
        """Test forward pass through edge embedder."""
        batch_size = 20

        # Create sample inputs
        edge_types = torch.randint(0, 5, (batch_size,))
        edge_distances = torch.randn(batch_size)
        edge_strengths = torch.rand(batch_size)
        temporal_evolution = torch.randn(batch_size)
        source_embeddings = torch.randn(batch_size, 256)
        target_embeddings = torch.randn(batch_size, 256)

        # Forward pass
        results = self.edge_embedder(
            edge_types=edge_types,
            edge_distances=edge_distances,
            edge_strengths=edge_strengths,
            temporal_evolution=temporal_evolution,
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
        )

        # Check outputs
        assert "edge_embeddings" in results
        assert "causal_scores" in results
        assert "supereruption_trigger_scores" in results

        assert results["edge_embeddings"].shape == (batch_size, self.embedding_dim)
        assert results["causal_scores"].shape == (batch_size, 1)
        assert results["supereruption_trigger_scores"].shape == (batch_size, 1)

class TestGeologicalGraphEmbedding:
    """Test geological graph-level embedding functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.node_embedding_dim = 256
        self.edge_embedding_dim = 128
        self.graph_embedding_dim = 512

        self.graph_embedder = GeologicalGraphEmbedding(
            node_embedding_dim=self.node_embedding_dim,
            edge_embedding_dim=self.edge_embedding_dim,
            graph_embedding_dim=self.graph_embedding_dim,
        )

    def test_forward_pass(self):
        """Test forward pass through graph embedder."""
        num_nodes = 50
        num_edges = 100

        # Create sample inputs
        node_embeddings = torch.randn(num_nodes, self.node_embedding_dim)
        edge_embeddings = torch.randn(num_edges, self.edge_embedding_dim)

        # Forward pass
        results = self.graph_embedder(
            node_embeddings=node_embeddings, edge_embeddings=edge_embeddings
        )

        # Check outputs
        assert "graph_embedding" in results
        assert "system_type_logits" in results
        assert "supereruption_system_score" in results
        assert "caldera_characteristics" in results

        assert results["graph_embedding"].shape == (1, self.graph_embedding_dim)
        assert results["system_type_logits"].shape == (1, 20)
        assert results["supereruption_system_score"].shape == (1, 1)

    def test_system_classification(self):
        """Test geological system classification."""
        graph_embedding = torch.randn(1, self.graph_embedding_dim)

        classification = self.graph_embedder.classify_geological_system(graph_embedding)

        assert "system_type" in classification
        assert "confidence" in classification
        assert "all_probabilities" in classification

        assert isinstance(classification["system_type"], str)
        assert 0.0 <= classification["confidence"] <= 1.0

class TestGeologicalSimilaritySearch:
    """Test geological similarity search functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.embedding_dim = 512
        self.similarity_search = GeologicalSimilaritySearch(
            embedding_dim=self.embedding_dim,
            use_faiss=False,  # Disable FAISS for testing
        )

    def test_add_geological_system(self):
        """Test adding geological systems to the database."""
        system_id = "test_volcano_1"
        embedding = np.random.randn(self.embedding_dim)
        metadata = {
            "system_type": "volcanic_arc",
            "location": [45.0, -122.0],
            "last_eruption": 1980,
        }

        self.similarity_search.add_geological_system(system_id, embedding, metadata)

        assert system_id in self.similarity_search.embeddings_db
        assert system_id in self.similarity_search.metadata_db
        assert np.array_equal(
            self.similarity_search.embeddings_db[system_id],
            embedding / np.linalg.norm(embedding),
        )

    def test_find_similar_systems(self):
        """Test finding similar geological systems."""
        # Add some systems to the database
        for i in range(5):
            system_id = f"volcano_{i}"
            embedding = np.random.randn(self.embedding_dim)
            metadata = {
                "system_type": "volcanic_arc",
                "location": [45.0 + i, -122.0 + i],
                "last_eruption": 1980 + i * 10,
            }
            self.similarity_search.add_geological_system(system_id, embedding, metadata)

        # Search for similar systems
        query_embedding = np.random.randn(self.embedding_dim)
        similar_systems = self.similarity_search.find_similar_systems(
            query_embedding, k=3, min_similarity=0.0
        )

        assert len(similar_systems) <= 3
        for system in similar_systems:
            assert "system_id" in system
            assert "similarity" in system
            assert "metadata" in system
            assert 0.0 <= system["similarity"] <= 1.0

    def test_find_analogous_volcanic_systems(self):
        """Test finding analogous volcanic systems."""
        # Add volcanic systems
        for i in range(3):
            system_id = f"volcano_{i}"
            embedding = np.random.randn(self.embedding_dim)
            metadata = {
                "system_type": "volcanic_arc",
                "eruption_frequency": 0.1 + i * 0.05,
                "max_vei": 3 + i,
                "magma_composition": 0.6 + i * 0.1,
            }
            self.similarity_search.add_geological_system(system_id, embedding, metadata)

        # Query system
        query_system = {
            "eruption_frequency": 0.12,
            "max_vei": 4,
            "magma_composition": 0.65,
        }

        analogous_systems = self.similarity_search.find_analogous_volcanic_systems(
            query_system, k=2
        )

        assert len(analogous_systems) <= 2
        for system in analogous_systems:
            assert "system_id" in system
            assert "similarity" in system
            assert "matching_features" in system

class TestSupereruptionAnalyzer:
    """Test supereruption analysis functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = SupereruptionAnalyzer()

    def test_analyze_supereruption_potential(self):
        """Test supereruption potential analysis."""
        caldera_system = {
            "system_id": "yellowstone",
            "magma_chamber_volume": 800.0,
            "caldera_diameter": 45.0,
        }

        monitoring_data = {
            "deformation_rate": 25.0,
            "seismic_swarm_intensity": 200.0,
            "gas_emission_anomaly": 3.0,
            "thermal_anomaly_extent": 15.0,
            "geochemical_indicators": {"he3_he4_ratio": 0.15, "co2_so2_ratio": 0.8},
        }

        signature = self.analyzer.analyze_supereruption_potential(
            caldera_system, monitoring_data
        )

        assert isinstance(signature, SupereruptionSignature)
        assert signature.caldera_id == "yellowstone"
        assert 0.0 <= signature.eruption_probability <= 1.0
        assert signature.magma_chamber_volume == 800.0

    def test_identify_magma_chamber_state(self):
        """Test magma chamber state identification."""
        chamber_embedding = torch.randn(256)
        monitoring_data = {
            "temperature": 850.0,
            "pressure": 200.0,
            "crystallinity": 0.4,
            "volatile_content": {"H2O": 3.5, "CO2": 1000.0},
        }

        chamber_state = self.analyzer.identify_magma_chamber_state(
            chamber_embedding, monitoring_data
        )

        assert "chamber_state" in chamber_state
        assert "magma_mobility" in chamber_state
        assert "explosive_potential" in chamber_state
        assert "eruption_style_prediction" in chamber_state

        assert chamber_state["chamber_state"] in [
            "eruptible",
            "potentially_eruptible",
            "crystallized",
            "dormant",
        ]

    def test_compare_with_historical_supereruptions(self):
        """Test comparison with historical supereruptions."""
        current_system = SupereruptionSignature(
            caldera_id="test_caldera",
            magma_chamber_volume=600.0,
            crustal_deformation_rate=30.0,
            seismic_swarm_intensity=150.0,
            gas_emission_anomaly=2.5,
            thermal_anomaly_extent=10.0,
            ground_uplift_pattern=np.array([1, 2, 3, 4, 5]),
            geochemical_indicators={"he3_he4_ratio": 0.1},
            eruption_probability=0.4,
        )

        comparisons = self.analyzer.compare_with_historical_supereruptions(
            current_system
        )

        assert len(comparisons) > 0
        for comparison in comparisons:
            assert "historical_eruption" in comparison
            assert "similarity" in comparison
            assert "volume_comparison" in comparison
            assert "implications" in comparison

class TestGeologicalNetworkRepresentationEngine:
    """Test the main representation engine."""

    def setup_method(self):
        """Setup test fixtures."""
        self.engine = GeologicalNetworkRepresentationEngine(
            node_embedding_dim=256, edge_embedding_dim=128, graph_embedding_dim=512
        )

    def test_process_geological_network(self):
        """Test processing a complete geological network."""
        # Create sample graph data
        num_nodes = 30
        num_edges = 50

        node_features = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        node_types = torch.randint(0, 10, (num_nodes,))
        edge_types = torch.randint(0, 5, (num_edges,))
        spatial_coords = torch.randn(num_nodes, 3)

        graph_data = GeologicalGraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=None,
            node_types=node_types,
            edge_types=edge_types,
            spatial_coords=spatial_coords,
        )

        # Process the network
        results = self.engine.process_geological_network(graph_data)

        # Check results
        assert "node_embeddings" in results
        assert "edge_embeddings" in results
        assert "graph_embedding" in results
        assert "system_classification" in results
        assert "supereruption_analysis" in results

        assert results["node_embeddings"].shape == (num_nodes, 256)
        assert results["edge_embeddings"].shape == (num_edges, 128)
        assert results["graph_embedding"].shape == (1, 512)

    def test_find_similar_geological_systems(self):
        """Test finding similar geological systems."""
        # Create sample graph data
        num_nodes = 20
        num_edges = 30

        node_features = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        node_types = torch.randint(0, 10, (num_nodes,))
        edge_types = torch.randint(0, 5, (num_edges,))
        spatial_coords = torch.randn(num_nodes, 3)

        graph_data = GeologicalGraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=None,
            node_types=node_types,
            edge_types=edge_types,
            spatial_coords=spatial_coords,
        )

        # Add some systems to the similarity database first
        for i in range(3):
            embedding = np.random.randn(512)
            self.engine.similarity_search.add_geological_system(
                f"system_{i}",
                embedding,
                {"system_type": "volcanic_arc", "location": [45.0, -122.0]},
            )

        # Find similar systems
        similar_systems = self.engine.find_similar_geological_systems(graph_data, k=2)

        assert len(similar_systems) <= 2

    def test_save_and_load_model(self):
        """Test saving and loading the model."""
        import tempfile

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save the model
            self.engine.save_model(tmp_path)

            # Create a new engine and load the model
            new_engine = GeologicalNetworkRepresentationEngine()
            new_engine.load_model(tmp_path)

            # Verify the model was loaded (basic check)
            assert new_engine.node_embedding_dim == self.engine.node_embedding_dim
            assert new_engine.edge_embedding_dim == self.engine.edge_embedding_dim
            assert new_engine.graph_embedding_dim == self.engine.graph_embedding_dim

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

def test_integration():
    """Integration test for the complete system."""
    print("Running integration test...")

    # Create a realistic geological network
    from geological_graph_utils import GeologicalGraphBuilder
    import torch.nn as nn
    import torch.nn.functional as F

    builder = GeologicalGraphBuilder()

    # Add volcanic systems
    builder.add_volcanic_system(
        "mount_st_helens",
        (46.2, -122.2, 2549),
        {"last_eruption": 2008, "vei": 5, "magma_chamber_volume": 50},
    )

    builder.add_volcanic_system(
        "mount_rainier",
        (46.9, -121.8, 4392),
        {"last_eruption": 1894, "vei": 3, "magma_chamber_volume": 30},
    )

    # Add magma chambers
    builder.add_magma_chamber(
        "msh_chamber", (46.2, -122.2, -5), {"volume": 50, "temperature": 900}
    )

    # Add monitoring stations
    builder.add_seismic_station(
        "station_1", (46.0, -122.0, 1000), {"installation_date": 1980}
    )

    # Build relationships
    builder.build_spatial_graph(max_distance=100.0)
    builder.add_causal_relationship("msh_chamber", "mount_st_helens", 0.9)

    # Convert to graph data
    graph_data = builder.to_geological_graph_data()

    # Initialize and test the representation engine
    engine = GeologicalNetworkRepresentationEngine()

    # Process the network
    results = engine.process_geological_network(graph_data)

    # Verify results
    assert results["node_embeddings"].shape[0] == graph_data.node_features.shape[0]
    assert results["system_classification"]["system_type"] is not None
    assert 0.0 <= results["system_classification"]["confidence"] <= 1.0

    print("Integration test passed!")

if __name__ == "__main__":
    # Run the integration test
    test_integration()

    print("All tests completed successfully!")
