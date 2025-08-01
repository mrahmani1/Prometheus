"""
Tests for geological graph neural network implementations.
"""

import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.geological_graph_utils import (
    GeologicalGraphAnalyzer,
    GeologicalGraphBuilder,
    create_volcanic_system_graph,
)
from core.gnn_training import GeologicalGNNTrainer, create_geological_gnn_model
from core.graph_neural_networks import (
    GeologicalGAT,
    GeologicalGCN,
    GeologicalGNNEnsemble,
    GeologicalGraphData,
    GeologicalGraphPooling,
    GeologicalGraphSAGE,
    import,
)
from core.graph_neural_networks import numpy as np
from core.graph_neural_networks import torch.nn as nn
from core.graph_neural_networks import torch.nn.functional as F

class TestGeologicalGraphData:
    """Test the GeologicalGraphData structure."""

    def test_geological_graph_data_creation(self):
        """Test creating GeologicalGraphData."""
        num_nodes = 50
        num_edges = 100
        feature_dim = 32

        node_features = torch.randn(num_nodes, feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        node_types = torch.randint(0, 5, (num_nodes,))
        edge_types = torch.randint(0, 3, (num_edges,))

        data = GeologicalGraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=None,
            node_types=node_types,
            edge_types=edge_types,
        )

        assert data.node_features.shape == (num_nodes, feature_dim)
        assert data.edge_index.shape == (2, num_edges)
        assert data.node_types.shape == (num_nodes,)
        assert data.edge_types.shape == (num_edges,)

    def test_geological_graph_data_with_spatial_coords(self):
        """Test GeologicalGraphData with spatial coordinates."""
        num_nodes = 30
        spatial_coords = torch.randn(num_nodes, 3)  # lat, lon, depth

        data = GeologicalGraphData(
            node_features=torch.randn(num_nodes, 16),
            edge_index=torch.randint(0, num_nodes, (2, 50)),
            edge_features=None,
            node_types=torch.randint(0, 5, (num_nodes,)),
            edge_types=torch.randint(0, 3, (50,)),
            spatial_coords=spatial_coords,
        )

        assert data.spatial_coords.shape == (num_nodes, 3)

class TestGeologicalGCN:
    """Test the Geological Graph Convolutional Network."""

    def test_gcn_initialization(self):
        """Test GCN model initialization."""
        input_dim = 64
        hidden_dims = [128, 64]
        output_dim = 32

        model = GeologicalGCN(input_dim, hidden_dims, output_dim)

        assert len(model.convs) == 3  # input + hidden + output
        assert len(model.batch_norms) == 2  # no batch norm for output
        assert model.input_dim == input_dim
        assert model.output_dim == output_dim

    def test_gcn_forward_pass(self):
        """Test GCN forward pass."""
        num_nodes = 100
        input_dim = 64
        hidden_dims = [128, 64]
        output_dim = 32

        model = GeologicalGCN(input_dim, hidden_dims, output_dim)

        # Create test data
        data = GeologicalGraphData(
            node_features=torch.randn(num_nodes, input_dim),
            edge_index=torch.randint(0, num_nodes, (2, 200)),
            edge_features=None,
            node_types=torch.randint(0, 10, (num_nodes,)),
            edge_types=None,
        )

        output = model(data)

        assert output.shape == (num_nodes, output_dim)
        assert not torch.isnan(output).any()

    def test_gcn_geological_embeddings(self):
        """Test extraction of geological embeddings."""
        num_nodes = 50
        input_dim = 32
        output_dim = 16

        model = GeologicalGCN(input_dim, [64], output_dim)

        data = GeologicalGraphData(
            node_features=torch.randn(num_nodes, input_dim),
            edge_index=torch.randint(0, num_nodes, (2, 100)),
            edge_features=None,
            node_types=torch.tensor(
                [0, 1, 2] * (num_nodes // 3) + [0] * (num_nodes % 3)
            ),
            edge_types=None,
        )

        embeddings = model.get_geological_embeddings(data)

        assert isinstance(embeddings, dict)
        assert len(embeddings) > 0
        for key, emb in embeddings.items():
            assert emb.shape[1] == output_dim

class TestGeologicalGAT:
    """Test the Geological Graph Attention Network."""

    def test_gat_initialization(self):
        """Test GAT model initialization."""
        input_dim = 64
        hidden_dims = [128, 64]
        output_dim = 32
        num_heads = [4, 2]

        model = GeologicalGAT(input_dim, hidden_dims, output_dim, num_heads)

        assert len(model.convs) == 3
        assert model.num_heads == num_heads

    def test_gat_forward_with_attention(self):
        """Test GAT forward pass with attention weights."""
        num_nodes = 80
        input_dim = 48
        hidden_dims = [96, 48]
        output_dim = 24
        num_heads = [4, 2]

        model = GeologicalGAT(input_dim, hidden_dims, output_dim, num_heads)

        data = GeologicalGraphData(
            node_features=torch.randn(num_nodes, input_dim),
            edge_index=torch.randint(0, num_nodes, (2, 160)),
            edge_features=None,
            node_types=torch.randint(0, 5, (num_nodes,)),
            edge_types=torch.randint(0, 3, (160,)),
        )

        output, attention_weights = model(data)

        assert output.shape == (num_nodes, output_dim)
        assert isinstance(attention_weights, dict)
        assert "final" in attention_weights
        assert "process_attention" in attention_weights

    def test_gat_relationship_importance(self):
        """Test relationship importance analysis."""
        num_nodes = 60
        input_dim = 32
        output_dim = 16

        model = GeologicalGAT(input_dim, [64], output_dim, [4])

        data = GeologicalGraphData(
            node_features=torch.randn(num_nodes, input_dim),
            edge_index=torch.randint(0, num_nodes, (2, 120)),
            edge_features=None,
            node_types=torch.randint(0, 5, (num_nodes,)),
            edge_types=torch.randint(0, 3, (120,)),
        )

        importance = model.get_relationship_importance(data)

        assert isinstance(importance, dict)
        for key, value in importance.items():
            assert isinstance(value, float)
            assert 0 <= value <= 1

class TestGeologicalGraphSAGE:
    """Test the Geological GraphSAGE implementation."""

    def test_graphsage_initialization(self):
        """Test GraphSAGE model initialization."""
        input_dim = 64
        hidden_dims = [128, 64]
        output_dim = 32

        model = GeologicalGraphSAGE(input_dim, hidden_dims, output_dim)

        assert len(model.convs) == 3
        assert model.input_dim == input_dim
        assert model.output_dim == output_dim

    def test_graphsage_forward_pass(self):
        """Test GraphSAGE forward pass."""
        num_nodes = 120
        input_dim = 56
        hidden_dims = [112, 56]
        output_dim = 28

        model = GeologicalGraphSAGE(input_dim, hidden_dims, output_dim)

        data = GeologicalGraphData(
            node_features=torch.randn(num_nodes, input_dim),
            edge_index=torch.randint(0, num_nodes, (2, 240)),
            edge_features=None,
            node_types=torch.randint(0, 8, (num_nodes,)),
            edge_types=None,
        )

        output = model(data)

        assert output.shape == (num_nodes, output_dim)
        assert not torch.isnan(output).any()

class TestGeologicalGNNEnsemble:
    """Test the Geological GNN Ensemble."""

    def test_ensemble_initialization(self):
        """Test ensemble model initialization."""
        input_dim = 64
        hidden_dims = [128, 64]
        output_dim = 32

        model = GeologicalGNNEnsemble(input_dim, hidden_dims, output_dim)

        assert hasattr(model, "gcn")
        assert hasattr(model, "gat")
        assert hasattr(model, "graphsage")
        assert hasattr(model, "fusion_layer")

    def test_ensemble_forward_pass(self):
        """Test ensemble forward pass."""
        num_nodes = 100
        input_dim = 48
        hidden_dims = [96, 48]
        output_dim = 24

        model = GeologicalGNNEnsemble(input_dim, hidden_dims, output_dim)

        data = GeologicalGraphData(
            node_features=torch.randn(num_nodes, input_dim),
            edge_index=torch.randint(0, num_nodes, (2, 200)),
            edge_features=None,
            node_types=torch.randint(0, 5, (num_nodes,)),
            edge_types=torch.randint(0, 3, (200,)),
        )

        outputs = model(data)

        assert "ensemble_prediction" in outputs
        assert "fused_prediction" in outputs
        assert "uncertainty" in outputs
        assert "individual_predictions" in outputs

        assert outputs["ensemble_prediction"].shape == (num_nodes, output_dim)
        assert outputs["fused_prediction"].shape == (num_nodes, output_dim)
        assert outputs["uncertainty"].shape == (num_nodes, output_dim)

    def test_ensemble_model_agreement(self):
        """Test model agreement calculation."""
        num_nodes = 80
        input_dim = 32
        output_dim = 16

        model = GeologicalGNNEnsemble(input_dim, [64], output_dim)

        data = GeologicalGraphData(
            node_features=torch.randn(num_nodes, input_dim),
            edge_index=torch.randint(0, num_nodes, (2, 160)),
            edge_features=None,
            node_types=torch.randint(0, 5, (num_nodes,)),
            edge_types=None,
        )

        agreement = model.get_model_agreement(data)

        assert agreement.shape == (num_nodes,)
        assert torch.all(agreement >= -1) and torch.all(agreement <= 1)

class TestGeologicalGraphPooling:
    """Test the Geological Graph Pooling methods."""

    def test_pooling_initialization(self):
        """Test pooling layer initialization."""
        input_dim = 64
        pooling = GeologicalGraphPooling(input_dim)

        assert pooling.input_dim == input_dim
        assert hasattr(pooling, "local_pool")
        assert hasattr(pooling, "regional_pool")
        assert hasattr(pooling, "global_pool")

    def test_multi_scale_pooling(self):
        """Test multi-scale pooling operation."""
        num_nodes = 100
        input_dim = 32

        pooling = GeologicalGraphPooling(input_dim)

        x = torch.randn(num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (2, 200))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        pooled = pooling(x, edge_index, batch)

        assert isinstance(pooled, dict)
        assert "graph_level" in pooled
        assert "mean" in pooled["graph_level"]
        assert "max" in pooled["graph_level"]
        assert "sum" in pooled["graph_level"]

    def test_hierarchical_pooling(self):
        """Test hierarchical pooling."""
        num_nodes = 80
        input_dim = 24

        pooling = GeologicalGraphPooling(input_dim)

        x = torch.randn(num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (2, 160))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        hierarchical = pooling.hierarchical_pooling(x, edge_index, batch)

        assert hierarchical.shape[0] == 1  # Single graph
        assert hierarchical.shape[1] == input_dim

class TestGeologicalGraphBuilder:
    """Test the Geological Graph Builder."""

    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = GeologicalGraphBuilder()

        assert len(builder.entities) == 0
        assert len(builder.relationships) == 0

    def test_add_volcanic_system(self):
        """Test adding volcanic systems."""
        builder = GeologicalGraphBuilder()

        builder.add_volcanic_system(
            "volcano_1", (45.0, -122.0, 2000.0), {"last_eruption": 1980, "vei": 5}
        )

        assert "volcano_1" in builder.entities
        assert builder.entities["volcano_1"].entity_type == "volcano"
        assert builder.entities["volcano_1"].coordinates == (45.0, -122.0, 2000.0)

    def test_add_relationships(self):
        """Test adding different types of relationships."""
        builder = GeologicalGraphBuilder()

        # Add entities first
        builder.add_volcanic_system("volcano_1", (45.0, -122.0, 2000.0), {})
        builder.add_seismic_station("station_1", (45.1, -122.1, 1500.0), {})

        # Add spatial relationship
        builder.add_spatial_relationship("volcano_1", "station_1", 15.0)

        # Add causal relationship
        builder.add_causal_relationship("volcano_1", "station_1", 0.8)

        assert len(builder.relationships) == 2
        assert builder.relationships[0].relationship_type == "spatial"
        assert builder.relationships[1].relationship_type == "causal"

    def test_build_spatial_graph(self):
        """Test automatic spatial graph building."""
        builder = GeologicalGraphBuilder()

        # Add multiple entities
        builder.add_volcanic_system("volcano_1", (45.0, -122.0, 2000.0), {})
        builder.add_volcanic_system("volcano_2", (45.1, -122.1, 1800.0), {})
        builder.add_seismic_station("station_1", (45.05, -122.05, 1500.0), {})

        initial_relationships = len(builder.relationships)
        builder.build_spatial_graph(max_distance=50.0)

        assert len(builder.relationships) > initial_relationships

    def test_to_geological_graph_data(self):
        """Test conversion to GeologicalGraphData."""
        builder = GeologicalGraphBuilder()

        # Add entities and relationships
        builder.add_volcanic_system("volcano_1", (45.0, -122.0, 2000.0), {"vei": 5})
        builder.add_seismic_station(
            "station_1", (45.1, -122.1, 1500.0), {"sensitivity": 0.8}
        )
        builder.add_spatial_relationship("volcano_1", "station_1", 15.0)

        graph_data = builder.to_geological_graph_data()

        assert isinstance(graph_data, GeologicalGraphData)
        assert graph_data.node_features.shape[0] == 2  # 2 entities
        assert graph_data.edge_index.shape[1] == 2  # 2 edges (bidirectional)
        assert graph_data.node_types.shape[0] == 2
        assert graph_data.spatial_coords.shape == (2, 3)

class TestGeologicalGraphAnalyzer:
    """Test the Geological Graph Analyzer."""

    def test_analyze_graph_structure(self):
        """Test graph structure analysis."""
        # Create a simple graph
        builder = GeologicalGraphBuilder()
        builder.add_volcanic_system("v1", (45.0, -122.0, 2000.0), {})
        builder.add_volcanic_system("v2", (45.1, -122.1, 1800.0), {})
        builder.add_spatial_relationship("v1", "v2", 15.0)

        graph_data = builder.to_geological_graph_data()
        analyzer = GeologicalGraphAnalyzer()

        metrics = analyzer.analyze_graph_structure(graph_data)

        assert "num_nodes" in metrics
        assert "num_edges" in metrics
        assert "density" in metrics
        assert "average_clustering" in metrics
        assert metrics["num_nodes"] == 2
        assert metrics["num_edges"] == 2  # Bidirectional

    def test_analyze_node_types(self):
        """Test node type analysis."""
        builder = GeologicalGraphBuilder()
        builder.add_volcanic_system("v1", (45.0, -122.0, 2000.0), {})
        builder.add_seismic_station("s1", (45.1, -122.1, 1500.0), {})

        graph_data = builder.to_geological_graph_data()
        analyzer = GeologicalGraphAnalyzer()

        node_types = analyzer.analyze_node_types(graph_data)

        assert isinstance(node_types, dict)
        assert "volcano" in node_types
        assert "seismic_station" in node_types
        assert node_types["volcano"] == 1
        assert node_types["seismic_station"] == 1

    def test_calculate_spatial_statistics(self):
        """Test spatial statistics calculation."""
        builder = GeologicalGraphBuilder()
        builder.add_volcanic_system("v1", (45.0, -122.0, 2000.0), {})
        builder.add_volcanic_system("v2", (46.0, -123.0, 1800.0), {})

        graph_data = builder.to_geological_graph_data()
        analyzer = GeologicalGraphAnalyzer()

        spatial_stats = analyzer.calculate_spatial_statistics(graph_data)

        assert "latitude_range" in spatial_stats
        assert "longitude_range" in spatial_stats
        assert "depth_range" in spatial_stats
        assert "center_latitude" in spatial_stats
        assert "mean_distance" in spatial_stats

        assert spatial_stats["latitude_range"] == 1.0
        assert spatial_stats["longitude_range"] == 1.0

class TestGNNTraining:
    """Test GNN training functionality."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = GeologicalGCN(32, [64], 16)
        trainer = GeologicalGNNTrainer(model, device="cpu")

        assert trainer.model == model
        assert trainer.device == "cpu"
        assert len(trainer.train_losses) == 0

    def test_physics_informed_loss(self):
        """Test physics-informed loss calculation."""
        model = GeologicalGCN(32, [64], 16)
        trainer = GeologicalGNNTrainer(model, device="cpu")

        # Create dummy data
        predictions = torch.randn(10, 16)
        data = GeologicalGraphData(
            node_features=torch.randn(10, 32),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_features=None,
            node_types=None,
            edge_types=None,
            spatial_coords=torch.randn(10, 3),
        )

        physics_loss = trainer.physics_informed_loss(predictions, data)

        assert isinstance(physics_loss, torch.Tensor)
        assert physics_loss.item() >= 0

class TestCreateVolcanicSystemGraph:
    """Test volcanic system graph creation."""

    def test_create_volcanic_system_graph(self):
        """Test creating a comprehensive volcanic system graph."""
        volcanoes = [
            {
                "volcano_id": "v1",
                "latitude": 45.0,
                "longitude": -122.0,
                "elevation": 2000,
                "properties": {"last_eruption": 1980},
            }
        ]

        seismic_events = [
            {
                "event_id": 1,
                "latitude": 45.05,
                "longitude": -122.05,
                "depth": 5.0,
                "magnitude": 3.2,
                "timestamp": 1640995200,
            }
        ]

        monitoring_stations = [
            {
                "station_id": "s1",
                "latitude": 45.02,
                "longitude": -122.02,
                "elevation": 1500,
                "properties": {"installation_date": 2000},
            }
        ]

        graph_data = create_volcanic_system_graph(
            volcanoes, seismic_events, monitoring_stations
        )

        assert isinstance(graph_data, GeologicalGraphData)
        assert graph_data.node_features.shape[0] == 3  # volcano + event + station
        assert graph_data.spatial_coords.shape[0] == 3

class TestModelFactory:
    """Test the model factory function."""

    def test_create_gcn_model(self):
        """Test creating GCN model."""
        model = create_geological_gnn_model("gcn", 32, [64], 16)
        assert isinstance(model, GeologicalGCN)

    def test_create_gat_model(self):
        """Test creating GAT model."""
        model = create_geological_gnn_model("gat", 32, [64], 16, num_heads=[4])
        assert isinstance(model, GeologicalGAT)

    def test_create_graphsage_model(self):
        """Test creating GraphSAGE model."""
        model = create_geological_gnn_model("graphsage", 32, [64], 16)
        assert isinstance(model, GeologicalGraphSAGE)

    def test_create_ensemble_model(self):
        """Test creating ensemble model."""
        model = create_geological_gnn_model("ensemble", 32, [64], 16)
        assert isinstance(model, GeologicalGNNEnsemble)

    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError):
            create_geological_gnn_model("invalid_type", 32, [64], 16)

# Integration tests
class TestGNNIntegration:
    """Integration tests for the complete GNN pipeline."""

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end GNN pipeline."""
        # 1. Create geological graph
        builder = GeologicalGraphBuilder()
        builder.add_volcanic_system("v1", (45.0, -122.0, 2000.0), {"vei": 5})
        builder.add_seismic_station("s1", (45.1, -122.1, 1500.0), {"sensitivity": 0.8})
        builder.build_spatial_graph(max_distance=50.0)

        graph_data = builder.to_geological_graph_data()

        # 2. Create and train model
        model = create_geological_gnn_model("gcn", 64, [128, 64], 3)
        trainer = GeologicalGNNTrainer(model, device="cpu")

        # Add dummy targets
        graph_data.y = torch.randint(0, 3, (graph_data.node_features.shape[0],))

        # 3. Test single training step
        criterion = torch.nn.CrossEntropyLoss()
        train_loader = [graph_data]

        loss, metrics = trainer.train_epoch(train_loader, criterion)

        assert isinstance(loss, float)
        assert "accuracy" in metrics

        # 4. Test prediction
        model.eval()
        with torch.no_grad():
            predictions = model(graph_data)

        assert predictions.shape == (graph_data.node_features.shape[0], 3)
        assert not torch.isnan(predictions).any()

if __name__ == "__main__":
    pytest.main([__file__])
