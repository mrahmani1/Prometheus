"""
Graph Neural Network architectures for geological data analysis.

This module implements various GNN architectures specifically designed for
geological process modeling, relationship discovery, and multi-scale analysis.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (ASAPooling, GATConv, GCNConv, SAGEConv,
                                SAGPooling, TopKPooling, global_add_pool,
                                global_max_pool, global_mean_pool)

logger = logging.getLogger(__name__)

@dataclass
class GeologicalGraphData:
    """Data structure for geological graph representations."""

    node_features: torch.Tensor  # Geological entity features
    edge_index: torch.Tensor  # Graph connectivity
    edge_features: torch.Tensor  # Relationship features
    node_types: torch.Tensor  # Entity types (volcano, fault, etc.)
    edge_types: torch.Tensor  # Relationship types (causal, spatial, etc.)
    temporal_info: Optional[torch.Tensor] = None  # Time information
    spatial_coords: Optional[torch.Tensor] = None  # Geographic coordinates
    batch: Optional[torch.Tensor] = None  # Batch information

class GeologicalGCN(nn.Module):
    """
    Graph Convolutional Network for geological process modeling.

    Designed to capture spatial relationships between geological entities
    and propagate information through geological process networks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_node_types: int = 10,
        num_edge_types: int = 5,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super(GeologicalGCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout

        # Node type embeddings for different geological entities
        self.node_type_embedding = nn.Embedding(num_node_types, input_dim // 4)

        # Edge type embeddings for different geological relationships
        self.edge_type_embedding = nn.Embedding(num_edge_types, input_dim // 4)

        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dims[0]))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))

        # Output layer
        self.convs.append(GCNConv(hidden_dims[-1], output_dim))

        # Activation function
        self.activation = getattr(F, activation)

        # Physics-informed constraints
        self.physics_constraint_layer = nn.Linear(output_dim, output_dim)

    def forward(self, data: GeologicalGraphData) -> torch.Tensor:
        """
        Forward pass through the geological GCN.

        Args:
            data: Geological graph data

        Returns:
            Node embeddings with geological process information
        """
        x, edge_index = data.node_features, data.edge_index

        # Add node type information
        if data.node_types is not None:
            node_type_emb = self.node_type_embedding(data.node_types)
            x = torch.cat([x, node_type_emb], dim=-1)

        # Apply GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer
        x = self.convs[-1](x, edge_index)

        # Apply physics constraints
        x = self.physics_constraint_layer(x)

        return x

    def get_geological_embeddings(
        self, data: GeologicalGraphData
    ) -> Dict[str, torch.Tensor]:
        """Extract embeddings for different geological entities."""
        embeddings = self.forward(data)

        # Group embeddings by node type
        geological_embeddings = {}
        if data.node_types is not None:
            unique_types = torch.unique(data.node_types)
            for node_type in unique_types:
                mask = data.node_types == node_type
                geological_embeddings[f"type_{node_type.item()}"] = embeddings[mask]

        return geological_embeddings

class GeologicalGAT(nn.Module):
    """
    Graph Attention Network for geological relationship weighting.

    Uses attention mechanisms to identify the most important geological
    relationships and processes for prediction tasks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_heads: List[int],
        num_node_types: int = 10,
        num_edge_types: int = 5,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
    ):
        super(GeologicalGAT, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # Node and edge type embeddings
        self.node_type_embedding = nn.Embedding(num_node_types, input_dim // 4)
        self.edge_type_embedding = nn.Embedding(num_edge_types, input_dim // 4)

        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(
            GATConv(
                input_dim,
                hidden_dims[0] // num_heads[0],
                heads=num_heads[0],
                dropout=attention_dropout,
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(
                GATConv(
                    hidden_dims[i],
                    hidden_dims[i + 1] // num_heads[i + 1],
                    heads=num_heads[i + 1],
                    dropout=attention_dropout,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))

        # Output layer (single head)
        self.convs.append(
            GATConv(hidden_dims[-1], output_dim, heads=1, dropout=attention_dropout)
        )

        # Geological process attention
        self.process_attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=4, dropout=attention_dropout
        )

    def forward(
        self, data: GeologicalGraphData
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with attention weights for geological relationships.

        Args:
            data: Geological graph data

        Returns:
            Tuple of (node embeddings, attention weights)
        """
        x, edge_index = data.node_features, data.edge_index
        attention_weights = {}

        # Add node type information
        if data.node_types is not None:
            node_type_emb = self.node_type_embedding(data.node_types)
            x = torch.cat([x, node_type_emb], dim=-1)

        # Apply GAT layers
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms)):
            x, att_weights = conv(x, edge_index, return_attention_weights=True)
            attention_weights[f"layer_{i}"] = att_weights
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer
        x, final_att_weights = self.convs[-1](
            x, edge_index, return_attention_weights=True
        )
        attention_weights["final"] = final_att_weights

        # Apply geological process attention
        x_reshaped = x.unsqueeze(0)  # Add sequence dimension
        process_attended, process_weights = self.process_attention(
            x_reshaped, x_reshaped, x_reshaped
        )
        x = process_attended.squeeze(0)
        attention_weights["process_attention"] = process_weights

        return x, attention_weights

    def get_relationship_importance(
        self, data: GeologicalGraphData
    ) -> Dict[str, float]:
        """Analyze the importance of different geological relationships."""
        _, attention_weights = self.forward(data)

        # Calculate average attention weights for different edge types
        relationship_importance = {}
        if data.edge_types is not None and "final" in attention_weights:
            edge_index, att_weights = attention_weights["final"]

            unique_edge_types = torch.unique(data.edge_types)
            for edge_type in unique_edge_types:
                mask = data.edge_types == edge_type
                avg_attention = att_weights[mask].mean().item()
                relationship_importance[f"edge_type_{edge_type.item()}"] = avg_attention

        return relationship_importance

class GeologicalGraphSAGE(nn.Module):
    """
    GraphSAGE for large-scale geological network analysis.

    Designed to handle massive geological datasets by sampling
    neighborhoods and aggregating information efficiently.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_layers: int = 3,
        aggregator: str = "mean",
        num_node_types: int = 10,
        dropout: float = 0.2,
    ):
        super(GeologicalGraphSAGE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Node type embeddings
        self.node_type_embedding = nn.Embedding(num_node_types, input_dim // 4)

        # SAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dims[0], aggr=aggregator))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(
                SAGEConv(hidden_dims[i], hidden_dims[i + 1], aggr=aggregator)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))

        # Output layer
        self.convs.append(SAGEConv(hidden_dims[-1], output_dim, aggr=aggregator))

        # Geological scale aggregation
        self.scale_aggregator = nn.Linear(
            output_dim * 3, output_dim
        )  # local, regional, global

    def forward(self, data: GeologicalGraphData) -> torch.Tensor:
        """
        Forward pass through GraphSAGE for large-scale geological analysis.

        Args:
            data: Geological graph data

        Returns:
            Multi-scale geological embeddings
        """
        x, edge_index = data.node_features, data.edge_index

        # Add node type information
        if data.node_types is not None:
            node_type_emb = self.node_type_embedding(data.node_types)
            x = torch.cat([x, node_type_emb], dim=-1)

        # Store embeddings at different scales
        scale_embeddings = []

        # Apply SAGE layers
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Store embeddings for multi-scale analysis
            if i < 3:  # First 3 layers represent different scales
                scale_embeddings.append(x)

        # Final layer
        x = self.convs[-1](x, edge_index)
        scale_embeddings.append(x)

        # Combine multi-scale information
        if len(scale_embeddings) >= 3:
            # Pad embeddings to same size if needed
            max_dim = max(emb.size(-1) for emb in scale_embeddings[:3])
            padded_embeddings = []
            for emb in scale_embeddings[:3]:
                if emb.size(-1) < max_dim:
                    padding = torch.zeros(
                        emb.size(0), max_dim - emb.size(-1), device=emb.device
                    )
                    emb = torch.cat([emb, padding], dim=-1)
                padded_embeddings.append(emb)

            multi_scale = torch.cat(padded_embeddings, dim=-1)
            x = self.scale_aggregator(multi_scale)

        return x

    def sample_subgraph(
        self, data: GeologicalGraphData, num_neighbors: List[int]
    ) -> GeologicalGraphData:
        """Sample a subgraph for efficient large-scale processing."""
        # This would implement neighborhood sampling for large graphs
        # For now, return the original data
        return data

class GeologicalGraphPooling(nn.Module):
    """
    Graph pooling methods for multi-scale geological analysis.

    Implements hierarchical pooling to capture geological processes
    at different spatial and temporal scales.
    """

    def __init__(
        self, input_dim: int, pooling_ratio: float = 0.5, pooling_method: str = "topk"
    ):
        super(GeologicalGraphPooling, self).__init__()

        self.input_dim = input_dim
        self.pooling_ratio = pooling_ratio
        self.pooling_method = pooling_method

        # Different pooling layers for different scales
        if pooling_method == "topk":
            self.local_pool = TopKPooling(input_dim, ratio=pooling_ratio)
            self.regional_pool = TopKPooling(input_dim, ratio=pooling_ratio * 0.7)
            self.global_pool = TopKPooling(input_dim, ratio=pooling_ratio * 0.5)
        elif pooling_method == "sag":
            self.local_pool = SAGPooling(input_dim, ratio=pooling_ratio)
            self.regional_pool = SAGPooling(input_dim, ratio=pooling_ratio * 0.7)
            self.global_pool = SAGPooling(input_dim, ratio=pooling_ratio * 0.5)
        elif pooling_method == "asa":
            self.local_pool = ASAPooling(input_dim, ratio=pooling_ratio)
            self.regional_pool = ASAPooling(input_dim, ratio=pooling_ratio * 0.7)
            self.global_pool = ASAPooling(input_dim, ratio=pooling_ratio * 0.5)

        # Scale fusion layer
        self.scale_fusion = nn.Linear(input_dim * 3, input_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply multi-scale pooling for geological analysis.

        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch information

        Returns:
            Dictionary with pooled representations at different scales
        """
        pooled_representations = {}

        # Local scale pooling (fine-grained geological features)
        if hasattr(self, "local_pool"):
            local_x, local_edge_index, _, local_batch, _, _ = self.local_pool(
                x, edge_index, batch=batch
            )
            pooled_representations["local"] = {
                "x": local_x,
                "edge_index": local_edge_index,
                "batch": local_batch,
            }

        # Regional scale pooling (intermediate geological processes)
        if hasattr(self, "regional_pool"):
            regional_x, regional_edge_index, _, regional_batch, _, _ = (
                self.regional_pool(x, edge_index, batch=batch)
            )
            pooled_representations["regional"] = {
                "x": regional_x,
                "edge_index": regional_edge_index,
                "batch": regional_batch,
            }

        # Global scale pooling (large-scale geological systems)
        if hasattr(self, "global_pool"):
            global_x, global_edge_index, _, global_batch, _, _ = self.global_pool(
                x, edge_index, batch=batch
            )
            pooled_representations["global"] = {
                "x": global_x,
                "edge_index": global_edge_index,
                "batch": global_batch,
            }

        # Graph-level representations
        pooled_representations["graph_level"] = {
            "mean": global_mean_pool(x, batch),
            "max": global_max_pool(x, batch),
            "sum": global_add_pool(x, batch),
        }

        return pooled_representations

    def hierarchical_pooling(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Apply hierarchical pooling to capture multi-scale geological processes."""
        # Apply pooling at different scales
        pooled = self.forward(x, edge_index, batch)

        # Combine representations from different scales
        scale_features = []

        if "local" in pooled:
            local_global = global_mean_pool(
                pooled["local"]["x"], pooled["local"]["batch"]
            )
            scale_features.append(local_global)

        if "regional" in pooled:
            regional_global = global_mean_pool(
                pooled["regional"]["x"], pooled["regional"]["batch"]
            )
            scale_features.append(regional_global)

        if "global" in pooled:
            global_global = global_mean_pool(
                pooled["global"]["x"], pooled["global"]["batch"]
            )
            scale_features.append(global_global)

        # Ensure all features have the same dimension
        if scale_features:
            max_dim = max(feat.size(-1) for feat in scale_features)
            padded_features = []
            for feat in scale_features:
                if feat.size(-1) < max_dim:
                    padding = torch.zeros(
                        feat.size(0), max_dim - feat.size(-1), device=feat.device
                    )
                    feat = torch.cat([feat, padding], dim=-1)
                padded_features.append(feat)

            # Concatenate and fuse
            multi_scale_features = torch.cat(padded_features, dim=-1)
            fused_features = self.scale_fusion(multi_scale_features)
            return fused_features

        return global_mean_pool(x, batch)

class GeologicalGNNEnsemble(nn.Module):
    """
    Ensemble of different GNN architectures for robust geological modeling.

    Combines GCN, GAT, and GraphSAGE for comprehensive geological analysis.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_node_types: int = 10,
        num_edge_types: int = 5,
        ensemble_weights: Optional[List[float]] = None,
    ):
        super(GeologicalGNNEnsemble, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize different GNN architectures
        self.gcn = GeologicalGCN(
            input_dim, hidden_dims, output_dim, num_node_types, num_edge_types
        )

        self.gat = GeologicalGAT(
            input_dim,
            hidden_dims,
            output_dim,
            num_heads=[4, 4, 2],
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
        )

        self.graphsage = GeologicalGraphSAGE(
            input_dim, hidden_dims, output_dim, num_node_types=num_node_types
        )

        # Ensemble weights
        if ensemble_weights is None:
            ensemble_weights = [1 / 3, 1 / 3, 1 / 3]
        self.register_buffer("ensemble_weights", torch.tensor(ensemble_weights))

        # Fusion layer
        self.fusion_layer = nn.Linear(output_dim * 3, output_dim)

        # Uncertainty quantification
        self.uncertainty_head = nn.Linear(output_dim, output_dim)

    def forward(self, data: GeologicalGraphData) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the ensemble of GNN architectures.

        Args:
            data: Geological graph data

        Returns:
            Dictionary with ensemble predictions and individual model outputs
        """
        # Get predictions from each model
        gcn_output = self.gcn(data)
        gat_output, attention_weights = self.gat(data)
        graphsage_output = self.graphsage(data)

        # Weighted ensemble
        weighted_outputs = [
            self.ensemble_weights[0] * gcn_output,
            self.ensemble_weights[1] * gat_output,
            self.ensemble_weights[2] * graphsage_output,
        ]
        ensemble_output = sum(weighted_outputs)

        # Alternative: Learned fusion
        concatenated = torch.cat([gcn_output, gat_output, graphsage_output], dim=-1)
        fused_output = self.fusion_layer(concatenated)

        # Uncertainty estimation based on model disagreement
        model_outputs = torch.stack([gcn_output, gat_output, graphsage_output], dim=0)
        uncertainty = torch.std(model_outputs, dim=0)
        uncertainty = self.uncertainty_head(uncertainty)

        return {
            "ensemble_prediction": ensemble_output,
            "fused_prediction": fused_output,
            "uncertainty": uncertainty,
            "individual_predictions": {
                "gcn": gcn_output,
                "gat": gat_output,
                "graphsage": graphsage_output,
            },
            "attention_weights": attention_weights,
        }

    def get_model_agreement(self, data: GeologicalGraphData) -> torch.Tensor:
        """Calculate agreement between different GNN models."""
        outputs = self.forward(data)
        individual_preds = outputs["individual_predictions"]

        # Calculate pairwise correlations
        gcn_pred = individual_preds["gcn"]
        gat_pred = individual_preds["gat"]
        sage_pred = individual_preds["graphsage"]

        # Compute correlation coefficients
        gcn_gat_corr = F.cosine_similarity(gcn_pred, gat_pred, dim=-1)
        gcn_sage_corr = F.cosine_similarity(gcn_pred, sage_pred, dim=-1)
        gat_sage_corr = F.cosine_similarity(gat_pred, sage_pred, dim=-1)

        # Average agreement
        agreement = (gcn_gat_corr + gcn_sage_corr + gat_sage_corr) / 3
        return agreement

def create_geological_graph_from_data(
    geological_entities: List[Dict],
    relationships: List[Dict],
    node_features: np.ndarray,
    edge_features: Optional[np.ndarray] = None,
) -> GeologicalGraphData:
    """
    Create a geological graph from raw geological data.

    Args:
        geological_entities: List of geological entities with metadata
        relationships: List of relationships between entities
        node_features: Feature matrix for nodes
        edge_features: Feature matrix for edges

    Returns:
        GeologicalGraphData object ready for GNN processing
    """
    # Create edge index from relationships
    edge_index = []
    edge_types = []

    for rel in relationships:
        source_idx = rel["source_id"]
        target_idx = rel["target_id"]
        rel_type = rel["relationship_type"]

        edge_index.append([source_idx, target_idx])
        edge_types.append(rel_type)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_types = torch.tensor(edge_types, dtype=torch.long)

    # Create node types
    node_types = []
    spatial_coords = []

    for entity in geological_entities:
        node_types.append(entity.get("entity_type", 0))
        if "coordinates" in entity:
            spatial_coords.append(entity["coordinates"])

    node_types = torch.tensor(node_types, dtype=torch.long)
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)

    # Optional edge features
    edge_features_tensor = None
    if edge_features is not None:
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float32)

    # Optional spatial coordinates
    spatial_coords_tensor = None
    if spatial_coords:
        spatial_coords_tensor = torch.tensor(spatial_coords, dtype=torch.float32)

    return GeologicalGraphData(
        node_features=node_features_tensor,
        edge_index=edge_index,
        edge_features=edge_features_tensor,
        node_types=node_types,
        edge_types=edge_types,
        spatial_coords=spatial_coords_tensor,
    )

# Example usage and testing functions
def test_geological_gnn_architectures():
    """Test function for geological GNN architectures."""
    # Create sample geological graph data
    num_nodes = 100
    num_edges = 200
    input_dim = 64

    # Sample node features (geological measurements)
    node_features = torch.randn(num_nodes, input_dim)

    # Sample edge index (geological relationships)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Sample node and edge types
    node_types = torch.randint(0, 10, (num_nodes,))
    edge_types = torch.randint(0, 5, (num_edges,))

    # Create geological graph data
    data = GeologicalGraphData(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=None,
        node_types=node_types,
        edge_types=edge_types,
    )

    # Test GCN
    gcn = GeologicalGCN(input_dim, [128, 64], 32)
    gcn_output = gcn(data)
    print(f"GCN output shape: {gcn_output.shape}")

    # Test GAT
    gat = GeologicalGAT(input_dim, [128, 64], 32, [4, 4])
    gat_output, attention_weights = gat(data)
    print(f"GAT output shape: {gat_output.shape}")
    print(f"Attention weights keys: {attention_weights.keys()}")

    # Test GraphSAGE
    graphsage = GeologicalGraphSAGE(input_dim, [128, 64], 32)
    sage_output = graphsage(data)
    print(f"GraphSAGE output shape: {sage_output.shape}")

    # Test Ensemble
    ensemble = GeologicalGNNEnsemble(input_dim, [128, 64], 32)
    ensemble_output = ensemble(data)
    print(f"Ensemble prediction shape: {ensemble_output['ensemble_prediction'].shape}")
    print(f"Uncertainty shape: {ensemble_output['uncertainty'].shape}")

    return True

if __name__ == "__main__":
    # Run tests
    test_geological_gnn_architectures()
    logger.info("Geological GNN architectures implemented and tested successfully")
