"""
Advanced Graph Neural Networks for Geological Process Modeling

This module implements state-of-the-art GNNs for geological relationship discovery including:
    - Temporal Graph Neural Networks for geological process evolution
- Heterogeneous GNNs for multi-type geological entities
- Graph Attention Networks for causal relationship discovery
- Variational Graph Autoencoders for uncertainty quantification
- Physics-informed GNNs with geological constraints
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, MessagePassing, SAGEConv

logger = logging.getLogger(__name__)


@dataclass
class GeologicalNode:
    """Geological process or entity node"""

    node_id: str
    node_type: str  # 'earthquake', 'volcano', 'fault', 'magma_chamber', etc.
    coordinates: Tuple[float, float, float]  # lat, lon, depth
    properties: Dict[str, float]
    temporal_features: List[float]
    uncertainty: float = 0.0
    confidence: float = 1.0

    def to_tensor(self) -> torch.Tensor:
        """Convert node to feature tensor"""
        features = []
        features.extend(self.coordinates)
        features.extend(list(self.properties.values()))
        features.extend(self.temporal_features)
        features.extend([self.uncertainty, self.confidence])
        return torch.tensor(features, dtype=torch.float32)


@dataclass
class GeologicalEdge:
    """Geological relationship edge"""

    source_id: str
    target_id: str
    edge_type: str  # 'causal', 'spatial', 'temporal', 'mechanical', etc.
    strength: float
    confidence: float
    temporal_lag: float  # Time delay in relationship
    distance: float  # Spatial distance
    properties: Dict[str, float]

    def to_tensor(self) -> torch.Tensor:
        """Convert edge to feature tensor"""
        features = [self.strength, self.confidence, self.temporal_lag, self.distance]
        features.extend(list(self.properties.values()))
        return torch.tensor(features, dtype=torch.float32)


class TemporalGraphAttention(MessagePassing):
    """Temporal Graph Attention mechanism for geological processes"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        temporal_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.temporal_dim = temporal_dim
        self.dropout = dropout

        # Attention components
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)

        # Temporal encoding
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, temporal_dim),
            nn.ReLU(),
            nn.Linear(temporal_dim, heads * out_channels),
        )

        # Output projection
        self.lin_out = nn.Linear(heads * out_channels, out_channels)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        temporal_lags: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with temporal attention"""

        # Compute attention components
        query = self.lin_query(x).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.heads, self.out_channels)

        # Temporal encoding
        temporal_encoding = self.temporal_encoder(temporal_lags.unsqueeze(-1))
        temporal_encoding = temporal_encoding.view(-1, self.heads, self.out_channels)

        # Message passing with temporal attention
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            temporal_encoding=temporal_encoding,
        )

        # Output projection
        out = out.view(-1, self.heads * self.out_channels)
        out = self.lin_out(out)
        out = self.dropout_layer(out)

        return out

    def message(
        self,
        query_i: torch.Tensor,
        key_j: torch.Tensor,
        value_j: torch.Tensor,
        edge_attr: torch.Tensor,
        temporal_encoding: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        """Compute messages with temporal attention"""

        # Attention scores
        attention = (query_i * (key_j + temporal_encoding)).sum(dim=-1) / np.sqrt(
            self.out_channels
        )
        attention = F.softmax(attention, dim=1)

        # Apply attention to values
        out = attention.unsqueeze(-1) * value_j

        return out


class PhysicsInformedGNN(nn.Module):
    """Physics-informed Graph Neural Network for geological processes"""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
        )

        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_features, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)
        )

        # Temporal attention layers
        self.temporal_layers = nn.ModuleList(
            [
                TemporalGraphAttention(hidden_dim, hidden_dim, num_heads)
                for _ in range(num_layers)
            ]
        )

        # Physics constraint layers
        self.physics_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                )
                for _ in range(2)
            ]
        )

        # Output layers
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),  # Predict activity/magnitude
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),  # Predict relationship strength
        )

        # Uncertainty estimation
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive uncertainty
        )

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """Forward pass through physics-informed GNN"""

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        temporal_lags = (
            data.temporal_lags
            if hasattr(data, "temporal_lags")
            else torch.zeros(edge_index.size(1))
        )

        # Initial embeddings
        x = self.node_embedding(x)
        self.edge_embedding(edge_attr)

        # Store initial features for residual connections
        x.clone()

        # Temporal attention layers
        for i, layer in enumerate(self.temporal_layers):
            x_new = layer(x, edge_index, edge_attr, temporal_lags)

            # Residual connection
            if i > 0:
                x = x + x_new
            else:
                x = x_new

            # Apply physics constraints
            if i < len(self.physics_layers):
                x = self.physics_layers[i](x) + x  # Another residual connection

        # Predictions
        node_predictions = self.node_predictor(x)
        uncertainty = self.uncertainty_predictor(x)

        # Edge predictions (for relationship strength)
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        edge_predictions = self.edge_predictor(edge_features)

        return {
            "node_predictions": node_predictions,
            "edge_predictions": edge_predictions,
            "node_embeddings": x,
            "uncertainty": uncertainty,
        }

    def physics_loss(
        self, predictions: Dict[str, torch.Tensor], data: Data
    ) -> torch.Tensor:
        """Compute physics-based loss constraints"""

        node_preds = predictions["node_predictions"]
        edge_preds = predictions["edge_predictions"]

        # Energy conservation constraint
        # Sum of energy changes should be conserved in the system
        energy_conservation = torch.sum(node_preds) ** 2

        # Causality constraint
        # Effects cannot precede causes (temporal ordering)
        temporal_lags = (
            data.temporal_lags
            if hasattr(data, "temporal_lags")
            else torch.zeros(data.edge_index.size(1))
        )
        causality_violation = F.relu(-temporal_lags * edge_preds.squeeze()).sum()

        # Spatial decay constraint
        # Relationship strength should decay with distance
        if hasattr(data, "distances"):
            distances = data.distances
            expected_decay = torch.exp(
                -distances / 100.0
            )  # 100km characteristic length
            spatial_constraint = F.mse_loss(edge_preds.squeeze(), expected_decay)
        else:
            spatial_constraint = torch.tensor(0.0)

        # Magnitude constraint
        # Predicted magnitudes should be physically reasonable
        magnitude_constraint = (
            F.relu(node_preds - 10.0).sum() + F.relu(-node_preds).sum()
        )

        total_physics_loss = (
            energy_conservation
            + causality_violation
            + spatial_constraint
            + magnitude_constraint
        )

        return total_physics_loss


class HeterogeneousGeologicalGNN(nn.Module):
    """Heterogeneous GNN for different types of geological entities"""

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[str],
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Type-specific embeddings
        self.node_embeddings = nn.ModuleDict(
            {
                node_type: nn.Linear(
                    64, hidden_dim
                )  # Assume 64 input features per type
                for node_type in node_types
            }
        )

        # Heterogeneous message passing layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                source_type, _, target_type = edge_type.partition("_to_")
                if source_type in node_types and target_type in node_types:
                    conv_dict[edge_type] = SAGEConv(hidden_dim, hidden_dim)
            self.conv_layers.append(nn.ModuleDict(conv_dict))

        # Type-specific output layers
        self.output_layers = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for node_type in node_types
            }
        )

    def forward(self, hetero_data: HeteroData) -> Dict[str, torch.Tensor]:
        """Forward pass through heterogeneous GNN"""

        # Initial embeddings for each node type
        x_dict = {}
        for node_type in self.node_types:
            if node_type in hetero_data.x_dict:
                x_dict[node_type] = self.node_embeddings[node_type](
                    hetero_data.x_dict[node_type]
                )

        # Message passing layers
        for conv_layer in self.conv_layers:
            x_dict_new = {}

            for edge_type, conv in conv_layer.items():
                source_type, _, target_type = edge_type.partition("_to_")

                if (
                    edge_type in hetero_data.edge_index_dict
                    and source_type in x_dict
                    and target_type in x_dict
                ):

                    edge_index = hetero_data.edge_index_dict[edge_type]

                    # Create homogeneous subgraph for this edge type
                    source_x = x_dict[source_type]
                    target_x = x_dict[target_type]

                    if source_type == target_type:
                        # Homogeneous case
                        x_new = conv(source_x, edge_index)
                    else:
                        # Heterogeneous case - concatenate and project
                        combined_x = torch.cat([source_x, target_x], dim=0)
                        # Adjust edge indices for concatenated tensor
                        edge_index_adj = edge_index.clone()
                        edge_index_adj[1] += source_x.size(0)
                        x_new = conv(combined_x, edge_index_adj)
                        # Split back
                        x_new = x_new[: source_x.size(0)]  # Take source part

                    if target_type not in x_dict_new:
                        x_dict_new[target_type] = []
                    x_dict_new[target_type].append(x_new)

            # Aggregate messages for each node type
            for node_type in self.node_types:
                if node_type in x_dict_new:
                    x_dict[node_type] = torch.stack(x_dict_new[node_type]).mean(dim=0)

        # Generate predictions for each node type
        predictions = {}
        for node_type in self.node_types:
            if node_type in x_dict:
                predictions[node_type] = self.output_layers[node_type](
                    x_dict[node_type]
                )

        return predictions


class VariationalGraphAutoencoder(nn.Module):
    """Variational Graph Autoencoder for uncertainty quantification"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv1 = GCNConv(input_dim, hidden_dim)
        self.encoder_conv2 = GCNConv(hidden_dim, hidden_dim)

        # Variational layers
        self.mu_layer = GCNConv(hidden_dim, latent_dim)
        self.logvar_layer = GCNConv(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Edge decoder for link prediction
        self.edge_decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def encode(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode nodes to latent space"""
        h = F.relu(self.encoder_conv1(x, edge_index))
        h = F.relu(self.encoder_conv2(h, edge_index))

        mu = self.mu_layer(h, edge_index)
        logvar = self.logvar_layer(h, edge_index)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_nodes(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representations to node features"""
        return self.decoder(z)

    def decode_edges(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Decode edge probabilities"""
        row, col = edge_index
        edge_features = torch.cat([z[row], z[col]], dim=1)
        return self.edge_decoder(edge_features)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through VGAE"""
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)

        x_recon = self.decode_nodes(z)
        edge_probs = self.decode_edges(z, edge_index)

        return {
            "x_recon": x_recon,
            "edge_probs": edge_probs,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def loss_function(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute VGAE loss"""
        x_recon = outputs["x_recon"]
        edge_probs = outputs["edge_probs"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")

        # Edge reconstruction loss
        edge_labels = torch.ones(edge_index.size(1))  # Assume all edges exist
        edge_loss = F.binary_cross_entropy(
            edge_probs.squeeze(), edge_labels, reduction="sum"
        )

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + edge_loss + kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "edge_loss": edge_loss,
            "kl_loss": kl_loss,
        }


class CausalDiscoveryGNN(nn.Module):
    """GNN for discovering causal relationships in geological processes"""

    def __init__(self, node_features: int, hidden_dim: int = 128, num_heads: int = 8):
        super().__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
        )

        # Causal attention mechanism
        self.causal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )

        # Temporal convolution for time-series analysis
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # Causal strength predictor
        self.causal_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Causal strength between 0 and 1
        )

        # Direction predictor (cause -> effect or effect -> cause)
        self.direction_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Binary classification
            nn.Softmax(dim=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for causal discovery"""

        # Feature extraction
        node_features = self.feature_extractor(x)

        # Temporal processing if available
        if temporal_features is not None:
            # Reshape for 1D convolution: (batch, channels, sequence)
            temp_features = temporal_features.transpose(1, 2)
            temp_features = self.temporal_conv(temp_features)
            temp_features = temp_features.transpose(1, 2)

            # Combine with node features
            node_features = node_features + temp_features.mean(dim=1)

        # Self-attention for capturing long-range dependencies
        attended_features, attention_weights = self.causal_attention(
            node_features.unsqueeze(0),
            node_features.unsqueeze(0),
            node_features.unsqueeze(0),
        )
        attended_features = attended_features.squeeze(0)

        # Edge-wise predictions
        row, col = edge_index
        edge_features = torch.cat(
            [attended_features[row], attended_features[col]], dim=1
        )

        causal_strength = self.causal_predictor(edge_features)
        causal_direction = self.direction_predictor(edge_features)

        return {
            "causal_strength": causal_strength,
            "causal_direction": causal_direction,
            "node_embeddings": attended_features,
            "attention_weights": attention_weights,
        }

    def discover_causal_graph(
        self, x: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Discover causal graph structure"""
        num_nodes = x.size(0)

        # Create fully connected graph for evaluation
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        # Add reverse edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Forward pass
        outputs = self.forward(x, edge_index)
        causal_strength = outputs["causal_strength"]
        causal_direction = outputs["causal_direction"]

        # Filter edges by strength threshold
        strong_edges = causal_strength.squeeze() > threshold
        filtered_edge_index = edge_index[:, strong_edges]
        filtered_strength = causal_strength[strong_edges]
        filtered_direction = causal_direction[strong_edges]

        # Create adjacency matrix
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        for i, (src, tgt) in enumerate(filtered_edge_index.t()):
            # Use direction probability to determine edge direction
            if filtered_direction[i, 0] > filtered_direction[i, 1]:  # src -> tgt
                adj_matrix[src, tgt] = filtered_strength[i]
            else:  # tgt -> src
                adj_matrix[tgt, src] = filtered_strength[i]

        return adj_matrix


class AdvancedGeologicalGNNEngine:
    """Advanced GNN engine for geological process modeling"""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.models = {}
        self.optimizers = {}
        self.training_history = {}

        logger.info(f"Initialized Advanced Geological GNN Engine on {device}")

    def create_physics_informed_model(
        self, node_features: int, edge_features: int, model_name: str = "physics_gnn"
    ) -> PhysicsInformedGNN:
        """Create physics-informed GNN model"""
        model = PhysicsInformedGNN(node_features, edge_features).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.models[model_name] = model
        self.optimizers[model_name] = optimizer
        self.training_history[model_name] = {
            "loss": [],
            "physics_loss": [],
            "pred_loss": [],
        }

        return model

    def create_heterogeneous_model(
        self,
        node_types: List[str],
        edge_types: List[str],
        model_name: str = "hetero_gnn",
    ) -> HeterogeneousGeologicalGNN:
        """Create heterogeneous GNN model"""
        model = HeterogeneousGeologicalGNN(node_types, edge_types).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.models[model_name] = model
        self.optimizers[model_name] = optimizer
        self.training_history[model_name] = {"loss": []}

        return model

    def create_variational_model(
        self, input_dim: int, model_name: str = "vgae"
    ) -> VariationalGraphAutoencoder:
        """Create variational graph autoencoder"""
        model = VariationalGraphAutoencoder(input_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        self.models[model_name] = model
        self.optimizers[model_name] = optimizer
        self.training_history[model_name] = {
            "total_loss": [],
            "recon_loss": [],
            "kl_loss": [],
        }

        return model

    def create_causal_discovery_model(
        self, node_features: int, model_name: str = "causal_gnn"
    ) -> CausalDiscoveryGNN:
        """Create causal discovery GNN model"""
        model = CausalDiscoveryGNN(node_features).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.models[model_name] = model
        self.optimizers[model_name] = optimizer
        self.training_history[model_name] = {"loss": [], "causal_accuracy": []}

        return model

    def train_physics_model(
        self,
        model_name: str,
        data_loader,
        epochs: int = 1000,
        physics_weight: float = 0.1,
    ) -> Dict[str, List[float]]:
        """Train physics-informed GNN model"""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        optimizer = self.optimizers[model_name]

        model.train()

        logger.info(f"Starting training for {model_name} with {epochs} epochs")

        for epoch in range(epochs):
            total_loss = 0.0
            total_physics_loss = 0.0
            total_pred_loss = 0.0

            for batch in data_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch)

                # Prediction loss
                if hasattr(batch, "y"):
                    pred_loss = F.mse_loss(outputs["node_predictions"], batch.y)
                else:
                    pred_loss = torch.tensor(0.0)

                # Physics loss
                physics_loss = model.physics_loss(outputs, batch)

                # Combined loss
                loss = pred_loss + physics_weight * physics_loss

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_physics_loss += physics_loss.item()
                total_pred_loss += pred_loss.item()

            # Average losses
            avg_loss = total_loss / len(data_loader)
            avg_physics_loss = total_physics_loss / len(data_loader)
            avg_pred_loss = total_pred_loss / len(data_loader)

            # Store history
            self.training_history[model_name]["loss"].append(avg_loss)
            self.training_history[model_name]["physics_loss"].append(avg_physics_loss)
            self.training_history[model_name]["pred_loss"].append(avg_pred_loss)

            if epoch % 100 == 0:
                logger.info(
                    f"Epoch {epoch}: Loss = {avg_loss:.6f}, "
                    f"Physics Loss = {avg_physics_loss:.6f}, "
                    f"Pred Loss = {avg_pred_loss:.6f}"
                )

        logger.info(f"Training completed for {model_name}")
        return self.training_history[model_name]

    def train_causal_model(
        self, model_name: str, data_loader, epochs: int = 1000
    ) -> Dict[str, List[float]]:
        """Train causal discovery GNN model"""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        optimizer = self.optimizers[model_name]

        model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for batch in data_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch.x, batch.edge_index)

                # Causal strength loss
                if hasattr(batch, "causal_labels"):
                    strength_loss = F.binary_cross_entropy(
                        outputs["causal_strength"].squeeze(),
                        batch.causal_labels.float(),
                    )

                    # Direction loss
                    if hasattr(batch, "direction_labels"):
                        direction_loss = F.cross_entropy(
                            outputs["causal_direction"], batch.direction_labels
                        )
                    else:
                        direction_loss = torch.tensor(0.0)

                    loss = strength_loss + direction_loss

                    # Calculate accuracy
                    pred_strength = (outputs["causal_strength"] > 0.5).float()
                    correct_predictions += (
                        (pred_strength.squeeze() == batch.causal_labels.float())
                        .sum()
                        .item()
                    )
                    total_predictions += batch.causal_labels.size(0)

                else:
                    # Unsupervised training - use consistency loss
                    loss = torch.mean(outputs["causal_strength"])  # Encourage sparsity

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Calculate metrics
            avg_loss = total_loss / len(data_loader)
            accuracy = (
                correct_predictions / total_predictions
                if total_predictions > 0
                else 0.0
            )

            # Store history
            self.training_history[model_name]["loss"].append(avg_loss)
            self.training_history[model_name]["causal_accuracy"].append(accuracy)

            if epoch % 100 == 0:
                logger.info(
                    f"Epoch {epoch}: Loss = {avg_loss:.6f}, Accuracy = {accuracy:.4f}"
                )

        return self.training_history[model_name]

    def predict_geological_relationships(
        self, model_name: str, data: Data
    ) -> Dict[str, torch.Tensor]:
        """Predict geological relationships using trained model"""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        model.eval()

        data = data.to(self.device)

        with torch.no_grad():
            if isinstance(model, PhysicsInformedGNN):
                outputs = model(data)
            elif isinstance(model, CausalDiscoveryGNN):
                outputs = model(data.x, data.edge_index)
            elif isinstance(model, VariationalGraphAutoencoder):
                outputs = model(data.x, data.edge_index)
            else:
                raise ValueError(f"Unknown model type for {model_name}")

        return outputs

    def discover_causal_network(
        self, model_name: str, node_features: torch.Tensor, threshold: float = 0.5
    ) -> nx.DiGraph:
        """Discover causal network structure"""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]

        if not isinstance(model, CausalDiscoveryGNN):
            raise ValueError(f"Model {model_name} is not a causal discovery model")

        model.eval()
        node_features = node_features.to(self.device)

        with torch.no_grad():
            adj_matrix = model.discover_causal_graph(node_features, threshold)

        # Convert to NetworkX graph
        adj_matrix_np = adj_matrix.cpu().numpy()
        G = nx.from_numpy_array(adj_matrix_np, create_using=nx.DiGraph)

        # Add edge weights
        for i, j, data in G.edges(data=True):
            data["weight"] = adj_matrix_np[i, j]
            data["causal_strength"] = adj_matrix_np[i, j]

        return G

    def visualize_causal_network(
        self,
        G: nx.DiGraph,
        node_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ):
        """Visualize discovered causal network"""

        plt.figure(figsize=(12, 8))

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes
        node_colors = [G.degree(node) for node in G.nodes()]
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            node_size=500,
            alpha=0.8,
        )

        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]["causal_strength"] for u, v in edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            width=weights,
            alpha=0.6,
            edge_color="gray",
            arrows=True,
            arrowsize=20,
        )

        # Draw labels
        if node_labels:
            labels = {i: label for i, label in enumerate(node_labels)}
            nx.draw_networkx_labels(G, pos, labels, font_size=10)

        plt.title("Discovered Causal Network of Geological Processes")
        plt.axis("o")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def save_model(self, model_name: str, filepath: str):
        """Save trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        torch.save(
            {
                "model_state_dict": self.models[model_name].state_dict(),
                "optimizer_state_dict": self.optimizers[model_name].state_dict(),
                "training_history": self.training_history[model_name],
            },
            filepath,
        )

        logger.info(f"Model {model_name} saved to {filepath}")

    def load_model(self, model_name: str, filepath: str, model_class, **model_kwargs):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Create model based on class
        if model_class == PhysicsInformedGNN:
            model = self.create_physics_informed_model(
                **model_kwargs, model_name=model_name
            )
        elif model_class == CausalDiscoveryGNN:
            model = self.create_causal_discovery_model(
                **model_kwargs, model_name=model_name
            )
        elif model_class == VariationalGraphAutoencoder:
            model = self.create_variational_model(**model_kwargs, model_name=model_name)
        else:
            raise ValueError(f"Unknown model class: {model_class}")

        # Load state
        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizers[model_name].load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history[model_name] = checkpoint["training_history"]

        logger.info(f"Model {model_name} loaded from {filepath}")
        return model
