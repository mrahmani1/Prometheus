"""
Deep Graph Library (DGL) Geological Graph Neural Networks
Advanced GNN models for geological structure analysis and deformation modeling
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# DGL imports with fallback
try:
    import dgl
    import dgl.nn as dglnn
    from dgl import DGLGraph
    from dgl.data import DGLDataset
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    warnings.warn("DGL not available. Graph neural network features will be limited.")

warnings.filterwarnings('ignore')

class GeologicalFeatureType(Enum):
    FAULT = "fault"
    FRACTURE = "fracture"
    BEDDING = "bedding"
    JOINT = "joint"
    DIKE = "dike"
    SILL = "sill"
    FOLD = "fold"
    CONTACT = "contact"

class GraphType(Enum):
    SPATIAL = "spatial"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    MULTI_LAYER = "multi_layer"

@dataclass
class GeologicalNode:
    """Node in geological graph representing a geological feature or measurement point"""
    id: str
    coordinates: np.ndarray  # [x, y, z]
    feature_type: GeologicalFeatureType
    properties: Dict[str, float]
    measurements: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert node to feature vector"""
        features = list(self.coordinates)
        features.extend(list(self.properties.values()))
        
        # Add one-hot encoding for feature type
        feature_type_encoding = [0.0] * len(GeologicalFeatureType)
        feature_type_encoding[list(GeologicalFeatureType).index(self.feature_type)] = 1.0
        features.extend(feature_type_encoding)
        
        return np.array(features)

@dataclass
class GeologicalEdge:
    """Edge in geological graph representing relationships between features"""
    source_id: str
    target_id: str
    edge_type: str  # 'spatial', 'structural', 'temporal', 'causal'
    weight: float
    properties: Dict[str, float] = field(default_factory=dict)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert edge to feature vector"""
        features = [self.weight]
        features.extend(list(self.properties.values()))
        return np.array(features)

class GeologicalGraphDataset:
    """Dataset for geological graph neural networks"""
    
    def __init__(self, nodes: List[GeologicalNode], edges: List[GeologicalEdge]):
        self.nodes = nodes
        self.edges = edges
        self.node_id_to_idx = {node.id: i for i, node in enumerate(nodes)}
        
        # Build feature matrices
        self.node_features = self._build_node_features()
        self.edge_features = self._build_edge_features()
        
        # Build adjacency information
        self.edge_indices = self._build_edge_indices()
        
        # Create DGL graph if available
        if DGL_AVAILABLE:
            self.dgl_graph = self._create_dgl_graph()
        else:
            self.dgl_graph = None
    
    def _build_node_features(self) -> np.ndarray:
        """Build node feature matrix"""
        if not self.nodes:
            return np.array([])
        
        feature_vectors = [node.to_feature_vector() for node in self.nodes]
        
        # Ensure all feature vectors have the same length
        max_length = max(len(fv) for fv in feature_vectors)
        padded_features = []
        
        for fv in feature_vectors:
            if len(fv) < max_length:
                padded = np.pad(fv, (0, max_length - len(fv)), mode='constant')
                padded_features.append(padded)
            else:
                padded_features.append(fv)
        
        return np.array(padded_features)
    
    def _build_edge_features(self) -> np.ndarray:
        """Build edge feature matrix"""
        if not self.edges:
            return np.array([])
        
        feature_vectors = [edge.to_feature_vector() for edge in self.edges]
        
        # Ensure all feature vectors have the same length
        if feature_vectors:
            max_length = max(len(fv) for fv in feature_vectors)
            padded_features = []
            
            for fv in feature_vectors:
                if len(fv) < max_length:
                    padded = np.pad(fv, (0, max_length - len(fv)), mode='constant')
                    padded_features.append(padded)
                else:
                    padded_features.append(fv)
            
            return np.array(padded_features)
        else:
            return np.array([])
    
    def _build_edge_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build edge indices for graph connectivity"""
        if not self.edges:
            return np.array([]), np.array([])
        
        source_indices = []
        target_indices = []
        
        for edge in self.edges:
            if edge.source_id in self.node_id_to_idx and edge.target_id in self.node_id_to_idx:
                source_indices.append(self.node_id_to_idx[edge.source_id])
                target_indices.append(self.node_id_to_idx[edge.target_id])
        
        return np.array(source_indices), np.array(target_indices)
    
    def _create_dgl_graph(self) -> Optional['DGLGraph']:
        """Create DGL graph from nodes and edges"""
        if not DGL_AVAILABLE or not self.edges:
            return None
        
        src_indices, dst_indices = self.edge_indices
        
        if len(src_indices) == 0:
            # Create graph with nodes but no edges
            g = dgl.graph(([], []), num_nodes=len(self.nodes))
        else:
            g = dgl.graph((src_indices, dst_indices), num_nodes=len(self.nodes))
        
        # Add node features
        g.ndata['feat'] = torch.FloatTensor(self.node_features)
        
        # Add edge features
        if len(self.edge_features) > 0:
            g.edata['feat'] = torch.FloatTensor(self.edge_features)
        
        return g

class GeologicalGCN(nn.Module):
    """Graph Convolutional Network for geological analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout: float = 0.1):
        super(GeologicalGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers
        if DGL_AVAILABLE:
            self.gcn_layers = nn.ModuleList([
                dglnn.GraphConv(hidden_dim, hidden_dim, activation=F.relu)
                for _ in range(num_layers - 1)
            ])
            self.gcn_layers.append(dglnn.GraphConv(hidden_dim, hidden_dim))
        else:
            # Fallback: simple MLP layers
            self.gcn_layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
        
        # Output layers
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, g, features):
        """Forward pass through GCN"""
        if not DGL_AVAILABLE:
            # Fallback: simple MLP
            h = self.input_proj(features)
            for layer in self.gcn_layers:
                h = F.relu(layer(h))
                h = self.dropout_layer(h)
            return self.output_proj(h)
        
        # DGL-based computation
        h = self.input_proj(features)
        
        for i, layer in enumerate(self.gcn_layers):
            h = layer(g, h)
            if i < len(self.gcn_layers) - 1:
                h = F.relu(h)
            h = self.dropout_layer(h)
        
        return self.output_proj(h)

class GeologicalGAT(nn.Module):
    """Graph Attention Network for geological analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_heads: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super(GeologicalGAT, self).__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        if DGL_AVAILABLE:
            self.gat_layers = nn.ModuleList()
            
            # First layer
            self.gat_layers.append(
                dglnn.GATConv(hidden_dim, hidden_dim // num_heads, num_heads, dropout=dropout, activation=F.elu)
            )
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.gat_layers.append(
                    dglnn.GATConv(hidden_dim, hidden_dim // num_heads, num_heads, dropout=dropout, activation=F.elu)
                )
            
            # Output layer
            self.gat_layers.append(
                dglnn.GATConv(hidden_dim, output_dim, 1, dropout=dropout, activation=None)
            )
        else:
            # Fallback: self-attention layers
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, g, features):
        """Forward pass through GAT"""
        if not DGL_AVAILABLE:
            # Fallback: self-attention
            h = self.input_proj(features)
            h = h.unsqueeze(1)  # Add sequence dimension
            
            for attention in self.attention_layers:
                h, _ = attention(h, h, h)
                h = self.dropout_layer(h)
            
            h = h.squeeze(1)  # Remove sequence dimension
            return self.output_proj(h)
        
        # DGL-based computation
        h = self.input_proj(features)
        
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i < len(self.gat_layers) - 1:
                h = h.flatten(1)  # Flatten multi-head dimension
            h = self.dropout_layer(h)
        
        return h.mean(1)  # Average over heads for final output

class GeologicalGraphSAGE(nn.Module):
    """GraphSAGE model for geological analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, aggregator_type: str = 'mean'):
        super(GeologicalGraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        if DGL_AVAILABLE:
            # SAGE layers
            self.sage_layers = nn.ModuleList([
                dglnn.SAGEConv(hidden_dim, hidden_dim, aggregator_type)
                for _ in range(num_layers - 1)
            ])
            self.sage_layers.append(dglnn.SAGEConv(hidden_dim, output_dim, aggregator_type))
        else:
            # Fallback: simple layers
            self.layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
            ])
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, g, features):
        """Forward pass through GraphSAGE"""
        if not DGL_AVAILABLE:
            # Fallback computation
            h = self.input_proj(features)
            for i, layer in enumerate(self.layers):
                h = layer(h)
                if i < len(self.layers) - 1:
                    h = F.relu(h)
                h = self.dropout(h)
            return h
        
        # DGL-based computation
        h = self.input_proj(features)
        
        for i, layer in enumerate(self.sage_layers):
            h = layer(g, h)
            if i < len(self.sage_layers) - 1:
                h = F.relu(h)
            h = self.dropout(h)
        
        return h

class GeologicalRelationPredictor(nn.Module):
    """GNN model for predicting geological relationships"""
    
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, 
                 hidden_dim: int = 128, num_relation_types: int = 8):
        super().__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Graph neural network
        if DGL_AVAILABLE:
            self.gnn = dglnn.GATConv(
                hidden_dim, hidden_dim, num_heads=4, dropout=0.1
            )
        else:
            self.gnn = nn.Linear(hidden_dim, hidden_dim)
        
        # Relation prediction head
        self.relation_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_relation_types)
        )
    
    def forward(self, g, node_features, edge_features, edge_pairs):
        """Predict geological relationships between node pairs"""
        # Encode nodes
        node_embeddings = self.node_encoder(node_features)
        
        # Apply GNN
        if DGL_AVAILABLE and g is not None:
            node_embeddings = self.gnn(g, node_embeddings).mean(1)  # Average over heads
        else:
            node_embeddings = F.relu(self.gnn(node_embeddings))
        
        # Encode edges
        if edge_features.size(0) > 0:
            edge_embeddings = self.edge_encoder(edge_features)
        else:
            edge_embeddings = torch.zeros(len(edge_pairs), self.edge_encoder[-1].out_features)
        
        # Predict relationships for edge pairs
        predictions = []
        for i, (src, dst) in enumerate(edge_pairs):
            src_emb = node_embeddings[src]
            dst_emb = node_embeddings[dst]
            edge_emb = edge_embeddings[i] if i < len(edge_embeddings) else torch.zeros(edge_embeddings.size(1))
            
            # Combine embeddings
            combined = torch.cat([src_emb, dst_emb, edge_emb])
            prediction = self.relation_predictor(combined)
            predictions.append(prediction)
        
        return torch.stack(predictions) if predictions else torch.zeros(0, self.relation_predictor[-1].out_features)

class GeologicalDeformationGNN(nn.Module):
    """GNN for predicting geological deformation patterns"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 3):
        super().__init__()
        
        # Feature encoders
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),  # x, y, z coordinates
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.property_encoder = nn.Sequential(
            nn.Linear(input_dim - 3, hidden_dim * 3 // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 3 // 4),
            nn.Linear(hidden_dim * 3 // 4, hidden_dim * 3 // 4)
        )
        
        # Graph neural network layers
        if DGL_AVAILABLE:
            self.gnn_layers = nn.ModuleList([
                dglnn.GraphConv(hidden_dim, hidden_dim, activation=F.relu),
                dglnn.GATConv(hidden_dim, hidden_dim // 4, num_heads=4, dropout=0.1),
                dglnn.SAGEConv(hidden_dim, hidden_dim, 'mean')
            ])
        else:
            self.gnn_layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
            ])
        
        # Physics-informed layers
        self.physics_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Output layers for deformation components
        self.displacement_head = nn.Linear(hidden_dim // 2, output_dim)
        self.uncertainty_head = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, g, features):
        """Predict deformation patterns"""
        # Split features
        spatial_features = features[:, :3]  # x, y, z
        property_features = features[:, 3:]  # geological properties
        
        # Encode features
        spatial_emb = self.spatial_encoder(spatial_features)
        
        if property_features.size(1) > 0:
            property_emb = self.property_encoder(property_features)
            combined_features = torch.cat([spatial_emb, property_emb], dim=1)
        else:
            combined_features = spatial_emb
            # Pad to expected size
            padding_size = self.property_encoder[-1].out_features
            padding = torch.zeros(spatial_emb.size(0), padding_size)
            combined_features = torch.cat([spatial_emb, padding], dim=1)
        
        # Apply GNN layers
        h = combined_features
        
        if DGL_AVAILABLE and g is not None:
            for i, layer in enumerate(self.gnn_layers):
                if isinstance(layer, dglnn.GATConv):
                    h = layer(g, h).mean(1)  # Average over attention heads
                else:
                    h = layer(g, h)
                if i < len(self.gnn_layers) - 1:
                    h = F.relu(h)
        else:
            # Fallback computation
            for i, layer in enumerate(self.gnn_layers):
                h = layer(h)
                if i < len(self.gnn_layers) - 1:
                    h = F.relu(h)
        
        # Physics-informed processing
        physics_features = self.physics_layer(h)
        
        # Predict displacements and uncertainties
        displacements = self.displacement_head(physics_features)
        uncertainties = torch.abs(self.uncertainty_head(physics_features))  # Ensure positive
        
        return displacements, uncertainties

class GeologicalGraphBuilder:
    """Build geological graphs from spatial and structural data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default graph building configuration"""
        return {
            'spatial_threshold': 1000,  # meters
            'structural_threshold': 30,  # degrees for orientation similarity
            'k_nearest_neighbors': 5,
            'edge_types': ['spatial', 'structural', 'temporal'],
            'node_features': ['coordinates', 'orientation', 'properties'],
            'graph_type': GraphType.MULTI_LAYER
        }
    
    def build_spatial_graph(self, nodes: List[GeologicalNode]) -> GeologicalGraphDataset:
        """Build spatial connectivity graph"""
        edges = []
        
        for i, node1 in enumerate(nodes):
            distances = []
            
            for j, node2 in enumerate(nodes):
                if i != j:
                    distance = np.linalg.norm(node1.coordinates - node2.coordinates)
                    distances.append((j, distance))
            
            # Sort by distance and take k nearest neighbors
            distances.sort(key=lambda x: x[1])
            k_nearest = distances[:self.config['k_nearest_neighbors']]
            
            # Create edges to nearest neighbors within threshold
            for neighbor_idx, distance in k_nearest:
                if distance <= self.config['spatial_threshold']:
                    edge = GeologicalEdge(
                        source_id=node1.id,
                        target_id=nodes[neighbor_idx].id,
                        edge_type='spatial',
                        weight=1.0 / (1.0 + distance),  # Inverse distance weighting
                        properties={'distance': distance}
                    )
                    edges.append(edge)
        
        return GeologicalGraphDataset(nodes, edges)
    
    def build_structural_graph(self, nodes: List[GeologicalNode]) -> GeologicalGraphDataset:
        """Build structural relationship graph based on geological features"""
        edges = []
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], start=i+1):
                # Check for structural relationships
                relationship_type, strength = self._analyze_structural_relationship(node1, node2)
                
                if relationship_type and strength > 0.1:
                    edge = GeologicalEdge(
                        source_id=node1.id,
                        target_id=node2.id,
                        edge_type='structural',
                        weight=strength,
                        properties={
                            'relationship_type': relationship_type,
                            'structural_strength': strength
                        }
                    )
                    edges.append(edge)
        
        return GeologicalGraphDataset(nodes, edges)
    
    def _analyze_structural_relationship(self, node1: GeologicalNode, 
                                       node2: GeologicalNode) -> Tuple[Optional[str], float]:
        """Analyze structural relationship between two nodes"""
        # Check feature type compatibility
        compatible_pairs = {
            (GeologicalFeatureType.FAULT, GeologicalFeatureType.FAULT): 'fault_intersection',
            (GeologicalFeatureType.DIKE, GeologicalFeatureType.SILL): 'intrusion_complex',
            (GeologicalFeatureType.FOLD, GeologicalFeatureType.FAULT): 'fault_fold_relationship',
            (GeologicalFeatureType.BEDDING, GeologicalFeatureType.FOLD): 'folded_bedding'
        }
        
        pair = (node1.feature_type, node2.feature_type)
        reverse_pair = (node2.feature_type, node1.feature_type)
        
        relationship_type = compatible_pairs.get(pair) or compatible_pairs.get(reverse_pair)
        
        if not relationship_type:
            return None, 0.0
        
        # Calculate relationship strength based on spatial proximity and orientation
        distance = np.linalg.norm(node1.coordinates - node2.coordinates)
        spatial_strength = np.exp(-distance / self.config['spatial_threshold'])
        
        # Orientation similarity (if available)
        orientation_strength = 1.0
        if 'strike' in node1.properties and 'strike' in node2.properties:
            strike_diff = abs(node1.properties['strike'] - node2.properties['strike'])
            strike_diff = min(strike_diff, 360 - strike_diff)  # Angular difference
            orientation_strength = 1.0 - (strike_diff / 180.0)
        
        total_strength = spatial_strength * orientation_strength
        
        return relationship_type, total_strength
    
    def build_temporal_graph(self, nodes: List[GeologicalNode]) -> GeologicalGraphDataset:
        """Build temporal relationship graph"""
        edges = []
        
        # Sort nodes by age if available
        nodes_with_age = [(node, node.properties.get('age', 0)) for node in nodes]
        nodes_with_age.sort(key=lambda x: x[1])
        
        for i, (node1, age1) in enumerate(nodes_with_age):
            for j, (node2, age2) in enumerate(nodes_with_age[i+1:], start=i+1):
                age_diff = abs(age2 - age1)
                
                if age_diff > 0:  # Temporal relationship exists
                    temporal_strength = 1.0 / (1.0 + age_diff / 1e6)  # Age in years
                    
                    edge = GeologicalEdge(
                        source_id=node1.id,
                        target_id=node2.id,
                        edge_type='temporal',
                        weight=temporal_strength,
                        properties={
                            'age_difference': age_diff,
                            'temporal_order': 'older' if age1 < age2 else 'younger'
                        }
                    )
                    edges.append(edge)
        
        return GeologicalGraphDataset(nodes, edges)
    
    def build_multi_layer_graph(self, nodes: List[GeologicalNode]) -> GeologicalGraphDataset:
        """Build multi-layer graph combining spatial, structural, and temporal relationships"""
        all_edges = []
        
        # Spatial layer
        spatial_dataset = self.build_spatial_graph(nodes)
        all_edges.extend(spatial_dataset.edges)
        
        # Structural layer
        structural_dataset = self.build_structural_graph(nodes)
        all_edges.extend(structural_dataset.edges)
        
        # Temporal layer
        temporal_dataset = self.build_temporal_graph(nodes)
        all_edges.extend(temporal_dataset.edges)
        
        return GeologicalGraphDataset(nodes, all_edges)

class GeologicalGNNTrainer:
    """Training framework for geological GNN models"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        self.metric_history = []
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration"""
        return {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'validation_split': 0.2,
            'loss_function': 'mse',
            'metrics': ['mae', 'r2'],
            'scheduler': 'cosine',
            'gradient_clipping': 1.0
        }
    
    def setup_training(self):
        """Setup training components"""
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        if self.config['scheduler'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config['epochs']
            )
        elif self.config['scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
    
    async def train(self, train_data: GeologicalGraphDataset, 
                   val_data: Optional[GeologicalGraphDataset] = None,
                   targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Train the GNN model"""
        
        self.setup_training()
        
        # Convert to tensors
        if train_data.dgl_graph is not None:
            train_graph = train_data.dgl_graph
            train_features = train_graph.ndata['feat']
        else:
            train_graph = None
            train_features = torch.FloatTensor(train_data.node_features)
        
        if targets is None:
            # Generate synthetic targets for demonstration
            targets = torch.randn(len(train_features), 3)  # 3D displacement
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, GeologicalDeformationGNN):
                predictions, uncertainties = self.model(train_graph, train_features)
                loss = self._calculate_loss(predictions, targets, uncertainties)
            else:
                predictions = self.model(train_graph, train_features)
                loss = self._calculate_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['gradient_clipping']
                )
            
            self.optimizer.step()
            
            # Validation phase
            val_loss = 0.0
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    if val_data.dgl_graph is not None:
                        val_graph = val_data.dgl_graph
                        val_features = val_graph.ndata['feat']
                    else:
                        val_graph = None
                        val_features = torch.FloatTensor(val_data.node_features)
                    
                    val_targets = targets[:len(val_features)]  # Adjust size if needed
                    
                    if isinstance(self.model, GeologicalDeformationGNN):
                        val_predictions, val_uncertainties = self.model(val_graph, val_features)
                        val_loss = self._calculate_loss(val_predictions, val_targets, val_uncertainties).item()
                    else:
                        val_predictions = self.model(val_graph, val_features)
                        val_loss = self._calculate_loss(val_predictions, val_targets).item()
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Record metrics
            self.loss_history.append(loss.item())
            if val_data is not None:
                self.metric_history.append({'epoch': epoch, 'train_loss': loss.item(), 'val_loss': val_loss})
            
            # Early stopping
            if val_data is not None:
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config['early_stopping_patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")
        
        return {
            'final_train_loss': self.loss_history[-1],
            'final_val_loss': val_loss if val_data else None,
            'best_val_loss': best_loss if val_data else None,
            'epochs_trained': len(self.loss_history),
            'loss_history': self.loss_history,
            'metric_history': self.metric_history
        }
    
    def _calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                      uncertainties: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate training loss"""
        if self.config['loss_function'] == 'mse':
            loss = F.mse_loss(predictions, targets)
        elif self.config['loss_function'] == 'mae':
            loss = F.l1_loss(predictions, targets)
        elif self.config['loss_function'] == 'huber':
            loss = F.smooth_l1_loss(predictions, targets)
        else:
            loss = F.mse_loss(predictions, targets)
        
        # Add uncertainty-based weighting if available
        if uncertainties is not None:
            # Weighted loss based on uncertainty
            weights = 1.0 / (uncertainties + 1e-8)
            weighted_residuals = weights * (predictions - targets) ** 2
            loss = weighted_residuals.mean()
        
        return loss

# Example usage and testing
async def example_geological_gnn():
    """Example of using geological GNN"""
    
    # Create synthetic geological nodes
    nodes = []
    np.random.seed(42)
    
    for i in range(50):
        node = GeologicalNode(
            id=f"node_{i}",
            coordinates=np.random.rand(3) * 10000,  # 10km x 10km x 10km
            feature_type=np.random.choice(list(GeologicalFeatureType)),
            properties={
                'strike': np.random.uniform(0, 360),
                'dip': np.random.uniform(0, 90),
                'age': np.random.uniform(0, 100e6)  # 0-100 Ma
            }
        )
        nodes.append(node)
    
    # Build graph
    graph_builder = GeologicalGraphBuilder()
    graph_dataset = graph_builder.build_multi_layer_graph(nodes)
    
    print(f"Created graph with {len(nodes)} nodes and {len(graph_dataset.edges)} edges")
    
    # Create and train model
    input_dim = graph_dataset.node_features.shape[1]
    model = GeologicalDeformationGNN(input_dim, hidden_dim=128, output_dim=3)
    
    # Training
    trainer = GeologicalGNNTrainer(model)
    training_results = await trainer.train(graph_dataset)
    
    print("Training completed!")
    print(f"Final training loss: {training_results['final_train_loss']:.6f}")
    print(f"Epochs trained: {training_results['epochs_trained']}")

if __name__ == "__main__":
    print(f"DGL available: {DGL_AVAILABLE}")
    
    # Run example
    asyncio.run(example_geological_gnn())