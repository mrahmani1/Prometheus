"""
Advanced Geological Graph Attention Networks (GATs)
Specialized GATs for geological relationship weighting with multi-head attention, hierarchical attention, and physics-informed constraints
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import math
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans, DBSCAN
import pickle
import hashlib
import os
from pathlib import Path

# DGL imports with fallback
try:
    import dgl
    import dgl.nn as dglnn
    from dgl import DGLGraph
    from dgl.data import DGLDataset
    from dgl.nn import GATConv, GATv2Conv
    from dgl.ops import edge_softmax
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    warnings.warn("DGL not available. GAT will use PyTorch Geometric fallback.")

# PyTorch Geometric fallback
try:
    import torch_geometric as pyg
    from torch_geometric.nn import GATConv as PyGGATConv, GATv2Conv as PyGGATv2Conv
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn("PyTorch Geometric not available. Using custom GAT implementation.")

warnings.filterwarnings('ignore')

class GeologicalRelationshipType(Enum):
    SPATIAL_PROXIMITY = "spatial_proximity"
    STRUCTURAL_CONTINUITY = "structural_continuity"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    CAUSAL_DEPENDENCY = "causal_dependency"
    CHEMICAL_SIMILARITY = "chemical_similarity"
    PHYSICAL_INTERACTION = "physical_interaction"
    PROCESS_COUPLING = "process_coupling"
    HIERARCHICAL_CONTAINMENT = "hierarchical_containment"

class AttentionMechanism(Enum):
    ADDITIVE = "additive"
    DOT_PRODUCT = "dot_product"
    SCALED_DOT_PRODUCT = "scaled_dot_product"
    MULTI_HEAD = "multi_head"
    HIERARCHICAL = "hierarchical"
    PHYSICS_INFORMED = "physics_informed"

@dataclass
class GeologicalGATConfig:
    """Configuration for Geological Graph Attention Networks"""
    # Model architecture
    input_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    output_dim: int = 64
    num_layers: int = 3
    
    # Attention configuration
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    attention_mechanism: AttentionMechanism = AttentionMechanism.MULTI_HEAD
    use_edge_features: bool = True
    edge_feature_dim: int = 32
    
    # Hierarchical attention
    use_hierarchical_attention: bool = True
    hierarchy_levels: List[str] = field(default_factory=lambda: ['local', 'regional', 'global'])
    level_attention_heads: Dict[str, int] = field(default_factory=lambda: {'local': 4, 'regional': 6, 'global': 8})
    
    # Physics-informed constraints
    use_physics_constraints: bool = True
    physics_feature_dim: int = 16
    constraint_weight: float = 0.1
    
    # Relationship weighting
    relationship_types: List[GeologicalRelationshipType] = field(default_factory=lambda: list(GeologicalRelationshipType))
    relationship_weights: Dict[str, float] = field(default_factory=lambda: {rel.value: 1.0 for rel in GeologicalRelationshipType})
    adaptive_weighting: bool = True
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    dropout: float = 0.1
    batch_size: int = 512
    epochs: int = 150
    early_stopping_patience: int = 15
    
    # Regularization
    attention_regularization: float = 0.01
    sparsity_regularization: float = 0.005
    consistency_regularization: float = 0.02
    
    # Advanced features
    use_graph_transformer: bool = True
    use_positional_encoding: bool = True
    max_path_length: int = 5
    use_cross_attention: bool = True

@dataclass
class GeologicalRelationship:
    """Geological relationship with attention weights"""
    source_id: str
    target_id: str
    relationship_type: GeologicalRelationshipType
    strength: float
    confidence: float
    features: np.ndarray
    spatial_distance: float
    temporal_distance: float
    physics_properties: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert relationship to feature vector"""
        feature_vec = [
            self.strength,
            self.confidence,
            self.spatial_distance,
            self.temporal_distance
        ]
        
        # Add relationship type encoding
        rel_type_encoding = [0.0] * len(GeologicalRelationshipType)
        rel_type_encoding[list(GeologicalRelationshipType).index(self.relationship_type)] = 1.0
        feature_vec.extend(rel_type_encoding)
        
        # Add features
        feature_vec.extend(self.features)
        
        # Add physics properties
        if self.physics_properties:
            feature_vec.extend(list(self.physics_properties.values()))
        
        return np.array(feature_vec)

class GeologicalMultiHeadAttention(nn.Module):
    """Multi-head attention with geological relationship weighting"""
    
    def __init__(self, embed_dim: int, num_heads: int, 
                 edge_feature_dim: int = 0, dropout: float = 0.1,
                 use_physics_constraints: bool = False):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.edge_feature_dim = edge_feature_dim
        self.use_physics_constraints = use_physics_constraints
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Edge feature projection
        if edge_feature_dim > 0:
            self.edge_proj = nn.Linear(edge_feature_dim, num_heads)
        
        # Physics constraint projection
        if use_physics_constraints:
            self.physics_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, num_heads)
            )
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Attention dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None,
                edge_indices: Optional[torch.Tensor] = None,
                physics_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with geological relationship weighting"""
        
        batch_size, seq_len, _ = query.size()
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add edge feature bias
        if edge_features is not None and self.edge_feature_dim > 0:
            edge_bias = self.edge_proj(edge_features)
            if edge_indices is not None:
                # Apply edge bias only to connected nodes
                edge_bias_matrix = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=scores.device)
                src_indices, dst_indices = edge_indices
                edge_bias_matrix[:, :, src_indices, dst_indices] = edge_bias.transpose(0, 1)
                scores = scores + edge_bias_matrix
        
        # Add physics constraint bias
        if self.use_physics_constraints and physics_features is not None:
            physics_bias = self.physics_proj(physics_features)
            # Broadcast physics bias across attention heads
            physics_bias = physics_bias.unsqueeze(-1).expand(-1, -1, -1, seq_len)
            scores = scores + physics_bias
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        output = self.out_proj(attended_values)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output, attention_weights.mean(dim=1)  # Average across heads

class HierarchicalAttentionLayer(nn.Module):
    """Hierarchical attention for multi-scale geological analysis"""
    
    def __init__(self, config: GeologicalGATConfig):
        super().__init__()
        
        self.config = config
        self.hierarchy_levels = config.hierarchy_levels
        
        # Attention layers for each hierarchy level
        self.level_attentions = nn.ModuleDict()
        for level in self.hierarchy_levels:
            num_heads = config.level_attention_heads.get(level, config.num_attention_heads)
            self.level_attentions[level] = GeologicalMultiHeadAttention(
                embed_dim=config.input_dim,
                num_heads=num_heads,
                edge_feature_dim=config.edge_feature_dim,
                dropout=config.attention_dropout,
                use_physics_constraints=config.use_physics_constraints
            )
        
        # Level fusion network
        self.level_fusion = nn.Sequential(
            nn.Linear(config.input_dim * len(self.hierarchy_levels), config.input_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.input_dim, config.input_dim)
        )
        
        # Level importance weights (learnable)
        self.level_weights = nn.Parameter(torch.ones(len(self.hierarchy_levels)) / len(self.hierarchy_levels))
        
    def forward(self, node_features: torch.Tensor, 
                level_adjacencies: Dict[str, torch.Tensor],
                level_edge_features: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with hierarchical attention"""
        
        level_outputs = []
        level_attentions = {}
        
        for i, level in enumerate(self.hierarchy_levels):
            # Get level-specific adjacency and edge features
            level_adj = level_adjacencies.get(level)
            level_edges = level_edge_features.get(level) if level_edge_features else None
            
            if level_adj is not None:
                # Apply level-specific attention
                level_output, level_attention = self.level_attentions[level](
                    node_features, node_features, node_features,
                    edge_features=level_edges,
                    edge_indices=level_adj
                )
                
                # Weight by learned importance
                weighted_output = self.level_weights[i] * level_output
                level_outputs.append(weighted_output)
                level_attentions[level] = level_attention
            else:
                # If no adjacency for this level, use identity
                level_outputs.append(torch.zeros_like(node_features))
                level_attentions[level] = torch.zeros(node_features.size(0), node_features.size(1), node_features.size(1))
        
        # Fuse multi-level representations
        if level_outputs:
            concatenated = torch.cat(level_outputs, dim=-1)
            fused_output = self.level_fusion(concatenated)
        else:
            fused_output = node_features
        
        return fused_output, level_attentions

class PhysicsInformedAttention(nn.Module):
    """Physics-informed attention mechanism for geological constraints"""
    
    def __init__(self, embed_dim: int, physics_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.physics_dim = physics_dim
        self.num_heads = num_heads
        
        # Physics feature encoder
        self.physics_encoder = nn.Sequential(
            nn.Linear(physics_dim, embed_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim // 2),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Physics-guided attention
        self.physics_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # Constraint validation network
        self.constraint_validator = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        # Integration layer
        self.integration_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
    
    def forward(self, node_features: torch.Tensor, 
                physics_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply physics-informed attention"""
        
        # Encode physics features
        encoded_physics = self.physics_encoder(physics_features)
        
        # Physics-guided attention
        attended_features, physics_attention = self.physics_attention(
            node_features, encoded_physics, encoded_physics
        )
        
        # Validate physics constraints
        combined_for_validation = torch.cat([node_features, attended_features], dim=-1)
        constraint_validity = self.constraint_validator(combined_for_validation)
        
        # Apply constraint weighting
        constraint_weighted = attended_features * constraint_validity
        
        # Integrate with original features
        integrated_features = torch.cat([node_features, constraint_weighted], dim=-1)
        final_features = self.integration_layer(integrated_features)
        
        return final_features, constraint_validity

class AdvancedGeologicalGAT(nn.Module):
    """Advanced Geological Graph Attention Network"""
    
    def __init__(self, config: GeologicalGATConfig):
        super().__init__()
        
        self.config = config
        self.num_layers = len(config.hidden_dims)
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dims[0]),
            nn.BatchNorm1d(config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Positional encoding
        if config.use_positional_encoding:
            self.positional_encoding = self._create_positional_encoding(config.hidden_dims[0])
        
        # Hierarchical attention layers
        if config.use_hierarchical_attention:
            self.hierarchical_attention = HierarchicalAttentionLayer(config)
        
        # Main GAT layers
        self.gat_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            input_dim = config.hidden_dims[i] if i > 0 else config.hidden_dims[0]
            output_dim = config.hidden_dims[i+1] if i < self.num_layers - 1 else config.output_dim
            
            if DGL_AVAILABLE:
                # Use DGL GAT
                gat_layer = dglnn.GATConv(
                    input_dim, output_dim // config.num_attention_heads,
                    num_heads=config.num_attention_heads,
                    feat_drop=config.dropout,
                    attn_drop=config.attention_dropout,
                    activation=F.elu if i < self.num_layers - 1 else None,
                    allow_zero_in_degree=True
                )
            else:
                # Use custom GAT
                gat_layer = GeologicalMultiHeadAttention(
                    embed_dim=input_dim,
                    num_heads=config.num_attention_heads,
                    edge_feature_dim=config.edge_feature_dim,
                    dropout=config.attention_dropout,
                    use_physics_constraints=config.use_physics_constraints
                )
            
            self.gat_layers.append(gat_layer)
        
        # Physics-informed attention
        if config.use_physics_constraints:
            self.physics_attention = PhysicsInformedAttention(
                embed_dim=config.output_dim,
                physics_dim=config.physics_feature_dim,
                num_heads=config.num_attention_heads // 2
            )
        
        # Cross-attention for multi-modal fusion
        if config.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                config.output_dim, config.num_attention_heads // 2,
                dropout=config.attention_dropout, batch_first=True
            )
        
        # Graph transformer layers
        if config.use_graph_transformer:
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config.output_dim,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.output_dim * 4,
                    dropout=config.dropout,
                    activation='gelu',
                    batch_first=True
                ) for _ in range(2)
            ])
        
        # Relationship classification heads
        self.relationship_classifiers = nn.ModuleDict()
        for rel_type in config.relationship_types:
            self.relationship_classifiers[rel_type.value] = nn.Sequential(
                nn.Linear(config.output_dim * 2, config.output_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.output_dim, 2)  # Binary classification
            )
        
        # Attention weight predictor
        self.attention_weight_predictor = nn.Sequential(
            nn.Linear(config.output_dim * 2 + config.edge_feature_dim, config.output_dim),
            nn.ReLU(),
            nn.Linear(config.output_dim, len(config.relationship_types)),
            nn.Softmax(dim=-1)
        )
        
        # Final output layers
        self.output_layers = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.output_dim // 2),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim // 2, config.output_dim)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim // 2),
            nn.ReLU(),
            nn.Linear(config.output_dim // 2, 1),
            nn.Softplus()
        )
    
    def _create_positional_encoding(self, embed_dim: int, max_len: int = 10000) -> nn.Parameter:
        """Create positional encoding for nodes"""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through advanced GAT"""
        
        node_features = graph_data['node_features']
        edge_indices = graph_data.get('edge_indices')
        edge_features = graph_data.get('edge_features')
        physics_features = graph_data.get('physics_features')
        level_adjacencies = graph_data.get('level_adjacencies', {})
        
        batch_size, num_nodes, _ = node_features.shape
        
        # Input projection
        h = self.input_projection(node_features.view(-1, node_features.size(-1)))
        h = h.view(batch_size, num_nodes, -1)
        
        # Add positional encoding
        if self.config.use_positional_encoding and hasattr(self, 'positional_encoding'):
            pos_enc = self.positional_encoding[:, :num_nodes, :].expand(batch_size, -1, -1)
            h = h + pos_enc
        
        # Hierarchical attention
        hierarchical_attentions = {}
        if self.config.use_hierarchical_attention:
            h, hierarchical_attentions = self.hierarchical_attention(h, level_adjacencies)
        
        # Main GAT layers
        layer_attentions = []
        
        for i, gat_layer in enumerate(self.gat_layers):
            if DGL_AVAILABLE:
                # Convert to DGL format
                g = self._create_dgl_graph(edge_indices, num_nodes, batch_size)
                h_flat = h.view(-1, h.size(-1))
                
                if i < len(self.gat_layers) - 1:
                    h_out = gat_layer(g, h_flat)
                    h_out = h_out.flatten(1)  # Flatten multi-head dimension
                else:
                    h_out, attention = gat_layer(g, h_flat, get_attention=True)
                    h_out = h_out.mean(1)  # Average across heads
                    layer_attentions.append(attention)
                
                h = h_out.view(batch_size, num_nodes, -1)
            else:
                # Use custom attention
                h_out, attention = gat_layer(h, h, h, edge_features, edge_indices)
                h = h_out
                layer_attentions.append(attention)
        
        # Physics-informed attention
        constraint_validities = None
        if self.config.use_physics_constraints and physics_features is not None:
            h, constraint_validities = self.physics_attention(h, physics_features)
        
        # Graph transformer layers
        if self.config.use_graph_transformer:
            h_flat = h.view(-1, h.size(-1))
            for transformer_layer in self.transformer_layers:
                h_flat = transformer_layer(h_flat.unsqueeze(0)).squeeze(0)
            h = h_flat.view(batch_size, num_nodes, -1)
        
        # Cross-attention (if multi-modal data available)
        cross_attention_weights = None
        if self.config.use_cross_attention and 'auxiliary_features' in graph_data:
            aux_features = graph_data['auxiliary_features']
            h_cross, cross_attention_weights = self.cross_attention(h, aux_features, aux_features)
            h = h + h_cross
        
        # Final output
        h_flat = h.view(-1, h.size(-1))
        embeddings = self.output_layers(h_flat)
        embeddings = embeddings.view(batch_size, num_nodes, -1)
        
        # Uncertainty estimation
        uncertainties = self.uncertainty_head(embeddings.view(-1, embeddings.size(-1)))
        uncertainties = uncertainties.view(batch_size, num_nodes, -1)
        
        # Relationship classification
        relationship_predictions = {}
        if edge_indices is not None:
            src_indices, dst_indices = edge_indices
            for rel_type in self.config.relationship_types:
                # Create edge embeddings by concatenating source and target node embeddings
                edge_embeddings = torch.cat([
                    embeddings[:, src_indices, :].view(-1, embeddings.size(-1)),
                    embeddings[:, dst_indices, :].view(-1, embeddings.size(-1))
                ], dim=-1)
                
                rel_pred = self.relationship_classifiers[rel_type.value](edge_embeddings)
                relationship_predictions[rel_type.value] = rel_pred
        
        # Attention weight prediction
        attention_weights = None
        if edge_indices is not None and edge_features is not None:
            edge_embeddings_with_features = torch.cat([
                embeddings[:, src_indices, :].view(-1, embeddings.size(-1)),
                embeddings[:, dst_indices, :].view(-1, embeddings.size(-1)),
                edge_features.view(-1, edge_features.size(-1))
            ], dim=-1)
            attention_weights = self.attention_weight_predictor(edge_embeddings_with_features)
        
        return {
            'node_embeddings': embeddings,
            'uncertainties': uncertainties,
            'layer_attentions': layer_attentions,
            'hierarchical_attentions': hierarchical_attentions,
            'relationship_predictions': relationship_predictions,
            'attention_weights': attention_weights,
            'constraint_validities': constraint_validities,
            'cross_attention_weights': cross_attention_weights
        }
    
    def _create_dgl_graph(self, edge_indices: torch.Tensor, num_nodes: int, batch_size: int):
        """Create DGL graph from edge indices"""
        if edge_indices is None:
            return dgl.graph(([], []), num_nodes=num_nodes * batch_size)
        
        src, dst = edge_indices
        # Expand for batch
        batch_src, batch_dst = [], []
        for b in range(batch_size):
            batch_src.extend([s + b * num_nodes for s in src])
            batch_dst.extend([d + b * num_nodes for d in dst])
        
        return dgl.graph((batch_src, batch_dst), num_nodes=num_nodes * batch_size)

class GeologicalGATDataset(Dataset):
    """Dataset for Geological GAT training"""
    
    def __init__(self, nodes: List[Dict], relationships: List[GeologicalRelationship], config: GeologicalGATConfig):
        self.nodes = nodes
        self.relationships = relationships
        self.config = config
        
        # Build mappings
        self.node_id_to_idx = {node['id']: i for i, node in enumerate(nodes)}
        
        # Prepare features
        self.node_features = self._prepare_node_features()
        self.edge_features, self.edge_indices = self._prepare_edge_features()
        
        # Create hierarchical adjacencies
        self.level_adjacencies = self._create_hierarchical_adjacencies()
    
    def _prepare_node_features(self) -> torch.Tensor:
        """Prepare node feature matrix"""
        feature_vectors = []
        
        for node in self.nodes:
            # Extract various feature types
            features = []
            
            # Basic features
            features.extend(node.get('coordinates', [0, 0, 0]))
            features.extend(node.get('properties', {}).values())
            
            # Geological type encoding
            geo_types = ['fault', 'fold', 'intrusion', 'sediment', 'mineral', 'structure']
            geo_type_encoding = [0.0] * len(geo_types)
            if node.get('geological_type') in geo_types:
                geo_type_encoding[geo_types.index(node['geological_type'])] = 1.0
            features.extend(geo_type_encoding)
            
            # Scale encoding
            scale_types = [scale.value for scale in GeologicalScaleType.__members__.values()]
            scale_encoding = [0.0] * len(scale_types)
            if node.get('scale_type') in scale_types:
                scale_encoding[scale_types.index(node['scale_type'])] = 1.0
            features.extend(scale_encoding)
            
            feature_vectors.append(features)
        
        # Pad to same length
        max_len = max(len(fv) for fv in feature_vectors)
        padded_features = []
        for fv in feature_vectors:
            if len(fv) < max_len:
                padded_fv = fv + [0.0] * (max_len - len(fv))
            else:
                padded_fv = fv
            padded_features.append(padded_fv)
        
        return torch.FloatTensor(padded_features)
    
    def _prepare_edge_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare edge features and indices"""
        edge_features = []
        edge_indices = [[], []]
        
        for rel in self.relationships:
            if rel.source_id in self.node_id_to_idx and rel.target_id in self.node_id_to_idx:
                src_idx = self.node_id_to_idx[rel.source_id]
                dst_idx = self.node_id_to_idx[rel.target_id]
                
                edge_indices[0].append(src_idx)
                edge_indices[1].append(dst_idx)
                
                edge_features.append(rel.to_feature_vector())
        
        if edge_features:
            # Pad edge features to same length
            max_edge_len = max(len(ef) for ef in edge_features)
            padded_edge_features = []
            for ef in edge_features:
                if len(ef) < max_edge_len:
                    padded_ef = np.pad(ef, (0, max_edge_len - len(ef)), mode='constant')
                else:
                    padded_ef = ef
                padded_edge_features.append(padded_ef)
            
            return torch.FloatTensor(padded_edge_features), torch.LongTensor(edge_indices)
        else:
            return torch.empty(0, 0), torch.empty(2, 0, dtype=torch.long)
    
    def _create_hierarchical_adjacencies(self) -> Dict[str, torch.Tensor]:
        """Create adjacency matrices for different hierarchy levels"""
        level_adjacencies = {}
        
        for level in self.config.hierarchy_levels:
            level_edges = [[], []]
            
            for rel in self.relationships:
                if rel.source_id in self.node_id_to_idx and rel.target_id in self.node_id_to_idx:
                    src_idx = self.node_id_to_idx[rel.source_id]
                    dst_idx = self.node_id_to_idx[rel.target_id]
                    
                    # Determine if relationship belongs to this level
                    if self._relationship_belongs_to_level(rel, level):
                        level_edges[0].append(src_idx)
                        level_edges[1].append(dst_idx)
            
            if level_edges[0]:
                level_adjacencies[level] = torch.LongTensor(level_edges)
            else:
                level_adjacencies[level] = torch.empty(2, 0, dtype=torch.long)
        
        return level_adjacencies
    
    def _relationship_belongs_to_level(self, rel: GeologicalRelationship, level: str) -> bool:
        """Determine if relationship belongs to hierarchy level"""
        if level == 'local':
            return rel.spatial_distance < 1000  # < 1km
        elif level == 'regional':
            return 1000 <= rel.spatial_distance < 100000  # 1km - 100km
        elif level == 'global':
            return rel.spatial_distance >= 100000  # > 100km
        else:
            return True
    
    def __len__(self):
        return 1  # Single graph dataset
    
    def __getitem__(self, idx):
        return {
            'node_features': self.node_features.unsqueeze(0),
            'edge_features': self.edge_features,
            'edge_indices': self.edge_indices,
            'level_adjacencies': self.level_adjacencies
        }

class GeologicalGATTrainer:
    """Advanced training system for Geological GAT"""
    
    def __init__(self, model: AdvancedGeologicalGAT, config: GeologicalGATConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.attention_metrics = []
    
    def setup_training(self):
        """Setup training components"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    
    def calculate_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate comprehensive loss"""
        total_loss = 0.0
        
        # Embedding consistency loss (self-supervised)
        embeddings = outputs['node_embeddings']
        batch_size, num_nodes, embed_dim = embeddings.shape
        
        # InfoNCE contrastive loss
        embeddings_flat = embeddings.view(-1, embed_dim)
        embeddings_norm = F.normalize(embeddings_flat, dim=1)
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # Create labels for contrastive learning
        labels = torch.arange(batch_size * num_nodes).to(embeddings.device)
        contrastive_loss = F.cross_entropy(similarity_matrix / 0.1, labels)
        total_loss += contrastive_loss
        
        # Relationship prediction loss
        if 'relationship_predictions' in outputs and 'relationship_targets' in targets:
            for rel_type, predictions in outputs['relationship_predictions'].items():
                if rel_type in targets['relationship_targets']:
                    rel_targets = targets['relationship_targets'][rel_type]
                    rel_loss = F.cross_entropy(predictions, rel_targets)
                    total_loss += rel_loss
        
        # Attention regularization
        if 'layer_attentions' in outputs:
            attention_reg = 0.0
            for attention in outputs['layer_attentions']:
                # Encourage attention sparsity
                attention_reg += torch.sum(attention ** 2)
            total_loss += self.config.attention_regularization * attention_reg
        
        # Physics constraint loss
        if 'constraint_validities' in outputs and outputs['constraint_validities'] is not None:
            constraint_loss = torch.mean((1 - outputs['constraint_validities']) ** 2)
            total_loss += self.config.constraint_weight * constraint_loss
        
        # Uncertainty regularization
        if 'uncertainties' in outputs:
            uncertainty_reg = torch.mean(outputs['uncertainties'])
            total_loss += 0.01 * uncertainty_reg
        
        return total_loss
    
    async def train(self, dataset: GeologicalGATDataset, val_dataset: Optional[GeologicalGATDataset] = None) -> Dict[str, Any]:
        """Main training loop"""
        
        self.setup_training()
        
        # Create data loaders
        train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) if val_dataset else None
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # Move to device
                device = next(self.model.parameters()).device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, torch.Tensor):
                                batch[key][sub_key] = sub_value.to(device)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate loss (self-supervised)
                loss = self.calculate_loss(outputs, {})
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        device = next(self.model.parameters()).device
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                batch[key] = value.to(device)
                        
                        outputs = self.model(batch)
                        val_loss += self.calculate_loss(outputs, {}).item()
                
                val_loss /= len(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Record metrics
            avg_train_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            if val_loader:
                self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loader and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            elif val_loader:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0.0
        }

# Example usage and testing
async def test_geological_gat():
    """Test Geological GAT implementation"""
    
    # Configuration
    config = GeologicalGATConfig(
        input_dim=64,
        hidden_dims=[128, 64, 32],
        output_dim=16,
        num_attention_heads=4,
        epochs=50
    )
    
    # Create synthetic data
    nodes = []
    for i in range(100):
        node = {
            'id': f'node_{i}',
            'coordinates': np.random.rand(3) * 10000,
            'geological_type': np.random.choice(['fault', 'fold', 'intrusion', 'sediment']),
            'scale_type': np.random.choice([scale.value for scale in GeologicalScaleType.__members__.values()]),
            'properties': {
                'porosity': np.random.uniform(0.1, 0.4),
                'density': np.random.uniform(2000, 3000)
            }
        }
        nodes.append(node)
    
    # Create relationships
    relationships = []
    for i in range(200):
        source_idx = np.random.randint(len(nodes))
        target_idx = np.random.randint(len(nodes))
        
        if source_idx != target_idx:
            rel = GeologicalRelationship(
                source_id=nodes[source_idx]['id'],
                target_id=nodes[target_idx]['id'],
                relationship_type=np.random.choice(list(GeologicalRelationshipType)),
                strength=np.random.uniform(0.1, 1.0),
                confidence=np.random.uniform(0.5, 1.0),
                features=np.random.rand(5),
                spatial_distance=np.random.uniform(100, 50000),
                temporal_distance=np.random.uniform(0, 1e6)
            )
            relationships.append(rel)
    
    print(f"Created dataset with {len(nodes)} nodes and {len(relationships)} relationships")
    
    # Create dataset
    dataset = GeologicalGATDataset(nodes, relationships, config)
    
    # Create model
    model = AdvancedGeologicalGAT(config)
    
    # Train model
    trainer = GeologicalGATTrainer(model, config)
    results = await trainer.train(dataset)
    
    print("Training completed!")
    print(f"Final training loss: {results['final_train_loss']:.6f}")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")

if __name__ == "__main__":
    print(f"DGL available: {DGL_AVAILABLE}")
    print(f"PyTorch Geometric available: {TORCH_GEOMETRIC_AVAILABLE}")
    
    # Run test
    asyncio.run(test_geological_gat())