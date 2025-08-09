"""
Large-Scale Geological GraphSAGE Implementation
Advanced GraphSAGE with distributed processing, mini-batch training, and geological domain specialization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle
import hashlib
import os
from pathlib import Path
import psutil
import gc

# DGL imports with fallback
try:
    import dgl
    import dgl.nn as dglnn
    from dgl import DGLGraph
    from dgl.data import DGLDataset
    from dgl.dataloading import GraphDataLoader, MultiLayerNeighborSampler
    from dgl.sampling import sample_neighbors
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    warnings.warn("DGL not available. GraphSAGE will use simplified implementations.")

# Redis for distributed caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Distributed caching will be limited.")

# Graph analytics
try:
    import networkx as nx
    import igraph as ig
    GRAPH_LIBS_AVAILABLE = True
except ImportError:
    GRAPH_LIBS_AVAILABLE = False
    warnings.warn("Advanced graph libraries not available.")

warnings.filterwarnings('ignore')

class GeologicalScaleType(Enum):
    MINERAL = "mineral"
    OUTCROP = "outcrop"
    FORMATION = "formation"
    BASIN = "basin"
    REGIONAL = "regional"
    CONTINENTAL = "continental"
    GLOBAL = "global"

class SamplingStrategy(Enum):
    UNIFORM = "uniform"
    IMPORTANCE = "importance"
    FASTGCN = "fastgcn"
    CONTROL_VARIATE = "control_variate"
    ADAPTIVE = "adaptive"

@dataclass
class GraphSAGEConfig:
    """Configuration for large-scale GraphSAGE"""
    # Model architecture
    input_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    output_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    
    # Aggregation
    aggregator_types: List[str] = field(default_factory=lambda: ['mean', 'max', 'lstm'])
    attention_heads: int = 8
    
    # Sampling
    neighbor_samples: List[int] = field(default_factory=lambda: [25, 10, 5])
    sampling_strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE
    batch_size: int = 1024
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 200
    early_stopping_patience: int = 20
    gradient_clipping: float = 1.0
    
    # Distributed processing
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    cache_embeddings: bool = True
    max_memory_usage: float = 0.8  # 80% of available memory
    
    # Geological domain
    scale_types: List[GeologicalScaleType] = field(default_factory=lambda: [GeologicalScaleType.FORMATION, GeologicalScaleType.BASIN])
    temporal_aggregation: bool = True
    spatial_weighting: bool = True
    physics_constraints: bool = True

@dataclass 
class GeologicalNodeFeatures:
    """Enhanced node features for large-scale analysis"""
    node_id: str
    coordinates: np.ndarray  # [x, y, z, time]
    geological_type: str
    properties: Dict[str, float]
    embeddings: Optional[np.ndarray] = None
    scale_type: GeologicalScaleType = GeologicalScaleType.FORMATION
    temporal_features: Optional[np.ndarray] = None
    spatial_features: Optional[np.ndarray] = None
    physics_features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to comprehensive feature vector"""
        features = []
        
        # Spatial-temporal coordinates
        features.extend(self.coordinates)
        
        # Geological properties
        features.extend(list(self.properties.values()))
        
        # Scale encoding
        scale_encoding = [0.0] * len(GeologicalScaleType)
        scale_encoding[list(GeologicalScaleType).index(self.scale_type)] = 1.0
        features.extend(scale_encoding)
        
        # Optional embeddings
        if self.embeddings is not None:
            features.extend(self.embeddings)
            
        # Temporal features
        if self.temporal_features is not None:
            features.extend(self.temporal_features)
            
        # Spatial features  
        if self.spatial_features is not None:
            features.extend(self.spatial_features)
            
        # Physics features
        if self.physics_features is not None:
            features.extend(self.physics_features)
            
        return np.array(features)

class AdaptiveNeighborSampler:
    """Adaptive neighbor sampling based on geological importance"""
    
    def __init__(self, config: GraphSAGEConfig):
        self.config = config
        self.importance_cache = {}
        self.sampling_history = {}
        
    def calculate_importance(self, node_features: GeologicalNodeFeatures, 
                           neighbors: List[GeologicalNodeFeatures]) -> np.ndarray:
        """Calculate geological importance of neighbors"""
        if not neighbors:
            return np.array([])
            
        importance_scores = []
        
        for neighbor in neighbors:
            # Spatial importance
            distance = np.linalg.norm(node_features.coordinates[:3] - neighbor.coordinates[:3])
            spatial_weight = np.exp(-distance / 1000)  # 1km characteristic distance
            
            # Temporal importance
            time_diff = abs(node_features.coordinates[3] - neighbor.coordinates[3])
            temporal_weight = np.exp(-time_diff / (365 * 24 * 3600))  # 1 year characteristic time
            
            # Scale compatibility
            scale_compatibility = 1.0 if node_features.scale_type == neighbor.scale_type else 0.5
            
            # Geological type similarity
            type_similarity = 1.0 if node_features.geological_type == neighbor.geological_type else 0.3
            
            # Physics-based importance
            physics_importance = 1.0
            if node_features.physics_features is not None and neighbor.physics_features is not None:
                physics_similarity = np.dot(node_features.physics_features, neighbor.physics_features)
                physics_importance = np.exp(physics_similarity)
                
            total_importance = (spatial_weight * temporal_weight * 
                              scale_compatibility * type_similarity * physics_importance)
            importance_scores.append(total_importance)
            
        return np.array(importance_scores)
    
    def sample_neighbors(self, node_id: str, neighbors: List[str], 
                        node_features_map: Dict[str, GeologicalNodeFeatures],
                        num_samples: int) -> List[str]:
        """Sample neighbors based on geological importance"""
        if len(neighbors) <= num_samples:
            return neighbors
            
        node_features = node_features_map[node_id]
        neighbor_features = [node_features_map[nid] for nid in neighbors if nid in node_features_map]
        
        if self.config.sampling_strategy == SamplingStrategy.UNIFORM:
            return np.random.choice(neighbors, size=num_samples, replace=False).tolist()
        elif self.config.sampling_strategy == SamplingStrategy.IMPORTANCE:
            importance_scores = self.calculate_importance(node_features, neighbor_features)
            probabilities = importance_scores / importance_scores.sum()
            sampled_indices = np.random.choice(len(neighbors), size=num_samples, 
                                             replace=False, p=probabilities)
            return [neighbors[i] for i in sampled_indices]
        elif self.config.sampling_strategy == SamplingStrategy.ADAPTIVE:
            # Combine importance and diversity
            importance_scores = self.calculate_importance(node_features, neighbor_features)
            
            # Select top important neighbors
            top_k = min(num_samples * 2, len(neighbors))
            top_indices = np.argsort(importance_scores)[-top_k:]
            
            # Apply diversity sampling within top candidates
            diverse_sample = self._diversity_sampling(
                [neighbors[i] for i in top_indices],
                [neighbor_features[i] for i in top_indices],
                num_samples
            )
            return diverse_sample
        else:
            return neighbors[:num_samples]
    
    def _diversity_sampling(self, candidates: List[str], 
                           candidate_features: List[GeologicalNodeFeatures],
                           num_samples: int) -> List[str]:
        """Sample for diversity within important candidates"""
        if len(candidates) <= num_samples:
            return candidates
            
        # Build similarity matrix
        n = len(candidates)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Calculate feature similarity
                feat1 = candidate_features[i].to_feature_vector()
                feat2 = candidate_features[j].to_feature_vector()
                
                # Ensure same length
                min_len = min(len(feat1), len(feat2))
                feat1_norm = feat1[:min_len] / (np.linalg.norm(feat1[:min_len]) + 1e-8)
                feat2_norm = feat2[:min_len] / (np.linalg.norm(feat2[:min_len]) + 1e-8)
                
                similarity = np.dot(feat1_norm, feat2_norm)
                similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
        
        # Greedy diversity selection
        selected_indices = []
        remaining_indices = list(range(n))
        
        # Start with most important (last in sorted order)
        selected_indices.append(remaining_indices.pop())
        
        while len(selected_indices) < num_samples and remaining_indices:
            # Find candidate with minimum similarity to selected
            min_max_sim = float('inf')
            best_candidate = None
            
            for candidate_idx in remaining_indices:
                max_sim = max(similarity_matrix[candidate_idx, sel_idx] 
                             for sel_idx in selected_indices)
                if max_sim < min_max_sim:
                    min_max_sim = max_sim
                    best_candidate = candidate_idx
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
        
        return [candidates[i] for i in selected_indices]

class GeologicalGraphSAGELayer(nn.Module):
    """Advanced GraphSAGE layer with geological constraints"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 aggregator_type: str = 'mean', dropout: float = 0.1,
                 use_attention: bool = True, use_physics_constraints: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator_type = aggregator_type
        self.use_attention = use_attention
        self.use_physics_constraints = use_physics_constraints
        
        # Neighbor aggregator
        if aggregator_type == 'mean':
            self.aggregator = self._mean_aggregator
        elif aggregator_type == 'max':
            self.aggregator = self._max_aggregator
        elif aggregator_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, output_dim // 2, batch_first=True)
            self.aggregator = self._lstm_aggregator
        else:
            self.aggregator = self._mean_aggregator
            
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(input_dim, num_heads=8, dropout=dropout)
            self.attention_norm = nn.LayerNorm(input_dim)
        
        # Feature transformations
        self.self_transform = nn.Linear(input_dim, output_dim)
        self.neighbor_transform = nn.Linear(input_dim, output_dim)
        self.combined_transform = nn.Linear(output_dim * 2, output_dim)
        
        # Physics constraints
        if use_physics_constraints:
            self.physics_constraint = nn.Sequential(
                nn.Linear(output_dim, output_dim // 2),
                nn.ReLU(),
                nn.Linear(output_dim // 2, output_dim)
            )
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def _mean_aggregator(self, neighbor_features: torch.Tensor) -> torch.Tensor:
        """Mean aggregation"""
        return torch.mean(neighbor_features, dim=1)
    
    def _max_aggregator(self, neighbor_features: torch.Tensor) -> torch.Tensor:
        """Max aggregation"""
        return torch.max(neighbor_features, dim=1)[0]
    
    def _lstm_aggregator(self, neighbor_features: torch.Tensor) -> torch.Tensor:
        """LSTM aggregation"""
        batch_size, num_neighbors, feature_dim = neighbor_features.shape
        
        # Random permutation for order invariance
        perm_indices = torch.randperm(num_neighbors)
        permuted_features = neighbor_features[:, perm_indices, :]
        
        lstm_out, (hidden, _) = self.lstm(permuted_features)
        return hidden[-1]  # Last hidden state
    
    def forward(self, self_features: torch.Tensor, 
                neighbor_features: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size = self_features.size(0)
        
        # Apply attention if enabled
        if self.use_attention and neighbor_features.size(1) > 0:
            # Reshape for attention
            seq_len = neighbor_features.size(1)
            neighbor_flat = neighbor_features.view(-1, neighbor_features.size(2))
            self_expanded = self_features.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, self_features.size(1))
            
            # Apply attention
            attended, _ = self.attention(
                neighbor_flat.unsqueeze(1),
                self_expanded.unsqueeze(1), 
                self_expanded.unsqueeze(1)
            )
            attended = attended.squeeze(1)
            attended = self.attention_norm(neighbor_flat + attended)
            
            # Reshape back
            neighbor_features = attended.view(batch_size, seq_len, -1)
        
        # Aggregate neighbors
        if neighbor_features.size(1) > 0:
            aggregated_neighbors = self.aggregator(neighbor_features)
        else:
            aggregated_neighbors = torch.zeros(batch_size, self.input_dim, device=self_features.device)
        
        # Transform features
        self_transformed = self.self_transform(self_features)
        neighbor_transformed = self.neighbor_transform(aggregated_neighbors)
        
        # Combine self and neighbor features
        combined = torch.cat([self_transformed, neighbor_transformed], dim=1)
        output = self.combined_transform(combined)
        
        # Apply physics constraints
        if self.use_physics_constraints:
            physics_correction = self.physics_constraint(output)
            output = output + physics_correction
        
        # Normalization and activation
        output = self.layer_norm(output)
        output = F.relu(output)
        output = self.dropout(output)
        
        return output

class LargeScaleGeologicalGraphSAGE(nn.Module):
    """Large-scale GraphSAGE for geological analysis"""
    
    def __init__(self, config: GraphSAGEConfig):
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
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            input_dim = config.hidden_dims[i] if i > 0 else config.hidden_dims[0]
            output_dim = config.hidden_dims[i+1] if i < self.num_layers - 1 else config.output_dim
            
            # Use different aggregator types for different layers
            aggregator_type = config.aggregator_types[i % len(config.aggregator_types)]
            
            layer = GeologicalGraphSAGELayer(
                input_dim=input_dim,
                output_dim=output_dim,
                aggregator_type=aggregator_type,
                dropout=config.dropout,
                use_attention=(i < 2),  # Only use attention in first 2 layers
                use_physics_constraints=config.physics_constraints
            )
            self.sage_layers.append(layer)
        
        # Multi-scale fusion
        self.scale_fusion = nn.ModuleDict({
            scale.value: nn.Sequential(
                nn.Linear(config.output_dim, config.output_dim),
                nn.ReLU(),
                nn.Linear(config.output_dim, config.output_dim)
            ) for scale in config.scale_types
        })
        
        # Temporal aggregation
        if config.temporal_aggregation:
            self.temporal_lstm = nn.LSTM(
                config.output_dim, config.output_dim, 
                batch_first=True, dropout=config.dropout
            )
            self.temporal_attention = nn.MultiheadAttention(
                config.output_dim, num_heads=config.attention_heads,
                dropout=config.dropout
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
            nn.Linear(config.output_dim // 2, config.output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, node_features: torch.Tensor, 
                neighbor_features_list: List[torch.Tensor],
                scale_types: Optional[torch.Tensor] = None,
                temporal_sequence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-scale and temporal processing"""
        
        # Input projection
        h = self.input_projection(node_features)
        
        # GraphSAGE layers
        for i, layer in enumerate(self.sage_layers):
            neighbor_features = neighbor_features_list[i] if i < len(neighbor_features_list) else torch.empty(h.size(0), 0, h.size(1))
            h = layer(h, neighbor_features)
        
        # Multi-scale fusion
        if scale_types is not None:
            scale_outputs = []
            for i, scale_type in enumerate(scale_types):
                scale_name = GeologicalScaleType(scale_type.item()).value
                if scale_name in self.scale_fusion:
                    scale_output = self.scale_fusion[scale_name](h[i:i+1])
                    scale_outputs.append(scale_output)
            
            if scale_outputs:
                h = torch.cat(scale_outputs, dim=0)
        
        # Temporal aggregation
        if self.config.temporal_aggregation and temporal_sequence is not None:
            batch_size, seq_len, feature_dim = temporal_sequence.shape
            
            # LSTM processing
            lstm_out, (final_hidden, _) = self.temporal_lstm(temporal_sequence)
            
            # Temporal attention
            attended, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
            
            # Combine with current features
            temporal_features = attended.mean(dim=1)  # Average over sequence
            h = h + temporal_features
        
        # Final output
        embeddings = self.output_layers(h)
        uncertainties = self.uncertainty_head(h)
        
        return {
            'embeddings': embeddings,
            'uncertainties': uncertainties,
            'intermediate_features': h
        }

class GeologicalGraphDataset(Dataset):
    """Dataset for large-scale geological graph processing"""
    
    def __init__(self, nodes: List[GeologicalNodeFeatures], 
                 edges: List[Tuple[str, str, Dict]], 
                 config: GraphSAGEConfig):
        self.nodes = nodes
        self.edges = edges
        self.config = config
        
        # Build mappings
        self.node_id_to_idx = {node.node_id: i for i, node in enumerate(nodes)}
        self.idx_to_node_id = {i: node.node_id for i, node in enumerate(nodes)}
        
        # Build adjacency lists
        self.adjacency_list = self._build_adjacency_list()
        
        # Prepare features
        self.node_features = self._prepare_node_features()
        
        # Initialize sampler
        self.sampler = AdaptiveNeighborSampler(config)
        
    def _build_adjacency_list(self) -> Dict[str, List[str]]:
        """Build adjacency list representation"""
        adj_list = {node.node_id: [] for node in self.nodes}
        
        for src, dst, edge_data in self.edges:
            if src in adj_list and dst in adj_list:
                adj_list[src].append(dst)
                adj_list[dst].append(src)  # Undirected graph
        
        return adj_list
    
    def _prepare_node_features(self) -> torch.Tensor:
        """Prepare node feature matrix"""
        feature_vectors = []
        max_length = 0
        
        # First pass: find maximum feature length
        for node in self.nodes:
            fv = node.to_feature_vector()
            max_length = max(max_length, len(fv))
        
        # Second pass: pad features to same length
        for node in self.nodes:
            fv = node.to_feature_vector()
            if len(fv) < max_length:
                padded_fv = np.pad(fv, (0, max_length - len(fv)), mode='constant')
            else:
                padded_fv = fv
            feature_vectors.append(padded_fv)
        
        return torch.FloatTensor(np.array(feature_vectors))
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, idx):
        node = self.nodes[idx]
        node_id = node.node_id
        
        # Get neighbors for each hop
        neighbor_features_list = []
        current_nodes = [node_id]
        
        for hop in range(len(self.config.neighbor_samples)):
            next_neighbors = []
            hop_neighbor_features = []
            
            for current_node in current_nodes:
                if current_node in self.adjacency_list:
                    neighbors = self.adjacency_list[current_node]
                    
                    # Sample neighbors
                    num_samples = self.config.neighbor_samples[hop]
                    sampled_neighbors = self.sampler.sample_neighbors(
                        current_node, neighbors, 
                        {n.node_id: n for n in self.nodes},
                        num_samples
                    )
                    
                    # Get neighbor features
                    for neighbor_id in sampled_neighbors:
                        if neighbor_id in self.node_id_to_idx:
                            neighbor_idx = self.node_id_to_idx[neighbor_id]
                            hop_neighbor_features.append(self.node_features[neighbor_idx])
                    
                    next_neighbors.extend(sampled_neighbors)
            
            # Pad or truncate to fixed size
            max_neighbors = self.config.neighbor_samples[hop] * len(current_nodes)
            
            if len(hop_neighbor_features) > max_neighbors:
                hop_neighbor_features = hop_neighbor_features[:max_neighbors]
            elif len(hop_neighbor_features) < max_neighbors:
                # Pad with zeros
                padding_size = max_neighbors - len(hop_neighbor_features)
                zero_features = torch.zeros(padding_size, self.node_features.size(1))
                hop_neighbor_features.extend(zero_features)
            
            if hop_neighbor_features:
                neighbor_features_tensor = torch.stack(hop_neighbor_features)
                neighbor_features_tensor = neighbor_features_tensor.view(
                    len(current_nodes), -1, self.node_features.size(1)
                )
            else:
                neighbor_features_tensor = torch.zeros(
                    len(current_nodes), 0, self.node_features.size(1)
                )
            
            neighbor_features_list.append(neighbor_features_tensor[0])  # Take first node's neighbors
            current_nodes = list(set(next_neighbors))
        
        # Additional data
        scale_type = torch.tensor([list(GeologicalScaleType).index(node.scale_type)])
        
        # Temporal sequence (if available)
        temporal_sequence = None
        if node.temporal_features is not None:
            temporal_sequence = torch.FloatTensor(node.temporal_features.reshape(1, -1))
        
        return {
            'node_features': self.node_features[idx],
            'neighbor_features_list': neighbor_features_list,
            'scale_type': scale_type,
            'temporal_sequence': temporal_sequence,
            'node_id': node_id
        }

class LargeScaleGraphSAGETrainer:
    """Advanced training system for large-scale GraphSAGE"""
    
    def __init__(self, model: LargeScaleGeologicalGraphSAGE, config: GraphSAGEConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        
        # Memory management
        self.memory_monitor = psutil.Process()
        
        # Distributed training setup
        self.is_distributed = False
        self.local_rank = 0
        self.world_size = 1
        
    def setup_distributed(self):
        """Setup distributed training"""
        if 'WORLD_SIZE' in os.environ:
            self.is_distributed = True
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            
            # Initialize process group
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
            
            # Wrap model with DDP
            self.model = DDP(self.model, device_ids=[self.local_rank])
    
    def setup_training(self):
        """Setup training components"""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.epochs,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Mixed precision scaler
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def create_data_loader(self, dataset: GeologicalGraphDataset, 
                          shuffle: bool = True) -> DataLoader:
        """Create data loader with appropriate settings"""
        
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # DistributedSampler handles shuffling
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            persistent_workers=True
        )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Memory management
            if batch_idx % 100 == 0:
                self._manage_memory()
            
            # Move to device
            device = next(self.model.parameters()).device
            node_features = batch['node_features'].to(device)
            neighbor_features_list = [nf.to(device) for nf in batch['neighbor_features_list']]
            scale_types = batch['scale_type'].to(device) if batch['scale_type'] is not None else None
            temporal_sequences = batch['temporal_sequence'].to(device) if batch['temporal_sequence'][0] is not None else None
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(node_features, neighbor_features_list, 
                                       scale_types, temporal_sequences)
                    loss = self._calculate_loss(outputs, batch)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clipping
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(node_features, neighbor_features_list, 
                                   scale_types, temporal_sequences)
                loss = self._calculate_loss(outputs, batch)
                
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clipping
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % 100 == 0 and self.local_rank == 0:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_embeddings = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in val_loader:
                device = next(self.model.parameters()).device
                node_features = batch['node_features'].to(device)
                neighbor_features_list = [nf.to(device) for nf in batch['neighbor_features_list']]
                scale_types = batch['scale_type'].to(device) if batch['scale_type'] is not None else None
                temporal_sequences = batch['temporal_sequence'].to(device) if batch['temporal_sequence'][0] is not None else None
                
                outputs = self.model(node_features, neighbor_features_list, 
                                   scale_types, temporal_sequences)
                loss = self._calculate_loss(outputs, batch)
                
                total_loss += loss.item()
                all_embeddings.append(outputs['embeddings'].cpu())
                all_uncertainties.append(outputs['uncertainties'].cpu())
        
        # Calculate additional metrics
        embeddings = torch.cat(all_embeddings, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0)
        
        metrics = {
            'embedding_mean_norm': torch.norm(embeddings, dim=1).mean().item(),
            'uncertainty_mean': uncertainties.mean().item(),
            'embedding_std': embeddings.std().item()
        }
        
        return total_loss / len(val_loader), metrics
    
    def _calculate_loss(self, outputs: Dict[str, torch.Tensor], 
                       batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate training loss"""
        embeddings = outputs['embeddings']
        uncertainties = outputs['uncertainties']
        
        # Contrastive loss for self-supervised learning
        batch_size = embeddings.size(0)
        
        # Create positive and negative pairs
        labels = torch.arange(batch_size).to(embeddings.device)
        
        # Compute similarity matrix
        embeddings_norm = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # InfoNCE loss
        temperature = 0.1
        similarity_matrix = similarity_matrix / temperature
        
        # Create mask for positive pairs (diagonal)
        mask = torch.eye(batch_size).bool().to(embeddings.device)
        
        # Compute InfoNCE loss
        positive_samples = similarity_matrix[mask].view(batch_size, -1)
        negative_samples = similarity_matrix[~mask].view(batch_size, -1)
        
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        labels = torch.zeros(batch_size).long().to(embeddings.device)
        
        contrastive_loss = F.cross_entropy(logits, labels)
        
        # Uncertainty regularization
        uncertainty_reg = uncertainties.mean()
        
        # Total loss
        total_loss = contrastive_loss + 0.01 * uncertainty_reg
        
        return total_loss
    
    def _manage_memory(self):
        """Manage GPU memory usage"""
        if torch.cuda.is_available():
            # Clear cache periodically
            torch.cuda.empty_cache()
            
            # Check memory usage
            memory_usage = self.memory_monitor.memory_percent()
            if memory_usage > self.config.max_memory_usage * 100:
                self.logger.warning(f"High memory usage: {memory_usage:.1f}%")
                gc.collect()
                torch.cuda.empty_cache()
    
    async def train(self, train_dataset: GeologicalGraphDataset,
                   val_dataset: Optional[GeologicalGraphDataset] = None) -> Dict[str, Any]:
        """Main training loop"""
        
        self.setup_distributed()
        self.setup_training()
        
        # Create data loaders
        train_loader = self.create_data_loader(train_dataset, shuffle=True)
        val_loader = None
        if val_dataset:
            val_loader = self.create_data_loader(val_dataset, shuffle=False)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            if self.is_distributed:
                train_loader.sampler.set_epoch(epoch)
            
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = 0.0
            metrics = {}
            if val_loader:
                val_loss, metrics = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                self.metrics_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **metrics
                })
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Early stopping
            if val_loader and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                if self.local_rank == 0:
                    self._save_checkpoint(epoch, best_val_loss)
            elif val_loader:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                if self.local_rank == 0:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            if self.local_rank == 0:
                self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'best_val_loss': best_val_loss
        }
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict() if not self.is_distributed else self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        checkpoint_path = Path("checkpoints") / f"graphsage_epoch_{epoch}.pt"
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

# Example usage and testing
async def test_large_scale_graphsage():
    """Test large-scale GraphSAGE implementation"""
    
    # Configuration
    config = GraphSAGEConfig(
        input_dim=128,
        hidden_dims=[256, 128, 64],
        output_dim=32,
        batch_size=64,
        epochs=50
    )
    
    # Create synthetic geological data
    nodes = []
    np.random.seed(42)
    
    for i in range(1000):  # Large-scale test
        node = GeologicalNodeFeatures(
            node_id=f"geo_node_{i}",
            coordinates=np.random.rand(4) * [10000, 10000, 5000, 365*24*3600],  # x,y,z,time
            geological_type=np.random.choice(['fault', 'fold', 'intrusion', 'sediment']),
            properties={
                'porosity': np.random.uniform(0.1, 0.4),
                'permeability': np.random.uniform(1e-15, 1e-12),
                'density': np.random.uniform(2000, 3000),
                'temperature': np.random.uniform(273, 573)
            },
            scale_type=np.random.choice(list(GeologicalScaleType)),
            temporal_features=np.random.rand(10),
            spatial_features=np.random.rand(15),
            physics_features=np.random.rand(8)
        )
        nodes.append(node)
    
    # Create synthetic edges
    edges = []
    for i in range(len(nodes)):
        for j in range(i+1, min(i+20, len(nodes))):  # Connect to nearby nodes
            distance = np.linalg.norm(nodes[i].coordinates[:3] - nodes[j].coordinates[:3])
            if distance < 2000:  # Within 2km
                weight = np.exp(-distance / 1000)
                edges.append((nodes[i].node_id, nodes[j].node_id, {'weight': weight, 'distance': distance}))
    
    print(f"Created dataset with {len(nodes)} nodes and {len(edges)} edges")
    
    # Create datasets
    dataset = GeologicalGraphDataset(nodes, edges, config)
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create model
    model = LargeScaleGeologicalGraphSAGE(config)
    
    # Train model
    trainer = LargeScaleGraphSAGETrainer(model, config)
    results = await trainer.train(train_dataset, val_dataset)
    
    print("Training completed!")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Final training loss: {results['train_losses'][-1]:.6f}")

if __name__ == "__main__":
    print(f"DGL available: {DGL_AVAILABLE}")
    print(f"Redis available: {REDIS_AVAILABLE}")
    
    # Run test
    asyncio.run(test_large_scale_graphsage())