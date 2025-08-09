"""
Geological Network Representation Learning System
Advanced embedding learning for geological entities, relationships, and system classification
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy
import pickle
import faiss
import math

# Advanced ML imports with fallback
try:
    import dgl
    import dgl.nn as dglnn
    from dgl import DGLGraph
    from dgl.data import DGLDataset
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    warnings.warn("DGL not available. Graph neural network features will be limited.")

try:
    from transformers import AutoModel, AutoTokenizer
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers or PyTorch Geometric not available. Advanced embeddings will be limited.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

warnings.filterwarnings('ignore')

class GeologicalEntityType(Enum):
    VOLCANO = "volcano"
    CALDERA = "caldera"
    FAULT_SYSTEM = "fault_system"
    MAGMA_CHAMBER = "magma_chamber"
    HYDROTHERMAL_SYSTEM = "hydrothermal_system"
    TECTONIC_PLATE = "tectonic_plate"
    SEDIMENTARY_BASIN = "sedimentary_basin"
    METAMORPHIC_CORE = "metamorphic_core"
    CRUSTAL_BLOCK = "crustal_block"
    MANTLE_PLUME = "mantle_plume"

class GeologicalProcessType(Enum):
    ERUPTION = "eruption"
    SUPERERUPTION = "supereruption"
    EARTHQUAKE = "earthquake"
    DEFORMATION = "deformation"
    INTRUSION = "intrusion"
    SUBDUCTION = "subduction"
    RIFTING = "rifting"
    METAMORPHISM = "metamorphism"
    SEDIMENTATION = "sedimentation"
    WEATHERING = "weathering"

class RelationshipType(Enum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    GENETIC = "genetic"
    STRUCTURAL = "structural"
    CHEMICAL = "chemical"
    THERMAL = "thermal"
    MECHANICAL = "mechanical"

@dataclass
class GeologicalEntity:
    """Represents a geological entity for embedding learning"""
    id: str
    entity_type: GeologicalEntityType
    coordinates: np.ndarray  # [lon, lat, depth]
    properties: Dict[str, float]
    temporal_properties: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_feature_vector(self, include_temporal: bool = True) -> np.ndarray:
        """Convert entity to feature vector"""
        features = []
        
        # Spatial features
        features.extend(self.coordinates)
        
        # Entity type one-hot encoding
        entity_encoding = [0.0] * len(GeologicalEntityType)
        entity_encoding[list(GeologicalEntityType).index(self.entity_type)] = 1.0
        features.extend(entity_encoding)
        
        # Property features
        features.extend(list(self.properties.values()))
        
        # Temporal features if requested
        if include_temporal and self.temporal_properties:
            for key, values in self.temporal_properties.items():
                if isinstance(values, np.ndarray):
                    # Statistical summaries of time series
                    features.extend([
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values),
                        np.median(values)
                    ])
        
        return np.array(features, dtype=np.float32)

@dataclass
class GeologicalProcess:
    """Represents a geological process for embedding learning"""
    id: str
    process_type: GeologicalProcessType
    entity_ids: List[str]  # Entities involved in the process
    parameters: Dict[str, float]
    time_range: Tuple[float, float]  # Start and end times
    spatial_extent: Dict[str, float]  # Spatial bounding information
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert process to feature vector"""
        features = []
        
        # Process type encoding
        process_encoding = [0.0] * len(GeologicalProcessType)
        process_encoding[list(GeologicalProcessType).index(self.process_type)] = 1.0
        features.extend(process_encoding)
        
        # Temporal features
        duration = self.time_range[1] - self.time_range[0]
        features.extend([self.time_range[0], self.time_range[1], duration])
        
        # Spatial features
        features.extend(list(self.spatial_extent.values()))
        
        # Process parameters
        features.extend(list(self.parameters.values()))
        
        # Number of entities involved
        features.append(float(len(self.entity_ids)))
        
        return np.array(features, dtype=np.float32)

@dataclass
class GeologicalRelationship:
    """Represents relationships between geological entities/processes"""
    id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float
    properties: Dict[str, float] = field(default_factory=dict)
    temporal_evolution: Optional[np.ndarray] = None
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert relationship to feature vector"""
        features = []
        
        # Relationship type encoding
        rel_encoding = [0.0] * len(RelationshipType)
        rel_encoding[list(RelationshipType).index(self.relationship_type)] = 1.0
        features.extend(rel_encoding)
        
        # Strength
        features.append(self.strength)
        
        # Properties
        features.extend(list(self.properties.values()))
        
        # Temporal evolution statistics if available
        if self.temporal_evolution is not None:
            features.extend([
                np.mean(self.temporal_evolution),
                np.std(self.temporal_evolution),
                np.min(self.temporal_evolution),
                np.max(self.temporal_evolution)
            ])
        
        return np.array(features, dtype=np.float32)

class NodeEmbeddingLearner(nn.Module):
    """Advanced node embedding learning for geological entities and processes"""
    
    def __init__(self, input_dim: int, embedding_dim: int = 256, 
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # Geological domain-specific transformations
        self.spatial_transform = nn.Sequential(
            nn.Linear(3, embedding_dim // 4),  # lon, lat, depth
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        self.temporal_transform = nn.Sequential(
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        self.property_transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        
        # Final embedding layer
        self.final_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()  # Normalized embeddings
        )
        
        # Contrastive learning components
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4)
        )
    
    def forward(self, features: torch.Tensor, 
                spatial_coords: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to generate node embeddings"""
        batch_size = features.size(0)
        
        # Basic feature encoding
        encoded_features = self.feature_encoder(features)
        
        # Apply attention mechanism
        encoded_features = encoded_features.unsqueeze(1)  # Add sequence dimension
        attended_features, attention_weights = self.attention(
            encoded_features, encoded_features, encoded_features
        )
        attended_features = attended_features.squeeze(1)
        
        # Domain-specific transformations
        if spatial_coords is not None and spatial_coords.size(1) >= 3:
            spatial_emb = self.spatial_transform(spatial_coords[:, :3])
            attended_features = torch.cat([attended_features[:, :-spatial_emb.size(1)], spatial_emb], dim=1)
        
        # Final embedding
        embeddings = self.final_embedding(attended_features)
        
        # Projection for contrastive learning
        projections = self.projection_head(embeddings)
        
        return embeddings, projections

class EdgeEmbeddingLearner(nn.Module):
    """Edge embedding learning for geological relationships"""
    
    def __init__(self, node_embedding_dim: int, edge_feature_dim: int,
                 embedding_dim: int = 128, num_layers: int = 3):
        super().__init__()
        
        self.node_embedding_dim = node_embedding_dim
        self.edge_feature_dim = edge_feature_dim
        self.embedding_dim = embedding_dim
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2)
        )
        
        # Node interaction encoder
        self.node_interaction = nn.Sequential(
            nn.Linear(node_embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2)
        )
        
        # Relationship type attention
        self.relationship_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # Final edge embedding
        self.final_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()
        )
    
    def forward(self, edge_features: torch.Tensor,
                source_embeddings: torch.Tensor,
                target_embeddings: torch.Tensor) -> torch.Tensor:
        """Generate edge embeddings from node embeddings and edge features"""
        
        # Encode edge features
        edge_encoded = self.edge_encoder(edge_features)
        
        # Encode node interactions
        node_interactions = torch.cat([source_embeddings, target_embeddings], dim=1)
        interaction_encoded = self.node_interaction(node_interactions)
        
        # Combine edge and interaction features
        combined_features = torch.cat([edge_encoded, interaction_encoded], dim=1)
        
        # Apply attention
        combined_features = combined_features.unsqueeze(1)
        attended_features, _ = self.relationship_attention(
            combined_features, combined_features, combined_features
        )
        attended_features = attended_features.squeeze(1)
        
        # Final embedding
        edge_embeddings = self.final_embedding(attended_features)
        
        return edge_embeddings

class GraphLevelEmbeddingLearner(nn.Module):
    """Graph-level embedding learning for geological system classification"""
    
    def __init__(self, node_embedding_dim: int, edge_embedding_dim: int,
                 graph_embedding_dim: int = 512, num_graph_types: int = 20):
        super().__init__()
        
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.graph_embedding_dim = graph_embedding_dim
        
        # Node aggregation mechanisms
        self.node_attention_pooling = nn.MultiheadAttention(
            embed_dim=node_embedding_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Edge aggregation mechanisms
        self.edge_attention_pooling = nn.MultiheadAttention(
            embed_dim=edge_embedding_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # Graph structure encoding
        self.structure_encoder = nn.Sequential(
            nn.Linear(10, graph_embedding_dim // 4),  # Graph statistics features
            nn.ReLU(),
            nn.Linear(graph_embedding_dim // 4, graph_embedding_dim // 4)
        )
        
        # Hierarchical graph representation
        self.hierarchical_encoder = nn.LSTM(
            input_size=node_embedding_dim + edge_embedding_dim,
            hidden_size=graph_embedding_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Final graph embedding
        self.graph_embedding = nn.Sequential(
            nn.Linear(node_embedding_dim + edge_embedding_dim + graph_embedding_dim // 4, graph_embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(graph_embedding_dim),
            nn.Dropout(0.1),
            nn.Linear(graph_embedding_dim, graph_embedding_dim),
            nn.Tanh()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(graph_embedding_dim, graph_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(graph_embedding_dim // 2, num_graph_types)
        )
    
    def forward(self, node_embeddings: torch.Tensor,
                edge_embeddings: torch.Tensor,
                graph_structure_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate graph-level embeddings and classifications"""
        
        batch_size = node_embeddings.size(0)
        
        # Aggregate node embeddings
        node_embeddings_seq = node_embeddings.unsqueeze(0)  # Add batch dimension for attention
        node_attended, _ = self.node_attention_pooling(
            node_embeddings_seq, node_embeddings_seq, node_embeddings_seq
        )
        node_aggregated = torch.mean(node_attended.squeeze(0), dim=0, keepdim=True)
        
        # Aggregate edge embeddings
        if edge_embeddings.size(0) > 0:
            edge_embeddings_seq = edge_embeddings.unsqueeze(0)
            edge_attended, _ = self.edge_attention_pooling(
                edge_embeddings_seq, edge_embeddings_seq, edge_embeddings_seq
            )
            edge_aggregated = torch.mean(edge_attended.squeeze(0), dim=0, keepdim=True)
        else:
            edge_aggregated = torch.zeros(1, self.edge_embedding_dim)
        
        # Encode graph structure
        structure_encoded = self.structure_encoder(graph_structure_features)
        
        # Combine all features
        combined_features = torch.cat([
            node_aggregated,
            edge_aggregated,
            structure_encoded
        ], dim=1)
        
        # Generate graph embedding
        graph_embedding = self.graph_embedding(combined_features)
        
        # Classification
        classification_logits = self.classifier(graph_embedding)
        
        return graph_embedding, classification_logits

class GeologicalSimilaritySearch:
    """Advanced similarity search for analogous geological systems"""
    
    def __init__(self, embedding_dim: int = 256, distance_metric: str = 'cosine'):
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        
        # Initialize FAISS index for efficient similarity search
        if distance_metric == 'cosine':
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        
        # Metadata storage
        self.entity_metadata = {}
        self.embedding_to_id = {}
        self.id_to_embedding = {}
        
        # Clustering for system categorization
        self.clusterer = None
        self.cluster_labels = None
        self.cluster_representatives = {}
        
    def add_embeddings(self, embeddings: np.ndarray, entity_ids: List[str],
                      metadata: List[Dict[str, Any]]):
        """Add embeddings to the search index"""
        
        # Normalize embeddings for cosine similarity
        if self.distance_metric == 'cosine':
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.index.add(normalized_embeddings.astype(np.float32))
        else:
            self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        for i, (entity_id, meta) in enumerate(zip(entity_ids, metadata)):
            idx = self.index.ntotal - len(entity_ids) + i
            self.entity_metadata[idx] = meta
            self.embedding_to_id[idx] = entity_id
            self.id_to_embedding[entity_id] = idx
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10,
                      filter_criteria: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar geological systems"""
        
        # Normalize query if using cosine similarity
        if self.distance_metric == 'cosine':
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32), k * 2  # Get more results for filtering
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.entity_metadata:
                entity_id = self.embedding_to_id[idx]
                metadata = self.entity_metadata[idx]
                
                # Apply filters if specified
                if filter_criteria:
                    if not self._matches_criteria(metadata, filter_criteria):
                        continue
                
                similarity_score = float(dist) if self.distance_metric != 'cosine' else 1.0 - float(dist)
                results.append((entity_id, similarity_score, metadata))
                
                if len(results) >= k:
                    break
        
        return results
    
    def _matches_criteria(self, metadata: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in criteria.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif isinstance(value, dict):
                if 'min' in value and metadata[key] < value['min']:
                    return False
                if 'max' in value and metadata[key] > value['max']:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def cluster_systems(self, n_clusters: int = 10) -> Dict[int, List[str]]:
        """Cluster geological systems based on embeddings"""
        
        if self.index.ntotal == 0:
            return {}
        
        # Extract all embeddings
        embeddings = np.zeros((self.index.ntotal, self.embedding_dim), dtype=np.float32)
        for i in range(self.index.ntotal):
            embeddings[i] = self.index.reconstruct(i)
        
        # Perform clustering
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = self.clusterer.fit_predict(embeddings)
        
        # Organize results by cluster
        clusters = {}
        for i, label in enumerate(self.cluster_labels):
            if label not in clusters:
                clusters[label] = []
            
            if i in self.embedding_to_id:
                clusters[label].append(self.embedding_to_id[i])
        
        # Find cluster representatives (closest to centroid)
        for cluster_id in clusters.keys():
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = self.clusterer.cluster_centers_[cluster_id]
            
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            representative_idx = cluster_indices[np.argmin(distances)]
            
            if representative_idx in self.embedding_to_id:
                self.cluster_representatives[cluster_id] = self.embedding_to_id[representative_idx]
        
        return clusters
    
    def find_analogous_systems(self, query_system_id: str, 
                             analogy_criteria: Dict[str, Any],
                             k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find analogous geological systems based on specific criteria"""
        
        if query_system_id not in self.id_to_embedding:
            return []
        
        query_idx = self.id_to_embedding[query_system_id]
        query_embedding = self.index.reconstruct(query_idx)
        
        # Enhanced similarity search with geological domain knowledge
        candidates = self.search_similar(query_embedding, k * 3, analogy_criteria)
        
        # Rank candidates based on geological similarity metrics
        ranked_candidates = []
        for candidate_id, similarity, metadata in candidates:
            analogy_score = self._calculate_analogy_score(
                query_system_id, candidate_id, analogy_criteria
            )
            
            combined_score = 0.7 * similarity + 0.3 * analogy_score
            ranked_candidates.append((candidate_id, combined_score, metadata))
        
        # Sort by combined score and return top k
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        return ranked_candidates[:k]
    
    def _calculate_analogy_score(self, query_id: str, candidate_id: str,
                               criteria: Dict[str, Any]) -> float:
        """Calculate geological analogy score between systems"""
        
        query_metadata = self.entity_metadata.get(self.id_to_embedding.get(query_id, -1), {})
        candidate_metadata = self.entity_metadata.get(self.id_to_embedding.get(candidate_id, -1), {})
        
        if not query_metadata or not candidate_metadata:
            return 0.0
        
        score = 0.0
        total_weight = 0.0
        
        # Geological similarity metrics
        similarity_metrics = {
            'entity_type': 0.3,
            'tectonic_setting': 0.2,
            'age_range': 0.15,
            'size_scale': 0.15,
            'composition': 0.1,
            'activity_level': 0.1
        }
        
        for metric, weight in similarity_metrics.items():
            if metric in query_metadata and metric in candidate_metadata:
                metric_score = self._calculate_metric_similarity(
                    query_metadata[metric], candidate_metadata[metric], metric
                )
                score += weight * metric_score
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_metric_similarity(self, value1: Any, value2: Any, metric_type: str) -> float:
        """Calculate similarity for specific geological metrics"""
        
        if metric_type == 'entity_type':
            return 1.0 if value1 == value2 else 0.0
        
        elif metric_type == 'tectonic_setting':
            # Tectonic setting compatibility matrix
            compatibility = {
                ('subduction', 'subduction'): 1.0,
                ('rift', 'rift'): 1.0,
                ('hotspot', 'hotspot'): 1.0,
                ('transform', 'transform'): 1.0,
                ('subduction', 'rift'): 0.3,
                ('rift', 'hotspot'): 0.5,
                ('subduction', 'hotspot'): 0.4
            }
            
            key = (value1, value2) if (value1, value2) in compatibility else (value2, value1)
            return compatibility.get(key, 0.1)
        
        elif metric_type in ['age_range', 'size_scale']:
            # Numerical similarity with logarithmic scaling for geological scales
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                log_diff = abs(np.log10(max(value1, 1)) - np.log10(max(value2, 1)))
                return np.exp(-log_diff / 2)
            return 0.0
        
        elif metric_type == 'composition':
            # Chemical composition similarity (simplified)
            if isinstance(value1, dict) and isinstance(value2, dict):
                common_elements = set(value1.keys()) & set(value2.keys())
                if not common_elements:
                    return 0.0
                
                similarity_sum = 0.0
                for element in common_elements:
                    diff = abs(value1[element] - value2[element])
                    similarity_sum += np.exp(-diff / 10)  # Exponential decay
                
                return similarity_sum / len(common_elements)
            return 0.0
        
        elif metric_type == 'activity_level':
            # Activity level similarity
            activity_levels = ['dormant', 'low', 'moderate', 'high', 'extreme']
            if value1 in activity_levels and value2 in activity_levels:
                idx1, idx2 = activity_levels.index(value1), activity_levels.index(value2)
                return 1.0 - abs(idx1 - idx2) / (len(activity_levels) - 1)
            return 0.0
        
        return 0.0

class GeologicalNetworkRepresentationLearning:
    """Main class for geological network representation learning"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.node_learner = None
        self.edge_learner = None
        self.graph_learner = None
        self.similarity_search = None
        
        # Data storage
        self.entities = {}
        self.processes = {}
        self.relationships = {}
        self.graphs = {}
        
        # Learned embeddings
        self.node_embeddings = {}
        self.edge_embeddings = {}
        self.graph_embeddings = {}
        
        # Training history
        self.training_history = []
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'node_embedding_dim': 256,
            'edge_embedding_dim': 128,
            'graph_embedding_dim': 512,
            'num_layers': 4,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'contrastive_temperature': 0.07,
            'similarity_metric': 'cosine',
            'clustering_algorithm': 'kmeans',
            'supereruption_threshold': 8.0,  # VEI >= 8
            'caldera_size_threshold': 10.0,  # km diameter
            'magma_chamber_depth_range': [2000, 50000]  # meters
        }
    
    def initialize_models(self):
        """Initialize embedding learning models"""
        
        # Estimate input dimensions from data if available
        node_input_dim = 50  # Default estimate
        edge_input_dim = 20
        
        if self.entities:
            sample_entity = next(iter(self.entities.values()))
            node_input_dim = len(sample_entity.to_feature_vector())
        
        if self.relationships:
            sample_relationship = next(iter(self.relationships.values()))
            edge_input_dim = len(sample_relationship.to_feature_vector())
        
        # Initialize models
        self.node_learner = NodeEmbeddingLearner(
            input_dim=node_input_dim,
            embedding_dim=self.config['node_embedding_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        self.edge_learner = EdgeEmbeddingLearner(
            node_embedding_dim=self.config['node_embedding_dim'],
            edge_feature_dim=edge_input_dim,
            embedding_dim=self.config['edge_embedding_dim']
        )
        
        self.graph_learner = GraphLevelEmbeddingLearner(
            node_embedding_dim=self.config['node_embedding_dim'],
            edge_embedding_dim=self.config['edge_embedding_dim'],
            graph_embedding_dim=self.config['graph_embedding_dim']
        )
        
        self.similarity_search = GeologicalSimilaritySearch(
            embedding_dim=self.config['node_embedding_dim'],
            distance_metric=self.config['similarity_metric']
        )
    
    def add_geological_entity(self, entity: GeologicalEntity):
        """Add a geological entity to the system"""
        self.entities[entity.id] = entity
        
        # Special handling for supereruption-related entities
        if entity.entity_type in [GeologicalEntityType.VOLCANO, GeologicalEntityType.CALDERA]:
            self._process_volcanic_entity(entity)
    
    def add_geological_process(self, process: GeologicalProcess):
        """Add a geological process to the system"""
        self.processes[process.id] = process
        
        # Special handling for supereruptions
        if process.process_type == GeologicalProcessType.SUPERERUPTION:
            self._process_supereruption(process)
    
    def add_geological_relationship(self, relationship: GeologicalRelationship):
        """Add a geological relationship to the system"""
        self.relationships[relationship.id] = relationship
    
    def _process_volcanic_entity(self, entity: GeologicalEntity):
        """Process volcanic entities with special focus on supereruption potential"""
        
        # Check for supereruption indicators
        supereruption_indicators = {
            'caldera_diameter': entity.properties.get('caldera_diameter', 0),
            'magma_chamber_volume': entity.properties.get('magma_chamber_volume', 0),
            'historical_vei_max': entity.properties.get('historical_vei_max', 0),
            'crustal_thickness': entity.properties.get('crustal_thickness', 0),
            'silica_content': entity.properties.get('silica_content', 0)
        }
        
        # Calculate supereruption potential score
        potential_score = self._calculate_supereruption_potential(supereruption_indicators)
        entity.properties['supereruption_potential'] = potential_score
        
        # Enhanced metadata for large caldera systems
        if entity.properties.get('caldera_diameter', 0) > self.config['caldera_size_threshold']:
            entity.metadata['large_caldera_system'] = True
            entity.metadata['monitoring_priority'] = 'critical'
            
            # Add magma chamber analysis requirements
            entity.metadata['magma_chamber_analysis_required'] = True
    
    def _process_supereruption(self, process: GeologicalProcess):
        """Process supereruption events with enhanced analysis"""
        
        # Enhanced supereruption characterization
        vei = process.parameters.get('vei', 0)
        if vei >= self.config['supereruption_threshold']:
            process.parameters['supereruption_classification'] = True
            process.parameters['global_impact_assessment'] = True
            
            # Add precursor analysis
            precursor_duration = process.parameters.get('precursor_duration', 0)
            if precursor_duration > 0:
                process.parameters['precursor_complexity_score'] = self._calculate_precursor_complexity(process)
    
    def _calculate_supereruption_potential(self, indicators: Dict[str, float]) -> float:
        """Calculate supereruption potential score"""
        
        score = 0.0
        
        # Caldera diameter contribution
        if indicators['caldera_diameter'] > 15:  # km
            score += 0.3
        elif indicators['caldera_diameter'] > 10:
            score += 0.15
        
        # Magma chamber volume contribution
        if indicators['magma_chamber_volume'] > 1000:  # km³
            score += 0.25
        elif indicators['magma_chamber_volume'] > 500:
            score += 0.1
        
        # Historical VEI contribution
        if indicators['historical_vei_max'] >= 6:
            score += 0.2
        elif indicators['historical_vei_max'] >= 5:
            score += 0.1
        
        # Crustal thickness contribution
        if 20 <= indicators['crustal_thickness'] <= 40:  # Optimal thickness for large chambers
            score += 0.15
        
        # Silica content contribution (high silica = explosive potential)
        if indicators['silica_content'] > 70:  # wt% SiO2
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_precursor_complexity(self, process: GeologicalProcess) -> float:
        """Calculate precursor complexity score for supereruptions"""
        
        complexity_factors = [
            'seismic_rate_increase',
            'ground_deformation_rate',
            'gas_emission_changes',
            'thermal_anomalies',
            'gravity_changes',
            'magnetic_field_changes'
        ]
        
        complexity_score = 0.0
        available_factors = 0
        
        for factor in complexity_factors:
            if factor in process.parameters:
                complexity_score += min(process.parameters[factor], 1.0)
                available_factors += 1
        
        return complexity_score / max(available_factors, 1)
    
    async def train_embeddings(self, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train all embedding models"""
        
        if not self.node_learner:
            self.initialize_models()
        
        # Prepare training data
        node_features, node_ids = self._prepare_node_training_data()
        edge_features, edge_pairs = self._prepare_edge_training_data()
        graph_features, graph_labels = self._prepare_graph_training_data()
        
        if len(node_features) == 0:
            self.logger.warning("No training data available")
            return {'status': 'no_data'}
        
        # Split data for validation
        train_indices, val_indices = train_test_split(
            range(len(node_features)), test_size=validation_split, random_state=42
        )
        
        # Training setup
        optimizer = torch.optim.Adam(
            list(self.node_learner.parameters()) + 
            list(self.edge_learner.parameters()) + 
            list(self.graph_learner.parameters()),
            lr=self.config['learning_rate']
        )
        
        # Training loop
        training_results = {
            'epoch_losses': [],
            'node_losses': [],
            'edge_losses': [],
            'graph_losses': [],
            'validation_scores': []
        }
        
        for epoch in range(self.config['epochs']):
            epoch_loss = await self._train_epoch(
                node_features, node_ids, edge_features, edge_pairs,
                graph_features, graph_labels, train_indices, optimizer
            )
            
            training_results['epoch_losses'].append(epoch_loss)
            
            # Validation
            if epoch % 10 == 0:
                val_score = await self._validate_embeddings(
                    node_features, node_ids, edge_features, edge_pairs,
                    graph_features, graph_labels, val_indices
                )
                training_results['validation_scores'].append(val_score)
                
                self.logger.info(f"Epoch {epoch}: loss={epoch_loss:.6f}, val_score={val_score:.6f}")
        
        # Generate final embeddings
        await self._generate_final_embeddings(node_features, node_ids, edge_features, edge_pairs)
        
        self.training_history.append(training_results)
        return training_results
    
    def _prepare_node_training_data(self) -> Tuple[torch.Tensor, List[str]]:
        """Prepare node training data"""
        features = []
        ids = []
        
        for entity in self.entities.values():
            features.append(entity.to_feature_vector())
            ids.append(entity.id)
        
        for process in self.processes.values():
            features.append(process.to_feature_vector())
            ids.append(process.id)
        
        if features:
            # Pad features to same length
            max_length = max(len(f) for f in features)
            padded_features = []
            for f in features:
                if len(f) < max_length:
                    padded = np.pad(f, (0, max_length - len(f)), mode='constant')
                    padded_features.append(padded)
                else:
                    padded_features.append(f)
            
            return torch.FloatTensor(padded_features), ids
        
        return torch.FloatTensor([]), []
    
    def _prepare_edge_training_data(self) -> Tuple[torch.Tensor, List[Tuple[str, str]]]:
        """Prepare edge training data"""
        features = []
        pairs = []
        
        for rel in self.relationships.values():
            features.append(rel.to_feature_vector())
            pairs.append((rel.source_id, rel.target_id))
        
        if features:
            # Pad features to same length
            max_length = max(len(f) for f in features)
            padded_features = []
            for f in features:
                if len(f) < max_length:
                    padded = np.pad(f, (0, max_length - len(f)), mode='constant')
                    padded_features.append(padded)
                else:
                    padded_features.append(f)
            
            return torch.FloatTensor(padded_features), pairs
        
        return torch.FloatTensor([]), []
    
    def _prepare_graph_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare graph-level training data"""
        # This would be implemented based on specific graph-level tasks
        # For now, return empty tensors
        return torch.FloatTensor([]), torch.LongTensor([])
    
    async def _train_epoch(self, node_features, node_ids, edge_features, edge_pairs,
                          graph_features, graph_labels, train_indices, optimizer) -> float:
        """Train one epoch"""
        
        self.node_learner.train()
        self.edge_learner.train()
        self.graph_learner.train()
        
        total_loss = 0.0
        
        # Node embedding training with contrastive learning
        if len(node_features) > 0:
            train_node_features = node_features[train_indices]
            
            node_embeddings, node_projections = self.node_learner(train_node_features)
            
            # Contrastive loss for node embeddings
            node_loss = self._contrastive_loss(node_projections)
            total_loss += node_loss.item()
            
            # Edge embedding training
            if len(edge_features) > 0 and len(edge_pairs) > 0:
                # Sample source and target embeddings for edges
                edge_source_embs = []
                edge_target_embs = []
                valid_edge_features = []
                
                for i, (source_id, target_id) in enumerate(edge_pairs):
                    if source_id in node_ids and target_id in node_ids:
                        source_idx = node_ids.index(source_id)
                        target_idx = node_ids.index(target_id)
                        
                        if source_idx in train_indices and target_idx in train_indices:
                            source_emb_idx = list(train_indices).index(source_idx)
                            target_emb_idx = list(train_indices).index(target_idx)
                            
                            edge_source_embs.append(node_embeddings[source_emb_idx])
                            edge_target_embs.append(node_embeddings[target_emb_idx])
                            valid_edge_features.append(edge_features[i])
                
                if edge_source_embs:
                    edge_source_tensor = torch.stack(edge_source_embs)
                    edge_target_tensor = torch.stack(edge_target_embs)
                    edge_feature_tensor = torch.stack([torch.FloatTensor(f) for f in valid_edge_features])
                    
                    edge_embeddings = self.edge_learner(
                        edge_feature_tensor, edge_source_tensor, edge_target_tensor
                    )
                    
                    # Edge reconstruction loss
                    edge_loss = self._edge_reconstruction_loss(
                        edge_embeddings, edge_source_tensor, edge_target_tensor
                    )
                    total_loss += edge_loss.item()
        
        # Optimization step
        optimizer.zero_grad()
        loss_tensor = torch.tensor(total_loss, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()
        
        return total_loss
    
    def _contrastive_loss(self, projections: torch.Tensor, temperature: float = None) -> torch.Tensor:
        """Calculate contrastive loss for node embeddings"""
        if temperature is None:
            temperature = self.config['contrastive_temperature']
        
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / temperature
        
        # Create labels (diagonal should be positive pairs)
        batch_size = projections.size(0)
        labels = torch.arange(batch_size).long()
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def _edge_reconstruction_loss(self, edge_embeddings: torch.Tensor,
                                source_embeddings: torch.Tensor,
                                target_embeddings: torch.Tensor) -> torch.Tensor:
        """Calculate edge reconstruction loss"""
        
        # Reconstruct edge features from node embeddings
        reconstructed_edges = torch.cat([source_embeddings, target_embeddings], dim=1)
        
        # Simple reconstruction loss
        loss = F.mse_loss(edge_embeddings, reconstructed_edges[:, :edge_embeddings.size(1)])
        
        return loss
    
    async def _validate_embeddings(self, node_features, node_ids, edge_features, edge_pairs,
                                 graph_features, graph_labels, val_indices) -> float:
        """Validate embeddings"""
        
        self.node_learner.eval()
        self.edge_learner.eval()
        self.graph_learner.eval()
        
        with torch.no_grad():
            if len(val_indices) > 0 and len(node_features) > 0:
                val_node_features = node_features[val_indices]
                node_embeddings, node_projections = self.node_learner(val_node_features)
                
                # Calculate embedding quality metrics
                embedding_variance = torch.var(node_embeddings).item()
                embedding_mean_norm = torch.norm(node_embeddings, dim=1).mean().item()
                
                return embedding_variance * embedding_mean_norm
        
        return 0.0
    
    async def _generate_final_embeddings(self, node_features, node_ids, edge_features, edge_pairs):
        """Generate final embeddings for all entities"""
        
        self.node_learner.eval()
        self.edge_learner.eval()
        self.graph_learner.eval()
        
        with torch.no_grad():
            # Generate node embeddings
            if len(node_features) > 0:
                node_embeddings, _ = self.node_learner(node_features)
                
                for i, node_id in enumerate(node_ids):
                    self.node_embeddings[node_id] = node_embeddings[i].numpy()
                
                # Add to similarity search
                embeddings_array = node_embeddings.numpy()
                metadata_list = []
                
                for node_id in node_ids:
                    if node_id in self.entities:
                        entity = self.entities[node_id]
                        metadata = {
                            'entity_type': entity.entity_type.value,
                            'coordinates': entity.coordinates.tolist(),
                            'properties': entity.properties.copy()
                        }
                    else:
                        process = self.processes[node_id]
                        metadata = {
                            'process_type': process.process_type.value,
                            'parameters': process.parameters.copy(),
                            'time_range': process.time_range
                        }
                    
                    metadata_list.append(metadata)
                
                self.similarity_search.add_embeddings(embeddings_array, node_ids, metadata_list)
    
    def find_similar_systems(self, query_id: str, k: int = 10,
                           filter_criteria: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find similar geological systems"""
        
        if query_id not in self.node_embeddings:
            return []
        
        query_embedding = self.node_embeddings[query_id]
        return self.similarity_search.search_similar(query_embedding, k, filter_criteria)
    
    def find_supereruption_analogs(self, caldera_id: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find analog systems for supereruption analysis"""
        
        analogy_criteria = {
            'entity_type': [GeologicalEntityType.CALDERA.value, GeologicalEntityType.VOLCANO.value],
            'caldera_diameter': {'min': 10.0},  # Large calderas only
            'supereruption_potential': {'min': 0.3}
        }
        
        return self.similarity_search.find_analogous_systems(caldera_id, analogy_criteria, k)
    
    def analyze_magma_reservoir_systems(self, system_id: str) -> Dict[str, Any]:
        """Analyze magma reservoir characteristics for large caldera systems"""
        
        if system_id not in self.entities:
            return {}
        
        entity = self.entities[system_id]
        
        analysis = {
            'system_id': system_id,
            'entity_type': entity.entity_type.value,
            'magma_chamber_analysis': {},
            'crustal_analysis': {},
            'thermal_analysis': {},
            'similarity_analysis': {}
        }
        
        # Magma chamber characteristics
        chamber_volume = entity.properties.get('magma_chamber_volume', 0)
        chamber_depth = entity.properties.get('magma_chamber_depth', 0)
        
        analysis['magma_chamber_analysis'] = {
            'volume_km3': chamber_volume,
            'depth_km': chamber_depth / 1000 if chamber_depth > 0 else None,
            'volume_class': self._classify_chamber_volume(chamber_volume),
            'depth_class': self._classify_chamber_depth(chamber_depth),
            'eruption_potential': self._assess_eruption_potential(entity)
        }
        
        # Crustal analysis
        crustal_thickness = entity.properties.get('crustal_thickness', 0)
        analysis['crustal_analysis'] = {
            'thickness_km': crustal_thickness,
            'crustal_type': self._classify_crustal_type(crustal_thickness),
            'thermal_gradient': entity.properties.get('thermal_gradient', 0),
            'melt_fraction': entity.properties.get('melt_fraction', 0)
        }
        
        # Find similar systems
        similar_systems = self.find_similar_systems(system_id, k=5)
        analysis['similarity_analysis'] = {
            'analog_systems': similar_systems,
            'supereruption_analogs': self.find_supereruption_analogs(system_id)
        }
        
        return analysis
    
    def _classify_chamber_volume(self, volume: float) -> str:
        """Classify magma chamber volume"""
        if volume >= 1000:
            return 'supereruption_scale'
        elif volume >= 500:
            return 'large_scale'
        elif volume >= 100:
            return 'moderate_scale'
        else:
            return 'small_scale'
    
    def _classify_chamber_depth(self, depth: float) -> str:
        """Classify magma chamber depth"""
        depth_km = depth / 1000 if depth > 0 else 0
        
        if depth_km <= 5:
            return 'shallow_crustal'
        elif depth_km <= 15:
            return 'mid_crustal'
        elif depth_km <= 30:
            return 'deep_crustal'
        else:
            return 'subcrustal'
    
    def _classify_crustal_type(self, thickness: float) -> str:
        """Classify crustal type based on thickness"""
        if thickness <= 15:
            return 'oceanic'
        elif thickness <= 25:
            return 'transitional'
        elif thickness <= 45:
            return 'continental'
        else:
            return 'thick_continental'
    
    def _assess_eruption_potential(self, entity: GeologicalEntity) -> Dict[str, float]:
        """Assess eruption potential for magma chamber system"""
        
        volume = entity.properties.get('magma_chamber_volume', 0)
        depth = entity.properties.get('magma_chamber_depth', 0) / 1000  # Convert to km
        silica = entity.properties.get('silica_content', 0)
        temperature = entity.properties.get('temperature', 0)
        
        # Eruption potential factors
        volume_factor = min(volume / 1000, 1.0)  # Normalize by supereruption scale
        depth_factor = max(0, 1.0 - depth / 50)  # Shallower = higher potential
        composition_factor = max(0, (silica - 50) / 30)  # Higher silica = more explosive
        thermal_factor = max(0, (temperature - 700) / 500)  # Higher temp = more active
        
        explosive_potential = (volume_factor * 0.4 + composition_factor * 0.3 + 
                              thermal_factor * 0.2 + depth_factor * 0.1)
        
        effusive_potential = max(0, 1.0 - explosive_potential)
        
        return {
            'explosive_potential': explosive_potential,
            'effusive_potential': effusive_potential,
            'overall_activity_score': (explosive_potential + effusive_potential) / 2
        }
    
    def save_embeddings(self, filepath: str):
        """Save learned embeddings to file"""
        
        save_data = {
            'node_embeddings': self.node_embeddings,
            'edge_embeddings': self.edge_embeddings,
            'graph_embeddings': self.graph_embeddings,
            'config': self.config,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str):
        """Load learned embeddings from file"""
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.node_embeddings = save_data.get('node_embeddings', {})
        self.edge_embeddings = save_data.get('edge_embeddings', {})
        self.graph_embeddings = save_data.get('graph_embeddings', {})
        self.config.update(save_data.get('config', {}))
        self.training_history = save_data.get('training_history', [])
        
        self.logger.info(f"Embeddings loaded from {filepath}")

# Example usage and testing
async def example_geological_network_representation():
    """Example of using geological network representation learning"""
    
    # Initialize system
    system = GeologicalNetworkRepresentationLearning()
    
    # Create sample geological entities with supereruption focus
    entities = []
    
    # Yellowstone Caldera (supereruption potential)
    yellowstone = GeologicalEntity(
        id="yellowstone_caldera",
        entity_type=GeologicalEntityType.CALDERA,
        coordinates=np.array([-110.5, 44.6, 0]),
        properties={
            'caldera_diameter': 55.0,  # km
            'magma_chamber_volume': 15000,  # km³
            'magma_chamber_depth': 8000,  # m
            'crustal_thickness': 45.0,  # km
            'silica_content': 75.0,  # wt% SiO2
            'temperature': 850.0,  # °C
            'historical_vei_max': 8.0,
            'age_last_eruption': 0.64e6,  # years
            'thermal_gradient': 60.0  # °C/km
        }
    )
    entities.append(yellowstone)
    
    # Toba Caldera (past supereruption)
    toba = GeologicalEntity(
        id="toba_caldera",
        entity_type=GeologicalEntityType.CALDERA,
        coordinates=np.array([98.8, 2.5, 0]),
        properties={
            'caldera_diameter': 100.0,
            'magma_chamber_volume': 20000,
            'magma_chamber_depth': 12000,
            'crustal_thickness': 35.0,
            'silica_content': 78.0,
            'temperature': 800.0,
            'historical_vei_max': 8.0,
            'age_last_eruption': 74000,
            'thermal_gradient': 45.0
        }
    )
    entities.append(toba)
    
    # Regular volcanic system for comparison
    mount_vesuvius = GeologicalEntity(
        id="mount_vesuvius",
        entity_type=GeologicalEntityType.VOLCANO,
        coordinates=np.array([14.4, 40.8, 0]),
        properties={
            'caldera_diameter': 2.0,
            'magma_chamber_volume': 50,
            'magma_chamber_depth': 3000,
            'crustal_thickness': 25.0,
            'silica_content': 60.0,
            'temperature': 1000.0,
            'historical_vei_max': 5.0,
            'age_last_eruption': 1944,
            'thermal_gradient': 30.0
        }
    )
    entities.append(mount_vesuvius)
    
    # Add entities to system
    for entity in entities:
        system.add_geological_entity(entity)
    
    # Create sample supereruption process
    supereruption = GeologicalProcess(
        id="toba_supereruption",
        process_type=GeologicalProcessType.SUPERERUPTION,
        entity_ids=["toba_caldera"],
        parameters={
            'vei': 8.0,
            'volume_erupted': 2800,  # km³ DRE
            'duration': 10,  # years
            'precursor_duration': 1000,  # years
            'seismic_rate_increase': 0.8,
            'ground_deformation_rate': 0.9,
            'gas_emission_changes': 0.7,
            'thermal_anomalies': 0.6
        },
        time_range=(74000, 73990),  # years ago
        spatial_extent={
            'affected_area': 10000000,  # km²
            'ash_thickness_max': 50.0  # m
        }
    )
    system.add_geological_process(supereruption)
    
    # Create relationships
    relationship = GeologicalRelationship(
        id="toba_supereruption_relationship",
        source_id="toba_caldera",
        target_id="toba_supereruption",
        relationship_type=RelationshipType.CAUSAL,
        strength=1.0,
        properties={
            'lead_time': 1000,  # years
            'magnitude_correlation': 0.95
        }
    )
    system.add_geological_relationship(relationship)
    
    print("Created geological network with supereruption focus:")
    print(f"Entities: {len(system.entities)}")
    print(f"Processes: {len(system.processes)}")
    print(f"Relationships: {len(system.relationships)}")
    
    # Train embeddings
    print("\nTraining embeddings...")
    training_results = await system.train_embeddings()
    print(f"Training completed with final loss: {training_results.get('epoch_losses', [0])[-1]:.6f}")
    
    # Test similarity search
    print("\nFinding systems similar to Yellowstone:")
    similar_systems = system.find_similar_systems("yellowstone_caldera", k=3)
    for system_id, similarity, metadata in similar_systems:
        print(f"  {system_id}: similarity={similarity:.3f}")
    
    # Test supereruption analog search
    print("\nFinding supereruption analogs for Yellowstone:")
    supereruption_analogs = system.find_supereruption_analogs("yellowstone_caldera", k=3)
    for system_id, score, metadata in supereruption_analogs:
        print(f"  {system_id}: analog_score={score:.3f}")
    
    # Analyze magma reservoir system
    print("\nAnalyzing Yellowstone magma reservoir system:")
    analysis = system.analyze_magma_reservoir_systems("yellowstone_caldera")
    
    chamber_analysis = analysis.get('magma_chamber_analysis', {})
    print(f"  Chamber volume: {chamber_analysis.get('volume_km3')} km³")
    print(f"  Volume class: {chamber_analysis.get('volume_class')}")
    print(f"  Depth class: {chamber_analysis.get('depth_class')}")
    
    eruption_potential = chamber_analysis.get('eruption_potential', {})
    print(f"  Explosive potential: {eruption_potential.get('explosive_potential', 0):.3f}")
    print(f"  Activity score: {eruption_potential.get('overall_activity_score', 0):.3f}")

if __name__ == "__main__":
    print("Geological Network Representation Learning System")
    print(f"DGL available: {DGL_AVAILABLE}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    print(f"UMAP available: {UMAP_AVAILABLE}")
    
    # Run example
    asyncio.run(example_geological_network_representation())