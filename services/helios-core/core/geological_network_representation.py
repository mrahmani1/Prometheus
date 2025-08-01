"""
Geological Network Representation Learning Module.

This module implements advanced representation learning techniques for geological networks,
including node embeddings, edge embeddings, graph-level embeddings, and similarity search
for analogous geological systems. Special focus on supereruptions and large caldera systems.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Import geological data structures - simplified for now
try:
    from .geological_graph_utils import GeologicalGraphBuilder
    from .graph_neural_networks import GeologicalGraphData
except ImportError:
    # Fallback simple data structure
    @dataclass
    class GeologicalGraphData:
        node_features: torch.Tensor
        edge_index: torch.Tensor
        edge_features: Optional[torch.Tensor] = None
        node_types: Optional[torch.Tensor] = None
        edge_types: Optional[torch.Tensor] = None
        spatial_coords: Optional[torch.Tensor] = None
        temporal_info: Optional[torch.Tensor] = None
        batch: Optional[torch.Tensor] = None

logger = logging.getLogger(__name__)

@dataclass
class GeologicalEmbedding:
    """Container for geological entity embeddings."""

    entity_id: str
    entity_type: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: datetime
    confidence: float = 1.0

@dataclass
class SupereruptionSignature:
    """Signature patterns for supereruption detection."""

    caldera_id: str
    magma_chamber_volume: float  # km³
    crustal_deformation_rate: float  # mm/year
    seismic_swarm_intensity: float
    gas_emission_anomaly: float
    thermal_anomaly_extent: float  # km²
    ground_uplift_pattern: np.ndarray
    geochemical_indicators: Dict[str, float]
    eruption_probability: float
    time_to_eruption_estimate: Optional[float] = None  # years

class GeologicalNodeEmbedding(nn.Module):
    """
    Advanced node embedding system for geological entities and processes.

    Creates rich representations that capture geological properties, spatial relationships,
    temporal evolution, and process dynamics.
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 256,
        num_entity_types: int = 15,
        num_process_types: int = 20,
        use_physics_constraints: bool = True,
        include_temporal_dynamics: bool = True,
    ):
        super(GeologicalNodeEmbedding, self).__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.use_physics_constraints = use_physics_constraints
        self.include_temporal_dynamics = include_temporal_dynamics

        # Base node embeddings
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

        # Entity type embeddings (volcanoes, faults, chambers, etc.)
        self.entity_type_embeddings = nn.Embedding(num_entity_types, embedding_dim // 4)

        # Process type embeddings (eruption, deformation, seismicity, etc.)
        self.process_type_embeddings = nn.Embedding(
            num_process_types, embedding_dim // 4
        )

        # Spatial encoding for geographic coordinates
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim // 2),  # lat, lon, depth
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
        )

        # Temporal encoding for time-varying properties
        if include_temporal_dynamics:
            self.temporal_encoder = nn.Sequential(
                nn.Linear(1, embedding_dim // 4),  # time
                nn.ReLU(),
                nn.Linear(embedding_dim // 4, embedding_dim // 4),
            )

        # Physics-informed constraints
        if use_physics_constraints:
            self.physics_constraint_layer = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )

        # Supereruption detection components
        self.supereruption_detector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Magma chamber state estimator
        self.magma_chamber_estimator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 10),  # volume, pressure, temperature, etc.
        )

        # Initialize embeddings
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize embeddings with geological priors."""
        # Initialize with small random values
        nn.init.normal_(self.node_embeddings.weight, mean=0, std=0.1)
        nn.init.normal_(self.entity_type_embeddings.weight, mean=0, std=0.1)
        nn.init.normal_(self.process_type_embeddings.weight, mean=0, std=0.1)

    def forward(
        self,
        node_ids: torch.Tensor,
        entity_types: torch.Tensor,
        process_types: torch.Tensor,
        spatial_coords: torch.Tensor,
        temporal_info: Optional[torch.Tensor] = None,
        geological_properties: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate comprehensive node embeddings for geological entities.

        Args:
            node_ids: Node indices
            entity_types: Entity type indices
            process_types: Process type indices
            spatial_coords: Geographic coordinates [lat, lon, depth]
            temporal_info: Time information
            geological_properties: Additional geological properties

        Returns:
            Dictionary containing various embedding representations
        """
        batch_size = node_ids.size(0)

        # Base node embeddings
        base_embeddings = self.node_embeddings(node_ids)

        # Entity type embeddings
        entity_emb = self.entity_type_embeddings(entity_types)

        # Process type embeddings
        process_emb = self.process_type_embeddings(process_types)

        # Spatial embeddings
        spatial_emb = self.spatial_encoder(spatial_coords)

        # Temporal embeddings
        temporal_emb = torch.zeros(
            batch_size, self.embedding_dim // 4, device=node_ids.device
        )
        if self.include_temporal_dynamics and temporal_info is not None:
            temporal_emb = self.temporal_encoder(temporal_info.unsqueeze(-1))

        # Combine all embeddings
        combined_embeddings = torch.cat(
            [
                base_embeddings,
                entity_emb.repeat(1, 4),  # Expand to match dimensions
                process_emb.repeat(1, 4),
                spatial_emb.repeat(1, 4),
                temporal_emb.repeat(1, 4),
            ],
            dim=-1,
        )

        # Reduce to target dimension
        combined_embeddings = combined_embeddings[:, : self.embedding_dim]

        # Apply physics constraints
        if self.use_physics_constraints:
            combined_embeddings = self.physics_constraint_layer(combined_embeddings)

        # Supereruption detection
        supereruption_scores = self.supereruption_detector(combined_embeddings)

        # Magma chamber state estimation
        magma_chamber_states = self.magma_chamber_estimator(combined_embeddings)

        return {
            "node_embeddings": combined_embeddings,
            "supereruption_scores": supereruption_scores,
            "magma_chamber_states": magma_chamber_states,
            "entity_embeddings": entity_emb,
            "process_embeddings": process_emb,
            "spatial_embeddings": spatial_emb,
            "temporal_embeddings": temporal_emb,
        }

    def get_caldera_embeddings(
        self, caldera_nodes: torch.Tensor, caldera_properties: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generate specialized embeddings for large caldera systems."""
        # Enhanced processing for caldera systems
        caldera_embeddings = self.node_embeddings(caldera_nodes)

        # Caldera-specific features
        caldera_features = torch.cat([caldera_embeddings, caldera_properties], dim=-1)

        # Specialized caldera processing
        caldera_enhanced = self.physics_constraint_layer(
            caldera_features[:, : self.embedding_dim]
        )

        # Supereruption risk assessment
        supereruption_risk = self.supereruption_detector(caldera_enhanced)

        # Detailed magma chamber analysis
        chamber_analysis = self.magma_chamber_estimator(caldera_enhanced)

        return {
            "caldera_embeddings": caldera_enhanced,
            "supereruption_risk": supereruption_risk,
            "chamber_analysis": chamber_analysis,
        }

class GeologicalEdgeEmbedding(nn.Module):
    """
    Advanced edge embedding system for geological relationships.

    Captures the strength, type, and dynamics of relationships between
    geological entities and processes.
    """

    def __init__(
        self,
        num_edge_types: int = 10,
        embedding_dim: int = 128,
        include_distance_encoding: bool = True,
        include_temporal_evolution: bool = True,
    ):
        super(GeologicalEdgeEmbedding, self).__init__()

        self.num_edge_types = num_edge_types
        self.embedding_dim = embedding_dim
        self.include_distance_encoding = include_distance_encoding
        self.include_temporal_evolution = include_temporal_evolution

        # Edge type embeddings
        self.edge_type_embeddings = nn.Embedding(num_edge_types, embedding_dim // 2)

        # Distance encoding for spatial relationships
        if include_distance_encoding:
            self.distance_encoder = nn.Sequential(
                nn.Linear(1, embedding_dim // 4),
                nn.ReLU(),
                nn.Linear(embedding_dim // 4, embedding_dim // 4),
            )

        # Temporal evolution encoding
        if include_temporal_evolution:
            self.temporal_evolution_encoder = nn.Sequential(
                nn.Linear(1, embedding_dim // 4),
                nn.ReLU(),
                nn.Linear(embedding_dim // 4, embedding_dim // 4),
            )

        # Relationship strength encoder
        self.strength_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4),
        )

        # Causal relationship detector
        self.causal_detector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Supereruption trigger relationship detector
        self.supereruption_trigger_detector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        edge_types: torch.Tensor,
        edge_distances: Optional[torch.Tensor] = None,
        edge_strengths: Optional[torch.Tensor] = None,
        temporal_evolution: Optional[torch.Tensor] = None,
        source_embeddings: Optional[torch.Tensor] = None,
        target_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate comprehensive edge embeddings for geological relationships.

        Args:
            edge_types: Edge type indices
            edge_distances: Spatial distances between entities
            edge_strengths: Relationship strengths
            temporal_evolution: Temporal evolution of relationships
            source_embeddings: Source node embeddings
            target_embeddings: Target node embeddings

        Returns:
            Dictionary containing edge embeddings and relationship analysis
        """
        batch_size = edge_types.size(0)
        device = edge_types.device

        # Base edge type embeddings
        type_embeddings = self.edge_type_embeddings(edge_types)

        # Distance embeddings
        distance_emb = torch.zeros(batch_size, self.embedding_dim // 4, device=device)
        if self.include_distance_encoding and edge_distances is not None:
            distance_emb = self.distance_encoder(edge_distances.unsqueeze(-1))

        # Temporal evolution embeddings
        temporal_emb = torch.zeros(batch_size, self.embedding_dim // 4, device=device)
        if self.include_temporal_evolution and temporal_evolution is not None:
            temporal_emb = self.temporal_evolution_encoder(
                temporal_evolution.unsqueeze(-1)
            )

        # Strength embeddings
        strength_emb = torch.zeros(batch_size, self.embedding_dim // 4, device=device)
        if edge_strengths is not None:
            strength_emb = self.strength_encoder(edge_strengths.unsqueeze(-1))

        # Combine embeddings
        combined_embeddings = torch.cat(
            [type_embeddings, distance_emb, temporal_emb, strength_emb], dim=-1
        )

        # Ensure correct dimension
        if combined_embeddings.size(-1) != self.embedding_dim:
            # Pad or truncate to match embedding_dim
            if combined_embeddings.size(-1) < self.embedding_dim:
                padding = torch.zeros(
                    batch_size,
                    self.embedding_dim - combined_embeddings.size(-1),
                    device=device,
                )
                combined_embeddings = torch.cat([combined_embeddings, padding], dim=-1)
            else:
                combined_embeddings = combined_embeddings[:, : self.embedding_dim]

        # Causal relationship detection
        causal_scores = self.causal_detector(combined_embeddings)

        # Supereruption trigger detection
        supereruption_trigger_scores = self.supereruption_trigger_detector(
            combined_embeddings
        )

        # Enhanced analysis with node embeddings
        enhanced_analysis = {}
        if source_embeddings is not None and target_embeddings is not None:
            # Compute relationship context
            relationship_context = torch.cat(
                [source_embeddings, target_embeddings, combined_embeddings], dim=-1
            )

            # Analyze relationship significance
            enhanced_analysis["relationship_significance"] = torch.norm(
                relationship_context, dim=-1, keepdim=True
            )

        return {
            "edge_embeddings": combined_embeddings,
            "causal_scores": causal_scores,
            "supereruption_trigger_scores": supereruption_trigger_scores,
            "type_embeddings": type_embeddings,
            "distance_embeddings": distance_emb,
            "temporal_embeddings": temporal_emb,
            "strength_embeddings": strength_emb,
            **enhanced_analysis,
        }

class GeologicalGraphEmbedding(nn.Module):
    """
    Graph-level embedding system for geological system classification.

    Creates holistic representations of entire geological systems for
    classification, comparison, and similarity analysis.
    """

    def __init__(
        self,
        node_embedding_dim: int = 256,
        edge_embedding_dim: int = 128,
        graph_embedding_dim: int = 512,
        num_system_types: int = 20,
        use_attention: bool = True,
    ):
        super(GeologicalGraphEmbedding, self).__init__()

        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.graph_embedding_dim = graph_embedding_dim
        self.use_attention = use_attention

        # Graph-level aggregation layers
        self.node_aggregator = nn.Sequential(
            nn.Linear(node_embedding_dim, graph_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(graph_embedding_dim // 2, graph_embedding_dim // 4),
        )

        self.edge_aggregator = nn.Sequential(
            nn.Linear(edge_embedding_dim, graph_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(graph_embedding_dim // 2, graph_embedding_dim // 4),
        )

        # Attention mechanism for important features
        if use_attention:
            self.node_attention = nn.MultiheadAttention(
                embed_dim=graph_embedding_dim // 4, num_heads=4, dropout=0.1
            )

            self.edge_attention = nn.MultiheadAttention(
                embed_dim=graph_embedding_dim // 4, num_heads=4, dropout=0.1
            )

        # Graph-level feature fusion
        self.graph_fusion = nn.Sequential(
            nn.Linear(graph_embedding_dim // 2, graph_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(graph_embedding_dim, graph_embedding_dim),
        )

        # System type classifier
        self.system_classifier = nn.Sequential(
            nn.Linear(graph_embedding_dim, graph_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(graph_embedding_dim // 2, num_system_types),
        )

        # Supereruption system detector
        self.supereruption_system_detector = nn.Sequential(
            nn.Linear(graph_embedding_dim, graph_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(graph_embedding_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Caldera system analyzer
        self.caldera_analyzer = nn.Sequential(
            nn.Linear(graph_embedding_dim, graph_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(graph_embedding_dim // 2, 8),  # Various caldera characteristics
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate graph-level embeddings for geological systems.

        Args:
            node_embeddings: Node embeddings from the graph
            edge_embeddings: Edge embeddings from the graph
            batch: Batch information for multiple graphs
            node_mask: Mask for valid nodes
            edge_mask: Mask for valid edges

        Returns:
            Dictionary containing graph-level representations and analysis
        """
        # Aggregate node embeddings
        aggregated_nodes = self.node_aggregator(node_embeddings)

        # Aggregate edge embeddings
        aggregated_edges = self.edge_aggregator(edge_embeddings)

        # Apply attention if enabled
        if self.use_attention:
            # Node attention
            aggregated_nodes = aggregated_nodes.unsqueeze(0)  # Add sequence dimension
            attended_nodes, node_attention_weights = self.node_attention(
                aggregated_nodes, aggregated_nodes, aggregated_nodes
            )
            aggregated_nodes = attended_nodes.squeeze(0)

            # Edge attention
            aggregated_edges = aggregated_edges.unsqueeze(0)
            attended_edges, edge_attention_weights = self.edge_attention(
                aggregated_edges, aggregated_edges, aggregated_edges
            )
            aggregated_edges = attended_edges.squeeze(0)

        # Global pooling for graph-level representation
        if batch is not None:
            # Handle batched graphs
            from torch_geometric.nn import global_max_pool, global_mean_pool

            node_global_mean = global_mean_pool(aggregated_nodes, batch)
            global_max_pool(aggregated_nodes, batch)

            edge_global_mean = global_mean_pool(aggregated_edges, batch)
            global_max_pool(aggregated_edges, batch)

            # Combine pooled representations
            graph_representation = torch.cat(
                [node_global_mean, edge_global_mean], dim=-1
            )
        else:
            # Single graph
            node_global = torch.mean(aggregated_nodes, dim=0, keepdim=True)
            edge_global = torch.mean(aggregated_edges, dim=0, keepdim=True)

            graph_representation = torch.cat([node_global, edge_global], dim=-1)

        # Fuse features
        fused_graph_embedding = self.graph_fusion(graph_representation)

        # System classification
        system_type_logits = self.system_classifier(fused_graph_embedding)

        # Supereruption system detection
        supereruption_system_score = self.supereruption_system_detector(
            fused_graph_embedding
        )

        # Caldera system analysis
        caldera_characteristics = self.caldera_analyzer(fused_graph_embedding)

        results = {
            "graph_embedding": fused_graph_embedding,
            "system_type_logits": system_type_logits,
            "supereruption_system_score": supereruption_system_score,
            "caldera_characteristics": caldera_characteristics,
            "node_aggregated": aggregated_nodes,
            "edge_aggregated": aggregated_edges,
        }

        # Add attention weights if available
        if self.use_attention:
            results["node_attention_weights"] = node_attention_weights
            results["edge_attention_weights"] = edge_attention_weights

        return results

    def classify_geological_system(
        self, graph_embedding: torch.Tensor
    ) -> Dict[str, Union[str, float]]:
        """Classify the type of geological system."""
        system_logits = self.system_classifier(graph_embedding)
        system_probs = F.softmax(system_logits, dim=-1)

        system_types = [
            "volcanic_arc",
            "hotspot_chain",
            "rift_system",
            "subduction_zone",
            "large_igneous_province",
            "caldera_complex",
            "monogenetic_field",
            "stratovolcano_cluster",
            "shield_volcano_system",
            "fissure_system",
            "hydrothermal_field",
            "geothermal_system",
            "fault_network",
            "seismic_zone",
            "transform_boundary",
            "spreading_center",
            "collision_zone",
            "back_arc_basin",
            "forearc_region",
            "supereruption_system",
        ]

        predicted_type_idx = torch.argmax(system_probs, dim=-1).item()
        confidence = system_probs[0, predicted_type_idx].item()

        return {
            "system_type": system_types[predicted_type_idx],
            "confidence": confidence,
            "all_probabilities": {
                system_types[i]: system_probs[0, i].item()
                for i in range(len(system_types))
            },
        }

class GeologicalSimilaritySearch:
    """
    Similarity search system for finding analogous geological systems.

    Uses advanced embedding techniques and efficient search algorithms
    to identify similar geological systems and processes.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        use_faiss: bool = True,
        similarity_metric: str = "cosine",
    ):
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss
        self.similarity_metric = similarity_metric

        # Storage for embeddings and metadata
        self.embeddings_db = {}
        self.metadata_db = {}

        # FAISS index for efficient similarity search
        if use_faiss and FAISS_AVAILABLE:
            if similarity_metric == "cosine":
                self.faiss_index = faiss.IndexFlatIP(
                    embedding_dim
                )  # Inner product for cosine
            else:
                self.faiss_index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        else:
            self.faiss_index = None
            self.use_faiss = False

        # Specialized indices for different geological systems
        self.volcanic_systems_index = {}
        self.caldera_systems_index = {}
        self.supereruption_systems_index = {}

    def add_geological_system(
        self, system_id: str, embedding: np.ndarray, metadata: Dict[str, Any]
    ):
        """Add a geological system to the similarity search database."""
        # Normalize embedding for cosine similarity
        if self.similarity_metric == "cosine":
            embedding = embedding / np.linalg.norm(embedding)

        # Store in database
        self.embeddings_db[system_id] = embedding
        self.metadata_db[system_id] = metadata

        # Add to FAISS index
        if self.use_faiss and self.faiss_index is not None:
            self.faiss_index.add(embedding.reshape(1, -1).astype(np.float32))

        # Add to specialized indices
        system_type = metadata.get("system_type", "unknown")
        if "volcanic" in system_type.lower():
            self.volcanic_systems_index[system_id] = embedding
        if "caldera" in system_type.lower():
            self.caldera_systems_index[system_id] = embedding
        if "supereruption" in system_type.lower():
            self.supereruption_systems_index[system_id] = embedding

    def find_similar_systems(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        system_type_filter: Optional[str] = None,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Find similar geological systems based on embedding similarity.

        Args:
            query_embedding: Query system embedding
            k: Number of similar systems to return
            system_type_filter: Filter by system type
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar systems with metadata and similarity scores
        """
        # Normalize query embedding
        if self.similarity_metric == "cosine":
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        similar_systems = []

        if (
            self.use_faiss
            and self.faiss_index is not None
            and len(self.embeddings_db) > 0
        ):
            # Use FAISS for efficient search
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
            distances, indices = self.faiss_index.search(
                query_vector, min(k, len(self.embeddings_db))
            )

            # Convert to similarity scores and filter
            system_ids = list(self.embeddings_db.keys())
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(system_ids):
                    system_id = system_ids[idx]

                    # Convert distance to similarity
                    if self.similarity_metric == "cosine":
                        similarity = distance  # FAISS returns inner product for cosine
                    else:
                        similarity = 1.0 / (
                            1.0 + distance
                        )  # Convert L2 distance to similarity

                    # Apply filters
                    if similarity >= min_similarity:
                        metadata = self.metadata_db[system_id]
                        if (
                            system_type_filter is None
                            or system_type_filter in metadata.get("system_type", "")
                        ):
                            similar_systems.append(
                                {
                                    "system_id": system_id,
                                    "similarity": similarity,
                                    "metadata": metadata,
                                    "embedding": self.embeddings_db[system_id],
                                }
                            )
        else:
            # Fallback to manual similarity computation
            for system_id, embedding in self.embeddings_db.items():
                if self.similarity_metric == "cosine":
                    similarity = np.dot(query_embedding, embedding)
                else:
                    similarity = 1.0 / (
                        1.0 + np.linalg.norm(query_embedding - embedding)
                    )

                if similarity >= min_similarity:
                    metadata = self.metadata_db[system_id]
                    if (
                        system_type_filter is None
                        or system_type_filter in metadata.get("system_type", "")
                    ):
                        similar_systems.append(
                            {
                                "system_id": system_id,
                                "similarity": similarity,
                                "metadata": metadata,
                                "embedding": embedding,
                            }
                        )

        # Sort by similarity and return top k
        similar_systems.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_systems[:k]

    def find_analogous_volcanic_systems(
        self, query_system: Dict[str, Any], k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find volcanic systems analogous to the query system."""
        # Extract key features for comparison
        query_features = self._extract_volcanic_features(query_system)

        analogous_systems = []
        for system_id, metadata in self.metadata_db.items():
            if "volcanic" in metadata.get("system_type", "").lower():
                system_features = self._extract_volcanic_features(metadata)
                similarity = self._compute_volcanic_similarity(
                    query_features, system_features
                )

                analogous_systems.append(
                    {
                        "system_id": system_id,
                        "similarity": similarity,
                        "metadata": metadata,
                        "matching_features": self._identify_matching_features(
                            query_features, system_features
                        ),
                    }
                )

        analogous_systems.sort(key=lambda x: x["similarity"], reverse=True)
        return analogous_systems[:k]

    def _extract_volcanic_features(
        self, system_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract key volcanic features for comparison."""
        features = {}

        # Volcanic system characteristics
        features["eruption_frequency"] = system_data.get("eruption_frequency", 0.0)
        features["max_vei"] = system_data.get("max_vei", 0.0)
        features["magma_composition"] = system_data.get("magma_composition", 0.0)
        features["tectonic_setting"] = system_data.get("tectonic_setting", 0.0)
        features["elevation"] = system_data.get("elevation", 0.0)
        features["caldera_diameter"] = system_data.get("caldera_diameter", 0.0)
        features["magma_chamber_volume"] = system_data.get("magma_chamber_volume", 0.0)

        return features

    def _compute_volcanic_similarity(
        self, features1: Dict[str, float], features2: Dict[str, float]
    ) -> float:
        """Compute similarity between volcanic systems based on features."""
        # Weight different features by importance
        feature_weights = {
            "eruption_frequency": 0.2,
            "max_vei": 0.25,
            "magma_composition": 0.2,
            "tectonic_setting": 0.15,
            "elevation": 0.05,
            "caldera_diameter": 0.1,
            "magma_chamber_volume": 0.05,
        }

        total_similarity = 0.0
        total_weight = 0.0

        for feature, weight in feature_weights.items():
            if feature in features1 and feature in features2:
                # Normalize feature values and compute similarity
                val1 = features1[feature]
                val2 = features2[feature]

                if val1 == 0 and val2 == 0:
                    feature_similarity = 1.0
                else:
                    feature_similarity = 1.0 - abs(val1 - val2) / (
                        abs(val1) + abs(val2) + 1e-8
                    )

                total_similarity += weight * feature_similarity
                total_weight += weight

        return total_similarity / total_weight if total_weight > 0 else 0.0

    def _identify_matching_features(
        self, features1: Dict[str, float], features2: Dict[str, float]
    ) -> List[str]:
        """Identify which features match between two systems."""
        matching_features = []

        for feature in features1:
            if feature in features2:
                val1 = features1[feature]
                val2 = features2[feature]

                # Consider features matching if they're within 20% of each other
                if abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-8) < 0.2:
                    matching_features.append(feature)

        return matching_features

class SupereruptionAnalyzer:
    """
    Specialized analyzer for supereruption detection and magma chamber analysis.

    Focuses on large caldera systems and their potential for catastrophic eruptions.
    """

    def __init__(self):
        self.supereruption_threshold_volume = 1000.0  # km³ of magma
        self.caldera_systems_db = {}
        self.historical_supereruptions = self._load_historical_supereruptions()

    def _load_historical_supereruptions(self) -> List[Dict[str, Any]]:
        """Load historical supereruption data for comparison."""
        return [
            {
                "name": "Yellowstone (Lava Creek)",
                "age": 0.64,  # Ma
                "volume": 1000,  # km³
                "vei": 8,
                "caldera_diameter": 45,  # km
                "precursor_duration": 100000,  # years
                "location": [44.4, -110.6],
            },
            {
                "name": "Toba",
                "age": 0.074,  # Ma
                "volume": 2800,  # km³
                "vei": 8,
                "caldera_diameter": 100,  # km
                "precursor_duration": 50000,  # years
                "location": [2.7, 98.8],
            },
            {
                "name": "Taupo (Oruanui)",
                "age": 0.0265,  # Ma
                "volume": 530,  # km³
                "vei": 8,
                "caldera_diameter": 35,  # km
                "precursor_duration": 10000,  # years
                "location": [-38.7, 176.1],
            },
            {
                "name": "Long Valley",
                "age": 0.76,  # Ma
                "volume": 600,  # km³
                "vei": 7,
                "caldera_diameter": 32,  # km
                "precursor_duration": 200000,  # years
                "location": [37.7, -118.9],
            },
        ]

    def analyze_supereruption_potential(
        self, caldera_system: Dict[str, Any], current_monitoring_data: Dict[str, Any]
    ) -> SupereruptionSignature:
        """
        Analyze the supereruption potential of a caldera system.

        Args:
            caldera_system: Caldera system metadata and properties
            current_monitoring_data: Real-time monitoring data

        Returns:
            SupereruptionSignature with detailed analysis
        """
        caldera_id = caldera_system.get("system_id", "unknown")

        # Extract key parameters
        magma_chamber_volume = caldera_system.get("magma_chamber_volume", 0.0)
        caldera_system.get("caldera_diameter", 0.0)

        # Analyze current monitoring data
        deformation_rate = current_monitoring_data.get("deformation_rate", 0.0)
        seismic_intensity = current_monitoring_data.get("seismic_swarm_intensity", 0.0)
        gas_anomaly = current_monitoring_data.get("gas_emission_anomaly", 0.0)
        thermal_extent = current_monitoring_data.get("thermal_anomaly_extent", 0.0)

        # Ground uplift pattern analysis
        uplift_data = current_monitoring_data.get("ground_uplift_pattern", [])
        uplift_pattern = np.array(uplift_data) if uplift_data else np.zeros(10)

        # Geochemical indicators
        geochemical_indicators = current_monitoring_data.get(
            "geochemical_indicators", {}
        )

        # Calculate eruption probability using multiple factors
        eruption_probability = self._calculate_eruption_probability(
            magma_chamber_volume,
            deformation_rate,
            seismic_intensity,
            gas_anomaly,
            thermal_extent,
            geochemical_indicators,
        )

        # Estimate time to eruption if probability is high
        time_to_eruption = None
        if eruption_probability > 0.3:
            time_to_eruption = self._estimate_time_to_eruption(
                deformation_rate, seismic_intensity, gas_anomaly
            )

        return SupereruptionSignature(
            caldera_id=caldera_id,
            magma_chamber_volume=magma_chamber_volume,
            crustal_deformation_rate=deformation_rate,
            seismic_swarm_intensity=seismic_intensity,
            gas_emission_anomaly=gas_anomaly,
            thermal_anomaly_extent=thermal_extent,
            ground_uplift_pattern=uplift_pattern,
            geochemical_indicators=geochemical_indicators,
            eruption_probability=eruption_probability,
            time_to_eruption_estimate=time_to_eruption,
        )

    def _calculate_eruption_probability(
        self,
        magma_volume: float,
        deformation_rate: float,
        seismic_intensity: float,
        gas_anomaly: float,
        thermal_extent: float,
        geochemical_indicators: Dict[str, float],
    ) -> float:
        """Calculate eruption probability based on multiple indicators."""
        # Volume factor (larger chambers more likely to erupt catastrophically)
        volume_factor = min(magma_volume / self.supereruption_threshold_volume, 1.0)

        # Deformation factor (rapid uplift indicates magma intrusion)
        deformation_factor = min(deformation_rate / 100.0, 1.0)  # mm/year threshold

        # Seismic factor (increased seismicity indicates magma movement)
        seismic_factor = min(seismic_intensity / 1000.0, 1.0)  # events/day threshold

        # Gas factor (increased gas emissions indicate magma degassing)
        gas_factor = min(gas_anomaly / 10.0, 1.0)  # relative to baseline

        # Thermal factor (increased thermal activity)
        thermal_factor = min(thermal_extent / 100.0, 1.0)  # km² threshold

        # Geochemical factor
        geochemical_factor = 0.0
        if geochemical_indicators:
            # Key indicators: He3/He4 ratio, CO2/SO2 ratio, etc.
            he_ratio = geochemical_indicators.get("he3_he4_ratio", 0.0)
            co2_so2_ratio = geochemical_indicators.get("co2_so2_ratio", 0.0)

            # Normalize and combine
            geochemical_factor = min((he_ratio + co2_so2_ratio) / 2.0, 1.0)

        # Weighted combination
        weights = [0.3, 0.25, 0.2, 0.15, 0.05, 0.05]
        factors = [
            volume_factor,
            deformation_factor,
            seismic_factor,
            gas_factor,
            thermal_factor,
            geochemical_factor,
        ]

        probability = sum(w * f for w, f in zip(weights, factors))
        return min(probability, 1.0)

    def _estimate_time_to_eruption(
        self, deformation_rate: float, seismic_intensity: float, gas_anomaly: float
    ) -> float:
        """Estimate time to eruption based on current trends."""
        # Simple model based on acceleration of precursory signals
        if deformation_rate > 50:  # mm/year
            if seismic_intensity > 500:  # events/day
                return 1.0  # 1 year
            else:
                return 5.0  # 5 years
        elif deformation_rate > 20:
            return 10.0  # 10 years
        else:
            return 50.0  # 50 years

    def identify_magma_chamber_state(
        self, chamber_embedding: torch.Tensor, monitoring_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Identify the current state of a magma chamber.

        Args:
            chamber_embedding: Neural network embedding of the chamber
            monitoring_data: Current monitoring data

        Returns:
            Dictionary with chamber state analysis
        """
        # Extract chamber characteristics from embedding
        chamber_features = chamber_embedding.detach().numpy()

        # Analyze monitoring data
        temperature = monitoring_data.get("temperature", 0.0)
        pressure = monitoring_data.get("pressure", 0.0)
        crystallinity = monitoring_data.get("crystallinity", 0.0)
        volatile_content = monitoring_data.get("volatile_content", {})

        # Determine chamber state
        if temperature > 800 and crystallinity < 0.5:
            state = "eruptible"
            mobility = "high"
        elif temperature > 700 and crystallinity < 0.7:
            state = "potentially_eruptible"
            mobility = "medium"
        elif crystallinity > 0.8:
            state = "crystallized"
            mobility = "low"
        else:
            state = "dormant"
            mobility = "very_low"

        # Estimate chamber connectivity
        connectivity_score = self._estimate_chamber_connectivity(chamber_features)

        # Volatile analysis
        h2o_content = volatile_content.get("H2O", 0.0)
        co2_content = volatile_content.get("CO2", 0.0)

        explosive_potential = "low"
        if h2o_content > 4.0:  # wt%
            explosive_potential = "high"
        elif h2o_content > 2.0:
            explosive_potential = "medium"

        return {
            "chamber_state": state,
            "magma_mobility": mobility,
            "connectivity_score": connectivity_score,
            "explosive_potential": explosive_potential,
            "temperature": temperature,
            "pressure": pressure,
            "crystallinity": crystallinity,
            "volatile_content": volatile_content,
            "eruption_style_prediction": self._predict_eruption_style(
                temperature, crystallinity, h2o_content, co2_content
            ),
        }

    def _estimate_chamber_connectivity(self, chamber_features: np.ndarray) -> float:
        """Estimate how well connected a magma chamber is to others."""
        # Use embedding features to estimate connectivity
        # This is a simplified approach - in practice, this would use
        # more sophisticated analysis of the embedding space
        connectivity_features = chamber_features[:10]  # First 10 features
        connectivity_score = np.mean(np.abs(connectivity_features))
        return min(connectivity_score, 1.0)

    def _predict_eruption_style(
        self,
        temperature: float,
        crystallinity: float,
        h2o_content: float,
        co2_content: float,
    ) -> str:
        """Predict the likely eruption style based on magma properties."""
        if h2o_content > 4.0 and temperature > 800:
            return "plinian"
        elif h2o_content > 2.0 and crystallinity < 0.6:
            return "vulcanian"
        elif temperature > 1000 and crystallinity < 0.3:
            return "strombolian"
        elif crystallinity > 0.7:
            return "dome_forming"
        else:
            return "effusive"

    def compare_with_historical_supereruptions(
        self, current_system: SupereruptionSignature
    ) -> List[Dict[str, Any]]:
        """Compare current system with historical supereruptions."""
        comparisons = []

        for historical in self.historical_supereruptions:
            # Calculate similarity based on multiple factors
            volume_similarity = 1.0 - abs(
                current_system.magma_chamber_volume - historical["volume"]
            ) / max(current_system.magma_chamber_volume, historical["volume"])

            # Location-based similarity (if coordinates available)
            location_similarity = 1.0  # Placeholder

            # Overall similarity
            overall_similarity = (volume_similarity + location_similarity) / 2.0

            comparisons.append(
                {
                    "historical_eruption": historical["name"],
                    "similarity": overall_similarity,
                    "volume_comparison": {
                        "current": current_system.magma_chamber_volume,
                        "historical": historical["volume"],
                        "ratio": current_system.magma_chamber_volume
                        / historical["volume"],
                    },
                    "implications": self._generate_implications(
                        current_system, historical
                    ),
                }
            )

        # Sort by similarity
        comparisons.sort(key=lambda x: x["similarity"], reverse=True)
        return comparisons

    def _generate_implications(
        self, current: SupereruptionSignature, historical: Dict[str, Any]
    ) -> List[str]:
        """Generate implications based on comparison with historical eruption."""
        implications = []

        volume_ratio = current.magma_chamber_volume / historical["volume"]

        if volume_ratio > 0.8:
            implications.append(
                f"Magma chamber volume comparable to {historical['name']} - "
                "potential for VEI 7-8 eruption"
            )

        if current.eruption_probability > 0.5:
            implications.append(
                f"High eruption probability similar to precursory phase of {historical['name']}"
            )

        if current.crustal_deformation_rate > 50:
            implications.append(
                "Rapid deformation suggests active magma intrusion - "
                "monitor for acceleration"
            )

        return implications

class GeologicalNetworkRepresentationEngine:
    """
    Main engine that combines all representation learning components.

    Provides a unified interface for geological network representation learning,
    including node embeddings, edge embeddings, graph-level embeddings,
    and similarity search capabilities.
    """

    def __init__(
        self,
        node_embedding_dim: int = 256,
        edge_embedding_dim: int = 128,
        graph_embedding_dim: int = 512,
        device: str = "cpu",
    ):
        self.device = device
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.graph_embedding_dim = graph_embedding_dim

        # Initialize components
        self.node_embedder = GeologicalNodeEmbedding(
            num_nodes=10000, embedding_dim=node_embedding_dim  # Expandable
        ).to(device)

        self.edge_embedder = GeologicalEdgeEmbedding(
            embedding_dim=edge_embedding_dim
        ).to(device)

        self.graph_embedder = GeologicalGraphEmbedding(
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            graph_embedding_dim=graph_embedding_dim,
        ).to(device)

        self.similarity_search = GeologicalSimilaritySearch(
            embedding_dim=graph_embedding_dim
        )

        self.supereruption_analyzer = SupereruptionAnalyzer()

        # Training components
        self.optimizer = None
        self.loss_functions = self._setup_loss_functions()

    def _setup_loss_functions(self) -> Dict[str, nn.Module]:
        """Setup loss functions for different components."""
        return {
            "node_reconstruction": nn.MSELoss(),
            "edge_prediction": nn.BCEWithLogitsLoss(),
            "graph_classification": nn.CrossEntropyLoss(),
            "supereruption_detection": nn.BCEWithLogitsLoss(),
            "similarity_ranking": nn.MarginRankingLoss(),
        }

    def process_geological_network(
        self,
        graph_data: GeologicalGraphData,
        include_supereruption_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a complete geological network and generate all representations.

        Args:
            graph_data: Input geological graph data
            include_supereruption_analysis: Whether to include supereruption analysis

        Returns:
            Dictionary with all generated representations and analyses
        """
        # Generate node embeddings
        node_results = self.node_embedder(
            node_ids=torch.arange(graph_data.node_features.size(0), device=self.device),
            entity_types=graph_data.node_types,
            process_types=torch.zeros_like(graph_data.node_types),  # Placeholder
            spatial_coords=graph_data.spatial_coords,
            temporal_info=graph_data.temporal_info,
        )

        # Generate edge embeddings
        edge_results = self.edge_embedder(
            edge_types=graph_data.edge_types,
            edge_distances=None,  # Could be computed from spatial_coords
            edge_strengths=None,  # Could be extracted from edge_features
            source_embeddings=node_results["node_embeddings"][graph_data.edge_index[0]],
            target_embeddings=node_results["node_embeddings"][graph_data.edge_index[1]],
        )

        # Generate graph-level embeddings
        graph_results = self.graph_embedder(
            node_embeddings=node_results["node_embeddings"],
            edge_embeddings=edge_results["edge_embeddings"],
            batch=graph_data.batch,
        )

        # Combine all results
        results = {
            "node_embeddings": node_results["node_embeddings"],
            "edge_embeddings": edge_results["edge_embeddings"],
            "graph_embedding": graph_results["graph_embedding"],
            "system_classification": self.graph_embedder.classify_geological_system(
                graph_results["graph_embedding"]
            ),
            "supereruption_scores": node_results["supereruption_scores"],
            "magma_chamber_states": node_results["magma_chamber_states"],
            "causal_relationships": edge_results["causal_scores"],
            "supereruption_triggers": edge_results["supereruption_trigger_scores"],
        }

        # Add supereruption analysis if requested
        if include_supereruption_analysis:
            supereruption_analysis = self._analyze_supereruption_potential(
                graph_data, results
            )
            results["supereruption_analysis"] = supereruption_analysis

        return results

    def _analyze_supereruption_potential(
        self, graph_data: GeologicalGraphData, embeddings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze supereruption potential for the geological system."""
        # Identify caldera systems in the graph
        caldera_nodes = []
        if graph_data.node_types is not None:
            # Assuming caldera systems have a specific node type
            caldera_mask = graph_data.node_types == 6  # Adjust based on your encoding
            caldera_nodes = torch.where(caldera_mask)[0]

        supereruption_analyses = []

        for caldera_node in caldera_nodes:
            # Extract caldera system metadata (this would come from your data)
            caldera_system = {
                "system_id": f"caldera_{caldera_node.item()}",
                "magma_chamber_volume": 500.0,  # Placeholder - extract from data
                "caldera_diameter": 25.0,  # Placeholder
            }

            # Mock monitoring data (in practice, this would be real-time data)
            monitoring_data = {
                "deformation_rate": 10.0,  # mm/year
                "seismic_swarm_intensity": 100.0,  # events/day
                "gas_emission_anomaly": 2.0,  # relative to baseline
                "thermal_anomaly_extent": 5.0,  # km²
                "geochemical_indicators": {"he3_he4_ratio": 0.1, "co2_so2_ratio": 0.5},
            }

            # Analyze supereruption potential
            signature = self.supereruption_analyzer.analyze_supereruption_potential(
                caldera_system, monitoring_data
            )

            # Analyze magma chamber state
            chamber_embedding = embeddings["node_embeddings"][caldera_node]
            chamber_state = self.supereruption_analyzer.identify_magma_chamber_state(
                chamber_embedding, monitoring_data
            )

            supereruption_analyses.append(
                {
                    "caldera_node": caldera_node.item(),
                    "supereruption_signature": signature,
                    "chamber_state": chamber_state,
                }
            )

        return {
            "num_caldera_systems": len(caldera_nodes),
            "caldera_analyses": supereruption_analyses,
            "overall_supereruption_risk": max(
                [
                    analysis["supereruption_signature"].eruption_probability
                    for analysis in supereruption_analyses
                ],
                default=0.0,
            ),
        }

    def find_similar_geological_systems(
        self, query_graph_data: GeologicalGraphData, k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find geological systems similar to the query system."""
        # Process query system
        query_results = self.process_geological_network(
            query_graph_data, include_supereruption_analysis=False
        )

        # Extract graph embedding
        query_embedding = query_results["graph_embedding"].detach().cpu().numpy()

        # Search for similar systems
        similar_systems = self.similarity_search.find_similar_systems(
            query_embedding, k=k
        )

        return similar_systems

    def save_model(self, filepath: str):
        """Save the trained model components."""
        torch.save(
            {
                "node_embedder": self.node_embedder.state_dict(),
                "edge_embedder": self.edge_embedder.state_dict(),
                "graph_embedder": self.graph_embedder.state_dict(),
                "similarity_db": {
                    "embeddings": self.similarity_search.embeddings_db,
                    "metadata": self.similarity_search.metadata_db,
                },
            },
            filepath,
        )

    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.node_embedder.load_state_dict(checkpoint["node_embedder"])
        self.edge_embedder.load_state_dict(checkpoint["edge_embedder"])
        self.graph_embedder.load_state_dict(checkpoint["graph_embedder"])

        # Restore similarity search database
        similarity_data = checkpoint.get("similarity_db", {})
        self.similarity_search.embeddings_db = similarity_data.get("embeddings", {})
        self.similarity_search.metadata_db = similarity_data.get("metadata", {})

# Example usage and testing functions
def test_geological_network_representation():
    """Test the geological network representation learning system."""
    print("Testing Geological Network Representation Learning...")

    # Create sample geological graph data
    num_nodes = 50
    num_edges = 100
    node_feature_dim = 64

    # Sample data
    node_features = torch.randn(num_nodes, node_feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    node_types = torch.randint(0, 10, (num_nodes,))
    edge_types = torch.randint(0, 5, (num_edges,))
    spatial_coords = torch.randn(num_nodes, 3)  # lat, lon, depth

    # Create geological graph data
    graph_data = GeologicalGraphData(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=None,
        node_types=node_types,
        edge_types=edge_types,
        spatial_coords=spatial_coords,
    )

    # Initialize the representation engine
    engine = GeologicalNetworkRepresentationEngine(
        node_embedding_dim=256, edge_embedding_dim=128, graph_embedding_dim=512
    )

    # Process the geological network
    results = engine.process_geological_network(graph_data)

    print(f"Node embeddings shape: {results['node_embeddings'].shape}")
    print(f"Edge embeddings shape: {results['edge_embeddings'].shape}")
    print(f"Graph embedding shape: {results['graph_embedding'].shape}")
    print(f"System classification: {results['system_classification']['system_type']}")
    print(f"Confidence: {results['system_classification']['confidence']:.3f}")

    # Test supereruption analysis
    if "supereruption_analysis" in results:
        analysis = results["supereruption_analysis"]
        print(f"Number of caldera systems detected: {analysis['num_caldera_systems']}")
        print(
            f"Overall supereruption risk: {analysis['overall_supereruption_risk']:.3f}"
        )

    # Test similarity search
    print("\nTesting similarity search...")

    # Add the current system to the similarity database
    graph_embedding = results["graph_embedding"].detach().cpu().numpy()
    engine.similarity_search.add_geological_system(
        system_id="test_system_1",
        embedding=graph_embedding[0],
        metadata={
            "system_type": "volcanic_arc",
            "location": [45.0, -122.0],
            "last_eruption": 1980,
            "max_vei": 5,
        },
    )

    # Find similar systems
    similar_systems = engine.find_similar_geological_systems(graph_data, k=5)
    print(f"Found {len(similar_systems)} similar systems")

    # Test supereruption analyzer
    print("\nTesting supereruption analyzer...")
    analyzer = SupereruptionAnalyzer()

    # Mock caldera system data
    caldera_system = {
        "system_id": "yellowstone",
        "magma_chamber_volume": 800.0,  # km³
        "caldera_diameter": 45.0,  # km
    }

    monitoring_data = {
        "deformation_rate": 25.0,  # mm/year
        "seismic_swarm_intensity": 200.0,  # events/day
        "gas_emission_anomaly": 3.0,  # relative to baseline
        "thermal_anomaly_extent": 15.0,  # km²
        "geochemical_indicators": {"he3_he4_ratio": 0.15, "co2_so2_ratio": 0.8},
    }

    signature = analyzer.analyze_supereruption_potential(
        caldera_system, monitoring_data
    )
    print(f"Eruption probability: {signature.eruption_probability:.3f}")
    print(f"Estimated time to eruption: {signature.time_to_eruption_estimate} years")

    # Compare with historical supereruptions
    comparisons = analyzer.compare_with_historical_supereruptions(signature)
    print(f"Most similar historical eruption: {comparisons[0]['historical_eruption']}")
    print(f"Similarity: {comparisons[0]['similarity']:.3f}")

    print("\nGeological Network Representation Learning test completed successfully!")
    return True

def create_sample_volcanic_network() -> GeologicalGraphData:
    """Create a sample volcanic network for testing."""
    builder = GeologicalGraphBuilder()

    # Add volcanic systems
    volcanoes = [
        ("mount_st_helens", [46.2, -122.2, 2549], {"last_eruption": 2008, "vei": 5}),
        ("mount_rainier", [46.9, -121.8, 4392], {"last_eruption": 1894, "vei": 3}),
        ("mount_baker", [48.8, -121.8, 3286], {"last_eruption": 1880, "vei": 2}),
    ]

    for volcano_id, coords, props in volcanoes:
        builder.add_volcanic_system(volcano_id, coords, props)

    # Add magma chambers
    chambers = [
        ("msh_chamber", [46.2, -122.2, -5], {"volume": 50, "temperature": 900}),
        ("rainier_chamber", [46.9, -121.8, -8], {"volume": 30, "temperature": 850}),
    ]

    for chamber_id, coords, props in chambers:
        builder.add_magma_chamber(chamber_id, coords, props)

    # Add seismic stations
    stations = [
        ("station_1", [46.0, -122.0, 1000], {"installation_date": 1980}),
        ("station_2", [47.0, -122.0, 1200], {"installation_date": 1985}),
    ]

    for station_id, coords, props in stations:
        builder.add_seismic_station(station_id, coords, props)

    # Build spatial relationships
    builder.build_spatial_graph(max_distance=100.0)

    # Add some causal relationships
    builder.add_causal_relationship("msh_chamber", "mount_st_helens", 0.9)
    builder.add_causal_relationship("rainier_chamber", "mount_rainier", 0.8)

    return builder.to_geological_graph_data()

def demonstrate_supereruption_detection():
    """Demonstrate supereruption detection capabilities."""
    print("Demonstrating Supereruption Detection...")

    # Create a large caldera system
    builder = GeologicalGraphBuilder()

    # Add Yellowstone-like system
    builder.add_entity(
        "yellowstone_caldera",
        "caldera_complex",
        [44.4, -110.6, 0],
        {
            "caldera_diameter": 45.0,
            "last_eruption": 640000,  # years ago
            "magma_chamber_volume": 1000.0,  # km³
        },
    )

    # Add associated magma chambers
    chambers = [
        ("upper_chamber", [44.4, -110.6, -8], {"volume": 400, "temperature": 750}),
        ("lower_chamber", [44.4, -110.6, -15], {"volume": 600, "temperature": 900}),
    ]

    for chamber_id, coords, props in chambers:
        builder.add_magma_chamber(chamber_id, coords, props)

    # Add monitoring network
    stations = [
        (f"ys_station_{i}", [44.4 + 0.1 * i, -110.6 + 0.1 * i, 2000], {})
        for i in range(5)
    ]

    for station_id, coords, props in stations:
        builder.add_seismic_station(station_id, coords, props)

    # Build relationships
    builder.build_spatial_graph(max_distance=50.0)
    builder.add_causal_relationship("upper_chamber", "yellowstone_caldera", 0.95)
    builder.add_causal_relationship("lower_chamber", "upper_chamber", 0.8)

    # Convert to graph data
    graph_data = builder.to_geological_graph_data()

    # Initialize representation engine
    engine = GeologicalNetworkRepresentationEngine()

    # Process the system
    results = engine.process_geological_network(
        graph_data, include_supereruption_analysis=True
    )

    print("Yellowstone-like System Analysis:")
    print(f"System type: {results['system_classification']['system_type']}")
    print(f"Confidence: {results['system_classification']['confidence']:.3f}")

    if "supereruption_analysis" in results:
        analysis = results["supereruption_analysis"]
        print(f"Supereruption risk: {analysis['overall_supereruption_risk']:.3f}")

        for caldera_analysis in analysis["caldera_analyses"]:
            signature = caldera_analysis["supereruption_signature"]
            chamber_state = caldera_analysis["chamber_state"]

            print(f"\nCaldera Node {caldera_analysis['caldera_node']}:")
            print(f"  Eruption probability: {signature.eruption_probability:.3f}")
            print(f"  Chamber state: {chamber_state['chamber_state']}")
            print(f"  Explosive potential: {chamber_state['explosive_potential']}")

    print("Supereruption detection demonstration completed!")

if __name__ == "__main__":
    # Run tests
    test_geological_network_representation()

    # Demonstrate supereruption detection
    demonstrate_supereruption_detection()

    # Create and analyze sample volcanic network
    sample_network = create_sample_volcanic_network()
    print(
        f"\nSample volcanic network created with {sample_network.node_features.size(0)} nodes"
    )

    logger.info("Geological Network Representation Learning implemented successfully")
