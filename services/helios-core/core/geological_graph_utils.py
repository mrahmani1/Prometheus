"""
Utilities for geological graph construction and processing.

This module provides tools for converting geological data into graph representations
and preprocessing functions for graph neural networks.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

from .graph_neural_networks import GeologicalGraphData

logger = logging.getLogger(__name__)

@dataclass
class GeologicalEntity:
    """Represents a geological entity in the graph."""

    entity_id: str
    entity_type: str  # volcano, fault, seismic_station, etc.
    coordinates: Tuple[float, float, float]  # lat, lon, depth
    properties: Dict[str, float]
    temporal_data: Optional[Dict[str, List[float]]] = None

@dataclass
class GeologicalRelationship:
    """Represents a relationship between geological entities."""

    source_id: str
    target_id: str
    relationship_type: str  # spatial, causal, temporal, etc.
    strength: float
    properties: Dict[str, float]
    temporal_evolution: Optional[List[float]] = None

class GeologicalGraphBuilder:
    """
    Builder class for constructing geological graphs from various data sources.
    """

    def __init__(self):
        self.entities = {}
        self.relationships = []
        self.feature_scalers = {}

    def add_volcanic_system(
        self,
        volcano_id: str,
        coordinates: Tuple[float, float, float],
        properties: Dict[str, float],
    ):
        """Add a volcanic system to the graph."""
        entity = GeologicalEntity(
            entity_id=volcano_id,
            entity_type="volcano",
            coordinates=coordinates,
            properties=properties,
        )
        self.entities[volcano_id] = entity

    def add_seismic_station(
        self,
        station_id: str,
        coordinates: Tuple[float, float, float],
        properties: Dict[str, float],
    ):
        """Add a seismic monitoring station to the graph."""
        entity = GeologicalEntity(
            entity_id=station_id,
            entity_type="seismic_station",
            coordinates=coordinates,
            properties=properties,
        )
        self.entities[station_id] = entity

    def add_fault_system(
        self,
        fault_id: str,
        coordinates: Tuple[float, float, float],
        properties: Dict[str, float],
    ):
        """Add a fault system to the graph."""
        entity = GeologicalEntity(
            entity_id=fault_id,
            entity_type="fault",
            coordinates=coordinates,
            properties=properties,
        )
        self.entities[fault_id] = entity

    def add_magma_chamber(
        self,
        chamber_id: str,
        coordinates: Tuple[float, float, float],
        properties: Dict[str, float],
    ):
        """Add a magma chamber to the graph."""
        entity = GeologicalEntity(
            entity_id=chamber_id,
            entity_type="magma_chamber",
            coordinates=coordinates,
            properties=properties,
        )
        self.entities[chamber_id] = entity

    def add_spatial_relationship(
        self,
        source_id: str,
        target_id: str,
        distance: float,
        properties: Optional[Dict[str, float]] = None,
    ):
        """Add a spatial relationship between entities."""
        if properties is None:
            properties = {}

        relationship = GeologicalRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type="spatial",
            strength=1.0 / (1.0 + distance),  # Inverse distance weighting
            properties={"distance": distance, **properties},
        )
        self.relationships.append(relationship)

    def add_causal_relationship(
        self,
        source_id: str,
        target_id: str,
        causal_strength: float,
        properties: Optional[Dict[str, float]] = None,
    ):
        """Add a causal relationship between entities."""
        if properties is None:
            properties = {}

        relationship = GeologicalRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type="causal",
            strength=causal_strength,
            properties=properties,
        )
        self.relationships.append(relationship)

    def add_temporal_relationship(
        self,
        source_id: str,
        target_id: str,
        temporal_lag: float,
        correlation: float,
        properties: Optional[Dict[str, float]] = None,
    ):
        """Add a temporal relationship between entities."""
        if properties is None:
            properties = {}

        relationship = GeologicalRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type="temporal",
            strength=abs(correlation),
            properties={
                "temporal_lag": temporal_lag,
                "correlation": correlation,
                **properties,
            },
        )
        self.relationships.append(relationship)

    def build_spatial_graph(
        self, max_distance: float = 100.0
    ) -> "GeologicalGraphBuilder":
        """Build spatial relationships based on geographic proximity."""
        entity_list = list(self.entities.values())

        for i, entity1 in enumerate(entity_list):
            for j, entity2 in enumerate(entity_list[i + 1 :], i + 1):
                # Calculate distance
                coord1 = np.array(entity1.coordinates)
                coord2 = np.array(entity2.coordinates)
                distance = np.linalg.norm(coord1 - coord2)

                if distance <= max_distance:
                    self.add_spatial_relationship(
                        entity1.entity_id, entity2.entity_id, distance
                    )

        return self

    def build_from_seismic_data(
        self,
        seismic_events: List[Dict],
        stations: List[Dict],
        correlation_threshold: float = 0.5,
    ) -> "GeologicalGraphBuilder":
        """Build graph from seismic monitoring data."""
        # Add seismic stations
        for station in stations:
            self.add_seismic_station(
                station["station_id"],
                (station["latitude"], station["longitude"], station["elevation"]),
                station.get("properties", {}),
            )

        # Add seismic events as entities
        for event in seismic_events:
            event_id = f"event_{event['event_id']}"
            self.add_entity(
                event_id,
                "seismic_event",
                (event["latitude"], event["longitude"], event["depth"]),
                {"magnitude": event["magnitude"], "timestamp": event["timestamp"]},
            )

        # Build spatial relationships
        self.build_spatial_graph(max_distance=50.0)

        return self

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        coordinates: Tuple[float, float, float],
        properties: Dict[str, float],
    ):
        """Generic method to add any geological entity."""
        entity = GeologicalEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            coordinates=coordinates,
            properties=properties,
        )
        self.entities[entity_id] = entity

    def to_geological_graph_data(self) -> GeologicalGraphData:
        """Convert the built graph to GeologicalGraphData format."""
        # Create entity ID to index mapping
        entity_ids = list(self.entities.keys())
        id_to_idx = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}

        # Extract node features
        node_features = []
        node_types = []
        spatial_coords = []

        # Define entity type mapping
        entity_type_map = {
            "volcano": 0,
            "seismic_station": 1,
            "fault": 2,
            "magma_chamber": 3,
            "seismic_event": 4,
            "hydrothermal_system": 5,
            "gas_emission_site": 6,
            "deformation_source": 7,
            "thermal_anomaly": 8,
            "unknown": 9,
        }

        # Define relationship type mapping
        relationship_type_map = {
            "spatial": 0,
            "causal": 1,
            "temporal": 2,
            "structural": 3,
            "chemical": 4,
        }

        for entity_id in entity_ids:
            entity = self.entities[entity_id]

            # Extract features from properties
            feature_vector = []
            for key in sorted(entity.properties.keys()):
                feature_vector.append(entity.properties[key])

            # Add coordinate features
            feature_vector.extend(entity.coordinates)

            # Pad or truncate to fixed size
            target_size = 64  # Fixed feature size
            if len(feature_vector) < target_size:
                feature_vector.extend([0.0] * (target_size - len(feature_vector)))
            else:
                feature_vector = feature_vector[:target_size]

            node_features.append(feature_vector)
            node_types.append(entity_type_map.get(entity.entity_type, 9))
            spatial_coords.append(entity.coordinates)

        # Extract edge information
        edge_index = []
        edge_types = []
        edge_features = []

        for relationship in self.relationships:
            source_idx = id_to_idx.get(relationship.source_id)
            target_idx = id_to_idx.get(relationship.target_id)

            if source_idx is not None and target_idx is not None:
                # Add both directions for undirected relationships
                edge_index.extend([[source_idx, target_idx], [target_idx, source_idx]])

                rel_type = relationship_type_map.get(relationship.relationship_type, 4)
                edge_types.extend([rel_type, rel_type])

                # Edge features
                edge_feature = [relationship.strength]
                for key in sorted(relationship.properties.keys()):
                    edge_feature.append(relationship.properties[key])

                # Pad edge features
                edge_target_size = 16
                if len(edge_feature) < edge_target_size:
                    edge_feature.extend([0.0] * (edge_target_size - len(edge_feature)))
                else:
                    edge_feature = edge_feature[:edge_target_size]

                edge_features.extend([edge_feature, edge_feature])

        # Convert to tensors
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features_tensor = (
            torch.tensor(edge_features, dtype=torch.float32) if edge_features else None
        )
        node_types_tensor = torch.tensor(node_types, dtype=torch.long)
        edge_types_tensor = (
            torch.tensor(edge_types, dtype=torch.long) if edge_types else None
        )
        spatial_coords_tensor = torch.tensor(spatial_coords, dtype=torch.float32)

        return GeologicalGraphData(
            node_features=node_features_tensor,
            edge_index=edge_index_tensor,
            edge_features=edge_features_tensor,
            node_types=node_types_tensor,
            edge_types=edge_types_tensor,
            spatial_coords=spatial_coords_tensor,
        )

class GeologicalGraphPreprocessor:
    """
    Preprocessor for geological graph data.
    """

    def __init__(self):
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.fitted = False

    def fit(self, graph_data: GeologicalGraphData):
        """Fit the preprocessor on graph data."""
        # Fit node feature scaler
        self.node_scaler.fit(graph_data.node_features.numpy())

        # Fit edge feature scaler if edge features exist
        if graph_data.edge_features is not None:
            self.edge_scaler.fit(graph_data.edge_features.numpy())

        self.fitted = True

    def transform(self, graph_data: GeologicalGraphData) -> GeologicalGraphData:
        """Transform graph data using fitted scalers."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transformation")

        # Scale node features
        scaled_node_features = self.node_scaler.transform(
            graph_data.node_features.numpy()
        )
        scaled_node_features = torch.tensor(scaled_node_features, dtype=torch.float32)

        # Scale edge features if they exist
        scaled_edge_features = graph_data.edge_features
        if graph_data.edge_features is not None:
            scaled_edge_features = self.edge_scaler.transform(
                graph_data.edge_features.numpy()
            )
            scaled_edge_features = torch.tensor(
                scaled_edge_features, dtype=torch.float32
            )

        return GeologicalGraphData(
            node_features=scaled_node_features,
            edge_index=graph_data.edge_index,
            edge_features=scaled_edge_features,
            node_types=graph_data.node_types,
            edge_types=graph_data.edge_types,
            spatial_coords=graph_data.spatial_coords,
            temporal_info=graph_data.temporal_info,
            batch=graph_data.batch,
        )

    def fit_transform(self, graph_data: GeologicalGraphData) -> GeologicalGraphData:
        """Fit and transform graph data in one step."""
        self.fit(graph_data)
        return self.transform(graph_data)

class GeologicalGraphAnalyzer:
    """
    Analyzer for geological graph properties and statistics.
    """

    @staticmethod
    def analyze_graph_structure(graph_data: GeologicalGraphData) -> Dict[str, float]:
        """Analyze the structure of a geological graph."""
        # Convert to NetworkX for analysis
        edge_index = graph_data.edge_index.numpy()
        G = nx.Graph()

        # Add nodes
        num_nodes = graph_data.node_features.size(0)
        G.add_nodes_from(range(num_nodes))

        # Add edges
        edges = [
            (edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])
        ]
        G.add_edges_from(edges)

        # Calculate graph metrics
        metrics = {
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "density": nx.density(G),
            "average_clustering": nx.average_clustering(G),
            "num_connected_components": nx.number_connected_components(G),
        }

        # Calculate centrality measures for connected graphs
        if nx.is_connected(G):
            metrics["average_shortest_path_length"] = nx.average_shortest_path_length(G)
            metrics["diameter"] = nx.diameter(G)
        else:
            # For disconnected graphs, calculate for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            metrics["largest_component_size"] = len(largest_cc)
            metrics["largest_component_avg_path_length"] = (
                nx.average_shortest_path_length(subgraph)
            )

        return metrics

    @staticmethod
    def analyze_node_types(graph_data: GeologicalGraphData) -> Dict[str, int]:
        """Analyze the distribution of node types."""
        if graph_data.node_types is None:
            return {}

        node_types = graph_data.node_types.numpy()
        unique_types, counts = np.unique(node_types, return_counts=True)

        type_names = {
            0: "volcano",
            1: "seismic_station",
            2: "fault",
            3: "magma_chamber",
            4: "seismic_event",
            5: "hydrothermal_system",
            6: "gas_emission_site",
            7: "deformation_source",
            8: "thermal_anomaly",
            9: "unknown",
        }

        return {
            type_names.get(t, f"type_{t}"): count
            for t, count in zip(unique_types, counts)
        }

    @staticmethod
    def analyze_edge_types(graph_data: GeologicalGraphData) -> Dict[str, int]:
        """Analyze the distribution of edge types."""
        if graph_data.edge_types is None:
            return {}

        edge_types = graph_data.edge_types.numpy()
        unique_types, counts = np.unique(edge_types, return_counts=True)

        type_names = {
            0: "spatial",
            1: "causal",
            2: "temporal",
            3: "structural",
            4: "chemical",
        }

        return {
            type_names.get(t, f"type_{t}"): count
            for t, count in zip(unique_types, counts)
        }

    @staticmethod
    def calculate_spatial_statistics(
        graph_data: GeologicalGraphData,
    ) -> Dict[str, float]:
        """Calculate spatial statistics for the geological graph."""
        if graph_data.spatial_coords is None:
            return {}

        coords = graph_data.spatial_coords.numpy()

        # Calculate spatial extent
        lat_range = coords[:, 0].max() - coords[:, 0].min()
        lon_range = coords[:, 1].max() - coords[:, 1].min()
        depth_range = coords[:, 2].max() - coords[:, 2].min()

        # Calculate center point
        center_lat = coords[:, 0].mean()
        center_lon = coords[:, 1].mean()
        center_depth = coords[:, 2].mean()

        # Calculate average distances between nodes
        distances = pdist(coords)

        return {
            "latitude_range": lat_range,
            "longitude_range": lon_range,
            "depth_range": depth_range,
            "center_latitude": center_lat,
            "center_longitude": center_lon,
            "center_depth": center_depth,
            "mean_distance": distances.mean(),
            "std_distance": distances.std(),
            "min_distance": distances.min(),
            "max_distance": distances.max(),
        }

def create_volcanic_system_graph(
    volcanoes: List[Dict],
    seismic_events: List[Dict],
    monitoring_stations: List[Dict],
    max_spatial_distance: float = 50.0,
) -> GeologicalGraphData:
    """
    Create a comprehensive volcanic system graph.

    Args:
        volcanoes: List of volcano data
        seismic_events: List of seismic event data
        monitoring_stations: List of monitoring station data
        max_spatial_distance: Maximum distance for spatial relationships (km)

    Returns:
        GeologicalGraphData representing the volcanic system
    """
    builder = GeologicalGraphBuilder()

    # Add volcanoes
    for volcano in volcanoes:
        builder.add_volcanic_system(
            volcano["volcano_id"],
            (volcano["latitude"], volcano["longitude"], volcano.get("elevation", 0)),
            volcano.get("properties", {}),
        )

    # Add seismic events
    for event in seismic_events:
        builder.add_entity(
            f"event_{event['event_id']}",
            "seismic_event",
            (event["latitude"], event["longitude"], event["depth"]),
            {"magnitude": event["magnitude"], "timestamp": event.get("timestamp", 0)},
        )

    # Add monitoring stations
    for station in monitoring_stations:
        builder.add_seismic_station(
            station["station_id"],
            (station["latitude"], station["longitude"], station.get("elevation", 0)),
            station.get("properties", {}),
        )

    # Build spatial relationships
    builder.build_spatial_graph(max_distance=max_spatial_distance)

    return builder.to_geological_graph_data()

# Example usage and testing
def test_geological_graph_utils():
    """Test geological graph utilities."""
    # Create sample data
    volcanoes = [
        {
            "volcano_id": "volcano_1",
            "latitude": 45.0,
            "longitude": -122.0,
            "elevation": 2000,
            "properties": {"last_eruption": 1980, "vei": 5},
        },
        {
            "volcano_id": "volcano_2",
            "latitude": 45.1,
            "longitude": -122.1,
            "elevation": 1800,
            "properties": {"last_eruption": 1800, "vei": 3},
        },
    ]

    seismic_events = [
        {
            "event_id": 1,
            "latitude": 45.05,
            "longitude": -122.05,
            "depth": 5.0,
            "magnitude": 3.2,
            "timestamp": 1640995200,
        },
        {
            "event_id": 2,
            "latitude": 45.02,
            "longitude": -122.02,
            "depth": 8.0,
            "magnitude": 2.8,
            "timestamp": 1640995800,
        },
    ]

    monitoring_stations = [
        {
            "station_id": "station_1",
            "latitude": 45.0,
            "longitude": -122.0,
            "elevation": 1500,
            "properties": {"installation_date": 2000},
        }
    ]

    # Create graph
    graph_data = create_volcanic_system_graph(
        volcanoes, seismic_events, monitoring_stations
    )

    print(f"Graph created with {graph_data.node_features.size(0)} nodes")
    print(f"Graph has {graph_data.edge_index.size(1)} edges")

    # Analyze graph
    analyzer = GeologicalGraphAnalyzer()
    structure_metrics = analyzer.analyze_graph_structure(graph_data)
    node_type_dist = analyzer.analyze_node_types(graph_data)
    spatial_stats = analyzer.calculate_spatial_statistics(graph_data)

    print("Graph structure metrics:", structure_metrics)
    print("Node type distribution:", node_type_dist)
    print("Spatial statistics:", spatial_stats)

    return True

if __name__ == "__main__":
    test_geological_graph_utils()
    logger.info("Geological graph utilities implemented and tested successfully")
