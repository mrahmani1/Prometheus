"""
Data models for geological processes and relationships in the graph database
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field


class ProcessType(str, Enum):
    """Types of geological processes"""

    VOLCANIC = "volcanic"
    SEISMIC = "seismic"
    TECTONIC = "tectonic"
    HYDROTHERMAL = "hydrothermal"
    MAGMATIC = "magmatic"
    METAMORPHIC = "metamorphic"
    SEDIMENTARY = "sedimentary"
    EROSIONAL = "erosional"
    DEFORMATIONAL = "deformational"
    GEOCHEMICAL = "geochemical"


class RelationshipType(str, Enum):
    """Types of relationships between geological processes"""

    CAUSAL = "causal"
    CORRELATIONAL = "correlational"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    INFLUENCES = "influences"
    TRIGGERS = "triggers"
    INHIBITS = "inhibits"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    COOCCURS = "cooccurs"


class GeologicalLocation(BaseModel):
    """Geological location information"""

    location_id: str
    name: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    elevation: Optional[float] = None
    depth: Optional[float] = None
    region: Optional[str] = None
    tectonic_setting: Optional[str] = None
    coordinate_system: str = "WGS84"


class GeologicalProcess(BaseModel):
    """Represents a geological process in the graph database"""

    process_id: str
    name: str
    process_type: ProcessType
    description: Optional[str] = None
    location_id: str

    # Physical parameters
    depth_range: Optional[List[float]] = None  # [min_depth, max_depth] in km
    temperature_range: Optional[List[float]] = None  # [min_temp, max_temp] in Celsius
    pressure_range: Optional[List[float]] = None  # [min_pressure, max_pressure] in MPa

    # Temporal characteristics
    timescale: Optional[str] = None  # e.g., "seconds", "years", "millions_of_years"
    duration: Optional[float] = None  # Duration in appropriate units
    frequency: Optional[float] = None  # Frequency of occurrence

    # Process-specific attributes
    magnitude: Optional[float] = None  # For seismic events, eruption magnitude, etc.
    intensity: Optional[float] = None  # Process intensity measure
    volume: Optional[float] = None  # Volume of material involved (m³)

    # Metadata
    confidence_level: float = Field(default=1.0, ge=0.0, le=1.0)
    data_sources: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


class GeologicalRelationship(BaseModel):
    """Represents a relationship between geological processes"""

    relationship_id: str
    source_process_id: str
    target_process_id: str
    relationship_type: RelationshipType

    # Relationship strength and confidence
    strength: float = Field(..., ge=0.0, le=1.0)  # Strength of the relationship
    confidence: float = Field(..., ge=0.0, le=1.0)  # Confidence in the relationship

    # Relationship characteristics
    mechanism: Optional[str] = None  # Physical mechanism of the relationship
    time_delay: Optional[float] = None  # Time delay between cause and effect
    spatial_scale: Optional[float] = None  # Spatial scale of influence (km)

    # Evidence and validation
    evidence_type: Optional[str] = None  # Type of evidence supporting the relationship
    statistical_significance: Optional[float] = None
    peer_reviewed: bool = False

    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


class CausalRelationship(BaseModel):
    """Represents a causal chain of geological processes"""

    chain_id: str
    process_chain: List[Dict[str, Any]]  # Ordered list of processes in the chain
    relationships: List[Dict[str, Any]]  # Relationships connecting the processes

    # Chain characteristics
    total_strength: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    chain_length: int = Field(..., ge=1)

    # Temporal characteristics
    total_time_delay: Optional[float] = None
    time_units: Optional[str] = None

    # Validation
    validated: bool = False
    validation_method: Optional[str] = None


class ProcessNetwork(BaseModel):
    """Represents a network of interconnected geological processes"""

    network_id: str
    name: str
    description: Optional[str] = None

    # Network components
    process_ids: List[str]
    relationship_ids: List[str]

    # Network properties
    node_count: int
    edge_count: int
    density: float = Field(..., ge=0.0, le=1.0)

    # Spatial and temporal bounds
    spatial_bounds: Optional[Dict[str, float]] = None  # lat/lon bounds
    temporal_bounds: Optional[Dict[str, datetime]] = None  # start/end times

    # Network analysis results
    centrality_measures: Optional[Dict[str, Dict[str, float]]] = None
    community_structure: Optional[Dict[str, List[str]]] = None

    # Metadata
    created_at: Optional[datetime] = None
    analysis_timestamp: Optional[datetime] = None


class NetworkAnalysisResult(BaseModel):
    """Results of network analysis on geological processes"""

    network_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

    # Basic network metrics
    node_count: int
    edge_count: int
    density: float

    # Centrality measures
    centrality_measures: Dict[str, Dict[str, float]]

    # Community detection results
    communities: Optional[List[List[str]]] = None
    modularity: Optional[float] = None

    # Path analysis
    shortest_paths: Optional[Dict[str, Dict[str, int]]] = None
    diameter: Optional[int] = None

    # Clustering metrics
    clustering_coefficient: Optional[float] = None
    transitivity: Optional[float] = None

    # Analysis metadata
    analysis_timestamp: datetime
    analysis_parameters: Optional[Dict[str, Any]] = None


class GraphQuery(BaseModel):
    """Represents a graph query for geological relationships"""

    query_id: str
    query_type: str  # e.g., "causal_chain", "network_analysis", "pattern_search"

    # Query parameters
    start_processes: Optional[List[str]] = None
    end_processes: Optional[List[str]] = None
    process_types: Optional[List[ProcessType]] = None
    relationship_types: Optional[List[RelationshipType]] = None

    # Filtering criteria
    min_strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_depth: Optional[int] = Field(None, ge=1, le=10)

    # Spatial and temporal filters
    spatial_bounds: Optional[Dict[str, float]] = None
    temporal_bounds: Optional[Dict[str, datetime]] = None

    # Result limits
    max_results: int = Field(default=100, ge=1, le=10000)

    class Config:
        use_enum_values = True


class GraphVisualizationConfig(BaseModel):
    """Configuration for graph visualization"""

    visualization_id: str
    layout_algorithm: str = "force_directed"  # force_directed, hierarchical, circular

    # Visual properties
    node_size_property: Optional[str] = None  # Property to determine node size
    node_color_property: Optional[str] = None  # Property to determine node color
    edge_width_property: Optional[str] = None  # Property to determine edge width

    # Filtering for visualization
    show_node_labels: bool = True
    show_edge_labels: bool = False
    min_edge_strength: float = Field(default=0.1, ge=0.0, le=1.0)

    # Layout parameters
    layout_parameters: Optional[Dict[str, Any]] = None

    # Export settings
    export_format: str = "json"  # json, graphml, gexf, cytoscape
    include_metadata: bool = True


class GeologicalEvent(BaseModel):
    """Represents a specific geological event"""

    event_id: str
    event_type: str
    timestamp: datetime
    location_id: str

    # Event characteristics
    magnitude: Optional[float] = None
    duration: Optional[float] = None
    depth: Optional[float] = None

    # Associated processes
    related_processes: Optional[List[str]] = None
    triggered_processes: Optional[List[str]] = None

    # Data and validation
    data_sources: Optional[List[str]] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    validated: bool = False


class ProcessEvolution(BaseModel):
    """Represents the evolution of a geological process over time"""

    evolution_id: str
    process_id: str

    # Time series data
    timestamps: List[datetime]
    parameter_values: Dict[str, List[float]]  # parameter_name -> time series values

    # Evolution characteristics
    trend: Optional[str] = None  # increasing, decreasing, stable, cyclic
    change_rate: Optional[float] = None

    # Significant events in evolution
    change_points: Optional[List[Dict[str, Any]]] = None
    anomalies: Optional[List[Dict[str, Any]]] = None

    # Analysis metadata
    analysis_method: Optional[str] = None
    analysis_timestamp: Optional[datetime] = None
