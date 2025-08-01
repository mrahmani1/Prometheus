"""
API endpoints for geological network representation learning.

Provides REST API access to node embeddings, edge embeddings, graph-level embeddings,
similarity search, and supereruption analysis capabilities.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.geological_network_representation import (
    GeologicalGraphData,
    GeologicalNetworkRepresentationEngine,
    SupereruptionAnalyzer,
    SupereruptionSignature,
)

logger = logging.getLogger(__name__)

# Initialize the router
router = APIRouter(
    prefix="/api/v1/geological-representation",
    tags=["Geological Network Representation"],
)

# Global engine instance (in production, this would be managed differently)
_engine = None
_analyzer = None

def get_engine() -> GeologicalNetworkRepresentationEngine:
    """Get or create the geological network representation engine."""
    global _engine
    if _engine is None:
        _engine = GeologicalNetworkRepresentationEngine(
            node_embedding_dim=256, edge_embedding_dim=128, graph_embedding_dim=512
        )
    return _engine

def get_analyzer() -> SupereruptionAnalyzer:
    """Get or create the supereruption analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SupereruptionAnalyzer()
    return _analyzer

# Pydantic models for API requests and responses

class GeologicalNodeData(BaseModel):
    """Input data for geological nodes."""

    node_ids: List[int]
    entity_types: List[int]
    process_types: List[int]
    spatial_coords: List[List[float]]  # [[lat, lon, depth], ...]
    temporal_info: Optional[List[float]] = None
    geological_properties: Optional[List[List[float]]] = None

class GeologicalEdgeData(BaseModel):
    """Input data for geological edges."""

    edge_types: List[int]
    edge_distances: Optional[List[float]] = None
    edge_strengths: Optional[List[float]] = None
    temporal_evolution: Optional[List[float]] = None

class GeologicalGraphInput(BaseModel):
    """Input data for a complete geological graph."""

    node_features: List[List[float]]
    edge_index: List[List[int]]  # [[source_nodes], [target_nodes]]
    node_types: Optional[List[int]] = None
    edge_types: Optional[List[int]] = None
    spatial_coords: Optional[List[List[float]]] = None
    temporal_info: Optional[List[float]] = None

class CalderaSystemData(BaseModel):
    """Input data for caldera system analysis."""

    system_id: str
    magma_chamber_volume: float = Field(..., description="Magma chamber volume in km³")
    caldera_diameter: float = Field(..., description="Caldera diameter in km")
    location: Optional[List[float]] = Field(None, description="[latitude, longitude]")

class MonitoringData(BaseModel):
    """Current monitoring data for supereruption analysis."""

    deformation_rate: float = Field(
        ..., description="Crustal deformation rate in mm/year"
    )
    seismic_swarm_intensity: float = Field(..., description="Seismic events per day")
    gas_emission_anomaly: float = Field(
        ..., description="Gas emission relative to baseline"
    )
    thermal_anomaly_extent: float = Field(
        ..., description="Thermal anomaly extent in km²"
    )
    ground_uplift_pattern: Optional[List[float]] = Field(
        None, description="Ground uplift measurements"
    )
    geochemical_indicators: Optional[Dict[str, float]] = Field(
        None, description="Geochemical measurements"
    )

class SimilaritySearchRequest(BaseModel):
    """Request for similarity search."""

    query_embedding: List[float]
    k: int = Field(10, description="Number of similar systems to return")
    system_type_filter: Optional[str] = None
    min_similarity: float = Field(0.5, description="Minimum similarity threshold")

class SystemMetadata(BaseModel):
    """Metadata for geological systems."""

    system_type: str
    location: Optional[List[float]] = None
    last_eruption: Optional[int] = None
    max_vei: Optional[int] = None
    magma_chamber_volume: Optional[float] = None
    caldera_diameter: Optional[float] = None
    eruption_frequency: Optional[float] = None

class AddSystemRequest(BaseModel):
    """Request to add a system to the similarity database."""

    system_id: str
    embedding: List[float]
    metadata: SystemMetadata

# Response models

class NodeEmbeddingResponse(BaseModel):
    """Response for node embedding generation."""

    node_embeddings: List[List[float]]
    supereruption_scores: List[float]
    magma_chamber_states: List[List[float]]
    entity_embeddings: List[List[float]]
    process_embeddings: List[List[float]]
    spatial_embeddings: List[List[float]]

class EdgeEmbeddingResponse(BaseModel):
    """Response for edge embedding generation."""

    edge_embeddings: List[List[float]]
    causal_scores: List[float]
    supereruption_trigger_scores: List[float]
    type_embeddings: List[List[float]]

class GraphEmbeddingResponse(BaseModel):
    """Response for graph-level embedding generation."""

    graph_embedding: List[float]
    system_type: str
    confidence: float
    all_probabilities: Dict[str, float]
    supereruption_system_score: float
    caldera_characteristics: List[float]

class SupereruptionAnalysisResponse(BaseModel):
    """Response for supereruption analysis."""

    caldera_id: str
    magma_chamber_volume: float
    eruption_probability: float
    crustal_deformation_rate: float
    seismic_swarm_intensity: float
    gas_emission_anomaly: float
    thermal_anomaly_extent: float
    ground_uplift_pattern: List[float]
    geochemical_indicators: Dict[str, float]
    time_to_eruption_estimate: Optional[float]

class HistoricalComparison(BaseModel):
    """Historical supereruption comparison."""

    historical_eruption: str
    similarity: float
    volume_comparison: Dict[str, float]
    implications: List[str]

class MagmaChamberState(BaseModel):
    """Magma chamber state analysis."""

    chamber_state: str
    magma_mobility: str
    connectivity_score: float
    explosive_potential: str
    temperature: float
    pressure: float
    crystallinity: float
    volatile_content: Dict[str, float]
    eruption_style_prediction: str

class SimilarSystem(BaseModel):
    """Similar geological system."""

    system_id: str
    similarity: float
    metadata: SystemMetadata

class NetworkAnalysisResponse(BaseModel):
    """Complete network analysis response."""

    node_embeddings: List[List[float]]
    edge_embeddings: List[List[float]]
    graph_embedding: List[float]
    system_classification: Dict[str, Any]
    supereruption_analysis: Optional[Dict[str, Any]] = None

# API Endpoints

@router.post("/node-embeddings", response_model=NodeEmbeddingResponse)
async def generate_node_embeddings(
    node_data: GeologicalNodeData,
    engine: GeologicalNetworkRepresentationEngine = Depends(get_engine),
):
    """
    Generate node embeddings for geological entities and processes.

    This endpoint creates rich representations of geological entities that capture
    their properties, spatial relationships, temporal evolution, and process dynamics.
    """
    try:
        # Convert input data to tensors
        node_ids = torch.tensor(node_data.node_ids, dtype=torch.long)
        entity_types = torch.tensor(node_data.entity_types, dtype=torch.long)
        process_types = torch.tensor(node_data.process_types, dtype=torch.long)
        spatial_coords = torch.tensor(node_data.spatial_coords, dtype=torch.float32)

        temporal_info = None
        if node_data.temporal_info:
            temporal_info = torch.tensor(node_data.temporal_info, dtype=torch.float32)

        geological_properties = None
        if node_data.geological_properties:
            geological_properties = torch.tensor(
                node_data.geological_properties, dtype=torch.float32
            )

        # Generate embeddings
        results = engine.node_embedder(
            node_ids=node_ids,
            entity_types=entity_types,
            process_types=process_types,
            spatial_coords=spatial_coords,
            temporal_info=temporal_info,
            geological_properties=geological_properties,
        )

        # Convert results to lists for JSON serialization
        return NodeEmbeddingResponse(
            node_embeddings=results["node_embeddings"].detach().cpu().numpy().tolist(),
            supereruption_scores=results["supereruption_scores"]
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .tolist(),
            magma_chamber_states=results["magma_chamber_states"]
            .detach()
            .cpu()
            .numpy()
            .tolist(),
            entity_embeddings=results["entity_embeddings"]
            .detach()
            .cpu()
            .numpy()
            .tolist(),
            process_embeddings=results["process_embeddings"]
            .detach()
            .cpu()
            .numpy()
            .tolist(),
            spatial_embeddings=results["spatial_embeddings"]
            .detach()
            .cpu()
            .numpy()
            .tolist(),
        )

    except Exception as e:
        logger.error(f"Error generating node embeddings: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate node embeddings: {str(e)}"
        )

@router.post("/edge-embeddings", response_model=EdgeEmbeddingResponse)
async def generate_edge_embeddings(
    edge_data: GeologicalEdgeData,
    engine: GeologicalNetworkRepresentationEngine = Depends(get_engine),
):
    """
    Generate edge embeddings for geological relationships.

    This endpoint creates representations that capture the strength, type, and
    dynamics of relationships between geological entities and processes.
    """
    try:
        # Convert input data to tensors
        edge_types = torch.tensor(edge_data.edge_types, dtype=torch.long)

        edge_distances = None
        if edge_data.edge_distances:
            edge_distances = torch.tensor(edge_data.edge_distances, dtype=torch.float32)

        edge_strengths = None
        if edge_data.edge_strengths:
            edge_strengths = torch.tensor(edge_data.edge_strengths, dtype=torch.float32)

        temporal_evolution = None
        if edge_data.temporal_evolution:
            temporal_evolution = torch.tensor(
                edge_data.temporal_evolution, dtype=torch.float32
            )

        # Generate embeddings
        results = engine.edge_embedder(
            edge_types=edge_types,
            edge_distances=edge_distances,
            edge_strengths=edge_strengths,
            temporal_evolution=temporal_evolution,
        )

        return EdgeEmbeddingResponse(
            edge_embeddings=results["edge_embeddings"].detach().cpu().numpy().tolist(),
            causal_scores=results["causal_scores"]
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .tolist(),
            supereruption_trigger_scores=results["supereruption_trigger_scores"]
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .tolist(),
            type_embeddings=results["type_embeddings"].detach().cpu().numpy().tolist(),
        )

    except Exception as e:
        logger.error(f"Error generating edge embeddings: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate edge embeddings: {str(e)}"
        )

@router.post("/graph-embeddings", response_model=GraphEmbeddingResponse)
async def generate_graph_embeddings(
    graph_input: GeologicalGraphInput,
    include_supereruption_analysis: bool = True,
    engine: GeologicalNetworkRepresentationEngine = Depends(get_engine),
):
    """
    Generate graph-level embeddings for geological system classification.

    This endpoint creates holistic representations of entire geological systems
    for classification, comparison, and similarity analysis.
    """
    try:
        # Convert input to GeologicalGraphData
        node_features = torch.tensor(graph_input.node_features, dtype=torch.float32)
        edge_index = torch.tensor(graph_input.edge_index, dtype=torch.long)

        node_types = None
        if graph_input.node_types:
            node_types = torch.tensor(graph_input.node_types, dtype=torch.long)

        edge_types = None
        if graph_input.edge_types:
            edge_types = torch.tensor(graph_input.edge_types, dtype=torch.long)

        spatial_coords = None
        if graph_input.spatial_coords:
            spatial_coords = torch.tensor(
                graph_input.spatial_coords, dtype=torch.float32
            )

        temporal_info = None
        if graph_input.temporal_info:
            temporal_info = torch.tensor(graph_input.temporal_info, dtype=torch.float32)

        graph_data = GeologicalGraphData(
            node_features=node_features,
            edge_index=edge_index,
            node_types=node_types,
            edge_types=edge_types,
            spatial_coords=spatial_coords,
            temporal_info=temporal_info,
        )

        # Process the geological network
        results = engine.process_geological_network(
            graph_data, include_supereruption_analysis=include_supereruption_analysis
        )

        return GraphEmbeddingResponse(
            graph_embedding=results["graph_embedding"]
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .tolist(),
            system_type=results["system_classification"]["system_type"],
            confidence=results["system_classification"]["confidence"],
            all_probabilities=results["system_classification"]["all_probabilities"],
            supereruption_system_score=float(
                results.get("supereruption_scores", torch.tensor([0.0])).mean().item()
            ),
            caldera_characteristics=results.get("magma_chamber_states", torch.zeros(8))
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy()
            .tolist(),
        )

    except Exception as e:
        logger.error(f"Error generating graph embeddings: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate graph embeddings: {str(e)}"
        )

@router.post("/analyze-supereruption", response_model=SupereruptionAnalysisResponse)
async def analyze_supereruption_potential(
    caldera_system: CalderaSystemData,
    monitoring_data: MonitoringData,
    analyzer: SupereruptionAnalyzer = Depends(get_analyzer),
):
    """
    Analyze the supereruption potential of a caldera system.

    This endpoint provides comprehensive analysis of supereruption risk based on
    current monitoring data and caldera system characteristics.
    """
    try:
        # Convert input data
        caldera_dict = {
            "system_id": caldera_system.system_id,
            "magma_chamber_volume": caldera_system.magma_chamber_volume,
            "caldera_diameter": caldera_system.caldera_diameter,
        }

        monitoring_dict = {
            "deformation_rate": monitoring_data.deformation_rate,
            "seismic_swarm_intensity": monitoring_data.seismic_swarm_intensity,
            "gas_emission_anomaly": monitoring_data.gas_emission_anomaly,
            "thermal_anomaly_extent": monitoring_data.thermal_anomaly_extent,
            "ground_uplift_pattern": monitoring_data.ground_uplift_pattern or [],
            "geochemical_indicators": monitoring_data.geochemical_indicators or {},
        }

        # Analyze supereruption potential
        signature = analyzer.analyze_supereruption_potential(
            caldera_dict, monitoring_dict
        )

        return SupereruptionAnalysisResponse(
            caldera_id=signature.caldera_id,
            magma_chamber_volume=signature.magma_chamber_volume,
            eruption_probability=signature.eruption_probability,
            crustal_deformation_rate=signature.crustal_deformation_rate,
            seismic_swarm_intensity=signature.seismic_swarm_intensity,
            gas_emission_anomaly=signature.gas_emission_anomaly,
            thermal_anomaly_extent=signature.thermal_anomaly_extent,
            ground_uplift_pattern=signature.ground_uplift_pattern.tolist(),
            geochemical_indicators=signature.geochemical_indicators,
            time_to_eruption_estimate=signature.time_to_eruption_estimate,
        )

    except Exception as e:
        logger.error(f"Error analyzing supereruption potential: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze supereruption potential: {str(e)}",
        )

@router.post(
    "/compare-historical-supereruptions", response_model=List[HistoricalComparison]
)
async def compare_with_historical_supereruptions(
    supereruption_data: SupereruptionAnalysisResponse,
    analyzer: SupereruptionAnalyzer = Depends(get_analyzer),
):
    """
    Compare a current supereruption signature with historical supereruptions.

    This endpoint provides insights by comparing current conditions with
    well-documented historical supereruptions.
    """
    try:
        # Convert response back to SupereruptionSignature
        signature = SupereruptionSignature(
            caldera_id=supereruption_data.caldera_id,
            magma_chamber_volume=supereruption_data.magma_chamber_volume,
            crustal_deformation_rate=supereruption_data.crustal_deformation_rate,
            seismic_swarm_intensity=supereruption_data.seismic_swarm_intensity,
            gas_emission_anomaly=supereruption_data.gas_emission_anomaly,
            thermal_anomaly_extent=supereruption_data.thermal_anomaly_extent,
            ground_uplift_pattern=np.array(supereruption_data.ground_uplift_pattern),
            geochemical_indicators=supereruption_data.geochemical_indicators,
            eruption_probability=supereruption_data.eruption_probability,
            time_to_eruption_estimate=supereruption_data.time_to_eruption_estimate,
        )

        # Compare with historical supereruptions
        comparisons = analyzer.compare_with_historical_supereruptions(signature)

        return [
            HistoricalComparison(
                historical_eruption=comp["historical_eruption"],
                similarity=comp["similarity"],
                volume_comparison=comp["volume_comparison"],
                implications=comp["implications"],
            )
            for comp in comparisons
        ]

    except Exception as e:
        logger.error(f"Error comparing with historical supereruptions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare with historical supereruptions: {str(e)}",
        )

@router.post("/analyze-magma-chamber", response_model=MagmaChamberState)
async def analyze_magma_chamber_state(
    chamber_embedding: List[float],
    monitoring_data: Dict[str, Any],
    analyzer: SupereruptionAnalyzer = Depends(get_analyzer),
):
    """
    Analyze the current state of a magma chamber.

    This endpoint provides detailed analysis of magma chamber conditions
    and eruption style predictions.
    """
    try:
        # Convert embedding to tensor
        embedding_tensor = torch.tensor(chamber_embedding, dtype=torch.float32)

        # Analyze chamber state
        chamber_state = analyzer.identify_magma_chamber_state(
            embedding_tensor, monitoring_data
        )

        return MagmaChamberState(**chamber_state)

    except Exception as e:
        logger.error(f"Error analyzing magma chamber state: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze magma chamber state: {str(e)}"
        )

@router.post("/add-system")
async def add_geological_system(
    request: AddSystemRequest,
    engine: GeologicalNetworkRepresentationEngine = Depends(get_engine),
):
    """
    Add a geological system to the similarity search database.

    This endpoint allows adding new geological systems for similarity comparison.
    """
    try:
        # Convert embedding to numpy array
        embedding = np.array(request.embedding, dtype=np.float32)

        # Convert metadata
        metadata = request.metadata.dict()

        # Add to similarity search database
        engine.similarity_search.add_geological_system(
            request.system_id, embedding, metadata
        )

        return {"message": f"Successfully added system {request.system_id} to database"}

    except Exception as e:
        logger.error(f"Error adding geological system: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to add geological system: {str(e)}"
        )

@router.post("/find-similar-systems", response_model=List[SimilarSystem])
async def find_similar_systems(
    request: SimilaritySearchRequest,
    engine: GeologicalNetworkRepresentationEngine = Depends(get_engine),
):
    """
    Find geological systems similar to the query system.

    This endpoint uses advanced embedding techniques to identify analogous
    geological systems for comparative analysis.
    """
    try:
        # Convert query embedding to numpy array
        query_embedding = np.array(request.query_embedding, dtype=np.float32)

        # Find similar systems
        similar_systems = engine.similarity_search.find_similar_systems(
            query_embedding,
            k=request.k,
            system_type_filter=request.system_type_filter,
            min_similarity=request.min_similarity,
        )

        return [
            SimilarSystem(
                system_id=system["system_id"],
                similarity=float(system["similarity"]),
                metadata=SystemMetadata(**system["metadata"]),
            )
            for system in similar_systems
        ]

    except Exception as e:
        logger.error(f"Error finding similar systems: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to find similar systems: {str(e)}"
        )

@router.post("/analyze-network", response_model=NetworkAnalysisResponse)
async def analyze_complete_network(
    graph_input: GeologicalGraphInput,
    include_supereruption_analysis: bool = True,
    engine: GeologicalNetworkRepresentationEngine = Depends(get_engine),
):
    """
    Perform complete geological network analysis.

    This endpoint provides comprehensive analysis including node embeddings,
    edge embeddings, graph-level embeddings, system classification, and
    optional supereruption analysis.
    """
    try:
        # Convert input to GeologicalGraphData (same as graph-embeddings endpoint)
        node_features = torch.tensor(graph_input.node_features, dtype=torch.float32)
        edge_index = torch.tensor(graph_input.edge_index, dtype=torch.long)

        node_types = None
        if graph_input.node_types:
            node_types = torch.tensor(graph_input.node_types, dtype=torch.long)

        edge_types = None
        if graph_input.edge_types:
            edge_types = torch.tensor(graph_input.edge_types, dtype=torch.long)

        spatial_coords = None
        if graph_input.spatial_coords:
            spatial_coords = torch.tensor(
                graph_input.spatial_coords, dtype=torch.float32
            )

        temporal_info = None
        if graph_input.temporal_info:
            temporal_info = torch.tensor(graph_input.temporal_info, dtype=torch.float32)

        graph_data = GeologicalGraphData(
            node_features=node_features,
            edge_index=edge_index,
            node_types=node_types,
            edge_types=edge_types,
            spatial_coords=spatial_coords,
            temporal_info=temporal_info,
        )

        # Process the complete geological network
        results = engine.process_geological_network(
            graph_data, include_supereruption_analysis=include_supereruption_analysis
        )

        return NetworkAnalysisResponse(
            node_embeddings=results["node_embeddings"].detach().cpu().numpy().tolist(),
            edge_embeddings=results["edge_embeddings"].detach().cpu().numpy().tolist(),
            graph_embedding=results["graph_embedding"]
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .tolist(),
            system_classification=results["system_classification"],
            supereruption_analysis=results.get("supereruption_analysis"),
        )

    except Exception as e:
        logger.error(f"Error analyzing complete network: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze complete network: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "geological-network-representation",
        "timestamp": datetime.utcnow().isoformat(),
    }
