"""
API endpoints for causal inference engine

Provides REST API endpoints for geological causal relationship discovery,
intervention analysis, and hypothesis generation.
"""

import io
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn.functional as F
from core.causal_inference_engine import (CausalAlgorithm, CausalGraph,
                                          CausalInferenceEngine)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/causal-inference", tags=["causal-inference"])

# Global engine instance
causal_engine = CausalInferenceEngine()


# Pydantic models for API
class CausalDiscoveryRequest(BaseModel):
    algorithm: str = Field(
        default="pc", description="Causal discovery algorithm (pc, fci, ges)"
    )
    confidence_threshold: float = Field(
        default=0.05, description="Statistical significance threshold"
    )
    data_source: str = Field(description="Source of geological data")
    variables: Optional[List[str]] = Field(
        default=None, description="Variables to include in analysis"
    )


class CausalDiscoveryResponse(BaseModel):
    graph_id: str
    nodes: List[str]
    edges: List[Dict[str, Any]]
    algorithm_used: str
    confidence_threshold: float
    creation_timestamp: str
    validation_metrics: Dict[str, float]


class InterventionRequest(BaseModel):
    graph_id: str = Field(description="ID of causal graph to use")
    intervention_variable: str = Field(description="Variable to intervene on")
    intervention_value: float = Field(
        description="Value to set intervention variable to"
    )
    target_variables: Optional[List[str]] = Field(
        default=None, description="Variables to analyze effects on"
    )


class InterventionResponse(BaseModel):
    intervention_variable: str
    intervention_value: float
    target_variables: List[str]
    causal_effects: Dict[str, float]
    confidence_intervals: Dict[str, List[float]]
    statistical_significance: Dict[str, float]
    geological_interpretation: str


class HypothesisGenerationRequest(BaseModel):
    graph_id: str = Field(description="ID of causal graph to use")
    confidence_threshold: float = Field(
        default=0.8, description="Minimum confidence for hypothesis generation"
    )


class HypothesisGenerationResponse(BaseModel):
    hypotheses: List[str]
    graph_summary: Dict[str, Any]
    generation_timestamp: str


class ValidationRequest(BaseModel):
    graph_id: str = Field(description="ID of causal graph to validate")
    validation_data_source: str = Field(description="Source of validation data")


class ValidationResponse(BaseModel):
    total_edges: int
    validated_edges: int
    failed_edges: int
    overall_accuracy: float
    domain_knowledge_consistency: float
    edge_validations: Dict[str, Dict[str, Any]]


# In-memory storage for discovered graphs (in production, use proper database)
discovered_graphs: Dict[str, CausalGraph] = {}
graph_data: Dict[str, pd.DataFrame] = {}


@router.post("/discover", response_model=CausalDiscoveryResponse)
async def discover_causal_relationships(
    request: CausalDiscoveryRequest,
    data_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Discover causal relationships in geological data

    Upload geological data and discover causal relationships using specified algorithm.
    Returns a causal graph with geological interpretations.
    """
    try:
        # Validate algorithm
        try:
            algorithm = CausalAlgorithm(request.algorithm.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {request.algorithm}. Use 'pc', 'fci', or 'ges'",
            )

        # Read uploaded data
        content = await data_file.read()

        try:
            if data_file.filename.endswith(".csv"):
                data = pd.read_csv(io.StringIO(content.decode("utf-8")))
            elif data_file.filename.endswith(".json"):
                data = pd.read_json(io.StringIO(content.decode("utf-8")))
            else:
                raise HTTPException(
                    status_code=400, detail="Unsupported file format. Use CSV or JSON."
                )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error reading data file: {str(e)}"
            )

        # Filter variables if specified
        if request.variables:
            missing_vars = [var for var in request.variables if var not in data.columns]
            if missing_vars:
                raise HTTPException(
                    status_code=400,
                    detail=f"Variables not found in data: {missing_vars}",
                )
            data = data[request.variables]

        # Validate data
        if len(data) < 10:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for causal discovery (minimum 10 samples required)",
            )

        if data.isnull().sum().sum() > len(data) * len(data.columns) * 0.5:
            raise HTTPException(
                status_code=400, detail="Too many missing values in data (>50%)"
            )

        logger.info(
            f"Starting causal discovery with {algorithm.value} algorithm on {len(data)} samples"
        )

        # Discover causal relationships
        causal_graph = causal_engine.discover_causal_relationships(
            data=data,
            algorithm=algorithm,
            confidence_threshold=request.confidence_threshold,
        )

        # Generate unique graph ID
        graph_id = f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{algorithm.value}"

        # Store graph and data
        discovered_graphs[graph_id] = causal_graph
        graph_data[graph_id] = data

        # Calculate validation metrics
        validation_metrics = {
            "edge_count": len(causal_graph.edges),
            "node_count": len(causal_graph.nodes),
            "average_edge_strength": (
                np.mean([edge.strength for edge in causal_graph.edges])
                if causal_graph.edges
                else 0.0
            ),
            "average_confidence": (
                np.mean([edge.confidence for edge in causal_graph.edges])
                if causal_graph.edges
                else 0.0
            ),
            "domain_knowledge_consistency": causal_engine._check_domain_consistency(
                causal_graph
            ),
        }

        # Prepare response
        edges_data = []
        for edge in causal_graph.edges:
            edges_data.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "edge_type": edge.edge_type.value,
                    "strength": edge.strength,
                    "confidence": edge.confidence,
                    "p_value": edge.p_value,
                    "geological_interpretation": edge.geological_interpretation,
                }
            )

        logger.info(
            f"Causal discovery completed. Found {len(causal_graph.edges)} relationships"
        )

        return CausalDiscoveryResponse(
            graph_id=graph_id,
            nodes=causal_graph.nodes,
            edges=edges_data,
            algorithm_used=algorithm.value,
            confidence_threshold=request.confidence_threshold,
            creation_timestamp=causal_graph.creation_timestamp.isoformat(),
            validation_metrics=validation_metrics,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in causal discovery: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/intervene", response_model=InterventionResponse)
async def perform_intervention_analysis(request: InterventionRequest):
    """
    Perform causal intervention analysis

    Analyze the causal effects of intervening on a specific variable
    in the discovered causal graph.
    """
    try:
        # Validate graph ID
        if request.graph_id not in discovered_graphs:
            raise HTTPException(
                status_code=404, detail=f"Causal graph not found: {request.graph_id}"
            )

        causal_graph = discovered_graphs[request.graph_id]
        data = graph_data[request.graph_id]

        # Validate intervention variable
        if request.intervention_variable not in causal_graph.nodes:
            raise HTTPException(
                status_code=400,
                detail=f"Intervention variable not found in graph: {request.intervention_variable}",
            )

        # Validate target variables if specified
        if request.target_variables:
            missing_targets = [
                var for var in request.target_variables if var not in causal_graph.nodes
            ]
            if missing_targets:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target variables not found in graph: {missing_targets}",
                )

        logger.info(
            f"Performing intervention analysis on {request.intervention_variable}"
        )

        # Perform intervention analysis
        intervention_result = causal_engine.perform_intervention_analysis(
            causal_graph=causal_graph,
            data=data,
            intervention_variable=request.intervention_variable,
            intervention_value=request.intervention_value,
            target_variables=request.target_variables,
        )

        # Convert confidence intervals to list format for JSON serialization
        confidence_intervals_list = {
            var: [ci[0], ci[1]]
            for var, ci in intervention_result.confidence_intervals.items()
        }

        logger.info(
            f"Intervention analysis completed for {len(intervention_result.target_variables)} targets"
        )

        return InterventionResponse(
            intervention_variable=intervention_result.intervention_variable,
            intervention_value=request.intervention_value,
            target_variables=intervention_result.target_variables,
            causal_effects=intervention_result.causal_effects,
            confidence_intervals=confidence_intervals_list,
            statistical_significance=intervention_result.statistical_significance,
            geological_interpretation=intervention_result.geological_interpretation,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in intervention analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/generate-hypotheses", response_model=HypothesisGenerationResponse)
async def generate_hypotheses(request: HypothesisGenerationRequest):
    """
    Generate testable geological hypotheses from causal graph

    Analyze the discovered causal relationships and generate
    testable scientific hypotheses with experimental designs.
    """
    try:
        # Validate graph ID
        if request.graph_id not in discovered_graphs:
            raise HTTPException(
                status_code=404, detail=f"Causal graph not found: {request.graph_id}"
            )

        causal_graph = discovered_graphs[request.graph_id]

        logger.info(f"Generating hypotheses from graph {request.graph_id}")

        # Generate hypotheses
        hypotheses = causal_engine.generate_hypothesis_from_graph(causal_graph)

        # Create graph summary
        high_confidence_edges = [
            edge
            for edge in causal_graph.edges
            if edge.confidence > request.confidence_threshold
        ]

        graph_summary = {
            "total_edges": len(causal_graph.edges),
            "high_confidence_edges": len(high_confidence_edges),
            "confidence_threshold": request.confidence_threshold,
            "algorithm_used": causal_graph.algorithm_used.value,
            "strongest_relationships": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "strength": edge.strength,
                    "confidence": edge.confidence,
                }
                for edge in sorted(
                    causal_graph.edges, key=lambda x: x.strength, reverse=True
                )[:5]
            ],
        }

        logger.info(f"Generated {len(hypotheses)} testable hypotheses")

        return HypothesisGenerationResponse(
            hypotheses=hypotheses,
            graph_summary=graph_summary,
            generation_timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in hypothesis generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/validate", response_model=ValidationResponse)
async def validate_causal_graph(
    request: ValidationRequest, validation_file: UploadFile = File(...)
):
    """
    Validate discovered causal relationships with independent data

    Upload independent geological data to validate the discovered
    causal relationships and assess their reliability.
    """
    try:
        # Validate graph ID
        if request.graph_id not in discovered_graphs:
            raise HTTPException(
                status_code=404, detail=f"Causal graph not found: {request.graph_id}"
            )

        causal_graph = discovered_graphs[request.graph_id]

        # Read validation data
        content = await validation_file.read()

        try:
            if validation_file.filename.endswith(".csv"):
                validation_data = pd.read_csv(io.StringIO(content.decode("utf-8")))
            elif validation_file.filename.endswith(".json"):
                validation_data = pd.read_json(io.StringIO(content.decode("utf-8")))
            else:
                raise HTTPException(
                    status_code=400, detail="Unsupported file format. Use CSV or JSON."
                )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error reading validation data file: {str(e)}"
            )

        # Validate data compatibility
        missing_vars = [
            var for var in causal_graph.nodes if var not in validation_data.columns
        ]
        if missing_vars:
            raise HTTPException(
                status_code=400,
                detail=f"Validation data missing required variables: {missing_vars}",
            )

        logger.info(
            f"Validating causal graph {request.graph_id} with {len(validation_data)} validation samples"
        )

        # Perform validation
        validation_results = causal_engine.validate_causal_relationships(
            causal_graph=causal_graph, validation_data=validation_data
        )

        logger.info(
            f"Validation completed. Accuracy: {validation_results['overall_accuracy']:.3f}"
        )

        return ValidationResponse(**validation_results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in causal graph validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/graphs")
async def list_discovered_graphs():
    """List all discovered causal graphs"""
    graphs_info = []

    for graph_id, graph in discovered_graphs.items():
        graphs_info.append(
            {
                "graph_id": graph_id,
                "algorithm": graph.algorithm_used.value,
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "creation_timestamp": graph.creation_timestamp.isoformat(),
                "confidence_threshold": graph.confidence_threshold,
            }
        )

    return {"graphs": graphs_info}


@router.get("/graphs/{graph_id}")
async def get_causal_graph(graph_id: str):
    """Get detailed information about a specific causal graph"""
    if graph_id not in discovered_graphs:
        raise HTTPException(
            status_code=404, detail=f"Causal graph not found: {graph_id}"
        )

    causal_graph = discovered_graphs[graph_id]

    # Prepare detailed response
    edges_data = []
    for edge in causal_graph.edges:
        edges_data.append(
            {
                "source": edge.source,
                "target": edge.target,
                "edge_type": edge.edge_type.value,
                "strength": edge.strength,
                "confidence": edge.confidence,
                "p_value": edge.p_value,
                "geological_interpretation": edge.geological_interpretation,
            }
        )

    return {
        "graph_id": graph_id,
        "nodes": causal_graph.nodes,
        "edges": edges_data,
        "algorithm_used": causal_graph.algorithm_used.value,
        "confidence_threshold": causal_graph.confidence_threshold,
        "creation_timestamp": causal_graph.creation_timestamp.isoformat(),
        "domain_knowledge": causal_graph.geological_domain_knowledge,
    }


@router.delete("/graphs/{graph_id}")
async def delete_causal_graph(graph_id: str):
    """Delete a discovered causal graph"""
    if graph_id not in discovered_graphs:
        raise HTTPException(
            status_code=404, detail=f"Causal graph not found: {graph_id}"
        )

    del discovered_graphs[graph_id]
    if graph_id in graph_data:
        del graph_data[graph_id]

    return {"message": f"Causal graph {graph_id} deleted successfully"}


@router.post("/export/{graph_id}")
async def export_causal_graph(graph_id: str):
    """Export causal graph to JSON format"""
    if graph_id not in discovered_graphs:
        raise HTTPException(
            status_code=404, detail=f"Causal graph not found: {graph_id}"
        )

    causal_graph = discovered_graphs[graph_id]

    # Create export data
    export_data = {
        "graph_id": graph_id,
        "nodes": causal_graph.nodes,
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "edge_type": edge.edge_type.value,
                "strength": edge.strength,
                "confidence": edge.confidence,
                "p_value": edge.p_value,
                "geological_interpretation": edge.geological_interpretation,
            }
            for edge in causal_graph.edges
        ],
        "algorithm_used": causal_graph.algorithm_used.value,
        "confidence_threshold": causal_graph.confidence_threshold,
        "creation_timestamp": causal_graph.creation_timestamp.isoformat(),
        "domain_knowledge": causal_graph.geological_domain_knowledge,
    }

    return export_data


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "causal-inference-engine",
        "timestamp": datetime.now().isoformat(),
        "discovered_graphs": len(discovered_graphs),
    }
