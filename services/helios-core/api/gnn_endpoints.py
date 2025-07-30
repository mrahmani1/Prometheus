"""
API endpoints for Graph Neural Network functionality.

This module provides REST API endpoints for geological graph neural network
operations including model training, inference, and analysis.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import torch
import numpy as np
import logging
from pathlib import Path
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.graph_neural_networks import (
    GeologicalGCN, GeologicalGAT, GeologicalGraphSAGE,
    GeologicalGNNEnsemble, GeologicalGraphData
)
from core.geological_graph_utils import (
    GeologicalGraphBuilder, GeologicalGraphPreprocessor,
    GeologicalGraphAnalyzer, create_volcanic_system_graph
)
from core.gnn_training import (
    GeologicalGNNTrainer, GeologicalGNNEvaluator,
    create_geological_gnn_model, save_training_results
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/gnn", tags=["Graph Neural Networks"])


# Pydantic models for API requests/responses
class GeologicalEntityRequest(BaseModel):
    entity_id: str
    entity_type: str = Field(..., description="Type of geological entity")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    depth: float = Field(default=0.0, description="Depth in kilometers")
    properties: Dict[str, float] = Field(default_factory=dict)


class GeologicalRelationshipRequest(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str = Field(..., description="Type of relationship")
    strength: float = Field(..., ge=0, le=1, description="Relationship strength")
    properties: Dict[str, float] = Field(default_factory=dict)


class GraphConstructionRequest(BaseModel):
    entities: List[GeologicalEntityRequest]
    relationships: Optional[List[GeologicalRelationshipRequest]] = None
    auto_build_spatial: bool = Field(default=True, description="Auto-build spatial relationships")
    max_spatial_distance: float = Field(default=50.0, description="Max distance for spatial relationships (km)")


class GNNModelConfig(BaseModel):
    model_type: str = Field(..., description="Type of GNN model (gcn, gat, graphsage, ensemble)")
    input_dim: int = Field(..., gt=0)
    hidden_dims: List[int] = Field(..., description="Hidden layer dimensions")
    output_dim: int = Field(..., gt=0)
    num_node_types: int = Field(default=10)
    num_edge_types: int = Field(default=5)
    dropout: float = Field(default=0.2, ge=0, le=1)
    num_heads: Optional[List[int]] = Field(default=None, description="Number of attention heads for GAT")


class TrainingConfig(BaseModel):
    num_epochs: int = Field(default=100, gt=0)
    learning_rate: float = Field(default=0.001, gt=0)
    batch_size: int = Field(default=32, gt=0)
    task_type: str = Field(default="node_classification", description="Type of task")
    early_stopping_patience: int = Field(default=20, gt=0)
    physics_loss_weight: float = Field(default=0.1, ge=0)


class GraphAnalysisResponse(BaseModel):
    structure_metrics: Dict[str, float]
    node_type_distribution: Dict[str, int]
    edge_type_distribution: Dict[str, int]
    spatial_statistics: Dict[str, float]


class ModelPredictionResponse(BaseModel):
    predictions: List[List[float]]
    uncertainties: Optional[List[List[float]]] = None
    attention_weights: Optional[Dict[str, Any]] = None
    model_agreement: Optional[List[float]] = None


class TrainingResultsResponse(BaseModel):
    training_completed: bool
    final_train_loss: float
    final_val_loss: float
    best_val_metrics: Dict[str, float]
    model_path: str


# Global model storage
_loaded_models: Dict[str, torch.nn.Module] = {}
_model_configs: Dict[str, Dict] = {}


@router.post("/construct-graph", response_model=Dict[str, Any])
async def construct_geological_graph(request: GraphConstructionRequest):
    """
    Construct a geological graph from entities and relationships.
    """
    try:
        builder = GeologicalGraphBuilder()
        
        # Add entities
        for entity in request.entities:
            builder.add_entity(
                entity.entity_id,
                entity.entity_type,
                (entity.latitude, entity.longitude, entity.depth),
                entity.properties
            )
        
        # Add explicit relationships
        if request.relationships:
            for rel in request.relationships:
                if rel.relationship_type == 'spatial':
                    builder.add_spatial_relationship(
                        rel.source_id, rel.target_id, 
                        rel.properties.get('distance', 1.0), rel.properties
                    )
                elif rel.relationship_type == 'causal':
                    builder.add_causal_relationship(
                        rel.source_id, rel.target_id, rel.strength, rel.properties
                    )
                elif rel.relationship_type == 'temporal':
                    builder.add_temporal_relationship(
                        rel.source_id, rel.target_id,
                        rel.properties.get('temporal_lag', 0.0),
                        rel.strength, rel.properties
                    )
        
        # Auto-build spatial relationships if requested
        if request.auto_build_spatial:
            builder.build_spatial_graph(max_distance=request.max_spatial_distance)
        
        # Convert to graph data
        graph_data = builder.to_geological_graph_data()
        
        # Analyze the constructed graph
        analyzer = GeologicalGraphAnalyzer()
        structure_metrics = analyzer.analyze_graph_structure(graph_data)
        node_type_dist = analyzer.analyze_node_types(graph_data)
        edge_type_dist = analyzer.analyze_edge_types(graph_data)
        spatial_stats = analyzer.calculate_spatial_statistics(graph_data)
        
        return {
            "graph_constructed": True,
            "num_nodes": graph_data.node_features.size(0),
            "num_edges": graph_data.edge_index.size(1),
            "structure_metrics": structure_metrics,
            "node_type_distribution": node_type_dist,
            "edge_type_distribution": edge_type_dist,
            "spatial_statistics": spatial_stats
        }
        
    except Exception as e:
        logger.error(f"Error constructing geological graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph construction failed: {str(e)}")


@router.post("/analyze-graph", response_model=GraphAnalysisResponse)
async def analyze_geological_graph(request: GraphConstructionRequest):
    """
    Analyze the structure and properties of a geological graph.
    """
    try:
        # First construct the graph
        builder = GeologicalGraphBuilder()
        
        for entity in request.entities:
            builder.add_entity(
                entity.entity_id,
                entity.entity_type,
                (entity.latitude, entity.longitude, entity.depth),
                entity.properties
            )
        
        if request.auto_build_spatial:
            builder.build_spatial_graph(max_distance=request.max_spatial_distance)
        
        graph_data = builder.to_geological_graph_data()
        
        # Analyze the graph
        analyzer = GeologicalGraphAnalyzer()
        structure_metrics = analyzer.analyze_graph_structure(graph_data)
        node_type_dist = analyzer.analyze_node_types(graph_data)
        edge_type_dist = analyzer.analyze_edge_types(graph_data)
        spatial_stats = analyzer.calculate_spatial_statistics(graph_data)
        
        return GraphAnalysisResponse(
            structure_metrics=structure_metrics,
            node_type_distribution=node_type_dist,
            edge_type_distribution=edge_type_dist,
            spatial_statistics=spatial_stats
        )
        
    except Exception as e:
        logger.error(f"Error analyzing geological graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph analysis failed: {str(e)}")


@router.post("/create-model/{model_id}")
async def create_gnn_model(model_id: str, config: GNNModelConfig):
    """
    Create and initialize a geological GNN model.
    """
    try:
        # Create model based on configuration
        model_kwargs = {
            'num_node_types': config.num_node_types,
            'num_edge_types': config.num_edge_types,
            'dropout': config.dropout
        }
        
        if config.model_type == 'gat' and config.num_heads:
            model_kwargs['num_heads'] = config.num_heads
        
        model = create_geological_gnn_model(
            config.model_type,
            config.input_dim,
            config.hidden_dims,
            config.output_dim,
            **model_kwargs
        )
        
        # Store model and configuration
        _loaded_models[model_id] = model
        _model_configs[model_id] = config.dict()
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_created": True,
            "model_id": model_id,
            "model_type": config.model_type,
            "total_parameters": num_params,
            "trainable_parameters": trainable_params,
            "model_config": config.dict()
        }
        
    except Exception as e:
        logger.error(f"Error creating GNN model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model creation failed: {str(e)}")


@router.post("/train-model/{model_id}")
async def train_gnn_model(
    model_id: str, 
    training_config: TrainingConfig,
    graph_request: GraphConstructionRequest,
    background_tasks: BackgroundTasks
):
    """
    Train a geological GNN model on provided graph data.
    """
    try:
        if model_id not in _loaded_models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        model = _loaded_models[model_id]
        
        # Construct training graph
        builder = GeologicalGraphBuilder()
        
        for entity in graph_request.entities:
            builder.add_entity(
                entity.entity_id,
                entity.entity_type,
                (entity.latitude, entity.longitude, entity.depth),
                entity.properties
            )
        
        if graph_request.auto_build_spatial:
            builder.build_spatial_graph(max_distance=graph_request.max_spatial_distance)
        
        graph_data = builder.to_geological_graph_data()
        
        # Add dummy targets for training (in real scenario, these would be provided)
        if training_config.task_type == 'node_classification':
            graph_data.y = torch.randint(0, _model_configs[model_id]['output_dim'], 
                                       (graph_data.node_features.size(0),))
        elif training_config.task_type == 'regression':
            graph_data.y = torch.randn(graph_data.node_features.size(0), 
                                     _model_configs[model_id]['output_dim'])
        
        # Create trainer
        trainer = GeologicalGNNTrainer(
            model,
            learning_rate=training_config.learning_rate
        )
        trainer.physics_loss_weight = training_config.physics_loss_weight
        
        # Create simple data loaders (in practice, would split data properly)
        train_loader = [graph_data]
        val_loader = [graph_data]
        
        # Choose loss function
        if training_config.task_type == 'node_classification':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()
        
        # Start training in background
        def train_model():
            try:
                results = trainer.train(
                    train_loader, val_loader,
                    training_config.num_epochs,
                    criterion,
                    training_config.task_type,
                    training_config.early_stopping_patience,
                    f"model_storage/{model_id}_best.pth"
                )
                
                # Save training results
                save_training_results(
                    results,
                    _model_configs[model_id],
                    f"model_storage/{model_id}"
                )
                
                logger.info(f"Training completed for model {model_id}")
                
            except Exception as e:
                logger.error(f"Training failed for model {model_id}: {str(e)}")
        
        background_tasks.add_task(train_model)
        
        return {
            "training_started": True,
            "model_id": model_id,
            "estimated_duration_minutes": training_config.num_epochs * 0.1,  # Rough estimate
            "training_config": training_config.dict()
        }
        
    except Exception as e:
        logger.error(f"Error starting training for model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/predict/{model_id}", response_model=ModelPredictionResponse)
async def predict_with_gnn_model(
    model_id: str,
    graph_request: GraphConstructionRequest
):
    """
    Make predictions using a trained geological GNN model.
    """
    try:
        if model_id not in _loaded_models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        model = _loaded_models[model_id]
        model.eval()
        
        # Construct graph for prediction
        builder = GeologicalGraphBuilder()
        
        for entity in graph_request.entities:
            builder.add_entity(
                entity.entity_id,
                entity.entity_type,
                (entity.latitude, entity.longitude, entity.depth),
                entity.properties
            )
        
        if graph_request.auto_build_spatial:
            builder.build_spatial_graph(max_distance=graph_request.max_spatial_distance)
        
        graph_data = builder.to_geological_graph_data()
        
        # Make predictions
        with torch.no_grad():
            if isinstance(model, GeologicalGNNEnsemble):
                outputs = model(graph_data)
                predictions = outputs['ensemble_prediction']
                uncertainties = outputs['uncertainty']
                attention_weights = outputs.get('attention_weights', {})
                model_agreement = model.get_model_agreement(graph_data)
                
                return ModelPredictionResponse(
                    predictions=predictions.cpu().numpy().tolist(),
                    uncertainties=uncertainties.cpu().numpy().tolist(),
                    attention_weights=attention_weights,
                    model_agreement=model_agreement.cpu().numpy().tolist()
                )
            else:
                if isinstance(model, GeologicalGAT):
                    predictions, attention_weights = model(graph_data)
                    return ModelPredictionResponse(
                        predictions=predictions.cpu().numpy().tolist(),
                        attention_weights=attention_weights
                    )
                else:
                    predictions = model(graph_data)
                    return ModelPredictionResponse(
                        predictions=predictions.cpu().numpy().tolist()
                    )
        
    except Exception as e:
        logger.error(f"Error making predictions with model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/models")
async def list_loaded_models():
    """
    List all currently loaded GNN models.
    """
    models_info = {}
    for model_id, model in _loaded_models.items():
        config = _model_configs.get(model_id, {})
        num_params = sum(p.numel() for p in model.parameters())
        
        models_info[model_id] = {
            "model_type": config.get("model_type", "unknown"),
            "parameters": num_params,
            "config": config
        }
    
    return {
        "loaded_models": models_info,
        "total_models": len(_loaded_models)
    }


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a loaded GNN model from memory.
    """
    if model_id not in _loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    del _loaded_models[model_id]
    if model_id in _model_configs:
        del _model_configs[model_id]
    
    return {
        "model_deleted": True,
        "model_id": model_id
    }


@router.post("/volcanic-system-graph")
async def create_volcanic_system_graph_endpoint(
    volcanoes: List[Dict],
    seismic_events: List[Dict],
    monitoring_stations: List[Dict],
    max_spatial_distance: float = 50.0
):
    """
    Create a specialized volcanic system graph.
    """
    try:
        graph_data = create_volcanic_system_graph(
            volcanoes, seismic_events, monitoring_stations, max_spatial_distance
        )
        
        # Analyze the created graph
        analyzer = GeologicalGraphAnalyzer()
        structure_metrics = analyzer.analyze_graph_structure(graph_data)
        node_type_dist = analyzer.analyze_node_types(graph_data)
        spatial_stats = analyzer.calculate_spatial_statistics(graph_data)
        
        return {
            "volcanic_system_graph_created": True,
            "num_nodes": graph_data.node_features.size(0),
            "num_edges": graph_data.edge_index.size(1),
            "structure_metrics": structure_metrics,
            "node_type_distribution": node_type_dist,
            "spatial_statistics": spatial_stats
        }
        
    except Exception as e:
        logger.error(f"Error creating volcanic system graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Volcanic system graph creation failed: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check for the GNN service.
    """
    return {
        "status": "healthy",
        "loaded_models": len(_loaded_models),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }