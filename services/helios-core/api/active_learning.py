"""
Active Learning API Endpoints

This module provides REST API endpoints for the active learning engine,
allowing users to select optimal geological samples for model improvement.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.active_learning_engine import (ActiveLearningConfig,
                                           ActiveLearningEngine,
                                           GeologicalSample)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/active-learning", tags=["active-learning"])

# Global active learning engine instance
active_learning_engine: Optional[ActiveLearningEngine] = None


class GeologicalSampleRequest(BaseModel):
    """Request model for geological sample"""

    id: str
    features: List[float]
    location: List[float] = Field(..., min_items=3, max_items=3)
    timestamp: Optional[str] = None
    sample_type: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None
    label: Optional[List[float]] = None
    uncertainty: Optional[float] = None


class ActiveLearningConfigRequest(BaseModel):
    """Request model for active learning configuration"""

    strategy: str = Field(..., regex="^(uncertainty|committee|diversity|hybrid)$")
    batch_size: int = Field(10, ge=1, le=100)
    uncertainty_threshold: float = Field(0.1, ge=0.0, le=1.0)
    diversity_weight: float = Field(0.3, ge=0.0, le=1.0)
    committee_size: int = Field(5, ge=2, le=20)
    geological_features: Optional[List[str]] = None
    evaluation_metrics: Optional[List[str]] = None


class SampleSelectionRequest(BaseModel):
    """Request model for sample selection"""

    unlabeled_pool: List[GeologicalSampleRequest]
    labeled_data: List[GeologicalSampleRequest]
    config: ActiveLearningConfigRequest
    model_id: Optional[str] = None
    use_committee: bool = False


class SampleSelectionResponse(BaseModel):
    """Response model for sample selection"""

    selected_samples: List[GeologicalSampleRequest]
    selection_scores: List[float]
    strategy_used: str
    evaluation_metrics: Dict[str, float]
    geological_coverage: Dict[str, float]
    timestamp: str
    processing_time_ms: float


class EvaluationSummaryResponse(BaseModel):
    """Response model for evaluation summary"""

    total_evaluations: int
    strategies_used: List[str]
    average_metrics: Dict[str, Dict[str, float]]
    best_performing_strategy: Optional[Dict[str, Any]]
    trends: Dict[str, Any]


class StrategyRecommendationResponse(BaseModel):
    """Response model for strategy recommendations"""

    primary: str
    reason: str
    secondary: Optional[str] = None
    coverage_note: Optional[str] = None


def get_active_learning_engine() -> ActiveLearningEngine:
    """Get or create active learning engine instance"""
    global active_learning_engine

    if active_learning_engine is None:
        config = ActiveLearningConfig()
        active_learning_engine = ActiveLearningEngine(config)

    return active_learning_engine


def convert_sample_request_to_geological_sample(
    sample_req: GeologicalSampleRequest,
) -> GeologicalSample:
    """Convert API request model to internal GeologicalSample"""
    return GeologicalSample(
        id=sample_req.id,
        features=np.array(sample_req.features),
        location=tuple(sample_req.location),
        timestamp=sample_req.timestamp,
        sample_type=sample_req.sample_type,
        metadata=sample_req.metadata or {},
        label=np.array(sample_req.label) if sample_req.label else None,
        uncertainty=sample_req.uncertainty,
    )


def convert_geological_sample_to_response(
    sample: GeologicalSample,
) -> GeologicalSampleRequest:
    """Convert internal GeologicalSample to API response model"""
    return GeologicalSampleRequest(
        id=sample.id,
        features=sample.features.tolist(),
        location=list(sample.location),
        timestamp=sample.timestamp,
        sample_type=sample.sample_type,
        metadata=sample.metadata,
        label=sample.label.tolist() if sample.label is not None else None,
        uncertainty=sample.uncertainty,
    )


@router.post("/select-samples", response_model=SampleSelectionResponse)
async def select_optimal_samples(
    request: SampleSelectionRequest,
    background_tasks: BackgroundTasks,
    engine: ActiveLearningEngine = Depends(get_active_learning_engine),
):
    """
    Select optimal geological samples for model improvement using active learning.

    This endpoint applies various active learning strategies to select the most
    informative samples from an unlabeled pool for geological model training.
    """
    try:
        start_time = datetime.now()

        # Convert request models to internal format
        unlabeled_pool = [
            convert_sample_request_to_geological_sample(sample)
            for sample in request.unlabeled_pool
        ]

        labeled_data = [
            convert_sample_request_to_geological_sample(sample)
            for sample in request.labeled_data
        ]

        # Update engine configuration
        config = ActiveLearningConfig(
            strategy=request.config.strategy,
            batch_size=request.config.batch_size,
            uncertainty_threshold=request.config.uncertainty_threshold,
            diversity_weight=request.config.diversity_weight,
            committee_size=request.config.committee_size,
            geological_features=request.config.geological_features,
            evaluation_metrics=request.config.evaluation_metrics,
        )

        engine.update_strategy_config(config)

        # Get or create model
        if request.model_id:
            # In a real implementation, you would load the specific model
            # For now, we'll use a simple placeholder model
            model = _create_placeholder_model(unlabeled_pool[0].features.shape[0])
        else:
            model = _create_placeholder_model(unlabeled_pool[0].features.shape[0])

        # Handle committee models if requested
        committee_models = None
        if request.use_committee and request.config.strategy in ["committee", "hybrid"]:
            committee_models = [
                _create_placeholder_model(unlabeled_pool[0].features.shape[0])
                for _ in range(request.config.committee_size)
            ]

        # Select optimal samples
        result = engine.select_optimal_samples(
            unlabeled_pool=unlabeled_pool,
            labeled_data=labeled_data,
            model=model,
            committee_models=committee_models,
        )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Convert result to response format
        response = SampleSelectionResponse(
            selected_samples=[
                convert_geological_sample_to_response(sample)
                for sample in result.selected_samples
            ],
            selection_scores=result.selection_scores.tolist(),
            strategy_used=result.strategy_used,
            evaluation_metrics=result.evaluation_metrics,
            geological_coverage=result.geological_coverage,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
        )

        # Log the selection for monitoring
        logger.info(
            f"Selected {len(result.selected_samples)} samples using {result.strategy_used} strategy. "
            f"Processing time: {processing_time:.2f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Error in sample selection: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Sample selection failed: {str(e)}"
        )


@router.get("/evaluation-summary", response_model=EvaluationSummaryResponse)
async def get_evaluation_summary(
    engine: ActiveLearningEngine = Depends(get_active_learning_engine),
):
    """
    Get comprehensive evaluation summary of active learning performance.

    Returns statistics and trends from all previous active learning evaluations.
    """
    try:
        summary = engine.evaluator.get_evaluation_summary()

        return EvaluationSummaryResponse(
            total_evaluations=summary.get("total_evaluations", 0),
            strategies_used=summary.get("strategies_used", []),
            average_metrics=summary.get("average_metrics", {}),
            best_performing_strategy=summary.get("best_performing_strategy"),
            trends=summary.get("trends", {}),
        )

    except Exception as e:
        logger.error(f"Error getting evaluation summary: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get evaluation summary: {str(e)}"
        )


@router.post("/strategy-recommendations", response_model=StrategyRecommendationResponse)
async def get_strategy_recommendations(
    unlabeled_pool: List[GeologicalSampleRequest],
    labeled_data: List[GeologicalSampleRequest],
    engine: ActiveLearningEngine = Depends(get_active_learning_engine),
):
    """
    Get recommendations for optimal active learning strategy selection.

    Analyzes the characteristics of the dataset to recommend the most
    appropriate active learning strategy.
    """
    try:
        # Convert to internal format
        unlabeled_samples = [
            convert_sample_request_to_geological_sample(sample)
            for sample in unlabeled_pool
        ]

        labeled_samples = [
            convert_sample_request_to_geological_sample(sample)
            for sample in labeled_data
        ]

        recommendations = engine.get_strategy_recommendations(
            unlabeled_samples, labeled_samples
        )

        return StrategyRecommendationResponse(
            primary=recommendations["primary"],
            reason=recommendations["reason"],
            secondary=recommendations.get("secondary"),
            coverage_note=recommendations.get("coverage_note"),
        )

    except Exception as e:
        logger.error(f"Error getting strategy recommendations: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get recommendations: {str(e)}"
        )


@router.post("/update-config")
async def update_active_learning_config(
    config: ActiveLearningConfigRequest,
    engine: ActiveLearningEngine = Depends(get_active_learning_engine),
):
    """
    Update the active learning engine configuration.

    Allows dynamic reconfiguration of active learning parameters.
    """
    try:
        new_config = ActiveLearningConfig(
            strategy=config.strategy,
            batch_size=config.batch_size,
            uncertainty_threshold=config.uncertainty_threshold,
            diversity_weight=config.diversity_weight,
            committee_size=config.committee_size,
            geological_features=config.geological_features,
            evaluation_metrics=config.evaluation_metrics,
        )

        engine.update_strategy_config(new_config)

        return {
            "message": "Active learning configuration updated successfully",
            "config": {
                "strategy": new_config.strategy,
                "batch_size": new_config.batch_size,
                "uncertainty_threshold": new_config.uncertainty_threshold,
                "diversity_weight": new_config.diversity_weight,
                "committee_size": new_config.committee_size,
            },
        }

    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update configuration: {str(e)}"
        )


@router.post("/export-evaluation-report")
async def export_evaluation_report(
    filepath: str,
    background_tasks: BackgroundTasks,
    engine: ActiveLearningEngine = Depends(get_active_learning_engine),
):
    """
    Export detailed evaluation report to file.

    Generates a comprehensive report of active learning performance metrics.
    """
    try:
        # Add export task to background
        background_tasks.add_task(engine.export_evaluation_report, filepath)

        return {
            "message": "Evaluation report export initiated",
            "filepath": filepath,
            "status": "processing",
        }

    except Exception as e:
        logger.error(f"Error exporting evaluation report: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to export report: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for active learning service"""
    try:
        engine = get_active_learning_engine()

        return {
            "status": "healthy",
            "service": "active-learning",
            "strategies_available": list(engine.strategies.keys()),
            "evaluations_performed": len(engine.evaluator.metrics_history),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


def _create_placeholder_model(input_size: int) -> torch.nn.Module:
    """Create a placeholder model for testing purposes"""

    class PlaceholderModel(torch.nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_size, 64)
            self.fc2 = torch.nn.Linear(64, 32)
            self.fc3 = torch.nn.Linear(32, 1)
            self.dropout = torch.nn.Dropout(0.2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    return PlaceholderModel(input_size)


# Example usage endpoints for testing
@router.post("/test/generate-sample-data")
async def generate_sample_data(
    n_samples: int = 100, sample_types: Optional[List[str]] = None
):
    """
    Generate sample geological data for testing active learning.

    This endpoint is useful for testing and demonstration purposes.
    """
    try:
        from ..core.active_learning_engine import create_sample_geological_data

        samples = create_sample_geological_data(n_samples)

        # Convert to response format
        sample_responses = [
            convert_geological_sample_to_response(sample) for sample in samples
        ]

        return {
            "message": f"Generated {n_samples} sample geological data points",
            "samples": sample_responses,
            "sample_types": list(set(sample.sample_type for sample in samples)),
        }

    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate sample data: {str(e)}"
        )


@router.post("/test/run-strategy-comparison")
async def run_strategy_comparison(n_samples: int = 500, batch_size: int = 20):
    """
    Run a comparison of different active learning strategies.

    This endpoint demonstrates the performance of different strategies
    on synthetic geological data.
    """
    try:

        # Generate test data
        all_samples = create_sample_geological_data(n_samples)
        labeled_data = all_samples[:100]
        unlabeled_pool = all_samples[100:]

        # Create test model
        model = _create_placeholder_model(all_samples[0].features.shape[0])

        # Test different strategies
        strategies = ["uncertainty", "diversity", "hybrid"]
        results = {}

        for strategy in strategies:
            config = ActiveLearningConfig(
                strategy=strategy, batch_size=batch_size, diversity_weight=0.3
            )

            engine = ActiveLearningEngine(config)

            result = engine.select_optimal_samples(unlabeled_pool, labeled_data, model)

            results[strategy] = {
                "selected_count": len(result.selected_samples),
                "evaluation_metrics": result.evaluation_metrics,
                "geological_coverage": result.geological_coverage,
                "strategy_used": result.strategy_used,
            }

        return {
            "message": "Strategy comparison completed",
            "test_parameters": {
                "total_samples": n_samples,
                "labeled_samples": len(labeled_data),
                "unlabeled_samples": len(unlabeled_pool),
                "batch_size": batch_size,
            },
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error in strategy comparison: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Strategy comparison failed: {str(e)}"
        )
