"""
API endpoints for Hypothesis Generation and Validation System

Provides REST API endpoints for automated hypothesis generation from discovered
causal relationships, mathematical formulation of geological laws, confidence
assessment, and experimental design suggestions.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from core.hypothesis_generation_engine import (
    CausalRelationship,
    HypothesisValidationSystem,
    ScientificHypothesis,
)
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hypothesis-generation", tags=["hypothesis-generation"])

# Pydantic models for API
class CausalRelationshipRequest(BaseModel):
    cause: str = Field(..., description="Cause variable name")
    effect: str = Field(..., description="Effect variable name")
    strength: float = Field(..., ge=-1, le=1, description="Causal strength coefficient")
    confidence: float = Field(..., ge=0, le=1, description="Statistical confidence")
    mechanism: Optional[str] = Field(None, description="Proposed physical mechanism")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")

class GeologicalContextRequest(BaseModel):
    geological_setting: str = Field(
        ..., description="Geological setting (e.g., volcanic_system)"
    )
    temporal_scale: str = Field(
        default="real_time", description="Temporal scale of analysis"
    )
    spatial_scale: str = Field(default="local", description="Spatial scale of analysis")
    location: Optional[str] = Field(None, description="Geographic location")
    additional_context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )

class ResourceConstraintsRequest(BaseModel):
    budget: str = Field(
        default="medium", description="Available budget (low/medium/high)"
    )
    time: str = Field(
        default="medium",
        description="Available time (hours_to_days/days_to_weeks/weeks_to_months/months_to_years)",
    )
    equipment: List[str] = Field(
        default_factory=list, description="Available equipment"
    )
    personnel: Optional[int] = Field(None, description="Available personnel count")

class HypothesisGenerationRequest(BaseModel):
    causal_relationships: List[CausalRelationshipRequest] = Field(
        ..., description="Discovered causal relationships"
    )
    geological_data: Dict[str, List[List[float]]] = Field(
        default_factory=dict, description="Supporting geological data patterns"
    )
    geological_context: GeologicalContextRequest = Field(
        ..., description="Geological context"
    )
    available_resources: ResourceConstraintsRequest = Field(
        ..., description="Available resources for validation"
    )

class ScientificHypothesisResponse(BaseModel):
    id: str
    statement: str
    mathematical_formulation: Optional[str] = None
    confidence_score: float
    testability_score: float
    novelty_score: float
    predicted_outcomes: List[str]
    experimental_design: Optional[List[Dict[str, Any]]] = None
    generated_timestamp: datetime

class GeologicalLawResponse(BaseModel):
    name: str
    mathematical_expression: str
    variables: List[str]
    domain_applicability: List[str]
    confidence: float
    validation_experiments: List[Dict[str, Any]]

class HypothesisGenerationResponse(BaseModel):
    generated_hypotheses: List[ScientificHypothesisResponse]
    formulated_laws: List[GeologicalLawResponse]
    total_hypotheses: int
    high_confidence_hypotheses: int
    testable_hypotheses: int
    generation_timestamp: datetime
    summary: Dict[str, Any]
    processing_time_seconds: float

# Initialize the hypothesis validation system
hypothesis_system = HypothesisValidationSystem()

@router.post("/generate", response_model=HypothesisGenerationResponse)
async def generate_hypotheses(
    request: HypothesisGenerationRequest, background_tasks: BackgroundTasks
) -> HypothesisGenerationResponse:
    """
    Generate scientific hypotheses from causal relationships

    This endpoint implements the complete hypothesis generation and validation workflow:
    - Generates testable hypotheses from discovered causal relationships (Requirement 15.3)
    - Formulates mathematical expressions for geological laws (Requirement 15.8)
    - Assesses confidence in generated hypotheses (Requirement 15.8)
    - Designs validation experiments (Requirement 15.7)
    """

    try:
        start_time = datetime.now()
        logger.info(
            f"Starting hypothesis generation for {len(request.causal_relationships)} causal relationships"
        )

        # Convert request models to internal data structures
        causal_relationships = [
            CausalRelationship(
                cause=rel.cause,
                effect=rel.effect,
                strength=rel.strength,
                confidence=rel.confidence,
                mechanism=rel.mechanism,
                evidence=rel.evidence,
            )
            for rel in request.causal_relationships
        ]

        # Convert geological data to numpy arrays
        geological_data = {}
        for key, data_list in request.geological_data.items():
            try:
                geological_data[key] = np.array(data_list)
            except Exception as e:
                logger.warning(f"Failed to convert data for {key}: {e}")
                continue

        # Convert context and resources
        geological_context = {
            "geological_setting": request.geological_context.geological_setting,
            "temporal_scale": request.geological_context.temporal_scale,
            "spatial_scale": request.geological_context.spatial_scale,
            "location": request.geological_context.location,
            **request.geological_context.additional_context,
        }

        available_resources = {
            "budget": request.available_resources.budget,
            "time": request.available_resources.time,
            "equipment": request.available_resources.equipment,
            "personnel": request.available_resources.personnel,
        }

        # Generate hypotheses and laws
        results = hypothesis_system.generate_and_validate_hypotheses(
            causal_relationships=causal_relationships,
            geological_data=geological_data,
            geological_context=geological_context,
            available_resources=available_resources,
        )

        # Convert results to response models
        hypothesis_responses = [
            ScientificHypothesisResponse(
                id=hyp.id,
                statement=hyp.statement,
                mathematical_formulation=hyp.mathematical_formulation,
                confidence_score=hyp.confidence_score,
                testability_score=hyp.testability_score,
                novelty_score=hyp.novelty_score,
                predicted_outcomes=hyp.predicted_outcomes,
                experimental_design=hyp.experimental_design,
                generated_timestamp=hyp.generated_timestamp,
            )
            for hyp in results["generated_hypotheses"]
        ]

        law_responses = [
            GeologicalLawResponse(
                name=law.name,
                mathematical_expression=law.mathematical_expression,
                variables=law.variables,
                domain_applicability=law.domain_applicability,
                confidence=law.confidence,
                validation_experiments=law.validation_experiments,
            )
            for law in results["formulated_laws"]
        ]

        processing_time = (datetime.now() - start_time).total_seconds()

        response = HypothesisGenerationResponse(
            generated_hypotheses=hypothesis_responses,
            formulated_laws=law_responses,
            total_hypotheses=results["total_hypotheses"],
            high_confidence_hypotheses=results["high_confidence_hypotheses"],
            testable_hypotheses=results["testable_hypotheses"],
            generation_timestamp=results["generation_timestamp"],
            summary=results["summary"],
            processing_time_seconds=processing_time,
        )

        # Log results for monitoring
        background_tasks.add_task(
            log_hypothesis_generation_results,
            len(causal_relationships),
            len(hypothesis_responses),
            len(law_responses),
            processing_time,
        )

        logger.info(
            f"Generated {len(hypothesis_responses)} hypotheses and {len(law_responses)} laws in {processing_time:.2f}s"
        )

        return response

    except Exception as e:
        logger.error(f"Error in hypothesis generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Hypothesis generation failed: {str(e)}"
        )

@router.get("/status")
async def get_system_status():
    """Get status of the hypothesis generation system"""

    try:
        return {
            "status": "operational",
            "system_components": {
                "hypothesis_generator": "active",
                "mathematical_formulator": "active",
                "confidence_assessor": "active",
                "experiment_designer": "active",
            },
            "capabilities": [
                "automated_hypothesis_generation",
                "mathematical_law_formulation",
                "confidence_assessment",
                "experimental_design",
            ],
            "supported_contexts": [
                "volcanic_systems",
                "tectonic_boundaries",
                "sedimentary_basins",
                "hydrothermal_systems",
            ],
            "timestamp": datetime.now(),
        }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

@router.post("/validate-hypothesis")
async def validate_single_hypothesis(
    hypothesis_statement: str = Field(
        ..., description="Hypothesis statement to validate"
    ),
    supporting_data: Dict[str, Any] = Field(
        default_factory=dict, description="Supporting data for validation"
    ),
    geological_context: Optional[GeologicalContextRequest] = None,
):
    """
    Validate a single hypothesis with confidence assessment

    This endpoint provides confidence assessment for individual hypotheses
    without the full generation workflow.
    """

    try:
        logger.info(f"Validating hypothesis: {hypothesis_statement[:100]}...")

        # Create a hypothesis object
        hypothesis = ScientificHypothesis(
            id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            statement=hypothesis_statement,
            supporting_relationships=[],
        )

        # Assess confidence
        confidence_assessor = hypothesis_system.confidence_assessor
        confidence_score = confidence_assessor.assess_hypothesis_confidence(
            hypothesis, supporting_data
        )

        # Determine testability
        testability_factors = {
            "measurable_variables": len(
                [
                    term
                    for term in ["temperature", "pressure", "density", "velocity"]
                    if term in hypothesis_statement.lower()
                ]
            ),
            "temporal_constraints": "time" in hypothesis_statement.lower(),
            "spatial_constraints": any(
                term in hypothesis_statement.lower()
                for term in ["location", "depth", "surface", "subsurface"]
            ),
            "physical_mechanism": any(
                term in hypothesis_statement.lower()
                for term in ["heat", "flow", "deformation", "reaction"]
            ),
        }

        testability_score = min(sum(testability_factors.values()) / 4, 1.0)

        return {
            "hypothesis_id": hypothesis.id,
            "statement": hypothesis_statement,
            "confidence_score": confidence_score,
            "testability_score": testability_score,
            "validation_timestamp": datetime.now(),
            "testability_factors": testability_factors,
            "recommendations": [
                (
                    "High confidence - proceed with experimental validation"
                    if confidence_score > 0.8
                    else (
                        "Medium confidence - gather additional supporting data"
                        if confidence_score > 0.6
                        else "Low confidence - requires significant additional evidence"
                    )
                )
            ],
        }

    except Exception as e:
        logger.error(f"Error validating hypothesis: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Hypothesis validation failed: {str(e)}"
        )

@router.post("/design-experiment")
async def design_validation_experiment(
    hypothesis_statement: str = Field(
        ..., description="Hypothesis to design experiment for"
    ),
    experiment_type: Optional[str] = Field(
        None, description="Preferred experiment type"
    ),
    geological_context: GeologicalContextRequest = Field(
        ..., description="Geological context"
    ),
    available_resources: ResourceConstraintsRequest = Field(
        ..., description="Available resources"
    ),
):
    """
    Design validation experiments for a specific hypothesis

    This endpoint focuses on experimental design suggestions for hypothesis validation
    (Requirement 15.7).
    """

    try:
        logger.info(
            f"Designing experiment for hypothesis: {hypothesis_statement[:100]}..."
        )

        # Create hypothesis object
        hypothesis = ScientificHypothesis(
            id=f"exp_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            statement=hypothesis_statement,
        )

        # Convert resources
        resources = {
            "budget": available_resources.budget,
            "time": available_resources.time,
            "equipment": available_resources.equipment,
            "personnel": available_resources.personnel,
        }

        # Design experiments
        experiment_designer = hypothesis_system.experiment_designer
        experiments = experiment_designer.design_validation_experiments(
            hypothesis=hypothesis,
            available_resources=resources,
            geological_context=geological_context.geological_setting,
        )

        # Filter by preferred type if specified
        if experiment_type:
            experiments = [
                exp
                for exp in experiments
                if exp.get("experiment_type") == experiment_type
            ]

        return {
            "hypothesis_statement": hypothesis_statement,
            "designed_experiments": experiments,
            "total_experiments": len(experiments),
            "design_timestamp": datetime.now(),
            "recommendations": [
                (
                    f"Prioritize {experiments[0]['experiment_type']} for best results"
                    if experiments
                    else "No suitable experiments found for current constraints"
                )
            ],
        }

    except Exception as e:
        logger.error(f"Error designing experiment: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Experiment design failed: {str(e)}"
        )

async def log_hypothesis_generation_results(
    input_relationships: int,
    generated_hypotheses: int,
    formulated_laws: int,
    processing_time: float,
):
    """Background task to log hypothesis generation results for monitoring"""

    logger.info(
        "Hypothesis Generation Metrics: "
        f"Input relationships: {input_relationships}, "
        f"Generated hypotheses: {generated_hypotheses}, "
        f"Formulated laws: {formulated_laws}, "
        f"Processing time: {processing_time:.2f}s"
    )

    # In a production system, this would write to a metrics database
    # or monitoring system for performance tracking
