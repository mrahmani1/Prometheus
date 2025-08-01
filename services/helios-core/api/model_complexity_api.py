"""
Model Complexity Management API

Provides REST endpoints for model complexity assessment and management.
"""

from typing import Any, Dict, List, Optional

import structlog
import torch
import torch.nn.functional as F
from core.model_complexity_manager import ComplexityLevel, ModelComplexityManager
from fastapi import APIRouter, HTTPException
from models.ai_models import GeologicalProcess
from pydantic import BaseModel, Field

logger = structlog.get_logger()
router = APIRouter(prefix="/complexity", tags=["Model Complexity"])

# Global complexity manager instance
complexity_manager = ModelComplexityManager()

class ComplexityAssessmentRequest(BaseModel):
    """Request model for complexity assessment"""

    geological_process: str = Field(..., description="Type of geological process")
    data_info: Dict[str, Any] = Field(
        ..., description="Information about available data"
    )
    target_accuracy: float = Field(
        0.8, ge=0.0, le=1.0, description="Target accuracy requirement"
    )

    class Config:
        schema_extra = {
            "example": {
                "geological_process": "volcanic_eruption",
                "data_info": {
                    "seismic_data": {"points": 10000, "quality": 0.85},
                    "deformation_data": {"points": 5000, "quality": 0.90},
                    "thermal_data": {"points": 2000, "quality": 0.75},
                },
                "target_accuracy": 0.85,
            }
        }

class ModelSelectionRequest(BaseModel):
    """Request model for optimal model selection"""

    data_assessment: Dict[str, Any] = Field(..., description="Data assessment results")
    resource_constraints: Optional[Dict[str, Any]] = Field(
        None, description="Resource constraints"
    )
    target_accuracy: float = Field(
        0.8, ge=0.0, le=1.0, description="Target accuracy requirement"
    )

class ComplexityAssessmentResponse(BaseModel):
    """Response model for complexity assessment"""

    status: str
    data_assessment: Dict[str, Any]
    resource_assessment: Dict[str, Any]
    complexity_assessments: Dict[str, Any]
    optimal_configuration: Dict[str, Any]
    cost_breakdown: Dict[str, float]
    recommendations: List[str]

class ModelHierarchyResponse(BaseModel):
    """Response model for model hierarchy"""

    hierarchy: Dict[str, Dict[str, Any]]
    description: str

@router.post("/assess", response_model=ComplexityAssessmentResponse)
async def assess_simulation_complexity(request: ComplexityAssessmentRequest):
    """
    Assess the complexity requirements for a geological simulation

    This endpoint analyzes the available data and computational resources
    to recommend the optimal model complexity level.
    """
    try:
        logger.info(f"Assessing complexity for {request.geological_process}")

        # Convert data info to torch tensors (simplified)
        data_tensors = {}
        for data_type, info in request.data_info.items():
            # Create dummy tensor based on data info
            points = info.get("points", 1000)
            quality = info.get("quality", 0.8)

            # Generate synthetic data for assessment
            tensor_data = torch.randn(points, 10) * quality
            data_tensors[data_type] = tensor_data

        # Perform complexity assessment
        assessment_result = complexity_manager.assess_simulation_complexity(
            geological_process=GeologicalProcess(request.geological_process),
            data=data_tensors,
            target_accuracy=request.target_accuracy,
        )

        # Convert result to serializable format
        serializable_result = complexity_manager._make_serializable(assessment_result)

        return ComplexityAssessmentResponse(
            status="success",
            data_assessment=serializable_result["data_assessment"],
            resource_assessment=serializable_result["resource_assessment"],
            complexity_assessments=serializable_result["complexity_assessments"],
            optimal_configuration=serializable_result["optimal_configuration"],
            cost_breakdown=serializable_result["cost_breakdown"],
            recommendations=serializable_result["recommendations"],
        )

    except Exception as e:
        logger.error(f"Complexity assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

@router.post("/select-model")
async def select_optimal_model(request: ModelSelectionRequest):
    """
    Select optimal model configuration based on data and resource constraints
    """
    try:
        # Convert request data to assessment objects
        from core.model_complexity_manager import DataAssessment, ResourceAssessment

        # Create data assessment from request
        data_assessment = DataAssessment(
            total_data_points=request.data_assessment.get("total_data_points", 1000),
            data_quality_score=request.data_assessment.get("data_quality_score", 0.8),
            temporal_coverage=request.data_assessment.get("temporal_coverage", 0.7),
            spatial_coverage=request.data_assessment.get("spatial_coverage", 0.7),
            data_types_available=request.data_assessment.get(
                "data_types_available", []
            ),
            missing_data_fraction=request.data_assessment.get(
                "missing_data_fraction", 0.1
            ),
            noise_level=request.data_assessment.get("noise_level", 0.2),
        )

        # Get current resource assessment or use provided constraints
        if request.resource_constraints:
            resource_assessment = ResourceAssessment(
                cpu_cores=request.resource_constraints.get("cpu_cores", 8),
                memory_gb=request.resource_constraints.get("memory_gb", 16.0),
                gpu_count=request.resource_constraints.get("gpu_count", 0),
                gpu_memory_gb=request.resource_constraints.get("gpu_memory_gb", 0.0),
                storage_gb=request.resource_constraints.get("storage_gb", 100.0),
                network_bandwidth_mbps=request.resource_constraints.get(
                    "network_bandwidth_mbps", 1000.0
                ),
            )
        else:
            resource_assessment = (
                complexity_manager.resource_assessor.assess_resources()
            )

        # Select optimal model
        optimal_config = complexity_manager.select_optimal_model(
            data_assessment, resource_assessment, request.target_accuracy
        )

        # Estimate costs
        cost_breakdown = complexity_manager.cost_estimator.estimate_cost(
            optimal_config, data_assessment.total_data_points, resource_assessment
        )

        return {
            "status": "success",
            "optimal_configuration": optimal_config.to_dict(),
            "cost_breakdown": cost_breakdown,
            "data_level": data_assessment.availability_level().value,
            "resource_level": resource_assessment.resource_level().value,
        }

    except Exception as e:
        logger.error(f"Model selection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model selection failed: {str(e)}")

@router.get("/hierarchy", response_model=ModelHierarchyResponse)
async def get_model_hierarchy():
    """
    Get the complete model complexity hierarchy with metrics
    """
    try:
        hierarchy = complexity_manager.get_complexity_hierarchy()

        return ModelHierarchyResponse(
            hierarchy=hierarchy,
            description="Complete model complexity hierarchy from minimal to full complexity",
        )

    except Exception as e:
        logger.error(f"Failed to get model hierarchy: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get hierarchy: {str(e)}"
        )

@router.get("/resources")
async def assess_current_resources():
    """
    Assess current computational resources
    """
    try:
        resource_assessment = complexity_manager.resource_assessor.assess_resources()

        return {
            "status": "success",
            "resources": {
                "cpu_cores": resource_assessment.cpu_cores,
                "memory_gb": resource_assessment.memory_gb,
                "gpu_count": resource_assessment.gpu_count,
                "gpu_memory_gb": resource_assessment.gpu_memory_gb,
                "storage_gb": resource_assessment.storage_gb,
                "network_bandwidth_mbps": resource_assessment.network_bandwidth_mbps,
                "resource_level": resource_assessment.resource_level().value,
            },
        }

    except Exception as e:
        logger.error(f"Resource assessment failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Resource assessment failed: {str(e)}"
        )

@router.post("/estimate-cost")
async def estimate_computational_cost(
    complexity_level: str,
    data_size: int = 10000,
    resource_constraints: Optional[Dict[str, Any]] = None,
):
    """
    Estimate computational cost for a specific complexity level
    """
    try:
        # Validate complexity level
        try:
            level = ComplexityLevel(complexity_level)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid complexity level: {complexity_level}"
            )

        # Get model configuration
        config = complexity_manager.model_hierarchy.get_config(level)

        # Get resource assessment
        if resource_constraints:
            from core.model_complexity_manager import ResourceAssessment

            resource_assessment = ResourceAssessment(
                cpu_cores=resource_constraints.get("cpu_cores", 8),
                memory_gb=resource_constraints.get("memory_gb", 16.0),
                gpu_count=resource_constraints.get("gpu_count", 0),
                gpu_memory_gb=resource_constraints.get("gpu_memory_gb", 0.0),
                storage_gb=resource_constraints.get("storage_gb", 100.0),
                network_bandwidth_mbps=resource_constraints.get(
                    "network_bandwidth_mbps", 1000.0
                ),
            )
        else:
            resource_assessment = (
                complexity_manager.resource_assessor.assess_resources()
            )

        # Estimate cost
        cost_breakdown = complexity_manager.cost_estimator.estimate_cost(
            config, data_size, resource_assessment
        )

        return {
            "status": "success",
            "complexity_level": complexity_level,
            "configuration": config.to_dict(),
            "cost_breakdown": cost_breakdown,
            "data_size": data_size,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")

@router.get("/recommendations/{geological_process}")
async def get_complexity_recommendations(
    geological_process: str,
    data_quality: float = 0.8,
    data_size: int = 10000,
    target_accuracy: float = 0.8,
):
    """
    Get complexity recommendations for a specific geological process
    """
    try:
        # Create synthetic data assessment
        from core.model_complexity_manager import DataAssessment

        data_assessment = DataAssessment(
            total_data_points=data_size,
            data_quality_score=data_quality,
            temporal_coverage=0.7,
            spatial_coverage=0.7,
            data_types_available=["seismic", "deformation", "thermal"],
            missing_data_fraction=0.1,
            noise_level=0.2,
        )

        # Get resource assessment
        resource_assessment = complexity_manager.resource_assessor.assess_resources()

        # Select optimal model
        optimal_config = complexity_manager.select_optimal_model(
            data_assessment, resource_assessment, target_accuracy
        )

        # Generate recommendations
        recommendations = complexity_manager._generate_recommendations(
            data_assessment, resource_assessment, optimal_config
        )

        return {
            "status": "success",
            "geological_process": geological_process,
            "recommended_complexity": optimal_config.complexity_level.value,
            "configuration": optimal_config.to_dict(),
            "recommendations": recommendations,
            "data_level": data_assessment.availability_level().value,
            "resource_level": resource_assessment.resource_level().value,
        }

    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get recommendations: {str(e)}"
        )

@router.post("/save-assessment")
async def save_complexity_assessment(
    assessment_data: Dict[str, Any], filename: str = "complexity_assessment.json"
):
    """
    Save complexity assessment results to file
    """
    try:
        filepath = f"model_storage/{filename}"
        complexity_manager.save_assessment(assessment_data, filepath)

        return {
            "status": "success",
            "message": f"Assessment saved to {filepath}",
            "filepath": filepath,
        }

    except Exception as e:
        logger.error(f"Failed to save assessment: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save assessment: {str(e)}"
        )
