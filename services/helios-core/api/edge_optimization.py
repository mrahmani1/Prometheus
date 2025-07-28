"""
Edge AI Optimization API

This module provides REST API endpoints for edge AI model optimization,
automated edge model generation, and deployment management.

Requirements addressed: 10.1, 10.5
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import torch
import logging
from pathlib import Path
import asyncio
from datetime import datetime

from core.edge_ai_optimizer import (
    EdgeAIOptimizer, HardwareConstraints, OptimizationConfig, OptimizationResult
)
from core.automated_edge_generation import (
    AutomatedEdgeGenerator, EdgeDeploymentProfile, EdgeModelMetadata, HeliosEdgeModelFactory
)
from core.helios_engine import HeliosCore

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/edge", tags=["edge-optimization"])

# Global instances (would be dependency injected in production)
helios_core = None
edge_generator = None
edge_factory = None


# Pydantic models for API
class HardwareConstraintsModel(BaseModel):
    max_memory_mb: int = Field(..., description="Maximum memory in MB")
    max_inference_time_ms: float = Field(..., description="Maximum inference time in milliseconds")
    cpu_cores: int = Field(..., description="Number of CPU cores")
    has_gpu: bool = Field(default=False, description="Whether GPU is available")
    gpu_memory_mb: Optional[int] = Field(None, description="GPU memory in MB")
    target_platform: str = Field(default="cpu", description="Target platform")


class OptimizationConfigModel(BaseModel):
    compression_ratio: float = Field(default=0.5, ge=0.1, le=1.0, description="Compression ratio")
    quantization_bits: int = Field(default=8, ge=4, le=32, description="Quantization bits")
    pruning_sparsity: float = Field(default=0.3, ge=0.0, le=0.9, description="Pruning sparsity")
    enable_quantization: bool = Field(default=True, description="Enable quantization")
    enable_pruning: bool = Field(default=True, description="Enable pruning")
    enable_knowledge_distillation: bool = Field(default=True, description="Enable knowledge distillation")
    validation_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Validation threshold")


class OptimizationRequest(BaseModel):
    model_id: str = Field(..., description="Source model identifier")
    hardware_constraints: HardwareConstraintsModel
    optimization_config: OptimizationConfigModel
    profile_name: Optional[str] = Field(None, description="Custom profile name")


class EdgeModelResponse(BaseModel):
    model_id: str
    source_model_id: str
    deployment_profile: str
    creation_timestamp: datetime
    optimization_summary: Dict[str, Any]
    validation_status: str
    deployment_ready: bool
    download_url: Optional[str] = None


class DeploymentProfileModel(BaseModel):
    name: str
    description: str
    hardware_constraints: HardwareConstraintsModel
    optimization_config: OptimizationConfigModel
    target_use_cases: List[str]
    priority: int = Field(default=1, ge=1, le=3)


class ModelRecommendationRequest(BaseModel):
    use_case: str = Field(..., description="Target use case")
    hardware_constraints: Optional[HardwareConstraintsModel] = None


# Dependency injection
async def get_edge_generator():
    global edge_generator
    if edge_generator is None:
        storage_path = Path("model_storage/edge_models")
        edge_generator = AutomatedEdgeGenerator(helios_core, storage_path)
    return edge_generator


async def get_edge_factory():
    global edge_factory
    if edge_factory is None:
        storage_path = Path("model_storage/edge_models")
        edge_factory = HeliosEdgeModelFactory(helios_core, storage_path)
    return edge_factory


# API Endpoints

@router.post("/optimize", response_model=EdgeModelResponse)
async def optimize_model_for_edge(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    generator: AutomatedEdgeGenerator = Depends(get_edge_generator)
):
    """
    Optimize a model for edge deployment with custom constraints
    """
    try:
        logger.info(f"Starting edge optimization for model: {request.model_id}")
        
        # Convert Pydantic models to internal types
        constraints = HardwareConstraints(
            max_memory_mb=request.hardware_constraints.max_memory_mb,
            max_inference_time_ms=request.hardware_constraints.max_inference_time_ms,
            cpu_cores=request.hardware_constraints.cpu_cores,
            has_gpu=request.hardware_constraints.has_gpu,
            gpu_memory_mb=request.hardware_constraints.gpu_memory_mb,
            target_platform=request.hardware_constraints.target_platform
        )
        
        config = OptimizationConfig(
            compression_ratio=request.optimization_config.compression_ratio,
            quantization_bits=request.optimization_config.quantization_bits,
            pruning_sparsity=request.optimization_config.pruning_sparsity,
            enable_quantization=request.optimization_config.enable_quantization,
            enable_pruning=request.optimization_config.enable_pruning,
            enable_knowledge_distillation=request.optimization_config.enable_knowledge_distillation,
            validation_threshold=request.optimization_config.validation_threshold
        )
        
        # Get source model (simplified - would load from model registry)
        source_model = _get_model_by_id(request.model_id)
        if source_model is None:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        # Generate optimized model
        profile_name = request.profile_name or "custom"
        metadata = await generator.generate_custom_edge_model(
            source_model, request.model_id, constraints, config, profile_name
        )
        
        # Create response
        optimization_report = generator.get_optimization_report(metadata.model_id)
        
        return EdgeModelResponse(
            model_id=metadata.model_id,
            source_model_id=metadata.source_model_id,
            deployment_profile=metadata.deployment_profile,
            creation_timestamp=metadata.creation_timestamp,
            optimization_summary=optimization_report['optimization_summary'],
            validation_status=metadata.validation_status,
            deployment_ready=metadata.deployment_ready,
            download_url=f"/api/v1/edge/models/{metadata.model_id}/download"
        )
        
    except Exception as e:
        logger.error(f"Edge optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-all/{model_id}")
async def generate_all_edge_variants(
    model_id: str,
    background_tasks: BackgroundTasks,
    generator: AutomatedEdgeGenerator = Depends(get_edge_generator)
):
    """
    Generate edge-optimized variants for all deployment profiles
    """
    try:
        logger.info(f"Generating all edge variants for model: {model_id}")
        
        # Get source model
        source_model = _get_model_by_id(model_id)
        if source_model is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Generate all variants in background
        background_tasks.add_task(
            _generate_all_variants_background,
            generator, source_model, model_id
        )
        
        return {
            "message": f"Started generating edge variants for model {model_id}",
            "status": "in_progress",
            "check_status_url": f"/api/v1/edge/generation-status/{model_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start edge variant generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[EdgeModelResponse])
async def list_edge_models(
    profile: Optional[str] = None,
    deployment_ready_only: bool = True,
    generator: AutomatedEdgeGenerator = Depends(get_edge_generator)
):
    """
    List available edge-optimized models
    """
    try:
        if profile:
            models = generator.registry.list_models_by_profile(profile)
        else:
            models = list(generator.registry.models.values())
        
        if deployment_ready_only:
            models = [m for m in models if m.deployment_ready]
        
        response_models = []
        for model in models:
            optimization_report = generator.get_optimization_report(model.model_id)
            response_models.append(EdgeModelResponse(
                model_id=model.model_id,
                source_model_id=model.source_model_id,
                deployment_profile=model.deployment_profile,
                creation_timestamp=model.creation_timestamp,
                optimization_summary=optimization_report['optimization_summary'],
                validation_status=model.validation_status,
                deployment_ready=model.deployment_ready,
                download_url=f"/api/v1/edge/models/{model.model_id}/download"
            ))
        
        return response_models
        
    except Exception as e:
        logger.error(f"Failed to list edge models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}", response_model=EdgeModelResponse)
async def get_edge_model(
    model_id: str,
    generator: AutomatedEdgeGenerator = Depends(get_edge_generator)
):
    """
    Get details of a specific edge model
    """
    try:
        metadata = generator.registry.get_model(model_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Edge model {model_id} not found")
        
        optimization_report = generator.get_optimization_report(model_id)
        
        return EdgeModelResponse(
            model_id=metadata.model_id,
            source_model_id=metadata.source_model_id,
            deployment_profile=metadata.deployment_profile,
            creation_timestamp=metadata.creation_timestamp,
            optimization_summary=optimization_report['optimization_summary'],
            validation_status=metadata.validation_status,
            deployment_ready=metadata.deployment_ready,
            download_url=f"/api/v1/edge/models/{model_id}/download"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get edge model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/download")
async def download_edge_model(
    model_id: str,
    generator: AutomatedEdgeGenerator = Depends(get_edge_generator)
):
    """
    Download an edge-optimized model
    """
    try:
        metadata = generator.registry.get_model(model_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Edge model {model_id} not found")
        
        if not metadata.deployment_ready:
            raise HTTPException(status_code=400, detail="Model is not ready for deployment")
        
        model_path = generator.registry.storage_path / model_id / "optimized_model.pth"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        
        return FileResponse(
            path=str(model_path),
            filename=f"{model_id}.pth",
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download edge model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend", response_model=EdgeModelResponse)
async def recommend_edge_model(
    request: ModelRecommendationRequest,
    generator: AutomatedEdgeGenerator = Depends(get_edge_generator)
):
    """
    Get recommended edge model for specific use case and constraints
    """
    try:
        constraints = None
        if request.hardware_constraints:
            constraints = HardwareConstraints(
                max_memory_mb=request.hardware_constraints.max_memory_mb,
                max_inference_time_ms=request.hardware_constraints.max_inference_time_ms,
                cpu_cores=request.hardware_constraints.cpu_cores,
                has_gpu=request.hardware_constraints.has_gpu,
                gpu_memory_mb=request.hardware_constraints.gpu_memory_mb,
                target_platform=request.hardware_constraints.target_platform
            )
        
        metadata = generator.get_recommended_model(request.use_case, constraints)
        if not metadata:
            raise HTTPException(
                status_code=404, 
                detail=f"No suitable edge model found for use case: {request.use_case}"
            )
        
        optimization_report = generator.get_optimization_report(metadata.model_id)
        
        return EdgeModelResponse(
            model_id=metadata.model_id,
            source_model_id=metadata.source_model_id,
            deployment_profile=metadata.deployment_profile,
            creation_timestamp=metadata.creation_timestamp,
            optimization_summary=optimization_report['optimization_summary'],
            validation_status=metadata.validation_status,
            deployment_ready=metadata.deployment_ready,
            download_url=f"/api/v1/edge/models/{metadata.model_id}/download"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to recommend edge model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles", response_model=List[DeploymentProfileModel])
async def list_deployment_profiles(
    generator: AutomatedEdgeGenerator = Depends(get_edge_generator)
):
    """
    List available deployment profiles
    """
    try:
        profiles = []
        for profile in generator.deployment_profiles.values():
            profiles.append(DeploymentProfileModel(
                name=profile.name,
                description=profile.description,
                hardware_constraints=HardwareConstraintsModel(
                    max_memory_mb=profile.hardware_constraints.max_memory_mb,
                    max_inference_time_ms=profile.hardware_constraints.max_inference_time_ms,
                    cpu_cores=profile.hardware_constraints.cpu_cores,
                    has_gpu=profile.hardware_constraints.has_gpu,
                    gpu_memory_mb=profile.hardware_constraints.gpu_memory_mb,
                    target_platform=profile.hardware_constraints.target_platform
                ),
                optimization_config=OptimizationConfigModel(
                    compression_ratio=profile.optimization_config.compression_ratio,
                    quantization_bits=profile.optimization_config.quantization_bits,
                    pruning_sparsity=profile.optimization_config.pruning_sparsity,
                    enable_quantization=profile.optimization_config.enable_quantization,
                    enable_pruning=profile.optimization_config.enable_pruning,
                    enable_knowledge_distillation=profile.optimization_config.enable_knowledge_distillation,
                    validation_threshold=profile.optimization_config.validation_threshold
                ),
                target_use_cases=profile.target_use_cases,
                priority=profile.priority
            ))
        
        return profiles
        
    except Exception as e:
        logger.error(f"Failed to list deployment profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profiles", response_model=DeploymentProfileModel)
async def create_deployment_profile(
    profile: DeploymentProfileModel,
    generator: AutomatedEdgeGenerator = Depends(get_edge_generator)
):
    """
    Create or update a deployment profile
    """
    try:
        # Convert to internal types
        constraints = HardwareConstraints(
            max_memory_mb=profile.hardware_constraints.max_memory_mb,
            max_inference_time_ms=profile.hardware_constraints.max_inference_time_ms,
            cpu_cores=profile.hardware_constraints.cpu_cores,
            has_gpu=profile.hardware_constraints.has_gpu,
            gpu_memory_mb=profile.hardware_constraints.gpu_memory_mb,
            target_platform=profile.hardware_constraints.target_platform
        )
        
        config = OptimizationConfig(
            compression_ratio=profile.optimization_config.compression_ratio,
            quantization_bits=profile.optimization_config.quantization_bits,
            pruning_sparsity=profile.optimization_config.pruning_sparsity,
            enable_quantization=profile.optimization_config.enable_quantization,
            enable_pruning=profile.optimization_config.enable_pruning,
            enable_knowledge_distillation=profile.optimization_config.enable_knowledge_distillation,
            validation_threshold=profile.optimization_config.validation_threshold
        )
        
        edge_profile = EdgeDeploymentProfile(
            name=profile.name,
            description=profile.description,
            hardware_constraints=constraints,
            optimization_config=config,
            target_use_cases=profile.target_use_cases,
            priority=profile.priority
        )
        
        generator.update_deployment_profile(profile.name, edge_profile)
        
        return profile
        
    except Exception as e:
        logger.error(f"Failed to create deployment profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/{model_id}")
async def get_optimization_report(
    model_id: str,
    generator: AutomatedEdgeGenerator = Depends(get_edge_generator)
):
    """
    Get detailed optimization report for a model
    """
    try:
        report = generator.get_optimization_report(model_id)
        if not report:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions

def _get_model_by_id(model_id: str) -> Optional[torch.nn.Module]:
    """
    Get model by ID from model registry
    This is simplified - in practice would load from actual model storage
    """
    # For demo purposes, return a simple model
    if "pinn" in model_id.lower():
        return torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        )
    elif "cnn" in model_id.lower():
        return torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 3)
        )
    else:
        return None


async def _generate_all_variants_background(generator: AutomatedEdgeGenerator,
                                          source_model: torch.nn.Module,
                                          model_id: str):
    """Background task for generating all edge variants"""
    try:
        await generator.generate_all_edge_models(source_model, model_id)
        logger.info(f"Completed generating all edge variants for {model_id}")
    except Exception as e:
        logger.error(f"Background generation failed for {model_id}: {e}")


# Initialize global instances when module is imported
def initialize_edge_api(helios_instance: HeliosCore):
    """Initialize the edge API with Helios core instance"""
    global helios_core
    helios_core = helios_instance
    logger.info("Edge optimization API initialized")