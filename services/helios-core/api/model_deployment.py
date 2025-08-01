"""
Model Deployment API Endpoints

This module provides REST API endpoints for the global model deployment system,
including model versioning, deployment management, performance monitoring,
and drift detection.

Requirements addressed: 5.3, 5.4, 5.5
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.global_model_deployment import GlobalModelDeploymentSystem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/model-deployment", tags=["model-deployment"])

# Global deployment system instance
deployment_system = GlobalModelDeploymentSystem()

# Pydantic models for API
class ModelRegistrationRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    training_data_hash: str = Field(..., description="Hash of training data")
    performance_metrics: Dict[str, float] = Field(
        ..., description="Model performance metrics"
    )
    geological_domains: List[str] = Field(
        ..., description="Applicable geological domains"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class DeploymentRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to deploy")
    region_name: str = Field(..., description="Target region name")
    geological_domain: str = Field(..., description="Geological domain")
    endpoint_url: str = Field(..., description="Deployment endpoint URL")
    deployment_type: str = Field("cloud", description="Deployment type")
    auto_update: bool = Field(True, description="Enable automatic updates")

class DeploymentUpdateRequest(BaseModel):
    new_version_id: str = Field(..., description="New model version ID")

class AutoDeploymentRequest(BaseModel):
    model_name: str = Field(..., description="Model name")
    target_domains: List[str] = Field(..., description="Target geological domains")

class PerformanceThresholdRequest(BaseModel):
    accuracy_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum accuracy threshold"
    )
    latency_threshold_ms: float = Field(
        1000, gt=0, description="Maximum latency threshold"
    )
    error_rate_threshold: float = Field(
        0.1, ge=0.0, le=1.0, description="Maximum error rate threshold"
    )
    drift_threshold: float = Field(
        0.3, ge=0.0, le=1.0, description="Drift detection threshold"
    )

# Dependency to get Helios engine
async def get_helios_engine():
    """Dependency to get the Helios engine instance"""
    from main import helios_engine

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

    if not helios_engine or not await helios_engine.is_ready():
        raise HTTPException(status_code=503, detail="Helios engine not ready")
    return helios_engine

@router.post("/models/register")
async def register_model(
    request: ModelRegistrationRequest, helios_engine=Depends(get_helios_engine)
):
    """Register a new model version for deployment"""
    try:
        # Get the model from Helios engine
        model = helios_engine.get_pinn_model()
        if model is None:
            raise HTTPException(
                status_code=404, detail="Model not found in Helios engine"
            )

        # Register the model
        version_id = await deployment_system.register_model(
            model=model,
            model_name=request.model_name,
            training_data_hash=request.training_data_hash,
            performance_metrics=request.performance_metrics,
            geological_domains=request.geological_domains,
            metadata=request.metadata,
        )

        logger.info(
            f"Registered model {request.model_name} with version ID {version_id}"
        )

        return {
            "version_id": version_id,
            "model_name": request.model_name,
            "status": "registered",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to register model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deployments/deploy")
async def deploy_model(request: DeploymentRequest, background_tasks: BackgroundTasks):
    """Deploy a model to a geological region"""
    try:
        deployment_id = await deployment_system.deploy_to_region(
            model_name=request.model_name,
            region_name=request.region_name,
            geological_domain=request.geological_domain,
            endpoint_url=request.endpoint_url,
            deployment_type=request.deployment_type,
            auto_update=request.auto_update,
        )

        logger.info(f"Started deployment {deployment_id} for {request.model_name}")

        return {
            "deployment_id": deployment_id,
            "model_name": request.model_name,
            "region_name": request.region_name,
            "geological_domain": request.geological_domain,
            "status": "deploying",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to deploy model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deployments/{deployment_id}/status")
async def get_deployment_status(deployment_id: str):
    """Get detailed status of a deployment"""
    try:
        status = await deployment_system.get_deployment_status(deployment_id)

        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get deployment status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deployments")
async def list_deployments():
    """List all deployments"""
    try:
        deployments = await deployment_system.list_deployments()

        return {
            "deployments": deployments,
            "total_count": len(deployments),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to list deployments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/deployments/{deployment_id}/update")
async def update_deployment(deployment_id: str, request: DeploymentUpdateRequest):
    """Update a deployment to a new model version"""
    try:
        success = await deployment_system.update_deployment(
            deployment_id, request.new_version_id
        )

        if not success:
            raise HTTPException(status_code=404, detail="Deployment not found")

        return {
            "deployment_id": deployment_id,
            "new_version_id": request.new_version_id,
            "status": "updating",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update deployment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deployments/{deployment_id}/rollback")
async def rollback_deployment(deployment_id: str):
    """Rollback a deployment to the previous version"""
    try:
        success = await deployment_system.orchestrator.rollback_deployment(
            deployment_id
        )

        if not success:
            raise HTTPException(
                status_code=404,
                detail="Deployment not found or no rollback version available",
            )

        return {
            "deployment_id": deployment_id,
            "status": "rolling_back",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rollback deployment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deployments/overview")
async def get_global_deployment_overview():
    """Get overview of all global deployments"""
    try:
        overview = await deployment_system.get_global_deployment_overview()
        return overview

    except Exception as e:
        logger.error(f"Failed to get deployment overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}/versions")
async def list_model_versions(model_name: str):
    """List all versions of a model"""
    try:
        versions = await deployment_system.version_manager.list_versions(model_name)

        version_list = []
        for version in versions:
            version_info = {
                "version_id": version.version_id,
                "version_number": version.version_number,
                "creation_timestamp": version.creation_timestamp.isoformat(),
                "model_hash": version.model_hash,
                "model_size_bytes": version.model_size_bytes,
                "performance_metrics": version.performance_metrics,
                "geological_domains": version.geological_domains,
            }
            version_list.append(version_info)

        return {
            "model_name": model_name,
            "versions": version_list,
            "total_versions": len(version_list),
        }

    except Exception as e:
        logger.error(f"Failed to list model versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}/latest")
async def get_latest_model_version(model_name: str):
    """Get the latest version of a model"""
    try:
        latest_version = await deployment_system.version_manager.get_latest_version(
            model_name
        )

        if not latest_version:
            raise HTTPException(
                status_code=404, detail=f"No versions found for model {model_name}"
            )

        return {
            "version_id": latest_version.version_id,
            "version_number": latest_version.version_number,
            "creation_timestamp": latest_version.creation_timestamp.isoformat(),
            "model_hash": latest_version.model_hash,
            "performance_metrics": latest_version.performance_metrics,
            "geological_domains": latest_version.geological_domains,
            "metadata": latest_version.metadata,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest model version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_name}/versions/{version_id}")
async def delete_model_version(model_name: str, version_id: str):
    """Delete a specific model version"""
    try:
        success = await deployment_system.version_manager.storage.delete_version(
            version_id
        )

        if not success:
            raise HTTPException(status_code=404, detail="Model version not found")

        return {
            "version_id": version_id,
            "model_name": model_name,
            "status": "deleted",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/{deployment_id}/metrics")
async def get_deployment_metrics(deployment_id: str, hours: int = 24):
    """Get performance metrics for a deployment"""
    try:
        metrics = await deployment_system.performance_monitor.get_metrics_history(
            deployment_id, hours
        )

        metrics_data = []
        for metric in metrics:
            metric_data = {
                "timestamp": metric.timestamp.isoformat(),
                "accuracy": metric.accuracy,
                "latency_ms": metric.latency_ms,
                "throughput_rps": metric.throughput_rps,
                "memory_usage_mb": metric.memory_usage_mb,
                "cpu_usage_percent": metric.cpu_usage_percent,
                "error_rate": metric.error_rate,
                "drift_score": metric.drift_score,
                "geological_consistency": metric.geological_consistency,
            }
            metrics_data.append(metric_data)

        return {
            "deployment_id": deployment_id,
            "metrics": metrics_data,
            "total_metrics": len(metrics_data),
            "time_range_hours": hours,
        }

    except Exception as e:
        logger.error(f"Failed to get deployment metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/{deployment_id}/performance")
async def get_performance_summary(deployment_id: str):
    """Get performance summary for a deployment"""
    try:
        summary = await deployment_system.performance_monitor.get_performance_summary(
            deployment_id
        )
        return summary

    except Exception as e:
        logger.error(f"Failed to get performance summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/{deployment_id}/drift")
async def check_model_drift(deployment_id: str):
    """Check for model drift in a deployment"""
    try:
        drift_info = await deployment_system.drift_detector.detect_drift(deployment_id)
        return drift_info

    except Exception as e:
        logger.error(f"Failed to check model drift: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitoring/{deployment_id}/baseline")
async def establish_performance_baseline(deployment_id: str):
    """Establish performance baseline for drift detection"""
    try:
        await deployment_system.drift_detector.establish_baseline(deployment_id)

        return {
            "deployment_id": deployment_id,
            "status": "baseline_established",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to establish baseline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/monitoring/thresholds")
async def update_performance_thresholds(request: PerformanceThresholdRequest):
    """Update performance monitoring thresholds"""
    try:
        # Update thresholds in performance monitor
        deployment_system.performance_monitor.alert_thresholds.update(
            {
                "accuracy": request.accuracy_threshold,
                "latency_ms": request.latency_threshold_ms,
                "error_rate": request.error_rate_threshold,
            }
        )

        # Update drift detection thresholds
        deployment_system.drift_detector.drift_thresholds.update(
            {"drift_score": request.drift_threshold}
        )

        return {
            "status": "thresholds_updated",
            "new_thresholds": {
                "accuracy": request.accuracy_threshold,
                "latency_ms": request.latency_threshold_ms,
                "error_rate": request.error_rate_threshold,
                "drift_score": request.drift_threshold,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to update thresholds: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/automation/enable-auto-deployment")
async def enable_auto_deployment(request: AutoDeploymentRequest):
    """Enable automatic deployment for new model versions"""
    try:
        await deployment_system.enable_auto_deployment(
            request.model_name, request.target_domains
        )

        return {
            "model_name": request.model_name,
            "target_domains": request.target_domains,
            "status": "auto_deployment_enabled",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to enable auto-deployment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/automation/{deployment_id}/trigger-retraining")
async def trigger_model_retraining(deployment_id: str):
    """Manually trigger model retraining due to drift"""
    try:
        # Get drift information
        drift_info = await deployment_system.drift_detector.detect_drift(deployment_id)

        # Trigger retraining
        success = await deployment_system.drift_detector.trigger_retraining(
            deployment_id, drift_info
        )

        if not success:
            return {
                "deployment_id": deployment_id,
                "status": "retraining_skipped",
                "reason": "recently_triggered",
                "timestamp": datetime.utcnow().isoformat(),
            }

        return {
            "deployment_id": deployment_id,
            "status": "retraining_triggered",
            "drift_info": drift_info,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to trigger retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/maintenance/cleanup-versions")
async def cleanup_old_versions(model_name: str, keep_versions: int = 5):
    """Clean up old model versions"""
    try:
        await deployment_system.cleanup_old_versions(model_name, keep_versions)

        return {
            "model_name": model_name,
            "keep_versions": keep_versions,
            "status": "cleanup_completed",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to cleanup versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/targets")
async def list_deployment_targets():
    """List all registered deployment targets"""
    try:
        targets = []
        for (
            target_id,
            target,
        ) in deployment_system.orchestrator.deployment_targets.items():
            target_info = {
                "target_id": target.target_id,
                "region_name": target.region_name,
                "geological_domain": target.geological_domain,
                "endpoint_url": target.endpoint_url,
                "deployment_type": target.deployment_type,
                "auto_update_enabled": target.auto_update_enabled,
                "monitoring_enabled": target.monitoring_enabled,
            }
            targets.append(target_info)

        return {"targets": targets, "total_targets": len(targets)}

    except Exception as e:
        logger.error(f"Failed to list deployment targets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/status")
async def get_system_status():
    """Get overall system status"""
    try:
        overview = await deployment_system.get_global_deployment_overview()

        # Add system health information
        system_status = {
            "deployment_system": {
                "status": "healthy",
                "auto_deployment_enabled": deployment_system.auto_deployment_enabled,
                "auto_retraining_enabled": deployment_system.auto_retraining_enabled,
            },
            "components": {
                "version_manager": "healthy",
                "orchestrator": "healthy",
                "performance_monitor": "healthy",
                "drift_detector": "healthy",
            },
            "deployment_overview": overview,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return system_status

    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for model deployment service"""
    return {
        "status": "healthy",
        "service": "model-deployment",
        "components": {
            "deployment_system": deployment_system is not None,
            "version_manager": deployment_system.version_manager is not None,
            "orchestrator": deployment_system.orchestrator is not None,
            "performance_monitor": deployment_system.performance_monitor is not None,
            "drift_detector": deployment_system.drift_detector is not None,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
