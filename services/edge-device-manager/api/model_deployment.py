"""
API endpoints for model deployment to edge devices
"""

import logging
from typing import List, Optional

from core.device_manager import EdgeDeviceManager
from core.model_deployer import ModelDeployer
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from models.device import (
    ModelDeployment,
    ModelDeploymentRequest,
    ModelDeploymentResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependencies
async def get_model_deployer() -> ModelDeployer:
    from main import model_deployer

    if not model_deployer:
        raise HTTPException(status_code=503, detail="Model deployer not available")
    return model_deployer

async def get_device_manager() -> EdgeDeviceManager:
    from main import device_manager

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not available")
    return device_manager

@router.post("/deploy", response_model=ModelDeploymentResponse)
async def deploy_model(
    request: ModelDeploymentRequest,
    background_tasks: BackgroundTasks,
    deployer: ModelDeployer = Depends(get_model_deployer),
    device_manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """Deploy a model to specified edge devices"""
    try:
        response = await deployer.deploy_model(request, device_manager)
        logger.info(f"Model deployment initiated: {response.deployment_id}")
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/deployments/{deployment_id}")
async def get_deployment_status(
    deployment_id: str, deployer: ModelDeployer = Depends(get_model_deployer)
):
    """Get status of a model deployment"""
    status = await deployer.get_deployment_status(deployment_id)
    if not status:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return status

@router.get("/deployments", response_model=List[ModelDeployment])
async def list_deployments(
    device_id: Optional[str] = None,
    deployer: ModelDeployer = Depends(get_model_deployer),
):
    """List model deployments, optionally filtered by device"""
    try:
        deployments = await deployer.list_deployments(device_id=device_id)
        return deployments
    except Exception as e:
        logger.error(f"Failed to list deployments: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/deployments/{deployment_id}/devices/{device_id}/stop")
async def stop_deployment(
    deployment_id: str,
    device_id: str,
    deployer: ModelDeployer = Depends(get_model_deployer),
):
    """Stop a running model deployment"""
    try:
        await deployer.stop_deployment(deployment_id, device_id)
        return {
            "status": "success",
            "message": f"Deployment {deployment_id} stopped on device {device_id}",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(
            f"Failed to stop deployment {deployment_id} on device {device_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/deployments/{deployment_id}/devices/{device_id}")
async def remove_deployment(
    deployment_id: str,
    device_id: str,
    background_tasks: BackgroundTasks,
    deployer: ModelDeployer = Depends(get_model_deployer),
):
    """Remove a model deployment from device"""
    try:
        # Remove deployment in background to avoid blocking
        background_tasks.add_task(deployer.remove_deployment, deployment_id, device_id)
        return {
            "status": "success",
            "message": f"Deployment {deployment_id} removal initiated for device {device_id}",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"Failed to remove deployment {deployment_id} from device {device_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/devices/{device_id}/deployments")
async def get_device_deployments(
    device_id: str, deployer: ModelDeployer = Depends(get_model_deployer)
):
    """Get all deployments for a specific device"""
    try:
        deployments = await deployer.list_deployments(device_id=device_id)
        return {
            "device_id": device_id,
            "deployments": deployments,
            "total_deployments": len(deployments),
        }
    except Exception as e:
        logger.error(f"Failed to get deployments for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/devices/{device_id}/models/{model_id}/restart")
async def restart_model(
    device_id: str, model_id: str, deployer: ModelDeployer = Depends(get_model_deployer)
):
    """Restart a specific model on a device"""
    try:
        # Find deployment for this model on this device
        deployments = await deployer.list_deployments(device_id=device_id)
        target_deployment = None

        for deployment in deployments:
            if deployment.model_id == model_id:
                target_deployment = deployment
                break

        if not target_deployment:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found on device {device_id}",
            )

        # Stop and restart (simplified - would implement actual restart logic)
        await deployer.stop_deployment(
            target_deployment.deployment_id.split("-")[0], device_id
        )

        return {
            "status": "success",
            "message": f"Model {model_id} restart initiated on device {device_id}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart model {model_id} on device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/models/{model_id}/deployments")
async def get_model_deployments(
    model_id: str, deployer: ModelDeployer = Depends(get_model_deployer)
):
    """Get all deployments for a specific model"""
    try:
        all_deployments = await deployer.list_deployments()
        model_deployments = [d for d in all_deployments if d.model_id == model_id]

        return {
            "model_id": model_id,
            "deployments": model_deployments,
            "total_deployments": len(model_deployments),
            "devices": list(set(d.device_id for d in model_deployments)),
        }
    except Exception as e:
        logger.error(f"Failed to get deployments for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/deployments/{deployment_id}/rollback")
async def rollback_deployment(
    deployment_id: str, deployer: ModelDeployer = Depends(get_model_deployer)
):
    """Rollback a deployment to previous version"""
    try:
        # This would implement actual rollback logic
        # For now, just return success message
        return {
            "status": "success",
            "message": f"Rollback initiated for deployment {deployment_id}",
            "deployment_id": deployment_id,
        }
    except Exception as e:
        logger.error(f"Failed to rollback deployment {deployment_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/deployment-metrics")
async def get_deployment_metrics(deployer: ModelDeployer = Depends(get_model_deployer)):
    """Get deployment metrics and statistics"""
    try:
        # Get basic deployment statistics
        all_deployments = await deployer.list_deployments()

        total_deployments = len(all_deployments)
        running_deployments = len(
            [d for d in all_deployments if d.status.value == "running"]
        )
        failed_deployments = len(
            [d for d in all_deployments if d.status.value == "failed"]
        )
        pending_deployments = len(
            [d for d in all_deployments if d.status.value == "pending"]
        )

        # Group by model
        models = {}
        for deployment in all_deployments:
            model_id = deployment.model_id
            if model_id not in models:
                models[model_id] = {"total": 0, "running": 0, "failed": 0}
            models[model_id]["total"] += 1
            if deployment.status.value == "running":
                models[model_id]["running"] += 1
            elif deployment.status.value == "failed":
                models[model_id]["failed"] += 1

        return {
            "total_deployments": total_deployments,
            "running_deployments": running_deployments,
            "failed_deployments": failed_deployments,
            "pending_deployments": pending_deployments,
            "success_rate": (
                (running_deployments / total_deployments * 100)
                if total_deployments > 0
                else 0
            ),
            "models": models,
            "last_updated": "2024-01-01T00:00:00Z",  # Would use actual timestamp
        }
    except Exception as e:
        logger.error(f"Failed to get deployment metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
