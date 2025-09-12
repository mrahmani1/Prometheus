"""
Data Fusion API Router
REST API endpoints for Google Cloud Data Fusion visual data pipelines
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional, Any
import structlog

from ..models.gcp_models import *

logger = structlog.get_logger(__name__)
router = APIRouter()

# Dependency to get Data Fusion service
async def get_data_fusion_service():
    from ..main import data_fusion_service
    if not data_fusion_service:
        raise HTTPException(status_code=503, detail="Data Fusion service not available")
    return data_fusion_service


@router.get("/status")
async def get_data_fusion_status(service = Depends(get_data_fusion_service)):
    """Get Data Fusion service status"""
    try:
        status = await service.get_status()
        return ServiceStatusResponse(
            service="Google Cloud Data Fusion",
            status=status["status"],
            timestamp=status["timestamp"],
            details=status
        )
    except Exception as e:
        logger.error("Error getting Data Fusion status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")


@router.post("/pipelines")
async def create_pipeline(
    request: DataFusionPipelineRequest,
    service = Depends(get_data_fusion_service)
):
    """Create Data Fusion pipeline"""
    try:
        # Placeholder implementation
        return {
            "pipeline_id": f"pipeline_{request.pipeline_name}",
            "status": "created",
            "message": "Data Fusion pipeline creation not fully implemented"
        }
    except Exception as e:
        logger.error("Error creating pipeline", error=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline creation error: {str(e)}")


@router.get("/pipelines")
async def list_pipelines(service = Depends(get_data_fusion_service)):
    """List Data Fusion pipelines"""
    try:
        return {"pipelines": [], "message": "Data Fusion pipeline listing not fully implemented"}
    except Exception as e:
        logger.error("Error listing pipelines", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error listing pipelines: {str(e)}")