"""
Dataplex API Router
REST API endpoints for Google Cloud Dataplex unified data management
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional, Any
import structlog

from ..models.gcp_models import *

logger = structlog.get_logger(__name__)
router = APIRouter()

# Dependency to get Dataplex service
async def get_dataplex_service():
    from ..main import dataplex_service
    if not dataplex_service:
        raise HTTPException(status_code=503, detail="Dataplex service not available")
    return dataplex_service


@router.get("/status")
async def get_dataplex_status(service = Depends(get_dataplex_service)):
    """Get Dataplex service status"""
    try:
        status = await service.get_status()
        return ServiceStatusResponse(
            service="Google Cloud Dataplex",
            status=status["status"],
            timestamp=status["timestamp"],
            details=status
        )
    except Exception as e:
        logger.error("Error getting Dataplex status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")


@router.get("/lakes")
async def list_data_lakes(service = Depends(get_dataplex_service)):
    """List Dataplex data lakes"""
    try:
        return {"lakes": [], "message": "Dataplex data lake listing not fully implemented"}
    except Exception as e:
        logger.error("Error listing data lakes", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error listing data lakes: {str(e)}")


@router.get("/assets")
async def list_assets(service = Depends(get_dataplex_service)):
    """List Dataplex assets"""
    try:
        return {"assets": [], "message": "Dataplex asset listing not fully implemented"}
    except Exception as e:
        logger.error("Error listing assets", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error listing assets: {str(e)}")