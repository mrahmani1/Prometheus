"""
Dataflow API Router
REST API endpoints for Dataflow geological data processing
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
import structlog

from ..models.gcp_models import *
from ..core.dataflow_processor import DataflowGeologicalProcessor

logger = structlog.get_logger(__name__)
router = APIRouter()

# Dependency to get Dataflow service
async def get_dataflow_service() -> DataflowGeologicalProcessor:
    from ..main import dataflow_service
    if not dataflow_service:
        raise HTTPException(status_code=503, detail="Dataflow service not available")
    return dataflow_service


@router.get("/status")
async def get_dataflow_status(
    service: DataflowGeologicalProcessor = Depends(get_dataflow_service)
):
    """Get Dataflow service status"""
    try:
        status = await service.get_status()
        return ServiceStatusResponse(
            service="Dataflow Geological Processor",
            status=status["status"],
            timestamp=status["timestamp"],
            details=status
        )
    except Exception as e:
        logger.error("Error getting Dataflow status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")


@router.post("/process-stream")
async def process_geological_stream(
    request: DataflowJobRequest,
    service: DataflowGeologicalProcessor = Depends(get_dataflow_service)
):
    """Process geological data stream with Dataflow"""
    try:
        logger.info("Starting Dataflow stream processing", 
                   job_name=request.job_name, processing_type=request.processing_type)
        
        result = await service.process_geological_stream(
            request.input_source,
            {
                "processing_type": request.processing_type.value,
                **request.parameters
            }
        )
        
        return AnalysisResponse(
            job_id=result["job_id"],
            status=result["status"],
            message="Dataflow processing started successfully",
            results=result
        )
        
    except Exception as e:
        logger.error("Error starting Dataflow processing", error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/jobs")
async def list_active_jobs(
    service: DataflowGeologicalProcessor = Depends(get_dataflow_service)
):
    """List active Dataflow jobs"""
    try:
        jobs = await service.list_active_jobs()
        return {"jobs": jobs}
        
    except Exception as e:
        logger.error("Error listing Dataflow jobs", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error listing jobs: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    service: DataflowGeologicalProcessor = Depends(get_dataflow_service)
):
    """Get Dataflow job status"""
    try:
        status = await service.get_job_status(job_id)
        return JobStatusResponse(
            job_id=status["job_id"],
            job_name=status["job_name"],
            status=status["status"],
            create_time=status.get("create_time"),
            start_time=status.get("start_time")
        )
        
    except Exception as e:
        logger.error("Error getting job status", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    service: DataflowGeologicalProcessor = Depends(get_dataflow_service)
):
    """Cancel Dataflow job"""
    try:
        result = await service.cancel_job(job_id)
        return {
            "job_id": result["job_id"],
            "status": result["status"],
            "message": result["message"]
        }
        
    except Exception as e:
        logger.error("Error cancelling job", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error cancelling job: {str(e)}")


@router.get("/templates")
async def list_pipeline_templates(
    service: DataflowGeologicalProcessor = Depends(get_dataflow_service)
):
    """List available pipeline templates"""
    try:
        return {
            "templates": list(service.pipeline_templates.keys()),
            "template_details": {
                name: {"path": path, "description": f"Template for {name.replace('_', ' ')}"}
                for name, path in service.pipeline_templates.items()
            }
        }
        
    except Exception as e:
        logger.error("Error listing templates", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error listing templates: {str(e)}")


@router.get("/configurations")
async def get_job_configurations(
    service: DataflowGeologicalProcessor = Depends(get_dataflow_service)
):
    """Get available job configurations"""
    try:
        return {
            "configurations": service.job_configs
        }
        
    except Exception as e:
        logger.error("Error getting configurations", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting configurations: {str(e)}")