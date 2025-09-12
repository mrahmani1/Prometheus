"""
Dataproc API Router
REST API endpoints for Dataproc Spark geological analytics
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional, Any
import structlog

from ..models.gcp_models import *
from ..core.dataproc_analytics import DataprocSparkAnalytics

logger = structlog.get_logger(__name__)
router = APIRouter()

# Dependency to get Dataproc service
async def get_dataproc_service() -> DataprocSparkAnalytics:
    from ..main import dataproc_service
    if not dataproc_service:
        raise HTTPException(status_code=503, detail="Dataproc service not available")
    return dataproc_service


@router.get("/status")
async def get_dataproc_status(
    service: DataprocSparkAnalytics = Depends(get_dataproc_service)
):
    """Get Dataproc service status"""
    try:
        status = await service.get_status()
        return ServiceStatusResponse(
            service="Dataproc Spark Analytics",
            status=status["status"],
            timestamp=status["timestamp"],
            details=status
        )
    except Exception as e:
        logger.error("Error getting Dataproc status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")


@router.post("/analyze")
async def run_geological_analytics(
    dataset_id: str,
    analysis_type: str,
    parameters: Optional[Dict[str, Any]] = None,
    service: DataprocSparkAnalytics = Depends(get_dataproc_service)
):
    """Run geological analytics using Spark on Dataproc"""
    try:
        logger.info("Starting Dataproc geological analytics", 
                   dataset_id=dataset_id, analysis_type=analysis_type)
        
        result = await service.run_geological_analytics(
            dataset_id,
            analysis_type,
            parameters or {}
        )
        
        return AnalysisResponse(
            job_id=result["job_id"],
            status=result["status"],
            message="Dataproc analysis started successfully",
            results=result
        )
        
    except Exception as e:
        logger.error("Error starting Dataproc analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/clusters")
async def list_clusters(
    service: DataprocSparkAnalytics = Depends(get_dataproc_service)
):
    """List Dataproc clusters"""
    try:
        clusters = await service.list_clusters()
        return {"clusters": clusters}
        
    except Exception as e:
        logger.error("Error listing clusters", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error listing clusters: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    service: DataprocSparkAnalytics = Depends(get_dataproc_service)
):
    """Get Dataproc job status"""
    try:
        status = await service.get_job_status(job_id)
        return JobStatusResponse(
            job_id=status["job_id"],
            job_name=job_id,
            status=status["status"],
            create_time=None,
            start_time=None
        )
        
    except Exception as e:
        logger.error("Error getting job status", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")


@router.get("/spark-jobs")
async def list_spark_job_templates(
    service: DataprocSparkAnalytics = Depends(get_dataproc_service)
):
    """List available Spark job templates"""
    try:
        return {
            "spark_jobs": list(service.spark_jobs.keys()),
            "job_descriptions": {
                "seismic_analysis": "Seismic data analysis and event detection",
                "mineral_prospectivity": "Mineral prospectivity mapping using ML",
                "geological_clustering": "Geological data clustering analysis",
                "time_series_analysis": "Time series analysis and forecasting",
                "spatial_analysis": "Spatial analysis using GeoSpark"
            }
        }
        
    except Exception as e:
        logger.error("Error listing Spark jobs", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error listing Spark jobs: {str(e)}")


@router.get("/cluster-configs")
async def get_cluster_configurations(
    service: DataprocSparkAnalytics = Depends(get_dataproc_service)
):
    """Get cluster configuration information"""
    try:
        return {
            "available_clusters": list(service.cluster_configs.keys()),
            "cluster_purposes": {
                "geological-analytics": "General geological data analytics",
                "seismic-processing": "High-memory seismic data processing",
                "mineral-analysis": "Mineral analysis with ML capabilities",
                "large-scale-modeling": "Large-scale geological modeling"
            }
        }
        
    except Exception as e:
        logger.error("Error getting cluster configs", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting cluster configs: {str(e)}")