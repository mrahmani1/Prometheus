"""
Health Check API Router
Health check endpoints for GCP integration service
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "GCP Data Analytics Integration",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        # Import here to avoid circular imports
        from ..main import (
            bigquery_service, dataflow_service, dataproc_service,
            composer_service, data_fusion_service, looker_service,
            data_catalog_service, dataplex_service, earth_engine_service
        )
        
        services_status = {}
        all_healthy = True
        
        # Check each service
        services = {
            "bigquery": bigquery_service,
            "dataflow": dataflow_service,
            "dataproc": dataproc_service,
            "composer": composer_service,
            "data_fusion": data_fusion_service,
            "looker": looker_service,
            "data_catalog": data_catalog_service,
            "dataplex": dataplex_service,
            "earth_engine": earth_engine_service
        }
        
        for name, service in services.items():
            if service:
                try:
                    status = await service.get_status()
                    services_status[name] = status.get("status", "unknown")
                    if status.get("status") != "healthy":
                        all_healthy = False
                except Exception as e:
                    services_status[name] = "error"
                    all_healthy = False
                    logger.error(f"Health check failed for {name}", error=str(e))
            else:
                services_status[name] = "not_initialized"
                all_healthy = False
        
        if all_healthy:
            return {
                "status": "ready",
                "services": services_status,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "services": services_status,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/live")
async def liveness_check():
    """Liveness check for Kubernetes"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with service information"""
    try:
        from ..main import (
            bigquery_service, dataflow_service, dataproc_service,
            composer_service, data_fusion_service, looker_service,
            data_catalog_service, dataplex_service, earth_engine_service
        )
        
        detailed_status = {
            "service": "GCP Data Analytics Integration",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        services = {
            "bigquery": bigquery_service,
            "dataflow": dataflow_service,
            "dataproc": dataproc_service,
            "composer": composer_service,
            "data_fusion": data_fusion_service,
            "looker": looker_service,
            "data_catalog": data_catalog_service,
            "dataplex": dataplex_service,
            "earth_engine": earth_engine_service
        }
        
        for name, service in services.items():
            if service:
                try:
                    status = await service.get_status()
                    detailed_status["services"][name] = status
                except Exception as e:
                    detailed_status["services"][name] = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
            else:
                detailed_status["services"][name] = {
                    "status": "not_initialized",
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        return detailed_status
        
    except Exception as e:
        logger.error("Detailed health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Health check error: {str(e)}")


@router.get("/metrics")
async def get_health_metrics():
    """Get health metrics for monitoring"""
    try:
        from ..main import (
            bigquery_service, dataflow_service, dataproc_service,
            composer_service, data_fusion_service, looker_service,
            data_catalog_service, dataplex_service, earth_engine_service
        )
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "service_count": 0,
            "healthy_services": 0,
            "unhealthy_services": 0,
            "not_initialized_services": 0,
            "services": {}
        }
        
        services = {
            "bigquery": bigquery_service,
            "dataflow": dataflow_service,
            "dataproc": dataproc_service,
            "composer": composer_service,
            "data_fusion": data_fusion_service,
            "looker": looker_service,
            "data_catalog": data_catalog_service,
            "dataplex": dataplex_service,
            "earth_engine": earth_engine_service
        }
        
        for name, service in services.items():
            metrics["service_count"] += 1
            
            if service:
                try:
                    status = await service.get_status()
                    service_status = status.get("status", "unknown")
                    
                    if service_status == "healthy":
                        metrics["healthy_services"] += 1
                    else:
                        metrics["unhealthy_services"] += 1
                    
                    metrics["services"][name] = {
                        "status": service_status,
                        "response_time_ms": 0  # Could add actual timing
                    }
                    
                except Exception as e:
                    metrics["unhealthy_services"] += 1
                    metrics["services"][name] = {
                        "status": "error",
                        "error": str(e)
                    }
            else:
                metrics["not_initialized_services"] += 1
                metrics["services"][name] = {
                    "status": "not_initialized"
                }
        
        return metrics
        
    except Exception as e:
        logger.error("Error getting health metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")