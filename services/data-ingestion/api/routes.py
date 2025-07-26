"""API routes for data ingestion service"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()

# Global reference to ingestion engine (set in main.py)
ingestion_engine = None

def get_ingestion_engine():
    """Dependency to get the ingestion engine"""
    if ingestion_engine is None:
        raise HTTPException(status_code=503, detail="Ingestion engine not available")
    return ingestion_engine

router = APIRouter()

class IngestionRequest(BaseModel):
    """Request model for manual data ingestion"""
    connector_name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    parameters: Optional[Dict[str, Any]] = None

class ConnectorStatusResponse(BaseModel):
    """Response model for connector status"""
    name: str
    healthy: bool
    ready: bool
    last_data_time: Optional[datetime] = None
    total_records_processed: int = 0
    error_count: int = 0

class IngestionStatisticsResponse(BaseModel):
    """Response model for ingestion statistics"""
    total_connectors: int
    active_connectors: int
    ingestion_tasks_running: int
    connectors: Dict[str, Any]

@router.get("/connectors", response_model=List[str])
async def list_connectors(engine = Depends(get_ingestion_engine)):
    """List all available data connectors"""
    try:
        connector_names = list(engine.connectors.keys())
        return connector_names
    except Exception as e:
        logger.error("Failed to list connectors", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list connectors")

@router.get("/connectors/status", response_model=Dict[str, ConnectorStatusResponse])
async def get_connectors_status(engine = Depends(get_ingestion_engine)):
    """Get status of all connectors"""
    try:
        status = await engine.get_connector_status()
        
        response = {}
        for name, connector_status in status.items():
            response[name] = ConnectorStatusResponse(
                name=name,
                healthy=connector_status.get("healthy", False),
                ready=connector_status.get("ready", False),
                last_data_time=connector_status.get("last_data_time"),
                total_records_processed=connector_status.get("total_records_processed", 0),
                error_count=connector_status.get("error_count", 0)
            )
        
        return response
        
    except Exception as e:
        logger.error("Failed to get connector status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get connector status")

@router.get("/connectors/{connector_name}/status", response_model=ConnectorStatusResponse)
async def get_connector_status(
    connector_name: str,
    engine = Depends(get_ingestion_engine)
):
    """Get status of a specific connector"""
    try:
        if connector_name not in engine.connectors:
            raise HTTPException(status_code=404, detail=f"Connector '{connector_name}' not found")
        
        status = await engine.get_connector_status()
        connector_status = status.get(connector_name, {})
        
        return ConnectorStatusResponse(
            name=connector_name,
            healthy=connector_status.get("healthy", False),
            ready=connector_status.get("ready", False),
            last_data_time=connector_status.get("last_data_time"),
            total_records_processed=connector_status.get("total_records_processed", 0),
            error_count=connector_status.get("error_count", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get connector status", connector=connector_name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get connector status")

@router.post("/connectors/{connector_name}/ingest")
async def trigger_manual_ingestion(
    connector_name: str,
    request: IngestionRequest,
    engine = Depends(get_ingestion_engine)
):
    """Trigger manual data ingestion for a specific connector"""
    try:
        if connector_name not in engine.connectors:
            raise HTTPException(status_code=404, detail=f"Connector '{connector_name}' not found")
        
        # Use request parameters or defaults
        start_time = request.start_time
        end_time = request.end_time
        parameters = request.parameters or {}
        
        # Set default time range if not provided
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(hours=1)
        
        result = await engine.trigger_manual_ingestion(
            connector_name=connector_name,
            start_time=start_time,
            end_time=end_time,
            parameters=parameters
        )
        
        return {
            "message": f"Manual ingestion triggered for {connector_name}",
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to trigger manual ingestion", connector=connector_name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to trigger manual ingestion")

@router.post("/connectors/{connector_name}/pause")
async def pause_connector(
    connector_name: str,
    engine = Depends(get_ingestion_engine)
):
    """Pause a specific connector"""
    try:
        if connector_name not in engine.connectors:
            raise HTTPException(status_code=404, detail=f"Connector '{connector_name}' not found")
        
        await engine.pause_connector(connector_name)
        
        return {
            "message": f"Connector '{connector_name}' paused successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to pause connector", connector=connector_name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to pause connector")

@router.post("/connectors/{connector_name}/resume")
async def resume_connector(
    connector_name: str,
    engine = Depends(get_ingestion_engine)
):
    """Resume a specific connector"""
    try:
        if connector_name not in engine.connectors:
            raise HTTPException(status_code=404, detail=f"Connector '{connector_name}' not found")
        
        await engine.resume_connector(connector_name)
        
        return {
            "message": f"Connector '{connector_name}' resumed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resume connector", connector=connector_name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to resume connector")

@router.get("/statistics", response_model=IngestionStatisticsResponse)
async def get_ingestion_statistics(engine = Depends(get_ingestion_engine)):
    """Get ingestion statistics for all connectors"""
    try:
        stats = await engine.get_ingestion_statistics()
        
        return IngestionStatisticsResponse(
            total_connectors=stats["total_connectors"],
            active_connectors=stats["active_connectors"],
            ingestion_tasks_running=stats["ingestion_tasks_running"],
            connectors=stats["connectors"]
        )
        
    except Exception as e:
        logger.error("Failed to get ingestion statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get ingestion statistics")

@router.get("/processing/statistics")
async def get_processing_statistics(engine = Depends(get_ingestion_engine)):
    """Get data processing statistics"""
    try:
        if not engine.data_processor:
            raise HTTPException(status_code=503, detail="Data processor not available")
        
        stats = await engine.data_processor.get_processing_statistics()
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get processing statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get processing statistics")

@router.get("/fiber-optic/geometry")
async def get_fiber_geometry(engine = Depends(get_ingestion_engine)):
    """Get fiber optic cable geometry information"""
    try:
        if "fiber_optic" not in engine.connectors:
            raise HTTPException(status_code=404, detail="Fiber optic connector not available")
        
        fiber_connector = engine.connectors["fiber_optic"]
        if not hasattr(fiber_connector, 'get_fiber_geometry'):
            raise HTTPException(status_code=501, detail="Fiber geometry not supported")
        
        geometry = await fiber_connector.get_fiber_geometry()
        return {
            "fiber_length_km": fiber_connector.fiber_length_km,
            "spatial_resolution_m": fiber_connector.spatial_resolution,
            "total_channels": len(geometry),
            "geometry": geometry
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get fiber geometry", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get fiber geometry")

@router.get("/data/recent")
async def get_recent_data(
    connector_name: Optional[str] = Query(None, description="Filter by connector name"),
    data_type: Optional[str] = Query(None, description="Filter by data type"),
    hours: int = Query(24, description="Hours of recent data to retrieve"),
    limit: int = Query(100, description="Maximum number of records to return"),
    engine = Depends(get_ingestion_engine)
):
    """Get recent ingested data"""
    try:
        # This would query the data storage layer for recent data
        # For now, return a placeholder response
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        return {
            "message": "Recent data retrieval - placeholder implementation",
            "filters": {
                "connector_name": connector_name,
                "data_type": data_type,
                "start_time": start_time,
                "end_time": end_time,
                "limit": limit
            },
            "data": []  # Would contain actual data records
        }
        
    except Exception as e:
        logger.error("Failed to get recent data", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get recent data")

# Set the global ingestion engine reference
def set_ingestion_engine(engine):
    """Set the global ingestion engine reference"""
    global ingestion_engine
    ingestion_engine = engine