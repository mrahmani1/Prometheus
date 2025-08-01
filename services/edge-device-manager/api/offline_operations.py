"""
API endpoints for offline operations management
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from core.offline_alerting import AlertSeverity
from core.offline_operations import OfflineOperationsCoordinator
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Global offline operations coordinator (will be injected)
offline_coordinator: Optional[OfflineOperationsCoordinator] = None

class SensorDataRequest(BaseModel):
    """Request model for sensor data ingestion"""

    data_type: str = Field(
        ..., description="Type of sensor data (seismic, thermal, gas, deformation)"
    )
    data: Dict[str, Any] = Field(..., description="Sensor data payload")
    location: Optional[Dict[str, float]] = Field(
        None, description="Sensor location (lat, lon)"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class AlertAcknowledgeRequest(BaseModel):
    """Request model for alert acknowledgment"""

    alert_id: str = Field(..., description="Alert ID to acknowledge")
    acknowledged_by: str = Field(
        default="api", description="Who acknowledged the alert"
    )

class SyncConfigurationRequest(BaseModel):
    """Request model for sync configuration updates"""

    bandwidth_limit_mbps: Optional[float] = Field(
        None, description="Bandwidth limit in Mbps"
    )
    max_concurrent_syncs: Optional[int] = Field(
        None, description="Maximum concurrent sync operations"
    )
    sync_batch_size: Optional[int] = Field(None, description="Sync batch size")
    connectivity_check_interval: Optional[int] = Field(
        None, description="Connectivity check interval in seconds"
    )

def set_offline_coordinator(coordinator: OfflineOperationsCoordinator):
    """Set the global offline operations coordinator"""
    global offline_coordinator
    offline_coordinator = coordinator

@router.post("/ingest", response_model=Dict[str, str])
async def ingest_sensor_data(request: SensorDataRequest):
    """Ingest sensor data for offline processing"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        record_id = await offline_coordinator.ingest_sensor_data(
            data_type=request.data_type, data=request.data, location=request.location
        )

        return {
            "record_id": record_id,
            "status": "ingested",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error ingesting sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=Dict[str, Any])
async def get_offline_status():
    """Get offline operations status"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        status = await offline_coordinator.get_operation_status()

        return {
            "is_offline": status.is_offline,
            "offline_since": (
                status.offline_since.isoformat() if status.offline_since else None
            ),
            "connectivity_score": status.connectivity_score,
            "data_processing_active": status.data_processing_active,
            "alert_generation_active": status.alert_generation_active,
            "storage_usage_percent": status.storage_usage_percent,
            "pending_sync_count": status.pending_sync_count,
            "active_alerts_count": status.active_alerts_count,
            "last_analysis_time": (
                status.last_analysis_time.isoformat()
                if status.last_analysis_time
                else None
            ),
        }

    except Exception as e:
        logger.error(f"Error getting offline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/comprehensive", response_model=Dict[str, Any])
async def get_comprehensive_status():
    """Get comprehensive status of all offline operations"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        return await offline_coordinator.get_comprehensive_status()

    except Exception as e:
        logger.error(f"Error getting comprehensive status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts", response_model=List[Dict[str, Any]])
async def get_active_alerts(
    severity: Optional[str] = Query(
        None, description="Filter by severity (low, medium, high, critical)"
    )
):
    """Get active alerts"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        # Parse severity filter
        severity_filter = None
        if severity:
            try:
                severity_filter = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid severity: {severity}"
                )

        alerts = await offline_coordinator.get_active_alerts(severity_filter)

        # Convert alerts to dict format
        alert_list = []
        for alert in alerts:
            alert_dict = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity.value,
                "status": alert.status.value,
                "title": alert.title,
                "message": alert.message,
                "data_source": alert.data_source,
                "trigger_value": alert.trigger_value,
                "threshold_value": alert.threshold_value,
                "location": alert.location,
                "metadata": alert.metadata,
                "acknowledged_at": (
                    alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
                ),
                "resolved_at": (
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ),
            }
            alert_list.append(alert_dict)

        return alert_list

    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/acknowledge", response_model=Dict[str, Any])
async def acknowledge_alert(request: AlertAcknowledgeRequest):
    """Acknowledge an alert"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        success = await offline_coordinator.acknowledge_alert(request.alert_id)

        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")

        return {
            "alert_id": request.alert_id,
            "status": "acknowledged",
            "acknowledged_by": request.acknowledged_by,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/resolve", response_model=Dict[str, Any])
async def resolve_alert(alert_id: str):
    """Resolve an alert"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        success = await offline_coordinator.resolve_alert(alert_id)

        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")

        return {
            "alert_id": alert_id,
            "status": "resolved",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connectivity", response_model=Dict[str, Any])
async def get_connectivity_status():
    """Get connectivity status"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        return await offline_coordinator.connectivity_manager.get_connectivity_status()

    except Exception as e:
        logger.error(f"Error getting connectivity status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sync/status", response_model=Dict[str, Any])
async def get_sync_status():
    """Get synchronization status"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        return await offline_coordinator.get_sync_status()

    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync/force", response_model=Dict[str, Any])
async def force_data_sync(background_tasks: BackgroundTasks):
    """Force synchronization of pending data"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        # Run sync in background
        sync_result = await offline_coordinator.force_data_sync()

        return {
            "status": "sync_initiated",
            "result": sync_result,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error forcing data sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/sync/config", response_model=Dict[str, Any])
async def update_sync_configuration(request: SyncConfigurationRequest):
    """Update synchronization configuration"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        # Convert request to dict, excluding None values
        config_updates = {k: v for k, v in request.dict().items() if v is not None}

        if not config_updates:
            raise HTTPException(
                status_code=400, detail="No configuration updates provided"
            )

        await offline_coordinator.connectivity_manager.update_sync_configuration(
            config_updates
        )

        return {
            "status": "configuration_updated",
            "updates": config_updates,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating sync configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync/retry-failed", response_model=Dict[str, Any])
async def retry_failed_syncs():
    """Retry failed synchronization tasks"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        result = await offline_coordinator.connectivity_manager.retry_failed_syncs()

        return {
            "status": "retry_initiated",
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error retrying failed syncs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sync/queue", response_model=Dict[str, Any])
async def clear_sync_queue():
    """Clear the synchronization queue"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        await offline_coordinator.connectivity_manager.clear_sync_queue()

        return {"status": "queue_cleared", "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Error clearing sync queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/storage/statistics", response_model=Dict[str, Any])
async def get_storage_statistics():
    """Get local storage statistics"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        return await offline_coordinator.storage_manager.get_storage_statistics()

    except Exception as e:
        logger.error(f"Error getting storage statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/storage/cleanup", response_model=Dict[str, Any])
async def cleanup_storage():
    """Cleanup expired storage data"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        cleanup_result = (
            await offline_coordinator.storage_manager.cleanup_expired_data()
        )

        return {
            "status": "cleanup_completed",
            "result": cleanup_result,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error cleaning up storage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processing/statistics", response_model=Dict[str, Any])
async def get_processing_statistics():
    """Get data processing statistics"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        return await offline_coordinator.offline_processor.get_processing_statistics()

    except Exception as e:
        logger.error(f"Error getting processing statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/history", response_model=List[Dict[str, Any]])
async def get_alert_history(
    limit: int = Query(100, description="Maximum number of alerts to return"),
    since: Optional[str] = Query(
        None, description="ISO timestamp to filter alerts since"
    ),
):
    """Get alert history"""
    try:
        if not offline_coordinator:
            raise HTTPException(
                status_code=503, detail="Offline operations not available"
            )

        # Parse since parameter
        since_datetime = None
        if since:
            try:
                since_datetime = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid timestamp format")

        alerts = offline_coordinator.alert_generator.get_alert_history(
            limit=limit, since=since_datetime
        )

        # Convert alerts to dict format
        alert_list = []
        for alert in alerts:
            alert_dict = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity.value,
                "status": alert.status.value,
                "title": alert.title,
                "message": alert.message,
                "data_source": alert.data_source,
                "trigger_value": alert.trigger_value,
                "threshold_value": alert.threshold_value,
                "location": alert.location,
                "metadata": alert.metadata,
                "acknowledged_at": (
                    alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
                ),
                "resolved_at": (
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ),
            }
            alert_list.append(alert_dict)

        return alert_list

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check for offline operations"""
    try:
        if not offline_coordinator:
            return {
                "status": "unavailable",
                "message": "Offline operations coordinator not initialized",
            }

        if not offline_coordinator.is_running:
            return {
                "status": "stopped",
                "message": "Offline operations coordinator is not running",
            }

        # Get basic status
        operation_status = await offline_coordinator.get_operation_status()

        return {
            "status": "healthy",
            "is_offline": operation_status.is_offline,
            "connectivity_score": operation_status.connectivity_score,
            "data_processing_active": operation_status.data_processing_active,
            "alert_generation_active": operation_status.alert_generation_active,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in offline operations health check: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
