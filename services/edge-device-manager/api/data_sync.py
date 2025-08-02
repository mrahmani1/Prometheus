"""
API endpoints for data synchronization
"""

import logging
from typing import List, Optional

from core.device_manager import EdgeDeviceManager
from core.sync_manager import DataSyncManager
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from models.config import SyncConfiguration
from models.device import DataSyncRequest, DataSyncResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# Dependencies
async def get_sync_manager() -> DataSyncManager:
    from main import sync_manager

    if not sync_manager:
        raise HTTPException(status_code=503, detail="Sync manager not available")
    return sync_manager


async def get_device_manager() -> EdgeDeviceManager:
    from main import device_manager

    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not available")
    return device_manager


@router.post("/request", response_model=DataSyncResponse)
async def request_sync(
    request: DataSyncRequest, sync_manager: DataSyncManager = Depends(get_sync_manager)
):
    """Request data synchronization for specified devices"""
    try:
        response = await sync_manager.request_sync(request)
        logger.info(f"Sync request initiated: {response.sync_id}")
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to request sync: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{sync_id}")
async def get_sync_status(
    sync_id: str, sync_manager: DataSyncManager = Depends(get_sync_manager)
):
    """Get status of a sync operation"""
    status = await sync_manager.get_sync_status(sync_id)
    if not status:
        raise HTTPException(status_code=404, detail="Sync operation not found")
    return status


@router.post("/cancel/{sync_id}")
async def cancel_sync(
    sync_id: str, sync_manager: DataSyncManager = Depends(get_sync_manager)
):
    """Cancel a sync operation"""
    try:
        await sync_manager.cancel_sync(sync_id)
        return {
            "status": "success",
            "message": f"Sync {sync_id} cancelled",
            "sync_id": sync_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to cancel sync {sync_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/devices/{device_id}/sync")
async def sync_device(
    device_id: str,
    data_types: List[str],
    background_tasks: BackgroundTasks,
    sync_manager: DataSyncManager = Depends(get_sync_manager),
    device_manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """Synchronize data for a specific device"""
    try:
        device = await device_manager.get_device(device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        # Create sync configuration (would be loaded from device config)
        sync_config = SyncConfiguration()

        # Start sync in background
        background_tasks.add_task(
            sync_manager.sync_device_data, device_id, data_types, sync_config
        )

        return {
            "status": "success",
            "message": f"Sync initiated for device {device_id}",
            "device_id": device_id,
            "data_types": data_types,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to sync device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/devices/{device_id}/schedule")
async def schedule_device_sync(
    device_id: str,
    sync_config: SyncConfiguration,
    sync_manager: DataSyncManager = Depends(get_sync_manager),
    device_manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """Schedule regular sync for a device"""
    try:
        device = await device_manager.get_device(device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        await sync_manager.schedule_device_sync(device_id, sync_config)

        return {
            "status": "success",
            "message": f"Sync scheduled for device {device_id}",
            "device_id": device_id,
            "sync_config": sync_config,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to schedule sync for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/metrics")
async def get_sync_metrics(sync_manager: DataSyncManager = Depends(get_sync_manager)):
    """Get sync metrics and statistics"""
    try:
        metrics = await sync_manager.get_sync_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get sync metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/devices/{device_id}/sync-status")
async def get_device_sync_status(
    device_id: str, device_manager: EdgeDeviceManager = Depends(get_device_manager)
):
    """Get sync status for a specific device"""
    try:
        device = await device_manager.get_device(device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        return {
            "device_id": device_id,
            "last_sync": device.last_sync,
            "sync_status": device.sync_status,
            "sync_enabled": True,  # Would be from device config
            "next_scheduled_sync": None,  # Would be calculated from schedule
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get sync status for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/active")
async def get_active_syncs(sync_manager: DataSyncManager = Depends(get_sync_manager)):
    """Get currently active sync operations"""
    try:
        metrics = await sync_manager.get_sync_metrics()
        active_syncs = metrics.get("recent_syncs", [])

        # Filter for active syncs
        active_syncs = [
            sync
            for sync in active_syncs
            if sync.get("status") in ["syncing", "pending"]
        ]

        return {"active_syncs": active_syncs, "total_active": len(active_syncs)}
    except Exception as e:
        logger.error(f"Failed to get active syncs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/bulk-sync")
async def bulk_sync_devices(
    device_ids: List[str],
    data_types: List[str],
    sync_manager: DataSyncManager = Depends(get_sync_manager),
):
    """Initiate bulk sync for multiple devices"""
    try:
        request = DataSyncRequest(
            device_ids=device_ids, data_types=data_types, priority="normal"
        )

        response = await sync_manager.request_sync(request)

        return {
            "status": "success",
            "message": f"Bulk sync initiated for {len(device_ids)} devices",
            "sync_id": response.sync_id,
            "device_count": len(device_ids),
            "data_types": data_types,
        }
    except Exception as e:
        logger.error(f"Failed to initiate bulk sync: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history")
async def get_sync_history(
    device_id: Optional[str] = None,
    hours: int = 24,
    sync_manager: DataSyncManager = Depends(get_sync_manager),
):
    """Get sync history with optional device filter"""
    try:
        metrics = await sync_manager.get_sync_metrics()
        recent_syncs = metrics.get("recent_syncs", [])

        # Filter by device if specified
        if device_id:
            # This would need to be implemented to filter by device
            pass

        return {
            "sync_history": recent_syncs,
            "time_range_hours": hours,
            "device_filter": device_id,
            "total_syncs": len(recent_syncs),
        }
    except Exception as e:
        logger.error(f"Failed to get sync history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/configuration/update")
async def update_sync_configuration(
    device_id: str,
    config: SyncConfiguration,
    device_manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """Update sync configuration for a device"""
    try:
        device = await device_manager.get_device(device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        # Update device configuration with sync settings
        config_updates = {
            "sync_enabled": config.sync_enabled,
            "sync_direction": config.sync_direction,
            "sync_data_types": {
                "sensor_data": config.sensor_data,
                "model_outputs": config.model_outputs,
                "logs": config.logs,
                "metrics": config.metrics,
            },
        }

        await device_manager.update_device_config(device_id, config_updates)

        return {
            "status": "success",
            "message": f"Sync configuration updated for device {device_id}",
            "device_id": device_id,
            "configuration": config,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update sync configuration for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status")
async def get_sync_service_status(
    sync_manager: DataSyncManager = Depends(get_sync_manager),
):
    """Get sync service status"""
    try:
        return {
            "service": "data-sync-manager",
            "status": "running" if sync_manager.is_ready() else "not_ready",
            "scheduler_running": sync_manager._sync_scheduler_running,
            "active_syncs": len(sync_manager.active_syncs),
            "queued_syncs": sync_manager.sync_queue.qsize(),
            "total_sync_tasks": len(sync_manager.sync_tasks),
        }
    except Exception as e:
        logger.error(f"Failed to get sync service status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
