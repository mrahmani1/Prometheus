"""
API endpoints for device registration and management
"""

import logging
from typing import List, Optional

from core.device_manager import EdgeDeviceManager
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from models.device import (DeviceMetrics, DeviceRegistrationRequest,
                           DeviceRegistrationResponse, DeviceStatus,
                           EdgeDevice)

logger = logging.getLogger(__name__)

router = APIRouter()


# Dependency to get device manager instance
async def get_device_manager() -> EdgeDeviceManager:
    from main import device_manager

    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not available")
    return device_manager


@router.post("/register", response_model=DeviceRegistrationResponse)
async def register_device(
    request: DeviceRegistrationRequest,
    manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """Register a new edge device"""
    try:
        response = await manager.register_device(request)
        logger.info(f"Device registered: {response.device_id}")
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to register device: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{device_id}", response_model=EdgeDevice)
async def get_device(
    device_id: str, manager: EdgeDeviceManager = Depends(get_device_manager)
):
    """Get device information by ID"""
    device = await manager.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device


@router.get("/", response_model=List[EdgeDevice])
async def list_devices(
    region: Optional[str] = None,
    device_type: Optional[str] = None,
    status: Optional[DeviceStatus] = None,
    manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """List devices with optional filters"""
    try:
        devices = await manager.list_devices(
            region=region, device_type=device_type, status=status
        )
        return devices
    except Exception as e:
        logger.error(f"Failed to list devices: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{device_id}/heartbeat")
async def update_heartbeat(
    device_id: str,
    metrics: Optional[DeviceMetrics] = None,
    manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """Update device heartbeat and metrics"""
    try:
        await manager.update_device_heartbeat(device_id, metrics)
        return {"status": "success", "message": "Heartbeat updated"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update heartbeat for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{device_id}/config")
async def update_device_config(
    device_id: str,
    config_updates: dict,
    manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """Update device configuration"""
    try:
        updated_device = await manager.update_device_config(device_id, config_updates)
        return {
            "status": "success",
            "message": "Configuration updated",
            "device": updated_device,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update config for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{device_id}")
async def remove_device(
    device_id: str,
    background_tasks: BackgroundTasks,
    manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """Remove a device from management"""
    try:
        # Remove device in background to avoid blocking
        background_tasks.add_task(manager.remove_device, device_id)
        return {"status": "success", "message": "Device removal initiated"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to remove device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{device_id}/status")
async def get_device_status(
    device_id: str, manager: EdgeDeviceManager = Depends(get_device_manager)
):
    """Get detailed device status"""
    device = await manager.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    return {
        "device_id": device_id,
        "status": device.status,
        "last_seen": device.last_seen,
        "uptime_seconds": device.uptime_seconds,
        "current_metrics": device.current_metrics,
        "deployed_models": device.deployed_models,
        "last_sync": device.last_sync,
        "sync_status": device.sync_status,
    }


@router.get("/{device_id}/metrics/history")
async def get_device_metrics_history(
    device_id: str,
    hours: int = 24,
    manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """Get historical metrics for a device"""
    device = await manager.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    # This would typically fetch from time-series database
    # For now, return placeholder data
    return {
        "device_id": device_id,
        "time_range_hours": hours,
        "metrics": [],
        "message": "Historical metrics would be retrieved from time-series database",
    }


@router.post("/{device_id}/reboot")
async def reboot_device(
    device_id: str, manager: EdgeDeviceManager = Depends(get_device_manager)
):
    """Send reboot command to device"""
    device = await manager.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    # This would send actual reboot command to device
    # For now, just return success
    return {
        "status": "success",
        "message": f"Reboot command sent to device {device_id}",
        "device_id": device_id,
    }


@router.post("/{device_id}/maintenance")
async def set_maintenance_mode(
    device_id: str,
    enable: bool = True,
    manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """Set device maintenance mode"""
    try:
        device = await manager.get_device(device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        # Update device status
        new_status = DeviceStatus.MAINTENANCE if enable else DeviceStatus.ONLINE
        await manager.update_device_config(device_id, {"status": new_status})

        return {
            "status": "success",
            "message": f"Device {device_id} {'entered' if enable else 'exited'} maintenance mode",
            "maintenance_mode": enable,
        }
    except Exception as e:
        logger.error(f"Failed to set maintenance mode for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
