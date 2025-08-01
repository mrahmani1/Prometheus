"""
API endpoints for health monitoring and diagnostics
"""

import logging
from typing import Optional

from core.device_manager import EdgeDeviceManager
from core.health_monitor import HealthMonitor
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from models.device import HealthCheckResult

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependencies
async def get_health_monitor() -> HealthMonitor:
    from main import health_monitor

    if not health_monitor:
        raise HTTPException(status_code=503, detail="Health monitor not available")
    return health_monitor

async def get_device_manager() -> EdgeDeviceManager:
    from main import device_manager

    if not device_manager:
        raise HTTPException(status_code=503, detail="Device manager not available")
    return device_manager

@router.get("/summary")
async def get_health_summary(monitor: HealthMonitor = Depends(get_health_monitor)):
    """Get health summary for all monitored devices"""
    try:
        summary = await monitor.get_health_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get health summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/devices/{device_id}", response_model=HealthCheckResult)
async def get_device_health(
    device_id: str, monitor: HealthMonitor = Depends(get_health_monitor)
):
    """Get latest health check result for a device"""
    health_result = await monitor.get_device_health(device_id)
    if not health_result:
        raise HTTPException(
            status_code=404, detail="No health data available for device"
        )
    return health_result

@router.post("/devices/{device_id}/check")
async def perform_health_check(
    device_id: str,
    background_tasks: BackgroundTasks,
    monitor: HealthMonitor = Depends(get_health_monitor),
    device_manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """Perform immediate health check on a device"""
    try:
        device = await device_manager.get_device(device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        # Perform health check in background
        background_tasks.add_task(monitor.perform_health_check, device)

        return {
            "status": "success",
            "message": f"Health check initiated for device {device_id}",
            "device_id": device_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initiate health check for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/devices/{device_id}/diagnostics")
async def get_device_diagnostics(
    device_id: str, monitor: HealthMonitor = Depends(get_health_monitor)
):
    """Get detailed diagnostics for a specific device"""
    try:
        diagnostics = await monitor.get_device_diagnostics(device_id)
        return diagnostics
    except Exception as e:
        logger.error(f"Failed to get diagnostics for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/devices/{device_id}/monitoring/start")
async def start_device_monitoring(
    device_id: str,
    monitor: HealthMonitor = Depends(get_health_monitor),
    device_manager: EdgeDeviceManager = Depends(get_device_manager),
):
    """Start health monitoring for a device"""
    try:
        device = await device_manager.get_device(device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        await monitor.add_device_monitoring(device)

        return {
            "status": "success",
            "message": f"Health monitoring started for device {device_id}",
            "device_id": device_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start monitoring for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/devices/{device_id}/monitoring/stop")
async def stop_device_monitoring(
    device_id: str, monitor: HealthMonitor = Depends(get_health_monitor)
):
    """Stop health monitoring for a device"""
    try:
        await monitor.remove_device_monitoring(device_id)

        return {
            "status": "success",
            "message": f"Health monitoring stopped for device {device_id}",
            "device_id": device_id,
        }
    except Exception as e:
        logger.error(f"Failed to stop monitoring for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/alerts")
async def get_health_alerts(
    severity: Optional[str] = None,
    device_id: Optional[str] = None,
    monitor: HealthMonitor = Depends(get_health_monitor),
):
    """Get health alerts with optional filters"""
    try:
        # Get health summary which includes recent issues
        summary = await monitor.get_health_summary()
        alerts = summary.get("recent_issues", [])

        # Apply filters
        if device_id:
            alerts = [alert for alert in alerts if alert.get("device_id") == device_id]

        if severity:
            # This would filter by severity if we had that field
            pass

        return {
            "alerts": alerts,
            "total_alerts": len(alerts),
            "filters": {"severity": severity, "device_id": device_id},
        }
    except Exception as e:
        logger.error(f"Failed to get health alerts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/metrics/system")
async def get_system_health_metrics(
    monitor: HealthMonitor = Depends(get_health_monitor),
):
    """Get system-wide health metrics"""
    try:
        summary = await monitor.get_health_summary()

        # Calculate additional metrics
        total_devices = summary.get("total_devices", 0)
        healthy_devices = summary.get("healthy_devices", 0)
        warning_devices = summary.get("warning_devices", 0)
        critical_devices = summary.get("critical_devices", 0)

        health_percentage = (
            (healthy_devices / total_devices * 100) if total_devices > 0 else 0
        )

        return {
            "total_devices": total_devices,
            "healthy_devices": healthy_devices,
            "warning_devices": warning_devices,
            "critical_devices": critical_devices,
            "health_percentage": round(health_percentage, 2),
            "availability": {
                "healthy": (
                    round((healthy_devices / total_devices * 100), 2)
                    if total_devices > 0
                    else 0
                ),
                "warning": (
                    round((warning_devices / total_devices * 100), 2)
                    if total_devices > 0
                    else 0
                ),
                "critical": (
                    round((critical_devices / total_devices * 100), 2)
                    if total_devices > 0
                    else 0
                ),
            },
            "last_updated": summary.get("last_updated"),
        }
    except Exception as e:
        logger.error(f"Failed to get system health metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/devices/{device_id}/metrics/history")
async def get_device_health_history(
    device_id: str,
    hours: int = 24,
    monitor: HealthMonitor = Depends(get_health_monitor),
):
    """Get historical health metrics for a device"""
    try:
        # This would typically fetch from time-series database
        # For now, return current diagnostics
        diagnostics = await monitor.get_device_diagnostics(device_id)

        return {
            "device_id": device_id,
            "time_range_hours": hours,
            "current_health": diagnostics,
            "historical_data": diagnostics.get("historical_metrics", []),
            "message": "Historical health data would be retrieved from time-series database",
        }
    except Exception as e:
        logger.error(f"Failed to get health history for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/thresholds/update")
async def update_health_thresholds(
    thresholds: dict, monitor: HealthMonitor = Depends(get_health_monitor)
):
    """Update health monitoring thresholds"""
    try:
        # This would update the monitoring thresholds
        # For now, just validate and return success

        valid_thresholds = [
            "cpu_warning_threshold",
            "cpu_critical_threshold",
            "memory_warning_threshold",
            "memory_critical_threshold",
            "storage_warning_threshold",
            "storage_critical_threshold",
            "temperature_warning_threshold",
            "temperature_critical_threshold",
        ]

        for key in thresholds.keys():
            if key not in valid_thresholds:
                raise HTTPException(status_code=400, detail=f"Invalid threshold: {key}")

        # Update thresholds (would be implemented in monitor)
        for key, value in thresholds.items():
            if hasattr(monitor, key):
                setattr(monitor, key, value)

        return {
            "status": "success",
            "message": "Health thresholds updated",
            "updated_thresholds": thresholds,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update health thresholds: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/status")
async def get_monitoring_status(monitor: HealthMonitor = Depends(get_health_monitor)):
    """Get health monitoring service status"""
    try:
        return {
            "service": "health-monitor",
            "status": "running" if monitor.is_ready() else "not_ready",
            "monitoring_active": monitor._monitoring,
            "monitored_devices": len(monitor.health_checks),
            "active_monitoring_tasks": len(monitor.monitoring_tasks),
        }
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
