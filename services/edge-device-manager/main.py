#!/usr/bin/env python3
"""
Edge Device Management Service

This service manages edge computing nodes for the Prometheus AI platform,
providing device registration, configuration management, remote model deployment,
health monitoring, and data synchronization capabilities.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch.nn as nn
import uvicorn
from api.data_sync import router as sync_router
from api.device_registration import router as registration_router
from api.health_monitoring import router as health_router
from api.model_deployment import router as deployment_router
from api.offline_operations import router as offline_router
from api.offline_operations import set_offline_coordinator
from core.device_manager import EdgeDeviceManager
from core.health_monitor import HealthMonitor
from core.model_deployer import ModelDeployer
from core.offline_operations import OfflineOperationsCoordinator
from core.sync_manager import DataSyncManager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.config import EdgeManagerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instances
device_manager: Optional[EdgeDeviceManager] = None
model_deployer: Optional[ModelDeployer] = None
health_monitor: Optional[HealthMonitor] = None
sync_manager: Optional[DataSyncManager] = None
offline_coordinator: Optional[OfflineOperationsCoordinator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global device_manager, model_deployer, health_monitor, sync_manager, offline_coordinator

    logger.info("Starting Edge Device Management Service...")

    # Load configuration
    config = EdgeManagerConfig()

    # Initialize core services
    device_manager = EdgeDeviceManager(config)
    model_deployer = ModelDeployer(config)
    health_monitor = HealthMonitor(config)
    sync_manager = DataSyncManager(config)

    # Initialize offline operations coordinator
    offline_coordinator = OfflineOperationsCoordinator(config)
    set_offline_coordinator(offline_coordinator)

    # Start background services
    await device_manager.initialize()
    await health_monitor.start_monitoring()
    await sync_manager.start_sync_scheduler()
    await offline_coordinator.start()

    logger.info("Edge Device Management Service started successfully")

    yield

    # Cleanup
    logger.info("Shutting down Edge Device Management Service...")
    await offline_coordinator.stop()
    await health_monitor.stop_monitoring()
    await sync_manager.stop_sync_scheduler()
    await device_manager.cleanup()
    logger.info("Edge Device Management Service stopped")


# Create FastAPI application
app = FastAPI(
    title="Prometheus Edge Device Manager",
    description="Edge device management service for distributed AI processing",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(
    registration_router, prefix="/api/v1/devices", tags=["Device Registration"]
)
app.include_router(
    deployment_router, prefix="/api/v1/models", tags=["Model Deployment"]
)
app.include_router(health_router, prefix="/api/v1/health", tags=["Health Monitoring"])
app.include_router(sync_router, prefix="/api/v1/sync", tags=["Data Synchronization"])
app.include_router(
    offline_router, prefix="/api/v1/offline", tags=["Offline Operations"]
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "edge-device-manager", "version": "1.0.0"}


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    if not all(
        [
            device_manager,
            model_deployer,
            health_monitor,
            sync_manager,
            offline_coordinator,
        ]
    ):
        raise HTTPException(status_code=503, detail="Service not ready")

    return {
        "status": "ready",
        "services": {
            "device_manager": device_manager.is_ready(),
            "model_deployer": model_deployer.is_ready(),
            "health_monitor": health_monitor.is_ready(),
            "sync_manager": sync_manager.is_ready(),
            "offline_coordinator": offline_coordinator.is_running,
        },
    }


@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    if not device_manager:
        raise HTTPException(status_code=503, detail="Service not ready")

    return await device_manager.get_metrics()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
    )
