#!/usr/bin/env python3
"""
Anomaly Detection Service for Geological Phenomena Discovery

This service implements advanced anomaly detection algorithms for discovering
new geological phenomena using multi-variate analysis, contextual knowledge,
and causal reinforcement learning.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch.nn as nn
import uvicorn
from api.routes import router as api_router
from core.anomaly_engine import AnomalyDetectionEngine
from core.config import get_settings
from core.database import DatabaseManager
from core.monitoring import setup_monitoring
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
anomaly_engine: Optional[AnomalyDetectionEngine] = None
db_manager: Optional[DatabaseManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global anomaly_engine, db_manager

    settings = get_settings()

    # Initialize database connections
    db_manager = DatabaseManager(settings)
    await db_manager.initialize()

    # Initialize anomaly detection engine
    anomaly_engine = AnomalyDetectionEngine(settings, db_manager)
    await anomaly_engine.initialize()

    # Start background monitoring
    monitoring_task = asyncio.create_task(anomaly_engine.start_continuous_monitoring())

    logger.info("Anomaly Detection Service initialized successfully")

    yield

    # Cleanup
    monitoring_task.cancel()
    await anomaly_engine.shutdown()
    await db_manager.close()
    logger.info("Anomaly Detection Service shutdown complete")

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()

    app = FastAPI(
        title="Prometheus Anomaly Detection Service",
        description="Advanced anomaly detection for geological phenomena discovery",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")

    # Setup monitoring
    setup_monitoring(app)

    return app

app = create_app()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy", service="anomaly-detection", version="1.0.0"
    )

@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint"""
    if anomaly_engine is None or db_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return HealthResponse(status="ready", service="anomaly-detection", version="1.0.0")

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
