import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app

from core.ingestion_engine import DataIngestionEngine
from core.service_registry import ServiceRegistry
from api.routes import router, set_ingestion_engine
from core.config import settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global instances
ingestion_engine = None
service_registry = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global ingestion_engine, service_registry
    
    logger.info("Starting Data Ingestion Service")
    
    try:
        # Initialize service registry
        service_registry = ServiceRegistry(
            consul_host=settings.consul_host,
            consul_port=settings.consul_port
        )
        await service_registry.initialize()
        
        # Initialize data ingestion engine
        ingestion_engine = DataIngestionEngine(
            config=settings,
            service_registry=service_registry
        )
        await ingestion_engine.initialize()
        
        # Set the global reference for API routes
        set_ingestion_engine(ingestion_engine)
        
        # Register service with Consul
        await service_registry.register_service(
            name="prometheus-data-ingestion",
            service_id="data-ingestion-1",
            address="data-ingestion",
            port=8001,
            health_check_url="/health"
        )
        
        logger.info("Data Ingestion Service initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to initialize Data Ingestion Service", error=str(e))
        raise
    finally:
        # Cleanup
        if service_registry:
            await service_registry.deregister_service("data-ingestion-1")
        if ingestion_engine:
            await ingestion_engine.cleanup()
        logger.info("Data Ingestion Service shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Prometheus Data Ingestion Service",
    description="Multi-source data ingestion and processing service for geological data",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if ingestion_engine and await ingestion_engine.is_healthy():
            return {
                "status": "healthy",
                "service": "data-ingestion",
                "version": "1.0.0",
                "timestamp": structlog.processors.TimeStamper().now()
            }
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        if ingestion_engine and await ingestion_engine.is_ready():
            return {
                "status": "ready",
                "service": "data-ingestion",
                "active_connectors": await ingestion_engine.get_active_connectors_count(),
                "timestamp": structlog.processors.TimeStamper().now()
            }
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Prometheus Data Ingestion Service",
        "description": "Multi-source data ingestion and processing service for geological data",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "api": "/api/v1",
            "docs": "/docs",
            "metrics": "/metrics"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )