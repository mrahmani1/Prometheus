import asyncio
import logging
import os
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app

from core.helios_engine import HeliosEngine
from core.service_registry import ServiceRegistry
from api.routes import router
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
helios_engine = None
service_registry = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global helios_engine, service_registry
    
    logger.info("Starting Helios Core Service")
    
    try:
        # Initialize service registry
        service_registry = ServiceRegistry(
            consul_host=settings.consul_host,
            consul_port=settings.consul_port
        )
        await service_registry.initialize()
        
        # Initialize Helios engine
        helios_engine = HeliosEngine(
            config=settings,
            service_registry=service_registry
        )
        await helios_engine.initialize()
        
        # Register service with Consul
        await service_registry.register_service(
            name="prometheus-helios-core",
            service_id="helios-core-1",
            address="helios-core",
            port=8000,
            health_check_url="/health"
        )
        
        logger.info("Helios Core Service initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to initialize Helios Core Service", error=str(e))
        raise
    finally:
        # Cleanup
        if service_registry:
            await service_registry.deregister_service("helios-core-1")
        if helios_engine:
            await helios_engine.cleanup()
        logger.info("Helios Core Service shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Helios AI Core Engine",
    description="Physics-Informed Neural Network engine for geodynamic simulations",
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
        # Check if Helios engine is ready
        if helios_engine and await helios_engine.is_healthy():
            return {
                "status": "healthy",
                "service": "helios-core",
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
        if helios_engine and await helios_engine.is_ready():
            return {
                "status": "ready",
                "service": "helios-core",
                "models_loaded": await helios_engine.get_loaded_models_count(),
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
        "service": "Helios AI Core Engine",
        "description": "Physics-Informed Neural Network engine for geodynamic simulations",
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
        port=8000,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )