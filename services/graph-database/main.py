#!/usr/bin/env python3
"""
Neo4j Graph Database Service for Geological Relationships
Main entry point for the graph database microservice
"""

import logging
import os
from contextlib import asynccontextmanager

import torch.nn as nn
import uvicorn
from api.routes import router as api_router
from core.database import GraphDatabaseManager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global database manager instance
db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global db_manager

    # Startup
    logger.info("Starting Graph Database Service...")
    try:
        db_manager = GraphDatabaseManager()
        await db_manager.connect()
        logger.info("Connected to Neo4j database")

        # Initialize graph schema
        await db_manager.initialize_schema()
        logger.info("Graph schema initialized")

        yield

    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise
    finally:
        # Shutdown
        if db_manager:
            await db_manager.close()
            logger.info("Database connections closed")

# Create FastAPI application
app = FastAPI(
    title="Prometheus Graph Database Service",
    description="Neo4j graph database service for geological process relationships",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if db_manager and await db_manager.is_connected():
            return {"status": "healthy", "database": "connected"}
        else:
            raise HTTPException(status_code=503, detail="Database not connected")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        if db_manager and await db_manager.is_ready():
            return {"status": "ready", "database": "operational"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development",
    )
