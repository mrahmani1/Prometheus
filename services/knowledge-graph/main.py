#!/usr/bin/env python3
"""
Scientific Knowledge Graph Service for Prometheus AI Platform
Handles automated knowledge extraction, semantic reasoning, and dynamic updates
"""

import os
from contextlib import asynccontextmanager

import pandas as pd
import structlog
import torch.nn as nn
import uvicorn
from api.routes import router as api_router
from core.federated_discovery import FederatedCausalDiscovery
from core.knowledge_extractor import ScientificKnowledgeExtractor
from core.knowledge_graph_manager import KnowledgeGraphManager
from core.semantic_reasoner import SemanticReasoner
from core.temporal_causal_engine import TemporalCausalEngine
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global service components
service_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Scientific Knowledge Graph Service...")

    try:
        # Initialize core components
        service_components["knowledge_extractor"] = ScientificKnowledgeExtractor()
        service_components["semantic_reasoner"] = SemanticReasoner()
        service_components["knowledge_graph_manager"] = KnowledgeGraphManager()
        service_components["temporal_causal_engine"] = TemporalCausalEngine()
        service_components["federated_discovery"] = FederatedCausalDiscovery()

        # Initialize connections
        await service_components["knowledge_graph_manager"].initialize()
        await service_components["semantic_reasoner"].initialize()

        logger.info("Scientific Knowledge Graph Service initialized successfully")
        yield

    except Exception as e:
        logger.error("Failed to start service", error=str(e))
        raise
    finally:
        # Cleanup
        for component in service_components.values():
            if hasattr(component, "close"):
                await component.close()
        logger.info("Service components closed")

# Create FastAPI application
app = FastAPI(
    title="Prometheus Scientific Knowledge Graph Service",
    description="Automated knowledge extraction and semantic reasoning for geological sciences",
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
        kg_manager = service_components.get("knowledge_graph_manager")
        if kg_manager and await kg_manager.is_healthy():
            return {"status": "healthy", "service": "knowledge-graph"}
        else:
            raise HTTPException(status_code=503, detail="Service unhealthy")
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        required_components = [
            "knowledge_extractor",
            "semantic_reasoner",
            "knowledge_graph_manager",
            "temporal_causal_engine",
            "federated_discovery",
        ]

        for component_name in required_components:
            if component_name not in service_components:
                raise HTTPException(
                    status_code=503,
                    detail=f"Component {component_name} not initialized",
                )

        return {"status": "ready", "service": "knowledge-graph"}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8007)),
        reload=os.getenv("ENVIRONMENT") == "development",
    )
