"""
Natural Language Processing Service for Prometheus AI Platform
Handles geological query understanding and conversational responses
"""

import os
from contextlib import asynccontextmanager

import structlog
import uvicorn
from api.nlp_routes import router as nlp_router
from core.entity_extractor import GeologicalEntityExtractor
from core.intent_classifier import IntentClassifier
from core.query_translator import QueryTranslator
from core.response_generator import ConversationalResponseGenerator
from fastapi import FastAPI
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

# Global NLP components
nlp_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize NLP components on startup"""
    logger.info("Initializing NLP service components")

    try:
        nlp_components["intent_classifier"] = IntentClassifier()
        nlp_components["entity_extractor"] = GeologicalEntityExtractor()
        nlp_components["query_translator"] = QueryTranslator()
        nlp_components["response_generator"] = ConversationalResponseGenerator()

        logger.info("NLP service components initialized successfully")
        yield

    except Exception as e:
        logger.error("Failed to initialize NLP components", error=str(e))
        raise
    finally:
        logger.info("Shutting down NLP service")

# Create FastAPI app
app = FastAPI(
    title="Prometheus NLP Service",
    description="Natural Language Processing service for geological queries and conversational AI",
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

# Include NLP routes
app.include_router(nlp_router, prefix="/api/v1/nlp", tags=["nlp"])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "nlp-service"}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Check if all NLP components are initialized
        required_components = [
            "intent_classifier",
            "entity_extractor",
            "query_translator",
            "response_generator",
        ]
        missing_components = []

        for component in required_components:
            if component not in nlp_components:
                missing_components.append(component)

        if missing_components:
            return {
                "status": "not_ready",
                "service": "nlp-service",
                "missing_components": missing_components,
            }

        return {"status": "ready", "service": "nlp-service"}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return {"status": "error", "service": "nlp-service", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=8005,
        reload=True,
        log_config=None,  # Use structlog configuration
    )
