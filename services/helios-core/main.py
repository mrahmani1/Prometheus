import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import consul
import redis.asyncio as redis
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from core.geological_physics_informed_neural_networks import GeologicalPINNEngine
from core.multi_physics_coupling_engine import MultiPhysicsCouplingEngine
from modules.volcano_watchtower import VolcanoWatchtowerCore
from modules.digital_twin_earth import DigitalTwinEarthCore
from modules.prospectivity_engine import ProspectivityEngineCore

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

# Configuration
CONSUL_HOST = os.getenv('CONSUL_HOST', 'localhost')
CONSUL_PORT = int(os.getenv('CONSUL_PORT', '8500'))
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
SERVICE_NAME = 'prometheus-helios-core'
SERVICE_PORT = int(os.getenv('PORT', '8001'))
SERVICE_HOST = os.getenv('SERVICE_HOST', 'localhost')

# Global instances
consul_client = None
redis_client = None
geological_engine = None
physics_engine = None
volcano_core = None
digital_twin = None
prospectivity_engine = None

# Pydantic models
class GeologicalAnalysisRequest(BaseModel):
    data_type: str = Field(..., description="Type of geological data")
    spatial_extent: Dict[str, float] = Field(..., description="Geographic bounds")
    temporal_range: Optional[Dict[str, str]] = Field(None, description="Time range")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")

class SimulationRequest(BaseModel):
    simulation_type: str = Field(..., description="Type of physics simulation")
    initial_conditions: Dict[str, Any] = Field(..., description="Initial conditions")
    boundary_conditions: Dict[str, Any] = Field(..., description="Boundary conditions")
    time_span: float = Field(..., description="Simulation time span")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Simulation parameters")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    modules_loaded: List[str]

class AnalysisResponse(BaseModel):
    status: str
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    correlation_id: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global consul_client, redis_client, geological_engine, physics_engine
    global volcano_core, digital_twin, prospectivity_engine
    
    try:
        # Initialize external connections
        consul_client = consul.Consul(host=CONSUL_HOST, port=CONSUL_PORT)
        redis_client = redis.from_url(REDIS_URL)
        
        # Test connections
        await redis_client.ping()
        consul_client.agent.self()
        
        logger.info("External connections established")
        
        # Initialize geological modules
        geological_engine = GeologicalPINNEngine()
        physics_engine = MultiPhysicsCouplingEngine()
        volcano_core = VolcanoWatchtowerCore()
        digital_twin = DigitalTwinEarthCore()
        prospectivity_engine = ProspectivityEngineCore()
        
        # Initialize engines
        await geological_engine.initialize()
        await physics_engine.initialize()
        await volcano_core.initialize()
        await digital_twin.initialize()
        await prospectivity_engine.initialize()
        
        logger.info("Geological modules initialized")
        
        # Register service with Consul
        service_id = f"{SERVICE_NAME}-{SERVICE_HOST}-{SERVICE_PORT}"
        consul_client.agent.service.register(
            name=SERVICE_NAME,
            service_id=service_id,
            address=SERVICE_HOST,
            port=SERVICE_PORT,
            check=consul.Check.http(f"http://{SERVICE_HOST}:{SERVICE_PORT}/health", interval="10s"),
            tags=['ai', 'geology', 'pinn', 'prometheus']
        )
        
        logger.info("Service registered with Consul", service_id=service_id)
        
        yield
        
    except Exception as e:
        logger.error("Failed to initialize service", error=str(e))
        raise
    finally:
        # Cleanup
        if consul_client:
            try:
                consul_client.agent.service.deregister(service_id)
                logger.info("Service deregistered from Consul")
            except Exception as e:
                logger.error("Failed to deregister service", error=str(e))
        
        if redis_client:
            await redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Helios Core Service",
    description="Physics-Informed Neural Network engine for geological analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Dependency to get correlation ID
async def get_correlation_id(x_correlation_id: Optional[str] = Header(None)) -> str:
    return x_correlation_id or "unknown"

# Dependency to get user context
async def get_user_context(
    x_user_id: Optional[str] = Header(None),
    x_user_role: Optional[str] = Header(None)
) -> Dict[str, str]:
    return {
        "user_id": x_user_id or "anonymous",
        "user_role": x_user_role or "user"
    }

# Health endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=structlog.processors.TimeStamper()._make_stamper("iso")(),
        version="1.0.0",
        modules_loaded=[
            "GeologicalPINNEngine",
            "MultiPhysicsCouplingEngine", 
            "VolcanoWatchtowerCore",
            "DigitalTwinEarthCore",
            "ProspectivityEngineCore"
        ]
    )

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Check Redis connection
        await redis_client.ping()
        
        # Check if modules are initialized
        if not all([geological_engine, physics_engine, volcano_core, digital_twin, prospectivity_engine]):
            raise HTTPException(status_code=503, detail="Modules not initialized")
        
        return {
            "status": "ready",
            "timestamp": structlog.processors.TimeStamper()._make_stamper("iso")(),
            "dependencies": {
                "redis": "connected",
                "consul": "connected",
                "modules": "initialized"
            }
        }
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    # TODO: Implement proper Prometheus metrics
    return {
        "requests_total": 0,
        "requests_duration_seconds": 0.0,
        "active_connections": 0
    }

# Geological analysis endpoints
@app.post("/analyze/geological", response_model=AnalysisResponse)
async def analyze_geological_data(
    request: GeologicalAnalysisRequest,
    background_tasks: BackgroundTasks,
    correlation_id: str = Depends(get_correlation_id),
    user_context: Dict[str, str] = Depends(get_user_context)
):
    """Perform geological data analysis using PINN engine"""
    try:
        logger.info("Geological analysis requested", 
                   correlation_id=correlation_id,
                   data_type=request.data_type,
                   user_id=user_context["user_id"])
        
        # Process with geological engine
        results = await geological_engine.analyze_geological_data(
            data_type=request.data_type,
            spatial_extent=request.spatial_extent,
            temporal_range=request.temporal_range,
            parameters=request.parameters
        )
        
        # Cache results in Redis
        background_tasks.add_task(
            cache_results,
            f"geological_analysis:{correlation_id}",
            results
        )
        
        return AnalysisResponse(
            status="success",
            results=results,
            metadata={
                "processing_time": results.get("processing_time", 0),
                "model_version": results.get("model_version", "1.0.0"),
                "confidence": results.get("confidence", 0.0)
            },
            correlation_id=correlation_id
        )
        
    except Exception as e:
        logger.error("Geological analysis failed", 
                    correlation_id=correlation_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/simulate/physics", response_model=AnalysisResponse)
async def run_physics_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    correlation_id: str = Depends(get_correlation_id),
    user_context: Dict[str, str] = Depends(get_user_context)
):
    """Run physics-informed simulation"""
    try:
        logger.info("Physics simulation requested",
                   correlation_id=correlation_id,
                   simulation_type=request.simulation_type,
                   user_id=user_context["user_id"])
        
        # Process with physics engine
        results = await physics_engine.run_simulation(
            simulation_type=request.simulation_type,
            initial_conditions=request.initial_conditions,
            boundary_conditions=request.boundary_conditions,
            time_span=request.time_span,
            parameters=request.parameters
        )
        
        # Cache results
        background_tasks.add_task(
            cache_results,
            f"physics_simulation:{correlation_id}",
            results
        )
        
        return AnalysisResponse(
            status="success",
            results=results,
            metadata={
                "simulation_time": results.get("simulation_time", 0),
                "convergence": results.get("convergence", False),
                "iterations": results.get("iterations", 0)
            },
            correlation_id=correlation_id
        )
        
    except Exception as e:
        logger.error("Physics simulation failed",
                    correlation_id=correlation_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@app.get("/models/status")
async def get_model_status(
    correlation_id: str = Depends(get_correlation_id)
):
    """Get status of loaded models"""
    try:
        status = {
            "geological_pinn": await geological_engine.get_status(),
            "physics_coupling": await physics_engine.get_status(),
            "volcano_watchtower": await volcano_core.get_status(),
            "digital_twin": await digital_twin.get_status(),
            "prospectivity": await prospectivity_engine.get_status()
        }
        
        return {
            "status": "success",
            "models": status,
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error("Failed to get model status",
                    correlation_id=correlation_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Background task for caching
async def cache_results(key: str, results: Dict[str, Any]):
    """Cache analysis results in Redis"""
    try:
        await redis_client.setex(
            key,
            3600,  # 1 hour TTL
            str(results)  # In production, use proper serialization
        )
        logger.info("Results cached", key=key)
    except Exception as e:
        logger.error("Failed to cache results", key=key, error=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error("HTTP exception", 
                status_code=exc.status_code,
                detail=exc.detail,
                path=request.url.path)
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Unhandled exception",
                error=str(exc),
                path=request.url.path)
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_config=None  # Use structlog configuration
    )