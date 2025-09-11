import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

import consul
import redis.asyncio as redis
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from monitoring.seismic_transformer import SeismicTransformer
from monitoring.eruption_forecaster_ml import EruptionForecasterML
from monitoring.caldera_system_classifier import CalderaSystemClassifier
from monitoring.multi_sensor_fusion import MultiSensorFusion
from monitoring.deformation_analyzer import DeformationAnalyzer
from monitoring.alert_generator import AlertGenerator

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
SERVICE_NAME = 'prometheus-volcano-watchtower'
SERVICE_PORT = int(os.getenv('PORT', '8003'))
SERVICE_HOST = os.getenv('SERVICE_HOST', 'localhost')

# Global instances
consul_client = None
redis_client = None
monitoring_systems = {}
websocket_connections = set()

# Pydantic models
class VolcanoMonitoringRequest(BaseModel):
    volcano_id: str = Field(..., description="Volcano identifier")
    monitoring_type: str = Field(..., description="Type of monitoring (seismic, deformation, gas, thermal)")
    time_window: Optional[Dict[str, str]] = Field(None, description="Time window for analysis")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Monitoring parameters")

class EruptionForecastRequest(BaseModel):
    volcano_id: str = Field(..., description="Volcano identifier")
    forecast_horizon: int = Field(default=30, description="Forecast horizon in days")
    confidence_level: float = Field(default=0.95, description="Confidence level for forecast")
    include_uncertainty: bool = Field(default=True, description="Include uncertainty quantification")

class AlertConfiguration(BaseModel):
    volcano_id: str = Field(..., description="Volcano identifier")
    alert_types: List[str] = Field(..., description="Types of alerts to monitor")
    thresholds: Dict[str, float] = Field(..., description="Alert thresholds")
    notification_channels: List[str] = Field(..., description="Notification channels")

class SeismicEventData(BaseModel):
    timestamp: datetime
    latitude: float
    longitude: float
    depth: float
    magnitude: float
    event_type: str
    confidence: float

class DeformationData(BaseModel):
    timestamp: datetime
    location: Dict[str, float]
    displacement: Dict[str, float]
    velocity: Dict[str, float]
    coherence: float

class MonitoringResponse(BaseModel):
    status: str
    volcano_id: str
    monitoring_type: str
    results: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ForecastResponse(BaseModel):
    status: str
    volcano_id: str
    forecast: Dict[str, Any]
    uncertainty: Optional[Dict[str, Any]] = None
    confidence: float
    valid_until: datetime

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global consul_client, redis_client, monitoring_systems
    
    try:
        # Initialize external connections
        consul_client = consul.Consul(host=CONSUL_HOST, port=CONSUL_PORT)
        redis_client = redis.from_url(REDIS_URL)
        
        # Test connections
        await redis_client.ping()
        consul_client.agent.self()
        
        logger.info("External connections established")
        
        # Initialize monitoring systems
        monitoring_systems = {
            'seismic_transformer': SeismicTransformer(),
            'eruption_forecaster': EruptionForecasterML(),
            'caldera_classifier': CalderaSystemClassifier(),
            'sensor_fusion': MultiSensorFusion(),
            'deformation_analyzer': DeformationAnalyzer(),
            'alert_generator': AlertGenerator()
        }
        
        # Initialize all monitoring systems
        for name, system in monitoring_systems.items():
            await system.initialize()
            logger.info(f"Initialized {name} monitoring system")
        
        logger.info("Volcano monitoring systems initialized")
        
        # Register service with Consul
        service_id = f"{SERVICE_NAME}-{SERVICE_HOST}-{SERVICE_PORT}"
        consul_client.agent.service.register(
            name=SERVICE_NAME,
            service_id=service_id,
            address=SERVICE_HOST,
            port=SERVICE_PORT,
            check=consul.Check.http(f"http://{SERVICE_HOST}:{SERVICE_PORT}/health", interval="10s"),
            tags=['volcano', 'monitoring', 'real-time', 'prometheus']
        )
        
        logger.info("Service registered with Consul", service_id=service_id)
        
        # Start background monitoring tasks
        asyncio.create_task(continuous_monitoring())
        
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
    title="Volcano Watchtower Service",
    description="Real-time volcanic monitoring and eruption forecasting service",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Dependencies
async def get_correlation_id(x_correlation_id: Optional[str] = Header(None)) -> str:
    return x_correlation_id or f"volcano-{datetime.now().isoformat()}"

async def get_user_context(
    x_user_id: Optional[str] = Header(None),
    x_user_role: Optional[str] = Header(None)
) -> Dict[str, str]:
    return {
        "user_id": x_user_id or "anonymous",
        "user_role": x_user_role or "user"
    }

# Health endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "monitoring_systems": list(monitoring_systems.keys()),
        "active_connections": len(websocket_connections)
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Check Redis connection
        await redis_client.ping()
        
        # Check monitoring systems
        system_status = {}
        for name, system in monitoring_systems.items():
            status = await system.get_status()
            system_status[name] = status
            if status.get('status') != 'ready':
                raise HTTPException(status_code=503, detail=f"System {name} not ready")
        
        return {
            "status": "ready",
            "timestamp": datetime.now().isoformat(),
            "dependencies": {
                "redis": "connected",
                "consul": "connected"
            },
            "monitoring_systems": system_status
        }
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

# Volcano monitoring endpoints
@app.post("/monitor", response_model=MonitoringResponse)
async def monitor_volcano(
    request: VolcanoMonitoringRequest,
    background_tasks: BackgroundTasks,
    correlation_id: str = Depends(get_correlation_id),
    user_context: Dict[str, str] = Depends(get_user_context)
):
    """Monitor volcano with specified parameters"""
    try:
        logger.info("Volcano monitoring requested",
                   correlation_id=correlation_id,
                   volcano_id=request.volcano_id,
                   monitoring_type=request.monitoring_type,
                   user_id=user_context["user_id"])
        
        # Get appropriate monitoring system
        if request.monitoring_type == "seismic":
            system = monitoring_systems['seismic_transformer']
        elif request.monitoring_type == "deformation":
            system = monitoring_systems['deformation_analyzer']
        elif request.monitoring_type == "multi_sensor":
            system = monitoring_systems['sensor_fusion']
        else:
            raise HTTPException(status_code=400, detail=f"Unknown monitoring type: {request.monitoring_type}")
        
        # Perform monitoring analysis
        results = await system.analyze_volcano(
            volcano_id=request.volcano_id,
            time_window=request.time_window,
            parameters=request.parameters
        )
        
        # Cache results
        cache_key = f"monitoring:{request.volcano_id}:{request.monitoring_type}:{correlation_id}"
        await redis_client.setex(
            cache_key,
            1800,  # 30 minutes TTL
            json.dumps(results, default=str)
        )
        
        # Broadcast to WebSocket connections
        background_tasks.add_task(
            broadcast_monitoring_update,
            volcano_id=request.volcano_id,
            monitoring_type=request.monitoring_type,
            results=results
        )
        
        logger.info("Volcano monitoring completed",
                   correlation_id=correlation_id,
                   volcano_id=request.volcano_id)
        
        return MonitoringResponse(
            status="success",
            volcano_id=request.volcano_id,
            monitoring_type=request.monitoring_type,
            results=results,
            timestamp=datetime.now(),
            metadata={
                "correlation_id": correlation_id,
                "processing_time": results.get("processing_time", 0),
                "data_sources": results.get("data_sources", [])
            }
        )
        
    except Exception as e:
        logger.error("Volcano monitoring failed",
                    correlation_id=correlation_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Monitoring failed: {str(e)}")

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_eruption(
    request: EruptionForecastRequest,
    correlation_id: str = Depends(get_correlation_id),
    user_context: Dict[str, str] = Depends(get_user_context)
):
    """Generate eruption forecast for volcano"""
    try:
        logger.info("Eruption forecast requested",
                   correlation_id=correlation_id,
                   volcano_id=request.volcano_id,
                   forecast_horizon=request.forecast_horizon,
                   user_id=user_context["user_id"])
        
        forecaster = monitoring_systems['eruption_forecaster']
        
        # Generate forecast
        forecast_result = await forecaster.generate_forecast(
            volcano_id=request.volcano_id,
            forecast_horizon=request.forecast_horizon,
            confidence_level=request.confidence_level,
            include_uncertainty=request.include_uncertainty
        )
        
        # Cache forecast
        cache_key = f"forecast:{request.volcano_id}:{correlation_id}"
        await redis_client.setex(
            cache_key,
            3600,  # 1 hour TTL
            json.dumps(forecast_result, default=str)
        )
        
        logger.info("Eruption forecast completed",
                   correlation_id=correlation_id,
                   volcano_id=request.volcano_id,
                   forecast_probability=forecast_result.get("eruption_probability", 0))
        
        return ForecastResponse(
            status="success",
            volcano_id=request.volcano_id,
            forecast=forecast_result["forecast"],
            uncertainty=forecast_result.get("uncertainty") if request.include_uncertainty else None,
            confidence=forecast_result["confidence"],
            valid_until=datetime.now() + timedelta(hours=24)
        )
        
    except Exception as e:
        logger.error("Eruption forecast failed",
                    correlation_id=correlation_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

@app.post("/alerts/configure")
async def configure_alerts(
    config: AlertConfiguration,
    correlation_id: str = Depends(get_correlation_id)
):
    """Configure alert system for volcano"""
    try:
        alert_generator = monitoring_systems['alert_generator']
        
        result = await alert_generator.configure_alerts(
            volcano_id=config.volcano_id,
            alert_types=config.alert_types,
            thresholds=config.thresholds,
            notification_channels=config.notification_channels
        )
        
        return {
            "status": "success",
            "message": f"Alerts configured for volcano {config.volcano_id}",
            "configuration": result,
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error("Alert configuration failed",
                    correlation_id=correlation_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Alert configuration failed: {str(e)}")

@app.get("/volcanoes/{volcano_id}/status")
async def get_volcano_status(
    volcano_id: str,
    correlation_id: str = Depends(get_correlation_id)
):
    """Get current status of volcano"""
    try:
        # Get cached monitoring data
        cache_pattern = f"monitoring:{volcano_id}:*"
        keys = await redis_client.keys(cache_pattern)
        
        status_data = {}
        for key in keys:
            data = await redis_client.get(key)
            if data:
                monitoring_type = key.decode().split(':')[2]
                status_data[monitoring_type] = json.loads(data)
        
        # Get latest forecast
        forecast_pattern = f"forecast:{volcano_id}:*"
        forecast_keys = await redis_client.keys(forecast_pattern)
        latest_forecast = None
        
        if forecast_keys:
            # Get most recent forecast
            latest_key = sorted(forecast_keys)[-1]
            forecast_data = await redis_client.get(latest_key)
            if forecast_data:
                latest_forecast = json.loads(forecast_data)
        
        return {
            "volcano_id": volcano_id,
            "status": "active" if status_data else "inactive",
            "monitoring_data": status_data,
            "latest_forecast": latest_forecast,
            "last_updated": datetime.now().isoformat(),
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error("Failed to get volcano status",
                    volcano_id=volcano_id,
                    correlation_id=correlation_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@app.get("/systems/status")
async def get_systems_status():
    """Get status of all monitoring systems"""
    try:
        status = {}
        for name, system in monitoring_systems.items():
            system_status = await system.get_status()
            status[name] = system_status
        
        return {
            "status": "success",
            "monitoring_systems": status,
            "total_systems": len(monitoring_systems),
            "active_systems": sum(1 for s in status.values() if s.get('status') == 'ready')
        }
        
    except Exception as e:
        logger.error("Failed to get systems status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Systems status failed: {str(e)}")

# WebSocket endpoint for real-time data streaming
@app.websocket("/ws/{volcano_id}")
async def websocket_endpoint(websocket: WebSocket, volcano_id: str):
    """WebSocket endpoint for real-time volcano monitoring data"""
    await websocket.accept()
    websocket_connections.add(websocket)
    
    try:
        logger.info("WebSocket connection established", volcano_id=volcano_id)
        
        # Send initial status
        await websocket.send_json({
            "type": "connection_established",
            "volcano_id": volcano_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_json()
                
                # Handle different message types
                if data.get("type") == "subscribe":
                    await websocket.send_json({
                        "type": "subscription_confirmed",
                        "volcano_id": volcano_id,
                        "monitoring_types": data.get("monitoring_types", [])
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("WebSocket error", volcano_id=volcano_id, error=str(e))
                break
    
    finally:
        websocket_connections.discard(websocket)
        logger.info("WebSocket connection closed", volcano_id=volcano_id)

# Background tasks
async def broadcast_monitoring_update(volcano_id: str, monitoring_type: str, results: Dict[str, Any]):
    """Broadcast monitoring updates to WebSocket connections"""
    if not websocket_connections:
        return
    
    message = {
        "type": "monitoring_update",
        "volcano_id": volcano_id,
        "monitoring_type": monitoring_type,
        "data": results,
        "timestamp": datetime.now().isoformat()
    }
    
    # Send to all connected clients
    disconnected = set()
    for websocket in websocket_connections:
        try:
            await websocket.send_json(message)
        except Exception:
            disconnected.add(websocket)
    
    # Remove disconnected clients
    websocket_connections -= disconnected

async def continuous_monitoring():
    """Continuous background monitoring task"""
    logger.info("Starting continuous monitoring")
    
    while True:
        try:
            # This would implement actual continuous monitoring
            # For now, just log the activity
            logger.info("Continuous monitoring cycle")
            
            # Sleep for monitoring interval
            await asyncio.sleep(60)  # 1 minute intervals
            
        except Exception as e:
            logger.error("Continuous monitoring error", error=str(e))
            await asyncio.sleep(10)  # Short delay before retry

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_config=None
    )