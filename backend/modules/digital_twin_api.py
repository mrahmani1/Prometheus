"""
Digital Twin Earth API Endpoints
RESTful API and real-time WebSocket interface for Digital Twin Earth system
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import time
import uuid
from pathlib import Path
import hashlib
import hmac
import jwt

# Web framework imports
try:
    from fastapi import FastAPI, HTTPException, Depends, Security, WebSocket, WebSocketDisconnect
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    warnings.warn("FastAPI not available. API endpoints will be limited.")

# Database imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Caching will be limited.")

# Import Digital Twin Earth modules
try:
    from .digital_twin_earth import DigitalTwinEarth, EarthSystemState, ModelConfiguration, ObservationData, PredictiveScenario
    from .digital_twin_integration import DigitalTwinEarthIntegrator, IntegrationConfig
    from .digital_twin_visualization import DigitalTwinVisualizationDashboard, VisualizationConfig
    TWIN_MODULES_AVAILABLE = True
except ImportError:
    TWIN_MODULES_AVAILABLE = False
    warnings.warn("Digital Twin modules not available.")

warnings.filterwarnings('ignore')

class APIVersion(Enum):
    V1 = "v1"
    V2 = "v2"

class AccessLevel(Enum):
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    PREMIUM = "premium"
    ADMIN = "admin"

# Pydantic models for API requests/responses
class ObservationRequest(BaseModel):
    observation_id: str = Field(..., description="Unique observation identifier")
    observation_type: str = Field(..., description="Type of observation")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_id: str = Field(..., description="Source identifier")
    location: Optional[Tuple[float, float, float]] = Field(None, description="Lat, Lon, Elevation")
    value: Union[float, Dict[str, Any]] = Field(..., description="Observation value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    quality_score: float = Field(1.0, ge=0.0, le=1.0, description="Quality score 0-1")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EarthStateResponse(BaseModel):
    timestamp: datetime
    atmosphere_state: Dict[str, Any]
    hydrosphere_state: Dict[str, Any]
    lithosphere_state: Dict[str, Any]
    biosphere_state: Dict[str, Any]
    cryosphere_state: Dict[str, Any]
    anthroposphere_state: Dict[str, Any]
    spatial_resolution: float
    uncertainty_metrics: Dict[str, Any]

class PredictionRequest(BaseModel):
    scenario_count: int = Field(5, ge=1, le=20, description="Number of scenarios to generate")
    time_horizon_hours: float = Field(24.0, ge=0.1, le=8760.0, description="Prediction time horizon in hours")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Confidence level")
    scenario_parameters: Dict[str, Any] = Field(default_factory=dict)

class PredictionResponse(BaseModel):
    scenario_id: str
    predicted_states: List[Dict[str, Any]]
    probability: float
    confidence_level: float
    time_horizon_hours: float
    uncertainty_bounds: Optional[List[float]]
    model_version: str

class IntegrationStatusResponse(BaseModel):
    is_initialized: bool
    active_integrations: List[str]
    sync_latency_ms: float
    data_throughput_mbps: float
    integration_health: str
    last_sync_timestamp: datetime
    component_status: Dict[str, bool]

class APIKeyModel(BaseModel):
    key_id: str
    user_id: str
    access_level: AccessLevel
    rate_limit_per_hour: int
    created_at: datetime
    expires_at: Optional[datetime]

@dataclass
class APIConfig:
    """Configuration for Digital Twin API"""
    host: str = "localhost"
    port: int = 8000
    enable_cors: bool = True
    enable_authentication: bool = True
    enable_rate_limiting: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    jwt_secret_key: str = "digital_twin_earth_secret"
    jwt_algorithm: str = "HS256"
    max_connections: int = 100
    enable_websockets: bool = True
    log_requests: bool = True

class DigitalTwinEarthAPI:
    """FastAPI application for Digital Twin Earth"""
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.app = None
        self.integrator = None
        self.visualization_dashboard = None
        
        # Authentication and authorization
        self.api_keys = {}
        self.rate_limits = {}
        self.security = HTTPBearer()
        
        # WebSocket connections
        self.websocket_connections = set()
        self.connection_rooms = {}
        
        # Caching
        self.cache = None
        if REDIS_AVAILABLE:
            try:
                self.cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}")
                self.cache = None
        
        # Request tracking
        self.request_count = 0
        self.start_time = datetime.utcnow()
    
    async def initialize(self, integrator: DigitalTwinEarthIntegrator = None, 
                        visualization_dashboard: DigitalTwinVisualizationDashboard = None):
        """Initialize the API with Digital Twin components"""
        try:
            if not FASTAPI_AVAILABLE:
                raise RuntimeError("FastAPI not available")
            
            self.integrator = integrator
            self.visualization_dashboard = visualization_dashboard
            
            # Create FastAPI app
            self.app = FastAPI(
                title="Digital Twin Earth API",
                description="RESTful API for Digital Twin Earth system",
                version="1.0.0",
                docs_url="/docs",
                redoc_url="/redoc"
            )
            
            # Setup CORS
            if self.config.enable_cors:
                self.app.add_middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )
            
            # Setup routes
            await self._setup_routes()
            
            # Setup WebSocket handlers
            if self.config.enable_websockets:
                await self._setup_websockets()
            
            # Setup authentication
            if self.config.enable_authentication:
                await self._setup_authentication()
            
            # Setup middleware
            await self._setup_middleware()
            
            self.logger.info("Digital Twin Earth API initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize API: {e}")
            raise
    
    async def _setup_routes(self):
        """Setup API routes"""
        
        # Health check endpoint
        @self.app.get("/health", tags=["System"])
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
                "request_count": self.request_count,
                "version": "1.0.0"
            }
        
        # System information endpoint
        @self.app.get("/info", tags=["System"])
        async def system_info():
            """Get system information"""
            return {
                "api_version": "1.0.0",
                "digital_twin_available": self.integrator is not None,
                "visualization_available": self.visualization_dashboard is not None,
                "features": {
                    "authentication": self.config.enable_authentication,
                    "rate_limiting": self.config.enable_rate_limiting,
                    "caching": self.config.enable_caching and self.cache is not None,
                    "websockets": self.config.enable_websockets
                }
            }
        
        # Digital Twin Earth state endpoints
        @self.app.get("/api/v1/earth/state", response_model=EarthStateResponse, tags=["Earth System"])
        async def get_earth_state(credentials: HTTPAuthorizationCredentials = Security(self.security)):
            """Get current Earth system state"""
            await self._verify_credentials(credentials)
            
            if not self.integrator or not self.integrator.digital_twin:
                raise HTTPException(status_code=503, detail="Digital Twin not available")
            
            # Check cache first
            cache_key = "earth_state_current"
            if self.cache:
                cached_state = self.cache.get(cache_key)
                if cached_state:
                    return json.loads(cached_state)
            
            try:
                current_state = self.integrator.digital_twin.get_current_state()
                if not current_state:
                    raise HTTPException(status_code=404, detail="No current Earth state available")
                
                response_data = EarthStateResponse(
                    timestamp=current_state.timestamp,
                    atmosphere_state=current_state.atmosphere_state,
                    hydrosphere_state=current_state.hydrosphere_state,
                    lithosphere_state=current_state.lithosphere_state,
                    biosphere_state=current_state.biosphere_state,
                    cryosphere_state=current_state.cryosphere_state,
                    anthroposphere_state=current_state.anthroposphere_state,
                    spatial_resolution=current_state.spatial_resolution,
                    uncertainty_metrics=current_state.uncertainty_metrics
                )
                
                # Cache the result
                if self.cache:
                    self.cache.setex(cache_key, self.config.cache_ttl_seconds, response_data.json())
                
                return response_data
                
            except Exception as e:
                self.logger.error(f"Error getting Earth state: {e}")
                raise HTTPException(status_code=500, detail="Failed to get Earth state")
        
        # Historical Earth state endpoint
        @self.app.get("/api/v1/earth/state/history", tags=["Earth System"])
        async def get_earth_state_history(
            start_time: datetime,
            end_time: datetime,
            resolution_minutes: int = 60,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Get historical Earth system states"""
            await self._verify_credentials(credentials)
            
            if not self.integrator:
                raise HTTPException(status_code=503, detail="Integration system not available")
            
            # Validate time range
            if end_time <= start_time:
                raise HTTPException(status_code=400, detail="End time must be after start time")
            
            if (end_time - start_time).days > 30:
                raise HTTPException(status_code=400, detail="Time range cannot exceed 30 days")
            
            try:
                # This would query historical data from database
                # For now, return mock data
                history = []
                current_time = start_time
                while current_time <= end_time:
                    # Generate mock historical state
                    mock_state = {
                        "timestamp": current_time.isoformat(),
                        "atmosphere_state": {"temperature": 15.0 + np.random.normal(0, 5)},
                        "hydrosphere_state": {"sea_level": 0.0 + np.random.normal(0, 0.1)},
                        "lithosphere_state": {"stress_field": np.random.normal(0, 1)},
                        "biosphere_state": {"vegetation_index": 0.7 + np.random.normal(0, 0.1)},
                        "cryosphere_state": {"ice_coverage": 0.1 + np.random.normal(0, 0.02)},
                        "anthroposphere_state": {"activity_level": 0.5 + np.random.normal(0, 0.1)}
                    }
                    history.append(mock_state)
                    current_time += timedelta(minutes=resolution_minutes)
                
                return {"history": history, "count": len(history)}
                
            except Exception as e:
                self.logger.error(f"Error getting Earth state history: {e}")
                raise HTTPException(status_code=500, detail="Failed to get Earth state history")
        
        # Observation endpoints
        @self.app.post("/api/v1/observations", tags=["Observations"])
        async def add_observation(
            observation: ObservationRequest,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Add real-time observation to Digital Twin"""
            await self._verify_credentials(credentials)
            
            if not self.integrator:
                raise HTTPException(status_code=503, detail="Integration system not available")
            
            try:
                # Convert to ObservationData object
                obs_data = ObservationData(
                    observation_id=observation.observation_id,
                    observation_type=observation.observation_type,
                    timestamp=observation.timestamp,
                    source_id=observation.source_id,
                    location=observation.location,
                    value=observation.value,
                    unit=observation.unit,
                    quality_score=observation.quality_score,
                    metadata=observation.metadata
                )
                
                # Add to integration system
                await self.integrator.add_real_time_observation(obs_data)
                
                # Broadcast to WebSocket clients
                await self._broadcast_to_websockets("observation_added", observation.dict())
                
                return {
                    "status": "success",
                    "observation_id": observation.observation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error adding observation: {e}")
                raise HTTPException(status_code=500, detail="Failed to add observation")
        
        @self.app.get("/api/v1/observations", tags=["Observations"])
        async def get_observations(
            start_time: datetime,
            end_time: datetime,
            observation_type: Optional[str] = None,
            source_id: Optional[str] = None,
            limit: int = Field(100, ge=1, le=1000),
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Get observations from specified time range"""
            await self._verify_credentials(credentials)
            
            # This would query the database for observations
            # For now, return mock data
            observations = []
            for i in range(min(limit, 10)):
                obs = {
                    "observation_id": f"obs_{i}",
                    "observation_type": observation_type or "temperature",
                    "timestamp": (start_time + timedelta(hours=i)).isoformat(),
                    "source_id": source_id or f"sensor_{i}",
                    "value": 20.0 + np.random.normal(0, 5),
                    "quality_score": 0.95
                }
                observations.append(obs)
            
            return {
                "observations": observations,
                "count": len(observations),
                "filters": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "observation_type": observation_type,
                    "source_id": source_id
                }
            }
        
        # Prediction endpoints
        @self.app.post("/api/v1/predictions", response_model=List[PredictionResponse], tags=["Predictions"])
        async def generate_predictions(
            request: PredictionRequest,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Generate predictive scenarios"""
            await self._verify_credentials(credentials)
            
            if not self.integrator or not self.integrator.digital_twin:
                raise HTTPException(status_code=503, detail="Digital Twin not available")
            
            try:
                time_horizon = timedelta(hours=request.time_horizon_hours)
                predictions = await self.integrator.get_current_predictions(time_horizon)
                
                # Convert to response format
                prediction_responses = []
                for pred in predictions[:request.scenario_count]:
                    response = PredictionResponse(
                        scenario_id=pred.scenario_id,
                        predicted_states=[state for state in pred.predicted_states],
                        probability=pred.probability,
                        confidence_level=pred.confidence_level,
                        time_horizon_hours=pred.time_horizon.total_seconds() / 3600,
                        uncertainty_bounds=pred.uncertainty_bounds,
                        model_version=pred.model_version
                    )
                    prediction_responses.append(response)
                
                # Broadcast to WebSocket clients
                await self._broadcast_to_websockets("predictions_generated", 
                                                  [pred.dict() for pred in prediction_responses])
                
                return prediction_responses
                
            except Exception as e:
                self.logger.error(f"Error generating predictions: {e}")
                raise HTTPException(status_code=500, detail="Failed to generate predictions")
        
        @self.app.get("/api/v1/predictions/{scenario_id}", tags=["Predictions"])
        async def get_prediction(
            scenario_id: str,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Get specific prediction scenario"""
            await self._verify_credentials(credentials)
            
            # This would query stored predictions
            # For now, return mock prediction
            return {
                "scenario_id": scenario_id,
                "predicted_states": [{"component": "atmosphere", "value": 15.5}],
                "probability": 0.75,
                "confidence_level": 0.95,
                "generated_at": datetime.utcnow().isoformat()
            }
        
        # Integration status endpoint
        @self.app.get("/api/v1/integration/status", response_model=IntegrationStatusResponse, tags=["Integration"])
        async def get_integration_status(credentials: HTTPAuthorizationCredentials = Security(self.security)):
            """Get integration system status"""
            await self._verify_credentials(credentials)
            
            if not self.integrator:
                raise HTTPException(status_code=503, detail="Integration system not available")
            
            try:
                status = await self.integrator.get_integration_status()
                
                return IntegrationStatusResponse(
                    is_initialized=status['is_initialized'],
                    active_integrations=status['active_integrations'],
                    sync_latency_ms=status['metrics']['sync_latency_ms'],
                    data_throughput_mbps=status['metrics']['data_throughput_mbps'],
                    integration_health=status['metrics']['integration_health'],
                    last_sync_timestamp=datetime.fromisoformat(status['metrics']['last_sync_timestamp']),
                    component_status=status['component_status']
                )
                
            except Exception as e:
                self.logger.error(f"Error getting integration status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get integration status")
        
        # Visualization endpoints
        @self.app.get("/api/v1/visualization/{viz_type}", tags=["Visualization"])
        async def get_visualization(
            viz_type: str,
            format: str = "json",
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Get visualization data"""
            await self._verify_credentials(credentials)
            
            if not self.visualization_dashboard:
                raise HTTPException(status_code=503, detail="Visualization dashboard not available")
            
            try:
                from .digital_twin_visualization import VisualizationType
                viz_enum = VisualizationType(viz_type)
                viz_data = await self.visualization_dashboard._generate_visualization(viz_enum)
                
                return {"visualization_type": viz_type, "data": viz_data, "format": format}
                
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid visualization type: {viz_type}")
            except Exception as e:
                self.logger.error(f"Error getting visualization: {e}")
                raise HTTPException(status_code=500, detail="Failed to get visualization")
        
        # Export endpoint
        @self.app.get("/api/v1/export/{data_type}", tags=["Export"])
        async def export_data(
            data_type: str,
            format: str = "json",
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Export Digital Twin data"""
            await self._verify_credentials(credentials)
            
            try:
                # This would implement actual data export
                export_data = {
                    "data_type": data_type,
                    "format": format,
                    "exported_at": datetime.utcnow().isoformat(),
                    "time_range": {
                        "start": start_time.isoformat() if start_time else None,
                        "end": end_time.isoformat() if end_time else None
                    },
                    "data": []  # Actual data would go here
                }
                
                return export_data
                
            except Exception as e:
                self.logger.error(f"Error exporting data: {e}")
                raise HTTPException(status_code=500, detail="Failed to export data")
    
    async def _setup_websockets(self):
        """Setup WebSocket handlers for real-time communication"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            connection_id = str(uuid.uuid4())
            self.websocket_connections.add(websocket)
            
            try:
                self.logger.info(f"WebSocket connection established: {connection_id}")
                
                # Send initial status
                if self.integrator:
                    status = await self.integrator.get_integration_status()
                    await websocket.send_json({
                        "type": "integration_status",
                        "data": status
                    })
                
                # Handle incoming messages
                while True:
                    data = await websocket.receive_json()
                    await self._handle_websocket_message(websocket, data)
                    
            except WebSocketDisconnect:
                self.websocket_connections.discard(websocket)
                self.logger.info(f"WebSocket connection closed: {connection_id}")
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                self.websocket_connections.discard(websocket)
        
        @self.app.websocket("/ws/earth-state")
        async def earth_state_websocket(websocket: WebSocket):
            await websocket.accept()
            
            try:
                # Send real-time Earth state updates
                while True:
                    if self.integrator and self.integrator.digital_twin:
                        current_state = self.integrator.digital_twin.get_current_state()
                        if current_state:
                            await websocket.send_json({
                                "type": "earth_state_update",
                                "timestamp": current_state.timestamp.isoformat(),
                                "data": self._serialize_earth_state(current_state)
                            })
                    
                    await asyncio.sleep(2)  # Send updates every 2 seconds
                    
            except WebSocketDisconnect:
                pass
            except Exception as e:
                self.logger.error(f"Earth state WebSocket error: {e}")
    
    async def _handle_websocket_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        try:
            message_type = data.get("type")
            
            if message_type == "subscribe":
                # Subscribe to specific data streams
                stream = data.get("stream")
                # Implementation would manage subscriptions
                await websocket.send_json({
                    "type": "subscription_confirmed",
                    "stream": stream
                })
            
            elif message_type == "request_prediction":
                # Request real-time prediction
                if self.integrator:
                    predictions = await self.integrator.get_current_predictions()
                    await websocket.send_json({
                        "type": "prediction_result",
                        "data": [self._serialize_prediction(pred) for pred in predictions]
                    })
            
            elif message_type == "add_observation":
                # Add observation via WebSocket
                obs_data = data.get("observation")
                if obs_data and self.integrator:
                    observation = ObservationData(**obs_data)
                    await self.integrator.add_real_time_observation(observation)
                    
                    await websocket.send_json({
                        "type": "observation_confirmed",
                        "observation_id": observation.observation_id
                    })
            
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    
    async def _broadcast_to_websockets(self, message_type: str, data: Any):
        """Broadcast message to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        message = {
            "type": message_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                self.logger.warning(f"Failed to send WebSocket message: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected
    
    async def _setup_authentication(self):
        """Setup authentication system"""
        # Create default API keys for testing
        self.api_keys = {
            "test_key_123": APIKeyModel(
                key_id="test_key_123",
                user_id="test_user",
                access_level=AccessLevel.AUTHENTICATED,
                rate_limit_per_hour=1000,
                created_at=datetime.utcnow(),
                expires_at=None
            )
        }
    
    async def _setup_middleware(self):
        """Setup API middleware"""
        
        @self.app.middleware("http")
        async def request_middleware(request, call_next):
            start_time = time.time()
            
            # Increment request count
            self.request_count += 1
            
            # Process request
            response = await call_next(request)
            
            # Log request if enabled
            if self.config.log_requests:
                process_time = time.time() - start_time
                self.logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
            
            return response
    
    async def _verify_credentials(self, credentials: HTTPAuthorizationCredentials):
        """Verify API credentials"""
        if not self.config.enable_authentication:
            return True
        
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        token = credentials.credentials
        
        # Check API key
        if token in self.api_keys:
            api_key = self.api_keys[token]
            
            # Check expiration
            if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                raise HTTPException(status_code=401, detail="API key expired")
            
            # Check rate limiting
            if self.config.enable_rate_limiting:
                if not await self._check_rate_limit(api_key):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            return api_key
        
        # Try JWT token
        try:
            payload = jwt.decode(token, self.config.jwt_secret_key, algorithms=[self.config.jwt_algorithm])
            # Validate JWT payload
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def _check_rate_limit(self, api_key: APIKeyModel) -> bool:
        """Check rate limiting for API key"""
        if not self.cache:
            return True  # No rate limiting without cache
        
        rate_key = f"rate_limit:{api_key.key_id}"
        current_hour = datetime.utcnow().strftime("%Y-%m-%d:%H")
        full_key = f"{rate_key}:{current_hour}"
        
        current_count = self.cache.get(full_key) or "0"
        if int(current_count) >= api_key.rate_limit_per_hour:
            return False
        
        # Increment counter
        self.cache.incr(full_key)
        self.cache.expire(full_key, 3600)  # Expire in 1 hour
        
        return True
    
    def _serialize_earth_state(self, earth_state: Any) -> Dict[str, Any]:
        """Serialize Earth system state for API response"""
        if not earth_state:
            return {}
        
        return {
            "timestamp": earth_state.timestamp.isoformat() if hasattr(earth_state, 'timestamp') else datetime.utcnow().isoformat(),
            "atmosphere_state": getattr(earth_state, 'atmosphere_state', {}),
            "hydrosphere_state": getattr(earth_state, 'hydrosphere_state', {}),
            "lithosphere_state": getattr(earth_state, 'lithosphere_state', {}),
            "biosphere_state": getattr(earth_state, 'biosphere_state', {}),
            "cryosphere_state": getattr(earth_state, 'cryosphere_state', {}),
            "anthroposphere_state": getattr(earth_state, 'anthroposphere_state', {}),
            "uncertainty_metrics": getattr(earth_state, 'uncertainty_metrics', {})
        }
    
    def _serialize_prediction(self, prediction: Any) -> Dict[str, Any]:
        """Serialize prediction for API response"""
        if not prediction:
            return {}
        
        return {
            "scenario_id": getattr(prediction, 'scenario_id', ''),
            "predicted_states": getattr(prediction, 'predicted_states', []),
            "probability": getattr(prediction, 'probability', 0.0),
            "confidence_level": getattr(prediction, 'confidence_level', 0.0),
            "time_horizon_hours": getattr(prediction, 'time_horizon', timedelta()).total_seconds() / 3600,
            "model_version": getattr(prediction, 'model_version', '1.0')
        }
    
    async def run_api(self, host: str = None, port: int = None, reload: bool = False):
        """Run the API server"""
        try:
            if not self.app:
                raise RuntimeError("API not initialized")
            
            host = host or self.config.host
            port = port or self.config.port
            
            self.logger.info(f"Starting Digital Twin Earth API on {host}:{port}")
            
            # Run with uvicorn
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                reload=reload,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Error running API server: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the API server"""
        try:
            # Close WebSocket connections
            for websocket in self.websocket_connections:
                try:
                    await websocket.close()
                except Exception:
                    pass
            
            self.websocket_connections.clear()
            
            # Close cache connection
            if self.cache:
                self.cache.close()
            
            self.logger.info("Digital Twin Earth API shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error during API shutdown: {e}")

# Example usage and testing
async def example_digital_twin_api():
    """Example of Digital Twin API usage"""
    
    print("=== Digital Twin Earth API Example ===")
    
    # Configuration
    api_config = APIConfig(
        host="localhost",
        port=8001,  # Use different port for testing
        enable_authentication=False,  # Disable auth for testing
        enable_rate_limiting=False,
        log_requests=True
    )
    
    # Initialize API
    api = DigitalTwinEarthAPI(api_config)
    
    try:
        # Initialize without integrator for demo
        await api.initialize()
        
        print("API initialized successfully")
        print(f"Available endpoints:")
        print(f"  GET  /health - Health check")
        print(f"  GET  /info - System information")
        print(f"  GET  /api/v1/earth/state - Current Earth state")
        print(f"  POST /api/v1/observations - Add observation")
        print(f"  POST /api/v1/predictions - Generate predictions")
        print(f"  GET  /api/v1/integration/status - Integration status")
        print(f"  WS   /ws - WebSocket endpoint")
        
        print(f"API documentation available at: http://{api_config.host}:{api_config.port}/docs")
        print("To run the API server, use: await api.run_api()")
        
    finally:
        await api.shutdown()
    
    print("Digital Twin API example completed!")

if __name__ == "__main__":
    print("=== Digital Twin Earth API System ===")
    print(f"Available features:")
    print(f"  FastAPI: {FASTAPI_AVAILABLE}")
    print(f"  Redis: {REDIS_AVAILABLE}")
    print(f"  Twin Modules: {TWIN_MODULES_AVAILABLE}")
    
    # Run example
    asyncio.run(example_digital_twin_api())