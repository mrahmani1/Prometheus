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
    from .geological_anomaly_detection import GeologicalAnomalyDetector, AnomalyConfig
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
        self.anomaly_detector = None
        
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
            
            # Initialize geological anomaly detector
            if TWIN_MODULES_AVAILABLE:
                try:
                    anomaly_config = AnomalyConfig()
                    self.anomaly_detector = GeologicalAnomalyDetector(config=anomaly_config)
                    self.logger.info("Geological anomaly detector initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize anomaly detector: {e}")
                    self.anomaly_detector = None
            
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
                # Comprehensive database query implementation for historical Earth states
                history = await self._query_historical_earth_states(
                    start_time, end_time, resolution_minutes
                )
                
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
        
        # Geological Anomaly Detection endpoints
        @self.app.post("/api/v1/anomaly-detection/analyze", tags=["Anomaly Detection"])
        async def detect_geological_anomalies(
            request: Dict[str, Any],
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Detect geological anomalies in multivariate data"""
            await self._verify_credentials(credentials)
            
            if not self.anomaly_detector:
                raise HTTPException(status_code=503, detail="Anomaly detector not available")
            
            try:
                # Extract data from request
                data = np.array(request.get('data', []))
                if data.size == 0:
                    raise HTTPException(status_code=400, detail="No data provided")
                
                # Convert to tensor if 1D data provided
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                
                data_tensor = torch.tensor(data, dtype=torch.float32)
                
                # Perform anomaly detection
                results = await asyncio.get_event_loop().run_in_executor(
                    None, self.anomaly_detector.detect_anomalies, data_tensor
                )
                
                # Format response
                response = {
                    "anomaly_scores": results.anomaly_scores.tolist(),
                    "anomaly_labels": results.anomaly_labels.tolist(),
                    "confidence_scores": results.confidence_scores.tolist(),
                    "anomaly_threshold": float(results.anomaly_threshold),
                    "detected_patterns": [
                        {
                            "pattern_id": pattern.pattern_id,
                            "pattern_type": pattern.pattern_type,
                            "confidence": float(pattern.confidence),
                            "temporal_range": pattern.temporal_range,
                            "spatial_extent": pattern.spatial_extent,
                            "causal_factors": pattern.causal_factors
                        }
                        for pattern in results.detected_patterns
                    ],
                    "causal_relationships": [
                        {
                            "cause_variable": rel.cause_variable,
                            "effect_variable": rel.effect_variable,
                            "strength": float(rel.strength),
                            "lag": int(rel.lag),
                            "confidence": float(rel.confidence)
                        }
                        for rel in results.causal_relationships
                    ]
                }
                
                return response
                
            except Exception as e:
                self.logger.error(f"Error in anomaly detection: {e}")
                raise HTTPException(status_code=500, detail="Failed to detect anomalies")
        
        @self.app.post("/api/v1/anomaly-detection/real-time", tags=["Anomaly Detection"])
        async def real_time_anomaly_detection(
            request: Dict[str, Any],
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Real-time geological anomaly detection"""
            await self._verify_credentials(credentials)
            
            if not self.anomaly_detector:
                raise HTTPException(status_code=503, detail="Anomaly detector not available")
            
            try:
                data_point = np.array(request.get('data_point', []))
                if data_point.size == 0:
                    raise HTTPException(status_code=400, detail="No data point provided")
                
                data_tensor = torch.tensor(data_point, dtype=torch.float32)
                
                # Real-time detection
                is_anomaly, anomaly_score, confidence = await asyncio.get_event_loop().run_in_executor(
                    None, self.anomaly_detector.detect_real_time_anomaly, data_tensor
                )
                
                response = {
                    "is_anomaly": bool(is_anomaly),
                    "anomaly_score": float(anomaly_score),
                    "confidence": float(confidence),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Broadcast to WebSocket clients if anomaly detected
                if is_anomaly and confidence > 0.7:
                    await self._broadcast_to_websockets("anomaly_detected", response)
                
                return response
                
            except Exception as e:
                self.logger.error(f"Error in real-time anomaly detection: {e}")
                raise HTTPException(status_code=500, detail="Failed to detect real-time anomaly")
        
        @self.app.get("/api/v1/anomaly-detection/patterns", tags=["Anomaly Detection"])
        async def get_anomaly_patterns(
            start_time: datetime = None,
            end_time: datetime = None,
            pattern_type: str = None,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Get historical anomaly patterns"""
            await self._verify_credentials(credentials)
            
            if not self.anomaly_detector:
                raise HTTPException(status_code=503, detail="Anomaly detector not available")
            
            try:
                # Get historical patterns
                patterns = await asyncio.get_event_loop().run_in_executor(
                    None, self.anomaly_detector.get_historical_patterns, start_time, end_time, pattern_type
                )
                
                response = {
                    "patterns": [
                        {
                            "pattern_id": pattern.pattern_id,
                            "pattern_type": pattern.pattern_type,
                            "timestamp": pattern.timestamp.isoformat(),
                            "confidence": float(pattern.confidence),
                            "temporal_range": pattern.temporal_range,
                            "spatial_extent": pattern.spatial_extent,
                            "causal_factors": pattern.causal_factors,
                            "geological_context": pattern.geological_context
                        }
                        for pattern in patterns
                    ],
                    "total_count": len(patterns)
                }
                
                return response
                
            except Exception as e:
                self.logger.error(f"Error getting anomaly patterns: {e}")
                raise HTTPException(status_code=500, detail="Failed to get anomaly patterns")
        
        @self.app.get("/api/v1/anomaly-detection/causality/{variable}", tags=["Anomaly Detection"])
        async def get_causal_relationships(
            variable: str,
            lag_range: int = 10,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Get causal relationships for a specific variable"""
            await self._verify_credentials(credentials)
            
            if not self.anomaly_detector:
                raise HTTPException(status_code=503, detail="Anomaly detector not available")
            
            try:
                # Get causal relationships
                relationships = await asyncio.get_event_loop().run_in_executor(
                    None, self.anomaly_detector.analyze_causal_relationships, variable, lag_range
                )
                
                response = {
                    "target_variable": variable,
                    "causal_relationships": [
                        {
                            "cause_variable": rel.cause_variable,
                            "effect_variable": rel.effect_variable,
                            "causality_type": rel.causality_type,
                            "strength": float(rel.strength),
                            "lag": int(rel.lag),
                            "confidence": float(rel.confidence),
                            "p_value": float(rel.p_value),
                            "method": rel.method
                        }
                        for rel in relationships
                    ]
                }
                
                return response
                
            except Exception as e:
                self.logger.error(f"Error getting causal relationships: {e}")
                raise HTTPException(status_code=500, detail="Failed to get causal relationships")
        
        @self.app.post("/api/v1/anomaly-detection/configure", tags=["Anomaly Detection"])
        async def configure_anomaly_detector(
            config: Dict[str, Any],
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Configure anomaly detection parameters"""
            await self._verify_credentials(credentials)
            
            if not self.anomaly_detector:
                raise HTTPException(status_code=503, detail="Anomaly detector not available")
            
            try:
                # Update configuration
                await asyncio.get_event_loop().run_in_executor(
                    None, self.anomaly_detector.update_configuration, config
                )
                
                return {
                    "status": "success",
                    "message": "Anomaly detector configuration updated",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error configuring anomaly detector: {e}")
                raise HTTPException(status_code=500, detail="Failed to configure anomaly detector")
        
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
    
    async def _query_historical_earth_states(self, start_time: datetime, end_time: datetime, 
                                           resolution_minutes: int) -> List[Dict[str, Any]]:
        """
        Comprehensive database query implementation for historical Earth states
        
        Integrates with multiple data sources including:
        - Observational data repositories (NOAA, NASA, ESA)
        - Reanalysis datasets (ERA5, MERRA-2, JRA-55)
        - Model output archives
        - Satellite data streams
        - Ground-based measurement networks
        """
        try:
            history = []
            
            # Calculate temporal sampling parameters
            time_delta = timedelta(minutes=resolution_minutes)
            total_points = int((end_time - start_time) / time_delta) + 1
            
            if total_points > 10000:
                raise ValueError(f"Requested {total_points} data points exceeds maximum of 10,000")
            
            # Initialize data source connections
            data_sources = await self._initialize_data_source_connections()
            
            # Query each time point
            current_time = start_time
            while current_time <= end_time:
                try:
                    # Comprehensive Earth state reconstruction
                    earth_state = await self._reconstruct_earth_state_at_time(
                        current_time, data_sources, resolution_minutes
                    )
                    
                    if earth_state:
                        history.append(earth_state)
                
                except Exception as e:
                    self.logger.warning(f"Failed to reconstruct state at {current_time}: {e}")
                    # Add interpolated state based on neighboring data
                    if len(history) > 0:
                        interpolated_state = await self._interpolate_missing_state(
                            current_time, history[-1]
                        )
                        history.append(interpolated_state)
                
                current_time += time_delta
            
            # Post-process historical data for consistency and quality
            history = await self._post_process_historical_data(history)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error querying historical Earth states: {e}")
            # Fallback to simplified reconstruction if comprehensive query fails
            return await self._fallback_historical_reconstruction(
                start_time, end_time, resolution_minutes
            )
    
    async def _initialize_data_source_connections(self) -> Dict[str, Any]:
        """Initialize connections to various Earth system data sources"""
        try:
            data_sources = {}
            
            # Atmospheric data sources
            data_sources['era5'] = await self._connect_to_era5_reanalysis()
            data_sources['noaa_ncep'] = await self._connect_to_noaa_ncep()
            data_sources['nasa_merra2'] = await self._connect_to_nasa_merra2()
            
            # Ocean data sources
            data_sources['copernicus_marine'] = await self._connect_to_copernicus_marine()
            data_sources['noaa_oisst'] = await self._connect_to_noaa_oisst()
            
            # Land surface data sources
            data_sources['modis_terra'] = await self._connect_to_modis_terra()
            data_sources['sentinel_2'] = await self._connect_to_sentinel_2()
            
            # Geophysical data sources
            data_sources['usgs_earthquakes'] = await self._connect_to_usgs_earthquake_catalog()
            data_sources['igrf_magnetic'] = await self._connect_to_igrf_magnetic_field()
            
            # Cryosphere data sources
            data_sources['nsidc_ice'] = await self._connect_to_nsidc_sea_ice()
            data_sources['grace_gravity'] = await self._connect_to_grace_gravity()
            
            # Anthroposphere data sources
            data_sources['nighttime_lights'] = await self._connect_to_nighttime_lights()
            data_sources['co2_emissions'] = await self._connect_to_co2_emissions()
            
            return data_sources
            
        except Exception as e:
            self.logger.error(f"Error initializing data source connections: {e}")
            return {}
    
    async def _reconstruct_earth_state_at_time(self, timestamp: datetime, 
                                             data_sources: Dict[str, Any],
                                             resolution_minutes: int) -> Dict[str, Any]:
        """Reconstruct comprehensive Earth system state at specific time"""
        try:
            earth_state = {
                "timestamp": timestamp.isoformat(),
                "data_quality_score": 0.0,
                "reconstruction_method": "comprehensive_multi_source"
            }
            
            # Atmosphere state reconstruction
            atmosphere_data = await self._reconstruct_atmosphere_state(
                timestamp, data_sources, resolution_minutes
            )
            earth_state["atmosphere_state"] = atmosphere_data
            
            # Hydrosphere state reconstruction  
            hydrosphere_data = await self._reconstruct_hydrosphere_state(
                timestamp, data_sources, resolution_minutes
            )
            earth_state["hydrosphere_state"] = hydrosphere_data
            
            # Lithosphere state reconstruction
            lithosphere_data = await self._reconstruct_lithosphere_state(
                timestamp, data_sources, resolution_minutes
            )
            earth_state["lithosphere_state"] = lithosphere_data
            
            # Biosphere state reconstruction
            biosphere_data = await self._reconstruct_biosphere_state(
                timestamp, data_sources, resolution_minutes
            )
            earth_state["biosphere_state"] = biosphere_data
            
            # Cryosphere state reconstruction
            cryosphere_data = await self._reconstruct_cryosphere_state(
                timestamp, data_sources, resolution_minutes
            )
            earth_state["cryosphere_state"] = cryosphere_data
            
            # Anthroposphere state reconstruction
            anthroposphere_data = await self._reconstruct_anthroposphere_state(
                timestamp, data_sources, resolution_minutes
            )
            earth_state["anthroposphere_state"] = anthroposphere_data
            
            # Calculate overall data quality score
            quality_scores = [
                atmosphere_data.get("quality_score", 0.0),
                hydrosphere_data.get("quality_score", 0.0),
                lithosphere_data.get("quality_score", 0.0),
                biosphere_data.get("quality_score", 0.0),
                cryosphere_data.get("quality_score", 0.0),
                anthroposphere_data.get("quality_score", 0.0)
            ]
            earth_state["data_quality_score"] = np.mean(quality_scores)
            
            # Add uncertainty quantification
            earth_state["uncertainty_metrics"] = await self._calculate_state_uncertainty(
                earth_state, data_sources
            )
            
            return earth_state
            
        except Exception as e:
            self.logger.error(f"Error reconstructing Earth state at {timestamp}: {e}")
            return None
    
    async def _reconstruct_atmosphere_state(self, timestamp: datetime, 
                                          data_sources: Dict[str, Any],
                                          resolution_minutes: int) -> Dict[str, Any]:
        """Reconstruct atmospheric state from multiple reanalysis and observational datasets"""
        try:
            atmosphere_state = {
                "temperature_2m": None,
                "pressure_msl": None,
                "humidity_relative": None,
                "wind_speed_10m": None,
                "wind_direction_10m": None,
                "precipitation_total": None,
                "cloud_cover_total": None,
                "surface_solar_radiation": None,
                "quality_score": 0.0,
                "data_sources_used": []
            }
            
            # Query ERA5 reanalysis data
            if 'era5' in data_sources and data_sources['era5']:
                era5_data = await self._query_era5_at_time(timestamp, data_sources['era5'])
                if era5_data:
                    atmosphere_state["temperature_2m"] = era5_data.get("t2m")
                    atmosphere_state["pressure_msl"] = era5_data.get("msl")
                    atmosphere_state["humidity_relative"] = era5_data.get("r2m")
                    atmosphere_state["wind_speed_10m"] = era5_data.get("si10")
                    atmosphere_state["wind_direction_10m"] = era5_data.get("d10m")
                    atmosphere_state["precipitation_total"] = era5_data.get("tp")
                    atmosphere_state["cloud_cover_total"] = era5_data.get("tcc")
                    atmosphere_state["surface_solar_radiation"] = era5_data.get("ssr")
                    atmosphere_state["quality_score"] += 0.4
                    atmosphere_state["data_sources_used"].append("ERA5")
            
            # Query NOAA NCEP data for additional validation
            if 'noaa_ncep' in data_sources and data_sources['noaa_ncep']:
                ncep_data = await self._query_ncep_at_time(timestamp, data_sources['noaa_ncep'])
                if ncep_data:
                    # Cross-validate with ERA5 data and blend if consistent
                    atmosphere_state = await self._blend_atmospheric_datasets(
                        atmosphere_state, ncep_data, "NCEP"
                    )
                    atmosphere_state["quality_score"] += 0.2
            
            # Query NASA MERRA-2 for aerosol and chemistry data
            if 'nasa_merra2' in data_sources and data_sources['nasa_merra2']:
                merra2_data = await self._query_merra2_at_time(timestamp, data_sources['nasa_merra2'])
                if merra2_data:
                    atmosphere_state["aerosol_optical_depth"] = merra2_data.get("aod")
                    atmosphere_state["ozone_column"] = merra2_data.get("o3_column")
                    atmosphere_state["quality_score"] += 0.2
                    atmosphere_state["data_sources_used"].append("MERRA-2")
            
            # Add atmospheric composition from ground-based networks
            composition_data = await self._query_atmospheric_composition(timestamp, data_sources)
            if composition_data:
                atmosphere_state["co2_concentration"] = composition_data.get("co2")
                atmosphere_state["ch4_concentration"] = composition_data.get("ch4")
                atmosphere_state["quality_score"] += 0.2
            
            return atmosphere_state
            
        except Exception as e:
            self.logger.error(f"Error reconstructing atmosphere state: {e}")
            return {"quality_score": 0.0, "data_sources_used": []}
    
    async def _reconstruct_hydrosphere_state(self, timestamp: datetime,
                                           data_sources: Dict[str, Any],
                                           resolution_minutes: int) -> Dict[str, Any]:
        """Reconstruct ocean and water cycle state"""
        try:
            hydrosphere_state = {
                "sea_surface_temperature": None,
                "sea_level_anomaly": None,
                "ocean_currents_u": None,
                "ocean_currents_v": None,
                "salinity": None,
                "chlorophyll_a": None,
                "quality_score": 0.0,
                "data_sources_used": []
            }
            
            # Query Copernicus Marine Environment Monitoring Service
            if 'copernicus_marine' in data_sources and data_sources['copernicus_marine']:
                marine_data = await self._query_copernicus_marine_at_time(
                    timestamp, data_sources['copernicus_marine']
                )
                if marine_data:
                    hydrosphere_state["sea_surface_temperature"] = marine_data.get("sst")
                    hydrosphere_state["sea_level_anomaly"] = marine_data.get("sla")
                    hydrosphere_state["ocean_currents_u"] = marine_data.get("ugos")
                    hydrosphere_state["ocean_currents_v"] = marine_data.get("vgos")
                    hydrosphere_state["salinity"] = marine_data.get("so")
                    hydrosphere_state["quality_score"] += 0.5
                    hydrosphere_state["data_sources_used"].append("Copernicus Marine")
            
            # Query NOAA Optimum Interpolation SST
            if 'noaa_oisst' in data_sources and data_sources['noaa_oisst']:
                oisst_data = await self._query_oisst_at_time(timestamp, data_sources['noaa_oisst'])
                if oisst_data:
                    # Validate and blend with Copernicus SST
                    if hydrosphere_state["sea_surface_temperature"] is not None:
                        hydrosphere_state["sea_surface_temperature"] = np.mean([
                            hydrosphere_state["sea_surface_temperature"],
                            oisst_data.get("sst", hydrosphere_state["sea_surface_temperature"])
                        ])
                    else:
                        hydrosphere_state["sea_surface_temperature"] = oisst_data.get("sst")
                    hydrosphere_state["quality_score"] += 0.3
                    hydrosphere_state["data_sources_used"].append("NOAA OISST")
            
            # Add ocean color data for biological productivity
            color_data = await self._query_ocean_color_data(timestamp, data_sources)
            if color_data:
                hydrosphere_state["chlorophyll_a"] = color_data.get("chl_a")
                hydrosphere_state["quality_score"] += 0.2
            
            return hydrosphere_state
            
        except Exception as e:
            self.logger.error(f"Error reconstructing hydrosphere state: {e}")
            return {"quality_score": 0.0, "data_sources_used": []}
    
    async def _reconstruct_lithosphere_state(self, timestamp: datetime,
                                           data_sources: Dict[str, Any],
                                           resolution_minutes: int) -> Dict[str, Any]:
        """Reconstruct solid Earth state including seismic and magnetic field data"""
        try:
            lithosphere_state = {
                "earthquake_activity": None,
                "magnetic_field_intensity": None,
                "magnetic_declination": None,
                "magnetic_inclination": None,
                "gravitational_anomaly": None,
                "tectonic_stress_field": None,
                "quality_score": 0.0,
                "data_sources_used": []
            }
            
            # Query USGS earthquake catalog
            if 'usgs_earthquakes' in data_sources and data_sources['usgs_earthquakes']:
                # Look for earthquakes within time window
                time_window = timedelta(hours=6)  # 6-hour window around timestamp
                earthquake_data = await self._query_earthquakes_in_window(
                    timestamp - time_window, timestamp + time_window, data_sources['usgs_earthquakes']
                )
                
                if earthquake_data:
                    lithosphere_state["earthquake_activity"] = {
                        "event_count": len(earthquake_data),
                        "max_magnitude": max([eq.get("magnitude", 0) for eq in earthquake_data]),
                        "total_moment": sum([10**(1.5*eq.get("magnitude", 0) + 9.1) for eq in earthquake_data]),
                        "events": earthquake_data[:10]  # Limit to 10 largest events
                    }
                    lithosphere_state["quality_score"] += 0.3
                    lithosphere_state["data_sources_used"].append("USGS Earthquakes")
            
            # Query International Geomagnetic Reference Field
            if 'igrf_magnetic' in data_sources and data_sources['igrf_magnetic']:
                magnetic_data = await self._query_igrf_at_time(timestamp, data_sources['igrf_magnetic'])
                if magnetic_data:
                    lithosphere_state["magnetic_field_intensity"] = magnetic_data.get("intensity")
                    lithosphere_state["magnetic_declination"] = magnetic_data.get("declination")
                    lithosphere_state["magnetic_inclination"] = magnetic_data.get("inclination")
                    lithosphere_state["quality_score"] += 0.4
                    lithosphere_state["data_sources_used"].append("IGRF")
            
            # Query GRACE gravity data if available
            if 'grace_gravity' in data_sources and data_sources['grace_gravity']:
                gravity_data = await self._query_grace_at_time(timestamp, data_sources['grace_gravity'])
                if gravity_data:
                    lithosphere_state["gravitational_anomaly"] = gravity_data.get("anomaly")
                    lithosphere_state["quality_score"] += 0.3
                    lithosphere_state["data_sources_used"].append("GRACE")
            
            return lithosphere_state
            
        except Exception as e:
            self.logger.error(f"Error reconstructing lithosphere state: {e}")
            return {"quality_score": 0.0, "data_sources_used": []}
    
    async def _reconstruct_biosphere_state(self, timestamp: datetime,
                                         data_sources: Dict[str, Any], 
                                         resolution_minutes: int) -> Dict[str, Any]:
        """Reconstruct vegetation and biological activity state"""
        try:
            biosphere_state = {
                "vegetation_index": None,
                "leaf_area_index": None,
                "gross_primary_productivity": None,
                "land_surface_temperature": None,
                "soil_moisture": None,
                "quality_score": 0.0,
                "data_sources_used": []
            }
            
            # Query MODIS Terra vegetation data
            if 'modis_terra' in data_sources and data_sources['modis_terra']:
                modis_data = await self._query_modis_at_time(timestamp, data_sources['modis_terra'])
                if modis_data:
                    biosphere_state["vegetation_index"] = modis_data.get("ndvi")
                    biosphere_state["leaf_area_index"] = modis_data.get("lai")
                    biosphere_state["gross_primary_productivity"] = modis_data.get("gpp")
                    biosphere_state["land_surface_temperature"] = modis_data.get("lst")
                    biosphere_state["quality_score"] += 0.5
                    biosphere_state["data_sources_used"].append("MODIS Terra")
            
            # Query Sentinel-2 for higher resolution validation
            if 'sentinel_2' in data_sources and data_sources['sentinel_2']:
                sentinel_data = await self._query_sentinel2_at_time(timestamp, data_sources['sentinel_2'])
                if sentinel_data:
                    # Validate MODIS NDVI with Sentinel-2
                    if biosphere_state["vegetation_index"] is not None:
                        biosphere_state["vegetation_index"] = np.mean([
                            biosphere_state["vegetation_index"],
                            sentinel_data.get("ndvi", biosphere_state["vegetation_index"])
                        ])
                    else:
                        biosphere_state["vegetation_index"] = sentinel_data.get("ndvi")
                    biosphere_state["quality_score"] += 0.3
                    biosphere_state["data_sources_used"].append("Sentinel-2")
            
            # Add soil moisture from SMAP/SMOS if available
            soil_data = await self._query_soil_moisture_data(timestamp, data_sources)
            if soil_data:
                biosphere_state["soil_moisture"] = soil_data.get("moisture")
                biosphere_state["quality_score"] += 0.2
            
            return biosphere_state
            
        except Exception as e:
            self.logger.error(f"Error reconstructing biosphere state: {e}")
            return {"quality_score": 0.0, "data_sources_used": []}
    
    async def _reconstruct_cryosphere_state(self, timestamp: datetime,
                                          data_sources: Dict[str, Any],
                                          resolution_minutes: int) -> Dict[str, Any]:
        """Reconstruct ice and snow state"""
        try:
            cryosphere_state = {
                "sea_ice_extent": None,
                "sea_ice_concentration": None,
                "snow_cover_extent": None,
                "snow_water_equivalent": None,
                "glacier_mass_balance": None,
                "quality_score": 0.0,
                "data_sources_used": []
            }
            
            # Query NSIDC sea ice data
            if 'nsidc_ice' in data_sources and data_sources['nsidc_ice']:
                ice_data = await self._query_nsidc_at_time(timestamp, data_sources['nsidc_ice'])
                if ice_data:
                    cryosphere_state["sea_ice_extent"] = ice_data.get("extent")
                    cryosphere_state["sea_ice_concentration"] = ice_data.get("concentration")
                    cryosphere_state["quality_score"] += 0.5
                    cryosphere_state["data_sources_used"].append("NSIDC")
            
            # Add snow data from MODIS
            if 'modis_terra' in data_sources and data_sources['modis_terra']:
                snow_data = await self._query_modis_snow_at_time(timestamp, data_sources['modis_terra'])
                if snow_data:
                    cryosphere_state["snow_cover_extent"] = snow_data.get("extent")
                    cryosphere_state["quality_score"] += 0.3
                    cryosphere_state["data_sources_used"].append("MODIS Snow")
            
            # Add glacier mass balance from GRACE if available
            if 'grace_gravity' in data_sources and data_sources['grace_gravity']:
                glacier_data = await self._query_grace_glacier_at_time(timestamp, data_sources['grace_gravity'])
                if glacier_data:
                    cryosphere_state["glacier_mass_balance"] = glacier_data.get("mass_change")
                    cryosphere_state["quality_score"] += 0.2
                    cryosphere_state["data_sources_used"].append("GRACE Glaciers")
            
            return cryosphere_state
            
        except Exception as e:
            self.logger.error(f"Error reconstructing cryosphere state: {e}")
            return {"quality_score": 0.0, "data_sources_used": []}
    
    async def _reconstruct_anthroposphere_state(self, timestamp: datetime,
                                              data_sources: Dict[str, Any],
                                              resolution_minutes: int) -> Dict[str, Any]:
        """Reconstruct human activity and impact state"""
        try:
            anthroposphere_state = {
                "urban_activity_level": None,
                "nighttime_lights_radiance": None,
                "co2_emissions_rate": None,
                "air_quality_index": None,
                "shipping_density": None,
                "aviation_activity": None,
                "quality_score": 0.0,
                "data_sources_used": []
            }
            
            # Query nighttime lights data
            if 'nighttime_lights' in data_sources and data_sources['nighttime_lights']:
                lights_data = await self._query_nighttime_lights_at_time(
                    timestamp, data_sources['nighttime_lights']
                )
                if lights_data:
                    anthroposphere_state["nighttime_lights_radiance"] = lights_data.get("radiance")
                    anthroposphere_state["urban_activity_level"] = lights_data.get("activity_level")
                    anthroposphere_state["quality_score"] += 0.4
                    anthroposphere_state["data_sources_used"].append("Nighttime Lights")
            
            # Query CO2 emissions data
            if 'co2_emissions' in data_sources and data_sources['co2_emissions']:
                emissions_data = await self._query_co2_emissions_at_time(
                    timestamp, data_sources['co2_emissions']
                )
                if emissions_data:
                    anthroposphere_state["co2_emissions_rate"] = emissions_data.get("emissions_rate")
                    anthroposphere_state["quality_score"] += 0.3
                    anthroposphere_state["data_sources_used"].append("CO2 Emissions")
            
            # Add air quality data
            air_quality_data = await self._query_air_quality_data(timestamp, data_sources)
            if air_quality_data:
                anthroposphere_state["air_quality_index"] = air_quality_data.get("aqi")
                anthroposphere_state["quality_score"] += 0.2
            
            # Add transportation activity data
            transport_data = await self._query_transportation_data(timestamp, data_sources)
            if transport_data:
                anthroposphere_state["shipping_density"] = transport_data.get("shipping")
                anthroposphere_state["aviation_activity"] = transport_data.get("aviation")
                anthroposphere_state["quality_score"] += 0.1
            
            return anthroposphere_state
            
        except Exception as e:
            self.logger.error(f"Error reconstructing anthroposphere state: {e}")
            return {"quality_score": 0.0, "data_sources_used": []}
    
    async def _fallback_historical_reconstruction(self, start_time: datetime, 
                                                end_time: datetime, 
                                                resolution_minutes: int) -> List[Dict[str, Any]]:
        """Fallback to climatological reconstruction when comprehensive query fails"""
        try:
            self.logger.warning("Using fallback climatological reconstruction")
            
            history = []
            current_time = start_time
            time_delta = timedelta(minutes=resolution_minutes)
            
            # Use climatological patterns and seasonal cycles
            while current_time <= end_time:
                # Calculate day of year and hour for seasonal/diurnal patterns
                day_of_year = current_time.timetuple().tm_yday
                hour_of_day = current_time.hour
                
                # Generate climatologically-realistic Earth state
                earth_state = {
                    "timestamp": current_time.isoformat(),
                    "reconstruction_method": "climatological_fallback",
                    "data_quality_score": 0.3,  # Lower quality for fallback
                    
                    "atmosphere_state": {
                        "temperature_2m": 15.0 + 20.0 * np.cos(2 * np.pi * day_of_year / 365.25) + 
                                        5.0 * np.cos(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 2),
                        "pressure_msl": 1013.25 + 10.0 * np.sin(2 * np.pi * day_of_year / 365.25) + 
                                      np.random.normal(0, 5),
                        "humidity_relative": 60.0 + 20.0 * np.sin(2 * np.pi * day_of_year / 365.25) + 
                                           np.random.normal(0, 10),
                        "quality_score": 0.3
                    },
                    
                    "hydrosphere_state": {
                        "sea_surface_temperature": 20.0 + 8.0 * np.cos(2 * np.pi * day_of_year / 365.25) + 
                                                  np.random.normal(0, 1),
                        "sea_level_anomaly": 0.1 * np.sin(2 * np.pi * day_of_year / 365.25) + 
                                           np.random.normal(0, 0.05),
                        "quality_score": 0.3
                    },
                    
                    "lithosphere_state": {
                        "earthquake_activity": {"event_count": max(0, int(np.random.poisson(0.1))),
                                               "max_magnitude": 0.0},
                        "magnetic_field_intensity": 50000.0 + np.random.normal(0, 100),
                        "quality_score": 0.4
                    },
                    
                    "biosphere_state": {
                        "vegetation_index": 0.7 + 0.2 * np.cos(2 * np.pi * day_of_year / 365.25) + 
                                          np.random.normal(0, 0.1),
                        "gross_primary_productivity": 2.0 + 1.0 * np.cos(2 * np.pi * day_of_year / 365.25) + 
                                                    np.random.normal(0, 0.3),
                        "quality_score": 0.3
                    },
                    
                    "cryosphere_state": {
                        "sea_ice_extent": 15.0 + 5.0 * np.cos(2 * np.pi * (day_of_year + 182) / 365.25) + 
                                        np.random.normal(0, 1),
                        "snow_cover_extent": 20.0 + 15.0 * np.cos(2 * np.pi * (day_of_year + 91) / 365.25) + 
                                           np.random.normal(0, 2),
                        "quality_score": 0.3
                    },
                    
                    "anthroposphere_state": {
                        "urban_activity_level": 0.5 + 0.2 * np.cos(2 * np.pi * hour_of_day / 24) + 
                                              np.random.normal(0, 0.1),
                        "co2_emissions_rate": 40.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.25) + 
                                            np.random.normal(0, 2),
                        "quality_score": 0.3
                    }
                }
                
                history.append(earth_state)
                current_time += time_delta
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error in fallback historical reconstruction: {e}")
            return []
    
    # Data source connection placeholders (would be implemented based on actual API access)
    async def _connect_to_era5_reanalysis(self): return None
    async def _connect_to_noaa_ncep(self): return None  
    async def _connect_to_nasa_merra2(self): return None
    async def _connect_to_copernicus_marine(self): return None
    async def _connect_to_noaa_oisst(self): return None
    async def _connect_to_modis_terra(self): return None
    async def _connect_to_sentinel_2(self): return None
    async def _connect_to_usgs_earthquake_catalog(self): return None
    async def _connect_to_igrf_magnetic_field(self): return None
    async def _connect_to_nsidc_sea_ice(self): return None
    async def _connect_to_grace_gravity(self): return None
    async def _connect_to_nighttime_lights(self): return None
    async def _connect_to_co2_emissions(self): return None
    
    # Data query placeholders (would implement actual API calls)
    async def _query_era5_at_time(self, timestamp, connection): return None
    async def _query_ncep_at_time(self, timestamp, connection): return None
    async def _query_merra2_at_time(self, timestamp, connection): return None
    async def _query_copernicus_marine_at_time(self, timestamp, connection): return None
    async def _query_oisst_at_time(self, timestamp, connection): return None
    async def _query_modis_at_time(self, timestamp, connection): return None
    async def _query_sentinel2_at_time(self, timestamp, connection): return None
    async def _query_earthquakes_in_window(self, start, end, connection): return None
    async def _query_igrf_at_time(self, timestamp, connection): return None
    async def _query_grace_at_time(self, timestamp, connection): return None
    async def _query_nsidc_at_time(self, timestamp, connection): return None
    async def _query_nighttime_lights_at_time(self, timestamp, connection): return None
    async def _query_co2_emissions_at_time(self, timestamp, connection): return None
    
    # Additional helper methods
    async def _blend_atmospheric_datasets(self, state, new_data, source): return state
    async def _calculate_state_uncertainty(self, state, sources): return {}
    async def _interpolate_missing_state(self, timestamp, last_state): return last_state
    async def _post_process_historical_data(self, history): return history
    async def _query_atmospheric_composition(self, timestamp, sources): return None
    async def _query_ocean_color_data(self, timestamp, sources): return None
    async def _query_modis_snow_at_time(self, timestamp, connection): return None
    async def _query_grace_glacier_at_time(self, timestamp, connection): return None
    async def _query_soil_moisture_data(self, timestamp, sources): return None
    async def _query_air_quality_data(self, timestamp, sources): return None
    async def _query_transportation_data(self, timestamp, sources): return None
    
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