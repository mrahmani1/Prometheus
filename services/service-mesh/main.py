"""
Main Service Mesh Manager for Prometheus AI Platform
Orchestrates all service mesh components and provides unified API
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis.asyncio as redis
import consul.aio
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Import service mesh components
from circuit_breaker import CircuitBreakerFactory, CircuitBreakerConfig, ChaosType
from rate_limiter import RateLimiterFactory, RateLimitConfig, RateLimitAlgorithm
from chaos_engineering import ChaosMonkey, ChaosExperiment, ChaosScope
from traffic_management import IntelligentTrafficManager, TrafficSplitRule, TrafficSplitType, ServiceEndpoint
from service_discovery import AdvancedServiceDiscovery, ServiceInstance, HealthCheck, HealthCheckType
from observability import ServiceMeshObservability, ObservabilityConfig, SecurityPolicy, SecurityPolicyType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ServiceRegistrationRequest(BaseModel):
    name: str
    host: str
    port: int
    version: str = "v1"
    tags: List[str] = []
    geological_capabilities: List[str] = []
    processing_capacity: float = 1.0

class TrafficSplitRequest(BaseModel):
    name: str
    service_name: str
    split_type: str
    splits: List[Dict[str, Any]]
    canary_percentage: Optional[float] = None

class ChaosExperimentRequest(BaseModel):
    name: str
    description: str
    chaos_type: str
    scope: str
    target_services: List[str] = []
    duration: int = 300
    intensity: float = 0.1

class SecurityPolicyRequest(BaseModel):
    name: str
    policy_type: str
    services: List[str] = []
    rules: Dict[str, Any] = {}
    enforcement_mode: str = "enforce"

class ServiceMeshManager:
    """
    Main service mesh manager that orchestrates all components
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Prometheus AI Service Mesh Manager",
            description="Advanced service mesh management for geological systems",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self.redis_client = None
        self.consul_client = None
        self.circuit_breaker_factory = None
        self.rate_limiter_factory = None
        self.chaos_monkey = None
        self.traffic_manager = None
        self.service_discovery = None
        self.observability = None
        
        # Setup routes
        self._setup_routes()
    
    async def initialize(self):
        """Initialize all service mesh components"""
        
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis.from_url("redis://redis:6379")
            await self.redis_client.ping()
            logger.info("Connected to Redis")
            
            # Initialize Consul connection
            self.consul_client = consul.aio.Consul(host="consul-server", port=8500)
            logger.info("Connected to Consul")
            
            # Initialize service mesh components
            self.circuit_breaker_factory = CircuitBreakerFactory(self.redis_client)
            self.rate_limiter_factory = RateLimiterFactory(self.redis_client)
            self.chaos_monkey = ChaosMonkey(self.redis_client)
            self.traffic_manager = IntelligentTrafficManager(self.redis_client)
            self.service_discovery = AdvancedServiceDiscovery(self.consul_client, self.redis_client)
            
            # Initialize observability
            observability_config = ObservabilityConfig(
                tracing_enabled=True,
                geological_metrics_enabled=True,
                volcanic_monitoring_traces=True,
                seismic_processing_metrics=True
            )
            self.observability = ServiceMeshObservability(observability_config)
            
            logger.info("Service mesh manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize service mesh manager: {e}")
            raise
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            await self.initialize()
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            if self.redis_client:
                await self.redis_client.close()
        
        # Health endpoints
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        @self.app.get("/ready")
        async def readiness_check():
            try:
                # Check Redis connection
                await self.redis_client.ping()
                
                # Check Consul connection
                await self.consul_client.agent.self()
                
                return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def metrics():
            return generate_latest()
        
        # Service Discovery endpoints
        @self.app.post("/services/register")
        async def register_service(request: ServiceRegistrationRequest):
            try:
                service = ServiceInstance(
                    id=f"{request.name}-{request.host}-{request.port}",
                    name=request.name,
                    host=request.host,
                    port=request.port,
                    version=request.version,
                    tags=request.tags,
                    geological_capabilities=request.geological_capabilities,
                    processing_capacity=request.processing_capacity,
                    health_checks=[
                        HealthCheck(
                            id=f"{request.name}-http-check",
                            name="HTTP Health Check",
                            check_type=HealthCheckType.HTTP,
                            url=f"http://{request.host}:{request.port}/health"
                        )
                    ]
                )
                
                success = await self.service_discovery.register_service(service)
                
                if success:
                    return {"status": "registered", "service_id": service.id}
                else:
                    raise HTTPException(status_code=500, detail="Failed to register service")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/services/{service_id}")
        async def deregister_service(service_id: str):
            try:
                success = await self.service_discovery.deregister_service(service_id)
                
                if success:
                    return {"status": "deregistered", "service_id": service_id}
                else:
                    raise HTTPException(status_code=404, detail="Service not found")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/services")
        async def discover_services(service_name: Optional[str] = None, healthy_only: bool = True):
            try:
                services = await self.service_discovery.discover_services(
                    service_name=service_name,
                    healthy_only=healthy_only
                )
                
                return {
                    "services": [
                        {
                            "id": service.id,
                            "name": service.name,
                            "host": service.host,
                            "port": service.port,
                            "status": service.status.value,
                            "geological_capabilities": service.geological_capabilities
                        }
                        for service in services
                    ]
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/services/{service_id}/health")
        async def get_service_health(service_id: str):
            try:
                health = await self.service_discovery.get_service_health(service_id)
                
                if health:
                    return health
                else:
                    raise HTTPException(status_code=404, detail="Service not found")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Traffic Management endpoints
        @self.app.post("/traffic/splits")
        async def create_traffic_split(request: TrafficSplitRequest):
            try:
                rule = TrafficSplitRule(
                    id=f"split-{request.name}",
                    name=request.name,
                    description=f"Traffic split for {request.service_name}",
                    split_type=TrafficSplitType(request.split_type),
                    service_name=request.service_name,
                    splits=request.splits,
                    canary_percentage=request.canary_percentage or 0.1
                )
                
                await self.traffic_manager.create_traffic_split_rule(rule)
                
                return {"status": "created", "rule_id": rule.id}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/traffic/splits/{rule_id}")
        async def get_traffic_split_status(rule_id: str):
            try:
                status = await self.traffic_manager.get_traffic_split_status(rule_id)
                
                if status:
                    return status
                else:
                    raise HTTPException(status_code=404, detail="Traffic split rule not found")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Circuit Breaker endpoints
        @self.app.post("/circuit-breakers/{service_name}")
        async def create_circuit_breaker(service_name: str, config: Dict[str, Any] = {}):
            try:
                cb_config = CircuitBreakerConfig(
                    name=service_name,
                    failure_threshold=config.get('failure_threshold', 5),
                    recovery_timeout=config.get('recovery_timeout', 60),
                    max_concurrent_calls=config.get('max_concurrent_calls', 100)
                )
                
                circuit_breaker = self.circuit_breaker_factory.get_circuit_breaker(service_name, cb_config)
                
                return {"status": "created", "service": service_name}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/circuit-breakers/{service_name}/status")
        async def get_circuit_breaker_status(service_name: str):
            try:
                circuit_breaker = self.circuit_breaker_factory.get_circuit_breaker(service_name)
                status = await circuit_breaker.get_health_status()
                
                return status
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Chaos Engineering endpoints
        @self.app.post("/chaos/experiments")
        async def create_chaos_experiment(request: ChaosExperimentRequest, background_tasks: BackgroundTasks):
            try:
                experiment = ChaosExperiment(
                    id=f"chaos-{request.name}",
                    name=request.name,
                    description=request.description,
                    chaos_type=ChaosType(request.chaos_type),
                    scope=ChaosScope(request.scope),
                    target_services=request.target_services,
                    duration=request.duration,
                    intensity=request.intensity
                )
                
                # Run experiment in background
                background_tasks.add_task(self.chaos_monkey.run_experiment, experiment)
                
                return {"status": "started", "experiment_id": experiment.id}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/chaos/experiments/{experiment_id}")
        async def get_chaos_experiment_status(experiment_id: str):
            try:
                status = await self.chaos_monkey.get_experiment_status(experiment_id)
                
                if status:
                    return status
                else:
                    raise HTTPException(status_code=404, detail="Experiment not found")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/chaos/experiments")
        async def list_chaos_experiments():
            try:
                experiments = await self.chaos_monkey.list_experiments()
                return experiments
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Security Policy endpoints
        @self.app.post("/security/policies")
        async def create_security_policy(request: SecurityPolicyRequest):
            try:
                policy = SecurityPolicy(
                    id=f"policy-{request.name}",
                    name=request.name,
                    policy_type=SecurityPolicyType(request.policy_type),
                    services=request.services,
                    rules=request.rules,
                    enforcement_mode=request.enforcement_mode
                )
                
                await self.observability.create_security_policy(policy)
                
                return {"status": "created", "policy_id": policy.id}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/security/policies")
        async def list_security_policies():
            try:
                policies = [
                    {
                        "id": policy.id,
                        "name": policy.name,
                        "type": policy.policy_type.value,
                        "services": policy.services,
                        "enforcement_mode": policy.enforcement_mode
                    }
                    for policy in self.observability.security_policies.values()
                ]
                
                return {"policies": policies}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Observability endpoints
        @self.app.get("/observability/metrics")
        async def get_observability_metrics():
            try:
                metrics = await self.observability.get_observability_metrics()
                return metrics
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/observability/traces")
        async def export_traces(
            start_time: Optional[str] = None,
            end_time: Optional[str] = None
        ):
            try:
                start_dt = datetime.fromisoformat(start_time) if start_time else datetime.utcnow()
                end_dt = datetime.fromisoformat(end_time) if end_time else datetime.utcnow()
                
                traces = await self.observability.export_traces(start_dt, end_dt)
                return {"traces": traces}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Service Mesh Status endpoint
        @self.app.get("/status")
        async def get_service_mesh_status():
            try:
                # Get status from all components
                circuit_breaker_status = await self.circuit_breaker_factory.get_all_health_status()
                rate_limiter_status = await self.rate_limiter_factory.get_all_status()
                dependency_map = await self.service_discovery.map_service_dependencies()
                observability_metrics = await self.observability.get_observability_metrics()
                
                return {
                    "service_mesh": {
                        "status": "healthy",
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "circuit_breakers": circuit_breaker_status,
                    "rate_limiters": rate_limiter_status,
                    "service_dependencies": dependency_map,
                    "observability": observability_metrics
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

# Create global service mesh manager instance
service_mesh_manager = ServiceMeshManager()

# FastAPI app instance for uvicorn
app = service_mesh_manager.app

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )