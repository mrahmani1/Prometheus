"""
Edge Computing Service - Main Entry Point

This is the main FastAPI service for edge computing and CDN integration.
It provides REST API endpoints for managing edge nodes, processing sensor data,
and optimizing content delivery for geological datasets.

Features:
- Edge node deployment and management
- Real-time sensor data processing at edge
- CDN caching and optimization for geological data
- Edge AI inference and model deployment
- Performance monitoring and analytics
- IoT sensor network integration
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.edge_manager import (
    EdgeManager, CDNManager, EdgeComputeConfig, EdgeNodeType,
    EdgeLocation, create_edge_manager, create_cdn_manager
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global service instances
edge_manager: Optional[EdgeManager] = None
cdn_manager: Optional[CDNManager] = None


# Pydantic models for API
class DeployEdgeNodeRequest(BaseModel):
    node_type: str
    location: str
    coordinates: List[float]
    compute_capacity: Dict[str, Any] = Field(default_factory=dict)
    storage_gb: int = 100
    bandwidth_mbps: int = 100
    power_watts: float = 50.0
    geological_context: Dict[str, Any] = Field(default_factory=dict)


class ProcessSensorDataRequest(BaseModel):
    node_id: str
    sensor_data: Dict[str, Any]


class CacheDatasetRequest(BaseModel):
    dataset_id: str
    size_mb: int
    content_type: str = "geological_data"
    geological_domain: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Service configuration
class ServiceConfig(BaseModel):
    """Service configuration"""
    edge_compute: EdgeComputeConfig = Field(default_factory=EdgeComputeConfig)
    
    # Service settings
    host: str = Field(default="0.0.0.0", description="Service host")
    port: int = Field(default=8083, description="Service port")
    debug: bool = Field(default=False, description="Debug mode")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")


# Load configuration
def load_config() -> ServiceConfig:
    """Load service configuration from environment or defaults"""
    config = ServiceConfig()
    
    # Override with environment variables
    if os.getenv("DEBUG"):
        config.debug = os.getenv("DEBUG").lower() == "true"
    
    if os.getenv("PORT"):
        config.port = int(os.getenv("PORT"))
    
    return config


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global edge_manager, cdn_manager
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize edge manager
        logger.info("Initializing edge manager...")
        edge_manager = await create_edge_manager(config.edge_compute)
        
        # Initialize CDN manager
        logger.info("Initializing CDN manager...")
        cdn_manager = await create_cdn_manager(config.edge_compute)
        
        logger.info("Edge Computing Service initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    finally:
        logger.info("Shutting down Edge Computing Service")


# Create FastAPI application
config = load_config()
app = FastAPI(
    title="Edge Computing Service",
    description="Edge computing and CDN integration service for Prometheus AI Platform",
    version="1.0.0",
    lifespan=lifespan,
    debug=config.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
async def get_edge_manager() -> EdgeManager:
    """Get edge manager instance"""
    if edge_manager is None:
        raise HTTPException(status_code=503, detail="Edge manager not initialized")
    return edge_manager


async def get_cdn_manager() -> CDNManager:
    """Get CDN manager instance"""
    if cdn_manager is None:
        raise HTTPException(status_code=503, detail="CDN manager not initialized")
    return cdn_manager


# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        if not all([edge_manager, cdn_manager]):
            raise HTTPException(status_code=503, detail="Services not ready")
        
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {e}")


# Edge node management endpoints
@app.post("/api/v1/edge/nodes/deploy")
async def deploy_edge_node(
    request: DeployEdgeNodeRequest,
    manager: EdgeManager = Depends(get_edge_manager)
):
    """Deploy new edge computing node"""
    try:
        node_config = {
            'node_type': request.node_type,
            'location': request.location,
            'coordinates': request.coordinates,
            'compute_capacity': request.compute_capacity,
            'storage_gb': request.storage_gb,
            'bandwidth_mbps': request.bandwidth_mbps,
            'power_watts': request.power_watts,
            'geological_context': request.geological_context
        }
        
        node_id = await manager.deploy_edge_node(node_config)
        
        return {
            "node_id": node_id,
            "status": "deployed",
            "node_type": request.node_type,
            "location": request.location,
            "coordinates": request.coordinates
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Edge node deployment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/edge/nodes")
async def list_edge_nodes(
    node_type: Optional[str] = None,
    location: Optional[str] = None,
    status: Optional[str] = None,
    manager: EdgeManager = Depends(get_edge_manager)
):
    """List edge computing nodes"""
    try:
        nodes = []
        
        for node in manager.edge_nodes.values():
            # Apply filters
            if node_type and node.node_type.value != node_type:
                continue
            if location and node.location.value != location:
                continue
            if status and node.status != status:
                continue
            
            nodes.append({
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "location": node.location.value,
                "coordinates": node.geographical_coordinates,
                "status": node.status,
                "compute_capacity": node.compute_capacity,
                "storage_capacity": node.storage_capacity,
                "network_bandwidth": node.network_bandwidth,
                "power_consumption": node.power_consumption,
                "last_heartbeat": node.last_heartbeat.isoformat(),
                "deployed_models": node.deployed_models,
                "cached_datasets": node.cached_datasets,
                "geological_context": node.geological_context
            })
        
        return {
            "nodes": nodes,
            "total_count": len(nodes),
            "node_types": list(set(n["node_type"] for n in nodes)),
            "locations": list(set(n["location"] for n in nodes))
        }
        
    except Exception as e:
        logger.error(f"Edge node listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/edge/nodes/{node_id}")
async def get_edge_node(
    node_id: str,
    manager: EdgeManager = Depends(get_edge_manager)
):
    """Get edge node details"""
    try:
        if node_id not in manager.edge_nodes:
            raise HTTPException(status_code=404, detail="Edge node not found")
        
        node = manager.edge_nodes[node_id]
        
        return {
            "node_id": node.node_id,
            "node_type": node.node_type.value,
            "location": node.location.value,
            "coordinates": node.geographical_coordinates,
            "status": node.status,
            "compute_capacity": node.compute_capacity,
            "storage_capacity": node.storage_capacity,
            "network_bandwidth": node.network_bandwidth,
            "power_consumption": node.power_consumption,
            "last_heartbeat": node.last_heartbeat.isoformat(),
            "deployed_models": node.deployed_models,
            "cached_datasets": node.cached_datasets,
            "geological_context": node.geological_context
        }
        
    except Exception as e:
        logger.error(f"Edge node retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Sensor data processing endpoints
@app.post("/api/v1/edge/process")
async def process_sensor_data(
    request: ProcessSensorDataRequest,
    manager: EdgeManager = Depends(get_edge_manager)
):
    """Process sensor data at edge node"""
    try:
        result = await manager.process_sensor_data(
            request.node_id,
            request.sensor_data
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Sensor data processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# CDN management endpoints
@app.post("/api/v1/cdn/cache")
async def cache_dataset(
    request: CacheDatasetRequest,
    manager: CDNManager = Depends(get_cdn_manager)
):
    """Cache geological dataset in CDN"""
    try:
        dataset_info = {
            'size_mb': request.size_mb,
            'content_type': request.content_type,
            'geological_domain': request.geological_domain,
            'metadata': request.metadata
        }
        
        success = await manager.cache_geological_dataset(
            request.dataset_id,
            dataset_info
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to cache dataset")
        
        return {
            "dataset_id": request.dataset_id,
            "status": "cached",
            "size_mb": request.size_mb,
            "geological_domain": request.geological_domain,
            "cache_locations": manager.content_registry[request.dataset_id]['cache_locations']
        }
        
    except Exception as e:
        logger.error(f"Dataset caching error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cdn/datasets")
async def list_cached_datasets(
    geological_domain: Optional[str] = None,
    manager: CDNManager = Depends(get_cdn_manager)
):
    """List cached datasets"""
    try:
        datasets = []
        
        for dataset_id, info in manager.content_registry.items():
            # Apply filter
            if geological_domain and info.get('geological_domain') != geological_domain:
                continue
            
            datasets.append({
                "dataset_id": dataset_id,
                "size_mb": info.get('size_mb', 0),
                "content_type": info.get('content_type', 'unknown'),
                "geological_domain": info.get('geological_domain', 'general'),
                "access_frequency": info.get('access_frequency', 0),
                "cached_at": info.get('cached_at').isoformat() if info.get('cached_at') else None,
                "cache_locations": info.get('cache_locations', [])
            })
        
        return {
            "datasets": datasets,
            "total_count": len(datasets),
            "geological_domains": list(set(d["geological_domain"] for d in datasets))
        }
        
    except Exception as e:
        logger.error(f"Dataset listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cdn/cache-stats")
async def get_cache_statistics(
    manager: CDNManager = Depends(get_cdn_manager)
):
    """Get CDN cache statistics"""
    try:
        stats = {
            "total_datasets": len(manager.content_registry),
            "total_cache_nodes": len(manager.cache_nodes),
            "cache_nodes": {},
            "dataset_distribution": {},
            "storage_utilization": {}
        }
        
        # Cache node statistics
        for location, node_info in manager.cache_nodes.items():
            stats["cache_nodes"][location] = {
                "cached_datasets": len(node_info['cached_datasets']),
                "total_size_mb": node_info['total_size_mb'],
                "capacity_mb": node_info['capacity_mb'],
                "utilization_percent": (node_info['total_size_mb'] / node_info['capacity_mb']) * 100,
                "hit_rate": node_info['hit_rate']
            }
        
        # Dataset distribution by geological domain
        domain_counts = {}
        for info in manager.content_registry.values():
            domain = info.get('geological_domain', 'general')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        stats["dataset_distribution"] = domain_counts
        
        # Storage utilization
        total_used = sum(node['total_size_mb'] for node in manager.cache_nodes.values())
        total_capacity = sum(node['capacity_mb'] for node in manager.cache_nodes.values())
        
        stats["storage_utilization"] = {
            "total_used_mb": total_used,
            "total_capacity_mb": total_capacity,
            "utilization_percent": (total_used / total_capacity) * 100 if total_capacity > 0 else 0
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Cache statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance monitoring endpoints
@app.get("/api/v1/performance/edge-nodes")
async def get_edge_performance(
    manager: EdgeManager = Depends(get_edge_manager)
):
    """Get edge node performance metrics"""
    try:
        performance_data = {
            "total_nodes": len(manager.edge_nodes),
            "active_nodes": sum(1 for node in manager.edge_nodes.values() if node.status == "active"),
            "node_performance": {},
            "average_metrics": {}
        }
        
        # Individual node performance
        total_power = 0
        total_storage = 0
        total_bandwidth = 0
        
        for node in manager.edge_nodes.values():
            performance_data["node_performance"][node.node_id] = {
                "status": node.status,
                "power_consumption": node.power_consumption,
                "storage_utilization": len(node.cached_datasets),
                "deployed_models": len(node.deployed_models),
                "last_heartbeat": node.last_heartbeat.isoformat(),
                "uptime_hours": (datetime.now() - node.last_heartbeat).total_seconds() / 3600
            }
            
            total_power += node.power_consumption
            total_storage += node.storage_capacity
            total_bandwidth += node.network_bandwidth
        
        # Average metrics
        node_count = len(manager.edge_nodes)
        if node_count > 0:
            performance_data["average_metrics"] = {
                "avg_power_consumption": total_power / node_count,
                "total_storage_capacity": total_storage,
                "total_bandwidth": total_bandwidth,
                "avg_storage_per_node": total_storage / node_count,
                "avg_bandwidth_per_node": total_bandwidth / node_count
            }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Edge performance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System information endpoints
@app.get("/api/v1/system/info")
async def get_system_info():
    """Get system information"""
    try:
        info = {
            "service_name": "Edge Computing Service",
            "version": "1.0.0",
            "supported_node_types": [node_type.value for node_type in EdgeNodeType],
            "supported_locations": [location.value for location in EdgeLocation],
            "features": [
                "Edge node deployment",
                "Real-time sensor processing",
                "CDN caching",
                "Edge AI inference",
                "Performance monitoring"
            ]
        }
        
        return info
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )


# Main entry point
if __name__ == "__main__":
    config = load_config()
    
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info" if not config.debug else "debug"
    )