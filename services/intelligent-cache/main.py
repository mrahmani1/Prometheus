"""
Intelligent Cache Service - Main Entry Point

This is the main FastAPI service for the Intelligent Cache and Data Optimization system.
It provides REST API endpoints for cache management, query optimization, prefetching,
and performance monitoring with geological domain awareness.

Features:
- RESTful API for cache operations
- Real-time performance monitoring endpoints
- Query optimization and result caching
- Intelligent prefetching management
- Geological workflow optimization
- Comprehensive metrics and analytics
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.cache_manager import (
    IntelligentCacheManager, CacheConfig, DataType, create_cache_manager
)
from core.query_optimizer import (
    QueryOptimizer, QueryOptimizerConfig, create_query_optimizer
)
from core.prefetch_engine import (
    IntelligentPrefetchEngine, PrefetchConfig, create_prefetch_engine
)
from core.performance_monitor import (
    PerformanceMonitor, PerformanceMonitorConfig, MetricType, 
    AlertSeverity, create_performance_monitor
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global service instances
cache_manager: Optional[IntelligentCacheManager] = None
query_optimizer: Optional[QueryOptimizer] = None
prefetch_engine: Optional[IntelligentPrefetchEngine] = None
performance_monitor: Optional[PerformanceMonitor] = None


# Pydantic models for API
class CacheGetRequest(BaseModel):
    key: str
    data_type: Optional[str] = None


class CacheSetRequest(BaseModel):
    key: str
    data: Any
    data_type: str
    geological_context: Optional[Dict[str, Any]] = None
    ttl_hours: Optional[int] = None


class QueryOptimizeRequest(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = None
    geological_context: Optional[Dict[str, Any]] = None


class PrefetchRequest(BaseModel):
    user_id: str
    session_id: str
    data_key: str
    geological_context: Optional[Dict[str, Any]] = None


class WorkflowWarmRequest(BaseModel):
    workflow_id: str
    geological_context: Dict[str, Any]


class MetricRecordRequest(BaseModel):
    metric_type: str
    value: float
    geological_context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceSummaryRequest(BaseModel):
    time_range_hours: Optional[int] = 1
    include_geological_analysis: bool = True


# Service configuration
class ServiceConfig(BaseModel):
    """Service configuration"""
    cache: CacheConfig = Field(default_factory=CacheConfig)
    query_optimizer: QueryOptimizerConfig = Field(default_factory=QueryOptimizerConfig)
    prefetch: PrefetchConfig = Field(default_factory=PrefetchConfig)
    performance_monitor: PerformanceMonitorConfig = Field(default_factory=PerformanceMonitorConfig)
    
    # Service settings
    host: str = Field(default="0.0.0.0", description="Service host")
    port: int = Field(default=8080, description="Service port")
    debug: bool = Field(default=False, description="Debug mode")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")


# Load configuration
def load_config() -> ServiceConfig:
    """Load service configuration from environment or defaults"""
    config = ServiceConfig()
    
    # Override with environment variables
    if os.getenv("REDIS_URL"):
        config.cache.redis = {"url": os.getenv("REDIS_URL")}
    
    if os.getenv("DEBUG"):
        config.debug = os.getenv("DEBUG").lower() == "true"
    
    if os.getenv("PORT"):
        config.port = int(os.getenv("PORT"))
    
    return config


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global cache_manager, query_optimizer, prefetch_engine, performance_monitor
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize cache manager
        logger.info("Initializing cache manager...")
        cache_manager = await create_cache_manager(config.cache)
        
        # Initialize query optimizer
        logger.info("Initializing query optimizer...")
        query_optimizer = await create_query_optimizer(cache_manager, config.query_optimizer)
        
        # Initialize prefetch engine
        logger.info("Initializing prefetch engine...")
        prefetch_engine = await create_prefetch_engine(cache_manager, config.prefetch)
        
        # Initialize performance monitor
        logger.info("Initializing performance monitor...")
        performance_monitor = await create_performance_monitor(cache_manager, config.performance_monitor)
        
        logger.info("Intelligent Cache Service initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    finally:
        logger.info("Shutting down Intelligent Cache Service")


# Create FastAPI application
config = load_config()
app = FastAPI(
    title="Intelligent Cache Service",
    description="Advanced caching and data optimization service for Prometheus AI Platform",
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
async def get_cache_manager() -> IntelligentCacheManager:
    """Get cache manager instance"""
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache manager not initialized")
    return cache_manager


async def get_query_optimizer() -> QueryOptimizer:
    """Get query optimizer instance"""
    if query_optimizer is None:
        raise HTTPException(status_code=503, detail="Query optimizer not initialized")
    return query_optimizer


async def get_prefetch_engine() -> IntelligentPrefetchEngine:
    """Get prefetch engine instance"""
    if prefetch_engine is None:
        raise HTTPException(status_code=503, detail="Prefetch engine not initialized")
    return prefetch_engine


async def get_performance_monitor() -> PerformanceMonitor:
    """Get performance monitor instance"""
    if performance_monitor is None:
        raise HTTPException(status_code=503, detail="Performance monitor not initialized")
    return performance_monitor


# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Check if all services are initialized
        if not all([cache_manager, query_optimizer, prefetch_engine, performance_monitor]):
            raise HTTPException(status_code=503, detail="Services not ready")
        
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {e}")


# Cache management endpoints
@app.get("/api/v1/cache/{key}")
async def get_cached_data(
    key: str,
    data_type: Optional[str] = None,
    cache_mgr: IntelligentCacheManager = Depends(get_cache_manager),
    perf_monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get data from cache"""
    start_time = datetime.now()
    
    try:
        # Convert data type if provided
        dt = DataType(data_type) if data_type else None
        
        # Get data from cache
        data = await cache_mgr.get(key, dt)
        
        # Record performance metrics
        duration = (datetime.now() - start_time).total_seconds()
        await perf_monitor.record_cache_operation(
            "get", duration, data is not None, 
            cache_mgr.l1_cache.get(key, {}).get('cache_level', 'unknown'),
            dt or DataType.GEOLOGICAL_MODEL,
            {}
        )
        
        if data is None:
            raise HTTPException(status_code=404, detail="Data not found in cache")
        
        return {"key": key, "data": data, "cached": True}
        
    except Exception as e:
        logger.error(f"Cache get error for key {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cache")
async def set_cached_data(
    request: CacheSetRequest,
    cache_mgr: IntelligentCacheManager = Depends(get_cache_manager),
    perf_monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Set data in cache"""
    start_time = datetime.now()
    
    try:
        # Convert data type
        data_type = DataType(request.data_type)
        
        # Calculate TTL
        ttl = timedelta(hours=request.ttl_hours) if request.ttl_hours else None
        
        # Set data in cache
        success = await cache_mgr.set(
            key=request.key,
            data=request.data,
            data_type=data_type,
            geological_context=request.geological_context,
            ttl=ttl
        )
        
        # Record performance metrics
        duration = (datetime.now() - start_time).total_seconds()
        await perf_monitor.record_cache_operation(
            "set", duration, success, 
            cache_mgr._determine_cache_level(request.key) if hasattr(cache_mgr, '_determine_cache_level') else 'l1_memory',
            data_type,
            request.geological_context or {}
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to cache data")
        
        return {"key": request.key, "cached": True, "ttl_hours": request.ttl_hours}
        
    except Exception as e:
        logger.error(f"Cache set error for key {request.key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/cache/{key}")
async def invalidate_cached_data(
    key: str,
    cascade: bool = True,
    cache_mgr: IntelligentCacheManager = Depends(get_cache_manager)
):
    """Invalidate cached data"""
    try:
        success = await cache_mgr.invalidate(key, cascade)
        
        if not success:
            raise HTTPException(status_code=404, detail="Data not found in cache")
        
        return {"key": key, "invalidated": True, "cascade": cascade}
        
    except Exception as e:
        logger.error(f"Cache invalidation error for key {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Query optimization endpoints
@app.post("/api/v1/query/optimize")
async def optimize_query(
    request: QueryOptimizeRequest,
    optimizer: QueryOptimizer = Depends(get_query_optimizer)
):
    """Optimize and execute query"""
    try:
        result = await optimizer.optimize_and_execute(
            query=request.query,
            parameters=request.parameters,
            database_session=None  # Would pass actual session in production
        )
        
        return {
            "query_hash": result.query_hash,
            "execution_time": result.execution_time,
            "result_size": result.result_size,
            "cached": result.cache_key in optimizer.cache_manager.l1_cache,
            "geological_context": result.geological_context
        }
        
    except Exception as e:
        logger.error(f"Query optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/query/recommendations")
async def get_query_recommendations(
    optimizer: QueryOptimizer = Depends(get_query_optimizer)
):
    """Get query optimization recommendations"""
    try:
        recommendations = optimizer.get_optimization_recommendations()
        return {"recommendations": recommendations}
        
    except Exception as e:
        logger.error(f"Query recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Prefetching endpoints
@app.post("/api/v1/prefetch/record-access")
async def record_data_access(
    request: PrefetchRequest,
    prefetch: IntelligentPrefetchEngine = Depends(get_prefetch_engine)
):
    """Record data access for predictive prefetching"""
    try:
        await prefetch.record_access(
            user_id=request.user_id,
            session_id=request.session_id,
            data_key=request.data_key,
            geological_context=request.geological_context
        )
        
        return {"recorded": True, "user_id": request.user_id, "data_key": request.data_key}
        
    except Exception as e:
        logger.error(f"Prefetch record access error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/prefetch/warm-workflow")
async def warm_workflow_cache(
    request: WorkflowWarmRequest,
    prefetch: IntelligentPrefetchEngine = Depends(get_prefetch_engine)
):
    """Warm cache for geological workflow"""
    try:
        success = await prefetch.warm_cache_for_workflow(
            workflow_id=request.workflow_id,
            geological_context=request.geological_context
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Invalid workflow ID")
        
        return {"workflow_id": request.workflow_id, "warming_initiated": True}
        
    except Exception as e:
        logger.error(f"Workflow cache warming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/prefetch/metrics")
async def get_prefetch_metrics(
    prefetch: IntelligentPrefetchEngine = Depends(get_prefetch_engine)
):
    """Get prefetch performance metrics"""
    try:
        metrics = prefetch.get_prefetch_metrics()
        return {"metrics": metrics}
        
    except Exception as e:
        logger.error(f"Prefetch metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance monitoring endpoints
@app.post("/api/v1/performance/record-metric")
async def record_performance_metric(
    request: MetricRecordRequest,
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Record performance metric"""
    try:
        metric_type = MetricType(request.metric_type)
        
        await monitor.record_metric(
            metric_type=metric_type,
            value=request.value,
            geological_context=request.geological_context,
            metadata=request.metadata
        )
        
        return {"recorded": True, "metric_type": request.metric_type, "value": request.value}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid metric type: {request.metric_type}")
    except Exception as e:
        logger.error(f"Performance metric recording error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/performance/summary")
async def get_performance_summary(
    request: PerformanceSummaryRequest,
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get performance summary"""
    try:
        time_range = timedelta(hours=request.time_range_hours)
        summary = monitor.get_performance_summary(time_range)
        
        return {"summary": summary}
        
    except Exception as e:
        logger.error(f"Performance summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/performance/alerts")
async def get_performance_alerts(
    severity: Optional[str] = None,
    hours: Optional[int] = 24,
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get performance alerts"""
    try:
        alert_severity = AlertSeverity(severity) if severity else None
        time_range = timedelta(hours=hours)
        
        alerts = monitor.get_recent_alerts(alert_severity, time_range)
        
        return {
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "metric_type": alert.metric_type.value,
                    "message": alert.message,
                    "threshold_value": alert.threshold_value,
                    "actual_value": alert.actual_value,
                    "timestamp": alert.timestamp.isoformat(),
                    "recommendations": alert.recommendations
                }
                for alert in alerts
            ]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
    except Exception as e:
        logger.error(f"Performance alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/performance/recommendations")
async def get_optimization_recommendations(
    priority_threshold: int = 5,
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """Get optimization recommendations"""
    try:
        recommendations = monitor.get_optimization_recommendations(priority_threshold)
        
        return {
            "recommendations": [
                {
                    "recommendation_id": rec.recommendation_id,
                    "category": rec.category,
                    "priority": rec.priority,
                    "description": rec.description,
                    "expected_improvement": rec.expected_improvement,
                    "implementation_effort": rec.implementation_effort,
                    "geological_impact": rec.geological_impact,
                    "cost_impact": rec.cost_impact,
                    "metrics_affected": [m.value for m in rec.metrics_affected],
                    "implementation_steps": rec.implementation_steps
                }
                for rec in recommendations
            ]
        }
        
    except Exception as e:
        logger.error(f"Optimization recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache optimization endpoints
@app.post("/api/v1/cache/optimize")
async def optimize_cache(
    background_tasks: BackgroundTasks,
    cache_mgr: IntelligentCacheManager = Depends(get_cache_manager)
):
    """Trigger cache optimization"""
    try:
        # Run optimization in background
        background_tasks.add_task(cache_mgr.optimize_cache)
        
        return {"optimization_initiated": True}
        
    except Exception as e:
        logger.error(f"Cache optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cache/metrics")
async def get_cache_metrics(
    cache_mgr: IntelligentCacheManager = Depends(get_cache_manager)
):
    """Get cache performance metrics"""
    try:
        metrics = cache_mgr.get_metrics()
        
        return {
            "metrics": {
                "hit_rate": metrics.hit_rate,
                "miss_rate": metrics.miss_rate,
                "eviction_rate": metrics.eviction_rate,
                "compression_ratio": metrics.compression_ratio,
                "memory_usage": metrics.memory_usage,
                "network_savings": metrics.network_savings,
                "query_acceleration": metrics.query_acceleration,
                "geological_accuracy_preservation": metrics.geological_accuracy_preservation
            }
        }
        
    except Exception as e:
        logger.error(f"Cache metrics error: {e}")
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