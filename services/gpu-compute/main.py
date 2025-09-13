"""
GPU Compute Service - Main Entry Point

This is the main FastAPI service for GPU acceleration and distributed computing.
It provides REST API endpoints for submitting compute jobs, managing GPU resources,
and monitoring performance with geological domain awareness.

Features:
- GPU-accelerated geological simulations
- Multi-cloud GPU resource management
- Distributed computing job scheduling
- Performance monitoring and optimization
- Cost optimization recommendations
- Real-time job status tracking
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

from core.gpu_accelerator import (
    DistributedComputeManager, GPUComputeConfig, WorkloadType,
    ComputeBackend, CloudProvider, create_distributed_compute_manager
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global service instances
compute_manager: Optional[DistributedComputeManager] = None


# Pydantic models for API
class SubmitJobRequest(BaseModel):
    workload_type: str
    input_data: Dict[str, Any]
    parameters: Dict[str, Any]
    geological_context: Dict[str, Any]
    priority: int = Field(default=5, ge=1, le=10)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    workload_type: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None


class GPUResourceRequest(BaseModel):
    provider: Optional[str] = None
    gpu_type: Optional[str] = None
    memory_gb: Optional[int] = None
    preemptible: bool = False


class SeismicProcessingRequest(BaseModel):
    velocity_model_shape: List[int]
    source_location: List[int]
    time_steps: int = 1000
    dt: float = 0.001
    memory_mb: int = 2048
    geological_context: Dict[str, Any]


class VolcanicSimulationRequest(BaseModel):
    grid_size: List[int]
    simulation_time: float = 3600.0
    dt: float = 1.0
    initial_temperature: float = 1200.0
    memory_mb: int = 1024
    geological_context: Dict[str, Any]


class GeologicalModelingRequest(BaseModel):
    model_dimensions: List[int]
    material_properties: Dict[str, Any]
    boundary_conditions: Dict[str, Any]
    memory_mb: int = 4096
    geological_context: Dict[str, Any]


# Service configuration
class ServiceConfig(BaseModel):
    """Service configuration"""
    gpu_compute: GPUComputeConfig = Field(default_factory=GPUComputeConfig)
    
    # Service settings
    host: str = Field(default="0.0.0.0", description="Service host")
    port: int = Field(default=8082, description="Service port")
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
    
    if os.getenv("DASK_SCHEDULER"):
        config.gpu_compute.dask_scheduler_address = os.getenv("DASK_SCHEDULER")
    
    if os.getenv("SPARK_MASTER"):
        config.gpu_compute.spark_master = os.getenv("SPARK_MASTER")
    
    return config


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global compute_manager
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize compute manager
        logger.info("Initializing GPU compute manager...")
        compute_manager = await create_distributed_compute_manager(config.gpu_compute)
        
        # Start job processing loop
        asyncio.create_task(job_processing_loop())
        
        logger.info("GPU Compute Service initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    finally:
        logger.info("Shutting down GPU Compute Service")


# Create FastAPI application
config = load_config()
app = FastAPI(
    title="GPU Compute Service",
    description="GPU acceleration and distributed computing service for Prometheus AI Platform",
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
async def get_compute_manager() -> DistributedComputeManager:
    """Get compute manager instance"""
    if compute_manager is None:
        raise HTTPException(status_code=503, detail="Compute manager not initialized")
    return compute_manager


# Background job processing
async def job_processing_loop():
    """Background loop for processing compute jobs"""
    while True:
        try:
            if compute_manager and compute_manager.job_queue:
                job = compute_manager.job_queue.popleft()
                # Execute job in background
                asyncio.create_task(compute_manager.execute_job(job))
            
            await asyncio.sleep(1)  # Check every second
            
        except Exception as e:
            logger.error(f"Job processing loop error: {e}")
            await asyncio.sleep(5)


# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        if compute_manager is None:
            raise HTTPException(status_code=503, detail="Compute manager not ready")
        
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {e}")


# Job management endpoints
@app.post("/api/v1/jobs/submit")
async def submit_job(
    request: SubmitJobRequest,
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """Submit compute job"""
    try:
        workload_type = WorkloadType(request.workload_type)
        
        job_id = await manager.submit_job(
            workload_type=workload_type,
            input_data=request.input_data,
            parameters=request.parameters,
            geological_context=request.geological_context,
            priority=request.priority
        )
        
        return {
            "job_id": job_id,
            "status": "submitted",
            "workload_type": request.workload_type,
            "priority": request.priority,
            "estimated_queue_time": len(manager.job_queue) * 30  # Rough estimate in seconds
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid workload type: {request.workload_type}")
    except Exception as e:
        logger.error(f"Job submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(
    job_id: str,
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """Get job status"""
    try:
        status = manager.get_job_status(job_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return status
        
    except Exception as e:
        logger.error(f"Job status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/jobs")
async def list_jobs(
    status: Optional[str] = None,
    workload_type: Optional[str] = None,
    limit: int = 100,
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """List jobs with optional filtering"""
    try:
        jobs = []
        
        # Get active jobs
        for job in manager.active_jobs.values():
            if status and job.status != status:
                continue
            if workload_type and job.workload_type.value != workload_type:
                continue
            
            jobs.append({
                "job_id": job.job_id,
                "status": job.status,
                "workload_type": job.workload_type.value,
                "priority": job.priority,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "geological_context": job.geological_context
            })
        
        # Get completed jobs
        for job in manager.completed_jobs[-limit:]:
            if status and job.status != status:
                continue
            if workload_type and job.workload_type.value != workload_type:
                continue
            
            jobs.append({
                "job_id": job.job_id,
                "status": job.status,
                "workload_type": job.workload_type.value,
                "priority": job.priority,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "geological_context": job.geological_context
            })
        
        # Sort by creation time (most recent first)
        jobs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        
        return {
            "jobs": jobs[:limit],
            "total_count": len(jobs),
            "active_jobs": len(manager.active_jobs),
            "queued_jobs": len(manager.job_queue)
        }
        
    except Exception as e:
        logger.error(f"Job listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Specialized job submission endpoints
@app.post("/api/v1/jobs/seismic-processing")
async def submit_seismic_processing_job(
    request: SeismicProcessingRequest,
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """Submit seismic processing job"""
    try:
        import numpy as np
        
        # Generate mock velocity model
        velocity_model = np.random.rand(*request.velocity_model_shape).astype(np.float32) * 5000 + 2000
        
        job_id = await manager.submit_job(
            workload_type=WorkloadType.SEISMIC_PROCESSING,
            input_data={'velocity_model': velocity_model},
            parameters={
                'source_location': request.source_location,
                'time_steps': request.time_steps,
                'dt': request.dt,
                'memory_mb': request.memory_mb
            },
            geological_context=request.geological_context,
            priority=8  # High priority for seismic processing
        )
        
        return {
            "job_id": job_id,
            "workload_type": "seismic_processing",
            "velocity_model_shape": request.velocity_model_shape,
            "source_location": request.source_location,
            "time_steps": request.time_steps
        }
        
    except Exception as e:
        logger.error(f"Seismic processing job submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/jobs/volcanic-simulation")
async def submit_volcanic_simulation_job(
    request: VolcanicSimulationRequest,
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """Submit volcanic simulation job"""
    try:
        import numpy as np
        
        # Generate initial conditions
        nx, ny = request.grid_size
        initial_conditions = {
            'temperature': np.full((nx, ny), request.initial_temperature, dtype=np.float32),
            'velocity_x': np.zeros((nx, ny), dtype=np.float32),
            'velocity_y': np.zeros((nx, ny), dtype=np.float32),
            'density': np.full((nx, ny), 1000.0, dtype=np.float32),
            'pressure': np.full((nx, ny), 101325.0, dtype=np.float32)
        }
        
        job_id = await manager.submit_job(
            workload_type=WorkloadType.VOLCANIC_SIMULATION,
            input_data={'initial_conditions': initial_conditions},
            parameters={
                'simulation_time': request.simulation_time,
                'dt': request.dt,
                'memory_mb': request.memory_mb
            },
            geological_context=request.geological_context,
            priority=7  # High priority for volcanic simulation
        )
        
        return {
            "job_id": job_id,
            "workload_type": "volcanic_simulation",
            "grid_size": request.grid_size,
            "simulation_time": request.simulation_time,
            "initial_temperature": request.initial_temperature
        }
        
    except Exception as e:
        logger.error(f"Volcanic simulation job submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/jobs/geological-modeling")
async def submit_geological_modeling_job(
    request: GeologicalModelingRequest,
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """Submit geological modeling job"""
    try:
        import numpy as np
        
        # Generate mock displacement field and material properties
        nx, ny = request.model_dimensions[:2]
        
        displacement_field = {
            'x': np.random.rand(nx, ny).astype(np.float32) * 0.01,
            'y': np.random.rand(nx, ny).astype(np.float32) * 0.01
        }
        
        material_properties = {
            'elastic_modulus': np.full((nx, ny), 30e9, dtype=np.float32),  # 30 GPa
            'poisson_ratio': np.full((nx, ny), 0.25, dtype=np.float32)
        }
        
        job_id = await manager.submit_job(
            workload_type=WorkloadType.GEOLOGICAL_MODELING,
            input_data={
                'displacement_field': displacement_field,
                'material_properties': material_properties
            },
            parameters={
                'model_dimensions': request.model_dimensions,
                'boundary_conditions': request.boundary_conditions,
                'memory_mb': request.memory_mb
            },
            geological_context=request.geological_context,
            priority=6  # Medium priority for geological modeling
        )
        
        return {
            "job_id": job_id,
            "workload_type": "geological_modeling",
            "model_dimensions": request.model_dimensions,
            "boundary_conditions": request.boundary_conditions
        }
        
    except Exception as e:
        logger.error(f"Geological modeling job submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# GPU resource management endpoints
@app.get("/api/v1/gpu/resources")
async def list_gpu_resources(
    provider: Optional[str] = None,
    available_only: bool = True,
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """List available GPU resources"""
    try:
        resources = []
        
        for resource in manager.gpu_manager.gpu_resources.values():
            if available_only and not resource.available:
                continue
            
            if provider and resource.provider.value != provider:
                continue
            
            resources.append({
                "resource_id": resource.resource_id,
                "provider": resource.provider.value,
                "gpu_type": resource.gpu_type,
                "memory_gb": resource.memory_gb,
                "cores": resource.cores,
                "cost_per_hour": resource.cost_per_hour,
                "location": resource.location,
                "available": resource.available,
                "current_utilization": resource.current_utilization,
                "preemptible": resource.preemptible
            })
        
        return {
            "resources": resources,
            "total_count": len(resources),
            "providers": list(set(r["provider"] for r in resources))
        }
        
    except Exception as e:
        logger.error(f"GPU resource listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/gpu/discover")
async def discover_gpu_resources(
    background_tasks: BackgroundTasks,
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """Discover GPU resources across cloud providers"""
    try:
        # Run discovery in background
        background_tasks.add_task(manager.gpu_manager.discover_gpu_resources)
        
        return {
            "status": "discovering",
            "message": "GPU resource discovery initiated in background"
        }
        
    except Exception as e:
        logger.error(f"GPU resource discovery error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/gpu/provision")
async def provision_gpu_resources(
    request: GPUResourceRequest,
    background_tasks: BackgroundTasks,
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """Provision GPU resources"""
    try:
        # Filter resources based on request
        suitable_resources = []
        
        for resource in manager.gpu_manager.gpu_resources.values():
            if request.provider and resource.provider.value != request.provider:
                continue
            if request.gpu_type and request.gpu_type not in resource.gpu_type:
                continue
            if request.memory_gb and resource.memory_gb < request.memory_gb:
                continue
            if request.preemptible != resource.preemptible:
                continue
            
            suitable_resources.append(resource)
        
        if not suitable_resources:
            raise HTTPException(status_code=404, detail="No suitable resources found")
        
        # Provision resources in background
        background_tasks.add_task(
            manager.gpu_manager.provision_resources,
            suitable_resources[:5]  # Provision up to 5 resources
        )
        
        return {
            "status": "provisioning",
            "message": f"Provisioning {len(suitable_resources[:5])} GPU resources",
            "resources": [r.resource_id for r in suitable_resources[:5]]
        }
        
    except Exception as e:
        logger.error(f"GPU resource provisioning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance monitoring endpoints
@app.get("/api/v1/performance/summary")
async def get_performance_summary(
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """Get performance summary"""
    try:
        summary = manager.get_performance_summary()
        return summary
        
    except Exception as e:
        logger.error(f"Performance summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/performance/metrics")
async def get_performance_metrics(
    limit: int = 100,
    workload_type: Optional[str] = None,
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """Get detailed performance metrics"""
    try:
        metrics = manager.performance_metrics[-limit:]
        
        if workload_type:
            # Filter by workload type (would need to add workload_type to PerformanceMetrics)
            pass
        
        return {
            "metrics": [
                {
                    "job_id": m.job_id,
                    "backend": m.backend.value,
                    "execution_time": m.execution_time,
                    "memory_usage": m.memory_usage,
                    "gpu_utilization": m.gpu_utilization,
                    "throughput": m.throughput,
                    "cost": m.cost,
                    "energy_consumption": m.energy_consumption,
                    "geological_accuracy": m.geological_accuracy,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in metrics
            ],
            "total_count": len(metrics)
        }
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cost optimization endpoints
@app.get("/api/v1/cost/optimization")
async def get_cost_optimization(
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """Get cost optimization recommendations"""
    try:
        optimization = manager.gpu_manager.calculate_cost_optimization()
        return optimization
        
    except Exception as e:
        logger.error(f"Cost optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cost/tracking")
async def get_cost_tracking(
    time_range_hours: int = 24,
    manager: DistributedComputeManager = Depends(get_compute_manager)
):
    """Get cost tracking information"""
    try:
        # Calculate costs for the specified time range
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        
        recent_metrics = [
            m for m in manager.performance_metrics
            if m.timestamp > cutoff_time
        ]
        
        total_cost = sum(m.cost for m in recent_metrics)
        cost_by_backend = {}
        
        for metric in recent_metrics:
            backend = metric.backend.value
            if backend not in cost_by_backend:
                cost_by_backend[backend] = 0
            cost_by_backend[backend] += metric.cost
        
        return {
            "time_range_hours": time_range_hours,
            "total_cost": total_cost,
            "cost_by_backend": cost_by_backend,
            "cost_per_hour": total_cost / time_range_hours if time_range_hours > 0 else 0,
            "job_count": len(recent_metrics)
        }
        
    except Exception as e:
        logger.error(f"Cost tracking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System information endpoints
@app.get("/api/v1/system/info")
async def get_system_info():
    """Get system information"""
    try:
        info = {
            "cuda_available": False,
            "torch_available": False,
            "dask_available": False,
            "spark_available": False,
            "gpu_count": 0,
            "system_memory": "unknown"
        }
        
        # Check CUDA availability
        try:
            import cupy as cp
            info["cuda_available"] = True
            info["gpu_count"] = cp.cuda.runtime.getDeviceCount()
        except:
            pass
        
        # Check PyTorch availability
        try:
            import torch
            info["torch_available"] = True
            if torch.cuda.is_available():
                info["torch_cuda_available"] = True
                info["torch_gpu_count"] = torch.cuda.device_count()
        except:
            pass
        
        # Check Dask availability
        try:
            import dask
            info["dask_available"] = True
        except:
            pass
        
        # Check Spark availability
        try:
            from pyspark.sql import SparkSession
            info["spark_available"] = True
        except:
            pass
        
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