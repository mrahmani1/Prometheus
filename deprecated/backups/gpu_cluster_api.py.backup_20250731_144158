"""
GPU Cluster API endpoints for managing distributed geological modeling.

This module provides REST API endpoints for GPU cluster management,
distributed training coordination, and performance monitoring.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime

from ..core.gpu_cluster_manager import (
    GPUClusterManager, ClusterConfig, ClusterStatus, GPUType,
    TrainingJob, ClusterMetrics, create_gpu_cluster_manager
)

logger = logging.getLogger(__name__)

# Global GPU cluster manager instance
cluster_manager: Optional[GPUClusterManager] = None

router = APIRouter(prefix="/gpu-cluster", tags=["GPU Cluster Management"])


class ClusterConfigRequest(BaseModel):
    """Request model for GPU cluster configuration."""
    cluster_name: str = Field(default="geological-gpu-cluster", description="Cluster name")
    max_nodes: int = Field(default=8, ge=1, le=64, description="Maximum number of nodes")
    gpus_per_node: int = Field(default=8, ge=1, le=8, description="GPUs per node")
    memory_threshold_gb: float = Field(default=2.0, ge=0.5, description="Minimum free memory per GPU")
    utilization_threshold: float = Field(default=90.0, ge=50.0, le=100.0, description="Maximum utilization percentage")
    temperature_threshold: float = Field(default=85.0, ge=60.0, le=95.0, description="Maximum temperature in Celsius")
    auto_scaling: bool = Field(default=True, description="Enable automatic scaling")
    mixed_precision: bool = Field(default=True, description="Use automatic mixed precision")
    batch_size_per_gpu: int = Field(default=32, ge=1, description="Batch size per GPU")
    distributed_backend: str = Field(default="nccl", description="Distributed backend")


class TrainingJobRequest(BaseModel):
    """Request model for training job submission."""
    model_name: str = Field(description="Name of the model to train")
    dataset_path: str = Field(description="Path to training dataset")
    num_epochs: int = Field(ge=1, le=1000, description="Number of training epochs")
    learning_rate: float = Field(gt=0, le=1.0, description="Learning rate")
    batch_size: int = Field(ge=1, description="Batch size")
    num_gpus: int = Field(ge=1, le=64, description="Number of GPUs required")
    priority: int = Field(default=1, ge=1, le=5, description="Job priority (1=highest, 5=lowest)")
    model_config: Optional[Dict[str, Any]] = Field(None, description="Model configuration parameters")


class BatchSizeOptimizationRequest(BaseModel):
    """Request model for batch size optimization."""
    job_id: str = Field(description="Training job ID")
    model_size_mb: Optional[float] = Field(None, description="Model size in MB")
    sample_size_mb: Optional[float] = Field(None, description="Sample size in MB")


class ClusterStatusResponse(BaseModel):
    """Response model for cluster status."""
    status: str
    cluster_name: str
    total_gpus: int
    active_gpus: int
    gpu_devices: List[Dict[str, Any]]
    cluster_metrics: Dict[str, float]
    training_jobs: Dict[str, int]


class TrainingJobResponse(BaseModel):
    """Response model for training job information."""
    job_id: str
    model_name: str
    status: str
    progress: float
    current_loss: float
    num_gpus: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    estimated_completion: Optional[datetime]


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    timestamp: datetime
    total_gpus: int
    active_gpus: int
    total_memory_gb: float
    used_memory_gb: float
    free_memory_gb: float
    average_utilization: float
    average_temperature: float
    total_power_usage_watts: float
    throughput_samples_per_sec: float
    active_jobs: int
    queued_jobs: int


def get_cluster_manager() -> GPUClusterManager:
    """Dependency to get GPU cluster manager instance."""
    global cluster_manager
    if cluster_manager is None:
        raise HTTPException(status_code=503, detail="GPU cluster not initialized")
    return cluster_manager


@router.post("/initialize", response_model=Dict[str, str])
async def initialize_gpu_cluster(config: ClusterConfigRequest):
    """
    Initialize GPU cluster with specified configuration.
    
    This endpoint initializes a GPU cluster for distributed geological modeling.
    The cluster will discover available GPUs and prepare them for training.
    """
    global cluster_manager
    
    try:
        # Convert request to ClusterConfig
        cluster_config = ClusterConfig(
            cluster_name=config.cluster_name,
            max_nodes=config.max_nodes,
            gpus_per_node=config.gpus_per_node,
            memory_threshold_gb=config.memory_threshold_gb,
            utilization_threshold=config.utilization_threshold,
            temperature_threshold=config.temperature_threshold,
            auto_scaling=config.auto_scaling,
            mixed_precision=config.mixed_precision,
            batch_size_per_gpu=config.batch_size_per_gpu,
            distributed_backend=config.distributed_backend
        )
        
        # Initialize GPU cluster manager
        cluster_manager = create_gpu_cluster_manager(cluster_config)
        
        logger.info(f"GPU cluster '{config.cluster_name}' initialized")
        
        return {
            "status": "success",
            "message": f"GPU cluster '{config.cluster_name}' initialized successfully",
            "total_gpus": len(cluster_manager.gpu_devices)
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize GPU cluster: {e}")
        raise HTTPException(status_code=500, detail=f"Cluster initialization failed: {str(e)}")


@router.get("/status", response_model=ClusterStatusResponse)
async def get_cluster_status(manager: GPUClusterManager = Depends(get_cluster_manager)):
    """
    Get current GPU cluster status and device information.
    
    Returns comprehensive information about cluster health, GPU utilization,
    memory usage, and active training jobs.
    """
    try:
        status_info = manager.get_cluster_status()
        
        return ClusterStatusResponse(
            status=status_info["status"],
            cluster_name=status_info["cluster_name"],
            total_gpus=status_info["total_gpus"],
            active_gpus=status_info["active_gpus"],
            gpu_devices=status_info["gpu_devices"],
            cluster_metrics=status_info["cluster_metrics"],
            training_jobs=status_info["training_jobs"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get cluster status: {e}")
        raise HTTPException(status_code=500, detail=f"Status query failed: {str(e)}")


@router.post("/jobs/submit", response_model=Dict[str, str])
async def submit_training_job(
    job_request: TrainingJobRequest,
    manager: GPUClusterManager = Depends(get_cluster_manager)
):
    """
    Submit a training job to the GPU cluster.
    
    This endpoint submits a geological model training job to the cluster queue.
    The job will be scheduled based on priority and resource availability.
    """
    try:
        # Convert request to job configuration
        job_config = {
            "model_name": job_request.model_name,
            "dataset_path": job_request.dataset_path,
            "num_epochs": job_request.num_epochs,
            "learning_rate": job_request.learning_rate,
            "batch_size": job_request.batch_size,
            "num_gpus": job_request.num_gpus,
            "priority": job_request.priority
        }
        
        if job_request.model_config:
            job_config.update(job_request.model_config)
        
        # Submit job
        job_id = manager.submit_training_job(job_config)
        
        logger.info(f"Training job {job_id} submitted successfully")
        
        return {
            "status": "success",
            "job_id": job_id,
            "message": f"Training job submitted with ID: {job_id}",
            "queue_position": len(manager.job_queue)
        }
        
    except Exception as e:
        logger.error(f"Failed to submit training job: {e}")
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")


@router.post("/jobs/{job_id}/start", response_model=Dict[str, str])
async def start_training_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    manager: GPUClusterManager = Depends(get_cluster_manager)
):
    """
    Start a training job on the GPU cluster.
    
    This endpoint starts distributed training for a queued job,
    allocating the required GPUs and initializing the training process.
    """
    try:
        # Start distributed training
        success = manager.start_distributed_training(job_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to start job {job_id}")
        
        # Simulate training progress in background
        async def simulate_training():
            try:
                import asyncio
                import random
                
                # Simulate training progress
                for epoch in range(1, 101):  # 100 epochs
                    await asyncio.sleep(0.1)  # Simulate training time
                    
                    if job_id in manager.active_jobs:
                        job = manager.active_jobs[job_id]["job"]
                        job.progress = epoch / 100.0
                        job.current_loss = 1.0 * (1.0 - job.progress) + random.uniform(-0.1, 0.1)
                        job.current_loss = max(0.01, job.current_loss)  # Minimum loss
                
                # Complete the job
                final_loss = manager.active_jobs[job_id]["job"].current_loss if job_id in manager.active_jobs else 0.01
                manager.complete_training_job(job_id, final_loss)
                
                logger.info(f"Training job {job_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Training simulation failed for job {job_id}: {e}")
        
        background_tasks.add_task(simulate_training)
        
        return {
            "status": "success",
            "job_id": job_id,
            "message": f"Training job {job_id} started successfully",
            "allocated_gpus": manager.active_jobs[job_id]["allocated_gpus"]
        }
        
    except Exception as e:
        logger.error(f"Failed to start training job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Job start failed: {str(e)}")


@router.get("/jobs/{job_id}/status", response_model=Dict[str, Any])
async def get_job_status(
    job_id: str,
    manager: GPUClusterManager = Depends(get_cluster_manager)
):
    """
    Get status and progress of a training job.
    
    Returns detailed information about job progress, resource usage,
    and performance metrics.
    """
    try:
        if job_id not in manager.training_jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Get job progress
        progress_info = manager.monitor_training_progress(job_id)
        
        # Get job details
        job = manager.training_jobs[job_id]
        
        return {
            "job_id": job_id,
            "model_name": job.model_name,
            "status": job.status,
            "progress": job.progress,
            "current_loss": job.current_loss,
            "num_gpus": job.num_gpus,
            "priority": job.priority,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "estimated_completion": job.estimated_completion.isoformat() if job.estimated_completion else None,
            "progress_info": progress_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Status query failed: {str(e)}")


@router.get("/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    status_filter: Optional[str] = None,
    limit: int = 50,
    manager: GPUClusterManager = Depends(get_cluster_manager)
):
    """
    List all training jobs with optional status filtering.
    
    Returns a list of training jobs with their current status and progress.
    """
    try:
        jobs = []
        
        for job in manager.training_jobs.values():
            if status_filter and job.status != status_filter:
                continue
            
            jobs.append(TrainingJobResponse(
                job_id=job.job_id,
                model_name=job.model_name,
                status=job.status,
                progress=job.progress,
                current_loss=job.current_loss,
                num_gpus=job.num_gpus,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                estimated_completion=job.estimated_completion
            ))
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return jobs[:limit]
        
    except Exception as e:
        logger.error(f"Failed to list training jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Job listing failed: {str(e)}")


@router.post("/jobs/{job_id}/optimize-batch-size", response_model=Dict[str, Any])
async def optimize_batch_size(
    job_id: str,
    request: BatchSizeOptimizationRequest,
    manager: GPUClusterManager = Depends(get_cluster_manager)
):
    """
    Optimize batch sizes for a training job.
    
    This endpoint calculates optimal batch sizes for each GPU
    based on available memory and model requirements.
    """
    try:
        if job_id not in manager.active_jobs:
            raise HTTPException(status_code=404, detail=f"Active job {job_id} not found")
        
        # Optimize batch sizes
        optimized_batch_sizes = manager.optimize_batch_sizes(job_id)
        
        return {
            "job_id": job_id,
            "optimized_batch_sizes": optimized_batch_sizes,
            "total_batch_size": sum(optimized_batch_sizes.values()),
            "optimization_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to optimize batch size for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Batch size optimization failed: {str(e)}")


@router.get("/metrics/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(manager: GPUClusterManager = Depends(get_cluster_manager)):
    """
    Get comprehensive cluster performance metrics.
    
    Returns detailed performance metrics including GPU utilization,
    memory usage, temperature, power consumption, and throughput.
    """
    try:
        metrics = manager.get_performance_metrics()
        
        return PerformanceMetricsResponse(
            timestamp=metrics.timestamp,
            total_gpus=metrics.total_gpus,
            active_gpus=metrics.active_gpus,
            total_memory_gb=metrics.total_memory_gb,
            used_memory_gb=metrics.used_memory_gb,
            free_memory_gb=metrics.free_memory_gb,
            average_utilization=metrics.average_utilization,
            average_temperature=metrics.average_temperature,
            total_power_usage_watts=metrics.total_power_usage_watts,
            throughput_samples_per_sec=metrics.throughput_samples_per_sec,
            active_jobs=metrics.active_jobs,
            queued_jobs=metrics.queued_jobs
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics query failed: {str(e)}")


@router.get("/metrics/history", response_model=List[PerformanceMetricsResponse])
async def get_metrics_history(
    limit: int = 100,
    manager: GPUClusterManager = Depends(get_cluster_manager)
):
    """
    Get historical performance metrics.
    
    Returns a time series of cluster performance metrics
    for monitoring and analysis purposes.
    """
    try:
        metrics_history = manager.metrics_history[-limit:] if manager.metrics_history else []
        
        response_metrics = []
        for metrics in metrics_history:
            response_metrics.append(PerformanceMetricsResponse(
                timestamp=metrics.timestamp,
                total_gpus=metrics.total_gpus,
                active_gpus=metrics.active_gpus,
                total_memory_gb=metrics.total_memory_gb,
                used_memory_gb=metrics.used_memory_gb,
                free_memory_gb=metrics.free_memory_gb,
                average_utilization=metrics.average_utilization,
                average_temperature=metrics.average_temperature,
                total_power_usage_watts=metrics.total_power_usage_watts,
                throughput_samples_per_sec=metrics.throughput_samples_per_sec,
                active_jobs=metrics.active_jobs,
                queued_jobs=metrics.queued_jobs
            ))
        
        return response_metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics history: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics history query failed: {str(e)}")


@router.post("/auto-scale", response_model=Dict[str, Any])
async def trigger_auto_scaling(manager: GPUClusterManager = Depends(get_cluster_manager)):
    """
    Trigger automatic cluster scaling.
    
    This endpoint manually triggers the auto-scaling algorithm
    to evaluate current workload and adjust cluster size if needed.
    """
    try:
        scaling_performed = manager.auto_scale_cluster()
        
        current_metrics = manager.get_performance_metrics()
        
        return {
            "scaling_performed": scaling_performed,
            "current_utilization": current_metrics.average_utilization,
            "active_jobs": current_metrics.active_jobs,
            "queued_jobs": current_metrics.queued_jobs,
            "recommendation": (
                "Scale up recommended" if current_metrics.average_utilization > 80 and current_metrics.queued_jobs > 2
                else "Scale down recommended" if current_metrics.average_utilization < 20 and current_metrics.queued_jobs == 0
                else "No scaling needed"
            ),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger auto-scaling: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-scaling failed: {str(e)}")


@router.post("/shutdown", response_model=Dict[str, str])
async def shutdown_cluster(manager: GPUClusterManager = Depends(get_cluster_manager)):
    """
    Shutdown GPU cluster and cleanup resources.
    
    This endpoint gracefully shuts down the GPU cluster,
    completes active jobs, and cleans up all resources.
    """
    global cluster_manager
    
    try:
        manager.shutdown_cluster()
        cluster_manager = None
        
        return {
            "status": "success",
            "message": "GPU cluster shutdown completed"
        }
        
    except Exception as e:
        logger.error(f"Failed to shutdown GPU cluster: {e}")
        raise HTTPException(status_code=500, detail=f"Shutdown failed: {str(e)}")


# Health check endpoint
@router.get("/health", response_model=Dict[str, str])
async def cluster_health_check():
    """Health check for GPU cluster service."""
    global cluster_manager
    
    if cluster_manager is None:
        return {"status": "not_initialized", "message": "GPU cluster not initialized"}
    
    try:
        status = cluster_manager.status.value
        return {"status": status, "message": f"GPU cluster status: {status}"}
    except Exception as e:
        return {"status": "error", "message": f"Health check failed: {str(e)}"}