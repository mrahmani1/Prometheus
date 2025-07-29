"""
GPU Cluster Manager for distributed geological model training.

This module provides GPU cluster management for distributed training
of geological AI models across V100/A100/H100 GPU clusters with
automatic memory optimization and performance monitoring.
"""

import os
import logging
import asyncio
import psutil
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    import torch.cuda.amp as amp
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("PyTorch GPU dependencies not available. GPU functionality will be disabled.")

from .physics_constraints import PhysicsConstraints
from .pinn_engine import PINNEngine
from ..models.geological import GeologicalData, SimulationParameters

logger = logging.getLogger(__name__)


class GPUType(Enum):
    """GPU type enumeration."""
    V100 = "V100"
    A100 = "A100"
    H100 = "H100"
    RTX_4090 = "RTX_4090"
    RTX_3090 = "RTX_3090"
    UNKNOWN = "UNKNOWN"


class ClusterStatus(Enum):
    """GPU cluster status enumeration."""
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    INFERENCE = "inference"
    SCALING = "scaling"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class GPUDevice:
    """GPU device information."""
    device_id: int
    name: str
    gpu_type: GPUType
    memory_total_gb: float
    memory_free_gb: float
    memory_used_gb: float
    utilization_percent: float
    temperature_celsius: float
    power_usage_watts: float
    compute_capability: Tuple[int, int]
    is_available: bool = True
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ClusterConfig:
    """Configuration for GPU cluster."""
    cluster_name: str = "geological-gpu-cluster"
    max_nodes: int = 8
    gpus_per_node: int = 8
    memory_threshold_gb: float = 2.0  # Minimum free memory per GPU
    utilization_threshold: float = 90.0  # Maximum utilization percentage
    temperature_threshold: float = 85.0  # Maximum temperature in Celsius
    auto_scaling: bool = True
    mixed_precision: bool = True  # Use automatic mixed precision
    gradient_checkpointing: bool = True
    batch_size_per_gpu: int = 32
    accumulation_steps: int = 1
    max_sequence_length: int = 1024
    distributed_backend: str = "nccl"  # NCCL for GPU communication


@dataclass
class TrainingJob:
    """Training job configuration."""
    job_id: str
    model_name: str
    dataset_path: str
    num_epochs: int
    learning_rate: float
    batch_size: int
    num_gpus: int
    priority: int = 1  # 1 = highest, 5 = lowest
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "queued"
    progress: float = 0.0
    current_loss: float = float('inf')
    estimated_completion: Optional[datetime] = None


@dataclass
class ClusterMetrics:
    """GPU cluster performance metrics."""
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


class GPUMemoryOptimizer:
    """Optimizes GPU memory usage and batch sizes."""
    
    def __init__(self, safety_margin: float = 0.1):
        self.safety_margin = safety_margin  # Reserve 10% of memory as safety margin
        self.memory_history = {}
        
    def get_optimal_batch_size(self, gpu_id: int, model_size_mb: float, 
                              sample_size_mb: float, max_batch_size: int = 512) -> int:
        """Calculate optimal batch size for given GPU and model."""
        if not GPU_AVAILABLE:
            return 32  # Default fallback
            
        try:
            # Get GPU memory info
            torch.cuda.set_device(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            cached_memory = torch.cuda.memory_reserved(gpu_id)
            
            # Calculate available memory in MB
            available_memory_mb = (total_memory - allocated_memory - cached_memory) / (1024 ** 2)
            available_memory_mb *= (1 - self.safety_margin)  # Apply safety margin
            
            # Account for model memory
            available_for_batch = available_memory_mb - model_size_mb
            
            if available_for_batch <= 0:
                logger.warning(f"Insufficient memory on GPU {gpu_id}")
                return 1
            
            # Calculate batch size
            optimal_batch_size = int(available_for_batch / sample_size_mb)
            optimal_batch_size = min(optimal_batch_size, max_batch_size)
            optimal_batch_size = max(optimal_batch_size, 1)
            
            logger.info(f"GPU {gpu_id}: Optimal batch size = {optimal_batch_size}")
            return optimal_batch_size
            
        except Exception as e:
            logger.error(f"Failed to calculate optimal batch size for GPU {gpu_id}: {e}")
            return 32  # Safe default
    
    def optimize_memory_usage(self, model: torch.nn.Module, gpu_id: int):
        """Optimize memory usage for a model on specific GPU."""
        if not GPU_AVAILABLE:
            return
            
        try:
            torch.cuda.set_device(gpu_id)
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                # Use memory efficient attention
                pass
            
            # Enable gradient checkpointing if model supports it
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            logger.info(f"Memory optimization applied for GPU {gpu_id}")
            
        except Exception as e:
            logger.error(f"Failed to optimize memory for GPU {gpu_id}: {e}")
    
    def monitor_memory_usage(self, gpu_id: int) -> Dict[str, float]:
        """Monitor memory usage for a specific GPU."""
        if not GPU_AVAILABLE:
            return {"total": 0, "allocated": 0, "cached": 0, "free": 0}
            
        try:
            torch.cuda.set_device(gpu_id)
            
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            cached_memory = torch.cuda.memory_reserved(gpu_id)
            free_memory = total_memory - allocated_memory
            
            # Convert to GB
            memory_info = {
                "total": total_memory / (1024 ** 3),
                "allocated": allocated_memory / (1024 ** 3),
                "cached": cached_memory / (1024 ** 3),
                "free": free_memory / (1024 ** 3)
            }
            
            # Store in history
            self.memory_history[gpu_id] = {
                "timestamp": datetime.now(),
                "memory_info": memory_info
            }
            
            return memory_info
            
        except Exception as e:
            logger.error(f"Failed to monitor memory for GPU {gpu_id}: {e}")
            return {"total": 0, "allocated": 0, "cached": 0, "free": 0}


class DistributedTrainingManager:
    """Manages distributed training across GPU cluster."""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.world_size = 0
        self.rank = 0
        self.local_rank = 0
        self.is_initialized = False
        
    def initialize_distributed(self, rank: int, world_size: int, master_addr: str = "localhost", 
                             master_port: str = "12355"):
        """Initialize distributed training environment."""
        try:
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['RANK'] = str(rank)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config.distributed_backend,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=30)
            )
            
            self.rank = rank
            self.world_size = world_size
            self.local_rank = rank % torch.cuda.device_count()
            self.is_initialized = True
            
            # Set device for this process
            torch.cuda.set_device(self.local_rank)
            
            logger.info(f"Distributed training initialized: rank={rank}, world_size={world_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    def create_distributed_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training."""
        if not self.is_initialized:
            raise RuntimeError("Distributed training not initialized")
        
        try:
            # Move model to GPU
            device = torch.device(f"cuda:{self.local_rank}")
            model = model.to(device)
            
            # Wrap with DistributedDataParallel
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
            
            logger.info(f"Model wrapped for distributed training on GPU {self.local_rank}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create distributed model: {e}")
            raise
    
    def create_distributed_dataloader(self, dataset, batch_size: int, 
                                    shuffle: bool = True) -> DataLoader:
        """Create distributed data loader."""
        if not self.is_initialized:
            raise RuntimeError("Distributed training not initialized")
        
        try:
            # Create distributed sampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            
            # Create data loader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )
            
            return dataloader
            
        except Exception as e:
            logger.error(f"Failed to create distributed dataloader: {e}")
            raise
    
    def cleanup_distributed(self):
        """Cleanup distributed training environment."""
        if self.is_initialized:
            try:
                dist.destroy_process_group()
                self.is_initialized = False
                logger.info("Distributed training cleanup completed")
            except Exception as e:
                logger.error(f"Failed to cleanup distributed training: {e}")


class GPUClusterManager:
    """
    Manages GPU cluster resources for distributed geological modeling.
    
    Handles GPU allocation, distributed training coordination, memory optimization,
    and performance monitoring across V100/A100/H100 GPU clusters.
    """
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.status = ClusterStatus.INITIALIZING
        self.gpu_devices: Dict[int, GPUDevice] = {}
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.job_queue: List[str] = []
        self.active_jobs: Dict[str, Any] = {}
        self.memory_optimizer = GPUMemoryOptimizer()
        self.distributed_manager = DistributedTrainingManager(config)
        self.metrics_history: List[ClusterMetrics] = []
        
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU cluster functionality requires PyTorch with CUDA support")
        
        self._initialize_cluster()
    
    def _initialize_cluster(self):
        """Initialize GPU cluster and discover available devices."""
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            
            num_gpus = torch.cuda.device_count()
            logger.info(f"Discovered {num_gpus} GPU devices")
            
            # Initialize each GPU device
            for gpu_id in range(num_gpus):
                device_props = torch.cuda.get_device_properties(gpu_id)
                
                # Determine GPU type
                gpu_type = self._identify_gpu_type(device_props.name)
                
                # Get memory info
                torch.cuda.set_device(gpu_id)
                total_memory = device_props.total_memory / (1024 ** 3)  # Convert to GB
                
                # Create GPU device object
                gpu_device = GPUDevice(
                    device_id=gpu_id,
                    name=device_props.name,
                    gpu_type=gpu_type,
                    memory_total_gb=total_memory,
                    memory_free_gb=total_memory,  # Initially all free
                    memory_used_gb=0.0,
                    utilization_percent=0.0,
                    temperature_celsius=0.0,
                    power_usage_watts=0.0,
                    compute_capability=(device_props.major, device_props.minor),
                    is_available=True
                )
                
                self.gpu_devices[gpu_id] = gpu_device
                logger.info(f"Initialized GPU {gpu_id}: {device_props.name} ({total_memory:.1f} GB)")
            
            self.status = ClusterStatus.READY
            logger.info("GPU cluster initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU cluster: {e}")
            self.status = ClusterStatus.ERROR
            raise
    
    def _identify_gpu_type(self, gpu_name: str) -> GPUType:
        """Identify GPU type from device name."""
        gpu_name_lower = gpu_name.lower()
        
        if "v100" in gpu_name_lower:
            return GPUType.V100
        elif "a100" in gpu_name_lower:
            return GPUType.A100
        elif "h100" in gpu_name_lower:
            return GPUType.H100
        elif "rtx 4090" in gpu_name_lower or "4090" in gpu_name_lower:
            return GPUType.RTX_4090
        elif "rtx 3090" in gpu_name_lower or "3090" in gpu_name_lower:
            return GPUType.RTX_3090
        else:
            return GPUType.UNKNOWN
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        try:
            # Update GPU device information
            self._update_gpu_metrics()
            
            # Calculate cluster statistics
            total_gpus = len(self.gpu_devices)
            active_gpus = sum(1 for gpu in self.gpu_devices.values() if gpu.is_available)
            total_memory = sum(gpu.memory_total_gb for gpu in self.gpu_devices.values())
            used_memory = sum(gpu.memory_used_gb for gpu in self.gpu_devices.values())
            free_memory = total_memory - used_memory
            avg_utilization = np.mean([gpu.utilization_percent for gpu in self.gpu_devices.values()])
            avg_temperature = np.mean([gpu.temperature_celsius for gpu in self.gpu_devices.values()])
            total_power = sum(gpu.power_usage_watts for gpu in self.gpu_devices.values())
            
            return {
                "status": self.status.value,
                "cluster_name": self.config.cluster_name,
                "total_gpus": total_gpus,
                "active_gpus": active_gpus,
                "gpu_devices": [
                    {
                        "device_id": gpu.device_id,
                        "name": gpu.name,
                        "gpu_type": gpu.gpu_type.value,
                        "memory_total_gb": gpu.memory_total_gb,
                        "memory_used_gb": gpu.memory_used_gb,
                        "memory_free_gb": gpu.memory_free_gb,
                        "utilization_percent": gpu.utilization_percent,
                        "temperature_celsius": gpu.temperature_celsius,
                        "power_usage_watts": gpu.power_usage_watts,
                        "is_available": gpu.is_available
                    }
                    for gpu in self.gpu_devices.values()
                ],
                "cluster_metrics": {
                    "total_memory_gb": total_memory,
                    "used_memory_gb": used_memory,
                    "free_memory_gb": free_memory,
                    "memory_utilization_percent": (used_memory / total_memory) * 100 if total_memory > 0 else 0,
                    "average_gpu_utilization": avg_utilization,
                    "average_temperature": avg_temperature,
                    "total_power_usage_watts": total_power
                },
                "training_jobs": {
                    "active": len(self.active_jobs),
                    "queued": len(self.job_queue),
                    "total": len(self.training_jobs)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {"status": "error", "error": str(e)}
    
    def _update_gpu_metrics(self):
        """Update GPU device metrics."""
        try:
            for gpu_id, gpu_device in self.gpu_devices.items():
                # Update memory information
                memory_info = self.memory_optimizer.monitor_memory_usage(gpu_id)
                gpu_device.memory_used_gb = memory_info["allocated"]
                gpu_device.memory_free_gb = memory_info["free"]
                
                # Update utilization (would use nvidia-ml-py in production)
                # For now, estimate based on memory usage
                gpu_device.utilization_percent = min(
                    (gpu_device.memory_used_gb / gpu_device.memory_total_gb) * 100, 100.0
                )
                
                # Update temperature and power (would use nvidia-ml-py in production)
                gpu_device.temperature_celsius = 65.0  # Estimated
                gpu_device.power_usage_watts = 250.0 * (gpu_device.utilization_percent / 100.0)
                
                gpu_device.last_updated = datetime.now()
                
        except Exception as e:
            logger.error(f"Failed to update GPU metrics: {e}")
    
    def allocate_gpus(self, num_gpus: int, memory_required_gb: float = 4.0) -> List[int]:
        """Allocate GPUs for training job."""
        try:
            available_gpus = []
            
            for gpu_id, gpu_device in self.gpu_devices.items():
                if (gpu_device.is_available and 
                    gpu_device.memory_free_gb >= memory_required_gb and
                    gpu_device.utilization_percent < self.config.utilization_threshold and
                    gpu_device.temperature_celsius < self.config.temperature_threshold):
                    available_gpus.append(gpu_id)
            
            if len(available_gpus) < num_gpus:
                raise RuntimeError(f"Insufficient GPUs available. Requested: {num_gpus}, Available: {len(available_gpus)}")
            
            # Select best GPUs (lowest utilization first)
            available_gpus.sort(key=lambda gpu_id: self.gpu_devices[gpu_id].utilization_percent)
            allocated_gpus = available_gpus[:num_gpus]
            
            # Mark GPUs as allocated
            for gpu_id in allocated_gpus:
                self.gpu_devices[gpu_id].is_available = False
            
            logger.info(f"Allocated GPUs {allocated_gpus} for training")
            return allocated_gpus
            
        except Exception as e:
            logger.error(f"Failed to allocate GPUs: {e}")
            raise
    
    def release_gpus(self, gpu_ids: List[int]):
        """Release GPUs after training completion."""
        try:
            for gpu_id in gpu_ids:
                if gpu_id in self.gpu_devices:
                    self.gpu_devices[gpu_id].is_available = True
                    # Clear GPU memory
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
            
            logger.info(f"Released GPUs {gpu_ids}")
            
        except Exception as e:
            logger.error(f"Failed to release GPUs: {e}")
    
    def submit_training_job(self, job_config: Dict[str, Any]) -> str:
        """Submit a training job to the cluster."""
        try:
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.training_jobs)}"
            
            training_job = TrainingJob(
                job_id=job_id,
                model_name=job_config.get("model_name", "geological_pinn"),
                dataset_path=job_config.get("dataset_path", ""),
                num_epochs=job_config.get("num_epochs", 100),
                learning_rate=job_config.get("learning_rate", 1e-3),
                batch_size=job_config.get("batch_size", 32),
                num_gpus=job_config.get("num_gpus", 1),
                priority=job_config.get("priority", 1)
            )
            
            self.training_jobs[job_id] = training_job
            self.job_queue.append(job_id)
            
            # Sort queue by priority
            self.job_queue.sort(key=lambda jid: self.training_jobs[jid].priority)
            
            logger.info(f"Submitted training job {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to submit training job: {e}")
            raise
    
    def start_distributed_training(self, job_id: str) -> bool:
        """Start distributed training for a job."""
        try:
            if job_id not in self.training_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self.training_jobs[job_id]
            
            # Allocate GPUs
            allocated_gpus = self.allocate_gpus(job.num_gpus, memory_required_gb=4.0)
            
            # Update job status
            job.status = "running"
            job.started_at = datetime.now()
            
            # Store job in active jobs
            self.active_jobs[job_id] = {
                "job": job,
                "allocated_gpus": allocated_gpus,
                "start_time": datetime.now()
            }
            
            # Remove from queue
            if job_id in self.job_queue:
                self.job_queue.remove(job_id)
            
            self.status = ClusterStatus.TRAINING
            
            logger.info(f"Started distributed training for job {job_id} on GPUs {allocated_gpus}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start distributed training for job {job_id}: {e}")
            return False
    
    def optimize_batch_sizes(self, job_id: str) -> Dict[int, int]:
        """Optimize batch sizes for each GPU in a training job."""
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Active job {job_id} not found")
            
            job_info = self.active_jobs[job_id]
            allocated_gpus = job_info["allocated_gpus"]
            
            # Estimate model and sample sizes (would be calculated from actual model)
            model_size_mb = 500.0  # Estimated model size
            sample_size_mb = 10.0   # Estimated sample size
            
            optimized_batch_sizes = {}
            
            for gpu_id in allocated_gpus:
                optimal_batch_size = self.memory_optimizer.get_optimal_batch_size(
                    gpu_id, model_size_mb, sample_size_mb
                )
                optimized_batch_sizes[gpu_id] = optimal_batch_size
                
                # Apply memory optimization
                # self.memory_optimizer.optimize_memory_usage(model, gpu_id)  # Would pass actual model
            
            logger.info(f"Optimized batch sizes for job {job_id}: {optimized_batch_sizes}")
            return optimized_batch_sizes
            
        except Exception as e:
            logger.error(f"Failed to optimize batch sizes for job {job_id}: {e}")
            return {}
    
    def monitor_training_progress(self, job_id: str) -> Dict[str, Any]:
        """Monitor training progress for a job."""
        try:
            if job_id not in self.active_jobs:
                return {"status": "not_found"}
            
            job_info = self.active_jobs[job_id]
            job = job_info["job"]
            allocated_gpus = job_info["allocated_gpus"]
            
            # Get GPU metrics for allocated GPUs
            gpu_metrics = {}
            for gpu_id in allocated_gpus:
                memory_info = self.memory_optimizer.monitor_memory_usage(gpu_id)
                gpu_metrics[gpu_id] = {
                    "memory_used_gb": memory_info["allocated"],
                    "memory_free_gb": memory_info["free"],
                    "utilization_percent": self.gpu_devices[gpu_id].utilization_percent,
                    "temperature_celsius": self.gpu_devices[gpu_id].temperature_celsius
                }
            
            # Calculate training statistics
            elapsed_time = datetime.now() - job_info["start_time"]
            estimated_completion = None
            if job.progress > 0:
                total_time_estimate = elapsed_time / job.progress
                estimated_completion = job.started_at + total_time_estimate
            
            return {
                "job_id": job_id,
                "status": job.status,
                "progress": job.progress,
                "current_loss": job.current_loss,
                "elapsed_time_seconds": elapsed_time.total_seconds(),
                "estimated_completion": estimated_completion.isoformat() if estimated_completion else None,
                "allocated_gpus": allocated_gpus,
                "gpu_metrics": gpu_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to monitor training progress for job {job_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    def complete_training_job(self, job_id: str, final_loss: float = None):
        """Complete a training job and release resources."""
        try:
            if job_id not in self.active_jobs:
                logger.warning(f"Job {job_id} not found in active jobs")
                return
            
            job_info = self.active_jobs[job_id]
            job = job_info["job"]
            allocated_gpus = job_info["allocated_gpus"]
            
            # Update job status
            job.status = "completed"
            job.completed_at = datetime.now()
            job.progress = 1.0
            if final_loss is not None:
                job.current_loss = final_loss
            
            # Release GPUs
            self.release_gpus(allocated_gpus)
            
            # Remove from active jobs
            del self.active_jobs[job_id]
            
            # Update cluster status
            if len(self.active_jobs) == 0:
                self.status = ClusterStatus.READY
            
            logger.info(f"Completed training job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to complete training job {job_id}: {e}")
    
    def get_performance_metrics(self) -> ClusterMetrics:
        """Get comprehensive cluster performance metrics."""
        try:
            self._update_gpu_metrics()
            
            total_gpus = len(self.gpu_devices)
            active_gpus = sum(1 for gpu in self.gpu_devices.values() if not gpu.is_available)
            total_memory = sum(gpu.memory_total_gb for gpu in self.gpu_devices.values())
            used_memory = sum(gpu.memory_used_gb for gpu in self.gpu_devices.values())
            free_memory = total_memory - used_memory
            avg_utilization = np.mean([gpu.utilization_percent for gpu in self.gpu_devices.values()])
            avg_temperature = np.mean([gpu.temperature_celsius for gpu in self.gpu_devices.values()])
            total_power = sum(gpu.power_usage_watts for gpu in self.gpu_devices.values())
            
            # Estimate throughput (would be calculated from actual training metrics)
            throughput = sum(
                self.config.batch_size_per_gpu * gpu.utilization_percent / 100.0
                for gpu in self.gpu_devices.values()
            )
            
            metrics = ClusterMetrics(
                timestamp=datetime.now(),
                total_gpus=total_gpus,
                active_gpus=active_gpus,
                total_memory_gb=total_memory,
                used_memory_gb=used_memory,
                free_memory_gb=free_memory,
                average_utilization=avg_utilization,
                average_temperature=avg_temperature,
                total_power_usage_watts=total_power,
                throughput_samples_per_sec=throughput,
                active_jobs=len(self.active_jobs),
                queued_jobs=len(self.job_queue)
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 metrics
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            raise
    
    def auto_scale_cluster(self) -> bool:
        """Automatically scale cluster based on workload."""
        try:
            if not self.config.auto_scaling:
                return False
            
            # Get current metrics
            metrics = self.get_performance_metrics()
            
            # Check if scaling is needed
            avg_utilization = metrics.average_utilization
            queue_length = len(self.job_queue)
            
            if avg_utilization > 80.0 and queue_length > 2:
                # Scale up - would implement node addition in cloud environment
                logger.info("Auto-scaling: High utilization detected, scaling up recommended")
                return True
            elif avg_utilization < 20.0 and queue_length == 0:
                # Scale down - would implement node removal in cloud environment
                logger.info("Auto-scaling: Low utilization detected, scaling down recommended")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to auto-scale cluster: {e}")
            return False
    
    def shutdown_cluster(self):
        """Shutdown GPU cluster and cleanup resources."""
        try:
            # Complete all active jobs
            for job_id in list(self.active_jobs.keys()):
                self.complete_training_job(job_id)
            
            # Clear job queue
            self.job_queue.clear()
            
            # Cleanup distributed training
            self.distributed_manager.cleanup_distributed()
            
            # Clear GPU memory
            for gpu_id in self.gpu_devices.keys():
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
            
            self.status = ClusterStatus.SHUTDOWN
            logger.info("GPU cluster shutdown completed")
            
        except Exception as e:
            logger.error(f"Failed to shutdown GPU cluster: {e}")


# Factory function for creating GPU cluster manager
def create_gpu_cluster_manager(config: Optional[ClusterConfig] = None) -> GPUClusterManager:
    """Create and initialize a GPU cluster manager."""
    if config is None:
        config = ClusterConfig()
    
    return GPUClusterManager(config)