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
    
class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def _build_model(self, input_channels: int, num_classes: int) -> nn.Module:
        """Build advanced CNN architecture with attention mechanisms"""
        
        class AttentionBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(channels, channels // 8, 1),
                    nn.ReLU(),
                    nn.Conv1d(channels // 8, channels, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                attention_weights = self.attention(x)
                return x * attention_weights
        
        class SeismicCNN(nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()
                
                # Feature extraction layers
                self.conv_blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        AttentionBlock(64),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(64, 128, kernel_size=5, padding=2),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        AttentionBlock(128),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(128, 256, kernel_size=3, padding=1),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        AttentionBlock(256),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(256, 512, kernel_size=3, padding=1),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        AttentionBlock(512),
                        nn.AdaptiveAvgPool1d(1)
                    )
                ])
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
                
            def forward(self, x):
                for block in self.conv_blocks:
                    x = block(x)
                
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        return SeismicCNN(input_channels, num_classes).to(self.device)
    
    def detect_seismic_events(self, waveform_data: np.ndarray, 
                            sampling_rate: float = 100.0) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""
        
        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Event classification mapping
        event_types = {
            0: 'background_noise',
            1: 'local_earthquake',
            2: 'regional_earthquake',
            3: 'volcanic_tremor',
            4: 'explosion'
        }
        
        return {
            'event_type': event_types[predicted_class.item()],
            'confidence': probabilities.max().item(),
            'all_probabilities': {
                event_types[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'magnitude_estimate': self._estimate_magnitude(waveform_data),
            'arrival_time': self._detect_p_wave_arrival(waveform_data, sampling_rate),
            'frequency_content': self._analyze_frequency_content(waveform_data, sampling_rate)
        }
    
    def _preprocess_waveform(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Advanced waveform preprocessing"""
        # Bandpass filter
        from scipy import signal
        nyquist = sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = 25.0 / nyquist
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        
        # Normalize
        normalized_data = (filtered_data - np.mean(filtered_data, axis=-1, keepdims=True)) / \
                         (np.std(filtered_data, axis=-1, keepdims=True) + 1e-8)
        
        return normalized_data
    
    def _estimate_magnitude(self, waveform_data: np.ndarray) -> float:
        """Estimate earthquake magnitude from waveform amplitude"""
        max_amplitude = np.max(np.abs(waveform_data))
        # Simplified magnitude estimation (would use proper calibration in practice)
        magnitude = np.log10(max_amplitude * 1e6) - 3.0
        return max(0.0, min(9.0, magnitude))
    
    def _detect_p_wave_arrival(self, data: np.ndarray, sampling_rate: float) -> float:
        """Detect P-wave arrival time using STA/LTA algorithm"""
        # Short-term average / Long-term average
        sta_window = int(0.5 * sampling_rate)  # 0.5 seconds
        lta_window = int(10.0 * sampling_rate)  # 10 seconds
        
        energy = np.square(data)
        sta = np.convolve(energy, np.ones(sta_window)/sta_window, mode='same')
        lta = np.convolve(energy, np.ones(lta_window)/lta_window, mode='same')
        
        sta_lta_ratio = sta / (lta + 1e-8)
        trigger_threshold = 3.0
        
        trigger_indices = np.where(sta_lta_ratio > trigger_threshold)[0]
        if len(trigger_indices) > 0:
            return trigger_indices[0] / sampling_rate
        return 0.0
    
    def _analyze_frequency_content(self, data: np.ndarray, sampling_rate: float) -> Dict[str, float]:
        """Analyze frequency content of seismic signal"""
        from scipy import signal
        
        frequencies, psd = signal.welch(data, fs=sampling_rate, nperseg=1024)
        
        # Frequency bands of interest
        bands = {
            'low_freq': (0.1, 1.0),
            'mid_freq': (1.0, 10.0),
            'high_freq': (10.0, 25.0)
        }
        
        band_powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (frequencies >= low) & (frequencies <= high)
            band_powers[band_name] = np.trapz(psd[band_mask], frequencies[band_mask])
        
        # Dominant frequency
        dominant_freq = frequencies[np.argmax(psd)]
        
        return {
            'dominant_frequency': dominant_freq,
            'band_powers': band_powers,
            'spectral_centroid': np.sum(frequencies * psd) / np.sum(psd),
            'spectral_bandwidth': np.sqrt(np.sum(((frequencies - dominant_freq) ** 2) * psd) / np.sum(psd))
        }
"Sample size in MB")


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
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def __init__(self):
        self.processors = {
            'time_series': TimeSeriesProcessor(),
            'spatial': SpatialDataProcessor(),
            'spectral': SpectralAnalysisProcessor(),
            'statistical': StatisticalProcessor(),
            'ml_preprocessing': MLPreprocessor()
        }
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', str(uuid.uuid4())),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps based on configuration
        for step_name, step_config in processing_config.items():
            if step_name in self.processors:
                try:
                    processor = self.processors[step_name]
                    processed_data, step_results = processor.process(data, step_config)
                    
                    data = processed_data
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'success',
                        'results': step_results,
                        'output_size': len(data) if isinstance(data, list) else 'N/A'
                    })
                    
                except Exception as e:
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results

class TimeSeriesProcessor:
    """Advanced time series processing for geological data"""
    
    def process(self, data: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process time series data"""
        
        if not isinstance(data, (list, np.ndarray)):
            return data, {'error': 'Invalid data type for time series processing'}
        
        data_array = np.array(data)
        results = {}
        
        # Detrending
        if config.get('detrend', False):
            from scipy import signal
            data_array = signal.detrend(data_array, type=config.get('detrend_type', 'linear'))
            results['detrended'] = True
        
        # Filtering
        if 'filter' in config:
            filter_config = config['filter']
            data_array = self._apply_filter(data_array, filter_config)
            results['filtered'] = filter_config
        
        # Resampling
        if 'resample' in config:
            resample_config = config['resample']
            data_array = self._resample_data(data_array, resample_config)
            results['resampled'] = resample_config
        
        # Anomaly detection
        if config.get('detect_anomalies', False):
            anomalies = self._detect_anomalies(data_array, config.get('anomaly_config', {}))
            results['anomalies'] = anomalies
        
        return data_array.tolist(), results
    
    def _apply_filter(self, data: np.ndarray, filter_config: Dict[str, Any]) -> np.ndarray:
        """Apply digital filter to data"""
        from scipy import signal
        
        filter_type = filter_config.get('type', 'bandpass')
        sampling_rate = filter_config.get('sampling_rate', 100.0)
        
        if filter_type == 'bandpass':
            low_freq = filter_config.get('low_freq', 1.0)
            high_freq = filter_config.get('high_freq', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                [low_freq/nyquist, high_freq/nyquist],
                btype='band'
            )
            return signal.filtfilt(b, a, data)
        
        elif filter_type == 'lowpass':
            cutoff = filter_config.get('cutoff', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                cutoff/nyquist,
                btype='low'
            )
            return signal.filtfilt(b, a, data)
        
        return data
    
    def _resample_data(self, data: np.ndarray, resample_config: Dict[str, Any]) -> np.ndarray:
        """Resample data to different sampling rate"""
        from scipy import signal
        
        original_rate = resample_config.get('original_rate', 100.0)
        target_rate = resample_config.get('target_rate', 50.0)
        
        if original_rate != target_rate:
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        
        return data
    
    def _detect_anomalies(self, data: np.ndarray, anomaly_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        method = anomaly_config.get('method', 'statistical')
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = anomaly_config.get('threshold', 3.0)
            
            anomaly_indices = np.where(np.abs(data - mean_val) > threshold * std_val)[0]
            
            return {
                'method': 'statistical',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'threshold_used': threshold
            }
        
        elif method == 'isolation_forest':
            # Machine learning based anomaly detection
            from sklearn.ensemble import IsolationForest
            
            contamination = anomaly_config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            
            data_reshaped = data.reshape(-1, 1)
            anomaly_labels = clf.fit_predict(data_reshaped)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'contamination': contamination
            }
        
        return {'method': 'none', 'anomaly_count': 0}

    
    try:
        status = cluster_manager.status.value
        return {"status": status, "message": f"GPU cluster status: {status}"}
    except Exception as e:
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def __init__(self):
        self.processors = {
            'time_series': TimeSeriesProcessor(),
            'spatial': SpatialDataProcessor(),
            'spectral': SpectralAnalysisProcessor(),
            'statistical': StatisticalProcessor(),
            'ml_preprocessing': MLPreprocessor()
        }
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', str(uuid.uuid4())),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps based on configuration
        for step_name, step_config in processing_config.items():
            if step_name in self.processors:
                try:
                    processor = self.processors[step_name]
                    processed_data, step_results = processor.process(data, step_config)
                    
                    data = processed_data
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'success',
                        'results': step_results,
                        'output_size': len(data) if isinstance(data, list) else 'N/A'
                    })
                    
                except Exception as e:
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results

class TimeSeriesProcessor:
    """Advanced time series processing for geological data"""
    
    def process(self, data: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process time series data"""
        
        if not isinstance(data, (list, np.ndarray)):
            return data, {'error': 'Invalid data type for time series processing'}
        
        data_array = np.array(data)
        results = {}
        
        # Detrending
        if config.get('detrend', False):
            from scipy import signal
            data_array = signal.detrend(data_array, type=config.get('detrend_type', 'linear'))
            results['detrended'] = True
        
        # Filtering
        if 'filter' in config:
            filter_config = config['filter']
            data_array = self._apply_filter(data_array, filter_config)
            results['filtered'] = filter_config
        
        # Resampling
        if 'resample' in config:
            resample_config = config['resample']
            data_array = self._resample_data(data_array, resample_config)
            results['resampled'] = resample_config
        
        # Anomaly detection
        if config.get('detect_anomalies', False):
            anomalies = self._detect_anomalies(data_array, config.get('anomaly_config', {}))
            results['anomalies'] = anomalies
        
        return data_array.tolist(), results
    
    def _apply_filter(self, data: np.ndarray, filter_config: Dict[str, Any]) -> np.ndarray:
        """Apply digital filter to data"""
        from scipy import signal
        
        filter_type = filter_config.get('type', 'bandpass')
        sampling_rate = filter_config.get('sampling_rate', 100.0)
        
        if filter_type == 'bandpass':
            low_freq = filter_config.get('low_freq', 1.0)
            high_freq = filter_config.get('high_freq', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                [low_freq/nyquist, high_freq/nyquist],
                btype='band'
            )
            return signal.filtfilt(b, a, data)
        
        elif filter_type == 'lowpass':
            cutoff = filter_config.get('cutoff', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                cutoff/nyquist,
                btype='low'
            )
            return signal.filtfilt(b, a, data)
        
        return data
    
    def _resample_data(self, data: np.ndarray, resample_config: Dict[str, Any]) -> np.ndarray:
        """Resample data to different sampling rate"""
        from scipy import signal
        
        original_rate = resample_config.get('original_rate', 100.0)
        target_rate = resample_config.get('target_rate', 50.0)
        
        if original_rate != target_rate:
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        
        return data
    
    def _detect_anomalies(self, data: np.ndarray, anomaly_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        method = anomaly_config.get('method', 'statistical')
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = anomaly_config.get('threshold', 3.0)
            
            anomaly_indices = np.where(np.abs(data - mean_val) > threshold * std_val)[0]
            
            return {
                'method': 'statistical',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'threshold_used': threshold
            }
        
        elif method == 'isolation_forest':
            # Machine learning based anomaly detection
            from sklearn.ensemble import IsolationForest
            
            contamination = anomaly_config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            
            data_reshaped = data.reshape(-1, 1)
            anomaly_labels = clf.fit_predict(data_reshaped)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'contamination': contamination
            }
        
        return {'method': 'none', 'anomaly_count': 0}
