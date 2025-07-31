"""
Tests for GPU Cluster Manager functionality.

This module contains comprehensive tests for GPU cluster management,
distributed training coordination, and performance monitoring.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

# Mock PyTorch dependencies for testing
with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'torch.cuda': MagicMock(),
    'torch.distributed': MagicMock(),
    'torch.multiprocessing': MagicMock(),
    'torch.nn.parallel': MagicMock(),
    'torch.utils.data': MagicMock(),
    'torch.cuda.amp': MagicMock()
}):
    from ..core.gpu_cluster_manager import (
        GPUClusterManager, ClusterConfig, ClusterStatus, GPUType, GPUDevice,
        TrainingJob, ClusterMetrics, GPUMemoryOptimizer, DistributedTrainingManager,
        create_gpu_cluster_manager
    )


class TestClusterConfig:
    """Test cluster configuration."""
    
    def test_default_config(self):
        """Test default cluster configuration."""
        config = ClusterConfig()
        
        assert config.cluster_name == "geological-gpu-cluster"
        assert config.max_nodes == 8
        assert config.gpus_per_node == 8
        assert config.memory_threshold_gb == 2.0
        assert config.utilization_threshold == 90.0
        assert config.temperature_threshold == 85.0
        assert config.auto_scaling == True
        assert config.mixed_precision == True
        assert config.batch_size_per_gpu == 32
        assert config.distributed_backend == "nccl"
    
    def test_custom_config(self):
        """Test custom cluster configuration."""
        config = ClusterConfig(
            cluster_name="test-cluster",
            max_nodes=4,
            gpus_per_node=4,
            memory_threshold_gb=4.0,
            auto_scaling=False,
            batch_size_per_gpu=64
        )
        
        assert config.cluster_name == "test-cluster"
        assert config.max_nodes == 4
        assert config.gpus_per_node == 4
        assert config.memory_threshold_gb == 4.0
        assert config.auto_scaling == False
        assert config.batch_size_per_gpu == 64


class TestGPUDevice:
    """Test GPU device data structure."""
    
    def test_gpu_device_creation(self):
        """Test GPU device creation."""
        device = GPUDevice(
            device_id=0,
            name="Tesla V100",
            gpu_type=GPUType.V100,
            memory_total_gb=32.0,
            memory_free_gb=30.0,
            memory_used_gb=2.0,
            utilization_percent=25.0,
            temperature_celsius=65.0,
            power_usage_watts=200.0,
            compute_capability=(7, 0)
        )
        
        assert device.device_id == 0
        assert device.name == "Tesla V100"
        assert device.gpu_type == GPUType.V100
        assert device.memory_total_gb == 32.0
        assert device.memory_free_gb == 30.0
        assert device.memory_used_gb == 2.0
        assert device.utilization_percent == 25.0
        assert device.temperature_celsius == 65.0
        assert device.power_usage_watts == 200.0
        assert device.compute_capability == (7, 0)
        assert device.is_available == True


class TestTrainingJob:
    """Test training job data structure."""
    
    def test_training_job_creation(self):
        """Test training job creation."""
        job = TrainingJob(
            job_id="test_job_001",
            model_name="geological_pinn",
            dataset_path="/data/geological",
            num_epochs=100,
            learning_rate=1e-3,
            batch_size=32,
            num_gpus=2,
            priority=1
        )
        
        assert job.job_id == "test_job_001"
        assert job.model_name == "geological_pinn"
        assert job.dataset_path == "/data/geological"
        assert job.num_epochs == 100
        assert job.learning_rate == 1e-3
        assert job.batch_size == 32
        assert job.num_gpus == 2
        assert job.priority == 1
        assert job.status == "queued"
        assert job.progress == 0.0
        assert job.current_loss == float('inf')


class TestGPUMemoryOptimizer:
    """Test GPU memory optimizer."""
    
    def test_optimizer_initialization(self):
        """Test memory optimizer initialization."""
        optimizer = GPUMemoryOptimizer(safety_margin=0.15)
        
        assert optimizer.safety_margin == 0.15
        assert optimizer.memory_history == {}
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.set_device')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_get_optimal_batch_size(self, mock_reserved, mock_allocated, 
                                   mock_properties, mock_set_device, mock_available):
        """Test optimal batch size calculation."""
        mock_available.return_value = True
        
        # Mock device properties
        mock_props = Mock()
        mock_props.total_memory = 32 * 1024**3  # 32 GB
        mock_properties.return_value = mock_props
        
        # Mock memory usage
        mock_allocated.return_value = 2 * 1024**3  # 2 GB allocated
        mock_reserved.return_value = 1 * 1024**3   # 1 GB cached
        
        optimizer = GPUMemoryOptimizer()
        
        # Test batch size calculation
        batch_size = optimizer.get_optimal_batch_size(
            gpu_id=0,
            model_size_mb=500,  # 500 MB model
            sample_size_mb=10   # 10 MB per sample
        )
        
        assert isinstance(batch_size, int)
        assert batch_size >= 1
        mock_set_device.assert_called_with(0)
    
    @patch('torch.cuda.is_available')
    def test_get_optimal_batch_size_no_cuda(self, mock_available):
        """Test batch size calculation when CUDA is not available."""
        mock_available.return_value = False
        
        optimizer = GPUMemoryOptimizer()
        batch_size = optimizer.get_optimal_batch_size(0, 500, 10)
        
        assert batch_size == 32  # Default fallback
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.set_device')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_monitor_memory_usage(self, mock_reserved, mock_allocated, 
                                 mock_properties, mock_set_device, mock_available):
        """Test memory usage monitoring."""
        mock_available.return_value = True
        
        # Mock device properties
        mock_props = Mock()
        mock_props.total_memory = 16 * 1024**3  # 16 GB
        mock_properties.return_value = mock_props
        
        # Mock memory usage
        mock_allocated.return_value = 4 * 1024**3  # 4 GB allocated
        mock_reserved.return_value = 2 * 1024**3   # 2 GB cached
        
        optimizer = GPUMemoryOptimizer()
        memory_info = optimizer.monitor_memory_usage(0)
        
        assert "total" in memory_info
        assert "allocated" in memory_info
        assert "cached" in memory_info
        assert "free" in memory_info
        assert memory_info["total"] == 16.0  # 16 GB
        assert memory_info["allocated"] == 4.0  # 4 GB
        assert memory_info["cached"] == 2.0  # 2 GB
        assert memory_info["free"] == 12.0  # 12 GB


class TestDistributedTrainingManager:
    """Test distributed training manager."""
    
    def test_manager_initialization(self):
        """Test distributed training manager initialization."""
        config = ClusterConfig()
        manager = DistributedTrainingManager(config)
        
        assert manager.config == config
        assert manager.world_size == 0
        assert manager.rank == 0
        assert manager.local_rank == 0
        assert manager.is_initialized == False
    
    @patch('torch.distributed.init_process_group')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.set_device')
    def test_initialize_distributed(self, mock_set_device, mock_device_count, mock_init_group):
        """Test distributed training initialization."""
        mock_device_count.return_value = 8
        
        config = ClusterConfig()
        manager = DistributedTrainingManager(config)
        
        manager.initialize_distributed(rank=0, world_size=4)
        
        assert manager.rank == 0
        assert manager.world_size == 4
        assert manager.local_rank == 0
        assert manager.is_initialized == True
        
        mock_init_group.assert_called_once()
        mock_set_device.assert_called_with(0)
    
    @patch('torch.distributed.destroy_process_group')
    def test_cleanup_distributed(self, mock_destroy_group):
        """Test distributed training cleanup."""
        config = ClusterConfig()
        manager = DistributedTrainingManager(config)
        manager.is_initialized = True
        
        manager.cleanup_distributed()
        
        assert manager.is_initialized == False
        mock_destroy_group.assert_called_once()


class TestGPUClusterManager:
    """Test GPU cluster manager functionality."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_manager_initialization_no_cuda(self, mock_device_count, mock_available):
        """Test manager initialization when CUDA is not available."""
        mock_available.return_value = False
        
        config = ClusterConfig()
        
        with pytest.raises(RuntimeError, match="GPU cluster functionality requires PyTorch with CUDA support"):
            GPUClusterManager(config)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.set_device')
    def test_manager_initialization_success(self, mock_set_device, mock_properties, 
                                          mock_device_count, mock_available):
        """Test successful manager initialization."""
        mock_available.return_value = True
        mock_device_count.return_value = 2
        
        # Mock device properties
        mock_props = Mock()
        mock_props.name = "Tesla V100"
        mock_props.total_memory = 16 * 1024**3  # 16 GB
        mock_props.major = 7
        mock_props.minor = 0
        mock_properties.return_value = mock_props
        
        config = ClusterConfig()
        
        with patch.object(GPUClusterManager, '_identify_gpu_type') as mock_identify:
            mock_identify.return_value = GPUType.V100
            
            manager = GPUClusterManager(config)
            
            assert manager.config == config
            assert manager.status == ClusterStatus.READY
            assert len(manager.gpu_devices) == 2
            assert 0 in manager.gpu_devices
            assert 1 in manager.gpu_devices
    
    def test_identify_gpu_type(self):
        """Test GPU type identification."""
        config = ClusterConfig()
        
        with patch.object(GPUClusterManager, '_initialize_cluster'):
            manager = GPUClusterManager(config)
            
            assert manager._identify_gpu_type("Tesla V100") == GPUType.V100
            assert manager._identify_gpu_type("Tesla A100") == GPUType.A100
            assert manager._identify_gpu_type("H100") == GPUType.H100
            assert manager._identify_gpu_type("RTX 4090") == GPUType.RTX_4090
            assert manager._identify_gpu_type("RTX 3090") == GPUType.RTX_3090
            assert manager._identify_gpu_type("Unknown GPU") == GPUType.UNKNOWN
    
    def test_submit_training_job(self):
        """Test training job submission."""
        config = ClusterConfig()
        
        with patch.object(GPUClusterManager, '_initialize_cluster'):
            manager = GPUClusterManager(config)
            
            job_config = {
                "model_name": "test_model",
                "dataset_path": "/data/test",
                "num_epochs": 50,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "num_gpus": 1,
                "priority": 1
            }
            
            job_id = manager.submit_training_job(job_config)
            
            assert job_id in manager.training_jobs
            assert job_id in manager.job_queue
            
            job = manager.training_jobs[job_id]
            assert job.model_name == "test_model"
            assert job.num_epochs == 50
            assert job.learning_rate == 1e-3
            assert job.status == "queued"
    
    def test_allocate_gpus(self):
        """Test GPU allocation."""
        config = ClusterConfig()
        
        with patch.object(GPUClusterManager, '_initialize_cluster'):
            manager = GPUClusterManager(config)
            
            # Mock GPU devices
            manager.gpu_devices = {
                0: GPUDevice(0, "GPU0", GPUType.V100, 16.0, 14.0, 2.0, 20.0, 65.0, 150.0, (7, 0)),
                1: GPUDevice(1, "GPU1", GPUType.V100, 16.0, 12.0, 4.0, 30.0, 70.0, 200.0, (7, 0)),
                2: GPUDevice(2, "GPU2", GPUType.V100, 16.0, 8.0, 8.0, 50.0, 75.0, 300.0, (7, 0))
            }
            
            # Allocate 2 GPUs
            allocated_gpus = manager.allocate_gpus(num_gpus=2, memory_required_gb=4.0)
            
            assert len(allocated_gpus) == 2
            assert 0 in allocated_gpus  # GPU 0 has lowest utilization
            assert 1 in allocated_gpus  # GPU 1 has second lowest utilization
            
            # Check that GPUs are marked as unavailable
            assert not manager.gpu_devices[0].is_available
            assert not manager.gpu_devices[1].is_available
            assert manager.gpu_devices[2].is_available  # Not allocated
    
    def test_allocate_gpus_insufficient(self):
        """Test GPU allocation when insufficient GPUs are available."""
        config = ClusterConfig()
        
        with patch.object(GPUClusterManager, '_initialize_cluster'):
            manager = GPUClusterManager(config)
            
            # Mock only 1 GPU device
            manager.gpu_devices = {
                0: GPUDevice(0, "GPU0", GPUType.V100, 16.0, 14.0, 2.0, 20.0, 65.0, 150.0, (7, 0))
            }
            
            # Try to allocate 2 GPUs
            with pytest.raises(RuntimeError, match="Insufficient GPUs available"):
                manager.allocate_gpus(num_gpus=2, memory_required_gb=4.0)
    
    def test_release_gpus(self):
        """Test GPU release."""
        config = ClusterConfig()
        
        with patch.object(GPUClusterManager, '_initialize_cluster'):
            with patch('torch.cuda.set_device') as mock_set_device:
                with patch('torch.cuda.empty_cache') as mock_empty_cache:
                    manager = GPUClusterManager(config)
                    
                    # Mock GPU devices
                    manager.gpu_devices = {
                        0: GPUDevice(0, "GPU0", GPUType.V100, 16.0, 14.0, 2.0, 20.0, 65.0, 150.0, (7, 0)),
                        1: GPUDevice(1, "GPU1", GPUType.V100, 16.0, 12.0, 4.0, 30.0, 70.0, 200.0, (7, 0))
                    }
                    
                    # Mark GPUs as unavailable
                    manager.gpu_devices[0].is_available = False
                    manager.gpu_devices[1].is_available = False
                    
                    # Release GPUs
                    manager.release_gpus([0, 1])
                    
                    # Check that GPUs are marked as available
                    assert manager.gpu_devices[0].is_available
                    assert manager.gpu_devices[1].is_available
                    
                    # Check that CUDA operations were called
                    assert mock_set_device.call_count == 2
                    assert mock_empty_cache.call_count == 2
    
    def test_get_performance_metrics(self):
        """Test performance metrics collection."""
        config = ClusterConfig()
        
        with patch.object(GPUClusterManager, '_initialize_cluster'):
            with patch.object(GPUClusterManager, '_update_gpu_metrics'):
                manager = GPUClusterManager(config)
                
                # Mock GPU devices
                manager.gpu_devices = {
                    0: GPUDevice(0, "GPU0", GPUType.V100, 16.0, 12.0, 4.0, 25.0, 65.0, 200.0, (7, 0)),
                    1: GPUDevice(1, "GPU1", GPUType.V100, 16.0, 10.0, 6.0, 37.5, 70.0, 300.0, (7, 0))
                }
                
                metrics = manager.get_performance_metrics()
                
                assert isinstance(metrics, ClusterMetrics)
                assert metrics.total_gpus == 2
                assert metrics.total_memory_gb == 32.0  # 16 + 16
                assert metrics.used_memory_gb == 10.0   # 4 + 6
                assert metrics.free_memory_gb == 22.0   # 12 + 10
                assert metrics.average_utilization == 31.25  # (25 + 37.5) / 2
                assert metrics.average_temperature == 67.5   # (65 + 70) / 2
                assert metrics.total_power_usage_watts == 500.0  # 200 + 300


class TestGPUClusterFactory:
    """Test GPU cluster factory function."""
    
    @patch.object(GPUClusterManager, '_initialize_cluster')
    def test_create_gpu_cluster_manager_default(self, mock_init):
        """Test factory function with default config."""
        manager = create_gpu_cluster_manager()
        
        assert isinstance(manager, GPUClusterManager)
        assert isinstance(manager.config, ClusterConfig)
    
    @patch.object(GPUClusterManager, '_initialize_cluster')
    def test_create_gpu_cluster_manager_custom(self, mock_init):
        """Test factory function with custom config."""
        config = ClusterConfig(cluster_name="test-cluster", max_nodes=4)
        manager = create_gpu_cluster_manager(config)
        
        assert isinstance(manager, GPUClusterManager)
        assert manager.config == config
        assert manager.config.cluster_name == "test-cluster"
        assert manager.config.max_nodes == 4


class TestGPUClusterIntegration:
    """Integration tests for GPU cluster functionality."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.set_device')
    def test_full_workflow_simulation(self, mock_set_device, mock_properties, 
                                     mock_device_count, mock_available):
        """Test complete GPU cluster workflow simulation."""
        mock_available.return_value = True
        mock_device_count.return_value = 4
        
        # Mock device properties
        mock_props = Mock()
        mock_props.name = "Tesla A100"
        mock_props.total_memory = 40 * 1024**3  # 40 GB
        mock_props.major = 8
        mock_props.minor = 0
        mock_properties.return_value = mock_props
        
        config = ClusterConfig(max_nodes=2, gpus_per_node=4)
        
        with patch.object(GPUClusterManager, '_identify_gpu_type') as mock_identify:
            mock_identify.return_value = GPUType.A100
            
            # Initialize cluster
            manager = GPUClusterManager(config)
            assert manager.status == ClusterStatus.READY
            assert len(manager.gpu_devices) == 4
            
            # Submit job
            job_config = {
                "model_name": "integration_test_model",
                "dataset_path": "/data/integration",
                "num_epochs": 10,
                "learning_rate": 1e-3,
                "batch_size": 64,
                "num_gpus": 2,
                "priority": 1
            }
            
            job_id = manager.submit_training_job(job_config)
            assert job_id in manager.training_jobs
            assert len(manager.job_queue) == 1
            
            # Start training
            with patch('torch.cuda.empty_cache'):
                success = manager.start_distributed_training(job_id)
                assert success
                assert job_id in manager.active_jobs
                assert len(manager.active_jobs) == 1
            
            # Monitor progress
            progress_info = manager.monitor_training_progress(job_id)
            assert progress_info["job_id"] == job_id
            assert "allocated_gpus" in progress_info
            
            # Get metrics
            metrics = manager.get_performance_metrics()
            assert metrics.total_gpus == 4
            assert metrics.active_gpus == 2  # 2 GPUs allocated for training
            
            # Complete job
            manager.complete_training_job(job_id, final_loss=0.01)
            assert job_id not in manager.active_jobs
            assert manager.training_jobs[job_id].status == "completed"
            
            # Shutdown
            manager.shutdown_cluster()
            assert manager.status == ClusterStatus.SHUTDOWN


if __name__ == "__main__":
    pytest.main([__file__])