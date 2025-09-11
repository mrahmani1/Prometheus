"""
Tests for Advanced Analytics Service
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Mock external dependencies
sys.modules['torch'] = MagicMock()
sys.modules['dask'] = MagicMock()
sys.modules['ray'] = MagicMock()
sys.modules['cupy'] = MagicMock()

# Add the service to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.distributed_processor import DistributedProcessor

class TestDistributedProcessor:
    """Test cases for DistributedProcessor"""
    
    @pytest.fixture
    def processor_config(self):
        """Configuration for testing"""
        return {
            'processing_mode': 'local',
            'max_workers': 2,
            'gpu_enabled': False,
            'memory_limit_gb': 4
        }
    
    @pytest.fixture
    def processor(self, processor_config):
        """Create test processor"""
        return DistributedProcessor(processor_config)
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor.processing_mode == 'local'
        assert processor.max_workers == 2
        assert processor.gpu_enabled is False
        assert processor.memory_limit == 4
        assert isinstance(processor.active_tasks, dict)
        assert isinstance(processor.task_results, dict)
    
    def test_estimate_data_size(self, processor):
        """Test data size estimation"""
        small_data = {'key': 'value'}
        large_data = {'key': 'x' * 1000}
        
        small_size = processor._estimate_data_size(small_data)
        large_size = processor._estimate_data_size(large_data)
        
        assert isinstance(small_size, int)
        assert isinstance(large_size, int)
        assert large_size > small_size
    
    def test_select_processing_strategy(self, processor):
        """Test processing strategy selection"""
        # Small data should use local parallel
        strategy = processor._select_processing_strategy(1024, 'geological_modeling')
        assert strategy == 'local_parallel'
        
        # Large data should prefer distributed (but fall back to local if not available)
        strategy = processor._select_processing_strategy(200 * 1024 * 1024, 'geological_modeling')
        assert strategy == 'local_parallel'  # Since dask/ray not available in test
    
    def test_calculate_optimal_batch_size(self, processor):
        """Test optimal batch size calculation"""
        # Test with different item counts and processing functions
        batch_size = processor._calculate_optimal_batch_size(100, 'geological_modeling')
        assert isinstance(batch_size, int)
        assert 1 <= batch_size <= 100
        
        # Complex processing should have smaller batches
        complex_batch = processor._calculate_optimal_batch_size(100, 'geological_modeling')
        simple_batch = processor._calculate_optimal_batch_size(100, 'simple_calculation')
        assert complex_batch <= simple_batch
    
    @pytest.mark.asyncio
    async def test_process_geological_modeling(self, processor):
        """Test geological modeling processing"""
        task_id = "test_geo_001"
        data = {
            'geological_units': ['granite', 'sandstone'],
            'coordinates': [[0, 0, 0], [1, 1, 1]]
        }
        parameters = {
            'resolution': 'high',
            'quality': 'ultra'
        }
        
        # Mock the computation method
        with patch.object(processor, '_geological_modeling_computation') as mock_compute:
            mock_compute.return_value = {
                'model_data': {'vertices': 1000, 'faces': 2000},
                'metadata': {'processing_time': 2.0, 'quality': 'high'},
                'visualization': {'mesh_file': 'model.vtk'},
                'quality_metrics': {'accuracy': 0.95, 'completeness': 0.98}
            }
            
            result = await processor.process_geological_modeling(task_id, data, parameters)
            
            # Check task was registered
            assert task_id in processor.active_tasks
            assert processor.active_tasks[task_id]['status'] == 'completed'
            assert processor.active_tasks[task_id]['progress'] == 100.0
            
            # Check result
            assert 'model_data' in result
            assert 'metadata' in result
            assert result['model_data']['vertices'] == 1000
    
    @pytest.mark.asyncio
    async def test_process_magmatic_analysis(self, processor):
        """Test magmatic analysis processing"""
        task_id = "test_mag_001"
        seismic_data = {
            'tomography_data': {'p_wave': [6.0, 6.5], 's_wave': [3.5, 3.8]},
            'velocity_models': {'layers': 3}
        }
        parameters = {
            'uncertainty_quantification': True,
            'physics_constraints': True
        }
        
        result = await processor.process_magmatic_analysis(task_id, seismic_data, parameters)
        
        # Check task was registered and completed
        assert task_id in processor.active_tasks
        assert processor.active_tasks[task_id]['status'] == 'completed'
        assert processor.active_tasks[task_id]['progress'] == 100.0
        
        # Check result structure
        assert 'final_model' in result
        assert 'validation_metrics' in result
        assert 'uncertainty_bounds' in result
    
    @pytest.mark.asyncio
    async def test_process_batch_analysis(self, processor):
        """Test batch analysis processing"""
        task_id = "test_batch_001"
        batch_data = [
            {'id': 1, 'data': 'sample1'},
            {'id': 2, 'data': 'sample2'},
            {'id': 3, 'data': 'sample3'}
        ]
        processing_function = 'geological_modeling'
        parameters = {'quality': 'high'}
        
        result = await processor.process_batch_analysis(
            task_id, batch_data, processing_function, parameters
        )
        
        # Check task was registered and completed
        assert task_id in processor.active_tasks
        assert processor.active_tasks[task_id]['status'] == 'completed'
        assert processor.active_tasks[task_id]['progress'] == 100.0
        
        # Check result
        assert isinstance(result, list)
        assert len(result) == len(batch_data)
        
        # Check each result item
        for i, item_result in enumerate(result):
            assert item_result['processed'] is True
            assert item_result['model_id'] == batch_data[i]['id']
    
    def test_process_batch_chunk(self, processor):
        """Test batch chunk processing"""
        batch = [
            {'id': 1, 'data': 'sample1'},
            {'id': 2, 'data': 'sample2'}
        ]
        processing_function = 'seismic_analysis'
        parameters = {}
        
        result = processor._process_batch_chunk(batch, processing_function, parameters)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['processed'] is True
        assert result[0]['event_id'] == 1
        assert result[1]['event_id'] == 2
    
    def test_geological_modeling_computation(self, processor):
        """Test core geological modeling computation"""
        data = {'test': 'data'}
        parameters = {'quality': 'high'}
        
        result = processor._geological_modeling_computation(data, parameters)
        
        assert isinstance(result, dict)
        assert 'model_data' in result
        assert 'metadata' in result
        assert 'visualization' in result
        assert 'quality_metrics' in result
        
        # Check specific values
        assert result['model_data']['vertices'] == 1000
        assert result['quality_metrics']['accuracy'] == 0.95
    
    def test_task_management(self, processor):
        """Test task status and result management"""
        task_id = "test_task_001"
        
        # Initially no task
        assert processor.get_task_status(task_id) is None
        assert processor.get_task_result(task_id) is None
        
        # Add task
        processor.active_tasks[task_id] = {
            'status': 'running',
            'progress': 50.0
        }
        processor.task_results[task_id] = {'result': 'test_data'}
        
        # Check task status and result
        status = processor.get_task_status(task_id)
        assert status['status'] == 'running'
        assert status['progress'] == 50.0
        
        result = processor.get_task_result(task_id)
        assert result['result'] == 'test_data'
        
        # Cancel task
        cancelled = processor.cancel_task(task_id)
        assert cancelled is True
        assert processor.active_tasks[task_id]['status'] == 'cancelled'
        
        # Try to cancel non-existent task
        cancelled = processor.cancel_task('non_existent')
        assert cancelled is False
    
    @pytest.mark.asyncio
    async def test_error_handling(self, processor):
        """Test error handling in processing"""
        task_id = "test_error_001"
        data = {'invalid': 'data'}
        parameters = {}
        
        # Mock computation to raise exception
        with patch.object(processor, '_geological_modeling_computation') as mock_compute:
            mock_compute.side_effect = Exception("Processing failed")
            
            with pytest.raises(Exception, match="Processing failed"):
                await processor.process_geological_modeling(task_id, data, parameters)
            
            # Check task status reflects error
            assert task_id in processor.active_tasks
            assert processor.active_tasks[task_id]['status'] == 'failed'
            assert 'error' in processor.active_tasks[task_id]

class TestServiceStructure:
    """Test service structure and configuration"""
    
    def test_service_files_exist(self):
        """Test that required service files exist"""
        service_root = os.path.join(os.path.dirname(__file__), '..')
        
        required_files = [
            'main.py',
            'requirements.txt',
            'Dockerfile',
            'core/__init__.py',
            'core/distributed_processor.py'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(service_root, file_name)
            assert os.path.exists(file_path), f"File {file_name} should exist"
    
    def test_dockerfile_content(self):
        """Test Dockerfile has required content"""
        service_root = os.path.join(os.path.dirname(__file__), '..')
        dockerfile_path = os.path.join(service_root, 'Dockerfile')
        
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for required Dockerfile elements
        assert 'FROM nvidia/cuda' in content
        assert 'EXPOSE 8004' in content
        assert 'HEALTHCHECK' in content
        assert 'requirements.txt' in content
    
    def test_requirements_content(self):
        """Test requirements.txt has required dependencies"""
        service_root = os.path.join(os.path.dirname(__file__), '..')
        requirements_path = os.path.join(service_root, 'requirements.txt')
        
        with open(requirements_path, 'r') as f:
            content = f.read()
        
        # Check for required dependencies
        required_deps = [
            'fastapi',
            'torch',
            'dask',
            'ray',
            'numpy',
            'scipy'
        ]
        
        for dep in required_deps:
            assert dep in content, f"Dependency {dep} should be in requirements.txt"

if __name__ == '__main__':
    pytest.main([__file__])