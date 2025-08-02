"""
Performance tests for scalability validation of Prometheus AI platform

Tests system performance under various load conditions and data volumes.
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import List, Dict, Any
import asyncio
import gc
import multiprocessing
import numpy as np
import psutil
import pytest
import queue
import threading
import time
import torch

# Performance test markers
pytestmark = pytest.mark.performance


class PerformanceMonitor:
    """Monitor system performance during tests"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        self.cpu_usage = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
        self.peak_memory = self.start_memory
        self.cpu_usage = []
        self.monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        self.end_time = time.time()
        self.end_memory = psutil.virtual_memory().used
    
    def _monitor_resources(self):
        """Monitor CPU and memory usage"""
        while self.monitoring:
            current_memory = psutil.virtual_memory().used
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage.append(cpu_percent)
            
            time.sleep(0.1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        execution_time = self.end_time - self.start_time if self.end_time else 0
        memory_used = (self.peak_memory - self.start_memory) / (1024 * 1024)  # MB
        avg_cpu = np.mean(self.cpu_usage) if self.cpu_usage else 0
        max_cpu = np.max(self.cpu_usage) if self.cpu_usage else 0
        
        return {
            'execution_time': execution_time,
            'memory_used_mb': memory_used,
            'peak_memory_mb': self.peak_memory / (1024 * 1024),
            'avg_cpu_percent': avg_cpu,
            'max_cpu_percent': max_cpu
        }


@pytest.fixture
def performance_monitor():
    """Provide performance monitor fixture"""
    return PerformanceMonitor()


class TestDataIngestionPerformance:
    """Test data ingestion performance and scalability"""
    
    @pytest.mark.parametrize("data_size", [1000, 10000, 100000])
    def test_seismic_data_ingestion_scalability(self, data_size, performance_monitor):
        """Test seismic data ingestion with increasing data sizes"""
        performance_monitor.start_monitoring()
        
        # Generate synthetic seismic data
        seismic_data = self._generate_seismic_data(data_size)
        
        # Process data
        processed_data = self._process_seismic_data(seismic_data)
        
        performance_monitor.stop_monitoring()
        metrics = performance_monitor.get_metrics()
        
        # Performance assertions
        assert len(processed_data) == data_size
        
        # Execution time should scale reasonably
        if data_size == 1000:
            assert metrics['execution_time'] < 1.0  # < 1 second for 1K events
        elif data_size == 10000:
            assert metrics['execution_time'] < 10.0  # < 10 seconds for 10K events
        elif data_size == 100000:
            assert metrics['execution_time'] < 100.0  # < 100 seconds for 100K events
        
        # Memory usage should be reasonable
        memory_per_event = metrics['memory_used_mb'] / data_size
        assert memory_per_event < 0.01  # < 10KB per event
        
        print(f"Data size: {data_size}, Time: {metrics['execution_time']:.2f}s, "
              f"Memory: {metrics['memory_used_mb']:.1f}MB, "
              f"CPU: {metrics['avg_cpu_percent']:.1f}%")
    
    @pytest.mark.parametrize("concurrent_streams", [1, 5, 10, 20])
    @pytest.mark.asyncio
    async def test_concurrent_data_streams(self, concurrent_streams, performance_monitor):
        """Test concurrent data stream processing"""
        performance_monitor.start_monitoring()
        
        # Create multiple concurrent data streams
        tasks = []
        for i in range(concurrent_streams):
            task = asyncio.create_task(
                self._simulate_data_stream(stream_id=i, duration=5.0)
            )
            tasks.append(task)
        
        # Process all streams concurrently
        results = await asyncio.gather(*tasks)
        
        performance_monitor.stop_monitoring()
        metrics = performance_monitor.get_metrics()
        
        # All streams should complete successfully
        assert len(results) == concurrent_streams
        for result in results:
            assert result['status'] == 'completed'
            assert result['data_points'] > 0
        
        # Performance should degrade gracefully with more streams
        throughput = sum(r['data_points'] for r in results) / metrics['execution_time']
        
        print(f"Concurrent streams: {concurrent_streams}, "
              f"Throughput: {throughput:.1f} events/sec, "
              f"CPU: {metrics['avg_cpu_percent']:.1f}%")
        
        # CPU usage should increase with concurrent streams but not exceed 90%
        assert metrics['max_cpu_percent'] < 90.0
    
    def test_memory_leak_detection(self, performance_monitor):
        """Test for memory leaks during continuous processing"""
        performance_monitor.start_monitoring()
        
        initial_memory = psutil.virtual_memory().used
        
        # Process data in batches to detect memory leaks
        for batch in range(10):
            # Generate and process data
            data = self._generate_seismic_data(1000)
            processed = self._process_seismic_data(data)
            
            # Force garbage collection
            del data, processed
            gc.collect()
            
            # Check memory usage
            current_memory = psutil.virtual_memory().used
            memory_increase = (current_memory - initial_memory) / (1024 * 1024)  # MB
            
            # Memory increase should be minimal after GC
            if batch > 2:  # Allow some warmup
                assert memory_increase < 50.0  # < 50MB increase
        
        performance_monitor.stop_monitoring()
        metrics = performance_monitor.get_metrics()
        
        print(f"Memory leak test completed. Peak memory: {metrics['peak_memory_mb']:.1f}MB")
    
    def _generate_seismic_data(self, count: int) -> List[Dict]:
        """Generate synthetic seismic data"""
        data = []
        base_time = datetime.utcnow()
        
        for i in range(count):
            event = {
                'timestamp': (base_time + timedelta(seconds=i)).isoformat(),
                'latitude': 44.428 + np.random.normal(0, 0.01),
                'longitude': -110.588 + np.random.normal(0, 0.01),
                'magnitude': np.random.uniform(1.0, 6.0),
                'depth': np.random.uniform(1.0, 50.0),
                'waveform': np.random.randn(1000).tolist()  # 1000-point waveform
            }
            data.append(event)
        
        return data
    
    def _process_seismic_data(self, data: List[Dict]) -> List[Dict]:
        """Process seismic data (mock processing)"""
        processed = []
        
        for event in data:
            # Simulate processing
            waveform = np.array(event['waveform'])
            
            # Simple processing: compute statistics
            processed_event = {
                'original_id': len(processed),
                'magnitude': event['magnitude'],
                'location': [event['latitude'], event['longitude']],
                'depth': event['depth'],
                'waveform_stats': {
                    'mean': float(np.mean(waveform)),
                    'std': float(np.std(waveform)),
                    'max': float(np.max(waveform)),
                    'min': float(np.min(waveform))
                }
            }
            processed.append(processed_event)
        
        return processed
    
    async def _simulate_data_stream(self, stream_id: int, duration: float) -> Dict:
        """Simulate a continuous data stream"""
        start_time = time.time()
        data_points = 0
        
        while time.time() - start_time < duration:
            # Simulate receiving data point
            data_point = {
                'stream_id': stream_id,
                'timestamp': datetime.utcnow().isoformat(),
                'value': np.random.randn()
            }
            
            # Simulate processing delay
            await asyncio.sleep(0.01)  # 10ms processing time
            data_points += 1
        
        return {
            'status': 'completed',
            'stream_id': stream_id,
            'data_points': data_points,
            'duration': time.time() - start_time
        }


class TestAIModelPerformance:
    """Test AI model performance and scalability"""
    
    @pytest.mark.parametrize("batch_size", [16, 64, 256, 1024])
    def test_pinn_model_batch_processing(self, batch_size, performance_monitor):
        """Test PINN model performance with different batch sizes"""
        performance_monitor.start_monitoring()
        
        # Create mock PINN model
        model = self._create_mock_pinn_model()
        
        # Generate test data
        input_data = torch.randn(batch_size, 4, requires_grad=True)
        target_data = torch.randn(batch_size, 3)
        
        # Forward pass
        with torch.no_grad():
            predictions = model(input_data)
        
        # Compute loss (including physics constraints)
        loss = self._compute_physics_informed_loss(predictions, target_data, input_data)
        
        performance_monitor.stop_monitoring()
        metrics = performance_monitor.get_metrics()
        
        # Verify output shape
        assert predictions.shape == (batch_size, 3)
        assert torch.isfinite(loss)
        
        # Performance should scale with batch size
        time_per_sample = metrics['execution_time'] / batch_size
        memory_per_sample = metrics['memory_used_mb'] / batch_size
        
        print(f"Batch size: {batch_size}, "
              f"Time/sample: {time_per_sample*1000:.2f}ms, "
              f"Memory/sample: {memory_per_sample:.3f}MB")
        
        # Time per sample should decrease with larger batches (better GPU utilization)
        if batch_size >= 64:
            assert time_per_sample < 0.01  # < 10ms per sample
    
    @pytest.mark.parametrize("model_complexity", ["simple", "medium", "complex"])
    def test_model_complexity_scaling(self, model_complexity, performance_monitor):
        """Test performance scaling with model complexity"""
        performance_monitor.start_monitoring()
        
        # Create models of different complexity
        if model_complexity == "simple":
            layer_sizes = [4, 32, 3]
        elif model_complexity == "medium":
            layer_sizes = [4, 64, 64, 3]
        else:  # complex
            layer_sizes = [4, 128, 128, 64, 3]
        
        model = self._create_mock_model(layer_sizes)
        
        # Test with fixed batch size
        batch_size = 64
        input_data = torch.randn(batch_size, 4)
        
        # Multiple forward passes to get stable timing
        num_passes = 100
        for _ in range(num_passes):
            with torch.no_grad():
                _ = model(input_data)
        
        performance_monitor.stop_monitoring()
        metrics = performance_monitor.get_metrics()
        
        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        time_per_pass = metrics['execution_time'] / num_passes
        
        print(f"Model complexity: {model_complexity}, "
              f"Parameters: {total_params}, "
              f"Time/pass: {time_per_pass*1000:.2f}ms")
        
        # More complex models should take longer but still be reasonable
        if model_complexity == "simple":
            assert time_per_pass < 0.001  # < 1ms
        elif model_complexity == "medium":
            assert time_per_pass < 0.005  # < 5ms
        else:  # complex
            assert time_per_pass < 0.02   # < 20ms
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_efficiency(self, performance_monitor):
        """Test GPU memory efficiency"""
        device = torch.device('cuda')
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        performance_monitor.start_monitoring()
        
        # Create model on GPU
        model = self._create_mock_pinn_model().to(device)
        
        # Process increasingly large batches
        max_batch_size = 0
        batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
        
        for batch_size in batch_sizes:
            try:
                input_data = torch.randn(batch_size, 4, device=device)
                
                with torch.no_grad():
                    output = model(input_data)
                
                max_batch_size = batch_size
                
                # Check GPU memory usage
                current_memory = torch.cuda.memory_allocated(device)
                memory_used = (current_memory - initial_memory) / (1024**2)  # MB
                
                print(f"Batch size: {batch_size}, GPU memory: {memory_used:.1f}MB")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU OOM at batch size: {batch_size}")
                    break
                else:
                    raise
        
        performance_monitor.stop_monitoring()
        
        # Should handle reasonable batch sizes
        assert max_batch_size >= 64
        
        # Clean up
        torch.cuda.empty_cache()
    
    def test_parallel_model_inference(self, performance_monitor):
        """Test parallel model inference"""
        performance_monitor.start_monitoring()
        
        # Create multiple models
        num_models = 4
        models = [self._create_mock_pinn_model() for _ in range(num_models)]
        
        # Test data
        test_data = [torch.randn(32, 4) for _ in range(num_models)]
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for model, data in zip(models, test_data):
            with torch.no_grad():
                result = model(data)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Parallel processing
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_models) as executor:
            futures = []
            for model, data in zip(models, test_data):
                future = executor.submit(self._model_inference, model, data)
                futures.append(future)
            
            parallel_results = [future.result() for future in futures]
        parallel_time = time.time() - start_time
        
        performance_monitor.stop_monitoring()
        
        # Results should be the same
        for seq_result, par_result in zip(sequential_results, parallel_results):
            assert torch.allclose(seq_result, par_result, atol=1e-6)
        
        # Parallel should be faster (if multiple cores available)
        speedup = sequential_time / parallel_time
        print(f"Sequential time: {sequential_time:.3f}s, "
              f"Parallel time: {parallel_time:.3f}s, "
              f"Speedup: {speedup:.2f}x")
        
        if multiprocessing.cpu_count() >= num_models:
            assert speedup > 1.5  # At least 1.5x speedup
    
    def _create_mock_pinn_model(self):
        """Create mock PINN model for testing"""
        import torch.nn as nn
        
        class MockPINNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(4, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, 3)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        return MockPINNModel()
    
    def _create_mock_model(self, layer_sizes):
        """Create mock model with specified layer sizes"""
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # No activation after last layer
                layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)
    
    def _compute_physics_informed_loss(self, predictions, targets, inputs):
        """Compute physics-informed loss"""
        # Data loss
        data_loss = torch.nn.functional.mse_loss(predictions, targets)
        
        # Mock physics constraint (gradient-based)
        gradients = torch.autograd.grad(
            predictions.sum(), inputs, create_graph=True
        )[0]
        physics_loss = torch.mean(gradients**2)
        
        return data_loss + 0.1 * physics_loss
    
    def _model_inference(self, model, data):
        """Perform model inference"""
        with torch.no_grad():
            return model(data)


class TestDatabasePerformance:
    """Test database performance and scalability"""
    
    @pytest.mark.parametrize("record_count", [1000, 10000, 100000])
    def test_time_series_insertion_performance(self, record_count, performance_monitor):
        """Test time series database insertion performance"""
        performance_monitor.start_monitoring()
        
        # Generate time series data
        time_series_data = self._generate_time_series_data(record_count)
        
        # Simulate database insertion
        insertion_results = self._simulate_influxdb_insertion(time_series_data)
        
        performance_monitor.stop_monitoring()
        metrics = performance_monitor.get_metrics()
        
        assert insertion_results['inserted_count'] == record_count
        assert insertion_results['errors'] == 0
        
        # Performance benchmarks
        insertion_rate = record_count / metrics['execution_time']
        
        print(f"Records: {record_count}, "
              f"Time: {metrics['execution_time']:.2f}s, "
              f"Rate: {insertion_rate:.0f} records/sec")
        
        # Should maintain reasonable insertion rates
        if record_count == 1000:
            assert insertion_rate > 500  # > 500 records/sec
        elif record_count == 10000:
            assert insertion_rate > 1000  # > 1000 records/sec
        elif record_count == 100000:
            assert insertion_rate > 2000  # > 2000 records/sec
    
    @pytest.mark.parametrize("query_complexity", ["simple", "complex", "aggregation"])
    def test_query_performance(self, query_complexity, performance_monitor):
        """Test database query performance"""
        performance_monitor.start_monitoring()
        
        # Simulate different query types
        if query_complexity == "simple":
            result = self._simulate_simple_query()
        elif query_complexity == "complex":
            result = self._simulate_complex_query()
        else:  # aggregation
            result = self._simulate_aggregation_query()
        
        performance_monitor.stop_monitoring()
        metrics = performance_monitor.get_metrics()
        
        assert result['status'] == 'success'
        assert result['row_count'] > 0
        
        print(f"Query complexity: {query_complexity}, "
              f"Time: {metrics['execution_time']:.3f}s, "
              f"Rows: {result['row_count']}")
        
        # Query time limits
        if query_complexity == "simple":
            assert metrics['execution_time'] < 0.1  # < 100ms
        elif query_complexity == "complex":
            assert metrics['execution_time'] < 1.0  # < 1s
        else:  # aggregation
            assert metrics['execution_time'] < 5.0  # < 5s
    
    def _generate_time_series_data(self, count: int) -> List[Dict]:
        """Generate time series data"""
        data = []
        base_time = datetime.utcnow()
        
        for i in range(count):
            record = {
                'timestamp': base_time + timedelta(seconds=i),
                'sensor_id': f'sensor_{i % 100}',  # 100 different sensors
                'value': np.random.randn(),
                'quality': np.random.choice(['good', 'fair', 'poor'], p=[0.8, 0.15, 0.05])
            }
            data.append(record)
        
        return data
    
    def _simulate_influxdb_insertion(self, data: List[Dict]) -> Dict:
        """Simulate InfluxDB insertion"""
        # Mock insertion with realistic timing
        time.sleep(len(data) * 0.00001)  # 10μs per record
        
        return {
            'inserted_count': len(data),
            'errors': 0,
            'duration': len(data) * 0.00001
        }
    
    def _simulate_simple_query(self) -> Dict:
        """Simulate simple database query"""
        time.sleep(0.05)  # 50ms query time
        return {'status': 'success', 'row_count': 1000}
    
    def _simulate_complex_query(self) -> Dict:
        """Simulate complex database query"""
        time.sleep(0.5)  # 500ms query time
        return {'status': 'success', 'row_count': 5000}
    
    def _simulate_aggregation_query(self) -> Dict:
        """Simulate aggregation query"""
        time.sleep(2.0)  # 2s query time
        return {'status': 'success', 'row_count': 100}


class TestSystemLoadPerformance:
    """Test system performance under various load conditions"""
    
    def test_stress_test_concurrent_users(self, performance_monitor):
        """Test system under concurrent user load"""
        performance_monitor.start_monitoring()
        
        num_users = 50
        requests_per_user = 10
        
        # Simulate concurrent users
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = []
            
            for user_id in range(num_users):
                future = executor.submit(
                    self._simulate_user_session, user_id, requests_per_user
                )
                futures.append(future)
            
            # Wait for all users to complete
            results = [future.result() for future in futures]
        
        performance_monitor.stop_monitoring()
        metrics = performance_monitor.get_metrics()
        
        # All users should complete successfully
        successful_users = sum(1 for r in results if r['status'] == 'success')
        total_requests = sum(r['requests_completed'] for r in results)
        
        assert successful_users == num_users
        assert total_requests == num_users * requests_per_user
        
        # Calculate throughput
        throughput = total_requests / metrics['execution_time']
        
        print(f"Concurrent users: {num_users}, "
              f"Total requests: {total_requests}, "
              f"Throughput: {throughput:.1f} req/sec, "
              f"CPU: {metrics['avg_cpu_percent']:.1f}%")
        
        # System should handle load gracefully
        assert throughput > 10  # > 10 requests/sec
        assert metrics['avg_cpu_percent'] < 80  # < 80% CPU usage
    
    def test_memory_pressure_handling(self, performance_monitor):
        """Test system behavior under memory pressure"""
        performance_monitor.start_monitoring()
        
        # Gradually increase memory usage
        memory_blocks = []
        block_size = 10 * 1024 * 1024  # 10MB blocks
        max_blocks = 100  # Up to 1GB
        
        try:
            for i in range(max_blocks):
                # Allocate memory block
                block = np.random.randn(block_size // 8)  # 8 bytes per float64
                memory_blocks.append(block)
                
                # Simulate some processing
                result = np.mean(block)
                assert np.isfinite(result)
                
                # Check if we're approaching memory limits
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 85:  # Stop at 85% memory usage
                    print(f"Stopped at {i+1} blocks due to memory pressure")
                    break
        
        except MemoryError:
            print(f"Memory error at {len(memory_blocks)} blocks")
        
        performance_monitor.stop_monitoring()
        metrics = performance_monitor.get_metrics()
        
        # Should handle at least some memory allocation
        assert len(memory_blocks) > 10
        
        print(f"Allocated blocks: {len(memory_blocks)}, "
              f"Memory used: {metrics['memory_used_mb']:.1f}MB")
        
        # Clean up
        del memory_blocks
        gc.collect()
    
    def _simulate_user_session(self, user_id: int, num_requests: int) -> Dict:
        """Simulate a user session with multiple requests"""
        requests_completed = 0
        
        try:
            for request_id in range(num_requests):
                # Simulate different types of requests
                request_type = np.random.choice(['query', 'analysis', 'visualization'])
                
                if request_type == 'query':
                    time.sleep(0.1)  # 100ms query
                elif request_type == 'analysis':
                    time.sleep(0.5)  # 500ms analysis
                else:  # visualization
                    time.sleep(0.2)  # 200ms visualization
                
                requests_completed += 1
            
            return {
                'status': 'success',
                'user_id': user_id,
                'requests_completed': requests_completed
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'user_id': user_id,
                'requests_completed': requests_completed,
                'error': str(e)
            }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])