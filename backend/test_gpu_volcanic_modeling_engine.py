"""
Test suite for GPU-Accelerated Volcanic Modeling Engine

This test suite validates the functionality of the GPU-accelerated volcanic modeling engine,
including molecular-scale calculations, optimization algorithms, linear solvers, and pattern recognition.
"""

import pytest
import asyncio
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.gpu_volcanic_modeling_engine import (
    GPUVolcanicModelingEngine,
    GPUMemoryManager,
    CUDAMagmaPropertyCalculator,
    GPUOptimizationEngine,
    GPULinearSolver,
    GPUPatternRecognizer,
    GPULoadBalancer,
    GPUPerformanceMonitor,
    MagmaComposition,
    PhysicalConditions,
    GPUComputationRequest,
    GPUSimulationResult
)

class TestGPUMemoryManager:
    """Test GPU memory management functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.memory_manager = GPUMemoryManager()
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization"""
        assert hasattr(self.memory_manager, 'device_count')
        assert hasattr(self.memory_manager, 'memory_pools')
        assert isinstance(self.memory_manager.memory_pools, dict)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_memory_stats(self):
        """Test memory statistics retrieval"""
        if torch.cuda.device_count() > 0:
            stats = self.memory_manager.get_memory_stats(0)
            
            assert 'total_memory' in stats
            assert 'allocated_memory' in stats
            assert 'free_memory' in stats
            assert 'utilization' in stats
            assert stats['total_memory'] > 0
            assert stats['utilization'] >= 0.0
            assert stats['utilization'] <= 1.0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_managed_allocation(self):
        """Test managed memory allocation context manager"""
        if torch.cuda.device_count() > 0:
            with self.memory_manager.managed_allocation(0, 1024) as allocation_id:
                assert allocation_id is not None
                assert isinstance(allocation_id, str)
                assert allocation_id.startswith('alloc_0_')

class TestGPUOptimizationEngine:
    """Test GPU optimization engine functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.optimization_engine = GPUOptimizationEngine()
    
    def test_optimization_engine_initialization(self):
        """Test optimization engine initialization"""
        assert hasattr(self.optimization_engine, 'device')
        assert hasattr(self.optimization_engine, 'scaler')
    
    @pytest.mark.asyncio
    async def test_volcanic_system_optimization(self):
        """Test volcanic system parameter optimization"""
        parameters = {
            'chamber_pressure': 50.0,
            'magma_temperature': 1100.0,
            'gas_content': 0.03
        }
        
        constraints = {
            'chamber_pressure': (10.0, 200.0),
            'magma_temperature': (800.0, 1400.0),
            'gas_content': (0.0, 0.1)
        }
        
        result = await self.optimization_engine.optimize_volcanic_system(
            parameters, constraints, "eruption_probability"
        )
        
        assert isinstance(result, GPUSimulationResult)
        assert 'optimized_parameters' in result.results
        assert result.execution_time > 0
        assert result.validation_score >= 0.0
        assert result.validation_score <= 1.0
        
        # Check that constraints are satisfied
        optimized_params = result.results['optimized_parameters']
        for param_name, value in optimized_params.items():
            min_val, max_val = constraints[param_name]
            assert min_val <= value <= max_val

class TestGPULinearSolver:
    """Test GPU linear solver functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.linear_solver = GPULinearSolver()
    
    def test_linear_solver_initialization(self):
        """Test linear solver initialization"""
        assert hasattr(self.linear_solver, 'device')
    
    @pytest.mark.asyncio
    async def test_multiphysics_system_solving(self):
        """Test multi-physics equation system solving"""
        # Create a simple test system
        n = 10
        A = np.random.rand(n, n) + np.eye(n) * 5  # Diagonally dominant
        boundary_conditions = {
            'dirichlet': {0: 1.0, n-1: 0.0}
        }
        
        result = await self.linear_solver.solve_multiphysics_system(
            A, np.ones(n), boundary_conditions
        )
        
        assert isinstance(result, GPUSimulationResult)
        assert 'solution' in result.results
        assert 'residual' in result.results
        assert len(result.results['solution']) == n
        assert result.execution_time > 0
        assert result.results['residual'] >= 0
    
    def test_boundary_condition_application(self):
        """Test boundary condition application"""
        n = 5
        A = torch.eye(n, device=self.linear_solver.device)
        b = torch.ones(n, device=self.linear_solver.device)
        
        boundary_conditions = {
            'dirichlet': {0: 2.0, 4: 3.0}
        }
        
        A_mod, b_mod = self.linear_solver._apply_boundary_conditions(A, b, boundary_conditions)
        
        # Check Dirichlet conditions
        assert A_mod[0, 0] == 1.0
        assert b_mod[0] == 2.0
        assert A_mod[4, 4] == 1.0
        assert b_mod[4] == 3.0
    
    def test_conjugate_gradient_solver(self):
        """Test conjugate gradient solver"""
        n = 5
        A = torch.eye(n, device=self.linear_solver.device) * 2
        b = torch.ones(n, device=self.linear_solver.device)
        
        solution = self.linear_solver._conjugate_gradient_gpu(A, b, max_iter=100, tol=1e-6)
        
        assert solution.shape == (n,)
        # Check solution accuracy
        residual = torch.norm(A @ solution - b)
        assert residual < 1e-5

class TestGPUPatternRecognizer:
    """Test GPU pattern recognition functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.pattern_recognizer = GPUPatternRecognizer()
    
    def test_pattern_recognizer_initialization(self):
        """Test pattern recognizer initialization"""
        assert hasattr(self.pattern_recognizer, 'device')
        assert hasattr(self.pattern_recognizer, 'model')
    
    @pytest.mark.asyncio
    async def test_unsupervised_pattern_recognition(self):
        """Test unsupervised pattern recognition"""
        # Create synthetic volcanic data
        np.random.seed(42)
        data = np.random.rand(100, 20)  # 100 samples, 20 features
        
        result = await self.pattern_recognizer.recognize_patterns(data)
        
        assert isinstance(result, GPUSimulationResult)
        assert 'cluster_labels' in result.results
        assert 'cluster_centers' in result.results
        assert len(result.results['cluster_labels']) == 100
        assert result.execution_time > 0
        assert result.validation_score >= 0.0

class TestGPULoadBalancer:
    """Test GPU load balancer functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.load_balancer = GPULoadBalancer()
    
    def test_load_balancer_initialization(self):
        """Test load balancer initialization"""
        assert hasattr(self.load_balancer, 'gpu_count')
        assert hasattr(self.load_balancer, 'cpu_count')
        assert hasattr(self.load_balancer, 'gpu_utilization')
    
    @pytest.mark.asyncio
    @patch('GPUtil.getGPUs')
    @patch('psutil.cpu_percent')
    async def test_balance_computation(self, mock_cpu_percent, mock_get_gpus):
        """Test computation load balancing"""
        # Mock GPU information
        mock_gpu = Mock()
        mock_gpu.memoryUtil = 0.3
        mock_gpu.load = 0.2
        mock_gpu.temperature = 65.0
        mock_get_gpus.return_value = [mock_gpu]
        
        mock_cpu_percent.return_value = 30.0
        
        computation_request = GPUComputationRequest(
            computation_id="test_computation",
            algorithm_type="test_algorithm",
            data_size=1000,
            memory_requirement=50.0
        )
        
        device_id = await self.load_balancer.balance_computation(computation_request)
        
        assert isinstance(device_id, int)
        assert device_id >= -1  # -1 for CPU, >= 0 for GPU

class TestGPUPerformanceMonitor:
    """Test GPU performance monitoring functionality"""
    
    def setup_method(self):
        """Setup test environment with mocked database"""
        with patch('sqlalchemy.create_engine'), \
             patch('redis.Redis'):
            self.performance_monitor = GPUPerformanceMonitor()
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization"""
        assert hasattr(self.performance_monitor, 'db_engine')
        assert hasattr(self.performance_monitor, 'redis_client')

class TestGPUVolcanicModelingEngine:
    """Test main GPU volcanic modeling engine"""
    
    def setup_method(self):
        """Setup test environment"""
        with patch('modules.gpu_volcanic_modeling_engine.CUDAMagmaPropertyCalculator'), \
             patch('sqlalchemy.create_engine'), \
             patch('redis.Redis'):
            self.engine = GPUVolcanicModelingEngine()
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        assert hasattr(self.engine, 'memory_manager')
        assert hasattr(self.engine, 'optimization_engine')
        assert hasattr(self.engine, 'linear_solver')
        assert hasattr(self.engine, 'pattern_recognizer')
        assert hasattr(self.engine, 'load_balancer')
        assert hasattr(self.engine, 'performance_monitor')
    
    @pytest.mark.asyncio
    async def test_cpu_fallback_magma_properties(self):
        """Test CPU fallback for magma property calculations"""
        composition = MagmaComposition(
            sio2=0.65, al2o3=0.15, fe2o3=0.03, feo=0.02, mgo=0.02,
            cao=0.04, na2o=0.04, k2o=0.03, tio2=0.01, p2o5=0.01,
            h2o=0.05, co2=0.01, temperature=1200.0, pressure=100.0
        )
        
        conditions = PhysicalConditions(
            temperature=1200.0, pressure=100.0, oxygen_fugacity=-10.0,
            water_content=0.05, crystal_fraction=0.1, bubble_fraction=0.05
        )
        
        result = await self.engine._cpu_fallback_magma_properties([composition], [conditions])
        
        assert isinstance(result, GPUSimulationResult)
        assert 'viscosities' in result.results
        assert 'densities' in result.results
        assert 'heat_capacities' in result.results
        assert len(result.results['viscosities']) == 1
        assert result.execution_time > 0
        assert result.gpu_utilization == 0.0  # CPU fallback
    
    @pytest.mark.asyncio
    async def test_cpu_fallback_optimization(self):
        """Test CPU fallback for optimization"""
        parameters = {
            'chamber_pressure': 50.0,
            'magma_temperature': 1100.0
        }
        
        constraints = {
            'chamber_pressure': (10.0, 200.0),
            'magma_temperature': (800.0, 1400.0)
        }
        
        result = await self.engine._cpu_fallback_optimization(parameters, constraints)
        
        assert isinstance(result, GPUSimulationResult)
        assert 'optimized_parameters' in result.results
        assert result.execution_time > 0
        assert result.gpu_utilization == 0.0
        
        # Check constraints are satisfied
        optimized_params = result.results['optimized_parameters']
        for param_name, value in optimized_params.items():
            min_val, max_val = constraints[param_name]
            assert min_val <= value <= max_val
    
    @pytest.mark.asyncio
    async def test_cpu_fallback_linear_solver(self):
        """Test CPU fallback for linear system solving"""
        n = 5
        A = np.random.rand(n, n) + np.eye(n) * 2
        boundary_conditions = {'dirichlet': {0: 1.0}}
        
        result = await self.engine._cpu_fallback_linear_solver(A, boundary_conditions)
        
        assert isinstance(result, GPUSimulationResult)
        assert 'solution' in result.results
        assert 'residual' in result.results
        assert len(result.results['solution']) == n
        assert result.execution_time > 0
        assert result.gpu_utilization == 0.0
    
    @pytest.mark.asyncio
    async def test_cpu_fallback_pattern_recognition(self):
        """Test CPU fallback for pattern recognition"""
        data = np.random.rand(50, 10)
        config = {'n_clusters': 3}
        
        result = await self.engine._cpu_fallback_pattern_recognition(data, config)
        
        assert isinstance(result, GPUSimulationResult)
        assert 'cluster_labels' in result.results
        assert 'cluster_centers' in result.results
        assert len(result.results['cluster_labels']) == 50
        assert len(result.results['cluster_centers']) == 3
        assert result.execution_time > 0
        assert result.gpu_utilization == 0.0

class TestDataStructures:
    """Test data structure classes"""
    
    def test_magma_composition_creation(self):
        """Test MagmaComposition dataclass"""
        composition = MagmaComposition(
            sio2=0.65, al2o3=0.15, fe2o3=0.03, feo=0.02, mgo=0.02,
            cao=0.04, na2o=0.04, k2o=0.03, tio2=0.01, p2o5=0.01,
            h2o=0.05, co2=0.01, temperature=1200.0, pressure=100.0
        )
        
        assert composition.sio2 == 0.65
        assert composition.temperature == 1200.0
        assert composition.pressure == 100.0
    
    def test_physical_conditions_creation(self):
        """Test PhysicalConditions dataclass"""
        conditions = PhysicalConditions(
            temperature=1200.0, pressure=100.0, oxygen_fugacity=-10.0,
            water_content=0.05, crystal_fraction=0.1, bubble_fraction=0.05
        )
        
        assert conditions.temperature == 1200.0
        assert conditions.pressure == 100.0
        assert conditions.oxygen_fugacity == -10.0
    
    def test_gpu_computation_request_creation(self):
        """Test GPUComputationRequest dataclass"""
        request = GPUComputationRequest(
            computation_id="test_id",
            algorithm_type="test_algorithm",
            data_size=1000,
            memory_requirement=50.0
        )
        
        assert request.computation_id == "test_id"
        assert request.algorithm_type == "test_algorithm"
        assert request.data_size == 1000
        assert request.memory_requirement == 50.0
        assert request.priority == 1  # Default value
    
    def test_gpu_simulation_result_creation(self):
        """Test GPUSimulationResult dataclass"""
        result = GPUSimulationResult(
            simulation_id="test_sim",
            results={'test': 'data'},
            execution_time=1.5,
            memory_peak=1024,
            gpu_utilization=85.0,
            accuracy_metrics={'accuracy': 0.95},
            uncertainty_bounds={'param': (0.1, 0.9)},
            validation_score=0.92
        )
        
        assert result.simulation_id == "test_sim"
        assert result.results == {'test': 'data'}
        assert result.execution_time == 1.5
        assert result.validation_score == 0.92

# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        with patch('modules.gpu_volcanic_modeling_engine.CUDAMagmaPropertyCalculator'), \
             patch('sqlalchemy.create_engine'), \
             patch('redis.Redis'), \
             patch('GPUtil.getGPUs') as mock_gpus:
            
            # Mock no GPUs available to force CPU fallback
            mock_gpus.return_value = []
            
            engine = GPUVolcanicModelingEngine()
            
            # Test magma property calculation
            composition = MagmaComposition(
                sio2=0.65, al2o3=0.15, fe2o3=0.03, feo=0.02, mgo=0.02,
                cao=0.04, na2o=0.04, k2o=0.03, tio2=0.01, p2o5=0.01,
                h2o=0.05, co2=0.01, temperature=1200.0, pressure=100.0
            )
            
            conditions = PhysicalConditions(
                temperature=1200.0, pressure=100.0, oxygen_fugacity=-10.0,
                water_content=0.05, crystal_fraction=0.1, bubble_fraction=0.05
            )
            
            # This should use CPU fallback
            with patch.object(engine, '_cpu_fallback_magma_properties') as mock_fallback:
                mock_result = GPUSimulationResult(
                    simulation_id="test", results={}, execution_time=0.1,
                    memory_peak=1024, gpu_utilization=0.0, accuracy_metrics={},
                    uncertainty_bounds={}, validation_score=0.8
                )
                mock_fallback.return_value = mock_result
                
                result = await engine.simulate_molecular_magma_properties(composition, conditions)
                assert isinstance(result, GPUSimulationResult)
                mock_fallback.assert_called_once()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])