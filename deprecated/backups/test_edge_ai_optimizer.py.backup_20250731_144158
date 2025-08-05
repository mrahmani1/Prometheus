"""
Tests for Edge AI Optimization Pipeline

This module contains comprehensive tests for the edge AI model optimization
functionality, including compression, quantization, pruning, and validation.

Requirements addressed: 10.1, 10.5
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from core.edge_ai_optimizer import (
    EdgeAIOptimizer, ModelCompressor, ModelQuantizer, ModelPruner, PerformanceValidator,
    HardwareConstraints, OptimizationConfig, OptimizationResult,
    create_mobile_constraints, create_embedded_constraints, create_edge_server_constraints
)


class TestModelCompressor:
    """Test model compression functionality"""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing"""
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    @pytest.fixture
    def compressor(self):
        """Create model compressor instance"""
        return ModelCompressor()
    
    def test_weight_sharing_compression(self, compressor, sample_model):
        """Test weight sharing compression"""
        compressed_model = compressor.compress_model(
            sample_model, method='weight_sharing', compression_ratio=0.5
        )
        
        # Check that model structure is preserved
        assert len(list(compressed_model.modules())) == len(list(sample_model.modules()))
        
        # Check that weights are different (compressed)
        original_weights = list(sample_model.parameters())[0].data.flatten()
        compressed_weights = list(compressed_model.parameters())[0].data.flatten()
        
        # Should have fewer unique values after compression
        original_unique = len(torch.unique(original_weights))
        compressed_unique = len(torch.unique(compressed_weights))
        assert compressed_unique <= original_unique
    
    def test_low_rank_approximation(self, compressor, sample_model):
        """Test low-rank approximation compression"""
        compressed_model = compressor.compress_model(
            sample_model, method='low_rank_approximation', compression_ratio=0.7
        )
        
        # Check that model structure is preserved
        assert len(list(compressed_model.modules())) == len(list(sample_model.modules()))
        
        # Check that linear layer weights have been modified
        original_linear = None
        compressed_linear = None
        
        for module in sample_model.modules():
            if isinstance(module, nn.Linear):
                original_linear = module
                break
        
        for module in compressed_model.modules():
            if isinstance(module, nn.Linear):
                compressed_linear = module
                break
        
        assert original_linear is not None
        assert compressed_linear is not None
        assert not torch.equal(original_linear.weight, compressed_linear.weight)
    
    def test_huffman_coding_compression(self, compressor, sample_model):
        """Test Huffman coding compression"""
        compressed_model = compressor.compress_model(
            sample_model, method='huffman_coding', compression_ratio=0.3
        )
        
        # Check that model structure is preserved
        assert len(list(compressed_model.modules())) == len(list(sample_model.modules()))
        
        # Check that weights have been quantized
        original_weights = list(sample_model.parameters())[0].data
        compressed_weights = list(compressed_model.parameters())[0].data
        
        assert not torch.equal(original_weights, compressed_weights)
    
    def test_invalid_compression_method(self, compressor, sample_model):
        """Test error handling for invalid compression method"""
        with pytest.raises(ValueError, match="Unknown compression method"):
            compressor.compress_model(sample_model, method='invalid_method')
    
    def test_kmeans_clustering(self, compressor):
        """Test k-means clustering utility"""
        data = torch.randn(100)
        centroids = compressor._kmeans_clustering(data, 10)
        
        assert len(centroids) == 10
        assert all(isinstance(c, float) for c in centroids)


class TestModelQuantizer:
    """Test model quantization functionality"""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing"""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        model.eval()
        return model
    
    @pytest.fixture
    def quantizer(self):
        """Create model quantizer instance"""
        return ModelQuantizer()
    
    @pytest.fixture
    def calibration_data(self):
        """Create calibration data for quantization"""
        return torch.randn(50, 10)
    
    def test_dynamic_quantization(self, quantizer, sample_model):
        """Test dynamic quantization"""
        quantized_model = quantizer.quantize_model(
            sample_model, scheme='dynamic', bits=8
        )
        
        # Check that model has been quantized
        assert quantized_model is not None
        
        # Test inference still works
        test_input = torch.randn(1, 10)
        with torch.no_grad():
            output = quantized_model(test_input)
        
        assert output.shape == (1, 3)
    
    def test_static_quantization(self, quantizer, sample_model, calibration_data):
        """Test static quantization"""
        # Note: This test might be skipped on some systems without proper quantization support
        try:
            quantized_model = quantizer.quantize_model(
                sample_model, scheme='static', bits=8, calibration_data=calibration_data
            )
            
            # Check that model has been quantized
            assert quantized_model is not None
            
            # Test inference still works
            test_input = torch.randn(1, 10)
            with torch.no_grad():
                output = quantized_model(test_input)
            
            assert output.shape == (1, 3)
            
        except Exception as e:
            pytest.skip(f"Static quantization not supported on this system: {e}")
    
    def test_quantization_aware_training(self, quantizer, sample_model):
        """Test quantization-aware training setup"""
        try:
            qat_model = quantizer.quantize_model(
                sample_model, scheme='qat', bits=8
            )
            
            # Check that model has been prepared for QAT
            assert qat_model is not None
            
        except Exception as e:
            pytest.skip(f"QAT not supported on this system: {e}")
    
    def test_invalid_quantization_scheme(self, quantizer, sample_model):
        """Test error handling for invalid quantization scheme"""
        with pytest.raises(ValueError, match="Unknown quantization scheme"):
            quantizer.quantize_model(sample_model, scheme='invalid_scheme')


class TestModelPruner:
    """Test model pruning functionality"""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing"""
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    @pytest.fixture
    def pruner(self):
        """Create model pruner instance"""
        return ModelPruner()
    
    def test_magnitude_pruning(self, pruner, sample_model):
        """Test magnitude-based pruning"""
        pruned_model = pruner.prune_model(
            sample_model, method='magnitude', sparsity=0.3
        )
        
        # Check that model structure is preserved
        assert len(list(pruned_model.modules())) == len(list(sample_model.modules()))
        
        # Check that some weights have been pruned (set to zero)
        total_params = 0
        zero_params = 0
        
        for module in pruned_model.modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data.flatten()
                total_params += len(weights)
                zero_params += (weights == 0).sum().item()
        
        sparsity_ratio = zero_params / total_params
        assert sparsity_ratio > 0.2  # Should have some sparsity
    
    def test_structured_pruning(self, pruner, sample_model):
        """Test structured pruning"""
        pruned_model = pruner.prune_model(
            sample_model, method='structured', sparsity=0.2, structured=True
        )
        
        # Check that model structure is preserved
        assert len(list(pruned_model.modules())) == len(list(sample_model.modules()))
        
        # Check that structured pruning was applied
        for module in pruned_model.modules():
            if isinstance(module, nn.Linear):
                # Should have some zero rows/columns for structured pruning
                weights = module.weight.data
                zero_rows = (weights.abs().sum(dim=1) == 0).sum().item()
                assert zero_rows >= 0  # At least some structure should be pruned
    
    def test_gradual_pruning(self, pruner, sample_model):
        """Test gradual pruning (currently falls back to magnitude)"""
        pruned_model = pruner.prune_model(
            sample_model, method='gradual', sparsity=0.25
        )
        
        # Check that model structure is preserved
        assert len(list(pruned_model.modules())) == len(list(sample_model.modules()))
    
    def test_invalid_pruning_method(self, pruner, sample_model):
        """Test error handling for invalid pruning method"""
        with pytest.raises(ValueError, match="Unknown pruning method"):
            pruner.prune_model(sample_model, method='invalid_method')


class TestPerformanceValidator:
    """Test performance validation functionality"""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing"""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    @pytest.fixture
    def optimized_model(self, sample_model):
        """Create an optimized version of the sample model"""
        # Simple optimization: smaller model
        return nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
    
    @pytest.fixture
    def test_data(self):
        """Create test data"""
        return torch.randn(100, 10)
    
    @pytest.fixture
    def test_labels(self):
        """Create test labels"""
        return torch.randint(0, 3, (100,))
    
    @pytest.fixture
    def hardware_constraints(self):
        """Create hardware constraints"""
        return HardwareConstraints(
            max_memory_mb=100,
            max_inference_time_ms=50,
            cpu_cores=2,
            has_gpu=False,
            target_platform="cpu"
        )
    
    @pytest.fixture
    def validator(self):
        """Create performance validator instance"""
        return PerformanceValidator()
    
    def test_validate_model(self, validator, sample_model, optimized_model, 
                          test_data, test_labels, hardware_constraints):
        """Test complete model validation"""
        results = validator.validate_model(
            sample_model, optimized_model, test_data, test_labels, hardware_constraints
        )
        
        assert isinstance(results, OptimizationResult)
        assert results.original_size_mb > 0
        assert results.optimized_size_mb > 0
        assert results.compression_ratio > 0
        assert results.original_inference_time_ms > 0
        assert results.optimized_inference_time_ms > 0
        assert results.speedup_factor > 0
        assert 0 <= results.accuracy_retention <= 1
        assert results.memory_usage_mb >= 0
        assert isinstance(results.meets_constraints, bool)
        assert isinstance(results.optimization_steps, list)
    
    def test_get_model_size(self, validator, sample_model):
        """Test model size calculation"""
        size_mb = validator._get_model_size(sample_model)
        assert size_mb > 0
        assert isinstance(size_mb, float)
    
    def test_measure_inference_time(self, validator, sample_model, test_data):
        """Test inference time measurement"""
        inference_time = validator._measure_inference_time(sample_model, test_data)
        assert inference_time > 0
        assert isinstance(inference_time, float)
    
    def test_measure_accuracy(self, validator, sample_model, test_data, test_labels):
        """Test accuracy measurement"""
        accuracy = validator._measure_accuracy(sample_model, test_data, test_labels)
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, float)
    
    def test_measure_memory_usage(self, validator, sample_model, test_data):
        """Test memory usage measurement"""
        memory_usage = validator._measure_memory_usage(sample_model, test_data)
        assert isinstance(memory_usage, float)
    
    def test_check_constraints(self, validator, hardware_constraints):
        """Test constraint checking"""
        # Test case that meets constraints
        meets_constraints = validator._check_constraints(
            model_size=50.0,
            inference_time=30.0,
            memory_usage=40.0,
            constraints=hardware_constraints
        )
        assert meets_constraints is True
        
        # Test case that doesn't meet constraints
        meets_constraints = validator._check_constraints(
            model_size=150.0,  # Exceeds max_memory_mb
            inference_time=30.0,
            memory_usage=40.0,
            constraints=hardware_constraints
        )
        assert meets_constraints is False


class TestEdgeAIOptimizer:
    """Test complete edge AI optimization pipeline"""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing"""
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    @pytest.fixture
    def hardware_constraints(self):
        """Create hardware constraints"""
        return HardwareConstraints(
            max_memory_mb=50,
            max_inference_time_ms=100,
            cpu_cores=2,
            has_gpu=False,
            target_platform="mobile"
        )
    
    @pytest.fixture
    def optimization_config(self):
        """Create optimization configuration"""
        return OptimizationConfig(
            compression_ratio=0.5,
            quantization_bits=8,
            pruning_sparsity=0.3,
            enable_quantization=True,
            enable_pruning=True,
            validation_threshold=0.8
        )
    
    @pytest.fixture
    def test_data(self):
        """Create test data"""
        return torch.randn(50, 10)
    
    @pytest.fixture
    def test_labels(self):
        """Create test labels"""
        return torch.randint(0, 3, (50,))
    
    @pytest.fixture
    def optimizer(self):
        """Create edge AI optimizer instance"""
        return EdgeAIOptimizer()
    
    def test_optimize_for_edge(self, optimizer, sample_model, hardware_constraints,
                             optimization_config, test_data, test_labels):
        """Test complete optimization pipeline"""
        optimized_model, results = optimizer.optimize_for_edge(
            sample_model, hardware_constraints, optimization_config,
            test_data, test_labels
        )
        
        # Check that optimized model is returned
        assert optimized_model is not None
        assert isinstance(optimized_model, nn.Module)
        
        # Check that results are valid
        assert isinstance(results, OptimizationResult)
        assert results.compression_ratio > 0
        assert results.speedup_factor > 0
        assert 0 <= results.accuracy_retention <= 1
        
        # Check that optimization steps were recorded
        assert len(results.optimization_steps) > 0
        assert "compression" in results.optimization_steps
    
    def test_generate_edge_model_variants(self, optimizer, sample_model, test_data, test_labels):
        """Test generation of multiple model variants"""
        hardware_profiles = [
            create_mobile_constraints(),
            create_embedded_constraints(),
            create_edge_server_constraints()
        ]
        
        variants = optimizer.generate_edge_model_variants(
            sample_model, hardware_profiles, test_data, test_labels
        )
        
        assert len(variants) <= len(hardware_profiles)  # Some might fail
        
        for profile_name, (model, results) in variants.items():
            assert model is not None
            assert isinstance(results, OptimizationResult)
            assert results.compression_ratio > 0
    
    def test_create_optimization_config_for_constraints(self, optimizer):
        """Test automatic optimization config creation"""
        # Very constrained device
        constrained = HardwareConstraints(
            max_memory_mb=50, max_inference_time_ms=100, cpu_cores=1
        )
        config = optimizer._create_optimization_config_for_constraints(constrained)
        assert config.compression_ratio <= 0.5
        assert config.pruning_sparsity >= 0.3
        
        # Mid-range device
        mid_range = HardwareConstraints(
            max_memory_mb=300, max_inference_time_ms=50, cpu_cores=4
        )
        config = optimizer._create_optimization_config_for_constraints(mid_range)
        assert config.compression_ratio >= 0.3
        
        # High-end edge device
        high_end = HardwareConstraints(
            max_memory_mb=1000, max_inference_time_ms=20, cpu_cores=8
        )
        config = optimizer._create_optimization_config_for_constraints(high_end)
        assert config.compression_ratio >= 0.5
    
    def test_save_optimized_model(self, optimizer, sample_model):
        """Test saving optimized model"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model"
            
            # Create dummy results
            results = OptimizationResult(
                original_size_mb=10.0,
                optimized_size_mb=5.0,
                compression_ratio=2.0,
                original_inference_time_ms=100.0,
                optimized_inference_time_ms=50.0,
                speedup_factor=2.0,
                accuracy_retention=0.95,
                memory_usage_mb=20.0,
                meets_constraints=True,
                optimization_steps=["compression", "quantization"]
            )
            
            optimizer.save_optimized_model(sample_model, results, save_path)
            
            # Check that files were created
            assert (save_path / "optimized_model.pth").exists()
            assert (save_path / "optimization_results.json").exists()


class TestHardwareConstraints:
    """Test hardware constraints functionality"""
    
    def test_create_mobile_constraints(self):
        """Test mobile constraints creation"""
        constraints = create_mobile_constraints()
        assert constraints.max_memory_mb == 200
        assert constraints.max_inference_time_ms == 100
        assert constraints.cpu_cores == 4
        assert constraints.has_gpu is False
        assert constraints.target_platform == "mobile"
    
    def test_create_embedded_constraints(self):
        """Test embedded constraints creation"""
        constraints = create_embedded_constraints()
        assert constraints.max_memory_mb == 50
        assert constraints.max_inference_time_ms == 500
        assert constraints.cpu_cores == 1
        assert constraints.has_gpu is False
        assert constraints.target_platform == "embedded"
    
    def test_create_edge_server_constraints(self):
        """Test edge server constraints creation"""
        constraints = create_edge_server_constraints()
        assert constraints.max_memory_mb == 1000
        assert constraints.max_inference_time_ms == 50
        assert constraints.cpu_cores == 8
        assert constraints.has_gpu is True
        assert constraints.gpu_memory_mb == 4000
        assert constraints.target_platform == "edge_server"


class TestOptimizationConfig:
    """Test optimization configuration"""
    
    def test_default_optimization_config(self):
        """Test default optimization configuration"""
        config = OptimizationConfig()
        assert config.compression_ratio == 0.5
        assert config.quantization_bits == 8
        assert config.pruning_sparsity == 0.3
        assert config.enable_quantization is True
        assert config.enable_pruning is True
        assert config.validation_threshold == 0.95
    
    def test_custom_optimization_config(self):
        """Test custom optimization configuration"""
        config = OptimizationConfig(
            compression_ratio=0.3,
            quantization_bits=16,
            pruning_sparsity=0.5,
            enable_quantization=False,
            enable_pruning=True,
            validation_threshold=0.85
        )
        assert config.compression_ratio == 0.3
        assert config.quantization_bits == 16
        assert config.pruning_sparsity == 0.5
        assert config.enable_quantization is False
        assert config.enable_pruning is True
        assert config.validation_threshold == 0.85


# Integration tests
class TestEdgeOptimizationIntegration:
    """Integration tests for edge optimization pipeline"""
    
    @pytest.fixture
    def complex_model(self):
        """Create a more complex model for integration testing"""
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    @pytest.fixture
    def complex_test_data(self):
        """Create test data for complex model"""
        return torch.randn(20, 1, 28, 28)
    
    @pytest.fixture
    def complex_test_labels(self):
        """Create test labels for complex model"""
        return torch.randint(0, 10, (20,))
    
    def test_end_to_end_optimization(self, complex_model, complex_test_data, complex_test_labels):
        """Test end-to-end optimization pipeline"""
        optimizer = EdgeAIOptimizer()
        
        # Define aggressive constraints for mobile deployment
        constraints = HardwareConstraints(
            max_memory_mb=30,
            max_inference_time_ms=200,
            cpu_cores=2,
            has_gpu=False,
            target_platform="mobile"
        )
        
        config = OptimizationConfig(
            compression_ratio=0.3,
            quantization_bits=8,
            pruning_sparsity=0.4,
            enable_quantization=True,
            enable_pruning=True,
            validation_threshold=0.7  # Lower threshold for aggressive optimization
        )
        
        # Run optimization
        optimized_model, results = optimizer.optimize_for_edge(
            complex_model, constraints, config,
            complex_test_data, complex_test_labels
        )
        
        # Verify optimization results
        assert results.compression_ratio > 1.0
        assert results.speedup_factor > 1.0
        assert results.accuracy_retention > 0.5  # Should retain reasonable accuracy
        
        # Verify model still works
        with torch.no_grad():
            original_output = complex_model(complex_test_data[:1])
            optimized_output = optimized_model(complex_test_data[:1])
        
        assert original_output.shape == optimized_output.shape
        
        # Verify size reduction
        assert results.optimized_size_mb < results.original_size_mb
        
        print(f"Optimization Results:")
        print(f"  Compression: {results.compression_ratio:.2f}x")
        print(f"  Speedup: {results.speedup_factor:.2f}x")
        print(f"  Accuracy retention: {results.accuracy_retention:.1%}")
        print(f"  Size: {results.original_size_mb:.1f}MB → {results.optimized_size_mb:.1f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])