"""
Tests for Automated Edge Model Generation Service

This module contains tests for the automated edge model generation functionality,
including model registry, deployment profiles, and automated optimization.

Requirements addressed: 10.1, 10.5
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import asyncio
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from core.automated_edge_generation import (
    AutomatedEdgeGenerator, EdgeModelRegistry, EdgeDeploymentProfile,
    EdgeModelMetadata, HeliosEdgeModelFactory
)
from core.edge_ai_optimizer import (
    HardwareConstraints, OptimizationConfig, OptimizationResult
)
from core.helios_engine import HeliosCore


class TestEdgeModelRegistry:
    """Test edge model registry functionality"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def registry(self, temp_storage):
        """Create edge model registry instance"""
        return EdgeModelRegistry(temp_storage)
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample edge model metadata"""
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
        
        return EdgeModelMetadata(
            model_id="test_model_001",
            source_model_id="helios_pinn_v1",
            deployment_profile="mobile",
            creation_timestamp=datetime.now(),
            optimization_results=results,
            model_hash="abc123def456",
            validation_status="validated",
            deployment_ready=True
        )
    
    def test_register_model(self, registry, sample_metadata):
        """Test model registration"""
        registry.register_model(sample_metadata)
        
        assert sample_metadata.model_id in registry.models
        assert registry.models[sample_metadata.model_id] == sample_metadata
    
    def test_get_model(self, registry, sample_metadata):
        """Test model retrieval"""
        registry.register_model(sample_metadata)
        
        retrieved = registry.get_model(sample_metadata.model_id)
        assert retrieved == sample_metadata
        
        # Test non-existent model
        assert registry.get_model("non_existent") is None
    
    def test_list_models_by_profile(self, registry, sample_metadata):
        """Test listing models by deployment profile"""
        # Create multiple models with different profiles
        mobile_metadata = sample_metadata
        
        embedded_metadata = EdgeModelMetadata(
            model_id="test_model_002",
            source_model_id="helios_pinn_v1",
            deployment_profile="embedded",
            creation_timestamp=datetime.now(),
            optimization_results=sample_metadata.optimization_results,
            model_hash="def456ghi789",
            validation_status="validated",
            deployment_ready=True
        )
        
        registry.register_model(mobile_metadata)
        registry.register_model(embedded_metadata)
        
        # Test filtering by profile
        mobile_models = registry.list_models_by_profile("mobile")
        assert len(mobile_models) == 1
        assert mobile_models[0].model_id == "test_model_001"
        
        embedded_models = registry.list_models_by_profile("embedded")
        assert len(embedded_models) == 1
        assert embedded_models[0].model_id == "test_model_002"
        
        # Test non-existent profile
        assert len(registry.list_models_by_profile("non_existent")) == 0
    
    def test_get_best_model_for_constraints(self, registry, sample_metadata):
        """Test finding best model for constraints"""
        registry.register_model(sample_metadata)
        
        # Constraints that the model meets
        compatible_constraints = HardwareConstraints(
            max_memory_mb=100,
            max_inference_time_ms=200,
            cpu_cores=2
        )
        
        best_model = registry.get_best_model_for_constraints(compatible_constraints)
        assert best_model == sample_metadata
        
        # Constraints that the model doesn't meet
        incompatible_constraints = HardwareConstraints(
            max_memory_mb=1,  # Too small
            max_inference_time_ms=10,  # Too fast
            cpu_cores=1
        )
        
        best_model = registry.get_best_model_for_constraints(incompatible_constraints)
        assert best_model is None
    
    def test_registry_persistence(self, temp_storage, sample_metadata):
        """Test registry persistence to disk"""
        # Create registry and add model
        registry1 = EdgeModelRegistry(temp_storage)
        registry1.register_model(sample_metadata)
        
        # Create new registry instance (should load from disk)
        registry2 = EdgeModelRegistry(temp_storage)
        
        assert sample_metadata.model_id in registry2.models
        retrieved = registry2.get_model(sample_metadata.model_id)
        assert retrieved.model_id == sample_metadata.model_id
        assert retrieved.source_model_id == sample_metadata.source_model_id
        assert retrieved.deployment_profile == sample_metadata.deployment_profile


class TestAutomatedEdgeGenerator:
    """Test automated edge model generation"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_helios_core(self):
        """Create mock Helios core"""
        mock_core = Mock(spec=HeliosCore)
        mock_core.pinn_engine = Mock()
        mock_core.pinn_engine.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        return mock_core
    
    @pytest.fixture
    def generator(self, mock_helios_core, temp_storage):
        """Create automated edge generator instance"""
        return AutomatedEdgeGenerator(mock_helios_core, temp_storage)
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model for testing"""
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    def test_create_default_profiles(self, generator):
        """Test creation of default deployment profiles"""
        profiles = generator.deployment_profiles
        
        assert "mobile" in profiles
        assert "embedded" in profiles
        assert "edge_server" in profiles
        assert "field_station" in profiles
        
        # Check mobile profile
        mobile = profiles["mobile"]
        assert mobile.name == "mobile"
        assert mobile.hardware_constraints.max_memory_mb == 200
        assert "field_monitoring" in mobile.target_use_cases
        
        # Check embedded profile
        embedded = profiles["embedded"]
        assert embedded.name == "embedded"
        assert embedded.hardware_constraints.max_memory_mb == 50
        assert "sensor_nodes" in embedded.target_use_cases
    
    @pytest.mark.asyncio
    async def test_generate_all_edge_models(self, generator, sample_model):
        """Test generation of all edge model variants"""
        with patch.object(generator.optimizer, 'optimize_for_edge') as mock_optimize:
            # Mock optimization results
            mock_results = OptimizationResult(
                original_size_mb=10.0,
                optimized_size_mb=5.0,
                compression_ratio=2.0,
                original_inference_time_ms=100.0,
                optimized_inference_time_ms=50.0,
                speedup_factor=2.0,
                accuracy_retention=0.95,
                memory_usage_mb=20.0,
                meets_constraints=True,
                optimization_steps=["compression"]
            )
            
            mock_optimize.return_value = (sample_model, mock_results)
            
            # Generate models
            generated_models = await generator.generate_all_edge_models(
                sample_model, "test_source_model"
            )
            
            # Check that models were generated for all profiles
            assert len(generated_models) == len(generator.deployment_profiles)
            
            for profile_name, metadata in generated_models.items():
                assert isinstance(metadata, EdgeModelMetadata)
                assert metadata.source_model_id == "test_source_model"
                assert metadata.deployment_profile == profile_name
                assert metadata.deployment_ready is True
    
    @pytest.mark.asyncio
    async def test_generate_custom_edge_model(self, generator, sample_model):
        """Test generation of custom edge model"""
        with patch.object(generator.optimizer, 'optimize_for_edge') as mock_optimize:
            # Mock optimization results
            mock_results = OptimizationResult(
                original_size_mb=10.0,
                optimized_size_mb=3.0,
                compression_ratio=3.33,
                original_inference_time_ms=100.0,
                optimized_inference_time_ms=30.0,
                speedup_factor=3.33,
                accuracy_retention=0.90,
                memory_usage_mb=15.0,
                meets_constraints=True,
                optimization_steps=["compression", "quantization", "pruning"]
            )
            
            mock_optimize.return_value = (sample_model, mock_results)
            
            # Custom constraints and config
            custom_constraints = HardwareConstraints(
                max_memory_mb=25,
                max_inference_time_ms=50,
                cpu_cores=1,
                target_platform="custom"
            )
            
            custom_config = OptimizationConfig(
                compression_ratio=0.2,
                quantization_bits=8,
                pruning_sparsity=0.5,
                validation_threshold=0.85
            )
            
            # Generate custom model
            metadata = await generator.generate_custom_edge_model(
                sample_model, "test_source", custom_constraints, custom_config, "custom_profile"
            )
            
            assert isinstance(metadata, EdgeModelMetadata)
            assert metadata.source_model_id == "test_source"
            assert metadata.deployment_profile == "custom_profile"
            assert metadata.deployment_ready is True
    
    def test_get_recommended_model(self, generator):
        """Test model recommendation functionality"""
        # Add a test model to registry
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
            optimization_steps=["compression"]
        )
        
        metadata = EdgeModelMetadata(
            model_id="test_model_001",
            source_model_id="helios_pinn_v1",
            deployment_profile="mobile",
            creation_timestamp=datetime.now(),
            optimization_results=results,
            model_hash="abc123",
            validation_status="validated",
            deployment_ready=True
        )
        
        generator.registry.register_model(metadata)
        
        # Test recommendation for supported use case
        recommended = generator.get_recommended_model("field_monitoring")
        assert recommended == metadata
        
        # Test recommendation for unsupported use case
        recommended = generator.get_recommended_model("unsupported_use_case")
        assert recommended is None
        
        # Test recommendation with constraints
        constraints = HardwareConstraints(
            max_memory_mb=100,
            max_inference_time_ms=200,
            cpu_cores=2
        )
        
        recommended = generator.get_recommended_model("field_monitoring", constraints)
        assert recommended == metadata
        
        # Test recommendation with incompatible constraints
        tight_constraints = HardwareConstraints(
            max_memory_mb=1,
            max_inference_time_ms=1,
            cpu_cores=1
        )
        
        recommended = generator.get_recommended_model("field_monitoring", tight_constraints)
        assert recommended is None
    
    def test_update_deployment_profile(self, generator):
        """Test updating deployment profiles"""
        new_profile = EdgeDeploymentProfile(
            name="test_profile",
            description="Test profile",
            hardware_constraints=HardwareConstraints(
                max_memory_mb=100,
                max_inference_time_ms=100,
                cpu_cores=2
            ),
            optimization_config=OptimizationConfig(),
            target_use_cases=["testing"],
            priority=1
        )
        
        generator.update_deployment_profile("test_profile", new_profile)
        
        assert "test_profile" in generator.deployment_profiles
        assert generator.deployment_profiles["test_profile"] == new_profile
    
    def test_get_optimization_report(self, generator):
        """Test optimization report generation"""
        # Add a test model to registry
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
        
        metadata = EdgeModelMetadata(
            model_id="test_model_001",
            source_model_id="helios_pinn_v1",
            deployment_profile="mobile",
            creation_timestamp=datetime.now(),
            optimization_results=results,
            model_hash="abc123",
            validation_status="validated",
            deployment_ready=True
        )
        
        generator.registry.register_model(metadata)
        
        # Get optimization report
        report = generator.get_optimization_report("test_model_001")
        
        assert report is not None
        assert report["model_id"] == "test_model_001"
        assert report["source_model_id"] == "helios_pinn_v1"
        assert report["deployment_profile"] == "mobile"
        assert "optimization_summary" in report
        assert "compression_ratio" in report["optimization_summary"]
        assert report["optimization_steps"] == ["compression", "quantization"]
        assert report["meets_constraints"] is True
        assert report["deployment_ready"] is True
        
        # Test non-existent model
        report = generator.get_optimization_report("non_existent")
        assert report is None
    
    def test_generate_model_id(self, generator):
        """Test model ID generation"""
        model_id = generator._generate_model_id("source_model", "mobile")
        
        assert "source_model_edge_mobile" in model_id
        assert len(model_id.split("_")) >= 4  # Should include timestamp
    
    def test_calculate_model_hash(self, generator, sample_model):
        """Test model hash calculation"""
        hash1 = generator._calculate_model_hash(sample_model)
        hash2 = generator._calculate_model_hash(sample_model)
        
        # Same model should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
        
        # Different model should produce different hash
        different_model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
        
        hash3 = generator._calculate_model_hash(different_model)
        assert hash1 != hash3


class TestHeliosEdgeModelFactory:
    """Test Helios edge model factory"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_helios_core(self):
        """Create mock Helios core"""
        mock_core = Mock(spec=HeliosCore)
        
        # Mock PINN engine
        mock_core.pinn_engine = Mock()
        mock_core.pinn_engine.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        # Mock GAN generator
        mock_core.gan_generator = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )
        
        return mock_core
    
    @pytest.fixture
    def factory(self, mock_helios_core, temp_storage):
        """Create Helios edge model factory instance"""
        return HeliosEdgeModelFactory(mock_helios_core, temp_storage)
    
    @pytest.mark.asyncio
    async def test_create_edge_pinn_models(self, factory):
        """Test creation of edge PINN models"""
        with patch.object(factory.edge_generator, 'generate_all_edge_models') as mock_generate:
            mock_generate.return_value = {"mobile": Mock(), "embedded": Mock()}
            
            models = await factory.create_edge_pinn_models()
            
            assert len(models) == 2
            assert "mobile" in models
            assert "embedded" in models
            
            # Verify that the PINN model was used
            mock_generate.assert_called_once()
            args = mock_generate.call_args[0]
            assert args[1] == "helios_pinn_v1"  # model_id
    
    @pytest.mark.asyncio
    async def test_create_edge_seismic_cnn_models(self, factory):
        """Test creation of edge seismic CNN models"""
        with patch.object(factory.edge_generator, 'generate_all_edge_models') as mock_generate:
            with patch('services.helios_core.core.automated_edge_generation.SeismicCNNProcessor') as mock_processor_class:
                # Mock the SeismicCNNProcessor
                mock_processor = Mock()
                mock_processor.model = nn.Sequential(
                    nn.Conv2d(1, 32, 3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(32, 3)
                )
                mock_processor_class.return_value = mock_processor
                
                mock_generate.return_value = {"mobile": Mock(), "embedded": Mock()}
                
                models = await factory.create_edge_seismic_cnn_models()
                
                assert len(models) == 2
                mock_generate.assert_called_once()
                args = mock_generate.call_args[0]
                assert args[1] == "helios_seismic_cnn_v1"  # model_id
    
    @pytest.mark.asyncio
    async def test_create_edge_gan_models(self, factory):
        """Test creation of edge GAN models"""
        with patch.object(factory.edge_generator, 'generate_all_edge_models') as mock_generate:
            mock_generate.return_value = {"mobile": Mock(), "embedded": Mock()}
            
            models = await factory.create_edge_gan_models()
            
            assert len(models) == 2
            mock_generate.assert_called_once()
            args = mock_generate.call_args[0]
            assert args[1] == "helios_gan_generator_v1"  # model_id
    
    def test_get_edge_model_for_deployment(self, factory):
        """Test getting edge model for deployment"""
        # Mock the edge generator's get_recommended_model method
        mock_metadata = Mock()
        mock_metadata.model_id = "test_model_001"
        
        with patch.object(factory.edge_generator, 'get_recommended_model') as mock_recommend:
            with patch('torch.load') as mock_load:
                mock_recommend.return_value = mock_metadata
                mock_load.return_value = nn.Linear(10, 3)
                
                constraints = HardwareConstraints(
                    max_memory_mb=100,
                    max_inference_time_ms=50,
                    cpu_cores=2
                )
                
                result = factory.get_edge_model_for_deployment("field_monitoring", constraints)
                
                assert result is not None
                model, metadata = result
                assert model is not None
                assert metadata == mock_metadata
                
                mock_recommend.assert_called_once_with("field_monitoring", constraints)
        
        # Test case where no model is found
        with patch.object(factory.edge_generator, 'get_recommended_model') as mock_recommend:
            mock_recommend.return_value = None
            
            result = factory.get_edge_model_for_deployment("unsupported_use_case", constraints)
            assert result is None


class TestEdgeDeploymentProfile:
    """Test edge deployment profile functionality"""
    
    def test_edge_deployment_profile_creation(self):
        """Test creation of edge deployment profile"""
        constraints = HardwareConstraints(
            max_memory_mb=100,
            max_inference_time_ms=50,
            cpu_cores=2
        )
        
        config = OptimizationConfig(
            compression_ratio=0.5,
            quantization_bits=8,
            pruning_sparsity=0.3
        )
        
        profile = EdgeDeploymentProfile(
            name="test_profile",
            description="Test deployment profile",
            hardware_constraints=constraints,
            optimization_config=config,
            target_use_cases=["testing", "validation"],
            priority=1
        )
        
        assert profile.name == "test_profile"
        assert profile.description == "Test deployment profile"
        assert profile.hardware_constraints == constraints
        assert profile.optimization_config == config
        assert profile.target_use_cases == ["testing", "validation"]
        assert profile.priority == 1


class TestEdgeModelMetadata:
    """Test edge model metadata functionality"""
    
    def test_edge_model_metadata_creation(self):
        """Test creation of edge model metadata"""
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
            optimization_steps=["compression"]
        )
        
        timestamp = datetime.now()
        
        metadata = EdgeModelMetadata(
            model_id="test_model_001",
            source_model_id="helios_pinn_v1",
            deployment_profile="mobile",
            creation_timestamp=timestamp,
            optimization_results=results,
            model_hash="abc123def456",
            validation_status="validated",
            deployment_ready=True
        )
        
        assert metadata.model_id == "test_model_001"
        assert metadata.source_model_id == "helios_pinn_v1"
        assert metadata.deployment_profile == "mobile"
        assert metadata.creation_timestamp == timestamp
        assert metadata.optimization_results == results
        assert metadata.model_hash == "abc123def456"
        assert metadata.validation_status == "validated"
        assert metadata.deployment_ready is True


# Integration tests
class TestEdgeGenerationIntegration:
    """Integration tests for edge model generation"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_helios_core(self):
        """Create mock Helios core"""
        mock_core = Mock(spec=HeliosCore)
        mock_core.pinn_engine = Mock()
        mock_core.pinn_engine.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        return mock_core
    
    @pytest.mark.asyncio
    async def test_full_edge_generation_workflow(self, mock_helios_core, temp_storage):
        """Test complete edge model generation workflow"""
        # Create generator
        generator = AutomatedEdgeGenerator(mock_helios_core, temp_storage)
        
        # Create test model
        test_model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        # Generate edge models (this will use real optimization)
        generated_models = await generator.generate_all_edge_models(
            test_model, "test_source_model"
        )
        
        # Verify that models were generated
        assert len(generated_models) > 0
        
        # Check that models are registered
        for profile_name, metadata in generated_models.items():
            assert metadata.model_id in generator.registry.models
            
            # Check that model files exist
            model_path = generator.registry.storage_path / metadata.model_id
            assert model_path.exists()
            assert (model_path / "optimized_model.pth").exists()
            assert (model_path / "optimization_results.json").exists()
        
        # Test model recommendation
        recommended = generator.get_recommended_model("field_monitoring")
        if recommended:  # Some models might not meet constraints
            assert recommended.deployment_ready
            assert "field_monitoring" in generator.deployment_profiles[recommended.deployment_profile].target_use_cases
        
        # Test optimization report
        if generated_models:
            first_model_id = list(generated_models.keys())[0]
            metadata = generated_models[first_model_id]
            report = generator.get_optimization_report(metadata.model_id)
            
            assert report is not None
            assert "optimization_summary" in report
            assert "compression_ratio" in report["optimization_summary"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])