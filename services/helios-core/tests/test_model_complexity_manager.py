"""
Tests for Model Complexity Management System
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model_complexity_manager import (
    ComplexityLevel,
    ComplexityMetrics,
    ComputationalCostEstimator,
    ComputationalResource,
    DataAssessment,
    DataAvailability,
    DataAvailabilityAssessor,
    ModelComplexityAssessor,
    ModelComplexityManager,
    ModelConfiguration,
    ModelHierarchy,
    ResourceAssessment,
    ResourceAssessor,
)
from models.ai_models import GeologicalProcess

class TestComplexityMetrics:
    """Test ComplexityMetrics class"""

    def test_complexity_score_calculation(self):
        """Test complexity score calculation"""
        metrics = ComplexityMetrics(
            parameter_count=1000000,
            computational_flops=1e12,
            memory_requirement=10.0,
            training_time_estimate=5.0,
            inference_time_estimate=0.1,
            accuracy_score=0.85,
            physics_constraint_count=3,
            coupling_terms=2,
        )

        score = metrics.complexity_score()
        assert isinstance(score, float)
        assert score > 0

class TestDataAssessment:
    """Test DataAssessment class"""

    def test_availability_level_sparse(self):
        """Test sparse data availability level"""
        assessment = DataAssessment(
            total_data_points=100,
            data_quality_score=0.5,
            temporal_coverage=0.1,
            spatial_coverage=0.1,
            data_types_available=["seismic"],
            missing_data_fraction=0.3,
            noise_level=0.4,
        )

        assert assessment.availability_level() == DataAvailability.SPARSE

    def test_availability_level_excellent(self):
        """Test excellent data availability level"""
        assessment = DataAssessment(
            total_data_points=100000,
            data_quality_score=0.95,
            temporal_coverage=0.95,
            spatial_coverage=0.95,
            data_types_available=["seismic", "deformation", "thermal"],
            missing_data_fraction=0.02,
            noise_level=0.1,
        )

        assert assessment.availability_level() == DataAvailability.EXCELLENT

class TestResourceAssessment:
    """Test ResourceAssessment class"""

    def test_resource_level_minimal(self):
        """Test minimal resource level"""
        assessment = ResourceAssessment(
            cpu_cores=2,
            memory_gb=2.0,
            gpu_count=0,
            gpu_memory_gb=0.0,
            storage_gb=50.0,
            network_bandwidth_mbps=100.0,
        )

        assert assessment.resource_level() == ComputationalResource.MINIMAL

    def test_resource_level_extreme(self):
        """Test extreme resource level"""
        assessment = ResourceAssessment(
            cpu_cores=64,
            memory_gb=512.0,
            gpu_count=8,
            gpu_memory_gb=320.0,
            storage_gb=10000.0,
            network_bandwidth_mbps=10000.0,
        )

        assert assessment.resource_level() == ComputationalResource.EXTREME

class TestModelConfiguration:
    """Test ModelConfiguration class"""

    def test_to_dict_conversion(self):
        """Test conversion to dictionary"""
        config = ModelConfiguration(
            complexity_level=ComplexityLevel.MODERATE,
            network_layers=[128, 64, 32],
            physics_constraints=["mass_conservation", "energy_conservation"],
            coupling_terms=["thermal_mechanical"],
            time_stepping_method="runge_kutta_4",
            spatial_resolution=250.0,
            temporal_resolution=900.0,
            batch_size=128,
            training_epochs=200,
            convergence_tolerance=1e-5,
        )

        config_dict = config.to_dict()

        assert config_dict["complexity_level"] == "moderate"
        assert config_dict["network_layers"] == [128, 64, 32]
        assert len(config_dict["physics_constraints"]) == 2
        assert config_dict["batch_size"] == 128

class TestModelComplexityAssessor:
    """Test ModelComplexityAssessor class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.assessor = ModelComplexityAssessor()

        # Create a simple test model
        self.test_model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def test_assess_model_complexity(self):
        """Test model complexity assessment"""
        input_shape = (1000, 10)

        metrics = self.assessor.assess_model_complexity(self.test_model, input_shape)

        assert isinstance(metrics, ComplexityMetrics)
        assert metrics.parameter_count > 0
        assert metrics.computational_flops > 0
        assert metrics.memory_requirement > 0
        assert metrics.training_time_estimate > 0
        assert metrics.inference_time_estimate > 0

    def test_flops_estimation(self):
        """Test FLOPs estimation"""
        input_shape = (100, 10)
        flops = self.assessor._estimate_flops(self.test_model, input_shape)

        assert flops > 0
        assert isinstance(flops, float)

    def test_memory_estimation(self):
        """Test memory estimation"""
        input_shape = (100, 10)
        memory = self.assessor._estimate_memory(self.test_model, input_shape)

        assert memory > 0
        assert isinstance(memory, float)

class TestDataAvailabilityAssessor:
    """Test DataAvailabilityAssessor class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.assessor = DataAvailabilityAssessor()

    def test_assess_data_availability(self):
        """Test data availability assessment"""
        # Create test data
        data = {
            "seismic": torch.randn(1000, 10),
            "deformation": torch.randn(800, 5),
            "thermal": torch.randn(500, 3),
        }

        assessment = self.assessor.assess_data_availability(data)

        assert isinstance(assessment, DataAssessment)
        assert assessment.total_data_points == 2300
        assert 0 <= assessment.data_quality_score <= 1
        assert 0 <= assessment.temporal_coverage <= 1
        assert 0 <= assessment.spatial_coverage <= 1
        assert len(assessment.data_types_available) == 3

    def test_assess_data_quality_with_nans(self):
        """Test data quality assessment with NaN values"""
        # Create data with NaN values
        data_with_nans = torch.randn(100, 10)
        data_with_nans[10:20, :] = float("nan")

        data = {"test_data": data_with_nans}

        quality_score = self.assessor._assess_data_quality(data)

        assert quality_score < 1.0  # Should be reduced due to NaN values
        assert quality_score >= 0.0

    def test_assess_missing_data(self):
        """Test missing data assessment"""
        # Create data with some NaN values
        data_tensor = torch.randn(100, 10)
        data_tensor[0:10, :] = float("nan")  # 10% missing

        data = {"test_data": data_tensor}

        missing_fraction = self.assessor._assess_missing_data(data)

        assert abs(missing_fraction - 0.1) < 0.01  # Should be approximately 10%

class TestResourceAssessor:
    """Test ResourceAssessor class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.assessor = ResourceAssessor()

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    def test_assess_resources(
        self,
        mock_device_count,
        mock_cuda_available,
        mock_disk_usage,
        mock_virtual_memory,
        mock_cpu_count,
    ):
        """Test resource assessment with mocked system info"""
        # Mock system information
        mock_cpu_count.return_value = 8
        mock_virtual_memory.return_value = Mock(total=16 * 1024**3)  # 16GB
        mock_disk_usage.return_value = Mock(total=1000 * 1024**3)  # 1TB
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2

        # Mock GPU properties
        with patch("torch.cuda.get_device_properties") as mock_gpu_props:
            mock_gpu_props.return_value = Mock(total_memory=8 * 1024**3)  # 8GB per GPU

            assessment = self.assessor.assess_resources()

            assert isinstance(assessment, ResourceAssessment)
            assert assessment.cpu_cores == 8
            assert assessment.memory_gb == 16.0
            assert assessment.gpu_count == 2
            assert assessment.gpu_memory_gb == 16.0  # 2 GPUs * 8GB each

class TestModelHierarchy:
    """Test ModelHierarchy class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.hierarchy = ModelHierarchy()

    def test_initialization(self):
        """Test hierarchy initialization"""
        configs = self.hierarchy.get_all_configs()

        assert len(configs) == 5  # All complexity levels
        assert ComplexityLevel.MINIMAL in configs
        assert ComplexityLevel.FULL in configs

    def test_get_config(self):
        """Test getting specific configuration"""
        config = self.hierarchy.get_config(ComplexityLevel.MODERATE)

        assert isinstance(config, ModelConfiguration)
        assert config.complexity_level == ComplexityLevel.MODERATE
        assert len(config.network_layers) > 0
        assert len(config.physics_constraints) > 0

    def test_complexity_progression(self):
        """Test that complexity increases across levels"""
        minimal = self.hierarchy.get_config(ComplexityLevel.MINIMAL)
        full = self.hierarchy.get_config(ComplexityLevel.FULL)

        # Full complexity should have more layers
        assert len(full.network_layers) > len(minimal.network_layers)

        # Full complexity should have more physics constraints
        assert len(full.physics_constraints) > len(minimal.physics_constraints)

        # Full complexity should have more coupling terms
        assert len(full.coupling_terms) > len(minimal.coupling_terms)

class TestComputationalCostEstimator:
    """Test ComputationalCostEstimator class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.estimator = ComputationalCostEstimator()

        # Create test configuration
        self.test_config = ModelConfiguration(
            complexity_level=ComplexityLevel.MODERATE,
            network_layers=[128, 64, 32],
            physics_constraints=["mass_conservation", "energy_conservation"],
            coupling_terms=["thermal_mechanical"],
            time_stepping_method="runge_kutta_4",
            spatial_resolution=250.0,
            temporal_resolution=900.0,
            batch_size=128,
            training_epochs=200,
            convergence_tolerance=1e-5,
        )

        # Create test resource assessment
        self.test_resources = ResourceAssessment(
            cpu_cores=8,
            memory_gb=32.0,
            gpu_count=1,
            gpu_memory_gb=8.0,
            storage_gb=1000.0,
            network_bandwidth_mbps=1000.0,
        )

    def test_estimate_cost(self):
        """Test cost estimation"""
        data_size = 10000

        cost_breakdown = self.estimator.estimate_cost(
            self.test_config, data_size, self.test_resources
        )

        assert isinstance(cost_breakdown, dict)
        assert "training_cost" in cost_breakdown
        assert "inference_cost" in cost_breakdown
        assert "memory_cost" in cost_breakdown
        assert "storage_cost" in cost_breakdown
        assert "total_cost" in cost_breakdown

        # All costs should be positive
        for cost in cost_breakdown.values():
            assert cost >= 0

    def test_optimize_configuration(self):
        """Test configuration optimization"""
        # Create test data assessment
        data_assessment = DataAssessment(
            total_data_points=10000,
            data_quality_score=0.8,
            temporal_coverage=0.7,
            spatial_coverage=0.7,
            data_types_available=["seismic", "deformation"],
            missing_data_fraction=0.1,
            noise_level=0.2,
        )

        # Create multiple configurations to choose from
        hierarchy = ModelHierarchy()
        configs = list(hierarchy.get_all_configs().values())

        optimal_config = self.estimator.optimize_configuration(
            configs, data_assessment, self.test_resources, target_accuracy=0.8
        )

        assert isinstance(optimal_config, ModelConfiguration)
        assert optimal_config in configs

class TestModelComplexityManager:
    """Test ModelComplexityManager class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.manager = ModelComplexityManager()

    def test_initialization(self):
        """Test manager initialization"""
        assert hasattr(self.manager, "complexity_assessor")
        assert hasattr(self.manager, "data_assessor")
        assert hasattr(self.manager, "resource_assessor")
        assert hasattr(self.manager, "model_hierarchy")
        assert hasattr(self.manager, "cost_estimator")

    def test_assess_simulation_complexity(self):
        """Test comprehensive simulation complexity assessment"""
        # Create test data
        data = {
            "seismic": torch.randn(1000, 10),
            "deformation": torch.randn(800, 5),
            "thermal": torch.randn(500, 3),
        }

        geological_process = GeologicalProcess.VOLCANIC_ERUPTION

        assessment_result = self.manager.assess_simulation_complexity(
            geological_process, data, target_accuracy=0.8
        )

        assert isinstance(assessment_result, dict)
        assert "data_assessment" in assessment_result
        assert "resource_assessment" in assessment_result
        assert "complexity_assessments" in assessment_result
        assert "optimal_configuration" in assessment_result
        assert "cost_breakdown" in assessment_result
        assert "recommendations" in assessment_result

    def test_select_optimal_model(self):
        """Test optimal model selection"""
        # Create test assessments
        data_assessment = DataAssessment(
            total_data_points=10000,
            data_quality_score=0.8,
            temporal_coverage=0.7,
            spatial_coverage=0.7,
            data_types_available=["seismic", "deformation"],
            missing_data_fraction=0.1,
            noise_level=0.2,
        )

        resource_assessment = ResourceAssessment(
            cpu_cores=8,
            memory_gb=32.0,
            gpu_count=1,
            gpu_memory_gb=8.0,
            storage_gb=1000.0,
            network_bandwidth_mbps=1000.0,
        )

        optimal_config = self.manager.select_optimal_model(
            data_assessment, resource_assessment, target_accuracy=0.8
        )

        assert isinstance(optimal_config, ModelConfiguration)
        assert optimal_config.complexity_level in ComplexityLevel

    def test_get_complexity_hierarchy(self):
        """Test getting complexity hierarchy"""
        hierarchy = self.manager.get_complexity_hierarchy()

        assert isinstance(hierarchy, dict)
        assert len(hierarchy) == 5  # All complexity levels

        for level, info in hierarchy.items():
            assert isinstance(level, ComplexityLevel)
            assert "configuration" in info
            assert "metrics" in info

    def test_generate_recommendations(self):
        """Test recommendation generation"""
        # Create test assessments
        data_assessment = DataAssessment(
            total_data_points=100,  # Sparse data
            data_quality_score=0.5,
            temporal_coverage=0.3,
            spatial_coverage=0.3,
            data_types_available=["seismic"],
            missing_data_fraction=0.3,  # High missing data
            noise_level=0.4,  # High noise
        )

        resource_assessment = ResourceAssessment(
            cpu_cores=2,
            memory_gb=4.0,  # Minimal resources
            gpu_count=0,
            gpu_memory_gb=0.0,
            storage_gb=50.0,
            network_bandwidth_mbps=100.0,
        )

        optimal_config = self.manager.model_hierarchy.get_config(
            ComplexityLevel.MINIMAL
        )

        recommendations = self.manager._generate_recommendations(
            data_assessment, resource_assessment, optimal_config
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Should recommend collecting more data
        assert any("data" in rec.lower() for rec in recommendations)

        # Should recommend better resources
        assert any(
            "resource" in rec.lower() or "gpu" in rec.lower() for rec in recommendations
        )

    def test_save_and_load_assessment(self):
        """Test saving assessment to file"""
        # Create test assessment
        test_assessment = {
            "test_key": "test_value",
            "complexity_level": ComplexityLevel.MODERATE,
            "tensor_data": torch.randn(10, 5),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            # Save assessment
            self.manager.save_assessment(test_assessment, filepath)

            # Verify file was created
            assert os.path.exists(filepath)

            # Load and verify content
            with open(filepath, "r") as f:
                loaded_data = json.load(f)

            assert "test_key" in loaded_data
            assert loaded_data["test_key"] == "test_value"

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_make_serializable(self):
        """Test making objects serializable"""
        # Test with various object types
        test_objects = {
            "enum": ComplexityLevel.MODERATE,
            "tensor": torch.randn(3, 3),
            "numpy_array": np.array([1, 2, 3]),
            "dict": {"nested": ComplexityLevel.FULL},
            "list": [ComplexityLevel.MINIMAL, torch.tensor([1, 2])],
            "string": "test_string",
            "number": 42,
        }

        serializable = self.manager._make_serializable(test_objects)

        # Should be JSON serializable
        json_str = json.dumps(serializable)
        assert isinstance(json_str, str)

        # Verify specific conversions
        assert serializable["enum"] == "moderate"
        assert isinstance(serializable["tensor"], list)
        assert isinstance(serializable["numpy_array"], list)
        assert serializable["dict"]["nested"] == "full"

class TestIntegration:
    """Integration tests for the complete system"""

    def setup_method(self):
        """Setup test fixtures"""
        self.manager = ModelComplexityManager()

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Step 1: Create realistic test data
        data = {
            "seismic_data": torch.randn(5000, 15),  # 5000 seismic measurements
            "deformation_data": torch.randn(2000, 8),  # 2000 GPS/InSAR measurements
            "thermal_data": torch.randn(1000, 5),  # 1000 thermal measurements
            "geochemical_data": torch.randn(500, 12),  # 500 geochemical samples
        }

        # Add some realistic noise and missing data
        for key, tensor in data.items():
            # Add some NaN values (missing data)
            mask = torch.rand(tensor.shape) < 0.05  # 5% missing data
            tensor[mask] = float("nan")

        # Step 2: Assess complexity
        geological_process = GeologicalProcess.VOLCANIC_ERUPTION
        assessment_result = self.manager.assess_simulation_complexity(
            geological_process, data, target_accuracy=0.85
        )

        # Step 3: Verify assessment results
        assert "optimal_configuration" in assessment_result
        assert "cost_breakdown" in assessment_result
        assert "recommendations" in assessment_result

        optimal_config = assessment_result["optimal_configuration"]
        assert optimal_config["complexity_level"] in [
            level.value for level in ComplexityLevel
        ]

        # Step 4: Verify cost breakdown
        cost_breakdown = assessment_result["cost_breakdown"]
        assert all(cost >= 0 for cost in cost_breakdown.values())

        # Step 5: Verify recommendations are reasonable
        recommendations = assessment_result["recommendations"]
        assert isinstance(recommendations, list)

        # Should have some recommendations given the test setup
        assert len(recommendations) > 0

if __name__ == "__main__":
    pytest.main([__file__])
