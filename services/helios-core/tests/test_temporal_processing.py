"""
Tests for Multi-Scale Temporal Processing

Tests the implementation of separate model branches for geological vs. real-time timescales,
temporal attention mechanisms, adaptive time-stepping, and temporal interpolation/extrapolation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from core.temporal_processing import (
    MultiScaleTemporalPINN, TemporalScale, TimeScale, GeologicalProcess,
    TemporalAttentionMechanism, MultiScaleTemporalBranch, AdaptiveTimeStepper,
    AdaptiveTimeStepConfig, TemporalInterpolator, create_default_temporal_scales,
    TemporalPositionalEncoding
)


class TestTemporalScale:
    """Test TemporalScale dataclass"""
    
    def test_temporal_scale_creation(self):
        """Test creating a temporal scale"""
        scale = TemporalScale(
            name=TimeScale.GEOLOGICAL,
            min_time=3.15e10,  # 1000 years
            max_time=3.15e13,  # 1 million years
            characteristic_time=3.15e11  # 10,000 years
        )
        
        assert scale.name == TimeScale.GEOLOGICAL
        assert scale.min_time == 3.15e10
        assert scale.max_time == 3.15e13
        assert scale.characteristic_time == 3.15e11
        assert scale.adaptive_stepping is True
    
    def test_temporal_scale_validation(self):
        """Test temporal scale validation"""
        # Test invalid time range
        with pytest.raises(ValueError, match="Minimum time must be less than maximum time"):
            TemporalScale(
                name=TimeScale.REAL_TIME,
                min_time=100.0,
                max_time=50.0,
                characteristic_time=75.0
            )
        
        # Test invalid characteristic time
        with pytest.raises(ValueError, match="Characteristic time must be positive"):
            TemporalScale(
                name=TimeScale.REAL_TIME,
                min_time=1.0,
                max_time=100.0,
                characteristic_time=-10.0
            )


class TestAdaptiveTimeStepConfig:
    """Test AdaptiveTimeStepConfig dataclass"""
    
    def test_config_creation(self):
        """Test creating adaptive time step configuration"""
        config = AdaptiveTimeStepConfig(
            initial_dt=1.0,
            min_dt=0.1,
            max_dt=10.0,
            tolerance=1e-6,
            safety_factor=0.9
        )
        
        assert config.initial_dt == 1.0
        assert config.min_dt == 0.1
        assert config.max_dt == 10.0
        assert config.tolerance == 1e-6
        assert config.safety_factor == 0.9
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid time step bounds
        with pytest.raises(ValueError, match="Time step bounds must satisfy"):
            AdaptiveTimeStepConfig(
                initial_dt=5.0,
                min_dt=10.0,
                max_dt=1.0
            )
        
        # Test invalid tolerance
        with pytest.raises(ValueError, match="Tolerance must be positive"):
            AdaptiveTimeStepConfig(
                initial_dt=1.0,
                min_dt=0.1,
                max_dt=10.0,
                tolerance=-1e-6
            )
        
        # Test invalid safety factor
        with pytest.raises(ValueError, match="Safety factor must be between 0 and 1"):
            AdaptiveTimeStepConfig(
                initial_dt=1.0,
                min_dt=0.1,
                max_dt=10.0,
                safety_factor=1.5
            )


class TestTemporalPositionalEncoding:
    """Test TemporalPositionalEncoding module"""
    
    def test_positional_encoding_creation(self):
        """Test creating temporal positional encoding"""
        encoding = TemporalPositionalEncoding(feature_dim=64)
        
        assert encoding.feature_dim == 64
        assert encoding.max_time_scale == 1e15
        assert hasattr(encoding, 'div_term')
    
    def test_positional_encoding_forward(self):
        """Test forward pass of positional encoding"""
        encoding = TemporalPositionalEncoding(feature_dim=64)
        
        batch_size, seq_len = 2, 10
        features = torch.randn(batch_size, seq_len, 64)
        time_coords = torch.linspace(0, 1000, seq_len).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
        
        encoded_features = encoding(features, time_coords)
        
        assert encoded_features.shape == features.shape
        assert not torch.equal(encoded_features, features)  # Should be different after encoding


class TestTemporalAttentionMechanism:
    """Test TemporalAttentionMechanism module"""
    
    def test_attention_creation(self):
        """Test creating temporal attention mechanism"""
        attention = TemporalAttentionMechanism(
            feature_dim=64,
            num_scales=3,
            num_heads=8
        )
        
        assert attention.feature_dim == 64
        assert attention.num_scales == 3
        assert attention.num_heads == 8
        assert len(attention.scale_projections) == 3
    
    def test_attention_forward(self):
        """Test forward pass of temporal attention"""
        attention = TemporalAttentionMechanism(
            feature_dim=64,
            num_scales=3,
            num_heads=8
        )
        
        batch_size, seq_len = 2, 10
        features = [
            torch.randn(batch_size, seq_len, 64) for _ in range(3)
        ]
        time_coords = [
            torch.linspace(0, 1000, seq_len).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
            for _ in range(3)
        ]
        
        fused_features = attention(features, time_coords)
        
        assert fused_features.shape == (batch_size, seq_len, 64)


class TestMultiScaleTemporalBranch:
    """Test MultiScaleTemporalBranch module"""
    
    def test_branch_creation(self):
        """Test creating temporal branch"""
        scale = TemporalScale(
            name=TimeScale.GEOLOGICAL,
            min_time=3.15e10,
            max_time=3.15e13,
            characteristic_time=3.15e11
        )
        
        branch = MultiScaleTemporalBranch(
            temporal_scale=scale,
            input_dim=4,
            hidden_dims=[64, 128, 64],
            output_dim=2
        )
        
        assert branch.temporal_scale == scale
        assert branch.input_dim == 4
        assert branch.output_dim == 2
    
    def test_branch_forward(self):
        """Test forward pass of temporal branch"""
        scale = TemporalScale(
            name=TimeScale.GEOLOGICAL,
            min_time=3.15e10,
            max_time=3.15e13,
            characteristic_time=3.15e11
        )
        
        branch = MultiScaleTemporalBranch(
            temporal_scale=scale,
            input_dim=4,
            hidden_dims=[64, 128, 64],
            output_dim=2
        )
        
        batch_size = 5
        x = torch.randn(batch_size, 4)
        time_coord = torch.randn(batch_size, 1)
        
        output = branch(x, time_coord)
        
        assert output.shape == (batch_size, 2)
    
    def test_characteristic_time(self):
        """Test getting characteristic time"""
        scale = TemporalScale(
            name=TimeScale.GEOLOGICAL,
            min_time=3.15e10,
            max_time=3.15e13,
            characteristic_time=3.15e11
        )
        
        branch = MultiScaleTemporalBranch(
            temporal_scale=scale,
            input_dim=4,
            hidden_dims=[64, 128, 64],
            output_dim=2
        )
        
        assert branch.get_characteristic_time() == 3.15e11


class TestAdaptiveTimeStepper:
    """Test AdaptiveTimeStepper class"""
    
    def test_stepper_creation(self):
        """Test creating adaptive time stepper"""
        config = AdaptiveTimeStepConfig(
            initial_dt=1.0,
            min_dt=0.1,
            max_dt=10.0
        )
        
        stepper = AdaptiveTimeStepper(config)
        
        assert stepper.config == config
        assert stepper.current_dt == 1.0
        assert stepper.step_count == 0
        assert stepper.error_history == []
    
    def test_compute_next_timestep(self):
        """Test computing next time step"""
        config = AdaptiveTimeStepConfig(
            initial_dt=1.0,
            min_dt=0.1,
            max_dt=10.0
        )
        
        stepper = AdaptiveTimeStepper(config)
        
        current_solution = torch.randn(10, 3)
        previous_solution = torch.randn(10, 3)
        
        next_dt = stepper.compute_next_timestep(
            current_solution, previous_solution, GeologicalProcess.CRUSTAL_DEFORMATION
        )
        
        assert 0.1 <= next_dt <= 10.0
        assert stepper.step_count == 1
        assert len(stepper.error_history) == 1
    
    def test_process_constraints(self):
        """Test process-specific time step constraints"""
        config = AdaptiveTimeStepConfig(
            initial_dt=1.0,
            min_dt=0.1,
            max_dt=1e15  # Very large max
        )
        
        stepper = AdaptiveTimeStepper(config)
        
        # Test seismic process (should have small time steps)
        current_solution = torch.randn(10, 3)
        previous_solution = torch.randn(10, 3)
        
        next_dt = stepper.compute_next_timestep(
            current_solution, previous_solution, GeologicalProcess.SEISMIC_WAVE_PROPAGATION
        )
        
        # Should be limited by seismic process characteristic time
        assert next_dt <= 1.0  # Should be limited to fraction of 1 second
    
    def test_stepper_reset(self):
        """Test resetting time stepper"""
        config = AdaptiveTimeStepConfig(
            initial_dt=1.0,
            min_dt=0.1,
            max_dt=10.0
        )
        
        stepper = AdaptiveTimeStepper(config)
        stepper.current_dt = 5.0
        stepper.step_count = 10
        stepper.error_history = [1.0, 2.0, 3.0]
        
        stepper.reset()
        
        assert stepper.current_dt == 1.0
        assert stepper.step_count == 0
        assert stepper.error_history == []


class TestTemporalInterpolator:
    """Test TemporalInterpolator class"""
    
    def test_interpolator_creation(self):
        """Test creating temporal interpolator"""
        interpolator = TemporalInterpolator(method="linear")
        assert interpolator.method == "linear"
        
        interpolator = TemporalInterpolator(method="cubic_spline")
        assert interpolator.method == "cubic_spline"
    
    def test_linear_interpolation(self):
        """Test linear interpolation"""
        interpolator = TemporalInterpolator(method="linear")
        
        time_points = torch.tensor([0.0, 1.0, 2.0])
        values = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
        target_times = torch.tensor([0.5, 1.5])
        
        interpolated = interpolator.interpolate(time_points, values, target_times)
        
        assert interpolated.shape == (2, 2)
        # Check interpolated values
        expected = torch.tensor([[0.5, 1.0], [1.5, 3.0]])
        assert torch.allclose(interpolated, expected, atol=1e-6)
    
    def test_neural_interpolation(self):
        """Test neural network interpolation"""
        interpolator = TemporalInterpolator(method="neural")
        
        time_points = torch.tensor([0.0, 1.0, 2.0])
        values = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
        target_times = torch.tensor([0.5, 1.5])
        
        interpolated = interpolator.interpolate(time_points, values, target_times)
        
        assert interpolated.shape == (2, 2)
        # Neural interpolation should produce reasonable results
        assert torch.all(torch.isfinite(interpolated))
    
    def test_linear_extrapolation(self):
        """Test linear extrapolation"""
        interpolator = TemporalInterpolator()
        
        time_points = torch.tensor([0.0, 1.0, 2.0])
        values = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
        target_times = torch.tensor([3.0, 4.0])
        
        extrapolated = interpolator.extrapolate(time_points, values, target_times, method="linear")
        
        assert extrapolated.shape == (2, 2)
        # Check extrapolated values (linear trend continues)
        expected = torch.tensor([[3.0, 6.0], [4.0, 8.0]])
        assert torch.allclose(extrapolated, expected, atol=1e-6)
    
    def test_polynomial_extrapolation(self):
        """Test polynomial extrapolation"""
        interpolator = TemporalInterpolator()
        
        time_points = torch.tensor([0.0, 1.0, 2.0])
        values = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
        target_times = torch.tensor([3.0])
        
        extrapolated = interpolator.extrapolate(time_points, values, target_times, method="polynomial")
        
        assert extrapolated.shape == (1, 2)
        assert torch.all(torch.isfinite(extrapolated))
    
    def test_exponential_extrapolation(self):
        """Test exponential extrapolation"""
        interpolator = TemporalInterpolator()
        
        time_points = torch.tensor([0.0, 1.0, 2.0])
        values = torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])  # Exponential growth
        target_times = torch.tensor([3.0])
        
        extrapolated = interpolator.extrapolate(time_points, values, target_times, method="exponential")
        
        assert extrapolated.shape == (1, 2)
        assert torch.all(torch.isfinite(extrapolated))
        # Should continue exponential trend
        assert torch.all(extrapolated > values[-1])


class TestMultiScaleTemporalPINN:
    """Test MultiScaleTemporalPINN module"""
    
    def test_model_creation(self):
        """Test creating multi-scale temporal PINN"""
        temporal_scales = create_default_temporal_scales()
        
        model = MultiScaleTemporalPINN(
            input_dim=4,
            output_dim=8,  # Changed to be divisible by 4 heads
            temporal_scales=temporal_scales,
            branch_hidden_dims=[32, 64, 32],
            attention_heads=4
        )
        
        assert model.input_dim == 4
        assert model.output_dim == 8
        assert model.num_scales == len(temporal_scales)
        assert len(model.temporal_branches) == len(temporal_scales)
    
    def test_model_forward(self):
        """Test forward pass of multi-scale temporal PINN"""
        temporal_scales = create_default_temporal_scales()[:3]  # Use first 3 scales for testing
        
        model = MultiScaleTemporalPINN(
            input_dim=4,
            output_dim=8,  # Changed to be divisible by 4 heads
            temporal_scales=temporal_scales,
            branch_hidden_dims=[32, 64, 32],
            attention_heads=4
        )
        
        batch_size = 5
        x = torch.randn(batch_size, 4)
        time_coords = torch.randn(batch_size, 1)
        
        results = model(x, time_coords)
        
        assert "fused_prediction" in results
        assert "branch_predictions" in results
        assert "attention_weights" in results
        
        assert results["fused_prediction"].shape == (batch_size, 8)
        assert len(results["branch_predictions"]) == 3
        
        for scale_name, prediction in results["branch_predictions"].items():
            assert prediction.shape == (batch_size, 8)
    
    def test_model_with_target_scale(self):
        """Test model forward with target scale"""
        temporal_scales = create_default_temporal_scales()[:3]
        
        model = MultiScaleTemporalPINN(
            input_dim=4,
            output_dim=8,  # Changed to be divisible by 4 heads
            temporal_scales=temporal_scales,
            branch_hidden_dims=[32, 64, 32],
            attention_heads=4
        )
        
        batch_size = 5
        x = torch.randn(batch_size, 4)
        time_coords = torch.randn(batch_size, 1)
        
        # Use a scale that exists in the first 3 scales
        results = model(x, time_coords, target_scale=TimeScale.MEDIUM_TERM)
        
        assert "target_scale_prediction" in results
        assert results["target_scale_prediction"].shape == (batch_size, 8)
    
    def test_adaptive_stepping_prediction(self):
        """Test prediction with adaptive time stepping"""
        temporal_scales = create_default_temporal_scales()[:3]
        
        model = MultiScaleTemporalPINN(
            input_dim=4,
            output_dim=8,  # Changed to be divisible by 4 heads
            temporal_scales=temporal_scales,
            branch_hidden_dims=[32, 64, 32],
            attention_heads=4
        )
        
        x = torch.randn(1, 4)
        time_start = 0.0
        time_end = 100.0
        process_type = GeologicalProcess.CRUSTAL_DEFORMATION
        
        results = model.predict_with_adaptive_stepping(
            x, time_start, time_end, process_type
        )
        
        assert "times" in results
        assert "solutions" in results
        assert "time_steps" in results
        assert "final_solution" in results
        assert "total_steps" in results
        
        assert len(results["times"]) == len(results["solutions"])
        assert len(results["time_steps"]) == len(results["times"]) - 1
        assert results["times"][0] == time_start
        assert results["times"][-1] <= time_end
    
    def test_temporal_interpolation(self):
        """Test temporal solution interpolation"""
        temporal_scales = create_default_temporal_scales()[:3]
        
        model = MultiScaleTemporalPINN(
            input_dim=4,
            output_dim=8,  # Changed to be divisible by 4 heads
            temporal_scales=temporal_scales,
            branch_hidden_dims=[32, 64, 32],
            attention_heads=4
        )
        
        known_times = torch.tensor([0.0, 1.0, 2.0])
        known_solutions = torch.randn(3, 8)
        target_times = torch.tensor([0.5, 1.5])
        
        interpolated = model.interpolate_temporal_solution(
            known_times, known_solutions, target_times
        )
        
        assert interpolated.shape == (2, 8)
    
    def test_temporal_extrapolation(self):
        """Test temporal solution extrapolation"""
        temporal_scales = create_default_temporal_scales()[:3]
        
        model = MultiScaleTemporalPINN(
            input_dim=4,
            output_dim=8,  # Changed to be divisible by 4 heads
            temporal_scales=temporal_scales,
            branch_hidden_dims=[32, 64, 32],
            attention_heads=4
        )
        
        known_times = torch.tensor([0.0, 1.0, 2.0])
        known_solutions = torch.randn(3, 8)
        target_times = torch.tensor([3.0, 4.0])
        
        extrapolated = model.extrapolate_temporal_solution(
            known_times, known_solutions, target_times, method="linear"
        )
        
        assert extrapolated.shape == (2, 8)


class TestDefaultTemporalScales:
    """Test default temporal scales creation"""
    
    def test_create_default_scales(self):
        """Test creating default temporal scales"""
        scales = create_default_temporal_scales()
        
        assert len(scales) == 5
        
        # Check that scales are ordered from shortest to longest
        for i in range(len(scales) - 1):
            assert scales[i].max_time <= scales[i + 1].min_time
        
        # Check specific scales
        scale_names = [scale.name for scale in scales]
        expected_names = [
            TimeScale.REAL_TIME,
            TimeScale.SHORT_TERM,
            TimeScale.MEDIUM_TERM,
            TimeScale.GEOLOGICAL,
            TimeScale.DEEP_TIME
        ]
        
        assert scale_names == expected_names
    
    def test_scale_characteristics(self):
        """Test characteristics of default scales"""
        scales = create_default_temporal_scales()
        
        # Real-time scale (seconds to days)
        real_time = scales[0]
        assert real_time.name == TimeScale.REAL_TIME
        assert real_time.min_time == 1.0  # 1 second
        assert real_time.max_time == 86400.0  # 1 day
        
        # Deep time scale (millions to billions of years)
        deep_time = scales[-1]
        assert deep_time.name == TimeScale.DEEP_TIME
        assert deep_time.min_time == 3.15e13  # 1 million years
        assert deep_time.max_time == 3.15e17  # 10 billion years


@pytest.fixture
def sample_temporal_model():
    """Fixture for creating a sample temporal model"""
    temporal_scales = create_default_temporal_scales()[:3]
    
    model = MultiScaleTemporalPINN(
        input_dim=4,
        output_dim=8,  # Changed to be divisible by 4 heads
        temporal_scales=temporal_scales,
        branch_hidden_dims=[32, 64, 32],
        attention_heads=4
    )
    
    return model


class TestIntegration:
    """Integration tests for temporal processing components"""
    
    def test_end_to_end_temporal_prediction(self, sample_temporal_model):
        """Test end-to-end temporal prediction workflow"""
        model = sample_temporal_model
        
        # Create sample data
        batch_size = 3
        spatial_coords = torch.randn(batch_size, 4)
        time_coords = torch.linspace(0, 1000, batch_size).unsqueeze(-1)
        
        # Make predictions
        results = model(spatial_coords, time_coords)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert "fused_prediction" in results
        assert "branch_predictions" in results
        
        # Verify shapes
        assert results["fused_prediction"].shape == (batch_size, 8)
        
        # Verify all branches produced outputs
        assert len(results["branch_predictions"]) == 3
        for prediction in results["branch_predictions"].values():
            assert prediction.shape == (batch_size, 8)
    
    def test_adaptive_stepping_workflow(self, sample_temporal_model):
        """Test adaptive time stepping workflow"""
        model = sample_temporal_model
        
        # Create initial conditions
        spatial_coords = torch.randn(1, 4)
        initial_solution = torch.randn(1, 8)
        
        # Run adaptive stepping
        results = model.predict_with_adaptive_stepping(
            spatial_coords,
            time_start=0.0,
            time_end=50.0,
            process_type=GeologicalProcess.CRUSTAL_DEFORMATION,
            initial_solution=initial_solution
        )
        
        # Verify adaptive stepping worked
        assert len(results["times"]) > 1
        assert len(results["solutions"]) == len(results["times"])
        assert results["total_steps"] > 0
        
        # Verify time progression
        times = results["times"]
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1]
    
    def test_interpolation_extrapolation_workflow(self, sample_temporal_model):
        """Test interpolation and extrapolation workflow"""
        model = sample_temporal_model
        
        # Create known data points
        known_times = torch.tensor([0.0, 10.0, 20.0])
        known_solutions = torch.randn(3, 8)
        
        # Test interpolation
        interp_times = torch.tensor([5.0, 15.0])
        interpolated = model.interpolate_temporal_solution(
            known_times, known_solutions, interp_times
        )
        
        assert interpolated.shape == (2, 8)
        
        # Test extrapolation
        extrap_times = torch.tensor([30.0, 40.0])
        extrapolated = model.extrapolate_temporal_solution(
            known_times, known_solutions, extrap_times, method="linear"
        )
        
        assert extrapolated.shape == (2, 8)


if __name__ == "__main__":
    pytest.main([__file__])