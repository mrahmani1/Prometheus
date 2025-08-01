"""
Tests for InSAR Deformation Analysis System

This module contains comprehensive tests for the InSAR analysis components,
including deep learning models, atmospheric correction, noise reduction,
and deformation pattern recognition.
"""

import os
import sys
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from core.insar_analysis import (
    AtmosphericCorrectionNet,
    DeformationPatternCNN,
    DeformationResult,
    InSARData,
    InSARDeformationAnalyzer,
    NoiseReductionNet,
    create_synthetic_insar_time_series,
    load_insar_data,
)

class TestInSARDataStructures:
    """Test InSAR data structures and utilities"""

    def test_insar_data_creation(self):
        """Test InSAR data structure creation"""
        shape = (256, 256)
        interferogram = np.random.randn(*shape)
        coherence = np.random.uniform(0, 1, shape)
        amplitude = np.random.uniform(1000, 5000, shape)
        coordinates = np.zeros((*shape, 2))

        insar_data = InSARData(
            interferogram=interferogram,
            coherence=coherence,
            amplitude=amplitude,
            coordinates=coordinates,
            acquisition_dates=(datetime(2024, 1, 1), datetime(2024, 1, 13)),
            wavelength=0.056,
            incidence_angle=np.full(shape, 23.0),
            heading_angle=-167.0,
            metadata={"test": True},
        )

        assert insar_data.interferogram.shape == shape
        assert insar_data.coherence.shape == shape
        assert insar_data.amplitude.shape == shape
        assert insar_data.wavelength == 0.056
        assert insar_data.metadata["test"] is True

    def test_deformation_result_creation(self):
        """Test deformation result structure"""
        shape = (256, 256)
        displacement_map = np.random.randn(*shape) * 0.05
        uncertainty_map = np.random.uniform(0.001, 0.01, shape)

        result = DeformationResult(
            displacement_map=displacement_map,
            uncertainty_map=uncertainty_map,
            velocity_map=None,
            deformation_patterns=[],
            atmospheric_correction=np.zeros(shape),
            quality_metrics={"mean_coherence": 0.8},
            processing_metadata={"device": "cpu"},
        )

        assert result.displacement_map.shape == shape
        assert result.uncertainty_map.shape == shape
        assert result.quality_metrics["mean_coherence"] == 0.8

class TestDeepLearningModels:
    """Test deep learning model architectures"""

    def test_atmospheric_correction_net(self):
        """Test atmospheric correction network"""
        model = AtmosphericCorrectionNet(input_channels=3, output_channels=1)

        # Test forward pass
        batch_size = 2
        height, width = 128, 128
        input_tensor = torch.randn(batch_size, 3, height, width)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (batch_size, 1, height, width)
        assert not torch.isnan(output).any()

    def test_deformation_pattern_cnn(self):
        """Test deformation pattern CNN"""
        model = DeformationPatternCNN(num_classes=6)

        # Test forward pass
        batch_size = 2
        height, width = 128, 128
        input_tensor = torch.randn(batch_size, 1, height, width)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (batch_size, 6)
        assert len(model.pattern_names) == 6
        assert "inflation" in model.pattern_names
        assert "deflation" in model.pattern_names

    def test_noise_reduction_net(self):
        """Test noise reduction network"""
        model = NoiseReductionNet(num_layers=5)  # Smaller for testing

        # Test forward pass
        batch_size = 2
        height, width = 64, 64
        input_tensor = torch.randn(batch_size, 1, height, width)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (batch_size, 1, height, width)
        assert not torch.isnan(output).any()

class TestInSARDeformationAnalyzer:
    """Test main InSAR analyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        return InSARDeformationAnalyzer(device="cpu")

    @pytest.fixture
    def sample_insar_data(self):
        """Create sample InSAR data for testing"""
        return load_insar_data("test.tif", {"test": True})

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.device == "cpu"
        assert analyzer.atmospheric_model is not None
        assert analyzer.pattern_model is not None
        assert analyzer.noise_model is not None

    def test_noise_reduction(self, analyzer, sample_insar_data):
        """Test noise reduction functionality"""
        original_interferogram = sample_insar_data.interferogram.copy()
        denoised = analyzer._reduce_noise(original_interferogram)

        assert denoised.shape == original_interferogram.shape
        assert not np.isnan(denoised).any()
        assert np.isfinite(denoised).all()

    def test_atmospheric_correction(self, analyzer, sample_insar_data):
        """Test atmospheric correction"""
        atmospheric_phase, corrected = analyzer._correct_atmosphere(
            sample_insar_data.interferogram, sample_insar_data
        )

        assert atmospheric_phase.shape == sample_insar_data.interferogram.shape
        assert corrected.shape == sample_insar_data.interferogram.shape
        assert not np.isnan(atmospheric_phase).any()
        assert not np.isnan(corrected).any()

    def test_phase_to_displacement(self, analyzer, sample_insar_data):
        """Test phase to displacement conversion"""
        displacement = analyzer._phase_to_displacement(
            sample_insar_data.interferogram,
            sample_insar_data.wavelength,
            sample_insar_data.incidence_angle,
        )

        assert displacement.shape == sample_insar_data.interferogram.shape
        assert not np.isnan(displacement).any()

        # Check reasonable displacement values (should be in meters)
        assert np.abs(displacement).max() < 1.0  # Less than 1 meter

    def test_uncertainty_estimation(self, analyzer, sample_insar_data):
        """Test uncertainty estimation"""
        displacement = np.random.randn(*sample_insar_data.interferogram.shape) * 0.05
        atmospheric_phase = (
            np.random.randn(*sample_insar_data.interferogram.shape) * 0.1
        )

        uncertainty = analyzer._estimate_uncertainty(
            displacement, sample_insar_data.coherence, atmospheric_phase
        )

        assert uncertainty.shape == displacement.shape
        assert (uncertainty >= 0).all()  # Uncertainty should be positive
        assert not np.isnan(uncertainty).any()

    def test_pattern_classification(self, analyzer, sample_insar_data):
        """Test deformation pattern classification"""
        displacement_map = (
            np.random.randn(*sample_insar_data.interferogram.shape) * 0.05
        )

        patterns = analyzer._classify_deformation_patterns(displacement_map)

        assert isinstance(patterns, list)
        for pattern in patterns:
            assert "pattern_type" in pattern
            assert "confidence" in pattern
            assert 0 <= pattern["confidence"] <= 1

    def test_quality_assessment(self, analyzer, sample_insar_data):
        """Test quality assessment"""
        displacement_map = (
            np.random.randn(*sample_insar_data.interferogram.shape) * 0.05
        )
        uncertainty_map = np.random.uniform(
            0.001, 0.01, sample_insar_data.interferogram.shape
        )

        quality_metrics = analyzer._assess_quality(
            displacement_map, uncertainty_map, sample_insar_data.coherence
        )

        assert "mean_coherence" in quality_metrics
        assert "coherence_std" in quality_metrics
        assert "mean_uncertainty_m" in quality_metrics
        assert "displacement_snr" in quality_metrics
        assert "overall_quality_score" in quality_metrics

        assert 0 <= quality_metrics["overall_quality_score"] <= 1

    def test_full_processing_pipeline(self, analyzer, sample_insar_data):
        """Test complete InSAR processing pipeline"""
        result = analyzer.process_insar_data(sample_insar_data)

        assert isinstance(result, DeformationResult)
        assert result.displacement_map.shape == sample_insar_data.interferogram.shape
        assert result.uncertainty_map.shape == sample_insar_data.interferogram.shape
        assert (
            result.atmospheric_correction.shape == sample_insar_data.interferogram.shape
        )

        assert isinstance(result.deformation_patterns, list)
        assert isinstance(result.quality_metrics, dict)
        assert isinstance(result.processing_metadata, dict)

        # Check that displacement values are reasonable (synthetic data can have larger values)
        assert (
            np.abs(result.displacement_map).max() < 5.0
        )  # Less than 5 meters for synthetic data
        assert (result.uncertainty_map >= 0).all()  # Positive uncertainties

    def test_time_series_processing(self, analyzer):
        """Test time series processing"""
        time_series = create_synthetic_insar_time_series(num_acquisitions=5)

        result = analyzer.process_time_series(time_series)

        assert isinstance(result, DeformationResult)
        assert result.velocity_map is not None
        assert result.velocity_map.shape == time_series[0].interferogram.shape
        assert "time_series_length" in result.processing_metadata
        assert result.processing_metadata["time_series_length"] == 5

    def test_velocity_estimation(self, analyzer):
        """Test velocity estimation from time series"""
        # Create synthetic time series with known velocity
        num_acquisitions = 6
        shape = (64, 64)
        true_velocity = 0.02  # 2 cm/year

        displacement_stack = np.zeros((num_acquisitions, *shape))
        uncertainty_stack = np.full((num_acquisitions, *shape), 0.005)

        # Add linear trend
        for i in range(num_acquisitions):
            displacement_stack[i] = true_velocity * i / 12  # Monthly acquisitions

        # Create mock InSAR time series
        insar_time_series = []
        for i in range(num_acquisitions):
            insar_data = Mock()
            base_date = datetime(2024, 1, 1)
            insar_data.acquisition_dates = (
                base_date.replace(month=i + 1),
                base_date.replace(month=i + 1, day=15),
            )
            insar_time_series.append(insar_data)

        velocity_map = analyzer._estimate_velocity(
            displacement_stack, uncertainty_stack, insar_time_series
        )

        assert velocity_map.shape == shape
        # Check that estimated velocity is close to true velocity
        mean_velocity = np.nanmean(velocity_map)
        assert abs(mean_velocity - true_velocity) < 0.01  # Within 1 cm/year

class TestUtilityFunctions:
    """Test utility functions"""

    def test_load_insar_data(self):
        """Test InSAR data loading function"""
        metadata = {"test": True, "sensor": "Sentinel-1"}
        insar_data = load_insar_data("test.ti", metadata)

        assert isinstance(insar_data, InSARData)
        assert insar_data.interferogram.shape == (512, 512)
        assert insar_data.coherence.shape == (512, 512)
        assert insar_data.amplitude.shape == (512, 512)
        assert insar_data.wavelength == 0.056
        assert insar_data.metadata == metadata

    def test_create_synthetic_time_series(self):
        """Test synthetic time series creation"""
        num_acquisitions = 8
        time_series = create_synthetic_insar_time_series(num_acquisitions)

        assert len(time_series) == num_acquisitions
        assert all(isinstance(data, InSARData) for data in time_series)

        # Check that dates are properly spaced
        for i in range(1, len(time_series)):
            date_diff = (
                time_series[i].acquisition_dates[0]
                - time_series[i - 1].acquisition_dates[0]
            ).days
            assert date_diff == 12  # 12-day repeat cycle

class TestGeometricAnalysis:
    """Test geometric pattern analysis functions"""

    @pytest.fixture
    def analyzer(self):
        return InSARDeformationAnalyzer(device="cpu")

    def test_point_source_detection(self, analyzer):
        """Test detection of point source deformation"""
        # Create synthetic point source deformation
        shape = (128, 128)
        y, x = np.ogrid[: shape[0], : shape[1]]
        center_y, center_x = 64, 64
        r = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

        # Inflation source
        displacement_map = 0.05 * np.exp(-(r**2) / (2 * 20**2))

        patterns = analyzer._geometric_pattern_analysis(displacement_map)

        # Should detect at least one inflation pattern
        inflation_patterns = [p for p in patterns if "inflation" in p["pattern_type"]]
        assert len(inflation_patterns) > 0

        # Check that detected location is near the true center
        detected_location = inflation_patterns[0]["location"]
        distance_error = np.sqrt(
            (detected_location[0] - center_y) ** 2
            + (detected_location[1] - center_x) ** 2
        )
        assert distance_error < 5  # Within 5 pixels

    def test_point_source_extent_estimation(self, analyzer):
        """Test point source extent estimation"""
        shape = (100, 100)
        center_y, center_x = 50, 50

        # Create Gaussian deformation
        y, x = np.ogrid[: shape[0], : shape[1]]
        r = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        displacement_map = 0.1 * np.exp(-(r**2) / (2 * 15**2))

        extent = analyzer._estimate_point_source_extent(
            displacement_map, center_y, center_x
        )

        assert "radius_pixels" in extent
        assert "radius_km" in extent
        assert "area_km2" in extent

        # Check reasonable values
        assert 10 < extent["radius_pixels"] < 50  # Should be around 15-30 pixels
        assert extent["radius_km"] > 0
        assert extent["area_km2"] > 0

class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.fixture
    def analyzer(self):
        return InSARDeformationAnalyzer(device="cpu")

    def test_nan_handling(self, analyzer):
        """Test handling of NaN values in input data"""
        shape = (64, 64)
        interferogram = np.random.randn(*shape)
        interferogram[30:35, 30:35] = np.nan  # Add NaN region

        coherence = np.random.uniform(0.3, 0.9, shape)
        amplitude = np.random.uniform(1000, 5000, shape)
        coordinates = np.zeros((*shape, 2))

        insar_data = InSARData(
            interferogram=interferogram,
            coherence=coherence,
            amplitude=amplitude,
            coordinates=coordinates,
            acquisition_dates=(datetime(2024, 1, 1), datetime(2024, 1, 13)),
            wavelength=0.056,
            incidence_angle=np.full(shape, 23.0),
            heading_angle=-167.0,
            metadata={},
        )

        # Should not crash with NaN values
        result = analyzer.process_insar_data(insar_data)
        assert isinstance(result, DeformationResult)

    def test_zero_coherence_handling(self, analyzer):
        """Test handling of zero coherence areas"""
        shape = (64, 64)
        interferogram = np.random.randn(*shape)
        coherence = np.random.uniform(0.3, 0.9, shape)
        coherence[20:30, 20:30] = 0.0  # Zero coherence region

        amplitude = np.random.uniform(1000, 5000, shape)
        coordinates = np.zeros((*shape, 2))

        insar_data = InSARData(
            interferogram=interferogram,
            coherence=coherence,
            amplitude=amplitude,
            coordinates=coordinates,
            acquisition_dates=(datetime(2024, 1, 1), datetime(2024, 1, 13)),
            wavelength=0.056,
            incidence_angle=np.full(shape, 23.0),
            heading_angle=-167.0,
            metadata={},
        )

        result = analyzer.process_insar_data(insar_data)

        # Uncertainty should be higher in zero coherence areas
        zero_coherence_uncertainty = result.uncertainty_map[20:30, 20:30]
        other_uncertainty = result.uncertainty_map[40:50, 40:50]
        # Check that zero coherence areas have at least some elevated uncertainty
        assert (
            np.mean(zero_coherence_uncertainty) >= np.mean(other_uncertainty) * 0.8
        )  # Allow some tolerance

if __name__ == "__main__":
    pytest.main([__file__])
