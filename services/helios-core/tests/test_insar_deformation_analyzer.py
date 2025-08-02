"""
Tests for InSAR Deformation Analysis System
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.insar_deformation_analyzer import (
    AtmosphericCorrection,
    AtmosphericCorrectionNet,
    DeformationPattern,
    DeformationPatternNet,
    InSARDataset,
    InSARDeformationAnalyzer,
    InSARScene,
    NoiseReductionFilter,
)
from models.geological import GeologicalLocation


class TestInSARScene:
    """Test InSAR scene data structure"""

    def test_insar_scene_creation(self):
        """Test creating a valid InSAR scene"""
        scene = InSARScene(
            scene_id="S1A_20230101_001",
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            satellite="Sentinel-1A",
            orbit_direction="ascending",
            look_angle=23.0,
            heading_angle=192.0,
            wavelength=0.056,
            spatial_resolution=20.0,
            coverage_area=(45.0, 46.0, -122.0, -121.0),
        )

        assert scene.scene_id == "S1A_20230101_001"
        assert scene.satellite == "Sentinel-1A"
        assert scene.look_angle == 23.0
        assert scene.wavelength == 0.056

    def test_insar_scene_validation(self):
        """Test InSAR scene parameter validation"""
        # Test invalid look angle
        with pytest.raises(ValueError, match="Look angle must be between 0 and 90"):
            InSARScene(
                scene_id="test",
                timestamp=datetime.now(),
                satellite="Sentinel-1A",
                orbit_direction="ascending",
                look_angle=95.0,  # Invalid
                heading_angle=192.0,
                wavelength=0.056,
                spatial_resolution=20.0,
                coverage_area=(45.0, 46.0, -122.0, -121.0),
            )

        # Test invalid wavelength
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            InSARScene(
                scene_id="test",
                timestamp=datetime.now(),
                satellite="Sentinel-1A",
                orbit_direction="ascending",
                look_angle=23.0,
                heading_angle=192.0,
                wavelength=-0.056,  # Invalid
                spatial_resolution=20.0,
                coverage_area=(45.0, 46.0, -122.0, -121.0),
            )

    def test_line_of_sight_vector(self):
        """Test line-of-sight vector calculation"""
        scene = InSARScene(
            scene_id="test",
            timestamp=datetime.now(),
            satellite="Sentinel-1A",
            orbit_direction="ascending",
            look_angle=23.0,
            heading_angle=192.0,
            wavelength=0.056,
            spatial_resolution=20.0,
            coverage_area=(45.0, 46.0, -122.0, -121.0),
        )

        los_vector = scene.get_line_of_sight_vector()

        # Check vector properties
        assert len(los_vector) == 3
        assert abs(np.linalg.norm(los_vector) - 1.0) < 1e-10  # Unit vector

        # Check components are reasonable
        assert los_vector[2] > 0  # Up component should be positive

    def test_phase_to_displacement(self):
        """Test phase to displacement conversion"""
        scene = InSARScene(
            scene_id="test",
            timestamp=datetime.now(),
            satellite="Sentinel-1A",
            orbit_direction="ascending",
            look_angle=23.0,
            heading_angle=192.0,
            wavelength=0.056,
            spatial_resolution=20.0,
            coverage_area=(45.0, 46.0, -122.0, -121.0),
        )

        # Test with known phase values
        phase = np.array([0, np.pi, 2 * np.pi])
        displacement = scene.phase_to_displacement(phase)

        expected = phase * scene.wavelength / (4 * np.pi)
        np.testing.assert_array_almost_equal(displacement, expected)


class TestAtmosphericCorrectionNet:
    """Test atmospheric correction neural network"""

    def test_network_creation(self):
        """Test creating atmospheric correction network"""
        model = AtmosphericCorrectionNet(in_channels=2, out_channels=1)

        # Test forward pass
        x = torch.randn(1, 2, 256, 256)
        output = model(x)

        assert output.shape == (1, 1, 256, 256)

    def test_network_with_different_sizes(self):
        """Test network with different input sizes"""
        model = AtmosphericCorrectionNet()

        # Test different input sizes
        for size in [128, 256, 512]:
            x = torch.randn(1, 2, size, size)
            output = model(x)
            assert output.shape == (1, 1, size, size)


class TestDeformationPatternNet:
    """Test deformation pattern recognition network"""

    def test_network_creation(self):
        """Test creating deformation pattern network"""
        model = DeformationPatternNet(in_channels=2, num_classes=8)

        # Test forward pass
        x = torch.randn(1, 2, 256, 256)
        classification, magnitude = model(x)

        assert classification.shape == (1, 8)
        assert magnitude.shape == (1, 1)

    def test_network_outputs(self):
        """Test network output ranges"""
        model = DeformationPatternNet()
        model.eval()

        x = torch.randn(1, 2, 256, 256)
        with torch.no_grad():
            classification, magnitude = model(x)

        # Classification should be logits (can be any value)
        assert classification.shape[1] == 8

        # Magnitude should be a single value
        assert magnitude.shape == (1, 1)


class TestNoiseReductionFilter:
    """Test noise reduction filters"""

    def test_lee_filter(self):
        """Test Lee filter implementation"""
        filter_obj = NoiseReductionFilter()

        # Create test data with noise
        np.random.seed(42)
        clean_data = np.ones((100, 100)) * 5.0
        noisy_data = clean_data + np.random.normal(0, 1, (100, 100))

        # Apply Lee filter
        filtered_data = filter_obj.apply_lee_filter(noisy_data, window_size=7)

        # Check that filtering reduces noise
        noise_before = np.std(noisy_data - clean_data)
        noise_after = np.std(filtered_data - clean_data)

        assert noise_after < noise_before
        assert filtered_data.shape == noisy_data.shape

    def test_bilateral_filter(self):
        """Test bilateral filter implementation"""
        filter_obj = NoiseReductionFilter()

        # Create test data with edges
        data = np.zeros((50, 50))
        data[:, 25:] = 10.0  # Step edge

        # Add noise
        np.random.seed(42)
        noisy_data = data + np.random.normal(0, 0.5, data.shape)

        # Apply bilateral filter
        filtered_data = filter_obj.apply_bilateral_filter(
            noisy_data, sigma_spatial=2.0, sigma_intensity=0.1
        )

        # Check that edges are preserved better than simple smoothing
        assert filtered_data.shape == noisy_data.shape

        # Edge should still be present
        left_mean = np.mean(filtered_data[:, :20])
        right_mean = np.mean(filtered_data[:, 30:])
        assert abs(right_mean - left_mean) > 5.0  # Edge preserved


class TestInSARDataset:
    """Test InSAR dataset for PyTorch"""

    def create_test_scene(self, scene_id: str) -> InSARScene:
        """Create a test InSAR scene"""
        scene = InSARScene(
            scene_id=scene_id,
            timestamp=datetime.now(),
            satellite="Sentinel-1A",
            orbit_direction="ascending",
            look_angle=23.0,
            heading_angle=192.0,
            wavelength=0.056,
            spatial_resolution=20.0,
            coverage_area=(45.0, 46.0, -122.0, -121.0),
        )

        # Add synthetic data
        scene.unwrapped_phase = np.random.randn(512, 512)
        scene.coherence = np.random.rand(512, 512)

        return scene

    def test_dataset_creation(self):
        """Test creating InSAR dataset"""
        scenes = [self.create_test_scene(f"scene_{i}") for i in range(3)]
        dataset = InSARDataset(scenes, patch_size=128, overlap=0.5)

        assert len(dataset) > 0

        # Test getting an item
        item = dataset[0]
        assert "data" in item
        assert "scene_id" in item
        assert "position" in item

        # Check data shape
        data = item["data"]
        assert data.shape[1] == 128  # Height
        assert data.shape[2] == 128  # Width

    def test_dataset_augmentation(self):
        """Test dataset augmentation"""
        scenes = [self.create_test_scene("test_scene")]
        dataset = InSARDataset(scenes, patch_size=64, augment=True)

        # Get multiple samples to test augmentation
        items = [dataset[0] for _ in range(10)]

        # All items should have same shape but potentially different values due to augmentation
        for item in items:
            assert item["data"].shape[1:] == (64, 64)


class TestInSARDeformationAnalyzer:
    """Test main InSAR deformation analyzer"""

    def create_test_scene_with_data(self) -> InSARScene:
        """Create a test scene with synthetic data"""
        scene = InSARScene(
            scene_id="test_scene",
            timestamp=datetime.now(),
            satellite="Sentinel-1A",
            orbit_direction="ascending",
            look_angle=23.0,
            heading_angle=192.0,
            wavelength=0.056,
            spatial_resolution=20.0,
            coverage_area=(45.0, 46.0, -122.0, -121.0),
        )

        # Create synthetic deformation pattern
        y, x = np.mgrid[0:256, 0:256]
        center_y, center_x = 128, 128

        # Gaussian deformation pattern
        deformation = 0.05 * np.exp(
            -((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * 30**2)
        )

        # Convert to phase
        phase = 4 * np.pi * deformation / scene.wavelength

        scene.unwrapped_phase = phase
        scene.coherence = 0.8 * np.ones_like(phase)

        return scene

    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        analyzer = InSARDeformationAnalyzer(device="cpu")

        assert analyzer.device == "cpu"
        assert analyzer.atmospheric_model is None
        assert analyzer.deformation_model is None
        assert analyzer.noise_filter is not None

    def test_process_insar_scene_basic(self):
        """Test basic InSAR scene processing"""
        analyzer = InSARDeformationAnalyzer(device="cpu")
        scene = self.create_test_scene_with_data()

        # Process without models (should still work for basic processing)
        results = analyzer.process_insar_scene(
            scene, apply_atmospheric_correction=False, apply_noise_reduction=True
        )

        assert results["status"] == "success"
        assert results["scene_id"] == "test_scene"
        assert "displacement_measurements" in results
        assert "uncertainty_estimates" in results
        assert results["noise_reduction_applied"] is True
        assert results["atmospheric_correction"] is None

    def test_ground_displacement_calculation(self):
        """Test ground displacement measurement calculation"""
        analyzer = InSARDeformationAnalyzer(device="cpu")
        scene = self.create_test_scene_with_data()

        # Calculate displacement measurements
        measurements = analyzer._calculate_ground_displacement(
            scene.unwrapped_phase, scene
        )

        assert len(measurements) > 0

        # Check measurement properties
        for measurement in measurements[:5]:  # Check first few
            assert isinstance(measurement.timestamp, datetime)
            assert isinstance(measurement.location, GeologicalLocation)
            assert len(measurement.displacement_vector) == 3
            assert measurement.measurement_type == "InSAR"

    def test_uncertainty_estimation(self):
        """Test uncertainty estimation"""
        analyzer = InSARDeformationAnalyzer(device="cpu")
        scene = self.create_test_scene_with_data()

        # Estimate uncertainty
        uncertainty = analyzer._estimate_uncertainty(
            scene.unwrapped_phase, scene.coherence, scene
        )

        assert "mean_displacement_uncertainty" in uncertainty
        assert "total_uncertainty" in uncertainty
        assert "atmospheric_uncertainty" in uncertainty
        assert "orbital_uncertainty" in uncertainty

        # Check reasonable values
        assert uncertainty["total_uncertainty"] > 0
        assert uncertainty["mean_displacement_uncertainty"] >= 0

    def test_noise_reduction_application(self):
        """Test noise reduction application"""
        analyzer = InSARDeformationAnalyzer(device="cpu")

        # Create noisy data
        np.random.seed(42)
        clean_phase = np.zeros((100, 100))
        noisy_phase = clean_phase + np.random.normal(0, 0.1, (100, 100))
        coherence = 0.8 * np.ones_like(noisy_phase)

        # Apply noise reduction
        filtered_phase = analyzer._apply_noise_reduction(noisy_phase, coherence)

        # Check that noise is reduced
        noise_before = np.std(noisy_phase - clean_phase)
        noise_after = np.std(filtered_phase - clean_phase)

        # Should reduce noise (allowing for some tolerance)
        assert noise_after <= noise_before * 1.1  # Allow 10% tolerance

    @patch("torch.cuda.is_available")
    def test_device_selection(self, mock_cuda):
        """Test device selection logic"""
        # Test CUDA available
        mock_cuda.return_value = True
        analyzer = InSARDeformationAnalyzer()
        assert analyzer.device == "cuda"

        # Test CUDA not available
        mock_cuda.return_value = False
        analyzer = InSARDeformationAnalyzer()
        assert analyzer.device == "cpu"

        # Test explicit device
        analyzer = InSARDeformationAnalyzer(device="cpu")
        assert analyzer.device == "cpu"

    def test_model_loading(self):
        """Test model loading functionality"""
        analyzer = InSARDeformationAnalyzer(device="cpu")

        # Test loading without files (should not crash)
        analyzer.load_models()

        assert analyzer.atmospheric_model is None
        assert analyzer.deformation_model is None

    def test_pattern_classification_mapping(self):
        """Test deformation pattern classification mapping"""
        analyzer = InSARDeformationAnalyzer(device="cpu")

        expected_classes = {
            0: "background",
            1: "volcanic_uplift",
            2: "volcanic_subsidence",
            3: "fault_slip",
            4: "landslide",
            5: "mining_subsidence",
            6: "groundwater_depletion",
            7: "thermal_expansion",
        }

        assert analyzer.pattern_classes == expected_classes

    def test_error_handling(self):
        """Test error handling in scene processing"""
        analyzer = InSARDeformationAnalyzer(device="cpu")

        # Create scene with missing data
        scene = InSARScene(
            scene_id="error_test",
            timestamp=datetime.now(),
            satellite="Sentinel-1A",
            orbit_direction="ascending",
            look_angle=23.0,
            heading_angle=192.0,
            wavelength=0.056,
            spatial_resolution=20.0,
            coverage_area=(45.0, 46.0, -122.0, -121.0),
        )
        # Don't add phase data

        # Should handle missing data gracefully
        results = analyzer.process_insar_scene(scene)

        # Should either succeed with limited processing or fail gracefully
        assert "status" in results
        assert results["scene_id"] == "error_test"


class TestDeformationPattern:
    """Test deformation pattern data structure"""

    def test_pattern_creation(self):
        """Test creating deformation pattern"""
        location = GeologicalLocation(latitude=45.5, longitude=-121.5, elevation=1000.0)

        pattern = DeformationPattern(
            pattern_id="test_pattern",
            pattern_type="volcanic_uplift",
            center_location=location,
            spatial_extent=(500.0, 800.0),
            magnitude=0.05,
            confidence=0.85,
        )

        assert pattern.pattern_id == "test_pattern"
        assert pattern.pattern_type == "volcanic_uplift"
        assert pattern.magnitude == 0.05
        assert pattern.confidence == 0.85
        assert pattern.spatial_extent == (500.0, 800.0)


class TestAtmosphericCorrection:
    """Test atmospheric correction data structure"""

    def test_correction_creation(self):
        """Test creating atmospheric correction result"""
        atmospheric_phase = np.random.randn(100, 100)
        corrected_phase = np.random.randn(100, 100)

        correction = AtmosphericCorrection(
            correction_method="deep_learning",
            atmospheric_phase=atmospheric_phase,
            corrected_phase=corrected_phase,
            correction_quality={"variance_reduction": 0.75},
        )

        assert correction.correction_method == "deep_learning"
        assert correction.atmospheric_phase.shape == (100, 100)
        assert correction.corrected_phase.shape == (100, 100)
        assert correction.correction_quality["variance_reduction"] == 0.75


if __name__ == "__main__":
    pytest.main([__file__])
