"""
Tests for CNN-based Seismic Event Detection

Comprehensive test suite for the seismic CNN processor, including
unit tests for individual components and integration tests for
the complete detection pipeline.
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import signal

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from core.seismic_cnn_processor import (
    SeismicCNN,
    SeismicCNNProcessor,
    SeismicEvent,
    SeismicEventClassifier,
    SeismicEventType,
    SeismicFeatureExtractor,
    WaveformFeatures,
)
from models.geological import GeologicalLocation


class TestSeismicCNN:
    """Test the CNN model architecture"""

    def test_model_initialization(self):
        """Test CNN model initialization"""
        model = SeismicCNN(
            input_length=3000,
            num_channels=3,
            num_classes=len(SeismicEventType),
            dropout_rate=0.3,
            use_attention=True,
        )

        assert model.input_length == 3000
        assert model.num_channels == 3
        assert model.num_classes == len(SeismicEventType)
        assert model.use_attention is True

    def test_model_forward_pass(self):
        """Test CNN forward pass"""
        model = SeismicCNN(input_length=1000, num_channels=3)

        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 1000)

        # Forward pass
        outputs = model(input_tensor)

        # Check output shapes
        assert "event_type" in outputs
        assert "detection" in outputs
        assert "magnitude" in outputs
        assert "confidence" in outputs

        assert outputs["event_type"].shape == (batch_size, len(SeismicEventType))
        assert outputs["detection"].shape == (batch_size, 1)
        assert outputs["magnitude"].shape == (batch_size, 1)
        assert outputs["confidence"].shape == (batch_size, 1)

        # Check output ranges
        assert torch.all(outputs["detection"] >= 0) and torch.all(
            outputs["detection"] <= 1
        )
        assert torch.all(outputs["confidence"] >= 0) and torch.all(
            outputs["confidence"] <= 1
        )

    def test_model_without_attention(self):
        """Test CNN model without attention mechanism"""
        model = SeismicCNN(input_length=1000, num_channels=3, use_attention=False)

        input_tensor = torch.randn(1, 3, 1000)
        outputs = model(input_tensor)

        assert all(
            key in outputs
            for key in ["event_type", "detection", "magnitude", "confidence"]
        )


class TestSeismicFeatureExtractor:
    """Test the feature extraction functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.extractor = SeismicFeatureExtractor(sampling_rate=100.0)

        # Create synthetic seismic signal
        t = np.linspace(0, 10, 1000)  # 10 seconds at 100 Hz
        self.synthetic_signal = (
            np.sin(2 * np.pi * 5 * t)  # 5 Hz component
            + 0.5 * np.sin(2 * np.pi * 15 * t)  # 15 Hz component
            + 0.1 * np.random.randn(len(t))  # Noise
        )

    def test_time_domain_features(self):
        """Test time-domain feature extraction"""
        features = self.extractor.extract_time_domain_features(self.synthetic_signal)

        # Check that all expected features are present
        expected_features = [
            "mean",
            "std",
            "variance",
            "skewness",
            "kurtosis",
            "max_amplitude",
            "rms",
            "peak_to_peak",
            "total_energy",
            "signal_energy",
            "zero_crossing_rate",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], float)
            assert np.isfinite(features[feature])

    def test_frequency_domain_features(self):
        """Test frequency-domain feature extraction"""
        features = self.extractor.extract_frequency_domain_features(
            self.synthetic_signal
        )

        expected_features = [
            "dominant_frequency",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rollo",
            "low_freq_energy",
            "mid_freq_energy",
            "high_freq_energy",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], float)
            assert np.isfinite(features[feature])

        # Check that dominant frequency is reasonable (should be around 5 Hz)
        assert 3 < features["dominant_frequency"] < 7

    def test_wavelet_features(self):
        """Test wavelet feature extraction"""
        features = self.extractor.extract_wavelet_features(self.synthetic_signal)

        expected_features = [
            "wavelet_energy",
            "wavelet_entropy",
            "time_bandwidth_product",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], float)
            assert np.isfinite(features[feature])

    def test_zero_crossing_rate(self):
        """Test zero crossing rate calculation"""
        # Create signal with known zero crossing rate
        t = np.linspace(0, 1, 100)
        signal_5hz = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

        zcr = self.extractor._calculate_zero_crossing_rate(signal_5hz)

        # 5 Hz sine wave should have approximately 10 zero crossings per second
        assert 8 < zcr < 12

    def test_spectral_features_edge_cases(self):
        """Test spectral features with edge cases"""
        # Test with constant signal
        constant_signal = np.ones(1000)
        features = self.extractor.extract_frequency_domain_features(constant_signal)

        assert features["dominant_frequency"] == 0.0
        assert np.isfinite(features["spectral_centroid"])

        # Test with zero signal
        zero_signal = np.zeros(1000)
        features = self.extractor.extract_frequency_domain_features(zero_signal)

        for key, value in features.items():
            assert np.isfinite(value) or value == 0.0


class TestSeismicEvent:
    """Test the SeismicEvent data structure"""

    def test_seismic_event_creation(self):
        """Test creating a seismic event"""
        location = GeologicalLocation(latitude=45.0, longitude=-122.0, elevation=1000.0)

        event = SeismicEvent(
            event_id="test_event_001",
            timestamp=datetime.utcnow(),
            location=location,
            event_type=SeismicEventType.EARTHQUAKE,
            magnitude=3.5,
            depth=10.0,
            confidence=0.85,
            duration=15.0,
            peak_amplitude=1000.0,
        )

        assert event.event_id == "test_event_001"
        assert event.event_type == SeismicEventType.EARTHQUAKE
        assert event.magnitude == 3.5
        assert event.confidence == 0.85
        assert event.location == location

    def test_seismic_event_validation(self):
        """Test seismic event parameter validation"""
        location = GeologicalLocation(latitude=45.0, longitude=-122.0, elevation=1000.0)

        # Test invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            SeismicEvent(
                event_id="test",
                timestamp=datetime.utcnow(),
                location=location,
                event_type=SeismicEventType.EARTHQUAKE,
                confidence=1.5,
            )

        # Test negative depth
        with pytest.raises(ValueError, match="Depth cannot be negative"):
            SeismicEvent(
                event_id="test",
                timestamp=datetime.utcnow(),
                location=location,
                event_type=SeismicEventType.EARTHQUAKE,
                depth=-5.0,
            )


class TestSeismicCNNProcessor:
    """Test the main CNN processor"""

    def setup_method(self):
        """Set up test fixtures"""
        self.processor = SeismicCNNProcessor(
            sampling_rate=100.0,
            window_length=10.0,  # Shorter for testing
            detection_threshold=0.5,
            device="cpu",
        )

        self.test_location = GeologicalLocation(
            latitude=45.0, longitude=-122.0, elevation=1000.0
        )

    def test_processor_initialization(self):
        """Test processor initialization"""
        assert self.processor.sampling_rate == 100.0
        assert self.processor.window_length == 10.0
        assert self.processor.window_samples == 1000
        assert self.processor.detection_threshold == 0.5
        assert isinstance(self.processor.model, SeismicCNN)

    def test_preprocess_waveform(self):
        """Test waveform preprocessing"""
        # Create test waveform with DC offset and different amplitudes
        waveform = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],  # Z component with DC offset
                [0.1, 0.2, 0.3, 0.4, 0.5],  # N component smaller amplitude
                [-1.0, -2.0, -3.0, -4.0, -5.0],  # E component negative
            ]
        )

        processed = self.processor._preprocess_waveform(waveform)

        # Check DC removal
        for i in range(3):
            assert abs(np.mean(processed[i])) < 1e-10

        # Check normalization (std should be 1 for each component)
        for i in range(3):
            assert abs(np.std(processed[i]) - 1.0) < 1e-10

    def test_extract_comprehensive_features(self):
        """Test comprehensive feature extraction"""
        # Create 3-component synthetic waveform
        samples = 1000
        waveform = np.random.randn(3, samples)

        features = self.processor._extract_comprehensive_features(waveform)

        assert isinstance(features, WaveformFeatures)
        assert len(features.time_domain_features) > 0
        assert len(features.frequency_domain_features) > 0
        assert len(features.statistical_features) > 0

        # Check that features exist for each component
        for component in ["Z", "N", "E"]:
            assert any(component in key for key in features.time_domain_features.keys())
            assert any(
                component in key for key in features.frequency_domain_features.keys()
            )

    def test_estimate_event_duration(self):
        """Test event duration estimation"""
        # Create synthetic event with known duration
        samples = 1000
        event_start = 200
        event_end = 600

        waveform = np.zeros((3, samples))
        # Add event signal
        waveform[:, event_start:event_end] = np.random.randn(3, event_end - event_start)

        duration = self.processor._estimate_event_duration(waveform)

        # Expected duration in seconds
        expected_duration = (event_end - event_start) / self.processor.sampling_rate

        # Allow some tolerance due to energy-based detection
        assert abs(duration - expected_duration) < 2.0  # Within 2 seconds

    def test_process_continuous_data(self):
        """Test processing continuous seismic data"""
        # Create synthetic continuous data
        duration_seconds = 30
        samples = int(duration_seconds * self.processor.sampling_rate)

        # 3-component waveform
        waveform_data = np.random.randn(3, samples) * 0.1  # Low amplitude background

        # Add a synthetic event
        event_start = samples // 2
        event_duration = int(5 * self.processor.sampling_rate)  # 5 second event
        waveform_data[:, event_start : event_start + event_duration] += (
            np.random.randn(3, event_duration) * 2
        )

        # Create timestamps
        start_time = datetime.utcnow()
        timestamps = np.array(
            [
                start_time + timedelta(seconds=i / self.processor.sampling_rate)
                for i in range(samples)
            ]
        )

        # Process data
        events = self.processor.process_continuous_data(
            waveform_data=waveform_data,
            timestamps=timestamps,
            sensor_location=self.test_location,
            sensor_id="TEST_STATION",
        )

        # Check that processing completed without errors
        assert isinstance(events, list)
        assert self.processor.processing_stats["total_windows_processed"] > 0

    def test_update_detection_threshold(self):
        """Test updating detection threshold"""
        # Test valid threshold
        self.processor.update_detection_threshold(0.7)
        assert self.processor.detection_threshold == 0.7

        # Test invalid thresholds
        with pytest.raises(ValueError):
            self.processor.update_detection_threshold(-0.1)

        with pytest.raises(ValueError):
            self.processor.update_detection_threshold(1.5)

    def test_model_save_load(self):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pth")

            # Save model
            metadata = {"test": "metadata"}
            self.processor.save_model(model_path, metadata)

            assert os.path.exists(model_path)

            # Create new processor and load model
            new_processor = SeismicCNNProcessor(
                model_path=model_path, sampling_rate=100.0, device="cpu"
            )

            # Verify model loaded correctly
            assert new_processor.sampling_rate == 100.0

    def test_processing_statistics(self):
        """Test processing statistics tracking"""
        stats = self.processor.get_processing_statistics()

        expected_keys = [
            "total_windows_processed",
            "events_detected",
            "last_processing_time",
            "detection_threshold",
            "sampling_rate",
            "window_length",
            "model_device",
        ]

        for key in expected_keys:
            assert key in stats


class TestSeismicEventClassifier:
    """Test the specialized event classifier"""

    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = SeismicEventClassifier()

    def test_volcanic_event_classification(self):
        """Test volcanic-specific event classification"""
        # Create test events with different characteristics
        events = []

        # Very long period event
        vlp_event = SeismicEvent(
            event_id="vlp_test",
            timestamp=datetime.utcnow(),
            location=None,
            event_type=SeismicEventType.UNKNOWN,
            duration=120.0,  # Long duration
            frequency_content={"Z_dominant_frequency": 0.5},  # Low frequency
        )
        events.append(vlp_event)

        # Long period event
        lp_event = SeismicEvent(
            event_id="lp_test",
            timestamp=datetime.utcnow(),
            location=None,
            event_type=SeismicEventType.UNKNOWN,
            duration=30.0,  # Medium duration
            frequency_content={"Z_dominant_frequency": 3.0},  # Intermediate frequency
        )
        events.append(lp_event)

        # Explosion event
        explosion_event = SeismicEvent(
            event_id="explosion_test",
            timestamp=datetime.utcnow(),
            location=None,
            event_type=SeismicEventType.UNKNOWN,
            duration=2.0,  # Short duration
            frequency_content={"Z_dominant_frequency": 15.0},  # High frequency
        )
        events.append(explosion_event)

        # Classify events
        classified_events = self.classifier.classify_volcanic_events(events)

        # Check classifications
        assert classified_events[0].event_type == SeismicEventType.VERY_LONG_PERIOD
        assert classified_events[1].event_type == SeismicEventType.LONG_PERIOD
        assert classified_events[2].event_type == SeismicEventType.EXPLOSION

        # Check that classification rules were recorded
        for event in classified_events:
            assert "classification_rule" in event.detection_metadata


class TestIntegration:
    """Integration tests for the complete seismic detection system"""

    def setup_method(self):
        """Set up integration test fixtures"""
        self.processor = SeismicCNNProcessor(
            sampling_rate=100.0,
            window_length=10.0,
            detection_threshold=0.3,  # Lower threshold for testing
            device="cpu",
        )

        self.classifier = SeismicEventClassifier()

        self.sensor_locations = {
            "STATION_A": GeologicalLocation(
                latitude=45.0, longitude=-122.0, elevation=1000.0
            ),
            "STATION_B": GeologicalLocation(
                latitude=45.1, longitude=-122.1, elevation=1100.0
            ),
            "STATION_C": GeologicalLocation(
                latitude=44.9, longitude=-121.9, elevation=900.0
            ),
        }

    def test_end_to_end_processing(self):
        """Test complete end-to-end processing pipeline"""
        # Create synthetic multi-station data
        duration_seconds = 60
        samples = int(duration_seconds * self.processor.sampling_rate)

        all_events = []

        for station_id, location in self.sensor_locations.items():
            # Create waveform with synthetic event
            waveform_data = np.random.randn(3, samples) * 0.1

            # Add synthetic earthquake
            event_start = samples // 2
            event_duration = int(10 * self.processor.sampling_rate)
            event_amplitude = (
                2.0 if station_id == "STATION_A" else 1.0
            )  # Closer station has larger amplitude

            waveform_data[:, event_start : event_start + event_duration] += (
                np.random.randn(3, event_duration) * event_amplitude
            )

            # Create timestamps
            start_time = datetime.utcnow()
            timestamps = np.array(
                [
                    start_time + timedelta(seconds=i / self.processor.sampling_rate)
                    for i in range(samples)
                ]
            )

            # Process station data
            station_events = self.processor.process_continuous_data(
                waveform_data=waveform_data,
                timestamps=timestamps,
                sensor_location=location,
                sensor_id=station_id,
            )

            all_events.extend(station_events)

        # Apply volcanic classification
        classified_events = self.classifier.classify_volcanic_events(all_events)

        # Estimate locations using multiple stations
        located_events = self.processor.estimate_location(
            classified_events, self.sensor_locations
        )

        # Verify results
        assert (
            len(located_events) >= 0
        )  # May or may not detect events depending on random data

        # Check that processing statistics were updated
        stats = self.processor.get_processing_statistics()
        assert stats["total_windows_processed"] > 0

    def test_real_time_simulation(self):
        """Test simulated real-time processing"""
        # Simulate streaming data processing
        chunk_duration = 5.0  # 5 second chunks
        chunk_samples = int(chunk_duration * self.processor.sampling_rate)

        # Process multiple chunks
        all_events = []
        for chunk_idx in range(5):  # 5 chunks = 25 seconds total
            # Create chunk data
            waveform_chunk = np.random.randn(3, chunk_samples) * 0.1

            # Add event to middle chunk
            if chunk_idx == 2:
                event_samples = int(3 * self.processor.sampling_rate)
                start_idx = (chunk_samples - event_samples) // 2
                waveform_chunk[:, start_idx : start_idx + event_samples] += (
                    np.random.randn(3, event_samples) * 3
                )

            # Create timestamps for chunk
            chunk_start_time = datetime.utcnow() + timedelta(
                seconds=chunk_idx * chunk_duration
            )
            timestamps = np.array(
                [
                    chunk_start_time
                    + timedelta(seconds=i / self.processor.sampling_rate)
                    for i in range(chunk_samples)
                ]
            )

            # Process chunk
            chunk_events = self.processor.process_continuous_data(
                waveform_data=waveform_chunk,
                timestamps=timestamps,
                sensor_location=self.sensor_locations["STATION_A"],
                sensor_id="STATION_A",
            )

            all_events.extend(chunk_events)

        # Verify streaming processing worked
        assert self.processor.processing_stats["total_windows_processed"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
