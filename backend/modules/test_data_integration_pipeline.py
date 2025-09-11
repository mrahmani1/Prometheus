"""
Comprehensive test suite for Advanced Data Integration Pipeline
Tests all major functionality including real-time volcanic data enhancement,
multi-source data integration, quality assessment, and feature engineering.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio
import logging

# Import the module under test
from data_integration_pipeline import (
    AdvancedDataIntegrationPipeline,
    DataLayer,
    DataType,
    ProcessingMethod,
    QualityLevel,
    AnomalyType,
    QualityMetrics,
    GeographicBounds,
    TemporalRange,
    QualityAssessmentEngine,
    AnomalyDetectionEngine,
    AutomatedFeatureEngineer
)

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestDataLayer:
    """Test suite for DataLayer class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sample_data = np.random.normal(0, 1, 1000)
        self.data_layer = DataLayer(
            name="test_layer",
            data_type=DataType.VOLCANIC,
            data=self.sample_data,
            metadata={"source": "test", "version": "1.0"}
        )
    
    def test_data_layer_initialization(self):
        """Test DataLayer initialization"""
        assert self.data_layer.name == "test_layer"
        assert self.data_layer.data_type == DataType.VOLCANIC
        assert np.array_equal(self.data_layer.data, self.sample_data)
        assert self.data_layer.metadata["source"] == "test"
        assert isinstance(self.data_layer.created_at, datetime)
    
    def test_real_time_volcanic_data_enhancement(self):
        """Test real-time volcanic data enhancement"""
        base_features = np.random.normal(0, 1, 500)
        enhanced_data = self.data_layer.get_real_time_volcanic_data(base_features)
        
        # Enhanced data should have same length as input
        assert len(enhanced_data) == len(base_features)
        
        # Enhanced data should be different from input (unless enhancement fails)
        assert not np.array_equal(enhanced_data, base_features)
        
        # Enhanced data should be finite
        assert np.all(np.isfinite(enhanced_data))
    
    def test_geological_data_enhancement(self):
        """Test geological data enhancement components"""
        features = np.random.normal(0, 1, 200)
        enhanced = self.data_layer._enhance_with_geological_data(features)
        
        assert len(enhanced) == len(features)
        assert np.all(np.isfinite(enhanced))
    
    def test_geophysical_data_enhancement(self):
        """Test geophysical data enhancement components"""
        features = np.random.normal(0, 1, 200)
        enhanced = self.data_layer._enhance_with_geophysical_data(features)
        
        assert len(enhanced) == len(features)
        assert np.all(np.isfinite(enhanced))
    
    def test_geochemical_data_enhancement(self):
        """Test geochemical data enhancement components"""
        features = np.random.normal(0, 1, 200)
        enhanced = self.data_layer._enhance_with_geochemical_data(features)
        
        assert len(enhanced) == len(features)
        assert np.all(np.isfinite(enhanced))
    
    def test_remote_sensing_enhancement(self):
        """Test remote sensing data enhancement"""
        features = np.random.normal(0, 1, 200)
        enhanced = self.data_layer._enhance_with_remote_sensing_data(features)
        
        assert len(enhanced) == len(features)
        assert np.all(np.isfinite(enhanced))
    
    def test_quality_assessment(self):
        """Test quality assessment functionality"""
        # Create data with known quality issues
        noisy_data = np.random.normal(0, 1, 100)
        noisy_data[10:15] = np.nan  # Missing data
        noisy_data[50] = 100  # Outlier
        
        layer = DataLayer(
            name="noisy_test",
            data_type=DataType.SEISMIC,
            data=noisy_data
        )
        
        quality_data = layer._apply_quality_assessment(noisy_data)
        
        assert len(quality_data) == len(noisy_data)
        assert layer.quality_metrics is not None
        assert 0 <= layer.quality_metrics.overall_score <= 1
    
    def test_automated_feature_engineering(self):
        """Test automated feature engineering"""
        # Create data with known patterns
        t = np.linspace(0, 10, 200)
        signal = np.sin(2 * np.pi * t) + 0.1 * np.random.normal(0, 1, 200)
        
        engineered = self.data_layer._apply_automated_feature_engineering(signal)
        
        assert len(engineered) == len(signal)
        assert np.all(np.isfinite(engineered))


class TestAdvancedDataIntegrationPipeline:
    """Test suite for AdvancedDataIntegrationPipeline class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.pipeline = AdvancedDataIntegrationPipeline()
        
        # Create sample data sources
        self.sample_data_sources = [
            {
                'type': 'seismic',
                'name': 'test_seismic',
                'data': np.random.normal(0, 1, 100),
                'metadata': {'station': 'TEST01', 'sampling_rate': 100}
            },
            {
                'type': 'gravity',
                'name': 'test_gravity',
                'data': np.random.normal(0, 0.1, 100),
                'metadata': {'station': 'GRAV01', 'precision': 0.001}
            },
            {
                'type': 'deformation',
                'name': 'test_deformation',
                'data': np.random.normal(0, 5, 100),
                'metadata': {'station': 'GPS01', 'coordinate': 'vertical'}
            }
        ]
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert isinstance(self.pipeline, AdvancedDataIntegrationPipeline)
        assert hasattr(self.pipeline, 'data_layers')
        assert hasattr(self.pipeline, 'quality_assessor')
        assert hasattr(self.pipeline, 'anomaly_detector')
        assert hasattr(self.pipeline, 'feature_engineer')
    
    @pytest.mark.asyncio
    async def test_real_time_volcanic_enhancement(self):
        """Test real-time volcanic enhancement main entry point"""
        base_data = np.random.normal(0, 1, 300)
        
        enhanced_data = await self.pipeline.get_real_time_volcanic_enhancement(base_data)
        
        assert len(enhanced_data) == len(base_data)
        assert np.all(np.isfinite(enhanced_data))
        
        # Enhancement should modify the data
        assert not np.array_equal(enhanced_data, base_data)
    
    def test_layer_fusion_weight_calculation(self):
        """Test layer fusion weight calculation"""
        # Create layer with high quality
        high_quality_layer = DataLayer(
            name="high_quality",
            data_type=DataType.SEISMIC,
            data=np.random.normal(0, 1, 100),
            quality_metrics=QualityMetrics(
                completeness=0.95, accuracy=0.9, precision=0.85,
                consistency=0.8, timeliness=0.9, overall_score=0.88,
                anomaly_count=2, missing_percentage=5.0, outlier_percentage=1.0
            ),
            created_at=datetime.now()
        )
        
        weight = self.pipeline._calculate_layer_fusion_weight(high_quality_layer)
        
        assert 0.1 <= weight <= 2.0
        assert weight > 1.0  # Should be above baseline for high quality seismic data
    
    def test_array_interpolation(self):
        """Test array interpolation functionality"""
        # Test upsampling
        short_array = np.array([1, 2, 3, 4, 5])
        interpolated = self.pipeline._interpolate_array(short_array, 10)
        
        assert len(interpolated) == 10
        assert np.all(np.isfinite(interpolated))
        
        # Test downsampling
        long_array = np.random.normal(0, 1, 100)
        interpolated = self.pipeline._interpolate_array(long_array, 50)
        
        assert len(interpolated) == 50
        assert np.all(np.isfinite(interpolated))
    
    def test_array_interpolation_with_nans(self):
        """Test array interpolation with NaN values"""
        array_with_nans = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8])
        interpolated = self.pipeline._interpolate_array(array_with_nans, 10)
        
        assert len(interpolated) == 10
        # Should handle NaN values gracefully
        assert not np.all(np.isnan(interpolated))


class TestQualityAssessmentEngine:
    """Test suite for QualityAssessmentEngine"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.quality_engine = QualityAssessmentEngine()
    
    @pytest.mark.asyncio
    async def test_quality_assessment_perfect_data(self):
        """Test quality assessment with perfect data"""
        # Create perfect synthetic data
        perfect_data = np.sin(np.linspace(0, 10, 100))
        
        layer = DataLayer(
            name="perfect_test",
            data_type=DataType.SEISMIC,
            data=perfect_data,
            created_at=datetime.now()
        )
        
        quality_metrics = await self.quality_engine.assess_layer_quality(layer)
        
        assert quality_metrics.completeness == 1.0  # No missing data
        assert quality_metrics.overall_score > 0.5  # Should be reasonably high
        assert quality_metrics.missing_percentage == 0.0
        assert quality_metrics.anomaly_count >= 0
    
    @pytest.mark.asyncio
    async def test_quality_assessment_poor_data(self):
        """Test quality assessment with poor quality data"""
        # Create poor quality data
        poor_data = np.random.normal(0, 1, 100)
        poor_data[::5] = np.nan  # 20% missing data
        poor_data[10] = 1000  # Large outlier
        poor_data[20] = -1000  # Another outlier
        
        layer = DataLayer(
            name="poor_test",
            data_type=DataType.GRAVITY,
            data=poor_data,
            created_at=datetime.now() - timedelta(hours=48)  # Old data
        )
        
        quality_metrics = await self.quality_engine.assess_layer_quality(layer)
        
        assert quality_metrics.completeness < 1.0  # Has missing data
        assert quality_metrics.missing_percentage > 0
        assert quality_metrics.anomaly_count > 0  # Should detect outliers
        assert quality_metrics.overall_score < 0.8  # Should be lower for poor data
    
    @pytest.mark.asyncio
    async def test_quality_assessment_empty_data(self):
        """Test quality assessment with empty/None data"""
        layer = DataLayer(
            name="empty_test",
            data_type=DataType.GEOCHEMICAL,
            data=None
        )
        
        quality_metrics = await self.quality_engine.assess_layer_quality(layer)
        
        assert quality_metrics.completeness == 0.0
        assert quality_metrics.overall_score == 0.0
        assert quality_metrics.missing_percentage == 100.0
    
    def test_outlier_detection(self):
        """Test outlier detection functionality"""
        # Create data with known outliers
        normal_data = np.random.normal(0, 1, 100)
        data_with_outliers = normal_data.copy()
        data_with_outliers[10] = 10  # Clear outlier
        data_with_outliers[50] = -10  # Another outlier
        
        outlier_mask = self.quality_engine._detect_outliers(data_with_outliers)
        
        assert isinstance(outlier_mask, np.ndarray)
        assert outlier_mask.dtype == bool
        assert np.sum(outlier_mask) >= 2  # Should detect at least the 2 outliers
        assert outlier_mask[10] == True  # Should detect first outlier
        assert outlier_mask[50] == True  # Should detect second outlier


class TestAnomalyDetectionEngine:
    """Test suite for AnomalyDetectionEngine"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.anomaly_engine = AnomalyDetectionEngine()
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_normal_data(self):
        """Test anomaly detection with normal data"""
        normal_data = np.random.normal(0, 1, 200)
        
        anomalies = await self.anomaly_engine.detect_anomalies(normal_data)
        
        assert isinstance(anomalies, dict)
        assert 'statistical_outliers' in anomalies
        assert 'temporal_anomalies' in anomalies
        assert 'pattern_anomalies' in anomalies
        assert 'instrument_anomalies' in anomalies
        
        # Normal data should have few anomalies
        assert len(anomalies['statistical_outliers']) < len(normal_data) * 0.1
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_with_outliers(self):
        """Test anomaly detection with known outliers"""
        data_with_outliers = np.random.normal(0, 1, 100)
        data_with_outliers[25] = 20  # Large positive outlier
        data_with_outliers[75] = -20  # Large negative outlier
        
        anomalies = await self.anomaly_engine.detect_anomalies(data_with_outliers)
        
        # Should detect the outliers
        outlier_indices = anomalies['statistical_outliers']
        assert 25 in outlier_indices or 75 in outlier_indices
    
    def test_statistical_outlier_detection(self):
        """Test statistical outlier detection method"""
        # Create data with known outliers
        clean_data = np.random.normal(0, 1, 100)
        data_with_outliers = clean_data.copy()
        data_with_outliers[30] = 15  # Clear outlier
        
        outlier_indices = self.anomaly_engine._detect_statistical_outliers(data_with_outliers)
        
        assert isinstance(outlier_indices, np.ndarray)
        assert len(outlier_indices) > 0
        assert 30 in outlier_indices  # Should detect the inserted outlier


class TestAutomatedFeatureEngineer:
    """Test suite for AutomatedFeatureEngineer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.feature_engineer = AutomatedFeatureEngineer()
    
    @pytest.mark.asyncio
    async def test_feature_engineering_basic(self):
        """Test basic feature engineering functionality"""
        # Create signal with known characteristics
        t = np.linspace(0, 10, 200)
        signal = np.sin(2 * np.pi * t) + 0.1 * np.random.normal(0, 1, 200)
        
        features = await self.feature_engineer.engineer_features(signal)
        
        assert len(features) == len(signal)
        assert np.all(np.isfinite(features))
        
        # Features should capture some signal characteristics
        assert np.std(features) > 0  # Should have variation
    
    def test_temporal_feature_extraction(self):
        """Test temporal feature extraction"""
        # Create trending data
        trend_data = np.linspace(0, 10, 100) + 0.1 * np.random.normal(0, 1, 100)
        
        temporal_features = self.feature_engineer._extract_temporal_features(trend_data)
        
        assert len(temporal_features) == len(trend_data)
        assert np.all(np.isfinite(temporal_features))
    
    def test_statistical_feature_extraction(self):
        """Test statistical feature extraction"""
        # Create data with varying statistics
        varying_data = np.concatenate([
            np.random.normal(0, 1, 50),
            np.random.normal(0, 3, 50)  # Different variance
        ])
        
        statistical_features = self.feature_engineer._extract_statistical_features(varying_data)
        
        assert len(statistical_features) == len(varying_data)
        assert np.all(np.isfinite(statistical_features))
    
    def test_spectral_feature_extraction(self):
        """Test spectral feature extraction"""
        # Create signal with known frequency
        t = np.linspace(0, 10, 200)
        freq_signal = np.sin(2 * np.pi * 2 * t)  # 2 Hz signal
        
        spectral_features = self.feature_engineer._extract_spectral_features(freq_signal)
        
        assert len(spectral_features) == len(freq_signal)
        assert np.all(np.isfinite(spectral_features))


class TestDataTypes:
    """Test suite for data type enumerations and structures"""
    
    def test_data_type_enum(self):
        """Test DataType enumeration"""
        assert DataType.SEISMIC.value == "seismic"
        assert DataType.VOLCANIC.value == "volcanic"
        assert DataType.GEOCHEMICAL.value == "geochemical"
        
        # Test all enum values are strings
        for data_type in DataType:
            assert isinstance(data_type.value, str)
    
    def test_processing_method_enum(self):
        """Test ProcessingMethod enumeration"""
        assert ProcessingMethod.KRIGING.value == "kriging"
        assert ProcessingMethod.GAUSSIAN_PROCESS.value == "gaussian_process"
        
        # Test all enum values are strings
        for method in ProcessingMethod:
            assert isinstance(method.value, str)
    
    def test_quality_metrics_structure(self):
        """Test QualityMetrics dataclass"""
        metrics = QualityMetrics(
            completeness=0.95,
            accuracy=0.90,
            precision=0.85,
            consistency=0.80,
            timeliness=0.75,
            overall_score=0.85,
            anomaly_count=5,
            missing_percentage=5.0,
            outlier_percentage=2.0
        )
        
        assert metrics.completeness == 0.95
        assert metrics.overall_score == 0.85
        assert metrics.anomaly_count == 5
    
    def test_geographic_bounds_structure(self):
        """Test GeographicBounds dataclass"""
        bounds = GeographicBounds(
            min_lon=-120.0,
            max_lon=-119.0,
            min_lat=45.0,
            max_lat=46.0,
            min_elevation=0.0,
            max_elevation=3000.0
        )
        
        assert bounds.min_lon == -120.0
        assert bounds.max_elevation == 3000.0
    
    def test_temporal_range_structure(self):
        """Test TemporalRange dataclass"""
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)
        resolution = timedelta(minutes=1)
        
        temporal_range = TemporalRange(
            start_time=start_time,
            end_time=end_time,
            resolution=resolution
        )
        
        assert temporal_range.start_time == start_time
        assert temporal_range.end_time == end_time
        assert temporal_range.resolution == resolution


class TestIntegrationScenarios:
    """Integration tests for realistic volcanic monitoring scenarios"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        self.pipeline = AdvancedDataIntegrationPipeline()
    
    @pytest.mark.asyncio
    async def test_volcanic_eruption_scenario(self):
        """Test pipeline with simulated volcanic eruption data"""
        # Simulate pre-eruption, eruption, and post-eruption data
        time_points = 1000
        
        # Pre-eruption: gradual increase
        pre_eruption = np.linspace(0, 2, 300) + 0.1 * np.random.normal(0, 1, 300)
        
        # Eruption: sharp spike
        eruption = 10 * np.exp(-np.linspace(0, 5, 200)) + np.random.normal(0, 0.5, 200)
        
        # Post-eruption: gradual decay
        post_eruption = 2 * np.exp(-np.linspace(0, 3, 500)) + 0.1 * np.random.normal(0, 1, 500)
        
        # Combine phases
        volcanic_signal = np.concatenate([pre_eruption, eruption, post_eruption])
        
        # Apply real-time enhancement
        enhanced_signal = await self.pipeline.get_real_time_volcanic_enhancement(volcanic_signal)
        
        assert len(enhanced_signal) == len(volcanic_signal)
        assert np.all(np.isfinite(enhanced_signal))
        
        # Enhanced signal should capture the eruption pattern
        eruption_start = len(pre_eruption)
        eruption_end = eruption_start + len(eruption)
        
        # Check that eruption period has higher values
        eruption_enhanced = enhanced_signal[eruption_start:eruption_end]
        pre_eruption_enhanced = enhanced_signal[:len(pre_eruption)]
        
        assert np.mean(np.abs(eruption_enhanced)) > np.mean(np.abs(pre_eruption_enhanced))
    
    @pytest.mark.asyncio
    async def test_multi_sensor_integration_scenario(self):
        """Test integration of multiple sensor types"""
        # Create correlated multi-sensor data
        base_signal = np.sin(np.linspace(0, 20, 500)) + 0.1 * np.random.normal(0, 1, 500)
        
        # Seismic data (amplified and with noise)
        seismic_data = 2 * base_signal + 0.2 * np.random.normal(0, 1, 500)
        
        # Deformation data (delayed and smoothed)
        deformation_data = np.roll(base_signal, 10) * 0.5
        deformation_data = pd.Series(deformation_data).rolling(window=5, center=True).mean().values
        deformation_data = np.nan_to_num(deformation_data)
        
        # GPS data (with trend)
        gps_data = base_signal + np.linspace(0, 1, 500) + 0.05 * np.random.normal(0, 1, 500)
        
        # Create data layers
        seismic_layer = DataLayer(
            name="seismic_station_01",
            data_type=DataType.SEISMIC,
            data=seismic_data,
            created_at=datetime.now()
        )
        
        deformation_layer = DataLayer(
            name="deformation_sensor_01",
            data_type=DataType.DEFORMATION,
            data=deformation_data,
            created_at=datetime.now()
        )
        
        gps_layer = DataLayer(
            name="gps_station_01",
            data_type=DataType.DEFORMATION,
            data=gps_data,
            created_at=datetime.now()
        )
        
        # Test individual enhancements
        seismic_enhanced = seismic_layer.get_real_time_volcanic_data(seismic_data)
        deformation_enhanced = deformation_layer.get_real_time_volcanic_data(deformation_data)
        gps_enhanced = gps_layer.get_real_time_volcanic_data(gps_data)
        
        # All should be enhanced successfully
        assert len(seismic_enhanced) == len(seismic_data)
        assert len(deformation_enhanced) == len(deformation_data)
        assert len(gps_enhanced) == len(gps_data)
        
        assert np.all(np.isfinite(seismic_enhanced))
        assert np.all(np.isfinite(deformation_enhanced))
        assert np.all(np.isfinite(gps_enhanced))
    
    def test_data_quality_degradation_scenario(self):
        """Test pipeline behavior with degrading data quality"""
        # Create data with progressively worse quality
        good_data = np.sin(np.linspace(0, 10, 200)) + 0.05 * np.random.normal(0, 1, 200)
        
        # Add missing data
        degraded_data = good_data.copy()
        degraded_data[50:60] = np.nan  # 10 missing points
        
        # Add outliers
        degraded_data[100] = 50  # Large outlier
        degraded_data[150] = -50  # Another outlier
        
        # Add noise
        degraded_data += 0.5 * np.random.normal(0, 1, 200)
        
        # Create layers with different quality levels
        good_layer = DataLayer(
            name="good_quality",
            data_type=DataType.SEISMIC,
            data=good_data,
            created_at=datetime.now()
        )
        
        degraded_layer = DataLayer(
            name="degraded_quality",
            data_type=DataType.SEISMIC,
            data=degraded_data,
            created_at=datetime.now() - timedelta(hours=12)  # Older data
        )
        
        # Test quality assessment
        good_enhanced = good_layer._apply_quality_assessment(good_data)
        degraded_enhanced = degraded_layer._apply_quality_assessment(degraded_data)
        
        # Good layer should have higher quality score
        assert good_layer.quality_metrics.overall_score > degraded_layer.quality_metrics.overall_score
        assert good_layer.quality_metrics.missing_percentage < degraded_layer.quality_metrics.missing_percentage
        assert good_layer.quality_metrics.anomaly_count <= degraded_layer.quality_metrics.anomaly_count


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])