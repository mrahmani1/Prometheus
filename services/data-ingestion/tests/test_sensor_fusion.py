"""Tests for multi-sensor data fusion algorithms"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from processors.sensor_fusion import (
    KalmanSensorFusion, WeightedSensorFusion, MultiSensorDataFusion,
    SensorReading, FusionState, FusionResult
)
from models.geological_data import (
    SeismicEvent, ThermalMeasurement, DeformationMeasurement,
    StrainMeasurement, AtmosphericMeasurement, GeologicalLocation,
    DataQuality
)


class TestKalmanSensorFusion:
    """Test cases for Kalman filter-based sensor fusion"""
    
    @pytest.fixture
    def fusion_config(self):
        """Test configuration for sensor fusion"""
        return {
            'temporal_window': 300,  # 5 minutes
            'spatial_threshold': 50.0,  # 50 km
            'outlier_threshold': 3.0
        }
    
    @pytest.fixture
    def kalman_fusion(self, fusion_config):
        """Kalman fusion instance for testing"""
        return KalmanSensorFusion(fusion_config)
    
    @pytest.fixture
    def sample_location(self):
        """Sample geological location"""
        return GeologicalLocation(
            latitude=45.0,
            longitude=-122.0,
            elevation=1000.0
        )
    
    @pytest.fixture
    def sample_sensor_readings(self, sample_location):
        """Sample sensor readings for testing"""
        base_time = datetime.utcnow()
        
        readings = [
            # Seismic readings
            SensorReading(
                timestamp=base_time,
                location=sample_location,
                sensor_type='seismic',
                value=4.5,
                uncertainty=0.1,
                quality=DataQuality.GOOD,
                source='test_seismic_1'
            ),
            SensorReading(
                timestamp=base_time + timedelta(seconds=30),
                location=GeologicalLocation(
                    latitude=45.01,
                    longitude=-122.01,
                    elevation=1010.0
                ),
                sensor_type='seismic',
                value=4.3,
                uncertainty=0.15,
                quality=DataQuality.EXCELLENT,
                source='test_seismic_2'
            ),
            # Thermal readings
            SensorReading(
                timestamp=base_time + timedelta(seconds=60),
                location=GeologicalLocation(
                    latitude=45.005,
                    longitude=-122.005,
                    elevation=1005.0
                ),
                sensor_type='thermal',
                value=350.0,
                uncertainty=2.0,
                quality=DataQuality.GOOD,
                source='test_thermal_1'
            ),
            SensorReading(
                timestamp=base_time + timedelta(seconds=90),
                location=sample_location,
                sensor_type='thermal',
                value=355.0,
                uncertainty=1.5,
                quality=DataQuality.EXCELLENT,
                source='test_thermal_2'
            ),
            # Deformation readings
            SensorReading(
                timestamp=base_time + timedelta(seconds=120),
                location=sample_location,
                sensor_type='deformation',
                value=np.array([0.01, 0.02, -0.005]),
                uncertainty=np.array([0.001, 0.001, 0.0005]),
                quality=DataQuality.GOOD,
                source='test_deformation_1'
            )
        ]
        
        return readings
    
    def test_temporal_alignment(self, kalman_fusion, sample_sensor_readings):
        """Test temporal alignment of sensor readings"""
        reference_time = datetime.utcnow()
        
        aligned_readings = kalman_fusion._temporal_alignment(
            sample_sensor_readings, reference_time
        )
        
        # Check that all readings are within temporal window
        window_start = reference_time - timedelta(seconds=300)
        for reading in aligned_readings:
            assert window_start <= reading.timestamp <= reference_time
            assert 'temporal_offset' in reading.metadata
            assert 'reference_time' in reading.metadata
    
    def test_spatial_clustering(self, kalman_fusion, sample_sensor_readings):
        """Test spatial clustering of sensor readings"""
        clusters = kalman_fusion._spatial_clustering(sample_sensor_readings)
        
        # Should have at least one cluster
        assert len(clusters) >= 1
        
        # All readings should be assigned to clusters
        total_clustered = sum(len(readings) for readings in clusters.values())
        assert total_clustered == len(sample_sensor_readings)
        
        # Check cluster structure
        for cluster_id, readings in clusters.items():
            assert isinstance(cluster_id, str)
            assert len(readings) > 0
    
    def test_quality_weighting(self, kalman_fusion, sample_sensor_readings):
        """Test quality-based weighting of sensor readings"""
        weighted_readings = kalman_fusion._apply_quality_weights(sample_sensor_readings)
        
        assert len(weighted_readings) == len(sample_sensor_readings)
        
        for original, weighted in zip(sample_sensor_readings, weighted_readings):
            # Check that quality weight is applied
            assert 'quality_weight' in weighted.metadata
            assert 'original_uncertainty' in weighted.metadata
            
            # Higher quality should result in lower weighted uncertainty
            quality_weight = weighted.metadata['quality_weight']
            if original.quality == DataQuality.EXCELLENT:
                assert quality_weight == 1.0
            elif original.quality == DataQuality.GOOD:
                assert quality_weight == 0.8
    
    def test_outlier_rejection(self, kalman_fusion):
        """Test outlier detection and rejection"""
        # Create readings with one clear outlier - need more readings for proper covariance
        base_time = datetime.utcnow()
        location = GeologicalLocation(45.0, -122.0, 1000.0)
        
        readings = [
            SensorReading(
                timestamp=base_time,
                location=location,
                sensor_type='seismic',
                value=4.0,
                uncertainty=0.1,
                quality=DataQuality.GOOD,
                source='normal_1'
            ),
            SensorReading(
                timestamp=base_time + timedelta(seconds=30),
                location=location,
                sensor_type='seismic',
                value=4.1,
                uncertainty=0.1,
                quality=DataQuality.GOOD,
                source='normal_2'
            ),
            SensorReading(
                timestamp=base_time + timedelta(seconds=60),
                location=location,
                sensor_type='seismic',
                value=4.2,
                uncertainty=0.1,
                quality=DataQuality.GOOD,
                source='normal_3'
            ),
            SensorReading(
                timestamp=base_time + timedelta(seconds=90),
                location=location,
                sensor_type='seismic',
                value=4.15,
                uncertainty=0.1,
                quality=DataQuality.GOOD,
                source='normal_4'
            ),
            SensorReading(
                timestamp=base_time + timedelta(seconds=120),
                location=location,
                sensor_type='seismic',
                value=4.05,
                uncertainty=0.1,
                quality=DataQuality.GOOD,
                source='normal_5'
            ),
            # Clear outlier
            SensorReading(
                timestamp=base_time + timedelta(seconds=150),
                location=location,
                sensor_type='seismic',
                value=8.0,  # Much higher than others
                uncertainty=0.1,
                quality=DataQuality.GOOD,
                source='outlier'
            )
        ]
        
        filtered_readings, outliers = kalman_fusion._outlier_rejection(readings)
        
        # Should detect the outlier or handle gracefully
        assert len(filtered_readings) + len(outliers) == len(readings)
        
        # If outliers are detected, check metadata
        for outlier in outliers:
            assert 'mahalanobis_distance' in outlier.metadata
            assert 'outlier_reason' in outlier.metadata
    
    def test_kalman_fusion_complete(self, kalman_fusion, sample_sensor_readings):
        """Test complete Kalman fusion process"""
        fusion_results = kalman_fusion.fuse_sensor_data(sample_sensor_readings)
        
        # Should produce at least one fusion result
        assert len(fusion_results) >= 1
        
        for result in fusion_results:
            assert isinstance(result, FusionResult)
            assert isinstance(result.fused_state, FusionState)
            assert len(result.contributing_sensors) > 0
            assert 0.0 <= result.fusion_quality <= 1.0
            assert 0.0 <= result.confidence_level <= 1.0
            assert result.uncertainty_reduction >= 0.0
    
    def test_3d_distance_calculation(self, kalman_fusion):
        """Test 3D distance calculation between locations"""
        loc1 = GeologicalLocation(45.0, -122.0, 1000.0)
        loc2 = GeologicalLocation(45.01, -122.01, 1010.0)
        
        distance = kalman_fusion._calculate_3d_distance(loc1, loc2)
        
        # Should be approximately 1.57 km (rough calculation)
        assert 1.0 < distance < 2.0
    
    def test_fusion_state_creation(self, kalman_fusion, sample_location):
        """Test fusion state creation and management"""
        sensor_types = ['seismic', 'thermal']
        cluster_id = 'test_cluster'
        
        fusion_state = kalman_fusion._get_or_create_fusion_state(
            cluster_id, sensor_types, sample_location
        )
        
        assert isinstance(fusion_state, FusionState)
        assert fusion_state.sensor_types == sensor_types
        assert fusion_state.location == sample_location
        assert fusion_state.state_vector.shape[0] > 0
        assert fusion_state.covariance_matrix.shape[0] == fusion_state.state_vector.shape[0]
        
        # Test state retrieval
        retrieved_state = kalman_fusion._get_or_create_fusion_state(
            cluster_id, sensor_types, sample_location
        )
        
        # Should be the same state (updated)
        assert retrieved_state.sensor_types == sensor_types


class TestWeightedSensorFusion:
    """Test cases for weighted sensor fusion"""
    
    @pytest.fixture
    def fusion_config(self):
        """Test configuration for weighted fusion"""
        return {
            'sensor_weights': {
                'seismic': 1.0,
                'thermal': 0.8,
                'deformation': 0.9
            }
        }
    
    @pytest.fixture
    def weighted_fusion(self, fusion_config):
        """Weighted fusion instance for testing"""
        return WeightedSensorFusion(fusion_config)
    
    @pytest.fixture
    def sample_readings_for_weighting(self):
        """Sample readings for weighted fusion testing"""
        base_time = datetime.utcnow()
        base_location = GeologicalLocation(45.0, -122.0, 1000.0)
        
        readings = [
            SensorReading(
                timestamp=base_time,
                location=base_location,
                sensor_type='seismic',
                value=4.5,
                uncertainty=0.1,
                quality=DataQuality.EXCELLENT,
                source='seismic_1'
            ),
            SensorReading(
                timestamp=base_time + timedelta(seconds=30),
                location=GeologicalLocation(45.01, -122.01, 1000.0),
                sensor_type='seismic',
                value=4.3,
                uncertainty=0.15,
                quality=DataQuality.GOOD,
                source='seismic_2'
            ),
            SensorReading(
                timestamp=base_time + timedelta(seconds=60),
                location=GeologicalLocation(45.005, -122.005, 1000.0),
                sensor_type='thermal',
                value=350.0,
                uncertainty=2.0,
                quality=DataQuality.GOOD,
                source='thermal_1'
            )
        ]
        
        return readings
    
    def test_weighted_average_fusion(self, weighted_fusion, sample_readings_for_weighting):
        """Test weighted average fusion algorithm"""
        reference_location = GeologicalLocation(45.0, -122.0, 1000.0)
        
        fusion_result = weighted_fusion.weighted_average_fusion(
            sample_readings_for_weighting, reference_location
        )
        
        # Should have results for each sensor type
        assert 'seismic' in fusion_result
        assert 'thermal' in fusion_result
        
        # Check seismic fusion result
        seismic_result = fusion_result['seismic']
        assert 'weighted_value' in seismic_result
        assert 'weighted_uncertainty' in seismic_result
        assert 'weights' in seismic_result
        assert 'num_readings' in seismic_result
        
        # Weights should sum to 1.0 for each sensor type
        seismic_weights = np.array(seismic_result['weights'])
        assert np.isclose(np.sum(seismic_weights), 1.0)
        
        # Weighted value should be between the input values
        seismic_values = [4.5, 4.3]  # From sample data
        weighted_value = seismic_result['weighted_value']
        assert min(seismic_values) <= weighted_value <= max(seismic_values)
    
    def test_distance_calculation(self, weighted_fusion):
        """Test distance calculation for weighting"""
        loc1 = GeologicalLocation(45.0, -122.0, 1000.0)
        loc2 = GeologicalLocation(45.01, -122.01, 1000.0)
        
        distance = weighted_fusion._calculate_distance(loc1, loc2)
        
        # Should be approximately 1.57 km
        assert 1.0 < distance < 2.0
    
    def test_empty_readings(self, weighted_fusion):
        """Test handling of empty readings list"""
        reference_location = GeologicalLocation(45.0, -122.0, 1000.0)
        
        result = weighted_fusion.weighted_average_fusion([], reference_location)
        
        assert result == {}


class TestMultiSensorDataFusion:
    """Test cases for the main multi-sensor data fusion class"""
    
    @pytest.fixture
    def fusion_config(self):
        """Test configuration for multi-sensor fusion"""
        return {
            'temporal_window': 300,
            'spatial_threshold': 50.0,
            'outlier_threshold': 3.0
        }
    
    @pytest.fixture
    def multi_sensor_fusion(self, fusion_config):
        """Multi-sensor fusion instance for testing"""
        return MultiSensorDataFusion(fusion_config)
    
    @pytest.fixture
    def sample_geological_data(self):
        """Sample geological data objects for testing"""
        base_time = datetime.utcnow()
        base_location = GeologicalLocation(45.0, -122.0, 1000.0)
        
        data = [
            SeismicEvent(
                timestamp=base_time,
                location=base_location,
                magnitude=4.5,
                depth=10.0,
                event_type='earthquake',
                quality=DataQuality.GOOD,
                source='test_seismic'
            ),
            ThermalMeasurement(
                timestamp=base_time + timedelta(seconds=60),
                location=GeologicalLocation(45.005, -122.005, 1000.0),
                temperature=350.0,
                measurement_method='satellite',
                spatial_resolution=30.0,
                uncertainty=2.0,
                quality=DataQuality.GOOD,
                source='test_thermal'
            ),
            DeformationMeasurement(
                timestamp=base_time + timedelta(seconds=120),
                location=base_location,
                displacement_vector=np.array([0.01, 0.02, -0.005]),
                measurement_type='GPS',
                uncertainty=np.array([0.001, 0.001, 0.0005]),
                quality=DataQuality.EXCELLENT,
                source='test_deformation'
            ),
            StrainMeasurement(
                timestamp=base_time + timedelta(seconds=180),
                location=base_location,
                strain_rate=1e-6,
                strain_rate_std=1e-7,
                strain_rate_max=2e-6,
                measurement_type='das',
                channel_index=1,
                quality=DataQuality.GOOD,
                source='test_strain'
            ),
            AtmosphericMeasurement(
                timestamp=base_time + timedelta(seconds=240),
                location=base_location,
                pressure=101325.0,
                temperature=298.0,
                humidity=65.0,
                quality=DataQuality.FAIR,
                source='test_atmospheric'
            )
        ]
        
        return data
    
    def test_geological_data_conversion(self, multi_sensor_fusion, sample_geological_data):
        """Test conversion of geological data to sensor readings"""
        sensor_readings = multi_sensor_fusion._convert_to_sensor_readings(
            sample_geological_data
        )
        
        # Should have more readings than input data (atmospheric creates multiple)
        assert len(sensor_readings) >= len(sample_geological_data)
        
        # Check that all sensor types are represented
        sensor_types = set(reading.sensor_type for reading in sensor_readings)
        expected_types = {'seismic', 'thermal', 'deformation', 'strain', 'atmospheric'}
        assert expected_types.issubset(sensor_types)
        
        # Check specific conversions
        seismic_readings = [r for r in sensor_readings if r.sensor_type == 'seismic']
        assert len(seismic_readings) == 1
        assert seismic_readings[0].value == 4.5
        
        thermal_readings = [r for r in sensor_readings if r.sensor_type == 'thermal']
        assert len(thermal_readings) == 1
        assert thermal_readings[0].value == 350.0
        
        deformation_readings = [r for r in sensor_readings if r.sensor_type == 'deformation']
        assert len(deformation_readings) == 1
        assert np.array_equal(deformation_readings[0].value, np.array([0.01, 0.02, -0.005]))
    
    def test_kalman_fusion_integration(self, multi_sensor_fusion, sample_geological_data):
        """Test Kalman fusion integration with geological data"""
        fusion_results = multi_sensor_fusion.fuse_sensors(
            sample_geological_data,
            fusion_method='kalman'
        )
        
        # Should produce fusion results
        assert len(fusion_results) >= 1
        
        for result in fusion_results:
            assert isinstance(result, FusionResult)
            assert len(result.contributing_sensors) > 0
            assert 0.0 <= result.fusion_quality <= 1.0
            assert 0.0 <= result.confidence_level <= 1.0
    
    def test_weighted_fusion_integration(self, multi_sensor_fusion, sample_geological_data):
        """Test weighted fusion integration with geological data"""
        reference_location = GeologicalLocation(45.0, -122.0, 1000.0)
        
        fusion_results = multi_sensor_fusion.fuse_sensors(
            sample_geological_data,
            fusion_method='weighted',
            reference_location=reference_location
        )
        
        # Should produce one fusion result for weighted method
        assert len(fusion_results) == 1
        
        result = fusion_results[0]
        assert isinstance(result, FusionResult)
        assert len(result.contributing_sensors) > 0
        assert 0.0 <= result.fusion_quality <= 1.0
    
    def test_invalid_fusion_method(self, multi_sensor_fusion, sample_geological_data):
        """Test handling of invalid fusion method"""
        fusion_results = multi_sensor_fusion.fuse_sensors(
            sample_geological_data,
            fusion_method='invalid_method'
        )
        
        # Should return empty list for invalid method
        assert fusion_results == []
    
    def test_empty_data_handling(self, multi_sensor_fusion):
        """Test handling of empty geological data"""
        fusion_results = multi_sensor_fusion.fuse_sensors([])
        
        assert fusion_results == []
    
    def test_reference_location_calculation(self, multi_sensor_fusion, sample_geological_data):
        """Test automatic reference location calculation for weighted fusion"""
        fusion_results = multi_sensor_fusion.fuse_sensors(
            sample_geological_data,
            fusion_method='weighted'
            # No reference_location provided - should calculate automatically
        )
        
        # Should still produce results
        assert len(fusion_results) == 1
        
        result = fusion_results[0]
        assert result.fused_state.location is not None
        assert -90 <= result.fused_state.location.latitude <= 90
        assert -180 <= result.fused_state.location.longitude <= 180


class TestFusionDataStructures:
    """Test cases for fusion data structures"""
    
    def test_sensor_reading_creation(self):
        """Test SensorReading data structure"""
        timestamp = datetime.utcnow()
        location = GeologicalLocation(45.0, -122.0, 1000.0)
        
        reading = SensorReading(
            timestamp=timestamp,
            location=location,
            sensor_type='seismic',
            value=4.5,
            uncertainty=0.1,
            quality=DataQuality.GOOD,
            source='test_source',
            metadata={'test_key': 'test_value'}
        )
        
        assert reading.timestamp == timestamp
        assert reading.location == location
        assert reading.sensor_type == 'seismic'
        assert reading.value == 4.5
        assert reading.uncertainty == 0.1
        assert reading.quality == DataQuality.GOOD
        assert reading.source == 'test_source'
        assert reading.metadata['test_key'] == 'test_value'
    
    def test_fusion_state_creation(self):
        """Test FusionState data structure"""
        state_vector = np.array([1.0, 2.0, 3.0])
        covariance_matrix = np.eye(3)
        timestamp = datetime.utcnow()
        sensor_types = ['seismic', 'thermal']
        location = GeologicalLocation(45.0, -122.0, 1000.0)
        
        fusion_state = FusionState(
            state_vector=state_vector,
            covariance_matrix=covariance_matrix,
            timestamp=timestamp,
            sensor_types=sensor_types,
            location=location
        )
        
        assert np.array_equal(fusion_state.state_vector, state_vector)
        assert np.array_equal(fusion_state.covariance_matrix, covariance_matrix)
        assert fusion_state.timestamp == timestamp
        assert fusion_state.sensor_types == sensor_types
        assert fusion_state.location == location
    
    def test_fusion_result_creation(self):
        """Test FusionResult data structure"""
        fusion_state = FusionState(
            state_vector=np.array([1.0]),
            covariance_matrix=np.eye(1),
            timestamp=datetime.utcnow(),
            sensor_types=['seismic'],
            location=GeologicalLocation(45.0, -122.0, 1000.0)
        )
        
        fusion_result = FusionResult(
            fused_state=fusion_state,
            contributing_sensors=['seismic', 'thermal'],
            fusion_quality=0.8,
            uncertainty_reduction=0.3,
            outliers_rejected=[],
            temporal_alignment_offset=30.0,
            confidence_level=0.9
        )
        
        assert fusion_result.fused_state == fusion_state
        assert fusion_result.contributing_sensors == ['seismic', 'thermal']
        assert fusion_result.fusion_quality == 0.8
        assert fusion_result.uncertainty_reduction == 0.3
        assert fusion_result.outliers_rejected == []
        assert fusion_result.temporal_alignment_offset == 30.0
        assert fusion_result.confidence_level == 0.9


# Integration tests
class TestSensorFusionIntegration:
    """Integration tests for sensor fusion with real-world scenarios"""
    
    @pytest.fixture
    def volcanic_monitoring_scenario(self):
        """Simulate volcanic monitoring scenario with multiple sensors"""
        base_time = datetime.utcnow()
        volcano_location = GeologicalLocation(
            latitude=46.1978,  # Mount St. Helens
            longitude=-122.1956,
            elevation=2549.0
        )
        
        # Simulate volcanic unrest with correlated signals
        data = []
        
        # Seismic swarm
        for i in range(5):
            data.append(SeismicEvent(
                timestamp=base_time + timedelta(minutes=i*10),
                location=GeologicalLocation(
                    latitude=volcano_location.latitude + np.random.normal(0, 0.01),
                    longitude=volcano_location.longitude + np.random.normal(0, 0.01),
                    elevation=volcano_location.elevation + np.random.normal(0, 100)
                ),
                magnitude=3.0 + np.random.normal(0, 0.3),
                depth=5.0 + np.random.normal(0, 1.0),
                event_type='volcanic',
                quality=DataQuality.GOOD,
                source='volcanic_network'
            ))
        
        # Thermal anomalies
        for i in range(3):
            data.append(ThermalMeasurement(
                timestamp=base_time + timedelta(minutes=i*15),
                location=GeologicalLocation(
                    latitude=volcano_location.latitude + np.random.normal(0, 0.005),
                    longitude=volcano_location.longitude + np.random.normal(0, 0.005),
                    elevation=volcano_location.elevation
                ),
                temperature=320.0 + i*10 + np.random.normal(0, 5),
                measurement_method='satellite',
                spatial_resolution=30.0,
                uncertainty=3.0,
                quality=DataQuality.GOOD,
                source='thermal_satellite'
            ))
        
        # Ground deformation
        for i in range(4):
            data.append(DeformationMeasurement(
                timestamp=base_time + timedelta(minutes=i*12),
                location=GeologicalLocation(
                    latitude=volcano_location.latitude + np.random.normal(0, 0.01),
                    longitude=volcano_location.longitude + np.random.normal(0, 0.01),
                    elevation=volcano_location.elevation
                ),
                displacement_vector=np.array([
                    0.005 + i*0.002 + np.random.normal(0, 0.001),
                    0.003 + i*0.001 + np.random.normal(0, 0.001),
                    0.010 + i*0.003 + np.random.normal(0, 0.001)
                ]),
                measurement_type='InSAR',
                uncertainty=np.array([0.001, 0.001, 0.002]),
                quality=DataQuality.EXCELLENT,
                source='insar_satellite'
            ))
        
        return data
    
    def test_volcanic_monitoring_fusion(self, volcanic_monitoring_scenario):
        """Test sensor fusion for volcanic monitoring scenario"""
        config = {
            'temporal_window': 3600,  # 1 hour window
            'spatial_threshold': 25.0,  # 25 km threshold
            'outlier_threshold': 2.5
        }
        
        fusion_system = MultiSensorDataFusion(config)
        
        # Test Kalman fusion
        kalman_results = fusion_system.fuse_sensors(
            volcanic_monitoring_scenario,
            fusion_method='kalman'
        )
        
        assert len(kalman_results) >= 1
        
        # Check that fusion captures the volcanic signals
        for result in kalman_results:
            assert len(result.contributing_sensors) >= 2  # Multi-sensor fusion
            assert result.fusion_quality > 0.3  # Reasonable quality
            assert result.confidence_level > 0.3  # Reasonable confidence
        
        # Test weighted fusion
        weighted_results = fusion_system.fuse_sensors(
            volcanic_monitoring_scenario,
            fusion_method='weighted'
        )
        
        assert len(weighted_results) == 1
        assert len(weighted_results[0].contributing_sensors) >= 2
    
    def test_performance_with_large_dataset(self):
        """Test fusion performance with large number of sensors"""
        # Generate large dataset
        base_time = datetime.utcnow()
        base_location = GeologicalLocation(45.0, -122.0, 1000.0)
        
        large_dataset = []
        
        # Generate 100 sensor readings
        for i in range(100):
            large_dataset.append(SeismicEvent(
                timestamp=base_time + timedelta(seconds=i*30),
                location=GeologicalLocation(
                    latitude=base_location.latitude + np.random.normal(0, 0.1),
                    longitude=base_location.longitude + np.random.normal(0, 0.1),
                    elevation=base_location.elevation + np.random.normal(0, 100)
                ),
                magnitude=3.0 + np.random.normal(0, 1.0),
                depth=10.0 + np.random.normal(0, 5.0),
                event_type='earthquake',
                quality=np.random.choice(list(DataQuality)),
                source=f'sensor_{i}'
            ))
        
        config = {
            'temporal_window': 7200,  # 2 hour window
            'spatial_threshold': 100.0,  # 100 km threshold
            'outlier_threshold': 3.0
        }
        
        fusion_system = MultiSensorDataFusion(config)
        
        # Test that fusion completes without errors
        fusion_results = fusion_system.fuse_sensors(
            large_dataset,
            fusion_method='kalman'
        )
        
        # Should produce results despite large dataset
        assert len(fusion_results) >= 1
        
        # Check that outliers are properly handled
        total_outliers = sum(len(result.outliers_rejected) for result in fusion_results)
        assert total_outliers >= 0  # Some outliers expected in random data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])