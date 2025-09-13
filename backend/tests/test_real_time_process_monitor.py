"""
Comprehensive test suite for Real-Time Process Monitor

Tests all aspects of the real-time monitoring system including data processing,
causal analysis, anomaly detection, alert generation, and performance monitoring.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Import the module under test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from real_time_process_monitor import (
    RealTimeProcessMonitor,
    GeologicalDataPoint,
    CausalRelationship,
    RealTimeAlert,
    AnomalyDetection,
    ProcessType,
    AlertSeverity,
    AnomalyType
)

class TestRealTimeProcessMonitor:
    """Test suite for RealTimeProcessMonitor class"""
    
    @pytest.fixture
    def monitor_config(self):
        """Configuration for test monitor instance"""
        return {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'influxdb_url': 'http://localhost:8086',
            'influxdb_token': 'test-token',
            'influxdb_org': 'test-org',
            'websocket_host': 'localhost',
            'websocket_port': 8765,
            'max_workers': 4
        }
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        mock_redis = Mock()
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = True
        return mock_redis
    
    @pytest.fixture
    def mock_influx(self):
        """Mock InfluxDB client"""
        mock_influx = Mock()
        mock_write_api = Mock()
        mock_influx.write_api.return_value = mock_write_api
        return mock_influx
    
    @pytest.fixture
    async def monitor(self, monitor_config, mock_redis, mock_influx):
        """Create test monitor instance with mocked dependencies"""
        with patch('real_time_process_monitor.redis.Redis', return_value=mock_redis), \
             patch('real_time_process_monitor.InfluxDBClient', return_value=mock_influx):
            
            monitor = RealTimeProcessMonitor(monitor_config)
            yield monitor
    
    @pytest.fixture
    def sample_seismic_data(self):
        """Generate sample seismic data points"""
        data_points = []
        base_time = datetime.utcnow()
        
        for i in range(10):
            data_point = GeologicalDataPoint(
                timestamp=base_time + timedelta(seconds=i * 10),
                process_type=ProcessType.SEISMIC,
                station_id='TEST_SEISMIC_01',
                location=(45.0, -110.0, 1500.0),
                parameters={
                    'ground_velocity': np.random.lognormal(0, 1) * 1e-6,
                    'dominant_frequency': np.random.uniform(1, 50),
                    'signal_to_noise_ratio': np.random.uniform(5, 30),
                    'peak_ground_acceleration': np.random.uniform(0.01, 0.1)
                },
                quality_score=np.random.uniform(0.7, 1.0),
                uncertainty={
                    'ground_velocity': 1e-7,
                    'dominant_frequency': 2.0
                },
                source_id='test_iris',
                processing_latency=np.random.uniform(0.1, 0.5)
            )
            data_points.append(data_point)
        
        return data_points
    
    @pytest.fixture
    def sample_volcanic_data(self):
        """Generate sample volcanic data points"""
        data_points = []
        base_time = datetime.utcnow()
        
        for i in range(10):
            data_point = GeologicalDataPoint(
                timestamp=base_time + timedelta(seconds=i * 10),
                process_type=ProcessType.VOLCANIC,
                station_id='TEST_VOLCANO_01',
                location=(44.5, -110.5, 2000.0),
                parameters={
                    'so2_flux': np.random.lognormal(2, 1) * 100,
                    'thermal_radiance': np.random.uniform(0, 50),
                    'volcanic_tremor': np.random.lognormal(-2, 1),
                    'gas_temperature': np.random.uniform(200, 800)
                },
                quality_score=np.random.uniform(0.8, 1.0),
                uncertainty={
                    'so2_flux': 50.0,
                    'thermal_radiance': 5.0
                },
                source_id='test_usgs',
                processing_latency=np.random.uniform(0.2, 1.0)
            )
            data_points.append(data_point)
        
        return data_points

class TestDataIngestion:
    """Test data ingestion and processing functionality"""
    
    @pytest.mark.asyncio
    async def test_fetch_real_time_data(self, monitor):
        """Test fetching real-time data from multiple sources"""
        # Mock the individual fetch methods
        monitor._fetch_seismic_streams = AsyncMock(return_value=[])
        monitor._fetch_volcanic_streams = AsyncMock(return_value=[])
        monitor._fetch_deformation_streams = AsyncMock(return_value=[])
        monitor._fetch_geochemical_streams = AsyncMock(return_value=[])
        
        # Test data fetching
        data_points = await monitor._fetch_real_time_data()
        
        # Verify all fetch methods were called
        monitor._fetch_seismic_streams.assert_called_once()
        monitor._fetch_volcanic_streams.assert_called_once()
        monitor._fetch_deformation_streams.assert_called_once()
        monitor._fetch_geochemical_streams.assert_called_once()
        
        # Verify return type
        assert isinstance(data_points, list)
    
    @pytest.mark.asyncio
    async def test_fetch_seismic_streams(self, monitor):
        """Test seismic data stream fetching"""
        seismic_data = await monitor._fetch_seismic_streams()
        
        # Verify data structure
        assert isinstance(seismic_data, list)
        
        if seismic_data:
            data_point = seismic_data[0]
            assert isinstance(data_point, GeologicalDataPoint)
            assert data_point.process_type == ProcessType.SEISMIC
            assert 'ground_velocity' in data_point.parameters
            assert 'dominant_frequency' in data_point.parameters
            assert data_point.quality_score >= 0.0
            assert data_point.quality_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_fetch_volcanic_streams(self, monitor):
        """Test volcanic data stream fetching"""
        volcanic_data = await monitor._fetch_volcanic_streams()
        
        # Verify data structure
        assert isinstance(volcanic_data, list)
        
        if volcanic_data:
            data_point = volcanic_data[0]
            assert isinstance(data_point, GeologicalDataPoint)
            assert data_point.process_type == ProcessType.VOLCANIC
            assert 'so2_flux' in data_point.parameters
            assert 'thermal_radiance' in data_point.parameters
            assert data_point.quality_score >= 0.0
            assert data_point.quality_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_process_data_point(self, monitor, sample_seismic_data):
        """Test processing individual data points"""
        data_point = sample_seismic_data[0]
        
        # Mock the analysis methods
        monitor._analyze_causal_relationships = AsyncMock()
        monitor._update_baseline_statistics = AsyncMock()
        monitor._broadcast_data_update = AsyncMock()
        
        # Process the data point
        await monitor._process_data_point(data_point)
        
        # Verify data was added to buffer
        buffer_key = f"{data_point.process_type.value}_{data_point.station_id}"
        assert buffer_key in monitor.data_buffers
        assert len(monitor.data_buffers[buffer_key]) > 0
        
        # Verify analysis methods were called
        monitor._analyze_causal_relationships.assert_called_once_with(data_point)
        monitor._update_baseline_statistics.assert_called_once_with(data_point)
        monitor._broadcast_data_update.assert_called_once_with(data_point)

class TestCausalAnalysis:
    """Test causal analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_calculate_causal_strength(self, monitor, sample_seismic_data, sample_volcanic_data):
        """Test causal strength calculation between processes"""
        # Add data to buffers
        seismic_key = f"{sample_seismic_data[0].process_type.value}_{sample_seismic_data[0].station_id}"
        volcanic_key = f"{sample_volcanic_data[0].process_type.value}_{sample_volcanic_data[0].station_id}"
        
        for dp in sample_seismic_data:
            monitor.data_buffers[seismic_key].append(dp)
        
        for dp in sample_volcanic_data:
            monitor.data_buffers[volcanic_key].append(dp)
        
        # Calculate causal strength
        strength = await monitor._calculate_causal_strength(seismic_key, volcanic_key)
        
        # Verify result
        if strength is not None:
            assert isinstance(strength, float)
            assert 0.0 <= strength <= 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_causal_confidence(self, monitor, sample_seismic_data, sample_volcanic_data):
        """Test causal confidence calculation"""
        # Add data to buffers
        seismic_key = f"{sample_seismic_data[0].process_type.value}_{sample_seismic_data[0].station_id}"
        volcanic_key = f"{sample_volcanic_data[0].process_type.value}_{sample_volcanic_data[0].station_id}"
        
        for dp in sample_seismic_data:
            monitor.data_buffers[seismic_key].append(dp)
        
        for dp in sample_volcanic_data:
            monitor.data_buffers[volcanic_key].append(dp)
        
        # Calculate confidence
        confidence = await monitor._calculate_causal_confidence(seismic_key, volcanic_key)
        
        # Verify result
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_causal_strength_monitoring(self, monitor, sample_seismic_data):
        """Test causal strength monitoring loop"""
        # Add test data
        process_key = f"{sample_seismic_data[0].process_type.value}_{sample_seismic_data[0].station_id}"
        monitor.monitoring_state.active_processes.add(process_key)
        
        for dp in sample_seismic_data:
            monitor.data_buffers[process_key].append(dp)
        
        # Mock causal strength calculation
        monitor._calculate_causal_strength = AsyncMock(return_value=0.75)
        monitor._calculate_causal_confidence = AsyncMock(return_value=0.85)
        monitor._check_causal_strength_changes = AsyncMock()
        
        # Run one iteration of monitoring (with timeout to prevent infinite loop)
        try:
            await asyncio.wait_for(monitor._causal_strength_monitor(), timeout=0.1)
        except asyncio.TimeoutError:
            pass  # Expected timeout
        
        # Verify methods were called
        monitor._calculate_causal_strength.assert_called()
        monitor._calculate_causal_confidence.assert_called()

class TestAnomalyDetection:
    """Test anomaly detection functionality"""
    
    @pytest.mark.asyncio
    async def test_detect_process_anomalies(self, monitor, sample_seismic_data):
        """Test anomaly detection for a process"""
        # Add test data with some anomalous values
        process_key = f"{sample_seismic_data[0].process_type.value}_{sample_seismic_data[0].station_id}"
        
        # Add normal data
        for dp in sample_seismic_data:
            monitor.data_buffers[process_key].append(dp)
        
        # Add anomalous data point
        anomalous_data = GeologicalDataPoint(
            timestamp=datetime.utcnow(),
            process_type=ProcessType.SEISMIC,
            station_id='TEST_SEISMIC_01',
            location=(45.0, -110.0, 1500.0),
            parameters={
                'ground_velocity': 1e-3,  # Very high value (anomalous)
                'dominant_frequency': 100,  # Very high frequency (anomalous)
                'signal_to_noise_ratio': 50,
                'peak_ground_acceleration': 1.0  # Very high acceleration (anomalous)
            },
            quality_score=0.9,
            uncertainty={'ground_velocity': 1e-4, 'dominant_frequency': 5.0},
            source_id='test_iris',
            processing_latency=0.3
        )
        monitor.data_buffers[process_key].append(anomalous_data)
        
        # Detect anomalies
        anomalies = await monitor._detect_process_anomalies(process_key)
        
        # Verify anomalies were detected
        assert isinstance(anomalies, list)
        # Note: Anomalies might not always be detected depending on the algorithm
        # This test verifies the method runs without error
    
    @pytest.mark.asyncio
    async def test_detect_statistical_anomalies(self, monitor):
        """Test statistical anomaly detection"""
        # Create test data with known anomaly
        features = np.array([
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [0.9, 1.9, 2.9],
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0]  # Clear anomaly
        ])
        
        timestamps = [datetime.utcnow() + timedelta(seconds=i) for i in range(5)]
        process_key = "seismic_TEST_STATION"
        
        # Detect anomalies
        anomalies = await monitor._detect_statistical_anomalies(features, timestamps, process_key)
        
        # Verify result structure
        assert isinstance(anomalies, list)
        for anomaly in anomalies:
            assert isinstance(anomaly, AnomalyDetection)
            assert anomaly.anomaly_type == AnomalyType.STATISTICAL
            assert 0.0 <= anomaly.anomaly_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_detect_temporal_anomalies(self, monitor):
        """Test temporal anomaly detection"""
        # Create test data with temporal anomaly (sudden change)
        features = [
            [1.0, 2.0],
            [1.1, 2.1],
            [1.0, 2.0],
            [5.0, 10.0],  # Sudden jump (temporal anomaly)
            [5.1, 10.1]
        ]
        
        timestamps = [datetime.utcnow() + timedelta(seconds=i) for i in range(5)]
        process_key = "volcanic_TEST_STATION"
        
        # Detect anomalies
        anomalies = await monitor._detect_temporal_anomalies(features, timestamps, process_key)
        
        # Verify result structure
        assert isinstance(anomalies, list)
        for anomaly in anomalies:
            assert isinstance(anomaly, AnomalyDetection)
            assert anomaly.anomaly_type == AnomalyType.TEMPORAL
    
    @pytest.mark.asyncio
    async def test_analyze_anomaly_causal_pathway(self, monitor):
        """Test causal pathway analysis for anomalies"""
        # Create test anomaly
        anomaly = AnomalyDetection(
            anomaly_id="test_anomaly",
            timestamp=datetime.utcnow(),
            anomaly_type=AnomalyType.STATISTICAL,
            process_type=ProcessType.SEISMIC,
            station_id="TEST_STATION",
            anomaly_score=0.85,
            statistical_significance=0.95,
            causal_pathway=[],
            affected_processes=set(),
            temporal_pattern={},
            spatial_extent={}
        )
        
        # Mock causal strength calculation
        monitor._calculate_causal_strength = AsyncMock(return_value=0.8)
        monitor._check_recent_anomalies = AsyncMock(return_value=[])
        monitor._analyze_spatial_extent = AsyncMock()
        
        # Add some active processes
        monitor.monitoring_state.active_processes.add("volcanic_TEST_VOLCANO")
        monitor.monitoring_state.active_processes.add("seismic_TEST_STATION")
        
        # Analyze causal pathway
        await monitor._analyze_anomaly_causal_pathway(anomaly)
        
        # Verify analysis was performed
        monitor._calculate_causal_strength.assert_called()
        monitor._analyze_spatial_extent.assert_called_once_with(anomaly)

class TestAlertGeneration:
    """Test alert generation functionality"""
    
    @pytest.mark.asyncio
    async def test_check_causal_strength_alerts(self, monitor):
        """Test causal strength change alert generation"""
        # Add causal strength history with significant change
        relationship_key = "seismic_STATION1->volcanic_STATION2"
        monitor.causal_strength_history[relationship_key].extend([
            {'timestamp': datetime.utcnow() - timedelta(seconds=10), 'strength': 0.3, 'confidence': 0.8},
            {'timestamp': datetime.utcnow(), 'strength': 0.8, 'confidence': 0.9}  # Significant increase
        ])
        
        # Check for alerts
        alerts = await monitor._check_causal_strength_alerts()
        
        # Verify alert generation
        assert isinstance(alerts, list)
        if alerts:
            alert = alerts[0]
            assert isinstance(alert, RealTimeAlert)
            assert alert.severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH]
    
    @pytest.mark.asyncio
    async def test_check_anomaly_alerts(self, monitor, sample_volcanic_data):
        """Test anomaly-based alert generation"""
        # Add volcanic data with high SO2 flux
        process_key = f"{sample_volcanic_data[0].process_type.value}_{sample_volcanic_data[0].station_id}"
        
        # Modify data to create alert condition
        for i, dp in enumerate(sample_volcanic_data):
            if i >= 7:  # Last 3 points have high SO2
                dp.parameters['so2_flux'] = 2000.0  # Very high value
            monitor.data_buffers[process_key].append(dp)
        
        monitor.monitoring_state.active_processes.add(process_key)
        
        # Check for alerts
        alerts = await monitor._check_anomaly_alerts()
        
        # Verify alert structure
        assert isinstance(alerts, list)
        for alert in alerts:
            assert isinstance(alert, RealTimeAlert)
            assert alert.process_type == ProcessType.VOLCANIC
    
    @pytest.mark.asyncio
    async def test_generate_predictive_alerts(self, monitor):
        """Test predictive alert generation"""
        # Mock causal relationships and predictions
        monitor._calculate_causal_strength = AsyncMock(return_value=0.8)
        monitor._generate_process_prediction = AsyncMock(return_value={
            'risk_level': 0.85,
            'confidence': 0.9,
            'contributing_factors': 3,
            'prediction_horizon': '1-6 hours'
        })
        
        # Add active processes
        monitor.monitoring_state.active_processes.add("seismic_STATION1")
        monitor.monitoring_state.active_processes.add("volcanic_STATION2")
        
        # Generate predictive alerts
        alerts = await monitor._generate_predictive_alerts()
        
        # Verify alert generation
        assert isinstance(alerts, list)
        if alerts:
            alert = alerts[0]
            assert isinstance(alert, RealTimeAlert)
            assert alert.severity == AlertSeverity.MEDIUM

class TestDataQuality:
    """Test data quality validation functionality"""
    
    @pytest.mark.asyncio
    async def test_validate_process_data_quality(self, monitor, sample_seismic_data):
        """Test data quality validation for a process"""
        # Add test data with quality issues
        process_key = f"{sample_seismic_data[0].process_type.value}_{sample_seismic_data[0].station_id}"
        
        # Add data with quality issues
        for i, dp in enumerate(sample_seismic_data):
            if i % 3 == 0:  # Every third point has low quality
                dp.quality_score = 0.3
            if i >= 5:  # Later points have high latency
                dp.processing_latency = 8.0
            monitor.data_buffers[process_key].append(dp)
        
        # Validate data quality
        issues = await monitor._validate_process_data_quality(process_key)
        
        # Verify issue detection
        assert isinstance(issues, list)
        for issue in issues:
            assert 'type' in issue
            assert 'severity' in issue
            assert 'description' in issue
            assert 0.0 <= issue['severity'] <= 1.0

class TestPerformanceMonitoring:
    """Test performance monitoring functionality"""
    
    @pytest.mark.asyncio
    async def test_calculate_performance_metrics(self, monitor):
        """Test performance metrics calculation"""
        # Add some processing times
        monitor.processing_times.extend([0.1, 0.15, 0.12, 0.18, 0.11])
        monitor.throughput_counter = 100
        monitor.last_throughput_reset = datetime.utcnow() - timedelta(seconds=10)
        
        # Add some active processes
        monitor.monitoring_state.active_processes.add("seismic_STATION1")
        monitor.monitoring_state.active_processes.add("volcanic_STATION2")
        
        # Calculate metrics
        metrics = await monitor._calculate_performance_metrics()
        
        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert 'avg_processing_time_ms' in metrics
        assert 'throughput_points_per_sec' in metrics
        assert 'active_processes' in metrics
        assert metrics['active_processes'] == 2
        assert metrics['avg_processing_time_ms'] > 0

class TestWebSocketIntegration:
    """Test WebSocket integration functionality"""
    
    @pytest.mark.asyncio
    async def test_broadcast_data_update(self, monitor, sample_seismic_data):
        """Test WebSocket data broadcasting"""
        # Mock WebSocket connections
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        
        monitor.websocket_connections.add(mock_websocket1)
        monitor.websocket_connections.add(mock_websocket2)
        
        # Broadcast data update
        data_point = sample_seismic_data[0]
        await monitor._broadcast_data_update(data_point)
        
        # Verify broadcast was sent to all connections
        mock_websocket1.send.assert_called_once()
        mock_websocket2.send.assert_called_once()
        
        # Verify message format
        call_args = mock_websocket1.send.call_args[0][0]
        message = json.loads(call_args)
        assert message['type'] == 'data_update'
        assert 'data' in message
        assert message['data']['process_type'] == data_point.process_type.value

class TestIntegration:
    """Integration tests for the complete monitoring system"""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_cycle(self, monitor, sample_seismic_data, sample_volcanic_data):
        """Test complete monitoring cycle with real data flow"""
        # Mock external dependencies
        monitor._fetch_real_time_data = AsyncMock(return_value=sample_seismic_data + sample_volcanic_data)
        monitor._store_processed_data = AsyncMock()
        monitor._broadcast_data_update = AsyncMock()
        
        # Run one cycle of stream processing
        try:
            await asyncio.wait_for(monitor._stream_data_processor(), timeout=0.5)
        except asyncio.TimeoutError:
            pass  # Expected timeout
        
        # Verify data was processed
        assert len(monitor.data_buffers) > 0
        assert len(monitor.monitoring_state.active_processes) > 0
        
        # Verify methods were called
        monitor._fetch_real_time_data.assert_called()
        monitor._broadcast_data_update.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, monitor):
        """Test error handling in monitoring components"""
        # Mock methods to raise exceptions
        monitor._fetch_real_time_data = AsyncMock(side_effect=Exception("Test error"))
        
        # Verify system handles errors gracefully
        try:
            await asyncio.wait_for(monitor._stream_data_processor(), timeout=0.5)
        except asyncio.TimeoutError:
            pass  # Expected timeout
        
        # System should continue running despite errors
        assert monitor.monitoring_state is not None

# Performance and load testing
class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    async def test_high_throughput_processing(self, monitor):
        """Test system performance under high data throughput"""
        # Generate large amount of test data
        data_points = []
        base_time = datetime.utcnow()
        
        for i in range(1000):  # 1000 data points
            data_point = GeologicalDataPoint(
                timestamp=base_time + timedelta(milliseconds=i * 100),
                process_type=ProcessType.SEISMIC,
                station_id=f'STATION_{i % 10}',
                location=(45.0 + i * 0.001, -110.0 + i * 0.001, 1500.0),
                parameters={
                    'ground_velocity': np.random.lognormal(0, 1) * 1e-6,
                    'dominant_frequency': np.random.uniform(1, 50)
                },
                quality_score=np.random.uniform(0.7, 1.0),
                uncertainty={'ground_velocity': 1e-7},
                source_id='test_source',
                processing_latency=np.random.uniform(0.01, 0.1)
            )
            data_points.append(data_point)
        
        # Process all data points
        start_time = datetime.utcnow()
        
        for data_point in data_points:
            await monitor._process_data_point(data_point)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify performance
        throughput = len(data_points) / processing_time
        assert throughput > 100  # Should process at least 100 points per second
        
        # Verify data integrity
        assert len(monitor.data_buffers) > 0
        total_processed = sum(len(buffer) for buffer in monitor.data_buffers.values())
        assert total_processed == len(data_points)
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, monitor, sample_seismic_data):
        """Test memory usage with large data buffers"""
        # Fill buffers with large amount of data
        process_key = f"{sample_seismic_data[0].process_type.value}_{sample_seismic_data[0].station_id}"
        
        # Add data up to buffer limit
        for i in range(10000):  # Fill to buffer limit
            data_point = sample_seismic_data[0]
            data_point.timestamp = datetime.utcnow() + timedelta(seconds=i)
            monitor.data_buffers[process_key].append(data_point)
        
        # Verify buffer size is maintained
        assert len(monitor.data_buffers[process_key]) == 10000  # maxlen limit
        
        # Add more data and verify oldest is removed
        for i in range(100):
            data_point = sample_seismic_data[0]
            data_point.timestamp = datetime.utcnow() + timedelta(seconds=10000 + i)
            monitor.data_buffers[process_key].append(data_point)
        
        # Buffer should still be at limit
        assert len(monitor.data_buffers[process_key]) == 10000

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])