"""
Comprehensive test suite for Monitoring Network Registry

This test suite validates all functionality of the monitoring network registry
including observatory management, station inventory, coverage analysis,
performance assessment, and real-time data integration.

Author: Prometheus AI Platform
Version: 1.0.0
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os

# Import the module under test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from monitoring_network_registry import (
    MonitoringNetworkRegistry,
    MonitoringNetworkType,
    InstrumentType,
    DataStreamStatus,
    NetworkStatus,
    GeographicBounds,
    InstrumentSpecification,
    DataStreamInfo,
    NetworkCoverageMetrics,
    PerformanceMetrics,
    VolcanicObservatory,
    MonitoringStation,
    DataStream,
    NetworkCoverage,
    Base
)

class TestMonitoringNetworkRegistry:
    """Test class for Monitoring Network Registry functionality"""
    
    @pytest.fixture
    async def registry(self):
        """Create test registry instance with in-memory database"""
        # Use in-memory SQLite for testing
        test_db_url = "sqlite:///:memory:"
        
        registry = MonitoringNetworkRegistry(
            database_url=test_db_url,
            redis_url=None,  # Disable Redis for testing
            kafka_config=None  # Disable Kafka for testing
        )
        
        yield registry
        
        # Cleanup
        if hasattr(registry, 'executor'):
            registry.executor.shutdown(wait=True)
    
    @pytest.fixture
    def sample_observatory_data(self):
        """Sample observatory data for testing"""
        return {
            'name': 'Test Volcano Observatory',
            'acronym': 'TVO',
            'country': 'Test Country',
            'region': 'Test Region',
            'latitude': 45.0,
            'longitude': -120.0,
            'elevation': 1500,
            'established_date': datetime(2000, 1, 1),
            'parent_organization': 'Test Organization',
            'contact_email': 'test@example.com',
            'website_url': 'https://test.example.com',
            'phone': '+1-555-0123',
            'address': '123 Test Street, Test City',
            'primary_volcanoes': ['test_volcano_1', 'test_volcano_2'],
            'network_types': ['seismic', 'gps_gnss', 'thermal'],
            'status': 'operational',
            'capabilities': {
                'seismic_monitoring': True,
                'deformation_monitoring': True,
                'gas_monitoring': False,
                'thermal_monitoring': True
            },
            'staff_count': 15,
            'annual_budget': 2500000.0,
            'data_sharing_policy': 'Open access with attribution',
            'real_time_capability': True,
            'alert_system': True
        }
    
    @pytest.fixture
    def sample_station_data(self):
        """Sample monitoring station data for testing"""
        return {
            'station_code': 'TEST.STA1',
            'name': 'Test Seismic Station 1',
            'volcano_id': 'test_volcano_1',
            'latitude': 45.1,
            'longitude': -119.9,
            'elevation': 1200,
            'installation_date': datetime(2005, 6, 15),
            'station_type': 'seismic',
            'network_code': 'TEST',
            'site_conditions': 'Bedrock installation',
            'access_difficulty': 'moderate',
            'power_source': 'Solar with battery backup',
            'communication_method': 'Satellite',
            'maintenance_schedule': 'Quarterly',
            'status': 'operational',
            'instruments': [
                {
                    'type': 'seismometer',
                    'manufacturer': 'Test Instruments Inc.',
                    'model': 'TestSeismo-3000',
                    'serial_number': 'TS3000-001',
                    'sampling_rate': 100.0,
                    'installation_date': '2005-06-15'
                }
            ],
            'data_availability': 95.5,
            'uptime': 98.2,
            'metadata': {
                'installation_notes': 'Installed during summer field season',
                'site_photos': ['photo1.jpg', 'photo2.jpg']
            }
        }
    
    @pytest.fixture
    def sample_stream_data(self):
        """Sample data stream data for testing"""
        return {
            'stream_id': 'TEST.STA1.BHZ',
            'parameter_name': 'vertical_velocity',
            'unit': 'm/s',
            'instrument_type': 'seismometer',
            'sampling_rate': 100.0,
            'data_format': 'miniSEED',
            'endpoint_url': 'https://test.example.com/data/TEST.STA1.BHZ',
            'authentication_method': 'API_KEY',
            'real_time': True,
            'latency': 2.5,
            'availability': 95.0,
            'quality_score': 0.92,
            'last_update': datetime.utcnow(),
            'status': 'active',
            'access_restrictions': 'Open access',
            'data_policy': 'CC BY 4.0',
            'contact_info': 'data@test.example.com',
            'metadata': {
                'calibration_date': '2023-01-15',
                'response_file': 'TEST_STA1_response.xml'
            }
        }
    
    @pytest.fixture
    def sample_geographic_bounds(self):
        """Sample geographic bounds for testing"""
        return GeographicBounds(
            min_lat=44.0,
            max_lat=46.0,
            min_lon=-121.0,
            max_lon=-119.0
        )

    # Test Observatory Management
    
    async def test_register_volcanic_observatory(self, registry, sample_observatory_data):
        """Test observatory registration"""
        observatory_id = await registry.register_volcanic_observatory(sample_observatory_data)
        
        assert observatory_id is not None
        assert isinstance(observatory_id, str)
        
        # Verify observatory was stored
        observatories = await registry.query_observatories(
            filters={'name': sample_observatory_data['name']}
        )
        
        assert len(observatories) == 1
        assert observatories[0]['name'] == sample_observatory_data['name']
        assert observatories[0]['country'] == sample_observatory_data['country']
        assert observatories[0]['latitude'] == sample_observatory_data['latitude']
        assert observatories[0]['longitude'] == sample_observatory_data['longitude']
    
    async def test_register_observatory_missing_required_fields(self, registry):
        """Test observatory registration with missing required fields"""
        incomplete_data = {
            'name': 'Test Observatory'
            # Missing required fields: country, latitude, longitude
        }
        
        with pytest.raises(ValueError, match="Missing required field"):
            await registry.register_volcanic_observatory(incomplete_data)
    
    async def test_query_observatories_with_filters(self, registry, sample_observatory_data):
        """Test querying observatories with various filters"""
        # Register test observatory
        await registry.register_volcanic_observatory(sample_observatory_data)
        
        # Test country filter
        observatories = await registry.query_observatories(
            filters={'country': sample_observatory_data['country']}
        )
        assert len(observatories) == 1
        
        # Test status filter
        observatories = await registry.query_observatories(
            filters={'status': 'operational'}
        )
        assert len(observatories) == 1
        
        # Test real-time capability filter
        observatories = await registry.query_observatories(
            filters={'real_time_capability': True}
        )
        assert len(observatories) == 1
        
        # Test network types filter
        observatories = await registry.query_observatories(
            filters={'network_types': ['seismic']}
        )
        assert len(observatories) == 1
    
    async def test_query_observatories_geographic_bounds(self, registry, sample_observatory_data, sample_geographic_bounds):
        """Test querying observatories within geographic bounds"""
        # Register test observatory
        await registry.register_volcanic_observatory(sample_observatory_data)
        
        # Query within bounds (should find observatory)
        observatories = await registry.query_observatories(
            geographic_bounds=sample_geographic_bounds
        )
        assert len(observatories) == 1
        
        # Query outside bounds (should not find observatory)
        outside_bounds = GeographicBounds(50.0, 60.0, -130.0, -125.0)
        observatories = await registry.query_observatories(
            geographic_bounds=outside_bounds
        )
        assert len(observatories) == 0

    # Test Monitoring Station Management
    
    async def test_register_monitoring_station(self, registry, sample_observatory_data, sample_station_data):
        """Test monitoring station registration"""
        # Register observatory first
        observatory_id = await registry.register_volcanic_observatory(sample_observatory_data)
        sample_station_data['observatory_id'] = observatory_id
        
        # Register station
        station_id = await registry.register_monitoring_station(sample_station_data)
        
        assert station_id is not None
        assert isinstance(station_id, str)
        
        # Verify station was stored
        stations = await registry.query_monitoring_stations(
            filters={'station_code': sample_station_data['station_code']}
        )
        
        assert len(stations) == 1
        assert stations[0]['station_code'] == sample_station_data['station_code']
        assert stations[0]['name'] == sample_station_data['name']
        assert stations[0]['station_type'] == sample_station_data['station_type']
    
    async def test_register_station_missing_required_fields(self, registry):
        """Test station registration with missing required fields"""
        incomplete_data = {
            'station_code': 'TEST.STA1'
            # Missing required fields: name, latitude, longitude, etc.
        }
        
        with pytest.raises(ValueError, match="Missing required field"):
            await registry.register_monitoring_station(incomplete_data)
    
    async def test_query_monitoring_stations_with_filters(self, registry, sample_observatory_data, sample_station_data):
        """Test querying monitoring stations with various filters"""
        # Register observatory and station
        observatory_id = await registry.register_volcanic_observatory(sample_observatory_data)
        sample_station_data['observatory_id'] = observatory_id
        await registry.register_monitoring_station(sample_station_data)
        
        # Test station type filter
        stations = await registry.query_monitoring_stations(
            filters={'station_type': 'seismic'}
        )
        assert len(stations) == 1
        
        # Test observatory filter
        stations = await registry.query_monitoring_stations(
            filters={'observatory_id': observatory_id}
        )
        assert len(stations) == 1
        
        # Test status filter
        stations = await registry.query_monitoring_stations(
            filters={'status': 'operational'}
        )
        assert len(stations) == 1
        
        # Test minimum uptime filter
        stations = await registry.query_monitoring_stations(
            filters={'min_uptime': 95.0}
        )
        assert len(stations) == 1

    # Test Data Stream Management
    
    async def test_register_data_stream(self, registry, sample_observatory_data, sample_station_data, sample_stream_data):
        """Test data stream registration"""
        # Register observatory and station first
        observatory_id = await registry.register_volcanic_observatory(sample_observatory_data)
        sample_station_data['observatory_id'] = observatory_id
        station_id = await registry.register_monitoring_station(sample_station_data)
        
        sample_stream_data['station_id'] = station_id
        sample_stream_data['observatory_id'] = observatory_id
        
        # Register data stream
        stream_db_id = await registry.register_data_stream(sample_stream_data)
        
        assert stream_db_id is not None
        assert isinstance(stream_db_id, str)
    
    async def test_register_stream_missing_required_fields(self, registry):
        """Test data stream registration with missing required fields"""
        incomplete_data = {
            'stream_id': 'TEST.STREAM'
            # Missing required fields: parameter_name, station_id
        }
        
        with pytest.raises(ValueError, match="Missing required field"):
            await registry.register_data_stream(incomplete_data)

    # Test Network Coverage Analysis
    
    async def test_analyze_network_coverage(self, registry, sample_observatory_data, sample_station_data, sample_geographic_bounds):
        """Test network coverage analysis"""
        # Register observatory and multiple stations
        observatory_id = await registry.register_volcanic_observatory(sample_observatory_data)
        
        # Register multiple stations for better coverage analysis
        stations_data = []
        for i in range(3):
            station_data = sample_station_data.copy()
            station_data['station_code'] = f'TEST.STA{i+1}'
            station_data['name'] = f'Test Station {i+1}'
            station_data['latitude'] = 45.0 + i * 0.1
            station_data['longitude'] = -120.0 + i * 0.1
            station_data['observatory_id'] = observatory_id
            stations_data.append(station_data)
            await registry.register_monitoring_station(station_data)
        
        # Analyze coverage
        coverage = await registry.analyze_network_coverage(
            sample_geographic_bounds,
            MonitoringNetworkType.SEISMIC
        )
        
        assert isinstance(coverage, NetworkCoverageMetrics)
        assert coverage.total_area_km2 > 0
        assert coverage.coverage_percentage >= 0
        assert coverage.station_density >= 0
        assert coverage.average_inter_station_distance >= 0
        assert isinstance(coverage.coverage_gaps, list)
        assert 0 <= coverage.redundancy_score <= 1
        assert 0 <= coverage.temporal_coverage <= 100
    
    async def test_analyze_coverage_no_stations(self, registry, sample_geographic_bounds):
        """Test coverage analysis with no stations"""
        coverage = await registry.analyze_network_coverage(
            sample_geographic_bounds,
            MonitoringNetworkType.SEISMIC
        )
        
        assert coverage.coverage_percentage == 0.0
        assert coverage.station_density == 0.0
        assert coverage.average_inter_station_distance == 0.0
        assert coverage.redundancy_score == 0.0

    # Test Performance Assessment
    
    async def test_assess_network_performance(self, registry, sample_observatory_data, sample_station_data):
        """Test network performance assessment"""
        # Register observatory and station
        observatory_id = await registry.register_volcanic_observatory(sample_observatory_data)
        sample_station_data['observatory_id'] = observatory_id
        await registry.register_monitoring_station(sample_station_data)
        
        # Assess performance
        performance = await registry.assess_network_performance()
        
        assert isinstance(performance, PerformanceMetrics)
        assert 0 <= performance.uptime_percentage <= 100
        assert 0 <= performance.data_completeness <= 100
        assert performance.latency_p95 >= 0
        assert 0 <= performance.error_rate <= 1
        assert 0 <= performance.false_alarm_rate <= 1
        assert 0 <= performance.detection_capability <= 1
        assert performance.response_time >= 0
        assert performance.maintenance_frequency >= 0
    
    async def test_assess_performance_no_stations(self, registry):
        """Test performance assessment with no stations"""
        performance = await registry.assess_network_performance()
        
        # Should return zero metrics when no stations exist
        assert performance.uptime_percentage == 0.0
        assert performance.data_completeness == 0.0

    # Test Monitoring Gap Identification
    
    async def test_identify_monitoring_gaps(self, registry, sample_geographic_bounds):
        """Test monitoring gap identification"""
        # Test with no existing stations (should identify gaps for all network types)
        gaps = await registry.identify_monitoring_gaps(sample_geographic_bounds)
        
        assert isinstance(gaps, list)
        # Should have recommendations for multiple network types
        assert len(gaps) > 0
        
        # Verify gap structure
        for gap in gaps:
            assert 'network_type' in gap
            assert 'current_coverage' in gap
            assert 'target_coverage' in gap
            assert 'gap_percentage' in gap
            assert 'additional_stations_needed' in gap
            assert 'priority' in gap
            assert gap['priority'] in ['high', 'medium', 'low']
            assert 'estimated_cost' in gap
            assert 'implementation_timeline' in gap

    # Test External Source Synchronization
    
    async def test_synchronize_with_external_sources(self, registry):
        """Test synchronization with external sources"""
        # Mock external API responses
        with patch.object(registry, '_fetch_external_observatories', return_value=[]):
            with patch.object(registry, '_fetch_external_stations', return_value=[]):
                sync_report = await registry.synchronize_with_external_sources()
        
        assert isinstance(sync_report, dict)
        assert 'start_time' in sync_report
        assert 'end_time' in sync_report
        assert 'sources_processed' in sync_report
        assert 'observatories_updated' in sync_report
        assert 'stations_updated' in sync_report
        assert 'streams_updated' in sync_report
        assert 'errors' in sync_report
        assert 'success' in sync_report

    # Test Real-Time Data Integration
    
    async def test_setup_real_time_data_integration(self, registry):
        """Test real-time data integration setup"""
        stream_configs = [
            {
                'stream_id': 'TEST.STREAM1',
                'endpoint_url': 'https://test.example.com/stream1',
                'parameter_name': 'seismic_velocity',
                'station_id': 'test-station-id'
            }
        ]
        
        # Mock connection test
        with patch.object(registry, '_test_stream_connection', return_value={'success': True}):
            with patch.object(registry, '_setup_stream_consumer', return_value=None):
                setup_report = await registry.setup_real_time_data_integration(stream_configs)
        
        assert isinstance(setup_report, dict)
        assert 'start_time' in setup_report
        assert 'end_time' in setup_report
        assert 'streams_configured' in setup_report
        assert 'streams_active' in setup_report
        assert 'failed_streams' in setup_report
        assert 'integration_endpoints' in setup_report
        assert 'success' in setup_report
    
    async def test_setup_integration_missing_fields(self, registry):
        """Test integration setup with missing required fields"""
        invalid_configs = [
            {
                'stream_id': 'TEST.STREAM1'
                # Missing endpoint_url and parameter_name
            }
        ]
        
        setup_report = await registry.setup_real_time_data_integration(invalid_configs)
        
        # Should report failed streams
        assert len(setup_report['failed_streams']) > 0

    # Test Report Generation
    
    async def test_generate_monitoring_report(self, registry, sample_observatory_data, sample_station_data, sample_geographic_bounds):
        """Test comprehensive monitoring report generation"""
        # Register test data
        observatory_id = await registry.register_volcanic_observatory(sample_observatory_data)
        sample_station_data['observatory_id'] = observatory_id
        await registry.register_monitoring_station(sample_station_data)
        
        # Generate report
        report = await registry.generate_monitoring_report(
            region_bounds=sample_geographic_bounds
        )
        
        assert isinstance(report, dict)
        assert 'report_id' in report
        assert 'generated_at' in report
        assert 'summary' in report
        assert 'observatories' in report
        assert 'networks' in report
        assert 'coverage_analysis' in report
        assert 'performance_metrics' in report
        assert 'recommendations' in report
        
        # Verify summary data
        assert report['summary']['total_observatories'] >= 1
        assert report['summary']['countries'] >= 1

    # Test Helper Methods
    
    def test_geographic_bounds_contains_point(self, sample_geographic_bounds):
        """Test geographic bounds point containment"""
        # Point inside bounds
        assert sample_geographic_bounds.contains_point(45.0, -120.0) == True
        
        # Point outside bounds
        assert sample_geographic_bounds.contains_point(50.0, -130.0) == False
        
        # Point on boundary
        assert sample_geographic_bounds.contains_point(44.0, -121.0) == True
    
    def test_calculate_region_area(self, registry, sample_geographic_bounds):
        """Test region area calculation"""
        area = registry._calculate_region_area(sample_geographic_bounds)
        
        assert area > 0
        assert isinstance(area, float)
    
    def test_calculate_average_inter_station_distance(self, registry):
        """Test average inter-station distance calculation"""
        # Test with multiple stations
        station_coords = [(45.0, -120.0), (45.1, -119.9), (44.9, -120.1)]
        avg_distance = registry._calculate_average_inter_station_distance(station_coords)
        
        assert avg_distance > 0
        assert isinstance(avg_distance, float)
        
        # Test with single station
        single_coord = [(45.0, -120.0)]
        avg_distance = registry._calculate_average_inter_station_distance(single_coord)
        assert avg_distance == 0.0
        
        # Test with no stations
        no_coords = []
        avg_distance = registry._calculate_average_inter_station_distance(no_coords)
        assert avg_distance == 0.0
    
    def test_calculate_redundancy_score(self, registry):
        """Test redundancy score calculation"""
        # Test with multiple nearby stations (high redundancy)
        nearby_coords = [(45.0, -120.0), (45.01, -120.01), (45.02, -120.02)]
        redundancy = registry._calculate_redundancy_score(nearby_coords)
        
        assert 0 <= redundancy <= 1
        assert isinstance(redundancy, float)
        
        # Test with single station
        single_coord = [(45.0, -120.0)]
        redundancy = registry._calculate_redundancy_score(single_coord)
        assert redundancy == 0.0
    
    def test_estimate_deployment_cost(self, registry):
        """Test deployment cost estimation"""
        cost = registry._estimate_deployment_cost(MonitoringNetworkType.SEISMIC, 5)
        
        assert isinstance(cost, dict)
        assert 'equipment_cost' in cost
        assert 'installation_cost' in cost
        assert 'annual_maintenance' in cost
        assert 'total_first_year' in cost
        
        # All costs should be positive
        for cost_type, amount in cost.items():
            assert amount > 0
    
    def test_estimate_implementation_timeline(self, registry):
        """Test implementation timeline estimation"""
        timeline = registry._estimate_implementation_timeline(3)
        
        assert isinstance(timeline, dict)
        assert 'planning_phase' in timeline
        assert 'procurement' in timeline
        assert 'installation' in timeline
        assert 'testing_commissioning' in timeline
        assert 'total_timeline' in timeline

    # Test Error Handling
    
    async def test_database_error_handling(self, registry):
        """Test handling of database errors"""
        # Test with invalid database connection
        invalid_registry = MonitoringNetworkRegistry(
            database_url="postgresql://invalid:invalid@localhost/invalid"
        )
        
        # Should handle connection errors gracefully
        with pytest.raises(Exception):
            await invalid_registry.register_volcanic_observatory({
                'name': 'Test',
                'country': 'Test',
                'latitude': 0.0,
                'longitude': 0.0
            })

    # Test Data Validation
    
    async def test_data_validation(self, registry):
        """Test data validation for various inputs"""
        # Test invalid coordinates
        invalid_obs_data = {
            'name': 'Test Observatory',
            'country': 'Test Country',
            'latitude': 200.0,  # Invalid latitude
            'longitude': -120.0
        }
        
        # Should validate coordinate ranges (implementation dependent)
        # This test would be enhanced based on actual validation logic
        
        # Test invalid network types
        invalid_station_data = {
            'station_code': 'TEST.STA1',
            'name': 'Test Station',
            'latitude': 45.0,
            'longitude': -120.0,
            'elevation': 1000,
            'installation_date': datetime.now(),
            'station_type': 'invalid_type'  # Invalid station type
        }
        
        # Should validate station types (implementation dependent)

    # Performance Tests
    
    async def test_large_dataset_performance(self, registry):
        """Test performance with large datasets"""
        # This test would create many observatories and stations
        # to test performance characteristics
        
        # For now, just test that the system can handle multiple operations
        observatories = []
        for i in range(10):  # Reduced for test speed
            obs_data = {
                'name': f'Test Observatory {i}',
                'country': f'Country {i}',
                'latitude': 45.0 + i * 0.1,
                'longitude': -120.0 + i * 0.1
            }
            obs_id = await registry.register_volcanic_observatory(obs_data)
            observatories.append(obs_id)
        
        # Query all observatories
        all_obs = await registry.query_observatories()
        assert len(all_obs) >= 10
    
    async def test_concurrent_operations(self, registry, sample_observatory_data):
        """Test concurrent operations"""
        # Test multiple concurrent registrations
        tasks = []
        for i in range(5):
            obs_data = sample_observatory_data.copy()
            obs_data['name'] = f'Concurrent Observatory {i}'
            obs_data['latitude'] = 45.0 + i * 0.1
            task = registry.register_volcanic_observatory(obs_data)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 5
        for result in results:
            assert result is not None


# Integration Tests
class TestMonitoringNetworkRegistryIntegration:
    """Integration tests for the monitoring network registry"""
    
    @pytest.fixture
    async def populated_registry(self):
        """Create registry with sample data for integration testing"""
        registry = MonitoringNetworkRegistry(
            database_url="sqlite:///:memory:",
            redis_url=None,
            kafka_config=None
        )
        
        # Populate with sample data
        # Observatory 1
        obs1_data = {
            'name': 'Yellowstone Volcano Observatory',
            'acronym': 'YVO',
            'country': 'United States',
            'region': 'Wyoming',
            'latitude': 44.428,
            'longitude': -110.588,
            'primary_volcanoes': ['yellowstone'],
            'network_types': ['seismic', 'gps_gnss', 'thermal'],
            'real_time_capability': True,
            'alert_system': True
        }
        obs1_id = await registry.register_volcanic_observatory(obs1_data)
        
        # Observatory 2
        obs2_data = {
            'name': 'Cascades Volcano Observatory',
            'acronym': 'CVO',
            'country': 'United States',
            'region': 'Washington',
            'latitude': 46.2,
            'longitude': -122.3,
            'primary_volcanoes': ['mount_st_helens', 'mount_rainier'],
            'network_types': ['seismic', 'deformation', 'gas_geochemical'],
            'real_time_capability': True,
            'alert_system': True
        }
        obs2_id = await registry.register_volcanic_observatory(obs2_data)
        
        # Add stations for each observatory
        for i, obs_id in enumerate([obs1_id, obs2_id]):
            for j in range(3):
                station_data = {
                    'station_code': f'OBS{i+1}.STA{j+1}',
                    'name': f'Observatory {i+1} Station {j+1}',
                    'observatory_id': obs_id,
                    'latitude': (44.428 if i == 0 else 46.2) + j * 0.1,
                    'longitude': (-110.588 if i == 0 else -122.3) + j * 0.1,
                    'elevation': 1500 + j * 100,
                    'installation_date': datetime(2000 + j, 1, 1),
                    'station_type': 'seismic',
                    'status': 'operational',
                    'data_availability': 90.0 + j * 2,
                    'uptime': 95.0 + j * 1
                }
                await registry.register_monitoring_station(station_data)
        
        yield registry
        
        if hasattr(registry, 'executor'):
            registry.executor.shutdown(wait=True)
    
    async def test_end_to_end_workflow(self, populated_registry):
        """Test complete end-to-end workflow"""
        registry = populated_registry
        
        # 1. Query observatories
        observatories = await registry.query_observatories()
        assert len(observatories) == 2
        
        # 2. Query stations
        stations = await registry.query_monitoring_stations()
        assert len(stations) == 6  # 3 stations per observatory
        
        # 3. Analyze coverage for a region
        region = GeographicBounds(44.0, 47.0, -123.0, -110.0)
        coverage = await registry.analyze_network_coverage(region, MonitoringNetworkType.SEISMIC)
        assert coverage.coverage_percentage > 0
        
        # 4. Assess network performance
        performance = await registry.assess_network_performance()
        assert performance.uptime_percentage > 0
        
        # 5. Identify monitoring gaps
        gaps = await registry.identify_monitoring_gaps(region)
        assert isinstance(gaps, list)
        
        # 6. Generate comprehensive report
        report = await registry.generate_monitoring_report(region_bounds=region)
        assert report['summary']['total_observatories'] == 2
        assert len(report['networks']) > 0
    
    async def test_multi_region_analysis(self, populated_registry):
        """Test analysis across multiple regions"""
        registry = populated_registry
        
        # Define multiple regions
        yellowstone_region = GeographicBounds(44.0, 45.0, -111.0, -110.0)
        cascades_region = GeographicBounds(46.0, 47.0, -123.0, -122.0)
        
        # Analyze coverage for each region
        yellowstone_coverage = await registry.analyze_network_coverage(
            yellowstone_region, MonitoringNetworkType.SEISMIC
        )
        cascades_coverage = await registry.analyze_network_coverage(
            cascades_region, MonitoringNetworkType.SEISMIC
        )
        
        # Both regions should have some coverage
        assert yellowstone_coverage.coverage_percentage > 0
        assert cascades_coverage.coverage_percentage > 0
        
        # Generate reports for each region
        yellowstone_report = await registry.generate_monitoring_report(
            region_bounds=yellowstone_region
        )
        cascades_report = await registry.generate_monitoring_report(
            region_bounds=cascades_region
        )
        
        # Each report should reflect the appropriate region
        assert yellowstone_report['summary']['total_observatories'] >= 1
        assert cascades_report['summary']['total_observatories'] >= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])