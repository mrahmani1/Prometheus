"""
Test Suite for Real Data Connectors Framework

Comprehensive tests for all real data source connectors, health monitoring,
and failover mechanisms.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from core.real_data_connectors import (
    BaseDataConnector,
    DataSourceConfig,
    ConnectionStatus,
    DataSourceType,
    GeologicalObservation,
    IRISConnector,
    USGSVolcanoConnector,
    NASAEarthDataConnector,
    CopernicusConnector,
    ASFHyp3Connector,
    GeochemConnector,
    GeologicalSurveyConnector,
    DataSourceHealthMonitor,
    ConnectorFactory
)


class TestDataSourceConfig:
    """Test DataSourceConfig class"""
    
    def test_config_creation(self):
        config = DataSourceConfig(
            source_id="test_source",
            name="Test Source",
            base_url="https://api.test.com",
            api_key="test_key",
            timeout=30,
            max_retries=3,
            rate_limit=100,
            data_type=DataSourceType.SEISMIC
        )
        
        assert config.source_id == "test_source"
        assert config.name == "Test Source"
        assert config.base_url == "https://api.test.com"
        assert config.api_key == "test_key"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.rate_limit == 100
        assert config.data_type == DataSourceType.SEISMIC


class TestGeologicalObservation:
    """Test GeologicalObservation data model"""
    
    def test_observation_creation(self):
        obs = GeologicalObservation(
            observation_id="test_obs_001",
            timestamp=datetime.now(),
            latitude=45.0,
            longitude=-120.0,
            elevation=1000.0,
            observation_type="earthquake_event",
            value={"magnitude": 5.2, "depth_km": 10.0},
            uncertainty=0.1,
            data_source="Test Source",
            quality_flags=["high_quality"],
            metadata={"network": "TEST"}
        )
        
        assert obs.observation_id == "test_obs_001"
        assert obs.latitude == 45.0
        assert obs.longitude == -120.0
        assert obs.elevation == 1000.0
        assert obs.observation_type == "earthquake_event"
        assert obs.value["magnitude"] == 5.2
        assert obs.uncertainty == 0.1
        assert obs.data_source == "Test Source"
        assert "high_quality" in obs.quality_flags
        assert obs.metadata["network"] == "TEST"


class MockConnector(BaseDataConnector):
    """Mock connector for testing"""
    
    def __init__(self, config: DataSourceConfig, should_fail: bool = False):
        super().__init__(config)
        self.should_fail = should_fail
        self.auth_calls = 0
        self.request_calls = 0
    
    async def _authenticate(self) -> bool:
        self.auth_calls += 1
        return not self.should_fail
    
    async def _get_auth_headers(self) -> dict:
        return {"Authorization": "Bearer test_token"}
    
    async def fetch_real_time_data(self, **kwargs):
        if self.should_fail:
            raise Exception("Mock failure")
        
        yield GeologicalObservation(
            observation_id="mock_obs_001",
            timestamp=datetime.now(),
            latitude=45.0,
            longitude=-120.0,
            observation_type="test_observation",
            value={"test_value": 123},
            data_source="Mock Source"
        )
    
    async def fetch_historical_data(self, start_time: datetime, end_time: datetime, **kwargs):
        if self.should_fail:
            raise Exception("Mock failure")
        
        return [
            GeologicalObservation(
                observation_id="mock_historical_001",
                timestamp=start_time,
                latitude=45.0,
                longitude=-120.0,
                observation_type="test_observation",
                value={"test_value": 456},
                data_source="Mock Source"
            )
        ]
    
    async def validate_data_quality(self, observation: GeologicalObservation) -> bool:
        return True


class TestBaseDataConnector:
    """Test BaseDataConnector functionality"""
    
    @pytest.fixture
    def config(self):
        return DataSourceConfig(
            source_id="test_connector",
            name="Test Connector",
            base_url="https://api.test.com",
            timeout=30,
            max_retries=3,
            rate_limit=100
        )
    
    @pytest.mark.asyncio
    async def test_connector_connection(self, config):
        connector = MockConnector(config)
        
        # Test successful connection
        status = await connector.connect()
        assert status == ConnectionStatus.CONNECTED
        assert connector.auth_calls == 1
        
        await connector.disconnect()
        assert connector.status == ConnectionStatus.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_connector_connection_failure(self, config):
        connector = MockConnector(config, should_fail=True)
        
        # Test failed connection
        status = await connector.connect()
        assert status == ConnectionStatus.ERROR
        assert connector.error_count > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, config):
        config.rate_limit = 2  # Very low limit for testing
        connector = MockConnector(config)
        await connector.connect()
        
        # Make requests that should trigger rate limiting
        start_time = datetime.now()
        
        # First two requests should be fast
        await connector._rate_limit_check()
        await connector._rate_limit_check()
        
        # Third request should be delayed
        await connector._rate_limit_check()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        # Should have been delayed due to rate limiting
        assert elapsed > 0.1  # Some delay expected
    
    @pytest.mark.asyncio
    async def test_health_check(self, config):
        connector = MockConnector(config)
        await connector.connect()
        
        health_result = await connector.health_check()
        
        assert health_result['status'] == 'healthy'
        assert 'response_time_ms' in health_result
        assert 'error_count' in health_result
        assert 'last_check' in health_result
    
    @pytest.mark.asyncio
    async def test_real_time_data_fetch(self, config):
        connector = MockConnector(config)
        await connector.connect()
        
        observations = []
        async for obs in connector.fetch_real_time_data():
            observations.append(obs)
            break  # Just get one observation for testing
        
        assert len(observations) == 1
        assert observations[0].observation_id == "mock_obs_001"
        assert observations[0].data_source == "Mock Source"
    
    @pytest.mark.asyncio
    async def test_historical_data_fetch(self, config):
        connector = MockConnector(config)
        await connector.connect()
        
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        
        observations = await connector.fetch_historical_data(start_time, end_time)
        
        assert len(observations) == 1
        assert observations[0].observation_id == "mock_historical_001"
        assert observations[0].data_source == "Mock Source"


class TestDataSourceHealthMonitor:
    """Test DataSourceHealthMonitor functionality"""
    
    @pytest.fixture
    def health_monitor(self):
        return DataSourceHealthMonitor()
    
    @pytest.fixture
    def mock_connector(self):
        config = DataSourceConfig(
            source_id="test_source",
            name="Test Source",
            base_url="https://api.test.com"
        )
        return MockConnector(config)
    
    @pytest.mark.asyncio
    async def test_connector_registration(self, health_monitor, mock_connector):
        source_id = "test_source_001"
        
        health_monitor.register_connector(source_id, mock_connector)
        
        assert source_id in health_monitor.connectors
        assert source_id in health_monitor.health_metrics
        assert source_id in health_monitor.monitoring_tasks
    
    @pytest.mark.asyncio
    async def test_health_status_tracking(self, health_monitor, mock_connector):
        source_id = "test_source_002"
        
        health_monitor.register_connector(source_id, mock_connector)
        
        # Simulate successful health check
        await health_monitor._record_success(source_id, 100.0)
        
        metrics = health_monitor.get_health_status(source_id)
        assert metrics.status.value == "healthy"
        assert metrics.response_time_ms == 100.0
        assert metrics.consecutive_failures == 0
        
        # Simulate failed health check
        await health_monitor._record_failure(source_id, "Test error")
        
        metrics = health_monitor.get_health_status(source_id)
        assert metrics.consecutive_failures == 1
        assert metrics.error_count == 1
    
    @pytest.mark.asyncio
    async def test_health_report_export(self, health_monitor, mock_connector):
        source_id = "test_source_003"
        
        health_monitor.register_connector(source_id, mock_connector)
        await health_monitor._record_success(source_id, 150.0)
        
        report = health_monitor.export_health_report()
        
        assert 'timestamp' in report
        assert 'sources' in report
        assert 'summary' in report
        assert source_id in report['sources']
        assert report['sources'][source_id]['status'] == 'healthy'
        assert report['summary']['healthy_sources'] == 1


class TestConnectorFactory:
    """Test ConnectorFactory functionality"""
    
    @pytest.fixture
    def factory(self):
        return ConnectorFactory()
    
    @pytest.mark.asyncio
    async def test_factory_initialization(self, factory):
        # Mock the config loading to avoid file dependencies
        with patch.object(factory, '_load_config') as mock_load:
            mock_load.return_value = {
                'data_sources': {},
                'health_monitoring': {},
                'failover': {}
            }
            
            await factory.initialize()
            
            assert mock_load.called
            assert isinstance(factory.health_monitor, DataSourceHealthMonitor)
    
    def test_connector_type_registration(self, factory):
        # Test that all expected connector types are registered
        expected_types = [
            'iris', 'usgs_volcano', 'nasa_earthdata', 
            'copernicus', 'asf_hyp3', 'geochem', 'geological_survey'
        ]
        
        for connector_type in expected_types:
            assert connector_type in factory.connector_types
    
    def test_default_config_creation(self, factory):
        # Test that default configurations exist for all connector types
        for connector_type in factory.connector_types.keys():
            assert connector_type in factory.default_configs
            
            config = factory.default_configs[connector_type]
            assert 'name' in config
            assert 'base_url' in config
            assert 'timeout' in config
            assert 'max_retries' in config
            assert 'rate_limit' in config
            assert 'data_type' in config
    
    @pytest.mark.asyncio
    async def test_connector_testing(self, factory):
        # Add a mock connector for testing
        config = DataSourceConfig(
            source_id="test_connector",
            name="Test Connector",
            base_url="https://api.test.com"
        )
        mock_connector = MockConnector(config)
        factory.connectors["test_connector"] = mock_connector
        
        # Test the connector
        result = await factory.test_connector("test_connector")
        
        assert result['status'] == 'success'
        assert result['source_id'] == 'test_connector'
        assert 'health_check' in result
        assert 'connection_status' in result


class TestIRISConnector:
    """Test IRIS/EarthScope connector"""
    
    @pytest.fixture
    def iris_config(self):
        return DataSourceConfig(
            source_id="iris_test",
            name="IRIS Test",
            base_url="http://service.iris.edu",
            data_type=DataSourceType.SEISMIC
        )
    
    @pytest.mark.asyncio
    async def test_iris_authentication(self, iris_config):
        connector = IRISConnector(iris_config)
        
        # Mock the FDSN client creation
        with patch('core.real_data_connectors.iris_connector.Client') as mock_client:
            mock_client.return_value.get_events.return_value = []
            
            auth_result = await connector._authenticate()
            assert auth_result is True
    
    def test_iris_quality_assessment(self, iris_config):
        connector = IRISConnector(iris_config)
        
        # Test event quality assessment
        origin_mock = Mock()
        origin_mock.quality.standard_error = 0.5
        origin_mock.quality.azimuthal_gap = 90
        origin_mock.quality.used_phase_count = 15
        
        magnitude_mock = Mock()
        magnitude_mock.mag_errors.uncertainty = 0.2
        
        flags = connector._assess_event_quality(origin_mock, magnitude_mock)
        assert len(flags) == 0  # Should be good quality
        
        # Test poor quality event
        origin_mock.quality.standard_error = 2.0
        origin_mock.quality.azimuthal_gap = 200
        origin_mock.quality.used_phase_count = 5
        magnitude_mock.mag_errors.uncertainty = 0.5
        
        flags = connector._assess_event_quality(origin_mock, magnitude_mock)
        assert "high_location_uncertainty" in flags
        assert "large_azimuthal_gap" in flags
        assert "few_phases" in flags
        assert "high_magnitude_uncertainty" in flags


class TestUSGSVolcanoConnector:
    """Test USGS Volcano Hazards Program connector"""
    
    @pytest.fixture
    def usgs_config(self):
        return DataSourceConfig(
            source_id="usgs_test",
            name="USGS Test",
            base_url="https://api.usgs.gov/volcano",
            data_type=DataSourceType.VOLCANIC
        )
    
    @pytest.mark.asyncio
    async def test_usgs_authentication(self, usgs_config):
        connector = USGSVolcanoConnector(usgs_config)
        
        # Mock successful API response
        with patch.object(connector, '_make_request') as mock_request:
            mock_request.return_value = {'volcanoes': []}
            
            auth_result = await connector._authenticate()
            assert auth_result is True
    
    def test_usgs_alert_quality_assessment(self, usgs_config):
        connector = USGSVolcanoConnector(usgs_config)
        
        # Test good quality alert
        alert = {
            'alert_level': 'orange',
            'description': 'Volcanic unrest detected',
            'confidence_level': 'high'
        }
        
        flags = connector._assess_alert_quality(alert)
        assert len(flags) == 0
        
        # Test poor quality alert
        alert = {
            'confidence_level': 'low'
        }
        
        flags = connector._assess_alert_quality(alert)
        assert "missing_alert_level" in flags
        assert "missing_description" in flags
        assert "low_confidence" in flags


@pytest.mark.asyncio
async def test_data_validation():
    """Test data quality validation across all connectors"""
    
    # Test valid observation
    valid_obs = GeologicalObservation(
        observation_id="test_001",
        timestamp=datetime.now(),
        latitude=45.0,
        longitude=-120.0,
        observation_type="earthquake_event",
        value={"magnitude": 5.2, "depth_km": 10.0},
        data_source="Test"
    )
    
    config = DataSourceConfig(
        source_id="test",
        name="Test",
        base_url="https://test.com"
    )
    
    connector = MockConnector(config)
    assert await connector.validate_data_quality(valid_obs) is True
    
    # Test invalid observation (bad coordinates)
    invalid_obs = GeologicalObservation(
        observation_id="test_002",
        timestamp=datetime.now(),
        latitude=95.0,  # Invalid latitude
        longitude=-200.0,  # Invalid longitude
        observation_type="earthquake_event",
        value={"magnitude": 5.2},
        data_source="Test"
    )
    
    # Should fail validation due to invalid coordinates
    # Note: This would need to be implemented in the actual validate_data_quality method


@pytest.mark.asyncio
async def test_integration_workflow():
    """Test complete integration workflow"""
    
    # Create factory
    factory = ConnectorFactory()
    
    # Mock configuration
    mock_config = {
        'data_sources': {
            'test_source': {
                'type': 'iris',
                'name': 'Test IRIS',
                'base_url': 'http://service.iris.edu',
                'enabled': True,
                'timeout': 30,
                'max_retries': 3,
                'rate_limit': 100,
                'data_type': 'seismic'
            }
        },
        'health_monitoring': {
            'check_interval': 60
        },
        'failover': {}
    }
    
    # Mock the config loading and connector creation
    with patch.object(factory, '_load_config') as mock_load, \
         patch.object(factory, '_create_connector') as mock_create:
        
        mock_load.return_value = mock_config
        
        await factory.initialize()
        
        # Verify initialization
        assert mock_load.called
        assert mock_create.called
        
        # Test health report generation
        report = factory.get_health_report()
        assert 'timestamp' in report
        assert 'sources' in report
        assert 'summary' in report


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])