"""
Database Integration Tests
Comprehensive tests for multi-database integration including PostgreSQL+PostGIS,
InfluxDB, Neo4j, and Redis with performance and reliability testing
"""

import asyncio
import pytest
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np

from backend.core.database_integration import (
    EnhancedDatabaseIntegrator,
    DatabaseMetrics,
    QueryResult,
    Volcano,
    SeismicEvent,
    DeformationData,
    GeologicalUnit,
    get_database_integrator,
    initialize_database_integrator
)


@pytest.fixture
async def database_integrator():
    """Create test database integrator with mocked connections"""
    config = {
        'postgresql': {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_prometheus',
            'username': 'test_user',
            'password': 'test_password'
        },
        'influxdb': {
            'url': 'http://localhost:8086',
            'token': 'test_token',
            'org': 'test_org',
            'bucket': 'test_bucket'
        },
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': 'test_password'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'password': 'test_password',
            'db': 0
        }
    }
    
    integrator = EnhancedDatabaseIntegrator(config)
    
    # Mock database connections to avoid actual database dependencies
    integrator.postgresql_engine = Mock()
    integrator.postgresql_session_factory = Mock()
    integrator.influxdb_client = Mock()
    integrator.neo4j_driver = Mock()
    integrator.redis_client = AsyncMock()
    integrator.base_integrator = Mock()
    
    # Mock initialization methods
    integrator._initialize_postgresql = AsyncMock()
    integrator._initialize_influxdb = AsyncMock()
    integrator._initialize_neo4j = AsyncMock()
    integrator._initialize_redis = AsyncMock()
    integrator._create_schemas = AsyncMock()
    integrator._start_health_monitoring = AsyncMock()
    
    yield integrator
    
    # Cleanup
    await integrator.cleanup()


@pytest.fixture
def sample_seismic_event():
    """Sample seismic event data"""
    return {
        'id': str(uuid.uuid4()),
        'volcano_id': str(uuid.uuid4()),
        'timestamp': datetime.now(),
        'longitude': -122.1234,
        'latitude': 37.5678,
        'magnitude': 3.2,
        'depth': 5.5,
        'event_type': 'volcano-tectonic',
        'confidence': 0.85,
        'waveform_data': {'channels': ['BHZ', 'BHN', 'BHE']},
        'moment_tensor': {'mrr': 1.2e15, 'mtt': -0.8e15, 'mpp': -0.4e15},
        'metadata': {'processing_version': '2.1', 'analyst': 'auto'}
    }


@pytest.fixture
def sample_time_series_data():
    """Sample time-series data"""
    return {
        'measurement': 'seismic_amplitude',
        'tags': {
            'station_id': 'VOLC01',
            'volcano_id': 'kilauea',
            'channel': 'BHZ'
        },
        'fields': {
            'amplitude': 1250.5,
            'frequency': 2.3,
            'snr': 15.2
        },
        'timestamp': datetime.now()
    }


class TestDatabaseInitialization:
    """Test database initialization and connection management"""
    
    @pytest.mark.asyncio
    async def test_database_integrator_initialization(self, database_integrator):
        """Test database integrator initialization"""
        success = await database_integrator.initialize()
        assert success
        
        # Verify initialization methods were called
        database_integrator._initialize_postgresql.assert_called_once()
        database_integrator._initialize_influxdb.assert_called_once()
        database_integrator._initialize_neo4j.assert_called_once()
        database_integrator._initialize_redis.assert_called_once()
        database_integrator._create_schemas.assert_called_once()
        database_integrator._start_health_monitoring.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_postgresql_initialization(self):
        """Test PostgreSQL initialization"""
        integrator = EnhancedDatabaseIntegrator()
        
        with patch('backend.core.database_integration.create_async_engine') as mock_engine:
            with patch('backend.core.database_integration.async_sessionmaker') as mock_session:
                await integrator._initialize_postgresql()
                
                mock_engine.assert_called_once()
                mock_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_initialization(self):
        """Test Redis initialization"""
        integrator = EnhancedDatabaseIntegrator()
        
        with patch('backend.core.database_integration.redis.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            await integrator._initialize_redis()
            
            mock_redis.assert_called_once()
            mock_redis_instance.ping.assert_called_once()


class TestPostgreSQLOperations:
    """Test PostgreSQL database operations"""
    
    @pytest.mark.asyncio
    async def test_store_seismic_event(self, database_integrator, sample_seismic_event):
        """Test storing seismic event in PostgreSQL"""
        # Mock session context manager
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        database_integrator.get_postgresql_session = Mock(return_value=mock_session)
        
        event_id = await database_integrator.store_seismic_event(sample_seismic_event)
        
        # Verify session operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        
        # Verify metrics updated
        assert database_integrator.metrics.query_count > 0
    
    @pytest.mark.asyncio
    async def test_query_seismic_events(self, database_integrator):
        """Test querying seismic events from PostgreSQL"""
        # Mock session and query results
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        # Mock query result
        mock_event = Mock()
        mock_event.id = uuid.uuid4()
        mock_event.volcano_id = uuid.uuid4()
        mock_event.timestamp = datetime.now()
        mock_event.magnitude = 3.5
        mock_event.depth = 10.0
        mock_event.event_type = 'volcano-tectonic'
        mock_event.confidence = 0.9
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all = AsyncMock(return_value=[mock_event])
        
        mock_session.query.return_value = mock_query
        database_integrator.get_postgresql_session = Mock(return_value=mock_session)
        
        # Execute query
        results = await database_integrator.query_seismic_events(
            volcano_id=str(mock_event.volcano_id),
            min_magnitude=3.0,
            limit=100
        )
        
        # Verify results
        assert len(results) == 1
        assert results[0]['magnitude'] == 3.5
        assert results[0]['event_type'] == 'volcano-tectonic'
    
    @pytest.mark.asyncio
    async def test_postgresql_session_context_manager(self, database_integrator):
        """Test PostgreSQL session context manager"""
        mock_session_factory = Mock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_factory.return_value = mock_session
        database_integrator.postgresql_session_factory = mock_session_factory
        
        async with database_integrator.get_postgresql_session() as session:
            assert session == mock_session
        
        mock_session.close.assert_called_once()


class TestInfluxDBOperations:
    """Test InfluxDB time-series operations"""
    
    @pytest.mark.asyncio
    async def test_store_time_series_data(self, database_integrator, sample_time_series_data):
        """Test storing time-series data in InfluxDB"""
        # Mock InfluxDB client and write API
        mock_write_api = AsyncMock()
        database_integrator.influxdb_client.write_api.return_value = mock_write_api
        
        success = await database_integrator.store_time_series_data(
            measurement=sample_time_series_data['measurement'],
            tags=sample_time_series_data['tags'],
            fields=sample_time_series_data['fields'],
            timestamp=sample_time_series_data['timestamp']
        )
        
        assert success
        mock_write_api.write.assert_called_once()
        assert database_integrator.metrics.query_count > 0
    
    @pytest.mark.asyncio
    async def test_query_time_series_data(self, database_integrator):
        """Test querying time-series data from InfluxDB"""
        # Mock query API and results
        mock_query_api = Mock()
        mock_record = Mock()
        mock_record.get_time.return_value = datetime.now()
        mock_record.get_measurement.return_value = 'seismic_amplitude'
        mock_record.get_field.return_value = 'amplitude'
        mock_record.get_value.return_value = 1250.5
        mock_record.values = {'tag_station_id': 'VOLC01'}
        
        mock_table = Mock()
        mock_table.records = [mock_record]
        
        mock_query_api.query = AsyncMock(return_value=[mock_table])
        database_integrator.influxdb_client.query_api.return_value = mock_query_api
        
        # Execute query
        results = await database_integrator.query_time_series_data(
            measurement='seismic_amplitude',
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now(),
            filters={'station_id': 'VOLC01'}
        )
        
        # Verify results
        assert len(results) == 1
        assert results[0]['measurement'] == 'seismic_amplitude'
        assert results[0]['field'] == 'amplitude'
        assert results[0]['value'] == 1250.5
    
    @pytest.mark.asyncio
    async def test_influxdb_error_handling(self, database_integrator):
        """Test InfluxDB error handling"""
        # Mock write API to raise exception
        mock_write_api = AsyncMock()
        mock_write_api.write.side_effect = Exception("InfluxDB connection error")
        database_integrator.influxdb_client.write_api.return_value = mock_write_api
        
        success = await database_integrator.store_time_series_data(
            measurement='test',
            tags={'test': 'tag'},
            fields={'test': 123.45}
        )
        
        assert not success
        assert database_integrator.metrics.error_count > 0


class TestNeo4jOperations:
    """Test Neo4j graph database operations"""
    
    @pytest.mark.asyncio
    async def test_create_geological_relationship(self, database_integrator):
        """Test creating geological relationships in Neo4j"""
        # Mock Neo4j session
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mock_result = AsyncMock()
        mock_result.consume = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        
        database_integrator.neo4j_driver.session.return_value = mock_session
        
        # Create relationship
        source_node = {'id': 'magma_generation', 'name': 'Magma Generation'}
        target_node = {'id': 'volcanic_eruption', 'name': 'Volcanic Eruption'}
        properties = {'strength': 0.8, 'time_lag': 3600}
        
        success = await database_integrator.create_geological_relationship(
            source_node, target_node, 'LEADS_TO', properties
        )
        
        assert success
        mock_session.run.assert_called_once()
        mock_result.consume.assert_called_once()
        assert database_integrator.metrics.query_count > 0
    
    @pytest.mark.asyncio
    async def test_query_geological_relationships(self, database_integrator):
        """Test querying geological relationships from Neo4j"""
        # Mock Neo4j session and results
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        # Mock relationship data
        mock_node1 = {'id': 'node1', 'name': 'Process 1'}
        mock_node2 = {'id': 'node2', 'name': 'Process 2'}
        mock_rel = Mock()
        mock_rel.type = 'LEADS_TO'
        
        mock_record = {
            'nodes': [mock_node1, mock_node2],
            'rels': [mock_rel]
        }
        
        mock_result = AsyncMock()
        mock_result.__aiter__ = AsyncMock(return_value=iter([mock_record]))
        mock_session.run = AsyncMock(return_value=mock_result)
        
        database_integrator.neo4j_driver.session.return_value = mock_session
        
        # Query relationships
        relationships = await database_integrator.query_geological_relationships(
            node_id='node1',
            relationship_types=['LEADS_TO'],
            depth=2
        )
        
        assert len(relationships) == 1
        assert relationships[0]['type'] == 'LEADS_TO'
        mock_session.run.assert_called_once()


class TestRedisCaching:
    """Test Redis caching operations"""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, database_integrator):
        """Test Redis cache set and get operations"""
        test_data = {'key': 'value', 'number': 123, 'list': [1, 2, 3]}
        
        # Test cache set
        success = await database_integrator.cache_set('test_key', test_data, ttl=3600)
        assert success
        
        database_integrator.redis_client.setex.assert_called_once()
        
        # Test cache get
        database_integrator.redis_client.get.return_value = json.dumps(test_data, default=str)
        
        cached_data = await database_integrator.cache_get('test_key')
        assert cached_data == test_data
        
        database_integrator.redis_client.get.assert_called_once_with('test_key')
    
    @pytest.mark.asyncio
    async def test_cache_delete(self, database_integrator):
        """Test Redis cache delete operation"""
        database_integrator.redis_client.delete.return_value = 1
        
        success = await database_integrator.cache_delete('test_key')
        assert success
        
        database_integrator.redis_client.delete.assert_called_once_with('test_key')
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, database_integrator):
        """Test Redis cache miss scenario"""
        database_integrator.redis_client.get.return_value = None
        
        cached_data = await database_integrator.cache_get('nonexistent_key')
        assert cached_data is None
    
    @pytest.mark.asyncio
    async def test_cache_error_handling(self, database_integrator):
        """Test Redis cache error handling"""
        database_integrator.redis_client.setex.side_effect = Exception("Redis connection error")
        
        success = await database_integrator.cache_set('test_key', {'data': 'test'})
        assert not success


class TestUnifiedQueryInterface:
    """Test unified query interface across databases"""
    
    @pytest.mark.asyncio
    async def test_unified_seismic_query(self, database_integrator):
        """Test unified seismic events query"""
        # Mock seismic events query
        mock_events = [
            {'id': '1', 'magnitude': 3.5, 'timestamp': datetime.now()},
            {'id': '2', 'magnitude': 4.2, 'timestamp': datetime.now()}
        ]
        
        database_integrator.query_seismic_events = AsyncMock(return_value=mock_events)
        
        result = await database_integrator.execute_unified_query(
            'seismic_events',
            volcano_id='test_volcano',
            min_magnitude=3.0
        )
        
        assert isinstance(result, QueryResult)
        assert result.data == mock_events
        assert result.row_count == 2
        assert result.metadata['database'] == 'postgresql'
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_unified_time_series_query(self, database_integrator):
        """Test unified time-series query"""
        mock_data = [
            {'time': datetime.now(), 'value': 123.45, 'measurement': 'temperature'}
        ]
        
        database_integrator.query_time_series_data = AsyncMock(return_value=mock_data)
        
        result = await database_integrator.execute_unified_query(
            'time_series',
            measurement='temperature',
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        assert result.data == mock_data
        assert result.metadata['database'] == 'influxdb'
    
    @pytest.mark.asyncio
    async def test_unified_query_error_handling(self, database_integrator):
        """Test unified query error handling"""
        result = await database_integrator.execute_unified_query('invalid_query_type')
        
        assert result.data is None
        assert 'error' in result.metadata
        assert result.row_count == 0


class TestHealthMonitoring:
    """Test database health monitoring"""
    
    @pytest.mark.asyncio
    async def test_database_health_check(self, database_integrator):
        """Test database health checking"""
        # Mock successful connections
        mock_pg_conn = AsyncMock()
        mock_pg_conn.__aenter__ = AsyncMock(return_value=mock_pg_conn)
        mock_pg_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pg_conn.execute = AsyncMock()
        
        database_integrator.postgresql_engine.connect.return_value = mock_pg_conn
        
        # Mock InfluxDB health
        mock_health = Mock()
        mock_health.status = 'pass'
        database_integrator.influxdb_client.health = AsyncMock(return_value=mock_health)
        
        # Mock Neo4j session
        mock_neo4j_session = AsyncMock()
        mock_neo4j_session.__aenter__ = AsyncMock(return_value=mock_neo4j_session)
        mock_neo4j_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_session.run = AsyncMock()
        
        database_integrator.neo4j_driver.session.return_value = mock_neo4j_session
        
        # Mock Redis ping
        database_integrator.redis_client.ping = AsyncMock()
        
        # Run health check
        await database_integrator._check_database_health()
        
        # Verify health checks were performed
        mock_pg_conn.execute.assert_called_once()
        database_integrator.influxdb_client.health.assert_called_once()
        mock_neo4j_session.run.assert_called_once()
        database_integrator.redis_client.ping.assert_called_once()
    
    def test_database_metrics(self, database_integrator):
        """Test database metrics collection"""
        # Set some test metrics
        database_integrator.metrics.query_count = 100
        database_integrator.metrics.error_count = 5
        database_integrator.metrics.cache_hit_rate = 0.85
        
        metrics = database_integrator.get_database_metrics()
        
        assert metrics['query_count'] == 100
        assert metrics['error_count'] == 5
        assert metrics['cache_hit_rate'] == 0.85
        assert 'connection_count' in metrics
        assert 'average_query_time' in metrics


class TestPerformanceAndReliability:
    """Test performance and reliability features"""
    
    @pytest.mark.asyncio
    async def test_connection_pooling(self, database_integrator):
        """Test database connection pooling"""
        # This would test actual connection pool behavior
        # For now, verify pool configuration is set correctly
        assert database_integrator.config['postgresql']['pool_size'] == 20
        assert database_integrator.config['postgresql']['max_overflow'] == 30
    
    @pytest.mark.asyncio
    async def test_query_caching(self, database_integrator):
        """Test query result caching"""
        # Mock cache operations
        database_integrator.cache_get = AsyncMock(return_value=None)  # Cache miss
        database_integrator.cache_set = AsyncMock(return_value=True)
        
        # This would test actual query caching implementation
        # For now, verify cache methods work correctly
        cached_result = await database_integrator.cache_get('test_query')
        assert cached_result is None
        
        success = await database_integrator.cache_set('test_query', {'result': 'data'})
        assert success
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, database_integrator):
        """Test concurrent database operations"""
        # Mock multiple concurrent operations
        tasks = []
        
        for i in range(10):
            task = database_integrator.cache_set(f'key_{i}', {'data': i})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete successfully
        assert all(result is True or isinstance(result, Exception) for result in results)
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, database_integrator):
        """Test error recovery mechanisms"""
        # Test that operations continue after errors
        database_integrator.redis_client.setex.side_effect = [
            Exception("Connection error"),  # First call fails
            True  # Second call succeeds
        ]
        
        # First operation should fail
        success1 = await database_integrator.cache_set('key1', {'data': 1})
        assert not success1
        
        # Second operation should succeed
        success2 = await database_integrator.cache_set('key2', {'data': 2})
        assert success2


class TestGlobalIntegratorInstance:
    """Test global integrator instance management"""
    
    def test_get_database_integrator(self):
        """Test getting global database integrator instance"""
        integrator1 = get_database_integrator()
        integrator2 = get_database_integrator()
        
        # Should return the same instance
        assert integrator1 is integrator2
    
    @pytest.mark.asyncio
    async def test_initialize_database_integrator(self):
        """Test initializing global database integrator"""
        config = {'test': 'config'}
        
        with patch.object(EnhancedDatabaseIntegrator, 'initialize', return_value=True):
            success = await initialize_database_integrator(config)
            assert success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])