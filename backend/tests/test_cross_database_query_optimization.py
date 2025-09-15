"""
Comprehensive Test Suite for Cross-Database Query Optimization System
Tests all components: optimization, caching, auditing, distributed execution, and integration
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

# Import components to test
from backend.core.cross_database_query_optimizer import (
    CrossDatabaseQueryOptimizer, QueryRouter, QueryCache, CrossDatabaseJoinExecutor,
    QueryPerformanceMonitor, QueryType, OptimizationStrategy
)
from backend.core.query_audit_logger import (
    QueryAuditLogger, QueryAuditEntry, PerformanceAlert, QueryCategory, AuditLevel
)
from backend.core.distributed_query_executor import (
    DistributedQueryExecutor, QueryPartitioner, SpatialPartitioner, TemporalPartitioner,
    PartitionStrategy, ExecutionMode
)
from backend.core.integrated_query_system import (
    IntegratedQuerySystem, QueryRequest, QueryResponse, QueryComplexity, ExecutionStrategy
)
from backend.core.unified_database_manager import (
    DatabaseType, QueryResult, UnifiedDatabaseManager
)

class TestQueryRouter:
    """Test query routing functionality"""
    
    @pytest.fixture
    def query_router(self):
        return QueryRouter()
    
    def test_spatial_query_routing(self, query_router):
        """Test routing of spatial queries to PostgreSQL"""
        spatial_query = "SELECT * FROM geological_observations WHERE ST_Within(location, ST_MakeEnvelope(-120, 35, -115, 40, 4326))"
        
        routed_db = query_router.route_query(spatial_query, QueryType.SPATIAL)
        assert routed_db == DatabaseType.POSTGRESQL
    
    def test_time_series_query_routing(self, query_router):
        """Test routing of time-series queries to InfluxDB"""
        time_series_query = "from(bucket: 'geological_data') |> range(start: -1h) |> filter(fn: (r) => r._measurement == 'seismic_data')"
        
        routed_db = query_router.route_query(time_series_query, QueryType.TIME_SERIES)
        assert routed_db == DatabaseType.INFLUXDB
    
    def test_graph_query_routing(self, query_router):
        """Test routing of graph queries to Neo4j"""
        graph_query = "MATCH (p:GeologicalProcess)-[:CAUSES]->(e:Event) RETURN p, e"
        
        routed_db = query_router.route_query(graph_query, QueryType.GRAPH)
        assert routed_db == DatabaseType.NEO4J
    
    def test_load_balancing(self, query_router):
        """Test load balancing functionality"""
        # Simulate high load on PostgreSQL
        query_router.load_balancing_weights[DatabaseType.POSTGRESQL] = 0.1
        query_router.load_balancing_weights[DatabaseType.TIMESCALEDB] = 1.0
        
        # Should route to alternative database under load
        spatial_query = "SELECT * FROM geological_observations WHERE location IS NOT NULL"
        routed_db = query_router.route_query(spatial_query, QueryType.SPATIAL)
        
        # Should consider load balancing (implementation dependent)
        assert routed_db in [DatabaseType.POSTGRESQL, DatabaseType.TIMESCALEDB]

class TestQueryCache:
    """Test query caching functionality"""
    
    @pytest.fixture
    def query_cache(self):
        return QueryCache(max_size_mb=10)
    
    @pytest.mark.asyncio
    async def test_cache_put_and_get(self, query_cache):
        """Test basic cache put and get operations"""
        query = "SELECT * FROM geological_observations LIMIT 10"
        result = QueryResult(
            data=[{'id': 1, 'value': 'test'}],
            metadata={'test': True},
            execution_time=0.5,
            row_count=1,
            database_type=DatabaseType.POSTGRESQL
        )
        
        # Cache the result
        await query_cache.put(query, result)
        
        # Retrieve from cache
        cached_result = await query_cache.get(query)
        
        assert cached_result is not None
        assert cached_result.cached is True
        assert cached_result.data == result.data
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, query_cache):
        """Test cache TTL expiration"""
        query = "SELECT * FROM test_table"
        result = QueryResult(
            data=[{'id': 1}],
            metadata={},
            execution_time=0.1,
            row_count=1,
            database_type=DatabaseType.POSTGRESQL
        )
        
        # Cache with short TTL
        await query_cache.put(query, result, ttl=timedelta(milliseconds=100))
        
        # Should be available immediately
        cached_result = await query_cache.get(query)
        assert cached_result is not None
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired
        cached_result = await query_cache.get(query)
        assert cached_result is None
    
    def test_cache_statistics(self, query_cache):
        """Test cache statistics tracking"""
        stats = query_cache.get_statistics()
        
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate' in stats
        assert 'current_size_mb' in stats

class TestCrossDatabaseJoinExecutor:
    """Test cross-database join execution"""
    
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock(spec=UnifiedDatabaseManager)
        
        # Mock query results
        left_result = QueryResult(
            data=[{'id': 1, 'name': 'volcano1'}, {'id': 2, 'name': 'volcano2'}],
            metadata={},
            execution_time=0.5,
            row_count=2,
            database_type=DatabaseType.POSTGRESQL
        )
        
        right_result = QueryResult(
            data=[{'volcano_id': 1, 'magnitude': 5.2}, {'volcano_id': 2, 'magnitude': 4.8}],
            metadata={},
            execution_time=0.3,
            row_count=2,
            database_type=DatabaseType.INFLUXDB
        )
        
        db_manager.execute_query = AsyncMock(side_effect=[left_result, right_result])
        
        return db_manager
    
    @pytest.fixture
    def join_executor(self, mock_db_manager):
        return CrossDatabaseJoinExecutor(mock_db_manager)
    
    @pytest.mark.asyncio
    async def test_cross_database_join(self, join_executor):
        """Test cross-database join execution"""
        left_query = "SELECT id, name FROM volcanoes"
        right_query = "SELECT volcano_id, magnitude FROM seismic_events"
        join_condition = "volcanoes.id = seismic_events.volcano_id"
        
        result = await join_executor.execute_cross_db_join(
            left_query, DatabaseType.POSTGRESQL,
            right_query, DatabaseType.INFLUXDB,
            join_condition, "INNER"
        )
        
        assert result is not None
        assert result.metadata['left_db'] == DatabaseType.POSTGRESQL.value
        assert result.metadata['right_db'] == DatabaseType.INFLUXDB.value
        assert 'join_strategy' in result.metadata

class TestQueryPerformanceMonitor:
    """Test query performance monitoring"""
    
    @pytest.fixture
    def performance_monitor(self):
        return QueryPerformanceMonitor()
    
    def test_query_classification(self, performance_monitor):
        """Test query type classification"""
        
        # Test different query types
        select_query = "SELECT * FROM geological_observations"
        classified = performance_monitor._classify_query(select_query)
        assert classified == 'select_simple'
        
        join_query = "SELECT * FROM a JOIN b ON a.id = b.id"
        classified = performance_monitor._classify_query(join_query)
        assert classified == 'select_join'
        
        spatial_query = "SELECT * FROM obs WHERE ST_Within(location, geom)"
        classified = performance_monitor._classify_query(spatial_query)
        assert classified == 'select_spatial'
    
    @pytest.mark.asyncio
    async def test_slow_query_detection(self, performance_monitor):
        """Test slow query detection and alerting"""
        
        # Mock alert callback
        alert_received = []
        
        async def mock_alert_callback(query, metrics):
            alert_received.append((query, metrics))
        
        performance_monitor.register_alert_callback(mock_alert_callback)
        
        # Record slow query
        from backend.core.cross_database_query_optimizer import QueryMetrics
        
        slow_metrics = QueryMetrics(
            execution_time=10.0,  # Exceeds threshold
            rows_processed=1000,
            memory_usage=100.0,
            cpu_usage=50.0,
            cache_hit=False,
            optimization_applied=[],
            database_type=DatabaseType.POSTGRESQL,
            query_hash="test_hash"
        )
        
        performance_monitor.record_query_metrics("SELECT * FROM large_table", slow_metrics)
        
        # Allow async callback to execute
        await asyncio.sleep(0.1)
        
        assert len(alert_received) == 1
    
    def test_performance_report_generation(self, performance_monitor):
        """Test performance report generation"""
        
        # Add some test metrics
        from backend.core.cross_database_query_optimizer import QueryMetrics
        
        for i in range(5):
            metrics = QueryMetrics(
                execution_time=i * 0.5,
                rows_processed=100 * i,
                memory_usage=10.0,
                cpu_usage=20.0,
                cache_hit=i % 2 == 0,
                optimization_applied=[],
                database_type=DatabaseType.POSTGRESQL,
                query_hash=f"hash_{i}"
            )
            performance_monitor.record_query_metrics(f"SELECT * FROM table_{i}", metrics)
        
        report = performance_monitor.get_performance_report()
        
        assert 'total_queries' in report
        assert 'avg_execution_time' in report
        assert 'database_statistics' in report

class TestQueryAuditLogger:
    """Test query audit logging functionality"""
    
    @pytest.fixture
    def audit_logger(self, tmp_path):
        return QueryAuditLogger(log_directory=str(tmp_path / "audit_logs"))
    
    @pytest.mark.asyncio
    async def test_query_execution_logging(self, audit_logger):
        """Test logging of query execution"""
        
        query = "SELECT * FROM geological_observations"
        result = QueryResult(
            data=[{'id': 1}],
            metadata={},
            execution_time=0.5,
            row_count=1,
            database_type=DatabaseType.POSTGRESQL
        )
        
        from backend.core.cross_database_query_optimizer import QueryMetrics
        
        metrics = QueryMetrics(
            execution_time=0.5,
            rows_processed=1,
            memory_usage=10.0,
            cpu_usage=5.0,
            cache_hit=False,
            optimization_applied=[],
            database_type=DatabaseType.POSTGRESQL,
            query_hash="test_hash"
        )
        
        query_id = await audit_logger.log_query_execution(query, result, metrics)
        
        assert query_id is not None
        assert query_id.startswith('q_')
    
    @pytest.mark.asyncio
    async def test_query_error_logging(self, audit_logger):
        """Test logging of query errors"""
        
        query = "SELECT * FROM nonexistent_table"
        error = Exception("Table does not exist")
        
        query_id = await audit_logger.log_query_error(
            query, error, DatabaseType.POSTGRESQL
        )
        
        assert query_id is not None
    
    @pytest.mark.asyncio
    async def test_audit_report_generation(self, audit_logger):
        """Test audit report generation"""
        
        # Add some test audit entries
        for i in range(3):
            query = f"SELECT * FROM table_{i}"
            result = QueryResult(
                data=[],
                metadata={},
                execution_time=i * 0.1,
                row_count=i * 10,
                database_type=DatabaseType.POSTGRESQL
            )
            
            from backend.core.cross_database_query_optimizer import QueryMetrics
            
            metrics = QueryMetrics(
                execution_time=i * 0.1,
                rows_processed=i * 10,
                memory_usage=5.0,
                cpu_usage=2.0,
                cache_hit=False,
                optimization_applied=[],
                database_type=DatabaseType.POSTGRESQL,
                query_hash=f"hash_{i}"
            )
            
            await audit_logger.log_query_execution(query, result, metrics)
        
        report = await audit_logger.get_audit_report()
        
        assert 'summary' in report
        assert 'database_statistics' in report
        assert report['summary']['total_queries'] == 3

class TestSpatialPartitioner:
    """Test spatial data partitioning"""
    
    @pytest.fixture
    def spatial_partitioner(self):
        return SpatialPartitioner()
    
    def test_spatial_grid_creation(self, spatial_partitioner):
        """Test creation of spatial grid partitions"""
        
        bounds = (-120.0, 35.0, -115.0, 40.0)  # California region
        num_partitions = 4
        
        partitions = spatial_partitioner.create_spatial_partitions(bounds, num_partitions)
        
        assert len(partitions) <= num_partitions
        
        # Check that partitions cover the bounds
        for partition in partitions:
            minx, miny, maxx, maxy = partition.bounds
            assert minx >= bounds[0]
            assert miny >= bounds[1]
            assert maxx <= bounds[2]
            assert maxy <= bounds[3]

class TestTemporalPartitioner:
    """Test temporal data partitioning"""
    
    @pytest.fixture
    def temporal_partitioner(self):
        return TemporalPartitioner()
    
    def test_temporal_partition_creation(self, temporal_partitioner):
        """Test creation of temporal partitions"""
        
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 2)
        partition_size = timedelta(hours=6)
        
        partitions = temporal_partitioner.create_temporal_partitions(
            start_time, end_time, partition_size
        )
        
        assert len(partitions) == 4  # 24 hours / 6 hours = 4 partitions
        
        # Check partition continuity
        for i in range(len(partitions) - 1):
            assert partitions[i][1] == partitions[i + 1][0]
    
    def test_adaptive_temporal_partitioning(self, temporal_partitioner):
        """Test adaptive temporal partitioning based on data density"""
        
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 2)
        
        # Mock data density (more data in certain hours)
        data_density = {}
        current_time = start_time
        while current_time < end_time:
            # Simulate higher density during certain hours
            if current_time.hour in [8, 12, 16, 20]:
                data_density[current_time] = 1000
            else:
                data_density[current_time] = 100
            current_time += timedelta(hours=1)
        
        partitions = temporal_partitioner.create_adaptive_temporal_partitions(
            start_time, end_time, data_density, target_partition_size=2000
        )
        
        assert len(partitions) > 0
        
        # Check that partitions respect the target size
        for partition_start, partition_end in partitions:
            partition_data = sum(
                count for time, count in data_density.items()
                if partition_start <= time < partition_end
            )
            # Should be close to target size (allowing some variance)
            assert partition_data <= 3000  # Some tolerance

class TestQueryPartitioner:
    """Test query partitioning logic"""
    
    @pytest.fixture
    def query_partitioner(self):
        return QueryPartitioner()
    
    @pytest.mark.asyncio
    async def test_spatial_query_partitioning(self, query_partitioner):
        """Test partitioning of spatial queries"""
        
        spatial_query = """
        SELECT * FROM geological_observations 
        WHERE ST_Within(location, ST_MakeEnvelope(-120, 35, -115, 40, 4326))
        """
        
        execution_plan = await query_partitioner.create_execution_plan(
            spatial_query, QueryType.SPATIAL, 100000, {'cpu_cores': 4, 'memory_gb': 8}
        )
        
        assert execution_plan.execution_mode in [ExecutionMode.PARALLEL, ExecutionMode.DISTRIBUTED]
        assert len(execution_plan.partitions) > 1
        assert all(p.partition_strategy == PartitionStrategy.SPATIAL for p in execution_plan.partitions)
    
    @pytest.mark.asyncio
    async def test_temporal_query_partitioning(self, query_partitioner):
        """Test partitioning of temporal queries"""
        
        temporal_query = """
        SELECT * FROM geological_observations 
        WHERE timestamp >= '2024-01-01' AND timestamp < '2024-01-02'
        """
        
        execution_plan = await query_partitioner.create_execution_plan(
            temporal_query, QueryType.TEMPORAL, 500000, {'cpu_cores': 8, 'memory_gb': 16}
        )
        
        assert len(execution_plan.partitions) > 1
        assert all(p.partition_strategy == PartitionStrategy.TEMPORAL for p in execution_plan.partitions)

class TestDistributedQueryExecutor:
    """Test distributed query execution"""
    
    @pytest.fixture
    def mock_db_manager(self):
        return Mock(spec=UnifiedDatabaseManager)
    
    @pytest.fixture
    def mock_query_optimizer(self):
        optimizer = Mock(spec=CrossDatabaseQueryOptimizer)
        
        # Mock successful query execution
        optimizer.execute_optimized_query = AsyncMock(return_value=QueryResult(
            data=[{'id': i, 'value': f'test_{i}'} for i in range(100)],
            metadata={},
            execution_time=0.5,
            row_count=100,
            database_type=DatabaseType.POSTGRESQL
        ))
        
        return optimizer
    
    @pytest.fixture
    def distributed_executor(self, mock_db_manager, mock_query_optimizer):
        return DistributedQueryExecutor(mock_db_manager, mock_query_optimizer)
    
    @pytest.mark.asyncio
    async def test_sequential_execution(self, distributed_executor):
        """Test sequential query execution"""
        
        query = "SELECT * FROM small_table"
        
        result = await distributed_executor.execute_distributed_query(
            query, QueryType.SELECT, estimated_data_size=1000
        )
        
        assert result is not None
        assert result.metadata['execution_mode'] == ExecutionMode.SEQUENTIAL.value
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, distributed_executor):
        """Test parallel query execution"""
        
        query = "SELECT * FROM medium_table WHERE condition = 'test'"
        
        result = await distributed_executor.execute_distributed_query(
            query, QueryType.SELECT, estimated_data_size=100000
        )
        
        assert result is not None
        assert result.metadata['execution_mode'] in [ExecutionMode.PARALLEL.value, ExecutionMode.DISTRIBUTED.value]
    
    def test_execution_statistics(self, distributed_executor):
        """Test execution statistics tracking"""
        
        stats = distributed_executor.get_execution_statistics()
        
        assert 'total_executions' in stats
        assert 'success_rate' in stats
        assert 'execution_modes' in stats

class TestIntegratedQuerySystem:
    """Test integrated query system functionality"""
    
    @pytest.fixture
    def mock_components(self):
        """Mock all system components"""
        
        # Mock database manager
        db_manager = Mock(spec=UnifiedDatabaseManager)
        db_manager.execute_query = AsyncMock(return_value=QueryResult(
            data=[{'id': 1, 'test': 'data'}],
            metadata={},
            execution_time=0.5,
            row_count=1,
            database_type=DatabaseType.POSTGRESQL
        ))
        
        # Mock query optimizer
        query_optimizer = Mock(spec=CrossDatabaseQueryOptimizer)
        query_optimizer.execute_optimized_query = AsyncMock(return_value=QueryResult(
            data=[{'id': 1, 'test': 'optimized'}],
            metadata={'optimizations': ['index_hint']},
            execution_time=0.3,
            row_count=1,
            cached=False,
            database_type=DatabaseType.POSTGRESQL
        ))
        query_optimizer.query_cache = Mock()
        query_optimizer.query_cache.get_statistics = Mock(return_value={
            'hit_rate': 0.75,
            'current_size_mb': 50.0
        })
        query_optimizer.get_optimization_report = AsyncMock(return_value={
            'cache_statistics': {'hit_rate': 0.75}
        })
        query_optimizer.cleanup = AsyncMock()
        
        # Mock audit logger
        audit_logger = Mock(spec=QueryAuditLogger)
        audit_logger.log_query_execution = AsyncMock(return_value='audit_123')
        audit_logger.log_query_error = AsyncMock(return_value='error_123')
        audit_logger.get_audit_report = AsyncMock(return_value={
            'summary': {'total_queries': 10}
        })
        audit_logger.start_background_tasks = AsyncMock()
        audit_logger.cleanup = AsyncMock()
        
        # Mock distributed executor
        distributed_executor = Mock(spec=DistributedQueryExecutor)
        distributed_executor.execute_distributed_query = AsyncMock(return_value=QueryResult(
            data=[{'id': 1, 'test': 'distributed'}],
            metadata={'execution_mode': 'distributed'},
            execution_time=1.0,
            row_count=1,
            database_type=DatabaseType.POSTGRESQL
        ))
        distributed_executor.get_execution_statistics = Mock(return_value={
            'total_executions': 5
        })
        distributed_executor.cleanup = AsyncMock()
        
        # Mock config manager
        config_manager = Mock()
        
        return {
            'db_manager': db_manager,
            'query_optimizer': query_optimizer,
            'audit_logger': audit_logger,
            'distributed_executor': distributed_executor,
            'config_manager': config_manager
        }
    
    @pytest.fixture
    def integrated_system(self, mock_components):
        system = IntegratedQuerySystem()
        
        # Inject mocked components
        system.db_manager = mock_components['db_manager']
        system.query_optimizer = mock_components['query_optimizer']
        system.audit_logger = mock_components['audit_logger']
        system.distributed_executor = mock_components['distributed_executor']
        system.config_manager = mock_components['config_manager']
        
        return system
    
    @pytest.mark.asyncio
    async def test_simple_query_execution(self, integrated_system):
        """Test execution of simple queries"""
        
        query_request = QueryRequest(
            query="SELECT * FROM geological_observations LIMIT 10",
            use_cache=True,
            enable_optimization=True
        )
        
        response = await integrated_system.execute_query(query_request)
        
        assert response is not None
        assert response.query_result is not None
        assert response.execution_strategy_used in [ExecutionStrategy.DIRECT, ExecutionStrategy.OPTIMIZED]
        assert response.audit_log_id is not None
    
    @pytest.mark.asyncio
    async def test_complex_query_execution(self, integrated_system):
        """Test execution of complex queries"""
        
        complex_query = """
        SELECT v.name, COUNT(e.id) as event_count, AVG(e.magnitude) as avg_magnitude
        FROM volcanoes v
        JOIN seismic_events e ON v.id = e.volcano_id
        WHERE v.location && ST_MakeEnvelope(-120, 35, -115, 40, 4326)
        AND e.timestamp >= '2024-01-01'
        GROUP BY v.id, v.name
        HAVING COUNT(e.id) > 10
        ORDER BY avg_magnitude DESC
        """
        
        query_request = QueryRequest(
            query=complex_query,
            enable_optimization=True,
            enable_distributed=True
        )
        
        response = await integrated_system.execute_query(query_request)
        
        assert response is not None
        assert response.execution_strategy_used in [
            ExecutionStrategy.OPTIMIZED, 
            ExecutionStrategy.DISTRIBUTED, 
            ExecutionStrategy.HYBRID
        ]
    
    @pytest.mark.asyncio
    async def test_system_status_reporting(self, integrated_system):
        """Test system status reporting"""
        
        status = await integrated_system.get_system_status()
        
        assert 'system_status' in status
        assert 'components' in status
        assert 'performance_statistics' in status
        assert 'system_metrics' in status
    
    @pytest.mark.asyncio
    async def test_performance_report_generation(self, integrated_system):
        """Test performance report generation"""
        
        report = await integrated_system.get_performance_report()
        
        assert 'audit_report' in report
        assert 'optimization_report' in report
        assert 'distributed_execution_statistics' in report
        assert 'system_status' in report
    
    @pytest.mark.asyncio
    async def test_system_optimization_analysis(self, integrated_system):
        """Test system optimization analysis"""
        
        # Set up some metrics that would trigger recommendations
        integrated_system.system_metrics = {
            'cache_hit_rate': 0.3,  # Low cache hit rate
            'avg_execution_time': 3.0,  # High execution time
            'system_load': 0.9,  # High system load
            'success_rate': 0.92  # Slightly low success rate
        }
        
        optimization_analysis = await integrated_system.optimize_system_configuration()
        
        assert 'recommendations' in optimization_analysis
        assert 'system_health_score' in optimization_analysis
        assert len(optimization_analysis['recommendations']) > 0
        
        # Check that recommendations are generated for the issues we set up
        recommendation_types = [r['type'] for r in optimization_analysis['recommendations']]
        assert 'cache_optimization' in recommendation_types
        assert 'resource_optimization' in recommendation_types

class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_geological_query_workflow(self):
        """Test complete geological query workflow"""
        
        # This would test the entire pipeline from query submission to result delivery
        # For now, we'll test the convenience function
        
        with patch('backend.core.integrated_query_system.get_integrated_query_system') as mock_get_system:
            
            # Mock the integrated system
            mock_system = Mock()
            mock_system.execute_query = AsyncMock(return_value=QueryResponse(
                request_id='test_123',
                query_result=QueryResult(
                    data=[{'volcano_name': 'Mount St. Helens', 'magnitude': 5.2}],
                    metadata={},
                    execution_time=0.8,
                    row_count=1,
                    database_type=DatabaseType.POSTGRESQL
                ),
                execution_strategy_used=ExecutionStrategy.OPTIMIZED,
                optimization_applied=['spatial_index'],
                cache_hit=False,
                execution_time=0.8,
                audit_log_id='audit_456',
                performance_metrics={'complexity': 'moderate'},
                recommendations=['Consider caching for frequent queries']
            ))
            
            mock_get_system.return_value = mock_system
            
            # Import and test the convenience function
            from backend.core.integrated_query_system import execute_geological_query
            
            response = await execute_geological_query(
                "SELECT * FROM volcanoes WHERE region = 'Cascades'",
                query_type=QueryType.SELECT,
                use_cache=True
            )
            
            assert response is not None
            assert response.query_result.row_count == 1
            assert len(response.recommendations) > 0

# Performance and load testing
class TestPerformanceAndLoad:
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    async def test_concurrent_query_execution(self):
        """Test concurrent query execution performance"""
        
        # Mock components for performance testing
        with patch('backend.core.integrated_query_system.get_integrated_query_system') as mock_get_system:
            
            mock_system = Mock()
            mock_system.execute_query = AsyncMock(return_value=QueryResponse(
                request_id='perf_test',
                query_result=QueryResult(
                    data=[{'id': 1}],
                    metadata={},
                    execution_time=0.1,
                    row_count=1,
                    database_type=DatabaseType.POSTGRESQL
                ),
                execution_strategy_used=ExecutionStrategy.OPTIMIZED,
                optimization_applied=[],
                cache_hit=False,
                execution_time=0.1,
                audit_log_id='audit_perf',
                performance_metrics={},
                recommendations=[]
            ))
            
            mock_get_system.return_value = mock_system
            
            from backend.core.integrated_query_system import execute_geological_query
            
            # Execute multiple queries concurrently
            tasks = []
            for i in range(10):
                task = execute_geological_query(f"SELECT * FROM table_{i}")
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            execution_time = time.time() - start_time
            
            assert len(results) == 10
            assert all(r is not None for r in results)
            assert execution_time < 5.0  # Should complete within 5 seconds
    
    def test_cache_performance_under_load(self):
        """Test cache performance under high load"""
        
        cache = QueryCache(max_size_mb=1)  # Small cache to test eviction
        
        # Fill cache beyond capacity
        for i in range(100):
            query = f"SELECT * FROM table_{i}"
            result = QueryResult(
                data=[{'id': i}],
                metadata={},
                execution_time=0.1,
                row_count=1,
                database_type=DatabaseType.POSTGRESQL
            )
            
            # This would be async in real usage, but for testing we'll simulate
            cache.cache[cache._generate_cache_key(query)] = cache.CacheEntry(
                result=result.data,
                metadata=result.metadata,
                created_at=datetime.utcnow(),
                size_bytes=100
            )
        
        # Cache should have evicted entries to stay within size limit
        stats = cache.get_statistics()
        assert stats['current_size_mb'] <= 1.0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])