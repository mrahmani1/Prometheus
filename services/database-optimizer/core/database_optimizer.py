"""
Advanced Database Optimization System for Prometheus AI Platform

This module implements comprehensive database optimization with geological domain awareness,
including sharding, partitioning, connection pooling, query optimization, and performance tuning.

Features:
- Intelligent database sharding and partitioning strategies
- Advanced connection pooling and management
- Query optimization with geological context
- Automated index management and recommendations
- Performance monitoring and tuning
- Database backup and recovery optimization
- Data lifecycle management and archival
- Multi-database coordination and optimization
"""

import asyncio
import hashlib
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import asyncpg
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy import text, MetaData, Table, Column, Index
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import QueuePool


class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    INFLUXDB = "influxdb"
    NEO4J = "neo4j"
    REDIS = "redis"
    SQLITE = "sqlite"


class OptimizationType(Enum):
    """Types of database optimizations"""
    SHARDING = "sharding"
    PARTITIONING = "partitioning"
    INDEXING = "indexing"
    QUERY_OPTIMIZATION = "query_optimization"
    CONNECTION_POOLING = "connection_pooling"
    CACHING = "caching"
    ARCHIVAL = "archival"
    REPLICATION = "replication"


class GeologicalDataPattern(Enum):
    """Geological data access patterns"""
    TEMPORAL_SERIES = "temporal_series"
    SPATIAL_RANGE = "spatial_range"
    VOLCANIC_MONITORING = "volcanic_monitoring"
    SEISMIC_ANALYSIS = "seismic_analysis"
    GEOLOGICAL_MODELING = "geological_modeling"
    RESOURCE_ASSESSMENT = "resource_assessment"


@dataclass
class DatabaseMetrics:
    """Database performance metrics"""
    connection_count: int = 0
    active_queries: int = 0
    avg_query_time: float = 0.0
    cache_hit_rate: float = 0.0
    index_usage: Dict[str, float] = field(default_factory=dict)
    table_sizes: Dict[str, int] = field(default_factory=dict)
    fragmentation_ratio: float = 0.0
    replication_lag: float = 0.0
    backup_status: str = "unknown"
    storage_usage: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0


@dataclass
class OptimizationRecommendation:
    """Database optimization recommendation"""
    recommendation_id: str
    optimization_type: OptimizationType
    priority: int  # 1-10, 10 being highest
    description: str
    expected_improvement: float
    implementation_effort: str  # "low", "medium", "high"
    geological_impact: str
    cost_impact: float
    affected_tables: List[str]
    sql_commands: List[str] = field(default_factory=list)
    rollback_commands: List[str] = field(default_factory=list)


class GeologicalShardingStrategy:
    """Geological domain-aware sharding strategies"""
    
    def __init__(self):
        self.sharding_strategies = {
            'volcanic_monitoring': self._volcanic_sharding_strategy,
            'seismic_analysis': self._seismic_sharding_strategy,
            'geological_modeling': self._geological_sharding_strategy,
            'resource_assessment': self._resource_sharding_strategy
        }
        
        self.partition_strategies = {
            'temporal': self._temporal_partitioning,
            'spatial': self._spatial_partitioning,
            'volcanic_region': self._volcanic_region_partitioning,
            'magnitude_based': self._magnitude_based_partitioning
        }
    
    def get_sharding_key(self, table_name: str, data: Dict[str, Any], 
                        geological_context: Dict[str, Any]) -> str:
        """Generate sharding key based on geological context"""
        domain = geological_context.get('domain', 'general')
        
        if domain in self.sharding_strategies:
            return self.sharding_strategies[domain](table_name, data, geological_context)
        else:
            return self._default_sharding_strategy(table_name, data, geological_context)
    
    def _volcanic_sharding_strategy(self, table_name: str, data: Dict[str, Any],
                                  geological_context: Dict[str, Any]) -> str:
        """Sharding strategy for volcanic monitoring data"""
        volcano_id = data.get('volcano_id') or geological_context.get('volcano_id')
        
        if volcano_id:
            # Shard by volcano ID with alert level consideration
            alert_level = data.get('alert_level', 0)
            if alert_level >= 3:  # High alert volcanoes get dedicated shards
                return f"volcanic_high_alert_{volcano_id}"
            else:
                # Hash volcano ID to distribute across shards
                shard_num = int(hashlib.md5(volcano_id.encode()).hexdigest(), 16) % 8
                return f"volcanic_shard_{shard_num}"
        else:
            # Fallback to temporal sharding
            timestamp = data.get('timestamp', datetime.now())
            return f"volcanic_temporal_{timestamp.strftime('%Y_%m')}"
    
    def _seismic_sharding_strategy(self, table_name: str, data: Dict[str, Any],
                                 geological_context: Dict[str, Any]) -> str:
        """Sharding strategy for seismic analysis data"""
        magnitude = data.get('magnitude', 0.0)
        
        if magnitude >= 7.0:  # Major earthquakes
            return "seismic_major"
        elif magnitude >= 5.0:  # Moderate earthquakes
            return "seismic_moderate"
        elif magnitude >= 3.0:  # Minor earthquakes
            return "seismic_minor"
        else:  # Micro earthquakes
            return "seismic_micro"
    
    def _geological_sharding_strategy(self, table_name: str, data: Dict[str, Any],
                                    geological_context: Dict[str, Any]) -> str:
        """Sharding strategy for geological modeling data"""
        # Shard by geological age or formation type
        geological_age = data.get('geological_age')
        formation_type = data.get('formation_type')
        
        if geological_age:
            if geological_age < 2.6e6:  # Quaternary
                return "geological_quaternary"
            elif geological_age < 66e6:  # Cenozoic
                return "geological_cenozoic"
            elif geological_age < 252e6:  # Mesozoic
                return "geological_mesozoic"
            else:  # Paleozoic and older
                return "geological_paleozoic"
        elif formation_type:
            formation_hash = int(hashlib.md5(formation_type.encode()).hexdigest(), 16) % 4
            return f"geological_formation_{formation_hash}"
        else:
            # Fallback to spatial sharding
            lat = data.get('latitude', 0.0)
            lon = data.get('longitude', 0.0)
            spatial_hash = int(hashlib.md5(f"{lat:.1f}_{lon:.1f}".encode()).hexdigest(), 16) % 8
            return f"geological_spatial_{spatial_hash}"
    
    def _resource_sharding_strategy(self, table_name: str, data: Dict[str, Any],
                                  geological_context: Dict[str, Any]) -> str:
        """Sharding strategy for resource assessment data"""
        resource_type = data.get('resource_type', 'unknown')
        
        # Shard by resource type
        if resource_type in ['gold', 'silver', 'platinum']:
            return "resource_precious_metals"
        elif resource_type in ['copper', 'iron', 'aluminum']:
            return "resource_base_metals"
        elif resource_type in ['coal', 'oil', 'gas']:
            return "resource_energy"
        elif resource_type in ['uranium', 'lithium', 'rare_earth']:
            return "resource_strategic"
        else:
            return "resource_other"
    
    def _default_sharding_strategy(self, table_name: str, data: Dict[str, Any],
                                 geological_context: Dict[str, Any]) -> str:
        """Default sharding strategy"""
        # Use timestamp-based sharding as fallback
        timestamp = data.get('timestamp', datetime.now())
        return f"default_{timestamp.strftime('%Y_%m')}"
    
    def _temporal_partitioning(self, table_name: str, partition_column: str) -> List[str]:
        """Generate temporal partitioning SQL"""
        current_year = datetime.now().year
        partitions = []
        
        # Create partitions for current year and next year
        for year in [current_year, current_year + 1]:
            for month in range(1, 13):
                partition_name = f"{table_name}_{year}_{month:02d}"
                start_date = f"{year}-{month:02d}-01"
                
                if month == 12:
                    end_date = f"{year + 1}-01-01"
                else:
                    end_date = f"{year}-{month + 1:02d}-01"
                
                partition_sql = f"""
                CREATE TABLE IF NOT EXISTS {partition_name} PARTITION OF {table_name}
                FOR VALUES FROM ('{start_date}') TO ('{end_date}');
                """
                partitions.append(partition_sql.strip())
        
        return partitions
    
    def _spatial_partitioning(self, table_name: str, partition_column: str) -> List[str]:
        """Generate spatial partitioning SQL"""
        # Create spatial partitions based on geographic regions
        regions = [
            ("north_america", -180, 25, -50, 85),
            ("south_america", -90, -60, -30, 25),
            ("europe_africa", -30, -40, 60, 85),
            ("asia", 60, -10, 180, 85),
            ("oceania", 110, -50, 180, -10),
            ("antarctica", -180, -90, 180, -60)
        ]
        
        partitions = []
        for region_name, lon_min, lat_min, lon_max, lat_max in regions:
            partition_name = f"{table_name}_{region_name}"
            partition_sql = f"""
            CREATE TABLE IF NOT EXISTS {partition_name} PARTITION OF {table_name}
            FOR VALUES FROM ({lon_min}, {lat_min}) TO ({lon_max}, {lat_max});
            """
            partitions.append(partition_sql.strip())
        
        return partitions
    
    def _volcanic_region_partitioning(self, table_name: str, partition_column: str) -> List[str]:
        """Generate volcanic region-based partitioning SQL"""
        volcanic_regions = [
            "pacific_ring_of_fire",
            "mid_atlantic_ridge",
            "mediterranean_volcanic_arc",
            "east_african_rift",
            "yellowstone_hotspot",
            "hawaii_hotspot",
            "iceland_hotspot"
        ]
        
        partitions = []
        for region in volcanic_regions:
            partition_name = f"{table_name}_{region}"
            partition_sql = f"""
            CREATE TABLE IF NOT EXISTS {partition_name} PARTITION OF {table_name}
            FOR VALUES IN ('{region}');
            """
            partitions.append(partition_sql.strip())
        
        return partitions
    
    def _magnitude_based_partitioning(self, table_name: str, partition_column: str) -> List[str]:
        """Generate magnitude-based partitioning SQL for seismic data"""
        magnitude_ranges = [
            ("micro", 0.0, 3.0),
            ("minor", 3.0, 5.0),
            ("moderate", 5.0, 7.0),
            ("major", 7.0, 8.0),
            ("great", 8.0, 10.0)
        ]
        
        partitions = []
        for range_name, min_mag, max_mag in magnitude_ranges:
            partition_name = f"{table_name}_{range_name}"
            partition_sql = f"""
            CREATE TABLE IF NOT EXISTS {partition_name} PARTITION OF {table_name}
            FOR VALUES FROM ({min_mag}) TO ({max_mag});
            """
            partitions.append(partition_sql.strip())
        
        return partitions


class AdvancedConnectionPoolManager:
    """Advanced connection pool management with geological workload awareness"""
    
    def __init__(self, database_configs: Dict[str, Dict[str, Any]]):
        self.database_configs = database_configs
        self.connection_pools: Dict[str, Any] = {}
        self.pool_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.geological_workload_patterns = {
            'volcanic_monitoring': {
                'read_heavy': True,
                'write_burst': True,
                'connection_multiplier': 1.5,
                'timeout_multiplier': 0.5
            },
            'seismic_analysis': {
                'read_heavy': True,
                'write_burst': False,
                'connection_multiplier': 2.0,
                'timeout_multiplier': 1.0
            },
            'geological_modeling': {
                'read_heavy': False,
                'write_burst': False,
                'connection_multiplier': 1.0,
                'timeout_multiplier': 2.0
            },
            'resource_assessment': {
                'read_heavy': True,
                'write_burst': False,
                'connection_multiplier': 1.2,
                'timeout_multiplier': 1.5
            }
        }
        self.logger = logging.getLogger(__name__)
    
    async def initialize_pools(self):
        """Initialize connection pools for all databases"""
        for db_name, config in self.database_configs.items():
            try:
                await self._create_connection_pool(db_name, config)
                self.logger.info(f"Initialized connection pool for {db_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize pool for {db_name}: {e}")
    
    async def _create_connection_pool(self, db_name: str, config: Dict[str, Any]):
        """Create optimized connection pool for specific database"""
        db_type = config.get('type', 'postgresql')
        
        if db_type == 'postgresql':
            await self._create_postgresql_pool(db_name, config)
        elif db_type == 'influxdb':
            await self._create_influxdb_pool(db_name, config)
        # Add other database types as needed
    
    async def _create_postgresql_pool(self, db_name: str, config: Dict[str, Any]):
        """Create optimized PostgreSQL connection pool"""
        base_pool_size = config.get('pool_size', 20)
        base_max_overflow = config.get('max_overflow', 30)
        
        # Adjust pool size based on geological workload
        geological_domain = config.get('geological_domain', 'general')
        workload_pattern = self.geological_workload_patterns.get(geological_domain, {})
        
        connection_multiplier = workload_pattern.get('connection_multiplier', 1.0)
        pool_size = int(base_pool_size * connection_multiplier)
        max_overflow = int(base_max_overflow * connection_multiplier)
        
        # Create SQLAlchemy engine with optimized settings
        engine = create_async_engine(
            config['url'],
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=config.get('echo', False),
            connect_args={
                "server_settings": {
                    "application_name": f"prometheus_ai_{db_name}",
                    "jit": "off",  # Disable JIT for consistent performance
                }
            }
        )
        
        self.connection_pools[db_name] = engine
        
        # Initialize pool metrics
        self.pool_metrics[db_name] = {
            'pool_size': pool_size,
            'max_overflow': max_overflow,
            'active_connections': 0,
            'total_requests': 0,
            'avg_response_time': 0.0,
            'error_count': 0
        }
    
    async def _create_influxdb_pool(self, db_name: str, config: Dict[str, Any]):
        """Create optimized InfluxDB connection pool"""
        # InfluxDB connection pool implementation
        # This would use the InfluxDB client library
        pass
    
    async def get_connection(self, db_name: str, geological_context: Dict[str, Any] = None):
        """Get database connection with geological context optimization"""
        if db_name not in self.connection_pools:
            raise ValueError(f"No connection pool for database: {db_name}")
        
        start_time = time.time()
        
        try:
            engine = self.connection_pools[db_name]
            connection = await engine.connect()
            
            # Update metrics
            self.pool_metrics[db_name]['active_connections'] += 1
            self.pool_metrics[db_name]['total_requests'] += 1
            
            # Apply geological context optimizations
            if geological_context:
                await self._apply_geological_optimizations(connection, geological_context)
            
            return connection
            
        except Exception as e:
            self.pool_metrics[db_name]['error_count'] += 1
            self.logger.error(f"Failed to get connection for {db_name}: {e}")
            raise
        finally:
            response_time = time.time() - start_time
            self._update_response_time_metric(db_name, response_time)
    
    async def _apply_geological_optimizations(self, connection, geological_context: Dict[str, Any]):
        """Apply geological context-specific optimizations to connection"""
        domain = geological_context.get('domain')
        
        if domain == 'volcanic_monitoring':
            # Optimize for real-time volcanic monitoring
            await connection.execute(text("SET statement_timeout = '30s'"))
            await connection.execute(text("SET work_mem = '256MB'"))
        elif domain == 'seismic_analysis':
            # Optimize for seismic data analysis
            await connection.execute(text("SET statement_timeout = '300s'"))
            await connection.execute(text("SET work_mem = '512MB'"))
            await connection.execute(text("SET enable_hashjoin = on"))
        elif domain == 'geological_modeling':
            # Optimize for complex geological modeling
            await connection.execute(text("SET statement_timeout = '1800s'"))
            await connection.execute(text("SET work_mem = '1GB'"))
            await connection.execute(text("SET enable_mergejoin = on"))
        elif domain == 'resource_assessment':
            # Optimize for resource assessment queries
            await connection.execute(text("SET statement_timeout = '600s'"))
            await connection.execute(text("SET work_mem = '512MB'"))
            await connection.execute(text("SET random_page_cost = 1.1"))
    
    def _update_response_time_metric(self, db_name: str, response_time: float):
        """Update average response time metric"""
        current_avg = self.pool_metrics[db_name]['avg_response_time']
        total_requests = self.pool_metrics[db_name]['total_requests']
        
        # Exponential moving average
        if total_requests == 1:
            self.pool_metrics[db_name]['avg_response_time'] = response_time
        else:
            alpha = 0.1  # Smoothing factor
            self.pool_metrics[db_name]['avg_response_time'] = (
                alpha * response_time + (1 - alpha) * current_avg
            )
    
    async def release_connection(self, db_name: str, connection):
        """Release database connection back to pool"""
        try:
            await connection.close()
            self.pool_metrics[db_name]['active_connections'] -= 1
        except Exception as e:
            self.logger.error(f"Error releasing connection for {db_name}: {e}")
    
    def get_pool_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get connection pool metrics"""
        return dict(self.pool_metrics)


class DatabaseOptimizer:
    """Advanced database optimizer with geological domain awareness"""
    
    def __init__(self, database_configs: Dict[str, Dict[str, Any]]):
        self.database_configs = database_configs
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.sharding_strategy = GeologicalShardingStrategy()
        self.connection_manager = AdvancedConnectionPoolManager(database_configs)
        
        # Optimization tracking
        self.optimization_history: List[OptimizationRecommendation] = []
        self.performance_metrics: Dict[str, DatabaseMetrics] = {}
        self.query_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Index management
        self.recommended_indexes: Dict[str, List[str]] = defaultdict(list)
        self.existing_indexes: Dict[str, Set[str]] = defaultdict(set)
        
        # Geological optimization patterns
        self.geological_table_patterns = {
            'seismic_events': {
                'partition_strategy': 'temporal',
                'partition_column': 'timestamp',
                'recommended_indexes': [
                    'CREATE INDEX CONCURRENTLY idx_seismic_magnitude ON seismic_events (magnitude DESC)',
                    'CREATE INDEX CONCURRENTLY idx_seismic_location ON seismic_events USING GIST (ST_Point(longitude, latitude))',
                    'CREATE INDEX CONCURRENTLY idx_seismic_time_mag ON seismic_events (timestamp, magnitude)',
                    'CREATE INDEX CONCURRENTLY idx_seismic_depth ON seismic_events (depth) WHERE depth IS NOT NULL'
                ]
            },
            'volcanic_data': {
                'partition_strategy': 'volcanic_region',
                'partition_column': 'volcano_id',
                'recommended_indexes': [
                    'CREATE INDEX CONCURRENTLY idx_volcanic_alert ON volcanic_data (volcano_id, alert_level, timestamp)',
                    'CREATE INDEX CONCURRENTLY idx_volcanic_location ON volcanic_data USING GIST (location)',
                    'CREATE INDEX CONCURRENTLY idx_volcanic_time ON volcanic_data (timestamp DESC)',
                    'CREATE INDEX CONCURRENTLY idx_volcanic_type ON volcanic_data (eruption_type) WHERE eruption_type IS NOT NULL'
                ]
            },
            'geological_units': {
                'partition_strategy': 'spatial',
                'partition_column': 'geometry',
                'recommended_indexes': [
                    'CREATE INDEX CONCURRENTLY idx_geological_spatial ON geological_units USING GIST (geometry)',
                    'CREATE INDEX CONCURRENTLY idx_geological_age ON geological_units (age_min, age_max)',
                    'CREATE INDEX CONCURRENTLY idx_geological_lithology ON geological_units (lithology)',
                    'CREATE INDEX CONCURRENTLY idx_geological_formation ON geological_units (formation_name)'
                ]
            },
            'resource_data': {
                'partition_strategy': 'temporal',
                'partition_column': 'discovery_date',
                'recommended_indexes': [
                    'CREATE INDEX CONCURRENTLY idx_resource_type ON resource_data (resource_type, grade DESC)',
                    'CREATE INDEX CONCURRENTLY idx_resource_location ON resource_data USING GIST (location)',
                    'CREATE INDEX CONCURRENTLY idx_resource_economic ON resource_data (economic_viability) WHERE economic_viability IS NOT NULL',
                    'CREATE INDEX CONCURRENTLY idx_resource_deposit ON resource_data (deposit_type, tonnage DESC)'
                ]
            }
        }
    
    async def initialize(self):
        """Initialize database optimizer"""
        try:
            await self.connection_manager.initialize_pools()
            await self._analyze_existing_schema()
            await self._collect_initial_metrics()
            self.logger.info("Database optimizer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database optimizer: {e}")
            raise
    
    async def _analyze_existing_schema(self):
        """Analyze existing database schema"""
        for db_name, config in self.database_configs.items():
            try:
                connection = await self.connection_manager.get_connection(db_name)
                
                # Get existing indexes
                indexes_query = """
                SELECT schemaname, tablename, indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
                """
                
                result = await connection.execute(text(indexes_query))
                for row in result:
                    table_name = row[1]
                    index_name = row[2]
                    self.existing_indexes[table_name].add(index_name)
                
                await self.connection_manager.release_connection(db_name, connection)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze schema for {db_name}: {e}")
    
    async def _collect_initial_metrics(self):
        """Collect initial performance metrics"""
        for db_name, config in self.database_configs.items():
            try:
                metrics = await self._collect_database_metrics(db_name)
                self.performance_metrics[db_name] = metrics
            except Exception as e:
                self.logger.error(f"Failed to collect metrics for {db_name}: {e}")
    
    async def _collect_database_metrics(self, db_name: str) -> DatabaseMetrics:
        """Collect comprehensive database metrics"""
        connection = await self.connection_manager.get_connection(db_name)
        
        try:
            metrics = DatabaseMetrics()
            
            # Connection metrics
            conn_query = "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
            result = await connection.execute(text(conn_query))
            metrics.connection_count = result.scalar()
            
            # Query performance metrics
            query_stats_query = """
            SELECT mean_exec_time, calls
            FROM pg_stat_statements
            ORDER BY mean_exec_time DESC
            LIMIT 100
            """
            
            try:
                result = await connection.execute(text(query_stats_query))
                query_times = [row[0] for row in result]
                if query_times:
                    metrics.avg_query_time = statistics.mean(query_times)
            except:
                # pg_stat_statements might not be enabled
                metrics.avg_query_time = 0.0
            
            # Table size metrics
            table_size_query = """
            SELECT schemaname, tablename, pg_total_relation_size(schemaname||'.'||tablename) as size
            FROM pg_tables
            WHERE schemaname = 'public'
            """
            
            result = await connection.execute(text(table_size_query))
            for row in result:
                table_name = row[1]
                size_bytes = row[2]
                metrics.table_sizes[table_name] = size_bytes
            
            # Index usage metrics
            index_usage_query = """
            SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
            FROM pg_stat_user_indexes
            ORDER BY idx_scan DESC
            """
            
            result = await connection.execute(text(index_usage_query))
            for row in result:
                index_name = row[2]
                idx_scan = row[3] or 0
                metrics.index_usage[index_name] = idx_scan
            
            # Cache hit rate
            cache_hit_query = """
            SELECT 
                sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as cache_hit_rate
            FROM pg_statio_user_tables
            """
            
            result = await connection.execute(text(cache_hit_query))
            cache_hit_rate = result.scalar()
            metrics.cache_hit_rate = cache_hit_rate if cache_hit_rate else 0.0
            
            return metrics
            
        finally:
            await self.connection_manager.release_connection(db_name, connection)
    
    async def optimize_database(self, db_name: str, 
                              geological_context: Dict[str, Any] = None) -> List[OptimizationRecommendation]:
        """Perform comprehensive database optimization"""
        recommendations = []
        
        try:
            # Collect current metrics
            current_metrics = await self._collect_database_metrics(db_name)
            self.performance_metrics[db_name] = current_metrics
            
            # Generate optimization recommendations
            recommendations.extend(await self._analyze_indexing_opportunities(db_name, geological_context))
            recommendations.extend(await self._analyze_partitioning_opportunities(db_name, geological_context))
            recommendations.extend(await self._analyze_query_optimization_opportunities(db_name))
            recommendations.extend(await self._analyze_storage_optimization_opportunities(db_name))
            
            # Sort recommendations by priority
            recommendations.sort(key=lambda x: x.priority, reverse=True)
            
            # Store recommendations
            self.optimization_history.extend(recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Database optimization failed for {db_name}: {e}")
            return []
    
    async def _analyze_indexing_opportunities(self, db_name: str,
                                           geological_context: Dict[str, Any] = None) -> List[OptimizationRecommendation]:
        """Analyze indexing optimization opportunities"""
        recommendations = []
        
        connection = await self.connection_manager.get_connection(db_name)
        
        try:
            # Find missing indexes for geological tables
            for table_name, pattern in self.geological_table_patterns.items():
                # Check if table exists
                table_exists_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = %s
                )
                """
                
                result = await connection.execute(text(table_exists_query), (table_name,))
                if not result.scalar():
                    continue
                
                # Check for missing recommended indexes
                recommended_indexes = pattern['recommended_indexes']
                existing_indexes = self.existing_indexes.get(table_name, set())
                
                for index_sql in recommended_indexes:
                    # Extract index name from SQL
                    index_name = self._extract_index_name(index_sql)
                    
                    if index_name not in existing_indexes:
                        recommendation = OptimizationRecommendation(
                            recommendation_id=str(uuid4()),
                            optimization_type=OptimizationType.INDEXING,
                            priority=8,
                            description=f"Create missing geological index: {index_name}",
                            expected_improvement=0.3,  # 30% query improvement
                            implementation_effort="low",
                            geological_impact=f"Improves {table_name} query performance for geological analysis",
                            cost_impact=0.05,  # Small storage cost increase
                            affected_tables=[table_name],
                            sql_commands=[index_sql],
                            rollback_commands=[f"DROP INDEX CONCURRENTLY IF EXISTS {index_name}"]
                        )
                        recommendations.append(recommendation)
            
            # Analyze unused indexes
            unused_indexes = await self._find_unused_indexes(connection)
            for index_info in unused_indexes:
                recommendation = OptimizationRecommendation(
                    recommendation_id=str(uuid4()),
                    optimization_type=OptimizationType.INDEXING,
                    priority=6,
                    description=f"Remove unused index: {index_info['index_name']}",
                    expected_improvement=0.1,  # 10% write improvement
                    implementation_effort="low",
                    geological_impact="Reduces storage overhead and improves write performance",
                    cost_impact=-0.02,  # Small storage cost reduction
                    affected_tables=[index_info['table_name']],
                    sql_commands=[f"DROP INDEX CONCURRENTLY {index_info['index_name']}"],
                    rollback_commands=[index_info['index_definition']]
                )
                recommendations.append(recommendation)
            
        finally:
            await self.connection_manager.release_connection(db_name, connection)
        
        return recommendations
    
    async def _analyze_partitioning_opportunities(self, db_name: str,
                                                geological_context: Dict[str, Any] = None) -> List[OptimizationRecommendation]:
        """Analyze table partitioning opportunities"""
        recommendations = []
        
        connection = await self.connection_manager.get_connection(db_name)
        
        try:
            # Check for large tables that could benefit from partitioning
            large_tables_query = """
            SELECT schemaname, tablename, pg_total_relation_size(schemaname||'.'||tablename) as size
            FROM pg_tables
            WHERE schemaname = 'public'
            AND pg_total_relation_size(schemaname||'.'||tablename) > 1073741824  -- 1GB
            ORDER BY size DESC
            """
            
            result = await connection.execute(text(large_tables_query))
            
            for row in result:
                table_name = row[1]
                table_size = row[2]
                
                # Check if table is already partitioned
                partition_check_query = """
                SELECT COUNT(*) FROM pg_inherits 
                WHERE inhparent = (
                    SELECT oid FROM pg_class WHERE relname = %s
                )
                """
                
                partition_result = await connection.execute(text(partition_check_query), (table_name,))
                is_partitioned = partition_result.scalar() > 0
                
                if not is_partitioned and table_name in self.geological_table_patterns:
                    pattern = self.geological_table_patterns[table_name]
                    partition_strategy = pattern['partition_strategy']
                    partition_column = pattern['partition_column']
                    
                    # Generate partitioning SQL
                    partition_sqls = self.sharding_strategy.partition_strategies[partition_strategy](
                        table_name, partition_column
                    )
                    
                    recommendation = OptimizationRecommendation(
                        recommendation_id=str(uuid4()),
                        optimization_type=OptimizationType.PARTITIONING,
                        priority=9,
                        description=f"Partition large table {table_name} using {partition_strategy} strategy",
                        expected_improvement=0.5,  # 50% query improvement for range queries
                        implementation_effort="high",
                        geological_impact=f"Significantly improves query performance for {table_name} geological analysis",
                        cost_impact=0.0,  # No additional cost
                        affected_tables=[table_name],
                        sql_commands=partition_sqls,
                        rollback_commands=[f"-- Manual rollback required for partitioning {table_name}"]
                    )
                    recommendations.append(recommendation)
            
        finally:
            await self.connection_manager.release_connection(db_name, connection)
        
        return recommendations
    
    async def _analyze_query_optimization_opportunities(self, db_name: str) -> List[OptimizationRecommendation]:
        """Analyze query optimization opportunities"""
        recommendations = []
        
        connection = await self.connection_manager.get_connection(db_name)
        
        try:
            # Find slow queries
            slow_queries_query = """
            SELECT query, mean_exec_time, calls, total_exec_time
            FROM pg_stat_statements
            WHERE mean_exec_time > 1000  -- Queries slower than 1 second
            ORDER BY mean_exec_time DESC
            LIMIT 10
            """
            
            try:
                result = await connection.execute(text(slow_queries_query))
                
                for row in result:
                    query = row[0]
                    mean_time = row[1]
                    calls = row[2]
                    
                    if calls > 10:  # Only optimize frequently called queries
                        recommendation = OptimizationRecommendation(
                            recommendation_id=str(uuid4()),
                            optimization_type=OptimizationType.QUERY_OPTIMIZATION,
                            priority=7,
                            description=f"Optimize slow query (avg: {mean_time:.2f}ms, calls: {calls})",
                            expected_improvement=0.4,  # 40% improvement
                            implementation_effort="medium",
                            geological_impact="Improves overall system responsiveness for geological queries",
                            cost_impact=0.0,
                            affected_tables=[],
                            sql_commands=[f"-- Analyze and optimize: {query[:100]}..."],
                            rollback_commands=["-- No rollback needed for query optimization"]
                        )
                        recommendations.append(recommendation)
                        
            except:
                # pg_stat_statements not available
                pass
            
            # Check for missing ANALYZE statistics
            analyze_check_query = """
            SELECT schemaname, tablename, last_analyze, last_autoanalyze
            FROM pg_stat_user_tables
            WHERE (last_analyze IS NULL OR last_analyze < NOW() - INTERVAL '7 days')
            AND (last_autoanalyze IS NULL OR last_autoanalyze < NOW() - INTERVAL '7 days')
            """
            
            result = await connection.execute(text(analyze_check_query))
            
            for row in result:
                table_name = row[1]
                
                recommendation = OptimizationRecommendation(
                    recommendation_id=str(uuid4()),
                    optimization_type=OptimizationType.QUERY_OPTIMIZATION,
                    priority=5,
                    description=f"Update table statistics for {table_name}",
                    expected_improvement=0.2,  # 20% improvement
                    implementation_effort="low",
                    geological_impact="Improves query planner decisions for geological data",
                    cost_impact=0.0,
                    affected_tables=[table_name],
                    sql_commands=[f"ANALYZE {table_name}"],
                    rollback_commands=["-- No rollback needed for ANALYZE"]
                )
                recommendations.append(recommendation)
            
        finally:
            await self.connection_manager.release_connection(db_name, connection)
        
        return recommendations
    
    async def _analyze_storage_optimization_opportunities(self, db_name: str) -> List[OptimizationRecommendation]:
        """Analyze storage optimization opportunities"""
        recommendations = []
        
        connection = await self.connection_manager.get_connection(db_name)
        
        try:
            # Check for bloated tables
            bloat_query = """
            SELECT schemaname, tablename, 
                   pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                   pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
            FROM pg_tables
            WHERE schemaname = 'public'
            AND pg_total_relation_size(schemaname||'.'||tablename) > 104857600  -- 100MB
            ORDER BY size_bytes DESC
            """
            
            result = await connection.execute(text(bloat_query))
            
            for row in result:
                table_name = row[1]
                size_bytes = row[3]
                
                # Estimate bloat (simplified heuristic)
                if size_bytes > 1073741824:  # 1GB
                    recommendation = OptimizationRecommendation(
                        recommendation_id=str(uuid4()),
                        optimization_type=OptimizationType.ARCHIVAL,
                        priority=4,
                        description=f"Consider archiving old data from large table {table_name}",
                        expected_improvement=0.3,  # 30% storage reduction
                        implementation_effort="medium",
                        geological_impact="Reduces storage costs while maintaining recent geological data access",
                        cost_impact=-0.2,  # 20% cost reduction
                        affected_tables=[table_name],
                        sql_commands=[f"-- Implement archival strategy for {table_name}"],
                        rollback_commands=[f"-- Restore archived data for {table_name}"]
                    )
                    recommendations.append(recommendation)
            
        finally:
            await self.connection_manager.release_connection(db_name, connection)
        
        return recommendations
    
    async def _find_unused_indexes(self, connection) -> List[Dict[str, Any]]:
        """Find unused indexes that can be dropped"""
        unused_indexes_query = """
        SELECT schemaname, tablename, indexname, indexdef, idx_scan
        FROM pg_stat_user_indexes pui
        JOIN pg_indexes pi ON pui.indexrelname = pi.indexname
        WHERE pui.idx_scan = 0
        AND pui.indexrelname NOT LIKE '%_pkey'  -- Exclude primary keys
        AND pi.indexdef NOT LIKE '%UNIQUE%'     -- Exclude unique constraints
        """
        
        result = await connection.execute(text(unused_indexes_query))
        
        unused_indexes = []
        for row in result:
            unused_indexes.append({
                'schema_name': row[0],
                'table_name': row[1],
                'index_name': row[2],
                'index_definition': row[3],
                'scan_count': row[4]
            })
        
        return unused_indexes
    
    def _extract_index_name(self, index_sql: str) -> str:
        """Extract index name from CREATE INDEX SQL"""
        # Simple regex to extract index name
        import re
        match = re.search(r'CREATE INDEX (?:CONCURRENTLY )?(\w+)', index_sql, re.IGNORECASE)
        return match.group(1) if match else "unknown_index"
    
    async def apply_optimization(self, db_name: str, recommendation: OptimizationRecommendation) -> bool:
        """Apply optimization recommendation"""
        connection = await self.connection_manager.get_connection(db_name)
        
        try:
            # Execute optimization SQL commands
            for sql_command in recommendation.sql_commands:
                if sql_command.strip() and not sql_command.strip().startswith('--'):
                    await connection.execute(text(sql_command))
            
            await connection.commit()
            
            self.logger.info(f"Applied optimization: {recommendation.description}")
            return True
            
        except Exception as e:
            await connection.rollback()
            self.logger.error(f"Failed to apply optimization {recommendation.recommendation_id}: {e}")
            return False
        finally:
            await self.connection_manager.release_connection(db_name, connection)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary and metrics"""
        summary = {
            'total_recommendations': len(self.optimization_history),
            'recommendations_by_type': defaultdict(int),
            'database_metrics': self.performance_metrics,
            'pool_metrics': self.connection_manager.get_pool_metrics(),
            'recent_recommendations': []
        }
        
        # Count recommendations by type
        for rec in self.optimization_history:
            summary['recommendations_by_type'][rec.optimization_type.value] += 1
        
        # Get recent recommendations (last 10)
        recent_recs = sorted(
            self.optimization_history,
            key=lambda x: x.priority,
            reverse=True
        )[:10]
        
        summary['recent_recommendations'] = [
            {
                'id': rec.recommendation_id,
                'type': rec.optimization_type.value,
                'priority': rec.priority,
                'description': rec.description,
                'expected_improvement': rec.expected_improvement,
                'implementation_effort': rec.implementation_effort
            }
            for rec in recent_recs
        ]
        
        return summary


# Configuration and factory
class DatabaseOptimizerConfig(BaseModel):
    """Database optimizer configuration"""
    databases: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Database configurations"
    )
    optimization_interval: int = Field(
        default=3600,
        description="Optimization analysis interval in seconds"
    )
    auto_apply_optimizations: bool = Field(
        default=False,
        description="Automatically apply low-risk optimizations"
    )
    geological_optimization: bool = Field(
        default=True,
        description="Enable geological domain-aware optimization"
    )


async def create_database_optimizer(config: DatabaseOptimizerConfig) -> DatabaseOptimizer:
    """Factory function to create database optimizer"""
    optimizer = DatabaseOptimizer(config.databases)
    await optimizer.initialize()
    return optimizer


# Example usage and testing
if __name__ == "__main__":
    async def test_database_optimizer():
        """Test the database optimizer"""
        config = DatabaseOptimizerConfig(
            databases={
                "prometheus_main": {
                    "type": "postgresql",
                    "url": "postgresql+asyncpg://user:password@localhost:5432/prometheus",
                    "pool_size": 20,
                    "max_overflow": 30,
                    "geological_domain": "volcanic_monitoring"
                }
            }
        )
        
        optimizer = await create_database_optimizer(config)
        
        # Test optimization analysis
        recommendations = await optimizer.optimize_database(
            "prometheus_main",
            {"domain": "volcanic_monitoring", "volcano_id": "kilauea_001"}
        )
        
        print(f"Generated {len(recommendations)} optimization recommendations:")
        for rec in recommendations:
            print(f"- {rec.optimization_type.value}: {rec.description} (Priority: {rec.priority})")
        
        # Get optimization summary
        summary = optimizer.get_optimization_summary()
        print(f"Optimization summary: {summary}")
    
    # Run test
    asyncio.run(test_database_optimizer())