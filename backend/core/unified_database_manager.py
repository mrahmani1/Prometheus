"""
Unified Database Management Layer
Centralized database operations supporting PostgreSQL+PostGIS, InfluxDB, Redis, Neo4j, TimescaleDB
Production-grade implementation with connection pooling, health monitoring, and performance optimization
"""

import asyncio
import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, AsyncIterator, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid

# Database clients
import asyncpg
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Index, MetaData
from sqlalchemy.dialects.postgresql import UUID, JSONB
from geoalchemy2 import Geometry

# InfluxDB
try:
    from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
    from influxdb_client import Point, WritePrecision
    from influxdb_client.client.write_api_async import WriteApiAsync
    from influxdb_client.client.query_api_async import QueryApiAsync
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

# Neo4j
try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# TimescaleDB
try:
    import timescaledb
    TIMESCALEDB_AVAILABLE = True
except ImportError:
    TIMESCALEDB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    INFLUXDB = "influxdb"
    REDIS = "redis"
    NEO4J = "neo4j"
    TIMESCALEDB = "timescaledb"

class ConnectionStatus(Enum):
    """Database connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RECONNECTING = "reconnecting"

@dataclass
class DatabaseConfig:
    """Database configuration with environment variable support"""
    db_type: DatabaseType
    host: str = "localhost"
    port: int = 5432
    database: str = "prometheus"
    username: str = "prometheus"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 30
    timeout: int = 30
    ssl_enabled: bool = False
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls, db_type: DatabaseType) -> 'DatabaseConfig':
        """Create configuration from environment variables"""
        env_prefix = db_type.value.upper()
        
        return cls(
            db_type=db_type,
            host=os.getenv(f'{env_prefix}_HOST', 'localhost'),
            port=int(os.getenv(f'{env_prefix}_PORT', cls._get_default_port(db_type))),
            database=os.getenv(f'{env_prefix}_DATABASE', 'prometheus'),
            username=os.getenv(f'{env_prefix}_USERNAME', 'prometheus'),
            password=os.getenv(f'{env_prefix}_PASSWORD', ''),
            pool_size=int(os.getenv(f'{env_prefix}_POOL_SIZE', '20')),
            max_overflow=int(os.getenv(f'{env_prefix}_MAX_OVERFLOW', '30')),
            timeout=int(os.getenv(f'{env_prefix}_TIMEOUT', '30')),
            ssl_enabled=os.getenv(f'{env_prefix}_SSL_ENABLED', 'false').lower() == 'true'
        )
    
    @staticmethod
    def _get_default_port(db_type: DatabaseType) -> str:
        """Get default port for database type"""
        defaults = {
            DatabaseType.POSTGRESQL: "5432",
            DatabaseType.INFLUXDB: "8086",
            DatabaseType.REDIS: "6379",
            DatabaseType.NEO4J: "7687",
            DatabaseType.TIMESCALEDB: "5432"
        }
        return defaults.get(db_type, "5432")

@dataclass
class HealthMetrics:
    """Database health and performance metrics"""
    connection_count: int = 0
    active_queries: int = 0
    total_queries: int = 0
    avg_query_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    storage_usage: float = 0.0
    last_backup: Optional[datetime] = None
    
@dataclass
class QueryResult:
    """Standardized query result"""
    data: Any
    metadata: Dict[str, Any]
    execution_time: float
    row_count: int
    cached: bool = False
    database_type: DatabaseType = DatabaseType.POSTGRESQL

class DatabaseConnection:
    """Individual database connection wrapper"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.connection = None
        self.pool = None
        self.last_ping = None
        self.error_count = 0
        self.metrics = HealthMetrics()
        self.lock = asyncio.Lock()
        
    async def connect(self) -> bool:
        """Establish database connection"""
        try:
            async with self.lock:
                if self.config.db_type == DatabaseType.POSTGRESQL:
                    await self._connect_postgresql()
                elif self.config.db_type == DatabaseType.INFLUXDB:
                    await self._connect_influxdb()
                elif self.config.db_type == DatabaseType.REDIS:
                    await self._connect_redis()
                elif self.config.db_type == DatabaseType.NEO4J:
                    await self._connect_neo4j()
                elif self.config.db_type == DatabaseType.TIMESCALEDB:
                    await self._connect_timescaledb()
                
                self.status = ConnectionStatus.CONNECTED
                self.last_ping = datetime.utcnow()
                logger.info(f"Connected to {self.config.db_type.value} database")
                return True
                
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Failed to connect to {self.config.db_type.value}: {e}")
            return False
    
    async def _connect_postgresql(self):
        """Connect to PostgreSQL/TimescaleDB"""
        connection_string = (
            f"postgresql://{self.config.username}:{self.config.password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )
        
        # Create async engine for SQLAlchemy
        self.pool = create_async_engine(
            connection_string,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.timeout,
            pool_recycle=3600,
            echo=False
        )
        
        # Test connection
        async with self.pool.connect() as conn:
            await conn.execute("SELECT 1")
    
    async def _connect_influxdb(self):
        """Connect to InfluxDB"""
        if not INFLUXDB_AVAILABLE:
            raise ImportError("InfluxDB client not available")
        
        url = f"http://{self.config.host}:{self.config.port}"
        self.connection = InfluxDBClientAsync(
            url=url,
            token=self.config.password,  # InfluxDB uses token as password
            org=self.config.additional_params.get('org', 'prometheus'),
            timeout=self.config.timeout * 1000
        )
        
        # Test connection
        health = await self.connection.health()
        if health.status != 'pass':
            raise ConnectionError(f"InfluxDB health check failed: {health.message}")
    
    async def _connect_redis(self):
        """Connect to Redis"""
        self.connection = redis.Redis(
            host=self.config.host,
            port=self.config.port,
            password=self.config.password if self.config.password else None,
            db=self.config.additional_params.get('db', 0),
            decode_responses=True,
            socket_timeout=self.config.timeout,
            socket_connect_timeout=self.config.timeout,
            retry_on_timeout=True,
            max_connections=self.config.pool_size
        )
        
        # Test connection
        await self.connection.ping()
    
    async def _connect_neo4j(self):
        """Connect to Neo4j"""
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available")
        
        uri = f"bolt://{self.config.host}:{self.config.port}"
        self.connection = AsyncGraphDatabase.driver(
            uri,
            auth=(self.config.username, self.config.password),
            max_connection_lifetime=3600,
            max_connection_pool_size=self.config.pool_size
        )
        
        # Test connection
        async with self.connection.session() as session:
            await session.run("RETURN 1")
    
    async def _connect_timescaledb(self):
        """Connect to TimescaleDB (same as PostgreSQL with extensions)"""
        await self._connect_postgresql()
        
        # Enable TimescaleDB extension
        async with self.pool.connect() as conn:
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
            except Exception as e:
                logger.warning(f"TimescaleDB extension setup failed: {e}")
    
    async def disconnect(self):
        """Disconnect from database"""
        try:
            async with self.lock:
                if self.connection:
                    if hasattr(self.connection, 'close'):
                        await self.connection.close()
                    elif hasattr(self.connection, 'aclose'):
                        await self.connection.aclose()
                
                if self.pool:
                    await self.pool.dispose()
                
                self.status = ConnectionStatus.DISCONNECTED
                logger.info(f"Disconnected from {self.config.db_type.value} database")
                
        except Exception as e:
            logger.error(f"Error disconnecting from {self.config.db_type.value}: {e}")
    
    async def ping(self) -> bool:
        """Ping database to check connectivity"""
        try:
            if self.config.db_type == DatabaseType.POSTGRESQL:
                async with self.pool.connect() as conn:
                    await conn.execute("SELECT 1")
            elif self.config.db_type == DatabaseType.INFLUXDB:
                health = await self.connection.health()
                return health.status == 'pass'
            elif self.config.db_type == DatabaseType.REDIS:
                await self.connection.ping()
            elif self.config.db_type == DatabaseType.NEO4J:
                async with self.connection.session() as session:
                    await session.run("RETURN 1")
            
            self.last_ping = datetime.utcnow()
            return True
            
        except Exception as e:
            logger.error(f"Ping failed for {self.config.db_type.value}: {e}")
            return False

class UnifiedDatabaseManager:
    """
    Unified Database Management Layer
    Centralized management for all database types with connection pooling,
    health monitoring, and performance optimization
    """
    
    def __init__(self, configs: Optional[Dict[DatabaseType, DatabaseConfig]] = None):
        self.configs = configs or self._load_default_configs()
        self.connections: Dict[DatabaseType, DatabaseConnection] = {}
        self.session_factories: Dict[DatabaseType, Any] = {}
        
        # Health monitoring
        self.health_monitor_task = None
        self.health_check_interval = 60  # seconds
        
        # Performance monitoring
        self.query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(minutes=15)
        self.performance_metrics: Dict[DatabaseType, HealthMetrics] = {}
        
        # Connection management
        self.connection_lock = asyncio.Lock()
        self.reconnect_attempts = {}
        self.max_reconnect_attempts = 3
        
        # Query optimization
        self.query_optimizer = QueryOptimizer()
        
        logger.info("Unified Database Manager initialized")
    
    def _load_default_configs(self) -> Dict[DatabaseType, DatabaseConfig]:
        """Load default database configurations from environment"""
        configs = {}
        
        for db_type in DatabaseType:
            try:
                config = DatabaseConfig.from_env(db_type)
                configs[db_type] = config
            except Exception as e:
                logger.warning(f"Failed to load config for {db_type.value}: {e}")
        
        return configs
    
    async def initialize(self) -> bool:
        """Initialize all database connections"""
        try:
            logger.info("Initializing unified database manager")
            
            # Connect to all configured databases
            connection_tasks = []
            for db_type, config in self.configs.items():
                connection = DatabaseConnection(config)
                self.connections[db_type] = connection
                connection_tasks.append(connection.connect())
            
            # Wait for all connections
            results = await asyncio.gather(*connection_tasks, return_exceptions=True)
            
            # Check results
            successful_connections = 0
            for i, (db_type, result) in enumerate(zip(self.configs.keys(), results)):
                if isinstance(result, Exception):
                    logger.error(f"Failed to connect to {db_type.value}: {result}")
                elif result:
                    successful_connections += 1
                    # Create session factories for SQLAlchemy-based databases
                    if db_type in [DatabaseType.POSTGRESQL, DatabaseType.TIMESCALEDB]:
                        self.session_factories[db_type] = async_sessionmaker(
                            self.connections[db_type].pool,
                            class_=AsyncSession,
                            expire_on_commit=False
                        )
            
            if successful_connections == 0:
                logger.error("No database connections established")
                return False
            
            # Initialize database schemas
            await self._initialize_schemas()
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            logger.info(f"Database manager initialized with {successful_connections} connections")
            return True
            
        except Exception as e:
            logger.error(f"Database manager initialization failed: {e}")
            return False
    
    async def _initialize_schemas(self):
        """Initialize database schemas and tables"""
        # PostgreSQL/TimescaleDB schema initialization
        if DatabaseType.POSTGRESQL in self.connections:
            await self._initialize_postgresql_schema()
        
        # InfluxDB bucket initialization
        if DatabaseType.INFLUXDB in self.connections:
            await self._initialize_influxdb_buckets()
        
        # Neo4j constraints and indexes
        if DatabaseType.NEO4J in self.connections:
            await self._initialize_neo4j_schema()
    
    async def _initialize_postgresql_schema(self):
        """Initialize PostgreSQL schema with PostGIS and TimescaleDB"""
        try:
            conn = self.connections[DatabaseType.POSTGRESQL]
            if conn.status != ConnectionStatus.CONNECTED:
                return
            
            async with conn.pool.connect() as db_conn:
                # Enable extensions
                await db_conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
                await db_conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
                
                # Create geological data tables
                await db_conn.execute("""
                    CREATE TABLE IF NOT EXISTS geological_observations (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        timestamp TIMESTAMPTZ NOT NULL,
                        location GEOMETRY(POINT, 4326) NOT NULL,
                        observation_type VARCHAR(50) NOT NULL,
                        value JSONB NOT NULL,
                        uncertainty FLOAT,
                        quality_score FLOAT,
                        data_source VARCHAR(100) NOT NULL,
                        processing_metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                # Create hypertable for time-series data
                try:
                    await db_conn.execute("""
                        SELECT create_hypertable('geological_observations', 'timestamp', 
                                               if_not_exists => TRUE)
                    """)
                except Exception as e:
                    logger.warning(f"Hypertable creation failed: {e}")
                
                # Create indexes
                await db_conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_geo_obs_timestamp 
                    ON geological_observations (timestamp DESC)
                """)
                await db_conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_geo_obs_location 
                    ON geological_observations USING GIST (location)
                """)
                await db_conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_geo_obs_type 
                    ON geological_observations (observation_type)
                """)
                
                await db_conn.commit()
                
        except Exception as e:
            logger.error(f"PostgreSQL schema initialization failed: {e}")
    
    async def _initialize_influxdb_buckets(self):
        """Initialize InfluxDB buckets"""
        try:
            conn = self.connections[DatabaseType.INFLUXDB]
            if conn.status != ConnectionStatus.CONNECTED:
                return
            
            # Create buckets for different data types
            buckets_api = conn.connection.buckets_api()
            org = self.configs[DatabaseType.INFLUXDB].additional_params.get('org', 'prometheus')
            
            bucket_configs = [
                {'name': 'geological_data', 'retention': '30d'},
                {'name': 'real_time_monitoring', 'retention': '7d'},
                {'name': 'causal_relationships', 'retention': '365d'},
                {'name': 'performance_metrics', 'retention': '90d'}
            ]
            
            for bucket_config in bucket_configs:
                try:
                    await buckets_api.create_bucket(
                        bucket_name=bucket_config['name'],
                        org=org,
                        retention_rules=[{'type': 'expire', 'everySeconds': 
                                       self._parse_retention(bucket_config['retention'])}]
                    )
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Bucket creation failed for {bucket_config['name']}: {e}")
                        
        except Exception as e:
            logger.error(f"InfluxDB bucket initialization failed: {e}")
    
    def _parse_retention(self, retention_str: str) -> int:
        """Parse retention string to seconds"""
        if retention_str.endswith('d'):
            return int(retention_str[:-1]) * 24 * 3600
        elif retention_str.endswith('h'):
            return int(retention_str[:-1]) * 3600
        elif retention_str.endswith('m'):
            return int(retention_str[:-1]) * 60
        else:
            return int(retention_str)
    
    async def _initialize_neo4j_schema(self):
        """Initialize Neo4j constraints and indexes"""
        try:
            conn = self.connections[DatabaseType.NEO4J]
            if conn.status != ConnectionStatus.CONNECTED:
                return
            
            async with conn.connection.session() as session:
                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT geological_process_id IF NOT EXISTS FOR (p:GeologicalProcess) REQUIRE p.id IS UNIQUE",
                    "CREATE CONSTRAINT causal_relationship_id IF NOT EXISTS FOR (r:CausalRelationship) REQUIRE r.id IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        await session.run(constraint)
                    except Exception as e:
                        logger.warning(f"Constraint creation failed: {e}")
                
                # Create indexes
                indexes = [
                    "CREATE INDEX geological_process_type IF NOT EXISTS FOR (p:GeologicalProcess) ON (p.process_type)",
                    "CREATE INDEX causal_strength IF NOT EXISTS FOR ()-[r:CAUSES]-() ON (r.strength)"
                ]
                
                for index in indexes:
                    try:
                        await session.run(index)
                    except Exception as e:
                        logger.warning(f"Index creation failed: {e}")
                        
        except Exception as e:
            logger.error(f"Neo4j schema initialization failed: {e}")    
    
    async def _start_health_monitoring(self):
        """Start health monitoring task"""
        async def health_monitor():
            while True:
                try:
                    await asyncio.sleep(self.health_check_interval)
                    await self._check_all_connections()
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
        
        self.health_monitor_task = asyncio.create_task(health_monitor())
        logger.info("Health monitoring started")
    
    async def _check_all_connections(self):
        """Check health of all database connections"""
        for db_type, connection in self.connections.items():
            try:
                is_healthy = await connection.ping()
                if not is_healthy and connection.status == ConnectionStatus.CONNECTED:
                    logger.warning(f"{db_type.value} connection unhealthy, attempting reconnect")
                    await self._attempt_reconnect(db_type)
                    
            except Exception as e:
                logger.error(f"Health check failed for {db_type.value}: {e}")
    
    async def _attempt_reconnect(self, db_type: DatabaseType):
        """Attempt to reconnect to database"""
        if db_type not in self.reconnect_attempts:
            self.reconnect_attempts[db_type] = 0
        
        if self.reconnect_attempts[db_type] >= self.max_reconnect_attempts:
            logger.error(f"Max reconnect attempts reached for {db_type.value}")
            return
        
        try:
            connection = self.connections[db_type]
            connection.status = ConnectionStatus.RECONNECTING
            
            await connection.disconnect()
            success = await connection.connect()
            
            if success:
                self.reconnect_attempts[db_type] = 0
                logger.info(f"Successfully reconnected to {db_type.value}")
            else:
                self.reconnect_attempts[db_type] += 1
                
        except Exception as e:
            self.reconnect_attempts[db_type] += 1
            logger.error(f"Reconnect failed for {db_type.value}: {e}")
    
    @asynccontextmanager
    async def get_session(self, db_type: DatabaseType):
        """Get database session with automatic cleanup"""
        if db_type not in self.session_factories:
            raise ValueError(f"No session factory for {db_type.value}")
        
        async with self.session_factories[db_type]() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def execute_query(self, db_type: DatabaseType, query: str, 
                          params: Optional[Dict[str, Any]] = None,
                          use_cache: bool = True) -> QueryResult:
        """Execute query with caching and performance monitoring"""
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(db_type, query, params)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
        
        try:
            # Execute query based on database type
            if db_type == DatabaseType.POSTGRESQL:
                result = await self._execute_postgresql_query(query, params)
            elif db_type == DatabaseType.INFLUXDB:
                result = await self._execute_influxdb_query(query, params)
            elif db_type == DatabaseType.REDIS:
                result = await self._execute_redis_query(query, params)
            elif db_type == DatabaseType.NEO4J:
                result = await self._execute_neo4j_query(query, params)
            else:
                raise ValueError(f"Unsupported database type: {db_type.value}")
            
            execution_time = time.time() - start_time
            
            query_result = QueryResult(
                data=result,
                metadata={'query': query, 'params': params},
                execution_time=execution_time,
                row_count=len(result) if isinstance(result, list) else 1,
                database_type=db_type
            )
            
            # Cache result
            if use_cache:
                self._cache_result(cache_key, query_result)
            
            # Update metrics
            self._update_performance_metrics(db_type, execution_time)
            
            return query_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed for {db_type.value}: {e}")
            
            return QueryResult(
                data=None,
                metadata={'error': str(e), 'query': query},
                execution_time=execution_time,
                row_count=0,
                database_type=db_type
            )
    
    async def _execute_postgresql_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute PostgreSQL query"""
        connection = self.connections[DatabaseType.POSTGRESQL]
        if connection.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("PostgreSQL not connected")
        
        async with connection.pool.connect() as conn:
            if params:
                result = await conn.execute(query, params)
            else:
                result = await conn.execute(query)
            
            if result.returns_rows:
                return [dict(row) for row in result.fetchall()]
            else:
                return [{'affected_rows': result.rowcount}]
    
    async def _execute_influxdb_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute InfluxDB Flux query"""
        connection = self.connections[DatabaseType.INFLUXDB]
        if connection.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("InfluxDB not connected")
        
        query_api = connection.connection.query_api()
        
        # Replace parameters in Flux query
        if params:
            for key, value in params.items():
                query = query.replace(f"${key}", str(value))
        
        tables = await query_api.query(query)
        
        results = []
        for table in tables:
            for record in table.records:
                results.append({
                    'time': record.get_time(),
                    'measurement': record.get_measurement(),
                    'field': record.get_field(),
                    'value': record.get_value(),
                    'tags': {k: v for k, v in record.values.items() 
                            if not k.startswith('_') and k != 'result' and k != 'table'}
                })
        
        return results
    
    async def _execute_redis_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute Redis command"""
        connection = self.connections[DatabaseType.REDIS]
        if connection.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Redis not connected")
        
        # Parse Redis command
        parts = query.split()
        command = parts[0].upper()
        args = parts[1:]
        
        # Replace parameters
        if params:
            for i, arg in enumerate(args):
                if arg.startswith('$'):
                    param_name = arg[1:]
                    if param_name in params:
                        args[i] = str(params[param_name])
        
        # Execute command
        if command == 'GET':
            result = await connection.connection.get(args[0])
            return [{'key': args[0], 'value': result}]
        elif command == 'SET':
            result = await connection.connection.set(args[0], args[1])
            return [{'success': result}]
        elif command == 'KEYS':
            result = await connection.connection.keys(args[0])
            return [{'keys': result}]
        else:
            # Generic command execution
            result = await connection.connection.execute_command(command, *args)
            return [{'result': result}]
    
    async def _execute_neo4j_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute Neo4j Cypher query"""
        connection = self.connections[DatabaseType.NEO4J]
        if connection.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Neo4j not connected")
        
        async with connection.connection.session() as session:
            result = await session.run(query, params or {})
            
            records = []
            async for record in result:
                records.append(dict(record))
            
            return records
    
    def _generate_cache_key(self, db_type: DatabaseType, query: str, 
                          params: Optional[Dict] = None) -> str:
        """Generate cache key for query"""
        key_data = f"{db_type.value}:{query}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[QueryResult]:
        """Get cached query result"""
        if cache_key in self.query_cache:
            result, timestamp = self.query_cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_ttl:
                result.cached = True
                return result
            else:
                del self.query_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: QueryResult):
        """Cache query result"""
        self.query_cache[cache_key] = (result, datetime.utcnow())
        
        # Clean old cache entries
        if len(self.query_cache) > 1000:
            self._clean_cache()
    
    def _clean_cache(self):
        """Clean expired cache entries"""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, (_, timestamp) in self.query_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.query_cache[key]
    
    def _update_performance_metrics(self, db_type: DatabaseType, execution_time: float):
        """Update performance metrics"""
        if db_type not in self.performance_metrics:
            self.performance_metrics[db_type] = HealthMetrics()
        
        metrics = self.performance_metrics[db_type]
        metrics.total_queries += 1
        
        # Update average query time
        if metrics.avg_query_time == 0:
            metrics.avg_query_time = execution_time
        else:
            metrics.avg_query_time = (metrics.avg_query_time + execution_time) / 2
    
    async def store_geological_observation(self, observation: Dict[str, Any]) -> str:
        """Store geological observation across appropriate databases"""
        observation_id = str(uuid.uuid4())
        
        try:
            # Store in PostgreSQL for geospatial queries
            if DatabaseType.POSTGRESQL in self.connections:
                await self._store_observation_postgresql(observation_id, observation)
            
            # Store in InfluxDB for time-series analysis
            if DatabaseType.INFLUXDB in self.connections:
                await self._store_observation_influxdb(observation_id, observation)
            
            # Cache in Redis for fast access
            if DatabaseType.REDIS in self.connections:
                await self._cache_observation_redis(observation_id, observation)
            
            return observation_id
            
        except Exception as e:
            logger.error(f"Failed to store geological observation: {e}")
            raise
    
    async def _store_observation_postgresql(self, obs_id: str, observation: Dict[str, Any]):
        """Store observation in PostgreSQL"""
        query = """
            INSERT INTO geological_observations 
            (id, timestamp, location, observation_type, value, uncertainty, 
             quality_score, data_source, processing_metadata)
            VALUES ($1, $2, ST_SetSRID(ST_MakePoint($3, $4, $5), 4326), $6, $7, $8, $9, $10, $11)
        """
        
        params = {
            'id': obs_id,
            'timestamp': observation['timestamp'],
            'longitude': observation['location'][0],
            'latitude': observation['location'][1], 
            'elevation': observation['location'][2],
            'observation_type': observation['observation_type'],
            'value': json.dumps(observation['value']),
            'uncertainty': observation.get('uncertainty'),
            'quality_score': observation.get('quality_score', 0.5),
            'data_source': observation['data_source'],
            'processing_metadata': json.dumps(observation.get('metadata', {}))
        }
        
        await self.execute_query(DatabaseType.POSTGRESQL, query, params, use_cache=False)
    
    async def _store_observation_influxdb(self, obs_id: str, observation: Dict[str, Any]):
        """Store observation in InfluxDB"""
        connection = self.connections[DatabaseType.INFLUXDB]
        if connection.status != ConnectionStatus.CONNECTED:
            return
        
        point = Point("geological_observation") \
            .tag("observation_type", observation['observation_type']) \
            .tag("data_source", observation['data_source']) \
            .tag("observation_id", obs_id) \
            .field("longitude", observation['location'][0]) \
            .field("latitude", observation['location'][1]) \
            .field("elevation", observation['location'][2]) \
            .field("quality_score", observation.get('quality_score', 0.5)) \
            .time(observation['timestamp'], WritePrecision.NS)
        
        # Add observation values as fields
        for key, value in observation['value'].items():
            if isinstance(value, (int, float)):
                point = point.field(f"value_{key}", value)
        
        write_api = connection.connection.write_api()
        bucket = self.configs[DatabaseType.INFLUXDB].additional_params.get('bucket', 'geological_data')
        
        await write_api.write(bucket=bucket, record=point)
    
    async def _cache_observation_redis(self, obs_id: str, observation: Dict[str, Any]):
        """Cache observation in Redis"""
        connection = self.connections[DatabaseType.REDIS]
        if connection.status != ConnectionStatus.CONNECTED:
            return
        
        cache_key = f"observation:{obs_id}"
        cache_data = json.dumps(observation, default=str)
        
        # Cache for 1 hour
        await connection.connection.setex(cache_key, 3600, cache_data)
    
    async def query_geological_observations(self, 
                                         start_time: Optional[datetime] = None,
                                         end_time: Optional[datetime] = None,
                                         observation_type: Optional[str] = None,
                                         bbox: Optional[Tuple[float, float, float, float]] = None,
                                         limit: int = 1000) -> List[Dict[str, Any]]:
        """Query geological observations with filters"""
        
        # Build PostgreSQL query for geospatial filtering
        query = """
            SELECT id, timestamp, ST_X(location) as longitude, ST_Y(location) as latitude,
                   ST_Z(location) as elevation, observation_type, value, uncertainty,
                   quality_score, data_source, processing_metadata
            FROM geological_observations
            WHERE 1=1
        """
        
        params = {}
        param_count = 1
        
        if start_time:
            query += f" AND timestamp >= ${param_count}"
            params[f'param{param_count}'] = start_time
            param_count += 1
        
        if end_time:
            query += f" AND timestamp <= ${param_count}"
            params[f'param{param_count}'] = end_time
            param_count += 1
        
        if observation_type:
            query += f" AND observation_type = ${param_count}"
            params[f'param{param_count}'] = observation_type
            param_count += 1
        
        if bbox:
            # bbox format: (min_lon, min_lat, max_lon, max_lat)
            query += f" AND ST_Within(location, ST_MakeEnvelope(${param_count}, ${param_count+1}, ${param_count+2}, ${param_count+3}, 4326))"
            params[f'param{param_count}'] = bbox[0]
            params[f'param{param_count+1}'] = bbox[1] 
            params[f'param{param_count+2}'] = bbox[2]
            params[f'param{param_count+3}'] = bbox[3]
            param_count += 4
        
        query += f" ORDER BY timestamp DESC LIMIT ${param_count}"
        params[f'param{param_count}'] = limit
        
        result = await self.execute_query(DatabaseType.POSTGRESQL, query, params)
        
        # Parse JSON fields
        observations = []
        for row in result.data or []:
            row['value'] = json.loads(row['value']) if row['value'] else {}
            row['processing_metadata'] = json.loads(row['processing_metadata']) if row['processing_metadata'] else {}
            observations.append(row)
        
        return observations
    
    async def get_database_health(self) -> Dict[DatabaseType, Dict[str, Any]]:
        """Get health status of all databases"""
        health_status = {}
        
        for db_type, connection in self.connections.items():
            try:
                is_healthy = await connection.ping()
                metrics = self.performance_metrics.get(db_type, HealthMetrics())
                
                health_status[db_type] = {
                    'status': connection.status.value,
                    'healthy': is_healthy,
                    'last_ping': connection.last_ping.isoformat() if connection.last_ping else None,
                    'error_count': connection.error_count,
                    'total_queries': metrics.total_queries,
                    'avg_query_time': metrics.avg_query_time,
                    'last_error': metrics.last_error
                }
                
            except Exception as e:
                health_status[db_type] = {
                    'status': 'error',
                    'healthy': False,
                    'error': str(e)
                }
        
        return health_status
    
    async def cleanup(self):
        """Clean up database connections and resources"""
        logger.info("Cleaning up unified database manager")
        
        # Stop health monitoring
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all databases
        disconnect_tasks = []
        for connection in self.connections.values():
            disconnect_tasks.append(connection.disconnect())
        
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        # Clear caches
        self.query_cache.clear()
        self.performance_metrics.clear()
        
        logger.info("Database manager cleanup complete")

class QueryOptimizer:
    """Query optimization and routing logic"""
    
    def __init__(self):
        self.query_patterns = {}
        self.performance_history = {}
    
    def optimize_query(self, db_type: DatabaseType, query: str) -> str:
        """Optimize query for specific database type"""
        if db_type == DatabaseType.POSTGRESQL:
            return self._optimize_postgresql_query(query)
        elif db_type == DatabaseType.INFLUXDB:
            return self._optimize_influxdb_query(query)
        elif db_type == DatabaseType.NEO4J:
            return self._optimize_neo4j_query(query)
        else:
            return query
    
    def _optimize_postgresql_query(self, query: str) -> str:
        """Optimize PostgreSQL query"""
        # Add query hints and optimizations
        optimized = query
        
        # Add spatial index hints for geospatial queries
        if 'ST_' in query and 'location' in query:
            optimized = f"/*+ USE_INDEX(geological_observations idx_geo_obs_location) */ {optimized}"
        
        # Add time-based index hints
        if 'timestamp' in query:
            optimized = f"/*+ USE_INDEX(geological_observations idx_geo_obs_timestamp) */ {optimized}"
        
        return optimized
    
    def _optimize_influxdb_query(self, query: str) -> str:
        """Optimize InfluxDB Flux query"""
        # Add optimization hints for InfluxDB
        optimized = query
        
        # Ensure proper time range filtering
        if 'range(' not in query and 'from(' in query:
            # Add default time range if not specified
            optimized = optimized.replace('from(', 'from(\n  |> range(start: -1h)\n  |> from(')
        
        return optimized
    
    def _optimize_neo4j_query(self, query: str) -> str:
        """Optimize Neo4j Cypher query"""
        # Add Cypher optimization hints
        optimized = query
        
        # Add index hints for node lookups
        if 'MATCH (p:GeologicalProcess' in query:
            optimized = optimized.replace(
                'MATCH (p:GeologicalProcess',
                'MATCH (p:GeologicalProcess USING INDEX p:GeologicalProcess(id)'
            )
        
        return optimized

# Global database manager instance
_db_manager: Optional[UnifiedDatabaseManager] = None

async def get_database_manager() -> UnifiedDatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = UnifiedDatabaseManager()
        await _db_manager.initialize()
    
    return _db_manager

async def cleanup_database_manager():
    """Cleanup global database manager"""
    global _db_manager
    
    if _db_manager:
        await _db_manager.cleanup()
        _db_manager = None