"""
Unified Database Management Layer for Real Geological Data Integration
Supports PostgreSQL+PostGIS, InfluxDB, Redis, Neo4j, and TimescaleDB with production-grade features

This module provides centralized database operations, connection pooling, health monitoring,
and real data integration for the Prometheus AI Platform.

Requirements: 9.1, 9.2, 9.3, 6.1, 6.2
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, AsyncIterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncpg
import aioredis
import aiohttp
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client import InfluxDBClient, Point
from neo4j import AsyncGraphDatabase
import psycopg2
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, DateTime, Integer
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import ssl
from contextlib import asynccontextmanager
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import threading

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
    host: str = field(default_factory=lambda: os.getenv('DB_HOST', 'localhost'))
    port: int = field(default_factory=lambda: int(os.getenv('DB_PORT', '5432')))
    database: str = field(default_factory=lambda: os.getenv('DB_NAME', 'prometheus'))
    username: str = field(default_factory=lambda: os.getenv('DB_USER', 'prometheus'))
    password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD', ''))
    ssl_enabled: bool = field(default_factory=lambda: os.getenv('DB_SSL', 'false').lower() == 'true')
    pool_size: int = field(default_factory=lambda: int(os.getenv('DB_POOL_SIZE', '10')))
    max_overflow: int = field(default_factory=lambda: int(os.getenv('DB_MAX_OVERFLOW', '20')))
    connection_timeout: int = field(default_factory=lambda: int(os.getenv('DB_TIMEOUT', '30')))
    retry_attempts: int = field(default_factory=lambda: int(os.getenv('DB_RETRY_ATTEMPTS', '3')))
    health_check_interval: int = field(default_factory=lambda: int(os.getenv('DB_HEALTH_CHECK_INTERVAL', '60')))

@dataclass
class ConnectionMetrics:
    """Database connection metrics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    average_response_time: float = 0.0
    last_health_check: Optional[datetime] = None
    uptime_percentage: float = 100.0

class DatabaseManager:
    """
    Unified database management system supporting multiple database types
    with production-grade features including connection pooling, health monitoring,
    automatic failover, and comprehensive error handling.
    """
    
    def __init__(self, configs: Dict[str, DatabaseConfig]):
        self.configs = configs
        self.connections = {}
        self.connection_pools = {}
        self.metrics = {db_name: ConnectionMetrics() for db_name in configs.keys()}
        self.health_check_tasks = {}
        self.circuit_breakers = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.Lock()
        
    async def initialize(self) -> Dict[str, bool]:
        """Initialize all database connections"""
        results = {}
        
        for db_name, config in self.configs.items():
            try:
                success = await self._initialize_database(db_name, config)
                results[db_name] = success
                
                if success:
                    # Start health monitoring
                    self.health_check_tasks[db_name] = asyncio.create_task(
                        self._health_check_loop(db_name, config)
                    )
                    logger.info(f"Successfully initialized {db_name} database")
                else:
                    logger.error(f"Failed to initialize {db_name} database")
                    
            except Exception as e:
                logger.error(f"Error initializing {db_name}: {e}")
                results[db_name] = False
                
        return results
    
    async def _initialize_database(self, db_name: str, config: DatabaseConfig) -> bool:
        """Initialize a specific database connection"""
        try:
            if config.db_type == DatabaseType.POSTGRESQL:
                return await self._init_postgresql(db_name, config)
            elif config.db_type == DatabaseType.INFLUXDB:
                return await self._init_influxdb(db_name, config)
            elif config.db_type == DatabaseType.REDIS:
                return await self._init_redis(db_name, config)
            elif config.db_type == DatabaseType.NEO4J:
                return await self._init_neo4j(db_name, config)
            elif config.db_type == DatabaseType.TIMESCALEDB:
                return await self._init_timescaledb(db_name, config)
            else:
                logger.error(f"Unsupported database type: {config.db_type}")
                return False
                
        except Exception as e:
            logger.error(f"Database initialization error for {db_name}: {e}")
            return False
    
    async def _init_postgresql(self, db_name: str, config: DatabaseConfig) -> bool:
        """Initialize PostgreSQL+PostGIS connection"""
        try:
            # Create connection string
            conn_str = f"postgresql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
            
            if config.ssl_enabled:
                conn_str += "?sslmode=require"
            
            # Create async engine with connection pooling
            engine = create_async_engine(
                conn_str,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
                pool_timeout=config.connection_timeout,
                pool_recycle=3600,  # Recycle connections every hour
                echo=False
            )
            
            # Test connection
            async with engine.begin() as conn:
                result = await conn.execute("SELECT version()")
                version = result.fetchone()[0]
                logger.info(f"PostgreSQL version: {version}")
                
                # Enable PostGIS if available
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology")
                    logger.info("PostGIS extensions enabled")
                except Exception as e:
                    logger.warning(f"PostGIS extension warning: {e}")
            
            self.connections[db_name] = engine
            return True
            
        except Exception as e:
            logger.error(f"PostgreSQL initialization error: {e}")
            return False
    
    async def _init_influxdb(self, db_name: str, config: DatabaseConfig) -> bool:
        """Initialize InfluxDB connection"""
        try:
            # Get InfluxDB specific configuration
            token = os.getenv('INFLUXDB_TOKEN', '')
            org = os.getenv('INFLUXDB_ORG', 'prometheus')
            bucket = os.getenv('INFLUXDB_BUCKET', 'geological_data')
            
            url = f"http{'s' if config.ssl_enabled else ''}://{config.host}:{config.port}"
            
            client = InfluxDBClient(
                url=url,
                token=token,
                org=org,
                timeout=config.connection_timeout * 1000  # Convert to milliseconds
            )
            
            # Test connection
            health = client.health()
            if health.status == "pass":
                logger.info(f"InfluxDB connection successful: {health.message}")
                self.connections[db_name] = {
                    'client': client,
                    'org': org,
                    'bucket': bucket,
                    'write_api': client.write_api(write_options=SYNCHRONOUS)
                }
                return True
            else:
                logger.error(f"InfluxDB health check failed: {health.message}")
                return False
                
        except Exception as e:
            logger.error(f"InfluxDB initialization error: {e}")
            return False
    
    async def _init_redis(self, db_name: str, config: DatabaseConfig) -> bool:
        """Initialize Redis connection"""
        try:
            redis_url = f"redis://{config.host}:{config.port}"
            if config.password:
                redis_url = f"redis://:{config.password}@{config.host}:{config.port}"
            
            redis_client = await aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=config.pool_size,
                socket_timeout=config.connection_timeout,
                socket_connect_timeout=config.connection_timeout
            )
            
            # Test connection
            await redis_client.ping()
            logger.info("Redis connection successful")
            
            self.connections[db_name] = redis_client
            return True
            
        except Exception as e:
            logger.error(f"Redis initialization error: {e}")
            return False
    
    async def _init_neo4j(self, db_name: str, config: DatabaseConfig) -> bool:
        """Initialize Neo4j connection"""
        try:
            uri = f"bolt{'s' if config.ssl_enabled else ''}://{config.host}:{config.port}"
            
            driver = AsyncGraphDatabase.driver(
                uri,
                auth=(config.username, config.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=config.pool_size,
                connection_timeout=config.connection_timeout
            )
            
            # Test connection
            async with driver.session() as session:
                result = await session.run("RETURN 'Connection successful' as message")
                record = await result.single()
                logger.info(f"Neo4j: {record['message']}")
            
            self.connections[db_name] = driver
            return True
            
        except Exception as e:
            logger.error(f"Neo4j initialization error: {e}")
            return False
    
    async def _init_timescaledb(self, db_name: str, config: DatabaseConfig) -> bool:
        """Initialize TimescaleDB connection (PostgreSQL with TimescaleDB extension)"""
        try:
            # Similar to PostgreSQL but with TimescaleDB extensions
            success = await self._init_postgresql(db_name, config)
            
            if success:
                # Enable TimescaleDB extension
                engine = self.connections[db_name]
                async with engine.begin() as conn:
                    try:
                        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                        logger.info("TimescaleDB extension enabled")
                    except Exception as e:
                        logger.warning(f"TimescaleDB extension warning: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"TimescaleDB initialization error: {e}")
            return False
    
    async def _health_check_loop(self, db_name: str, config: DatabaseConfig):
        """Continuous health monitoring for database connections"""
        while True:
            try:
                await asyncio.sleep(config.health_check_interval)
                
                start_time = time.time()
                is_healthy = await self._perform_health_check(db_name, config)
                response_time = time.time() - start_time
                
                # Update metrics
                with self._lock:
                    metrics = self.metrics[db_name]
                    metrics.last_health_check = datetime.now()
                    metrics.average_response_time = (
                        metrics.average_response_time * 0.9 + response_time * 0.1
                    )
                    
                    if is_healthy:
                        metrics.uptime_percentage = min(100.0, metrics.uptime_percentage + 0.1)
                    else:
                        metrics.uptime_percentage = max(0.0, metrics.uptime_percentage - 1.0)
                        metrics.failed_connections += 1
                
                if not is_healthy:
                    logger.warning(f"Health check failed for {db_name}")
                    await self._attempt_reconnection(db_name, config)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for {db_name}: {e}")
    
    async def _perform_health_check(self, db_name: str, config: DatabaseConfig) -> bool:
        """Perform health check for specific database"""
        try:
            if config.db_type == DatabaseType.POSTGRESQL or config.db_type == DatabaseType.TIMESCALEDB:
                engine = self.connections.get(db_name)
                if engine:
                    async with engine.begin() as conn:
                        await conn.execute("SELECT 1")
                    return True
                    
            elif config.db_type == DatabaseType.INFLUXDB:
                influx_conn = self.connections.get(db_name)
                if influx_conn:
                    health = influx_conn['client'].health()
                    return health.status == "pass"
                    
            elif config.db_type == DatabaseType.REDIS:
                redis_client = self.connections.get(db_name)
                if redis_client:
                    await redis_client.ping()
                    return True
                    
            elif config.db_type == DatabaseType.NEO4J:
                driver = self.connections.get(db_name)
                if driver:
                    async with driver.session() as session:
                        await session.run("RETURN 1")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Health check failed for {db_name}: {e}")
            return False
    
    async def _attempt_reconnection(self, db_name: str, config: DatabaseConfig):
        """Attempt to reconnect to failed database"""
        logger.info(f"Attempting reconnection to {db_name}")
        
        for attempt in range(config.retry_attempts):
            try:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                success = await self._initialize_database(db_name, config)
                if success:
                    logger.info(f"Successfully reconnected to {db_name}")
                    return
                    
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed for {db_name}: {e}")
        
        logger.error(f"All reconnection attempts failed for {db_name}")
    
    async def execute_query(self, db_name: str, query: str, params: Optional[Dict] = None) -> Any:
        """Execute query on specified database"""
        try:
            config = self.configs[db_name]
            connection = self.connections.get(db_name)
            
            if not connection:
                raise Exception(f"No connection available for {db_name}")
            
            if config.db_type == DatabaseType.POSTGRESQL or config.db_type == DatabaseType.TIMESCALEDB:
                async with connection.begin() as conn:
                    result = await conn.execute(query, params or {})
                    return result.fetchall()
                    
            elif config.db_type == DatabaseType.NEO4J:
                async with connection.session() as session:
                    result = await session.run(query, params or {})
                    return [record.data() for record in await result.data()]
                    
            else:
                raise Exception(f"Query execution not supported for {config.db_type}")
                
        except Exception as e:
            logger.error(f"Query execution error on {db_name}: {e}")
            raise
    
    async def write_time_series(self, db_name: str, measurement: str, tags: Dict, fields: Dict, timestamp: Optional[datetime] = None) -> bool:
        """Write time-series data to InfluxDB or TimescaleDB"""
        try:
            config = self.configs[db_name]
            connection = self.connections.get(db_name)
            
            if not connection:
                raise Exception(f"No connection available for {db_name}")
            
            if config.db_type == DatabaseType.INFLUXDB:
                point = Point(measurement)
                
                for tag_key, tag_value in tags.items():
                    point = point.tag(tag_key, tag_value)
                
                for field_key, field_value in fields.items():
                    point = point.field(field_key, field_value)
                
                if timestamp:
                    point = point.time(timestamp)
                
                write_api = connection['write_api']
                write_api.write(bucket=connection['bucket'], record=point)
                return True
                
            elif config.db_type == DatabaseType.TIMESCALEDB:
                # Implement TimescaleDB time-series insertion
                columns = list(tags.keys()) + list(fields.keys()) + ['timestamp']
                values = list(tags.values()) + list(fields.values()) + [timestamp or datetime.now()]
                
                placeholders = ', '.join(['$' + str(i+1) for i in range(len(values))])
                column_names = ', '.join(columns)
                
                query = f"INSERT INTO {measurement} ({column_names}) VALUES ({placeholders})"
                
                async with connection.begin() as conn:
                    await conn.execute(query, values)
                return True
                
            else:
                raise Exception(f"Time-series writing not supported for {config.db_type}")
                
        except Exception as e:
            logger.error(f"Time-series write error on {db_name}: {e}")
            return False
    
    async def cache_set(self, db_name: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cache value in Redis"""
        try:
            config = self.configs[db_name]
            
            if config.db_type != DatabaseType.REDIS:
                raise Exception(f"Caching only supported for Redis, not {config.db_type}")
            
            redis_client = self.connections.get(db_name)
            if not redis_client:
                raise Exception(f"No Redis connection available for {db_name}")
            
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            
            if ttl:
                await redis_client.setex(key, ttl, serialized_value)
            else:
                await redis_client.set(key, serialized_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error on {db_name}: {e}")
            return False
    
    async def cache_get(self, db_name: str, key: str) -> Optional[Any]:
        """Get cache value from Redis"""
        try:
            config = self.configs[db_name]
            
            if config.db_type != DatabaseType.REDIS:
                raise Exception(f"Caching only supported for Redis, not {config.db_type}")
            
            redis_client = self.connections.get(db_name)
            if not redis_client:
                return None
            
            value = await redis_client.get(key)
            if value is None:
                return None
            
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
                
        except Exception as e:
            logger.error(f"Cache get error on {db_name}: {e}")
            return None
    
    def get_metrics(self, db_name: Optional[str] = None) -> Union[ConnectionMetrics, Dict[str, ConnectionMetrics]]:
        """Get connection metrics"""
        if db_name:
            return self.metrics.get(db_name)
        return self.metrics.copy()
    
    async def close_all_connections(self):
        """Close all database connections"""
        # Cancel health check tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Close connections
        for db_name, connection in self.connections.items():
            try:
                config = self.configs[db_name]
                
                if config.db_type == DatabaseType.POSTGRESQL or config.db_type == DatabaseType.TIMESCALEDB:
                    await connection.dispose()
                elif config.db_type == DatabaseType.INFLUXDB:
                    connection['client'].close()
                elif config.db_type == DatabaseType.REDIS:
                    await connection.close()
                elif config.db_type == DatabaseType.NEO4J:
                    await connection.close()
                    
                logger.info(f"Closed connection to {db_name}")
                
            except Exception as e:
                logger.error(f"Error closing connection to {db_name}: {e}")
        
        self.executor.shutdown(wait=True)

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

async def get_database_manager() -> DatabaseManager:
    """Get or create global database manager instance"""
    global _db_manager
    
    if _db_manager is None:
        # Load configuration from environment
        configs = {
            'postgresql': DatabaseConfig(
                db_type=DatabaseType.POSTGRESQL,
                port=int(os.getenv('POSTGRES_PORT', '5432'))
            ),
            'influxdb': DatabaseConfig(
                db_type=DatabaseType.INFLUXDB,
                port=int(os.getenv('INFLUXDB_PORT', '8086'))
            ),
            'redis': DatabaseConfig(
                db_type=DatabaseType.REDIS,
                port=int(os.getenv('REDIS_PORT', '6379'))
            ),
            'neo4j': DatabaseConfig(
                db_type=DatabaseType.NEO4J,
                port=int(os.getenv('NEO4J_PORT', '7687'))
            ),
            'timescaledb': DatabaseConfig(
                db_type=DatabaseType.TIMESCALEDB,
                port=int(os.getenv('TIMESCALEDB_PORT', '5433'))
            )
        }
        
        _db_manager = DatabaseManager(configs)
        await _db_manager.initialize()
    
    return _db_manager

@asynccontextmanager
async def database_transaction(db_name: str):
    """Context manager for database transactions"""
    db_manager = await get_database_manager()
    config = db_manager.configs[db_name]
    connection = db_manager.connections.get(db_name)
    
    if not connection:
        raise Exception(f"No connection available for {db_name}")
    
    if config.db_type == DatabaseType.POSTGRESQL or config.db_type == DatabaseType.TIMESCALEDB:
        async with connection.begin() as trans:
            try:
                yield trans
                await trans.commit()
            except Exception:
                await trans.rollback()
                raise
    else:
        # For databases that don't support transactions, just yield the connection
        yield connection