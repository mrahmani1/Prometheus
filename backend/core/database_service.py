"""
Enhanced Database Service with Standardized Configuration Management
Provides centralized database operations with automatic configuration discovery,
connection pooling, health monitoring, and production-grade features.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, AsyncIterator, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path

# Database clients
import asyncpg
import redis.asyncio as redis
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from neo4j import AsyncGraphDatabase
import motor.motor_asyncio

from backend.core.database_config_manager import (
    DatabaseConfigManager, DatabaseConfig, DatabaseType, Environment,
    ConnectionPoolConfig, DatabaseCredentials
)


@dataclass
class ConnectionMetrics:
    """Connection pool metrics"""
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    failed_connections: int = 0
    connection_errors: int = 0
    avg_connection_time: float = 0.0
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"


@dataclass
class QueryMetrics:
    """Query execution metrics"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_query_time: float = 0.0
    slow_queries: int = 0
    last_query_time: Optional[datetime] = None


class DatabaseConnectionPool:
    """
    Enhanced database connection pool with health monitoring,
    automatic failover, and performance metrics.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + f".ConnectionPool.{config.name}")
        
        # Connection pool
        self.pool = None
        self.is_initialized = False
        self.is_healthy = False
        
        # Metrics
        self.connection_metrics = ConnectionMetrics()
        self.query_metrics = QueryMetrics()
        
        # Health monitoring
        self.last_health_check = None
        self.health_check_interval = timedelta(seconds=config.pool_config.health_check_interval)
        
        # Failover and retry
        self.retry_count = 0
        self.max_retries = config.pool_config.retry_attempts
        self.retry_delay = config.pool_config.retry_delay
        
    async def initialize(self) -> bool:
        """Initialize connection pool"""
        try:
            self.logger.info(f"Initializing connection pool for {self.config.name}")
            
            if self.config.type == DatabaseType.POSTGRESQL:
                await self._initialize_postgresql_pool()
            elif self.config.type == DatabaseType.REDIS:
                await self._initialize_redis_pool()
            elif self.config.type == DatabaseType.INFLUXDB:
                await self._initialize_influxdb_pool()
            elif self.config.type == DatabaseType.NEO4J:
                await self._initialize_neo4j_pool()
            elif self.config.type == DatabaseType.TIMESCALEDB:
                await self._initialize_timescaledb_pool()
            else:
                raise ValueError(f"Unsupported database type: {self.config.type}")
            
            self.is_initialized = True
            await self.health_check()
            
            self.logger.info(f"Connection pool initialized successfully for {self.config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool for {self.config.name}: {e}")
            self.is_initialized = False
            return False
    
    async def _initialize_postgresql_pool(self):
        """Initialize PostgreSQL connection pool"""
        connection_kwargs = {
            'host': self.config.host,
            'port': self.config.port,
            'database': self.config.database,
            'min_size': self.config.pool_config.min_connections,
            'max_size': self.config.pool_config.max_connections,
            'command_timeout': self.config.timeout,
        }
        
        if self.config.credentials:
            connection_kwargs.update({
                'user': self.config.credentials.username,
                'password': self.config.credentials.password,
            })
        
        if self.config.ssl_enabled:
            ssl_context = self._create_ssl_context()
            connection_kwargs['ssl'] = ssl_context
        
        self.pool = await asyncpg.create_pool(**connection_kwargs)
    
    async def _initialize_redis_pool(self):
        """Initialize Redis connection pool"""
        connection_kwargs = {
            'host': self.config.host,
            'port': self.config.port,
            'db': int(self.config.database) if self.config.database else 0,
            'max_connections': self.config.pool_config.max_connections,
            'socket_timeout': self.config.timeout,
            'socket_connect_timeout': self.config.timeout,
            'retry_on_timeout': True,
            'health_check_interval': self.config.pool_config.health_check_interval,
        }
        
        if self.config.credentials and self.config.credentials.password:
            connection_kwargs['password'] = self.config.credentials.password
        
        if self.config.ssl_enabled:
            connection_kwargs['ssl'] = True
            if self.config.ssl_ca_path:
                connection_kwargs['ssl_ca_certs'] = self.config.ssl_ca_path
        
        self.pool = redis.ConnectionPool(**connection_kwargs)
    
    async def _initialize_influxdb_pool(self):
        """Initialize InfluxDB connection pool"""
        url = self.config.get_connection_string(include_credentials=False)
        token = self.config.credentials.token if self.config.credentials else None
        
        self.pool = InfluxDBClientAsync(
            url=url,
            token=token,
            timeout=self.config.timeout * 1000,  # Convert to milliseconds
            enable_gzip=True
        )
    
    async def _initialize_neo4j_pool(self):
        """Initialize Neo4j connection pool"""
        uri = self.config.get_connection_string(include_credentials=False)
        
        auth = None
        if self.config.credentials:
            auth = (self.config.credentials.username, self.config.credentials.password)
        
        self.pool = AsyncGraphDatabase.driver(
            uri,
            auth=auth,
            max_connection_lifetime=self.config.pool_config.pool_recycle,
            max_connection_pool_size=self.config.pool_config.max_connections,
            connection_timeout=self.config.timeout,
            encrypted=self.config.ssl_enabled
        )
    
    async def _initialize_timescaledb_pool(self):
        """Initialize TimescaleDB connection pool (same as PostgreSQL)"""
        await self._initialize_postgresql_pool()
    
    def _create_ssl_context(self):
        """Create SSL context for secure connections"""
        import ssl
        
        context = ssl.create_default_context()
        
        if not self.config.ssl_verify:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        
        if self.config.ssl_ca_path:
            context.load_verify_locations(self.config.ssl_ca_path)
        
        if self.config.ssl_cert_path and self.config.ssl_key_path:
            context.load_cert_chain(self.config.ssl_cert_path, self.config.ssl_key_path)
        
        return context
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool with automatic cleanup"""
        connection = None
        start_time = datetime.utcnow()
        
        try:
            if not self.is_initialized:
                raise RuntimeError(f"Connection pool not initialized for {self.config.name}")
            
            if self.config.type == DatabaseType.POSTGRESQL or self.config.type == DatabaseType.TIMESCALEDB:
                connection = await self.pool.acquire()
            elif self.config.type == DatabaseType.REDIS:
                connection = redis.Redis(connection_pool=self.pool)
            elif self.config.type == DatabaseType.INFLUXDB:
                connection = self.pool
            elif self.config.type == DatabaseType.NEO4J:
                connection = self.pool.session()
            
            # Update metrics
            connection_time = (datetime.utcnow() - start_time).total_seconds()
            self.connection_metrics.active_connections += 1
            self.connection_metrics.total_connections += 1
            self.connection_metrics.avg_connection_time = (
                (self.connection_metrics.avg_connection_time * (self.connection_metrics.total_connections - 1) + connection_time) /
                self.connection_metrics.total_connections
            )
            
            yield connection
            
        except Exception as e:
            self.connection_metrics.connection_errors += 1
            self.connection_metrics.failed_connections += 1
            self.logger.error(f"Connection error for {self.config.name}: {e}")
            raise
            
        finally:
            if connection:
                try:
                    if self.config.type == DatabaseType.POSTGRESQL or self.config.type == DatabaseType.TIMESCALEDB:
                        await self.pool.release(connection)
                    elif self.config.type == DatabaseType.REDIS:
                        await connection.close()
                    elif self.config.type == DatabaseType.NEO4J:
                        await connection.close()
                    # InfluxDB connections are managed by the client
                    
                    self.connection_metrics.active_connections -= 1
                    
                except Exception as e:
                    self.logger.error(f"Error releasing connection for {self.config.name}: {e}")
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute query with metrics tracking"""
        start_time = datetime.utcnow()
        
        try:
            async with self.get_connection() as conn:
                if self.config.type == DatabaseType.POSTGRESQL or self.config.type == DatabaseType.TIMESCALEDB:
                    if params:
                        result = await conn.fetch(query, *params.values())
                    else:
                        result = await conn.fetch(query)
                elif self.config.type == DatabaseType.REDIS:
                    # Redis commands are different
                    result = await conn.execute_command(*query.split())
                elif self.config.type == DatabaseType.NEO4J:
                    result = await conn.run(query, params or {})
                    result = await result.data()
                else:
                    raise ValueError(f"Query execution not implemented for {self.config.type}")
                
                # Update metrics
                query_time = (datetime.utcnow() - start_time).total_seconds()
                self.query_metrics.total_queries += 1
                self.query_metrics.successful_queries += 1
                self.query_metrics.avg_query_time = (
                    (self.query_metrics.avg_query_time * (self.query_metrics.total_queries - 1) + query_time) /
                    self.query_metrics.total_queries
                )
                self.query_metrics.last_query_time = datetime.utcnow()
                
                # Track slow queries
                if query_time > 1.0:  # Queries taking more than 1 second
                    self.query_metrics.slow_queries += 1
                
                return result
                
        except Exception as e:
            self.query_metrics.total_queries += 1
            self.query_metrics.failed_queries += 1
            self.logger.error(f"Query execution error for {self.config.name}: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Perform health check on the connection pool"""
        try:
            if self.config.type == DatabaseType.POSTGRESQL or self.config.type == DatabaseType.TIMESCALEDB:
                result = await self.execute_query("SELECT 1")
                self.is_healthy = len(result) > 0
            elif self.config.type == DatabaseType.REDIS:
                async with self.get_connection() as conn:
                    await conn.ping()
                self.is_healthy = True
            elif self.config.type == DatabaseType.INFLUXDB:
                health = await self.pool.health()
                self.is_healthy = health.status == 'pass'
            elif self.config.type == DatabaseType.NEO4J:
                async with self.get_connection() as session:
                    result = await session.run("RETURN 1")
                    await result.consume()
                self.is_healthy = True
            
            self.connection_metrics.last_health_check = datetime.utcnow()
            self.connection_metrics.health_status = "healthy" if self.is_healthy else "unhealthy"
            
            if self.is_healthy:
                self.retry_count = 0
            
            return self.is_healthy
            
        except Exception as e:
            self.is_healthy = False
            self.connection_metrics.health_status = f"unhealthy: {str(e)}"
            self.logger.error(f"Health check failed for {self.config.name}: {e}")
            return False
    
    async def close(self):
        """Close connection pool"""
        try:
            if self.pool:
                if self.config.type == DatabaseType.POSTGRESQL or self.config.type == DatabaseType.TIMESCALEDB:
                    await self.pool.close()
                elif self.config.type == DatabaseType.REDIS:
                    await self.pool.disconnect()
                elif self.config.type == DatabaseType.INFLUXDB:
                    await self.pool.close()
                elif self.config.type == DatabaseType.NEO4J:
                    await self.pool.close()
                
                self.is_initialized = False
                self.is_healthy = False
                self.logger.info(f"Connection pool closed for {self.config.name}")
                
        except Exception as e:
            self.logger.error(f"Error closing connection pool for {self.config.name}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get connection and query metrics"""
        return {
            'connection_metrics': {
                'active_connections': self.connection_metrics.active_connections,
                'idle_connections': self.connection_metrics.idle_connections,
                'total_connections': self.connection_metrics.total_connections,
                'failed_connections': self.connection_metrics.failed_connections,
                'connection_errors': self.connection_metrics.connection_errors,
                'avg_connection_time': self.connection_metrics.avg_connection_time,
                'last_health_check': self.connection_metrics.last_health_check.isoformat() if self.connection_metrics.last_health_check else None,
                'health_status': self.connection_metrics.health_status,
            },
            'query_metrics': {
                'total_queries': self.query_metrics.total_queries,
                'successful_queries': self.query_metrics.successful_queries,
                'failed_queries': self.query_metrics.failed_queries,
                'avg_query_time': self.query_metrics.avg_query_time,
                'slow_queries': self.query_metrics.slow_queries,
                'last_query_time': self.query_metrics.last_query_time.isoformat() if self.query_metrics.last_query_time else None,
            },
            'pool_status': {
                'is_initialized': self.is_initialized,
                'is_healthy': self.is_healthy,
                'retry_count': self.retry_count,
                'max_retries': self.max_retries,
            }
        }


class DatabaseService:
    """
    Centralized Database Service with standardized configuration management,
    automatic service discovery, connection pooling, and health monitoring.
    """
    
    def __init__(self, 
                 config_manager: Optional[DatabaseConfigManager] = None,
                 environment: Environment = Environment.DEVELOPMENT):
        self.config_manager = config_manager
        self.environment = environment
        self.logger = logging.getLogger(__name__ + ".DatabaseService")
        
        # Connection pools
        self.pools: Dict[str, DatabaseConnectionPool] = {}
        
        # Service status
        self.is_initialized = False
        self.health_check_task = None
        
        # Metrics aggregation
        self.service_metrics = {
            'total_databases': 0,
            'healthy_databases': 0,
            'unhealthy_databases': 0,
            'total_connections': 0,
            'total_queries': 0,
            'last_health_check': None
        }
    
    async def initialize(self) -> bool:
        """Initialize database service"""
        try:
            self.logger.info("Initializing Database Service")
            
            # Initialize config manager if not provided
            if not self.config_manager:
                from backend.core.database_config_manager import create_config_manager
                self.config_manager = await create_config_manager(self.environment.value)
            
            # Load database configurations
            configs = self.config_manager.list_database_configs()
            
            # Initialize connection pools
            for config in configs:
                pool = DatabaseConnectionPool(config)
                success = await pool.initialize()
                
                if success:
                    self.pools[config.name] = pool
                    self.logger.info(f"Initialized connection pool for {config.name}")
                else:
                    self.logger.error(f"Failed to initialize connection pool for {config.name}")
            
            self.service_metrics['total_databases'] = len(self.pools)
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            self.is_initialized = True
            self.logger.info(f"Database Service initialized with {len(self.pools)} databases")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Database Service initialization failed: {e}")
            return False
    
    async def _start_health_monitoring(self):
        """Start background health monitoring task"""
        async def health_monitor():
            while self.is_initialized:
                try:
                    await self.perform_health_checks()
                    await asyncio.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(60)
        
        self.health_check_task = asyncio.create_task(health_monitor())
    
    async def perform_health_checks(self):
        """Perform health checks on all database connections"""
        healthy_count = 0
        unhealthy_count = 0
        
        for name, pool in self.pools.items():
            try:
                is_healthy = await pool.health_check()
                if is_healthy:
                    healthy_count += 1
                else:
                    unhealthy_count += 1
                    self.logger.warning(f"Database {name} is unhealthy")
                    
            except Exception as e:
                unhealthy_count += 1
                self.logger.error(f"Health check failed for {name}: {e}")
        
        self.service_metrics.update({
            'healthy_databases': healthy_count,
            'unhealthy_databases': unhealthy_count,
            'last_health_check': datetime.utcnow().isoformat()
        })
    
    def get_database_pool(self, name: str) -> Optional[DatabaseConnectionPool]:
        """Get database connection pool by name"""
        return self.pools.get(name)
    
    def get_database_pools_by_type(self, db_type: DatabaseType) -> List[DatabaseConnectionPool]:
        """Get database connection pools by type"""
        return [
            pool for pool in self.pools.values()
            if pool.config.type == db_type
        ]
    
    async def execute_query(self, database_name: str, query: str, params: Optional[Dict] = None) -> Any:
        """Execute query on specified database"""
        pool = self.get_database_pool(database_name)
        if not pool:
            raise ValueError(f"Database {database_name} not found")
        
        return await pool.execute_query(query, params)
    
    @asynccontextmanager
    async def get_connection(self, database_name: str):
        """Get connection from specified database pool"""
        pool = self.get_database_pool(database_name)
        if not pool:
            raise ValueError(f"Database {database_name} not found")
        
        async with pool.get_connection() as conn:
            yield conn
    
    async def add_database(self, config: DatabaseConfig) -> bool:
        """Add new database configuration and initialize connection pool"""
        try:
            # Add to config manager
            await self.config_manager.create_database_config(
                name=config.name,
                db_type=config.type,
                host=config.host,
                port=config.port,
                database=config.database
            )
            
            # Initialize connection pool
            pool = DatabaseConnectionPool(config)
            success = await pool.initialize()
            
            if success:
                self.pools[config.name] = pool
                self.service_metrics['total_databases'] = len(self.pools)
                self.logger.info(f"Added database {config.name}")
                return True
            else:
                self.logger.error(f"Failed to initialize pool for {config.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding database {config.name}: {e}")
            return False
    
    async def remove_database(self, name: str) -> bool:
        """Remove database configuration and close connection pool"""
        try:
            if name in self.pools:
                await self.pools[name].close()
                del self.pools[name]
                self.service_metrics['total_databases'] = len(self.pools)
                self.logger.info(f"Removed database {name}")
                return True
            else:
                self.logger.warning(f"Database {name} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing database {name}: {e}")
            return False
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics"""
        # Aggregate metrics from all pools
        total_connections = sum(pool.connection_metrics.total_connections for pool in self.pools.values())
        total_queries = sum(pool.query_metrics.total_queries for pool in self.pools.values())
        
        self.service_metrics.update({
            'total_connections': total_connections,
            'total_queries': total_queries
        })
        
        # Include individual pool metrics
        pool_metrics = {}
        for name, pool in self.pools.items():
            pool_metrics[name] = pool.get_metrics()
        
        return {
            'service_metrics': self.service_metrics,
            'pool_metrics': pool_metrics,
            'service_status': {
                'is_initialized': self.is_initialized,
                'environment': self.environment.value,
                'total_pools': len(self.pools)
            }
        }
    
    async def get_database_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a database"""
        pool = self.get_database_pool(name)
        if not pool:
            return None
        
        config = pool.config
        metrics = pool.get_metrics()
        
        return {
            'name': config.name,
            'type': config.type.value,
            'host': config.host,
            'port': config.port,
            'database': config.database,
            'ssl_enabled': config.ssl_enabled,
            'environment': config.environment.value,
            'tags': config.tags,
            'metadata': config.metadata,
            'connection_string': config.get_connection_string(include_credentials=False),
            'metrics': metrics,
            'created_at': config.created_at.isoformat(),
            'updated_at': config.updated_at.isoformat(),
            'version': config.version
        }
    
    async def list_databases(self) -> List[Dict[str, Any]]:
        """List all configured databases with their information"""
        databases = []
        for name in self.pools.keys():
            db_info = await self.get_database_info(name)
            if db_info:
                databases.append(db_info)
        return databases
    
    async def test_all_connections(self) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Test all database connections"""
        results = {}
        for name, pool in self.pools.items():
            try:
                is_healthy = await pool.health_check()
                results[name] = (is_healthy, None)
            except Exception as e:
                results[name] = (False, str(e))
        return results
    
    async def cleanup(self):
        """Cleanup database service resources"""
        self.logger.info("Cleaning up Database Service")
        
        # Stop health monitoring
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connection pools
        for name, pool in self.pools.items():
            try:
                await pool.close()
                self.logger.info(f"Closed connection pool for {name}")
            except Exception as e:
                self.logger.error(f"Error closing pool for {name}: {e}")
        
        self.pools.clear()
        self.is_initialized = False
        
        # Cleanup config manager
        if self.config_manager:
            await self.config_manager.cleanup()
        
        self.logger.info("Database Service cleanup complete")


# Global database service instance
_database_service: Optional[DatabaseService] = None


async def get_database_service(environment: str = None) -> DatabaseService:
    """Get or create global database service instance"""
    global _database_service
    
    if _database_service is None or not _database_service.is_initialized:
        env = Environment(environment or os.getenv('ENVIRONMENT', 'development'))
        _database_service = DatabaseService(environment=env)
        await _database_service.initialize()
    
    return _database_service


async def cleanup_database_service():
    """Cleanup global database service instance"""
    global _database_service
    
    if _database_service:
        await _database_service.cleanup()
        _database_service = None


# Convenience functions for common database operations
async def get_postgres_connection(database_name: str = None):
    """Get PostgreSQL connection"""
    service = await get_database_service()
    
    if database_name:
        return service.get_connection(database_name)
    else:
        # Get first PostgreSQL database
        pools = service.get_database_pools_by_type(DatabaseType.POSTGRESQL)
        if pools:
            return service.get_connection(pools[0].config.name)
        else:
            raise ValueError("No PostgreSQL databases configured")


async def get_redis_connection(database_name: str = None):
    """Get Redis connection"""
    service = await get_database_service()
    
    if database_name:
        return service.get_connection(database_name)
    else:
        # Get first Redis database
        pools = service.get_database_pools_by_type(DatabaseType.REDIS)
        if pools:
            return service.get_connection(pools[0].config.name)
        else:
            raise ValueError("No Redis databases configured")


async def get_influxdb_connection(database_name: str = None):
    """Get InfluxDB connection"""
    service = await get_database_service()
    
    if database_name:
        return service.get_connection(database_name)
    else:
        # Get first InfluxDB database
        pools = service.get_database_pools_by_type(DatabaseType.INFLUXDB)
        if pools:
            return service.get_connection(pools[0].config.name)
        else:
            raise ValueError("No InfluxDB databases configured")


async def get_neo4j_connection(database_name: str = None):
    """Get Neo4j connection"""
    service = await get_database_service()
    
    if database_name:
        return service.get_connection(database_name)
    else:
        # Get first Neo4j database
        pools = service.get_database_pools_by_type(DatabaseType.NEO4J)
        if pools:
            return service.get_connection(pools[0].config.name)
        else:
            raise ValueError("No Neo4j databases configured")