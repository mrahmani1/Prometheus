"""
Database Configuration Service
Integrates the DatabaseConfigManager with the existing database integration layer
Provides centralized database access with standardized configuration management
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Type
from contextlib import asynccontextmanager
from datetime import datetime

# Import existing database integration
from backend.core.database_integration import EnhancedDatabaseIntegrator, QueryResult
from backend.core.database_config_manager import (
    DatabaseConfigManager, DatabaseConfig, DatabaseType, Environment,
    create_config_manager, validate_db_config
)

# Import database clients
try:
    import asyncpg
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False


class DatabaseConnectionPool:
    """Manages database connection pools with configuration-based setup"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + f".ConnectionPool.{config.name}")
        self.connection_pool = None
        self.is_initialized = False
        self.health_status = "unknown"
        self.last_health_check = None
        
    async def initialize(self) -> bool:
        """Initialize connection pool based on database type"""
        try:
            if self.config.type == DatabaseType.POSTGRESQL:
                await self._initialize_postgresql_pool()
            elif self.config.type == DatabaseType.INFLUXDB:
                await self._initialize_influxdb_pool()
            elif self.config.type == DatabaseType.NEO4J:
                await self._initialize_neo4j_pool()
            elif self.config.type == DatabaseType.REDIS:
                await self._initialize_redis_pool()
            elif self.config.type == DatabaseType.MONGODB:
                await self._initialize_mongodb_pool()
            else:
                raise ValueError(f"Unsupported database type: {self.config.type}")
            
            self.is_initialized = True
            self.health_status = "healthy"
            self.logger.info(f"Connection pool initialized for {self.config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool for {self.config.name}: {e}")
            self.health_status = "error"
            return False
    
    async def _initialize_postgresql_pool(self):
        """Initialize PostgreSQL connection pool"""
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("PostgreSQL dependencies not available")
        
        # Create SQLAlchemy async engine
        connection_string = self.config.get_connection_string()
        
        # Add SSL parameters if enabled
        connect_args = {}
        if self.config.ssl_enabled:
            connect_args['ssl'] = 'require' if self.config.ssl_verify else 'prefer'
            if self.config.ssl_ca_path:
                connect_args['sslrootcert'] = self.config.ssl_ca_path
            if self.config.ssl_cert_path:
                connect_args['sslcert'] = self.config.ssl_cert_path
            if self.config.ssl_key_path:
                connect_args['sslkey'] = self.config.ssl_key_path
        
        self.connection_pool = create_async_engine(
            connection_string,
            pool_size=self.config.pool_config.min_connections,
            max_overflow=self.config.pool_config.max_overflow,
            pool_timeout=self.config.pool_config.pool_timeout,
            pool_recycle=self.config.pool_config.pool_recycle,
            pool_pre_ping=self.config.pool_config.pool_pre_ping,
            connect_args=connect_args,
            echo=False
        )
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            self.connection_pool,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def _initialize_influxdb_pool(self):
        """Initialize InfluxDB connection pool"""
        if not INFLUXDB_AVAILABLE:
            raise ImportError("InfluxDB dependencies not available")
        
        self.connection_pool = InfluxDBClientAsync(
            url=self.config.get_connection_string(include_credentials=False),
            token=self.config.credentials.token if self.config.credentials else None,
            org=self.config.metadata.get('org', 'prometheus'),
            timeout=self.config.timeout * 1000,
            verify_ssl=self.config.ssl_verify if self.config.ssl_enabled else True
        )
    
    async def _initialize_neo4j_pool(self):
        """Initialize Neo4j connection pool"""
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j dependencies not available")
        
        auth = None
        if self.config.credentials:
            auth = (self.config.credentials.username, self.config.credentials.password)
        
        self.connection_pool = AsyncGraphDatabase.driver(
            self.config.get_connection_string(include_credentials=False),
            auth=auth,
            max_connection_lifetime=self.config.pool_config.pool_recycle,
            max_connection_pool_size=self.config.pool_config.max_connections,
            connection_timeout=self.config.timeout,
            encrypted=self.config.ssl_enabled,
            trust=self.config.ssl_verify if self.config.ssl_enabled else True
        )
    
    async def _initialize_redis_pool(self):
        """Initialize Redis connection pool"""
        if not REDIS_AVAILABLE:
            raise ImportError("Redis dependencies not available")
        
        connection_kwargs = {
            'host': self.config.host,
            'port': self.config.port,
            'db': int(self.config.database) if self.config.database else 0,
            'password': self.config.credentials.password if self.config.credentials else None,
            'socket_timeout': self.config.timeout,
            'socket_connect_timeout': self.config.timeout,
            'decode_responses': True,
            'max_connections': self.config.pool_config.max_connections
        }
        
        if self.config.ssl_enabled:
            connection_kwargs['ssl'] = True
            connection_kwargs['ssl_check_hostname'] = self.config.ssl_verify
            if self.config.ssl_ca_path:
                connection_kwargs['ssl_ca_certs'] = self.config.ssl_ca_path
            if self.config.ssl_cert_path:
                connection_kwargs['ssl_certfile'] = self.config.ssl_cert_path
            if self.config.ssl_key_path:
                connection_kwargs['ssl_keyfile'] = self.config.ssl_key_path
        
        self.connection_pool = redis.ConnectionPool(**connection_kwargs)
    
    async def _initialize_mongodb_pool(self):
        """Initialize MongoDB connection pool"""
        if not MONGODB_AVAILABLE:
            raise ImportError("MongoDB dependencies not available")
        
        connection_string = self.config.get_connection_string()
        
        # Add connection options
        options = {
            'maxPoolSize': self.config.pool_config.max_connections,
            'minPoolSize': self.config.pool_config.min_connections,
            'maxIdleTimeMS': self.config.pool_config.pool_recycle * 1000,
            'connectTimeoutMS': self.config.timeout * 1000,
            'serverSelectionTimeoutMS': self.config.timeout * 1000
        }
        
        if self.config.ssl_enabled:
            options['ssl'] = True
            options['ssl_cert_reqs'] = 'CERT_REQUIRED' if self.config.ssl_verify else 'CERT_NONE'
            if self.config.ssl_ca_path:
                options['ssl_ca_certs'] = self.config.ssl_ca_path
            if self.config.ssl_cert_path:
                options['ssl_certfile'] = self.config.ssl_cert_path
            if self.config.ssl_key_path:
                options['ssl_keyfile'] = self.config.ssl_key_path
        
        self.connection_pool = AsyncIOMotorClient(connection_string, **options)
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.is_initialized:
            raise RuntimeError(f"Connection pool not initialized for {self.config.name}")
        
        if self.config.type == DatabaseType.POSTGRESQL:
            async with self.session_factory() as session:
                try:
                    yield session
                except Exception:
                    await session.rollback()
                    raise
                finally:
                    await session.close()
        
        elif self.config.type == DatabaseType.REDIS:
            client = redis.Redis(connection_pool=self.connection_pool)
            try:
                yield client
            finally:
                await client.close()
        
        elif self.config.type == DatabaseType.NEO4J:
            async with self.connection_pool.session() as session:
                yield session
        
        elif self.config.type == DatabaseType.INFLUXDB:
            yield self.connection_pool
        
        elif self.config.type == DatabaseType.MONGODB:
            yield self.connection_pool[self.config.database]
        
        else:
            yield self.connection_pool
    
    async def health_check(self) -> bool:
        """Perform health check on connection pool"""
        try:
            if self.config.type == DatabaseType.POSTGRESQL:
                async with self.get_connection() as conn:
                    result = await conn.execute("SELECT 1")
                    
            elif self.config.type == DatabaseType.REDIS:
                async with self.get_connection() as conn:
                    await conn.ping()
                    
            elif self.config.type == DatabaseType.NEO4J:
                async with self.get_connection() as conn:
                    await conn.run("RETURN 1")
                    
            elif self.config.type == DatabaseType.INFLUXDB:
                async with self.get_connection() as conn:
                    health = await conn.health()
                    if health.status != 'pass':
                        return False
                        
            elif self.config.type == DatabaseType.MONGODB:
                async with self.get_connection() as conn:
                    await conn.admin.command('ping')
            
            self.health_status = "healthy"
            self.last_health_check = datetime.utcnow()
            return True
            
        except Exception as e:
            self.logger.warning(f"Health check failed for {self.config.name}: {e}")
            self.health_status = "unhealthy"
            self.last_health_check = datetime.utcnow()
            return False
    
    async def cleanup(self):
        """Cleanup connection pool"""
        if self.connection_pool:
            if self.config.type == DatabaseType.POSTGRESQL:
                await self.connection_pool.dispose()
            elif self.config.type == DatabaseType.INFLUXDB:
                await self.connection_pool.close()
            elif self.config.type == DatabaseType.NEO4J:
                await self.connection_pool.close()
            elif self.config.type == DatabaseType.REDIS:
                await self.connection_pool.disconnect()
            elif self.config.type == DatabaseType.MONGODB:
                self.connection_pool.close()
        
        self.is_initialized = False
        self.logger.info(f"Connection pool cleaned up for {self.config.name}")


class DatabaseService:
    """
    Centralized Database Service
    Provides unified database access with configuration management integration
    """
    
    def __init__(self, environment: str = None):
        self.environment = Environment(environment or os.getenv('ENVIRONMENT', 'development'))
        self.logger = logging.getLogger(__name__ + ".DatabaseService")
        
        # Configuration manager
        self.config_manager: Optional[DatabaseConfigManager] = None
        
        # Connection pools
        self.connection_pools: Dict[str, DatabaseConnectionPool] = {}
        
        # Legacy integrator for backward compatibility
        self.legacy_integrator: Optional[EnhancedDatabaseIntegrator] = None
        
        # Service status
        self.is_initialized = False
        self.health_monitor_task = None
        
        self.logger.info(f"Database Service initialized for {self.environment.value} environment")
    
    async def initialize(self) -> bool:
        """Initialize database service"""
        try:
            self.logger.info("Initializing Database Service")
            
            # Initialize configuration manager
            self.config_manager = await create_config_manager(self.environment.value)
            
            # Initialize connection pools for all configured databases
            await self._initialize_connection_pools()
            
            # Initialize legacy integrator for backward compatibility
            await self._initialize_legacy_integrator()
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            self.is_initialized = True
            self.logger.info("Database Service initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Database Service initialization failed: {e}")
            return False
    
    async def _initialize_connection_pools(self):
        """Initialize connection pools for all configured databases"""
        configs = self.config_manager.list_database_configs()
        
        for config in configs:
            if config.environment == self.environment:
                pool = DatabaseConnectionPool(config)
                success = await pool.initialize()
                
                if success:
                    self.connection_pools[config.name] = pool
                    self.logger.info(f"Initialized connection pool for {config.name}")
                else:
                    self.logger.warning(f"Failed to initialize connection pool for {config.name}")
    
    async def _initialize_legacy_integrator(self):
        """Initialize legacy database integrator for backward compatibility"""
        try:
            # Convert new configurations to legacy format
            legacy_config = {}
            
            for name, pool in self.connection_pools.items():
                config = pool.config
                
                if config.type == DatabaseType.POSTGRESQL:
                    legacy_config['postgresql'] = {
                        'host': config.host,
                        'port': config.port,
                        'database': config.database,
                        'username': config.credentials.username if config.credentials else None,
                        'password': config.credentials.password if config.credentials else None,
                        'pool_size': config.pool_config.max_connections,
                        'max_overflow': config.pool_config.max_overflow
                    }
                
                elif config.type == DatabaseType.INFLUXDB:
                    legacy_config['influxdb'] = {
                        'url': config.get_connection_string(include_credentials=False),
                        'token': config.credentials.token if config.credentials else None,
                        'org': config.metadata.get('org', 'prometheus'),
                        'bucket': config.database or 'geological_data'
                    }
                
                elif config.type == DatabaseType.NEO4J:
                    legacy_config['neo4j'] = {
                        'uri': config.get_connection_string(include_credentials=False),
                        'username': config.credentials.username if config.credentials else None,
                        'password': config.credentials.password if config.credentials else None
                    }
                
                elif config.type == DatabaseType.REDIS:
                    legacy_config['redis'] = {
                        'host': config.host,
                        'port': config.port,
                        'password': config.credentials.password if config.credentials else None,
                        'db': int(config.database) if config.database else 0
                    }
            
            # Initialize legacy integrator
            self.legacy_integrator = EnhancedDatabaseIntegrator(legacy_config)
            await self.legacy_integrator.initialize()
            
            self.logger.info("Legacy database integrator initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize legacy integrator: {e}")
    
    async def _start_health_monitoring(self):
        """Start health monitoring for all connection pools"""
        async def health_monitor():
            while self.is_initialized:
                try:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    
                    for name, pool in self.connection_pools.items():
                        await pool.health_check()
                    
                    # Log unhealthy pools
                    unhealthy_pools = [
                        name for name, pool in self.connection_pools.items()
                        if pool.health_status != "healthy"
                    ]
                    
                    if unhealthy_pools:
                        self.logger.warning(f"Unhealthy database pools: {unhealthy_pools}")
                    
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
        
        self.health_monitor_task = asyncio.create_task(health_monitor())
    
    def get_database_config(self, name: str) -> Optional[DatabaseConfig]:
        """Get database configuration by name"""
        return self.config_manager.get_database_config(name) if self.config_manager else None
    
    def get_connection_pool(self, name: str) -> Optional[DatabaseConnectionPool]:
        """Get connection pool by database name"""
        return self.connection_pools.get(name)
    
    @asynccontextmanager
    async def get_database_connection(self, name: str):
        """Get database connection by name"""
        pool = self.get_connection_pool(name)
        if not pool:
            raise ValueError(f"Database connection pool not found: {name}")
        
        async with pool.get_connection() as conn:
            yield conn
    
    async def execute_query(self, database_name: str, query_type: str, **kwargs) -> QueryResult:
        """Execute query using unified interface"""
        if self.legacy_integrator:
            return await self.legacy_integrator.execute_unified_query(query_type, **kwargs)
        else:
            raise RuntimeError("Legacy integrator not available")
    
    async def test_all_connections(self) -> Dict[str, bool]:
        """Test all database connections"""
        results = {}
        
        for name, pool in self.connection_pools.items():
            results[name] = await pool.health_check()
        
        return results
    
    async def add_database(self, 
                          name: str,
                          db_type: DatabaseType,
                          host: str,
                          port: int,
                          **kwargs) -> bool:
        """Add new database configuration and connection pool"""
        try:
            # Create configuration
            config = await self.config_manager.create_database_config(
                name=name,
                db_type=db_type,
                host=host,
                port=port,
                **kwargs
            )
            
            # Initialize connection pool
            pool = DatabaseConnectionPool(config)
            success = await pool.initialize()
            
            if success:
                self.connection_pools[name] = pool
                self.logger.info(f"Added database: {name}")
                return True
            else:
                self.logger.error(f"Failed to initialize connection pool for new database: {name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to add database {name}: {e}")
            return False
    
    async def remove_database(self, name: str) -> bool:
        """Remove database configuration and connection pool"""
        try:
            # Cleanup connection pool
            if name in self.connection_pools:
                await self.connection_pools[name].cleanup()
                del self.connection_pools[name]
            
            # Remove configuration (implementation depends on config manager)
            # This would need to be implemented in DatabaseConfigManager
            
            self.logger.info(f"Removed database: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove database {name}: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and health information"""
        return {
            'initialized': self.is_initialized,
            'environment': self.environment.value,
            'total_databases': len(self.connection_pools),
            'healthy_databases': len([
                pool for pool in self.connection_pools.values()
                if pool.health_status == "healthy"
            ]),
            'database_status': {
                name: {
                    'health': pool.health_status,
                    'last_check': pool.last_health_check.isoformat() if pool.last_health_check else None,
                    'type': pool.config.type.value,
                    'host': pool.config.host,
                    'port': pool.config.port
                }
                for name, pool in self.connection_pools.items()
            }
        }
    
    async def reload_configurations(self) -> bool:
        """Reload database configurations from files"""
        try:
            # Cleanup existing pools
            for pool in self.connection_pools.values():
                await pool.cleanup()
            self.connection_pools.clear()
            
            # Reinitialize configuration manager
            self.config_manager = await create_config_manager(self.environment.value)
            
            # Reinitialize connection pools
            await self._initialize_connection_pools()
            
            self.logger.info("Database configurations reloaded")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reload configurations: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup database service"""
        self.logger.info("Cleaning up Database Service")
        
        # Stop health monitoring
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup connection pools
        for pool in self.connection_pools.values():
            await pool.cleanup()
        self.connection_pools.clear()
        
        # Cleanup legacy integrator
        if self.legacy_integrator:
            await self.legacy_integrator.cleanup()
        
        # Cleanup configuration manager
        if self.config_manager:
            await self.config_manager.cleanup()
        
        self.is_initialized = False
        self.logger.info("Database Service cleanup complete")


# Global database service instance
_database_service: Optional[DatabaseService] = None


async def get_database_service(environment: str = None) -> DatabaseService:
    """Get global database service instance"""
    global _database_service
    
    if _database_service is None:
        _database_service = DatabaseService(environment)
        await _database_service.initialize()
    
    return _database_service


async def cleanup_database_service():
    """Cleanup global database service instance"""
    global _database_service
    
    if _database_service:
        await _database_service.cleanup()
        _database_service = None


# Convenience functions for common database operations
@validate_db_config("postgresql")
async def get_postgresql_connection(database_name: str = "postgresql"):
    """Get PostgreSQL connection"""
    service = await get_database_service()
    async with service.get_database_connection(database_name) as conn:
        yield conn


@validate_db_config("redis")
async def get_redis_connection(database_name: str = "redis"):
    """Get Redis connection"""
    service = await get_database_service()
    async with service.get_database_connection(database_name) as conn:
        yield conn


@validate_db_config("influxdb")
async def get_influxdb_connection(database_name: str = "influxdb"):
    """Get InfluxDB connection"""
    service = await get_database_service()
    async with service.get_database_connection(database_name) as conn:
        yield conn


@validate_db_config("neo4j")
async def get_neo4j_connection(database_name: str = "neo4j"):
    """Get Neo4j connection"""
    service = await get_database_service()
    async with service.get_database_connection(database_name) as conn:
        yield conn