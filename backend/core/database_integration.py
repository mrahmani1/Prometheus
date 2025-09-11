"""
Enhanced Database Integration Layer
Multi-database support with InfluxDB, PostgreSQL+PostGIS, Neo4j, and Redis
Comprehensive geological data management with performance optimization
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncpg
import psycopg2
import redis.asyncio as redis
from contextlib import asynccontextmanager
import warnings

# Import existing database integration
from backend.modules.advanced_database_integration import AdvancedDatabaseIntegrator, GeologicalDataPoint

# InfluxDB client
try:
    from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
    from influxdb_client import Point, WritePrecision
    from influxdb_client.client.write_api_async import WriteApiAsync
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    warnings.warn("InfluxDB client not available")

# Neo4j driver
try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    warnings.warn("Neo4j driver not available")

# SQLAlchemy for advanced ORM
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from geoalchemy2 import Geometry


# Database configuration
DATABASE_CONFIG = {
    'postgresql': {
        'host': 'localhost',
        'port': 5432,
        'database': 'prometheus_geo',
        'username': 'prometheus_admin',
        'password': 'GeoPhys2024!PrometheusDB#',
        'pool_size': 20,
        'max_overflow': 30
    },
    'influxdb': {
        'url': 'http://localhost:8086',
        'token': 'prometheus_influx_token',
        'org': 'prometheus_org',
        'bucket': 'geological_data'
    },
    'neo4j': {
        'uri': 'bolt://localhost:7687',
        'username': 'neo4j',
        'password': 'prometheus_neo4j'
    },
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'password': 'prometheus_redis',
        'db': 0
    }
}


@dataclass
class DatabaseMetrics:
    """Database performance and health metrics"""
    connection_count: int = 0
    query_count: int = 0
    average_query_time: float = 0.0
    error_count: int = 0
    cache_hit_rate: float = 0.0
    storage_usage: Dict[str, float] = field(default_factory=dict)
    last_backup: Optional[datetime] = None


@dataclass
class QueryResult:
    """Standardized query result structure"""
    data: Any
    metadata: Dict[str, Any]
    execution_time: float
    row_count: int
    cached: bool = False


# SQLAlchemy Models
Base = declarative_base()


class Volcano(Base):
    """Volcano entity model"""
    __tablename__ = 'volcanoes'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False)
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    elevation = Column(Float)
    volcano_type = Column(String(100))
    last_eruption = Column(DateTime)
    hazard_level = Column(Integer)
    monitoring_status = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    seismic_events = relationship("SeismicEvent", back_populates="volcano")
    deformation_data = relationship("DeformationData", back_populates="volcano")
    
    __table_args__ = (
        Index('idx_volcano_location', 'location', postgresql_using='gist'),
        Index('idx_volcano_name', 'name'),
    )


class SeismicEvent(Base):
    """Seismic event model"""
    __tablename__ = 'seismic_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    volcano_id = Column(UUID(as_uuid=True), ForeignKey('volcanoes.id'))
    timestamp = Column(DateTime, nullable=False)
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    magnitude = Column(Float, nullable=False)
    depth = Column(Float, nullable=False)
    event_type = Column(String(50))
    confidence = Column(Float)
    waveform_data = Column(JSONB)
    moment_tensor = Column(JSONB)
    processing_metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    volcano = relationship("Volcano", back_populates="seismic_events")
    
    __table_args__ = (
        Index('idx_seismic_timestamp', 'timestamp'),
        Index('idx_seismic_location', 'location', postgresql_using='gist'),
        Index('idx_seismic_magnitude', 'magnitude'),
    )


class DeformationData(Base):
    """Ground deformation data model"""
    __tablename__ = 'deformation_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    volcano_id = Column(UUID(as_uuid=True), ForeignKey('volcanoes.id'))
    timestamp = Column(DateTime, nullable=False)
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    displacement_x = Column(Float)
    displacement_y = Column(Float)
    displacement_z = Column(Float)
    coherence = Column(Float)
    measurement_type = Column(String(50))  # InSAR, GPS, etc.
    uncertainty = Column(JSONB)
    processing_parameters = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    volcano = relationship("Volcano", back_populates="deformation_data")
    
    __table_args__ = (
        Index('idx_deformation_timestamp', 'timestamp'),
        Index('idx_deformation_location', 'location', postgresql_using='gist'),
    )


class GeologicalUnit(Base):
    """Geological unit model"""
    __tablename__ = 'geological_units'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False)
    age_min = Column(Float)  # Ma
    age_max = Column(Float)  # Ma
    lithology = Column(String(100))
    formation = Column(String(255))
    geometry = Column(Geometry('MULTIPOLYGON', srid=4326))
    properties = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_geological_geometry', 'geometry', postgresql_using='gist'),
        Index('idx_geological_age', 'age_min', 'age_max'),
    )


class EnhancedDatabaseIntegrator:
    """
    Enhanced Database Integration Layer
    Provides unified access to multiple database systems with geological data optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DATABASE_CONFIG
        self.logger = logging.getLogger(__name__ + ".EnhancedDatabaseIntegrator")
        
        # Database connections
        self.postgresql_engine = None
        self.postgresql_session_factory = None
        self.influxdb_client = None
        self.neo4j_driver = None
        self.redis_client = None
        
        # Base integrator
        self.base_integrator = AdvancedDatabaseIntegrator(config)
        
        # Performance metrics
        self.metrics = DatabaseMetrics()
        
        # Query cache
        self.query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Connection pools
        self.connection_pools = {}
        
        self.logger.info("Enhanced Database Integrator initialized")
    
    async def initialize(self) -> bool:
        """Initialize all database connections"""
        try:
            self.logger.info("Initializing database connections")
            
            # Initialize PostgreSQL with PostGIS
            await self._initialize_postgresql()
            
            # Initialize InfluxDB for time-series data
            await self._initialize_influxdb()
            
            # Initialize Neo4j for knowledge graphs
            await self._initialize_neo4j()
            
            # Initialize Redis for caching
            await self._initialize_redis()
            
            # Initialize base integrator
            if hasattr(self.base_integrator, 'initialize'):
                await self.base_integrator.initialize()
            
            # Create database schemas
            await self._create_schemas()
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            self.logger.info("Database integration initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return False
    
    async def _initialize_postgresql(self):
        """Initialize PostgreSQL connection with PostGIS"""
        pg_config = self.config.get('postgresql', {})
        
        # Create async engine
        database_url = (
            f"postgresql+asyncpg://{pg_config['username']}:{pg_config['password']}"
            f"@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
        )
        
        self.postgresql_engine = create_async_engine(
            database_url,
            pool_size=pg_config.get('pool_size', 20),
            max_overflow=pg_config.get('max_overflow', 30),
            pool_timeout=30,
            pool_recycle=3600,
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.postgresql_session_factory = async_sessionmaker(
            self.postgresql_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        self.logger.info("PostgreSQL connection initialized")
    
    async def _initialize_influxdb(self):
        """Initialize InfluxDB connection for time-series data"""
        if not INFLUXDB_AVAILABLE:
            self.logger.warning("InfluxDB client not available, skipping initialization")
            return
        
        influx_config = self.config.get('influxdb', {})
        
        self.influxdb_client = InfluxDBClientAsync(
            url=influx_config.get('url', 'http://localhost:8086'),
            token=influx_config.get('token'),
            org=influx_config.get('org'),
            timeout=30000
        )
        
        self.logger.info("InfluxDB connection initialized")
    
    async def _initialize_neo4j(self):
        """Initialize Neo4j connection for knowledge graphs"""
        if not NEO4J_AVAILABLE:
            self.logger.warning("Neo4j driver not available, skipping initialization")
            return
        
        neo4j_config = self.config.get('neo4j', {})
        
        self.neo4j_driver = AsyncGraphDatabase.driver(
            neo4j_config.get('uri', 'bolt://localhost:7687'),
            auth=(neo4j_config.get('username'), neo4j_config.get('password')),
            max_connection_lifetime=3600,
            max_connection_pool_size=50
        )
        
        self.logger.info("Neo4j connection initialized")
    
    async def _initialize_redis(self):
        """Initialize Redis connection for caching"""
        redis_config = self.config.get('redis', {})
        
        self.redis_client = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            password=redis_config.get('password'),
            db=redis_config.get('db', 0),
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=30
        )
        
        # Test connection
        await self.redis_client.ping()
        
        self.logger.info("Redis connection initialized")
    
    async def _create_schemas(self):
        """Create database schemas and tables"""
        if self.postgresql_engine:
            async with self.postgresql_engine.begin() as conn:
                # Enable PostGIS extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
                await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
                
                # Create tables
                await conn.run_sync(Base.metadata.create_all)
                
                # Create hypertables for time-series data
                try:
                    await conn.execute(
                        "SELECT create_hypertable('seismic_events', 'timestamp', if_not_exists => TRUE)"
                    )
                    await conn.execute(
                        "SELECT create_hypertable('deformation_data', 'timestamp', if_not_exists => TRUE)"
                    )
                except Exception as e:
                    self.logger.warning(f"TimescaleDB hypertable creation failed: {e}")
        
        self.logger.info("Database schemas created")
    
    # PostgreSQL Operations
    @asynccontextmanager
    async def get_postgresql_session(self) -> AsyncSession:
        """Get PostgreSQL session with automatic cleanup"""
        async with self.postgresql_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def store_seismic_event(self, event_data: Dict[str, Any]) -> str:
        """Store seismic event in PostgreSQL"""
        async with self.get_postgresql_session() as session:
            event = SeismicEvent(
                id=event_data.get('id'),
                volcano_id=event_data.get('volcano_id'),
                timestamp=event_data['timestamp'],
                location=f"POINT({event_data['longitude']} {event_data['latitude']})",
                magnitude=event_data['magnitude'],
                depth=event_data['depth'],
                event_type=event_data.get('event_type'),
                confidence=event_data.get('confidence'),
                waveform_data=event_data.get('waveform_data'),
                moment_tensor=event_data.get('moment_tensor'),
                processing_metadata=event_data.get('metadata', {})
            )
            
            session.add(event)
            await session.commit()
            
            self.metrics.query_count += 1
            return str(event.id)
    
    async def query_seismic_events(self, 
                                  volcano_id: Optional[str] = None,
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None,
                                  min_magnitude: Optional[float] = None,
                                  limit: int = 1000) -> List[Dict[str, Any]]:
        """Query seismic events with filters"""
        async with self.get_postgresql_session() as session:
            query = session.query(SeismicEvent)
            
            if volcano_id:
                query = query.filter(SeismicEvent.volcano_id == volcano_id)
            if start_time:
                query = query.filter(SeismicEvent.timestamp >= start_time)
            if end_time:
                query = query.filter(SeismicEvent.timestamp <= end_time)
            if min_magnitude:
                query = query.filter(SeismicEvent.magnitude >= min_magnitude)
            
            query = query.order_by(SeismicEvent.timestamp.desc()).limit(limit)
            
            result = await query.all()
            
            self.metrics.query_count += 1
            
            return [
                {
                    'id': str(event.id),
                    'volcano_id': str(event.volcano_id) if event.volcano_id else None,
                    'timestamp': event.timestamp,
                    'magnitude': event.magnitude,
                    'depth': event.depth,
                    'event_type': event.event_type,
                    'confidence': event.confidence
                }
                for event in result
            ]
    
    # InfluxDB Operations
    async def store_time_series_data(self, measurement: str, 
                                   tags: Dict[str, str],
                                   fields: Dict[str, Union[float, int, str]],
                                   timestamp: Optional[datetime] = None) -> bool:
        """Store time-series data in InfluxDB"""
        if not self.influxdb_client:
            return False
        
        try:
            point = Point(measurement)
            
            # Add tags
            for key, value in tags.items():
                point = point.tag(key, value)
            
            # Add fields
            for key, value in fields.items():
                point = point.field(key, value)
            
            # Set timestamp
            if timestamp:
                point = point.time(timestamp, WritePrecision.NS)
            
            # Write to InfluxDB
            write_api = self.influxdb_client.write_api()
            bucket = self.config.get('influxdb', {}).get('bucket', 'geological_data')
            
            await write_api.write(bucket=bucket, record=point)
            
            self.metrics.query_count += 1
            return True
            
        except Exception as e:
            self.logger.error(f"InfluxDB write error: {e}")
            self.metrics.error_count += 1
            return False
    
    async def query_time_series_data(self, 
                                   measurement: str,
                                   start_time: datetime,
                                   end_time: datetime,
                                   filters: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Query time-series data from InfluxDB"""
        if not self.influxdb_client:
            return []
        
        try:
            # Build Flux query
            bucket = self.config.get('influxdb', {}).get('bucket', 'geological_data')
            
            query = f'''
                from(bucket: "{bucket}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r._measurement == "{measurement}")
            '''
            
            # Add filters
            if filters:
                for key, value in filters.items():
                    query += f'|> filter(fn: (r) => r.{key} == "{value}")\n'
            
            # Execute query
            query_api = self.influxdb_client.query_api()
            result = await query_api.query(query)
            
            # Process results
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        'time': record.get_time(),
                        'measurement': record.get_measurement(),
                        'field': record.get_field(),
                        'value': record.get_value(),
                        'tags': {k: v for k, v in record.values.items() if k.startswith('tag_')}
                    })
            
            self.metrics.query_count += 1
            return data
            
        except Exception as e:
            self.logger.error(f"InfluxDB query error: {e}")
            self.metrics.error_count += 1
            return []
    
    # Neo4j Operations
    async def create_geological_relationship(self, 
                                           source_node: Dict[str, Any],
                                           target_node: Dict[str, Any],
                                           relationship_type: str,
                                           properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create geological process relationship in Neo4j"""
        if not self.neo4j_driver:
            return False
        
        try:
            async with self.neo4j_driver.session() as session:
                query = """
                MERGE (source:GeologicalProcess {id: $source_id, name: $source_name})
                MERGE (target:GeologicalProcess {id: $target_id, name: $target_name})
                MERGE (source)-[r:%s]->(target)
                SET r += $properties
                RETURN r
                """ % relationship_type
                
                result = await session.run(
                    query,
                    source_id=source_node['id'],
                    source_name=source_node['name'],
                    target_id=target_node['id'],
                    target_name=target_node['name'],
                    properties=properties or {}
                )
                
                await result.consume()
                
                self.metrics.query_count += 1
                return True
                
        except Exception as e:
            self.logger.error(f"Neo4j relationship creation error: {e}")
            self.metrics.error_count += 1
            return False
    
    async def query_geological_relationships(self, 
                                           node_id: str,
                                           relationship_types: Optional[List[str]] = None,
                                           depth: int = 2) -> List[Dict[str, Any]]:
        """Query geological process relationships from Neo4j"""
        if not self.neo4j_driver:
            return []
        
        try:
            async with self.neo4j_driver.session() as session:
                # Build relationship type filter
                rel_filter = ""
                if relationship_types:
                    rel_types = "|".join(relationship_types)
                    rel_filter = f":{rel_types}"
                
                query = f"""
                MATCH path = (start:GeologicalProcess {{id: $node_id}})-[r{rel_filter}*1..{depth}]-(connected)
                RETURN path, relationships(path) as rels, nodes(path) as nodes
                """
                
                result = await session.run(query, node_id=node_id)
                
                relationships = []
                async for record in result:
                    path_nodes = record['nodes']
                    path_rels = record['rels']
                    
                    for i, rel in enumerate(path_rels):
                        relationships.append({
                            'source': dict(path_nodes[i]),
                            'target': dict(path_nodes[i + 1]),
                            'relationship': dict(rel),
                            'type': rel.type
                        })
                
                self.metrics.query_count += 1
                return relationships
                
        except Exception as e:
            self.logger.error(f"Neo4j relationship query error: {e}")
            self.metrics.error_count += 1
            return []
    
    # Redis Caching Operations
    async def cache_set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            serialized_value = json.dumps(value, default=str)
            await self.redis_client.setex(key, ttl, serialized_value)
            return True
            
        except Exception as e:
            self.logger.error(f"Redis cache set error: {e}")
            return False
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                self.metrics.cache_hit_rate += 1
                return json.loads(value)
            return None
            
        except Exception as e:
            self.logger.error(f"Redis cache get error: {e}")
            return None
    
    async def cache_delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Redis cache delete error: {e}")
            return False
    
    # Unified Query Interface
    async def execute_unified_query(self, query_type: str, **kwargs) -> QueryResult:
        """Execute unified query across multiple databases"""
        start_time = datetime.now()
        
        try:
            if query_type == "seismic_events":
                data = await self.query_seismic_events(**kwargs)
                database = "postgresql"
            elif query_type == "time_series":
                data = await self.query_time_series_data(**kwargs)
                database = "influxdb"
            elif query_type == "relationships":
                data = await self.query_geological_relationships(**kwargs)
                database = "neo4j"
            else:
                raise ValueError(f"Unknown query type: {query_type}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                data=data,
                metadata={
                    'database': database,
                    'query_type': query_type,
                    'parameters': kwargs
                },
                execution_time=execution_time,
                row_count=len(data) if isinstance(data, list) else 1
            )
            
        except Exception as e:
            self.logger.error(f"Unified query error: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                data=None,
                metadata={'error': str(e)},
                execution_time=execution_time,
                row_count=0
            )
    
    # Health Monitoring
    async def _start_health_monitoring(self):
        """Start database health monitoring"""
        async def health_monitor():
            while True:
                try:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    await self._check_database_health()
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
        
        asyncio.create_task(health_monitor())
    
    async def _check_database_health(self):
        """Check health of all database connections"""
        health_status = {}
        
        # Check PostgreSQL
        if self.postgresql_engine:
            try:
                async with self.postgresql_engine.connect() as conn:
                    await conn.execute("SELECT 1")
                health_status['postgresql'] = 'healthy'
            except Exception as e:
                health_status['postgresql'] = f'error: {e}'
        
        # Check InfluxDB
        if self.influxdb_client:
            try:
                health = await self.influxdb_client.health()
                health_status['influxdb'] = 'healthy' if health.status == 'pass' else 'degraded'
            except Exception as e:
                health_status['influxdb'] = f'error: {e}'
        
        # Check Neo4j
        if self.neo4j_driver:
            try:
                async with self.neo4j_driver.session() as session:
                    await session.run("RETURN 1")
                health_status['neo4j'] = 'healthy'
            except Exception as e:
                health_status['neo4j'] = f'error: {e}'
        
        # Check Redis
        if self.redis_client:
            try:
                await self.redis_client.ping()
                health_status['redis'] = 'healthy'
            except Exception as e:
                health_status['redis'] = f'error: {e}'
        
        # Log health status
        unhealthy_dbs = [db for db, status in health_status.items() if status != 'healthy']
        if unhealthy_dbs:
            self.logger.warning(f"Unhealthy databases: {unhealthy_dbs}")
        else:
            self.logger.debug("All databases healthy")
    
    def get_database_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics"""
        return {
            'connection_count': self.metrics.connection_count,
            'query_count': self.metrics.query_count,
            'average_query_time': self.metrics.average_query_time,
            'error_count': self.metrics.error_count,
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'storage_usage': self.metrics.storage_usage
        }
    
    async def cleanup(self):
        """Clean up database connections"""
        self.logger.info("Cleaning up database connections")
        
        if self.postgresql_engine:
            await self.postgresql_engine.dispose()
        
        if self.influxdb_client:
            await self.influxdb_client.close()
        
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("Database cleanup complete")


# Global database integrator instance
_database_integrator: Optional[EnhancedDatabaseIntegrator] = None


def get_database_integrator() -> EnhancedDatabaseIntegrator:
    """Get the global database integrator instance"""
    global _database_integrator
    if _database_integrator is None:
        _database_integrator = EnhancedDatabaseIntegrator()
    return _database_integrator


async def initialize_database_integrator(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the global database integrator instance"""
    global _database_integrator
    _database_integrator = EnhancedDatabaseIntegrator(config)
    return await _database_integrator.initialize()