"""
Advanced Database Integration Module
Real-time database connectivity with comprehensive schema integration and advanced querying
Integrates with PostgreSQL, PostGIS, TimescaleDB and external geological databases
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncpg
import psycopg2
import psycopg2.extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import aiohttp
import requests
from contextlib import asynccontextmanager
import warnings

# Database configuration from deployment config
DATABASE_CONFIG = {
    'database': 'prometheus_geo',
    'user': 'prometheus_admin',
    'password': 'GeoPhys2024!PrometheusDB#',
    'host': 'localhost',
    'port': 5432,
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600
}

@dataclass
class GeologicalDataPoint:
    """Standardized geological data point structure"""
    id: str
    location: Tuple[float, float, float]  # lon, lat, depth
    timestamp: datetime
    data_type: str
    values: Dict[str, Any]
    quality_grade: str
    source: str
    uncertainty: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class AdvancedDatabaseIntegrator:
    """Advanced database integration with real-time geological data management"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize advanced database integrator"""
        self.config = config or DATABASE_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Database connections
        self.pg_engine = None
        self.async_engine = None
        self.async_session = None
        self.redis_client = None
        
        # Connection pools
        self.connection_pools = {}
        
        # Schema metadata
        self.metadata = None
        self.table_schemas = {}
        
        # Real-time data streams
        self.active_streams = {}
        
        # External API clients
        self.external_apis = {}
        
        # Data validation rules
        self.validation_rules = {}
        
        # Initialize connection
        self._initialize_connections()
    
    async def _initialize_connections(self):
        """Initialize all database and external service connections"""
        try:
            # PostgreSQL with PostGIS and TimescaleDB
            await self._initialize_postgresql_connection()
            
            # Redis for caching and real-time streams
            await self._initialize_redis_connection()
            
            # External geological data sources
            await self._initialize_external_apis()
            
            # Load database schemas
            await self._load_database_schemas()
            
            # Setup validation rules
            await self._setup_validation_rules()
            
            self.logger.info("Advanced database integrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database integrator initialization failed: {e}")
            raise
    
    async def _initialize_postgresql_connection(self):
        """Initialize PostgreSQL connection with PostGIS and TimescaleDB extensions"""
        try:
            # Create async engine for high-performance operations
            connection_string = (
                f"postgresql+asyncpg://{self.config['user']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
            
            self.async_engine = create_async_engine(
                connection_string,
                pool_size=self.config['pool_size'],
                max_overflow=self.config['max_overflow'],
                pool_timeout=self.config['pool_timeout'],
                pool_recycle=self.config['pool_recycle'],
                echo=False
            )
            
            # Create session factory
            self.async_session = sessionmaker(
                self.async_engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # Test connection
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("SELECT version();"))
                version = result.scalar()
                self.logger.info(f"Connected to PostgreSQL: {version}")
            
            # Verify PostGIS extension
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("SELECT PostGIS_version();"))
                postgis_version = result.scalar()
                self.logger.info(f"PostGIS version: {postgis_version}")
            
            # Verify TimescaleDB extension
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"))
                timescale_version = result.scalar()
                if timescale_version:
                    self.logger.info(f"TimescaleDB version: {timescale_version}")
                
        except Exception as e:
            self.logger.error(f"PostgreSQL connection initialization failed: {e}")
            raise
    
    async def _initialize_redis_connection(self):
        """Initialize Redis connection for caching and real-time streams"""
        try:
            import aioredis
            
            self.redis_client = aioredis.from_url(
                "redis://localhost:6379",
                decode_responses=True,
                max_connections=50
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            self.logger.info("Connected to Redis for caching and streaming")
            
        except ImportError:
            self.logger.warning("aioredis not available - Redis functionality disabled")
            self.redis_client = None
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def _initialize_external_apis(self):
        """Initialize external geological data API connections"""
        try:
            # USGS Earthquake Hazards Program API
            self.external_apis['usgs_earthquake'] = {
                'base_url': 'https://earthquake.usgs.gov/fdsnws/event/1/',
                'session': aiohttp.ClientSession(),
                'rate_limit': 60  # requests per minute
            }
            
            # IRIS (Incorporated Research Institutions for Seismology) API
            self.external_apis['iris_seismic'] = {
                'base_url': 'https://service.iris.edu/fdsnws/',
                'session': aiohttp.ClientSession(),
                'rate_limit': 100
            }
            
            # NASA Earth Data API
            self.external_apis['nasa_earthdata'] = {
                'base_url': 'https://cmr.earthdata.nasa.gov/search/',
                'session': aiohttp.ClientSession(),
                'rate_limit': 1000
            }
            
            # NOAA Global Monitoring Laboratory
            self.external_apis['noaa_gml'] = {
                'base_url': 'https://gml.noaa.gov/webdata/',
                'session': aiohttp.ClientSession(),
                'rate_limit': 50
            }
            
            # European Space Agency (ESA) Copernicus API
            self.external_apis['esa_copernicus'] = {
                'base_url': 'https://scihub.copernicus.eu/dhus/',
                'session': aiohttp.ClientSession(),
                'rate_limit': 20
            }
            
            self.logger.info("External API connections initialized")
            
        except Exception as e:
            self.logger.error(f"External API initialization failed: {e}")
    
    async def _load_database_schemas(self):
        """Load database schemas and table definitions"""
        try:
            async with self.async_engine.begin() as conn:
                # Load metadata
                self.metadata = MetaData()
                await conn.run_sync(self.metadata.reflect)
                
                # Load specific table schemas from deployment config
                schema_tables = [
                    'rock_mechanical_properties',
                    'literature_rock_properties',
                    'volcanic_chambers',
                    'critical_failure_zones',
                    'real_time_earthquake_events',
                    'usgs_earthquake_catalog',
                    'igs_gps_stations',
                    'sentinel_insar_data'
                ]
                
                for table_name in schema_tables:
                    if table_name in self.metadata.tables:
                        self.table_schemas[table_name] = self.metadata.tables[table_name]
                        self.logger.debug(f"Loaded schema for table: {table_name}")
                
                self.logger.info(f"Loaded schemas for {len(self.table_schemas)} tables")
                
        except Exception as e:
            self.logger.error(f"Schema loading failed: {e}")
    
    async def _setup_validation_rules(self):
        """Setup data validation rules based on geological constraints"""
        self.validation_rules = {
            'rock_mechanical_properties': {
                'unconfined_compressive_strength_mpa': {'min': 0.1, 'max': 500.0},
                'tensile_strength_mpa': {'min': 0.001, 'max': 50.0},
                'cohesion_mpa': {'min': 0.0, 'max': 100.0},
                'friction_angle_deg': {'min': 0.0, 'max': 90.0},
                'elastic_modulus_gpa': {'min': 0.1, 'max': 200.0},
                'poisson_ratio': {'min': -1.0, 'max': 0.5},
                'porosity': {'min': 0.0, 'max': 1.0},
                'data_quality_grade': {'values': ['A', 'B', 'C', 'D']}
            },
            'volcanic_chambers': {
                'chamber_volume_km3': {'min': 0.001, 'max': 10000.0},
                'chamber_depth_km': {'min': 0.1, 'max': 50.0},
                'temperature_c': {'min': 700.0, 'max': 1400.0},
                'pressure_mpa': {'min': 1.0, 'max': 2000.0},
                'melt_fraction': {'min': 0.0, 'max': 1.0}
            }
        }
    
    async def insert_geological_data(self, data_points: List[GeologicalDataPoint]) -> Dict[str, Any]:
        """Insert geological data points with validation and quality control"""
        try:
            results = {
                'inserted': 0,
                'failed': 0,
                'validation_errors': [],
                'database_errors': []
            }
            
            async with self.async_session() as session:
                for data_point in data_points:
                    try:
                        # Validate data point
                        validation_result = self._validate_data_point(data_point)
                        if not validation_result['valid']:
                            results['failed'] += 1
                            results['validation_errors'].append({
                                'id': data_point.id,
                                'errors': validation_result['errors']
                            })
                            continue
                        
                        # Insert into appropriate table based on data type
                        await self._insert_single_data_point(session, data_point)
                        results['inserted'] += 1
                        
                    except Exception as e:
                        results['failed'] += 1
                        results['database_errors'].append({
                            'id': data_point.id,
                            'error': str(e)
                        })
                
                await session.commit()
            
            # Update caches and indices
            await self._update_spatial_indices()
            await self._update_redis_cache()
            
            self.logger.info(f"Inserted {results['inserted']} geological data points")
            return results
            
        except Exception as e:
            self.logger.error(f"Geological data insertion failed: {e}")
            raise
    
    async def query_geological_data(self, query_params: Dict[str, Any]) -> List[GeologicalDataPoint]:
        """Query geological data with spatial, temporal, and thematic filters"""
        try:
            # Parse query parameters
            spatial_filter = query_params.get('spatial_filter')
            temporal_filter = query_params.get('temporal_filter')
            data_types = query_params.get('data_types', [])
            quality_grades = query_params.get('quality_grades', ['A', 'B', 'C'])
            
            # Build SQL query with PostGIS spatial operations
            query_parts = []
            params = {}
            
            # Spatial filter
            if spatial_filter:
                if spatial_filter['type'] == 'bbox':
                    query_parts.append("""
                        ST_Within(location, ST_MakeEnvelope(
                            :min_lon, :min_lat, :max_lon, :max_lat, 4326
                        ))
                    """)
                    params.update(spatial_filter['bounds'])
                
                elif spatial_filter['type'] == 'radius':
                    query_parts.append("""
                        ST_DWithin(location, ST_Point(:center_lon, :center_lat), :radius_m)
                    """)
                    params.update(spatial_filter)
            
            # Temporal filter
            if temporal_filter:
                if temporal_filter.get('start_time'):
                    query_parts.append("timestamp >= :start_time")
                    params['start_time'] = temporal_filter['start_time']
                
                if temporal_filter.get('end_time'):
                    query_parts.append("timestamp <= :end_time")
                    params['end_time'] = temporal_filter['end_time']
            
            # Data type filter
            if data_types:
                query_parts.append("data_type = ANY(:data_types)")
                params['data_types'] = data_types
            
            # Quality filter
            if quality_grades:
                query_parts.append("quality_grade = ANY(:quality_grades)")
                params['quality_grades'] = quality_grades
            
            # Execute query across relevant tables
            results = []
            
            async with self.async_engine.begin() as conn:
                # Query each relevant table
                for table_name, schema in self.table_schemas.items():
                    if self._table_matches_query(table_name, query_params):
                        table_query = self._build_table_query(table_name, query_parts)
                        
                        result = await conn.execute(text(table_query), params)
                        rows = result.fetchall()
                        
                        for row in rows:
                            data_point = self._row_to_data_point(row, table_name)
                            results.append(data_point)
            
            self.logger.info(f"Retrieved {len(results)} geological data points")
            return results
            
        except Exception as e:
            self.logger.error(f"Geological data query failed: {e}")
            raise
    
    async def stream_real_time_data(self, data_types: List[str], callback: callable) -> str:
        """Setup real-time data streaming from geological monitoring networks"""
        try:
            stream_id = f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize stream configurations
            stream_configs = {}
            
            for data_type in data_types:
                if data_type == 'earthquake':
                    stream_configs['earthquake'] = {
                        'source': 'usgs_earthquake',
                        'endpoint': 'query',
                        'params': {
                            'format': 'geojson',
                            'minmagnitude': 2.0,
                            'eventtype': 'earthquake',
                            'orderby': 'time'
                        },
                        'poll_interval': 60  # seconds
                    }
                
                elif data_type == 'gps':
                    stream_configs['gps'] = {
                        'source': 'igs_stations',
                        'poll_interval': 300  # 5 minutes
                    }
                
                elif data_type == 'insar':
                    stream_configs['insar'] = {
                        'source': 'esa_copernicus',
                        'product_type': 'IW_SLC',
                        'poll_interval': 3600  # 1 hour
                    }
            
            # Start streaming tasks
            streaming_tasks = []
            for stream_name, config in stream_configs.items():
                task = asyncio.create_task(
                    self._stream_data_source(stream_id, stream_name, config, callback)
                )
                streaming_tasks.append(task)
            
            self.active_streams[stream_id] = {
                'tasks': streaming_tasks,
                'start_time': datetime.now(),
                'callback': callback,
                'data_types': data_types
            }
            
            self.logger.info(f"Started real-time streaming: {stream_id}")
            return stream_id
            
        except Exception as e:
            self.logger.error(f"Real-time streaming setup failed: {e}")
            raise
    
    async def _stream_data_source(self, stream_id: str, stream_name: str, 
                                config: Dict, callback: callable):
        """Stream data from a specific source"""
        try:
            last_update = datetime.now()
            
            while stream_id in self.active_streams:
                try:
                    # Fetch new data
                    new_data = await self._fetch_stream_data(stream_name, config, last_update)
                    
                    if new_data:
                        # Process and validate data
                        processed_data = []
                        for item in new_data:
                            data_point = self._convert_to_data_point(item, stream_name)
                            if self._validate_data_point(data_point)['valid']:
                                processed_data.append(data_point)
                        
                        if processed_data:
                            # Store in database
                            await self.insert_geological_data(processed_data)
                            
                            # Call user callback
                            await callback(stream_id, stream_name, processed_data)
                            
                            last_update = datetime.now()
                    
                    # Wait for next poll
                    await asyncio.sleep(config['poll_interval'])
                    
                except Exception as e:
                    self.logger.error(f"Stream {stream_name} error: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except Exception as e:
            self.logger.error(f"Streaming task {stream_name} failed: {e}")
    
    async def _fetch_stream_data(self, stream_name: str, config: Dict, 
                               since: datetime) -> List[Dict]:
        """Fetch new data from external stream source"""
        try:
            if config['source'] == 'usgs_earthquake':
                return await self._fetch_usgs_earthquake_data(config, since)
            elif config['source'] == 'igs_stations':
                return await self._fetch_igs_gps_data(config, since)
            elif config['source'] == 'esa_copernicus':
                return await self._fetch_sentinel_insar_data(config, since)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Data fetch from {stream_name} failed: {e}")
            return []
    
    async def _fetch_usgs_earthquake_data(self, config: Dict, since: datetime) -> List[Dict]:
        """Fetch earthquake data from USGS API"""
        try:
            api_config = self.external_apis['usgs_earthquake']
            
            params = config['params'].copy()
            params['starttime'] = since.isoformat()
            params['endtime'] = datetime.now().isoformat()
            
            url = f"{api_config['base_url']}{config['endpoint']}"
            
            async with api_config['session'].get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('features', [])
                else:
                    self.logger.warning(f"USGS API error: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"USGS earthquake data fetch failed: {e}")
            return []
    
    def _validate_data_point(self, data_point: GeologicalDataPoint) -> Dict[str, Any]:
        """Validate geological data point against schema and domain constraints"""
        try:
            errors = []
            
            # Basic structure validation
            if not data_point.id:
                errors.append("Missing data point ID")
            
            if not data_point.location or len(data_point.location) != 3:
                errors.append("Invalid location coordinates")
            
            if not data_point.timestamp:
                errors.append("Missing timestamp")
            
            # Domain-specific validation
            if data_point.data_type in self.validation_rules:
                rules = self.validation_rules[data_point.data_type]
                
                for field, rule in rules.items():
                    if field in data_point.values:
                        value = data_point.values[field]
                        
                        if 'min' in rule and value < rule['min']:
                            errors.append(f"{field} below minimum: {value} < {rule['min']}")
                        
                        if 'max' in rule and value > rule['max']:
                            errors.append(f"{field} above maximum: {value} > {rule['max']}")
                        
                        if 'values' in rule and value not in rule['values']:
                            errors.append(f"{field} invalid value: {value}")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"]
            }
    
    async def close(self):
        """Close all database connections and cleanup resources"""
        try:
            # Stop all active streams
            for stream_id, stream_info in self.active_streams.items():
                for task in stream_info['tasks']:
                    task.cancel()
            
            # Close database connections
            if self.async_engine:
                await self.async_engine.dispose()
            
            if self.redis_client:
                await self.redis_client.close()
            
            # Close external API sessions
            for api_name, api_config in self.external_apis.items():
                if 'session' in api_config:
                    await api_config['session'].close()
            
            self.logger.info("Advanced database integrator closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database integrator: {e}")

# Global integrator instance
_integrator = None

async def get_database_integrator() -> AdvancedDatabaseIntegrator:
    """Get global database integrator instance"""
    global _integrator
    if _integrator is None:
        _integrator = AdvancedDatabaseIntegrator()
        await _integrator._initialize_connections()
    return _integrator