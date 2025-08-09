"""
PostgreSQL GIS and TimeScaleDB Integration
Local database integration for geological data storage and real-time processing
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from contextlib import asynccontextmanager
import uuid

# Database imports
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_batch
    from psycopg2.pool import ThreadedConnectionPool
    import psycopg2.extensions
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    warnings.warn("PostgreSQL not available. Database features will be limited.")

# GIS imports
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiPolygon
    from shapely.wkt import dumps as wkt_dumps, loads as wkt_loads
    import pyproj
    from geopy.distance import geodesic
    GIS_AVAILABLE = True
except ImportError:
    GIS_AVAILABLE = False
    warnings.warn("GIS libraries not available. Geographic features will be limited.")

# AsyncPG for async operations
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    warnings.warn("AsyncPG not available. Async database operations will be limited.")

# SQLAlchemy for ORM operations
try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.dialects.postgresql import UUID, JSONB
    from geoalchemy2 import Geometry, Geography
    from geoalchemy2.functions import ST_Distance, ST_Within, ST_Contains
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    warnings.warn("SQLAlchemy/GeoAlchemy2 not available. ORM features will be limited.")

warnings.filterwarnings('ignore')

class DatabaseType(Enum):
    POSTGRESQL = "postgresql"
    TIMESCALEDB = "timescaledb"
    POSTGIS = "postgis"

class DataRetentionPolicy(Enum):
    REAL_TIME = timedelta(hours=24)
    SHORT_TERM = timedelta(days=30)
    MEDIUM_TERM = timedelta(days=365)
    LONG_TERM = timedelta(days=3650)  # 10 years

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "prometheus_geo"
    username: str = "prometheus_user"
    password: str = "prometheus_password"
    max_connections: int = 20
    enable_postgis: bool = True
    enable_timescaledb: bool = True
    ssl_mode: str = "prefer"
    
    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string"""
        return (f"postgresql://{self.username}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}")
    
    def get_async_connection_string(self) -> str:
        """Get AsyncPG connection string"""
        return (f"postgresql://{self.username}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}")

# SQLAlchemy Models
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class GeologicalEvent(Base):
        """Geological event model"""
        __tablename__ = 'geological_events'
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        event_id = Column(String(255), unique=True, nullable=False)
        event_type = Column(String(100), nullable=False)
        timestamp = Column(DateTime, nullable=False)
        source_id = Column(String(255), nullable=False)
        location = Column(Geography('POINTZ', srid=4326))  # WGS84 with elevation
        magnitude = Column(Float)
        depth = Column(Float)
        data = Column(JSONB)
        metadata = Column(JSONB)
        created_at = Column(DateTime, default=datetime.utcnow)
    
    class SeismicStation(Base):
        """Seismic monitoring station model"""
        __tablename__ = 'seismic_stations'
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        station_id = Column(String(50), unique=True, nullable=False)
        name = Column(String(255), nullable=False)
        location = Column(Geography('POINTZ', srid=4326))
        elevation = Column(Float)
        network = Column(String(50))
        instrument_type = Column(String(100))
        sampling_rate = Column(Float)
        active = Column(Boolean, default=True)
        metadata = Column(JSONB)
        created_at = Column(DateTime, default=datetime.utcnow)
    
    class GPSStation(Base):
        """GPS monitoring station model"""
        __tablename__ = 'gps_stations'
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        station_id = Column(String(50), unique=True, nullable=False)
        name = Column(String(255), nullable=False)
        location = Column(Geography('POINTZ', srid=4326))
        reference_frame = Column(String(50), default='ITRF2014')
        monument_type = Column(String(100))
        receiver_type = Column(String(100))
        antenna_type = Column(String(100))
        active = Column(Boolean, default=True)
        metadata = Column(JSONB)
        created_at = Column(DateTime, default=datetime.utcnow)
    
    class DeformationMeasurement(Base):
        """Deformation measurement time series"""
        __tablename__ = 'deformation_measurements'
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        station_id = Column(String(50), nullable=False)
        timestamp = Column(DateTime, nullable=False)
        displacement_x = Column(Float)  # East component (mm)
        displacement_y = Column(Float)  # North component (mm) 
        displacement_z = Column(Float)  # Up component (mm)
        uncertainty_x = Column(Float)
        uncertainty_y = Column(Float)
        uncertainty_z = Column(Float)
        processing_method = Column(String(100))
        quality_flag = Column(String(20))
        metadata = Column(JSONB)
        created_at = Column(DateTime, default=datetime.utcnow)
    
    class GeologicalStructure(Base):
        """Geological structure model"""
        __tablename__ = 'geological_structures'
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        structure_id = Column(String(255), unique=True, nullable=False)
        structure_type = Column(String(100), nullable=False)  # fault, fold, etc.
        geometry = Column(Geography('GEOMETRYZ', srid=4326))  # Line/Polygon with elevation
        strike = Column(Float)  # degrees
        dip = Column(Float)    # degrees
        age = Column(Float)    # Ma (millions of years)
        lithology = Column(String(255))
        properties = Column(JSONB)
        metadata = Column(JSONB)
        created_at = Column(DateTime, default=datetime.utcnow)
    
    class VolcanicEvent(Base):
        """Volcanic event model"""
        __tablename__ = 'volcanic_events'
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        volcano_id = Column(String(100), nullable=False)
        event_type = Column(String(100), nullable=False)  # eruption, gas, thermal, etc.
        timestamp = Column(DateTime, nullable=False)
        location = Column(Geography('POINTZ', srid=4326))
        vei = Column(Integer)  # Volcanic Explosivity Index
        magnitude = Column(Float)
        duration = Column(Float)  # minutes
        properties = Column(JSONB)
        metadata = Column(JSONB)
        created_at = Column(DateTime, default=datetime.utcnow)

class PostgreSQLGISManager:
    """PostgreSQL with PostGIS integration manager"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connection_pool = None
        self.engine = None
        self.session_maker = None
        
        if POSTGRES_AVAILABLE:
            self._setup_connection_pool()
        
        if SQLALCHEMY_AVAILABLE:
            self._setup_sqlalchemy()
    
    def _setup_connection_pool(self):
        """Setup PostgreSQL connection pool"""
        try:
            self.connection_pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                sslmode=self.config.ssl_mode
            )
            
            # Test connection and setup extensions
            with self._get_connection() as conn:
                self._setup_database_extensions(conn)
            
            self.logger.info("PostgreSQL connection pool initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup PostgreSQL connection pool: {e}")
    
    def _setup_sqlalchemy(self):
        """Setup SQLAlchemy engine and session"""
        try:
            self.engine = create_engine(
                self.config.get_connection_string(),
                pool_size=self.config.max_connections,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600
            )
            
            self.session_maker = sessionmaker(bind=self.engine)
            
            # Create tables
            if hasattr(self, '_create_tables'):
                Base.metadata.create_all(self.engine)
            
            self.logger.info("SQLAlchemy engine initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup SQLAlchemy: {e}")
    
    def _setup_database_extensions(self, conn):
        """Setup required PostgreSQL extensions"""
        extensions = []
        
        if self.config.enable_postgis:
            extensions.extend(['postgis', 'postgis_topology'])
        
        if self.config.enable_timescaledb:
            extensions.append('timescaledb')
        
        with conn.cursor() as cursor:
            for ext in extensions:
                try:
                    cursor.execute(f"CREATE EXTENSION IF NOT EXISTS {ext};")
                    self.logger.info(f"Enabled extension: {ext}")
                except Exception as e:
                    self.logger.warning(f"Failed to enable extension {ext}: {e}")
            
            conn.commit()
    
    @asynccontextmanager
    async def _get_connection(self):
        """Get database connection from pool"""
        if not self.connection_pool:
            raise RuntimeError("Connection pool not available")
        
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)
    
    @asynccontextmanager
    async def get_session(self) -> Session:
        """Get SQLAlchemy session"""
        if not self.session_maker:
            raise RuntimeError("SQLAlchemy not available")
        
        session = self.session_maker()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    async def create_tables(self):
        """Create database tables"""
        if not SQLALCHEMY_AVAILABLE or not self.engine:
            return
        
        try:
            # Create tables
            Base.metadata.create_all(self.engine)
            
            # Setup TimescaleDB hypertables
            if self.config.enable_timescaledb:
                await self._setup_timescale_hypertables()
            
            # Create indexes
            await self._create_indexes()
            
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
    
    async def _setup_timescale_hypertables(self):
        """Setup TimescaleDB hypertables for time-series data"""
        hypertables = [
            ('geological_events', 'timestamp'),
            ('deformation_measurements', 'timestamp'),
            ('volcanic_events', 'timestamp')
        ]
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                for table_name, time_column in hypertables:
                    try:
                        # Create hypertable
                        cursor.execute(f"""
                            SELECT create_hypertable('{table_name}', '{time_column}',
                                                    if_not_exists => TRUE);
                        """)
                        
                        # Set retention policy
                        cursor.execute(f"""
                            SELECT add_retention_policy('{table_name}', INTERVAL '2 years',
                                                       if_not_exists => TRUE);
                        """)
                        
                        self.logger.info(f"Created hypertable: {table_name}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create hypertable {table_name}: {e}")
                
                conn.commit()
    
    async def _create_indexes(self):
        """Create database indexes for performance"""
        indexes = [
            ("idx_geological_events_timestamp", "geological_events", "timestamp"),
            ("idx_geological_events_type", "geological_events", "event_type"),
            ("idx_geological_events_source", "geological_events", "source_id"),
            ("idx_geological_events_location", "geological_events", "location USING GIST"),
            ("idx_deformation_station", "deformation_measurements", "station_id"),
            ("idx_deformation_timestamp", "deformation_measurements", "timestamp"),
            ("idx_seismic_stations_location", "seismic_stations", "location USING GIST"),
            ("idx_gps_stations_location", "gps_stations", "location USING GIST"),
            ("idx_geological_structures_geometry", "geological_structures", "geometry USING GIST"),
            ("idx_volcanic_events_timestamp", "volcanic_events", "timestamp"),
            ("idx_volcanic_events_location", "volcanic_events", "location USING GIST")
        ]
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                for index_name, table, column in indexes:
                    try:
                        cursor.execute(f"""
                            CREATE INDEX IF NOT EXISTS {index_name} 
                            ON {table} ({column});
                        """)
                        self.logger.debug(f"Created index: {index_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to create index {index_name}: {e}")
                
                conn.commit()
    
    async def insert_geological_event(self, event_data: Dict[str, Any]) -> str:
        """Insert geological event"""
        if not SQLALCHEMY_AVAILABLE:
            return await self._insert_event_raw(event_data)
        
        try:
            async with self.get_session() as session:
                # Create geometry from location
                location_geom = None
                if event_data.get('location'):
                    lat, lon, elev = event_data['location']
                    location_geom = f"SRID=4326;POINTZ({lon} {lat} {elev})"
                
                event = GeologicalEvent(
                    event_id=event_data['event_id'],
                    event_type=event_data['event_type'],
                    timestamp=event_data['timestamp'],
                    source_id=event_data['source_id'],
                    location=location_geom,
                    magnitude=event_data.get('magnitude'),
                    depth=event_data.get('depth'),
                    data=event_data.get('data', {}),
                    metadata=event_data.get('metadata', {})
                )
                
                session.add(event)
                session.flush()
                
                return str(event.id)
                
        except Exception as e:
            self.logger.error(f"Failed to insert geological event: {e}")
            raise
    
    async def _insert_event_raw(self, event_data: Dict[str, Any]) -> str:
        """Insert event using raw SQL"""
        event_id = str(uuid.uuid4())
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Create location geometry
                location_wkt = None
                if event_data.get('location'):
                    lat, lon, elev = event_data['location']
                    location_wkt = f"SRID=4326;POINTZ({lon} {lat} {elev})"
                
                cursor.execute("""
                    INSERT INTO geological_events 
                    (id, event_id, event_type, timestamp, source_id, location, 
                     magnitude, depth, data, metadata)
                    VALUES (%s, %s, %s, %s, %s, ST_GeomFromText(%s), %s, %s, %s, %s)
                """, (
                    event_id,
                    event_data['event_id'],
                    event_data['event_type'],
                    event_data['timestamp'],
                    event_data['source_id'],
                    location_wkt,
                    event_data.get('magnitude'),
                    event_data.get('depth'),
                    json.dumps(event_data.get('data', {})),
                    json.dumps(event_data.get('metadata', {}))
                ))
                
                conn.commit()
        
        return event_id
    
    async def insert_deformation_measurement(self, measurement_data: Dict[str, Any]) -> str:
        """Insert deformation measurement"""
        try:
            async with self.get_session() as session:
                measurement = DeformationMeasurement(
                    station_id=measurement_data['station_id'],
                    timestamp=measurement_data['timestamp'],
                    displacement_x=measurement_data.get('displacement_x'),
                    displacement_y=measurement_data.get('displacement_y'),
                    displacement_z=measurement_data.get('displacement_z'),
                    uncertainty_x=measurement_data.get('uncertainty_x'),
                    uncertainty_y=measurement_data.get('uncertainty_y'),
                    uncertainty_z=measurement_data.get('uncertainty_z'),
                    processing_method=measurement_data.get('processing_method'),
                    quality_flag=measurement_data.get('quality_flag', 'GOOD'),
                    metadata=measurement_data.get('metadata', {})
                )
                
                session.add(measurement)
                session.flush()
                
                return str(measurement.id)
                
        except Exception as e:
            self.logger.error(f"Failed to insert deformation measurement: {e}")
            raise
    
    async def query_events_by_location(self, center_lat: float, center_lon: float, 
                                     radius_km: float, 
                                     start_time: datetime = None,
                                     end_time: datetime = None) -> List[Dict[str, Any]]:
        """Query events by geographic location"""
        try:
            async with self.get_session() as session:
                # Create center point
                center_point = f"SRID=4326;POINT({center_lon} {center_lat})"
                
                query = session.query(GeologicalEvent).filter(
                    ST_Distance(
                        GeologicalEvent.location,
                        center_point
                    ) <= radius_km * 1000  # Convert km to meters
                )
                
                # Add time filters
                if start_time:
                    query = query.filter(GeologicalEvent.timestamp >= start_time)
                if end_time:
                    query = query.filter(GeologicalEvent.timestamp <= end_time)
                
                results = query.all()
                
                # Convert to dictionaries
                events = []
                for event in results:
                    event_dict = {
                        'id': str(event.id),
                        'event_id': event.event_id,
                        'event_type': event.event_type,
                        'timestamp': event.timestamp.isoformat(),
                        'source_id': event.source_id,
                        'magnitude': event.magnitude,
                        'depth': event.depth,
                        'data': event.data,
                        'metadata': event.metadata
                    }
                    
                    # Extract location
                    if event.location:
                        # This would need proper geometry extraction
                        event_dict['location'] = self._extract_location(event.location)
                    
                    events.append(event_dict)
                
                return events
                
        except Exception as e:
            self.logger.error(f"Failed to query events by location: {e}")
            return []
    
    def _extract_location(self, geometry) -> Optional[Tuple[float, float, float]]:
        """Extract location from PostGIS geometry"""
        # This would need proper implementation with PostGIS functions
        # For now, return None
        return None
    
    async def query_deformation_time_series(self, station_id: str, 
                                          start_time: datetime, 
                                          end_time: datetime) -> pd.DataFrame:
        """Query deformation time series data"""
        try:
            async with self.get_session() as session:
                query = session.query(DeformationMeasurement).filter(
                    DeformationMeasurement.station_id == station_id,
                    DeformationMeasurement.timestamp >= start_time,
                    DeformationMeasurement.timestamp <= end_time
                ).order_by(DeformationMeasurement.timestamp)
                
                results = query.all()
                
                # Convert to DataFrame
                data = []
                for measurement in results:
                    data.append({
                        'timestamp': measurement.timestamp,
                        'displacement_x': measurement.displacement_x,
                        'displacement_y': measurement.displacement_y,
                        'displacement_z': measurement.displacement_z,
                        'uncertainty_x': measurement.uncertainty_x,
                        'uncertainty_y': measurement.uncertainty_y,
                        'uncertainty_z': measurement.uncertainty_z,
                        'quality_flag': measurement.quality_flag
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            self.logger.error(f"Failed to query deformation time series: {e}")
            return pd.DataFrame()
    
    async def create_spatial_indexes(self):
        """Create spatial indexes for geographic queries"""
        if not GIS_AVAILABLE:
            return
        
        spatial_indexes = [
            ("geological_events", "location"),
            ("seismic_stations", "location"),
            ("gps_stations", "location"),
            ("geological_structures", "geometry"),
            ("volcanic_events", "location")
        ]
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                for table, column in spatial_indexes:
                    try:
                        cursor.execute(f"""
                            CREATE INDEX IF NOT EXISTS idx_{table}_{column}_gist 
                            ON {table} USING GIST ({column});
                        """)
                        
                        # Also create spatial clustering index
                        cursor.execute(f"""
                            CLUSTER {table} USING idx_{table}_{column}_gist;
                        """)
                        
                        self.logger.info(f"Created spatial index for {table}.{column}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create spatial index for {table}.{column}: {e}")
                
                conn.commit()
    
    async def analyze_spatial_patterns(self, event_type: str, 
                                     time_window: timedelta) -> Dict[str, Any]:
        """Analyze spatial patterns in geological events"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - time_window
            
            async with self.get_session() as session:
                # Query events in time window
                events = session.query(GeologicalEvent).filter(
                    GeologicalEvent.event_type == event_type,
                    GeologicalEvent.timestamp >= start_time,
                    GeologicalEvent.timestamp <= end_time,
                    GeologicalEvent.location.isnot(None)
                ).all()
                
                if not events:
                    return {'event_count': 0}
                
                # Spatial analysis would go here
                analysis = {
                    'event_count': len(events),
                    'time_window': time_window.total_seconds(),
                    'spatial_distribution': self._calculate_spatial_distribution(events),
                    'temporal_pattern': self._calculate_temporal_pattern(events)
                }
                
                return analysis
                
        except Exception as e:
            self.logger.error(f"Failed to analyze spatial patterns: {e}")
            return {'error': str(e)}
    
    def _calculate_spatial_distribution(self, events) -> Dict[str, Any]:
        """Calculate spatial distribution statistics"""
        # This would implement spatial clustering and distribution analysis
        return {
            'centroid': None,
            'spread': None,
            'clusters': []
        }
    
    def _calculate_temporal_pattern(self, events) -> Dict[str, Any]:
        """Calculate temporal pattern statistics"""
        timestamps = [event.timestamp for event in events]
        
        if len(timestamps) < 2:
            return {}
        
        # Calculate time intervals
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        return {
            'mean_interval': np.mean(intervals),
            'std_interval': np.std(intervals),
            'min_interval': np.min(intervals),
            'max_interval': np.max(intervals)
        }

class TimeScaleDBManager:
    """TimescaleDB-specific functionality for time-series data"""
    
    def __init__(self, postgres_manager: PostgreSQLGISManager):
        self.postgres_manager = postgres_manager
        self.logger = logging.getLogger(__name__)
    
    async def create_continuous_aggregates(self):
        """Create continuous aggregates for real-time dashboards"""
        aggregates = [
            {
                'name': 'hourly_event_summary',
                'table': 'geological_events',
                'time_column': 'timestamp',
                'interval': '1 hour'
            },
            {
                'name': 'daily_deformation_summary',
                'table': 'deformation_measurements',
                'time_column': 'timestamp',
                'interval': '1 day'
            }
        ]
        
        with self.postgres_manager._get_connection() as conn:
            with conn.cursor() as cursor:
                for agg in aggregates:
                    try:
                        cursor.execute(f"""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS {agg['name']}
                            WITH (timescaledb.continuous) AS
                            SELECT 
                                time_bucket('{agg['interval']}', {agg['time_column']}) AS bucket,
                                COUNT(*) as event_count,
                                AVG(magnitude) as avg_magnitude,
                                MAX(magnitude) as max_magnitude,
                                MIN(magnitude) as min_magnitude
                            FROM {agg['table']}
                            WHERE magnitude IS NOT NULL
                            GROUP BY bucket;
                        """)
                        
                        # Add refresh policy
                        cursor.execute(f"""
                            SELECT add_continuous_aggregate_policy('{agg['name']}',
                                start_offset => INTERVAL '3 hours',
                                end_offset => INTERVAL '1 hour',
                                schedule_interval => INTERVAL '1 hour');
                        """)
                        
                        self.logger.info(f"Created continuous aggregate: {agg['name']}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create continuous aggregate {agg['name']}: {e}")
                
                conn.commit()
    
    async def get_real_time_metrics(self, time_window: timedelta) -> Dict[str, Any]:
        """Get real-time metrics using TimescaleDB functions"""
        end_time = datetime.utcnow()
        start_time = end_time - time_window
        
        metrics = {}
        
        with self.postgres_manager._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Event rate metrics
                cursor.execute("""
                    SELECT 
                        time_bucket('5 minutes', timestamp) AS bucket,
                        COUNT(*) as event_count,
                        AVG(magnitude) as avg_magnitude
                    FROM geological_events
                    WHERE timestamp >= %s AND timestamp <= %s
                      AND magnitude IS NOT NULL
                    GROUP BY bucket
                    ORDER BY bucket;
                """, (start_time, end_time))
                
                metrics['event_rate'] = cursor.fetchall()
                
                # Deformation trends
                cursor.execute("""
                    SELECT 
                        station_id,
                        LAST(displacement_x, timestamp) - FIRST(displacement_x, timestamp) AS dx_trend,
                        LAST(displacement_y, timestamp) - FIRST(displacement_y, timestamp) AS dy_trend,
                        LAST(displacement_z, timestamp) - FIRST(displacement_z, timestamp) AS dz_trend
                    FROM deformation_measurements
                    WHERE timestamp >= %s AND timestamp <= %s
                    GROUP BY station_id;
                """, (start_time, end_time))
                
                metrics['deformation_trends'] = cursor.fetchall()
        
        return metrics

# Example usage and testing
async def example_database_integration():
    """Example of using database integration"""
    
    print("=== Database Integration Example ===")
    
    # Initialize database manager
    config = DatabaseConfig(
        host="localhost",
        database="prometheus_geo_test",
        username="postgres",
        password="password"
    )
    
    db_manager = PostgreSQLGISManager(config)
    timescale_manager = TimeScaleDBManager(db_manager)
    
    # Create tables
    await db_manager.create_tables()
    
    # Create TimescaleDB continuous aggregates
    await timescale_manager.create_continuous_aggregates()
    
    # Insert sample geological events
    sample_events = []
    for i in range(10):
        event_data = {
            'event_id': f'test_event_{i}',
            'event_type': 'earthquake',
            'timestamp': datetime.utcnow() - timedelta(hours=i),
            'source_id': 'test_seismic_station',
            'location': (35.0 + i * 0.01, -118.0 + i * 0.01, 0.0),
            'magnitude': 4.0 + i * 0.2,
            'depth': 10.0 + i,
            'data': {'processed_by': 'prometheus_ai'},
            'metadata': {'quality': 'high'}
        }
        
        event_id = await db_manager.insert_geological_event(event_data)
        print(f"Inserted event {i}: {event_id}")
    
    # Query events by location
    events = await db_manager.query_events_by_location(
        center_lat=35.05,
        center_lon=-117.95,
        radius_km=10.0,
        start_time=datetime.utcnow() - timedelta(hours=24)
    )
    
    print(f"Found {len(events)} events within 10km of query location")
    
    # Insert sample deformation measurements
    for i in range(24):  # 24 hours of hourly data
        measurement_data = {
            'station_id': 'GPS_STATION_001',
            'timestamp': datetime.utcnow() - timedelta(hours=i),
            'displacement_x': np.random.normal(0, 2),  # mm
            'displacement_y': np.random.normal(0, 2),  # mm
            'displacement_z': np.random.normal(0, 5),  # mm
            'uncertainty_x': 1.0,
            'uncertainty_y': 1.0,
            'uncertainty_z': 2.0,
            'processing_method': 'PPP',
            'quality_flag': 'GOOD'
        }
        
        await db_manager.insert_deformation_measurement(measurement_data)
    
    # Query deformation time series
    time_series = await db_manager.query_deformation_time_series(
        'GPS_STATION_001',
        datetime.utcnow() - timedelta(hours=24),
        datetime.utcnow()
    )
    
    print(f"Retrieved {len(time_series)} deformation measurements")
    if not time_series.empty:
        print(f"Mean displacement: X={time_series['displacement_x'].mean():.2f}mm, "
              f"Y={time_series['displacement_y'].mean():.2f}mm, "
              f"Z={time_series['displacement_z'].mean():.2f}mm")
    
    # Get real-time metrics
    metrics = await timescale_manager.get_real_time_metrics(timedelta(hours=12))
    print(f"Real-time metrics retrieved: {len(metrics)} metric types")
    
    print("Database integration example completed!")

if __name__ == "__main__":
    print("=== PostgreSQL GIS and TimescaleDB Integration ===")
    print(f"Available features:")
    print(f"  PostgreSQL: {POSTGRES_AVAILABLE}")
    print(f"  GIS Libraries: {GIS_AVAILABLE}")
    print(f"  AsyncPG: {ASYNCPG_AVAILABLE}")
    print(f"  SQLAlchemy: {SQLALCHEMY_AVAILABLE}")
    
    # Run example
    asyncio.run(example_database_integration())