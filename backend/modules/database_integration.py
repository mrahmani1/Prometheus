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
        """Extract location from PostGIS geometry using advanced spatial processing"""
        if not geometry or not GIS_AVAILABLE:
            return None
        
        try:
            # Handle different geometry input types
            if hasattr(geometry, '__geo_interface__'):
                # GeoAlchemy2 geometry object
                geom_dict = geometry.__geo_interface__
            elif isinstance(geometry, str):
                # WKT string
                try:
                    from shapely.wkt import loads as wkt_loads
                    shapely_geom = wkt_loads(geometry)
                    geom_dict = shapely_geom.__geo_interface__
                except Exception:
                    return None
            elif hasattr(geometry, 'coordinates'):
                # GeoJSON-like object
                geom_dict = geometry
            else:
                return None
            
            # Extract coordinates based on geometry type
            geom_type = geom_dict.get('type', '').lower()
            coordinates = geom_dict.get('coordinates', [])
            
            if not coordinates:
                return None
            
            if geom_type == 'point':
                # Point geometry: [lon, lat] or [lon, lat, elevation]
                if len(coordinates) >= 2:
                    lon, lat = coordinates[0], coordinates[1]
                    elev = coordinates[2] if len(coordinates) > 2 else 0.0
                    return (lat, lon, elev)
            
            elif geom_type in ['linestring', 'multilinestring']:
                # For lines, extract centroid
                if geom_type == 'linestring':
                    coords = coordinates
                else:
                    coords = coordinates[0] if coordinates else []
                
                if coords:
                    # Calculate weighted centroid along the line
                    total_length = 0.0
                    weighted_lon = 0.0
                    weighted_lat = 0.0
                    weighted_elev = 0.0
                    
                    for i in range(len(coords) - 1):
                        p1 = coords[i]
                        p2 = coords[i + 1]
                        
                        # Calculate segment length using haversine formula
                        if len(p1) >= 2 and len(p2) >= 2:
                            from math import radians, cos, sin, asin, sqrt
                            
                            lon1, lat1 = radians(p1[0]), radians(p1[1])
                            lon2, lat2 = radians(p2[0]), radians(p2[1])
                            
                            # Haversine formula
                            dlat = lat2 - lat1
                            dlon = lon2 - lon1
                            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                            segment_length = 2 * asin(sqrt(a)) * 6371000  # Earth radius in meters
                            
                            if segment_length > 0:
                                mid_lon = (p1[0] + p2[0]) / 2
                                mid_lat = (p1[1] + p2[1]) / 2
                                mid_elev = ((p1[2] if len(p1) > 2 else 0) + 
                                           (p2[2] if len(p2) > 2 else 0)) / 2
                                
                                weighted_lon += mid_lon * segment_length
                                weighted_lat += mid_lat * segment_length
                                weighted_elev += mid_elev * segment_length
                                total_length += segment_length
                    
                    if total_length > 0:
                        return (weighted_lat / total_length, 
                               weighted_lon / total_length,
                               weighted_elev / total_length)
            
            elif geom_type in ['polygon', 'multipolygon']:
                # For polygons, calculate area-weighted centroid
                if geom_type == 'polygon':
                    rings = coordinates
                else:
                    rings = coordinates[0] if coordinates else []
                
                if rings and rings[0]:  # Exterior ring
                    exterior_coords = rings[0]
                    
                    # Use shoelace formula for centroid calculation
                    if len(exterior_coords) >= 4:  # Minimum for a closed polygon
                        area = 0.0
                        cx = 0.0
                        cy = 0.0
                        cz = 0.0
                        
                        n = len(exterior_coords) - 1  # Exclude closing point
                        
                        for i in range(n):
                            j = (i + 1) % n
                            xi, yi = exterior_coords[i][0], exterior_coords[i][1]
                            xj, yj = exterior_coords[j][0], exterior_coords[j][1]
                            
                            cross = xi * yj - xj * yi
                            area += cross
                            cx += (xi + xj) * cross
                            cy += (yi + yj) * cross
                            
                            # Handle elevation
                            zi = exterior_coords[i][2] if len(exterior_coords[i]) > 2 else 0
                            zj = exterior_coords[j][2] if len(exterior_coords[j]) > 2 else 0
                            cz += (zi + zj) * cross
                        
                        area /= 2.0
                        if abs(area) > 1e-10:  # Avoid division by zero
                            cx /= (6.0 * area)
                            cy /= (6.0 * area)
                            cz /= (6.0 * area)
                            return (cy, cx, cz)  # Return as (lat, lon, elev)
            
            elif geom_type in ['multipoint', 'geometrycollection']:
                # For collections, calculate average of all points
                all_points = []
                
                if geom_type == 'multipoint':
                    all_points = coordinates
                else:
                    # Extract points from geometry collection
                    geometries = geom_dict.get('geometries', [])
                    for geom in geometries:
                        if geom.get('type', '').lower() == 'point':
                            point_coords = geom.get('coordinates', [])
                            if len(point_coords) >= 2:
                                all_points.append(point_coords)
                
                if all_points:
                    # Calculate arithmetic mean
                    mean_lon = sum(p[0] for p in all_points) / len(all_points)
                    mean_lat = sum(p[1] for p in all_points) / len(all_points)
                    mean_elev = sum(p[2] if len(p) > 2 else 0 for p in all_points) / len(all_points)
                    return (mean_lat, mean_lon, mean_elev)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract location from geometry: {e}")
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
        """Calculate comprehensive spatial distribution statistics with advanced clustering analysis"""
        if not events or not GIS_AVAILABLE:
            return {
                'centroid': None,
                'spread': None,
                'clusters': [],
                'spatial_metrics': {},
                'error': 'No events or GIS not available'
            }
        
        try:
            # Extract locations from events
            locations = []
            magnitudes = []
            depths = []
            
            for event in events:
                if hasattr(event, 'location') and event.location:
                    location = self._extract_location(event.location)
                    if location:
                        lat, lon, elev = location
                        locations.append([lon, lat, elev])  # [lon, lat, depth/elevation]
                        magnitudes.append(event.magnitude if event.magnitude else 0.0)
                        depths.append(event.depth if event.depth else 0.0)
            
            if not locations:
                return {
                    'centroid': None,
                    'spread': None,
                    'clusters': [],
                    'spatial_metrics': {},
                    'error': 'No valid locations found'
                }
            
            locations = np.array(locations)
            magnitudes = np.array(magnitudes)
            depths = np.array(depths)
            
            # Calculate spatial centroid (weighted by magnitude)
            if np.sum(magnitudes) > 0:
                weights = magnitudes / np.sum(magnitudes)
                centroid_lon = np.sum(locations[:, 0] * weights)
                centroid_lat = np.sum(locations[:, 1] * weights) 
                centroid_elev = np.sum(locations[:, 2] * weights)
            else:
                centroid_lon = np.mean(locations[:, 0])
                centroid_lat = np.mean(locations[:, 1])
                centroid_elev = np.mean(locations[:, 2])
            
            centroid = (centroid_lat, centroid_lon, centroid_elev)
            
            # Calculate spatial spread metrics
            spread_metrics = self._calculate_spatial_spread(locations, magnitudes, centroid)
            
            # Perform clustering analysis
            clustering_results = self._perform_spatial_clustering(locations, magnitudes, depths)
            
            # Calculate spatial autocorrelation (Moran's I)
            morans_i = self._calculate_morans_i(locations, magnitudes)
            
            # Calculate nearest neighbor analysis
            nn_analysis = self._calculate_nearest_neighbor_analysis(locations)
            
            # Calculate spatial hotspots (Getis-Ord Gi*)
            hotspots = self._calculate_hotspots(locations, magnitudes)
            
            # Calculate fractal dimension
            fractal_dim = self._calculate_fractal_dimension(locations)
            
            # Directional analysis
            directional_stats = self._calculate_directional_statistics(locations, centroid)
            
            # Kernel density estimation
            kde_analysis = self._perform_kde_analysis(locations, magnitudes)
            
            return {
                'centroid': centroid,
                'spread': spread_metrics,
                'clusters': clustering_results,
                'spatial_metrics': {
                    'morans_i': morans_i,
                    'nearest_neighbor': nn_analysis,
                    'hotspots': hotspots,
                    'fractal_dimension': fractal_dim,
                    'directional_stats': directional_stats,
                    'kde_analysis': kde_analysis,
                    'total_events': len(events),
                    'valid_locations': len(locations),
                    'spatial_extent': {
                        'min_lon': float(np.min(locations[:, 0])),
                        'max_lon': float(np.max(locations[:, 0])),
                        'min_lat': float(np.min(locations[:, 1])),
                        'max_lat': float(np.max(locations[:, 1])),
                        'min_elev': float(np.min(locations[:, 2])),
                        'max_elev': float(np.max(locations[:, 2]))
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate spatial distribution: {e}")
            return {
                'centroid': None,
                'spread': None,
                'clusters': [],
                'spatial_metrics': {},
                'error': str(e)
            }
    
    def _calculate_spatial_spread(self, locations: np.ndarray, magnitudes: np.ndarray, 
                                centroid: Tuple[float, float, float]) -> Dict[str, float]:
        """Calculate various spatial spread metrics"""
        try:
            centroid_lon, centroid_lat, _ = centroid[1], centroid[0], centroid[2]
            
            # Calculate distances from centroid using haversine formula
            distances = []
            for i, (lon, lat, _) in enumerate(locations):
                from math import radians, cos, sin, asin, sqrt
                
                lat1, lon1 = radians(centroid_lat), radians(centroid_lon)
                lat2, lon2 = radians(lat), radians(lon)
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                distance = 2 * asin(sqrt(a)) * 6371000  # Earth radius in meters
                distances.append(distance)
            
            distances = np.array(distances)
            
            # Standard distance (spatial standard deviation)
            if np.sum(magnitudes) > 0:
                weights = magnitudes / np.sum(magnitudes)
                std_distance = np.sqrt(np.sum(weights * distances**2))
            else:
                std_distance = np.std(distances)
            
            # Calculate additional spread metrics
            mean_distance = np.mean(distances)
            median_distance = np.median(distances)
            max_distance = np.max(distances)
            range_distance = np.max(distances) - np.min(distances)
            
            # Calculate spatial dispersion index
            variance_distance = np.var(distances)
            dispersion_index = variance_distance / mean_distance if mean_distance > 0 else 0
            
            # Calculate convex hull area
            try:
                from scipy.spatial import ConvexHull
                if len(locations) >= 3:
                    hull = ConvexHull(locations[:, :2])  # Use only lon, lat
                    hull_area = hull.volume  # In 2D, volume is area
                else:
                    hull_area = 0
            except:
                hull_area = 0
            
            return {
                'standard_distance': float(std_distance),
                'mean_distance': float(mean_distance),
                'median_distance': float(median_distance),
                'max_distance': float(max_distance),
                'range_distance': float(range_distance),
                'dispersion_index': float(dispersion_index),
                'convex_hull_area': float(hull_area)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate spatial spread: {e}")
            return {}
    
    def _perform_spatial_clustering(self, locations: np.ndarray, magnitudes: np.ndarray, 
                                  depths: np.ndarray) -> List[Dict[str, Any]]:
        """Perform advanced spatial clustering analysis"""
        try:
            clusters = []
            
            # K-means clustering
            try:
                from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
                from sklearn.metrics import silhouette_score
                
                # Determine optimal number of clusters using elbow method
                max_clusters = min(10, len(locations) // 2)
                if max_clusters >= 2:
                    inertias = []
                    k_range = range(2, max_clusters + 1)
                    
                    for k in k_range:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(locations[:, :2])  # Use only lon, lat
                        inertias.append(kmeans.inertia_)
                    
                    # Find elbow point
                    if len(inertias) >= 2:
                        # Simple elbow detection
                        diffs = np.diff(inertias)
                        optimal_k = np.argmax(diffs[:-1] - diffs[1:]) + 2 if len(diffs) > 1 else 2
                        optimal_k = min(optimal_k, max_clusters)
                        
                        # Perform K-means with optimal k
                        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(locations[:, :2])
                        silhouette_avg = silhouette_score(locations[:, :2], labels)
                        
                        cluster_centers = kmeans.cluster_centers_
                        for i, center in enumerate(cluster_centers):
                            mask = labels == i
                            cluster_points = locations[mask]
                            cluster_mags = magnitudes[mask]
                            cluster_depths = depths[mask]
                            
                            clusters.append({
                                'algorithm': 'k-means',
                                'cluster_id': int(i),
                                'center': (float(center[1]), float(center[0])),  # (lat, lon)
                                'size': int(np.sum(mask)),
                                'mean_magnitude': float(np.mean(cluster_mags)) if len(cluster_mags) > 0 else 0,
                                'mean_depth': float(np.mean(cluster_depths)) if len(cluster_depths) > 0 else 0,
                                'silhouette_score': float(silhouette_avg)
                            })
                
                # DBSCAN clustering (density-based)
                eps_values = [0.01, 0.05, 0.1, 0.2]  # Different epsilon values to try
                best_dbscan = None
                best_score = -1
                
                for eps in eps_values:
                    dbscan = DBSCAN(eps=eps, min_samples=max(2, len(locations) // 10))
                    labels = dbscan.fit_predict(locations[:, :2])
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    if n_clusters > 0 and n_clusters < len(locations) // 2:
                        try:
                            score = silhouette_score(locations[:, :2], labels)
                            if score > best_score:
                                best_score = score
                                best_dbscan = (labels, n_clusters, n_noise, eps)
                        except:
                            pass
                
                if best_dbscan:
                    labels, n_clusters, n_noise, eps = best_dbscan
                    for cluster_id in set(labels):
                        if cluster_id != -1:  # Skip noise points
                            mask = labels == cluster_id
                            cluster_points = locations[mask]
                            cluster_mags = magnitudes[mask]
                            cluster_depths = depths[mask]
                            
                            center_lon = np.mean(cluster_points[:, 0])
                            center_lat = np.mean(cluster_points[:, 1])
                            
                            clusters.append({
                                'algorithm': 'dbscan',
                                'cluster_id': int(cluster_id),
                                'center': (float(center_lat), float(center_lon)),
                                'size': int(np.sum(mask)),
                                'mean_magnitude': float(np.mean(cluster_mags)) if len(cluster_mags) > 0 else 0,
                                'mean_depth': float(np.mean(cluster_depths)) if len(cluster_depths) > 0 else 0,
                                'eps': float(eps),
                                'noise_points': int(n_noise)
                            })
                
            except ImportError:
                self.logger.warning("Scikit-learn not available for clustering analysis")
            except Exception as e:
                self.logger.warning(f"Clustering analysis failed: {e}")
            
            return clusters
            
        except Exception as e:
            self.logger.warning(f"Failed to perform spatial clustering: {e}")
            return []
    
    def _calculate_morans_i(self, locations: np.ndarray, magnitudes: np.ndarray) -> Dict[str, float]:
        """Calculate Moran's I spatial autocorrelation index"""
        try:
            n = len(locations)
            if n < 3:
                return {'morans_i': 0.0, 'expected_i': 0.0, 'z_score': 0.0, 'p_value': 1.0}
            
            # Create spatial weights matrix (inverse distance)
            W = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # Calculate distance
                        from math import radians, cos, sin, asin, sqrt
                        lat1, lon1 = radians(locations[i, 1]), radians(locations[i, 0])
                        lat2, lon2 = radians(locations[j, 1]), radians(locations[j, 0])
                        
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                        distance = 2 * asin(sqrt(a)) * 6371000  # meters
                        
                        if distance > 0:
                            W[i, j] = 1.0 / distance
            
            # Row normalize weights
            row_sums = np.sum(W, axis=1)
            for i in range(n):
                if row_sums[i] > 0:
                    W[i, :] = W[i, :] / row_sums[i]
            
            # Calculate Moran's I
            x = magnitudes - np.mean(magnitudes)
            numerator = 0.0
            denominator = np.sum(x**2)
            
            for i in range(n):
                for j in range(n):
                    numerator += W[i, j] * x[i] * x[j]
            
            if denominator > 0:
                morans_i = (n * numerator) / (np.sum(W) * denominator)
            else:
                morans_i = 0.0
            
            # Expected value and variance
            expected_i = -1.0 / (n - 1)
            
            # Advanced z-score calculation with proper Moran's I variance
            if n > 3:
                # Calculate proper variance using the complete Moran's I formula
                # Variance of Moran's I under null hypothesis
                w_sum = np.sum(weights_matrix)
                w_sum_squared = np.sum(weights_matrix ** 2)
                
                # Sum of squared row sums and column sums
                w_row_sums = np.sum(weights_matrix, axis=1)
                w_col_sums = np.sum(weights_matrix, axis=0)
                s1 = 0.5 * np.sum((weights_matrix + weights_matrix.T) ** 2)
                s2 = np.sum((w_row_sums + w_col_sums) ** 2)
                
                # Fourth moment of the standardized variable
                if len(values) > 0:
                    mean_val = np.mean(values)
                    centered_vals = values - mean_val
                    variance_val = np.var(values, ddof=0)
                    if variance_val > 0:
                        standardized_vals = centered_vals / np.sqrt(variance_val)
                        b2 = np.sum(standardized_vals ** 4) / n
                    else:
                        b2 = 3.0  # Normal distribution kurtosis
                else:
                    b2 = 3.0
                
                # Calculate variance components
                a = (n - 1) * s1
                b = (2 * b2 - n) * s2
                c = (n - 1) * (n - 2) * (n - 3) * w_sum ** 2
                
                # Complete variance formula for Moran's I
                if c > 0:
                    var_i = (a - b) / c - expected_i ** 2
                    var_i = max(var_i, 1e-10)  # Ensure positive variance
                else:
                    var_i = 2.0 / (n * (n - 1))  # Fallback to simplified
                
                z_score = (morans_i - expected_i) / np.sqrt(var_i) if var_i > 0 else 0.0
                
                # Apply finite sample correction for small n
                if n < 30:
                    correction_factor = np.sqrt((n - 1) / n)
                    z_score *= correction_factor
            else:
                # Fallback for very small samples
                var_i = 2.0 / (n * (n - 1))
                z_score = (morans_i - expected_i) / np.sqrt(var_i) if var_i > 0 else 0.0
            
            # Approximate p-value (two-tailed)
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
            
            return {
                'morans_i': float(morans_i),
                'expected_i': float(expected_i),
                'z_score': float(z_score),
                'p_value': float(p_value)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate Moran's I: {e}")
            return {'morans_i': 0.0, 'expected_i': 0.0, 'z_score': 0.0, 'p_value': 1.0}
    
    def _calculate_nearest_neighbor_analysis(self, locations: np.ndarray) -> Dict[str, float]:
        """Calculate nearest neighbor analysis statistics"""
        try:
            n = len(locations)
            if n < 2:
                return {'nn_index': 1.0, 'mean_nn_distance': 0.0, 'expected_distance': 0.0}
            
            # Calculate nearest neighbor distances
            nn_distances = []
            for i in range(n):
                min_dist = float('inf')
                for j in range(n):
                    if i != j:
                        # Calculate distance
                        from math import radians, cos, sin, asin, sqrt
                        lat1, lon1 = radians(locations[i, 1]), radians(locations[i, 0])
                        lat2, lon2 = radians(locations[j, 1]), radians(locations[j, 0])
                        
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                        distance = 2 * asin(sqrt(a)) * 6371000  # meters
                        
                        if distance < min_dist:
                            min_dist = distance
                
                nn_distances.append(min_dist)
            
            mean_nn_distance = np.mean(nn_distances)
            
            # Calculate expected distance for random distribution
            # Approximate area (bounding box)
            min_lon, max_lon = np.min(locations[:, 0]), np.max(locations[:, 0])
            min_lat, max_lat = np.min(locations[:, 1]), np.max(locations[:, 1])
            
            # Convert to approximate area in square meters
            area_km2 = ((max_lon - min_lon) * 111.32 * np.cos(radians((min_lat + max_lat) / 2))) * \
                      ((max_lat - min_lat) * 110.57)
            area_m2 = area_km2 * 1e6
            
            density = n / area_m2 if area_m2 > 0 else 0
            expected_distance = 0.5 / np.sqrt(density) if density > 0 else 0
            
            # Nearest neighbor index
            nn_index = mean_nn_distance / expected_distance if expected_distance > 0 else 1.0
            
            return {
                'nn_index': float(nn_index),
                'mean_nn_distance': float(mean_nn_distance),
                'expected_distance': float(expected_distance),
                'interpretation': 'clustered' if nn_index < 1.0 else 'dispersed' if nn_index > 1.0 else 'random'
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate nearest neighbor analysis: {e}")
            return {'nn_index': 1.0, 'mean_nn_distance': 0.0, 'expected_distance': 0.0}
    
    def _calculate_hotspots(self, locations: np.ndarray, magnitudes: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate spatial hotspots using Getis-Ord Gi* statistic"""
        try:
            n = len(locations)
            if n < 3:
                return []
            
            hotspots = []
            
            # Calculate Gi* for each location
            for i in range(n):
                # Define neighborhood (within certain distance)
                neighborhood_distance = 50000  # 50km in meters
                neighbors = []
                weights = []
                
                for j in range(n):
                    # Calculate distance
                    from math import radians, cos, sin, asin, sqrt
                    lat1, lon1 = radians(locations[i, 1]), radians(locations[i, 0])
                    lat2, lon2 = radians(locations[j, 1]), radians(locations[j, 0])
                    
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    distance = 2 * asin(sqrt(a)) * 6371000  # meters
                    
                    if distance <= neighborhood_distance:
                        neighbors.append(j)
                        # Inverse distance weighting
                        weight = 1.0 / (distance + 1.0)  # Add 1 to avoid division by zero
                        weights.append(weight)
                
                if len(neighbors) > 1:
                    weights = np.array(weights)
                    weights = weights / np.sum(weights)  # Normalize
                    
                    # Calculate Gi*
                    local_sum = np.sum([weights[k] * magnitudes[neighbors[k]] for k in range(len(neighbors))])
                    global_mean = np.mean(magnitudes)
                    
                    # Simplified Gi* calculation
                    gi_star = (local_sum - global_mean * np.sum(weights)) / np.std(magnitudes)
                    
                    # Determine if it's a hotspot (high values) or coldspot (low values)
                    if abs(gi_star) > 1.96:  # 95% confidence level
                        hotspot_type = 'hot' if gi_star > 0 else 'cold'
                        significance = 'high' if abs(gi_star) > 2.58 else 'moderate'  # 99% vs 95%
                        
                        hotspots.append({
                            'location': (float(locations[i, 1]), float(locations[i, 0])),
                            'gi_star': float(gi_star),
                            'type': hotspot_type,
                            'significance': significance,
                            'local_mean': float(local_sum),
                            'global_mean': float(global_mean),
                            'neighbor_count': len(neighbors)
                        })
            
            return hotspots
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate hotspots: {e}")
            return []
    
    def _calculate_fractal_dimension(self, locations: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            if len(locations) < 10:
                return 0.0
            
            # Normalize coordinates to [0, 1] range
            coords = locations[:, :2].copy()  # Use only lon, lat
            mins = np.min(coords, axis=0)
            maxs = np.max(coords, axis=0)
            ranges = maxs - mins
            
            if np.any(ranges == 0):
                return 0.0
            
            coords = (coords - mins) / ranges
            
            # Box sizes (as fractions of the total range)
            box_sizes = np.logspace(-3, -1, 10)  # From 0.001 to 0.1
            box_counts = []
            
            for box_size in box_sizes:
                # Count boxes containing at least one point
                n_boxes_x = int(1.0 / box_size) + 1
                n_boxes_y = int(1.0 / box_size) + 1
                
                occupied_boxes = set()
                for coord in coords:
                    box_x = int(coord[0] / box_size)
                    box_y = int(coord[1] / box_size)
                    occupied_boxes.add((box_x, box_y))
                
                box_counts.append(len(occupied_boxes))
            
            # Calculate fractal dimension using linear regression
            log_box_sizes = np.log(box_sizes)
            log_box_counts = np.log(box_counts)
            
            # Linear regression
            coeffs = np.polyfit(log_box_sizes, log_box_counts, 1)
            fractal_dimension = -coeffs[0]  # Negative slope
            
            return float(fractal_dimension)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate fractal dimension: {e}")
            return 0.0
    
    def _calculate_directional_statistics(self, locations: np.ndarray, 
                                        centroid: Tuple[float, float, float]) -> Dict[str, Any]:
        """Calculate directional statistics relative to centroid"""
        try:
            if len(locations) < 2:
                return {}
            
            centroid_lat, centroid_lon = centroid[0], centroid[1]
            
            # Calculate bearings from centroid to each point
            bearings = []
            distances = []
            
            for lon, lat, _ in locations:
                from math import atan2, degrees, radians, cos, sin, asin, sqrt
                
                # Calculate bearing
                lat1, lon1 = radians(centroid_lat), radians(centroid_lon)
                lat2, lon2 = radians(lat), radians(lon)
                
                dlon = lon2 - lon1
                y = sin(dlon) * cos(lat2)
                x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
                
                bearing = degrees(atan2(y, x))
                bearing = (bearing + 360) % 360  # Normalize to 0-360
                bearings.append(bearing)
                
                # Calculate distance
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                distance = 2 * asin(sqrt(a)) * 6371000  # meters
                distances.append(distance)
            
            bearings = np.array(bearings)
            distances = np.array(distances)
            
            # Convert bearings to radians for circular statistics
            bearings_rad = np.radians(bearings)
            
            # Calculate circular mean and variance
            cos_bearings = np.cos(bearings_rad)
            sin_bearings = np.sin(bearings_rad)
            
            mean_cos = np.mean(cos_bearings)
            mean_sin = np.mean(sin_bearings)
            
            # Resultant vector length (measure of concentration)
            R = np.sqrt(mean_cos**2 + mean_sin**2)
            
            # Mean direction
            mean_bearing = degrees(atan2(mean_sin, mean_cos))
            mean_bearing = (mean_bearing + 360) % 360
            
            # Circular variance
            circular_variance = 1 - R
            
            # Concentration parameter (von Mises distribution)
            if R < 0.53:
                kappa = 2 * R + R**3 + 5 * R**5 / 6
            elif R < 0.85:
                kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
            else:
                kappa = 1 / (R**3 - 4 * R**2 + 3 * R)
            
            return {
                'mean_bearing': float(mean_bearing),
                'resultant_length': float(R),
                'circular_variance': float(circular_variance),
                'concentration': float(kappa),
                'num_points': len(bearings),
                'mean_distance': float(np.mean(distances)),
                'max_distance': float(np.max(distances))
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate directional statistics: {e}")
            return {}
    
    def _perform_kde_analysis(self, locations: np.ndarray, magnitudes: np.ndarray) -> Dict[str, Any]:
        """Perform kernel density estimation analysis"""
        try:
            if len(locations) < 3:
                return {}
            
            # Grid resolution for KDE
            grid_resolution = 50
            
            # Create grid
            min_lon, max_lon = np.min(locations[:, 0]), np.max(locations[:, 0])
            min_lat, max_lat = np.min(locations[:, 1]), np.max(locations[:, 1])
            
            # Add padding
            lon_range = max_lon - min_lon
            lat_range = max_lat - min_lat
            padding = 0.1  # 10% padding
            
            lon_grid = np.linspace(min_lon - padding * lon_range, 
                                 max_lon + padding * lon_range, grid_resolution)
            lat_grid = np.linspace(min_lat - padding * lat_range,
                                 max_lat + padding * lat_range, grid_resolution)
            
            # Advanced adaptive Gaussian KDE implementation with optimal bandwidth selection
            from scipy.spatial.distance import cdist
            from scipy.optimize import minimize_scalar
            
            # Adaptive bandwidth selection using multiple methods
            if len(coordinates) > 1:
                # Silverman's rule of thumb
                n = len(coordinates)
                coords_array = np.array(coordinates)
                std_x, std_y = np.std(coords_array, axis=0)
                silverman_bw = (4/3)**0.2 * min(std_x, std_y) * n**(-1/6)
                
                # Scott's rule
                scott_bw = n**(-1/6) * np.mean([std_x, std_y])
                
                # Cross-validation bandwidth optimization
                def kde_likelihood(bw):
                    if bw <= 0:
                        return np.inf
                    total_likelihood = 0
                    for i, coord in enumerate(coords_array):
                        # Leave-one-out density estimation
                        other_coords = np.delete(coords_array, i, axis=0)
                        distances = cdist([coord], other_coords, metric='euclidean')[0]
                        kernel_vals = np.exp(-0.5 * (distances / bw)**2) / (bw * np.sqrt(2 * np.pi))
                        density = np.mean(kernel_vals) if len(kernel_vals) > 0 else 1e-10
                        total_likelihood -= np.log(max(density, 1e-10))
                    return total_likelihood
                
                # Optimize bandwidth using cross-validation
                try:
                    cv_result = minimize_scalar(kde_likelihood, 
                                              bounds=(0.01, max(silverman_bw, scott_bw) * 2),
                                              method='bounded')
                    cv_bandwidth = cv_result.x if cv_result.success else silverman_bw
                except:
                    cv_bandwidth = silverman_bw
                
                # Ensemble bandwidth selection (weighted average)
                bandwidth = (0.4 * silverman_bw + 0.3 * scott_bw + 0.3 * cv_bandwidth)
                bandwidth = max(bandwidth, 0.01)  # Ensure minimum bandwidth
            else:
                bandwidth = 0.1  # Fallback for single point
            
            # Initialize high-resolution density grid with optimized computation
            density_grid = np.zeros((grid_resolution, grid_resolution))
            
            for i, lat in enumerate(lat_grid):
                for j, lon in enumerate(lon_grid):
                    density = 0.0
                    for k, (event_lon, event_lat, _) in enumerate(locations):
                        # Gaussian kernel
                        distance_sq = (lon - event_lon)**2 + (lat - event_lat)**2
                        kernel_value = np.exp(-distance_sq / (2 * bandwidth**2))
                        # Weight by magnitude
                        weight = magnitudes[k] if magnitudes[k] > 0 else 1.0
                        density += kernel_value * weight
                    
                    density_grid[i, j] = density
            
            # Find peaks in density
            max_density = np.max(density_grid)
            peak_threshold = 0.7 * max_density
            
            peaks = []
            for i in range(1, grid_resolution - 1):
                for j in range(1, grid_resolution - 1):
                    if (density_grid[i, j] > peak_threshold and
                        density_grid[i, j] > density_grid[i-1, j] and
                        density_grid[i, j] > density_grid[i+1, j] and
                        density_grid[i, j] > density_grid[i, j-1] and
                        density_grid[i, j] > density_grid[i, j+1]):
                        
                        peaks.append({
                            'location': (float(lat_grid[i]), float(lon_grid[j])),
                            'density': float(density_grid[i, j]),
                            'relative_density': float(density_grid[i, j] / max_density)
                        })
            
            # Sort peaks by density
            peaks.sort(key=lambda x: x['density'], reverse=True)
            
            return {
                'bandwidth': float(bandwidth),
                'max_density': float(max_density),
                'num_peaks': len(peaks),
                'peaks': peaks[:10],  # Top 10 peaks
                'grid_resolution': grid_resolution,
                'spatial_extent': {
                    'min_lon': float(min_lon),
                    'max_lon': float(max_lon),
                    'min_lat': float(min_lat),
                    'max_lat': float(max_lat)
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to perform KDE analysis: {e}")
            return {}
    
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