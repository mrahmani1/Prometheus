"""
GIS-Enabled PostgreSQL Operations Module
Comprehensive PostGIS integration for geospatial operations across the Prometheus AI Platform
Advanced spatial indexing, queries, transformations, and analysis for geological datasets
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncpg
from contextlib import asynccontextmanager
import warnings
from pathlib import Path

# Geospatial libraries
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
from shapely import wkt, wkb
import pyproj
from pyproj import CRS, Transformer
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds

# SQLAlchemy with PostGIS
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Index, func, select, and_, or_
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from geoalchemy2 import Geometry, Geography
from geoalchemy2.functions import ST_DWithin, ST_Contains, ST_Intersects, ST_Distance, ST_Area, ST_Transform, ST_Buffer, ST_Union, ST_Centroid, ST_AsGeoJSON, ST_GeomFromText, ST_MakePoint, ST_SetSRID

# Import existing database integration
from backend.core.database_integration import EnhancedDatabaseIntegrator, DatabaseMetrics


class CoordinateSystem(Enum):
    """Standard coordinate systems for geological data"""
    WGS84 = 4326  # World Geodetic System 1984
    WEB_MERCATOR = 3857  # Web Mercator projection
    UTM_NORTH = 32600  # UTM Northern Hemisphere base
    UTM_SOUTH = 32700  # UTM Southern Hemisphere base
    ALBERS_EQUAL_AREA = 3005  # North America Albers Equal Area Conic
    LAMBERT_CONFORMAL = 3347  # Statistics Canada Lambert
    POLAR_STEREOGRAPHIC = 3031  # Antarctic Polar Stereographic


class SpatialIndexType(Enum):
    """Spatial index types for optimization"""
    GIST = "GIST"
    SPGIST = "SP-GIST"
    BRIN = "BRIN"
    BTREE = "BTREE"


class ExportFormat(Enum):
    """Supported geospatial export formats"""
    GEOJSON = "geojson"
    KML = "kml"
    SHAPEFILE = "shapefile"
    GPKG = "gpkg"
    WKT = "wkt"
    WKB = "wkb"


@dataclass
class SpatialQuery:
    """Spatial query configuration"""
    geometry_column: str
    spatial_operation: str  # intersects, contains, within, dwithin, etc.
    query_geometry: Union[str, Dict]  # WKT string or GeoJSON
    buffer_distance: Optional[float] = None
    coordinate_system: int = 4326
    additional_filters: Optional[Dict[str, Any]] = None


@dataclass
class SpatialAnalysisResult:
    """Result of spatial analysis operation"""
    operation_type: str
    result_geometry: Optional[str]  # WKT format
    statistics: Dict[str, float]
    feature_count: int
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class CoordinateTransformation:
    """Coordinate system transformation configuration"""
    source_crs: int
    target_crs: int
    transformation_accuracy: float
    area_of_use: Optional[Dict[str, float]] = None


# Enhanced SQLAlchemy Models with PostGIS
Base = declarative_base()


class VolcanicSystem(Base):
    """Enhanced volcanic system model with comprehensive geospatial data"""
    __tablename__ = 'volcanic_systems'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    gvp_id = Column(String(20), unique=True)  # Global Volcanism Program ID
    name = Column(String(255), nullable=False)
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    elevation = Column(Float)
    volcano_type = Column(String(100))
    tectonic_setting = Column(String(100))
    
    # Spatial extents
    hazard_zone = Column(Geometry('MULTIPOLYGON', srid=4326))
    monitoring_network = Column(Geometry('MULTIPOLYGON', srid=4326))
    
    # Temporal data
    last_eruption = Column(DateTime)
    eruption_history = Column(JSONB)
    
    # Classification and metadata
    hazard_level = Column(Integer)
    monitoring_status = Column(String(50))
    population_exposure = Column(JSONB)
    geological_context = Column(JSONB)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    seismic_events = relationship("SeismicEvent", back_populates="volcanic_system")
    deformation_measurements = relationship("DeformationMeasurement", back_populates="volcanic_system")
    monitoring_stations = relationship("MonitoringStation", back_populates="volcanic_system")
    
    __table_args__ = (
        Index('idx_volcanic_location_gist', 'location', postgresql_using='gist'),
        Index('idx_volcanic_hazard_zone_gist', 'hazard_zone', postgresql_using='gist'),
        Index('idx_volcanic_gvp_id', 'gvp_id'),
        Index('idx_volcanic_tectonic_setting', 'tectonic_setting'),
    )


class SeismicEvent(Base):
    """Enhanced seismic event model with spatial-temporal indexing"""
    __tablename__ = 'seismic_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    volcanic_system_id = Column(UUID(as_uuid=True), ForeignKey('volcanic_systems.id'))
    
    # Temporal data
    origin_time = Column(DateTime, nullable=False)
    
    # Spatial data
    hypocenter = Column(Geometry('POINTZ', srid=4326), nullable=False)  # 3D point with depth
    epicenter = Column(Geometry('POINT', srid=4326), nullable=False)
    
    # Seismic parameters
    magnitude = Column(Float, nullable=False)
    magnitude_type = Column(String(10))
    depth = Column(Float, nullable=False)
    
    # Event classification
    event_type = Column(String(50))
    event_class = Column(String(50))  # VT, LP, Hybrid, etc.
    
    # Quality metrics
    location_uncertainty = Column(Float)
    magnitude_uncertainty = Column(Float)
    azimuthal_gap = Column(Float)
    station_count = Column(Integer)
    
    # Processing metadata
    network_code = Column(String(20))
    processing_agency = Column(String(50))
    waveform_data = Column(JSONB)
    moment_tensor = Column(JSONB)
    focal_mechanism = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    volcanic_system = relationship("VolcanicSystem", back_populates="seismic_events")
    
    __table_args__ = (
        Index('idx_seismic_hypocenter_gist', 'hypocenter', postgresql_using='gist'),
        Index('idx_seismic_epicenter_gist', 'epicenter', postgresql_using='gist'),
        Index('idx_seismic_origin_time_brin', 'origin_time', postgresql_using='brin'),
        Index('idx_seismic_magnitude', 'magnitude'),
        Index('idx_seismic_depth', 'depth'),
        Index('idx_seismic_volcanic_system', 'volcanic_system_id'),
    )


class DeformationMeasurement(Base):
    """Ground deformation measurements with MT-SAR integration"""
    __tablename__ = 'deformation_measurements'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    volcanic_system_id = Column(UUID(as_uuid=True), ForeignKey('volcanic_systems.id'))
    
    # Temporal data
    acquisition_time = Column(DateTime, nullable=False)
    
    # Spatial data
    measurement_point = Column(Geometry('POINT', srid=4326), nullable=False)
    measurement_area = Column(Geometry('POLYGON', srid=4326))
    
    # Deformation vectors
    displacement_east = Column(Float)
    displacement_north = Column(Float)
    displacement_up = Column(Float)
    line_of_sight_displacement = Column(Float)
    
    # Quality metrics
    coherence = Column(Float)
    uncertainty_east = Column(Float)
    uncertainty_north = Column(Float)
    uncertainty_up = Column(Float)
    
    # Measurement metadata
    measurement_type = Column(String(50))  # InSAR, GPS, leveling, etc.
    satellite_platform = Column(String(50))  # Sentinel-1, ALOS-2, etc.
    orbit_direction = Column(String(20))  # ascending/descending
    processing_parameters = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    volcanic_system = relationship("VolcanicSystem", back_populates="deformation_measurements")
    
    __table_args__ = (
        Index('idx_deformation_point_gist', 'measurement_point', postgresql_using='gist'),
        Index('idx_deformation_area_gist', 'measurement_area', postgresql_using='gist'),
        Index('idx_deformation_time_brin', 'acquisition_time', postgresql_using='brin'),
        Index('idx_deformation_type', 'measurement_type'),
        Index('idx_deformation_volcanic_system', 'volcanic_system_id'),
    )


class MonitoringStation(Base):
    """Monitoring station network with comprehensive metadata"""
    __tablename__ = 'monitoring_stations'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    volcanic_system_id = Column(UUID(as_uuid=True), ForeignKey('volcanic_systems.id'))
    
    # Station identification
    station_code = Column(String(20), unique=True, nullable=False)
    network_code = Column(String(20))
    station_name = Column(String(255))
    
    # Spatial data
    location = Column(Geometry('POINTZ', srid=4326), nullable=False)  # 3D with elevation
    
    # Station metadata
    station_type = Column(String(50))  # seismic, GPS, tilt, gas, thermal, etc.
    instrument_type = Column(String(100))
    installation_date = Column(DateTime)
    operational_status = Column(String(20))
    
    # Data characteristics
    sampling_rate = Column(Float)
    data_availability = Column(JSONB)
    quality_metrics = Column(JSONB)
    
    # Network information
    operator_agency = Column(String(100))
    data_access_policy = Column(String(50))
    real_time_availability = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    volcanic_system = relationship("VolcanicSystem", back_populates="monitoring_stations")
    
    __table_args__ = (
        Index('idx_monitoring_location_gist', 'location', postgresql_using='gist'),
        Index('idx_monitoring_station_code', 'station_code'),
        Index('idx_monitoring_network_code', 'network_code'),
        Index('idx_monitoring_type', 'station_type'),
        Index('idx_monitoring_volcanic_system', 'volcanic_system_id'),
    )


class GeologicalUnit(Base):
    """Geological units with comprehensive spatial and temporal data"""
    __tablename__ = 'geological_units'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    
    # Unit identification
    unit_name = Column(String(255), nullable=False)
    formation_name = Column(String(255))
    member_name = Column(String(255))
    
    # Spatial data
    geometry = Column(Geometry('MULTIPOLYGON', srid=4326), nullable=False)
    representative_point = Column(Geometry('POINT', srid=4326))
    
    # Temporal data
    age_min = Column(Float)  # Ma (million years)
    age_max = Column(Float)  # Ma
    geological_period = Column(String(50))
    geological_era = Column(String(50))
    
    # Lithological data
    primary_lithology = Column(String(100))
    secondary_lithology = Column(String(100))
    rock_type = Column(String(50))
    
    # Geochemical data
    geochemical_composition = Column(JSONB)
    mineral_assemblage = Column(ARRAY(String))
    
    # Structural data
    structural_features = Column(JSONB)
    metamorphic_grade = Column(String(50))
    
    # Metadata
    data_source = Column(String(100))
    mapping_scale = Column(String(20))
    confidence_level = Column(String(20))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_geological_geometry_gist', 'geometry', postgresql_using='gist'),
        Index('idx_geological_point_gist', 'representative_point', postgresql_using='gist'),
        Index('idx_geological_age', 'age_min', 'age_max'),
        Index('idx_geological_lithology', 'primary_lithology'),
        Index('idx_geological_period', 'geological_period'),
    )


class TectonicFeature(Base):
    """Tectonic features with kinematic and geometric data"""
    __tablename__ = 'tectonic_features'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    
    # Feature identification
    feature_name = Column(String(255))
    feature_type = Column(String(100))  # fault, fold, shear_zone, etc.
    
    # Spatial data
    geometry = Column(Geometry('MULTILINESTRING', srid=4326))  # Linear features
    zone_geometry = Column(Geometry('MULTIPOLYGON', srid=4326))  # Areal extent
    
    # Kinematic data
    movement_sense = Column(String(50))  # normal, reverse, strike-slip, etc.
    dip_direction = Column(Float)
    dip_angle = Column(Float)
    strike = Column(Float)
    
    # Activity data
    activity_status = Column(String(50))  # active, inactive, uncertain
    last_activity = Column(Float)  # Ma
    slip_rate = Column(Float)  # mm/year
    
    # Seismic data
    maximum_magnitude = Column(Float)
    recurrence_interval = Column(Float)
    
    # Metadata
    data_source = Column(String(100))
    confidence_level = Column(String(20))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_tectonic_geometry_gist', 'geometry', postgresql_using='gist'),
        Index('idx_tectonic_zone_gist', 'zone_geometry', postgresql_using='gist'),
        Index('idx_tectonic_type', 'feature_type'),
        Index('idx_tectonic_activity', 'activity_status'),
    )


class GISPostgreSQLOperations:
    """
    Comprehensive GIS-enabled PostgreSQL operations for geological data
    Advanced spatial indexing, queries, transformations, and analysis
    """
    
    def __init__(self, database_integrator: EnhancedDatabaseIntegrator):
        self.db = database_integrator
        self.logger = logging.getLogger(__name__ + ".GISPostgreSQLOperations")
        
        # Coordinate system transformations
        self.transformers = {}
        self.coordinate_systems = {}
        
        # Spatial analysis cache
        self.analysis_cache = {}
        
        # Performance metrics
        self.spatial_metrics = {
            'queries_executed': 0,
            'transformations_performed': 0,
            'indexes_created': 0,
            'exports_completed': 0
        }
        
        self.logger.info("GIS PostgreSQL Operations initialized")
    
    async def initialize_spatial_extensions(self) -> bool:
        """Initialize PostGIS and related spatial extensions"""
        try:
            async with self.db.get_postgresql_session() as session:
                # Enable PostGIS extensions
                await session.execute("CREATE EXTENSION IF NOT EXISTS postgis")
                await session.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology")
                await session.execute("CREATE EXTENSION IF NOT EXISTS postgis_raster")
                await session.execute("CREATE EXTENSION IF NOT EXISTS postgis_sfcgal")
                await session.execute("CREATE EXTENSION IF NOT EXISTS address_standardizer")
                await session.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch")
                
                # Enable TimescaleDB for time-series spatial data
                await session.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
                
                # Create custom spatial functions
                await self._create_custom_spatial_functions(session)
                
                # Create spatial reference systems
                await self._initialize_coordinate_systems(session)
                
                await session.commit()
                
            self.logger.info("Spatial extensions initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize spatial extensions: {e}")
            return False
    
    async def _create_custom_spatial_functions(self, session: AsyncSession):
        """Create custom spatial functions for geological analysis"""
        
        # Function to calculate volcanic hazard zones
        hazard_zone_function = """
        CREATE OR REPLACE FUNCTION calculate_volcanic_hazard_zone(
            volcano_location geometry,
            vei integer DEFAULT 3,
            wind_direction float DEFAULT 0
        ) RETURNS geometry AS $$
        DECLARE
            base_radius float;
            hazard_zone geometry;
        BEGIN
            -- Calculate base radius based on VEI
            base_radius := CASE 
                WHEN vei <= 2 THEN 5000
                WHEN vei = 3 THEN 10000
                WHEN vei = 4 THEN 25000
                WHEN vei = 5 THEN 50000
                WHEN vei >= 6 THEN 100000
                ELSE 10000
            END;
            
            -- Create elliptical hazard zone considering wind direction
            hazard_zone := ST_Buffer(
                ST_Transform(volcano_location, 3857),
                base_radius,
                'quad_segs=16'
            );
            
            RETURN ST_Transform(hazard_zone, 4326);
        END;
        $$ LANGUAGE plpgsql;
        """
        
        # Function to calculate seismic network coverage
        network_coverage_function = """
        CREATE OR REPLACE FUNCTION calculate_seismic_network_coverage(
            station_locations geometry[],
            detection_threshold float DEFAULT 2.0
        ) RETURNS geometry AS $$
        DECLARE
            coverage_area geometry;
            station geometry;
            station_buffer geometry;
        BEGIN
            coverage_area := NULL;
            
            FOREACH station IN ARRAY station_locations LOOP
                -- Calculate detection radius based on threshold
                station_buffer := ST_Buffer(
                    ST_Transform(station, 3857),
                    (detection_threshold * 50000),  -- 50km per magnitude unit
                    'quad_segs=16'
                );
                
                IF coverage_area IS NULL THEN
                    coverage_area := station_buffer;
                ELSE
                    coverage_area := ST_Union(coverage_area, station_buffer);
                END IF;
            END LOOP;
            
            RETURN ST_Transform(coverage_area, 4326);
        END;
        $$ LANGUAGE plpgsql;
        """
        
        # Function to calculate tectonic stress orientation
        stress_orientation_function = """
        CREATE OR REPLACE FUNCTION calculate_stress_orientation(
            fault_geometries geometry[],
            analysis_point geometry
        ) RETURNS float AS $$
        DECLARE
            weighted_orientation float := 0;
            total_weight float := 0;
            fault_geom geometry;
            distance float;
            weight float;
            orientation float;
        BEGIN
            FOREACH fault_geom IN ARRAY fault_geometries LOOP
                distance := ST_Distance(
                    ST_Transform(analysis_point, 3857),
                    ST_Transform(fault_geom, 3857)
                );
                
                -- Weight by inverse distance
                weight := 1.0 / (distance + 1000);  -- Add 1km to avoid division by zero
                
                -- Calculate fault orientation
                orientation := degrees(ST_Azimuth(
                    ST_StartPoint(fault_geom),
                    ST_EndPoint(fault_geom)
                ));
                
                weighted_orientation := weighted_orientation + (orientation * weight);
                total_weight := total_weight + weight;
            END LOOP;
            
            IF total_weight > 0 THEN
                RETURN weighted_orientation / total_weight;
            ELSE
                RETURN NULL;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        await session.execute(hazard_zone_function)
        await session.execute(network_coverage_function)
        await session.execute(stress_orientation_function)
    
    async def _initialize_coordinate_systems(self, session: AsyncSession):
        """Initialize commonly used coordinate systems"""
        
        # Store coordinate system information
        self.coordinate_systems = {
            CoordinateSystem.WGS84.value: {
                'name': 'WGS 84',
                'proj4': '+proj=longlat +datum=WGS84 +no_defs',
                'units': 'degrees',
                'area_of_use': 'World'
            },
            CoordinateSystem.WEB_MERCATOR.value: {
                'name': 'WGS 84 / Pseudo-Mercator',
                'proj4': '+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs',
                'units': 'meters',
                'area_of_use': 'World - between 85.06°S and 85.06°N'
            }
        }
        
        # Verify spatial reference systems exist
        for srid, info in self.coordinate_systems.items():
            result = await session.execute(
                "SELECT COUNT(*) FROM spatial_ref_sys WHERE srid = :srid",
                {'srid': srid}
            )
            count = result.scalar()
            
            if count == 0:
                self.logger.warning(f"SRID {srid} not found in spatial_ref_sys")
    
    # Spatial Indexing Operations
    async def create_spatial_indexes(self, table_name: str, geometry_columns: List[str], 
                                   index_type: SpatialIndexType = SpatialIndexType.GIST) -> bool:
        """Create optimized spatial indexes for large-scale geological datasets"""
        try:
            async with self.db.get_postgresql_session() as session:
                for column in geometry_columns:
                    index_name = f"idx_{table_name}_{column}_{index_type.value.lower()}"
                    
                    # Drop existing index if it exists
                    await session.execute(f"DROP INDEX IF EXISTS {index_name}")
                    
                    # Create spatial index
                    if index_type == SpatialIndexType.GIST:
                        index_sql = f"""
                        CREATE INDEX {index_name} ON {table_name} 
                        USING GIST ({column})
                        WITH (fillfactor = 90)
                        """
                    elif index_type == SpatialIndexType.SPGIST:
                        index_sql = f"""
                        CREATE INDEX {index_name} ON {table_name} 
                        USING SPGIST ({column})
                        """
                    elif index_type == SpatialIndexType.BRIN:
                        index_sql = f"""
                        CREATE INDEX {index_name} ON {table_name} 
                        USING BRIN ({column})
                        WITH (pages_per_range = 128)
                        """
                    else:
                        continue
                    
                    await session.execute(index_sql)
                    
                    # Analyze table for query planner
                    await session.execute(f"ANALYZE {table_name}")
                    
                    self.spatial_metrics['indexes_created'] += 1
                
                await session.commit()
                
            self.logger.info(f"Created spatial indexes for {table_name}: {geometry_columns}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create spatial indexes: {e}")
            return False
    
    async def optimize_spatial_queries(self, table_name: str) -> Dict[str, Any]:
        """Optimize spatial queries for large-scale geological datasets"""
        try:
            async with self.db.get_postgresql_session() as session:
                optimization_results = {}
                
                # Update table statistics
                await session.execute(f"ANALYZE {table_name}")
                
                # Vacuum and reindex
                await session.execute(f"VACUUM ANALYZE {table_name}")
                
                # Check index usage statistics
                stats_query = """
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes 
                WHERE tablename = :table_name
                """
                
                result = await session.execute(stats_query, {'table_name': table_name})
                index_stats = result.fetchall()
                
                optimization_results['index_statistics'] = [
                    {
                        'index_name': row.indexname,
                        'scans': row.idx_scan,
                        'tuples_read': row.idx_tup_read,
                        'tuples_fetched': row.idx_tup_fetch
                    }
                    for row in index_stats
                ]
                
                # Get table size information
                size_query = """
                SELECT 
                    pg_size_pretty(pg_total_relation_size(:table_name)) as total_size,
                    pg_size_pretty(pg_relation_size(:table_name)) as table_size,
                    pg_size_pretty(pg_indexes_size(:table_name)) as indexes_size
                """
                
                result = await session.execute(size_query, {'table_name': table_name})
                size_info = result.fetchone()
                
                optimization_results['storage_info'] = {
                    'total_size': size_info.total_size,
                    'table_size': size_info.table_size,
                    'indexes_size': size_info.indexes_size
                }
                
                return optimization_results
                
        except Exception as e:
            self.logger.error(f"Failed to optimize spatial queries: {e}")
            return {}
    
    # Advanced Spatial Queries
    async def query_volcanic_systems_by_region(self, 
                                             region_geometry: Union[str, Dict],
                                             buffer_distance: Optional[float] = None,
                                             tectonic_settings: Optional[List[str]] = None,
                                             hazard_levels: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Query volcanic systems within a specified region with advanced filtering"""
        try:
            async with self.db.get_postgresql_session() as session:
                # Convert geometry input
                if isinstance(region_geometry, dict):
                    # GeoJSON input
                    geom_wkt = f"ST_GeomFromGeoJSON('{json.dumps(region_geometry)}')"
                else:
                    # WKT input
                    geom_wkt = f"ST_GeomFromText('{region_geometry}', 4326)"
                
                # Build query
                query = select(VolcanicSystem).where(
                    ST_Intersects(
                        VolcanicSystem.location,
                        func.ST_Transform(func.text(geom_wkt), 4326)
                    )
                )
                
                # Add buffer if specified
                if buffer_distance:
                    query = query.where(
                        ST_DWithin(
                            ST_Transform(VolcanicSystem.location, 3857),
                            ST_Transform(func.text(geom_wkt), 3857),
                            buffer_distance
                        )
                    )
                
                # Add tectonic setting filter
                if tectonic_settings:
                    query = query.where(VolcanicSystem.tectonic_setting.in_(tectonic_settings))
                
                # Add hazard level filter
                if hazard_levels:
                    query = query.where(VolcanicSystem.hazard_level.in_(hazard_levels))
                
                result = await session.execute(query)
                volcanic_systems = result.scalars().all()
                
                # Convert to dictionary format
                systems_data = []
                for system in volcanic_systems:
                    system_dict = {
                        'id': str(system.id),
                        'gvp_id': system.gvp_id,
                        'name': system.name,
                        'location': {
                            'type': 'Point',
                            'coordinates': [
                                session.scalar(func.ST_X(system.location)),
                                session.scalar(func.ST_Y(system.location))
                            ]
                        },
                        'elevation': system.elevation,
                        'volcano_type': system.volcano_type,
                        'tectonic_setting': system.tectonic_setting,
                        'hazard_level': system.hazard_level,
                        'monitoring_status': system.monitoring_status,
                        'last_eruption': system.last_eruption.isoformat() if system.last_eruption else None
                    }
                    systems_data.append(system_dict)
                
                self.spatial_metrics['queries_executed'] += 1
                return systems_data
                
        except Exception as e:
            self.logger.error(f"Failed to query volcanic systems by region: {e}")
            return []
    
    async def query_seismic_events_spatial_temporal(self,
                                                  spatial_bounds: Dict[str, float],
                                                  time_range: Tuple[datetime, datetime],
                                                  magnitude_range: Optional[Tuple[float, float]] = None,
                                                  depth_range: Optional[Tuple[float, float]] = None,
                                                  event_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Advanced spatial-temporal query for seismic events"""
        try:
            async with self.db.get_postgresql_session() as session:
                # Create bounding box geometry
                bbox_wkt = f"""POLYGON(({spatial_bounds['min_lon']} {spatial_bounds['min_lat']},
                                      {spatial_bounds['max_lon']} {spatial_bounds['min_lat']},
                                      {spatial_bounds['max_lon']} {spatial_bounds['max_lat']},
                                      {spatial_bounds['min_lon']} {spatial_bounds['max_lat']},
                                      {spatial_bounds['min_lon']} {spatial_bounds['min_lat']}))"""
                
                # Build query
                query = select(SeismicEvent).where(
                    and_(
                        ST_Intersects(
                            SeismicEvent.epicenter,
                            func.ST_GeomFromText(bbox_wkt, 4326)
                        ),
                        SeismicEvent.origin_time >= time_range[0],
                        SeismicEvent.origin_time <= time_range[1]
                    )
                )
                
                # Add magnitude filter
                if magnitude_range:
                    query = query.where(
                        and_(
                            SeismicEvent.magnitude >= magnitude_range[0],
                            SeismicEvent.magnitude <= magnitude_range[1]
                        )
                    )
                
                # Add depth filter
                if depth_range:
                    query = query.where(
                        and_(
                            SeismicEvent.depth >= depth_range[0],
                            SeismicEvent.depth <= depth_range[1]
                        )
                    )
                
                # Add event type filter
                if event_types:
                    query = query.where(SeismicEvent.event_type.in_(event_types))
                
                # Order by time
                query = query.order_by(SeismicEvent.origin_time.desc())
                
                result = await session.execute(query)
                events = result.scalars().all()
                
                # Convert to dictionary format
                events_data = []
                for event in events:
                    event_dict = {
                        'id': str(event.id),
                        'origin_time': event.origin_time.isoformat(),
                        'epicenter': {
                            'type': 'Point',
                            'coordinates': [
                                session.scalar(func.ST_X(event.epicenter)),
                                session.scalar(func.ST_Y(event.epicenter))
                            ]
                        },
                        'hypocenter': {
                            'type': 'Point',
                            'coordinates': [
                                session.scalar(func.ST_X(event.hypocenter)),
                                session.scalar(func.ST_Y(event.hypocenter)),
                                session.scalar(func.ST_Z(event.hypocenter))
                            ]
                        },
                        'magnitude': event.magnitude,
                        'magnitude_type': event.magnitude_type,
                        'depth': event.depth,
                        'event_type': event.event_type,
                        'event_class': event.event_class,
                        'location_uncertainty': event.location_uncertainty,
                        'station_count': event.station_count
                    }
                    events_data.append(event_dict)
                
                self.spatial_metrics['queries_executed'] += 1
                return events_data
                
        except Exception as e:
            self.logger.error(f"Failed to query seismic events: {e}")
            return []
            
    async def query_monitoring_network_coverage(self,
                                              volcanic_system_id: str,
                                              station_types: Optional[List[str]] = None,
                                              operational_only: bool = True) -> Dict[str, Any]:
        """Query monitoring network coverage for a volcanic system"""
        try:
            async with self.db.get_postgresql_session() as session:
                # Get volcanic system
                volcano_query = select(VolcanicSystem).where(VolcanicSystem.id == volcanic_system_id)
                volcano_result = await session.execute(volcano_query)
                volcano = volcano_result.scalar_one_or_none()
                
                if not volcano:
                    return {'error': 'Volcanic system not found'}
                
                # Build monitoring stations query
                stations_query = select(MonitoringStation).where(
                    MonitoringStation.volcanic_system_id == volcanic_system_id
                )
                
                if station_types:
                    stations_query = stations_query.where(MonitoringStation.station_type.in_(station_types))
                
                if operational_only:
                    stations_query = stations_query.where(MonitoringStation.operational_status == 'operational')
                
                stations_result = await session.execute(stations_query)
                stations = stations_result.scalars().all()
                
                # Calculate network coverage using custom function
                if stations:
                    station_locations = [
                        session.scalar(func.ST_AsText(station.location))
                        for station in stations
                    ]
                    
                    # Use custom function to calculate coverage
                    coverage_query = """
                    SELECT ST_AsGeoJSON(calculate_seismic_network_coverage(
                        ARRAY[%s]::geometry[],
                        2.0
                    )) as coverage_geojson
                    """ % ','.join([f"ST_GeomFromText('{loc}', 4326)" for loc in station_locations])
                    
                    coverage_result = await session.execute(coverage_query)
                    coverage_geojson = coverage_result.scalar()
                    
                    coverage_data = {
                        'volcanic_system_id': volcanic_system_id,
                        'volcano_name': volcano.name,
                        'station_count': len(stations),
                        'station_types': list(set(station.station_type for station in stations)),
                        'coverage_area': json.loads(coverage_geojson) if coverage_geojson else None,
                        'stations': [
                            {
                                'id': str(station.id),
                                'code': station.station_code,
                                'name': station.station_name,
                                'type': station.station_type,
                                'location': {
                                    'type': 'Point',
                                    'coordinates': [
                                        session.scalar(func.ST_X(station.location)),
                                        session.scalar(func.ST_Y(station.location)),
                                        session.scalar(func.ST_Z(station.location))
                                    ]
                                },
                                'operational_status': station.operational_status,
                                'real_time_availability': station.real_time_availability
                            }
                            for station in stations
                        ]
                    }
                else:
                    coverage_data = {
                        'volcanic_system_id': volcanic_system_id,
                        'volcano_name': volcano.name,
                        'station_count': 0,
                        'station_types': [],
                        'coverage_area': None,
                        'stations': []
                    }
                
                self.spatial_metrics['queries_executed'] += 1
                return coverage_data
                
        except Exception as e:
            self.logger.error(f"Failed to query monitoring network coverage: {e}")
            return {'error': str(e)}
    
    # Coordinate System Transformations
    async def transform_coordinates(self, 
                                  geometries: List[str],
                                  source_crs: int,
                                  target_crs: int) -> List[str]:
        """Transform coordinates between different coordinate reference systems"""
        try:
            async with self.db.get_postgresql_session() as session:
                transformed_geometries = []
                
                for geom_wkt in geometries:
                    # Transform using PostGIS
                    transform_query = """
                    SELECT ST_AsText(ST_Transform(ST_GeomFromText(:geom_wkt, :source_crs), :target_crs))
                    """
                    
                    result = await session.execute(
                        transform_query,
                        {
                            'geom_wkt': geom_wkt,
                            'source_crs': source_crs,
                            'target_crs': target_crs
                        }
                    )
                    
                    transformed_wkt = result.scalar()
                    transformed_geometries.append(transformed_wkt)
                
                self.spatial_metrics['transformations_performed'] += len(geometries)
                return transformed_geometries
                
        except Exception as e:
            self.logger.error(f"Failed to transform coordinates: {e}")
            return []
    
    async def get_optimal_utm_zone(self, longitude: float, latitude: float) -> int:
        """Get optimal UTM zone for a given location"""
        try:
            # Calculate UTM zone
            utm_zone = int((longitude + 180) / 6) + 1
            
            # Determine hemisphere
            if latitude >= 0:
                # Northern hemisphere
                epsg_code = 32600 + utm_zone
            else:
                # Southern hemisphere
                epsg_code = 32700 + utm_zone
            
            return epsg_code
            
        except Exception as e:
            self.logger.error(f"Failed to determine UTM zone: {e}")
            return CoordinateSystem.WGS84.value
    
    async def validate_coordinate_system(self, srid: int) -> Dict[str, Any]:
        """Validate and get information about a coordinate reference system"""
        try:
            async with self.db.get_postgresql_session() as session:
                # Query spatial reference system information
                srs_query = """
                SELECT 
                    srid,
                    auth_name,
                    auth_srid,
                    srtext,
                    proj4text
                FROM spatial_ref_sys 
                WHERE srid = :srid
                """
                
                result = await session.execute(srs_query, {'srid': srid})
                srs_info = result.fetchone()
                
                if srs_info:
                    return {
                        'valid': True,
                        'srid': srs_info.srid,
                        'authority': srs_info.auth_name,
                        'authority_code': srs_info.auth_srid,
                        'wkt': srs_info.srtext,
                        'proj4': srs_info.proj4text
                    }
                else:
                    return {
                        'valid': False,
                        'error': f'SRID {srid} not found in spatial_ref_sys'
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to validate coordinate system: {e}")
            return {'valid': False, 'error': str(e)}
    
    # Geospatial Data Validation and Quality Control
    async def validate_geometry_quality(self, 
                                      table_name: str,
                                      geometry_column: str = 'geometry') -> Dict[str, Any]:
        """Comprehensive geometry validation and quality control"""
        try:
            async with self.db.get_postgresql_session() as session:
                validation_results = {}
                
                # Check for invalid geometries
                invalid_query = f"""
                SELECT COUNT(*) as invalid_count
                FROM {table_name}
                WHERE NOT ST_IsValid({geometry_column})
                """
                
                result = await session.execute(invalid_query)
                invalid_count = result.scalar()
                validation_results['invalid_geometries'] = invalid_count
                
                # Check for empty geometries
                empty_query = f"""
                SELECT COUNT(*) as empty_count
                FROM {table_name}
                WHERE ST_IsEmpty({geometry_column})
                """
                
                result = await session.execute(empty_query)
                empty_count = result.scalar()
                validation_results['empty_geometries'] = empty_count
                
                # Check for null geometries
                null_query = f"""
                SELECT COUNT(*) as null_count
                FROM {table_name}
                WHERE {geometry_column} IS NULL
                """
                
                result = await session.execute(null_query)
                null_count = result.scalar()
                validation_results['null_geometries'] = null_count
                
                # Check coordinate system consistency
                srid_query = f"""
                SELECT ST_SRID({geometry_column}) as srid, COUNT(*) as count
                FROM {table_name}
                WHERE {geometry_column} IS NOT NULL
                GROUP BY ST_SRID({geometry_column})
                ORDER BY count DESC
                """
                
                result = await session.execute(srid_query)
                srid_distribution = result.fetchall()
                validation_results['srid_distribution'] = [
                    {'srid': row.srid, 'count': row.count}
                    for row in srid_distribution
                ]
                
                # Check geometry types
                type_query = f"""
                SELECT ST_GeometryType({geometry_column}) as geom_type, COUNT(*) as count
                FROM {table_name}
                WHERE {geometry_column} IS NOT NULL
                GROUP BY ST_GeometryType({geometry_column})
                ORDER BY count DESC
                """
                
                result = await session.execute(type_query)
                type_distribution = result.fetchall()
                validation_results['geometry_types'] = [
                    {'type': row.geom_type, 'count': row.count}
                    for row in type_distribution
                ]
                
                # Calculate bounding box
                bbox_query = f"""
                SELECT 
                    ST_XMin(ST_Extent({geometry_column})) as min_x,
                    ST_YMin(ST_Extent({geometry_column})) as min_y,
                    ST_XMax(ST_Extent({geometry_column})) as max_x,
                    ST_YMax(ST_Extent({geometry_column})) as max_y
                FROM {table_name}
                WHERE {geometry_column} IS NOT NULL
                """
                
                result = await session.execute(bbox_query)
                bbox = result.fetchone()
                if bbox:
                    validation_results['bounding_box'] = {
                        'min_x': bbox.min_x,
                        'min_y': bbox.min_y,
                        'max_x': bbox.max_x,
                        'max_y': bbox.max_y
                    }
                
                # Get total record count
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                result = await session.execute(count_query)
                total_count = result.scalar()
                validation_results['total_records'] = total_count
                
                # Calculate quality score
                if total_count > 0:
                    valid_count = total_count - invalid_count - empty_count - null_count
                    quality_score = (valid_count / total_count) * 100
                    validation_results['quality_score'] = round(quality_score, 2)
                else:
                    validation_results['quality_score'] = 0
                
                return validation_results
                
        except Exception as e:
            self.logger.error(f"Failed to validate geometry quality: {e}")
            return {'error': str(e)}
    
    async def repair_invalid_geometries(self, 
                                      table_name: str,
                                      geometry_column: str = 'geometry',
                                      id_column: str = 'id') -> Dict[str, Any]:
        """Repair invalid geometries using PostGIS functions"""
        try:
            async with self.db.get_postgresql_session() as session:
                repair_results = {}
                
                # Find invalid geometries
                invalid_query = f"""
                SELECT {id_column}
                FROM {table_name}
                WHERE NOT ST_IsValid({geometry_column})
                """
                
                result = await session.execute(invalid_query)
                invalid_ids = [row[0] for row in result.fetchall()]
                repair_results['invalid_count'] = len(invalid_ids)
                
                if invalid_ids:
                    # Attempt to repair using ST_MakeValid
                    repair_query = f"""
                    UPDATE {table_name}
                    SET {geometry_column} = ST_MakeValid({geometry_column})
                    WHERE {id_column} = ANY(:invalid_ids)
                    AND NOT ST_IsValid({geometry_column})
                    """
                    
                    await session.execute(repair_query, {'invalid_ids': invalid_ids})
                    
                    # Check how many were successfully repaired
                    still_invalid_query = f"""
                    SELECT COUNT(*)
                    FROM {table_name}
                    WHERE {id_column} = ANY(:invalid_ids)
                    AND NOT ST_IsValid({geometry_column})
                    """
                    
                    result = await session.execute(still_invalid_query, {'invalid_ids': invalid_ids})
                    still_invalid_count = result.scalar()
                    
                    repaired_count = len(invalid_ids) - still_invalid_count
                    repair_results['repaired_count'] = repaired_count
                    repair_results['still_invalid_count'] = still_invalid_count
                    
                    await session.commit()
                else:
                    repair_results['repaired_count'] = 0
                    repair_results['still_invalid_count'] = 0
                
                return repair_results
                
        except Exception as e:
            self.logger.error(f"Failed to repair invalid geometries: {e}")
            await session.rollback()
            return {'error': str(e)}
    
    # Spatial Clustering and Analysis
    async def perform_spatial_clustering(self,
                                       table_name: str,
                                       geometry_column: str,
                                       cluster_distance: float,
                                       min_points: int = 3) -> Dict[str, Any]:
        """Perform spatial clustering analysis on geological features"""
        try:
            async with self.db.get_postgresql_session() as session:
                # Create clustering using ST_ClusterDBSCAN
                clustering_query = f"""
                WITH clustered_data AS (
                    SELECT 
                        *,
                        ST_ClusterDBSCAN({geometry_column}, eps := :cluster_distance, minpoints := :min_points) 
                        OVER () AS cluster_id
                    FROM {table_name}
                    WHERE {geometry_column} IS NOT NULL
                )
                SELECT 
                    cluster_id,
                    COUNT(*) as point_count,
                    ST_AsText(ST_Centroid(ST_Union({geometry_column}))) as cluster_center,
                    ST_AsText(ST_ConvexHull(ST_Union({geometry_column}))) as cluster_hull,
                    ST_Area(ST_ConvexHull(ST_Union({geometry_column}))) as cluster_area
                FROM clustered_data
                WHERE cluster_id IS NOT NULL
                GROUP BY cluster_id
                ORDER BY point_count DESC
                """
                
                result = await session.execute(
                    clustering_query,
                    {
                        'cluster_distance': cluster_distance,
                        'min_points': min_points
                    }
                )
                
                clusters = result.fetchall()
                
                clustering_results = {
                    'cluster_count': len(clusters),
                    'clustering_parameters': {
                        'distance_threshold': cluster_distance,
                        'minimum_points': min_points
                    },
                    'clusters': [
                        {
                            'cluster_id': cluster.cluster_id,
                            'point_count': cluster.point_count,
                            'center_wkt': cluster.cluster_center,
                            'convex_hull_wkt': cluster.cluster_hull,
                            'area': cluster.cluster_area
                        }
                        for cluster in clusters
                    ]
                }
                
                # Calculate clustering statistics
                if clusters:
                    point_counts = [cluster.point_count for cluster in clusters]
                    clustering_results['statistics'] = {
                        'mean_cluster_size': np.mean(point_counts),
                        'median_cluster_size': np.median(point_counts),
                        'max_cluster_size': max(point_counts),
                        'min_cluster_size': min(point_counts),
                        'total_clustered_points': sum(point_counts)
                    }
                
                return clustering_results
                
        except Exception as e:
            self.logger.error(f"Failed to perform spatial clustering: {e}")
            return {'error': str(e)}
    
    async def calculate_spatial_statistics(self,
                                         table_name: str,
                                         geometry_column: str,
                                         analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """Calculate comprehensive spatial statistics for geological datasets"""
        try:
            async with self.db.get_postgresql_session() as session:
                statistics = {}
                
                # Basic geometric statistics
                basic_stats_query = f"""
                SELECT 
                    COUNT(*) as feature_count,
                    ST_AsText(ST_Centroid(ST_Union({geometry_column}))) as geometric_center,
                    ST_Area(ST_Union({geometry_column})) as total_area,
                    ST_Perimeter(ST_Union({geometry_column})) as total_perimeter,
                    ST_XMin(ST_Extent({geometry_column})) as min_x,
                    ST_YMin(ST_Extent({geometry_column})) as min_y,
                    ST_XMax(ST_Extent({geometry_column})) as max_x,
                    ST_YMax(ST_Extent({geometry_column})) as max_y
                FROM {table_name}
                WHERE {geometry_column} IS NOT NULL
                """
                
                result = await session.execute(basic_stats_query)
                basic_stats = result.fetchone()
                
                if basic_stats:
                    statistics['basic_statistics'] = {
                        'feature_count': basic_stats.feature_count,
                        'geometric_center_wkt': basic_stats.geometric_center,
                        'total_area': basic_stats.total_area,
                        'total_perimeter': basic_stats.total_perimeter,
                        'bounding_box': {
                            'min_x': basic_stats.min_x,
                            'min_y': basic_stats.min_y,
                            'max_x': basic_stats.max_x,
                            'max_y': basic_stats.max_y
                        }
                    }
                
                if analysis_type == 'comprehensive':
                    # Individual feature statistics
                    feature_stats_query = f"""
                    SELECT 
                        AVG(ST_Area({geometry_column})) as mean_area,
                        STDDEV(ST_Area({geometry_column})) as stddev_area,
                        MIN(ST_Area({geometry_column})) as min_area,
                        MAX(ST_Area({geometry_column})) as max_area,
                        AVG(ST_Perimeter({geometry_column})) as mean_perimeter,
                        STDDEV(ST_Perimeter({geometry_column})) as stddev_perimeter
                    FROM {table_name}
                    WHERE {geometry_column} IS NOT NULL
                    AND ST_GeometryType({geometry_column}) IN ('ST_Polygon', 'ST_MultiPolygon')
                    """
                    
                    result = await session.execute(feature_stats_query)
                    feature_stats = result.fetchone()
                    
                    if feature_stats:
                        statistics['feature_statistics'] = {
                            'area': {
                                'mean': feature_stats.mean_area,
                                'standard_deviation': feature_stats.stddev_area,
                                'minimum': feature_stats.min_area,
                                'maximum': feature_stats.max_area
                            },
                            'perimeter': {
                                'mean': feature_stats.mean_perimeter,
                                'standard_deviation': feature_stats.stddev_perimeter
                            }
                        }
                    
                    # Spatial distribution analysis
                    distribution_query = f"""
                    WITH grid AS (
                        SELECT 
                            i, j,
                            ST_MakeEnvelope(
                                ST_XMin(extent) + (ST_XMax(extent) - ST_XMin(extent)) * i / 10.0,
                                ST_YMin(extent) + (ST_YMax(extent) - ST_YMin(extent)) * j / 10.0,
                                ST_XMin(extent) + (ST_XMax(extent) - ST_XMin(extent)) * (i + 1) / 10.0,
                                ST_YMin(extent) + (ST_YMax(extent) - ST_YMin(extent)) * (j + 1) / 10.0,
                                4326
                            ) as cell
                        FROM 
                            (SELECT ST_Extent({geometry_column}) as extent FROM {table_name}) as bounds,
                            generate_series(0, 9) as i,
                            generate_series(0, 9) as j
                    ),
                    grid_counts AS (
                        SELECT 
                            g.i, g.j,
                            COUNT(t.*) as feature_count
                        FROM grid g
                        LEFT JOIN {table_name} t ON ST_Intersects(t.{geometry_column}, g.cell)
                        GROUP BY g.i, g.j
                    )
                    SELECT 
                        AVG(feature_count) as mean_density,
                        STDDEV(feature_count) as stddev_density,
                        MIN(feature_count) as min_density,
                        MAX(feature_count) as max_density
                    FROM grid_counts
                    """
                    
                    result = await session.execute(distribution_query)
                    distribution_stats = result.fetchone()
                    
                    if distribution_stats:
                        statistics['spatial_distribution'] = {
                            'mean_density': distribution_stats.mean_density,
                            'density_standard_deviation': distribution_stats.stddev_density,
                            'minimum_density': distribution_stats.min_density,
                            'maximum_density': distribution_stats.max_density
                        }
                
                return statistics
                
        except Exception as e:
            self.logger.error(f"Failed to calculate spatial statistics: {e}")
            return {'error': str(e)}
    
    # Geospatial Data Export
    async def export_geospatial_data(self,
                                   table_name: str,
                                   export_format: ExportFormat,
                                   output_path: str,
                                   geometry_column: str = 'geometry',
                                   where_clause: Optional[str] = None,
                                   coordinate_system: Optional[int] = None) -> Dict[str, Any]:
        """Export geospatial data in various formats"""
        try:
            async with self.db.get_postgresql_session() as session:
                # Build query
                select_columns = f"*, ST_AsText({geometry_column}) as geometry_wkt"
                
                if export_format == ExportFormat.GEOJSON:
                    select_columns = f"*, ST_AsGeoJSON({geometry_column}) as geometry_geojson"
                elif coordinate_system and coordinate_system != 4326:
                    select_columns = f"*, ST_AsText(ST_Transform({geometry_column}, {coordinate_system})) as geometry_wkt"
                
                query = f"SELECT {select_columns} FROM {table_name}"
                
                if where_clause:
                    query += f" WHERE {where_clause}"
                
                result = await session.execute(query)
                rows = result.fetchall()
                
                export_results = {
                    'format': export_format.value,
                    'output_path': output_path,
                    'record_count': len(rows),
                    'coordinate_system': coordinate_system or 4326
                }
                
                if export_format == ExportFormat.GEOJSON:
                    # Create GeoJSON
                    features = []
                    for row in rows:
                        row_dict = dict(row._mapping)
                        geometry_json = json.loads(row_dict.pop('geometry_geojson', '{}'))
                        
                        feature = {
                            'type': 'Feature',
                            'geometry': geometry_json,
                            'properties': {k: v for k, v in row_dict.items() 
                                         if k != geometry_column and v is not None}
                        }
                        features.append(feature)
                    
                    geojson_data = {
                        'type': 'FeatureCollection',
                        'features': features,
                        'crs': {
                            'type': 'name',
                            'properties': {
                                'name': f'EPSG:{coordinate_system or 4326}'
                            }
                        }
                    }
                    
                    # Write to file
                    with open(output_path, 'w') as f:
                        json.dump(geojson_data, f, indent=2, default=str)
                
                elif export_format == ExportFormat.KML:
                    # Create KML
                    kml_content = ['<?xml version="1.0" encoding="UTF-8"?>']
                    kml_content.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
                    kml_content.append('<Document>')
                    kml_content.append(f'<name>{table_name}</name>')
                    
                    for row in rows:
                        row_dict = dict(row._mapping)
                        geometry_wkt = row_dict.get('geometry_wkt', '')
                        
                        # Convert WKT to KML coordinates (simplified)
                        if geometry_wkt.startswith('POINT'):
                            coords = geometry_wkt.replace('POINT(', '').replace(')', '')
                            lon, lat = coords.split()
                            
                            kml_content.append('<Placemark>')
                            kml_content.append(f'<name>{row_dict.get("name", "Feature")}</name>')
                            kml_content.append('<Point>')
                            kml_content.append(f'<coordinates>{lon},{lat},0</coordinates>')
                            kml_content.append('</Point>')
                            kml_content.append('</Placemark>')
                    
                    kml_content.append('</Document>')
                    kml_content.append('</kml>')
                    
                    # Write to file
                    with open(output_path, 'w') as f:
                        f.write('\n'.join(kml_content))
                
                elif export_format == ExportFormat.WKT:
                    # Export as WKT
                    wkt_lines = []
                    for row in rows:
                        row_dict = dict(row._mapping)
                        geometry_wkt = row_dict.get('geometry_wkt', '')
                        properties = {k: v for k, v in row_dict.items() 
                                    if k not in [geometry_column, 'geometry_wkt'] and v is not None}
                        
                        wkt_lines.append(f"{geometry_wkt}\t{json.dumps(properties)}")
                    
                    # Write to file
                    with open(output_path, 'w') as f:
                        f.write('\n'.join(wkt_lines))
                
                self.spatial_metrics['exports_completed'] += 1
                return export_results
                
        except Exception as e:
            self.logger.error(f"Failed to export geospatial data: {e}")
            return {'error': str(e)}
    
    # Performance Monitoring
    def get_spatial_metrics(self) -> Dict[str, Any]:
        """Get spatial operations performance metrics"""
        return {
            'queries_executed': self.spatial_metrics['queries_executed'],
            'transformations_performed': self.spatial_metrics['transformations_performed'],
            'indexes_created': self.spatial_metrics['indexes_created'],
            'exports_completed': self.spatial_metrics['exports_completed']
        }
    
    async def analyze_query_performance(self, 
                                      table_name: str,
                                      sample_queries: List[str]) -> Dict[str, Any]:
        """Analyze spatial query performance"""
        try:
            async with self.db.get_postgresql_session() as session:
                performance_results = {}
                
                for i, query in enumerate(sample_queries):
                    # Enable query timing
                    await session.execute("SET track_io_timing = on")
                    
                    # Execute EXPLAIN ANALYZE
                    explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
                    
                    start_time = datetime.now()
                    result = await session.execute(explain_query)
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    explain_result = result.fetchone()[0]
                    
                    performance_results[f'query_{i+1}'] = {
                        'query': query,
                        'execution_time_seconds': execution_time,
                        'explain_analyze': explain_result
                    }
                
                return performance_results
                
        except Exception as e:
            self.logger.error(f"Failed to analyze query performance: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up GIS PostgreSQL operations")
        
        # Clear caches
        self.analysis_cache.clear()
        self.transformers.clear()
        
        self.logger.info("GIS PostgreSQL operations cleanup complete")


# Utility functions for common spatial operations
class SpatialUtils:
    """Utility functions for spatial operations"""
    
    @staticmethod
    def create_point_from_coordinates(longitude: float, latitude: float, elevation: Optional[float] = None) -> str:
        """Create WKT point from coordinates"""
        if elevation is not None:
            return f"POINT Z ({longitude} {latitude} {elevation})"
        else:
            return f"POINT({longitude} {latitude})"
    
    @staticmethod
    def create_polygon_from_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> str:
        """Create WKT polygon from bounding box"""
        return f"""POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, 
                           {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"""
    
    @staticmethod
    def validate_wkt_geometry(wkt_string: str) -> bool:
        """Validate WKT geometry string"""
        try:
            from shapely import wkt
            geom = wkt.loads(wkt_string)
            return geom.is_valid
        except Exception:
            return False
    
    @staticmethod
    def convert_geojson_to_wkt(geojson_geometry: Dict) -> str:
        """Convert GeoJSON geometry to WKT"""
        try:
            from shapely.geometry import shape
            geom = shape(geojson_geometry)
            return geom.wkt
        except Exception as e:
            raise ValueError(f"Invalid GeoJSON geometry: {e}")
    
    @staticmethod
    def calculate_distance_meters(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two points in meters using Haversine formula"""
        from math import radians, cos, sin, asin, sqrt
        
        lon1, lat1 = point1
        lon2, lat2 = point2
        
        # Convert to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth radius in meters
        r = 6371000
        
        return c * r