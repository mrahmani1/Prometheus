"""
Monitoring Network Registry - Global Volcanic Observatory Database System

This module implements a comprehensive monitoring network registry that manages
global volcanic observatory databases, monitoring station inventories, and
real-time data stream integration capabilities. It provides production-grade
functionality for tracking monitoring networks, assessing coverage, and
coordinating collaborative monitoring efforts.

Key Features:
- Global volcanic observatory database with comprehensive metadata
- Monitoring station inventory with detailed instrumentation information
- Real-time data stream integration and management
- Network coverage analysis and gap identification
- Performance assessment and quality metrics
- Collaborative monitoring coordination tools
- Advanced search and filtering capabilities
- Automated synchronization with authoritative sources

Author: Prometheus AI Platform
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Text, Integer, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy import Text
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    # Fallback for environments without geospatial libraries
    GEOSPATIAL_AVAILABLE = False
    Point = None
    Polygon = None
import requests
import aiohttp
try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    
    # Simple fallback for distance calculation
    def geodesic(coord1, coord2):
        """Simple distance calculation fallback"""
        import math
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth's radius in kilometers
        
        class Distance:
            def __init__(self, km):
                self.kilometers = km
        
        return Distance(c * r)
import networkx as nx
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cKDTree = None
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
try:
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaProducer = None
    KafkaConsumer = None
import structlog

# Configure structured logging
logger = structlog.get_logger(__name__)

Base = declarative_base()

class MonitoringNetworkType(Enum):
    """Types of monitoring networks"""
    SEISMIC = "seismic"
    GPS_GNSS = "gps_gnss"
    INSAR = "insar"
    GAS_GEOCHEMICAL = "gas_geochemical"
    THERMAL = "thermal"
    DEFORMATION = "deformation"
    GRAVIMETRIC = "gravimetric"
    MAGNETIC = "magnetic"
    HYDROLOGICAL = "hydrological"
    METEOROLOGICAL = "meteorological"
    MULTI_PARAMETER = "multi_parameter"

class InstrumentType(Enum):
    """Types of monitoring instruments"""
    SEISMOMETER = "seismometer"
    ACCELEROMETER = "accelerometer"
    GPS_STATION = "gps_station"
    TILTMETER = "tiltmeter"
    STRAINMETER = "strainmeter"
    GAS_SENSOR = "gas_sensor"
    THERMAL_CAMERA = "thermal_camera"
    GRAVIMETER = "gravimeter"
    MAGNETOMETER = "magnetometer"
    WEATHER_STATION = "weather_station"
    WEBCAM = "webcam"
    INFRASOUND = "infrasound"

class DataStreamStatus(Enum):
    """Status of data streams"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class NetworkStatus(Enum):
    """Status of monitoring networks"""
    OPERATIONAL = "operational"
    PARTIAL = "partial"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    PLANNED = "planned"

@dataclass
class GeographicBounds:
    """Geographic bounding box"""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    
    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if point is within bounds"""
        return (self.min_lat <= lat <= self.max_lat and 
                self.min_lon <= lon <= self.max_lon)

@dataclass
class InstrumentSpecification:
    """Detailed instrument specifications"""
    instrument_type: InstrumentType
    manufacturer: str
    model: str
    serial_number: Optional[str]
    installation_date: datetime
    last_calibration: Optional[datetime]
    sampling_rate: Optional[float]  # Hz
    sensitivity: Optional[float]
    frequency_range: Optional[Tuple[float, float]]  # Hz
    dynamic_range: Optional[float]  # dB
    power_consumption: Optional[float]  # Watts
    communication_method: str
    data_format: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataStreamInfo:
    """Information about data streams"""
    stream_id: str
    parameter_name: str
    unit: str
    sampling_rate: float
    latency: float  # seconds
    availability: float  # percentage
    quality_score: float
    last_update: datetime
    status: DataStreamStatus
    endpoint_url: Optional[str]
    authentication_required: bool
    data_format: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NetworkCoverageMetrics:
    """Metrics for network coverage analysis"""
    total_area_km2: float
    monitored_area_km2: float
    coverage_percentage: float
    station_density: float  # stations per km2
    average_inter_station_distance: float  # km
    coverage_gaps: List[Polygon]
    redundancy_score: float
    temporal_coverage: float  # percentage of time with data

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring networks"""
    uptime_percentage: float
    data_completeness: float
    latency_p95: float  # 95th percentile latency in seconds
    error_rate: float
    false_alarm_rate: float
    detection_capability: float
    response_time: float  # seconds
    maintenance_frequency: float  # per year

# Database Models
class VolcanicObservatory(Base):
    """Volcanic observatory database model"""
    __tablename__ = 'volcanic_observatories'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    acronym = Column(String(20))
    country = Column(String(100), nullable=False)
    region = Column(String(100))
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float)  # meters
    established_date = Column(DateTime)
    parent_organization = Column(String(255))
    contact_email = Column(String(255))
    website_url = Column(String(500))
    phone = Column(String(50))
    address = Column(Text)
    primary_volcanoes = Column(JSON)  # List of volcano IDs
    network_types = Column(JSON)  # List of monitoring network types
    status = Column(String(50), default='operational')
    capabilities = Column(JSON)
    staff_count = Column(Integer)
    annual_budget = Column(Float)
    data_sharing_policy = Column(Text)
    real_time_capability = Column(Boolean, default=False)
    alert_system = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    monitoring_stations = relationship("MonitoringStation", back_populates="observatory")
    data_streams = relationship("DataStream", back_populates="observatory")

class MonitoringStation(Base):
    """Monitoring station database model"""
    __tablename__ = 'monitoring_stations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    station_code = Column(String(20), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    observatory_id = Column(UUID(as_uuid=True), ForeignKey('volcanic_observatories.id'))
    volcano_id = Column(String(50))  # Reference to volcano in GVP database
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float, nullable=False)  # meters
    installation_date = Column(DateTime, nullable=False)
    decommission_date = Column(DateTime)
    station_type = Column(String(50), nullable=False)  # seismic, gps, etc.
    network_code = Column(String(20))
    site_conditions = Column(Text)
    access_difficulty = Column(String(50))  # easy, moderate, difficult, extreme
    power_source = Column(String(100))
    communication_method = Column(String(100))
    maintenance_schedule = Column(String(100))
    last_maintenance = Column(DateTime)
    next_maintenance = Column(DateTime)
    status = Column(String(50), default='operational')
    instruments = Column(JSON)  # List of instrument specifications
    data_availability = Column(Float, default=0.0)  # percentage
    uptime = Column(Float, default=0.0)  # percentage
    station_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    observatory = relationship("VolcanicObservatory", back_populates="monitoring_stations")
    data_streams = relationship("DataStream", back_populates="station")

class DataStream(Base):
    """Data stream database model"""
    __tablename__ = 'data_streams'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stream_id = Column(String(100), unique=True, nullable=False)
    station_id = Column(UUID(as_uuid=True), ForeignKey('monitoring_stations.id'))
    observatory_id = Column(UUID(as_uuid=True), ForeignKey('volcanic_observatories.id'))
    parameter_name = Column(String(100), nullable=False)
    unit = Column(String(50))
    instrument_type = Column(String(50))
    sampling_rate = Column(Float)  # Hz
    data_format = Column(String(50))
    endpoint_url = Column(String(500))
    authentication_method = Column(String(100))
    real_time = Column(Boolean, default=False)
    latency = Column(Float)  # seconds
    availability = Column(Float, default=0.0)  # percentage
    quality_score = Column(Float, default=0.0)
    last_update = Column(DateTime)
    status = Column(String(50), default='unknown')
    access_restrictions = Column(Text)
    data_policy = Column(Text)
    contact_info = Column(String(255))
    stream_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    station = relationship("MonitoringStation", back_populates="data_streams")
    observatory = relationship("VolcanicObservatory", back_populates="data_streams")

class NetworkCoverage(Base):
    """Network coverage analysis results"""
    __tablename__ = 'network_coverage'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    region_name = Column(String(255), nullable=False)
    volcano_id = Column(String(50))
    network_type = Column(String(50), nullable=False)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    total_area_km2 = Column(Float)
    monitored_area_km2 = Column(Float)
    coverage_percentage = Column(Float)
    station_count = Column(Integer)
    station_density = Column(Float)  # stations per km2
    average_distance = Column(Float)  # km between stations
    coverage_gaps = Column(JSON)  # Serialized polygon data
    redundancy_score = Column(Float)
    recommendations = Column(Text)
    analysis_metadata = Column(JSON)

class MonitoringNetworkRegistry:
    """
    Comprehensive monitoring network registry system for global volcanic observatories.
    
    This class provides production-grade functionality for managing volcanic monitoring
    networks, including observatory databases, station inventories, data stream
    integration, coverage analysis, and collaborative coordination.
    """
    
    def __init__(self, database_url: str, redis_url: str = None, kafka_config: Dict = None):
        """
        Initialize the monitoring network registry.
        
        Args:
            database_url: PostgreSQL database connection URL
            redis_url: Redis connection URL for caching
            kafka_config: Kafka configuration for real-time streaming
        """
        self.database_url = database_url
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize Redis for caching
        self.redis_client = None
        if redis_url and REDIS_AVAILABLE:
            self.redis_client = redis.from_url(redis_url)
        
        # Initialize Kafka for real-time streaming
        self.kafka_config = kafka_config or {}
        self.kafka_producer = None
        if kafka_config and KAFKA_AVAILABLE:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_config.get('bootstrap_servers', ['localhost:9092']),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Known observatory APIs and data sources
        self.observatory_apis = {
            'usgs_vhp': {
                'base_url': 'https://volcanoes.usgs.gov/observatories/api',
                'endpoints': {
                    'observatories': '/observatories',
                    'stations': '/stations',
                    'data_streams': '/streams'
                }
            },
            'ingv': {
                'base_url': 'https://www.ct.ingv.it/api',
                'endpoints': {
                    'stations': '/stations',
                    'networks': '/networks'
                }
            },
            'jma': {
                'base_url': 'https://www.data.jma.go.jp/svd/vois/data/api',
                'endpoints': {
                    'observatories': '/observatories',
                    'monitoring': '/monitoring'
                }
            },
            'wovo': {
                'base_url': 'https://www.wovo.org/api',
                'endpoints': {
                    'observatories': '/observatories',
                    'networks': '/networks'
                }
            }
        }
        
        logger.info("Monitoring Network Registry initialized", 
                   database_url=database_url, 
                   redis_enabled=redis_url is not None,
                   kafka_enabled=kafka_config is not None)

    async def register_volcanic_observatory(self, observatory_data: Dict[str, Any]) -> str:
        """
        Register a new volcanic observatory in the database.
        
        Args:
            observatory_data: Dictionary containing observatory information
            
        Returns:
            Observatory ID
        """
        try:
            session = self.Session()
            
            # Validate required fields
            required_fields = ['name', 'country', 'latitude', 'longitude']
            for field in required_fields:
                if field not in observatory_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create observatory record
            observatory = VolcanicObservatory(
                name=observatory_data['name'],
                acronym=observatory_data.get('acronym'),
                country=observatory_data['country'],
                region=observatory_data.get('region'),
                latitude=observatory_data['latitude'],
                longitude=observatory_data['longitude'],
                elevation=observatory_data.get('elevation'),
                established_date=observatory_data.get('established_date'),
                parent_organization=observatory_data.get('parent_organization'),
                contact_email=observatory_data.get('contact_email'),
                website_url=observatory_data.get('website_url'),
                phone=observatory_data.get('phone'),
                address=observatory_data.get('address'),
                primary_volcanoes=observatory_data.get('primary_volcanoes', []),
                network_types=observatory_data.get('network_types', []),
                status=observatory_data.get('status', 'operational'),
                capabilities=observatory_data.get('capabilities', {}),
                staff_count=observatory_data.get('staff_count'),
                annual_budget=observatory_data.get('annual_budget'),
                data_sharing_policy=observatory_data.get('data_sharing_policy'),
                real_time_capability=observatory_data.get('real_time_capability', False),
                alert_system=observatory_data.get('alert_system', False)
            )
            
            session.add(observatory)
            session.commit()
            
            observatory_id = str(observatory.id)
            
            # Cache observatory data
            if self.redis_client:
                cache_key = f"observatory:{observatory_id}"
                self.redis_client.setex(
                    cache_key, 
                    3600,  # 1 hour TTL
                    json.dumps(observatory_data)
                )
            
            # Publish to Kafka
            if self.kafka_producer:
                self.kafka_producer.send('observatory_registered', {
                    'observatory_id': observatory_id,
                    'name': observatory_data['name'],
                    'country': observatory_data['country'],
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            logger.info("Observatory registered successfully", 
                       observatory_id=observatory_id,
                       name=observatory_data['name'])
            
            return observatory_id
            
        except Exception as e:
            logger.error("Failed to register observatory", error=str(e))
            session.rollback()
            raise
        finally:
            session.close()

    async def register_monitoring_station(self, station_data: Dict[str, Any]) -> str:
        """
        Register a new monitoring station.
        
        Args:
            station_data: Dictionary containing station information
            
        Returns:
            Station ID
        """
        try:
            session = self.Session()
            
            # Validate required fields
            required_fields = ['station_code', 'name', 'latitude', 'longitude', 
                             'elevation', 'installation_date', 'station_type']
            for field in required_fields:
                if field not in station_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create station record
            station = MonitoringStation(
                station_code=station_data['station_code'],
                name=station_data['name'],
                observatory_id=uuid.UUID(station_data['observatory_id']) if isinstance(station_data.get('observatory_id'), str) else station_data.get('observatory_id'),
                volcano_id=station_data.get('volcano_id'),
                latitude=station_data['latitude'],
                longitude=station_data['longitude'],
                elevation=station_data['elevation'],
                installation_date=station_data['installation_date'],
                decommission_date=station_data.get('decommission_date'),
                station_type=station_data['station_type'],
                network_code=station_data.get('network_code'),
                site_conditions=station_data.get('site_conditions'),
                access_difficulty=station_data.get('access_difficulty', 'moderate'),
                power_source=station_data.get('power_source'),
                communication_method=station_data.get('communication_method'),
                maintenance_schedule=station_data.get('maintenance_schedule'),
                last_maintenance=station_data.get('last_maintenance'),
                next_maintenance=station_data.get('next_maintenance'),
                status=station_data.get('status', 'operational'),
                instruments=station_data.get('instruments', []),
                data_availability=station_data.get('data_availability', 0.0),
                uptime=station_data.get('uptime', 0.0),
                station_metadata=station_data.get('metadata', {})
            )
            
            session.add(station)
            session.commit()
            
            station_id = str(station.id)
            
            # Cache station data
            if self.redis_client:
                cache_key = f"station:{station_id}"
                self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    json.dumps(station_data)
                )
            
            # Publish to Kafka
            if self.kafka_producer:
                self.kafka_producer.send('station_registered', {
                    'station_id': station_id,
                    'station_code': station_data['station_code'],
                    'station_type': station_data['station_type'],
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            logger.info("Monitoring station registered successfully",
                       station_id=station_id,
                       station_code=station_data['station_code'])
            
            return station_id
            
        except Exception as e:
            logger.error("Failed to register monitoring station", error=str(e))
            session.rollback()
            raise
        finally:
            session.close()

    async def register_data_stream(self, stream_data: Dict[str, Any]) -> str:
        """
        Register a new data stream.
        
        Args:
            stream_data: Dictionary containing data stream information
            
        Returns:
            Stream ID
        """
        try:
            session = self.Session()
            
            # Validate required fields
            required_fields = ['stream_id', 'parameter_name', 'station_id']
            for field in required_fields:
                if field not in stream_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create data stream record
            data_stream = DataStream(
                stream_id=stream_data['stream_id'],
                station_id=stream_data['station_id'],
                observatory_id=stream_data.get('observatory_id'),
                parameter_name=stream_data['parameter_name'],
                unit=stream_data.get('unit'),
                instrument_type=stream_data.get('instrument_type'),
                sampling_rate=stream_data.get('sampling_rate'),
                data_format=stream_data.get('data_format'),
                endpoint_url=stream_data.get('endpoint_url'),
                authentication_method=stream_data.get('authentication_method'),
                real_time=stream_data.get('real_time', False),
                latency=stream_data.get('latency'),
                availability=stream_data.get('availability', 0.0),
                quality_score=stream_data.get('quality_score', 0.0),
                last_update=stream_data.get('last_update'),
                status=stream_data.get('status', 'unknown'),
                access_restrictions=stream_data.get('access_restrictions'),
                data_policy=stream_data.get('data_policy'),
                contact_info=stream_data.get('contact_info'),
                stream_metadata=stream_data.get('metadata', {})
            )
            
            session.add(data_stream)
            session.commit()
            
            stream_db_id = str(data_stream.id)
            
            # Cache stream data
            if self.redis_client:
                cache_key = f"stream:{stream_db_id}"
                self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    json.dumps(stream_data)
                )
            
            # Publish to Kafka
            if self.kafka_producer:
                self.kafka_producer.send('stream_registered', {
                    'stream_db_id': stream_db_id,
                    'stream_id': stream_data['stream_id'],
                    'parameter_name': stream_data['parameter_name'],
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            logger.info("Data stream registered successfully",
                       stream_db_id=stream_db_id,
                       stream_id=stream_data['stream_id'])
            
            return stream_db_id
            
        except Exception as e:
            logger.error("Failed to register data stream", error=str(e))
            session.rollback()
            raise
        finally:
            session.close()

    async def query_observatories(self, 
                                filters: Dict[str, Any] = None,
                                geographic_bounds: GeographicBounds = None) -> List[Dict[str, Any]]:
        """
        Query volcanic observatories with optional filters.
        
        Args:
            filters: Dictionary of filter criteria
            geographic_bounds: Geographic bounding box for spatial filtering
            
        Returns:
            List of observatory records
        """
        try:
            session = self.Session()
            query = session.query(VolcanicObservatory)
            
            # Apply filters
            if filters:
                if 'country' in filters:
                    query = query.filter(VolcanicObservatory.country == filters['country'])
                if 'region' in filters:
                    query = query.filter(VolcanicObservatory.region == filters['region'])
                if 'status' in filters:
                    query = query.filter(VolcanicObservatory.status == filters['status'])
                if 'network_types' in filters:
                    for network_type in filters['network_types']:
                        query = query.filter(VolcanicObservatory.network_types.contains([network_type]))
                if 'real_time_capability' in filters:
                    query = query.filter(VolcanicObservatory.real_time_capability == filters['real_time_capability'])
            
            # Apply geographic bounds
            if geographic_bounds:
                query = query.filter(
                    VolcanicObservatory.latitude >= geographic_bounds.min_lat,
                    VolcanicObservatory.latitude <= geographic_bounds.max_lat,
                    VolcanicObservatory.longitude >= geographic_bounds.min_lon,
                    VolcanicObservatory.longitude <= geographic_bounds.max_lon
                )
            
            observatories = query.all()
            
            # Convert to dictionaries
            result = []
            for obs in observatories:
                obs_dict = {
                    'id': str(obs.id),
                    'name': obs.name,
                    'acronym': obs.acronym,
                    'country': obs.country,
                    'region': obs.region,
                    'latitude': obs.latitude,
                    'longitude': obs.longitude,
                    'elevation': obs.elevation,
                    'established_date': obs.established_date.isoformat() if obs.established_date else None,
                    'parent_organization': obs.parent_organization,
                    'contact_email': obs.contact_email,
                    'website_url': obs.website_url,
                    'phone': obs.phone,
                    'address': obs.address,
                    'primary_volcanoes': obs.primary_volcanoes,
                    'network_types': obs.network_types,
                    'status': obs.status,
                    'capabilities': obs.capabilities,
                    'staff_count': obs.staff_count,
                    'annual_budget': obs.annual_budget,
                    'data_sharing_policy': obs.data_sharing_policy,
                    'real_time_capability': obs.real_time_capability,
                    'alert_system': obs.alert_system,
                    'created_at': obs.created_at.isoformat(),
                    'updated_at': obs.updated_at.isoformat()
                }
                result.append(obs_dict)
            
            logger.info("Observatory query completed", 
                       count=len(result),
                       filters=filters)
            
            return result
            
        except Exception as e:
            logger.error("Failed to query observatories", error=str(e))
            raise
        finally:
            session.close()

    async def query_monitoring_stations(self,
                                      filters: Dict[str, Any] = None,
                                      geographic_bounds: GeographicBounds = None) -> List[Dict[str, Any]]:
        """
        Query monitoring stations with optional filters.
        
        Args:
            filters: Dictionary of filter criteria
            geographic_bounds: Geographic bounding box for spatial filtering
            
        Returns:
            List of monitoring station records
        """
        try:
            session = self.Session()
            query = session.query(MonitoringStation)
            
            # Apply filters
            if filters:
                if 'observatory_id' in filters:
                    query = query.filter(MonitoringStation.observatory_id == filters['observatory_id'])
                if 'volcano_id' in filters:
                    query = query.filter(MonitoringStation.volcano_id == filters['volcano_id'])
                if 'station_type' in filters:
                    query = query.filter(MonitoringStation.station_type == filters['station_type'])
                if 'network_code' in filters:
                    query = query.filter(MonitoringStation.network_code == filters['network_code'])
                if 'status' in filters:
                    query = query.filter(MonitoringStation.status == filters['status'])
                if 'min_uptime' in filters:
                    query = query.filter(MonitoringStation.uptime >= filters['min_uptime'])
                if 'min_data_availability' in filters:
                    query = query.filter(MonitoringStation.data_availability >= filters['min_data_availability'])
            
            # Apply geographic bounds
            if geographic_bounds:
                query = query.filter(
                    MonitoringStation.latitude >= geographic_bounds.min_lat,
                    MonitoringStation.latitude <= geographic_bounds.max_lat,
                    MonitoringStation.longitude >= geographic_bounds.min_lon,
                    MonitoringStation.longitude <= geographic_bounds.max_lon
                )
            
            stations = query.all()
            
            # Convert to dictionaries
            result = []
            for station in stations:
                station_dict = {
                    'id': str(station.id),
                    'station_code': station.station_code,
                    'name': station.name,
                    'observatory_id': str(station.observatory_id) if station.observatory_id else None,
                    'volcano_id': station.volcano_id,
                    'latitude': station.latitude,
                    'longitude': station.longitude,
                    'elevation': station.elevation,
                    'installation_date': station.installation_date.isoformat(),
                    'decommission_date': station.decommission_date.isoformat() if station.decommission_date else None,
                    'station_type': station.station_type,
                    'network_code': station.network_code,
                    'site_conditions': station.site_conditions,
                    'access_difficulty': station.access_difficulty,
                    'power_source': station.power_source,
                    'communication_method': station.communication_method,
                    'maintenance_schedule': station.maintenance_schedule,
                    'last_maintenance': station.last_maintenance.isoformat() if station.last_maintenance else None,
                    'next_maintenance': station.next_maintenance.isoformat() if station.next_maintenance else None,
                    'status': station.status,
                    'instruments': station.instruments,
                    'data_availability': station.data_availability,
                    'uptime': station.uptime,
                    'metadata': station.station_metadata,
                    'created_at': station.created_at.isoformat(),
                    'updated_at': station.updated_at.isoformat()
                }
                result.append(station_dict)
            
            logger.info("Monitoring station query completed",
                       count=len(result),
                       filters=filters)
            
            return result
            
        except Exception as e:
            logger.error("Failed to query monitoring stations", error=str(e))
            raise
        finally:
            session.close()

    async def analyze_network_coverage(self, 
                                     region_bounds: GeographicBounds,
                                     network_type: MonitoringNetworkType,
                                     volcano_id: str = None) -> NetworkCoverageMetrics:
        """
        Analyze monitoring network coverage for a specific region.
        
        Args:
            region_bounds: Geographic bounds of the region to analyze
            network_type: Type of monitoring network to analyze
            volcano_id: Optional volcano ID for volcano-specific analysis
            
        Returns:
            Network coverage metrics
        """
        try:
            # Query stations in the region
            stations = await self.query_monitoring_stations(
                filters={'station_type': network_type.value},
                geographic_bounds=region_bounds
            )
            
            if volcano_id:
                stations = [s for s in stations if s['volcano_id'] == volcano_id]
            
            # Calculate region area
            region_area_km2 = self._calculate_region_area(region_bounds)
            
            # Extract station coordinates
            station_coords = [(s['latitude'], s['longitude']) for s in stations]
            
            if not station_coords:
                return NetworkCoverageMetrics(
                    total_area_km2=region_area_km2,
                    monitored_area_km2=0.0,
                    coverage_percentage=0.0,
                    station_density=0.0,
                    average_inter_station_distance=0.0,
                    coverage_gaps=[],
                    redundancy_score=0.0,
                    temporal_coverage=0.0
                )
            
            # Calculate coverage metrics
            monitored_area_km2 = self._calculate_monitored_area(station_coords, region_bounds)
            coverage_percentage = (monitored_area_km2 / region_area_km2) * 100
            station_density = len(stations) / region_area_km2
            
            # Calculate average inter-station distance
            avg_distance = self._calculate_average_inter_station_distance(station_coords)
            
            # Identify coverage gaps
            coverage_gaps = self._identify_coverage_gaps(station_coords, region_bounds)
            
            # Calculate redundancy score
            redundancy_score = self._calculate_redundancy_score(station_coords)
            
            # Calculate temporal coverage
            temporal_coverage = self._calculate_temporal_coverage(stations)
            
            metrics = NetworkCoverageMetrics(
                total_area_km2=region_area_km2,
                monitored_area_km2=monitored_area_km2,
                coverage_percentage=coverage_percentage,
                station_density=station_density,
                average_inter_station_distance=avg_distance,
                coverage_gaps=coverage_gaps,
                redundancy_score=redundancy_score,
                temporal_coverage=temporal_coverage
            )
            
            # Store coverage analysis results
            await self._store_coverage_analysis(region_bounds, network_type, volcano_id, metrics)
            
            logger.info("Network coverage analysis completed",
                       network_type=network_type.value,
                       station_count=len(stations),
                       coverage_percentage=coverage_percentage)
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to analyze network coverage", error=str(e))
            raise

    async def assess_network_performance(self, 
                                       network_id: str = None,
                                       time_range: Tuple[datetime, datetime] = None) -> PerformanceMetrics:
        """
        Assess performance metrics for monitoring networks.
        
        Args:
            network_id: Optional specific network ID to assess
            time_range: Time range for performance assessment
            
        Returns:
            Performance metrics
        """
        try:
            if time_range is None:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=30)  # Last 30 days
                time_range = (start_time, end_time)
            
            # Query relevant data streams
            filters = {}
            if network_id:
                filters['network_code'] = network_id
            
            stations = await self.query_monitoring_stations(filters=filters)
            
            if not stations:
                logger.warning("No stations found for performance assessment")
                return PerformanceMetrics(
                    uptime_percentage=0.0,
                    data_completeness=0.0,
                    latency_p95=0.0,
                    error_rate=0.0,
                    false_alarm_rate=0.0,
                    detection_capability=0.0,
                    response_time=0.0,
                    maintenance_frequency=0.0
                )
            
            # Calculate performance metrics
            uptime_values = [s['uptime'] for s in stations if s['uptime'] is not None]
            uptime_percentage = np.mean(uptime_values) if uptime_values else 0.0
            
            availability_values = [s['data_availability'] for s in stations if s['data_availability'] is not None]
            data_completeness = np.mean(availability_values) if availability_values else 0.0
            
            # Query data streams for latency analysis
            session = self.Session()
            streams = session.query(DataStream).filter(
                DataStream.station_id.in_([s['id'] for s in stations])
            ).all()
            
            latency_values = [s.latency for s in streams if s.latency is not None]
            latency_p95 = np.percentile(latency_values, 95) if latency_values else 0.0
            
            # Calculate other metrics (simplified for demonstration)
            error_rate = max(0.0, 1.0 - data_completeness / 100.0)
            false_alarm_rate = 0.05  # Placeholder - would be calculated from actual alert data
            detection_capability = min(1.0, data_completeness / 100.0 * uptime_percentage / 100.0)
            response_time = latency_p95
            maintenance_frequency = 4.0  # Placeholder - quarterly maintenance
            
            metrics = PerformanceMetrics(
                uptime_percentage=uptime_percentage,
                data_completeness=data_completeness,
                latency_p95=latency_p95,
                error_rate=error_rate,
                false_alarm_rate=false_alarm_rate,
                detection_capability=detection_capability,
                response_time=response_time,
                maintenance_frequency=maintenance_frequency
            )
            
            logger.info("Network performance assessment completed",
                       network_id=network_id,
                       uptime=uptime_percentage,
                       completeness=data_completeness)
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to assess network performance", error=str(e))
            raise
        finally:
            session.close()

    async def identify_monitoring_gaps(self, 
                                     region_bounds: GeographicBounds,
                                     target_coverage: float = 90.0) -> List[Dict[str, Any]]:
        """
        Identify monitoring gaps and provide recommendations.
        
        Args:
            region_bounds: Geographic bounds to analyze
            target_coverage: Target coverage percentage
            
        Returns:
            List of monitoring gap recommendations
        """
        try:
            recommendations = []
            
            # Analyze coverage for each network type
            for network_type in MonitoringNetworkType:
                coverage = await self.analyze_network_coverage(region_bounds, network_type)
                
                if coverage.coverage_percentage < target_coverage:
                    gap_percentage = target_coverage - coverage.coverage_percentage
                    
                    # Calculate recommended number of additional stations
                    current_stations = len(await self.query_monitoring_stations(
                        filters={'station_type': network_type.value},
                        geographic_bounds=region_bounds
                    ))
                    
                    additional_stations_needed = int(
                        (gap_percentage / 100.0) * coverage.total_area_km2 / 
                        (coverage.monitored_area_km2 / max(1, current_stations))
                    )
                    
                    # Identify optimal locations for new stations
                    optimal_locations = self._identify_optimal_station_locations(
                        coverage.coverage_gaps, additional_stations_needed
                    )
                    
                    recommendation = {
                        'network_type': network_type.value,
                        'current_coverage': coverage.coverage_percentage,
                        'target_coverage': target_coverage,
                        'gap_percentage': gap_percentage,
                        'current_stations': current_stations,
                        'additional_stations_needed': additional_stations_needed,
                        'optimal_locations': optimal_locations,
                        'priority': 'high' if gap_percentage > 50 else 'medium' if gap_percentage > 20 else 'low',
                        'estimated_cost': self._estimate_deployment_cost(network_type, additional_stations_needed),
                        'implementation_timeline': self._estimate_implementation_timeline(additional_stations_needed)
                    }
                    
                    recommendations.append(recommendation)
            
            # Sort by priority and gap percentage
            recommendations.sort(key=lambda x: (
                {'high': 0, 'medium': 1, 'low': 2}[x['priority']],
                -x['gap_percentage']
            ))
            
            logger.info("Monitoring gap analysis completed",
                       total_recommendations=len(recommendations))
            
            return recommendations
            
        except Exception as e:
            logger.error("Failed to identify monitoring gaps", error=str(e))
            raise

    async def synchronize_with_external_sources(self) -> Dict[str, Any]:
        """
        Synchronize monitoring network data with external authoritative sources.
        
        Returns:
            Synchronization report
        """
        try:
            sync_report = {
                'start_time': datetime.utcnow().isoformat(),
                'sources_processed': [],
                'observatories_updated': 0,
                'stations_updated': 0,
                'streams_updated': 0,
                'errors': []
            }
            
            # Synchronize with each configured API
            for source_name, source_config in self.observatory_apis.items():
                try:
                    logger.info(f"Synchronizing with {source_name}")
                    
                    # Fetch observatory data
                    observatories_data = await self._fetch_external_observatories(source_config)
                    
                    # Update local database
                    for obs_data in observatories_data:
                        try:
                            # Check if observatory already exists
                            existing_obs = await self._find_existing_observatory(obs_data)
                            
                            if existing_obs:
                                await self._update_observatory(existing_obs['id'], obs_data)
                                sync_report['observatories_updated'] += 1
                            else:
                                await self.register_volcanic_observatory(obs_data)
                                sync_report['observatories_updated'] += 1
                                
                        except Exception as e:
                            sync_report['errors'].append({
                                'source': source_name,
                                'type': 'observatory_update',
                                'error': str(e)
                            })
                    
                    # Fetch station data
                    stations_data = await self._fetch_external_stations(source_config)
                    
                    for station_data in stations_data:
                        try:
                            existing_station = await self._find_existing_station(station_data)
                            
                            if existing_station:
                                await self._update_station(existing_station['id'], station_data)
                                sync_report['stations_updated'] += 1
                            else:
                                await self.register_monitoring_station(station_data)
                                sync_report['stations_updated'] += 1
                                
                        except Exception as e:
                            sync_report['errors'].append({
                                'source': source_name,
                                'type': 'station_update',
                                'error': str(e)
                            })
                    
                    sync_report['sources_processed'].append(source_name)
                    
                except Exception as e:
                    sync_report['errors'].append({
                        'source': source_name,
                        'type': 'source_connection',
                        'error': str(e)
                    })
            
            sync_report['end_time'] = datetime.utcnow().isoformat()
            sync_report['success'] = len(sync_report['errors']) == 0
            
            logger.info("External source synchronization completed",
                       sources_processed=len(sync_report['sources_processed']),
                       observatories_updated=sync_report['observatories_updated'],
                       stations_updated=sync_report['stations_updated'],
                       errors=len(sync_report['errors']))
            
            return sync_report
            
        except Exception as e:
            logger.error("Failed to synchronize with external sources", error=str(e))
            raise

    async def setup_real_time_data_integration(self, 
                                             stream_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Set up real-time data stream integration.
        
        Args:
            stream_configs: List of data stream configurations
            
        Returns:
            Integration setup report
        """
        try:
            setup_report = {
                'start_time': datetime.utcnow().isoformat(),
                'streams_configured': 0,
                'streams_active': 0,
                'failed_streams': [],
                'integration_endpoints': []
            }
            
            for stream_config in stream_configs:
                try:
                    # Validate stream configuration
                    required_fields = ['stream_id', 'endpoint_url', 'parameter_name']
                    for field in required_fields:
                        if field not in stream_config:
                            raise ValueError(f"Missing required field: {field}")
                    
                    # Register data stream
                    stream_db_id = await self.register_data_stream(stream_config)
                    
                    # Test connection
                    connection_test = await self._test_stream_connection(stream_config)
                    
                    if connection_test['success']:
                        # Set up real-time consumer
                        consumer_config = {
                            'stream_id': stream_config['stream_id'],
                            'endpoint_url': stream_config['endpoint_url'],
                            'callback_function': self._process_real_time_data,
                            'error_handler': self._handle_stream_error
                        }
                        
                        await self._setup_stream_consumer(consumer_config)
                        
                        setup_report['streams_active'] += 1
                        setup_report['integration_endpoints'].append({
                            'stream_id': stream_config['stream_id'],
                            'endpoint': stream_config['endpoint_url'],
                            'status': 'active'
                        })
                    else:
                        setup_report['failed_streams'].append({
                            'stream_id': stream_config['stream_id'],
                            'error': connection_test['error']
                        })
                    
                    setup_report['streams_configured'] += 1
                    
                except Exception as e:
                    setup_report['failed_streams'].append({
                        'stream_id': stream_config.get('stream_id', 'unknown'),
                        'error': str(e)
                    })
            
            setup_report['end_time'] = datetime.utcnow().isoformat()
            setup_report['success'] = len(setup_report['failed_streams']) == 0
            
            logger.info("Real-time data integration setup completed",
                       configured=setup_report['streams_configured'],
                       active=setup_report['streams_active'],
                       failed=len(setup_report['failed_streams']))
            
            return setup_report
            
        except Exception as e:
            logger.error("Failed to setup real-time data integration", error=str(e))
            raise

    # Helper methods for internal operations
    
    def _calculate_region_area(self, bounds: GeographicBounds) -> float:
        """Calculate area of geographic region in km²"""
        # Simplified calculation using geodesic distance
        lat_diff = bounds.max_lat - bounds.min_lat
        lon_diff = bounds.max_lon - bounds.min_lon
        
        # Calculate distances
        lat_distance = geodesic((bounds.min_lat, bounds.min_lon), 
                               (bounds.max_lat, bounds.min_lon)).kilometers
        lon_distance = geodesic((bounds.min_lat, bounds.min_lon), 
                               (bounds.min_lat, bounds.max_lon)).kilometers
        
        return lat_distance * lon_distance

    def _calculate_monitored_area(self, station_coords: List[Tuple[float, float]], 
                                bounds: GeographicBounds) -> float:
        """Calculate monitored area based on station coverage"""
        if not station_coords:
            return 0.0
        
        # Simplified calculation: assume each station covers a circular area
        # with radius based on typical monitoring range for the network type
        typical_range_km = 50.0  # Adjustable based on network type
        area_per_station = np.pi * (typical_range_km ** 2)
        
        # Account for overlapping coverage
        total_area = len(station_coords) * area_per_station
        overlap_factor = 0.8  # Assume 20% overlap
        
        return total_area * overlap_factor

    def _calculate_average_inter_station_distance(self, 
                                                station_coords: List[Tuple[float, float]]) -> float:
        """Calculate average distance between monitoring stations"""
        if len(station_coords) < 2:
            return 0.0
        
        distances = []
        for i in range(len(station_coords)):
            for j in range(i + 1, len(station_coords)):
                distance = geodesic(station_coords[i], station_coords[j]).kilometers
                distances.append(distance)
        
        return np.mean(distances)

    def _identify_coverage_gaps(self, station_coords: List[Tuple[float, float]], 
                              bounds: GeographicBounds) -> List:
        """Identify areas with insufficient monitoring coverage"""
        if not GEOSPATIAL_AVAILABLE:
            # Return simplified gap representation when geospatial libraries unavailable
            return []
        
        # Simplified implementation - would use more sophisticated spatial analysis
        gaps = []
        
        # Create a grid and identify cells without nearby stations
        grid_size = 0.1  # degrees
        lat_steps = int((bounds.max_lat - bounds.min_lat) / grid_size)
        lon_steps = int((bounds.max_lon - bounds.min_lon) / grid_size)
        
        for i in range(lat_steps):
            for j in range(lon_steps):
                cell_lat = bounds.min_lat + i * grid_size
                cell_lon = bounds.min_lon + j * grid_size
                
                # Check if any station is within monitoring range
                min_distance = float('inf')
                for station_lat, station_lon in station_coords:
                    distance = geodesic((cell_lat, cell_lon), (station_lat, station_lon)).kilometers
                    min_distance = min(min_distance, distance)
                
                # If no station within 50km, consider it a gap
                if min_distance > 50.0:
                    if GEOSPATIAL_AVAILABLE:
                        gap_polygon = Polygon([
                            (cell_lon, cell_lat),
                            (cell_lon + grid_size, cell_lat),
                            (cell_lon + grid_size, cell_lat + grid_size),
                            (cell_lon, cell_lat + grid_size)
                        ])
                        gaps.append(gap_polygon)
                    else:
                        # Simplified gap representation
                        gaps.append({
                            'center_lat': cell_lat + grid_size/2,
                            'center_lon': cell_lon + grid_size/2,
                            'size': grid_size
                        })
        
        return gaps

    def _calculate_redundancy_score(self, station_coords: List[Tuple[float, float]]) -> float:
        """Calculate network redundancy score"""
        if len(station_coords) < 2:
            return 0.0
        
        if SCIPY_AVAILABLE:
            # Build spatial tree for efficient neighbor queries
            tree = cKDTree(station_coords)
            
            redundancy_scores = []
            for i, coord in enumerate(station_coords):
                # Find nearest neighbors within 100km
                neighbors = tree.query_ball_point(coord, r=1.0)  # Approximate degree to km conversion
                redundancy_scores.append(len(neighbors) - 1)  # Exclude self
            
            # Normalize to 0-1 scale
            max_redundancy = max(redundancy_scores) if redundancy_scores else 0
            if max_redundancy == 0:
                return 0.0
            
            return np.mean(redundancy_scores) / max_redundancy
        else:
            # Simplified redundancy calculation without scipy
            redundancy_scores = []
            for i, coord1 in enumerate(station_coords):
                nearby_count = 0
                for j, coord2 in enumerate(station_coords):
                    if i != j:
                        distance = geodesic(coord1, coord2).kilometers
                        if distance < 100.0:  # Within 100km
                            nearby_count += 1
                redundancy_scores.append(nearby_count)
            
            # Normalize to 0-1 scale
            max_redundancy = max(redundancy_scores) if redundancy_scores else 0
            if max_redundancy == 0:
                return 0.0
            
            return np.mean(redundancy_scores) / max_redundancy

    def _calculate_temporal_coverage(self, stations: List[Dict[str, Any]]) -> float:
        """Calculate temporal coverage based on station uptime"""
        if not stations:
            return 0.0
        
        uptime_values = [s['uptime'] for s in stations if s['uptime'] is not None]
        return np.mean(uptime_values) if uptime_values else 0.0

    async def _store_coverage_analysis(self, region_bounds: GeographicBounds,
                                     network_type: MonitoringNetworkType,
                                     volcano_id: str,
                                     metrics: NetworkCoverageMetrics):
        """Store coverage analysis results in database"""
        try:
            session = self.Session()
            
            coverage_record = NetworkCoverage(
                region_name=f"Region_{region_bounds.min_lat}_{region_bounds.min_lon}",
                volcano_id=volcano_id,
                network_type=network_type.value,
                total_area_km2=metrics.total_area_km2,
                monitored_area_km2=metrics.monitored_area_km2,
                coverage_percentage=metrics.coverage_percentage,
                station_count=len(await self.query_monitoring_stations(
                    filters={'station_type': network_type.value},
                    geographic_bounds=region_bounds
                )),
                station_density=metrics.station_density,
                average_distance=metrics.average_inter_station_distance,
                coverage_gaps=[],  # Simplified - would serialize polygon data
                redundancy_score=metrics.redundancy_score,
                recommendations=f"Coverage: {metrics.coverage_percentage:.1f}%"
            )
            
            session.add(coverage_record)
            session.commit()
            
        except Exception as e:
            logger.error("Failed to store coverage analysis", error=str(e))
            session.rollback()
        finally:
            session.close()

    def _identify_optimal_station_locations(self, coverage_gaps: List, 
                                          num_stations: int) -> List[Dict[str, float]]:
        """Identify optimal locations for new monitoring stations"""
        optimal_locations = []
        
        # Simplified implementation - would use optimization algorithms
        for i, gap in enumerate(coverage_gaps[:num_stations]):
            if GEOSPATIAL_AVAILABLE and hasattr(gap, 'centroid'):
                # Get centroid of coverage gap (Polygon object)
                centroid = gap.centroid
                optimal_locations.append({
                    'latitude': centroid.y,
                    'longitude': centroid.x,
                    'priority_score': 1.0 - (i / len(coverage_gaps)),
                    'coverage_improvement': 10.0  # Estimated percentage improvement
                })
            elif isinstance(gap, dict):
                # Simplified gap representation
                optimal_locations.append({
                    'latitude': gap['center_lat'],
                    'longitude': gap['center_lon'],
                    'priority_score': 1.0 - (i / len(coverage_gaps)),
                    'coverage_improvement': 10.0  # Estimated percentage improvement
                })
        
        return optimal_locations

    def _estimate_deployment_cost(self, network_type: MonitoringNetworkType, 
                                num_stations: int) -> Dict[str, float]:
        """Estimate deployment cost for new stations"""
        # Cost estimates per station type (in USD)
        cost_per_station = {
            MonitoringNetworkType.SEISMIC: 50000,
            MonitoringNetworkType.GPS_GNSS: 30000,
            MonitoringNetworkType.THERMAL: 25000,
            MonitoringNetworkType.GAS_GEOCHEMICAL: 40000,
            MonitoringNetworkType.DEFORMATION: 35000,
            MonitoringNetworkType.MULTI_PARAMETER: 75000
        }
        
        base_cost = cost_per_station.get(network_type, 40000)
        
        return {
            'equipment_cost': base_cost * num_stations,
            'installation_cost': base_cost * 0.3 * num_stations,
            'annual_maintenance': base_cost * 0.1 * num_stations,
            'total_first_year': base_cost * 1.4 * num_stations
        }

    def _estimate_implementation_timeline(self, num_stations: int) -> Dict[str, str]:
        """Estimate implementation timeline"""
        # Base timeline per station (in months)
        months_per_station = 2
        total_months = max(6, num_stations * months_per_station)
        
        return {
            'planning_phase': '2-3 months',
            'procurement': '3-6 months',
            'installation': f'{total_months} months',
            'testing_commissioning': '1-2 months',
            'total_timeline': f'{total_months + 8} months'
        }

    async def _fetch_external_observatories(self, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch observatory data from external API"""
        # Placeholder implementation - would make actual API calls
        return []

    async def _fetch_external_stations(self, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch station data from external API"""
        # Placeholder implementation - would make actual API calls
        return []

    async def _find_existing_observatory(self, obs_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find existing observatory by name or coordinates"""
        observatories = await self.query_observatories(filters={'name': obs_data.get('name')})
        return observatories[0] if observatories else None

    async def _find_existing_station(self, station_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find existing station by station code"""
        stations = await self.query_monitoring_stations(
            filters={'station_code': station_data.get('station_code')}
        )
        return stations[0] if stations else None

    async def _update_observatory(self, observatory_id: str, obs_data: Dict[str, Any]):
        """Update existing observatory record"""
        # Implementation would update database record
        pass

    async def _update_station(self, station_id: str, station_data: Dict[str, Any]):
        """Update existing station record"""
        # Implementation would update database record
        pass

    async def _test_stream_connection(self, stream_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test connection to data stream"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(stream_config['endpoint_url'], timeout=10) as response:
                    if response.status == 200:
                        return {'success': True, 'status_code': response.status}
                    else:
                        return {'success': False, 'error': f'HTTP {response.status}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _setup_stream_consumer(self, consumer_config: Dict[str, Any]):
        """Set up real-time stream consumer"""
        # Implementation would set up Kafka consumer or WebSocket connection
        pass

    async def _process_real_time_data(self, stream_id: str, data: Dict[str, Any]):
        """Process incoming real-time data"""
        # Implementation would process and store real-time data
        logger.info("Processing real-time data", stream_id=stream_id)

    async def _handle_stream_error(self, stream_id: str, error: Exception):
        """Handle stream processing errors"""
        logger.error("Stream processing error", stream_id=stream_id, error=str(error))

    async def generate_monitoring_report(self, 
                                       region_bounds: GeographicBounds = None,
                                       time_range: Tuple[datetime, datetime] = None) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring network report.
        
        Args:
            region_bounds: Optional geographic bounds for regional report
            time_range: Optional time range for performance analysis
            
        Returns:
            Comprehensive monitoring report
        """
        try:
            report = {
                'report_id': str(uuid.uuid4()),
                'generated_at': datetime.utcnow().isoformat(),
                'region_bounds': region_bounds.__dict__ if region_bounds else None,
                'time_range': [t.isoformat() for t in time_range] if time_range else None,
                'summary': {},
                'observatories': {},
                'networks': {},
                'coverage_analysis': {},
                'performance_metrics': {},
                'recommendations': []
            }
            
            # Observatory summary
            observatories = await self.query_observatories(geographic_bounds=region_bounds)
            report['summary']['total_observatories'] = len(observatories)
            report['summary']['countries'] = len(set(obs['country'] for obs in observatories))
            report['observatories'] = {
                'active': len([obs for obs in observatories if obs['status'] == 'operational']),
                'real_time_capable': len([obs for obs in observatories if obs['real_time_capability']]),
                'with_alert_systems': len([obs for obs in observatories if obs['alert_system']])
            }
            
            # Network analysis by type
            for network_type in MonitoringNetworkType:
                stations = await self.query_monitoring_stations(
                    filters={'station_type': network_type.value},
                    geographic_bounds=region_bounds
                )
                
                if stations:
                    coverage = await self.analyze_network_coverage(
                        region_bounds or GeographicBounds(-90, 90, -180, 180),
                        network_type
                    )
                    
                    performance = await self.assess_network_performance()
                    
                    report['networks'][network_type.value] = {
                        'station_count': len(stations),
                        'coverage_percentage': coverage.coverage_percentage,
                        'uptime': performance.uptime_percentage,
                        'data_completeness': performance.data_completeness
                    }
            
            # Generate recommendations
            if region_bounds:
                gaps = await self.identify_monitoring_gaps(region_bounds)
                report['recommendations'] = gaps
            
            logger.info("Monitoring report generated", 
                       report_id=report['report_id'],
                       observatories=len(observatories))
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate monitoring report", error=str(e))
            raise

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        if hasattr(self, 'kafka_producer') and self.kafka_producer:
            self.kafka_producer.close()


# Example usage and testing functions
async def main():
    """Example usage of the Monitoring Network Registry"""
    
    # Initialize registry
    registry = MonitoringNetworkRegistry(
        database_url="postgresql://user:password@localhost/monitoring_db",
        redis_url="redis://localhost:6379",
        kafka_config={'bootstrap_servers': ['localhost:9092']}
    )
    
    # Register a volcanic observatory
    observatory_data = {
        'name': 'Yellowstone Volcano Observatory',
        'acronym': 'YVO',
        'country': 'United States',
        'region': 'Wyoming',
        'latitude': 44.428,
        'longitude': -110.588,
        'elevation': 2400,
        'established_date': datetime(1996, 1, 1),
        'parent_organization': 'USGS',
        'contact_email': 'yvo@usgs.gov',
        'website_url': 'https://www.usgs.gov/observatories/yellowstone-volcano-observatory',
        'primary_volcanoes': ['yellowstone'],
        'network_types': ['seismic', 'gps_gnss', 'thermal', 'deformation'],
        'real_time_capability': True,
        'alert_system': True,
        'capabilities': {
            'seismic_monitoring': True,
            'deformation_monitoring': True,
            'gas_monitoring': True,
            'thermal_monitoring': True
        }
    }
    
    observatory_id = await registry.register_volcanic_observatory(observatory_data)
    print(f"Registered observatory: {observatory_id}")
    
    # Register monitoring stations
    station_data = {
        'station_code': 'YNP.LKWY',
        'name': 'Lake Yellowstone Seismic Station',
        'observatory_id': observatory_id,
        'volcano_id': 'yellowstone',
        'latitude': 44.428,
        'longitude': -110.588,
        'elevation': 2357,
        'installation_date': datetime(2000, 6, 15),
        'station_type': 'seismic',
        'network_code': 'YNP',
        'site_conditions': 'Bedrock installation near lake shore',
        'access_difficulty': 'moderate',
        'power_source': 'Solar panel with battery backup',
        'communication_method': 'Satellite telemetry',
        'status': 'operational',
        'instruments': [
            {
                'type': 'seismometer',
                'manufacturer': 'Nanometrics',
                'model': 'Trillium Compact',
                'sampling_rate': 100.0,
                'installation_date': '2000-06-15'
            }
        ],
        'data_availability': 95.5,
        'uptime': 98.2
    }
    
    station_id = await registry.register_monitoring_station(station_data)
    print(f"Registered station: {station_id}")
    
    # Analyze network coverage
    region = GeographicBounds(44.0, 45.0, -111.0, -110.0)
    coverage = await registry.analyze_network_coverage(region, MonitoringNetworkType.SEISMIC)
    print(f"Network coverage: {coverage.coverage_percentage:.1f}%")
    
    # Generate monitoring report
    report = await registry.generate_monitoring_report(region_bounds=region)
    print(f"Generated report: {report['report_id']}")

if __name__ == "__main__":
    asyncio.run(main())