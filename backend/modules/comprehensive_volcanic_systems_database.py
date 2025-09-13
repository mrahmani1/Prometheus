"""
Comprehensive Volcanic Systems Database
Integrates Global Volcanism Program, NASA FIRMS, tectonic settings, and monitoring networks
Production-grade implementation with advanced features and real data integration
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import urlencode, quote
import warnings
import hashlib
import time
from enum import Enum
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import asynccontextmanager

# Geospatial and scientific libraries
try:
    import geopandas as gpd
    import shapely.geometry as geom
    from pyproj import Transformer, CRS
    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.ops import transform
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    warnings.warn("Geospatial libraries not available. Some features will be limited.")

# Advanced analytics libraries
try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from scipy.spatial.distance import cdist
    from scipy import stats, interpolate
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    warnings.warn("Analytics libraries not available. Advanced features will be limited.")

class TectonicSetting(Enum):
    """Comprehensive tectonic setting classification"""
    SUBDUCTION_ZONE = "subduction_zone"
    MID_OCEAN_RIDGE = "mid_ocean_ridge"
    HOTSPOT = "hotspot"
    CONTINENTAL_RIFT = "continental_rift"
    BACK_ARC = "back_arc"
    INTRAPLATE = "intraplate"
    TRANSFORM_FAULT = "transform_fault"
    CONTINENTAL_ARC = "continental_arc"
    ISLAND_ARC = "island_arc"
    FLOOD_BASALT = "flood_basalt"

class VolcanoType(Enum):
    """Detailed volcano type classification"""
    STRATOVOLCANO = "stratovolcano"
    SHIELD = "shield"
    CINDER_CONE = "cinder_cone"
    CALDERA = "caldera"
    COMPLEX = "complex"
    LAVA_DOME = "lava_dome"
    MAAR = "maar"
    FISSURE_VENT = "fissure_vent"
    SUBMARINE = "submarine"
    PYROCLASTIC_CONE = "pyroclastic_cone"

class AlertLevel(Enum):
    """Standardized volcanic alert levels"""
    NORMAL = "normal"
    ADVISORY = "advisory"
    WATCH = "watch"
    WARNING = "warning"
    UNASSIGNED = "unassigned"

class AviationColor(Enum):
    """Aviation color codes"""
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"
    UNASSIGNED = "unassigned"

@dataclass
class VolcanicSystem:
    """Comprehensive volcanic system data structure"""
    volcano_id: str
    name: str
    location: Tuple[float, float, float]  # lat, lon, elevation
    country: str
    region: str
    volcano_type: VolcanoType
    tectonic_setting: TectonicSetting
    elevation_m: float
    
    # Optional fields with defaults
    gvp_id: Optional[str] = None
    subregion: Optional[str] = None
    prominence_m: Optional[float] = None
    edifice_volume_km3: Optional[float] = None
    last_known_eruption: Optional[datetime] = None
    eruption_certainty: Optional[str] = None
    population_within_5km: Optional[int] = None
    population_within_10km: Optional[int] = None
    population_within_30km: Optional[int] = None
    population_within_100km: Optional[int] = None
    monitoring_level: Optional[str] = None
    morphology: Optional[str] = None
    
    # Lists with default factories
    activity_evidence: List[str] = field(default_factory=list)
    monitoring_networks: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    rock_types: List[str] = field(default_factory=list)
    thermal_features: List[str] = field(default_factory=list)
    
    # Metadata with defaults
    last_updated: datetime = field(default_factory=datetime.utcnow)
    quality_score: float = 0.0
    
@dataclass
class EruptionEvent:
    """Historical eruption event data"""
    eruption_id: str
    volcano_id: str
    description: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    vei: Optional[int] = None  # Volcanic Explosivity Index
    eruption_type: Optional[str] = None
    fatalities: Optional[int] = None
    evacuations: Optional[int] = None
    evidence: List[str] = field(default_factory=list)
    
@dataclass
class ThermalAnomaly:
    """NASA FIRMS thermal anomaly data"""
    anomaly_id: str
    volcano_id: str
    detection_time: datetime
    latitude: float
    longitude: float
    brightness_temp_k: float
    confidence: int
    satellite: str  # MODIS, VIIRS
    instrument: str
    fire_radiative_power: Optional[float] = None
    
@dataclass
class MonitoringStation:
    """Volcanic monitoring station information"""
    station_id: str
    volcano_id: str
    station_name: str
    location: Tuple[float, float, float]
    station_type: str  # seismic, GPS, gas, thermal, etc.
    operator: str
    status: str  # active, inactive, maintenance
    real_time_capability: bool
    instruments: List[str] = field(default_factory=list)
    data_availability: Dict[str, Any] = field(default_factory=dict)
    installation_date: Optional[datetime] = None
    
@dataclass
class TectonicContext:
    """Detailed tectonic setting context"""
    volcano_id: str
    tectonic_setting: TectonicSetting
    plate_boundary_distance_km: Optional[float]
    convergence_rate_mm_yr: Optional[float]
    convergence_obliquity_deg: Optional[float]
    slab_depth_km: Optional[float]
    crustal_thickness_km: Optional[float]
    regional_stress_field: Optional[Dict[str, float]]
    
class ComprehensiveVolcanicSystemsDatabase:
    """
    Production-grade volcanic systems database integrating multiple authoritative sources
    Combines GVP, NASA FIRMS, tectonic analysis, and monitoring network data
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._default_config()
        
        # Database connections
        self.postgres_pool = None
        self.redis_client = None
        self.session = None
        
        # Data source configurations
        self.data_sources = {
            'smithsonian_gvp': {
                'base_url': 'https://volcano.si.edu/api/v1/',
                'endpoints': {
                    'volcanoes': 'volcanoes',
                    'eruptions': 'eruptions',
                    'reports': 'reports',
                    'deformation': 'deformation'
                },
                'rate_limit': 100,
                'api_key': self.config.get('gvp_api_key')
            },
            'nasa_firms': {
                'base_url': 'https://firms.modaps.eosdis.nasa.gov/api/',
                'endpoints': {
                    'area': 'area/csv',
                    'country': 'country/csv'
                },
                'rate_limit': 1000,
                'api_key': self.config.get('nasa_firms_api_key')
            },
            'usgs_volcano_hazards': {
                'base_url': 'https://volcanoes.usgs.gov/vsc-sign/api/',
                'endpoints': {
                    'alerts': 'alerts',
                    'volcano_info': 'volcano',
                    'monitoring': 'monitoring'
                },
                'rate_limit': 60
            },
            'plate_motion_model': {
                'base_url': 'https://www.unavco.org/software/geodetic-utilities/',
                'endpoints': {
                    'plate_motion': 'plate-motion-calculator'
                },
                'rate_limit': 50
            }
        }
        
        # Rate limiters
        self.rate_limiters = {
            source: asyncio.Semaphore(config['rate_limit']) 
            for source, config in self.data_sources.items()
        }
        
        # Caching configuration
        self.cache_ttl = {
            'volcano_catalog': 86400,  # 24 hours
            'eruption_history': 43200,  # 12 hours
            'thermal_anomalies': 3600,  # 1 hour
            'monitoring_networks': 21600,  # 6 hours
            'tectonic_context': 604800  # 1 week
        }
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Synchronization locks
        self.sync_locks = {
            'gvp_sync': asyncio.Lock(),
            'firms_sync': asyncio.Lock(),
            'monitoring_sync': asyncio.Lock()
        }
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the database system"""
        return {
            'postgres_dsn': 'postgresql://prometheus:prometheus123@localhost:5432/prometheus',
            'redis_url': 'redis://localhost:6379/0',
            'gvp_api_key': None,  # Optional API key for enhanced access
            'nasa_firms_api_key': None,  # Required for NASA FIRMS access
            'cache_enabled': True,
            'auto_sync_interval': 3600,  # 1 hour
            'quality_threshold': 0.7,
            'max_concurrent_requests': 10
        }
    
    async def initialize(self):
        """Initialize the comprehensive volcanic systems database"""
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
                headers={
                    'User-Agent': 'Prometheus-VolcanicSystems/2.0',
                    'Accept': 'application/json, application/xml, text/csv'
                }
            )
            
            # Initialize database connections
            await self._initialize_databases()
            
            # Create database schema if needed
            await self._create_database_schema()
            
            # Test data source connections
            await self._test_data_source_connections()
            
            # Start background synchronization if configured
            if self.config.get('auto_sync_enabled', True):
                asyncio.create_task(self._background_synchronization())
            
            self.logger.info("Comprehensive volcanic systems database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize volcanic systems database: {e}")
            raise
    
    async def _initialize_databases(self):
        """Initialize PostgreSQL and Redis connections"""
        try:
            # Initialize PostgreSQL connection pool
            import asyncpg
            self.postgres_pool = await asyncpg.create_pool(
                self.config['postgres_dsn'],
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Initialize Redis client
            if self.config.get('cache_enabled', True):
                import aioredis
                self.redis_client = await aioredis.from_url(
                    self.config['redis_url'],
                    encoding='utf-8',
                    decode_responses=True
                )
            
            self.logger.info("Database connections initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def _create_database_schema(self):
        """Create comprehensive database schema for volcanic systems"""
        schema_sql = """
        -- Create volcanic systems schema
        CREATE SCHEMA IF NOT EXISTS volcanic_systems;
        
        -- Global volcano catalog table
        CREATE TABLE IF NOT EXISTS volcanic_systems.global_volcano_catalog (
            id SERIAL PRIMARY KEY,
            volcano_id VARCHAR(50) UNIQUE NOT NULL,
            gvp_id VARCHAR(20),
            name VARCHAR(255) NOT NULL,
            location GEOGRAPHY(POINTZ, 4326) NOT NULL,
            country VARCHAR(100),
            region VARCHAR(100),
            subregion VARCHAR(100),
            volcano_type VARCHAR(50),
            tectonic_setting VARCHAR(50),
            elevation_m REAL,
            prominence_m REAL,
            edifice_volume_km3 REAL,
            last_known_eruption TIMESTAMPTZ,
            eruption_certainty VARCHAR(50),
            activity_evidence JSONB,
            population_within_5km INTEGER,
            population_within_10km INTEGER,
            population_within_30km INTEGER,
            population_within_100km INTEGER,
            monitoring_level VARCHAR(50),
            monitoring_networks JSONB,
            data_sources JSONB,
            quality_score REAL DEFAULT 0.0,
            rock_types JSONB,
            morphology VARCHAR(100),
            thermal_features JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Eruption history table
        CREATE TABLE IF NOT EXISTS volcanic_systems.eruption_history (
            id SERIAL PRIMARY KEY,
            eruption_id VARCHAR(50) UNIQUE NOT NULL,
            volcano_id VARCHAR(50) REFERENCES volcanic_systems.global_volcano_catalog(volcano_id),
            start_date TIMESTAMPTZ,
            end_date TIMESTAMPTZ,
            vei INTEGER CHECK (vei >= 0 AND vei <= 8),
            eruption_type VARCHAR(100),
            description TEXT,
            fatalities INTEGER,
            evacuations INTEGER,
            evidence JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- NASA FIRMS thermal anomalies table
        CREATE TABLE IF NOT EXISTS volcanic_systems.thermal_anomalies (
            id SERIAL PRIMARY KEY,
            anomaly_id VARCHAR(100) UNIQUE NOT NULL,
            volcano_id VARCHAR(50),
            detection_time TIMESTAMPTZ NOT NULL,
            location GEOGRAPHY(POINT, 4326) NOT NULL,
            brightness_temp_k REAL,
            fire_radiative_power REAL,
            confidence INTEGER,
            satellite VARCHAR(20),
            instrument VARCHAR(20),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Monitoring stations table
        CREATE TABLE IF NOT EXISTS volcanic_systems.monitoring_stations (
            id SERIAL PRIMARY KEY,
            station_id VARCHAR(100) UNIQUE NOT NULL,
            volcano_id VARCHAR(50),
            station_name VARCHAR(255),
            location GEOGRAPHY(POINTZ, 4326) NOT NULL,
            station_type VARCHAR(50),
            instruments JSONB,
            operator VARCHAR(100),
            installation_date DATE,
            status VARCHAR(20),
            data_availability JSONB,
            real_time_capability BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Tectonic context table
        CREATE TABLE IF NOT EXISTS volcanic_systems.tectonic_context (
            id SERIAL PRIMARY KEY,
            volcano_id VARCHAR(50) UNIQUE REFERENCES volcanic_systems.global_volcano_catalog(volcano_id),
            tectonic_setting VARCHAR(50),
            plate_boundary_distance_km REAL,
            convergence_rate_mm_yr REAL,
            convergence_obliquity_deg REAL,
            slab_depth_km REAL,
            crustal_thickness_km REAL,
            regional_stress_field JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Create spatial indexes
        CREATE INDEX IF NOT EXISTS idx_volcano_location ON volcanic_systems.global_volcano_catalog USING GIST (location);
        CREATE INDEX IF NOT EXISTS idx_thermal_location ON volcanic_systems.thermal_anomalies USING GIST (location);
        CREATE INDEX IF NOT EXISTS idx_station_location ON volcanic_systems.monitoring_stations USING GIST (location);
        
        -- Create other indexes
        CREATE INDEX IF NOT EXISTS idx_volcano_tectonic ON volcanic_systems.global_volcano_catalog (tectonic_setting);
        CREATE INDEX IF NOT EXISTS idx_volcano_type ON volcanic_systems.global_volcano_catalog (volcano_type);
        CREATE INDEX IF NOT EXISTS idx_eruption_volcano ON volcanic_systems.eruption_history (volcano_id);
        CREATE INDEX IF NOT EXISTS idx_thermal_volcano ON volcanic_systems.thermal_anomalies (volcano_id);
        CREATE INDEX IF NOT EXISTS idx_thermal_time ON volcanic_systems.thermal_anomalies (detection_time);
        """
        
        async with self.postgres_pool.acquire() as conn:
            await conn.execute(schema_sql)
        
        self.logger.info("Database schema created successfully")
    
    async def integrate_gvp_database(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Integrate complete Global Volcanism Program database
        Implements subtask 4.1 with comprehensive volcano catalog and metadata
        """
        async with self.sync_locks['gvp_sync']:
            try:
                self.logger.info("Starting GVP database integration")
                
                # Check cache first
                if not force_refresh and self.redis_client:
                    cached_result = await self.redis_client.get('gvp_integration_status')
                    if cached_result:
                        return json.loads(cached_result)
                
                integration_stats = {
                    'volcanoes_processed': 0,
                    'eruptions_processed': 0,
                    'new_volcanoes': 0,
                    'updated_volcanoes': 0,
                    'errors': [],
                    'start_time': datetime.utcnow().isoformat(),
                    'status': 'in_progress'
                }
                
                # Fetch volcano catalog from GVP
                volcanoes = await self._fetch_gvp_volcano_catalog()
                
                for volcano_data in volcanoes:
                    try:
                        # Process volcano data
                        volcano = await self._process_gvp_volcano_data(volcano_data)
                        
                        # Store in database
                        stored = await self._store_volcano_data(volcano)
                        
                        if stored['is_new']:
                            integration_stats['new_volcanoes'] += 1
                        else:
                            integration_stats['updated_volcanoes'] += 1
                        
                        integration_stats['volcanoes_processed'] += 1
                        
                        # Fetch and store eruption history
                        eruptions = await self._fetch_gvp_eruption_history(volcano.volcano_id)
                        for eruption in eruptions:
                            await self._store_eruption_data(eruption)
                            integration_stats['eruptions_processed'] += 1
                        
                        # Rate limiting
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        error_msg = f"Failed to process volcano {volcano_data.get('name', 'unknown')}: {e}"
                        self.logger.warning(error_msg)
                        integration_stats['errors'].append(error_msg)
                
                integration_stats['status'] = 'completed'
                integration_stats['end_time'] = datetime.utcnow().isoformat()
                
                # Cache results
                if self.redis_client:
                    await self.redis_client.setex(
                        'gvp_integration_status',
                        self.cache_ttl['volcano_catalog'],
                        json.dumps(integration_stats)
                    )
                
                self.logger.info(f"GVP integration completed: {integration_stats['volcanoes_processed']} volcanoes, {integration_stats['eruptions_processed']} eruptions")
                
                return integration_stats
                
            except Exception as e:
                self.logger.error(f"GVP database integration failed: {e}")
                raise
    
    async def integrate_nasa_firms_thermal_data(self, time_window: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """
        Integrate NASA FIRMS thermal anomaly data for real-time volcanic monitoring
        Implements subtask 4.2 with MODIS and VIIRS thermal data
        """
        async with self.sync_locks['firms_sync']:
            try:
                self.logger.info("Starting NASA FIRMS thermal data integration")
                
                if not self.config.get('nasa_firms_api_key'):
                    raise ValueError("NASA FIRMS API key required for thermal data access")
                
                integration_stats = {
                    'anomalies_processed': 0,
                    'new_anomalies': 0,
                    'volcanoes_with_thermal_activity': 0,
                    'satellites_processed': [],
                    'time_range': {
                        'start': (datetime.utcnow() - time_window).isoformat(),
                        'end': datetime.utcnow().isoformat()
                    },
                    'errors': [],
                    'status': 'in_progress'
                }
                
                # Get all volcanoes for thermal monitoring
                volcanoes = await self._get_all_volcanoes()
                
                # Process thermal data for each volcano
                for volcano in volcanoes:
                    try:
                        # Fetch thermal anomalies around volcano
                        thermal_data = await self._fetch_nasa_firms_data(
                            volcano.location[:2],  # lat, lon
                            time_window
                        )
                        
                        volcano_has_activity = False
                        
                        for anomaly_data in thermal_data:
                            # Process and validate thermal anomaly
                            anomaly = await self._process_thermal_anomaly(
                                anomaly_data, volcano.volcano_id
                            )
                            
                            if anomaly:
                                # Store thermal anomaly
                                stored = await self._store_thermal_anomaly(anomaly)
                                
                                if stored['is_new']:
                                    integration_stats['new_anomalies'] += 1
                                    volcano_has_activity = True
                                
                                integration_stats['anomalies_processed'] += 1
                        
                        if volcano_has_activity:
                            integration_stats['volcanoes_with_thermal_activity'] += 1
                        
                        # Rate limiting
                        await asyncio.sleep(0.05)
                        
                    except Exception as e:
                        error_msg = f"Failed to process thermal data for {volcano.name}: {e}"
                        self.logger.warning(error_msg)
                        integration_stats['errors'].append(error_msg)
                
                integration_stats['status'] = 'completed'
                integration_stats['satellites_processed'] = ['MODIS', 'VIIRS']
                
                self.logger.info(f"NASA FIRMS integration completed: {integration_stats['anomalies_processed']} anomalies processed")
                
                return integration_stats
                
            except Exception as e:
                self.logger.error(f"NASA FIRMS thermal data integration failed: {e}")
                raise
    
    async def classify_tectonic_settings(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Comprehensive tectonic setting classification for all volcanic systems
        Implements subtask 4.3 with detailed tectonic context analysis
        """
        try:
            self.logger.info("Starting tectonic setting classification")
            
            classification_stats = {
                'volcanoes_classified': 0,
                'tectonic_settings': {},
                'plate_boundary_analysis': 0,
                'stress_field_analysis': 0,
                'errors': [],
                'status': 'in_progress'
            }
            
            # Get all volcanoes
            volcanoes = await self._get_all_volcanoes()
            
            # Load tectonic reference data
            plate_boundaries = await self._load_plate_boundary_data()
            hotspot_locations = await self._load_hotspot_data()
            
            for volcano in volcanoes:
                try:
                    # Analyze tectonic context
                    tectonic_context = await self._analyze_tectonic_context(
                        volcano, plate_boundaries, hotspot_locations
                    )
                    
                    # Store tectonic context
                    await self._store_tectonic_context(tectonic_context)
                    
                    # Update volcano with tectonic setting
                    await self._update_volcano_tectonic_setting(
                        volcano.volcano_id, tectonic_context.tectonic_setting
                    )
                    
                    classification_stats['volcanoes_classified'] += 1
                    
                    # Update statistics
                    setting_name = tectonic_context.tectonic_setting.value
                    classification_stats['tectonic_settings'][setting_name] = \
                        classification_stats['tectonic_settings'].get(setting_name, 0) + 1
                    
                    if tectonic_context.plate_boundary_distance_km is not None:
                        classification_stats['plate_boundary_analysis'] += 1
                    
                    if tectonic_context.regional_stress_field:
                        classification_stats['stress_field_analysis'] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to classify tectonic setting for {volcano.name}: {e}"
                    self.logger.warning(error_msg)
                    classification_stats['errors'].append(error_msg)
            
            classification_stats['status'] = 'completed'
            
            self.logger.info(f"Tectonic classification completed: {classification_stats['volcanoes_classified']} volcanoes classified")
            
            return classification_stats
            
        except Exception as e:
            self.logger.error(f"Tectonic setting classification failed: {e}")
            raise
    
    async def create_monitoring_network_registry(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Create comprehensive global volcanic monitoring network registry
        Implements subtask 4.4 with detailed monitoring station inventory
        """
        async with self.sync_locks['monitoring_sync']:
            try:
                self.logger.info("Creating monitoring network registry")
                
                registry_stats = {
                    'stations_processed': 0,
                    'networks_identified': 0,
                    'real_time_stations': 0,
                    'station_types': {},
                    'coverage_analysis': {},
                    'errors': [],
                    'status': 'in_progress'
                }
                
                # Fetch monitoring data from multiple sources
                monitoring_sources = [
                    self._fetch_usgs_monitoring_stations(),
                    self._fetch_iris_seismic_stations(),
                    self._fetch_unavco_gps_stations(),
                    self._fetch_international_monitoring_networks()
                ]
                
                all_stations = []
                for source_stations in await asyncio.gather(*monitoring_sources, return_exceptions=True):
                    if isinstance(source_stations, Exception):
                        registry_stats['errors'].append(str(source_stations))
                    else:
                        all_stations.extend(source_stations)
                
                # Process and store monitoring stations
                for station_data in all_stations:
                    try:
                        # Process station data
                        station = await self._process_monitoring_station_data(station_data)
                        
                        # Associate with nearest volcano
                        volcano_id = await self._find_nearest_volcano(station.location[:2])
                        if volcano_id:
                            station.volcano_id = volcano_id
                        
                        # Store station data
                        await self._store_monitoring_station(station)
                        
                        registry_stats['stations_processed'] += 1
                        
                        # Update statistics
                        station_type = station.station_type
                        registry_stats['station_types'][station_type] = \
                            registry_stats['station_types'].get(station_type, 0) + 1
                        
                        if station.real_time_capability:
                            registry_stats['real_time_stations'] += 1
                        
                    except Exception as e:
                        error_msg = f"Failed to process monitoring station: {e}"
                        self.logger.warning(error_msg)
                        registry_stats['errors'].append(error_msg)
                
                # Analyze monitoring coverage
                coverage_analysis = await self._analyze_monitoring_coverage()
                registry_stats['coverage_analysis'] = coverage_analysis
                
                registry_stats['status'] = 'completed'
                registry_stats['networks_identified'] = len(set(
                    station.operator for station in all_stations if hasattr(station, 'operator')
                ))
                
                self.logger.info(f"Monitoring network registry completed: {registry_stats['stations_processed']} stations processed")
                
                return registry_stats
                
            except Exception as e:
                self.logger.error(f"Monitoring network registry creation failed: {e}")
                raise 
   
    async def synchronize_with_authoritative_sources(self) -> Dict[str, Any]:
        """
        Automated synchronization with all authoritative volcanic data sources
        Implements comprehensive data validation and quality control
        """
        try:
            self.logger.info("Starting automated synchronization with authoritative sources")
            
            sync_results = {
                'gvp_sync': {},
                'firms_sync': {},
                'tectonic_sync': {},
                'monitoring_sync': {},
                'validation_results': {},
                'overall_status': 'in_progress',
                'start_time': datetime.utcnow().isoformat()
            }
            
            # Parallel synchronization of all data sources
            sync_tasks = [
                self.integrate_gvp_database(force_refresh=True),
                self.integrate_nasa_firms_thermal_data(),
                self.classify_tectonic_settings(force_refresh=True),
                self.create_monitoring_network_registry(force_refresh=True)
            ]
            
            results = await asyncio.gather(*sync_tasks, return_exceptions=True)
            
            # Process results
            sync_results['gvp_sync'] = results[0] if not isinstance(results[0], Exception) else {'error': str(results[0])}
            sync_results['firms_sync'] = results[1] if not isinstance(results[1], Exception) else {'error': str(results[1])}
            sync_results['tectonic_sync'] = results[2] if not isinstance(results[2], Exception) else {'error': str(results[2])}
            sync_results['monitoring_sync'] = results[3] if not isinstance(results[3], Exception) else {'error': str(results[3])}
            
            # Comprehensive data validation
            validation_results = await self._comprehensive_data_validation()
            sync_results['validation_results'] = validation_results
            
            sync_results['overall_status'] = 'completed'
            sync_results['end_time'] = datetime.utcnow().isoformat()
            
            # Update synchronization metadata
            await self._update_sync_metadata(sync_results)
            
            self.logger.info("Automated synchronization completed successfully")
            
            return sync_results
            
        except Exception as e:
            self.logger.error(f"Automated synchronization failed: {e}")
            raise
    
    async def query_volcanic_systems(self, 
                                   region: Optional[str] = None,
                                   tectonic_setting: Optional[TectonicSetting] = None,
                                   activity_since: Optional[datetime] = None,
                                   monitoring_level: Optional[str] = None,
                                   bbox: Optional[Tuple[float, float, float, float]] = None) -> List[VolcanicSystem]:
        """
        Advanced querying of volcanic systems with multiple filter criteria
        """
        try:
            query_conditions = []
            query_params = []
            
            base_query = """
            SELECT v.*, tc.tectonic_setting, tc.plate_boundary_distance_km,
                   tc.convergence_rate_mm_yr, tc.slab_depth_km
            FROM volcanic_systems.global_volcano_catalog v
            LEFT JOIN volcanic_systems.tectonic_context tc ON v.volcano_id = tc.volcano_id
            WHERE 1=1
            """
            
            if region:
                query_conditions.append("v.region = $" + str(len(query_params) + 1))
                query_params.append(region)
            
            if tectonic_setting:
                query_conditions.append("tc.tectonic_setting = $" + str(len(query_params) + 1))
                query_params.append(tectonic_setting.value)
            
            if activity_since:
                query_conditions.append("v.last_known_eruption >= $" + str(len(query_params) + 1))
                query_params.append(activity_since)
            
            if monitoring_level:
                query_conditions.append("v.monitoring_level = $" + str(len(query_params) + 1))
                query_params.append(monitoring_level)
            
            if bbox:
                # Spatial bounding box query
                query_conditions.append(
                    "ST_Intersects(v.location, ST_MakeEnvelope($" + str(len(query_params) + 1) + 
                    ", $" + str(len(query_params) + 2) + ", $" + str(len(query_params) + 3) + 
                    ", $" + str(len(query_params) + 4) + ", 4326))"
                )
                query_params.extend(bbox)
            
            if query_conditions:
                base_query += " AND " + " AND ".join(query_conditions)
            
            base_query += " ORDER BY v.name"
            
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(base_query, *query_params)
                
                volcanoes = []
                for row in rows:
                    volcano = await self._row_to_volcanic_system(row)
                    volcanoes.append(volcano)
                
                return volcanoes
                
        except Exception as e:
            self.logger.error(f"Failed to query volcanic systems: {e}")
            raise
    
    async def get_thermal_activity_analysis(self, 
                                          volcano_id: str,
                                          time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """
        Comprehensive thermal activity analysis for a specific volcano
        """
        try:
            query = """
            SELECT * FROM volcanic_systems.thermal_anomalies
            WHERE volcano_id = $1 AND detection_time BETWEEN $2 AND $3
            ORDER BY detection_time
            """
            
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query, volcano_id, time_range[0], time_range[1])
            
            if not rows:
                return {
                    'volcano_id': volcano_id,
                    'time_range': [t.isoformat() for t in time_range],
                    'thermal_anomalies': 0,
                    'analysis': 'no_thermal_activity'
                }
            
            # Convert to thermal anomaly objects
            anomalies = [await self._row_to_thermal_anomaly(row) for row in rows]
            
            # Perform thermal analysis
            analysis = await self._analyze_thermal_patterns(anomalies, time_range)
            
            return {
                'volcano_id': volcano_id,
                'time_range': [t.isoformat() for t in time_range],
                'thermal_anomalies': len(anomalies),
                'analysis': analysis,
                'anomaly_data': [
                    {
                        'detection_time': a.detection_time.isoformat(),
                        'brightness_temp_k': a.brightness_temp_k,
                        'confidence': a.confidence,
                        'satellite': a.satellite
                    } for a in anomalies
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze thermal activity for {volcano_id}: {e}")
            raise
    
    async def get_monitoring_network_status(self, volcano_id: str) -> Dict[str, Any]:
        """
        Get comprehensive monitoring network status for a volcano
        """
        try:
            query = """
            SELECT * FROM volcanic_systems.monitoring_stations
            WHERE volcano_id = $1
            ORDER BY station_type, station_name
            """
            
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query, volcano_id)
            
            stations = [await self._row_to_monitoring_station(row) for row in rows]
            
            # Analyze monitoring capabilities
            monitoring_analysis = {
                'total_stations': len(stations),
                'station_types': {},
                'real_time_stations': 0,
                'operators': set(),
                'coverage_assessment': {},
                'data_availability': {}
            }
            
            for station in stations:
                # Count by type
                station_type = station.station_type
                monitoring_analysis['station_types'][station_type] = \
                    monitoring_analysis['station_types'].get(station_type, 0) + 1
                
                # Real-time capability
                if station.real_time_capability:
                    monitoring_analysis['real_time_stations'] += 1
                
                # Operators
                monitoring_analysis['operators'].add(station.operator)
                
                # Data availability
                if station.data_availability:
                    for data_type, availability in station.data_availability.items():
                        if data_type not in monitoring_analysis['data_availability']:
                            monitoring_analysis['data_availability'][data_type] = []
                        monitoring_analysis['data_availability'][data_type].append(availability)
            
            # Convert sets to lists for JSON serialization
            monitoring_analysis['operators'] = list(monitoring_analysis['operators'])
            
            # Calculate coverage assessment
            monitoring_analysis['coverage_assessment'] = await self._assess_monitoring_coverage(volcano_id, stations)
            
            return {
                'volcano_id': volcano_id,
                'monitoring_analysis': monitoring_analysis,
                'stations': [
                    {
                        'station_id': s.station_id,
                        'name': s.station_name,
                        'type': s.station_type,
                        'operator': s.operator,
                        'real_time': s.real_time_capability,
                        'status': s.status
                    } for s in stations
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get monitoring network status for {volcano_id}: {e}")
            raise
    
    async def _fetch_gvp_volcano_catalog(self) -> List[Dict[str, Any]]:
        """Fetch complete volcano catalog from Smithsonian GVP"""
        try:
            async with self.rate_limiters['smithsonian_gvp']:
                url = f"{self.data_sources['smithsonian_gvp']['base_url']}{self.data_sources['smithsonian_gvp']['endpoints']['volcanoes']}"
                
                params = {
                    'format': 'json',
                    'limit': 10000  # Get all volcanoes
                }
                
                if self.data_sources['smithsonian_gvp']['api_key']:
                    params['api_key'] = self.data_sources['smithsonian_gvp']['api_key']
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('volcanoes', [])
                    else:
                        self.logger.warning(f"GVP catalog request failed: {response.status}")
                        return []
        
        except Exception as e:
            self.logger.error(f"Failed to fetch GVP volcano catalog: {e}")
            return []
    
    async def _process_gvp_volcano_data(self, volcano_data: Dict[str, Any]) -> VolcanicSystem:
        """Process raw GVP volcano data into standardized format"""
        try:
            # Extract basic information
            volcano_id = volcano_data.get('volcano_id') or volcano_data.get('gvp_id', '')
            name = volcano_data.get('volcano_name', '')
            
            # Location data
            latitude = float(volcano_data.get('latitude', 0))
            longitude = float(volcano_data.get('longitude', 0))
            elevation = float(volcano_data.get('elevation', 0))
            
            # Classification
            volcano_type_str = volcano_data.get('primary_volcano_type', '').lower()
            volcano_type = self._map_volcano_type(volcano_type_str)
            
            tectonic_setting_str = volcano_data.get('tectonic_setting', '').lower()
            tectonic_setting = self._map_tectonic_setting(tectonic_setting_str)
            
            # Activity information
            last_eruption_str = volcano_data.get('last_known_eruption')
            last_eruption = None
            if last_eruption_str:
                try:
                    last_eruption = datetime.fromisoformat(last_eruption_str.replace('Z', '+00:00'))
                except:
                    pass
            
            # Population data
            population_data = volcano_data.get('population_exposure', {})
            
            return VolcanicSystem(
                volcano_id=volcano_id,
                gvp_id=volcano_data.get('gvp_id'),
                name=name,
                location=(latitude, longitude, elevation),
                country=volcano_data.get('country', ''),
                region=volcano_data.get('region', ''),
                subregion=volcano_data.get('subregion'),
                volcano_type=volcano_type,
                tectonic_setting=tectonic_setting,
                elevation_m=elevation,
                prominence_m=volcano_data.get('prominence'),
                edifice_volume_km3=volcano_data.get('edifice_volume'),
                last_known_eruption=last_eruption,
                eruption_certainty=volcano_data.get('eruption_certainty'),
                activity_evidence=volcano_data.get('activity_evidence', []),
                population_within_5km=population_data.get('within_5km'),
                population_within_10km=population_data.get('within_10km'),
                population_within_30km=population_data.get('within_30km'),
                population_within_100km=population_data.get('within_100km'),
                monitoring_level=volcano_data.get('monitoring_level'),
                monitoring_networks=volcano_data.get('monitoring_networks', []),
                data_sources=['smithsonian_gvp'],
                rock_types=volcano_data.get('rock_types', []),
                morphology=volcano_data.get('morphology'),
                thermal_features=volcano_data.get('thermal_features', [])
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process GVP volcano data: {e}")
            raise
    
    def _map_volcano_type(self, type_str: str) -> VolcanoType:
        """Map GVP volcano type string to enum"""
        type_mapping = {
            'stratovolcano': VolcanoType.STRATOVOLCANO,
            'shield': VolcanoType.SHIELD,
            'cinder cone': VolcanoType.CINDER_CONE,
            'caldera': VolcanoType.CALDERA,
            'complex': VolcanoType.COMPLEX,
            'lava dome': VolcanoType.LAVA_DOME,
            'maar': VolcanoType.MAAR,
            'fissure vent': VolcanoType.FISSURE_VENT,
            'submarine': VolcanoType.SUBMARINE,
            'pyroclastic cone': VolcanoType.PYROCLASTIC_CONE
        }
        
        return type_mapping.get(type_str.lower(), VolcanoType.COMPLEX)
    
    def _map_tectonic_setting(self, setting_str: str) -> TectonicSetting:
        """Map tectonic setting string to enum"""
        setting_mapping = {
            'subduction zone': TectonicSetting.SUBDUCTION_ZONE,
            'mid-ocean ridge': TectonicSetting.MID_OCEAN_RIDGE,
            'hotspot': TectonicSetting.HOTSPOT,
            'continental rift': TectonicSetting.CONTINENTAL_RIFT,
            'back-arc': TectonicSetting.BACK_ARC,
            'intraplate': TectonicSetting.INTRAPLATE,
            'transform fault': TectonicSetting.TRANSFORM_FAULT,
            'continental arc': TectonicSetting.CONTINENTAL_ARC,
            'island arc': TectonicSetting.ISLAND_ARC
        }
        
        return setting_mapping.get(setting_str.lower(), TectonicSetting.INTRAPLATE)
    
    async def _background_synchronization(self):
        """Background task for automated data synchronization"""
        while True:
            try:
                await asyncio.sleep(self.config.get('auto_sync_interval', 3600))
                
                self.logger.info("Starting background synchronization")
                await self.synchronize_with_authoritative_sources()
                
            except Exception as e:
                self.logger.error(f"Background synchronization failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def close(self):
        """Close all connections and cleanup resources"""
        try:
            if self.session:
                await self.session.close()
            
            if self.postgres_pool:
                await self.postgres_pool.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            if self.executor:
                self.executor.shutdown(wait=True)
            
            self.logger.info("Comprehensive volcanic systems database closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing volcanic systems database: {e}")

# Factory function for easy instantiation
async def create_volcanic_systems_database(config: Dict[str, Any] = None) -> ComprehensiveVolcanicSystemsDatabase:
    """Factory function to create and initialize the volcanic systems database"""
    db = ComprehensiveVolcanicSystemsDatabase(config)
    await db.initialize()
    return db