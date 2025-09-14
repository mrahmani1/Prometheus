"""
GIS PostgreSQL Configuration
Configuration settings for GIS-enabled PostgreSQL operations
Includes spatial indexing, coordinate systems, and performance tuning parameters
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from backend.core.gis_postgresql_operations import CoordinateSystem, SpatialIndexType, ExportFormat


class DatabaseEnvironment(Enum):
    """Database environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class SpatialIndexConfig:
    """Configuration for spatial indexing"""
    index_type: SpatialIndexType = SpatialIndexType.GIST
    fillfactor: int = 90
    pages_per_range: int = 128  # For BRIN indexes
    quad_segs: int = 16  # For buffer operations
    auto_vacuum: bool = True
    auto_analyze: bool = True


@dataclass
class CoordinateSystemConfig:
    """Configuration for coordinate system handling"""
    default_srid: int = CoordinateSystem.WGS84.value
    supported_systems: List[int] = field(default_factory=lambda: [
        CoordinateSystem.WGS84.value,
        CoordinateSystem.WEB_MERCATOR.value,
        CoordinateSystem.UTM_NORTH.value,
        CoordinateSystem.UTM_SOUTH.value
    ])
    transformation_accuracy: float = 1.0  # meters
    auto_detect_utm: bool = True


@dataclass
class QueryOptimizationConfig:
    """Configuration for query optimization"""
    enable_parallel_queries: bool = True
    max_parallel_workers: int = 4
    work_mem: str = "256MB"
    shared_buffers: str = "1GB"
    effective_cache_size: str = "4GB"
    random_page_cost: float = 1.1
    seq_page_cost: float = 1.0
    cpu_tuple_cost: float = 0.01
    cpu_index_tuple_cost: float = 0.005
    cpu_operator_cost: float = 0.0025


@dataclass
class CacheConfig:
    """Configuration for spatial query caching"""
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    max_cache_size: int = 1000  # Maximum number of cached queries
    cache_large_results: bool = False  # Cache results with > 1000 features
    cache_key_prefix: str = "gis_query"


@dataclass
class ExportConfig:
    """Configuration for geospatial data export"""
    default_format: ExportFormat = ExportFormat.GEOJSON
    max_export_features: int = 100000
    enable_compression: bool = True
    coordinate_precision: int = 6  # Decimal places for coordinates
    include_metadata: bool = True
    chunk_size: int = 10000  # Features per chunk for large exports


@dataclass
class ValidationConfig:
    """Configuration for data quality validation"""
    auto_repair_geometries: bool = True
    validate_on_insert: bool = True
    tolerance_meters: float = 0.01
    max_validation_features: int = 50000
    geometry_simplification_tolerance: float = 0.0001


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring"""
    enable_query_logging: bool = True
    log_slow_queries: bool = True
    slow_query_threshold_ms: int = 1000
    enable_metrics_collection: bool = True
    metrics_retention_days: int = 30


class GISPostgreSQLConfig:
    """Main configuration class for GIS PostgreSQL operations"""
    
    def __init__(self, environment: DatabaseEnvironment = DatabaseEnvironment.DEVELOPMENT):
        self.environment = environment
        
        # Load configuration based on environment
        self._load_environment_config()
        
        # Initialize configuration sections
        self.spatial_indexing = SpatialIndexConfig()
        self.coordinate_systems = CoordinateSystemConfig()
        self.query_optimization = QueryOptimizationConfig()
        self.caching = CacheConfig()
        self.export = ExportConfig()
        self.validation = ValidationConfig()
        self.performance = PerformanceConfig()
        
        # Database connection settings
        self.database = self._get_database_config()
        
        # Spatial extensions settings
        self.extensions = self._get_extensions_config()
        
        # Table-specific configurations
        self.table_configs = self._get_table_configs()
    
    def _load_environment_config(self):
        """Load environment-specific configuration"""
        if self.environment == DatabaseEnvironment.PRODUCTION:
            self._load_production_config()
        elif self.environment == DatabaseEnvironment.STAGING:
            self._load_staging_config()
        elif self.environment == DatabaseEnvironment.TESTING:
            self._load_testing_config()
        else:
            self._load_development_config()
    
    def _load_production_config(self):
        """Load production environment configuration"""
        # Production settings prioritize performance and reliability
        pass
    
    def _load_staging_config(self):
        """Load staging environment configuration"""
        # Staging settings mirror production but with more logging
        pass
    
    def _load_testing_config(self):
        """Load testing environment configuration"""
        # Testing settings prioritize speed and isolation
        pass
    
    def _load_development_config(self):
        """Load development environment configuration"""
        # Development settings prioritize debugging and flexibility
        pass
    
    def _get_database_config(self) -> Dict[str, Any]:
        """Get database connection configuration"""
        base_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'prometheus_geo'),
            'username': os.getenv('POSTGRES_USER', 'prometheus_admin'),
            'password': os.getenv('POSTGRES_PASSWORD', 'GeoPhys2024!PrometheusDB#'),
            'pool_size': int(os.getenv('POSTGRES_POOL_SIZE', 20)),
            'max_overflow': int(os.getenv('POSTGRES_MAX_OVERFLOW', 30)),
            'pool_timeout': int(os.getenv('POSTGRES_POOL_TIMEOUT', 30)),
            'pool_recycle': int(os.getenv('POSTGRES_POOL_RECYCLE', 3600)),
            'echo': os.getenv('POSTGRES_ECHO', 'false').lower() == 'true'
        }
        
        # Environment-specific overrides
        if self.environment == DatabaseEnvironment.PRODUCTION:
            base_config.update({
                'pool_size': 50,
                'max_overflow': 100,
                'echo': False
            })
        elif self.environment == DatabaseEnvironment.TESTING:
            base_config.update({
                'database': 'prometheus_geo_test',
                'pool_size': 5,
                'max_overflow': 10,
                'echo': True
            })
        
        return base_config
    
    def _get_extensions_config(self) -> Dict[str, Any]:
        """Get PostGIS extensions configuration"""
        return {
            'required_extensions': [
                'postgis',
                'postgis_topology',
                'postgis_raster',
                'postgis_sfcgal',
                'timescaledb'
            ],
            'optional_extensions': [
                'address_standardizer',
                'fuzzystrmatch',
                'pg_stat_statements',
                'pg_trgm'
            ],
            'extension_settings': {
                'postgis.gdal_enabled_drivers': 'ENABLE_ALL',
                'postgis.enable_outdb_rasters': 'true',
                'timescaledb.max_background_workers': '8'
            }
        }
    
    def _get_table_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get table-specific configurations"""
        return {
            'volcanic_systems': {
                'spatial_columns': ['location', 'hazard_zone', 'monitoring_network'],
                'primary_spatial_column': 'location',
                'index_type': SpatialIndexType.GIST,
                'enable_clustering': True,
                'partition_by': None,
                'hypertable': False
            },
            'seismic_events': {
                'spatial_columns': ['hypocenter', 'epicenter'],
                'primary_spatial_column': 'epicenter',
                'index_type': SpatialIndexType.GIST,
                'enable_clustering': True,
                'partition_by': 'origin_time',
                'hypertable': True,
                'hypertable_time_column': 'origin_time',
                'hypertable_chunk_interval': '1 month'
            },
            'deformation_measurements': {
                'spatial_columns': ['measurement_point', 'measurement_area'],
                'primary_spatial_column': 'measurement_point',
                'index_type': SpatialIndexType.GIST,
                'enable_clustering': True,
                'partition_by': 'acquisition_time',
                'hypertable': True,
                'hypertable_time_column': 'acquisition_time',
                'hypertable_chunk_interval': '1 week'
            },
            'monitoring_stations': {
                'spatial_columns': ['location'],
                'primary_spatial_column': 'location',
                'index_type': SpatialIndexType.GIST,
                'enable_clustering': False,
                'partition_by': None,
                'hypertable': False
            },
            'geological_units': {
                'spatial_columns': ['geometry', 'representative_point'],
                'primary_spatial_column': 'geometry',
                'index_type': SpatialIndexType.GIST,
                'enable_clustering': True,
                'partition_by': None,
                'hypertable': False
            },
            'tectonic_features': {
                'spatial_columns': ['geometry', 'zone_geometry'],
                'primary_spatial_column': 'geometry',
                'index_type': SpatialIndexType.GIST,
                'enable_clustering': True,
                'partition_by': None,
                'hypertable': False
            }
        }
    
    def get_postgresql_settings(self) -> Dict[str, str]:
        """Get PostgreSQL configuration settings for optimal spatial performance"""
        base_settings = {
            # Memory settings
            'shared_buffers': self.query_optimization.shared_buffers,
            'work_mem': self.query_optimization.work_mem,
            'effective_cache_size': self.query_optimization.effective_cache_size,
            'maintenance_work_mem': '512MB',
            
            # Query planner settings
            'random_page_cost': str(self.query_optimization.random_page_cost),
            'seq_page_cost': str(self.query_optimization.seq_page_cost),
            'cpu_tuple_cost': str(self.query_optimization.cpu_tuple_cost),
            'cpu_index_tuple_cost': str(self.query_optimization.cpu_index_tuple_cost),
            'cpu_operator_cost': str(self.query_optimization.cpu_operator_cost),
            
            # Parallel query settings
            'max_parallel_workers_per_gather': str(self.query_optimization.max_parallel_workers),
            'max_parallel_workers': str(self.query_optimization.max_parallel_workers * 2),
            'parallel_tuple_cost': '0.1',
            'parallel_setup_cost': '1000',
            
            # WAL and checkpoint settings
            'wal_buffers': '64MB',
            'checkpoint_completion_target': '0.9',
            'checkpoint_timeout': '15min',
            'max_wal_size': '4GB',
            'min_wal_size': '1GB',
            
            # Connection settings
            'max_connections': '200',
            'superuser_reserved_connections': '3',
            
            # Logging settings
            'log_min_duration_statement': str(self.performance.slow_query_threshold_ms),
            'log_checkpoints': 'on',
            'log_connections': 'on',
            'log_disconnections': 'on',
            'log_lock_waits': 'on',
            
            # Statistics settings
            'track_activities': 'on',
            'track_counts': 'on',
            'track_io_timing': 'on',
            'track_functions': 'all',
            
            # PostGIS specific settings
            'postgis.gdal_enabled_drivers': 'ENABLE_ALL',
            'postgis.enable_outdb_rasters': 'true'
        }
        
        # Environment-specific overrides
        if self.environment == DatabaseEnvironment.PRODUCTION:
            base_settings.update({
                'shared_buffers': '8GB',
                'effective_cache_size': '24GB',
                'work_mem': '512MB',
                'maintenance_work_mem': '2GB',
                'max_connections': '500',
                'log_min_duration_statement': '5000'  # 5 seconds
            })
        elif self.environment == DatabaseEnvironment.DEVELOPMENT:
            base_settings.update({
                'shared_buffers': '256MB',
                'effective_cache_size': '1GB',
                'work_mem': '64MB',
                'maintenance_work_mem': '256MB',
                'max_connections': '50',
                'log_min_duration_statement': '1000'  # 1 second
            })
        
        return base_settings
    
    def get_spatial_index_settings(self, table_name: str) -> Dict[str, Any]:
        """Get spatial index settings for a specific table"""
        table_config = self.table_configs.get(table_name, {})
        
        return {
            'index_type': table_config.get('index_type', self.spatial_indexing.index_type),
            'fillfactor': self.spatial_indexing.fillfactor,
            'pages_per_range': self.spatial_indexing.pages_per_range,
            'spatial_columns': table_config.get('spatial_columns', []),
            'primary_spatial_column': table_config.get('primary_spatial_column'),
            'enable_clustering': table_config.get('enable_clustering', False)
        }
    
    def get_hypertable_settings(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get TimescaleDB hypertable settings for a specific table"""
        table_config = self.table_configs.get(table_name, {})
        
        if not table_config.get('hypertable', False):
            return None
        
        return {
            'time_column': table_config.get('hypertable_time_column'),
            'chunk_time_interval': table_config.get('hypertable_chunk_interval', '1 month'),
            'create_default_indexes': True,
            'if_not_exists': True,
            'partitioning_column': table_config.get('partition_by'),
            'number_partitions': 4,
            'associated_schema_name': None,
            'associated_table_prefix': None,
            'migrate_data': False
        }
    
    def get_export_settings(self, export_format: ExportFormat) -> Dict[str, Any]:
        """Get export settings for a specific format"""
        base_settings = {
            'max_features': self.export.max_export_features,
            'coordinate_precision': self.export.coordinate_precision,
            'include_metadata': self.export.include_metadata,
            'chunk_size': self.export.chunk_size,
            'enable_compression': self.export.enable_compression
        }
        
        # Format-specific settings
        if export_format == ExportFormat.GEOJSON:
            base_settings.update({
                'pretty_print': self.environment == DatabaseEnvironment.DEVELOPMENT,
                'include_crs': True,
                'coordinate_order': 'lon_lat'
            })
        elif export_format == ExportFormat.KML:
            base_settings.update({
                'include_description': True,
                'include_extended_data': True,
                'altitude_mode': 'clampToGround'
            })
        elif export_format == ExportFormat.SHAPEFILE:
            base_settings.update({
                'encoding': 'UTF-8',
                'include_prj': True,
                'max_field_length': 254
            })
        
        return base_settings
    
    def validate_config(self) -> List[str]:
        """Validate configuration settings and return any issues"""
        issues = []
        
        # Validate database settings
        if not self.database.get('host'):
            issues.append("Database host not specified")
        
        if not self.database.get('database'):
            issues.append("Database name not specified")
        
        # Validate coordinate systems
        if self.coordinate_systems.default_srid not in self.coordinate_systems.supported_systems:
            issues.append("Default SRID not in supported systems list")
        
        # Validate memory settings
        try:
            work_mem_mb = int(self.query_optimization.work_mem.replace('MB', ''))
            shared_buffers_mb = int(self.query_optimization.shared_buffers.replace('GB', '')) * 1024
            
            if work_mem_mb > shared_buffers_mb / 4:
                issues.append("work_mem should not exceed 25% of shared_buffers")
        except ValueError:
            issues.append("Invalid memory configuration format")
        
        # Validate table configurations
        for table_name, config in self.table_configs.items():
            if config.get('hypertable') and not config.get('hypertable_time_column'):
                issues.append(f"Hypertable {table_name} missing time column specification")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment.value,
            'database': self.database,
            'spatial_indexing': self.spatial_indexing.__dict__,
            'coordinate_systems': self.coordinate_systems.__dict__,
            'query_optimization': self.query_optimization.__dict__,
            'caching': self.caching.__dict__,
            'export': self.export.__dict__,
            'validation': self.validation.__dict__,
            'performance': self.performance.__dict__,
            'extensions': self.extensions,
            'table_configs': self.table_configs
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GISPostgreSQLConfig':
        """Create configuration from dictionary"""
        environment = DatabaseEnvironment(config_dict.get('environment', 'development'))
        config = cls(environment)
        
        # Update configuration sections
        if 'spatial_indexing' in config_dict:
            config.spatial_indexing = SpatialIndexConfig(**config_dict['spatial_indexing'])
        
        if 'coordinate_systems' in config_dict:
            config.coordinate_systems = CoordinateSystemConfig(**config_dict['coordinate_systems'])
        
        if 'query_optimization' in config_dict:
            config.query_optimization = QueryOptimizationConfig(**config_dict['query_optimization'])
        
        if 'caching' in config_dict:
            config.caching = CacheConfig(**config_dict['caching'])
        
        if 'export' in config_dict:
            config.export = ExportConfig(**config_dict['export'])
        
        if 'validation' in config_dict:
            config.validation = ValidationConfig(**config_dict['validation'])
        
        if 'performance' in config_dict:
            config.performance = PerformanceConfig(**config_dict['performance'])
        
        # Update other sections
        if 'database' in config_dict:
            config.database.update(config_dict['database'])
        
        if 'extensions' in config_dict:
            config.extensions.update(config_dict['extensions'])
        
        if 'table_configs' in config_dict:
            config.table_configs.update(config_dict['table_configs'])
        
        return config


# Default configuration instances
DEFAULT_CONFIG = GISPostgreSQLConfig(DatabaseEnvironment.DEVELOPMENT)
PRODUCTION_CONFIG = GISPostgreSQLConfig(DatabaseEnvironment.PRODUCTION)
TESTING_CONFIG = GISPostgreSQLConfig(DatabaseEnvironment.TESTING)