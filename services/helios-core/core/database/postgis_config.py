"""
PostgreSQL with PostGIS Configuration for Prometheus AI Platform

Configuration settings for advanced spatial-temporal database operations,
materialized view refresh schedules, and performance optimization.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import timedelta

@dataclass
class PostGISConfig:
    """Configuration for PostgreSQL with PostGIS operations"""
    
    # Database connection settings
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "prometheus")
    username: str = os.getenv("POSTGRES_USER", "prometheus")
    password: str = os.getenv("POSTGRES_PASSWORD", "password")
    
    # Connection pool settings
    min_pool_size: int = int(os.getenv("POSTGRES_MIN_POOL_SIZE", "5"))
    max_pool_size: int = int(os.getenv("POSTGRES_MAX_POOL_SIZE", "20"))
    command_timeout: int = int(os.getenv("POSTGRES_COMMAND_TIMEOUT", "60"))
    
    # Spatial reference systems
    default_srid: int = 4326  # WGS84
    utm_srid: int = 9001  # Custom UTM projection for geological analysis
    
    # Materialized view refresh settings
    auto_refresh_enabled: bool = os.getenv("POSTGRES_AUTO_REFRESH", "true").lower() == "true"
    refresh_interval_hours: int = int(os.getenv("POSTGRES_REFRESH_INTERVAL", "6"))
    
    # Query performance settings
    enable_spatial_indexes: bool = True
    enable_temporal_indexes: bool = True
    enable_composite_indexes: bool = True
    
    # Data retention settings
    seismic_data_retention_days: int = int(os.getenv("SEISMIC_RETENTION_DAYS", "3650"))  # 10 years
    deformation_data_retention_days: int = int(os.getenv("DEFORMATION_RETENTION_DAYS", "1825"))  # 5 years
    thermal_data_retention_days: int = int(os.getenv("THERMAL_RETENTION_DAYS", "1095"))  # 3 years
    
    # Spatial analysis settings
    default_spatial_radius: float = 50000.0  # 50km default search radius
    max_spatial_radius: float = 500000.0  # 500km maximum search radius
    default_temporal_window_days: int = 30
    max_temporal_window_days: int = 365
    
    # Geological calculation parameters
    default_shear_modulus: float = 3e10  # Pa, for stress drop calculations
    earth_radius: float = 6371000.0  # meters
    
    # Performance optimization settings
    enable_parallel_queries: bool = True
    work_mem_mb: int = int(os.getenv("POSTGRES_WORK_MEM_MB", "256"))
    shared_buffers_mb: int = int(os.getenv("POSTGRES_SHARED_BUFFERS_MB", "1024"))
    
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string"""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
    
    @property
    def async_connection_string(self) -> str:
        """Generate async PostgreSQL connection string"""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
    
    def get_materialized_view_refresh_schedule(self) -> Dict[str, timedelta]:
        """Get refresh schedule for materialized views"""
        return {
            "mv_seismic_activity_summary": timedelta(hours=1),
            "mv_deformation_analysis": timedelta(hours=6),
            "mv_thermal_anomalies": timedelta(hours=2),
            "mv_geological_correlation": timedelta(hours=12)
        }
    
    def get_spatial_index_configurations(self) -> List[Dict[str, str]]:
        """Get spatial index configurations for optimization"""
        return [
            {
                "table": "geological.locations",
                "column": "geom",
                "index_type": "GIST",
                "index_name": "idx_locations_geom"
            },
            {
                "table": "geological.locations",
                "column": "geom_utm",
                "index_type": "GIST",
                "index_name": "idx_locations_geom_utm"
            },
            {
                "table": "geological.locations",
                "column": "geom_3d",
                "index_type": "GIST",
                "index_name": "idx_locations_geom_3d"
            },
            {
                "table": "simulations.simulation_results",
                "column": "spatial_location",
                "index_type": "GIST",
                "index_name": "idx_simulation_results_location"
            }
        ]
    
    def get_temporal_index_configurations(self) -> List[Dict[str, str]]:
        """Get temporal index configurations for optimization"""
        return [
            {
                "table": "geological.seismic_events",
                "column": "event_time",
                "index_type": "BTREE",
                "index_name": "idx_seismic_events_time"
            },
            {
                "table": "geological.deformation_measurements",
                "column": "measurement_time",
                "index_type": "BTREE",
                "index_name": "idx_deformation_time"
            },
            {
                "table": "geological.thermal_measurements",
                "column": "measurement_time",
                "index_type": "BTREE",
                "index_name": "idx_thermal_time"
            },
            {
                "table": "monitoring.sensor_readings",
                "column": "reading_time",
                "index_type": "BTREE",
                "index_name": "idx_sensor_readings_time"
            }
        ]
    
    def get_composite_index_configurations(self) -> List[Dict[str, str]]:
        """Get composite index configurations for complex queries"""
        return [
            {
                "table": "geological.seismic_events",
                "columns": ["magnitude", "depth"],
                "index_type": "BTREE",
                "index_name": "idx_seismic_magnitude_depth"
            },
            {
                "table": "geological.thermal_measurements",
                "columns": ["temperature", "measurement_time"],
                "index_type": "BTREE",
                "index_name": "idx_thermal_temp_time"
            },
            {
                "table": "geological.locations",
                "columns": ["elevation", "depth"],
                "index_type": "BTREE",
                "index_name": "idx_locations_elevation_depth",
                "condition": "WHERE elevation IS NOT NULL AND depth IS NOT NULL"
            }
        ]
    
    def get_partition_configurations(self) -> List[Dict[str, str]]:
        """Get table partitioning configurations for large datasets"""
        return [
            {
                "table": "monitoring.sensor_readings",
                "partition_type": "RANGE",
                "partition_column": "reading_time",
                "partition_interval": "MONTH"
            },
            {
                "table": "geological.seismic_events",
                "partition_type": "RANGE",
                "partition_column": "event_time",
                "partition_interval": "YEAR"
            }
        ]
    
    def get_data_retention_policies(self) -> Dict[str, Dict[str, int]]:
        """Get data retention policies for different data types"""
        return {
            "geological.seismic_events": {
                "retention_days": self.seismic_data_retention_days,
                "archive_after_days": self.seismic_data_retention_days // 2
            },
            "geological.deformation_measurements": {
                "retention_days": self.deformation_data_retention_days,
                "archive_after_days": self.deformation_data_retention_days // 2
            },
            "geological.thermal_measurements": {
                "retention_days": self.thermal_data_retention_days,
                "archive_after_days": self.thermal_data_retention_days // 2
            },
            "monitoring.sensor_readings": {
                "retention_days": 365,  # 1 year for sensor readings
                "archive_after_days": 90   # Archive after 3 months
            }
        }
    
    def get_performance_tuning_parameters(self) -> Dict[str, str]:
        """Get PostgreSQL performance tuning parameters"""
        return {
            "work_mem": f"{self.work_mem_mb}MB",
            "shared_buffers": f"{self.shared_buffers_mb}MB",
            "effective_cache_size": f"{self.shared_buffers_mb * 4}MB",
            "random_page_cost": "1.1",  # For SSD storage
            "seq_page_cost": "1.0",
            "cpu_tuple_cost": "0.01",
            "cpu_index_tuple_cost": "0.005",
            "cpu_operator_cost": "0.0025",
            "max_parallel_workers_per_gather": "4" if self.enable_parallel_queries else "0",
            "max_parallel_workers": "8" if self.enable_parallel_queries else "0",
            "enable_partitionwise_join": "on",
            "enable_partitionwise_aggregate": "on"
        }
    
    def get_geological_constants(self) -> Dict[str, float]:
        """Get geological and geophysical constants for calculations"""
        return {
            "earth_radius_m": self.earth_radius,
            "gravitational_acceleration": 9.81,  # m/s²
            "average_crustal_density": 2700.0,  # kg/m³
            "average_mantle_density": 3300.0,  # kg/m³
            "average_crustal_thickness": 35000.0,  # meters
            "seismic_wave_velocity_p": 6000.0,  # m/s (average P-wave velocity)
            "seismic_wave_velocity_s": 3500.0,  # m/s (average S-wave velocity)
            "shear_modulus_crust": self.default_shear_modulus,  # Pa
            "bulk_modulus_crust": 5e10,  # Pa
            "poisson_ratio_crust": 0.25,  # dimensionless
            "thermal_diffusivity_rock": 1e-6,  # m²/s
            "specific_heat_rock": 1000.0,  # J/(kg·K)
            "thermal_conductivity_rock": 3.0,  # W/(m·K)
            "gas_constant": 8.314,  # J/(mol·K)
            "avogadro_number": 6.022e23,  # mol⁻¹
            "stefan_boltzmann_constant": 5.67e-8  # W/(m²·K⁴)
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration parameters and return any errors"""
        errors = []
        
        if self.min_pool_size <= 0:
            errors.append("min_pool_size must be greater than 0")
        
        if self.max_pool_size <= self.min_pool_size:
            errors.append("max_pool_size must be greater than min_pool_size")
        
        if self.command_timeout <= 0:
            errors.append("command_timeout must be greater than 0")
        
        if self.default_spatial_radius <= 0:
            errors.append("default_spatial_radius must be greater than 0")
        
        if self.max_spatial_radius <= self.default_spatial_radius:
            errors.append("max_spatial_radius must be greater than default_spatial_radius")
        
        if self.default_temporal_window_days <= 0:
            errors.append("default_temporal_window_days must be greater than 0")
        
        if self.max_temporal_window_days <= self.default_temporal_window_days:
            errors.append("max_temporal_window_days must be greater than default_temporal_window_days")
        
        if not self.host:
            errors.append("host cannot be empty")
        
        if not self.database:
            errors.append("database cannot be empty")
        
        if not self.username:
            errors.append("username cannot be empty")
        
        return errors

# Global configuration instance
postgis_config = PostGISConfig()

# Validate configuration on import
config_errors = postgis_config.validate_configuration()
if config_errors:
    raise ValueError(f"PostGIS configuration errors: {', '.join(config_errors)}")