"""Configuration settings for data ingestion service"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseSettings, Field
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DataSourceType(str, Enum):
    USGS = "usgs"
    NASA = "nasa"
    ESA = "esa"
    COMMERCIAL_SATELLITE = "commercial_satellite"
    FIBER_OPTIC = "fiber_optic"
    SEISMOMETER = "seismometer"
    ATMOSPHERIC = "atmospheric"
    HYDROACOUSTIC = "hydroacoustic"

class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    service_name: str = "prometheus-data-ingestion"
    service_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    
    # Server configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8001, env="PORT")
    
    # Service discovery
    consul_host: str = Field(default="consul", env="CONSUL_HOST")
    consul_port: int = Field(default=8500, env="CONSUL_PORT")
    
    # Database connections
    mongodb_url: str = Field(default="mongodb://mongodb:27017", env="MONGODB_URL")
    postgresql_url: str = Field(default="postgresql://postgres:password@postgresql:5432/prometheus", env="POSTGRESQL_URL")
    redis_url: str = Field(default="redis://redis:6379", env="REDIS_URL")
    influxdb_url: str = Field(default="http://influxdb:8086", env="INFLUXDB_URL")
    influxdb_token: str = Field(default="", env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="prometheus", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="geological_data", env="INFLUXDB_BUCKET")
    
    # Kafka configuration
    kafka_bootstrap_servers: List[str] = Field(default=["kafka:9092"], env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_topic_prefix: str = Field(default="prometheus", env="KAFKA_TOPIC_PREFIX")
    
    # Data source configurations
    usgs_api_base_url: str = Field(default="https://earthquake.usgs.gov/fdsnws", env="USGS_API_BASE_URL")
    usgs_api_key: Optional[str] = Field(default=None, env="USGS_API_KEY")
    
    nasa_earthdata_username: Optional[str] = Field(default=None, env="NASA_EARTHDATA_USERNAME")
    nasa_earthdata_password: Optional[str] = Field(default=None, env="NASA_EARTHDATA_PASSWORD")
    
    esa_copernicus_username: Optional[str] = Field(default=None, env="ESA_COPERNICUS_USERNAME")
    esa_copernicus_password: Optional[str] = Field(default=None, env="ESA_COPERNICUS_PASSWORD")
    
    # Fiber optic sensing configuration
    das_sampling_rate: int = Field(default=1000, env="DAS_SAMPLING_RATE")  # Hz
    dts_sampling_rate: int = Field(default=1, env="DTS_SAMPLING_RATE")  # Hz
    fiber_length_km: float = Field(default=100.0, env="FIBER_LENGTH_KM")
    
    # Seismometer network configuration
    seismic_networks: List[str] = Field(default=["II", "IU", "US", "TA"], env="SEISMIC_NETWORKS")
    seismic_stations: List[str] = Field(default=[], env="SEISMIC_STATIONS")
    seismic_channels: List[str] = Field(default=["BHZ", "BHN", "BHE"], env="SEISMIC_CHANNELS")
    
    # Data processing configuration
    max_concurrent_downloads: int = Field(default=10, env="MAX_CONCURRENT_DOWNLOADS")
    data_retention_days: int = Field(default=365, env="DATA_RETENTION_DAYS")
    batch_size: int = Field(default=1000, env="BATCH_SIZE")
    
    # Quality control thresholds
    seismic_quality_threshold: float = Field(default=0.8, env="SEISMIC_QUALITY_THRESHOLD")
    deformation_quality_threshold: float = Field(default=0.9, env="DEFORMATION_QUALITY_THRESHOLD")
    atmospheric_quality_threshold: float = Field(default=0.7, env="ATMOSPHERIC_QUALITY_THRESHOLD")
    
    # Rate limiting
    api_rate_limit_per_minute: int = Field(default=60, env="API_RATE_LIMIT_PER_MINUTE")
    
    # Security
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()