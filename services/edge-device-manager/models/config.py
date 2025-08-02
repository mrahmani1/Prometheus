"""
Configuration models for Edge Device Manager
"""

from typing import Dict, List, Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class EdgeManagerConfig(BaseSettings):
    """Main configuration for Edge Device Manager"""

    # Service configuration
    service_name: str = "edge-device-manager"
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Database connections
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    postgres_url: str = Field(
        default="postgresql://prometheus:prometheus123@localhost:5432/prometheus",
        env="POSTGRES_URL",
    )

    # Service discovery
    consul_host: str = Field(default="localhost", env="CONSUL_HOST")
    consul_port: int = Field(default=8500, env="CONSUL_PORT")

    # Edge device configuration
    device_heartbeat_interval: int = Field(
        default=30, env="DEVICE_HEARTBEAT_INTERVAL"
    )  # seconds
    device_timeout: int = Field(default=300, env="DEVICE_TIMEOUT")  # seconds
    max_devices_per_region: int = Field(default=100, env="MAX_DEVICES_PER_REGION")

    # Model deployment
    model_registry_url: str = Field(
        default="http://localhost:8081", env="MODEL_REGISTRY_URL"
    )
    deployment_timeout: int = Field(default=600, env="DEPLOYMENT_TIMEOUT")  # seconds
    max_concurrent_deployments: int = Field(default=5, env="MAX_CONCURRENT_DEPLOYMENTS")

    # Data synchronization
    sync_interval: int = Field(default=300, env="SYNC_INTERVAL")  # seconds
    max_sync_retries: int = Field(default=3, env="MAX_SYNC_RETRIES")
    sync_batch_size: int = Field(default=100, env="SYNC_BATCH_SIZE")

    # Security
    jwt_secret: str = Field(default="your-secret-key", env="JWT_SECRET")
    device_auth_token_expiry: int = Field(
        default=86400, env="DEVICE_AUTH_TOKEN_EXPIRY"
    )  # seconds

    # Monitoring
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_interval: int = Field(
        default=60, env="HEALTH_CHECK_INTERVAL"
    )  # seconds

    class Config:
        env_file = ".env"


class DeviceCapabilities(BaseModel):
    """Device hardware and software capabilities"""

    cpu_cores: int
    memory_gb: float
    storage_gb: float
    gpu_available: bool = False
    gpu_memory_gb: Optional[float] = None
    network_bandwidth_mbps: Optional[float] = None
    power_source: str = "battery"  # battery, solar, grid
    operating_system: str
    python_version: str
    docker_available: bool = False
    kubernetes_available: bool = False


class DeviceLocation(BaseModel):
    """Device geographical location"""

    latitude: float
    longitude: float
    altitude: Optional[float] = None
    region: str
    site_name: Optional[str] = None
    timezone: str = "UTC"


class EdgeDeviceConfig(BaseModel):
    """Configuration for an edge device"""

    device_id: str
    device_name: str
    device_type: str  # seismic_station, weather_station, camera, etc.
    location: DeviceLocation
    capabilities: DeviceCapabilities

    # Network configuration
    ip_address: str
    port: int = 8080
    ssl_enabled: bool = False

    # Authentication
    auth_token: str
    certificate_path: Optional[str] = None

    # Operational parameters
    max_cpu_usage: float = 80.0  # percentage
    max_memory_usage: float = 80.0  # percentage
    max_storage_usage: float = 90.0  # percentage

    # Data collection
    data_collection_enabled: bool = True
    data_retention_days: int = 7

    # Model execution
    model_execution_enabled: bool = True
    max_concurrent_models: int = 3


class ModelDeploymentConfig(BaseModel):
    """Configuration for model deployment to edge devices"""

    model_id: str
    model_name: str
    model_version: str
    model_type: str  # pinn, cnn, gan, etc.

    # Resource requirements
    min_cpu_cores: int = 1
    min_memory_gb: float = 1.0
    min_storage_gb: float = 0.5
    requires_gpu: bool = False
    min_gpu_memory_gb: Optional[float] = None

    # Deployment parameters
    container_image: str
    environment_variables: Dict[str, str] = {}
    command_args: List[str] = []

    # Execution configuration
    auto_start: bool = True
    restart_policy: str = "always"  # always, on-failure, never
    health_check_endpoint: str = "/health"

    # Data processing
    input_data_sources: List[str] = []
    output_data_targets: List[str] = []
    processing_interval: Optional[int] = None  # seconds


class SyncConfiguration(BaseModel):
    """Data synchronization configuration"""

    sync_enabled: bool = True
    sync_direction: str = "bidirectional"  # upload, download, bidirectional

    # Data types to sync
    sensor_data: bool = True
    model_outputs: bool = True
    logs: bool = True
    metrics: bool = True

    # Sync parameters
    compression_enabled: bool = True
    encryption_enabled: bool = True
    delta_sync_enabled: bool = True

    # Bandwidth management
    max_bandwidth_mbps: Optional[float] = None
    sync_during_hours: Optional[List[int]] = None  # Hours of day when sync is allowed

    # Conflict resolution
    conflict_resolution: str = "cloud_wins"  # cloud_wins, edge_wins, manual
