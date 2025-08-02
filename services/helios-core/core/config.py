"""Configuration management for Helios Core Service"""

from typing import List, Optional

import torch.nn as nn
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Service configuration
    service_name: str = "helios-core"
    service_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Server configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")

    # Service discovery
    consul_host: str = Field(default="localhost", env="CONSUL_HOST")
    consul_port: int = Field(default=8500, env="CONSUL_PORT")

    # Database connections
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="prometheus", env="POSTGRES_DB")
    postgres_user: str = Field(default="prometheus", env="POSTGRES_USER")
    postgres_password: str = Field(default="prometheus123", env="POSTGRES_PASSWORD")

    influxdb_host: str = Field(default="localhost", env="INFLUXDB_HOST")
    influxdb_port: int = Field(default=8086, env="INFLUXDB_PORT")
    influxdb_org: str = Field(default="prometheus", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="geological-data", env="INFLUXDB_BUCKET")
    influxdb_token: Optional[str] = Field(default=None, env="INFLUXDB_TOKEN")

    neo4j_host: str = Field(default="localhost", env="NEO4J_HOST")
    neo4j_port: int = Field(default=7687, env="NEO4J_PORT")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="prometheus123", env="NEO4J_PASSWORD")

    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

    # AI/ML configuration
    model_cache_dir: str = Field(default="/app/models", env="MODEL_CACHE_DIR")
    max_model_memory: int = Field(default=4096, env="MAX_MODEL_MEMORY_MB")  # MB
    gpu_enabled: bool = Field(default=False, env="GPU_ENABLED")
    gpu_device_ids: List[int] = Field(default_factory=list, env="GPU_DEVICE_IDS")

    # Physics simulation parameters
    default_simulation_resolution: float = Field(
        default=1000.0, env="DEFAULT_SIMULATION_RESOLUTION"
    )  # meters
    max_simulation_time: int = Field(default=3600, env="MAX_SIMULATION_TIME")  # seconds
    physics_solver_tolerance: float = Field(
        default=1e-6, env="PHYSICS_SOLVER_TOLERANCE"
    )

    # Security
    secret_key: str = Field(default="prometheus-secret-key", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
