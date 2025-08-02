"""Configuration management for anomaly detection service"""

from functools import lru_cache
from typing import List, Optional

import pandas as pd
import torch.nn as nn
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # Service configuration
    port: int = Field(default=8007, description="Service port")
    debug: bool = Field(default=False, description="Debug mode")
    allowed_origins: List[str] = Field(
        default=["*"], description="CORS allowed origins"
    )

    # Database connections
    postgres_url: str = Field(
        default="postgresql://prometheus:prometheus123@localhost:5432/prometheus",
        description="PostgreSQL connection URL",
    )
    influxdb_url: str = Field(
        default="http://localhost:8086", description="InfluxDB connection URL"
    )
    influxdb_token: str = Field(
        default="prometheus-token", description="InfluxDB authentication token"
    )
    influxdb_org: str = Field(default="prometheus", description="InfluxDB organization")
    influxdb_bucket: str = Field(
        default="geological_data", description="InfluxDB bucket name"
    )
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis connection URL"
    )
    neo4j_url: str = Field(
        default="bolt://localhost:7687", description="Neo4j connection URL"
    )
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="prometheus123", description="Neo4j password")

    # Anomaly detection parameters
    anomaly_threshold: float = Field(
        default=0.95, description="Anomaly detection threshold (0-1)"
    )
    multivariate_window_size: int = Field(
        default=100, description="Window size for multivariate analysis"
    )
    contextual_memory_size: int = Field(
        default=1000, description="Size of contextual memory for domain knowledge"
    )
    collective_sequence_length: int = Field(
        default=50, description="Sequence length for collective anomaly detection"
    )

    # Reinforcement learning parameters
    rl_learning_rate: float = Field(
        default=0.001, description="Learning rate for RL agent"
    )
    rl_exploration_rate: float = Field(
        default=0.1, description="Exploration rate for RL agent"
    )
    rl_memory_size: int = Field(default=10000, description="Memory size for RL agent")

    # Continuous learning parameters
    continuous_learning_enabled: bool = Field(
        default=True, description="Enable continuous learning"
    )
    model_update_frequency: int = Field(
        default=3600, description="Model update frequency in seconds"  # 1 hour
    )
    adaptation_rate: float = Field(
        default=0.01, description="Adaptation rate for continuous learning"
    )

    # Multi-scale analysis parameters
    temporal_scales: List[str] = Field(
        default=["seconds", "minutes", "hours", "days", "months", "years"],
        description="Temporal scales for multi-scale analysis",
    )
    spatial_scales: List[str] = Field(
        default=["local", "regional", "global"],
        description="Spatial scales for multi-scale analysis",
    )

    # Model paths
    model_cache_dir: str = Field(
        default="./models/cache", description="Directory for model caching"
    )
    pretrained_models_dir: str = Field(
        default="./models/pretrained", description="Directory for pretrained models"
    )

    # Monitoring and alerting
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    alert_webhook_url: Optional[str] = Field(
        default=None, description="Webhook URL for anomaly alerts"
    )

    # External services
    helios_core_url: str = Field(
        default="http://helios-core:8001", description="Helios core service URL"
    )
    knowledge_graph_url: str = Field(
        default="http://knowledge-graph:8006", description="Knowledge graph service URL"
    )
    volcano_watchtower_url: str = Field(
        default="http://volcano-watchtower:8003",
        description="Volcano Watchtower service URL",
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
