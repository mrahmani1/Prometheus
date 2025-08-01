"""
Configuration settings for the Graph Database Service
"""

from typing import Optional

import torch.nn as nn
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""

    # Service configuration
    service_name: str = "graph-database"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # Neo4j configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="prometheus123", env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="prometheus", env="NEO4J_DATABASE")

    # Connection pool settings
    max_connection_lifetime: int = Field(
        default=3600, env="NEO4J_MAX_CONNECTION_LIFETIME"
    )
    max_connection_pool_size: int = Field(
        default=50, env="NEO4J_MAX_CONNECTION_POOL_SIZE"
    )
    connection_acquisition_timeout: int = Field(
        default=60, env="NEO4J_CONNECTION_TIMEOUT"
    )

    # API configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    # Logging configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Security settings
    enable_auth: bool = Field(default=True, env="ENABLE_AUTH")
    jwt_secret: Optional[str] = Field(default=None, env="JWT_SECRET")

    # Performance settings
    query_timeout: int = Field(default=300, env="QUERY_TIMEOUT")  # 5 minutes
    batch_size: int = Field(default=1000, env="BATCH_SIZE")

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
