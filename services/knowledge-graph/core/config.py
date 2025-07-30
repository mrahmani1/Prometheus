"""
Configuration settings for the Scientific Knowledge Graph Service
"""

import os
from typing import List, Dict, Any
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    service_name: str = "knowledge-graph"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Neo4j configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="prometheus123", env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="prometheus", env="NEO4J_DATABASE")
    
    # External database connections
    postgres_uri: str = Field(
        default="postgresql://prometheus:prometheus123@localhost:5432/prometheus",
        env="POSTGRES_URI"
    )
    influxdb_url: str = Field(default="http://localhost:8086", env="INFLUXDB_URL")
    influxdb_token: str = Field(default="prometheus-token", env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="prometheus", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="geological_data", env="INFLUXDB_BUCKET")
    
    # External API configurations
    semantic_scholar_api_key: str = Field(default="", env="SEMANTIC_SCHOLAR_API_KEY")
    crossref_api_key: str = Field(default="", env="CROSSREF_API_KEY")
    pubmed_api_key: str = Field(default="", env="PUBMED_API_KEY")
    
    # Knowledge extraction settings
    literature_sources: List[str] = [
        "semantic_scholar",
        "crossref",
        "pubmed",
        "arxiv",
        "geonames"
    ]
    
    # Ontology sources
    ontology_sources: Dict[str, str] = {
        "geonames": "http://www.geonames.org/ontology#",
        "sweet": "http://sweetontology.net/",
        "envo": "http://purl.obolibrary.org/obo/envo.owl",
        "geological_time": "http://resource.geosciml.org/ontology/timescale/gts.owl"
    }
    
    # Causal discovery parameters
    causal_discovery_confidence_threshold: float = 0.7
    temporal_window_size: int = 100  # Time steps for temporal analysis
    federated_sync_interval: int = 3600  # Seconds between federated updates
    
    # Model paths
    nlp_model_path: str = Field(default="./models/geological_nlp", env="NLP_MODEL_PATH")
    embedding_model_path: str = Field(default="./models/embeddings", env="EMBEDDING_MODEL_PATH")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()