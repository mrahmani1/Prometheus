"""Database management for anomaly detection service"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import asyncpg
import pandas as pd
import redis.asyncio as redis
import torch.nn as nn
import torch.nn.functional as F
from core.config import Settings
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client.client.query_api_async import QueryApiAsync
from influxdb_client.client.write_api_async import WriteApiAsync
from models.schemas import (
    AnomalyCluster,
    DetectedAnomaly,
    GeologicalDataStream,
    PhenomenaDiscovery,
)
from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database connections and operations"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.influxdb_client: Optional[InfluxDBClientAsync] = None
        self.neo4j_driver: Optional[Any] = None

        # InfluxDB API clients
        self.influx_write_api: Optional[WriteApiAsync] = None
        self.influx_query_api: Optional[QueryApiAsync] = None

    async def initialize(self):
        """Initialize all database connections"""
        try:
            # PostgreSQL connection pool
            self.postgres_pool = await asyncpg.create_pool(
                self.settings.postgres_url, min_size=5, max_size=20, command_timeout=60
            )
            logger.info("PostgreSQL connection pool initialized")

            # Redis connection
            self.redis_client = redis.from_url(
                self.settings.redis_url, decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")

            # InfluxDB connection
            self.influxdb_client = InfluxDBClientAsync(
                url=self.settings.influxdb_url,
                token=self.settings.influxdb_token,
                org=self.settings.influxdb_org,
            )
            self.influx_write_api = self.influxdb_client.write_api()
            self.influx_query_api = self.influxdb_client.query_api()
            logger.info("InfluxDB connection established")

            # Neo4j connection
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_url,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )
            # Test connection
            async with self.neo4j_driver.session() as session:
                await session.run("RETURN 1")
            logger.info("Neo4j connection established")

            # Initialize database schemas
            await self._initialize_schemas()

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def close(self):
        """Close all database connections"""
        if self.postgres_pool:
            await self.postgres_pool.close()

        if self.redis_client:
            await self.redis_client.close()

        if self.influxdb_client:
            await self.influxdb_client.close()

        if self.neo4j_driver:
            await self.neo4j_driver.close()

        logger.info("All database connections closed")

    async def _initialize_schemas(self):
        """Initialize database schemas and indexes"""
        # PostgreSQL schema
        async with self.postgres_pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detected_anomalies (
                    anomaly_id VARCHAR PRIMARY KEY,
                    detection_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    anomaly_type VARCHAR NOT NULL,
                    severity VARCHAR NOT NULL,
                    temporal_start TIMESTAMP WITH TIME ZONE,
                    temporal_end TIMESTAMP WITH TIME ZONE,
                    spatial_bounds JSONB,
                    affected_variables TEXT[],
                    anomaly_score FLOAT,
                    deviation_magnitude FLOAT,
                    baseline_comparison JSONB,
                    explanation JSONB,
                    significance JSONB,
                    geological_context TEXT,
                    related_events TEXT[],
                    detection_method VARCHAR,
                    model_version VARCHAR,
                    processing_time FLOAT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp
                ON detected_anomalies(detection_timestamp);

                CREATE INDEX IF NOT EXISTS idx_anomalies_type
                ON detected_anomalies(anomaly_type);

                CREATE INDEX IF NOT EXISTS idx_anomalies_severity
                ON detected_anomalies(severity);

                CREATE TABLE IF NOT EXISTS anomaly_clusters (
                    cluster_id VARCHAR PRIMARY KEY,
                    anomaly_ids TEXT[],
                    cluster_center JSONB,
                    cluster_radius FLOAT,
                    coherence_score FLOAT,
                    temporal_pattern VARCHAR,
                    spatial_pattern VARCHAR,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS phenomena_discoveries (
                    discovery_id VARCHAR PRIMARY KEY,
                    discovery_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    phenomena_name VARCHAR NOT NULL,
                    description TEXT,
                    supporting_anomalies TEXT[],
                    statistical_evidence JSONB,
                    causal_evidence JSONB,
                    phenomena_type VARCHAR,
                    temporal_characteristics JSONB,
                    spatial_characteristics JSONB,
                    confidence_level FLOAT,
                    validation_status VARCHAR,
                    peer_review_status VARCHAR,
                    research_significance VARCHAR,
                    suggested_experiments TEXT[],
                    related_literature TEXT[],
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS continuous_learning_updates (
                    update_id VARCHAR PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    model_component VARCHAR NOT NULL,
                    learning_data JSONB,
                    performance_metrics JSONB,
                    adaptation_rate FLOAT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """
            )

        logger.info("Database schemas initialized")

    # Anomaly storage and retrieval
    async def store_anomaly(self, anomaly: DetectedAnomaly) -> bool:
        """Store detected anomaly in PostgreSQL"""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO detected_anomalies (
                        anomaly_id, detection_timestamp, anomaly_type, severity,
                        temporal_start, temporal_end, spatial_bounds, affected_variables,
                        anomaly_score, deviation_magnitude, baseline_comparison,
                        explanation, significance, geological_context, related_events,
                        detection_method, model_version, processing_time
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                """,
                    anomaly.anomaly_id,
                    anomaly.detection_timestamp,
                    anomaly.anomaly_type.value,
                    anomaly.severity.value,
                    anomaly.temporal_extent.get("start"),
                    anomaly.temporal_extent.get("end"),
                    anomaly.spatial_extent,
                    anomaly.affected_variables,
                    anomaly.anomaly_score,
                    anomaly.deviation_magnitude,
                    anomaly.baseline_comparison,
                    anomaly.explanation.dict() if anomaly.explanation else None,
                    anomaly.significance.dict() if anomaly.significance else None,
                    anomaly.geological_context,
                    anomaly.related_events,
                    anomaly.detection_method,
                    anomaly.model_version,
                    anomaly.processing_time,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to store anomaly {anomaly.anomaly_id}: {e}")
            return False

    async def get_recent_anomalies(
        self,
        hours: int = 24,
        anomaly_type: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent anomalies from PostgreSQL"""
        try:
            query = """
                SELECT * FROM detected_anomalies
                WHERE detection_timestamp > $1
            """
            params = [datetime.utcnow() - timedelta(hours=hours)]

            if anomaly_type:
                query += " AND anomaly_type = $2"
                params.append(anomaly_type)

            if severity:
                if anomaly_type:
                    query += " AND severity = $3"
                else:
                    query += " AND severity = $2"
                params.append(severity)

            query += " ORDER BY detection_timestamp DESC"

            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get recent anomalies: {e}")
            return []

    # Time series data operations
    async def store_geological_data(self, data_stream: GeologicalDataStream):
        """Store geological data in InfluxDB"""
        try:
            points = []
            for data_point in data_stream.data_points:
                point = {
                    "measurement": "geological_data",
                    "tags": {
                        "stream_id": data_stream.stream_id,
                        "sensor_type": data_point.sensor_type,
                        "measurement_type": data_point.measurement_type,
                        "data_source": data_stream.data_source,
                    },
                    "fields": {
                        "value": data_point.value,
                        "quality_score": data_point.quality_score,
                        "latitude": data_point.location.get("lat", 0.0),
                        "longitude": data_point.location.get("lon", 0.0),
                        "elevation": data_point.location.get("elevation", 0.0),
                        "depth": data_point.location.get("depth", 0.0),
                    },
                    "time": data_point.timestamp,
                }
                points.append(point)

            await self.influx_write_api.write(
                bucket=self.settings.influxdb_bucket, record=points
            )

        except Exception as e:
            logger.error(f"Failed to store geological data: {e}")
            raise

    async def query_geological_data(
        self,
        measurement_type: str,
        start_time: datetime,
        end_time: datetime,
        location_filter: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Query geological data from InfluxDB"""
        try:
            query = """
                from(bucket: "{self.settings.influxdb_bucket}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "geological_data")
                |> filter(fn: (r) => r.measurement_type == "{measurement_type}")
            """

            if location_filter:
                # Add spatial filtering if needed
                pass

            tables = await self.influx_query_api.query(query)

            results = []
            for table in tables:
                for record in table.records:
                    results.append(
                        {
                            "timestamp": record.get_time(),
                            "value": record.get_value(),
                            "measurement_type": record.values.get("measurement_type"),
                            "sensor_type": record.values.get("sensor_type"),
                            "quality_score": record.values.get("quality_score"),
                            "location": {
                                "lat": record.values.get("latitude"),
                                "lon": record.values.get("longitude"),
                                "elevation": record.values.get("elevation"),
                                "depth": record.values.get("depth"),
                            },
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Failed to query geological data: {e}")
            return []

    # Caching operations
    async def cache_anomaly_features(self, anomaly_id: str, features: Dict[str, Any]):
        """Cache anomaly features in Redis"""
        try:
            await self.redis_client.hset(
                f"anomaly_features:{anomaly_id}", mapping=features
            )
            await self.redis_client.expire(
                f"anomaly_features:{anomaly_id}", 86400
            )  # 24 hours
        except Exception as e:
            logger.error(f"Failed to cache anomaly features: {e}")

    async def get_cached_features(self, anomaly_id: str) -> Optional[Dict[str, Any]]:
        """Get cached anomaly features from Redis"""
        try:
            features = await self.redis_client.hgetall(f"anomaly_features:{anomaly_id}")
            return features if features else None
        except Exception as e:
            logger.error(f"Failed to get cached features: {e}")
            return None

    # Graph operations for causal relationships
    async def store_causal_relationship(
        self,
        cause: str,
        effect: str,
        strength: float,
        confidence: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Store causal relationship in Neo4j"""
        try:
            async with self.neo4j_driver.session() as session:
                await session.run(
                    """
                    MERGE (c:Variable {name: $cause})
                    MERGE (e:Variable {name: $effect})
                    MERGE (c)-[r:CAUSES]->(e)
                    SET r.strength = $strength,
                        r.confidence = $confidence,
                        r.context = $context,
                        r.updated_at = datetime()
                """,
                    cause=cause,
                    effect=effect,
                    strength=strength,
                    confidence=confidence,
                    context=context,
                )
        except Exception as e:
            logger.error(f"Failed to store causal relationship: {e}")

    async def get_causal_network(self, variable: str, depth: int = 2) -> Dict[str, Any]:
        """Get causal network around a variable from Neo4j"""
        try:
            async with self.neo4j_driver.session() as session:
                result = await session.run(
                    """
                    MATCH path = (v:Variable {name: $variable})-[r:CAUSES*1..$depth]-(connected)
                    RETURN path, r
                """,
                    variable=variable,
                    depth=depth,
                )

                # Process results into network structure
                network = {"nodes": [], "edges": []}
                async for record in result:
                    # Extract nodes and relationships
                    record["path"]
                    # Process path data...

                return network
        except Exception as e:
            logger.error(f"Failed to get causal network: {e}")
            return {"nodes": [], "edges": []}

    # Cluster operations
    async def store_anomaly_cluster(self, cluster: AnomalyCluster) -> bool:
        """Store anomaly cluster"""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO anomaly_clusters (
                        cluster_id, anomaly_ids, cluster_center, cluster_radius,
                        coherence_score, temporal_pattern, spatial_pattern
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                    cluster.cluster_id,
                    cluster.anomaly_ids,
                    cluster.cluster_center,
                    cluster.cluster_radius,
                    cluster.coherence_score,
                    cluster.temporal_pattern,
                    cluster.spatial_pattern,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to store anomaly cluster: {e}")
            return False

    # Phenomena discovery operations
    async def store_phenomena_discovery(self, discovery: PhenomenaDiscovery) -> bool:
        """Store discovered phenomena"""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO phenomena_discoveries (
                        discovery_id, discovery_timestamp, phenomena_name, description,
                        supporting_anomalies, statistical_evidence, causal_evidence,
                        phenomena_type, temporal_characteristics, spatial_characteristics,
                        confidence_level, validation_status, peer_review_status,
                        research_significance, suggested_experiments, related_literature
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                    discovery.discovery_id,
                    discovery.discovery_timestamp,
                    discovery.phenomena_name,
                    discovery.description,
                    discovery.supporting_anomalies,
                    discovery.statistical_evidence,
                    [rel.dict() for rel in discovery.causal_evidence],
                    discovery.phenomena_type,
                    discovery.temporal_characteristics,
                    discovery.spatial_characteristics,
                    discovery.confidence_level,
                    discovery.validation_status,
                    discovery.peer_review_status,
                    discovery.research_significance,
                    discovery.suggested_experiments,
                    discovery.related_literature,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to store phenomena discovery: {e}")
            return False
