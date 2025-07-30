"""Data storage layer for geological data"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import structlog
import json
import motor.motor_asyncio
import asyncpg
import redis.asyncio as redis
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import Point

from models.geological_data import (
    SeismicEvent, ThermalMeasurement, SatelliteObservation,
    FiberOpticMeasurement, StrainMeasurement, GeochemicalSample, AtmosphericMeasurement
)
from storage.mongodb_storage import MongoDBStorage

logger = structlog.get_logger()

class DataStorage:
    """Data storage layer supporting multiple database backends"""
    
    def __init__(self, config: Any):
        self.config = config
        self.is_initialized = False
        
        # Database connections
        self.mongodb_client = None
        self.mongodb_db = None
        self.mongodb_storage = None
        self.postgresql_pool = None
        self.redis_client = None
        self.influxdb_client = None
        
        # Connection strings
        self.mongodb_url = config.mongodb_url
        self.postgresql_url = config.postgresql_url
        self.redis_url = config.redis_url
        self.influxdb_url = config.influxdb_url
        self.influxdb_token = config.influxdb_token
        self.influxdb_org = config.influxdb_org
        self.influxdb_bucket = config.influxdb_bucket
        
    async def initialize(self):
        """Initialize all database connections"""
        try:
            # Initialize MongoDB
            await self._initialize_mongodb()
            
            # Initialize PostgreSQL
            await self._initialize_postgresql()
            
            # Initialize Redis
            await self._initialize_redis()
            
            # Initialize InfluxDB
            await self._initialize_influxdb()
            
            self.is_initialized = True
            logger.info("Data storage initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize data storage", error=str(e))
            raise
    
    async def _initialize_mongodb(self):
        """Initialize MongoDB connection and storage"""
        try:
            # Initialize the dedicated MongoDB storage
            self.mongodb_storage = MongoDBStorage(self.mongodb_url, "prometheus")
            await self.mongodb_storage.initialize()
            
            # Keep the original client for backward compatibility
            self.mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_url)
            self.mongodb_db = self.mongodb_client.prometheus
            
            # Test connection
            await self.mongodb_client.admin.command('ping')
            logger.info("MongoDB connection and storage established")
            
        except Exception as e:
            logger.error("Failed to connect to MongoDB", error=str(e))
            raise
    
    async def _initialize_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.postgresql_pool = await asyncpg.create_pool(
                self.postgresql_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Test connection and create tables if needed
            async with self.postgresql_pool.acquire() as conn:
                await self._create_postgresql_tables(conn)
            
            logger.info("PostgreSQL connection pool established")
            
        except Exception as e:
            logger.error("Failed to connect to PostgreSQL", error=str(e))
            raise
    
    async def _create_postgresql_tables(self, conn):
        """Create PostgreSQL tables for geological data"""
        
        # Enable PostGIS extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        
        # Seismic events table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS seismic_events (
                id SERIAL PRIMARY KEY,
                event_id VARCHAR(255) UNIQUE,
                timestamp TIMESTAMPTZ NOT NULL,
                location GEOGRAPHY(POINT, 4326),
                magnitude REAL,
                depth REAL,
                event_type VARCHAR(100),
                magnitude_type VARCHAR(50),
                source VARCHAR(100),
                confidence REAL,
                quality VARCHAR(50),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Thermal measurements table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS thermal_measurements (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                location GEOGRAPHY(POINT, 4326),
                temperature REAL,
                measurement_method VARCHAR(100),
                spatial_resolution REAL,
                uncertainty REAL,
                confidence REAL,
                source VARCHAR(100),
                quality VARCHAR(50),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Satellite observations table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS satellite_observations (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                location GEOGRAPHY(POINT, 4326),
                satellite VARCHAR(100),
                instrument VARCHAR(100),
                bands TEXT[],
                spatial_resolution REAL,
                cloud_cover REAL,
                scene_id VARCHAR(255),
                source VARCHAR(100),
                quality VARCHAR(50),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_seismic_events_timestamp ON seismic_events(timestamp);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_seismic_events_location ON seismic_events USING GIST(location);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_thermal_measurements_timestamp ON thermal_measurements(timestamp);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_thermal_measurements_location ON thermal_measurements USING GIST(location);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_satellite_observations_timestamp ON satellite_observations(timestamp);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_satellite_observations_location ON satellite_observations USING GIST(location);")
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise
    
    async def _initialize_influxdb(self):
        """Initialize InfluxDB connection"""
        try:
            if self.influxdb_token:
                self.influxdb_client = InfluxDBClientAsync(
                    url=self.influxdb_url,
                    token=self.influxdb_token,
                    org=self.influxdb_org
                )
                
                # Test connection
                health = await self.influxdb_client.health()
                if health.status == "pass":
                    logger.info("InfluxDB connection established")
                else:
                    logger.warning("InfluxDB health check failed")
            else:
                logger.warning("InfluxDB token not provided, time-series storage disabled")
                
        except Exception as e:
            logger.error("Failed to connect to InfluxDB", error=str(e))
            # Don't raise - InfluxDB is optional
    
    async def is_healthy(self) -> bool:
        """Check if all storage backends are healthy"""
        try:
            # Check MongoDB
            if self.mongodb_client:
                await self.mongodb_client.admin.command('ping')
            
            # Check PostgreSQL
            if self.postgresql_pool:
                async with self.postgresql_pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')
            
            # Check Redis
            if self.redis_client:
                await self.redis_client.ping()
            
            # Check InfluxDB (optional)
            if self.influxdb_client:
                health = await self.influxdb_client.health()
                if health.status != "pass":
                    logger.warning("InfluxDB health check failed")
            
            return True
            
        except Exception as e:
            logger.error("Storage health check failed", error=str(e))
            return False
    
    async def store_seismic_event(self, event: SeismicEvent):
        """Store seismic event in PostgreSQL and InfluxDB"""
        try:
            # Store in PostgreSQL
            if self.postgresql_pool:
                async with self.postgresql_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO seismic_events (
                            event_id, timestamp, location, magnitude, depth,
                            event_type, magnitude_type, source, confidence, quality
                        ) VALUES ($1, $2, ST_Point($3, $4), $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (event_id) DO UPDATE SET
                            timestamp = EXCLUDED.timestamp,
                            location = EXCLUDED.location,
                            magnitude = EXCLUDED.magnitude,
                            depth = EXCLUDED.depth,
                            event_type = EXCLUDED.event_type,
                            magnitude_type = EXCLUDED.magnitude_type,
                            source = EXCLUDED.source,
                            confidence = EXCLUDED.confidence,
                            quality = EXCLUDED.quality
                    """, 
                    event.event_id, event.timestamp, 
                    event.location.longitude, event.location.latitude,
                    event.magnitude, event.depth, event.event_type,
                    event.magnitude_type, event.source, event.confidence,
                    event.quality.value
                    )
            
            # Store in InfluxDB for time-series analysis
            if self.influxdb_client:
                point = Point("seismic_events") \
                    .tag("event_type", event.event_type) \
                    .tag("magnitude_type", event.magnitude_type) \
                    .tag("source", event.source) \
                    .tag("quality", event.quality.value) \
                    .field("magnitude", event.magnitude) \
                    .field("depth", event.depth) \
                    .field("latitude", event.location.latitude) \
                    .field("longitude", event.location.longitude) \
                    .field("confidence", event.confidence) \
                    .time(event.timestamp)
                
                write_api = self.influxdb_client.write_api()
                await write_api.write(bucket=self.influxdb_bucket, record=point)
            
            logger.debug("Stored seismic event", event_id=event.event_id)
            
        except Exception as e:
            logger.error("Failed to store seismic event", event_id=event.event_id, error=str(e))
            raise
    
    async def store_thermal_measurement(self, measurement: ThermalMeasurement):
        """Store thermal measurement in PostgreSQL and InfluxDB"""
        try:
            # Store in PostgreSQL
            if self.postgresql_pool:
                async with self.postgresql_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO thermal_measurements (
                            timestamp, location, temperature, measurement_method,
                            spatial_resolution, uncertainty, confidence, source, quality
                        ) VALUES ($1, ST_Point($2, $3), $4, $5, $6, $7, $8, $9, $10)
                    """, 
                    measurement.timestamp,
                    measurement.location.longitude, measurement.location.latitude,
                    measurement.temperature, measurement.measurement_method,
                    measurement.spatial_resolution, measurement.uncertainty,
                    measurement.confidence, measurement.source, measurement.quality.value
                    )
            
            # Store in InfluxDB
            if self.influxdb_client:
                point = Point("thermal_measurements") \
                    .tag("measurement_method", measurement.measurement_method) \
                    .tag("source", measurement.source) \
                    .tag("quality", measurement.quality.value) \
                    .field("temperature", measurement.temperature) \
                    .field("spatial_resolution", measurement.spatial_resolution) \
                    .field("uncertainty", measurement.uncertainty) \
                    .field("confidence", measurement.confidence) \
                    .field("latitude", measurement.location.latitude) \
                    .field("longitude", measurement.location.longitude) \
                    .time(measurement.timestamp)
                
                write_api = self.influxdb_client.write_api()
                await write_api.write(bucket=self.influxdb_bucket, record=point)
            
            logger.debug("Stored thermal measurement")
            
        except Exception as e:
            logger.error("Failed to store thermal measurement", error=str(e))
            raise
    
    async def store_satellite_observation(self, observation: SatelliteObservation):
        """Store satellite observation in PostgreSQL"""
        try:
            if self.postgresql_pool:
                async with self.postgresql_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO satellite_observations (
                            timestamp, location, satellite, instrument, bands,
                            spatial_resolution, cloud_cover, scene_id, source, quality
                        ) VALUES ($1, ST_Point($2, $3), $4, $5, $6, $7, $8, $9, $10, $11)
                    """, 
                    observation.timestamp,
                    observation.location.longitude, observation.location.latitude,
                    observation.satellite, observation.instrument, observation.bands,
                    observation.spatial_resolution, observation.cloud_cover,
                    observation.scene_id, observation.source, observation.quality.value
                    )
            
            logger.debug("Stored satellite observation", scene_id=observation.scene_id)
            
        except Exception as e:
            logger.error("Failed to store satellite observation", error=str(e))
            raise
    
    async def store_fiber_optic_measurement(self, measurement: FiberOpticMeasurement):
        """Store fiber optic measurement in MongoDB and InfluxDB"""
        try:
            # Store in MongoDB for flexible schema
            if self.mongodb_client:
                doc = {
                    "timestamp": measurement.timestamp,
                    "location": {
                        "latitude": measurement.location.latitude,
                        "longitude": measurement.location.longitude,
                        "elevation": measurement.location.elevation,
                        "depth": measurement.location.depth
                    },
                    "measurement_type": measurement.measurement_type,
                    "temperature": measurement.temperature,
                    "temperature_std": measurement.temperature_std,
                    "strain_rate": measurement.strain_rate,
                    "strain_rate_std": measurement.strain_rate_std,
                    "channel_index": measurement.channel_index,
                    "spatial_resolution": measurement.spatial_resolution,
                    "sampling_rate": measurement.sampling_rate,
                    "source": measurement.source,
                    "quality": measurement.quality.value,
                    "created_at": datetime.utcnow()
                }
                
                await self.mongodb_db.fiber_optic_measurements.insert_one(doc)
            
            # Store in InfluxDB for time-series analysis
            if self.influxdb_client:
                point = Point("fiber_optic_measurements") \
                    .tag("measurement_type", measurement.measurement_type) \
                    .tag("source", measurement.source) \
                    .tag("quality", measurement.quality.value) \
                    .field("channel_index", measurement.channel_index) \
                    .field("spatial_resolution", measurement.spatial_resolution) \
                    .field("sampling_rate", measurement.sampling_rate) \
                    .field("latitude", measurement.location.latitude) \
                    .field("longitude", measurement.location.longitude)
                
                if measurement.temperature is not None:
                    point = point.field("temperature", measurement.temperature)
                if measurement.temperature_std is not None:
                    point = point.field("temperature_std", measurement.temperature_std)
                if measurement.strain_rate is not None:
                    point = point.field("strain_rate", measurement.strain_rate)
                if measurement.strain_rate_std is not None:
                    point = point.field("strain_rate_std", measurement.strain_rate_std)
                
                point = point.time(measurement.timestamp)
                
                write_api = self.influxdb_client.write_api()
                await write_api.write(bucket=self.influxdb_bucket, record=point)
            
            logger.debug("Stored fiber optic measurement", channel=measurement.channel_index)
            
        except Exception as e:
            logger.error("Failed to store fiber optic measurement", error=str(e))
            raise
    
    async def store_strain_measurement(self, measurement: StrainMeasurement):
        """Store strain measurement in InfluxDB"""
        try:
            if self.influxdb_client:
                point = Point("strain_measurements") \
                    .tag("measurement_type", measurement.measurement_type) \
                    .tag("source", measurement.source) \
                    .tag("quality", measurement.quality.value) \
                    .field("strain_rate", measurement.strain_rate) \
                    .field("strain_rate_std", measurement.strain_rate_std) \
                    .field("strain_rate_max", measurement.strain_rate_max) \
                    .field("channel_index", measurement.channel_index) \
                    .field("gauge_length", measurement.gauge_length) \
                    .field("sampling_rate", measurement.sampling_rate) \
                    .field("latitude", measurement.location.latitude) \
                    .field("longitude", measurement.location.longitude) \
                    .time(measurement.timestamp)
                
                write_api = self.influxdb_client.write_api()
                await write_api.write(bucket=self.influxdb_bucket, record=point)
            
            logger.debug("Stored strain measurement", channel=measurement.channel_index)
            
        except Exception as e:
            logger.error("Failed to store strain measurement", error=str(e))
            raise
    
    async def store_gravity_measurement(self, measurement: Dict[str, Any]):
        """Store gravity measurement"""
        try:
            # Store in MongoDB using the dedicated storage
            if self.mongodb_storage:
                await self.mongodb_storage.store_unstructured_data(measurement, "gravity_measurements")
            elif self.mongodb_client:
                # Fallback to direct MongoDB client
                doc = {
                    **measurement,
                    "created_at": datetime.utcnow()
                }
                await self.mongodb_db.gravity_measurements.insert_one(doc)
            
            logger.debug("Stored gravity measurement")
            
        except Exception as e:
            logger.error("Failed to store gravity measurement", error=str(e))
            raise
    
    async def store_geochemical_sample(self, sample: GeochemicalSample):
        """Store geochemical sample using MongoDB storage"""
        try:
            if self.mongodb_storage:
                document_id = await self.mongodb_storage.store_geochemical_sample(sample)
                logger.debug("Stored geochemical sample", id=document_id)
                return document_id
            else:
                raise Exception("MongoDB storage not initialized")
        except Exception as e:
            logger.error("Failed to store geochemical sample", error=str(e))
            raise
    
    async def store_atmospheric_measurement_mongodb(self, measurement: AtmosphericMeasurement):
        """Store atmospheric measurement using MongoDB storage"""
        try:
            if self.mongodb_storage:
                document_id = await self.mongodb_storage.store_atmospheric_measurement(measurement)
                logger.debug("Stored atmospheric measurement", id=document_id)
                return document_id
            else:
                raise Exception("MongoDB storage not initialized")
        except Exception as e:
            logger.error("Failed to store atmospheric measurement", error=str(e))
            raise
    
    async def store_unstructured_geological_data(self, data: Dict[str, Any], collection_name: str):
        """Store arbitrary unstructured geological data"""
        try:
            if self.mongodb_storage:
                document_id = await self.mongodb_storage.store_unstructured_data(data, collection_name)
                logger.debug("Stored unstructured geological data", id=document_id, collection=collection_name)
                return document_id
            else:
                raise Exception("MongoDB storage not initialized")
        except Exception as e:
            logger.error("Failed to store unstructured geological data", error=str(e))
            raise
    
    async def bulk_insert_geological_data(self, documents: List[Dict[str, Any]], collection_name: str):
        """Bulk insert geological documents"""
        try:
            if self.mongodb_storage:
                document_ids = await self.mongodb_storage.bulk_insert(documents, collection_name)
                logger.info("Bulk inserted geological data", count=len(document_ids), collection=collection_name)
                return document_ids
            else:
                raise Exception("MongoDB storage not initialized")
        except Exception as e:
            logger.error("Failed bulk insert geological data", error=str(e))
            raise
    
    async def query_geological_data_by_location(self, longitude: float, latitude: float, 
                                               radius_km: float, collection_name: str, limit: int = 100):
        """Query geological data by location"""
        try:
            if self.mongodb_storage:
                results = await self.mongodb_storage.query_by_location(
                    longitude, latitude, radius_km, collection_name, limit
                )
                logger.debug("Location query completed", count=len(results), collection=collection_name)
                return results
            else:
                raise Exception("MongoDB storage not initialized")
        except Exception as e:
            logger.error("Failed location query", error=str(e))
            raise
    
    async def get_geological_data_quality_report(self, collection_name: str):
        """Get data quality report for geological data"""
        try:
            if self.mongodb_storage:
                report = await self.mongodb_storage.get_data_quality_report(collection_name)
                logger.debug("Generated data quality report", collection=collection_name)
                return report
            else:
                raise Exception("MongoDB storage not initialized")
        except Exception as e:
            logger.error("Failed to generate data quality report", error=str(e))
            raise
    
    async def cleanup(self):
        """Cleanup all database connections"""
        try:
            if self.mongodb_storage:
                await self.mongodb_storage.cleanup()
            
            if self.mongodb_client:
                self.mongodb_client.close()
            
            if self.postgresql_pool:
                await self.postgresql_pool.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            if self.influxdb_client:
                await self.influxdb_client.close()
            
            logger.info("Data storage cleanup completed")
            
        except Exception as e:
            logger.error("Error during data storage cleanup", error=str(e))