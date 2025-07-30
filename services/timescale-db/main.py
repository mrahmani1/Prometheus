#!/usr/bin/env python3
"""
TimescaleDB Service for Prometheus AI Platform
Specialized time-series database optimization for high-frequency sensor data
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import asyncpg
import structlog
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import os
from contextlib import asynccontextmanager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Database connection pool
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global db_pool
    
    # Startup
    logger.info("Starting TimescaleDB service")
    db_pool = await create_db_pool()
    await initialize_timescale_extensions()
    await setup_hypertables()
    await setup_compression_policies()
    await setup_retention_policies()
    await setup_continuous_aggregates()
    
    yield
    
    # Shutdown
    logger.info("Shutting down TimescaleDB service")
    if db_pool:
        await db_pool.close()

app = FastAPI(
    title="TimescaleDB Service",
    description="Specialized time-series database for geological sensor data",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SensorReading(BaseModel):
    sensor_id: str
    timestamp: datetime
    measurement_type: str
    value: float
    unit: str
    quality_flag: str = "good"
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    location_elevation: Optional[float] = None

class TimeSeriesQuery(BaseModel):
    sensor_ids: List[str]
    start_time: datetime
    end_time: datetime
    measurement_types: Optional[List[str]] = None
    aggregation: Optional[str] = None  # avg, min, max, sum, count
    interval: Optional[str] = "1h"  # 1m, 5m, 1h, 1d, etc.

class TrendAnalysisRequest(BaseModel):
    sensor_id: str
    measurement_type: str
    start_time: datetime
    end_time: datetime
    trend_window: str = "1d"  # sliding window for trend analysis

async def create_db_pool():
    """Create database connection pool"""
    database_url = os.getenv(
        "TIMESCALE_DATABASE_URL",
        "postgresql://prometheus:prometheus123@localhost:5432/prometheus_timeseries"
    )
    
    try:
        pool = await asyncpg.create_pool(
            database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        logger.info("Database connection pool created successfully")
        return pool
    except Exception as e:
        logger.error("Failed to create database pool", error=str(e))
        raise

async def get_db_connection():
    """Get database connection from pool"""
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")
    return db_pool

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "healthy", "timestamp": datetime.utcnow()}
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        async with db_pool.acquire() as conn:
            # Check if TimescaleDB extension is available
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'"
            )
            if result == 0:
                raise HTTPException(status_code=503, detail="TimescaleDB extension not available")
        
        return {"status": "ready", "timestamp": datetime.utcnow()}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("ENVIRONMENT") == "development"
    )

async def initialize_timescale_extensions():
    """Initialize TimescaleDB extensions and functions"""
    async with db_pool.acquire() as conn:
        try:
            # Enable TimescaleDB extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            
            # Enable PostGIS for spatial data
            await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis CASCADE;")
            
            # Enable additional extensions for advanced analytics
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements CASCADE;")
            
            logger.info("TimescaleDB extensions initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize TimescaleDB extensions", error=str(e))
            raise

async def setup_hypertables():
    """Set up hypertables for time-series data"""
    async with db_pool.acquire() as conn:
        try:
            # Create sensor readings table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    time TIMESTAMPTZ NOT NULL,
                    sensor_id TEXT NOT NULL,
                    measurement_type TEXT NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    unit TEXT NOT NULL,
                    quality_flag TEXT DEFAULT 'good',
                    location GEOMETRY(POINT, 4326),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create hypertable if not exists
            await conn.execute("""
                SELECT create_hypertable('sensor_readings', 'time', 
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE);
            """)
            
            # Create indexes for efficient queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_id_time 
                ON sensor_readings (sensor_id, time DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sensor_readings_measurement_type_time 
                ON sensor_readings (measurement_type, time DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sensor_readings_location 
                ON sensor_readings USING GIST (location);
            """)
            
            # Create seismic events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS seismic_events (
                    time TIMESTAMPTZ NOT NULL,
                    event_id TEXT NOT NULL,
                    magnitude DOUBLE PRECISION,
                    depth DOUBLE PRECISION,
                    location GEOMETRY(POINT, 4326),
                    event_type TEXT,
                    confidence DOUBLE PRECISION,
                    waveform_data BYTEA,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            await conn.execute("""
                SELECT create_hypertable('seismic_events', 'time', 
                    chunk_time_interval => INTERVAL '1 week',
                    if_not_exists => TRUE);
            """)
            
            # Create volcanic monitoring table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS volcanic_monitoring (
                    time TIMESTAMPTZ NOT NULL,
                    volcano_id TEXT NOT NULL,
                    alert_level INTEGER,
                    deformation_rate DOUBLE PRECISION,
                    gas_emission_rate DOUBLE PRECISION,
                    thermal_anomaly DOUBLE PRECISION,
                    seismic_activity_count INTEGER,
                    risk_assessment JSONB,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            await conn.execute("""
                SELECT create_hypertable('volcanic_monitoring', 'time', 
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE);
            """)
            
            logger.info("Hypertables set up successfully")
        except Exception as e:
            logger.error("Failed to set up hypertables", error=str(e))
            raise

async def setup_compression_policies():
    """Set up compression policies for long-term data storage"""
    async with db_pool.acquire() as conn:
        try:
            # Compress sensor readings older than 7 days
            await conn.execute("""
                SELECT add_compression_policy('sensor_readings', INTERVAL '7 days', if_not_exists => TRUE);
            """)
            
            # Compress seismic events older than 30 days
            await conn.execute("""
                SELECT add_compression_policy('seismic_events', INTERVAL '30 days', if_not_exists => TRUE);
            """)
            
            # Compress volcanic monitoring data older than 14 days
            await conn.execute("""
                SELECT add_compression_policy('volcanic_monitoring', INTERVAL '14 days', if_not_exists => TRUE);
            """)
            
            logger.info("Compression policies set up successfully")
        except Exception as e:
            logger.error("Failed to set up compression policies", error=str(e))
            raise

async def setup_retention_policies():
    """Set up retention policies for automatic data cleanup"""
    async with db_pool.acquire() as conn:
        try:
            # Retain sensor readings for 5 years
            await conn.execute("""
                SELECT add_retention_policy('sensor_readings', INTERVAL '5 years', if_not_exists => TRUE);
            """)
            
            # Retain seismic events for 10 years
            await conn.execute("""
                SELECT add_retention_policy('seismic_events', INTERVAL '10 years', if_not_exists => TRUE);
            """)
            
            # Retain volcanic monitoring data for 7 years
            await conn.execute("""
                SELECT add_retention_policy('volcanic_monitoring', INTERVAL '7 years', if_not_exists => TRUE);
            """)
            
            logger.info("Retention policies set up successfully")
        except Exception as e:
            logger.error("Failed to set up retention policies", error=str(e))
            raise

async def setup_continuous_aggregates():
    """Set up continuous aggregates for real-time monitoring"""
    async with db_pool.acquire() as conn:
        try:
            # Hourly sensor readings aggregates
            await conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_hourly
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 hour', time) AS bucket,
                    sensor_id,
                    measurement_type,
                    AVG(value) AS avg_value,
                    MIN(value) AS min_value,
                    MAX(value) AS max_value,
                    STDDEV(value) AS stddev_value,
                    COUNT(*) AS reading_count
                FROM sensor_readings
                GROUP BY bucket, sensor_id, measurement_type
                WITH NO DATA;
            """)
            
            # Daily sensor readings aggregates
            await conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_daily
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 day', time) AS bucket,
                    sensor_id,
                    measurement_type,
                    AVG(value) AS avg_value,
                    MIN(value) AS min_value,
                    MAX(value) AS max_value,
                    STDDEV(value) AS stddev_value,
                    COUNT(*) AS reading_count
                FROM sensor_readings
                GROUP BY bucket, sensor_id, measurement_type
                WITH NO DATA;
            """)
            
            # Volcanic activity hourly aggregates
            await conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS volcanic_activity_hourly
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 hour', time) AS bucket,
                    volcano_id,
                    AVG(alert_level) AS avg_alert_level,
                    MAX(alert_level) AS max_alert_level,
                    AVG(deformation_rate) AS avg_deformation_rate,
                    AVG(gas_emission_rate) AS avg_gas_emission_rate,
                    AVG(thermal_anomaly) AS avg_thermal_anomaly,
                    SUM(seismic_activity_count) AS total_seismic_events
                FROM volcanic_monitoring
                GROUP BY bucket, volcano_id
                WITH NO DATA;
            """)
            
            # Set up refresh policies for continuous aggregates
            await conn.execute("""
                SELECT add_continuous_aggregate_policy('sensor_readings_hourly',
                    start_offset => INTERVAL '3 hours',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE);
            """)
            
            await conn.execute("""
                SELECT add_continuous_aggregate_policy('sensor_readings_daily',
                    start_offset => INTERVAL '3 days',
                    end_offset => INTERVAL '1 day',
                    schedule_interval => INTERVAL '1 day',
                    if_not_exists => TRUE);
            """)
            
            await conn.execute("""
                SELECT add_continuous_aggregate_policy('volcanic_activity_hourly',
                    start_offset => INTERVAL '3 hours',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE);
            """)
            
            logger.info("Continuous aggregates set up successfully")
        except Exception as e:
            logger.error("Failed to set up continuous aggregates", error=str(e))
            raise

@app.post("/sensor-readings/batch")
async def insert_sensor_readings_batch(readings: List[SensorReading]):
    """Insert batch of sensor readings"""
    async with db_pool.acquire() as conn:
        try:
            # Prepare data for batch insert
            data = []
            for reading in readings:
                location_wkt = None
                if reading.location_lat is not None and reading.location_lon is not None:
                    location_wkt = f"POINT({reading.location_lon} {reading.location_lat})"
                
                data.append((
                    reading.timestamp,
                    reading.sensor_id,
                    reading.measurement_type,
                    reading.value,
                    reading.unit,
                    reading.quality_flag,
                    location_wkt
                ))
            
            # Batch insert using COPY
            await conn.copy_records_to_table(
                'sensor_readings',
                records=data,
                columns=['time', 'sensor_id', 'measurement_type', 'value', 'unit', 'quality_flag', 'location']
            )
            
            logger.info(f"Inserted {len(readings)} sensor readings")
            return {"status": "success", "inserted_count": len(readings)}
            
        except Exception as e:
            logger.error("Failed to insert sensor readings", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/time-series")
async def query_time_series(query: TimeSeriesQuery):
    """Query time-series data with optional aggregation"""
    async with db_pool.acquire() as conn:
        try:
            # Build query based on parameters
            where_conditions = [
                "time >= $1",
                "time <= $2",
                "sensor_id = ANY($3)"
            ]
            params = [query.start_time, query.end_time, query.sensor_ids]
            param_count = 3
            
            if query.measurement_types:
                param_count += 1
                where_conditions.append(f"measurement_type = ANY(${param_count})")
                params.append(query.measurement_types)
            
            where_clause = " AND ".join(where_conditions)
            
            if query.aggregation:
                # Use continuous aggregates for better performance
                if query.interval in ['1h', '1 hour']:
                    table = "sensor_readings_hourly"
                    time_col = "bucket"
                    value_col = f"{query.aggregation}_value"
                elif query.interval in ['1d', '1 day']:
                    table = "sensor_readings_daily"
                    time_col = "bucket"
                    value_col = f"{query.aggregation}_value"
                else:
                    # Custom aggregation
                    table = "sensor_readings"
                    time_col = f"time_bucket('{query.interval}', time)"
                    value_col = f"{query.aggregation.upper()}(value)"
                
                sql = f"""
                    SELECT {time_col} as time, sensor_id, measurement_type, {value_col} as value
                    FROM {table}
                    WHERE {where_clause}
                    ORDER BY time DESC, sensor_id
                """
            else:
                # Raw data query
                sql = f"""
                    SELECT time, sensor_id, measurement_type, value, unit, quality_flag
                    FROM sensor_readings
                    WHERE {where_clause}
                    ORDER BY time DESC, sensor_id
                    LIMIT 10000
                """
            
            rows = await conn.fetch(sql, *params)
            
            # Convert to list of dictionaries
            results = [dict(row) for row in rows]
            
            return {
                "status": "success",
                "data": results,
                "count": len(results)
            }
            
        except Exception as e:
            logger.error("Failed to query time-series data", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))@app.post(
"/analysis/trend")
async def analyze_geological_trends(request: TrendAnalysisRequest):
    """Analyze geological trends using time-series analysis functions"""
    async with db_pool.acquire() as conn:
        try:
            # Calculate trend using linear regression
            trend_sql = """
                WITH trend_data AS (
                    SELECT 
                        time,
                        value,
                        EXTRACT(EPOCH FROM time) AS time_epoch
                    FROM sensor_readings
                    WHERE sensor_id = $1 
                        AND measurement_type = $2
                        AND time >= $3 
                        AND time <= $4
                        AND quality_flag = 'good'
                    ORDER BY time
                ),
                regression AS (
                    SELECT 
                        regr_slope(value, time_epoch) AS slope,
                        regr_intercept(value, time_epoch) AS intercept,
                        regr_r2(value, time_epoch) AS r_squared,
                        COUNT(*) AS data_points,
                        AVG(value) AS mean_value,
                        STDDEV(value) AS stddev_value
                    FROM trend_data
                )
                SELECT * FROM regression;
            """
            
            trend_result = await conn.fetchrow(
                trend_sql, 
                request.sensor_id, 
                request.measurement_type,
                request.start_time, 
                request.end_time
            )
            
            # Calculate moving averages and anomalies
            anomaly_sql = """
                WITH windowed_data AS (
                    SELECT 
                        time,
                        value,
                        AVG(value) OVER (
                            ORDER BY time 
                            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                        ) AS moving_avg_24h,
                        STDDEV(value) OVER (
                            ORDER BY time 
                            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                        ) AS moving_stddev_24h
                    FROM sensor_readings
                    WHERE sensor_id = $1 
                        AND measurement_type = $2
                        AND time >= $3 
                        AND time <= $4
                        AND quality_flag = 'good'
                    ORDER BY time
                ),
                anomalies AS (
                    SELECT 
                        time,
                        value,
                        moving_avg_24h,
                        moving_stddev_24h,
                        ABS(value - moving_avg_24h) / NULLIF(moving_stddev_24h, 0) AS z_score,
                        CASE 
                            WHEN ABS(value - moving_avg_24h) / NULLIF(moving_stddev_24h, 0) > 3 
                            THEN 'anomaly'
                            ELSE 'normal'
                        END AS anomaly_flag
                    FROM windowed_data
                    WHERE moving_stddev_24h IS NOT NULL AND moving_stddev_24h > 0
                )
                SELECT 
                    COUNT(*) FILTER (WHERE anomaly_flag = 'anomaly') AS anomaly_count,
                    COUNT(*) AS total_points,
                    MAX(z_score) AS max_z_score,
                    AVG(z_score) AS avg_z_score
                FROM anomalies;
            """
            
            anomaly_result = await conn.fetchrow(
                anomaly_sql,
                request.sensor_id,
                request.measurement_type,
                request.start_time,
                request.end_time
            )
            
            # Calculate seasonal decomposition components
            seasonal_sql = """
                WITH hourly_data AS (
                    SELECT 
                        time_bucket('1 hour', time) AS hour_bucket,
                        AVG(value) AS avg_value
                    FROM sensor_readings
                    WHERE sensor_id = $1 
                        AND measurement_type = $2
                        AND time >= $3 
                        AND time <= $4
                        AND quality_flag = 'good'
                    GROUP BY hour_bucket
                    ORDER BY hour_bucket
                ),
                seasonal_components AS (
                    SELECT 
                        hour_bucket,
                        avg_value,
                        AVG(avg_value) OVER () AS overall_mean,
                        AVG(avg_value) OVER (
                            ORDER BY hour_bucket 
                            ROWS BETWEEN 11 PRECEDING AND 12 FOLLOWING
                        ) AS trend_component,
                        avg_value - AVG(avg_value) OVER (
                            ORDER BY hour_bucket 
                            ROWS BETWEEN 11 PRECEDING AND 12 FOLLOWING
                        ) AS detrended
                    FROM hourly_data
                )
                SELECT 
                    STDDEV(trend_component) AS trend_strength,
                    STDDEV(detrended) AS seasonal_strength,
                    STDDEV(avg_value) AS total_variance
                FROM seasonal_components;
            """
            
            seasonal_result = await conn.fetchrow(
                seasonal_sql,
                request.sensor_id,
                request.measurement_type,
                request.start_time,
                request.end_time
            )
            
            # Compile comprehensive trend analysis
            analysis_result = {
                "sensor_id": request.sensor_id,
                "measurement_type": request.measurement_type,
                "analysis_period": {
                    "start": request.start_time.isoformat(),
                    "end": request.end_time.isoformat()
                },
                "trend_analysis": {
                    "slope": float(trend_result['slope']) if trend_result['slope'] else 0,
                    "intercept": float(trend_result['intercept']) if trend_result['intercept'] else 0,
                    "r_squared": float(trend_result['r_squared']) if trend_result['r_squared'] else 0,
                    "data_points": int(trend_result['data_points']) if trend_result['data_points'] else 0,
                    "mean_value": float(trend_result['mean_value']) if trend_result['mean_value'] else 0,
                    "standard_deviation": float(trend_result['stddev_value']) if trend_result['stddev_value'] else 0
                },
                "anomaly_detection": {
                    "anomaly_count": int(anomaly_result['anomaly_count']) if anomaly_result['anomaly_count'] else 0,
                    "total_points": int(anomaly_result['total_points']) if anomaly_result['total_points'] else 0,
                    "anomaly_rate": float(anomaly_result['anomaly_count'] / max(anomaly_result['total_points'], 1)) if anomaly_result['anomaly_count'] else 0,
                    "max_z_score": float(anomaly_result['max_z_score']) if anomaly_result['max_z_score'] else 0,
                    "avg_z_score": float(anomaly_result['avg_z_score']) if anomaly_result['avg_z_score'] else 0
                },
                "seasonal_analysis": {
                    "trend_strength": float(seasonal_result['trend_strength']) if seasonal_result['trend_strength'] else 0,
                    "seasonal_strength": float(seasonal_result['seasonal_strength']) if seasonal_result['seasonal_strength'] else 0,
                    "total_variance": float(seasonal_result['total_variance']) if seasonal_result['total_variance'] else 0
                }
            }
            
            # Geological interpretation
            interpretation = interpret_geological_trends(analysis_result)
            analysis_result["geological_interpretation"] = interpretation
            
            return analysis_result
            
        except Exception as e:
            logger.error("Failed to analyze geological trends", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

def interpret_geological_trends(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Interpret geological trends based on statistical analysis"""
    interpretation = {
        "trend_significance": "none",
        "anomaly_level": "normal",
        "geological_implications": [],
        "recommendations": []
    }
    
    # Interpret trend significance
    r_squared = analysis["trend_analysis"]["r_squared"]
    slope = analysis["trend_analysis"]["slope"]
    
    if r_squared > 0.7:
        if abs(slope) > 0.001:  # Threshold depends on measurement type
            interpretation["trend_significance"] = "strong"
            if slope > 0:
                interpretation["geological_implications"].append("Increasing trend detected - possible volcanic unrest")
            else:
                interpretation["geological_implications"].append("Decreasing trend detected - possible system stabilization")
        else:
            interpretation["trend_significance"] = "stable"
    elif r_squared > 0.3:
        interpretation["trend_significance"] = "moderate"
    
    # Interpret anomaly level
    anomaly_rate = analysis["anomaly_detection"]["anomaly_rate"]
    max_z_score = analysis["anomaly_detection"]["max_z_score"]
    
    if anomaly_rate > 0.1 or max_z_score > 5:
        interpretation["anomaly_level"] = "high"
        interpretation["geological_implications"].append("High anomaly rate - requires immediate investigation")
        interpretation["recommendations"].append("Increase monitoring frequency")
    elif anomaly_rate > 0.05 or max_z_score > 3:
        interpretation["anomaly_level"] = "moderate"
        interpretation["geological_implications"].append("Moderate anomalies detected - enhanced monitoring recommended")
    
    # Seasonal analysis interpretation
    trend_strength = analysis["seasonal_analysis"]["trend_strength"]
    seasonal_strength = analysis["seasonal_analysis"]["seasonal_strength"]
    
    if seasonal_strength > trend_strength * 2:
        interpretation["geological_implications"].append("Strong seasonal patterns - consider environmental factors")
    
    return interpretation

@app.get("/statistics/database")
async def get_database_statistics():
    """Get database statistics and performance metrics"""
    async with db_pool.acquire() as conn:
        try:
            # Hypertable statistics
            hypertable_stats = await conn.fetch("""
                SELECT 
                    hypertable_name,
                    num_chunks,
                    table_bytes,
                    index_bytes,
                    toast_bytes,
                    total_bytes
                FROM timescaledb_information.hypertables h
                JOIN timescaledb_information.hypertable_detailed_size(h.hypertable_name) s
                ON h.hypertable_name = s.table_name;
            """)
            
            # Compression statistics
            compression_stats = await conn.fetch("""
                SELECT 
                    hypertable_name,
                    compression_status,
                    uncompressed_heap_bytes,
                    uncompressed_index_bytes,
                    uncompressed_toast_bytes,
                    uncompressed_total_bytes,
                    compressed_heap_bytes,
                    compressed_index_bytes,
                    compressed_toast_bytes,
                    compressed_total_bytes
                FROM timescaledb_information.hypertables h
                JOIN timescaledb_information.compression_settings cs
                ON h.hypertable_name = cs.hypertable_name;
            """)
            
            # Recent data ingestion rates
            ingestion_stats = await conn.fetch("""
                SELECT 
                    'sensor_readings' as table_name,
                    COUNT(*) as recent_inserts,
                    MIN(time) as oldest_recent,
                    MAX(time) as newest_recent
                FROM sensor_readings 
                WHERE time >= NOW() - INTERVAL '1 hour'
                UNION ALL
                SELECT 
                    'seismic_events' as table_name,
                    COUNT(*) as recent_inserts,
                    MIN(time) as oldest_recent,
                    MAX(time) as newest_recent
                FROM seismic_events 
                WHERE time >= NOW() - INTERVAL '1 hour'
                UNION ALL
                SELECT 
                    'volcanic_monitoring' as table_name,
                    COUNT(*) as recent_inserts,
                    MIN(time) as oldest_recent,
                    MAX(time) as newest_recent
                FROM volcanic_monitoring 
                WHERE time >= NOW() - INTERVAL '1 hour';
            """)
            
            return {
                "hypertable_statistics": [dict(row) for row in hypertable_stats],
                "compression_statistics": [dict(row) for row in compression_stats],
                "ingestion_statistics": [dict(row) for row in ingestion_stats],
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error("Failed to get database statistics", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/maintenance/compress")
async def manual_compression():
    """Manually trigger compression for eligible chunks"""
    async with db_pool.acquire() as conn:
        try:
            # Compress eligible chunks
            compression_results = await conn.fetch("""
                SELECT compress_chunk(chunk_name) as compressed_chunk
                FROM timescaledb_information.chunks 
                WHERE hypertable_name IN ('sensor_readings', 'seismic_events', 'volcanic_monitoring')
                AND is_compressed = false
                AND range_end < NOW() - INTERVAL '1 day';
            """)
            
            compressed_chunks = [row['compressed_chunk'] for row in compression_results]
            
            return {
                "status": "success",
                "compressed_chunks": compressed_chunks,
                "count": len(compressed_chunks)
            }
            
        except Exception as e:
            logger.error("Failed to perform manual compression", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))