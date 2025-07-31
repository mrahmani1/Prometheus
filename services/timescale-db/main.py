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
        
class AdvancedGeologicalDatabase:
    """Advanced database operations for geological data with spatial indexing and time-series optimization"""
    
    def __init__(self, connection_params: Dict[str, str]):
        self.postgres_params = connection_params.get('postgres', {})
        self.influx_params = connection_params.get('influxdb', {})
        self.neo4j_params = connection_params.get('neo4j', {})
        self.redis_params = connection_params.get('redis', {})
        
        self.connections = {}
        self._initialize_connections()
        
    def _initialize_connections(self):
        """Initialize database connections with connection pooling"""
        import psycopg2.pool
        import influxdb_client
        from neo4j import GraphDatabase
        import redis
        
        # PostgreSQL connection pool
        if self.postgres_params:
            self.connections['postgres'] = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=20, **self.postgres_params
            )
        
        # InfluxDB client
        if self.influx_params:
            self.connections['influxdb'] = influxdb_client.InfluxDBClient(
                url=self.influx_params['url'],
                token=self.influx_params['token'],
                org=self.influx_params['org']
            )
        
        # Neo4j driver
        if self.neo4j_params:
            self.connections['neo4j'] = GraphDatabase.driver(
                self.neo4j_params['uri'],
                auth=(self.neo4j_params['user'], self.neo4j_params['password'])
            )
        
        # Redis connection
        if self.redis_params:
            self.connections['redis'] = redis.Redis(**self.redis_params)
    
    def store_seismic_data(self, seismic_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store seismic data with spatial and temporal indexing"""
        
        stored_count = 0
        errors = []
        
        # Get PostgreSQL connection
        postgres_conn = self.connections['postgres'].getconn()
        
        try:
            with postgres_conn.cursor() as cursor:
                # Create spatial index if not exists
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_seismic_location 
                    ON seismic_events USING GIST (location);
                    
                    CREATE INDEX IF NOT EXISTS idx_seismic_time 
                    ON seismic_events (event_time);
                    
                    CREATE INDEX IF NOT EXISTS idx_seismic_magnitude 
                    ON seismic_events (magnitude);
                """)
                
                # Prepare batch insert
                insert_query = """
                    INSERT INTO seismic_events 
                    (event_id, event_time, location, magnitude, depth, event_type, 
                     waveform_data, frequency_content, uncertainty_metrics)
                    VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO UPDATE SET
                        magnitude = EXCLUDED.magnitude,
                        depth = EXCLUDED.depth,
                        event_type = EXCLUDED.event_type,
                        waveform_data = EXCLUDED.waveform_data,
                        frequency_content = EXCLUDED.frequency_content,
                        uncertainty_metrics = EXCLUDED.uncertainty_metrics;
                """
                
                # Process events in batches
                batch_size = 1000
                for i in range(0, len(seismic_events), batch_size):
                    batch = seismic_events[i:i + batch_size]
                    batch_data = []
                    
                    for event in batch:
                        try:
                            # Convert location to WKT format
                            location_wkt = f"POINT({event['longitude']} {event['latitude']})"
                            
                            batch_data.append((
                                event['event_id'],
                                event['event_time'],
                                location_wkt,
                                event['magnitude'],
                                event['depth'],
                                event['event_type'],
                                json.dumps(event.get('waveform_data', {})),
                                json.dumps(event.get('frequency_content', {})),
                                json.dumps(event.get('uncertainty_metrics', {}))
                            ))
                            
                        except Exception as e:
                            errors.append(f"Event {event.get('event_id', 'unknown')}: {str(e)}")
                    
                    # Execute batch insert
                    cursor.executemany(insert_query, batch_data)
                    stored_count += len(batch_data)
                
                postgres_conn.commit()
                
        except Exception as e:
            postgres_conn.rollback()
            errors.append(f"Database error: {str(e)}")
        
        finally:
            self.connections['postgres'].putconn(postgres_conn)
        
        # Store time-series data in InfluxDB
        if 'influxdb' in self.connections:
            self._store_timeseries_data(seismic_events)
        
        return {
            'stored_count': stored_count,
            'total_events': len(seismic_events),
            'errors': errors,
            'success_rate': stored_count / len(seismic_events) if seismic_events else 0
        }
    
    def _store_timeseries_data(self, seismic_events: List[Dict[str, Any]]):
        """Store time-series data in InfluxDB"""
        from influxdb_client.client.write_api import SYNCHRONOUS
        
        write_api = self.connections['influxdb'].write_api(write_options=SYNCHRONOUS)
        
        points = []
        for event in seismic_events:
            # Create point for magnitude time series
            point = {
                "measurement": "seismic_magnitude",
                "tags": {
                    "event_type": event.get('event_type', 'unknown'),
                    "region": event.get('region', 'unknown'),
                    "station": event.get('station', 'unknown')
                },
                "fields": {
                    "magnitude": float(event.get('magnitude', 0)),
                    "depth": float(event.get('depth', 0)),
                    "latitude": float(event.get('latitude', 0)),
                    "longitude": float(event.get('longitude', 0))
                },
                "time": event.get('event_time')
            }
            points.append(point)
            
            # Store waveform data if available
            if 'waveform_data' in event:
                waveform_point = {
                    "measurement": "seismic_waveform",
                    "tags": {
                        "event_id": event.get('event_id'),
                        "component": "combined"
                    },
                    "fields": {
                        "amplitude": float(np.max(np.abs(event['waveform_data']))),
                        "rms": float(np.sqrt(np.mean(np.square(event['waveform_data'])))),
                        "peak_frequency": float(event.get('frequency_content', {}).get('dominant_frequency', 0))
                    },
                    "time": event.get('event_time')
                }
                points.append(waveform_point)
        
        try:
            write_api.write(bucket="geological_data", record=points)
        except Exception as e:
            logger.error(f"InfluxDB write error: {e}")
    
    def query_spatial_events(self, bbox: Tuple[float, float, float, float],
                           time_range: Tuple[str, str],
                           magnitude_range: Tuple[float, float] = (0, 10)) -> List[Dict[str, Any]]:
        """Query seismic events within spatial and temporal bounds"""
        
        min_lon, min_lat, max_lon, max_lat = bbox
        start_time, end_time = time_range
        min_mag, max_mag = magnitude_range
        
        postgres_conn = self.connections['postgres'].getconn()
        
        try:
            with postgres_conn.cursor() as cursor:
                query = """
                    SELECT event_id, event_time, ST_X(location) as longitude, 
                           ST_Y(location) as latitude, magnitude, depth, event_type,
                           waveform_data, frequency_content, uncertainty_metrics
                    FROM seismic_events
                    WHERE location && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
                      AND event_time BETWEEN %s AND %s
                      AND magnitude BETWEEN %s AND %s
                    ORDER BY event_time DESC
                    LIMIT 10000;
                """
                
                cursor.execute(query, (min_lon, min_lat, max_lon, max_lat, 
                                     start_time, end_time, min_mag, max_mag))
                
                results = []
                for row in cursor.fetchall():
                    event = {
                        'event_id': row[0],
                        'event_time': row[1],
                        'longitude': row[2],
                        'latitude': row[3],
                        'magnitude': row[4],
                        'depth': row[5],
                        'event_type': row[6],
                        'waveform_data': json.loads(row[7]) if row[7] else {},
                        'frequency_content': json.loads(row[8]) if row[8] else {},
                        'uncertainty_metrics': json.loads(row[9]) if row[9] else {}
                    }
                    results.append(event)
                
                return results
                
        except Exception as e:
            logger.error(f"Spatial query error: {e}")
            return []
        
        finally:
            self.connections['postgres'].putconn(postgres_conn)
    
    def create_geological_relationships(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create geological process relationships in Neo4j graph database"""
        
        if 'neo4j' not in self.connections:
            return {'error': 'Neo4j connection not available'}
        
        created_count = 0
        errors = []
        
        with self.connections['neo4j'].session() as session:
            try:
                # Create nodes and relationships
                for rel in relationships:
                    try:
                        # Create or update source node
                        session.run("""
                            MERGE (source:GeologicalProcess {id: $source_id})
                            SET source.name = $source_name,
                                source.type = $source_type,
                                source.properties = $source_props
                        """, 
                        source_id=rel['source_id'],
                        source_name=rel['source_name'],
                        source_type=rel['source_type'],
                        source_props=json.dumps(rel.get('source_properties', {})))
                        
                        # Create or update target node
                        session.run("""
                            MERGE (target:GeologicalProcess {id: $target_id})
                            SET target.name = $target_name,
                                target.type = $target_type,
                                target.properties = $target_props
                        """,
                        target_id=rel['target_id'],
                        target_name=rel['target_name'],
                        target_type=rel['target_type'],
                        target_props=json.dumps(rel.get('target_properties', {})))
                        
                        # Create relationship
                        session.run("""
                            MATCH (source:GeologicalProcess {id: $source_id})
                            MATCH (target:GeologicalProcess {id: $target_id})
                            MERGE (source)-[r:INFLUENCES {type: $rel_type}]->(target)
                            SET r.strength = $strength,
                                r.confidence = $confidence,
                                r.temporal_lag = $temporal_lag,
                                r.properties = $rel_props
                        """,
                        source_id=rel['source_id'],
                        target_id=rel['target_id'],
                        rel_type=rel['relationship_type'],
                        strength=rel.get('strength', 1.0),
                        confidence=rel.get('confidence', 0.5),
                        temporal_lag=rel.get('temporal_lag', 0),
                        rel_props=json.dumps(rel.get('relationship_properties', {})))
                        
                        created_count += 1
                        
                    except Exception as e:
                        errors.append(f"Relationship {rel.get('source_id', 'unknown')} -> {rel.get('target_id', 'unknown')}: {str(e)}")
                
            except Exception as e:
                errors.append(f"Neo4j session error: {str(e)}")
        
        return {
            'created_count': created_count,
            'total_relationships': len(relationships),
            'errors': errors,
            'success_rate': created_count / len(relationships) if relationships else 0
        }
    
    def cache_analysis_results(self, analysis_id: str, results: Dict[str, Any], 
                             expiration_hours: int = 24) -> bool:
        """Cache analysis results in Redis with expiration"""
        
        if 'redis' not in self.connections:
            return False
        
        try:
            redis_client = self.connections['redis']
            
            # Store results with expiration
            cache_key = f"analysis:{analysis_id}"
            redis_client.setex(
                cache_key,
                expiration_hours * 3600,
                json.dumps(results, default=str)
            )
            
            # Store metadata
            metadata_key = f"analysis_meta:{analysis_id}"
            metadata = {
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=expiration_hours)).isoformat(),
                'size_bytes': len(json.dumps(results, default=str))
            }
            redis_client.setex(
                metadata_key,
                expiration_hours * 3600,
                json.dumps(metadata)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            return False
    
    def get_cached_results(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis results"""
        
        if 'redis' not in self.connections:
            return None
        
        try:
            redis_client = self.connections['redis']
            cache_key = f"analysis:{analysis_id}"
            
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Redis retrieval error: {e}")
            return None

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
        
        
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime, timedelta
import uuid

class AdvancedGeologicalAPI:
    """Advanced API endpoints for geological data and analysis"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Prometheus Geological Analysis API",
            description="Advanced geological simulation and monitoring API",
            version="2.0.0"
        )
        self.security = HTTPBearer()
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_middleware(self):
        """Setup API middleware for logging, CORS, rate limiting"""
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/v2/seismic/analyze", response_model=SeismicAnalysisResponse)
        async def analyze_seismic_data(
            request: SeismicAnalysisRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Advanced seismic data analysis endpoint"""
            
            # Validate authentication
            user_id = await self._validate_token(credentials.credentials)
            
            # Generate analysis ID
            analysis_id = str(uuid.uuid4())
            
            try:
                # Initialize seismic processor
                processor = AdvancedSeismicProcessor(sampling_rate=request.sampling_rate)
                
                # Process seismic data
                analysis_result = processor.process_seismic_stream(
                    data=np.array(request.waveform_data),
                    station_info=request.station_info.dict()
                )
                
                # Add metadata
                analysis_result.update({
                    'analysis_id': analysis_id,
                    'user_id': user_id,
                    'request_timestamp': datetime.now().isoformat(),
                    'processing_time_ms': 0  # Would be calculated
                })
                
                # Store results asynchronously
                background_tasks.add_task(
                    self._store_analysis_results,
                    analysis_id, analysis_result, 'seismic'
                )
                
                return SeismicAnalysisResponse(**analysis_result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        @self.app.post("/api/v2/volcanic/monitor", response_model=VolcanicMonitoringResponse)
        async def monitor_volcanic_activity(
            request: VolcanicMonitoringRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Comprehensive volcanic monitoring endpoint"""
            
            user_id = await self._validate_token(credentials.credentials)
            monitoring_id = str(uuid.uuid4())
            
            try:
                # Initialize volcanic monitor
                monitor = AdvancedVolcanicMonitor()
                
                # Perform comprehensive assessment
                assessment = monitor.comprehensive_volcanic_assessment(
                    monitoring_data=request.monitoring_data,
                    volcano_info=request.volcano_info.dict()
                )
                
                # Add metadata
                assessment.update({
                    'monitoring_id': monitoring_id,
                    'user_id': user_id,
                    'request_timestamp': datetime.now().isoformat()
                })
                
                # Store results and trigger alerts if necessary
                background_tasks.add_task(
                    self._process_volcanic_assessment,
                    monitoring_id, assessment
                )
                
                return VolcanicMonitoringResponse(**assessment)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Monitoring failed: {str(e)}")
        
        @self.app.post("/api/v2/geochemical/analyze", response_model=GeochemicalAnalysisResponse)
        async def analyze_geochemical_composition(
            request: GeochemicalAnalysisRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Advanced geochemical composition analysis"""
            
            user_id = await self._validate_token(credentials.credentials)
            analysis_id = str(uuid.uuid4())
            
            try:
                # Initialize geochemical analyzer
                analyzer = AdvancedGeochemicalAnalyzer()
                
                # Perform analysis
                analysis_result = analyzer.analyze_rock_composition(
                    composition=request.composition,
                    temperature=request.temperature,
                    pressure=request.pressure
                )
                
                # Add metadata
                analysis_result.update({
                    'analysis_id': analysis_id,
                    'user_id': user_id,
                    'sample_id': request.sample_id,
                    'request_timestamp': datetime.now().isoformat()
                })
                
                # Store results
                background_tasks.add_task(
                    self._store_analysis_results,
                    analysis_id, analysis_result, 'geochemical'
                )
                
                return GeochemicalAnalysisResponse(**analysis_result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Geochemical analysis failed: {str(e)}")
        
        @self.app.get("/api/v2/analysis/{analysis_id}", response_model=AnalysisStatusResponse)
        async def get_analysis_status(
            analysis_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get analysis status and results"""
            
            user_id = await self._validate_token(credentials.credentials)
            
            try:
                # Retrieve analysis from database/cache
                analysis_data = await self._get_analysis_results(analysis_id, user_id)
                
                if not analysis_data:
                    raise HTTPException(status_code=404, detail="Analysis not found")
                
                return AnalysisStatusResponse(**analysis_data)
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis: {str(e)}")
        
        @self.app.get("/api/v2/spatial/query", response_model=SpatialQueryResponse)
        async def spatial_geological_query(
            bbox: str = Query(..., description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
            start_time: str = Query(..., description="Start time (ISO format)"),
            end_time: str = Query(..., description="End time (ISO format)"),
            data_types: List[str] = Query(["seismic"], description="Data types to include"),
            min_magnitude: float = Query(0.0, description="Minimum magnitude"),
            max_magnitude: float = Query(10.0, description="Maximum magnitude"),
            limit: int = Query(1000, le=10000, description="Maximum results"),
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Spatial query for geological data"""
            
            user_id = await self._validate_token(credentials.credentials)
            
            try:
                # Parse bounding box
                bbox_coords = [float(x) for x in bbox.split(',')]
                if len(bbox_coords) != 4:
                    raise ValueError("Invalid bounding box format")
                
                # Initialize database
                db = AdvancedGeologicalDatabase({
                    'postgres': {'host': 'localhost', 'database': 'geological_data'},
                    'influxdb': {'url': 'http://localhost:8086'}
                })
                
                # Perform spatial query
                results = db.query_spatial_events(
                    bbox=tuple(bbox_coords),
                    time_range=(start_time, end_time),
                    magnitude_range=(min_magnitude, max_magnitude)
                )
                
                # Limit results
                results = results[:limit]
                
                return SpatialQueryResponse(
                    query_id=str(uuid.uuid4()),
                    user_id=user_id,
                    bbox=bbox_coords,
                    time_range=[start_time, end_time],
                    result_count=len(results),
                    results=results,
                    query_timestamp=datetime.now().isoformat()
                )
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Spatial query failed: {str(e)}")
    
    async def _validate_token(self, token: str) -> str:
        """Validate JWT token and return user ID"""
        # Simplified token validation - implement proper JWT validation
        if not token or len(token) < 10:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Extract user ID from token (simplified)
        return f"user_{hash(token) % 10000}"
    
    async def _store_analysis_results(self, analysis_id: str, results: Dict[str, Any], analysis_type: str):
        """Store analysis results in database"""
        try:
            # Initialize database connection
            db = AdvancedGeologicalDatabase({
                'redis': {'host': 'localhost', 'port': 6379}
            })
            
            # Cache results
            db.cache_analysis_results(analysis_id, results, expiration_hours=24)
            
            logger.info(f"Stored {analysis_type} analysis results for {analysis_id}")
            
        except Exception as e:
            logger.error(f"Failed to store analysis results: {e}")
    
    async def _get_analysis_results(self, analysis_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis results from database"""
        try:
            db = AdvancedGeologicalDatabase({
                'redis': {'host': 'localhost', 'port': 6379}
            })
            
            results = db.get_cached_results(analysis_id)
            
            # Verify user has access to this analysis
            if results and results.get('user_id') == user_id:
                return results
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve analysis results: {e}")
            return None

# Pydantic models for request/response validation
class StationInfo(BaseModel):
    station_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    elevation_m: float
    network: str

class SeismicAnalysisRequest(BaseModel):
    waveform_data: List[List[float]]
    sampling_rate: float = Field(..., gt=0, le=1000)
    station_info: StationInfo
    analysis_parameters: Optional[Dict[str, Any]] = {}

class SeismicAnalysisResponse(BaseModel):
    analysis_id: str
    station_id: str
    processing_timestamp: str
    components: List[Dict[str, Any]]
    cross_component_analysis: Dict[str, Any]
    overall_quality: float

class VolcanicMonitoringRequest(BaseModel):
    volcano_info: Dict[str, Any]
    monitoring_data: Dict[str, Any]
    analysis_parameters: Optional[Dict[str, Any]] = {}

class VolcanicMonitoringResponse(BaseModel):
    monitoring_id: str
    volcano_id: str
    assessment_timestamp: str
    current_alert_level: str
    eruption_probability: Dict[str, float]
    recommendations: List[str]

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
            
class AdvancedGeologicalDatabase:
    """Advanced database operations for geological data with spatial indexing and time-series optimization"""
    
    def __init__(self, connection_params: Dict[str, str]):
        self.postgres_params = connection_params.get('postgres', {})
        self.influx_params = connection_params.get('influxdb', {})
        self.neo4j_params = connection_params.get('neo4j', {})
        self.redis_params = connection_params.get('redis', {})
        
        self.connections = {}
        self._initialize_connections()
        
    def _initialize_connections(self):
        """Initialize database connections with connection pooling"""
        import psycopg2.pool
        import influxdb_client
        from neo4j import GraphDatabase
        import redis
        
        # PostgreSQL connection pool
        if self.postgres_params:
            self.connections['postgres'] = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=20, **self.postgres_params
            )
        
        # InfluxDB client
        if self.influx_params:
            self.connections['influxdb'] = influxdb_client.InfluxDBClient(
                url=self.influx_params['url'],
                token=self.influx_params['token'],
                org=self.influx_params['org']
            )
        
        # Neo4j driver
        if self.neo4j_params:
            self.connections['neo4j'] = GraphDatabase.driver(
                self.neo4j_params['uri'],
                auth=(self.neo4j_params['user'], self.neo4j_params['password'])
            )
        
        # Redis connection
        if self.redis_params:
            self.connections['redis'] = redis.Redis(**self.redis_params)
    
    def store_seismic_data(self, seismic_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store seismic data with spatial and temporal indexing"""
        
        stored_count = 0
        errors = []
        
        # Get PostgreSQL connection
        postgres_conn = self.connections['postgres'].getconn()
        
        try:
            with postgres_conn.cursor() as cursor:
                # Create spatial index if not exists
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_seismic_location 
                    ON seismic_events USING GIST (location);
                    
                    CREATE INDEX IF NOT EXISTS idx_seismic_time 
                    ON seismic_events (event_time);
                    
                    CREATE INDEX IF NOT EXISTS idx_seismic_magnitude 
                    ON seismic_events (magnitude);
                """)
                
                # Prepare batch insert
                insert_query = """
                    INSERT INTO seismic_events 
                    (event_id, event_time, location, magnitude, depth, event_type, 
                     waveform_data, frequency_content, uncertainty_metrics)
                    VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO UPDATE SET
                        magnitude = EXCLUDED.magnitude,
                        depth = EXCLUDED.depth,
                        event_type = EXCLUDED.event_type,
                        waveform_data = EXCLUDED.waveform_data,
                        frequency_content = EXCLUDED.frequency_content,
                        uncertainty_metrics = EXCLUDED.uncertainty_metrics;
                """
                
                # Process events in batches
                batch_size = 1000
                for i in range(0, len(seismic_events), batch_size):
                    batch = seismic_events[i:i + batch_size]
                    batch_data = []
                    
                    for event in batch:
                        try:
                            # Convert location to WKT format
                            location_wkt = f"POINT({event['longitude']} {event['latitude']})"
                            
                            batch_data.append((
                                event['event_id'],
                                event['event_time'],
                                location_wkt,
                                event['magnitude'],
                                event['depth'],
                                event['event_type'],
                                json.dumps(event.get('waveform_data', {})),
                                json.dumps(event.get('frequency_content', {})),
                                json.dumps(event.get('uncertainty_metrics', {}))
                            ))
                            
                        except Exception as e:
                            errors.append(f"Event {event.get('event_id', 'unknown')}: {str(e)}")
                    
                    # Execute batch insert
                    cursor.executemany(insert_query, batch_data)
                    stored_count += len(batch_data)
                
                postgres_conn.commit()
                
        except Exception as e:
            postgres_conn.rollback()
            errors.append(f"Database error: {str(e)}")
        
        finally:
            self.connections['postgres'].putconn(postgres_conn)
        
        # Store time-series data in InfluxDB
        if 'influxdb' in self.connections:
            self._store_timeseries_data(seismic_events)
        
        return {
            'stored_count': stored_count,
            'total_events': len(seismic_events),
            'errors': errors,
            'success_rate': stored_count / len(seismic_events) if seismic_events else 0
        }
    
    def _store_timeseries_data(self, seismic_events: List[Dict[str, Any]]):
        """Store time-series data in InfluxDB"""
        from influxdb_client.client.write_api import SYNCHRONOUS
        
        write_api = self.connections['influxdb'].write_api(write_options=SYNCHRONOUS)
        
        points = []
        for event in seismic_events:
            # Create point for magnitude time series
            point = {
                "measurement": "seismic_magnitude",
                "tags": {
                    "event_type": event.get('event_type', 'unknown'),
                    "region": event.get('region', 'unknown'),
                    "station": event.get('station', 'unknown')
                },
                "fields": {
                    "magnitude": float(event.get('magnitude', 0)),
                    "depth": float(event.get('depth', 0)),
                    "latitude": float(event.get('latitude', 0)),
                    "longitude": float(event.get('longitude', 0))
                },
                "time": event.get('event_time')
            }
            points.append(point)
            
            # Store waveform data if available
            if 'waveform_data' in event:
                waveform_point = {
                    "measurement": "seismic_waveform",
                    "tags": {
                        "event_id": event.get('event_id'),
                        "component": "combined"
                    },
                    "fields": {
                        "amplitude": float(np.max(np.abs(event['waveform_data']))),
                        "rms": float(np.sqrt(np.mean(np.square(event['waveform_data'])))),
                        "peak_frequency": float(event.get('frequency_content', {}).get('dominant_frequency', 0))
                    },
                    "time": event.get('event_time')
                }
                points.append(waveform_point)
        
        try:
            write_api.write(bucket="geological_data", record=points)
        except Exception as e:
            logger.error(f"InfluxDB write error: {e}")
    
    def query_spatial_events(self, bbox: Tuple[float, float, float, float],
                           time_range: Tuple[str, str],
                           magnitude_range: Tuple[float, float] = (0, 10)) -> List[Dict[str, Any]]:
        """Query seismic events within spatial and temporal bounds"""
        
        min_lon, min_lat, max_lon, max_lat = bbox
        start_time, end_time = time_range
        min_mag, max_mag = magnitude_range
        
        postgres_conn = self.connections['postgres'].getconn()
        
        try:
            with postgres_conn.cursor() as cursor:
                query = """
                    SELECT event_id, event_time, ST_X(location) as longitude, 
                           ST_Y(location) as latitude, magnitude, depth, event_type,
                           waveform_data, frequency_content, uncertainty_metrics
                    FROM seismic_events
                    WHERE location && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
                      AND event_time BETWEEN %s AND %s
                      AND magnitude BETWEEN %s AND %s
                    ORDER BY event_time DESC
                    LIMIT 10000;
                """
                
                cursor.execute(query, (min_lon, min_lat, max_lon, max_lat, 
                                     start_time, end_time, min_mag, max_mag))
                
                results = []
                for row in cursor.fetchall():
                    event = {
                        'event_id': row[0],
                        'event_time': row[1],
                        'longitude': row[2],
                        'latitude': row[3],
                        'magnitude': row[4],
                        'depth': row[5],
                        'event_type': row[6],
                        'waveform_data': json.loads(row[7]) if row[7] else {},
                        'frequency_content': json.loads(row[8]) if row[8] else {},
                        'uncertainty_metrics': json.loads(row[9]) if row[9] else {}
                    }
                    results.append(event)
                
                return results
                
        except Exception as e:
            logger.error(f"Spatial query error: {e}")
            return []
        
        finally:
            self.connections['postgres'].putconn(postgres_conn)
    
    def create_geological_relationships(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create geological process relationships in Neo4j graph database"""
        
        if 'neo4j' not in self.connections:
            return {'error': 'Neo4j connection not available'}
        
        created_count = 0
        errors = []
        
        with self.connections['neo4j'].session() as session:
            try:
                # Create nodes and relationships
                for rel in relationships:
                    try:
                        # Create or update source node
                        session.run("""
                            MERGE (source:GeologicalProcess {id: $source_id})
                            SET source.name = $source_name,
                                source.type = $source_type,
                                source.properties = $source_props
                        """, 
                        source_id=rel['source_id'],
                        source_name=rel['source_name'],
                        source_type=rel['source_type'],
                        source_props=json.dumps(rel.get('source_properties', {})))
                        
                        # Create or update target node
                        session.run("""
                            MERGE (target:GeologicalProcess {id: $target_id})
                            SET target.name = $target_name,
                                target.type = $target_type,
                                target.properties = $target_props
                        """,
                        target_id=rel['target_id'],
                        target_name=rel['target_name'],
                        target_type=rel['target_type'],
                        target_props=json.dumps(rel.get('target_properties', {})))
                        
                        # Create relationship
                        session.run("""
                            MATCH (source:GeologicalProcess {id: $source_id})
                            MATCH (target:GeologicalProcess {id: $target_id})
                            MERGE (source)-[r:INFLUENCES {type: $rel_type}]->(target)
                            SET r.strength = $strength,
                                r.confidence = $confidence,
                                r.temporal_lag = $temporal_lag,
                                r.properties = $rel_props
                        """,
                        source_id=rel['source_id'],
                        target_id=rel['target_id'],
                        rel_type=rel['relationship_type'],
                        strength=rel.get('strength', 1.0),
                        confidence=rel.get('confidence', 0.5),
                        temporal_lag=rel.get('temporal_lag', 0),
                        rel_props=json.dumps(rel.get('relationship_properties', {})))
                        
                        created_count += 1
                        
                    except Exception as e:
                        errors.append(f"Relationship {rel.get('source_id', 'unknown')} -> {rel.get('target_id', 'unknown')}: {str(e)}")
                
            except Exception as e:
                errors.append(f"Neo4j session error: {str(e)}")
        
        return {
            'created_count': created_count,
            'total_relationships': len(relationships),
            'errors': errors,
            'success_rate': created_count / len(relationships) if relationships else 0
        }
    
    def cache_analysis_results(self, analysis_id: str, results: Dict[str, Any], 
                             expiration_hours: int = 24) -> bool:
        """Cache analysis results in Redis with expiration"""
        
        if 'redis' not in self.connections:
            return False
        
        try:
            redis_client = self.connections['redis']
            
            # Store results with expiration
            cache_key = f"analysis:{analysis_id}"
            redis_client.setex(
                cache_key,
                expiration_hours * 3600,
                json.dumps(results, default=str)
            )
            
            # Store metadata
            metadata_key = f"analysis_meta:{analysis_id}"
            metadata = {
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=expiration_hours)).isoformat(),
                'size_bytes': len(json.dumps(results, default=str))
            }
            redis_client.setex(
                metadata_key,
                expiration_hours * 3600,
                json.dumps(metadata)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            return False
    
    def get_cached_results(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis results"""
        
        if 'redis' not in self.connections:
            return None
        
        try:
            redis_client = self.connections['redis']
            cache_key = f"analysis:{analysis_id}"
            
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Redis retrieval error: {e}")
            return None

            
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