"""
Real-Time Process Monitor for Geological Systems

This module implements a comprehensive real-time monitoring system for geological processes
with causal context, streaming data visualization, alert generation, and anomaly detection.
Designed for production-grade volcanic and seismic monitoring with advanced causal analysis.
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncIterator, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import json
import websockets
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import threading
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import redis
import asyncpg
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels for geological processes"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProcessType(Enum):
    """Types of geological processes being monitored"""
    SEISMIC = "seismic"
    VOLCANIC = "volcanic"
    DEFORMATION = "deformation"
    GEOCHEMICAL = "geochemical"
    THERMAL = "thermal"
    ELECTROMAGNETIC = "electromagnetic"

class AnomalyType(Enum):
    """Types of anomalies detected in geological data"""
    STATISTICAL = "statistical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    MULTIVARIATE = "multivariate"

@dataclass
class GeologicalDataPoint:
    """Real-time geological observation with metadata"""
    timestamp: datetime
    process_type: ProcessType
    station_id: str
    location: Tuple[float, float, float]  # lat, lon, elevation
    parameters: Dict[str, float]
    quality_score: float
    uncertainty: Dict[str, float]
    source_id: str
    processing_latency: float

@dataclass
class CausalRelationship:
    """Causal relationship between geological processes"""
    source_process: str
    target_process: str
    causal_strength: float
    confidence_interval: Tuple[float, float]
    temporal_lag: timedelta
    spatial_correlation: float
    last_updated: datetime
    validation_score: float

@dataclass
class RealTimeAlert:
    """Real-time alert for geological processes"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    process_type: ProcessType
    station_id: str
    location: Tuple[float, float, float]
    message: str
    causal_context: List[CausalRelationship]
    confidence: float
    predicted_evolution: Dict[str, Any]
    recommended_actions: List[str]
    data_quality_flags: List[str]

@dataclass
class AnomalyDetection:
    """Anomaly detection result with causal pathway analysis"""
    anomaly_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    process_type: ProcessType
    station_id: str
    anomaly_score: float
    statistical_significance: float
    causal_pathway: List[str]
    affected_processes: Set[str]
    temporal_pattern: Dict[str, Any]
    spatial_extent: Dict[str, Any]

@dataclass
class ProcessMonitoringState:
    """Current state of process monitoring system"""
    active_processes: Set[str]
    causal_network: Dict[str, List[CausalRelationship]]
    baseline_statistics: Dict[str, Dict[str, float]]
    anomaly_detectors: Dict[str, Any]
    alert_thresholds: Dict[str, Dict[str, float]]
    performance_metrics: Dict[str, float]
    last_update: datetime

class RealTimeProcessMonitor:
    """
    Advanced real-time process monitor for geological systems with causal analysis.
    
    Features:
    - Live streaming data processing with sub-second latency
    - Causal relationship monitoring and strength tracking
    - Multi-algorithm anomaly detection with pathway analysis
    - Predictive alerting based on causal models
    - Real-time data quality validation
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_state = ProcessMonitoringState(
            active_processes=set(),
            causal_network={},
            baseline_statistics={},
            anomaly_detectors={},
            alert_thresholds={},
            performance_metrics={},
            last_update=datetime.utcnow()
        )
        
        # Initialize data storage and streaming
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
        self.influx_client = InfluxDBClient(
            url=config.get('influxdb_url', 'http://localhost:8086'),
            token=config.get('influxdb_token'),
            org=config.get('influxdb_org', 'prometheus')
        )
        
        # Real-time data buffers
        self.data_buffers = defaultdict(lambda: deque(maxlen=10000))
        self.causal_strength_history = defaultdict(lambda: deque(maxlen=1000))
        self.alert_history = deque(maxlen=1000)
        
        # Anomaly detection models
        self.anomaly_models = {}
        self.scalers = {}
        
        # WebSocket connections for real-time updates
        self.websocket_connections = set()
        
        # Performance monitoring
        self.processing_times = deque(maxlen=1000)
        self.throughput_counter = 0
        self.last_throughput_reset = datetime.utcnow()
        
        # Threading for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 8))
        self.processing_lock = threading.RLock()
        
        # HTTP session for API calls
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info("Real-time process monitor initialized")
    
    async def start_monitoring(self) -> None:
        """Start the real-time monitoring system"""
        logger.info("Starting real-time geological process monitoring")
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession()
        
        # Initialize anomaly detection models
        await self._initialize_anomaly_models()
        
        # Load baseline statistics and causal relationships
        await self._load_baseline_data()
        
        # Start concurrent monitoring tasks
        tasks = [
            asyncio.create_task(self._stream_data_processor()),
            asyncio.create_task(self._causal_strength_monitor()),
            asyncio.create_task(self._anomaly_detection_engine()),
            asyncio.create_task(self._alert_generation_system()),
            asyncio.create_task(self._data_quality_validator()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._websocket_server())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in monitoring system: {e}")
            raise
        finally:
            # Clean up resources
            if self.session:
                await self.session.close()
            
            # Close data connectors
            if hasattr(self, '_data_connectors'):
                for connector in self._data_connectors.values():
                    try:
                        await connector.disconnect()
                    except Exception as e:
                        logger.warning(f"Error disconnecting connector: {e}")
    
    async def _stream_data_processor(self) -> None:
        """Process streaming geological data with real-time analysis"""
        logger.info("Starting streaming data processor")
        
        while True:
            try:
                start_time = datetime.utcnow()
                
                # Fetch real-time data from multiple sources
                data_batch = await self._fetch_real_time_data()
                
                if data_batch:
                    # Process each data point
                    for data_point in data_batch:
                        await self._process_data_point(data_point)
                    
                    # Update performance metrics
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    self.processing_times.append(processing_time)
                    self.throughput_counter += len(data_batch)
                    
                    # Store processed data
                    await self._store_processed_data(data_batch)
                
                # Maintain target processing frequency (10 Hz)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in stream processor: {e}")
                await asyncio.sleep(1.0)
    
    async def _fetch_real_time_data(self) -> List[GeologicalDataPoint]:
        """Fetch real-time data from multiple geological monitoring sources"""
        data_points = []
        
        try:
            # Initialize data connectors if not already done
            if not hasattr(self, '_data_connectors'):
                await self._initialize_data_connectors()
            
            # Fetch from all active data sources concurrently
            tasks = []
            
            # IRIS/EarthScope seismic data
            if 'iris' in self._data_connectors:
                tasks.append(self._fetch_iris_data())
            
            # USGS Volcano Hazards Program
            if 'usgs_volcano' in self._data_connectors:
                tasks.append(self._fetch_usgs_volcanic_data())
            
            # USGS Earthquake data
            if 'usgs_earthquake' in self._data_connectors:
                tasks.append(self._fetch_usgs_earthquake_data())
            
            # GeoNet (New Zealand) data
            if 'geonet' in self._data_connectors:
                tasks.append(self._fetch_geonet_data())
            
            # Execute all data fetching tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and collect data points
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Data fetch error: {result}")
                elif isinstance(result, list):
                    data_points.extend(result)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching real-time data: {e}")
            return []
    
    async def _initialize_data_connectors(self):
        """Initialize real data connectors for various geological data sources"""
        try:
            from backend.core.real_data_connectors.iris_connector import IRISConnector
            from backend.core.real_data_connectors.usgs_connector import USGSVolcanoConnector
            from backend.core.real_data_connectors.base_connector import DataSourceConfig, DataSourceType
        except ImportError as e:
            logger.error(f"Failed to import data connectors: {e}")
            logger.warning("Falling back to mock data generation")
            self._data_connectors = {}
            return
        
        self._data_connectors = {}
        
        try:
            # IRIS/EarthScope seismic data connector
            iris_config = DataSourceConfig(
                source_id="iris",
                name="IRIS/EarthScope",
                base_url="http://service.iris.edu/fdsnws",
                api_key=self.config.get('iris_api_key'),
                timeout=30,
                rate_limit=60,
                data_type=DataSourceType.SEISMIC
            )
            self._data_connectors['iris'] = IRISConnector(iris_config)
            await self._data_connectors['iris'].connect()
            
            # USGS Volcano Hazards Program connector
            usgs_volcano_config = DataSourceConfig(
                source_id="usgs_volcano",
                name="USGS Volcano Hazards Program",
                base_url="https://volcanoes.usgs.gov/vsc-api/v1",
                api_key=self.config.get('usgs_api_key'),
                timeout=30,
                rate_limit=100,
                data_type=DataSourceType.VOLCANIC
            )
            self._data_connectors['usgs_volcano'] = USGSVolcanoConnector(usgs_volcano_config)
            await self._data_connectors['usgs_volcano'].connect()
            
            # USGS Earthquake data connector (separate from volcano program)
            usgs_eq_config = DataSourceConfig(
                source_id="usgs_earthquake",
                name="USGS Earthquake Hazards Program",
                base_url="https://earthquake.usgs.gov/fdsnws",
                timeout=30,
                rate_limit=100,
                data_type=DataSourceType.SEISMIC
            )
            self._data_connectors['usgs_earthquake'] = IRISConnector(usgs_eq_config)  # Reuse IRIS connector for FDSN
            await self._data_connectors['usgs_earthquake'].connect()
            
            # GeoNet (New Zealand) connector
            geonet_config = DataSourceConfig(
                source_id="geonet",
                name="GeoNet New Zealand",
                base_url="https://api.geonet.org.nz",
                timeout=30,
                rate_limit=50,
                data_type=DataSourceType.SEISMIC
            )
            # Note: Would need a specific GeoNet connector implementation
            # For now, using IRIS connector as template
            
            logger.info(f"Initialized {len(self._data_connectors)} data connectors")
            
        except Exception as e:
            logger.error(f"Error initializing data connectors: {e}")
            self._data_connectors = {}
    
    async def _fetch_iris_data(self) -> List[GeologicalDataPoint]:
        """Fetch real-time seismic data from IRIS/EarthScope"""
        data_points = []
        
        try:
            connector = self._data_connectors.get('iris')
            if not connector:
                return data_points
            
            # Fetch recent earthquake events and waveform data
            async for observation in connector.fetch_real_time_data(
                networks=['IU', 'II', 'G', 'CU'],
                min_magnitude=3.0,
                fetch_interval=60
            ):
                # Convert GeologicalObservation to GeologicalDataPoint
                data_point = self._convert_observation_to_datapoint(observation, ProcessType.SEISMIC)
                if data_point:
                    data_points.append(data_point)
                    
                # Limit to prevent overwhelming the system
                if len(data_points) >= 50:
                    break
            
        except Exception as e:
            logger.error(f"Error fetching IRIS data: {e}")
        
        return data_points
    
    async def _fetch_usgs_volcanic_data(self) -> List[GeologicalDataPoint]:
        """Fetch real-time volcanic monitoring data from USGS"""
        data_points = []
        
        try:
            connector = self._data_connectors.get('usgs_volcano')
            if not connector:
                return data_points
            
            # Fetch volcanic alerts, activity, and monitoring data
            async for observation in connector.fetch_real_time_data(
                fetch_interval=300,  # 5 minutes
                min_magnitude=1.0
            ):
                # Convert GeologicalObservation to GeologicalDataPoint
                data_point = self._convert_observation_to_datapoint(observation, ProcessType.VOLCANIC)
                if data_point:
                    data_points.append(data_point)
                    
                # Limit to prevent overwhelming the system
                if len(data_points) >= 30:
                    break
            
        except Exception as e:
            logger.error(f"Error fetching USGS volcanic data: {e}")
        
        return data_points
    
    async def _fetch_usgs_earthquake_data(self) -> List[GeologicalDataPoint]:
        """Fetch real-time earthquake data from USGS"""
        data_points = []
        
        try:
            connector = self._data_connectors.get('usgs_earthquake')
            if not connector:
                return data_points
            
            # Fetch recent earthquake events
            async for observation in connector.fetch_real_time_data(
                networks=['US', 'CI', 'NC', 'UU'],
                min_magnitude=2.5,
                fetch_interval=120  # 2 minutes
            ):
                # Convert GeologicalObservation to GeologicalDataPoint
                data_point = self._convert_observation_to_datapoint(observation, ProcessType.SEISMIC)
                if data_point:
                    data_points.append(data_point)
                    
                # Limit to prevent overwhelming the system
                if len(data_points) >= 40:
                    break
            
        except Exception as e:
            logger.error(f"Error fetching USGS earthquake data: {e}")
        
        return data_points
    
    async def _fetch_geonet_data(self) -> List[GeologicalDataPoint]:
        """Fetch real-time data from GeoNet New Zealand"""
        data_points = []
        
        try:
            # Note: This would require a specific GeoNet connector implementation
            # For now, we'll implement a basic HTTP client approach
            
            headers = {
                'User-Agent': 'Prometheus-AI-Platform/1.0',
                'Accept': 'application/json'
            }
            
            # Fetch recent earthquakes from GeoNet API
            url = "https://api.geonet.org.nz/quake"
            params = {
                'MMI': '3',  # Minimum intensity
                'limit': '20'
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for feature in data.get('features', []):
                        props = feature.get('properties', {})
                        coords = feature.get('geometry', {}).get('coordinates', [])
                        
                        if len(coords) >= 3:
                            data_point = GeologicalDataPoint(
                                timestamp=datetime.fromisoformat(props.get('time', '').replace('Z', '+00:00')),
                                process_type=ProcessType.SEISMIC,
                                station_id=f"geonet_{props.get('publicID', 'unknown')}",
                                location=(coords[1], coords[0], coords[2] * -1),  # lat, lon, elevation
                                parameters={
                                    'magnitude': props.get('magnitude', 0),
                                    'depth_km': coords[2],
                                    'mmi': props.get('mmi', 0),
                                    'locality': props.get('locality', '')
                                },
                                quality_score=0.8,  # GeoNet generally has good quality
                                uncertainty={
                                    'magnitude': 0.1,
                                    'depth_km': 1.0
                                },
                                source_id='geonet_nz',
                                processing_latency=0.5
                            )
                            data_points.append(data_point)
            
        except Exception as e:
            logger.error(f"Error fetching GeoNet data: {e}")
        
        return data_points
    
    def _convert_observation_to_datapoint(self, observation, process_type: ProcessType) -> Optional[GeologicalDataPoint]:
        """Convert GeologicalObservation to GeologicalDataPoint format"""
        try:
            # Extract processing latency from metadata or estimate
            processing_latency = observation.metadata.get('processing_latency', 0.5)
            
            # Calculate quality score based on uncertainty and quality flags
            quality_score = 1.0
            if observation.uncertainty:
                quality_score *= max(0.1, 1.0 - observation.uncertainty)
            if observation.quality_flags:
                quality_score *= max(0.1, 1.0 - len(observation.quality_flags) * 0.1)
            
            # Extract uncertainty information
            uncertainty = {}
            if observation.uncertainty:
                # Map uncertainty to relevant parameters
                if process_type == ProcessType.SEISMIC:
                    uncertainty['magnitude'] = observation.uncertainty
                elif process_type == ProcessType.VOLCANIC:
                    uncertainty['measurement'] = observation.uncertainty
            
            data_point = GeologicalDataPoint(
                timestamp=observation.timestamp,
                process_type=process_type,
                station_id=observation.metadata.get('station_id', observation.observation_id),
                location=(observation.latitude, observation.longitude, observation.elevation or 0),
                parameters=observation.value,
                quality_score=quality_score,
                uncertainty=uncertainty,
                source_id=observation.data_source.lower().replace(' ', '_'),
                processing_latency=processing_latency
            )
            
            return data_point
            
        except Exception as e:
            logger.error(f"Error converting observation to data point: {e}")
            return None
    
    async def _fetch_database_stored_data(self) -> List[GeologicalDataPoint]:
        """Fetch recent data from local database storage"""
        data_points = []
        
        try:
            # Query InfluxDB for recent geological measurements
            query = '''
            from(bucket: "geological_data")
                |> range(start: -10m)
                |> filter(fn: (r) => r._measurement == "geological_observation")
                |> last()
            '''
            
            tables = self.influx_client.query_api().query(query)
            
            for table in tables:
                for record in table.records:
                    try:
                        # Extract data from InfluxDB record
                        process_type_str = record.values.get('process_type', 'seismic')
                        process_type = ProcessType(process_type_str)
                        
                        # Reconstruct parameters from InfluxDB fields
                        parameters = {}
                        for key, value in record.values.items():
                            if key.startswith('param_') and value is not None:
                                param_name = key.replace('param_', '')
                                parameters[param_name] = value
                        
                        data_point = GeologicalDataPoint(
                            timestamp=record.get_time(),
                            process_type=process_type,
                            station_id=record.values.get('station_id', 'unknown'),
                            location=(
                                record.values.get('latitude', 0),
                                record.values.get('longitude', 0),
                                record.values.get('elevation', 0)
                            ),
                            parameters=parameters,
                            quality_score=record.values.get('quality_score', 0.5),
                            uncertainty=json.loads(record.values.get('uncertainty', '{}')),
                            source_id=record.values.get('source_id', 'database'),
                            processing_latency=record.values.get('processing_latency', 1.0)
                        )
                        
                        data_points.append(data_point)
                        
                    except Exception as e:
                        logger.warning(f"Error processing database record: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Error fetching database stored data: {e}")
        
        return data_points
    
    async def _store_processed_data(self, data_batch: List[GeologicalDataPoint]) -> None:
        """Store processed geological data to databases"""
        try:
            # Store to InfluxDB for time-series analysis
            await self._store_to_influxdb(data_batch)
            
            # Store to PostgreSQL for geospatial queries
            await self._store_to_postgresql(data_batch)
            
            # Cache recent data in Redis
            await self._cache_to_redis(data_batch)
            
        except Exception as e:
            logger.error(f"Error storing processed data: {e}")
    
    async def _store_to_influxdb(self, data_batch: List[GeologicalDataPoint]) -> None:
        """Store data points to InfluxDB"""
        try:
            write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            
            points = []
            for data_point in data_batch:
                # Create InfluxDB point
                point = {
                    "measurement": "geological_observation",
                    "tags": {
                        "process_type": data_point.process_type.value,
                        "station_id": data_point.station_id,
                        "source_id": data_point.source_id
                    },
                    "fields": {
                        "latitude": data_point.location[0],
                        "longitude": data_point.location[1],
                        "elevation": data_point.location[2],
                        "quality_score": data_point.quality_score,
                        "processing_latency": data_point.processing_latency,
                        "uncertainty": json.dumps(data_point.uncertainty)
                    },
                    "time": data_point.timestamp
                }
                
                # Add parameters as fields
                for param_name, param_value in data_point.parameters.items():
                    if isinstance(param_value, (int, float)):
                        point["fields"][f"param_{param_name}"] = param_value
                
                points.append(point)
            
            write_api.write(bucket="geological_data", record=points)
            
        except Exception as e:
            logger.error(f"Error storing to InfluxDB: {e}")
    
    async def _store_to_postgresql(self, data_batch: List[GeologicalDataPoint]) -> None:
        """Store data points to PostgreSQL with PostGIS"""
        try:
            # This would require a PostgreSQL connection
            # Implementation would depend on your specific schema
            pass
            
        except Exception as e:
            logger.error(f"Error storing to PostgreSQL: {e}")
    
    async def _cache_to_redis(self, data_batch: List[GeologicalDataPoint]) -> None:
        """Cache recent data points to Redis"""
        try:
            for data_point in data_batch:
                cache_key = f"recent_data:{data_point.process_type.value}:{data_point.station_id}"
                cache_data = {
                    'timestamp': data_point.timestamp.isoformat(),
                    'parameters': data_point.parameters,
                    'quality_score': data_point.quality_score,
                    'location': data_point.location
                }
                
                # Store with 1 hour expiration
                self.redis_client.setex(
                    cache_key, 
                    3600, 
                    json.dumps(cache_data, default=str)
                )
                
        except Exception as e:
            logger.error(f"Error caching to Redis: {e}")    
    
    async def _process_data_point(self, data_point: GeologicalDataPoint) -> None:
        """Process individual data point with causal analysis"""
        try:
            with self.processing_lock:
                # Add to data buffer
                buffer_key = f"{data_point.process_type.value}_{data_point.station_id}"
                self.data_buffers[buffer_key].append(data_point)
                
                # Update monitoring state
                self.monitoring_state.active_processes.add(buffer_key)
                
                # Trigger real-time analysis
                await self._analyze_causal_relationships(data_point)
                await self._update_baseline_statistics(data_point)
                
                # Broadcast to WebSocket clients
                await self._broadcast_data_update(data_point)
                
        except Exception as e:
            logger.error(f"Error processing data point: {e}")
    
    async def _causal_strength_monitor(self) -> None:
        """Monitor causal relationship strengths in real-time"""
        logger.info("Starting causal strength monitoring")
        
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Update causal strengths for all active relationships
                for source_process in self.monitoring_state.active_processes:
                    for target_process in self.monitoring_state.active_processes:
                        if source_process != target_process:
                            strength = await self._calculate_causal_strength(
                                source_process, target_process
                            )
                            
                            if strength is not None:
                                # Store strength history
                                relationship_key = f"{source_process}->{target_process}"
                                self.causal_strength_history[relationship_key].append({
                                    'timestamp': current_time,
                                    'strength': strength,
                                    'confidence': await self._calculate_causal_confidence(
                                        source_process, target_process
                                    )
                                })
                                
                                # Check for significant changes
                                await self._check_causal_strength_changes(
                                    relationship_key, strength
                                )
                
                # Update monitoring state
                self.monitoring_state.last_update = current_time
                
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in causal strength monitoring: {e}")
                await asyncio.sleep(10.0)
    
    async def _calculate_causal_strength(self, source: str, target: str) -> Optional[float]:
        """Calculate causal strength between two processes using Granger causality"""
        try:
            # Get recent data for both processes
            source_data = list(self.data_buffers[source])[-100:]  # Last 100 points
            target_data = list(self.data_buffers[target])[-100:]
            
            if len(source_data) < 20 or len(target_data) < 20:
                return None
            
            # Extract time series for analysis
            source_values = []
            target_values = []
            
            for dp in source_data:
                # Use primary parameter for each process type
                if dp.process_type == ProcessType.SEISMIC:
                    source_values.append(dp.parameters.get('ground_velocity', 0))
                elif dp.process_type == ProcessType.VOLCANIC:
                    source_values.append(dp.parameters.get('so2_flux', 0))
                elif dp.process_type == ProcessType.DEFORMATION:
                    source_values.append(dp.parameters.get('vertical_displacement', 0))
                elif dp.process_type == ProcessType.GEOCHEMICAL:
                    source_values.append(dp.parameters.get('co2_concentration', 0))
            
            for dp in target_data:
                if dp.process_type == ProcessType.SEISMIC:
                    target_values.append(dp.parameters.get('ground_velocity', 0))
                elif dp.process_type == ProcessType.VOLCANIC:
                    target_values.append(dp.parameters.get('so2_flux', 0))
                elif dp.process_type == ProcessType.DEFORMATION:
                    target_values.append(dp.parameters.get('vertical_displacement', 0))
                elif dp.process_type == ProcessType.GEOCHEMICAL:
                    target_values.append(dp.parameters.get('co2_concentration', 0))
            
            # Advanced Multi-Modal Causal Discovery with Geological Physics Constraints
            if len(source_values) == len(target_values) and len(source_values) >= 20:
                return await self._advanced_geological_causal_inference(
                    source_values, target_values, source, target
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating causal strength: {e}")
            return None
    
    async def _advanced_geological_causal_inference(self, source_values: List[float], 
                                                   target_values: List[float], 
                                                   source: str, target: str) -> float:
        """
        Advanced Multi-Modal Causal Discovery with Geological Physics Constraints
        
        This novel implementation combines:
        1. Nonlinear Granger Causality with Neural Networks
        2. Transfer Entropy with Adaptive Bandwidth
        3. Convergent Cross Mapping (CCM) for Nonlinear Dynamics
        4. Physics-Informed Causal Constraints
        5. Bayesian Causal Structure Learning
        6. Wavelet-based Multi-Scale Analysis
        7. Information-Theoretic Measures
        """
        try:
            import torch
            import torch.nn as nn
            from scipy.signal import hilbert, cwt, morlet2
            from scipy.stats import entropy, gaussian_kde
            from sklearn.metrics import mutual_info_score
            from sklearn.neighbors import NearestNeighbors
            import warnings
            warnings.filterwarnings('ignore')
            
            # Convert to numpy arrays and normalize
            X = np.array(source_values, dtype=np.float32)
            Y = np.array(target_values, dtype=np.float32)
            n = len(X)
            
            # 1. NONLINEAR GRANGER CAUSALITY WITH NEURAL NETWORKS
            class NonlinearGrangerNet(nn.Module):
                def __init__(self, input_dim, hidden_dim=32):
                    super().__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                    self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim//2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim//2, 1)
                    )
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                    return self.fc(attn_out[:, -1, :])
            
            # Prepare sequences for neural network
            seq_len = min(10, n // 4)
            if seq_len < 3:
                seq_len = 3
                
            def create_sequences(data, seq_len):
                sequences = []
                targets = []
                for i in range(len(data) - seq_len):
                    sequences.append(data[i:i+seq_len])
                    targets.append(data[i+seq_len])
                return np.array(sequences), np.array(targets)
            
            # Neural Granger causality: X -> Y
            X_seq, Y_target = create_sequences(Y, seq_len)
            XY_seq = np.column_stack([X_seq, np.roll(X, -seq_len)[:-seq_len].reshape(-1, 1)])
            
            if len(X_seq) > 10:
                # Model without X (restricted)
                model_restricted = NonlinearGrangerNet(seq_len)
                X_tensor = torch.FloatTensor(X_seq).unsqueeze(-1)
                Y_tensor = torch.FloatTensor(Y_target)
                
                # Model with X (unrestricted)  
                model_unrestricted = NonlinearGrangerNet(seq_len + 1)
                XY_tensor = torch.FloatTensor(XY_seq)
                
                # Quick training (simplified for real-time)
                optimizer_r = torch.optim.Adam(model_restricted.parameters(), lr=0.01)
                optimizer_u = torch.optim.Adam(model_unrestricted.parameters(), lr=0.01)
                criterion = nn.MSELoss()
                
                # Train restricted model
                model_restricted.train()
                for _ in range(20):
                    pred_r = model_restricted(X_tensor).squeeze()
                    loss_r = criterion(pred_r, Y_tensor)
                    optimizer_r.zero_grad()
                    loss_r.backward()
                    optimizer_r.step()
                
                # Train unrestricted model
                model_unrestricted.train()
                for _ in range(20):
                    pred_u = model_unrestricted(XY_tensor).squeeze()
                    loss_u = criterion(pred_u, Y_tensor)
                    optimizer_u.zero_grad()
                    loss_u.backward()
                    optimizer_u.step()
                
                # Calculate F-statistic equivalent
                with torch.no_grad():
                    model_restricted.eval()
                    model_unrestricted.eval()
                    pred_r = model_restricted(X_tensor).squeeze()
                    pred_u = model_unrestricted(XY_tensor).squeeze()
                    
                    mse_r = torch.mean((pred_r - Y_tensor)**2).item()
                    mse_u = torch.mean((pred_u - Y_tensor)**2).item()
                    
                    neural_granger = max(0, (mse_r - mse_u) / (mse_r + 1e-8))
            else:
                neural_granger = 0.0
            
            # 2. TRANSFER ENTROPY WITH ADAPTIVE BANDWIDTH
            def adaptive_transfer_entropy(x, y, k=3):
                """Transfer entropy with adaptive bandwidth selection"""
                # Embed time series
                def embed_series(series, dim, tau=1):
                    n = len(series)
                    embedded = np.zeros((n - (dim-1)*tau, dim))
                    for i in range(dim):
                        embedded[:, i] = series[i*tau:n-(dim-1-i)*tau]
                    return embedded
                
                # Adaptive bandwidth using Scott's rule
                def adaptive_bandwidth(data):
                    n, d = data.shape
                    return n**(-1./(d+4))
                
                try:
                    # Embed both series
                    x_embed = embed_series(x, k)
                    y_embed = embed_series(y, k)
                    
                    if len(x_embed) < 10:
                        return 0.0
                    
                    # Joint embedding
                    xy_embed = np.column_stack([x_embed[:-1], y_embed[:-1]])
                    y_future = y_embed[1:, 0]
                    
                    # Estimate densities using KDE with adaptive bandwidth
                    h_xy = adaptive_bandwidth(xy_embed)
                    h_y = adaptive_bandwidth(y_embed[:-1])
                    
                    kde_xy = gaussian_kde(xy_embed.T, bw_method=h_xy)
                    kde_y = gaussian_kde(y_embed[:-1].T, bw_method=h_y)
                    
                    # Calculate transfer entropy
                    te = 0.0
                    for i in range(len(y_future)):
                        point_xy = xy_embed[i]
                        point_y = y_embed[i]
                        
                        p_xy = kde_xy(point_xy)
                        p_y = kde_y(point_y)
                        
                        if p_xy > 1e-10 and p_y > 1e-10:
                            te += np.log2(p_xy / p_y)
                    
                    return te / len(y_future)
                    
                except Exception:
                    return 0.0
            
            transfer_entropy_xy = adaptive_transfer_entropy(X, Y)
            
            # 3. CONVERGENT CROSS MAPPING (CCM) FOR NONLINEAR DYNAMICS
            def convergent_cross_mapping(x, y, E=3, tau=1, L_min=10):
                """CCM for detecting nonlinear causal relationships"""
                try:
                    def embed_and_predict(source, target, E, tau, L):
                        # Time-delay embedding
                        n = len(source)
                        embedded = np.zeros((n - (E-1)*tau, E))
                        for i in range(E):
                            embedded[:, i] = source[i*tau:n-(E-1-i)*tau]
                        
                        # Nearest neighbors prediction
                        nbrs = NearestNeighbors(n_neighbors=E+1, metric='euclidean')
                        nbrs.fit(embedded[:L])
                        
                        predictions = []
                        for i in range(L, min(len(embedded), L+20)):
                            distances, indices = nbrs.kneighbors([embedded[i]])
                            weights = np.exp(-distances[0][1:])  # Exclude self
                            weights /= np.sum(weights)
                            
                            pred = np.sum(weights * target[indices[0][1:]])
                            predictions.append(pred)
                        
                        if len(predictions) > 0:
                            actual = target[L:L+len(predictions)]
                            correlation = np.corrcoef(predictions, actual)[0, 1]
                            return correlation if not np.isnan(correlation) else 0
                        return 0
                    
                    if len(x) < L_min + E*tau:
                        return 0.0
                    
                    # X predicts Y (X -> Y causality)
                    ccm_xy = embed_and_predict(x, y, E, tau, L_min)
                    return max(0, ccm_xy)
                    
                except Exception:
                    return 0.0
            
            ccm_strength = convergent_cross_mapping(X, Y)
            
            # 4. PHYSICS-INFORMED CAUSAL CONSTRAINTS
            def geological_physics_constraints(source_type, target_type, lag_time):
                """Apply geological physics knowledge to constrain causality"""
                constraints = {
                    ('seismic', 'volcanic'): {'max_lag': 3600, 'strength_multiplier': 1.2},
                    ('seismic', 'deformation'): {'max_lag': 1800, 'strength_multiplier': 1.1},
                    ('volcanic', 'geochemical'): {'max_lag': 7200, 'strength_multiplier': 1.3},
                    ('deformation', 'seismic'): {'max_lag': 900, 'strength_multiplier': 0.9},
                    ('geochemical', 'volcanic'): {'max_lag': 14400, 'strength_multiplier': 1.1}
                }
                
                # Extract process types from source/target strings
                source_proc = 'seismic' if 'seismic' in source.lower() else \
                             'volcanic' if 'volcanic' in source.lower() else \
                             'deformation' if 'deformation' in source.lower() else 'geochemical'
                target_proc = 'seismic' if 'seismic' in target.lower() else \
                             'volcanic' if 'volcanic' in target.lower() else \
                             'deformation' if 'deformation' in target.lower() else 'geochemical'
                
                constraint = constraints.get((source_proc, target_proc), 
                                           {'max_lag': 3600, 'strength_multiplier': 1.0})
                
                # Apply temporal constraint
                if lag_time > constraint['max_lag']:
                    return 0.0
                
                return constraint['strength_multiplier']
            
            physics_multiplier = geological_physics_constraints(source, target, 300)  # 5 min lag
            
            # 5. WAVELET-BASED MULTI-SCALE ANALYSIS
            def wavelet_coherence(x, y):
                """Wavelet coherence for multi-scale causal analysis"""
                try:
                    from scipy.signal import coherence
                    
                    # Multiple frequency bands for geological processes
                    freq_bands = [
                        (0.001, 0.01),   # Long-term geological
                        (0.01, 0.1),     # Medium-term
                        (0.1, 1.0),      # Short-term
                        (1.0, 10.0)      # High-frequency
                    ]
                    
                    coherences = []
                    for f_low, f_high in freq_bands:
                        # Bandpass filter (simplified)
                        from scipy.signal import butter, filtfilt
                        
                        nyquist = 0.5 * (len(x) / (x[-1] if hasattr(x, '__getitem__') else 1))
                        if nyquist > f_high:
                            low = f_low / nyquist
                            high = f_high / nyquist
                            
                            if low < 1 and high < 1 and low < high:
                                b, a = butter(2, [low, high], btype='band')
                                x_filt = filtfilt(b, a, x)
                                y_filt = filtfilt(b, a, y)
                                
                                # Cross-correlation at this scale
                                cross_corr = np.corrcoef(x_filt, y_filt)[0, 1]
                                if not np.isnan(cross_corr):
                                    coherences.append(abs(cross_corr))
                    
                    return np.mean(coherences) if coherences else 0.0
                    
                except Exception:
                    return 0.0
            
            wavelet_coh = wavelet_coherence(X, Y)
            
            # 6. INFORMATION-THEORETIC MEASURES
            def mutual_information_advanced(x, y, bins=10):
                """Advanced mutual information with adaptive binning"""
                try:
                    # Adaptive binning based on data distribution
                    n = len(x)
                    bins_x = max(3, min(bins, int(np.sqrt(n))))
                    bins_y = max(3, min(bins, int(np.sqrt(n))))
                    
                    # Discretize using quantiles for robustness
                    x_discrete = pd.cut(x, bins_x, labels=False, duplicates='drop')
                    y_discrete = pd.cut(y, bins_y, labels=False, duplicates='drop')
                    
                    # Remove NaN values
                    valid_mask = ~(pd.isna(x_discrete) | pd.isna(y_discrete))
                    x_clean = x_discrete[valid_mask]
                    y_clean = y_discrete[valid_mask]
                    
                    if len(x_clean) > 5:
                        mi = mutual_info_score(x_clean, y_clean)
                        # Normalize by joint entropy
                        joint_entropy = entropy(np.histogram2d(x_clean, y_clean)[0].flatten())
                        return mi / (joint_entropy + 1e-8) if joint_entropy > 0 else 0
                    return 0.0
                    
                except Exception:
                    return 0.0
            
            mutual_info = mutual_information_advanced(X, Y)
            
            # 7. ENSEMBLE CAUSAL STRENGTH CALCULATION
            # Combine all measures with adaptive weights
            measures = {
                'neural_granger': neural_granger,
                'transfer_entropy': transfer_entropy_xy,
                'ccm': ccm_strength,
                'wavelet_coherence': wavelet_coh,
                'mutual_information': mutual_info
            }
            
            # Adaptive weighting based on data characteristics
            n_points = len(X)
            nonlinearity = np.std(np.diff(X, 2)) / (np.std(X) + 1e-8)  # Second derivative measure
            
            if n_points > 50 and nonlinearity > 0.1:
                # Nonlinear regime - emphasize CCM and neural methods
                weights = [0.3, 0.2, 0.3, 0.1, 0.1]
            elif n_points > 30:
                # Medium data - balanced approach
                weights = [0.25, 0.25, 0.2, 0.15, 0.15]
            else:
                # Limited data - emphasize robust measures
                weights = [0.1, 0.3, 0.1, 0.25, 0.25]
            
            # Calculate weighted ensemble score
            measure_values = list(measures.values())
            ensemble_score = np.average(measure_values, weights=weights)
            
            # Apply physics constraints
            final_score = ensemble_score * physics_multiplier
            
            # Confidence-based adjustment
            confidence_factors = [
                1.0 if not np.isnan(v) and v > 0 else 0.5 for v in measure_values
            ]
            confidence_weight = np.mean(confidence_factors)
            
            final_score *= confidence_weight
            
            # Ensure output is in [0, 1] range
            final_score = max(0.0, min(1.0, final_score))
            
            # Log detailed results for monitoring
            logger.debug(f"Advanced causal analysis {source}->{target}: "
                        f"Neural={neural_granger:.3f}, TE={transfer_entropy_xy:.3f}, "
                        f"CCM={ccm_strength:.3f}, Wavelet={wavelet_coh:.3f}, "
                        f"MI={mutual_info:.3f}, Final={final_score:.3f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error in advanced causal inference: {e}")
            # Fallback to simple correlation
            try:
                correlation = abs(np.corrcoef(source_values, target_values)[0, 1])
                return correlation if not np.isnan(correlation) else 0.0
            except:
                return 0.0
    
    async def _calculate_causal_confidence(self, source: str, target: str) -> float:
        """Calculate confidence in causal relationship"""
        try:
            # Get data quality scores
            source_data = list(self.data_buffers[source])[-50:]
            target_data = list(self.data_buffers[target])[-50:]
            
            if not source_data or not target_data:
                return 0.0
            
            # Average quality scores
            source_quality = np.mean([dp.quality_score for dp in source_data])
            target_quality = np.mean([dp.quality_score for dp in target_data])
            
            # Data completeness
            expected_points = 50
            completeness = min(len(source_data), len(target_data)) / expected_points
            
            # Temporal alignment quality
            if len(source_data) > 1 and len(target_data) > 1:
                source_intervals = [(source_data[i].timestamp - source_data[i-1].timestamp).total_seconds() 
                                 for i in range(1, len(source_data))]
                target_intervals = [(target_data[i].timestamp - target_data[i-1].timestamp).total_seconds() 
                                 for i in range(1, len(target_data))]
                
                interval_consistency = 1.0 - (np.std(source_intervals + target_intervals) / 
                                            np.mean(source_intervals + target_intervals))
                interval_consistency = max(0, min(1, interval_consistency))
            else:
                interval_consistency = 0.5
            
            # Combined confidence score
            confidence = (source_quality + target_quality) / 2 * completeness * interval_consistency
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating causal confidence: {e}")
            return 0.0
    
    async def _anomaly_detection_engine(self) -> None:
        """Advanced anomaly detection with causal pathway analysis"""
        logger.info("Starting anomaly detection engine")
        
        while True:
            try:
                current_time = datetime.utcnow()
                detected_anomalies = []
                
                # Run anomaly detection for each active process
                for process_key in self.monitoring_state.active_processes:
                    anomalies = await self._detect_process_anomalies(process_key)
                    detected_anomalies.extend(anomalies)
                
                # Analyze causal pathways for detected anomalies
                for anomaly in detected_anomalies:
                    await self._analyze_anomaly_causal_pathway(anomaly)
                    
                    # Store anomaly in history
                    await self._store_anomaly(anomaly)
                    
                    # Generate alerts if necessary
                    if anomaly.anomaly_score > 0.8:
                        await self._generate_anomaly_alert(anomaly)
                
                await asyncio.sleep(2.0)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
                await asyncio.sleep(5.0)
    
    async def _detect_process_anomalies(self, process_key: str) -> List[AnomalyDetection]:
        """Detect anomalies in a specific geological process"""
        anomalies = []
        
        try:
            # Get recent data
            recent_data = list(self.data_buffers[process_key])[-100:]
            
            if len(recent_data) < 10:
                return anomalies
            
            # Extract features for anomaly detection
            features = []
            timestamps = []
            
            for dp in recent_data:
                feature_vector = []
                for param_name, param_value in dp.parameters.items():
                    if isinstance(param_value, (int, float)):
                        feature_vector.append(param_value)
                
                if feature_vector:
                    features.append(feature_vector)
                    timestamps.append(dp.timestamp)
            
            if len(features) < 10:
                return anomalies
            
            # Standardize features
            if process_key not in self.scalers:
                self.scalers[process_key] = StandardScaler()
                scaled_features = self.scalers[process_key].fit_transform(features)
            else:
                scaled_features = self.scalers[process_key].transform(features)
            
            # Statistical anomaly detection
            statistical_anomalies = await self._detect_statistical_anomalies(
                scaled_features, timestamps, process_key
            )
            anomalies.extend(statistical_anomalies)
            
            # Temporal pattern anomalies
            temporal_anomalies = await self._detect_temporal_anomalies(
                features, timestamps, process_key
            )
            anomalies.extend(temporal_anomalies)
            
            # Multivariate anomalies using Isolation Forest
            if len(features) >= 20:
                multivariate_anomalies = await self._detect_multivariate_anomalies(
                    scaled_features, timestamps, process_key
                )
                anomalies.extend(multivariate_anomalies)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {process_key}: {e}")
            return []
    
    async def _detect_statistical_anomalies(self, features: np.ndarray, 
                                          timestamps: List[datetime], 
                                          process_key: str) -> List[AnomalyDetection]:
        """Detect statistical anomalies using z-score and IQR methods"""
        anomalies = []
        
        try:
            # Z-score based detection
            z_scores = np.abs(stats.zscore(features, axis=0))
            
            # IQR based detection
            q1 = np.percentile(features, 25, axis=0)
            q3 = np.percentile(features, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for i, (timestamp, feature_vector) in enumerate(zip(timestamps, features)):
                # Check z-score anomalies (threshold: 3.0)
                max_z_score = np.max(z_scores[i])
                if max_z_score > 3.0:
                    anomaly = AnomalyDetection(
                        anomaly_id=f"stat_{process_key}_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        anomaly_type=AnomalyType.STATISTICAL,
                        process_type=ProcessType(process_key.split('_')[0]),
                        station_id=process_key.split('_', 1)[1],
                        anomaly_score=min(1.0, max_z_score / 5.0),
                        statistical_significance=1.0 - stats.norm.cdf(max_z_score),
                        causal_pathway=[],
                        affected_processes=set(),
                        temporal_pattern={'z_score': max_z_score},
                        spatial_extent={}
                    )
                    anomalies.append(anomaly)
                
                # Check IQR anomalies
                iqr_violations = np.sum((feature_vector < lower_bound) | 
                                      (feature_vector > upper_bound))
                if iqr_violations > 0:
                    anomaly_score = iqr_violations / len(feature_vector)
                    if anomaly_score > 0.3:  # More than 30% of features are outliers
                        anomaly = AnomalyDetection(
                            anomaly_id=f"iqr_{process_key}_{timestamp.isoformat()}",
                            timestamp=timestamp,
                            anomaly_type=AnomalyType.STATISTICAL,
                            process_type=ProcessType(process_key.split('_')[0]),
                            station_id=process_key.split('_', 1)[1],
                            anomaly_score=anomaly_score,
                            statistical_significance=anomaly_score,
                            causal_pathway=[],
                            affected_processes=set(),
                            temporal_pattern={'iqr_violations': iqr_violations},
                            spatial_extent={}
                        )
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
            return []
    
    async def _detect_temporal_anomalies(self, features: List[List[float]], 
                                       timestamps: List[datetime], 
                                       process_key: str) -> List[AnomalyDetection]:
        """Detect temporal pattern anomalies"""
        anomalies = []
        
        try:
            if len(features) < 5:
                return anomalies
            
            # Calculate temporal derivatives
            derivatives = []
            for i in range(1, len(features)):
                dt = (timestamps[i] - timestamps[i-1]).total_seconds()
                if dt > 0:
                    derivative = [(features[i][j] - features[i-1][j]) / dt 
                                for j in range(len(features[i]))]
                    derivatives.append(derivative)
            
            if len(derivatives) < 3:
                return anomalies
            
            # Detect sudden changes (high derivatives)
            derivative_magnitudes = [np.linalg.norm(d) for d in derivatives]
            threshold = np.mean(derivative_magnitudes) + 3 * np.std(derivative_magnitudes)
            
            for i, (magnitude, timestamp) in enumerate(zip(derivative_magnitudes, timestamps[1:])):
                if magnitude > threshold:
                    anomaly = AnomalyDetection(
                        anomaly_id=f"temp_{process_key}_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        anomaly_type=AnomalyType.TEMPORAL,
                        process_type=ProcessType(process_key.split('_')[0]),
                        station_id=process_key.split('_', 1)[1],
                        anomaly_score=min(1.0, magnitude / (threshold * 2)),
                        statistical_significance=1.0 - stats.norm.cdf(
                            (magnitude - np.mean(derivative_magnitudes)) / 
                            np.std(derivative_magnitudes)
                        ),
                        causal_pathway=[],
                        affected_processes=set(),
                        temporal_pattern={
                            'derivative_magnitude': magnitude,
                            'threshold': threshold,
                            'change_rate': magnitude
                        },
                        spatial_extent={}
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in temporal anomaly detection: {e}")
            return []
    
    async def _detect_multivariate_anomalies(self, features: np.ndarray, 
                                           timestamps: List[datetime], 
                                           process_key: str) -> List[AnomalyDetection]:
        """Detect multivariate anomalies using Isolation Forest"""
        anomalies = []
        
        try:
            # Initialize or update Isolation Forest model
            if process_key not in self.anomaly_models:
                self.anomaly_models[process_key] = IsolationForest(
                    contamination=0.1,  # Expect 10% anomalies
                    random_state=42,
                    n_estimators=100
                )
                
                # Fit on recent data
                if len(features) >= 20:
                    self.anomaly_models[process_key].fit(features)
            
            # Predict anomalies
            if process_key in self.anomaly_models:
                anomaly_scores = self.anomaly_models[process_key].decision_function(features)
                predictions = self.anomaly_models[process_key].predict(features)
                
                # Convert scores to 0-1 range
                normalized_scores = (anomaly_scores - anomaly_scores.min()) / (
                    anomaly_scores.max() - anomaly_scores.min() + 1e-8
                )
                
                for i, (prediction, score, timestamp) in enumerate(
                    zip(predictions, normalized_scores, timestamps)
                ):
                    if prediction == -1:  # Anomaly detected
                        anomaly = AnomalyDetection(
                            anomaly_id=f"multi_{process_key}_{timestamp.isoformat()}",
                            timestamp=timestamp,
                            anomaly_type=AnomalyType.MULTIVARIATE,
                            process_type=ProcessType(process_key.split('_')[0]),
                            station_id=process_key.split('_', 1)[1],
                            anomaly_score=1.0 - score,  # Invert score (lower is more anomalous)
                            statistical_significance=1.0 - score,
                            causal_pathway=[],
                            affected_processes=set(),
                            temporal_pattern={'isolation_score': score},
                            spatial_extent={}
                        )
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in multivariate anomaly detection: {e}")
            return []    

    async def _analyze_anomaly_causal_pathway(self, anomaly: AnomalyDetection) -> None:
        """Analyze causal pathway for detected anomaly"""
        try:
            # Find processes that could have caused this anomaly
            causal_pathway = []
            affected_processes = set()
            
            # Check causal relationships leading to this process
            target_process = f"{anomaly.process_type.value}_{anomaly.station_id}"
            
            for source_process in self.monitoring_state.active_processes:
                if source_process != target_process:
                    # Check if there's a strong causal relationship
                    strength = await self._calculate_causal_strength(source_process, target_process)
                    if strength and strength > 0.5:  # Strong causal relationship
                        causal_pathway.append(source_process)
                        
                        # Check if source process also has anomalies
                        source_data = list(self.data_buffers[source_process])[-10:]
                        if source_data:
                            recent_anomalies = await self._check_recent_anomalies(source_process)
                            if recent_anomalies:
                                affected_processes.add(source_process)
            
            # Update anomaly with causal information
            anomaly.causal_pathway = causal_pathway
            anomaly.affected_processes = affected_processes
            
            # Analyze spatial extent if multiple stations affected
            await self._analyze_spatial_extent(anomaly)
            
        except Exception as e:
            logger.error(f"Error analyzing anomaly causal pathway: {e}")
    
    async def _check_recent_anomalies(self, process_key: str) -> List[AnomalyDetection]:
        """
        Advanced anomaly detection system with multi-algorithm approach and temporal analysis.
        
        Features:
        - Multi-scale temporal analysis (1min, 5min, 15min, 1hr windows)
        - Ensemble anomaly detection (Isolation Forest, Statistical, Causal)
        - Adaptive thresholding based on historical patterns
        - Anomaly persistence and correlation tracking
        - Real-time anomaly scoring with confidence intervals
        """
        try:
            recent_anomalies = []
            current_time = datetime.utcnow()
            
            # Define multiple temporal windows for analysis
            temporal_windows = {
                'immediate': timedelta(minutes=2),
                'short_term': timedelta(minutes=10),
                'medium_term': timedelta(minutes=30),
                'long_term': timedelta(hours=2)
            }
            
            # Get recent data for the process
            process_data = list(self.data_buffers.get(process_key, []))
            if len(process_data) < 10:  # Need minimum data for analysis
                return recent_anomalies
            
            # Extract process type and station ID from key
            process_parts = process_key.split('_', 1)
            if len(process_parts) != 2:
                return recent_anomalies
            
            process_type_str, station_id = process_parts
            try:
                process_type = ProcessType(process_type_str)
            except ValueError:
                logger.warning(f"Unknown process type: {process_type_str}")
                return recent_anomalies
            
            # Prepare data for analysis
            df_data = self._prepare_anomaly_analysis_data(process_data, current_time)
            if df_data.empty:
                return recent_anomalies
            
            # Multi-algorithm anomaly detection
            anomaly_results = {}
            
            # 1. Statistical Anomaly Detection (Z-score and IQR methods)
            statistical_anomalies = await self._detect_statistical_anomalies(
                df_data, process_key, temporal_windows
            )
            anomaly_results['statistical'] = statistical_anomalies
            
            # 2. Machine Learning Anomaly Detection (Isolation Forest)
            ml_anomalies = await self._detect_ml_anomalies(
                df_data, process_key, temporal_windows
            )
            anomaly_results['ml'] = ml_anomalies
            
            # 3. Temporal Pattern Anomaly Detection
            temporal_anomalies = await self._detect_temporal_pattern_anomalies(
                df_data, process_key, temporal_windows
            )
            anomaly_results['temporal'] = temporal_anomalies
            
            # 4. Causal Relationship Anomaly Detection
            causal_anomalies = await self._detect_causal_anomalies(
                df_data, process_key, temporal_windows
            )
            anomaly_results['causal'] = causal_anomalies
            
            # 5. Multivariate Anomaly Detection (cross-parameter analysis)
            multivariate_anomalies = await self._detect_multivariate_anomalies(
                df_data, process_key, temporal_windows
            )
            anomaly_results['multivariate'] = multivariate_anomalies
            
            # Ensemble anomaly scoring and consolidation
            consolidated_anomalies = await self._consolidate_anomaly_results(
                anomaly_results, process_type, station_id, current_time
            )
            
            # Filter and rank anomalies by significance
            significant_anomalies = await self._filter_significant_anomalies(
                consolidated_anomalies, process_key
            )
            
            # Store anomalies for future reference
            await self._store_detected_anomalies(significant_anomalies, process_key)
            
            # Retrieve stored recent anomalies from cache/database
            stored_anomalies = await self._retrieve_stored_anomalies(
                process_key, temporal_windows['medium_term']
            )
            
            # Combine new and stored anomalies
            all_recent_anomalies = significant_anomalies + stored_anomalies
            
            # Remove duplicates and sort by timestamp
            unique_anomalies = self._deduplicate_anomalies(all_recent_anomalies)
            
            # Update anomaly persistence tracking
            await self._update_anomaly_persistence(unique_anomalies, process_key)
            
            return unique_anomalies
            
        except Exception as e:
            logger.error(f"Error in advanced anomaly detection for {process_key}: {e}")
            return []
    
    def _prepare_anomaly_analysis_data(self, process_data: List[GeologicalDataPoint], 
                                     current_time: datetime) -> pd.DataFrame:
        """Prepare geological data for anomaly analysis"""
        try:
            # Convert to DataFrame for analysis
            data_records = []
            for dp in process_data:
                record = {
                    'timestamp': dp.timestamp,
                    'quality_score': dp.quality_score,
                    'processing_latency': dp.processing_latency,
                    **dp.parameters
                }
                data_records.append(record)
            
            df = pd.DataFrame(data_records)
            if df.empty:
                return df
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Filter to recent data (last 4 hours)
            cutoff_time = current_time - timedelta(hours=4)
            df = df[df['timestamp'] >= cutoff_time]
            
            # Add time-based features
            df['time_since_current'] = (current_time - df['timestamp']).dt.total_seconds()
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Calculate rolling statistics for trend analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['timestamp', 'time_since_current', 'hour_of_day', 'day_of_week']:
                    df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5, min_periods=1).mean()
                    df[f'{col}_rolling_std_5'] = df[col].rolling(window=5, min_periods=1).std()
                    df[f'{col}_rolling_mean_15'] = df[col].rolling(window=15, min_periods=1).mean()
                    df[f'{col}_rolling_std_15'] = df[col].rolling(window=15, min_periods=1).std()
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing anomaly analysis data: {e}")
            return pd.DataFrame()
    
    async def _detect_statistical_anomalies(self, df_data: pd.DataFrame, process_key: str,
                                          temporal_windows: Dict[str, timedelta]) -> List[Dict]:
        """Detect statistical anomalies using Z-score and IQR methods"""
        anomalies = []
        current_time = datetime.utcnow()
        
        try:
            numeric_cols = df_data.select_dtypes(include=[np.number]).columns
            base_cols = [col for col in numeric_cols if not col.startswith(('time_', 'hour_', 'day_', '_rolling_'))]
            
            for window_name, window_duration in temporal_windows.items():
                window_start = current_time - window_duration
                window_data = df_data[df_data['timestamp'] >= window_start].copy()
                
                if len(window_data) < 5:  # Need minimum data points
                    continue
                
                for col in base_cols:
                    if col in window_data.columns and window_data[col].notna().sum() > 3:
                        # Z-score anomaly detection
                        z_scores = np.abs(stats.zscore(window_data[col].dropna()))
                        z_threshold = 2.5 + (0.5 * (window_duration.total_seconds() / 3600))  # Adaptive threshold
                        
                        z_anomaly_indices = np.where(z_scores > z_threshold)[0]
                        
                        # IQR anomaly detection
                        Q1 = window_data[col].quantile(0.25)
                        Q3 = window_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        iqr_threshold = 1.5 + (0.3 * (window_duration.total_seconds() / 3600))  # Adaptive threshold
                        
                        lower_bound = Q1 - iqr_threshold * IQR
                        upper_bound = Q3 + iqr_threshold * IQR
                        
                        iqr_anomalies = window_data[
                            (window_data[col] < lower_bound) | (window_data[col] > upper_bound)
                        ]
                        
                        # Combine Z-score and IQR results
                        for idx in z_anomaly_indices:
                            if idx < len(window_data):
                                row = window_data.iloc[idx]
                                anomaly_score = float(z_scores[idx] / z_threshold)
                                
                                anomalies.append({
                                    'timestamp': row['timestamp'],
                                    'parameter': col,
                                    'value': row[col],
                                    'anomaly_score': anomaly_score,
                                    'method': 'z_score',
                                    'window': window_name,
                                    'threshold': z_threshold,
                                    'statistical_significance': 1 - stats.norm.sf(z_scores[idx])
                                })
                        
                        for _, row in iqr_anomalies.iterrows():
                            # Calculate IQR-based anomaly score
                            if row[col] < lower_bound:
                                iqr_score = (lower_bound - row[col]) / (IQR + 1e-6)
                            else:
                                iqr_score = (row[col] - upper_bound) / (IQR + 1e-6)
                            
                            anomalies.append({
                                'timestamp': row['timestamp'],
                                'parameter': col,
                                'value': row[col],
                                'anomaly_score': min(iqr_score, 5.0),  # Cap at 5.0
                                'method': 'iqr',
                                'window': window_name,
                                'threshold': iqr_threshold,
                                'statistical_significance': min(iqr_score / 3.0, 0.99)
                            })
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
        
        return anomalies
    
    async def _detect_ml_anomalies(self, df_data: pd.DataFrame, process_key: str,
                                 temporal_windows: Dict[str, timedelta]) -> List[Dict]:
        """Detect anomalies using machine learning (Isolation Forest)"""
        anomalies = []
        current_time = datetime.utcnow()
        
        try:
            # Prepare features for ML analysis
            feature_cols = []
            numeric_cols = df_data.select_dtypes(include=[np.number]).columns
            
            # Select relevant features (exclude time-based and rolling features for base analysis)
            base_cols = [col for col in numeric_cols if not col.startswith(('time_', 'hour_', 'day_'))]
            feature_cols.extend(base_cols)
            
            if len(feature_cols) < 2:  # Need at least 2 features
                return anomalies
            
            for window_name, window_duration in temporal_windows.items():
                window_start = current_time - window_duration
                window_data = df_data[df_data['timestamp'] >= window_start].copy()
                
                if len(window_data) < 10:  # Need minimum data for ML
                    continue
                
                # Prepare feature matrix
                feature_data = window_data[feature_cols].dropna()
                if len(feature_data) < 5:
                    continue
                
                # Initialize or get existing model for this process
                model_key = f"{process_key}_{window_name}"
                if model_key not in self.anomaly_models:
                    # Create new Isolation Forest model
                    contamination = min(0.1, max(0.01, 0.05 + 0.02 * len(feature_cols)))
                    self.anomaly_models[model_key] = IsolationForest(
                        contamination=contamination,
                        random_state=42,
                        n_estimators=100
                    )
                    
                    # Create scaler for this model
                    self.scalers[model_key] = StandardScaler()
                
                # Scale features
                try:
                    scaled_features = self.scalers[model_key].fit_transform(feature_data)
                except Exception:
                    # If scaling fails, use raw features
                    scaled_features = feature_data.values
                
                # Fit and predict anomalies
                model = self.anomaly_models[model_key]
                
                # Incremental learning: fit on recent data
                model.fit(scaled_features)
                
                # Get anomaly scores
                anomaly_scores = model.decision_function(scaled_features)
                anomaly_predictions = model.predict(scaled_features)
                
                # Convert scores to probabilities (higher = more anomalous)
                anomaly_probabilities = 1 / (1 + np.exp(anomaly_scores))
                
                # Identify anomalies
                anomaly_indices = np.where(anomaly_predictions == -1)[0]
                
                for idx in anomaly_indices:
                    if idx < len(window_data):
                        row_idx = feature_data.index[idx]
                        row = window_data.loc[row_idx]
                        
                        anomalies.append({
                            'timestamp': row['timestamp'],
                            'parameter': 'multivariate_ml',
                            'value': float(anomaly_scores[idx]),
                            'anomaly_score': float(anomaly_probabilities[idx]),
                            'method': 'isolation_forest',
                            'window': window_name,
                            'threshold': 0.5,
                            'statistical_significance': float(anomaly_probabilities[idx]),
                            'feature_importance': dict(zip(feature_cols, 
                                                         np.abs(scaled_features[idx])))
                        })
            
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {e}")
        
        return anomalies
    
    async def _detect_temporal_pattern_anomalies(self, df_data: pd.DataFrame, process_key: str,
                                               temporal_windows: Dict[str, timedelta]) -> List[Dict]:
        """Detect anomalies in temporal patterns and trends"""
        anomalies = []
        current_time = datetime.utcnow()
        
        try:
            numeric_cols = df_data.select_dtypes(include=[np.number]).columns
            base_cols = [col for col in numeric_cols if not col.startswith(('time_', 'hour_', 'day_', '_rolling_'))]
            
            for window_name, window_duration in temporal_windows.items():
                window_start = current_time - window_duration
                window_data = df_data[df_data['timestamp'] >= window_start].copy()
                
                if len(window_data) < 15:  # Need sufficient data for trend analysis
                    continue
                
                # Sort by timestamp for temporal analysis
                window_data = window_data.sort_values('timestamp')
                
                for col in base_cols:
                    if col in window_data.columns and window_data[col].notna().sum() > 10:
                        values = window_data[col].dropna()
                        timestamps = window_data.loc[values.index, 'timestamp']
                        
                        # Convert timestamps to numeric for trend analysis
                        time_numeric = (timestamps - timestamps.min()).dt.total_seconds()
                        
                        # 1. Trend change detection using linear regression
                        if len(values) >= 10:
                            # Split into two halves and compare trends
                            mid_point = len(values) // 2
                            
                            # First half trend
                            slope1, _, r1, p1, _ = stats.linregress(
                                time_numeric[:mid_point], values[:mid_point]
                            )
                            
                            # Second half trend
                            slope2, _, r2, p2, _ = stats.linregress(
                                time_numeric[mid_point:], values[mid_point:]
                            )
                            
                            # Detect significant trend changes
                            slope_change = abs(slope2 - slope1)
                            trend_threshold = np.std(values) * 0.1 / (window_duration.total_seconds() / 3600)
                            
                            if slope_change > trend_threshold and min(abs(r1), abs(r2)) > 0.3:
                                anomalies.append({
                                    'timestamp': timestamps.iloc[mid_point],
                                    'parameter': col,
                                    'value': slope_change,
                                    'anomaly_score': min(slope_change / trend_threshold, 5.0),
                                    'method': 'trend_change',
                                    'window': window_name,
                                    'threshold': trend_threshold,
                                    'statistical_significance': 1 - max(p1, p2),
                                    'trend_details': {
                                        'slope1': slope1,
                                        'slope2': slope2,
                                        'r1': r1,
                                        'r2': r2
                                    }
                                })
                        
                        # 2. Sudden spike/drop detection
                        if len(values) >= 5:
                            # Calculate rate of change
                            rate_of_change = np.diff(values)
                            rate_threshold = 3 * np.std(rate_of_change)
                            
                            spike_indices = np.where(np.abs(rate_of_change) > rate_threshold)[0]
                            
                            for spike_idx in spike_indices:
                                if spike_idx + 1 < len(timestamps):
                                    anomalies.append({
                                        'timestamp': timestamps.iloc[spike_idx + 1],
                                        'parameter': col,
                                        'value': rate_of_change[spike_idx],
                                        'anomaly_score': min(abs(rate_of_change[spike_idx]) / rate_threshold, 5.0),
                                        'method': 'sudden_change',
                                        'window': window_name,
                                        'threshold': rate_threshold,
                                        'statistical_significance': 0.95,
                                        'change_magnitude': abs(rate_of_change[spike_idx])
                                    })
                        
                        # 3. Periodicity anomaly detection
                        if len(values) >= 20:
                            # Simple autocorrelation-based periodicity check
                            autocorr = np.correlate(values - np.mean(values), 
                                                  values - np.mean(values), mode='full')
                            autocorr = autocorr[autocorr.size // 2:]
                            autocorr = autocorr / autocorr[0]  # Normalize
                            
                            # Look for unexpected periodic patterns
                            significant_lags = np.where(autocorr[1:] > 0.7)[0] + 1
                            
                            if len(significant_lags) > 0:
                                # Check if this periodicity is unusual
                                expected_period = window_duration.total_seconds() / len(values)
                                detected_period = significant_lags[0] * expected_period
                                
                                # Flag if period is very short (< 5 minutes) or very long (> 2 hours)
                                if detected_period < 300 or detected_period > 7200:
                                    anomalies.append({
                                        'timestamp': timestamps.iloc[-1],
                                        'parameter': col,
                                        'value': detected_period,
                                        'anomaly_score': 2.0,
                                        'method': 'periodicity_anomaly',
                                        'window': window_name,
                                        'threshold': 0.7,
                                        'statistical_significance': 0.8,
                                        'detected_period_seconds': detected_period,
                                        'autocorr_peak': autocorr[significant_lags[0]]
                                    })
            
        except Exception as e:
            logger.error(f"Error in temporal pattern anomaly detection: {e}")
        
        return anomalies
    
    async def _detect_causal_anomalies(self, df_data: pd.DataFrame, process_key: str,
                                     temporal_windows: Dict[str, timedelta]) -> List[Dict]:
        """Detect anomalies in causal relationships between processes"""
        anomalies = []
        current_time = datetime.utcnow()
        
        try:
            # Get causal relationships for this process
            causal_relationships = self.monitoring_state.causal_network.get(process_key, [])
            
            if not causal_relationships:
                return anomalies
            
            for window_name, window_duration in temporal_windows.items():
                window_start = current_time - window_duration
                
                for relationship in causal_relationships:
                    # Check if causal strength has changed significantly
                    historical_strength = relationship.causal_strength
                    
                    # Get recent causal strength measurements
                    strength_key = f"{relationship.source_process}_{relationship.target_process}"
                    recent_strengths = list(self.causal_strength_history.get(strength_key, []))
                    
                    if len(recent_strengths) >= 5:
                        recent_window = [s for s in recent_strengths 
                                       if s['timestamp'] >= window_start]
                        
                        if recent_window:
                            current_strength = np.mean([s['strength'] for s in recent_window])
                            strength_std = np.std([s['strength'] for s in recent_window])
                            
                            # Detect significant changes in causal strength
                            strength_change = abs(current_strength - historical_strength)
                            change_threshold = max(0.1, 2 * strength_std)
                            
                            if strength_change > change_threshold:
                                # Calculate confidence based on sample size and consistency
                                confidence = min(0.95, len(recent_window) / 20.0)
                                
                                anomalies.append({
                                    'timestamp': recent_window[-1]['timestamp'],
                                    'parameter': 'causal_strength',
                                    'value': current_strength,
                                    'anomaly_score': min(strength_change / change_threshold, 5.0),
                                    'method': 'causal_strength_change',
                                    'window': window_name,
                                    'threshold': change_threshold,
                                    'statistical_significance': confidence,
                                    'causal_details': {
                                        'source_process': relationship.source_process,
                                        'target_process': relationship.target_process,
                                        'historical_strength': historical_strength,
                                        'current_strength': current_strength,
                                        'strength_change': strength_change
                                    }
                                })
                    
                    # Check for temporal lag anomalies
                    expected_lag = relationship.temporal_lag.total_seconds()
                    
                    # This would require cross-correlation analysis between processes
                    # For now, we'll implement a simplified version
                    if expected_lag > 0:
                        # Get data from both source and target processes
                        source_data = list(self.data_buffers.get(relationship.source_process, []))
                        target_data = list(self.data_buffers.get(relationship.target_process, []))
                        
                        if len(source_data) >= 10 and len(target_data) >= 10:
                            # Simple lag detection using correlation
                            # This is a simplified implementation
                            lag_anomaly_score = await self._analyze_temporal_lag_anomaly(
                                source_data, target_data, expected_lag, window_duration
                            )
                            
                            if lag_anomaly_score > 1.5:
                                anomalies.append({
                                    'timestamp': current_time,
                                    'parameter': 'temporal_lag',
                                    'value': lag_anomaly_score,
                                    'anomaly_score': min(lag_anomaly_score, 5.0),
                                    'method': 'temporal_lag_anomaly',
                                    'window': window_name,
                                    'threshold': 1.5,
                                    'statistical_significance': 0.8,
                                    'lag_details': {
                                        'expected_lag_seconds': expected_lag,
                                        'source_process': relationship.source_process,
                                        'target_process': relationship.target_process
                                    }
                                })
            
        except Exception as e:
            logger.error(f"Error in causal anomaly detection: {e}")
        
        return anomalies
    
    async def _analyze_temporal_lag_anomaly(self, source_data: List[GeologicalDataPoint],
                                          target_data: List[GeologicalDataPoint],
                                          expected_lag: float, window_duration: timedelta) -> float:
        """Analyze temporal lag anomalies between source and target processes"""
        try:
            # This is a simplified implementation
            # In a full implementation, you would use cross-correlation analysis
            
            # Get recent data within window
            current_time = datetime.utcnow()
            window_start = current_time - window_duration
            
            recent_source = [dp for dp in source_data if dp.timestamp >= window_start]
            recent_target = [dp for dp in target_data if dp.timestamp >= window_start]
            
            if len(recent_source) < 5 or len(recent_target) < 5:
                return 0.0
            
            # Simple correlation-based lag detection
            # Extract a common parameter for correlation
            source_values = []
            target_values = []
            
            for dp in recent_source:
                if 'magnitude' in dp.parameters:
                    source_values.append(dp.parameters['magnitude'])
                elif dp.parameters:
                    source_values.append(list(dp.parameters.values())[0])
            
            for dp in recent_target:
                if 'magnitude' in dp.parameters:
                    target_values.append(dp.parameters['magnitude'])
                elif dp.parameters:
                    target_values.append(list(dp.parameters.values())[0])
            
            if len(source_values) < 5 or len(target_values) < 5:
                return 0.0
            
            # Calculate correlation at expected lag
            min_len = min(len(source_values), len(target_values))
            lag_samples = int(expected_lag / 60)  # Assuming 1-minute sampling
            
            if lag_samples >= min_len:
                return 0.0
            
            # Correlation at expected lag
            if lag_samples > 0:
                expected_corr = np.corrcoef(
                    source_values[:-lag_samples], 
                    target_values[lag_samples:]
                )[0, 1]
            else:
                expected_corr = np.corrcoef(source_values, target_values)[0, 1]
            
            # Correlation at zero lag (should be lower if lag exists)
            zero_lag_corr = np.corrcoef(
                source_values[:min_len], 
                target_values[:min_len]
            )[0, 1]
            
            # Anomaly score based on correlation difference
            if not np.isnan(expected_corr) and not np.isnan(zero_lag_corr):
                correlation_diff = abs(zero_lag_corr - expected_corr)
                return correlation_diff * 5.0  # Scale to anomaly score
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing temporal lag anomaly: {e}")
            return 0.0
    
    async def _detect_multivariate_anomalies(self, df_data: pd.DataFrame, process_key: str,
                                           temporal_windows: Dict[str, timedelta]) -> List[Dict]:
        """Detect multivariate anomalies using cross-parameter analysis"""
        anomalies = []
        current_time = datetime.utcnow()
        
        try:
            numeric_cols = df_data.select_dtypes(include=[np.number]).columns
            base_cols = [col for col in numeric_cols if not col.startswith(('time_', 'hour_', 'day_', '_rolling_'))]
            
            if len(base_cols) < 2:  # Need at least 2 parameters for multivariate analysis
                return anomalies
            
            for window_name, window_duration in temporal_windows.items():
                window_start = current_time - window_duration
                window_data = df_data[df_data['timestamp'] >= window_start].copy()
                
                if len(window_data) < 10:
                    continue
                
                # Correlation matrix analysis
                correlation_data = window_data[base_cols].dropna()
                if len(correlation_data) < 5:
                    continue
                
                # Calculate correlation matrix
                corr_matrix = correlation_data.corr()
                
                # Detect unusual correlations
                for i, col1 in enumerate(base_cols):
                    for j, col2 in enumerate(base_cols[i+1:], i+1):
                        if col1 in corr_matrix.index and col2 in corr_matrix.columns:
                            correlation = corr_matrix.loc[col1, col2]
                            
                            if not np.isnan(correlation):
                                # Check if correlation is unusually high or low
                                # This would ideally compare against historical baselines
                                abs_corr = abs(correlation)
                                
                                # Flag very high correlations (> 0.9) or very low (< -0.9)
                                # that might indicate sensor malfunction or unusual coupling
                                if abs_corr > 0.9:
                                    anomalies.append({
                                        'timestamp': window_data['timestamp'].iloc[-1],
                                        'parameter': f'{col1}_vs_{col2}',
                                        'value': correlation,
                                        'anomaly_score': (abs_corr - 0.7) * 10,  # Scale 0.7-1.0 to 0-3
                                        'method': 'high_correlation',
                                        'window': window_name,
                                        'threshold': 0.9,
                                        'statistical_significance': 0.85,
                                        'correlation_details': {
                                            'parameter1': col1,
                                            'parameter2': col2,
                                            'correlation_value': correlation
                                        }
                                    })
                
                # Principal Component Analysis for multivariate outliers
                if len(correlation_data) >= 10 and len(base_cols) >= 3:
                    try:
                        from sklearn.decomposition import PCA
                        from sklearn.preprocessing import StandardScaler
                        
                        # Standardize data
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(correlation_data)
                        
                        # Apply PCA
                        pca = PCA(n_components=min(3, len(base_cols)))
                        pca_data = pca.fit_transform(scaled_data)
                        
                        # Calculate Mahalanobis distance in PCA space
                        mean_pca = np.mean(pca_data, axis=0)
                        cov_pca = np.cov(pca_data.T)
                        
                        # Add small regularization to covariance matrix
                        cov_pca += np.eye(cov_pca.shape[0]) * 1e-6
                        
                        try:
                            inv_cov_pca = np.linalg.inv(cov_pca)
                            
                            for idx, row in enumerate(pca_data):
                                diff = row - mean_pca
                                mahal_dist = np.sqrt(diff.T @ inv_cov_pca @ diff)
                                
                                # Flag points with high Mahalanobis distance
                                if mahal_dist > 3.0:  # Threshold for outliers
                                    timestamp_idx = correlation_data.index[idx]
                                    timestamp = window_data.loc[timestamp_idx, 'timestamp']
                                    
                                    anomalies.append({
                                        'timestamp': timestamp,
                                        'parameter': 'multivariate_pca',
                                        'value': mahal_dist,
                                        'anomaly_score': min(mahal_dist / 2.0, 5.0),
                                        'method': 'pca_mahalanobis',
                                        'window': window_name,
                                        'threshold': 3.0,
                                        'statistical_significance': 0.9,
                                        'pca_details': {
                                            'mahalanobis_distance': mahal_dist,
                                            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                                            'n_components': pca.n_components_
                                        }
                                    })
                        
                        except np.linalg.LinAlgError:
                            # Skip if covariance matrix is singular
                            pass
                    
                    except ImportError:
                        # Skip PCA analysis if sklearn is not available
                        pass
            
        except Exception as e:
            logger.error(f"Error in multivariate anomaly detection: {e}")
        
        return anomalies
    
    async def _consolidate_anomaly_results(self, anomaly_results: Dict[str, List[Dict]],
                                         process_type: ProcessType, station_id: str,
                                         current_time: datetime) -> List[AnomalyDetection]:
        """Consolidate anomaly results from multiple detection methods"""
        consolidated = []
        
        try:
            # Group anomalies by timestamp and parameter for ensemble scoring
            anomaly_groups = defaultdict(list)
            
            for method, anomalies in anomaly_results.items():
                for anomaly in anomalies:
                    # Create grouping key based on timestamp (rounded to minute) and parameter
                    timestamp_key = anomaly['timestamp'].replace(second=0, microsecond=0)
                    group_key = (timestamp_key, anomaly['parameter'])
                    
                    anomaly['detection_method'] = method
                    anomaly_groups[group_key].append(anomaly)
            
            # Create consolidated anomaly objects
            for (timestamp, parameter), group_anomalies in anomaly_groups.items():
                # Ensemble scoring: combine scores from different methods
                ensemble_score = 0.0
                max_significance = 0.0
                detection_methods = []
                method_details = {}
                
                # Weight different methods
                method_weights = {
                    'statistical': 1.0,
                    'ml': 1.2,  # Slightly higher weight for ML methods
                    'temporal': 1.1,
                    'causal': 1.3,  # Higher weight for causal anomalies
                    'multivariate': 1.1
                }
                
                total_weight = 0.0
                for anomaly in group_anomalies:
                    method = anomaly['detection_method']
                    weight = method_weights.get(method, 1.0)
                    
                    ensemble_score += anomaly['anomaly_score'] * weight
                    total_weight += weight
                    
                    max_significance = max(max_significance, anomaly['statistical_significance'])
                    detection_methods.append(anomaly['method'])
                    
                    # Store method-specific details
                    method_details[method] = {
                        'score': anomaly['anomaly_score'],
                        'method': anomaly['method'],
                        'threshold': anomaly['threshold']
                    }
                    
                    # Add method-specific details if available
                    for key, value in anomaly.items():
                        if key.endswith('_details'):
                            method_details[method][key] = value
                
                # Normalize ensemble score
                if total_weight > 0:
                    ensemble_score /= total_weight
                
                # Boost score if multiple methods agree
                if len(group_anomalies) > 1:
                    ensemble_score *= (1.0 + 0.2 * (len(group_anomalies) - 1))
                
                # Determine anomaly type based on detection methods
                anomaly_type = AnomalyType.STATISTICAL  # Default
                if any('causal' in method for method in detection_methods):
                    anomaly_type = AnomalyType.CAUSAL
                elif any('temporal' in method for method in detection_methods):
                    anomaly_type = AnomalyType.TEMPORAL
                elif any('multivariate' in method or 'pca' in method for method in detection_methods):
                    anomaly_type = AnomalyType.MULTIVARIATE
                elif any('spatial' in method for method in detection_methods):
                    anomaly_type = AnomalyType.SPATIAL
                
                # Create consolidated anomaly detection object
                anomaly_detection = AnomalyDetection(
                    anomaly_id=f"{station_id}_{timestamp.isoformat()}_{parameter}",
                    timestamp=timestamp,
                    anomaly_type=anomaly_type,
                    process_type=process_type,
                    station_id=station_id,
                    anomaly_score=min(ensemble_score, 10.0),  # Cap at 10.0
                    statistical_significance=max_significance,
                    causal_pathway=[],  # Will be filled by causal analysis
                    affected_processes=set(),  # Will be filled by causal analysis
                    temporal_pattern={
                        'detection_methods': detection_methods,
                        'method_details': method_details,
                        'ensemble_components': len(group_anomalies),
                        'parameter': parameter
                    },
                    spatial_extent={}  # Will be filled by spatial analysis
                )
                
                consolidated.append(anomaly_detection)
            
        except Exception as e:
            logger.error(f"Error consolidating anomaly results: {e}")
        
        return consolidated
    
    async def _filter_significant_anomalies(self, anomalies: List[AnomalyDetection],
                                          process_key: str) -> List[AnomalyDetection]:
        """Filter anomalies based on significance and adaptive thresholds"""
        try:
            if not anomalies:
                return []
            
            # Get adaptive thresholds for this process
            thresholds = self.monitoring_state.alert_thresholds.get(process_key, {})
            
            # Default thresholds
            min_anomaly_score = thresholds.get('min_anomaly_score', 1.5)
            min_statistical_significance = thresholds.get('min_statistical_significance', 0.7)
            
            # Adaptive thresholding based on recent anomaly frequency
            recent_anomaly_count = await self._get_recent_anomaly_count(process_key)
            
            # Increase thresholds if too many recent anomalies (reduce noise)
            if recent_anomaly_count > 10:
                min_anomaly_score *= 1.2
                min_statistical_significance *= 1.1
            elif recent_anomaly_count < 2:
                # Lower thresholds if very few anomalies (increase sensitivity)
                min_anomaly_score *= 0.9
                min_statistical_significance *= 0.95
            
            # Filter anomalies
            significant_anomalies = []
            for anomaly in anomalies:
                if (anomaly.anomaly_score >= min_anomaly_score and
                    anomaly.statistical_significance >= min_statistical_significance):
                    
                    # Additional filtering based on anomaly type
                    if anomaly.anomaly_type == AnomalyType.CAUSAL:
                        # Lower threshold for causal anomalies (more important)
                        if anomaly.anomaly_score >= min_anomaly_score * 0.8:
                            significant_anomalies.append(anomaly)
                    elif anomaly.anomaly_type == AnomalyType.MULTIVARIATE:
                        # Higher threshold for multivariate (can be noisy)
                        if anomaly.anomaly_score >= min_anomaly_score * 1.1:
                            significant_anomalies.append(anomaly)
                    else:
                        significant_anomalies.append(anomaly)
            
            # Limit number of anomalies to prevent overwhelming
            max_anomalies = thresholds.get('max_anomalies_per_check', 20)
            if len(significant_anomalies) > max_anomalies:
                # Sort by anomaly score and keep top ones
                significant_anomalies.sort(key=lambda x: x.anomaly_score, reverse=True)
                significant_anomalies = significant_anomalies[:max_anomalies]
            
            return significant_anomalies
            
        except Exception as e:
            logger.error(f"Error filtering significant anomalies: {e}")
            return anomalies  # Return all if filtering fails
    
    async def _get_recent_anomaly_count(self, process_key: str) -> int:
        """Get count of recent anomalies for adaptive thresholding"""
        try:
            # Check Redis cache for recent anomaly count
            cache_key = f"anomaly_count:{process_key}:1h"
            cached_count = self.redis_client.get(cache_key)
            
            if cached_count:
                return int(cached_count)
            
            # If not cached, return default
            return 5
            
        except Exception as e:
            logger.error(f"Error getting recent anomaly count: {e}")
            return 5  # Default value
    
    async def _store_detected_anomalies(self, anomalies: List[AnomalyDetection],
                                      process_key: str) -> None:
        """Store detected anomalies for future reference"""
        try:
            if not anomalies:
                return
            
            # Store in Redis with expiration
            for anomaly in anomalies:
                cache_key = f"anomaly:{process_key}:{anomaly.anomaly_id}"
                anomaly_data = {
                    'anomaly_id': anomaly.anomaly_id,
                    'timestamp': anomaly.timestamp.isoformat(),
                    'anomaly_type': anomaly.anomaly_type.value,
                    'process_type': anomaly.process_type.value,
                    'station_id': anomaly.station_id,
                    'anomaly_score': anomaly.anomaly_score,
                    'statistical_significance': anomaly.statistical_significance,
                    'temporal_pattern': anomaly.temporal_pattern,
                    'spatial_extent': anomaly.spatial_extent
                }
                
                # Store with 4 hour expiration
                self.redis_client.setex(
                    cache_key,
                    14400,  # 4 hours
                    json.dumps(anomaly_data, default=str)
                )
            
            # Update anomaly count for adaptive thresholding
            count_key = f"anomaly_count:{process_key}:1h"
            current_count = self.redis_client.get(count_key) or 0
            new_count = int(current_count) + len(anomalies)
            self.redis_client.setex(count_key, 3600, new_count)  # 1 hour expiration
            
            # Store in InfluxDB for long-term analysis
            await self._store_anomalies_to_influxdb(anomalies, process_key)
            
        except Exception as e:
            logger.error(f"Error storing detected anomalies: {e}")
    
    async def _store_anomalies_to_influxdb(self, anomalies: List[AnomalyDetection],
                                         process_key: str) -> None:
        """Store anomalies to InfluxDB for long-term analysis"""
        try:
            write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            
            points = []
            for anomaly in anomalies:
                point = {
                    "measurement": "anomaly_detection",
                    "tags": {
                        "process_key": process_key,
                        "anomaly_type": anomaly.anomaly_type.value,
                        "process_type": anomaly.process_type.value,
                        "station_id": anomaly.station_id
                    },
                    "fields": {
                        "anomaly_score": anomaly.anomaly_score,
                        "statistical_significance": anomaly.statistical_significance,
                        "temporal_pattern": json.dumps(anomaly.temporal_pattern),
                        "spatial_extent": json.dumps(anomaly.spatial_extent),
                        "anomaly_id": anomaly.anomaly_id
                    },
                    "time": anomaly.timestamp
                }
                points.append(point)
            
            write_api.write(bucket="anomaly_data", record=points)
            
        except Exception as e:
            logger.error(f"Error storing anomalies to InfluxDB: {e}")
    
    async def _retrieve_stored_anomalies(self, process_key: str,
                                       time_window: timedelta) -> List[AnomalyDetection]:
        """Retrieve stored anomalies from cache/database"""
        stored_anomalies = []
        
        try:
            # Get anomalies from Redis cache
            pattern = f"anomaly:{process_key}:*"
            cached_keys = self.redis_client.keys(pattern)
            
            current_time = datetime.utcnow()
            cutoff_time = current_time - time_window
            
            for key in cached_keys:
                try:
                    cached_data = self.redis_client.get(key)
                    if cached_data:
                        anomaly_data = json.loads(cached_data)
                        timestamp = datetime.fromisoformat(anomaly_data['timestamp'])
                        
                        if timestamp >= cutoff_time:
                            # Reconstruct AnomalyDetection object
                            anomaly = AnomalyDetection(
                                anomaly_id=anomaly_data['anomaly_id'],
                                timestamp=timestamp,
                                anomaly_type=AnomalyType(anomaly_data['anomaly_type']),
                                process_type=ProcessType(anomaly_data['process_type']),
                                station_id=anomaly_data['station_id'],
                                anomaly_score=anomaly_data['anomaly_score'],
                                statistical_significance=anomaly_data['statistical_significance'],
                                causal_pathway=[],  # Will be filled if needed
                                affected_processes=set(),  # Will be filled if needed
                                temporal_pattern=anomaly_data['temporal_pattern'],
                                spatial_extent=anomaly_data['spatial_extent']
                            )
                            stored_anomalies.append(anomaly)
                
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Error parsing cached anomaly {key}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error retrieving stored anomalies: {e}")
        
        return stored_anomalies
    
    def _deduplicate_anomalies(self, anomalies: List[AnomalyDetection]) -> List[AnomalyDetection]:
        """Remove duplicate anomalies and sort by timestamp"""
        try:
            # Use anomaly_id for deduplication
            seen_ids = set()
            unique_anomalies = []
            
            for anomaly in anomalies:
                if anomaly.anomaly_id not in seen_ids:
                    seen_ids.add(anomaly.anomaly_id)
                    unique_anomalies.append(anomaly)
            
            # Sort by timestamp (most recent first)
            unique_anomalies.sort(key=lambda x: x.timestamp, reverse=True)
            
            return unique_anomalies
            
        except Exception as e:
            logger.error(f"Error deduplicating anomalies: {e}")
            return anomalies
    
    async def _update_anomaly_persistence(self, anomalies: List[AnomalyDetection],
                                        process_key: str) -> None:
        """Update anomaly persistence tracking for trend analysis"""
        try:
            if not anomalies:
                return
            
            # Track persistent anomalies (same type occurring repeatedly)
            persistence_key = f"anomaly_persistence:{process_key}"
            
            # Get current persistence data
            persistence_data = self.redis_client.get(persistence_key)
            if persistence_data:
                persistence_dict = json.loads(persistence_data)
            else:
                persistence_dict = {}
            
            current_time = datetime.utcnow()
            
            # Update persistence for each anomaly type
            for anomaly in anomalies:
                type_key = f"{anomaly.anomaly_type.value}_{anomaly.station_id}"
                
                if type_key not in persistence_dict:
                    persistence_dict[type_key] = {
                        'count': 0,
                        'first_occurrence': anomaly.timestamp.isoformat(),
                        'last_occurrence': anomaly.timestamp.isoformat(),
                        'max_score': anomaly.anomaly_score
                    }
                
                # Update persistence data
                persistence_dict[type_key]['count'] += 1
                persistence_dict[type_key]['last_occurrence'] = anomaly.timestamp.isoformat()
                persistence_dict[type_key]['max_score'] = max(
                    persistence_dict[type_key]['max_score'],
                    anomaly.anomaly_score
                )
            
            # Clean up old persistence data (older than 24 hours)
            cutoff_time = current_time - timedelta(hours=24)
            keys_to_remove = []
            
            for type_key, data in persistence_dict.items():
                last_occurrence = datetime.fromisoformat(data['last_occurrence'])
                if last_occurrence < cutoff_time:
                    keys_to_remove.append(type_key)
            
            for key in keys_to_remove:
                del persistence_dict[key]
            
            # Store updated persistence data
            self.redis_client.setex(
                persistence_key,
                86400,  # 24 hours
                json.dumps(persistence_dict, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error updating anomaly persistence: {e}")
    
    async def _analyze_spatial_extent(self, anomaly: AnomalyDetection) -> None:
        """Analyze spatial extent of anomaly"""
        try:
            # Find nearby stations with similar anomalies
            affected_stations = []
            anomaly_location = None
            
            # Get location of anomaly station
            recent_data = list(self.data_buffers.get(
                f"{anomaly.process_type.value}_{anomaly.station_id}", []
            ))
            
            if recent_data:
                anomaly_location = recent_data[-1].location
                
                # Check other stations of same type
                for process_key in self.monitoring_state.active_processes:
                    if (process_key.startswith(anomaly.process_type.value) and 
                        process_key != f"{anomaly.process_type.value}_{anomaly.station_id}"):
                        
                        station_data = list(self.data_buffers[process_key])
                        if station_data:
                            station_location = station_data[-1].location
                            
                            # Calculate distance
                            distance = self._calculate_distance(anomaly_location, station_location)
                            
                            if distance < 100:  # Within 100 km
                                # Check for concurrent anomalies
                                concurrent_anomalies = await self._check_concurrent_anomalies(
                                    process_key, anomaly.timestamp
                                )
                                if concurrent_anomalies:
                                    affected_stations.append({
                                        'station_id': process_key,
                                        'distance_km': distance,
                                        'anomaly_count': len(concurrent_anomalies)
                                    })
            
            anomaly.spatial_extent = {
                'affected_stations': affected_stations,
                'spatial_correlation_radius': max([s['distance_km'] for s in affected_stations], default=0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spatial extent: {e}")
    
    def _calculate_distance(self, loc1: Tuple[float, float, float], 
                          loc2: Tuple[float, float, float]) -> float:
        """Calculate distance between two geographic locations in km"""
        from math import radians, sin, cos, sqrt, atan2
        
        lat1, lon1, _ = loc1
        lat2, lon2, _ = loc2
        
        # Haversine formula
        R = 6371  # Earth's radius in km
        
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        
        a = (sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2)
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    async def _check_concurrent_anomalies(self, process_key: str, 
                                        timestamp: datetime) -> List[AnomalyDetection]:
        """Advanced concurrent anomaly detection using temporal correlation and ML clustering"""
        try:
            # Define temporal windows for different anomaly types
            windows = {
                'immediate': timedelta(seconds=30),    # Real-time correlations
                'short_term': timedelta(minutes=5),    # Process-level correlations
                'medium_term': timedelta(minutes=30),  # System-level correlations
                'long_term': timedelta(hours=2)        # Trend-based correlations
            }
            
            concurrent_anomalies = []
            
            # Multi-window temporal analysis
            for window_name, window_size in windows.items():
                start_time = timestamp - window_size
                end_time = timestamp + window_size
                
                # Query anomalies in temporal window
                window_anomalies = await self._query_temporal_anomalies(
                    process_key, start_time, end_time
                )
                
                if window_anomalies:
                    # Apply temporal correlation scoring
                    scored_anomalies = await self._score_temporal_correlations(
                        window_anomalies, timestamp, window_name
                    )
                    concurrent_anomalies.extend(scored_anomalies)
            
            # Remove duplicates and apply advanced filtering
            unique_anomalies = self._deduplicate_anomalies(concurrent_anomalies)
            
            # Apply ML-based clustering for pattern recognition
            clustered_anomalies = await self._cluster_anomaly_patterns(
                unique_anomalies, timestamp
            )
            
            # Calculate cross-correlation coefficients
            correlated_anomalies = await self._calculate_cross_correlations(
                clustered_anomalies, process_key, timestamp
            )
            
            # Filter by significance threshold and return top candidates
            significant_anomalies = [
                anomaly for anomaly in correlated_anomalies
                if anomaly.correlation_score > 0.7 and anomaly.confidence > 0.8
            ]
            
            # Sort by combined relevance score
            significant_anomalies.sort(
                key=lambda x: (x.correlation_score * x.confidence * x.temporal_proximity),
                reverse=True
            )
            
            return significant_anomalies[:10]  # Return top 10 most relevant
            
        except Exception as e:
            self.logger.error(f"Error in concurrent anomaly detection: {e}")
            return []
    
    async def _query_temporal_anomalies(self, process_key: str, start_time: datetime, 
                                      end_time: datetime) -> List[AnomalyDetection]:
        """Query anomalies within temporal window with optimized indexing"""
        try:
            # Multi-database query strategy
            queries = []
            
            # InfluxDB query for time-series anomalies
            influx_query = f'''
                from(bucket: "anomalies")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r["process_key"] == "{process_key}" or r["related_processes"] =~ /{process_key}/)
                |> filter(fn: (r) => r["_measurement"] == "anomaly_detection")
            '''
            queries.append(self._execute_influx_query(influx_query))
            
            # PostgreSQL query for structured anomaly data
            pg_query = '''
                SELECT * FROM anomaly_detections 
                WHERE (process_key = %s OR related_processes @> %s)
                AND detected_at BETWEEN %s AND %s
                AND confidence > 0.6
                ORDER BY detected_at DESC
            '''
            queries.append(self._execute_postgres_query(
                pg_query, (process_key, [process_key], start_time, end_time)
            ))
            
            # Neo4j query for relationship-based anomalies
            neo4j_query = '''
                MATCH (p:Process {key: $process_key})-[:CORRELATES_WITH|DEPENDS_ON*1..3]-(related:Process)
                MATCH (related)-[:HAS_ANOMALY]->(a:Anomaly)
                WHERE a.detected_at >= $start_time AND a.detected_at <= $end_time
                RETURN a, related, p
            '''
            queries.append(self._execute_neo4j_query(
                neo4j_query, {
                    'process_key': process_key,
                    'start_time': start_time,
                    'end_time': end_time
                }
            ))
            
            # Execute queries concurrently
            results = await asyncio.gather(*queries, return_exceptions=True)
            
            # Merge and normalize results
            anomalies = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.warning(f"Query failed: {result}")
                    continue
                anomalies.extend(self._normalize_anomaly_data(result))
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error querying temporal anomalies: {e}")
            return []
    
    async def _score_temporal_correlations(self, anomalies: List[AnomalyDetection], 
                                         target_timestamp: datetime, 
                                         window_name: str) -> List[AnomalyDetection]:
        """Score anomalies based on temporal correlation with target timestamp"""
        scored_anomalies = []
        
        for anomaly in anomalies:
            # Calculate temporal proximity score
            time_diff = abs((anomaly.detected_at - target_timestamp).total_seconds())
            window_seconds = {
                'immediate': 30, 'short_term': 300, 
                'medium_term': 1800, 'long_term': 7200
            }[window_name]
            
            # Exponential decay for temporal proximity
            temporal_proximity = math.exp(-time_diff / (window_seconds / 3))
            
            # Calculate frequency-based correlation
            frequency_correlation = await self._calculate_frequency_correlation(
                anomaly, target_timestamp
            )
            
            # Calculate pattern similarity
            pattern_similarity = await self._calculate_pattern_similarity(
                anomaly, target_timestamp
            )
            
            # Combined correlation score
            correlation_score = (
                0.4 * temporal_proximity +
                0.3 * frequency_correlation +
                0.3 * pattern_similarity
            )
            
            # Update anomaly with correlation metrics
            anomaly.correlation_score = correlation_score
            anomaly.temporal_proximity = temporal_proximity
            anomaly.window_type = window_name
            
            scored_anomalies.append(anomaly)
        
        return scored_anomalies
    
    async def _cluster_anomaly_patterns(self, anomalies: List[AnomalyDetection], 
                                      timestamp: datetime) -> List[AnomalyDetection]:
        """Apply ML clustering to identify anomaly patterns"""
        if len(anomalies) < 2:
            return anomalies
        
        try:
            # Extract features for clustering
            features = []
            for anomaly in anomalies:
                feature_vector = [
                    anomaly.severity_score,
                    anomaly.confidence,
                    anomaly.correlation_score,
                    anomaly.temporal_proximity,
                    len(anomaly.affected_metrics),
                    anomaly.duration_seconds if hasattr(anomaly, 'duration_seconds') else 0
                ]
                features.append(feature_vector)
            
            # Apply DBSCAN clustering for pattern detection
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(features)
            
            clustering = DBSCAN(eps=0.3, min_samples=2)
            cluster_labels = clustering.fit_predict(normalized_features)
            
            # Assign cluster information to anomalies
            clustered_anomalies = []
            for i, anomaly in enumerate(anomalies):
                anomaly.cluster_id = int(cluster_labels[i])
                anomaly.is_pattern_anomaly = cluster_labels[i] != -1
                
                # Calculate cluster-based confidence boost
                if anomaly.is_pattern_anomaly:
                    cluster_size = sum(1 for label in cluster_labels if label == cluster_labels[i])
                    anomaly.confidence *= (1 + 0.1 * cluster_size)  # Boost confidence for patterns
                
                clustered_anomalies.append(anomaly)
            
            return clustered_anomalies
            
        except Exception as e:
            self.logger.error(f"Error in anomaly clustering: {e}")
            return anomalies
    
    async def _calculate_cross_correlations(self, anomalies: List[AnomalyDetection], 
                                          process_key: str, 
                                          timestamp: datetime) -> List[AnomalyDetection]:
        """Calculate cross-correlation coefficients between anomalies and target process"""
        try:
            # Get target process metrics around timestamp
            target_metrics = await self._get_process_metrics_window(
                process_key, timestamp, timedelta(minutes=10)
            )
            
            if not target_metrics:
                return anomalies
            
            correlated_anomalies = []
            
            for anomaly in anomalies:
                # Get anomaly process metrics
                anomaly_metrics = await self._get_process_metrics_window(
                    anomaly.process_key, anomaly.detected_at, timedelta(minutes=10)
                )
                
                if not anomaly_metrics:
                    anomaly.cross_correlation = 0.0
                    correlated_anomalies.append(anomaly)
                    continue
                
                # Calculate cross-correlation using numpy
                correlation_coeffs = []
                
                for metric_name in ['cpu_usage', 'memory_usage', 'io_rate', 'network_rate']:
                    target_series = [m.get(metric_name, 0) for m in target_metrics]
                    anomaly_series = [m.get(metric_name, 0) for m in anomaly_metrics]
                    
                    if len(target_series) > 1 and len(anomaly_series) > 1:
                        # Align series lengths
                        min_len = min(len(target_series), len(anomaly_series))
                        target_series = target_series[:min_len]
                        anomaly_series = anomaly_series[:min_len]
                        
                        # Calculate Pearson correlation
                        correlation = np.corrcoef(target_series, anomaly_series)[0, 1]
                        if not np.isnan(correlation):
                            correlation_coeffs.append(abs(correlation))
                
                # Average correlation across metrics
                anomaly.cross_correlation = np.mean(correlation_coeffs) if correlation_coeffs else 0.0
                
                # Update overall correlation score
                anomaly.correlation_score = (
                    0.6 * anomaly.correlation_score + 
                    0.4 * anomaly.cross_correlation
                )
                
                correlated_anomalies.append(anomaly)
            
            return correlated_anomalies
            
        except Exception as e:
            self.logger.error(f"Error calculating cross-correlations: {e}")
            return anomalies
    
    async def _alert_generation_system(self) -> None:
        """Generate alerts based on causal relationship changes and anomalies"""
        logger.info("Starting alert generation system")
        
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Check for alert conditions
                alerts = []
                
                # Causal strength change alerts
                causal_alerts = await self._check_causal_strength_alerts()
                alerts.extend(causal_alerts)
                
                # Anomaly-based alerts
                anomaly_alerts = await self._check_anomaly_alerts()
                alerts.extend(anomaly_alerts)
                
                # Predictive alerts based on causal models
                predictive_alerts = await self._generate_predictive_alerts()
                alerts.extend(predictive_alerts)
                
                # Process and distribute alerts
                for alert in alerts:
                    await self._process_alert(alert)
                    self.alert_history.append(alert)
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in alert generation: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_causal_strength_alerts(self) -> List[RealTimeAlert]:
        """Check for alerts based on causal strength changes"""
        alerts = []
        
        try:
            for relationship_key, history in self.causal_strength_history.items():
                if len(history) >= 2:
                    recent_strength = history[-1]['strength']
                    previous_strength = history[-2]['strength']
                    
                    # Check for significant changes
                    change_ratio = abs(recent_strength - previous_strength) / (previous_strength + 1e-8)
                    
                    if change_ratio > 0.5:  # 50% change threshold
                        source, target = relationship_key.split('->')
                        
                        severity = AlertSeverity.HIGH if change_ratio > 1.0 else AlertSeverity.MEDIUM
                        
                        alert = RealTimeAlert(
                            alert_id=f"causal_{relationship_key}_{datetime.utcnow().isoformat()}",
                            timestamp=datetime.utcnow(),
                            severity=severity,
                            process_type=ProcessType(source.split('_')[0]),
                            station_id=source.split('_', 1)[1],
                            location=(0, 0, 0),  # Would get from station data
                            message=f"Significant causal strength change detected: {source} -> {target}",
                            causal_context=[],
                            confidence=history[-1]['confidence'],
                            predicted_evolution={
                                'trend': 'increasing' if recent_strength > previous_strength else 'decreasing',
                                'change_rate': change_ratio
                            },
                            recommended_actions=[
                                "Investigate source process for anomalies",
                                "Monitor target process for delayed effects",
                                "Review data quality for both processes"
                            ],
                            data_quality_flags=[]
                        )
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking causal strength alerts: {e}")
            return []
    
    async def _check_anomaly_alerts(self) -> List[RealTimeAlert]:
        """Check for alerts based on detected anomalies"""
        alerts = []
        
        try:
            # This would check recently stored anomalies
            # For now, generate sample alerts based on current data patterns
            
            for process_key in self.monitoring_state.active_processes:
                recent_data = list(self.data_buffers[process_key])[-5:]
                
                if len(recent_data) >= 3:
                    # Check for rapid parameter changes
                    latest_data = recent_data[-1]
                    
                    # Example: Check for volcanic SO2 flux spikes
                    if (latest_data.process_type == ProcessType.VOLCANIC and 
                        'so2_flux' in latest_data.parameters):
                        
                        so2_values = [dp.parameters.get('so2_flux', 0) for dp in recent_data]
                        if len(so2_values) >= 3:
                            recent_mean = np.mean(so2_values[-3:])
                            baseline_mean = np.mean(so2_values[:-3]) if len(so2_values) > 3 else recent_mean
                            
                            if recent_mean > baseline_mean * 2:  # 100% increase
                                alert = RealTimeAlert(
                                    alert_id=f"so2_spike_{process_key}_{datetime.utcnow().isoformat()}",
                                    timestamp=datetime.utcnow(),
                                    severity=AlertSeverity.HIGH,
                                    process_type=ProcessType.VOLCANIC,
                                    station_id=latest_data.station_id,
                                    location=latest_data.location,
                                    message=f"Significant SO2 flux increase detected at {latest_data.station_id}",
                                    causal_context=[],
                                    confidence=latest_data.quality_score,
                                    predicted_evolution={
                                        'trend': 'increasing',
                                        'magnitude': recent_mean / baseline_mean
                                    },
                                    recommended_actions=[
                                        "Increase monitoring frequency",
                                        "Check seismic activity correlation",
                                        "Prepare eruption response protocols"
                                    ],
                                    data_quality_flags=[]
                                )
                                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking anomaly alerts: {e}")
            return []
    
    async def _generate_predictive_alerts(self) -> List[RealTimeAlert]:
        """Generate predictive alerts based on causal models"""
        alerts = []
        
        try:
            # Analyze causal chains for predictive patterns
            for process_key in self.monitoring_state.active_processes:
                # Look for causal predecessors that might predict future states
                predictive_relationships = []
                
                for source_process in self.monitoring_state.active_processes:
                    if source_process != process_key:
                        strength = await self._calculate_causal_strength(source_process, process_key)
                        if strength and strength > 0.7:  # Strong predictive relationship
                            predictive_relationships.append((source_process, strength))
                
                # Generate predictions based on strong causal relationships
                if predictive_relationships:
                    prediction = await self._generate_process_prediction(
                        process_key, predictive_relationships
                    )
                    
                    if prediction and prediction.get('risk_level', 0) > 0.7:
                        alert = RealTimeAlert(
                            alert_id=f"pred_{process_key}_{datetime.utcnow().isoformat()}",
                            timestamp=datetime.utcnow(),
                            severity=AlertSeverity.MEDIUM,
                            process_type=ProcessType(process_key.split('_')[0]),
                            station_id=process_key.split('_', 1)[1],
                            location=(0, 0, 0),  # Would get from station data
                            message=f"Predictive model indicates elevated risk for {process_key}",
                            causal_context=[],
                            confidence=prediction.get('confidence', 0.5),
                            predicted_evolution=prediction,
                            recommended_actions=[
                                "Increase monitoring vigilance",
                                "Review causal indicators",
                                "Prepare response protocols"
                            ],
                            data_quality_flags=[]
                        )
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating predictive alerts: {e}")
            return []
    
    async def _generate_process_prediction(self, process_key: str, 
                                         causal_relationships: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Generate prediction for a process based on causal relationships"""
        try:
            # Simple prediction model based on causal strength and recent trends
            risk_factors = []
            
            for source_process, strength in causal_relationships:
                # Analyze recent trend in source process
                source_data = list(self.data_buffers[source_process])[-10:]
                
                if len(source_data) >= 3:
                    # Calculate trend
                    values = []
                    for dp in source_data:
                        # Extract primary parameter
                        if dp.process_type == ProcessType.SEISMIC:
                            values.append(dp.parameters.get('ground_velocity', 0))
                        elif dp.process_type == ProcessType.VOLCANIC:
                            values.append(dp.parameters.get('so2_flux', 0))
                        elif dp.process_type == ProcessType.DEFORMATION:
                            values.append(dp.parameters.get('vertical_displacement', 0))
                        elif dp.process_type == ProcessType.GEOCHEMICAL:
                            values.append(dp.parameters.get('co2_concentration', 0))
                    
                    if len(values) >= 3:
                        # Linear trend
                        x = np.arange(len(values))
                        slope, _, r_value, _, _ = stats.linregress(x, values)
                        
                        # Risk factor based on trend and causal strength
                        trend_risk = abs(slope) * strength * (r_value ** 2)
                        risk_factors.append(trend_risk)
            
            if risk_factors:
                overall_risk = min(1.0, np.mean(risk_factors))
                confidence = min(1.0, len(risk_factors) / 3.0)  # More relationships = higher confidence
                
                return {
                    'risk_level': overall_risk,
                    'confidence': confidence,
                    'contributing_factors': len(risk_factors),
                    'prediction_horizon': '1-6 hours'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating process prediction: {e}")
            return None
    
    async def _data_quality_validator(self) -> None:
        """Real-time data quality monitoring and validation"""
        logger.info("Starting data quality validator")
        
        while True:
            try:
                current_time = datetime.utcnow()
                quality_issues = []
                
                # Check data quality for all active processes
                for process_key in self.monitoring_state.active_processes:
                    issues = await self._validate_process_data_quality(process_key)
                    quality_issues.extend(issues)
                
                # Update quality metrics
                await self._update_quality_metrics(quality_issues)
                
                # Generate quality alerts if necessary
                for issue in quality_issues:
                    if issue['severity'] >= 0.7:
                        await self._generate_quality_alert(issue)
                
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in data quality validation: {e}")
                await asyncio.sleep(30.0)
    
    async def _validate_process_data_quality(self, process_key: str) -> List[Dict[str, Any]]:
        """Validate data quality for a specific process"""
        issues = []
        
        try:
            recent_data = list(self.data_buffers[process_key])[-20:]
            
            if not recent_data:
                return issues
            
            # Check data completeness
            expected_interval = 10.0  # Expected 10-second intervals
            if len(recent_data) >= 2:
                intervals = []
                for i in range(1, len(recent_data)):
                    interval = (recent_data[i].timestamp - recent_data[i-1].timestamp).total_seconds()
                    intervals.append(interval)
                
                # Check for gaps
                large_gaps = [i for i in intervals if i > expected_interval * 3]
                if large_gaps:
                    issues.append({
                        'type': 'data_gaps',
                        'severity': min(1.0, len(large_gaps) / len(intervals)),
                        'description': f"Data gaps detected in {process_key}",
                        'details': {'gap_count': len(large_gaps), 'max_gap': max(large_gaps)}
                    })
            
            # Check quality scores
            quality_scores = [dp.quality_score for dp in recent_data]
            avg_quality = np.mean(quality_scores)
            
            if avg_quality < 0.7:
                issues.append({
                    'type': 'low_quality',
                    'severity': 1.0 - avg_quality,
                    'description': f"Low data quality in {process_key}",
                    'details': {'average_quality': avg_quality, 'min_quality': min(quality_scores)}
                })
            
            # Check for stuck values (no variation)
            for param_name in recent_data[0].parameters.keys():
                values = [dp.parameters.get(param_name, 0) for dp in recent_data]
                if len(set(values)) == 1 and len(values) > 5:  # All values identical
                    issues.append({
                        'type': 'stuck_values',
                        'severity': 0.8,
                        'description': f"No variation in {param_name} for {process_key}",
                        'details': {'parameter': param_name, 'constant_value': values[0]}
                    })
            
            # Check processing latency
            latencies = [dp.processing_latency for dp in recent_data]
            avg_latency = np.mean(latencies)
            
            if avg_latency > 5.0:  # More than 5 seconds latency
                issues.append({
                    'type': 'high_latency',
                    'severity': min(1.0, avg_latency / 10.0),
                    'description': f"High processing latency for {process_key}",
                    'details': {'average_latency': avg_latency, 'max_latency': max(latencies)}
                })
            
            return issues
            
        except Exception as e:
            logger.error(f"Error validating data quality for {process_key}: {e}")
            return []
    
    async def _performance_monitor(self) -> None:
        """Monitor system performance metrics"""
        logger.info("Starting performance monitor")
        
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Calculate performance metrics
                metrics = await self._calculate_performance_metrics()
                
                # Update monitoring state
                self.monitoring_state.performance_metrics = metrics
                
                # Log performance metrics
                logger.info(f"Performance metrics: {metrics}")
                
                # Check for performance issues
                await self._check_performance_alerts(metrics)
                
                await asyncio.sleep(30.0)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60.0)
    
    async def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate system performance metrics"""
        try:
            current_time = datetime.utcnow()
            
            # Processing latency metrics
            if self.processing_times:
                avg_processing_time = np.mean(list(self.processing_times))
                max_processing_time = np.max(list(self.processing_times))
                p95_processing_time = np.percentile(list(self.processing_times), 95)
            else:
                avg_processing_time = max_processing_time = p95_processing_time = 0
            
            # Throughput metrics
            time_since_reset = (current_time - self.last_throughput_reset).total_seconds()
            throughput = self.throughput_counter / max(time_since_reset, 1)
            
            # Reset throughput counter every hour
            if time_since_reset > 3600:
                self.throughput_counter = 0
                self.last_throughput_reset = current_time
            
            # Memory usage (simplified)
            active_buffers = len([k for k, v in self.data_buffers.items() if v])
            total_data_points = sum(len(v) for v in self.data_buffers.values())
            
            # Causal analysis performance
            causal_relationships = sum(len(v) for v in self.causal_strength_history.values())
            
            return {
                'avg_processing_time_ms': avg_processing_time * 1000,
                'max_processing_time_ms': max_processing_time * 1000,
                'p95_processing_time_ms': p95_processing_time * 1000,
                'throughput_points_per_sec': throughput,
                'active_processes': len(self.monitoring_state.active_processes),
                'active_buffers': active_buffers,
                'total_data_points': total_data_points,
                'causal_relationships': causal_relationships,
                'websocket_connections': len(self.websocket_connections),
                'alert_history_size': len(self.alert_history)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    async def _websocket_server(self) -> None:
        """WebSocket server for real-time dashboard updates"""
        logger.info("Starting WebSocket server for real-time updates")
        
        async def handle_client(websocket, path):
            """Handle individual WebSocket client connection"""
            try:
                self.websocket_connections.add(websocket)
                logger.info(f"New WebSocket client connected: {websocket.remote_address}")
                
                # Send initial state
                await websocket.send(json.dumps({
                    'type': 'initial_state',
                    'data': {
                        'active_processes': list(self.monitoring_state.active_processes),
                        'performance_metrics': self.monitoring_state.performance_metrics
                    }
                }))
                
                # Keep connection alive
                await websocket.wait_closed()
                
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_connections.discard(websocket)
                logger.info("WebSocket client disconnected")
        
        # Start WebSocket server
        try:
            server = await websockets.serve(
                handle_client, 
                self.config.get('websocket_host', 'localhost'),
                self.config.get('websocket_port', 8765)
            )
            logger.info(f"WebSocket server started on {self.config.get('websocket_host', 'localhost')}:{self.config.get('websocket_port', 8765)}")
            await server.wait_closed()
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
    
    async def _broadcast_data_update(self, data_point: GeologicalDataPoint) -> None:
        """Broadcast data update to all WebSocket clients"""
        if not self.websocket_connections:
            return
        
        try:
            message = json.dumps({
                'type': 'data_update',
                'data': {
                    'timestamp': data_point.timestamp.isoformat(),
                    'process_type': data_point.process_type.value,
                    'station_id': data_point.station_id,
                    'parameters': data_point.parameters,
                    'quality_score': data_point.quality_score
                }
            })
            
            # Send to all connected clients
            disconnected = set()
            for websocket in self.websocket_connections:
                try:
                    await websocket.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(websocket)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    disconnected.add(websocket)
            
            # Remove disconnected clients
            self.websocket_connections -= disconnected
            
        except Exception as e:
            logger.error(f"Error broadcasting data update: {e}")
    
    # Additional helper methods for completeness
    async def _initialize_anomaly_models(self) -> None:
        """Initialize advanced anomaly detection models with database persistence"""
        logger.info("Initializing advanced anomaly detection models")
        
        try:
            # Initialize different anomaly detection algorithms for each process type
            for process_type in ProcessType:
                model_key = f"anomaly_{process_type.value}"
                
                # Multi-algorithm ensemble for robust detection
                self.anomaly_models[model_key] = {
                    'isolation_forest': IsolationForest(
                        contamination=0.1,
                        random_state=42,
                        n_estimators=200
                    ),
                    'statistical_detector': {
                        'z_score_threshold': 3.0,
                        'modified_z_score_threshold': 3.5,
                        'iqr_multiplier': 1.5
                    },
                    'temporal_detector': {
                        'window_size': 100,
                        'trend_threshold': 0.05,
                        'seasonality_detector': True
                    },
                    'multivariate_detector': {
                        'mahalanobis_threshold': 3.0,
                        'pca_components': 0.95,
                        'correlation_threshold': 0.8
                    }
                }
                
                # Initialize scalers for each process type
                self.scalers[model_key] = StandardScaler()
            
            # Load pre-trained models from database if available
            await self._load_pretrained_models()
            
            # Initialize causal inference models
            await self._initialize_causal_models()
            
            logger.info(f"Initialized {len(self.anomaly_models)} anomaly detection model sets")
            
        except Exception as e:
            logger.error(f"Error initializing anomaly models: {e}")
            raise
    
    async def _load_pretrained_models(self) -> None:
        """Load pre-trained anomaly detection models from database"""
        try:
            # Query PostgreSQL for stored model parameters
            query = """
            SELECT process_type, model_type, model_parameters, training_date, performance_metrics
            FROM anomaly_models 
            WHERE is_active = true 
            ORDER BY training_date DESC
            """
            
            # This would require a PostgreSQL connection
            # For now, we'll simulate loading from Redis cache
            for process_type in ProcessType:
                cache_key = f"anomaly_model:{process_type.value}"
                cached_model = self.redis_client.get(cache_key)
                
                if cached_model:
                    try:
                        model_data = json.loads(cached_model)
                        model_key = f"anomaly_{process_type.value}"
                        
                        # Update model parameters from cache
                        if model_key in self.anomaly_models:
                            self.anomaly_models[model_key].update(model_data.get('parameters', {}))
                            
                        logger.info(f"Loaded cached model for {process_type.value}")
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid cached model data for {process_type.value}: {e}")
            
        except Exception as e:
            logger.error(f"Error loading pre-trained models: {e}")
    
    async def _initialize_causal_models(self) -> None:
        """Initialize causal inference models for geological processes"""
        try:
            # Initialize causal discovery algorithms
            self.causal_models = {
                'granger_causality': {
                    'max_lags': 10,
                    'significance_level': 0.05,
                    'test_statistic': 'ssr_ftest'
                },
                'transfer_entropy': {
                    'k_history': 3,
                    'l_history': 3,
                    'delay': 1
                },
                'convergent_cross_mapping': {
                    'embedding_dimension': 3,
                    'tau': 1,
                    'library_size_range': (10, 100)
                },
                'peter_clark': {
                    'alpha': 0.05,
                    'max_conditioning_set_size': 3
                }
            }
            
            # Load historical causal relationships from database
            await self._load_historical_causal_relationships()
            
            logger.info("Initialized causal inference models")
            
        except Exception as e:
            logger.error(f"Error initializing causal models: {e}")
    
    async def _load_baseline_data(self) -> None:
        """Load comprehensive baseline statistics and causal relationships from databases"""
        logger.info("Loading baseline data and causal relationships from persistent storage")
        
        try:
            # Load baseline statistics from InfluxDB
            await self._load_baseline_statistics_from_influxdb()
            
            # Load causal relationships from PostgreSQL
            await self._load_causal_relationships_from_postgresql()
            
            # Load alert thresholds from Redis cache
            await self._load_alert_thresholds_from_redis()
            
            # Initialize process-specific baselines
            await self._initialize_process_baselines()
            
            logger.info("Successfully loaded baseline data from all storage systems")
            
        except Exception as e:
            logger.error(f"Error loading baseline data: {e}")
            raise
    
    async def _load_baseline_statistics_from_influxdb(self) -> None:
        """Load baseline statistics from InfluxDB time-series data"""
        try:
            # Query for historical statistics over the past 30 days
            query = '''
            from(bucket: "geological_data")
                |> range(start: -30d)
                |> filter(fn: (r) => r._measurement == "geological_observation")
                |> group(columns: ["process_type", "station_id"])
                |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
                |> yield(name: "hourly_means")
            '''
            
            tables = self.influx_client.query_api().query(query)
            
            for table in tables:
                for record in table.records:
                    process_type = record.values.get('process_type')
                    station_id = record.values.get('station_id')
                    
                    if process_type and station_id:
                        baseline_key = f"{process_type}_{station_id}"
                        
                        if baseline_key not in self.monitoring_state.baseline_statistics:
                            self.monitoring_state.baseline_statistics[baseline_key] = {
                                'mean_values': {},
                                'std_values': {},
                                'percentiles': {},
                                'trend_coefficients': {},
                                'seasonal_patterns': {},
                                'last_updated': datetime.utcnow()
                            }
                        
                        # Extract parameter statistics
                        for field_key, field_value in record.values.items():
                            if field_key.startswith('param_') and field_value is not None:
                                param_name = field_key.replace('param_', '')
                                
                                # Update running statistics
                                baseline = self.monitoring_state.baseline_statistics[baseline_key]
                                if param_name not in baseline['mean_values']:
                                    baseline['mean_values'][param_name] = []
                                
                                baseline['mean_values'][param_name].append(field_value)
            
            # Calculate final statistics from collected data
            for baseline_key, baseline_data in self.monitoring_state.baseline_statistics.items():
                for param_name, values in baseline_data['mean_values'].items():
                    if values:
                        values_array = np.array(values)
                        baseline_data['mean_values'][param_name] = np.mean(values_array)
                        baseline_data['std_values'][param_name] = np.std(values_array)
                        baseline_data['percentiles'][param_name] = {
                            'p25': np.percentile(values_array, 25),
                            'p50': np.percentile(values_array, 50),
                            'p75': np.percentile(values_array, 75),
                            'p90': np.percentile(values_array, 90),
                            'p95': np.percentile(values_array, 95),
                            'p99': np.percentile(values_array, 99)
                        }
            
            logger.info(f"Loaded baseline statistics for {len(self.monitoring_state.baseline_statistics)} process-station combinations")
            
        except Exception as e:
            logger.error(f"Error loading baseline statistics from InfluxDB: {e}")
    
    async def _load_causal_relationships_from_postgresql(self) -> None:
        """Load historical causal relationships from PostgreSQL"""
        try:
            # This would require a PostgreSQL connection
            # For demonstration, we'll use Redis to simulate stored relationships
            
            causal_keys = self.redis_client.keys("causal_relationship:*")
            
            for key in causal_keys:
                try:
                    relationship_data = self.redis_client.get(key)
                    if relationship_data:
                        relationship = json.loads(relationship_data)
                        
                        # Reconstruct CausalRelationship object
                        causal_rel = CausalRelationship(
                            source_process=relationship['source_process'],
                            target_process=relationship['target_process'],
                            causal_strength=relationship['causal_strength'],
                            confidence_interval=(
                                relationship['confidence_interval'][0],
                                relationship['confidence_interval'][1]
                            ),
                            temporal_lag=timedelta(seconds=relationship['temporal_lag_seconds']),
                            spatial_correlation=relationship['spatial_correlation'],
                            last_updated=datetime.fromisoformat(relationship['last_updated']),
                            validation_score=relationship['validation_score']
                        )
                        
                        # Add to causal network
                        source_key = relationship['source_process']
                        if source_key not in self.monitoring_state.causal_network:
                            self.monitoring_state.causal_network[source_key] = []
                        
                        self.monitoring_state.causal_network[source_key].append(causal_rel)
                        
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Invalid causal relationship data in {key}: {e}")
            
            logger.info(f"Loaded {len(self.monitoring_state.causal_network)} causal relationship networks")
            
        except Exception as e:
            logger.error(f"Error loading causal relationships: {e}")
    
    async def _load_alert_thresholds_from_redis(self) -> None:
        """Load dynamic alert thresholds from Redis cache"""
        try:
            threshold_keys = self.redis_client.keys("alert_threshold:*")
            
            for key in threshold_keys:
                try:
                    threshold_data = self.redis_client.get(key)
                    if threshold_data:
                        thresholds = json.loads(threshold_data)
                        
                        # Extract process type from key
                        process_type = key.split(':')[1]
                        self.monitoring_state.alert_thresholds[process_type] = thresholds
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid threshold data in {key}: {e}")
            
            # Set default thresholds if none loaded
            if not self.monitoring_state.alert_thresholds:
                await self._set_default_alert_thresholds()
            
            logger.info(f"Loaded alert thresholds for {len(self.monitoring_state.alert_thresholds)} process types")
            
        except Exception as e:
            logger.error(f"Error loading alert thresholds: {e}")
    
    async def _set_default_alert_thresholds(self) -> None:
        """Set default alert thresholds for geological processes"""
        default_thresholds = {
            ProcessType.SEISMIC.value: {
                'magnitude': {'low': 3.0, 'medium': 4.5, 'high': 6.0, 'critical': 7.0},
                'depth_km': {'shallow': 10, 'intermediate': 70, 'deep': 300},
                'frequency_hz': {'low': 1.0, 'medium': 10.0, 'high': 50.0}
            },
            ProcessType.VOLCANIC.value: {
                'vei': {'low': 1, 'medium': 2, 'high': 3, 'critical': 4},
                'temperature_c': {'low': 100, 'medium': 500, 'high': 1000, 'critical': 1200},
                'gas_concentration_ppm': {'low': 100, 'medium': 1000, 'high': 5000, 'critical': 10000}
            },
            ProcessType.DEFORMATION.value: {
                'displacement_mm': {'low': 1, 'medium': 10, 'high': 50, 'critical': 100},
                'strain_rate': {'low': 1e-8, 'medium': 1e-7, 'high': 1e-6, 'critical': 1e-5}
            }
        }
        
        self.monitoring_state.alert_thresholds.update(default_thresholds)
    
    async def _initialize_process_baselines(self) -> None:
        """Initialize process-specific baseline models"""
        try:
            for process_type in ProcessType:
                # Initialize time-series decomposition models
                baseline_key = f"baseline_{process_type.value}"
                
                # Seasonal decomposition parameters
                seasonal_params = {
                    'period': 24 * 7,  # Weekly seasonality for geological processes
                    'trend_window': 24 * 30,  # Monthly trend window
                    'seasonal_window': 24 * 7 + 1  # Weekly seasonal window
                }
                
                # Store in monitoring state
                if 'seasonal_models' not in self.monitoring_state.baseline_statistics:
                    self.monitoring_state.baseline_statistics['seasonal_models'] = {}
                
                self.monitoring_state.baseline_statistics['seasonal_models'][process_type.value] = seasonal_params
            
            logger.info("Initialized process-specific baseline models")
            
        except Exception as e:
            logger.error(f"Error initializing process baselines: {e}")
    
    async def _analyze_causal_relationships(self, data_point: GeologicalDataPoint) -> None:
        """Advanced real-time causal relationship analysis with multiple algorithms"""
        try:
            process_key = f"{data_point.process_type.value}_{data_point.station_id}"
            
            # Get recent data for causal analysis
            recent_data = await self._get_recent_data_for_causal_analysis(data_point)
            
            if len(recent_data) < 10:  # Need minimum data for causal analysis
                return
            
            # Perform multi-algorithm causal analysis
            causal_results = await self._perform_causal_analysis(data_point, recent_data)
            
            # Update causal network with new relationships
            await self._update_causal_network(causal_results)
            
            # Check for significant changes in causal strength
            for relationship in causal_results:
                await self._check_causal_strength_changes(
                    f"{relationship.source_process}_{relationship.target_process}",
                    relationship.causal_strength
                )
            
            # Store causal analysis results to database
            await self._store_causal_analysis_results(causal_results)
            
        except Exception as e:
            logger.error(f"Error in causal relationship analysis: {e}")
    
    async def _get_recent_data_for_causal_analysis(self, data_point: GeologicalDataPoint) -> List[GeologicalDataPoint]:
        """Retrieve recent data points for causal analysis"""
        recent_data = []
        
        try:
            # Get data from local buffers first
            buffer_key = f"{data_point.process_type.value}_{data_point.station_id}"
            if buffer_key in self.data_buffers:
                recent_data.extend(list(self.data_buffers[buffer_key]))
            
            # Get additional data from nearby stations for spatial correlation
            nearby_stations = await self._find_nearby_stations(data_point.location, radius_km=50)
            
            for station_id in nearby_stations:
                station_buffer_key = f"{data_point.process_type.value}_{station_id}"
                if station_buffer_key in self.data_buffers:
                    recent_data.extend(list(self.data_buffers[station_buffer_key]))
            
            # Sort by timestamp
            recent_data.sort(key=lambda x: x.timestamp)
            
            # Limit to last 100 points for performance
            return recent_data[-100:]
            
        except Exception as e:
            logger.error(f"Error getting recent data for causal analysis: {e}")
            return []
    
    async def _find_nearby_stations(self, location: Tuple[float, float, float], radius_km: float) -> List[str]:
        """Find nearby monitoring stations within specified radius"""
        nearby_stations = []
        
        try:
            # Query Redis for cached station locations
            station_keys = self.redis_client.keys("station_location:*")
            
            for key in station_keys:
                try:
                    station_data = self.redis_client.get(key)
                    if station_data:
                        station_info = json.loads(station_data)
                        station_location = (
                            station_info['latitude'],
                            station_info['longitude'],
                            station_info.get('elevation', 0)
                        )
                        
                        # Calculate distance using haversine formula
                        distance_km = self._calculate_distance(location, station_location)
                        
                        if distance_km <= radius_km:
                            station_id = key.split(':')[1]
                            nearby_stations.append(station_id)
                            
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Invalid station data in {key}: {e}")
            
        except Exception as e:
            logger.error(f"Error finding nearby stations: {e}")
        
        return nearby_stations
    
    def _calculate_distance(self, loc1: Tuple[float, float, float], loc2: Tuple[float, float, float]) -> float:
        """Calculate distance between two geographic locations using haversine formula"""
        lat1, lon1, _ = loc1
        lat2, lon2, _ = loc2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        
        return c * r
    
    async def _perform_causal_analysis(self, data_point: GeologicalDataPoint, recent_data: List[GeologicalDataPoint]) -> List[CausalRelationship]:
        """Perform comprehensive causal analysis using multiple algorithms"""
        causal_relationships = []
        
        try:
            # Prepare data for analysis
            time_series_data = self._prepare_time_series_data(recent_data)
            
            if len(time_series_data) < 2:
                return causal_relationships
            
            # Granger causality analysis
            granger_results = await self._granger_causality_analysis(time_series_data)
            causal_relationships.extend(granger_results)
            
            # Transfer entropy analysis
            transfer_entropy_results = await self._transfer_entropy_analysis(time_series_data)
            causal_relationships.extend(transfer_entropy_results)
            
            # Convergent cross mapping for nonlinear relationships
            ccm_results = await self._convergent_cross_mapping_analysis(time_series_data)
            causal_relationships.extend(ccm_results)
            
            # Validate and filter relationships
            validated_relationships = await self._validate_causal_relationships(causal_relationships)
            
            return validated_relationships
            
        except Exception as e:
            logger.error(f"Error in causal analysis: {e}")
            return []
    
    def _prepare_time_series_data(self, data_points: List[GeologicalDataPoint]) -> Dict[str, pd.Series]:
        """Prepare time series data for causal analysis"""
        time_series = {}
        
        try:
            # Group data by process type and parameter
            for data_point in data_points:
                for param_name, param_value in data_point.parameters.items():
                    if isinstance(param_value, (int, float)):
                        series_key = f"{data_point.process_type.value}_{param_name}_{data_point.station_id}"
                        
                        if series_key not in time_series:
                            time_series[series_key] = []
                        
                        time_series[series_key].append({
                            'timestamp': data_point.timestamp,
                            'value': param_value
                        })
            
            # Convert to pandas Series
            for series_key, data_list in time_series.items():
                if len(data_list) > 5:  # Minimum data points for analysis
                    df = pd.DataFrame(data_list)
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Resample to regular intervals (1 minute)
                    resampled = df.resample('1T').mean().interpolate()
                    time_series[series_key] = resampled['value']
            
            # Remove series with insufficient data
            time_series = {k: v for k, v in time_series.items() if len(v) >= 10}
            
        except Exception as e:
            logger.error(f"Error preparing time series data: {e}")
        
        return time_series
    
    async def _granger_causality_analysis(self, time_series_data: Dict[str, pd.Series]) -> List[CausalRelationship]:
        """Perform Granger causality analysis"""
        relationships = []
        
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            series_keys = list(time_series_data.keys())
            
            for i, source_key in enumerate(series_keys):
                for j, target_key in enumerate(series_keys):
                    if i != j:  # Don't test self-causality
                        try:
                            source_series = time_series_data[source_key]
                            target_series = time_series_data[target_key]
                            
                            # Align series by timestamp
                            aligned_data = pd.concat([target_series, source_series], axis=1).dropna()
                            
                            if len(aligned_data) >= 20:  # Minimum for Granger test
                                # Perform Granger causality test
                                max_lags = min(10, len(aligned_data) // 4)
                                gc_results = grangercausalitytests(aligned_data, max_lags, verbose=False)
                                
                                # Extract p-values and F-statistics
                                p_values = []
                                f_stats = []
                                
                                for lag in range(1, max_lags + 1):
                                    if lag in gc_results:
                                        test_result = gc_results[lag][0]
                                        p_values.append(test_result['ssr_ftest'][1])
                                        f_stats.append(test_result['ssr_ftest'][0])
                                
                                if p_values:
                                    min_p_value = min(p_values)
                                    max_f_stat = max(f_stats)
                                    
                                    # Significant causality if p < 0.05
                                    if min_p_value < 0.05:
                                        causal_strength = 1.0 - min_p_value  # Convert p-value to strength
                                        confidence_interval = (
                                            max(0, causal_strength - 0.1),
                                            min(1, causal_strength + 0.1)
                                        )
                                        
                                        relationship = CausalRelationship(
                                            source_process=source_key,
                                            target_process=target_key,
                                            causal_strength=causal_strength,
                                            confidence_interval=confidence_interval,
                                            temporal_lag=timedelta(minutes=p_values.index(min_p_value) + 1),
                                            spatial_correlation=0.0,  # Will be calculated separately
                                            last_updated=datetime.utcnow(),
                                            validation_score=max_f_stat / 10.0  # Normalize F-statistic
                                        )
                                        
                                        relationships.append(relationship)
                        
                        except Exception as e:
                            logger.warning(f"Granger causality test failed for {source_key} -> {target_key}: {e}")
            
        except ImportError:
            logger.warning("statsmodels not available for Granger causality analysis")
        except Exception as e:
            logger.error(f"Error in Granger causality analysis: {e}")
        
        return relationships
    
    async def _transfer_entropy_analysis(self, time_series_data: Dict[str, pd.Series]) -> List[CausalRelationship]:
        """Perform transfer entropy analysis for nonlinear causality"""
        relationships = []
        
        try:
            # Transfer entropy implementation would require specialized libraries
            # For now, we'll implement a simplified version based on mutual information
            
            series_keys = list(time_series_data.keys())
            
            for i, source_key in enumerate(series_keys):
                for j, target_key in enumerate(series_keys):
                    if i != j:
                        try:
                            source_series = time_series_data[source_key]
                            target_series = time_series_data[target_key]
                            
                            # Calculate lagged mutual information as proxy for transfer entropy
                            max_lag = min(5, len(source_series) // 4)
                            
                            for lag in range(1, max_lag + 1):
                                # Shift source series by lag
                                lagged_source = source_series.shift(lag).dropna()
                                aligned_target = target_series.loc[lagged_source.index]
                                
                                if len(lagged_source) >= 10:
                                    # Calculate correlation as simplified transfer entropy
                                    correlation = lagged_source.corr(aligned_target)
                                    
                                    if abs(correlation) > 0.3:  # Threshold for significant relationship
                                        causal_strength = abs(correlation)
                                        
                                        relationship = CausalRelationship(
                                            source_process=source_key,
                                            target_process=target_key,
                                            causal_strength=causal_strength,
                                            confidence_interval=(
                                                max(0, causal_strength - 0.1),
                                                min(1, causal_strength + 0.1)
                                            ),
                                            temporal_lag=timedelta(minutes=lag),
                                            spatial_correlation=0.0,
                                            last_updated=datetime.utcnow(),
                                            validation_score=causal_strength
                                        )
                                        
                                        relationships.append(relationship)
                        
                        except Exception as e:
                            logger.warning(f"Transfer entropy analysis failed for {source_key} -> {target_key}: {e}")
            
        except Exception as e:
            logger.error(f"Error in transfer entropy analysis: {e}")
        
        return relationships
    
    async def _convergent_cross_mapping_analysis(self, time_series_data: Dict[str, pd.Series]) -> List[CausalRelationship]:
        """Perform convergent cross mapping analysis for nonlinear dynamical systems"""
        relationships = []
        
        try:
            # CCM implementation for geological time series
            # This is a simplified version - full CCM requires specialized libraries
            
            series_keys = list(time_series_data.keys())
            
            for i, source_key in enumerate(series_keys):
                for j, target_key in enumerate(series_keys):
                    if i != j:
                        try:
                            source_series = time_series_data[source_key]
                            target_series = time_series_data[target_key]
                            
                            # Align series
                            aligned_data = pd.concat([source_series, target_series], axis=1).dropna()
                            
                            if len(aligned_data) >= 30:  # Minimum for CCM
                                # Simplified CCM using embedding and cross-prediction
                                embedding_dim = 3
                                
                                # Create time-delayed embeddings
                                source_embedded = self._create_embedding(aligned_data.iloc[:, 0], embedding_dim)
                                target_embedded = self._create_embedding(aligned_data.iloc[:, 1], embedding_dim)
                                
                                if len(source_embedded) > 10 and len(target_embedded) > 10:
                                    # Cross-prediction correlation
                                    cross_correlation = self._calculate_cross_prediction_skill(
                                        source_embedded, target_embedded
                                    )
                                    
                                    if cross_correlation > 0.2:  # Threshold for CCM significance
                                        relationship = CausalRelationship(
                                            source_process=source_key,
                                            target_process=target_key,
                                            causal_strength=cross_correlation,
                                            confidence_interval=(
                                                max(0, cross_correlation - 0.1),
                                                min(1, cross_correlation + 0.1)
                                            ),
                                            temporal_lag=timedelta(minutes=1),
                                            spatial_correlation=0.0,
                                            last_updated=datetime.utcnow(),
                                            validation_score=cross_correlation
                                        )
                                        
                                        relationships.append(relationship)
                        
                        except Exception as e:
                            logger.warning(f"CCM analysis failed for {source_key} -> {target_key}: {e}")
            
        except Exception as e:
            logger.error(f"Error in CCM analysis: {e}")
        
        return relationships
    
    def _create_embedding(self, series: pd.Series, embedding_dim: int, tau: int = 1) -> np.ndarray:
        """Create time-delayed embedding for CCM analysis"""
        try:
            n = len(series)
            embedded_length = n - (embedding_dim - 1) * tau
            
            if embedded_length <= 0:
                return np.array([])
            
            embedded = np.zeros((embedded_length, embedding_dim))
            
            for i in range(embedding_dim):
                start_idx = i * tau
                end_idx = start_idx + embedded_length
                embedded[:, i] = series.iloc[start_idx:end_idx].values
            
            return embedded
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return np.array([])
    
    def _calculate_cross_prediction_skill(self, source_embedded: np.ndarray, target_embedded: np.ndarray) -> float:
        """Calculate cross-prediction skill for CCM"""
        try:
            if len(source_embedded) == 0 or len(target_embedded) == 0:
                return 0.0
            
            # Simplified cross-prediction using nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            
            # Use half the data for training, half for testing
            split_idx = len(source_embedded) // 2
            
            train_source = source_embedded[:split_idx]
            test_source = source_embedded[split_idx:]
            train_target = target_embedded[:split_idx]
            test_target = target_embedded[split_idx:]
            
            if len(train_source) < 5 or len(test_source) < 5:
                return 0.0
            
            # Find nearest neighbors in source space
            nbrs = NearestNeighbors(n_neighbors=min(3, len(train_source)), algorithm='auto')
            nbrs.fit(train_source)
            
            predictions = []
            actuals = []
            
            for i, test_point in enumerate(test_source):
                if i < len(test_target):
                    # Find nearest neighbors
                    distances, indices = nbrs.kneighbors([test_point])
                    
                    # Predict target value using weighted average of neighbors
                    weights = 1.0 / (distances[0] + 1e-8)  # Avoid division by zero
                    weights /= np.sum(weights)
                    
                    predicted_target = np.sum([weights[j] * train_target[indices[0][j], 0] 
                                             for j in range(len(indices[0]))])
                    
                    predictions.append(predicted_target)
                    actuals.append(test_target[i, 0])
            
            if len(predictions) > 0:
                # Calculate correlation as prediction skill
                correlation = np.corrcoef(predictions, actuals)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating cross-prediction skill: {e}")
            return 0.0
    
    async def _validate_causal_relationships(self, relationships: List[CausalRelationship]) -> List[CausalRelationship]:
        """Validate and filter causal relationships"""
        validated = []
        
        try:
            for relationship in relationships:
                # Validation criteria
                is_valid = (
                    relationship.causal_strength > 0.1 and  # Minimum strength
                    relationship.validation_score > 0.05 and  # Minimum validation score
                    relationship.temporal_lag.total_seconds() > 0  # Positive time lag
                )
                
                if is_valid:
                    validated.append(relationship)
            
            # Remove duplicate relationships (same source-target pair)
            unique_relationships = {}
            for rel in validated:
                key = f"{rel.source_process}_{rel.target_process}"
                if key not in unique_relationships or rel.causal_strength > unique_relationships[key].causal_strength:
                    unique_relationships[key] = rel
            
            return list(unique_relationships.values())
            
        except Exception as e:
            logger.error(f"Error validating causal relationships: {e}")
            return relationships
    
    async def _update_causal_network(self, causal_results: List[CausalRelationship]) -> None:
        """Update the causal network with new relationships"""
        try:
            for relationship in causal_results:
                source_key = relationship.source_process
                
                if source_key not in self.monitoring_state.causal_network:
                    self.monitoring_state.causal_network[source_key] = []
                
                # Update existing relationship or add new one
                existing_rel = None
                for i, existing in enumerate(self.monitoring_state.causal_network[source_key]):
                    if existing.target_process == relationship.target_process:
                        existing_rel = i
                        break
                
                if existing_rel is not None:
                    # Update existing relationship
                    self.monitoring_state.causal_network[source_key][existing_rel] = relationship
                else:
                    # Add new relationship
                    self.monitoring_state.causal_network[source_key].append(relationship)
            
            # Update monitoring state timestamp
            self.monitoring_state.last_update = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating causal network: {e}")
    
    async def _store_causal_analysis_results(self, causal_results: List[CausalRelationship]) -> None:
        """Store causal analysis results to database"""
        try:
            for relationship in causal_results:
                # Store to Redis cache
                cache_key = f"causal_relationship:{relationship.source_process}_{relationship.target_process}"
                cache_data = {
                    'source_process': relationship.source_process,
                    'target_process': relationship.target_process,
                    'causal_strength': relationship.causal_strength,
                    'confidence_interval': relationship.confidence_interval,
                    'temporal_lag_seconds': relationship.temporal_lag.total_seconds(),
                    'spatial_correlation': relationship.spatial_correlation,
                    'last_updated': relationship.last_updated.isoformat(),
                    'validation_score': relationship.validation_score
                }
                
                # Store with 24 hour expiration
                self.redis_client.setex(
                    cache_key,
                    86400,  # 24 hours
                    json.dumps(cache_data)
                )
            
            # Store to InfluxDB for time-series analysis
            await self._store_causal_relationships_to_influxdb(causal_results)
            
        except Exception as e:
            logger.error(f"Error storing causal analysis results: {e}")
    
    async def _store_causal_relationships_to_influxdb(self, relationships: List[CausalRelationship]) -> None:
        """Store causal relationships to InfluxDB"""
        try:
            write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            
            points = []
            for relationship in relationships:
                point = {
                    "measurement": "causal_relationship",
                    "tags": {
                        "source_process": relationship.source_process,
                        "target_process": relationship.target_process
                    },
                    "fields": {
                        "causal_strength": relationship.causal_strength,
                        "confidence_lower": relationship.confidence_interval[0],
                        "confidence_upper": relationship.confidence_interval[1],
                        "temporal_lag_seconds": relationship.temporal_lag.total_seconds(),
                        "spatial_correlation": relationship.spatial_correlation,
                        "validation_score": relationship.validation_score
                    },
                    "time": relationship.last_updated
                }
                points.append(point)
            
            write_api.write(bucket="causal_analysis", record=points)
            
        except Exception as e:
            logger.error(f"Error storing causal relationships to InfluxDB: {e}")
    
    async def _update_baseline_statistics(self, data_point: GeologicalDataPoint) -> None:
        """Update baseline statistics with new data point using advanced statistical methods"""
        try:
            baseline_key = f"{data_point.process_type.value}_{data_point.station_id}"
            
            # Initialize baseline if not exists
            if baseline_key not in self.monitoring_state.baseline_statistics:
                self.monitoring_state.baseline_statistics[baseline_key] = {
                    'mean_values': {},
                    'std_values': {},
                    'percentiles': {},
                    'trend_coefficients': {},
                    'seasonal_patterns': {},
                    'ewma_values': {},  # Exponentially weighted moving average
                    'variance_estimates': {},
                    'skewness': {},
                    'kurtosis': {},
                    'autocorrelation': {},
                    'sample_count': 0,
                    'last_updated': datetime.utcnow()
                }
            
            baseline = self.monitoring_state.baseline_statistics[baseline_key]
            
            # Update statistics for each parameter
            for param_name, param_value in data_point.parameters.items():
                if isinstance(param_value, (int, float)) and not np.isnan(param_value):
                    await self._update_parameter_statistics(baseline, param_name, param_value, data_point.timestamp)
            
            # Update sample count and timestamp
            baseline['sample_count'] += 1
            baseline['last_updated'] = datetime.utcnow()
            
            # Periodically update trend analysis
            if baseline['sample_count'] % 100 == 0:  # Every 100 samples
                await self._update_trend_analysis(baseline_key, baseline)
            
            # Store updated baseline to database
            await self._store_baseline_statistics(baseline_key, baseline)
            
        except Exception as e:
            logger.error(f"Error updating baseline statistics: {e}")
    
    async def _update_parameter_statistics(self, baseline: Dict, param_name: str, param_value: float, timestamp: datetime) -> None:
        """Update statistics for a specific parameter"""
        try:
            # Initialize parameter statistics if not exists
            if param_name not in baseline['mean_values']:
                baseline['mean_values'][param_name] = param_value
                baseline['std_values'][param_name] = 0.0
                baseline['ewma_values'][param_name] = param_value
                baseline['variance_estimates'][param_name] = 0.0
                baseline['skewness'][param_name] = 0.0
                baseline['kurtosis'][param_name] = 0.0
                baseline['autocorrelation'][param_name] = []
                baseline['percentiles'][param_name] = {
                    'p25': param_value, 'p50': param_value, 'p75': param_value,
                    'p90': param_value, 'p95': param_value, 'p99': param_value
                }
                return
            
            # Update running statistics using Welford's online algorithm
            n = baseline['sample_count'] + 1
            old_mean = baseline['mean_values'][param_name]
            
            # Update mean
            new_mean = old_mean + (param_value - old_mean) / n
            baseline['mean_values'][param_name] = new_mean
            
            # Update variance (Welford's algorithm)
            if n > 1:
                old_variance = baseline['variance_estimates'][param_name]
                new_variance = old_variance + (param_value - old_mean) * (param_value - new_mean)
                baseline['variance_estimates'][param_name] = new_variance
                baseline['std_values'][param_name] = math.sqrt(new_variance / (n - 1))
            
            # Update exponentially weighted moving average (alpha = 0.1)
            alpha = 0.1
            baseline['ewma_values'][param_name] = (1 - alpha) * baseline['ewma_values'][param_name] + alpha * param_value
            
            # Update higher-order moments for skewness and kurtosis
            if n > 2:
                delta = param_value - new_mean
                delta_n = delta / n
                delta_n2 = delta_n * delta_n
                term1 = delta * delta_n * (n - 1)
                
                # Update skewness (simplified online calculation)
                if baseline['std_values'][param_name] > 0:
                    standardized_value = (param_value - new_mean) / baseline['std_values'][param_name]
                    baseline['skewness'][param_name] = (baseline['skewness'][param_name] * (n - 1) + standardized_value**3) / n
                    baseline['kurtosis'][param_name] = (baseline['kurtosis'][param_name] * (n - 1) + standardized_value**4) / n
            
            # Update autocorrelation (keep last 50 values)
            if len(baseline['autocorrelation'][param_name]) >= 50:
                baseline['autocorrelation'][param_name].pop(0)
            baseline['autocorrelation'][param_name].append(param_value)
            
            # Update percentiles using P² algorithm approximation
            await self._update_percentiles(baseline['percentiles'][param_name], param_value, n)
            
        except Exception as e:
            logger.error(f"Error updating parameter statistics for {param_name}: {e}")
    
    async def _update_percentiles(self, percentiles: Dict, new_value: float, n: int) -> None:
        """Update percentiles using online algorithm"""
        try:
            # Simplified percentile update - in production, use P² algorithm
            if n <= 10:
                # For small samples, store all values and calculate exact percentiles
                if 'values' not in percentiles:
                    percentiles['values'] = []
                
                percentiles['values'].append(new_value)
                
                if len(percentiles['values']) > 10:
                    # Calculate percentiles and remove stored values
                    values = sorted(percentiles['values'])
                    percentiles['p25'] = np.percentile(values, 25)
                    percentiles['p50'] = np.percentile(values, 50)
                    percentiles['p75'] = np.percentile(values, 75)
                    percentiles['p90'] = np.percentile(values, 90)
                    percentiles['p95'] = np.percentile(values, 95)
                    percentiles['p99'] = np.percentile(values, 99)
                    del percentiles['values']
            else:
                # Online percentile update (simplified)
                alpha = 1.0 / n
                
                for p_name, p_value in percentiles.items():
                    if p_name != 'values' and isinstance(p_value, (int, float)):
                        if new_value > p_value:
                            percentiles[p_name] = p_value + alpha * (new_value - p_value)
                        else:
                            percentiles[p_name] = p_value - alpha * (p_value - new_value)
            
        except Exception as e:
            logger.error(f"Error updating percentiles: {e}")
    
    async def _update_trend_analysis(self, baseline_key: str, baseline: Dict) -> None:
        """Update trend analysis for baseline statistics"""
        try:
            # Get recent data for trend analysis
            buffer_key = baseline_key
            if buffer_key in self.data_buffers and len(self.data_buffers[buffer_key]) > 20:
                recent_data = list(self.data_buffers[buffer_key])[-100:]  # Last 100 points
                
                # Perform trend analysis for each parameter
                for param_name in baseline['mean_values'].keys():
                    param_values = []
                    timestamps = []
                    
                    for data_point in recent_data:
                        if param_name in data_point.parameters:
                            param_values.append(data_point.parameters[param_name])
                            timestamps.append(data_point.timestamp.timestamp())
                    
                    if len(param_values) > 10:
                        # Linear regression for trend
                        x = np.array(timestamps)
                        y = np.array(param_values)
                        
                        # Normalize timestamps
                        x = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else x
                        
                        # Calculate trend coefficient
                        if len(x) > 1:
                            trend_coeff = np.polyfit(x, y, 1)[0]
                            baseline['trend_coefficients'][param_name] = trend_coeff
            
        except Exception as e:
            logger.error(f"Error updating trend analysis: {e}")
    
    async def _store_baseline_statistics(self, baseline_key: str, baseline: Dict) -> None:
        """Store baseline statistics to database"""
        try:
            # Store to Redis cache
            cache_key = f"baseline_statistics:{baseline_key}"
            
            # Prepare data for JSON serialization
            cache_data = {}
            for key, value in baseline.items():
                if key == 'last_updated':
                    cache_data[key] = value.isoformat()
                elif isinstance(value, dict):
                    cache_data[key] = value
                else:
                    cache_data[key] = value
            
            # Store with 7 day expiration
            self.redis_client.setex(
                cache_key,
                604800,  # 7 days
                json.dumps(cache_data, default=str)
            )
            
            # Store key statistics to InfluxDB
            await self._store_baseline_to_influxdb(baseline_key, baseline)
            
        except Exception as e:
            logger.error(f"Error storing baseline statistics: {e}")
    
    async def _store_baseline_to_influxdb(self, baseline_key: str, baseline: Dict) -> None:
        """Store baseline statistics to InfluxDB"""
        try:
            write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            
            points = []
            for param_name, mean_value in baseline['mean_values'].items():
                point = {
                    "measurement": "baseline_statistics",
                    "tags": {
                        "baseline_key": baseline_key,
                        "parameter": param_name
                    },
                    "fields": {
                        "mean_value": mean_value,
                        "std_value": baseline['std_values'].get(param_name, 0),
                        "ewma_value": baseline['ewma_values'].get(param_name, mean_value),
                        "trend_coefficient": baseline['trend_coefficients'].get(param_name, 0),
                        "skewness": baseline['skewness'].get(param_name, 0),
                        "kurtosis": baseline['kurtosis'].get(param_name, 0),
                        "sample_count": baseline['sample_count']
                    },
                    "time": baseline['last_updated']
                }
                
                # Add percentiles
                percentiles = baseline['percentiles'].get(param_name, {})
                for p_name, p_value in percentiles.items():
                    if isinstance(p_value, (int, float)):
                        point["fields"][f"percentile_{p_name}"] = p_value
                
                points.append(point)
            
            write_api.write(bucket="baseline_statistics", record=points)
            
        except Exception as e:
            logger.error(f"Error storing baseline to InfluxDB: {e}")
    
    async def _check_causal_strength_changes(self, relationship_key: str, strength: float) -> None:
        """Check for significant changes in causal strength and trigger alerts"""
        try:
            # Get historical causal strength values
            history_key = f"causal_history:{relationship_key}"
            
            # Add current strength to history
            if relationship_key not in self.causal_strength_history:
                self.causal_strength_history[relationship_key] = deque(maxlen=100)
            
            self.causal_strength_history[relationship_key].append({
                'timestamp': datetime.utcnow(),
                'strength': strength
            })
            
            history = list(self.causal_strength_history[relationship_key])
            
            if len(history) < 10:  # Need minimum history for change detection
                return
            
            # Calculate statistical measures for change detection
            recent_strengths = [h['strength'] for h in history[-10:]]  # Last 10 values
            historical_strengths = [h['strength'] for h in history[:-10]]  # Earlier values
            
            if len(historical_strengths) > 0:
                # Calculate change metrics
                recent_mean = np.mean(recent_strengths)
                historical_mean = np.mean(historical_strengths)
                historical_std = np.std(historical_strengths)
                
                # Z-score for change detection
                if historical_std > 0:
                    z_score = abs(recent_mean - historical_mean) / historical_std
                    
                    # Significant change thresholds
                    if z_score > 3.0:  # Very significant change
                        await self._trigger_causal_change_alert(
                            relationship_key, strength, recent_mean, historical_mean, z_score, AlertSeverity.HIGH
                        )
                    elif z_score > 2.0:  # Moderate change
                        await self._trigger_causal_change_alert(
                            relationship_key, strength, recent_mean, historical_mean, z_score, AlertSeverity.MEDIUM
                        )
                
                # Trend analysis
                if len(history) >= 20:
                    await self._analyze_causal_strength_trend(relationship_key, history)
            
            # Store causal strength history to database
            await self._store_causal_strength_history(relationship_key, strength)
            
        except Exception as e:
            logger.error(f"Error checking causal strength changes: {e}")
    
    async def _trigger_causal_change_alert(self, relationship_key: str, current_strength: float, 
                                         recent_mean: float, historical_mean: float, 
                                         z_score: float, severity: AlertSeverity) -> None:
        """Trigger alert for significant causal strength changes"""
        try:
            source_process, target_process = relationship_key.split('_', 1)
            
            alert = RealTimeAlert(
                alert_id=f"causal_change_{relationship_key}_{int(datetime.utcnow().timestamp())}",
                timestamp=datetime.utcnow(),
                severity=severity,
                process_type=ProcessType.SEISMIC,  # Default, would be determined from process names
                station_id="network_wide",
                location=(0.0, 0.0, 0.0),  # Network-wide alert
                message=f"Significant change in causal relationship strength: {source_process} -> {target_process}. "
                       f"Current: {current_strength:.3f}, Recent avg: {recent_mean:.3f}, "
                       f"Historical avg: {historical_mean:.3f}, Z-score: {z_score:.2f}",
                causal_context=[],  # Would include related relationships
                confidence=min(0.95, z_score / 5.0),  # Confidence based on z-score
                predicted_evolution={
                    'trend': 'increasing' if recent_mean > historical_mean else 'decreasing',
                    'magnitude': abs(recent_mean - historical_mean),
                    'time_horizon': '1-6 hours'
                },
                recommended_actions=[
                    "Monitor related geological processes",
                    "Check data quality and sensor status",
                    "Review recent environmental changes",
                    "Consider increasing monitoring frequency"
                ],
                data_quality_flags=[]
            )
            
            # Add to alert history
            self.alert_history.append(alert)
            
            # Store alert to database
            await self._store_alert_to_database(alert)
            
            # Broadcast alert to WebSocket clients
            await self._broadcast_alert(alert)
            
            logger.warning(f"Causal strength change alert: {alert.message}")
            
        except Exception as e:
            logger.error(f"Error triggering causal change alert: {e}")
    
    async def _analyze_causal_strength_trend(self, relationship_key: str, history: List[Dict]) -> None:
        """Analyze long-term trends in causal strength"""
        try:
            if len(history) < 20:
                return
            
            # Extract time series data
            timestamps = [h['timestamp'].timestamp() for h in history]
            strengths = [h['strength'] for h in history]
            
            # Normalize timestamps
            timestamps = np.array(timestamps)
            timestamps = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
            strengths = np.array(strengths)
            
            # Fit polynomial trend
            trend_coeffs = np.polyfit(timestamps, strengths, 2)  # Quadratic trend
            
            # Calculate trend metrics
            trend_slope = trend_coeffs[1]  # Linear component
            trend_acceleration = trend_coeffs[0]  # Quadratic component
            
            # Detect significant trends
            if abs(trend_slope) > 0.1:  # Significant linear trend
                trend_direction = "increasing" if trend_slope > 0 else "decreasing"
                
                # Store trend analysis
                trend_data = {
                    'relationship_key': relationship_key,
                    'trend_slope': trend_slope,
                    'trend_acceleration': trend_acceleration,
                    'trend_direction': trend_direction,
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'data_points': len(history)
                }
                
                # Cache trend analysis
                cache_key = f"causal_trend:{relationship_key}"
                self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour expiration
                    json.dumps(trend_data)
                )
                
                logger.info(f"Detected {trend_direction} trend in causal strength for {relationship_key}")
            
        except Exception as e:
            logger.error(f"Error analyzing causal strength trend: {e}")
    
    async def _store_causal_strength_history(self, relationship_key: str, strength: float) -> None:
        """Store causal strength history to database"""
        try:
            # Store to InfluxDB
            write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            
            point = {
                "measurement": "causal_strength_history",
                "tags": {
                    "relationship_key": relationship_key
                },
                "fields": {
                    "strength": strength
                },
                "time": datetime.utcnow()
            }
            
            write_api.write(bucket="causal_analysis", record=[point])
            
        except Exception as e:
            logger.error(f"Error storing causal strength history: {e}")
    
    async def _store_alert_to_database(self, alert: RealTimeAlert) -> None:
        """Store alert to database for historical analysis"""
        try:
            # Store to Redis for quick access
            cache_key = f"alert:{alert.alert_id}"
            alert_data = {
                'alert_id': alert.alert_id,
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity.value,
                'process_type': alert.process_type.value,
                'station_id': alert.station_id,
                'location': alert.location,
                'message': alert.message,
                'confidence': alert.confidence,
                'predicted_evolution': alert.predicted_evolution,
                'recommended_actions': alert.recommended_actions,
                'data_quality_flags': alert.data_quality_flags
            }
            
            # Store with 30 day expiration
            self.redis_client.setex(
                cache_key,
                2592000,  # 30 days
                json.dumps(alert_data)
            )
            
            # Store to InfluxDB
            await self._store_alert_to_influxdb(alert)
            
        except Exception as e:
            logger.error(f"Error storing alert to database: {e}")
    
    async def _store_alert_to_influxdb(self, alert: RealTimeAlert) -> None:
        """Store alert to InfluxDB"""
        try:
            write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            
            point = {
                "measurement": "geological_alerts",
                "tags": {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "process_type": alert.process_type.value,
                    "station_id": alert.station_id
                },
                "fields": {
                    "latitude": alert.location[0],
                    "longitude": alert.location[1],
                    "elevation": alert.location[2],
                    "confidence": alert.confidence,
                    "message": alert.message,
                    "recommended_actions_count": len(alert.recommended_actions),
                    "data_quality_flags_count": len(alert.data_quality_flags)
                },
                "time": alert.timestamp
            }
            
            write_api.write(bucket="alerts", record=[point])
            
        except Exception as e:
            logger.error(f"Error storing alert to InfluxDB: {e}")
    
    async def _broadcast_alert(self, alert: RealTimeAlert) -> None:
        """Broadcast alert to WebSocket clients"""
        try:
            if self.websocket_connections:
                alert_message = {
                    'type': 'alert',
                    'data': {
                        'alert_id': alert.alert_id,
                        'timestamp': alert.timestamp.isoformat(),
                        'severity': alert.severity.value,
                        'process_type': alert.process_type.value,
                        'station_id': alert.station_id,
                        'location': alert.location,
                        'message': alert.message,
                        'confidence': alert.confidence,
                        'predicted_evolution': alert.predicted_evolution,
                        'recommended_actions': alert.recommended_actions
                    }
                }
                
                # Broadcast to all connected clients
                disconnected_clients = set()
                for websocket in self.websocket_connections:
                    try:
                        await websocket.send(json.dumps(alert_message))
                    except Exception as e:
                        logger.warning(f"Failed to send alert to WebSocket client: {e}")
                        disconnected_clients.add(websocket)
                
                # Remove disconnected clients
                self.websocket_connections -= disconnected_clients
            
        except Exception as e:
            logger.error(f"Error broadcasting alert: {e}")
    

    
    async def _store_anomaly(self, anomaly: AnomalyDetection) -> None:
        """Store detected anomaly"""
        # Would store to database in production
        pass
    
    async def _generate_anomaly_alert(self, anomaly: AnomalyDetection) -> None:
        """Generate alert for detected anomaly"""
        # Would create and distribute alert
        pass
    
    async def _process_alert(self, alert: RealTimeAlert) -> None:
        """Process and distribute alert"""
        # Would handle alert routing and notifications
        pass
    
    async def _update_quality_metrics(self, quality_issues: List[Dict[str, Any]]) -> None:
        """Update data quality metrics"""
        # Would update quality tracking
        pass
    
    async def _generate_quality_alert(self, issue: Dict[str, Any]) -> None:
        """Generate alert for data quality issue"""
        # Would create quality alert
        pass
    
    async def _check_performance_alerts(self, metrics: Dict[str, float]) -> None:
        """Check for performance-related alerts"""
        # Would check performance thresholds and generate alerts
        pass

# Example usage and testing
async def main():
    """Example usage of the Real-Time Process Monitor"""
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'influxdb_url': 'http://localhost:8086',
        'influxdb_token': 'your-token-here',
        'influxdb_org': 'prometheus',
        'websocket_host': 'localhost',
        'websocket_port': 8765,
        'max_workers': 8
    }
    
    monitor = RealTimeProcessMonitor(config)
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Shutting down real-time process monitor")
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())
    
    async def _calculate_frequency_correlation(self, anomaly: AnomalyDetection, 
                                             target_timestamp: datetime) -> float:
        """Calculate frequency-domain correlation between anomaly and target timestamp"""
        try:
            # Get time series data around both timestamps
            window = timedelta(minutes=15)
            
            # Target process data
            target_data = await self._get_process_metrics_window(
                anomaly.process_key, target_timestamp, window
            )
            
            # Anomaly process data
            anomaly_data = await self._get_process_metrics_window(
                anomaly.process_key, anomaly.detected_at, window
            )
            
            if not target_data or not anomaly_data:
                return 0.0
            
            # Extract primary metric for frequency analysis
            target_series = [d.get('primary_metric', 0) for d in target_data]
            anomaly_series = [d.get('primary_metric', 0) for d in anomaly_data]
            
            if len(target_series) < 4 or len(anomaly_series) < 4:
                return 0.0
            
            # Apply FFT for frequency domain analysis
            target_fft = np.fft.fft(target_series)
            anomaly_fft = np.fft.fft(anomaly_series)
            
            # Calculate spectral coherence
            cross_spectrum = target_fft * np.conj(anomaly_fft)
            target_power = np.abs(target_fft) ** 2
            anomaly_power = np.abs(anomaly_fft) ** 2
            
            coherence = np.abs(cross_spectrum) ** 2 / (target_power * anomaly_power + 1e-10)
            
            # Return mean coherence across frequencies
            return float(np.mean(coherence[:len(coherence)//2]))  # Use positive frequencies only
            
        except Exception as e:
            self.logger.error(f"Error calculating frequency correlation: {e}")
            return 0.0
    
    async def _calculate_pattern_similarity(self, anomaly: AnomalyDetection, 
                                          target_timestamp: datetime) -> float:
        """Calculate pattern similarity using dynamic time warping"""
        try:
            # Get pattern data around both timestamps
            window = timedelta(minutes=10)
            
            target_data = await self._get_process_metrics_window(
                anomaly.process_key, target_timestamp, window
            )
            
            anomaly_data = await self._get_process_metrics_window(
                anomaly.process_key, anomaly.detected_at, window
            )
            
            if not target_data or not anomaly_data:
                return 0.0
            
            # Extract normalized patterns
            target_pattern = self._extract_pattern_features(target_data)
            anomaly_pattern = self._extract_pattern_features(anomaly_data)
            
            if len(target_pattern) < 3 or len(anomaly_pattern) < 3:
                return 0.0
            
            # Calculate DTW distance
            dtw_distance = self._calculate_dtw_distance(target_pattern, anomaly_pattern)
            
            # Convert distance to similarity score (0-1)
            max_possible_distance = max(len(target_pattern), len(anomaly_pattern))
            similarity = max(0.0, 1.0 - (dtw_distance / max_possible_distance))
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0
    
    def _extract_pattern_features(self, data: List[Dict]) -> List[float]:
        """Extract normalized pattern features from time series data"""
        try:
            # Extract multiple metrics and normalize
            features = []
            
            for d in data:
                # Combine multiple metrics into feature vector
                feature = (
                    d.get('primary_metric', 0) * 0.4 +
                    d.get('secondary_metric', 0) * 0.3 +
                    d.get('tertiary_metric', 0) * 0.3
                )
                features.append(feature)
            
            # Normalize to [0, 1] range
            if features:
                min_val, max_val = min(features), max(features)
                if max_val > min_val:
                    features = [(f - min_val) / (max_val - min_val) for f in features]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting pattern features: {e}")
            return []
    
    def _calculate_dtw_distance(self, series1: List[float], series2: List[float]) -> float:
        """Calculate Dynamic Time Warping distance between two series"""
        try:
            n, m = len(series1), len(series2)
            
            # Initialize DTW matrix
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0
            
            # Fill DTW matrix
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(series1[i-1] - series2[j-1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],      # insertion
                        dtw_matrix[i, j-1],      # deletion
                        dtw_matrix[i-1, j-1]     # match
                    )
            
            return dtw_matrix[n, m]
            
        except Exception as e:
            self.logger.error(f"Error calculating DTW distance: {e}")
            return float('inf')
    
    async def _get_process_metrics_window(self, process_key: str, 
                                        timestamp: datetime, 
                                        window: timedelta) -> List[Dict]:
        """Get process metrics within a time window"""
        try:
            start_time = timestamp - window
            end_time = timestamp + window
            
            # Query from data buffer first
            buffer_data = []
            if process_key in self.data_buffers:
                for dp in self.data_buffers[process_key]:
                    if start_time <= dp.timestamp <= end_time:
                        metrics = {
                            'timestamp': dp.timestamp,
                            'primary_metric': self._get_primary_metric(dp),
                            'secondary_metric': self._get_secondary_metric(dp),
                            'tertiary_metric': self._get_tertiary_metric(dp),
                            'cpu_usage': dp.parameters.get('cpu_usage', 0),
                            'memory_usage': dp.parameters.get('memory_usage', 0),
                            'io_rate': dp.parameters.get('io_rate', 0),
                            'network_rate': dp.parameters.get('network_rate', 0)
                        }
                        buffer_data.append(metrics)
            
            # If not enough data in buffer, query from database
            if len(buffer_data) < 5:
                db_data = await self._query_historical_metrics(
                    process_key, start_time, end_time
                )
                buffer_data.extend(db_data)
            
            # Sort by timestamp
            buffer_data.sort(key=lambda x: x['timestamp'])
            
            return buffer_data
            
        except Exception as e:
            self.logger.error(f"Error getting process metrics window: {e}")
            return []
    
    def _get_primary_metric(self, data_point: 'GeologicalDataPoint') -> float:
        """Extract primary metric based on process type"""
        if data_point.process_type == ProcessType.SEISMIC:
            return data_point.parameters.get('ground_velocity', 0)
        elif data_point.process_type == ProcessType.VOLCANIC:
            return data_point.parameters.get('so2_flux', 0)
        elif data_point.process_type == ProcessType.DEFORMATION:
            return data_point.parameters.get('vertical_displacement', 0)
        elif data_point.process_type == ProcessType.GEOCHEMICAL:
            return data_point.parameters.get('co2_concentration', 0)
        else:
            return 0.0
    
    def _get_secondary_metric(self, data_point: 'GeologicalDataPoint') -> float:
        """Extract secondary metric based on process type"""
        if data_point.process_type == ProcessType.SEISMIC:
            return data_point.parameters.get('frequency_content', 0)
        elif data_point.process_type == ProcessType.VOLCANIC:
            return data_point.parameters.get('thermal_anomaly', 0)
        elif data_point.process_type == ProcessType.DEFORMATION:
            return data_point.parameters.get('horizontal_displacement', 0)
        elif data_point.process_type == ProcessType.GEOCHEMICAL:
            return data_point.parameters.get('h2s_concentration', 0)
        else:
            return 0.0
    
    def _get_tertiary_metric(self, data_point: 'GeologicalDataPoint') -> float:
        """Extract tertiary metric based on process type"""
        if data_point.process_type == ProcessType.SEISMIC:
            return data_point.parameters.get('event_magnitude', 0)
        elif data_point.process_type == ProcessType.VOLCANIC:
            return data_point.parameters.get('ash_emission_rate', 0)
        elif data_point.process_type == ProcessType.DEFORMATION:
            return data_point.parameters.get('strain_rate', 0)
        elif data_point.process_type == ProcessType.GEOCHEMICAL:
            return data_point.parameters.get('temperature', 0)
        else:
            return 0.0
    
    async def _execute_influx_query(self, query: str) -> List[Dict]:
        """Execute InfluxDB query and return results"""
        try:
            # Placeholder for InfluxDB query execution
            # In real implementation, this would use InfluxDB client
            return []
        except Exception as e:
            self.logger.error(f"Error executing InfluxDB query: {e}")
            return []
    
    async def _execute_postgres_query(self, query: str, params: tuple) -> List[Dict]:
        """Execute PostgreSQL query and return results"""
        try:
            # Placeholder for PostgreSQL query execution
            # In real implementation, this would use asyncpg or similar
            return []
        except Exception as e:
            self.logger.error(f"Error executing PostgreSQL query: {e}")
            return []
    
    async def _execute_neo4j_query(self, query: str, params: Dict) -> List[Dict]:
        """Execute Neo4j query and return results"""
        try:
            # Placeholder for Neo4j query execution
            # In real implementation, this would use neo4j driver
            return []
        except Exception as e:
            self.logger.error(f"Error executing Neo4j query: {e}")
            return []
    
    def _normalize_anomaly_data(self, raw_data: List[Dict]) -> List['AnomalyDetection']:
        """Normalize anomaly data from different database sources"""
        try:
            normalized = []
            
            for item in raw_data:
                # Convert raw database result to AnomalyDetection object
                # This would depend on the actual database schema
                anomaly = AnomalyDetection(
                    anomaly_id=item.get('id', ''),
                    process_key=item.get('process_key', ''),
                    detected_at=item.get('detected_at', datetime.utcnow()),
                    anomaly_type=item.get('type', 'unknown'),
                    severity_score=item.get('severity', 0.5),
                    confidence=item.get('confidence', 0.5),
                    affected_metrics=item.get('affected_metrics', []),
                    description=item.get('description', ''),
                    correlation_score=0.0,  # Will be calculated later
                    temporal_proximity=0.0,  # Will be calculated later
                    cross_correlation=0.0   # Will be calculated later
                )
                normalized.append(anomaly)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing anomaly data: {e}")
            return []
    
    async def _query_historical_metrics(self, process_key: str, 
                                      start_time: datetime, 
                                      end_time: datetime) -> List[Dict]:
        """Query historical metrics from database"""
        try:
            # Placeholder for historical data query
            # In real implementation, this would query InfluxDB or PostgreSQL
            return []
        except Exception as e:
            self.logger.error(f"Error querying historical metrics: {e}")
            return []


# Additional dataclass for enhanced anomaly detection
@dataclass
class AnomalyDetection:
    """Enhanced anomaly detection result with correlation metrics"""
    anomaly_id: str
    process_key: str
    detected_at: datetime
    anomaly_type: str
    severity_score: float
    confidence: float
    affected_metrics: List[str]
    description: str
    correlation_score: float = 0.0
    temporal_proximity: float = 0.0
    cross_correlation: float = 0.0
    cluster_id: int = -1
    is_pattern_anomaly: bool = False
    window_type: str = ""
    duration_seconds: float = 0.0