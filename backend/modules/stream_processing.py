"""
Real-time Stream Processing Framework
Apache Kafka integration, Apache Flink processing, real-time aggregation, and event-driven architecture
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import struct
from abc import ABC, abstractmethod

# Kafka imports with fallback
try:
    from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
    from kafka.admin.new_topics import NewTopic
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    warnings.warn("Apache Kafka not available. Stream processing will use in-memory queues.")

# Flink imports with fallback
try:
    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.table import StreamTableEnvironment, TableEnvironment
    from pyflink.datastream.connectors import FlinkKafkaProducer, FlinkKafkaConsumer
    FLINK_AVAILABLE = True
except ImportError:
    FLINK_AVAILABLE = False
    warnings.warn("Apache Flink not available. Using custom stream processing engine.")

# Database imports
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import psycopg2.extensions
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    warnings.warn("PostgreSQL not available. Database features will be limited.")

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, LineString
    import pyproj
    GIS_AVAILABLE = True
except ImportError:
    GIS_AVAILABLE = False
    warnings.warn("GIS libraries not available. Geographic features will be limited.")

warnings.filterwarnings('ignore')

class StreamEventType(Enum):
    SENSOR_DATA = "sensor_data"
    GEOLOGICAL_EVENT = "geological_event"
    EARTHQUAKE = "earthquake"
    VOLCANIC_ACTIVITY = "volcanic_activity"
    DEFORMATION = "deformation"
    GPS_DATA = "gps_data"
    INSAR_DATA = "insar_data"
    SEISMIC_DATA = "seismic_data"
    ALERT = "alert"
    STATUS_UPDATE = "status_update"

class ProcessingStrategy(Enum):
    WINDOWED = "windowed"
    CONTINUOUS = "continuous"
    BATCH = "batch"
    REAL_TIME = "real_time"

@dataclass
class StreamEvent:
    """Event in the stream processing system"""
    event_id: str
    event_type: StreamEventType
    timestamp: datetime
    source_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[Tuple[float, float, float]] = None  # lat, lon, elevation
    
    def to_json(self) -> str:
        """Convert event to JSON string"""
        event_dict = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source_id': self.source_id,
            'data': self.data,
            'metadata': self.metadata,
            'location': self.location
        }
        return json.dumps(event_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StreamEvent':
        """Create event from JSON string"""
        data = json.loads(json_str)
        return cls(
            event_id=data['event_id'],
            event_type=StreamEventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            source_id=data['source_id'],
            data=data['data'],
            metadata=data.get('metadata', {}),
            location=data.get('location')
        )

class KafkaStreamManager:
    """Apache Kafka integration for stream processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        self.producer = None
        self.consumer = None
        self.admin_client = None
        
        if KAFKA_AVAILABLE:
            self._setup_kafka()
        else:
            # Fallback to in-memory queues
            self._setup_memory_queues()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Kafka configuration"""
        return {
            'bootstrap_servers': ['localhost:9092'],
            'topics': {
                'geological_events': 'geological_events',
                'sensor_data': 'sensor_data',
                'alerts': 'alerts',
                'processed_data': 'processed_data'
            },
            'producer_config': {
                'acks': 'all',
                'retries': 3,
                'batch_size': 16384,
                'linger_ms': 10,
                'buffer_memory': 33554432
            },
            'consumer_config': {
                'group_id': 'prometheus_consumer_group',
                'auto_offset_reset': 'latest',
                'enable_auto_commit': True,
                'auto_commit_interval_ms': 1000,
                'session_timeout_ms': 10000
            }
        }
    
    def _setup_kafka(self):
        """Setup Kafka connections"""
        try:
            # Setup producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.config['bootstrap_servers'],
                value_serializer=lambda x: x.encode('utf-8'),
                **self.config['producer_config']
            )
            
            # Setup admin client
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=self.config['bootstrap_servers']
            )
            
            # Create topics if they don't exist
            self._create_topics()
            
            self.logger.info("Kafka setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Kafka: {e}")
            self._setup_memory_queues()
    
    def _setup_memory_queues(self):
        """Setup in-memory queues as fallback"""
        self.memory_queues = {
            topic: queue.Queue() for topic in self.config['topics'].values()
        }
        self.logger.info("Using in-memory queues for stream processing")
    
    def _create_topics(self):
        """Create Kafka topics"""
        if not KAFKA_AVAILABLE or not self.admin_client:
            return
        
        topics = []
        for topic_name in self.config['topics'].values():
            topics.append(NewTopic(
                name=topic_name,
                num_partitions=3,
                replication_factor=1
            ))
        
        try:
            existing_topics = self.admin_client.list_topics()
            new_topics = [topic for topic in topics if topic.name not in existing_topics]
            
            if new_topics:
                self.admin_client.create_topics(new_topics)
                self.logger.info(f"Created {len(new_topics)} new topics")
        except Exception as e:
            self.logger.warning(f"Failed to create topics: {e}")
    
    async def produce_event(self, event: StreamEvent, topic: str = None):
        """Produce event to stream"""
        if topic is None:
            topic = self.config['topics']['geological_events']
        
        try:
            if KAFKA_AVAILABLE and self.producer:
                # Send to Kafka
                future = self.producer.send(topic, event.to_json())
                self.producer.flush()  # Ensure delivery
                
                # Optional: wait for confirmation
                record_metadata = future.get(timeout=10)
                self.logger.debug(f"Event sent to {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
            
            else:
                # Send to memory queue
                if topic in self.memory_queues:
                    self.memory_queues[topic].put(event.to_json())
                    
        except Exception as e:
            self.logger.error(f"Failed to produce event: {e}")
    
    async def consume_events(self, topic: str, callback: Callable[[StreamEvent], None], 
                           timeout_ms: int = 1000):
        """Consume events from stream"""
        if KAFKA_AVAILABLE:
            await self._consume_kafka_events(topic, callback, timeout_ms)
        else:
            await self._consume_memory_events(topic, callback, timeout_ms)
    
    async def _consume_kafka_events(self, topic: str, callback: Callable[[StreamEvent], None], 
                                  timeout_ms: int):
        """Consume events from Kafka"""
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.config['bootstrap_servers'],
                **self.config['consumer_config']
            )
            
            for message in consumer:
                try:
                    event = StreamEvent.from_json(message.value.decode('utf-8'))
                    await asyncio.create_task(self._run_callback(callback, event))
                except Exception as e:
                    self.logger.error(f"Failed to process message: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to consume from Kafka: {e}")
    
    async def _consume_memory_events(self, topic: str, callback: Callable[[StreamEvent], None], 
                                   timeout_ms: int):
        """Consume events from memory queue"""
        if topic not in self.memory_queues:
            return
        
        while True:
            try:
                message = self.memory_queues[topic].get(timeout=timeout_ms/1000)
                event = StreamEvent.from_json(message)
                await asyncio.create_task(self._run_callback(callback, event))
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Failed to process memory queue message: {e}")
    
    async def _run_callback(self, callback: Callable[[StreamEvent], None], event: StreamEvent):
        """Run callback function for event processing"""
        if asyncio.iscoroutinefunction(callback):
            await callback(event)
        else:
            callback(event)

class FlinkStreamProcessor:
    """Apache Flink integration for advanced stream processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        self.env = None
        self.table_env = None
        
        if FLINK_AVAILABLE:
            self._setup_flink()
        else:
            self._setup_custom_processor()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Flink configuration"""
        return {
            'parallelism': 4,
            'checkpoint_interval': 60000,  # 1 minute
            'checkpoint_mode': 'EXACTLY_ONCE',
            'window_size_seconds': 30,
            'slide_interval_seconds': 10,
            'kafka_config': {
                'bootstrap_servers': 'localhost:9092',
                'group_id': 'flink_consumer_group'
            }
        }
    
    def _setup_flink(self):
        """Setup Flink processing environment"""
        try:
            # Create execution environment
            self.env = StreamExecutionEnvironment.get_execution_environment()
            self.env.set_parallelism(self.config['parallelism'])
            self.env.enable_checkpointing(self.config['checkpoint_interval'])
            
            # Create table environment
            self.table_env = StreamTableEnvironment.create(self.env)
            
            self.logger.info("Flink environment setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Flink: {e}")
            self._setup_custom_processor()
    
    def _setup_custom_processor(self):
        """Setup custom stream processor as fallback"""
        self.processing_queues = {}
        self.processors = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config['parallelism'])
        self.logger.info("Using custom stream processor")
    
    async def create_geological_stream_pipeline(self, kafka_manager: KafkaStreamManager):
        """Create geological data processing pipeline"""
        if FLINK_AVAILABLE and self.env:
            await self._create_flink_pipeline(kafka_manager)
        else:
            await self._create_custom_pipeline(kafka_manager)
    
    async def _create_flink_pipeline(self, kafka_manager: KafkaStreamManager):
        """Create Flink-based processing pipeline"""
        # This would be implemented with actual Flink streaming operations
        # For now, we'll create a conceptual implementation
        
        self.logger.info("Creating Flink geological stream pipeline")
        
        # Example Flink pipeline structure:
        # 1. Source: Kafka consumer
        # 2. Parse and validate geological events
        # 3. Apply windowed aggregations
        # 4. Detect anomalies and patterns
        # 5. Sink: Multiple outputs (database, alerts, processed stream)
        
        pass
    
    async def _create_custom_pipeline(self, kafka_manager: KafkaStreamManager):
        """Create custom processing pipeline"""
        self.logger.info("Creating custom geological stream pipeline")
        
        # Setup processing stages
        processors = [
            self._data_validation_processor,
            self._real_time_aggregation_processor,
            self._anomaly_detection_processor,
            self._alert_generation_processor
        ]
        
        # Start processing pipeline
        for i, processor in enumerate(processors):
            queue_name = f"stage_{i}"
            self.processing_queues[queue_name] = asyncio.Queue()
            asyncio.create_task(self._run_processor(processor, queue_name))
    
    async def _run_processor(self, processor: Callable, queue_name: str):
        """Run individual processor"""
        while True:
            try:
                if queue_name in self.processing_queues:
                    event = await self.processing_queues[queue_name].get()
                    result = await processor(event)
                    
                    # Pass to next stage
                    next_stage = f"stage_{int(queue_name.split('_')[1]) + 1}"
                    if next_stage in self.processing_queues:
                        await self.processing_queues[next_stage].put(result)
                        
            except Exception as e:
                self.logger.error(f"Processor {queue_name} error: {e}")
                await asyncio.sleep(1)
    
    async def _data_validation_processor(self, event: StreamEvent) -> StreamEvent:
        """Validate and clean incoming data"""
        # Validate required fields
        if not event.event_id or not event.timestamp or not event.data:
            raise ValueError("Invalid event: missing required fields")
        
        # Validate location data if present
        if event.location:
            lat, lon, elev = event.location
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                self.logger.warning(f"Invalid location in event {event.event_id}")
                event.location = None
        
        # Add validation metadata
        event.metadata['validated'] = True
        event.metadata['validation_timestamp'] = datetime.utcnow().isoformat()
        
        return event
    
    async def _real_time_aggregation_processor(self, event: StreamEvent) -> StreamEvent:
        """Perform real-time aggregations"""
        # Implement windowed aggregations
        window_key = f"{event.event_type.value}_{event.source_id}"
        
        # Store in time-based windows
        if not hasattr(self, 'time_windows'):
            self.time_windows = {}
        
        current_window = datetime.utcnow().replace(second=0, microsecond=0)
        if window_key not in self.time_windows:
            self.time_windows[window_key] = {}
        
        if current_window not in self.time_windows[window_key]:
            self.time_windows[window_key][current_window] = []
        
        self.time_windows[window_key][current_window].append(event)
        
        # Calculate aggregations
        window_events = self.time_windows[window_key][current_window]
        if len(window_events) >= 5:  # Minimum events for aggregation
            aggregated_data = self._calculate_aggregations(window_events)
            event.metadata['aggregated_metrics'] = aggregated_data
        
        return event
    
    def _calculate_aggregations(self, events: List[StreamEvent]) -> Dict[str, Any]:
        """Calculate aggregations for event window"""
        aggregations = {
            'count': len(events),
            'event_types': {},
            'sources': set(),
            'time_range': {
                'start': min(e.timestamp for e in events).isoformat(),
                'end': max(e.timestamp for e in events).isoformat()
            }
        }
        
        # Count event types
        for event in events:
            event_type = event.event_type.value
            aggregations['event_types'][event_type] = aggregations['event_types'].get(event_type, 0) + 1
            aggregations['sources'].add(event.source_id)
        
        aggregations['sources'] = list(aggregations['sources'])
        
        # Calculate numerical aggregations if applicable
        numerical_fields = []
        for event in events:
            for key, value in event.data.items():
                if isinstance(value, (int, float)):
                    numerical_fields.append((key, value))
        
        if numerical_fields:
            field_stats = {}
            field_groups = {}
            
            for field_name, value in numerical_fields:
                if field_name not in field_groups:
                    field_groups[field_name] = []
                field_groups[field_name].append(value)
            
            for field_name, values in field_groups.items():
                field_stats[field_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
            
            aggregations['numerical_stats'] = field_stats
        
        return aggregations
    
    async def _anomaly_detection_processor(self, event: StreamEvent) -> StreamEvent:
        """Detect anomalies in geological data"""
        anomaly_score = 0.0
        anomaly_reasons = []
        
        # Check for temporal anomalies
        if 'aggregated_metrics' in event.metadata:
            metrics = event.metadata['aggregated_metrics']
            
            # High event frequency anomaly
            if metrics['count'] > 20:  # Threshold for high activity
                anomaly_score += 0.3
                anomaly_reasons.append('high_event_frequency')
            
            # Unusual numerical values
            if 'numerical_stats' in metrics:
                for field, stats in metrics['numerical_stats'].items():
                    if stats['std'] > 2 * stats['mean']:  # High variance
                        anomaly_score += 0.2
                        anomaly_reasons.append(f'high_variance_{field}')
        
        # Check individual event anomalies
        if event.event_type == StreamEventType.EARTHQUAKE:
            magnitude = event.data.get('magnitude', 0)
            if magnitude > 7.0:
                anomaly_score += 0.5
                anomaly_reasons.append('large_earthquake')
        
        # Add anomaly information
        if anomaly_score > 0.5:  # Anomaly threshold
            event.metadata['anomaly_detected'] = True
            event.metadata['anomaly_score'] = anomaly_score
            event.metadata['anomaly_reasons'] = anomaly_reasons
        
        return event
    
    async def _alert_generation_processor(self, event: StreamEvent) -> StreamEvent:
        """Generate alerts based on processed data"""
        alerts = []
        
        # Check for high-priority events
        if event.metadata.get('anomaly_detected', False):
            alert_level = 'HIGH' if event.metadata['anomaly_score'] > 0.7 else 'MEDIUM'
            alerts.append({
                'level': alert_level,
                'type': 'anomaly_detected',
                'message': f"Anomaly detected in {event.event_type.value} from {event.source_id}",
                'reasons': event.metadata.get('anomaly_reasons', []),
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Earthquake-specific alerts
        if event.event_type == StreamEventType.EARTHQUAKE:
            magnitude = event.data.get('magnitude', 0)
            if magnitude >= 6.0:
                alerts.append({
                    'level': 'CRITICAL' if magnitude >= 7.0 else 'HIGH',
                    'type': 'earthquake_alert',
                    'message': f"Magnitude {magnitude} earthquake detected",
                    'location': event.location,
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        if alerts:
            event.metadata['alerts'] = alerts
        
        return event

class RealTimeAggregator:
    """Real-time aggregation system for geological data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        self.aggregation_windows = {}
        self.aggregation_results = {}
        
        # Start aggregation tasks
        asyncio.create_task(self._cleanup_old_windows())
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default aggregation configuration"""
        return {
            'window_sizes': {
                'minute': timedelta(minutes=1),
                'hour': timedelta(hours=1),
                'day': timedelta(days=1)
            },
            'slide_intervals': {
                'minute': timedelta(seconds=10),
                'hour': timedelta(minutes=5),
                'day': timedelta(hours=1)
            },
            'max_window_age': timedelta(days=7),
            'aggregation_functions': [
                'count', 'mean', 'std', 'min', 'max', 'percentiles'
            ]
        }
    
    async def add_event(self, event: StreamEvent):
        """Add event to aggregation windows"""
        for window_name, window_size in self.config['window_sizes'].items():
            await self._add_to_window(event, window_name, window_size)
    
    async def _add_to_window(self, event: StreamEvent, window_name: str, window_size: timedelta):
        """Add event to specific time window"""
        # Calculate window boundaries
        current_time = datetime.utcnow()
        window_start = current_time.replace(second=0, microsecond=0)
        
        if window_name == 'hour':
            window_start = window_start.replace(minute=0)
        elif window_name == 'day':
            window_start = window_start.replace(hour=0, minute=0)
        
        # Create window key
        window_key = f"{window_name}_{window_start.isoformat()}"
        
        # Initialize window if not exists
        if window_key not in self.aggregation_windows:
            self.aggregation_windows[window_key] = {
                'events': [],
                'start_time': window_start,
                'end_time': window_start + window_size,
                'window_type': window_name
            }
        
        # Add event to window
        self.aggregation_windows[window_key]['events'].append(event)
        
        # Trigger aggregation if window is full enough
        if len(self.aggregation_windows[window_key]['events']) % 10 == 0:
            await self._compute_aggregations(window_key)
    
    async def _compute_aggregations(self, window_key: str):
        """Compute aggregations for window"""
        if window_key not in self.aggregation_windows:
            return
        
        window_data = self.aggregation_windows[window_key]
        events = window_data['events']
        
        if not events:
            return
        
        # Basic aggregations
        aggregations = {
            'window_key': window_key,
            'window_type': window_data['window_type'],
            'start_time': window_data['start_time'].isoformat(),
            'end_time': window_data['end_time'].isoformat(),
            'event_count': len(events),
            'event_types': {},
            'sources': {},
            'locations': []
        }
        
        # Count by event type and source
        for event in events:
            event_type = event.event_type.value
            aggregations['event_types'][event_type] = aggregations['event_types'].get(event_type, 0) + 1
            aggregations['sources'][event.source_id] = aggregations['sources'].get(event.source_id, 0) + 1
            
            if event.location:
                aggregations['locations'].append(event.location)
        
        # Numerical field aggregations
        numerical_data = {}
        for event in events:
            for field, value in event.data.items():
                if isinstance(value, (int, float)):
                    if field not in numerical_data:
                        numerical_data[field] = []
                    numerical_data[field].append(value)
        
        # Calculate statistics for numerical fields
        if numerical_data:
            aggregations['numerical_stats'] = {}
            for field, values in numerical_data.items():
                values_array = np.array(values)
                aggregations['numerical_stats'][field] = {
                    'count': len(values),
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'percentiles': {
                        'p25': float(np.percentile(values_array, 25)),
                        'p50': float(np.percentile(values_array, 50)),
                        'p75': float(np.percentile(values_array, 75)),
                        'p90': float(np.percentile(values_array, 90)),
                        'p95': float(np.percentile(values_array, 95))
                    }
                }
        
        # Geographic aggregations if locations available
        if aggregations['locations']:
            locations = np.array(aggregations['locations'])
            aggregations['geographic_stats'] = {
                'center_lat': float(np.mean(locations[:, 0])),
                'center_lon': float(np.mean(locations[:, 1])),
                'lat_range': [float(np.min(locations[:, 0])), float(np.max(locations[:, 0]))],
                'lon_range': [float(np.min(locations[:, 1])), float(np.max(locations[:, 1]))],
                'elevation_stats': {
                    'mean': float(np.mean(locations[:, 2])),
                    'std': float(np.std(locations[:, 2])),
                    'range': [float(np.min(locations[:, 2])), float(np.max(locations[:, 2]))]
                } if locations.shape[1] > 2 else None
            }
        
        # Store aggregation results
        self.aggregation_results[window_key] = aggregations
        
        self.logger.debug(f"Computed aggregations for window {window_key}: {aggregations['event_count']} events")
    
    async def get_aggregations(self, window_type: str = None, 
                             time_range: Tuple[datetime, datetime] = None) -> List[Dict[str, Any]]:
        """Get aggregation results"""
        results = []
        
        for window_key, aggregation in self.aggregation_results.items():
            # Filter by window type
            if window_type and aggregation['window_type'] != window_type:
                continue
            
            # Filter by time range
            if time_range:
                start_time = datetime.fromisoformat(aggregation['start_time'])
                end_time = datetime.fromisoformat(aggregation['end_time'])
                
                if not (time_range[0] <= start_time <= time_range[1] or 
                       time_range[0] <= end_time <= time_range[1]):
                    continue
            
            results.append(aggregation)
        
        return sorted(results, key=lambda x: x['start_time'])
    
    async def _cleanup_old_windows(self):
        """Clean up old aggregation windows"""
        while True:
            try:
                current_time = datetime.utcnow()
                max_age = self.config['max_window_age']
                
                # Clean up old windows
                windows_to_remove = []
                for window_key, window_data in self.aggregation_windows.items():
                    if current_time - window_data['start_time'] > max_age:
                        windows_to_remove.append(window_key)
                
                for window_key in windows_to_remove:
                    del self.aggregation_windows[window_key]
                    if window_key in self.aggregation_results:
                        del self.aggregation_results[window_key]
                
                if windows_to_remove:
                    self.logger.info(f"Cleaned up {len(windows_to_remove)} old windows")
                
                await asyncio.sleep(3600)  # Run cleanup every hour
                
            except Exception as e:
                self.logger.error(f"Error in window cleanup: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes before retry

class EventDrivenArchitecture:
    """Event-driven architecture for geological monitoring"""
    
    def __init__(self, kafka_manager: KafkaStreamManager):
        self.kafka_manager = kafka_manager
        self.logger = logging.getLogger(__name__)
        self.event_handlers = {}
        self.middleware = []
        self.subscribers = {}
    
    def register_handler(self, event_type: StreamEventType, 
                        handler: Callable[[StreamEvent], Any]):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        self.logger.info(f"Registered handler for {event_type.value}")
    
    def add_middleware(self, middleware: Callable[[StreamEvent], StreamEvent]):
        """Add middleware for event processing"""
        self.middleware.append(middleware)
    
    async def publish_event(self, event: StreamEvent, topic: str = None):
        """Publish event to the system"""
        # Apply middleware
        for middleware_func in self.middleware:
            try:
                if asyncio.iscoroutinefunction(middleware_func):
                    event = await middleware_func(event)
                else:
                    event = middleware_func(event)
            except Exception as e:
                self.logger.error(f"Middleware error: {e}")
        
        # Send to Kafka
        await self.kafka_manager.produce_event(event, topic)
        
        # Process local handlers
        await self._process_local_handlers(event)
    
    async def _process_local_handlers(self, event: StreamEvent):
        """Process local event handlers"""
        if event.event_type in self.event_handlers:
            handlers = self.event_handlers[event.event_type]
            
            # Run handlers concurrently
            tasks = []
            for handler in handlers:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(asyncio.create_task(handler(event)))
                else:
                    tasks.append(asyncio.create_task(
                        asyncio.to_thread(handler, event)
                    ))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any errors
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Handler {i} error: {result}")
    
    async def start_event_processing(self, topics: List[str] = None):
        """Start processing events from streams"""
        if topics is None:
            topics = list(self.kafka_manager.config['topics'].values())
        
        # Start consumers for each topic
        for topic in topics:
            asyncio.create_task(
                self.kafka_manager.consume_events(topic, self._handle_consumed_event)
            )
        
        self.logger.info(f"Started event processing for topics: {topics}")
    
    async def _handle_consumed_event(self, event: StreamEvent):
        """Handle event consumed from stream"""
        await self._process_local_handlers(event)

# Example usage and testing
async def example_stream_processing():
    """Example of using stream processing system"""
    
    print("=== Stream Processing System Example ===")
    
    # Initialize components
    kafka_manager = KafkaStreamManager()
    flink_processor = FlinkStreamProcessor()
    aggregator = RealTimeAggregator()
    event_architecture = EventDrivenArchitecture(kafka_manager)
    
    # Register event handlers
    async def earthquake_handler(event: StreamEvent):
        print(f"Earthquake handler: {event.data.get('magnitude', 'N/A')} magnitude event")
    
    async def alert_handler(event: StreamEvent):
        if 'alerts' in event.metadata:
            for alert in event.metadata['alerts']:
                print(f"ALERT [{alert['level']}]: {alert['message']}")
    
    event_architecture.register_handler(StreamEventType.EARTHQUAKE, earthquake_handler)
    event_architecture.register_handler(StreamEventType.ALERT, alert_handler)
    
    # Start event processing
    await event_architecture.start_event_processing()
    
    # Create processing pipeline
    await flink_processor.create_geological_stream_pipeline(kafka_manager)
    
    # Generate sample events
    sample_events = [
        StreamEvent(
            event_id=f"eq_{i}",
            event_type=StreamEventType.EARTHQUAKE,
            timestamp=datetime.utcnow(),
            source_id="seismic_station_1",
            data={'magnitude': 4.5 + i * 0.5, 'depth': 10.0},
            location=(35.0 + i * 0.1, -118.0 + i * 0.1, 0)
        ) for i in range(10)
    ]
    
    # Process events
    for event in sample_events:
        await kafka_manager.produce_event(event)
        await aggregator.add_event(event)
        await event_architecture.publish_event(event)
        await asyncio.sleep(0.1)
    
    # Get aggregations
    await asyncio.sleep(2)  # Wait for processing
    aggregations = await aggregator.get_aggregations('minute')
    
    if aggregations:
        print(f"\nAggregation results:")
        for agg in aggregations[-1:]:  # Show latest
            print(f"  Window: {agg['window_type']}")
            print(f"  Events: {agg['event_count']}")
            print(f"  Event types: {agg['event_types']}")
            if 'numerical_stats' in agg:
                print(f"  Magnitude stats: {agg['numerical_stats'].get('magnitude', {})}")
    
    print("Stream processing example completed!")

if __name__ == "__main__":
    print("=== Real-time Stream Processing Framework ===")
    print(f"Available features:")
    print(f"  Apache Kafka: {KAFKA_AVAILABLE}")
    print(f"  Apache Flink: {FLINK_AVAILABLE}")
    print(f"  PostgreSQL: {POSTGRES_AVAILABLE}")
    print(f"  GIS Libraries: {GIS_AVAILABLE}")
    
    # Run example
    asyncio.run(example_stream_processing())