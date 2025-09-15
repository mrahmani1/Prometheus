"""
Advanced Apache Kafka Integration for Real-Time Geological Data Streaming

This module provides production-grade Kafka integration for streaming geological data
between databases with advanced features including schema registry, exactly-once
semantics, and geological data serialization.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
import avro.schema
import numpy as np
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class GeologicalDataType(Enum):
    """Enumeration of geological data types for Kafka topic routing"""
    SEISMIC_WAVEFORM = "seismic_waveform"
    GPS_DEFORMATION = "gps_deformation"
    MTSAR_INTERFEROMETRY = "mtsar_interferometry"
    GEOCHEMICAL_ANALYSIS = "geochemical_analysis"
    THERMAL_ANOMALY = "thermal_anomaly"
    VOLCANIC_GAS = "volcanic_gas"
    GRAVITY_MAGNETIC = "gravity_magnetic"
    BOREHOLE_DATA = "borehole_data"
    GEOLOGICAL_SURVEY = "geological_survey"
    CAUSAL_RELATIONSHIP = "causal_relationship"


@dataclass
class GeologicalDataMessage:
    """Standardized geological data message for Kafka streaming"""
    message_id: str
    timestamp: datetime
    data_type: GeologicalDataType
    source_system: str
    location: Dict[str, float]  # lat, lon, elevation
    payload: Dict[str, Any]
    quality_score: float
    processing_metadata: Dict[str, Any]
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['data_type'] = self.data_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeologicalDataMessage':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['data_type'] = GeologicalDataType(data['data_type'])
        return cls(**data)


class KafkaTopicManager:
    """Advanced Kafka topic management for geological data streams"""
    
    def __init__(self, bootstrap_servers: str, schema_registry_url: str):
        self.bootstrap_servers = bootstrap_servers
        self.schema_registry_url = schema_registry_url
        self.schema_registry_client = SchemaRegistryClient({'url': schema_registry_url})
        self.topic_configs = self._get_geological_topic_configs()
        
    def _get_geological_topic_configs(self) -> Dict[str, Dict[str, Any]]:
        """Define topic configurations for different geological data types"""
        return {
            'geological-seismic-realtime': {
                'partitions': 12,
                'replication_factor': 3,
                'config': {
                    'retention.ms': str(7 * 24 * 60 * 60 * 1000),  # 7 days
                    'compression.type': 'lz4',
                    'max.message.bytes': str(10 * 1024 * 1024),  # 10MB
                    'min.insync.replicas': '2'
                }
            },
            'geological-deformation-timeseries': {
                'partitions': 8,
                'replication_factor': 3,
                'config': {
                    'retention.ms': str(30 * 24 * 60 * 60 * 1000),  # 30 days
                    'compression.type': 'snappy',
                    'cleanup.policy': 'compact,delete'
                }
            },
            'geological-geochemical-analysis': {
                'partitions': 4,
                'replication_factor': 3,
                'config': {
                    'retention.ms': str(365 * 24 * 60 * 60 * 1000),  # 1 year
                    'compression.type': 'gzip'
                }
            },
            'geological-causal-relationships': {
                'partitions': 6,
                'replication_factor': 3,
                'config': {
                    'retention.ms': str(90 * 24 * 60 * 60 * 1000),  # 90 days
                    'compression.type': 'snappy'
                }
            },
            'geological-data-quality-metrics': {
                'partitions': 2,
                'replication_factor': 3,
                'config': {
                    'retention.ms': str(14 * 24 * 60 * 60 * 1000),  # 14 days
                    'compression.type': 'lz4'
                }
            }
        }
    
    async def create_topics(self) -> None:
        """Create all geological data topics with appropriate configurations"""
        from aiokafka.admin import AIOKafkaAdminClient, NewTopic
        
        admin_client = AIOKafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
        
        try:
            await admin_client.start()
            
            # Get existing topics
            existing_topics = await admin_client.list_topics()
            
            # Create new topics
            topics_to_create = []
            for topic_name, config in self.topic_configs.items():
                if topic_name not in existing_topics:
                    new_topic = NewTopic(
                        name=topic_name,
                        num_partitions=config['partitions'],
                        replication_factor=config['replication_factor'],
                        topic_configs=config['config']
                    )
                    topics_to_create.append(new_topic)
            
            if topics_to_create:
                await admin_client.create_topics(topics_to_create)
                logger.info("Created Kafka topics", topics=len(topics_to_create))
            
        finally:
            await admin_client.close()


class GeologicalDataSerializer:
    """Advanced serialization for geological data with schema evolution support"""
    
    def __init__(self, schema_registry_client: SchemaRegistryClient):
        self.schema_registry = schema_registry_client
        self.avro_serializers = {}
        self.avro_deserializers = {}
        self._register_schemas()
    
    def _register_schemas(self) -> None:
        """Register Avro schemas for geological data types"""
        geological_data_schema = {
            "type": "record",
            "name": "GeologicalDataMessage",
            "fields": [
                {"name": "message_id", "type": "string"},
                {"name": "timestamp", "type": "string"},
                {"name": "data_type", "type": "string"},
                {"name": "source_system", "type": "string"},
                {"name": "location", "type": {"type": "map", "values": "double"}},
                {"name": "payload", "type": {"type": "map", "values": ["null", "string", "double", "boolean"]}},
                {"name": "quality_score", "type": "double"},
                {"name": "processing_metadata", "type": {"type": "map", "values": "string"}},
                {"name": "correlation_id", "type": ["null", "string"], "default": None}
            ]
        }
        
        # Register schema with schema registry
        schema_str = json.dumps(geological_data_schema)
        self.schema_registry.register_schema(
            "geological-data-value", 
            avro.schema.parse(schema_str)
        )
    
    def get_serializer(self, topic: str) -> AvroSerializer:
        """Get Avro serializer for topic"""
        if topic not in self.avro_serializers:
            schema_str = self.schema_registry.get_latest_version("geological-data-value")
            self.avro_serializers[topic] = AvroSerializer(
                self.schema_registry,
                schema_str.schema.schema_str
            )
        return self.avro_serializers[topic]
    
    def get_deserializer(self, topic: str) -> AvroDeserializer:
        """Get Avro deserializer for topic"""
        if topic not in self.avro_deserializers:
            self.avro_deserializers[topic] = AvroDeserializer(
                self.schema_registry,
                "geological-data-value"
            )
        return self.avro_deserializers[topic]


class GeologicalKafkaProducer:
    """High-performance Kafka producer for geological data streams"""
    
    def __init__(self, 
                 bootstrap_servers: str,
                 schema_registry_url: str,
                 enable_idempotence: bool = True,
                 max_batch_size: int = 16384,
                 linger_ms: int = 10):
        self.bootstrap_servers = bootstrap_servers
        self.schema_registry_url = schema_registry_url
        self.producer = None
        self.serializer = None
        self.topic_manager = KafkaTopicManager(bootstrap_servers, schema_registry_url)
        
        # Producer configuration for exactly-once semantics
        self.producer_config = {
            'bootstrap_servers': bootstrap_servers,
            'enable_idempotence': enable_idempotence,
            'acks': 'all',
            'retries': 2147483647,  # Max retries
            'max_in_flight_requests_per_connection': 5,
            'batch_size': max_batch_size,
            'linger_ms': linger_ms,
            'compression_type': 'snappy',
            'request_timeout_ms': 30000,
            'delivery_timeout_ms': 120000
        }
    
    async def start(self) -> None:
        """Initialize producer and create topics"""
        self.producer = AIOKafkaProducer(**self.producer_config)
        await self.producer.start()
        
        # Initialize serializer
        schema_registry_client = SchemaRegistryClient({'url': self.schema_registry_url})
        self.serializer = GeologicalDataSerializer(schema_registry_client)
        
        # Create topics
        await self.topic_manager.create_topics()
        
        logger.info("Geological Kafka producer started")
    
    async def stop(self) -> None:
        """Stop producer"""
        if self.producer:
            await self.producer.stop()
            logger.info("Geological Kafka producer stopped")
    
    def _get_topic_for_data_type(self, data_type: GeologicalDataType) -> str:
        """Map geological data type to Kafka topic"""
        topic_mapping = {
            GeologicalDataType.SEISMIC_WAVEFORM: 'geological-seismic-realtime',
            GeologicalDataType.GPS_DEFORMATION: 'geological-deformation-timeseries',
            GeologicalDataType.MTSAR_INTERFEROMETRY: 'geological-deformation-timeseries',
            GeologicalDataType.GEOCHEMICAL_ANALYSIS: 'geological-geochemical-analysis',
            GeologicalDataType.THERMAL_ANOMALY: 'geological-seismic-realtime',
            GeologicalDataType.VOLCANIC_GAS: 'geological-geochemical-analysis',
            GeologicalDataType.GRAVITY_MAGNETIC: 'geological-deformation-timeseries',
            GeologicalDataType.BOREHOLE_DATA: 'geological-geochemical-analysis',
            GeologicalDataType.GEOLOGICAL_SURVEY: 'geological-geochemical-analysis',
            GeologicalDataType.CAUSAL_RELATIONSHIP: 'geological-causal-relationships'
        }
        return topic_mapping.get(data_type, 'geological-seismic-realtime')
    
    async def send_geological_data(self, 
                                 message: GeologicalDataMessage,
                                 partition_key: Optional[str] = None) -> None:
        """Send geological data message to appropriate Kafka topic"""
        try:
            topic = self._get_topic_for_data_type(message.data_type)
            
            # Serialize message
            serialized_data = json.dumps(message.to_dict()).encode('utf-8')
            
            # Use location-based partitioning if no key provided
            if partition_key is None:
                partition_key = f"{message.location['lat']:.2f},{message.location['lon']:.2f}"
            
            # Send to Kafka
            await self.producer.send_and_wait(
                topic=topic,
                value=serialized_data,
                key=partition_key.encode('utf-8'),
                headers=[
                    ('data_type', message.data_type.value.encode('utf-8')),
                    ('source_system', message.source_system.encode('utf-8')),
                    ('quality_score', str(message.quality_score).encode('utf-8'))
                ]
            )
            
            logger.debug("Sent geological data message", 
                        message_id=message.message_id,
                        topic=topic,
                        data_type=message.data_type.value)
            
        except Exception as e:
            logger.error("Failed to send geological data message", 
                        message_id=message.message_id,
                        error=str(e))
            raise
    
    async def send_batch_geological_data(self, 
                                       messages: List[GeologicalDataMessage]) -> None:
        """Send batch of geological data messages efficiently"""
        try:
            # Group messages by topic for efficient batching
            topic_groups = {}
            for message in messages:
                topic = self._get_topic_for_data_type(message.data_type)
                if topic not in topic_groups:
                    topic_groups[topic] = []
                topic_groups[topic].append(message)
            
            # Send batches to each topic
            for topic, topic_messages in topic_groups.items():
                batch_futures = []
                
                for message in topic_messages:
                    serialized_data = json.dumps(message.to_dict()).encode('utf-8')
                    partition_key = f"{message.location['lat']:.2f},{message.location['lon']:.2f}"
                    
                    future = self.producer.send(
                        topic=topic,
                        value=serialized_data,
                        key=partition_key.encode('utf-8'),
                        headers=[
                            ('data_type', message.data_type.value.encode('utf-8')),
                            ('source_system', message.source_system.encode('utf-8'))
                        ]
                    )
                    batch_futures.append(future)
                
                # Wait for all messages in batch to be sent
                await asyncio.gather(*batch_futures)
            
            logger.info("Sent batch geological data", 
                       total_messages=len(messages),
                       topics=list(topic_groups.keys()))
            
        except Exception as e:
            logger.error("Failed to send batch geological data", error=str(e))
            raise


class GeologicalKafkaConsumer:
    """High-performance Kafka consumer for geological data streams"""
    
    def __init__(self, 
                 bootstrap_servers: str,
                 schema_registry_url: str,
                 consumer_group: str,
                 topics: List[str],
                 auto_offset_reset: str = 'latest'):
        self.bootstrap_servers = bootstrap_servers
        self.schema_registry_url = schema_registry_url
        self.consumer_group = consumer_group
        self.topics = topics
        self.consumer = None
        self.deserializer = None
        self.message_handlers: Dict[GeologicalDataType, Callable] = {}
        
        # Consumer configuration
        self.consumer_config = {
            'bootstrap_servers': bootstrap_servers,
            'group_id': consumer_group,
            'auto_offset_reset': auto_offset_reset,
            'enable_auto_commit': False,  # Manual commit for exactly-once processing
            'max_poll_records': 500,
            'fetch_min_bytes': 1024,
            'fetch_max_wait_ms': 500,
            'session_timeout_ms': 30000,
            'heartbeat_interval_ms': 10000
        }
    
    async def start(self) -> None:
        """Initialize consumer"""
        self.consumer = AIOKafkaConsumer(
            *self.topics,
            **self.consumer_config
        )
        await self.consumer.start()
        
        # Initialize deserializer
        schema_registry_client = SchemaRegistryClient({'url': self.schema_registry_url})
        self.deserializer = GeologicalDataSerializer(schema_registry_client)
        
        logger.info("Geological Kafka consumer started", 
                   group=self.consumer_group,
                   topics=self.topics)
    
    async def stop(self) -> None:
        """Stop consumer"""
        if self.consumer:
            await self.consumer.stop()
            logger.info("Geological Kafka consumer stopped")
    
    def register_handler(self, 
                        data_type: GeologicalDataType, 
                        handler: Callable[[GeologicalDataMessage], None]) -> None:
        """Register message handler for specific geological data type"""
        self.message_handlers[data_type] = handler
        logger.info("Registered handler", data_type=data_type.value)
    
    async def consume_messages(self) -> AsyncIterator[GeologicalDataMessage]:
        """Consume geological data messages from Kafka"""
        try:
            async for message in self.consumer:
                try:
                    # Deserialize message
                    message_data = json.loads(message.value.decode('utf-8'))
                    geological_message = GeologicalDataMessage.from_dict(message_data)
                    
                    # Process message with registered handler
                    if geological_message.data_type in self.message_handlers:
                        handler = self.message_handlers[geological_message.data_type]
                        await handler(geological_message)
                    
                    # Commit offset after successful processing
                    await self.consumer.commit()
                    
                    yield geological_message
                    
                except Exception as e:
                    logger.error("Failed to process message", 
                               topic=message.topic,
                               partition=message.partition,
                               offset=message.offset,
                               error=str(e))
                    # Skip problematic message and continue
                    await self.consumer.commit()
                    
        except Exception as e:
            logger.error("Consumer error", error=str(e))
            raise


class GeologicalDataStreamProcessor:
    """Advanced stream processor for real-time geological data analysis"""
    
    def __init__(self, 
                 bootstrap_servers: str,
                 schema_registry_url: str):
        self.bootstrap_servers = bootstrap_servers
        self.schema_registry_url = schema_registry_url
        self.producer = GeologicalKafkaProducer(bootstrap_servers, schema_registry_url)
        self.consumers: Dict[str, GeologicalKafkaConsumer] = {}
        self.processing_pipelines: Dict[str, Callable] = {}
        
    async def start(self) -> None:
        """Start stream processor"""
        await self.producer.start()
        logger.info("Geological data stream processor started")
    
    async def stop(self) -> None:
        """Stop stream processor"""
        await self.producer.stop()
        for consumer in self.consumers.values():
            await consumer.stop()
        logger.info("Geological data stream processor stopped")
    
    def register_processing_pipeline(self, 
                                   pipeline_name: str,
                                   input_topics: List[str],
                                   output_topic: str,
                                   processor_func: Callable) -> None:
        """Register data processing pipeline"""
        self.processing_pipelines[pipeline_name] = {
            'input_topics': input_topics,
            'output_topic': output_topic,
            'processor': processor_func
        }
        
        # Create consumer for input topics
        consumer = GeologicalKafkaConsumer(
            self.bootstrap_servers,
            self.schema_registry_url,
            f"processor-{pipeline_name}",
            input_topics
        )
        self.consumers[pipeline_name] = consumer
    
    async def run_processing_pipeline(self, pipeline_name: str) -> None:
        """Run specific processing pipeline"""
        if pipeline_name not in self.processing_pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not registered")
        
        pipeline = self.processing_pipelines[pipeline_name]
        consumer = self.consumers[pipeline_name]
        
        await consumer.start()
        
        try:
            async for message in consumer.consume_messages():
                # Process message through pipeline
                processed_data = await pipeline['processor'](message)
                
                if processed_data:
                    # Send processed data to output topic
                    await self.producer.send_geological_data(processed_data)
                    
        except Exception as e:
            logger.error("Pipeline processing error", 
                        pipeline=pipeline_name,
                        error=str(e))
            raise
        finally:
            await consumer.stop()


# Example usage and testing
async def example_kafka_integration():
    """Example of using the geological Kafka integration"""
    
    # Initialize producer
    producer = GeologicalKafkaProducer(
        bootstrap_servers='localhost:9092',
        schema_registry_url='http://localhost:8081'
    )
    
    await producer.start()
    
    try:
        # Create sample geological data message
        sample_message = GeologicalDataMessage(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            data_type=GeologicalDataType.SEISMIC_WAVEFORM,
            source_system='IRIS-USGS',
            location={'lat': 19.4194, 'lon': -155.2885, 'elevation': 1222.0},
            payload={
                'station': 'KILAUEA-HVO',
                'magnitude': 2.3,
                'depth_km': 5.2,
                'waveform_data': [0.1, 0.2, 0.15, 0.3, 0.25]
            },
            quality_score=0.95,
            processing_metadata={
                'processing_time': datetime.utcnow().isoformat(),
                'algorithm_version': '2.1.0',
                'data_source': 'real-time'
            }
        )
        
        # Send message
        await producer.send_geological_data(sample_message)
        logger.info("Sent sample geological data message")
        
    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(example_kafka_integration())