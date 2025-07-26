"""Redis Streams for low-latency caching and immediate analysis"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import structlog
import redis.asyncio as redis
from redis.exceptions import RedisError

logger = structlog.get_logger()

class RedisStreamsManager:
    """Redis Streams manager for low-latency geological data processing"""
    
    def __init__(self, config: Any):
        self.config = config
        self.redis_url = config.redis_url
        self.redis_client = None
        self.is_initialized = False
        
        # Stream configuration
        self.stream_prefix = "prometheus"
        self.streams = {
            "seismic_events": f"{self.stream_prefix}:seismic:events",
            "thermal_measurements": f"{self.stream_prefix}:thermal:measurements",
            "strain_measurements": f"{self.stream_prefix}:strain:measurements",
            "alerts": f"{self.stream_prefix}:alerts",
            "real_time_analysis": f"{self.stream_prefix}:analysis:realtime",
            "quality_metrics": f"{self.stream_prefix}:quality:metrics"
        }
        
        # Consumer groups
        self.consumer_groups = {
            "data_processor": "data-processor-group",
            "alert_handler": "alert-handler-group",
            "quality_monitor": "quality-monitor-group",
            "real_time_analyzer": "realtime-analyzer-group"
        }
        
        # Stream settings
        self.max_stream_length = 10000  # Keep last 10k messages per stream
        self.consumer_timeout_ms = 1000  # 1 second timeout
        self.block_time_ms = 100  # 100ms block time for XREAD
        
        # Statistics
        self.messages_added = 0
        self.messages_read = 0
        self.bytes_cached = 0
        
    async def initialize(self):
        """Initialize Redis streams"""
        try:
            # Create Redis connection
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Create consumer groups for all streams
            await self._create_consumer_groups()
            
            self.is_initialized = True
            
            logger.info(
                "Redis streams initialized",
                redis_url=self.redis_url,
                streams=list(self.streams.values()),
                consumer_groups=list(self.consumer_groups.values())
            )
            
        except Exception as e:
            logger.error("Failed to initialize Redis streams", error=str(e))
            raise
    
    async def _create_consumer_groups(self):
        """Create consumer groups for all streams"""
        for stream_name in self.streams.values():
            for group_name in self.consumer_groups.values():
                try:
                    await self.redis_client.xgroup_create(
                        stream_name, group_name, id='0', mkstream=True
                    )
                    logger.debug(f"Created consumer group {group_name} for stream {stream_name}")
                except redis.ResponseError as e:
                    if "BUSYGROUP" in str(e):
                        # Group already exists
                        logger.debug(f"Consumer group {group_name} already exists for {stream_name}")
                    else:
                        logger.warning(f"Error creating consumer group: {e}")
    
    async def add_seismic_event(self, event_data: Dict[str, Any], event_id: Optional[str] = None) -> str:
        """Add seismic event to stream"""
        return await self._add_to_stream(
            stream_name=self.streams["seismic_events"],
            data=event_data,
            message_id=event_id
        )
    
    async def add_thermal_measurement(self, measurement_data: Dict[str, Any], measurement_id: Optional[str] = None) -> str:
        """Add thermal measurement to stream"""
        return await self._add_to_stream(
            stream_name=self.streams["thermal_measurements"],
            data=measurement_data,
            message_id=measurement_id
        )
    
    async def add_strain_measurement(self, measurement_data: Dict[str, Any], measurement_id: Optional[str] = None) -> str:
        """Add strain measurement to stream"""
        return await self._add_to_stream(
            stream_name=self.streams["strain_measurements"],
            data=measurement_data,
            message_id=measurement_id
        )
    
    async def add_alert(self, alert_data: Dict[str, Any], alert_id: Optional[str] = None) -> str:
        """Add alert to stream"""
        return await self._add_to_stream(
            stream_name=self.streams["alerts"],
            data=alert_data,
            message_id=alert_id
        )
    
    async def add_real_time_analysis(self, analysis_data: Dict[str, Any], analysis_id: Optional[str] = None) -> str:
        """Add real-time analysis result to stream"""
        return await self._add_to_stream(
            stream_name=self.streams["real_time_analysis"],
            data=analysis_data,
            message_id=analysis_id
        )
    
    async def add_quality_metrics(self, metrics_data: Dict[str, Any], metrics_id: Optional[str] = None) -> str:
        """Add quality metrics to stream"""
        return await self._add_to_stream(
            stream_name=self.streams["quality_metrics"],
            data=metrics_data,
            message_id=metrics_id
        )
    
    async def _add_to_stream(
        self, 
        stream_name: str, 
        data: Dict[str, Any], 
        message_id: Optional[str] = None,
        max_length: Optional[int] = None
    ) -> str:
        """Add message to Redis stream"""
        if not self.is_initialized:
            raise RuntimeError("Redis streams not initialized")
        
        try:
            # Prepare message data
            message_data = {
                "data": json.dumps(data, default=self._json_serializer),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "prometheus-data-ingestion"
            }
            
            # Add to stream with optional trimming
            stream_id = await self.redis_client.xadd(
                stream_name,
                message_data,
                id=message_id or '*',
                maxlen=max_length or self.max_stream_length,
                approximate=True  # Use approximate trimming for better performance
            )\n            \n            # Update statistics\n            self.messages_added += 1\n            self.bytes_cached += len(json.dumps(message_data))\n            \n            logger.debug(\n                \"Message added to stream\",\n                stream=stream_name,\n                message_id=stream_id,\n                data_size=len(json.dumps(data))\n            )\n            \n            return stream_id\n            \n        except RedisError as e:\n            logger.error(\n                \"Redis error adding to stream\",\n                stream=stream_name,\n                error=str(e)\n            )\n            raise\n        except Exception as e:\n            logger.error(\n                \"Error adding to stream\",\n                stream=stream_name,\n                error=str(e)\n            )\n            raise\n    \n    def _json_serializer(self, obj: Any) -> Any:\n        \"\"\"Custom JSON serializer for special types\"\"\"\n        if isinstance(obj, datetime):\n            return obj.isoformat()\n        elif hasattr(obj, 'value'):  # Enum\n            return obj.value\n        elif hasattr(obj, '__dict__'):\n            return obj.__dict__\n        else:\n            return str(obj)\n    \n    async def read_stream_messages(\n        self, \n        stream_name: str, \n        consumer_group: str,\n        consumer_name: str,\n        count: int = 10,\n        block_ms: Optional[int] = None\n    ) -> List[Tuple[str, Dict[str, Any]]]:\n        \"\"\"Read messages from stream using consumer group\"\"\"\n        if not self.is_initialized:\n            raise RuntimeError(\"Redis streams not initialized\")\n        \n        try:\n            # Read messages from consumer group\n            messages = await self.redis_client.xreadgroup(\n                consumer_group,\n                consumer_name,\n                {stream_name: '>'},\n                count=count,\n                block=block_ms or self.block_time_ms\n            )\n            \n            processed_messages = []\n            \n            for stream, stream_messages in messages:\n                for message_id, fields in stream_messages:\n                    try:\n                        # Parse message data\n                        data = json.loads(fields.get('data', '{}'))\n                        timestamp = fields.get('timestamp')\n                        source = fields.get('source')\n                        \n                        message_info = {\n                            'id': message_id,\n                            'data': data,\n                            'timestamp': timestamp,\n                            'source': source,\n                            'stream': stream\n                        }\n                        \n                        processed_messages.append((message_id, message_info))\n                        \n                    except json.JSONDecodeError as e:\n                        logger.warning(\n                            \"Error parsing message data\",\n                            message_id=message_id,\n                            error=str(e)\n                        )\n            \n            self.messages_read += len(processed_messages)\n            \n            logger.debug(\n                \"Read messages from stream\",\n                stream=stream_name,\n                consumer_group=consumer_group,\n                consumer_name=consumer_name,\n                count=len(processed_messages)\n            )\n            \n            return processed_messages\n            \n        except RedisError as e:\n            logger.error(\n                \"Redis error reading from stream\",\n                stream=stream_name,\n                consumer_group=consumer_group,\n                error=str(e)\n            )\n            return []\n        except Exception as e:\n            logger.error(\n                \"Error reading from stream\",\n                stream=stream_name,\n                consumer_group=consumer_group,\n                error=str(e)\n            )\n            return []\n    \n    async def acknowledge_messages(\n        self, \n        stream_name: str, \n        consumer_group: str, \n        message_ids: List[str]\n    ) -> int:\n        \"\"\"Acknowledge processed messages\"\"\"\n        if not self.is_initialized:\n            raise RuntimeError(\"Redis streams not initialized\")\n        \n        try:\n            ack_count = await self.redis_client.xack(\n                stream_name, consumer_group, *message_ids\n            )\n            \n            logger.debug(\n                \"Acknowledged messages\",\n                stream=stream_name,\n                consumer_group=consumer_group,\n                count=ack_count\n            )\n            \n            return ack_count\n            \n        except RedisError as e:\n            logger.error(\n                \"Redis error acknowledging messages\",\n                stream=stream_name,\n                consumer_group=consumer_group,\n                error=str(e)\n            )\n            return 0\n        except Exception as e:\n            logger.error(\n                \"Error acknowledging messages\",\n                stream=stream_name,\n                consumer_group=consumer_group,\n                error=str(e)\n            )\n            return 0\n    \n    async def get_stream_info(self, stream_name: str) -> Dict[str, Any]:\n        \"\"\"Get information about a stream\"\"\"\n        if not self.is_initialized:\n            raise RuntimeError(\"Redis streams not initialized\")\n        \n        try:\n            info = await self.redis_client.xinfo_stream(stream_name)\n            \n            return {\n                \"length\": info.get('length', 0),\n                \"radix_tree_keys\": info.get('radix-tree-keys', 0),\n                \"radix_tree_nodes\": info.get('radix-tree-nodes', 0),\n                \"groups\": info.get('groups', 0),\n                \"last_generated_id\": info.get('last-generated-id', ''),\n                \"first_entry\": info.get('first-entry'),\n                \"last_entry\": info.get('last-entry')\n            }\n            \n        except RedisError as e:\n            logger.error(\n                \"Redis error getting stream info\",\n                stream=stream_name,\n                error=str(e)\n            )\n            return {}\n        except Exception as e:\n            logger.error(\n                \"Error getting stream info\",\n                stream=stream_name,\n                error=str(e)\n            )\n            return {}\n    \n    async def get_consumer_group_info(self, stream_name: str) -> List[Dict[str, Any]]:\n        \"\"\"Get information about consumer groups for a stream\"\"\"\n        if not self.is_initialized:\n            raise RuntimeError(\"Redis streams not initialized\")\n        \n        try:\n            groups = await self.redis_client.xinfo_groups(stream_name)\n            \n            group_info = []\n            for group in groups:\n                group_info.append({\n                    \"name\": group.get('name'),\n                    \"consumers\": group.get('consumers', 0),\n                    \"pending\": group.get('pending', 0),\n                    \"last_delivered_id\": group.get('last-delivered-id')\n                })\n            \n            return group_info\n            \n        except RedisError as e:\n            logger.error(\n                \"Redis error getting consumer group info\",\n                stream=stream_name,\n                error=str(e)\n            )\n            return []\n        except Exception as e:\n            logger.error(\n                \"Error getting consumer group info\",\n                stream=stream_name,\n                error=str(e)\n            )\n            return []\n    \n    async def get_pending_messages(\n        self, \n        stream_name: str, \n        consumer_group: str,\n        consumer_name: Optional[str] = None\n    ) -> List[Dict[str, Any]]:\n        \"\"\"Get pending messages for consumer group or specific consumer\"\"\"\n        if not self.is_initialized:\n            raise RuntimeError(\"Redis streams not initialized\")\n        \n        try:\n            if consumer_name:\n                # Get pending messages for specific consumer\n                pending = await self.redis_client.xpending_range(\n                    stream_name, consumer_group, \n                    min='-', max='+', count=100,\n                    consumername=consumer_name\n                )\n            else:\n                # Get pending messages for entire group\n                pending = await self.redis_client.xpending_range(\n                    stream_name, consumer_group,\n                    min='-', max='+', count=100\n                )\n            \n            pending_info = []\n            for msg in pending:\n                pending_info.append({\n                    \"message_id\": msg['message_id'],\n                    \"consumer\": msg['consumer'],\n                    \"time_since_delivered\": msg['time_since_delivered'],\n                    \"times_delivered\": msg['times_delivered']\n                })\n            \n            return pending_info\n            \n        except RedisError as e:\n            logger.error(\n                \"Redis error getting pending messages\",\n                stream=stream_name,\n                consumer_group=consumer_group,\n                error=str(e)\n            )\n            return []\n        except Exception as e:\n            logger.error(\n                \"Error getting pending messages\",\n                stream=stream_name,\n                consumer_group=consumer_group,\n                error=str(e)\n            )\n            return []\n    \n    async def trim_stream(self, stream_name: str, max_length: int) -> int:\n        \"\"\"Trim stream to maximum length\"\"\"\n        if not self.is_initialized:\n            raise RuntimeError(\"Redis streams not initialized\")\n        \n        try:\n            trimmed = await self.redis_client.xtrim(\n                stream_name, maxlen=max_length, approximate=True\n            )\n            \n            logger.info(\n                \"Trimmed stream\",\n                stream=stream_name,\n                max_length=max_length,\n                trimmed_count=trimmed\n            )\n            \n            return trimmed\n            \n        except RedisError as e:\n            logger.error(\n                \"Redis error trimming stream\",\n                stream=stream_name,\n                error=str(e)\n            )\n            return 0\n        except Exception as e:\n            logger.error(\n                \"Error trimming stream\",\n                stream=stream_name,\n                error=str(e)\n            )\n            return 0\n    \n    async def get_recent_messages(\n        self, \n        stream_name: str, \n        count: int = 100,\n        start_id: str = '-',\n        end_id: str = '+'\n    ) -> List[Tuple[str, Dict[str, Any]]]:\n        \"\"\"Get recent messages from stream without consumer group\"\"\"\n        if not self.is_initialized:\n            raise RuntimeError(\"Redis streams not initialized\")\n        \n        try:\n            messages = await self.redis_client.xrange(\n                stream_name, min=start_id, max=end_id, count=count\n            )\n            \n            processed_messages = []\n            \n            for message_id, fields in messages:\n                try:\n                    data = json.loads(fields.get('data', '{}'))\n                    timestamp = fields.get('timestamp')\n                    source = fields.get('source')\n                    \n                    message_info = {\n                        'id': message_id,\n                        'data': data,\n                        'timestamp': timestamp,\n                        'source': source,\n                        'stream': stream_name\n                    }\n                    \n                    processed_messages.append((message_id, message_info))\n                    \n                except json.JSONDecodeError as e:\n                    logger.warning(\n                        \"Error parsing message data\",\n                        message_id=message_id,\n                        error=str(e)\n                    )\n            \n            logger.debug(\n                \"Retrieved recent messages\",\n                stream=stream_name,\n                count=len(processed_messages)\n            )\n            \n            return processed_messages\n            \n        except RedisError as e:\n            logger.error(\n                \"Redis error getting recent messages\",\n                stream=stream_name,\n                error=str(e)\n            )\n            return []\n        except Exception as e:\n            logger.error(\n                \"Error getting recent messages\",\n                stream=stream_name,\n                error=str(e)\n            )\n            return []\n    \n    async def get_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get Redis streams statistics\"\"\"\n        stats = {\n            \"is_initialized\": self.is_initialized,\n            \"messages_added\": self.messages_added,\n            \"messages_read\": self.messages_read,\n            \"bytes_cached\": self.bytes_cached,\n            \"streams\": {},\n            \"consumer_groups\": self.consumer_groups\n        }\n        \n        if self.is_initialized:\n            # Get info for each stream\n            for stream_key, stream_name in self.streams.items():\n                stream_info = await self.get_stream_info(stream_name)\n                group_info = await self.get_consumer_group_info(stream_name)\n                \n                stats[\"streams\"][stream_key] = {\n                    \"name\": stream_name,\n                    \"info\": stream_info,\n                    \"consumer_groups\": group_info\n                }\n        \n        return stats\n    \n    async def cleanup(self):\n        \"\"\"Cleanup Redis streams\"\"\"\n        if self.redis_client:\n            try:\n                await self.redis_client.close()\n                logger.info(\"Redis streams connection closed\")\n            except Exception as e:\n                logger.error(\"Error closing Redis connection\", error=str(e))\n        \n        self.is_initialized = False