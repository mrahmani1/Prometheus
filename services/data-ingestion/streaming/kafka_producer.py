"""Kafka producer for streaming geological data"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
from models.geological_data import (
    ProcessedDataBatch,
    SeismicEvent,
    StrainMeasurement,
    ThermalMeasurement,
)

logger = structlog.get_logger()

class KafkaDataProducer:
    """Kafka producer for streaming geological data"""

    def __init__(self, config: Any):
        self.config = config
        self.bootstrap_servers = config.kafka_bootstrap_servers
        self.topic_prefix = config.kafka_topic_prefix
        self.producer = None
        self.is_initialized = False

        # Topic configuration
        self.topics = {
            "seismic_events": f"{self.topic_prefix}.seismic.events",
            "thermal_measurements": f"{self.topic_prefix}.thermal.measurements",
            "strain_measurements": f"{self.topic_prefix}.strain.measurements",
            "satellite_observations": f"{self.topic_prefix}.satellite.observations",
            "fiber_optic_measurements": f"{self.topic_prefix}.fiber.measurements",
            "processing_batches": f"{self.topic_prefix}.processing.batches",
            "alerts": f"{self.topic_prefix}.alerts",
            "quality_metrics": f"{self.topic_prefix}.quality.metrics",
        }

        # Producer statistics
        self.messages_sent = 0
        self.messages_failed = 0
        self.bytes_sent = 0

    async def initialize(self):
        """Initialize Kafka producer"""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=self._serialize_message,
                key_serializer=self._serialize_key,
                compression_type="gzip",
                batch_size=16384,  # 16KB batches
                linger_ms=10,  # Wait up to 10ms to batch messages
                acks="all",  # Wait for all replicas to acknowledge
                retries=3,
                retry_backoff_ms=100,
                request_timeout_ms=30000,
                enable_idempotence=True,
            )

            await self.producer.start()
            self.is_initialized = True

            logger.info(
                "Kafka producer initialized",
                bootstrap_servers=self.bootstrap_servers,
                topics=list(self.topics.values()),
            )

        except Exception as e:
            logger.error("Failed to initialize Kafka producer", error=str(e))
            raise

    def _serialize_message(self, message: Any) -> bytes:
        """Serialize message to JSON bytes"""
        try:
            if hasattr(message, "__dict__"):
                # Convert dataclass or object to dict
                message_dict = self._convert_to_dict(message)
            else:
                message_dict = message

            # Add metadata
            message_dict["_metadata"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "producer": "prometheus-data-ingestion",
                "version": "1.0.0",
            }

            return json.dumps(message_dict, default=self._json_serializer).encode(
                "utf-8"
            )

        except Exception as e:
            logger.error("Error serializing message", error=str(e))
            return json.dumps({"error": str(e)}).encode("utf-8")

    def _serialize_key(self, key: Any) -> bytes:
        """Serialize message key"""
        if key is None:
            return None
        return str(key).encode("utf-8")

    def _convert_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert object to dictionary for serialization"""
        if hasattr(obj, "__dict__"):
            result = {}
            for key, value in obj.__dict__.items():
                if key.startswith("_"):
                    continue

                if hasattr(value, "__dict__"):
                    result[key] = self._convert_to_dict(value)
                elif isinstance(value, (list, tuple)):
                    result[key] = [
                        (
                            self._convert_to_dict(item)
                            if hasattr(item, "__dict__")
                            else item
                        )
                        for item in value
                    ]
                else:
                    result[key] = value
            return result
        else:
            return obj

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "value"):  # Enum
            return obj.value
        elif hasattr(obj, "__dict__"):
            return self._convert_to_dict(obj)
        else:
            return str(obj)

    async def send_seismic_event(
        self, event: SeismicEvent, key: Optional[str] = None
    ) -> bool:
        """Send seismic event to Kafka"""
        return await self._send_message(
            topic=self.topics["seismic_events"],
            message=event,
            key=key or event.event_id,
        )

    async def send_thermal_measurement(
        self, measurement: ThermalMeasurement, key: Optional[str] = None
    ) -> bool:
        """Send thermal measurement to Kafka"""
        return await self._send_message(
            topic=self.topics["thermal_measurements"],
            message=measurement,
            key=key
            or f"{measurement.location.latitude}_{measurement.location.longitude}_{measurement.timestamp.isoformat()}",
        )

    async def send_strain_measurement(
        self, measurement: StrainMeasurement, key: Optional[str] = None
    ) -> bool:
        """Send strain measurement to Kafka"""
        return await self._send_message(
            topic=self.topics["strain_measurements"],
            message=measurement,
            key=key
            or f"ch_{measurement.channel_index}_{measurement.timestamp.isoformat()}",
        )

    async def send_processing_batch(
        self, batch: ProcessedDataBatch, key: Optional[str] = None
    ) -> bool:
        """Send processing batch metadata to Kafka"""
        return await self._send_message(
            topic=self.topics["processing_batches"],
            message=batch,
            key=key or batch.batch_id,
        )

    async def send_alert(
        self, alert: Dict[str, Any], key: Optional[str] = None
    ) -> bool:
        """Send alert to Kafka"""
        return await self._send_message(
            topic=self.topics["alerts"],
            message=alert,
            key=key or alert.get("alert_id", "unknown"),
        )

    async def send_quality_metrics(
        self, metrics: Dict[str, Any], key: Optional[str] = None
    ) -> bool:
        """Send quality metrics to Kafka"""
        return await self._send_message(
            topic=self.topics["quality_metrics"],
            message=metrics,
            key=key or f"metrics_{datetime.utcnow().isoformat()}",
        )

    async def _send_message(
        self, topic: str, message: Any, key: Optional[str] = None
    ) -> bool:
        """Send message to Kafka topic"""
        if not self.is_initialized:
            logger.error("Kafka producer not initialized")
            return False

        try:
            # Send message
            future = await self.producer.send(topic, message, key=key)
            record_metadata = await future

            # Update statistics
            self.messages_sent += 1
            message_bytes = len(self._serialize_message(message))
            self.bytes_sent += message_bytes

            logger.debug(
                "Message sent to Kafka",
                topic=topic,
                partition=record_metadata.partition,
                offset=record_metadata.offset,
                key=key,
                size_bytes=message_bytes,
            )

            return True

        except KafkaError as e:
            logger.error(
                "Kafka error sending message", topic=topic, key=key, error=str(e)
            )
            self.messages_failed += 1
            return False
        except Exception as e:
            logger.error(
                "Error sending message to Kafka", topic=topic, key=key, error=str(e)
            )
            self.messages_failed += 1
            return False

    async def send_batch_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Send multiple messages in batch"""
        if not self.is_initialized:
            logger.error("Kafka producer not initialized")
            return {"sent": 0, "failed": len(messages)}

        results = {"sent": 0, "failed": 0}

        try:
            # Send all messages
            futures = []
            for msg_data in messages:
                topic = msg_data.get("topic")
                message = msg_data.get("message")
                key = msg_data.get("key")

                if topic and message:
                    future = self.producer.send(topic, message, key=key)
                    futures.append((future, topic, key))
                else:
                    results["failed"] += 1

            # Wait for all messages to be sent
            for future, topic, key in futures:
                try:
                    await future
                    results["sent"] += 1
                    self.messages_sent += 1
                except Exception as e:
                    logger.error(
                        "Error in batch message", topic=topic, key=key, error=str(e)
                    )
                    results["failed"] += 1
                    self.messages_failed += 1

            logger.info(
                "Batch messages sent",
                total=len(messages),
                sent=results["sent"],
                failed=results["failed"],
            )

        except Exception as e:
            logger.error("Error sending batch messages", error=str(e))
            results["failed"] = len(messages)

        return results

    async def flush(self, timeout_ms: int = 10000):
        """Flush all pending messages"""
        if self.producer:
            try:
                await asyncio.wait_for(self.producer.flush(), timeout=timeout_ms / 1000)
                logger.debug("Kafka producer flushed")
            except asyncio.TimeoutError:
                logger.warning("Kafka producer flush timeout")
            except Exception as e:
                logger.error("Error flushing Kafka producer", error=str(e))

    async def get_statistics(self) -> Dict[str, Any]:
        """Get producer statistics"""
        return {
            "is_initialized": self.is_initialized,
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed,
            "bytes_sent": self.bytes_sent,
            "success_rate": (
                self.messages_sent / (self.messages_sent + self.messages_failed) * 100
                if (self.messages_sent + self.messages_failed) > 0
                else 0
            ),
            "topics": self.topics,
        }

    async def cleanup(self):
        """Cleanup Kafka producer"""
        if self.producer:
            try:
                await self.flush()
                await self.producer.stop()
                logger.info("Kafka producer stopped")
            except Exception as e:
                logger.error("Error stopping Kafka producer", error=str(e))

        self.is_initialized = False
