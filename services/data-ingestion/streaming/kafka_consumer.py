"""Kafka consumer for processing geological data streams"""

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import structlog
from aiokafka import AIOKafkaConsumer

logger = structlog.get_logger()


class KafkaDataConsumer:
    """Kafka consumer for processing geological data streams"""

    def __init__(self, config: Any):
        self.config = config
        self.bootstrap_servers = config.kafka_bootstrap_servers
        self.topic_prefix = config.kafka_topic_prefix
        self.consumer = None
        self.is_initialized = False
        self.is_consuming = False

        # Consumer configuration
        self.group_id = "prometheus-data-processor"
        self.auto_offset_reset = "latest"
        self.enable_auto_commit = True
        self.auto_commit_interval_ms = 1000

        # Message handlers
        self.message_handlers = {}

        # Consumer statistics
        self.messages_consumed = 0
        self.messages_processed = 0
        self.messages_failed = 0
        self.bytes_consumed = 0

        # Topics to consume
        self.topics = [
            f"{self.topic_prefix}.seismic.events",
            f"{self.topic_prefix}.thermal.measurements",
            f"{self.topic_prefix}.strain.measurements",
            f"{self.topic_prefix}.satellite.observations",
            f"{self.topic_prefix}.fiber.measurements",
            f"{self.topic_prefix}.processing.batches",
            f"{self.topic_prefix}.alerts",
            f"{self.topic_prefix}.quality.metrics",
        ]

    async def initialize(self):
        """Initialize Kafka consumer"""
        try:
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=self.enable_auto_commit,
                auto_commit_interval_ms=self.auto_commit_interval_ms,
                value_deserializer=self._deserialize_message,
                key_deserializer=self._deserialize_key,
                consumer_timeout_ms=1000,  # 1 second timeout
                fetch_max_wait_ms=500,  # Max wait for fetch
                max_partition_fetch_bytes=1048576,  # 1MB max per partition
                session_timeout_ms=30000,  # 30 second session timeout
                heartbeat_interval_ms=3000,  # 3 second heartbeat
            )

            await self.consumer.start()
            self.is_initialized = True

            logger.info(
                "Kafka consumer initialized",
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                topics=self.topics,
            )

        except Exception as e:
            logger.error("Failed to initialize Kafka consumer", error=str(e))
            raise

    def _deserialize_message(self, message_bytes: bytes) -> Dict[str, Any]:
        """Deserialize message from JSON bytes"""
        try:
            if message_bytes is None:
                return {}

            message_str = message_bytes.decode("utf-8")
            return json.loads(message_str)

        except Exception as e:
            logger.error("Error deserializing message", error=str(e))
            return {"error": f"Deserialization failed: {str(e)}"}

    def _deserialize_key(self, key_bytes: bytes) -> Optional[str]:
        """Deserialize message key"""
        if key_bytes is None:
            return None
        return key_bytes.decode("utf-8")

    def register_handler(
        self, topic_pattern: str, handler: Callable[[Dict[str, Any]], None]
    ):
        """Register message handler for topic pattern"""
        self.message_handlers[topic_pattern] = handler
        logger.info(f"Registered handler for topic pattern: {topic_pattern}")

    async def start_consuming(self):
        """Start consuming messages"""
        if not self.is_initialized:
            raise RuntimeError("Consumer not initialized")

        self.is_consuming = True
        logger.info("Starting Kafka message consumption")

        try:
            async for message in self.consumer:
                if not self.is_consuming:
                    break

                await self._process_message(message)

        except Exception as e:
            logger.error("Error in message consumption loop", error=str(e))
        finally:
            self.is_consuming = False
            logger.info("Kafka message consumption stopped")

    async def _process_message(self, message):
        """Process a single Kafka message"""
        try:
            # Update statistics
            self.messages_consumed += 1
            self.bytes_consumed += len(message.value) if message.value else 0

            # Extract message data
            topic = message.topic
            key = message.key
            value = message.value
            partition = message.partition
            offset = message.offset
            timestamp = message.timestamp

            logger.debug(
                "Processing message",
                topic=topic,
                key=key,
                partition=partition,
                offset=offset,
                timestamp=timestamp,
            )

            # Find matching handler
            handler = self._find_handler(topic)

            if handler:
                try:
                    # Create message context
                    message_context = {
                        "topic": topic,
                        "key": key,
                        "value": value,
                        "partition": partition,
                        "offset": offset,
                        "timestamp": timestamp,
                        "received_at": datetime.utcnow(),
                    }

                    # Call handler
                    await handler(message_context)
                    self.messages_processed += 1

                except Exception as e:
                    logger.error(
                        "Error in message handler", topic=topic, key=key, error=str(e)
                    )
                    self.messages_failed += 1
            else:
                logger.debug(f"No handler found for topic: {topic}")
                self.messages_failed += 1

        except Exception as e:
            logger.error("Error processing message", error=str(e))
            self.messages_failed += 1

    def _find_handler(self, topic: str) -> Optional[Callable]:
        """Find handler for topic"""
        # Exact match first
        if topic in self.message_handlers:
            return self.message_handlers[topic]

        # Pattern matching
        for pattern, handler in self.message_handlers.items():
            if self._topic_matches_pattern(topic, pattern):
                return handler

        return None

    def _topic_matches_pattern(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern"""
        # Simple pattern matching (can be enhanced with regex)
        if pattern == "*":
            return True

        if "*" in pattern:
            # Replace * with regex pattern
            import re

            regex_pattern = pattern.replace("*", ".*")
            return bool(re.match(regex_pattern, topic))

        return topic == pattern

    async def consume_single_message(
        self, timeout_ms: int = 5000
    ) -> Optional[Dict[str, Any]]:
        """Consume a single message with timeout"""
        if not self.is_initialized:
            raise RuntimeError("Consumer not initialized")

        try:
            # Get message with timeout
            message = await asyncio.wait_for(
                self.consumer.getone(), timeout=timeout_ms / 1000
            )

            if message:
                self.messages_consumed += 1
                self.bytes_consumed += len(message.value) if message.value else 0

                return {
                    "topic": message.topic,
                    "key": message.key,
                    "value": message.value,
                    "partition": message.partition,
                    "offset": message.offset,
                    "timestamp": message.timestamp,
                    "received_at": datetime.utcnow(),
                }

        except asyncio.TimeoutError:
            logger.debug("No message received within timeout")
        except Exception as e:
            logger.error("Error consuming single message", error=str(e))

        return None

    async def consume_batch(
        self, batch_size: int = 100, timeout_ms: int = 5000
    ) -> List[Dict[str, Any]]:
        """Consume a batch of messages"""
        if not self.is_initialized:
            raise RuntimeError("Consumer not initialized")

        messages = []
        start_time = datetime.utcnow()

        try:
            while len(messages) < batch_size:
                # Check timeout
                elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
                if elapsed >= timeout_ms:
                    break

                # Get message
                remaining_timeout = max(100, timeout_ms - elapsed)
                message_data = await self.consume_single_message(remaining_timeout)

                if message_data:
                    messages.append(message_data)
                else:
                    break  # No more messages available

            logger.debug(f"Consumed batch of {len(messages)} messages")

        except Exception as e:
            logger.error("Error consuming message batch", error=str(e))

        return messages

    async def seek_to_beginning(self, topic: str, partition: int):
        """Seek to beginning of topic partition"""
        if not self.is_initialized:
            raise RuntimeError("Consumer not initialized")

        try:
            from kafka import TopicPartition

            tp = TopicPartition(topic, partition)
            self.consumer.seek_to_beginning(tp)
            logger.info(f"Seeked to beginning of {topic}:{partition}")

        except Exception as e:
            logger.error(
                "Error seeking to beginning",
                topic=topic,
                partition=partition,
                error=str(e),
            )

    async def seek_to_end(self, topic: str, partition: int):
        """Seek to end of topic partition"""
        if not self.is_initialized:
            raise RuntimeError("Consumer not initialized")

        try:

            tp = TopicPartition(topic, partition)
            self.consumer.seek_to_end(tp)
            logger.info(f"Seeked to end of {topic}:{partition}")

        except Exception as e:
            logger.error(
                "Error seeking to end", topic=topic, partition=partition, error=str(e)
            )

    async def commit_offsets(self):
        """Manually commit current offsets"""
        if not self.is_initialized:
            raise RuntimeError("Consumer not initialized")

        try:
            await self.consumer.commit()
            logger.debug("Committed consumer offsets")

        except Exception as e:
            logger.error("Error committing offsets", error=str(e))

    async def get_statistics(self) -> Dict[str, Any]:
        """Get consumer statistics"""
        return {
            "is_initialized": self.is_initialized,
            "is_consuming": self.is_consuming,
            "messages_consumed": self.messages_consumed,
            "messages_processed": self.messages_processed,
            "messages_failed": self.messages_failed,
            "bytes_consumed": self.bytes_consumed,
            "processing_rate": (
                self.messages_processed / self.messages_consumed * 100
                if self.messages_consumed > 0
                else 0
            ),
            "topics": self.topics,
            "handlers_registered": len(self.message_handlers),
        }

    async def stop_consuming(self):
        """Stop consuming messages"""
        self.is_consuming = False
        logger.info("Stopping Kafka message consumption")

    async def cleanup(self):
        """Cleanup Kafka consumer"""
        self.is_consuming = False

        if self.consumer:
            try:
                await self.consumer.stop()
                logger.info("Kafka consumer stopped")
            except Exception as e:
                logger.error("Error stopping Kafka consumer", error=str(e))

        self.is_initialized = False
