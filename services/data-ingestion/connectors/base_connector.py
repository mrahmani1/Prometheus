"""Base connector class for all data source connectors"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger()


class BaseConnector(ABC):
    """Abstract base class for all data source connectors"""

    def __init__(self, config: Any, data_processor: Any):
        self.config = config
        self.data_processor = data_processor
        self.name = "base"
        self.is_initialized = False
        self.is_paused = False
        self.last_data_time = None
        self.total_records_processed = 0
        self.error_count = 0
        self.health_status = True

    async def initialize(self):
        """Initialize the connector"""
        self.is_initialized = True
        logger.info(f"Initialized {self.name} connector")

    async def is_healthy(self) -> bool:
        """Check if the connector is healthy"""
        return self.is_initialized and self.health_status

    async def is_ready(self) -> bool:
        """Check if the connector is ready to process data"""
        return await self.is_healthy() and not self.is_paused

    async def pause(self):
        """Pause the connector"""
        self.is_paused = True
        logger.info(f"Paused {self.name} connector")

    async def resume(self):
        """Resume the connector"""
        self.is_paused = False
        logger.info(f"Resumed {self.name} connector")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics"""
        return {
            "name": self.name,
            "is_healthy": await self.is_healthy(),
            "is_ready": await self.is_ready(),
            "is_paused": self.is_paused,
            "last_data_time": (
                self.last_data_time.isoformat() if self.last_data_time else None
            ),
            "total_records_processed": self.total_records_processed,
            "error_count": self.error_count,
        }

    @abstractmethod
    async def ingest_data_range(
        self,
        start_time: datetime,
        end_time: datetime,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest data for a specific time range"""

    @abstractmethod
    async def start_continuous_ingestion(self):
        """Start continuous data ingestion"""

    async def cleanup(self):
        """Cleanup the connector"""
        self.is_initialized = False
        logger.info(f"Cleaned up {self.name} connector")
