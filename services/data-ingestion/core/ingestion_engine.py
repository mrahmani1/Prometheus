"""Main data ingestion engine for multi-source geological data"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import structlog
import torch.nn as nn
from connectors.atmospheric_connector import AtmosphericConnector
from connectors.esa_connector import ESAConnector
from connectors.fiber_optic_connector import FiberOpticConnector
from connectors.hydroacoustic_connector import HydroacousticConnector
from connectors.nasa_connector import NASAConnector
from connectors.satellite_connector import CommercialSatelliteConnector
from connectors.seismometer_connector import SeismometerConnector
from connectors.usgs_connector import USGSConnector
from processors.data_processor import DataProcessor
from storage.data_storage import DataStorage
from streaming.stream_processor import StreamProcessor

from .service_registry import ServiceRegistry

logger = structlog.get_logger()


class DataIngestionEngine:
    """Main engine for coordinating multi-source data ingestion"""

    def __init__(self, config: Any, service_registry: ServiceRegistry):
        self.config = config
        self.service_registry = service_registry
        self.connectors = {}
        self.data_processor = None
        self.data_storage = None
        self.stream_processor = None
        self.ingestion_tasks = {}
        self.is_running = False

    async def initialize(self):
        """Initialize the data ingestion engine"""
        logger.info("Initializing Data Ingestion Engine")

        try:
            # Initialize data storage
            self.data_storage = DataStorage(self.config)
            await self.data_storage.initialize()

            # Initialize data processor
            self.data_processor = DataProcessor(self.config, self.data_storage)
            await self.data_processor.initialize()

            # Initialize stream processor
            self.stream_processor = StreamProcessor(self.config, self.data_processor)
            await self.stream_processor.initialize()

            # Initialize connectors
            await self._initialize_connectors()

            # Start ingestion tasks
            await self._start_ingestion_tasks()

            # Start stream processing
            asyncio.create_task(self.stream_processor.start_processing())

            self.is_running = True
            logger.info("Data Ingestion Engine initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize Data Ingestion Engine", error=str(e))
            raise

    async def _initialize_connectors(self):
        """Initialize all data source connectors"""
        connector_configs = [
            ("usgs", USGSConnector),
            ("nasa", NASAConnector),
            ("esa", ESAConnector),
            ("commercial_satellite", CommercialSatelliteConnector),
            ("fiber_optic", FiberOpticConnector),
            ("seismometer", SeismometerConnector),
            ("atmospheric", AtmosphericConnector),
            ("hydroacoustic", HydroacousticConnector),
        ]

        for connector_name, connector_class in connector_configs:
            try:
                connector = connector_class(self.config, self.data_processor)
                await connector.initialize()
                self.connectors[connector_name] = connector
                logger.info(f"Initialized {connector_name} connector")
            except Exception as e:
                logger.error(
                    f"Failed to initialize {connector_name} connector", error=str(e)
                )
                # Continue with other connectors even if one fails

    async def _start_ingestion_tasks(self):
        """Start background ingestion tasks for each connector"""
        for connector_name, connector in self.connectors.items():
            if hasattr(connector, "start_continuous_ingestion"):
                task = asyncio.create_task(
                    connector.start_continuous_ingestion(),
                    name=f"ingestion_{connector_name}",
                )
                self.ingestion_tasks[connector_name] = task
                logger.info(f"Started ingestion task for {connector_name}")

    async def is_healthy(self) -> bool:
        """Check if the ingestion engine is healthy"""
        try:
            if not self.is_running:
                return False

            # Check data storage health
            if not await self.data_storage.is_healthy():
                return False

            # Check if at least 50% of connectors are healthy
            healthy_connectors = 0
            total_connectors = len(self.connectors)

            for connector in self.connectors.values():
                if await connector.is_healthy():
                    healthy_connectors += 1

            health_ratio = (
                healthy_connectors / total_connectors if total_connectors > 0 else 0
            )
            return health_ratio >= 0.5

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False

    async def is_ready(self) -> bool:
        """Check if the ingestion engine is ready to serve requests"""
        try:
            if not self.is_running:
                return False

            # Check if data processor is ready
            if not await self.data_processor.is_ready():
                return False

            # Check if at least one connector is ready
            for connector in self.connectors.values():
                if await connector.is_ready():
                    return True

            return False

        except Exception as e:
            logger.error("Readiness check failed", error=str(e))
            return False

    async def get_active_connectors_count(self) -> int:
        """Get the number of active connectors"""
        active_count = 0
        for connector in self.connectors.values():
            if await connector.is_healthy():
                active_count += 1
        return active_count

    async def get_connector_status(self) -> Dict[str, Any]:
        """Get status of all connectors"""
        status = {}
        for name, connector in self.connectors.items():
            try:
                status[name] = {
                    "healthy": await connector.is_healthy(),
                    "ready": await connector.is_ready(),
                    "last_data_time": getattr(connector, "last_data_time", None),
                    "total_records_processed": getattr(
                        connector, "total_records_processed", 0
                    ),
                    "error_count": getattr(connector, "error_count", 0),
                }
            except Exception as e:
                status[name] = {"healthy": False, "ready": False, "error": str(e)}
        return status

    async def trigger_manual_ingestion(
        self,
        connector_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Trigger manual data ingestion for a specific connector"""
        if connector_name not in self.connectors:
            raise ValueError(f"Unknown connector: {connector_name}")

        connector = self.connectors[connector_name]

        try:
            # Set default time range if not provided
            if not end_time:
                end_time = datetime.utcnow()
            if not start_time:
                start_time = end_time - timedelta(hours=1)

            logger.info(
                f"Starting manual ingestion for {connector_name}",
                start_time=start_time,
                end_time=end_time,
                parameters=parameters,
            )

            result = await connector.ingest_data_range(
                start_time=start_time, end_time=end_time, parameters=parameters or {}
            )

            logger.info(
                f"Manual ingestion completed for {connector_name}",
                records_processed=result.get("records_processed", 0),
            )

            return result

        except Exception as e:
            logger.error(f"Manual ingestion failed for {connector_name}", error=str(e))
            raise

    async def get_ingestion_statistics(self) -> Dict[str, Any]:
        """Get ingestion statistics for all connectors"""
        stats = {
            "total_connectors": len(self.connectors),
            "active_connectors": await self.get_active_connectors_count(),
            "ingestion_tasks_running": len(
                [task for task in self.ingestion_tasks.values() if not task.done()]
            ),
            "connectors": {},
        }

        for name, connector in self.connectors.items():
            try:
                connector_stats = await connector.get_statistics()
                stats["connectors"][name] = connector_stats
            except Exception as e:
                stats["connectors"][name] = {"error": str(e)}

        return stats

    async def pause_connector(self, connector_name: str):
        """Pause a specific connector"""
        if connector_name not in self.connectors:
            raise ValueError(f"Unknown connector: {connector_name}")

        connector = self.connectors[connector_name]
        await connector.pause()

        # Cancel the ingestion task if it exists
        if connector_name in self.ingestion_tasks:
            task = self.ingestion_tasks[connector_name]
            if not task.done():
                task.cancel()

        logger.info(f"Paused connector: {connector_name}")

    async def resume_connector(self, connector_name: str):
        """Resume a specific connector"""
        if connector_name not in self.connectors:
            raise ValueError(f"Unknown connector: {connector_name}")

        connector = self.connectors[connector_name]
        await connector.resume()

        # Restart the ingestion task
        if hasattr(connector, "start_continuous_ingestion"):
            task = asyncio.create_task(
                connector.start_continuous_ingestion(),
                name=f"ingestion_{connector_name}",
            )
            self.ingestion_tasks[connector_name] = task

        logger.info(f"Resumed connector: {connector_name}")

    async def cleanup(self):
        """Cleanup the data ingestion engine"""
        logger.info("Cleaning up Data Ingestion Engine")

        self.is_running = False

        # Cancel all ingestion tasks
        for task_name, task in self.ingestion_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.info(f"Cancelled ingestion task: {task_name}")

        # Cleanup connectors
        for connector_name, connector in self.connectors.items():
            try:
                await connector.cleanup()
                logger.info(f"Cleaned up connector: {connector_name}")
            except Exception as e:
                logger.error(
                    f"Error cleaning up connector {connector_name}", error=str(e)
                )

        # Cleanup stream processor
        if self.stream_processor:
            await self.stream_processor.cleanup()

        # Cleanup data processor
        if self.data_processor:
            await self.data_processor.cleanup()

        # Cleanup data storage
        if self.data_storage:
            await self.data_storage.cleanup()

        logger.info("Data Ingestion Engine cleanup completed")
