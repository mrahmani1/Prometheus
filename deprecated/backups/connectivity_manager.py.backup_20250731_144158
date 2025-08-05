"""
Connectivity restoration and data synchronization procedures
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import hashlib

from .local_storage import LocalStorageManager, StorageType, StorageRecord
from models.device import SyncStatus, DeviceMetrics
from models.config import EdgeDeviceConfig

logger = logging.getLogger(__name__)


class ConnectivityStatus(str, Enum):
    """Connectivity status"""
    ONLINE = "online"
    OFFLINE = "offline"
    LIMITED = "limited"
    RECONNECTING = "reconnecting"


class SyncPriority(str, Enum):
    """Data synchronization priority"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class ConnectivityTest:
    """Connectivity test configuration"""
    name: str
    url: str
    method: str = "GET"
    timeout: int = 10
    expected_status: int = 200
    weight: float = 1.0


@dataclass
class SyncTask:
    """Data synchronization task"""
    task_id: str
    record_id: str
    storage_type: StorageType
    priority: SyncPriority
    created_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    data_size: int = 0


class ConnectivityManager:
    """Manages connectivity monitoring and data synchronization"""
    
    def __init__(self, config: EdgeDeviceConfig, storage_manager: LocalStorageManager):
        self.config = config
        self.storage_manager = storage_manager
        
        # Connectivity state
        self.connectivity_status = ConnectivityStatus.OFFLINE
        self.last_online_time: Optional[datetime] = None
        self.connectivity_score = 0.0
        
        # Connectivity tests
        self.connectivity_tests = [
            ConnectivityTest(
                name="cloud_api",
                url=f"http://localhost:8080/health",  # API Gateway health
                weight=2.0
            ),
            ConnectivityTest(
                name="data_ingestion",
                url=f"http://localhost:8082/health",  # Data ingestion service
                weight=1.5
            ),
            ConnectivityTest(
                name="internet",
                url="https://8.8.8.8",  # Google DNS
                method="HEAD",
                timeout=5,
                weight=1.0
            )
        ]
        
        # Synchronization state
        self.sync_queue: List[SyncTask] = []
        self.sync_in_progress: Dict[str, SyncTask] = {}
        self.sync_statistics = {
            "total_synced": 0,
            "sync_errors": 0,
            "bytes_synced": 0,
            "last_sync_time": None,
            "sync_rate_mbps": 0.0
        }
        
        # Configuration
        self.connectivity_check_interval = 30  # seconds
        self.sync_batch_size = 10
        self.max_concurrent_syncs = 3
        self.bandwidth_limit_mbps = 10.0
        
        # Event callbacks
        self.connectivity_callbacks: List[Callable] = []
        self.sync_callbacks: List[Callable] = []
        
        # Background tasks
        self._connectivity_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start connectivity monitoring and synchronization"""
        try:
            self._running = True
            
            # Start background tasks
            self._connectivity_task = asyncio.create_task(self._connectivity_monitor())
            self._sync_task = asyncio.create_task(self._sync_manager())
            
            logger.info("Connectivity manager started")
            
        except Exception as e:
            logger.error(f"Error starting connectivity manager: {e}")
            raise
    
    async def stop(self):
        """Stop connectivity monitoring and synchronization"""
        try:
            self._running = False
            
            # Cancel background tasks
            if self._connectivity_task:
                self._connectivity_task.cancel()
                try:
                    await self._connectivity_task
                except asyncio.CancelledError:
                    pass
            
            if self._sync_task:
                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Connectivity manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping connectivity manager: {e}")
    
    async def _connectivity_monitor(self):
        """Monitor connectivity status"""
        while self._running:
            try:
                # Test connectivity
                connectivity_score = await self._test_connectivity()
                
                # Update connectivity status
                previous_status = self.connectivity_status
                self.connectivity_score = connectivity_score
                
                if connectivity_score >= 0.8:
                    self.connectivity_status = ConnectivityStatus.ONLINE
                elif connectivity_score >= 0.5:
                    self.connectivity_status = ConnectivityStatus.LIMITED
                elif previous_status == ConnectivityStatus.ONLINE and connectivity_score > 0:
                    self.connectivity_status = ConnectivityStatus.RECONNECTING
                else:
                    self.connectivity_status = ConnectivityStatus.OFFLINE
                
                # Track online time
                if self.connectivity_status == ConnectivityStatus.ONLINE:
                    if previous_status != ConnectivityStatus.ONLINE:
                        logger.info("Connectivity restored")
                        await self._on_connectivity_restored()
                    self.last_online_time = datetime.utcnow()
                elif previous_status == ConnectivityStatus.ONLINE:
                    logger.warning("Connectivity lost")
                    await self._on_connectivity_lost()
                
                # Notify callbacks if status changed
                if previous_status != self.connectivity_status:
                    await self._notify_connectivity_callbacks()
                
                await asyncio.sleep(self.connectivity_check_interval)
                
            except Exception as e:
                logger.error(f"Error in connectivity monitor: {e}")
                await asyncio.sleep(self.connectivity_check_interval)
    
    async def _test_connectivity(self) -> float:
        """Test connectivity and return score (0.0 to 1.0)"""
        try:
            total_weight = sum(test.weight for test in self.connectivity_tests)
            weighted_score = 0.0
            
            async with aiohttp.ClientSession() as session:
                for test in self.connectivity_tests:
                    try:
                        start_time = time.time()
                        
                        async with session.request(
                            test.method,
                            test.url,
                            timeout=aiohttp.ClientTimeout(total=test.timeout)
                        ) as response:
                            response_time = time.time() - start_time
                            
                            # Check if response is successful
                            if response.status == test.expected_status:
                                # Score based on response time (faster = better)
                                time_score = max(0.0, 1.0 - (response_time / test.timeout))
                                weighted_score += test.weight * time_score
                            
                    except Exception as e:
                        logger.debug(f"Connectivity test {test.name} failed: {e}")
                        # Test failed, contributes 0 to score
                        continue
            
            return min(1.0, weighted_score / total_weight) if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error testing connectivity: {e}")
            return 0.0
    
    async def _on_connectivity_restored(self):
        """Handle connectivity restoration"""
        try:
            logger.info("Initiating data synchronization after connectivity restoration")
            
            # Queue pending data for synchronization
            await self._queue_pending_sync_tasks()
            
            # Update device status if possible
            await self._update_device_status()
            
        except Exception as e:
            logger.error(f"Error handling connectivity restoration: {e}")
    
    async def _on_connectivity_lost(self):
        """Handle connectivity loss"""
        try:
            logger.warning("Connectivity lost - switching to offline mode")
            
            # Cancel ongoing sync tasks
            for task_id in list(self.sync_in_progress.keys()):
                sync_task = self.sync_in_progress.pop(task_id)
                logger.debug(f"Cancelled sync task: {task_id}")
            
        except Exception as e:
            logger.error(f"Error handling connectivity loss: {e}")
    
    async def _sync_manager(self):
        """Manage data synchronization tasks"""
        while self._running:
            try:
                if self.connectivity_status in [ConnectivityStatus.ONLINE, ConnectivityStatus.LIMITED]:
                    # Process sync queue
                    await self._process_sync_queue()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in sync manager: {e}")
                await asyncio.sleep(5)
    
    async def _queue_pending_sync_tasks(self):
        """Queue pending data for synchronization"""
        try:
            # Get unsynchronized records
            pending_records = await self.storage_manager.list_records(
                sync_status="pending",
                limit=1000
            )
            
            for record in pending_records:
                # Determine priority based on storage type and age
                priority = self._determine_sync_priority(record)
                
                sync_task = SyncTask(
                    task_id=f"sync_{record.record_id}_{datetime.utcnow().timestamp()}",
                    record_id=record.record_id,
                    storage_type=record.storage_type,
                    priority=priority,
                    created_at=datetime.utcnow(),
                    data_size=record.compressed_size
                )
                
                self.sync_queue.append(sync_task)
            
            # Sort queue by priority
            self.sync_queue.sort(key=lambda x: (x.priority.value, x.created_at))
            
            logger.info(f"Queued {len(pending_records)} records for synchronization")
            
        except Exception as e:
            logger.error(f"Error queuing sync tasks: {e}")
    
    def _determine_sync_priority(self, record: StorageRecord) -> SyncPriority:
        """Determine synchronization priority for a record"""
        # Critical data types
        if record.storage_type in [StorageType.ALERT_HISTORY, StorageType.SYSTEM_LOGS]:
            return SyncPriority.CRITICAL
        
        # High priority for recent analysis results
        if record.storage_type == StorageType.ANALYSIS_RESULTS:
            age_hours = (datetime.utcnow() - record.timestamp).total_seconds() / 3600
            if age_hours < 1:
                return SyncPriority.HIGH
            elif age_hours < 24:
                return SyncPriority.NORMAL
            else:
                return SyncPriority.LOW
        
        # Normal priority for sensor data
        if record.storage_type == StorageType.SENSOR_DATA:
            age_hours = (datetime.utcnow() - record.timestamp).total_seconds() / 3600
            if age_hours < 6:
                return SyncPriority.NORMAL
            else:
                return SyncPriority.LOW
        
        # Default to normal priority
        return SyncPriority.NORMAL
    
    async def _process_sync_queue(self):
        """Process synchronization queue"""
        try:
            # Limit concurrent syncs
            if len(self.sync_in_progress) >= self.max_concurrent_syncs:
                return
            
            # Get next batch of tasks
            batch_size = min(self.sync_batch_size, self.max_concurrent_syncs - len(self.sync_in_progress))
            tasks_to_process = []
            
            for _ in range(batch_size):
                if not self.sync_queue:
                    break
                
                task = self.sync_queue.pop(0)
                tasks_to_process.append(task)
                self.sync_in_progress[task.task_id] = task
            
            # Process tasks concurrently
            if tasks_to_process:
                await asyncio.gather(
                    *[self._sync_single_record(task) for task in tasks_to_process],
                    return_exceptions=True
                )
            
        except Exception as e:
            logger.error(f"Error processing sync queue: {e}")
    
    async def _sync_single_record(self, sync_task: SyncTask):
        """Synchronize a single record"""
        try:
            start_time = time.time()
            
            # Retrieve data from local storage
            data_result = await self.storage_manager.retrieve_data(sync_task.record_id)
            if not data_result:
                logger.error(f"Failed to retrieve data for sync: {sync_task.record_id}")
                return
            
            data_bytes, metadata = data_result
            
            # Prepare sync payload
            sync_payload = {
                "record_id": sync_task.record_id,
                "storage_type": sync_task.storage_type.value,
                "device_id": self.config.device_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data_size": len(data_bytes),
                "metadata": metadata,
                "data": data_bytes.hex(),  # Hex encode for JSON transport
                "checksum": hashlib.sha256(data_bytes).hexdigest()
            }
            
            # Send to cloud service
            success = await self._send_sync_data(sync_payload)
            
            if success:
                # Update sync status
                await self.storage_manager.update_sync_status(sync_task.record_id, "synced")
                
                # Update statistics
                sync_time = time.time() - start_time
                self.sync_statistics["total_synced"] += 1
                self.sync_statistics["bytes_synced"] += len(data_bytes)
                self.sync_statistics["last_sync_time"] = datetime.utcnow()
                
                if sync_time > 0:
                    mbps = (len(data_bytes) * 8) / (sync_time * 1_000_000)
                    self.sync_statistics["sync_rate_mbps"] = mbps
                
                logger.debug(f"Successfully synced record: {sync_task.record_id}")
                
            else:
                # Handle sync failure
                sync_task.retry_count += 1
                sync_task.last_error = "Sync failed"
                
                if sync_task.retry_count < sync_task.max_retries:
                    # Re-queue for retry
                    self.sync_queue.append(sync_task)
                    logger.warning(f"Sync failed, retrying: {sync_task.record_id} (attempt {sync_task.retry_count})")
                else:
                    # Mark as failed
                    await self.storage_manager.update_sync_status(sync_task.record_id, "failed")
                    self.sync_statistics["sync_errors"] += 1
                    logger.error(f"Sync failed permanently: {sync_task.record_id}")
            
        except Exception as e:
            logger.error(f"Error syncing record {sync_task.record_id}: {e}")
            sync_task.retry_count += 1
            sync_task.last_error = str(e)
            
            if sync_task.retry_count < sync_task.max_retries:
                self.sync_queue.append(sync_task)
        
        finally:
            # Remove from in-progress
            self.sync_in_progress.pop(sync_task.task_id, None)
    
    async def _send_sync_data(self, payload: Dict[str, Any]) -> bool:
        """Send synchronization data to cloud service"""
        try:
            sync_endpoint = f"http://localhost:8082/api/v1/sync/upload"  # Data ingestion service
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    sync_endpoint,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.config.auth_token}",
                        "Content-Type": "application/json"
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status == 200:
                        return True
                    else:
                        logger.error(f"Sync upload failed with status {response.status}: {await response.text()}")
                        return False
            
        except Exception as e:
            logger.error(f"Error sending sync data: {e}")
            return False
    
    async def _update_device_status(self):
        """Update device status with cloud service"""
        try:
            # Prepare device status update
            status_update = {
                "device_id": self.config.device_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "online",
                "connectivity_score": self.connectivity_score,
                "sync_statistics": self.sync_statistics,
                "storage_statistics": await self.storage_manager.get_storage_statistics()
            }
            
            # Send status update
            status_endpoint = f"http://localhost:8080/api/v1/devices/{self.config.device_id}/status"
            
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    status_endpoint,
                    json=status_update,
                    headers={
                        "Authorization": f"Bearer {self.config.auth_token}",
                        "Content-Type": "application/json"
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        logger.debug("Device status updated successfully")
                    else:
                        logger.warning(f"Failed to update device status: {response.status}")
            
        except Exception as e:
            logger.debug(f"Error updating device status: {e}")  # Debug level since this is not critical
    
    async def _notify_connectivity_callbacks(self):
        """Notify registered connectivity callbacks"""
        try:
            for callback in self.connectivity_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self.connectivity_status, self.connectivity_score)
                    else:
                        callback(self.connectivity_status, self.connectivity_score)
                except Exception as e:
                    logger.error(f"Error in connectivity callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error notifying connectivity callbacks: {e}")
    
    def register_connectivity_callback(self, callback: Callable):
        """Register callback for connectivity status changes"""
        self.connectivity_callbacks.append(callback)
    
    def register_sync_callback(self, callback: Callable):
        """Register callback for sync events"""
        self.sync_callbacks.append(callback)
    
    async def force_sync(self, storage_type: Optional[StorageType] = None) -> Dict[str, Any]:
        """Force immediate synchronization of pending data"""
        try:
            if self.connectivity_status == ConnectivityStatus.OFFLINE:
                return {"error": "No connectivity available for sync"}
            
            # Get pending records
            pending_records = await self.storage_manager.list_records(
                storage_type=storage_type,
                sync_status="pending",
                limit=100
            )
            
            # Queue for immediate sync
            sync_tasks = []
            for record in pending_records:
                sync_task = SyncTask(
                    task_id=f"force_sync_{record.record_id}_{datetime.utcnow().timestamp()}",
                    record_id=record.record_id,
                    storage_type=record.storage_type,
                    priority=SyncPriority.HIGH,
                    created_at=datetime.utcnow(),
                    data_size=record.compressed_size
                )
                sync_tasks.append(sync_task)
            
            # Process sync tasks
            results = await asyncio.gather(
                *[self._sync_single_record(task) for task in sync_tasks],
                return_exceptions=True
            )
            
            # Count successes and failures
            successes = sum(1 for r in results if not isinstance(r, Exception))
            failures = len(results) - successes
            
            return {
                "total_records": len(pending_records),
                "successful_syncs": successes,
                "failed_syncs": failures,
                "sync_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in force sync: {e}")
            return {"error": str(e)}
    
    async def get_connectivity_status(self) -> Dict[str, Any]:
        """Get current connectivity status and statistics"""
        return {
            "status": self.connectivity_status.value,
            "connectivity_score": self.connectivity_score,
            "last_online_time": self.last_online_time.isoformat() if self.last_online_time else None,
            "sync_queue_size": len(self.sync_queue),
            "sync_in_progress": len(self.sync_in_progress),
            "sync_statistics": self.sync_statistics.copy(),
            "connectivity_tests": [
                {
                    "name": test.name,
                    "url": test.url,
                    "weight": test.weight
                }
                for test in self.connectivity_tests
            ]
        }
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status and statistics"""
        # Get sync statistics from storage
        storage_stats = await self.storage_manager.get_storage_statistics()
        sync_stats = storage_stats.get("sync_statistics", {})
        
        return {
            "sync_statistics": self.sync_statistics.copy(),
            "queue_size": len(self.sync_queue),
            "in_progress": len(self.sync_in_progress),
            "storage_sync_stats": sync_stats,
            "bandwidth_limit_mbps": self.bandwidth_limit_mbps,
            "max_concurrent_syncs": self.max_concurrent_syncs
        }
    
    async def update_sync_configuration(self, config: Dict[str, Any]):
        """Update synchronization configuration"""
        try:
            if "bandwidth_limit_mbps" in config:
                self.bandwidth_limit_mbps = float(config["bandwidth_limit_mbps"])
            
            if "max_concurrent_syncs" in config:
                self.max_concurrent_syncs = int(config["max_concurrent_syncs"])
            
            if "sync_batch_size" in config:
                self.sync_batch_size = int(config["sync_batch_size"])
            
            if "connectivity_check_interval" in config:
                self.connectivity_check_interval = int(config["connectivity_check_interval"])
            
            logger.info("Sync configuration updated")
            
        except Exception as e:
            logger.error(f"Error updating sync configuration: {e}")
            raise
    
    async def clear_sync_queue(self):
        """Clear the synchronization queue"""
        try:
            cleared_count = len(self.sync_queue)
            self.sync_queue.clear()
            logger.info(f"Cleared {cleared_count} items from sync queue")
            
        except Exception as e:
            logger.error(f"Error clearing sync queue: {e}")
    
    async def retry_failed_syncs(self) -> Dict[str, Any]:
        """Retry failed synchronization tasks"""
        try:
            # Get failed records
            failed_records = await self.storage_manager.list_records(
                sync_status="failed",
                limit=100
            )
            
            # Reset to pending and queue for retry
            retry_count = 0
            for record in failed_records:
                await self.storage_manager.update_sync_status(record.record_id, "pending")
                
                sync_task = SyncTask(
                    task_id=f"retry_{record.record_id}_{datetime.utcnow().timestamp()}",
                    record_id=record.record_id,
                    storage_type=record.storage_type,
                    priority=SyncPriority.NORMAL,
                    created_at=datetime.utcnow(),
                    data_size=record.compressed_size
                )
                
                self.sync_queue.append(sync_task)
                retry_count += 1
            
            # Sort queue by priority
            self.sync_queue.sort(key=lambda x: (x.priority.value, x.created_at))
            
            return {
                "retried_records": retry_count,
                "queue_size": len(self.sync_queue)
            }
            
        except Exception as e:
            logger.error(f"Error retrying failed syncs: {e}")
            return {"error": str(e)}