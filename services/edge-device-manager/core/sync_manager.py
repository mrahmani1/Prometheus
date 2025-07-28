"""
Data synchronization manager for edge devices
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
import uuid
import hashlib

import aioredis
import httpx
from models.config import EdgeManagerConfig, SyncConfiguration
from models.device import (
    EdgeDevice, SyncStatus, DataSyncRequest, DataSyncResponse
)

logger = logging.getLogger(__name__)


class DataSyncManager:
    """Manages data synchronization between edge devices and cloud"""
    
    def __init__(self, config: EdgeManagerConfig):
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
        self.sync_tasks: Dict[str, Dict] = {}
        self.sync_queue: asyncio.Queue = asyncio.Queue()
        self._ready = False
        self._sync_scheduler_running = False
        
        # Sync settings
        self.sync_interval = config.sync_interval
        self.max_sync_retries = config.max_sync_retries
        self.sync_batch_size = config.sync_batch_size
        
        # Active sync tracking
        self.active_syncs: Set[str] = set()
        
    async def initialize(self):
        """Initialize the sync manager"""
        try:
            # Connect to Redis
            self.redis = aioredis.from_url(self.config.redis_url)
            await self.redis.ping()
            
            # Load existing sync tasks
            await self._load_sync_tasks_from_storage()
            
            # Start sync workers
            for i in range(3):  # 3 concurrent sync workers
                asyncio.create_task(self._sync_worker(f"sync-worker-{i}"))
            
            self._ready = True
            logger.info(f"Sync manager initialized with {len(self.sync_tasks)} sync tasks")
            
        except Exception as e:
            logger.error(f"Failed to initialize sync manager: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if the sync manager is ready"""
        return self._ready
    
    async def start_sync_scheduler(self):
        """Start the sync scheduler"""
        if self._sync_scheduler_running:
            return
        
        self._sync_scheduler_running = True
        asyncio.create_task(self._sync_scheduler())
        logger.info("Started sync scheduler")
    
    async def stop_sync_scheduler(self):
        """Stop the sync scheduler"""
        self._sync_scheduler_running = False
        logger.info("Stopped sync scheduler")
    
    async def request_sync(self, request: DataSyncRequest) -> DataSyncResponse:
        """Request data synchronization for specified devices"""
        try:
            sync_id = f"sync-{uuid.uuid4().hex[:12]}"
            
            # Create sync task
            sync_task = {
                "sync_id": sync_id,
                "device_ids": request.device_ids,
                "data_types": request.data_types,
                "start_time": request.start_time,
                "end_time": request.end_time,
                "force_full_sync": request.force_full_sync,
                "max_bandwidth_mbps": request.max_bandwidth_mbps,
                "priority": request.priority,
                "status": SyncStatus.IDLE,
                "created_at": datetime.utcnow(),
                "device_status": {device_id: {"status": "pending"} for device_id in request.device_ids}
            }
            
            # Store sync task
            self.sync_tasks[sync_id] = sync_task
            await self._save_sync_task_to_storage(sync_task)
            
            # Queue for processing
            await self.sync_queue.put(sync_task)
            
            logger.info(f"Queued sync request {sync_id} for {len(request.device_ids)} devices")
            
            response = DataSyncResponse(
                sync_id=sync_id,
                device_ids=request.device_ids,
                status=SyncStatus.IDLE,
                device_status=sync_task["device_status"]
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to request sync: {e}")
            raise
    
    async def get_sync_status(self, sync_id: str) -> Optional[Dict]:
        """Get status of a sync operation"""
        sync_task = self.sync_tasks.get(sync_id)
        if not sync_task:
            return None
        
        # Calculate progress
        total_devices = len(sync_task["device_ids"])
        completed_devices = len([
            status for status in sync_task["device_status"].values()
            if status.get("status") == "completed"
        ])
        failed_devices = len([
            status for status in sync_task["device_status"].values()
            if status.get("status") == "failed"
        ])
        
        return {
            "sync_id": sync_id,
            "status": sync_task["status"],
            "total_devices": total_devices,
            "completed_devices": completed_devices,
            "failed_devices": failed_devices,
            "progress_percentage": (completed_devices / total_devices) * 100 if total_devices > 0 else 0,
            "device_status": sync_task["device_status"],
            "created_at": sync_task["created_at"],
            "estimated_completion": sync_task.get("estimated_completion")
        }
    
    async def cancel_sync(self, sync_id: str):
        """Cancel a sync operation"""
        if sync_id not in self.sync_tasks:
            raise ValueError(f"Sync {sync_id} not found")
        
        sync_task = self.sync_tasks[sync_id]
        
        if sync_task["status"] in [SyncStatus.COMPLETED, SyncStatus.FAILED]:
            raise ValueError(f"Cannot cancel sync {sync_id} - already {sync_task['status']}")
        
        # Update status
        sync_task["status"] = SyncStatus.FAILED
        sync_task["cancelled_at"] = datetime.utcnow()
        
        # Update device statuses
        for device_id in sync_task["device_ids"]:
            if sync_task["device_status"][device_id]["status"] in ["pending", "syncing"]:
                sync_task["device_status"][device_id]["status"] = "cancelled"
        
        await self._save_sync_task_to_storage(sync_task)
        logger.info(f"Cancelled sync {sync_id}")
    
    async def schedule_device_sync(self, device_id: str, sync_config: SyncConfiguration):
        """Schedule regular sync for a device"""
        if not sync_config.sync_enabled:
            return
        
        # Create scheduled sync task
        sync_request = DataSyncRequest(
            device_ids=[device_id],
            data_types=self._get_enabled_data_types(sync_config)
        )
        
        await self.request_sync(sync_request)
        logger.info(f"Scheduled sync for device {device_id}")
    
    async def sync_device_data(self, device_id: str, data_types: List[str], 
                             sync_config: SyncConfiguration) -> Dict:
        """Synchronize data for a specific device"""
        try:
            sync_result = {
                "device_id": device_id,
                "data_types": data_types,
                "status": "success",
                "synced_bytes": 0,
                "sync_duration_seconds": 0,
                "errors": []
            }
            
            start_time = datetime.utcnow()
            
            # Sync each data type
            for data_type in data_types:
                try:
                    bytes_synced = await self._sync_data_type(device_id, data_type, sync_config)
                    sync_result["synced_bytes"] += bytes_synced
                    logger.debug(f"Synced {bytes_synced} bytes of {data_type} for device {device_id}")
                    
                except Exception as e:
                    error_msg = f"Failed to sync {data_type}: {e}"
                    sync_result["errors"].append(error_msg)
                    logger.error(f"Device {device_id} - {error_msg}")
            
            # Calculate duration
            sync_result["sync_duration_seconds"] = (datetime.utcnow() - start_time).total_seconds()
            
            # Update device sync status
            await self._update_device_sync_status(device_id, sync_result)
            
            if sync_result["errors"]:
                sync_result["status"] = "partial_success"
            
            return sync_result
            
        except Exception as e:
            logger.error(f"Failed to sync device {device_id}: {e}")
            return {
                "device_id": device_id,
                "status": "failed",
                "error": str(e),
                "sync_duration_seconds": 0,
                "synced_bytes": 0
            }
    
    async def get_sync_metrics(self) -> Dict:
        """Get sync metrics and statistics"""
        # Count sync tasks by status
        status_counts = {}
        total_bytes_synced = 0
        
        for sync_task in self.sync_tasks.values():
            status = sync_task["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Sum bytes synced (would be tracked in real implementation)
            total_bytes_synced += sync_task.get("total_bytes_synced", 0)
        
        # Get recent sync activity
        recent_syncs = []
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for sync_task in self.sync_tasks.values():
            if sync_task["created_at"] > cutoff_time:
                recent_syncs.append({
                    "sync_id": sync_task["sync_id"],
                    "device_count": len(sync_task["device_ids"]),
                    "status": sync_task["status"],
                    "created_at": sync_task["created_at"]
                })
        
        return {
            "total_sync_tasks": len(self.sync_tasks),
            "status_counts": status_counts,
            "active_syncs": len(self.active_syncs),
            "total_bytes_synced": total_bytes_synced,
            "recent_syncs": recent_syncs[-20:],  # Last 20 syncs
            "last_updated": datetime.utcnow()
        }
    
    async def _sync_scheduler(self):
        """Periodic sync scheduler"""
        while self._sync_scheduler_running:
            try:
                # Check for devices that need scheduled sync
                await self._check_scheduled_syncs()
                
                # Clean up old sync tasks
                await self._cleanup_old_sync_tasks()
                
                # Sleep for sync interval
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Error in sync scheduler: {e}")
                await asyncio.sleep(self.sync_interval)
    
    async def _sync_worker(self, worker_id: str):
        """Worker process for handling sync tasks"""
        logger.info(f"Started sync worker {worker_id}")
        
        while self._ready:
            try:
                # Get sync task from queue
                sync_task = await asyncio.wait_for(self.sync_queue.get(), timeout=1.0)
                
                logger.info(f"Worker {worker_id} processing sync {sync_task['sync_id']}")
                
                # Add to active syncs
                self.active_syncs.add(sync_task["sync_id"])
                
                try:
                    # Execute sync
                    await self._execute_sync_task(sync_task)
                    
                except Exception as e:
                    logger.error(f"Sync {sync_task['sync_id']} failed: {e}")
                    sync_task["status"] = SyncStatus.FAILED
                    sync_task["error"] = str(e)
                    await self._save_sync_task_to_storage(sync_task)
                
                finally:
                    # Remove from active syncs
                    self.active_syncs.discard(sync_task["sync_id"])
                    self.sync_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in sync worker {worker_id}: {e}")
                await asyncio.sleep(5)
    
    async def _execute_sync_task(self, sync_task: Dict):
        """Execute a sync task"""
        try:
            sync_task["status"] = SyncStatus.SYNCING
            sync_task["started_at"] = datetime.utcnow()
            await self._save_sync_task_to_storage(sync_task)
            
            # Process each device
            for device_id in sync_task["device_ids"]:
                try:
                    # Update device status
                    sync_task["device_status"][device_id]["status"] = "syncing"
                    sync_task["device_status"][device_id]["started_at"] = datetime.utcnow()
                    
                    # Create sync config (would be loaded from device config)
                    sync_config = SyncConfiguration()
                    
                    # Sync device data
                    sync_result = await self.sync_device_data(
                        device_id, sync_task["data_types"], sync_config
                    )
                    
                    # Update device status with result
                    sync_task["device_status"][device_id].update({
                        "status": "completed" if sync_result["status"] == "success" else "failed",
                        "completed_at": datetime.utcnow(),
                        "synced_bytes": sync_result["synced_bytes"],
                        "errors": sync_result.get("errors", [])
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to sync device {device_id}: {e}")
                    sync_task["device_status"][device_id].update({
                        "status": "failed",
                        "completed_at": datetime.utcnow(),
                        "error": str(e)
                    })
            
            # Determine overall sync status
            device_statuses = [status["status"] for status in sync_task["device_status"].values()]
            
            if all(status == "completed" for status in device_statuses):
                sync_task["status"] = SyncStatus.COMPLETED
            elif any(status == "completed" for status in device_statuses):
                sync_task["status"] = SyncStatus.COMPLETED  # Partial success still counts as completed
            else:
                sync_task["status"] = SyncStatus.FAILED
            
            sync_task["completed_at"] = datetime.utcnow()
            await self._save_sync_task_to_storage(sync_task)
            
            logger.info(f"Completed sync {sync_task['sync_id']} with status {sync_task['status']}")
            
        except Exception as e:
            logger.error(f"Failed to execute sync task {sync_task['sync_id']}: {e}")
            raise
    
    async def _sync_data_type(self, device_id: str, data_type: str, 
                            sync_config: SyncConfiguration) -> int:
        """Sync a specific data type for a device"""
        try:
            # This would implement actual data synchronization
            # For now, simulate sync process
            
            if data_type == "sensor_data":
                # Sync sensor readings
                bytes_synced = await self._sync_sensor_data(device_id, sync_config)
            elif data_type == "model_outputs":
                # Sync model prediction outputs
                bytes_synced = await self._sync_model_outputs(device_id, sync_config)
            elif data_type == "logs":
                # Sync application logs
                bytes_synced = await self._sync_logs(device_id, sync_config)
            elif data_type == "metrics":
                # Sync performance metrics
                bytes_synced = await self._sync_metrics(device_id, sync_config)
            else:
                logger.warning(f"Unknown data type: {data_type}")
                bytes_synced = 0
            
            return bytes_synced
            
        except Exception as e:
            logger.error(f"Failed to sync {data_type} for device {device_id}: {e}")
            raise
    
    async def _sync_sensor_data(self, device_id: str, sync_config: SyncConfiguration) -> int:
        """Sync sensor data for a device"""
        # Simulate sensor data sync
        await asyncio.sleep(0.5)  # Simulate network transfer
        return 1024 * 100  # 100KB simulated
    
    async def _sync_model_outputs(self, device_id: str, sync_config: SyncConfiguration) -> int:
        """Sync model outputs for a device"""
        # Simulate model output sync
        await asyncio.sleep(0.3)
        return 1024 * 50  # 50KB simulated
    
    async def _sync_logs(self, device_id: str, sync_config: SyncConfiguration) -> int:
        """Sync logs for a device"""
        # Simulate log sync
        await asyncio.sleep(0.2)
        return 1024 * 20  # 20KB simulated
    
    async def _sync_metrics(self, device_id: str, sync_config: SyncConfiguration) -> int:
        """Sync metrics for a device"""
        # Simulate metrics sync
        await asyncio.sleep(0.1)
        return 1024 * 10  # 10KB simulated
    
    def _get_enabled_data_types(self, sync_config: SyncConfiguration) -> List[str]:
        """Get list of enabled data types from sync configuration"""
        enabled_types = []
        
        if sync_config.sensor_data:
            enabled_types.append("sensor_data")
        if sync_config.model_outputs:
            enabled_types.append("model_outputs")
        if sync_config.logs:
            enabled_types.append("logs")
        if sync_config.metrics:
            enabled_types.append("metrics")
        
        return enabled_types
    
    async def _check_scheduled_syncs(self):
        """Check for devices that need scheduled sync"""
        # This would check device configurations and schedule syncs as needed
        # For now, just log that we're checking
        logger.debug("Checking for scheduled syncs")
    
    async def _cleanup_old_sync_tasks(self):
        """Clean up old sync tasks"""
        # Remove sync tasks older than 30 days
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        
        to_remove = []
        for sync_id, sync_task in self.sync_tasks.items():
            if sync_task["created_at"] < cutoff_time:
                to_remove.append(sync_id)
        
        for sync_id in to_remove:
            del self.sync_tasks[sync_id]
            await self._remove_sync_task_from_storage(sync_id)
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old sync tasks")
    
    async def _update_device_sync_status(self, device_id: str, sync_result: Dict):
        """Update device sync status in Redis"""
        if not self.redis:
            return
        
        try:
            sync_status_key = f"device_sync:{device_id}"
            sync_status = {
                "last_sync": datetime.utcnow().isoformat(),
                "status": sync_result["status"],
                "synced_bytes": sync_result["synced_bytes"],
                "sync_duration": sync_result["sync_duration_seconds"],
                "errors": sync_result.get("errors", [])
            }
            
            await self.redis.setex(sync_status_key, 86400, json.dumps(sync_status))
            
        except Exception as e:
            logger.error(f"Failed to update sync status for device {device_id}: {e}")
    
    async def _load_sync_tasks_from_storage(self):
        """Load sync tasks from Redis storage"""
        if not self.redis:
            return
        
        try:
            sync_keys = await self.redis.keys("sync_task:*")
            for key in sync_keys:
                sync_data = await self.redis.get(key)
                if sync_data:
                    sync_task = json.loads(sync_data)
                    # Convert datetime strings back to datetime objects
                    sync_task["created_at"] = datetime.fromisoformat(sync_task["created_at"])
                    if "started_at" in sync_task:
                        sync_task["started_at"] = datetime.fromisoformat(sync_task["started_at"])
                    if "completed_at" in sync_task:
                        sync_task["completed_at"] = datetime.fromisoformat(sync_task["completed_at"])
                    
                    self.sync_tasks[sync_task["sync_id"]] = sync_task
            
            logger.info(f"Loaded {len(self.sync_tasks)} sync tasks from storage")
            
        except Exception as e:
            logger.error(f"Failed to load sync tasks from storage: {e}")
    
    async def _save_sync_task_to_storage(self, sync_task: Dict):
        """Save sync task to Redis storage"""
        if not self.redis:
            return
        
        try:
            sync_key = f"sync_task:{sync_task['sync_id']}"
            
            # Convert datetime objects to strings for JSON serialization
            sync_data = sync_task.copy()
            sync_data["created_at"] = sync_task["created_at"].isoformat()
            if "started_at" in sync_task:
                sync_data["started_at"] = sync_task["started_at"].isoformat()
            if "completed_at" in sync_task:
                sync_data["completed_at"] = sync_task["completed_at"].isoformat()
            
            await self.redis.set(sync_key, json.dumps(sync_data))
            
        except Exception as e:
            logger.error(f"Failed to save sync task {sync_task['sync_id']} to storage: {e}")
    
    async def _remove_sync_task_from_storage(self, sync_id: str):
        """Remove sync task from Redis storage"""
        if not self.redis:
            return
        
        try:
            sync_key = f"sync_task:{sync_id}"
            await self.redis.delete(sync_key)
            
        except Exception as e:
            logger.error(f"Failed to remove sync task {sync_id} from storage: {e}")