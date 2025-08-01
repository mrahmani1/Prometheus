"""
Local data storage and management system for offline operation
"""

import gzip
import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class StorageType(str, Enum):
    """Data storage types"""

    SENSOR_DATA = "sensor_data"
    ANALYSIS_RESULTS = "analysis_results"
    MODEL_OUTPUTS = "model_outputs"
    SYSTEM_LOGS = "system_logs"
    ALERT_HISTORY = "alert_history"
    CONFIGURATION = "configuration"

class CompressionType(str, Enum):
    """Data compression types"""

    NONE = "none"
    GZIP = "gzip"
    PICKLE = "pickle"

@dataclass
class StorageRecord:
    """Storage record metadata"""

    record_id: str
    storage_type: StorageType
    timestamp: datetime
    data_size: int
    compressed_size: int
    compression_type: CompressionType
    checksum: str
    metadata: Dict[str, Any]
    file_path: Optional[str] = None
    sync_status: str = "pending"
    retention_date: Optional[datetime] = None

class LocalStorageManager:
    """Manages local data storage for offline operation"""

    def __init__(
        self, storage_root: str = "./local_storage", max_storage_gb: float = 10.0
    ):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)

        # Storage configuration
        self.max_storage_bytes = int(max_storage_gb * 1024 * 1024 * 1024)
        self.compression_threshold = 1024  # Compress files larger than 1KB

        # Database for metadata
        self.db_path = self.storage_root / "storage_metadata.db"
        self.db_connection: Optional[sqlite3.Connection] = None
        self.db_lock = threading.Lock()

        # Storage directories
        self.data_dirs = {
            StorageType.SENSOR_DATA: self.storage_root / "sensor_data",
            StorageType.ANALYSIS_RESULTS: self.storage_root / "analysis_results",
            StorageType.MODEL_OUTPUTS: self.storage_root / "model_outputs",
            StorageType.SYSTEM_LOGS: self.storage_root / "system_logs",
            StorageType.ALERT_HISTORY: self.storage_root / "alert_history",
            StorageType.CONFIGURATION: self.storage_root / "configuration",
        }

        # Create storage directories
        for storage_dir in self.data_dirs.values():
            storage_dir.mkdir(parents=True, exist_ok=True)

        # Storage statistics
        self.stats = {
            "total_records": 0,
            "total_size_bytes": 0,
            "records_by_type": {storage_type.value: 0 for storage_type in StorageType},
            "size_by_type": {storage_type.value: 0 for storage_type in StorageType},
            "compression_ratio": 0.0,
            "last_cleanup": None,
        }

        self._initialize_database()
        self._update_statistics()

    def _initialize_database(self):
        """Initialize SQLite database for storage metadata"""
        try:
            with self.db_lock:
                self.db_connection = sqlite3.connect(
                    str(self.db_path), check_same_thread=False
                )
                cursor = self.db_connection.cursor()

                # Create storage metadata table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS storage_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        record_id TEXT UNIQUE,
                        storage_type TEXT,
                        timestamp REAL,
                        data_size INTEGER,
                        compressed_size INTEGER,
                        compression_type TEXT,
                        checksum TEXT,
                        metadata TEXT,
                        file_path TEXT,
                        sync_status TEXT DEFAULT 'pending',
                        retention_date REAL,
                        created_at REAL DEFAULT (julianday('now'))
                    )
                """
                )

                # Create indexes
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_record_id ON storage_records(record_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_storage_type ON storage_records(storage_type)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON storage_records(timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sync_status ON storage_records(sync_status)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_retention_date ON storage_records(retention_date)"
                )

                self.db_connection.commit()
                logger.info("Local storage database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize storage database: {e}")
            raise

    async def store_data(
        self,
        storage_type: StorageType,
        data: Union[bytes, Dict[str, Any], str],
        metadata: Optional[Dict[str, Any]] = None,
        retention_days: Optional[int] = None,
    ) -> str:
        """Store data locally with optional compression"""
        try:
            # Generate record ID
            timestamp = datetime.utcnow()
            record_id = f"{storage_type.value}_{timestamp.timestamp()}_{hash(str(data)) % 10000}"

            # Serialize data if needed
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode("utf-8")
                compression_type = CompressionType.NONE
            elif isinstance(data, str):
                data_bytes = data.encode("utf-8")
                compression_type = CompressionType.NONE
            else:
                data_bytes = data
                compression_type = CompressionType.NONE

            original_size = len(data_bytes)

            # Apply compression if data is large enough
            if original_size > self.compression_threshold:
                compressed_data = gzip.compress(data_bytes)
                if (
                    len(compressed_data) < original_size * 0.9
                ):  # Only use if significant compression
                    data_bytes = compressed_data
                    compression_type = CompressionType.GZIP

            compressed_size = len(data_bytes)

            # Calculate checksum
            checksum = hashlib.sha256(data_bytes).hexdigest()

            # Determine file path
            storage_dir = self.data_dirs[storage_type]
            date_subdir = storage_dir / timestamp.strftime("%Y/%m/%d")
            date_subdir.mkdir(parents=True, exist_ok=True)

            file_extension = (
                ".gz" if compression_type == CompressionType.GZIP else ".dat"
            )
            file_path = date_subdir / f"{record_id}{file_extension}"

            # Write data to file
            with open(file_path, "wb") as f:
                f.write(data_bytes)

            # Calculate retention date
            retention_date = None
            if retention_days:
                retention_date = timestamp + timedelta(days=retention_days)

            # Create storage record
            storage_record = StorageRecord(
                record_id=record_id,
                storage_type=storage_type,
                timestamp=timestamp,
                data_size=original_size,
                compressed_size=compressed_size,
                compression_type=compression_type,
                checksum=checksum,
                metadata=metadata or {},
                file_path=str(file_path.relative_to(self.storage_root)),
                retention_date=retention_date,
            )

            # Store metadata in database
            await self._store_record_metadata(storage_record)

            # Update statistics
            self._update_record_stats(storage_record, added=True)

            logger.debug(
                f"Stored data record: {record_id} ({original_size} -> {compressed_size} bytes)"
            )

            # Check storage limits
            await self._check_storage_limits()

            return record_id

        except Exception as e:
            logger.error(f"Error storing data: {e}")
            raise

    async def retrieve_data(
        self, record_id: str
    ) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Retrieve data by record ID"""
        try:
            # Get record metadata
            record = await self._get_record_metadata(record_id)
            if not record:
                return None

            # Read data from file
            file_path = self.storage_root / record.file_path
            if not file_path.exists():
                logger.error(f"Data file not found: {file_path}")
                return None

            with open(file_path, "rb") as f:
                data_bytes = f.read()

            # Verify checksum
            actual_checksum = hashlib.sha256(data_bytes).hexdigest()
            if actual_checksum != record.checksum:
                logger.error(f"Checksum mismatch for record {record_id}")
                return None

            # Decompress if needed
            if record.compression_type == CompressionType.GZIP:
                data_bytes = gzip.decompress(data_bytes)

            return data_bytes, record.metadata

        except Exception as e:
            logger.error(f"Error retrieving data {record_id}: {e}")
            return None

    async def list_records(
        self,
        storage_type: Optional[StorageType] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        sync_status: Optional[str] = None,
        limit: int = 100,
    ) -> List[StorageRecord]:
        """List storage records with optional filters"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()

                query = "SELECT * FROM storage_records WHERE 1=1"
                params = []

                if storage_type:
                    query += " AND storage_type = ?"
                    params.append(storage_type.value)

                if since:
                    query += " AND timestamp >= ?"
                    params.append(since.timestamp())

                if until:
                    query += " AND timestamp <= ?"
                    params.append(until.timestamp())

                if sync_status:
                    query += " AND sync_status = ?"
                    params.append(sync_status)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                records = []
                for row in rows:
                    record = self._row_to_storage_record(row)
                    records.append(record)

                return records

        except Exception as e:
            logger.error(f"Error listing records: {e}")
            return []

    async def delete_record(self, record_id: str) -> bool:
        """Delete a storage record and its data file"""
        try:
            # Get record metadata
            record = await self._get_record_metadata(record_id)
            if not record:
                return False

            # Delete data file
            file_path = self.storage_root / record.file_path
            if file_path.exists():
                file_path.unlink()

            # Delete metadata from database
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute(
                    "DELETE FROM storage_records WHERE record_id = ?", (record_id,)
                )
                self.db_connection.commit()

            # Update statistics
            self._update_record_stats(record, added=False)

            logger.debug(f"Deleted storage record: {record_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting record {record_id}: {e}")
            return False

    async def update_sync_status(self, record_id: str, sync_status: str) -> bool:
        """Update sync status for a record"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute(
                    "UPDATE storage_records SET sync_status = ? WHERE record_id = ?",
                    (sync_status, record_id),
                )
                self.db_connection.commit()

                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error updating sync status for {record_id}: {e}")
            return False

    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        await self._update_statistics()

        # Calculate storage usage percentage
        usage_percentage = (
            self.stats["total_size_bytes"] / self.max_storage_bytes
        ) * 100

        # Get sync statistics
        sync_stats = await self._get_sync_statistics()

        return {
            **self.stats,
            "max_storage_bytes": self.max_storage_bytes,
            "usage_percentage": usage_percentage,
            "available_bytes": self.max_storage_bytes - self.stats["total_size_bytes"],
            "sync_statistics": sync_stats,
        }

    async def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data based on retention policies"""
        try:
            current_time = datetime.utcnow()
            cleanup_stats = {"deleted_records": 0, "freed_bytes": 0}

            # Find expired records
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute(
                    """
                    SELECT record_id, file_path, compressed_size
                    FROM storage_records
                    WHERE retention_date IS NOT NULL AND retention_date < ?
                """,
                    (current_time.timestamp(),),
                )

                expired_records = cursor.fetchall()

            # Delete expired records
            for record_id, file_path, compressed_size in expired_records:
                try:
                    # Delete file
                    full_path = self.storage_root / file_path
                    if full_path.exists():
                        full_path.unlink()

                    # Delete metadata
                    with self.db_lock:
                        cursor = self.db_connection.cursor()
                        cursor.execute(
                            "DELETE FROM storage_records WHERE record_id = ?",
                            (record_id,),
                        )
                        self.db_connection.commit()

                    cleanup_stats["deleted_records"] += 1
                    cleanup_stats["freed_bytes"] += compressed_size

                except Exception as e:
                    logger.error(f"Error deleting expired record {record_id}: {e}")

            self.stats["last_cleanup"] = current_time
            logger.info(
                f"Cleanup completed: {cleanup_stats['deleted_records']} records, {cleanup_stats['freed_bytes']} bytes freed"
            )

            return cleanup_stats

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"deleted_records": 0, "freed_bytes": 0}

    async def _check_storage_limits(self):
        """Check storage limits and cleanup if necessary"""
        try:
            if (
                self.stats["total_size_bytes"] > self.max_storage_bytes * 0.9
            ):  # 90% threshold
                logger.warning("Storage usage above 90%, initiating cleanup")

                # First, clean up expired data
                await self.cleanup_expired_data()

                # If still over limit, clean up oldest non-critical data
                if self.stats["total_size_bytes"] > self.max_storage_bytes * 0.85:
                    await self._cleanup_oldest_data()

        except Exception as e:
            logger.error(f"Error checking storage limits: {e}")

    async def _cleanup_oldest_data(self):
        """Clean up oldest data to free space"""
        try:
            # Get oldest records (excluding critical data)
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute(
                    """
                    SELECT record_id FROM storage_records
                    WHERE storage_type NOT IN ('configuration', 'alert_history')
                    AND sync_status = 'synced'
                    ORDER BY timestamp ASC
                    LIMIT 100
                """
                )

                old_records = [row[0] for row in cursor.fetchall()]

            # Delete old records
            deleted_count = 0
            for record_id in old_records:
                if await self.delete_record(record_id):
                    deleted_count += 1

                # Stop if we've freed enough space
                if self.stats["total_size_bytes"] < self.max_storage_bytes * 0.8:
                    break

            logger.info(f"Cleaned up {deleted_count} old records to free space")

        except Exception as e:
            logger.error(f"Error cleaning up oldest data: {e}")

    async def _store_record_metadata(self, record: StorageRecord):
        """Store record metadata in database"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute(
                    """
                    INSERT INTO storage_records
                    (record_id, storage_type, timestamp, data_size, compressed_size,
                     compression_type, checksum, metadata, file_path, sync_status, retention_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record.record_id,
                        record.storage_type.value,
                        record.timestamp.timestamp(),
                        record.data_size,
                        record.compressed_size,
                        record.compression_type.value,
                        record.checksum,
                        json.dumps(record.metadata),
                        record.file_path,
                        record.sync_status,
                        (
                            record.retention_date.timestamp()
                            if record.retention_date
                            else None
                        ),
                    ),
                )

                self.db_connection.commit()

        except Exception as e:
            logger.error(f"Error storing record metadata: {e}")
            raise

    async def _get_record_metadata(self, record_id: str) -> Optional[StorageRecord]:
        """Get record metadata from database"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute(
                    "SELECT * FROM storage_records WHERE record_id = ?", (record_id,)
                )
                row = cursor.fetchone()

                if row:
                    return self._row_to_storage_record(row)

                return None

        except Exception as e:
            logger.error(f"Error getting record metadata: {e}")
            return None

    def _row_to_storage_record(self, row) -> StorageRecord:
        """Convert database row to StorageRecord"""
        return StorageRecord(
            record_id=row[1],
            storage_type=StorageType(row[2]),
            timestamp=datetime.fromtimestamp(row[3]),
            data_size=row[4],
            compressed_size=row[5],
            compression_type=CompressionType(row[6]),
            checksum=row[7],
            metadata=json.loads(row[8]) if row[8] else {},
            file_path=row[9],
            sync_status=row[10],
            retention_date=datetime.fromtimestamp(row[11]) if row[11] else None,
        )

    def _update_record_stats(self, record: StorageRecord, added: bool):
        """Update statistics when record is added or removed"""
        multiplier = 1 if added else -1

        self.stats["total_records"] += multiplier
        self.stats["total_size_bytes"] += record.compressed_size * multiplier
        self.stats["records_by_type"][record.storage_type.value] += multiplier
        self.stats["size_by_type"][record.storage_type.value] += (
            record.compressed_size * multiplier
        )

    async def _update_statistics(self):
        """Update storage statistics from database"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()

                # Total records and size
                cursor.execute(
                    "SELECT COUNT(*), SUM(compressed_size) FROM storage_records"
                )
                total_records, total_size = cursor.fetchone()

                self.stats["total_records"] = total_records or 0
                self.stats["total_size_bytes"] = total_size or 0

                # Records by type
                cursor.execute(
                    """
                    SELECT storage_type, COUNT(*), SUM(compressed_size)
                    FROM storage_records
                    GROUP BY storage_type
                """
                )

                # Reset type stats
                for storage_type in StorageType:
                    self.stats["records_by_type"][storage_type.value] = 0
                    self.stats["size_by_type"][storage_type.value] = 0

                for storage_type, count, size in cursor.fetchall():
                    self.stats["records_by_type"][storage_type] = count
                    self.stats["size_by_type"][storage_type] = size or 0

                # Compression ratio
                cursor.execute(
                    "SELECT SUM(data_size), SUM(compressed_size) FROM storage_records"
                )
                total_original, total_compressed = cursor.fetchone()

                if total_original and total_compressed and total_original > 0:
                    self.stats["compression_ratio"] = 1.0 - (
                        total_compressed / total_original
                    )
                else:
                    self.stats["compression_ratio"] = 0.0

        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    async def _get_sync_statistics(self) -> Dict[str, int]:
        """Get synchronization statistics"""
        try:
            with self.db_lock:
                cursor = self.db_connection.cursor()

                cursor.execute(
                    """
                    SELECT sync_status, COUNT(*)
                    FROM storage_records
                    GROUP BY sync_status
                """
                )

                sync_stats = {}
                for status, count in cursor.fetchall():
                    sync_stats[status] = count

                return sync_stats

        except Exception as e:
            logger.error(f"Error getting sync statistics: {e}")
            return {}

    async def export_data(
        self,
        storage_type: Optional[StorageType] = None,
        since: Optional[datetime] = None,
        export_path: Optional[str] = None,
    ) -> str:
        """Export data for backup or transfer"""
        try:
            if not export_path:
                export_path = (
                    self.storage_root
                    / f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                )

            # Get records to export
            records = await self.list_records(
                storage_type=storage_type,
                since=since,
                limit=10000,  # Large limit for export
            )

            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "storage_type": storage_type.value if storage_type else "all",
                "record_count": len(records),
                "records": [],
            }

            for record in records:
                # Get data content
                data_content, metadata = await self.retrieve_data(record.record_id)

                if data_content:
                    export_data["records"].append(
                        {
                            "record_id": record.record_id,
                            "storage_type": record.storage_type.value,
                            "timestamp": record.timestamp.isoformat(),
                            "metadata": record.metadata,
                            "data": data_content.hex(),  # Hex encode binary data
                            "checksum": record.checksum,
                        }
                    )

            # Write export file
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Exported {len(records)} records to {export_path}")
            return str(export_path)

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise

    async def shutdown(self):
        """Shutdown storage manager and cleanup resources"""
        try:
            if self.db_connection:
                self.db_connection.close()
            logger.info("Local storage manager shutdown complete")

        except Exception as e:
            logger.error(f"Error during storage manager shutdown: {e}")
