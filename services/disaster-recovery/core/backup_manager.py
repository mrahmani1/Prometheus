"""
Advanced Backup Manager
Comprehensive backup management system with automated strategies,
multi-database support, and intelligent scheduling.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid
import structlog
import boto3
from google.cloud import storage as gcs
import psycopg2
import redis
from influxdb_client import InfluxDBClient
from neo4j import GraphDatabase

from ..models.disaster_recovery_models import (
    BackupConfiguration, BackupMetadata, BackupType, BackupStatus,
    DatabaseType, BackupStrategy, IntegrityCheckResult
)

logger = structlog.get_logger(__name__)

class BackupManager:
    """Advanced backup management system"""
    
    def __init__(self):
        self.backup_configs: Dict[str, BackupConfiguration] = {}
        self.active_backups: Dict[str, BackupMetadata] = {}
        self.backup_history: List[BackupMetadata] = []
        self.storage_clients = {}
        self.database_connections = {}
        
    async def initialize(self):
        """Initialize backup manager"""
        try:
            await self._initialize_storage_clients()
            await self._initialize_database_connections()
            await self._load_backup_configurations()
            await self._validate_storage_access()
            
            logger.info("Backup manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize backup manager", error=str(e))
            raise
    
    async def _initialize_storage_clients(self):
        """Initialize storage clients for different providers"""
        try:
            # AWS S3 client
            if os.getenv('AWS_ACCESS_KEY_ID'):
                self.storage_clients['s3'] = boto3.client(
                    's3',
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-west-2')
                )
            
            # Google Cloud Storage client
            if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                self.storage_clients['gcs'] = gcs.Client()
            
            # Azure Blob Storage would be initialized here
            # Local storage is always available
            self.storage_clients['local'] = Path(os.getenv('BACKUP_LOCAL_PATH', '/data/backups'))
            self.storage_clients['local'].mkdir(parents=True, exist_ok=True)
            
            logger.info("Storage clients initialized", 
                       providers=list(self.storage_clients.keys()))
            
        except Exception as e:
            logger.error("Failed to initialize storage clients", error=str(e))
            raise
    
    async def _initialize_database_connections(self):
        """Initialize database connections for backup operations"""
        try:
            # PostgreSQL connection
            if os.getenv('POSTGRES_HOST'):
                self.database_connections['postgresql'] = {
                    'host': os.getenv('POSTGRES_HOST'),
                    'port': int(os.getenv('POSTGRES_PORT', 5432)),
                    'database': os.getenv('POSTGRES_DB'),
                    'user': os.getenv('POSTGRES_USER'),
                    'password': os.getenv('POSTGRES_PASSWORD')
                }
            
            # InfluxDB connection
            if os.getenv('INFLUXDB_URL'):
                self.database_connections['influxdb'] = {
                    'url': os.getenv('INFLUXDB_URL'),
                    'token': os.getenv('INFLUXDB_TOKEN'),
                    'org': os.getenv('INFLUXDB_ORG'),
                    'bucket': os.getenv('INFLUXDB_BUCKET')
                }
            
            # Neo4j connection
            if os.getenv('NEO4J_URI'):
                self.database_connections['neo4j'] = {
                    'uri': os.getenv('NEO4J_URI'),
                    'user': os.getenv('NEO4J_USER'),
                    'password': os.getenv('NEO4J_PASSWORD')
                }
            
            # Redis connection
            if os.getenv('REDIS_HOST'):
                self.database_connections['redis'] = {
                    'host': os.getenv('REDIS_HOST'),
                    'port': int(os.getenv('REDIS_PORT', 6379)),
                    'password': os.getenv('REDIS_PASSWORD'),
                    'db': int(os.getenv('REDIS_DB', 0))
                }
            
            logger.info("Database connections initialized",
                       databases=list(self.database_connections.keys()))
            
        except Exception as e:
            logger.error("Failed to initialize database connections", error=str(e))
            raise
    
    async def _load_backup_configurations(self):
        """Load backup configurations from storage"""
        try:
            config_file = Path('/etc/prometheus/backup-configs.json')
            if config_file.exists():
                with open(config_file, 'r') as f:
                    configs_data = json.load(f)
                
                for config_data in configs_data.get('configurations', []):
                    config = BackupConfiguration(**config_data)
                    self.backup_configs[config.database_type.value] = config
            
            # Create default configurations if none exist
            if not self.backup_configs:
                await self._create_default_configurations()
            
            logger.info("Backup configurations loaded",
                       count=len(self.backup_configs))
            
        except Exception as e:
            logger.error("Failed to load backup configurations", error=str(e))
            # Continue with default configurations
            await self._create_default_configurations()
    
    async def _create_default_configurations(self):
        """Create default backup configurations"""
        default_configs = [
            BackupConfiguration(
                database_type=DatabaseType.POSTGRESQL,
                backup_type=BackupType.FULL,
                schedule_cron="0 2 * * *",  # Daily at 2 AM
                retention_days=30,
                compression_enabled=True,
                encryption_enabled=True,
                verification_enabled=True
            ),
            BackupConfiguration(
                database_type=DatabaseType.INFLUXDB,
                backup_type=BackupType.SNAPSHOT,
                schedule_cron="0 3 * * *",  # Daily at 3 AM
                retention_days=14,
                compression_enabled=True,
                encryption_enabled=True,
                verification_enabled=True
            ),
            BackupConfiguration(
                database_type=DatabaseType.NEO4J,
                backup_type=BackupType.FULL,
                schedule_cron="0 4 * * 0",  # Weekly on Sunday at 4 AM
                retention_days=60,
                compression_enabled=True,
                encryption_enabled=True,
                verification_enabled=True
            ),
            BackupConfiguration(
                database_type=DatabaseType.REDIS,
                backup_type=BackupType.SNAPSHOT,
                schedule_cron="0 */6 * * *",  # Every 6 hours
                retention_days=7,
                compression_enabled=True,
                encryption_enabled=True,
                verification_enabled=True
            )
        ]
        
        for config in default_configs:
            self.backup_configs[config.database_type.value] = config
    
    async def _validate_storage_access(self):
        """Validate access to all configured storage locations"""
        validation_results = {}
        
        for provider, client in self.storage_clients.items():
            try:
                if provider == 's3':
                    # Test S3 access
                    bucket_name = os.getenv('BACKUP_S3_BUCKET')
                    if bucket_name:
                        client.head_bucket(Bucket=bucket_name)
                        validation_results[provider] = True
                
                elif provider == 'gcs':
                    # Test GCS access
                    bucket_name = os.getenv('BACKUP_GCS_BUCKET')
                    if bucket_name:
                        bucket = client.bucket(bucket_name)
                        bucket.exists()
                        validation_results[provider] = True
                
                elif provider == 'local':
                    # Test local storage access
                    test_file = client / 'test_access.txt'
                    test_file.write_text('test')
                    test_file.unlink()
                    validation_results[provider] = True
                
            except Exception as e:
                logger.warning(f"Storage validation failed for {provider}", error=str(e))
                validation_results[provider] = False
        
        logger.info("Storage validation completed", results=validation_results)
    
    async def create_backup(self, backup_request: Dict[str, Any]) -> str:
        """Create a new backup"""
        try:
            database_type = DatabaseType(backup_request['database_type'])
            backup_type = BackupType(backup_request.get('backup_type', 'full'))
            
            backup_metadata = BackupMetadata(
                backup_id=str(uuid.uuid4()),
                database_name=backup_request['database_name'],
                database_type=database_type,
                backup_type=backup_type,
                start_time=datetime.utcnow(),
                status=BackupStatus.PENDING
            )
            
            self.active_backups[backup_metadata.backup_id] = backup_metadata
            
            logger.info("Backup created", 
                       backup_id=backup_metadata.backup_id,
                       database_type=database_type.value,
                       backup_type=backup_type.value)
            
            return backup_metadata.backup_id
            
        except Exception as e:
            logger.error("Failed to create backup", error=str(e))
            raise
    
    async def execute_backup(self, backup_id: str):
        """Execute a backup operation"""
        try:
            backup_metadata = self.active_backups.get(backup_id)
            if not backup_metadata:
                raise ValueError(f"Backup {backup_id} not found")
            
            backup_metadata.status = BackupStatus.IN_PROGRESS
            backup_metadata.start_time = datetime.utcnow()
            
            logger.info("Starting backup execution", backup_id=backup_id)
            
            # Execute database-specific backup
            if backup_metadata.database_type == DatabaseType.POSTGRESQL:
                await self._backup_postgresql(backup_metadata)
            elif backup_metadata.database_type == DatabaseType.INFLUXDB:
                await self._backup_influxdb(backup_metadata)
            elif backup_metadata.database_type == DatabaseType.NEO4J:
                await self._backup_neo4j(backup_metadata)
            elif backup_metadata.database_type == DatabaseType.REDIS:
                await self._backup_redis(backup_metadata)
            else:
                raise ValueError(f"Unsupported database type: {backup_metadata.database_type}")
            
            # Upload to storage locations
            await self._upload_backup_to_storage(backup_metadata)
            
            # Verify backup integrity
            if self.backup_configs[backup_metadata.database_type.value].verification_enabled:
                await self._verify_backup_integrity(backup_metadata)
            
            backup_metadata.status = BackupStatus.COMPLETED
            backup_metadata.end_time = datetime.utcnow()
            
            # Move to history
            self.backup_history.append(backup_metadata)
            del self.active_backups[backup_id]
            
            logger.info("Backup completed successfully", 
                       backup_id=backup_id,
                       duration_minutes=(backup_metadata.end_time - backup_metadata.start_time).total_seconds() / 60)
            
        except Exception as e:
            backup_metadata.status = BackupStatus.FAILED
            backup_metadata.error_message = str(e)
            backup_metadata.end_time = datetime.utcnow()
            
            logger.error("Backup execution failed", 
                        backup_id=backup_id, 
                        error=str(e))
            raise
    
    async def _backup_postgresql(self, backup_metadata: BackupMetadata):
        """Execute PostgreSQL backup"""
        try:
            conn_params = self.database_connections['postgresql']
            backup_file = f"/tmp/{backup_metadata.backup_id}.sql"
            
            # Build pg_dump command
            cmd = [
                'pg_dump',
                f"--host={conn_params['host']}",
                f"--port={conn_params['port']}",
                f"--username={conn_params['user']}",
                f"--dbname={backup_metadata.database_name}",
                '--verbose',
                '--no-password',
                f"--file={backup_file}"
            ]
            
            if backup_metadata.backup_type == BackupType.FULL:
                cmd.extend(['--create', '--clean'])
            
            # Set password via environment
            env = os.environ.copy()
            env['PGPASSWORD'] = conn_params['password']
            
            # Execute backup
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"pg_dump failed: {stderr.decode()}")
            
            # Compress if enabled
            if self.backup_configs[backup_metadata.database_type.value].compression_enabled:
                compressed_file = f"{backup_file}.gz"
                await self._compress_file(backup_file, compressed_file)
                backup_file = compressed_file
            
            # Calculate file size and checksum
            backup_metadata.size_bytes = os.path.getsize(backup_file)
            backup_metadata.checksum = await self._calculate_checksum(backup_file)
            backup_metadata.storage_location = backup_file
            
            logger.info("PostgreSQL backup completed", 
                       backup_id=backup_metadata.backup_id,
                       size_mb=backup_metadata.size_bytes / (1024 * 1024))
            
        except Exception as e:
            logger.error("PostgreSQL backup failed", 
                        backup_id=backup_metadata.backup_id,
                        error=str(e))
            raise
    
    async def _backup_influxdb(self, backup_metadata: BackupMetadata):
        """Execute InfluxDB backup"""
        try:
            conn_params = self.database_connections['influxdb']
            backup_dir = f"/tmp/{backup_metadata.backup_id}_influx"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Use InfluxDB backup command
            cmd = [
                'influx', 'backup',
                '--host', conn_params['url'],
                '--token', conn_params['token'],
                '--org', conn_params['org'],
                '--bucket', backup_metadata.database_name,
                backup_dir
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"InfluxDB backup failed: {stderr.decode()}")
            
            # Create tar archive
            backup_file = f"/tmp/{backup_metadata.backup_id}.tar"
            await self._create_tar_archive(backup_dir, backup_file)
            
            # Compress if enabled
            if self.backup_configs[backup_metadata.database_type.value].compression_enabled:
                compressed_file = f"{backup_file}.gz"
                await self._compress_file(backup_file, compressed_file)
                backup_file = compressed_file
            
            backup_metadata.size_bytes = os.path.getsize(backup_file)
            backup_metadata.checksum = await self._calculate_checksum(backup_file)
            backup_metadata.storage_location = backup_file
            
            logger.info("InfluxDB backup completed", 
                       backup_id=backup_metadata.backup_id,
                       size_mb=backup_metadata.size_bytes / (1024 * 1024))
            
        except Exception as e:
            logger.error("InfluxDB backup failed", 
                        backup_id=backup_metadata.backup_id,
                        error=str(e))
            raise
    
    async def _backup_neo4j(self, backup_metadata: BackupMetadata):
        """Execute Neo4j backup"""
        try:
            conn_params = self.database_connections['neo4j']
            backup_file = f"/tmp/{backup_metadata.backup_id}.dump"
            
            # Use neo4j-admin backup command
            cmd = [
                'neo4j-admin', 'backup',
                '--from', conn_params['uri'],
                '--backup-dir', '/tmp',
                '--name', backup_metadata.backup_id
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Neo4j backup failed: {stderr.decode()}")
            
            # Compress if enabled
            if self.backup_configs[backup_metadata.database_type.value].compression_enabled:
                compressed_file = f"{backup_file}.gz"
                await self._compress_file(backup_file, compressed_file)
                backup_file = compressed_file
            
            backup_metadata.size_bytes = os.path.getsize(backup_file)
            backup_metadata.checksum = await self._calculate_checksum(backup_file)
            backup_metadata.storage_location = backup_file
            
            logger.info("Neo4j backup completed", 
                       backup_id=backup_metadata.backup_id,
                       size_mb=backup_metadata.size_bytes / (1024 * 1024))
            
        except Exception as e:
            logger.error("Neo4j backup failed", 
                        backup_id=backup_metadata.backup_id,
                        error=str(e))
            raise
    
    async def _backup_redis(self, backup_metadata: BackupMetadata):
        """Execute Redis backup"""
        try:
            conn_params = self.database_connections['redis']
            backup_file = f"/tmp/{backup_metadata.backup_id}.rdb"
            
            # Connect to Redis and trigger BGSAVE
            r = redis.Redis(
                host=conn_params['host'],
                port=conn_params['port'],
                password=conn_params['password'],
                db=conn_params['db']
            )
            
            # Trigger background save
            r.bgsave()
            
            # Wait for save to complete
            while r.lastsave() == r.lastsave():
                await asyncio.sleep(1)
            
            # Copy RDB file
            redis_data_dir = os.getenv('REDIS_DATA_DIR', '/data')
            source_rdb = os.path.join(redis_data_dir, 'dump.rdb')
            
            if os.path.exists(source_rdb):
                import shutil
                shutil.copy2(source_rdb, backup_file)
            else:
                raise Exception("Redis RDB file not found")
            
            # Compress if enabled
            if self.backup_configs[backup_metadata.database_type.value].compression_enabled:
                compressed_file = f"{backup_file}.gz"
                await self._compress_file(backup_file, compressed_file)
                backup_file = compressed_file
            
            backup_metadata.size_bytes = os.path.getsize(backup_file)
            backup_metadata.checksum = await self._calculate_checksum(backup_file)
            backup_metadata.storage_location = backup_file
            
            logger.info("Redis backup completed", 
                       backup_id=backup_metadata.backup_id,
                       size_mb=backup_metadata.size_bytes / (1024 * 1024))
            
        except Exception as e:
            logger.error("Redis backup failed", 
                        backup_id=backup_metadata.backup_id,
                        error=str(e))
            raise
    
    async def _compress_file(self, source_file: str, target_file: str):
        """Compress a file using gzip"""
        cmd = ['gzip', '-c', source_file]
        
        with open(target_file, 'wb') as f:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=f,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Compression failed: {stderr.decode()}")
        
        # Remove original file
        os.unlink(source_file)
    
    async def _create_tar_archive(self, source_dir: str, target_file: str):
        """Create tar archive from directory"""
        cmd = ['tar', '-cf', target_file, '-C', os.path.dirname(source_dir), os.path.basename(source_dir)]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Tar archive creation failed: {stderr.decode()}")
    
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def _upload_backup_to_storage(self, backup_metadata: BackupMetadata):
        """Upload backup to configured storage locations"""
        config = self.backup_configs[backup_metadata.database_type.value]
        
        for storage_location in config.storage_locations:
            try:
                if storage_location == 's3':
                    await self._upload_to_s3(backup_metadata)
                elif storage_location == 'gcs':
                    await self._upload_to_gcs(backup_metadata)
                elif storage_location == 'local':
                    await self._copy_to_local_storage(backup_metadata)
                
                logger.info("Backup uploaded to storage", 
                           backup_id=backup_metadata.backup_id,
                           storage=storage_location)
                
            except Exception as e:
                logger.error("Failed to upload backup to storage", 
                            backup_id=backup_metadata.backup_id,
                            storage=storage_location,
                            error=str(e))
                # Continue with other storage locations
    
    async def _upload_to_s3(self, backup_metadata: BackupMetadata):
        """Upload backup to AWS S3"""
        if 's3' not in self.storage_clients:
            return
        
        bucket_name = os.getenv('BACKUP_S3_BUCKET')
        if not bucket_name:
            return
        
        s3_key = f"backups/{backup_metadata.database_type.value}/{backup_metadata.backup_id}"
        
        self.storage_clients['s3'].upload_file(
            backup_metadata.storage_location,
            bucket_name,
            s3_key
        )
    
    async def _upload_to_gcs(self, backup_metadata: BackupMetadata):
        """Upload backup to Google Cloud Storage"""
        if 'gcs' not in self.storage_clients:
            return
        
        bucket_name = os.getenv('BACKUP_GCS_BUCKET')
        if not bucket_name:
            return
        
        bucket = self.storage_clients['gcs'].bucket(bucket_name)
        blob_name = f"backups/{backup_metadata.database_type.value}/{backup_metadata.backup_id}"
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(backup_metadata.storage_location)
    
    async def _copy_to_local_storage(self, backup_metadata: BackupMetadata):
        """Copy backup to local storage"""
        local_path = self.storage_clients['local']
        target_dir = local_path / backup_metadata.database_type.value
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_file = target_dir / f"{backup_metadata.backup_id}"
        
        import shutil
        shutil.copy2(backup_metadata.storage_location, target_file)
    
    async def _verify_backup_integrity(self, backup_metadata: BackupMetadata):
        """Verify backup integrity"""
        try:
            backup_metadata.status = BackupStatus.VERIFYING
            
            # Verify checksum
            current_checksum = await self._calculate_checksum(backup_metadata.storage_location)
            if current_checksum != backup_metadata.checksum:
                raise Exception("Checksum verification failed")
            
            # Database-specific verification
            if backup_metadata.database_type == DatabaseType.POSTGRESQL:
                await self._verify_postgresql_backup(backup_metadata)
            
            backup_metadata.verification_status = "passed"
            backup_metadata.status = BackupStatus.VERIFIED
            
            logger.info("Backup verification passed", 
                       backup_id=backup_metadata.backup_id)
            
        except Exception as e:
            backup_metadata.verification_status = f"failed: {str(e)}"
            logger.error("Backup verification failed", 
                        backup_id=backup_metadata.backup_id,
                        error=str(e))
            raise
    
    async def _verify_postgresql_backup(self, backup_metadata: BackupMetadata):
        """Verify PostgreSQL backup by attempting restore to test database"""
        try:
            # This would restore to a test database to verify integrity
            # Implementation would depend on available test environment
            pass
        except Exception as e:
            logger.error("PostgreSQL backup verification failed", error=str(e))
            raise
    
    async def execute_scheduled_backups(self):
        """Execute scheduled backup operations"""
        try:
            current_time = datetime.utcnow()
            
            for db_type, config in self.backup_configs.items():
                # Check if backup should run based on schedule
                # This is a simplified check - in production, use a proper cron parser
                if await self._should_run_backup(config, current_time):
                    backup_request = {
                        'database_type': db_type,
                        'database_name': self._get_database_name(db_type),
                        'backup_type': config.backup_type.value
                    }
                    
                    backup_id = await self.create_backup(backup_request)
                    asyncio.create_task(self.execute_backup(backup_id))
                    
                    logger.info("Scheduled backup initiated", 
                               database_type=db_type,
                               backup_id=backup_id)
            
        except Exception as e:
            logger.error("Scheduled backup execution failed", error=str(e))
    
    async def _should_run_backup(self, config: BackupConfiguration, current_time: datetime) -> bool:
        """Check if backup should run based on schedule"""
        # Simplified implementation - in production, use croniter or similar
        if config.schedule_cron == "0 2 * * *":  # Daily at 2 AM
            return current_time.hour == 2 and current_time.minute < 5
        elif config.schedule_cron == "0 3 * * *":  # Daily at 3 AM
            return current_time.hour == 3 and current_time.minute < 5
        elif config.schedule_cron == "0 4 * * 0":  # Weekly on Sunday at 4 AM
            return current_time.weekday() == 6 and current_time.hour == 4 and current_time.minute < 5
        elif config.schedule_cron == "0 */6 * * *":  # Every 6 hours
            return current_time.hour % 6 == 0 and current_time.minute < 5
        
        return False
    
    def _get_database_name(self, db_type: str) -> str:
        """Get database name for backup"""
        db_names = {
            'postgresql': os.getenv('POSTGRES_DB', 'prometheus'),
            'influxdb': os.getenv('INFLUXDB_BUCKET', 'prometheus'),
            'neo4j': 'neo4j',
            'redis': '0'
        }
        return db_names.get(db_type, 'default')
    
    async def get_backup_status(self, backup_id: str) -> Dict[str, Any]:
        """Get backup status"""
        # Check active backups first
        if backup_id in self.active_backups:
            backup = self.active_backups[backup_id]
        else:
            # Check backup history
            backup = next((b for b in self.backup_history if b.backup_id == backup_id), None)
        
        if not backup:
            raise ValueError(f"Backup {backup_id} not found")
        
        return {
            'backup_id': backup.backup_id,
            'database_name': backup.database_name,
            'database_type': backup.database_type.value,
            'backup_type': backup.backup_type.value,
            'status': backup.status.value,
            'start_time': backup.start_time.isoformat() if backup.start_time else None,
            'end_time': backup.end_time.isoformat() if backup.end_time else None,
            'size_bytes': backup.size_bytes,
            'checksum': backup.checksum,
            'verification_status': backup.verification_status,
            'error_message': backup.error_message
        }
    
    async def list_backups(
        self, 
        database: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = self.backup_history.copy()
        
        # Add active backups
        backups.extend(self.active_backups.values())
        
        # Filter by database type
        if database:
            backups = [b for b in backups if b.database_type.value == database]
        
        # Filter by date range
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
            backups = [b for b in backups if b.start_time >= start_dt]
        
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
            backups = [b for b in backups if b.start_time <= end_dt]
        
        # Sort by start time (newest first)
        backups.sort(key=lambda x: x.start_time, reverse=True)
        
        return [
            {
                'backup_id': b.backup_id,
                'database_name': b.database_name,
                'database_type': b.database_type.value,
                'backup_type': b.backup_type.value,
                'status': b.status.value,
                'start_time': b.start_time.isoformat() if b.start_time else None,
                'end_time': b.end_time.isoformat() if b.end_time else None,
                'size_bytes': b.size_bytes
            }
            for b in backups
        ]
    
    async def is_ready(self) -> bool:
        """Check if backup manager is ready"""
        return (
            len(self.storage_clients) > 0 and
            len(self.database_connections) > 0 and
            len(self.backup_configs) > 0
        )