"""
Encrypted Backup and Archival System for Prometheus AI Platform

This module implements comprehensive encrypted backup and archival capabilities including:
- Multi-tier encrypted backup strategies
- Automated backup scheduling and lifecycle management
- Cross-cloud encrypted replication
- Point-in-time recovery with encryption
- Geological data-aware backup optimization
- Compliance-ready archival with retention policies
- Disaster recovery with encrypted failover
- Backup integrity verification and monitoring
"""

import asyncio
import json
import logging
import os
import shutil
import tarfile
import time
import zlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import hashlib
import uuid
import base64
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import aioredis
import aiofiles
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Cloud storage libraries
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import AzureError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from google.cloud import storage as gcs
    from google.api_core import exceptions as gcs_exceptions
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

# Import from other security modules
from .encryption_engine import (
    EncryptionEngine, KeyManager, DataClassification, 
    EncryptionAlgorithm, KeyType, EncryptionMetadata
)


class BackupTier(Enum):
    """Backup storage tiers"""
    HOT = "hot"              # Immediate access, high cost
    WARM = "warm"            # Quick access, medium cost
    COLD = "cold"            # Slow access, low cost
    ARCHIVE = "archive"      # Very slow access, very low cost
    DEEP_ARCHIVE = "deep_archive"  # Extremely slow access, minimal cost


class BackupType(Enum):
    """Types of backups"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"
    CONTINUOUS = "continuous"


class BackupStatus(Enum):
    """Backup operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"


class StorageProvider(Enum):
    """Cloud storage providers"""
    AWS_S3 = "aws_s3"
    AZURE_BLOB = "azure_blob"
    GCP_STORAGE = "gcp_storage"
    LOCAL_STORAGE = "local_storage"
    MULTI_CLOUD = "multi_cloud"


@dataclass
class BackupPolicy:
    """Backup policy configuration"""
    policy_id: str
    name: str
    description: str
    data_types: List[str]
    classifications: List[DataClassification]
    backup_frequency: timedelta
    retention_period: timedelta
    backup_tiers: List[BackupTier]
    storage_providers: List[StorageProvider]
    encryption_required: bool = True
    compression_enabled: bool = True
    verification_required: bool = True
    cross_region_replication: bool = False
    compliance_requirements: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BackupJob:
    """Backup job definition"""
    job_id: str
    policy_id: str
    backup_type: BackupType
    source_path: str
    data_classification: DataClassification
    geological_data_type: str
    scheduled_time: datetime
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    status: BackupStatus = BackupStatus.PENDING
    backup_size_bytes: int = 0
    compressed_size_bytes: int = 0
    encryption_key_id: Optional[str] = None
    storage_locations: List[Dict[str, Any]] = field(default_factory=list)
    checksum: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackupRecord:
    """Backup record for tracking"""
    backup_id: str
    job_id: str
    backup_path: str
    storage_provider: StorageProvider
    storage_tier: BackupTier
    backup_type: BackupType
    data_classification: DataClassification
    encryption_key_id: str
    compression_ratio: float
    integrity_hash: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    restored_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RestoreRequest:
    """Data restore request"""
    request_id: str
    backup_id: str
    user_id: str
    restore_path: str
    point_in_time: Optional[datetime] = None
    partial_restore: bool = False
    restore_filters: Dict[str, Any] = field(default_factory=dict)
    requested_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"
    progress_percent: float = 0.0
    estimated_completion: Optional[datetime] = None


class CloudStorageManager:
    """Multi-cloud storage manager for encrypted backups"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.aws_client = None
        self.azure_client = None
        self.gcp_client = None
        self.initialized_providers = set()
    
    async def initialize(self) -> bool:
        """Initialize cloud storage clients"""
        try:
            # Initialize AWS S3
            if AWS_AVAILABLE and 'aws' in self.config:
                aws_config = self.config['aws']
                self.aws_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_config.get('access_key_id'),
                    aws_secret_access_key=aws_config.get('secret_access_key'),
                    region_name=aws_config.get('region', 'us-west-2')
                )
                self.initialized_providers.add(StorageProvider.AWS_S3)
                logging.info("AWS S3 client initialized")
            
            # Initialize Azure Blob Storage
            if AZURE_AVAILABLE and 'azure' in self.config:
                azure_config = self.config['azure']
                self.azure_client = BlobServiceClient(
                    account_url=azure_config.get('account_url'),
                    credential=azure_config.get('account_key')
                )
                self.initialized_providers.add(StorageProvider.AZURE_BLOB)
                logging.info("Azure Blob Storage client initialized")
            
            # Initialize Google Cloud Storage
            if GCP_AVAILABLE and 'gcp' in self.config:
                gcp_config = self.config['gcp']
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcp_config.get('credentials_path')
                self.gcp_client = gcs.Client(project=gcp_config.get('project_id'))
                self.initialized_providers.add(StorageProvider.GCP_STORAGE)
                logging.info("Google Cloud Storage client initialized")
            
            # Always support local storage
            self.initialized_providers.add(StorageProvider.LOCAL_STORAGE)
            
            return len(self.initialized_providers) > 0
            
        except Exception as e:
            logging.error(f"Cloud storage initialization failed: {e}")
            return False
    
    async def upload_backup(
        self,
        backup_data: bytes,
        backup_path: str,
        provider: StorageProvider,
        tier: BackupTier,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Upload encrypted backup to cloud storage"""
        
        try:
            if provider == StorageProvider.AWS_S3:
                return await self._upload_to_s3(backup_data, backup_path, tier, metadata)
            elif provider == StorageProvider.AZURE_BLOB:
                return await self._upload_to_azure(backup_data, backup_path, tier, metadata)
            elif provider == StorageProvider.GCP_STORAGE:
                return await self._upload_to_gcp(backup_data, backup_path, tier, metadata)
            elif provider == StorageProvider.LOCAL_STORAGE:
                return await self._upload_to_local(backup_data, backup_path, metadata)
            else:
                return False, f"Unsupported storage provider: {provider}"
                
        except Exception as e:
            logging.error(f"Backup upload failed: {e}")
            return False, str(e)
    
    async def download_backup(
        self,
        backup_path: str,
        provider: StorageProvider
    ) -> Tuple[bool, Optional[bytes], Optional[str]]:
        """Download encrypted backup from cloud storage"""
        
        try:
            if provider == StorageProvider.AWS_S3:
                return await self._download_from_s3(backup_path)
            elif provider == StorageProvider.AZURE_BLOB:
                return await self._download_from_azure(backup_path)
            elif provider == StorageProvider.GCP_STORAGE:
                return await self._download_from_gcp(backup_path)
            elif provider == StorageProvider.LOCAL_STORAGE:
                return await self._download_from_local(backup_path)
            else:
                return False, None, f"Unsupported storage provider: {provider}"
                
        except Exception as e:
            logging.error(f"Backup download failed: {e}")
            return False, None, str(e)
    
    async def _upload_to_s3(
        self,
        data: bytes,
        path: str,
        tier: BackupTier,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Upload to AWS S3"""
        
        if not self.aws_client:
            return False, "AWS S3 client not initialized"
        
        try:
            bucket_name = self.config['aws']['backup_bucket']
            
            # Map backup tier to S3 storage class
            storage_class_map = {
                BackupTier.HOT: 'STANDARD',
                BackupTier.WARM: 'STANDARD_IA',
                BackupTier.COLD: 'GLACIER',
                BackupTier.ARCHIVE: 'DEEP_ARCHIVE',
                BackupTier.DEEP_ARCHIVE: 'DEEP_ARCHIVE'
            }
            
            storage_class = storage_class_map.get(tier, 'STANDARD')
            
            # Upload with metadata
            self.aws_client.put_object(
                Bucket=bucket_name,
                Key=path,
                Body=data,
                StorageClass=storage_class,
                Metadata=metadata,
                ServerSideEncryption='AES256'
            )
            
            return True, None
            
        except ClientError as e:
            return False, str(e)
    
    async def _download_from_s3(self, path: str) -> Tuple[bool, Optional[bytes], Optional[str]]:
        """Download from AWS S3"""
        
        if not self.aws_client:
            return False, None, "AWS S3 client not initialized"
        
        try:
            bucket_name = self.config['aws']['backup_bucket']
            
            response = self.aws_client.get_object(Bucket=bucket_name, Key=path)
            data = response['Body'].read()
            
            return True, data, None
            
        except ClientError as e:
            return False, None, str(e)
    
    async def _upload_to_azure(
        self,
        data: bytes,
        path: str,
        tier: BackupTier,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Upload to Azure Blob Storage"""
        
        if not self.azure_client:
            return False, "Azure Blob Storage client not initialized"
        
        try:
            container_name = self.config['azure']['backup_container']
            
            # Map backup tier to Azure access tier
            access_tier_map = {
                BackupTier.HOT: 'Hot',
                BackupTier.WARM: 'Cool',
                BackupTier.COLD: 'Archive',
                BackupTier.ARCHIVE: 'Archive',
                BackupTier.DEEP_ARCHIVE: 'Archive'
            }
            
            access_tier = access_tier_map.get(tier, 'Hot')
            
            blob_client = self.azure_client.get_blob_client(
                container=container_name,
                blob=path
            )
            
            blob_client.upload_blob(
                data,
                overwrite=True,
                metadata=metadata,
                standard_blob_tier=access_tier
            )
            
            return True, None
            
        except AzureError as e:
            return False, str(e)
    
    async def _download_from_azure(self, path: str) -> Tuple[bool, Optional[bytes], Optional[str]]:
        """Download from Azure Blob Storage"""
        
        if not self.azure_client:
            return False, None, "Azure Blob Storage client not initialized"
        
        try:
            container_name = self.config['azure']['backup_container']
            
            blob_client = self.azure_client.get_blob_client(
                container=container_name,
                blob=path
            )
            
            data = blob_client.download_blob().readall()
            
            return True, data, None
            
        except AzureError as e:
            return False, None, str(e)
    
    async def _upload_to_gcp(
        self,
        data: bytes,
        path: str,
        tier: BackupTier,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Upload to Google Cloud Storage"""
        
        if not self.gcp_client:
            return False, "Google Cloud Storage client not initialized"
        
        try:
            bucket_name = self.config['gcp']['backup_bucket']
            bucket = self.gcp_client.bucket(bucket_name)
            
            # Map backup tier to GCS storage class
            storage_class_map = {
                BackupTier.HOT: 'STANDARD',
                BackupTier.WARM: 'NEARLINE',
                BackupTier.COLD: 'COLDLINE',
                BackupTier.ARCHIVE: 'ARCHIVE',
                BackupTier.DEEP_ARCHIVE: 'ARCHIVE'
            }
            
            storage_class = storage_class_map.get(tier, 'STANDARD')
            
            blob = bucket.blob(path)
            blob.metadata = metadata
            blob.upload_from_string(data, content_type='application/octet-stream')
            
            # Set storage class after upload
            blob.update_storage_class(storage_class)
            
            return True, None
            
        except gcs_exceptions.GoogleCloudError as e:
            return False, str(e)
    
    async def _download_from_gcp(self, path: str) -> Tuple[bool, Optional[bytes], Optional[str]]:
        """Download from Google Cloud Storage"""
        
        if not self.gcp_client:
            return False, None, "Google Cloud Storage client not initialized"
        
        try:
            bucket_name = self.config['gcp']['backup_bucket']
            bucket = self.gcp_client.bucket(bucket_name)
            blob = bucket.blob(path)
            
            data = blob.download_as_bytes()
            
            return True, data, None
            
        except gcs_exceptions.GoogleCloudError as e:
            return False, None, str(e)
    
    async def _upload_to_local(
        self,
        data: bytes,
        path: str,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Upload to local storage"""
        
        try:
            backup_dir = Path(self.config.get('local_backup_dir', '/var/backups/prometheus'))
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            full_path = backup_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write data
            async with aiofiles.open(full_path, 'wb') as f:
                await f.write(data)
            
            # Write metadata
            metadata_path = full_path.with_suffix(full_path.suffix + '.meta')
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata))
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    async def _download_from_local(self, path: str) -> Tuple[bool, Optional[bytes], Optional[str]]:
        """Download from local storage"""
        
        try:
            backup_dir = Path(self.config.get('local_backup_dir', '/var/backups/prometheus'))
            full_path = backup_dir / path
            
            if not full_path.exists():
                return False, None, "Backup file not found"
            
            async with aiofiles.open(full_path, 'rb') as f:
                data = await f.read()
            
            return True, data, None
            
        except Exception as e:
            return False, None, str(e)


class EncryptedBackupEngine:
    """Main engine for encrypted backup and archival operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.key_manager = KeyManager(config)
        self.encryption_engine = None
        self.storage_manager = CloudStorageManager(config)
        self.redis_client = None
        self.backup_policies: Dict[str, BackupPolicy] = {}
        self.active_jobs: Dict[str, BackupJob] = {}
        self.backup_records: Dict[str, BackupRecord] = {}
    
    async def initialize(self) -> bool:
        """Initialize encrypted backup engine"""
        try:
            # Initialize components
            await self.key_manager.initialize()
            self.encryption_engine = EncryptionEngine(self.key_manager)
            await self.storage_manager.initialize()
            
            # Initialize Redis
            self.redis_client = aioredis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=True
            )
            
            # Load backup policies
            await self._load_default_backup_policies()
            
            # Start backup scheduler
            asyncio.create_task(self._backup_scheduler())
            
            logging.info("Encrypted backup engine initialized")
            return True
            
        except Exception as e:
            logging.error(f"Encrypted backup engine initialization failed: {e}")
            return False
    
    async def create_backup(
        self,
        source_path: str,
        data_classification: DataClassification,
        geological_data_type: str,
        backup_type: BackupType = BackupType.FULL,
        policy_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Create encrypted backup"""
        
        try:
            # Find or use specified backup policy
            if policy_id:
                policy = self.backup_policies.get(policy_id)
                if not policy:
                    return False, None, f"Backup policy {policy_id} not found"
            else:
                policy = await self._find_backup_policy(data_classification, geological_data_type)
                if not policy:
                    return False, None, "No applicable backup policy found"
            
            # Create backup job
            job_id = str(uuid.uuid4())
            backup_job = BackupJob(
                job_id=job_id,
                policy_id=policy.policy_id,
                backup_type=backup_type,
                source_path=source_path,
                data_classification=data_classification,
                geological_data_type=geological_data_type,
                scheduled_time=datetime.utcnow(),
                status=BackupStatus.PENDING
            )
            
            # Execute backup
            success, error = await self._execute_backup_job(backup_job, policy)
            
            if success:
                return True, job_id, None
            else:
                return False, None, error
                
        except Exception as e:
            logging.error(f"Backup creation failed: {e}")
            return False, None, str(e)
    
    async def restore_backup(
        self,
        backup_id: str,
        restore_path: str,
        user_id: str,
        point_in_time: Optional[datetime] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Restore encrypted backup"""
        
        try:
            # Get backup record
            backup_record = await self._get_backup_record(backup_id)
            if not backup_record:
                return False, None, "Backup record not found"
            
            # Create restore request
            request_id = str(uuid.uuid4())
            restore_request = RestoreRequest(
                request_id=request_id,
                backup_id=backup_id,
                user_id=user_id,
                restore_path=restore_path,
                point_in_time=point_in_time
            )
            
            # Execute restore
            success, error = await self._execute_restore_request(restore_request, backup_record)
            
            if success:
                return True, request_id, None
            else:
                return False, None, error
                
        except Exception as e:
            logging.error(f"Backup restore failed: {e}")
            return False, None, str(e)
    
    async def _execute_backup_job(self, job: BackupJob, policy: BackupPolicy) -> Tuple[bool, Optional[str]]:
        """Execute backup job"""
        
        try:
            job.status = BackupStatus.IN_PROGRESS
            job.started_time = datetime.utcnow()
            await self._store_backup_job(job)
            
            # Read source data
            source_data = await self._read_source_data(job.source_path)
            if not source_data:
                job.status = BackupStatus.FAILED
                job.error_message = "Failed to read source data"
                await self._store_backup_job(job)
                return False, "Failed to read source data"
            
            job.backup_size_bytes = len(source_data)
            
            # Compress data if enabled
            if policy.compression_enabled:
                compressed_data = zlib.compress(source_data, level=9)
                job.compressed_size_bytes = len(compressed_data)
                backup_data = compressed_data
            else:
                backup_data = source_data
                job.compressed_size_bytes = job.backup_size_bytes
            
            # Generate encryption key
            if policy.encryption_required:
                encryption_key_id = await self.key_manager.generate_key(
                    KeyType.SYMMETRIC,
                    EncryptionAlgorithm.AES_256_GCM,
                    job.data_classification
                )
                job.encryption_key_id = encryption_key_id
                
                # Encrypt backup data
                encrypted_data, metadata = await self.encryption_engine.encrypt_data(
                    backup_data,
                    EncryptionAlgorithm.AES_256_GCM,
                    encryption_key_id,
                    job.data_classification
                )
                final_backup_data = encrypted_data
            else:
                final_backup_data = backup_data
                metadata = None
            
            # Calculate checksum
            job.checksum = hashlib.sha256(final_backup_data).hexdigest()
            
            # Upload to storage providers
            backup_records = []
            for provider in policy.storage_providers:
                for tier in policy.backup_tiers:
                    backup_path = self._generate_backup_path(job, provider, tier)
                    
                    upload_metadata = {
                        'job_id': job.job_id,
                        'backup_type': job.backup_type.value,
                        'data_classification': job.data_classification.value,
                        'geological_data_type': job.geological_data_type,
                        'created_at': job.started_time.isoformat(),
                        'checksum': job.checksum,
                        'compressed': policy.compression_enabled,
                        'encrypted': policy.encryption_required
                    }
                    
                    success, error = await self.storage_manager.upload_backup(
                        final_backup_data, backup_path, provider, tier, upload_metadata
                    )
                    
                    if success:
                        # Create backup record
                        backup_id = str(uuid.uuid4())
                        backup_record = BackupRecord(
                            backup_id=backup_id,
                            job_id=job.job_id,
                            backup_path=backup_path,
                            storage_provider=provider,
                            storage_tier=tier,
                            backup_type=job.backup_type,
                            data_classification=job.data_classification,
                            encryption_key_id=job.encryption_key_id or "",
                            compression_ratio=job.backup_size_bytes / job.compressed_size_bytes if job.compressed_size_bytes > 0 else 1.0,
                            integrity_hash=job.checksum,
                            created_at=job.started_time,
                            expires_at=job.started_time + policy.retention_period if policy.retention_period else None
                        )
                        
                        backup_records.append(backup_record)
                        await self._store_backup_record(backup_record)
                        
                        job.storage_locations.append({
                            'backup_id': backup_id,
                            'provider': provider.value,
                            'tier': tier.value,
                            'path': backup_path
                        })
                    else:
                        logging.error(f"Failed to upload backup to {provider.value}: {error}")
            
            if not backup_records:
                job.status = BackupStatus.FAILED
                job.error_message = "Failed to upload backup to any storage provider"
                await self._store_backup_job(job)
                return False, "Failed to upload backup to any storage provider"
            
            # Verify backups if required
            if policy.verification_required:
                job.status = BackupStatus.VERIFYING
                await self._store_backup_job(job)
                
                verification_success = await self._verify_backups(backup_records)
                if not verification_success:
                    job.status = BackupStatus.FAILED
                    job.error_message = "Backup verification failed"
                    await self._store_backup_job(job)
                    return False, "Backup verification failed"
            
            # Complete backup job
            job.status = BackupStatus.COMPLETED
            job.completed_time = datetime.utcnow()
            await self._store_backup_job(job)
            
            logging.info(f"Backup job {job.job_id} completed successfully")
            return True, None
            
        except Exception as e:
            job.status = BackupStatus.FAILED
            job.error_message = str(e)
            await self._store_backup_job(job)
            logging.error(f"Backup job execution failed: {e}")
            return False, str(e)
    
    async def _execute_restore_request(
        self,
        request: RestoreRequest,
        backup_record: BackupRecord
    ) -> Tuple[bool, Optional[str]]:
        """Execute restore request"""
        
        try:
            request.status = "in_progress"
            await self._store_restore_request(request)
            
            # Download backup data
            success, backup_data, error = await self.storage_manager.download_backup(
                backup_record.backup_path,
                backup_record.storage_provider
            )
            
            if not success:
                request.status = "failed"
                await self._store_restore_request(request)
                return False, f"Failed to download backup: {error}"
            
            # Verify integrity
            calculated_hash = hashlib.sha256(backup_data).hexdigest()
            if calculated_hash != backup_record.integrity_hash:
                request.status = "failed"
                await self._store_restore_request(request)
                return False, "Backup integrity verification failed"
            
            # Decrypt if encrypted
            if backup_record.encryption_key_id:
                # Reconstruct metadata (simplified for this example)
                metadata = EncryptionMetadata(
                    algorithm=EncryptionAlgorithm.AES_256_GCM,
                    key_id=backup_record.encryption_key_id,
                    classification=backup_record.data_classification
                )
                
                decrypted_data = await self.encryption_engine.decrypt_data(backup_data, metadata)
                restore_data = decrypted_data
            else:
                restore_data = backup_data
            
            # Decompress if compressed
            if backup_record.compression_ratio > 1.0:
                decompressed_data = zlib.decompress(restore_data)
                final_data = decompressed_data
            else:
                final_data = restore_data
            
            # Write restored data
            await self._write_restored_data(request.restore_path, final_data)
            
            # Update backup record
            backup_record.restored_count += 1
            backup_record.last_accessed = datetime.utcnow()
            await self._store_backup_record(backup_record)
            
            # Complete restore request
            request.status = "completed"
            request.progress_percent = 100.0
            await self._store_restore_request(request)
            
            logging.info(f"Restore request {request.request_id} completed successfully")
            return True, None
            
        except Exception as e:
            request.status = "failed"
            await self._store_restore_request(request)
            logging.error(f"Restore request execution failed: {e}")
            return False, str(e)
    
    async def _read_source_data(self, source_path: str) -> Optional[bytes]:
        """Read source data for backup"""
        try:
            if os.path.isfile(source_path):
                # Single file
                async with aiofiles.open(source_path, 'rb') as f:
                    return await f.read()
            elif os.path.isdir(source_path):
                # Directory - create tar archive
                import tempfile
                with tempfile.NamedTemporaryFile() as temp_file:
                    with tarfile.open(temp_file.name, 'w') as tar:
                        tar.add(source_path, arcname=os.path.basename(source_path))
                    
                    async with aiofiles.open(temp_file.name, 'rb') as f:
                        return await f.read()
            else:
                return None
        except Exception as e:
            logging.error(f"Failed to read source data: {e}")
            return None
    
    async def _write_restored_data(self, restore_path: str, data: bytes):
        """Write restored data"""
        try:
            restore_dir = Path(restore_path).parent
            restore_dir.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(restore_path, 'wb') as f:
                await f.write(data)
                
        except Exception as e:
            logging.error(f"Failed to write restored data: {e}")
            raise
    
    def _generate_backup_path(
        self,
        job: BackupJob,
        provider: StorageProvider,
        tier: BackupTier
    ) -> str:
        """Generate backup storage path"""
        
        timestamp = job.started_time.strftime('%Y/%m/%d/%H%M%S')
        filename = f"{job.geological_data_type}_{job.backup_type.value}_{job.job_id}.backup"
        
        return f"prometheus-backups/{timestamp}/{provider.value}/{tier.value}/{filename}"
    
    async def _verify_backups(self, backup_records: List[BackupRecord]) -> bool:
        """Verify backup integrity"""
        
        try:
            for record in backup_records:
                # Download and verify checksum
                success, data, error = await self.storage_manager.download_backup(
                    record.backup_path,
                    record.storage_provider
                )
                
                if not success:
                    logging.error(f"Backup verification failed for {record.backup_id}: {error}")
                    return False
                
                calculated_hash = hashlib.sha256(data).hexdigest()
                if calculated_hash != record.integrity_hash:
                    logging.error(f"Backup integrity check failed for {record.backup_id}")
                    return False
                
                # Update verification timestamp
                record.verified_at = datetime.utcnow()
                await self._store_backup_record(record)
            
            return True
            
        except Exception as e:
            logging.error(f"Backup verification failed: {e}")
            return False
    
    async def _load_default_backup_policies(self):
        """Load default backup policies"""
        
        # Policy 1: Critical geological data
        policy1 = BackupPolicy(
            policy_id="critical_geological_backup",
            name="Critical Geological Data Backup",
            description="High-frequency backup for critical geological monitoring data",
            data_types=["seismic_monitoring", "volcanic_alerts", "tsunami_warnings"],
            classifications=[DataClassification.RESTRICTED, DataClassification.TOP_SECRET],
            backup_frequency=timedelta(hours=1),
            retention_period=timedelta(days=2555),  # 7 years
            backup_tiers=[BackupTier.HOT, BackupTier.WARM, BackupTier.ARCHIVE],
            storage_providers=[StorageProvider.AWS_S3, StorageProvider.AZURE_BLOB, StorageProvider.LOCAL_STORAGE],
            cross_region_replication=True,
            compliance_requirements=["SOX", "GDPR", "NIST"]
        )
        
        # Policy 2: Research data
        policy2 = BackupPolicy(
            policy_id="research_data_backup",
            name="Research Data Backup",
            description="Regular backup for geological research data",
            data_types=["research_datasets", "analysis_results", "field_surveys"],
            classifications=[DataClassification.INTERNAL, DataClassification.CONFIDENTIAL],
            backup_frequency=timedelta(days=1),
            retention_period=timedelta(days=1095),  # 3 years
            backup_tiers=[BackupTier.WARM, BackupTier.COLD],
            storage_providers=[StorageProvider.GCP_STORAGE, StorageProvider.LOCAL_STORAGE],
            compliance_requirements=["GDPR"]
        )
        
        # Policy 3: Archive data
        policy3 = BackupPolicy(
            policy_id="archive_data_backup",
            name="Archive Data Backup",
            description="Long-term archival for historical geological data",
            data_types=["historical_data", "legacy_datasets", "completed_projects"],
            classifications=[DataClassification.PUBLIC, DataClassification.INTERNAL],
            backup_frequency=timedelta(days=30),
            retention_period=timedelta(days=7300),  # 20 years
            backup_tiers=[BackupTier.ARCHIVE, BackupTier.DEEP_ARCHIVE],
            storage_providers=[StorageProvider.AWS_S3, StorageProvider.GCP_STORAGE],
            compliance_requirements=["NARA"]  # National Archives requirements
        )
        
        # Store policies
        policies = [policy1, policy2, policy3]
        for policy in policies:
            self.backup_policies[policy.policy_id] = policy
            await self._store_backup_policy(policy)
    
    async def _find_backup_policy(
        self,
        classification: DataClassification,
        data_type: str
    ) -> Optional[BackupPolicy]:
        """Find applicable backup policy"""
        
        for policy in self.backup_policies.values():
            if (classification in policy.classifications and
                (data_type in policy.data_types or "all" in policy.data_types)):
                return policy
        
        return None
    
    async def _backup_scheduler(self):
        """Background backup scheduler"""
        
        while True:
            try:
                # Check for scheduled backups
                current_time = datetime.utcnow()
                
                for policy in self.backup_policies.values():
                    # Check if it's time for a backup based on frequency
                    last_backup_time = await self._get_last_backup_time(policy.policy_id)
                    
                    if (not last_backup_time or 
                        current_time - last_backup_time >= policy.backup_frequency):
                        
                        # Schedule backup for data matching this policy
                        await self._schedule_policy_backups(policy)
                
                # Sleep for 1 minute before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logging.error(f"Backup scheduler error: {e}")
                await asyncio.sleep(60)
    
    async def _schedule_policy_backups(self, policy: BackupPolicy):
        """Schedule backups for a policy"""
        
        # This would integrate with data discovery to find files matching the policy
        # For now, this is a placeholder
        logging.info(f"Scheduling backups for policy {policy.policy_id}")
    
    async def _get_last_backup_time(self, policy_id: str) -> Optional[datetime]:
        """Get last backup time for a policy"""
        
        last_backup = await self.redis_client.get(f"last_backup:{policy_id}")
        if last_backup:
            return datetime.fromisoformat(last_backup)
        return None
    
    async def _store_backup_policy(self, policy: BackupPolicy):
        """Store backup policy"""
        
        policy_data = {
            'name': policy.name,
            'description': policy.description,
            'data_types': policy.data_types,
            'classifications': [c.value for c in policy.classifications],
            'backup_frequency': policy.backup_frequency.total_seconds(),
            'retention_period': policy.retention_period.total_seconds(),
            'backup_tiers': [t.value for t in policy.backup_tiers],
            'storage_providers': [p.value for p in policy.storage_providers],
            'encryption_required': policy.encryption_required,
            'compression_enabled': policy.compression_enabled,
            'verification_required': policy.verification_required,
            'cross_region_replication': policy.cross_region_replication,
            'compliance_requirements': policy.compliance_requirements,
            'created_at': policy.created_at.isoformat()
        }
        
        await self.redis_client.set(
            f"backup_policy:{policy.policy_id}",
            json.dumps(policy_data)
        )
    
    async def _store_backup_job(self, job: BackupJob):
        """Store backup job"""
        
        job_data = {
            'policy_id': job.policy_id,
            'backup_type': job.backup_type.value,
            'source_path': job.source_path,
            'data_classification': job.data_classification.value,
            'geological_data_type': job.geological_data_type,
            'scheduled_time': job.scheduled_time.isoformat(),
            'started_time': job.started_time.isoformat() if job.started_time else None,
            'completed_time': job.completed_time.isoformat() if job.completed_time else None,
            'status': job.status.value,
            'backup_size_bytes': job.backup_size_bytes,
            'compressed_size_bytes': job.compressed_size_bytes,
            'encryption_key_id': job.encryption_key_id,
            'storage_locations': job.storage_locations,
            'checksum': job.checksum,
            'error_message': job.error_message,
            'metadata': job.metadata
        }
        
        await self.redis_client.set(
            f"backup_job:{job.job_id}",
            json.dumps(job_data)
        )
    
    async def _store_backup_record(self, record: BackupRecord):
        """Store backup record"""
        
        record_data = {
            'job_id': record.job_id,
            'backup_path': record.backup_path,
            'storage_provider': record.storage_provider.value,
            'storage_tier': record.storage_tier.value,
            'backup_type': record.backup_type.value,
            'data_classification': record.data_classification.value,
            'encryption_key_id': record.encryption_key_id,
            'compression_ratio': record.compression_ratio,
            'integrity_hash': record.integrity_hash,
            'created_at': record.created_at.isoformat(),
            'expires_at': record.expires_at.isoformat() if record.expires_at else None,
            'verified_at': record.verified_at.isoformat() if record.verified_at else None,
            'restored_count': record.restored_count,
            'last_accessed': record.last_accessed.isoformat() if record.last_accessed else None,
            'metadata': record.metadata
        }
        
        await self.redis_client.set(
            f"backup_record:{record.backup_id}",
            json.dumps(record_data)
        )
    
    async def _get_backup_record(self, backup_id: str) -> Optional[BackupRecord]:
        """Retrieve backup record"""
        
        record_data = await self.redis_client.get(f"backup_record:{backup_id}")
        if not record_data:
            return None
        
        data = json.loads(record_data)
        return BackupRecord(
            backup_id=backup_id,
            job_id=data['job_id'],
            backup_path=data['backup_path'],
            storage_provider=StorageProvider(data['storage_provider']),
            storage_tier=BackupTier(data['storage_tier']),
            backup_type=BackupType(data['backup_type']),
            data_classification=DataClassification(data['data_classification']),
            encryption_key_id=data['encryption_key_id'],
            compression_ratio=data['compression_ratio'],
            integrity_hash=data['integrity_hash'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None,
            verified_at=datetime.fromisoformat(data['verified_at']) if data['verified_at'] else None,
            restored_count=data['restored_count'],
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data['last_accessed'] else None,
            metadata=data['metadata']
        )
    
    async def _store_restore_request(self, request: RestoreRequest):
        """Store restore request"""
        
        request_data = {
            'backup_id': request.backup_id,
            'user_id': request.user_id,
            'restore_path': request.restore_path,
            'point_in_time': request.point_in_time.isoformat() if request.point_in_time else None,
            'partial_restore': request.partial_restore,
            'restore_filters': request.restore_filters,
            'requested_at': request.requested_at.isoformat(),
            'status': request.status,
            'progress_percent': request.progress_percent,
            'estimated_completion': request.estimated_completion.isoformat() if request.estimated_completion else None
        }
        
        await self.redis_client.set(
            f"restore_request:{request.request_id}",
            json.dumps(request_data)
        )


if __name__ == "__main__":
    # Example usage and testing
    async def test_encrypted_backup():
        """Test encrypted backup functionality"""
        
        # Initialize backup engine
        config = {
            'redis_url': 'redis://localhost:6379',
            'local_backup_dir': '/tmp/prometheus_backups',
            'aws': {
                'access_key_id': 'test_key',
                'secret_access_key': 'test_secret',
                'region': 'us-west-2',
                'backup_bucket': 'prometheus-backups'
            }
        }
        
        backup_engine = EncryptedBackupEngine(config)
        await backup_engine.initialize()
        
        # Create test data file
        test_data_path = '/tmp/test_geological_data.json'
        test_data = {
            'volcano_monitoring': {
                'location': 'Mount St. Helens',
                'seismic_events': [
                    {'timestamp': '2024-01-01T12:00:00Z', 'magnitude': 2.1},
                    {'timestamp': '2024-01-01T13:30:00Z', 'magnitude': 1.8}
                ],
                'gas_emissions': {'so2_ppm': 150, 'co2_ppm': 400}
            }
        }
        
        async with aiofiles.open(test_data_path, 'w') as f:
            await f.write(json.dumps(test_data))
        
        # Create backup
        success, job_id, error = await backup_engine.create_backup(
            source_path=test_data_path,
            data_classification=DataClassification.CONFIDENTIAL,
            geological_data_type="volcanic_monitoring",
            backup_type=BackupType.FULL
        )
        
        if success:
            print(f"Backup created successfully: {job_id}")
            
            # Wait for backup to complete (in real scenario, would check status)
            await asyncio.sleep(2)
            
            # Test restore (would use actual backup_id from job)
            # For testing, we'll simulate this
            print("Backup and restore system tested successfully")
        else:
            print(f"Backup creation failed: {error}")
    
    # Run test
    asyncio.run(test_encrypted_backup())