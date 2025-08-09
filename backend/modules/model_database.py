#!/usr/bin/env python3
"""
Model Database Management System
================================

Advanced model management system for seismic tomography and geophysical models
with version control, metadata management, and distributed storage capabilities.

Author: Prometheus Platform
Date: 2025-08-08
Version: 2.0.0
"""

import numpy as np
import h5py
import sqlite3
import hashlib
import json
import asyncio
import aiofiles
import aiohttp
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import uuid
from datetime import datetime, timezone
import logging
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import shutil
import zipfile
import tarfile
import urllib.parse
from contextlib import contextmanager
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Text, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
import boto3
from minio import Minio
import paramiko
import requests
import netCDF4 as nc
from scipy.io import savemat, loadmat
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

@dataclass
class ModelMetadata:
    """Comprehensive model metadata structure"""
    id: str
    name: str
    version: str
    description: str
    author: str
    institution: str
    creation_date: datetime
    last_modified: datetime
    model_type: str  # 'seismic_p', 'seismic_s', 'density', 'attenuation', etc.
    region: str      # 'global', 'regional', 'local'
    resolution: Dict[str, float]  # grid spacing in km
    dimensions: Dict[str, int]    # grid dimensions
    coordinate_system: str        # 'geographic', 'cartesian', 'spherical'
    reference_model: Optional[str] = None
    processing_parameters: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    validation_status: str = 'pending'  # 'pending', 'validated', 'rejected'
    tags: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    license: str = 'CC-BY-4.0'
    file_format: str = 'hdf5'
    file_size: int = 0
    checksum: str = ''
    storage_locations: List[str] = field(default_factory=list)

@dataclass
class ModelVersion:
    """Model version information"""
    version: str
    parent_version: Optional[str]
    changes: str
    timestamp: datetime
    author: str
    checksum: str
    file_path: str

class ModelRecord(Base):
    """SQLAlchemy model for database storage"""
    __tablename__ = 'models'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    description = Column(Text)
    author = Column(String)
    institution = Column(String)
    creation_date = Column(DateTime)
    last_modified = Column(DateTime)
    model_type = Column(String)
    region = Column(String)
    resolution = Column(JSONB)
    dimensions = Column(JSONB)
    coordinate_system = Column(String)
    reference_model = Column(String)
    processing_parameters = Column(JSONB)
    quality_metrics = Column(JSONB)
    validation_status = Column(String)
    tags = Column(JSONB)
    doi = Column(String)
    license = Column(String)
    file_format = Column(String)
    file_size = Column(Integer)
    checksum = Column(String)
    storage_locations = Column(JSONB)

class ModelVersionRecord(Base):
    """SQLAlchemy model for version tracking"""
    __tablename__ = 'model_versions'
    
    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False)
    version = Column(String, nullable=False)
    parent_version = Column(String)
    changes = Column(Text)
    timestamp = Column(DateTime)
    author = Column(String)
    checksum = Column(String)
    file_path = Column(String)

class StorageBackend:
    """Abstract base class for storage backends"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    async def upload(self, local_path: str, remote_path: str) -> str:
        """Upload file to storage backend"""
        raise NotImplementedError
    
    async def download(self, remote_path: str, local_path: str) -> bool:
        """Download file from storage backend"""
        raise NotImplementedError
    
    async def delete(self, remote_path: str) -> bool:
        """Delete file from storage backend"""
        raise NotImplementedError
    
    async def exists(self, remote_path: str) -> bool:
        """Check if file exists in storage backend"""
        raise NotImplementedError
    
    async def get_metadata(self, remote_path: str) -> Dict:
        """Get file metadata from storage backend"""
        raise NotImplementedError

class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_path = Path(config.get('base_path', './model_storage'))
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def upload(self, local_path: str, remote_path: str) -> str:
        """Copy file to local storage"""
        full_remote_path = self.base_path / remote_path
        full_remote_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use async file operations
        async with aiofiles.open(local_path, 'rb') as src:
            async with aiofiles.open(full_remote_path, 'wb') as dst:
                while chunk := await src.read(8192):
                    await dst.write(chunk)
        
        return str(full_remote_path)
    
    async def download(self, remote_path: str, local_path: str) -> bool:
        """Copy file from local storage"""
        full_remote_path = self.base_path / remote_path
        
        if not full_remote_path.exists():
            return False
        
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(full_remote_path, 'rb') as src:
            async with aiofiles.open(local_path, 'wb') as dst:
                while chunk := await src.read(8192):
                    await dst.write(chunk)
        
        return True
    
    async def delete(self, remote_path: str) -> bool:
        """Delete file from local storage"""
        full_remote_path = self.base_path / remote_path
        
        try:
            full_remote_path.unlink()
            return True
        except FileNotFoundError:
            return False
    
    async def exists(self, remote_path: str) -> bool:
        """Check if file exists in local storage"""
        full_remote_path = self.base_path / remote_path
        return full_remote_path.exists()
    
    async def get_metadata(self, remote_path: str) -> Dict:
        """Get file metadata from local storage"""
        full_remote_path = self.base_path / remote_path
        
        if not full_remote_path.exists():
            return {}
        
        stat = full_remote_path.stat()
        return {
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            'created': datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        }

class S3StorageBackend(StorageBackend):
    """Amazon S3 storage backend"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.get('access_key'),
            aws_secret_access_key=config.get('secret_key'),
            region_name=config.get('region', 'us-east-1')
        )
        self.bucket = config.get('bucket')
    
    async def upload(self, local_path: str, remote_path: str) -> str:
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(local_path, self.bucket, remote_path)
            return f"s3://{self.bucket}/{remote_path}"
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    async def download(self, remote_path: str, local_path: str) -> bool:
        """Download file from S3"""
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket, remote_path, local_path)
            return True
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False
    
    async def delete(self, remote_path: str) -> bool:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=remote_path)
            return True
        except Exception as e:
            logger.error(f"S3 delete failed: {e}")
            return False
    
    async def exists(self, remote_path: str) -> bool:
        """Check if file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except:
            return False
    
    async def get_metadata(self, remote_path: str) -> Dict:
        """Get file metadata from S3"""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=remote_path)
            return {
                'size': response.get('ContentLength', 0),
                'modified': response.get('LastModified'),
                'etag': response.get('ETag', '').strip('"')
            }
        except:
            return {}

class MinIOStorageBackend(StorageBackend):
    """MinIO object storage backend"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.client = Minio(
            config.get('endpoint'),
            access_key=config.get('access_key'),
            secret_key=config.get('secret_key'),
            secure=config.get('secure', True)
        )
        self.bucket = config.get('bucket')
        
        # Ensure bucket exists
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
    
    async def upload(self, local_path: str, remote_path: str) -> str:
        """Upload file to MinIO"""
        try:
            self.client.fput_object(self.bucket, remote_path, local_path)
            return f"minio://{self.bucket}/{remote_path}"
        except Exception as e:
            logger.error(f"MinIO upload failed: {e}")
            raise
    
    async def download(self, remote_path: str, local_path: str) -> bool:
        """Download file from MinIO"""
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.client.fget_object(self.bucket, remote_path, local_path)
            return True
        except Exception as e:
            logger.error(f"MinIO download failed: {e}")
            return False
    
    async def delete(self, remote_path: str) -> bool:
        """Delete file from MinIO"""
        try:
            self.client.remove_object(self.bucket, remote_path)
            return True
        except Exception as e:
            logger.error(f"MinIO delete failed: {e}")
            return False
    
    async def exists(self, remote_path: str) -> bool:
        """Check if file exists in MinIO"""
        try:
            self.client.stat_object(self.bucket, remote_path)
            return True
        except:
            return False
    
    async def get_metadata(self, remote_path: str) -> Dict:
        """Get file metadata from MinIO"""
        try:
            stat = self.client.stat_object(self.bucket, remote_path)
            return {
                'size': stat.size,
                'modified': stat.last_modified,
                'etag': stat.etag
            }
        except:
            return {}

class ModelDatabase:
    """Advanced model database management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_database()
        self.setup_storage_backends()
        self.setup_cache()
        
        # Thread pools for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        
        logger.info("Model Database initialized successfully")
    
    def setup_database(self):
        """Initialize database connection"""
        db_config = self.config.get('database', {})
        db_url = db_config.get('url', 'sqlite:///models.db')
        
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        logger.info(f"Database initialized: {db_url}")
    
    def setup_storage_backends(self):
        """Initialize storage backends"""
        self.storage_backends = {}
        
        storage_config = self.config.get('storage', {})
        
        # Local storage (always available)
        if 'local' in storage_config:
            self.storage_backends['local'] = LocalStorageBackend(storage_config['local'])
        
        # S3 storage
        if 's3' in storage_config:
            self.storage_backends['s3'] = S3StorageBackend(storage_config['s3'])
        
        # MinIO storage
        if 'minio' in storage_config:
            self.storage_backends['minio'] = MinIOStorageBackend(storage_config['minio'])
        
        # Default to local storage if none configured
        if not self.storage_backends:
            self.storage_backends['local'] = LocalStorageBackend({'base_path': './model_storage'})
        
        logger.info(f"Storage backends initialized: {list(self.storage_backends.keys())}")
    
    def setup_cache(self):
        """Initialize caching system"""
        cache_config = self.config.get('cache', {})
        
        if cache_config.get('type') == 'redis':
            self.cache = redis.Redis(
                host=cache_config.get('host', 'localhost'),
                port=cache_config.get('port', 6379),
                db=cache_config.get('db', 0)
            )
        else:
            # In-memory cache fallback
            self.cache = {}
        
        logger.info("Cache system initialized")
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    def calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def register_model(self, 
                           model_path: str,
                           metadata: ModelMetadata,
                           storage_backends: List[str] = None) -> str:
        """Register a new model in the database"""
        
        # Validate file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Calculate checksum
        metadata.checksum = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool, self.calculate_checksum, model_path
        )
        metadata.file_size = Path(model_path).stat().st_size
        
        # Generate unique ID if not provided
        if not metadata.id:
            metadata.id = str(uuid.uuid4())
        
        # Set timestamps
        now = datetime.now(timezone.utc)
        metadata.creation_date = now
        metadata.last_modified = now
        
        # Use default storage backends if not specified
        if storage_backends is None:
            storage_backends = list(self.storage_backends.keys())
        
        # Upload to storage backends
        storage_locations = []
        for backend_name in storage_backends:
            if backend_name in self.storage_backends:
                backend = self.storage_backends[backend_name]
                remote_path = f"{metadata.model_type}/{metadata.region}/{metadata.id}/{metadata.version}.{metadata.file_format}"
                
                try:
                    location = await backend.upload(model_path, remote_path)
                    storage_locations.append(location)
                    logger.info(f"Model uploaded to {backend_name}: {location}")
                except Exception as e:
                    logger.error(f"Failed to upload to {backend_name}: {e}")
        
        metadata.storage_locations = storage_locations
        
        # Store in database
        with self.get_session() as session:
            # Check if model already exists
            existing = session.query(ModelRecord).filter_by(
                name=metadata.name, version=metadata.version
            ).first()
            
            if existing:
                raise ValueError(f"Model {metadata.name} version {metadata.version} already exists")
            
            # Create database record
            model_record = ModelRecord(**asdict(metadata))
            session.add(model_record)
            
            # Create version record
            version_record = ModelVersionRecord(
                id=str(uuid.uuid4()),
                model_id=metadata.id,
                version=metadata.version,
                parent_version=None,
                changes="Initial version",
                timestamp=now,
                author=metadata.author,
                checksum=metadata.checksum,
                file_path=storage_locations[0] if storage_locations else model_path
            )
            session.add(version_record)
            
            session.commit()
        
        # Cache model metadata
        await self.cache_metadata(metadata.id, metadata)
        
        logger.info(f"Model registered successfully: {metadata.id}")
        return metadata.id
    
    async def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata"""
        
        # Check cache first
        cached = await self.get_cached_metadata(model_id)
        if cached:
            return cached
        
        # Query database
        with self.get_session() as session:
            record = session.query(ModelRecord).filter_by(id=model_id).first()
            
            if not record:
                return None
            
            # Convert to dataclass
            metadata = ModelMetadata(**{
                column.name: getattr(record, column.name)
                for column in ModelRecord.__table__.columns
            })
            
            # Cache for future use
            await self.cache_metadata(model_id, metadata)
            
            return metadata
    
    async def search_models(self, 
                          query: Dict[str, Any],
                          limit: int = 100,
                          offset: int = 0) -> List[ModelMetadata]:
        """Search models with flexible query parameters"""
        
        with self.get_session() as session:
            db_query = session.query(ModelRecord)
            
            # Apply filters
            if 'name' in query:
                db_query = db_query.filter(ModelRecord.name.like(f"%{query['name']}%"))
            
            if 'model_type' in query:
                db_query = db_query.filter(ModelRecord.model_type == query['model_type'])
            
            if 'region' in query:
                db_query = db_query.filter(ModelRecord.region == query['region'])
            
            if 'author' in query:
                db_query = db_query.filter(ModelRecord.author.like(f"%{query['author']}%"))
            
            if 'tags' in query:
                # Search for models containing any of the specified tags
                tags = query['tags'] if isinstance(query['tags'], list) else [query['tags']]
                db_query = db_query.filter(ModelRecord.tags.op('&&')(tags))
            
            if 'created_after' in query:
                db_query = db_query.filter(ModelRecord.creation_date >= query['created_after'])
            
            if 'created_before' in query:
                db_query = db_query.filter(ModelRecord.creation_date <= query['created_before'])
            
            # Apply sorting
            sort_by = query.get('sort_by', 'creation_date')
            sort_order = query.get('sort_order', 'desc')
            
            if hasattr(ModelRecord, sort_by):
                column = getattr(ModelRecord, sort_by)
                if sort_order.lower() == 'desc':
                    db_query = db_query.order_by(column.desc())
                else:
                    db_query = db_query.order_by(column.asc())
            
            # Apply pagination
            records = db_query.offset(offset).limit(limit).all()
            
            # Convert to dataclasses
            models = []
            for record in records:
                metadata = ModelMetadata(**{
                    column.name: getattr(record, column.name)
                    for column in ModelRecord.__table__.columns
                })
                models.append(metadata)
            
            return models
    
    async def download_model(self, 
                           model_id: str,
                           version: str = None,
                           local_path: str = None,
                           preferred_backend: str = None) -> str:
        """Download model data to local filesystem"""
        
        # Get model metadata
        metadata = await self.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")
        
        # If version is specified and different from current, get version-specific metadata
        if version and version != metadata.version:
            version_metadata = await self.get_model_version(model_id, version)
            if not version_metadata:
                raise ValueError(f"Version {version} of model {model_id} not found")
            # Use version-specific file path
            # This would require additional logic to handle version-specific storage
        
        # Determine local path
        if not local_path:
            local_path = f"./downloads/{metadata.name}_{metadata.version}.{metadata.file_format}"
        
        # Try to download from preferred backend first
        storage_locations = metadata.storage_locations.copy()
        if preferred_backend:
            # Move preferred backend to front of list
            preferred_locations = [loc for loc in storage_locations 
                                 if loc.startswith(f"{preferred_backend}://")]
            other_locations = [loc for loc in storage_locations 
                             if not loc.startswith(f"{preferred_backend}://")]
            storage_locations = preferred_locations + other_locations
        
        # Try each storage location until successful
        for location in storage_locations:
            try:
                backend_name = location.split('://')[0]
                if backend_name in self.storage_backends:
                    backend = self.storage_backends[backend_name]
                    
                    # Extract remote path from location
                    if '://' in location:
                        remote_path = location.split('/', 3)[-1]  # Remove protocol and bucket/host
                    else:
                        remote_path = location
                    
                    success = await backend.download(remote_path, local_path)
                    if success:
                        # Verify checksum
                        downloaded_checksum = await asyncio.get_event_loop().run_in_executor(
                            self.thread_pool, self.calculate_checksum, local_path
                        )
                        
                        if downloaded_checksum == metadata.checksum:
                            logger.info(f"Model {model_id} downloaded successfully from {backend_name}")
                            return local_path
                        else:
                            logger.warning(f"Checksum mismatch for model {model_id} from {backend_name}")
                            Path(local_path).unlink(missing_ok=True)
                            continue
                    
            except Exception as e:
                logger.error(f"Failed to download from {location}: {e}")
                continue
        
        raise RuntimeError(f"Failed to download model {model_id} from any storage location")
    
    async def create_model_version(self,
                                 model_id: str,
                                 new_model_path: str,
                                 version: str,
                                 changes: str,
                                 author: str) -> str:
        """Create a new version of an existing model"""
        
        # Get current model metadata
        current_metadata = await self.get_model_metadata(model_id)
        if not current_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        # Calculate checksum of new model
        new_checksum = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool, self.calculate_checksum, new_model_path
        )
        
        # Create new metadata with updated version
        new_metadata = ModelMetadata(**asdict(current_metadata))
        new_metadata.version = version
        new_metadata.last_modified = datetime.now(timezone.utc)
        new_metadata.checksum = new_checksum
        new_metadata.file_size = Path(new_model_path).stat().st_size
        
        # Upload to storage backends
        storage_locations = []
        for backend_name in self.storage_backends:
            backend = self.storage_backends[backend_name]
            remote_path = f"{new_metadata.model_type}/{new_metadata.region}/{model_id}/{version}.{new_metadata.file_format}"
            
            try:
                location = await backend.upload(new_model_path, remote_path)
                storage_locations.append(location)
            except Exception as e:
                logger.error(f"Failed to upload version to {backend_name}: {e}")
        
        new_metadata.storage_locations = storage_locations
        
        # Update database
        with self.get_session() as session:
            # Update main model record
            model_record = session.query(ModelRecord).filter_by(id=model_id).first()
            if model_record:
                model_record.version = version
                model_record.last_modified = new_metadata.last_modified
                model_record.checksum = new_checksum
                model_record.file_size = new_metadata.file_size
                model_record.storage_locations = storage_locations
            
            # Create version record
            version_record = ModelVersionRecord(
                id=str(uuid.uuid4()),
                model_id=model_id,
                version=version,
                parent_version=current_metadata.version,
                changes=changes,
                timestamp=new_metadata.last_modified,
                author=author,
                checksum=new_checksum,
                file_path=storage_locations[0] if storage_locations else new_model_path
            )
            session.add(version_record)
            
            session.commit()
        
        # Update cache
        await self.cache_metadata(model_id, new_metadata)
        
        logger.info(f"New version {version} created for model {model_id}")
        return version
    
    async def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        
        with self.get_session() as session:
            version_records = session.query(ModelVersionRecord)\
                                   .filter_by(model_id=model_id)\
                                   .order_by(ModelVersionRecord.timestamp.desc())\
                                   .all()
            
            versions = []
            for record in version_records:
                version = ModelVersion(
                    version=record.version,
                    parent_version=record.parent_version,
                    changes=record.changes,
                    timestamp=record.timestamp,
                    author=record.author,
                    checksum=record.checksum,
                    file_path=record.file_path
                )
                versions.append(version)
            
            return versions
    
    async def get_model_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get specific version of a model"""
        
        with self.get_session() as session:
            record = session.query(ModelVersionRecord)\
                           .filter_by(model_id=model_id, version=version)\
                           .first()
            
            if not record:
                return None
            
            return ModelVersion(
                version=record.version,
                parent_version=record.parent_version,
                changes=record.changes,
                timestamp=record.timestamp,
                author=record.author,
                checksum=record.checksum,
                file_path=record.file_path
            )
    
    async def validate_model(self, model_id: str, validation_results: Dict[str, Any]) -> bool:
        """Update model validation status"""
        
        with self.get_session() as session:
            model_record = session.query(ModelRecord).filter_by(id=model_id).first()
            
            if not model_record:
                return False
            
            # Update validation status
            model_record.validation_status = validation_results.get('status', 'validated')
            
            # Update quality metrics
            if 'quality_metrics' in validation_results:
                model_record.quality_metrics = validation_results['quality_metrics']
            
            session.commit()
        
        # Update cache
        await self.invalidate_cache(model_id)
        
        logger.info(f"Model {model_id} validation updated: {validation_results.get('status')}")
        return True
    
    async def delete_model(self, model_id: str, version: str = None) -> bool:
        """Delete model or specific version"""
        
        metadata = await self.get_model_metadata(model_id)
        if not metadata:
            return False
        
        if version:
            # Delete specific version
            with self.get_session() as session:
                version_record = session.query(ModelVersionRecord)\
                               .filter_by(model_id=model_id, version=version)\
                               .first()
                
                if version_record:
                    # Delete from storage backends
                    for location in metadata.storage_locations:
                        try:
                            backend_name = location.split('://')[0]
                            if backend_name in self.storage_backends:
                                backend = self.storage_backends[backend_name]
                                remote_path = f"{metadata.model_type}/{metadata.region}/{model_id}/{version}.{metadata.file_format}"
                                await backend.delete(remote_path)
                        except Exception as e:
                            logger.error(f"Failed to delete from {backend_name}: {e}")
                    
                    session.delete(version_record)
                    session.commit()
                    
                    logger.info(f"Model version {version} deleted for {model_id}")
                    return True
        
        else:
            # Delete entire model
            with self.get_session() as session:
                # Delete all versions first
                session.query(ModelVersionRecord).filter_by(model_id=model_id).delete()
                
                # Delete main model record
                model_record = session.query(ModelRecord).filter_by(id=model_id).first()
                if model_record:
                    # Delete from all storage backends
                    for location in metadata.storage_locations:
                        try:
                            backend_name = location.split('://')[0]
                            if backend_name in self.storage_backends:
                                backend = self.storage_backends[backend_name]
                                # Delete entire model directory
                                model_dir = f"{metadata.model_type}/{metadata.region}/{model_id}/"
                                await backend.delete(model_dir)
                        except Exception as e:
                            logger.error(f"Failed to delete from {backend_name}: {e}")
                    
                    session.delete(model_record)
                    session.commit()
                    
                    # Remove from cache
                    await self.invalidate_cache(model_id)
                    
                    logger.info(f"Model {model_id} deleted completely")
                    return True
        
        return False
    
    async def export_model_catalog(self, output_path: str, format: str = 'json') -> bool:
        """Export model catalog to file"""
        
        try:
            # Get all models
            all_models = await self.search_models({}, limit=10000)
            
            if format.lower() == 'json':
                catalog_data = {
                    'export_date': datetime.now(timezone.utc).isoformat(),
                    'total_models': len(all_models),
                    'models': [asdict(model) for model in all_models]
                }
                
                with open(output_path, 'w') as f:
                    json.dump(catalog_data, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                import pandas as pd
                
                # Convert to DataFrame
                df_data = []
                for model in all_models:
                    model_dict = asdict(model)
                    # Flatten complex fields
                    model_dict['tags'] = ','.join(model_dict.get('tags', []))
                    model_dict['storage_locations'] = ','.join(model_dict.get('storage_locations', []))
                    df_data.append(model_dict)
                
                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)
            
            elif format.lower() == 'xml':
                import xml.etree.ElementTree as ET
                
                root = ET.Element('model_catalog')
                root.set('export_date', datetime.now(timezone.utc).isoformat())
                root.set('total_models', str(len(all_models)))
                
                for model in all_models:
                    model_elem = ET.SubElement(root, 'model')
                    model_dict = asdict(model)
                    
                    for key, value in model_dict.items():
                        if value is not None:
                            elem = ET.SubElement(model_elem, key)
                            elem.text = str(value)
                
                tree = ET.ElementTree(root)
                tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            logger.info(f"Model catalog exported to {output_path} ({format} format)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export model catalog: {e}")
            return False
    
    async def import_model_catalog(self, catalog_path: str, format: str = 'json') -> int:
        """Import models from catalog file"""
        
        try:
            imported_count = 0
            
            if format.lower() == 'json':
                with open(catalog_path, 'r') as f:
                    catalog_data = json.load(f)
                
                for model_data in catalog_data.get('models', []):
                    try:
                        # Convert date strings back to datetime objects
                        if 'creation_date' in model_data:
                            model_data['creation_date'] = datetime.fromisoformat(
                                model_data['creation_date'].replace('Z', '+00:00')
                            )
                        if 'last_modified' in model_data:
                            model_data['last_modified'] = datetime.fromisoformat(
                                model_data['last_modified'].replace('Z', '+00:00')
                            )
                        
                        metadata = ModelMetadata(**model_data)
                        
                        # Register model (without file upload since it's metadata only)
                        with self.get_session() as session:
                            existing = session.query(ModelRecord)\
                                            .filter_by(id=metadata.id)\
                                            .first()
                            
                            if not existing:
                                model_record = ModelRecord(**asdict(metadata))
                                session.add(model_record)
                                session.commit()
                                imported_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to import model {model_data.get('id', 'unknown')}: {e}")
                        continue
            
            logger.info(f"Imported {imported_count} models from catalog")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to import model catalog: {e}")
            return 0
    
    async def sync_with_remote(self, remote_config: Dict[str, str]) -> bool:
        """Synchronize with remote model database"""
        
        try:
            remote_url = remote_config.get('url')
            api_key = remote_config.get('api_key')
            
            # Fetch remote catalog
            headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{remote_url}/api/models", headers=headers) as response:
                    if response.status == 200:
                        remote_models = await response.json()
                        
                        sync_count = 0
                        for model_data in remote_models.get('models', []):
                            # Check if model exists locally
                            local_model = await self.get_model_metadata(model_data.get('id'))
                            
                            if not local_model or local_model.last_modified < \
                               datetime.fromisoformat(model_data.get('last_modified')):
                                
                                # Download and register model
                                # This would require additional API endpoints for file download
                                logger.info(f"Syncing model: {model_data.get('name')}")
                                sync_count += 1
                        
                        logger.info(f"Synchronized {sync_count} models from remote")
                        return True
                    else:
                        logger.error(f"Failed to fetch remote catalog: {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Sync with remote failed: {e}")
            return False
    
    async def cache_metadata(self, model_id: str, metadata: ModelMetadata):
        """Cache model metadata"""
        cache_key = f"model_metadata:{model_id}"
        
        if hasattr(self.cache, 'set'):
            # Redis cache
            metadata_json = json.dumps(asdict(metadata), default=str)
            self.cache.setex(cache_key, 3600, metadata_json)  # Cache for 1 hour
        else:
            # In-memory cache
            self.cache[cache_key] = metadata
    
    async def get_cached_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieve cached model metadata"""
        cache_key = f"model_metadata:{model_id}"
        
        if hasattr(self.cache, 'get'):
            # Redis cache
            cached_data = self.cache.get(cache_key)
            if cached_data:
                metadata_dict = json.loads(cached_data)
                # Convert date strings back to datetime objects
                for date_field in ['creation_date', 'last_modified']:
                    if date_field in metadata_dict and metadata_dict[date_field]:
                        metadata_dict[date_field] = datetime.fromisoformat(
                            metadata_dict[date_field].replace('Z', '+00:00')
                        )
                return ModelMetadata(**metadata_dict)
        else:
            # In-memory cache
            return self.cache.get(cache_key)
        
        return None
    
    async def invalidate_cache(self, model_id: str):
        """Invalidate cached model metadata"""
        cache_key = f"model_metadata:{model_id}"
        
        if hasattr(self.cache, 'delete'):
            # Redis cache
            self.cache.delete(cache_key)
        else:
            # In-memory cache
            self.cache.pop(cache_key, None)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        with self.get_session() as session:
            total_models = session.query(ModelRecord).count()
            
            # Statistics by model type
            type_stats = session.query(
                ModelRecord.model_type,
                session.query(ModelRecord).filter_by(model_type=ModelRecord.model_type).count()
            ).distinct().all()
            
            # Statistics by region
            region_stats = session.query(
                ModelRecord.region,
                session.query(ModelRecord).filter_by(region=ModelRecord.region).count()
            ).distinct().all()
            
            # Storage usage
            total_size = session.query(session.query(ModelRecord.file_size).label('total')).scalar() or 0
            
            # Recent activity
            recent_models = session.query(ModelRecord)\
                                 .order_by(ModelRecord.creation_date.desc())\
                                 .limit(5)\
                                 .all()
            
            return {
                'total_models': total_models,
                'type_distribution': dict(type_stats),
                'region_distribution': dict(region_stats),
                'total_storage_bytes': total_size,
                'total_storage_gb': round(total_size / (1024**3), 2),
                'recent_models': [
                    {'name': m.name, 'version': m.version, 'created': m.creation_date}
                    for m in recent_models
                ]
            }
    
    def close(self):
        """Clean up database connections and resources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)
        
        if hasattr(self.cache, 'close'):
            self.cache.close()
        
        logger.info("Model Database closed")

class ModelValidationService:
    """Service for validating model integrity and quality"""
    
    def __init__(self, database: ModelDatabase):
        self.database = database
        
    async def validate_model_file(self, file_path: str, expected_format: str) -> Dict[str, Any]:
        """Validate model file format and structure"""
        
        validation_results = {
            'status': 'pending',
            'errors': [],
            'warnings': [],
            'quality_metrics': {}
        }
        
        try:
            if expected_format == 'hdf5':
                validation_results.update(await self._validate_hdf5(file_path))
            elif expected_format == 'netcdf':
                validation_results.update(await self._validate_netcdf(file_path))
            elif expected_format == 'binary':
                validation_results.update(await self._validate_binary(file_path))
            else:
                validation_results['errors'].append(f"Unsupported format: {expected_format}")
                validation_results['status'] = 'rejected'
        
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['status'] = 'rejected'
        
        # Overall status
        if not validation_results['errors']:
            validation_results['status'] = 'validated' if not validation_results['warnings'] else 'validated_with_warnings'
        else:
            validation_results['status'] = 'rejected'
        
        return validation_results
    
    async def _validate_hdf5(self, file_path: str) -> Dict[str, Any]:
        """Validate HDF5 model file"""
        results = {'errors': [], 'warnings': [], 'quality_metrics': {}}
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Check required datasets
                required_datasets = ['velocity_p', 'velocity_s', 'density']
                
                for dataset_name in required_datasets:
                    if dataset_name not in f:
                        results['errors'].append(f"Missing required dataset: {dataset_name}")
                    else:
                        dataset = f[dataset_name]
                        
                        # Check data validity
                        data = dataset[:]
                        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                            results['warnings'].append(f"Dataset {dataset_name} contains NaN or Inf values")
                        
                        # Quality metrics
                        results['quality_metrics'][f'{dataset_name}_min'] = float(np.min(data))
                        results['quality_metrics'][f'{dataset_name}_max'] = float(np.max(data))
                        results['quality_metrics'][f'{dataset_name}_mean'] = float(np.mean(data))
                        results['quality_metrics'][f'{dataset_name}_std'] = float(np.std(data))
                
                # Check coordinate arrays
                coord_arrays = ['longitude', 'latitude', 'depth']
                for coord_name in coord_arrays:
                    if coord_name in f:
                        coord_data = f[coord_name][:]
                        if not np.all(np.diff(coord_data) > 0):
                            results['warnings'].append(f"Coordinate array {coord_name} is not monotonically increasing")
        
        except Exception as e:
            results['errors'].append(f"HDF5 validation error: {str(e)}")
        
        return results
    
    async def _validate_netcdf(self, file_path: str) -> Dict[str, Any]:
        """Validate NetCDF model file"""
        results = {'errors': [], 'warnings': [], 'quality_metrics': {}}
        
        try:
            with nc.Dataset(file_path, 'r') as ds:
                # Check required variables
                required_vars = ['velocity_p', 'velocity_s', 'density']
                
                for var_name in required_vars:
                    if var_name not in ds.variables:
                        results['errors'].append(f"Missing required variable: {var_name}")
                    else:
                        var = ds.variables[var_name]
                        data = var[:]
                        
                        # Check data validity
                        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                            results['warnings'].append(f"Variable {var_name} contains NaN or Inf values")
                        
                        # Quality metrics
                        results['quality_metrics'][f'{var_name}_min'] = float(np.min(data))
                        results['quality_metrics'][f'{var_name}_max'] = float(np.max(data))
                        results['quality_metrics'][f'{var_name}_mean'] = float(np.mean(data))
                        results['quality_metrics'][f'{var_name}_std'] = float(np.std(data))
                
                # Check global attributes
                required_attrs = ['title', 'institution', 'source']
                for attr_name in required_attrs:
                    if not hasattr(ds, attr_name):
                        results['warnings'].append(f"Missing recommended attribute: {attr_name}")
        
        except Exception as e:
            results['errors'].append(f"NetCDF validation error: {str(e)}")
        
        return results
    
    async def _validate_binary(self, file_path: str) -> Dict[str, Any]:
        """Validate binary model file"""
        results = {'errors': [], 'warnings': [], 'quality_metrics': {}}
        
        try:
            # Basic file size and readability check
            file_size = Path(file_path).stat().st_size
            
            if file_size == 0:
                results['errors'].append("File is empty")
            else:
                results['quality_metrics']['file_size_bytes'] = file_size
                
                # Try to read first few bytes to check format
                with open(file_path, 'rb') as f:
                    header = f.read(64)
                    
                    if len(header) < 64:
                        results['warnings'].append("File is smaller than expected header size")
        
        except Exception as e:
            results['errors'].append(f"Binary validation error: {str(e)}")
        
        return results

# Main execution and utility functions
async def main():
    """Main execution function for demonstration"""
    
    # Configuration
    config = {
        'database': {
            'url': 'sqlite:///models.db'
        },
        'storage': {
            'local': {
                'base_path': './model_storage'
            }
        },
        'cache': {
            'type': 'memory'
        }
    }
    
    # Initialize database
    db = ModelDatabase(config)
    
    # Example usage
    try:
        # Create example model metadata
        metadata = ModelMetadata(
            id=str(uuid.uuid4()),
            name="Global_P_Wave_Tomography",
            version="1.0",
            description="Global P-wave velocity model from full waveform inversion",
            author="Prometheus Platform",
            institution="Advanced Seismology Institute",
            creation_date=datetime.now(timezone.utc),
            last_modified=datetime.now(timezone.utc),
            model_type="seismic_p",
            region="global",
            resolution={"horizontal": 1.0, "vertical": 25.0},
            dimensions={"nx": 360, "ny": 180, "nz": 50},
            coordinate_system="geographic",
            tags=["tomography", "global", "p-wave", "fwi"]
        )
        
        # Get database statistics
        stats = await db.get_statistics()
        logger.info(f"Database statistics: {stats}")
        
        # Search models
        search_results = await db.search_models({"model_type": "seismic_p"})
        logger.info(f"Found {len(search_results)} P-wave models")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
    
    finally:
        db.close()
    
    logger.info("Model Database demonstration completed")

if __name__ == "__main__":
    asyncio.run(main())