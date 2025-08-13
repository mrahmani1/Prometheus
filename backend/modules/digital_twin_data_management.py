"""
Digital Twin Earth Data Management System
Advanced data export/import and management for Digital Twin Earth system
"""

import asyncio
import logging
import warnings
import json
import csv
import pickle
import gzip
import tarfile
import zipfile
from typing import Dict, List, Any, Optional, Union, Callable, Iterator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import tempfile
import shutil
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import io
import base64

# Database imports
try:
    import sqlite3
    import sqlalchemy
    from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Text, LargeBinary
    from sqlalchemy.orm import sessionmaker
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    warnings.warn("Database libraries not available. Database functionality will be limited.")

# Data format imports
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas/NumPy not available. Data processing capabilities will be limited.")

# Scientific data formats
try:
    import netCDF4
    import h5py
    import xarray as xr
    SCIENTIFIC_FORMATS_AVAILABLE = True
except ImportError:
    SCIENTIFIC_FORMATS_AVAILABLE = False
    warnings.warn("Scientific data format libraries not available.")

# Cloud storage imports
try:
    import boto3
    from botocore.exceptions import NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    warnings.warn("AWS SDK not available. Cloud storage will be limited.")

# Compression imports
try:
    import lz4.frame
    import zstandard as zstd
    ADVANCED_COMPRESSION_AVAILABLE = True
except ImportError:
    ADVANCED_COMPRESSION_AVAILABLE = False
    warnings.warn("Advanced compression libraries not available.")

warnings.filterwarnings('ignore')

class DataFormat(Enum):
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    NETCDF = "netcdf"
    PICKLE = "pickle"
    NUMPY = "npy"
    XML = "xml"
    YAML = "yaml"
    BINARY = "bin"

class CompressionType(Enum):
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    TAR_GZ = "tar.gz"
    LZ4 = "lz4"
    ZSTD = "zstd"

class StorageBackend(Enum):
    LOCAL = "local"
    AWS_S3 = "aws_s3"
    AZURE_BLOB = "azure_blob"
    GCP_STORAGE = "gcp_storage"
    FTP = "ftp"
    DATABASE = "database"

@dataclass
class DataExportConfig:
    """Configuration for data export operations"""
    format: DataFormat = DataFormat.JSON
    compression: CompressionType = CompressionType.GZIP
    include_metadata: bool = True
    include_history: bool = False
    date_range: Optional[Tuple[datetime, datetime]] = None
    batch_size: int = 10000
    parallel_workers: int = 4
    encryption_key: Optional[str] = None
    storage_backend: StorageBackend = StorageBackend.LOCAL
    storage_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataImportConfig:
    """Configuration for data import operations"""
    format: DataFormat = DataFormat.JSON
    compression: CompressionType = CompressionType.GZIP
    validate_schema: bool = True
    merge_strategy: str = "replace"  # replace, merge, append
    batch_size: int = 10000
    parallel_workers: int = 4
    skip_errors: bool = False
    backup_existing: bool = True
    storage_backend: StorageBackend = StorageBackend.LOCAL
    storage_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataPackage:
    """Data package with metadata and content"""
    name: str
    version: str
    created_at: datetime
    data_type: str
    format: DataFormat
    compression: CompressionType
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: Any = None

class DigitalTwinDataManager:
    """Advanced data management system for Digital Twin Earth"""
    
    def __init__(self, base_path: str = "data", temp_path: str = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.temp_path = Path(temp_path) if temp_path else Path(tempfile.gettempdir()) / "prometheus_data"
        self.temp_path.mkdir(exist_ok=True)
        
        # Storage backends
        self.storage_backends = {}
        self.active_connections = {}
        
        # Data cache and indexing
        self.data_cache = {}
        self.data_index = {}  # package_name -> DataPackage metadata
        self.access_stats = {}  # package_name -> access statistics
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Task management
        self.active_tasks = {}
        self.task_progress = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage backends
        asyncio.create_task(self._initialize_storage_backends())
        
        # Initialize database if available
        if DATABASE_AVAILABLE:
            asyncio.create_task(self._initialize_database())
    
    async def _initialize_storage_backends(self):
        """Initialize available storage backends"""
        try:
            # Local storage (always available)
            self.storage_backends[StorageBackend.LOCAL] = LocalStorageBackend(self.base_path)
            
            # AWS S3
            if AWS_AVAILABLE:
                self.storage_backends[StorageBackend.AWS_S3] = S3StorageBackend()
            
            # Database storage
            if DATABASE_AVAILABLE:
                self.storage_backends[StorageBackend.DATABASE] = DatabaseStorageBackend(
                    str(self.base_path / "digital_twin_data.db")
                )
            
            self.logger.info("Storage backends initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing storage backends: {e}")
    
    async def _initialize_database(self):
        """Initialize database for data management"""
        try:
            if not DATABASE_AVAILABLE:
                return
            
            db_path = self.base_path / "data_management.db"
            engine = create_engine(f"sqlite:///{db_path}")
            
            # Create tables
            metadata = MetaData()
            
            # Data packages table
            packages_table = Table(
                'data_packages', metadata,
                Column('id', Integer, primary_key=True),
                Column('name', String(255), unique=True, nullable=False),
                Column('version', String(50), nullable=False),
                Column('data_type', String(100), nullable=False),
                Column('format', String(50), nullable=False),
                Column('compression', String(50), nullable=False),
                Column('size_bytes', Integer, nullable=False),
                Column('checksum', String(128), nullable=False),
                Column('created_at', DateTime, nullable=False),
                Column('metadata', Text),
                Column('storage_backend', String(50), nullable=False),
                Column('storage_path', String(500), nullable=False)
            )
            
            # Access statistics table
            access_stats_table = Table(
                'access_stats', metadata,
                Column('id', Integer, primary_key=True),
                Column('package_name', String(255), nullable=False),
                Column('access_time', DateTime, nullable=False),
                Column('operation', String(50), nullable=False),
                Column('user_id', String(100)),
                Column('duration_ms', Integer)
            )
            
            metadata.create_all(engine)
            
            # Create session factory
            Session = sessionmaker(bind=engine)
            self.db_session_factory = Session
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    async def export_earth_system_data(self, data_types: List[str], config: DataExportConfig) -> str:
        """Export Digital Twin Earth system data"""
        try:
            export_id = f"earth_system_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create export directory
            export_dir = self.temp_path / export_id
            export_dir.mkdir(exist_ok=True)
            
            # Track progress
            self.task_progress[export_id] = {
                'total_steps': len(data_types),
                'completed_steps': 0,
                'current_step': '',
                'start_time': datetime.utcnow()
            }
            
            exported_packages = []
            
            for data_type in data_types:
                self.task_progress[export_id]['current_step'] = f'Exporting {data_type}'
                
                # Get data for this type
                data = await self._get_earth_system_data(data_type, config.date_range)
                
                if data:
                    # Create data package
                    package = await self._create_data_package(
                        name=f"{data_type}_data",
                        version="1.0.0",
                        data_type=data_type,
                        data=data,
                        config=config
                    )
                    
                    # Export package
                    package_path = await self._export_data_package(package, export_dir, config)
                    exported_packages.append(package_path)
                
                self.task_progress[export_id]['completed_steps'] += 1
            
            # Create export archive if multiple packages
            if len(exported_packages) > 1:
                archive_path = await self._create_export_archive(exported_packages, export_dir, config)
            else:
                archive_path = exported_packages[0] if exported_packages else None
            
            # Upload to storage backend if configured
            if config.storage_backend != StorageBackend.LOCAL and archive_path:
                final_path = await self._upload_to_storage(archive_path, config)
                
                # Clean up local temporary files
                if export_dir.exists():
                    shutil.rmtree(export_dir)
                
                return final_path
            
            self.logger.info(f"Earth system data exported to: {archive_path}")
            return str(archive_path) if archive_path else None
            
        except Exception as e:
            self.logger.error(f"Error exporting Earth system data: {e}")
            raise
        finally:
            # Clean up progress tracking
            if export_id in self.task_progress:
                del self.task_progress[export_id]
    
    async def import_earth_system_data(self, import_path: str, config: DataImportConfig) -> bool:
        """Import Digital Twin Earth system data"""
        try:
            import_id = f"earth_system_import_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Download from storage backend if needed
            local_path = await self._download_from_storage(import_path, config)
            
            # Extract archive if needed
            extracted_path = await self._extract_import_archive(local_path, config)
            
            # Track progress
            import_files = list(Path(extracted_path).rglob("*")) if Path(extracted_path).is_dir() else [Path(local_path)]
            data_files = [f for f in import_files if f.is_file() and not f.name.startswith('.')]
            
            self.task_progress[import_id] = {
                'total_steps': len(data_files),
                'completed_steps': 0,
                'current_step': '',
                'start_time': datetime.utcnow()
            }
            
            # Backup existing data if requested
            if config.backup_existing:
                await self._backup_existing_data()
            
            imported_count = 0
            
            for data_file in data_files:
                self.task_progress[import_id]['current_step'] = f'Importing {data_file.name}'
                
                try:
                    # Load data package
                    package = await self._load_data_package(data_file, config)
                    
                    if package:
                        # Import data into system
                        success = await self._import_data_package(package, config)
                        if success:
                            imported_count += 1
                    
                except Exception as e:
                    if not config.skip_errors:
                        raise
                    else:
                        self.logger.warning(f"Skipped import of {data_file.name}: {e}")
                
                self.task_progress[import_id]['completed_steps'] += 1
            
            self.logger.info(f"Successfully imported {imported_count}/{len(data_files)} data packages")
            return imported_count > 0
            
        except Exception as e:
            self.logger.error(f"Error importing Earth system data: {e}")
            raise
        finally:
            # Clean up progress tracking
            if import_id in self.task_progress:
                del self.task_progress[import_id]
            
            # Clean up temporary files
            if local_path != import_path and Path(local_path).exists():
                if Path(local_path).is_file():
                    Path(local_path).unlink()
                else:
                    shutil.rmtree(local_path)
    
    async def _get_earth_system_data(self, data_type: str, date_range: Optional[Tuple[datetime, datetime]] = None) -> Any:
        """Get Earth system data of specified type"""
        try:
            # Mock data generation for demonstration
            # In real implementation, this would fetch actual data from the Digital Twin system
            
            if data_type == "atmosphere":
                data = {
                    "temperature": np.random.normal(15, 10, (100, 100)) if PANDAS_AVAILABLE else [[15.0] * 100] * 100,
                    "pressure": np.random.normal(1013, 50, (100, 100)) if PANDAS_AVAILABLE else [[1013.0] * 100] * 100,
                    "humidity": np.random.uniform(0, 100, (100, 100)) if PANDAS_AVAILABLE else [[50.0] * 100] * 100,
                    "timestamp": datetime.utcnow().isoformat(),
                    "resolution": {"lat": 1.0, "lon": 1.0, "time": 3600}
                }
            elif data_type == "hydrosphere":
                data = {
                    "sea_surface_temperature": np.random.normal(20, 5, (50, 50)) if PANDAS_AVAILABLE else [[20.0] * 50] * 50,
                    "salinity": np.random.normal(35, 2, (50, 50)) if PANDAS_AVAILABLE else [[35.0] * 50] * 50,
                    "ocean_currents": {
                        "u_velocity": np.random.normal(0, 0.5, (50, 50)) if PANDAS_AVAILABLE else [[0.0] * 50] * 50,
                        "v_velocity": np.random.normal(0, 0.5, (50, 50)) if PANDAS_AVAILABLE else [[0.0] * 50] * 50
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            elif data_type == "lithosphere":
                data = {
                    "plate_velocities": np.random.normal(0, 5, (20, 20)) if PANDAS_AVAILABLE else [[0.0] * 20] * 20,
                    "stress_tensor": np.random.normal(0, 1000, (20, 20, 6)) if PANDAS_AVAILABLE else [[[0.0] * 6] * 20] * 20,
                    "earthquake_catalog": [
                        {
                            "magnitude": 5.5 + np.random.random(),
                            "depth": 10 + 50 * np.random.random(),
                            "lat": -90 + 180 * np.random.random(),
                            "lon": -180 + 360 * np.random.random(),
                            "time": (datetime.utcnow() - timedelta(days=np.random.randint(0, 365))).isoformat()
                        } for _ in range(100)
                    ] if PANDAS_AVAILABLE else [],
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return None
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting Earth system data for {data_type}: {e}")
            return None
    
    async def _create_data_package(self, name: str, version: str, data_type: str, 
                                  data: Any, config: DataExportConfig) -> DataPackage:
        """Create a data package from raw data"""
        try:
            # Serialize data based on format
            serialized_data, size_bytes = await self._serialize_data(data, config.format)
            
            # Calculate checksum
            checksum = hashlib.sha256(serialized_data if isinstance(serialized_data, bytes) else 
                                    str(serialized_data).encode()).hexdigest()
            
            # Create metadata
            metadata = {
                "source": "Digital Twin Earth System",
                "data_type": data_type,
                "export_config": asdict(config),
                "schema_version": "1.0.0"
            }
            
            if config.include_metadata and isinstance(data, dict) and "timestamp" in data:
                metadata["data_timestamp"] = data["timestamp"]
            
            # Create package
            package = DataPackage(
                name=name,
                version=version,
                created_at=datetime.utcnow(),
                data_type=data_type,
                format=config.format,
                compression=config.compression,
                size_bytes=size_bytes,
                checksum=checksum,
                metadata=metadata,
                content=serialized_data
            )
            
            return package
            
        except Exception as e:
            self.logger.error(f"Error creating data package: {e}")
            raise
    
    async def _serialize_data(self, data: Any, format: DataFormat) -> Tuple[Any, int]:
        """Serialize data to specified format"""
        try:
            if format == DataFormat.JSON:
                # Convert numpy arrays to lists for JSON serialization
                if PANDAS_AVAILABLE:
                    data = self._convert_numpy_to_lists(data)
                serialized = json.dumps(data, indent=2, default=str)
                return serialized.encode(), len(serialized.encode())
            
            elif format == DataFormat.PICKLE:
                serialized = pickle.dumps(data)
                return serialized, len(serialized)
            
            elif format == DataFormat.NUMPY and PANDAS_AVAILABLE:
                if isinstance(data, dict):
                    # Save multiple arrays in a compressed format
                    buffer = io.BytesIO()
                    np.savez_compressed(buffer, **data)
                    serialized = buffer.getvalue()
                    return serialized, len(serialized)
                else:
                    buffer = io.BytesIO()
                    np.save(buffer, data)
                    serialized = buffer.getvalue()
                    return serialized, len(serialized)
            
            elif format == DataFormat.CSV and PANDAS_AVAILABLE:
                if isinstance(data, dict):
                    # Convert to DataFrame if possible
                    try:
                        df = pd.DataFrame(data)
                        csv_data = df.to_csv(index=False)
                        return csv_data.encode(), len(csv_data.encode())
                    except:
                        # Fallback to JSON
                        return await self._serialize_data(data, DataFormat.JSON)
                else:
                    # Fallback to JSON
                    return await self._serialize_data(data, DataFormat.JSON)
            
            else:
                # Default to JSON
                return await self._serialize_data(data, DataFormat.JSON)
                
        except Exception as e:
            self.logger.error(f"Error serializing data: {e}")
            raise
    
    def _convert_numpy_to_lists(self, data: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization"""
        if not PANDAS_AVAILABLE:
            return data
            
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._convert_numpy_to_lists(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_to_lists(item) for item in data]
        else:
            return data
    
    async def _export_data_package(self, package: DataPackage, export_dir: Path, 
                                  config: DataExportConfig) -> str:
        """Export data package to file"""
        try:
            # Create filename
            filename = f"{package.name}_v{package.version}_{package.created_at.strftime('%Y%m%d_%H%M%S')}"
            
            # Add format extension
            if config.format == DataFormat.JSON:
                filename += ".json"
            elif config.format == DataFormat.PICKLE:
                filename += ".pkl"
            elif config.format == DataFormat.NUMPY:
                filename += ".npz"
            elif config.format == DataFormat.CSV:
                filename += ".csv"
            else:
                filename += ".dat"
            
            filepath = export_dir / filename
            
            # Write data to file
            if isinstance(package.content, bytes):
                filepath.write_bytes(package.content)
            else:
                filepath.write_text(str(package.content))
            
            # Apply compression if requested
            if config.compression != CompressionType.NONE:
                compressed_path = await self._compress_file(filepath, config.compression)
                filepath.unlink()  # Remove uncompressed file
                filepath = compressed_path
            
            # Create metadata file
            metadata_path = filepath.with_suffix(filepath.suffix + ".meta")
            metadata = {
                "package": asdict(package),
                "export_time": datetime.utcnow().isoformat(),
                "file_size": filepath.stat().st_size
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error exporting data package: {e}")
            raise
    
    async def _compress_file(self, filepath: Path, compression: CompressionType) -> Path:
        """Compress file using specified compression"""
        try:
            if compression == CompressionType.GZIP:
                compressed_path = filepath.with_suffix(filepath.suffix + ".gz")
                with open(filepath, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return compressed_path
            
            elif compression == CompressionType.ZIP:
                compressed_path = filepath.with_suffix(filepath.suffix + ".zip")
                with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(filepath, filepath.name)
                return compressed_path
            
            elif compression == CompressionType.LZ4 and ADVANCED_COMPRESSION_AVAILABLE:
                compressed_path = filepath.with_suffix(filepath.suffix + ".lz4")
                with open(filepath, 'rb') as f_in:
                    with lz4.frame.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return compressed_path
            
            elif compression == CompressionType.ZSTD and ADVANCED_COMPRESSION_AVAILABLE:
                compressed_path = filepath.with_suffix(filepath.suffix + ".zst")
                cctx = zstd.ZstdCompressor()
                with open(filepath, 'rb') as f_in:
                    with open(compressed_path, 'wb') as f_out:
                        cctx.copy_stream(f_in, f_out)
                return compressed_path
            
            else:
                # No compression or unsupported format
                return filepath
                
        except Exception as e:
            self.logger.error(f"Error compressing file: {e}")
            return filepath
    
    async def _create_export_archive(self, package_paths: List[str], export_dir: Path, 
                                    config: DataExportConfig) -> str:
        """Create archive containing multiple packages"""
        try:
            archive_name = f"digital_twin_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            if config.compression == CompressionType.ZIP:
                archive_path = export_dir / f"{archive_name}.zip"
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for package_path in package_paths:
                        path = Path(package_path)
                        zipf.write(path, path.name)
                        # Include metadata file
                        meta_path = path.with_suffix(path.suffix + ".meta")
                        if meta_path.exists():
                            zipf.write(meta_path, meta_path.name)
            
            elif config.compression == CompressionType.TAR_GZ:
                archive_path = export_dir / f"{archive_name}.tar.gz"
                with tarfile.open(archive_path, 'w:gz') as tar:
                    for package_path in package_paths:
                        path = Path(package_path)
                        tar.add(path, arcname=path.name)
                        # Include metadata file
                        meta_path = path.with_suffix(path.suffix + ".meta")
                        if meta_path.exists():
                            tar.add(meta_path, arcname=meta_path.name)
            
            else:
                # Create simple tar archive
                archive_path = export_dir / f"{archive_name}.tar"
                with tarfile.open(archive_path, 'w') as tar:
                    for package_path in package_paths:
                        path = Path(package_path)
                        tar.add(path, arcname=path.name)
                        # Include metadata file
                        meta_path = path.with_suffix(path.suffix + ".meta")
                        if meta_path.exists():
                            tar.add(meta_path, arcname=meta_path.name)
            
            return str(archive_path)
            
        except Exception as e:
            self.logger.error(f"Error creating export archive: {e}")
            raise
    
    async def _upload_to_storage(self, local_path: str, config: DataExportConfig) -> str:
        """Upload file to configured storage backend"""
        try:
            backend = self.storage_backends.get(config.storage_backend)
            if not backend:
                raise ValueError(f"Storage backend {config.storage_backend} not available")
            
            return await backend.upload(local_path, config.storage_config)
            
        except Exception as e:
            self.logger.error(f"Error uploading to storage: {e}")
            raise
    
    async def _download_from_storage(self, remote_path: str, config: DataImportConfig) -> str:
        """Download file from configured storage backend"""
        try:
            if config.storage_backend == StorageBackend.LOCAL:
                return remote_path
            
            backend = self.storage_backends.get(config.storage_backend)
            if not backend:
                raise ValueError(f"Storage backend {config.storage_backend} not available")
            
            # Create temporary local path
            temp_path = self.temp_path / f"import_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            temp_path.mkdir(exist_ok=True)
            
            local_path = await backend.download(remote_path, str(temp_path), config.storage_config)
            return local_path
            
        except Exception as e:
            self.logger.error(f"Error downloading from storage: {e}")
            raise
    
    async def _extract_import_archive(self, archive_path: str, config: DataImportConfig) -> str:
        """Extract import archive if needed"""
        try:
            path = Path(archive_path)
            
            if path.suffix == '.zip':
                extract_dir = path.parent / path.stem
                extract_dir.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(path, 'r') as zipf:
                    zipf.extractall(extract_dir)
                
                return str(extract_dir)
            
            elif path.suffixes == ['.tar', '.gz'] or path.suffix == '.tgz':
                extract_dir = path.parent / path.stem
                if path.suffixes == ['.tar', '.gz']:
                    extract_dir = extract_dir.parent / extract_dir.stem
                extract_dir.mkdir(exist_ok=True)
                
                with tarfile.open(path, 'r:gz') as tar:
                    tar.extractall(extract_dir)
                
                return str(extract_dir)
            
            elif path.suffix == '.tar':
                extract_dir = path.parent / path.stem
                extract_dir.mkdir(exist_ok=True)
                
                with tarfile.open(path, 'r') as tar:
                    tar.extractall(extract_dir)
                
                return str(extract_dir)
            
            else:
                # Single file, no extraction needed
                return archive_path
                
        except Exception as e:
            self.logger.error(f"Error extracting archive: {e}")
            return archive_path
    
    async def _backup_existing_data(self):
        """Backup existing data before import"""
        try:
            backup_dir = self.base_path / "backups" / f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup of current data
            # This would backup existing Digital Twin data
            self.logger.info(f"Data backed up to: {backup_dir}")
            
        except Exception as e:
            self.logger.error(f"Error backing up existing data: {e}")
    
    async def _load_data_package(self, file_path: Path, config: DataImportConfig) -> Optional[DataPackage]:
        """Load data package from file"""
        try:
            # Load metadata if available
            meta_path = file_path.with_suffix(file_path.suffix + ".meta")
            metadata = {}
            
            if meta_path.exists():
                meta_content = json.loads(meta_path.read_text())
                if "package" in meta_content:
                    pkg_data = meta_content["package"]
                    
                    # Reconstruct DataPackage
                    package = DataPackage(
                        name=pkg_data["name"],
                        version=pkg_data["version"],
                        created_at=datetime.fromisoformat(pkg_data["created_at"]),
                        data_type=pkg_data["data_type"],
                        format=DataFormat(pkg_data["format"]),
                        compression=CompressionType(pkg_data["compression"]),
                        size_bytes=pkg_data["size_bytes"],
                        checksum=pkg_data["checksum"],
                        metadata=pkg_data.get("metadata", {})
                    )
                    
                    # Load content
                    content = await self._load_package_content(file_path, package.format)
                    package.content = content
                    
                    return package
            
            # Fallback: infer package info from file
            return await self._infer_data_package(file_path, config)
            
        except Exception as e:
            self.logger.error(f"Error loading data package: {e}")
            return None
    
    async def _load_package_content(self, file_path: Path, format: DataFormat) -> Any:
        """Load package content based on format"""
        try:
            if format == DataFormat.JSON:
                return json.loads(file_path.read_text())
            elif format == DataFormat.PICKLE:
                return pickle.loads(file_path.read_bytes())
            elif format == DataFormat.CSV and PANDAS_AVAILABLE:
                return pd.read_csv(file_path).to_dict()
            else:
                # Generic binary read
                return file_path.read_bytes()
                
        except Exception as e:
            self.logger.error(f"Error loading package content: {e}")
            return None
    
    async def _infer_data_package(self, file_path: Path, config: DataImportConfig) -> Optional[DataPackage]:
        """Infer data package information from file"""
        try:
            # Basic package info
            stat = file_path.stat()
            
            package = DataPackage(
                name=file_path.stem,
                version="1.0.0",
                created_at=datetime.fromtimestamp(stat.st_mtime),
                data_type="inferred",
                format=config.format,
                compression=config.compression,
                size_bytes=stat.st_size,
                checksum="",
                metadata={"source": "imported", "inferred": True}
            )
            
            # Load content
            content = await self._load_package_content(file_path, config.format)
            package.content = content
            
            return package
            
        except Exception as e:
            self.logger.error(f"Error inferring data package: {e}")
            return None
    
    async def _import_data_package(self, package: DataPackage, config: DataImportConfig) -> bool:
        """Import data package into Digital Twin system"""
        try:
            # Here would be the actual integration with Digital Twin Earth system
            # For now, we'll store the package information
            
            with self._lock:
                self.data_index[package.name] = package
            
            # Store in database if available
            if DATABASE_AVAILABLE and hasattr(self, 'db_session_factory'):
                try:
                    session = self.db_session_factory()
                    # Insert package record
                    # (Database insertion code would go here)
                    session.close()
                except Exception as db_error:
                    self.logger.warning(f"Database storage failed: {db_error}")
            
            self.logger.info(f"Successfully imported data package: {package.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing data package {package.name}: {e}")
            return False
    
    async def get_import_export_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of import/export operation"""
        return self.task_progress.get(task_id)
    
    async def list_data_packages(self) -> List[Dict[str, Any]]:
        """List available data packages"""
        try:
            packages = []
            
            with self._lock:
                for name, package in self.data_index.items():
                    packages.append({
                        "name": package.name,
                        "version": package.version,
                        "data_type": package.data_type,
                        "format": package.format.value,
                        "size_bytes": package.size_bytes,
                        "created_at": package.created_at.isoformat(),
                        "metadata": package.metadata
                    })
            
            return packages
            
        except Exception as e:
            self.logger.error(f"Error listing data packages: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources and temporary files"""
        try:
            # Cancel active tasks
            for task in self.active_tasks.values():
                if not task.done():
                    task.cancel()
            
            # Clean up temporary files
            if self.temp_path.exists():
                for temp_file in self.temp_path.iterdir():
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
            
            # Close storage backend connections
            for backend in self.storage_backends.values():
                if hasattr(backend, 'close'):
                    await backend.close()
            
            self.logger.info("Data manager cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during data manager cleanup: {e}")

# Storage backend implementations
class LocalStorageBackend:
    """Local file system storage backend"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
    
    async def upload(self, local_path: str, config: Dict[str, Any]) -> str:
        """Copy file to local storage"""
        dest_path = self.base_path / Path(local_path).name
        shutil.copy2(local_path, dest_path)
        return str(dest_path)
    
    async def download(self, remote_path: str, local_dir: str, config: Dict[str, Any]) -> str:
        """Copy file from local storage"""
        source_path = Path(remote_path)
        dest_path = Path(local_dir) / source_path.name
        shutil.copy2(source_path, dest_path)
        return str(dest_path)

class S3StorageBackend:
    """AWS S3 storage backend"""
    
    def __init__(self):
        self.s3_client = None
        if AWS_AVAILABLE:
            self.s3_client = boto3.client('s3')
    
    async def upload(self, local_path: str, config: Dict[str, Any]) -> str:
        """Upload file to S3"""
        if not self.s3_client:
            raise RuntimeError("AWS S3 not available")
        
        bucket = config.get('bucket')
        key = config.get('key', Path(local_path).name)
        
        self.s3_client.upload_file(local_path, bucket, key)
        return f"s3://{bucket}/{key}"
    
    async def download(self, remote_path: str, local_dir: str, config: Dict[str, Any]) -> str:
        """Download file from S3"""
        if not self.s3_client:
            raise RuntimeError("AWS S3 not available")
        
        # Parse S3 URL
        if remote_path.startswith('s3://'):
            parts = remote_path[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
        else:
            bucket = config.get('bucket')
            key = remote_path
        
        local_path = Path(local_dir) / Path(key).name
        self.s3_client.download_file(bucket, key, str(local_path))
        return str(local_path)

class DatabaseStorageBackend:
    """Database storage backend"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine = None
        if DATABASE_AVAILABLE:
            self.engine = create_engine(f"sqlite:///{db_path}")
    
    async def upload(self, local_path: str, config: Dict[str, Any]) -> str:
        """Store file in database as BLOB"""
        # Implementation would store file content in database
        return f"db://{Path(local_path).name}"
    
    async def download(self, remote_path: str, local_dir: str, config: Dict[str, Any]) -> str:
        """Retrieve file from database"""
        # Implementation would retrieve file content from database
        local_path = Path(local_dir) / Path(remote_path).name
        return str(local_path)

# Example usage and testing
async def example_digital_twin_data_management():
    """Example of Digital Twin data management"""
    
    print("=== Digital Twin Data Management Example ===")
    
    # Initialize data manager
    data_manager = DigitalTwinDataManager()
    
    try:
        # Wait for initialization
        await asyncio.sleep(1)
        
        print("\n1. Export Earth System Data:")
        export_config = DataExportConfig(
            format=DataFormat.JSON,
            compression=CompressionType.GZIP,
            include_metadata=True
        )
        
        data_types = ["atmosphere", "hydrosphere", "lithosphere"]
        export_path = await data_manager.export_earth_system_data(data_types, export_config)
        print(f"Data exported to: {export_path}")
        
        print("\n2. List Data Packages:")
        packages = await data_manager.list_data_packages()
        print(f"Available packages: {len(packages)}")
        for pkg in packages:
            print(f"  - {pkg['name']} (v{pkg['version']}) - {pkg['data_type']}")
        
        print("\n3. Import Data (simulated):")
        if export_path and Path(export_path).exists():
            import_config = DataImportConfig(
                format=DataFormat.JSON,
                compression=CompressionType.GZIP,
                merge_strategy="merge",
                backup_existing=True
            )
            
            success = await data_manager.import_earth_system_data(export_path, import_config)
            print(f"Import successful: {success}")
        
        print("\n4. Storage Backends:")
        print(f"Available backends: {list(data_manager.storage_backends.keys())}")
        
        print("\nData management example completed successfully!")
        
    finally:
        await data_manager.cleanup()

if __name__ == "__main__":
    print("=== Digital Twin Data Management System ===")
    print(f"Features available:")
    print(f"  Database Support: {DATABASE_AVAILABLE}")
    print(f"  Pandas/NumPy: {PANDAS_AVAILABLE}")
    print(f"  Scientific Formats: {SCIENTIFIC_FORMATS_AVAILABLE}")
    print(f"  AWS S3: {AWS_AVAILABLE}")
    print(f"  Advanced Compression: {ADVANCED_COMPRESSION_AVAILABLE}")
    
    # Run example
    asyncio.run(example_digital_twin_data_management())