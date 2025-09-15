"""
Automated Data Archiving and Retention Policies System

This module provides comprehensive data archiving, retention management, and
lifecycle policies for geological data across all database systems with
intelligent tiering, compression, and compliance features.
"""

import asyncio
import json
import logging
import gzip
import bz2
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import os
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import asyncpg
import redis.asyncio as redis
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
import boto3
from azure.storage.blob.aio import BlobServiceClient
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class StorageTier(Enum):
    """Storage tiers for data lifecycle management"""
    HOT = "hot"              # Frequently accessed, high performance
    WARM = "warm"            # Occasionally accessed, balanced performance/cost
    COLD = "cold"            # Rarely accessed, low cost
    ARCHIVE = "archive"      # Long-term retention, lowest cost
    DEEP_ARCHIVE = "deep_archive"  # Compliance/regulatory, very low cost


class CompressionType(Enum):
    """Compression algorithms for archived data"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZ4 = "lz4"
    ZSTD = "zstd"


class RetentionAction(Enum):
    """Actions to take when retention period expires"""
    DELETE = "delete"
    ARCHIVE = "archive"
    COMPRESS = "compress"
    MIGRATE = "migrate"
    NOTIFY = "notify"


@dataclass
class RetentionPolicy:
    """Data retention policy definition"""
    policy_id: str
    policy_name: str
    database_name: str
    table_name: str
    retention_period_days: int
    action: RetentionAction
    storage_tier: StorageTier
    compression_type: CompressionType
    archive_location: Optional[str]
    conditions: Dict[str, Any]
    enabled: bool = True
    created_by: str = "system"
    created_timestamp: datetime = None
    
    def __post_init__(self):
        if self.created_timestamp is None:
            self.created_timestamp = datetime.utcnow()


@dataclass
class ArchiveJob:
    """Data archiving job definition"""
    job_id: str
    policy_id: str
    database_name: str
    table_name: str
    start_date: datetime
    end_date: datetime
    estimated_records: int
    estimated_size_mb: float
    target_location: str
    compression_type: CompressionType
    status: str = "pending"
    created_timestamp: datetime = None
    started_timestamp: Optional[datetime] = None
    completed_timestamp: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_timestamp is None:
            self.created_timestamp = datetime.utcnow()


@dataclass
class ArchiveMetadata:
    """Metadata for archived data"""
    archive_id: str
    original_database: str
    original_table: str
    archive_location: str
    compression_type: CompressionType
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    record_count: int
    date_range_start: datetime
    date_range_end: datetime
    archive_timestamp: datetime
    checksum: str
    retrieval_instructions: str


class DataArchivingManager:
    """Advanced data archiving and retention management system"""
    
    def __init__(self,
                 postgres_connection_string: str,
                 influxdb_url: str,
                 influxdb_token: str,
                 influxdb_org: str,
                 redis_url: str,
                 s3_config: Optional[Dict[str, str]] = None,
                 azure_config: Optional[Dict[str, str]] = None,
                 local_archive_path: str = "/data/archives"):
        
        self.postgres_connection_string = postgres_connection_string
        self.influxdb_url = influxdb_url
        self.influxdb_token = influxdb_token
        self.influxdb_org = influxdb_org
        self.redis_url = redis_url
        self.local_archive_path = Path(local_archive_path)
        
        # Cloud storage configurations
        self.s3_config = s3_config
        self.azure_config = azure_config
        
        # Database connections
        self.postgres_conn = None
        self.influxdb_client = None
        self.redis_client = None
        
        # Cloud storage clients
        self.s3_client = None
        self.azure_client = None
        
        # Retention policies registry
        self.retention_policies: Dict[str, RetentionPolicy] = {}
        
        # Active archive jobs
        self.active_jobs: Dict[str, asyncio.Task] = {}
        
        # Archiving statistics
        self.archiving_stats = {
            'total_archived_records': 0,
            'total_archived_size_mb': 0,
            'total_compression_savings_mb': 0,
            'active_policies': 0,
            'completed_jobs': 0,
            'failed_jobs': 0
        }
    
    async def initialize(self) -> None:
        """Initialize database connections and create schema"""
        try:
            # Initialize database connections
            self.postgres_conn = await asyncpg.connect(self.postgres_connection_string)
            self.influxdb_client = InfluxDBClientAsync(
                url=self.influxdb_url,
                token=self.influxdb_token,
                org=self.influxdb_org
            )
            self.redis_client = redis.from_url(self.redis_url)
            
            # Initialize cloud storage clients
            if self.s3_config:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.s3_config.get('access_key_id'),
                    aws_secret_access_key=self.s3_config.get('secret_access_key'),
                    region_name=self.s3_config.get('region', 'us-east-1')
                )
            
            if self.azure_config:
                self.azure_client = BlobServiceClient(
                    account_url=self.azure_config.get('account_url'),
                    credential=self.azure_config.get('credential')
                )
            
            # Create local archive directory
            self.local_archive_path.mkdir(parents=True, exist_ok=True)
            
            # Create database schema
            await self._create_archiving_schema()
            
            # Load existing retention policies
            await self._load_retention_policies()
            
            logger.info("Data archiving manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize data archiving manager", error=str(e))
            raise
    
    async def close(self) -> None:
        """Close database connections and cleanup"""
        try:
            # Cancel active jobs
            for job_id, task in self.active_jobs.items():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Close database connections
            if self.postgres_conn:
                await self.postgres_conn.close()
            
            if self.influxdb_client:
                await self.influxdb_client.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            if self.azure_client:
                await self.azure_client.close()
            
            logger.info("Data archiving manager connections closed")
            
        except Exception as e:
            logger.error("Failed to close connections", error=str(e))
    
    async def _create_archiving_schema(self) -> None:
        """Create database schema for archiving system"""
        try:
            # Create retention policies table
            await self.postgres_conn.execute("""
                CREATE TABLE IF NOT EXISTS retention_policies (
                    policy_id VARCHAR(100) PRIMARY KEY,
                    policy_name VARCHAR(255) NOT NULL,
                    database_name VARCHAR(100) NOT NULL,
                    table_name VARCHAR(100) NOT NULL,
                    retention_period_days INTEGER NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    storage_tier VARCHAR(50) NOT NULL,
                    compression_type VARCHAR(50) NOT NULL,
                    archive_location TEXT,
                    conditions JSONB,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_by VARCHAR(100),
                    created_timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create archive jobs table
            await self.postgres_conn.execute("""
                CREATE TABLE IF NOT EXISTS archive_jobs (
                    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    policy_id VARCHAR(100) REFERENCES retention_policies(policy_id),
                    database_name VARCHAR(100) NOT NULL,
                    table_name VARCHAR(100) NOT NULL,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP NOT NULL,
                    estimated_records BIGINT,
                    estimated_size_mb FLOAT,
                    target_location TEXT NOT NULL,
                    compression_type VARCHAR(50) NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    created_timestamp TIMESTAMP DEFAULT NOW(),
                    started_timestamp TIMESTAMP,
                    completed_timestamp TIMESTAMP,
                    error_message TEXT
                )
            """)
            
            # Create archive metadata table
            await self.postgres_conn.execute("""
                CREATE TABLE IF NOT EXISTS archive_metadata (
                    archive_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    original_database VARCHAR(100) NOT NULL,
                    original_table VARCHAR(100) NOT NULL,
                    archive_location TEXT NOT NULL,
                    compression_type VARCHAR(50) NOT NULL,
                    original_size_mb FLOAT NOT NULL,
                    compressed_size_mb FLOAT NOT NULL,
                    compression_ratio FLOAT NOT NULL,
                    record_count BIGINT NOT NULL,
                    date_range_start TIMESTAMP NOT NULL,
                    date_range_end TIMESTAMP NOT NULL,
                    archive_timestamp TIMESTAMP DEFAULT NOW(),
                    checksum VARCHAR(64) NOT NULL,
                    retrieval_instructions TEXT
                )
            """)
            
            # Create data lifecycle events table
            await self.postgres_conn.execute("""
                CREATE TABLE IF NOT EXISTS data_lifecycle_events (
                    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    database_name VARCHAR(100) NOT NULL,
                    table_name VARCHAR(100) NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    event_timestamp TIMESTAMP DEFAULT NOW(),
                    records_affected BIGINT,
                    size_mb FLOAT,
                    details JSONB,
                    policy_id VARCHAR(100)
                )
            """)
            
            # Create indexes for performance
            await self.postgres_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_retention_policies_table ON retention_policies(database_name, table_name);
                CREATE INDEX IF NOT EXISTS idx_archive_jobs_status ON archive_jobs(status);
                CREATE INDEX IF NOT EXISTS idx_archive_jobs_timestamp ON archive_jobs(created_timestamp);
                CREATE INDEX IF NOT EXISTS idx_archive_metadata_table ON archive_metadata(original_database, original_table);
                CREATE INDEX IF NOT EXISTS idx_archive_metadata_date_range ON archive_metadata(date_range_start, date_range_end);
                CREATE INDEX IF NOT EXISTS idx_lifecycle_events_timestamp ON data_lifecycle_events(event_timestamp);
            """)
            
            logger.info("Archiving database schema created successfully")
            
        except Exception as e:
            logger.error("Failed to create archiving schema", error=str(e))
            raise
    
    async def _load_retention_policies(self) -> None:
        """Load existing retention policies from database"""
        try:
            policies_query = "SELECT * FROM retention_policies WHERE enabled = TRUE"
            policy_records = await self.postgres_conn.fetch(policies_query)
            
            for record in policy_records:
                policy = RetentionPolicy(
                    policy_id=record['policy_id'],
                    policy_name=record['policy_name'],
                    database_name=record['database_name'],
                    table_name=record['table_name'],
                    retention_period_days=record['retention_period_days'],
                    action=RetentionAction(record['action']),
                    storage_tier=StorageTier(record['storage_tier']),
                    compression_type=CompressionType(record['compression_type']),
                    archive_location=record['archive_location'],
                    conditions=record['conditions'] or {},
                    enabled=record['enabled'],
                    created_by=record['created_by'] or 'system',
                    created_timestamp=record['created_timestamp']
                )
                
                self.retention_policies[policy.policy_id] = policy
            
            self.archiving_stats['active_policies'] = len(self.retention_policies)
            
            logger.info("Loaded retention policies", 
                       total_policies=len(self.retention_policies))
            
        except Exception as e:
            logger.error("Failed to load retention policies", error=str(e))
            raise
    
    async def create_retention_policy(self, 
                                    policy_name: str,
                                    database_name: str,
                                    table_name: str,
                                    retention_period_days: int,
                                    action: RetentionAction,
                                    storage_tier: StorageTier = StorageTier.COLD,
                                    compression_type: CompressionType = CompressionType.GZIP,
                                    archive_location: Optional[str] = None,
                                    conditions: Dict[str, Any] = None,
                                    created_by: str = "system") -> str:
        """Create a new retention policy"""
        try:
            policy_id = f"{database_name}_{table_name}_{uuid.uuid4().hex[:8]}"
            
            policy = RetentionPolicy(
                policy_id=policy_id,
                policy_name=policy_name,
                database_name=database_name,
                table_name=table_name,
                retention_period_days=retention_period_days,
                action=action,
                storage_tier=storage_tier,
                compression_type=compression_type,
                archive_location=archive_location,
                conditions=conditions or {},
                created_by=created_by
            )
            
            # Store in database
            await self.postgres_conn.execute("""
                INSERT INTO retention_policies 
                (policy_id, policy_name, database_name, table_name, retention_period_days,
                 action, storage_tier, compression_type, archive_location, conditions, created_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                policy.policy_id, policy.policy_name, policy.database_name,
                policy.table_name, policy.retention_period_days, policy.action.value,
                policy.storage_tier.value, policy.compression_type.value,
                policy.archive_location, json.dumps(policy.conditions), policy.created_by
            )
            
            # Add to memory
            self.retention_policies[policy_id] = policy
            self.archiving_stats['active_policies'] += 1
            
            logger.info("Created retention policy", 
                       policy_id=policy_id,
                       policy_name=policy_name,
                       retention_days=retention_period_days)
            
            return policy_id
            
        except Exception as e:
            logger.error("Failed to create retention policy", 
                        policy_name=policy_name,
                        error=str(e))
            raise
    
    async def execute_retention_policies(self) -> Dict[str, Any]:
        """Execute all active retention policies"""
        try:
            execution_results = {
                'policies_executed': 0,
                'records_processed': 0,
                'jobs_created': 0,
                'errors': []
            }
            
            for policy_id, policy in self.retention_policies.items():
                try:
                    if not policy.enabled:
                        continue
                    
                    # Find data eligible for retention action
                    eligible_data = await self._find_eligible_data(policy)
                    
                    if eligible_data['record_count'] > 0:
                        # Create archive job
                        job_id = await self._create_archive_job(policy, eligible_data)
                        
                        # Execute job asynchronously
                        task = asyncio.create_task(self._execute_archive_job(job_id))
                        self.active_jobs[job_id] = task
                        
                        execution_results['jobs_created'] += 1
                        execution_results['records_processed'] += eligible_data['record_count']
                    
                    execution_results['policies_executed'] += 1
                    
                except Exception as e:
                    error_msg = f"Policy {policy_id} execution failed: {str(e)}"
                    execution_results['errors'].append(error_msg)
                    logger.error("Retention policy execution failed", 
                               policy_id=policy_id,
                               error=str(e))
            
            logger.info("Retention policies execution completed", 
                       **execution_results)
            
            return execution_results
            
        except Exception as e:
            logger.error("Failed to execute retention policies", error=str(e))
            raise
    
    async def _find_eligible_data(self, policy: RetentionPolicy) -> Dict[str, Any]:
        """Find data eligible for retention action based on policy"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_period_days)
            
            # Build query based on database type and conditions
            if policy.database_name == 'postgresql':
                return await self._find_eligible_postgresql_data(policy, cutoff_date)
            elif policy.database_name == 'influxdb':
                return await self._find_eligible_influxdb_data(policy, cutoff_date)
            else:
                logger.warning("Unsupported database for retention", 
                             database=policy.database_name)
                return {'record_count': 0, 'size_mb': 0}
            
        except Exception as e:
            logger.error("Failed to find eligible data", 
                        policy_id=policy.policy_id,
                        error=str(e))
            return {'record_count': 0, 'size_mb': 0}
    
    async def _find_eligible_postgresql_data(self, 
                                           policy: RetentionPolicy,
                                           cutoff_date: datetime) -> Dict[str, Any]:
        """Find eligible PostgreSQL data for retention"""
        try:
            # Determine timestamp column
            timestamp_column = policy.conditions.get('timestamp_column', 'timestamp')
            
            # Build base query
            base_query = f"""
                SELECT 
                    COUNT(*) as record_count,
                    MIN({timestamp_column}) as min_date,
                    MAX({timestamp_column}) as max_date
                FROM {policy.table_name}
                WHERE {timestamp_column} < $1
            """
            
            # Add additional conditions
            additional_conditions = policy.conditions.get('where_clause')
            if additional_conditions:
                base_query += f" AND ({additional_conditions})"
            
            result = await self.postgres_conn.fetchrow(base_query, cutoff_date)
            
            # Estimate size (rough calculation)
            if result['record_count'] > 0:
                size_query = f"""
                    SELECT pg_total_relation_size('{policy.table_name}') as table_size_bytes
                """
                size_result = await self.postgres_conn.fetchrow(size_query)
                
                # Estimate size based on proportion of records
                total_records_query = f"SELECT COUNT(*) as total FROM {policy.table_name}"
                total_result = await self.postgres_conn.fetchrow(total_records_query)
                
                if total_result['total'] > 0:
                    size_proportion = result['record_count'] / total_result['total']
                    estimated_size_mb = (size_result['table_size_bytes'] * size_proportion) / (1024 * 1024)
                else:
                    estimated_size_mb = 0
            else:
                estimated_size_mb = 0
            
            return {
                'record_count': result['record_count'],
                'size_mb': estimated_size_mb,
                'date_range_start': result['min_date'],
                'date_range_end': result['max_date']
            }
            
        except Exception as e:
            logger.error("Failed to find eligible PostgreSQL data", error=str(e))
            return {'record_count': 0, 'size_mb': 0}
    
    async def _find_eligible_influxdb_data(self, 
                                         policy: RetentionPolicy,
                                         cutoff_date: datetime) -> Dict[str, Any]:
        """Find eligible InfluxDB data for retention"""
        try:
            # Build Flux query for InfluxDB
            flux_query = f'''
                from(bucket: "{policy.conditions.get('bucket', 'geological_data')}")
                |> range(start: 0, stop: {cutoff_date.isoformat()}Z)
                |> filter(fn: (r) => r._measurement == "{policy.table_name}")
                |> count()
            '''
            
            query_api = self.influxdb_client.query_api()
            result = await query_api.query(flux_query)
            
            # Process result to get count
            record_count = 0
            if result and len(result) > 0:
                for table in result:
                    for record in table.records:
                        record_count += record.get_value()
            
            # Estimate size (InfluxDB doesn't provide easy size estimation)
            estimated_size_mb = record_count * 0.001  # Rough estimate: 1KB per record
            
            return {
                'record_count': record_count,
                'size_mb': estimated_size_mb,
                'date_range_start': datetime.min,
                'date_range_end': cutoff_date
            }
            
        except Exception as e:
            logger.error("Failed to find eligible InfluxDB data", error=str(e))
            return {'record_count': 0, 'size_mb': 0}
    
    async def _create_archive_job(self, 
                                policy: RetentionPolicy,
                                eligible_data: Dict[str, Any]) -> str:
        """Create an archive job for eligible data"""
        try:
            job_id = str(uuid.uuid4())
            
            # Determine target location
            if policy.archive_location:
                target_location = policy.archive_location
            else:
                target_location = str(self.local_archive_path / f"{policy.database_name}_{policy.table_name}_{job_id}")
            
            job = ArchiveJob(
                job_id=job_id,
                policy_id=policy.policy_id,
                database_name=policy.database_name,
                table_name=policy.table_name,
                start_date=eligible_data.get('date_range_start', datetime.min),
                end_date=eligible_data.get('date_range_end', datetime.utcnow()),
                estimated_records=eligible_data['record_count'],
                estimated_size_mb=eligible_data['size_mb'],
                target_location=target_location,
                compression_type=policy.compression_type
            )
            
            # Store in database
            await self.postgres_conn.execute("""
                INSERT INTO archive_jobs 
                (job_id, policy_id, database_name, table_name, start_date, end_date,
                 estimated_records, estimated_size_mb, target_location, compression_type)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                job.job_id, job.policy_id, job.database_name, job.table_name,
                job.start_date, job.end_date, job.estimated_records,
                job.estimated_size_mb, job.target_location, job.compression_type.value
            )
            
            logger.info("Created archive job", 
                       job_id=job_id,
                       policy_id=policy.policy_id,
                       estimated_records=job.estimated_records)
            
            return job_id
            
        except Exception as e:
            logger.error("Failed to create archive job", error=str(e))
            raise
    
    async def _execute_archive_job(self, job_id: str) -> None:
        """Execute an archive job"""
        try:
            # Update job status to running
            await self.postgres_conn.execute("""
                UPDATE archive_jobs 
                SET status = 'running', started_timestamp = NOW()
                WHERE job_id = $1
            """, job_id)
            
            # Get job details
            job_record = await self.postgres_conn.fetchrow("""
                SELECT * FROM archive_jobs WHERE job_id = $1
            """, job_id)
            
            if not job_record:
                raise ValueError(f"Archive job {job_id} not found")
            
            # Get policy details
            policy = self.retention_policies.get(job_record['policy_id'])
            if not policy:
                raise ValueError(f"Policy {job_record['policy_id']} not found")
            
            logger.info("Starting archive job execution", 
                       job_id=job_id,
                       database=job_record['database_name'],
                       table=job_record['table_name'])
            
            # Execute archiving based on database type
            if job_record['database_name'] == 'postgresql':
                archive_metadata = await self._archive_postgresql_data(job_record, policy)
            elif job_record['database_name'] == 'influxdb':
                archive_metadata = await self._archive_influxdb_data(job_record, policy)
            else:
                raise ValueError(f"Unsupported database: {job_record['database_name']}")
            
            # Store archive metadata
            await self._store_archive_metadata(archive_metadata)
            
            # Update job status to completed
            await self.postgres_conn.execute("""
                UPDATE archive_jobs 
                SET status = 'completed', completed_timestamp = NOW()
                WHERE job_id = $1
            """, job_id)
            
            # Update statistics
            self.archiving_stats['completed_jobs'] += 1
            self.archiving_stats['total_archived_records'] += archive_metadata.record_count
            self.archiving_stats['total_archived_size_mb'] += archive_metadata.original_size_mb
            self.archiving_stats['total_compression_savings_mb'] += (
                archive_metadata.original_size_mb - archive_metadata.compressed_size_mb
            )
            
            # Log lifecycle event
            await self._log_lifecycle_event(
                database_name=job_record['database_name'],
                table_name=job_record['table_name'],
                event_type='archived',
                records_affected=archive_metadata.record_count,
                size_mb=archive_metadata.original_size_mb,
                details={
                    'job_id': job_id,
                    'archive_location': archive_metadata.archive_location,
                    'compression_ratio': archive_metadata.compression_ratio
                },
                policy_id=policy.policy_id
            )
            
            logger.info("Archive job completed successfully", 
                       job_id=job_id,
                       records_archived=archive_metadata.record_count,
                       compression_ratio=archive_metadata.compression_ratio)
            
        except Exception as e:
            # Update job status to failed
            await self.postgres_conn.execute("""
                UPDATE archive_jobs 
                SET status = 'failed', error_message = $2, completed_timestamp = NOW()
                WHERE job_id = $1
            """, job_id, str(e))
            
            self.archiving_stats['failed_jobs'] += 1
            
            logger.error("Archive job failed", 
                        job_id=job_id,
                        error=str(e))
            raise
        
        finally:
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]


# Continue with archiving implementation in next part...   
    async def _archive_postgresql_data(self, 
                                      job_record: Dict[str, Any],
                                      policy: RetentionPolicy) -> ArchiveMetadata:
        """Archive PostgreSQL data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_period_days)
            timestamp_column = policy.conditions.get('timestamp_column', 'timestamp')
            
            # Extract data to archive
            query = f"""
                SELECT * FROM {job_record['table_name']}
                WHERE {timestamp_column} < $1
            """
            
            # Add additional conditions
            additional_conditions = policy.conditions.get('where_clause')
            if additional_conditions:
                query += f" AND ({additional_conditions})"
            
            query += f" ORDER BY {timestamp_column}"
            
            # Execute query and get data
            records = await self.postgres_conn.fetch(query, cutoff_date)
            
            if not records:
                raise ValueError("No data found to archive")
            
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(records)
            
            # Calculate original size
            original_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Create archive file
            archive_location = await self._create_archive_file(
                df, 
                job_record['target_location'],
                CompressionType(job_record['compression_type'])
            )
            
            # Calculate compressed size
            compressed_size_mb = os.path.getsize(archive_location) / (1024 * 1024)
            compression_ratio = original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 1.0
            
            # Calculate checksum
            checksum = await self._calculate_file_checksum(archive_location)
            
            # Delete original data if policy action is DELETE
            if policy.action == RetentionAction.DELETE:
                delete_query = f"""
                    DELETE FROM {job_record['table_name']}
                    WHERE {timestamp_column} < $1
                """
                
                if additional_conditions:
                    delete_query += f" AND ({additional_conditions})"
                
                deleted_count = await self.postgres_conn.execute(delete_query, cutoff_date)
                logger.info("Deleted original data after archiving", 
                           table=job_record['table_name'],
                           deleted_records=deleted_count)
            
            # Create archive metadata
            archive_metadata = ArchiveMetadata(
                archive_id=str(uuid.uuid4()),
                original_database=job_record['database_name'],
                original_table=job_record['table_name'],
                archive_location=archive_location,
                compression_type=CompressionType(job_record['compression_type']),
                original_size_mb=original_size_mb,
                compressed_size_mb=compressed_size_mb,
                compression_ratio=compression_ratio,
                record_count=len(df),
                date_range_start=job_record['start_date'],
                date_range_end=job_record['end_date'],
                archive_timestamp=datetime.utcnow(),
                checksum=checksum,
                retrieval_instructions=f"Use pandas.read_parquet('{archive_location}') to restore data"
            )
            
            return archive_metadata
            
        except Exception as e:
            logger.error("Failed to archive PostgreSQL data", error=str(e))
            raise
    
    async def _archive_influxdb_data(self, 
                                   job_record: Dict[str, Any],
                                   policy: RetentionPolicy) -> ArchiveMetadata:
        """Archive InfluxDB data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_period_days)
            bucket = policy.conditions.get('bucket', 'geological_data')
            
            # Build Flux query to extract data
            flux_query = f'''
                from(bucket: "{bucket}")
                |> range(start: 0, stop: {cutoff_date.isoformat()}Z)
                |> filter(fn: (r) => r._measurement == "{job_record['table_name']}")
            '''
            
            # Add additional filters
            additional_filters = policy.conditions.get('flux_filters')
            if additional_filters:
                flux_query += f"\n{additional_filters}"
            
            # Execute query
            query_api = self.influxdb_client.query_api()
            result = await query_api.query_data_frame(flux_query)
            
            if result.empty:
                raise ValueError("No data found to archive")
            
            # Calculate original size
            original_size_mb = result.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Create archive file
            archive_location = await self._create_archive_file(
                result, 
                job_record['target_location'],
                CompressionType(job_record['compression_type'])
            )
            
            # Calculate compressed size
            compressed_size_mb = os.path.getsize(archive_location) / (1024 * 1024)
            compression_ratio = original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 1.0
            
            # Calculate checksum
            checksum = await self._calculate_file_checksum(archive_location)
            
            # Delete original data if policy action is DELETE
            if policy.action == RetentionAction.DELETE:
                delete_api = self.influxdb_client.delete_api()
                await delete_api.delete(
                    start="1970-01-01T00:00:00Z",
                    stop=cutoff_date.isoformat() + "Z",
                    predicate=f'_measurement="{job_record["table_name"]}"',
                    bucket=bucket,
                    org=self.influxdb_org
                )
                logger.info("Deleted original InfluxDB data after archiving", 
                           measurement=job_record['table_name'])
            
            # Create archive metadata
            archive_metadata = ArchiveMetadata(
                archive_id=str(uuid.uuid4()),
                original_database=job_record['database_name'],
                original_table=job_record['table_name'],
                archive_location=archive_location,
                compression_type=CompressionType(job_record['compression_type']),
                original_size_mb=original_size_mb,
                compressed_size_mb=compressed_size_mb,
                compression_ratio=compression_ratio,
                record_count=len(result),
                date_range_start=job_record['start_date'],
                date_range_end=job_record['end_date'],
                archive_timestamp=datetime.utcnow(),
                checksum=checksum,
                retrieval_instructions=f"Use pandas.read_parquet('{archive_location}') to restore data"
            )
            
            return archive_metadata
            
        except Exception as e:
            logger.error("Failed to archive InfluxDB data", error=str(e))
            raise
    
    async def _create_archive_file(self, 
                                 df: pd.DataFrame,
                                 target_location: str,
                                 compression_type: CompressionType) -> str:
        """Create compressed archive file from DataFrame"""
        try:
            # Ensure target directory exists
            target_path = Path(target_location)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine file format and compression
            if compression_type == CompressionType.NONE:
                archive_file = f"{target_location}.parquet"
                df.to_parquet(archive_file, compression=None)
            
            elif compression_type == CompressionType.GZIP:
                archive_file = f"{target_location}.parquet.gz"
                df.to_parquet(archive_file, compression='gzip')
            
            elif compression_type == CompressionType.BZIP2:
                archive_file = f"{target_location}.parquet.bz2"
                df.to_parquet(archive_file, compression='brotli')
            
            elif compression_type == CompressionType.LZ4:
                archive_file = f"{target_location}.parquet.lz4"
                df.to_parquet(archive_file, compression='lz4')
            
            elif compression_type == CompressionType.ZSTD:
                archive_file = f"{target_location}.parquet.zstd"
                df.to_parquet(archive_file, compression='zstd')
            
            else:
                # Default to gzip
                archive_file = f"{target_location}.parquet.gz"
                df.to_parquet(archive_file, compression='gzip')
            
            # Upload to cloud storage if configured
            if self.s3_config:
                cloud_location = await self._upload_to_s3(archive_file)
                return cloud_location
            elif self.azure_config:
                cloud_location = await self._upload_to_azure(archive_file)
                return cloud_location
            
            return archive_file
            
        except Exception as e:
            logger.error("Failed to create archive file", error=str(e))
            raise
    
    async def _upload_to_s3(self, local_file_path: str) -> str:
        """Upload archive file to AWS S3"""
        try:
            bucket_name = self.s3_config.get('bucket_name')
            if not bucket_name:
                raise ValueError("S3 bucket name not configured")
            
            # Generate S3 key
            file_name = Path(local_file_path).name
            s3_key = f"geological_archives/{datetime.utcnow().strftime('%Y/%m/%d')}/{file_name}"
            
            # Upload file
            self.s3_client.upload_file(local_file_path, bucket_name, s3_key)
            
            # Set storage class based on tier
            storage_class = 'STANDARD'  # Default
            # Could be configured based on storage tier
            
            s3_location = f"s3://{bucket_name}/{s3_key}"
            
            # Remove local file after successful upload
            os.remove(local_file_path)
            
            logger.info("Uploaded archive to S3", 
                       s3_location=s3_location)
            
            return s3_location
            
        except Exception as e:
            logger.error("Failed to upload to S3", error=str(e))
            raise
    
    async def _upload_to_azure(self, local_file_path: str) -> str:
        """Upload archive file to Azure Blob Storage"""
        try:
            container_name = self.azure_config.get('container_name')
            if not container_name:
                raise ValueError("Azure container name not configured")
            
            # Generate blob name
            file_name = Path(local_file_path).name
            blob_name = f"geological_archives/{datetime.utcnow().strftime('%Y/%m/%d')}/{file_name}"
            
            # Upload file
            blob_client = self.azure_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            with open(local_file_path, 'rb') as data:
                await blob_client.upload_blob(data, overwrite=True)
            
            azure_location = f"azure://{container_name}/{blob_name}"
            
            # Remove local file after successful upload
            os.remove(local_file_path)
            
            logger.info("Uploaded archive to Azure", 
                       azure_location=azure_location)
            
            return azure_location
            
        except Exception as e:
            logger.error("Failed to upload to Azure", error=str(e))
            raise
    
    async def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file"""
        try:
            import hashlib
            
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            return sha256_hash.hexdigest()
            
        except Exception as e:
            logger.error("Failed to calculate checksum", error=str(e))
            return ""
    
    async def _store_archive_metadata(self, metadata: ArchiveMetadata) -> None:
        """Store archive metadata in database"""
        try:
            await self.postgres_conn.execute("""
                INSERT INTO archive_metadata 
                (archive_id, original_database, original_table, archive_location,
                 compression_type, original_size_mb, compressed_size_mb, compression_ratio,
                 record_count, date_range_start, date_range_end, checksum, retrieval_instructions)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                metadata.archive_id, metadata.original_database, metadata.original_table,
                metadata.archive_location, metadata.compression_type.value,
                metadata.original_size_mb, metadata.compressed_size_mb, metadata.compression_ratio,
                metadata.record_count, metadata.date_range_start, metadata.date_range_end,
                metadata.checksum, metadata.retrieval_instructions
            )
            
        except Exception as e:
            logger.error("Failed to store archive metadata", error=str(e))
            raise
    
    async def _log_lifecycle_event(self,
                                 database_name: str,
                                 table_name: str,
                                 event_type: str,
                                 records_affected: int,
                                 size_mb: float,
                                 details: Dict[str, Any],
                                 policy_id: Optional[str] = None) -> None:
        """Log data lifecycle event"""
        try:
            await self.postgres_conn.execute("""
                INSERT INTO data_lifecycle_events 
                (database_name, table_name, event_type, records_affected, size_mb, details, policy_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                database_name, table_name, event_type, records_affected,
                size_mb, json.dumps(details), policy_id
            )
            
        except Exception as e:
            logger.error("Failed to log lifecycle event", error=str(e))
    
    async def restore_archived_data(self, 
                                  archive_id: str,
                                  target_table: Optional[str] = None) -> Dict[str, Any]:
        """Restore data from archive"""
        try:
            # Get archive metadata
            metadata_record = await self.postgres_conn.fetchrow("""
                SELECT * FROM archive_metadata WHERE archive_id = $1
            """, archive_id)
            
            if not metadata_record:
                raise ValueError(f"Archive {archive_id} not found")
            
            # Download from cloud storage if needed
            local_file_path = await self._download_archive_file(
                metadata_record['archive_location']
            )
            
            # Verify checksum
            calculated_checksum = await self._calculate_file_checksum(local_file_path)
            if calculated_checksum != metadata_record['checksum']:
                raise ValueError("Archive file checksum mismatch - file may be corrupted")
            
            # Load data
            df = pd.read_parquet(local_file_path)
            
            # Restore to database if target table specified
            if target_table:
                await self._restore_to_database(
                    df,
                    metadata_record['original_database'],
                    target_table
                )
            
            # Log restoration event
            await self._log_lifecycle_event(
                database_name=metadata_record['original_database'],
                table_name=target_table or metadata_record['original_table'],
                event_type='restored',
                records_affected=len(df),
                size_mb=metadata_record['original_size_mb'],
                details={
                    'archive_id': archive_id,
                    'archive_location': metadata_record['archive_location'],
                    'restoration_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Cleanup local file
            if local_file_path != metadata_record['archive_location']:
                os.remove(local_file_path)
            
            logger.info("Archive data restored successfully", 
                       archive_id=archive_id,
                       records_restored=len(df),
                       target_table=target_table)
            
            return {
                'archive_id': archive_id,
                'records_restored': len(df),
                'target_table': target_table,
                'restoration_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to restore archived data", 
                        archive_id=archive_id,
                        error=str(e))
            raise
    
    async def _download_archive_file(self, archive_location: str) -> str:
        """Download archive file from cloud storage if needed"""
        try:
            if archive_location.startswith('s3://'):
                return await self._download_from_s3(archive_location)
            elif archive_location.startswith('azure://'):
                return await self._download_from_azure(archive_location)
            else:
                # Local file
                return archive_location
            
        except Exception as e:
            logger.error("Failed to download archive file", error=str(e))
            raise
    
    async def _download_from_s3(self, s3_location: str) -> str:
        """Download file from S3"""
        try:
            # Parse S3 location
            s3_parts = s3_location.replace('s3://', '').split('/', 1)
            bucket_name = s3_parts[0]
            s3_key = s3_parts[1]
            
            # Create local file path
            local_file_path = self.local_archive_path / f"temp_{uuid.uuid4().hex}.parquet"
            
            # Download file
            self.s3_client.download_file(bucket_name, s3_key, str(local_file_path))
            
            return str(local_file_path)
            
        except Exception as e:
            logger.error("Failed to download from S3", error=str(e))
            raise
    
    async def _download_from_azure(self, azure_location: str) -> str:
        """Download file from Azure Blob Storage"""
        try:
            # Parse Azure location
            azure_parts = azure_location.replace('azure://', '').split('/', 1)
            container_name = azure_parts[0]
            blob_name = azure_parts[1]
            
            # Create local file path
            local_file_path = self.local_archive_path / f"temp_{uuid.uuid4().hex}.parquet"
            
            # Download file
            blob_client = self.azure_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            with open(local_file_path, 'wb') as download_file:
                download_stream = await blob_client.download_blob()
                async for chunk in download_stream.chunks():
                    download_file.write(chunk)
            
            return str(local_file_path)
            
        except Exception as e:
            logger.error("Failed to download from Azure", error=str(e))
            raise
    
    async def _restore_to_database(self, 
                                 df: pd.DataFrame,
                                 database_name: str,
                                 table_name: str) -> None:
        """Restore DataFrame to database"""
        try:
            if database_name == 'postgresql':
                # Convert DataFrame to records and insert
                records = df.to_dict('records')
                
                if records:
                    # Generate insert query
                    columns = list(df.columns)
                    placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
                    
                    insert_query = f"""
                        INSERT INTO {table_name} ({', '.join(columns)})
                        VALUES ({placeholders})
                    """
                    
                    # Execute batch insert
                    values_list = [list(record.values()) for record in records]
                    await self.postgres_conn.executemany(insert_query, values_list)
            
            elif database_name == 'influxdb':
                # Convert DataFrame to InfluxDB line protocol and write
                write_api = self.influxdb_client.write_api()
                
                # This would need proper conversion to InfluxDB format
                # Implementation depends on DataFrame structure
                pass
            
        except Exception as e:
            logger.error("Failed to restore to database", error=str(e))
            raise
    
    async def get_archiving_statistics(self) -> Dict[str, Any]:
        """Get comprehensive archiving statistics"""
        try:
            # Get recent archive jobs
            recent_jobs = await self.postgres_conn.fetch("""
                SELECT status, COUNT(*) as count
                FROM archive_jobs
                WHERE created_timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY status
            """)
            
            # Get storage usage by location
            storage_usage = await self.postgres_conn.fetch("""
                SELECT 
                    CASE 
                        WHEN archive_location LIKE 's3://%' THEN 'S3'
                        WHEN archive_location LIKE 'azure://%' THEN 'Azure'
                        ELSE 'Local'
                    END as storage_type,
                    COUNT(*) as archive_count,
                    SUM(compressed_size_mb) as total_size_mb,
                    AVG(compression_ratio) as avg_compression_ratio
                FROM archive_metadata
                GROUP BY storage_type
            """)
            
            # Get retention policy effectiveness
            policy_stats = await self.postgres_conn.fetch("""
                SELECT 
                    rp.policy_name,
                    COUNT(aj.job_id) as jobs_executed,
                    SUM(aj.estimated_records) as total_records_processed,
                    AVG(EXTRACT(EPOCH FROM (aj.completed_timestamp - aj.started_timestamp))/60) as avg_duration_minutes
                FROM retention_policies rp
                LEFT JOIN archive_jobs aj ON rp.policy_id = aj.policy_id
                WHERE rp.enabled = TRUE
                GROUP BY rp.policy_id, rp.policy_name
            """)
            
            statistics = {
                'current_stats': self.archiving_stats,
                'recent_jobs': {
                    record['status']: record['count']
                    for record in recent_jobs
                },
                'storage_usage': [
                    {
                        'storage_type': record['storage_type'],
                        'archive_count': record['archive_count'],
                        'total_size_mb': float(record['total_size_mb']) if record['total_size_mb'] else 0,
                        'avg_compression_ratio': float(record['avg_compression_ratio']) if record['avg_compression_ratio'] else 0
                    }
                    for record in storage_usage
                ],
                'policy_effectiveness': [
                    {
                        'policy_name': record['policy_name'],
                        'jobs_executed': record['jobs_executed'] or 0,
                        'total_records_processed': record['total_records_processed'] or 0,
                        'avg_duration_minutes': float(record['avg_duration_minutes']) if record['avg_duration_minutes'] else 0
                    }
                    for record in policy_stats
                ],
                'active_jobs': len(self.active_jobs),
                'statistics_timestamp': datetime.utcnow().isoformat()
            }
            
            return statistics
            
        except Exception as e:
            logger.error("Failed to get archiving statistics", error=str(e))
            return {}


# Example usage and configuration
async def example_data_archiving():
    """Example of comprehensive data archiving setup"""
    
    # Initialize archiving manager
    archiving_manager = DataArchivingManager(
        postgres_connection_string='postgresql://prometheus:prometheus123@localhost:5432/prometheus',
        influxdb_url='http://localhost:8086',
        influxdb_token='prometheus-token',
        influxdb_org='prometheus',
        redis_url='redis://localhost:6379',
        s3_config={
            'access_key_id': 'your_access_key',
            'secret_access_key': 'your_secret_key',
            'bucket_name': 'geological-data-archive',
            'region': 'us-east-1'
        },
        local_archive_path='/data/archives'
    )
    
    await archiving_manager.initialize()
    
    try:
        # Create retention policies for different data types
        
        # Seismic data: Archive after 1 year, compress with gzip
        seismic_policy_id = await archiving_manager.create_retention_policy(
            policy_name="Seismic Data Long-term Archival",
            database_name="postgresql",
            table_name="seismic_events",
            retention_period_days=365,
            action=RetentionAction.ARCHIVE,
            storage_tier=StorageTier.COLD,
            compression_type=CompressionType.GZIP,
            conditions={
                'timestamp_column': 'timestamp',
                'where_clause': 'magnitude < 3.0'  # Only archive smaller events
            },
            created_by="data_admin"
        )
        
        # MT-SAR data: Archive after 6 months, high compression
        mtsar_policy_id = await archiving_manager.create_retention_policy(
            policy_name="MT-SAR Interferogram Archival",
            database_name="postgresql",
            table_name="mtsar_interferograms",
            retention_period_days=180,
            action=RetentionAction.ARCHIVE,
            storage_tier=StorageTier.COLD,
            compression_type=CompressionType.ZSTD,
            conditions={
                'timestamp_column': 'acquisition_date',
                'where_clause': 'average_coherence < 0.5'  # Archive low-quality data
            },
            created_by="data_admin"
        )
        
        # Geochemical data: Keep for 5 years, then deep archive
        geochem_policy_id = await archiving_manager.create_retention_policy(
            policy_name="Geochemical Data Deep Archive",
            database_name="postgresql",
            table_name="geochemical_analyses",
            retention_period_days=1825,  # 5 years
            action=RetentionAction.ARCHIVE,
            storage_tier=StorageTier.DEEP_ARCHIVE,
            compression_type=CompressionType.BZIP2,
            conditions={
                'timestamp_column': 'collection_date'
            },
            created_by="data_admin"
        )
        
        # Execute retention policies
        execution_results = await archiving_manager.execute_retention_policies()
        
        # Get archiving statistics
        statistics = await archiving_manager.get_archiving_statistics()
        
        logger.info("Data archiving example completed",
                   policies_created=3,
                   execution_results=execution_results,
                   statistics=statistics)
        
    finally:
        await archiving_manager.close()


if __name__ == "__main__":
    asyncio.run(example_data_archiving())