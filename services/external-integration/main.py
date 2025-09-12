#!/usr/bin/env python3
"""
Prometheus AI Platform - External System Integration Framework
Advanced integration with external systems, APIs, and cloud services
"""

import asyncio
import json
import logging
import uuid
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from pydantic import BaseModel, Field, validator
import jwt
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, Float, LargeBinary, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import aiohttp
import aiofiles
from google.cloud import storage as gcs
from google.cloud import bigquery
from google.cloud import pubsub_v1
from google.oauth2 import service_account
import boto3
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import requests
from requests_oauthlib import OAuth2Session

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Database Models
Base = declarative_base()

class ExternalIntegration(Base):
    __tablename__ = "external_integrations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    integration_name = Column(String(255), nullable=False)
    integration_type = Column(String(50), nullable=False)  # api, storage, auth, gis, scientific
    provider = Column(String(100), nullable=False)  # google, aws, azure, arcgis, etc.
    configuration = Column(JSONB, nullable=False)
    credentials = Column(JSONB)  # Encrypted credentials
    status = Column(String(20), default="active")  # active, inactive, error
    created_by = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_used = Column(DateTime)
    usage_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    metadata = Column(JSONB)

class WebhookEndpoint(Base):
    __tablename__ = "webhook_endpoints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    endpoint_name = Column(String(255), nullable=False)
    url = Column(String(500), nullable=False)
    secret_key = Column(String(255))
    event_types = Column(JSONB)  # List of event types to send
    headers = Column(JSONB)  # Custom headers
    retry_policy = Column(JSONB)
    is_active = Column(Boolean, default=True)
    created_by = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_triggered = Column(DateTime)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)

class DataExportJob(Base):
    __tablename__ = "data_export_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_name = Column(String(255), nullable=False)
    export_type = Column(String(50), nullable=False)  # csv, json, geojson, shapefile, etc.
    data_source = Column(String(100), nullable=False)
    filters = Column(JSONB)
    destination = Column(JSONB)  # Export destination configuration
    schedule = Column(JSONB)  # Cron-like schedule
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    created_by = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    file_path = Column(String(500))
    file_size = Column(Integer)
    record_count = Column(Integer)
    error_message = Column(Text)

class APIGatewayLog(Base):
    __tablename__ = "api_gateway_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)
    path = Column(String(500), nullable=False)
    query_params = Column(JSONB)
    headers = Column(JSONB)
    request_body = Column(Text)
    response_status = Column(Integer)
    response_body = Column(Text)
    response_time_ms = Column(Float)
    user_id = Column(UUID(as_uuid=True))
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    timestamp = Column(DateTime, default=datetime.utcnow)
    integration_id = Column(UUID(as_uuid=True), ForeignKey("external_integrations.id"))

# Pydantic Models
class IntegrationConfig(BaseModel):
    """Base integration configuration"""
    integration_name: str
    integration_type: str
    provider: str
    configuration: Dict[str, Any]
    credentials: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WebhookConfig(BaseModel):
    """Webhook configuration"""
    endpoint_name: str
    url: str
    secret_key: Optional[str] = None
    event_types: List[str] = Field(default_factory=list)
    headers: Dict[str, str] = Field(default_factory=dict)
    retry_policy: Dict[str, Any] = Field(default_factory=dict)

class ExportJobConfig(BaseModel):
    """Data export job configuration"""
    job_name: str
    export_type: str
    data_source: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    destination: Dict[str, Any]
    schedule: Optional[Dict[str, Any]] = None

class CloudStorageConfig(BaseModel):
    """Cloud storage configuration"""
    provider: str  # gcs, s3, azure
    bucket_name: str
    credentials: Dict[str, Any]
    region: Optional[str] = None
    encryption: Optional[Dict[str, Any]] = None

class AuthenticationConfig(BaseModel):
    """Authentication configuration"""
    auth_type: str  # oauth2, jwt, api_key, saml
    provider: str  # google, azure, aws, custom
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    redirect_uri: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    additional_params: Dict[str, Any] = Field(default_factory=dict)

class GISIntegrationConfig(BaseModel):
    """GIS system integration configuration"""
    system_type: str  # arcgis, qgis, google_earth_engine
    server_url: str
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    services: List[str] = Field(default_factory=list)

class ScientificSoftwareConfig(BaseModel):
    """Scientific software integration configuration"""
    software_type: str  # matlab, r, python, colab
    execution_environment: str  # local, cloud, container
    credentials: Optional[Dict[str, Any]] = None
    resource_limits: Dict[str, Any] = Field(default_factory=dict)

# Enums
class IntegrationType(Enum):
    API_GATEWAY = "api_gateway"
    CLOUD_STORAGE = "cloud_storage"
    AUTHENTICATION = "authentication"
    GIS_SYSTEM = "gis_system"
    SCIENTIFIC_SOFTWARE = "scientific_software"
    DATABASE = "database"
    MESSAGING = "messaging"
    MONITORING = "monitoring"

class ExportFormat(Enum):
    CSV = "csv"
    JSON = "json"
    GEOJSON = "geojson"
    SHAPEFILE = "shapefile"
    KML = "kml"
    NETCDF = "netcdf"
    HDF5 = "hdf5"
    PARQUET = "parquet"

# Advanced Integration Engine
class ExternalIntegrationEngine:
    """Advanced external system integration engine"""
    
    def __init__(self, redis_client: redis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.integrations: Dict[str, Any] = {}
        self.webhook_handlers: Dict[str, Callable] = {}
        
    async def register_integration(self, config: IntegrationConfig, user_id: str) -> ExternalIntegration:
        """Register new external integration"""
        try:
            # Encrypt credentials if provided
            encrypted_credentials = None
            if config.credentials:
                encrypted_credentials = await self._encrypt_credentials(config.credentials)
            
            integration = ExternalIntegration(
                integration_name=config.integration_name,
                integration_type=config.integration_type,
                provider=config.provider,
                configuration=config.configuration,
                credentials=encrypted_credentials,
                created_by=uuid.UUID(user_id),
                metadata=config.metadata
            )
            
            self.db.add(integration)
            await self.db.commit()
            await self.db.refresh(integration)
            
            # Initialize integration client
            await self._initialize_integration_client(integration)
            
            logger.info("Integration registered", 
                       integration_id=str(integration.id),
                       integration_type=config.integration_type,
                       provider=config.provider)
            
            return integration
            
        except Exception as e:
            logger.error("Failed to register integration", error=str(e))
            raise
    
    async def create_webhook_endpoint(self, config: WebhookConfig, user_id: str) -> WebhookEndpoint:
        """Create webhook endpoint"""
        try:
            webhook = WebhookEndpoint(
                endpoint_name=config.endpoint_name,
                url=config.url,
                secret_key=config.secret_key,
                event_types=config.event_types,
                headers=config.headers,
                retry_policy=config.retry_policy,
                created_by=uuid.UUID(user_id)
            )
            
            self.db.add(webhook)
            await self.db.commit()
            await self.db.refresh(webhook)
            
            logger.info("Webhook endpoint created", 
                       webhook_id=str(webhook.id),
                       endpoint_name=config.endpoint_name)
            
            return webhook
            
        except Exception as e:
            logger.error("Failed to create webhook", error=str(e))
            raise
    
    async def trigger_webhook(self, event_type: str, data: Dict[str, Any]):
        """Trigger webhooks for event type"""
        try:
            # Get active webhooks for event type
            from sqlalchemy import select
            query = select(WebhookEndpoint).where(
                WebhookEndpoint.is_active == True,
                WebhookEndpoint.event_types.contains([event_type])
            )
            result = await self.db.execute(query)
            webhooks = result.scalars().all()
            
            # Trigger webhooks asynchronously
            tasks = []
            for webhook in webhooks:
                task = asyncio.create_task(self._send_webhook(webhook, event_type, data))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error("Failed to trigger webhooks", event_type=event_type, error=str(e))
    
    async def export_data(self, config: ExportJobConfig, user_id: str) -> DataExportJob:
        """Create and execute data export job"""
        try:
            export_job = DataExportJob(
                job_name=config.job_name,
                export_type=config.export_type,
                data_source=config.data_source,
                filters=config.filters,
                destination=config.destination,
                schedule=config.schedule,
                created_by=uuid.UUID(user_id)
            )
            
            self.db.add(export_job)
            await self.db.commit()
            await self.db.refresh(export_job)
            
            # Execute export job
            await self._execute_export_job(export_job)
            
            logger.info("Export job created", 
                       job_id=str(export_job.id),
                       export_type=config.export_type)
            
            return export_job
            
        except Exception as e:
            logger.error("Failed to create export job", error=str(e))
            raise
    
    async def authenticate_external_service(self, integration_id: str, auth_config: AuthenticationConfig) -> Dict[str, Any]:
        """Authenticate with external service"""
        try:
            if auth_config.auth_type == "oauth2":
                return await self._oauth2_authentication(auth_config)
            elif auth_config.auth_type == "jwt":
                return await self._jwt_authentication(auth_config)
            elif auth_config.auth_type == "api_key":
                return await self._api_key_authentication(auth_config)
            else:
                raise ValueError(f"Unsupported authentication type: {auth_config.auth_type}")
                
        except Exception as e:
            logger.error("Authentication failed", 
                        integration_id=integration_id,
                        auth_type=auth_config.auth_type,
                        error=str(e))
            raise
    
    async def sync_with_gis_system(self, integration_id: str, gis_config: GISIntegrationConfig) -> Dict[str, Any]:
        """Synchronize data with GIS system"""
        try:
            if gis_config.system_type == "arcgis":
                return await self._sync_with_arcgis(gis_config)
            elif gis_config.system_type == "qgis":
                return await self._sync_with_qgis(gis_config)
            elif gis_config.system_type == "google_earth_engine":
                return await self._sync_with_google_earth_engine(gis_config)
            else:
                raise ValueError(f"Unsupported GIS system: {gis_config.system_type}")
                
        except Exception as e:
            logger.error("GIS synchronization failed", 
                        integration_id=integration_id,
                        system_type=gis_config.system_type,
                        error=str(e))
            raise
    
    async def execute_scientific_software(self, integration_id: str, software_config: ScientificSoftwareConfig, script: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute script in scientific software environment"""
        try:
            if software_config.software_type == "matlab":
                return await self._execute_matlab_script(software_config, script, data)
            elif software_config.software_type == "r":
                return await self._execute_r_script(software_config, script, data)
            elif software_config.software_type == "python":
                return await self._execute_python_script(software_config, script, data)
            elif software_config.software_type == "colab":
                return await self._execute_colab_notebook(software_config, script, data)
            else:
                raise ValueError(f"Unsupported software type: {software_config.software_type}")
                
        except Exception as e:
            logger.error("Scientific software execution failed", 
                        integration_id=integration_id,
                        software_type=software_config.software_type,
                        error=str(e))
            raise
    
    async def upload_to_cloud_storage(self, storage_config: CloudStorageConfig, file_path: str, destination_path: str) -> Dict[str, Any]:
        """Upload file to cloud storage"""
        try:
            if storage_config.provider == "gcs":
                return await self._upload_to_gcs(storage_config, file_path, destination_path)
            elif storage_config.provider == "s3":
                return await self._upload_to_s3(storage_config, file_path, destination_path)
            elif storage_config.provider == "azure":
                return await self._upload_to_azure(storage_config, file_path, destination_path)
            else:
                raise ValueError(f"Unsupported storage provider: {storage_config.provider}")
                
        except Exception as e:
            logger.error("Cloud storage upload failed", 
                        provider=storage_config.provider,
                        error=str(e))
            raise
    
    async def query_bigquery(self, integration_id: str, query: str) -> Dict[str, Any]:
        """Execute BigQuery query"""
        try:
            # Get integration configuration
            integration = await self._get_integration(integration_id)
            if not integration or integration.provider != "google":
                raise ValueError("Invalid BigQuery integration")
            
            # Initialize BigQuery client
            credentials = await self._decrypt_credentials(integration.credentials)
            client = bigquery.Client.from_service_account_info(credentials)
            
            # Execute query
            query_job = client.query(query)
            results = query_job.result()
            
            # Convert results to list of dictionaries
            rows = []
            for row in results:
                rows.append(dict(row))
            
            return {
                "rows": rows,
                "total_rows": results.total_rows,
                "job_id": query_job.job_id,
                "bytes_processed": query_job.total_bytes_processed
            }
            
        except Exception as e:
            logger.error("BigQuery execution failed", 
                        integration_id=integration_id,
                        error=str(e))
            raise
    
    async def publish_to_pubsub(self, integration_id: str, topic: str, message: Dict[str, Any]) -> str:
        """Publish message to Google Cloud Pub/Sub"""
        try:
            # Get integration configuration
            integration = await self._get_integration(integration_id)
            if not integration or integration.provider != "google":
                raise ValueError("Invalid Pub/Sub integration")
            
            # Initialize Pub/Sub client
            credentials = await self._decrypt_credentials(integration.credentials)
            publisher = pubsub_v1.PublisherClient(credentials=service_account.Credentials.from_service_account_info(credentials))
            
            # Publish message
            topic_path = publisher.topic_path(credentials["project_id"], topic)
            message_data = json.dumps(message).encode("utf-8")
            
            future = publisher.publish(topic_path, message_data)
            message_id = future.result()
            
            logger.info("Message published to Pub/Sub", 
                       topic=topic,
                       message_id=message_id)
            
            return message_id
            
        except Exception as e:
            logger.error("Pub/Sub publish failed", 
                        integration_id=integration_id,
                        topic=topic,
                        error=str(e))
            raise
    
    # Private helper methods
    async def _initialize_integration_client(self, integration: ExternalIntegration):
        """Initialize integration client"""
        try:
            client_key = f"{integration.integration_type}:{integration.provider}:{integration.id}"
            
            if integration.integration_type == "cloud_storage":
                if integration.provider == "google":
                    credentials = await self._decrypt_credentials(integration.credentials)
                    client = gcs.Client.from_service_account_info(credentials)
                    self.integrations[client_key] = client
                elif integration.provider == "aws":
                    credentials = await self._decrypt_credentials(integration.credentials)
                    client = boto3.client('s3', **credentials)
                    self.integrations[client_key] = client
                elif integration.provider == "azure":
                    credentials = await self._decrypt_credentials(integration.credentials)
                    client = BlobServiceClient(account_url=credentials["account_url"], credential=credentials["credential"])
                    self.integrations[client_key] = client
            
        except Exception as e:
            logger.warning("Failed to initialize integration client", 
                          integration_id=str(integration.id),
                          error=str(e))
    
    async def _encrypt_credentials(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt credentials for storage"""
        # In production, use proper encryption (e.g., Google Cloud KMS, AWS KMS, Azure Key Vault)
        # For demo, we'll use base64 encoding (NOT secure for production)
        encrypted = {}
        for key, value in credentials.items():
            if isinstance(value, str):
                encrypted[key] = base64.b64encode(value.encode()).decode()
            else:
                encrypted[key] = value
        return encrypted
    
    async def _decrypt_credentials(self, encrypted_credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt credentials for use"""
        # In production, use proper decryption
        # For demo, we'll use base64 decoding
        decrypted = {}
        for key, value in encrypted_credentials.items():
            if isinstance(value, str):
                try:
                    decrypted[key] = base64.b64decode(value.encode()).decode()
                except:
                    decrypted[key] = value
            else:
                decrypted[key] = value
        return decrypted
    
    async def _send_webhook(self, webhook: WebhookEndpoint, event_type: str, data: Dict[str, Any]):
        """Send webhook notification"""
        try:
            payload = {
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            headers = webhook.headers.copy() if webhook.headers else {}
            headers["Content-Type"] = "application/json"
            
            if webhook.secret_key:
                # Add signature for verification
                signature = hashlib.sha256(
                    (webhook.secret_key + json.dumps(payload)).encode()
                ).hexdigest()
                headers["X-Prometheus-Signature"] = signature
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        webhook.success_count += 1
                        webhook.last_triggered = datetime.utcnow()
                    else:
                        webhook.failure_count += 1
                        logger.warning("Webhook failed", 
                                     webhook_id=str(webhook.id),
                                     status=response.status)
            
            await self.db.commit()
            
        except Exception as e:
            webhook.failure_count += 1
            await self.db.commit()
            logger.error("Webhook send failed", 
                        webhook_id=str(webhook.id),
                        error=str(e))
    
    async def _execute_export_job(self, export_job: DataExportJob):
        """Execute data export job"""
        try:
            export_job.status = "running"
            export_job.started_at = datetime.utcnow()
            await self.db.commit()
            
            # Get data based on source and filters
            data = await self._get_export_data(export_job.data_source, export_job.filters)
            
            # Export data in specified format
            file_path = await self._export_data_to_format(data, export_job.export_type, export_job.id)
            
            # Upload to destination if specified
            if export_job.destination:
                await self._upload_export_file(file_path, export_job.destination)
            
            # Update job status
            export_job.status = "completed"
            export_job.completed_at = datetime.utcnow()
            export_job.file_path = str(file_path)
            export_job.file_size = file_path.stat().st_size if file_path.exists() else 0
            export_job.record_count = len(data)
            
            await self.db.commit()
            
        except Exception as e:
            export_job.status = "failed"
            export_job.error_message = str(e)
            await self.db.commit()
            logger.error("Export job failed", 
                        job_id=str(export_job.id),
                        error=str(e))
    
    async def _oauth2_authentication(self, auth_config: AuthenticationConfig) -> Dict[str, Any]:
        """Handle OAuth2 authentication"""
        # Implementation would handle OAuth2 flow
        return {"access_token": "dummy_token", "token_type": "Bearer"}
    
    async def _jwt_authentication(self, auth_config: AuthenticationConfig) -> Dict[str, Any]:
        """Handle JWT authentication"""
        # Implementation would handle JWT token generation/validation
        return {"token": "dummy_jwt_token"}
    
    async def _api_key_authentication(self, auth_config: AuthenticationConfig) -> Dict[str, Any]:
        """Handle API key authentication"""
        # Implementation would handle API key validation
        return {"api_key": auth_config.additional_params.get("api_key")}
    
    async def _sync_with_arcgis(self, gis_config: GISIntegrationConfig) -> Dict[str, Any]:
        """Synchronize with ArcGIS system"""
        # Implementation would handle ArcGIS REST API integration
        return {"status": "synced", "features_updated": 0}
    
    async def _sync_with_qgis(self, gis_config: GISIntegrationConfig) -> Dict[str, Any]:
        """Synchronize with QGIS system"""
        # Implementation would handle QGIS server integration
        return {"status": "synced", "layers_updated": 0}
    
    async def _sync_with_google_earth_engine(self, gis_config: GISIntegrationConfig) -> Dict[str, Any]:
        """Synchronize with Google Earth Engine"""
        # Implementation would handle Google Earth Engine API
        return {"status": "synced", "assets_updated": 0}
    
    async def _execute_matlab_script(self, software_config: ScientificSoftwareConfig, script: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MATLAB script"""
        # Implementation would handle MATLAB execution
        return {"status": "completed", "results": {}}
    
    async def _execute_r_script(self, software_config: ScientificSoftwareConfig, script: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute R script"""
        # Implementation would handle R execution
        return {"status": "completed", "results": {}}
    
    async def _execute_python_script(self, software_config: ScientificSoftwareConfig, script: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python script"""
        # Implementation would handle Python execution in isolated environment
        return {"status": "completed", "results": {}}
    
    async def _execute_colab_notebook(self, software_config: ScientificSoftwareConfig, script: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Google Colab notebook"""
        # Implementation would handle Google Colab API
        return {"status": "completed", "results": {}}
    
    async def _upload_to_gcs(self, storage_config: CloudStorageConfig, file_path: str, destination_path: str) -> Dict[str, Any]:
        """Upload file to Google Cloud Storage"""
        try:
            credentials = storage_config.credentials
            client = gcs.Client.from_service_account_info(credentials)
            bucket = client.bucket(storage_config.bucket_name)
            blob = bucket.blob(destination_path)
            
            blob.upload_from_filename(file_path)
            
            return {
                "provider": "gcs",
                "bucket": storage_config.bucket_name,
                "path": destination_path,
                "url": f"gs://{storage_config.bucket_name}/{destination_path}"
            }
            
        except Exception as e:
            logger.error("GCS upload failed", error=str(e))
            raise
    
    async def _upload_to_s3(self, storage_config: CloudStorageConfig, file_path: str, destination_path: str) -> Dict[str, Any]:
        """Upload file to AWS S3"""
        try:
            credentials = storage_config.credentials
            s3_client = boto3.client('s3', **credentials)
            
            s3_client.upload_file(file_path, storage_config.bucket_name, destination_path)
            
            return {
                "provider": "s3",
                "bucket": storage_config.bucket_name,
                "path": destination_path,
                "url": f"s3://{storage_config.bucket_name}/{destination_path}"
            }
            
        except Exception as e:
            logger.error("S3 upload failed", error=str(e))
            raise
    
    async def _upload_to_azure(self, storage_config: CloudStorageConfig, file_path: str, destination_path: str) -> Dict[str, Any]:
        """Upload file to Azure Blob Storage"""
        try:
            credentials = storage_config.credentials
            blob_service_client = BlobServiceClient(
                account_url=credentials["account_url"],
                credential=credentials["credential"]
            )
            
            blob_client = blob_service_client.get_blob_client(
                container=storage_config.bucket_name,
                blob=destination_path
            )
            
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            return {
                "provider": "azure",
                "container": storage_config.bucket_name,
                "path": destination_path,
                "url": f"https://{credentials['account_name']}.blob.core.windows.net/{storage_config.bucket_name}/{destination_path}"
            }
            
        except Exception as e:
            logger.error("Azure upload failed", error=str(e))
            raise
    
    async def _get_integration(self, integration_id: str) -> Optional[ExternalIntegration]:
        """Get integration by ID"""
        from sqlalchemy import select
        query = select(ExternalIntegration).where(ExternalIntegration.id == uuid.UUID(integration_id))
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def _get_export_data(self, data_source: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get data for export based on source and filters"""
        # Implementation would query appropriate data source
        return [{"sample": "data"}]  # Placeholder
    
    async def _export_data_to_format(self, data: List[Dict[str, Any]], export_format: str, job_id: uuid.UUID) -> Path:
        """Export data to specified format"""
        export_dir = Path("/app/data/exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = export_dir / f"{job_id}.{export_format}"
        
        if export_format == "json":
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        elif export_format == "csv":
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
        # Add more format handlers as needed
        
        return file_path
    
    async def _upload_export_file(self, file_path: Path, destination: Dict[str, Any]):
        """Upload export file to destination"""
        # Implementation would handle various destination types
        pass

# FastAPI Application
app = FastAPI(
    title="Prometheus External Integration Framework",
    description="Advanced external system integration and API gateway",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
redis_client = None
db_engine = None
SessionLocal = None
integration_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global redis_client, db_engine, SessionLocal, integration_engine
    
    # Initialize Redis
    redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
    
    # Initialize Database
    db_engine = create_async_engine("postgresql+asyncpg://prometheus:prometheus123@localhost/prometheus")
    SessionLocal = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    
    # Initialize integration engine
    async with SessionLocal() as session:
        integration_engine = ExternalIntegrationEngine(redis_client, session)
    
    logger.info("External Integration Framework started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
    if db_engine:
        await db_engine.dispose()
    
    logger.info("External Integration Framework shutdown complete")

# Dependency injection
async def get_db():
    async with SessionLocal() as session:
        yield session

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract user from JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, "secret", algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Health endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        await redis_client.ping()
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")

# Include API routes
from .api.integration_routes import router as integration_router
app.include_router(integration_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)