"""
External Integration API Routes
Advanced external system integration endpoints
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func
from pydantic import BaseModel, Field

from ..main import (
    get_db, get_current_user, ExternalIntegration, WebhookEndpoint, DataExportJob,
    IntegrationConfig, WebhookConfig, ExportJobConfig, CloudStorageConfig,
    AuthenticationConfig, GISIntegrationConfig, ScientificSoftwareConfig,
    integration_engine
)
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/integration", tags=["external-integration"])

# Request/Response Models
class IntegrationResponse(BaseModel):
    id: str
    integration_name: str
    integration_type: str
    provider: str
    status: str
    created_at: datetime
    updated_at: datetime
    last_used: Optional[datetime]
    usage_count: int
    error_count: int
    metadata: Dict[str, Any]

class WebhookResponse(BaseModel):
    id: str
    endpoint_name: str
    url: str
    event_types: List[str]
    is_active: bool
    created_at: datetime
    last_triggered: Optional[datetime]
    success_count: int
    failure_count: int

class ExportJobResponse(BaseModel):
    id: str
    job_name: str
    export_type: str
    data_source: str
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    file_size: Optional[int]
    record_count: Optional[int]
    error_message: Optional[str]

class AuthenticationRequest(BaseModel):
    integration_id: str
    auth_config: AuthenticationConfig

class GISIntegrationRequest(BaseModel):
    integration_id: str
    gis_config: GISIntegrationConfig
    operation: str  # sync, upload, download, query

class ScientificExecutionRequest(BaseModel):
    integration_id: str
    software_config: ScientificSoftwareConfig
    script: str
    input_data: Dict[str, Any]
    execution_params: Dict[str, Any] = Field(default_factory=dict)

class CloudStorageRequest(BaseModel):
    storage_config: CloudStorageConfig
    operation: str  # upload, download, list, delete
    source_path: Optional[str] = None
    destination_path: Optional[str] = None

class BigQueryRequest(BaseModel):
    integration_id: str
    query: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False

class PubSubRequest(BaseModel):
    integration_id: str
    topic: str
    message: Dict[str, Any]
    attributes: Dict[str, str] = Field(default_factory=dict)

class IntegrationAnalytics(BaseModel):
    total_integrations: int
    active_integrations: int
    integration_types: Dict[str, int]
    providers: Dict[str, int]
    total_webhooks: int
    active_webhooks: int
    webhook_success_rate: float
    total_exports: int
    successful_exports: int
    export_success_rate: float

# Integration Management
@router.post("/integrations", response_model=IntegrationResponse)
async def create_integration(
    config: IntegrationConfig,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create new external integration"""
    try:
        integration = await integration_engine.register_integration(config, current_user)
        
        return IntegrationResponse(
            id=str(integration.id),
            integration_name=integration.integration_name,
            integration_type=integration.integration_type,
            provider=integration.provider,
            status=integration.status,
            created_at=integration.created_at,
            updated_at=integration.updated_at,
            last_used=integration.last_used,
            usage_count=integration.usage_count,
            error_count=integration.error_count,
            metadata=integration.metadata or {}
        )
        
    except Exception as e:
        logger.error("Failed to create integration", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create integration")

@router.get("/integrations", response_model=List[IntegrationResponse])
async def list_integrations(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    integration_type: Optional[str] = Query(None),
    provider: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """List user's integrations"""
    try:
        query = select(ExternalIntegration).where(ExternalIntegration.created_by == UUID(current_user))
        
        if integration_type:
            query = query.where(ExternalIntegration.integration_type == integration_type)
        
        if provider:
            query = query.where(ExternalIntegration.provider == provider)
        
        if status:
            query = query.where(ExternalIntegration.status == status)
        
        query = query.offset(offset).limit(limit).order_by(desc(ExternalIntegration.updated_at))
        
        result = await db.execute(query)
        integrations = result.scalars().all()
        
        return [
            IntegrationResponse(
                id=str(integration.id),
                integration_name=integration.integration_name,
                integration_type=integration.integration_type,
                provider=integration.provider,
                status=integration.status,
                created_at=integration.created_at,
                updated_at=integration.updated_at,
                last_used=integration.last_used,
                usage_count=integration.usage_count,
                error_count=integration.error_count,
                metadata=integration.metadata or {}
            )
            for integration in integrations
        ]
        
    except Exception as e:
        logger.error("Failed to list integrations", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list integrations")

@router.get("/integrations/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    integration_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get integration details"""
    try:
        query = select(ExternalIntegration).where(
            and_(
                ExternalIntegration.id == UUID(integration_id),
                ExternalIntegration.created_by == UUID(current_user)
            )
        )
        result = await db.execute(query)
        integration = result.scalar_one_or_none()
        
        if not integration:
            raise HTTPException(status_code=404, detail="Integration not found")
        
        return IntegrationResponse(
            id=str(integration.id),
            integration_name=integration.integration_name,
            integration_type=integration.integration_type,
            provider=integration.provider,
            status=integration.status,
            created_at=integration.created_at,
            updated_at=integration.updated_at,
            last_used=integration.last_used,
            usage_count=integration.usage_count,
            error_count=integration.error_count,
            metadata=integration.metadata or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get integration", integration_id=integration_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get integration")

# Webhook Management
@router.post("/webhooks", response_model=WebhookResponse)
async def create_webhook(
    config: WebhookConfig,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create webhook endpoint"""
    try:
        webhook = await integration_engine.create_webhook_endpoint(config, current_user)
        
        return WebhookResponse(
            id=str(webhook.id),
            endpoint_name=webhook.endpoint_name,
            url=webhook.url,
            event_types=webhook.event_types or [],
            is_active=webhook.is_active,
            created_at=webhook.created_at,
            last_triggered=webhook.last_triggered,
            success_count=webhook.success_count,
            failure_count=webhook.failure_count
        )
        
    except Exception as e:
        logger.error("Failed to create webhook", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create webhook")

@router.get("/webhooks", response_model=List[WebhookResponse])
async def list_webhooks(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    is_active: Optional[bool] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """List user's webhooks"""
    try:
        query = select(WebhookEndpoint).where(WebhookEndpoint.created_by == UUID(current_user))
        
        if is_active is not None:
            query = query.where(WebhookEndpoint.is_active == is_active)
        
        query = query.offset(offset).limit(limit).order_by(desc(WebhookEndpoint.created_at))
        
        result = await db.execute(query)
        webhooks = result.scalars().all()
        
        return [
            WebhookResponse(
                id=str(webhook.id),
                endpoint_name=webhook.endpoint_name,
                url=webhook.url,
                event_types=webhook.event_types or [],
                is_active=webhook.is_active,
                created_at=webhook.created_at,
                last_triggered=webhook.last_triggered,
                success_count=webhook.success_count,
                failure_count=webhook.failure_count
            )
            for webhook in webhooks
        ]
        
    except Exception as e:
        logger.error("Failed to list webhooks", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list webhooks")

@router.post("/webhooks/trigger")
async def trigger_webhook_event(
    event_type: str,
    data: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Trigger webhook event"""
    try:
        # Add user context to event data
        event_data = {
            "user_id": current_user,
            "timestamp": datetime.utcnow().isoformat(),
            **data
        }
        
        # Trigger webhooks in background
        background_tasks.add_task(integration_engine.trigger_webhook, event_type, event_data)
        
        return {"message": "Webhook event triggered", "event_type": event_type}
        
    except Exception as e:
        logger.error("Failed to trigger webhook", event_type=event_type, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to trigger webhook")

# Data Export Management
@router.post("/exports", response_model=ExportJobResponse)
async def create_export_job(
    config: ExportJobConfig,
    current_user: str = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create data export job"""
    try:
        export_job = await integration_engine.export_data(config, current_user)
        
        return ExportJobResponse(
            id=str(export_job.id),
            job_name=export_job.job_name,
            export_type=export_job.export_type,
            data_source=export_job.data_source,
            status=export_job.status,
            created_at=export_job.created_at,
            started_at=export_job.started_at,
            completed_at=export_job.completed_at,
            file_size=export_job.file_size,
            record_count=export_job.record_count,
            error_message=export_job.error_message
        )
        
    except Exception as e:
        logger.error("Failed to create export job", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create export job")

@router.get("/exports", response_model=List[ExportJobResponse])
async def list_export_jobs(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    status: Optional[str] = Query(None),
    export_type: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """List user's export jobs"""
    try:
        query = select(DataExportJob).where(DataExportJob.created_by == UUID(current_user))
        
        if status:
            query = query.where(DataExportJob.status == status)
        
        if export_type:
            query = query.where(DataExportJob.export_type == export_type)
        
        query = query.offset(offset).limit(limit).order_by(desc(DataExportJob.created_at))
        
        result = await db.execute(query)
        export_jobs = result.scalars().all()
        
        return [
            ExportJobResponse(
                id=str(job.id),
                job_name=job.job_name,
                export_type=job.export_type,
                data_source=job.data_source,
                status=job.status,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                file_size=job.file_size,
                record_count=job.record_count,
                error_message=job.error_message
            )
            for job in export_jobs
        ]
        
    except Exception as e:
        logger.error("Failed to list export jobs", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list export jobs")

@router.get("/exports/{job_id}/download")
async def download_export_file(
    job_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Download export file"""
    try:
        query = select(DataExportJob).where(
            and_(
                DataExportJob.id == UUID(job_id),
                DataExportJob.created_by == UUID(current_user)
            )
        )
        result = await db.execute(query)
        export_job = result.scalar_one_or_none()
        
        if not export_job:
            raise HTTPException(status_code=404, detail="Export job not found")
        
        if export_job.status != "completed" or not export_job.file_path:
            raise HTTPException(status_code=400, detail="Export file not available")
        
        return FileResponse(
            export_job.file_path,
            filename=f"{export_job.job_name}.{export_job.export_type}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download export file", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to download export file")

# Authentication Services
@router.post("/auth/authenticate")
async def authenticate_external_service(
    request: AuthenticationRequest,
    current_user: str = Depends(get_current_user)
):
    """Authenticate with external service"""
    try:
        auth_result = await integration_engine.authenticate_external_service(
            request.integration_id,
            request.auth_config
        )
        
        return {
            "status": "authenticated",
            "integration_id": request.integration_id,
            "auth_data": auth_result
        }
        
    except Exception as e:
        logger.error("Authentication failed", 
                    integration_id=request.integration_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail="Authentication failed")

# GIS Integration Services
@router.post("/gis/sync")
async def sync_gis_system(
    request: GISIntegrationRequest,
    current_user: str = Depends(get_current_user)
):
    """Synchronize with GIS system"""
    try:
        sync_result = await integration_engine.sync_with_gis_system(
            request.integration_id,
            request.gis_config
        )
        
        return {
            "status": "synchronized",
            "integration_id": request.integration_id,
            "operation": request.operation,
            "result": sync_result
        }
        
    except Exception as e:
        logger.error("GIS synchronization failed", 
                    integration_id=request.integration_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail="GIS synchronization failed")

# Scientific Software Integration
@router.post("/scientific/execute")
async def execute_scientific_software(
    request: ScientificExecutionRequest,
    current_user: str = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Execute script in scientific software environment"""
    try:
        execution_result = await integration_engine.execute_scientific_software(
            request.integration_id,
            request.software_config,
            request.script,
            request.input_data
        )
        
        return {
            "status": "executed",
            "integration_id": request.integration_id,
            "software_type": request.software_config.software_type,
            "result": execution_result
        }
        
    except Exception as e:
        logger.error("Scientific software execution failed", 
                    integration_id=request.integration_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail="Scientific software execution failed")

# Cloud Storage Services
@router.post("/storage/upload")
async def upload_to_cloud_storage(
    request: CloudStorageRequest,
    current_user: str = Depends(get_current_user)
):
    """Upload file to cloud storage"""
    try:
        if request.operation != "upload":
            raise HTTPException(status_code=400, detail="Invalid operation for upload endpoint")
        
        if not request.source_path or not request.destination_path:
            raise HTTPException(status_code=400, detail="Source and destination paths required")
        
        upload_result = await integration_engine.upload_to_cloud_storage(
            request.storage_config,
            request.source_path,
            request.destination_path
        )
        
        return {
            "status": "uploaded",
            "provider": request.storage_config.provider,
            "result": upload_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Cloud storage upload failed", error=str(e))
        raise HTTPException(status_code=500, detail="Cloud storage upload failed")

# Google Cloud Services
@router.post("/google/bigquery")
async def execute_bigquery(
    request: BigQueryRequest,
    current_user: str = Depends(get_current_user)
):
    """Execute BigQuery query"""
    try:
        query_result = await integration_engine.query_bigquery(
            request.integration_id,
            request.query
        )
        
        return {
            "status": "executed",
            "integration_id": request.integration_id,
            "result": query_result
        }
        
    except Exception as e:
        logger.error("BigQuery execution failed", 
                    integration_id=request.integration_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail="BigQuery execution failed")

@router.post("/google/pubsub")
async def publish_to_pubsub(
    request: PubSubRequest,
    current_user: str = Depends(get_current_user)
):
    """Publish message to Google Cloud Pub/Sub"""
    try:
        message_id = await integration_engine.publish_to_pubsub(
            request.integration_id,
            request.topic,
            request.message
        )
        
        return {
            "status": "published",
            "integration_id": request.integration_id,
            "topic": request.topic,
            "message_id": message_id
        }
        
    except Exception as e:
        logger.error("Pub/Sub publish failed", 
                    integration_id=request.integration_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail="Pub/Sub publish failed")

# Analytics and Monitoring
@router.get("/analytics", response_model=IntegrationAnalytics)
async def get_integration_analytics(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    days: int = Query(30, ge=1, le=365)
):
    """Get integration analytics"""
    try:
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Count integrations
        integration_query = select(func.count(ExternalIntegration.id)).where(
            and_(
                ExternalIntegration.created_by == UUID(current_user),
                ExternalIntegration.created_at >= since_date
            )
        )
        integration_result = await db.execute(integration_query)
        total_integrations = integration_result.scalar() or 0
        
        # Count active integrations
        active_integration_query = select(func.count(ExternalIntegration.id)).where(
            and_(
                ExternalIntegration.created_by == UUID(current_user),
                ExternalIntegration.status == "active"
            )
        )
        active_integration_result = await db.execute(active_integration_query)
        active_integrations = active_integration_result.scalar() or 0
        
        # Count webhooks
        webhook_query = select(func.count(WebhookEndpoint.id)).where(
            WebhookEndpoint.created_by == UUID(current_user)
        )
        webhook_result = await db.execute(webhook_query)
        total_webhooks = webhook_result.scalar() or 0
        
        # Count active webhooks
        active_webhook_query = select(func.count(WebhookEndpoint.id)).where(
            and_(
                WebhookEndpoint.created_by == UUID(current_user),
                WebhookEndpoint.is_active == True
            )
        )
        active_webhook_result = await db.execute(active_webhook_query)
        active_webhooks = active_webhook_result.scalar() or 0
        
        # Calculate webhook success rate
        webhook_success_query = select(
            func.sum(WebhookEndpoint.success_count),
            func.sum(WebhookEndpoint.failure_count)
        ).where(WebhookEndpoint.created_by == UUID(current_user))
        webhook_success_result = await db.execute(webhook_success_query)
        success_count, failure_count = webhook_success_result.first() or (0, 0)
        
        webhook_success_rate = 0.0
        if success_count or failure_count:
            webhook_success_rate = (success_count or 0) / ((success_count or 0) + (failure_count or 0))
        
        # Count export jobs
        export_query = select(func.count(DataExportJob.id)).where(
            DataExportJob.created_by == UUID(current_user)
        )
        export_result = await db.execute(export_query)
        total_exports = export_result.scalar() or 0
        
        # Count successful exports
        successful_export_query = select(func.count(DataExportJob.id)).where(
            and_(
                DataExportJob.created_by == UUID(current_user),
                DataExportJob.status == "completed"
            )
        )
        successful_export_result = await db.execute(successful_export_query)
        successful_exports = successful_export_result.scalar() or 0
        
        export_success_rate = 0.0
        if total_exports > 0:
            export_success_rate = successful_exports / total_exports
        
        # Get integration types and providers (simplified)
        integration_types = {"api_gateway": 0, "cloud_storage": 0, "authentication": 0}
        providers = {"google": 0, "aws": 0, "azure": 0}
        
        return IntegrationAnalytics(
            total_integrations=total_integrations,
            active_integrations=active_integrations,
            integration_types=integration_types,
            providers=providers,
            total_webhooks=total_webhooks,
            active_webhooks=active_webhooks,
            webhook_success_rate=round(webhook_success_rate, 3),
            total_exports=total_exports,
            successful_exports=successful_exports,
            export_success_rate=round(export_success_rate, 3)
        )
        
    except Exception as e:
        logger.error("Failed to get integration analytics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get integration analytics")