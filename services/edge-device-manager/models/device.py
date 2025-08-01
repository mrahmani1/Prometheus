"""
Data models for edge devices
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from .config import DeviceCapabilities, DeviceLocation, EdgeDeviceConfig

class DeviceStatus(str, Enum):
    """Device operational status"""

    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    UNKNOWN = "unknown"

class ModelStatus(str, Enum):
    """Model deployment status"""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    INSTALLING = "installing"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    UPDATING = "updating"

class SyncStatus(str, Enum):
    """Data synchronization status"""

    IDLE = "idle"
    SYNCING = "syncing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class DeviceMetrics(BaseModel):
    """Real-time device metrics"""

    timestamp: datetime
    device_id: str

    # System metrics
    cpu_usage_percent: float
    memory_usage_percent: float
    storage_usage_percent: float
    temperature_celsius: Optional[float] = None

    # Network metrics
    network_latency_ms: Optional[float] = None
    bandwidth_usage_mbps: Optional[float] = None

    # Power metrics
    battery_level_percent: Optional[float] = None
    power_consumption_watts: Optional[float] = None

    # Application metrics
    active_models: int = 0
    data_processing_rate: Optional[float] = None  # samples per second
    error_count: int = 0

class EdgeDevice(BaseModel):
    """Edge device representation"""

    device_id: str
    device_name: str
    device_type: str
    location: DeviceLocation
    capabilities: DeviceCapabilities
    config: EdgeDeviceConfig

    # Status information
    status: DeviceStatus = DeviceStatus.UNKNOWN
    last_seen: Optional[datetime] = None
    uptime_seconds: Optional[int] = None

    # Registration information
    registered_at: datetime
    last_updated: datetime
    firmware_version: Optional[str] = None
    software_version: Optional[str] = None

    # Current metrics
    current_metrics: Optional[DeviceMetrics] = None

    # Deployed models
    deployed_models: List[str] = []

    # Sync status
    last_sync: Optional[datetime] = None
    sync_status: SyncStatus = SyncStatus.IDLE

class ModelDeployment(BaseModel):
    """Model deployment on edge device"""

    deployment_id: str
    device_id: str
    model_id: str
    model_name: str
    model_version: str

    # Deployment status
    status: ModelStatus = ModelStatus.PENDING
    deployed_at: Optional[datetime] = None
    last_updated: datetime

    # Resource usage
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    storage_usage_mb: Optional[float] = None

    # Performance metrics
    inference_count: int = 0
    average_inference_time_ms: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None

    # Configuration
    container_id: Optional[str] = None
    port: Optional[int] = None
    health_status: str = "unknown"

class DeviceRegistrationRequest(BaseModel):
    """Request to register a new edge device"""

    device_name: str
    device_type: str
    location: DeviceLocation
    capabilities: DeviceCapabilities

    # Network information
    ip_address: str
    port: int = 8080
    ssl_enabled: bool = False

    # Authentication
    certificate: Optional[str] = None
    public_key: Optional[str] = None

    # Additional metadata
    firmware_version: Optional[str] = None
    software_version: Optional[str] = None
    metadata: Dict[str, Any] = {}

class DeviceRegistrationResponse(BaseModel):
    """Response to device registration"""

    device_id: str
    auth_token: str
    config: EdgeDeviceConfig

    # Service endpoints
    management_endpoint: str
    sync_endpoint: str
    model_registry_endpoint: str

    # Certificates and keys
    ca_certificate: Optional[str] = None
    client_certificate: Optional[str] = None
    client_private_key: Optional[str] = None

class ModelDeploymentRequest(BaseModel):
    """Request to deploy a model to edge devices"""

    model_id: str
    target_devices: List[str]  # Device IDs or selection criteria

    # Deployment options
    force_update: bool = False
    rollback_on_failure: bool = True
    max_concurrent_deployments: Optional[int] = None

    # Configuration overrides
    environment_variables: Optional[Dict[str, str]] = None
    resource_limits: Optional[Dict[str, Any]] = None

class ModelDeploymentResponse(BaseModel):
    """Response to model deployment request"""

    deployment_id: str
    target_devices: List[str]
    status: str

    # Progress tracking
    total_devices: int
    successful_deployments: int
    failed_deployments: int
    pending_deployments: int

    # Deployment details
    deployment_details: List[Dict[str, Any]] = []

class DataSyncRequest(BaseModel):
    """Request to synchronize data with edge devices"""

    device_ids: List[str]
    data_types: List[str]  # sensor_data, model_outputs, logs, metrics

    # Sync parameters
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    force_full_sync: bool = False

    # Bandwidth management
    max_bandwidth_mbps: Optional[float] = None
    priority: str = "normal"  # low, normal, high

class DataSyncResponse(BaseModel):
    """Response to data sync request"""

    sync_id: str
    device_ids: List[str]
    status: SyncStatus

    # Progress information
    total_bytes: Optional[int] = None
    transferred_bytes: Optional[int] = None
    estimated_completion: Optional[datetime] = None

    # Per-device status
    device_status: Dict[str, Dict[str, Any]] = {}

class HealthCheckResult(BaseModel):
    """Health check result for a device"""

    device_id: str
    timestamp: datetime
    status: DeviceStatus

    # Health indicators
    response_time_ms: float
    system_health: Dict[str, Any]
    model_health: Dict[str, Any]
    connectivity_health: Dict[str, Any]

    # Issues and warnings
    issues: List[str] = []
    warnings: List[str] = []

    # Recommendations
    recommendations: List[str] = []
