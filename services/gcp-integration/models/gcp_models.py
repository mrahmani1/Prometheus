"""
Google Cloud Platform Data Models
Comprehensive data models for GCP data analytics integration
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json

from pydantic import BaseModel, Field, validator


class AnalysisType(str, Enum):
    """Types of geological analysis"""
    SEISMIC = "seismic"
    VOLCANIC = "volcanic"
    MINERAL_PROSPECTIVITY = "mineral_prospectivity"
    GEOLOGICAL_CLUSTERING = "geological_clustering"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    THERMAL_ANOMALY = "thermal_anomaly"
    VEGETATION_STRESS = "vegetation_stress"
    LANDSLIDE_DETECTION = "landslide_detection"
    COMPREHENSIVE = "comprehensive"


class ProcessingType(str, Enum):
    """Types of data processing"""
    SEISMIC = "seismic"
    VOLCANIC = "volcanic"
    SATELLITE = "satellite"
    GEOCHEMICAL = "geochemical"
    GENERAL = "general"


class JobStatus(str, Enum):
    """Job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AlertLevel(str, Enum):
    """Alert level enumeration"""
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


# Pydantic Models for API requests/responses

class GeographicRegion(BaseModel):
    """Geographic region specification"""
    coordinates: Optional[List[List[float]]] = None
    bounds: Optional[Dict[str, float]] = None
    center: Optional[Dict[str, float]] = None
    radius: Optional[float] = None
    
    @validator('bounds')
    def validate_bounds(cls, v):
        if v and not all(k in v for k in ['north', 'south', 'east', 'west']):
            raise ValueError('Bounds must contain north, south, east, west')
        return v


class GeologicalAnalysisRequest(BaseModel):
    """Request for geological analysis"""
    dataset_id: str = Field(..., description="Dataset identifier")
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")
    region: Optional[GeographicRegion] = Field(None, description="Geographic region")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range for analysis")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")


class DataflowJobRequest(BaseModel):
    """Request for Dataflow job"""
    job_name: str = Field(..., description="Job name")
    processing_type: ProcessingType = Field(..., description="Type of processing")
    input_source: str = Field(..., description="Input data source")
    output_destination: str = Field(..., description="Output destination")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Job parameters")


class BigQueryQueryRequest(BaseModel):
    """Request for BigQuery query"""
    query: str = Field(..., description="SQL query")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Query parameters")
    use_legacy_sql: bool = Field(False, description="Use legacy SQL")
    dry_run: bool = Field(False, description="Perform dry run")


class EarthEngineAnalysisRequest(BaseModel):
    """Request for Earth Engine analysis"""
    analysis_id: str = Field(..., description="Analysis identifier")
    region: GeographicRegion = Field(..., description="Region of interest")
    analysis_type: str = Field("comprehensive", description="Type of analysis")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")


class ComposerWorkflowRequest(BaseModel):
    """Request for Composer workflow"""
    workflow_name: str = Field(..., description="Workflow name")
    dag_id: str = Field(..., description="DAG identifier")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Workflow parameters")
    schedule_interval: Optional[str] = Field(None, description="Schedule interval")


class DataFusionPipelineRequest(BaseModel):
    """Request for Data Fusion pipeline"""
    pipeline_name: str = Field(..., description="Pipeline name")
    source_config: Dict[str, Any] = Field(..., description="Source configuration")
    sink_config: Dict[str, Any] = Field(..., description="Sink configuration")
    transformations: List[Dict[str, Any]] = Field(default_factory=list, description="Transformations")


class LookerDashboardRequest(BaseModel):
    """Request for Looker dashboard"""
    dashboard_name: str = Field(..., description="Dashboard name")
    data_source: str = Field(..., description="Data source")
    visualizations: List[Dict[str, Any]] = Field(..., description="Visualization configurations")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Dashboard filters")


# Dataclass Models for internal use

@dataclass
class GeologicalAnalysisJob:
    """Geological analysis job"""
    job_id: str
    dataset_id: str
    analysis_type: str
    parameters: Dict[str, Any]
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataflowJob:
    """Dataflow job information"""
    job_id: str
    job_name: str
    processing_type: str
    status: str
    created_at: datetime
    parameters: Dict[str, Any]
    input_source: str
    output_destination: str
    error: Optional[str] = None


@dataclass
class BigQueryTable:
    """BigQuery table information"""
    project_id: str
    dataset_id: str
    table_id: str
    schema: List[Dict[str, Any]]
    description: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    partitioning: Optional[Dict[str, Any]] = None
    clustering: Optional[List[str]] = None


@dataclass
class BigQueryMLModel:
    """BigQuery ML model information"""
    project_id: str
    dataset_id: str
    model_id: str
    model_type: str
    training_query: str
    feature_columns: List[str]
    label_column: str
    created_at: datetime
    last_modified: datetime
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EarthEngineAnalysis:
    """Earth Engine analysis result"""
    analysis_id: str
    analysis_type: str
    region: Dict[str, Any]
    date_range: Dict[str, str]
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    export_tasks: List[str] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


@dataclass
class ComposerWorkflow:
    """Composer workflow information"""
    workflow_id: str
    workflow_name: str
    dag_id: str
    parameters: Dict[str, Any]
    schedule_interval: Optional[str]
    status: str
    created_at: datetime
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


@dataclass
class DataFusionPipeline:
    """Data Fusion pipeline information"""
    pipeline_id: str
    pipeline_name: str
    source_config: Dict[str, Any]
    sink_config: Dict[str, Any]
    transformations: List[Dict[str, Any]]
    status: str
    created_at: datetime
    last_run: Optional[datetime] = None


@dataclass
class LookerDashboard:
    """Looker dashboard information"""
    dashboard_id: str
    dashboard_name: str
    data_source: str
    visualizations: List[Dict[str, Any]]
    filters: Dict[str, Any]
    url: str
    created_at: datetime
    last_modified: datetime


@dataclass
class DataCatalogEntry:
    """Data Catalog entry information"""
    entry_id: str
    entry_name: str
    entry_type: str
    description: str
    schema: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataplexAsset:
    """Dataplex asset information"""
    asset_id: str
    asset_name: str
    asset_type: str
    location: str
    resource_spec: Dict[str, Any]
    discovery_spec: Dict[str, Any]
    status: str
    created_at: datetime
    last_updated: datetime


@dataclass
class SeismicEvent:
    """Seismic event data"""
    event_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    depth: float
    magnitude: float
    magnitude_type: str
    region: str
    event_type: Optional[str] = None
    p_wave_velocity: Optional[float] = None
    s_wave_velocity: Optional[float] = None
    frequency_content: Optional[float] = None


@dataclass
class VolcanicMonitoring:
    """Volcanic monitoring data"""
    volcano_id: str
    volcano_name: str
    timestamp: datetime
    deformation_rate: Optional[float] = None
    gas_so2_flux: Optional[float] = None
    thermal_anomaly_intensity: Optional[float] = None
    seismic_energy: Optional[float] = None
    ground_temperature: Optional[float] = None
    alert_level: Optional[str] = None
    eruption_probability: Optional[float] = None


@dataclass
class MineralDeposit:
    """Mineral deposit data"""
    deposit_id: str
    deposit_name: str
    latitude: float
    longitude: float
    deposit_type: str
    commodity: str
    tonnage: Optional[float] = None
    grade: Optional[float] = None
    discovery_date: Optional[datetime] = None
    country: Optional[str] = None
    geological_setting: Optional[str] = None


@dataclass
class GeochemicalSample:
    """Geochemical sample data"""
    sample_id: str
    latitude: float
    longitude: float
    sample_type: str
    collection_date: datetime
    elements: Dict[str, float] = field(default_factory=dict)
    oxides: Dict[str, float] = field(default_factory=dict)
    quality_score: Optional[float] = None


@dataclass
class SatelliteImage:
    """Satellite image metadata"""
    image_id: str
    timestamp: datetime
    satellite: str
    sensor: str
    bounds: Dict[str, float]
    cloud_cover: Optional[float] = None
    resolution_meters: Optional[float] = None
    processing_level: Optional[str] = None
    gcs_path: Optional[str] = None


# Response Models

class AnalysisResponse(BaseModel):
    """Analysis response"""
    job_id: str
    status: str
    message: str
    estimated_completion: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    job_name: str
    status: str
    create_time: Optional[str] = None
    start_time: Optional[str] = None
    completion_time: Optional[str] = None
    error_message: Optional[str] = None


class ServiceStatusResponse(BaseModel):
    """Service status response"""
    service: str
    status: str
    timestamp: str
    details: Dict[str, Any] = Field(default_factory=dict)


class QueryResultResponse(BaseModel):
    """Query result response"""
    query_id: str
    status: str
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    total_rows: int = 0
    bytes_processed: Optional[int] = None
    execution_time_ms: Optional[int] = None


class ExportTaskResponse(BaseModel):
    """Export task response"""
    task_id: str
    status: str
    export_url: Optional[str] = None
    created_at: str
    estimated_completion: Optional[str] = None


# Utility functions

def serialize_dataclass(obj) -> Dict[str, Any]:
    """Serialize dataclass to dictionary"""
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name, field_value in asdict(obj).items():
            if isinstance(field_value, datetime):
                result[field_name] = field_value.isoformat()
            else:
                result[field_name] = field_value
        return result
    return obj


def deserialize_datetime(date_str: str) -> datetime:
    """Deserialize datetime string"""
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except ValueError:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')


class GeologicalDataEncoder(json.JSONEncoder):
    """Custom JSON encoder for geological data"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dataclass_fields__'):
            return serialize_dataclass(obj)
        elif isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


# Configuration classes

@dataclass
class BigQueryConfig:
    """BigQuery configuration"""
    project_id: str
    dataset_id: str
    location: str = "US"
    use_legacy_sql: bool = False
    maximum_bytes_billed: Optional[int] = None


@dataclass
class DataflowConfig:
    """Dataflow configuration"""
    project_id: str
    region: str
    temp_location: str
    staging_location: str
    machine_type: str = "n1-standard-1"
    max_num_workers: int = 10
    use_public_ips: bool = True


@dataclass
class EarthEngineConfig:
    """Earth Engine configuration"""
    project_id: str
    service_account_key: Optional[str] = None
    output_bucket: str = "prometheus-earth-engine-outputs"
    temp_bucket: str = "prometheus-earth-engine-temp"


@dataclass
class ComposerConfig:
    """Composer configuration"""
    project_id: str
    location: str
    environment_name: str
    node_count: int = 3
    machine_type: str = "n1-standard-1"
    disk_size_gb: int = 100


@dataclass
class DataFusionConfig:
    """Data Fusion configuration"""
    project_id: str
    location: str
    instance_name: str
    instance_type: str = "BASIC"
    enable_stackdriver_logging: bool = True
    enable_stackdriver_monitoring: bool = True


@dataclass
class LookerConfig:
    """Looker configuration"""
    base_url: str
    client_id: str
    client_secret: str
    verify_ssl: bool = True
    timeout: int = 120


@dataclass
class DataCatalogConfig:
    """Data Catalog configuration"""
    project_id: str
    location: str = "us-central1"
    entry_group_id: str = "geological-data"


@dataclass
class DataplexConfig:
    """Dataplex configuration"""
    project_id: str
    location: str
    lake_id: str = "geological-data-lake"
    zone_id: str = "geological-zone"