"""
API models and schemas for Model Management Service
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator

class ModelType(str, Enum):
    ERUPTION = "eruption"
    VEI = "vei"
    TIME_TO_EVENT = "time"
    MULTI_TASK = "multi_task"
    SEISMIC_CLASSIFICATION = "seismic_classification"
    DEFORMATION_ANALYSIS = "deformation_analysis"

class ModelStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

class DeploymentStrategy(str, Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    AB_TEST = "ab_test"

class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CloudProvider(str, Enum):
    AZURE = "azure"
    GCP = "gcp"
    AWS = "aws"
    LOCAL = "local"

# Request models
class ModelConfigRequest(BaseModel):
    model_type: ModelType = Field(..., description="Type of model")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    azure_vm_endpoint: str = Field("", description="Azure VM endpoint")
    google_cloud_project: str = Field("", description="Google Cloud project ID")
    aws_region: str = Field("us-west-2", description="AWS region")
    dataset_path: str = Field("", description="Path to training dataset")
    model_save_path: str = Field("", description="Path to save models")
    retrain_interval: int = Field(24, ge=1, le=168, description="Retraining interval in hours")
    multi_cloud_training: bool = Field(True, description="Enable multi-cloud training")
    auto_deploy: bool = Field(True, description="Enable automatic deployment")
    enable_vertex_ai: bool = Field(True, description="Enable Google Vertex AI")
    enable_sagemaker: bool = Field(True, description="Enable AWS SageMaker")
    enable_azure_ml: bool = Field(True, description="Enable Azure ML")
    min_accuracy: float = Field(0.75, ge=0.0, le=1.0, description="Minimum model accuracy threshold")
    max_latency_ms: float = Field(100.0, ge=0.0, description="Maximum latency threshold")
    enable_uncertainty_quantification: bool = Field(True, description="Enable uncertainty quantification")
    enable_physics_constraints: bool = Field(True, description="Enable physics-informed constraints")

class TrainingRequest(BaseModel):
    model_type: ModelType = Field(..., description="Type of model to train")
    force_retrain: bool = Field(False, description="Force retraining even if recent")
    cloud_providers: List[CloudProvider] = Field(default_factory=list, description="Specific cloud providers to use")
    experiment_name: str = Field("", description="MLflow experiment name")
    hyperparameter_tuning: bool = Field(True, description="Enable hyperparameter tuning")
    cross_validation_folds: int = Field(5, ge=2, le=10, description="Number of CV folds")
    early_stopping: bool = Field(True, description="Enable early stopping")
    data_augmentation: bool = Field(False, description="Enable data augmentation")
    
    @validator('cloud_providers')
    def validate_cloud_providers(cls, v):
        if not v:
            return [CloudProvider.LOCAL]
        return v

class DeploymentRequest(BaseModel):
    model_id: str = Field(..., description="Model ID to deploy")
    deployment_strategy: DeploymentStrategy = Field(DeploymentStrategy.ROLLING, description="Deployment strategy")
    target_environment: ModelStage = Field(ModelStage.STAGING, description="Target environment")
    cloud_providers: List[CloudProvider] = Field(default_factory=list, description="Target cloud providers")
    traffic_percentage: float = Field(100.0, ge=0.0, le=100.0, description="Traffic percentage for canary/AB testing")
    rollback_threshold: float = Field(0.05, ge=0.0, le=1.0, description="Error rate threshold for automatic rollback")
    monitoring_duration_minutes: int = Field(60, ge=5, le=1440, description="Monitoring duration before full deployment")

class PredictionRequest(BaseModel):
    model_type: ModelType = Field(..., description="Type of model for prediction")
    features: List[float] = Field(..., description="Input features for prediction")
    uncertainty_estimation: bool = Field(True, description="Include uncertainty estimation")
    n_samples: int = Field(100, ge=10, le=1000, description="Number of samples for uncertainty estimation")
    return_probabilities: bool = Field(False, description="Return class probabilities for classification")
    explain_prediction: bool = Field(False, description="Include prediction explanation")

class BatchPredictionRequest(BaseModel):
    model_type: ModelType = Field(..., description="Type of model for prediction")
    features_batch: List[List[float]] = Field(..., description="Batch of input features")
    uncertainty_estimation: bool = Field(True, description="Include uncertainty estimation")
    n_samples: int = Field(100, ge=10, le=1000, description="Number of samples for uncertainty estimation")
    batch_size: int = Field(32, ge=1, le=1000, description="Processing batch size")

class ModelVersionRequest(BaseModel):
    model_id: str = Field(..., description="Model ID")
    version_tag: str = Field(..., description="Version tag")
    description: str = Field("", description="Version description")
    is_production: bool = Field(False, description="Mark as production version")

# Response models
class ModelPerformanceResponse(BaseModel):
    model_id: str
    model_type: ModelType
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    latency_ms: float
    throughput_rps: float
    memory_usage_mb: float
    last_updated: datetime
    deployment_status: str
    drift_score: Optional[float] = None
    uncertainty_calibration: Optional[float] = None
    physics_constraint_violations: Optional[int] = None

class TrainingJobResponse(BaseModel):
    job_id: str
    model_type: ModelType
    status: TrainingStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    cloud_provider: CloudProvider
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts: Dict[str, str] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    error_message: Optional[str] = None

class DeploymentResponse(BaseModel):
    deployment_id: str
    model_id: str
    model_version: str
    deployment_strategy: DeploymentStrategy
    target_environment: ModelStage
    status: str
    start_time: datetime
    completion_time: Optional[datetime] = None
    cloud_providers: List[CloudProvider]
    traffic_percentage: float
    endpoint_urls: Dict[str, str] = Field(default_factory=dict)
    health_status: Dict[str, str] = Field(default_factory=dict)

class PredictionResponse(BaseModel):
    prediction: Union[float, List[float], Dict[str, Any]]
    uncertainty: Optional[Union[float, List[float]]] = None
    confidence: Optional[float] = None
    model_type: ModelType
    model_version: str
    prediction_time_ms: float
    explanation: Optional[Dict[str, Any]] = None
    physics_constraints_satisfied: Optional[bool] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[Union[float, List[float], Dict[str, Any]]]
    uncertainties: Optional[List[Union[float, List[float]]]] = None
    confidences: Optional[List[float]] = None
    model_type: ModelType
    model_version: str
    batch_size: int
    total_prediction_time_ms: float
    average_prediction_time_ms: float

class ModelListResponse(BaseModel):
    model_id: str
    model_type: ModelType
    version: str
    stage: ModelStage
    created_at: datetime
    last_updated: datetime
    accuracy: Optional[float] = None
    deployment_status: str
    cloud_providers: List[CloudProvider]
    size_mb: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]
    model_count: int
    active_deployments: int
    active_training_jobs: int
    system_resources: Dict[str, Any] = Field(default_factory=dict)

class MetricsResponse(BaseModel):
    timestamp: datetime
    model_metrics: Dict[str, Dict[str, float]]
    system_metrics: Dict[str, float]
    deployment_metrics: Dict[str, Dict[str, float]]
    training_metrics: Dict[str, Dict[str, float]]

class ModelVersionResponse(BaseModel):
    model_id: str
    version: str
    version_tag: str
    description: str
    created_at: datetime
    is_production: bool
    performance_metrics: Dict[str, float]
    artifacts: Dict[str, str]
    parent_version: Optional[str] = None

class DriftDetectionResponse(BaseModel):
    model_id: str
    drift_score: float
    drift_threshold: float
    is_drifting: bool
    drift_type: str  # "data", "concept", "prediction"
    detection_time: datetime
    affected_features: List[str]
    recommended_actions: List[str]
    confidence: float

class ABTestResponse(BaseModel):
    test_id: str
    model_a_id: str
    model_b_id: str
    traffic_split: Dict[str, float]
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics_comparison: Dict[str, Dict[str, float]]
    statistical_significance: Optional[float] = None
    winner: Optional[str] = None
    recommendation: str

# Error response models
class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None