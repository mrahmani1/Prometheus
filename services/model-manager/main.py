"""
Model Management Service - FastAPI microservice for ML model lifecycle management
Integrates with existing model management modules and provides REST API endpoints
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Import existing model management modules
from modules.model_manager import ModelManager, ModelConfig
from modules.mlops_pipeline import MLOpsPipeline, PipelineConfig, ModelStage, DeploymentStrategy
from modules.model_database import ModelDatabase, ModelMetadata
from modules.advanced_ml_model_management import AdvancedMLModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service state
model_manager: Optional[ModelManager] = None
mlops_pipeline: Optional[MLOpsPipeline] = None
model_database: Optional[ModelDatabase] = None
advanced_manager: Optional[AdvancedMLModelManager] = None

# Pydantic models for API
class ModelConfigRequest(BaseModel):
    model_type: str = Field(..., description="Type of model (eruption, vei, time)")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    azure_vm_endpoint: str = Field("", description="Azure VM endpoint")
    google_cloud_project: str = Field("", description="Google Cloud project ID")
    aws_region: str = Field("us-west-2", description="AWS region")
    dataset_path: str = Field("", description="Path to training dataset")
    model_save_path: str = Field("", description="Path to save models")
    retrain_interval: int = Field(24, description="Retraining interval in hours")
    multi_cloud_training: bool = Field(True, description="Enable multi-cloud training")
    auto_deploy: bool = Field(True, description="Enable automatic deployment")
    enable_vertex_ai: bool = Field(True, description="Enable Google Vertex AI")
    enable_sagemaker: bool = Field(True, description="Enable AWS SageMaker")
    enable_azure_ml: bool = Field(True, description="Enable Azure ML")
    min_accuracy: float = Field(0.75, description="Minimum model accuracy threshold")
    max_latency_ms: float = Field(100.0, description="Maximum latency threshold")

class TrainingRequest(BaseModel):
    model_type: str = Field(..., description="Type of model to train")
    force_retrain: bool = Field(False, description="Force retraining even if recent")
    cloud_providers: List[str] = Field(default_factory=list, description="Specific cloud providers to use")
    experiment_name: str = Field("", description="MLflow experiment name")

class DeploymentRequest(BaseModel):
    model_id: str = Field(..., description="Model ID to deploy")
    deployment_strategy: str = Field("rolling", description="Deployment strategy")
    target_environment: str = Field("staging", description="Target environment")
    cloud_providers: List[str] = Field(default_factory=list, description="Target cloud providers")
    traffic_percentage: float = Field(100.0, description="Traffic percentage for canary/AB testing")

class PredictionRequest(BaseModel):
    model_type: str = Field(..., description="Type of model for prediction")
    features: List[float] = Field(..., description="Input features for prediction")
    uncertainty_estimation: bool = Field(True, description="Include uncertainty estimation")
    n_samples: int = Field(100, description="Number of samples for uncertainty estimation")

class ModelPerformanceResponse(BaseModel):
    model_id: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    throughput_rps: float
    last_updated: datetime
    deployment_status: str
    drift_score: Optional[float] = None

class TrainingJobResponse(BaseModel):
    job_id: str
    model_type: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    cloud_provider: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts: Dict[str, str] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]
    model_count: int
    active_deployments: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Model Management Service")
    await initialize_services()
    yield
    # Shutdown
    logger.info("Shutting down Model Management Service")
    await cleanup_services()

# Create FastAPI app
app = FastAPI(
    title="Model Management Service",
    description="ML model lifecycle management for Prometheus AI Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_services():
    """Initialize all model management services"""
    global model_manager, mlops_pipeline, model_database, advanced_manager
    
    try:
        # Initialize model database
        db_config = {
            'database_url': os.getenv('MODEL_DATABASE_URL', 'sqlite:///model_manager.db'),
            'storage_backend': 'multi_cloud',
            'storage_path': os.getenv('MODEL_STORAGE_PATH', './models'),
        }
        model_database = ModelDatabase(db_config)
        await model_database.initialize()
        logger.info("Model database initialized")
        
        # Initialize MLOps pipeline
        pipeline_config = PipelineConfig(
            model_registry_url=os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
            experiment_tracking_url=os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
            data_drift_threshold=0.1,
            model_performance_threshold=0.75,
            retraining_frequency=24,
            validation_split=0.2,
            monitoring_interval=300,
            multi_cloud_enabled=True,
            auto_deploy=True,
            enable_a_b_testing=True,
            enable_canary_deployment=True
        )
        mlops_pipeline = MLOpsPipeline(pipeline_config)
        await mlops_pipeline.initialize()
        logger.info("MLOps pipeline initialized")
        
        # Initialize advanced model manager
        advanced_config = {
            'enable_multi_cloud': True,
            'enable_automl': True,
            'enable_federated_learning': False,
            'performance_monitoring': True,
            'automated_retraining': True
        }
        advanced_manager = AdvancedMLModelManager(advanced_config)
        logger.info("Advanced model manager initialized")
        
        # Initialize basic model manager with default config
        default_config = ModelConfig(
            model_type="volcanic_prediction",
            hyperparameters={},
            azure_vm_endpoint=os.getenv('AZURE_VM_ENDPOINT', ''),
            google_cloud_project=os.getenv('GOOGLE_CLOUD_PROJECT', ''),
            aws_region=os.getenv('AWS_REGION', 'us-west-2'),
            dataset_path=os.getenv('DATASET_PATH', './data'),
            model_save_path=os.getenv('MODEL_SAVE_PATH', './models')
        )
        model_manager = ModelManager(default_config)
        logger.info("Model manager initialized")
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        raise

async def cleanup_services():
    """Cleanup services on shutdown"""
    global model_manager, mlops_pipeline, model_database, advanced_manager
    
    try:
        if mlops_pipeline:
            await mlops_pipeline.shutdown()
        if model_database:
            await model_database.close()
        logger.info("Services cleaned up successfully")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        components = {
            "model_manager": "healthy" if model_manager else "unavailable",
            "mlops_pipeline": "healthy" if mlops_pipeline else "unavailable",
            "model_database": "healthy" if model_database else "unavailable",
            "advanced_manager": "healthy" if advanced_manager else "unavailable"
        }
        
        model_count = len(model_manager.models) if model_manager else 0
        active_deployments = len(model_manager.deployments) if model_manager else 0
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            components=components,
            model_count=model_count,
            active_deployments=active_deployments
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    if not all([model_manager, mlops_pipeline, model_database]):
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

# Model management endpoints
@app.post("/models/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a new model"""
    try:
        if not model_manager:
            raise HTTPException(status_code=503, detail="Model manager not available")
        
        # Start training in background
        job_id = f"train_{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        background_tasks.add_task(
            _train_model_background,
            job_id,
            request.model_type,
            request.force_retrain,
            request.cloud_providers,
            request.experiment_name
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": f"Training job for {request.model_type} started"
        }
        
    except Exception as e:
        logger.error(f"Training request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _train_model_background(job_id: str, model_type: str, force_retrain: bool, 
                                cloud_providers: List[str], experiment_name: str):
    """Background task for model training"""
    try:
        logger.info(f"Starting training job {job_id} for model type {model_type}")
        
        # Use advanced manager for comprehensive training
        if advanced_manager:
            result = await advanced_manager.train_model(
                model_id=model_type,
                force_retrain=force_retrain
            )
        else:
            # Fallback to basic model manager
            result = await model_manager.train_multi_cloud_models(force_retrain=force_retrain)
        
        logger.info(f"Training job {job_id} completed: {result}")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")

@app.post("/models/deploy")
async def deploy_model(request: DeploymentRequest):
    """Deploy a model to specified environment"""
    try:
        if not model_manager or not mlops_pipeline:
            raise HTTPException(status_code=503, detail="Services not available")
        
        # Use MLOps pipeline for deployment
        deployment_result = await mlops_pipeline.deploy_model(
            model_id=request.model_id,
            strategy=DeploymentStrategy(request.deployment_strategy),
            target_stage=ModelStage(request.target_environment),
            traffic_percentage=request.traffic_percentage
        )
        
        return {
            "deployment_id": deployment_result.get("deployment_id"),
            "status": deployment_result.get("status"),
            "message": f"Model {request.model_id} deployment initiated"
        }
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}/performance", response_model=ModelPerformanceResponse)
async def get_model_performance(model_id: str):
    """Get model performance metrics"""
    try:
        if not advanced_manager:
            raise HTTPException(status_code=503, detail="Advanced manager not available")
        
        performance = await advanced_manager.get_model_performance(model_id)
        
        return ModelPerformanceResponse(
            model_id=model_id,
            model_type=performance.get("model_type", "unknown"),
            accuracy=performance.get("accuracy", 0.0),
            precision=performance.get("precision", 0.0),
            recall=performance.get("recall", 0.0),
            f1_score=performance.get("f1_score", 0.0),
            latency_ms=performance.get("latency_ms", 0.0),
            throughput_rps=performance.get("throughput_rps", 0.0),
            last_updated=performance.get("last_updated", datetime.now()),
            deployment_status=performance.get("deployment_status", "unknown"),
            drift_score=performance.get("drift_score")
        )
        
    except Exception as e:
        logger.error(f"Performance retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/predict")
async def predict(request: PredictionRequest):
    """Make predictions using specified model"""
    try:
        if not model_manager:
            raise HTTPException(status_code=503, detail="Model manager not available")
        
        # Get model
        if request.model_type not in model_manager.models:
            raise HTTPException(status_code=404, detail=f"Model {request.model_type} not found")
        
        model = model_manager.models[request.model_type]
        scaler = model_manager.scalers.get(request.model_type)
        
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        if scaler:
            features = scaler.transform(features)
        
        # Make prediction
        if hasattr(model, 'predict_with_uncertainty') and request.uncertainty_estimation:
            prediction, uncertainty = model.predict_with_uncertainty(
                torch.FloatTensor(features), 
                n_samples=request.n_samples
            )
            return {
                "prediction": prediction.tolist(),
                "uncertainty": uncertainty.tolist(),
                "model_type": request.model_type
            }
        else:
            prediction = model.predict(features)
            return {
                "prediction": prediction.tolist(),
                "model_type": request.model_type
            }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List all available models"""
    try:
        if not model_manager:
            raise HTTPException(status_code=503, detail="Model manager not available")
        
        models = []
        for model_type, model in model_manager.models.items():
            metadata = model_manager.model_metadata.get(model_type, {})
            models.append({
                "model_type": model_type,
                "model_class": model.__class__.__name__,
                "last_trained": metadata.get("last_trained"),
                "accuracy": metadata.get("accuracy"),
                "version": metadata.get("version")
            })
        
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Model listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training-jobs")
async def list_training_jobs():
    """List all training jobs"""
    try:
        if not model_manager:
            raise HTTPException(status_code=503, detail="Model manager not available")
        
        jobs = []
        for job_id, job in model_manager.training_jobs.items():
            jobs.append(TrainingJobResponse(
                job_id=job_id,
                model_type=job.model_type,
                status=job.status,
                start_time=job.start_time,
                end_time=job.end_time,
                cloud_provider=job.cloud_provider,
                metrics=job.metrics,
                artifacts=job.artifacts
            ))
        
        return {"training_jobs": jobs}
        
    except Exception as e:
        logger.error(f"Training job listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    try:
        if not model_manager:
            raise HTTPException(status_code=503, detail="Model manager not available")
        
        if model_id in model_manager.models:
            del model_manager.models[model_id]
            if model_id in model_manager.scalers:
                del model_manager.scalers[model_id]
            if model_id in model_manager.model_metadata:
                del model_manager.model_metadata[model_id]
            
            return {"message": f"Model {model_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
    except Exception as e:
        logger.error(f"Model deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring endpoints
@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    try:
        if not advanced_manager:
            raise HTTPException(status_code=503, detail="Advanced manager not available")
        
        # Get comprehensive metrics from advanced manager
        metrics = await advanced_manager.get_system_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the service
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8003)),
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info"
    )