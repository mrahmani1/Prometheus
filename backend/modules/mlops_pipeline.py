"""
MLOps Pipeline - Automated ML Lifecycle Management
Comprehensive model management, monitoring, and deployment system for geological AI models
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Protocol
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import time
import uuid
import hashlib
import pickle
import os
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
import tempfile
import zipfile
from abc import ABC, abstractmethod

# ML and model management imports
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Monitoring and drift detection
try:
    from scipy import stats
    from scipy.stats import ks_2samp, wasserstein_distance
    import matplotlib.pyplot as plt
    import seaborn as sns
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    warnings.warn("Statistical libraries not available. Drift detection will be limited.")

# MLflow integration
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Model tracking will be limited.")

# FastAPI for model serving
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    import uvicorn
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    warnings.warn("FastAPI not available. Model serving will be limited.")

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    warnings.warn("Prometheus client not available. Advanced metrics collection will be limited.")

warnings.filterwarnings('ignore')

class ModelStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

class DeploymentStrategy(Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    AB_TEST = "ab_test"

class ModelType(Enum):
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    TENSORFLOW = "tensorflow"

class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ModelMetadata:
    """Metadata for ML models"""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    stage: ModelStage
    created_at: datetime
    created_by: str
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_data_hash: Optional[str] = None
    model_size_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        result['stage'] = self.stage.value
        result['model_type'] = self.model_type.value
        return result

@dataclass
class ModelVersion:
    """Version information for models"""
    version: str
    model_id: str
    artifact_path: str
    metadata: ModelMetadata
    checksum: str
    parent_version: Optional[str] = None
    deployment_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    strategy: DeploymentStrategy
    traffic_percentage: float = 100.0
    max_replicas: int = 3
    min_replicas: int = 1
    resource_limits: Dict[str, str] = field(default_factory=lambda: {"cpu": "1000m", "memory": "2Gi"})
    health_check_path: str = "/health"
    readiness_timeout: int = 30
    liveness_timeout: int = 30

@dataclass
class PerformanceMetrics:
    """Model performance metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DriftReport:
    """Data drift detection report"""
    feature_name: str
    drift_detected: bool
    drift_score: float
    statistical_test: str
    p_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

class ModelRegistry:
    """Model version control and registry"""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__ + ".ModelRegistry")
        
        # Initialize registry database (SQLite in production)
        self.models: Dict[str, List[ModelVersion]] = {}
        self.active_models: Dict[str, str] = {}  # model_name -> version
        
        # Setup MLflow if available
        if MLFLOW_AVAILABLE:
            self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            mlflow.set_tracking_uri(f"file://{self.registry_path}/mlruns")
            self.mlflow_client = MlflowClient()
            self.logger.info("MLflow tracking initialized")
        except Exception as e:
            self.logger.warning(f"Failed to setup MLflow: {e}")
    
    def register_model(self, model: Any, metadata: ModelMetadata, 
                      model_path: Optional[str] = None) -> str:
        """Register a new model version"""
        try:
            version_id = f"{metadata.model_id}_v{metadata.version}"
            
            # Create model directory
            model_dir = self.registry_path / metadata.model_id / metadata.version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            if model_path:
                # Copy existing model file
                shutil.copy2(model_path, model_dir / "model.pkl")
            else:
                # Save model based on type
                if metadata.model_type == ModelType.PYTORCH:
                    torch.save(model.state_dict(), model_dir / "model.pth")
                    # Save model architecture
                    with open(model_dir / "model_info.json", 'w') as f:
                        json.dump({"model_class": model.__class__.__name__}, f)
                elif metadata.model_type == ModelType.SKLEARN:
                    joblib.dump(model, model_dir / "model.pkl")
            
            # Save metadata
            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Calculate model checksum
            checksum = self._calculate_checksum(model_dir)
            
            # Create version record
            model_version = ModelVersion(
                version=metadata.version,
                model_id=metadata.model_id,
                artifact_path=str(model_dir),
                metadata=metadata,
                checksum=checksum
            )
            
            # Update registry
            if metadata.model_id not in self.models:
                self.models[metadata.model_id] = []
            self.models[metadata.model_id].append(model_version)
            
            # MLflow logging
            if MLFLOW_AVAILABLE:
                self._log_to_mlflow(model, metadata)
            
            self.logger.info(f"Registered model {metadata.model_id} version {metadata.version}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise
    
    def _calculate_checksum(self, model_path: Path) -> str:
        """Calculate model checksum"""
        hasher = hashlib.sha256()
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
        return hasher.hexdigest()
    
    def _log_to_mlflow(self, model: Any, metadata: ModelMetadata):
        """Log model to MLflow"""
        try:
            with mlflow.start_run(run_name=f"{metadata.model_id}_v{metadata.version}"):
                # Log parameters
                mlflow.log_params(metadata.hyperparameters)
                
                # Log metrics
                mlflow.log_metrics(metadata.metrics)
                
                # Log model
                if metadata.model_type == ModelType.PYTORCH:
                    mlflow.pytorch.log_model(model, "model")
                elif metadata.model_type == ModelType.SKLEARN:
                    mlflow.sklearn.log_model(model, "model")
                
                # Log metadata
                mlflow.log_dict(metadata.to_dict(), "metadata.json")
                
        except Exception as e:
            self.logger.warning(f"Failed to log to MLflow: {e}")
    
    def get_model(self, model_id: str, version: Optional[str] = None) -> Tuple[Any, ModelMetadata]:
        """Retrieve model by ID and version"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        if version is None:
            # Get latest version
            version = self.models[model_id][-1].version
        
        # Find model version
        model_version = None
        for mv in self.models[model_id]:
            if mv.version == version:
                model_version = mv
                break
        
        if model_version is None:
            raise ValueError(f"Model {model_id} version {version} not found")
        
        # Load model
        model_path = Path(model_version.artifact_path)
        
        if model_version.metadata.model_type == ModelType.PYTORCH:
            # Load PyTorch model (would need model class definition)
            state_dict = torch.load(model_path / "model.pth")
            # Model instantiation would require architecture info
            model = state_dict  # Return state dict for now
        elif model_version.metadata.model_type == ModelType.SKLEARN:
            model = joblib.load(model_path / "model.pkl")
        else:
            raise ValueError(f"Unsupported model type: {model_version.metadata.model_type}")
        
        return model, model_version.metadata
    
    def promote_model(self, model_id: str, version: str, target_stage: ModelStage):
        """Promote model to different stage"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        for model_version in self.models[model_id]:
            if model_version.version == version:
                model_version.metadata.stage = target_stage
                
                # Update metadata file
                model_path = Path(model_version.artifact_path)
                with open(model_path / "metadata.json", 'w') as f:
                    json.dump(model_version.metadata.to_dict(), f, indent=2)
                
                self.logger.info(f"Promoted model {model_id} v{version} to {target_stage.value}")
                return
        
        raise ValueError(f"Model {model_id} version {version} not found")
    
    def list_models(self, stage: Optional[ModelStage] = None) -> List[ModelMetadata]:
        """List all models, optionally filtered by stage"""
        result = []
        for model_versions in self.models.values():
            for model_version in model_versions:
                if stage is None or model_version.metadata.stage == stage:
                    result.append(model_version.metadata)
        return result

class ABTestManager:
    """A/B testing infrastructure for models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".ABTestManager")
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_ab_test(self, test_name: str, model_a_id: str, model_b_id: str,
                      traffic_split: Tuple[float, float] = (0.5, 0.5),
                      duration_hours: int = 24,
                      success_metrics: List[str] = None) -> str:
        """Create new A/B test"""
        if success_metrics is None:
            success_metrics = ["accuracy", "response_time", "error_rate"]
        
        test_config = {
            "test_id": str(uuid.uuid4()),
            "test_name": test_name,
            "model_a_id": model_a_id,
            "model_b_id": model_b_id,
            "traffic_split": traffic_split,
            "start_time": datetime.utcnow(),
            "end_time": datetime.utcnow() + timedelta(hours=duration_hours),
            "success_metrics": success_metrics,
            "status": "active",
            "requests_a": 0,
            "requests_b": 0
        }
        
        test_id = test_config["test_id"]
        self.active_tests[test_id] = test_config
        self.test_results[test_id] = []
        
        self.logger.info(f"Created A/B test {test_name} ({test_id})")
        return test_id
    
    def route_request(self, test_id: str, request_id: str) -> str:
        """Route request to appropriate model"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        
        # Check if test is still active
        if datetime.utcnow() > test_config["end_time"]:
            test_config["status"] = "completed"
            return test_config["model_a_id"]  # Default to model A
        
        # Determine routing based on traffic split
        import random
        if random.random() < test_config["traffic_split"][0]:
            test_config["requests_a"] += 1
            return test_config["model_a_id"]
        else:
            test_config["requests_b"] += 1
            return test_config["model_b_id"]
    
    def log_result(self, test_id: str, model_id: str, metrics: Dict[str, float]):
        """Log A/B test result"""
        if test_id not in self.test_results:
            return
        
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "metrics": metrics
        }
        
        self.test_results[test_id].append(result)
    
    def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        if test_id not in self.active_tests or test_id not in self.test_results:
            raise ValueError(f"Test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        results = self.test_results[test_id]
        
        # Separate results by model
        results_a = [r for r in results if r["model_id"] == test_config["model_a_id"]]
        results_b = [r for r in results if r["model_id"] == test_config["model_b_id"]]
        
        analysis = {
            "test_id": test_id,
            "test_name": test_config["test_name"],
            "status": test_config["status"],
            "duration_hours": (datetime.utcnow() - test_config["start_time"]).total_seconds() / 3600,
            "total_requests": len(results),
            "model_a_requests": len(results_a),
            "model_b_requests": len(results_b),
            "model_a_metrics": {},
            "model_b_metrics": {},
            "statistical_significance": {}
        }
        
        # Calculate average metrics for each model
        if results_a:
            for metric in test_config["success_metrics"]:
                values = [r["metrics"].get(metric, 0) for r in results_a if metric in r["metrics"]]
                if values:
                    analysis["model_a_metrics"][metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values)
                    }
        
        if results_b:
            for metric in test_config["success_metrics"]:
                values = [r["metrics"].get(metric, 0) for r in results_b if metric in r["metrics"]]
                if values:
                    analysis["model_b_metrics"][metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values)
                    }
        
        # Statistical significance testing
        if STATS_AVAILABLE and results_a and results_b:
            for metric in test_config["success_metrics"]:
                values_a = [r["metrics"].get(metric, 0) for r in results_a if metric in r["metrics"]]
                values_b = [r["metrics"].get(metric, 0) for r in results_b if metric in r["metrics"]]
                
                if len(values_a) > 5 and len(values_b) > 5:
                    try:
                        statistic, p_value = stats.ttest_ind(values_a, values_b)
                        analysis["statistical_significance"][metric] = {
                            "statistic": statistic,
                            "p_value": p_value,
                            "significant": p_value < 0.05
                        }
                    except Exception as e:
                        self.logger.warning(f"Statistical test failed for {metric}: {e}")
        
        return analysis

class GradualRolloutManager:
    """Gradual rollout system for model deployments"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".GradualRolloutManager")
        self.active_rollouts: Dict[str, Dict[str, Any]] = {}
    
    def start_rollout(self, rollout_name: str, old_model_id: str, new_model_id: str,
                     stages: List[Dict[str, Any]] = None,
                     success_criteria: Dict[str, float] = None) -> str:
        """Start gradual rollout"""
        if stages is None:
            stages = [
                {"traffic_percentage": 5, "duration_minutes": 30},
                {"traffic_percentage": 25, "duration_minutes": 60},
                {"traffic_percentage": 50, "duration_minutes": 120},
                {"traffic_percentage": 100, "duration_minutes": 0}
            ]
        
        if success_criteria is None:
            success_criteria = {
                "error_rate_threshold": 0.05,
                "latency_p95_threshold": 500,  # ms
                "accuracy_threshold": 0.9
            }
        
        rollout_id = str(uuid.uuid4())
        rollout_config = {
            "rollout_id": rollout_id,
            "rollout_name": rollout_name,
            "old_model_id": old_model_id,
            "new_model_id": new_model_id,
            "stages": stages,
            "current_stage": 0,
            "current_traffic": 0,
            "start_time": datetime.utcnow(),
            "stage_start_time": datetime.utcnow(),
            "success_criteria": success_criteria,
            "status": "active",
            "metrics_history": []
        }
        
        self.active_rollouts[rollout_id] = rollout_config
        
        self.logger.info(f"Started gradual rollout {rollout_name} ({rollout_id})")
        return rollout_id
    
    def get_traffic_routing(self, rollout_id: str) -> Tuple[str, float]:
        """Get current traffic routing for rollout"""
        if rollout_id not in self.active_rollouts:
            raise ValueError(f"Rollout {rollout_id} not found")
        
        rollout = self.active_rollouts[rollout_id]
        
        if rollout["status"] != "active":
            return rollout["old_model_id"], 100.0
        
        # Check if should advance to next stage
        self._check_stage_advancement(rollout_id)
        
        current_traffic = rollout["current_traffic"]
        new_model_id = rollout["new_model_id"]
        
        return new_model_id, current_traffic
    
    def _check_stage_advancement(self, rollout_id: str):
        """Check if rollout should advance to next stage"""
        rollout = self.active_rollouts[rollout_id]
        
        if rollout["current_stage"] >= len(rollout["stages"]):
            rollout["status"] = "completed"
            return
        
        current_stage = rollout["stages"][rollout["current_stage"]]
        stage_duration = timedelta(minutes=current_stage["duration_minutes"])
        
        # Check if stage duration has passed
        if datetime.utcnow() - rollout["stage_start_time"] >= stage_duration:
            # Check success criteria before advancing
            if self._evaluate_success_criteria(rollout_id):
                # Advance to next stage
                rollout["current_stage"] += 1
                rollout["stage_start_time"] = datetime.utcnow()
                
                if rollout["current_stage"] < len(rollout["stages"]):
                    next_stage = rollout["stages"][rollout["current_stage"]]
                    rollout["current_traffic"] = next_stage["traffic_percentage"]
                    self.logger.info(f"Advanced rollout {rollout_id} to stage {rollout['current_stage']}")
                else:
                    rollout["status"] = "completed"
                    rollout["current_traffic"] = 100
                    self.logger.info(f"Completed rollout {rollout_id}")
            else:
                # Rollback due to failed criteria
                self._rollback(rollout_id)
    
    def _evaluate_success_criteria(self, rollout_id: str) -> bool:
        """Evaluate if success criteria are met"""
        rollout = self.active_rollouts[rollout_id]
        
        # In a real implementation, this would check actual metrics
        # For now, return True (successful)
        return True
    
    def _rollback(self, rollout_id: str):
        """Rollback deployment"""
        rollout = self.active_rollouts[rollout_id]
        rollout["status"] = "rolled_back"
        rollout["current_traffic"] = 0
        
        self.logger.warning(f"Rolled back deployment {rollout_id}")
    
    def manual_rollback(self, rollout_id: str):
        """Manually trigger rollback"""
        if rollout_id in self.active_rollouts:
            self._rollback(rollout_id)

class DataDriftDetector:
    """Data drift detection system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".DataDriftDetector")
        self.reference_data: Dict[str, np.ndarray] = {}
        self.feature_statistics: Dict[str, Dict[str, Any]] = {}
    
    def set_reference_data(self, feature_name: str, reference_data: np.ndarray):
        """Set reference data for drift detection"""
        self.reference_data[feature_name] = reference_data
        
        # Calculate reference statistics
        self.feature_statistics[feature_name] = {
            "mean": np.mean(reference_data),
            "std": np.std(reference_data),
            "min": np.min(reference_data),
            "max": np.max(reference_data),
            "quantiles": np.quantile(reference_data, [0.25, 0.5, 0.75])
        }
        
        self.logger.info(f"Set reference data for feature {feature_name}")
    
    def detect_drift(self, feature_name: str, current_data: np.ndarray,
                    method: str = "ks_test", threshold: float = 0.05) -> DriftReport:
        """Detect data drift"""
        if feature_name not in self.reference_data:
            raise ValueError(f"No reference data for feature {feature_name}")
        
        reference_data = self.reference_data[feature_name]
        
        if method == "ks_test" and STATS_AVAILABLE:
            # Kolmogorov-Smirnov test
            statistic, p_value = ks_2samp(reference_data, current_data)
            drift_detected = p_value < threshold
            drift_score = 1 - p_value
            
        elif method == "wasserstein" and STATS_AVAILABLE:
            # Wasserstein distance
            distance = wasserstein_distance(reference_data, current_data)
            # Normalize by reference data range
            ref_range = np.max(reference_data) - np.min(reference_data)
            drift_score = distance / (ref_range + 1e-8)
            drift_detected = drift_score > threshold
            p_value = 1 - drift_score  # Approximate
            
        else:
            # Simple statistical comparison
            ref_mean = np.mean(reference_data)
            ref_std = np.std(reference_data)
            curr_mean = np.mean(current_data)
            
            # Z-score based drift detection
            z_score = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
            drift_score = z_score / 3.0  # Normalize to [0,1] approximately
            drift_detected = z_score > 2.0  # 2 standard deviations
            p_value = 1 - drift_score
        
        report = DriftReport(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=drift_score,
            statistical_test=method,
            p_value=p_value,
            threshold=threshold
        )
        
        if drift_detected:
            self.logger.warning(f"Drift detected for feature {feature_name}: score={drift_score:.3f}")
        else:
            self.logger.debug(f"No drift detected for feature {feature_name}")
        
        return report

class ModelMonitor:
    """Comprehensive model monitoring system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".ModelMonitor")
        self.metrics_history: Dict[str, List[PerformanceMetrics]] = {}
        self.drift_detector = DataDriftDetector()
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        self.resource_metrics: Dict[str, List[Dict[str, Any]]] = {}
        
        # Setup Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.prediction_counter = Counter('model_predictions_total', 'Total predictions made', ['model_id'])
        self.prediction_latency = Histogram('model_prediction_duration_seconds', 'Prediction latency', ['model_id'])
        self.accuracy_gauge = Gauge('model_accuracy', 'Model accuracy', ['model_id'])
        self.drift_gauge = Gauge('feature_drift_score', 'Feature drift score', ['feature_name'])
    
    def log_prediction_metrics(self, model_id: str, metrics: PerformanceMetrics):
        """Log prediction performance metrics"""
        if model_id not in self.metrics_history:
            self.metrics_history[model_id] = []
        
        self.metrics_history[model_id].append(metrics)
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            if metrics.accuracy is not None:
                self.accuracy_gauge.labels(model_id=model_id).set(metrics.accuracy)
            self.prediction_counter.labels(model_id=model_id).inc()
        
        # Check for degradation
        self._check_model_degradation(model_id, metrics)
    
    def _check_model_degradation(self, model_id: str, current_metrics: PerformanceMetrics):
        """Check for model performance degradation"""
        if model_id not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[model_id]
        alerts = []
        
        # Check accuracy degradation
        if current_metrics.accuracy is not None and "accuracy_threshold" in thresholds:
            if current_metrics.accuracy < thresholds["accuracy_threshold"]:
                alerts.append({
                    "type": "accuracy_degradation",
                    "level": AlertLevel.HIGH,
                    "message": f"Accuracy dropped to {current_metrics.accuracy:.3f}",
                    "threshold": thresholds["accuracy_threshold"],
                    "current_value": current_metrics.accuracy
                })
        
        # Check error rate increase
        error_rate = current_metrics.custom_metrics.get("error_rate")
        if error_rate is not None and "error_rate_threshold" in thresholds:
            if error_rate > thresholds["error_rate_threshold"]:
                alerts.append({
                    "type": "error_rate_increase",
                    "level": AlertLevel.CRITICAL,
                    "message": f"Error rate increased to {error_rate:.3f}",
                    "threshold": thresholds["error_rate_threshold"],
                    "current_value": error_rate
                })
        
        # Send alerts
        for alert in alerts:
            self._send_alert(model_id, alert)
    
    def _send_alert(self, model_id: str, alert: Dict[str, Any]):
        """Send model degradation alert"""
        self.logger.warning(f"ALERT for model {model_id}: {alert['message']}")
        
        # In production, this would integrate with alerting systems like:
        # - Slack notifications
        # - Email alerts  
        # - PagerDuty
        # - Webhook callbacks
    
    def set_alert_thresholds(self, model_id: str, thresholds: Dict[str, float]):
        """Set alert thresholds for model"""
        self.alert_thresholds[model_id] = thresholds
        self.logger.info(f"Set alert thresholds for model {model_id}: {thresholds}")
    
    def log_resource_metrics(self, model_id: str, cpu_usage: float, memory_usage: float, 
                           gpu_usage: Optional[float] = None):
        """Log resource utilization metrics"""
        if model_id not in self.resource_metrics:
            self.resource_metrics[model_id] = []
        
        resource_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "gpu_usage": gpu_usage
        }
        
        self.resource_metrics[model_id].append(resource_data)
        
        # Keep only last 1000 entries
        if len(self.resource_metrics[model_id]) > 1000:
            self.resource_metrics[model_id] = self.resource_metrics[model_id][-1000:]
    
    def generate_monitoring_report(self, model_id: str, 
                                 time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        if time_range is None:
            time_range = timedelta(hours=24)
        
        cutoff_time = datetime.utcnow() - time_range
        
        report = {
            "model_id": model_id,
            "report_time": datetime.utcnow().isoformat(),
            "time_range_hours": time_range.total_seconds() / 3600,
            "performance_summary": {},
            "resource_summary": {},
            "drift_summary": [],
            "alerts": []
        }
        
        # Performance metrics summary
        if model_id in self.metrics_history:
            recent_metrics = [
                m for m in self.metrics_history[model_id]
                if m.timestamp >= cutoff_time
            ]
            
            if recent_metrics:
                accuracies = [m.accuracy for m in recent_metrics if m.accuracy is not None]
                if accuracies:
                    report["performance_summary"]["accuracy"] = {
                        "mean": np.mean(accuracies),
                        "std": np.std(accuracies),
                        "min": np.min(accuracies),
                        "max": np.max(accuracies),
                        "trend": "stable"  # Would calculate actual trend
                    }
        
        # Resource utilization summary
        if model_id in self.resource_metrics:
            recent_resources = [
                r for r in self.resource_metrics[model_id]
                if datetime.fromisoformat(r["timestamp"]) >= cutoff_time
            ]
            
            if recent_resources:
                cpu_usage = [r["cpu_usage"] for r in recent_resources]
                memory_usage = [r["memory_usage"] for r in recent_resources]
                
                report["resource_summary"] = {
                    "cpu": {
                        "mean": np.mean(cpu_usage),
                        "max": np.max(cpu_usage),
                        "p95": np.percentile(cpu_usage, 95)
                    },
                    "memory": {
                        "mean": np.mean(memory_usage),
                        "max": np.max(memory_usage),
                        "p95": np.percentile(memory_usage, 95)
                    }
                }
        
        return report

class MLOpsPipeline:
    """Main MLOps pipeline orchestrator"""
    
    def __init__(self, registry_path: str = "./mlops_registry"):
        self.logger = logging.getLogger(__name__ + ".MLOpsPipeline")
        
        # Initialize components
        self.model_registry = ModelRegistry(registry_path)
        self.ab_test_manager = ABTestManager()
        self.rollout_manager = GradualRolloutManager()
        self.monitor = ModelMonitor()
        
        # Pipeline state
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("MLOps Pipeline initialized")
    
    async def deploy_model(self, model: Any, metadata: ModelMetadata,
                          deployment_config: DeploymentConfig) -> str:
        """Deploy model with specified strategy"""
        try:
            # Register model
            version_id = self.model_registry.register_model(model, metadata)
            
            # Create deployment record
            deployment_id = str(uuid.uuid4())
            deployment_record = {
                "deployment_id": deployment_id,
                "model_id": metadata.model_id,
                "version": metadata.version,
                "strategy": deployment_config.strategy,
                "status": "deploying",
                "created_at": datetime.utcnow(),
                "config": asdict(deployment_config)
            }
            
            self.active_deployments[deployment_id] = deployment_record
            
            # Execute deployment based on strategy
            if deployment_config.strategy == DeploymentStrategy.CANARY:
                await self._deploy_canary(deployment_id, deployment_config)
            elif deployment_config.strategy == DeploymentStrategy.AB_TEST:
                await self._deploy_ab_test(deployment_id, deployment_config)
            elif deployment_config.strategy == DeploymentStrategy.ROLLING:
                await self._deploy_rolling(deployment_id, deployment_config)
            else:
                # Blue-green deployment
                await self._deploy_blue_green(deployment_id, deployment_config)
            
            deployment_record["status"] = "deployed"
            
            # Setup monitoring
            self.monitor.set_alert_thresholds(metadata.model_id, {
                "accuracy_threshold": 0.85,
                "error_rate_threshold": 0.1
            })
            
            self.logger.info(f"Successfully deployed model {metadata.model_id} v{metadata.version}")
            return deployment_id
            
        except Exception as e:
            self.logger.error(f"Failed to deploy model: {e}")
            raise
    
    async def _deploy_canary(self, deployment_id: str, config: DeploymentConfig):
        """Deploy using canary strategy"""
        deployment = self.active_deployments[deployment_id]
        
        # Start gradual rollout
        rollout_id = self.rollout_manager.start_rollout(
            rollout_name=f"canary_{deployment_id}",
            old_model_id="current_production",  # Would be actual current model
            new_model_id=deployment["model_id"]
        )
        
        deployment["rollout_id"] = rollout_id
        self.logger.info(f"Started canary deployment {deployment_id}")
    
    async def _deploy_ab_test(self, deployment_id: str, config: DeploymentConfig):
        """Deploy using A/B testing strategy"""
        deployment = self.active_deployments[deployment_id]
        
        # Create A/B test
        test_id = self.ab_test_manager.create_ab_test(
            test_name=f"ab_test_{deployment_id}",
            model_a_id="current_production",
            model_b_id=deployment["model_id"],
            traffic_split=(0.8, 0.2)  # 20% to new model
        )
        
        deployment["ab_test_id"] = test_id
        self.logger.info(f"Started A/B test deployment {deployment_id}")
    
    async def _deploy_rolling(self, deployment_id: str, config: DeploymentConfig):
        """Deploy using rolling update strategy"""
        # In production, this would coordinate with container orchestration
        self.logger.info(f"Started rolling deployment {deployment_id}")
    
    async def _deploy_blue_green(self, deployment_id: str, config: DeploymentConfig):
        """Deploy using blue-green strategy"""
        # In production, this would setup parallel environment and switch traffic
        self.logger.info(f"Started blue-green deployment {deployment_id}")
    
    def rollback_deployment(self, deployment_id: str):
        """Rollback deployment"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.active_deployments[deployment_id]
        
        # Execute rollback based on strategy
        if "rollout_id" in deployment:
            self.rollout_manager.manual_rollback(deployment["rollout_id"])
        
        deployment["status"] = "rolled_back"
        deployment["rollback_time"] = datetime.utcnow()
        
        self.logger.info(f"Rolled back deployment {deployment_id}")
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        return self.active_deployments[deployment_id]
    
    def get_pipeline_dashboard(self) -> Dict[str, Any]:
        """Get overall pipeline status dashboard"""
        dashboard = {
            "timestamp": datetime.utcnow().isoformat(),
            "models": {
                "total_registered": len(self.model_registry.models),
                "production_models": len(self.model_registry.list_models(ModelStage.PRODUCTION)),
                "staging_models": len(self.model_registry.list_models(ModelStage.STAGING))
            },
            "deployments": {
                "active_deployments": len([d for d in self.active_deployments.values() 
                                         if d["status"] == "deployed"]),
                "total_deployments": len(self.active_deployments)
            },
            "testing": {
                "active_ab_tests": len([t for t in self.ab_test_manager.active_tests.values() 
                                      if t["status"] == "active"]),
                "active_rollouts": len([r for r in self.rollout_manager.active_rollouts.values() 
                                      if r["status"] == "active"])
            },
            "monitoring": {
                "monitored_models": len(self.monitor.metrics_history),
                "resource_tracked_models": len(self.monitor.resource_metrics)
            }
        }
        
        return dashboard

# Example usage and testing
async def example_mlops_pipeline():
    """Example of using MLOps pipeline"""
    
    print("=== MLOps Pipeline Example ===")
    
    # Initialize pipeline
    pipeline = MLOpsPipeline("./example_mlops_registry")
    
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = SimpleModel()
    
    # Model metadata
    metadata = ModelMetadata(
        model_id="earthquake_predictor",
        name="Earthquake Prediction Model",
        version="1.0.0",
        model_type=ModelType.PYTORCH,
        stage=ModelStage.STAGING,
        created_at=datetime.utcnow(),
        created_by="data_scientist_1",
        description="Deep learning model for earthquake prediction",
        tags={"domain": "seismology", "type": "prediction"},
        metrics={"accuracy": 0.92, "f1_score": 0.89},
        hyperparameters={"learning_rate": 0.001, "batch_size": 32}
    )
    
    # Deployment configuration
    deployment_config = DeploymentConfig(
        strategy=DeploymentStrategy.CANARY,
        traffic_percentage=10.0,
        max_replicas=3
    )
    
    # Deploy model
    deployment_id = await pipeline.deploy_model(model, metadata, deployment_config)
    print(f"Deployed model with deployment ID: {deployment_id}")
    
    # Simulate some monitoring data
    for i in range(10):
        metrics = PerformanceMetrics(
            accuracy=0.92 - i * 0.001,  # Slight degradation
            f1_score=0.89,
            custom_metrics={"response_time_ms": 150 + i * 5}
        )
        pipeline.monitor.log_prediction_metrics("earthquake_predictor", metrics)
        
        # Resource metrics
        pipeline.monitor.log_resource_metrics(
            "earthquake_predictor",
            cpu_usage=50 + i * 2,
            memory_usage=1024 + i * 10
        )
    
    # Create A/B test
    test_id = pipeline.ab_test_manager.create_ab_test(
        "earthquake_model_comparison",
        "earthquake_predictor_v1",
        "earthquake_predictor_v2",
        duration_hours=1
    )
    print(f"Created A/B test: {test_id}")
    
    # Simulate test results
    for i in range(50):
        model_id = pipeline.ab_test_manager.route_request(test_id, f"request_{i}")
        test_metrics = {"accuracy": 0.91 + np.random.normal(0, 0.02)}
        pipeline.ab_test_manager.log_result(test_id, model_id, test_metrics)
    
    # Analyze A/B test
    test_analysis = pipeline.ab_test_manager.analyze_test(test_id)
    print(f"A/B Test Analysis: {test_analysis['model_a_requests']} vs {test_analysis['model_b_requests']} requests")
    
    # Generate monitoring report
    monitoring_report = pipeline.monitor.generate_monitoring_report("earthquake_predictor")
    print(f"Monitoring Report: {monitoring_report['performance_summary']}")
    
    # Get pipeline dashboard
    dashboard = pipeline.get_pipeline_dashboard()
    print(f"Pipeline Dashboard:")
    print(f"  Total registered models: {dashboard['models']['total_registered']}")
    print(f"  Active deployments: {dashboard['deployments']['active_deployments']}")
    print(f"  Active A/B tests: {dashboard['testing']['active_ab_tests']}")
    
    # Demonstrate drift detection
    reference_data = np.random.normal(0, 1, 1000)
    current_data = np.random.normal(0.5, 1.2, 1000)  # Shifted and scaled
    
    pipeline.monitor.drift_detector.set_reference_data("seismic_feature", reference_data)
    drift_report = pipeline.monitor.drift_detector.detect_drift("seismic_feature", current_data)
    
    print(f"Drift Detection: {drift_report.drift_detected}, Score: {drift_report.drift_score:.3f}")
    
    print("MLOps pipeline example completed!")

if __name__ == "__main__":
    print("=== MLOps Pipeline - Automated ML Lifecycle Management ===")
    print(f"Available features:")
    print(f"  Statistical Analysis: {STATS_AVAILABLE}")
    print(f"  MLflow Tracking: {MLFLOW_AVAILABLE}")
    print(f"  FastAPI Serving: {FASTAPI_AVAILABLE}")
    print(f"  Prometheus Metrics: {PROMETHEUS_AVAILABLE}")
    
    # Run example
    asyncio.run(example_mlops_pipeline())