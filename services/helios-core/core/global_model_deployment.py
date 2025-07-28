"""
Global Model Deployment System

This module implements model versioning, distribution infrastructure, automated
deployment to different geological regions, performance monitoring, and drift detection
for the Helios AI models.

Requirements addressed: 5.3, 5.4, 5.5
"""

import torch
import torch.nn as nn
import numpy as np
import json
import hashlib
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
import pickle
import zipfile
import shutil
from enum import Enum
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
import time

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLBACK = "rollback"
    DEPRECATED = "deprecated"


class ModelPerformanceStatus(Enum):
    """Model performance status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DRIFT_DETECTED = "drift_detected"


@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_name: str
    version_number: str
    creation_timestamp: datetime
    model_hash: str
    model_size_bytes: int
    training_data_hash: str
    performance_metrics: Dict[str, float]
    geological_domains: List[str]
    dependencies: List[str]
    metadata: Dict[str, Any]
    deployment_config: Dict[str, Any]


@dataclass
class DeploymentTarget:
    """Deployment target information"""
    target_id: str
    region_name: str
    geological_domain: str
    endpoint_url: str
    deployment_type: str  # "cloud", "edge", "on_premise"
    hardware_specs: Dict[str, Any]
    network_config: Dict[str, Any]
    security_config: Dict[str, Any]
    auto_update_enabled: bool = True
    monitoring_enabled: bool = True


@dataclass
class DeploymentRecord:
    """Deployment record"""
    deployment_id: str
    model_version: ModelVersion
    target: DeploymentTarget
    deployment_timestamp: datetime
    status: DeploymentStatus
    deployment_logs: List[str]
    performance_history: List[Dict[str, Any]]
    last_health_check: Optional[datetime] = None
    rollback_version: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for deployed models"""
    timestamp: datetime
    target_id: str
    model_version: str
    accuracy: float
    latency_ms: float
    throughput_rps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate: float
    drift_score: float
    geological_consistency: float


class ModelStorage(ABC):
    """Abstract base class for model storage backends"""
    
    @abstractmethod
    async def store_model(self, model_version: ModelVersion, model_data: bytes) -> str:
        """Store model data and return storage path"""
        pass
    
    @abstractmethod
    async def retrieve_model(self, version_id: str) -> Tuple[ModelVersion, bytes]:
        """Retrieve model data by version ID"""
        pass
    
    @abstractmethod
    async def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model"""
        pass
    
    @abstractmethod
    async def delete_version(self, version_id: str) -> bool:
        """Delete a model version"""
        pass


class LocalModelStorage(ModelStorage):
    """Local filesystem model storage"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_path / "model_registry.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load model metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save model metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    async def store_model(self, model_version: ModelVersion, model_data: bytes) -> str:
        """Store model data locally"""
        model_dir = self.storage_path / model_version.model_name / model_version.version_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Store model data
        model_file = model_dir / "model.pkl"
        with open(model_file, 'wb') as f:
            f.write(model_data)
        
        # Store metadata
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(model_version), f, indent=2, default=str)
        
        # Update registry
        if model_version.model_name not in self.metadata:
            self.metadata[model_version.model_name] = {}
        
        self.metadata[model_version.model_name][model_version.version_id] = {
            'version_number': model_version.version_number,
            'creation_timestamp': model_version.creation_timestamp.isoformat(),
            'model_hash': model_version.model_hash,
            'storage_path': str(model_file)
        }
        
        self._save_metadata()
        
        logger.info(f"Stored model {model_version.model_name} v{model_version.version_number}")
        return str(model_file)
    
    async def retrieve_model(self, version_id: str) -> Tuple[ModelVersion, bytes]:
        """Retrieve model data from local storage"""
        # Find model in metadata
        for model_name, versions in self.metadata.items():
            if version_id in versions:
                model_dir = self.storage_path / model_name / version_id
                
                # Load metadata
                metadata_file = model_dir / "metadata.json"
                with open(metadata_file, 'r') as f:
                    version_data = json.load(f)
                
                # Reconstruct ModelVersion object
                version_data['creation_timestamp'] = datetime.fromisoformat(version_data['creation_timestamp'])
                model_version = ModelVersion(**version_data)
                
                # Load model data
                model_file = model_dir / "model.pkl"
                with open(model_file, 'rb') as f:
                    model_data = f.read()
                
                return model_version, model_data
        
        raise ValueError(f"Model version {version_id} not found")
    
    async def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model"""
        if model_name not in self.metadata:
            return []
        
        versions = []
        for version_id in self.metadata[model_name]:
            try:
                model_version, _ = await self.retrieve_model(version_id)
                versions.append(model_version)
            except Exception as e:
                logger.warning(f"Failed to load version {version_id}: {e}")
        
        return sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)
    
    async def delete_version(self, version_id: str) -> bool:
        """Delete a model version"""
        for model_name, versions in self.metadata.items():
            if version_id in versions:
                model_dir = self.storage_path / model_name / version_id
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                
                del self.metadata[model_name][version_id]
                self._save_metadata()
                
                logger.info(f"Deleted model version {version_id}")
                return True
        
        return False


class CloudModelStorage(ModelStorage):
    """Cloud-based model storage (S3-compatible)"""
    
    def __init__(self, bucket_name: str, access_key: str, secret_key: str, endpoint_url: str):
        self.bucket_name = bucket_name
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint_url = endpoint_url
        # In a real implementation, would use boto3 or similar
        logger.info(f"Initialized cloud storage for bucket: {bucket_name}")
    
    async def store_model(self, model_version: ModelVersion, model_data: bytes) -> str:
        """Store model in cloud storage"""
        # Placeholder implementation
        storage_key = f"{model_version.model_name}/{model_version.version_id}/model.pkl"
        logger.info(f"Would store model to cloud: {storage_key}")
        return storage_key
    
    async def retrieve_model(self, version_id: str) -> Tuple[ModelVersion, bytes]:
        """Retrieve model from cloud storage"""
        # Placeholder implementation
        raise NotImplementedError("Cloud storage retrieval not implemented in demo")
    
    async def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List model versions from cloud storage"""
        # Placeholder implementation
        return []
    
    async def delete_version(self, version_id: str) -> bool:
        """Delete model version from cloud storage"""
        # Placeholder implementation
        return True


class ModelVersionManager:
    """Manages model versions and deployment lifecycle"""
    
    def __init__(self, storage: ModelStorage):
        self.storage = storage
        self.version_counter = {}
    
    def _generate_version_id(self, model_name: str) -> str:
        """Generate unique version ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        counter = self.version_counter.get(model_name, 0) + 1
        self.version_counter[model_name] = counter
        return f"{model_name}_{timestamp}_{counter:03d}"
    
    def _calculate_model_hash(self, model: nn.Module) -> str:
        """Calculate hash of model parameters"""
        model_str = ""
        for name, param in model.named_parameters():
            model_str += f"{name}:{param.data.cpu().numpy().tobytes().hex()}"
        
        return hashlib.sha256(model_str.encode()).hexdigest()[:16]
    
    def _serialize_model(self, model: nn.Module) -> bytes:
        """Serialize model to bytes"""
        return pickle.dumps({
            'state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'architecture': str(model)
        })
    
    def _deserialize_model(self, model_data: bytes, model_class: type) -> nn.Module:
        """Deserialize model from bytes"""
        data = pickle.loads(model_data)
        
        # This is simplified - in practice would need proper model reconstruction
        model = model_class()
        model.load_state_dict(data['state_dict'])
        
        return model
    
    async def create_version(
        self,
        model: nn.Module,
        model_name: str,
        training_data_hash: str,
        performance_metrics: Dict[str, float],
        geological_domains: List[str],
        metadata: Dict[str, Any] = None
    ) -> ModelVersion:
        """Create a new model version"""
        version_id = self._generate_version_id(model_name)
        model_hash = self._calculate_model_hash(model)
        model_data = self._serialize_model(model)
        
        # Generate version number
        existing_versions = await self.storage.list_versions(model_name)
        version_number = f"1.{len(existing_versions)}.0"
        
        model_version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            version_number=version_number,
            creation_timestamp=datetime.utcnow(),
            model_hash=model_hash,
            model_size_bytes=len(model_data),
            training_data_hash=training_data_hash,
            performance_metrics=performance_metrics,
            geological_domains=geological_domains,
            dependencies=[],  # Would be populated with actual dependencies
            metadata=metadata or {},
            deployment_config={}
        )
        
        # Store model
        await self.storage.store_model(model_version, model_data)
        
        logger.info(f"Created model version {version_id} for {model_name}")
        return model_version
    
    async def get_version(self, version_id: str) -> ModelVersion:
        """Get model version by ID"""
        model_version, _ = await self.storage.retrieve_model(version_id)
        return model_version
    
    async def get_model(self, version_id: str, model_class: type) -> nn.Module:
        """Get model instance by version ID"""
        model_version, model_data = await self.storage.retrieve_model(version_id)
        return self._deserialize_model(model_data, model_class)
    
    async def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model"""
        return await self.storage.list_versions(model_name)
    
    async def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model"""
        versions = await self.list_versions(model_name)
        return versions[0] if versions else None
    
    async def deprecate_version(self, version_id: str) -> bool:
        """Mark a version as deprecated"""
        # In a real implementation, would update metadata
        logger.info(f"Deprecated model version {version_id}")
        return True


class DeploymentOrchestrator:
    """Orchestrates model deployments across different targets"""
    
    def __init__(self, version_manager: ModelVersionManager):
        self.version_manager = version_manager
        self.deployment_targets = {}
        self.deployment_records = {}
        self.deployment_queue = asyncio.Queue()
        self.deployment_worker_running = False
    
    def register_target(self, target: DeploymentTarget):
        """Register a deployment target"""
        self.deployment_targets[target.target_id] = target
        logger.info(f"Registered deployment target: {target.region_name}")
    
    def unregister_target(self, target_id: str):
        """Unregister a deployment target"""
        if target_id in self.deployment_targets:
            del self.deployment_targets[target_id]
            logger.info(f"Unregistered deployment target: {target_id}")
    
    async def deploy_model(
        self,
        version_id: str,
        target_id: str,
        deployment_config: Dict[str, Any] = None
    ) -> str:
        """Deploy a model version to a target"""
        if target_id not in self.deployment_targets:
            raise ValueError(f"Deployment target {target_id} not found")
        
        model_version = await self.version_manager.get_version(version_id)
        target = self.deployment_targets[target_id]
        
        deployment_id = f"deploy_{version_id}_{target_id}_{int(time.time())}"
        
        deployment_record = DeploymentRecord(
            deployment_id=deployment_id,
            model_version=model_version,
            target=target,
            deployment_timestamp=datetime.utcnow(),
            status=DeploymentStatus.PENDING,
            deployment_logs=[],
            performance_history=[]
        )
        
        self.deployment_records[deployment_id] = deployment_record
        
        # Add to deployment queue
        await self.deployment_queue.put((deployment_id, deployment_config or {}))
        
        # Start deployment worker if not running
        if not self.deployment_worker_running:
            asyncio.create_task(self._deployment_worker())
        
        logger.info(f"Queued deployment {deployment_id}")
        return deployment_id
    
    async def _deployment_worker(self):
        """Background worker for processing deployments"""
        self.deployment_worker_running = True
        
        try:
            while True:
                try:
                    deployment_id, config = await asyncio.wait_for(
                        self.deployment_queue.get(), timeout=1.0
                    )
                    await self._execute_deployment(deployment_id, config)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Deployment worker error: {e}")
        finally:
            self.deployment_worker_running = False
    
    async def _execute_deployment(self, deployment_id: str, config: Dict[str, Any]):
        """Execute a single deployment"""
        record = self.deployment_records[deployment_id]
        
        try:
            record.status = DeploymentStatus.DEPLOYING
            record.deployment_logs.append(f"Started deployment at {datetime.utcnow()}")
            
            # Simulate deployment steps
            await self._prepare_deployment_environment(record, config)
            await self._transfer_model_artifacts(record)
            await self._install_dependencies(record)
            await self._start_model_service(record)
            await self._verify_deployment(record)
            
            record.status = DeploymentStatus.DEPLOYED
            record.deployment_logs.append(f"Deployment completed at {datetime.utcnow()}")
            
            logger.info(f"Successfully deployed {deployment_id}")
            
        except Exception as e:
            record.status = DeploymentStatus.FAILED
            record.deployment_logs.append(f"Deployment failed: {str(e)}")
            logger.error(f"Deployment {deployment_id} failed: {e}")
    
    async def _prepare_deployment_environment(self, record: DeploymentRecord, config: Dict[str, Any]):
        """Prepare deployment environment"""
        record.deployment_logs.append("Preparing deployment environment")
        await asyncio.sleep(0.1)  # Simulate work
    
    async def _transfer_model_artifacts(self, record: DeploymentRecord):
        """Transfer model artifacts to target"""
        record.deployment_logs.append("Transferring model artifacts")
        await asyncio.sleep(0.2)  # Simulate transfer
    
    async def _install_dependencies(self, record: DeploymentRecord):
        """Install model dependencies"""
        record.deployment_logs.append("Installing dependencies")
        await asyncio.sleep(0.1)  # Simulate installation
    
    async def _start_model_service(self, record: DeploymentRecord):
        """Start model service"""
        record.deployment_logs.append("Starting model service")
        await asyncio.sleep(0.1)  # Simulate service start
    
    async def _verify_deployment(self, record: DeploymentRecord):
        """Verify deployment is working"""
        record.deployment_logs.append("Verifying deployment")
        
        # Simulate health check
        if record.target.endpoint_url:
            # Would make actual HTTP request in real implementation
            record.deployment_logs.append("Health check passed")
        
        await asyncio.sleep(0.1)
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """Get deployment status"""
        return self.deployment_records.get(deployment_id)
    
    async def list_deployments(self, target_id: str = None) -> List[DeploymentRecord]:
        """List deployments, optionally filtered by target"""
        deployments = list(self.deployment_records.values())
        
        if target_id:
            deployments = [d for d in deployments if d.target.target_id == target_id]
        
        return sorted(deployments, key=lambda d: d.deployment_timestamp, reverse=True)
    
    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment to previous version"""
        record = self.deployment_records.get(deployment_id)
        if not record:
            return False
        
        if record.rollback_version:
            # Deploy rollback version
            rollback_deployment_id = await self.deploy_model(
                record.rollback_version,
                record.target.target_id
            )
            
            record.status = DeploymentStatus.ROLLBACK
            record.deployment_logs.append(f"Rolled back to {record.rollback_version}")
            
            logger.info(f"Rolled back deployment {deployment_id}")
            return True
        
        return False


class PerformanceMonitor:
    """Monitors performance of deployed models"""
    
    def __init__(self):
        self.metrics_history = {}
        self.alert_thresholds = {
            'accuracy': 0.7,
            'latency_ms': 1000,
            'error_rate': 0.1,
            'drift_score': 0.3
        }
        self.monitoring_active = {}
    
    async def start_monitoring(self, deployment_id: str, target: DeploymentTarget):
        """Start monitoring a deployment"""
        self.monitoring_active[deployment_id] = True
        self.metrics_history[deployment_id] = []
        
        # Start monitoring task
        asyncio.create_task(self._monitoring_loop(deployment_id, target))
        
        logger.info(f"Started monitoring deployment {deployment_id}")
    
    async def stop_monitoring(self, deployment_id: str):
        """Stop monitoring a deployment"""
        self.monitoring_active[deployment_id] = False
        logger.info(f"Stopped monitoring deployment {deployment_id}")
    
    async def _monitoring_loop(self, deployment_id: str, target: DeploymentTarget):
        """Main monitoring loop"""
        while self.monitoring_active.get(deployment_id, False):
            try:
                metrics = await self._collect_metrics(deployment_id, target)
                self.metrics_history[deployment_id].append(metrics)
                
                # Check for alerts
                await self._check_alerts(deployment_id, metrics)
                
                # Keep only recent metrics (last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics_history[deployment_id] = [
                    m for m in self.metrics_history[deployment_id]
                    if m.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Monitoring error for {deployment_id}: {e}")
                await asyncio.sleep(60)
    
    async def _collect_metrics(self, deployment_id: str, target: DeploymentTarget) -> PerformanceMetrics:
        """Collect performance metrics from deployment"""
        # Simulate metric collection
        # In real implementation, would query the deployed service
        
        base_accuracy = 0.85
        base_latency = 200
        
        # Add some random variation
        import random
        accuracy = base_accuracy + random.uniform(-0.1, 0.05)
        latency = base_latency + random.uniform(-50, 100)
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            target_id=target.target_id,
            model_version="1.0.0",  # Would get from deployment record
            accuracy=max(0.0, min(1.0, accuracy)),
            latency_ms=max(0, latency),
            throughput_rps=random.uniform(10, 50),
            memory_usage_mb=random.uniform(500, 1000),
            cpu_usage_percent=random.uniform(20, 80),
            error_rate=random.uniform(0, 0.05),
            drift_score=random.uniform(0, 0.2),
            geological_consistency=random.uniform(0.8, 1.0)
        )
    
    async def _check_alerts(self, deployment_id: str, metrics: PerformanceMetrics):
        """Check metrics against alert thresholds"""
        alerts = []
        
        if metrics.accuracy < self.alert_thresholds['accuracy']:
            alerts.append(f"Low accuracy: {metrics.accuracy:.3f}")
        
        if metrics.latency_ms > self.alert_thresholds['latency_ms']:
            alerts.append(f"High latency: {metrics.latency_ms:.1f}ms")
        
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {metrics.error_rate:.3f}")
        
        if metrics.drift_score > self.alert_thresholds['drift_score']:
            alerts.append(f"Model drift detected: {metrics.drift_score:.3f}")
        
        if alerts:
            logger.warning(f"Performance alerts for {deployment_id}: {', '.join(alerts)}")
    
    async def get_metrics_history(self, deployment_id: str, hours: int = 24) -> List[PerformanceMetrics]:
        """Get metrics history for a deployment"""
        if deployment_id not in self.metrics_history:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            m for m in self.metrics_history[deployment_id]
            if m.timestamp > cutoff_time
        ]
    
    async def get_performance_summary(self, deployment_id: str) -> Dict[str, Any]:
        """Get performance summary for a deployment"""
        metrics = await self.get_metrics_history(deployment_id, hours=1)
        
        if not metrics:
            return {"status": "no_data"}
        
        latest = metrics[-1]
        avg_accuracy = np.mean([m.accuracy for m in metrics])
        avg_latency = np.mean([m.latency_ms for m in metrics])
        avg_error_rate = np.mean([m.error_rate for m in metrics])
        
        status = ModelPerformanceStatus.HEALTHY
        if latest.drift_score > self.alert_thresholds['drift_score']:
            status = ModelPerformanceStatus.DRIFT_DETECTED
        elif (latest.accuracy < self.alert_thresholds['accuracy'] or 
              latest.latency_ms > self.alert_thresholds['latency_ms']):
            status = ModelPerformanceStatus.DEGRADED
        elif latest.error_rate > self.alert_thresholds['error_rate']:
            status = ModelPerformanceStatus.CRITICAL
        
        return {
            "status": status.value,
            "latest_metrics": asdict(latest),
            "averages": {
                "accuracy": avg_accuracy,
                "latency_ms": avg_latency,
                "error_rate": avg_error_rate
            },
            "metrics_count": len(metrics),
            "monitoring_duration_hours": 1
        }


class DriftDetector:
    """Detects model drift and triggers retraining"""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.drift_thresholds = {
            'accuracy_drop': 0.1,  # 10% drop in accuracy
            'latency_increase': 2.0,  # 2x increase in latency
            'error_rate_increase': 0.05,  # 5% increase in error rate
            'drift_score': 0.3  # Direct drift score threshold
        }
        self.baseline_metrics = {}
        self.retraining_triggers = {}
    
    async def establish_baseline(self, deployment_id: str):
        """Establish baseline metrics for drift detection"""
        metrics = await self.performance_monitor.get_metrics_history(deployment_id, hours=24)
        
        if len(metrics) < 10:  # Need sufficient data
            logger.warning(f"Insufficient data to establish baseline for {deployment_id}")
            return
        
        # Calculate baseline statistics
        accuracies = [m.accuracy for m in metrics]
        latencies = [m.latency_ms for m in metrics]
        error_rates = [m.error_rate for m in metrics]
        
        self.baseline_metrics[deployment_id] = {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'latency_mean': np.mean(latencies),
            'latency_std': np.std(latencies),
            'error_rate_mean': np.mean(error_rates),
            'error_rate_std': np.std(error_rates),
            'established_at': datetime.utcnow()
        }
        
        logger.info(f"Established baseline metrics for {deployment_id}")
    
    async def detect_drift(self, deployment_id: str) -> Dict[str, Any]:
        """Detect drift in model performance"""
        if deployment_id not in self.baseline_metrics:
            await self.establish_baseline(deployment_id)
            return {"drift_detected": False, "reason": "baseline_not_established"}
        
        recent_metrics = await self.performance_monitor.get_metrics_history(deployment_id, hours=1)
        
        if not recent_metrics:
            return {"drift_detected": False, "reason": "no_recent_data"}
        
        baseline = self.baseline_metrics[deployment_id]
        latest = recent_metrics[-1]
        
        drift_indicators = []
        
        # Check accuracy drop
        accuracy_drop = baseline['accuracy_mean'] - latest.accuracy
        if accuracy_drop > self.drift_thresholds['accuracy_drop']:
            drift_indicators.append(f"accuracy_drop: {accuracy_drop:.3f}")
        
        # Check latency increase
        latency_ratio = latest.latency_ms / baseline['latency_mean']
        if latency_ratio > self.drift_thresholds['latency_increase']:
            drift_indicators.append(f"latency_increase: {latency_ratio:.2f}x")
        
        # Check error rate increase
        error_rate_increase = latest.error_rate - baseline['error_rate_mean']
        if error_rate_increase > self.drift_thresholds['error_rate_increase']:
            drift_indicators.append(f"error_rate_increase: {error_rate_increase:.3f}")
        
        # Check direct drift score
        if latest.drift_score > self.drift_thresholds['drift_score']:
            drift_indicators.append(f"drift_score: {latest.drift_score:.3f}")
        
        drift_detected = len(drift_indicators) > 0
        
        if drift_detected:
            logger.warning(f"Drift detected for {deployment_id}: {', '.join(drift_indicators)}")
        
        return {
            "drift_detected": drift_detected,
            "drift_indicators": drift_indicators,
            "latest_metrics": asdict(latest),
            "baseline_metrics": baseline,
            "detection_timestamp": datetime.utcnow().isoformat()
        }
    
    async def trigger_retraining(self, deployment_id: str, drift_info: Dict[str, Any]):
        """Trigger model retraining due to drift"""
        if deployment_id in self.retraining_triggers:
            last_trigger = self.retraining_triggers[deployment_id]
            if datetime.utcnow() - last_trigger < timedelta(hours=6):
                logger.info(f"Retraining recently triggered for {deployment_id}, skipping")
                return False
        
        self.retraining_triggers[deployment_id] = datetime.utcnow()
        
        # In a real implementation, would trigger actual retraining pipeline
        logger.info(f"Triggered retraining for {deployment_id} due to drift")
        
        return True


class GlobalModelDeploymentSystem:
    """Main system for global model deployment and management"""
    
    def __init__(self, storage_path: str = "model_storage"):
        # Initialize components
        self.storage = LocalModelStorage(storage_path)
        self.version_manager = ModelVersionManager(self.storage)
        self.orchestrator = DeploymentOrchestrator(self.version_manager)
        self.performance_monitor = PerformanceMonitor()
        self.drift_detector = DriftDetector(self.performance_monitor)
        
        # System state
        self.auto_deployment_enabled = True
        self.auto_retraining_enabled = True
        
        logger.info("Global Model Deployment System initialized")
    
    async def register_model(
        self,
        model: nn.Module,
        model_name: str,
        training_data_hash: str,
        performance_metrics: Dict[str, float],
        geological_domains: List[str],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Register a new model version"""
        model_version = await self.version_manager.create_version(
            model=model,
            model_name=model_name,
            training_data_hash=training_data_hash,
            performance_metrics=performance_metrics,
            geological_domains=geological_domains,
            metadata=metadata
        )
        
        logger.info(f"Registered model {model_name} version {model_version.version_number}")
        return model_version.version_id
    
    async def deploy_to_region(
        self,
        model_name: str,
        region_name: str,
        geological_domain: str,
        endpoint_url: str,
        deployment_type: str = "cloud",
        auto_update: bool = True
    ) -> str:
        """Deploy latest model version to a geological region"""
        # Get latest model version
        latest_version = await self.version_manager.get_latest_version(model_name)
        if not latest_version:
            raise ValueError(f"No versions found for model {model_name}")
        
        # Create deployment target
        target_id = f"{region_name}_{geological_domain}".replace(" ", "_").lower()
        target = DeploymentTarget(
            target_id=target_id,
            region_name=region_name,
            geological_domain=geological_domain,
            endpoint_url=endpoint_url,
            deployment_type=deployment_type,
            hardware_specs={},
            network_config={},
            security_config={},
            auto_update_enabled=auto_update,
            monitoring_enabled=True
        )
        
        # Register target
        self.orchestrator.register_target(target)
        
        # Deploy model
        deployment_id = await self.orchestrator.deploy_model(
            latest_version.version_id,
            target_id
        )
        
        # Start monitoring
        await self.performance_monitor.start_monitoring(deployment_id, target)
        
        logger.info(f"Deployed {model_name} to {region_name} ({geological_domain})")
        return deployment_id
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get comprehensive deployment status"""
        deployment_record = await self.orchestrator.get_deployment_status(deployment_id)
        if not deployment_record:
            return {"error": "Deployment not found"}
        
        performance_summary = await self.performance_monitor.get_performance_summary(deployment_id)
        drift_info = await self.drift_detector.detect_drift(deployment_id)
        
        return {
            "deployment_id": deployment_id,
            "status": deployment_record.status.value,
            "model_version": deployment_record.model_version.version_number,
            "target_region": deployment_record.target.region_name,
            "geological_domain": deployment_record.target.geological_domain,
            "deployment_timestamp": deployment_record.deployment_timestamp.isoformat(),
            "performance": performance_summary,
            "drift_detection": drift_info,
            "logs": deployment_record.deployment_logs[-5:]  # Last 5 log entries
        }
    
    async def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments with their status"""
        deployments = await self.orchestrator.list_deployments()
        
        deployment_list = []
        for deployment in deployments:
            deployment_info = {
                "deployment_id": deployment.deployment_id,
                "model_name": deployment.model_version.model_name,
                "model_version": deployment.model_version.version_number,
                "target_region": deployment.target.region_name,
                "geological_domain": deployment.target.geological_domain,
                "status": deployment.status.value,
                "deployment_timestamp": deployment.deployment_timestamp.isoformat()
            }
            deployment_list.append(deployment_info)
        
        return deployment_list
    
    async def update_deployment(self, deployment_id: str, new_version_id: str) -> bool:
        """Update a deployment to a new model version"""
        deployment_record = await self.orchestrator.get_deployment_status(deployment_id)
        if not deployment_record:
            return False
        
        # Set rollback version
        deployment_record.rollback_version = deployment_record.model_version.version_id
        
        # Deploy new version
        new_deployment_id = await self.orchestrator.deploy_model(
            new_version_id,
            deployment_record.target.target_id
        )
        
        logger.info(f"Updated deployment {deployment_id} to version {new_version_id}")
        return True
    
    async def get_global_deployment_overview(self) -> Dict[str, Any]:
        """Get overview of all global deployments"""
        deployments = await self.orchestrator.list_deployments()
        
        # Aggregate statistics
        total_deployments = len(deployments)
        active_deployments = len([d for d in deployments if d.status == DeploymentStatus.DEPLOYED])
        failed_deployments = len([d for d in deployments if d.status == DeploymentStatus.FAILED])
        
        # Group by geological domain
        domain_stats = {}
        for deployment in deployments:
            domain = deployment.target.geological_domain
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "active": 0, "failed": 0}
            
            domain_stats[domain]["total"] += 1
            if deployment.status == DeploymentStatus.DEPLOYED:
                domain_stats[domain]["active"] += 1
            elif deployment.status == DeploymentStatus.FAILED:
                domain_stats[domain]["failed"] += 1
        
        return {
            "total_deployments": total_deployments,
            "active_deployments": active_deployments,
            "failed_deployments": failed_deployments,
            "success_rate": active_deployments / total_deployments if total_deployments > 0 else 0,
            "domain_statistics": domain_stats,
            "auto_deployment_enabled": self.auto_deployment_enabled,
            "auto_retraining_enabled": self.auto_retraining_enabled,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def enable_auto_deployment(self, model_name: str, target_domains: List[str]):
        """Enable automatic deployment for new model versions"""
        # In a real implementation, would set up triggers for new versions
        logger.info(f"Enabled auto-deployment for {model_name} to domains: {target_domains}")
    
    async def cleanup_old_versions(self, model_name: str, keep_versions: int = 5):
        """Clean up old model versions, keeping only the most recent"""
        versions = await self.version_manager.list_versions(model_name)
        
        if len(versions) <= keep_versions:
            return
        
        versions_to_delete = versions[keep_versions:]
        
        for version in versions_to_delete:
            await self.version_manager.storage.delete_version(version.version_id)
        
        logger.info(f"Cleaned up {len(versions_to_delete)} old versions of {model_name}")
    
    async def shutdown(self):
        """Shutdown the deployment system"""
        # Stop all monitoring
        for deployment_id in list(self.performance_monitor.monitoring_active.keys()):
            await self.performance_monitor.stop_monitoring(deployment_id)
        
        logger.info("Global Model Deployment System shutdown complete")