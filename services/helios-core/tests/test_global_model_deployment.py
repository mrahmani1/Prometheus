"""
Tests for Global Model Deployment System

This module contains comprehensive tests for the global model deployment system,
including model versioning, deployment orchestration, performance monitoring,
and drift detection.

Requirements addressed: 5.3, 5.4, 5.5
"""

import asyncio
import shutil
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import torch.nn as nn
import torch.nn.functional as F
from core.global_model_deployment import (
    DeploymentOrchestrator,
    DeploymentStatus,
    DeploymentTarget,
    DriftDetector,
    GlobalModelDeploymentSystem,
    LocalModelStorage,
    ModelVersion,
    ModelVersionManager,
    PerformanceMetrics,
    PerformanceMonitor,
)

class SimpleTestModel(nn.Module):
    """Simple neural network for testing"""

    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)

@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for model storage"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_model():
    """Create a sample model for testing"""
    return SimpleTestModel()

@pytest.fixture
def sample_model_version():
    """Create a sample model version"""
    return ModelVersion(
        version_id="test_model_20240101_001",
        model_name="test_model",
        version_number="1.0.0",
        creation_timestamp=datetime.utcnow(),
        model_hash="abc123",
        model_size_bytes=1024,
        training_data_hash="def456",
        performance_metrics={"accuracy": 0.85, "loss": 0.15},
        geological_domains=["yellowstone", "vesuvius"],
        dependencies=["torch", "numpy"],
        metadata={"description": "Test model"},
        deployment_config={},
    )

@pytest.fixture
def sample_deployment_target():
    """Create a sample deployment target"""
    return DeploymentTarget(
        target_id="test_target",
        region_name="Test Region",
        geological_domain="test_domain",
        endpoint_url="http://test.example.com",
        deployment_type="cloud",
        hardware_specs={"cpu": "4 cores", "memory": "8GB"},
        network_config={"bandwidth": "1Gbps"},
        security_config={"encryption": "TLS"},
        auto_update_enabled=True,
        monitoring_enabled=True,
    )

class TestLocalModelStorage:
    """Test local model storage functionality"""

    def test_storage_initialization(self, temp_storage_dir):
        """Test storage initialization"""
        storage = LocalModelStorage(temp_storage_dir)

        assert storage.storage_path.exists()
        assert storage.metadata_file.exists()
        assert isinstance(storage.metadata, dict)

    @pytest.mark.asyncio
    async def test_store_and_retrieve_model(
        self, temp_storage_dir, sample_model_version
    ):
        """Test storing and retrieving a model"""
        storage = LocalModelStorage(temp_storage_dir)

        # Create dummy model data
        model_data = b"dummy model data"

        # Store model
        storage_path = await storage.store_model(sample_model_version, model_data)
        assert storage_path is not None

        # Retrieve model
        retrieved_version, retrieved_data = await storage.retrieve_model(
            sample_model_version.version_id
        )

        assert retrieved_version.version_id == sample_model_version.version_id
        assert retrieved_version.model_name == sample_model_version.model_name
        assert retrieved_data == model_data

    @pytest.mark.asyncio
    async def test_list_versions(self, temp_storage_dir, sample_model_version):
        """Test listing model versions"""
        storage = LocalModelStorage(temp_storage_dir)

        # Initially no versions
        versions = await storage.list_versions("test_model")
        assert len(versions) == 0

        # Store a model
        model_data = b"dummy model data"
        await storage.store_model(sample_model_version, model_data)

        # Now should have one version
        versions = await storage.list_versions("test_model")
        assert len(versions) == 1
        assert versions[0].version_id == sample_model_version.version_id

    @pytest.mark.asyncio
    async def test_delete_version(self, temp_storage_dir, sample_model_version):
        """Test deleting a model version"""
        storage = LocalModelStorage(temp_storage_dir)

        # Store a model
        model_data = b"dummy model data"
        await storage.store_model(sample_model_version, model_data)

        # Verify it exists
        versions = await storage.list_versions("test_model")
        assert len(versions) == 1

        # Delete it
        success = await storage.delete_version(sample_model_version.version_id)
        assert success

        # Verify it's gone
        versions = await storage.list_versions("test_model")
        assert len(versions) == 0

class TestModelVersionManager:
    """Test model version management"""

    @pytest.mark.asyncio
    async def test_create_version(self, temp_storage_dir, sample_model):
        """Test creating a new model version"""
        storage = LocalModelStorage(temp_storage_dir)
        version_manager = ModelVersionManager(storage)

        model_version = await version_manager.create_version(
            model=sample_model,
            model_name="test_model",
            training_data_hash="training123",
            performance_metrics={"accuracy": 0.9},
            geological_domains=["test_domain"],
            metadata={"test": "metadata"},
        )

        assert model_version.model_name == "test_model"
        assert model_version.training_data_hash == "training123"
        assert model_version.performance_metrics["accuracy"] == 0.9
        assert "test_domain" in model_version.geological_domains
        assert model_version.version_number == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_latest_version(self, temp_storage_dir, sample_model):
        """Test getting the latest model version"""
        storage = LocalModelStorage(temp_storage_dir)
        version_manager = ModelVersionManager(storage)

        # No versions initially
        latest = await version_manager.get_latest_version("test_model")
        assert latest is None

        # Create a version
        version1 = await version_manager.create_version(
            model=sample_model,
            model_name="test_model",
            training_data_hash="training123",
            performance_metrics={"accuracy": 0.9},
            geological_domains=["test_domain"],
        )

        # Should be the latest
        latest = await version_manager.get_latest_version("test_model")
        assert latest.version_id == version1.version_id

        # Create another version
        await asyncio.sleep(0.01)  # Ensure different timestamp
        version2 = await version_manager.create_version(
            model=sample_model,
            model_name="test_model",
            training_data_hash="training456",
            performance_metrics={"accuracy": 0.95},
            geological_domains=["test_domain"],
        )

        # Should now be the latest
        latest = await version_manager.get_latest_version("test_model")
        assert latest.version_id == version2.version_id

    def test_calculate_model_hash(self, temp_storage_dir, sample_model):
        """Test model hash calculation"""
        storage = LocalModelStorage(temp_storage_dir)
        version_manager = ModelVersionManager(storage)

        hash1 = version_manager._calculate_model_hash(sample_model)
        hash2 = version_manager._calculate_model_hash(sample_model)

        # Same model should produce same hash
        assert hash1 == hash2

        # Different model should produce different hash
        different_model = SimpleTestModel(input_size=20)
        hash3 = version_manager._calculate_model_hash(different_model)
        assert hash1 != hash3

class TestDeploymentOrchestrator:
    """Test deployment orchestration"""

    @pytest.mark.asyncio
    async def test_register_target(self, temp_storage_dir, sample_deployment_target):
        """Test registering a deployment target"""
        storage = LocalModelStorage(temp_storage_dir)
        version_manager = ModelVersionManager(storage)
        orchestrator = DeploymentOrchestrator(version_manager)

        orchestrator.register_target(sample_deployment_target)

        assert sample_deployment_target.target_id in orchestrator.deployment_targets
        assert (
            orchestrator.deployment_targets[sample_deployment_target.target_id]
            == sample_deployment_target
        )

    @pytest.mark.asyncio
    async def test_deploy_model(
        self, temp_storage_dir, sample_model, sample_deployment_target
    ):
        """Test model deployment"""
        storage = LocalModelStorage(temp_storage_dir)
        version_manager = ModelVersionManager(storage)
        orchestrator = DeploymentOrchestrator(version_manager)

        # Create a model version
        model_version = await version_manager.create_version(
            model=sample_model,
            model_name="test_model",
            training_data_hash="training123",
            performance_metrics={"accuracy": 0.9},
            geological_domains=["test_domain"],
        )

        # Register target
        orchestrator.register_target(sample_deployment_target)

        # Deploy model
        deployment_id = await orchestrator.deploy_model(
            model_version.version_id, sample_deployment_target.target_id
        )

        assert deployment_id is not None
        assert deployment_id in orchestrator.deployment_records

        # Check deployment record
        record = orchestrator.deployment_records[deployment_id]
        assert record.model_version.version_id == model_version.version_id
        assert record.target.target_id == sample_deployment_target.target_id
        assert record.status == DeploymentStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_deployment_status(
        self, temp_storage_dir, sample_model, sample_deployment_target
    ):
        """Test getting deployment status"""
        storage = LocalModelStorage(temp_storage_dir)
        version_manager = ModelVersionManager(storage)
        orchestrator = DeploymentOrchestrator(version_manager)

        # Create and deploy model
        model_version = await version_manager.create_version(
            model=sample_model,
            model_name="test_model",
            training_data_hash="training123",
            performance_metrics={"accuracy": 0.9},
            geological_domains=["test_domain"],
        )

        orchestrator.register_target(sample_deployment_target)
        deployment_id = await orchestrator.deploy_model(
            model_version.version_id, sample_deployment_target.target_id
        )

        # Get status
        status = await orchestrator.get_deployment_status(deployment_id)
        assert status is not None
        assert status.deployment_id == deployment_id

    @pytest.mark.asyncio
    async def test_list_deployments(
        self, temp_storage_dir, sample_model, sample_deployment_target
    ):
        """Test listing deployments"""
        storage = LocalModelStorage(temp_storage_dir)
        version_manager = ModelVersionManager(storage)
        orchestrator = DeploymentOrchestrator(version_manager)

        # Initially no deployments
        deployments = await orchestrator.list_deployments()
        assert len(deployments) == 0

        # Create and deploy model
        model_version = await version_manager.create_version(
            model=sample_model,
            model_name="test_model",
            training_data_hash="training123",
            performance_metrics={"accuracy": 0.9},
            geological_domains=["test_domain"],
        )

        orchestrator.register_target(sample_deployment_target)
        await orchestrator.deploy_model(
            model_version.version_id, sample_deployment_target.target_id
        )

        # Should have one deployment
        deployments = await orchestrator.list_deployments()
        assert len(deployments) == 1

class TestPerformanceMonitor:
    """Test performance monitoring"""

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, sample_deployment_target):
        """Test starting and stopping monitoring"""
        monitor = PerformanceMonitor()

        deployment_id = "test_deployment"

        # Start monitoring
        await monitor.start_monitoring(deployment_id, sample_deployment_target)
        assert deployment_id in monitor.monitoring_active
        assert monitor.monitoring_active[deployment_id] is True

        # Stop monitoring
        await monitor.stop_monitoring(deployment_id)
        assert monitor.monitoring_active[deployment_id] is False

    @pytest.mark.asyncio
    async def test_collect_metrics(self, sample_deployment_target):
        """Test metrics collection"""
        monitor = PerformanceMonitor()

        deployment_id = "test_deployment"
        metrics = await monitor._collect_metrics(
            deployment_id, sample_deployment_target
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.target_id == sample_deployment_target.target_id
        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.latency_ms >= 0
        assert metrics.throughput_rps >= 0

    @pytest.mark.asyncio
    async def test_get_metrics_history(self, sample_deployment_target):
        """Test getting metrics history"""
        monitor = PerformanceMonitor()

        deployment_id = "test_deployment"

        # Initially no history
        history = await monitor.get_metrics_history(deployment_id)
        assert len(history) == 0

        # Add some metrics manually
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            target_id=sample_deployment_target.target_id,
            model_version="1.0.0",
            accuracy=0.85,
            latency_ms=200,
            throughput_rps=25,
            memory_usage_mb=750,
            cpu_usage_percent=50,
            error_rate=0.02,
            drift_score=0.1,
            geological_consistency=0.9,
        )

        monitor.metrics_history[deployment_id] = [metrics]

        # Should have history now
        history = await monitor.get_metrics_history(deployment_id)
        assert len(history) == 1
        assert history[0].accuracy == 0.85

    @pytest.mark.asyncio
    async def test_performance_summary(self, sample_deployment_target):
        """Test performance summary generation"""
        monitor = PerformanceMonitor()

        deployment_id = "test_deployment"

        # Add some metrics
        metrics = [
            PerformanceMetrics(
                timestamp=datetime.utcnow() - timedelta(minutes=i),
                target_id=sample_deployment_target.target_id,
                model_version="1.0.0",
                accuracy=0.85 + i * 0.01,
                latency_ms=200 + i * 10,
                throughput_rps=25,
                memory_usage_mb=750,
                cpu_usage_percent=50,
                error_rate=0.02,
                drift_score=0.1,
                geological_consistency=0.9,
            )
            for i in range(5)
        ]

        monitor.metrics_history[deployment_id] = metrics

        summary = await monitor.get_performance_summary(deployment_id)

        assert "status" in summary
        assert "latest_metrics" in summary
        assert "averages" in summary
        assert summary["metrics_count"] == 5

class TestDriftDetector:
    """Test drift detection"""

    @pytest.mark.asyncio
    async def test_establish_baseline(self, sample_deployment_target):
        """Test establishing performance baseline"""
        monitor = PerformanceMonitor()
        drift_detector = DriftDetector(monitor)

        deployment_id = "test_deployment"

        # Add metrics for baseline
        metrics = [
            PerformanceMetrics(
                timestamp=datetime.utcnow() - timedelta(hours=i),
                target_id=sample_deployment_target.target_id,
                model_version="1.0.0",
                accuracy=0.85,
                latency_ms=200,
                throughput_rps=25,
                memory_usage_mb=750,
                cpu_usage_percent=50,
                error_rate=0.02,
                drift_score=0.1,
                geological_consistency=0.9,
            )
            for i in range(15)  # 15 metrics for sufficient baseline
        ]

        monitor.metrics_history[deployment_id] = metrics

        await drift_detector.establish_baseline(deployment_id)

        assert deployment_id in drift_detector.baseline_metrics
        baseline = drift_detector.baseline_metrics[deployment_id]
        assert "accuracy_mean" in baseline
        assert "latency_mean" in baseline
        assert "error_rate_mean" in baseline

    @pytest.mark.asyncio
    async def test_detect_drift(self, sample_deployment_target):
        """Test drift detection"""
        monitor = PerformanceMonitor()
        drift_detector = DriftDetector(monitor)

        deployment_id = "test_deployment"

        # Establish baseline with good metrics
        baseline_metrics = [
            PerformanceMetrics(
                timestamp=datetime.utcnow() - timedelta(hours=i),
                target_id=sample_deployment_target.target_id,
                model_version="1.0.0",
                accuracy=0.85,
                latency_ms=200,
                throughput_rps=25,
                memory_usage_mb=750,
                cpu_usage_percent=50,
                error_rate=0.02,
                drift_score=0.1,
                geological_consistency=0.9,
            )
            for i in range(15)
        ]

        monitor.metrics_history[deployment_id] = baseline_metrics
        await drift_detector.establish_baseline(deployment_id)

        # Add recent metrics with drift
        recent_metrics = [
            PerformanceMetrics(
                timestamp=datetime.utcnow(),
                target_id=sample_deployment_target.target_id,
                model_version="1.0.0",
                accuracy=0.70,  # Significant drop
                latency_ms=500,  # Significant increase
                throughput_rps=25,
                memory_usage_mb=750,
                cpu_usage_percent=50,
                error_rate=0.08,  # Significant increase
                drift_score=0.4,  # High drift score
                geological_consistency=0.9,
            )
        ]

        monitor.metrics_history[deployment_id].extend(recent_metrics)

        drift_info = await drift_detector.detect_drift(deployment_id)

        assert drift_info["drift_detected"] is True
        assert len(drift_info["drift_indicators"]) > 0

    @pytest.mark.asyncio
    async def test_trigger_retraining(self, sample_deployment_target):
        """Test triggering retraining"""
        monitor = PerformanceMonitor()
        drift_detector = DriftDetector(monitor)

        deployment_id = "test_deployment"
        drift_info = {"drift_detected": True, "drift_indicators": ["accuracy_drop"]}

        # First trigger should succeed
        success = await drift_detector.trigger_retraining(deployment_id, drift_info)
        assert success is True

        # Second trigger within 6 hours should be skipped
        success = await drift_detector.trigger_retraining(deployment_id, drift_info)
        assert success is False

class TestGlobalModelDeploymentSystem:
    """Test the complete global model deployment system"""

    @pytest.mark.asyncio
    async def test_system_initialization(self, temp_storage_dir):
        """Test system initialization"""
        system = GlobalModelDeploymentSystem(temp_storage_dir)

        assert system.storage is not None
        assert system.version_manager is not None
        assert system.orchestrator is not None
        assert system.performance_monitor is not None
        assert system.drift_detector is not None

    @pytest.mark.asyncio
    async def test_register_model(self, temp_storage_dir, sample_model):
        """Test model registration"""
        system = GlobalModelDeploymentSystem(temp_storage_dir)

        version_id = await system.register_model(
            model=sample_model,
            model_name="test_model",
            training_data_hash="training123",
            performance_metrics={"accuracy": 0.9},
            geological_domains=["test_domain"],
            metadata={"description": "Test model"},
        )

        assert version_id is not None

        # Verify model was registered
        latest_version = await system.version_manager.get_latest_version("test_model")
        assert latest_version is not None
        assert latest_version.version_id == version_id

    @pytest.mark.asyncio
    async def test_deploy_to_region(self, temp_storage_dir, sample_model):
        """Test deploying model to a region"""
        system = GlobalModelDeploymentSystem(temp_storage_dir)

        # Register model first
        await system.register_model(
            model=sample_model,
            model_name="test_model",
            training_data_hash="training123",
            performance_metrics={"accuracy": 0.9},
            geological_domains=["test_domain"],
        )

        # Deploy to region
        deployment_id = await system.deploy_to_region(
            model_name="test_model",
            region_name="Test Region",
            geological_domain="test_domain",
            endpoint_url="http://test.example.com",
        )

        assert deployment_id is not None

        # Verify deployment
        status = await system.get_deployment_status(deployment_id)
        assert "error" not in status
        assert status["model_version"] is not None

    @pytest.mark.asyncio
    async def test_list_deployments(self, temp_storage_dir, sample_model):
        """Test listing deployments"""
        system = GlobalModelDeploymentSystem(temp_storage_dir)

        # Initially no deployments
        deployments = await system.list_deployments()
        assert len(deployments) == 0

        # Register and deploy model
        await system.register_model(
            model=sample_model,
            model_name="test_model",
            training_data_hash="training123",
            performance_metrics={"accuracy": 0.9},
            geological_domains=["test_domain"],
        )

        await system.deploy_to_region(
            model_name="test_model",
            region_name="Test Region",
            geological_domain="test_domain",
            endpoint_url="http://test.example.com",
        )

        # Should have one deployment
        deployments = await system.list_deployments()
        assert len(deployments) == 1
        assert deployments[0]["model_name"] == "test_model"

    @pytest.mark.asyncio
    async def test_global_deployment_overview(self, temp_storage_dir, sample_model):
        """Test getting global deployment overview"""
        system = GlobalModelDeploymentSystem(temp_storage_dir)

        # Register and deploy model
        await system.register_model(
            model=sample_model,
            model_name="test_model",
            training_data_hash="training123",
            performance_metrics={"accuracy": 0.9},
            geological_domains=["test_domain"],
        )

        await system.deploy_to_region(
            model_name="test_model",
            region_name="Test Region",
            geological_domain="test_domain",
            endpoint_url="http://test.example.com",
        )

        overview = await system.get_global_deployment_overview()

        assert "total_deployments" in overview
        assert "active_deployments" in overview
        assert "domain_statistics" in overview
        assert overview["total_deployments"] >= 1

    @pytest.mark.asyncio
    async def test_cleanup_old_versions(self, temp_storage_dir, sample_model):
        """Test cleaning up old model versions"""
        system = GlobalModelDeploymentSystem(temp_storage_dir)

        # Register multiple versions
        for i in range(10):
            await asyncio.sleep(0.01)  # Ensure different timestamps
            await system.register_model(
                model=sample_model,
                model_name="test_model",
                training_data_hash=f"training{i}",
                performance_metrics={"accuracy": 0.9 + i * 0.01},
                geological_domains=["test_domain"],
            )

        # Should have 10 versions
        versions = await system.version_manager.list_versions("test_model")
        assert len(versions) == 10

        # Cleanup, keeping only 3
        await system.cleanup_old_versions("test_model", keep_versions=3)

        # Should have 3 versions left
        versions = await system.version_manager.list_versions("test_model")
        assert len(versions) == 3

# Integration tests
class TestDeploymentIntegration:
    """Integration tests for the complete deployment workflow"""

    @pytest.mark.asyncio
    async def test_end_to_end_deployment_workflow(self, temp_storage_dir, sample_model):
        """Test complete deployment workflow from registration to monitoring"""
        system = GlobalModelDeploymentSystem(temp_storage_dir)

        # 1. Register model
        version_id = await system.register_model(
            model=sample_model,
            model_name="integration_test_model",
            training_data_hash="training123",
            performance_metrics={"accuracy": 0.9, "loss": 0.1},
            geological_domains=["yellowstone", "vesuvius"],
            metadata={"description": "Integration test model"},
        )

        # 2. Deploy to multiple regions
        deployment_ids = []
        regions = [
            ("Yellowstone", "yellowstone", "http://yellowstone.example.com"),
            ("Vesuvius", "vesuvius", "http://vesuvius.example.com"),
        ]

        for region_name, domain, endpoint in regions:
            deployment_id = await system.deploy_to_region(
                model_name="integration_test_model",
                region_name=region_name,
                geological_domain=domain,
                endpoint_url=endpoint,
            )
            deployment_ids.append(deployment_id)

        # 3. Wait for deployments to process
        await asyncio.sleep(0.5)

        # 4. Check deployment statuses
        for deployment_id in deployment_ids:
            status = await system.get_deployment_status(deployment_id)
            assert "error" not in status
            assert status["deployment_id"] == deployment_id

        # 5. Get global overview
        overview = await system.get_global_deployment_overview()
        assert overview["total_deployments"] >= 2
        assert len(overview["domain_statistics"]) >= 2

        # 6. List all deployments
        deployments = await system.list_deployments()
        assert len(deployments) >= 2

        # Verify all deployments are for our model
        for deployment in deployments:
            assert deployment["model_name"] == "integration_test_model"

if __name__ == "__main__":
    pytest.main([__file__])
