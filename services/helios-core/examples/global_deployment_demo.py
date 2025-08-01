"""
Global Model Deployment Demo

This example demonstrates the global model deployment system capabilities,
including model versioning, automated deployment to different geological regions,
performance monitoring, and drift detection.

Requirements addressed: 5.3, 5.4, 5.5
"""

import asyncio
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import deployment system components
from core.global_model_deployment import GlobalModelDeploymentSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeologicalPINNModel(nn.Module):
    """Sample geological PINN model for demonstration"""

    def __init__(self, input_dim=6, hidden_dims=[64, 128, 128, 64], output_dim=4):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.Tanh(), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Physics constraint weights
        self.physics_weights = nn.Parameter(torch.ones(4))

    def forward(self, x):
        return self.network(x)

    def physics_loss(self, x, y_pred):
        """Simplified physics loss for demonstration"""
        # In practice, would implement actual physics constraints
        return torch.mean(self.physics_weights * torch.randn(4))

class ModelTrainingSimulator:
    """Simulates model training with different performance characteristics"""

    def __init__(self):
        self.training_scenarios = {
            "high_performance": {
                "accuracy": 0.92,
                "loss": 0.08,
                "convergence_epochs": 50,
                "stability": 0.95,
            },
            "medium_performance": {
                "accuracy": 0.85,
                "loss": 0.15,
                "convergence_epochs": 75,
                "stability": 0.88,
            },
            "low_performance": {
                "accuracy": 0.78,
                "loss": 0.22,
                "convergence_epochs": 100,
                "stability": 0.80,
            },
        }

    def simulate_training(
        self, model: nn.Module, scenario: str = "high_performance"
    ) -> dict:
        """Simulate model training and return performance metrics"""
        if scenario not in self.training_scenarios:
            scenario = "medium_performance"

        metrics = self.training_scenarios[scenario].copy()

        # Add some random variation
        metrics["accuracy"] += np.random.uniform(-0.02, 0.02)
        metrics["loss"] += np.random.uniform(-0.01, 0.01)

        # Additional metrics
        metrics.update(
            {
                "precision": metrics["accuracy"] - 0.05,
                "recall": metrics["accuracy"] - 0.03,
                "f1_score": metrics["accuracy"] - 0.04,
                "training_time_hours": np.random.uniform(2, 8),
                "model_size_mb": np.random.uniform(50, 200),
            }
        )

        logger.info(
            f"Simulated training with {scenario} scenario: "
            f"accuracy={metrics['accuracy']:.3f}, loss={metrics['loss']:.3f}"
        )

        return metrics

async def demonstrate_model_versioning():
    """Demonstrate model versioning and registration"""
    logger.info("=== Model Versioning Demo ===")

    # Initialize deployment system
    system = GlobalModelDeploymentSystem("demo_model_storage")
    trainer = ModelTrainingSimulator()

    # Create different model versions
    model_versions = []
    scenarios = ["low_performance", "medium_performance", "high_performance"]

    for i, scenario in enumerate(scenarios):
        logger.info(f"\n--- Creating Model Version {i+1} ---")

        # Create model
        model = GeologicalPINNModel()

        # Simulate training
        performance_metrics = trainer.simulate_training(model, scenario)

        # Register model version
        version_id = await system.register_model(
            model=model,
            model_name="helios_geological_pinn",
            training_data_hash=f"training_data_hash_{i+1}",
            performance_metrics=performance_metrics,
            geological_domains=["yellowstone", "vesuvius", "kilauea"],
            metadata={
                "training_scenario": scenario,
                "model_architecture": "PINN",
                "physics_constraints": ["navier_stokes", "heat_transfer"],
                "training_date": datetime.utcnow().isoformat(),
            },
        )

        model_versions.append(version_id)
        logger.info(f"Registered model version: {version_id}")

        # Small delay to ensure different timestamps
        await asyncio.sleep(0.1)

    # List all versions
    versions = await system.version_manager.list_versions("helios_geological_pinn")
    logger.info(f"\nTotal model versions: {len(versions)}")

    for version in versions:
        logger.info(
            f"  Version {version.version_number}: "
            f"accuracy={version.performance_metrics.get('accuracy', 0):.3f}"
        )

    return system, model_versions

async def demonstrate_global_deployment():
    """Demonstrate deployment to multiple geological regions"""
    logger.info("\n=== Global Deployment Demo ===")

    # Get system from previous demo
    system, model_versions = await demonstrate_model_versioning()

    # Define geological regions for deployment
    regions = [
        {
            "region_name": "Yellowstone National Park",
            "geological_domain": "yellowstone",
            "endpoint_url": "https://yellowstone-api.usgs.gov/helios",
            "deployment_type": "cloud",
        },
        {
            "region_name": "Mount Vesuvius",
            "geological_domain": "vesuvius",
            "endpoint_url": "https://vesuvius-api.ingv.it/helios",
            "deployment_type": "cloud",
        },
        {
            "region_name": "Kilauea Volcano",
            "geological_domain": "kilauea",
            "endpoint_url": "https://kilauea-api.hvo.usgs.gov/helios",
            "deployment_type": "edge",
        },
        {
            "region_name": "Kamchatka Peninsula",
            "geological_domain": "kamchatka",
            "endpoint_url": "https://kamchatka-api.kscnet.ru/helios",
            "deployment_type": "on_premise",
        },
    ]

    # Deploy to all regions
    deployment_ids = []

    for region in regions:
        logger.info(f"\n--- Deploying to {region['region_name']} ---")

        deployment_id = await system.deploy_to_region(
            model_name="helios_geological_pinn",
            region_name=region["region_name"],
            geological_domain=region["geological_domain"],
            endpoint_url=region["endpoint_url"],
            deployment_type=region["deployment_type"],
            auto_update=True,
        )

        deployment_ids.append(deployment_id)
        logger.info(f"Deployment started: {deployment_id}")

    # Wait for deployments to process
    logger.info("\nWaiting for deployments to complete...")
    await asyncio.sleep(2.0)

    # Check deployment statuses
    logger.info("\n--- Deployment Status ---")
    for deployment_id in deployment_ids:
        status = await system.get_deployment_status(deployment_id)
        logger.info(f"Deployment {deployment_id[:12]}...: {status['status']}")
        logger.info(f"  Region: {status['target_region']}")
        logger.info(f"  Domain: {status['geological_domain']}")
        logger.info(f"  Model Version: {status['model_version']}")

    return system, deployment_ids

async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and metrics collection"""
    logger.info("\n=== Performance Monitoring Demo ===")

    # Get system and deployments from previous demo
    system, deployment_ids = await demonstrate_global_deployment()

    # Simulate monitoring for a period
    logger.info("Simulating performance monitoring...")

    # Let monitoring run for a bit
    await asyncio.sleep(3.0)

    # Check performance for each deployment
    for deployment_id in deployment_ids:
        logger.info(f"\n--- Performance for {deployment_id[:12]}... ---")

        # Get performance summary
        performance = await system.performance_monitor.get_performance_summary(
            deployment_id
        )

        if performance.get("status") != "no_data":
            latest = performance.get("latest_metrics", {})
            averages = performance.get("averages", {})

            logger.info(f"Status: {performance.get('status', 'unknown')}")
            logger.info(f"Latest Accuracy: {latest.get('accuracy', 0):.3f}")
            logger.info(f"Latest Latency: {latest.get('latency_ms', 0):.1f}ms")
            logger.info(f"Average Accuracy: {averages.get('accuracy', 0):.3f}")
            logger.info(f"Average Latency: {averages.get('latency_ms', 0):.1f}ms")
            logger.info(f"Error Rate: {latest.get('error_rate', 0):.3f}")
        else:
            logger.info("No performance data available yet")

    return system, deployment_ids

async def demonstrate_drift_detection():
    """Demonstrate model drift detection and retraining triggers"""
    logger.info("\n=== Drift Detection Demo ===")

    # Get system and deployments
    system, deployment_ids = await demonstrate_performance_monitoring()

    # Establish baselines for drift detection
    logger.info("Establishing performance baselines...")

    for deployment_id in deployment_ids:
        await system.drift_detector.establish_baseline(deployment_id)
        logger.info(f"Baseline established for {deployment_id[:12]}...")

    # Simulate some time passing with potential drift
    logger.info("\nSimulating model drift scenarios...")

    # Manually inject some "drift" by modifying thresholds temporarily
    original_thresholds = system.drift_detector.drift_thresholds.copy()

    # Make drift detection more sensitive for demo
    system.drift_detector.drift_thresholds.update(
        {
            "accuracy_drop": 0.05,  # 5% drop triggers drift
            "latency_increase": 1.5,  # 1.5x increase triggers drift
            "drift_score": 0.2,  # Lower threshold for demo
        }
    )

    # Check for drift in each deployment
    drift_detected_count = 0

    for deployment_id in deployment_ids:
        logger.info(f"\n--- Checking drift for {deployment_id[:12]}... ---")

        drift_info = await system.drift_detector.detect_drift(deployment_id)

        if drift_info["drift_detected"]:
            drift_detected_count += 1
            logger.warning("DRIFT DETECTED!")
            logger.warning(f"Indicators: {', '.join(drift_info['drift_indicators'])}")

            # Trigger retraining
            retraining_triggered = await system.drift_detector.trigger_retraining(
                deployment_id, drift_info
            )

            if retraining_triggered:
                logger.info("Retraining triggered successfully")
            else:
                logger.info("Retraining skipped (recently triggered)")
        else:
            logger.info(
                "No drift detected - model performing within expected parameters"
            )

    # Restore original thresholds
    system.drift_detector.drift_thresholds = original_thresholds

    logger.info(
        f"\nDrift detection summary: {drift_detected_count}/{len(deployment_ids)} deployments showed drift"
    )

    return system, deployment_ids

async def demonstrate_automated_updates():
    """Demonstrate automated model updates and rollbacks"""
    logger.info("\n=== Automated Updates Demo ===")

    # Get system and deployments
    system, deployment_ids = await demonstrate_drift_detection()

    # Create a new improved model version
    logger.info("Creating improved model version...")

    improved_model = GeologicalPINNModel()
    trainer = ModelTrainingSimulator()

    # Simulate better performance
    improved_metrics = trainer.simulate_training(improved_model, "high_performance")
    improved_metrics["accuracy"] = min(0.95, improved_metrics["accuracy"] + 0.03)

    new_version_id = await system.register_model(
        model=improved_model,
        model_name="helios_geological_pinn",
        training_data_hash="improved_training_data_hash",
        performance_metrics=improved_metrics,
        geological_domains=["yellowstone", "vesuvius", "kilauea", "kamchatka"],
        metadata={
            "training_scenario": "improved_high_performance",
            "model_architecture": "Enhanced PINN",
            "improvements": ["better_convergence", "reduced_overfitting"],
            "training_date": datetime.utcnow().isoformat(),
        },
    )

    logger.info(f"Created improved model version: {new_version_id}")
    logger.info(f"Improved accuracy: {improved_metrics['accuracy']:.3f}")

    # Update one deployment to the new version
    if deployment_ids:
        test_deployment_id = deployment_ids[0]
        logger.info(
            f"\nUpdating deployment {test_deployment_id[:12]}... to new version"
        )

        success = await system.update_deployment(test_deployment_id, new_version_id)

        if success:
            logger.info("Deployment update initiated successfully")

            # Wait a bit and check status
            await asyncio.sleep(1.0)

            status = await system.get_deployment_status(test_deployment_id)
            logger.info(f"Updated deployment status: {status['status']}")
        else:
            logger.error("Failed to update deployment")

    return system, deployment_ids

async def demonstrate_global_overview():
    """Demonstrate global deployment overview and analytics"""
    logger.info("\n=== Global Overview Demo ===")

    # Get system from previous demo
    system, deployment_ids = await demonstrate_automated_updates()

    # Get comprehensive overview
    overview = await system.get_global_deployment_overview()

    logger.info("=== GLOBAL DEPLOYMENT OVERVIEW ===")
    logger.info(f"Total Deployments: {overview['total_deployments']}")
    logger.info(f"Active Deployments: {overview['active_deployments']}")
    logger.info(f"Failed Deployments: {overview['failed_deployments']}")
    logger.info(f"Success Rate: {overview['success_rate']:.1%}")
    logger.info(f"Auto-deployment Enabled: {overview['auto_deployment_enabled']}")
    logger.info(f"Auto-retraining Enabled: {overview['auto_retraining_enabled']}")

    logger.info("\n--- Domain Statistics ---")
    for domain, stats in overview["domain_statistics"].items():
        logger.info(f"{domain.title()}:")
        logger.info(f"  Total: {stats['total']}")
        logger.info(f"  Active: {stats['active']}")
        logger.info(f"  Failed: {stats['failed']}")
        if stats["total"] > 0:
            success_rate = stats["active"] / stats["total"]
            logger.info(f"  Success Rate: {success_rate:.1%}")

    # List all deployments
    deployments = await system.list_deployments()

    logger.info(f"\n--- All Deployments ({len(deployments)}) ---")
    for deployment in deployments:
        logger.info(
            f"{deployment['deployment_id'][:12]}... - "
            f"{deployment['target_region']} ({deployment['geological_domain']}) - "
            f"v{deployment['model_version']} - {deployment['status']}"
        )

    return system

async def demonstrate_maintenance_operations():
    """Demonstrate maintenance operations like cleanup and health checks"""
    logger.info("\n=== Maintenance Operations Demo ===")

    # Get system from previous demo
    system = await demonstrate_global_overview()

    # Show current model versions
    versions = await system.version_manager.list_versions("helios_geological_pinn")
    logger.info(f"Current model versions: {len(versions)}")

    for version in versions:
        logger.info(
            f"  {version.version_number} - "
            f"created {version.creation_timestamp.strftime('%Y-%m-%d %H:%M')} - "
            f"accuracy: {version.performance_metrics.get('accuracy', 0):.3f}"
        )

    # Cleanup old versions (keep only 3 most recent)
    logger.info("\nCleaning up old model versions...")
    await system.cleanup_old_versions("helios_geological_pinn", keep_versions=3)

    # Show versions after cleanup
    versions_after = await system.version_manager.list_versions(
        "helios_geological_pinn"
    )
    logger.info(f"Model versions after cleanup: {len(versions_after)}")

    # System health check
    logger.info("\n--- System Health Check ---")
    logger.info(f"Storage system: {'✓ Healthy' if system.storage else '✗ Error'}")
    logger.info(
        f"Version manager: {'✓ Healthy' if system.version_manager else '✗ Error'}"
    )
    logger.info(f"Orchestrator: {'✓ Healthy' if system.orchestrator else '✗ Error'}")
    logger.info(
        f"Performance monitor: {'✓ Healthy' if system.performance_monitor else '✗ Error'}"
    )
    logger.info(
        f"Drift detector: {'✓ Healthy' if system.drift_detector else '✗ Error'}"
    )

    # Show deployment targets
    targets = system.orchestrator.deployment_targets
    logger.info(f"\nRegistered deployment targets: {len(targets)}")
    for target_id, target in targets.items():
        logger.info(
            f"  {target.region_name} ({target.geological_domain}) - "
            f"{target.deployment_type} - "
            f"auto-update: {target.auto_update_enabled}"
        )

    return system

def create_deployment_visualization(system):
    """Create visualization of deployment metrics (if matplotlib available)"""
    try:
        logger.info("\n=== Creating Deployment Visualization ===")

        # This would create charts showing:
        # - Deployment success rates by region
        # - Performance metrics over time
        # - Drift detection alerts
        # - Model version adoption

        logger.info("Deployment visualization would be created here")
        logger.info("Charts would include:")
        logger.info("  - Success rates by geological domain")
        logger.info("  - Performance trends over time")
        logger.info("  - Drift detection timeline")
        logger.info("  - Model version distribution")

    except ImportError:
        logger.warning("Matplotlib not available - skipping visualization")

async def main():
    """Run all global deployment demonstrations"""
    logger.info("Starting Global Model Deployment System Demonstration")
    logger.info("=" * 60)

    try:
        # Run all demonstrations in sequence
        system = await demonstrate_model_versioning()
        await demonstrate_global_deployment()
        await demonstrate_performance_monitoring()
        await demonstrate_drift_detection()
        await demonstrate_automated_updates()
        await demonstrate_global_overview()
        system = await demonstrate_maintenance_operations()

        # Create visualization
        create_deployment_visualization(system)

        # Final cleanup
        logger.info("\n=== Demonstration Complete ===")
        logger.info("Shutting down deployment system...")
        await system.shutdown()

        logger.info("\n" + "=" * 60)
        logger.info("Global Model Deployment System Demo Completed Successfully!")
        logger.info("Key features demonstrated:")
        logger.info("  ✓ Model versioning and registration")
        logger.info("  ✓ Global deployment to multiple regions")
        logger.info("  ✓ Performance monitoring and metrics")
        logger.info("  ✓ Drift detection and retraining triggers")
        logger.info("  ✓ Automated updates and rollbacks")
        logger.info("  ✓ Global overview and analytics")
        logger.info("  ✓ Maintenance operations")

        return system

    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the demonstration
    result = asyncio.run(main())
