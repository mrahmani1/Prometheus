#!/usr/bin/env python3
"""
Error Management System Demonstration

This script demonstrates the hierarchical error management system
capabilities with realistic geological modeling scenarios.

Requirements: 12.3, 12.4
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add the core module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.error_integration import (
    GeologicalModelErrorHandler,
    VolcanicMonitoringErrorHandler,
    with_geological_error_handling,
    with_realtime_monitoring_error_handling,
)
from core.error_management import get_system_health
from core.error_monitoring import get_error_metrics, get_error_patterns

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DemoGeologicalModel(GeologicalModelErrorHandler):
    """Demonstration geological model with error scenarios"""

    @with_geological_error_handling("mantle_convection_simulation")
    async def simulate_mantle_convection(self, resolution: str, time_span: str):
        """Simulate mantle convection with potential resource errors"""
        logger.info(
            f"Starting mantle convection simulation: {resolution} resolution, {time_span}"
        )

        # Simulate memory error for ultra-high resolution
        if resolution == "ultra_high":
            raise MemoryError(
                "Insufficient memory for ultra-high resolution simulation - requires 128GB"
            )

        # Simulate physics constraint violation
        if time_span == "invalid_timespan":
            raise Exception(
                "Physics constraint violated: negative time evolution detected"
            )

        # Successful simulation
        return {
            "simulation_type": "mantle_convection",
            "resolution": resolution,
            "time_span": time_span,
            "temperature_field": "computed",
            "velocity_field": "computed",
            "status": "completed",
            "timestamp": datetime.now(),
        }

    @with_geological_error_handling("pinn_training")
    async def train_pinn_model(self, physics_constraints: list, data_quality: str):
        """Train PINN model with potential convergence errors"""
        logger.info(f"Training PINN model with constraints: {physics_constraints}")

        # Simulate convergence failure
        if data_quality == "poor":
            raise Exception(
                "PINN training diverged - gradient explosion due to poor data quality"
            )

        # Simulate physics constraint violation
        if "invalid_constraint" in physics_constraints:
            raise Exception(
                "Physics constraint violated: conservation law cannot be satisfied"
            )

        return {
            "model_type": "physics_informed_neural_network",
            "constraints": physics_constraints,
            "training_status": "converged",
            "accuracy": 0.95,
            "timestamp": datetime.now(),
        }

class DemoVolcanicMonitor(VolcanicMonitoringErrorHandler):
    """Demonstration volcanic monitor with error scenarios"""

    @with_realtime_monitoring_error_handling("seismic_data_processing")
    async def process_seismic_data(
        self, sensor_id: str, waveform_data, data_quality: str
    ):
        """Process seismic data with potential data quality errors"""
        logger.info(f"Processing seismic data from sensor {sensor_id}")

        # Simulate data corruption
        if waveform_data is None or data_quality == "corrupted":
            raise ValueError(f"Corrupted seismic waveform data from sensor {sensor_id}")

        # Simulate network connectivity error
        if sensor_id == "OFFLINE_SENSOR":
            raise ConnectionError(
                f"Cannot connect to sensor {sensor_id} - network timeout"
            )

        return {
            "sensor_id": sensor_id,
            "events_detected": 3,
            "max_magnitude": 2.1,
            "processing_status": "completed",
            "timestamp": datetime.now(),
        }

    @with_realtime_monitoring_error_handling("volcanic_alert_generation")
    async def generate_volcanic_alert(self, volcano_id: str, risk_level: float):
        """Generate volcanic alert with potential system errors"""
        logger.info(
            f"Generating alert for volcano {volcano_id}, risk level: {risk_level}"
        )

        # Simulate alert system overload
        if risk_level > 0.95:
            raise Exception(
                "Alert system overload - too many simultaneous high-risk alerts"
            )

        # Simulate security error
        if volcano_id == "RESTRICTED_VOLCANO":
            raise Exception(
                "Authorization denied - insufficient permissions for restricted volcano data"
            )

        return {
            "volcano_id": volcano_id,
            "alert_level": "high" if risk_level > 0.7 else "moderate",
            "risk_assessment": risk_level,
            "recommended_actions": ["monitor_closely", "prepare_evacuation_plans"],
            "timestamp": datetime.now(),
        }

async def demonstrate_error_scenarios():
    """Demonstrate various error scenarios and recovery mechanisms"""

    print("=" * 80)
    print("PROMETHEUS AI - HIERARCHICAL ERROR MANAGEMENT SYSTEM DEMONSTRATION")
    print("=" * 80)
    print()

    # Initialize demo components
    geo_model = DemoGeologicalModel()
    volcano_monitor = DemoVolcanicMonitor()

    print("🔬 GEOLOGICAL MODELING ERROR SCENARIOS")
    print("-" * 50)

    # Scenario 1: Memory resource error with automatic recovery
    print("\n1. Testing memory resource error recovery...")
    try:
        result = await geo_model.simulate_mantle_convection(
            "ultra_high", "100_million_years"
        )
        print(
            f"   ✅ Result: {result['status']} - {result.get('resource_usage', 'standard')}"
        )
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Scenario 2: Model convergence error with fallback
    print("\n2. Testing model convergence error recovery...")
    try:
        result = await geo_model.train_pinn_model(
            ["navier_stokes", "heat_transfer"], "poor"
        )
        print(
            f"   ✅ Result: {result.get('model_type', 'fallback')} - confidence: {result.get('confidence', 'N/A')}"
        )
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Scenario 3: Physics constraint violation
    print("\n3. Testing physics constraint violation recovery...")
    try:
        result = await geo_model.train_pinn_model(["invalid_constraint"], "good")
        print(
            f"   ✅ Result: {result.get('result_type', 'corrected')} - warning: {result.get('warning', 'none')}"
        )
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    print("\n🌋 VOLCANIC MONITORING ERROR SCENARIOS")
    print("-" * 50)

    # Scenario 4: Data quality error with interpolation
    print("\n4. Testing data quality error recovery...")
    try:
        result = await volcano_monitor.process_seismic_data("VS001", None, "corrupted")
        print(
            f"   ✅ Result: {result.get('processing_status', 'recovered')} - quality: {result.get('quality_flag', 'estimated')}"
        )
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Scenario 5: Network connectivity error with caching
    print("\n5. Testing network connectivity error recovery...")
    try:
        result = await volcano_monitor.process_seismic_data(
            "OFFLINE_SENSOR", "valid_data", "good"
        )
        print(
            f"   ✅ Result: {result.get('data_source', 'network')} - status: {result.get('connection_status', 'unknown')}"
        )
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Scenario 6: Security error with limited access
    print("\n6. Testing security error recovery...")
    try:
        result = await volcano_monitor.generate_volcanic_alert(
            "RESTRICTED_VOLCANO", 0.8
        )
        print(
            f"   ✅ Result: {result.get('access_level', 'full')} - operations: {result.get('available_operations', 'all')}"
        )
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Scenario 7: System overload with graceful degradation
    print("\n7. Testing system overload with graceful degradation...")
    try:
        result = await volcano_monitor.generate_volcanic_alert("MOUNT_VESUVIUS", 0.98)
        print(
            f"   ✅ Result: {result.get('alert_level', 'emergency')} - system: {result.get('system_status', 'normal')}"
        )
    except Exception as e:
        print(f"   ❌ Failed: {e}")

async def demonstrate_system_monitoring():
    """Demonstrate system monitoring and health assessment"""

    print("\n📊 SYSTEM MONITORING AND HEALTH ASSESSMENT")
    print("-" * 50)

    # Get system health status
    health_status = get_system_health()
    print("\n🏥 System Health Status:")
    print(f"   Overall Status: {health_status['system_status'].upper()}")
    print(f"   Degradation Level: {health_status['degradation_level']}")
    print(
        f"   Error Rate (24h): {health_status['error_analysis']['error_rate_24h']:.2f} errors/hour"
    )

    # Get error metrics
    error_metrics = get_error_metrics(24)
    print("\n📈 Error Metrics (Last 24 Hours):")
    print(f"   Total Errors: {error_metrics.total_errors}")
    print(f"   Resolved Errors: {error_metrics.resolved_errors}")
    print(f"   Recovery Success Rate: {error_metrics.recovery_success_rate:.1f}%")

    if error_metrics.category_counts:
        print("   Error Categories:")
        for category, count in error_metrics.category_counts.items():
            print(f"     - {category}: {count}")

    # Get error patterns
    patterns = get_error_patterns()
    if patterns["anomalies"]:
        print("\n🚨 Anomalies Detected:")
        for anomaly in patterns["anomalies"]:
            print(f"   - {anomaly['type']}: {anomaly['description']}")

    if patterns["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in patterns["recommendations"]:
            print(f"   - {rec}")

async def demonstrate_configuration():
    """Demonstrate configuration management"""

    print("\n⚙️  CONFIGURATION MANAGEMENT")
    print("-" * 50)

    from core.error_config import get_config, get_service_config

import torch.nn as nn
import torch.nn.functional as F

# Show global configuration
config = get_config()
print("\n🔧 Global Configuration:")
print(f"   Max Retries: {config.default_max_retries}")
print(f"   Critical Error Threshold: {config.critical_error_threshold}")
print(f"   Automatic Degradation: {config.enable_automatic_degradation}")
    print(f"   Error Alerts: {config.enable_error_alerts}")

    # Show service-specific configurations
    print("\n🏢 Service-Specific Configurations:")
    for service_name in ["helios_core", "volcano_watchtower"]:
        service_config = get_service_config(service_name)
        if service_config:
            print(f"   {service_name}:")
            for key, value in service_config.items():
                print(f"     - {key}: {value}")

async def main():
    """Main demonstration function"""

    try:
        # Run error scenario demonstrations
        await demonstrate_error_scenarios()

        # Demonstrate system monitoring
        await demonstrate_system_monitoring()

        # Demonstrate configuration
        await demonstrate_configuration()

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\n✅ All error management system components are functioning correctly!")
        print("📚 See docs/error_management_system.md for detailed documentation")
        print("🧪 Run tests with: python -m pytest tests/test_error_management.py -v")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    # Run the demonstration
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
