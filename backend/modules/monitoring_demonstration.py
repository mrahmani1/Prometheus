"""
Comprehensive Logging and Monitoring System Demonstration
Advanced demonstration of monitoring, logging, and performance tracking capabilities
"""

import asyncio
import time
import json
import logging
import numpy as np
from typing import Dict, Any, List
import threading
from datetime import datetime, timedelta

# Import monitoring system
try:
    from .comprehensive_logging_monitoring import (
        ComprehensiveLoggingMonitoring,
        LoggingConfig,
        MonitoringConfig,
        setup_geological_monitoring,
        create_geological_logger
    )
    MONITORING_AVAILABLE = True
except ImportError:
    print("Comprehensive monitoring system not available")
    MONITORING_AVAILABLE = False

# Import parallel computing framework
try:
    from .parallel_computing import ParallelComputingFramework
    PARALLEL_COMPUTING_AVAILABLE = True
except ImportError:
    print("Parallel computing framework not available")
    PARALLEL_COMPUTING_AVAILABLE = False

class GeologicalSimulationEngine:
    """Simulated geological computation engine for demonstration"""
    
    def __init__(self, monitoring_system=None):
        self.monitoring_system = monitoring_system
        self.logger = create_geological_logger("simulation_engine") if monitoring_system else logging.getLogger(__name__)
        self.models = ["seismic_transformer", "volcanic_predictor", "tectonic_analyzer"]
        self.computation_history = []
    
    def simulate_seismic_analysis(self, data_size: int = 10000, complexity: str = "medium") -> Dict[str, Any]:
        """Simulate seismic data analysis with monitoring"""
        
        operation_name = "seismic_analysis"
        start_time = time.time()
        
        # Configure simulation parameters based on complexity
        complexity_params = {
            "low": {"iterations": 100, "sleep_time": 0.01},
            "medium": {"iterations": 500, "sleep_time": 0.02},
            "high": {"iterations": 1000, "sleep_time": 0.05}
        }
        
        params = complexity_params.get(complexity, complexity_params["medium"])
        
        try:
            if self.monitoring_system:
                with self.monitoring_system.trace_operation(
                    operation_name,
                    data_size=data_size,
                    complexity=complexity,
                    model="seismic_transformer"
                ) as correlation_id:
                    
                    self.monitoring_system.log("INFO", f"Starting seismic analysis", 
                                             data_size=data_size, complexity=complexity)
                    
                    # Simulate computation
                    result = self._perform_seismic_computation(data_size, params)
                    
                    # Record metrics
                    self.monitoring_system.record_metric("seismic_events_detected", result["events_detected"])
                    self.monitoring_system.record_metric("model_confidence", result["confidence"])
                    
                    duration = time.time() - start_time
                    self.monitoring_system.log_performance(operation_name, duration, 
                                                         data_size=data_size, complexity=complexity)
                    
                    result["correlation_id"] = correlation_id
            else:
                # Fallback without monitoring
                self.logger.info(f"Starting seismic analysis (data_size={data_size}, complexity={complexity})")
                result = self._perform_seismic_computation(data_size, params)
                duration = time.time() - start_time
                self.logger.info(f"Seismic analysis completed in {duration:.3f}s")
            
            result["duration"] = duration
            result["success"] = True
            self.computation_history.append(result)
            
            return result
            
        except Exception as e:
            if self.monitoring_system:
                self.monitoring_system.log_error("Seismic analysis failed", e, 
                                                operation=operation_name, data_size=data_size)
            else:
                self.logger.error(f"Seismic analysis failed: {e}")
            
            return {"success": False, "error": str(e), "duration": time.time() - start_time}
    
    def _perform_seismic_computation(self, data_size: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform actual seismic computation simulation"""
        
        # Generate synthetic seismic data
        seismic_data = np.random.normal(0, 1, (data_size, 3))  # 3D seismic data
        
        # Simulate processing iterations
        detected_events = 0
        for i in range(params["iterations"]):
            # Simulate computation load
            time.sleep(params["sleep_time"])
            
            # Simulate event detection
            if np.random.random() > 0.95:  # 5% chance of event detection
                detected_events += 1
        
        # Calculate synthetic results
        signal_strength = np.mean(np.abs(seismic_data))
        noise_level = np.std(seismic_data)
        confidence = max(0.0, min(1.0, 0.8 + np.random.normal(0, 0.1)))
        
        return {
            "events_detected": detected_events,
            "signal_strength": float(signal_strength),
            "noise_level": float(noise_level),
            "confidence": confidence,
            "data_points_processed": data_size,
            "processing_iterations": params["iterations"]
        }
    
    def simulate_volcanic_prediction(self, forecast_days: int = 30) -> Dict[str, Any]:
        """Simulate volcanic eruption prediction"""
        
        operation_name = "volcanic_prediction"
        start_time = time.time()
        
        try:
            if self.monitoring_system:
                with self.monitoring_system.trace_operation(
                    operation_name,
                    forecast_days=forecast_days,
                    model="volcanic_predictor"
                ):
                    result = self._perform_volcanic_computation(forecast_days)
                    
                    # Record volcanic-specific metrics
                    self.monitoring_system.record_metric("eruption_probability", result["eruption_probability"])
                    self.monitoring_system.record_metric("alert_level", result["alert_level"])
                    
                    duration = time.time() - start_time
                    self.monitoring_system.log_performance(operation_name, duration, 
                                                         forecast_days=forecast_days)
            else:
                result = self._perform_volcanic_computation(forecast_days)
                duration = time.time() - start_time
                self.logger.info(f"Volcanic prediction completed in {duration:.3f}s")
            
            result["duration"] = duration
            result["success"] = True
            
            return result
            
        except Exception as e:
            if self.monitoring_system:
                self.monitoring_system.log_error("Volcanic prediction failed", e,
                                                operation=operation_name, forecast_days=forecast_days)
            else:
                self.logger.error(f"Volcanic prediction failed: {e}")
            
            return {"success": False, "error": str(e), "duration": time.time() - start_time}
    
    def _perform_volcanic_computation(self, forecast_days: int) -> Dict[str, Any]:
        """Perform volcanic prediction computation"""
        
        # Simulate computation time based on forecast period
        base_time = 0.5
        complexity_factor = forecast_days / 30.0
        computation_time = base_time * complexity_factor
        time.sleep(computation_time)
        
        # Generate synthetic volcanic prediction data
        eruption_probability = max(0.0, min(1.0, np.random.beta(2, 8)))  # Skewed toward low probability
        alert_level = int(np.clip(eruption_probability * 5, 0, 4))  # 0-4 alert scale
        
        # Simulate various volcanic indicators
        gas_emissions = np.random.exponential(1.0)
        ground_deformation = np.random.normal(0, 0.1)
        seismic_activity = np.random.gamma(2, 0.5)
        temperature_anomaly = np.random.normal(0, 2.0)
        
        return {
            "eruption_probability": eruption_probability,
            "alert_level": alert_level,
            "forecast_days": forecast_days,
            "indicators": {
                "gas_emissions": gas_emissions,
                "ground_deformation": ground_deformation,
                "seismic_activity": seismic_activity,
                "temperature_anomaly": temperature_anomaly
            },
            "confidence": max(0.6, min(0.99, 0.8 + np.random.normal(0, 0.05)))
        }
    
    def simulate_tectonic_analysis(self, region: str = "pacific_rim") -> Dict[str, Any]:
        """Simulate tectonic plate analysis"""
        
        operation_name = "tectonic_analysis"
        start_time = time.time()
        
        try:
            if self.monitoring_system:
                with self.monitoring_system.trace_operation(
                    operation_name,
                    region=region,
                    model="tectonic_analyzer"
                ):
                    result = self._perform_tectonic_computation(region)
                    
                    # Record tectonic metrics
                    self.monitoring_system.record_metric("plate_stress_level", result["stress_level"])
                    self.monitoring_system.record_metric("fault_activity", result["fault_activity"])
                    
                    duration = time.time() - start_time
                    self.monitoring_system.log_performance(operation_name, duration, region=region)
            else:
                result = self._perform_tectonic_computation(region)
                duration = time.time() - start_time
                self.logger.info(f"Tectonic analysis completed in {duration:.3f}s")
            
            result["duration"] = duration
            result["success"] = True
            
            return result
            
        except Exception as e:
            if self.monitoring_system:
                self.monitoring_system.log_error("Tectonic analysis failed", e,
                                                operation=operation_name, region=region)
            else:
                self.logger.error(f"Tectonic analysis failed: {e}")
            
            return {"success": False, "error": str(e), "duration": time.time() - start_time}
    
    def _perform_tectonic_computation(self, region: str) -> Dict[str, Any]:
        """Perform tectonic analysis computation"""
        
        # Simulate region-specific computation complexity
        region_complexity = {
            "pacific_rim": 0.8,
            "mediterranean": 0.6,
            "mid_atlantic": 0.4,
            "himalayan": 0.9
        }
        
        complexity = region_complexity.get(region, 0.5)
        computation_time = 0.3 + complexity * 0.7
        time.sleep(computation_time)
        
        # Generate synthetic tectonic data
        stress_level = np.random.gamma(2, complexity)
        fault_activity = np.random.exponential(complexity)
        plate_velocity = np.random.normal(5.0, 2.0)  # cm/year
        earthquake_risk = max(0.0, min(1.0, complexity + np.random.normal(0, 0.1)))
        
        return {
            "region": region,
            "stress_level": stress_level,
            "fault_activity": fault_activity,
            "plate_velocity": plate_velocity,
            "earthquake_risk": earthquake_risk,
            "analysis_complexity": complexity,
            "confidence": max(0.7, min(0.98, 0.85 + np.random.normal(0, 0.05)))
        }
    
    def get_computation_statistics(self) -> Dict[str, Any]:
        """Get computation statistics"""
        if not self.computation_history:
            return {"total_computations": 0}
        
        successful_computations = [c for c in self.computation_history if c.get("success", False)]
        total_time = sum(c.get("duration", 0) for c in self.computation_history)
        avg_time = total_time / len(self.computation_history) if self.computation_history else 0
        
        return {
            "total_computations": len(self.computation_history),
            "successful_computations": len(successful_computations),
            "success_rate": len(successful_computations) / len(self.computation_history),
            "total_processing_time": total_time,
            "average_processing_time": avg_time,
            "last_computation": self.computation_history[-1] if self.computation_history else None
        }

async def run_comprehensive_monitoring_demo():
    """Run comprehensive demonstration of monitoring system"""
    
    if not MONITORING_AVAILABLE:
        print("Monitoring system not available. Skipping demonstration.")
        return
    
    print("=== Comprehensive Logging and Monitoring System Demonstration ===\n")
    
    # Initialize monitoring system with custom configuration
    logging_config = LoggingConfig(
        log_level="INFO",
        log_format="json",
        enable_structured_logging=True,
        enable_performance_logging=True,
        performance_threshold_ms=1000.0,
        enable_async_logging=True
    )
    
    monitoring_config = MonitoringConfig(
        enable_prometheus=True,
        enable_alerting=True,
        metrics_prefix="prometheus_demo",
        cpu_threshold=80.0,
        memory_threshold=85.0,
        performance_collection_interval=2.0,
        enable_health_checks=True,
        health_check_interval=10.0
    )
    
    print("Initializing comprehensive monitoring system...")
    monitoring_system = ComprehensiveLoggingMonitoring(logging_config, monitoring_config)
    
    # Start performance monitoring
    monitoring_system.start_performance_monitoring()
    
    print("✓ Monitoring system initialized")
    print("✓ Performance monitoring started")
    print("✓ Health checks enabled")
    
    # Initialize geological simulation engine
    print("\nInitializing geological simulation engine...")
    geo_engine = GeologicalSimulationEngine(monitoring_system)
    print("✓ Geological simulation engine ready")
    
    # Demonstrate various monitoring capabilities
    print("\n=== Monitoring Capabilities Demonstration ===")
    
    # 1. Basic logging and tracing
    print("\n1. Basic Logging and Distributed Tracing")
    with monitoring_system.trace_operation("demo_initialization", demo_phase="basic_logging"):
        monitoring_system.log("INFO", "Starting basic logging demonstration", component="demo")
        time.sleep(0.5)  # Simulate some work
        monitoring_system.log("DEBUG", "Basic logging demonstration completed", component="demo")
    
    # 2. Performance monitoring
    print("\n2. Performance Monitoring")
    with monitoring_system.time_operation("performance_demo", {"operation_type": "demo"}):
        # Simulate CPU-intensive work
        data = np.random.random((5000, 5000))
        result = np.linalg.svd(data, compute_uv=False)
        monitoring_system.record_metric("matrix_size", data.size)
        monitoring_system.record_metric("singular_values_computed", len(result))
    
    # 3. Error handling and alerting
    print("\n3. Error Handling and Monitoring")
    try:
        with monitoring_system.trace_operation("error_demo", error_simulation=True):
            # Simulate an error condition
            monitoring_system.log("WARNING", "Simulating error condition for demonstration")
            raise ValueError("Simulated error for monitoring demonstration")
    except Exception as e:
        monitoring_system.log_error("Demonstration error handled successfully", e, component="demo")
    
    # 4. Geological computations with monitoring
    print("\n4. Geological Computations with Comprehensive Monitoring")
    
    # Run multiple geological simulations
    simulation_tasks = [
        ("seismic_analysis", {"data_size": 15000, "complexity": "high"}),
        ("volcanic_prediction", {"forecast_days": 45}),
        ("tectonic_analysis", {"region": "pacific_rim"}),
        ("seismic_analysis", {"data_size": 8000, "complexity": "medium"}),
        ("volcanic_prediction", {"forecast_days": 14})
    ]
    
    results = []
    for task_name, params in simulation_tasks:
        print(f"  Running {task_name}...")
        
        if task_name == "seismic_analysis":
            result = geo_engine.simulate_seismic_analysis(**params)
        elif task_name == "volcanic_prediction":
            result = geo_engine.simulate_volcanic_prediction(**params)
        elif task_name == "tectonic_analysis":
            result = geo_engine.simulate_tectonic_analysis(**params)
        
        results.append(result)
        
        # Brief pause between tasks
        await asyncio.sleep(0.2)
    
    # 5. System health and metrics collection
    print("\n5. System Health and Metrics Collection")
    
    # Wait a moment for metrics to be collected
    await asyncio.sleep(3)
    
    # Get comprehensive system status
    system_status = monitoring_system.get_system_status()
    health_status = system_status.get('health', {})
    
    print(f"  Overall Health Status: {health_status.get('overall_status', 'unknown')}")
    print(f"  Active Health Checks: {len(health_status.get('checks', {}))}")
    
    # Display health check results
    for check_name, check_result in health_status.get('checks', {}).items():
        status = check_result.get('status', 'unknown')
        print(f"    {check_name}: {status}")
    
    # 6. Custom metrics and performance statistics
    print("\n6. Performance Statistics and Custom Metrics")
    
    computation_stats = geo_engine.get_computation_statistics()
    print(f"  Total Computations: {computation_stats['total_computations']}")
    print(f"  Success Rate: {computation_stats['success_rate']:.2%}")
    print(f"  Average Processing Time: {computation_stats['average_processing_time']:.3f}s")
    
    # Record summary metrics
    monitoring_system.record_metric("demo_total_computations", computation_stats['total_computations'])
    monitoring_system.record_metric("demo_success_rate", computation_stats['success_rate'])
    
    # 7. Demonstrate parallel processing integration
    if PARALLEL_COMPUTING_AVAILABLE:
        print("\n7. Parallel Processing Integration")
        
        parallel_config = {
            'enable_prometheus': True,
            'enable_alerting': True,
            'log_level': 'INFO'
        }
        
        parallel_framework = ParallelComputingFramework(parallel_config)
        
        # Run parallel geological computation with monitoring
        with parallel_framework.trace_geological_computation("parallel_demo", "ensemble_model"):
            # Simulate parallel seismic processing
            data = np.random.random((2000, 2000))
            parallel_result = np.mean(data ** 2)
            
            parallel_framework.record_geological_metric("parallel_data_processed", data.size)
            parallel_framework.record_geological_metric("parallel_result_accuracy", 0.94)
        
        # Get parallel framework health
        parallel_health = parallel_framework.get_health_status()
        print(f"  Parallel Framework Health: {parallel_health.get('overall_status', 'unknown')}")
    
    # 8. Final system statistics
    print("\n=== Final System Statistics ===")
    
    final_status = monitoring_system.get_system_status()
    final_metrics = final_status.get('metrics', {})
    
    print(f"Performance History Size: {final_metrics.get('performance_history_size', 0)}")
    print(f"Active Alerts: {final_metrics.get('active_alerts', 0)}")
    print(f"Custom Metrics Tracked: {final_metrics.get('custom_metrics_count', 0)}")
    
    configuration = final_status.get('configuration', {})
    print(f"Monitoring Enabled: {configuration.get('monitoring_enabled', False)}")
    print(f"Alerting Enabled: {configuration.get('alerting_enabled', False)}")
    print(f"Tracing Enabled: {configuration.get('tracing_enabled', False)}")
    
    # 9. Demonstrate metrics endpoints (for production integration)
    print("\n9. Production Integration Endpoints")
    
    # Get Prometheus metrics endpoint handler
    metrics_handler = monitoring_system.get_metrics_endpoint()
    health_handler = monitoring_system.get_health_endpoint()
    
    print("  ✓ Prometheus metrics endpoint available")
    print("  ✓ Health check endpoint available")
    print("  ✓ JSON structured logging enabled")
    print("  ✓ Distributed tracing configured")
    print("  ✓ Alert system operational")
    
    # Cleanup
    print("\n=== Monitoring Demonstration Completed Successfully ===")
    monitoring_system.shutdown()
    
    return {
        "geological_computations": len(results),
        "successful_operations": len([r for r in results if r.get('success', False)]),
        "system_health": health_status.get('overall_status', 'unknown'),
        "monitoring_features_demonstrated": [
            "structured_logging",
            "distributed_tracing", 
            "performance_monitoring",
            "error_handling",
            "health_checks",
            "custom_metrics",
            "alerting_system",
            "prometheus_integration"
        ]
    }

def run_basic_monitoring_demo():
    """Run basic monitoring demonstration (synchronous version)"""
    
    print("=== Basic Monitoring System Demonstration ===\n")
    
    if not MONITORING_AVAILABLE:
        print("Comprehensive monitoring system not available.")
        print("Running basic demonstration with standard logging...")
        
        # Basic logging demonstration
        logger = logging.getLogger("basic_demo")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        logger.info("Basic logging demonstration started")
        
        # Simulate geological computation
        geo_engine = GeologicalSimulationEngine()
        result = geo_engine.simulate_seismic_analysis(data_size=5000, complexity="low")
        
        if result["success"]:
            logger.info(f"Seismic analysis completed: {result['events_detected']} events detected")
        else:
            logger.error(f"Seismic analysis failed: {result.get('error', 'Unknown error')}")
        
        logger.info("Basic demonstration completed")
        return result
    
    # Use setup utility for quick initialization
    monitoring_system = setup_geological_monitoring(
        log_level="INFO",
        enable_prometheus=True,
        enable_alerting=True
    )
    
    print("✓ Geological monitoring system initialized")
    
    # Create geological logger
    geo_logger = create_geological_logger("basic_demo")
    
    # Initialize simulation engine
    geo_engine = GeologicalSimulationEngine(monitoring_system)
    
    # Run basic geological simulations
    print("\nRunning geological simulations with monitoring...")
    
    # Seismic analysis
    seismic_result = geo_engine.simulate_seismic_analysis(data_size=8000, complexity="medium")
    if seismic_result["success"]:
        print(f"  ✓ Seismic Analysis: {seismic_result['events_detected']} events detected")
    
    # Volcanic prediction  
    volcanic_result = geo_engine.simulate_volcanic_prediction(forecast_days=21)
    if volcanic_result["success"]:
        print(f"  ✓ Volcanic Prediction: {volcanic_result['eruption_probability']:.1%} probability")
    
    # Tectonic analysis
    tectonic_result = geo_engine.simulate_tectonic_analysis(region="mediterranean")
    if tectonic_result["success"]:
        print(f"  ✓ Tectonic Analysis: Stress level {tectonic_result['stress_level']:.2f}")
    
    # Get final statistics
    stats = geo_engine.get_computation_statistics()
    print(f"\nComputation Statistics:")
    print(f"  Total: {stats['total_computations']}")
    print(f"  Success Rate: {stats['success_rate']:.2%}")
    print(f"  Average Time: {stats['average_processing_time']:.3f}s")
    
    # Get system health
    health = monitoring_system.get_system_status()
    print(f"  System Health: {health['health']['overall_status']}")
    
    print("\n=== Basic Monitoring Demonstration Completed ===")
    
    return {
        "seismic_result": seismic_result,
        "volcanic_result": volcanic_result, 
        "tectonic_result": tectonic_result,
        "statistics": stats,
        "health_status": health['health']['overall_status']
    }

if __name__ == "__main__":
    # Check available components
    print("=== Monitoring System Component Availability ===")
    print(f"Comprehensive Monitoring: {MONITORING_AVAILABLE}")
    print(f"Parallel Computing: {PARALLEL_COMPUTING_AVAILABLE}")
    
    import sys
    
    # Run appropriate demonstration based on command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "basic":
        # Run basic demo
        result = run_basic_monitoring_demo()
    else:
        # Run comprehensive demo (default)
        if MONITORING_AVAILABLE:
            result = asyncio.run(run_comprehensive_monitoring_demo())
        else:
            result = run_basic_monitoring_demo()
    
    print(f"\nDemonstration completed successfully!")
    print(f"Result summary: {json.dumps(result, indent=2, default=str)}")