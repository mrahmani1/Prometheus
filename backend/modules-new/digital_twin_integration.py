"""
Digital Twin Earth Integration System
Integrates Digital Twin Earth with existing Prometheus AI infrastructure
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import uuid
from pathlib import Path

# Import existing Prometheus modules
try:
    from .digital_twin_earth import (
        DigitalTwinEarth, EarthSystemState, ModelConfiguration, 
        ObservationData, PredictiveScenario
    )
    from .stream_processing import (
        KafkaStreamManager, FlinkStreamProcessor, RealTimeAggregator,
        EventDrivenArchitecture, StreamEvent, StreamEventType
    )
    from .database_integration import PostgreSQLGISManager, DatabaseConfig
    from .optimization_framework import ModelOptimizer, OptimizationConfig
    from .federated_learning import FederatedLearningOrchestrator, FederatedConfig
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    warnings.warn(f"Prometheus modules not fully available: {e}")

# Additional ML and scientific computing imports
try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("ML libraries not available. Some features will be limited.")

try:
    import xarray as xr
    import netCDF4
    SCIENTIFIC_AVAILABLE = True
except ImportError:
    SCIENTIFIC_AVAILABLE = False
    warnings.warn("Scientific computing libraries not available.")

warnings.filterwarnings('ignore')

class IntegrationType(Enum):
    STREAM_PROCESSING = "stream_processing"
    DATABASE_STORAGE = "database_storage"
    ML_OPTIMIZATION = "ml_optimization"
    FEDERATED_LEARNING = "federated_learning"
    REAL_TIME_MONITORING = "real_time_monitoring"

class DataFlowDirection(Enum):
    EARTH_TO_PROMETHEUS = "earth_to_prometheus"
    PROMETHEUS_TO_EARTH = "prometheus_to_earth"
    BIDIRECTIONAL = "bidirectional"

@dataclass
class IntegrationConfig:
    """Configuration for Digital Twin Earth integration"""
    stream_processing_enabled: bool = True
    database_storage_enabled: bool = True
    ml_optimization_enabled: bool = True
    federated_learning_enabled: bool = True
    real_time_sync_interval: float = 1.0  # seconds
    data_validation_enabled: bool = True
    performance_monitoring_enabled: bool = True
    max_concurrent_operations: int = 10
    
@dataclass
class IntegrationMetrics:
    """Metrics for integration performance"""
    sync_latency_ms: float
    data_throughput_mbps: float
    error_rate_percent: float
    active_connections: int
    last_sync_timestamp: datetime
    integration_health: str  # HEALTHY, DEGRADED, CRITICAL

class DigitalTwinEarthIntegrator:
    """Main integration system for Digital Twin Earth"""
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.digital_twin = None
        self.stream_manager = None
        self.database_manager = None
        self.optimizer = None
        self.federated_orchestrator = None
        
        # Integration state
        self.is_initialized = False
        self.active_integrations = set()
        self.integration_metrics = IntegrationMetrics(
            sync_latency_ms=0.0,
            data_throughput_mbps=0.0,
            error_rate_percent=0.0,
            active_connections=0,
            last_sync_timestamp=datetime.utcnow(),
            integration_health="INITIALIZING"
        )
        
        # Data flow management
        self.data_queues = {}
        self.sync_tasks = {}
        self.performance_monitor = None
        
    async def initialize(self, components_config: Dict[str, Any] = None):
        """Initialize all integration components"""
        try:
            components_config = components_config or {}
            
            # Initialize Digital Twin Earth
            earth_config = components_config.get('digital_twin', {})
            model_config = ModelConfiguration(**earth_config)
            self.digital_twin = DigitalTwinEarth(model_config)
            await self.digital_twin.initialize()
            
            # Initialize stream processing if enabled
            if self.config.stream_processing_enabled:
                kafka_config = components_config.get('kafka', {})
                self.stream_manager = KafkaStreamManager(kafka_config)
                self.active_integrations.add(IntegrationType.STREAM_PROCESSING)
            
            # Initialize database if enabled
            if self.config.database_storage_enabled:
                db_config_dict = components_config.get('database', {})
                db_config = DatabaseConfig(**db_config_dict)
                self.database_manager = PostgreSQLGISManager(db_config)
                await self.database_manager.create_tables()
                self.active_integrations.add(IntegrationType.DATABASE_STORAGE)
            
            # Initialize ML optimization if enabled
            if self.config.ml_optimization_enabled and ML_AVAILABLE:
                opt_config_dict = components_config.get('optimization', {})
                opt_config = OptimizationConfig(**opt_config_dict)
                self.optimizer = ModelOptimizer(opt_config)
                self.active_integrations.add(IntegrationType.ML_OPTIMIZATION)
            
            # Initialize federated learning if enabled
            if self.config.federated_learning_enabled:
                fed_config_dict = components_config.get('federated_learning', {})
                fed_config = FederatedConfig(**fed_config_dict)
                self.federated_orchestrator = FederatedLearningOrchestrator(fed_config)
                self.active_integrations.add(IntegrationType.FEDERATED_LEARNING)
            
            # Setup data flow queues
            await self._setup_data_flows()
            
            # Start performance monitoring
            if self.config.performance_monitoring_enabled:
                await self._start_performance_monitoring()
            
            # Start real-time synchronization
            await self._start_real_time_sync()
            
            self.is_initialized = True
            self.integration_metrics.integration_health = "HEALTHY"
            
            self.logger.info(f"Digital Twin Earth integration initialized with {len(self.active_integrations)} active integrations")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize integration: {e}")
            self.integration_metrics.integration_health = "CRITICAL"
            raise
    
    async def _setup_data_flows(self):
        """Setup data flow queues between components"""
        flow_types = [
            "earth_state_updates",
            "observation_data",
            "prediction_results",
            "anomaly_alerts",
            "performance_metrics"
        ]
        
        for flow_type in flow_types:
            self.data_queues[flow_type] = asyncio.Queue(maxsize=1000)
        
        # Setup data flow processors
        asyncio.create_task(self._process_earth_state_updates())
        asyncio.create_task(self._process_observation_data())
        asyncio.create_task(self._process_prediction_results())
        asyncio.create_task(self._process_anomaly_alerts())
    
    async def _process_earth_state_updates(self):
        """Process Earth system state updates"""
        while True:
            try:
                if "earth_state_updates" not in self.data_queues:
                    await asyncio.sleep(0.1)
                    continue
                
                earth_state = await self.data_queues["earth_state_updates"].get()
                
                # Stream to Kafka if enabled
                if self.stream_manager:
                    event = self._create_stream_event(earth_state, StreamEventType.GEOLOGICAL_EVENT)
                    await self.stream_manager.produce_event(event, "earth_system_updates")
                
                # Store in database if enabled
                if self.database_manager:
                    await self._store_earth_state(earth_state)
                
                # Update federated learning if enabled
                if self.federated_orchestrator:
                    await self._update_federated_models(earth_state)
                
            except Exception as e:
                self.logger.error(f"Error processing earth state updates: {e}")
                await asyncio.sleep(1)
    
    async def _process_observation_data(self):
        """Process incoming observation data"""
        while True:
            try:
                if "observation_data" not in self.data_queues:
                    await asyncio.sleep(0.1)
                    continue
                
                observation = await self.data_queues["observation_data"].get()
                
                # Validate data if enabled
                if self.config.data_validation_enabled:
                    if not await self._validate_observation(observation):
                        continue
                
                # Assimilate into Digital Twin Earth
                if self.digital_twin:
                    await self.digital_twin.assimilate_real_time_observation(observation)
                
                # Store in database
                if self.database_manager:
                    await self._store_observation(observation)
                
            except Exception as e:
                self.logger.error(f"Error processing observation data: {e}")
                await asyncio.sleep(1)
    
    async def _process_prediction_results(self):
        """Process prediction results from Digital Twin"""
        while True:
            try:
                if "prediction_results" not in self.data_queues:
                    await asyncio.sleep(0.1)
                    continue
                
                prediction = await self.data_queues["prediction_results"].get()
                
                # Stream predictions to real-time systems
                if self.stream_manager:
                    event = self._create_prediction_event(prediction)
                    await self.stream_manager.produce_event(event, "predictions")
                
                # Store predictions in database
                if self.database_manager:
                    await self._store_prediction(prediction)
                
                # Check for anomalies
                if await self._detect_prediction_anomaly(prediction):
                    await self.data_queues["anomaly_alerts"].put(prediction)
                
            except Exception as e:
                self.logger.error(f"Error processing prediction results: {e}")
                await asyncio.sleep(1)
    
    async def _process_anomaly_alerts(self):
        """Process anomaly alerts"""
        while True:
            try:
                if "anomaly_alerts" not in self.data_queues:
                    await asyncio.sleep(0.1)
                    continue
                
                alert_data = await self.data_queues["anomaly_alerts"].get()
                
                # Create high-priority stream event
                if self.stream_manager:
                    alert_event = self._create_alert_event(alert_data)
                    await self.stream_manager.produce_event(alert_event, "alerts")
                
                # Store alert in database
                if self.database_manager:
                    await self._store_alert(alert_data)
                
                self.logger.warning(f"Anomaly detected: {alert_data}")
                
            except Exception as e:
                self.logger.error(f"Error processing anomaly alerts: {e}")
                await asyncio.sleep(1)
    
    def _create_stream_event(self, earth_state: EarthSystemState, event_type: StreamEventType) -> StreamEvent:
        """Create stream event from Earth system state"""
        return StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=earth_state.timestamp,
            source_id="digital_twin_earth",
            data={
                "atmosphere": earth_state.atmosphere_state,
                "hydrosphere": earth_state.hydrosphere_state,
                "lithosphere": earth_state.lithosphere_state,
                "biosphere": earth_state.biosphere_state,
                "cryosphere": earth_state.cryosphere_state,
                "anthroposphere": earth_state.anthroposphere_state
            },
            metadata={
                "model_resolution": earth_state.spatial_resolution,
                "uncertainty_level": earth_state.uncertainty_metrics.get("total_uncertainty", 0.0),
                "integration_version": "1.0"
            }
        )
    
    def _create_prediction_event(self, prediction: PredictiveScenario) -> StreamEvent:
        """Create stream event from prediction results"""
        return StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=StreamEventType.GEOLOGICAL_EVENT,
            timestamp=datetime.utcnow(),
            source_id="digital_twin_predictor",
            data={
                "scenario_id": prediction.scenario_id,
                "predicted_states": prediction.predicted_states,
                "probability": prediction.probability,
                "time_horizon": prediction.time_horizon.total_seconds()
            },
            metadata={
                "model_version": prediction.model_version,
                "confidence_level": prediction.confidence_level,
                "uncertainty_bounds": prediction.uncertainty_bounds
            }
        )
    
    def _create_alert_event(self, alert_data: Any) -> StreamEvent:
        """Create alert stream event"""
        return StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=StreamEventType.ALERT,
            timestamp=datetime.utcnow(),
            source_id="digital_twin_anomaly_detector",
            data={
                "alert_type": "anomaly_detected",
                "severity": "HIGH",
                "description": f"Anomaly detected in Digital Twin Earth system",
                "alert_data": str(alert_data)
            },
            metadata={
                "priority": "high",
                "requires_action": True
            }
        )
    
    async def _validate_observation(self, observation: ObservationData) -> bool:
        """Validate incoming observation data"""
        try:
            # Check required fields
            if not observation.observation_id or not observation.timestamp or not observation.value:
                return False
            
            # Check data quality
            if observation.quality_score < 0.5:  # Minimum quality threshold
                return False
            
            # Check temporal consistency
            time_diff = abs((datetime.utcnow() - observation.timestamp).total_seconds())
            if time_diff > 3600:  # More than 1 hour old
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating observation: {e}")
            return False
    
    async def _detect_prediction_anomaly(self, prediction: PredictiveScenario) -> bool:
        """Detect anomalies in prediction results"""
        try:
            # Check confidence level
            if prediction.confidence_level < 0.3:  # Low confidence
                return True
            
            # Check prediction magnitude
            if hasattr(prediction, 'magnitude') and prediction.magnitude > 7.0:
                return True
            
            # Check uncertainty bounds
            if prediction.uncertainty_bounds and max(prediction.uncertainty_bounds) > 0.8:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting prediction anomaly: {e}")
            return False
    
    async def _store_earth_state(self, earth_state: EarthSystemState):
        """Store Earth system state in database"""
        try:
            event_data = {
                'event_id': f"earth_state_{earth_state.timestamp.isoformat()}",
                'event_type': 'earth_system_update',
                'timestamp': earth_state.timestamp,
                'source_id': 'digital_twin_earth',
                'data': {
                    'atmosphere': earth_state.atmosphere_state,
                    'hydrosphere': earth_state.hydrosphere_state,
                    'lithosphere': earth_state.lithosphere_state,
                    'biosphere': earth_state.biosphere_state,
                    'cryosphere': earth_state.cryosphere_state,
                    'anthroposphere': earth_state.anthroposphere_state
                },
                'metadata': {
                    'spatial_resolution': earth_state.spatial_resolution,
                    'uncertainty_metrics': earth_state.uncertainty_metrics
                }
            }
            
            await self.database_manager.insert_geological_event(event_data)
            
        except Exception as e:
            self.logger.error(f"Error storing earth state: {e}")
    
    async def _store_observation(self, observation: ObservationData):
        """Store observation data in database"""
        try:
            event_data = {
                'event_id': observation.observation_id,
                'event_type': observation.observation_type,
                'timestamp': observation.timestamp,
                'source_id': observation.source_id,
                'location': observation.location,
                'data': {
                    'value': observation.value,
                    'unit': observation.unit,
                    'quality_score': observation.quality_score
                },
                'metadata': observation.metadata
            }
            
            await self.database_manager.insert_geological_event(event_data)
            
        except Exception as e:
            self.logger.error(f"Error storing observation: {e}")
    
    async def _store_prediction(self, prediction: PredictiveScenario):
        """Store prediction results in database"""
        try:
            event_data = {
                'event_id': prediction.scenario_id,
                'event_type': 'prediction',
                'timestamp': datetime.utcnow(),
                'source_id': 'digital_twin_predictor',
                'data': {
                    'predicted_states': prediction.predicted_states,
                    'probability': prediction.probability,
                    'time_horizon_seconds': prediction.time_horizon.total_seconds()
                },
                'metadata': {
                    'model_version': prediction.model_version,
                    'confidence_level': prediction.confidence_level,
                    'uncertainty_bounds': prediction.uncertainty_bounds
                }
            }
            
            await self.database_manager.insert_geological_event(event_data)
            
        except Exception as e:
            self.logger.error(f"Error storing prediction: {e}")
    
    async def _store_alert(self, alert_data: Any):
        """Store alert in database"""
        try:
            event_data = {
                'event_id': f"alert_{datetime.utcnow().isoformat()}",
                'event_type': 'anomaly_alert',
                'timestamp': datetime.utcnow(),
                'source_id': 'digital_twin_anomaly_detector',
                'data': {
                    'alert_data': str(alert_data),
                    'severity': 'HIGH'
                },
                'metadata': {
                    'requires_action': True,
                    'detection_method': 'digital_twin_integration'
                }
            }
            
            await self.database_manager.insert_geological_event(event_data)
            
        except Exception as e:
            self.logger.error(f"Error storing alert: {e}")
    
    async def _update_federated_models(self, earth_state: EarthSystemState):
        """Update federated learning models with Earth state data"""
        try:
            if not self.federated_orchestrator:
                return
            
            # Convert earth state to training data format
            features = np.array([
                earth_state.atmosphere_state.get('temperature', 0),
                earth_state.atmosphere_state.get('pressure', 0),
                earth_state.hydrosphere_state.get('sea_level', 0),
                earth_state.lithosphere_state.get('stress_field', 0),
                earth_state.biosphere_state.get('vegetation_index', 0),
                earth_state.cryosphere_state.get('ice_coverage', 0),
                earth_state.anthroposphere_state.get('activity_level', 0)
            ]).reshape(1, -1)
            
            # This would trigger federated model updates
            # Implementation depends on specific federated learning architecture
            
        except Exception as e:
            self.logger.error(f"Error updating federated models: {e}")
    
    async def _start_real_time_sync(self):
        """Start real-time synchronization between components"""
        async def sync_loop():
            while True:
                try:
                    start_time = time.time()
                    
                    # Get current Earth system state
                    if self.digital_twin:
                        current_state = self.digital_twin.get_current_state()
                        if current_state:
                            await self.data_queues["earth_state_updates"].put(current_state)
                    
                    # Update sync metrics
                    sync_time = (time.time() - start_time) * 1000
                    self.integration_metrics.sync_latency_ms = sync_time
                    self.integration_metrics.last_sync_timestamp = datetime.utcnow()
                    
                    await asyncio.sleep(self.config.real_time_sync_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in real-time sync: {e}")
                    await asyncio.sleep(5)
        
        self.sync_tasks['main_sync'] = asyncio.create_task(sync_loop())
    
    async def _start_performance_monitoring(self):
        """Start performance monitoring"""
        async def monitor_loop():
            while True:
                try:
                    # Calculate performance metrics
                    await self._update_performance_metrics()
                    
                    # Check health status
                    await self._check_integration_health()
                    
                    await asyncio.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in performance monitoring: {e}")
                    await asyncio.sleep(60)
        
        self.performance_monitor = asyncio.create_task(monitor_loop())
    
    async def _update_performance_metrics(self):
        """Update integration performance metrics"""
        try:
            # Count active connections
            active_connections = len(self.active_integrations)
            
            # Calculate data throughput (simplified)
            throughput = 0.0
            for queue_name, queue in self.data_queues.items():
                throughput += queue.qsize() * 0.001  # Approximate MB
            
            # Update metrics
            self.integration_metrics.active_connections = active_connections
            self.integration_metrics.data_throughput_mbps = throughput
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _check_integration_health(self):
        """Check overall integration health"""
        try:
            health_score = 1.0
            
            # Check component availability
            if not self.digital_twin:
                health_score -= 0.3
            if self.config.stream_processing_enabled and not self.stream_manager:
                health_score -= 0.2
            if self.config.database_storage_enabled and not self.database_manager:
                health_score -= 0.2
            
            # Check sync latency
            if self.integration_metrics.sync_latency_ms > 1000:  # >1 second
                health_score -= 0.2
            
            # Check queue sizes
            for queue in self.data_queues.values():
                if queue.qsize() > 500:  # Queue getting full
                    health_score -= 0.1
            
            # Update health status
            if health_score >= 0.8:
                self.integration_metrics.integration_health = "HEALTHY"
            elif health_score >= 0.5:
                self.integration_metrics.integration_health = "DEGRADED"
            else:
                self.integration_metrics.integration_health = "CRITICAL"
            
        except Exception as e:
            self.logger.error(f"Error checking integration health: {e}")
            self.integration_metrics.integration_health = "CRITICAL"
    
    async def add_real_time_observation(self, observation: ObservationData):
        """Add real-time observation to the integration system"""
        try:
            await self.data_queues["observation_data"].put(observation)
            
        except Exception as e:
            self.logger.error(f"Error adding real-time observation: {e}")
    
    async def get_current_predictions(self, time_horizon: timedelta = None) -> List[PredictiveScenario]:
        """Get current predictions from Digital Twin Earth"""
        try:
            if not self.digital_twin:
                return []
            
            time_horizon = time_horizon or timedelta(hours=24)
            predictions = await self.digital_twin.generate_predictive_scenarios(
                scenario_count=5,
                time_horizon=time_horizon
            )
            
            # Add predictions to processing queue
            for prediction in predictions:
                await self.data_queues["prediction_results"].put(prediction)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error getting current predictions: {e}")
            return []
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        return {
            'is_initialized': self.is_initialized,
            'active_integrations': [integration.value for integration in self.active_integrations],
            'metrics': {
                'sync_latency_ms': self.integration_metrics.sync_latency_ms,
                'data_throughput_mbps': self.integration_metrics.data_throughput_mbps,
                'error_rate_percent': self.integration_metrics.error_rate_percent,
                'active_connections': self.integration_metrics.active_connections,
                'last_sync_timestamp': self.integration_metrics.last_sync_timestamp.isoformat(),
                'integration_health': self.integration_metrics.integration_health
            },
            'data_queue_sizes': {
                queue_name: queue.qsize() 
                for queue_name, queue in self.data_queues.items()
            },
            'component_status': {
                'digital_twin': self.digital_twin is not None,
                'stream_manager': self.stream_manager is not None,
                'database_manager': self.database_manager is not None,
                'optimizer': self.optimizer is not None,
                'federated_orchestrator': self.federated_orchestrator is not None
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown the integration system"""
        try:
            self.logger.info("Shutting down Digital Twin Earth integration...")
            
            # Cancel sync tasks
            for task_name, task in self.sync_tasks.items():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Cancel performance monitor
            if self.performance_monitor:
                self.performance_monitor.cancel()
                try:
                    await self.performance_monitor
                except asyncio.CancelledError:
                    pass
            
            # Shutdown Digital Twin Earth
            if self.digital_twin:
                await self.digital_twin.shutdown()
            
            # Clear queues
            for queue in self.data_queues.values():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            
            self.is_initialized = False
            self.integration_metrics.integration_health = "SHUTDOWN"
            
            self.logger.info("Digital Twin Earth integration shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Example usage and testing
async def example_digital_twin_integration():
    """Example of Digital Twin Earth integration"""
    
    print("=== Digital Twin Earth Integration Example ===")
    
    # Configuration
    integration_config = IntegrationConfig(
        stream_processing_enabled=True,
        database_storage_enabled=True,
        ml_optimization_enabled=True,
        federated_learning_enabled=True,
        real_time_sync_interval=2.0
    )
    
    # Component configurations
    components_config = {
        'digital_twin': {
            'spatial_resolution': 1.0,  # 1 degree
            'temporal_resolution': 300,  # 5 minutes
            'enable_coupling': True,
            'enable_data_assimilation': True
        },
        'kafka': {
            'bootstrap_servers': ['localhost:9092'],
            'topics': {
                'earth_system_updates': 'earth_system_updates',
                'predictions': 'predictions',
                'alerts': 'alerts'
            }
        },
        'database': {
            'host': 'localhost',
            'database': 'prometheus_earth_twin',
            'username': 'postgres',
            'password': 'password'
        }
    }
    
    # Initialize integrator
    integrator = DigitalTwinEarthIntegrator(integration_config)
    
    try:
        # Initialize integration
        await integrator.initialize(components_config)
        
        # Simulate real-time observations
        for i in range(5):
            observation = ObservationData(
                observation_id=f"test_obs_{i}",
                observation_type="temperature",
                timestamp=datetime.utcnow(),
                source_id=f"sensor_{i}",
                location=(35.0 + i * 0.1, -118.0 + i * 0.1, 0.0),
                value=20.0 + i * 2.0,
                unit="celsius",
                quality_score=0.95,
                metadata={"sensor_type": "thermometer"}
            )
            
            await integrator.add_real_time_observation(observation)
            print(f"Added observation {i}: {observation.value}°C")
        
        # Wait for processing
        await asyncio.sleep(5)
        
        # Get predictions
        predictions = await integrator.get_current_predictions(timedelta(hours=12))
        print(f"Generated {len(predictions)} predictions")
        
        # Check integration status
        status = await integrator.get_integration_status()
        print(f"Integration status: {status['metrics']['integration_health']}")
        print(f"Active integrations: {status['active_integrations']}")
        print(f"Sync latency: {status['metrics']['sync_latency_ms']:.2f}ms")
        
        # Run for a bit to demonstrate real-time operations
        print("Running integration for 10 seconds...")
        await asyncio.sleep(10)
        
        # Final status check
        final_status = await integrator.get_integration_status()
        print(f"Final sync latency: {final_status['metrics']['sync_latency_ms']:.2f}ms")
        print(f"Data throughput: {final_status['metrics']['data_throughput_mbps']:.2f} MB/s")
        
    finally:
        # Shutdown
        await integrator.shutdown()
    
    print("Digital Twin Earth integration example completed!")

if __name__ == "__main__":
    print("=== Digital Twin Earth Integration System ===")
    print(f"Available features:")
    print(f"  Prometheus Modules: {MODULES_AVAILABLE}")
    print(f"  ML Libraries: {ML_AVAILABLE}")
    print(f"  Scientific Computing: {SCIENTIFIC_AVAILABLE}")
    
    # Run example
    asyncio.run(example_digital_twin_integration())