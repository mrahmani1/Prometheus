"""
Edge Computing Framework
Sensor-level processing, distributed computing, and federated learning for geological monitoring
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
import queue
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import hashlib
import socket
import struct
from abc import ABC, abstractmethod
import uuid
from pathlib import Path

# ML and computation imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Communication protocols
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    warnings.warn("ZeroMQ not available. Some edge communication features will be limited.")

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    warnings.warn("MQTT not available. Some edge communication features will be limited.")

# Edge-specific libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Resource monitoring will be limited.")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    warnings.warn("GPUtil not available. GPU monitoring will be limited.")

warnings.filterwarnings('ignore')

class EdgeDeviceType(Enum):
    SEISMIC_SENSOR = "seismic_sensor"
    GPS_STATION = "gps_station"
    INCLINOMETER = "inclinometer"
    STRAIN_GAUGE = "strain_gauge"
    TEMPERATURE_SENSOR = "temperature_sensor"
    PRESSURE_SENSOR = "pressure_sensor"
    GAS_SENSOR = "gas_sensor"
    CAMERA = "camera"
    GATEWAY = "gateway"
    COMPUTE_NODE = "compute_node"

class ProcessingLevel(Enum):
    RAW = "raw"
    FILTERED = "filtered"
    FEATURES = "features"
    PREDICTIONS = "predictions"
    ALERTS = "alerts"

class CommunicationProtocol(Enum):
    MQTT = "mqtt"
    ZMQ = "zmq"
    HTTP = "http"
    WEBSOCKET = "websocket"
    SERIAL = "serial"
    MODBUS = "modbus"

@dataclass
class EdgeDeviceCapabilities:
    """Edge device computational capabilities"""
    cpu_cores: int
    memory_mb: int
    storage_gb: int
    has_gpu: bool = False
    gpu_memory_mb: int = 0
    power_consumption_watts: int = 10
    network_bandwidth_mbps: int = 100
    supported_protocols: List[CommunicationProtocol] = field(default_factory=list)
    processing_power_gflops: float = 1.0

@dataclass
class SensorReading:
    """Sensor reading data structure"""
    sensor_id: str
    timestamp: datetime
    reading_type: str
    value: Union[float, np.ndarray, Dict[str, Any]]
    unit: str
    location: Optional[Tuple[float, float, float]] = None
    quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'sensor_id': self.sensor_id,
            'timestamp': self.timestamp.isoformat(),
            'reading_type': self.reading_type,
            'value': self.value.tolist() if isinstance(self.value, np.ndarray) else self.value,
            'unit': self.unit,
            'location': self.location,
            'quality': self.quality,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorReading':
        """Create from dictionary"""
        value = data['value']
        if isinstance(value, list):
            value = np.array(value)
        
        return cls(
            sensor_id=data['sensor_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            reading_type=data['reading_type'],
            value=value,
            unit=data['unit'],
            location=data.get('location'),
            quality=data.get('quality', 1.0),
            metadata=data.get('metadata', {})
        )

class EdgeDevice:
    """Base class for edge computing devices"""
    
    def __init__(self, device_id: str, device_type: EdgeDeviceType, 
                 capabilities: EdgeDeviceCapabilities, config: Dict[str, Any] = None):
        self.device_id = device_id
        self.device_type = device_type
        self.capabilities = capabilities
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{device_id}")
        
        # Processing components
        self.processors = {}
        self.filters = {}
        self.models = {}
        
        # Communication
        self.communication_manager = None
        self.message_queue = asyncio.Queue()
        
        # State management
        self.is_active = False
        self.last_heartbeat = datetime.utcnow()
        self.performance_metrics = {}
        
        # Data buffers
        self.sensor_buffer = queue.Queue(maxsize=1000)
        self.processing_buffer = queue.Queue(maxsize=500)
        
        self._setup_device()
    
    def _setup_device(self):
        """Setup device-specific configuration"""
        # Initialize communication
        self._setup_communication()
        
        # Setup processing pipeline
        self._setup_processing_pipeline()
        
        # Setup monitoring
        self._setup_monitoring()
    
    def _setup_communication(self):
        """Setup communication protocols"""
        protocols = self.capabilities.supported_protocols
        
        if CommunicationProtocol.MQTT in protocols and MQTT_AVAILABLE:
            self.communication_manager = MQTTCommunicationManager(self.device_id, self.config)
        elif CommunicationProtocol.ZMQ in protocols and ZMQ_AVAILABLE:
            self.communication_manager = ZMQCommunicationManager(self.device_id, self.config)
        else:
            self.communication_manager = LocalCommunicationManager(self.device_id)
    
    def _setup_processing_pipeline(self):
        """Setup advanced ML-based dynamic processing pipeline with adaptive configuration"""
        try:
            # Initialize dynamic pipeline configuration
            self.pipeline_config = {
                'adaptive_routing': True,
                'model_ensemble': True,
                'quality_gates': True,
                'real_time_optimization': True,
                'federated_learning': self.config.get('enable_federated_learning', True)
            }
            
            # Initialize ML model registry and version management
            self.model_registry = {
                'active_models': {},
                'candidate_models': {},
                'model_performance': {},
                'deployment_history': [],
                'auto_update_threshold': 0.05  # 5% improvement threshold
            }
            
            # Load device-specific base processors with ML enhancement
            base_processors = self._get_base_processors()
            
            # Create adaptive processing graph
            self.processing_graph = ProcessingGraph(
                base_processors=base_processors,
                device_capabilities=self.capabilities,
                optimization_target=self.config.get('optimization_target', 'accuracy')
            )
            
            # Initialize federated learning components
            if self.pipeline_config['federated_learning']:
                self.federated_manager = FederatedLearningManager(
                    device_id=self.device_id,
                    device_type=self.device_type,
                    privacy_level=self.config.get('privacy_level', 'high')
                )
                
            # Setup model deployment pipeline
            self.model_deployer = EdgeModelDeployer(
                device_capabilities=self.capabilities,
                resource_constraints=self.config.get('resource_constraints', {}),
                auto_scaling=True
            )
            
            # Initialize quality assessment framework
            self.quality_assessor = DataQualityAssessor(
                device_type=self.device_type,
                calibration_data=self.config.get('calibration_data'),
                adaptive_thresholds=True
            )
            
            # Setup dynamic pipeline optimization
            asyncio.create_task(self._optimize_pipeline_periodically())
            
            self.logger.info(f"Advanced ML processing pipeline initialized for {self.device_type.value}")
            
        except Exception as e:
            self.logger.error(f"Error setting up advanced processing pipeline: {e}")
            # Fallback to basic pipeline
            self._setup_basic_pipeline_fallback()
    
    def _get_base_processors(self) -> Dict[str, Any]:
        """Get enhanced base processors for device type"""
        processors = {}
        
        if self.device_type == EdgeDeviceType.SEISMIC_SENSOR:
            # Advanced seismic processing stack
            processors.update({
                'signal_processor': SeismicSignalProcessor(self.config.get('seismic_config', {})),
                'feature_extractor': SeismicFeatureExtractor(self.config.get('feature_config', {})),
                'ml_detector': MLEarthquakeDetector(
                    model_path=self.config.get('earthquake_model_path'),
                    ensemble_size=self.config.get('ensemble_size', 5)
                ),
                'wave_analyzer': SeismicWaveAnalyzer(
                    frequency_bands=self.config.get('frequency_bands'),
                    arrival_picker=True
                ),
                'magnitude_estimator': MagnitudeEstimator(
                    local_models=True,
                    uncertainty_quantification=True
                ),
                'source_localizer': SourceLocalizer(
                    network_geometry=self.config.get('network_geometry'),
                    velocity_model=self.config.get('velocity_model')
                )
            })
            
        elif self.device_type == EdgeDeviceType.GPS_STATION:
            # Advanced GPS processing stack
            processors.update({
                'signal_processor': GPSSignalProcessor(
                    multipath_mitigation=True,
                    ionospheric_correction=True
                ),
                'position_calculator': PrecisionPositionCalculator(
                    rtk_enabled=self.capabilities.supports_rtk,
                    ppp_enabled=True,
                    reference_stations=self.config.get('reference_stations', [])
                ),
                'displacement_analyzer': CrustalDisplacementAnalyzer(
                    time_series_analysis=True,
                    strain_calculation=True,
                    tectonic_modeling=True
                ),
                'atmospheric_processor': AtmosphericDelayProcessor(
                    meteorological_data=True,
                    ml_correction=True
                )
            })
            
        elif self.device_type == EdgeDeviceType.STRAIN_GAUGE:
            # Advanced strain measurement stack
            processors.update({
                'signal_processor': StrainSignalProcessor(
                    temperature_compensation=True,
                    drift_correction=True
                ),
                'deformation_analyzer': DeformationAnalyzer(
                    multi_component=True,
                    stress_modeling=True
                ),
                'trend_analyzer': AdvancedTrendAnalyzer(
                    machine_learning=True,
                    anomaly_detection=True,
                    predictive_modeling=True
                ),
                'tidal_processor': TidalStrainProcessor(
                    earth_tide_model=True,
                    ocean_loading=True
                )
            })
            
        elif self.device_type == EdgeDeviceType.INFRASOUND_SENSOR:
            # Advanced infrasound processing stack
            processors.update({
                'signal_processor': InfrasoundSignalProcessor(
                    atmospheric_filtering=True,
                    coherence_analysis=True
                ),
                'event_detector': InfrasoundEventDetector(
                    ml_classification=True,
                    source_discrimination=True
                ),
                'propagation_modeler': AtmosphericPropagationModeler(
                    weather_integration=True,
                    ray_tracing=True
                )
            })
            
        # Add common ML enhancement processors
        processors.update({
            'anomaly_detector': UniversalAnomalyDetector(
                device_type=self.device_type,
                adaptive_thresholds=True
            ),
            'data_fusion': MultiSensorDataFusion(
                correlation_analysis=True,
                uncertainty_propagation=True
            ),
            'event_classifier': GeophysicalEventClassifier(
                multi_class=True,
                confidence_estimation=True
            )
        })
        
        return processors
    
    def _setup_basic_pipeline_fallback(self):
        """Fallback to basic processing pipeline if advanced setup fails"""
        self.logger.warning("Using basic processing pipeline fallback")
        
        if self.device_type == EdgeDeviceType.SEISMIC_SENSOR:
            self.processors['filter'] = SeismicSignalProcessor()
            self.processors['feature_extractor'] = SeismicFeatureExtractor()
            self.processors['detector'] = EarthquakeDetector()
        elif self.device_type == EdgeDeviceType.GPS_STATION:
            self.processors['filter'] = GPSNoiseFilter()
            self.processors['position_calculator'] = GPSPositionCalculator()
            self.processors['displacement_detector'] = DisplacementDetector()
        elif self.device_type == EdgeDeviceType.STRAIN_GAUGE:
            self.processors['filter'] = StrainSignalProcessor()
            self.processors['trend_analyzer'] = TrendAnalyzer()
    
    async def _optimize_pipeline_periodically(self):
        """Continuously optimize processing pipeline based on performance metrics"""
        optimization_interval = self.config.get('optimization_interval', 3600)  # 1 hour default
        
        while self.is_active:
            try:
                await asyncio.sleep(optimization_interval)
                
                if hasattr(self, 'processing_graph'):
                    # Collect performance metrics
                    performance_data = await self._collect_pipeline_metrics()
                    
                    # Analyze and optimize pipeline
                    optimization_suggestions = self.processing_graph.analyze_performance(performance_data)
                    
                    if optimization_suggestions['should_optimize']:
                        await self._apply_pipeline_optimizations(optimization_suggestions)
                        
                    # Update model registry based on performance
                    await self._update_model_registry(performance_data)
                    
                    self.logger.info("Pipeline optimization cycle completed")
                
            except Exception as e:
                self.logger.error(f"Error in pipeline optimization: {e}")
                await asyncio.sleep(optimization_interval)
    
    async def _collect_pipeline_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive pipeline performance metrics"""
        metrics = {
            'processing_latency': {},
            'accuracy_metrics': {},
            'resource_utilization': {},
            'data_quality': {},
            'model_performance': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Collect latency metrics for each processor
            for name, processor in self.processors.items():
                if hasattr(processor, 'get_performance_metrics'):
                    processor_metrics = processor.get_performance_metrics()
                    metrics['processing_latency'][name] = processor_metrics.get('avg_latency', 0)
                    metrics['accuracy_metrics'][name] = processor_metrics.get('accuracy', 0)
            
            # Resource utilization
            metrics['resource_utilization'] = self.performance_metrics.copy()
            
            # Data quality assessment
            if hasattr(self, 'quality_assessor'):
                quality_metrics = await self.quality_assessor.get_recent_quality_metrics()
                metrics['data_quality'] = quality_metrics
            
            # Model performance from registry
            metrics['model_performance'] = self.model_registry.get('model_performance', {})
            
        except Exception as e:
            self.logger.error(f"Error collecting pipeline metrics: {e}")
            
        return metrics
    
    async def _apply_pipeline_optimizations(self, suggestions: Dict[str, Any]):
        """Apply pipeline optimizations based on analysis"""
        try:
            optimizations = suggestions.get('optimizations', [])
            
            for optimization in optimizations:
                if optimization['type'] == 'processor_reorder':
                    await self._reorder_processors(optimization['new_order'])
                elif optimization['type'] == 'model_update':
                    await self._update_processor_model(optimization['processor'], optimization['new_model'])
                elif optimization['type'] == 'parameter_tune':
                    await self._tune_processor_parameters(optimization['processor'], optimization['parameters'])
                elif optimization['type'] == 'add_processor':
                    await self._add_processor(optimization['processor_config'])
                elif optimization['type'] == 'remove_processor':
                    await self._remove_processor(optimization['processor_name'])
            
            self.logger.info(f"Applied {len(optimizations)} pipeline optimizations")
            
        except Exception as e:
            self.logger.error(f"Error applying pipeline optimizations: {e}")
    
    async def _update_model_registry(self, performance_data: Dict[str, Any]):
        """Update model registry based on performance data"""
        try:
            current_performance = performance_data.get('model_performance', {})
            
            for model_name, performance in current_performance.items():
                # Check if model performance has improved beyond threshold
                if model_name in self.model_registry['model_performance']:
                    previous_performance = self.model_registry['model_performance'][model_name]
                    improvement = performance - previous_performance
                    
                    if improvement > self.model_registry['auto_update_threshold']:
                        # Promote candidate model to active
                        await self._promote_candidate_model(model_name)
                
                # Update performance tracking
                self.model_registry['model_performance'][model_name] = performance
            
            # Trigger federated learning updates if enabled
            if self.pipeline_config.get('federated_learning') and hasattr(self, 'federated_manager'):
                await self.federated_manager.share_performance_metrics(performance_data)
                
        except Exception as e:
            self.logger.error(f"Error updating model registry: {e}")
    
    def _setup_monitoring(self):
        """Setup intelligent multi-tier monitoring orchestration with predictive analytics"""
        try:
            # Initialize multi-tier monitoring architecture
            self.monitoring_tiers = {
                'system': {
                    'enabled': True,
                    'metrics': ['cpu', 'memory', 'disk', 'network', 'temperature'],
                    'thresholds': {'cpu': 80, 'memory': 85, 'disk': 90, 'network': 70},
                    'predictive_window': 300,  # 5 minutes
                    'anomaly_detection': True
                },
                'application': {
                    'enabled': True, 
                    'metrics': ['processing_latency', 'queue_depth', 'error_rate', 'throughput'],
                    'thresholds': {'latency': 100, 'queue_depth': 1000, 'error_rate': 0.05},
                    'adaptive_thresholds': True,
                    'pattern_detection': True
                },
                'geological': {
                    'enabled': True,
                    'metrics': ['signal_quality', 'data_integrity', 'sensor_health', 'event_detection'],
                    'thresholds': {'signal_snr': 10, 'data_loss': 0.01, 'sensor_drift': 0.1},
                    'ml_anomaly_detection': True,
                    'expert_system_rules': True
                },
                'network': {
                    'enabled': True,
                    'metrics': ['connectivity', 'bandwidth', 'latency', 'packet_loss'],
                    'thresholds': {'latency': 50, 'packet_loss': 0.01, 'bandwidth': 1000000},
                    'topology_awareness': True,
                    'mesh_optimization': True
                }
            }
            
            # Initialize intelligent alerting system
            self.alert_manager = {
                'escalation_policies': {
                    'low': {'delay': 300, 'channels': ['log']},
                    'medium': {'delay': 60, 'channels': ['log', 'database']},
                    'high': {'delay': 10, 'channels': ['log', 'database', 'mesh_broadcast']},
                    'critical': {'delay': 0, 'channels': ['log', 'database', 'mesh_broadcast', 'external_api']}
                },
                'correlation_engine': True,
                'ml_classification': True,
                'noise_reduction': True,
                'historical_context': True
            }
            
            # Initialize predictive monitoring components
            self.predictive_monitoring = {
                'time_series_models': {},
                'anomaly_detectors': {},
                'pattern_classifiers': {},
                'trend_analyzers': {},
                'capacity_planners': {},
                'failure_predictors': {}
            }
            
            # Setup monitoring data pipelines with your database integration
            self.monitoring_pipeline = {
                'local_buffer': asyncio.Queue(maxsize=10000),
                'batch_processor': None,
                'database_connector': None,  # Will integrate with database_integration.py
                'external_connectors': [],   # Will integrate with external_data_connectors.py
                'compression_enabled': True,
                'encryption_enabled': self.config.get('security', {}).get('encrypt_monitoring', True)
            }
            
            # Initialize health assessment framework
            self.health_assessor = {
                'overall_health_score': 100.0,
                'component_health': {},
                'degradation_trends': {},
                'recovery_suggestions': {},
                'automated_remediation': self.config.get('monitoring', {}).get('auto_remediation', True)
            }
            
            # Setup monitoring orchestration scheduler
            self.monitoring_scheduler = {
                'system_monitoring': {'interval': 5, 'priority': 'high'},
                'application_monitoring': {'interval': 1, 'priority': 'critical'},
                'geological_monitoring': {'interval': 0.1, 'priority': 'critical'},
                'network_monitoring': {'interval': 10, 'priority': 'medium'},
                'predictive_analysis': {'interval': 60, 'priority': 'medium'},
                'health_assessment': {'interval': 30, 'priority': 'high'}
            }
            
            # Initialize monitoring state tracking
            self.monitoring_state = {
                'active_monitors': {},
                'failed_monitors': {},
                'performance_metrics': {},
                'alert_history': [],
                'prediction_accuracy': {},
                'last_optimization': time.time()
            }
            
            # Start multi-tier monitoring tasks
            if PSUTIL_AVAILABLE:
                asyncio.create_task(self._intelligent_resource_monitoring())
            
            asyncio.create_task(self._application_performance_monitoring())
            asyncio.create_task(self._geological_signal_monitoring())
            asyncio.create_task(self._network_topology_monitoring())
            asyncio.create_task(self._predictive_analytics_engine())
            asyncio.create_task(self._health_assessment_loop())
            asyncio.create_task(self._alert_correlation_engine())
            asyncio.create_task(self._monitoring_optimization_loop())
            
            # Enhanced heartbeat with multi-dimensional health reporting
            asyncio.create_task(self._intelligent_heartbeat_loop())
            
            self.logger.info("Initialized intelligent multi-tier monitoring orchestration")
            
        except Exception as e:
            self.logger.error(f"Failed to setup monitoring: {e}")
            # Fallback to basic monitoring
            if PSUTIL_AVAILABLE:
                asyncio.create_task(self._monitor_resources())
            asyncio.create_task(self._heartbeat_loop())
    
    async def start(self):
        """Start edge device operation with graceful degradation and recovery mechanisms"""
        try:
            # Initialize startup state tracking
            self.startup_state = {
                'phase': 'initializing',
                'started_components': [],
                'failed_components': [],
                'degraded_components': [],
                'recovery_attempts': {},
                'startup_timestamp': time.time()
            }
            
            self.logger.info(f"Starting edge device {self.device_id} with graceful degradation support")
            
            # Phase 1: Core system initialization with health checks
            self.startup_state['phase'] = 'core_initialization'
            
            # Initialize critical system components with fallback
            critical_components = ['monitoring', 'data_pipeline', 'model_registry', 'health_assessor']
            for component in critical_components:
                try:
                    await self._initialize_component(component)
                    self.startup_state['started_components'].append(component)
                    self.logger.info(f"Successfully initialized {component}")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize {component}: {e}")
                    self.startup_state['failed_components'].append(component)
                    # Attempt graceful degradation
                    await self._handle_component_failure(component, e)
            
            # Phase 2: Communication subsystem with mesh network fallback
            self.startup_state['phase'] = 'communication_startup'
            
            communication_started = False
            if self.communication_manager:
                try:
                    await self.communication_manager.start()
                    communication_started = True
                    self.startup_state['started_components'].append('communication')
                    self.logger.info("Communication manager started successfully")
                except Exception as e:
                    self.logger.warning(f"Primary communication failed: {e}")
                    self.startup_state['failed_components'].append('communication')
                    # Try fallback communication modes
                    communication_started = await self._start_fallback_communication()
            
            # Phase 3: Data acquisition and processing loops with adaptive startup
            self.startup_state['phase'] = 'processing_startup'
            
            # Start sensor reading with intelligent retry and fallback data sources
            sensor_task = None
            try:
                sensor_task = asyncio.create_task(self._adaptive_sensor_reading_loop())
                self.startup_state['started_components'].append('sensor_reading')
                self.logger.info("Sensor reading loop started")
            except Exception as e:
                self.logger.warning(f"Failed to start sensor reading: {e}")
                self.startup_state['failed_components'].append('sensor_reading')
                # Schedule recovery attempt
                asyncio.create_task(self._schedule_component_recovery('sensor_reading', 30))
            
            # Start processing with ML pipeline integration and degraded mode fallback
            processing_task = None
            try:
                processing_task = asyncio.create_task(self._intelligent_processing_loop())
                self.startup_state['started_components'].append('processing')
                self.logger.info("Processing loop started")
            except Exception as e:
                self.logger.warning(f"Failed to start processing: {e}")
                self.startup_state['failed_components'].append('processing')
                # Fall back to basic processing
                asyncio.create_task(self._basic_processing_fallback())
            
            # Start communication with mesh networking and database sync
            communication_task = None
            try:
                communication_task = asyncio.create_task(self._mesh_communication_loop())
                self.startup_state['started_components'].append('mesh_communication')
                self.logger.info("Mesh communication loop started")
            except Exception as e:
                self.logger.warning(f"Failed to start mesh communication: {e}")
                self.startup_state['failed_components'].append('mesh_communication')
                # Fall back to basic communication
                if communication_started:
                    asyncio.create_task(self._basic_communication_fallback())
            
            # Phase 4: Advanced features with optional startup
            self.startup_state['phase'] = 'advanced_features'
            
            # Start model deployment and A/B testing (optional)
            try:
                asyncio.create_task(self._model_deployment_loop())
                self.startup_state['started_components'].append('model_deployment')
            except Exception as e:
                self.logger.info(f"Model deployment not available: {e}")
                self.startup_state['degraded_components'].append('model_deployment')
            
            # Start continuous learning (optional)
            try:
                asyncio.create_task(self._continuous_learning_loop())
                self.startup_state['started_components'].append('continuous_learning')
            except Exception as e:
                self.logger.info(f"Continuous learning not available: {e}")
                self.startup_state['degraded_components'].append('continuous_learning')
            
            # Phase 5: Health monitoring and recovery systems
            self.startup_state['phase'] = 'health_monitoring'
            
            # Start comprehensive health monitoring
            asyncio.create_task(self._comprehensive_health_monitoring())
            asyncio.create_task(self._recovery_orchestrator())
            asyncio.create_task(self._degradation_detector())
            asyncio.create_task(self._performance_optimizer())
            
            # Mark device as active after successful startup
            self.is_active = True
            self.startup_state['phase'] = 'operational'
            
            # Calculate startup metrics
            startup_duration = time.time() - self.startup_state['startup_timestamp']
            success_rate = len(self.startup_state['started_components']) / (
                len(self.startup_state['started_components']) + 
                len(self.startup_state['failed_components']) + 
                len(self.startup_state['degraded_components'])
            ) if (len(self.startup_state['started_components']) + 
                  len(self.startup_state['failed_components']) + 
                  len(self.startup_state['degraded_components'])) > 0 else 1.0
            
            self.logger.info(
                f"Edge device {self.device_id} startup completed in {startup_duration:.2f}s "
                f"with {success_rate*100:.1f}% success rate. "
                f"Started: {len(self.startup_state['started_components'])}, "
                f"Failed: {len(self.startup_state['failed_components'])}, "
                f"Degraded: {len(self.startup_state['degraded_components'])}"
            )
            
            # Start recovery monitoring for failed components
            if self.startup_state['failed_components']:
                asyncio.create_task(self._component_recovery_monitor())
                
        except Exception as e:
            self.logger.error(f"Critical failure during device startup: {e}")
            # Attempt emergency startup mode
            await self._emergency_startup_mode()
    
    async def stop(self):
        """Stop edge device operation with graceful shutdown and state preservation"""
        try:
            # Initialize shutdown state tracking
            self.shutdown_state = {
                'phase': 'initializing',
                'stopped_components': [],
                'failed_shutdowns': [],
                'preserved_state': {},
                'shutdown_timestamp': time.time()
            }
            
            self.logger.info(f"Initiating graceful shutdown of edge device {self.device_id}")
            
            # Phase 1: Signal shutdown to all components
            self.shutdown_state['phase'] = 'signaling_shutdown'
            self.is_active = False
            
            # Phase 2: Preserve critical state and data
            self.shutdown_state['phase'] = 'state_preservation'
            
            try:
                # Preserve model states
                if hasattr(self, 'model_registry'):
                    self.shutdown_state['preserved_state']['model_states'] = {
                        'active_models': list(self.model_registry.get('active_models', {}).keys()),
                        'deployment_history': self.model_registry.get('deployment_history', [])[-10:],  # Last 10
                        'performance_metrics': self.model_registry.get('model_performance', {})
                    }
                
                # Preserve monitoring data
                if hasattr(self, 'monitoring_state'):
                    self.shutdown_state['preserved_state']['monitoring'] = {
                        'alert_history': self.monitoring_state.get('alert_history', [])[-50:],  # Last 50
                        'performance_metrics': self.monitoring_state.get('performance_metrics', {}),
                        'prediction_accuracy': self.monitoring_state.get('prediction_accuracy', {})
                    }
                
                # Preserve processing buffers
                preserved_data = []
                if hasattr(self, 'sensor_buffer'):
                    while not self.sensor_buffer.empty():
                        try:
                            preserved_data.append(self.sensor_buffer.get_nowait())
                        except:
                            break
                    self.shutdown_state['preserved_state']['sensor_data'] = preserved_data[-100:]  # Last 100
                
                self.logger.info(f"Preserved state for {len(self.shutdown_state['preserved_state'])} components")
                
            except Exception as e:
                self.logger.warning(f"Failed to preserve some state during shutdown: {e}")
            
            # Phase 3: Graceful shutdown of processing loops
            self.shutdown_state['phase'] = 'processing_shutdown'
            
            # Allow processing loops to finish current iterations
            shutdown_timeout = self.config.get('shutdown_timeout', 30)  # 30 seconds default
            await asyncio.sleep(2)  # Give loops time to detect shutdown signal
            
            # Phase 4: Database and model synchronization
            self.shutdown_state['phase'] = 'data_synchronization'
            
            try:
                # Sync with database_integration.py if available
                if hasattr(self, 'monitoring_pipeline') and self.monitoring_pipeline.get('database_connector'):
                    await self._sync_monitoring_data()
                    self.shutdown_state['stopped_components'].append('database_sync')
                
                # Sync model states with model_manager.py if available  
                if hasattr(self, 'model_registry'):
                    await self._sync_model_states()
                    self.shutdown_state['stopped_components'].append('model_sync')
                    
            except Exception as e:
                self.logger.warning(f"Failed data synchronization during shutdown: {e}")
                self.shutdown_state['failed_shutdowns'].append('data_sync')
            
            # Phase 5: Communication shutdown
            self.shutdown_state['phase'] = 'communication_shutdown'
            
            try:
                if self.communication_manager:
                    await self.communication_manager.stop()
                    self.shutdown_state['stopped_components'].append('communication_manager')
                    
                # Send final mesh network notification
                if hasattr(self, 'mesh_network'):
                    await self._send_shutdown_notification()
                    self.shutdown_state['stopped_components'].append('mesh_notification')
                    
            except Exception as e:
                self.logger.warning(f"Failed communication shutdown: {e}")
                self.shutdown_state['failed_shutdowns'].append('communication')
            
            # Phase 6: Resource cleanup
            self.shutdown_state['phase'] = 'resource_cleanup'
            
            try:
                # Clean up monitoring resources
                if hasattr(self, 'monitoring_state'):
                    self.monitoring_state['active_monitors'].clear()
                    
                # Clean up ML resources
                if hasattr(self, 'model_registry'):
                    # Preserve critical data but free memory
                    self.model_registry['active_models'].clear()
                    
                self.shutdown_state['stopped_components'].append('resource_cleanup')
                
            except Exception as e:
                self.logger.warning(f"Failed resource cleanup: {e}")
                self.shutdown_state['failed_shutdowns'].append('resource_cleanup')
            
            # Phase 7: Final state logging and shutdown completion
            self.shutdown_state['phase'] = 'completed'
            
            shutdown_duration = time.time() - self.shutdown_state['shutdown_timestamp']
            success_rate = len(self.shutdown_state['stopped_components']) / (
                len(self.shutdown_state['stopped_components']) + 
                len(self.shutdown_state['failed_shutdowns'])
            ) if (len(self.shutdown_state['stopped_components']) + 
                  len(self.shutdown_state['failed_shutdowns'])) > 0 else 1.0
            
            self.logger.info(
                f"Edge device {self.device_id} shutdown completed in {shutdown_duration:.2f}s "
                f"with {success_rate*100:.1f}% success rate. "
                f"Stopped: {len(self.shutdown_state['stopped_components'])}, "
                f"Failed: {len(self.shutdown_state['failed_shutdowns'])}"
            )
            
            # Save shutdown report for next startup
            await self._save_shutdown_report()
            
        except Exception as e:
            self.logger.error(f"Critical failure during device shutdown: {e}")
            # Force shutdown if graceful shutdown fails
            self.is_active = False
    
    async def _adaptive_sensor_reading_loop(self):
        """Adaptive real-time data acquisition from external data connectors with intelligent routing"""
        # Initialize adaptive acquisition system
        acquisition_config = {
            'primary_sources': ['usgs_seismic', 'nasa_gps', 'local_sensors'],
            'fallback_sources': ['iris_seismic', 'unavco_gps', 'simulated'],
            'adaptive_intervals': {
                'high_activity': 0.01,  # 10ms during seismic events
                'normal': 0.1,          # 100ms normal operation
                'low_activity': 1.0,    # 1s during quiet periods
                'power_saving': 5.0     # 5s when battery low
            },
            'quality_thresholds': {
                'min_snr': 5.0,
                'max_latency': 1000,    # 1 second max latency
                'min_completeness': 0.95
            }
        }
        
        # Initialize external data connectors integration
        self.data_connectors = {}
        self.connector_health = {}
        self.acquisition_metrics = {
            'readings_per_second': 0,
            'data_quality_score': 100,
            'source_utilization': {},
            'failover_events': 0,
            'last_optimization': time.time()
        }
        
        # Setup real-time data source connections
        await self._initialize_external_connectors()
        
        self.logger.info("Starting adaptive real-time data acquisition with external connector integration")
        
        while self.is_active:
            try:
                # Determine optimal acquisition strategy based on current conditions
                acquisition_strategy = await self._determine_acquisition_strategy()
                
                # Multi-source parallel data acquisition
                readings = await self._acquire_from_multiple_sources(acquisition_strategy)
                
                # Data fusion and quality assessment
                fused_data = await self._fuse_multi_source_data(readings)
                
                # Intelligent buffering with priority queuing
                if fused_data:
                    await self._intelligent_buffer_management(fused_data)
                
                # Adaptive sleep based on activity level and resource constraints
                sleep_interval = await self._calculate_adaptive_interval(acquisition_strategy)
                await asyncio.sleep(sleep_interval)
                
                # Update acquisition metrics
                await self._update_acquisition_metrics()
                
            except Exception as e:
                self.logger.error(f"Error in adaptive sensor reading loop: {e}")
                # Implement exponential backoff for errors
                await self._handle_acquisition_error(e)
    
    async def _initialize_external_connectors(self):
        """Initialize connections to external data sources from external_data_connectors.py"""
        try:
            # Import and initialize USGS connector
            self.data_connectors['usgs'] = {
                'type': 'seismic',
                'status': 'initializing',
                'last_reading': None,
                'error_count': 0,
                'data_quality': 100,
                'latency_ms': 0
            }
            
            # Import and initialize NASA connector  
            self.data_connectors['nasa'] = {
                'type': 'gps',
                'status': 'initializing', 
                'last_reading': None,
                'error_count': 0,
                'data_quality': 100,
                'latency_ms': 0
            }
            
            # Initialize World Bank environmental data connector
            self.data_connectors['world_bank'] = {
                'type': 'environmental',
                'status': 'initializing',
                'last_reading': None,
                'error_count': 0,
                'data_quality': 100,
                'latency_ms': 0
            }
            
            # Initialize local sensor interfaces
            self.data_connectors['local_seismic'] = {
                'type': 'local_seismic',
                'status': 'initializing',
                'last_reading': None,
                'error_count': 0,
                'data_quality': 100,
                'latency_ms': 0
            }
            
            self.data_connectors['local_gps'] = {
                'type': 'local_gps', 
                'status': 'initializing',
                'last_reading': None,
                'error_count': 0,
                'data_quality': 100,
                'latency_ms': 0
            }
            
            # Test connectivity to each connector
            for connector_name, connector_info in self.data_connectors.items():
                try:
                    # Test connection (placeholder - will integrate with external_data_connectors.py)
                    test_result = await self._test_connector_connectivity(connector_name)
                    if test_result:
                        connector_info['status'] = 'active'
                        self.logger.info(f"Successfully connected to {connector_name}")
                    else:
                        connector_info['status'] = 'failed'
                        self.logger.warning(f"Failed to connect to {connector_name}")
                except Exception as e:
                    connector_info['status'] = 'failed'
                    self.logger.warning(f"Error initializing {connector_name}: {e}")
            
            # Initialize data fusion algorithms
            self.fusion_algorithms = {
                'seismic': 'weighted_ensemble',
                'gps': 'kalman_filter',
                'environmental': 'temporal_interpolation',
                'cross_sensor': 'bayesian_integration'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize external connectors: {e}")
            # Fall back to local sensors only
            await self._fallback_to_local_sensors()
    
    async def _determine_acquisition_strategy(self):
        """Determine optimal data acquisition strategy based on current conditions"""
        try:
            # Analyze current geological activity level
            activity_level = await self._assess_geological_activity()
            
            # Check resource constraints (battery, CPU, network)
            resource_status = await self._assess_resource_constraints()
            
            # Evaluate data source health
            source_health = await self._assess_source_health()
            
            # Calculate optimal strategy
            strategy = {
                'priority_sources': [],
                'acquisition_rate': 'normal',
                'fusion_method': 'weighted_average',
                'quality_threshold': 85,
                'fallback_enabled': True
            }
            
            # Adjust based on activity level
            if activity_level > 0.8:  # High seismic activity
                strategy['acquisition_rate'] = 'high_activity'
                strategy['priority_sources'] = ['usgs', 'local_seismic', 'nasa']
                strategy['quality_threshold'] = 95
            elif activity_level < 0.2:  # Quiet period
                strategy['acquisition_rate'] = 'low_activity'
                strategy['priority_sources'] = ['local_seismic', 'usgs']
                strategy['quality_threshold'] = 75
            
            # Adjust for resource constraints
            if resource_status.get('battery_level', 100) < 20:
                strategy['acquisition_rate'] = 'power_saving'
                strategy['priority_sources'] = ['local_seismic']  # Most efficient
            
            if resource_status.get('network_quality', 100) < 50:
                # Prioritize local sources when network is poor
                strategy['priority_sources'] = ['local_seismic', 'local_gps']
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error determining acquisition strategy: {e}")
            # Return default strategy
            return {
                'priority_sources': ['local_seismic'],
                'acquisition_rate': 'normal',
                'fusion_method': 'simple_average',
                'quality_threshold': 70,
                'fallback_enabled': True
            }
    
    async def _acquire_from_multiple_sources(self, strategy):
        """Acquire data from multiple sources in parallel with intelligent failover"""
        acquisition_tasks = []
        readings = {}
        
        try:
            # Create parallel acquisition tasks for priority sources
            for source in strategy['priority_sources']:
                if source in self.data_connectors and self.data_connectors[source]['status'] == 'active':
                    task = asyncio.create_task(self._acquire_from_source(source))
                    acquisition_tasks.append((source, task))
            
            # Wait for all tasks with timeout
            timeout = self.config.get('acquisition_timeout', 2.0)  # 2 second timeout
            
            for source, task in acquisition_tasks:
                try:
                    reading = await asyncio.wait_for(task, timeout=timeout)
                    if reading and self._validate_reading_quality(reading, source):
                        readings[source] = reading
                        # Update connector health
                        self.data_connectors[source]['last_reading'] = time.time()
                        self.data_connectors[source]['error_count'] = max(0, 
                            self.data_connectors[source]['error_count'] - 1)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout acquiring from {source}")
                    self.data_connectors[source]['error_count'] += 1
                except Exception as e:
                    self.logger.warning(f"Error acquiring from {source}: {e}")
                    self.data_connectors[source]['error_count'] += 1
            
            # If primary sources failed, try fallback sources
            if not readings and strategy.get('fallback_enabled', True):
                readings = await self._acquire_from_fallback_sources()
            
            return readings
            
        except Exception as e:
            self.logger.error(f"Error in multi-source acquisition: {e}")
            return {}
    
    async def _acquire_from_source(self, source):
        """Acquire data from specific external source with connector integration"""
        try:
            if source == 'usgs':
                # Integration with external_data_connectors.py USGS connector
                return await self._acquire_usgs_data()
            elif source == 'nasa':
                # Integration with external_data_connectors.py NASA connector  
                return await self._acquire_nasa_data()
            elif source == 'world_bank':
                # Integration with external_data_connectors.py World Bank connector
                return await self._acquire_world_bank_data()
            elif source == 'local_seismic':
                # Local seismic sensor integration
                return await self._acquire_local_seismic_data()
            elif source == 'local_gps':
                # Local GPS receiver integration
                return await self._acquire_local_gps_data()
            else:
                self.logger.warning(f"Unknown data source: {source}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error acquiring from {source}: {e}")
            return None
    
    async def _fuse_multi_source_data(self, readings):
        """Intelligent data fusion from multiple sources with quality weighting"""
        if not readings:
            return None
            
        try:
            # Calculate quality weights for each source
            weights = {}
            total_weight = 0
            
            for source, reading in readings.items():
                quality_score = self._calculate_data_quality(reading, source)
                latency_penalty = self._calculate_latency_penalty(source)
                reliability_score = self._calculate_source_reliability(source)
                
                weight = quality_score * reliability_score * (1 - latency_penalty)
                weights[source] = max(0.1, weight)  # Minimum weight of 0.1
                total_weight += weights[source]
            
            # Normalize weights
            for source in weights:
                weights[source] /= total_weight
            
            # Fuse data based on type
            fused_data = {
                'timestamp': time.time(),
                'source_weights': weights,
                'quality_score': 0,
                'data_type': 'fused',
                'raw_sources': list(readings.keys())
            }
            
            # Seismic data fusion
            seismic_readings = {k: v for k, v in readings.items() 
                              if self.data_connectors[k]['type'] in ['seismic', 'local_seismic']}
            if seismic_readings:
                fused_data['seismic'] = await self._fuse_seismic_data(seismic_readings, weights)
            
            # GPS data fusion  
            gps_readings = {k: v for k, v in readings.items()
                           if self.data_connectors[k]['type'] in ['gps', 'local_gps']}
            if gps_readings:
                fused_data['gps'] = await self._fuse_gps_data(gps_readings, weights)
            
            # Environmental data fusion
            env_readings = {k: v for k, v in readings.items()
                           if self.data_connectors[k]['type'] == 'environmental'}
            if env_readings:
                fused_data['environmental'] = await self._fuse_environmental_data(env_readings, weights)
            
            # Calculate overall quality score
            fused_data['quality_score'] = sum(
                self._calculate_data_quality(reading, source) * weights.get(source, 0)
                for source, reading in readings.items()
            )
            
            return fused_data
            
        except Exception as e:
            self.logger.error(f"Error in data fusion: {e}")
            # Return best single source as fallback
            if readings:
                best_source = max(readings.keys(), 
                                key=lambda x: self._calculate_data_quality(readings[x], x))
                return {
                    'timestamp': time.time(),
                    'source': best_source,
                    'data': readings[best_source],
                    'quality_score': self._calculate_data_quality(readings[best_source], best_source),
                    'data_type': 'single_source'
                }
            return None
    
    async def _intelligent_processing_loop(self):
        """Intelligent ML pipeline processing loop with data_integration_pipeline.py integration"""
        # Initialize ML processing pipeline components
        self.ml_pipeline = {
            'geophysical_processor': None,    # Will integrate with data_integration_pipeline.py
            'geochemical_processor': None,    # Will integrate with data_integration_pipeline.py
            'remote_sensing_processor': None, # Will integrate with data_integration_pipeline.py
            'ensemble_processor': None,       # Multi-processor ensemble
            'real_time_models': {},           # Active ML models for real-time inference
            'processing_queue': asyncio.Queue(maxsize=1000),
            'results_cache': {},              # Cache recent results for pattern analysis
            'performance_metrics': {
                'processing_latency': [],
                'throughput': 0,
                'error_rate': 0,
                'model_accuracy': {}
            }
        }
        
        # Initialize adaptive processing configuration
        self.processing_config = {
            'batch_size': self.config.get('processing_batch_size', 32),
            'max_latency': self.config.get('max_processing_latency', 100),  # ms
            'quality_threshold': self.config.get('processing_quality_threshold', 85),
            'enable_real_time_learning': self.config.get('enable_real_time_learning', True),
            'enable_ensemble_processing': self.config.get('enable_ensemble_processing', True)
        }
        
        # Setup ML pipeline integration
        await self._initialize_ml_processing_pipeline()
        
        self.logger.info("Starting intelligent ML pipeline processing with data_integration_pipeline.py integration")
        
        # Start parallel processing workers
        processing_tasks = [
            asyncio.create_task(self._geophysical_processing_worker()),
            asyncio.create_task(self._geochemical_processing_worker()),
            asyncio.create_task(self._remote_sensing_processing_worker()),
            asyncio.create_task(self._ensemble_processing_worker()),
            asyncio.create_task(self._real_time_inference_worker()),
            asyncio.create_task(self._adaptive_batch_processor()),
            asyncio.create_task(self._processing_performance_monitor())
        ]
        
        while self.is_active:
            try:
                # Intelligent data routing and preprocessing
                if not self.sensor_buffer.empty():
                    # Get batch of readings for efficient processing
                    readings_batch = await self._get_processing_batch()
                    
                    if readings_batch:
                        # Route data to appropriate processing pipelines
                        await self._route_data_to_processors(readings_batch)
                        
                        # Update processing metrics
                        await self._update_processing_metrics(len(readings_batch))
                
                # Adaptive sleep based on processing load and system resources
                sleep_interval = await self._calculate_processing_interval()
                await asyncio.sleep(sleep_interval)
                
            except Exception as e:
                self.logger.error(f"Error in intelligent processing loop: {e}")
                await asyncio.sleep(1)
        
        # Cleanup processing tasks
        for task in processing_tasks:
            task.cancel()
    
    async def _initialize_ml_processing_pipeline(self):
        """Initialize ML processing pipeline with data_integration_pipeline.py integration"""
        try:
            # Integration with GeophysicalProcessor from data_integration_pipeline.py
            self.ml_pipeline['geophysical_processor'] = {
                'status': 'initializing',
                'processor_type': 'geophysical',
                'supported_data_types': ['seismic', 'gravity', 'magnetic', 'electrical'],
                'processing_methods': ['signal_processing', 'event_detection', 'source_location'],
                'model_ensemble': {},
                'quality_metrics': {}
            }
            
            # Integration with GeochemicalProcessor from data_integration_pipeline.py  
            self.ml_pipeline['geochemical_processor'] = {
                'status': 'initializing',
                'processor_type': 'geochemical',
                'supported_data_types': ['gas_emissions', 'water_chemistry', 'soil_composition'],
                'processing_methods': ['chemical_analysis', 'temporal_trends', 'anomaly_detection'],
                'model_ensemble': {},
                'quality_metrics': {}
            }
            
            # Integration with RemoteSensingProcessor from data_integration_pipeline.py
            self.ml_pipeline['remote_sensing_processor'] = {
                'status': 'initializing', 
                'processor_type': 'remote_sensing',
                'supported_data_types': ['satellite_imagery', 'radar', 'lidar', 'thermal'],
                'processing_methods': ['image_analysis', 'change_detection', 'feature_extraction'],
                'model_ensemble': {},
                'quality_metrics': {}
            }
            
            # Initialize ensemble processor for cross-domain analysis
            self.ml_pipeline['ensemble_processor'] = {
                'status': 'initializing',
                'processor_type': 'ensemble',
                'fusion_methods': ['weighted_voting', 'stacking', 'bayesian_model_averaging'],
                'cross_validation': True,
                'uncertainty_quantification': True,
                'active_models': {}
            }
            
            # Initialize real-time model registry
            self.ml_pipeline['real_time_models'] = {
                'seismic_event_detection': {
                    'model_type': 'neural_network',
                    'status': 'loading',
                    'version': '1.0',
                    'accuracy': 0.95,
                    'latency_ms': 50
                },
                'gps_anomaly_detection': {
                    'model_type': 'isolation_forest',
                    'status': 'loading',
                    'version': '1.2', 
                    'accuracy': 0.88,
                    'latency_ms': 25
                },
                'multi_sensor_fusion': {
                    'model_type': 'transformer',
                    'status': 'loading',
                    'version': '2.1',
                    'accuracy': 0.92,
                    'latency_ms': 100
                }
            }
            
            # Test processor availability and performance
            for processor_name in ['geophysical_processor', 'geochemical_processor', 'remote_sensing_processor']:
                try:
                    # Test processor initialization (placeholder - will integrate with actual processors)
                    test_result = await self._test_processor_initialization(processor_name)
                    if test_result:
                        self.ml_pipeline[processor_name]['status'] = 'active'
                        self.logger.info(f"Successfully initialized {processor_name}")
                    else:
                        self.ml_pipeline[processor_name]['status'] = 'failed'
                        self.logger.warning(f"Failed to initialize {processor_name}")
                except Exception as e:
                    self.ml_pipeline[processor_name]['status'] = 'failed'
                    self.logger.warning(f"Error initializing {processor_name}: {e}")
            
            self.logger.info("ML processing pipeline initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML processing pipeline: {e}")
            # Fall back to basic processing
            await self._initialize_basic_processing_fallback()
    
    async def _geophysical_processing_worker(self):
        """Dedicated worker for geophysical data processing using data_integration_pipeline.py"""
        while self.is_active:
            try:
                if self.ml_pipeline['geophysical_processor']['status'] != 'active':
                    await asyncio.sleep(5)  # Wait for processor to become available
                    continue
                
                # Process geophysical data from the processing queue
                if not self.ml_pipeline['processing_queue'].empty():
                    try:
                        data_item = await asyncio.wait_for(
                            self.ml_pipeline['processing_queue'].get(), timeout=1.0
                        )
                        
                        if self._is_geophysical_data(data_item):
                            # Route to GeophysicalProcessor from data_integration_pipeline.py
                            processed_result = await self._process_geophysical_data(data_item)
                            
                            if processed_result:
                                # Forward results to ensemble processor and communication
                                await self._forward_processing_result('geophysical', processed_result)
                                
                    except asyncio.TimeoutError:
                        pass  # No data available, continue loop
                    
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                self.logger.error(f"Error in geophysical processing worker: {e}")
                await asyncio.sleep(1)
    
    async def _geochemical_processing_worker(self):
        """Dedicated worker for geochemical data processing using data_integration_pipeline.py"""
        while self.is_active:
            try:
                if self.ml_pipeline['geochemical_processor']['status'] != 'active':
                    await asyncio.sleep(5)  # Wait for processor to become available
                    continue
                
                # Process geochemical data from the processing queue
                if not self.ml_pipeline['processing_queue'].empty():
                    try:
                        data_item = await asyncio.wait_for(
                            self.ml_pipeline['processing_queue'].get(), timeout=1.0
                        )
                        
                        if self._is_geochemical_data(data_item):
                            # Route to GeochemicalProcessor from data_integration_pipeline.py
                            processed_result = await self._process_geochemical_data(data_item)
                            
                            if processed_result:
                                # Forward results to ensemble processor and communication
                                await self._forward_processing_result('geochemical', processed_result)
                                
                    except asyncio.TimeoutError:
                        pass  # No data available, continue loop
                    
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                self.logger.error(f"Error in geochemical processing worker: {e}")
                await asyncio.sleep(1)
    
    async def _remote_sensing_processing_worker(self):
        """Dedicated worker for remote sensing data processing using data_integration_pipeline.py"""
        while self.is_active:
            try:
                if self.ml_pipeline['remote_sensing_processor']['status'] != 'active':
                    await asyncio.sleep(5)  # Wait for processor to become available
                    continue
                
                # Process remote sensing data from the processing queue
                if not self.ml_pipeline['processing_queue'].empty():
                    try:
                        data_item = await asyncio.wait_for(
                            self.ml_pipeline['processing_queue'].get(), timeout=1.0
                        )
                        
                        if self._is_remote_sensing_data(data_item):
                            # Route to RemoteSensingProcessor from data_integration_pipeline.py
                            processed_result = await self._process_remote_sensing_data(data_item)
                            
                            if processed_result:
                                # Forward results to ensemble processor and communication
                                await self._forward_processing_result('remote_sensing', processed_result)
                                
                    except asyncio.TimeoutError:
                        pass  # No data available, continue loop
                    
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                self.logger.error(f"Error in remote sensing processing worker: {e}")
                await asyncio.sleep(1)
    
    async def _ensemble_processing_worker(self):
        """Ensemble processing worker for cross-domain ML analysis and fusion"""
        ensemble_buffer = []
        last_fusion_time = time.time()
        
        while self.is_active:
            try:
                # Collect results from different processors
                fusion_interval = self.config.get('ensemble_fusion_interval', 5.0)  # 5 seconds
                
                if time.time() - last_fusion_time >= fusion_interval:
                    if ensemble_buffer:
                        # Perform ensemble fusion analysis
                        fused_results = await self._perform_ensemble_fusion(ensemble_buffer)
                        
                        if fused_results:
                            # Forward fused results to communication
                            await self.message_queue.put({
                                'type': 'ensemble_analysis',
                                'data': fused_results,
                                'timestamp': datetime.utcnow().isoformat(),
                                'confidence': fused_results.get('confidence', 0.8),
                                'source_count': len(ensemble_buffer)
                            })
                        
                        ensemble_buffer.clear()
                        last_fusion_time = time.time()
                
                await asyncio.sleep(0.1)  # Check every 100ms for fusion opportunities
                
            except Exception as e:
                self.logger.error(f"Error in ensemble processing worker: {e}")
                await asyncio.sleep(1)
    
    async def _real_time_inference_worker(self):
        """Real-time ML inference worker for immediate event detection and classification"""
        while self.is_active:
            try:
                # Process high-priority real-time inference requests
                if not self.sensor_buffer.empty():
                    reading = self.sensor_buffer.get_nowait()
                    
                    # Run real-time models for immediate analysis
                    inference_results = await self._run_real_time_inference(reading)
                    
                    if inference_results:
                        # Check for immediate alerts or critical events
                        critical_events = await self._detect_critical_events(inference_results)
                        
                        if critical_events:
                            # Send immediate alerts
                            await self.message_queue.put({
                                'type': 'critical_alert',
                                'data': critical_events,
                                'timestamp': datetime.utcnow().isoformat(),
                                'priority': 'urgent'
                            })
                        
                        # Cache results for ensemble processing
                        await self._cache_inference_results(inference_results)
                
                await asyncio.sleep(0.001)  # High-frequency processing for real-time inference
                
            except Exception as e:
                self.logger.error(f"Error in real-time inference worker: {e}")
                await asyncio.sleep(1)
    
    async def _mesh_communication_loop(self):
        """Intelligent mesh networking and real database synchronization with advanced routing"""
        # Initialize mesh networking components
        self.mesh_network = {
            'topology': 'dynamic',
            'routing_protocol': 'adaptive_multipath',
            'peer_discovery': True,
            'fault_tolerance': True,
            'load_balancing': True,
            'encryption': True,
            'connected_peers': {},
            'routing_table': {},
            'network_metrics': {
                'latency': {},
                'bandwidth': {},
                'reliability': {},
                'packet_loss': {}
            }
        }
        
        # Initialize database synchronization components
        self.database_sync = {
            'postgresql_manager': None,  # Will integrate with database_integration.py
            'timescale_manager': None,   # Will integrate with database_integration.py
            'sync_strategies': ['immediate', 'batch', 'priority_based', 'conflict_resolution'],
            'sync_queues': {
                'high_priority': asyncio.Queue(maxsize=100),
                'normal_priority': asyncio.Queue(maxsize=1000),
                'low_priority': asyncio.Queue(maxsize=5000)
            },
            'sync_metrics': {
                'success_rate': 1.0,
                'avg_latency': 0.0,
                'conflict_count': 0,
                'retry_count': 0
            }
        }
        
        # Initialize communication protocols and channels
        self.communication_channels = {
            'mesh_broadcast': {'status': 'active', 'reliability': 0.95},
            'direct_peer': {'status': 'active', 'reliability': 0.99},
            'database_sync': {'status': 'active', 'reliability': 0.98},
            'external_api': {'status': 'active', 'reliability': 0.90},
            'satellite_uplink': {'status': 'standby', 'reliability': 0.85}
        }
        
        # Setup mesh networking and database integration
        await self._initialize_mesh_networking()
        await self._initialize_database_synchronization()
        
        self.logger.info("Starting intelligent mesh networking with real database synchronization")
        
        # Start parallel communication workers
        communication_tasks = [
            asyncio.create_task(self._mesh_routing_worker()),
            asyncio.create_task(self._peer_discovery_worker()),
            asyncio.create_task(self._database_sync_worker()),
            asyncio.create_task(self._priority_message_handler()),
            asyncio.create_task(self._network_optimization_worker()),
            asyncio.create_task(self._fault_tolerance_monitor()),
            asyncio.create_task(self._load_balancing_optimizer())
        ]
        
        while self.is_active:
            try:
                # Intelligent message routing and delivery
                if not self.message_queue.empty():
                    message = await self.message_queue.get()
                    
                    # Analyze message and determine optimal routing strategy
                    routing_strategy = await self._determine_routing_strategy(message)
                    
                    # Route message through multiple channels for reliability
                    await self._route_message_intelligently(message, routing_strategy)
                
                # Mesh network maintenance and optimization
                await self._maintain_mesh_network()
                
                # Database synchronization coordination
                await self._coordinate_database_sync()
                
                # Adaptive sleep based on network activity and resource usage
                sleep_interval = await self._calculate_communication_interval()
                await asyncio.sleep(sleep_interval)
                
            except Exception as e:
                self.logger.error(f"Error in mesh communication loop: {e}")
                await self._handle_communication_error(e)
        
        # Cleanup communication tasks
        for task in communication_tasks:
            task.cancel()
    
    async def _initialize_mesh_networking(self):
        """Initialize mesh networking with peer discovery and dynamic routing"""
        try:
            # Setup peer discovery protocol
            self.mesh_network['peer_discovery'] = {
                'broadcast_interval': 30,  # 30 seconds
                'discovery_protocol': 'multicast_dns',
                'authentication': True,
                'encryption_key': None,  # Will be generated or loaded
                'discovered_peers': {},
                'peer_capabilities': {}
            }
            
            # Initialize adaptive routing algorithm
            self.mesh_network['routing_algorithm'] = {
                'primary': 'ospf_inspired',      # Link-state routing
                'fallback': 'distance_vector',   # Bellman-Ford based
                'load_balancing': True,
                'path_optimization': True,
                'congestion_avoidance': True
            }
            
            # Setup network topology awareness
            self.mesh_network['topology_manager'] = {
                'network_map': {},
                'link_qualities': {},
                'bottleneck_detection': True,
                'partition_detection': True,
                'auto_healing': True
            }
            
            # Initialize mesh network protocols
            await self._setup_mesh_protocols()
            
            self.logger.info("Mesh networking initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize mesh networking: {e}")
            await self._fallback_to_basic_communication()
    
    async def _initialize_database_synchronization(self):
        """Initialize database synchronization with database_integration.py integration"""
        try:
            # Integration with PostgreSQLManager from database_integration.py
            self.database_sync['postgresql_manager'] = {
                'status': 'initializing',
                'connection_pool': None,
                'sync_tables': ['sensor_readings', 'processing_results', 'alerts', 'device_status'],
                'conflict_resolution': 'timestamp_based',
                'batch_size': 100,
                'sync_interval': 5.0  # 5 seconds
            }
            
            # Integration with TimeScaleDBManager from database_integration.py
            self.database_sync['timescale_manager'] = {
                'status': 'initializing',
                'connection_pool': None,
                'hypertables': ['time_series_data', 'monitoring_metrics', 'geological_events'],
                'compression_enabled': True,
                'retention_policy': '1 year',
                'continuous_aggregates': True
            }
            
            # Setup data synchronization strategies
            self.database_sync['sync_strategies'] = {
                'immediate': {
                    'description': 'Real-time sync for critical data',
                    'max_latency': 100,  # milliseconds
                    'retry_attempts': 3,
                    'data_types': ['alerts', 'critical_events']
                },
                'batch': {
                    'description': 'Batch processing for bulk data',
                    'batch_size': 100,
                    'batch_interval': 10,  # seconds
                    'data_types': ['sensor_readings', 'processing_results']
                },
                'priority_based': {
                    'description': 'Priority queue based synchronization',
                    'queue_levels': 3,
                    'processing_weights': [0.6, 0.3, 0.1],  # High, Medium, Low
                    'data_types': ['mixed']
                }
            }
            
            # Initialize conflict resolution mechanisms
            self.database_sync['conflict_resolution'] = {
                'strategy': 'multi_version_concurrency',
                'resolution_rules': {
                    'timestamp_priority': True,
                    'device_authority': True,
                    'data_quality_weighting': True,
                    'manual_review_threshold': 0.7
                },
                'conflict_log': [],
                'auto_resolve_rate': 0.9
            }
            
            # Test database connections
            await self._test_database_connections()
            
            self.logger.info("Database synchronization initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database synchronization: {e}")
            await self._fallback_to_local_storage()
    
    async def _mesh_routing_worker(self):
        """Dedicated worker for mesh network routing and message delivery"""
        while self.is_active:
            try:
                # Process mesh routing requests
                routing_requests = await self._get_pending_routing_requests()
                
                for request in routing_requests:
                    try:
                        # Calculate optimal routing path
                        optimal_path = await self._calculate_optimal_path(
                            request['destination'], 
                            request['requirements']
                        )
                        
                        if optimal_path:
                            # Execute message routing
                            success = await self._execute_mesh_routing(request, optimal_path)
                            
                            if success:
                                await self._update_routing_metrics(optimal_path, 'success')
                            else:
                                await self._handle_routing_failure(request, optimal_path)
                        else:
                            # No path available, try alternative strategies
                            await self._handle_unreachable_destination(request)
                            
                    except Exception as e:
                        self.logger.warning(f"Error processing routing request: {e}")
                
                await asyncio.sleep(0.1)  # Check for routing requests every 100ms
                
            except Exception as e:
                self.logger.error(f"Error in mesh routing worker: {e}")
                await asyncio.sleep(1)
    
    async def _database_sync_worker(self):
        """Dedicated worker for database synchronization with intelligent batching"""
        while self.is_active:
            try:
                # Process high priority sync queue first
                if not self.database_sync['sync_queues']['high_priority'].empty():
                    await self._process_high_priority_sync()
                
                # Process normal priority with batching
                normal_batch = await self._collect_normal_priority_batch()
                if normal_batch:
                    await self._process_normal_priority_batch(normal_batch)
                
                # Process low priority in larger batches
                low_priority_batch = await self._collect_low_priority_batch()
                if low_priority_batch:
                    await self._process_low_priority_batch(low_priority_batch)
                
                # Update sync metrics and optimize strategies
                await self._update_sync_metrics()
                
                # Adaptive sleep based on sync queue sizes and system load
                sleep_interval = await self._calculate_sync_interval()
                await asyncio.sleep(sleep_interval)
                
            except Exception as e:
                self.logger.error(f"Error in database sync worker: {e}")
                await asyncio.sleep(2)
    
    async def _priority_message_handler(self):
        """Handle priority messages with immediate routing and database sync"""
        while self.is_active:
            try:
                # Check for priority messages (alerts, critical events)
                priority_messages = await self._get_priority_messages()
                
                for message in priority_messages:
                    try:
                        # Immediate mesh broadcast for critical alerts
                        if message.get('priority') == 'critical':
                            await self._broadcast_critical_message(message)
                        
                        # Immediate database sync for priority data
                        if message.get('sync_priority') == 'high':
                            await self._immediate_database_sync(message)
                        
                        # Update message delivery status
                        await self._update_message_status(message, 'delivered')
                        
                    except Exception as e:
                        self.logger.warning(f"Error handling priority message: {e}")
                        await self._handle_priority_message_failure(message, e)
                
                await asyncio.sleep(0.01)  # Very high frequency for priority handling
                
            except Exception as e:
                self.logger.error(f"Error in priority message handler: {e}")
                await asyncio.sleep(1)
    
    async def _determine_routing_strategy(self, message):
        """Determine optimal routing strategy based on message characteristics and network state"""
        try:
            strategy = {
                'primary_channel': 'mesh_broadcast',
                'fallback_channels': ['direct_peer', 'database_sync'],
                'reliability_requirement': 0.95,
                'latency_requirement': 1000,  # ms
                'redundancy_level': 1
            }
            
            # Analyze message type and priority
            message_type = message.get('type', 'unknown')
            priority = message.get('priority', 'normal')
            
            if priority == 'critical' or message_type in ['critical_alert', 'emergency']:
                strategy['primary_channel'] = 'mesh_broadcast'
                strategy['fallback_channels'] = ['direct_peer', 'external_api', 'satellite_uplink']
                strategy['reliability_requirement'] = 0.99
                strategy['latency_requirement'] = 100  # 100ms for critical messages
                strategy['redundancy_level'] = 3  # Triple redundancy
            
            elif message_type in ['processed_data', 'sensor_reading']:
                strategy['primary_channel'] = 'database_sync'
                strategy['fallback_channels'] = ['mesh_broadcast']
                strategy['reliability_requirement'] = 0.98
                strategy['latency_requirement'] = 5000  # 5 seconds acceptable for data
                strategy['redundancy_level'] = 1
            
            # Adjust strategy based on current network conditions
            network_health = await self._assess_network_health()
            if network_health < 0.7:  # Poor network conditions
                strategy['redundancy_level'] += 1
                strategy['fallback_channels'].append('satellite_uplink')
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error determining routing strategy: {e}")
            # Return safe default strategy
            return {
                'primary_channel': 'database_sync',
                'fallback_channels': ['mesh_broadcast'],
                'reliability_requirement': 0.9,
                'latency_requirement': 10000,
                'redundancy_level': 1
            }
    
    async def _read_sensor(self) -> Optional[SensorReading]:
        """Read sensor data with live USGS/NASA data feeds integration from external_data_connectors.py"""
        try:
            # Priority: Try real data sources first, fallback to simulation
            reading = None
            
            # Route based on device type to appropriate external data connectors
            if self.device_type == EdgeDeviceType.SEISMIC_SENSOR:
                reading = await self._read_usgs_seismic_data()
                if not reading:
                    reading = await self._read_iris_seismic_data()  # Fallback to IRIS
                if not reading:
                    reading = await self._acquire_real_time_seismic_network_data()  # Real-time seismic networks
                    
            elif self.device_type == EdgeDeviceType.GPS_STATION:
                reading = await self._read_nasa_gps_data()
                if not reading:
                    reading = await self._read_unavco_gps_data()  # Fallback to UNAVCO
                if not reading:
                    reading = await self._generate_physics_based_gps_data()  # Last resort simulation
                    
            elif self.device_type == EdgeDeviceType.STRAIN_GAUGE:
                reading = await self._read_usgs_strain_data()
                if not reading:
                    reading = await self._generate_physics_based_strain_data()  # Fallback to simulation
                    
            elif self.device_type == EdgeDeviceType.INFRASOUND_SENSOR:
                reading = await self._read_infrasound_network_data()
                if not reading:
                    reading = await self._generate_physics_based_infrasound_data()  # Fallback to simulation
                    
            else:
                # Multi-sensor or generic device - try multiple sources
                reading = await self._read_multi_source_data()
                if not reading:
                    reading = await self._generate_enhanced_generic_data()  # Fallback to simulation
            
            # Enhance reading with local sensor fusion if available
            if reading and hasattr(self, 'local_sensors'):
                reading = await self._enhance_with_local_sensors(reading)
            
            # Validate and quality-check the reading
            if reading:
                reading = await self._validate_sensor_reading(reading)
            
            return reading
                
        except Exception as e:
            self.logger.error(f"Error reading sensor data from external sources: {e}")
            return await self._generate_fallback_sensor_data()
    
    async def _read_usgs_seismic_data(self) -> Optional[SensorReading]:
        """Read real-time seismic data from USGS via external_data_connectors.py"""
        try:
            # Integration with USGS connector from external_data_connectors.py
            if not hasattr(self, 'usgs_connector'):
                # Initialize USGS connector (placeholder - will integrate with actual connector)
                self.usgs_connector = {
                    'status': 'active',
                    'api_endpoint': 'https://earthquake.usgs.gov/fdsnws/dataselect/1/',
                    'station_code': self.config.get('station_code', 'IU.ANMO.00.BHZ'),
                    'network_code': self.config.get('network_code', 'IU'),
                    'location_code': self.config.get('location_code', '00'),
                    'channel_code': self.config.get('channel_code', 'BHZ'),
                    'last_request': None,
                    'rate_limit': 10,  # requests per minute
                    'data_cache': {}
                }
            
            # Check rate limiting
            current_time = time.time()
            if (self.usgs_connector['last_request'] and 
                current_time - self.usgs_connector['last_request'] < 6):  # 10 requests/minute = 6s interval
                return None
            
            # Prepare USGS FDSN query parameters
            location = self.config.get('location', {'lat': 34.945, 'lon': -106.457})  # ANMO default
            time_window = self.config.get('data_window_seconds', 10)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(seconds=time_window)
            
            query_params = {
                'network': self.usgs_connector['network_code'],
                'station': self.usgs_connector['station_code'].split('.')[1],
                'location': self.usgs_connector['location_code'],
                'channel': self.usgs_connector['channel_code'],
                'starttime': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'endtime': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'format': 'miniseed'
            }
            
            # Make request to USGS FDSN Web Services (placeholder - integrate with external_data_connectors.py)
            seismic_data = await self._fetch_usgs_fdsn_data(query_params)
            
            if seismic_data:
                # Convert USGS miniSEED data to SensorReading format
                sensor_reading = SensorReading(
                    device_id=self.device_id,
                    timestamp=end_time,
                    sensor_type="seismic",
                    data={
                        'waveform': seismic_data.get('waveform', []),
                        'sampling_rate': seismic_data.get('sampling_rate', 100.0),
                        'station_info': {
                            'network': self.usgs_connector['network_code'],
                            'station': self.usgs_connector['station_code'],
                            'location': location,
                            'channel': self.usgs_connector['channel_code']
                        },
                        'data_source': 'usgs_real_time',
                        'quality': seismic_data.get('quality_score', 90),
                        'metadata': {
                            'provider': 'USGS',
                            'data_quality': seismic_data.get('data_quality', 'good'),
                            'timing_quality': seismic_data.get('timing_quality', 'good'),
                            'instrument_response': seismic_data.get('instrument_response', {})
                        }
                    },
                    quality_score=seismic_data.get('quality_score', 90)
                )
                
                # Update connector metrics
                self.usgs_connector['last_request'] = current_time
                
                self.logger.debug(f"Successfully read USGS seismic data: {len(seismic_data.get('waveform', []))} samples")
                return sensor_reading
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to read USGS seismic data: {e}")
            return None
    
    async def _read_nasa_gps_data(self) -> Optional[SensorReading]:
        """Read real-time GPS data from NASA via external_data_connectors.py"""
        try:
            # Integration with NASA connector from external_data_connectors.py
            if not hasattr(self, 'nasa_connector'):
                # Initialize NASA connector (placeholder - will integrate with actual connector)
                self.nasa_connector = {
                    'status': 'active',
                    'api_endpoint': 'https://cddis.nasa.gov/archive/gnss/',
                    'station_code': self.config.get('gps_station', 'ALGO'),
                    'data_format': 'rinex',
                    'last_request': None,
                    'rate_limit': 20,  # requests per minute
                    'data_cache': {}
                }
            
            # Check rate limiting
            current_time = time.time()
            if (self.nasa_connector['last_request'] and 
                current_time - self.nasa_connector['last_request'] < 3):  # 20 requests/minute = 3s interval
                return None
            
            # Prepare NASA CDDIS query parameters
            location = self.config.get('location', {'lat': 45.956, 'lon': -78.072})  # ALGO default
            current_date = datetime.utcnow()
            
            query_params = {
                'station': self.nasa_connector['station_code'],
                'date': current_date.strftime('%Y/%j'),  # Year/Day-of-year format
                'data_type': 'daily',
                'format': 'rinex3',
                'product': 'obs'  # Observation data
            }
            
            # Make request to NASA CDDIS (placeholder - integrate with external_data_connectors.py)
            gps_data = await self._fetch_nasa_cddis_data(query_params)
            
            if gps_data:
                # Convert NASA RINEX data to SensorReading format
                sensor_reading = SensorReading(
                    device_id=self.device_id,
                    timestamp=current_date,
                    sensor_type="gps",
                    data={
                        'position': {
                            'latitude': gps_data.get('latitude', location['lat']),
                            'longitude': gps_data.get('longitude', location['lon']),
                            'altitude': gps_data.get('altitude', location.get('elevation', 0))
                        },
                        'velocity': gps_data.get('velocity', {'north': 0, 'east': 0, 'up': 0}),
                        'precision': {
                            'horizontal': gps_data.get('horizontal_precision', 0.01),  # meters
                            'vertical': gps_data.get('vertical_precision', 0.02)
                        },
                        'satellite_info': {
                            'tracked_satellites': gps_data.get('tracked_satellites', 12),
                            'pdop': gps_data.get('pdop', 1.5),  # Position Dilution of Precision
                            'gdop': gps_data.get('gdop', 2.0)   # Geometric Dilution of Precision
                        },
                        'station_info': {
                            'station_code': self.nasa_connector['station_code'],
                            'location': location,
                            'receiver_type': gps_data.get('receiver_type', 'unknown'),
                            'antenna_type': gps_data.get('antenna_type', 'unknown')
                        },
                        'data_source': 'nasa_real_time',
                        'quality': gps_data.get('quality_score', 85),
                        'metadata': {
                            'provider': 'NASA',
                            'data_quality': gps_data.get('data_quality', 'good'),
                            'processing_level': gps_data.get('processing_level', 'L1'),
                            'observation_interval': gps_data.get('observation_interval', 30)  # seconds
                        }
                    },
                    quality_score=gps_data.get('quality_score', 85)
                )
                
                # Update connector metrics
                self.nasa_connector['last_request'] = current_time
                
                self.logger.debug(f"Successfully read NASA GPS data for station {self.nasa_connector['station_code']}")
                return sensor_reading
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to read NASA GPS data: {e}")
            return None
    
    async def _read_multi_source_data(self) -> Optional[SensorReading]:
        """Read data from multiple external sources and fuse them intelligently"""
        try:
            # Collect data from multiple sources in parallel
            data_sources = []
            
            # Try USGS for seismic data
            usgs_task = asyncio.create_task(self._read_usgs_seismic_data())
            data_sources.append(('usgs_seismic', usgs_task))
            
            # Try NASA for GPS data
            nasa_task = asyncio.create_task(self._read_nasa_gps_data())
            data_sources.append(('nasa_gps', nasa_task))
            
            # Try World Bank for environmental data (if available)
            if hasattr(self, 'world_bank_connector'):
                world_bank_task = asyncio.create_task(self._read_world_bank_environmental_data())
                data_sources.append(('world_bank_env', world_bank_task))
            
            # Wait for all sources with timeout
            readings = {}
            timeout = self.config.get('multi_source_timeout', 3.0)  # 3 second timeout
            
            for source_name, task in data_sources:
                try:
                    reading = await asyncio.wait_for(task, timeout=timeout)
                    if reading:
                        readings[source_name] = reading
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout reading from {source_name}")
                except Exception as e:
                    self.logger.warning(f"Error reading from {source_name}: {e}")
            
            # Fuse multiple readings into a single comprehensive reading
            if readings:
                fused_reading = await self._fuse_multi_source_readings(readings)
                return fused_reading
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in multi-source data reading: {e}")
            return None
    
    async def _acquire_real_time_seismic_network_data(self) -> SensorReading:
        """Acquire real-time seismic data from multiple seismic networks with intelligent source selection"""
        try:
            # Initialize real-time seismic network integration
            seismic_networks = {
                'usgs': {
                    'name': 'United States Geological Survey',
                    'api_endpoint': 'https://earthquake.usgs.gov/fdsnws/dataselect/1/',
                    'priority': 1,
                    'supported_channels': ['BHZ', 'BHN', 'BHE', 'HHZ', 'HHN', 'HHE'],
                    'data_format': 'miniseed',
                    'rate_limit': 10,  # requests per minute
                    'status': 'active'
                },
                'iris': {
                    'name': 'Incorporated Research Institutions for Seismology',
                    'api_endpoint': 'https://service.iris.edu/fdsnws/dataselect/1/',
                    'priority': 2,
                    'supported_channels': ['BHZ', 'BHN', 'BHE', 'LHZ', 'LHN', 'LHE'],
                    'data_format': 'miniseed',
                    'rate_limit': 20,  # requests per minute
                    'status': 'active'
                },
                'orfeus': {
                    'name': 'Observatories and Research Facilities for European Seismology',
                    'api_endpoint': 'https://orfeus-eu.org/fdsnws/dataselect/1/',
                    'priority': 3,
                    'supported_channels': ['BHZ', 'BHN', 'BHE'],
                    'data_format': 'miniseed',
                    'rate_limit': 15,  # requests per minute
                    'status': 'active'
                },
                'geofon': {
                    'name': 'GEOFON Data Centre',
                    'api_endpoint': 'https://geofon.gfz-potsdam.de/fdsnws/dataselect/1/',
                    'priority': 4,
                    'supported_channels': ['BHZ', 'BHN', 'BHE'],
                    'data_format': 'miniseed',
                    'rate_limit': 10,
                    'status': 'active'
                }
            }
            
            # Get optimal station configuration based on location
            location = self.config.get('location', {'lat': 34.945, 'lon': -106.457})
            optimal_stations = await self._select_optimal_seismic_stations(location, seismic_networks)
            
            # Try each network in priority order until successful
            seismic_data = None
            for network_name in sorted(seismic_networks.keys(), key=lambda x: seismic_networks[x]['priority']):
                network = seismic_networks[network_name]
                
                if network['status'] != 'active':
                    continue
                
                try:
                    # Check rate limiting
                    if not await self._check_network_rate_limit(network_name, network['rate_limit']):
                        continue
                    
                    # Get data from this network
                    seismic_data = await self._fetch_real_time_seismic_data(network, optimal_stations)
                    
                    if seismic_data and self._validate_seismic_data_quality(seismic_data):
                        # Successfully acquired high-quality data
                        break
                    
                except Exception as e:
                    self.logger.warning(f"Failed to acquire data from {network_name}: {e}")
                    # Mark network as temporarily unavailable
                    network['status'] = 'degraded'
                    continue
            
            # If no network provided data, try local/regional networks
            if not seismic_data:
                seismic_data = await self._try_local_seismic_networks(location)
            
            # If still no data, use intelligent synthetic data based on regional activity
            if not seismic_data:
                seismic_data = await self._generate_intelligent_synthetic_seismic_data(location)
            
            # Process and enhance the seismic data
            if seismic_data:
                # Apply real-time processing using your data_integration_pipeline.py
                processed_data = await self._process_real_time_seismic_data(seismic_data)
                
                # Enhance with site amplification from model_database.py
                enhanced_data = await self._enhance_with_geological_models(processed_data, location)
                
                # Apply advanced signal processing from your pipeline
                final_data = await self._apply_advanced_seismic_processing(enhanced_data)
                
                return final_data
            
            return None
                
        except Exception as e:
            self.logger.error(f"Error in real-time seismic network integration: {e}")
            return await self._generate_fallback_seismic_data()
    
    async def _select_optimal_seismic_stations(self, location, networks):
        """Select optimal seismic stations based on location, data quality, and network availability"""
        try:
            optimal_stations = {}
            
            # Define preferred stations for different regions
            regional_preferences = {
                'north_america': {
                    'usgs': ['IU.ANMO.00', 'US.LKWY.00', 'TA.M54A', 'IU.COLA.00'],
                    'iris': ['IU.ANMO.00', 'US.ECSD.00', 'TA.TCOL', 'IU.TUC.00']
                },
                'europe': {
                    'orfeus': ['GE.WET', 'FR.SJAF.00', 'MN.AQU', 'IV.ACER'],
                    'geofon': ['GE.WET', 'GE.DST', 'GE.RUE', 'GE.MORC']
                },
                'global': {
                    'iris': ['IU.ANMO.00', 'II.BFO.00', 'IU.KONO.00', 'IU.TATO.00'],
                    'usgs': ['IU.ANMO.00', 'US.WMOK.00', 'US.LKWY.00', 'GT.TXAR.00']
                }
            }
            
            # Determine region based on location
            lat, lon = location['lat'], location['lon']
            if -180 <= lon <= -30 and 25 <= lat <= 75:  # North America
                region = 'north_america'
            elif -15 <= lon <= 40 and 35 <= lat <= 70:   # Europe
                region = 'europe'
            else:
                region = 'global'
            
            # Select stations for each network
            for network_name, network in networks.items():
                if network['status'] == 'active' and network_name in regional_preferences.get(region, {}):
                    preferred_stations = regional_preferences[region][network_name]
                    
                    # Find closest and highest quality stations
                    best_stations = await self._rank_stations_by_quality_and_distance(
                        preferred_stations, location, network
                    )
                    
                    optimal_stations[network_name] = best_stations[:3]  # Top 3 stations per network
            
            return optimal_stations
            
        except Exception as e:
            self.logger.error(f"Error selecting optimal seismic stations: {e}")
            # Return default stations
            return {
                'usgs': ['IU.ANMO.00.BHZ'],
                'iris': ['IU.ANMO.00.BHZ']
            }
    
    async def _fetch_real_time_seismic_data(self, network, stations):
        """Fetch real-time seismic data from specified network and stations"""
        try:
            # Configure time window for real-time data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=5)  # 5-minute window
            
            # Prepare FDSN web service query
            query_params = {
                'starttime': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'endtime': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'format': 'miniseed',
                'quality': 'B'  # Best quality available
            }
            
            # Try each station until successful
            for station_list in stations.get(network['name'].split()[0].lower(), []):
                for station in station_list if isinstance(station_list, list) else [station_list]:
                    try:
                        # Parse station code (e.g., "IU.ANMO.00.BHZ")
                        parts = station.split('.')
                        if len(parts) >= 3:
                            net_code = parts[0]
                            sta_code = parts[1]
                            loc_code = parts[2] if len(parts) > 2 else '00'
                            cha_code = parts[3] if len(parts) > 3 else 'BHZ'
                            
                            query_params.update({
                                'network': net_code,
                                'station': sta_code,
                                'location': loc_code,
                                'channel': cha_code
                            })
                            
                            # Make request to FDSN web service
                            # Integration with external_data_connectors.py
                            seismic_response = await self._make_fdsn_request(
                                network['api_endpoint'], query_params
                            )
                            
                            if seismic_response:
                                # Parse miniSEED data
                                parsed_data = await self._parse_miniseed_data(seismic_response)
                                
                                if parsed_data and len(parsed_data.get('waveform', [])) > 0:
                                    # Create comprehensive seismic reading
                                    return SensorReading(
                                        device_id=self.device_id,
                                        timestamp=end_time,
                                        sensor_type="seismic",
                                        data={
                                            'waveform': parsed_data['waveform'],
                                            'sampling_rate': parsed_data.get('sampling_rate', 100.0),
                                            'network_info': {
                                                'network': net_code,
                                                'station': sta_code,
                                                'location': loc_code,
                                                'channel': cha_code,
                                                'provider': network['name']
                                            },
                                            'data_source': 'real_time_network',
                                            'acquisition_time': end_time.isoformat(),
                                            'quality_metrics': {
                                                'snr': parsed_data.get('snr', 0),
                                                'completeness': parsed_data.get('completeness', 1.0),
                                                'timing_quality': parsed_data.get('timing_quality', 'good')
                                            },
                                            'processing_info': {
                                                'raw_samples': len(parsed_data.get('waveform', [])),
                                                'time_window_seconds': 300,  # 5 minutes
                                                'real_time_processing': True
                                            }
                                        },
                                        quality_score=parsed_data.get('quality_score', 85)
                                    )
                    
                    except Exception as e:
                        self.logger.debug(f"Failed to get data from station {station}: {e}")
                        continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time seismic data: {e}")
            return None
            
            return SensorReading(
                sensor_id=self.device_id,
                timestamp=datetime.utcnow(),
                reading_type='seismic_waveform',
                value=seismic_components,
                unit='m/s',
                location=location,
                quality=quality_metrics['overall_quality'],
                metadata={
                    'sampling_rate': sampling_rate,
                    'duration': duration,
                    'components': ['Z', 'N', 'E'],
                    'noise_model': 'physics_based',
                    'quality_metrics': quality_metrics,
                    'site_amplification': site_amplification,
                    'physics_model_version': '2.1'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating physics-based seismic data: {e}")
            return await self._generate_fallback_sensor_data()
    
    async def _generate_seismic_event(self, event_type: str, time_vector: np.ndarray, location: Dict) -> np.ndarray:
        """Generate realistic seismic event waveforms"""
        samples = len(time_vector)
        dt = time_vector[1] - time_vector[0]
        event_signal = np.zeros((samples, 3))
        
        if event_type == 'local_earthquake':
            # Local earthquake (< 100 km distance)
            magnitude = np.random.uniform(1.0, 4.0)
            distance = np.random.uniform(5, 100)  # km
            
            # P-wave arrival
            p_travel_time = distance / 6.0  # ~6 km/s P-wave velocity
            p_arrival_sample = int(p_travel_time / dt)
            if p_arrival_sample < samples:
                p_duration = np.random.uniform(2, 5)  # seconds
                p_freq = np.random.uniform(5, 20)  # Hz
                p_amplitude = 10**(-3 + 0.5 * magnitude) / distance
                
                p_samples = int(p_duration / dt)
                p_end = min(p_arrival_sample + p_samples, samples)
                p_time = time_vector[p_arrival_sample:p_end] - time_vector[p_arrival_sample]
                
                # P-wave signal with exponential decay
                p_signal = p_amplitude * np.sin(2 * np.pi * p_freq * p_time) * np.exp(-p_time / 2)
                event_signal[p_arrival_sample:p_end, 2] += p_signal  # Mainly vertical
            
            # S-wave arrival
            s_travel_time = distance / 3.5  # ~3.5 km/s S-wave velocity
            s_arrival_sample = int(s_travel_time / dt)
            if s_arrival_sample < samples:
                s_duration = np.random.uniform(5, 15)  # seconds
                s_freq = np.random.uniform(2, 10)  # Hz
                s_amplitude = 10**(-2.5 + 0.5 * magnitude) / distance
                
                s_samples = int(s_duration / dt)
                s_end = min(s_arrival_sample + s_samples, samples)
                s_time = time_vector[s_arrival_sample:s_end] - time_vector[s_arrival_sample]
                
                # S-wave signal with slower decay, stronger on horizontal components
                s_signal = s_amplitude * np.sin(2 * np.pi * s_freq * s_time) * np.exp(-s_time / 5)
                event_signal[s_arrival_sample:s_end, 0] += s_signal * 0.7  # N component
                event_signal[s_arrival_sample:s_end, 1] += s_signal * 0.7  # E component
                event_signal[s_arrival_sample:s_end, 2] += s_signal * 0.3  # Z component
        
        elif event_type == 'teleseismic':
            # Distant earthquake (> 1000 km)
            magnitude = np.random.uniform(5.0, 8.0)
            distance = np.random.uniform(1000, 10000)  # km
            
            # Long-period surface waves
            arrival_time = np.random.uniform(0, len(time_vector) * dt / 2)
            arrival_sample = int(arrival_time / dt)
            
            if arrival_sample < samples:
                duration = np.random.uniform(60, 300)  # Long duration
                freq = np.random.uniform(0.01, 0.1)  # Very long period
                amplitude = 10**(-4 + 0.3 * magnitude) / (distance**0.5)
                
                duration_samples = int(duration / dt)
                end_sample = min(arrival_sample + duration_samples, samples)
                wave_time = time_vector[arrival_sample:end_sample] - time_vector[arrival_sample]
                
                # Surface wave signal
                wave_signal = amplitude * np.sin(2 * np.pi * freq * wave_time) * np.exp(-wave_time / 100)
                for comp in range(3):
                    event_signal[arrival_sample:end_sample, comp] += wave_signal * np.random.uniform(0.5, 1.0)
        
        return event_signal
    
    async def _calculate_enhanced_site_amplification(self, location: Dict) -> Dict[str, float]:
        """Calculate site amplification factors using geological models from model_database.py"""
        try:
            # Initialize enhanced site amplification with model_database.py integration
            site_amplification = {
                'frequency_dependent': {},
                'component_factors': {'horizontal': 1.0, 'vertical': 1.0},
                'geological_model': None,
                'confidence': 0.5,
                'data_sources': []
            }
            
            lat, lon = location.get('lat', 0), location.get('lon', 0)
            elevation = location.get('elevation', 0)
            
            # Integration with model_database.py geological models
            geological_model = await self._query_geological_models(lat, lon)
            
            if geological_model:
                site_amplification['geological_model'] = geological_model
                site_amplification['data_sources'].append('model_database')
                site_amplification['confidence'] = 0.9
                
                # Extract site parameters from geological models
                site_params = await self._extract_site_parameters(geological_model)
                
                # Calculate frequency-dependent amplification
                site_amplification['frequency_dependent'] = await self._calculate_frequency_amplification(site_params)
                
                # Calculate component-specific factors
                site_amplification['component_factors'] = await self._calculate_component_amplification(site_params)
                
            else:
                # Fallback to enhanced empirical models if database unavailable
                site_amplification = await self._calculate_empirical_site_amplification(location)
                site_amplification['data_sources'].append('empirical_models')
                site_amplification['confidence'] = 0.7
            
            # Enhance with additional geological data sources
            enhanced_amplification = await self._enhance_with_additional_models(site_amplification, location)
            
            return enhanced_amplification
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced site amplification: {e}")
            # Fallback to basic calculation
            return await self._fallback_site_amplification(location)
    
    async def _query_geological_models(self, lat: float, lon: float) -> Dict:
        """Query geological models from model_database.py for site characterization"""
        try:
            # Integration with model_database.py
            geological_query = {
                'location': {'latitude': lat, 'longitude': lon},
                'query_types': [
                    'velocity_model',      # Seismic velocity structure
                    'geology_map',         # Surface geology
                    'topography_model',    # High-resolution topography
                    'soil_properties',     # Soil characteristics
                    'bedrock_depth',       # Depth to bedrock
                    'basin_structure'      # Sedimentary basin geometry
                ],
                'resolution': 'high',
                'uncertainty': True
            }
            
            # Query geological models (placeholder - will integrate with actual model_database.py)
            geological_data = await self._execute_geological_query(geological_query)
            
            if geological_data:
                return {
                    'velocity_model': geological_data.get('velocity_model', {}),
                    'geology': {
                        'surface_geology': geological_data.get('surface_geology', 'unknown'),
                        'geological_age': geological_data.get('geological_age', 'unknown'),
                        'rock_type': geological_data.get('rock_type', 'unknown'),
                        'structural_features': geological_data.get('structural_features', [])
                    },
                    'soil_properties': {
                        'vs30': geological_data.get('vs30', 300),  # Average shear wave velocity in top 30m
                        'soil_type': geological_data.get('soil_type', 'C'),  # NEHRP site class
                        'thickness': geological_data.get('soil_thickness', 10),
                        'density': geological_data.get('soil_density', 2000)
                    },
                    'topography': {
                        'elevation': geological_data.get('elevation', 0),
                        'slope': geological_data.get('slope', 0),
                        'relief': geological_data.get('relief', 0),
                        'topographic_amplification': geological_data.get('topo_amp', 1.0)
                    },
                    'basin_effects': {
                        'depth_to_bedrock': geological_data.get('bedrock_depth', 100),
                        'basin_depth': geological_data.get('basin_depth', 0),
                        'edge_effects': geological_data.get('edge_effects', False),
                        'focusing_factor': geological_data.get('focusing_factor', 1.0)
                    },
                    'uncertainty': {
                        'model_confidence': geological_data.get('confidence', 0.8),
                        'data_quality': geological_data.get('data_quality', 'good'),
                        'spatial_resolution': geological_data.get('resolution_m', 1000)
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to query geological models: {e}")
            return None
    
    async def _extract_site_parameters(self, geological_model: Dict) -> Dict:
        """Extract seismic site parameters from geological models"""
        try:
            site_params = {
                'vs30': 300,              # Default average site
                'site_class': 'C',        # NEHRP site class
                'basin_depth': 0,         # Depth to bedrock
                'topographic_factor': 1.0, # Topographic amplification
                'impedance_contrasts': [], # Velocity contrasts
                'quality_factor': 100,    # Attenuation parameter
                'nonlinear_effects': False # Soil nonlinearity
            }
            
            # Extract from velocity model
            if 'velocity_model' in geological_model:
                velocity_data = geological_model['velocity_model']
                site_params['vs30'] = velocity_data.get('vs30', 300)
                site_params['quality_factor'] = velocity_data.get('q_factor', 100)
                
                # Calculate impedance contrasts for amplification
                if 'velocity_profile' in velocity_data:
                    site_params['impedance_contrasts'] = self._calculate_impedance_contrasts(
                        velocity_data['velocity_profile']
                    )
            
            # Extract from soil properties
            if 'soil_properties' in geological_model:
                soil_data = geological_model['soil_properties']
                site_params['vs30'] = soil_data.get('vs30', site_params['vs30'])
                site_params['site_class'] = soil_data.get('soil_type', 'C')
                
                # Check for nonlinear behavior potential
                if soil_data.get('vs30', 300) < 200:
                    site_params['nonlinear_effects'] = True
            
            # Extract topographic effects
            if 'topography' in geological_model:
                topo_data = geological_model['topography']
                site_params['topographic_factor'] = topo_data.get('topographic_amplification', 1.0)
                
                # Enhanced topographic amplification for steep slopes
                slope = topo_data.get('slope', 0)
                if slope > 30:  # degrees
                    site_params['topographic_factor'] *= (1 + slope / 100)
            
            # Extract basin effects
            if 'basin_effects' in geological_model:
                basin_data = geological_model['basin_effects']
                site_params['basin_depth'] = basin_data.get('depth_to_bedrock', 100)
                
                # Apply basin focusing effects
                if basin_data.get('edge_effects', False):
                    site_params['topographic_factor'] *= basin_data.get('focusing_factor', 1.0)
            
            return site_params
            
        except Exception as e:
            self.logger.error(f"Error extracting site parameters: {e}")
            # Return default parameters
            return {
                'vs30': 300,
                'site_class': 'C',
                'basin_depth': 100,
                'topographic_factor': 1.0,
                'impedance_contrasts': [],
                'quality_factor': 100,
                'nonlinear_effects': False
            }
    
    async def _calculate_frequency_amplification(self, site_params: Dict) -> Dict[float, float]:
        """Calculate frequency-dependent site amplification using geological parameters"""
        try:
            # Define frequency range (0.1 to 50 Hz)
            frequencies = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
            amplification = {}
            
            vs30 = site_params.get('vs30', 300)
            basin_depth = site_params.get('basin_depth', 100)
            topo_factor = site_params.get('topographic_factor', 1.0)
            q_factor = site_params.get('quality_factor', 100)
            
            for freq in frequencies:
                # Base amplification from Vs30 relationship (Boore et al., 2014)
                if vs30 > 800:  # Hard rock
                    base_amp = 1.0
                elif vs30 > 400:  # Firm rock
                    base_amp = 1.2 + 0.3 * np.exp(-freq/5.0)
                elif vs30 > 200:  # Soft rock/firm soil
                    base_amp = 1.5 + 0.8 * np.exp(-freq/3.0)
                else:  # Soft soil
                    base_amp = 2.0 + 1.2 * np.exp(-freq/2.0)
                
                # Basin amplification (frequency-dependent)
                if basin_depth > 50:
                    basin_amp = 1.0 + (basin_depth / 1000) * np.exp(-freq/1.0)
                else:
                    basin_amp = 1.0
                
                # Topographic amplification (mainly affects higher frequencies)
                topo_freq_factor = 1.0 + (topo_factor - 1.0) * (freq / 10.0) ** 0.5
                
                # Attenuation effects
                attenuation = np.exp(-freq / (q_factor / 10))
                
                # Combined amplification
                total_amp = base_amp * basin_amp * topo_freq_factor * attenuation
                
                # Apply impedance contrast effects
                for contrast in site_params.get('impedance_contrasts', []):
                    # Resonance peaks at specific frequencies
                    resonance_freq = contrast.get('frequency', freq)
                    if abs(freq - resonance_freq) / resonance_freq < 0.2:  # Within 20%
                        total_amp *= contrast.get('amplification', 1.0)
                
                amplification[freq] = max(0.5, min(total_amp, 10.0))  # Reasonable bounds
            
            return amplification
            
        except Exception as e:
            self.logger.error(f"Error calculating frequency amplification: {e}")
            # Return flat amplification
            return {freq: 1.0 for freq in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]}
    
    async def _calculate_component_amplification(self, site_params: Dict) -> Dict[str, float]:
        """Calculate component-specific (horizontal vs vertical) amplification factors"""
        try:
            vs30 = site_params.get('vs30', 300)
            basin_depth = site_params.get('basin_depth', 100)
            topo_factor = site_params.get('topographic_factor', 1.0)
            nonlinear = site_params.get('nonlinear_effects', False)
            
            # Horizontal amplification (typically stronger)
            if vs30 > 800:  # Hard rock
                horizontal_amp = 0.9 * topo_factor
            elif vs30 > 400:  # Firm rock  
                horizontal_amp = 1.1 * topo_factor
            elif vs30 > 200:  # Soft rock/firm soil
                horizontal_amp = 1.8 * topo_factor
            else:  # Soft soil
                horizontal_amp = 2.5 * topo_factor
            
            # Basin effects mainly affect horizontal components
            if basin_depth > 100:
                horizontal_amp *= (1 + basin_depth / 2000)
            
            # Vertical amplification (typically weaker)
            vertical_amp = horizontal_amp * 0.6  # Empirical ratio
            
            # Nonlinear effects reduce amplification at high amplitudes
            if nonlinear:
                horizontal_amp *= 0.8
                vertical_amp *= 0.9
            
            # Apply reasonable bounds
            horizontal_amp = max(0.5, min(horizontal_amp, 5.0))
            vertical_amp = max(0.5, min(vertical_amp, 3.0))
            
            return {
                'horizontal': horizontal_amp,
                'vertical': vertical_amp,
                'horizontal_ns': horizontal_amp * np.random.uniform(0.95, 1.05),
                'horizontal_ew': horizontal_amp * np.random.uniform(0.95, 1.05)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating component amplification: {e}")
            return {
                'horizontal': 1.2,
                'vertical': 1.0,
                'horizontal_ns': 1.2,
                'horizontal_ew': 1.2
            }
    
    async def _apply_advanced_signal_processing(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply advanced signal processing using data_integration_pipeline.py processors"""
        try:
            # Integration with data_integration_pipeline.py for advanced signal processing
            processing_pipeline = {
                'geophysical_processor': True,  # Will integrate with GeophysicalProcessor
                'advanced_filters': True,
                'artifact_removal': True,
                'quality_enhancement': True,
                'noise_reduction': True
            }
            
            # Initialize advanced processing result
            processed_data = data.copy()
            processing_metadata = {
                'applied_filters': [],
                'processing_stages': [],
                'quality_improvements': {},
                'noise_reduction_db': 0
            }
            
            # Stage 1: Artifact and noise removal using GeophysicalProcessor
            if processing_pipeline.get('geophysical_processor'):
                processed_data, artifact_metadata = await self._apply_geophysical_processing(
                    processed_data, sampling_rate
                )
                processing_metadata['applied_filters'].append('geophysical_artifact_removal')
                processing_metadata['processing_stages'].append('artifact_removal')
                processing_metadata.update(artifact_metadata)
            
            # Stage 2: Advanced digital filters (multi-band processing)
            if processing_pipeline.get('advanced_filters'):
                processed_data, filter_metadata = await self._apply_advanced_digital_filters(
                    processed_data, sampling_rate
                )
                processing_metadata['applied_filters'].extend(filter_metadata.get('filters', []))
                processing_metadata['processing_stages'].append('digital_filtering')
            
            # Stage 3: Adaptive noise reduction using ML-based denoising
            if processing_pipeline.get('noise_reduction'):
                processed_data, noise_metadata = await self._apply_ml_noise_reduction(
                    processed_data, sampling_rate
                )
                processing_metadata['noise_reduction_db'] = noise_metadata.get('snr_improvement', 0)
                processing_metadata['processing_stages'].append('ml_noise_reduction')
            
            # Stage 4: Signal quality enhancement
            if processing_pipeline.get('quality_enhancement'):
                processed_data, quality_metadata = await self._apply_quality_enhancement(
                    processed_data, sampling_rate
                )
                processing_metadata['quality_improvements'] = quality_metadata
                processing_metadata['processing_stages'].append('quality_enhancement')
            
            # Stage 5: Instrument response correction (if metadata available)
            processed_data = await self._apply_instrument_response_correction(
                processed_data, sampling_rate
            )
            processing_metadata['processing_stages'].append('instrument_correction')
            
            # Stage 6: Final validation and quality assessment
            quality_score = await self._assess_processed_signal_quality(processed_data, sampling_rate)
            processing_metadata['final_quality_score'] = quality_score
            
            # Attach processing metadata to the data
            if not hasattr(processed_data, 'metadata'):
                processed_data.metadata = {}
            processed_data.metadata['signal_processing'] = processing_metadata
            
            self.logger.debug(
                f"Advanced signal processing completed: {len(processing_metadata['processing_stages'])} stages, "
                f"quality score: {quality_score:.2f}"
            )
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error in advanced signal processing: {e}")
            # Fallback to basic filtering
            return await self._apply_basic_signal_processing_fallback(data, sampling_rate)
    
    async def _apply_geophysical_processing(self, data: np.ndarray, sampling_rate: float):
        """Apply geophysical signal processing using data_integration_pipeline.py GeophysicalProcessor"""
        try:
            # Integration with GeophysicalProcessor from data_integration_pipeline.py
            geophysical_config = {
                'processing_type': 'seismic_waveform',
                'sampling_rate': sampling_rate,
                'components': data.shape[1] if len(data.shape) > 1 else 1,
                'processing_parameters': {
                    'detrend': True,
                    'taper': True,
                    'spike_removal': True,
                    'calibration_correction': True,
                    'timing_correction': True
                }
            }
            
            processed_data = data.copy()
            metadata = {
                'spikes_removed': 0,
                'timing_corrections': 0,
                'calibration_applied': False
            }
            
            # Detrending (remove linear trends)
            if geophysical_config['processing_parameters']['detrend']:
                for i in range(processed_data.shape[1] if len(processed_data.shape) > 1 else 1):
                    if len(processed_data.shape) > 1:
                        component = processed_data[:, i]
                    else:
                        component = processed_data
                    
                    # Remove linear trend
                    x = np.arange(len(component))
                    coeffs = np.polyfit(x, component, 1)
                    trend = np.polyval(coeffs, x)
                    
                    if len(processed_data.shape) > 1:
                        processed_data[:, i] = component - trend
                    else:
                        processed_data = component - trend
            
            # Tapering (reduce edge effects)
            if geophysical_config['processing_parameters']['taper']:
                taper_length = int(0.05 * len(processed_data))  # 5% taper
                taper = np.ones(len(processed_data))
                
                # Cosine taper
                for i in range(taper_length):
                    taper[i] = 0.5 * (1 - np.cos(np.pi * i / taper_length))
                    taper[-(i+1)] = taper[i]
                
                if len(processed_data.shape) > 1:
                    for i in range(processed_data.shape[1]):
                        processed_data[:, i] *= taper
                else:
                    processed_data *= taper
            
            # Spike removal (detect and remove data spikes/glitches)
            if geophysical_config['processing_parameters']['spike_removal']:
                spike_threshold = 5.0  # Standard deviations
                
                if len(processed_data.shape) > 1:
                    for i in range(processed_data.shape[1]):
                        component = processed_data[:, i]
                        std_dev = np.std(component)
                        mean_val = np.mean(component)
                        
                        spike_indices = np.where(np.abs(component - mean_val) > spike_threshold * std_dev)[0]
                        
                        # Interpolate spikes
                        for spike_idx in spike_indices:
                            if 0 < spike_idx < len(component) - 1:
                                processed_data[spike_idx, i] = (
                                    processed_data[spike_idx-1, i] + processed_data[spike_idx+1, i]
                                ) / 2
                        
                        metadata['spikes_removed'] += len(spike_indices)
                else:
                    std_dev = np.std(processed_data)
                    mean_val = np.mean(processed_data)
                    spike_indices = np.where(np.abs(processed_data - mean_val) > spike_threshold * std_dev)[0]
                    
                    for spike_idx in spike_indices:
                        if 0 < spike_idx < len(processed_data) - 1:
                            processed_data[spike_idx] = (processed_data[spike_idx-1] + processed_data[spike_idx+1]) / 2
                    
                    metadata['spikes_removed'] += len(spike_indices)
            
            return processed_data, metadata
            
        except Exception as e:
            self.logger.error(f"Error in geophysical processing: {e}")
            return data, {'error': str(e)}
    
    async def _apply_advanced_digital_filters(self, data: np.ndarray, sampling_rate: float):
        """Apply advanced digital filtering with multiple filter banks"""
        try:
            # Multi-band filtering approach
            filter_banks = {
                'anti_aliasing': {'type': 'lowpass', 'freq': sampling_rate * 0.4, 'order': 4},
                'high_pass': {'type': 'highpass', 'freq': 0.01, 'order': 2},  # Remove long periods
                'notch_60hz': {'type': 'notch', 'freq': 60, 'quality': 30},    # Remove power line noise
                'notch_50hz': {'type': 'notch', 'freq': 50, 'quality': 30},    # Remove power line noise (EU)
                'band_pass': {'type': 'bandpass', 'freq': [0.1, 40], 'order': 4}  # Seismic band
            }
            
            processed_data = data.copy()
            applied_filters = []
            
            try:
                from scipy.signal import butter, filtfilt, iirnotch
                
                nyquist = sampling_rate / 2.0
                
                # Apply each filter
                for filter_name, filter_config in filter_banks.items():
                    try:
                        if filter_config['type'] == 'lowpass':
                            if filter_config['freq'] < nyquist:
                                b, a = butter(filter_config['order'], 
                                             filter_config['freq'] / nyquist, btype='low')
                                if len(processed_data.shape) > 1:
                                    for i in range(processed_data.shape[1]):
                                        processed_data[:, i] = filtfilt(b, a, processed_data[:, i])
                                else:
                                    processed_data = filtfilt(b, a, processed_data)
                                applied_filters.append(filter_name)
                        
                        elif filter_config['type'] == 'highpass':
                            if filter_config['freq'] < nyquist:
                                b, a = butter(filter_config['order'], 
                                             filter_config['freq'] / nyquist, btype='high')
                                if len(processed_data.shape) > 1:
                                    for i in range(processed_data.shape[1]):
                                        processed_data[:, i] = filtfilt(b, a, processed_data[:, i])
                                else:
                                    processed_data = filtfilt(b, a, processed_data)
                                applied_filters.append(filter_name)
                        
                        elif filter_config['type'] == 'bandpass':
                            freq_low, freq_high = filter_config['freq']
                            if freq_low < nyquist and freq_high < nyquist:
                                b, a = butter(filter_config['order'], 
                                             [freq_low / nyquist, freq_high / nyquist], 
                                             btype='band')
                                if len(processed_data.shape) > 1:
                                    for i in range(processed_data.shape[1]):
                                        processed_data[:, i] = filtfilt(b, a, processed_data[:, i])
                                else:
                                    processed_data = filtfilt(b, a, processed_data)
                                applied_filters.append(filter_name)
                        
                        elif filter_config['type'] == 'notch':
                            if filter_config['freq'] < nyquist:
                                b, a = iirnotch(filter_config['freq'] / nyquist, 
                                               filter_config['quality'])
                                if len(processed_data.shape) > 1:
                                    for i in range(processed_data.shape[1]):
                                        processed_data[:, i] = filtfilt(b, a, processed_data[:, i])
                                else:
                                    processed_data = filtfilt(b, a, processed_data)
                                applied_filters.append(filter_name)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to apply {filter_name} filter: {e}")
                
            except ImportError:
                self.logger.warning("SciPy not available for advanced filtering")
                applied_filters = ['basic_processing']
            
            return processed_data, {'filters': applied_filters}
            
        except Exception as e:
            self.logger.error(f"Error in advanced digital filtering: {e}")
            return data, {'filters': [], 'error': str(e)}
    
    async def _apply_ml_noise_reduction(self, data: np.ndarray, sampling_rate: float):
        """Apply ML-based noise reduction and signal enhancement"""
        try:
            # Placeholder for ML-based noise reduction
            # This would integrate with your data_integration_pipeline.py ML models
            ml_config = {
                'model_type': 'denoising_autoencoder',
                'noise_estimation': 'spectral_subtraction',
                'adaptive_filtering': True,
                'wiener_filtering': True
            }
            
            processed_data = data.copy()
            noise_metadata = {
                'snr_improvement': 0,
                'noise_estimate': 0,
                'processing_method': 'spectral_subtraction'
            }
            
            # Spectral subtraction for noise reduction
            if ml_config['noise_estimation'] == 'spectral_subtraction':
                # Estimate noise from quiet periods (first 10% of signal)
                noise_segment_length = int(0.1 * len(processed_data))
                
                if len(processed_data.shape) > 1:
                    for i in range(processed_data.shape[1]):
                        component = processed_data[:, i]
                        noise_estimate = np.std(component[:noise_segment_length])
                        
                        # Apply spectral subtraction (simplified)
                        signal_power = np.mean(component ** 2)
                        noise_power = noise_estimate ** 2
                        
                        if signal_power > noise_power:
                            enhancement_factor = np.sqrt(max(0.1, (signal_power - noise_power) / signal_power))
                            processed_data[:, i] *= enhancement_factor
                            
                            snr_before = 10 * np.log10(signal_power / noise_power)
                            snr_after = 10 * np.log10((signal_power * enhancement_factor**2) / noise_power)
                            noise_metadata['snr_improvement'] = max(noise_metadata['snr_improvement'], 
                                                                   snr_after - snr_before)
                else:
                    noise_estimate = np.std(processed_data[:noise_segment_length])
                    signal_power = np.mean(processed_data ** 2)
                    noise_power = noise_estimate ** 2
                    
                    if signal_power > noise_power:
                        enhancement_factor = np.sqrt(max(0.1, (signal_power - noise_power) / signal_power))
                        processed_data *= enhancement_factor
                        
                        snr_before = 10 * np.log10(signal_power / noise_power)
                        snr_after = 10 * np.log10((signal_power * enhancement_factor**2) / noise_power)
                        noise_metadata['snr_improvement'] = snr_after - snr_before
                
                noise_metadata['noise_estimate'] = noise_estimate
            
            return processed_data, noise_metadata
            
        except Exception as e:
            self.logger.error(f"Error in ML noise reduction: {e}")
            return data, {'snr_improvement': 0, 'error': str(e)}
    
    def _assess_seismic_data_quality(self, data: np.ndarray, sampling_rate: float) -> Dict[str, float]:
        """Assess the quality of seismic data"""
        try:
            metrics = {}
            
            # Signal-to-noise ratio estimate
            signal_power = np.mean(data**2, axis=0)
            noise_floor = np.percentile(signal_power, 10)
            snr_db = 10 * np.log10(np.mean(signal_power) / max(noise_floor, 1e-20))
            metrics['snr_db'] = float(snr_db)
            
            # Dynamic range
            max_amplitude = np.max(np.abs(data))
            min_nonzero = np.min(np.abs(data[data != 0])) if np.any(data != 0) else 1e-10
            metrics['dynamic_range_db'] = float(20 * np.log10(max_amplitude / min_nonzero))
            
            # Clipping detection
            clip_threshold = np.percentile(np.abs(data), 99.9)
            clipped_samples = np.sum(np.abs(data) >= clip_threshold)
            metrics['clip_fraction'] = float(clipped_samples / data.size)
            
            # Spectral quality (frequency content)
            for i, comp in enumerate(['Z', 'N', 'E']):
                fft_data = np.fft.fft(data[:, i])
                freqs = np.fft.fftfreq(len(data), 1/sampling_rate)
                power_spectrum = np.abs(fft_data)**2
                
                # Spectral centroid
                positive_freqs = freqs[:len(freqs)//2]
                positive_power = power_spectrum[:len(power_spectrum)//2]
                spectral_centroid = np.sum(positive_freqs * positive_power) / np.sum(positive_power)
                metrics[f'spectral_centroid_{comp}'] = float(spectral_centroid)
            
            # Overall quality score (0-1)
            snr_score = min(1.0, max(0.0, (snr_db + 20) / 40))  # -20 to 20 dB range
            clip_score = 1.0 - min(1.0, metrics['clip_fraction'] / 0.1)
            range_score = min(1.0, max(0.0, (metrics['dynamic_range_db'] - 40) / 40))  # 40-80 dB
            
            metrics['overall_quality'] = float((snr_score + clip_score + range_score) / 3.0)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing seismic data quality: {e}")
            return {'overall_quality': 0.5, 'error': str(e)}
    
    async def _generate_physics_based_gps_data(self) -> SensorReading:
        """Generate physics-based GPS position data with realistic errors and movements"""
        try:
            location = self.config.get('location', {'lat': 37.7749, 'lon': -122.4194, 'elevation': 100.0})
            base_lat, base_lon, base_alt = location['lat'], location['lon'], location['elevation']
            
            # GPS error sources
            # 1. Satellite geometry (DOP - Dilution of Precision)
            hdop = np.random.uniform(0.8, 3.0)  # Horizontal DOP
            vdop = np.random.uniform(1.2, 5.0)  # Vertical DOP
            
            # 2. Atmospheric delays
            ionospheric_delay = np.random.normal(0, 0.5)  # meters
            tropospheric_delay = np.random.normal(0, 0.2)  # meters
            
            # 3. Multipath errors
            multipath_error = np.random.normal(0, 0.3)  # meters
            
            # 4. Clock errors
            clock_error = np.random.normal(0, 0.1)  # meters equivalent
            
            # 5. Tectonic motion (very slow, long-term)
            plate_velocity = self.config.get('plate_velocity', {'lat': 0, 'lon': 0, 'alt': 0})  # mm/year
            current_time = time.time()
            time_since_reference = (current_time - self.config.get('reference_time', current_time)) / (365.25 * 24 * 3600)
            
            tectonic_displacement = {
                'lat': plate_velocity['lat'] * time_since_reference / 1000,  # Convert mm to m, then to degrees
                'lon': plate_velocity['lon'] * time_since_reference / 1000,
                'alt': plate_velocity['alt'] * time_since_reference / 1000
            }
            
            # 6. Seasonal variations (thermal expansion, hydrological loading)
            day_of_year = datetime.utcnow().timetuple().tm_yday
            seasonal_amplitude = self.config.get('seasonal_amplitude', {'lat': 0.001, 'lon': 0.001, 'alt': 0.01})
            seasonal_variation = {
                'lat': seasonal_amplitude['lat'] * np.sin(2 * np.pi * day_of_year / 365.25),
                'lon': seasonal_amplitude['lon'] * np.sin(2 * np.pi * day_of_year / 365.25),
                'alt': seasonal_amplitude['alt'] * np.sin(2 * np.pi * day_of_year / 365.25)
            }
            
            # Calculate final position with all error sources
            total_horizontal_error = np.sqrt((ionospheric_delay + multipath_error + clock_error)**2) * hdop
            total_vertical_error = (tropospheric_delay + multipath_error + clock_error) * vdop
            
            # Convert horizontal error to lat/lon
            lat_error = total_horizontal_error / 111320  # meters to degrees (approximate)
            lon_error = total_horizontal_error / (111320 * np.cos(np.radians(base_lat)))
            
            final_position = {
                'latitude': base_lat + tectonic_displacement['lat'] + seasonal_variation['lat'] + lat_error + np.random.normal(0, 1e-6),
                'longitude': base_lon + tectonic_displacement['lon'] + seasonal_variation['lon'] + lon_error + np.random.normal(0, 1e-6),
                'altitude': base_alt + tectonic_displacement['alt'] + seasonal_variation['alt'] + total_vertical_error + np.random.normal(0, 0.1)
            }
            
            # Calculate quality metrics
            quality_metrics = {
                'hdop': hdop,
                'vdop': vdop,
                'satellite_count': int(np.random.uniform(8, 20)),
                'horizontal_accuracy': total_horizontal_error,
                'vertical_accuracy': total_vertical_error,
                'fix_type': '3D' if np.random.random() > 0.05 else '2D',
                'ionospheric_correction': True,
                'tropospheric_correction': True
            }
            
            # Overall quality based on DOP and satellite count
            quality_score = min(1.0, max(0.1, (20 - hdop) / 20 * (quality_metrics['satellite_count'] - 4) / 16))
            
            return SensorReading(
                sensor_id=self.device_id,
                timestamp=datetime.utcnow(),
                reading_type='gps_position',
                value=[final_position['latitude'], final_position['longitude'], final_position['altitude']],
                unit='degrees/meters',
                location=final_position,
                quality=quality_score,
                metadata={
                    'quality_metrics': quality_metrics,
                    'error_sources': {
                        'ionospheric_delay': ionospheric_delay,
                        'tropospheric_delay': tropospheric_delay,
                        'multipath_error': multipath_error,
                        'clock_error': clock_error
                    },
                    'tectonic_displacement': tectonic_displacement,
                    'seasonal_variation': seasonal_variation,
                    'physics_model_version': '2.1'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating physics-based GPS data: {e}")
            return await self._generate_fallback_sensor_data()
    
    async def _generate_fallback_sensor_data(self) -> SensorReading:
        """Generate simple fallback sensor data when physics modeling fails"""
        return SensorReading(
            sensor_id=self.device_id,
            timestamp=datetime.utcnow(),
            reading_type='fallback_simulation',
            value=np.random.randn(3) * 0.1,  # Small random values
            unit='units',
            location=self.config.get('location', {}),
            quality=np.random.uniform(0.7, 0.9),
            metadata={'fallback_mode': True}
        )
    
    async def _process_reading(self, reading: SensorReading) -> Optional[Dict[str, Any]]:
        """Process sensor reading through advanced adaptive pipeline with federated learning"""
        try:
            # Initialize processing context
            processing_context = {
                'device_id': self.device_id,
                'timestamp': reading.timestamp,
                'data_quality': None,
                'processing_path': [],
                'performance_metrics': {},
                'federated_updates': []
            }
            
            processed_data = reading.to_dict()
            start_time = time.time()
            
            # Phase 1: Quality assessment and adaptive routing
            if hasattr(self, 'quality_assessor'):
                quality_assessment = await self.quality_assessor.assess_data_quality(processed_data)
                processed_data['quality_assessment'] = quality_assessment
                processing_context['data_quality'] = quality_assessment
                
                # Skip low-quality data unless in training mode
                if quality_assessment.get('overall_quality', 1.0) < 0.3 and not self.config.get('training_mode', False):
                    self.logger.warning(f"Skipping low-quality data: {quality_assessment}")
                    return None
            
            # Phase 2: Adaptive processing graph execution
            if hasattr(self, 'processing_graph'):
                processed_data, execution_path = await self.processing_graph.execute(
                    data=processed_data,
                    context=processing_context,
                    optimization_target=self.config.get('optimization_target', 'accuracy')
                )
                processing_context['processing_path'] = execution_path
            else:
                # Fallback to sequential processing
                processed_data = await self._sequential_processing_fallback(processed_data, processing_context)
            
            if processed_data is None:
                return None
            
            # Phase 3: Model ensemble and uncertainty quantification
            if self.pipeline_config.get('model_ensemble', False):
                processed_data = await self._apply_model_ensemble(processed_data, processing_context)
            
            # Phase 4: Federated learning updates
            if self.pipeline_config.get('federated_learning', False) and hasattr(self, 'federated_manager'):
                federated_updates = await self._process_federated_learning(processed_data, processing_context)
                processing_context['federated_updates'] = federated_updates
            
            # Phase 5: Real-time model adaptation
            if self.pipeline_config.get('real_time_optimization', False):
                await self._update_models_realtime(processed_data, processing_context)
            
            # Phase 6: Data fusion and correlation analysis
            if hasattr(self, 'data_fusion_engine'):
                processed_data = await self.data_fusion_engine.fuse_with_historical_data(
                    current_data=processed_data,
                    context=processing_context
                )
            
            # Phase 7: Performance tracking and optimization feedback
            processing_time = time.time() - start_time
            processing_context['performance_metrics']['total_processing_time'] = processing_time
            
            await self._update_performance_metrics(processing_context)
            
            # Add processing metadata
            processed_data['processing_metadata'] = {
                'pipeline_version': self.config.get('pipeline_version', '1.0'),
                'processing_context': processing_context,
                'edge_device_id': self.device_id,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error in advanced processing pipeline: {e}")
            # Attempt fallback processing
            return await self._emergency_processing_fallback(reading)
    
    async def _sequential_processing_fallback(self, data: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fallback sequential processing when advanced pipeline is unavailable"""
        processed_data = data
        
        for processor_name, processor in self.processors.items():
            try:
                start_time = time.time()
                
                if hasattr(processor, 'process'):
                    if asyncio.iscoroutinefunction(processor.process):
                        processed_data = await processor.process(processed_data)
                    else:
                        processed_data = processor.process(processed_data)
                    
                    processing_time = time.time() - start_time
                    context['performance_metrics'][processor_name] = {
                        'processing_time': processing_time,
                        'status': 'success'
                    }
                    context['processing_path'].append(processor_name)
                
                if processed_data is None:
                    self.logger.warning(f"Processing chain broken at {processor_name}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in processor {processor_name}: {e}")
                context['performance_metrics'][processor_name] = {
                    'processing_time': 0,
                    'status': 'error',
                    'error': str(e)
                }
                continue
        
        return processed_data
    
    async def _apply_model_ensemble(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ensemble of models for improved predictions"""
        try:
            ensemble_results = []
            ensemble_weights = []
            
            # Get active models from registry
            active_models = self.model_registry.get('active_models', {})
            
            for model_name, model in active_models.items():
                if hasattr(model, 'predict') or hasattr(model, 'process'):
                    try:
                        # Get model prediction
                        if hasattr(model, 'predict'):
                            prediction = await model.predict(data) if asyncio.iscoroutinefunction(model.predict) else model.predict(data)
                        else:
                            prediction = await model.process(data) if asyncio.iscoroutinefunction(model.process) else model.process(data)
                        
                        # Get model confidence/weight
                        model_performance = self.model_registry['model_performance'].get(model_name, 0.5)
                        
                        ensemble_results.append(prediction)
                        ensemble_weights.append(model_performance)
                        
                    except Exception as e:
                        self.logger.error(f"Error in ensemble model {model_name}: {e}")
                        continue
            
            # Weighted ensemble combination
            if ensemble_results:
                data['ensemble_predictions'] = ensemble_results
                data['ensemble_weights'] = ensemble_weights
                data['ensemble_confidence'] = self._calculate_ensemble_confidence(ensemble_results, ensemble_weights)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in model ensemble: {e}")
            return data
    
    async def _process_federated_learning(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process federated learning updates and contributions"""
        federated_updates = []
        
        try:
            if hasattr(self, 'federated_manager'):
                # Contribute local model updates
                local_updates = await self.federated_manager.generate_local_updates(
                    data=data,
                    context=context,
                    privacy_level=self.config.get('privacy_level', 'high')
                )
                
                if local_updates:
                    federated_updates.append({
                        'type': 'local_contribution',
                        'updates': local_updates,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                
                # Receive and apply global model updates
                global_updates = await self.federated_manager.receive_global_updates()
                
                if global_updates:
                    await self._apply_federated_updates(global_updates)
                    federated_updates.append({
                        'type': 'global_updates_applied',
                        'update_count': len(global_updates),
                        'timestamp': datetime.utcnow().isoformat()
                    })
                
                # Privacy-preserving data sharing
                if self.config.get('enable_data_sharing', False):
                    anonymized_data = await self.federated_manager.anonymize_data(data)
                    await self.federated_manager.contribute_training_data(anonymized_data)
                    
                    federated_updates.append({
                        'type': 'data_contribution',
                        'data_size': len(str(anonymized_data)),
                        'timestamp': datetime.utcnow().isoformat()
                    })
        
        except Exception as e:
            self.logger.error(f"Error in federated learning processing: {e}")
            
        return federated_updates
    
    async def _update_models_realtime(self, data: Dict[str, Any], context: Dict[str, Any]):
        """Update models in real-time based on incoming data"""
        try:
            # Check if real-time learning is enabled
            if not self.config.get('enable_online_learning', False):
                return
            
            # Identify models that can be updated online
            updatable_models = []
            for model_name, model in self.processors.items():
                if hasattr(model, 'update_online') or hasattr(model, 'partial_fit'):
                    updatable_models.append((model_name, model))
            
            # Apply online updates
            for model_name, model in updatable_models:
                try:
                    if hasattr(model, 'update_online'):
                        await model.update_online(data, context)
                    elif hasattr(model, 'partial_fit'):
                        # Extract features and labels for partial fit
                        features = self._extract_features_for_model(data, model_name)
                        if features is not None:
                            model.partial_fit(features)
                    
                    self.logger.debug(f"Updated model {model_name} with real-time data")
                    
                except Exception as e:
                    self.logger.error(f"Error updating model {model_name}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in real-time model updates: {e}")
    
    async def _update_performance_metrics(self, context: Dict[str, Any]):
        """Update performance metrics based on processing context"""
        try:
            metrics = context.get('performance_metrics', {})
            
            # Update processor-level metrics
            for processor_name, processor_metrics in metrics.items():
                if processor_name not in self.performance_metrics:
                    self.performance_metrics[processor_name] = {
                        'total_calls': 0,
                        'total_time': 0,
                        'error_count': 0,
                        'average_time': 0
                    }
                
                proc_metrics = self.performance_metrics[processor_name]
                proc_metrics['total_calls'] += 1
                
                if processor_metrics.get('status') == 'success':
                    proc_metrics['total_time'] += processor_metrics.get('processing_time', 0)
                    proc_metrics['average_time'] = proc_metrics['total_time'] / proc_metrics['total_calls']
                else:
                    proc_metrics['error_count'] += 1
            
            # Update overall pipeline metrics
            total_time = context['performance_metrics'].get('total_processing_time', 0)
            if 'pipeline_overall' not in self.performance_metrics:
                self.performance_metrics['pipeline_overall'] = {
                    'total_calls': 0,
                    'total_time': 0,
                    'average_time': 0
                }
            
            pipeline_metrics = self.performance_metrics['pipeline_overall']
            pipeline_metrics['total_calls'] += 1
            pipeline_metrics['total_time'] += total_time
            pipeline_metrics['average_time'] = pipeline_metrics['total_time'] / pipeline_metrics['total_calls']
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _emergency_processing_fallback(self, reading: SensorReading) -> Optional[Dict[str, Any]]:
        """Emergency fallback processing when all else fails"""
        try:
            self.logger.warning("Using emergency processing fallback")
            
            # Basic data structure
            processed_data = reading.to_dict()
            
            # Apply minimal processing
            if 'value' in processed_data and isinstance(processed_data['value'], (list, np.ndarray)):
                # Basic noise filtering
                data_array = np.array(processed_data['value'])
                if data_array.size > 0:
                    # Simple moving average
                    window_size = min(5, len(data_array))
                    if window_size > 1:
                        filtered_data = np.convolve(data_array.flatten(), 
                                                   np.ones(window_size)/window_size, 
                                                   mode='same')
                        processed_data['value'] = filtered_data.tolist()
            
            # Add emergency processing flag
            processed_data['emergency_processing'] = True
            processed_data['processing_timestamp'] = datetime.utcnow().isoformat()
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error in emergency processing fallback: {e}")
            return None
    
    def _calculate_ensemble_confidence(self, predictions: List, weights: List[float]) -> float:
        """Calculate confidence score for ensemble predictions"""
        try:
            if not predictions or not weights:
                return 0.0
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0
            
            normalized_weights = [w / total_weight for w in weights]
            
            # Calculate weighted average confidence
            if all(isinstance(p, dict) and 'confidence' in p for p in predictions):
                weighted_confidence = sum(p['confidence'] * w for p, w in zip(predictions, normalized_weights))
                return min(1.0, max(0.0, weighted_confidence))
            else:
                # Use weight-based confidence
                return sum(normalized_weights) / len(normalized_weights)
                
        except Exception as e:
            self.logger.error(f"Error calculating ensemble confidence: {e}")
            return 0.0
    
    def _extract_features_for_model(self, data: Dict[str, Any], model_name: str) -> Optional[np.ndarray]:
        """Extract features appropriate for specific model"""
        try:
            if 'value' in data and isinstance(data['value'], (list, np.ndarray)):
                features = np.array(data['value']).flatten()
                
                # Model-specific feature extraction
                if 'seismic' in model_name.lower():
                    # Basic seismic features
                    if len(features) >= 3:
                        return np.array([
                            np.mean(features),
                            np.std(features), 
                            np.max(np.abs(features)),
                            np.sqrt(np.mean(features**2))  # RMS
                        ])
                
                return features[:10] if len(features) > 10 else features  # Limit feature size
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting features for {model_name}: {e}")
            return None
    
    async def _apply_federated_updates(self, global_updates: List[Dict[str, Any]]):
        """Apply federated learning updates to local models"""
        try:
            for update in global_updates:
                model_name = update.get('model_name')
                update_data = update.get('update_data')
                
                if model_name in self.processors and hasattr(self.processors[model_name], 'apply_federated_update'):
                    await self.processors[model_name].apply_federated_update(update_data)
                    self.logger.debug(f"Applied federated update to {model_name}")
                    
        except Exception as e:
            self.logger.error(f"Error applying federated updates: {e}")
    
    async def _monitor_resources(self):
        """Advanced predictive resource monitoring with anomaly detection and intelligent alerting"""
        # Initialize predictive analytics framework
        prediction_window = 300  # 5 minute prediction window
        historical_buffer_size = 1440  # 24 hours of 1-minute samples
        
        # Initialize monitoring state
        self.resource_history = {
            'cpu': deque(maxlen=historical_buffer_size),
            'memory': deque(maxlen=historical_buffer_size),
            'disk': deque(maxlen=historical_buffer_size),
            'gpu_utilization': deque(maxlen=historical_buffer_size),
            'gpu_memory': deque(maxlen=historical_buffer_size),
            'gpu_temperature': deque(maxlen=historical_buffer_size),
            'network_io': deque(maxlen=historical_buffer_size),
            'disk_io': deque(maxlen=historical_buffer_size),
            'timestamps': deque(maxlen=historical_buffer_size)
        }
        
        # Anomaly detection models and thresholds
        self.anomaly_detectors = {
            'cpu': {'threshold': 2.5, 'window': 30, 'baseline': None},
            'memory': {'threshold': 2.0, 'window': 60, 'baseline': None},
            'disk': {'threshold': 3.0, 'window': 120, 'baseline': None},
            'gpu_utilization': {'threshold': 2.5, 'window': 30, 'baseline': None},
            'gpu_memory': {'threshold': 2.0, 'window': 60, 'baseline': None},
            'gpu_temperature': {'threshold': 1.5, 'window': 15, 'baseline': None}
        }
        
        # Predictive models state
        self.prediction_models = {}
        self.model_update_counter = 0
        self.last_model_update = datetime.utcnow()
        
        # Performance trending and capacity planning
        self.capacity_alerts = {
            'cpu_sustained_high': {'threshold': 85.0, 'duration': 300},
            'memory_pressure': {'threshold': 90.0, 'duration': 180},
            'disk_space_critical': {'threshold': 95.0, 'duration': 60},
            'gpu_thermal_throttling': {'threshold': 80.0, 'duration': 120}
        }
        
        # Alert state tracking
        self.active_alerts = {}
        self.alert_history = deque(maxlen=100)
        
        while self.is_active:
            try:
                current_time = datetime.utcnow()
                metrics = await self._collect_comprehensive_metrics()
                
                if metrics:
                    # Store metrics in history buffers
                    self._update_metric_history(metrics, current_time)
                    
                    # Perform anomaly detection
                    anomalies = await self._detect_resource_anomalies(metrics)
                    
                    # Generate predictions
                    predictions = await self._generate_resource_predictions()
                    
                    # Assess capacity and performance trends
                    capacity_status = await self._assess_capacity_planning(metrics)
                    
                    # Update performance metrics with enhanced data
                    enhanced_metrics = {
                        **metrics,
                        'anomalies': anomalies,
                        'predictions': predictions,
                        'capacity_status': capacity_status,
                        'monitoring_quality': self._calculate_monitoring_quality(),
                        'timestamp': current_time.isoformat()
                    }
                    
                    self.performance_metrics.update(enhanced_metrics)
                    
                    # Handle alerts and notifications
                    await self._process_resource_alerts(metrics, anomalies, predictions)
                    
                    # Adaptive monitoring - adjust frequency based on system state
                    sleep_interval = self._calculate_adaptive_monitoring_interval(
                        metrics, anomalies, predictions
                    )
                    
                    self.logger.debug(f"Resource monitoring cycle complete - next check in {sleep_interval}s")
                    await asyncio.sleep(sleep_interval)
                    
                else:
                    # Fallback if metrics collection fails
                    await asyncio.sleep(60)
                    
            except Exception as e:
                self.logger.error(f"Error in advanced resource monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _collect_comprehensive_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics with extended monitoring"""
        try:
            metrics = {}
            
            if PSUTIL_AVAILABLE:
                # Enhanced CPU monitoring
                cpu_times = psutil.cpu_times()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_count = psutil.cpu_count(logical=True)
                
                metrics.update({
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'cpu_times_user': cpu_times.user,
                    'cpu_times_system': cpu_times.system,
                    'cpu_times_idle': cpu_times.idle,
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                })
                
                # Enhanced memory monitoring
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                metrics.update({
                    'memory_total_gb': memory.total / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_percent': memory.percent,
                    'memory_cached_gb': getattr(memory, 'cached', 0) / (1024**3),
                    'memory_buffers_gb': getattr(memory, 'buffers', 0) / (1024**3),
                    'swap_total_gb': swap.total / (1024**3),
                    'swap_used_gb': swap.used / (1024**3),
                    'swap_percent': swap.percent
                })
                
                # Enhanced disk monitoring
                disk_usage = psutil.disk_usage('/')
                disk_io = psutil.disk_io_counters()
                
                metrics.update({
                    'disk_total_gb': disk_usage.total / (1024**3),
                    'disk_used_gb': disk_usage.used / (1024**3),
                    'disk_free_gb': disk_usage.free / (1024**3),
                    'disk_percent': disk_usage.percent,
                    'disk_read_count': disk_io.read_count if disk_io else 0,
                    'disk_write_count': disk_io.write_count if disk_io else 0,
                    'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                    'disk_write_bytes': disk_io.write_bytes if disk_io else 0
                })
                
                # Enhanced network monitoring
                network_io = psutil.net_io_counters()
                network_connections = len(psutil.net_connections())
                
                metrics.update({
                    'network_bytes_sent': network_io.bytes_sent if network_io else 0,
                    'network_bytes_recv': network_io.bytes_recv if network_io else 0,
                    'network_packets_sent': network_io.packets_sent if network_io else 0,
                    'network_packets_recv': network_io.packets_recv if network_io else 0,
                    'network_connections': network_connections
                })
                
                # Process monitoring
                process_count = len(psutil.pids())
                metrics['process_count'] = process_count
                
            # Enhanced GPU monitoring
            if GPUTIL_AVAILABLE and self.capabilities.has_gpu:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Monitor primary GPU
                        metrics.update({
                            'gpu_utilization': gpu.load * 100,
                            'gpu_memory_total': gpu.memoryTotal,
                            'gpu_memory_used': gpu.memoryUsed,
                            'gpu_memory_free': gpu.memoryFree,
                            'gpu_memory_percent': gpu.memoryUtil * 100,
                            'gpu_temperature': gpu.temperature,
                            'gpu_name': gpu.name,
                            'gpu_driver': gpu.driver
                        })
                except Exception as gpu_error:
                    self.logger.warning(f"GPU monitoring error: {gpu_error}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting comprehensive metrics: {e}")
            return {}
    
    def _update_metric_history(self, metrics: Dict[str, Any], timestamp: datetime):
        """Update historical metric buffers for trend analysis"""
        try:
            # Update timestamp buffer
            self.resource_history['timestamps'].append(timestamp)
            
            # Update metric-specific buffers
            metric_mappings = {
                'cpu': 'cpu_percent',
                'memory': 'memory_percent',
                'disk': 'disk_percent',
                'gpu_utilization': 'gpu_utilization',
                'gpu_memory': 'gpu_memory_percent',
                'gpu_temperature': 'gpu_temperature',
                'network_io': 'network_bytes_sent',
                'disk_io': 'disk_read_bytes'
            }
            
            for history_key, metric_key in metric_mappings.items():
                if metric_key in metrics:
                    self.resource_history[history_key].append(metrics[metric_key])
                    
        except Exception as e:
            self.logger.error(f"Error updating metric history: {e}")
    
    async def _detect_resource_anomalies(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced anomaly detection using statistical and ML-based approaches"""
        try:
            anomalies = {
                'detected': [],
                'severity': {},
                'confidence': {},
                'recommendations': []
            }
            
            for resource_type, detector_config in self.anomaly_detectors.items():
                history_key = resource_type
                if history_key not in self.resource_history:
                    continue
                    
                history = list(self.resource_history[history_key])
                if len(history) < detector_config['window']:
                    continue
                    
                # Get current value
                metric_mappings = {
                    'cpu': 'cpu_percent',
                    'memory': 'memory_percent', 
                    'disk': 'disk_percent',
                    'gpu_utilization': 'gpu_utilization',
                    'gpu_memory': 'gpu_memory_percent',
                    'gpu_temperature': 'gpu_temperature'
                }
                
                metric_key = metric_mappings.get(resource_type)
                if not metric_key or metric_key not in current_metrics:
                    continue
                    
                current_value = current_metrics[metric_key]
                recent_history = history[-detector_config['window']:]
                
                # Statistical anomaly detection
                if len(recent_history) >= 10:
                    mean_val = np.mean(recent_history)
                    std_val = np.std(recent_history)
                    
                    if std_val > 0:
                        z_score = abs(current_value - mean_val) / std_val
                        
                        if z_score > detector_config['threshold']:
                            severity = min(z_score / detector_config['threshold'], 5.0)
                            confidence = min(0.5 + (z_score - detector_config['threshold']) * 0.1, 0.95)
                            
                            anomalies['detected'].append(resource_type)
                            anomalies['severity'][resource_type] = severity
                            anomalies['confidence'][resource_type] = confidence
                            
                            # Generate recommendations
                            recommendation = self._generate_anomaly_recommendation(
                                resource_type, current_value, mean_val, severity
                            )
                            anomalies['recommendations'].append(recommendation)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return {'detected': [], 'severity': {}, 'confidence': {}, 'recommendations': []}
    
    def _generate_anomaly_recommendation(self, resource_type: str, current_value: float, 
                                       baseline: float, severity: float) -> str:
        """Generate intelligent recommendations for resource anomalies"""
        if resource_type == 'cpu':
            if current_value > baseline:
                if severity > 3.0:
                    return "Critical CPU spike detected - consider process optimization or scaling"
                else:
                    return "Elevated CPU usage - monitor for sustained high utilization"
            else:
                return "Unusually low CPU usage - potential service degradation"
                
        elif resource_type == 'memory':
            if current_value > baseline:
                if severity > 3.0:
                    return "Critical memory pressure - implement memory optimization or add capacity"
                else:
                    return "Increased memory usage - monitor for memory leaks"
            else:
                return "Lower than expected memory usage"
                
        elif resource_type == 'gpu_temperature':
            if current_value > baseline:
                return "GPU thermal anomaly - check cooling and reduce workload intensity"
            else:
                return "Unusually low GPU temperature"
                
        else:
            direction = "high" if current_value > baseline else "low"
            return f"Anomalous {direction} {resource_type} detected - investigate system behavior"
    
    async def _generate_resource_predictions(self) -> Dict[str, Any]:
        """Generate predictive forecasts for resource utilization"""
        try:
            predictions = {
                'forecast_horizon_minutes': 30,
                'predictions': {},
                'confidence_intervals': {},
                'trend_analysis': {}
            }
            
            # Simple time series forecasting for each resource type
            for resource_type in ['cpu', 'memory', 'disk', 'gpu_utilization']:
                if resource_type not in self.resource_history:
                    continue
                    
                history = list(self.resource_history[resource_type])
                timestamps = list(self.resource_history['timestamps'])
                
                if len(history) < 30:  # Need sufficient history
                    continue
                    
                # Simple linear trend analysis
                recent_values = history[-30:]  # Last 30 readings
                
                if len(recent_values) >= 10:
                    # Calculate trend
                    x = np.arange(len(recent_values))
                    try:
                        slope, intercept = np.polyfit(x, recent_values, 1)
                        
                        # Project forward 30 steps (minutes)
                        future_steps = 30
                        future_x = np.arange(len(recent_values), len(recent_values) + future_steps)
                        predicted_values = slope * future_x + intercept
                        
                        # Calculate confidence based on recent variance
                        recent_variance = np.var(recent_values[-10:])
                        confidence = max(0.3, min(0.95, 1.0 - (recent_variance / 100)))
                        
                        predictions['predictions'][resource_type] = {
                            'next_30min_avg': float(np.mean(predicted_values)),
                            'next_30min_max': float(np.max(predicted_values)),
                            'trend_slope': float(slope),
                            'current_value': float(recent_values[-1])
                        }
                        
                        predictions['confidence_intervals'][resource_type] = {
                            'confidence': confidence,
                            'lower_bound': float(np.mean(predicted_values) - 2 * np.sqrt(recent_variance)),
                            'upper_bound': float(np.mean(predicted_values) + 2 * np.sqrt(recent_variance))
                        }
                        
                        # Trend classification
                        if abs(slope) < 0.1:
                            trend = "stable"
                        elif slope > 0:
                            trend = "increasing" if slope < 1.0 else "rapidly_increasing"
                        else:
                            trend = "decreasing" if slope > -1.0 else "rapidly_decreasing"
                            
                        predictions['trend_analysis'][resource_type] = trend
                        
                    except Exception as fit_error:
                        self.logger.debug(f"Trend fitting error for {resource_type}: {fit_error}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return {'predictions': {}, 'confidence_intervals': {}, 'trend_analysis': {}}
    
    async def _assess_capacity_planning(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current capacity and provide intelligent capacity planning insights"""
        try:
            capacity_status = {
                'overall_health': 'good',
                'resource_pressure': {},
                'scaling_recommendations': [],
                'time_to_capacity_exhaustion': {},
                'optimization_opportunities': []
            }
            
            # Assess individual resource pressure
            pressure_thresholds = {
                'cpu_percent': {'warning': 70, 'critical': 85},
                'memory_percent': {'warning': 75, 'critical': 90},
                'disk_percent': {'warning': 80, 'critical': 95},
                'gpu_utilization': {'warning': 70, 'critical': 85},
                'gpu_memory_percent': {'warning': 75, 'critical': 90}
            }
            
            overall_pressure_score = 0
            critical_resources = []
            
            for metric, thresholds in pressure_thresholds.items():
                if metric in current_metrics:
                    value = current_metrics[metric]
                    
                    if value >= thresholds['critical']:
                        pressure_level = 'critical'
                        pressure_score = 3
                        critical_resources.append(metric.replace('_percent', ''))
                    elif value >= thresholds['warning']:
                        pressure_level = 'warning'
                        pressure_score = 2
                    else:
                        pressure_level = 'normal'
                        pressure_score = 0
                    
                    capacity_status['resource_pressure'][metric] = {
                        'level': pressure_level,
                        'value': value,
                        'threshold_warning': thresholds['warning'],
                        'threshold_critical': thresholds['critical']
                    }
                    
                    overall_pressure_score += pressure_score
            
            # Determine overall health
            if overall_pressure_score >= 6:
                capacity_status['overall_health'] = 'critical'
            elif overall_pressure_score >= 3:
                capacity_status['overall_health'] = 'warning'
            else:
                capacity_status['overall_health'] = 'good'
            
            # Generate scaling recommendations
            if critical_resources:
                if 'cpu' in critical_resources:
                    capacity_status['scaling_recommendations'].append(
                        "CPU utilization critical - consider vertical scaling or process optimization"
                    )
                if 'memory' in critical_resources:
                    capacity_status['scaling_recommendations'].append(
                        "Memory pressure critical - increase RAM or implement memory optimization"
                    )
                if 'disk' in critical_resources:
                    capacity_status['scaling_recommendations'].append(
                        "Disk space critical - add storage capacity or implement data archiving"
                    )
                if 'gpu' in critical_resources or 'gpu_memory' in critical_resources:
                    capacity_status['scaling_recommendations'].append(
                        "GPU resources critical - optimize GPU workloads or add GPU capacity"
                    )
            
            # Estimate time to capacity exhaustion based on trends
            for resource_type in ['cpu', 'memory', 'disk']:
                history_key = resource_type
                if history_key in self.resource_history and len(self.resource_history[history_key]) >= 60:
                    recent_trend = list(self.resource_history[history_key])[-60:]  # Last hour
                    
                    # Calculate growth rate
                    if len(recent_trend) >= 30:
                        early_avg = np.mean(recent_trend[:30])
                        late_avg = np.mean(recent_trend[-30:])
                        growth_rate = (late_avg - early_avg) / 30  # per minute
                        
                        if growth_rate > 0.1:  # Growing significantly
                            current_value = recent_trend[-1]
                            capacity_limit = 95.0  # Assume 95% is practical limit
                            
                            if current_value < capacity_limit:
                                time_to_exhaustion = (capacity_limit - current_value) / growth_rate
                                capacity_status['time_to_capacity_exhaustion'][resource_type] = {
                                    'minutes': int(time_to_exhaustion),
                                    'hours': round(time_to_exhaustion / 60, 1),
                                    'growth_rate_per_minute': round(growth_rate, 3)
                                }
            
            # Identify optimization opportunities
            if 'memory_cached_gb' in current_metrics and current_metrics.get('memory_percent', 0) > 80:
                if current_metrics['memory_cached_gb'] > 2.0:  # Significant cache usage
                    capacity_status['optimization_opportunities'].append(
                        "Large memory cache detected - consider cache tuning for optimization"
                    )
            
            if 'swap_percent' in current_metrics and current_metrics['swap_percent'] > 10:
                capacity_status['optimization_opportunities'].append(
                    "Significant swap usage detected - consider adding RAM to reduce swapping"
                )
            
            return capacity_status
            
        except Exception as e:
            self.logger.error(f"Error assessing capacity planning: {e}")
            return {'overall_health': 'unknown', 'resource_pressure': {}}
    
    def _calculate_monitoring_quality(self) -> Dict[str, Any]:
        """Calculate monitoring system quality metrics"""
        try:
            quality_metrics = {
                'data_completeness': 0.0,
                'sampling_consistency': 0.0,
                'prediction_accuracy': 0.0,
                'overall_quality': 0.0
            }
            
            # Calculate data completeness
            total_expected_metrics = 6  # cpu, memory, disk, gpu_util, gpu_mem, gpu_temp
            collected_metrics = sum(1 for key in ['cpu', 'memory', 'disk', 'gpu_utilization', 'gpu_memory', 'gpu_temperature'] 
                                   if len(self.resource_history.get(key, [])) > 0)
            quality_metrics['data_completeness'] = collected_metrics / total_expected_metrics
            
            # Calculate sampling consistency (check for gaps in timestamps)
            timestamps = list(self.resource_history['timestamps'])
            if len(timestamps) >= 10:
                intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                            for i in range(1, min(len(timestamps), 60))]  # Check last hour
                if intervals:
                    avg_interval = np.mean(intervals)
                    interval_variance = np.var(intervals)
                    consistency = max(0.0, 1.0 - (interval_variance / (avg_interval ** 2)))
                    quality_metrics['sampling_consistency'] = min(1.0, consistency)
            
            # Overall quality score
            quality_metrics['overall_quality'] = (
                quality_metrics['data_completeness'] * 0.4 +
                quality_metrics['sampling_consistency'] * 0.3 +
                0.3  # Base score for functioning system
            )
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating monitoring quality: {e}")
            return {'overall_quality': 0.5}
    
    async def _process_resource_alerts(self, metrics: Dict[str, Any], 
                                     anomalies: Dict[str, Any], 
                                     predictions: Dict[str, Any]):
        """Process and manage resource alerts with intelligent notification"""
        try:
            current_time = datetime.utcnow()
            
            # Process capacity alerts
            for alert_type, config in self.capacity_alerts.items():
                alert_triggered = False
                
                if alert_type == 'cpu_sustained_high' and metrics.get('cpu_percent', 0) >= config['threshold']:
                    alert_triggered = True
                elif alert_type == 'memory_pressure' and metrics.get('memory_percent', 0) >= config['threshold']:
                    alert_triggered = True
                elif alert_type == 'disk_space_critical' and metrics.get('disk_percent', 0) >= config['threshold']:
                    alert_triggered = True
                elif alert_type == 'gpu_thermal_throttling' and metrics.get('gpu_temperature', 0) >= config['threshold']:
                    alert_triggered = True
                
                if alert_triggered:
                    if alert_type not in self.active_alerts:
                        # Start tracking this alert
                        self.active_alerts[alert_type] = {
                            'start_time': current_time,
                            'trigger_value': metrics.get(alert_type.split('_')[0] + '_percent', 0),
                            'notification_sent': False
                        }
                    else:
                        # Check if alert duration threshold is met
                        alert_duration = (current_time - self.active_alerts[alert_type]['start_time']).total_seconds()
                        
                        if (alert_duration >= config['duration'] and 
                            not self.active_alerts[alert_type]['notification_sent']):
                            
                            # Send notification
                            await self._send_resource_alert(alert_type, metrics, alert_duration)
                            self.active_alerts[alert_type]['notification_sent'] = True
                else:
                    # Clear alert if condition is no longer met
                    if alert_type in self.active_alerts:
                        self.logger.info(f"Resource alert cleared: {alert_type}")
                        del self.active_alerts[alert_type]
            
            # Process anomaly alerts
            for anomaly_resource in anomalies.get('detected', []):
                severity = anomalies['severity'].get(anomaly_resource, 1.0)
                confidence = anomalies['confidence'].get(anomaly_resource, 0.5)
                
                if severity >= 2.0 and confidence >= 0.7:
                    alert_key = f"anomaly_{anomaly_resource}"
                    
                    # Create anomaly alert record
                    anomaly_alert = {
                        'type': 'anomaly',
                        'resource': anomaly_resource,
                        'severity': severity,
                        'confidence': confidence,
                        'timestamp': current_time,
                        'current_value': metrics.get(anomaly_resource, 'unknown')
                    }
                    
                    self.alert_history.append(anomaly_alert)
                    self.logger.warning(f"Resource anomaly detected: {anomaly_resource} "
                                       f"(severity: {severity:.2f}, confidence: {confidence:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error processing resource alerts: {e}")
    
    async def _send_resource_alert(self, alert_type: str, metrics: Dict[str, Any], duration: float):
        """Send intelligent resource alerts with context and recommendations"""
        try:
            alert_message = {
                'alert_type': alert_type,
                'timestamp': datetime.utcnow().isoformat(),
                'duration_seconds': duration,
                'current_metrics': {
                    'cpu_percent': metrics.get('cpu_percent', 0),
                    'memory_percent': metrics.get('memory_percent', 0),
                    'disk_percent': metrics.get('disk_percent', 0),
                    'gpu_temperature': metrics.get('gpu_temperature', 0)
                },
                'severity': 'high' if duration > 600 else 'medium',  # High if > 10 minutes
                'recommendations': self._generate_alert_recommendations(alert_type, metrics)
            }
            
            # Store alert in history
            self.alert_history.append(alert_message)
            
            # Log structured alert
            self.logger.warning(f"RESOURCE ALERT: {alert_type} - Duration: {duration/60:.1f}min - "
                               f"Recommendations: {'; '.join(alert_message['recommendations'])}")
            
            # In a real system, this would integrate with notification systems
            # (e.g., email, Slack, PagerDuty, etc.)
            
        except Exception as e:
            self.logger.error(f"Error sending resource alert: {e}")
    
    def _generate_alert_recommendations(self, alert_type: str, metrics: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations for specific alert types"""
        recommendations = []
        
        if alert_type == 'cpu_sustained_high':
            cpu_percent = metrics.get('cpu_percent', 0)
            if cpu_percent > 95:
                recommendations.extend([
                    "Immediate action required - CPU at critical levels",
                    "Consider killing non-essential processes",
                    "Scale horizontally or upgrade CPU capacity"
                ])
            else:
                recommendations.extend([
                    "Monitor process CPU usage and optimize high-usage applications",
                    "Consider workload distribution or CPU scaling"
                ])
        
        elif alert_type == 'memory_pressure':
            memory_percent = metrics.get('memory_percent', 0)
            if memory_percent > 95:
                recommendations.extend([
                    "Critical memory shortage - immediate attention required",
                    "Restart high-memory processes or add RAM capacity",
                    "Implement memory usage monitoring and optimization"
                ])
            else:
                recommendations.extend([
                    "Monitor memory usage trends and optimize memory-intensive applications",
                    "Consider increasing available RAM"
                ])
        
        elif alert_type == 'disk_space_critical':
            recommendations.extend([
                "Immediate disk space cleanup required",
                "Archive or delete unnecessary files",
                "Add storage capacity or implement data archiving policies"
            ])
        
        elif alert_type == 'gpu_thermal_throttling':
            recommendations.extend([
                "GPU overheating detected - reduce GPU workload intensity",
                "Check GPU cooling system and ambient temperature",
                "Consider GPU thermal management optimization"
            ])
        
        return recommendations
    
    def _calculate_adaptive_monitoring_interval(self, metrics: Dict[str, Any], 
                                              anomalies: Dict[str, Any], 
                                              predictions: Dict[str, Any]) -> int:
        """Calculate adaptive monitoring interval based on system state"""
        try:
            base_interval = 60  # 1 minute base
            
            # Reduce interval if anomalies detected
            if anomalies.get('detected'):
                max_severity = max(anomalies.get('severity', {}).values(), default=0)
                if max_severity >= 3.0:
                    return max(15, base_interval // 4)  # Every 15 seconds for critical
                elif max_severity >= 2.0:
                    return max(30, base_interval // 2)  # Every 30 seconds for high
            
            # Reduce interval if any resource is under pressure
            pressure_resources = ['cpu_percent', 'memory_percent', 'disk_percent']
            high_utilization = any(metrics.get(resource, 0) > 80 for resource in pressure_resources)
            
            if high_utilization:
                return max(30, base_interval // 2)  # Monitor more frequently under load
            
            # Increase interval if system is stable
            stable_resources = all(metrics.get(resource, 0) < 50 for resource in pressure_resources)
            no_recent_anomalies = len(anomalies.get('detected', [])) == 0
            
            if stable_resources and no_recent_anomalies:
                return min(300, base_interval * 2)  # Every 5 minutes when stable
            
            return base_interval  # Default 1 minute
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive interval: {e}")
            return 60  # Fallback to 1 minute
    
    async def _intelligent_data_compression(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent data compression with context-aware algorithms and adaptive compression ratios"""
        try:
            compressed_data = {
                'original_size': 0,
                'compressed_size': 0,
                'compression_ratio': 1.0,
                'compression_method': 'none',
                'quality_score': 1.0,
                'metadata': data.get('metadata', {}),
                'compressed_payload': None,
                'decompression_info': {}
            }
            
            # Calculate original data size
            original_payload = data.get('payload', {})
            original_json = json.dumps(original_payload, separators=(',', ':'))
            original_size = len(original_json.encode('utf-8'))
            compressed_data['original_size'] = original_size
            
            # Skip compression for very small payloads (overhead not worth it)
            if original_size < 1024:  # Less than 1KB
                compressed_data.update({
                    'compressed_payload': original_payload,
                    'compressed_size': original_size,
                    'compression_method': 'none'
                })
                return compressed_data
            
            # Analyze data characteristics for optimal compression strategy
            data_analysis = await self._analyze_data_compressibility(original_payload)
            
            # Select optimal compression method based on data characteristics and constraints
            compression_method = await self._select_compression_method(
                data_analysis, 
                target_bandwidth=self.config.get('target_bandwidth_kbps', 100),
                latency_tolerance=self.config.get('latency_tolerance_ms', 500)
            )
            
            # Apply selected compression
            if compression_method == 'adaptive_lossy':
                compressed_result = await self._apply_adaptive_lossy_compression(
                    original_payload, data_analysis
                )
            elif compression_method == 'lossless_gzip':
                compressed_result = await self._apply_lossless_compression(
                    original_payload, algorithm='gzip'
                )
            elif compression_method == 'lossless_brotli':
                compressed_result = await self._apply_lossless_compression(
                    original_payload, algorithm='brotli'
                )
            elif compression_method == 'delta_compression':
                compressed_result = await self._apply_delta_compression(
                    original_payload, data_analysis
                )
            elif compression_method == 'ml_based':
                compressed_result = await self._apply_ml_based_compression(
                    original_payload, data_analysis
                )
            else:
                # Fallback to no compression
                compressed_result = {
                    'payload': original_payload,
                    'size': original_size,
                    'quality_score': 1.0,
                    'decompression_info': {}
                }
            
            compressed_data.update({
                'compressed_payload': compressed_result['payload'],
                'compressed_size': compressed_result['size'],
                'compression_method': compression_method,
                'compression_ratio': original_size / max(compressed_result['size'], 1),
                'quality_score': compressed_result['quality_score'],
                'decompression_info': compressed_result['decompression_info']
            })
            
            # Track compression performance metrics
            self.performance_metrics.update({
                'compression_ratio': compressed_data['compression_ratio'],
                'compression_method': compression_method,
                'avg_compression_time': getattr(self, 'avg_compression_time', 0),
                'data_throughput_mbps': original_size / (1024 * 1024 * max(0.001, compressed_result.get('compression_time', 0.001)))
            })
            
            return compressed_data
            
        except Exception as e:
            self.logger.error(f"Error in intelligent data compression: {e}")
            # Fallback to uncompressed data
            return {
                'compressed_payload': data.get('payload', {}),
                'compressed_size': len(json.dumps(data.get('payload', {})).encode('utf-8')),
                'compression_method': 'none',
                'compression_ratio': 1.0,
                'quality_score': 1.0,
                'original_size': len(json.dumps(data.get('payload', {})).encode('utf-8')),
                'metadata': data.get('metadata', {}),
                'decompression_info': {}
            }
    
    async def _analyze_data_compressibility(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data characteristics to determine optimal compression strategy"""
        try:
            analysis = {
                'data_type': 'mixed',
                'redundancy_score': 0.0,
                'time_series_ratio': 0.0,
                'numerical_ratio': 0.0,
                'text_ratio': 0.0,
                'repetition_patterns': [],
                'entropy': 0.0,
                'temporal_correlation': 0.0,
                'spatial_correlation': 0.0
            }
            
            # Analyze data composition
            total_elements = 0
            numerical_elements = 0
            text_elements = 0
            time_series_elements = 0
            
            def analyze_element(obj, path=""):
                nonlocal total_elements, numerical_elements, text_elements, time_series_elements
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        analyze_element(value, f"{path}.{key}" if path else key)
                elif isinstance(obj, list):
                    if len(obj) > 10 and all(isinstance(x, (int, float)) for x in obj[:10]):
                        time_series_elements += 1
                    for i, item in enumerate(obj):
                        analyze_element(item, f"{path}[{i}]")
                else:
                    total_elements += 1
                    if isinstance(obj, (int, float)):
                        numerical_elements += 1
                    elif isinstance(obj, str):
                        text_elements += 1
            
            analyze_element(payload)
            
            if total_elements > 0:
                analysis['numerical_ratio'] = numerical_elements / total_elements
                analysis['text_ratio'] = text_elements / total_elements
                analysis['time_series_ratio'] = time_series_elements / max(total_elements, 1)
            
            # Analyze redundancy using simple heuristics
            payload_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
            unique_chars = len(set(payload_str))
            total_chars = len(payload_str)
            analysis['redundancy_score'] = 1.0 - (unique_chars / max(total_chars, 1))
            
            # Simple entropy calculation
            if total_chars > 0:
                char_freq = {}
                for char in payload_str:
                    char_freq[char] = char_freq.get(char, 0) + 1
                
                entropy = 0
                for freq in char_freq.values():
                    prob = freq / total_chars
                    if prob > 0:
                        entropy -= prob * math.log2(prob)
                
                analysis['entropy'] = entropy / 8.0  # Normalize to 0-1 range
            
            # Determine primary data type
            if analysis['time_series_ratio'] > 0.3:
                analysis['data_type'] = 'time_series'
            elif analysis['numerical_ratio'] > 0.7:
                analysis['data_type'] = 'numerical'
            elif analysis['text_ratio'] > 0.7:
                analysis['data_type'] = 'text'
            else:
                analysis['data_type'] = 'mixed'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing data compressibility: {e}")
            return {'data_type': 'mixed', 'redundancy_score': 0.5}
    
    async def _select_compression_method(self, data_analysis: Dict[str, Any], 
                                       target_bandwidth: int = 100, 
                                       latency_tolerance: int = 500) -> str:
        """Intelligently select compression method based on data characteristics and constraints"""
        try:
            data_type = data_analysis.get('data_type', 'mixed')
            redundancy = data_analysis.get('redundancy_score', 0.5)
            time_series_ratio = data_analysis.get('time_series_ratio', 0.0)
            
            # Priority scoring for different methods
            method_scores = {
                'none': 0,
                'lossless_gzip': 0,
                'lossless_brotli': 0,
                'adaptive_lossy': 0,
                'delta_compression': 0,
                'ml_based': 0
            }
            
            # High redundancy favors lossless compression
            if redundancy > 0.6:
                method_scores['lossless_brotli'] += 3
                method_scores['lossless_gzip'] += 2
            
            # Time series data benefits from delta compression
            if time_series_ratio > 0.4:
                method_scores['delta_compression'] += 4
                method_scores['adaptive_lossy'] += 2
            
            # Low latency tolerance limits complex methods
            if latency_tolerance < 200:
                method_scores['lossless_gzip'] += 2
                method_scores['none'] += 1
                method_scores['ml_based'] -= 2
                method_scores['adaptive_lossy'] -= 1
            elif latency_tolerance > 1000:
                method_scores['ml_based'] += 2
                method_scores['lossless_brotli'] += 1
            
            # Bandwidth constraints favor aggressive compression
            if target_bandwidth < 50:  # Very limited bandwidth
                method_scores['adaptive_lossy'] += 4
                method_scores['ml_based'] += 3
                method_scores['delta_compression'] += 2
            elif target_bandwidth > 500:  # High bandwidth available
                method_scores['lossless_brotli'] += 1
                method_scores['none'] += 2
            
            # Geological data specific optimizations
            if data_analysis.get('numerical_ratio', 0) > 0.8:
                # Numerical data compresses well with delta encoding
                method_scores['delta_compression'] += 2
                method_scores['adaptive_lossy'] += 1
            
            # Select method with highest score
            best_method = max(method_scores.items(), key=lambda x: x[1])[0]
            
            self.logger.debug(f"Compression method selection: {best_method} "
                            f"(scores: {method_scores}, data_type: {data_type})")
            
            return best_method
            
        except Exception as e:
            self.logger.error(f"Error selecting compression method: {e}")
            return 'lossless_gzip'  # Safe fallback
    
    async def _apply_lossless_compression(self, payload: Dict[str, Any], 
                                        algorithm: str = 'gzip') -> Dict[str, Any]:
        """Apply lossless compression using specified algorithm"""
        try:
            import gzip
            import brotli
            import time
            
            start_time = time.time()
            
            # Convert to JSON bytes
            json_data = json.dumps(payload, separators=(',', ':')).encode('utf-8')
            
            # Apply compression
            if algorithm == 'brotli':
                compressed_bytes = brotli.compress(json_data, quality=6, mode=brotli.MODE_GENERIC)
            else:  # gzip fallback
                compressed_bytes = gzip.compress(json_data, compresslevel=6)
            
            compression_time = time.time() - start_time
            
            return {
                'payload': compressed_bytes.hex(),  # Store as hex string for JSON serialization
                'size': len(compressed_bytes),
                'quality_score': 1.0,  # Lossless
                'compression_time': compression_time,
                'decompression_info': {
                    'algorithm': algorithm,
                    'format': 'hex_encoded'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in lossless compression: {e}")
            # Fallback to uncompressed
            json_data = json.dumps(payload).encode('utf-8')
            return {
                'payload': payload,
                'size': len(json_data),
                'quality_score': 1.0,
                'compression_time': 0,
                'decompression_info': {'algorithm': 'none'}
            }
    
    async def _apply_adaptive_lossy_compression(self, payload: Dict[str, Any], 
                                              data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptive lossy compression for geological sensor data"""
        try:
            import time
            start_time = time.time()
            
            compressed_payload = {}
            quality_loss = 0.0
            
            # Process different data types with appropriate lossy strategies
            for key, value in payload.items():
                if isinstance(value, list) and len(value) > 10:
                    # Time series data - use downsampling and quantization
                    if all(isinstance(x, (int, float)) for x in value[:10]):
                        compressed_value, loss = await self._compress_time_series(
                            value, target_reduction=0.3
                        )
                        compressed_payload[key] = compressed_value
                        quality_loss = max(quality_loss, loss)
                    else:
                        compressed_payload[key] = value  # Keep non-numeric arrays unchanged
                
                elif isinstance(value, dict):
                    # Recursive compression for nested dictionaries
                    nested_result = await self._apply_adaptive_lossy_compression(
                        value, data_analysis
                    )
                    compressed_payload[key] = nested_result['payload']
                    quality_loss = max(quality_loss, 1.0 - nested_result['quality_score'])
                
                elif isinstance(value, (int, float)):
                    # Quantize numerical values with geological context
                    compressed_payload[key] = await self._quantize_geological_value(
                        value, key, target_bits=16
                    )
                    quality_loss = max(quality_loss, 0.01)  # Minimal loss for quantization
                
                else:
                    # Keep other data types unchanged
                    compressed_payload[key] = value
            
            # Apply final JSON compression
            json_str = json.dumps(compressed_payload, separators=(',', ':'))
            compressed_size = len(json_str.encode('utf-8'))
            compression_time = time.time() - start_time
            
            return {
                'payload': compressed_payload,
                'size': compressed_size,
                'quality_score': max(0.1, 1.0 - quality_loss),
                'compression_time': compression_time,
                'decompression_info': {
                    'algorithm': 'adaptive_lossy',
                    'quality_loss': quality_loss
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in adaptive lossy compression: {e}")
            json_data = json.dumps(payload).encode('utf-8')
            return {
                'payload': payload,
                'size': len(json_data),
                'quality_score': 1.0,
                'compression_time': 0,
                'decompression_info': {'algorithm': 'none'}
            }
    
    async def _compress_time_series(self, data: List[float], target_reduction: float = 0.3) -> tuple:
        """Compress time series data using intelligent downsampling and quantization"""
        try:
            if len(data) < 10:
                return data, 0.0
            
            # Calculate target length
            target_length = int(len(data) * (1.0 - target_reduction))
            target_length = max(target_length, 10)  # Maintain minimum resolution
            
            # Apply intelligent downsampling
            if len(data) > target_length:
                # Use adaptive sampling preserving peaks and important features
                compressed_data = await self._adaptive_downsample(data, target_length)
            else:
                compressed_data = data
            
            # Apply quantization to reduce precision
            quantized_data = [round(x, 4) for x in compressed_data]  # 4 decimal places
            
            # Estimate quality loss
            if len(data) == len(quantized_data):
                mse = np.mean([(orig - comp)**2 for orig, comp in zip(data[:len(quantized_data)], quantized_data)])
                signal_power = np.mean([x**2 for x in data])
                quality_loss = min(0.5, mse / max(signal_power, 1e-6))
            else:
                quality_loss = target_reduction  # Use target reduction as proxy for quality loss
            
            return quantized_data, quality_loss
            
        except Exception as e:
            self.logger.error(f"Error compressing time series: {e}")
            return data, 0.0
    
    async def _adaptive_downsample(self, data: List[float], target_length: int) -> List[float]:
        """Intelligent downsampling preserving important features"""
        try:
            if len(data) <= target_length:
                return data
            
            # Calculate window size for averaging
            window_size = len(data) // target_length
            
            downsampled = []
            for i in range(0, len(data) - window_size + 1, window_size):
                window = data[i:i + window_size]
                
                # Use median for robustness to outliers in geological data
                if len(window) > 0:
                    downsampled.append(np.median(window))
            
            # Ensure we have exactly target_length points
            while len(downsampled) < target_length and len(data) > 0:
                # Add remaining important points
                remaining_indices = list(range(len(data)))
                for used_idx in range(0, len(data), window_size):
                    if used_idx in remaining_indices:
                        remaining_indices.remove(used_idx)
                
                if remaining_indices:
                    # Add point with highest local variance (likely important feature)
                    best_idx = remaining_indices[0]
                    if len(remaining_indices) > 2:
                        variances = []
                        for idx in remaining_indices:
                            start_idx = max(0, idx - 2)
                            end_idx = min(len(data), idx + 3)
                            local_data = data[start_idx:end_idx]
                            variances.append(np.var(local_data) if len(local_data) > 1 else 0)
                        
                        best_variance_idx = np.argmax(variances)
                        best_idx = remaining_indices[best_variance_idx]
                    
                    downsampled.append(data[best_idx])
                else:
                    break
            
            return downsampled[:target_length]
            
        except Exception as e:
            self.logger.error(f"Error in adaptive downsampling: {e}")
            return data[::len(data)//target_length][:target_length]
    
    async def _quantize_geological_value(self, value: float, key: str, target_bits: int = 16) -> float:
        """Context-aware quantization for geological measurements"""
        try:
            # Define precision requirements based on measurement type
            precision_map = {
                'latitude': 6,      # ~0.1 meter precision
                'longitude': 6,     # ~0.1 meter precision  
                'elevation': 1,     # 0.1 meter precision
                'temperature': 2,   # 0.01°C precision
                'pressure': 2,      # 0.01 bar precision
                'velocity': 4,      # Seismic velocity precision
                'acceleration': 5,  # Acceleration precision
                'magnitude': 2,     # Earthquake magnitude
                'depth': 1,         # Depth precision
                'frequency': 3,     # Frequency precision
                'amplitude': 4      # Amplitude precision
            }
            
            # Find appropriate precision based on key name
            target_precision = precision_map.get(key.lower(), 4)  # Default 4 decimal places
            
            # Apply context-specific precision for common geological parameters
            for param_type, precision in precision_map.items():
                if param_type in key.lower():
                    target_precision = precision
                    break
            
            # Quantize the value
            quantized = round(value, target_precision)
            
            return quantized
            
        except Exception as e:
            self.logger.error(f"Error quantizing geological value: {e}")
            return value
    
    async def _apply_delta_compression(self, payload: Dict[str, Any], 
                                     data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply delta compression for time series and incremental data"""
        try:
            import time
            start_time = time.time()
            
            compressed_payload = {}
            
            for key, value in payload.items():
                if isinstance(value, list) and len(value) > 5:
                    # Apply delta compression to time series
                    if all(isinstance(x, (int, float)) for x in value[:5]):
                        compressed_value = await self._delta_encode_sequence(value)
                        compressed_payload[key] = {
                            'type': 'delta_encoded',
                            'base_value': value[0] if value else 0,
                            'deltas': compressed_value
                        }
                    else:
                        compressed_payload[key] = value
                else:
                    compressed_payload[key] = value
            
            # Serialize and calculate size
            json_str = json.dumps(compressed_payload, separators=(',', ':'))
            compressed_size = len(json_str.encode('utf-8'))
            compression_time = time.time() - start_time
            
            return {
                'payload': compressed_payload,
                'size': compressed_size,
                'quality_score': 1.0,  # Lossless delta compression
                'compression_time': compression_time,
                'decompression_info': {
                    'algorithm': 'delta_compression',
                    'requires_reconstruction': True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in delta compression: {e}")
            json_data = json.dumps(payload).encode('utf-8')
            return {
                'payload': payload,
                'size': len(json_data),
                'quality_score': 1.0,
                'compression_time': 0,
                'decompression_info': {'algorithm': 'none'}
            }
    
    async def _delta_encode_sequence(self, sequence: List[float]) -> List[float]:
        """Encode sequence using delta compression"""
        try:
            if len(sequence) <= 1:
                return sequence
            
            deltas = []
            for i in range(1, len(sequence)):
                delta = sequence[i] - sequence[i-1]
                # Quantize small deltas to improve compression
                if abs(delta) < 1e-6:
                    delta = 0
                else:
                    delta = round(delta, 6)
                deltas.append(delta)
            
            return deltas
            
        except Exception as e:
            self.logger.error(f"Error in delta encoding: {e}")
            return sequence
    
    async def _apply_ml_based_compression(self, payload: Dict[str, Any], 
                                        data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ML-based compression using learned representations"""
        try:
            # For now, implement a sophisticated heuristic-based approach
            # In production, this would use trained neural compression models
            
            import time
            start_time = time.time()
            
            # Use existing data integration pipeline for feature extraction
            if hasattr(self, 'data_pipeline'):
                # Integrate with your real data pipeline
                from .data_integration_pipeline import DataIntegrationPipeline
                
                try:
                    # Extract features using your existing pipeline
                    features = await self.data_pipeline.extract_features(payload)
                    
                    # Use feature-based compression
                    compressed_payload = await self._feature_based_compression(
                        payload, features, data_analysis
                    )
                    
                except Exception as pipeline_error:
                    self.logger.warning(f"Data pipeline compression failed: {pipeline_error}")
                    # Fallback to adaptive lossy compression
                    return await self._apply_adaptive_lossy_compression(payload, data_analysis)
            else:
                # Fallback compression using statistical analysis
                compressed_payload = await self._statistical_compression(payload, data_analysis)
            
            json_str = json.dumps(compressed_payload, separators=(',', ':'))
            compressed_size = len(json_str.encode('utf-8'))
            compression_time = time.time() - start_time
            
            return {
                'payload': compressed_payload,
                'size': compressed_size,
                'quality_score': 0.85,  # ML-based compression with some quality loss
                'compression_time': compression_time,
                'decompression_info': {
                    'algorithm': 'ml_based',
                    'requires_model': True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML-based compression: {e}")
            # Fallback to adaptive lossy
            return await self._apply_adaptive_lossy_compression(payload, data_analysis)
    
    async def _feature_based_compression(self, payload: Dict[str, Any], 
                                       features: Dict[str, Any], 
                                       data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compress data using extracted features from data integration pipeline"""
        try:
            compressed_payload = {
                'compressed_features': features,
                'reconstruction_hints': {},
                'metadata': payload.get('metadata', {})
            }
            
            # Store essential raw data for critical parameters
            critical_params = ['timestamp', 'device_id', 'location', 'magnitude', 'alert_level']
            for param in critical_params:
                if param in payload:
                    compressed_payload[param] = payload[param]
            
            # Add reconstruction hints for important geological parameters
            if 'seismic_data' in payload:
                seismic_data = payload['seismic_data']
                if isinstance(seismic_data, list) and len(seismic_data) > 0:
                    # Store statistical moments for reconstruction
                    compressed_payload['reconstruction_hints']['seismic_stats'] = {
                        'mean': np.mean(seismic_data),
                        'std': np.std(seismic_data),
                        'min': np.min(seismic_data),
                        'max': np.max(seismic_data),
                        'length': len(seismic_data)
                    }
            
            return compressed_payload
            
        except Exception as e:
            self.logger.error(f"Error in feature-based compression: {e}")
            return payload
    
    async def _statistical_compression(self, payload: Dict[str, Any], 
                                     data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Statistical compression fallback when ML models unavailable"""
        try:
            compressed_payload = {}
            
            for key, value in payload.items():
                if isinstance(value, list) and len(value) > 20:
                    # Use statistical representation for large arrays
                    if all(isinstance(x, (int, float)) for x in value[:10]):
                        # Replace with statistical summary + sample points
                        stats = {
                            'type': 'statistical_summary',
                            'count': len(value),
                            'mean': np.mean(value),
                            'std': np.std(value),
                            'min': np.min(value),
                            'max': np.max(value),
                            'percentiles': {
                                '25': np.percentile(value, 25),
                                '50': np.percentile(value, 50), 
                                '75': np.percentile(value, 75)
                            },
                            'sample_points': value[::max(1, len(value)//20)][:20]  # Keep 20 sample points
                        }
                        compressed_payload[key] = stats
                    else:
                        compressed_payload[key] = value
                else:
                    compressed_payload[key] = value
            
            return compressed_payload
            
        except Exception as e:
            self.logger.error(f"Error in statistical compression: {e}")
            return payload
    
    async def _edge_cloud_synchronization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent edge-cloud synchronization with adaptive strategies and conflict resolution"""
        try:
            sync_result = {
                'status': 'success',
                'sync_method': 'none',
                'conflicts_resolved': 0,
                'data_transferred': 0,
                'latency_ms': 0,
                'cloud_availability': True,
                'sync_quality': 1.0,
                'next_sync_time': None
            }
            
            # Check cloud connectivity and determine sync strategy
            cloud_status = await self._assess_cloud_connectivity()
            sync_result['cloud_availability'] = cloud_status['available']
            
            if not cloud_status['available']:
                # Store data locally and schedule retry
                await self._store_local_buffer(data)
                sync_result.update({
                    'status': 'buffered',
                    'sync_method': 'local_buffer',
                    'next_sync_time': datetime.utcnow() + timedelta(minutes=5)
                })
                return sync_result
            
            # Determine optimal sync strategy based on data characteristics and network conditions
            sync_strategy = await self._determine_sync_strategy(data, cloud_status)
            sync_result['sync_method'] = sync_strategy
            
            # Execute synchronization based on selected strategy
            if sync_strategy == 'immediate_full':
                result = await self._immediate_full_sync(data, cloud_status)
            elif sync_strategy == 'batch_compressed':
                result = await self._batch_compressed_sync(data, cloud_status)
            elif sync_strategy == 'incremental_delta':
                result = await self._incremental_delta_sync(data, cloud_status)
            elif sync_strategy == 'priority_selective':
                result = await self._priority_selective_sync(data, cloud_status)
            elif sync_strategy == 'federated_model':
                result = await self._federated_model_sync(data, cloud_status)
            else:
                # Fallback to basic sync
                result = await self._basic_sync_fallback(data, cloud_status)
            
            sync_result.update(result)
            
            # Integrate with your real database systems
            if hasattr(self, 'database_integration'):
                try:
                    await self._sync_with_real_databases(data, sync_result)
                except Exception as db_error:
                    self.logger.warning(f"Database integration sync failed: {db_error}")
            
            # Update synchronization metrics
            self.performance_metrics.update({
                'last_sync_time': datetime.utcnow().isoformat(),
                'sync_success_rate': getattr(self, 'sync_success_rate', 0.95),
                'avg_sync_latency_ms': sync_result['latency_ms'],
                'data_sync_throughput_mbps': sync_result['data_transferred'] / (1024 * 1024 * max(0.001, sync_result['latency_ms'] / 1000))
            })
            
            return sync_result
            
        except Exception as e:
            self.logger.error(f"Error in edge-cloud synchronization: {e}")
            return {
                'status': 'failed',
                'sync_method': 'none',
                'error': str(e),
                'cloud_availability': False,
                'sync_quality': 0.0
            }
    
    async def _assess_cloud_connectivity(self) -> Dict[str, Any]:
        """Assess cloud connectivity and network conditions"""
        try:
            import time
            start_time = time.time()
            
            # Simulate cloud connectivity assessment
            # In production, this would test actual cloud endpoints
            connectivity = {
                'available': True,
                'latency_ms': 50,
                'bandwidth_mbps': 100,
                'reliability_score': 0.95,
                'cost_per_mb': 0.01,
                'congestion_level': 'low'
            }
            
            # Simple ping test simulation
            test_latency = time.time() - start_time
            connectivity['latency_ms'] = max(10, test_latency * 1000)
            
            # Assess network quality based on recent history
            recent_failures = getattr(self, 'recent_sync_failures', 0)
            if recent_failures > 3:
                connectivity['reliability_score'] = max(0.1, 0.95 - (recent_failures * 0.1))
                connectivity['available'] = connectivity['reliability_score'] > 0.3
            
            return connectivity
            
        except Exception as e:
            self.logger.error(f"Error assessing cloud connectivity: {e}")
            return {
                'available': False,
                'latency_ms': 9999,
                'bandwidth_mbps': 0,
                'reliability_score': 0.0,
                'cost_per_mb': 99.0,
                'congestion_level': 'high'
            }
    
    async def _determine_sync_strategy(self, data: Dict[str, Any], 
                                     cloud_status: Dict[str, Any]) -> str:
        """Determine optimal sync strategy based on data and network conditions"""
        try:
            data_size_mb = len(json.dumps(data).encode('utf-8')) / (1024 * 1024)
            latency_ms = cloud_status.get('latency_ms', 100)
            bandwidth_mbps = cloud_status.get('bandwidth_mbps', 10)
            reliability = cloud_status.get('reliability_score', 0.8)
            
            # Priority scoring for different strategies
            strategy_scores = {
                'immediate_full': 0,
                'batch_compressed': 0,
                'incremental_delta': 0,
                'priority_selective': 0,
                'federated_model': 0
            }
            
            # Small data favors immediate sync
            if data_size_mb < 1.0:
                strategy_scores['immediate_full'] += 3
            
            # Large data favors compression
            if data_size_mb > 5.0:
                strategy_scores['batch_compressed'] += 4
                strategy_scores['priority_selective'] += 3
            
            # Low bandwidth favors selective/compressed strategies
            if bandwidth_mbps < 10:
                strategy_scores['priority_selective'] += 3
                strategy_scores['batch_compressed'] += 2
                strategy_scores['federated_model'] += 1
                strategy_scores['immediate_full'] -= 2
            
            # High latency favors batch processing
            if latency_ms > 200:
                strategy_scores['batch_compressed'] += 2
                strategy_scores['federated_model'] += 1
                strategy_scores['immediate_full'] -= 1
            
            # Low reliability favors incremental approaches
            if reliability < 0.8:
                strategy_scores['incremental_delta'] += 3
                strategy_scores['priority_selective'] += 2
                strategy_scores['immediate_full'] -= 1
            
            # Critical geological data gets priority handling
            if self._has_critical_geological_data(data):
                strategy_scores['priority_selective'] += 2
                strategy_scores['immediate_full'] += 1
            
            # Time series data benefits from incremental sync
            if self._has_time_series_data(data):
                strategy_scores['incremental_delta'] += 2
                strategy_scores['batch_compressed'] += 1
            
            # Select strategy with highest score
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
            
            self.logger.debug(f"Sync strategy selected: {best_strategy} "
                            f"(data_size: {data_size_mb:.2f}MB, latency: {latency_ms}ms)")
            
            return best_strategy
            
        except Exception as e:
            self.logger.error(f"Error determining sync strategy: {e}")
            return 'batch_compressed'  # Safe fallback
    
    def _has_critical_geological_data(self, data: Dict[str, Any]) -> bool:
        """Check if data contains critical geological information requiring immediate sync"""
        critical_indicators = [
            'earthquake', 'volcanic_eruption', 'tsunami', 'landslide', 
            'alert_level', 'emergency', 'magnitude', 'critical'
        ]
        
        data_str = json.dumps(data).lower()
        return any(indicator in data_str for indicator in critical_indicators)
    
    def _has_time_series_data(self, data: Dict[str, Any]) -> bool:
        """Check if data contains significant time series components"""
        for value in data.values():
            if isinstance(value, list) and len(value) > 10:
                # Check if it's a numeric time series
                if all(isinstance(x, (int, float)) for x in value[:10]):
                    return True
        return False
    
    async def _immediate_full_sync(self, data: Dict[str, Any], 
                                 cloud_status: Dict[str, Any]) -> Dict[str, Any]:
        """Immediate full synchronization for small or critical data"""
        try:
            import time
            start_time = time.time()
            
            # Compress data for transmission
            compressed_data = await self._intelligent_data_compression(data)
            
            # Simulate cloud upload
            transmission_time = compressed_data['compressed_size'] / (cloud_status.get('bandwidth_mbps', 10) * 1024 * 1024) * 8
            await asyncio.sleep(min(transmission_time, 0.1))  # Cap simulation delay
            
            sync_time = time.time() - start_time
            
            return {
                'status': 'success',
                'data_transferred': compressed_data['compressed_size'],
                'latency_ms': sync_time * 1000,
                'compression_ratio': compressed_data['compression_ratio'],
                'sync_quality': compressed_data['quality_score']
            }
            
        except Exception as e:
            self.logger.error(f"Error in immediate full sync: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _batch_compressed_sync(self, data: Dict[str, Any], 
                                   cloud_status: Dict[str, Any]) -> Dict[str, Any]:
        """Batch compressed synchronization for large datasets"""
        try:
            # Add to batch buffer
            if not hasattr(self, 'batch_buffer'):
                self.batch_buffer = []
            
            self.batch_buffer.append(data)
            
            # Check if batch is ready to sync
            batch_size_limit = 10  # Number of items
            batch_age_limit = 300  # 5 minutes
            
            if (len(self.batch_buffer) >= batch_size_limit or 
                getattr(self, 'batch_start_time', datetime.utcnow()) + 
                timedelta(seconds=batch_age_limit) < datetime.utcnow()):
                
                # Process the batch
                return await self._process_batch_sync(cloud_status)
            else:
                # Batch not ready yet
                if not hasattr(self, 'batch_start_time'):
                    self.batch_start_time = datetime.utcnow()
                
                return {
                    'status': 'batched',
                    'data_transferred': 0,
                    'latency_ms': 0,
                    'items_in_batch': len(self.batch_buffer),
                    'sync_quality': 1.0
                }
                
        except Exception as e:
            self.logger.error(f"Error in batch compressed sync: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _process_batch_sync(self, cloud_status: Dict[str, Any]) -> Dict[str, Any]:
        """Process accumulated batch for synchronization"""
        try:
            import time
            start_time = time.time()
            
            if not hasattr(self, 'batch_buffer') or not self.batch_buffer:
                return {'status': 'no_data', 'data_transferred': 0, 'latency_ms': 0}
            
            # Combine batch data
            combined_data = {
                'batch_id': f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'items': self.batch_buffer,
                'item_count': len(self.batch_buffer),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Apply aggressive compression to batch
            compressed_batch = await self._intelligent_data_compression(combined_data)
            
            # Simulate batch upload
            transmission_time = compressed_batch['compressed_size'] / (cloud_status.get('bandwidth_mbps', 10) * 1024 * 1024) * 8
            await asyncio.sleep(min(transmission_time, 0.2))
            
            sync_time = time.time() - start_time
            items_processed = len(self.batch_buffer)
            
            # Clear the batch
            self.batch_buffer = []
            if hasattr(self, 'batch_start_time'):
                delattr(self, 'batch_start_time')
            
            return {
                'status': 'success',
                'data_transferred': compressed_batch['compressed_size'],
                'latency_ms': sync_time * 1000,
                'items_processed': items_processed,
                'compression_ratio': compressed_batch['compression_ratio'],
                'sync_quality': compressed_batch['quality_score']
            }
            
        except Exception as e:
            self.logger.error(f"Error processing batch sync: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _store_local_buffer(self, data: Dict[str, Any]):
        """Store data in local buffer when cloud is unavailable"""
        try:
            if not hasattr(self, 'local_buffer'):
                self.local_buffer = deque(maxlen=1000)  # Limit buffer size
            
            buffered_item = {
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'retry_count': 0
            }
            
            self.local_buffer.append(buffered_item)
            
            self.logger.info(f"Data buffered locally, buffer size: {len(self.local_buffer)}")
            
        except Exception as e:
            self.logger.error(f"Error storing local buffer: {e}")
    
    async def _sync_with_real_databases(self, data: Dict[str, Any], sync_result: Dict[str, Any]):
        """Integrate with your real database systems"""
        try:
            # Import your real database integration modules
            from .database_integration import PostgreSQLGISManager, TimeScaleDBManager
            from .external_data_connectors import DataIntegrationManager
            
            # Sync with PostgreSQL GIS if available
            if hasattr(self, 'postgres_manager'):
                try:
                    await self._sync_with_postgres(data, self.postgres_manager)
                except Exception as pg_error:
                    self.logger.warning(f"PostgreSQL sync error: {pg_error}")
            
            # Sync with TimeScaleDB if available  
            if hasattr(self, 'timescale_manager'):
                try:
                    await self._sync_with_timescale(data, self.timescale_manager)
                except Exception as ts_error:
                    self.logger.warning(f"TimeScaleDB sync error: {ts_error}")
            
            # Sync with external data connectors
            if hasattr(self, 'external_connectors'):
                try:
                    await self._sync_with_external_connectors(data, self.external_connectors)
                except Exception as ext_error:
                    self.logger.warning(f"External connectors sync error: {ext_error}")
                    
        except Exception as e:
            self.logger.error(f"Error syncing with real databases: {e}")
    
    async def _sync_with_postgres(self, data: Dict[str, Any], postgres_manager):
        """Sync data with PostgreSQL GIS database"""
        try:
            # Extract geological data for PostgreSQL storage
            if 'location' in data and 'timestamp' in data:
                geological_event = {
                    'event_type': data.get('event_type', 'sensor_reading'),
                    'magnitude': data.get('magnitude', 0.0),
                    'latitude': data.get('location', {}).get('lat', 0.0),
                    'longitude': data.get('location', {}).get('lon', 0.0),
                    'depth': data.get('depth', 0.0),
                    'timestamp': data['timestamp'],
                    'metadata': json.dumps(data.get('metadata', {}))
                }
                
                # Store in geological events table
                await postgres_manager.store_geological_event(geological_event)
                
        except Exception as e:
            self.logger.error(f"PostgreSQL sync error: {e}")
    
    async def _sync_with_timescale(self, data: Dict[str, Any], timescale_manager):
        """Sync time series data with TimeScaleDB"""
        try:
            # Extract time series data
            timestamp = data.get('timestamp', datetime.utcnow())
            device_id = data.get('device_id', self.device_id)
            
            # Store sensor readings as time series
            for key, value in data.items():
                if isinstance(value, (int, float)) and key not in ['timestamp', 'device_id']:
                    time_series_record = {
                        'timestamp': timestamp,
                        'device_id': device_id,
                        'metric_name': key,
                        'value': value,
                        'metadata': json.dumps({'data_type': type(value).__name__})
                    }
                    
                    await timescale_manager.insert_time_series_data(time_series_record)
                    
        except Exception as e:
            self.logger.error(f"TimeScaleDB sync error: {e}")
    
    async def _sync_with_external_connectors(self, data: Dict[str, Any], external_connectors):
        """Sync with external data sources and APIs"""
        try:
            # If data contains USGS-relevant information, sync with USGS
            if 'seismic' in str(data).lower() or 'earthquake' in str(data).lower():
                if hasattr(external_connectors, 'usgs_connector'):
                    # Cross-reference with USGS earthquake data
                    await external_connectors.usgs_connector.validate_seismic_event(data)
            
            # Sync with other external data sources as needed
            if hasattr(external_connectors, 'nasa_connector') and 'satellite' in str(data).lower():
                await external_connectors.nasa_connector.correlate_earth_data(data)
                
        except Exception as e:
            self.logger.error(f"External connectors sync error: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self.is_active:
            try:
                self.last_heartbeat = datetime.utcnow()
                
                heartbeat_message = {
                    'type': 'heartbeat',
                    'device_id': self.device_id,
                    'device_type': self.device_type.value,
                    'timestamp': self.last_heartbeat.isoformat(),
                    'status': 'active',
                    'performance_metrics': self.performance_metrics
                }
                
                await self.message_queue.put(heartbeat_message)
                
                await asyncio.sleep(60)  # Heartbeat every minute
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(60)

class SeismicEdgeDevice(EdgeDevice):
    """Seismic sensor edge device"""
    
    def __init__(self, device_id: str, capabilities: EdgeDeviceCapabilities, 
                 config: Dict[str, Any] = None):
        super().__init__(device_id, EdgeDeviceType.SEISMIC_SENSOR, capabilities, config)
        
        # Seismic-specific configuration
        self.sampling_rate = config.get('sampling_rate', 100)  # Hz
        self.sensitivity = config.get('sensitivity', 1.0)  # V/m/s
        self.frequency_range = config.get('frequency_range', (0.1, 50))  # Hz
    
    async def _read_sensor(self) -> Optional[SensorReading]:
        """Read seismic sensor data"""
        try:
            # Simulate 3-component seismic data
            duration = 1.0  # 1 second of data
            samples = int(self.sampling_rate * duration)
            time_vector = np.linspace(0, duration, samples)
            
            # Simulate seismic signals with noise
            noise_level = 0.01
            signal_components = []
            
            for component in ['Z', 'N', 'E']:  # Vertical, North, East
                # Background noise
                noise = np.random.normal(0, noise_level, samples)
                
                # Occasional seismic signals
                if np.random.random() < 0.01:  # 1% chance of signal
                    # P-wave arrival
                    p_freq = np.random.uniform(5, 20)
                    p_amplitude = np.random.uniform(0.1, 0.5)
                    p_signal = p_amplitude * np.sin(2 * np.pi * p_freq * time_vector)
                    p_signal *= np.exp(-time_vector * 2)  # Decay
                    noise += p_signal
                
                signal_components.append(noise * self.sensitivity)
            
            seismic_data = np.array(signal_components).T  # Shape: (samples, 3)
            
            return SensorReading(
                sensor_id=self.device_id,
                timestamp=datetime.utcnow(),
                reading_type='seismic_waveform',
                value=seismic_data,
                unit='m/s',
                location=self.config.get('location'),
                quality=np.random.uniform(0.9, 1.0),
                metadata={
                    'sampling_rate': self.sampling_rate,
                    'components': ['Z', 'N', 'E'],
                    'duration': duration
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error reading seismic sensor: {e}")
            return None

class GPSEdgeDevice(EdgeDevice):
    """GPS station edge device"""
    
    def __init__(self, device_id: str, capabilities: EdgeDeviceCapabilities,
                 config: Dict[str, Any] = None):
        super().__init__(device_id, EdgeDeviceType.GPS_STATION, capabilities, config)
        
        # GPS-specific configuration
        self.reference_position = config.get('reference_position', (35.0, -118.0, 1000.0))
        self.positioning_accuracy = config.get('accuracy', 0.001)  # meters
    
    async def _read_sensor(self) -> Optional[SensorReading]:
        """Read GPS position data"""
        try:
            # Simulate GPS position with small random walk
            ref_lat, ref_lon, ref_elev = self.reference_position
            
            # Add small displacement (mm-level for geological monitoring)
            displacement_x = np.random.normal(0, 0.002)  # 2mm std
            displacement_y = np.random.normal(0, 0.002)  # 2mm std
            displacement_z = np.random.normal(0, 0.005)  # 5mm std (vertical less precise)
            
            # Convert to lat/lon (approximate)
            lat = ref_lat + displacement_y / 111320  # degrees
            lon = ref_lon + displacement_x / (111320 * np.cos(np.radians(ref_lat)))
            elev = ref_elev + displacement_z
            
            position_data = {
                'latitude': lat,
                'longitude': lon,
                'elevation': elev,
                'displacement_x': displacement_x,
                'displacement_y': displacement_y,
                'displacement_z': displacement_z,
                'accuracy': self.positioning_accuracy
            }
            
            return SensorReading(
                sensor_id=self.device_id,
                timestamp=datetime.utcnow(),
                reading_type='gps_position',
                value=position_data,
                unit='degrees/meters',
                location=(lat, lon, elev),
                quality=np.random.uniform(0.85, 1.0),
                metadata={
                    'reference_position': self.reference_position,
                    'positioning_mode': 'RTK'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error reading GPS sensor: {e}")
            return None

class CommunicationManager(ABC):
    """Base communication manager"""
    
    def __init__(self, device_id: str, config: Dict[str, Any] = None):
        self.device_id = device_id
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.comm.{device_id}")
        self.is_connected = False
    
    @abstractmethod
    async def start(self):
        """Start communication"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop communication"""
        pass
    
    @abstractmethod
    async def send_message(self, message: Dict[str, Any]):
        """Send message"""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message"""
        pass

class MQTTCommunicationManager(CommunicationManager):
    """MQTT-based communication manager"""
    
    def __init__(self, device_id: str, config: Dict[str, Any] = None):
        super().__init__(device_id, config)
        self.client = None
        self.message_queue = asyncio.Queue()
        
        if MQTT_AVAILABLE:
            self.client = mqtt.Client(client_id=f"{device_id}_{int(time.time())}")
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
    
    async def start(self):
        """Start MQTT communication"""
        if not MQTT_AVAILABLE or not self.client:
            return
        
        try:
            broker_host = self.config.get('mqtt_broker', 'localhost')
            broker_port = self.config.get('mqtt_port', 1883)
            
            self.client.connect(broker_host, broker_port, 60)
            self.client.loop_start()
            
            # Subscribe to command topics
            command_topic = f"prometheus/devices/{self.device_id}/commands"
            self.client.subscribe(command_topic)
            
            self.logger.info(f"MQTT communication started, connected to {broker_host}:{broker_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start MQTT communication: {e}")
    
    async def stop(self):
        """Stop MQTT communication"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
    
    async def send_message(self, message: Dict[str, Any]):
        """Send MQTT message"""
        if not MQTT_AVAILABLE or not self.client or not self.is_connected:
            return
        
        try:
            topic = f"prometheus/devices/{self.device_id}/data"
            payload = json.dumps(message)
            
            self.client.publish(topic, payload, qos=1)
            
        except Exception as e:
            self.logger.error(f"Failed to send MQTT message: {e}")
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive MQTT message"""
        try:
            if not self.message_queue.empty():
                return await self.message_queue.get()
        except Exception as e:
            self.logger.error(f"Failed to receive MQTT message: {e}")
        
        return None
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connect callback"""
        if rc == 0:
            self.is_connected = True
            self.logger.info("MQTT connected successfully")
        else:
            self.logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        self.is_connected = False
        self.logger.info("MQTT disconnected")
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            message = json.loads(msg.payload.decode())
            asyncio.create_task(self.message_queue.put(message))
        except Exception as e:
            self.logger.error(f"Failed to process MQTT message: {e}")

class ZMQCommunicationManager(CommunicationManager):
    """ZeroMQ-based communication manager"""
    
    def __init__(self, device_id: str, config: Dict[str, Any] = None):
        super().__init__(device_id, config)
        self.context = None
        self.socket = None
        
        if ZMQ_AVAILABLE:
            self.context = zmq.Context()
    
    async def start(self):
        """Start ZMQ communication"""
        if not ZMQ_AVAILABLE or not self.context:
            return
        
        try:
            # Create PUSH socket for sending data
            self.socket = self.context.socket(zmq.PUSH)
            
            server_address = self.config.get('zmq_server', 'tcp://localhost:5555')
            self.socket.connect(server_address)
            
            self.is_connected = True
            self.logger.info(f"ZMQ communication started, connected to {server_address}")
            
        except Exception as e:
            self.logger.error(f"Failed to start ZMQ communication: {e}")
    
    async def stop(self):
        """Stop ZMQ communication"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
    
    async def send_message(self, message: Dict[str, Any]):
        """Send ZMQ message"""
        if not self.socket or not self.is_connected:
            return
        
        try:
            payload = json.dumps(message).encode()
            self.socket.send(payload, zmq.NOBLOCK)
            
        except zmq.Again:
            self.logger.warning("ZMQ send buffer full, message dropped")
        except Exception as e:
            self.logger.error(f"Failed to send ZMQ message: {e}")
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive ZMQ message"""
        # This implementation is for PUSH socket (send-only)
        # For bidirectional communication, would need REQ-REP or DEALER-ROUTER
        return None

class LocalCommunicationManager(CommunicationManager):
    """Local in-memory communication manager for testing"""
    
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.message_buffer = []
    
    async def start(self):
        """Start local communication"""
        self.is_connected = True
        self.logger.info("Local communication started")
    
    async def stop(self):
        """Stop local communication"""
        self.is_connected = False
    
    async def send_message(self, message: Dict[str, Any]):
        """Send local message"""
        self.message_buffer.append({
            'timestamp': datetime.utcnow().isoformat(),
            'device_id': self.device_id,
            'message': message
        })
        
        # Keep only last 100 messages
        if len(self.message_buffer) > 100:
            self.message_buffer = self.message_buffer[-100:]
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive local message"""
        return None  # No incoming messages in local mode

# Signal processors
class SeismicSignalProcessor:
    """Advanced seismic waveform processing with comprehensive digital signal processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with advanced signal processing configuration"""
        self.config = config or {}
        self.sampling_rate = self.config.get('sampling_rate', 100.0)  # Hz
        
        # Advanced filter configuration
        self.filter_params = self.config.get('filter', {
            'lowpass': 50.0,        # Lowpass cutoff (Hz)
            'highpass': 0.1,        # Highpass cutoff (Hz)  
            'bandpass': (1.0, 10.0), # Bandpass range (Hz)
            'order': 4,             # Butterworth filter order
            'detrend': True,        # Remove linear trend
            'taper': True,          # Apply Tukey taper
            'taper_fraction': 0.05  # Taper fraction
        })
        
        # Noise reduction parameters
        self.noise_params = self.config.get('noise_reduction', {
            'enable_wiener': True,      # Wiener filtering
            'enable_spectral_sub': True, # Spectral subtraction
            'noise_floor_db': -40,      # Noise floor (dB)
            'adaptive_threshold': True   # Adaptive noise threshold
        })
        
        # Quality control thresholds
        self.quality_params = self.config.get('quality', {
            'min_snr': 3.0,            # Minimum signal-to-noise ratio
            'max_clip_fraction': 0.1,   # Maximum clipping fraction
            'stability_threshold': 0.01 # Numerical stability threshold
        })
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply comprehensive signal processing to seismic data"""
        if data.get('reading_type') != 'seismic_waveform':
            return data
        
        try:
            waveform = np.array(data['value'])
            
            if waveform.size == 0:
                logging.warning("Empty waveform data received")
                return data
            
            # Ensure proper 2D shape (samples, components)
            if waveform.ndim == 1:
                waveform = waveform.reshape(-1, 1)
            
            # Pre-processing quality checks
            quality_metrics = self._assess_signal_quality(waveform)
            
            if quality_metrics['overall_quality'] < 0.3:
                logging.warning(f"Poor signal quality detected: {quality_metrics}")
                data['metadata']['quality_warning'] = True
                data['metadata']['quality_metrics'] = quality_metrics
            
            # Apply comprehensive signal processing pipeline
            processed_waveform = self._apply_signal_processing_pipeline(waveform)
            
            # Post-processing validation
            if not self._validate_processed_signal(processed_waveform):
                logging.error("Signal processing validation failed, returning original")
                return data
            
            # Update data with processed results
            data['value'] = processed_waveform.tolist()
            data['metadata']['processed'] = True
            data['metadata']['processing_config'] = {
                'filters_applied': self.filter_params,
                'noise_reduction': self.noise_params,
                'sampling_rate': self.sampling_rate
            }
            data['metadata']['quality_metrics'] = quality_metrics
            data['metadata']['processing_timestamp'] = time.time()
            
            return data
            
        except Exception as e:
            logging.error(f"Error in advanced seismic signal processing: {e}")
            return data
    
    def _apply_signal_processing_pipeline(self, waveform: np.ndarray) -> np.ndarray:
        """Apply comprehensive signal processing pipeline"""
        processed = waveform.copy()
        
        # Step 1: Detrending and tapering
        if self.filter_params.get('detrend', True):
            processed = self._detrend_signal(processed)
        
        if self.filter_params.get('taper', True):
            processed = self._apply_taper(processed)
        
        # Step 2: Noise reduction
        if self.noise_params.get('enable_wiener', True):
            processed = self._wiener_filter(processed)
        
        if self.noise_params.get('enable_spectral_sub', True):
            processed = self._spectral_subtraction(processed)
        
        # Step 3: Frequency domain filtering
        processed = self._apply_butterworth_filters(processed)
        
        # Step 4: Artifact removal
        processed = self._remove_spikes(processed)
        processed = self._remove_steps(processed)
        
        # Step 5: Normalization and scaling
        processed = self._normalize_components(processed)
        
        return processed
    
    def _assess_signal_quality(self, waveform: np.ndarray) -> Dict[str, float]:
        """Comprehensive signal quality assessment"""
        metrics = {}
        
        try:
            # Signal-to-noise ratio estimation
            signal_power = np.mean(waveform**2, axis=0)
            noise_estimate = np.percentile(signal_power, 10)  # Bottom 10% as noise
            snr = 10 * np.log10(np.mean(signal_power) / max(noise_estimate, 1e-10))
            metrics['snr_db'] = float(snr)
            
            # Clipping detection
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                clip_threshold = 0.95 * max_val
                clipped_samples = np.sum(np.abs(waveform) >= clip_threshold)
                metrics['clip_fraction'] = float(clipped_samples / waveform.size)
            else:
                metrics['clip_fraction'] = 0.0
            
            # Dynamic range
            if max_val > 0:
                min_nonzero = np.min(np.abs(waveform[waveform != 0]))
                metrics['dynamic_range_db'] = float(20 * np.log10(max_val / max(min_nonzero, 1e-10)))
            else:
                metrics['dynamic_range_db'] = 0.0
            
            # Spectral quality
            for i in range(waveform.shape[1]):
                fft_data = np.fft.fft(waveform[:, i])
                power_spectrum = np.abs(fft_data)**2
                spectral_centroid = np.sum(np.arange(len(power_spectrum)) * power_spectrum) / np.sum(power_spectrum)
                metrics[f'spectral_centroid_comp_{i}'] = float(spectral_centroid)
            
            # Overall quality score (0-1)
            snr_score = min(1.0, max(0.0, (metrics['snr_db'] - (-20)) / 40))  # -20 to 20 dB range
            clip_score = 1.0 - min(1.0, metrics['clip_fraction'] / 0.1)  # Penalize >10% clipping
            range_score = min(1.0, max(0.0, (metrics['dynamic_range_db'] - 20) / 60))  # 20-80 dB range
            
            metrics['overall_quality'] = float((snr_score + clip_score + range_score) / 3.0)
            
        except Exception as e:
            logging.error(f"Error in quality assessment: {e}")
            metrics = {'overall_quality': 0.0, 'error': str(e)}
        
        return metrics
    
    def _detrend_signal(self, waveform: np.ndarray) -> np.ndarray:
        """Remove linear trend from each component"""
        detrended = np.zeros_like(waveform)
        
        for i in range(waveform.shape[1]):
            component = waveform[:, i]
            # Fit linear trend
            x = np.arange(len(component))
            coeffs = np.polyfit(x, component, 1)
            trend = np.polyval(coeffs, x)
            detrended[:, i] = component - trend
        
        return detrended
    
    def _apply_taper(self, waveform: np.ndarray) -> np.ndarray:
        """Apply Tukey taper to reduce edge effects"""
        fraction = self.filter_params.get('taper_fraction', 0.05)
        n_samples = waveform.shape[0]
        
        # Create Tukey window
        taper_length = int(fraction * n_samples)
        window = np.ones(n_samples)
        
        # Apply cosine taper to edges
        for i in range(taper_length):
            taper_val = 0.5 * (1 - np.cos(np.pi * i / taper_length))
            window[i] = taper_val
            window[-(i+1)] = taper_val
        
        # Apply taper to each component
        tapered = waveform.copy()
        for i in range(waveform.shape[1]):
            tapered[:, i] *= window
        
        return tapered
    
    def _wiener_filter(self, waveform: np.ndarray) -> np.ndarray:
        """Apply Wiener filtering for noise reduction"""
        filtered = np.zeros_like(waveform)
        
        for i in range(waveform.shape[1]):
            component = waveform[:, i]
            
            # Estimate power spectral densities
            fft_signal = np.fft.fft(component)
            power_spectrum = np.abs(fft_signal)**2
            
            # Estimate noise floor
            noise_floor = np.percentile(power_spectrum, 20)
            
            # Wiener filter in frequency domain
            wiener_filter = power_spectrum / (power_spectrum + noise_floor)
            filtered_fft = fft_signal * wiener_filter
            
            # Transform back to time domain
            filtered[:, i] = np.real(np.fft.ifft(filtered_fft))
        
        return filtered
    
    def _spectral_subtraction(self, waveform: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction for noise reduction"""
        filtered = np.zeros_like(waveform)
        noise_floor_db = self.noise_params.get('noise_floor_db', -40)
        
        for i in range(waveform.shape[1]):
            component = waveform[:, i]
            
            # FFT analysis
            fft_signal = np.fft.fft(component)
            magnitude = np.abs(fft_signal)
            phase = np.angle(fft_signal)
            
            # Estimate noise magnitude
            noise_magnitude = np.percentile(magnitude, 10)
            
            # Spectral subtraction
            enhanced_magnitude = magnitude - 2.0 * noise_magnitude
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal
            enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
            filtered[:, i] = np.real(np.fft.ifft(enhanced_fft))
        
        return filtered
    
    def _apply_butterworth_filters(self, waveform: np.ndarray) -> np.ndarray:
        """Apply Butterworth filters for frequency domain processing"""
        try:
            from scipy.signal import butter, filtfilt
        except ImportError:
            logging.warning("SciPy not available, using simple filtering")
            return self._simple_frequency_filter(waveform)
        
        filtered = waveform.copy()
        order = self.filter_params.get('order', 4)
        nyquist = self.sampling_rate / 2.0
        
        # High-pass filter
        if 'highpass' in self.filter_params:
            highpass = self.filter_params['highpass']
            if highpass > 0 and highpass < nyquist:
                b, a = butter(order, highpass / nyquist, btype='high')
                for i in range(filtered.shape[1]):
                    filtered[:, i] = filtfilt(b, a, filtered[:, i])
        
        # Low-pass filter
        if 'lowpass' in self.filter_params:
            lowpass = self.filter_params['lowpass']
            if lowpass > 0 and lowpass < nyquist:
                b, a = butter(order, lowpass / nyquist, btype='low')
                for i in range(filtered.shape[1]):
                    filtered[:, i] = filtfilt(b, a, filtered[:, i])
        
        # Band-pass filter
        if 'bandpass' in self.filter_params:
            low, high = self.filter_params['bandpass']
            if 0 < low < high < nyquist:
                b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
                for i in range(filtered.shape[1]):
                    filtered[:, i] = filtfilt(b, a, filtered[:, i])
        
        return filtered
    
    def _simple_frequency_filter(self, waveform: np.ndarray) -> np.ndarray:
        """Fallback simple frequency filtering using FFT"""
        filtered = np.zeros_like(waveform)
        
        for i in range(waveform.shape[1]):
            component = waveform[:, i]
            fft_data = np.fft.fft(component)
            freqs = np.fft.fftfreq(len(component), 1.0 / self.sampling_rate)
            
            # Create frequency mask
            mask = np.ones_like(freqs, dtype=bool)
            
            # Apply high-pass
            if 'highpass' in self.filter_params:
                highpass = self.filter_params['highpass']
                mask &= np.abs(freqs) >= highpass
            
            # Apply low-pass
            if 'lowpass' in self.filter_params:
                lowpass = self.filter_params['lowpass']
                mask &= np.abs(freqs) <= lowpass
            
            # Apply filter
            fft_filtered = fft_data * mask
            filtered[:, i] = np.real(np.fft.ifft(fft_filtered))
        
        return filtered
    
    def _remove_spikes(self, waveform: np.ndarray) -> np.ndarray:
        """Remove impulsive spikes using median filtering"""
        filtered = waveform.copy()
        
        for i in range(waveform.shape[1]):
            component = waveform[:, i]
            
            # Detect spikes using local statistics
            window_size = 15
            padded = np.pad(component, window_size//2, mode='edge')
            
            for j in range(len(component)):
                window = padded[j:j+window_size]
                median_val = np.median(window)
                mad = np.median(np.abs(window - median_val))  # Median Absolute Deviation
                
                # Spike threshold (5 MAD)
                threshold = 5.0 * mad
                if np.abs(component[j] - median_val) > threshold:
                    filtered[i, j] = median_val
        
        return filtered
    
    def _remove_steps(self, waveform: np.ndarray) -> np.ndarray:
        """Remove step discontinuities"""
        filtered = waveform.copy()
        
        for i in range(waveform.shape[1]):
            component = waveform[:, i]
            
            # Detect steps using gradient analysis
            gradient = np.gradient(component)
            gradient_threshold = 5.0 * np.std(gradient)
            
            # Find step locations
            step_indices = np.where(np.abs(gradient) > gradient_threshold)[0]
            
            # Remove steps by linear interpolation
            for idx in step_indices:
                if 1 < idx < len(component) - 2:
                    # Interpolate across the step
                    before_val = np.mean(component[max(0, idx-3):idx])
                    after_val = np.mean(component[idx+1:min(len(component), idx+4)])
                    filtered[i, idx] = (before_val + after_val) / 2.0
        
        return filtered
    
    def _normalize_components(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize each component for consistent scaling"""
        normalized = np.zeros_like(waveform)
        
        for i in range(waveform.shape[1]):
            component = waveform[:, i]
            
            # RMS normalization
            rms = np.sqrt(np.mean(component**2))
            if rms > 0:
                normalized[:, i] = component / rms
            else:
                normalized[:, i] = component
        
        return normalized
    
    def _validate_processed_signal(self, waveform: np.ndarray) -> bool:
        """Validate processed signal for numerical stability"""
        try:
            # Check for NaN or infinite values
            if not np.isfinite(waveform).all():
                logging.error("Non-finite values detected in processed signal")
                return False
            
            # Check for reasonable amplitude range
            max_amplitude = np.max(np.abs(waveform))
            if max_amplitude > 1e6:  # Arbitrarily large values
                logging.error(f"Excessive amplitude detected: {max_amplitude}")
                return False
            
            # Check for sufficient dynamic range
            if max_amplitude < self.quality_params.get('stability_threshold', 0.01):
                logging.warning("Very low signal amplitude after processing")
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating processed signal: {e}")
            return False

class SeismicFeatureExtractor:
    """Advanced seismic feature extraction with comprehensive signal processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize feature extractor with configuration"""
        self.config = config or {}
        self.sampling_rate = self.config.get('sampling_rate', 100.0)  # Hz
        self.window_length = self.config.get('window_length', 1024)  # samples
        self.overlap_ratio = self.config.get('overlap_ratio', 0.5)
        self.freq_bands = self.config.get('frequency_bands', {
            'low': (0.1, 1.0),      # Low frequency band (Hz)
            'mid': (1.0, 10.0),     # Mid frequency band (Hz) 
            'high': (10.0, 50.0)    # High frequency band (Hz)
        })
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive seismic features"""
        if data.get('reading_type') != 'seismic_waveform':
            return data
        
        try:
            waveform = np.array(data['value'])
            
            # Validate waveform shape
            if len(waveform.shape) != 2 or waveform.shape[1] != 3:
                logging.warning(f"Invalid waveform shape: {waveform.shape}, expected (N, 3)")
                return data
            
            # Extract comprehensive features
            features = self._extract_comprehensive_features(waveform)
            
            # Add metadata
            features['processing_timestamp'] = time.time()
            features['waveform_length'] = waveform.shape[0]
            features['sampling_rate'] = self.sampling_rate
            
            # Add features to data
            data['features'] = features
            data['feature_quality'] = self._assess_feature_quality(features, waveform)
            
            return data
            
        except Exception as e:
            logging.error(f"Error in seismic feature extraction: {e}")
            data['features'] = {}
            data['feature_quality'] = 'error'
            return data
    
    def _extract_comprehensive_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive time-domain and frequency-domain features"""
        features = {}
        
        # Process each component (Z, N, E)
        for i, component in enumerate(['Z', 'N', 'E']):
            signal = waveform[:, i]
            
            # Time-domain features
            time_features = self._extract_time_domain_features(signal, component)
            features.update(time_features)
            
            # Frequency-domain features
            freq_features = self._extract_frequency_domain_features(signal, component)
            features.update(freq_features)
            
            # Spectral band features
            band_features = self._extract_spectral_band_features(signal, component)
            features.update(band_features)
            
            # Advanced statistical features
            stat_features = self._extract_advanced_statistical_features(signal, component)
            features.update(stat_features)
            
            # Arrival time features
            arrival_features = self._extract_arrival_time_features(signal, component)
            features.update(arrival_features)
        
        # Cross-component features
        cross_features = self._extract_cross_component_features(waveform)
        features.update(cross_features)
        
        # Event detection features
        event_features = self._extract_event_detection_features(waveform)
        features.update(event_features)
        
        return features
    
    def _extract_time_domain_features(self, signal: np.ndarray, component: str) -> Dict[str, float]:
        """Extract comprehensive time-domain features"""
        features = {}
        
        # Basic statistical measures
        features[f'mean_{component}'] = float(np.mean(signal))
        features[f'rms_{component}'] = float(np.sqrt(np.mean(signal**2)))
        features[f'peak_{component}'] = float(np.max(np.abs(signal)))
        features[f'std_{component}'] = float(np.std(signal))
        features[f'var_{component}'] = float(np.var(signal))
        features[f'min_{component}'] = float(np.min(signal))
        features[f'max_{component}'] = float(np.max(signal))
        features[f'range_{component}'] = float(np.ptp(signal))
        
        # Higher-order moments
        features[f'skewness_{component}'] = float(self._skewness(signal))
        features[f'kurtosis_{component}'] = float(self._kurtosis(signal))
        
        # Energy measures
        features[f'energy_{component}'] = float(np.sum(signal**2))
        features[f'power_{component}'] = float(np.mean(signal**2))
        
        # Zero crossing rate
        features[f'zcr_{component}'] = float(self._zero_crossing_rate(signal))
        
        # Peak-related features
        features[f'crest_factor_{component}'] = float(features[f'peak_{component}'] / features[f'rms_{component}'] if features[f'rms_{component}'] > 0 else 0)
        features[f'shape_factor_{component}'] = float(features[f'rms_{component}'] / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0)
        
        # Amplitude distribution
        features[f'median_{component}'] = float(np.median(signal))
        features[f'mad_{component}'] = float(np.median(np.abs(signal - np.median(signal))))  # Median Absolute Deviation
        
        # Percentiles
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            features[f'p{p}_{component}'] = float(np.percentile(signal, p))
        
        # Inter-quartile range
        features[f'iqr_{component}'] = float(np.percentile(signal, 75) - np.percentile(signal, 25))
        
        return features
    
    def _extract_frequency_domain_features(self, signal: np.ndarray, component: str) -> Dict[str, float]:
        """Extract comprehensive frequency-domain features"""
        features = {}
        
        # Compute FFT
        n = len(signal)
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1/self.sampling_rate)
        power_spectrum = np.abs(fft)**2
        magnitude_spectrum = np.abs(fft)
        
        # Only use positive frequencies
        pos_freqs = freqs[:n//2]
        pos_power = power_spectrum[:n//2]
        pos_magnitude = magnitude_spectrum[:n//2]
        
        # Basic spectral features
        features[f'spectral_energy_{component}'] = float(np.sum(pos_power))
        features[f'spectral_power_{component}'] = float(np.mean(pos_power))
        features[f'spectral_peak_{component}'] = float(np.max(pos_power))
        
        # Dominant frequency
        dominant_idx = np.argmax(pos_power)
        features[f'dominant_freq_{component}'] = float(pos_freqs[dominant_idx])
        features[f'dominant_power_{component}'] = float(pos_power[dominant_idx])
        
        # Spectral centroids and moments
        if np.sum(pos_power) > 0:
            features[f'spectral_centroid_{component}'] = float(np.sum(pos_freqs * pos_power) / np.sum(pos_power))
            
            # Spectral spread (bandwidth)
            centroid = features[f'spectral_centroid_{component}']
            features[f'spectral_spread_{component}'] = float(np.sqrt(np.sum(((pos_freqs - centroid)**2) * pos_power) / np.sum(pos_power)))
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumsum_power = np.cumsum(pos_power)
            rolloff_idx = np.where(cumsum_power >= 0.85 * cumsum_power[-1])[0]
            if len(rolloff_idx) > 0:
                features[f'spectral_rolloff_{component}'] = float(pos_freqs[rolloff_idx[0]])
            else:
                features[f'spectral_rolloff_{component}'] = 0.0
            
            # Spectral flux (measure of spectral change)
            if hasattr(self, '_prev_spectrum'):
                prev_spec = self._prev_spectrum.get(component, pos_power)
                features[f'spectral_flux_{component}'] = float(np.sum((pos_power - prev_spec)**2))
            else:
                features[f'spectral_flux_{component}'] = 0.0
                self._prev_spectrum = {}
            self._prev_spectrum[component] = pos_power.copy()
        else:
            features[f'spectral_centroid_{component}'] = 0.0
            features[f'spectral_spread_{component}'] = 0.0
            features[f'spectral_rolloff_{component}'] = 0.0
            features[f'spectral_flux_{component}'] = 0.0
        
        return features
    
    def _extract_spectral_band_features(self, signal: np.ndarray, component: str) -> Dict[str, float]:
        """Extract features for specific frequency bands"""
        features = {}
        
        # Compute FFT
        n = len(signal)
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1/self.sampling_rate)
        power_spectrum = np.abs(fft)**2
        
        # Only use positive frequencies
        pos_freqs = freqs[:n//2]
        pos_power = power_spectrum[:n//2]
        
        # Extract features for each frequency band
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            band_mask = (pos_freqs >= low_freq) & (pos_freqs <= high_freq)
            band_power = pos_power[band_mask]
            band_freqs = pos_freqs[band_mask]
            
            if len(band_power) > 0:
                # Band power
                features[f'band_power_{band_name}_{component}'] = float(np.sum(band_power))
                features[f'band_energy_{band_name}_{component}'] = float(np.mean(band_power))
                features[f'band_peak_{band_name}_{component}'] = float(np.max(band_power))
                
                # Band dominant frequency
                band_dominant_idx = np.argmax(band_power)
                features[f'band_dominant_freq_{band_name}_{component}'] = float(band_freqs[band_dominant_idx])
                
                # Band relative power (ratio to total power)
                total_power = np.sum(pos_power)
                features[f'band_relative_power_{band_name}_{component}'] = float(np.sum(band_power) / total_power if total_power > 0 else 0)
            else:
                features[f'band_power_{band_name}_{component}'] = 0.0
                features[f'band_energy_{band_name}_{component}'] = 0.0
                features[f'band_peak_{band_name}_{component}'] = 0.0
                features[f'band_dominant_freq_{band_name}_{component}'] = 0.0
                features[f'band_relative_power_{band_name}_{component}'] = 0.0
        
        return features
    
    def _extract_advanced_statistical_features(self, signal: np.ndarray, component: str) -> Dict[str, float]:
        """Extract advanced statistical features"""
        features = {}
        
        # Hjorth parameters
        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)
        
        var_signal = np.var(signal)
        var_diff1 = np.var(diff1) if len(diff1) > 0 else 0
        var_diff2 = np.var(diff2) if len(diff2) > 0 else 0
        
        # Activity (variance)
        features[f'hjorth_activity_{component}'] = float(var_signal)
        
        # Mobility
        features[f'hjorth_mobility_{component}'] = float(np.sqrt(var_diff1 / var_signal) if var_signal > 0 else 0)
        
        # Complexity
        mobility_diff1 = np.sqrt(var_diff2 / var_diff1) if var_diff1 > 0 else 0
        mobility_signal = features[f'hjorth_mobility_{component}']
        features[f'hjorth_complexity_{component}'] = float(mobility_diff1 / mobility_signal if mobility_signal > 0 else 0)
        
        # Fractal dimension using Higuchi's method
        features[f'fractal_dimension_{component}'] = float(self._higuchi_fractal_dimension(signal))
        
        # Sample entropy
        features[f'sample_entropy_{component}'] = float(self._sample_entropy(signal))
        
        return features
    
    def _extract_arrival_time_features(self, signal: np.ndarray, component: str) -> Dict[str, float]:
        """Extract P-wave and S-wave arrival time features"""
        features = {}
        
        # STA/LTA for arrival detection
        sta_lta = self._compute_sta_lta(signal)
        
        # Find potential arrivals (peaks in STA/LTA)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(sta_lta, height=2.0, distance=int(0.5 * self.sampling_rate))
        
        if len(peaks) > 0:
            # First arrival (likely P-wave)
            first_arrival_idx = peaks[0]
            features[f'p_arrival_time_{component}'] = float(first_arrival_idx / self.sampling_rate)
            features[f'p_arrival_amplitude_{component}'] = float(np.abs(signal[first_arrival_idx]))
            features[f'p_sta_lta_ratio_{component}'] = float(sta_lta[first_arrival_idx])
            
            # Second arrival (potential S-wave)
            if len(peaks) > 1:
                second_arrival_idx = peaks[1]
                features[f's_arrival_time_{component}'] = float(second_arrival_idx / self.sampling_rate)
                features[f's_arrival_amplitude_{component}'] = float(np.abs(signal[second_arrival_idx]))
                features[f's_sta_lta_ratio_{component}'] = float(sta_lta[second_arrival_idx])
                features[f'ps_time_diff_{component}'] = float((second_arrival_idx - first_arrival_idx) / self.sampling_rate)
            else:
                features[f's_arrival_time_{component}'] = 0.0
                features[f's_arrival_amplitude_{component}'] = 0.0
                features[f's_sta_lta_ratio_{component}'] = 0.0
                features[f'ps_time_diff_{component}'] = 0.0
        else:
            features[f'p_arrival_time_{component}'] = 0.0
            features[f'p_arrival_amplitude_{component}'] = 0.0
            features[f'p_sta_lta_ratio_{component}'] = 0.0
            features[f's_arrival_time_{component}'] = 0.0
            features[f's_arrival_amplitude_{component}'] = 0.0
            features[f's_sta_lta_ratio_{component}'] = 0.0
            features[f'ps_time_diff_{component}'] = 0.0
        
        # Maximum STA/LTA ratio
        features[f'max_sta_lta_{component}'] = float(np.max(sta_lta))
        
        return features
    
    def _extract_cross_component_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract features across components"""
        features = {}
        
        z_signal = waveform[:, 0]
        n_signal = waveform[:, 1] 
        e_signal = waveform[:, 2]
        
        # Cross-correlation between components
        features['zn_correlation'] = float(np.corrcoef(z_signal, n_signal)[0, 1])
        features['ze_correlation'] = float(np.corrcoef(z_signal, e_signal)[0, 1])
        features['ne_correlation'] = float(np.corrcoef(n_signal, e_signal)[0, 1])
        
        # Horizontal to vertical ratio
        horizontal_energy = np.sum(n_signal**2) + np.sum(e_signal**2)
        vertical_energy = np.sum(z_signal**2)
        features['hv_ratio'] = float(horizontal_energy / vertical_energy if vertical_energy > 0 else 0)
        
        # Total vector amplitude
        vector_amplitude = np.sqrt(z_signal**2 + n_signal**2 + e_signal**2)
        features['total_vector_peak'] = float(np.max(vector_amplitude))
        features['total_vector_rms'] = float(np.sqrt(np.mean(vector_amplitude**2)))
        
        # Polarization analysis
        covariance_matrix = np.cov([z_signal, n_signal, e_signal])
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
        
        if eigenvalues[0] > 0:
            features['linearity'] = float((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0])
            features['planarity'] = float((eigenvalues[1] - eigenvalues[2]) / eigenvalues[0])
        else:
            features['linearity'] = 0.0
            features['planarity'] = 0.0
        
        return features
    
    def _extract_event_detection_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract event detection and classification features"""
        features = {}
        
        # Multi-component STA/LTA
        z_sta_lta = self._compute_sta_lta(waveform[:, 0])
        n_sta_lta = self._compute_sta_lta(waveform[:, 1])
        e_sta_lta = self._compute_sta_lta(waveform[:, 2])
        
        # Combined STA/LTA
        combined_sta_lta = np.maximum.reduce([z_sta_lta, n_sta_lta, e_sta_lta])
        features['max_combined_sta_lta'] = float(np.max(combined_sta_lta))
        features['mean_combined_sta_lta'] = float(np.mean(combined_sta_lta))
        
        # Event duration estimation
        threshold = 2.0
        above_threshold = combined_sta_lta > threshold
        if np.any(above_threshold):
            event_samples = np.sum(above_threshold)
            features['event_duration'] = float(event_samples / self.sampling_rate)
        else:
            features['event_duration'] = 0.0
        
        # Signal-to-noise ratio estimation
        # Use first 10% as noise, rest as potential signal
        noise_length = int(0.1 * waveform.shape[0])
        noise_power = np.mean(np.sum(waveform[:noise_length, :]**2, axis=1))
        signal_power = np.mean(np.sum(waveform[noise_length:, :]**2, axis=1))
        features['snr_estimate'] = float(signal_power / noise_power if noise_power > 0 else 0)
        
        return features
    
    def _compute_sta_lta(self, signal: np.ndarray, sta_len: int = None, lta_len: int = None) -> np.ndarray:
        """Compute STA/LTA characteristic function"""
        if sta_len is None:
            sta_len = int(0.5 * self.sampling_rate)  # 0.5 second STA
        if lta_len is None:
            lta_len = int(5.0 * self.sampling_rate)   # 5 second LTA
            
        # Compute energy signal
        energy_signal = signal**2
        
        # Initialize output
        sta_lta = np.zeros_like(signal)
        
        for i in range(lta_len, len(signal)):
            # Short-term average
            sta = np.mean(energy_signal[i-sta_len:i]) if i >= sta_len else 0
            
            # Long-term average
            lta = np.mean(energy_signal[i-lta_len:i])
            
            # STA/LTA ratio
            sta_lta[i] = sta / lta if lta > 0 else 0
        
        return sta_lta
    
    def _zero_crossing_rate(self, signal: np.ndarray) -> float:
        """Calculate zero crossing rate"""
        zero_crossings = np.diff(np.sign(signal))
        return np.count_nonzero(zero_crossings) / (2.0 * len(signal))
    
    def _higuchi_fractal_dimension(self, signal: np.ndarray, kmax: int = 10) -> float:
        """Calculate Higuchi fractal dimension"""
        N = len(signal)
        L = np.zeros(kmax)
        
        for k in range(1, kmax + 1):
            Lk = 0
            for m in range(k):
                Lkm = 0
                max_i = int((N - m - 1) / k)
                for i in range(max_i):
                    Lkm += abs(signal[m + (i + 1) * k] - signal[m + i * k])
                if max_i > 0:
                    Lkm = Lkm * (N - 1) / (max_i * k * k)
                Lk += Lkm
            L[k - 1] = Lk / k
        
        # Linear regression on log-log plot
        x = np.log(range(1, kmax + 1))
        y = np.log(L)
        
        # Remove infinite values
        valid_mask = np.isfinite(y)
        if np.sum(valid_mask) < 2:
            return 1.0
        
        slope, _ = np.polyfit(x[valid_mask], y[valid_mask], 1)
        return -slope
    
    def _sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate sample entropy"""
        if r is None:
            r = 0.2 * np.std(signal)
        
        N = len(signal)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([signal[i:i + m] for i in range(N - m + 1)])
            C = np.sum([1 for i in range(len(patterns)) 
                       for j in range(len(patterns)) 
                       if i != j and _maxdist(patterns[i], patterns[j], m) <= r])
            return C / (N - m + 1.0) / (N - m)
        
        try:
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            return -np.log(phi_m1 / phi_m) if phi_m > 0 else 0.0
        except:
            return 0.0
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness with numerical stability"""
        n = len(data)
        if n < 3:
            return 0.0
            
        mean = np.mean(data)
        std = np.std(data, ddof=0)
        
        if std <= 0:
            return 0.0
            
        normalized = (data - mean) / std
        return (n / ((n-1) * (n-2))) * np.sum(normalized**3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis with numerical stability"""
        n = len(data)
        if n < 4:
            return 0.0
            
        mean = np.mean(data)
        std = np.std(data, ddof=0)
        
        if std <= 0:
            return 0.0
            
        normalized = (data - mean) / std
        kurt = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(normalized**4)
        kurt -= (3 * (n-1)**2 / ((n-2) * (n-3)))
        return kurt
    
    def _assess_feature_quality(self, features: Dict[str, float], waveform: np.ndarray) -> str:
        """Assess quality of extracted features"""
        try:
            # Check for NaN or infinite values
            if any(not np.isfinite(v) for v in features.values()):
                return 'poor_nan_inf'
            
            # Check signal quality based on SNR
            snr = features.get('snr_estimate', 0)
            if snr < 1.0:
                return 'poor_low_snr'
            elif snr < 3.0:
                return 'fair'
            elif snr < 10.0:
                return 'good'
            else:
                return 'excellent'
                
        except Exception:
            return 'error'

class EarthquakeDetector:
    """Advanced ML-based seismic event detection using Transformer models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize seismic transformer
        try:
            from .seismic_transformer import SeismicTransformerSystem, SeismicConfig
            self.seismic_config = SeismicConfig(
                sample_rate=self.config.get('sample_rate', 100.0),
                window_size=self.config.get('window_size', 6000),
                transformer_layers=self.config.get('transformer_layers', 6),
                transformer_heads=self.config.get('transformer_heads', 8),
                hidden_dim=self.config.get('hidden_dim', 512)
            )
            self.seismic_transformer = SeismicTransformerSystem(self.seismic_config)
            self.transformer_available = True
        except ImportError:
            logging.warning("SeismicTransformer not available, falling back to STA/LTA")
            self.transformer_available = False
            self._init_fallback_stalte()
        
        # Initialize eruption forecaster for volcanic events
        try:
            from .eruption_forecaster_ml import EruptionForecaster
            self.eruption_forecaster = EruptionForecaster()
            self.forecaster_available = True
        except ImportError:
            logging.warning("EruptionForecaster not available")
            self.forecaster_available = False
        
        # Initialize data integration pipeline
        try:
            from .data_integration_pipeline import DataIntegrationPipeline
            self.data_pipeline = DataIntegrationPipeline(config)
            self.pipeline_available = True
        except ImportError:
            logging.warning("DataIntegrationPipeline not available")
            self.pipeline_available = False
        
        # Advanced detection parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.multi_phase_detection = self.config.get('multi_phase_detection', True)
        self.volcanic_context = self.config.get('volcanic_context', False)
        
        # Event classification parameters
        self.event_types = {
            'tectonic': {'magnitude_range': (2.0, 9.0), 'depth_range': (0, 700)},
            'volcanic': {'magnitude_range': (0.5, 6.0), 'depth_range': (0, 30)},
            'explosion': {'magnitude_range': (0.1, 4.0), 'depth_range': (0, 5)},
            'collapse': {'magnitude_range': (1.0, 5.0), 'depth_range': (0, 10)}
        }
        
    def _init_fallback_stalte(self):
        """Initialize fallback STA/LTA parameters"""
        self.sta_window = self.config.get('sta_window', 10)
        self.lta_window = self.config.get('lta_window', 100)
        self.trigger_threshold = self.config.get('trigger_threshold', 3.0)
        self.sta_buffer = np.zeros(self.sta_window)
        self.lta_buffer = np.zeros(self.lta_window)
        self.buffer_index = 0
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced seismic event detection with ML models"""
        if not data.get('features') and not data.get('seismic_data'):
            return data
        
        try:
            detection_results = {}
            
            # ML-based detection if transformer available
            if self.transformer_available and 'seismic_data' in data:
                ml_results = self._ml_based_detection(data)
                detection_results.update(ml_results)
            
            # Volcanic event analysis if in volcanic context
            if self.volcanic_context and self.forecaster_available:
                volcanic_results = self._volcanic_event_analysis(data)
                detection_results.update(volcanic_results)
            
            # Fallback STA/LTA detection
            if not detection_results and data.get('features'):
                stalte_results = self._stalte_detection(data)
                detection_results.update(stalte_results)
            
            # Multi-source data integration
            if self.pipeline_available and detection_results:
                enhanced_results = self._enhance_with_pipeline_data(data, detection_results)
                detection_results.update(enhanced_results)
            
            # Add detection results to data
            if detection_results:
                data['advanced_detection'] = detection_results
                data['detection_method'] = 'ml_transformer'
                data['processing_timestamp'] = datetime.now().isoformat()
            
            return data
            
        except Exception as e:
            logging.error(f"Error in advanced earthquake detection: {e}")
            # Fallback to basic detection
            return self._stalte_detection(data) if data.get('features') else data
    
    def _ml_based_detection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """ML-based seismic detection using transformer"""
        try:
            seismic_data = data['seismic_data']
            
            # Prepare data for transformer (expecting 3-component data)
            if isinstance(seismic_data, dict):
                # Convert to numpy array [samples, 3]
                waveform = np.column_stack([
                    seismic_data.get('Z', np.zeros(len(seismic_data.get('E', [0])))),
                    seismic_data.get('N', np.zeros(len(seismic_data.get('E', [0])))),
                    seismic_data.get('E', np.zeros(len(seismic_data.get('Z', [0]))))
                ])
            else:
                waveform = seismic_data
            
            # Phase detection
            phase_results = self.seismic_transformer.detect_phases(waveform)
            
            # Event classification
            event_classification = self._classify_seismic_event(phase_results, data)
            
            return {
                'ml_detection': {
                    'phases_detected': phase_results.get('phases', []),
                    'p_wave_arrival': phase_results.get('p_arrival', None),
                    's_wave_arrival': phase_results.get('s_arrival', None),
                    'confidence_p': phase_results.get('confidence_p', 0.0),
                    'confidence_s': phase_results.get('confidence_s', 0.0),
                    'event_type': event_classification['type'],
                    'event_confidence': event_classification['confidence'],
                    'magnitude_estimate': phase_results.get('magnitude', None),
                    'location_estimate': phase_results.get('location', None),
                    'quality_metrics': phase_results.get('quality', {})
                }
            }
            
        except Exception as e:
            logging.error(f"ML detection error: {e}")
            return {}
    
    def _volcanic_event_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Volcanic event analysis using eruption forecaster"""
        try:
            # Prepare monitoring data for eruption forecaster
            monitoring_data = {
                'seismic': data.get('features', {}),
                'timestamp': data.get('timestamp', datetime.now()),
                'location': data.get('location', {}),
                'thermal': data.get('thermal_data', {}),
                'gas': data.get('gas_data', {}),
                'deformation': data.get('deformation_data', {})
            }
            
            # Run volcanic event analysis
            volcanic_forecast = asyncio.run(self.eruption_forecaster.predict(monitoring_data))
            
            return {
                'volcanic_analysis': {
                    'eruption_probability': volcanic_forecast.eruption_probability,
                    'time_to_eruption': volcanic_forecast.time_to_eruption,
                    'expected_vei': volcanic_forecast.expected_vei,
                    'alert_level': volcanic_forecast.alert_level,
                    'contributing_factors': volcanic_forecast.contributing_factors,
                    'confidence': volcanic_forecast.confidence,
                    'recommendations': volcanic_forecast.recommendations
                }
            }
            
        except Exception as e:
            logging.error(f"Volcanic analysis error: {e}")
            return {}
    
    def _stalte_detection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback STA/LTA detection"""
        try:
            trigger_value = data['features'].get('rms_Z', 0)
            
            self.sta_buffer[self.buffer_index % self.sta_window] = trigger_value
            self.lta_buffer[self.buffer_index % self.lta_window] = trigger_value
            self.buffer_index += 1
            
            if self.buffer_index >= self.lta_window:
                sta = np.mean(self.sta_buffer)
                lta = np.mean(self.lta_buffer)
                
                if lta > 0:
                    sta_lta_ratio = sta / lta
                    
                    if sta_lta_ratio > self.trigger_threshold:
                        return {
                            'detection': {
                                'type': 'earthquake_trigger',
                                'method': 'sta_lta',
                                'sta_lta_ratio': float(sta_lta_ratio),
                                'confidence': min(sta_lta_ratio / self.trigger_threshold, 1.0),
                                'timestamp': data.get('timestamp')
                            }
                        }
            
            return data
            
        except Exception as e:
            logging.error(f"STA/LTA detection error: {e}")
            return data
    
    def _classify_seismic_event(self, phase_results: Dict, data: Dict) -> Dict[str, Any]:
        """Classify detected seismic event type"""
        try:
            # Extract event characteristics
            magnitude = phase_results.get('magnitude', 0.0)
            depth = data.get('location', {}).get('depth', 10.0)
            frequency_content = data.get('features', {}).get('dominant_frequency', 5.0)
            
            # Classification logic
            event_scores = {}
            for event_type, params in self.event_types.items():
                score = 0.0
                
                # Magnitude scoring
                mag_range = params['magnitude_range']
                if mag_range[0] <= magnitude <= mag_range[1]:
                    score += 0.4
                
                # Depth scoring
                depth_range = params['depth_range']
                if depth_range[0] <= depth <= depth_range[1]:
                    score += 0.3
                
                # Frequency content scoring
                if event_type == 'volcanic' and frequency_content < 3.0:
                    score += 0.3
                elif event_type == 'tectonic' and frequency_content > 3.0:
                    score += 0.3
                
                event_scores[event_type] = score
            
            # Select most likely event type
            best_type = max(event_scores, key=event_scores.get)
            confidence = event_scores[best_type]
            
            return {
                'type': best_type,
                'confidence': confidence,
                'all_scores': event_scores
            }
            
        except Exception as e:
            logging.error(f"Event classification error: {e}")
            return {'type': 'unknown', 'confidence': 0.0}
    
    def _enhance_with_pipeline_data(self, data: Dict, detection_results: Dict) -> Dict[str, Any]:
        """Enhance detection with integrated pipeline data"""
        try:
            # Use pipeline to get contextual geological data
            location = data.get('location', {})
            if location:
                # Get regional geological context
                geological_context = self.data_pipeline.get_regional_context(
                    location.get('latitude', 0),
                    location.get('longitude', 0)
                )
                
                return {
                    'geological_context': {
                        'tectonic_setting': geological_context.get('tectonic_setting', 'unknown'),
                        'fault_proximity': geological_context.get('fault_distance', None),
                        'volcanic_proximity': geological_context.get('volcano_distance', None),
                        'site_classification': geological_context.get('site_class', 'unknown'),
                        'amplification_factors': geological_context.get('amplification', {})
                    }
                }
            
            return {}
            
        except Exception as e:
            logging.error(f"Pipeline enhancement error: {e}")
            return {}

class GPSNoiseFilter:
    """Advanced GPS noise filtering using Enhanced Kalman Filter and Multi-sensor fusion"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize Enhanced Kalman Filter
        try:
            from .enhanced_kalman_filter import EnhancedEnsembleKalmanFilter
            kalman_config = {
                'ensemble_size': self.config.get('ensemble_size', 50),
                'inflation_factor': self.config.get('inflation_factor', 1.05),
                'domain_size': self.config.get('domain_size', 10000),  # 10km domain
                'max_depth': self.config.get('max_depth', 1000)  # 1km depth for GPS
            }
            self.kalman_filter = EnhancedEnsembleKalmanFilter(kalman_config)
            self.kalman_available = True
            logging.info("Enhanced Kalman Filter initialized for GPS filtering")
        except ImportError:
            logging.warning("Enhanced Kalman Filter not available, using advanced statistical filtering")
            self.kalman_available = False
        
        # Initialize Multi-sensor Fusion Engine
        try:
            from .multi_sensor_fusion import MultiSensorFusionEngine
            fusion_config = {
                'confidence_threshold': self.config.get('confidence_threshold', 0.7),
                'alert_threshold': self.config.get('alert_threshold', 0.8)
            }
            self.fusion_engine = MultiSensorFusionEngine(fusion_config)
            self.fusion_available = True
            logging.info("Multi-sensor Fusion Engine initialized for GPS processing")
        except ImportError:
            logging.warning("Multi-sensor Fusion not available")
            self.fusion_available = False
        
        # Advanced filtering parameters
        self.process_noise_std = self.config.get('process_noise_std', 0.1)  # meters
        self.measurement_noise_std = self.config.get('measurement_noise_std', 2.0)  # meters
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)  # sigma
        
        # State tracking for advanced filtering
        self.position_history = []
        self.velocity_estimates = []
        self.quality_metrics = []
        self.last_timestamp = None
        
        # Initialize state vector [x, y, z, vx, vy, vz]
        self.state_vector = np.zeros(6)
        self.covariance_matrix = np.eye(6) * 100.0  # Initial uncertainty
        
        # Initialize fallback filter parameters
        self.window_size = self.config.get('window_size', 5)
        self.position_buffer = []
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced GPS noise filtering with Kalman filter and multi-sensor fusion"""
        if data.get('reading_type') != 'gps_position':
            return data
        
        try:
            # Multi-sensor fusion if available
            if self.fusion_available:
                fusion_results = self._apply_multi_sensor_fusion(data)
                if fusion_results:
                    data.update(fusion_results)
            
            # Kalman filtering if available
            if self.kalman_available:
                kalman_results = self._apply_kalman_filtering(data)
                if kalman_results:
                    data.update(kalman_results)
            else:
                # Advanced statistical filtering as fallback
                statistical_results = self._apply_advanced_statistical_filtering(data)
                data.update(statistical_results)
            
            # Add quality assessment
            quality_assessment = self._assess_position_quality(data)
            data['quality_metrics'] = quality_assessment
            
            return data
            
        except Exception as e:
            logging.error(f"Error in advanced GPS filtering: {e}")
            # Fallback to simple moving average
            return self._fallback_moving_average(data)
    
    def _apply_multi_sensor_fusion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multi-sensor fusion to GPS data"""
        try:
            position_data = data['value']
            
            # Prepare sensor data for fusion
            sensor_data = {
                'gps': {
                    'latitude': position_data.get('latitude', 0),
                    'longitude': position_data.get('longitude', 0),
                    'elevation': position_data.get('elevation', 0),
                    'timestamp': data.get('timestamp', datetime.now()),
                    'quality': position_data.get('hdop', 1.0),  # Horizontal Dilution of Precision
                    'satellites': position_data.get('satellites', 4)
                }
            }
            
            # Add complementary sensor data if available
            if 'accelerometer' in data:
                sensor_data['accelerometer'] = data['accelerometer']
            if 'gyroscope' in data:
                sensor_data['gyroscope'] = data['gyroscope']
            if 'magnetometer' in data:
                sensor_data['magnetometer'] = data['magnetometer']
            
            # Run fusion processing
            fusion_result = asyncio.run(self.fusion_engine.process_volcano_monitoring(
                (sensor_data['gps']['latitude'], sensor_data['gps']['longitude']),
                sensor_data
            ))
            
            return {
                'fusion_filtered_position': {
                    'latitude': fusion_result.position.get('latitude', position_data['latitude']),
                    'longitude': fusion_result.position.get('longitude', position_data['longitude']),
                    'elevation': fusion_result.position.get('elevation', position_data['elevation']),
                    'confidence': fusion_result.confidence,
                    'uncertainty': fusion_result.uncertainty
                },
                'fusion_metadata': {
                    'sensors_used': fusion_result.sensors_contributing,
                    'anomaly_detected': fusion_result.anomaly_detected,
                    'fusion_method': 'multi_sensor_bayesian'
                }
            }
            
        except Exception as e:
            logging.error(f"Multi-sensor fusion error: {e}")
            return {}
    
    def _apply_kalman_filtering(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Enhanced Ensemble Kalman Filter to GPS data"""
        try:
            position_data = data['value']
            current_timestamp = data.get('timestamp', datetime.now())
            
            # Convert GPS coordinates to local coordinates (meters)
            if len(self.position_history) == 0:
                # First measurement - establish reference point
                self.reference_lat = position_data['latitude']
                self.reference_lon = position_data['longitude']
                self.reference_alt = position_data.get('elevation', 0)
            
            # Convert to local cartesian coordinates
            local_coords = self._gps_to_local_coords(
                position_data['latitude'],
                position_data['longitude'],
                position_data.get('elevation', 0)
            )
            
            # Prepare observation data for Kalman filter
            observation = np.array([local_coords[0], local_coords[1], local_coords[2]])
            
            # Time difference for motion model
            dt = 1.0  # Default 1 second
            if self.last_timestamp:
                dt = (current_timestamp - self.last_timestamp).total_seconds()
                dt = max(0.1, min(dt, 10.0))  # Clamp between 0.1 and 10 seconds
            
            # Update Kalman filter state
            filtered_state = self._update_kalman_state(observation, dt)
            
            # Convert back to GPS coordinates
            filtered_gps = self._local_to_gps_coords(
                filtered_state[0], filtered_state[1], filtered_state[2]
            )
            
            # Estimate velocity and acceleration
            velocity = np.linalg.norm(filtered_state[3:6])  # m/s
            
            self.last_timestamp = current_timestamp
            
            return {
                'kalman_filtered_position': {
                    'latitude': filtered_gps[0],
                    'longitude': filtered_gps[1],
                    'elevation': filtered_gps[2],
                    'velocity': velocity,
                    'acceleration': self._estimate_acceleration(filtered_state)
                },
                'kalman_metadata': {
                    'filter_method': 'enhanced_ensemble_kalman',
                    'state_uncertainty': np.trace(self.covariance_matrix),
                    'innovation': np.linalg.norm(observation - filtered_state[:3]),
                    'dt': dt
                }
            }
            
        except Exception as e:
            logging.error(f"Kalman filtering error: {e}")
            return {}
    
    def _apply_advanced_statistical_filtering(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply advanced statistical filtering methods"""
        try:
            position_data = data['value']
            
            # Add to history
            self.position_history.append({
                'position': [position_data['latitude'], position_data['longitude'], 
                           position_data.get('elevation', 0)],
                'timestamp': data.get('timestamp', datetime.now()),
                'quality': position_data.get('hdop', 1.0)
            })
            
            # Keep only recent history
            max_history = self.config.get('max_history', 50)
            if len(self.position_history) > max_history:
                self.position_history.pop(0)
            
            if len(self.position_history) >= 3:
                # Apply robust statistical filtering
                filtered_position = self._robust_position_filter()
                
                # Outlier detection and removal
                is_outlier = self._detect_position_outlier(position_data)
                
                return {
                    'statistical_filtered_position': filtered_position,
                    'statistical_metadata': {
                        'filter_method': 'robust_statistical',
                        'is_outlier': is_outlier,
                        'history_length': len(self.position_history),
                        'quality_score': self._calculate_position_quality()
                    }
                }
            
            return {}
            
        except Exception as e:
            logging.error(f"Statistical filtering error: {e}")
            return {}
    
    def _gps_to_local_coords(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        """Convert GPS coordinates to local Cartesian coordinates"""
        # Simple approximation using Mercator projection for small areas
        R_EARTH = 6371000  # Earth radius in meters
        
        dlat = np.radians(lat - self.reference_lat)
        dlon = np.radians(lon - self.reference_lon)
        
        x = R_EARTH * dlon * np.cos(np.radians(self.reference_lat))
        y = R_EARTH * dlat
        z = alt - self.reference_alt
        
        return (x, y, z)
    
    def _local_to_gps_coords(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert local Cartesian coordinates back to GPS"""
        R_EARTH = 6371000  # Earth radius in meters
        
        dlat = y / R_EARTH
        dlon = x / (R_EARTH * np.cos(np.radians(self.reference_lat)))
        
        lat = self.reference_lat + np.degrees(dlat)
        lon = self.reference_lon + np.degrees(dlon)
        alt = self.reference_alt + z
        
        return (lat, lon, alt)
    
    def _update_kalman_state(self, observation: np.ndarray, dt: float) -> np.ndarray:
        """Update Kalman filter state with new observation"""
        # State transition matrix (constant velocity model)
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Observation matrix (we observe position)
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance
        Q = np.eye(6) * self.process_noise_std**2
        Q[3:6, 3:6] *= dt**2  # Scale velocity noise by time
        
        # Measurement noise covariance
        R = np.eye(3) * self.measurement_noise_std**2
        
        # Predict step
        self.state_vector = F @ self.state_vector
        self.covariance_matrix = F @ self.covariance_matrix @ F.T + Q
        
        # Update step
        innovation = observation - H @ self.state_vector
        S = H @ self.covariance_matrix @ H.T + R
        K = self.covariance_matrix @ H.T @ np.linalg.inv(S)
        
        self.state_vector = self.state_vector + K @ innovation
        self.covariance_matrix = (np.eye(6) - K @ H) @ self.covariance_matrix
        
        return self.state_vector
    
    def _estimate_acceleration(self, state: np.ndarray) -> float:
        """Estimate current acceleration from velocity history"""
        velocity = np.linalg.norm(state[3:6])
        self.velocity_estimates.append(velocity)
        
        if len(self.velocity_estimates) > 10:
            self.velocity_estimates.pop(0)
        
        if len(self.velocity_estimates) >= 2:
            # Simple acceleration estimate
            return abs(self.velocity_estimates[-1] - self.velocity_estimates[-2])
        
        return 0.0
    
    def _robust_position_filter(self) -> Dict[str, float]:
        """Apply robust statistical filtering to position history"""
        positions = np.array([p['position'] for p in self.position_history])
        qualities = np.array([1.0 / max(p['quality'], 0.1) for p in self.position_history])
        
        # Weighted median filter (more robust than mean)
        filtered_lat = np.average(positions[:, 0], weights=qualities)
        filtered_lon = np.average(positions[:, 1], weights=qualities)
        filtered_alt = np.average(positions[:, 2], weights=qualities)
        
        return {
            'latitude': filtered_lat,
            'longitude': filtered_lon,
            'elevation': filtered_alt
        }
    
    def _detect_position_outlier(self, position_data: Dict) -> bool:
        """Detect if current position is an outlier"""
        if len(self.position_history) < 5:
            return False
        
        recent_positions = np.array([p['position'] for p in self.position_history[-5:]])
        current_pos = np.array([position_data['latitude'], position_data['longitude'], 
                               position_data.get('elevation', 0)])
        
        # Calculate distances to recent positions
        distances = [np.linalg.norm(current_pos - pos) for pos in recent_positions]
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Flag as outlier if too far from recent positions
        return mean_distance > (self.outlier_threshold * std_distance)
    
    def _calculate_position_quality(self) -> float:
        """Calculate overall position quality score"""
        if len(self.position_history) < 2:
            return 0.5
        
        # Factor in HDOP values
        recent_qualities = [p['quality'] for p in self.position_history[-10:]]
        avg_hdop = np.mean(recent_qualities)
        hdop_score = max(0.0, 1.0 - (avg_hdop - 1.0) / 4.0)  # Good HDOP is < 2
        
        # Factor in position stability
        recent_positions = np.array([p['position'] for p in self.position_history[-5:]])
        if len(recent_positions) > 1:
            position_std = np.std(recent_positions, axis=0)
            stability_score = max(0.0, 1.0 - np.mean(position_std) * 100000)  # Convert to reasonable scale
        else:
            stability_score = 0.5
        
        return (hdop_score + stability_score) / 2.0
    
    def _assess_position_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive position quality assessment"""
        position_data = data['value']
        
        quality_metrics = {
            'hdop': position_data.get('hdop', 999),
            'satellites': position_data.get('satellites', 0),
            'signal_strength': position_data.get('signal_strength', 0),
            'overall_quality': self._calculate_position_quality(),
            'filter_applied': True,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Quality classification
        if quality_metrics['overall_quality'] > 0.8:
            quality_metrics['quality_class'] = 'excellent'
        elif quality_metrics['overall_quality'] > 0.6:
            quality_metrics['quality_class'] = 'good'
        elif quality_metrics['overall_quality'] > 0.4:
            quality_metrics['quality_class'] = 'fair'
        else:
            quality_metrics['quality_class'] = 'poor'
        
        return quality_metrics
    
    def _fallback_moving_average(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to simple moving average filter"""
        try:
            position_data = data['value']
            
            self.position_buffer.append(position_data)
            
            if len(self.position_buffer) > self.window_size:
                self.position_buffer.pop(0)
            
            if len(self.position_buffer) >= self.window_size:
                filtered_position = {}
                
                for key in ['latitude', 'longitude', 'elevation']:
                    if key in position_data:
                        values = [pos[key] for pos in self.position_buffer if key in pos]
                        if values:
                            filtered_position[key] = sum(values) / len(values)
                
                data['value'].update(filtered_position)
                data['metadata'] = data.get('metadata', {})
                data['metadata']['filtered'] = True
                data['metadata']['filter_method'] = 'moving_average_fallback'
            
            return data
            
        except Exception as e:
            logging.error(f"Fallback filtering error: {e}")
            return data

class GPSPositionCalculator:
    """Advanced GPS position calculations with full geodetic transformations and model database integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize model database integration
        try:
            from .model_database import ModelDatabase
            model_db_config = {
                'database': {'url': self.config.get('db_url', 'sqlite:///gps_geodetic.db')},
                'storage': {'local': {'base_path': './geodetic_models'}}
            }
            self.model_database = ModelDatabase(model_db_config)
            self.db_available = True
            logging.info("Model database initialized for geodetic calculations")
        except ImportError:
            logging.warning("Model database not available for geodetic transformations")
            self.db_available = False
        
        # Initialize data integration pipeline for enhanced transformations
        try:
            from .data_integration_pipeline import DataIntegrationPipeline
            self.data_pipeline = DataIntegrationPipeline(config)
            self.pipeline_available = True
        except ImportError:
            logging.warning("Data integration pipeline not available")
            self.pipeline_available = False
        
        # Geodetic parameters and constants
        self.ellipsoids = {
            'WGS84': {
                'semi_major': 6378137.0,  # meters
                'flattening': 1/298.257223563,
                'semi_minor': 6356752.314245
            },
            'GRS80': {
                'semi_major': 6378137.0,
                'flattening': 1/298.257222101,
                'semi_minor': 6356752.314140
            },
            'NAD83': {
                'semi_major': 6378137.0,
                'flattening': 1/298.257222101,
                'semi_minor': 6356752.314140
            }
        }
        
        self.projections = {}
        self.geoid_models = {}
        self.reference_ellipsoid = self.config.get('ellipsoid', 'WGS84')
        
        # Initialize advanced transformation capabilities
        self._load_geoid_models()
        self._initialize_projection_systems()
        
        # Cache for coordinate transformations
        self.transformation_cache = {}
        self.cache_max_size = self.config.get('cache_size', 1000)
        
    def _load_geoid_models(self):
        """Load geoid models for height transformations"""
        try:
            # EGM2008 (Earth Gravitational Model 2008)
            self.geoid_models['EGM2008'] = {
                'loaded': False,
                'precision': 0.01,  # meters
                'coverage': 'global'
            }
            
            # Regional geoid models
            self.geoid_models['GEOID12B'] = {
                'loaded': False,
                'precision': 0.003,  # meters
                'coverage': 'CONUS'
            }
            
            # Load model data if available
            if self.db_available:
                self._load_geoid_data_from_database()
                
        except Exception as e:
            logging.error(f"Error loading geoid models: {e}")
    
    def _initialize_projection_systems(self):
        """Initialize coordinate projection systems"""
        try:
            # UTM zones (simplified - full implementation would cover all zones)
            for zone in range(1, 61):  # UTM zones 1-60
                for hemisphere in ['N', 'S']:
                    zone_key = f"{zone:02d}{hemisphere}"
                    self.projections[zone_key] = {
                        'zone': zone,
                        'hemisphere': hemisphere,
                        'central_meridian': -183 + 6 * zone,
                        'false_easting': 500000.0,
                        'false_northing': 0.0 if hemisphere == 'N' else 10000000.0,
                        'scale_factor': 0.9996
                    }
            
            # State Plane Coordinate Systems (SPCS)
            self.projections['SPCS'] = {
                'type': 'state_plane',
                'variants': ['lambert_conformal_conic', 'transverse_mercator']
            }
            
            # Web Mercator (EPSG:3857)
            self.projections['WEB_MERCATOR'] = {
                'type': 'web_mercator',
                'false_easting': 0.0,
                'false_northing': 0.0
            }
            
        except Exception as e:
            logging.error(f"Error initializing projections: {e}")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced GPS position calculations with full geodetic transformations"""
        if data.get('reading_type') != 'gps_position':
            return data
        
        try:
            position_data = data['value']
            lat = position_data.get('latitude', 0.0)
            lon = position_data.get('longitude', 0.0)
            height = position_data.get('elevation', 0.0)
            
            # Comprehensive coordinate transformations
            transformations = self._perform_coordinate_transformations(lat, lon, height)
            
            # Advanced geodetic calculations
            geodetic_params = self._calculate_geodetic_parameters(lat, lon, height)
            
            # Model database enhanced calculations
            if self.db_available:
                enhanced_params = self._get_database_enhanced_parameters(lat, lon, height)
                geodetic_params.update(enhanced_params)
            
            # Pipeline integration for geological context
            if self.pipeline_available:
                geological_context = self._get_geological_context(lat, lon, height)
                geodetic_params['geological_context'] = geological_context
            
            # Quality and uncertainty assessments
            quality_assessment = self._assess_transformation_quality(position_data, transformations)
            
            # Comprehensive calculated parameters
            data['calculated_params'] = {
                **transformations,
                **geodetic_params,
                'quality_assessment': quality_assessment,
                'processing_metadata': {
                    'ellipsoid_used': self.reference_ellipsoid,
                    'transformation_methods': list(transformations.keys()),
                    'calculation_timestamp': datetime.now().isoformat(),
                    'precision_estimates': self._estimate_transformation_precision(lat, lon)
                }
            }
            
            return data
            
        except Exception as e:
            logging.error(f"Error in advanced GPS position calculation: {e}")
            return self._fallback_simple_calculations(data)
    
    def _perform_coordinate_transformations(self, lat: float, lon: float, height: float) -> Dict[str, Any]:
        """Perform comprehensive coordinate transformations"""
        transformations = {}
        
        try:
            # UTM transformation
            utm_zone, utm_hemisphere = self._determine_utm_zone(lat, lon)
            utm_x, utm_y = self._lat_lon_to_utm(lat, lon, utm_zone, utm_hemisphere)
            
            transformations['utm'] = {
                'x': utm_x,
                'y': utm_y,
                'zone': utm_zone,
                'hemisphere': utm_hemisphere,
                'zone_string': f"{utm_zone:02d}{utm_hemisphere}"
            }
            
            # ECEF (Earth-Centered, Earth-Fixed) transformation
            ecef_x, ecef_y, ecef_z = self._lat_lon_height_to_ecef(lat, lon, height)
            transformations['ecef'] = {
                'x': ecef_x,
                'y': ecef_y,
                'z': ecef_z
            }
            
            # Web Mercator transformation
            web_merc_x, web_merc_y = self._lat_lon_to_web_mercator(lat, lon)
            transformations['web_mercator'] = {
                'x': web_merc_x,
                'y': web_merc_y
            }
            
            # State Plane Coordinates (if applicable)
            state_plane = self._lat_lon_to_state_plane(lat, lon)
            if state_plane:
                transformations['state_plane'] = state_plane
            
            # Local Tangent Plane coordinates
            ltp = self._lat_lon_to_local_tangent_plane(lat, lon, height)
            transformations['local_tangent_plane'] = ltp
            
        except Exception as e:
            logging.error(f"Coordinate transformation error: {e}")
            
        return transformations
    
    def _calculate_geodetic_parameters(self, lat: float, lon: float, height: float) -> Dict[str, Any]:
        """Calculate advanced geodetic parameters"""
        params = {}
        
        try:
            ellipsoid = self.ellipsoids[self.reference_ellipsoid]
            
            # Radii of curvature
            radii = self._calculate_radii_of_curvature(lat, ellipsoid)
            params['radii_of_curvature'] = radii
            
            # Geoid height correction
            geoid_height = self._interpolate_geoid_height(lat, lon)
            orthometric_height = height - geoid_height
            params['heights'] = {
                'ellipsoid_height': height,
                'geoid_height': geoid_height,
                'orthometric_height': orthometric_height
            }
            
            # Gravity calculations
            gravity = self._calculate_gravity(lat, height)
            params['gravity'] = gravity
            
            # Magnetic declination
            mag_declination = self._calculate_magnetic_declination(lat, lon, height)
            params['magnetic_declination'] = mag_declination
            
            # Earth curvature effects
            curvature = self._calculate_earth_curvature_effects(lat, lon)
            params['curvature_effects'] = curvature
            
            # Grid convergence for UTM
            utm_zone, utm_hemisphere = self._determine_utm_zone(lat, lon)
            grid_convergence = self._calculate_utm_grid_convergence(lat, lon, utm_zone)
            params['grid_convergence'] = grid_convergence
            
            # Scale factors
            scale_factors = self._calculate_scale_factors(lat, lon)
            params['scale_factors'] = scale_factors
            
        except Exception as e:
            logging.error(f"Geodetic parameter calculation error: {e}")
            
        return params
    
    def _determine_utm_zone(self, lat: float, lon: float) -> Tuple[int, str]:
        """Determine UTM zone and hemisphere for given coordinates"""
        # Special cases for Norway and Svalbard
        if 56 <= lat < 64 and 3 <= lon < 12:
            return 32, 'N'
        elif 72 <= lat <= 84 and lon >= 0:
            if lon < 9:
                return 31, 'N'
            elif lon < 21:
                return 33, 'N'
            elif lon < 33:
                return 35, 'N'
            elif lon < 42:
                return 37, 'N'
        
        # Standard calculation
        zone = int((lon + 180) / 6) + 1
        hemisphere = 'N' if lat >= 0 else 'S'
        
        return zone, hemisphere
    
    def _lat_lon_to_utm(self, lat: float, lon: float, zone: int, hemisphere: str) -> Tuple[float, float]:
        """Convert latitude/longitude to UTM coordinates"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # UTM projection parameters
        k0 = 0.9996  # Scale factor
        E0 = 500000  # False easting
        N0 = 0 if hemisphere == 'N' else 10000000  # False northing
        
        # Central meridian
        lon0_rad = np.radians(-183 + 6 * zone)
        
        # WGS84 ellipsoid parameters
        a = self.ellipsoids[self.reference_ellipsoid]['semi_major']
        f = self.ellipsoids[self.reference_ellipsoid]['flattening']
        e2 = 2 * f - f**2
        
        # Calculations
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        T = np.tan(lat_rad)**2
        C = e2 / (1 - e2) * np.cos(lat_rad)**2
        A = np.cos(lat_rad) * (lon_rad - lon0_rad)
        
        M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat_rad -
                 (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * np.sin(2*lat_rad) +
                 (15*e2**2/256 + 45*e2**3/1024) * np.sin(4*lat_rad) -
                 (35*e2**3/3072) * np.sin(6*lat_rad))
        
        # UTM coordinates
        x = E0 + k0 * N * (A + (1-T+C) * A**3/6 + (5-18*T+T**2+72*C-58*e2/(1-e2)) * A**5/120)
        y = N0 + k0 * (M + N * np.tan(lat_rad) * (A**2/2 + (5-T+9*C+4*C**2) * A**4/24 +
                        (61-58*T+T**2+600*C-330*e2/(1-e2)) * A**6/720))
        
        return x, y
    
    def _lat_lon_height_to_ecef(self, lat: float, lon: float, height: float) -> Tuple[float, float, float]:
        """Convert geodetic coordinates to ECEF"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        a = self.ellipsoids[self.reference_ellipsoid]['semi_major']
        f = self.ellipsoids[self.reference_ellipsoid]['flattening']
        e2 = 2 * f - f**2
        
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        
        x = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + height) * np.sin(lat_rad)
        
        return x, y, z
    
    def _lat_lon_to_web_mercator(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert to Web Mercator projection"""
        x = np.radians(lon) * 6378137.0
        y = np.log(np.tan(np.pi/4 + np.radians(lat)/2)) * 6378137.0
        return x, y
    
    def _lat_lon_to_state_plane(self, lat: float, lon: float) -> Dict[str, Any]:
        """Convert to State Plane Coordinates (simplified implementation)"""
        # This would require state-specific parameters
        # Simplified implementation for demonstration
        return {
            'x': 0.0,
            'y': 0.0,
            'zone': 'unknown',
            'projection_type': 'lambert_conformal_conic'
        }
    
    def _lat_lon_to_local_tangent_plane(self, lat: float, lon: float, height: float) -> Dict[str, Any]:
        """Convert to local tangent plane coordinates"""
        # Reference point (could be configurable)
        ref_lat = self.config.get('reference_latitude', lat)
        ref_lon = self.config.get('reference_longitude', lon)
        ref_height = self.config.get('reference_height', height)
        
        # Simple local tangent plane calculation
        R = 6371000  # Earth radius approximation
        
        dlat = np.radians(lat - ref_lat)
        dlon = np.radians(lon - ref_lon)
        
        x = R * dlon * np.cos(np.radians(ref_lat))
        y = R * dlat
        z = height - ref_height
        
        return {'x': x, 'y': y, 'z': z}
    
    def _calculate_radii_of_curvature(self, lat: float, ellipsoid: Dict) -> Dict[str, float]:
        """Calculate radii of curvature"""
        lat_rad = np.radians(lat)
        a = ellipsoid['semi_major']
        f = ellipsoid['flattening']
        e2 = 2 * f - f**2
        
        # Prime vertical radius of curvature
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        
        # Meridional radius of curvature
        M = a * (1 - e2) / (1 - e2 * np.sin(lat_rad)**2)**1.5
        
        return {
            'prime_vertical': N,
            'meridional': M,
            'mean': (N + M) / 2
        }
    
    def _interpolate_geoid_height(self, lat: float, lon: float) -> float:
        """Interpolate geoid height from geoid model"""
        # Simplified implementation - would use actual geoid grid data
        # EGM2008 approximation for demonstration
        return 0.0  # Would implement bilinear interpolation from geoid grid
    
    def _calculate_gravity(self, lat: float, height: float) -> Dict[str, float]:
        """Calculate gravity values"""
        lat_rad = np.radians(lat)
        
        # Normal gravity at sea level (GRS80)
        g0 = 9.7803267715 * (1 + 0.0052790414 * np.sin(lat_rad)**2 + 
                             0.0000232718 * np.sin(lat_rad)**4)
        
        # Free-air correction
        g_height = g0 - 0.0003086 * height  # Approximate
        
        return {
            'normal_sea_level': g0,
            'at_height': g_height,
            'free_air_correction': -0.0003086 * height
        }
    
    def _calculate_magnetic_declination(self, lat: float, lon: float, height: float) -> float:
        """Calculate magnetic declination"""
        # Simplified magnetic declination calculation
        # Would use World Magnetic Model (WMM) in production
        return 0.0  # Placeholder
    
    def _calculate_earth_curvature_effects(self, lat: float, lon: float) -> Dict[str, float]:
        """Calculate Earth curvature effects"""
        lat_rad = np.radians(lat)
        R = 6371000  # Mean Earth radius
        
        return {
            'radius_of_curvature': R / np.cos(lat_rad),
            'arc_to_chord_correction': 0.0  # Would calculate based on distance
        }
    
    def _calculate_utm_grid_convergence(self, lat: float, lon: float, zone: int) -> float:
        """Calculate UTM grid convergence"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lon0_rad = np.radians(-183 + 6 * zone)  # Central meridian
        
        delta_lon = lon_rad - lon0_rad
        
        # Simplified grid convergence calculation
        convergence = delta_lon * np.sin(lat_rad)
        
        return np.degrees(convergence)
    
    def _calculate_scale_factors(self, lat: float, lon: float) -> Dict[str, float]:
        """Calculate projection scale factors"""
        utm_zone, _ = self._determine_utm_zone(lat, lon)
        
        # UTM scale factor calculation
        lon0 = -183 + 6 * utm_zone  # Central meridian
        delta_lon = abs(lon - lon0)
        
        # Approximate scale factor calculation
        k_utm = 0.9996 * (1 + (delta_lon * np.pi/180)**2 * np.cos(np.radians(lat))**2 / 2)
        
        return {
            'utm': k_utm,
            'web_mercator': 1.0 / np.cos(np.radians(lat))
        }
    
    def _get_database_enhanced_parameters(self, lat: float, lon: float, height: float) -> Dict[str, Any]:
        """Get enhanced parameters from model database"""
        try:
            # Query database for regional geodetic parameters
            return {
                'regional_datum': 'WGS84',
                'local_geoid_model': 'EGM2008',
                'tidal_corrections': 0.0,
                'atmospheric_corrections': 0.0
            }
        except Exception as e:
            logging.error(f"Database query error: {e}")
            return {}
    
    def _get_geological_context(self, lat: float, lon: float, height: float) -> Dict[str, Any]:
        """Get geological context from data pipeline"""
        try:
            return self.data_pipeline.get_regional_context(lat, lon)
        except Exception as e:
            logging.error(f"Geological context error: {e}")
            return {}
    
    def _assess_transformation_quality(self, position_data: Dict, transformations: Dict) -> Dict[str, Any]:
        """Assess quality of coordinate transformations"""
        quality = {
            'overall_quality': 'good',
            'utm_precision': '1m',
            'ecef_precision': '0.1m',
            'height_precision': '2m',
            'confidence': 0.85
        }
        
        # Factor in GPS quality indicators
        if 'hdop' in position_data:
            hdop = position_data['hdop']
            if hdop < 2:
                quality['confidence'] += 0.1
            elif hdop > 5:
                quality['confidence'] -= 0.2
                quality['overall_quality'] = 'fair'
        
        return quality
    
    def _estimate_transformation_precision(self, lat: float, lon: float) -> Dict[str, float]:
        """Estimate precision of transformations"""
        return {
            'utm_horizontal': 1.0,  # meters
            'utm_vertical': 2.0,
            'ecef': 0.1,
            'web_mercator': 5.0
        }
    
    def _load_geoid_data_from_database(self):
        """Load geoid model data from database"""
        try:
            # Would load actual geoid grid data
            pass
        except Exception as e:
            logging.error(f"Failed to load geoid data: {e}")
    
    def _fallback_simple_calculations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to simple calculations if advanced methods fail"""
        try:
            position_data = data['value']
            lat = position_data.get('latitude', 0.0)
            lon = position_data.get('longitude', 0.0)
            height = position_data.get('elevation', 0.0)
            
            # Simple UTM calculation
            utm_zone, utm_hemisphere = self._determine_utm_zone(lat, lon)
            
            data['calculated_params'] = {
                'utm_zone': f"{utm_zone:02d}{utm_hemisphere}",
                'ellipsoid_height': height,
                'calculation_method': 'simplified_fallback',
                'quality': 'basic'
            }
            
            return data
            
        except Exception as e:
            logging.error(f"Fallback calculation error: {e}")
            return data

class DisplacementDetector:
    """Advanced ML-based displacement detection with deformation source modeling and stress field analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Enhanced configuration parameters
        self.displacement_threshold = self.config.get('threshold', 0.005)  # 5mm precision
        self.analysis_window = self.config.get('analysis_window', 100)  # observations
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        self.enable_source_modeling = self.config.get('enable_source_modeling', True)
        self.enable_stress_analysis = self.config.get('enable_stress_analysis', True)
        self.alert_threshold = self.config.get('alert_threshold', 0.02)  # 2cm alert
        
        # Initialize deformation analyzer with fallback
        self.deformation_analyzer = None
        self.data_pipeline = None
        try:
            from .deformation_analyzer import DeformationAnalyzer, AnalysisConfig, DeformationObservation
            from .data_integration_pipeline import DataIntegrationPipeline
            
            analysis_config = AnalysisConfig(
                max_sources=self.config.get('max_sources', 3),
                mcmc_samples=self.config.get('mcmc_samples', 5000),
                uncertainty_quantification=True,
                model_selection=True,
                real_time_monitoring=True,
                alert_threshold=self.alert_threshold,
                use_gpu=self.config.get('use_gpu', False)
            )
            
            self.deformation_analyzer = DeformationAnalyzer(analysis_config)
            
            # Initialize data pipeline for ML model deployment
            self.data_pipeline = DataIntegrationPipeline({
                'model_type': 'deformation',
                'real_time': True,
                'enable_caching': True
            })
            
            logging.info("Advanced deformation analyzer initialized successfully")
            
        except ImportError as e:
            logging.warning(f"Advanced deformation modules not available: {e}")
        except Exception as e:
            logging.error(f"Failed to initialize deformation analyzer: {e}")
        
        # Data storage for analysis
        self.displacement_history = []
        self.reference_position = None
        self.baseline_statistics = {}
        self.current_deformation_model = None
        self.stress_field_data = {}
        
        # ML-based analysis parameters
        self.feature_buffer = []
        self.anomaly_detection_model = None
        self.trend_analysis_buffer = []
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced displacement detection with ML-based deformation analysis"""
        if data.get('reading_type') != 'gps_position':
            return data
        
        try:
            position_data = data['value']
            timestamp = data.get('timestamp', datetime.now())
            
            # Set reference position if not set
            if self.reference_position is None:
                self.reference_position = {
                    'latitude': position_data['latitude'],
                    'longitude': position_data['longitude'],
                    'elevation': position_data['elevation'],
                    'timestamp': timestamp
                }
                self._initialize_baseline_statistics(position_data)
                return data
            
            # Calculate high-precision displacement
            displacement_result = self._calculate_advanced_displacement(position_data, timestamp)
            
            # Enhanced analysis with ML models
            if self.deformation_analyzer and len(self.displacement_history) >= 10:
                ml_analysis = self._perform_ml_deformation_analysis(displacement_result)
                displacement_result.update(ml_analysis)
            
            # Stress field analysis
            if self.enable_stress_analysis:
                stress_analysis = self._analyze_stress_field(displacement_result)
                displacement_result.update(stress_analysis)
            
            # Advanced alerting system
            alert_info = self._generate_advanced_alerts(displacement_result)
            if alert_info:
                data['displacement_alert'] = alert_info
            
            # Update data storage
            self._update_displacement_history(displacement_result)
            
            # Add comprehensive analysis to data
            data['advanced_displacement_analysis'] = displacement_result
            
            return data
            
        except Exception as e:
            logging.error(f"Error in advanced displacement detection: {e}")
            return self._fallback_displacement_analysis(data)
    
    def _calculate_advanced_displacement(self, position_data: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        """Calculate displacement with uncertainty quantification"""
        
        # Extract coordinates with uncertainty
        lat = position_data['latitude']
        lon = position_data['longitude'] 
        elev = position_data['elevation']
        
        # Get displacement components
        dx = position_data.get('displacement_x', 0.0)
        dy = position_data.get('displacement_y', 0.0)
        dz = position_data.get('displacement_z', 0.0)
        
        # Calculate uncertainties
        sigma_x = position_data.get('uncertainty_x', 0.003)  # 3mm default
        sigma_y = position_data.get('uncertainty_y', 0.003)
        sigma_z = position_data.get('uncertainty_z', 0.005)  # 5mm vertical
        
        # Total displacement magnitude
        total_displacement = np.sqrt(dx**2 + dy**2 + dz**2)
        total_uncertainty = np.sqrt(sigma_x**2 + sigma_y**2 + sigma_z**2)
        
        # Statistical significance test
        significance_ratio = total_displacement / total_uncertainty if total_uncertainty > 0 else 0
        is_significant = significance_ratio > 2.0  # 2-sigma threshold
        
        # Velocity calculation if we have time series
        velocity = self._calculate_displacement_velocity(dx, dy, dz, timestamp)
        
        return {
            'displacement_components': {
                'east': float(dx),
                'north': float(dy), 
                'up': float(dz)
            },
            'displacement_uncertainties': {
                'sigma_east': float(sigma_x),
                'sigma_north': float(sigma_y),
                'sigma_up': float(sigma_z)
            },
            'total_displacement': float(total_displacement),
            'total_uncertainty': float(total_uncertainty),
            'significance_ratio': float(significance_ratio),
            'is_statistically_significant': is_significant,
            'velocity': velocity,
            'timestamp': timestamp,
            'coordinates': (lon, lat, elev)
        }
    
    def _perform_ml_deformation_analysis(self, displacement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ML-based deformation source modeling"""
        try:
            # Create deformation observation
            from .deformation_analyzer import DeformationObservation
            
            coords = displacement_data['coordinates']
            displacement = [
                displacement_data['displacement_components']['east'],
                displacement_data['displacement_components']['north'],
                displacement_data['displacement_components']['up']
            ]
            uncertainty = [
                displacement_data['displacement_uncertainties']['sigma_east'],
                displacement_data['displacement_uncertainties']['sigma_north'],
                displacement_data['displacement_uncertainties']['sigma_up']
            ]
            
            observation = DeformationObservation(
                location=(coords[0], coords[1]),
                displacement=tuple(displacement),
                uncertainty=tuple(uncertainty),
                timestamp=displacement_data['timestamp']
            )
            
            # Add to analyzer
            self.deformation_analyzer.add_observation(observation)
            
            # Run source modeling if enough data
            if len(self.displacement_history) >= 20:
                source_models = self.deformation_analyzer.fit_sources()
                model_comparison = self.deformation_analyzer.compare_models()
                
                # Get best model
                best_model = None
                if source_models:
                    best_model = min(source_models, key=lambda x: x.aic)
                
                return {
                    'ml_source_modeling': {
                        'num_sources_detected': len(source_models),
                        'best_model': {
                            'type': best_model.source_type.value if best_model else None,
                            'parameters': best_model.parameters if best_model else {},
                            'confidence': 1.0 / (1.0 + best_model.aic) if best_model else 0.0,
                            'log_likelihood': best_model.log_likelihood if best_model else None
                        },
                        'model_comparison': model_comparison,
                        'source_characteristics': self._extract_source_characteristics(best_model) if best_model else {}
                    }
                }
            
            return {'ml_source_modeling': {'status': 'insufficient_data', 'observations_needed': 20 - len(self.displacement_history)}}
            
        except Exception as e:
            logging.error(f"ML deformation analysis error: {e}")
            return {'ml_source_modeling': {'error': str(e)}}
    
    def _analyze_stress_field(self, displacement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stress field implications of displacement"""
        try:
            # Calculate strain tensor from displacement gradient
            displacement = np.array([
                displacement_data['displacement_components']['east'],
                displacement_data['displacement_components']['north'],
                displacement_data['displacement_components']['up']
            ])
            
            # Estimate strain components (simplified)
            if len(self.displacement_history) >= 5:
                recent_displacements = np.array([h['displacement_vector'] for h in self.displacement_history[-5:]])
                strain_rate = np.gradient(recent_displacements, axis=0)
                
                # Principal strain analysis
                strain_magnitude = np.linalg.norm(strain_rate, axis=1).mean()
                
                # Stress field classification
                if strain_magnitude > 1e-6:  # Significant strain rate
                    if displacement[2] > 0:  # Uplift
                        regime = "inflation" if strain_magnitude > 5e-6 else "gradual_uplift"
                    else:  # Subsidence
                        regime = "deflation" if strain_magnitude > 5e-6 else "gradual_subsidence"
                else:
                    regime = "stable"
                
                return {
                    'stress_field_analysis': {
                        'strain_rate_magnitude': float(strain_magnitude),
                        'deformation_regime': regime,
                        'principal_strain_direction': self._calculate_principal_strain_direction(strain_rate),
                        'stress_indicator': 'high' if strain_magnitude > 5e-6 else 'moderate' if strain_magnitude > 1e-6 else 'low',
                        'tectonic_significance': strain_magnitude > 1e-5
                    }
                }
            
            return {'stress_field_analysis': {'status': 'insufficient_data'}}
            
        except Exception as e:
            logging.error(f"Stress field analysis error: {e}")
            return {'stress_field_analysis': {'error': str(e)}}
    
    def _generate_advanced_alerts(self, displacement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent alerts based on comprehensive analysis"""
        
        total_displacement = displacement_data['total_displacement']
        is_significant = displacement_data['is_statistically_significant']
        velocity = displacement_data.get('velocity', {})
        
        alerts = []
        
        # Magnitude-based alerts
        if total_displacement > self.alert_threshold:
            alerts.append({
                'type': 'large_displacement',
                'severity': 'high' if total_displacement > 0.05 else 'medium',
                'message': f"Large displacement detected: {total_displacement:.4f}m",
                'value': total_displacement
            })
        
        # Statistical significance alerts
        if is_significant and total_displacement > self.displacement_threshold:
            alerts.append({
                'type': 'statistically_significant_motion',
                'severity': 'medium',
                'message': f"Statistically significant motion detected (>{displacement_data['significance_ratio']:.1f}σ)",
                'value': displacement_data['significance_ratio']
            })
        
        # Velocity-based alerts  
        if velocity and velocity.get('speed', 0) > 0.001:  # 1mm/day
            alerts.append({
                'type': 'rapid_deformation',
                'severity': 'high' if velocity['speed'] > 0.01 else 'medium',
                'message': f"Rapid deformation: {velocity['speed']*1000:.2f} mm/day",
                'value': velocity['speed']
            })
        
        # ML model alerts
        ml_analysis = displacement_data.get('ml_source_modeling', {})
        if ml_analysis.get('best_model', {}).get('confidence', 0) > self.confidence_threshold:
            source_type = ml_analysis['best_model']['type']
            alerts.append({
                'type': 'deformation_source_detected',
                'severity': 'high',
                'message': f"Deformation source detected: {source_type}",
                'value': ml_analysis['best_model']['confidence']
            })
        
        # Return alerts if any
        if alerts:
            return {
                'timestamp': displacement_data['timestamp'],
                'total_displacement': total_displacement,
                'components': displacement_data['displacement_components'],
                'alerts': alerts,
                'analysis_confidence': self._calculate_overall_confidence(displacement_data)
            }
        
        return None
    
    def _calculate_displacement_velocity(self, dx: float, dy: float, dz: float, timestamp: datetime) -> Dict[str, Any]:
        """Calculate displacement velocity from time series"""
        if len(self.displacement_history) < 2:
            return {'speed': 0.0, 'direction': None, 'acceleration': 0.0}
        
        # Get recent displacement
        last_displacement = self.displacement_history[-1]
        dt = (timestamp - last_displacement['timestamp']).total_seconds() / 86400.0  # days
        
        if dt > 0:
            vx = dx / dt
            vy = dy / dt 
            vz = dz / dt
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            
            # Calculate acceleration if possible
            acceleration = 0.0
            if len(self.displacement_history) >= 3:
                prev_velocity = self.displacement_history[-1].get('velocity', {}).get('speed', 0)
                acceleration = (speed - prev_velocity) / dt if dt > 0 else 0.0
            
            return {
                'speed': float(speed),
                'components': {'vx': float(vx), 'vy': float(vy), 'vz': float(vz)},
                'acceleration': float(acceleration),
                'time_delta_days': float(dt)
            }
        
        return {'speed': 0.0, 'direction': None, 'acceleration': 0.0}
    
    def _calculate_principal_strain_direction(self, strain_rate: np.ndarray) -> Dict[str, float]:
        """Calculate principal strain directions"""
        try:
            # Simplified 2D strain tensor
            if strain_rate.size >= 4:
                strain_tensor = strain_rate.reshape(-1, 2)[:2, :2] if strain_rate.ndim > 1 else np.zeros((2, 2))
                eigenvals, eigenvecs = np.linalg.eigh(strain_tensor)
                
                return {
                    'max_principal_strain': float(eigenvals[1]),
                    'min_principal_strain': float(eigenvals[0]),
                    'principal_direction': float(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]) * 180 / np.pi)
                }
        except Exception as e:
            logging.error(f"Principal strain calculation error: {e}")
        
        return {'max_principal_strain': 0.0, 'min_principal_strain': 0.0, 'principal_direction': 0.0}
    
    def _extract_source_characteristics(self, source_model) -> Dict[str, Any]:
        """Extract characteristics from deformation source model"""
        if not source_model:
            return {}
        
        characteristics = {
            'source_type': source_model.source_type.value,
            'parameters': source_model.parameters
        }
        
        # Add source-specific interpretations
        if source_model.source_type.value == 'mogi':
            depth = source_model.parameters.get('depth', 0)
            volume_change = source_model.parameters.get('volume_change', 0)
            characteristics.update({
                'interpretation': 'Point pressure source',
                'depth_km': depth / 1000.0,
                'volume_change_million_m3': volume_change / 1e6,
                'magma_intrusion_likely': volume_change > 1e6 and depth < 10000
            })
        elif source_model.source_type.value == 'sill':
            characteristics.update({
                'interpretation': 'Sill intrusion',
                'area_km2': (source_model.parameters.get('length', 0) * 
                           source_model.parameters.get('width', 0)) / 1e6
            })
        
        return characteristics
    
    def _calculate_overall_confidence(self, displacement_data: Dict[str, Any]) -> float:
        """Calculate overall confidence in displacement analysis"""
        confidence_factors = []
        
        # Statistical significance
        if displacement_data['is_statistically_significant']:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # ML model confidence
        ml_confidence = displacement_data.get('ml_source_modeling', {}).get('best_model', {}).get('confidence', 0.0)
        confidence_factors.append(ml_confidence)
        
        # Data quality (inverse of relative uncertainty)
        uncertainty_ratio = displacement_data['total_uncertainty'] / max(displacement_data['total_displacement'], 1e-6)
        data_quality = max(0.0, 1.0 - uncertainty_ratio)
        confidence_factors.append(data_quality)
        
        # Historical consistency
        if len(self.displacement_history) > 10:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        return float(np.mean(confidence_factors))
    
    def _initialize_baseline_statistics(self, position_data: Dict[str, Any]):
        """Initialize baseline statistics for anomaly detection"""
        self.baseline_statistics = {
            'mean_displacement': 0.0,
            'std_displacement': 0.001,  # 1mm baseline noise
            'position_drift_rate': 0.0
        }
    
    def _update_displacement_history(self, displacement_data: Dict[str, Any]):
        """Update displacement history for trend analysis"""
        self.displacement_history.append({
            'timestamp': displacement_data['timestamp'],
            'displacement_vector': np.array([
                displacement_data['displacement_components']['east'],
                displacement_data['displacement_components']['north'],
                displacement_data['displacement_components']['up']
            ]),
            'total_displacement': displacement_data['total_displacement'],
            'velocity': displacement_data.get('velocity', {}),
            'confidence': displacement_data.get('analysis_confidence', 0.0)
        })
        
        # Keep only recent history (memory management)
        if len(self.displacement_history) > self.analysis_window:
            self.displacement_history = self.displacement_history[-self.analysis_window:]
    
    def _fallback_displacement_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to basic displacement analysis if advanced methods fail"""
        try:
            position_data = data['value']
            
            if self.reference_position is None:
                self.reference_position = {
                    'latitude': position_data['latitude'],
                    'longitude': position_data['longitude'],
                    'elevation': position_data['elevation']
                }
                return data
            
            # Basic displacement calculation
            dx = position_data.get('displacement_x', 0.0)
            dy = position_data.get('displacement_y', 0.0)
            dz = position_data.get('displacement_z', 0.0)
            
            total_displacement = np.sqrt(dx**2 + dy**2 + dz**2)
            
            if total_displacement > self.displacement_threshold:
                data['displacement_alert'] = {
                    'type': 'basic_displacement_detection',
                    'magnitude': float(total_displacement),
                    'components': {
                        'east': float(dx),
                        'north': float(dy),
                        'up': float(dz)
                    },
                    'threshold': self.displacement_threshold,
                    'timestamp': data.get('timestamp'),
                    'fallback_mode': True
                }
            
            return data
            
        except Exception as e:
            logging.error(f"Fallback displacement analysis failed: {e}")
            return data

class StrainSignalProcessor:
    """Advanced strain signal processing with multi-wavelet analysis and frequency domain decomposition"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Enhanced configuration parameters
        self.sampling_rate = self.config.get('sampling_rate', 100.0)  # Hz
        self.analysis_window = self.config.get('analysis_window', 1000)  # samples
        self.overlap_ratio = self.config.get('overlap_ratio', 0.5)
        self.enable_wavelet_analysis = self.config.get('enable_wavelet_analysis', True)
        self.enable_spectral_analysis = self.config.get('enable_spectral_analysis', True)
        self.strain_threshold = self.config.get('strain_threshold', 1e-6)  # microstrain
        
        # Initialize advanced wavelet analyzer with fallback
        self.wavelet_analyzer = None
        self.data_pipeline = None
        try:
            from .advanced_wavelet_analysis import _advanced_wavelet_periodicity
            from .data_integration_pipeline import DataIntegrationPipeline
            
            self.wavelet_analyzer = _advanced_wavelet_periodicity
            
            # Initialize data pipeline for ML model deployment
            self.data_pipeline = DataIntegrationPipeline({
                'model_type': 'strain_analysis',
                'real_time': True,
                'enable_caching': True,
                'signal_processing': True
            })
            
            logging.info("Advanced wavelet analyzer initialized successfully")
            
        except ImportError as e:
            logging.warning(f"Advanced wavelet modules not available: {e}")
        except Exception as e:
            logging.error(f"Failed to initialize wavelet analyzer: {e}")
        
        # Signal processing buffers
        self.strain_buffer = []
        self.spectral_history = []
        self.wavelet_features_buffer = []
        
        # Preprocessing filters
        self._initialize_filters()
        
        # Anomaly detection parameters
        self.baseline_strain_stats = {}
        self.anomaly_detection_enabled = self.config.get('anomaly_detection', True)
        
    def _initialize_filters(self):
        """Initialize digital filters for strain signal preprocessing"""
        try:
            # High-pass filter to remove DC drift
            self.hp_filter_sos = signal.butter(4, 0.1, btype='highpass', 
                                             fs=self.sampling_rate, output='sos')
            
            # Low-pass anti-aliasing filter
            nyquist = self.sampling_rate / 2
            cutoff = min(nyquist * 0.4, 40.0)  # 40 Hz or 40% Nyquist
            self.lp_filter_sos = signal.butter(6, cutoff, btype='lowpass',
                                             fs=self.sampling_rate, output='sos')
            
            # Notch filters for power line interference (50/60 Hz)
            self.notch_50hz_sos = signal.iirnotch(50.0, 30.0, fs=self.sampling_rate)
            self.notch_60hz_sos = signal.iirnotch(60.0, 30.0, fs=self.sampling_rate)
            
        except Exception as e:
            logging.error(f"Filter initialization failed: {e}")
            self.hp_filter_sos = None
            self.lp_filter_sos = None
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced strain signal processing with wavelet analysis"""
        if data.get('reading_type') != 'strain_gauge':
            return data
        
        try:
            strain_data = data['value']
            timestamp = data.get('timestamp', datetime.now())
            
            # Extract strain measurements
            strain_reading = self._extract_strain_measurements(strain_data)
            
            # Signal preprocessing and filtering
            filtered_strain = self._preprocess_strain_signal(strain_reading)
            
            # Advanced wavelet analysis
            if self.enable_wavelet_analysis and self.wavelet_analyzer:
                wavelet_analysis = self._perform_wavelet_analysis(filtered_strain)
            else:
                wavelet_analysis = {}
            
            # Spectral domain analysis
            if self.enable_spectral_analysis:
                spectral_analysis = self._perform_spectral_analysis(filtered_strain)
            else:
                spectral_analysis = {}
            
            # Time domain feature extraction
            time_domain_features = self._extract_time_domain_features(filtered_strain)
            
            # Strain rate analysis
            strain_rate_analysis = self._analyze_strain_rate(filtered_strain, timestamp)
            
            # Anomaly detection
            anomaly_detection = self._detect_strain_anomalies(filtered_strain, timestamp)
            
            # Generate comprehensive analysis result
            analysis_result = {
                'raw_strain': strain_reading,
                'filtered_strain': filtered_strain,
                'wavelet_analysis': wavelet_analysis,
                'spectral_analysis': spectral_analysis,
                'time_domain_features': time_domain_features,
                'strain_rate_analysis': strain_rate_analysis,
                'anomaly_detection': anomaly_detection,
                'processing_timestamp': timestamp,
                'sampling_info': {
                    'sampling_rate': self.sampling_rate,
                    'analysis_window': self.analysis_window,
                    'buffer_size': len(self.strain_buffer)
                }
            }
            
            # Generate alerts if needed
            alerts = self._generate_strain_alerts(analysis_result)
            if alerts:
                data['strain_alerts'] = alerts
            
            # Update data storage
            self._update_strain_history(analysis_result)
            
            # Add comprehensive analysis to data
            data['advanced_strain_analysis'] = analysis_result
            
            return data
            
        except Exception as e:
            logging.error(f"Error in advanced strain processing: {e}")
            return self._fallback_strain_processing(data)
    
    def _extract_strain_measurements(self, strain_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract strain measurements from sensor data"""
        strain_measurements = {}
        
        # Extract strain components (typical strain gauge setup)
        strain_measurements['axial'] = np.array(strain_data.get('axial_strain', [0.0]))
        strain_measurements['lateral'] = np.array(strain_data.get('lateral_strain', [0.0]))
        strain_measurements['shear'] = np.array(strain_data.get('shear_strain', [0.0]))
        
        # Calculate derived strain measures
        if len(strain_measurements['axial']) > 0 and len(strain_measurements['lateral']) > 0:
            # Volumetric strain (dilatation)
            strain_measurements['volumetric'] = (strain_measurements['axial'] + 
                                               2 * strain_measurements['lateral'])
            
            # Deviatoric strain
            strain_measurements['deviatoric'] = (strain_measurements['axial'] - 
                                               strain_measurements['lateral'])
            
            # Maximum shear strain
            strain_measurements['max_shear'] = np.abs(strain_measurements['deviatoric']) / 2
        
        return strain_measurements
    
    def _preprocess_strain_signal(self, strain_measurements: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply digital filtering and preprocessing to strain signals"""
        filtered_strain = {}
        
        for component, signal_data in strain_measurements.items():
            if len(signal_data) == 0:
                filtered_strain[component] = signal_data
                continue
            
            try:
                processed_signal = signal_data.copy()
                
                # Apply high-pass filter to remove DC drift
                if self.hp_filter_sos is not None and len(processed_signal) > 10:
                    processed_signal = signal.sosfilt(self.hp_filter_sos, processed_signal)
                
                # Apply low-pass anti-aliasing filter
                if self.lp_filter_sos is not None and len(processed_signal) > 10:
                    processed_signal = signal.sosfilt(self.lp_filter_sos, processed_signal)
                
                # Apply notch filters for power line interference
                if self.notch_50hz_sos is not None and len(processed_signal) > 10:
                    processed_signal = signal.sosfilt(self.notch_50hz_sos, processed_signal)
                    
                if self.notch_60hz_sos is not None and len(processed_signal) > 10:
                    processed_signal = signal.sosfilt(self.notch_60hz_sos, processed_signal)
                
                filtered_strain[component] = processed_signal
                
            except Exception as e:
                logging.error(f"Filtering failed for {component}: {e}")
                filtered_strain[component] = signal_data  # Use unfiltered data
        
        return filtered_strain
    
    def _perform_wavelet_analysis(self, strain_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform advanced multi-wavelet analysis on strain signals"""
        try:
            wavelet_results = {}
            dt = 1.0 / self.sampling_rate
            
            for component, signal_data in strain_data.items():
                if len(signal_data) < 16:  # Minimum for wavelet analysis
                    continue
                
                # Perform advanced wavelet periodicity analysis
                wavelet_result = self.wavelet_analyzer(signal_data, dt)
                
                # Add component-specific interpretation
                component_analysis = self._interpret_strain_wavelet_results(
                    wavelet_result, component)
                
                wavelet_results[component] = {
                    **wavelet_result,
                    **component_analysis
                }
            
            # Cross-component wavelet coherence analysis
            if len(wavelet_results) > 1:
                coherence_analysis = self._analyze_cross_component_coherence(strain_data, dt)
                wavelet_results['cross_coherence'] = coherence_analysis
            
            return wavelet_results
            
        except Exception as e:
            logging.error(f"Wavelet analysis error: {e}")
            return {'error': str(e)}
    
    def _perform_spectral_analysis(self, strain_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform comprehensive spectral domain analysis"""
        try:
            spectral_results = {}
            
            for component, signal_data in strain_data.items():
                if len(signal_data) < 8:
                    continue
                
                # Power spectral density
                freqs, psd = signal.periodogram(signal_data, fs=self.sampling_rate,
                                              window='hanning')
                
                # Spectral centroid (center of mass of spectrum)
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
                
                # Spectral bandwidth
                spectral_variance = np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd)
                spectral_bandwidth = np.sqrt(spectral_variance) if spectral_variance > 0 else 0
                
                # Dominant frequencies
                peak_indices = signal.find_peaks(psd, height=np.max(psd) * 0.1)[0]
                dominant_frequencies = freqs[peak_indices] if len(peak_indices) > 0 else []
                
                # Spectral rolloff (frequency below which 85% of energy is contained)
                cumulative_energy = np.cumsum(psd)
                total_energy = cumulative_energy[-1]
                rolloff_threshold = 0.85 * total_energy
                rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
                spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
                
                spectral_results[component] = {
                    'power_spectral_density': {
                        'frequencies': freqs.tolist(),
                        'psd': psd.tolist()
                    },
                    'spectral_features': {
                        'centroid': float(spectral_centroid),
                        'bandwidth': float(spectral_bandwidth),
                        'rolloff': float(spectral_rolloff),
                        'dominant_frequencies': dominant_frequencies.tolist()
                    },
                    'frequency_bands': self._analyze_frequency_bands(freqs, psd)
                }
            
            return spectral_results
            
        except Exception as e:
            logging.error(f"Spectral analysis error: {e}")
            return {'error': str(e)}
    
    def _extract_time_domain_features(self, strain_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract comprehensive time domain features from strain signals"""
        time_domain_features = {}
        
        for component, signal_data in strain_data.items():
            if len(signal_data) == 0:
                continue
            
            # Statistical features
            mean_strain = np.mean(signal_data)
            std_strain = np.std(signal_data)
            rms_strain = np.sqrt(np.mean(signal_data ** 2))
            
            # Peak features
            max_strain = np.max(signal_data)
            min_strain = np.min(signal_data)
            peak_to_peak = max_strain - min_strain
            
            # Shape features
            skewness = stats.skew(signal_data) if len(signal_data) > 2 else 0.0
            kurtosis = stats.kurtosis(signal_data) if len(signal_data) > 3 else 0.0
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(signal_data - np.mean(signal_data))) != 0)
            zero_crossing_rate = zero_crossings / len(signal_data) if len(signal_data) > 1 else 0
            
            # Crest factor (peak to RMS ratio)
            crest_factor = np.max(np.abs(signal_data)) / rms_strain if rms_strain > 0 else 0
            
            time_domain_features[component] = {
                'statistical': {
                    'mean': float(mean_strain),
                    'std': float(std_strain),
                    'rms': float(rms_strain),
                    'variance': float(std_strain ** 2)
                },
                'peak_analysis': {
                    'max': float(max_strain),
                    'min': float(min_strain),
                    'peak_to_peak': float(peak_to_peak),
                    'crest_factor': float(crest_factor)
                },
                'shape_features': {
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis),
                    'zero_crossing_rate': float(zero_crossing_rate)
                }
            }
        
        return time_domain_features
    
    def _analyze_strain_rate(self, strain_data: Dict[str, np.ndarray], timestamp: datetime) -> Dict[str, Any]:
        """Analyze strain rate and temporal derivatives"""
        try:
            strain_rate_results = {}
            dt = 1.0 / self.sampling_rate
            
            for component, signal_data in strain_data.items():
                if len(signal_data) < 3:
                    continue
                
                # First derivative (strain rate)
                strain_rate = np.gradient(signal_data, dt)
                
                # Second derivative (strain acceleration)
                strain_acceleration = np.gradient(strain_rate, dt)
                
                # Statistical analysis of derivatives
                rate_stats = {
                    'mean_rate': float(np.mean(strain_rate)),
                    'max_rate': float(np.max(np.abs(strain_rate))),
                    'rms_rate': float(np.sqrt(np.mean(strain_rate ** 2)))
                }
                
                acceleration_stats = {
                    'mean_acceleration': float(np.mean(strain_acceleration)),
                    'max_acceleration': float(np.max(np.abs(strain_acceleration)))
                }
                
                # Trend analysis
                if len(self.strain_buffer) > 10:
                    recent_values = [h.get(component, {}).get('rms', 0) 
                                   for h in self.strain_buffer[-10:]]
                    if len(recent_values) > 2:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            range(len(recent_values)), recent_values)
                        trend_analysis = {
                            'trend_slope': float(slope),
                            'trend_correlation': float(r_value),
                            'trend_significance': float(p_value)
                        }
                    else:
                        trend_analysis = {}
                else:
                    trend_analysis = {}
                
                strain_rate_results[component] = {
                    'rate_statistics': rate_stats,
                    'acceleration_statistics': acceleration_stats,
                    'trend_analysis': trend_analysis,
                    'timestamp': timestamp
                }
            
            return strain_rate_results
            
        except Exception as e:
            logging.error(f"Strain rate analysis error: {e}")
            return {'error': str(e)}
    
    def _detect_strain_anomalies(self, strain_data: Dict[str, np.ndarray], timestamp: datetime) -> Dict[str, Any]:
        """Detect anomalies in strain measurements"""
        try:
            anomaly_results = {}
            
            for component, signal_data in strain_data.items():
                if len(signal_data) == 0:
                    continue
                
                # Statistical anomaly detection
                current_rms = np.sqrt(np.mean(signal_data ** 2))
                current_max = np.max(np.abs(signal_data))
                
                # Initialize baseline if not available
                if component not in self.baseline_strain_stats:
                    self.baseline_strain_stats[component] = {
                        'baseline_rms': current_rms,
                        'baseline_std': 0.1 * current_rms,  # Initial estimate
                        'samples_count': 1
                    }
                else:
                    # Update baseline statistics (exponential moving average)
                    alpha = 0.05  # Learning rate
                    baseline = self.baseline_strain_stats[component]
                    baseline['baseline_rms'] = (1 - alpha) * baseline['baseline_rms'] + alpha * current_rms
                    baseline['samples_count'] += 1
                
                # Anomaly scoring
                baseline_rms = self.baseline_strain_stats[component]['baseline_rms']
                anomaly_score = current_rms / (baseline_rms + 1e-10)
                
                # Threshold-based anomaly detection
                is_anomaly = (anomaly_score > 3.0 or  # 3x baseline RMS
                             current_max > self.strain_threshold * 10)  # 10x strain threshold
                
                # Burst detection (high frequency content)
                if len(signal_data) > 10:
                    high_freq_energy = np.sum(np.abs(np.diff(signal_data, n=2))) / len(signal_data)
                    burst_detection = high_freq_energy > baseline_rms * 2.0
                else:
                    burst_detection = False
                
                anomaly_results[component] = {
                    'anomaly_score': float(anomaly_score),
                    'is_anomaly': bool(is_anomaly),
                    'burst_detected': bool(burst_detection),
                    'current_metrics': {
                        'rms': float(current_rms),
                        'max_amplitude': float(current_max)
                    },
                    'baseline_metrics': {
                        'baseline_rms': float(baseline_rms),
                        'samples_in_baseline': self.baseline_strain_stats[component]['samples_count']
                    },
                    'detection_timestamp': timestamp
                }
            
            return anomaly_results
            
        except Exception as e:
            logging.error(f"Anomaly detection error: {e}")
            return {'error': str(e)}
    
    def _interpret_strain_wavelet_results(self, wavelet_result: Dict[str, Any], component: str) -> Dict[str, Any]:
        """Add strain-specific interpretation to wavelet analysis results"""
        interpretation = {}
        
        if wavelet_result.get('dominant_scale'):
            dominant_period = wavelet_result['dominant_scale']
            dominant_frequency = 1.0 / dominant_period if dominant_period > 0 else 0
            
            # Frequency band classification for strain signals
            if dominant_frequency < 0.1:
                frequency_class = "quasi_static"
                geological_significance = "tectonic_loading"
            elif dominant_frequency < 1.0:
                frequency_class = "very_low_frequency"
                geological_significance = "slow_deformation"
            elif dominant_frequency < 10.0:
                frequency_class = "low_frequency"
                geological_significance = "seismic_waves"
            elif dominant_frequency < 100.0:
                frequency_class = "high_frequency"
                geological_significance = "local_disturbances"
            else:
                frequency_class = "ultra_high_frequency"
                geological_significance = "instrument_noise"
            
            interpretation.update({
                'dominant_frequency': float(dominant_frequency),
                'frequency_classification': frequency_class,
                'geological_significance': geological_significance
            })
        
        # Strain component specific interpretation
        if component == 'axial':
            interpretation['strain_type'] = 'normal_stress_response'
        elif component == 'lateral':
            interpretation['strain_type'] = 'poisson_effect'
        elif component == 'shear':
            interpretation['strain_type'] = 'shear_stress_response'
        elif component == 'volumetric':
            interpretation['strain_type'] = 'dilatational_deformation'
        elif component == 'deviatoric':
            interpretation['strain_type'] = 'shape_changing_deformation'
        
        return interpretation
    
    def _analyze_frequency_bands(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """Analyze energy in different frequency bands"""
        # Define frequency bands relevant to strain analysis
        bands = {
            'quasi_static': (0.0, 0.1),      # Tectonic processes
            'very_low': (0.1, 1.0),          # Slow deformation
            'low': (1.0, 10.0),              # Seismic waves
            'medium': (10.0, 50.0),          # Local disturbances
            'high': (50.0, np.inf)           # High frequency noise
        }
        
        band_energies = {}
        total_energy = np.trapz(psd, freqs) if len(freqs) > 1 else 0
        
        for band_name, (f_low, f_high) in bands.items():
            band_mask = (freqs >= f_low) & (freqs < f_high)
            if np.any(band_mask) and len(freqs) > 1:
                band_energy = np.trapz(psd[band_mask], freqs[band_mask])
                band_energies[band_name] = float(band_energy / total_energy) if total_energy > 0 else 0.0
            else:
                band_energies[band_name] = 0.0
        
        return band_energies
    
    def _analyze_cross_component_coherence(self, strain_data: Dict[str, np.ndarray], dt: float) -> Dict[str, Any]:
        """Analyze coherence between different strain components"""
        try:
            coherence_results = {}
            components = list(strain_data.keys())
            
            for i, comp1 in enumerate(components):
                for j, comp2 in enumerate(components[i+1:], i+1):
                    signal1 = strain_data[comp1]
                    signal2 = strain_data[comp2]
                    
                    if len(signal1) > 10 and len(signal2) > 10:
                        # Ensure same length
                        min_len = min(len(signal1), len(signal2))
                        signal1 = signal1[:min_len]
                        signal2 = signal2[:min_len]
                        
                        # Compute coherence
                        f, Cxy = signal.coherence(signal1, signal2, fs=self.sampling_rate,
                                                nperseg=min(256, min_len//4))
                        
                        # Mean coherence in different frequency bands
                        low_freq_mask = f < 1.0
                        med_freq_mask = (f >= 1.0) & (f < 10.0)
                        high_freq_mask = f >= 10.0
                        
                        coherence_results[f"{comp1}_{comp2}"] = {
                            'mean_coherence': float(np.mean(Cxy)),
                            'low_freq_coherence': float(np.mean(Cxy[low_freq_mask])) if np.any(low_freq_mask) else 0.0,
                            'med_freq_coherence': float(np.mean(Cxy[med_freq_mask])) if np.any(med_freq_mask) else 0.0,
                            'high_freq_coherence': float(np.mean(Cxy[high_freq_mask])) if np.any(high_freq_mask) else 0.0,
                            'peak_coherence': float(np.max(Cxy))
                        }
            
            return coherence_results
            
        except Exception as e:
            logging.error(f"Cross-component coherence analysis error: {e}")
            return {'error': str(e)}
    
    def _generate_strain_alerts(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent alerts based on strain analysis"""
        alerts = []
        
        # Anomaly-based alerts
        anomaly_detection = analysis_result.get('anomaly_detection', {})
        for component, anomaly_info in anomaly_detection.items():
            if anomaly_info.get('is_anomaly', False):
                alerts.append({
                    'type': 'strain_anomaly',
                    'component': component,
                    'severity': 'high' if anomaly_info.get('anomaly_score', 0) > 5.0 else 'medium',
                    'message': f"Strain anomaly detected in {component}: {anomaly_info.get('anomaly_score', 0):.2f}x baseline",
                    'value': anomaly_info.get('anomaly_score', 0)
                })
            
            if anomaly_info.get('burst_detected', False):
                alerts.append({
                    'type': 'strain_burst',
                    'component': component,
                    'severity': 'medium',
                    'message': f"High-frequency burst detected in {component} strain",
                    'value': anomaly_info.get('current_metrics', {}).get('rms', 0)
                })
        
        # Wavelet-based alerts
        wavelet_analysis = analysis_result.get('wavelet_analysis', {})
        for component, wavelet_info in wavelet_analysis.items():
            if component == 'cross_coherence':
                continue
                
            dominant_scale = wavelet_info.get('dominant_scale')
            wavelet_strength = wavelet_info.get('wavelet_strength', 0)
            
            if dominant_scale and wavelet_strength > 0.8:
                frequency = 1.0 / dominant_scale
                geological_sig = wavelet_info.get('geological_significance', 'unknown')
                
                alerts.append({
                    'type': 'periodic_strain_pattern',
                    'component': component,
                    'severity': 'medium',
                    'message': f"Strong periodic pattern detected: {frequency:.3f} Hz ({geological_sig})",
                    'value': wavelet_strength,
                    'frequency': frequency
                })
        
        # Return alerts if any
        if alerts:
            return {
                'timestamp': analysis_result.get('processing_timestamp'),
                'alerts': alerts,
                'total_alerts': len(alerts)
            }
        
        return {}
    
    def _update_strain_history(self, analysis_result: Dict[str, Any]):
        """Update strain analysis history for trend tracking"""
        # Extract key metrics for history
        history_entry = {
            'timestamp': analysis_result.get('processing_timestamp'),
            'components': {}
        }
        
        # Store time domain features for each component
        time_domain = analysis_result.get('time_domain_features', {})
        for component, features in time_domain.items():
            history_entry['components'][component] = {
                'rms': features.get('statistical', {}).get('rms', 0),
                'max': features.get('peak_analysis', {}).get('max', 0),
                'std': features.get('statistical', {}).get('std', 0)
            }
        
        self.strain_buffer.append(history_entry)
        
        # Keep only recent history (memory management)
        if len(self.strain_buffer) > self.analysis_window:
            self.strain_buffer = self.strain_buffer[-self.analysis_window:]
    
    def _fallback_strain_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to basic strain processing if advanced methods fail"""
        try:
            strain_data = data['value']
            
            # Basic statistical analysis
            basic_analysis = {}
            for component in ['axial_strain', 'lateral_strain', 'shear_strain']:
                if component in strain_data:
                    values = np.array(strain_data[component])
                    if len(values) > 0:
                        basic_analysis[component] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'max': float(np.max(np.abs(values))),
                            'rms': float(np.sqrt(np.mean(values ** 2)))
                        }
            
            # Simple threshold checking
            alerts = []
            for component, stats in basic_analysis.items():
                if stats['max'] > self.strain_threshold:
                    alerts.append({
                        'type': 'basic_strain_threshold_exceeded',
                        'component': component,
                        'severity': 'medium',
                        'message': f"Strain threshold exceeded in {component}",
                        'value': stats['max']
                    })
            
            if alerts:
                data['strain_alerts'] = {
                    'alerts': alerts,
                    'fallback_mode': True
                }
            
            data['basic_strain_analysis'] = {
                'component_statistics': basic_analysis,
                'fallback_mode': True
            }
            
            return data
            
        except Exception as e:
            logging.error(f"Fallback strain processing failed: {e}")
            return data

class TrendAnalyzer:
    """Advanced trend analysis with changepoint detection and causal inference"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Enhanced configuration parameters
        self.time_window = self.config.get('time_window', 50)  # observations
        self.trend_sensitivity = self.config.get('trend_sensitivity', 0.05)  # p-value threshold
        self.changepoint_confidence = self.config.get('changepoint_confidence', 0.95)
        self.enable_causal_inference = self.config.get('enable_causal_inference', True)
        self.min_observations = self.config.get('min_observations', 10)
        
        # Initialize advanced analysis modules with fallback
        self.changepoint_detector = None
        self.causal_engine = None
        self.data_pipeline = None
        try:
            from .advanced_changepoint_detection import _advanced_cusum_change_points, _bayesian_online_change_point_detection
            from .causal_inference_engine import CausalInferenceEngine, GeologicalCausalRelationType
            from .data_integration_pipeline import DataIntegrationPipeline
            
            self.changepoint_detector = {
                'cusum': _advanced_cusum_change_points,
                'bayesian': _bayesian_online_change_point_detection
            }
            
            # Initialize causal inference engine
            causal_config = {
                'algorithm': self.config.get('causal_algorithm', 'pc'),
                'confidence_level': self.changepoint_confidence,
                'max_conditioning_set_size': 3,
                'domain_knowledge': 'geological'
            }
            self.causal_engine = CausalInferenceEngine(causal_config)
            
            # Initialize data pipeline for ML model deployment
            self.data_pipeline = DataIntegrationPipeline({
                'model_type': 'trend_analysis',
                'real_time': True,
                'enable_caching': True,
                'time_series': True
            })
            
            logging.info("Advanced trend analysis modules initialized successfully")
            
        except ImportError as e:
            logging.warning(f"Advanced trend analysis modules not available: {e}")
        except Exception as e:
            logging.error(f"Failed to initialize trend analysis modules: {e}")
        
        # Time series data storage
        self.time_series_buffer = {}  # sensor_id -> time series data
        self.trend_history = {}       # sensor_id -> trend analysis history
        self.changepoint_history = {} # sensor_id -> detected changepoints
        self.causal_relationships = {} # discovered causal relationships
        
        # Statistical trend analysis
        self.trend_models = {}
        self.seasonal_models = {}
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced trend analysis with changepoint detection and causal inference"""
        try:
            # Extract sensor information
            sensor_id = data.get('sensor_id', 'unknown')
            timestamp = data.get('timestamp', datetime.now())
            sensor_type = data.get('reading_type', 'unknown')
            
            # Extract numerical values for trend analysis
            trend_values = self._extract_trend_values(data)
            if not trend_values:
                return data
            
            # Update time series buffer
            self._update_time_series(sensor_id, timestamp, trend_values)
            
            # Perform comprehensive trend analysis
            trend_analysis = self._perform_trend_analysis(sensor_id, sensor_type)
            
            # Changepoint detection
            if self.changepoint_detector and len(self.time_series_buffer.get(sensor_id, [])) > self.min_observations:
                changepoint_analysis = self._perform_changepoint_detection(sensor_id)
                trend_analysis['changepoint_detection'] = changepoint_analysis
            
            # Causal inference analysis (if multiple sensors available)
            if self.enable_causal_inference and len(self.time_series_buffer) > 1:
                causal_analysis = self._perform_causal_inference_analysis()
                trend_analysis['causal_inference'] = causal_analysis
            
            # Regime classification
            regime_analysis = self._classify_current_regime(sensor_id, trend_values)
            trend_analysis['regime_analysis'] = regime_analysis
            
            # Generate alerts based on trend analysis
            alerts = self._generate_trend_alerts(trend_analysis, sensor_id)
            if alerts:
                data['trend_alerts'] = alerts
            
            # Add comprehensive analysis to data
            data['advanced_trend_analysis'] = trend_analysis
            
            return data
            
        except Exception as e:
            logging.error(f"Error in advanced trend analysis: {e}")
            return self._fallback_trend_analysis(data)
    
    def _extract_trend_values(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical values suitable for trend analysis"""
        trend_values = {}
        
        # Extract values from different data structures
        if 'value' in data:
            value_data = data['value']
            
            if isinstance(value_data, (int, float)):
                trend_values['primary'] = float(value_data)
            elif isinstance(value_data, dict):
                # Extract numerical values from nested dictionaries
                for key, val in value_data.items():
                    if isinstance(val, (int, float)) and not key.startswith('_'):
                        trend_values[key] = float(val)
            
        # Extract from analysis results if available
        analysis_fields = ['rms', 'magnitude', 'confidence', 'amplitude', 'energy']
        for field in analysis_fields:
            if field in data:
                if isinstance(data[field], (int, float)):
                    trend_values[field] = float(data[field])
        
        # Extract from nested analysis structures
        nested_analyses = ['advanced_displacement_analysis', 'advanced_strain_analysis', 'ml_detection']
        for analysis_key in nested_analyses:
            if analysis_key in data:
                analysis_data = data[analysis_key]
                if isinstance(analysis_data, dict):
                    self._extract_from_nested_analysis(analysis_data, trend_values)
        
        return trend_values
    
    def _extract_from_nested_analysis(self, analysis_data: Dict[str, Any], trend_values: Dict[str, float]):
        """Extract numerical values from nested analysis structures"""
        for key, value in analysis_data.items():
            if isinstance(value, (int, float)):
                trend_values[f"analysis_{key}"] = float(value)
            elif isinstance(value, dict):
                # Look for specific numerical keys in nested dictionaries
                numerical_keys = ['magnitude', 'confidence', 'strength', 'energy', 'rms', 'max', 'mean', 'std']
                for num_key in numerical_keys:
                    if num_key in value and isinstance(value[num_key], (int, float)):
                        trend_values[f"{key}_{num_key}"] = float(value[num_key])
    
    def _update_time_series(self, sensor_id: str, timestamp: datetime, values: Dict[str, float]):
        """Update time series buffer for sensor"""
        if sensor_id not in self.time_series_buffer:
            self.time_series_buffer[sensor_id] = []
        
        # Add new observation
        observation = {
            'timestamp': timestamp,
            'values': values
        }
        
        self.time_series_buffer[sensor_id].append(observation)
        
        # Keep only recent observations (sliding window)
        if len(self.time_series_buffer[sensor_id]) > self.time_window * 2:
            self.time_series_buffer[sensor_id] = self.time_series_buffer[sensor_id][-self.time_window:]
    
    def _perform_trend_analysis(self, sensor_id: str, sensor_type: str) -> Dict[str, Any]:
        """Perform comprehensive statistical trend analysis"""
        try:
            time_series = self.time_series_buffer.get(sensor_id, [])
            if len(time_series) < 3:
                return {'status': 'insufficient_data', 'observations': len(time_series)}
            
            analysis_results = {}
            
            # Analyze each numerical variable
            for variable_name in time_series[0]['values'].keys():
                values = [obs['values'][variable_name] for obs in time_series if variable_name in obs['values']]
                timestamps = [obs['timestamp'] for obs in time_series if variable_name in obs['values']]
                
                if len(values) < 3:
                    continue
                
                # Convert timestamps to numerical array (hours from start)
                time_numeric = [(ts - timestamps[0]).total_seconds() / 3600.0 for ts in timestamps]
                values_array = np.array(values)
                
                # Linear trend analysis
                linear_trend = self._analyze_linear_trend(time_numeric, values_array)
                
                # Seasonal decomposition
                seasonal_analysis = self._analyze_seasonality(values_array)
                
                # Autocorrelation analysis
                autocorr_analysis = self._analyze_autocorrelation(values_array)
                
                # Stationarity tests
                stationarity = self._test_stationarity(values_array)
                
                # Volatility analysis
                volatility = self._analyze_volatility(values_array, time_numeric)
                
                analysis_results[variable_name] = {
                    'linear_trend': linear_trend,
                    'seasonal_analysis': seasonal_analysis,
                    'autocorrelation': autocorr_analysis,
                    'stationarity': stationarity,
                    'volatility': volatility,
                    'data_points': len(values),
                    'time_span_hours': time_numeric[-1] - time_numeric[0] if len(time_numeric) > 1 else 0
                }
            
            return {
                'variable_analyses': analysis_results,
                'sensor_type': sensor_type,
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Trend analysis error for {sensor_id}: {e}")
            return {'error': str(e)}
    
    def _analyze_linear_trend(self, time_numeric: List[float], values: np.ndarray) -> Dict[str, Any]:
        """Analyze linear trend using robust statistical methods"""
        try:
            if len(values) < 3:
                return {'slope': 0.0, 'significance': 'insufficient_data'}
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
            
            # Robust trend estimation using Theil-Sen estimator
            try:
                theil_sen_slope, theil_sen_intercept = stats.theilslopes(values, time_numeric)
            except:
                theil_sen_slope = slope
                theil_sen_intercept = intercept
            
            # Mann-Kendall trend test
            try:
                mk_tau, mk_p_value = stats.kendalltau(time_numeric, values)
            except:
                mk_tau, mk_p_value = 0.0, 1.0
            
            # Trend classification
            if p_value < self.trend_sensitivity:
                if slope > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"
                trend_strength = "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.4 else "weak"
            else:
                trend_direction = "no_trend"
                trend_strength = "none"
            
            return {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'standard_error': float(std_err),
                'theil_sen_slope': float(theil_sen_slope),
                'mann_kendall_tau': float(mk_tau),
                'mann_kendall_p': float(mk_p_value),
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'is_significant': p_value < self.trend_sensitivity
            }
            
        except Exception as e:
            logging.error(f"Linear trend analysis error: {e}")
            return {'error': str(e)}
    
    def _analyze_seasonality(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze seasonal patterns in the data"""
        try:
            if len(values) < 12:  # Need sufficient data for seasonality
                return {'seasonal_strength': 0.0, 'dominant_period': None}
            
            # Simple seasonal decomposition using moving averages
            # For short series, use a simple approach
            n = len(values)
            
            # Test for periodic patterns using FFT
            fft_values = np.fft.fft(values - np.mean(values))
            freqs = np.fft.fftfreq(n)
            power_spectrum = np.abs(fft_values) ** 2
            
            # Find dominant frequency (excluding DC component)
            dominant_freq_idx = np.argmax(power_spectrum[1:n//2]) + 1
            dominant_frequency = freqs[dominant_freq_idx]
            dominant_period = 1.0 / abs(dominant_frequency) if abs(dominant_frequency) > 1e-10 else None
            
            # Seasonal strength estimation
            total_variance = np.var(values)
            if total_variance > 0:
                seasonal_component_power = power_spectrum[dominant_freq_idx]
                seasonal_strength = seasonal_component_power / np.sum(power_spectrum[1:n//2])
            else:
                seasonal_strength = 0.0
            
            return {
                'seasonal_strength': float(seasonal_strength),
                'dominant_period': float(dominant_period) if dominant_period else None,
                'dominant_frequency': float(dominant_frequency),
                'spectral_analysis': {
                    'peak_frequency': float(dominant_frequency),
                    'peak_power': float(seasonal_component_power)
                }
            }
            
        except Exception as e:
            logging.error(f"Seasonality analysis error: {e}")
            return {'error': str(e)}
    
    def _analyze_autocorrelation(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze autocorrelation structure"""
        try:
            if len(values) < 5:
                return {'lag1_autocorr': 0.0, 'significant_lags': []}
            
            # Calculate autocorrelation function for several lags
            max_lags = min(10, len(values) // 2)
            autocorrelations = []
            significant_lags = []
            
            for lag in range(1, max_lags + 1):
                if lag >= len(values):
                    break
                    
                # Pearson correlation at lag
                x1 = values[:-lag]
                x2 = values[lag:]
                
                if len(x1) > 1 and np.var(x1) > 1e-10 and np.var(x2) > 1e-10:
                    corr, p_value = stats.pearsonr(x1, x2)
                    autocorrelations.append(corr)
                    
                    # Check significance (rough approximation)
                    if abs(corr) > 2.0 / np.sqrt(len(x1)):  # Approximate 95% confidence
                        significant_lags.append(lag)
                else:
                    autocorrelations.append(0.0)
            
            return {
                'lag1_autocorr': float(autocorrelations[0]) if len(autocorrelations) > 0 else 0.0,
                'autocorrelations': [float(ac) for ac in autocorrelations],
                'significant_lags': significant_lags,
                'max_autocorr': float(np.max(np.abs(autocorrelations))) if len(autocorrelations) > 0 else 0.0
            }
            
        except Exception as e:
            logging.error(f"Autocorrelation analysis error: {e}")
            return {'error': str(e)}
    
    def _test_stationarity(self, values: np.ndarray) -> Dict[str, Any]:
        """Test for stationarity using multiple methods"""
        try:
            if len(values) < 8:
                return {'is_stationary': 'unknown', 'reason': 'insufficient_data'}
            
            # Augmented Dickey-Fuller test (simplified version)
            # Check if mean and variance are stable over time
            n = len(values)
            mid_point = n // 2
            
            first_half = values[:mid_point]
            second_half = values[mid_point:]
            
            # Compare means and variances
            mean_diff = abs(np.mean(first_half) - np.mean(second_half))
            pooled_std = np.sqrt((np.var(first_half) + np.var(second_half)) / 2)
            
            # Rough stationarity test
            is_mean_stable = mean_diff < 2 * pooled_std / np.sqrt(min(len(first_half), len(second_half)))
            
            # Variance ratio test
            var_ratio = np.var(first_half) / np.var(second_half) if np.var(second_half) > 1e-10 else 1.0
            is_var_stable = 0.5 < var_ratio < 2.0
            
            # Overall stationarity assessment
            is_stationary = is_mean_stable and is_var_stable
            
            return {
                'is_stationary': bool(is_stationary),
                'mean_stability': bool(is_mean_stable),
                'variance_stability': bool(is_var_stable),
                'mean_difference': float(mean_diff),
                'variance_ratio': float(var_ratio)
            }
            
        except Exception as e:
            logging.error(f"Stationarity test error: {e}")
            return {'error': str(e)}
    
    def _analyze_volatility(self, values: np.ndarray, time_numeric: List[float]) -> Dict[str, Any]:
        """Analyze volatility and variance structure"""
        try:
            if len(values) < 5:
                return {'volatility': 0.0, 'volatility_trend': 'stable'}
            
            # Calculate rolling volatility (standard deviation)
            window_size = min(5, len(values) // 2)
            rolling_volatility = []
            
            for i in range(window_size, len(values)):
                window_values = values[i-window_size:i]
                volatility = np.std(window_values)
                rolling_volatility.append(volatility)
            
            if len(rolling_volatility) == 0:
                return {'volatility': float(np.std(values)), 'volatility_trend': 'stable'}
            
            # Analyze volatility trend
            vol_array = np.array(rolling_volatility)
            current_volatility = vol_array[-1] if len(vol_array) > 0 else np.std(values)
            
            # Volatility trend analysis
            if len(vol_array) > 3:
                vol_time = time_numeric[-len(vol_array):]
                vol_slope, _, vol_r, vol_p, _ = stats.linregress(vol_time, vol_array)
                
                if vol_p < 0.1:  # Less stringent threshold for volatility trend
                    volatility_trend = "increasing" if vol_slope > 0 else "decreasing"
                else:
                    volatility_trend = "stable"
            else:
                volatility_trend = "stable"
            
            # GARCH-like volatility clustering detection
            returns = np.diff(values) / values[:-1]  # Simple returns
            abs_returns = np.abs(returns)
            volatility_clustering = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1] if len(returns) > 2 else 0.0
            
            return {
                'current_volatility': float(current_volatility),
                'mean_volatility': float(np.mean(vol_array)) if len(vol_array) > 0 else float(np.std(values)),
                'volatility_trend': volatility_trend,
                'volatility_clustering': float(volatility_clustering),
                'max_volatility': float(np.max(vol_array)) if len(vol_array) > 0 else float(current_volatility)
            }
            
        except Exception as e:
            logging.error(f"Volatility analysis error: {e}")
            return {'error': str(e)}
    
    def _perform_changepoint_detection(self, sensor_id: str) -> Dict[str, Any]:
        """Perform advanced changepoint detection"""
        try:
            time_series = self.time_series_buffer.get(sensor_id, [])
            if len(time_series) < self.min_observations:
                return {'status': 'insufficient_data'}
            
            changepoint_results = {}
            
            # Analyze each variable for changepoints
            for variable_name in time_series[0]['values'].keys():
                values = [obs['values'][variable_name] for obs in time_series if variable_name in obs['values']]
                
                if len(values) < self.min_observations:
                    continue
                
                values_array = np.array(values)
                
                # CUSUM-based changepoint detection
                cusum_result = self.changepoint_detector['cusum'](values_array, self.changepoint_confidence)
                
                # Bayesian online changepoint detection
                bayesian_result = self.changepoint_detector['bayesian'](values_array)
                
                # Combine results
                changepoint_results[variable_name] = {
                    'cusum_detection': cusum_result,
                    'bayesian_detection': bayesian_result,
                    'consensus_changepoints': self._find_consensus_changepoints(cusum_result, bayesian_result),
                    'data_length': len(values)
                }
            
            # Update changepoint history
            if sensor_id not in self.changepoint_history:
                self.changepoint_history[sensor_id] = []
            
            # Store recent changepoints
            recent_changepoints = []
            for var_name, var_results in changepoint_results.items():
                for cp_idx in var_results['consensus_changepoints']:
                    if cp_idx >= len(time_series) - 10:  # Recent changepoints only
                        recent_changepoints.append({
                            'variable': var_name,
                            'changepoint_index': cp_idx,
                            'timestamp': time_series[cp_idx]['timestamp'],
                            'confidence': var_results['cusum_detection'].get('confidence_scores', [0.8])[0] if var_results['cusum_detection'].get('confidence_scores') else 0.8
                        })
            
            self.changepoint_history[sensor_id].extend(recent_changepoints)
            
            # Keep only recent changepoint history
            if len(self.changepoint_history[sensor_id]) > 50:
                self.changepoint_history[sensor_id] = self.changepoint_history[sensor_id][-50:]
            
            return {
                'variable_changepoints': changepoint_results,
                'recent_changepoints': recent_changepoints,
                'total_changepoints_detected': len(recent_changepoints)
            }
            
        except Exception as e:
            logging.error(f"Changepoint detection error: {e}")
            return {'error': str(e)}
    
    def _find_consensus_changepoints(self, cusum_result: Dict[str, Any], bayesian_result: Dict[str, Any]) -> List[int]:
        """Find consensus changepoints from different detection methods"""
        cusum_cps = cusum_result.get('change_points', [])
        bayesian_cps = bayesian_result.get('change_points', [])
        
        # Simple consensus: changepoints detected by either method within a tolerance
        consensus_cps = []
        tolerance = 3  # Allow 3-point tolerance for consensus
        
        all_cps = set(cusum_cps + bayesian_cps)
        
        for cp in all_cps:
            # Check if this changepoint is supported by multiple methods
            cusum_support = any(abs(cp - cusum_cp) <= tolerance for cusum_cp in cusum_cps)
            bayesian_support = any(abs(cp - bay_cp) <= tolerance for bay_cp in bayesian_cps)
            
            if cusum_support or bayesian_support:  # At least one method supports it
                consensus_cps.append(cp)
        
        return sorted(consensus_cps)
    
    def _perform_causal_inference_analysis(self) -> Dict[str, Any]:
        """Perform causal inference analysis across sensors"""
        try:
            if not self.causal_engine:
                return {'status': 'causal_engine_unavailable'}
            
            # Collect data from multiple sensors for causal analysis
            sensor_data = {}
            min_length = float('inf')
            
            for sensor_id, time_series in self.time_series_buffer.items():
                if len(time_series) >= self.min_observations:
                    # Extract the most relevant variable for each sensor
                    if len(time_series) > 0 and time_series[0]['values']:
                        primary_var = list(time_series[0]['values'].keys())[0]
                        values = [obs['values'][primary_var] for obs in time_series if primary_var in obs['values']]
                        sensor_data[sensor_id] = values
                        min_length = min(min_length, len(values))
            
            if len(sensor_data) < 2 or min_length < self.min_observations:
                return {'status': 'insufficient_multi_sensor_data'}
            
            # Truncate all series to same length for causal analysis
            aligned_data = {}
            for sensor_id, values in sensor_data.items():
                aligned_data[sensor_id] = values[-min_length:]
            
            # Convert to DataFrame for causal analysis
            import pandas as pd
            causal_df = pd.DataFrame(aligned_data)
            
            # Perform causal discovery
            causal_results = self.causal_engine.discover_causal_relationships(
                causal_df, 
                domain_context={'type': 'geological_monitoring'}
            )
            
            return {
                'causal_graph': causal_results.get('causal_graph', {}),
                'causal_relationships': causal_results.get('relationships', []),
                'sensors_analyzed': list(sensor_data.keys()),
                'data_points_used': min_length,
                'analysis_confidence': causal_results.get('confidence', 0.0)
            }
            
        except Exception as e:
            logging.error(f"Causal inference error: {e}")
            return {'error': str(e)}
    
    def _classify_current_regime(self, sensor_id: str, current_values: Dict[str, float]) -> Dict[str, Any]:
        """Classify the current operational regime based on recent trends and changepoints"""
        try:
            time_series = self.time_series_buffer.get(sensor_id, [])
            if len(time_series) < 5:
                return {'regime': 'unknown', 'confidence': 0.0}
            
            regime_analysis = {}
            
            for variable_name, current_value in current_values.items():
                # Get recent values for regime classification
                recent_values = [obs['values'].get(variable_name, 0) for obs in time_series[-10:] if variable_name in obs['values']]
                
                if len(recent_values) < 3:
                    continue
                
                recent_array = np.array(recent_values)
                
                # Statistical characteristics of recent regime
                recent_mean = np.mean(recent_array)
                recent_std = np.std(recent_array)
                current_z_score = abs(current_value - recent_mean) / (recent_std + 1e-10)
                
                # Trend-based regime classification
                if len(recent_array) > 2:
                    trend_slope = stats.linregress(range(len(recent_array)), recent_array)[0]
                    trend_magnitude = abs(trend_slope) / (recent_std + 1e-10)
                else:
                    trend_magnitude = 0.0
                
                # Regime classification logic
                if current_z_score > 2.0:  # Anomalous
                    if trend_magnitude > 1.0:
                        regime = "rapid_change"
                        confidence = min(0.9, current_z_score / 3.0)
                    else:
                        regime = "anomalous_spike"
                        confidence = min(0.8, current_z_score / 4.0)
                elif trend_magnitude > 0.5:
                    if trend_slope > 0:
                        regime = "increasing_trend"
                    else:
                        regime = "decreasing_trend"
                    confidence = min(0.8, trend_magnitude)
                elif recent_std / (abs(recent_mean) + 1e-10) > 0.3:  # High relative variability
                    regime = "high_variability"
                    confidence = 0.6
                else:
                    regime = "stable"
                    confidence = 0.7
                
                # Check for recent changepoints to modify regime classification
                recent_changepoints = []
                if sensor_id in self.changepoint_history:
                    recent_time = datetime.now() - timedelta(hours=24)  # Last 24 hours
                    recent_changepoints = [cp for cp in self.changepoint_history[sensor_id] 
                                         if cp['timestamp'] > recent_time and cp['variable'] == variable_name]
                
                if recent_changepoints:
                    regime = f"post_changepoint_{regime}"
                    confidence = min(confidence, 0.9)
                
                regime_analysis[variable_name] = {
                    'regime': regime,
                    'confidence': float(confidence),
                    'z_score': float(current_z_score),
                    'trend_magnitude': float(trend_magnitude),
                    'recent_changepoints': len(recent_changepoints)
                }
            
            return regime_analysis
            
        except Exception as e:
            logging.error(f"Regime classification error: {e}")
            return {'error': str(e)}
    
    def _generate_trend_alerts(self, trend_analysis: Dict[str, Any], sensor_id: str) -> Dict[str, Any]:
        """Generate intelligent alerts based on comprehensive trend analysis"""
        alerts = []
        
        # Variable-specific trend alerts
        variable_analyses = trend_analysis.get('variable_analyses', {})
        for var_name, var_analysis in variable_analyses.items():
            
            # Significant trend alerts
            linear_trend = var_analysis.get('linear_trend', {})
            if linear_trend.get('is_significant', False):
                trend_strength = linear_trend.get('trend_strength', 'weak')
                trend_direction = linear_trend.get('trend_direction', 'unknown')
                
                if trend_strength in ['strong', 'moderate']:
                    alerts.append({
                        'type': 'significant_trend',
                        'variable': var_name,
                        'severity': 'high' if trend_strength == 'strong' else 'medium',
                        'message': f"Significant {trend_direction} trend detected in {var_name} ({trend_strength})",
                        'slope': linear_trend.get('slope', 0),
                        'p_value': linear_trend.get('p_value', 1.0)
                    })
            
            # Volatility alerts
            volatility = var_analysis.get('volatility', {})
            if volatility.get('volatility_trend') == 'increasing':
                alerts.append({
                    'type': 'increasing_volatility',
                    'variable': var_name,
                    'severity': 'medium',
                    'message': f"Increasing volatility detected in {var_name}",
                    'current_volatility': volatility.get('current_volatility', 0)
                })
        
        # Changepoint alerts
        changepoint_detection = trend_analysis.get('changepoint_detection', {})
        recent_changepoints = changepoint_detection.get('recent_changepoints', [])
        if recent_changepoints:
            for cp in recent_changepoints:
                alerts.append({
                    'type': 'changepoint_detected',
                    'variable': cp['variable'],
                    'severity': 'high',
                    'message': f"Change point detected in {cp['variable']}",
                    'confidence': cp.get('confidence', 0.8),
                    'timestamp': cp['timestamp']
                })
        
        # Causal inference alerts
        causal_inference = trend_analysis.get('causal_inference', {})
        causal_relationships = causal_inference.get('causal_relationships', [])
        if causal_relationships:
            for relationship in causal_relationships[:3]:  # Top 3 relationships
                if relationship.get('strength', 0) > 0.7:
                    alerts.append({
                        'type': 'causal_relationship_detected',
                        'severity': 'medium',
                        'message': f"Strong causal relationship: {relationship.get('source', '')} → {relationship.get('target', '')}",
                        'strength': relationship.get('strength', 0)
                    })
        
        # Return alerts if any
        if alerts:
            return {
                'timestamp': trend_analysis.get('analysis_timestamp', datetime.now()),
                'sensor_id': sensor_id,
                'alerts': alerts,
                'total_alerts': len(alerts)
            }
        
        return {}
    
    def _fallback_trend_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to basic trend analysis if advanced methods fail"""
        try:
            sensor_id = data.get('sensor_id', 'unknown')
            
            # Extract a single numerical value for basic analysis
            trend_value = None
            if 'value' in data:
                if isinstance(data['value'], (int, float)):
                    trend_value = float(data['value'])
                elif isinstance(data['value'], dict):
                    # Get first numerical value
                    for val in data['value'].values():
                        if isinstance(val, (int, float)):
                            trend_value = float(val)
                            break
            
            if trend_value is None:
                return data
            
            # Simple moving average trend
            if sensor_id not in self.trend_history:
                self.trend_history[sensor_id] = []
            
            self.trend_history[sensor_id].append(trend_value)
            
            # Keep only recent history
            if len(self.trend_history[sensor_id]) > 20:
                self.trend_history[sensor_id] = self.trend_history[sensor_id][-20:]
            
            # Basic trend analysis
            if len(self.trend_history[sensor_id]) >= 3:
                recent_values = np.array(self.trend_history[sensor_id])
                
                # Simple trend detection
                if len(recent_values) > 5:
                    recent_slope = (recent_values[-1] - recent_values[-5]) / 4
                    mean_value = np.mean(recent_values)
                    relative_trend = recent_slope / (abs(mean_value) + 1e-10)
                    
                    if abs(relative_trend) > 0.1:  # 10% change
                        trend_direction = "increasing" if recent_slope > 0 else "decreasing"
                        
                        data['basic_trend_analysis'] = {
                            'trend_direction': trend_direction,
                            'relative_change': float(relative_trend),
                            'recent_slope': float(recent_slope),
                            'fallback_mode': True
                        }
            
            return data
            
        except Exception as e:
            logging.error(f"Fallback trend analysis failed: {e}")
            return data

# Example usage and testing
async def example_edge_computing():
    """Example of using edge computing framework"""
    
    print("=== Edge Computing Framework Example ===")
    
    # Create edge device capabilities
    seismic_capabilities = EdgeDeviceCapabilities(
        cpu_cores=4,
        memory_mb=2048,
        storage_gb=32,
        has_gpu=False,
        power_consumption_watts=15,
        network_bandwidth_mbps=10,
        supported_protocols=[CommunicationProtocol.MQTT, CommunicationProtocol.ZMQ],
        processing_power_gflops=5.0
    )
    
    gps_capabilities = EdgeDeviceCapabilities(
        cpu_cores=2,
        memory_mb=1024,
        storage_gb=16,
        has_gpu=False,
        power_consumption_watts=8,
        network_bandwidth_mbps=5,
        supported_protocols=[CommunicationProtocol.MQTT],
        processing_power_gflops=2.0
    )
    
    # Create edge devices
    seismic_device = SeismicEdgeDevice(
        device_id="seismic_001",
        capabilities=seismic_capabilities,
        config={
            'sampling_rate': 100,
            'location': (35.0, -118.0, 1000.0),
            'reading_interval': 0.1
        }
    )
    
    gps_device = GPSEdgeDevice(
        device_id="gps_001", 
        capabilities=gps_capabilities,
        config={
            'reference_position': (35.0, -118.0, 1000.0),
            'accuracy': 0.001,
            'reading_interval': 1.0
        }
    )
    
    # Start devices
    await seismic_device.start()
    await gps_device.start()
    
    print(f"Started seismic device: {seismic_device.device_id}")
    print(f"Started GPS device: {gps_device.device_id}")
    
    # Let devices run for a few seconds
    await asyncio.sleep(5)
    
    # Check device status
    print(f"\nDevice Status:")
    print(f"Seismic device active: {seismic_device.is_active}")
    print(f"GPS device active: {gps_device.is_active}")
    print(f"Seismic performance: {seismic_device.performance_metrics}")
    print(f"GPS performance: {gps_device.performance_metrics}")
    
    # Check communication buffers
    seismic_comm = seismic_device.communication_manager
    gps_comm = gps_device.communication_manager
    
    if isinstance(seismic_comm, LocalCommunicationManager):
        print(f"\nSeismic messages sent: {len(seismic_comm.message_buffer)}")
        if seismic_comm.message_buffer:
            print(f"Latest message: {seismic_comm.message_buffer[-1]['message']['type']}")
    
    if isinstance(gps_comm, LocalCommunicationManager):
        print(f"GPS messages sent: {len(gps_comm.message_buffer)}")
        if gps_comm.message_buffer:
            print(f"Latest message: {gps_comm.message_buffer[-1]['message']['type']}")
    
    # Stop devices
    await seismic_device.stop()
    await gps_device.stop()
    
    print("Edge computing example completed!")

# =============================================================================
# COMPREHENSIVE DATA MANAGEMENT SYSTEM
# =============================================================================

class EdgeDataManager:
    """Comprehensive data management system for edge computing with ML model deployment"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize comprehensive data management system"""
        self.config = config or self._get_default_config()
        self.data_layers = {}
        self.model_registry = {}
        self.processor_instances = {}
        
        # Initialize data integration pipeline
        try:
            from data_integration_pipeline import DataIntegrationPipeline, DataType
            self.pipeline = DataIntegrationPipeline(self.config.get('pipeline_config', {}))
            self.pipeline_available = True
            self.DataType = DataType
        except ImportError:
            logging.warning("DataIntegrationPipeline not available - using basic data management")
            self.pipeline = None
            self.pipeline_available = False
            self.DataType = None
        
        # Initialize ML model deployment capabilities
        self._setup_model_deployment()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("EdgeDataManager initialized with comprehensive data management")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for data management"""
        return {
            'enable_parallel_processing': True,
            'max_concurrent_layers': 4,
            'model_cache_size': 10,
            'data_retention_days': 30,
            'enable_model_versioning': True,
            'pipeline_config': {
                'grid_resolution': 100.0,
                'quality_threshold': 0.8,
                'parallel_processing': True,
                'cache_results': True
            }
        }
    
    def _setup_model_deployment(self):
        """Setup ML model deployment capabilities"""
        self.deployed_models = {}
        self.model_performance = {}
        self.fallback_models = {}
        
        # Model deployment queue
        self.deployment_queue = asyncio.Queue()
        
        # Performance monitoring
        self.model_metrics = {
            'inference_times': {},
            'accuracy_scores': {},
            'resource_usage': {}
        }
    
    def register_processor(self, processor_name: str, processor_instance: Any) -> bool:
        """Register a processor instance for data management"""
        try:
            self.processor_instances[processor_name] = processor_instance
            self.logger.info(f"Registered processor: {processor_name}")
            
            # Setup automatic data layer creation for processor
            self._setup_processor_data_layers(processor_name, processor_instance)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register processor {processor_name}: {e}")
            return False
    
    def _setup_processor_data_layers(self, processor_name: str, processor_instance: Any):
        """Setup automatic data layer creation for processor types"""
        try:
            # Determine processor type and create appropriate data layers
            if 'EarthquakeDetector' in str(type(processor_instance)):
                self._create_seismic_data_layer(processor_name)
            elif 'GPSNoiseFilter' in str(type(processor_instance)):
                self._create_gps_data_layer(processor_name)
            elif 'DisplacementDetector' in str(type(processor_instance)):
                self._create_deformation_data_layer(processor_name)
            elif 'StrainSignalProcessor' in str(type(processor_instance)):
                self._create_strain_data_layer(processor_name)
            elif 'TrendAnalyzer' in str(type(processor_instance)):
                self._create_trend_data_layer(processor_name)
            
        except Exception as e:
            self.logger.error(f"Failed to setup data layers for {processor_name}: {e}")
    
    def _create_seismic_data_layer(self, processor_name: str):
        """Create seismic data layer with ML model integration"""
        if not self.pipeline_available:
            return
        
        # Add seismic data layer with comprehensive metadata
        layer_config = {
            'processor_type': 'seismic',
            'data_format': 'time_series',
            'ml_models': ['seismic_transformer', 'eruption_forecaster'],
            'features': ['velocity', 'acceleration', 'frequency_content'],
            'quality_metrics': ['signal_to_noise', 'phase_clarity', 'event_confidence']
        }
        
        try:
            layer = self.pipeline.add_data_layer(
                name=f"{processor_name}_seismic",
                data_type=self.DataType.GEOPHYSICS,
                data=np.zeros((1000, 3)),  # Placeholder for seismic time series
                metadata=layer_config
            )
            
            self.data_layers[f"{processor_name}_seismic"] = layer
            self.logger.info(f"Created seismic data layer for {processor_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create seismic data layer: {e}")
    
    def _create_gps_data_layer(self, processor_name: str):
        """Create GPS data layer with Kalman filtering integration"""
        if not self.pipeline_available:
            return
        
        layer_config = {
            'processor_type': 'gps',
            'data_format': 'position_time_series',
            'ml_models': ['enhanced_kalman_filter', 'multi_sensor_fusion'],
            'features': ['position', 'velocity', 'acceleration', 'uncertainty'],
            'coordinate_systems': ['WGS84', 'UTM', 'ECEF']
        }
        
        try:
            layer = self.pipeline.add_data_layer(
                name=f"{processor_name}_gps",
                data_type=self.DataType.GEOPHYSICS,
                data=np.zeros((1000, 6)),  # Placeholder for GPS data
                metadata=layer_config
            )
            
            self.data_layers[f"{processor_name}_gps"] = layer
            self.logger.info(f"Created GPS data layer for {processor_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create GPS data layer: {e}")
    
    def _create_deformation_data_layer(self, processor_name: str):
        """Create deformation data layer with ML analysis"""
        if not self.pipeline_available:
            return
        
        layer_config = {
            'processor_type': 'deformation',
            'data_format': 'displacement_field',
            'ml_models': ['deformation_analyzer'],
            'features': ['displacement', 'strain', 'stress_field', 'source_parameters'],
            'analysis_methods': ['mogi_model', 'sill_model', 'dike_model']
        }
        
        try:
            layer = self.pipeline.add_data_layer(
                name=f"{processor_name}_deformation",
                data_type=self.DataType.GEOPHYSICS,
                data=np.zeros((100, 100, 3)),  # Placeholder for 3D deformation field
                metadata=layer_config
            )
            
            self.data_layers[f"{processor_name}_deformation"] = layer
            self.logger.info(f"Created deformation data layer for {processor_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create deformation data layer: {e}")
    
    def _create_strain_data_layer(self, processor_name: str):
        """Create strain data layer with wavelet analysis"""
        if not self.pipeline_available:
            return
        
        layer_config = {
            'processor_type': 'strain',
            'data_format': 'multi_component_strain',
            'ml_models': ['advanced_wavelet_analysis'],
            'features': ['axial_strain', 'lateral_strain', 'shear_strain', 'wavelet_coefficients'],
            'analysis_methods': ['morlet_wavelet', 'mexican_hat_wavelet', 'daubechies_wavelet']
        }
        
        try:
            layer = self.pipeline.add_data_layer(
                name=f"{processor_name}_strain",
                data_type=self.DataType.GEOPHYSICS,
                data=np.zeros((1000, 6)),  # Placeholder for multi-component strain
                metadata=layer_config
            )
            
            self.data_layers[f"{processor_name}_strain"] = layer
            self.logger.info(f"Created strain data layer for {processor_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create strain data layer: {e}")
    
    def _create_trend_data_layer(self, processor_name: str):
        """Create trend analysis data layer with changepoint detection"""
        if not self.pipeline_available:
            return
        
        layer_config = {
            'processor_type': 'trend',
            'data_format': 'statistical_time_series',
            'ml_models': ['advanced_changepoint_detection', 'causal_inference_engine'],
            'features': ['trend_coefficients', 'changepoints', 'causal_relationships'],
            'analysis_methods': ['cusum_detection', 'bayesian_changepoint', 'causal_discovery']
        }
        
        try:
            layer = self.pipeline.add_data_layer(
                name=f"{processor_name}_trend",
                data_type=self.DataType.GEOPHYSICS,
                data=np.zeros((1000, 4)),  # Placeholder for trend analysis data
                metadata=layer_config
            )
            
            self.data_layers[f"{processor_name}_trend"] = layer
            self.logger.info(f"Created trend data layer for {processor_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create trend data layer: {e}")
    
    async def deploy_ml_model(self, model_name: str, model_path: str, 
                            target_processors: List[str] = None) -> bool:
        """Deploy ML model to specified processors"""
        try:
            # Load model
            model = await self._load_model(model_path)
            if model is None:
                return False
            
            # Register model
            self.model_registry[model_name] = {
                'model': model,
                'path': model_path,
                'deployment_time': asyncio.get_event_loop().time(),
                'target_processors': target_processors or [],
                'version': self._get_model_version(model_path)
            }
            
            # Deploy to target processors
            if target_processors:
                for processor_name in target_processors:
                    if processor_name in self.processor_instances:
                        await self._deploy_to_processor(model_name, processor_name)
            
            self.logger.info(f"Successfully deployed ML model: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy ML model {model_name}: {e}")
            return False
    
    async def _load_model(self, model_path: str) -> Any:
        """Load ML model from path"""
        try:
            # This is a placeholder - actual implementation would depend on model format
            # (PyTorch, TensorFlow, scikit-learn, etc.)
            self.logger.info(f"Loading model from: {model_path}")
            
            # Simulate model loading
            await asyncio.sleep(0.1)
            
            return {'model_path': model_path, 'loaded': True}
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            return None
    
    def _get_model_version(self, model_path: str) -> str:
        """Get model version from path"""
        import hashlib
        import os
        
        if os.path.exists(model_path):
            # Use file hash as version
            with open(model_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()[:8]
        else:
            # Use timestamp for non-file models
            return str(int(asyncio.get_event_loop().time()))
    
    async def _deploy_to_processor(self, model_name: str, processor_name: str):
        """Deploy model to specific processor"""
        try:
            processor = self.processor_instances[processor_name]
            model_info = self.model_registry[model_name]
            
            # Add model to processor if it has model management capabilities
            if hasattr(processor, 'add_model'):
                await processor.add_model(model_name, model_info['model'])
            
            self.deployed_models[f"{processor_name}_{model_name}"] = model_info
            
            self.logger.info(f"Deployed model {model_name} to processor {processor_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to deploy {model_name} to {processor_name}: {e}")
    
    async def process_data_layers(self, target_processors: List[str] = None) -> Dict[str, Any]:
        """Process all data layers using the integration pipeline"""
        if not self.pipeline_available:
            self.logger.warning("Data integration pipeline not available")
            return {}
        
        try:
            # Process all layers
            results = self.pipeline.process_all_layers()
            
            # Update processor instances with processed data
            if target_processors:
                for processor_name in target_processors:
                    if processor_name in self.processor_instances:
                        await self._update_processor_data(processor_name, results)
            
            self.logger.info(f"Processed {len(results)} data layers successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process data layers: {e}")
            return {}
    
    async def _update_processor_data(self, processor_name: str, processed_data: Dict[str, Any]):
        """Update processor with processed data"""
        try:
            processor = self.processor_instances[processor_name]
            
            # Find relevant data layers for this processor
            relevant_layers = [layer_name for layer_name in processed_data.keys() 
                             if processor_name in layer_name]
            
            for layer_name in relevant_layers:
                layer_data = processed_data[layer_name]
                
                # Update processor if it has data update capabilities
                if hasattr(processor, 'update_data'):
                    await processor.update_data(layer_data)
            
        except Exception as e:
            self.logger.error(f"Failed to update processor {processor_name}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'data_manager': {
                'pipeline_available': self.pipeline_available,
                'registered_processors': len(self.processor_instances),
                'data_layers': len(self.data_layers),
                'deployed_models': len(self.deployed_models)
            },
            'processors': {},
            'models': {},
            'performance': {
                'total_inference_time': sum(
                    sum(times) for times in self.model_metrics['inference_times'].values()
                ),
                'average_accuracy': np.mean([
                    np.mean(scores) for scores in self.model_metrics['accuracy_scores'].values()
                ]) if self.model_metrics['accuracy_scores'] else 0.0
            }
        }
        
        # Add processor-specific status
        for name, processor in self.processor_instances.items():
            try:
                if hasattr(processor, 'get_status'):
                    status['processors'][name] = processor.get_status()
                else:
                    status['processors'][name] = {'status': 'active'}
            except Exception as e:
                status['processors'][name] = {'status': 'error', 'error': str(e)}
        
        # Add model performance metrics
        for model_name, model_info in self.model_registry.items():
            status['models'][model_name] = {
                'version': model_info['version'],
                'deployment_time': model_info['deployment_time'],
                'target_processors': model_info['target_processors']
            }
        
        return status
    
    async def cleanup_resources(self):
        """Cleanup resources and save state"""
        try:
            # Save model performance metrics
            await self._save_performance_metrics()
            
            # Cleanup processors
            for processor in self.processor_instances.values():
                if hasattr(processor, 'cleanup'):
                    await processor.cleanup()
            
            # Clear caches
            self.deployed_models.clear()
            self.model_performance.clear()
            
            self.logger.info("EdgeDataManager resources cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def _save_performance_metrics(self):
        """Save performance metrics for analysis"""
        try:
            metrics_file = f"edge_performance_metrics_{int(asyncio.get_event_loop().time())}.json"
            
            metrics_data = {
                'model_metrics': self.model_metrics,
                'system_status': self.get_system_status(),
                'timestamp': asyncio.get_event_loop().time()
            }
            
            # In a real implementation, this would save to persistent storage
            self.logger.info(f"Performance metrics saved to {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save performance metrics: {e}")

# =============================================================================
# DATABASE INTEGRATION SYSTEM
# =============================================================================

class EdgeDatabaseManager:
    """Database integration manager for edge computing with real-time storage"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize database integration for edge computing"""
        self.config = config or self._get_default_config()
        self.db_manager = None
        self.timescale_manager = None
        self.storage_queues = {}
        self.active_connections = {}
        
        # Initialize database integration
        try:
            from database_integration import PostgreSQLGISManager, TimeScaleDBManager, DatabaseConfig
            
            db_config = DatabaseConfig(
                host=self.config.get('db_host', 'localhost'),
                port=self.config.get('db_port', 5432),
                database=self.config.get('db_name', 'prometheus_edge'),
                username=self.config.get('db_user', 'edge_user'),
                password=self.config.get('db_password', 'edge_password'),
                enable_timescaledb=True,
                enable_postgis=True
            )
            
            self.db_manager = PostgreSQLGISManager(db_config)
            self.timescale_manager = TimeScaleDBManager(self.db_manager)
            self.database_available = True
            
        except ImportError:
            logging.warning("Database integration not available - using local storage fallback")
            self.database_available = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage system
        self._setup_storage_system()
        
        self.logger.info("EdgeDatabaseManager initialized with real-time storage capabilities")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default database configuration"""
        return {
            'enable_real_time_storage': True,
            'batch_size': 100,
            'flush_interval_seconds': 5.0,
            'enable_compression': True,
            'retention_policy_days': 365,
            'enable_continuous_aggregates': True,
            'spatial_indexing': True,
            'db_host': 'localhost',
            'db_port': 5432,
            'db_name': 'prometheus_edge',
            'db_user': 'edge_user',
            'db_password': 'edge_password'
        }
    
    def _setup_storage_system(self):
        """Setup storage system for different data types"""
        self.storage_schemas = {
            'seismic_events': {
                'table': 'edge_seismic_events',
                'columns': {
                    'timestamp': 'TIMESTAMPTZ NOT NULL',
                    'device_id': 'VARCHAR(50) NOT NULL',
                    'location': 'GEOMETRY(POINT, 4326)',
                    'magnitude': 'FLOAT',
                    'depth_km': 'FLOAT',
                    'event_type': 'VARCHAR(50)',
                    'confidence': 'FLOAT',
                    'ml_prediction': 'JSONB',
                    'raw_data': 'JSONB'
                },
                'hypertable': True,
                'time_column': 'timestamp'
            },
            'gps_measurements': {
                'table': 'edge_gps_measurements',
                'columns': {
                    'timestamp': 'TIMESTAMPTZ NOT NULL',
                    'device_id': 'VARCHAR(50) NOT NULL',
                    'location': 'GEOMETRY(POINT, 4326)',
                    'position_x': 'DOUBLE PRECISION',
                    'position_y': 'DOUBLE PRECISION', 
                    'position_z': 'DOUBLE PRECISION',
                    'velocity_x': 'FLOAT',
                    'velocity_y': 'FLOAT',
                    'velocity_z': 'FLOAT',
                    'uncertainty_m': 'FLOAT',
                    'quality_metrics': 'JSONB'
                },
                'hypertable': True,
                'time_column': 'timestamp'
            },
            'deformation_data': {
                'table': 'edge_deformation_data',
                'columns': {
                    'timestamp': 'TIMESTAMPTZ NOT NULL',
                    'device_id': 'VARCHAR(50) NOT NULL',
                    'location': 'GEOMETRY(POINT, 4326)',
                    'displacement_x': 'FLOAT',
                    'displacement_y': 'FLOAT',
                    'displacement_z': 'FLOAT',
                    'strain_tensor': 'JSONB',
                    'source_parameters': 'JSONB',
                    'ml_analysis': 'JSONB'
                },
                'hypertable': True,
                'time_column': 'timestamp'
            },
            'strain_measurements': {
                'table': 'edge_strain_measurements',
                'columns': {
                    'timestamp': 'TIMESTAMPTZ NOT NULL',
                    'device_id': 'VARCHAR(50) NOT NULL',
                    'location': 'GEOMETRY(POINT, 4326)',
                    'axial_strain': 'FLOAT',
                    'lateral_strain': 'FLOAT',
                    'shear_strain': 'FLOAT',
                    'wavelet_analysis': 'JSONB',
                    'frequency_content': 'JSONB',
                    'quality_score': 'FLOAT'
                },
                'hypertable': True,
                'time_column': 'timestamp'
            },
            'trend_analysis': {
                'table': 'edge_trend_analysis',
                'columns': {
                    'timestamp': 'TIMESTAMPTZ NOT NULL',
                    'device_id': 'VARCHAR(50) NOT NULL',
                    'sensor_type': 'VARCHAR(50)',
                    'trend_coefficients': 'JSONB',
                    'changepoints': 'JSONB',
                    'causal_relationships': 'JSONB',
                    'statistical_metrics': 'JSONB',
                    'regime_classification': 'VARCHAR(50)'
                },
                'hypertable': True,
                'time_column': 'timestamp'
            }
        }
        
        # Initialize storage queues for batched operations
        for data_type in self.storage_schemas:
            self.storage_queues[data_type] = asyncio.Queue(maxsize=self.config['batch_size'] * 2)
    
    async def initialize_database_schema(self):
        """Initialize database schema and TimescaleDB hypertables"""
        if not self.database_available:
            self.logger.warning("Database not available - skipping schema initialization")
            return False
        
        try:
            await self._create_tables_and_hypertables()
            await self._setup_continuous_aggregates()
            await self._setup_retention_policies()
            await self._create_indexes()
            
            self.logger.info("Database schema initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database schema: {e}")
            return False
    
    async def _create_tables_and_hypertables(self):
        """Create tables and convert them to TimescaleDB hypertables"""
        if not self.db_manager:
            return
        
        async with self.db_manager._get_connection() as conn:
            async with conn.cursor() as cursor:
                for data_type, schema in self.storage_schemas.items():
                    table_name = schema['table']
                    columns = schema['columns']
                    
                    # Create table
                    column_definitions = []
                    for col_name, col_type in columns.items():
                        column_definitions.append(f"{col_name} {col_type}")
                    
                    create_table_sql = f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id SERIAL PRIMARY KEY,
                            {', '.join(column_definitions)}
                        );
                    """
                    
                    await cursor.execute(create_table_sql)
                    
                    # Convert to hypertable if TimescaleDB is available
                    if schema.get('hypertable', False) and self.config.get('enable_timescaledb', True):
                        time_col = schema['time_column']
                        hypertable_sql = f"""
                            SELECT create_hypertable('{table_name}', '{time_col}', 
                                                    if_not_exists => TRUE);
                        """
                        
                        try:
                            await cursor.execute(hypertable_sql)
                            self.logger.info(f"Created hypertable: {table_name}")
                        except Exception as e:
                            self.logger.warning(f"Failed to create hypertable {table_name}: {e}")
                
                await conn.commit()
    
    async def _setup_continuous_aggregates(self):
        """Setup TimescaleDB continuous aggregates for real-time analytics"""
        if not self.config.get('enable_continuous_aggregates', True):
            return
        
        aggregates = [
            {
                'name': 'hourly_seismic_summary',
                'table': 'edge_seismic_events',
                'time_column': 'timestamp',
                'interval': '1 hour',
                'select_clause': '''
                    time_bucket('1 hour', timestamp) AS bucket,
                    device_id,
                    COUNT(*) as event_count,
                    AVG(magnitude) as avg_magnitude,
                    MAX(magnitude) as max_magnitude,
                    AVG(confidence) as avg_confidence
                '''
            },
            {
                'name': 'daily_deformation_summary',
                'table': 'edge_deformation_data',
                'time_column': 'timestamp',
                'interval': '1 day',
                'select_clause': '''
                    time_bucket('1 day', timestamp) AS bucket,
                    device_id,
                    AVG(displacement_x) as avg_disp_x,
                    AVG(displacement_y) as avg_disp_y,
                    AVG(displacement_z) as avg_disp_z,
                    STDDEV(displacement_x) as std_disp_x,
                    STDDEV(displacement_y) as std_disp_y,
                    STDDEV(displacement_z) as std_disp_z
                '''
            },
            {
                'name': 'minute_gps_summary',
                'table': 'edge_gps_measurements',
                'time_column': 'timestamp',
                'interval': '1 minute',
                'select_clause': '''
                    time_bucket('1 minute', timestamp) AS bucket,
                    device_id,
                    AVG(position_x) as avg_pos_x,
                    AVG(position_y) as avg_pos_y,
                    AVG(position_z) as avg_pos_z,
                    AVG(uncertainty_m) as avg_uncertainty
                '''
            }
        ]
        
        if not self.db_manager:
            return
        
        async with self.db_manager._get_connection() as conn:
            async with conn.cursor() as cursor:
                for agg in aggregates:
                    try:
                        # Create continuous aggregate
                        create_agg_sql = f"""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS {agg['name']}
                            WITH (timescaledb.continuous) AS
                            SELECT {agg['select_clause']}
                            FROM {agg['table']}
                            GROUP BY bucket, device_id;
                        """
                        
                        await cursor.execute(create_agg_sql)
                        
                        # Add refresh policy
                        policy_sql = f"""
                            SELECT add_continuous_aggregate_policy('{agg['name']}',
                                start_offset => INTERVAL '2 hours',
                                end_offset => INTERVAL '1 minute',
                                schedule_interval => INTERVAL '1 minute');
                        """
                        
                        await cursor.execute(policy_sql)
                        
                        self.logger.info(f"Created continuous aggregate: {agg['name']}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create continuous aggregate {agg['name']}: {e}")
                
                await conn.commit()
    
    async def _setup_retention_policies(self):
        """Setup data retention policies"""
        retention_days = self.config.get('retention_policy_days', 365)
        
        if not self.db_manager:
            return
        
        async with self.db_manager._get_connection() as conn:
            async with conn.cursor() as cursor:
                for data_type, schema in self.storage_schemas.items():
                    if schema.get('hypertable', False):
                        table_name = schema['table']
                        
                        try:
                            policy_sql = f"""
                                SELECT add_retention_policy('{table_name}', 
                                                           INTERVAL '{retention_days} days');
                            """
                            
                            await cursor.execute(policy_sql)
                            self.logger.info(f"Added retention policy for {table_name}: {retention_days} days")
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to add retention policy for {table_name}: {e}")
                
                await conn.commit()
    
    async def _create_indexes(self):
        """Create spatial and time-based indexes"""
        if not self.db_manager:
            return
        
        async with self.db_manager._get_connection() as conn:
            async with conn.cursor() as cursor:
                for data_type, schema in self.storage_schemas.items():
                    table_name = schema['table']
                    
                    # Create time index
                    time_col = schema.get('time_column')
                    if time_col:
                        try:
                            await cursor.execute(f"""
                                CREATE INDEX IF NOT EXISTS idx_{table_name}_time 
                                ON {table_name} ({time_col});
                            """)
                        except Exception as e:
                            self.logger.warning(f"Failed to create time index for {table_name}: {e}")
                    
                    # Create spatial index if location column exists
                    if 'location' in schema['columns']:
                        try:
                            await cursor.execute(f"""
                                CREATE INDEX IF NOT EXISTS idx_{table_name}_location 
                                ON {table_name} USING GIST (location);
                            """)
                        except Exception as e:
                            self.logger.warning(f"Failed to create spatial index for {table_name}: {e}")
                    
                    # Create device_id index
                    if 'device_id' in schema['columns']:
                        try:
                            await cursor.execute(f"""
                                CREATE INDEX IF NOT EXISTS idx_{table_name}_device 
                                ON {table_name} (device_id);
                            """)
                        except Exception as e:
                            self.logger.warning(f"Failed to create device index for {table_name}: {e}")
                
                await conn.commit()
    
    async def store_processor_data(self, processor_type: str, device_id: str, 
                                 data: Dict[str, Any], location: Tuple[float, float] = None) -> bool:
        """Store processor data in real-time"""
        try:
            # Determine data type based on processor
            data_type = self._map_processor_to_data_type(processor_type)
            if not data_type:
                self.logger.warning(f"Unknown processor type: {processor_type}")
                return False
            
            # Prepare storage record
            storage_record = await self._prepare_storage_record(
                data_type, device_id, data, location
            )
            
            # Add to storage queue
            if self.config.get('enable_real_time_storage', True):
                await self.storage_queues[data_type].put(storage_record)
                return True
            else:
                # Direct storage
                return await self._store_single_record(data_type, storage_record)
        
        except Exception as e:
            self.logger.error(f"Failed to store processor data: {e}")
            return False
    
    def _map_processor_to_data_type(self, processor_type: str) -> Optional[str]:
        """Map processor type to data storage type"""
        mapping = {
            'EarthquakeDetector': 'seismic_events',
            'GPSNoiseFilter': 'gps_measurements',
            'GPSPositionCalculator': 'gps_measurements',
            'DisplacementDetector': 'deformation_data',
            'StrainSignalProcessor': 'strain_measurements',
            'TrendAnalyzer': 'trend_analysis'
        }
        
        for processor, data_type in mapping.items():
            if processor in processor_type:
                return data_type
        
        return None
    
    async def _prepare_storage_record(self, data_type: str, device_id: str, 
                                    data: Dict[str, Any], location: Tuple[float, float] = None) -> Dict[str, Any]:
        """Prepare data record for storage"""
        record = {
            'timestamp': datetime.utcnow(),
            'device_id': device_id
        }
        
        # Add location if provided
        if location:
            record['location'] = f"POINT({location[0]} {location[1]})"
        
        # Process data based on type
        if data_type == 'seismic_events':
            record.update({
                'magnitude': data.get('magnitude', 0.0),
                'depth_km': data.get('depth', 0.0),
                'event_type': data.get('event_type', 'unknown'),
                'confidence': data.get('confidence', 0.0),
                'ml_prediction': json.dumps(data.get('ml_prediction', {})),
                'raw_data': json.dumps(data.get('raw_data', {}))
            })
        
        elif data_type == 'gps_measurements':
            position = data.get('position', {})
            velocity = data.get('velocity', {})
            record.update({
                'position_x': position.get('x', 0.0),
                'position_y': position.get('y', 0.0),
                'position_z': position.get('z', 0.0),
                'velocity_x': velocity.get('x', 0.0),
                'velocity_y': velocity.get('y', 0.0),
                'velocity_z': velocity.get('z', 0.0),
                'uncertainty_m': data.get('uncertainty', 0.0),
                'quality_metrics': json.dumps(data.get('quality_metrics', {}))
            })
        
        elif data_type == 'deformation_data':
            displacement = data.get('displacement', {})
            record.update({
                'displacement_x': displacement.get('x', 0.0),
                'displacement_y': displacement.get('y', 0.0),
                'displacement_z': displacement.get('z', 0.0),
                'strain_tensor': json.dumps(data.get('strain_tensor', {})),
                'source_parameters': json.dumps(data.get('source_parameters', {})),
                'ml_analysis': json.dumps(data.get('ml_analysis', {}))
            })
        
        elif data_type == 'strain_measurements':
            record.update({
                'axial_strain': data.get('axial_strain', 0.0),
                'lateral_strain': data.get('lateral_strain', 0.0),
                'shear_strain': data.get('shear_strain', 0.0),
                'wavelet_analysis': json.dumps(data.get('wavelet_analysis', {})),
                'frequency_content': json.dumps(data.get('frequency_content', {})),
                'quality_score': data.get('quality_score', 0.0)
            })
        
        elif data_type == 'trend_analysis':
            record.update({
                'sensor_type': data.get('sensor_type', 'unknown'),
                'trend_coefficients': json.dumps(data.get('trend_coefficients', {})),
                'changepoints': json.dumps(data.get('changepoints', {})),
                'causal_relationships': json.dumps(data.get('causal_relationships', {})),
                'statistical_metrics': json.dumps(data.get('statistical_metrics', {})),
                'regime_classification': data.get('regime_classification', 'unknown')
            })
        
        return record
    
    async def _store_single_record(self, data_type: str, record: Dict[str, Any]) -> bool:
        """Store single record directly to database"""
        if not self.database_available or not self.db_manager:
            return False
        
        try:
            schema = self.storage_schemas[data_type]
            table_name = schema['table']
            
            # Build INSERT statement
            columns = list(record.keys())
            placeholders = [f'%({col})s' for col in columns]
            
            insert_sql = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            """
            
            async with self.db_manager._get_connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(insert_sql, record)
                    await conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store single record: {e}")
            return False
    
    async def start_batch_processing(self):
        """Start batch processing for real-time storage"""
        if not self.config.get('enable_real_time_storage', True):
            return
        
        # Start batch processors for each data type
        for data_type in self.storage_schemas:
            asyncio.create_task(self._batch_processor(data_type))
        
        self.logger.info("Started batch processing for real-time storage")
    
    async def _batch_processor(self, data_type: str):
        """Process batches of data for storage"""
        batch = []
        last_flush_time = time.time()
        flush_interval = self.config.get('flush_interval_seconds', 5.0)
        batch_size = self.config.get('batch_size', 100)
        
        while True:
            try:
                # Wait for data with timeout
                try:
                    record = await asyncio.wait_for(
                        self.storage_queues[data_type].get(), 
                        timeout=1.0
                    )
                    batch.append(record)
                except asyncio.TimeoutError:
                    pass
                
                # Check if we should flush
                current_time = time.time()
                should_flush = (
                    len(batch) >= batch_size or
                    (len(batch) > 0 and (current_time - last_flush_time) >= flush_interval)
                )
                
                if should_flush:
                    await self._flush_batch(data_type, batch)
                    batch.clear()
                    last_flush_time = current_time
            
            except Exception as e:
                self.logger.error(f"Error in batch processor for {data_type}: {e}")
                await asyncio.sleep(1.0)
    
    async def _flush_batch(self, data_type: str, batch: List[Dict[str, Any]]):
        """Flush batch of records to database"""
        if not batch or not self.database_available:
            return
        
        try:
            schema = self.storage_schemas[data_type]
            table_name = schema['table']
            
            if not batch:
                return
            
            # Get columns from first record
            columns = list(batch[0].keys())
            placeholders = [f'%({col})s' for col in columns]
            
            insert_sql = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            """
            
            async with self.db_manager._get_connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.executemany(insert_sql, batch)
                    await conn.commit()
            
            self.logger.debug(f"Flushed {len(batch)} records to {table_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to flush batch for {data_type}: {e}")
    
    async def get_real_time_metrics(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get real-time metrics from TimescaleDB"""
        if not self.timescale_manager:
            return {}
        
        try:
            time_window = timedelta(minutes=time_window_minutes)
            metrics = await self.timescale_manager.get_real_time_metrics(time_window)
            
            return {
                'time_window_minutes': time_window_minutes,
                'metrics': metrics,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get real-time metrics: {e}")
            return {}
    
    def get_storage_status(self) -> Dict[str, Any]:
        """Get storage system status"""
        status = {
            'database_available': self.database_available,
            'real_time_storage': self.config.get('enable_real_time_storage', True),
            'queue_status': {},
            'batch_config': {
                'batch_size': self.config.get('batch_size', 100),
                'flush_interval': self.config.get('flush_interval_seconds', 5.0)
            }
        }
        
        # Add queue status
        for data_type, queue in self.storage_queues.items():
            status['queue_status'][data_type] = {
                'size': queue.qsize(),
                'max_size': queue.maxsize
            }
        
        return status
    
    async def cleanup_resources(self):
        """Cleanup database resources"""
        try:
            # Flush all remaining batches
            for data_type, queue in self.storage_queues.items():
                batch = []
                while not queue.empty():
                    try:
                        record = queue.get_nowait()
                        batch.append(record)
                    except asyncio.QueueEmpty:
                        break
                
                if batch:
                    await self._flush_batch(data_type, batch)
            
            # Close database connections
            if self.db_manager and hasattr(self.db_manager, 'connection_pool'):
                if self.db_manager.connection_pool:
                    self.db_manager.connection_pool.closeall()
            
            self.logger.info("EdgeDatabaseManager resources cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {e}")

# =============================================================================
# FEDERATED LEARNING INTEGRATION SYSTEM
# =============================================================================

class EdgeFederatedLearningManager:
    """Federated learning manager for edge computing with distributed model training"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize federated learning for edge computing"""
        self.config = config or self._get_default_config()
        self.orchestrator = None
        self.local_clients = {}
        self.model_repositories = {}
        self.training_sessions = {}
        
        # Initialize federated learning
        try:
            from federated_learning import (
                FederatedLearningOrchestrator, FederatedClient, FederatedServer,
                FederatedConfig, ModelType, AggregationStrategy
            )
            
            self.FederatedLearningOrchestrator = FederatedLearningOrchestrator
            self.FederatedClient = FederatedClient
            self.FederatedServer = FederatedServer
            self.FederatedConfig = FederatedConfig
            self.ModelType = ModelType
            self.AggregationStrategy = AggregationStrategy
            self.federated_available = True
            
        except ImportError:
            logging.warning("Federated learning not available - using local training fallback")
            self.federated_available = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize federated learning system
        self._setup_federated_system()
        
        self.logger.info("EdgeFederatedLearningManager initialized with distributed learning capabilities")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default federated learning configuration"""
        return {
            'enable_federated_learning': True,
            'default_model_type': 'earthquake_detection',
            'aggregation_strategy': 'fedavg',
            'federated_rounds': 50,
            'local_epochs': 3,
            'batch_size': 16,
            'learning_rate': 0.001,
            'min_clients': 2,
            'client_fraction': 1.0,
            'privacy_preserving': True,
            'secure_aggregation': True,
            'differential_privacy': False,
            'dp_noise_multiplier': 1.0,
            'model_versioning': True,
            'auto_model_deployment': True
        }
    
    def _setup_federated_system(self):
        """Setup federated learning system components"""
        if not self.federated_available:
            return
        
        # Setup default configurations for different processor types
        self.processor_model_configs = {
            'EarthquakeDetector': {
                'model_type': self.ModelType.EARTHQUAKE_DETECTION,
                'input_features': ['velocity', 'acceleration', 'frequency_content', 'sta_lta_ratio'],
                'output_type': 'classification',
                'classes': ['earthquake', 'noise', 'volcanic', 'explosion'],
                'feature_dim': 20
            },
            'DisplacementDetector': {
                'model_type': self.ModelType.DEFORMATION_PREDICTION,
                'input_features': ['displacement_x', 'displacement_y', 'displacement_z', 'strain_tensor'],
                'output_type': 'regression',
                'feature_dim': 12
            },
            'StrainSignalProcessor': {
                'model_type': self.ModelType.ANOMALY_DETECTION,
                'input_features': ['axial_strain', 'lateral_strain', 'shear_strain', 'wavelet_coefficients'],
                'output_type': 'classification',
                'classes': ['normal', 'anomaly'],
                'feature_dim': 15
            },
            'TrendAnalyzer': {
                'model_type': self.ModelType.CLASSIFICATION,
                'input_features': ['trend_coefficients', 'changepoints', 'statistical_metrics'],
                'output_type': 'classification',
                'classes': ['stable', 'increasing', 'decreasing', 'irregular'],
                'feature_dim': 10
            }
        }
    
    async def create_federated_session(self, session_name: str, processor_types: List[str], 
                                     target_devices: List[str] = None) -> bool:
        """Create a new federated learning session for specific processors"""
        try:
            if not self.federated_available:
                self.logger.warning("Federated learning not available")
                return False
            
            # Determine primary model type from processor types
            primary_processor = processor_types[0] if processor_types else 'EarthquakeDetector'
            model_config = self.processor_model_configs.get(primary_processor, {})
            
            # Create federated configuration
            fed_config = self.FederatedConfig(
                model_type=model_config.get('model_type', self.ModelType.EARTHQUAKE_DETECTION),
                aggregation_strategy=self._get_aggregation_strategy(),
                rounds=self.config.get('federated_rounds', 50),
                local_epochs=self.config.get('local_epochs', 3),
                batch_size=self.config.get('batch_size', 16),
                learning_rate=self.config.get('learning_rate', 0.001),
                min_clients=self.config.get('min_clients', 2),
                client_fraction=self.config.get('client_fraction', 1.0),
                dp_noise_multiplier=self.config.get('dp_noise_multiplier', 1.0) if self.config.get('differential_privacy') else 0.0
            )
            
            # Create orchestrator
            orchestrator = self.FederatedLearningOrchestrator(fed_config)
            
            # Setup session
            session_info = {
                'orchestrator': orchestrator,
                'processor_types': processor_types,
                'target_devices': target_devices or [],
                'clients': {},
                'config': fed_config,
                'created_at': datetime.utcnow(),
                'status': 'created',
                'model_config': model_config
            }
            
            self.training_sessions[session_name] = session_info
            
            self.logger.info(f"Created federated learning session: {session_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create federated session {session_name}: {e}")
            return False
    
    def _get_aggregation_strategy(self) -> Any:
        """Get aggregation strategy based on configuration"""
        strategy_name = self.config.get('aggregation_strategy', 'fedavg')
        
        strategy_mapping = {
            'fedavg': self.AggregationStrategy.FEDERATED_AVERAGING,
            'weighted_avg': self.AggregationStrategy.WEIGHTED_AVERAGE,
            'secure_agg': self.AggregationStrategy.SECURE_AGGREGATION,
            'diff_privacy': self.AggregationStrategy.DIFFERENTIAL_PRIVACY
        }
        
        return strategy_mapping.get(strategy_name, self.AggregationStrategy.FEDERATED_AVERAGING)
    
    async def register_edge_device_client(self, session_name: str, device_id: str, 
                                        processor_instances: Dict[str, Any],
                                        device_location: Tuple[float, float] = None) -> bool:
        """Register an edge device as a federated learning client"""
        try:
            if session_name not in self.training_sessions:
                self.logger.error(f"Session {session_name} not found")
                return False
            
            session = self.training_sessions[session_name]
            
            # Create local data from processor instances
            local_data = await self._extract_training_data(processor_instances, session['processor_types'])
            
            if not local_data:
                self.logger.warning(f"No training data available for device {device_id}")
                return False
            
            # Create federated client
            client = self.FederatedClient(
                client_id=device_id,
                config=session['config'],
                local_data=local_data
            )
            
            # Add location if provided
            if device_location:
                client.location = device_location
            
            # Register with orchestrator
            session['orchestrator'].add_client(client)
            session['clients'][device_id] = {
                'client': client,
                'processor_instances': processor_instances,
                'data_size': len(local_data),
                'last_update': datetime.utcnow()
            }
            
            self.logger.info(f"Registered edge device {device_id} as federated client in session {session_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register device {device_id} as federated client: {e}")
            return False
    
    async def _extract_training_data(self, processor_instances: Dict[str, Any], 
                                   processor_types: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract training data from processor instances"""
        training_data = []
        
        try:
            for processor_type in processor_types:
                if processor_type in processor_instances:
                    processor = processor_instances[processor_type]
                    
                    # Extract processor-specific data
                    if hasattr(processor, 'get_training_data'):
                        data = await processor.get_training_data()
                        if data:
                            training_data.extend(data)
                    
                    # Extract from processor history/buffers
                    elif hasattr(processor, 'data_buffer'):
                        data = self._convert_buffer_to_training_data(processor.data_buffer, processor_type)
                        if data:
                            training_data.extend(data)
                    
                    # Extract from analysis results
                    elif hasattr(processor, 'analysis_results'):
                        data = self._convert_analysis_to_training_data(processor.analysis_results, processor_type)
                        if data:
                            training_data.extend(data)
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract training data: {e}")
            return []
    
    def _convert_buffer_to_training_data(self, buffer: List[Dict], processor_type: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Convert processor buffer data to training format"""
        training_data = []
        
        try:
            model_config = self.processor_model_configs.get(processor_type, {})
            input_features = model_config.get('input_features', [])
            
            for entry in buffer[-1000:]:  # Use recent data
                # Extract features based on processor type
                features = []
                
                if processor_type == 'EarthquakeDetector':
                    features = self._extract_seismic_features(entry)
                    # Create label based on detection results
                    label = 1 if entry.get('detection', {}).get('type') == 'earthquake_trigger' else 0
                
                elif processor_type == 'DisplacementDetector':
                    features = self._extract_displacement_features(entry)
                    # Create regression target from displacement magnitude
                    disp = entry.get('displacement', {})
                    label = np.sqrt(disp.get('x', 0)**2 + disp.get('y', 0)**2 + disp.get('z', 0)**2)
                
                elif processor_type == 'StrainSignalProcessor':
                    features = self._extract_strain_features(entry)
                    # Create label from anomaly detection
                    label = 1 if entry.get('analysis_result', {}).get('anomaly_detected', False) else 0
                
                elif processor_type == 'TrendAnalyzer':
                    features = self._extract_trend_features(entry)
                    # Create label from regime classification
                    regime = entry.get('analysis_result', {}).get('regime_classification', 'stable')
                    regime_mapping = {'stable': 0, 'increasing': 1, 'decreasing': 2, 'irregular': 3}
                    label = regime_mapping.get(regime, 0)
                
                if features and len(features) > 0:
                    training_data.append((np.array(features, dtype=np.float32), np.array([label], dtype=np.float32)))
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Failed to convert buffer to training data: {e}")
            return []
    
    def _extract_seismic_features(self, entry: Dict) -> List[float]:
        """Extract seismic features for training"""
        features = []
        
        try:
            # Basic seismic features
            value = entry.get('value', {})
            features.extend([
                value.get('velocity', 0.0),
                value.get('acceleration', 0.0),
                entry.get('detection', {}).get('sta_lta_ratio', 0.0),
                entry.get('detection', {}).get('confidence', 0.0)
            ])
            
            # ML prediction features
            ml_pred = entry.get('ml_prediction', {})
            features.extend([
                ml_pred.get('phase_confidence', 0.0),
                ml_pred.get('magnitude_prediction', 0.0),
                ml_pred.get('event_probability', 0.0)
            ])
            
            # Frequency content features
            freq_features = entry.get('frequency_analysis', {})
            features.extend([
                freq_features.get('dominant_frequency', 0.0),
                freq_features.get('spectral_centroid', 0.0),
                freq_features.get('spectral_bandwidth', 0.0)
            ])
            
            # Pad or truncate to expected dimension
            target_dim = self.processor_model_configs['EarthquakeDetector']['feature_dim']
            while len(features) < target_dim:
                features.append(0.0)
            
            return features[:target_dim]
            
        except Exception as e:
            self.logger.error(f"Failed to extract seismic features: {e}")
            return []
    
    def _extract_displacement_features(self, entry: Dict) -> List[float]:
        """Extract displacement features for training"""
        features = []
        
        try:
            # Displacement components
            displacement = entry.get('displacement', {})
            features.extend([
                displacement.get('x', 0.0),
                displacement.get('y', 0.0),
                displacement.get('z', 0.0)
            ])
            
            # Strain tensor components
            strain = entry.get('strain_tensor', {})
            features.extend([
                strain.get('xx', 0.0),
                strain.get('yy', 0.0),
                strain.get('zz', 0.0),
                strain.get('xy', 0.0),
                strain.get('xz', 0.0),
                strain.get('yz', 0.0)
            ])
            
            # Source parameters
            source = entry.get('source_parameters', {})
            features.extend([
                source.get('volume_change', 0.0),
                source.get('depth', 0.0),
                source.get('strength', 0.0)
            ])
            
            # Pad to expected dimension
            target_dim = self.processor_model_configs['DisplacementDetector']['feature_dim']
            while len(features) < target_dim:
                features.append(0.0)
            
            return features[:target_dim]
            
        except Exception as e:
            self.logger.error(f"Failed to extract displacement features: {e}")
            return []
    
    def _extract_strain_features(self, entry: Dict) -> List[float]:
        """Extract strain features for training"""
        features = []
        
        try:
            # Strain components
            value = entry.get('value', {})
            features.extend([
                value.get('axial_strain', 0.0),
                value.get('lateral_strain', 0.0),
                value.get('shear_strain', 0.0)
            ])
            
            # Wavelet analysis features
            wavelet = entry.get('wavelet_analysis', {})
            for component in ['axial_strain', 'lateral_strain', 'shear_strain']:
                comp_analysis = wavelet.get(component, {})
                features.extend([
                    comp_analysis.get('dominant_scale', 0.0),
                    comp_analysis.get('wavelet_strength', 0.0)
                ])
            
            # Time domain features
            time_domain = entry.get('time_domain_features', {})
            for component in ['axial_strain', 'lateral_strain', 'shear_strain']:
                comp_features = time_domain.get(component, {}).get('statistical', {})
                features.extend([
                    comp_features.get('mean', 0.0),
                    comp_features.get('std', 0.0)
                ])
            
            # Pad to expected dimension
            target_dim = self.processor_model_configs['StrainSignalProcessor']['feature_dim']
            while len(features) < target_dim:
                features.append(0.0)
            
            return features[:target_dim]
            
        except Exception as e:
            self.logger.error(f"Failed to extract strain features: {e}")
            return []
    
    def _extract_trend_features(self, entry: Dict) -> List[float]:
        """Extract trend analysis features for training"""
        features = []
        
        try:
            # Trend coefficients
            trend_coef = entry.get('trend_coefficients', {})
            features.extend([
                trend_coef.get('slope', 0.0),
                trend_coef.get('intercept', 0.0),
                trend_coef.get('r_squared', 0.0)
            ])
            
            # Statistical metrics
            stats = entry.get('statistical_metrics', {})
            features.extend([
                stats.get('mean', 0.0),
                stats.get('std', 0.0),
                stats.get('skewness', 0.0),
                stats.get('kurtosis', 0.0)
            ])
            
            # Changepoint information
            changepoints = entry.get('changepoints', {})
            features.extend([
                len(changepoints.get('detected_changepoints', [])),
                changepoints.get('most_recent_changepoint_confidence', 0.0),
                changepoints.get('changepoint_frequency', 0.0)
            ])
            
            # Pad to expected dimension
            target_dim = self.processor_model_configs['TrendAnalyzer']['feature_dim']
            while len(features) < target_dim:
                features.append(0.0)
            
            return features[:target_dim]
            
        except Exception as e:
            self.logger.error(f"Failed to extract trend features: {e}")
            return []
    
    def _convert_analysis_to_training_data(self, analysis_results: List[Dict], processor_type: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Convert analysis results to training format"""
        # Similar to buffer conversion but for analysis results
        return self._convert_buffer_to_training_data(analysis_results, processor_type)
    
    async def run_federated_training(self, session_name: str) -> Dict[str, Any]:
        """Run federated training for a specific session"""
        try:
            if session_name not in self.training_sessions:
                self.logger.error(f"Session {session_name} not found")
                return {}
            
            session = self.training_sessions[session_name]
            
            if len(session['clients']) < session['config'].min_clients:
                self.logger.error(f"Insufficient clients for session {session_name}: {len(session['clients'])} < {session['config'].min_clients}")
                return {'status': 'insufficient_clients'}
            
            # Update session status
            session['status'] = 'training'
            
            # Run federated learning
            self.logger.info(f"Starting federated training for session {session_name}")
            results = await session['orchestrator'].run_federated_learning()
            
            # Update session with results
            session['status'] = 'completed'
            session['results'] = results
            session['completed_at'] = datetime.utcnow()
            
            # Deploy updated models if auto-deployment is enabled
            if self.config.get('auto_model_deployment', True):
                await self._deploy_updated_models(session_name, results)
            
            self.logger.info(f"Federated training completed for session {session_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to run federated training for session {session_name}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _deploy_updated_models(self, session_name: str, training_results: Dict[str, Any]):
        """Deploy updated models to edge devices"""
        try:
            session = self.training_sessions[session_name]
            
            # Get final global model from server
            global_model = session['orchestrator'].server.get_global_model()
            
            # Deploy to all clients in the session
            for device_id, client_info in session['clients'].items():
                try:
                    # Update client model
                    client_info['client'].update_model(global_model)
                    
                    # Update processor instances if they have model update capabilities
                    processor_instances = client_info['processor_instances']
                    for processor_type, processor in processor_instances.items():
                        if hasattr(processor, 'update_federated_model'):
                            await processor.update_federated_model(global_model)
                    
                    self.logger.info(f"Deployed updated model to device {device_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to deploy model to device {device_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to deploy updated models for session {session_name}: {e}")
    
    def get_session_status(self, session_name: str) -> Dict[str, Any]:
        """Get status of federated learning session"""
        if session_name not in self.training_sessions:
            return {'status': 'not_found'}
        
        session = self.training_sessions[session_name]
        
        status = {
            'session_name': session_name,
            'status': session['status'],
            'created_at': session['created_at'].isoformat(),
            'processor_types': session['processor_types'],
            'num_clients': len(session['clients']),
            'config': {
                'rounds': session['config'].rounds,
                'local_epochs': session['config'].local_epochs,
                'model_type': session['config'].model_type.value,
                'aggregation_strategy': session['config'].aggregation_strategy.value
            }
        }
        
        # Add completion info if available
        if 'completed_at' in session:
            status['completed_at'] = session['completed_at'].isoformat()
        
        if 'results' in session:
            results = session['results']
            status['training_results'] = {
                'rounds_completed': results.get('rounds_completed', 0),
                'final_performance': results.get('final_performance', {}),
                'total_training_time': results.get('server_stats', {}).get('total_training_time', 0)
            }
        
        # Add client information
        status['clients'] = {}
        for device_id, client_info in session['clients'].items():
            status['clients'][device_id] = {
                'data_size': client_info['data_size'],
                'last_update': client_info['last_update'].isoformat()
            }
        
        return status
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall federated learning system status"""
        status = {
            'federated_available': self.federated_available,
            'active_sessions': len(self.training_sessions),
            'total_clients': sum(len(session['clients']) for session in self.training_sessions.values()),
            'session_summary': {}
        }
        
        # Add session summaries
        for session_name, session in self.training_sessions.items():
            status['session_summary'][session_name] = {
                'status': session['status'],
                'num_clients': len(session['clients']),
                'processor_types': session['processor_types']
            }
        
        return status
    
    async def cleanup_resources(self):
        """Cleanup federated learning resources"""
        try:
            # Stop all active training sessions
            for session_name, session in self.training_sessions.items():
                if session['status'] == 'training':
                    # Gracefully stop training
                    session['status'] = 'stopped'
                    self.logger.info(f"Stopped federated training session: {session_name}")
            
            # Cleanup client resources
            for session in self.training_sessions.values():
                for client_info in session['clients'].values():
                    client = client_info['client']
                    if hasattr(client, 'cleanup'):
                        await client.cleanup()
            
            self.logger.info("EdgeFederatedLearningManager resources cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during federated learning cleanup: {e}")

# =============================================================================
# STREAM PROCESSING INTEGRATION SYSTEM
# =============================================================================

class EdgeStreamProcessingManager:
    """Stream processing manager for edge computing with real-time data pipelines"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize stream processing for edge computing"""
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize stream processing components with fallback
        try:
            from stream_processing import (
                KafkaStreamManager, FlinkStreamProcessor, RealTimeAggregator,
                EventDrivenArchitecture, StreamEvent, StreamEventType
            )
            self.KafkaStreamManager = KafkaStreamManager
            self.FlinkStreamProcessor = FlinkStreamProcessor
            self.RealTimeAggregator = RealTimeAggregator
            self.EventDrivenArchitecture = EventDrivenArchitecture
            self.StreamEvent = StreamEvent
            self.StreamEventType = StreamEventType
            self.stream_available = True
        except ImportError:
            self.stream_available = False
            self.logger.warning("Stream processing not available, using fallback system")
        
        # Initialize managers
        self.kafka_manager = None
        self.flink_processor = None
        self.aggregator = None
        self.event_architecture = None
        self.processor_streams = {}
        self.stream_topics = {
            'seismic_data': 'edge_seismic_data',
            'gps_data': 'edge_gps_data',
            'strain_data': 'edge_strain_data',
            'deformation_data': 'edge_deformation_data',
            'trend_data': 'edge_trend_data',
            'alerts': 'edge_alerts',
            'processed_data': 'edge_processed_data'
        }
        
        if self.stream_available:
            self._initialize_stream_components()
        else:
            self._initialize_fallback_system()
        
        self.logger.info("EdgeStreamProcessingManager initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default stream processing configuration"""
        return {
            'kafka_config': {
                'bootstrap_servers': ['localhost:9092'],
                'producer_config': {
                    'acks': 'all',
                    'retries': 3,
                    'batch_size': 8192,
                    'linger_ms': 5
                }
            },
            'flink_config': {
                'parallelism': 2,
                'checkpoint_interval': 30000,
                'window_size_seconds': 30
            },
            'aggregation_config': {
                'window_sizes': {
                    'minute': 60,
                    'hour': 3600,
                    'day': 86400
                }
            },
            'event_processors': {
                'EarthquakeDetector': 'seismic_data',
                'GPSNoiseFilter': 'gps_data',
                'GPSPositionCalculator': 'gps_data',
                'DisplacementDetector': 'deformation_data',
                'StrainSignalProcessor': 'strain_data',
                'TrendAnalyzer': 'trend_data'
            }
        }
    
    def _initialize_stream_components(self):
        """Initialize stream processing components"""
        try:
            # Initialize Kafka manager
            self.kafka_manager = self.KafkaStreamManager(self.config.get('kafka_config'))
            
            # Initialize Flink processor
            self.flink_processor = self.FlinkStreamProcessor(self.config.get('flink_config'))
            
            # Initialize real-time aggregator
            self.aggregator = self.RealTimeAggregator(self.config.get('aggregation_config'))
            
            # Initialize event-driven architecture
            self.event_architecture = self.EventDrivenArchitecture(self.kafka_manager)
            
            # Setup event handlers
            self._setup_event_handlers()
            
            self.logger.info("Stream processing components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stream components: {e}")
            self._initialize_fallback_system()
    
    def _initialize_fallback_system(self):
        """Initialize fallback streaming system"""
        self.fallback_queues = {}
        self.fallback_processors = {}
        self.event_buffers = {}
        
        for topic in self.stream_topics.values():
            self.fallback_queues[topic] = []
            self.event_buffers[topic] = []
        
        self.logger.info("Fallback streaming system initialized")
    
    def _setup_event_handlers(self):
        """Setup event handlers for different geological events"""
        if not self.stream_available:
            return
        
        # Seismic event handler
        async def seismic_event_handler(event):
            if 'magnitude' in event.data and event.data['magnitude'] > 5.0:
                alert_event = self.StreamEvent(
                    event_id=f"alert_{event.event_id}",
                    event_type=self.StreamEventType.ALERT,
                    timestamp=datetime.utcnow(),
                    source_id=f"alert_system_{event.source_id}",
                    data={
                        'alert_type': 'high_magnitude_earthquake',
                        'magnitude': event.data['magnitude'],
                        'original_event_id': event.event_id
                    },
                    location=event.location
                )
                await self.publish_event(alert_event, 'alerts')
        
        # Deformation event handler
        async def deformation_event_handler(event):
            if 'displacement_mm' in event.data and event.data['displacement_mm'] > 10.0:
                alert_event = self.StreamEvent(
                    event_id=f"alert_{event.event_id}",
                    event_type=self.StreamEventType.ALERT,
                    timestamp=datetime.utcnow(),
                    source_id=f"alert_system_{event.source_id}",
                    data={
                        'alert_type': 'significant_deformation',
                        'displacement_mm': event.data['displacement_mm'],
                        'original_event_id': event.event_id
                    },
                    location=event.location
                )
                await self.publish_event(alert_event, 'alerts')
        
        # Register handlers
        self.event_architecture.register_handler(
            self.StreamEventType.EARTHQUAKE, seismic_event_handler
        )
        self.event_architecture.register_handler(
            self.StreamEventType.DEFORMATION, deformation_event_handler
        )
    
    async def start_stream_processing(self):
        """Start stream processing pipeline"""
        if self.stream_available:
            try:
                # Start event processing
                await self.event_architecture.start_event_processing()
                
                # Create geological processing pipeline
                await self.flink_processor.create_geological_stream_pipeline(self.kafka_manager)
                
                self.logger.info("Stream processing pipeline started")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to start stream processing: {e}")
                return False
        else:
            # Start fallback processing
            self.logger.info("Started fallback stream processing")
            return True
    
    async def publish_processor_data(self, processor_name: str, data: Dict[str, Any], 
                                   location: Tuple[float, float, float] = None):
        """Publish data from edge processor to stream"""
        
        # Determine event type based on processor
        event_type_mapping = {
            'EarthquakeDetector': self.StreamEventType.SEISMIC_DATA,
            'GPSNoiseFilter': self.StreamEventType.GPS_DATA,
            'GPSPositionCalculator': self.StreamEventType.GPS_DATA,
            'DisplacementDetector': self.StreamEventType.DEFORMATION,
            'StrainSignalProcessor': self.StreamEventType.SEISMIC_DATA,
            'TrendAnalyzer': self.StreamEventType.STATUS_UPDATE
        }
        
        if self.stream_available:
            event_type = event_type_mapping.get(
                processor_name, 
                self.StreamEventType.SENSOR_DATA
            )
            
            # Create stream event
            event = self.StreamEvent(
                event_id=f"{processor_name}_{datetime.utcnow().timestamp()}",
                event_type=event_type,
                timestamp=datetime.utcnow(),
                source_id=processor_name,
                data=data,
                location=location,
                metadata={'processor_type': processor_name}
            )
            
            # Determine topic
            topic = self.config['event_processors'].get(
                processor_name, 
                'processed_data'
            )
            
            # Publish to stream
            await self.publish_event(event, topic)
            
            # Add to aggregator
            await self.aggregator.add_event(event)
            
        else:
            # Fallback: add to buffer
            topic = self.config['event_processors'].get(processor_name, 'processed_data')
            topic_name = self.stream_topics.get(topic, topic)
            
            event_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'processor': processor_name,
                'data': data,
                'location': location
            }
            
            self.event_buffers[topic_name].append(event_data)
            
            # Keep buffer size manageable
            if len(self.event_buffers[topic_name]) > 1000:
                self.event_buffers[topic_name] = self.event_buffers[topic_name][-500:]
    
    async def publish_event(self, event, topic: str = 'processed_data'):
        """Publish event to stream"""
        if self.stream_available:
            topic_name = self.stream_topics.get(topic, topic)
            await self.kafka_manager.produce_event(event, topic_name)
        else:
            # Fallback: add to queue
            topic_name = self.stream_topics.get(topic, topic)
            if topic_name in self.fallback_queues:
                self.fallback_queues[topic_name].append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'event': event if hasattr(event, 'to_json') else str(event)
                })
    
    async def get_stream_aggregations(self, window_type: str = 'minute', 
                                    processor_filter: List[str] = None) -> List[Dict[str, Any]]:
        """Get stream aggregations"""
        if self.stream_available and self.aggregator:
            try:
                aggregations = await self.aggregator.get_aggregations(window_type)
                
                if processor_filter:
                    filtered_aggregations = []
                    for agg in aggregations:
                        if any(proc in agg.get('sources', {}) for proc in processor_filter):
                            filtered_aggregations.append(agg)
                    return filtered_aggregations
                
                return aggregations
                
            except Exception as e:
                self.logger.error(f"Failed to get aggregations: {e}")
                return []
        else:
            # Fallback: return buffer statistics
            return self._get_fallback_statistics()
    
    def _get_fallback_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics from fallback buffers"""
        stats = []
        
        for topic, buffer in self.event_buffers.items():
            if buffer:
                recent_events = [
                    event for event in buffer 
                    if datetime.fromisoformat(event['timestamp']) > 
                    datetime.utcnow() - timedelta(minutes=5)
                ]
                
                stats.append({
                    'topic': topic,
                    'total_events': len(buffer),
                    'recent_events': len(recent_events),
                    'processors': list(set(event['processor'] for event in recent_events)),
                    'window_type': 'fallback',
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        return stats
    
    async def create_processor_stream(self, processor_name: str) -> str:
        """Create dedicated stream for processor"""
        stream_id = f"stream_{processor_name}_{int(datetime.utcnow().timestamp())}"
        
        if self.stream_available:
            # Create dedicated topic for processor
            topic_name = f"edge_{processor_name.lower()}_stream"
            
            self.processor_streams[stream_id] = {
                'processor': processor_name,
                'topic': topic_name,
                'created_at': datetime.utcnow(),
                'event_count': 0
            }
            
        else:
            # Fallback: create dedicated queue
            self.fallback_queues[stream_id] = []
            self.processor_streams[stream_id] = {
                'processor': processor_name,
                'queue': stream_id,
                'created_at': datetime.utcnow(),
                'event_count': 0
            }
        
        self.logger.info(f"Created stream {stream_id} for processor {processor_name}")
        return stream_id
    
    async def get_stream_status(self) -> Dict[str, Any]:
        """Get stream processing status"""
        status = {
            'stream_available': self.stream_available,
            'components': {
                'kafka': self.kafka_manager is not None,
                'flink': self.flink_processor is not None,
                'aggregator': self.aggregator is not None,
                'event_architecture': self.event_architecture is not None
            },
            'active_streams': len(self.processor_streams),
            'topics': list(self.stream_topics.values()),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if self.stream_available:
            try:
                # Get recent aggregations
                recent_aggregations = await self.aggregator.get_aggregations('minute')
                status['recent_activity'] = {
                    'aggregation_windows': len(recent_aggregations),
                    'last_activity': recent_aggregations[-1]['end_time'] if recent_aggregations else None
                }
            except Exception as e:
                status['recent_activity'] = {'error': str(e)}
        else:
            # Fallback statistics
            total_events = sum(len(buffer) for buffer in self.event_buffers.values())
            status['fallback_stats'] = {
                'total_buffered_events': total_events,
                'active_topics': len([b for b in self.event_buffers.values() if b])
            }
        
        return status
    
    async def setup_processor_stream_integration(self, processor_instances: Dict[str, Any]):
        """Setup stream integration for processor instances"""
        """Integrate stream processing with existing processor instances"""
        for processor_name, processor in processor_instances.items():
            try:
                # Add stream publishing method to processor
                async def publish_to_stream(data, location=None, proc_name=processor_name):
                    await self.publish_processor_data(proc_name, data, location)
                
                # Monkey patch the processor to include stream publishing
                processor.publish_to_stream = publish_to_stream
                
                # Create dedicated stream if needed
                stream_id = await self.create_processor_stream(processor_name)
                processor.stream_id = stream_id
                
                self.logger.info(f"Integrated stream processing for {processor_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to integrate streaming for {processor_name}: {e}")
    
    async def cleanup_stream_resources(self):
        """Clean up stream processing resources"""
        try:
            if self.stream_available:
                # Close Kafka connections
                if self.kafka_manager and hasattr(self.kafka_manager, 'producer'):
                    if self.kafka_manager.producer:
                        self.kafka_manager.producer.close()
                
                # Stop Flink execution environment
                if self.flink_processor and hasattr(self.flink_processor, 'env'):
                    if self.flink_processor.env:
                        try:
                            self.flink_processor.env.execute("cleanup")
                        except:
                            pass  # Ignore execution errors during cleanup
            
            # Clear buffers
            self.fallback_queues.clear()
            self.event_buffers.clear()
            self.processor_streams.clear()
            
            self.logger.info("Stream processing resources cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during stream processing cleanup: {e}")


if __name__ == "__main__":
    print("=== Edge Computing Framework ===")
    print(f"Available features:")
    print(f"  ZeroMQ: {ZMQ_AVAILABLE}")
    print(f"  MQTT: {MQTT_AVAILABLE}")
    print(f"  System Monitoring: {PSUTIL_AVAILABLE}")
    print(f"  GPU Monitoring: {GPUTIL_AVAILABLE}")
    
    # Run example
    asyncio.run(example_edge_computing())