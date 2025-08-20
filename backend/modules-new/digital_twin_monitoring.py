"""
Digital Twin Earth Real-time Monitoring System
Comprehensive monitoring, alerting, and health tracking for Digital Twin Earth infrastructure
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
import uuid
import psutil
import socket
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Monitoring and observability imports
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    warnings.warn("Prometheus client not available. Metrics collection will be limited.")

try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    warnings.warn("OpenTelemetry not available. Distributed tracing will be limited.")

# Alerting imports
try:
    import requests
    import slack_sdk
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False
    warnings.warn("Alerting libraries not available. Alert notifications will be limited.")

# Database imports for monitoring data
try:
    import influxdb_client
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    warnings.warn("InfluxDB client not available. Time-series monitoring will be limited.")

# Import Digital Twin Earth modules
try:
    from .digital_twin_earth import DigitalTwinEarth, EarthSystemState
    from .digital_twin_integration import DigitalTwinEarthIntegrator, IntegrationConfig
    TWIN_MODULES_AVAILABLE = True
except ImportError:
    TWIN_MODULES_AVAILABLE = False
    warnings.warn("Digital Twin modules not available.")

warnings.filterwarnings('ignore')

class MonitoringLevel(Enum):
    BASIC = "basic"
    DETAILED = "detailed"
    DEBUG = "debug"
    COMPREHENSIVE = "comprehensive"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"

class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"

class SystemComponent(Enum):
    DIGITAL_TWIN = "digital_twin"
    INTEGRATION = "integration"
    API = "api"
    VISUALIZATION = "visualization"
    DATABASE = "database"
    STREAM_PROCESSING = "stream_processing"
    ML_MODELS = "ml_models"

@dataclass
class MonitoringMetric:
    """Individual monitoring metric"""
    name: str
    value: Union[float, int, str]
    unit: str
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    component: SystemComponent = SystemComponent.DIGITAL_TWIN
    help_text: str = ""

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    component: SystemComponent
    metric_name: str
    condition: str  # e.g., "value > 0.8", "value < 0.1", "rate > 100"
    threshold: float
    severity: AlertSeverity
    evaluation_window_seconds: int = 300
    channels: List[AlertChannel] = field(default_factory=list)
    enabled: bool = True
    cooldown_seconds: int = 300

@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    name: str
    description: str
    severity: AlertSeverity
    component: SystemComponent
    triggered_at: datetime
    value: float
    threshold: float
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    monitoring_level: MonitoringLevel = MonitoringLevel.DETAILED
    collection_interval_seconds: int = 30
    metric_retention_days: int = 30
    enable_prometheus_metrics: bool = True
    enable_distributed_tracing: bool = True
    enable_health_checks: bool = True
    enable_performance_profiling: bool = True
    enable_alerting: bool = True
    alert_evaluation_interval_seconds: int = 60
    
    # InfluxDB configuration
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = "monitoring_token"
    influxdb_org: str = "prometheus_ai"
    influxdb_bucket: str = "digital_twin_metrics"
    
    # Alert channels configuration
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    slack_webhook_url: str = ""
    webhook_urls: List[str] = field(default_factory=list)

class DigitalTwinMonitoringSystem:
    """Comprehensive monitoring system for Digital Twin Earth"""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        
        # Monitoring components
        self.integrator = None
        self.metrics_registry = None
        self.tracer = None
        self.influxdb_client = None
        
        # Metrics storage
        self.current_metrics = {}
        self.metric_history = {}
        self.health_status = {}
        
        # Alerting system
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = []
        self.alert_cooldowns = {}
        
        # Performance tracking
        self.performance_stats = {}
        self.system_resources = {}
        
        # Monitoring tasks
        self.monitoring_tasks = {}
        self.is_monitoring_active = False
        
        # Initialize components
        asyncio.create_task(self._initialize_monitoring_components())
    
    async def _initialize_monitoring_components(self):
        """Initialize monitoring components"""
        try:
            # Setup Prometheus metrics
            if self.config.enable_prometheus_metrics and PROMETHEUS_AVAILABLE:
                await self._setup_prometheus_metrics()
            
            # Setup distributed tracing
            if self.config.enable_distributed_tracing and OPENTELEMETRY_AVAILABLE:
                await self._setup_distributed_tracing()
            
            # Setup InfluxDB connection
            if INFLUXDB_AVAILABLE:
                await self._setup_influxdb_connection()
            
            # Setup default alert rules
            await self._setup_default_alert_rules()
            
            self.logger.info("Monitoring system components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring components: {e}")
    
    async def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collection"""
        try:
            self.metrics_registry = CollectorRegistry()
            
            # System metrics
            self.cpu_usage_gauge = Gauge(
                'digital_twin_cpu_usage_percent',
                'CPU usage percentage',
                ['component'],
                registry=self.metrics_registry
            )
            
            self.memory_usage_gauge = Gauge(
                'digital_twin_memory_usage_bytes',
                'Memory usage in bytes',
                ['component'],
                registry=self.metrics_registry
            )
            
            self.request_count = Counter(
                'digital_twin_requests_total',
                'Total number of requests',
                ['component', 'method', 'status'],
                registry=self.metrics_registry
            )
            
            self.request_duration = Histogram(
                'digital_twin_request_duration_seconds',
                'Request duration in seconds',
                ['component', 'method'],
                registry=self.metrics_registry
            )
            
            # Digital Twin specific metrics
            self.earth_state_updates = Counter(
                'digital_twin_earth_state_updates_total',
                'Total Earth state updates',
                ['component'],
                registry=self.metrics_registry
            )
            
            self.prediction_accuracy = Gauge(
                'digital_twin_prediction_accuracy',
                'Prediction accuracy score',
                ['model_type'],
                registry=self.metrics_registry
            )
            
            self.uncertainty_level = Gauge(
                'digital_twin_uncertainty_level',
                'System uncertainty level',
                ['component'],
                registry=self.metrics_registry
            )
            
            self.integration_health = Gauge(
                'digital_twin_integration_health',
                'Integration health score 0-1',
                ['integration_type'],
                registry=self.metrics_registry
            )
            
            self.logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Prometheus metrics: {e}")
    
    async def _setup_distributed_tracing(self):
        """Setup distributed tracing with OpenTelemetry"""
        try:
            trace.set_tracer_provider(TracerProvider())
            
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.tracer = trace.get_tracer(__name__)
            
            # Instrument asyncio
            AsyncioInstrumentor().instrument()
            
            self.logger.info("Distributed tracing initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup distributed tracing: {e}")
    
    async def _setup_influxdb_connection(self):
        """Setup InfluxDB connection for time-series metrics"""
        try:
            self.influxdb_client = InfluxDBClient(
                url=self.config.influxdb_url,
                token=self.config.influxdb_token,
                org=self.config.influxdb_org
            )
            
            # Test connection
            health = self.influxdb_client.health()
            if health.status == "pass":
                self.logger.info("InfluxDB connection established")
            else:
                self.logger.warning("InfluxDB health check failed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup InfluxDB connection: {e}")
            self.influxdb_client = None
    
    async def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU usage above 80%",
                component=SystemComponent.DIGITAL_TWIN,
                metric_name="cpu_usage_percent",
                condition="value > 80",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD]
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="Memory usage above 85%",
                component=SystemComponent.DIGITAL_TWIN,
                metric_name="memory_usage_percent",
                condition="value > 85",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD]
            ),
            AlertRule(
                rule_id="integration_health_low",
                name="Integration Health Low",
                description="Integration health score below 0.5",
                component=SystemComponent.INTEGRATION,
                metric_name="integration_health_score",
                condition="value < 0.5",
                threshold=0.5,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD]
            ),
            AlertRule(
                rule_id="high_uncertainty",
                name="High System Uncertainty",
                description="System uncertainty above 0.8",
                component=SystemComponent.DIGITAL_TWIN,
                metric_name="uncertainty_level",
                condition="value > 0.8",
                threshold=0.8,
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD]
            ),
            AlertRule(
                rule_id="prediction_accuracy_low",
                name="Low Prediction Accuracy",
                description="Prediction accuracy below 0.7",
                component=SystemComponent.ML_MODELS,
                metric_name="prediction_accuracy",
                condition="value < 0.7",
                threshold=0.7,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.DASHBOARD]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    async def initialize(self, integrator: DigitalTwinEarthIntegrator = None):
        """Initialize monitoring system with Digital Twin components"""
        try:
            self.integrator = integrator
            
            # Start monitoring tasks
            if self.config.enable_health_checks:
                self.monitoring_tasks['health_checks'] = asyncio.create_task(self._health_check_loop())
            
            self.monitoring_tasks['metrics_collection'] = asyncio.create_task(self._metrics_collection_loop())
            
            if self.config.enable_alerting:
                self.monitoring_tasks['alert_evaluation'] = asyncio.create_task(self._alert_evaluation_loop())
            
            if self.config.enable_performance_profiling:
                self.monitoring_tasks['performance_monitoring'] = asyncio.create_task(self._performance_monitoring_loop())
            
            self.is_monitoring_active = True
            
            self.logger.info("Digital Twin monitoring system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring system: {e}")
            raise
    
    async def _health_check_loop(self):
        """Continuous health checking loop"""
        while self.is_monitoring_active:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(60)  # Health checks every minute
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_collection_loop(self):
        """Continuous metrics collection loop"""
        while self.is_monitoring_active:
            try:
                await self._collect_system_metrics()
                await self._collect_digital_twin_metrics()
                await self._store_metrics_in_influxdb()
                await asyncio.sleep(self.config.collection_interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)
    
    async def _alert_evaluation_loop(self):
        """Continuous alert evaluation loop"""
        while self.is_monitoring_active:
            try:
                await self._evaluate_alert_rules()
                await asyncio.sleep(self.config.alert_evaluation_interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(10)
    
    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring loop"""
        while self.is_monitoring_active:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(30)  # Performance metrics every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _perform_health_checks(self):
        """Perform comprehensive health checks"""
        try:
            health_results = {}
            
            # Check Digital Twin Earth health
            if self.integrator and self.integrator.digital_twin:
                try:
                    current_state = self.integrator.digital_twin.get_current_state()
                    health_results['digital_twin'] = {
                        'status': 'healthy' if current_state else 'unhealthy',
                        'last_update': current_state.timestamp.isoformat() if current_state else None,
                        'component_count': 6  # Number of Earth system components
                    }
                except Exception as e:
                    health_results['digital_twin'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            else:
                health_results['digital_twin'] = {'status': 'unavailable'}
            
            # Check integration system health
            if self.integrator:
                try:
                    integration_status = await self.integrator.get_integration_status()
                    health_results['integration'] = {
                        'status': integration_status['metrics']['integration_health'].lower(),
                        'active_integrations': len(integration_status['active_integrations']),
                        'sync_latency_ms': integration_status['metrics']['sync_latency_ms']
                    }
                except Exception as e:
                    health_results['integration'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            else:
                health_results['integration'] = {'status': 'unavailable'}
            
            # Check system resources
            health_results['system'] = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
            
            # Check network connectivity
            health_results['network'] = await self._check_network_connectivity()
            
            # Check database connections
            health_results['database'] = await self._check_database_connectivity()
            
            # Update health status
            self.health_status = {
                'timestamp': datetime.utcnow(),
                'overall_status': self._calculate_overall_health(health_results),
                'components': health_results
            }
            
            # Record health metrics
            await self._record_health_metrics(health_results)
            
        except Exception as e:
            self.logger.error(f"Error performing health checks: {e}")
    
    async def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            # Test external connectivity
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('8.8.8.8', 53))  # Google DNS
            sock.close()
            
            return {
                'status': 'healthy' if result == 0 else 'degraded',
                'external_connectivity': result == 0
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            db_status = {}
            
            # Check InfluxDB
            if self.influxdb_client:
                try:
                    health = self.influxdb_client.health()
                    db_status['influxdb'] = {
                        'status': 'healthy' if health.status == "pass" else 'unhealthy'
                    }
                except Exception as e:
                    db_status['influxdb'] = {'status': 'error', 'error': str(e)}
            else:
                db_status['influxdb'] = {'status': 'unavailable'}
            
            return db_status
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_overall_health(self, health_results: Dict[str, Any]) -> str:
        """Calculate overall system health from component health"""
        try:
            healthy_count = 0
            total_count = 0
            
            for component, status in health_results.items():
                total_count += 1
                if isinstance(status, dict) and status.get('status') == 'healthy':
                    healthy_count += 1
            
            health_ratio = healthy_count / total_count if total_count > 0 else 0
            
            if health_ratio >= 0.9:
                return 'healthy'
            elif health_ratio >= 0.7:
                return 'degraded'
            elif health_ratio >= 0.5:
                return 'unhealthy'
            else:
                return 'critical'
        
        except Exception as e:
            self.logger.error(f"Error calculating overall health: {e}")
            return 'unknown'
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            timestamp = datetime.utcnow()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.current_metrics['cpu_usage_percent'] = MonitoringMetric(
                name='cpu_usage_percent',
                value=cpu_percent,
                unit='percent',
                timestamp=timestamp,
                component=SystemComponent.DIGITAL_TWIN
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.current_metrics['memory_usage_percent'] = MonitoringMetric(
                name='memory_usage_percent',
                value=memory.percent,
                unit='percent',
                timestamp=timestamp,
                component=SystemComponent.DIGITAL_TWIN
            )
            
            self.current_metrics['memory_usage_bytes'] = MonitoringMetric(
                name='memory_usage_bytes',
                value=memory.used,
                unit='bytes',
                timestamp=timestamp,
                component=SystemComponent.DIGITAL_TWIN
            )
            
            # Disk metrics
            if hasattr(psutil, 'disk_usage'):
                disk = psutil.disk_usage('/')
                self.current_metrics['disk_usage_percent'] = MonitoringMetric(
                    name='disk_usage_percent',
                    value=(disk.used / disk.total) * 100,
                    unit='percent',
                    timestamp=timestamp,
                    component=SystemComponent.DIGITAL_TWIN
                )
            
            # Network metrics
            if hasattr(psutil, 'net_io_counters'):
                net_io = psutil.net_io_counters()
                self.current_metrics['network_bytes_sent'] = MonitoringMetric(
                    name='network_bytes_sent',
                    value=net_io.bytes_sent,
                    unit='bytes',
                    timestamp=timestamp,
                    component=SystemComponent.DIGITAL_TWIN
                )
                
                self.current_metrics['network_bytes_recv'] = MonitoringMetric(
                    name='network_bytes_recv',
                    value=net_io.bytes_recv,
                    unit='bytes',
                    timestamp=timestamp,
                    component=SystemComponent.DIGITAL_TWIN
                )
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and self.metrics_registry:
                self.cpu_usage_gauge.labels(component='system').set(cpu_percent)
                self.memory_usage_gauge.labels(component='system').set(memory.used)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_digital_twin_metrics(self):
        """Collect Digital Twin specific metrics"""
        try:
            if not self.integrator:
                return
            
            timestamp = datetime.utcnow()
            
            # Integration status metrics
            try:
                integration_status = await self.integrator.get_integration_status()
                
                self.current_metrics['integration_health_score'] = MonitoringMetric(
                    name='integration_health_score',
                    value=self._health_to_score(integration_status['metrics']['integration_health']),
                    unit='score',
                    timestamp=timestamp,
                    component=SystemComponent.INTEGRATION
                )
                
                self.current_metrics['sync_latency_ms'] = MonitoringMetric(
                    name='sync_latency_ms',
                    value=integration_status['metrics']['sync_latency_ms'],
                    unit='milliseconds',
                    timestamp=timestamp,
                    component=SystemComponent.INTEGRATION
                )
                
                self.current_metrics['data_throughput_mbps'] = MonitoringMetric(
                    name='data_throughput_mbps',
                    value=integration_status['metrics']['data_throughput_mbps'],
                    unit='mbps',
                    timestamp=timestamp,
                    component=SystemComponent.INTEGRATION
                )
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE and self.metrics_registry:
                    health_score = self._health_to_score(integration_status['metrics']['integration_health'])
                    self.integration_health.labels(integration_type='main').set(health_score)
                
            except Exception as e:
                self.logger.error(f"Error collecting integration metrics: {e}")
            
            # Digital Twin Earth metrics
            try:
                if self.integrator.digital_twin:
                    current_state = self.integrator.digital_twin.get_current_state()
                    if current_state:
                        # Calculate uncertainty level
                        uncertainty = current_state.uncertainty_metrics.get('total_uncertainty', 0.0)
                        self.current_metrics['uncertainty_level'] = MonitoringMetric(
                            name='uncertainty_level',
                            value=uncertainty,
                            unit='score',
                            timestamp=timestamp,
                            component=SystemComponent.DIGITAL_TWIN
                        )
                        
                        # Component state metrics
                        for component in ['atmosphere', 'hydrosphere', 'lithosphere', 'biosphere', 'cryosphere', 'anthroposphere']:
                            state_data = getattr(current_state, f'{component}_state', {})
                            if state_data and isinstance(state_data, dict):
                                # Calculate component health score
                                component_score = self._calculate_component_health_score(state_data)
                                self.current_metrics[f'{component}_health'] = MonitoringMetric(
                                    name=f'{component}_health',
                                    value=component_score,
                                    unit='score',
                                    timestamp=timestamp,
                                    component=SystemComponent.DIGITAL_TWIN,
                                    labels={'earth_component': component}
                                )
                        
                        # Update Prometheus metrics
                        if PROMETHEUS_AVAILABLE and self.metrics_registry:
                            self.uncertainty_level.labels(component='digital_twin').set(uncertainty)
            
            except Exception as e:
                self.logger.error(f"Error collecting Digital Twin metrics: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in Digital Twin metrics collection: {e}")
    
    def _health_to_score(self, health_status: str) -> float:
        """Convert health status to numeric score"""
        health_scores = {
            'healthy': 1.0,
            'degraded': 0.7,
            'unhealthy': 0.4,
            'critical': 0.2,
            'unknown': 0.0
        }
        return health_scores.get(health_status.lower(), 0.0)
    
    def _calculate_component_health_score(self, component_data: Dict[str, Any]) -> float:
        """Calculate health score for Earth system component"""
        try:
            # Simple health scoring based on data completeness and validity
            if not component_data:
                return 0.0
            
            valid_values = 0
            total_values = 0
            
            for key, value in component_data.items():
                total_values += 1
                if value is not None and not np.isnan(float(value)) if isinstance(value, (int, float)) else True:
                    valid_values += 1
            
            return valid_values / total_values if total_values > 0 else 0.0
        
        except Exception as e:
            self.logger.error(f"Error calculating component health score: {e}")
            return 0.0
    
    async def _store_metrics_in_influxdb(self):
        """Store collected metrics in InfluxDB"""
        try:
            if not self.influxdb_client:
                return
            
            write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)
            
            points = []
            for metric_name, metric in self.current_metrics.items():
                point = Point("digital_twin_metrics") \
                    .tag("component", metric.component.value) \
                    .tag("metric_name", metric.name) \
                    .field("value", float(metric.value)) \
                    .time(metric.timestamp, WritePrecision.NS)
                
                # Add labels as tags
                for label_key, label_value in metric.labels.items():
                    point = point.tag(label_key, label_value)
                
                points.append(point)
            
            if points:
                write_api.write(bucket=self.config.influxdb_bucket, record=points)
                
        except Exception as e:
            self.logger.error(f"Error storing metrics in InfluxDB: {e}")
    
    async def _collect_performance_metrics(self):
        """Collect performance-related metrics"""
        try:
            timestamp = datetime.utcnow()
            
            # Process-specific metrics
            process = psutil.Process()
            
            self.performance_stats['process_cpu_percent'] = process.cpu_percent(interval=1)
            self.performance_stats['process_memory_mb'] = process.memory_info().rss / (1024 * 1024)
            self.performance_stats['process_threads'] = process.num_threads()
            
            # Python-specific metrics
            import gc
            self.performance_stats['gc_objects'] = len(gc.get_objects())
            self.performance_stats['gc_collections'] = gc.get_count()
            
            # Store as monitoring metrics
            for perf_name, perf_value in self.performance_stats.items():
                self.current_metrics[perf_name] = MonitoringMetric(
                    name=perf_name,
                    value=perf_value if isinstance(perf_value, (int, float)) else 0,
                    unit='count',
                    timestamp=timestamp,
                    component=SystemComponent.DIGITAL_TWIN,
                    labels={'performance_metric': perf_name}
                )
        
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
    
    async def _evaluate_alert_rules(self):
        """Evaluate all alert rules against current metrics"""
        try:
            current_time = datetime.utcnow()
            
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if rule_id in self.alert_cooldowns:
                    if current_time < self.alert_cooldowns[rule_id]:
                        continue
                
                # Get current metric value
                if rule.metric_name not in self.current_metrics:
                    continue
                
                metric = self.current_metrics[rule.metric_name]
                
                # Evaluate condition
                if await self._evaluate_condition(metric.value, rule):
                    # Create or update alert
                    alert_id = f"{rule_id}_{int(current_time.timestamp())}"
                    
                    alert = Alert(
                        alert_id=alert_id,
                        rule_id=rule_id,
                        name=rule.name,
                        description=rule.description,
                        severity=rule.severity,
                        component=rule.component,
                        triggered_at=current_time,
                        value=float(metric.value),
                        threshold=rule.threshold,
                        metadata={
                            'metric_name': rule.metric_name,
                            'condition': rule.condition,
                            'evaluation_window': rule.evaluation_window_seconds
                        }
                    )
                    
                    # Store active alert
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    
                    # Set cooldown
                    self.alert_cooldowns[rule_id] = current_time + timedelta(seconds=rule.cooldown_seconds)
                    
                    # Send alert notifications
                    await self._send_alert_notifications(alert, rule)
                    
                    self.logger.warning(f"Alert triggered: {alert.name} - {alert.description}")
        
        except Exception as e:
            self.logger.error(f"Error evaluating alert rules: {e}")
    
    async def _evaluate_condition(self, value: Union[float, int], rule: AlertRule) -> bool:
        """Evaluate alert condition"""
        try:
            # Simple condition evaluation
            if ">" in rule.condition:
                return float(value) > rule.threshold
            elif "<" in rule.condition:
                return float(value) < rule.threshold
            elif "==" in rule.condition:
                return float(value) == rule.threshold
            elif "!=" in rule.condition:
                return float(value) != rule.threshold
            else:
                return False
        
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return False
    
    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send alert notifications to configured channels"""
        try:
            for channel in rule.channels:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert)
                elif channel == AlertChannel.DASHBOARD:
                    await self._send_dashboard_alert(alert)
        
        except Exception as e:
            self.logger.error(f"Error sending alert notifications: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert notification"""
        try:
            if not self.config.email_username or not self.config.email_password:
                return
            
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = self.config.email_username  # Send to same email for demo
            msg['Subject'] = f"Digital Twin Alert: {alert.name} [{alert.severity.value.upper()}]"
            
            body = f"""
Digital Twin Earth Alert

Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Component: {alert.component.value}
Description: {alert.description}

Current Value: {alert.value}
Threshold: {alert.threshold}
Triggered At: {alert.triggered_at.isoformat()}

Please investigate this alert immediately.

Digital Twin Monitoring System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert notification"""
        try:
            if not self.config.slack_webhook_url:
                return
            
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.FATAL: "danger"
            }
            
            payload = {
                "text": f"Digital Twin Alert: {alert.name}",
                "attachments": [{
                    "color": color_map.get(alert.severity, "danger"),
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Component", "value": alert.component.value, "short": True},
                        {"title": "Current Value", "value": str(alert.value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                        {"title": "Description", "value": alert.description, "short": False}
                    ]
                }]
            }
            
            if ALERTING_AVAILABLE:
                response = requests.post(self.config.slack_webhook_url, json=payload)
                response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert notification"""
        try:
            if not self.config.webhook_urls:
                return
            
            payload = {
                "alert_id": alert.alert_id,
                "name": alert.name,
                "severity": alert.severity.value,
                "component": alert.component.value,
                "description": alert.description,
                "value": alert.value,
                "threshold": alert.threshold,
                "triggered_at": alert.triggered_at.isoformat(),
                "metadata": alert.metadata
            }
            
            if ALERTING_AVAILABLE:
                for webhook_url in self.config.webhook_urls:
                    response = requests.post(webhook_url, json=payload)
                    response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}")
    
    async def _send_dashboard_alert(self, alert: Alert):
        """Send dashboard alert notification"""
        try:
            # Dashboard alerts would be handled by the visualization system
            # For now, just log the alert
            self.logger.info(f"Dashboard alert: {alert.name} - {alert.description}")
        except Exception as e:
            self.logger.error(f"Error sending dashboard alert: {e}")
    
    async def _record_health_metrics(self, health_results: Dict[str, Any]):
        """Record health check results as metrics"""
        try:
            timestamp = datetime.utcnow()
            
            for component, health_data in health_results.items():
                if isinstance(health_data, dict):
                    status = health_data.get('status', 'unknown')
                    health_score = self._health_to_score(status)
                    
                    metric_name = f'{component}_health'
                    self.current_metrics[metric_name] = MonitoringMetric(
                        name=metric_name,
                        value=health_score,
                        unit='score',
                        timestamp=timestamp,
                        component=SystemComponent.DIGITAL_TWIN,
                        labels={'health_component': component}
                    )
        
        except Exception as e:
            self.logger.error(f"Error recording health metrics: {e}")
    
    async def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data for dashboard"""
        try:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'health_status': self.health_status,
                'current_metrics': {
                    name: {
                        'value': metric.value,
                        'unit': metric.unit,
                        'component': metric.component.value,
                        'timestamp': metric.timestamp.isoformat()
                    }
                    for name, metric in self.current_metrics.items()
                },
                'active_alerts': {
                    alert_id: {
                        'name': alert.name,
                        'severity': alert.severity.value,
                        'component': alert.component.value,
                        'triggered_at': alert.triggered_at.isoformat(),
                        'value': alert.value,
                        'threshold': alert.threshold
                    }
                    for alert_id, alert in self.active_alerts.items()
                },
                'performance_stats': self.performance_stats,
                'system_resources': {
                    'cpu_percent': psutil.cpu_percent(interval=None),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error getting monitoring dashboard data: {e}")
            return {'error': str(e)}
    
    async def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule"""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    async def remove_alert_rule(self, rule_id: str):
        """Remove alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")
    
    async def resolve_alert(self, alert_id: str):
        """Manually resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.utcnow()
            del self.active_alerts[alert_id]
            self.logger.info(f"Resolved alert: {alert.name}")
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        try:
            if self.metrics_registry:
                return generate_latest(self.metrics_registry).decode('utf-8')
            else:
                return "# Prometheus metrics not available\n"
        except Exception as e:
            self.logger.error(f"Error getting Prometheus metrics: {e}")
            return f"# Error: {e}\n"
    
    async def shutdown(self):
        """Shutdown the monitoring system"""
        try:
            self.is_monitoring_active = False
            
            # Cancel monitoring tasks
            for task_name, task in self.monitoring_tasks.items():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Close connections
            if self.influxdb_client:
                self.influxdb_client.close()
            
            # Clear data
            self.current_metrics.clear()
            self.active_alerts.clear()
            
            self.logger.info("Digital Twin monitoring system shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error during monitoring system shutdown: {e}")

# Example usage and testing
async def example_digital_twin_monitoring():
    """Example of Digital Twin monitoring system"""
    
    print("=== Digital Twin Earth Monitoring System Example ===")
    
    # Configuration
    monitoring_config = MonitoringConfig(
        monitoring_level=MonitoringLevel.DETAILED,
        collection_interval_seconds=10,
        enable_prometheus_metrics=True,
        enable_alerting=True,
        alert_evaluation_interval_seconds=30
    )
    
    # Initialize monitoring system
    monitoring_system = DigitalTwinMonitoringSystem(monitoring_config)
    
    try:
        # Initialize without integrator for demo
        await monitoring_system.initialize()
        
        print("Monitoring system initialized successfully")
        print(f"Active monitoring level: {monitoring_config.monitoring_level.value}")
        print(f"Collection interval: {monitoring_config.collection_interval_seconds}s")
        print(f"Alert rules configured: {len(monitoring_system.alert_rules)}")
        
        # Wait for some metrics collection
        await asyncio.sleep(15)
        
        # Get monitoring dashboard data
        dashboard_data = await monitoring_system.get_monitoring_dashboard_data()
        print(f"Current metrics collected: {len(dashboard_data['current_metrics'])}")
        print(f"Active alerts: {len(dashboard_data['active_alerts'])}")
        print(f"Overall health status: {dashboard_data['health_status']['overall_status']}")
        
        # Show some metrics
        if dashboard_data['current_metrics']:
            print("Sample metrics:")
            for metric_name, metric_data in list(dashboard_data['current_metrics'].items())[:3]:
                print(f"  {metric_name}: {metric_data['value']} {metric_data['unit']}")
        
        # Get Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            prom_metrics = monitoring_system.get_prometheus_metrics()
            print(f"Prometheus metrics format available: {len(prom_metrics)} characters")
        
        print("Monitoring system running successfully!")
        
    finally:
        await monitoring_system.shutdown()
    
    print("Digital Twin monitoring example completed!")

if __name__ == "__main__":
    print("=== Digital Twin Earth Monitoring System ===")
    print(f"Available features:")
    print(f"  Prometheus Client: {PROMETHEUS_AVAILABLE}")
    print(f"  OpenTelemetry: {OPENTELEMETRY_AVAILABLE}")
    print(f"  InfluxDB: {INFLUXDB_AVAILABLE}")
    print(f"  Alerting: {ALERTING_AVAILABLE}")
    print(f"  Twin Modules: {TWIN_MODULES_AVAILABLE}")
    
    # Run example
    asyncio.run(example_digital_twin_monitoring())