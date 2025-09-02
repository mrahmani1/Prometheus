"""
Comprehensive Logging and Monitoring System for Prometheus AI Geodynamic Platform
Advanced logging, metrics collection, performance tracking, and observability framework
"""

import os
import sys
import json
import time
import logging
import logging.config
import traceback
import threading
import asyncio
import uuid
import socket
import psutil
import threading
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import signal
import platform

# Advanced monitoring imports
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    warnings.warn("Prometheus client not available")

try:
    from opentelemetry import trace, metrics, baggage
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    warnings.warn("OpenTelemetry not available")

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    warnings.warn("Structlog not available - using standard logging")

try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.asyncio import AsyncioIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    warnings.warn("Sentry SDK not available")

try:
    import influxdb_client
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    warnings.warn("InfluxDB client not available")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available")

try:
    import elasticsearch
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    warnings.warn("Elasticsearch not available")

try:
    import requests
    import slack_sdk
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False
    warnings.warn("Alerting libraries not available")

warnings.filterwarnings('ignore')

class LogLevel(Enum):
    """Enhanced log levels"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    FATAL = 60

class MetricType(Enum):
    """Metric types for monitoring"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"

class MonitoringComponent(Enum):
    """System components for monitoring"""
    GEOLOGICAL_MODELS = "geological_models"
    ML_INFERENCE = "ml_inference"
    DATA_PIPELINE = "data_pipeline"
    API_GATEWAY = "api_gateway"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    WORKER = "worker"
    SCHEDULER = "scheduler"
    EXTERNAL_API = "external_api"

@dataclass
class LoggingConfig:
    """Comprehensive logging configuration"""
    log_level: str = "INFO"
    log_format: str = "json"  # json, text, structured
    log_file_path: Optional[str] = None
    log_rotation_size: str = "100MB"
    log_rotation_backup_count: int = 10
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    enable_structured_logging: bool = True
    enable_distributed_tracing: bool = True
    enable_sentry_integration: bool = False
    sentry_dsn: Optional[str] = None
    correlation_id_header: str = "X-Correlation-ID"
    
    # Performance logging
    enable_performance_logging: bool = True
    performance_threshold_ms: float = 1000.0
    
    # Security logging
    enable_security_logging: bool = True
    log_sensitive_data: bool = False
    
    # Async logging
    enable_async_logging: bool = True
    async_queue_size: int = 10000

@dataclass
class MonitoringConfig:
    """Comprehensive monitoring configuration"""
    enable_metrics: bool = True
    enable_prometheus: bool = True
    prometheus_port: int = 8000
    metrics_prefix: str = "prometheus_ai"
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_collection_interval: float = 5.0
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    
    # Application metrics
    enable_custom_metrics: bool = True
    enable_business_metrics: bool = True
    
    # Time series storage
    enable_influxdb: bool = False
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: Optional[str] = None
    influxdb_org: str = "prometheus-ai"
    influxdb_bucket: str = "metrics"
    
    # Distributed tracing
    enable_jaeger: bool = False
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    
    # Health checks
    enable_health_checks: bool = True
    health_check_interval: float = 30.0
    
    # Alert configuration
    enable_alerting: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["email", "webhook"])
    webhook_url: Optional[str] = None
    email_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    thread_count: int
    open_files: int
    response_times: Dict[str, float]
    throughput: Dict[str, float]
    error_rates: Dict[str, float]
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

class CorrelationContext:
    """Thread-local correlation context for distributed tracing"""
    
    def __init__(self):
        self._local = threading.local()
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current context"""
        self._local.correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get correlation ID from current context"""
        return getattr(self._local, 'correlation_id', None)
    
    def clear(self):
        """Clear correlation context"""
        if hasattr(self._local, 'correlation_id'):
            delattr(self._local, 'correlation_id')

class StructuredLogger:
    """Advanced structured logging implementation"""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.correlation_context = CorrelationContext()
        self._setup_logging()
        self._setup_structured_logging()
        self._setup_sentry()
        self._setup_async_logging()
    
    def _setup_logging(self):
        """Setup standard logging configuration"""
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'json': {
                    'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                    'format': '%(asctime)s %(name)s %(levelname)s %(correlation_id)s %(message)s'
                },
                'detailed': {
                    'format': '[%(asctime)s] [%(name)s] [%(levelname)s] [%(correlation_id)s] %(message)s'
                },
                'simple': {
                    'format': '%(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': self.config.log_level,
                    'formatter': 'json' if self.config.log_format == 'json' else 'detailed',
                    'stream': 'ext://sys.stdout'
                }
            },
            'root': {
                'level': self.config.log_level,
                'handlers': ['console']
            }
        }
        
        # Add file handler if enabled
        if self.config.enable_file_logging and self.config.log_file_path:
            logging_config['handlers']['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': self.config.log_level,
                'formatter': 'json' if self.config.log_format == 'json' else 'detailed',
                'filename': self.config.log_file_path,
                'maxBytes': self._parse_size(self.config.log_rotation_size),
                'backupCount': self.config.log_rotation_backup_count
            }
            logging_config['root']['handlers'].append('file')
        
        logging.config.dictConfig(logging_config)
        self.logger = logging.getLogger(__name__)
    
    def _setup_structured_logging(self):
        """Setup structured logging with structlog if available"""
        if STRUCTLOG_AVAILABLE and self.config.enable_structured_logging:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
    
    def _setup_sentry(self):
        """Setup Sentry error tracking if enabled"""
        if SENTRY_AVAILABLE and self.config.enable_sentry_integration and self.config.sentry_dsn:
            sentry_logging = LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR
            )
            
            sentry_sdk.init(
                dsn=self.config.sentry_dsn,
                integrations=[sentry_logging, AsyncioIntegration()],
                traces_sample_rate=1.0,
                attach_stacktrace=True,
                environment=os.getenv('ENVIRONMENT', 'development')
            )
    
    def _setup_async_logging(self):
        """Setup async logging queue if enabled"""
        if self.config.enable_async_logging:
            self.log_queue = queue.Queue(maxsize=self.config.async_queue_size)
            self.async_handler_thread = threading.Thread(
                target=self._async_log_handler,
                daemon=True
            )
            self.async_handler_thread.start()
    
    def _async_log_handler(self):
        """Async log handler thread"""
        while True:
            try:
                log_record = self.log_queue.get(timeout=1.0)
                if log_record is None:
                    break
                
                # Process log record
                self.logger.handle(log_record)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Async logging error: {e}")
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes"""
        size_str = size_str.upper()
        if size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    @contextmanager
    def correlation_context(self, correlation_id: str = None):
        """Context manager for correlation tracking"""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        
        self.correlation_context.set_correlation_id(correlation_id)
        try:
            yield correlation_id
        finally:
            self.correlation_context.clear()
    
    def log_with_context(self, level: str, message: str, **kwargs):
        """Log with correlation context"""
        correlation_id = self.correlation_context.get_correlation_id()
        extra = {
            'correlation_id': correlation_id or 'unknown',
            **kwargs
        }
        
        getattr(self.logger, level.lower())(message, extra=extra)

class MetricsCollector:
    """Advanced metrics collection system"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.custom_metrics = {}
        self.performance_history = deque(maxlen=10000)
        
        # Initialize metric collectors
        self._setup_prometheus_metrics()
        self._setup_influxdb()
        self._setup_custom_metrics()
        
        # Start collection threads
        self._start_collection_threads()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics if available"""
        if not PROMETHEUS_AVAILABLE or not self.config.enable_prometheus:
            return
        
        prefix = self.config.metrics_prefix
        
        # System metrics
        self.cpu_usage = Gauge(
            f'{prefix}_cpu_usage_percent',
            'CPU usage percentage',
            ['component', 'instance'],
            registry=self.metrics_registry
        )
        
        self.memory_usage = Gauge(
            f'{prefix}_memory_usage_bytes',
            'Memory usage in bytes',
            ['type', 'instance'],
            registry=self.metrics_registry
        )
        
        self.disk_usage = Gauge(
            f'{prefix}_disk_usage_percent',
            'Disk usage percentage',
            ['mount_point', 'instance'],
            registry=self.metrics_registry
        )
        
        # Application metrics
        self.request_count = Counter(
            f'{prefix}_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.metrics_registry
        )
        
        self.request_duration = Histogram(
            f'{prefix}_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.metrics_registry
        )
        
        self.active_connections = Gauge(
            f'{prefix}_active_connections',
            'Number of active connections',
            ['type'],
            registry=self.metrics_registry
        )
        
        # Geological computation metrics
        self.geological_computation_time = Histogram(
            f'{prefix}_geological_computation_seconds',
            'Time spent on geological computations',
            ['computation_type', 'model'],
            registry=self.metrics_registry
        )
        
        self.ml_inference_time = Histogram(
            f'{prefix}_ml_inference_seconds',
            'ML inference time',
            ['model_type', 'model_version'],
            registry=self.metrics_registry
        )
        
        self.data_processing_rate = Gauge(
            f'{prefix}_data_processing_rate_per_second',
            'Data processing rate per second',
            ['data_type', 'pipeline'],
            registry=self.metrics_registry
        )
        
        # Error metrics
        self.error_count = Counter(
            f'{prefix}_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.metrics_registry
        )
    
    def _setup_influxdb(self):
        """Setup InfluxDB connection if enabled"""
        if not INFLUXDB_AVAILABLE or not self.config.enable_influxdb:
            self.influxdb_client = None
            return
        
        try:
            self.influxdb_client = InfluxDBClient(
                url=self.config.influxdb_url,
                token=self.config.influxdb_token,
                org=self.config.influxdb_org
            )
            self.influxdb_write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)
        except Exception as e:
            self.logger.error(f"Failed to setup InfluxDB: {e}")
            self.influxdb_client = None
    
    def _setup_custom_metrics(self):
        """Setup custom application metrics"""
        self.custom_metrics = {
            'geological_model_accuracy': deque(maxlen=1000),
            'seismic_event_detection_rate': deque(maxlen=1000),
            'volcanic_eruption_predictions': deque(maxlen=1000),
            'data_ingestion_rate': deque(maxlen=1000),
            'api_response_times': defaultdict(lambda: deque(maxlen=1000)),
            'cache_hit_rates': defaultdict(lambda: deque(maxlen=1000)),
            'queue_depths': defaultdict(lambda: deque(maxlen=1000))
        }
    
    def _start_collection_threads(self):
        """Start background metric collection threads"""
        if self.config.enable_performance_monitoring:
            self.performance_thread = threading.Thread(
                target=self._collect_performance_metrics,
                daemon=True
            )
            self.performance_thread.start()
    
    def _collect_performance_metrics(self):
        """Background thread for performance metrics collection"""
        while True:
            try:
                metrics = self._gather_system_metrics()
                self.performance_history.append(metrics)
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE and self.config.enable_prometheus:
                    self._update_prometheus_metrics(metrics)
                
                # Send to InfluxDB
                if self.influxdb_client:
                    self._send_to_influxdb(metrics)
                
                time.sleep(self.config.performance_collection_interval)
                
            except Exception as e:
                self.logger.error(f"Performance metrics collection error: {e}")
                time.sleep(self.config.performance_collection_interval)
    
    def _gather_system_metrics(self) -> PerformanceMetrics:
        """Gather comprehensive system metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = usage.percent
                except PermissionError:
                    continue
            
            # Network I/O
            network_io = psutil.net_io_counters()._asdict()
            
            # Process information
            process = psutil.Process()
            process_info = {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'num_threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }
            
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=max(disk_usage.values()) if disk_usage else 0.0,
                network_io=network_io,
                process_count=len(psutil.pids()),
                thread_count=process_info['num_threads'],
                open_files=process_info['open_files'],
                response_times={},
                throughput={},
                error_rates={},
                custom_metrics=process_info
            )
            
        except Exception as e:
            self.logger.error(f"Error gathering system metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                process_count=0,
                thread_count=0,
                open_files=0,
                response_times={},
                throughput={},
                error_rates={}
            )
    
    def _update_prometheus_metrics(self, metrics: PerformanceMetrics):
        """Update Prometheus metrics"""
        try:
            instance = socket.gethostname()
            
            self.cpu_usage.labels(component='system', instance=instance).set(metrics.cpu_usage)
            self.memory_usage.labels(type='virtual', instance=instance).set(
                metrics.memory_usage * 1024 * 1024  # Convert to bytes
            )
            self.disk_usage.labels(mount_point='/', instance=instance).set(metrics.disk_usage)
            
        except Exception as e:
            self.logger.error(f"Error updating Prometheus metrics: {e}")
    
    def _send_to_influxdb(self, metrics: PerformanceMetrics):
        """Send metrics to InfluxDB"""
        try:
            points = [
                Point("system_metrics")
                .tag("host", socket.gethostname())
                .field("cpu_usage", metrics.cpu_usage)
                .field("memory_usage", metrics.memory_usage)
                .field("disk_usage", metrics.disk_usage)
                .field("process_count", metrics.process_count)
                .field("thread_count", metrics.thread_count)
                .time(metrics.timestamp, WritePrecision.MS)
            ]
            
            self.influxdb_write_api.write(
                bucket=self.config.influxdb_bucket,
                record=points
            )
            
        except Exception as e:
            self.logger.error(f"Error sending metrics to InfluxDB: {e}")
    
    def record_custom_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record custom application metric"""
        if metric_name in self.custom_metrics:
            if isinstance(self.custom_metrics[metric_name], dict):
                key = labels.get('key', 'default') if labels else 'default'
                self.custom_metrics[metric_name][key].append(value)
            else:
                self.custom_metrics[metric_name].append(value)
    
    @contextmanager
    def time_operation(self, operation_name: str, labels: Dict[str, str] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            if PROMETHEUS_AVAILABLE and hasattr(self, 'request_duration'):
                label_values = [labels.get('method', 'unknown'), labels.get('endpoint', operation_name)]
                self.request_duration.labels(*label_values).observe(duration)
            
            # Record in custom metrics
            self.record_custom_metric(f'{operation_name}_duration', duration, labels)

class DistributedTracer:
    """Distributed tracing implementation"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_tracing()
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        if not OPENTELEMETRY_AVAILABLE or not self.config.enable_jaeger:
            self.tracer = None
            return
        
        try:
            # Setup trace provider
            trace.set_tracer_provider(TracerProvider())
            
            # Setup Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            
            # Instrument asyncio
            AsyncioInstrumentor().instrument()
            
        except Exception as e:
            self.logger.error(f"Failed to setup distributed tracing: {e}")
            self.tracer = None
    
    @contextmanager
    def trace_operation(self, operation_name: str, attributes: Dict[str, Any] = None):
        """Context manager for tracing operations"""
        if not self.tracer:
            yield
            return
        
        with self.tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

class AlertManager:
    """Advanced alerting system"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.cooldown_periods = {}
        
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self):
        """Setup default system alert rules"""
        self.alert_rules = {
            'high_cpu_usage': {
                'condition': lambda metrics: metrics.cpu_usage > self.config.cpu_threshold,
                'severity': AlertSeverity.WARNING,
                'message': 'High CPU usage detected: {cpu_usage:.1f}%',
                'cooldown': 300  # 5 minutes
            },
            'high_memory_usage': {
                'condition': lambda metrics: metrics.memory_usage > self.config.memory_threshold,
                'severity': AlertSeverity.WARNING,
                'message': 'High memory usage detected: {memory_usage:.1f}%',
                'cooldown': 300
            },
            'high_disk_usage': {
                'condition': lambda metrics: metrics.disk_usage > self.config.disk_threshold,
                'severity': AlertSeverity.ERROR,
                'message': 'High disk usage detected: {disk_usage:.1f}%',
                'cooldown': 600  # 10 minutes
            },
            'process_crash': {
                'condition': lambda metrics: metrics.process_count < 1,
                'severity': AlertSeverity.CRITICAL,
                'message': 'Process crash detected',
                'cooldown': 0  # Immediate alert
            }
        }
    
    def add_alert_rule(self, rule_name: str, condition: Callable, severity: AlertSeverity, 
                      message: str, cooldown: int = 300):
        """Add custom alert rule"""
        self.alert_rules[rule_name] = {
            'condition': condition,
            'severity': severity,
            'message': message,
            'cooldown': cooldown
        }
    
    def check_alerts(self, metrics: PerformanceMetrics):
        """Check all alert rules against current metrics"""
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule['condition'](metrics):
                    # Check cooldown
                    last_alert_time = self.cooldown_periods.get(rule_name, 0)
                    if current_time - last_alert_time < rule['cooldown']:
                        continue
                    
                    # Format message
                    message = rule['message'].format(**metrics.__dict__)
                    
                    # Create alert
                    alert = {
                        'rule_name': rule_name,
                        'severity': rule['severity'],
                        'message': message,
                        'timestamp': datetime.utcnow(),
                        'metrics_snapshot': metrics
                    }
                    
                    self._trigger_alert(alert)
                    self.cooldown_periods[rule_name] = current_time
                    
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger alert through configured channels"""
        alert_id = str(uuid.uuid4())
        alert['alert_id'] = alert_id
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.warning(f"ALERT: {alert['message']}")
        
        # Send through configured channels
        for channel in self.config.alert_channels:
            try:
                if channel == 'webhook' and self.config.webhook_url:
                    self._send_webhook_alert(alert)
                elif channel == 'email' and self.config.email_config:
                    self._send_email_alert(alert)
                    
            except Exception as e:
                self.logger.error(f"Failed to send alert through {channel}: {e}")
    
    def _send_webhook_alert(self, alert: Dict[str, Any]):
        """Send alert via webhook"""
        if not ALERTING_AVAILABLE:
            return
        
        payload = {
            'alert_id': alert['alert_id'],
            'severity': alert['severity'].value,
            'message': alert['message'],
            'timestamp': alert['timestamp'].isoformat(),
            'rule_name': alert['rule_name']
        }
        
        response = requests.post(
            self.config.webhook_url,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send alert via email"""
        # Implementation would depend on email configuration
        pass

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.health_checks = {}
        self.health_status = {}
        
        self._register_default_health_checks()
        
        if config.enable_health_checks:
            self._start_health_check_thread()
    
    def _register_default_health_checks(self):
        """Register default health checks"""
        self.health_checks = {
            'system_resources': self._check_system_resources,
            'disk_space': self._check_disk_space,
            'network_connectivity': self._check_network_connectivity,
            'database_connectivity': self._check_database_connectivity
        }
    
    def _start_health_check_thread(self):
        """Start background health checking"""
        self.health_thread = threading.Thread(
            target=self._run_health_checks,
            daemon=True
        )
        self.health_thread.start()
    
    def _run_health_checks(self):
        """Run health checks periodically"""
        while True:
            try:
                for check_name, check_func in self.health_checks.items():
                    try:
                        result = check_func()
                        self.health_status[check_name] = {
                            'status': 'healthy' if result else 'unhealthy',
                            'last_check': datetime.utcnow(),
                            'result': result
                        }
                    except Exception as e:
                        self.health_status[check_name] = {
                            'status': 'error',
                            'last_check': datetime.utcnow(),
                            'error': str(e)
                        }
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                time.sleep(self.config.health_check_interval)
    
    def _check_system_resources(self) -> bool:
        """Check system resource availability"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            return cpu_usage < 95.0 and memory_usage < 95.0
        except:
            return False
    
    def _check_disk_space(self) -> bool:
        """Check disk space availability"""
        try:
            disk_usage = psutil.disk_usage('/').percent
            return disk_usage < 95.0
        except:
            return False
    
    def _check_network_connectivity(self) -> bool:
        """Check network connectivity"""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False
    
    def _check_database_connectivity(self) -> bool:
        """Check database connectivity"""
        # Implementation would depend on specific database connections
        return True
    
    def add_health_check(self, name: str, check_func: Callable[[], bool]):
        """Add custom health check"""
        self.health_checks[name] = check_func
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        overall_status = 'healthy'
        
        for check_name, status in self.health_status.items():
            if status['status'] != 'healthy':
                overall_status = 'unhealthy'
                break
        
        return {
            'overall_status': overall_status,
            'checks': self.health_status,
            'timestamp': datetime.utcnow().isoformat()
        }

class ComprehensiveLoggingMonitoring:
    """Main comprehensive logging and monitoring system"""
    
    def __init__(self, 
                 logging_config: LoggingConfig = None,
                 monitoring_config: MonitoringConfig = None):
        
        self.logging_config = logging_config or LoggingConfig()
        self.monitoring_config = monitoring_config or MonitoringConfig()
        
        # Initialize components
        self.structured_logger = StructuredLogger(self.logging_config)
        self.metrics_collector = MetricsCollector(self.monitoring_config)
        self.distributed_tracer = DistributedTracer(self.monitoring_config)
        self.alert_manager = AlertManager(self.monitoring_config)
        self.health_checker = HealthChecker(self.monitoring_config)
        
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)
        
        self.logger.info("Comprehensive logging and monitoring system initialized")
    
    def _shutdown_handler(self, signum, frame):
        """Handle graceful shutdown"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def log(self, level: str, message: str, **kwargs):
        """Log message with context"""
        self.structured_logger.log_with_context(level, message, **kwargs)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        self.log("INFO", f"Performance: {operation} took {duration:.3f}s", 
                operation=operation, duration=duration, **kwargs)
        
        if self.logging_config.enable_performance_logging and \
           duration * 1000 > self.logging_config.performance_threshold_ms:
            self.log("WARNING", f"Slow operation detected: {operation} took {duration:.3f}s",
                    operation=operation, duration=duration, slow_operation=True, **kwargs)
    
    def log_error(self, message: str, error: Exception = None, **kwargs):
        """Log error with context"""
        error_context = {
            'error_type': type(error).__name__ if error else 'Unknown',
            'error_message': str(error) if error else message,
            'traceback': traceback.format_exc() if error else None,
            **kwargs
        }
        
        self.log("ERROR", message, **error_context)
        
        # Record error metric
        if hasattr(self.metrics_collector, 'error_count'):
            self.metrics_collector.error_count.labels(
                error_type=error_context['error_type'],
                component=kwargs.get('component', 'unknown')
            ).inc()
    
    def record_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record custom metric"""
        self.metrics_collector.record_custom_metric(metric_name, value, labels)
    
    @contextmanager
    def trace_operation(self, operation_name: str, **kwargs):
        """Trace operation with timing and error handling"""
        start_time = time.time()
        
        with self.structured_logger.correlation_context() as correlation_id:
            with self.distributed_tracer.trace_operation(operation_name, kwargs) as span:
                try:
                    self.log("DEBUG", f"Starting operation: {operation_name}", 
                            operation=operation_name, correlation_id=correlation_id, **kwargs)
                    
                    yield correlation_id
                    
                    duration = time.time() - start_time
                    self.log_performance(operation_name, duration, **kwargs)
                    
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_error(f"Operation failed: {operation_name}", e, 
                                 operation=operation_name, duration=duration, **kwargs)
                    raise
    
    @contextmanager
    def time_operation(self, operation_name: str, labels: Dict[str, str] = None):
        """Time operation and record metrics"""
        with self.metrics_collector.time_operation(operation_name, labels):
            yield
    
    def get_metrics_endpoint(self):
        """Get Prometheus metrics endpoint handler"""
        if not PROMETHEUS_AVAILABLE or not self.metrics_collector.metrics_registry:
            return lambda: ("Metrics not available", 503)
        
        def metrics_handler():
            return generate_latest(self.metrics_collector.metrics_registry), 200, {
                'Content-Type': CONTENT_TYPE_LATEST
            }
        
        return metrics_handler
    
    def get_health_endpoint(self):
        """Get health check endpoint handler"""
        def health_handler():
            health_status = self.health_checker.get_health_status()
            status_code = 200 if health_status['overall_status'] == 'healthy' else 503
            
            return json.dumps(health_status), status_code, {
                'Content-Type': 'application/json'
            }
        
        return health_handler
    
    def start_performance_monitoring(self):
        """Start performance monitoring with alerts"""
        def monitor_and_alert():
            while True:
                try:
                    # Get latest metrics
                    if self.metrics_collector.performance_history:
                        latest_metrics = self.metrics_collector.performance_history[-1]
                        
                        # Check alerts
                        self.alert_manager.check_alerts(latest_metrics)
                    
                    time.sleep(self.monitoring_config.performance_collection_interval)
                    
                except Exception as e:
                    self.log_error("Performance monitoring error", e)
                    time.sleep(self.monitoring_config.performance_collection_interval)
        
        monitor_thread = threading.Thread(target=monitor_and_alert, daemon=True)
        monitor_thread.start()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'health': self.health_checker.get_health_status(),
            'metrics': {
                'performance_history_size': len(self.metrics_collector.performance_history),
                'active_alerts': len(self.alert_manager.active_alerts),
                'custom_metrics_count': len(self.metrics_collector.custom_metrics)
            },
            'configuration': {
                'logging_level': self.logging_config.log_level,
                'monitoring_enabled': self.monitoring_config.enable_metrics,
                'alerting_enabled': self.monitoring_config.enable_alerting,
                'tracing_enabled': self.monitoring_config.enable_jaeger
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def shutdown(self):
        """Graceful shutdown of monitoring system"""
        self.logger.info("Shutting down comprehensive logging and monitoring system")
        
        # Close InfluxDB client
        if hasattr(self.metrics_collector, 'influxdb_client') and \
           self.metrics_collector.influxdb_client:
            self.metrics_collector.influxdb_client.close()
        
        self.logger.info("Comprehensive logging and monitoring system shutdown complete")

# Utility functions for easy integration

def setup_geological_monitoring(log_level: str = "INFO", 
                               enable_prometheus: bool = True,
                               enable_alerting: bool = True) -> ComprehensiveLoggingMonitoring:
    """Setup monitoring system optimized for geological applications"""
    
    logging_config = LoggingConfig(
        log_level=log_level,
        log_format="json",
        enable_structured_logging=True,
        enable_performance_logging=True,
        performance_threshold_ms=2000.0  # Geological computations can be slower
    )
    
    monitoring_config = MonitoringConfig(
        enable_prometheus=enable_prometheus,
        enable_alerting=enable_alerting,
        metrics_prefix="prometheus_geological",
        cpu_threshold=85.0,  # Higher threshold for compute-intensive operations
        memory_threshold=90.0,
        performance_collection_interval=10.0  # More frequent for critical systems
    )
    
    return ComprehensiveLoggingMonitoring(logging_config, monitoring_config)

def create_geological_logger(component_name: str) -> logging.Logger:
    """Create logger specifically for geological components"""
    logger = logging.getLogger(f"geological.{component_name}")
    return logger

# Example usage and integration patterns
if __name__ == "__main__":
    # Example setup
    monitor = setup_geological_monitoring()
    
    # Example usage
    with monitor.trace_operation("seismic_analysis", model_type="transformer"):
        # Simulate geological computation
        time.sleep(1.5)
        monitor.record_metric("seismic_events_processed", 42.0)
    
    # Get system status
    status = monitor.get_system_status()
    print(json.dumps(status, indent=2, default=str))