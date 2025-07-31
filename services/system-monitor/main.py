#!/usr/bin/env python3
"""
System Monitoring and Alerting Service
Provides comprehensive monitoring, health checks, and automated incident response
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import aiohttp
import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import kubernetes
from kubernetes import client, config

# Configure structured logging
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

logger = structlog.get_logger()

# Prometheus metrics
HEALTH_CHECK_COUNTER = Counter('health_checks_total', 'Total health checks performed', ['service', 'status'])
RESPONSE_TIME_HISTOGRAM = Histogram('service_response_time_seconds', 'Service response time', ['service'])
SYSTEM_LOAD_GAUGE = Gauge('system_load_average', 'System load average', ['period'])
INCIDENT_COUNTER = Counter('incidents_total', 'Total incidents detected', ['severity', 'service'])
SCALING_ACTIONS_COUNTER = Counter('scaling_actions_total', 'Total scaling actions performed', ['service', 'direction'])

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DOWN = "down"

@dataclass
class HealthCheckResult:
    service_name: str
    status: ServiceStatus
    response_time: float
    timestamp: datetime
    details: Dict[str, Any]
    dependencies: List[str]

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    queue_depth: int
    timestamp: datetime

@dataclass
class Alert:
    id: str
    severity: AlertSeverity
    service: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    escalated: bool = False

class MonitoringService:
    def __init__(self):
        self.app = FastAPI(title="System Monitoring Service", version="1.0.0")
        self.redis_client = None
        self.db_pool = None
        self.k8s_client = None
        self.active_alerts: Dict[str, Alert] = {}
        self.service_configs = self._load_service_configs()
        self.setup_routes()
        
    def _load_service_configs(self) -> Dict[str, Dict]:
        """Load service monitoring configurations"""
        return {
            "api-gateway": {
                "url": "http://api-gateway-service:8080",
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics",
                "critical": True,
                "dependencies": ["redis", "consul"],
                "scaling": {
                    "min_replicas": 2,
                    "max_replicas": 10,
                    "cpu_threshold": 70,
                    "memory_threshold": 80
                }
            },
            "helios-core": {
                "url": "http://helios-core-service:8000",
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics",
                "critical": True,
                "dependencies": ["postgresql", "redis", "influxdb"],
                "scaling": {
                    "min_replicas": 3,
                    "max_replicas": 20,
                    "cpu_threshold": 80,
                    "memory_threshold": 85
                }
            },
            "volcano-watchtower": {
                "url": "http://volcano-watchtower-service:8000",
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics",
                "critical": True,
                "dependencies": ["kafka", "influxdb", "redis"],
                "scaling": {
                    "min_replicas": 2,
                    "max_replicas": 8,
                    "cpu_threshold": 75,
                    "memory_threshold": 80
                }
            },
            "data-ingestion": {
                "url": "http://data-ingestion-service:8000",
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics",
                "critical": True,
                "dependencies": ["kafka", "postgresql", "influxdb"],
                "scaling": {
                    "min_replicas": 2,
                    "max_replicas": 15,
                    "cpu_threshold": 70,
                    "memory_threshold": 75
                }
            },
            "nlp-service": {
                "url": "http://nlp-service:8000",
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics",
                "critical": False,
                "dependencies": ["redis"],
                "scaling": {
                    "min_replicas": 1,
                    "max_replicas": 5,
                    "cpu_threshold": 80,
                    "memory_threshold": 85
                }
            }
        }

    async def initialize(self):
        """Initialize connections and Kubernetes client"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'redis'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                decode_responses=True
            )
            
            # Initialize PostgreSQL connection pool
            self.db_pool = await asyncpg.create_pool(
                host=os.getenv('POSTGRES_HOST', 'postgresql'),
                port=int(os.getenv('POSTGRES_PORT', 5432)),
                user=os.getenv('POSTGRES_USER', 'prometheus'),
                password=os.getenv('POSTGRES_PASSWORD', 'prometheus123'),
                database=os.getenv('POSTGRES_DB', 'prometheus'),
                min_size=2,
                max_size=10
            )
            
            # Initialize Kubernetes client
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            
            self.k8s_client = client.AppsV1Api()
            
            logger.info("Monitoring service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize monitoring service", error=str(e))
            raise

    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def __init__(self):
        self.processors = {
            'time_series': TimeSeriesProcessor(),
            'spatial': SpatialDataProcessor(),
            'spectral': SpectralAnalysisProcessor(),
            'statistical': StatisticalProcessor(),
            'ml_preprocessing': MLPreprocessor()
        }
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', str(uuid.uuid4())),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps based on configuration
        for step_name, step_config in processing_config.items():
            if step_name in self.processors:
                try:
                    processor = self.processors[step_name]
                    processed_data, step_results = processor.process(data, step_config)
                    
                    data = processed_data
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'success',
                        'results': step_results,
                        'output_size': len(data) if isinstance(data, list) else 'N/A'
                    })
                    
                except Exception as e:
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results

class TimeSeriesProcessor:
    """Advanced time series processing for geological data"""
    
    def process(self, data: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process time series data"""
        
        if not isinstance(data, (list, np.ndarray)):
            return data, {'error': 'Invalid data type for time series processing'}
        
        data_array = np.array(data)
        results = {}
        
        # Detrending
        if config.get('detrend', False):
            from scipy import signal
            data_array = signal.detrend(data_array, type=config.get('detrend_type', 'linear'))
            results['detrended'] = True
        
        # Filtering
        if 'filter' in config:
            filter_config = config['filter']
            data_array = self._apply_filter(data_array, filter_config)
            results['filtered'] = filter_config
        
        # Resampling
        if 'resample' in config:
            resample_config = config['resample']
            data_array = self._resample_data(data_array, resample_config)
            results['resampled'] = resample_config
        
        # Anomaly detection
        if config.get('detect_anomalies', False):
            anomalies = self._detect_anomalies(data_array, config.get('anomaly_config', {}))
            results['anomalies'] = anomalies
        
        return data_array.tolist(), results
    
    def _apply_filter(self, data: np.ndarray, filter_config: Dict[str, Any]) -> np.ndarray:
        """Apply digital filter to data"""
        from scipy import signal
        
        filter_type = filter_config.get('type', 'bandpass')
        sampling_rate = filter_config.get('sampling_rate', 100.0)
        
        if filter_type == 'bandpass':
            low_freq = filter_config.get('low_freq', 1.0)
            high_freq = filter_config.get('high_freq', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                [low_freq/nyquist, high_freq/nyquist],
                btype='band'
            )
            return signal.filtfilt(b, a, data)
        
        elif filter_type == 'lowpass':
            cutoff = filter_config.get('cutoff', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                cutoff/nyquist,
                btype='low'
            )
            return signal.filtfilt(b, a, data)
        
        return data
    
    def _resample_data(self, data: np.ndarray, resample_config: Dict[str, Any]) -> np.ndarray:
        """Resample data to different sampling rate"""
        from scipy import signal
        
        original_rate = resample_config.get('original_rate', 100.0)
        target_rate = resample_config.get('target_rate', 50.0)
        
        if original_rate != target_rate:
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        
        return data
    
    def _detect_anomalies(self, data: np.ndarray, anomaly_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        method = anomaly_config.get('method', 'statistical')
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = anomaly_config.get('threshold', 3.0)
            
            anomaly_indices = np.where(np.abs(data - mean_val) > threshold * std_val)[0]
            
            return {
                'method': 'statistical',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'threshold_used': threshold
            }
        
        elif method == 'isolation_forest':
            # Machine learning based anomaly detection
            from sklearn.ensemble import IsolationForest
            
            contamination = anomaly_config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            
            data_reshaped = data.reshape(-1, 1)
            anomaly_labels = clf.fit_predict(data_reshaped)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'contamination': contamination
            }
        
        return {'method': 'none', 'anomaly_count': 0}

        
        @self.app.get("/metrics")
        async def metrics():
            return generate_latest()
        
        @self.app.get("/system/status")
        async def system_status():
            """Get overall system status"""
            try:
                health_results = await self.perform_health_checks()
                system_metrics = await self.collect_system_metrics()
                active_alerts = [asdict(alert) for alert in self.active_alerts.values()]
                
                overall_status = self._determine_overall_status(health_results)
                
                return {
                    "overall_status": overall_status.value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "services": [asdict(result) for result in health_results],
                    "system_metrics": asdict(system_metrics),
                    "active_alerts": active_alerts,
                    "total_services": len(health_results),
                    "healthy_services": len([r for r in health_results if r.status == ServiceStatus.HEALTHY])
                }
            except Exception as e:
                logger.error("Failed to get system status", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to get system status")
        
        @self.app.post("/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """Manually resolve an alert"""
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                await self._log_alert_resolution(alert_id)
                return {"message": f"Alert {alert_id} resolved"}
            else:
                raise HTTPException(status_code=404, detail="Alert not found")
        
        @self.app.get("/scaling/status")
        async def scaling_status():
            """Get current scaling status for all services"""
            try:
                scaling_info = await self.get_scaling_status()
                return scaling_info
            except Exception as e:
                logger.error("Failed to get scaling status", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to get scaling status")

    async def perform_health_checks(self) -> List[HealthCheckResult]:
        """Perform health checks on all monitored services"""
        health_results = []
        
        for service_name, config in self.service_configs.items():
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    health_url = f"{config['url']}{config['health_endpoint']}"
                    
                    async with session.get(health_url) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            response_data = await response.json()
                            status = ServiceStatus.HEALTHY
                            details = response_data
                        else:
                            status = ServiceStatus.UNHEALTHY
                            details = {"error": f"HTTP {response.status}"}
                        
                        # Check dependencies
                        dependency_status = await self._check_dependencies(config.get('dependencies', []))
                        if not dependency_status['all_healthy']:
                            status = ServiceStatus.DEGRADED
                            details.update(dependency_status)
                        
                        result = HealthCheckResult(
                            service_name=service_name,
                            status=status,
                            response_time=response_time,
                            timestamp=datetime.utcnow(),
                            details=details,
                            dependencies=config.get('dependencies', [])
                        )
                        
                        health_results.append(result)
                        
                        # Update Prometheus metrics
                        HEALTH_CHECK_COUNTER.labels(service=service_name, status=status.value).inc()
                        RESPONSE_TIME_HISTOGRAM.labels(service=service_name).observe(response_time)
                        
            except Exception as e:
                logger.error(f"Health check failed for {service_name}", error=str(e))
                
                result = HealthCheckResult(
                    service_name=service_name,
                    status=ServiceStatus.DOWN,
                    response_time=0.0,
                    timestamp=datetime.utcnow(),
                    details={"error": str(e)},
                    dependencies=config.get('dependencies', [])
                )
                
                health_results.append(result)
                HEALTH_CHECK_COUNTER.labels(service=service_name, status="down").inc()
        
        return health_results

    async def _check_dependencies(self, dependencies: List[str]) -> Dict[str, Any]:
        """Check the health of service dependencies"""
        dependency_results = {}
        all_healthy = True
        
        for dep in dependencies:
            try:
                if dep == "redis":
                    await self.redis_client.ping()
                    dependency_results[dep] = "healthy"
                elif dep == "postgresql":
                    async with self.db_pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                    dependency_results[dep] = "healthy"
                elif dep in ["kafka", "influxdb", "consul"]:
                    # For now, assume these are healthy if the service is running
                    # In production, implement specific health checks
                    dependency_results[dep] = "healthy"
                else:
                    dependency_results[dep] = "unknown"
                    
            except Exception as e:
                dependency_results[dep] = f"unhealthy: {str(e)}"
                all_healthy = False
        
        return {
            "all_healthy": all_healthy,
            "dependencies": dependency_results
        }

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-wide metrics"""
        try:
            # Get Kubernetes cluster metrics
            pods = self.k8s_client.list_namespaced_pod(namespace="prometheus-ai")
            
            # Calculate aggregate metrics
            total_cpu_requests = 0
            total_memory_requests = 0
            active_pods = 0
            
            for pod in pods.items:
                if pod.status.phase == "Running":
                    active_pods += 1
                    if pod.spec.containers:
                        for container in pod.spec.containers:
                            if container.resources and container.resources.requests:
                                if 'cpu' in container.resources.requests:
                                    total_cpu_requests += self._parse_cpu(container.resources.requests['cpu'])
                                if 'memory' in container.resources.requests:
                                    total_memory_requests += self._parse_memory(container.resources.requests['memory'])
            
            # Get Redis queue depth
            queue_depth = 0
            try:
                queue_keys = await self.redis_client.keys("queue:*")
                for key in queue_keys:
                    length = await self.redis_client.llen(key)
                    queue_depth += length
            except:
                queue_depth = 0
            
            return SystemMetrics(
                cpu_usage=min(total_cpu_requests / 1000 * 100, 100),  # Convert to percentage
                memory_usage=min(total_memory_requests / (1024**3) * 100, 100),  # Convert to percentage
                disk_usage=0.0,  # Would need node metrics for accurate disk usage
                network_io={"ingress": 0.0, "egress": 0.0},  # Would need network metrics
                active_connections=active_pods,
                queue_depth=queue_depth,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            return SystemMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={"ingress": 0.0, "egress": 0.0},
                active_connections=0,
                queue_depth=0,
                timestamp=datetime.utcnow()
            )

    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse Kubernetes CPU resource string to millicores"""
        if cpu_str.endswith('m'):
            return float(cpu_str[:-1])
        else:
            return float(cpu_str) * 1000

    def _parse_memory(self, memory_str: str) -> float:
        """Parse Kubernetes memory resource string to bytes"""
        units = {'Ki': 1024, 'Mi': 1024**2, 'Gi': 1024**3, 'Ti': 1024**4}
        for unit, multiplier in units.items():
            if memory_str.endswith(unit):
                return float(memory_str[:-len(unit)]) * multiplier
        return float(memory_str)

    def _determine_overall_status(self, health_results: List[HealthCheckResult]) -> ServiceStatus:
        """Determine overall system status based on individual service health"""
        critical_services = [
            result for result in health_results 
            if self.service_configs[result.service_name].get('critical', False)
        ]
        
        # If any critical service is down, system is down
        if any(result.status == ServiceStatus.DOWN for result in critical_services):
            return ServiceStatus.DOWN
        
        # If any critical service is unhealthy, system is unhealthy
        if any(result.status == ServiceStatus.UNHEALTHY for result in critical_services):
            return ServiceStatus.UNHEALTHY
        
        # If any service is degraded, system is degraded
        if any(result.status == ServiceStatus.DEGRADED for result in health_results):
            return ServiceStatus.DEGRADED
        
        return ServiceStatus.HEALTHY

    async def check_scaling_requirements(self):
        """Check if any services need scaling based on metrics"""
        try:
            for service_name, config in self.service_configs.items():
                scaling_config = config.get('scaling', {})
                if not scaling_config:
                    continue
                
                # Get current replica count
                try:
                    deployment = self.k8s_client.read_namespaced_deployment(
                        name=f"{service_name}-deployment",
                        namespace="prometheus-ai"
                    )
                    current_replicas = deployment.status.replicas or 0
                    
                    # Check if scaling is needed based on metrics
                    # This is a simplified example - in production, you'd use actual metrics
                    should_scale_up = await self._should_scale_up(service_name, scaling_config)
                    should_scale_down = await self._should_scale_down(service_name, scaling_config)
                    
                    if should_scale_up and current_replicas < scaling_config['max_replicas']:
                        await self._scale_service(service_name, current_replicas + 1)
                    elif should_scale_down and current_replicas > scaling_config['min_replicas']:
                        await self._scale_service(service_name, current_replicas - 1)
                        
                except Exception as e:
                    logger.error(f"Failed to check scaling for {service_name}", error=str(e))
                    
        except Exception as e:
            logger.error("Failed to check scaling requirements", error=str(e))

    async def _should_scale_up(self, service_name: str, scaling_config: Dict) -> bool:
        """Determine if service should scale up based on metrics"""
        
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime, timedelta
import uuid

class AdvancedGeologicalAPI:
    """Advanced API endpoints for geological data and analysis"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Prometheus Geological Analysis API",
            description="Advanced geological simulation and monitoring API",
            version="2.0.0"
        )
        self.security = HTTPBearer()
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_middleware(self):
        """Setup API middleware for logging, CORS, rate limiting"""
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/v2/seismic/analyze", response_model=SeismicAnalysisResponse)
        async def analyze_seismic_data(
            request: SeismicAnalysisRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Advanced seismic data analysis endpoint"""
            
            # Validate authentication
            user_id = await self._validate_token(credentials.credentials)
            
            # Generate analysis ID
            analysis_id = str(uuid.uuid4())
            
            try:
                # Initialize seismic processor
                processor = AdvancedSeismicProcessor(sampling_rate=request.sampling_rate)
                
                # Process seismic data
                analysis_result = processor.process_seismic_stream(
                    data=np.array(request.waveform_data),
                    station_info=request.station_info.dict()
                )
                
                # Add metadata
                analysis_result.update({
                    'analysis_id': analysis_id,
                    'user_id': user_id,
                    'request_timestamp': datetime.now().isoformat(),
                    'processing_time_ms': 0  # Would be calculated
                })
                
                # Store results asynchronously
                background_tasks.add_task(
                    self._store_analysis_results,
                    analysis_id, analysis_result, 'seismic'
                )
                
                return SeismicAnalysisResponse(**analysis_result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        @self.app.post("/api/v2/volcanic/monitor", response_model=VolcanicMonitoringResponse)
        async def monitor_volcanic_activity(
            request: VolcanicMonitoringRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Comprehensive volcanic monitoring endpoint"""
            
            user_id = await self._validate_token(credentials.credentials)
            monitoring_id = str(uuid.uuid4())
            
            try:
                # Initialize volcanic monitor
                monitor = AdvancedVolcanicMonitor()
                
                # Perform comprehensive assessment
                assessment = monitor.comprehensive_volcanic_assessment(
                    monitoring_data=request.monitoring_data,
                    volcano_info=request.volcano_info.dict()
                )
                
                # Add metadata
                assessment.update({
                    'monitoring_id': monitoring_id,
                    'user_id': user_id,
                    'request_timestamp': datetime.now().isoformat()
                })
                
                # Store results and trigger alerts if necessary
                background_tasks.add_task(
                    self._process_volcanic_assessment,
                    monitoring_id, assessment
                )
                
                return VolcanicMonitoringResponse(**assessment)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Monitoring failed: {str(e)}")
        
        @self.app.post("/api/v2/geochemical/analyze", response_model=GeochemicalAnalysisResponse)
        async def analyze_geochemical_composition(
            request: GeochemicalAnalysisRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Advanced geochemical composition analysis"""
            
            user_id = await self._validate_token(credentials.credentials)
            analysis_id = str(uuid.uuid4())
            
            try:
                # Initialize geochemical analyzer
                analyzer = AdvancedGeochemicalAnalyzer()
                
                # Perform analysis
                analysis_result = analyzer.analyze_rock_composition(
                    composition=request.composition,
                    temperature=request.temperature,
                    pressure=request.pressure
                )
                
                # Add metadata
                analysis_result.update({
                    'analysis_id': analysis_id,
                    'user_id': user_id,
                    'sample_id': request.sample_id,
                    'request_timestamp': datetime.now().isoformat()
                })
                
                # Store results
                background_tasks.add_task(
                    self._store_analysis_results,
                    analysis_id, analysis_result, 'geochemical'
                )
                
                return GeochemicalAnalysisResponse(**analysis_result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Geochemical analysis failed: {str(e)}")
        
        @self.app.get("/api/v2/analysis/{analysis_id}", response_model=AnalysisStatusResponse)
        async def get_analysis_status(
            analysis_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get analysis status and results"""
            
            user_id = await self._validate_token(credentials.credentials)
            
            try:
                # Retrieve analysis from database/cache
                analysis_data = await self._get_analysis_results(analysis_id, user_id)
                
                if not analysis_data:
                    raise HTTPException(status_code=404, detail="Analysis not found")
                
                return AnalysisStatusResponse(**analysis_data)
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis: {str(e)}")
        
        @self.app.get("/api/v2/spatial/query", response_model=SpatialQueryResponse)
        async def spatial_geological_query(
            bbox: str = Query(..., description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
            start_time: str = Query(..., description="Start time (ISO format)"),
            end_time: str = Query(..., description="End time (ISO format)"),
            data_types: List[str] = Query(["seismic"], description="Data types to include"),
            min_magnitude: float = Query(0.0, description="Minimum magnitude"),
            max_magnitude: float = Query(10.0, description="Maximum magnitude"),
            limit: int = Query(1000, le=10000, description="Maximum results"),
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Spatial query for geological data"""
            
            user_id = await self._validate_token(credentials.credentials)
            
            try:
                # Parse bounding box
                bbox_coords = [float(x) for x in bbox.split(',')]
                if len(bbox_coords) != 4:
                    raise ValueError("Invalid bounding box format")
                
                # Initialize database
                db = AdvancedGeologicalDatabase({
                    'postgres': {'host': 'localhost', 'database': 'geological_data'},
                    'influxdb': {'url': 'http://localhost:8086'}
                })
                
                # Perform spatial query
                results = db.query_spatial_events(
                    bbox=tuple(bbox_coords),
                    time_range=(start_time, end_time),
                    magnitude_range=(min_magnitude, max_magnitude)
                )
                
                # Limit results
                results = results[:limit]
                
                return SpatialQueryResponse(
                    query_id=str(uuid.uuid4()),
                    user_id=user_id,
                    bbox=bbox_coords,
                    time_range=[start_time, end_time],
                    result_count=len(results),
                    results=results,
                    query_timestamp=datetime.now().isoformat()
                )
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Spatial query failed: {str(e)}")
    
    async def _validate_token(self, token: str) -> str:
        """Validate JWT token and return user ID"""
        # Simplified token validation - implement proper JWT validation
        if not token or len(token) < 10:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Extract user ID from token (simplified)
        return f"user_{hash(token) % 10000}"
    
    async def _store_analysis_results(self, analysis_id: str, results: Dict[str, Any], analysis_type: str):
        """Store analysis results in database"""
        try:
            # Initialize database connection
            db = AdvancedGeologicalDatabase({
                'redis': {'host': 'localhost', 'port': 6379}
            })
            
            # Cache results
            db.cache_analysis_results(analysis_id, results, expiration_hours=24)
            
            logger.info(f"Stored {analysis_type} analysis results for {analysis_id}")
            
        except Exception as e:
            logger.error(f"Failed to store analysis results: {e}")
    
    async def _get_analysis_results(self, analysis_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis results from database"""
        try:
            db = AdvancedGeologicalDatabase({
                'redis': {'host': 'localhost', 'port': 6379}
            })
            
            results = db.get_cached_results(analysis_id)
            
            # Verify user has access to this analysis
            if results and results.get('user_id') == user_id:
                return results
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve analysis results: {e}")
            return None

# Pydantic models for request/response validation
class StationInfo(BaseModel):
    station_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    elevation_m: float
    network: str

class SeismicAnalysisRequest(BaseModel):
    waveform_data: List[List[float]]
    sampling_rate: float = Field(..., gt=0, le=1000)
    station_info: StationInfo
    analysis_parameters: Optional[Dict[str, Any]] = {}

class SeismicAnalysisResponse(BaseModel):
    analysis_id: str
    station_id: str
    processing_timestamp: str
    components: List[Dict[str, Any]]
    cross_component_analysis: Dict[str, Any]
    overall_quality: float

class VolcanicMonitoringRequest(BaseModel):
    volcano_info: Dict[str, Any]
    monitoring_data: Dict[str, Any]
    analysis_parameters: Optional[Dict[str, Any]] = {}

class VolcanicMonitoringResponse(BaseModel):
    monitoring_id: str
    volcano_id: str
    assessment_timestamp: str
    current_alert_level: str
    eruption_probability: Dict[str, float]
    recommendations: List[str]

        try:
            # Check recent response times
            health_results = await self.perform_health_checks()
            service_health = next((r for r in health_results if r.service_name == service_name), None)
            
            if service_health and service_health.response_time > 2.0:  # 2 second threshold
                return True
                
            # Check queue depth for data processing services
            if service_name in ["data-ingestion", "volcano-watchtower"]:
                queue_depth = await self.redis_client.llen(f"queue:{service_name}")
                if queue_depth > 100:  # Arbitrary threshold
                    return True
                    
            return False
            
        except Exception:
            return False

    async def _should_scale_down(self, service_name: str, scaling_config: Dict) -> bool:
        """Determine if service should scale down based on metrics"""
        try:
            # Check if service has been underutilized
            health_results = await self.perform_health_checks()
            service_health = next((r for r in health_results if r.service_name == service_name), None)
            
            if service_health and service_health.response_time < 0.1:  # Very fast response
                # Check queue depth is low
                if service_name in ["data-ingestion", "volcano-watchtower"]:
                    queue_depth = await self.redis_client.llen(f"queue:{service_name}")
                    if queue_depth < 10:
                        return True
                else:
                    return True
                    
            return False
            
        except Exception:
            return False

    async def _scale_service(self, service_name: str, target_replicas: int):
        """Scale a service to the target number of replicas"""
        try:
            deployment_name = f"{service_name}-deployment"
            
            # Update the deployment
            deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name,
                namespace="prometheus-ai"
            )
            
            deployment.spec.replicas = target_replicas
            
            self.k8s_client.patch_namespaced_deployment(
                name=deployment_name,
                namespace="prometheus-ai",
                body=deployment
            )
            
            direction = "up" if target_replicas > deployment.status.replicas else "down"
            SCALING_ACTIONS_COUNTER.labels(service=service_name, direction=direction).inc()
            
            logger.info(
                f"Scaled {service_name} to {target_replicas} replicas",
                service=service_name,
                target_replicas=target_replicas,
                direction=direction
            )
            
            # Create alert for scaling action
            alert = Alert(
                id=f"scaling-{service_name}-{int(time.time())}",
                severity=AlertSeverity.LOW,
                service=service_name,
                message=f"Service scaled {direction} to {target_replicas} replicas",
                details={"target_replicas": target_replicas, "direction": direction},
                timestamp=datetime.utcnow()
            )
            
            await self._process_alert(alert)
            
        except Exception as e:
            logger.error(f"Failed to scale {service_name}", error=str(e))

    async def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status for all services"""
        scaling_status = {}
        
        for service_name, config in self.service_configs.items():
            scaling_config = config.get('scaling', {})
            if not scaling_config:
                continue
                
            try:
                deployment = self.k8s_client.read_namespaced_deployment(
                    name=f"{service_name}-deployment",
                    namespace="prometheus-ai"
                )
                
                scaling_status[service_name] = {
                    "current_replicas": deployment.status.replicas or 0,
                    "desired_replicas": deployment.spec.replicas or 0,
                    "min_replicas": scaling_config['min_replicas'],
                    "max_replicas": scaling_config['max_replicas'],
                    "ready_replicas": deployment.status.ready_replicas or 0,
                    "updated_replicas": deployment.status.updated_replicas or 0
                }
                
            except Exception as e:
                scaling_status[service_name] = {
                    "error": str(e)
                }
        
        return scaling_status

    async def _process_alert(self, alert: Alert):
        """Process and potentially escalate an alert"""
        self.active_alerts[alert.id] = alert
        
        # Log the alert
        logger.warning(
            "Alert generated",
            alert_id=alert.id,
            severity=alert.severity.value,
            service=alert.service,
            message=alert.message
        )
        
        # Store in database
        await self._store_alert(alert)
        
        # Send notifications based on severity
        await self._send_alert_notifications(alert)
        
        # Update Prometheus metrics
        INCIDENT_COUNTER.labels(severity=alert.severity.value, service=alert.service).inc()

    async def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO alerts (id, severity, service, message, details, timestamp, resolved, escalated)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, alert.id, alert.severity.value, alert.service, alert.message, 
                    str(alert.details), alert.timestamp, alert.resolved, alert.escalated)
        except Exception as e:
            logger.error("Failed to store alert", error=str(e))

    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications based on severity"""
        try:
            # Store notification in Redis for external systems to pick up
            notification = {
                "alert_id": alert.id,
                "severity": alert.severity.value,
                "service": alert.service,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "details": alert.details
            }
            
            # Different channels based on severity
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                await self.redis_client.lpush("alerts:critical", str(notification))
            elif alert.severity == AlertSeverity.HIGH:
                await self.redis_client.lpush("alerts:high", str(notification))
            else:
                await self.redis_client.lpush("alerts:normal", str(notification))
                
        except Exception as e:
            logger.error("Failed to send alert notifications", error=str(e))

    async def _log_alert_resolution(self, alert_id: str):
        """Log alert resolution"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE alerts SET resolved = true, resolved_at = $1 WHERE id = $2
                """, datetime.utcnow(), alert_id)
                
            logger.info("Alert resolved", alert_id=alert_id)
            
        except Exception as e:
            logger.error("Failed to log alert resolution", error=str(e))

    async def monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Perform health checks
                health_results = await self.perform_health_checks()
                
                # Check for issues and generate alerts
                await self._analyze_health_results(health_results)
                
                # Check scaling requirements
                await self.check_scaling_requirements()
                
                # Clean up resolved alerts
                await self._cleanup_resolved_alerts()
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error

    async def _analyze_health_results(self, health_results: List[HealthCheckResult]):
        """Analyze health check results and generate alerts if needed"""
        for result in health_results:
            service_config = self.service_configs[result.service_name]
            
            # Check for service down
            if result.status == ServiceStatus.DOWN:
                severity = AlertSeverity.CRITICAL if service_config.get('critical') else AlertSeverity.HIGH
                
                # Special handling for volcanic monitoring system
                if result.service_name == "volcano-watchtower":
                    severity = AlertSeverity.EMERGENCY
                
                alert = Alert(
                    id=f"service-down-{result.service_name}-{int(time.time())}",
                    severity=severity,
                    service=result.service_name,
                    message=f"Service {result.service_name} is down",
                    details=result.details,
                    timestamp=datetime.utcnow()
                )
                
                await self._process_alert(alert)
            
            # Check for high response times
            elif result.response_time > 5.0:  # 5 second threshold
                alert = Alert(
                    id=f"high-latency-{result.service_name}-{int(time.time())}",
                    severity=AlertSeverity.MEDIUM,
                    service=result.service_name,
                    message=f"High response time detected for {result.service_name}: {result.response_time:.2f}s",
                    details={"response_time": result.response_time},
                    timestamp=datetime.utcnow()
                )
                
                await self._process_alert(alert)

    async def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            resolved_alerts = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.resolved and alert.timestamp < cutoff_time
            ]
            
            for alert_id in resolved_alerts:
                del self.active_alerts[alert_id]
                
        except Exception as e:
            logger.error("Failed to cleanup resolved alerts", error=str(e))

# Initialize the monitoring service
monitoring_service = MonitoringService()

@monitoring_service.app.on_event("startup")
async def startup_event():
    await monitoring_service.initialize()
    # Start the monitoring loop in the background
    asyncio.create_task(monitoring_service.monitoring_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:monitoring_service.app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_config=None  # Use structlog configuration
    )