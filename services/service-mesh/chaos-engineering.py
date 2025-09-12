"""
Chaos Engineering Implementation for Prometheus AI Platform
Advanced chaos testing with intelligent failure injection and monitoring
"""

import asyncio
import random
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import psutil
import docker
import kubernetes
from prometheus_client import Counter, Histogram, Gauge
import yaml

logger = logging.getLogger(__name__)

class ChaosType(Enum):
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    NETWORK_LOSS = "network_loss"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    SERVICE_KILL = "service_kill"
    DATABASE_FAILURE = "database_failure"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_DRIFT = "configuration_drift"
    TIME_SKEW = "time_skew"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

class ChaosScope(Enum):
    SINGLE_INSTANCE = "single_instance"
    SERVICE_CLUSTER = "service_cluster"
    AVAILABILITY_ZONE = "availability_zone"
    REGION = "region"
    GLOBAL = "global"

class ChaosStatus(Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ChaosTarget:
    """Target for chaos experiment"""
    service_name: str
    instance_id: Optional[str] = None
    namespace: str = "prometheus-ai"
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Resource constraints
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    
    # Network configuration
    network_interface: str = "eth0"
    port_ranges: List[str] = field(default_factory=list)

@dataclass
class ChaosParameters:
    """Parameters for chaos experiment"""
    # Duration and timing
    duration: int = 300  # seconds
    delay: int = 0  # seconds before starting
    
    # Intensity parameters
    intensity: float = 1.0  # 0.0 to 1.0
    percentage: float = 100.0  # percentage of targets affected
    
    # Network chaos parameters
    latency_ms: int = 100
    packet_loss_percent: float = 10.0
    bandwidth_limit: str = "1mbps"
    
    # Resource chaos parameters
    cpu_percent: float = 80.0
    memory_percent: float = 80.0
    disk_io_percent: float = 80.0
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChaosExperiment:
    """Complete chaos experiment definition"""
    name: str
    description: str
    chaos_type: ChaosType
    scope: ChaosScope
    targets: List[ChaosTarget]
    parameters: ChaosParameters
    
    # Experiment metadata
    id: str = field(default_factory=lambda: f"chaos-{int(time.time())}")
    created_at: datetime = field(default_factory=datetime.now)
    status: ChaosStatus = ChaosStatus.SCHEDULED
    
    # Safety and rollback
    safety_checks: List[str] = field(default_factory=list)
    rollback_strategy: str = "immediate"
    max_blast_radius: float = 0.5  # maximum percentage of infrastructure affected
    
    # Monitoring and validation
    success_criteria: List[str] = field(default_factory=list)
    failure_criteria: List[str] = field(default_factory=list)
    monitoring_queries: List[str] = field(default_factory=list)
    
    # Results
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

class ChaosMonkey:
    """
    Advanced Chaos Engineering platform with intelligent failure injection,
    safety mechanisms, and comprehensive monitoring
    """
    
    def __init__(self, 
                 kubernetes_config: Optional[str] = None,
                 docker_client: Optional[docker.DockerClient] = None,
                 prometheus_url: str = "http://prometheus:9090"):
        
        # Initialize clients
        if kubernetes_config:
            kubernetes.config.load_kube_config(kubernetes_config)
            self.k8s_client = kubernetes.client.ApiClient()
            self.k8s_apps_v1 = kubernetes.client.AppsV1Api()
            self.k8s_core_v1 = kubernetes.client.CoreV1Api()
        else:
            self.k8s_client = None
        
        self.docker_client = docker_client or docker.from_env()
        self.prometheus_url = prometheus_url
        
        # Experiment management
        self.active_experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_history: List[ChaosExperiment] = []
        
        # Safety mechanisms
        self.safety_enabled = True
        self.max_concurrent_experiments = 3
        self.global_blast_radius_limit = 0.3
        
        # Metrics
        self.experiments_total = Counter(
            'chaos_experiments_total',
            'Total number of chaos experiments',
            ['type', 'scope', 'status']
        )
        self.experiment_duration = Histogram(
            'chaos_experiment_duration_seconds',
            'Duration of chaos experiments',
            ['type', 'scope']
        )
        self.active_experiments_gauge = Gauge(
            'chaos_active_experiments',
            'Number of active chaos experiments'
        )
        
        logger.info("Initialized Chaos Monkey with safety mechanisms enabled")

    async def schedule_experiment(self, experiment: ChaosExperiment) -> bool:
        """Schedule a chaos experiment with safety checks"""
        try:
            # Perform safety checks
            if not await self._perform_safety_checks(experiment):
                logger.error(f"Safety checks failed for experiment: {experiment.name}")
                return False
            
            # Check blast radius
            if not await self._check_blast_radius(experiment):
                logger.error(f"Blast radius too large for experiment: {experiment.name}")
                return False
            
            # Check concurrent experiment limits
            if len(self.active_experiments) >= self.max_concurrent_experiments:
                logger.error(f"Too many concurrent experiments running")
                return False
            
            # Schedule the experiment
            experiment.status = ChaosStatus.SCHEDULED
            self.active_experiments[experiment.id] = experiment
            
            # Start experiment execution
            asyncio.create_task(self._execute_experiment(experiment))
            
            logger.info(f"Scheduled chaos experiment: {experiment.name} ({experiment.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule experiment {experiment.name}: {e}")
            return False

    async def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running chaos experiment"""
        if experiment_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[experiment_id]
        experiment.status = ChaosStatus.CANCELLED
        
        # Perform immediate rollback
        await self._rollback_experiment(experiment)
        
        # Move to history
        self._move_to_history(experiment)
        
        logger.info(f"Cancelled chaos experiment: {experiment.name}")
        return True

    async def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a chaos experiment"""
        experiment = None
        
        # Check active experiments
        if experiment_id in self.active_experiments:
            experiment = self.active_experiments[experiment_id]
        else:
            # Check history
            for hist_exp in self.experiment_history:
                if hist_exp.id == experiment_id:
                    experiment = hist_exp
                    break
        
        if not experiment:
            return None
        
        # Get current metrics
        current_metrics = await self._collect_experiment_metrics(experiment)
        
        return {
            'id': experiment.id,
            'name': experiment.name,
            'description': experiment.description,
            'type': experiment.chaos_type.value,
            'scope': experiment.scope.value,
            'status': experiment.status.value,
            'created_at': experiment.created_at.isoformat(),
            'start_time': experiment.start_time.isoformat() if experiment.start_time else None,
            'end_time': experiment.end_time.isoformat() if experiment.end_time else None,
            'duration': (experiment.end_time - experiment.start_time).total_seconds() if experiment.start_time and experiment.end_time else None,
            'targets': [
                {
                    'service_name': target.service_name,
                    'instance_id': target.instance_id,
                    'namespace': target.namespace
                }
                for target in experiment.targets
            ],
            'parameters': {
                'duration': experiment.parameters.duration,
                'intensity': experiment.parameters.intensity,
                'percentage': experiment.parameters.percentage
            },
            'results': experiment.results,
            'current_metrics': current_metrics,
            'safety_status': await self._check_safety_status(experiment)
        }

    async def _execute_experiment(self, experiment: ChaosExperiment):
        """Execute a chaos experiment with monitoring and safety checks"""
        try:
            experiment.status = ChaosStatus.RUNNING
            experiment.start_time = datetime.now()
            
            # Update metrics
            self.active_experiments_gauge.set(len(self.active_experiments))
            
            # Wait for delay if specified
            if experiment.parameters.delay > 0:
                await asyncio.sleep(experiment.parameters.delay)
            
            # Execute chaos based on type
            chaos_task = None
            
            if experiment.chaos_type == ChaosType.NETWORK_LATENCY:
                chaos_task = self._inject_network_latency(experiment)
            elif experiment.chaos_type == ChaosType.NETWORK_PARTITION:
                chaos_task = self._inject_network_partition(experiment)
            elif experiment.chaos_type == ChaosType.CPU_STRESS:
                chaos_task = self._inject_cpu_stress(experiment)
            elif experiment.chaos_type == ChaosType.MEMORY_STRESS:
                chaos_task = self._inject_memory_stress(experiment)
            elif experiment.chaos_type == ChaosType.SERVICE_KILL:
                chaos_task = self._inject_service_kill(experiment)
            elif experiment.chaos_type == ChaosType.DATABASE_FAILURE:
                chaos_task = self._inject_database_failure(experiment)
            else:
                raise ValueError(f"Unsupported chaos type: {experiment.chaos_type}")
            
            # Run chaos injection with monitoring
            monitoring_task = asyncio.create_task(self._monitor_experiment(experiment))
            
            if chaos_task:
                await asyncio.gather(chaos_task, monitoring_task)
            
            # Collect final results
            experiment.results = await self._collect_experiment_results(experiment)
            experiment.status = ChaosStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Experiment execution failed: {e}")
            experiment.status = ChaosStatus.FAILED
            experiment.results['error'] = str(e)
            
            # Perform emergency rollback
            await self._rollback_experiment(experiment)
        
        finally:
            experiment.end_time = datetime.now()
            
            # Update metrics
            duration = (experiment.end_time - experiment.start_time).total_seconds()
            self.experiment_duration.labels(
                type=experiment.chaos_type.value,
                scope=experiment.scope.value
            ).observe(duration)
            
            self.experiments_total.labels(
                type=experiment.chaos_type.value,
                scope=experiment.scope.value,
                status=experiment.status.value
            ).inc()
            
            # Move to history
            self._move_to_history(experiment)

    async def _inject_network_latency(self, experiment: ChaosExperiment):
        """Inject network latency using tc (traffic control)"""
        latency = experiment.parameters.latency_ms
        
        for target in experiment.targets:
            if self.k8s_client:
                # Kubernetes implementation
                await self._k8s_inject_network_latency(target, latency, experiment.parameters.duration)
            else:
                # Docker implementation
                await self._docker_inject_network_latency(target, latency, experiment.parameters.duration)

    async def _inject_network_partition(self, experiment: ChaosExperiment):
        """Inject network partition by blocking traffic"""
        for target in experiment.targets:
            if self.k8s_client:
                await self._k8s_inject_network_partition(target, experiment.parameters.duration)
            else:
                await self._docker_inject_network_partition(target, experiment.parameters.duration)

    async def _inject_cpu_stress(self, experiment: ChaosExperiment):
        """Inject CPU stress using stress-ng"""
        cpu_percent = experiment.parameters.cpu_percent
        
        for target in experiment.targets:
            if self.k8s_client:
                await self._k8s_inject_cpu_stress(target, cpu_percent, experiment.parameters.duration)
            else:
                await self._docker_inject_cpu_stress(target, cpu_percent, experiment.parameters.duration)

    async def _inject_memory_stress(self, experiment: ChaosExperiment):
        """Inject memory stress"""
        memory_percent = experiment.parameters.memory_percent
        
        for target in experiment.targets:
            if self.k8s_client:
                await self._k8s_inject_memory_stress(target, memory_percent, experiment.parameters.duration)
            else:
                await self._docker_inject_memory_stress(target, memory_percent, experiment.parameters.duration)

    async def _inject_service_kill(self, experiment: ChaosExperiment):
        """Kill service instances"""
        for target in experiment.targets:
            if self.k8s_client:
                await self._k8s_kill_service(target)
            else:
                await self._docker_kill_service(target)
            
            # Wait before killing next instance if percentage < 100%
            if experiment.parameters.percentage < 100.0:
                delay = experiment.parameters.duration / len(experiment.targets)
                await asyncio.sleep(delay)

    async def _inject_database_failure(self, experiment: ChaosExperiment):
        """Inject database failures"""
        for target in experiment.targets:
            # Simulate database connection failures
            await self._simulate_database_failure(target, experiment.parameters.duration)

    async def _k8s_inject_network_latency(self, target: ChaosTarget, latency_ms: int, duration: int):
        """Inject network latency in Kubernetes pod"""
        # Create a chaos pod that injects latency
        chaos_pod_spec = {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name': f'chaos-latency-{target.service_name}-{int(time.time())}',
                'namespace': target.namespace,
                'labels': {'chaos-experiment': 'network-latency'}
            },
            'spec': {
                'hostNetwork': True,
                'containers': [{
                    'name': 'chaos-agent',
                    'image': 'nicolaka/netshoot:latest',
                    'command': ['sh', '-c'],
                    'args': [f'''
                        tc qdisc add dev {target.network_interface} root netem delay {latency_ms}ms &&
                        sleep {duration} &&
                        tc qdisc del dev {target.network_interface} root
                    '''],
                    'securityContext': {
                        'privileged': True,
                        'capabilities': {'add': ['NET_ADMIN']}
                    }
                }],
                'restartPolicy': 'Never'
            }
        }
        
        # Create and monitor the chaos pod
        await self._create_and_monitor_chaos_pod(chaos_pod_spec, target.namespace)

    async def _docker_inject_network_latency(self, target: ChaosTarget, latency_ms: int, duration: int):
        """Inject network latency in Docker container"""
        try:
            container = self.docker_client.containers.get(target.instance_id or target.service_name)
            
            # Execute tc command in container
            exec_result = container.exec_run(
                f'tc qdisc add dev {target.network_interface} root netem delay {latency_ms}ms',
                privileged=True
            )
            
            if exec_result.exit_code == 0:
                # Wait for duration
                await asyncio.sleep(duration)
                
                # Remove latency
                container.exec_run(
                    f'tc qdisc del dev {target.network_interface} root',
                    privileged=True
                )
            
        except Exception as e:
            logger.error(f"Failed to inject network latency in Docker: {e}")

    async def _monitor_experiment(self, experiment: ChaosExperiment):
        """Monitor experiment progress and safety metrics"""
        start_time = time.time()
        
        while experiment.status == ChaosStatus.RUNNING:
            # Check safety criteria
            if not await self._check_safety_status(experiment):
                logger.warning(f"Safety violation detected in experiment {experiment.name}")
                experiment.status = ChaosStatus.FAILED
                await self._rollback_experiment(experiment)
                break
            
            # Collect metrics
            metrics = await self._collect_experiment_metrics(experiment)
            experiment.metrics[int(time.time())] = metrics
            
            # Check if experiment duration exceeded
            if time.time() - start_time > experiment.parameters.duration:
                break
            
            await asyncio.sleep(10)  # Check every 10 seconds

    async def _collect_experiment_metrics(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Collect metrics during experiment execution"""
        metrics = {}
        
        try:
            # Query Prometheus for system metrics
            async with aiohttp.ClientSession() as session:
                # CPU usage
                cpu_query = f'avg(cpu_usage_percent{{service="{experiment.targets[0].service_name}"}})'
                cpu_result = await self._query_prometheus(session, cpu_query)
                metrics['cpu_usage'] = cpu_result
                
                # Memory usage
                memory_query = f'avg(memory_usage_percent{{service="{experiment.targets[0].service_name}"}})'
                memory_result = await self._query_prometheus(session, memory_query)
                metrics['memory_usage'] = memory_result
                
                # Request rate
                request_query = f'rate(http_requests_total{{service="{experiment.targets[0].service_name}"}}[5m])'
                request_result = await self._query_prometheus(session, request_query)
                metrics['request_rate'] = request_result
                
                # Error rate
                error_query = f'rate(http_requests_total{{service="{experiment.targets[0].service_name}",status=~"5.."}}[5m])'
                error_result = await self._query_prometheus(session, error_query)
                metrics['error_rate'] = error_result
        
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics

    async def _query_prometheus(self, session: aiohttp.ClientSession, query: str) -> float:
        """Query Prometheus for metrics"""
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            params = {'query': query}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['status'] == 'success' and data['data']['result']:
                        return float(data['data']['result'][0]['value'][1])
                return 0.0
        except Exception:
            return 0.0

    async def _perform_safety_checks(self, experiment: ChaosExperiment) -> bool:
        """Perform comprehensive safety checks before experiment"""
        if not self.safety_enabled:
            return True
        
        # Check if it's business hours (avoid chaos during peak times)
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:  # Business hours
            logger.warning("Chaos experiment scheduled during business hours")
            return False
        
        # Check system health
        system_health = await self._check_system_health()
        if not system_health:
            logger.warning("System health check failed")
            return False
        
        # Check if critical services are healthy
        for target in experiment.targets:
            if not await self._check_service_health(target.service_name):
                logger.warning(f"Target service {target.service_name} is not healthy")
                return False
        
        return True

    async def _check_blast_radius(self, experiment: ChaosExperiment) -> bool:
        """Check if experiment blast radius is within limits"""
        # Calculate affected infrastructure percentage
        total_instances = await self._get_total_service_instances()
        affected_instances = len(experiment.targets)
        
        blast_radius = affected_instances / total_instances if total_instances > 0 else 0
        
        if blast_radius > experiment.max_blast_radius:
            return False
        
        if blast_radius > self.global_blast_radius_limit:
            return False
        
        return True

    async def _check_system_health(self) -> bool:
        """Check overall system health"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                return False
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 80:
                return False
            
            return True
        except Exception:
            return False

    async def _check_service_health(self, service_name: str) -> bool:
        """Check health of a specific service"""
        try:
            # This would integrate with your service discovery
            # For now, return True as placeholder
            return True
        except Exception:
            return False

    async def _get_total_service_instances(self) -> int:
        """Get total number of service instances"""
        try:
            if self.k8s_client:
                pods = self.k8s_core_v1.list_namespaced_pod(namespace="prometheus-ai")
                return len(pods.items)
            else:
                containers = self.docker_client.containers.list()
                return len(containers)
        except Exception:
            return 1  # Fallback

    async def _check_safety_status(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Check current safety status during experiment"""
        return {
            'system_health': await self._check_system_health(),
            'blast_radius_ok': await self._check_blast_radius(experiment),
            'service_health': {
                target.service_name: await self._check_service_health(target.service_name)
                for target in experiment.targets
            }
        }

    async def _rollback_experiment(self, experiment: ChaosExperiment):
        """Rollback chaos experiment immediately"""
        logger.info(f"Rolling back experiment: {experiment.name}")
        
        # Implement rollback based on chaos type
        if experiment.chaos_type in [ChaosType.NETWORK_LATENCY, ChaosType.NETWORK_PARTITION]:
            await self._rollback_network_chaos(experiment)
        elif experiment.chaos_type in [ChaosType.CPU_STRESS, ChaosType.MEMORY_STRESS]:
            await self._rollback_resource_chaos(experiment)
        elif experiment.chaos_type == ChaosType.SERVICE_KILL:
            await self._rollback_service_kill(experiment)

    async def _rollback_network_chaos(self, experiment: ChaosExperiment):
        """Rollback network-related chaos"""
        for target in experiment.targets:
            try:
                if self.k8s_client:
                    # Remove chaos pods
                    pods = self.k8s_core_v1.list_namespaced_pod(
                        namespace=target.namespace,
                        label_selector="chaos-experiment=network-latency"
                    )
                    for pod in pods.items:
                        self.k8s_core_v1.delete_namespaced_pod(
                            name=pod.metadata.name,
                            namespace=target.namespace
                        )
                else:
                    # Remove tc rules from Docker containers
                    container = self.docker_client.containers.get(target.instance_id or target.service_name)
                    container.exec_run(f'tc qdisc del dev {target.network_interface} root', privileged=True)
            except Exception as e:
                logger.error(f"Failed to rollback network chaos for {target.service_name}: {e}")

    async def _rollback_resource_chaos(self, experiment: ChaosExperiment):
        """Rollback resource-related chaos"""
        # Kill stress processes
        for target in experiment.targets:
            try:
                if self.k8s_client:
                    # Remove stress pods
                    pods = self.k8s_core_v1.list_namespaced_pod(
                        namespace=target.namespace,
                        label_selector="chaos-experiment=resource-stress"
                    )
                    for pod in pods.items:
                        self.k8s_core_v1.delete_namespaced_pod(
                            name=pod.metadata.name,
                            namespace=target.namespace
                        )
                else:
                    # Kill stress processes in Docker
                    container = self.docker_client.containers.get(target.instance_id or target.service_name)
                    container.exec_run('pkill -f stress-ng')
            except Exception as e:
                logger.error(f"Failed to rollback resource chaos for {target.service_name}: {e}")

    async def _rollback_service_kill(self, experiment: ChaosExperiment):
        """Rollback service kill by restarting services"""
        for target in experiment.targets:
            try:
                if self.k8s_client:
                    # Scale deployment back up
                    deployment = self.k8s_apps_v1.read_namespaced_deployment(
                        name=target.service_name,
                        namespace=target.namespace
                    )
                    if deployment.spec.replicas == 0:
                        deployment.spec.replicas = 1
                        self.k8s_apps_v1.patch_namespaced_deployment(
                            name=target.service_name,
                            namespace=target.namespace,
                            body=deployment
                        )
                else:
                    # Restart Docker container
                    try:
                        container = self.docker_client.containers.get(target.instance_id or target.service_name)
                        container.restart()
                    except docker.errors.NotFound:
                        # Container was killed, need to recreate
                        logger.warning(f"Container {target.service_name} was killed and needs manual recreation")
            except Exception as e:
                logger.error(f"Failed to rollback service kill for {target.service_name}: {e}")

    async def _collect_experiment_results(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Collect final experiment results"""
        results = {
            'duration': (experiment.end_time - experiment.start_time).total_seconds(),
            'targets_affected': len(experiment.targets),
            'metrics_collected': len(experiment.metrics),
            'success_criteria_met': [],
            'failure_criteria_met': [],
            'recommendations': []
        }
        
        # Analyze metrics to determine success/failure criteria
        if experiment.metrics:
            avg_metrics = self._calculate_average_metrics(experiment.metrics)
            results['average_metrics'] = avg_metrics
            
            # Check success criteria
            for criteria in experiment.success_criteria:
                if self._evaluate_criteria(criteria, avg_metrics):
                    results['success_criteria_met'].append(criteria)
            
            # Check failure criteria
            for criteria in experiment.failure_criteria:
                if self._evaluate_criteria(criteria, avg_metrics):
                    results['failure_criteria_met'].append(criteria)
        
        return results

    def _calculate_average_metrics(self, metrics: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average metrics across experiment duration"""
        if not metrics:
            return {}
        
        metric_sums = {}
        metric_counts = {}
        
        for timestamp, metric_data in metrics.items():
            for metric_name, value in metric_data.items():
                if isinstance(value, (int, float)):
                    metric_sums[metric_name] = metric_sums.get(metric_name, 0) + value
                    metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
        
        return {
            metric_name: metric_sums[metric_name] / metric_counts[metric_name]
            for metric_name in metric_sums
        }

    def _evaluate_criteria(self, criteria: str, metrics: Dict[str, float]) -> bool:
        """Evaluate success/failure criteria against metrics"""
        # Simple criteria evaluation (could be enhanced with more complex logic)
        try:
            # Example: "error_rate < 0.05"
            for metric_name, value in metrics.items():
                if metric_name in criteria:
                    # This is a simplified evaluation - in practice, you'd want a proper expression parser
                    if "< 0.05" in criteria and value < 0.05:
                        return True
                    elif "> 0.1" in criteria and value > 0.1:
                        return True
            return False
        except Exception:
            return False

    def _move_to_history(self, experiment: ChaosExperiment):
        """Move experiment from active to history"""
        if experiment.id in self.active_experiments:
            del self.active_experiments[experiment.id]
        
        self.experiment_history.append(experiment)
        
        # Keep only last 100 experiments in history
        if len(self.experiment_history) > 100:
            self.experiment_history.pop(0)
        
        # Update metrics
        self.active_experiments_gauge.set(len(self.active_experiments))

    async def _create_and_monitor_chaos_pod(self, pod_spec: Dict, namespace: str):
        """Create and monitor a chaos pod in Kubernetes"""
        try:
            # Create the pod
            pod = self.k8s_core_v1.create_namespaced_pod(
                namespace=namespace,
                body=pod_spec
            )
            
            # Wait for pod to complete
            pod_name = pod.metadata.name
            while True:
                pod_status = self.k8s_core_v1.read_namespaced_pod_status(
                    name=pod_name,
                    namespace=namespace
                )
                
                if pod_status.status.phase in ['Succeeded', 'Failed']:
                    break
                
                await asyncio.sleep(5)
            
            # Clean up the pod
            self.k8s_core_v1.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace
            )
            
        except Exception as e:
            logger.error(f"Failed to create/monitor chaos pod: {e}")

# Example usage and experiment definitions
async def setup_chaos_engineering():
    """Example setup for chaos engineering in Prometheus AI Platform"""
    
    # Initialize Chaos Monkey
    chaos_monkey = ChaosMonkey(
        prometheus_url="http://prometheus:9090"
    )
    
    # Define chaos experiments
    
    # Network latency experiment
    network_latency_experiment = ChaosExperiment(
        name="helios-core-network-latency",
        description="Test Helios Core resilience to network latency",
        chaos_type=ChaosType.NETWORK_LATENCY,
        scope=ChaosScope.SINGLE_INSTANCE,
        targets=[
            ChaosTarget(
                service_name="helios-core",
                namespace="prometheus-ai",
                labels={"app": "helios-core"}
            )
        ],
        parameters=ChaosParameters(
            duration=300,  # 5 minutes
            latency_ms=200,
            intensity=0.5
        ),
        success_criteria=[
            "error_rate < 0.05",
            "response_time_p95 < 2000"
        ],
        failure_criteria=[
            "error_rate > 0.2",
            "response_time_p95 > 5000"
        ]
    )
    
    # CPU stress experiment
    cpu_stress_experiment = ChaosExperiment(
        name="volcano-watchtower-cpu-stress",
        description="Test Volcano Watchtower under CPU stress",
        chaos_type=ChaosType.CPU_STRESS,
        scope=ChaosScope.SINGLE_INSTANCE,
        targets=[
            ChaosTarget(
                service_name="volcano-watchtower",
                namespace="prometheus-ai"
            )
        ],
        parameters=ChaosParameters(
            duration=180,  # 3 minutes
            cpu_percent=80.0,
            intensity=0.8
        ),
        success_criteria=[
            "cpu_usage < 90",
            "memory_usage < 80"
        ]
    )
    
    return chaos_monkey, [network_latency_experiment, cpu_stress_experiment]

if __name__ == "__main__":
    async def test_chaos_engineering():
        chaos_monkey, experiments = await setup_chaos_engineering()
        
        # Schedule experiments
        for experiment in experiments:
            success = await chaos_monkey.schedule_experiment(experiment)
            print(f"Scheduled experiment {experiment.name}: {success}")
        
        # Wait and check status
        await asyncio.sleep(10)
        
        for experiment in experiments:
            status = await chaos_monkey.get_experiment_status(experiment.id)
            if status:
                print(f"Experiment status: {json.dumps(status, indent=2)}")
    
    asyncio.run(test_chaos_engineering())