"""
Advanced Service Discovery and Dependency Mapping for Prometheus AI Platform
Comprehensive service registry with health monitoring and visualization
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import consul.aio
from prometheus_client import Counter, Gauge, Histogram
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class DependencyType(Enum):
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    MESSAGE_QUEUE = "message_queue"

@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    host: str
    port: int
    protocol: str = "http"
    path: str = ""
    
    @property
    def url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"

@dataclass
class ServiceDependency:
    """Service dependency relationship"""
    service_name: str
    dependency_type: DependencyType
    endpoint: ServiceEndpoint
    required: bool = True
    timeout: float = 30.0
    retry_count: int = 3
    circuit_breaker: bool = True
    
    # Health check configuration
    health_check_interval: int = 30
    health_check_timeout: float = 5.0
    health_check_path: str = "/health"

@dataclass
class ServiceMetadata:
    """Extended service metadata"""
    version: str
    environment: str
    region: str
    zone: str
    tags: List[str] = field(default_factory=list)
    
    # Resource requirements
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    
    # Scaling configuration
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    
    # Security configuration
    requires_auth: bool = True
    allowed_origins: List[str] = field(default_factory=list)
    
    # Monitoring configuration
    metrics_path: str = "/metrics"
    log_level: str = "INFO"

@dataclass
class ServiceRegistration:
    """Complete service registration information"""
    name: str
    id: str
    endpoint: ServiceEndpoint
    status: ServiceStatus = ServiceStatus.UNKNOWN
    metadata: Optional[ServiceMetadata] = None
    dependencies: List[ServiceDependency] = field(default_factory=list)
    
    # Health and performance metrics
    last_health_check: Optional[datetime] = None
    response_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    
    # Registration metadata
    registered_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

class AdvancedServiceDiscovery:
    """
    Advanced service discovery with comprehensive health monitoring,
    dependency mapping, and visualization capabilities
    """
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.consul_client = consul.aio.Consul(host=consul_host, port=consul_port)
        self.services: Dict[str, ServiceRegistration] = {}
        self.dependency_graph = nx.DiGraph()
        
        # Health monitoring
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.health_check_interval = 30  # seconds
        
        # Metrics
        self.service_count = Gauge(
            'service_discovery_registered_services',
            'Number of registered services',
            ['status']
        )
        self.health_check_duration = Histogram(
            'service_discovery_health_check_duration_seconds',
            'Duration of health checks',
            ['service']
        )
        self.dependency_check_total = Counter(
            'service_discovery_dependency_checks_total',
            'Total dependency checks performed',
            ['service', 'dependency', 'result']
        )
        
        logger.info(f"Initialized service discovery with Consul at {consul_host}:{consul_port}")

    async def register_service(self, registration: ServiceRegistration) -> bool:
        """Register a service with comprehensive metadata"""
        try:
            # Prepare Consul service registration
            consul_service = {
                'name': registration.name,
                'id': registration.id,
                'address': registration.endpoint.host,
                'port': registration.endpoint.port,
                'tags': registration.metadata.tags if registration.metadata else [],
                'meta': self._prepare_metadata(registration),
                'check': {
                    'http': f"{registration.endpoint.url}/health",
                    'interval': '30s',
                    'timeout': '10s',
                    'deregister_critical_service_after': '5m'
                }
            }
            
            # Register with Consul
            await self.consul_client.agent.service.register(**consul_service)
            
            # Store locally
            self.services[registration.id] = registration
            
            # Update dependency graph
            self._update_dependency_graph(registration)
            
            # Start health monitoring
            await self._start_health_monitoring(registration)
            
            # Update metrics
            self._update_service_metrics()
            
            logger.info(f"Registered service: {registration.name} ({registration.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {registration.name}: {e}")
            return False

    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service"""
        try:
            # Deregister from Consul
            await self.consul_client.agent.service.deregister(service_id)
            
            # Stop health monitoring
            if service_id in self.health_check_tasks:
                self.health_check_tasks[service_id].cancel()
                del self.health_check_tasks[service_id]
            
            # Remove from local storage
            if service_id in self.services:
                service = self.services[service_id]
                del self.services[service_id]
                
                # Update dependency graph
                self.dependency_graph.remove_node(service.name)
            
            # Update metrics
            self._update_service_metrics()
            
            logger.info(f"Deregistered service: {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            return False

    async def discover_services(self, service_name: Optional[str] = None, 
                              tags: Optional[List[str]] = None,
                              healthy_only: bool = True) -> List[ServiceRegistration]:
        """Discover services with filtering options"""
        try:
            # Query Consul for services
            if service_name:
                _, services = await self.consul_client.health.service(
                    service_name, 
                    passing=healthy_only,
                    tag=tags[0] if tags else None
                )
            else:
                _, services = await self.consul_client.health.state('any')
            
            discovered_services = []
            
            for service_data in services:
                if isinstance(service_data, dict) and 'Service' in service_data:
                    service_info = service_data['Service']
                    
                    # Create service registration from Consul data
                    registration = self._create_registration_from_consul(service_info, service_data)
                    
                    # Apply tag filtering
                    if tags and not any(tag in registration.metadata.tags for tag in tags):
                        continue
                    
                    discovered_services.append(registration)
            
            logger.debug(f"Discovered {len(discovered_services)} services")
            return discovered_services
            
        except Exception as e:
            logger.error(f"Failed to discover services: {e}")
            return []

    async def get_service_health(self, service_id: str) -> Dict[str, Any]:
        """Get comprehensive health information for a service"""
        if service_id not in self.services:
            return {'error': 'Service not found'}
        
        service = self.services[service_id]
        
        # Perform health check
        health_status = await self._perform_health_check(service)
        
        # Check dependencies
        dependency_status = await self._check_dependencies(service)
        
        # Get performance metrics
        performance_metrics = await self._get_performance_metrics(service)
        
        return {
            'service_id': service_id,
            'service_name': service.name,
            'status': health_status['status'].value,
            'endpoint': service.endpoint.url,
            'last_health_check': service.last_health_check.isoformat() if service.last_health_check else None,
            'response_time': service.response_time,
            'error_rate': service.error_rate,
            'throughput': service.throughput,
            'health_details': health_status,
            'dependencies': dependency_status,
            'performance': performance_metrics,
            'metadata': {
                'version': service.metadata.version if service.metadata else 'unknown',
                'environment': service.metadata.environment if service.metadata else 'unknown',
                'region': service.metadata.region if service.metadata else 'unknown'
            }
        }

    async def get_dependency_map(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive dependency map"""
        if service_name:
            # Get subgraph for specific service
            if service_name in self.dependency_graph:
                # Get all nodes connected to this service (both directions)
                connected_nodes = set()
                connected_nodes.add(service_name)
                
                # Add predecessors (services that depend on this one)
                connected_nodes.update(self.dependency_graph.predecessors(service_name))
                
                # Add successors (services this one depends on)
                connected_nodes.update(self.dependency_graph.successors(service_name))
                
                subgraph = self.dependency_graph.subgraph(connected_nodes)
            else:
                subgraph = nx.DiGraph()
        else:
            subgraph = self.dependency_graph
        
        # Generate dependency analysis
        dependency_analysis = {
            'nodes': [],
            'edges': [],
            'metrics': {
                'total_services': subgraph.number_of_nodes(),
                'total_dependencies': subgraph.number_of_edges(),
                'strongly_connected_components': len(list(nx.strongly_connected_components(subgraph))),
                'weakly_connected_components': len(list(nx.weakly_connected_components(subgraph)))
            },
            'critical_paths': [],
            'circular_dependencies': []
        }
        
        # Add nodes with metadata
        for node in subgraph.nodes():
            service = self._find_service_by_name(node)
            node_data = {
                'name': node,
                'status': service.status.value if service else 'unknown',
                'endpoint': service.endpoint.url if service else 'unknown',
                'in_degree': subgraph.in_degree(node),
                'out_degree': subgraph.out_degree(node),
                'metadata': {
                    'version': service.metadata.version if service and service.metadata else 'unknown',
                    'environment': service.metadata.environment if service and service.metadata else 'unknown'
                }
            }
            dependency_analysis['nodes'].append(node_data)
        
        # Add edges with dependency information
        for source, target in subgraph.edges():
            edge_data = {
                'source': source,
                'target': target,
                'dependency_type': 'unknown',  # Would need to be stored in graph
                'required': True  # Would need to be stored in graph
            }
            dependency_analysis['edges'].append(edge_data)
        
        # Find critical paths (longest paths)
        try:
            if nx.is_directed_acyclic_graph(subgraph):
                # Find longest paths in DAG
                for source in subgraph.nodes():
                    if subgraph.in_degree(source) == 0:  # Source nodes
                        for target in subgraph.nodes():
                            if subgraph.out_degree(target) == 0:  # Sink nodes
                                try:
                                    path = nx.shortest_path(subgraph, source, target)
                                    if len(path) > 2:  # Only include non-trivial paths
                                        dependency_analysis['critical_paths'].append({
                                            'path': path,
                                            'length': len(path) - 1
                                        })
                                except nx.NetworkXNoPath:
                                    continue
        except:
            pass
        
        # Find circular dependencies
        try:
            cycles = list(nx.simple_cycles(subgraph))
            for cycle in cycles:
                dependency_analysis['circular_dependencies'].append({
                    'cycle': cycle,
                    'length': len(cycle)
                })
        except:
            pass
        
        return dependency_analysis

    async def generate_dependency_visualization(self, service_name: Optional[str] = None) -> str:
        """Generate visual dependency map as base64 encoded image"""
        try:
            dependency_map = await self.get_dependency_map(service_name)
            
            # Create matplotlib figure
            plt.figure(figsize=(12, 8))
            
            # Create networkx graph from dependency map
            G = nx.DiGraph()
            
            # Add nodes
            for node in dependency_map['nodes']:
                G.add_node(node['name'], **node)
            
            # Add edges
            for edge in dependency_map['edges']:
                G.add_edge(edge['source'], edge['target'])
            
            # Generate layout
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, k=3, iterations=50)
                
                # Draw nodes with status-based colors
                node_colors = []
                for node in G.nodes():
                    node_data = G.nodes[node]
                    status = node_data.get('status', 'unknown')
                    if status == 'healthy':
                        node_colors.append('green')
                    elif status == 'unhealthy':
                        node_colors.append('red')
                    elif status == 'warning':
                        node_colors.append('orange')
                    else:
                        node_colors.append('gray')
                
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                     node_size=1000, alpha=0.7)
                nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
                nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                     arrows=True, arrowsize=20, alpha=0.6)
                
                plt.title(f"Service Dependency Map{' - ' + service_name if service_name else ''}")
                plt.axis('off')
            else:
                plt.text(0.5, 0.5, 'No services found', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes, fontsize=16)
                plt.title("Service Dependency Map - No Data")
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Failed to generate dependency visualization: {e}")
            return ""

    async def _perform_health_check(self, service: ServiceRegistration) -> Dict[str, Any]:
        """Perform comprehensive health check on a service"""
        start_time = time.time()
        
        try:
            health_url = f"{service.endpoint.url}/health"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(health_url) as response:
                    duration = time.time() - start_time
                    
                    if response.status == 200:
                        health_data = await response.json()
                        status = ServiceStatus.HEALTHY
                    else:
                        health_data = {'error': f'HTTP {response.status}'}
                        status = ServiceStatus.UNHEALTHY
                    
                    # Update service metrics
                    service.last_health_check = datetime.now()
                    service.response_time = duration
                    service.status = status
                    
                    # Record metrics
                    self.health_check_duration.labels(service=service.name).observe(duration)
                    
                    return {
                        'status': status,
                        'response_time': duration,
                        'health_data': health_data,
                        'timestamp': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            duration = time.time() - start_time
            service.last_health_check = datetime.now()
            service.response_time = duration
            service.status = ServiceStatus.UNHEALTHY
            
            return {
                'status': ServiceStatus.UNHEALTHY,
                'response_time': duration,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def _check_dependencies(self, service: ServiceRegistration) -> List[Dict[str, Any]]:
        """Check health of service dependencies"""
        dependency_results = []
        
        for dependency in service.dependencies:
            start_time = time.time()
            
            try:
                health_url = f"{dependency.endpoint.url}{dependency.health_check_path}"
                
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=dependency.health_check_timeout)
                ) as session:
                    async with session.get(health_url) as response:
                        duration = time.time() - start_time
                        
                        result = {
                            'service_name': dependency.service_name,
                            'dependency_type': dependency.dependency_type.value,
                            'required': dependency.required,
                            'status': 'healthy' if response.status == 200 else 'unhealthy',
                            'response_time': duration,
                            'endpoint': dependency.endpoint.url
                        }
                        
                        # Record metrics
                        self.dependency_check_total.labels(
                            service=service.name,
                            dependency=dependency.service_name,
                            result=result['status']
                        ).inc()
                        
                        dependency_results.append(result)
                        
            except Exception as e:
                duration = time.time() - start_time
                
                result = {
                    'service_name': dependency.service_name,
                    'dependency_type': dependency.dependency_type.value,
                    'required': dependency.required,
                    'status': 'error',
                    'response_time': duration,
                    'error': str(e),
                    'endpoint': dependency.endpoint.url
                }
                
                # Record metrics
                self.dependency_check_total.labels(
                    service=service.name,
                    dependency=dependency.service_name,
                    result='error'
                ).inc()
                
                dependency_results.append(result)
        
        return dependency_results

    async def _get_performance_metrics(self, service: ServiceRegistration) -> Dict[str, Any]:
        """Get performance metrics for a service"""
        try:
            metrics_url = f"{service.endpoint.url}/metrics"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(metrics_url) as response:
                    if response.status == 200:
                        metrics_text = await response.text()
                        
                        # Parse basic Prometheus metrics
                        metrics = self._parse_prometheus_metrics(metrics_text)
                        
                        return {
                            'cpu_usage': metrics.get('cpu_usage_percent', 0),
                            'memory_usage': metrics.get('memory_usage_bytes', 0),
                            'request_rate': metrics.get('http_requests_per_second', 0),
                            'error_rate': metrics.get('http_error_rate', 0),
                            'response_time_p95': metrics.get('http_request_duration_p95', 0)
                        }
                    else:
                        return {'error': f'HTTP {response.status}'}
                        
        except Exception as e:
            return {'error': str(e)}

    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, float]:
        """Basic Prometheus metrics parser"""
        metrics = {}
        
        for line in metrics_text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            
            try:
                parts = line.split(' ')
                if len(parts) >= 2:
                    metric_name = parts[0]
                    metric_value = float(parts[1])
                    metrics[metric_name] = metric_value
            except (ValueError, IndexError):
                continue
        
        return metrics

    def _prepare_metadata(self, registration: ServiceRegistration) -> Dict[str, str]:
        """Prepare metadata for Consul registration"""
        metadata = {}
        
        if registration.metadata:
            metadata.update({
                'version': registration.metadata.version,
                'environment': registration.metadata.environment,
                'region': registration.metadata.region,
                'zone': registration.metadata.zone,
                'cpu_limit': registration.metadata.cpu_limit or '',
                'memory_limit': registration.metadata.memory_limit or '',
                'min_instances': str(registration.metadata.min_instances),
                'max_instances': str(registration.metadata.max_instances),
                'requires_auth': str(registration.metadata.requires_auth),
                'metrics_path': registration.metadata.metrics_path,
                'log_level': registration.metadata.log_level
            })
        
        # Add dependency information
        if registration.dependencies:
            dependency_names = [dep.service_name for dep in registration.dependencies]
            metadata['dependencies'] = ','.join(dependency_names)
        
        return metadata

    def _create_registration_from_consul(self, service_info: Dict, health_data: Dict) -> ServiceRegistration:
        """Create ServiceRegistration from Consul service data"""
        endpoint = ServiceEndpoint(
            host=service_info.get('Address', 'localhost'),
            port=service_info.get('Port', 80),
            protocol='http'
        )
        
        # Parse metadata
        meta = service_info.get('Meta', {})
        metadata = ServiceMetadata(
            version=meta.get('version', 'unknown'),
            environment=meta.get('environment', 'unknown'),
            region=meta.get('region', 'unknown'),
            zone=meta.get('zone', 'unknown'),
            tags=service_info.get('Tags', [])
        )
        
        # Determine status from health checks
        status = ServiceStatus.HEALTHY
        if 'Checks' in health_data:
            for check in health_data['Checks']:
                if check.get('Status') != 'passing':
                    status = ServiceStatus.UNHEALTHY
                    break
        
        return ServiceRegistration(
            name=service_info.get('Service', 'unknown'),
            id=service_info.get('ID', 'unknown'),
            endpoint=endpoint,
            status=status,
            metadata=metadata
        )

    def _update_dependency_graph(self, service: ServiceRegistration):
        """Update the dependency graph with service information"""
        # Add service node
        self.dependency_graph.add_node(service.name, service_id=service.id)
        
        # Add dependency edges
        for dependency in service.dependencies:
            self.dependency_graph.add_edge(
                service.name,
                dependency.service_name,
                dependency_type=dependency.dependency_type.value,
                required=dependency.required
            )

    def _find_service_by_name(self, service_name: str) -> Optional[ServiceRegistration]:
        """Find service registration by name"""
        for service in self.services.values():
            if service.name == service_name:
                return service
        return None

    async def _start_health_monitoring(self, service: ServiceRegistration):
        """Start continuous health monitoring for a service"""
        async def health_monitor():
            while True:
                try:
                    await self._perform_health_check(service)
                    await asyncio.sleep(self.health_check_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health monitoring error for {service.name}: {e}")
                    await asyncio.sleep(self.health_check_interval)
        
        # Cancel existing task if any
        if service.id in self.health_check_tasks:
            self.health_check_tasks[service.id].cancel()
        
        # Start new monitoring task
        self.health_check_tasks[service.id] = asyncio.create_task(health_monitor())

    def _update_service_metrics(self):
        """Update Prometheus metrics for service counts"""
        status_counts = {}
        for service in self.services.values():
            status = service.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Reset all gauges
        for status in ServiceStatus:
            self.service_count.labels(status=status.value).set(0)
        
        # Set current counts
        for status, count in status_counts.items():
            self.service_count.labels(status=status).set(count)

# Example usage and setup
async def setup_service_discovery():
    """Example setup for service discovery in Prometheus AI Platform"""
    
    # Initialize service discovery
    discovery = AdvancedServiceDiscovery()
    
    # Register Helios Core service
    helios_metadata = ServiceMetadata(
        version="1.0.0",
        environment="production",
        region="us-west-2",
        zone="us-west-2a",
        tags=["ai", "ml", "geological"],
        cpu_limit="2000m",
        memory_limit="4Gi",
        min_instances=2,
        max_instances=10,
        requires_auth=True
    )
    
    helios_dependencies = [
        ServiceDependency(
            service_name="postgres",
            dependency_type=DependencyType.DATABASE,
            endpoint=ServiceEndpoint("postgres", 5432, "postgresql"),
            required=True
        ),
        ServiceDependency(
            service_name="redis",
            dependency_type=DependencyType.CACHE,
            endpoint=ServiceEndpoint("redis", 6379, "redis"),
            required=False
        )
    ]
    
    helios_registration = ServiceRegistration(
        name="helios-core",
        id="helios-core-001",
        endpoint=ServiceEndpoint("helios-core", 8000, "http"),
        metadata=helios_metadata,
        dependencies=helios_dependencies
    )
    
    await discovery.register_service(helios_registration)
    
    return discovery

if __name__ == "__main__":
    async def test_service_discovery():
        discovery = await setup_service_discovery()
        
        # Wait a bit for health checks
        await asyncio.sleep(5)
        
        # Get service health
        health = await discovery.get_service_health("helios-core-001")
        print(f"Service health: {json.dumps(health, indent=2)}")
        
        # Get dependency map
        dep_map = await discovery.get_dependency_map("helios-core")
        print(f"Dependency map: {json.dumps(dep_map, indent=2)}")
        
        # Generate visualization
        viz = await discovery.generate_dependency_visualization()
        if viz:
            print(f"Generated visualization (base64 length: {len(viz)})")
    
    asyncio.run(test_service_discovery())