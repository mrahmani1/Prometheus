"""
Automated Database Discovery and Service Registration

Automatically discovers database services in the environment and registers them
with the configuration manager and service discovery system.
"""

import asyncio
import logging
import socket
import subprocess
import json
import yaml
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import os
import re

from database_config_manager import (
    DatabaseConfigManager,
    DatabaseType,
    Environment,
    DatabaseConfig,
    DatabaseCredentials,
    CredentialSource
)

# Network scanning
try:
    import nmap
    NMAP_AVAILABLE = True
except ImportError:
    NMAP_AVAILABLE = False

# Docker integration
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# Kubernetes integration
try:
    from kubernetes import client, config as k8s_config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False


@dataclass
class DiscoveredService:
    """Represents a discovered database service"""
    name: str
    type: DatabaseType
    host: str
    port: int
    database: Optional[str] = None
    version: Optional[str] = None
    status: str = "unknown"
    discovery_method: str = "unknown"
    metadata: Dict[str, Any] = None
    discovered_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.discovered_at is None:
            self.discovered_at = datetime.utcnow()


class DatabaseDiscoveryService:
    """
    Automated Database Discovery and Registration Service
    
    Features:
    - Network scanning for database services
    - Docker container discovery
    - Kubernetes service discovery
    - Automatic configuration generation
    - Service health monitoring
    - Dynamic registration/deregistration
    """
    
    def __init__(self, 
                 config_manager: DatabaseConfigManager,
                 discovery_config: Optional[Dict[str, Any]] = None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__ + ".DatabaseDiscoveryService")
        
        # Discovery configuration
        self.discovery_config = discovery_config or {}
        self.scan_interval = self.discovery_config.get('scan_interval', 300)  # 5 minutes
        self.network_ranges = self.discovery_config.get('network_ranges', ['127.0.0.1/32'])
        self.port_ranges = self.discovery_config.get('port_ranges', {
            DatabaseType.POSTGRESQL: [5432, 5433, 5434],
            DatabaseType.REDIS: [6379, 6380, 6381],
            DatabaseType.INFLUXDB: [8086, 8087, 8088],
            DatabaseType.NEO4J: [7687, 7688, 7689],
            DatabaseType.MONGODB: [27017, 27018, 27019],
            DatabaseType.ELASTICSEARCH: [9200, 9201, 9202],
            DatabaseType.KAFKA: [9092, 9093, 9094]
        })
        
        # Discovery methods
        self.discovery_methods = {
            'network_scan': self.discovery_config.get('network_scan', True),
            'docker_discovery': self.discovery_config.get('docker_discovery', True),
            'kubernetes_discovery': self.discovery_config.get('kubernetes_discovery', True),
            'consul_discovery': self.discovery_config.get('consul_discovery', True)
        }
        
        # Discovered services cache
        self.discovered_services: Dict[str, DiscoveredService] = {}
        self.last_discovery_run: Optional[datetime] = None
        
        # External clients
        self.docker_client = None
        self.k8s_client = None
        self.nmap_scanner = None
        
        # Running state
        self.is_running = False
        self.discovery_task: Optional[asyncio.Task] = None
        
        self.logger.info("Database Discovery Service initialized")
    
    async def initialize(self) -> bool:
        """Initialize discovery service and external clients"""
        try:
            self.logger.info("Initializing Database Discovery Service")
            
            # Initialize Docker client
            if self.discovery_methods['docker_discovery'] and DOCKER_AVAILABLE:
                try:
                    self.docker_client = docker.from_env()
                    self.docker_client.ping()
                    self.logger.info("Docker client initialized")
                except Exception as e:
                    self.logger.warning(f"Docker client initialization failed: {e}")
                    self.docker_client = None
            
            # Initialize Kubernetes client
            if self.discovery_methods['kubernetes_discovery'] and KUBERNETES_AVAILABLE:
                try:
                    # Try in-cluster config first, then local config
                    try:
                        k8s_config.load_incluster_config()
                    except k8s_config.ConfigException:
                        k8s_config.load_kube_config()
                    
                    self.k8s_client = client.CoreV1Api()
                    self.logger.info("Kubernetes client initialized")
                    
                except Exception as e:
                    self.logger.warning(f"Kubernetes client initialization failed: {e}")
                    self.k8s_client = None
            
            # Initialize nmap scanner
            if self.discovery_methods['network_scan'] and NMAP_AVAILABLE:
                try:
                    self.nmap_scanner = nmap.PortScanner()
                    self.logger.info("Nmap scanner initialized")
                except Exception as e:
                    self.logger.warning(f"Nmap scanner initialization failed: {e}")
                    self.nmap_scanner = None
            
            self.logger.info("Database Discovery Service initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Discovery service initialization failed: {e}")
            return False
    
    async def start_discovery(self):
        """Start continuous discovery process"""
        if self.is_running:
            self.logger.warning("Discovery service is already running")
            return
        
        self.is_running = True
        self.discovery_task = asyncio.create_task(self._discovery_loop())
        self.logger.info("Started database discovery service")
    
    async def stop_discovery(self):
        """Stop continuous discovery process"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.discovery_task:
            self.discovery_task.cancel()
            try:
                await self.discovery_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped database discovery service")
    
    async def _discovery_loop(self):
        """Main discovery loop"""
        while self.is_running:
            try:
                await self.run_discovery()
                await asyncio.sleep(self.scan_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def run_discovery(self) -> Dict[str, List[DiscoveredService]]:
        """Run complete discovery process"""
        self.logger.info("Starting database discovery scan")
        
        discovery_results = {
            'network_scan': [],
            'docker_discovery': [],
            'kubernetes_discovery': [],
            'consul_discovery': []
        }
        
        # Network scanning
        if self.discovery_methods['network_scan']:
            discovery_results['network_scan'] = await self._network_discovery()
        
        # Docker discovery
        if self.discovery_methods['docker_discovery']:
            discovery_results['docker_discovery'] = await self._docker_discovery()
        
        # Kubernetes discovery
        if self.discovery_methods['kubernetes_discovery']:
            discovery_results['kubernetes_discovery'] = await self._kubernetes_discovery()
        
        # Consul discovery
        if self.discovery_methods['consul_discovery']:
            discovery_results['consul_discovery'] = await self._consul_discovery()
        
        # Process and register discovered services
        await self._process_discovery_results(discovery_results)
        
        self.last_discovery_run = datetime.utcnow()
        
        total_discovered = sum(len(services) for services in discovery_results.values())
        self.logger.info(f"Discovery scan complete: {total_discovered} services found")
        
        return discovery_results
    
    async def _network_discovery(self) -> List[DiscoveredService]:
        """Discover database services via network scanning"""
        if not self.nmap_scanner:
            return []
        
        discovered = []
        
        for network_range in self.network_ranges:
            try:
                self.logger.debug(f"Scanning network range: {network_range}")
                
                # Build port list for scanning
                all_ports = []
                for db_type, ports in self.port_ranges.items():
                    all_ports.extend(ports)
                
                port_list = ','.join(map(str, set(all_ports)))
                
                # Perform network scan
                scan_result = self.nmap_scanner.scan(
                    hosts=network_range,
                    ports=port_list,
                    arguments='-sT -T4 --open'
                )
                
                # Process scan results
                for host in scan_result['scan']:
                    host_info = scan_result['scan'][host]
                    
                    if host_info['status']['state'] != 'up':
                        continue
                    
                    for port_info in host_info.get('tcp', {}).values():
                        if port_info['state'] != 'open':
                            continue
                        
                        port = int(port_info['name'])
                        service_name = port_info.get('name', 'unknown')
                        
                        # Determine database type from port
                        db_type = self._identify_database_type_by_port(port)
                        if not db_type:
                            continue
                        
                        # Create discovered service
                        service = DiscoveredService(
                            name=f"{db_type.value}_{host}_{port}",
                            type=db_type,
                            host=host,
                            port=port,
                            status="detected",
                            discovery_method="network_scan",
                            metadata={
                                'service_name': service_name,
                                'scan_result': port_info
                            }
                        )
                        
                        # Test connection to verify service
                        if await self._verify_service_connection(service):
                            discovered.append(service)
                
            except Exception as e:
                self.logger.error(f"Network scan failed for {network_range}: {e}")
        
        self.logger.info(f"Network discovery found {len(discovered)} services")
        return discovered
    
    async def _docker_discovery(self) -> List[DiscoveredService]:
        """Discover database services in Docker containers"""
        if not self.docker_client:
            return []
        
        discovered = []
        
        try:
            containers = self.docker_client.containers.list(all=True)
            
            for container in containers:
                try:
                    # Check if container is a database service
                    db_type = self._identify_database_type_by_image(container.image.tags)
                    if not db_type:
                        continue
                    
                    # Get container network information
                    container_info = container.attrs
                    network_settings = container_info.get('NetworkSettings', {})
                    
                    # Extract port mappings
                    port_bindings = network_settings.get('Ports', {})
                    
                    for container_port, host_bindings in port_bindings.items():
                        if not host_bindings:
                            continue
                        
                        port_num = int(container_port.split('/')[0])
                        
                        for binding in host_bindings:
                            host_ip = binding.get('HostIp', '127.0.0.1')
                            host_port = int(binding.get('HostPort', port_num))
                            
                            # Create discovered service
                            service = DiscoveredService(
                                name=f"{container.name}_{db_type.value}",
                                type=db_type,
                                host=host_ip if host_ip != '0.0.0.0' else '127.0.0.1',
                                port=host_port,
                                status=container.status,
                                discovery_method="docker_discovery",
                                metadata={
                                    'container_id': container.id,
                                    'container_name': container.name,
                                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                                    'container_port': port_num
                                }
                            )
                            
                            # Only add running containers or test connection
                            if container.status == 'running':
                                if await self._verify_service_connection(service):
                                    discovered.append(service)
                            else:
                                discovered.append(service)
                
                except Exception as e:
                    self.logger.warning(f"Failed to process container {container.name}: {e}")
        
        except Exception as e:
            self.logger.error(f"Docker discovery failed: {e}")
        
        self.logger.info(f"Docker discovery found {len(discovered)} services")
        return discovered
    
    async def _kubernetes_discovery(self) -> List[DiscoveredService]:
        """Discover database services in Kubernetes"""
        if not self.k8s_client:
            return []
        
        discovered = []
        
        try:
            # Get all services in all namespaces
            services = self.k8s_client.list_service_for_all_namespaces()
            
            for service in services.items:
                try:
                    # Check if service is a database service
                    db_type = self._identify_database_type_by_labels(service.metadata.labels or {})
                    if not db_type:
                        db_type = self._identify_database_type_by_name(service.metadata.name)
                    
                    if not db_type:
                        continue
                    
                    # Extract service information
                    for port in service.spec.ports or []:
                        service_name = f"{service.metadata.name}_{service.metadata.namespace}"
                        
                        # Determine host (use cluster IP or external IP)
                        host = service.spec.cluster_ip
                        if service.spec.external_i_ps:
                            host = service.spec.external_i_ps[0]
                        elif service.status.load_balancer and service.status.load_balancer.ingress:
                            ingress = service.status.load_balancer.ingress[0]
                            host = ingress.ip or ingress.hostname
                        
                        if not host or host == 'None':
                            continue
                        
                        # Create discovered service
                        discovered_service = DiscoveredService(
                            name=service_name,
                            type=db_type,
                            host=host,
                            port=port.port,
                            status="active",
                            discovery_method="kubernetes_discovery",
                            metadata={
                                'namespace': service.metadata.namespace,
                                'service_name': service.metadata.name,
                                'labels': service.metadata.labels or {},
                                'annotations': service.metadata.annotations or {},
                                'port_name': port.name,
                                'target_port': port.target_port
                            }
                        )
                        
                        discovered.append(discovered_service)
                
                except Exception as e:
                    self.logger.warning(f"Failed to process K8s service {service.metadata.name}: {e}")
        
        except Exception as e:
            self.logger.error(f"Kubernetes discovery failed: {e}")
        
        self.logger.info(f"Kubernetes discovery found {len(discovered)} services")
        return discovered
    
    async def _consul_discovery(self) -> List[DiscoveredService]:
        """Discover database services via Consul"""
        if not self.config_manager.consul_client:
            return []
        
        discovered = []
        
        try:
            # Get all services from Consul
            services = self.config_manager.consul_client.catalog.services()[1]
            
            for service_name, tags in services.items():
                try:
                    # Check if service is a database service
                    db_type = self._identify_database_type_by_tags(tags)
                    if not db_type:
                        continue
                    
                    # Get service details
                    service_info = self.config_manager.consul_client.catalog.service(service_name)[1]
                    
                    for instance in service_info:
                        discovered_service = DiscoveredService(
                            name=service_name,
                            type=db_type,
                            host=instance['ServiceAddress'] or instance['Address'],
                            port=instance['ServicePort'],
                            status="registered",
                            discovery_method="consul_discovery",
                            metadata={
                                'consul_service_id': instance['ServiceID'],
                                'consul_node': instance['Node'],
                                'consul_tags': instance['ServiceTags'] or [],
                                'consul_meta': instance['ServiceMeta'] or {}
                            }
                        )
                        
                        discovered.append(discovered_service)
                
                except Exception as e:
                    self.logger.warning(f"Failed to process Consul service {service_name}: {e}")
        
        except Exception as e:
            self.logger.error(f"Consul discovery failed: {e}")
        
        self.logger.info(f"Consul discovery found {len(discovered)} services")
        return discovered
    
    def _identify_database_type_by_port(self, port: int) -> Optional[DatabaseType]:
        """Identify database type by port number"""
        for db_type, ports in self.port_ranges.items():
            if port in ports:
                return db_type
        return None
    
    def _identify_database_type_by_image(self, image_tags: List[str]) -> Optional[DatabaseType]:
        """Identify database type by Docker image tags"""
        for tag in image_tags:
            tag_lower = tag.lower()
            
            if 'postgres' in tag_lower:
                return DatabaseType.POSTGRESQL
            elif 'redis' in tag_lower:
                return DatabaseType.REDIS
            elif 'influx' in tag_lower:
                return DatabaseType.INFLUXDB
            elif 'neo4j' in tag_lower:
                return DatabaseType.NEO4J
            elif 'mongo' in tag_lower:
                return DatabaseType.MONGODB
            elif 'elastic' in tag_lower:
                return DatabaseType.ELASTICSEARCH
            elif 'kafka' in tag_lower:
                return DatabaseType.KAFKA
            elif 'timescale' in tag_lower:
                return DatabaseType.TIMESCALEDB
        
        return None
    
    def _identify_database_type_by_labels(self, labels: Dict[str, str]) -> Optional[DatabaseType]:
        """Identify database type by Kubernetes labels"""
        # Check common label patterns
        app_label = labels.get('app', '').lower()
        component_label = labels.get('component', '').lower()
        
        for label_value in [app_label, component_label]:
            if 'postgres' in label_value:
                return DatabaseType.POSTGRESQL
            elif 'redis' in label_value:
                return DatabaseType.REDIS
            elif 'influx' in label_value:
                return DatabaseType.INFLUXDB
            elif 'neo4j' in label_value:
                return DatabaseType.NEO4J
            elif 'mongo' in label_value:
                return DatabaseType.MONGODB
            elif 'elastic' in label_value:
                return DatabaseType.ELASTICSEARCH
            elif 'kafka' in label_value:
                return DatabaseType.KAFKA
            elif 'timescale' in label_value:
                return DatabaseType.TIMESCALEDB
        
        return None
    
    def _identify_database_type_by_name(self, name: str) -> Optional[DatabaseType]:
        """Identify database type by service name"""
        name_lower = name.lower()
        
        if 'postgres' in name_lower:
            return DatabaseType.POSTGRESQL
        elif 'redis' in name_lower:
            return DatabaseType.REDIS
        elif 'influx' in name_lower:
            return DatabaseType.INFLUXDB
        elif 'neo4j' in name_lower:
            return DatabaseType.NEO4J
        elif 'mongo' in name_lower:
            return DatabaseType.MONGODB
        elif 'elastic' in name_lower:
            return DatabaseType.ELASTICSEARCH
        elif 'kafka' in name_lower:
            return DatabaseType.KAFKA
        elif 'timescale' in name_lower:
            return DatabaseType.TIMESCALEDB
        
        return None
    
    def _identify_database_type_by_tags(self, tags: List[str]) -> Optional[DatabaseType]:
        """Identify database type by Consul tags"""
        for tag in tags:
            if tag.startswith('type:'):
                db_type_str = tag.split(':', 1)[1]
                try:
                    return DatabaseType(db_type_str)
                except ValueError:
                    continue
        
        return None
    
    async def _verify_service_connection(self, service: DiscoveredService) -> bool:
        """Verify that a discovered service is actually accessible"""
        try:
            # Create temporary configuration for testing
            temp_config = DatabaseConfig(
                name=f"temp_{service.name}",
                type=service.type,
                host=service.host,
                port=service.port,
                database=service.database,
                timeout=5  # Short timeout for discovery
            )
            
            # Test connection
            success, _ = await self.config_manager._test_config_connection(temp_config)
            return success
            
        except Exception as e:
            self.logger.debug(f"Connection verification failed for {service.name}: {e}")
            return False
    
    async def _process_discovery_results(self, results: Dict[str, List[DiscoveredService]]):
        """Process discovery results and register new services"""
        all_discovered = []
        for method_results in results.values():
            all_discovered.extend(method_results)
        
        # Deduplicate services by host:port
        unique_services = {}
        for service in all_discovered:
            key = f"{service.host}:{service.port}"
            if key not in unique_services or service.status == "running":
                unique_services[key] = service
        
        # Register new services
        for service in unique_services.values():
            await self._register_discovered_service(service)
        
        # Update discovered services cache
        self.discovered_services = {s.name: s for s in unique_services.values()}
    
    async def _register_discovered_service(self, service: DiscoveredService):
        """Register a discovered service with the configuration manager"""
        try:
            # Check if service is already registered
            if service.name in self.config_manager.configs:
                self.logger.debug(f"Service {service.name} already registered")
                return
            
            # Create database configuration
            config = await self.config_manager.create_database_config(
                name=service.name,
                db_type=service.type,
                host=service.host,
                port=service.port,
                database=service.database,
                tags={
                    'discovered': 'true',
                    'discovery_method': service.discovery_method,
                    'auto_registered': 'true'
                },
                metadata={
                    'discovered_at': service.discovered_at.isoformat(),
                    'discovery_metadata': service.metadata
                }
            )
            
            self.logger.info(f"Registered discovered service: {service.name} ({service.type.value})")
            
        except Exception as e:
            self.logger.error(f"Failed to register discovered service {service.name}: {e}")
    
    async def get_discovery_status(self) -> Dict[str, Any]:
        """Get current discovery service status"""
        return {
            'is_running': self.is_running,
            'last_discovery_run': self.last_discovery_run.isoformat() if self.last_discovery_run else None,
            'discovered_services_count': len(self.discovered_services),
            'discovery_methods': self.discovery_methods,
            'scan_interval': self.scan_interval,
            'clients_available': {
                'docker': self.docker_client is not None,
                'kubernetes': self.k8s_client is not None,
                'nmap': self.nmap_scanner is not None,
                'consul': self.config_manager.consul_client is not None
            }
        }
    
    async def get_discovered_services(self) -> List[Dict[str, Any]]:
        """Get list of all discovered services"""
        return [
            {
                'name': service.name,
                'type': service.type.value,
                'host': service.host,
                'port': service.port,
                'database': service.database,
                'status': service.status,
                'discovery_method': service.discovery_method,
                'discovered_at': service.discovered_at.isoformat(),
                'metadata': service.metadata
            }
            for service in self.discovered_services.values()
        ]