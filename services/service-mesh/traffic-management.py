"""
Advanced Traffic Management and A/B Testing for Prometheus AI Platform
Intelligent traffic routing, load balancing, and canary deployments
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import hashlib
import random
from prometheus_client import Counter, Histogram, Gauge
import yaml

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"
    GEOGRAPHIC = "geographic"
    RANDOM = "random"

class TrafficSplitType(Enum):
    PERCENTAGE = "percentage"
    HEADER_BASED = "header_based"
    USER_BASED = "user_based"
    GEOGRAPHIC = "geographic"
    TIME_BASED = "time_based"
    FEATURE_FLAG = "feature_flag"

class DeploymentStrategy(Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    A_B_TEST = "a_b_test"
    SHADOW = "shadow"

@dataclass
class ServiceEndpoint:
    """Service endpoint with health and performance metrics"""
    host: str
    port: int
    weight: float = 1.0
    protocol: str = "http"
    
    # Health metrics
    healthy: bool = True
    response_time: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    
    # Metadata
    version: str = "v1"
    region: str = "us-west-2"
    zone: str = "us-west-2a"
    tags: Dict[str, str] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"

@dataclass
class TrafficRule:
    """Traffic routing rule configuration"""
    name: str
    service_name: str
    priority: int = 1
    enabled: bool = True
    
    # Matching criteria
    path_patterns: List[str] = field(default_factory=list)
    header_conditions: Dict[str, str] = field(default_factory=dict)
    query_conditions: Dict[str, str] = field(default_factory=dict)
    
    # Routing configuration
    strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN
    endpoints: List[ServiceEndpoint] = field(default_factory=list)
    
    # Advanced features
    timeout: float = 30.0
    retry_count: int = 3
    circuit_breaker: bool = True
    rate_limit: Optional[int] = None
    
    # Traffic splitting
    traffic_split: Optional[Dict[str, float]] = None
    split_type: TrafficSplitType = TrafficSplitType.PERCENTAGE

@dataclass
class ABTestConfig:
    """A/B test configuration"""
    name: str
    description: str
    service_name: str
    
    # Test variants
    control_version: str = "v1"
    treatment_versions: List[str] = field(default_factory=list)
    
    # Traffic allocation
    traffic_allocation: Dict[str, float] = field(default_factory=dict)
    
    # Test criteria
    success_metrics: List[str] = field(default_factory=list)
    duration_hours: int = 24
    min_sample_size: int = 1000
    
    # Statistical configuration
    confidence_level: float = 0.95
    statistical_power: float = 0.8
    minimum_detectable_effect: float = 0.05
    
    # Status
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "draft"
    results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CanaryDeployment:
    """Canary deployment configuration"""
    name: str
    service_name: str
    
    # Version configuration
    stable_version: str
    canary_version: str
    
    # Traffic progression
    initial_traffic_percent: float = 5.0
    target_traffic_percent: float = 100.0
    increment_percent: float = 10.0
    increment_interval_minutes: int = 15
    
    # Success criteria
    success_criteria: Dict[str, float] = field(default_factory=dict)
    failure_criteria: Dict[str, float] = field(default_factory=dict)
    
    # Rollback configuration
    auto_rollback: bool = True
    rollback_threshold: float = 0.1  # 10% error rate
    
    # Status
    current_traffic_percent: float = 0.0
    status: str = "pending"
    start_time: Optional[datetime] = None
    last_increment_time: Optional[datetime] = None

class AdvancedTrafficManager:
    """
    Advanced traffic management system with intelligent routing,
    A/B testing, and canary deployment capabilities
    """
    
    def __init__(self, consul_client=None, prometheus_url: str = "http://prometheus:9090"):
        self.consul_client = consul_client
        self.prometheus_url = prometheus_url
        
        # Traffic management state
        self.traffic_rules: Dict[str, TrafficRule] = {}
        self.service_endpoints: Dict[str, List[ServiceEndpoint]] = {}
        self.active_ab_tests: Dict[str, ABTestConfig] = {}
        self.active_canaries: Dict[str, CanaryDeployment] = {}
        
        # Load balancing state
        self.round_robin_counters: Dict[str, int] = {}
        self.connection_counts: Dict[str, Dict[str, int]] = {}
        
        # Metrics
        self.requests_total = Counter(
            'traffic_manager_requests_total',
            'Total requests processed by traffic manager',
            ['service', 'version', 'status']
        )
        self.request_duration = Histogram(
            'traffic_manager_request_duration_seconds',
            'Request duration through traffic manager',
            ['service', 'version']
        )
        self.active_connections = Gauge(
            'traffic_manager_active_connections',
            'Active connections per endpoint',
            ['service', 'endpoint']
        )
        
        logger.info("Initialized Advanced Traffic Manager")

    async def add_traffic_rule(self, rule: TrafficRule) -> bool:
        """Add a traffic routing rule"""
        try:
            # Validate rule
            if not await self._validate_traffic_rule(rule):
                return False
            
            # Store rule
            self.traffic_rules[rule.name] = rule
            
            # Initialize service endpoints if not exists
            if rule.service_name not in self.service_endpoints:
                self.service_endpoints[rule.service_name] = rule.endpoints
            
            # Initialize load balancing state
            if rule.service_name not in self.round_robin_counters:
                self.round_robin_counters[rule.service_name] = 0
            
            if rule.service_name not in self.connection_counts:
                self.connection_counts[rule.service_name] = {}
                for endpoint in rule.endpoints:
                    self.connection_counts[rule.service_name][endpoint.url] = 0
            
            logger.info(f"Added traffic rule: {rule.name} for service {rule.service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add traffic rule {rule.name}: {e}")
            return False

    async def route_request(self, 
                          service_name: str, 
                          request_info: Dict[str, Any]) -> Optional[ServiceEndpoint]:
        """Route request to appropriate endpoint based on traffic rules"""
        
        # Find matching traffic rule
        matching_rule = await self._find_matching_rule(service_name, request_info)
        if not matching_rule:
            logger.warning(f"No matching rule found for service: {service_name}")
            return None
        
        # Get healthy endpoints
        healthy_endpoints = [ep for ep in matching_rule.endpoints if ep.healthy]
        if not healthy_endpoints:
            logger.error(f"No healthy endpoints available for service: {service_name}")
            return None
        
        # Apply traffic splitting if configured
        if matching_rule.traffic_split:
            endpoint = await self._apply_traffic_split(matching_rule, request_info, healthy_endpoints)
        else:
            # Use load balancing strategy
            endpoint = await self._select_endpoint_by_strategy(matching_rule, healthy_endpoints, request_info)
        
        if endpoint:
            # Update connection count
            self.connection_counts[service_name][endpoint.url] += 1
            self.active_connections.labels(
                service=service_name,
                endpoint=endpoint.url
            ).set(self.connection_counts[service_name][endpoint.url])
        
        return endpoint

    async def start_ab_test(self, ab_test: ABTestConfig) -> bool:
        """Start an A/B test"""
        try:
            # Validate A/B test configuration
            if not await self._validate_ab_test(ab_test):
                return False
            
            # Set up traffic rules for A/B test
            await self._setup_ab_test_routing(ab_test)
            
            # Start the test
            ab_test.start_time = datetime.now()
            ab_test.status = "running"
            self.active_ab_tests[ab_test.name] = ab_test
            
            # Schedule automatic analysis
            asyncio.create_task(self._monitor_ab_test(ab_test))
            
            logger.info(f"Started A/B test: {ab_test.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start A/B test {ab_test.name}: {e}")
            return False

    async def start_canary_deployment(self, canary: CanaryDeployment) -> bool:
        """Start a canary deployment"""
        try:
            # Validate canary configuration
            if not await self._validate_canary_deployment(canary):
                return False
            
            # Initialize canary traffic
            canary.current_traffic_percent = canary.initial_traffic_percent
            canary.start_time = datetime.now()
            canary.last_increment_time = datetime.now()
            canary.status = "running"
            
            # Set up traffic routing
            await self._setup_canary_routing(canary)
            
            # Store canary deployment
            self.active_canaries[canary.name] = canary
            
            # Schedule automatic progression
            asyncio.create_task(self._manage_canary_progression(canary))
            
            logger.info(f"Started canary deployment: {canary.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start canary deployment {canary.name}: {e}")
            return False

    async def _find_matching_rule(self, service_name: str, request_info: Dict[str, Any]) -> Optional[TrafficRule]:
        """Find the best matching traffic rule for a request"""
        matching_rules = []
        
        for rule in self.traffic_rules.values():
            if rule.service_name != service_name or not rule.enabled:
                continue
            
            # Check path patterns
            if rule.path_patterns:
                path = request_info.get('path', '')
                if not any(self._match_pattern(path, pattern) for pattern in rule.path_patterns):
                    continue
            
            # Check header conditions
            if rule.header_conditions:
                headers = request_info.get('headers', {})
                if not all(headers.get(k) == v for k, v in rule.header_conditions.items()):
                    continue
            
            # Check query conditions
            if rule.query_conditions:
                query_params = request_info.get('query_params', {})
                if not all(query_params.get(k) == v for k, v in rule.query_conditions.items()):
                    continue
            
            matching_rules.append(rule)
        
        # Return highest priority rule
        if matching_rules:
            return max(matching_rules, key=lambda r: r.priority)
        
        return None

    def _match_pattern(self, path: str, pattern: str) -> bool:
        """Simple pattern matching (could be enhanced with regex)"""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return path.startswith(pattern[:-1])
        if pattern.startswith("*"):
            return path.endswith(pattern[1:])
        return path == pattern

    async def _select_endpoint_by_strategy(self, 
                                         rule: TrafficRule, 
                                         endpoints: List[ServiceEndpoint],
                                         request_info: Dict[str, Any]) -> ServiceEndpoint:
        """Select endpoint based on load balancing strategy"""
        
        if rule.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(rule.service_name, endpoints)
        
        elif rule.strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(endpoints)
        
        elif rule.strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(rule.service_name, endpoints)
        
        elif rule.strategy == RoutingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(endpoints)
        
        elif rule.strategy == RoutingStrategy.IP_HASH:
            return self._ip_hash_selection(endpoints, request_info.get('client_ip', ''))
        
        elif rule.strategy == RoutingStrategy.CONSISTENT_HASH:
            return self._consistent_hash_selection(endpoints, request_info.get('user_id', ''))
        
        elif rule.strategy == RoutingStrategy.GEOGRAPHIC:
            return self._geographic_selection(endpoints, request_info.get('client_region', ''))
        
        elif rule.strategy == RoutingStrategy.RANDOM:
            return random.choice(endpoints)
        
        else:
            return endpoints[0]  # Fallback

    def _round_robin_selection(self, service_name: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round robin endpoint selection"""
        counter = self.round_robin_counters[service_name]
        selected = endpoints[counter % len(endpoints)]
        self.round_robin_counters[service_name] = (counter + 1) % len(endpoints)
        return selected

    def _weighted_round_robin_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted round robin selection based on endpoint weights"""
        total_weight = sum(ep.weight for ep in endpoints)
        random_weight = random.uniform(0, total_weight)
        
        current_weight = 0
        for endpoint in endpoints:
            current_weight += endpoint.weight
            if random_weight <= current_weight:
                return endpoint
        
        return endpoints[-1]  # Fallback

    def _least_connections_selection(self, service_name: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select endpoint with least active connections"""
        connection_counts = self.connection_counts.get(service_name, {})
        return min(endpoints, key=lambda ep: connection_counts.get(ep.url, 0))

    def _least_response_time_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select endpoint with lowest response time"""
        return min(endpoints, key=lambda ep: ep.response_time)

    def _ip_hash_selection(self, endpoints: List[ServiceEndpoint], client_ip: str) -> ServiceEndpoint:
        """Select endpoint based on client IP hash"""
        if not client_ip:
            return endpoints[0]
        
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return endpoints[hash_value % len(endpoints)]

    def _consistent_hash_selection(self, endpoints: List[ServiceEndpoint], user_id: str) -> ServiceEndpoint:
        """Select endpoint using consistent hashing"""
        if not user_id:
            return endpoints[0]
        
        hash_value = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        return endpoints[hash_value % len(endpoints)]

    def _geographic_selection(self, endpoints: List[ServiceEndpoint], client_region: str) -> ServiceEndpoint:
        """Select endpoint based on geographic proximity"""
        if not client_region:
            return endpoints[0]
        
        # Find endpoints in same region
        same_region = [ep for ep in endpoints if ep.region == client_region]
        if same_region:
            return random.choice(same_region)
        
        return endpoints[0]  # Fallback

    async def _apply_traffic_split(self, 
                                 rule: TrafficRule, 
                                 request_info: Dict[str, Any],
                                 endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Apply traffic splitting based on configuration"""
        
        if rule.split_type == TrafficSplitType.PERCENTAGE:
            return self._percentage_split(rule.traffic_split, endpoints)
        
        elif rule.split_type == TrafficSplitType.HEADER_BASED:
            return self._header_based_split(rule.traffic_split, request_info, endpoints)
        
        elif rule.split_type == TrafficSplitType.USER_BASED:
            return self._user_based_split(rule.traffic_split, request_info, endpoints)
        
        else:
            return self._percentage_split(rule.traffic_split, endpoints)

    def _percentage_split(self, traffic_split: Dict[str, float], endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Split traffic based on percentage"""
        random_value = random.uniform(0, 100)
        cumulative_percent = 0
        
        for version, percent in traffic_split.items():
            cumulative_percent += percent
            if random_value <= cumulative_percent:
                # Find endpoint with matching version
                for endpoint in endpoints:
                    if endpoint.version == version:
                        return endpoint
        
        return endpoints[0]  # Fallback

    def _header_based_split(self, 
                          traffic_split: Dict[str, float], 
                          request_info: Dict[str, Any],
                          endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Split traffic based on request headers"""
        headers = request_info.get('headers', {})
        
        # Check for version header
        version_header = headers.get('X-Version', headers.get('x-version'))
        if version_header and version_header in traffic_split:
            for endpoint in endpoints:
                if endpoint.version == version_header:
                    return endpoint
        
        # Fallback to percentage split
        return self._percentage_split(traffic_split, endpoints)

    def _user_based_split(self, 
                        traffic_split: Dict[str, float], 
                        request_info: Dict[str, Any],
                        endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Split traffic based on user ID hash"""
        user_id = request_info.get('user_id', '')
        if not user_id:
            return self._percentage_split(traffic_split, endpoints)
        
        # Use consistent hashing for user-based splitting
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        
        cumulative_percent = 0
        for version, percent in traffic_split.items():
            cumulative_percent += percent
            if hash_value < cumulative_percent:
                for endpoint in endpoints:
                    if endpoint.version == version:
                        return endpoint
        
        return endpoints[0]  # Fallback

    async def _setup_ab_test_routing(self, ab_test: ABTestConfig):
        """Set up traffic routing for A/B test"""
        # Create traffic rule for A/B test
        rule_name = f"ab_test_{ab_test.name}"
        
        # Get service endpoints
        service_endpoints = self.service_endpoints.get(ab_test.service_name, [])
        
        # Filter endpoints by versions
        test_endpoints = []
        for endpoint in service_endpoints:
            if endpoint.version in [ab_test.control_version] + ab_test.treatment_versions:
                test_endpoints.append(endpoint)
        
        # Create traffic rule
        traffic_rule = TrafficRule(
            name=rule_name,
            service_name=ab_test.service_name,
            priority=100,  # High priority for A/B tests
            strategy=RoutingStrategy.WEIGHTED_ROUND_ROBIN,
            endpoints=test_endpoints,
            traffic_split=ab_test.traffic_allocation,
            split_type=TrafficSplitType.USER_BASED
        )
        
        await self.add_traffic_rule(traffic_rule)

    async def _setup_canary_routing(self, canary: CanaryDeployment):
        """Set up traffic routing for canary deployment"""
        rule_name = f"canary_{canary.name}"
        
        # Calculate traffic split
        traffic_split = {
            canary.stable_version: 100.0 - canary.current_traffic_percent,
            canary.canary_version: canary.current_traffic_percent
        }
        
        # Get service endpoints
        service_endpoints = self.service_endpoints.get(canary.service_name, [])
        
        # Filter endpoints by versions
        canary_endpoints = []
        for endpoint in service_endpoints:
            if endpoint.version in [canary.stable_version, canary.canary_version]:
                canary_endpoints.append(endpoint)
        
        # Create or update traffic rule
        traffic_rule = TrafficRule(
            name=rule_name,
            service_name=canary.service_name,
            priority=90,  # High priority for canary deployments
            strategy=RoutingStrategy.WEIGHTED_ROUND_ROBIN,
            endpoints=canary_endpoints,
            traffic_split=traffic_split,
            split_type=TrafficSplitType.PERCENTAGE
        )
        
        await self.add_traffic_rule(traffic_rule)

    async def _monitor_ab_test(self, ab_test: ABTestConfig):
        """Monitor A/B test progress and analyze results"""
        while ab_test.status == "running":
            try:
                # Check if test duration exceeded
                if datetime.now() - ab_test.start_time > timedelta(hours=ab_test.duration_hours):
                    await self._complete_ab_test(ab_test)
                    break
                
                # Collect metrics
                metrics = await self._collect_ab_test_metrics(ab_test)
                ab_test.results['current_metrics'] = metrics
                
                # Check for statistical significance
                if await self._check_statistical_significance(ab_test, metrics):
                    await self._complete_ab_test(ab_test)
                    break
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring A/B test {ab_test.name}: {e}")
                await asyncio.sleep(60)

    async def _manage_canary_progression(self, canary: CanaryDeployment):
        """Manage canary deployment progression"""
        while canary.status == "running":
            try:
                # Check if it's time for next increment
                time_since_last_increment = datetime.now() - canary.last_increment_time
                if time_since_last_increment < timedelta(minutes=canary.increment_interval_minutes):
                    await asyncio.sleep(60)  # Check every minute
                    continue
                
                # Collect metrics
                metrics = await self._collect_canary_metrics(canary)
                
                # Check success/failure criteria
                if await self._check_canary_failure_criteria(canary, metrics):
                    await self._rollback_canary(canary)
                    break
                
                if await self._check_canary_success_criteria(canary, metrics):
                    # Increment traffic
                    new_traffic_percent = min(
                        canary.current_traffic_percent + canary.increment_percent,
                        canary.target_traffic_percent
                    )
                    
                    await self._update_canary_traffic(canary, new_traffic_percent)
                    
                    # Check if deployment is complete
                    if canary.current_traffic_percent >= canary.target_traffic_percent:
                        await self._complete_canary(canary)
                        break
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error managing canary deployment {canary.name}: {e}")
                await asyncio.sleep(60)

    async def _collect_ab_test_metrics(self, ab_test: ABTestConfig) -> Dict[str, Any]:
        """Collect metrics for A/B test analysis"""
        metrics = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                for version in [ab_test.control_version] + ab_test.treatment_versions:
                    version_metrics = {}
                    
                    # Request count
                    query = f'sum(rate(http_requests_total{{service="{ab_test.service_name}",version="{version}"}}[5m]))'
                    version_metrics['request_rate'] = await self._query_prometheus(session, query)
                    
                    # Error rate
                    query = f'sum(rate(http_requests_total{{service="{ab_test.service_name}",version="{version}",status=~"5.."}}[5m])) / sum(rate(http_requests_total{{service="{ab_test.service_name}",version="{version}"}}[5m]))'
                    version_metrics['error_rate'] = await self._query_prometheus(session, query)
                    
                    # Response time
                    query = f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{service="{ab_test.service_name}",version="{version}"}}[5m]))'
                    version_metrics['response_time_p95'] = await self._query_prometheus(session, query)
                    
                    metrics[version] = version_metrics
        
        except Exception as e:
            logger.error(f"Failed to collect A/B test metrics: {e}")
        
        return metrics

    async def _collect_canary_metrics(self, canary: CanaryDeployment) -> Dict[str, Any]:
        """Collect metrics for canary deployment analysis"""
        metrics = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                for version in [canary.stable_version, canary.canary_version]:
                    version_metrics = {}
                    
                    # Error rate
                    query = f'sum(rate(http_requests_total{{service="{canary.service_name}",version="{version}",status=~"5.."}}[5m])) / sum(rate(http_requests_total{{service="{canary.service_name}",version="{version}"}}[5m]))'
                    version_metrics['error_rate'] = await self._query_prometheus(session, query)
                    
                    # Response time
                    query = f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{service="{canary.service_name}",version="{version}"}}[5m]))'
                    version_metrics['response_time_p95'] = await self._query_prometheus(session, query)
                    
                    # CPU usage
                    query = f'avg(cpu_usage_percent{{service="{canary.service_name}",version="{version}"}})'
                    version_metrics['cpu_usage'] = await self._query_prometheus(session, query)
                    
                    metrics[version] = version_metrics
        
        except Exception as e:
            logger.error(f"Failed to collect canary metrics: {e}")
        
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

    async def _check_statistical_significance(self, ab_test: ABTestConfig, metrics: Dict[str, Any]) -> bool:
        """Check if A/B test results are statistically significant"""
        # Simplified statistical significance check
        # In practice, you'd want proper statistical tests (t-test, chi-square, etc.)
        
        control_metrics = metrics.get(ab_test.control_version, {})
        
        for treatment_version in ab_test.treatment_versions:
            treatment_metrics = metrics.get(treatment_version, {})
            
            # Check if we have enough samples
            control_requests = control_metrics.get('request_rate', 0) * 300  # 5 minutes * 60 seconds
            treatment_requests = treatment_metrics.get('request_rate', 0) * 300
            
            if control_requests < ab_test.min_sample_size or treatment_requests < ab_test.min_sample_size:
                continue
            
            # Simple effect size check
            for metric in ab_test.success_metrics:
                control_value = control_metrics.get(metric, 0)
                treatment_value = treatment_metrics.get(metric, 0)
                
                if control_value > 0:
                    effect_size = abs(treatment_value - control_value) / control_value
                    if effect_size >= ab_test.minimum_detectable_effect:
                        return True
        
        return False

    async def _check_canary_success_criteria(self, canary: CanaryDeployment, metrics: Dict[str, Any]) -> bool:
        """Check if canary deployment meets success criteria"""
        canary_metrics = metrics.get(canary.canary_version, {})
        
        for metric, threshold in canary.success_criteria.items():
            value = canary_metrics.get(metric, float('inf'))
            if value > threshold:
                return False
        
        return True

    async def _check_canary_failure_criteria(self, canary: CanaryDeployment, metrics: Dict[str, Any]) -> bool:
        """Check if canary deployment meets failure criteria"""
        canary_metrics = metrics.get(canary.canary_version, {})
        
        for metric, threshold in canary.failure_criteria.items():
            value = canary_metrics.get(metric, 0)
            if value > threshold:
                return True
        
        return False

    async def _update_canary_traffic(self, canary: CanaryDeployment, new_traffic_percent: float):
        """Update canary deployment traffic percentage"""
        canary.current_traffic_percent = new_traffic_percent
        canary.last_increment_time = datetime.now()
        
        # Update traffic rule
        await self._setup_canary_routing(canary)
        
        logger.info(f"Updated canary {canary.name} traffic to {new_traffic_percent}%")

    async def _complete_ab_test(self, ab_test: ABTestConfig):
        """Complete A/B test and analyze final results"""
        ab_test.status = "completed"
        ab_test.end_time = datetime.now()
        
        # Perform final analysis
        final_metrics = await self._collect_ab_test_metrics(ab_test)
        ab_test.results['final_metrics'] = final_metrics
        ab_test.results['winner'] = await self._determine_ab_test_winner(ab_test, final_metrics)
        
        logger.info(f"Completed A/B test: {ab_test.name}")

    async def _complete_canary(self, canary: CanaryDeployment):
        """Complete canary deployment successfully"""
        canary.status = "completed"
        logger.info(f"Completed canary deployment: {canary.name}")

    async def _rollback_canary(self, canary: CanaryDeployment):
        """Rollback canary deployment due to failure"""
        canary.status = "rolled_back"
        canary.current_traffic_percent = 0.0
        
        # Update traffic to route all traffic to stable version
        await self._setup_canary_routing(canary)
        
        logger.warning(f"Rolled back canary deployment: {canary.name}")

    async def _determine_ab_test_winner(self, ab_test: ABTestConfig, metrics: Dict[str, Any]) -> str:
        """Determine winner of A/B test based on success metrics"""
        # Simplified winner determination
        # In practice, you'd want proper statistical analysis
        
        best_version = ab_test.control_version
        best_score = 0
        
        for version, version_metrics in metrics.items():
            score = 0
            for metric in ab_test.success_metrics:
                # Higher is better for some metrics, lower for others
                if metric in ['error_rate', 'response_time_p95']:
                    score -= version_metrics.get(metric, 0)
                else:
                    score += version_metrics.get(metric, 0)
            
            if score > best_score:
                best_score = score
                best_version = version
        
        return best_version

    async def _validate_traffic_rule(self, rule: TrafficRule) -> bool:
        """Validate traffic rule configuration"""
        if not rule.name or not rule.service_name:
            return False
        
        if not rule.endpoints:
            return False
        
        # Validate traffic split percentages
        if rule.traffic_split:
            total_percent = sum(rule.traffic_split.values())
            if abs(total_percent - 100.0) > 0.01:  # Allow small floating point errors
                logger.error(f"Traffic split percentages must sum to 100%, got {total_percent}")
                return False
        
        return True

    async def _validate_ab_test(self, ab_test: ABTestConfig) -> bool:
        """Validate A/B test configuration"""
        if not ab_test.name or not ab_test.service_name:
            return False
        
        if not ab_test.treatment_versions:
            return False
        
        # Validate traffic allocation
        total_allocation = sum(ab_test.traffic_allocation.values())
        if abs(total_allocation - 100.0) > 0.01:
            logger.error(f"A/B test traffic allocation must sum to 100%, got {total_allocation}")
            return False
        
        return True

    async def _validate_canary_deployment(self, canary: CanaryDeployment) -> bool:
        """Validate canary deployment configuration"""
        if not canary.name or not canary.service_name:
            return False
        
        if canary.stable_version == canary.canary_version:
            return False
        
        if canary.initial_traffic_percent < 0 or canary.initial_traffic_percent > 100:
            return False
        
        return True

    async def get_traffic_stats(self) -> Dict[str, Any]:
        """Get comprehensive traffic management statistics"""
        stats = {
            'traffic_rules': len(self.traffic_rules),
            'active_ab_tests': len(self.active_ab_tests),
            'active_canaries': len(self.active_canaries),
            'service_endpoints': {
                service: len(endpoints) 
                for service, endpoints in self.service_endpoints.items()
            },
            'rules': [
                {
                    'name': rule.name,
                    'service': rule.service_name,
                    'strategy': rule.strategy.value,
                    'enabled': rule.enabled,
                    'endpoints': len(rule.endpoints)
                }
                for rule in self.traffic_rules.values()
            ],
            'ab_tests': [
                {
                    'name': test.name,
                    'service': test.service_name,
                    'status': test.status,
                    'duration_hours': test.duration_hours
                }
                for test in self.active_ab_tests.values()
            ],
            'canaries': [
                {
                    'name': canary.name,
                    'service': canary.service_name,
                    'status': canary.status,
                    'current_traffic_percent': canary.current_traffic_percent
                }
                for canary in self.active_canaries.values()
            ]
        }
        
        return stats

# Example usage and configuration
async def setup_traffic_management():
    """Example setup for traffic management in Prometheus AI Platform"""
    
    # Initialize traffic manager
    traffic_manager = AdvancedTrafficManager()
    
    # Define service endpoints
    helios_endpoints = [
        ServiceEndpoint("helios-core-v1", 8000, weight=0.8, version="v1"),
        ServiceEndpoint("helios-core-v2", 8000, weight=0.2, version="v2")
    ]
    
    # Add traffic rule
    helios_rule = TrafficRule(
        name="helios-core-routing",
        service_name="helios-core",
        strategy=RoutingStrategy.WEIGHTED_ROUND_ROBIN,
        endpoints=helios_endpoints,
        path_patterns=["/api/v1/geological/*"],
        timeout=60.0,
        retry_count=3
    )
    
    await traffic_manager.add_traffic_rule(helios_rule)
    
    # Set up A/B test
    ab_test = ABTestConfig(
        name="helios-algorithm-test",
        service_name="helios-core",
        control_version="v1",
        treatment_versions=["v2"],
        traffic_allocation={"v1": 70.0, "v2": 30.0},
        success_metrics=["response_time_p95", "error_rate"],
        duration_hours=48,
        min_sample_size=1000
    )
    
    await traffic_manager.start_ab_test(ab_test)
    
    # Set up canary deployment
    canary = CanaryDeployment(
        name="volcano-watchtower-v2",
        service_name="volcano-watchtower",
        stable_version="v1",
        canary_version="v2",
        initial_traffic_percent=5.0,
        increment_percent=10.0,
        increment_interval_minutes=15,
        success_criteria={"error_rate": 0.05, "response_time_p95": 2.0},
        failure_criteria={"error_rate": 0.1}
    )
    
    await traffic_manager.start_canary_deployment(canary)
    
    return traffic_manager

if __name__ == "__main__":
    async def test_traffic_management():
        traffic_manager = await setup_traffic_management()
        
        # Simulate request routing
        request_info = {
            'path': '/api/v1/geological/analysis',
            'headers': {'user-agent': 'test-client'},
            'client_ip': '192.168.1.100',
            'user_id': 'user123'
        }
        
        endpoint = await traffic_manager.route_request("helios-core", request_info)
        if endpoint:
            print(f"Routed to: {endpoint.url} (version: {endpoint.version})")
        
        # Get traffic stats
        stats = await traffic_manager.get_traffic_stats()
        print(f"Traffic stats: {json.dumps(stats, indent=2)}")
    
    asyncio.run(test_traffic_management())