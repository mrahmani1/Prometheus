"""
Service Mesh Observability and Security Policies for Prometheus AI Platform
Comprehensive monitoring, tracing, and security policy enforcement
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import jaeger_client
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram, Gauge, Summary
import hashlib
import jwt

logger = logging.getLogger(__name__)

class SecurityPolicyType(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMITING = "rate_limiting"
    ENCRYPTION = "encryption"
    NETWORK_POLICY = "network_policy"
    DATA_CLASSIFICATION = "data_classification"

class TraceLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    DETAILED = "detailed"
    FULL = "full"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    name: str
    policy_type: SecurityPolicyType
    service_selector: Dict[str, str]
    enabled: bool = True
    
    # Authentication policies
    require_jwt: bool = False
    allowed_issuers: List[str] = field(default_factory=list)
    required_scopes: List[str] = field(default_factory=list)
    
    # Authorization policies
    allowed_roles: List[str] = field(default_factory=list)
    allowed_users: List[str] = field(default_factory=list)
    resource_permissions: Dict[str, List[str]] = field(default_factory=dict)
    
    # Network policies
    allowed_sources: List[str] = field(default_factory=list)
    allowed_destinations: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=list)
    
    # Encryption policies
    require_tls: bool = True
    min_tls_version: str = "1.2"
    cipher_suites: List[str] = field(default_factory=list)
    
    # Data classification
    data_classification: str = "internal"
    retention_policy: str = "30d"
    encryption_at_rest: bool = True

@dataclass
class ObservabilityConfig:
    """Observability configuration for services"""
    service_name: str
    
    # Tracing configuration
    trace_level: TraceLevel = TraceLevel.BASIC
    sample_rate: float = 0.1
    jaeger_endpoint: str = "http://jaeger:14268/api/traces"
    
    # Metrics configuration
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    custom_metrics: List[str] = field(default_factory=list)
    
    # Logging configuration
    log_level: str = "INFO"
    structured_logging: bool = True
    log_sampling_rate: float = 1.0
    
    # Health check configuration
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    liveness_check_path: str = "/live"

@dataclass
class ServiceMeshMetrics:
    """Service mesh metrics collection"""
    service_name: str
    timestamp: datetime
    
    # Request metrics
    request_count: int = 0
    request_rate: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    
    # Latency metrics
    response_time_p50: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    
    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_io: float = 0.0
    
    # Security metrics
    auth_failures: int = 0
    policy_violations: int = 0
    tls_handshake_failures: int = 0

@dataclass
class Alert:
    """Alert definition and state"""
    name: str
    description: str
    severity: AlertSeverity
    service_name: str
    
    # Alert conditions
    metric_name: str
    threshold: float
    comparison: str = ">"  # >, <, >=, <=, ==, !=
    duration: int = 300  # seconds
    
    # Alert state
    active: bool = False
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Notification configuration
    notification_channels: List[str] = field(default_factory=list)
    escalation_policy: Optional[str] = None

class ServiceMeshObservability:
    """
    Comprehensive service mesh observability with distributed tracing,
    metrics collection, security monitoring, and alerting
    """
    
    def __init__(self, 
                 jaeger_endpoint: str = "http://jaeger:14268/api/traces",
                 prometheus_url: str = "http://prometheus:9090"):
        
        self.jaeger_endpoint = jaeger_endpoint
        self.prometheus_url = prometheus_url
        
        # Initialize tracing
        self._setup_tracing()
        
        # Service configurations
        self.observability_configs: Dict[str, ObservabilityConfig] = {}
        self.security_policies: Dict[str, SecurityPolicy] = {}
        
        # Metrics storage
        self.service_metrics: Dict[str, ServiceMeshMetrics] = {}
        self.alerts: Dict[str, Alert] = {}
        
        # Prometheus metrics
        self.mesh_requests_total = Counter(
            'service_mesh_requests_total',
            'Total requests in service mesh',
            ['source_service', 'destination_service', 'method', 'status']
        )
        self.mesh_request_duration = Histogram(
            'service_mesh_request_duration_seconds',
            'Request duration in service mesh',
            ['source_service', 'destination_service', 'method']
        )
        self.mesh_security_violations = Counter(
            'service_mesh_security_violations_total',
            'Security policy violations',
            ['service', 'policy_type', 'violation_type']
        )
        self.mesh_active_connections = Gauge(
            'service_mesh_active_connections',
            'Active connections in service mesh',
            ['source_service', 'destination_service']
        )
        
        logger.info("Initialized Service Mesh Observability")

    def _setup_tracing(self):
        """Initialize distributed tracing with Jaeger"""
        try:
            # Configure Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="jaeger",
                agent_port=6831,
            )
            
            # Set up tracer provider
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)
            
            # Add span processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.tracer = tracer
            logger.info("Initialized distributed tracing with Jaeger")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
            self.tracer = None

    async def add_observability_config(self, config: ObservabilityConfig):
        """Add observability configuration for a service"""
        self.observability_configs[config.service_name] = config
        
        # Initialize metrics for service
        self.service_metrics[config.service_name] = ServiceMeshMetrics(
            service_name=config.service_name,
            timestamp=datetime.now()
        )
        
        logger.info(f"Added observability config for service: {config.service_name}")

    async def add_security_policy(self, policy: SecurityPolicy):
        """Add security policy for service mesh"""
        self.security_policies[policy.name] = policy
        logger.info(f"Added security policy: {policy.name}")

    async def trace_request(self, 
                          source_service: str,
                          destination_service: str,
                          operation_name: str,
                          request_data: Dict[str, Any]) -> Optional[str]:
        """Start distributed trace for a request"""
        if not self.tracer:
            return None
        
        try:
            # Create span
            with self.tracer.start_as_current_span(operation_name) as span:
                # Add span attributes
                span.set_attribute("source.service", source_service)
                span.set_attribute("destination.service", destination_service)
                span.set_attribute("http.method", request_data.get("method", "GET"))
                span.set_attribute("http.url", request_data.get("url", ""))
                span.set_attribute("user.id", request_data.get("user_id", ""))
                
                # Add custom attributes based on geological context
                if "geological_analysis" in request_data:
                    span.set_attribute("geological.analysis_type", request_data["geological_analysis"])
                if "volcano_id" in request_data:
                    span.set_attribute("geological.volcano_id", request_data["volcano_id"])
                
                # Generate trace ID
                trace_id = format(span.get_span_context().trace_id, '032x')
                
                return trace_id
                
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None

    async def record_request_metrics(self,
                                   source_service: str,
                                   destination_service: str,
                                   method: str,
                                   status_code: int,
                                   duration: float,
                                   trace_id: Optional[str] = None):
        """Record request metrics for service mesh"""
        
        # Update Prometheus metrics
        self.mesh_requests_total.labels(
            source_service=source_service,
            destination_service=destination_service,
            method=method,
            status=str(status_code)
        ).inc()
        
        self.mesh_request_duration.labels(
            source_service=source_service,
            destination_service=destination_service,
            method=method
        ).observe(duration)
        
        # Update service metrics
        if destination_service in self.service_metrics:
            metrics = self.service_metrics[destination_service]
            metrics.request_count += 1
            metrics.timestamp = datetime.now()
            
            if status_code >= 400:
                metrics.error_count += 1
            
            # Update error rate (simple moving average)
            metrics.error_rate = metrics.error_count / metrics.request_count if metrics.request_count > 0 else 0
            
            # Update response time percentiles (simplified)
            # In production, you'd use a proper percentile calculation
            if duration > metrics.response_time_p99:
                metrics.response_time_p99 = duration
            if duration > metrics.response_time_p95:
                metrics.response_time_p95 = duration
            metrics.response_time_p50 = (metrics.response_time_p50 + duration) / 2

    async def enforce_security_policy(self,
                                    service_name: str,
                                    request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce security policies for a service request"""
        
        enforcement_result = {
            'allowed': True,
            'violations': [],
            'applied_policies': []
        }
        
        # Find applicable policies
        applicable_policies = []
        for policy in self.security_policies.values():
            if self._policy_applies_to_service(policy, service_name):
                applicable_policies.append(policy)
        
        # Enforce each applicable policy
        for policy in applicable_policies:
            policy_result = await self._enforce_single_policy(policy, request_data)
            
            if not policy_result['allowed']:
                enforcement_result['allowed'] = False
                enforcement_result['violations'].extend(policy_result['violations'])
            
            enforcement_result['applied_policies'].append(policy.name)
        
        # Record security metrics
        if enforcement_result['violations']:
            for violation in enforcement_result['violations']:
                self.mesh_security_violations.labels(
                    service=service_name,
                    policy_type=violation['policy_type'],
                    violation_type=violation['type']
                ).inc()
        
        return enforcement_result

    def _policy_applies_to_service(self, policy: SecurityPolicy, service_name: str) -> bool:
        """Check if security policy applies to a service"""
        if not policy.enabled:
            return False
        
        # Check service selector
        if policy.service_selector:
            # Simple label matching (could be enhanced with more complex selectors)
            if 'service' in policy.service_selector:
                return policy.service_selector['service'] == service_name
        
        return True  # Default: policy applies to all services

    async def _enforce_single_policy(self, 
                                   policy: SecurityPolicy, 
                                   request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce a single security policy"""
        
        result = {
            'allowed': True,
            'violations': []
        }
        
        if policy.policy_type == SecurityPolicyType.AUTHENTICATION:
            auth_result = await self._enforce_authentication_policy(policy, request_data)
            if not auth_result['allowed']:
                result['allowed'] = False
                result['violations'].extend(auth_result['violations'])
        
        elif policy.policy_type == SecurityPolicyType.AUTHORIZATION:
            authz_result = await self._enforce_authorization_policy(policy, request_data)
            if not authz_result['allowed']:
                result['allowed'] = False
                result['violations'].extend(authz_result['violations'])
        
        elif policy.policy_type == SecurityPolicyType.NETWORK_POLICY:
            network_result = await self._enforce_network_policy(policy, request_data)
            if not network_result['allowed']:
                result['allowed'] = False
                result['violations'].extend(network_result['violations'])
        
        elif policy.policy_type == SecurityPolicyType.ENCRYPTION:
            encryption_result = await self._enforce_encryption_policy(policy, request_data)
            if not encryption_result['allowed']:
                result['allowed'] = False
                result['violations'].extend(encryption_result['violations'])
        
        return result

    async def _enforce_authentication_policy(self, 
                                           policy: SecurityPolicy, 
                                           request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce authentication policy"""
        
        result = {'allowed': True, 'violations': []}
        
        if policy.require_jwt:
            auth_header = request_data.get('headers', {}).get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                result['allowed'] = False
                result['violations'].append({
                    'policy_type': 'authentication',
                    'type': 'missing_jwt_token',
                    'message': 'JWT token required but not provided'
                })
                return result
            
            token = auth_header[7:]  # Remove 'Bearer ' prefix
            
            try:
                # Decode JWT (without verification for demo - in production, verify signature)
                decoded_token = jwt.decode(token, options={"verify_signature": False})
                
                # Check issuer
                if policy.allowed_issuers:
                    issuer = decoded_token.get('iss', '')
                    if issuer not in policy.allowed_issuers:
                        result['allowed'] = False
                        result['violations'].append({
                            'policy_type': 'authentication',
                            'type': 'invalid_issuer',
                            'message': f'Token issuer {issuer} not in allowed list'
                        })
                
                # Check scopes
                if policy.required_scopes:
                    token_scopes = decoded_token.get('scope', '').split()
                    missing_scopes = set(policy.required_scopes) - set(token_scopes)
                    if missing_scopes:
                        result['allowed'] = False
                        result['violations'].append({
                            'policy_type': 'authentication',
                            'type': 'insufficient_scopes',
                            'message': f'Missing required scopes: {missing_scopes}'
                        })
                
            except jwt.InvalidTokenError as e:
                result['allowed'] = False
                result['violations'].append({
                    'policy_type': 'authentication',
                    'type': 'invalid_jwt_token',
                    'message': f'Invalid JWT token: {str(e)}'
                })
        
        return result

    async def _enforce_authorization_policy(self, 
                                          policy: SecurityPolicy, 
                                          request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce authorization policy"""
        
        result = {'allowed': True, 'violations': []}
        
        # Extract user information from request
        user_id = request_data.get('user_id', '')
        user_roles = request_data.get('user_roles', [])
        
        # Check allowed users
        if policy.allowed_users and user_id not in policy.allowed_users:
            result['allowed'] = False
            result['violations'].append({
                'policy_type': 'authorization',
                'type': 'user_not_allowed',
                'message': f'User {user_id} not in allowed users list'
            })
        
        # Check allowed roles
        if policy.allowed_roles:
            if not any(role in policy.allowed_roles for role in user_roles):
                result['allowed'] = False
                result['violations'].append({
                    'policy_type': 'authorization',
                    'type': 'insufficient_roles',
                    'message': f'User roles {user_roles} do not match required roles {policy.allowed_roles}'
                })
        
        # Check resource permissions
        if policy.resource_permissions:
            requested_resource = request_data.get('resource', '')
            requested_action = request_data.get('action', 'read')
            
            if requested_resource in policy.resource_permissions:
                allowed_actions = policy.resource_permissions[requested_resource]
                if requested_action not in allowed_actions:
                    result['allowed'] = False
                    result['violations'].append({
                        'policy_type': 'authorization',
                        'type': 'action_not_permitted',
                        'message': f'Action {requested_action} not permitted on resource {requested_resource}'
                    })
        
        return result

    async def _enforce_network_policy(self, 
                                    policy: SecurityPolicy, 
                                    request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce network policy"""
        
        result = {'allowed': True, 'violations': []}
        
        source_ip = request_data.get('source_ip', '')
        destination_port = request_data.get('destination_port', 80)
        
        # Check allowed sources
        if policy.allowed_sources and source_ip:
            if not any(self._ip_in_range(source_ip, allowed) for allowed in policy.allowed_sources):
                result['allowed'] = False
                result['violations'].append({
                    'policy_type': 'network_policy',
                    'type': 'source_not_allowed',
                    'message': f'Source IP {source_ip} not in allowed ranges'
                })
        
        # Check allowed ports
        if policy.allowed_ports and destination_port not in policy.allowed_ports:
            result['allowed'] = False
            result['violations'].append({
                'policy_type': 'network_policy',
                'type': 'port_not_allowed',
                'message': f'Destination port {destination_port} not in allowed ports'
            })
        
        return result

    async def _enforce_encryption_policy(self, 
                                       policy: SecurityPolicy, 
                                       request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce encryption policy"""
        
        result = {'allowed': True, 'violations': []}
        
        # Check TLS requirement
        if policy.require_tls:
            is_tls = request_data.get('is_tls', False)
            if not is_tls:
                result['allowed'] = False
                result['violations'].append({
                    'policy_type': 'encryption',
                    'type': 'tls_required',
                    'message': 'TLS encryption required but not used'
                })
        
        # Check TLS version
        tls_version = request_data.get('tls_version', '')
        if tls_version and policy.min_tls_version:
            if self._compare_tls_versions(tls_version, policy.min_tls_version) < 0:
                result['allowed'] = False
                result['violations'].append({
                    'policy_type': 'encryption',
                    'type': 'tls_version_too_low',
                    'message': f'TLS version {tls_version} below minimum {policy.min_tls_version}'
                })
        
        return result

    def _ip_in_range(self, ip: str, ip_range: str) -> bool:
        """Check if IP is in range (simplified implementation)"""
        # This is a simplified implementation
        # In production, use proper CIDR matching
        if '/' not in ip_range:
            return ip == ip_range
        
        # Basic CIDR check (would need proper implementation)
        network, prefix = ip_range.split('/')
        return ip.startswith(network.rsplit('.', 1)[0])

    def _compare_tls_versions(self, version1: str, version2: str) -> int:
        """Compare TLS versions"""
        version_map = {'1.0': 10, '1.1': 11, '1.2': 12, '1.3': 13}
        v1 = version_map.get(version1, 0)
        v2 = version_map.get(version2, 0)
        return v1 - v2

    async def add_alert(self, alert: Alert):
        """Add alert configuration"""
        self.alerts[alert.name] = alert
        logger.info(f"Added alert: {alert.name}")

    async def check_alerts(self):
        """Check all configured alerts"""
        for alert in self.alerts.values():
            await self._evaluate_alert(alert)

    async def _evaluate_alert(self, alert: Alert):
        """Evaluate a single alert condition"""
        try:
            # Get current metric value
            current_value = await self._get_metric_value(alert.service_name, alert.metric_name)
            
            # Evaluate condition
            condition_met = self._evaluate_condition(current_value, alert.threshold, alert.comparison)
            
            if condition_met and not alert.active:
                # Alert triggered
                alert.active = True
                alert.triggered_at = datetime.now()
                await self._send_alert_notification(alert, "triggered")
                logger.warning(f"Alert triggered: {alert.name}")
                
            elif not condition_met and alert.active:
                # Alert resolved
                alert.active = False
                alert.resolved_at = datetime.now()
                await self._send_alert_notification(alert, "resolved")
                logger.info(f"Alert resolved: {alert.name}")
                
        except Exception as e:
            logger.error(f"Failed to evaluate alert {alert.name}: {e}")

    async def _get_metric_value(self, service_name: str, metric_name: str) -> float:
        """Get current metric value for a service"""
        if service_name in self.service_metrics:
            metrics = self.service_metrics[service_name]
            return getattr(metrics, metric_name, 0.0)
        return 0.0

    def _evaluate_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate alert condition"""
        if comparison == ">":
            return value > threshold
        elif comparison == "<":
            return value < threshold
        elif comparison == ">=":
            return value >= threshold
        elif comparison == "<=":
            return value <= threshold
        elif comparison == "==":
            return value == threshold
        elif comparison == "!=":
            return value != threshold
        return False

    async def _send_alert_notification(self, alert: Alert, action: str):
        """Send alert notification"""
        # This would integrate with notification systems (Slack, PagerDuty, etc.)
        notification = {
            'alert_name': alert.name,
            'service_name': alert.service_name,
            'severity': alert.severity.value,
            'action': action,
            'description': alert.description,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Alert notification: {json.dumps(notification)}")

    async def get_service_topology(self) -> Dict[str, Any]:
        """Get service mesh topology and relationships"""
        topology = {
            'services': [],
            'connections': [],
            'security_policies': [],
            'metrics_summary': {}
        }
        
        # Add services
        for service_name, config in self.observability_configs.items():
            service_info = {
                'name': service_name,
                'trace_level': config.trace_level.value,
                'metrics_enabled': config.metrics_enabled,
                'health_status': 'healthy'  # Would check actual health
            }
            topology['services'].append(service_info)
        
        # Add security policies
        for policy_name, policy in self.security_policies.items():
            policy_info = {
                'name': policy_name,
                'type': policy.policy_type.value,
                'enabled': policy.enabled,
                'service_selector': policy.service_selector
            }
            topology['security_policies'].append(policy_info)
        
        # Add metrics summary
        for service_name, metrics in self.service_metrics.items():
            topology['metrics_summary'][service_name] = {
                'request_count': metrics.request_count,
                'error_rate': metrics.error_rate,
                'response_time_p95': metrics.response_time_p95
            }
        
        return topology

    async def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data"""
        dashboard = {
            'policy_violations': {},
            'authentication_failures': 0,
            'authorization_failures': 0,
            'tls_handshake_failures': 0,
            'active_policies': len([p for p in self.security_policies.values() if p.enabled]),
            'services_monitored': len(self.observability_configs)
        }
        
        # Aggregate security metrics
        for service_name, metrics in self.service_metrics.items():
            dashboard['policy_violations'][service_name] = metrics.policy_violations
            dashboard['authentication_failures'] += metrics.auth_failures
            dashboard['tls_handshake_failures'] += metrics.tls_handshake_failures
        
        return dashboard

# Example usage and configuration
async def setup_service_mesh_observability():
    """Example setup for service mesh observability"""
    
    # Initialize observability
    observability = ServiceMeshObservability()
    
    # Add observability configs for services
    helios_config = ObservabilityConfig(
        service_name="helios-core",
        trace_level=TraceLevel.DETAILED,
        sample_rate=0.2,
        custom_metrics=["geological_analysis_duration", "model_accuracy"]
    )
    
    volcano_config = ObservabilityConfig(
        service_name="volcano-watchtower",
        trace_level=TraceLevel.FULL,
        sample_rate=0.5,
        custom_metrics=["eruption_predictions", "seismic_events_processed"]
    )
    
    await observability.add_observability_config(helios_config)
    await observability.add_observability_config(volcano_config)
    
    # Add security policies
    auth_policy = SecurityPolicy(
        name="geological-services-auth",
        policy_type=SecurityPolicyType.AUTHENTICATION,
        service_selector={"app": "geological"},
        require_jwt=True,
        allowed_issuers=["prometheus-ai-auth"],
        required_scopes=["geological:read", "geological:analyze"]
    )
    
    network_policy = SecurityPolicy(
        name="internal-network-only",
        policy_type=SecurityPolicyType.NETWORK_POLICY,
        service_selector={"tier": "backend"},
        allowed_sources=["10.0.0.0/8", "192.168.0.0/16"]
    )
    
    encryption_policy = SecurityPolicy(
        name="tls-required",
        policy_type=SecurityPolicyType.ENCRYPTION,
        service_selector={},  # Apply to all services
        require_tls=True,
        min_tls_version="1.2"
    )
    
    await observability.add_security_policy(auth_policy)
    await observability.add_security_policy(network_policy)
    await observability.add_security_policy(encryption_policy)
    
    # Add alerts
    error_rate_alert = Alert(
        name="high-error-rate",
        description="Service error rate is too high",
        severity=AlertSeverity.ERROR,
        service_name="helios-core",
        metric_name="error_rate",
        threshold=0.05,
        comparison=">",
        duration=300
    )
    
    response_time_alert = Alert(
        name="high-response-time",
        description="Service response time is too high",
        severity=AlertSeverity.WARNING,
        service_name="volcano-watchtower",
        metric_name="response_time_p95",
        threshold=2.0,
        comparison=">",
        duration=180
    )
    
    await observability.add_alert(error_rate_alert)
    await observability.add_alert(response_time_alert)
    
    return observability

if __name__ == "__main__":
    async def test_observability():
        observability = await setup_service_mesh_observability()
        
        # Simulate request tracing
        trace_id = await observability.trace_request(
            "api-gateway",
            "helios-core",
            "geological_analysis",
            {
                "method": "POST",
                "url": "/api/v1/geological/analysis",
                "user_id": "user123",
                "geological_analysis": "volcanic_risk_assessment",
                "volcano_id": "mount_st_helens"
            }
        )
        
        print(f"Started trace: {trace_id}")
        
        # Record metrics
        await observability.record_request_metrics(
            "api-gateway",
            "helios-core",
            "POST",
            200,
            1.5,
            trace_id
        )
        
        # Test security policy enforcement
        security_result = await observability.enforce_security_policy(
            "helios-core",
            {
                "headers": {"Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."},
                "user_id": "user123",
                "user_roles": ["geologist", "researcher"],
                "source_ip": "192.168.1.100",
                "is_tls": True,
                "tls_version": "1.2"
            }
        )
        
        print(f"Security enforcement result: {json.dumps(security_result, indent=2)}")
        
        # Get topology
        topology = await observability.get_service_topology()
        print(f"Service topology: {json.dumps(topology, indent=2)}")
        
        # Get security dashboard
        security_dashboard = await observability.get_security_dashboard()
        print(f"Security dashboard: {json.dumps(security_dashboard, indent=2)}")
    
    asyncio.run(test_observability())