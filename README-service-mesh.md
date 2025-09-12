# Service Mesh Implementation for Prometheus AI Platform

## Overview

This implementation provides a comprehensive service mesh solution for the Prometheus AI Platform using Consul Connect with advanced features including circuit breakers, distributed rate limiting, chaos engineering, traffic management, and comprehensive observability.

## Architecture Components

### Core Service Mesh Components

1. **Consul Connect Service Mesh**
   - Multi-node Consul cluster with service discovery
   - Automatic service registration and health checking
   - mTLS encryption for service-to-service communication
   - Service intentions for fine-grained access control

2. **Circuit Breaker Service**
   - Advanced circuit breaker implementation with sliding window metrics
   - Bulkhead pattern for resource isolation
   - Automatic failure detection and recovery
   - Comprehensive metrics and monitoring

3. **Distributed Rate Limiter**
   - Multiple rate limiting algorithms (token bucket, sliding window, etc.)
   - Redis-backed distributed coordination
   - Hierarchical rate limiting rules
   - Real-time metrics and alerting

4. **Traffic Management**
   - Intelligent load balancing strategies
   - A/B testing and canary deployment support
   - Traffic splitting and routing rules
   - Geographic and user-based routing

5. **Service Discovery**
   - Automated service registration with Consul
   - Health monitoring and dependency mapping
   - Visual dependency graph generation
   - Service topology visualization

6. **Chaos Engineering**
   - Intelligent failure injection (network, CPU, memory, service kills)
   - Safety mechanisms and blast radius controls
   - Automated experiment management
   - Comprehensive result analysis

7. **Observability and Security**
   - Distributed tracing with Jaeger integration
   - Security policy enforcement
   - Real-time metrics collection
   - Automated alerting and notification

## Features Implemented

### Circuit Breaker Features
- **Advanced State Management**: Closed, Open, Half-Open states with intelligent transitions
- **Sliding Window Metrics**: Real-time failure rate and slow call detection
- **Bulkhead Pattern**: Concurrent request limiting with semaphores
- **Comprehensive Monitoring**: Prometheus metrics and health dashboards
- **Cluster Coordination**: Redis-backed state persistence for distributed deployments

### Rate Limiting Features
- **Multiple Algorithms**: Token bucket, sliding window log, sliding window counter, fixed window, leaky bucket
- **Distributed Coordination**: Redis-backed rate limit state sharing
- **Hierarchical Rules**: Per-user, per-IP, per-service, and global rate limits
- **Real-time Monitoring**: Live usage statistics and violation tracking
- **Dynamic Configuration**: Runtime rule updates and adjustments

### Traffic Management Features
- **Intelligent Load Balancing**: Round robin, weighted, least connections, geographic routing
- **A/B Testing**: Statistical significance testing with automated winner determination
- **Canary Deployments**: Automated traffic progression with safety checks
- **Traffic Splitting**: Percentage, header-based, and user-based routing
- **Performance Monitoring**: Real-time metrics and automated rollback

### Service Discovery Features
- **Automated Registration**: Consul integration with health monitoring
- **Dependency Mapping**: Visual service relationship graphs
- **Health Monitoring**: Comprehensive service and dependency health checks
- **Performance Metrics**: Response time, error rate, and throughput tracking
- **Topology Visualization**: Interactive service mesh topology maps

### Chaos Engineering Features
- **Intelligent Failure Injection**: Network latency, partitions, resource stress, service kills
- **Safety Mechanisms**: Business hours protection, blast radius limits, health checks
- **Automated Experiments**: Scheduled chaos with monitoring and rollback
- **Comprehensive Analysis**: Statistical analysis and recommendation generation
- **Kubernetes Integration**: Native pod and deployment manipulation

### Observability Features
- **Distributed Tracing**: Jaeger integration with geological context awareness
- **Security Policy Enforcement**: Authentication, authorization, encryption, network policies
- **Real-time Monitoring**: Service mesh metrics and performance tracking
- **Automated Alerting**: Configurable alerts with notification channels
- **Security Dashboard**: Policy violations and security metrics visualization

## Deployment

### Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3.x
- Redis instance
- Prometheus and Grafana (optional but recommended)

### Quick Start

1. **Deploy the service mesh:**
   ```bash
   ./scripts/deploy-service-mesh.sh
   ```

2. **Validate deployment:**
   ```bash
   ./scripts/deploy-service-mesh.sh validate
   ```

3. **Access Consul UI:**
   ```bash
   kubectl port-forward -n prometheus-ai-mesh svc/consul-ui 8500:8500
   # Open http://localhost:8500
   ```

### Manual Deployment Steps

1. **Create namespace and secrets:**
   ```bash
   kubectl create namespace prometheus-ai-mesh
   kubectl create secret generic consul-gossip-key \
     --namespace=prometheus-ai-mesh \
     --from-literal=key=$(openssl rand -base64 32)
   ```

2. **Deploy Consul using Helm:**
   ```bash
   helm repo add hashicorp https://helm.releases.hashicorp.com
   helm install consul hashicorp/consul \
     --namespace=prometheus-ai-mesh \
     --set global.datacenter=prometheus-ai \
     --set server.replicas=3 \
     --set connectInject.enabled=true \
     --set ui.enabled=true
   ```

3. **Deploy service mesh components:**
   ```bash
   kubectl apply -f kubernetes/service-mesh-deployment.yaml
   ```

## Configuration

### Circuit Breaker Configuration

```python
from services.service_mesh.circuit_breaker import CircuitBreakerConfig, AdvancedCircuitBreaker

config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60,
    timeout=30.0,
    max_concurrent_calls=100,
    error_rate_threshold=0.5,
    slow_call_duration_threshold=10.0
)

circuit_breaker = AdvancedCircuitBreaker("helios-core", config)
```

### Rate Limiting Configuration

```python
from services.service_mesh.rate_limiter import RateLimitRule, RateLimitAlgorithm, RateLimitScope

rule = RateLimitRule(
    name="api_rate_limit",
    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
    scope=RateLimitScope.PER_USER,
    limit=100,  # requests per minute
    window_size=60,
    burst_limit=150
)
```

### Traffic Management Configuration

```python
from services.service_mesh.traffic_management import ABTestConfig, CanaryDeployment

# A/B Test
ab_test = ABTestConfig(
    name="algorithm_test",
    service_name="helios-core",
    control_version="v1",
    treatment_versions=["v2"],
    traffic_allocation={"v1": 70.0, "v2": 30.0},
    duration_hours=48
)

# Canary Deployment
canary = CanaryDeployment(
    name="new_feature",
    service_name="volcano-watchtower",
    stable_version="v1",
    canary_version="v2",
    initial_traffic_percent=5.0,
    increment_percent=10.0
)
```

### Security Policy Configuration

```python
from services.service_mesh.observability import SecurityPolicy, SecurityPolicyType

auth_policy = SecurityPolicy(
    name="jwt_authentication",
    policy_type=SecurityPolicyType.AUTHENTICATION,
    service_selector={"app": "geological"},
    require_jwt=True,
    allowed_issuers=["prometheus-ai-auth"],
    required_scopes=["geological:read"]
)
```

## Monitoring and Observability

### Metrics Available

1. **Circuit Breaker Metrics:**
   - `circuit_breaker_calls_total` - Total calls through circuit breaker
   - `circuit_breaker_call_duration_seconds` - Call duration histogram
   - `circuit_breaker_state` - Current circuit breaker state
   - `circuit_breaker_concurrent_calls` - Active concurrent calls

2. **Rate Limiting Metrics:**
   - `rate_limiter_requests_total` - Total rate limit checks
   - `rate_limiter_check_duration_seconds` - Rate limit check duration
   - `rate_limiter_current_usage` - Current usage per rule

3. **Traffic Management Metrics:**
   - `traffic_manager_requests_total` - Requests processed by traffic manager
   - `traffic_manager_request_duration_seconds` - Request duration
   - `traffic_manager_active_connections` - Active connections per endpoint

4. **Service Mesh Metrics:**
   - `service_mesh_requests_total` - Total service mesh requests
   - `service_mesh_request_duration_seconds` - Request duration
   - `service_mesh_security_violations_total` - Security violations
   - `service_mesh_active_connections` - Active mesh connections

### Dashboards

The deployment includes Grafana dashboards for:
- Service mesh overview
- Circuit breaker status
- Rate limiting statistics
- Traffic management metrics
- Security policy violations
- Chaos engineering results

### Alerting

Pre-configured alerts include:
- High error rates
- Circuit breaker state changes
- Rate limit violations
- Security policy violations
- Service health degradation

## API Usage Examples

### Circuit Breaker Usage

```python
import asyncio
from services.service_mesh.circuit_breaker import circuit_breaker, CircuitBreakerConfig

@circuit_breaker("external_api", CircuitBreakerConfig(failure_threshold=3))
async def call_external_api():
    # Your API call here
    pass

# Manual usage
async def main():
    try:
        result = await call_external_api()
        print(f"API call successful: {result}")
    except CircuitBreakerOpenException:
        print("Circuit breaker is open - service unavailable")
```

### Rate Limiting Usage

```python
from services.service_mesh.rate_limiter import DistributedRateLimiter

rate_limiter = DistributedRateLimiter(redis_client)

async def handle_request(user_id, request_info):
    result = await rate_limiter.check_rate_limit(
        user_id, 
        rule_names=["per_user_limit"],
        metadata=request_info
    )
    
    if not result.allowed:
        raise HTTPException(429, "Rate limit exceeded")
    
    # Process request
    return await process_request(request_info)
```

### Service Discovery Usage

```python
from services.service_mesh.service_discovery import AdvancedServiceDiscovery

discovery = AdvancedServiceDiscovery()

# Register service
registration = ServiceRegistration(
    name="helios-core",
    id="helios-core-001",
    endpoint=ServiceEndpoint("helios-core", 8000),
    metadata=ServiceMetadata(version="1.0.0", environment="production")
)

await discovery.register_service(registration)

# Discover services
services = await discovery.discover_services("helios-core", healthy_only=True)
```

## Troubleshooting

### Common Issues

1. **Consul cluster not forming:**
   - Check network policies and firewall rules
   - Verify gossip encryption key is consistent
   - Check DNS resolution between pods

2. **Circuit breaker not working:**
   - Verify Redis connectivity
   - Check service health endpoints
   - Review circuit breaker configuration

3. **Rate limiting not enforcing:**
   - Confirm Redis is accessible
   - Check rate limit rule configuration
   - Verify request metadata extraction

4. **Service discovery issues:**
   - Check Consul agent connectivity
   - Verify service registration format
   - Review health check configuration

### Debug Commands

```bash
# Check Consul cluster status
kubectl exec -n prometheus-ai-mesh consul-server-0 -- consul members

# View service registrations
kubectl exec -n prometheus-ai-mesh consul-server-0 -- consul catalog services

# Check Connect status
kubectl exec -n prometheus-ai-mesh consul-server-0 -- consul connect ca get-config

# View service mesh logs
kubectl logs -n prometheus-ai-mesh -l app=circuit-breaker-service
kubectl logs -n prometheus-ai-mesh -l app=rate-limiter-service
kubectl logs -n prometheus-ai-mesh -l app=traffic-manager-service
```

## Security Considerations

1. **mTLS Encryption:** All service-to-service communication is encrypted
2. **Service Intentions:** Fine-grained access control between services
3. **Network Policies:** Kubernetes network policies restrict traffic
4. **Secret Management:** Consul gossip keys and certificates are stored as secrets
5. **RBAC:** Proper Kubernetes RBAC for service accounts
6. **Security Policies:** Comprehensive security policy enforcement

## Performance Tuning

### Circuit Breaker Tuning
- Adjust failure thresholds based on service SLAs
- Configure appropriate timeout values
- Set bulkhead limits based on resource capacity

### Rate Limiting Tuning
- Choose appropriate algorithms for use cases
- Configure Redis for high availability
- Set reasonable rate limits to prevent abuse

### Traffic Management Tuning
- Configure load balancing strategies for optimal performance
- Set appropriate health check intervals
- Tune A/B test sample sizes for statistical significance

## Integration with Prometheus AI Platform

The service mesh integrates seamlessly with the Prometheus AI Platform services:

1. **Helios Core Service:** Circuit breaker protection for AI model inference
2. **Volcano Watchtower:** Rate limiting for real-time monitoring endpoints
3. **Data Ingestion:** Traffic management for high-volume data processing
4. **API Gateway:** Comprehensive security policy enforcement

## Future Enhancements

1. **Multi-cluster Support:** Cross-cluster service mesh federation
2. **Advanced Security:** Integration with external identity providers
3. **ML-based Optimization:** AI-driven traffic management and failure prediction
4. **Enhanced Observability:** Custom geological domain metrics and dashboards
5. **Automated Scaling:** Integration with HPA and VPA for intelligent scaling

## Support and Maintenance

For support and maintenance:
1. Monitor service mesh health dashboards
2. Review circuit breaker and rate limiting metrics regularly
3. Update Consul and Envoy versions as needed
4. Perform regular chaos engineering experiments
5. Review and update security policies based on threat landscape