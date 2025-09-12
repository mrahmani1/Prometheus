# Prometheus AI Service Mesh

Advanced service mesh implementation for the Prometheus AI Platform with geological system awareness, intelligent traffic management, and comprehensive observability.

## Overview

The Prometheus AI Service Mesh provides:

- **Advanced Service Discovery**: Consul-based service registry with health monitoring
- **Intelligent Traffic Management**: ML-powered routing with geological workload awareness
- **Circuit Breaker Pattern**: Resilience patterns with bulkhead isolation
- **Distributed Rate Limiting**: Multi-algorithm rate limiting with Redis backend
- **Chaos Engineering**: Geological system-aware chaos testing
- **Comprehensive Observability**: Distributed tracing, metrics, and security policies
- **Security Policies**: Zero-trust architecture with geological data protection

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Service Mesh Control Plane                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │   Consul    │  │   Traffic   │  │   Chaos     │  │  Observ │ │
│  │   Connect   │  │  Manager    │  │  Monkey     │  │ ability │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      Data Plane Services                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ API Gateway │  │ Helios Core │  │  Volcano    │  │  Data   │ │
│  │             │  │             │  │ Watchtower  │  │Ingestion│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Service Discovery (`service_discovery.py`)

Advanced service discovery with Consul integration:

- **Automated Registration**: Services auto-register with health checks
- **Health Monitoring**: Comprehensive health assessment including geological metrics
- **Dependency Mapping**: Automatic service dependency discovery and visualization
- **Geological Capabilities**: Service classification by geological processing capabilities

### 2. Traffic Management (`traffic_management.py`)

Intelligent traffic routing and load balancing:

- **ML-Powered Routing**: Machine learning for optimal endpoint selection
- **Geological Workload Awareness**: Route based on geological processing requirements
- **A/B Testing**: Traffic splitting for geological model comparisons
- **Canary Deployments**: Safe rollout of new geological algorithms

### 3. Circuit Breaker (`circuit_breaker.py`)

Resilience patterns for geological systems:

- **Advanced Circuit Breaker**: State management with geological context
- **Bulkhead Pattern**: Resource isolation for different geological workflows
- **Distributed State**: Redis-backed state sharing across instances
- **Performance Metrics**: Real-time monitoring of geological processing health

### 4. Rate Limiting (`rate_limiter.py`)

Distributed rate limiting with multiple algorithms:

- **Token Bucket**: Burst handling for geological data ingestion
- **Sliding Window**: Smooth rate limiting for real-time monitoring
- **Geological Context**: Rate limits based on geological data sensitivity
- **Adaptive Limiting**: Dynamic adjustment based on system load

### 5. Chaos Engineering (`chaos_engineering.py`)

Geological system resilience testing:

- **Geological Chaos**: Specialized chaos experiments for volcanic/seismic systems
- **Safety Mechanisms**: Comprehensive safety checks for critical geological workflows
- **Data Corruption Testing**: Validate geological data validation and recovery
- **Alert Storm Simulation**: Test volcanic alert processing under load

### 6. Observability (`observability.py`)

Comprehensive monitoring and security:

- **Distributed Tracing**: Jaeger integration with geological context
- **Security Policies**: Zero-trust security with geological data protection
- **Geological Metrics**: Specialized metrics for volcanic and seismic workflows
- **Compliance**: GDPR and data protection for sensitive geological data

## Quick Start

### 1. Deploy Service Mesh

```bash
# Deploy complete service mesh infrastructure
./scripts/deploy-service-mesh.sh

# Check deployment status
kubectl get pods -n prometheus-ai-service-mesh
```

### 2. Register a Service

```python
import aiohttp
import asyncio

async def register_service():
    async with aiohttp.ClientSession() as session:
        service_data = {
            "name": "helios-core",
            "host": "helios-core.prometheus-ai.svc.cluster.local",
            "port": 8000,
            "geological_capabilities": ["seismic_analysis", "volcanic_monitoring"],
            "processing_capacity": 0.9
        }
        
        async with session.post(
            "http://service-mesh-manager:8080/services/register",
            json=service_data
        ) as response:
            result = await response.json()
            print(f"Service registered: {result}")

asyncio.run(register_service())
```

### 3. Create Traffic Split Rule

```python
async def create_traffic_split():
    async with aiohttp.ClientSession() as session:
        split_data = {
            "name": "geological-model-comparison",
            "service_name": "helios-core",
            "split_type": "geological_model_comparison",
            "splits": [
                {"endpoint_id": "helios-v1", "weight": 0.8},
                {"endpoint_id": "helios-v2", "weight": 0.2}
            ]
        }
        
        async with session.post(
            "http://service-mesh-manager:8080/traffic/splits",
            json=split_data
        ) as response:
            result = await response.json()
            print(f"Traffic split created: {result}")

asyncio.run(create_traffic_split())
```

### 4. Run Chaos Experiment

```python
async def run_chaos_experiment():
    async with aiohttp.ClientSession() as session:
        experiment_data = {
            "name": "seismic-data-corruption",
            "description": "Test seismic data validation under corruption",
            "chaos_type": "geological_data_corruption",
            "scope": "geological_workflow",
            "target_services": ["volcano-watchtower"],
            "duration": 300,
            "intensity": 0.1
        }
        
        async with session.post(
            "http://service-mesh-manager:8080/chaos/experiments",
            json=experiment_data
        ) as response:
            result = await response.json()
            print(f"Chaos experiment started: {result}")

asyncio.run(run_chaos_experiment())
```

## Configuration

### Environment Variables

```bash
# Consul Configuration
CONSUL_HTTP_ADDR=consul-server:8500
CONSUL_DATACENTER=prometheus-ai

# Redis Configuration
REDIS_URL=redis://redis:6379

# Observability
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
PROMETHEUS_ENDPOINT=http://prometheus:9090

# Geological System Features
GEOLOGICAL_MONITORING_ENABLED=true
VOLCANIC_ALERT_PROCESSING=true
SEISMIC_DATA_ANALYSIS=true
```

### Service Mesh Configuration

```yaml
service_mesh:
  consul:
    address: "consul-server:8500"
    datacenter: "prometheus-ai"
    
  traffic_management:
    intelligent_routing: true
    geological_workload_aware: true
    ml_routing_enabled: true
    
  circuit_breaker:
    default_failure_threshold: 5
    default_recovery_timeout: 60
    bulkhead_max_concurrent: 100
    
  rate_limiting:
    algorithm: "token_bucket"
    default_requests_per_second: 10.0
    default_burst_size: 20
    
  chaos_engineering:
    enabled: true
    safety_checks_enabled: true
    geological_chaos_enabled: true
    
  observability:
    tracing_enabled: true
    metrics_enabled: true
    geological_metrics_enabled: true
    security_policies_enabled: true
```

## API Reference

### Service Discovery

- `POST /services/register` - Register a service
- `DELETE /services/{service_id}` - Deregister a service
- `GET /services` - Discover services
- `GET /services/{service_id}/health` - Get service health

### Traffic Management

- `POST /traffic/splits` - Create traffic split rule
- `GET /traffic/splits/{rule_id}` - Get traffic split status
- `PUT /traffic/splits/{rule_id}` - Update traffic split rule
- `DELETE /traffic/splits/{rule_id}` - Delete traffic split rule

### Circuit Breakers

- `POST /circuit-breakers/{service_name}` - Create circuit breaker
- `GET /circuit-breakers/{service_name}/status` - Get circuit breaker status
- `PUT /circuit-breakers/{service_name}` - Update circuit breaker configuration

### Chaos Engineering

- `POST /chaos/experiments` - Create chaos experiment
- `GET /chaos/experiments/{experiment_id}` - Get experiment status
- `GET /chaos/experiments` - List all experiments
- `DELETE /chaos/experiments/{experiment_id}` - Stop experiment

### Security Policies

- `POST /security/policies` - Create security policy
- `GET /security/policies` - List security policies
- `PUT /security/policies/{policy_id}` - Update security policy
- `DELETE /security/policies/{policy_id}` - Delete security policy

### Observability

- `GET /observability/metrics` - Get observability metrics
- `GET /observability/traces` - Export traces
- `GET /status` - Get comprehensive service mesh status

## Monitoring and Alerting

### Prometheus Metrics

The service mesh exposes comprehensive metrics:

```
# Service Discovery
service_discovery_registered_services{status="healthy"} 5
service_discovery_health_checks_total{service="helios-core",check_type="http",result="success"} 150

# Traffic Management
traffic_manager_routing_decisions_total{service="helios-core",endpoint="helios-v1",strategy="geological_workload_aware"} 1000
traffic_manager_routing_latency_seconds_bucket{le="0.1"} 950

# Circuit Breakers
circuit_breaker_requests_total{service="volcano-watchtower",state="closed",result="success"} 2000
circuit_breaker_state{service="volcano-watchtower"} 0

# Geological Workflows
geological_workflow_duration_seconds{workflow_type="seismic_analysis",service="helios-core"} 2.5
volcanic_alert_processing_latency_seconds{alert_type="eruption",processing_stage="classification"} 0.15
```

### Grafana Dashboards

Pre-configured dashboards for:

- Service Mesh Overview
- Geological Workflow Performance
- Circuit Breaker Status
- Traffic Management Analytics
- Security Policy Violations
- Chaos Engineering Results

## Security

### Zero-Trust Architecture

- **Service-to-Service Authentication**: mTLS with Consul Connect
- **Authorization Policies**: Fine-grained access control
- **Data Protection**: Encryption for sensitive geological data
- **Audit Logging**: Comprehensive security event logging

### Geological Data Protection

- **Data Classification**: Automatic classification of geological data sensitivity
- **Access Control**: Role-based access for geological workflows
- **Compliance**: GDPR compliance for geological research data
- **Encryption**: End-to-end encryption for sensitive volcanic/seismic data

## Troubleshooting

### Common Issues

1. **Service Registration Failures**
   ```bash
   # Check Consul connectivity
   kubectl exec -n prometheus-ai-service-mesh consul-server-0 -- consul members
   
   # Check service mesh manager logs
   kubectl logs -n prometheus-ai-service-mesh -l app=service-mesh-manager
   ```

2. **Circuit Breaker Issues**
   ```bash
   # Check circuit breaker status
   curl http://service-mesh-manager:8080/circuit-breakers/helios-core/status
   
   # Reset circuit breaker
   kubectl delete pod -n prometheus-ai-service-mesh -l app=service-mesh-manager
   ```

3. **Traffic Routing Problems**
   ```bash
   # Check traffic split rules
   curl http://service-mesh-manager:8080/traffic/splits
   
   # Validate service endpoints
   curl http://service-mesh-manager:8080/services
   ```

### Debug Mode

Enable debug logging:

```bash
kubectl set env deployment/service-mesh-manager -n prometheus-ai-service-mesh LOG_LEVEL=DEBUG
```

## Development

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis and Consul (using Docker)
docker run -d --name redis -p 6379:6379 redis:7-alpine
docker run -d --name consul -p 8500:8500 consul:1.16.1 agent -dev -client=0.0.0.0

# Run service mesh manager
python main.py
```

### Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/integration/ -v

# Run chaos engineering tests
pytest tests/chaos/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.