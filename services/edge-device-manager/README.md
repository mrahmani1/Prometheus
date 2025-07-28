# Edge Device Manager

The Edge Device Manager is a core component of the Prometheus AI platform that handles registration, configuration, health monitoring, model deployment, and data synchronization for distributed edge computing nodes.

## Overview

This service provides comprehensive management capabilities for edge devices deployed in remote locations for geological monitoring and AI model execution. It supports various device types including seismic stations, weather stations, cameras, and other IoT sensors.

## Features

### 🔧 Device Registration and Configuration Management
- Automatic device registration with capability detection
- Dynamic configuration updates
- Regional device limits and management
- Authentication token generation and management

### 🚀 Remote Model Deployment
- Automated model deployment to compatible edge devices
- Resource requirement validation
- Deployment status tracking and health monitoring
- Rollback capabilities for failed deployments

### 💓 Health Monitoring and Diagnostics
- Real-time device health monitoring
- System metrics collection (CPU, memory, storage, temperature)
- Automated alerting for critical issues
- Comprehensive diagnostics and troubleshooting

### 🔄 Data Synchronization
- Bidirectional data sync between edge and cloud
- Configurable sync schedules and data types
- Bandwidth management and optimization
- Conflict resolution and retry mechanisms

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Edge Device Manager                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Device Manager  │  │ Model Deployer  │  │ Health       │ │
│  │                 │  │                 │  │ Monitor      │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Sync Manager    │  │ API Gateway     │  │ Storage      │ │
│  │                 │  │                 │  │ Layer        │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

### Device Registration
- `POST /api/v1/devices/register` - Register new edge device
- `GET /api/v1/devices/{device_id}` - Get device information
- `GET /api/v1/devices/` - List devices with filters
- `POST /api/v1/devices/{device_id}/heartbeat` - Update device heartbeat
- `PUT /api/v1/devices/{device_id}/config` - Update device configuration
- `DELETE /api/v1/devices/{device_id}` - Remove device

### Model Deployment
- `POST /api/v1/models/deploy` - Deploy model to devices
- `GET /api/v1/models/deployments/{deployment_id}` - Get deployment status
- `GET /api/v1/models/deployments` - List deployments
- `POST /api/v1/models/deployments/{deployment_id}/devices/{device_id}/stop` - Stop deployment
- `DELETE /api/v1/models/deployments/{deployment_id}/devices/{device_id}` - Remove deployment

### Health Monitoring
- `GET /api/v1/health/summary` - Get health summary
- `GET /api/v1/health/devices/{device_id}` - Get device health
- `POST /api/v1/health/devices/{device_id}/check` - Perform health check
- `GET /api/v1/health/devices/{device_id}/diagnostics` - Get diagnostics
- `GET /api/v1/health/alerts` - Get health alerts

### Data Synchronization
- `POST /api/v1/sync/request` - Request data sync
- `GET /api/v1/sync/status/{sync_id}` - Get sync status
- `POST /api/v1/sync/cancel/{sync_id}` - Cancel sync
- `POST /api/v1/sync/devices/{device_id}/sync` - Sync specific device
- `GET /api/v1/sync/metrics` - Get sync metrics

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `development` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `PORT` | Service port | `8080` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `POSTGRES_URL` | PostgreSQL connection URL | `postgresql://...` |
| `CONSUL_HOST` | Consul host | `localhost` |
| `CONSUL_PORT` | Consul port | `8500` |
| `DEVICE_HEARTBEAT_INTERVAL` | Heartbeat interval (seconds) | `30` |
| `DEVICE_TIMEOUT` | Device timeout (seconds) | `300` |
| `MAX_DEVICES_PER_REGION` | Max devices per region | `100` |
| `DEPLOYMENT_TIMEOUT` | Model deployment timeout | `600` |
| `SYNC_INTERVAL` | Data sync interval | `300` |
| `HEALTH_CHECK_INTERVAL` | Health check interval | `60` |

### Device Configuration

```json
{
  "device_name": "yellowstone-seismic-01",
  "device_type": "seismic_station",
  "location": {
    "latitude": 44.4280,
    "longitude": -110.5885,
    "altitude": 2400.0,
    "region": "us-west-1",
    "site_name": "Yellowstone National Park",
    "timezone": "America/Denver"
  },
  "capabilities": {
    "cpu_cores": 4,
    "memory_gb": 8.0,
    "storage_gb": 500.0,
    "gpu_available": false,
    "power_source": "solar",
    "operating_system": "Linux",
    "python_version": "3.11",
    "docker_available": true
  },
  "ip_address": "192.168.100.10",
  "port": 8080,
  "ssl_enabled": true
}
```

## Development

### Prerequisites
- Python 3.11+
- Redis 7.2+
- PostgreSQL 14+ with PostGIS
- Docker (optional)

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export REDIS_URL="redis://localhost:6379"
   export POSTGRES_URL="postgresql://user:pass@localhost:5432/prometheus"
   ```

3. **Run the service:**
   ```bash
   python main.py
   ```

### Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run integration tests
pytest tests/integration/ -v
```

### Docker Development

```bash
# Build image
docker build -t prometheus-ai/edge-device-manager .

# Run container
docker run -p 8080:8080 \
  -e REDIS_URL=redis://host.docker.internal:6379 \
  prometheus-ai/edge-device-manager
```

## Deployment

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/edge-device-manager-deployment.yaml

# Check deployment status
kubectl get pods -l app=edge-device-manager -n prometheus-ai

# View logs
kubectl logs -f deployment/edge-device-manager -n prometheus-ai
```

### Docker Compose

```yaml
version: '3.8'
services:
  edge-device-manager:
    build: .
    ports:
      - "8080:8080"
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://postgres:password@postgres:5432/prometheus
    depends_on:
      - redis
      - postgres
```

## Usage Examples

### Register a Device

```python
import aiohttp
import asyncio

async def register_device():
    device_config = {
        "device_name": "test-station-01",
        "device_type": "seismic_station",
        "location": {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "region": "us-west-1",
            "site_name": "San Francisco"
        },
        "capabilities": {
            "cpu_cores": 4,
            "memory_gb": 8.0,
            "storage_gb": 100.0,
            "gpu_available": False,
            "operating_system": "Linux",
            "python_version": "3.11"
        },
        "ip_address": "192.168.1.100",
        "port": 8080
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8080/api/v1/devices/register",
            json=device_config
        ) as response:
            result = await response.json()
            print(f"Device registered: {result['device_id']}")
            return result

# Run the example
asyncio.run(register_device())
```

### Deploy a Model

```python
async def deploy_model():
    deployment_request = {
        "model_id": "seismic-cnn-v1.0",
        "target_devices": ["edge-abc123def456"],
        "force_update": False,
        "rollback_on_failure": True
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8080/api/v1/models/deploy",
            json=deployment_request
        ) as response:
            result = await response.json()
            print(f"Deployment initiated: {result['deployment_id']}")
            return result
```

### Send Heartbeat with Metrics

```python
async def send_heartbeat(device_id):
    metrics = {
        "timestamp": "2024-01-01T12:00:00Z",
        "device_id": device_id,
        "cpu_usage_percent": 45.0,
        "memory_usage_percent": 60.0,
        "storage_usage_percent": 25.0,
        "temperature_celsius": 35.0,
        "active_models": 2,
        "data_processing_rate": 100.0,
        "error_count": 0
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:8080/api/v1/devices/{device_id}/heartbeat",
            json=metrics
        ) as response:
            result = await response.json()
            print(f"Heartbeat sent: {result['status']}")
```

## Monitoring and Observability

### Metrics

The service exposes Prometheus metrics on port 9090:

- `edge_devices_total` - Total number of registered devices
- `edge_devices_online` - Number of online devices
- `edge_model_deployments_total` - Total model deployments
- `edge_sync_operations_total` - Total sync operations
- `edge_health_checks_total` - Total health checks performed

### Health Checks

- **Liveness**: `GET /health` - Basic service health
- **Readiness**: `GET /ready` - Service readiness with dependencies

### Logging

Structured JSON logging with correlation IDs for request tracing:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "service": "edge-device-manager",
  "correlation_id": "abc123def456",
  "message": "Device registered successfully",
  "device_id": "edge-xyz789",
  "device_type": "seismic_station"
}
```

## Security

### Authentication
- JWT tokens for device authentication
- API key authentication for service-to-service communication
- Certificate-based authentication for edge devices

### Network Security
- TLS encryption for all communications
- Network policies for pod-to-pod communication
- Firewall rules for edge device access

### Data Protection
- Encryption at rest for sensitive device data
- Secure credential storage in Kubernetes secrets
- Regular security scanning of container images

## Troubleshooting

### Common Issues

1. **Device Registration Fails**
   - Check network connectivity
   - Verify device capabilities meet minimum requirements
   - Ensure regional limits not exceeded

2. **Model Deployment Fails**
   - Verify device has sufficient resources
   - Check model compatibility with device OS
   - Review deployment logs for specific errors

3. **Health Checks Failing**
   - Check device network connectivity
   - Verify device is responding to heartbeats
   - Review system resource usage

4. **Data Sync Issues**
   - Check network bandwidth and connectivity
   - Verify sync configuration settings
   - Review sync logs for specific errors

### Debug Commands

```bash
# Check service logs
kubectl logs -f deployment/edge-device-manager -n prometheus-ai

# Get device status
curl http://localhost:8080/api/v1/devices/{device_id}/status

# Check health summary
curl http://localhost:8080/api/v1/health/summary

# View sync metrics
curl http://localhost:8080/api/v1/sync/metrics
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is part of the Prometheus AI platform and is subject to the project's licensing terms.