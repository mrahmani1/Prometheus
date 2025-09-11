# Prometheus AI Platform - Microservices Architecture

This directory contains the microservices implementation of the Prometheus AI Platform, providing a scalable, distributed architecture for geological monitoring and analysis.

## Architecture Overview

The platform consists of four core microservices:

### 1. API Gateway (`api-gateway/`)
- **Technology**: Node.js + Express
- **Port**: 3000
- **Purpose**: Central entry point with authentication, rate limiting, and service discovery
- **Features**:
  - JWT authentication with role-based access control
  - Dynamic routing to backend services via Consul
  - Rate limiting and security middleware
  - Health checks and monitoring endpoints

### 2. Helios Core (`helios-core/`)
- **Technology**: Python + FastAPI
- **Port**: 8001
- **Purpose**: Physics-informed neural networks and geological analysis
- **Features**:
  - Geological PINN engine integration
  - Multi-physics coupling simulations
  - Volcano monitoring core functionality
  - Digital twin Earth system integration
  - Resource prospectivity analysis

### 3. Data Ingestion (`data-ingestion/`)
- **Technology**: Python + FastAPI
- **Port**: 8002
- **Purpose**: Multi-source geological data processing and streaming
- **Features**:
  - IRIS/EarthScope seismic data connector
  - Copernicus satellite data integration
  - ASF-Hyp3 InSAR processing
  - USGS monitoring data feeds
  - NASA EarthData integration
  - Real-time Kafka streaming
  - Data validation and quality control

### 4. Volcano Watchtower (`volcano-watchtower/`)
- **Technology**: Python + FastAPI + WebSockets
- **Port**: 8003
- **Purpose**: Real-time volcanic monitoring and eruption forecasting
- **Features**:
  - Seismic transformer for event detection
  - ML-based eruption forecasting
  - Caldera system classification
  - Multi-sensor data fusion
  - Real-time alert generation
  - WebSocket streaming for live data

## Infrastructure Services

### Service Discovery
- **Consul**: Service registration and health checking
- **Port**: 8500

### Data Storage
- **PostgreSQL + PostGIS**: Geospatial data (Port: 5432)
- **InfluxDB**: Time-series monitoring data (Port: 8086)
- **Neo4j**: Knowledge graphs and relationships (Port: 7474/7687)
- **Redis**: Caching and session management (Port: 6379)

### Message Streaming
- **Apache Kafka**: Real-time data streaming (Port: 9092)
- **Zookeeper**: Kafka coordination (Port: 2181)

## Development Setup

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

### Quick Start
```bash
# Start all services
./scripts/deploy.sh development

# Or manually with Docker Compose
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Individual Service Development
```bash
# API Gateway
cd services/api-gateway
npm install
npm run dev

# Python services (Helios Core, Data Ingestion, Volcano Watchtower)
cd services/helios-core  # or data-ingestion, volcano-watchtower
pip install -r requirements.txt
python main.py
```

## Service Communication

### Authentication Flow
1. Client authenticates with API Gateway (`POST /auth/login`)
2. Gateway returns JWT token
3. All subsequent requests include `Authorization: Bearer <token>`
4. Gateway validates token and forwards requests to appropriate services

### Service Discovery
- Services register with Consul on startup
- API Gateway discovers services dynamically
- Health checks ensure only healthy services receive traffic
- Load balancing via round-robin across service instances

### Data Flow
1. **Data Ingestion** → Kafka → **Processing Services**
2. **Helios Core** ← API Gateway ← **Client Requests**
3. **Volcano Watchtower** → WebSocket → **Real-time Updates**
4. All services → Redis (caching) and databases (persistence)

## API Endpoints

### API Gateway (Port 3000)
- `GET /health` - Health check
- `POST /auth/login` - Authentication
- `GET /api/services` - Service status
- `/api/v1/helios/*` - Proxy to Helios Core
- `/api/v1/data/*` - Proxy to Data Ingestion
- `/api/v1/volcano/*` - Proxy to Volcano Watchtower

### Helios Core (Port 8001)
- `POST /analyze/geological` - Geological analysis
- `POST /simulate/physics` - Physics simulations
- `GET /models/status` - Model status

### Data Ingestion (Port 8002)
- `POST /ingest` - Ingest data from sources
- `POST /validate` - Validate data
- `POST /quality-control` - Quality assessment
- `GET /sources` - List available sources

### Volcano Watchtower (Port 8003)
- `POST /monitor` - Monitor volcano
- `POST /forecast` - Eruption forecast
- `POST /alerts/configure` - Configure alerts
- `GET /volcanoes/{id}/status` - Volcano status
- `WS /ws/{volcano_id}` - Real-time WebSocket

## Configuration

### Environment Variables
Each service supports configuration via environment variables:

```bash
# Common
PORT=8001
CONSUL_HOST=localhost
CONSUL_PORT=8500
REDIS_URL=redis://localhost:6379

# API Gateway specific
JWT_SECRET=your-secret-key
NODE_ENV=development

# Python services specific
ENVIRONMENT=development
```

### Docker Compose Override
Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.8'
services:
  helios-core:
    environment:
      - DEBUG=true
    volumes:
      - ./custom-models:/app/models
```

## Testing

### Unit Tests
```bash
# Node.js services
cd services/api-gateway
npm test

# Python services
cd services/helios-core
pytest tests/ -v --cov=.
```

### Integration Tests
```bash
# Test API endpoints
newman run tests/integration/prometheus-api.postman_collection.json

# Test service communication
python tests/integration/test_service_communication.py
```

### Load Testing
```bash
# Test API Gateway performance
k6 run tests/load/api-gateway-load-test.js
```

## Monitoring and Observability

### Health Checks
- All services expose `/health` and `/ready` endpoints
- Consul performs periodic health checks
- Docker health checks for container monitoring

### Logging
- Structured JSON logging with correlation IDs
- Centralized logging via Docker log drivers
- Log levels: DEBUG, INFO, WARNING, ERROR

### Metrics
- Prometheus metrics endpoints (`/metrics`)
- Service-specific performance metrics
- Infrastructure metrics via Docker stats

## Security

### Authentication & Authorization
- JWT tokens with configurable expiration
- Role-based access control (RBAC)
- API key support for service-to-service communication

### Network Security
- Services communicate within Docker network
- External access only through API Gateway
- TLS termination at gateway level

### Data Protection
- Encrypted credentials via Kubernetes secrets
- Database connection encryption
- Input validation and sanitization

## Deployment

### Development
- Docker Compose for local development
- Hot reloading for code changes
- Debug configurations available

### Production
- Kubernetes deployment manifests (see `kubernetes/` directory)
- Horizontal pod autoscaling
- Rolling updates and blue-green deployment
- Production-grade security configurations

## Troubleshooting

### Common Issues

1. **Service Discovery Failures**
   ```bash
   # Check Consul status
   curl http://localhost:8500/v1/status/leader
   
   # View service registrations
   curl http://localhost:8500/v1/catalog/services
   ```

2. **Database Connection Issues**
   ```bash
   # Check PostgreSQL
   docker exec prometheus-postgres pg_isready
   
   # Check Redis
   docker exec prometheus-redis redis-cli ping
   ```

3. **Service Communication Errors**
   ```bash
   # Check service logs
   docker-compose logs -f helios-core
   
   # Test service endpoints directly
   curl http://localhost:8001/health
   ```

### Debug Mode
Enable debug logging by setting environment variables:
```bash
export DEBUG=true
export LOG_LEVEL=debug
```

## Contributing

1. Follow the established service structure pattern
2. Implement proper health checks and metrics
3. Add comprehensive tests (unit + integration)
4. Update documentation for new endpoints
5. Ensure proper error handling and logging

## Next Steps

1. **Kubernetes Deployment** - Production-ready orchestration
2. **Enhanced Monitoring** - Prometheus + Grafana dashboards
3. **API Documentation** - OpenAPI/Swagger specifications
4. **Performance Optimization** - Caching strategies and optimization
5. **Security Hardening** - Advanced security measures