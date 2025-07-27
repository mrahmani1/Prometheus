# Project Structure

## Root Directory Organization

```
prometheus-ai-platform/
├── .kiro/                    # Kiro IDE configuration and steering rules
├── services/                 # Microservices implementations
├── kubernetes/              # Kubernetes deployment manifests
├── database/                # Database initialization scripts
├── scripts/                 # Deployment and utility scripts
├── docker-compose.yml       # Development environment setup
└── project.md              # Comprehensive project documentation
```

## Services Architecture

### Core Services
- **api-gateway/**: Node.js Express gateway with service discovery
- **helios-core/**: Python FastAPI service for AI/ML geological modeling
- **data-ingestion/**: Python service for multi-source data processing
- **volcano-watchtower/**: Python service for real-time volcanic monitoring

### Service Structure Pattern
Each service follows this consistent structure:
```
services/{service-name}/
├── api/                     # API route definitions
├── core/                    # Business logic and algorithms
├── models/                  # Data models and schemas
├── tests/                   # Unit and integration tests
├── examples/                # Usage examples and demos
├── Dockerfile              # Container build configuration
├── main.py                 # Service entry point (Python)
├── requirements.txt        # Python dependencies
└── healthcheck.py          # Health check endpoint
```

## Infrastructure Configuration

### Kubernetes Manifests
- **namespace.yaml**: Isolated namespace for all resources
- **configmap.yaml**: Environment configuration
- **secrets.yaml**: Encrypted credentials and API keys
- **persistent-volumes.yaml**: Storage for databases
- **database-deployments.yaml**: Database service deployments
- **api-gateway-deployment.yaml**: Gateway service deployment
- **monitoring.yaml**: Prometheus and Grafana setup

### Database Organization
```
database/
└── postgres/
    └── init.sql            # PostgreSQL schema initialization
```

## Development Conventions

### File Naming
- **Python**: snake_case for files and modules
- **JavaScript**: camelCase for files, kebab-case for directories
- **Kubernetes**: kebab-case for all manifest files
- **Docker**: Dockerfile (capitalized) in each service root

### Directory Patterns
- **API Routes**: `/api/v1/{resource}` pattern for all endpoints
- **Tests**: Mirror source structure in `tests/` directory
- **Documentation**: Service-specific docs in `docs/` subdirectory
- **Examples**: Runnable examples in `examples/` subdirectory

### Configuration Management
- **Environment Variables**: Use consistent naming (SERVICE_NAME_CONFIG)
- **Secrets**: Store in Kubernetes secrets, never in code
- **Service Discovery**: All services register with Consul
- **Health Checks**: Every service implements `/health` and `/ready` endpoints

## Data Flow Architecture

### Request Flow
1. **API Gateway** → Service discovery via Consul
2. **Load Balancing** → Route to healthy service instances
3. **Authentication** → JWT validation and RBAC
4. **Service Processing** → Business logic execution
5. **Database Access** → Appropriate database for data type
6. **Response** → JSON API responses with correlation IDs

### Data Storage Patterns
- **Time-series data** → InfluxDB (sensor readings, monitoring)
- **Geospatial data** → PostgreSQL + PostGIS (locations, boundaries)
- **Graph relationships** → Neo4j (geological processes, dependencies)
- **Cache/Sessions** → Redis (temporary data, user sessions)
- **Event streaming** → Kafka (real-time data processing)

## Deployment Patterns

### Environment Separation
- **Development**: Docker Compose for local development
- **Staging**: Kubernetes cluster with integration testing
- **Production**: Multi-region Kubernetes with blue-green deployment

### Service Dependencies
Services should be designed to handle dependency failures gracefully:
- **Circuit breakers** for external service calls
- **Retry logic** with exponential backoff
- **Health checks** for dependency monitoring
- **Graceful degradation** when services are unavailable

## Code Organization Best Practices

### Python Services
- Use **FastAPI** for all HTTP services
- Implement **Pydantic** models for request/response validation
- Follow **async/await** patterns for I/O operations
- Use **structlog** for structured logging

### Node.js Services
- Use **Express.js** with middleware pattern
- Implement **Joi** for request validation
- Use **Winston** for structured logging
- Follow **async/await** over callbacks

### Testing Strategy
- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test service-to-service communication
- **End-to-end tests**: Test complete user workflows
- **Load tests**: Validate performance under load