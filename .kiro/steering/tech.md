# Technology Stack

## Architecture
- **Microservices Architecture**: Containerized services with Docker and Kubernetes
- **Service Discovery**: Consul for dynamic service registration and health checking
- **API Gateway**: Node.js Express gateway with rate limiting and JWT authentication
- **Message Streaming**: Apache Kafka for real-time data processing

## Backend Technologies

### Node.js Services (API Gateway)
- **Framework**: Express.js 4.18+
- **Key Libraries**: helmet, cors, express-rate-limit, consul, redis, winston
- **Authentication**: JWT tokens with role-based access control
- **Node Version**: 18.0.0+

### Python Services (Core AI/ML)
- **Framework**: FastAPI with uvicorn
- **AI/ML Stack**: PyTorch 2.0+, scikit-learn, pytorch-lightning
- **Scientific Computing**: numpy, scipy, pandas, xarray
- **Geospatial**: geopandas, shapely, pyproj, rasterio
- **Physics Libraries**: For PINNs and geological modeling
- **Python Version**: 3.11+

## Database Stack
- **InfluxDB 2.7**: Time-series data for real-time sensor readings
- **PostgreSQL + PostGIS**: Geospatial data with geological schemas
- **Neo4j 5.12**: Graph database for geological process relationships
- **Redis 7.2**: Caching and session management
- **Kafka**: Message streaming and event processing

## Infrastructure
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Kubernetes with namespace isolation
- **Service Mesh**: Consul Connect for service-to-service communication
- **Monitoring**: Prometheus metrics with Grafana dashboards
- **Logging**: Structured JSON logging with correlation IDs

## Development Tools
- **Testing**: Jest (Node.js), pytest (Python)
- **Code Quality**: ESLint (JS), Black (Python)
- **API Documentation**: OpenAPI/Swagger with interactive docs
- **Load Testing**: k6 for performance testing

## Common Commands

### Development Environment
```bash
# Start full development stack
./scripts/deploy.sh development

# Individual service development
cd services/helios-core
pip install -r requirements.txt
python main.py

cd services/api-gateway  
npm install
npm run dev
```

### Testing
```bash
# Python services
cd services/helios-core
pytest tests/ -v --cov=.

# Node.js services
cd services/api-gateway
npm test
npm run test:watch

# Integration tests
newman run tests/integration/prometheus-api.postman_collection.json
```

### Production Deployment
```bash
# Deploy to Kubernetes
./scripts/deploy.sh production

# Check deployment status
kubectl get pods -n prometheus-ai
kubectl logs -f deployment/helios-core -n prometheus-ai
```

### Database Operations
```bash
# Access databases in development
docker-compose exec postgres psql -U prometheus -d prometheus
docker-compose exec influxdb influx
docker-compose exec neo4j cypher-shell -u neo4j -p prometheus123
```

## Security Requirements
- All external APIs require JWT authentication
- Database connections use encrypted credentials from Kubernetes secrets
- Container images scanned with Trivy for vulnerabilities
- Network policies enforce pod-to-pod communication restrictions