# Intelligent Cache Service

Advanced caching and data optimization service for the Prometheus AI Platform with geological domain awareness.

## Features

### Core Caching Capabilities
- **Multi-Level Caching**: L1 (Memory), L2 (Redis), L3 (Distributed), L4 (CDN)
- **Intelligent Cache Invalidation**: Geological context-aware invalidation strategies
- **Advanced Data Compression**: Type-specific compression with scientific accuracy preservation
- **Data Deduplication**: Hash-based deduplication to reduce storage overhead
- **Cache Coherence**: Distributed cache consistency protocols

### Query Optimization
- **Geological Domain-Aware Optimization**: Specialized optimization for volcanic, seismic, and geological queries
- **Result Caching**: Intelligent caching of query results with semantic understanding
- **Query Plan Optimization**: Automated query rewriting and optimization
- **Index Recommendations**: Automated database index suggestions
- **Performance Monitoring**: Real-time query performance tracking

### Intelligent Prefetching
- **Predictive Prefetching**: ML-based prediction of data access patterns
- **Workflow-Based Prefetching**: Geological workflow-aware data prefetching
- **User Behavior Analysis**: Personalized prefetching based on user patterns
- **Spatial-Temporal Locality**: Geographic and temporal data locality optimization
- **Progressive Loading**: Priority-based data loading strategies

### Performance Monitoring
- **Real-Time Metrics**: Comprehensive performance metrics collection
- **Geological Performance Analysis**: Domain-specific performance analysis
- **Automated Optimization**: Self-optimizing cache strategies
- **Alert System**: Intelligent alerting with geological context
- **Cost Optimization**: Multi-cloud cost optimization recommendations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI REST API                         │
├─────────────────────────────────────────────────────────────┤
│  Cache Manager  │ Query Optimizer │ Prefetch Engine │ Monitor │
├─────────────────────────────────────────────────────────────┤
│                Multi-Level Cache Storage                    │
│  L1: Memory  │  L2: Redis  │  L3: Distributed  │  L4: CDN  │
├─────────────────────────────────────────────────────────────┤
│              Geological Domain Intelligence                 │
│  Volcanic    │  Seismic    │  Geological   │  Resource     │
│  Monitoring  │  Analysis   │  Modeling     │  Assessment   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+
- Redis 7.0+
- PostgreSQL 14+ (optional)
- Docker (optional)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd services/intelligent-cache
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure the service**
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your settings
```

4. **Start Redis (if not already running)**
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

5. **Run the service**
```bash
python main.py
```

### Docker Deployment

1. **Build the image**
```bash
docker build -t intelligent-cache:latest .
```

2. **Run with Docker Compose**
```bash
docker-compose up -d
```

## API Documentation

### Cache Operations

#### Get Cached Data
```http
GET /api/v1/cache/{key}?data_type=seismic_waveform
```

#### Set Cached Data
```http
POST /api/v1/cache
Content-Type: application/json

{
  "key": "seismic_events_kilauea_001",
  "data": {...},
  "data_type": "seismic_waveform",
  "geological_context": {
    "volcano_id": "kilauea_001",
    "domain": "volcanic",
    "alert_level": 3
  },
  "ttl_hours": 24
}
```

#### Invalidate Cache
```http
DELETE /api/v1/cache/{key}?cascade=true
```

### Query Optimization

#### Optimize Query
```http
POST /api/v1/query/optimize
Content-Type: application/json

{
  "query": "SELECT * FROM seismic_events WHERE magnitude >= 4.0",
  "parameters": {
    "start_time": "2024-01-01T00:00:00Z",
    "end_time": "2024-01-02T00:00:00Z"
  },
  "geological_context": {
    "domain": "seismic",
    "region": "pacific_ring_of_fire"
  }
}
```

#### Get Optimization Recommendations
```http
GET /api/v1/query/recommendations
```

### Prefetching

#### Record Data Access
```http
POST /api/v1/prefetch/record-access
Content-Type: application/json

{
  "user_id": "researcher_001",
  "session_id": "session_123",
  "data_key": "seismic_events",
  "geological_context": {
    "domain": "seismic",
    "workflow": "earthquake_analysis"
  }
}
```

#### Warm Workflow Cache
```http
POST /api/v1/prefetch/warm-workflow
Content-Type: application/json

{
  "workflow_id": "eruption_monitoring",
  "geological_context": {
    "volcano_id": "kilauea_001",
    "alert_level": 3
  }
}
```

### Performance Monitoring

#### Get Performance Summary
```http
POST /api/v1/performance/summary
Content-Type: application/json

{
  "time_range_hours": 24,
  "include_geological_analysis": true
}
```

#### Get Performance Alerts
```http
GET /api/v1/performance/alerts?severity=critical&hours=24
```

#### Get Optimization Recommendations
```http
GET /api/v1/performance/recommendations?priority_threshold=7
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `DEBUG` | Enable debug mode | `false` |
| `PORT` | Service port | `8080` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Geological Domain Configuration

The service includes specialized configuration for different geological domains:

#### Volcanic Monitoring
- Real-time cache TTL: 5 minutes
- Alert level priorities: 1 (Low) to 4 (Critical)
- Automatic cache warming for high-alert volcanoes

#### Seismic Analysis
- Waveform compression enabled
- Parallel processing for large datasets
- Magnitude-based indexing optimization

#### Geological Modeling
- Extended cache TTL: 24 hours
- 3D spatial indexing
- High compression for model data

#### Resource Assessment
- Economic data caching: 12 hours
- Grade threshold optimization
- Prospectivity map caching

## Performance Optimization

### Cache Optimization Strategies

1. **Geological Context Awareness**
   - Volcanic alert level-based prioritization
   - Seismic magnitude-based caching
   - Temporal locality for real-time data

2. **Compression Optimization**
   - Seismic waveform: Wavelet compression
   - Geological models: High-ratio compression
   - Visualization data: Balanced compression

3. **Prefetching Strategies**
   - Workflow-based prediction
   - User behavior analysis
   - Spatial-temporal locality

### Monitoring and Alerting

The service provides comprehensive monitoring with geological domain awareness:

- **Latency Monitoring**: Real-time latency tracking with geological context
- **Hit Rate Analysis**: Cache efficiency by geological domain
- **Cost Optimization**: Multi-cloud cost tracking and optimization
- **Geological Impact Scoring**: Performance impact on scientific workflows

## Development

### Running Tests
```bash
pytest tests/ -v --cov=core
```

### Code Quality
```bash
black core/ tests/
isort core/ tests/
mypy core/
```

### Adding New Geological Domains

1. **Extend GeologicalWorkflowPredictor**
```python
# In core/prefetch_engine.py
self.workflows['new_domain'] = GeologicalWorkflow(
    workflow_id='new_domain',
    name='New Domain Analysis',
    data_sequence=['data1', 'data2', 'data3'],
    # ... other parameters
)
```

2. **Add Domain-Specific Optimization**
```python
# In core/query_optimizer.py
async def _optimize_new_domain_query(self, query, parameters, pattern):
    # Domain-specific optimization logic
    return optimized_query, optimized_params
```

3. **Configure Performance Thresholds**
```python
# In core/performance_monitor.py
self.geological_performance_patterns['new_domain'] = {
    'expected_latency': 1.0,
    'expected_hit_rate': 0.8,
    # ... other thresholds
}
```

## Deployment

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: intelligent-cache
spec:
  replicas: 3
  selector:
    matchLabels:
      app: intelligent-cache
  template:
    metadata:
      labels:
        app: intelligent-cache
    spec:
      containers:
      - name: intelligent-cache
        image: intelligent-cache:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Production Considerations

1. **Security**
   - Enable JWT authentication
   - Use TLS for all connections
   - Implement rate limiting

2. **Scalability**
   - Use Redis Cluster for L2 cache
   - Implement horizontal pod autoscaling
   - Configure distributed caching

3. **Monitoring**
   - Export metrics to Prometheus
   - Set up Grafana dashboards
   - Configure alerting rules

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check L1 cache size configuration
   - Enable compression
   - Implement more aggressive eviction

2. **Low Hit Rate**
   - Analyze access patterns
   - Adjust TTL settings
   - Improve prefetching strategies

3. **High Latency**
   - Check Redis connection
   - Optimize query patterns
   - Enable parallel processing

### Debug Mode

Enable debug mode for detailed logging:
```bash
export DEBUG=true
python main.py
```

### Health Checks

The service provides health check endpoints:
- `/health` - Basic health check
- `/ready` - Readiness check with dependency validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation wiki