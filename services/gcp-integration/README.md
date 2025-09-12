# GCP Data Analytics Integration Service

Comprehensive Google Cloud Platform data analytics and BigQuery integration for geological data processing.

## Features

### Core Services
- **BigQuery Geological Analytics**: Petabyte-scale geological data analysis with BigQuery ML
- **Dataflow Real-time Processing**: Real-time geological data processing using Apache Beam
- **Dataproc Spark Analytics**: Distributed geological analytics using Apache Spark
- **Earth Engine Integration**: Satellite data analysis for geological monitoring
- **Composer Workflow Orchestration**: Automated workflow management
- **Data Fusion Visual Pipelines**: Visual data pipeline creation and management
- **Looker Geological BI**: Business intelligence and data visualization
- **Data Catalog Management**: Data discovery and governance
- **Dataplex Unified Management**: Unified data lake management

### Geological Analysis Capabilities
- Seismic event detection and classification
- Volcanic monitoring and eruption forecasting
- Mineral prospectivity mapping
- Geological data clustering and classification
- Time series analysis and forecasting
- Thermal anomaly detection
- Vegetation stress analysis
- Landslide detection and monitoring
- Spatial geological analysis

## Quick Start

### Prerequisites
- Google Cloud Project with enabled APIs:
  - BigQuery API
  - Dataflow API
  - Dataproc API
  - Earth Engine API
  - Composer API
  - Data Fusion API
  - Data Catalog API
  - Dataplex API
- Service account with appropriate permissions
- Docker (for containerized deployment)

### Local Development

1. **Clone and setup**:
```bash
cd services/gcp-integration
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_REGION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

3. **Run the service**:
```bash
python main.py
```

### Docker Deployment

1. **Build the image**:
```bash
docker build -t gcp-integration:latest .
```

2. **Run the container**:
```bash
docker run -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT="your-project-id" \
  -e GOOGLE_CLOUD_REGION="us-central1" \
  -v /path/to/service-account.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json" \
  gcp-integration:latest
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gcp-integration
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gcp-integration
  template:
    metadata:
      labels:
        app: gcp-integration
    spec:
      containers:
      - name: gcp-integration
        image: gcp-integration:latest
        ports:
        - containerPort: 8080
        env:
        - name: GOOGLE_CLOUD_PROJECT
          value: "your-project-id"
        - name: GOOGLE_CLOUD_REGION
          value: "us-central1"
        volumeMounts:
        - name: gcp-credentials
          mountPath: /app/credentials.json
          subPath: credentials.json
      volumes:
      - name: gcp-credentials
        secret:
          secretName: gcp-service-account
```

## API Documentation

### Health Endpoints
- `GET /health/` - Basic health check
- `GET /health/ready` - Readiness check
- `GET /health/live` - Liveness check
- `GET /health/detailed` - Detailed health status

### BigQuery Endpoints
- `POST /api/v1/bigquery/analyze` - Analyze geological dataset
- `POST /api/v1/bigquery/query` - Execute custom query
- `GET /api/v1/bigquery/datasets` - List datasets
- `GET /api/v1/bigquery/ml-models` - List ML models

### Dataflow Endpoints
- `POST /api/v1/dataflow/process-stream` - Process data stream with specialized pipelines
- `GET /api/v1/dataflow/jobs` - List active jobs
- `GET /api/v1/dataflow/jobs/{job_id}` - Get job status and details
- `DELETE /api/v1/dataflow/jobs/{job_id}` - Cancel running job

#### Supported Processing Types
- **seismic**: Real-time seismic event detection and classification
- **volcanic**: Multi-parameter volcanic monitoring and alert generation
- **satellite**: Satellite imagery processing and analysis
- **geochemical**: Geochemical data processing and mineral enrichment
- **real_time_alerts**: ✅ **NEW** - Intelligent alert processing with prioritization and routing

### Dataproc Endpoints
- `POST /api/v1/dataproc/analyze` - Run Spark analytics
- `GET /api/v1/dataproc/clusters` - List clusters
- `GET /api/v1/dataproc/jobs/{job_id}` - Get job status

### Earth Engine Endpoints
- `POST /api/v1/earth-engine/analyze` - Analyze satellite data
- `GET /api/v1/earth-engine/datasets` - List available datasets
- `POST /api/v1/earth-engine/thermal-anomaly` - Detect thermal anomalies
- `POST /api/v1/earth-engine/mineral-mapping` - Perform mineral mapping

## Configuration

### Environment Variables
- `GOOGLE_CLOUD_PROJECT` - GCP project ID
- `GOOGLE_CLOUD_REGION` - GCP region (default: us-central1)
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to service account JSON
- `PORT` - Service port (default: 8080)
- `ENVIRONMENT` - Environment (development/production)

### Service Account Permissions
Required IAM roles:
- BigQuery Admin
- Dataflow Admin
- Dataproc Admin
- Earth Engine Resource Admin
- Composer Admin
- Data Fusion Admin
- Data Catalog Admin
- Dataplex Admin
- Storage Admin

## Monitoring and Logging

### Health Checks
The service provides comprehensive health checks for Kubernetes:
- Liveness probe: `/health/live`
- Readiness probe: `/health/ready`

### Metrics
Prometheus metrics available at `/metrics`:
- Request count and duration
- Service health status
- GCP service connectivity

### Logging
Structured JSON logging with correlation IDs for request tracing.

## Development

### Project Structure
```
services/gcp-integration/
├── main.py                 # FastAPI application entry point
├── config.yaml            # Service configuration
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build configuration
├── api/                   # REST API endpoints
│   ├── bigquery.py
│   ├── dataflow.py
│   ├── dataproc.py
│   ├── earth_engine.py
│   └── ...
├── core/                  # Core service implementations
│   ├── bigquery_integration.py
│   ├── dataflow_processor.py
│   ├── dataproc_analytics.py
│   ├── earth_engine_integration.py
│   └── ...
└── models/                # Data models and schemas
    └── gcp_models.py
```

### Testing
```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## Examples

### Analyze Seismic Data
```python
import httpx

response = httpx.post("http://localhost:8080/api/v1/bigquery/analyze", json={
    "dataset_id": "seismic_events_2024",
    "analysis_type": "seismic_analysis",
    "parameters": {
        "magnitude_threshold": 4.0,
        "region": "california"
    }
})
```

### Process Real-time Volcanic Data
```python
response = httpx.post("http://localhost:8080/api/v1/dataflow/process-stream", json={
    "job_name": "volcanic-monitoring",
    "processing_type": "volcanic",
    "input_source": "volcanic-data-stream",
    "output_destination": "volcanic-analysis-results",
    "parameters": {
        "volcano_ids": ["mount_st_helens", "yellowstone"]
    }
})
```

### Detect Thermal Anomalies
```python
response = httpx.post("http://localhost:8080/api/v1/earth-engine/thermal-anomaly", json={
    "region": {
        "bounds": {
            "north": 46.2,
            "south": 46.1,
            "east": -122.1,
            "west": -122.3
        }
    },
    "date_range": {
        "start": "2024-01-01",
        "end": "2024-12-31"
    },
    "threshold_multiplier": 2.5
})
```

### Process Real-time Alerts
```python
response = httpx.post("http://localhost:8080/api/v1/dataflow/process-stream", json={
    "stream_name": "geological-alerts-stream",
    "processing_type": "real_time_alerts",
    "parameters": {
        "alert_types": ["seismic", "volcanic", "landslide"],
        "priority_levels": ["critical", "high", "medium"],
        "notification_channels": ["email", "sms", "webhook"],
        "escalation_rules": {
            "critical": {"timeout_minutes": 5, "escalate_to": "emergency_team"},
            "high": {"timeout_minutes": 15, "escalate_to": "duty_officer"}
        }
    }
})
```

## Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review logs for error details
3. Verify GCP service permissions
4. Check service health at `/health/detailed`

## License

This service is part of the Prometheus AI Platform project.