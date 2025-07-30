# TimescaleDB Service

Specialized time-series database service for the Prometheus AI Geodynamic Platform. This service provides optimized storage, compression, and analysis capabilities for high-frequency geological sensor data.

## Features

- **High-Performance Time-Series Storage**: Optimized for geological sensor data with automatic partitioning
- **Intelligent Compression**: Automatic compression policies for long-term data retention
- **Real-Time Analytics**: Continuous aggregates for real-time monitoring and alerting
- **Geological Trend Analysis**: Advanced statistical analysis for geological trend detection
- **Anomaly Detection**: Automated anomaly detection using statistical methods
- **Spatial-Temporal Queries**: PostGIS integration for location-based queries
- **Multi-Schema Organization**: Organized schemas for sensors, seismic, volcanic, and analytics data

## Architecture

### Database Schema Organization

```
prometheus_timeseries/
├── sensors/              # Sensor readings and geochemical samples
│   ├── readings          # High-frequency sensor data
│   └── geochemical_samples # Laboratory analysis results
├── seismic/              # Seismic event data
│   └── events            # Earthquake and volcanic seismic events
├── volcanic/             # Volcanic monitoring data
│   └── monitoring        # Real-time volcanic activity data
└── analytics/            # Continuous aggregates and analysis views
    ├── sensor_readings_hourly
    ├── sensor_readings_daily
    ├── volcanic_activity_hourly
    └── seismic_activity_daily
```

### Hypertable Configuration

- **Sensor Readings**: 1-day chunks, 7-day compression, 5-year retention
- **Seismic Events**: 1-week chunks, 30-day compression, 10-year retention
- **Volcanic Monitoring**: 1-day chunks, 14-day compression, 7-year retention
- **Geochemical Samples**: 1-month chunks, 90-day compression, 20-year retention

## API Endpoints

### Health and Status

- `GET /health` - Service health check
- `GET /ready` - Service readiness check
- `GET /statistics/database` - Database performance statistics

### Data Ingestion

- `POST /sensor-readings/batch` - Batch insert sensor readings

### Data Querying

- `POST /query/time-series` - Query time-series data with optional aggregation

### Analysis

- `POST /analysis/trend` - Geological trend analysis with anomaly detection

### Maintenance

- `POST /maintenance/compress` - Manual compression trigger

## Usage Examples

### Insert Sensor Data

```python
import httpx
from datetime import datetime

readings = [
    {
        "sensor_id": "TEMP_001",
        "timestamp": "2024-01-01T12:00:00Z",
        "measurement_type": "temperature",
        "value": 25.5,
        "unit": "celsius",
        "quality_flag": "good",
        "location_lat": 37.7749,
        "location_lon": -122.4194
    }
]

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://timescale-service:8080/sensor-readings/batch",
        json=readings
    )
    print(response.json())
```

### Query Time-Series Data

```python
query = {
    "sensor_ids": ["TEMP_001", "TEMP_002"],
    "start_time": "2024-01-01T00:00:00Z",
    "end_time": "2024-01-01T23:59:59Z",
    "measurement_types": ["temperature"],
    "aggregation": "avg",
    "interval": "1h"
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://timescale-service:8080/query/time-series",
        json=query
    )
    data = response.json()
    print(f"Retrieved {data['count']} data points")
```

### Analyze Geological Trends

```python
analysis_request = {
    "sensor_id": "SEISMO_001",
    "measurement_type": "seismic_amplitude",
    "start_time": "2024-01-01T00:00:00Z",
    "end_time": "2024-01-07T23:59:59Z",
    "trend_window": "1d"
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://timescale-service:8080/analysis/trend",
        json=analysis_request
    )
    analysis = response.json()
    
    print(f"Trend significance: {analysis['geological_interpretation']['trend_significance']}")
    print(f"Anomaly level: {analysis['geological_interpretation']['anomaly_level']}")
```

## Configuration

### Environment Variables

- `TIMESCALE_DATABASE_URL`: PostgreSQL connection string for TimescaleDB
- `ENVIRONMENT`: Deployment environment (development/production)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `PORT`: Service port (default: 8080)

### Database Configuration

The service automatically configures:

- **Compression Policies**: Automatic compression based on data age
- **Retention Policies**: Automatic data cleanup based on retention periods
- **Continuous Aggregates**: Pre-computed aggregations for fast queries
- **Indexes**: Optimized indexes for common query patterns

## Deployment

### Docker

```bash
# Build the image
docker build -t prometheus-ai/timescale-db-service .

# Run the container
docker run -p 8080:8080 \
  -e TIMESCALE_DATABASE_URL="postgresql://user:pass@host:5432/db" \
  prometheus-ai/timescale-db-service
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/timescale-db-deployment.yaml

# Check deployment status
kubectl get pods -l app=timescale-db-service -n prometheus-ai
```

## Development

### Setup Development Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export TIMESCALE_DATABASE_URL="postgresql://prometheus:prometheus123@localhost:5432/prometheus_timeseries"

# Run the service
python main.py
```

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=.
```

### Database Setup

```bash
# Initialize TimescaleDB
psql -h localhost -U prometheus -d prometheus_timeseries -f database/timescale/init.sql

# Verify setup
python healthcheck.py
```

## Performance Optimization

### Compression

The service automatically compresses data based on age:

- **Sensor readings**: Compressed after 7 days
- **Seismic events**: Compressed after 30 days
- **Volcanic monitoring**: Compressed after 14 days

Compression typically achieves 5-10x size reduction for geological time-series data.

### Continuous Aggregates

Pre-computed aggregations provide fast query performance:

- **Hourly aggregates**: Updated every hour with 1-hour lag
- **Daily aggregates**: Updated daily with 1-day lag
- **Custom intervals**: Configurable aggregation intervals

### Query Optimization

- Use continuous aggregates for historical analysis
- Leverage spatial indexes for location-based queries
- Use appropriate time ranges to minimize data scanning
- Consider data quality flags in queries

## Monitoring

### Metrics

The service exposes metrics for:

- Database connection pool usage
- Query performance statistics
- Compression ratios
- Data ingestion rates
- Error rates and response times

### Alerts

Configure alerts for:

- High connection pool usage (>80%)
- Slow queries (>10 seconds)
- High error rates (>5%)
- Disk space usage
- Compression job failures

## Geological Analysis Features

### Trend Analysis

- **Linear regression**: Slope, intercept, R-squared calculation
- **Statistical significance**: Confidence intervals and p-values
- **Seasonal decomposition**: Trend, seasonal, and residual components

### Anomaly Detection

- **Z-score analysis**: Statistical outlier detection
- **Moving window**: Adaptive baseline calculation
- **Contextual anomalies**: Domain-specific anomaly rules

### Geological Interpretation

- **Trend significance**: Weak, moderate, strong trend classification
- **Volcanic implications**: Unrest detection and risk assessment
- **Recommendations**: Automated monitoring recommendations

## Troubleshooting

### Common Issues

1. **Connection timeouts**: Increase connection pool size or timeout values
2. **Slow queries**: Check indexes and consider using continuous aggregates
3. **High memory usage**: Adjust chunk intervals or compression policies
4. **Compression failures**: Check disk space and TimescaleDB version compatibility

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python main.py
```

### Health Checks

```bash
# Check service health
curl http://localhost:8080/health

# Check database readiness
curl http://localhost:8080/ready

# Get database statistics
curl http://localhost:8080/statistics/database
```

## Contributing

1. Follow the existing code structure and naming conventions
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure geological domain expertise is reflected in analysis functions
5. Consider performance implications for high-frequency data

## License

This service is part of the Prometheus AI Geodynamic Platform and follows the project's licensing terms.