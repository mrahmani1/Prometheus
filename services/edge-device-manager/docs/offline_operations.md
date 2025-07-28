# Offline Operations Documentation

## Overview

The Offline Operations system provides comprehensive capabilities for edge devices to continue functioning when internet connectivity is intermittent or unavailable. This system implements the requirements for offline operation capabilities (Requirements 10.2 and 10.3) from the Prometheus AI Geodynamic Platform specification.

## Key Features

### 1. Local Data Processing and Analysis
- **Offline Data Processor**: Processes sensor data using cached models and rule-based algorithms
- **Multi-sensor Support**: Handles seismic, thermal, gas composition, and deformation data
- **Real-time Analysis**: Provides immediate analysis capabilities without cloud connectivity
- **Quality Validation**: Validates and cleans incoming sensor data

### 2. Offline Alert Generation
- **Pre-computed Thresholds**: Uses pre-configured thresholds for immediate alert generation
- **Rule-based Alerting**: Implements geological monitoring rules for different data types
- **Alert Prioritization**: Categorizes alerts by severity (low, medium, high, critical)
- **Alert Management**: Supports acknowledgment and resolution of alerts

### 3. Local Data Storage and Management
- **SQLite Database**: Lightweight database for metadata and small data storage
- **File-based Storage**: Efficient storage of large sensor data with compression
- **Data Retention**: Configurable retention policies for different data types
- **Storage Optimization**: Automatic compression and cleanup of old data

### 4. Connectivity Restoration and Synchronization
- **Connectivity Monitoring**: Continuous monitoring of internet connectivity
- **Automatic Sync**: Automatic synchronization when connectivity is restored
- **Priority-based Sync**: Prioritizes critical data for synchronization
- **Retry Logic**: Robust retry mechanisms for failed synchronizations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Offline Operations Coordinator               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Offline Data    │  │ Alert Generator │  │ Local Storage│ │
│  │ Processor       │  │                 │  │ Manager      │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Connectivity Manager                          │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Components

### OfflineDataProcessor
Handles local data processing using simplified models:

- **Seismic Classification**: Rule-based event classification (earthquake, volcanic, explosion)
- **Deformation Analysis**: Threshold-based ground movement detection
- **Thermal Monitoring**: Temperature anomaly detection
- **Gas Analysis**: Composition analysis with normal range validation

### OfflineAlertGenerator
Generates alerts using pre-computed thresholds:

- **Alert Rules**: Configurable rules for different geological phenomena
- **Cooldown Management**: Prevents alert spam with configurable cooldown periods
- **Severity Assessment**: Automatic severity assignment based on trigger values
- **Alert History**: Maintains history of all generated alerts

### LocalStorageManager
Manages local data storage:

- **Multi-format Support**: Handles JSON, binary, and compressed data
- **Metadata Management**: SQLite database for efficient metadata queries
- **Storage Limits**: Enforces storage limits with automatic cleanup
- **Data Integrity**: Checksum validation for stored data

### ConnectivityManager
Handles connectivity and synchronization:

- **Multi-endpoint Testing**: Tests connectivity to multiple services
- **Adaptive Sync**: Adjusts sync behavior based on connectivity quality
- **Bandwidth Management**: Respects bandwidth limits during synchronization
- **Failure Recovery**: Robust error handling and retry logic

## API Endpoints

### Data Ingestion
```http
POST /api/v1/offline/ingest
Content-Type: application/json

{
  "data_type": "seismic",
  "data": {
    "magnitude": 3.2,
    "depth": 15.0,
    "duration": 45.0
  },
  "location": {
    "latitude": 45.0,
    "longitude": -122.0
  }
}
```

### Status Monitoring
```http
GET /api/v1/offline/status
```

Returns comprehensive offline operation status including:
- Connectivity status
- Processing activity
- Storage usage
- Alert counts
- Sync queue status

### Alert Management
```http
GET /api/v1/offline/alerts
POST /api/v1/offline/alerts/acknowledge
POST /api/v1/offline/alerts/{alert_id}/resolve
```

### Synchronization Control
```http
GET /api/v1/offline/sync/status
POST /api/v1/offline/sync/force
PUT /api/v1/offline/sync/config
```

## Configuration

### Alert Thresholds
Alert thresholds are configurable in `alert_config.json`:

```json
{
  "alert_rules": [
    {
      "rule_id": "seismic_magnitude_high",
      "name": "High Magnitude Seismic Event",
      "data_type": "seismic",
      "condition": "magnitude >= threshold",
      "threshold_value": 4.5,
      "severity": "high",
      "cooldown_minutes": 5
    }
  ]
}
```

### Storage Configuration
Storage settings in device configuration:

```python
storage_config = {
    "max_storage_gb": 10.0,
    "compression_threshold": 1024,  # bytes
    "retention_days": 7,
    "cleanup_interval": 3600  # seconds
}
```

### Sync Configuration
Synchronization parameters:

```python
sync_config = {
    "bandwidth_limit_mbps": 10.0,
    "max_concurrent_syncs": 3,
    "sync_batch_size": 10,
    "connectivity_check_interval": 30  # seconds
}
```

## Usage Examples

### Basic Setup
```python
from core.offline_operations import OfflineOperationsCoordinator
from models.config import EdgeDeviceConfig

# Create configuration
config = EdgeDeviceConfig(
    device_id="edge-001",
    device_name="Seismic Station 1",
    device_type="seismic_station",
    # ... other config
)

# Initialize coordinator
coordinator = OfflineOperationsCoordinator(config)

# Start offline operations
await coordinator.start()
```

### Data Ingestion
```python
# Ingest seismic data
seismic_data = {
    "magnitude": 3.2,
    "depth": 15.0,
    "duration": 45.0
}

record_id = await coordinator.ingest_sensor_data(
    data_type="seismic",
    data=seismic_data,
    location={"latitude": 45.0, "longitude": -122.0}
)
```

### Alert Handling
```python
# Get active alerts
alerts = await coordinator.get_active_alerts()

# Acknowledge alert
await coordinator.acknowledge_alert(alert_id)

# Resolve alert
await coordinator.resolve_alert(alert_id)
```

### Status Monitoring
```python
# Get operation status
status = await coordinator.get_operation_status()

# Get comprehensive status
comprehensive = await coordinator.get_comprehensive_status()
```

## Data Flow

1. **Data Ingestion**: Sensor data is ingested via API or direct calls
2. **Local Storage**: Data is stored locally with metadata
3. **Processing Queue**: Data is queued for offline processing
4. **Analysis**: Data is analyzed using cached models and rules
5. **Alert Generation**: Alerts are generated based on analysis results
6. **Storage**: Results and alerts are stored locally
7. **Connectivity Check**: System continuously monitors connectivity
8. **Synchronization**: When online, data is synchronized with cloud services

## Performance Characteristics

### Processing Performance
- **Throughput**: ~100 sensor readings per minute
- **Latency**: <1 second for immediate alert evaluation
- **Storage**: ~1MB per 1000 sensor readings (compressed)

### Storage Efficiency
- **Compression**: ~70% size reduction for typical geological data
- **Retention**: Configurable retention policies (default 7 days)
- **Cleanup**: Automatic cleanup of expired data

### Connectivity Resilience
- **Offline Duration**: Supports weeks of offline operation
- **Sync Recovery**: Automatic sync when connectivity restored
- **Data Integrity**: Checksum validation prevents data corruption

## Monitoring and Diagnostics

### Health Checks
```http
GET /api/v1/offline/health
```

### Statistics
```http
GET /api/v1/offline/storage/statistics
GET /api/v1/offline/processing/statistics
```

### Logs
Structured logging with correlation IDs for troubleshooting:
- Processing events
- Alert generation
- Connectivity changes
- Sync operations
- Error conditions

## Error Handling

### Data Processing Errors
- Invalid data format → Data cleaning and retry
- Model failures → Fallback to simplified models
- Storage errors → Alternative storage locations

### Connectivity Errors
- Network timeouts → Exponential backoff retry
- Authentication failures → Token refresh
- Service unavailable → Queue for later retry

### Storage Errors
- Disk full → Automatic cleanup of old data
- Corruption → Checksum validation and recovery
- Permission errors → Fallback storage locations

## Security Considerations

### Data Protection
- Local data encryption at rest
- Secure token storage
- Access control for API endpoints

### Network Security
- TLS encryption for sync operations
- Certificate validation
- Rate limiting for API endpoints

## Testing

### Unit Tests
```bash
cd services/edge-device-manager
python -m pytest tests/test_offline_operations.py
```

### Integration Tests
```bash
python test_offline_operations.py
```

### Load Testing
```bash
python tests/load_test_offline.py
```

## Troubleshooting

### Common Issues

1. **High Storage Usage**
   - Check retention policies
   - Run manual cleanup
   - Increase storage limits

2. **Sync Failures**
   - Verify connectivity
   - Check authentication tokens
   - Review sync queue

3. **Missing Alerts**
   - Verify alert thresholds
   - Check rule configuration
   - Review processing logs

4. **Performance Issues**
   - Monitor CPU/memory usage
   - Check processing queue size
   - Review data compression

### Debug Commands
```bash
# Check offline status
curl http://localhost:8080/api/v1/offline/status

# View active alerts
curl http://localhost:8080/api/v1/offline/alerts

# Check storage statistics
curl http://localhost:8080/api/v1/offline/storage/statistics

# Force synchronization
curl -X POST http://localhost:8080/api/v1/offline/sync/force
```

## Future Enhancements

### Planned Features
- Machine learning model caching
- Advanced anomaly detection
- Peer-to-peer sync between edge devices
- Enhanced compression algorithms
- Real-time dashboard for offline operations

### Performance Improvements
- Parallel processing pipelines
- Advanced caching strategies
- Optimized storage formats
- Intelligent sync prioritization