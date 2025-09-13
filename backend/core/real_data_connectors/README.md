# Real Data Connectors Framework

The Real Data Connectors Framework provides authenticated access to real geological data sources, replacing all placeholder and mock data implementations with authentic data streams from authoritative sources.

## Overview

This framework systematically eliminates placeholder implementations across the Prometheus AI Platform by providing:

- **Authenticated Connectors**: Direct access to IRIS/EarthScope, USGS, NASA EarthData, Copernicus, ASF-Hyp3, and geochemical databases
- **Real-Time Streaming**: Continuous data ingestion with configurable intervals
- **Health Monitoring**: Comprehensive monitoring with automated failover capabilities
- **Data Quality Validation**: Built-in validation and quality assessment
- **Standardized Interface**: Consistent API across all data sources

## Supported Data Sources

### 1. IRIS/EarthScope Seismic Data (`IRISConnector`)
- **Real-time earthquake events** from global seismic networks
- **Continuous waveform data** from broadband stations
- **Station metadata** and instrument information
- **Quality assessment** based on location uncertainty and phase counts

### 2. USGS Volcano Hazards Program (`USGSVolcanoConnector`)
- **Volcanic alerts** with real-time status updates
- **Volcanic activity reports** and eruption notifications
- **Volcanic earthquake monitoring** from observatory networks
- **Gas emission measurements** from monitoring stations

### 3. NASA EarthData (`NASAEarthDataConnector`)
- **MODIS thermal anomalies** for fire and volcanic activity detection
- **VIIRS fire detection** with high temporal resolution
- **Landsat thermal infrared** data for surface temperature monitoring
- **Satellite granule metadata** for historical analysis

### 4. Copernicus Satellites (`CopernicusConnector`)
- **Sentinel-1 SAR data** for interferometric analysis
- **Sentinel-2 optical imagery** with low cloud cover filtering
- **Sentinel-3 thermal data** for land surface temperature
- **Sentinel-5P atmospheric data** for volcanic gas monitoring

### 5. ASF-Hyp3 InSAR (`ASFHyp3Connector`)
- **InSAR interferograms** with deformation measurements
- **Ground motion time series** from persistent scatterer analysis
- **Processing job monitoring** with automated result retrieval
- **Quality metrics** including coherence and unwrapping quality

### 6. Geochemical Databases (`GeochemConnector`)
- **EarthChem** comprehensive geochemical database
- **GEOROC** volcanic rock compositions
- **NAVDAT** North American volcanic database
- **PetDB** marine igneous petrology database

### 7. National Geological Surveys (`GeologicalSurveyConnector`)
- **USGS** geological formations and mineral deposits
- **BGS** borehole data and geological maps
- **GSC** geochronology and geological surveys
- **GA** geophysical surveys and mineral resources

## Quick Start

### Basic Usage

```python
import asyncio
from backend.core.real_data_connectors import ConnectorFactory

async def main():
    # Initialize connector factory
    factory = ConnectorFactory("backend/config/data_sources.yaml")
    await factory.initialize()
    
    # Get real-time seismic data
    iris_connector = factory.get_active_connector('seismic_data')
    async for observation in iris_connector.fetch_real_time_data(min_magnitude=4.0):
        print(f"Earthquake M{observation.value['magnitude']} at "
              f"{observation.latitude}, {observation.longitude}")
    
    await factory.shutdown()

asyncio.run(main())
```

### Configuration

Create a `data_sources.yaml` configuration file:

```yaml
data_sources:
  iris_primary:
    type: iris
    name: "IRIS/EarthScope Seismic Data"
    base_url: "http://service.iris.edu"
    enabled: true
    timeout: 30
    max_retries: 3
    rate_limit: 100
    
  usgs_volcano_primary:
    type: usgs_volcano
    name: "USGS Volcano Hazards Program"
    base_url: "https://api.usgs.gov/volcano"
    enabled: true
    api_key: "${USGS_API_KEY}"
    
health_monitoring:
  check_interval: 60
  alert_thresholds:
    response_time_ms: 5000
    success_rate: 0.95
    consecutive_failures: 3
```

## Architecture

### Core Components

```
Real Data Connectors Framework
├── BaseDataConnector          # Abstract base class
├── Specific Connectors        # IRIS, USGS, NASA, etc.
├── DataSourceHealthMonitor    # Health monitoring and failover
├── ConnectorFactory          # Factory and configuration management
└── GeologicalObservation     # Standardized data model
```

### Data Flow

```
External APIs → Connectors → Health Monitor → Quality Validation → Standardized Output
```

## Health Monitoring and Failover

The framework includes comprehensive health monitoring:

### Health Metrics
- **Response Time**: API response latency monitoring
- **Success Rate**: Percentage of successful requests
- **Error Count**: Total and consecutive failure tracking
- **Uptime**: Service availability percentage

### Automatic Failover
- **Primary/Fallback Sources**: Configurable failover chains
- **Failure Thresholds**: Customizable failure detection
- **Auto-Recovery**: Automatic failback when primary recovers
- **Manual Override**: Administrative failover controls

### Alerting
```python
# Add custom alert handler
async def alert_handler(source_id, status, alert_data):
    print(f"Alert: {source_id} status changed to {status}")
    # Send to Slack, PagerDuty, etc.

factory.health_monitor.add_alert_callback(alert_handler)
```

## Data Quality Validation

Each connector implements comprehensive quality validation:

### Validation Checks
- **Coordinate Validation**: Latitude/longitude bounds checking
- **Value Range Validation**: Reasonable ranges for measurements
- **Temporal Validation**: Timestamp consistency and ordering
- **Cross-Source Validation**: Consistency across multiple sources

### Quality Flags
- **Data Quality Issues**: Missing data, high uncertainty
- **Instrument Problems**: Clipping, flat lines, calibration issues
- **Processing Issues**: Algorithm failures, poor convergence

## API Reference

### BaseDataConnector

```python
class BaseDataConnector:
    async def connect() -> ConnectionStatus
    async def disconnect()
    async def fetch_real_time_data(**kwargs) -> AsyncIterator[GeologicalObservation]
    async def fetch_historical_data(start_time, end_time, **kwargs) -> List[GeologicalObservation]
    async def validate_data_quality(observation) -> bool
    async def health_check() -> Dict[str, Any]
```

### GeologicalObservation

```python
@dataclass
class GeologicalObservation:
    observation_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    elevation: Optional[float]
    observation_type: str
    value: Dict[str, Any]
    uncertainty: Optional[float]
    data_source: str
    quality_flags: List[str]
    metadata: Dict[str, Any]
```

### ConnectorFactory

```python
class ConnectorFactory:
    async def initialize()
    def get_connector(source_id: str) -> BaseDataConnector
    def get_active_connector(service_name: str) -> BaseDataConnector
    async def add_connector(source_id: str, connector_type: str, config: Dict)
    async def remove_connector(source_id: str)
    async def test_connector(source_id: str) -> Dict[str, Any]
    def get_health_report() -> Dict[str, Any]
```

## Authentication

### Environment Variables

Set the following environment variables for authenticated access:

```bash
# NASA EarthData
export NASA_EARTHDATA_USERNAME="your_username"
export NASA_EARTHDATA_PASSWORD="your_password"

# Copernicus Data Space
export COPERNICUS_USERNAME="your_username"
export COPERNICUS_PASSWORD="your_password"

# ASF-Hyp3
export ASF_USERNAME="your_nasa_username"
export ASF_PASSWORD="your_nasa_password"

# API Keys
export USGS_API_KEY="your_usgs_key"
export EARTHCHEM_API_KEY="your_earthchem_key"
```

### Account Setup

1. **NASA EarthData**: Register at https://urs.earthdata.nasa.gov/
2. **Copernicus**: Register at https://dataspace.copernicus.eu/
3. **ASF-Hyp3**: Uses NASA EarthData credentials
4. **USGS**: Request API key from USGS Web Services
5. **EarthChem**: Register at https://www.earthchem.org/

## Performance Optimization

### Connection Pooling
```python
# Configure connection pool size
factory = ConnectorFactory()
factory.connection_pool_size = 20
factory.max_concurrent_requests = 100
```

### Rate Limiting
```python
# Automatic rate limiting per source
config = DataSourceConfig(
    source_id="example",
    rate_limit=60,  # requests per minute
    timeout=30,
    max_retries=3
)
```

### Caching
```python
# Enable response caching
factory.enable_caching = True
factory.cache_ttl_seconds = 300  # 5 minutes
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest backend/tests/test_real_data_connectors.py -v

# Run specific test categories
python -m pytest backend/tests/test_real_data_connectors.py::TestIRISConnector -v
python -m pytest backend/tests/test_real_data_connectors.py::TestHealthMonitor -v

# Run with coverage
python -m pytest backend/tests/test_real_data_connectors.py --cov=backend.core.real_data_connectors
```

## Examples

See `backend/examples/real_data_connectors_example.py` for comprehensive usage examples including:

- Real-time data streaming from multiple sources
- Historical data fetching with time ranges
- Health monitoring and failover demonstration
- Dynamic connector management
- Custom alert handling

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify environment variables are set correctly
   - Check account credentials and permissions
   - Ensure API keys are valid and not expired

2. **Rate Limiting**
   - Reduce `rate_limit` in configuration
   - Increase `fetch_interval` for real-time streams
   - Monitor rate limit headers in responses

3. **Connection Timeouts**
   - Increase `timeout` values in configuration
   - Check network connectivity to external APIs
   - Verify firewall and proxy settings

4. **Data Quality Issues**
   - Review quality flags in observations
   - Adjust validation thresholds if needed
   - Cross-validate with multiple sources

### Logging

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('backend.core.real_data_connectors')
logger.setLevel(logging.DEBUG)
```

## Contributing

### Adding New Connectors

1. Create a new connector class inheriting from `BaseDataConnector`
2. Implement required abstract methods
3. Add connector type to `ConnectorFactory.connector_types`
4. Add default configuration to `ConnectorFactory.default_configs`
5. Write comprehensive tests
6. Update documentation

### Example New Connector

```python
class CustomConnector(BaseDataConnector):
    async def _authenticate(self) -> bool:
        # Implement authentication logic
        pass
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        # Return authentication headers
        pass
    
    async def fetch_real_time_data(self, **kwargs):
        # Implement real-time data fetching
        pass
    
    async def fetch_historical_data(self, start_time, end_time, **kwargs):
        # Implement historical data fetching
        pass
    
    async def validate_data_quality(self, observation):
        # Implement data quality validation
        pass
```

## License

This framework is part of the Prometheus AI Platform and follows the project's licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the test suite for usage patterns
3. Examine the example code for implementation details
4. Create an issue in the project repository