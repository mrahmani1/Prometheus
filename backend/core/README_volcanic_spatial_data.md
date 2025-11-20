# Volcanic Spatial Data Connector

## Overview

The Volcanic Spatial Data Connector provides real-time and historical volcanic spatial data integration for the Prometheus AI platform's emergent behavior detection system. It connects to multiple authoritative volcanic databases and provides a unified interface for accessing volcanic spatial data with geological context.

## Features

- **Multi-Source Data Integration**: Connects to USGS, Global Volcanism Program (GVP), NASA EarthData, and other geological survey databases
- **Real-Time Monitoring**: Streams real-time volcanic alerts and activity data
- **Spatial Analysis**: Performs volcanic cluster analysis with emergent property detection
- **Data Quality Assurance**: Validates and scores data quality from multiple sources
- **Intelligent Caching**: Caches spatial data to reduce API calls and improve performance
- **Failover Support**: Automatic failover between data sources for high availability
- **Regional Filtering**: Supports geographic bounding box filtering for focused analysis

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Emergent Behavior Detection System           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│            Volcanic Spatial Data Connector                 │
├─────────────────────────────────────────────────────────────┤
│  • Data Source Management                                  │
│  • Spatial Data Processing                                 │
│  • Cluster Analysis                                        │
│  • Quality Validation                                      │
│  • Caching & Performance                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                Connector Factory                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │    USGS     │ │     GVP     │ │    NASA     │          │
│  │ Connector   │ │ Connector   │ │ Connector   │   ...    │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Installation and Setup

### Prerequisites

- Python 3.11+
- PostgreSQL with PostGIS extension
- Required Python packages (see requirements.txt)

### Configuration

1. **Database Setup**: Ensure PostgreSQL with PostGIS is running and accessible
2. **Configuration File**: Update `backend/config/data_sources.yaml` with your API credentials
3. **Environment Variables**: Set required API keys and credentials

```bash
# Example environment variables
export USGS_API_KEY="your_usgs_api_key"
export NASA_USERNAME="your_nasa_username"
export NASA_PASSWORD="your_nasa_password"
export COPERNICUS_USERNAME="your_copernicus_username"
export COPERNICUS_PASSWORD="your_copernicus_password"
```

### Database Tables

The connector automatically creates the following database tables:

- `volcanic_spatial_data`: Main table for volcanic point data
- Spatial indexes for efficient geographic queries
- Quality and clustering indexes for performance

## Usage

### Basic Usage

```python
from backend.core.volcanic_spatial_data_connector import VolcanicSpatialDataConnector
from backend.core.database_integration import DatabaseIntegration

# Initialize
db_integration = DatabaseIntegration()
await db_integration.initialize()

connector = VolcanicSpatialDataConnector(db_integration)
await connector.initialize()

# Get global volcanic data
spatial_data = await connector.get_real_volcanic_spatial_data()

# spatial_data is a numpy array with shape (n_volcanoes, 4)
# Columns: [latitude, longitude, elevation, activity_score]
```

### Regional Analysis

```python
# Define region bounds (min_lat, min_lon, max_lat, max_lon)
cascade_range = (42, -125, 49, -120)

# Get regional data with activity filtering
regional_data = await connector.get_real_volcanic_spatial_data(
    region_bounds=cascade_range,
    min_activity_level='yellow',  # Only yellow, orange, red alerts
    include_historical=True
)
```

### Cluster Analysis

```python
# Get volcanic clusters with emergent properties
clusters = await connector.get_volcanic_clusters(region_bounds)

for cluster in clusters:
    print(f"Cluster {cluster.cluster_id}:")
    print(f"  Location: {cluster.center_lat:.2f}°N, {cluster.center_lon:.2f}°E")
    print(f"  Volcanoes: {cluster.volcano_count}")
    print(f"  Radius: {cluster.radius_km:.1f} km")
    print(f"  Activity Level: {cluster.activity_level}")
    print(f"  Emergent Properties: {cluster.emergent_properties}")
```

### Integration with Emergent Behavior Detection

```python
# This is how the emergent behavior detection system uses the connector
async def _get_real_volcanic_spatial_data(self) -> Optional[np.ndarray]:
    """Get real volcanic spatial data from database"""
    try:
        if not hasattr(self, '_volcanic_connector'):
            self._volcanic_connector = VolcanicSpatialDataConnector(self._db_integration)
            await self._volcanic_connector.initialize()
        
        spatial_data = await self._volcanic_connector.get_real_volcanic_spatial_data(
            region_bounds=None,  # Global data
            min_activity_level='green',  # Include all activity levels
            include_historical=True
        )
        
        return spatial_data
        
    except Exception as e:
        logger.warning(f"Could not retrieve real volcanic spatial data: {e}")
        return None  # Falls back to synthetic data
```

## Data Sources

### USGS Volcano Hazards Program
- **URL**: https://volcano.wr.usgs.gov/vsc/api/volcanoApi
- **Data Types**: Real-time alerts, volcanic activity, earthquakes, gas emissions
- **Update Frequency**: Real-time to hourly
- **Coverage**: Global, with focus on US volcanoes

### Global Volcanism Program (Smithsonian)
- **URL**: https://volcano.si.edu/api
- **Data Types**: Volcano catalog, eruption history, volcano types
- **Update Frequency**: Weekly to monthly
- **Coverage**: Global comprehensive database

### NASA EarthData
- **URL**: https://api.earthdata.nasa.gov
- **Data Types**: Satellite thermal monitoring, MODIS data
- **Update Frequency**: Daily
- **Coverage**: Global satellite coverage

### Copernicus Sentinel
- **URL**: https://catalogue.dataspace.copernicus.eu
- **Data Types**: Sentinel-2/3 thermal and optical data
- **Update Frequency**: 2-5 days
- **Coverage**: Global

## Data Model

### VolcanicSpatialPoint
```python
@dataclass
class VolcanicSpatialPoint:
    volcano_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float
    volcano_type: str
    last_eruption: Optional[datetime]
    alert_level: str  # 'green', 'yellow', 'orange', 'red'
    tectonic_setting: str
    rock_type: str
    activity_status: str
    spatial_cluster_id: Optional[str]
    uncertainty_radius_km: float
    data_quality_score: float  # 0.0 to 1.0
```

### VolcanicSpatialCluster
```python
@dataclass
class VolcanicSpatialCluster:
    cluster_id: str
    center_lat: float
    center_lon: float
    radius_km: float
    volcano_count: int
    dominant_type: str
    tectonic_setting: str
    activity_level: str
    formation_age_ma: Optional[float]
    magma_composition: str
    emergent_properties: Dict[str, Any]
```

## Configuration

### Data Sources Configuration (`backend/config/data_sources.yaml`)

```yaml
data_sources:
  usgs_volcano_primary:
    type: usgs_volcano
    enabled: true
    name: "USGS Volcano Hazards Program"
    base_url: "https://volcano.wr.usgs.gov/vsc/api/volcanoApi"
    timeout: 30
    max_retries: 3
    rate_limit: 60
    api_key: null  # Set via environment variable

health_monitoring:
  check_interval: 60
  alert_thresholds:
    response_time_ms: 5000
    success_rate: 0.95
    consecutive_failures: 3

failover:
  volcanic_data:
    primary_source: "usgs_volcano_primary"
    fallback_sources: ["gvp_primary"]
    max_failures: 3
    auto_failback: true
```

## Performance Considerations

### Caching Strategy
- **Cache Duration**: 1 hour for spatial data
- **Cache Keys**: Based on region bounds and filter parameters
- **Cache Size**: Limited to 100MB to prevent memory issues

### Rate Limiting
- **USGS**: 60 requests/minute
- **GVP**: 30 requests/minute
- **NASA**: 50 requests/minute
- **Copernicus**: 40 requests/minute

### Database Optimization
- Spatial indexes on latitude/longitude columns
- Composite indexes on alert_level and spatial_cluster_id
- Regular VACUUM and ANALYZE operations recommended

## Error Handling and Resilience

### Graceful Degradation
1. **Primary Source Failure**: Automatic failover to secondary sources
2. **All Sources Unavailable**: Returns `None` to trigger synthetic data generation
3. **Partial Data**: Continues processing with available data, logs warnings
4. **Network Issues**: Exponential backoff retry with circuit breaker pattern

### Data Quality Validation
- Coordinate validation (latitude: -90 to 90, longitude: -180 to 180)
- Alert level validation against known values
- Magnitude and measurement range validation
- Quality scoring based on data completeness and source reliability

## Testing

### Integration Tests
```bash
# Run integration tests
python backend/core/test_volcanic_spatial_integration.py
```

### Usage Examples
```bash
# Run usage examples
python backend/core/examples/volcanic_spatial_data_example.py
```

### Test Coverage
- Basic connectivity and initialization
- Data retrieval from multiple sources
- Regional filtering and activity level filtering
- Cluster analysis and emergent properties
- Cache functionality and performance
- Error handling and fallback mechanisms

## Monitoring and Logging

### Health Monitoring
- Automatic health checks every 60 seconds
- Response time monitoring
- Success rate tracking
- Failure count and consecutive failure detection

### Logging
- Structured JSON logging with correlation IDs
- Log levels: DEBUG, INFO, WARNING, ERROR
- Separate log files for different components
- Log rotation and retention policies

### Metrics
- Data retrieval latency
- Cache hit/miss ratios
- API call success rates
- Data quality scores
- Cluster analysis performance

## Troubleshooting

### Common Issues

1. **No Data Retrieved**
   - Check API credentials and network connectivity
   - Verify data source endpoints are accessible
   - Check rate limiting and quota usage

2. **Poor Performance**
   - Review cache configuration and hit rates
   - Check database indexes and query performance
   - Monitor API response times

3. **Data Quality Issues**
   - Review data validation rules
   - Check source data quality scores
   - Verify coordinate and measurement ranges

4. **Integration Failures**
   - Check database connectivity and schema
   - Verify module imports and dependencies
   - Review error logs for specific failure points

### Debug Mode
```python
# Enable debug logging
import logging
logging.getLogger('backend.core.volcanic_spatial_data_connector').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features
- **Machine Learning Integration**: Predictive modeling for volcanic activity
- **Real-Time Streaming**: WebSocket connections for live data feeds
- **Advanced Clustering**: Hierarchical and density-based clustering algorithms
- **Geochemical Integration**: Rock composition and magma chemistry data
- **3D Visualization**: Integration with Vulcan UI for 3D spatial visualization

### API Extensions
- RESTful API endpoints for external access
- GraphQL interface for flexible data queries
- Webhook support for real-time notifications
- Bulk data export capabilities

## Contributing

### Development Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up PostgreSQL with PostGIS
4. Configure environment variables
5. Run tests to verify setup

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for all public methods
- Include unit tests for new features

### Pull Request Process
1. Create feature branch from main
2. Implement changes with tests
3. Update documentation
4. Submit pull request with description
5. Address review feedback

## License

This project is part of the Prometheus AI platform and follows the same licensing terms.

## Support

For technical support and questions:
- Create an issue in the project repository
- Contact the development team
- Review the troubleshooting section above

---

*Last updated: January 2025*