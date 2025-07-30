# PostgreSQL with PostGIS Implementation

## Overview

This document describes the advanced PostgreSQL with PostGIS implementation for the Prometheus AI Platform, providing sophisticated spatial-temporal database operations for geological data analysis.

## Features Implemented

### 1. Advanced Spatial Indexing for Geological Coordinate Systems

#### Custom Spatial Reference Systems
- **UTM Zone 10N WGS84 Geological (SRID: 9001)**: Custom projection optimized for geological analysis
- **3D Point Support**: Full 3D spatial indexing with elevation and depth coordinates
- **Multi-Coordinate System Support**: WGS84, UTM, and custom geological projections

#### Spatial Index Types
```sql
-- Standard 2D spatial index
CREATE INDEX idx_locations_geom ON geological.locations USING GIST (geom);

-- UTM coordinate system index for precise distance calculations
CREATE INDEX idx_locations_geom_utm ON geological.locations USING GIST (geom_utm);

-- 3D spatial index including elevation/depth
CREATE INDEX idx_locations_geom_3d ON geological.locations USING GIST (geom_3d);

-- Composite spatial-temporal indexes
CREATE INDEX idx_seismic_events_spatio_temporal ON geological.seismic_events 
USING GIST (
    (SELECT geom FROM geological.locations WHERE id = location_id), 
    event_time
);
```

### 2. Complex Spatial-Temporal Queries for Geological Processes

#### Spatial-Temporal Event Search
```python
# Find events within spatial-temporal window
events = await manager.find_events_in_spacetime(
    SpatialTemporalQuery(
        center_lat=46.1912,
        center_lon=-122.1944,
        spatial_radius=50000,  # 50km radius
        time_center=datetime.now(),
        time_window=timedelta(days=30),
        event_types=["volcanic", "earthquake"]
    )
)
```

#### Advanced Spatial Analysis Views
- **Seismic Spatial-Temporal Analysis**: Clustering analysis with distance and time relationships
- **Deformation Velocity Analysis**: Ground movement velocity calculations between measurements
- **Multi-Parameter Correlation**: Cross-correlation between seismic, deformation, and thermal data

### 3. Materialized Views for High-Performance Geological Analysis

#### Seismic Activity Summary
```sql
CREATE MATERIALIZED VIEW geological.mv_seismic_activity_summary AS
SELECT 
    DATE_TRUNC('day', se.event_time) as activity_date,
    se.event_type,
    COUNT(*) as event_count,
    AVG(se.magnitude) as avg_magnitude,
    MAX(se.magnitude) as max_magnitude,
    MIN(se.magnitude) as min_magnitude,
    STDDEV(se.magnitude) as magnitude_stddev,
    AVG(se.depth) as avg_depth,
    ST_Centroid(ST_Collect(l.geom)) as centroid_location,
    ST_ConvexHull(ST_Collect(l.geom)) as activity_envelope
FROM geological.seismic_events se
JOIN geological.locations l ON se.location_id = l.id
WHERE se.event_time >= NOW() - INTERVAL '1 year'
GROUP BY DATE_TRUNC('day', se.event_time), se.event_type;
```

#### Deformation Analysis
- **Monthly aggregation** of deformation measurements
- **Statistical analysis** including mean, max, and standard deviation
- **Multi-component displacement** tracking (X, Y, Z components)

#### Thermal Anomaly Detection
- **Automated anomaly classification** (NORMAL, MODERATE_ANOMALY, HIGH_ANOMALY)
- **Statistical baseline comparison** using historical data
- **Weekly temporal aggregation** for trend analysis

#### Multi-Parameter Geological Correlation
- **Cross-correlation analysis** between different measurement types
- **Temporal alignment** of diverse geological data
- **Comprehensive location-based analysis**

### 4. Stored Procedures for Common Geological Calculations

#### Distance Calculations
```sql
-- Calculate distance between geological locations
SELECT geological.calculate_distance(loc1_id, loc2_id, 'spheroidal');
```

#### Seismic Calculations
```sql
-- Calculate seismic moment from magnitude
SELECT geological.calculate_seismic_moment(5.0, 'Mw');

-- Calculate stress drop from seismic parameters
SELECT geological.calculate_stress_drop(seismic_moment, rupture_radius);
```

#### Volcanic Calculations
```sql
-- Estimate Volcanic Explosivity Index
SELECT geological.estimate_vei(eruption_volume, eruption_height);

-- Calculate magma ascent rate
SELECT geological.calculate_magma_ascent_rate(depth_change, time_interval);
```

#### Hazard Assessment
```sql
-- Calculate geological hazard probability
SELECT geological.calculate_hazard_probability(
    location_geom, 
    'volcanic', 
    '1 year'::INTERVAL
);
```

## Database Schema

### Core Tables

#### Geological Locations
```sql
CREATE TABLE geological.locations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    elevation DOUBLE PRECISION,
    depth DOUBLE PRECISION,
    coordinate_system VARCHAR(50) DEFAULT 'WGS84',
    geom GEOMETRY(POINT, 4326),
    geom_utm GEOMETRY(POINT, 9001),
    geom_3d GEOMETRY(POINTZ, 4326),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Seismic Events
```sql
CREATE TABLE geological.seismic_events (
    id SERIAL PRIMARY KEY,
    event_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location_id INTEGER REFERENCES geological.locations(id),
    magnitude DOUBLE PRECISION NOT NULL,
    depth DOUBLE PRECISION NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    confidence DOUBLE PRECISION DEFAULT 1.0,
    waveform_data BYTEA,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Deformation Measurements
```sql
CREATE TABLE geological.deformation_measurements (
    id SERIAL PRIMARY KEY,
    measurement_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location_id INTEGER REFERENCES geological.locations(id),
    displacement_x DOUBLE PRECISION NOT NULL,
    displacement_y DOUBLE PRECISION NOT NULL,
    displacement_z DOUBLE PRECISION NOT NULL,
    uncertainty_x DOUBLE PRECISION,
    uncertainty_y DOUBLE PRECISION,
    uncertainty_z DOUBLE PRECISION,
    measurement_type VARCHAR(50) NOT NULL,
    reference_frame VARCHAR(50) DEFAULT 'ITRF2014',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Python API Usage

### Basic Operations

#### Initialize Manager
```python
from core.database.postgis_manager import PostGISManager

manager = PostGISManager("postgresql://user:pass@localhost:5432/prometheus")
await manager.initialize()
```

#### Create Geological Location
```python
location = GeologicalLocation(
    id=None,
    name="Mount St. Helens",
    latitude=46.1912,
    longitude=-122.1944,
    elevation=2549.0,
    depth=None
)
location_id = await manager.create_geological_location(location)
```

#### Insert Seismic Event
```python
event = SeismicEvent(
    id=None,
    event_time=datetime.now(),
    location_id=location_id,
    magnitude=3.2,
    depth=5000.0,
    event_type="volcanic",
    confidence=0.95,
    metadata={"source": "USGS", "quality": "A"}
)
await manager.insert_seismic_event(event)
```

### Advanced Analysis

#### Spatial-Temporal Queries
```python
# Find events in space-time window
query_params = SpatialTemporalQuery(
    center_lat=46.1912,
    center_lon=-122.1944,
    spatial_radius=50000,
    time_center=datetime.now(),
    time_window=timedelta(days=30),
    event_types=["volcanic", "earthquake"]
)
events = await manager.find_events_in_spacetime(query_params)
```

#### Materialized View Analysis
```python
# Get seismic activity summary
summary = await manager.get_seismic_activity_summary(days_back=30)

# Get deformation analysis
deformation = await manager.get_deformation_analysis(location_id=1, months_back=12)

# Get thermal anomalies
anomalies = await manager.get_thermal_anomalies(anomaly_level="HIGH_ANOMALY")

# Get geological correlation
correlation = await manager.get_geological_correlation(location_id=1)
```

#### Geological Calculations
```python
# Calculate distance between locations
distance = await manager.calculate_geological_distance(loc1_id, loc2_id, 'spheroidal')

# Calculate seismic moment
moment = await manager.calculate_seismic_moment(5.0, 'Mw')

# Estimate VEI
vei = await manager.estimate_vei(eruption_volume=1e8, eruption_height=20000.0)

# Calculate hazard probability
probability = await manager.calculate_hazard_probability(
    46.1912, -122.1944, 'volcanic', timedelta(days=365)
)
```

## Performance Optimization

### Indexing Strategy

#### Spatial Indexes
- **GIST indexes** for all geometry columns
- **Multi-coordinate system** indexing (WGS84, UTM, 3D)
- **Composite spatial-temporal** indexes for complex queries

#### Temporal Indexes
- **B-tree indexes** on all timestamp columns
- **Composite indexes** combining temporal and scalar columns
- **Partial indexes** for frequently queried subsets

#### Specialized Indexes
- **GIN indexes** for JSONB metadata columns
- **Multi-column indexes** for common query patterns
- **Conditional indexes** for specific data subsets

### Materialized View Management

#### Refresh Strategy
```python
# Automatic refresh schedule
refresh_schedule = {
    "mv_seismic_activity_summary": timedelta(hours=1),
    "mv_deformation_analysis": timedelta(hours=6),
    "mv_thermal_anomalies": timedelta(hours=2),
    "mv_geological_correlation": timedelta(hours=12)
}

# Manual refresh
await manager.refresh_materialized_views()
```

#### Concurrent Refresh
- **CONCURRENTLY** option for zero-downtime updates
- **Unique indexes** required for concurrent refresh
- **Automated scheduling** for regular updates

### Query Optimization

#### Spatial Query Optimization
- **Spatial index usage** for all geometric operations
- **Bounding box pre-filtering** for complex spatial queries
- **Coordinate system optimization** for distance calculations

#### Temporal Query Optimization
- **Time-based partitioning** for large tables
- **Index-only scans** for temporal range queries
- **Parallel query execution** for large datasets

## Configuration

### Environment Variables
```bash
# Database connection
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=prometheus
POSTGRES_USER=prometheus
POSTGRES_PASSWORD=password

# Connection pool
POSTGRES_MIN_POOL_SIZE=5
POSTGRES_MAX_POOL_SIZE=20
POSTGRES_COMMAND_TIMEOUT=60

# Performance tuning
POSTGRES_WORK_MEM_MB=256
POSTGRES_SHARED_BUFFERS_MB=1024

# Data retention
SEISMIC_RETENTION_DAYS=3650
DEFORMATION_RETENTION_DAYS=1825
THERMAL_RETENTION_DAYS=1095

# Materialized views
POSTGRES_AUTO_REFRESH=true
POSTGRES_REFRESH_INTERVAL=6
```

### Configuration Class
```python
from core.database.postgis_config import PostGISConfig

config = PostGISConfig()
print(config.connection_string)
print(config.get_materialized_view_refresh_schedule())
print(config.get_spatial_index_configurations())
```

## Testing

### Unit Tests
```bash
# Run unit tests
cd services/helios-core
pytest tests/test_postgis_manager.py -v
```

### Integration Tests
```bash
# Run integration tests (requires database)
pytest tests/test_postgis_manager.py::TestPostGISIntegration -v --integration
```

### Performance Tests
```bash
# Run performance benchmarks
pytest tests/test_postgis_performance.py -v --benchmark
```

## Monitoring and Maintenance

### Database Statistics
```python
# Get comprehensive database statistics
stats = await manager.get_database_statistics()
print(f"Table counts: {stats}")
print(f"Index usage: {stats['top_indexes']}")
print(f"Materialized views: {stats['materialized_views']}")
```

### Performance Monitoring
- **Query execution time** tracking
- **Index usage statistics** monitoring
- **Materialized view freshness** checking
- **Connection pool utilization** monitoring

### Maintenance Tasks
- **Automated VACUUM** and ANALYZE
- **Index maintenance** and rebuilding
- **Materialized view refresh** scheduling
- **Data archival** and cleanup

## Security Considerations

### Access Control
- **Role-based permissions** for different user types
- **Schema-level security** for data isolation
- **Function execution permissions** for stored procedures

### Data Protection
- **Connection encryption** using SSL/TLS
- **Password security** using environment variables
- **Audit logging** for sensitive operations

### Network Security
- **Connection pooling** with secure credentials
- **IP-based access restrictions**
- **VPN/private network** deployment

## Troubleshooting

### Common Issues

#### Performance Issues
- **Slow spatial queries**: Check spatial index usage
- **Materialized view lag**: Verify refresh schedule
- **Connection pool exhaustion**: Increase pool size

#### Data Issues
- **Coordinate system errors**: Verify SRID consistency
- **Temporal alignment**: Check timezone handling
- **Missing spatial data**: Validate geometry creation

#### Configuration Issues
- **Connection failures**: Verify credentials and network
- **Permission errors**: Check role assignments
- **Extension errors**: Ensure PostGIS installation

### Debugging Tools
```python
# Enable query logging
import logging
logging.getLogger('asyncpg').setLevel(logging.DEBUG)

# Check spatial index usage
await manager.execute_spatial_query("""
    EXPLAIN (ANALYZE, BUFFERS) 
    SELECT * FROM geological.locations 
    WHERE ST_DWithin(geom, ST_MakePoint(-122.1944, 46.1912), 50000)
""")
```

## Future Enhancements

### Planned Features
- **Temporal partitioning** for very large datasets
- **Distributed spatial queries** across multiple nodes
- **Real-time spatial triggers** for immediate analysis
- **Advanced spatial statistics** and clustering algorithms

### Performance Improvements
- **Parallel spatial operations** for large datasets
- **Spatial data compression** for storage optimization
- **Cached spatial calculations** for frequently accessed data
- **GPU-accelerated spatial operations** for complex analysis

### Integration Enhancements
- **Time-series database integration** for sensor data
- **Graph database connectivity** for relationship analysis
- **Machine learning model integration** for predictive analysis
- **Real-time streaming** data ingestion and processing