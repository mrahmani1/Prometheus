# Database Schema Documentation
## Prometheus AI Platform - Comprehensive Geological Database Architecture

---

## 🏗️ Overview

The Prometheus AI Platform utilizes a sophisticated multi-database architecture designed to handle diverse geological, geophysical, and machine learning data types. The database schema has been significantly enhanced in Phase 2 to include 50+ specialized geological tables, comprehensive rock mechanical properties, volcanic chamber modeling, and advanced ML model lifecycle management.

## 🗄️ Database Architecture

### Multi-Database System
- **PostgreSQL + PostGIS + TimeScaleDB**: Primary geospatial and geological data with time-series optimization
- **InfluxDB**: High-frequency time-series data for real-time sensor readings
- **Neo4j**: Graph database for geological process relationships and knowledge graphs
- **Redis**: Caching, session management, and real-time data processing
- **Model Database System**: ML model versioning and lifecycle management with multi-backend storage

---

## 📊 Core Database Schema

### 1. Geological Structures Table

The foundation table for all geological data with comprehensive age information and metadata.

```sql
CREATE TABLE geological_structures (
    id SERIAL PRIMARY KEY,
    structure_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,  -- 3D coordinates with elevation
    geometry GEOMETRY(GEOMETRYZ, 4326),         -- Complex 3D geological structures
    structure_type VARCHAR(100) NOT NULL,
    lithology VARCHAR(200),
    
    -- Age information (multiple dating methods)
    age REAL CHECK (age >= 0 AND age <= 4500),  -- Primary age in Ma
    properties JSONB DEFAULT '{}'::jsonb,       -- Flexible age data storage
    
    -- Geometric properties
    strike REAL CHECK (strike >= 0 AND strike <= 360),
    dip REAL CHECK (dip >= 0 AND dip <= 90),
    
    -- Data provenance and quality
    data_source VARCHAR(200) NOT NULL,
    data_quality_grade CHAR(1) DEFAULT 'C' CHECK (data_quality_grade IN ('A', 'B', 'C', 'D')),
    confidence_level REAL DEFAULT 0.5 CHECK (confidence_level >= 0 AND confidence_level <= 1),
    
    -- Temporal tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Age Properties JSONB Structure:**
```json
{
    "formation_age": 150.5,           // Stratigraphic age (Ma)
    "radiometric_age": 148.2,         // U-Pb, Ar-Ar, K-Ar age (Ma)
    "stratigraphic_age": "Jurassic",   // Geological period
    "age_uncertainty": 2.5,           // Age uncertainty (Ma)
    "dating_method": "U-Pb zircon",   // Dating methodology
    "age_reference": "Smith et al. 2023", // Literature reference
    "geological_unit": "Morrison Formation", // Formation name
    "data_quality": "high",           // Quality assessment
    "confidence_level": 0.85,         // Numerical confidence (0-1)
    "isotope_system": "U-Pb",         // Isotopic system used
    "material_dated": "zircon",       // Mineral/material analyzed
    "analytical_precision": 1.2,     // Analytical precision (Ma)
    "geological_context": "intrusive", // Geological setting
    "sample_description": "granitic pluton" // Sample details
}
```

### 2. Rock Mechanical Properties Table

Comprehensive rock mechanical properties for geotechnical analysis and cap rock stability assessment.

```sql
CREATE TABLE rock_mechanical_properties (
    id SERIAL PRIMARY KEY,
    formation_id VARCHAR(100) NOT NULL,
    lithology VARCHAR(100) NOT NULL,
    unconfined_compressive_strength_mpa REAL CHECK (unconfined_compressive_strength_mpa > 0),
    tensile_strength_mpa REAL CHECK (tensile_strength_mpa > 0),
    cohesion_mpa REAL CHECK (cohesion_mpa >= 0),
    friction_angle_deg REAL CHECK (friction_angle_deg BETWEEN 0 AND 90),
    elastic_modulus_gpa REAL CHECK (elastic_modulus_gpa > 0),
    poisson_ratio REAL CHECK (poisson_ratio BETWEEN -1 AND 0.5),
    permeability_m2 REAL CHECK (permeability_m2 > 0),
    thermal_conductivity_w_m_k REAL CHECK (thermal_conductivity_w_m_k > 0),
    thermal_expansion_per_k REAL CHECK (thermal_expansion_per_k > 0),
    porosity REAL CHECK (porosity BETWEEN 0 AND 1),
    fracture_density_per_m3 REAL CHECK (fracture_density_per_m3 >= 0),
    data_quality_grade CHAR(1) CHECK (data_quality_grade IN ('A', 'B', 'C', 'D')),
    test_standard VARCHAR(20) CHECK (test_standard IN ('ASTM', 'ISRM', 'BS', 'DIN', 'ISO')),
    sample_location GEOGRAPHY(POINTZ, 4326),
    collected_date DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 3. Volcanic Chambers Table

Advanced volcanic chamber modeling with comprehensive geometric and thermodynamic properties.

```sql
CREATE TABLE volcanic_chambers (
    id SERIAL PRIMARY KEY,
    chamber_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    chamber_volume_km3 REAL CHECK (chamber_volume_km3 > 0),
    chamber_depth_km REAL CHECK (chamber_depth_km > 0),
    chamber_width_km REAL CHECK (chamber_width_km > 0),
    chamber_length_km REAL CHECK (chamber_length_km > 0),
    chamber_height_km REAL CHECK (chamber_height_km > 0),
    roof_depth_km REAL CHECK (roof_depth_km > 0),
    chamber_shape VARCHAR(50) CHECK (chamber_shape IN ('spherical', 'ellipsoid', 'cylindrical', 'irregular')),
    connectivity_factor REAL CHECK (connectivity_factor BETWEEN 0 AND 1),
    melt_fraction REAL CHECK (melt_fraction BETWEEN 0 AND 1),
    temperature_c REAL CHECK (temperature_c > 0),
    pressure_mpa REAL CHECK (pressure_mpa > 0),
    volatile_content JSONB DEFAULT '{}'::jsonb,
    last_updated TIMESTAMP DEFAULT NOW(),
    data_source VARCHAR(100),
    confidence_level REAL CHECK (confidence_level BETWEEN 0 AND 1) DEFAULT 0.5
);
```

### 4. Critical Failure Zones Table

Identification and analysis of critical failure zones for volcanic hazard assessment.

```sql
CREATE TABLE critical_failure_zones (
    id SERIAL PRIMARY KEY,
    zone_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    critical_point GEOGRAPHY(POINTZ, 4326) NOT NULL,
    depth_km REAL CHECK (depth_km > 0),
    failure_mechanism VARCHAR(50) CHECK (failure_mechanism IN ('mohr_coulomb', 'tensile', 'thermal', 'combined')),
    stress_ratio REAL CHECK (stress_ratio >= 0),
    confidence_level REAL CHECK (confidence_level BETWEEN 0 AND 1),
    historical_failures INTEGER DEFAULT 0,
    last_failure_date DATE,
    monitoring_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🤖 Model Database System

### Model Metadata Table

Comprehensive ML model lifecycle management with versioning and performance tracking.

```sql
CREATE TABLE model_metadata (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) UNIQUE NOT NULL,
    model_name VARCHAR(200) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    description TEXT,
    created_by VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    tags TEXT[],
    model_category VARCHAR(100),
    application_domain VARCHAR(100)
);

CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version_id VARCHAR(100) UNIQUE NOT NULL,
    model_id VARCHAR(100) REFERENCES model_metadata(model_id),
    version VARCHAR(50) NOT NULL,
    parent_version_id VARCHAR(100),
    storage_backend VARCHAR(50) CHECK (storage_backend IN ('local', 's3', 'minio')),
    model_path TEXT NOT NULL,
    model_size_bytes BIGINT,
    validation_status VARCHAR(50) DEFAULT 'pending',
    deployment_status VARCHAR(50) DEFAULT 'development',
    performance_metrics JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    change_summary TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 📡 Real-Time Data Tables

### Seismic Data (TimeScaleDB)

```sql
CREATE TABLE earthquakes (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    magnitude REAL NOT NULL,
    depth_km REAL,
    occurrence_time TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50),
    magnitude_type VARCHAR(10),
    data_source VARCHAR(50),
    uncertainty_km REAL,
    quality_grade CHAR(1) CHECK (quality_grade IN ('A', 'B', 'C', 'D')),
    focal_mechanism JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('earthquakes', 'occurrence_time');
```

### Volcanic Monitoring (TimeScaleDB)

```sql
CREATE TABLE volcanic_monitoring (
    id SERIAL PRIMARY KEY,
    volcano_id VARCHAR(20) REFERENCES volcanoes(volcano_id),
    timestamp TIMESTAMPTZ NOT NULL,
    seismic_activity_level REAL,
    gas_emission_so2_tons_day REAL,
    thermal_anomaly_count INTEGER,
    deformation_rate_cm_yr REAL,
    ground_temperature_celsius REAL,
    monitoring_station_count INTEGER,
    data_quality_score REAL CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
    alert_changes JSONB,
    last_update_timestamp TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('volcanic_monitoring', 'timestamp');
```

### InSAR Deformation Data (TimeScaleDB)

```sql
CREATE TABLE insar_deformation (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    line_of_sight_velocity_mm_yr REAL,
    vertical_velocity_mm_yr REAL,
    east_velocity_mm_yr REAL,
    north_velocity_mm_yr REAL,
    coherence_quality REAL CHECK (coherence_quality >= 0 AND coherence_quality <= 1),
    temporal_baseline_days INTEGER,
    perpendicular_baseline_m REAL,
    satellite_mission VARCHAR(50),
    acquisition_date TIMESTAMPTZ,
    processing_level VARCHAR(20),
    uncertainty_mm_yr REAL
);

SELECT create_hypertable('insar_deformation', 'acquisition_date');
```

---

## 🔍 Indexing Strategy

### Spatial Indexes (PostGIS)
```sql
-- Geological structures spatial indexing
CREATE INDEX idx_geological_structures_spatial ON geological_structures USING GIST (location);
CREATE INDEX idx_geological_structures_geometry ON geological_structures USING GIST (geometry);

-- Volcanic chambers spatial indexing
CREATE INDEX idx_volcanic_chambers_location ON volcanic_chambers USING GIST (location);

-- Critical failure zones spatial indexing
CREATE INDEX idx_critical_zones_location ON critical_failure_zones USING GIST (location);
```

### JSONB Indexes for Flexible Data
```sql
-- GIN index for JSONB properties (age data)
CREATE INDEX idx_geological_structures_properties ON geological_structures USING GIN (properties);

-- Specialized indexes for age queries
CREATE INDEX idx_geological_structures_age_exists ON geological_structures 
WHERE age IS NOT NULL OR properties ? 'formation_age' OR properties ? 'radiometric_age';

CREATE INDEX idx_geological_structures_age_confidence ON geological_structures 
((properties->>'confidence_level')::float, age) 
WHERE (properties->>'confidence_level') IS NOT NULL;
```

### Time-Series Indexes (TimeScaleDB)
```sql
-- Earthquake data indexing
CREATE INDEX idx_earthquakes_spatial ON earthquakes USING GIST (location);
CREATE INDEX idx_earthquakes_magnitude ON earthquakes (magnitude);
CREATE INDEX idx_earthquakes_depth ON earthquakes (depth_km);

-- Volcanic monitoring indexing
CREATE INDEX idx_volcanic_monitoring_volcano ON volcanic_monitoring (volcano_id);
CREATE INDEX idx_volcanic_monitoring_quality ON volcanic_monitoring (data_quality_score);
```

---

## 🔧 Database Configuration

### Environment Variables

```bash
# Primary database connection
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=prometheus_geo
POSTGRES_USER=prometheus_user
POSTGRES_PASSWORD=prometheus_password

# Specialized geological age service credentials
POSTGRES_USER_GEOLOGICAL_AGE=prometheus_age_service
POSTGRES_PASSWORD_GEOLOGICAL_AGE=GeoAge2024!SecurePass#

# ModelManager configuration for age prediction
AZURE_VM_ENDPOINT=https://prometheus-ml.azurewebsites.net
AGE_TRAINING_DATA_PATH=/data/geological_age/training_datasets/
AGE_MODEL_SAVE_PATH=/models/geological_age/
MODEL_REGISTRY_URI=postgresql://prometheus_ml:MLModels2024#@prometheus-db.internal:5432/prometheus_ml

# MLOps pipeline configuration
MLFLOW_TRACKING_URI=postgresql://prometheus_ml:MLModels2024#@prometheus-db.internal:5432/mlflow_tracking
ARTIFACT_STORE_URI=/storage/mlflow_artifacts/
PROMETHEUS_METRICS_ENDPOINT=http://prometheus:9090

# External geological age data sources
EARTHCHEM_API_KEY=earthchem_api_key_here
GEOROC_API_ENDPOINT=https://georoc.mpch-mainz.gwdg.de/georoc/
NAVDAT_API_ENDPOINT=https://navdat.org/api/v1/
PETDB_API_ENDPOINT=https://petdb.ldeo.columbia.edu/api/
```

### Required Extensions

```sql
-- PostGIS for spatial data support
CREATE EXTENSION IF NOT EXISTS postgis;

-- TimeScaleDB for time-series data optimization
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Text search capabilities
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- GiST indexing support
CREATE EXTENSION IF NOT EXISTS btree_gist;
```

---

## 📈 Performance Optimization

### Partitioning Strategy

```sql
-- Partition earthquakes table by time
SELECT add_dimension('earthquakes', 'location', number_partitions => 4);

-- Partition volcanic monitoring by volcano_id
SELECT add_dimension('volcanic_monitoring', 'volcano_id', number_partitions => 8);
```

### Materialized Views for Complex Queries

```sql
-- Materialized view for geological age statistics
CREATE MATERIALIZED VIEW geological_age_summary AS
SELECT 
    lithology,
    COUNT(*) as sample_count,
    AVG(age) as avg_age,
    STDDEV(age) as age_stddev,
    AVG((properties->>'confidence_level')::float) as avg_confidence
FROM geological_structures 
WHERE age IS NOT NULL
GROUP BY lithology;

-- Refresh strategy
CREATE INDEX ON geological_age_summary (lithology);
```

### Query Optimization

```sql
-- Optimized query for spatial-temporal data
EXPLAIN (ANALYZE, BUFFERS) 
SELECT gs.*, rmp.unconfined_compressive_strength_mpa
FROM geological_structures gs
JOIN rock_mechanical_properties rmp ON gs.structure_id = rmp.formation_id
WHERE ST_DWithin(gs.location, ST_GeogFromText('POINT(-122.4194 37.7749)'), 10000)
  AND gs.age BETWEEN 100 AND 200
  AND gs.data_quality_grade IN ('A', 'B')
ORDER BY gs.confidence_level DESC
LIMIT 100;
```

---

## 🔄 Data Migration and Versioning

### Migration Scripts

```sql
-- Migration script example: Add new column to geological_structures
ALTER TABLE geological_structures 
ADD COLUMN metamorphic_grade VARCHAR(50);

-- Update existing records with default values
UPDATE geological_structures 
SET metamorphic_grade = 'unknown' 
WHERE metamorphic_grade IS NULL;

-- Add constraint after data update
ALTER TABLE geological_structures 
ADD CONSTRAINT check_metamorphic_grade 
CHECK (metamorphic_grade IN ('low', 'medium', 'high', 'unknown'));
```

### Version Control

```sql
-- Database schema version tracking
CREATE TABLE schema_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) NOT NULL,
    description TEXT,
    applied_at TIMESTAMPTZ DEFAULT NOW(),
    applied_by VARCHAR(100)
);

-- Insert current version
INSERT INTO schema_versions (version, description, applied_by) 
VALUES ('2.1.0', 'Phase 2 database schema integration', 'system');
```

---

## 🔐 Security and Access Control

### Role-Based Access Control

```sql
-- Create specialized roles
CREATE ROLE prometheus_read_only;
CREATE ROLE prometheus_data_scientist;
CREATE ROLE prometheus_admin;

-- Grant appropriate permissions
GRANT SELECT ON ALL TABLES IN SCHEMA public TO prometheus_read_only;
GRANT SELECT, INSERT, UPDATE ON geological_structures TO prometheus_data_scientist;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO prometheus_admin;
```

### Row-Level Security

```sql
-- Enable row-level security for sensitive data
ALTER TABLE rock_mechanical_properties ENABLE ROW LEVEL SECURITY;

-- Create policy for data quality access
CREATE POLICY quality_access_policy ON rock_mechanical_properties
FOR SELECT USING (
    data_quality_grade IN ('A', 'B') OR 
    current_user IN ('prometheus_admin', 'prometheus_data_scientist')
);
```

---

## 📊 Monitoring and Maintenance

### Performance Monitoring

```sql
-- Query performance monitoring
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements 
WHERE query LIKE '%geological_structures%'
ORDER BY total_time DESC
LIMIT 10;

-- Index usage statistics
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;
```

### Maintenance Tasks

```sql
-- Regular maintenance procedures
VACUUM ANALYZE geological_structures;
REINDEX INDEX CONCURRENTLY idx_geological_structures_spatial;
REFRESH MATERIALIZED VIEW CONCURRENTLY geological_age_summary;

-- TimeScaleDB compression
SELECT add_compression_policy('earthquakes', INTERVAL '7 days');
SELECT add_compression_policy('volcanic_monitoring', INTERVAL '30 days');
```

---

## 🔄 Advanced Data Integration Pipeline (Task 2.3 - Active Development)

The Advanced Data Integration Pipeline is currently being implemented to provide comprehensive multi-source geological data processing capabilities:

### Key Components Under Development

1. **DataLayer Class Implementation**
   - Real-time volcanic data enhancement and processing
   - Multi-parameter geophysical data integration (seismic, gravity, deformation, EM)
   - Advanced geochemical analysis with gas emission monitoring
   - Remote sensing data integration with multi-sensor thermal anomaly detection

2. **Data Quality Assessment System**
   - Automated data validation and quality control
   - Anomaly detection for geological datasets
   - Data consistency checking across multiple sources
   - Quality scoring and confidence metrics

3. **Automated Feature Engineering**
   - Geological domain knowledge integration
   - Temporal and spatial feature extraction
   - Multi-scale data fusion techniques
   - Physics-informed feature generation

4. **Multi-Source Data Integration**
   - DEM and geological map integration
   - Geophysical survey data processing
   - Satellite imagery and remote sensing data
   - Real-time sensor network integration

### Database Schema Extensions for Data Integration

```sql
-- Data integration pipeline tracking
CREATE TABLE data_integration_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(100) UNIQUE NOT NULL,
    data_source VARCHAR(100) NOT NULL,
    integration_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    records_processed INTEGER DEFAULT 0,
    quality_score REAL CHECK (quality_score BETWEEN 0 AND 1),
    error_log TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Data quality assessment results
CREATE TABLE data_quality_assessments (
    id SERIAL PRIMARY KEY,
    assessment_id VARCHAR(100) UNIQUE NOT NULL,
    data_source VARCHAR(100) NOT NULL,
    assessment_date TIMESTAMPTZ DEFAULT NOW(),
    completeness_score REAL CHECK (completeness_score BETWEEN 0 AND 1),
    accuracy_score REAL CHECK (accuracy_score BETWEEN 0 AND 1),
    consistency_score REAL CHECK (consistency_score BETWEEN 0 AND 1),
    timeliness_score REAL CHECK (timeliness_score BETWEEN 0 AND 1),
    overall_quality_score REAL CHECK (overall_quality_score BETWEEN 0 AND 1),
    anomalies_detected INTEGER DEFAULT 0,
    recommendations TEXT,
    assessment_details JSONB DEFAULT '{}'::jsonb
);
```

---

## 🚀 Future Enhancements

### Planned Schema Extensions

1. **Enhanced Geochemical Analysis Tables**
   - Isotope systematics tracking
   - Thermodynamic modeling results
   - Geochemical fingerprinting

2. **Advanced ML Model Integration**
   - Model performance tracking
   - A/B testing results
   - Feature importance storage

3. **Real-time Data Streaming**
   - Kafka integration tables
   - Stream processing results
   - Event sourcing patterns

4. **Global Data Integration**
   - External API synchronization
   - Data quality assessment
   - Multi-source data fusion

---

This comprehensive database schema provides the foundation for advanced geological analysis, real-time monitoring, and machine learning model management in the Prometheus AI Platform. The schema is designed for scalability, performance, and scientific accuracy while maintaining flexibility for future enhancements.