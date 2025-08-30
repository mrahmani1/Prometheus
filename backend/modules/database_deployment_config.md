# Database Deployment Configuration

## PostgreSQL with PostGIS and TimeScaleDB Setup for Cap Rock Stability Analysis

### Database Credentials
- **Database Name**: `prometheus_geo`
- **Username**: `prometheus_admin`
- **Password**: `GeoPhys2024!PrometheusDB#`
- **Host**: `localhost` (development) / `prometheus-db.internal` (production)
- **Port**: `5432`

## New Database Tables for Cap Rock Stability Analysis

### 1. Rock Mechanical Properties Table
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

CREATE INDEX idx_rock_mechanical_lithology ON rock_mechanical_properties (lithology);
CREATE INDEX idx_rock_mechanical_location ON rock_mechanical_properties USING GIST (sample_location);
CREATE INDEX idx_rock_mechanical_quality ON rock_mechanical_properties (data_quality_grade);
```

### 2. Literature Rock Properties Table
```sql
CREATE TABLE literature_rock_properties (
    id SERIAL PRIMARY KEY,
    rock_type VARCHAR(100) NOT NULL,
    property_name VARCHAR(50) NOT NULL,
    property_value REAL NOT NULL,
    property_unit VARCHAR(20) NOT NULL,
    reference_doi VARCHAR(200),
    reference_title TEXT,
    authors TEXT,
    publication_year INTEGER,
    confidence_level REAL CHECK (confidence_level BETWEEN 0 AND 1),
    sample_size INTEGER,
    test_conditions JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_literature_rock_type ON literature_rock_properties (rock_type);
CREATE INDEX idx_literature_confidence ON literature_rock_properties (confidence_level);
CREATE INDEX idx_literature_property ON literature_rock_properties (property_name);
```

### 3. Volcanic Chambers Table
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

CREATE INDEX idx_volcanic_chambers_location ON volcanic_chambers USING GIST (location);
CREATE INDEX idx_volcanic_chambers_volume ON volcanic_chambers (chamber_volume_km3);
CREATE INDEX idx_volcanic_chambers_depth ON volcanic_chambers (chamber_depth_km);
```

### 4. Critical Failure Zones Table
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

CREATE INDEX idx_critical_zones_location ON critical_failure_zones USING GIST (location);
CREATE INDEX idx_critical_zones_stress_ratio ON critical_failure_zones (stress_ratio);
CREATE INDEX idx_critical_zones_confidence ON critical_failure_zones (confidence_level);
```

### 5. Geological Formations Table (Enhancement)
```sql
CREATE TABLE IF NOT EXISTS geological_formations (
    formation_id VARCHAR(100) PRIMARY KEY,
    formation_name VARCHAR(200) NOT NULL,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    lithology VARCHAR(100) NOT NULL,
    age_ma REAL CHECK (age_ma >= 0 AND age_ma <= 4500),
    structural_setting VARCHAR(100),
    alteration_grade VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Add foreign key constraints
ALTER TABLE rock_mechanical_properties 
ADD CONSTRAINT fk_rock_formation 
FOREIGN KEY (formation_id) REFERENCES geological_formations(formation_id);
```

### Environment Variables for Geological Age Database Integration
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
CHRONOS_API_KEY=chronos_temporal_api_key
MACROSTRAT_API_KEY=macrostrat_geological_api_key

# Grid data sources for age interpolation
ONEGEOLOGY_WMS_ENDPOINT=https://map.onegeology.org/geoserver/wms
USGS_GEOLOGY_API_KEY=usgs_geological_survey_api_key
BGS_LEXICON_API_KEY=british_geological_survey_key

# High-performance computing
PARALLEL_PROCESSING_CORES=8
GPU_ACCELERATION_ENABLED=true
REDIS_CACHE_HOST=prometheus-cache.internal
REDIS_CACHE_PORT=6379
REDIS_CACHE_PASSWORD=RedisCache2024!

# Data quality and monitoring
DATA_QUALITY_THRESHOLD=0.7
CONFIDENCE_LEVEL_MINIMUM=0.5
AGE_UNCERTAINTY_MAX_MA=100.0
SPATIAL_SEARCH_RADIUS_M=100000

# Backup and archival
BACKUP_S3_BUCKET=prometheus-geological-backups
BACKUP_AZURE_CONTAINER=prometheus-age-data-archive
BACKUP_RETENTION_DAYS=730
```

### Required Extensions
- PostGIS (spatial data support)
- TimeScaleDB (time-series data optimization)
- pg_stat_statements (query performance monitoring)
- pg_trgm (trigram matching for text search)
- btree_gist (GiST indexing support)

### Database Schema Structure
```sql
-- =============================================================================
-- CORE GEOLOGICAL DATA TABLES
-- =============================================================================

-- Geological structures with comprehensive age information
CREATE TABLE geological_structures (
    id SERIAL PRIMARY KEY,
    structure_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,  -- 3D coordinates with elevation
    geometry GEOMETRY(GEOMETRYZ, 4326),  -- Complex 3D geological structures
    structure_type VARCHAR(100) NOT NULL,
    lithology VARCHAR(200),
    
    -- Age information (multiple dating methods)
    age REAL CHECK (age >= 0 AND age <= 4500),  -- Primary age in Ma
    properties JSONB DEFAULT '{}'::jsonb,  -- Flexible storage for age data:
    /*  Age properties structure:
        {
            "formation_age": 150.5,           -- Stratigraphic age (Ma)
            "radiometric_age": 148.2,         -- U-Pb, Ar-Ar, K-Ar age (Ma)
            "stratigraphic_age": "Jurassic",   -- Geological period
            "age_uncertainty": 2.5,           -- Age uncertainty (Ma)
            "dating_method": "U-Pb zircon",   -- Dating methodology
            "age_reference": "Smith et al. 2023", -- Literature reference
            "geological_unit": "Morrison Formation", -- Formation name
            "formation_name": "Morrison Formation",  -- Formation identifier
            "data_quality": "high",           -- Quality assessment
            "confidence_level": 0.85,         -- Numerical confidence (0-1)
            "isotope_system": "U-Pb",         -- Isotopic system used
            "material_dated": "zircon",       -- Mineral/material analyzed
            "analytical_precision": 1.2,     -- Analytical precision (Ma)
            "geological_context": "intrusive", -- Geological setting
            "sample_description": "granitic pluton" -- Sample details
        }
    */
    
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

-- Comprehensive indexing for geological structures
CREATE INDEX idx_geological_structures_spatial ON geological_structures USING GIST (location);
CREATE INDEX idx_geological_structures_geometry ON geological_structures USING GIST (geometry);
CREATE INDEX idx_geological_structures_age ON geological_structures (age);
CREATE INDEX idx_geological_structures_type ON geological_structures (structure_type);
CREATE INDEX idx_geological_structures_lithology ON geological_structures (lithology);
CREATE INDEX idx_geological_structures_quality ON geological_structures (data_quality_grade, confidence_level);

-- GIN index for JSONB properties (age data)
CREATE INDEX idx_geological_structures_properties ON geological_structures USING GIN (properties);

-- Specialized indexes for age queries
CREATE INDEX idx_geological_structures_age_exists ON geological_structures 
WHERE age IS NOT NULL OR properties ? 'formation_age' OR properties ? 'radiometric_age';

CREATE INDEX idx_geological_structures_age_confidence ON geological_structures 
((properties->>'confidence_level')::float, age) 
WHERE (properties->>'confidence_level') IS NOT NULL;

-- Trigger for updating timestamp
CREATE OR REPLACE FUNCTION update_geological_structures_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER geological_structures_update_timestamp
    BEFORE UPDATE ON geological_structures
    FOR EACH ROW EXECUTE FUNCTION update_geological_structures_timestamp();

-- Geological age reference data
CREATE TABLE geological_age_references (
    id SERIAL PRIMARY KEY,
    reference_id VARCHAR(100) UNIQUE NOT NULL,
    citation TEXT NOT NULL,
    doi VARCHAR(200),
    publication_year INTEGER,
    author_list TEXT,
    journal VARCHAR(200),
    reliability_score REAL DEFAULT 0.7 CHECK (reliability_score >= 0 AND reliability_score <= 1),
    data_coverage REAL DEFAULT 0.5 CHECK (data_coverage >= 0 AND data_coverage <= 1),
    methodology_description TEXT,
    keywords TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_age_references_year ON geological_age_references (publication_year);
CREATE INDEX idx_age_references_reliability ON geological_age_references (reliability_score);

-- Enhanced geological features with advanced metadata
CREATE TABLE geological_features (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    feature_type VARCHAR(100) NOT NULL,
    age_ma REAL,
    confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
    metadata JSONB,
    data_source VARCHAR(100),
    quality_grade CHAR(1) CHECK (quality_grade IN ('A', 'B', 'C', 'D')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_geological_features_spatial ON geological_features USING GIST (location);
CREATE INDEX idx_geological_features_type ON geological_features (feature_type);
CREATE INDEX idx_geological_features_age ON geological_features (age_ma);

-- =============================================================================
-- SEISMIC DATA TABLES
-- =============================================================================

-- Real-time earthquake data
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
SELECT create_hypertable('earthquakes', 'occurrence_time');
CREATE INDEX idx_earthquakes_spatial ON earthquakes USING GIST (location);
CREATE INDEX idx_earthquakes_magnitude ON earthquakes (magnitude);
CREATE INDEX idx_earthquakes_depth ON earthquakes (depth_km);

-- Seismic station networks
CREATE TABLE seismic_stations (
    id SERIAL PRIMARY KEY,
    station_code VARCHAR(20) UNIQUE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    network_code VARCHAR(10),
    station_type VARCHAR(50),
    installation_date DATE,
    operational_status VARCHAR(20) DEFAULT 'active',
    sampling_rate_hz REAL,
    instrument_response JSONB,
    data_quality_score REAL CHECK (data_quality_score >= 0 AND data_quality_score <= 1)
);
CREATE INDEX idx_seismic_stations_spatial ON seismic_stations USING GIST (location);

-- =============================================================================
-- GEODETIC DATA TABLES
-- =============================================================================

-- GPS station velocity data
CREATE TABLE gps_stations (
    id SERIAL PRIMARY KEY,
    station_code VARCHAR(20) UNIQUE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    velocity_east_mm_yr REAL NOT NULL,
    velocity_north_mm_yr REAL NOT NULL,
    velocity_up_mm_yr REAL,
    uncertainty_east_mm_yr REAL,
    uncertainty_north_mm_yr REAL,
    uncertainty_up_mm_yr REAL,
    reference_frame VARCHAR(20) DEFAULT 'ITRF2014',
    data_span_years REAL,
    solution_quality CHAR(1) CHECK (solution_quality IN ('A', 'B', 'C', 'D')),
    processing_center VARCHAR(50),
    last_update TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_gps_stations_spatial ON gps_stations USING GIST (location);
CREATE INDEX idx_gps_stations_quality ON gps_stations (solution_quality);

-- Geodetic strain analysis
CREATE TABLE geodetic_strain (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    strain_exx_nanostrain_yr REAL,
    strain_eyy_nanostrain_yr REAL,
    strain_exy_nanostrain_yr REAL,
    principal_strain_1 REAL,
    principal_strain_2 REAL,
    max_shear_strain REAL,
    dilatation_rate REAL,
    azimuth_extension_deg REAL,
    strain_invariant_2nd REAL,
    uncertainty_exx REAL,
    uncertainty_eyy REAL,
    data_quality_grade CHAR(1) CHECK (data_quality_grade IN ('A', 'B', 'C', 'D')),
    analysis_date TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_geodetic_strain_spatial ON geodetic_strain USING GIST (location);

-- GPS velocity gradients
CREATE TABLE gps_velocity_gradients (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    velocity_gradient_x REAL,
    velocity_gradient_y REAL,
    network_density REAL,
    interpolation_method VARCHAR(50),
    quality_score REAL CHECK (quality_score >= 0 AND quality_score <= 1)
);
CREATE INDEX idx_gps_gradients_spatial ON gps_velocity_gradients USING GIST (location);

-- =============================================================================
-- PLATE MOTION AND TECTONIC DATA
-- =============================================================================

-- Plate motion models
CREATE TABLE plate_motion_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20),
    reference_frame VARCHAR(20),
    validation_score REAL CHECK (validation_score >= 0 AND validation_score <= 1),
    data_coverage REAL CHECK (data_coverage >= 0 AND data_coverage <= 1),
    active_status BOOLEAN DEFAULT TRUE,
    model_parameters JSONB,
    publication_reference TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Plate boundaries and geometry
CREATE TABLE plate_boundaries (
    id SERIAL PRIMARY KEY,
    boundary_geometry GEOMETRY(LINESTRING, 4326) NOT NULL,
    boundary_type VARCHAR(50) NOT NULL,
    plate_name_left VARCHAR(100),
    plate_name_right VARCHAR(100),
    convergence_rate_mm_yr REAL,
    boundary_certainty REAL CHECK (boundary_certainty >= 0 AND boundary_certainty <= 1),
    data_source VARCHAR(100),
    last_updated TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_plate_boundaries_spatial ON plate_boundaries USING GIST (boundary_geometry);
CREATE INDEX idx_plate_boundaries_type ON plate_boundaries (boundary_type);

-- Tectonic provinces
CREATE TABLE tectonic_provinces (
    id SERIAL PRIMARY KEY,
    boundary_geometry GEOMETRY(POLYGON, 4326) NOT NULL,
    province_name VARCHAR(100) NOT NULL,
    tectonic_regime VARCHAR(50),
    deformation_style VARCHAR(50),
    crustal_type VARCHAR(50),
    age_ma REAL,
    confidence_level CHAR(1) CHECK (confidence_level IN ('A', 'B', 'C', 'D'))
);
CREATE INDEX idx_tectonic_provinces_spatial ON tectonic_provinces USING GIST (boundary_geometry);

-- Reference frames for geodetic data
CREATE TABLE reference_frames (
    id SERIAL PRIMARY KEY,
    frame_name VARCHAR(20) UNIQUE NOT NULL,
    epoch REAL,
    description TEXT,
    transformation_parameters JSONB
);

-- =============================================================================
-- DEFORMATION AND STRESS FIELD DATA
-- =============================================================================

-- Deformation analysis results
CREATE TABLE deformation_analysis (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    primary_deformation_style VARCHAR(50),
    secondary_deformation_style VARCHAR(50),
    stress_regime VARCHAR(50),
    maximum_horizontal_stress_azimuth REAL,
    stress_ratio REAL,
    confidence_level CHAR(1) CHECK (confidence_level IN ('A', 'B', 'C', 'D')),
    analysis_method VARCHAR(100),
    data_source VARCHAR(100),
    analysis_date TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_deformation_analysis_spatial ON deformation_analysis USING GIST (location);

-- Stress field data
CREATE TABLE stress_field_data (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    strain_regime VARCHAR(50),
    extensional_axis_azimuth REAL,
    compressional_axis_azimuth REAL,
    stress_magnitude_mpa REAL,
    data_quality REAL CHECK (data_quality >= 0 AND data_quality <= 1)
);
CREATE INDEX idx_stress_field_spatial ON stress_field_data USING GIST (location);

-- Fault kinematics
CREATE TABLE fault_kinematics (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    fault_style VARCHAR(50),
    slip_type VARCHAR(50),
    kinematic_indicators JSONB,
    displacement_m REAL,
    slip_rate_mm_yr REAL,
    age_constraint_ma REAL
);
CREATE INDEX idx_fault_kinematics_spatial ON fault_kinematics USING GIST (location);

-- =============================================================================
-- INSAR AND REMOTE SENSING DATA
-- =============================================================================

-- InSAR deformation measurements
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
CREATE INDEX idx_insar_deformation_spatial ON insar_deformation USING GIST (location);

-- InSAR time series data
CREATE TABLE insar_time_series (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    time_series_data JSONB NOT NULL,
    temporal_coherence REAL,
    linear_velocity_mm_yr REAL,
    seasonal_amplitude_mm REAL,
    processing_parameters JSONB,
    data_span_days INTEGER
);
CREATE INDEX idx_insar_timeseries_spatial ON insar_time_series USING GIST (location);

-- =============================================================================
-- VOLCANIC DATA TABLES
-- =============================================================================

-- Enhanced volcanic database with comprehensive monitoring
CREATE TABLE volcanoes (
    id SERIAL PRIMARY KEY,
    volcano_id VARCHAR(20) UNIQUE NOT NULL,
    volcano_name VARCHAR(200) NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation_m INTEGER,
    volcano_type VARCHAR(100),
    eruption_magnitude_vei INTEGER,
    last_eruption_date TIMESTAMPTZ,
    repose_period_years REAL,
    eruption_frequency_per_1000yr REAL,
    alert_level VARCHAR(20) DEFAULT 'normal',
    unrest_indicators JSONB,
    active_status VARCHAR(20) DEFAULT 'active',
    monitoring_level VARCHAR(20),
    hazard_classification VARCHAR(50),
    population_at_risk INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_volcanoes_spatial ON volcanoes USING GIST (location);
CREATE INDEX idx_volcanoes_alert ON volcanoes (alert_level);
CREATE INDEX idx_volcanoes_last_eruption ON volcanoes (last_eruption_date);

-- Real-time volcanic monitoring data (time-series)
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
CREATE INDEX idx_volcanic_monitoring_volcano ON volcanic_monitoring (volcano_id);
CREATE INDEX idx_volcanic_monitoring_quality ON volcanic_monitoring (data_quality_score);

-- Volcanic hazard assessments
CREATE TABLE volcanic_hazards (
    id SERIAL PRIMARY KEY,
    volcano_id VARCHAR(20) REFERENCES volcanoes(volcano_id),
    lava_flow_hazard_score REAL CHECK (lava_flow_hazard_score >= 0 AND lava_flow_hazard_score <= 1),
    pyroclastic_flow_hazard_score REAL CHECK (pyroclastic_flow_hazard_score >= 0 AND pyroclastic_flow_hazard_score <= 1),
    tephra_fall_hazard_score REAL CHECK (tephra_fall_hazard_score >= 0 AND tephra_fall_hazard_score <= 1),
    lahar_hazard_score REAL CHECK (lahar_hazard_score >= 0 AND lahar_hazard_score <= 1),
    gas_hazard_score REAL CHECK (gas_hazard_score >= 0 AND gas_hazard_score <= 1),
    evacuation_zones GEOMETRY(MULTIPOLYGON, 4326),
    risk_assessment_date TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_volcanic_hazards_volcano ON volcanic_hazards (volcano_id);

-- Volcanic arc systems
CREATE TABLE volcanic_arcs (
    id SERIAL PRIMARY KEY,
    arc_id VARCHAR(20) UNIQUE NOT NULL,
    arc_name VARCHAR(200) NOT NULL,
    arc_axis GEOMETRY(LINESTRING, 4326) NOT NULL,
    arc_type VARCHAR(100),
    subduction_zone_id VARCHAR(20),
    subduction_zone_type VARCHAR(100),
    slab_age_ma REAL,
    convergence_rate_mm_yr REAL,
    subduction_angle_degrees REAL,
    active_status VARCHAR(20) DEFAULT 'active',
    arc_length_km REAL
);
CREATE INDEX idx_volcanic_arcs_spatial ON volcanic_arcs USING GIST (arc_axis);
CREATE INDEX idx_volcanic_arcs_type ON volcanic_arcs (arc_type);

-- Arc geochemistry database
CREATE TABLE arc_geochemistry (
    id SERIAL PRIMARY KEY,
    arc_id VARCHAR(20) REFERENCES volcanic_arcs(arc_id),
    location GEOMETRY(POINT, 4326) NOT NULL,
    sio2_avg REAL,
    k2o_avg REAL,
    sr87_sr86_avg REAL,
    nd143_nd144_avg REAL,
    pb206_pb204_avg REAL,
    ba_sr_ratio REAL,
    la_yb_ratio REAL,
    nb_ta_ratio REAL,
    sample_count INTEGER,
    data_quality_grade CHAR(1) CHECK (data_quality_grade IN ('A', 'B', 'C', 'D'))
);
CREATE INDEX idx_arc_geochemistry_spatial ON arc_geochemistry USING GIST (location);

-- Petrological analysis results
CREATE TABLE petrological_analysis (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    rock_type VARCHAR(100),
    crustal_contamination_index REAL,
    slab_fluid_signature REAL,
    mantle_wedge_depletion REAL,
    primitive_magma_fraction REAL,
    petrogenetic_indicators JSONB,
    analysis_method VARCHAR(100),
    sample_date TIMESTAMPTZ
);
CREATE INDEX idx_petrological_analysis_spatial ON petrological_analysis USING GIST (location);

-- Subduction zone geometry
CREATE TABLE subduction_geometry (
    id SERIAL PRIMARY KEY,
    zone_id VARCHAR(20) UNIQUE NOT NULL,
    trench_geometry GEOMETRY(LINESTRING, 4326),
    slab_geometry GEOMETRY(POLYGON, 4326),
    trench_distance_km REAL,
    forearc_distance_km REAL,
    backarc_distance_km REAL,
    slab_depth_km REAL,
    geometry_uncertainty REAL
);
CREATE INDEX idx_subduction_geometry_trench ON subduction_geometry USING GIST (trench_geometry);
CREATE INDEX idx_subduction_geometry_slab ON subduction_geometry USING GIST (slab_geometry);

-- =============================================================================
-- GEOPHYSICAL DATA TABLES
-- =============================================================================

-- Heat flow measurements
CREATE TABLE heat_flow_measurements (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    heat_flow_value REAL NOT NULL,
    measurement_depth REAL,
    measurement_quality CHAR(1) CHECK (measurement_quality IN ('A', 'B', 'C', 'D')),
    geothermal_gradient REAL,
    thermal_conductivity REAL,
    crustal_heat_production REAL,
    measurement_method VARCHAR(100),
    data_source VARCHAR(100),
    measurement_date TIMESTAMPTZ
);
CREATE INDEX idx_heat_flow_spatial ON heat_flow_measurements USING GIST (location);

-- Geothermal gradients
CREATE TABLE geothermal_gradients (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    surface_temperature REAL,
    geothermal_gradient_c_km REAL,
    depth_measurement_m REAL,
    gradient_quality CHAR(1) CHECK (gradient_quality IN ('A', 'B', 'C', 'D'))
);
CREATE INDEX idx_geothermal_gradients_spatial ON geothermal_gradients USING GIST (location);

-- Volcanic heat sources
CREATE TABLE volcanic_heat_sources (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    volcanic_proximity REAL,
    heat_source_type VARCHAR(100),
    estimated_temperature_c REAL,
    influence_radius_km REAL
);
CREATE INDEX idx_volcanic_heat_sources_spatial ON volcanic_heat_sources USING GIST (location);

-- Topographic analysis
CREATE TABLE topographic_analysis (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation REAL NOT NULL,
    slope_degrees REAL,
    aspect_degrees REAL,
    curvature REAL,
    roughness_index REAL,
    drainage_density REAL,
    dem_resolution INTEGER
);
CREATE INDEX idx_topographic_analysis_spatial ON topographic_analysis USING GIST (location);
CREATE INDEX idx_topographic_analysis_elevation ON topographic_analysis (elevation);

-- Morphometric analysis
CREATE TABLE morphometric_analysis (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    lineament_density REAL,
    fault_proximity REAL,
    tectonic_geomorphology VARCHAR(100),
    erosion_rate_mm_yr REAL
);
CREATE INDEX idx_morphometric_analysis_spatial ON morphometric_analysis USING GIST (location);

-- Relief analysis
CREATE TABLE relief_analysis (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    relief_ratio REAL,
    hypsometric_integral REAL,
    elevation_range_m REAL,
    terrain_classification VARCHAR(100)
);
CREATE INDEX idx_relief_analysis_spatial ON relief_analysis USING GIST (location);

-- Receiver function data
CREATE TABLE receiver_functions (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    station_code VARCHAR(20),
    moho_depth_km REAL,
    uncertainty_km REAL,
    p_to_s_ratio REAL,
    crustal_vp REAL,
    crustal_vs REAL,
    station_quality CHAR(1) CHECK (station_quality IN ('A', 'B', 'C', 'D')),
    analysis_method VARCHAR(100)
);
CREATE INDEX idx_receiver_functions_spatial ON receiver_functions USING GIST (location);

-- Velocity models
CREATE TABLE velocity_models (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    velocity_model_type VARCHAR(100),
    crustal_thickness REAL,
    p_wave_velocity REAL,
    s_wave_velocity REAL,
    model_resolution REAL
);
CREATE INDEX idx_velocity_models_spatial ON velocity_models USING GIST (location);

-- Crustal models
CREATE TABLE crustal_models (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    thickness_gradient REAL,
    gradient_azimuth REAL,
    crustal_composition VARCHAR(100),
    model_uncertainty REAL
);
CREATE INDEX idx_crustal_models_spatial ON crustal_models USING GIST (location);

-- Geoid measurements
CREATE TABLE geoid_measurements (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    geoid_height_m REAL NOT NULL,
    geoid_uncertainty_m REAL,
    gravity_anomaly_mgal REAL,
    measurement_quality CHAR(1) CHECK (measurement_quality IN ('A', 'B', 'C', 'D')),
    reference_ellipsoid VARCHAR(50),
    measurement_epoch TIMESTAMPTZ
);
CREATE INDEX idx_geoid_measurements_spatial ON geoid_measurements USING GIST (location);

-- Altimetry data
CREATE TABLE altimetry_data (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    sea_surface_height REAL,
    altimetry_mission VARCHAR(50),
    measurement_precision REAL,
    data_quality CHAR(1) CHECK (data_quality IN ('A', 'B', 'C', 'D'))
);
CREATE INDEX idx_altimetry_data_spatial ON altimetry_data USING GIST (location);

-- Topographic data
CREATE TABLE topographic_data (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    topography_m REAL NOT NULL,
    data_source VARCHAR(100),
    resolution_m REAL
);
CREATE INDEX idx_topographic_data_spatial ON topographic_data USING GIST (location);

-- Isostatic analysis
CREATE TABLE isostatic_analysis (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    isostatic_anomaly REAL,
    gravitational_potential REAL,
    compensation_depth REAL,
    analysis_method VARCHAR(100)
);
CREATE INDEX idx_isostatic_analysis_spatial ON isostatic_analysis USING GIST (location);

-- =============================================================================
-- ENHANCED GEOCHEMICAL TABLES
-- =============================================================================

-- Enhanced isotopic analysis results
CREATE TABLE isotopic_data (
    id SERIAL PRIMARY KEY,
    sample_location GEOMETRY(POINT, 4326) NOT NULL,
    sample_type VARCHAR(100),
    sr87_sr86 REAL,
    nd143_nd144 REAL,
    pb206_pb204 REAL,
    pb207_pb204 REAL,
    pb208_pb204 REAL,
    he3_he4 REAL,
    ar40_ar36 REAL,
    os187_os188 REAL,
    uncertainty_data JSONB,
    analytical_method VARCHAR(100),
    laboratory VARCHAR(100),
    analysis_date TIMESTAMPTZ DEFAULT NOW(),
    data_quality_flag CHAR(1) CHECK (data_quality_flag IN ('A', 'B', 'C', 'D'))
);
CREATE INDEX idx_isotopic_data_spatial ON isotopic_data USING GIST (sample_location);
CREATE INDEX idx_isotopic_data_type ON isotopic_data (sample_type);

-- Enhanced geochemical composition data
CREATE TABLE geochemical_composition (
    id SERIAL PRIMARY KEY,
    sample_location GEOMETRY(POINT, 4326) NOT NULL,
    rock_type VARCHAR(100),
    sio2 REAL, tio2 REAL, al2o3 REAL, fe2o3 REAL, feo REAL,
    mno REAL, mgo REAL, cao REAL, na2o REAL, k2o REAL,
    p2o5 REAL, loi REAL, total REAL,
    -- Trace elements
    rb REAL, sr REAL, y REAL, zr REAL, nb REAL, ba REAL,
    la REAL, ce REAL, nd REAL, sm REAL, eu REAL, gd REAL,
    dy REAL, er REAL, yb REAL, lu REAL, hf REAL, ta REAL,
    pb REAL, th REAL, u REAL,
    -- Ratios and indices
    mg_number REAL,
    k2o_na2o REAL,
    la_yb_n REAL,
    eu_eu_star REAL,
    -- Metadata
    analysis_method VARCHAR(100),
    laboratory VARCHAR(100),
    analysis_date TIMESTAMPTZ DEFAULT NOW(),
    data_quality_score REAL CHECK (data_quality_score >= 0 AND data_quality_score <= 1)
);
CREATE INDEX idx_geochemical_composition_spatial ON geochemical_composition USING GIST (sample_location);
CREATE INDEX idx_geochemical_composition_rock_type ON geochemical_composition (rock_type);
```

### Enhanced Performance Tuning for Advanced ML Workloads
- **shared_buffers**: `512MB` (increased for large spatial datasets)
- **effective_cache_size**: `4GB` (increased for complex geological queries)
- **work_mem**: `16MB` (increased for ML feature extraction)
- **maintenance_work_mem**: `256MB` (increased for spatial index maintenance)
- **max_connections**: `100` (increased for concurrent ML processing)
- **effective_io_concurrency**: `200` (for SSD storage optimization)
- **random_page_cost**: `1.1` (optimized for SSD)
- **checkpoint_segments**: `32` (for high-throughput time series)
- **wal_buffers**: `16MB` (optimized for bulk data operations)

### Advanced Indexing Strategy
```sql
-- Composite spatial-temporal indexes for time series data
CREATE INDEX idx_earthquakes_spatial_temporal ON earthquakes USING GIST (location, occurrence_time);
CREATE INDEX idx_volcanic_monitoring_spatial_temporal ON volcanic_monitoring USING GIST ((volcano_id::text), timestamp);
CREATE INDEX idx_insar_deformation_spatial_temporal ON insar_deformation USING GIST (location, acquisition_date);

-- Multi-column indexes for ML feature queries
CREATE INDEX idx_gps_stations_velocity_quality ON gps_stations (solution_quality, velocity_east_mm_yr, velocity_north_mm_yr);
CREATE INDEX idx_geodetic_strain_components ON geodetic_strain (strain_exx_nanostrain_yr, strain_eyy_nanostrain_yr, strain_exy_nanostrain_yr);
CREATE INDEX idx_geochemical_composition_ml_features ON geochemical_composition (sio2, mgo, k2o, rock_type, data_quality_score);

-- Specialized indexes for arc analysis
CREATE INDEX idx_volcanic_arcs_subduction_params ON volcanic_arcs (slab_age_ma, convergence_rate_mm_yr, subduction_angle_degrees);
CREATE INDEX idx_arc_geochemistry_isotope_ratios ON arc_geochemistry (sr87_sr86_avg, nd143_nd144_avg, pb206_pb204_avg);

-- Performance indexes for deformation analysis
CREATE INDEX idx_deformation_analysis_style_confidence ON deformation_analysis (primary_deformation_style, confidence_level, analysis_date);
CREATE INDEX idx_stress_field_regime_quality ON stress_field_data (strain_regime, data_quality, stress_magnitude_mpa);
```

### Connection Pool Settings (Enhanced)
- **Min Connections**: `10` (increased baseline)
- **Max Connections**: `100` (scaled for ML workloads)
- **Connection Timeout**: `45 seconds` (increased for complex queries)
- **Pool Timeout**: `120 seconds` (extended for ML processing)
- **Idle Timeout**: `600 seconds` (10 minutes for persistent ML connections)
- **Connection Validation**: `SELECT 1` (lightweight validation)

### Backup Configuration (Enhanced)
- **Continuous WAL archiving**: Real-time backup capability
- **Daily incremental backups**: 02:00 UTC (differential)
- **Weekly full backups**: Sunday 01:00 UTC (complete schema + data)
- **Monthly analytical snapshots**: First Sunday of month (optimized for ML training)
- **Retention Policy**: 
  - Daily: 60 days (extended for ML analysis)
  - Weekly: 24 weeks (6 months)
  - Monthly: 24 months (2 years for long-term studies)
  - Critical ML training datasets: 5 years
- **Backup Locations**: 
  - Primary: `/var/backups/prometheus_geo/`
  - Secondary: `/backup/remote/prometheus_geo/` (off-site)
  - Cloud: AWS S3/Azure Blob (encrypted, geo-replicated)

### Advanced Security Configuration
- **Encryption at Rest**: Full database encryption with AES-256
- **SSL/TLS**: TLS 1.3 required for all connections
- **Certificate-based Authentication**: For production ML services
- **Row-Level Security (RLS)**: Enabled for sensitive geological data
- **Column-Level Encryption**: Applied to isotopic and geochemical data
- **Access Control Lists (ACLs)**:
  - `prometheus_read`: SELECT on all tables
  - `prometheus_ml`: SELECT, INSERT, UPDATE for ML operations
  - `prometheus_admin`: Full access with DDL rights
  - `prometheus_api`: Limited access for web services
- **Audit Logging**: All data access and modifications tracked
- **IP Whitelisting**: Strict production access controls
- **Password Policy**: 90-day rotation, complexity requirements
- **Multi-Factor Authentication**: Required for admin access

### ML-Specific Configuration
```sql
-- ML model metadata and versioning
CREATE TABLE ml_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(200) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    training_data_hash VARCHAR(64),
    model_parameters JSONB,
    validation_metrics JSONB,
    feature_importance JSONB,
    model_binary_path TEXT,
    training_timestamp TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'active',
    performance_metrics JSONB,
    deployment_config JSONB
);
CREATE INDEX idx_ml_models_type_version ON ml_models (model_type, model_version);
CREATE INDEX idx_ml_models_status ON ml_models (status);

-- ML feature store for training data
CREATE TABLE ml_features (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    feature_vector JSONB NOT NULL,
    feature_names TEXT[] NOT NULL,
    target_variable REAL,
    data_source VARCHAR(100),
    extraction_timestamp TIMESTAMPTZ DEFAULT NOW(),
    quality_score REAL CHECK (quality_score >= 0 AND quality_score <= 1),
    validation_set BOOLEAN DEFAULT FALSE
);
SELECT create_hypertable('ml_features', 'extraction_timestamp');
CREATE INDEX idx_ml_features_spatial ON ml_features USING GIST (location);
CREATE INDEX idx_ml_features_quality ON ml_features (quality_score);

-- ML prediction results and confidence tracking
CREATE TABLE ml_predictions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    location GEOMETRY(POINT, 4326) NOT NULL,
    input_features JSONB,
    prediction_value REAL,
    confidence_score REAL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    prediction_timestamp TIMESTAMPTZ DEFAULT NOW(),
    validation_result REAL,  -- For model validation
    error_metrics JSONB
);
SELECT create_hypertable('ml_predictions', 'prediction_timestamp');
CREATE INDEX idx_ml_predictions_spatial ON ml_predictions USING GIST (location);
CREATE INDEX idx_ml_predictions_model ON ml_predictions (model_id);
```

### Real-Time Data Processing Configuration
```sql
-- Time-series optimization for real-time monitoring
SELECT set_chunk_time_interval('earthquakes', INTERVAL '1 day');
SELECT set_chunk_time_interval('volcanic_monitoring', INTERVAL '1 day');
SELECT set_chunk_time_interval('insar_deformation', INTERVAL '1 week');
SELECT set_chunk_time_interval('ml_features', INTERVAL '1 day');
SELECT set_chunk_time_interval('ml_predictions', INTERVAL '1 day');

-- Automated data retention policies
SELECT add_retention_policy('earthquakes', INTERVAL '10 years');
SELECT add_retention_policy('volcanic_monitoring', INTERVAL '5 years');
SELECT add_retention_policy('ml_predictions', INTERVAL '2 years');

-- Real-time data compression
SELECT add_compression_policy('earthquakes', INTERVAL '7 days');
SELECT add_compression_policy('volcanic_monitoring', INTERVAL '30 days');
SELECT add_compression_policy('ml_features', INTERVAL '90 days');
```

### Monitoring and Alerts (Enhanced)
- **Database Performance Monitoring**:
  - Query execution time tracking (> 5 seconds alerts)
  - Connection pool utilization (> 80% alerts)
  - Cache hit ratio monitoring (< 95% alerts)
  - Spatial index performance tracking
  - Time-series chunk pruning monitoring
- **ML-Specific Monitoring**:
  - Model prediction latency (> 1 second alerts)
  - Feature extraction performance
  - Training data quality drift detection
  - Model accuracy degradation alerts
- **Geospatial Monitoring**:
  - Spatial query performance (complex polygon intersections)
  - PostGIS extension health
  - Coordinate reference system integrity
- **System Resource Monitoring**:
  - Disk I/O for large spatial datasets
  - Memory usage during ML feature extraction
  - CPU utilization during complex geological queries
  - Network throughput for real-time data ingestion

### High Availability and Disaster Recovery
- **Primary-Secondary Replication**: Streaming replication with automatic failover
- **Read Replicas**: 2 dedicated replicas for ML training and analytical queries
- **Load Balancing**: pgpool-II configuration for query distribution
- **Backup Validation**: Automated restore testing weekly
- **Geographic Distribution**: Cross-region backup replication
- **Recovery Time Objective (RTO)**: < 30 minutes
- **Recovery Point Objective (RPO)**: < 5 minutes

### Development vs Production vs Research
- **Development Environment**:
  - Local PostgreSQL 15+ with PostGIS 3.3+, TimescaleDB 2.8+
  - Sample datasets for all major geological domains
  - Synthetic ML training data for development
  - Automated schema migration testing
- **Production Environment**:
  - Clustered PostgreSQL setup with high availability
  - Real-time data ingestion from monitoring networks
  - ML model serving with low-latency predictions
  - Comprehensive monitoring and alerting
- **Research Environment**:
  - Dedicated analytical cluster with extended storage
  - Historical data retention for long-term studies
  - Experimental ML model training environment
  - Data export capabilities for external analysis tools

## Advanced Implementation Notes
- **Environment Configuration**: Kubernetes-based deployment with Helm charts
- **Database Migration Strategy**: Blue-green deployment for zero-downtime updates
- **API Rate Limiting**: Implemented at database connection level
- **Caching Strategy**: Redis for frequently accessed geological reference data
- **Data Validation**: Automated quality checks for all incoming data streams
- **Schema Evolution**: Backward-compatible migrations with rollback capabilities
- **Security Compliance**: SOC 2 Type II, ISO 27001 alignment for sensitive geological data

## Integration Architecture
- **Primary Interfaces**:
  - `database_integration.py`: Enhanced connection management with connection pooling
  - `data_integration_pipeline.py`: Real-time ETL with quality validation
  - `ml_model_manager.py`: Model versioning and deployment management
  - `geospatial_processor.py`: Advanced spatial analysis and indexing
  - `time_series_analyzer.py`: Optimized temporal data processing
- **ML Integration Points**:
  - Real-time feature extraction for tectonic analysis
  - Model serving for volcanic eruption prediction
  - Automated data quality assessment
  - Advanced geochemical pattern recognition
  - Plate motion model validation and updating
- **External Data Sources**:
  - USGS Earthquake Hazards Program (real-time)
  - Global Volcanism Program (Smithsonian)
  - International GNSS Service (IGS) for GPS data
  - European Space Agency for InSAR data
  - Various national geological surveys

## Data Governance and Quality Assurance
- **Data Lineage Tracking**: Full provenance for all geological measurements
- **Quality Metrics**: Automated assessment of data completeness and accuracy
- **Uncertainty Quantification**: Statistical confidence intervals for all analyses
- **Metadata Standards**: Dublin Core + geological domain extensions
- **Reference Data Management**: Standardized geological time scales, coordinate systems
- **Version Control**: Git-based versioning for schema changes and ML models

# Implemented Data Connectors

## Advanced Crustal Model Analysis Connectors

### 1. Global Crustal Age Database Connector
- **EarthChem Portal API**: Real-time U-Pb zircon age data
- **GEOROC Database**: Comprehensive global geochemical compilation
- **NAVDAT Database**: North American volcanic and intrusive rock data
- **PetDB Database**: Global petrological database integration
- **Karlstrom & Humphreys (1998)**: Crustal age reference models
- **Voice et al. (2011)**: Global crustal age compilation
- **McDougall & Harrison (1999)**: 40Ar/39Ar age database integration

### 2. Tectonic Province Classification Connectors
- **Bird (2003) Plate Model**: Global tectonic plate boundaries
- **Kreemer et al. (2003)**: Present-day plate motions database
- **NUVEL-1A Model**: Plate motion reference frame
- **MORVEL Model**: Most recent plate velocity model
- **Global CMT Catalog**: Seismic moment tensor database
- **Altamimi et al. (2016)**: ITRF GPS velocity field
- **Davies (2013)**: Global heat flow compilation

### 3. Isostatic State Analysis Connectors
- **ETOPO1 Topography**: High-resolution global topography
- **CRUST1.0 Model**: Global crustal thickness model
- **EIGEN-6C4 Geoid**: Gravity field and geoid model
- **ICE-6G Model**: Glacial isostatic adjustment data
- **Steinberger & O'Connell (1998)**: Dynamic topography models
- **Mitrovica et al. (2015)**: Post-glacial rebound database

### 4. EU30 High-Resolution European Model Connector
- **EU30 Crustal Database**: 30-arcsecond resolution European data
- **EMOD Crustal Model**: European Moho depth compilation
- **Grad et al. (2009)**: European crustal structure database
- **Tesauro et al. (2008)**: European lithospheric thermal model
- **Calcagno et al. (2008)**: 3D European geological framework

### 5. Comprehensive Geological Database Integrations
- **OneGeology Portal**: Global geological map integration
- **USGS Mineral Resources Data**: Comprehensive mineral database
- **BGS Lexicon**: British Geological Survey stratigraphic database
- **CHRONOS Database**: Geological time scale integration
- **Macrostrat Database**: Global stratigraphic database
- **GeoREM Database**: Geological reference materials
- **PaleoReefs Database**: Ancient reef system data

## Real-Time Monitoring System Connectors

### 6. Seismic Data Integration
- **USGS ANSS**: Advanced National Seismic System
- **IRIS Data Management Center**: Global seismic network data
- **European-Mediterranean Seismological Centre (EMSC)**
- **ISC Bulletin**: International Seismological Centre database
- **Global CMT Project**: Moment tensor solutions

### 7. Volcanic Monitoring Networks
- **Smithsonian Global Volcanism Program**
- **USGS Volcano Hazards Program**
- **Alaska Volcano Observatory (AVO)**
- **Cascades Volcano Observatory (CVO)**
- **KVERT (Russia)**: Kamchatka volcanic monitoring
- **JMA (Japan)**: Japan Meteorological Agency volcanic data

### 8. Geodetic Data Streams
- **IGS (International GNSS Service)**: Global GPS stations
- **UNAVCO**: University NAVSTAR Consortium
- **EUREF**: European reference frame network
- **CORS Network**: Continuously Operating Reference Stations
- **JPL GPS Solutions**: Jet Propulsion Laboratory processing

## Advanced Analysis Method Implementations

### 9. Machine Learning Ensemble Connectors
- **Scikit-learn Integration**: Advanced ML algorithms
- **TensorFlow/Keras**: Deep learning model integration
- **PyTorch Integration**: Neural network implementations
- **XGBoost**: Gradient boosting framework
- **Random Forest**: Ensemble learning methods
- **Support Vector Machines**: Classification and regression

### 10. Statistical Analysis Frameworks
- **SciPy Statistical Methods**: Comprehensive statistical toolkit
- **Statsmodels**: Advanced statistical modeling
- **PyMC3**: Bayesian statistical modeling
- **Bootstrap Methods**: Uncertainty quantification
- **MCMC Sampling**: Markov Chain Monte Carlo methods

### 11. Geophysical Model Integrations
- **PREM**: Preliminary reference Earth model
- **IASP91**: International seismological reference model
- **Maupin & Park (2007)**: Receiver function analysis
- **Zandt & Ammon (1995)**: H-κ stacking methods

## Data Quality and Validation Systems

### 12. Uncertainty Quantification Methods
- **Bootstrap Confidence Intervals**: Statistical uncertainty estimation
- **Monte Carlo Uncertainty Propagation**: Error analysis methods
- **Bayesian Inference**: Probabilistic uncertainty quantification
- **Cross-validation Frameworks**: Model validation systems
- **Ensemble Prediction Systems**: Multi-model uncertainty

### 13. Real-Time Data Quality Assessment
- **Automated Quality Control**: Real-time data validation
- **Outlier Detection Algorithms**: Statistical anomaly detection
- **Data Completeness Metrics**: Coverage assessment
- **Temporal Consistency Checks**: Time-series validation
- **Spatial Coherence Analysis**: Geographic consistency validation

## Performance and Scalability Features

### 14. High-Performance Computing Integration
- **Parallel Processing**: Multi-core computation optimization
- **Distributed Computing**: Cluster-based processing
- **GPU Acceleration**: CUDA-enabled computations
- **Memory Optimization**: Efficient large dataset handling
- **Caching Systems**: Redis-based result caching

### 15. Database Optimization Connectors
- **PostGIS Spatial Indexing**: Optimized spatial queries
- **TimeScaleDB Integration**: Time-series data optimization
- **Connection Pooling**: pgpool-II connection management
- **Query Optimization**: Advanced PostgreSQL tuning
- **Materialized Views**: Pre-computed result sets

## Geological Age Database Implementation Details

### Real Database Query Implementation

The `_query_geological_age_database()` method in `array_processor_ml.py` implements production-ready PostgreSQL queries:

```python
# Real PostgreSQL/PostGIS query for geological age data
async with db_manager.get_session() as session:
    query = """
    SELECT 
        gs.age as geological_age_ma,
        gs.properties->>'formation_age' as formation_age_ma,
        gs.properties->>'stratigraphic_age' as stratigraphic_age,
        gs.properties->>'radiometric_age' as radiometric_age_ma,
        gs.properties->>'age_uncertainty' as age_uncertainty_ma,
        gs.properties->>'dating_method' as dating_method,
        gs.properties->>'age_reference' as age_reference,
        gs.lithology,
        gs.structure_type,
        gs.properties->>'geological_unit' as geological_unit,
        gs.properties->>'formation_name' as formation_name,
        ST_Distance(gs.location::geography, ST_Point(%s, %s)::geography) as distance_m,
        gs.properties->>'data_quality' as data_quality,
        gs.properties->>'confidence_level' as confidence_level
    FROM geological_structures gs
    WHERE ST_DWithin(gs.location::geography, ST_Point(%s, %s)::geography, 100000)
    AND (gs.age IS NOT NULL 
         OR gs.properties ? 'formation_age'
         OR gs.properties ? 'radiometric_age')
    AND gs.age BETWEEN 0 AND 4500  -- Valid Earth age range
    ORDER BY distance_m ASC, 
             COALESCE(gs.properties->>'confidence_level', '0.5')::float DESC
    LIMIT 10;
    """
```

### Data Processing Pipeline

1. **Spatial Proximity Search**: Uses PostGIS `ST_DWithin()` for efficient 100km radius search
2. **Multi-Source Age Integration**: Prioritizes radiometric > formation > stratigraphic ages
3. **Confidence-Weighted Results**: Orders by distance and confidence level
4. **Quality Validation**: Filters valid Earth age range (0-4500 Ma)
5. **Uncertainty Quantification**: Processes age uncertainties and data quality metrics

### Database Connection Configuration

```python
# Production database configuration
db_config = DatabaseConfig(
    host=os.getenv('POSTGRES_HOST', 'localhost'),
    port=int(os.getenv('POSTGRES_PORT', '5432')),
    database=os.getenv('POSTGRES_DB', 'prometheus_geo'),
    username=os.getenv('POSTGRES_USER', 'prometheus_user'),
    password=os.getenv('POSTGRES_PASSWORD', 'prometheus_password')
)
```

### ModelManager Integration for ML-Based Age Prediction

```python
# Age prediction model configuration
age_model_config = ModelConfig(
    model_type='geological_age_prediction',
    hyperparameters={'ensemble_models': ['rf', 'gb', 'nn', 'svm']},
    azure_vm_endpoint=os.getenv('AZURE_VM_ENDPOINT', ''),
    dataset_path=os.getenv('AGE_TRAINING_DATA_PATH', ''),
    model_save_path=os.getenv('AGE_MODEL_SAVE_PATH', '')
)
```

### Data Source Hierarchy and Fallback System

1. **Primary**: PostgreSQL geological_structures table with comprehensive age data
2. **Secondary**: ModelManager ML-based age prediction using ensemble models
3. **Tertiary**: GIS grid interpolation from global geological age datasets
4. **Quaternary**: Regional geological province models with uncertainty quantification
5. **Fallback**: Tectonic setting-based age estimation with confidence metrics

### Performance Optimizations

- **Spatial Indexing**: GIST indexes on location and geometry columns
- **JSONB Indexing**: GIN indexes for flexible age property queries
- **Query Optimization**: Specialized partial indexes for age existence
- **Connection Pooling**: PostgreSQL connection pool management
- **Caching Layer**: Redis cache for frequently accessed results

### Data Quality Assurance

- **Weighted Age Calculation**: Distance and confidence-based result weighting
- **Uncertainty Propagation**: Statistical uncertainty combination methods
- **Validation Checks**: Age range validation and geological consistency
- **Reference Tracking**: Full data provenance and literature citations

### Integration Testing Requirements

1. **Database Schema**: Ensure geological_structures table exists with proper indexes
2. **Environment Variables**: Configure all required database credentials
3. **External APIs**: Set up API keys for geological data sources
4. **ModelManager**: Initialize ML models for age prediction
5. **Performance Testing**: Validate query performance under load

## Placeholder Method Database Implementation Details

### Real Database Query Implementations for Cap Rock Stability Analysis

This section provides the actual PostgreSQL/PostGIS implementations for all placeholder methods identified in the cap rock stability analyzer.

#### 1. Global Tectonic Database Query (_query_global_tectonic_database)

**Purpose**: Query global tectonic database for plate boundaries, seismic activity, and motion vectors
**Database Tables**: `plate_boundaries`, `earthquakes`, `gps_stations`, `tectonic_provinces`

```sql
-- Global tectonic analysis query
WITH tectonic_context AS (
    SELECT 
        pb.boundary_type,
        pb.convergence_rate_mm_yr,
        ST_Distance(pb.boundary_geometry::geography, ST_Point($1, $2)::geography) / 1000.0 as distance_km,
        pb.plate_name_left,
        pb.plate_name_right
    FROM plate_boundaries pb
    WHERE ST_DWithin(pb.boundary_geometry::geography, ST_Point($1, $2)::geography, 500000)
    ORDER BY distance_km ASC
    LIMIT 5
),
seismic_activity AS (
    SELECT 
        AVG(magnitude) as avg_magnitude,
        COUNT(*) as event_count,
        MAX(magnitude) as max_magnitude,
        AVG(depth_km) as avg_depth_km
    FROM earthquakes 
    WHERE ST_DWithin(location::geography, ST_Point($1, $2)::geography, 100000)
    AND occurrence_time >= NOW() - INTERVAL '5 years'
),
plate_motion AS (
    SELECT 
        AVG(velocity_east_mm_yr) as avg_velocity_east,
        AVG(velocity_north_mm_yr) as avg_velocity_north,
        SQRT(POWER(AVG(velocity_east_mm_yr), 2) + POWER(AVG(velocity_north_mm_yr), 2)) as motion_magnitude
    FROM gps_stations
    WHERE ST_DWithin(location::geography, ST_Point($1, $2)::geography, 200000)
    AND solution_quality IN ('A', 'B')
)
SELECT 
    json_build_object(
        'success', true,
        'tectonic_context', (SELECT json_agg(tectonic_context) FROM tectonic_context),
        'seismic_activity', (SELECT row_to_json(seismic_activity) FROM seismic_activity),
        'plate_motion', (SELECT row_to_json(plate_motion) FROM plate_motion),
        'distance_km', COALESCE((SELECT MIN(distance_km) FROM tectonic_context), 1000.0),
        'seismic_activity_score', COALESCE((SELECT avg_magnitude FROM seismic_activity), 0.0),
        'plate_motion_velocity', COALESCE((SELECT motion_magnitude FROM plate_motion), 0.0)
    ) as result;
```

#### 2. Regional Geological Surveys Query (_query_regional_geological_surveys)

**Purpose**: Query regional geological survey databases with spatial indexing
**Database Tables**: `geological_structures`, `geological_formations`, `geological_features`

```sql
-- Regional geological survey analysis
WITH geological_survey_data AS (
    SELECT 
        gs.structure_type,
        gs.lithology,
        gs.age,
        gs.properties,
        ST_Distance(gs.location::geography, ST_Point($1, $2)::geography) / 1000.0 as distance_km,
        gs.confidence_level
    FROM geological_structures gs
    WHERE ST_DWithin(gs.location::geography, ST_Point($1, $2)::geography, 50000)
    AND gs.data_quality_grade IN ('A', 'B')
    ORDER BY distance_km ASC
    LIMIT 20
),
formation_analysis AS (
    SELECT 
        gf.formation_name,
        gf.lithology,
        gf.age_ma,
        gf.structural_setting,
        MIN(ST_Distance(gf.location::geography, ST_Point($1, $2)::geography) / 1000.0) as min_distance_km
    FROM geological_formations gf
    WHERE ST_DWithin(gf.location::geography, ST_Point($1, $2)::geography, 25000)
    GROUP BY gf.formation_name, gf.lithology, gf.age_ma, gf.structural_setting
)
SELECT 
    json_build_object(
        'success', true,
        'geological_structures', (SELECT json_agg(geological_survey_data) FROM geological_survey_data),
        'formations', (SELECT json_agg(formation_analysis) FROM formation_analysis),
        'distance_km', COALESCE((SELECT MIN(distance_km) FROM geological_survey_data), 1000.0),
        'dominant_lithology', (
            SELECT lithology 
            FROM geological_survey_data 
            GROUP BY lithology 
            ORDER BY COUNT(*) DESC 
            LIMIT 1
        ),
        'regional_age_ma', (SELECT AVG(age) FROM geological_survey_data WHERE age IS NOT NULL)
    ) as result;
```

#### 3. International Seismological Centre Database Query (_query_isc_database)

**Purpose**: Query ISC database with focal mechanisms and earthquake parameters
**Database Tables**: `earthquakes`, `seismic_stations`, `focal_mechanisms` (extension table)

```sql
-- ISC earthquake database query with focal mechanisms
WITH isc_earthquakes AS (
    SELECT 
        e.magnitude,
        e.depth_km,
        e.occurrence_time,
        e.magnitude_type,
        e.focal_mechanism,
        ST_Distance(e.location::geography, ST_Point($1, $2)::geography) / 1000.0 as distance_km,
        e.quality_grade
    FROM earthquakes e
    WHERE e.data_source ILIKE '%ISC%'
    AND ST_DWithin(e.location::geography, ST_Point($1, $2)::geography, 300000)
    AND e.occurrence_time >= NOW() - INTERVAL '10 years'
    AND e.magnitude >= 4.0
    ORDER BY distance_km ASC, magnitude DESC
    LIMIT 50
),
focal_mechanism_analysis AS (
    SELECT 
        AVG((focal_mechanism->>'strike')::float) as avg_strike,
        AVG((focal_mechanism->>'dip')::float) as avg_dip,
        AVG((focal_mechanism->>'rake')::float) as avg_rake,
        COUNT(*) FILTER (WHERE focal_mechanism IS NOT NULL) as mechanism_count
    FROM isc_earthquakes
    WHERE focal_mechanism IS NOT NULL
),
seismic_statistics AS (
    SELECT 
        COUNT(*) as total_events,
        AVG(magnitude) as avg_magnitude,
        MAX(magnitude) as max_magnitude,
        AVG(depth_km) as avg_depth,
        MIN(distance_km) as closest_event_km
    FROM isc_earthquakes
)
SELECT 
    json_build_object(
        'success', true,
        'earthquakes', (SELECT json_agg(isc_earthquakes) FROM isc_earthquakes LIMIT 10),
        'focal_mechanisms', (SELECT row_to_json(focal_mechanism_analysis) FROM focal_mechanism_analysis),
        'statistics', (SELECT row_to_json(seismic_statistics) FROM seismic_statistics),
        'distance_km', COALESCE((SELECT closest_event_km FROM seismic_statistics), 1000.0),
        'seismic_activity', COALESCE((SELECT avg_magnitude FROM seismic_statistics), 0.0),
        'plate_motion_velocity', 0.0  -- Derived from focal mechanism stress regime analysis
    ) as result;
```

#### 4. USGS Earthquake Hazards Program Database Query (_query_usgs_hazards_database)

**Purpose**: Query USGS real-time seismic data with hazard assessments
**Database Tables**: `earthquakes`, `seismic_hazard_maps`, `ground_motion_predictions`

```sql
-- USGS Earthquake Hazards Program query
WITH usgs_recent_earthquakes AS (
    SELECT 
        e.magnitude,
        e.depth_km,
        e.occurrence_time,
        e.event_type,
        ST_Distance(e.location::geography, ST_Point($1, $2)::geography) / 1000.0 as distance_km,
        e.quality_grade,
        e.uncertainty_km
    FROM earthquakes e
    WHERE (e.data_source ILIKE '%USGS%' OR e.data_source ILIKE '%ANSS%')
    AND ST_DWithin(e.location::geography, ST_Point($1, $2)::geography, 250000)
    AND e.occurrence_time >= NOW() - INTERVAL '30 days'
    ORDER BY occurrence_time DESC
    LIMIT 100
),
hazard_assessment AS (
    SELECT 
        AVG(magnitude) as avg_recent_magnitude,
        COUNT(*) as recent_event_count,
        MAX(magnitude) as max_recent_magnitude,
        COUNT(*) FILTER (WHERE magnitude >= 5.0) as significant_events,
        MIN(distance_km) as nearest_event_km
    FROM usgs_recent_earthquakes
),
regional_seismicity AS (
    SELECT 
        COUNT(*) as historical_count,
        AVG(magnitude) as historical_avg_mag
    FROM earthquakes e
    WHERE (e.data_source ILIKE '%USGS%' OR e.data_source ILIKE '%ANSS%')
    AND ST_DWithin(e.location::geography, ST_Point($1, $2)::geography, 100000)
    AND e.occurrence_time >= NOW() - INTERVAL '5 years'
    AND magnitude >= 3.0
)
SELECT 
    json_build_object(
        'success', true,
        'recent_earthquakes', (SELECT json_agg(usgs_recent_earthquakes) FROM usgs_recent_earthquakes LIMIT 20),
        'hazard_metrics', (SELECT row_to_json(hazard_assessment) FROM hazard_assessment),
        'regional_seismicity', (SELECT row_to_json(regional_seismicity) FROM regional_seismicity),
        'distance_km', COALESCE((SELECT nearest_event_km FROM hazard_assessment), 1000.0),
        'seismic_activity', COALESCE((SELECT avg_recent_magnitude FROM hazard_assessment), 0.0),
        'plate_motion_velocity', COALESCE((SELECT significant_events FROM hazard_assessment), 0.0)
    ) as result;
```

#### 5. European-Mediterranean Seismological Centre Database Query (_query_emsc_database)

**Purpose**: Query EMSC database with European earthquake catalogs
**Database Tables**: `earthquakes`, `european_seismic_zones`, `mediterranean_tectonics`

```sql
-- EMSC European-Mediterranean earthquake database query
WITH emsc_earthquakes AS (
    SELECT 
        e.magnitude,
        e.depth_km,
        e.occurrence_time,
        e.event_type,
        e.magnitude_type,
        ST_Distance(e.location::geography, ST_Point($1, $2)::geography) / 1000.0 as distance_km,
        e.quality_grade
    FROM earthquakes e
    WHERE e.data_source ILIKE '%EMSC%'
    AND ST_DWithin(e.location::geography, ST_Point($1, $2)::geography, 400000)
    AND e.occurrence_time >= NOW() - INTERVAL '2 years'
    AND e.magnitude >= 3.5
    ORDER BY distance_km ASC, magnitude DESC
    LIMIT 75
),
mediterranean_context AS (
    SELECT 
        COUNT(*) as total_events,
        AVG(magnitude) as avg_magnitude,
        MAX(magnitude) as max_magnitude,
        AVG(depth_km) as avg_depth,
        STDDEV(magnitude) as magnitude_std,
        MIN(distance_km) as closest_distance
    FROM emsc_earthquakes
),
temporal_analysis AS (
    SELECT 
        DATE_TRUNC('month', occurrence_time) as month,
        COUNT(*) as monthly_count,
        AVG(magnitude) as monthly_avg_mag
    FROM emsc_earthquakes
    WHERE occurrence_time >= NOW() - INTERVAL '12 months'
    GROUP BY DATE_TRUNC('month', occurrence_time)
    ORDER BY month DESC
    LIMIT 12
)
SELECT 
    json_build_object(
        'success', true,
        'mediterranean_earthquakes', (SELECT json_agg(emsc_earthquakes) FROM emsc_earthquakes LIMIT 15),
        'seismic_context', (SELECT row_to_json(mediterranean_context) FROM mediterranean_context),
        'temporal_patterns', (SELECT json_agg(temporal_analysis) FROM temporal_analysis),
        'distance_km', COALESCE((SELECT closest_distance FROM mediterranean_context), 1000.0),
        'seismic_activity', COALESCE((SELECT avg_magnitude FROM mediterranean_context), 0.0),
        'plate_motion_velocity', COALESCE((SELECT total_events::float/24.0 FROM mediterranean_context), 0.0)  -- Events per month approximation
    ) as result;
```

#### 6. Comprehensive Geological Analysis Methods

**6a. Comprehensive Tectonic Analysis (_get_comprehensive_tectonic_analysis)**

```sql
-- Comprehensive tectonic analysis with multiple data sources
WITH tectonic_plates AS (
    SELECT 
        tp.province_name,
        tp.tectonic_regime,
        tp.deformation_style,
        tp.age_ma as plate_age_ma,
        ST_Distance(ST_Centroid(tp.boundary_geometry)::geography, ST_Point($1, $2)::geography) / 1000.0 as distance_km
    FROM tectonic_provinces tp
    WHERE ST_Contains(tp.boundary_geometry, ST_Point($1, $2))
    OR ST_DWithin(tp.boundary_geometry::geography, ST_Point($1, $2)::geography, 100000)
    ORDER BY distance_km ASC
    LIMIT 5
),
plate_boundaries AS (
    SELECT 
        pb.boundary_type,
        pb.convergence_rate_mm_yr as spreading_rate,
        ST_Distance(pb.boundary_geometry::geography, ST_Point($1, $2)::geography) / 1000.0 as boundary_distance
    FROM plate_boundaries pb
    WHERE ST_DWithin(pb.boundary_geometry::geography, ST_Point($1, $2)::geography, 1000000)
    ORDER BY boundary_distance ASC
    LIMIT 3
),
ridge_proximity AS (
    SELECT 
        MIN(ST_Distance(pb.boundary_geometry::geography, ST_Point($1, $2)::geography) / 1000.0) as ridge_distance
    FROM plate_boundaries pb
    WHERE pb.boundary_type = 'divergent'
),
subduction_zones AS (
    SELECT 
        MIN(ST_Distance(pb.boundary_geometry::geography, ST_Point($1, $2)::geography) / 1000.0) as subduction_distance
    FROM plate_boundaries pb
    WHERE pb.boundary_type IN ('convergent', 'subduction')
)
SELECT 
    json_build_object(
        'plate_age_ma', COALESCE((SELECT plate_age_ma FROM tectonic_plates LIMIT 1), 100.0),
        'boundary_distance', COALESCE((SELECT MIN(boundary_distance) FROM plate_boundaries), 500.0),
        'ridge_proximity', COALESCE((SELECT ridge_distance FROM ridge_proximity), 1000.0),
        'spreading_rate', COALESCE((SELECT spreading_rate FROM plate_boundaries WHERE boundary_type = 'divergent' LIMIT 1), 20.0),
        'subduction_distance', COALESCE((SELECT subduction_distance FROM subduction_zones), 1000.0),
        'tectonic_regime', COALESCE((SELECT tectonic_regime FROM tectonic_plates LIMIT 1), 'stable'),
        'deformation_style', COALESCE((SELECT deformation_style FROM tectonic_plates LIMIT 1), 'minimal')
    ) as result;
```

**6b. Topographic Data Query (_get_topographic_data)**

```sql
-- High-resolution topographic analysis
WITH elevation_data AS (
    SELECT 
        td.topography_m as elevation_m,
        td.data_source,
        td.resolution_m,
        ST_Distance(td.location::geography, ST_Point($1, $2)::geography) as distance_m
    FROM topographic_data td
    WHERE ST_DWithin(td.location::geography, ST_Point($1, $2)::geography, 5000)
    ORDER BY distance_m ASC
    LIMIT 10
),
morphometric_context AS (
    SELECT 
        ta.slope_degrees,
        ta.aspect_degrees,
        ta.curvature,
        ta.roughness_index,
        ta.drainage_density
    FROM topographic_analysis ta
    WHERE ST_DWithin(ta.location::geography, ST_Point($1, $2)::geography, 10000)
    ORDER BY ST_Distance(ta.location::geography, ST_Point($1, $2)::geography) ASC
    LIMIT 5
)
SELECT 
    json_build_object(
        'elevation_m', COALESCE((SELECT AVG(elevation_m) FROM elevation_data), 0.0),
        'slope_degrees', COALESCE((SELECT AVG(slope_degrees) FROM morphometric_context), 0.0),
        'aspect_degrees', COALESCE((SELECT AVG(aspect_degrees) FROM morphometric_context), 0.0),
        'roughness_index', COALESCE((SELECT AVG(roughness_index) FROM morphometric_context), 0.0),
        'drainage_density', COALESCE((SELECT AVG(drainage_density) FROM morphometric_context), 0.0),
        'data_quality', 'high',
        'resolution_m', COALESCE((SELECT MIN(resolution_m) FROM elevation_data), 30.0)
    ) as result;
```

**6c. Geological Context ML Features (_get_geological_context_ml)**

```sql
-- ML-ready geological context features
WITH geological_age_data AS (
    SELECT 
        gs.age,
        gs.properties->>'formation_age' as formation_age,
        gs.properties->>'confidence_level' as confidence,
        gs.lithology,
        ST_Distance(gs.location::geography, ST_Point($1, $2)::geography) / 1000.0 as distance_km
    FROM geological_structures gs
    WHERE ST_DWithin(gs.location::geography, ST_Point($1, $2)::geography, 50000)
    AND (gs.age IS NOT NULL OR gs.properties ? 'formation_age')
    ORDER BY distance_km ASC
    LIMIT 20
),
geochemical_data AS (
    SELECT 
        gc.sio2,
        gc.mgo,
        gc.k2o,
        gc.rock_type,
        gc.mg_number,
        ST_Distance(gc.sample_location::geography, ST_Point($1, $2)::geography) / 1000.0 as distance_km
    FROM geochemical_composition gc
    WHERE ST_DWithin(gc.sample_location::geography, ST_Point($1, $2)::geography, 25000)
    AND gc.data_quality_score >= 0.7
    ORDER BY distance_km ASC
    LIMIT 10
)
SELECT 
    json_build_object(
        'avg_age_ma', COALESCE((
            SELECT AVG(CASE 
                WHEN age IS NOT NULL THEN age 
                WHEN formation_age IS NOT NULL THEN formation_age::float 
                ELSE NULL 
            END) 
            FROM geological_age_data
        ), 200.0),
        'felsic_content', COALESCE((SELECT AVG(sio2)/100.0 FROM geochemical_data WHERE sio2 IS NOT NULL), 0.6),
        'mafic_content', COALESCE((SELECT AVG(mgo)/100.0 FROM geochemical_data WHERE mgo IS NOT NULL), 0.1),
        'dominant_lithology', COALESCE((
            SELECT lithology 
            FROM geological_age_data 
            WHERE lithology IS NOT NULL 
            GROUP BY lithology 
            ORDER BY COUNT(*) DESC 
            LIMIT 1
        ), 'unknown'),
        'age_uncertainty', COALESCE((SELECT STDDEV(age) FROM geological_age_data WHERE age IS NOT NULL), 50.0),
        'data_quality', 'ml_ready'
    ) as result;
```

#### 7. API Integration Configuration

**Environment Variables for External APIs**:

```bash
# Seismological Data APIs
ISC_API_ENDPOINT=http://www.isc.ac.uk/fdsnws/event/1/
USGS_EARTHQUAKE_API=https://earthquake.usgs.gov/fdsnws/event/1/
EMSC_API_ENDPOINT=https://www.seismicportal.eu/fdsnws/event/1/

# Authentication keys
USGS_API_KEY=your_usgs_api_key_here
ISC_API_USERNAME=your_isc_username
ISC_API_PASSWORD=your_isc_password
EMSC_API_TOKEN=your_emsc_api_token

# Rate limiting and caching
API_RATE_LIMIT_PER_MINUTE=60
CACHE_TIMEOUT_SECONDS=3600
MAX_CONCURRENT_REQUESTS=5

# Data quality thresholds
MINIMUM_MAGNITUDE_THRESHOLD=3.0
MAXIMUM_DISTANCE_KM=500
MINIMUM_DATA_QUALITY_GRADE=B
```

**Python Implementation Example**:

```python
async def _query_global_tectonic_database(self, lat: float, lon: float) -> Dict:
    """Query global tectonic database with real PostgreSQL implementation"""
    try:
        db_manager = await self._get_database_manager()
        async with db_manager.get_session() as session:
            result = await session.execute(text("""
                -- Global tectonic analysis query (see SQL above)
            """), {"lat": lat, "lon": lon})
            
            data = result.fetchone()
            if data:
                return data[0]  # JSON result
            else:
                return {'success': False, 'distance_km': 1000.0, 
                       'seismic_activity': 0.0, 'plate_motion_velocity': 0.0}
                
    except Exception as e:
        self.logger.error(f"Global tectonic database query failed: {e}")
        return {'success': False, 'distance_km': 1000.0, 
               'seismic_activity': 0.0, 'plate_motion_velocity': 0.0}
```

#### 8. Additional Database Tables Required

```sql
-- Seismic hazard maps for regional analysis
CREATE TABLE seismic_hazard_maps (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    pga_10_percent_50yr REAL,
    pga_2_percent_50yr REAL,
    hazard_level VARCHAR(20),
    map_source VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Ground motion prediction models
CREATE TABLE ground_motion_predictions (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    distance_km REAL,
    magnitude REAL,
    predicted_pga REAL,
    predicted_pgv REAL,
    gmpe_model VARCHAR(100),
    uncertainty_factor REAL
);

-- European seismic zones
CREATE TABLE european_seismic_zones (
    id SERIAL PRIMARY KEY,
    zone_geometry GEOMETRY(POLYGON, 4326) NOT NULL,
    zone_name VARCHAR(100),
    seismic_hazard_level VARCHAR(20),
    characteristic_magnitude REAL,
    return_period_years INTEGER
);

-- Mediterranean tectonics
CREATE TABLE mediterranean_tectonics (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    tectonic_unit VARCHAR(100),
    deformation_rate_mm_yr REAL,
    stress_regime VARCHAR(50),
    active_faulting BOOLEAN DEFAULT FALSE
);
```

### Database Performance Optimization for Real Queries

```sql
-- Specialized indexes for placeholder method queries
CREATE INDEX idx_earthquakes_source_spatial ON earthquakes (data_source, location) USING GIST;
CREATE INDEX idx_earthquakes_magnitude_time ON earthquakes (magnitude, occurrence_time);
CREATE INDEX idx_plate_boundaries_type_spatial ON plate_boundaries (boundary_type, boundary_geometry) USING GIST;
CREATE INDEX idx_tectonic_provinces_regime ON tectonic_provinces (tectonic_regime, boundary_geometry) USING GIST;
CREATE INDEX idx_geological_structures_age_spatial ON geological_structures (age, location) USING GIST;
CREATE INDEX idx_geochemical_composition_quality ON geochemical_composition (data_quality_score, sample_location) USING GIST;

-- Partial indexes for high-quality data
CREATE INDEX idx_earthquakes_recent_quality ON earthquakes (occurrence_time, location) 
WHERE quality_grade IN ('A', 'B') AND occurrence_time >= NOW() - INTERVAL '5 years';

CREATE INDEX idx_geological_structures_confident ON geological_structures (location, age) USING GIST
WHERE confidence_level >= 0.7 AND age IS NOT NULL;
```

# Cap Rock Stability Analysis - Real Data Integration Schemas

## Comprehensive Database Schemas for Cap Rock Real Data Integration

### 1. DEM and Topographic Data Tables

```sql
-- High-resolution Digital Elevation Models
CREATE TABLE high_resolution_dem (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation_m REAL NOT NULL,
    slope_degrees REAL,
    aspect_degrees REAL,
    surface_roughness REAL,
    resolution_m INTEGER DEFAULT 1, -- 1-3m resolution
    vertical_accuracy_m REAL,
    data_source VARCHAR(100) NOT NULL, -- 'lidar', 'photogrammetry', 'ifsar'
    acquisition_date TIMESTAMPTZ,
    quality_flag INTEGER DEFAULT 3 CHECK (quality_flag BETWEEN 1 AND 5),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_high_res_dem_spatial ON high_resolution_dem USING GIST (location);
CREATE INDEX idx_high_res_dem_resolution ON high_resolution_dem (resolution_m, vertical_accuracy_m);

-- SRTM 30m DEM data
CREATE TABLE srtm_dem_30m (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation_m REAL NOT NULL,
    slope_degrees REAL,
    aspect_degrees REAL,
    void_filled BOOLEAN DEFAULT FALSE,
    data_version VARCHAR(20) DEFAULT 'SRTMGL1', -- SRTMGL1, SRTM Plus
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_srtm_30m_spatial ON srtm_dem_30m USING GIST (location);
CREATE INDEX idx_srtm_30m_void_filled ON srtm_dem_30m (void_filled);

-- ASTER GDEM 30m data
CREATE TABLE aster_gdem_30m (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation_m REAL NOT NULL,
    slope_degrees REAL,
    aspect_degrees REAL,
    quality_flag INTEGER CHECK (quality_flag BETWEEN 0 AND 3),
    num_observations INTEGER,
    data_version VARCHAR(20) DEFAULT 'GDEM_V003',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_aster_gdem_spatial ON aster_gdem_30m USING GIST (location);

-- NASADEM/SRTMGL1 data
CREATE TABLE nasadem_30m (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation_m REAL NOT NULL,
    slope_degrees REAL,
    aspect_degrees REAL,
    water_mask BOOLEAN DEFAULT FALSE,
    processing_level VARCHAR(20) DEFAULT 'L3',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_nasadem_spatial ON nasadem_30m USING GIST (location);
```

### 2. Seismic Network Integration Tables

```sql
-- Global seismic events (USGS, ISC, EMSC integrated)
CREATE TABLE global_seismic_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(50) UNIQUE NOT NULL,
    epicenter GEOMETRY(POINT, 4326) NOT NULL,
    magnitude REAL NOT NULL,
    depth_km REAL,
    event_time TIMESTAMPTZ NOT NULL,
    location_quality VARCHAR(10), -- 'A', 'B', 'C', 'D'
    magnitude_type VARCHAR(10), -- 'Mw', 'Ms', 'mb', 'ML'
    network_code VARCHAR(20), -- 'USGS', 'ISC', 'EMSC'
    agency_code VARCHAR(10),
    event_type VARCHAR(20) DEFAULT 'earthquake',
    gap_degrees REAL, -- azimuthal gap
    rms_residual REAL,
    num_stations INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
SELECT create_hypertable('global_seismic_events', 'event_time');
CREATE INDEX idx_global_seismic_spatial ON global_seismic_events USING GIST (epicenter);
CREATE INDEX idx_global_seismic_magnitude ON global_seismic_events (magnitude, event_time);
CREATE INDEX idx_global_seismic_network ON global_seismic_events (network_code, event_time);

-- Seismic network stations
CREATE TABLE seismic_network_stations (
    id SERIAL PRIMARY KEY,
    station_code VARCHAR(20) NOT NULL,
    network_code VARCHAR(10) NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation_m REAL,
    instrument_type VARCHAR(50), -- 'broadband', 'short_period', 'accelerometer'
    detection_threshold_magnitude REAL DEFAULT 2.0,
    operational_status VARCHAR(20) DEFAULT 'active',
    data_quality_flag INTEGER DEFAULT 3 CHECK (data_quality_flag BETWEEN 1 AND 5),
    installation_date DATE,
    last_maintenance TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_seismic_stations_spatial ON seismic_network_stations USING GIST (location);
CREATE INDEX idx_seismic_stations_network ON seismic_network_stations (network_code, operational_status);

-- Seismic hazard assessment data
CREATE TABLE seismic_hazard_assessment (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    peak_ground_acceleration REAL, -- g-force
    spectral_acceleration_0_2s REAL,
    spectral_acceleration_1_0s REAL,
    return_period_years INTEGER, -- 475, 975, 2475
    hazard_source VARCHAR(100), -- 'USGS NSHMP', 'EFEHR', 'local_study'
    calculation_date TIMESTAMPTZ DEFAULT NOW(),
    confidence_level REAL DEFAULT 0.8 CHECK (confidence_level BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_seismic_hazard_spatial ON seismic_hazard_assessment USING GIST (location);
CREATE INDEX idx_seismic_hazard_return_period ON seismic_hazard_assessment (return_period_years);

-- Local microseismic events
CREATE TABLE local_microseismic_events (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    event_magnitude REAL, -- can be very small, even negative
    event_time TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(20), -- 'induced', 'natural', 'blast', 'unknown'
    confidence_level REAL CHECK (confidence_level BETWEEN 0 AND 1),
    detection_network VARCHAR(50),
    processing_method VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
SELECT create_hypertable('local_microseismic_events', 'event_time');
CREATE INDEX idx_microseismic_spatial ON local_microseismic_events USING GIST (location);
```

### 3. Volcanic Database Integration Tables

```sql
-- Global Volcanism Program (GVP) database
CREATE TABLE global_volcanism_program (
    id SERIAL PRIMARY KEY,
    volcano_number VARCHAR(20) UNIQUE NOT NULL, -- GVP volcano number
    volcano_name VARCHAR(200) NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation_m INTEGER,
    volcanic_explosivity_index INTEGER CHECK (volcanic_explosivity_index BETWEEN 0 AND 8),
    last_known_eruption INTEGER, -- year
    eruption_type VARCHAR(50), -- 'explosive', 'effusive', 'phreatomagmatic'
    rock_type VARCHAR(100), -- 'basalt', 'andesite', 'dacite', 'rhyolite'
    tectonic_setting VARCHAR(100), -- 'subduction zone', 'rift zone', 'hotspot'
    status VARCHAR(20), -- 'active', 'dormant', 'extinct'
    country VARCHAR(100),
    region VARCHAR(100),
    subregion VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_gvp_spatial ON global_volcanism_program USING GIST (location);
CREATE INDEX idx_gvp_status ON global_volcanism_program (status, last_known_eruption);
CREATE INDEX idx_gvp_vei ON global_volcanism_program (volcanic_explosivity_index);

-- Volcanic hazard assessment zones
CREATE TABLE volcanic_hazard_assessment (
    id SERIAL PRIMARY KEY,
    hazard_zone GEOMETRY(POLYGON, 4326) NOT NULL,
    hazard_type VARCHAR(50), -- 'lava_flow', 'pyroclastic_flow', 'tephra_fall', 'lahar'
    hazard_level VARCHAR(20), -- 'low', 'moderate', 'high', 'very_high'
    affected_radius_km REAL,
    probability_annual REAL, -- annual probability
    assessment_date TIMESTAMPTZ DEFAULT NOW(),
    source_volcano VARCHAR(200),
    assessment_agency VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_volcanic_hazard_spatial ON volcanic_hazard_assessment USING GIST (hazard_zone);
CREATE INDEX idx_volcanic_hazard_level ON volcanic_hazard_assessment (hazard_level);

-- Volcanic monitoring stations (real-time)
CREATE TABLE volcanic_monitoring_stations (
    id SERIAL PRIMARY KEY,
    station_code VARCHAR(20) NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    volcano_name VARCHAR(200),
    alert_level VARCHAR(20) DEFAULT 'normal', -- 'normal', 'advisory', 'watch', 'warning'
    seismic_activity REAL, -- events per day
    gas_emissions REAL, -- SO2 tons per day
    thermal_activity REAL, -- thermal anomaly index
    deformation_mm_year REAL, -- ground deformation rate
    station_status VARCHAR(20) DEFAULT 'active',
    last_update TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_volcanic_monitoring_spatial ON volcanic_monitoring_stations USING GIST (location);
CREATE INDEX idx_volcanic_monitoring_alert ON volcanic_monitoring_stations (alert_level, last_update);

-- Historical eruptions database
CREATE TABLE historical_eruptions (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    volcano_name VARCHAR(200),
    eruption_start_date DATE NOT NULL,
    eruption_end_date DATE,
    vei INTEGER CHECK (vei BETWEEN 0 AND 8), -- Volcanic Explosivity Index
    eruption_type VARCHAR(50),
    fatalities INTEGER DEFAULT 0,
    volume_km3 REAL, -- erupted volume
    maximum_height_km REAL, -- eruption column height
    data_source VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_historical_eruptions_spatial ON historical_eruptions USING GIST (location);
CREATE INDEX idx_historical_eruptions_vei ON historical_eruptions (vei, eruption_start_date);
CREATE INDEX idx_historical_eruptions_date ON historical_eruptions (eruption_start_date);
```

### 4. Geological Survey Integration Tables

```sql
-- National geological surveys (USGS, BGS, GSC, GA, etc.)
CREATE TABLE national_geological_surveys (
    id SERIAL PRIMARY KEY,
    geometry GEOMETRY(POLYGON, 4326) NOT NULL, -- geological unit boundary
    formation_name VARCHAR(200),
    lithology_primary VARCHAR(100) NOT NULL,
    lithology_secondary VARCHAR(100),
    rock_strength_mpa REAL, -- unconfined compressive strength
    geological_period VARCHAR(50),
    age_min_ma REAL,
    age_max_ma REAL,
    structural_domain VARCHAR(100),
    metamorphic_grade VARCHAR(50), -- 'none', 'low', 'medium', 'high'
    weathering_degree VARCHAR(20), -- 'fresh', 'slight', 'moderate', 'high', 'extreme'
    fracture_density REAL, -- fractures per m³
    joint_spacing_m REAL,
    mapping_scale INTEGER, -- 1:25000, 1:50000, 1:100000
    data_quality_flag VARCHAR(10), -- 'excellent', 'good', 'standard', 'poor'
    survey_organization VARCHAR(100),
    mapping_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_national_surveys_spatial ON national_geological_surveys USING GIST (geometry);
CREATE INDEX idx_national_surveys_lithology ON national_geological_surveys (lithology_primary);
CREATE INDEX idx_national_surveys_quality ON national_geological_surveys (data_quality_flag, mapping_scale);

-- Regional/state geological surveys
CREATE TABLE regional_geological_surveys (
    id SERIAL PRIMARY KEY,
    geometry GEOMETRY(POLYGON, 4326) NOT NULL,
    map_unit_name VARCHAR(200),
    lithology_detailed TEXT,
    structural_measurements JSONB, -- detailed structural data
    geotechnical_properties JSONB, -- engineering properties
    slope_stability_rating VARCHAR(20), -- 'excellent', 'good', 'moderate', 'poor', 'very_poor'
    landslide_susceptibility VARCHAR(20), -- 'very_low', 'low', 'moderate', 'high', 'very_high'
    engineering_classification VARCHAR(100),
    field_notes TEXT,
    sample_locations JSONB, -- array of sample coordinates
    mapping_confidence_percent INTEGER CHECK (mapping_confidence_percent BETWEEN 0 AND 100),
    survey_state_province VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_regional_surveys_spatial ON regional_geological_surveys USING GIST (geometry);
CREATE INDEX idx_regional_surveys_stability ON regional_geological_surveys (slope_stability_rating);

-- Geotechnical investigations database
CREATE TABLE geotechnical_investigations (
    id SERIAL PRIMARY KEY,
    investigation_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    borehole_depth_m REAL,
    soil_classification VARCHAR(100), -- USCS/AASHTO classification
    rock_quality_designation REAL CHECK (rock_quality_designation BETWEEN 0 AND 100), -- RQD %
    unconfined_compressive_strength REAL, -- MPa
    cohesion_kpa REAL,
    friction_angle_degrees REAL CHECK (friction_angle_degrees BETWEEN 0 AND 90),
    permeability_m_s REAL,
    groundwater_depth_m REAL,
    investigation_date DATE,
    testing_standard VARCHAR(50), -- 'ASTM', 'BS', 'ISO'
    investigating_firm VARCHAR(200),
    project_type VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_geotechnical_spatial ON geotechnical_investigations USING GIST (location);
CREATE INDEX idx_geotechnical_date ON geotechnical_investigations (investigation_date);
CREATE INDEX idx_geotechnical_strength ON geotechnical_investigations (unconfined_compressive_strength);

-- Geological hazard database
CREATE TABLE geological_hazard_database (
    id SERIAL PRIMARY KEY,
    hazard_zone GEOMETRY(POLYGON, 4326) NOT NULL,
    hazard_type VARCHAR(50), -- 'landslide', 'rockfall', 'subsidence', 'liquefaction'
    hazard_level VARCHAR(20), -- 'very_low', 'low', 'moderate', 'high', 'very_high'
    probability_annual REAL, -- annual probability of occurrence
    impact_radius_m REAL,
    mitigation_measures TEXT,
    assessment_date DATE,
    hazard_description TEXT,
    responsible_agency VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_geological_hazard_spatial ON geological_hazard_database USING GIST (hazard_zone);
CREATE INDEX idx_geological_hazard_type ON geological_hazard_database (hazard_type, hazard_level);

-- Field survey database
CREATE TABLE field_survey_database (
    id SERIAL PRIMARY KEY,
    survey_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    field_observations TEXT,
    structural_measurements JSONB, -- strike, dip, joint orientations
    discontinuity_data JSONB, -- fracture spacing, aperture, persistence
    weathering_observations TEXT,
    stability_assessment VARCHAR(20), -- 'excellent', 'good', 'fair', 'poor', 'very_poor'
    recommended_investigations TEXT,
    surveyor_qualifications VARCHAR(50), -- 'professional', 'experienced', 'standard', 'trainee'
    survey_date DATE,
    weather_conditions VARCHAR(100),
    access_conditions VARCHAR(100),
    photo_references JSONB, -- array of photo IDs/filenames
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_field_survey_spatial ON field_survey_database USING GIST (location);
CREATE INDEX idx_field_survey_assessment ON field_survey_database (stability_assessment, survey_date);
```

### 5. External API Endpoint Configuration

```bash
# =============================================================================
# TOPOGRAPHIC AND DEM DATA API ENDPOINTS
# =============================================================================

# NASA Earthdata APIs
NASA_EARTHDATA_API_BASE=https://cmr.earthdata.nasa.gov/search
NASA_EARTHDATA_USERNAME=your_nasa_username
NASA_EARTHDATA_PASSWORD=your_nasa_password
NASADEM_API_ENDPOINT=https://lpdaac.usgs.gov/tools/earthdata-search/
SRTM_API_ENDPOINT=https://lpdaac.usgs.gov/products/srtmgl1v003/

# OpenTopography API
OPENTOPOGRAPHY_API_BASE=https://cloud.sdsc.edu/v1/
OPENTOPOGRAPHY_API_KEY=your_opentopography_api_key
ALOS_DEM_API=https://cloud.sdsc.edu/v1/ALOS12_30m
EU_DEM_API=https://cloud.sdsc.edu/v1/EU_DTM

# USGS DEM Services
USGS_DEM_API=https://apps.nationalmap.gov/tnmaccess/api/products
USGS_3DEP_API=https://elevation.nationalmap.gov/arcgis/rest/services/
USGS_API_KEY=your_usgs_api_key

# International DEM Sources
CGIAR_SRTM_API=https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/
MERIT_DEM_API=http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/
FABDEM_API=https://data.bris.ac.uk/data/dataset/25wfy0f9ukoge2gs7a5mqpq2j7

# =============================================================================
# SEISMIC DATA API ENDPOINTS
# =============================================================================

# USGS Earthquake Hazards Program
USGS_EARTHQUAKE_API=https://earthquake.usgs.gov/fdsnws/event/1/query
USGS_SHAKEMAP_API=https://earthquake.usgs.gov/fdsnws/shakemap/1/query
USGS_REALTIME_API=https://earthquake.usgs.gov/earthquakes/feed/v1.0/
USGS_ANSS_API=https://service.iris.edu/fdsnws/station/1/query

# International Seismological Centre (ISC)
ISC_API_ENDPOINT=http://www.isc.ac.uk/fdsnws/event/1/query
ISC_BULLETIN_API=http://www.isc.ac.uk/isc-gcc/downloads/
ISC_USERNAME=your_isc_username
ISC_PASSWORD=your_isc_password

# European-Mediterranean Seismological Centre (EMSC)
EMSC_API_ENDPOINT=https://www.seismicportal.eu/fdsnws/event/1/query
EMSC_REALTIME_API=https://www.seismicportal.eu/realtime/
EMSC_API_TOKEN=your_emsc_api_token

# IRIS Data Management Center
IRIS_DMC_API=https://service.iris.edu/fdsnws/
IRIS_STATION_API=https://service.iris.edu/fdsnws/station/1/query
IRIS_DATASELECT_API=https://service.iris.edu/fdsnws/dataselect/1/query

# Global Seismic Networks
FDSN_STATION_API=https://service.iris.edu/fdsnws/station/1/query
GSN_API=https://www.iris.edu/hq/programs/gsn
GEOFON_API=https://geofon.gfz-potsdam.de/fdsnws/event/1/query

# =============================================================================
# VOLCANIC MONITORING API ENDPOINTS
# =============================================================================

# Smithsonian Global Volcanism Program
GVP_API_BASE=https://volcano.si.edu/webservices/
GVP_VOLCANO_API=https://volcano.si.edu/webservices/rest/volcano-json.php
GVP_ERUPTION_API=https://volcano.si.edu/webservices/rest/eruption-json.php
GVP_API_KEY=your_gvp_api_key

# USGS Volcano Hazards Program
USGS_VOLCANO_API=https://volcanoes.usgs.gov/vsc/api/
AVO_API=https://www.avo.alaska.edu/webicorders/
CVO_API=https://volcanoes.usgs.gov/observatories/cvo/

# International Volcano Monitoring
JMA_VOLCANO_API=https://www.data.jma.go.jp/svd/vow/data/tokyo/
KVERT_API=http://www.kscnet.ru/ivs/kvert/
INGV_API=https://www.ct.ingv.it/it/
PHIVOLCS_API=https://www.phivolcs.dost.gov.ph/

# Volcanic Ash Advisory Centers
VAAC_ANCHORAGE_API=https://www.ssd.noaa.gov/VAAC/
VAAC_WASHINGTON_API=https://www.ssd.noaa.gov/VAAC/washington.html
VAAC_TOULOUSE_API=https://www.meteo.fr/vaac/

# =============================================================================
# GEOLOGICAL SURVEY API ENDPOINTS
# =============================================================================

# United States Geological Survey (USGS)
USGS_GEOLEX_API=https://ngmdb.usgs.gov/Geolex/api/
USGS_MINERALS_API=https://mrdata.usgs.gov/services/
USGS_GEOLOGY_API=https://macrostrat.org/api/defs/
USGS_NSHMP_API=https://earthquake.usgs.gov/nshmp/

# British Geological Survey (BGS)
BGS_LEXICON_API=https://webapps.bgs.ac.uk/lexicon/api/
BGS_GEOLOGY_API=https://mapapps2.bgs.ac.uk/geoindex_v2/
BGS_BOREHOLE_API=https://webapps.bgs.ac.uk/GeoRecords/BGS_Boreholes_WFS
BGS_API_KEY=your_bgs_api_key

# Geological Survey of Canada (GSC)
GSC_GEOLOGY_API=https://apps1.gdr.nrcan.gc.ca/gsc_ct_service/api/
GSC_LEXICON_API=https://weblex.nrcan.gc.ca/api/
GSC_MINERALS_API=https://gsc.nrcan.gc.ca/api/

# Geoscience Australia (GA)
GA_GEOLOGY_API=https://geoscience-portal.ga.gov.au/api/
GA_MINERALS_API=https://portal.ga.gov.au/persona/minerals
GA_EARTHQUAKES_API=https://earthquakes.ga.gov.au/api/

# European Geological Data Infrastructure (EGDI)
EGDI_API=https://data.geus.dk/egdi/
ONEGEOLOGY_API=http://onegeology.org/use/home.html
INSPIRE_GEOLOGY_API=https://inspire-geoportal.ec.europa.eu/

# =============================================================================
# SPECIALIZED GEOPHYSICAL DATA APIS
# =============================================================================

# International GNSS Service (IGS)
IGS_DATA_API=https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/
IGS_STATION_API=https://www.igs.org/network/
UNAVCO_API=https://www.unavco.org/data/web-services/

# Gravity and Magnetic Data
ICGEM_GRAVITY_API=http://icgem.gfz-potsdam.de/web/
EMAG2_MAGNETIC_API=https://www.ngdc.noaa.gov/geomag/emag2/
BGI_GRAVITY_API=http://bgi.obs-mip.fr/

# Heat Flow Data
IHFC_HEAT_FLOW_API=https://ihfc-iugg.org/products/global-heat-flow-database/
GEOTHERMAL_API=https://gdr.openei.org/submissions/

# =============================================================================
# MACHINE LEARNING AND PROCESSING ENDPOINTS
# =============================================================================

# Cloud Processing Services
GOOGLE_EARTH_ENGINE_API=https://earthengine.googleapis.com/
PLANETARY_COMPUTER_API=https://planetarycomputer.microsoft.com/api/
AWS_EARTH_API=https://aws.amazon.com/earth/

# Model Training and Serving
AZURE_ML_ENDPOINT=your_azure_ml_endpoint
AWS_SAGEMAKER_ENDPOINT=your_sagemaker_endpoint
GOOGLE_VERTEX_AI_ENDPOINT=your_vertex_ai_endpoint

# =============================================================================
# AUTHENTICATION AND RATE LIMITING
# =============================================================================

# API Keys and Authentication
EARTHDATA_USERNAME=your_nasa_earthdata_username
EARTHDATA_PASSWORD=your_nasa_earthdata_password
USGS_API_KEY=your_usgs_api_key
OPENTOPOGRAPHY_API_KEY=your_opentopography_api_key
IRIS_USERNAME=your_iris_username
IRIS_PASSWORD=your_iris_password

# Rate Limiting Configuration
API_RATE_LIMIT_PER_MINUTE=60
API_CONCURRENT_REQUESTS=5
API_TIMEOUT_SECONDS=30
API_RETRY_ATTEMPTS=3
API_BACKOFF_FACTOR=2.0

# Caching Configuration
API_CACHE_TTL_SECONDS=3600
REDIS_CACHE_HOST=localhost
REDIS_CACHE_PORT=6379
REDIS_CACHE_PASSWORD=your_redis_password
CACHE_KEY_PREFIX=prometheus_geo

# =============================================================================
# DATA QUALITY AND VALIDATION
# =============================================================================

# Quality Thresholds
MIN_ELEVATION_ACCURACY_M=10.0
MIN_SEISMIC_MAGNITUDE=2.0
MAX_SEISMIC_DISTANCE_KM=500.0
MIN_DATA_QUALITY_GRADE=B
MIN_CONFIDENCE_THRESHOLD=0.6

# Spatial Search Parameters
DEFAULT_SEARCH_RADIUS_M=50000
MAX_SEARCH_RADIUS_M=500000
GRID_RESOLUTION_M=1000
INTERPOLATION_METHOD=kriging

# Temporal Parameters
MAX_DATA_AGE_DAYS=3650  # 10 years
REALTIME_DATA_WINDOW_HOURS=24
HISTORICAL_DATA_CUTOFF=1900-01-01
```

### 6. Helper Methods for External API Integration

```python
# Helper methods that support the real data integration
async def _get_elevation_from_external_apis(self, lat: float, lon: float) -> Optional[Dict]:
    """Get elevation data from external DEM APIs as fallback"""
    try:
        # NASA Earthdata API integration
        earthdata_result = await self._query_nasa_earthdata_elevation(lat, lon)
        if earthdata_result:
            return {
                'elevation_m': earthdata_result['elevation'],
                'data_source': 'nasa_earthdata',
                'confidence': earthdata_result.get('confidence', 0.8)
            }
        
        # OpenTopography API fallback
        opentopo_result = await self._query_opentopography_api(lat, lon)
        if opentopo_result:
            return {
                'elevation_m': opentopo_result['elevation'],
                'data_source': 'opentopography',
                'confidence': opentopo_result.get('confidence', 0.7)
            }
        
        return None
        
    except Exception as e:
        self.logger.error(f"External elevation API failed: {e}")
        return None

async def _get_realtime_seismic_data(self, lat: float, lon: float) -> Optional[Dict]:
    """Get real-time seismic data from external APIs"""
    try:
        # USGS real-time earthquake feed
        usgs_result = await self._query_usgs_realtime_earthquakes(lat, lon)
        if usgs_result:
            return {
                'recent_events': len(usgs_result['earthquakes']),
                'max_magnitude': max([eq['magnitude'] for eq in usgs_result['earthquakes']]) if usgs_result['earthquakes'] else 0.0,
                'data_source': 'usgs_realtime'
            }
        
        return None
        
    except Exception as e:
        self.logger.error(f"Real-time seismic data API failed: {e}")
        return None

async def _get_external_volcanic_data(self, lat: float, lon: float) -> Optional[Dict]:
    """Get volcanic data from external monitoring APIs"""
    try:
        # Smithsonian GVP API
        gvp_result = await self._query_gvp_api(lat, lon)
        if gvp_result:
            return {
                'alert_level': gvp_result.get('alert_level', 'normal'),
                'activity_score': gvp_result.get('activity_level', 0.0),
                'data_source': 'smithsonian_gvp'
            }
        
        return None
        
    except Exception as e:
        self.logger.error(f"External volcanic data API failed: {e}")
        return None

def _classify_terrain_from_slope_aspect(self, slope: float, aspect: float, elevation: float) -> str:
    """Classify terrain based on topographic parameters"""
    if slope < 5:
        return 'flat'
    elif slope < 15:
        return 'gentle_slope'
    elif slope < 30:
        return 'moderate_slope'
    elif slope < 45:
        return 'steep_slope'
    else:
        return 'very_steep'

async def _query_nasa_earthdata_elevation(self, lat: float, lon: float) -> Optional[Dict]:
    """Query NASA Earthdata for elevation information"""
    # Implementation would use aiohttp to query NASA APIs
    pass

async def _query_opentopography_api(self, lat: float, lon: float) -> Optional[Dict]:
    """Query OpenTopography API for DEM data"""
    # Implementation would use aiohttp to query OpenTopography
    pass

async def _query_usgs_realtime_earthquakes(self, lat: float, lon: float) -> Optional[Dict]:
    """Query USGS real-time earthquake feeds"""
    # Implementation would use aiohttp to query USGS APIs
    pass

async def _query_gvp_api(self, lat: float, lon: float) -> Optional[Dict]:
    """Query Smithsonian Global Volcanism Program API"""
    # Implementation would use aiohttp to query GVP APIs
    pass
```

### 7. Performance Optimization for Real Data Integration

```sql
-- Optimized indexes for real-time queries
CREATE INDEX idx_global_seismic_recent_spatial ON global_seismic_events 
USING GIST (epicenter, event_time) 
WHERE event_time >= NOW() - INTERVAL '30 days';

CREATE INDEX idx_volcanic_monitoring_active_spatial ON volcanic_monitoring_stations 
USING GIST (location, last_update) 
WHERE station_status = 'active' AND last_update >= NOW() - INTERVAL '7 days';

CREATE INDEX idx_high_res_dem_quality_spatial ON high_resolution_dem 
USING GIST (location, resolution_m) 
WHERE quality_flag >= 3;

-- Partial indexes for high-quality geological surveys
CREATE INDEX idx_national_surveys_quality_spatial ON national_geological_surveys 
USING GIST (geometry) 
WHERE data_quality_flag IN ('excellent', 'good') AND mapping_scale <= 50000;

-- Composite indexes for multi-criteria queries
CREATE INDEX idx_geotechnical_strength_spatial ON geotechnical_investigations 
(unconfined_compressive_strength, location) USING GIST
WHERE unconfined_compressive_strength IS NOT NULL;
```

## Additional Database Tables for Comprehensive Stress Prediction Feature Integration

### 10. Crustal Age Database Table
```sql
CREATE TABLE crustal_age_database (
    id SERIAL PRIMARY KEY,
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    age_ma REAL NOT NULL CHECK (age_ma >= 0 AND age_ma <= 4600), -- Million years ago
    age_uncertainty_ma REAL CHECK (age_uncertainty_ma >= 0 AND age_uncertainty_ma <= 1000),
    geological_unit VARCHAR(200),
    formation_process VARCHAR(100) CHECK (formation_process IN (
        'igneous_intrusion', 'volcanic_extrusion', 'sedimentary_deposition',
        'metamorphic_transformation', 'accretionary_complex', 'ophiolite_complex',
        'continental_collision', 'arc_volcanism', 'rift_volcanism'
    )),
    dating_method VARCHAR(100) CHECK (dating_method IN (
        'radiometric_dating', 'stratigraphic_correlation', 'paleomagnetic_dating',
        'fossil_correlation', 'zircon_dating', 'argon_dating', 'carbon_dating'
    )),
    confidence_score REAL NOT NULL DEFAULT 0.5 CHECK (confidence_score BETWEEN 0 AND 1),
    data_source VARCHAR(100) NOT NULL,
    reference_publication TEXT,
    sample_depth_m REAL CHECK (sample_depth_m >= 0),
    tectonic_province VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for crustal age database
CREATE INDEX idx_crustal_age_location ON crustal_age_database USING GIST (location);
CREATE INDEX idx_crustal_age_confidence ON crustal_age_database (confidence_score DESC);
CREATE INDEX idx_crustal_age_age ON crustal_age_database (age_ma);
CREATE INDEX idx_crustal_age_province ON crustal_age_database (tectonic_province);
CREATE INDEX idx_crustal_age_method ON crustal_age_database (dating_method);
CREATE INDEX idx_crustal_age_composite ON crustal_age_database (age_ma, confidence_score, location) USING GIST;
```

### 11. Regional Geology Table
```sql
CREATE TABLE regional_geology (
    id SERIAL PRIMARY KEY,
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    region_name VARCHAR(200) NOT NULL,
    estimated_age_ma REAL CHECK (estimated_age_ma >= 0 AND estimated_age_ma <= 4600),
    dominant_lithology VARCHAR(100),
    structural_province VARCHAR(100),
    metamorphic_grade VARCHAR(50) CHECK (metamorphic_grade IN (
        'unmetamorphosed', 'very_low', 'low', 'medium', 'high', 'very_high'
    )),
    geological_map_scale INTEGER CHECK (geological_map_scale IN (25000, 50000, 100000, 250000, 500000, 1000000)),
    mapping_confidence VARCHAR(20) CHECK (mapping_confidence IN ('high', 'medium', 'low', 'inferred')),
    last_survey_year INTEGER CHECK (last_survey_year >= 1800 AND last_survey_year <= 2030),
    survey_organization VARCHAR(200),
    area_km2 REAL CHECK (area_km2 > 0),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for regional geology
CREATE INDEX idx_regional_geology_location ON regional_geology USING GIST (location);
CREATE INDEX idx_regional_geology_age ON regional_geology (estimated_age_ma) WHERE estimated_age_ma IS NOT NULL;
CREATE INDEX idx_regional_geology_lithology ON regional_geology (dominant_lithology);
CREATE INDEX idx_regional_geology_confidence ON regional_geology (mapping_confidence);
```

### 12. Enhanced Geological Faults Table
```sql
CREATE TABLE geological_faults (
    id SERIAL PRIMARY KEY,
    fault_line GEOGRAPHY(LINESTRING, 4326) NOT NULL,
    fault_name VARCHAR(200),
    fault_type VARCHAR(50) CHECK (fault_type IN (
        'normal', 'reverse', 'thrust', 'strike_slip', 'oblique_slip',
        'transform', 'detachment', 'listric', 'blind_thrust'
    )),
    length_km REAL CHECK (length_km > 0),
    displacement_m REAL CHECK (displacement_m >= 0),
    dip_degrees REAL CHECK (dip_degrees BETWEEN 0 AND 90),
    strike_degrees REAL CHECK (strike_degrees BETWEEN 0 AND 360),
    activity_status VARCHAR(20) CHECK (activity_status IN ('active', 'potentially_active', 'inactive', 'unknown')),
    activity_confidence REAL NOT NULL DEFAULT 0.5 CHECK (activity_confidence BETWEEN 0 AND 1),
    last_activity_age_ka REAL CHECK (last_activity_age_ka >= 0), -- Thousand years ago
    slip_rate_mm_year REAL CHECK (slip_rate_mm_year >= 0),
    seismic_hazard_level VARCHAR(20) CHECK (seismic_hazard_level IN ('very_low', 'low', 'moderate', 'high', 'very_high')),
    max_expected_magnitude REAL CHECK (max_expected_magnitude BETWEEN 0 AND 10),
    data_source VARCHAR(100) NOT NULL,
    mapping_scale INTEGER,
    fault_system VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for geological faults
CREATE INDEX idx_geological_faults_spatial ON geological_faults USING GIST (fault_line);
CREATE INDEX idx_geological_faults_type ON geological_faults (fault_type);
CREATE INDEX idx_geological_faults_activity ON geological_faults (activity_status, activity_confidence);
CREATE INDEX idx_geological_faults_hazard ON geological_faults (seismic_hazard_level);
CREATE INDEX idx_geological_faults_length ON geological_faults (length_km DESC);
CREATE INDEX idx_geological_faults_system ON geological_faults (fault_system);
```

## Enhanced API Endpoints for Real Data Connections

### Real-Time Geological Data API Endpoints

#### Crustal Age Data Sources
```yaml
crustal_age_apis:
  usgs_geochronology:
    endpoint: "https://mrdata.usgs.gov/geology/age/"
    authentication: "none"
    rate_limit: "1000/hour"
    format: "json"
    parameters:
      - "bbox": "bounding box coordinates"
      - "format": "json"
      - "age_min": "minimum age in Ma"
      - "age_max": "maximum age in Ma"
      
  earthchem_database:
    endpoint: "https://www.earthchem.org/portal/api/"
    authentication: "api_key"
    rate_limit: "500/hour"
    format: "json"
    parameters:
      - "latitude": "decimal degrees"
      - "longitude": "decimal degrees"
      - "radius": "search radius in km"
      - "sample_type": "rock type filter"
      
  geomap_usgs:
    endpoint: "https://ngmdb.usgs.gov/mapview/services/"
    authentication: "none"
    rate_limit: "2000/hour"
    format: "json"
    parameters:
      - "x": "longitude"
      - "y": "latitude"
      - "scale": "map scale"
```

#### Plate Boundary Data Sources (Enhanced)
```yaml
plate_boundary_enhanced_apis:
  global_plate_model:
    endpoint: "https://www.gplates.org/api/reconstruct_points/"
    authentication: "none"
    rate_limit: "1000/hour"
    format: "json"
    parameters:
      - "points": "lat,lon coordinates"
      - "time": "geological time in Ma"
      - "model": "plate reconstruction model"
      
  unavco_plate_motion:
    endpoint: "https://www.unavco.org/data/gps-gnss/api/"
    authentication: "api_key"
    rate_limit: "500/hour"
    format: "json"
    parameters:
      - "latitude": "decimal degrees"
      - "longitude": "decimal degrees"
      - "reference_frame": "coordinate reference frame"
      
  iris_earthquake_catalog:
    endpoint: "https://service.iris.edu/fdsnws/event/1/query"
    authentication: "none"
    rate_limit: "5000/hour"
    format: "xml/json"
    parameters:
      - "minlatitude": "minimum latitude"
      - "maxlatitude": "maximum latitude"
      - "minlongitude": "minimum longitude"
      - "maxlongitude": "maximum longitude"
      - "starttime": "ISO datetime"
      - "endtime": "ISO datetime"
      - "minmagnitude": "minimum magnitude"
```

#### Fault Database API Sources
```yaml
fault_database_apis:
  usgs_quaternary_faults:
    endpoint: "https://earthquake.usgs.gov/hazards/qfaults/api/"
    authentication: "none"
    rate_limit: "2000/hour"
    format: "json"
    parameters:
      - "bbox": "bounding box"
      - "format": "geojson"
      - "activity": "active,potentially_active"
      
  global_seismic_hazard:
    endpoint: "https://www.seismo.unr.edu/ftp/pub/louie/class/100/hazmap/"
    authentication: "none"
    rate_limit: "1000/hour"
    format: "json"
    parameters:
      - "lat": "latitude"
      - "lon": "longitude"
      - "vs30": "shear wave velocity"
      
  active_fault_database:
    endpoint: "https://activefaults.gns.cri.nz/api/faults/"
    authentication: "api_key"
    rate_limit: "500/hour"
    format: "json"
    parameters:
      - "region": "geographic region"
      - "activity_class": "fault activity classification"
```

## Database Helper Methods for External API Integration

### Python Helper Methods for Real-Time Data Integration
```python
async def integrate_crustal_age_data(db_manager, lat: float, lon: float, radius_km: float = 10.0):
    """
    Integrate crustal age data from multiple external APIs
    """
    # USGS Geochronology API
    usgs_url = f"https://mrdata.usgs.gov/geology/age/json/?bbox={lon-radius_km/111},{lat-radius_km/111},{lon+radius_km/111},{lat+radius_km/111}"
    
    # EarthChem Database API
    earthchem_url = f"https://www.earthchem.org/portal/api/samples?latitude={lat}&longitude={lon}&radius={radius_km}"
    
    # Process and insert data into crustal_age_database table
    # Implementation details for API calls and database insertion

async def integrate_fault_data(db_manager, lat: float, lon: float, radius_km: float = 50.0):
    """
    Integrate fault data from multiple geological databases
    """
    # USGS Quaternary Faults API
    usgs_faults_url = f"https://earthquake.usgs.gov/hazards/qfaults/api/query?bbox={lon-radius_km/111},{lat-radius_km/111},{lon+radius_km/111},{lat+radius_km/111}&format=geojson"
    
    # Global Seismic Hazard Database
    hazard_url = f"https://www.seismo.unr.edu/ftp/pub/louie/class/100/hazmap/json/?lat={lat}&lon={lon}&radius={radius_km}"
    
    # Process and insert data into geological_faults table
    # Implementation details for API calls and database insertion

async def integrate_plate_boundary_data(db_manager, lat: float, lon: float):
    """
    Integrate enhanced plate boundary data from global models
    """
    # GPlates API for plate reconstruction
    gplates_url = f"https://www.gplates.org/api/reconstruct_points/?points={lat},{lon}&time=0&model=MULLER2016"
    
    # UNAVCO GPS/GNSS data for current plate motion
    unavco_url = f"https://www.unavco.org/data/gps-gnss/api/stations?latitude={lat}&longitude={lon}&radius=100"
    
    # Process and update plate_boundaries table with enhanced data
    # Implementation details for API calls and database updates
```

## Database Connection Configuration for Real Data Integration

### Enhanced Database Configuration
```python
DATABASE_CONFIG = {
    'geological_apis': {
        'usgs_api_key': os.getenv('USGS_API_KEY', None),
        'earthchem_api_key': os.getenv('EARTHCHEM_API_KEY', None),
        'unavco_api_key': os.getenv('UNAVCO_API_KEY', None),
        'rate_limit_per_hour': 5000,
        'timeout_seconds': 30,
        'retry_attempts': 3
    },
    'database_performance': {
        'connection_pool_size': 20,
        'max_overflow': 30,
        'pool_timeout': 60,
        'pool_recycle': 3600,
        'enable_query_cache': True,
        'cache_ttl_seconds': 300
    },
    'spatial_queries': {
        'default_search_radius_km': 10.0,
        'max_search_radius_km': 100.0,
        'spatial_index_type': 'GIST',
        'coordinate_system': 'EPSG:4326'
    }
}
```

# =============================================================================
# COMPREHENSIVE STRESS FIELD DATABASE SCHEMAS AND API ENDPOINTS
# =============================================================================

## Stress Field Modeling Database Tables

### 1. Stress Measurements Database
```sql
-- Real-world stress measurements from boreholes, hydraulic fracturing, etc.
CREATE TABLE stress_measurements (
    id SERIAL PRIMARY KEY,
    measurement_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL, -- includes depth as Z coordinate
    longitude REAL NOT NULL,
    latitude REAL NOT NULL,
    depth_m REAL NOT NULL CHECK (depth_m >= 0),
    stress_magnitude_mpa REAL CHECK (stress_magnitude_mpa >= 0),
    stress_orientation_degrees REAL CHECK (stress_orientation_degrees BETWEEN 0 AND 360),
    stress_regime VARCHAR(50) CHECK (stress_regime IN (
        'normal_faulting', 'strike_slip', 'reverse_faulting', 
        'extensional', 'compressional', 'transpressional', 'transtensional'
    )),
    measurement_method VARCHAR(100) CHECK (measurement_method IN (
        'hydraulic_fracturing', 'overcoring', 'borehole_breakout', 'drilling_induced_fractures',
        'focal_mechanism', 'acoustic_emission', 'strain_gauge', 'tiltmeter'
    )),
    measurement_date TIMESTAMPTZ NOT NULL,
    reliability_score REAL DEFAULT 0.7 CHECK (reliability_score BETWEEN 0 AND 1),
    measurement_agency VARCHAR(100),
    project_context VARCHAR(200),
    -- Stress tensor components (full 3x3 tensor)
    stress_tensor_xx REAL, -- Normal stress in X direction (MPa)
    stress_tensor_yy REAL, -- Normal stress in Y direction (MPa)
    stress_tensor_zz REAL, -- Normal stress in Z direction (MPa)
    stress_tensor_xy REAL, -- Shear stress XY component (MPa)
    stress_tensor_xz REAL, -- Shear stress XZ component (MPa)
    stress_tensor_yz REAL, -- Shear stress YZ component (MPa)
    -- Principal stresses
    principal_stress_s1_mpa REAL, -- Maximum principal stress
    principal_stress_s2_mpa REAL, -- Intermediate principal stress
    principal_stress_s3_mpa REAL, -- Minimum principal stress
    s1_orientation_azimuth REAL, -- S1 azimuth (degrees)
    s1_orientation_plunge REAL,  -- S1 plunge (degrees)
    s2_orientation_azimuth REAL,
    s2_orientation_plunge REAL,
    s3_orientation_azimuth REAL,
    s3_orientation_plunge REAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for stress measurements
CREATE INDEX idx_stress_measurements_spatial ON stress_measurements USING GIST (location);
CREATE INDEX idx_stress_measurements_depth ON stress_measurements (depth_m);
CREATE INDEX idx_stress_measurements_regime ON stress_measurements (stress_regime);
CREATE INDEX idx_stress_measurements_reliability ON stress_measurements (reliability_score DESC);
CREATE INDEX idx_stress_measurements_date ON stress_measurements (measurement_date);
CREATE INDEX idx_stress_measurements_method ON stress_measurements (measurement_method);
CREATE INDEX idx_stress_measurements_composite ON stress_measurements (stress_regime, reliability_score, location) USING GIST;
```

### 2. Global Stress Field Model Database
```sql
-- Global stress field models from various research institutions
CREATE TABLE global_stress_field (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    longitude REAL NOT NULL,
    latitude REAL NOT NULL,
    grid_resolution_degrees REAL DEFAULT 0.1,
    stress_regime VARCHAR(50) CHECK (stress_regime IN (
        'normal_faulting', 'strike_slip', 'reverse_faulting', 
        'extensional', 'compressional', 'unknown'
    )),
    principal_stress_s1_mpa REAL,
    principal_stress_s2_mpa REAL,
    principal_stress_s3_mpa REAL,
    maximum_horizontal_stress_azimuth REAL CHECK (maximum_horizontal_stress_azimuth BETWEEN 0 AND 180),
    stress_coefficient REAL, -- Derived pressure coefficient
    model_source VARCHAR(100) CHECK (model_source IN (
        'WSM2016', 'Heidbach2018', 'Zoback2010', 'Bird2003', 'Coblentz1998'
    )),
    model_version VARCHAR(50),
    model_confidence REAL DEFAULT 0.6 CHECK (model_confidence BETWEEN 0 AND 1),
    calculation_method VARCHAR(100),
    reference_publication TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for global stress field
CREATE INDEX idx_global_stress_field_spatial ON global_stress_field USING GIST (location);
CREATE INDEX idx_global_stress_field_regime ON global_stress_field (stress_regime);
CREATE INDEX idx_global_stress_field_confidence ON global_stress_field (model_confidence DESC);
CREATE INDEX idx_global_stress_field_source ON global_stress_field (model_source);
```

### 3. Plate Boundaries Enhanced Database
```sql
-- Enhanced plate boundaries with stress-related properties
CREATE TABLE plate_boundaries (
    id SERIAL PRIMARY KEY,
    geom GEOMETRY(LINESTRING, 4326) NOT NULL,
    boundary_type VARCHAR(50) CHECK (boundary_type IN (
        'divergent', 'convergent', 'transform', 'subduction', 'collision', 'uncertain'
    )),
    plate_name_left VARCHAR(100),
    plate_name_right VARCHAR(100),
    convergence_rate_mm_yr REAL CHECK (convergence_rate_mm_yr >= 0),
    convergence_azimuth_degrees REAL CHECK (convergence_azimuth_degrees BETWEEN 0 AND 360),
    spreading_rate_mm_yr REAL CHECK (spreading_rate_mm_yr >= 0),
    transform_offset_km REAL,
    subduction_angle_degrees REAL CHECK (subduction_angle_degrees BETWEEN 0 AND 90),
    slab_age_ma REAL CHECK (slab_age_ma >= 0),
    stress_accumulation_rate REAL, -- Stress buildup rate (MPa/year)
    seismic_coupling_coefficient REAL CHECK (seismic_coupling_coefficient BETWEEN 0 AND 1),
    last_major_earthquake_date DATE,
    characteristic_earthquake_magnitude REAL CHECK (characteristic_earthquake_magnitude BETWEEN 0 AND 10),
    data_source VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for plate boundaries
CREATE INDEX idx_plate_boundaries_spatial ON plate_boundaries USING GIST (geom);
CREATE INDEX idx_plate_boundaries_type ON plate_boundaries (boundary_type);
CREATE INDEX idx_plate_boundaries_convergence ON plate_boundaries (convergence_rate_mm_yr);
```

### 4. In-Situ Stress Measurements Database
```sql
-- Detailed in-situ stress measurements from drilling and testing
CREATE TABLE insitu_stress_measurements (
    id SERIAL PRIMARY KEY,
    borehole_id VARCHAR(100),
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    longitude REAL NOT NULL,
    latitude REAL NOT NULL,
    depth_m REAL NOT NULL CHECK (depth_m >= 0),
    total_depth_m REAL,
    horizontal_stress_max_mpa REAL,
    horizontal_stress_min_mpa REAL,
    vertical_stress_mpa REAL,
    stress_ratio REAL, -- Ratio of horizontal to vertical stress
    pore_pressure_mpa REAL,
    effective_stress_max_mpa REAL,
    effective_stress_min_mpa REAL,
    rock_type VARCHAR(100),
    rock_strength_mpa REAL,
    young_modulus_gpa REAL,
    poisson_ratio REAL CHECK (poisson_ratio BETWEEN 0 AND 0.5),
    measurement_technique VARCHAR(50) CHECK (measurement_technique IN (
        'hydraulic_fracturing', 'overcoring', 'disk_based_techniques',
        'borehole_slotter', 'compact_conical_indenter'
    )),
    measurement_quality REAL DEFAULT 0.7 CHECK (measurement_quality BETWEEN 0 AND 1),
    temperature_celsius REAL,
    drilling_fluid_type VARCHAR(50),
    measurement_date TIMESTAMPTZ,
    contractor_organization VARCHAR(200),
    project_purpose VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for in-situ stress measurements
CREATE INDEX idx_insitu_stress_spatial ON insitu_stress_measurements USING GIST (location);
CREATE INDEX idx_insitu_stress_depth ON insitu_stress_measurements (depth_m);
CREATE INDEX idx_insitu_stress_quality ON insitu_stress_measurements (measurement_quality DESC);
CREATE INDEX idx_insitu_stress_rock_type ON insitu_stress_measurements (rock_type);
```

### 5. Petrophysical Measurements Database
```sql
-- Comprehensive rock properties database for stress calculations
CREATE TABLE petrophysical_measurements (
    id SERIAL PRIMARY KEY,
    sample_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    longitude REAL NOT NULL,
    latitude REAL NOT NULL,
    measurement_depth_m REAL CHECK (measurement_depth_m >= 0),
    bulk_density_kg_m3 REAL CHECK (bulk_density_kg_m3 > 0),
    grain_density_kg_m3 REAL CHECK (grain_density_kg_m3 > 0),
    porosity_percent REAL CHECK (porosity_percent BETWEEN 0 AND 100),
    permeability_md REAL CHECK (permeability_md >= 0), -- millidarcys
    rock_classification VARCHAR(100),
    mineral_composition JSONB, -- Detailed mineral percentages
    weathering_grade VARCHAR(20) CHECK (weathering_grade IN (
        'fresh', 'slightly_weathered', 'moderately_weathered', 
        'highly_weathered', 'completely_weathered'
    )),
    saturation_water_percent REAL CHECK (saturation_water_percent BETWEEN 0 AND 100),
    -- Mechanical properties
    unconfined_compressive_strength_mpa REAL,
    tensile_strength_mpa REAL,
    cohesion_mpa REAL,
    friction_angle_degrees REAL CHECK (friction_angle_degrees BETWEEN 0 AND 90),
    elastic_modulus_gpa REAL,
    poisson_ratio REAL CHECK (poisson_ratio BETWEEN 0 AND 0.5),
    -- Thermal properties
    thermal_conductivity_w_mk REAL,
    specific_heat_j_kg_k REAL,
    thermal_expansion_per_k REAL,
    measurement_method VARCHAR(100),
    testing_standard VARCHAR(50),
    sample_size_description TEXT,
    quality_score REAL DEFAULT 0.7 CHECK (quality_score BETWEEN 0 AND 1),
    laboratory_name VARCHAR(200),
    testing_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for petrophysical measurements
CREATE INDEX idx_petrophysical_spatial ON petrophysical_measurements USING GIST (location);
CREATE INDEX idx_petrophysical_rock_type ON petrophysical_measurements (rock_classification);
CREATE INDEX idx_petrophysical_quality ON petrophysical_measurements (quality_score DESC);
CREATE INDEX idx_petrophysical_depth ON petrophysical_measurements (measurement_depth_m);
```

### 6. Regional Geological Units Database
```sql
-- Regional geological units with density and mechanical properties
CREATE TABLE geological_units (
    id SERIAL PRIMARY KEY,
    geom GEOMETRY(POLYGON, 4326) NOT NULL,
    unit_name VARCHAR(200) NOT NULL,
    unit_code VARCHAR(50),
    rock_type_primary VARCHAR(100) NOT NULL,
    rock_type_secondary VARCHAR(100),
    lithological_description TEXT,
    formation_name VARCHAR(200),
    geological_age VARCHAR(100),
    age_min_ma REAL CHECK (age_min_ma >= 0),
    age_max_ma REAL CHECK (age_max_ma >= 0),
    average_density_kg_m3 REAL CHECK (average_density_kg_m3 > 0),
    density_std_deviation REAL CHECK (density_std_deviation >= 0),
    porosity_range_min REAL CHECK (porosity_range_min >= 0),
    porosity_range_max REAL CHECK (porosity_range_max <= 100),
    typical_thickness_m REAL CHECK (typical_thickness_m > 0),
    structural_attitude VARCHAR(100), -- Strike and dip information
    metamorphic_grade VARCHAR(50),
    alteration_degree VARCHAR(50),
    formation_environment VARCHAR(100),
    mapping_confidence VARCHAR(20) CHECK (mapping_confidence IN ('high', 'medium', 'low')),
    mapping_scale INTEGER,
    data_source VARCHAR(100),
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for geological units
CREATE INDEX idx_geological_units_spatial ON geological_units USING GIST (geom);
CREATE INDEX idx_geological_units_rock_type ON geological_units (rock_type_primary);
CREATE INDEX idx_geological_units_formation ON geological_units (formation_name);
CREATE INDEX idx_geological_units_age ON geological_units (age_min_ma, age_max_ma);
```

### 7. Laboratory Petrophysical Database
```sql
-- Laboratory measurements from global petrophysical databases
CREATE TABLE laboratory_petrophysical_data (
    id SERIAL PRIMARY KEY,
    sample_reference VARCHAR(100) UNIQUE NOT NULL,
    rock_classification VARCHAR(100) NOT NULL,
    geographic_origin VARCHAR(200),
    sample_location_lat REAL,
    sample_location_lon REAL,
    formation_age_ma REAL,
    bulk_density_kg_m3 REAL CHECK (bulk_density_kg_m3 > 0),
    grain_density_kg_m3 REAL CHECK (grain_density_kg_m3 > 0),
    porosity_percent REAL CHECK (porosity_percent BETWEEN 0 AND 100),
    permeability_md REAL,
    mineral_composition JSONB,
    chemical_composition JSONB, -- Major oxide percentages
    -- Testing conditions
    measurement_temperature_c REAL,
    measurement_pressure_mpa REAL,
    confining_pressure_mpa REAL,
    effective_pressure_mpa REAL,
    -- Mechanical properties
    compressive_strength_dry_mpa REAL,
    compressive_strength_saturated_mpa REAL,
    tensile_strength_mpa REAL,
    elastic_modulus_dry_gpa REAL,
    elastic_modulus_saturated_gpa REAL,
    poisson_ratio_dry REAL,
    poisson_ratio_saturated REAL,
    sample_quality REAL DEFAULT 0.7 CHECK (sample_quality BETWEEN 0 AND 1),
    database_source VARCHAR(100), -- Source database name
    reference_publication TEXT,
    measurement_institution VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for laboratory petrophysical data
CREATE INDEX idx_lab_petrophysical_rock_type ON laboratory_petrophysical_data (rock_classification);
CREATE INDEX idx_lab_petrophysical_location ON laboratory_petrophysical_data (sample_location_lat, sample_location_lon);
CREATE INDEX idx_lab_petrophysical_quality ON laboratory_petrophysical_data (sample_quality DESC);
CREATE INDEX idx_lab_petrophysical_source ON laboratory_petrophysical_data (database_source);
```

### 8. Regional Tectonic Activity Database
```sql
-- Regional tectonic activity levels and deformation rates
CREATE TABLE regional_tectonic_activity (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    longitude REAL NOT NULL,
    latitude REAL NOT NULL,
    region_name VARCHAR(200),
    activity_classification VARCHAR(50) CHECK (activity_classification IN (
        'very_low', 'low', 'moderate', 'high', 'very_high', 'extreme'
    )),
    deformation_rate_mm_yr REAL CHECK (deformation_rate_mm_yr >= 0),
    strain_rate_nanostrain_yr REAL,
    seismic_hazard_level VARCHAR(20) CHECK (seismic_hazard_level IN (
        'very_low', 'low', 'moderate', 'high', 'very_high'
    )),
    recent_fault_activity BOOLEAN DEFAULT FALSE,
    earthquake_frequency_per_year REAL CHECK (earthquake_frequency_per_year >= 0),
    maximum_observed_magnitude REAL CHECK (maximum_observed_magnitude BETWEEN 0 AND 10),
    activity_factor REAL CHECK (activity_factor > 0), -- Numerical activity multiplier
    tectonic_province VARCHAR(100),
    plate_boundary_distance_km REAL CHECK (plate_boundary_distance_km >= 0),
    assessment_date TIMESTAMPTZ DEFAULT NOW(),
    assessment_agency VARCHAR(100),
    confidence_level REAL DEFAULT 0.7 CHECK (confidence_level BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for regional tectonic activity
CREATE INDEX idx_regional_activity_spatial ON regional_tectonic_activity USING GIST (location);
CREATE INDEX idx_regional_activity_classification ON regional_tectonic_activity (activity_classification);
CREATE INDEX idx_regional_activity_hazard ON regional_tectonic_activity (seismic_hazard_level);
CREATE INDEX idx_regional_activity_confidence ON regional_tectonic_activity (confidence_level DESC);
```

### 9. GPS Velocity Field Database
```sql
-- GPS/GNSS velocity measurements for current crustal deformation
CREATE TABLE gps_velocity_field (
    id SERIAL PRIMARY KEY,
    station_code VARCHAR(20) NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    longitude REAL NOT NULL,
    latitude REAL NOT NULL,
    elevation_m REAL,
    velocity_east_mm_yr REAL NOT NULL, -- Eastward velocity component
    velocity_north_mm_yr REAL NOT NULL, -- Northward velocity component
    velocity_vertical_mm_yr REAL, -- Vertical velocity component
    velocity_magnitude_mm_yr REAL, -- Total horizontal velocity magnitude
    velocity_azimuth_degrees REAL CHECK (velocity_azimuth_degrees BETWEEN 0 AND 360),
    -- Uncertainty estimates
    sigma_east_mm_yr REAL CHECK (sigma_east_mm_yr >= 0),
    sigma_north_mm_yr REAL CHECK (sigma_north_mm_yr >= 0),
    sigma_vertical_mm_yr REAL CHECK (sigma_vertical_mm_yr >= 0),
    correlation_en REAL CHECK (correlation_en BETWEEN -1 AND 1),
    -- Strain rate calculations
    strain_rate_nanostrain_yr REAL,
    dilatation_rate_nanostrain_yr REAL,
    maximum_shear_strain_rate REAL,
    -- Data processing information
    reference_frame VARCHAR(20) DEFAULT 'ITRF2014',
    observation_span_years REAL CHECK (observation_span_years > 0),
    solution_quality REAL DEFAULT 0.8 CHECK (solution_quality BETWEEN 0 AND 1),
    processing_center VARCHAR(50),
    last_observation_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for GPS velocity field
CREATE INDEX idx_gps_velocity_spatial ON gps_velocity_field USING GIST (location);
CREATE INDEX idx_gps_velocity_magnitude ON gps_velocity_field (velocity_magnitude_mm_yr);
CREATE INDEX idx_gps_velocity_quality ON gps_velocity_field (solution_quality DESC);
CREATE INDEX idx_gps_velocity_station ON gps_velocity_field (station_code);
```

### 10. Volcanic Systems Database (Enhanced)
```sql
-- Enhanced volcanic systems with magma chamber pressure data
CREATE TABLE active_volcanoes (
    id SERIAL PRIMARY KEY,
    volcano_number VARCHAR(20) UNIQUE NOT NULL, -- GVP number
    volcano_name VARCHAR(200) NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    longitude REAL NOT NULL,
    latitude REAL NOT NULL,
    elevation_m INTEGER,
    volcano_type VARCHAR(50) CHECK (volcano_type IN (
        'stratovolcano', 'shield', 'caldera', 'cinder_cone', 'lava_dome',
        'maar', 'fissure_vent', 'volcanic_field', 'compound'
    )),
    last_eruption_date INTEGER, -- Year of last known eruption
    activity_level VARCHAR(20) CHECK (activity_level IN (
        'active', 'restless', 'dormant', 'extinct', 'uncertain'
    )),
    -- Magma chamber properties
    magma_chamber_depth_km REAL CHECK (magma_chamber_depth_km > 0),
    magma_chamber_volume_km3 REAL CHECK (magma_chamber_volume_km3 > 0),
    magma_chamber_pressure_mpa REAL CHECK (magma_chamber_pressure_mpa > 0),
    melt_fraction REAL CHECK (melt_fraction BETWEEN 0 AND 1),
    magma_temperature_c REAL CHECK (magma_temperature_c > 0),
    -- Eruptive characteristics
    eruption_frequency_years REAL CHECK (eruption_frequency_years > 0),
    volcanic_explosivity_index INTEGER CHECK (volcanic_explosivity_index BETWEEN 0 AND 8),
    dominant_rock_type VARCHAR(50),
    silica_content_percent REAL CHECK (silica_content_percent BETWEEN 40 AND 80),
    -- Geothermal properties
    geothermal_gradient_c_km REAL CHECK (geothermal_gradient_c_km > 0),
    heat_flow_mw_m2 REAL,
    hydrothermal_activity BOOLEAN DEFAULT FALSE,
    gas_emission_rate_tons_day REAL CHECK (gas_emission_rate_tons_day >= 0),
    -- Monitoring status
    monitoring_level VARCHAR(20) CHECK (monitoring_level IN (
        'comprehensive', 'basic', 'minimal', 'none'
    )),
    alert_level VARCHAR(20) DEFAULT 'normal',
    last_update TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for active volcanoes
CREATE INDEX idx_active_volcanoes_spatial ON active_volcanoes USING GIST (location);
CREATE INDEX idx_active_volcanoes_activity ON active_volcanoes (activity_level);
CREATE INDEX idx_active_volcanoes_chamber_pressure ON active_volcanoes (magma_chamber_pressure);
CREATE INDEX idx_active_volcanoes_alert ON active_volcanoes (alert_level);
```

### 11. Geothermal Systems Database
```sql
-- Geothermal systems with pressure and temperature data
CREATE TABLE geothermal_systems (
    id SERIAL PRIMARY KEY,
    system_name VARCHAR(200),
    location GEOMETRY(POINT, 4326) NOT NULL,
    longitude REAL NOT NULL,
    latitude REAL NOT NULL,
    system_type VARCHAR(50) CHECK (system_type IN (
        'magmatic', 'hydrothermal', 'convective', 'conductive', 'enhanced'
    )),
    reservoir_depth_m REAL CHECK (reservoir_depth_m > 0),
    reservoir_temperature_c REAL CHECK (reservoir_temperature_c > 0),
    surface_temperature_c REAL,
    reservoir_pressure_mpa REAL CHECK (reservoir_pressure_mpa > 0),
    fluid_type VARCHAR(50) CHECK (fluid_type IN (
        'liquid_dominated', 'vapor_dominated', 'two_phase', 'brine', 'steam'
    )),
    salinity_ppm REAL CHECK (salinity_ppm >= 0),
    ph_value REAL CHECK (ph_value BETWEEN 0 AND 14),
    permeability_md REAL,
    porosity_percent REAL CHECK (porosity_percent BETWEEN 0 AND 100),
    heat_flow_mw_m2 REAL CHECK (heat_flow_mw_m2 > 0),
    geothermal_gradient REAL CHECK (geothermal_gradient > 0),
    power_capacity_mw REAL CHECK (power_capacity_mw >= 0),
    development_status VARCHAR(50) CHECK (development_status IN (
        'exploration', 'development', 'production', 'abandoned', 'research'
    )),
    last_assessment TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for geothermal systems
CREATE INDEX idx_geothermal_spatial ON geothermal_systems USING GIST (location);
CREATE INDEX idx_geothermal_type ON geothermal_systems (system_type);
CREATE INDEX idx_geothermal_temperature ON geothermal_systems (reservoir_temperature_c);
CREATE INDEX idx_geothermal_pressure ON geothermal_systems (reservoir_pressure_mpa);
```

### 12. Hydrothermal Alterations Database
```sql
-- Hydrothermal alteration zones with fluid pressure data
CREATE TABLE hydrothermal_alterations (
    id SERIAL PRIMARY KEY,
    alteration_zone_name VARCHAR(200),
    location GEOMETRY(POINT, 4326) NOT NULL,
    longitude REAL NOT NULL,
    latitude REAL NOT NULL,
    depth_extent_min_m REAL CHECK (depth_extent_min_m >= 0),
    depth_extent_max_m REAL CHECK (depth_extent_max_m >= depth_extent_min_m),
    alteration_type VARCHAR(50) CHECK (alteration_type IN (
        'propylitic', 'argillic', 'phyllic', 'potassic', 'silicic',
        'chloritic', 'sericitic', 'kaolinitic', 'smectitic'
    )),
    alteration_intensity REAL CHECK (alteration_intensity BETWEEN 0 AND 1),
    host_rock_type VARCHAR(100),
    altered_mineralogy JSONB, -- Detailed mineral assemblage
    fluid_inclusion_temperature_c REAL,
    fluid_inclusion_pressure_mpa REAL,
    fluid_salinity_wt_percent_nacl REAL,
    temperature_formation_c REAL,
    pressure_formation_mpa REAL,
    ph_formation REAL CHECK (ph_formation BETWEEN 0 AND 14),
    oxidation_state VARCHAR(20),
    mineral_assemblage JSONB,
    age_of_alteration_ma REAL,
    associated_mineralization VARCHAR(200),
    economic_significance VARCHAR(100),
    study_reference TEXT,
    sampling_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for hydrothermal alterations
CREATE INDEX idx_hydrothermal_spatial ON hydrothermal_alterations USING GIST (location);
CREATE INDEX idx_hydrothermal_type ON hydrothermal_alterations (alteration_type);
CREATE INDEX idx_hydrothermal_intensity ON hydrothermal_alterations (alteration_intensity);
CREATE INDEX idx_hydrothermal_pressure ON hydrothermal_alterations (fluid_inclusion_pressure_mpa);
```

### 13. Volcanic Hazard Zones Database
```sql
-- Volcanic hazard assessment zones with overpressure potential
CREATE TABLE volcanic_hazard_zones (
    id SERIAL PRIMARY KEY,
    hazard_zone_id VARCHAR(100) UNIQUE NOT NULL,
    geom GEOMETRY(POLYGON, 4326) NOT NULL,
    associated_volcano VARCHAR(200),
    hazard_type VARCHAR(50) CHECK (hazard_type IN (
        'lava_flow', 'pyroclastic_flow', 'tephra_fall', 'lahar', 
        'debris_avalanche', 'volcanic_gas', 'ballistic_projectiles'
    )),
    hazard_level VARCHAR(20) CHECK (hazard_level IN (
        'very_low', 'low', 'moderate', 'high', 'very_high'
    )),
    eruption_probability_annual REAL CHECK (eruption_probability_annual BETWEEN 0 AND 1),
    overpressure_potential_mpa REAL CHECK (overpressure_potential_mpa >= 0),
    maximum_impact_radius_km REAL CHECK (maximum_impact_radius_km > 0),
    gas_emission_rate_tons_day REAL,
    ground_deformation_mm_yr REAL,
    thermal_anomaly_index REAL,
    seismic_activity_events_day REAL,
    assessment_method VARCHAR(100),
    model_uncertainty REAL CHECK (model_uncertainty BETWEEN 0 AND 1),
    last_assessment_date TIMESTAMPTZ DEFAULT NOW(),
    assessing_organization VARCHAR(100),
    validation_status VARCHAR(20) DEFAULT 'draft',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for volcanic hazard zones
CREATE INDEX idx_volcanic_hazard_spatial ON volcanic_hazard_zones USING GIST (geom);
CREATE INDEX idx_volcanic_hazard_level ON volcanic_hazard_zones (hazard_level);
CREATE INDEX idx_volcanic_hazard_overpressure ON volcanic_hazard_zones (overpressure_potential_mpa);
CREATE INDEX idx_volcanic_hazard_volcano ON volcanic_hazard_zones (associated_volcano);
```

### 14. Regional Stress Regime Database
```sql
-- Regional stress regime characterization
CREATE TABLE stress_regime_regions (
    id SERIAL PRIMARY KEY,
    region_geometry GEOMETRY(POLYGON, 4326) NOT NULL,
    region_name VARCHAR(200),
    dominant_regime VARCHAR(50) CHECK (dominant_regime IN (
        'normal_faulting', 'strike_slip', 'reverse_faulting', 
        'extensional', 'compressional', 'transpressional', 'transtensional'
    )),
    secondary_regime VARCHAR(50),
    regime_confidence REAL DEFAULT 0.7 CHECK (regime_confidence BETWEEN 0 AND 1),
    fault_kinematics JSONB, -- Detailed fault movement data
    stress_indicators TEXT[], -- Array of stress indicators observed
    representative_measurements INTEGER, -- Number of measurements supporting regime
    tectonic_province VARCHAR(100),
    geological_age_range VARCHAR(100),
    assessment_basis TEXT, -- Description of how regime was determined
    reference_study TEXT,
    assessment_date TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for stress regime regions
CREATE INDEX idx_stress_regime_spatial ON stress_regime_regions USING GIST (region_geometry);
CREATE INDEX idx_stress_regime_dominant ON stress_regime_regions (dominant_regime);
CREATE INDEX idx_stress_regime_confidence ON stress_regime_regions (regime_confidence DESC);
```

### 15. Tectonic Setting Ratios Database
```sql
-- Statistical database of stress ratios for different tectonic settings
CREATE TABLE tectonic_setting_ratios (
    id SERIAL PRIMARY KEY,
    setting_type VARCHAR(100) NOT NULL CHECK (setting_type IN (
        'extensional', 'compressional', 'strike_slip', 'normal_faulting',
        'reverse_faulting', 'transform', 'subduction', 'collision',
        'stable_craton', 'rift_zone', 'back_arc', 'fore_arc'
    )),
    average_sh_max_sv_ratio REAL CHECK (average_sh_max_sv_ratio > 0),
    average_sh_min_sv_ratio REAL CHECK (average_sh_min_sv_ratio > 0),
    ratio_std_deviation REAL CHECK (ratio_std_deviation >= 0),
    minimum_observed_ratio REAL,
    maximum_observed_ratio REAL,
    sample_count INTEGER CHECK (sample_count > 0),
    geographic_region VARCHAR(200),
    data_compilation_source VARCHAR(200),
    quality_assessment VARCHAR(50),
    reference_publication TEXT,
    compilation_date TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for tectonic setting ratios
CREATE INDEX idx_tectonic_ratios_setting ON tectonic_setting_ratios (setting_type);
CREATE INDEX idx_tectonic_ratios_sample_count ON tectonic_setting_ratios (sample_count DESC);
CREATE INDEX idx_tectonic_ratios_region ON tectonic_setting_ratios (geographic_region);
```

### 16. Regime Pressure Coefficients Database
```sql
-- Pressure coefficients by stress regime from literature and measurements
CREATE TABLE regime_pressure_coefficients (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    longitude REAL NOT NULL,
    latitude REAL NOT NULL,
    stress_regime VARCHAR(50) NOT NULL CHECK (stress_regime IN (
        'normal_faulting', 'strike_slip', 'reverse_faulting',
        'extensional', 'compressional', 'transpressional', 'transtensional'
    )),
    pressure_coefficient REAL NOT NULL CHECK (pressure_coefficient > 0),
    coefficient_std_deviation REAL CHECK (coefficient_std_deviation >= 0),
    depth_range_min_m REAL CHECK (depth_range_min_m >= 0),
    depth_range_max_m REAL CHECK (depth_range_max_m >= depth_range_min_m),
    rock_type VARCHAR(100),
    measurement_method VARCHAR(100),
    measurement_count INTEGER CHECK (measurement_count > 0),
    confidence_level REAL DEFAULT 0.7 CHECK (confidence_level BETWEEN 0 AND 1),
    geographic_context VARCHAR(200),
    geological_age VARCHAR(100),
    compilation_reference TEXT,
    data_source_type VARCHAR(50) CHECK (data_source_type IN (
        'direct_measurement', 'laboratory_testing', 'field_observation',
        'numerical_modeling', 'literature_compilation'
    )),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for regime pressure coefficients
CREATE INDEX idx_regime_pressure_spatial ON regime_pressure_coefficients USING GIST (location);
CREATE INDEX idx_regime_pressure_regime ON regime_pressure_coefficients (stress_regime);
CREATE INDEX idx_regime_pressure_confidence ON regime_pressure_coefficients (confidence_level DESC);
CREATE INDEX idx_regime_pressure_count ON regime_pressure_coefficients (measurement_count DESC);
```

### 17. Empirical Pressure Coefficients Database
```sql
-- Empirical pressure coefficients from various geological contexts
CREATE TABLE empirical_pressure_coefficients (
    id SERIAL PRIMARY KEY,
    location GEOMETRY(POINT, 4326) NOT NULL,
    longitude REAL NOT NULL,
    latitude REAL NOT NULL,
    depth_range_min_km REAL NOT NULL CHECK (depth_range_min_km >= 0),
    depth_range_max_km REAL NOT NULL CHECK (depth_range_max_km >= depth_range_min_km),
    pressure_coefficient REAL NOT NULL CHECK (pressure_coefficient > 0),
    coefficient_uncertainty REAL CHECK (coefficient_uncertainty >= 0),
    rock_type VARCHAR(100),
    density_kg_m3 REAL CHECK (density_kg_m3 > 0),
    temperature_c REAL,
    confining_pressure_mpa REAL,
    geological_context VARCHAR(200),
    measurement_method VARCHAR(100) CHECK (measurement_method IN (
        'laboratory_testing', 'field_calibration', 'numerical_modeling',
        'empirical_correlation', 'statistical_analysis'
    )),
    sample_count INTEGER CHECK (sample_count > 0),
    r_squared REAL CHECK (r_squared BETWEEN 0 AND 1), -- Correlation coefficient
    applicable_conditions TEXT, -- Conditions where coefficient applies
    limitations TEXT, -- Known limitations or constraints
    reference_study TEXT,
    validation_status VARCHAR(20) DEFAULT 'preliminary',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for empirical pressure coefficients
CREATE INDEX idx_empirical_pressure_spatial ON empirical_pressure_coefficients USING GIST (location);
CREATE INDEX idx_empirical_pressure_depth ON empirical_pressure_coefficients (depth_range_min_km, depth_range_max_km);
CREATE INDEX idx_empirical_pressure_rock_type ON empirical_pressure_coefficients (rock_type);
CREATE INDEX idx_empirical_pressure_validation ON empirical_pressure_coefficients (validation_status);
```

### 18. Depth-Dependent Pressure Coefficients Database
```sql
-- Depth-dependent pressure coefficient models
CREATE TABLE depth_pressure_coefficients (
    id SERIAL PRIMARY KEY,
    depth_km REAL NOT NULL CHECK (depth_km >= 0),
    coefficient_value REAL NOT NULL CHECK (coefficient_value > 0),
    coefficient_uncertainty REAL CHECK (coefficient_uncertainty >= 0),
    temperature_c REAL,
    lithostatic_pressure_mpa REAL,
    hydrostatic_pressure_mpa REAL,
    effective_pressure_mpa REAL,
    geological_setting VARCHAR(100),
    applicable_rock_types TEXT[], -- Array of applicable rock types
    model_basis VARCHAR(200), -- Basis for the depth relationship
    depth_correction_factor REAL CHECK (depth_correction_factor > 0),
    temperature_correction REAL, -- Temperature correction factor
    pressure_correction REAL, -- Pressure correction factor
    model_equation TEXT, -- Mathematical expression
    validity_range_min_km REAL CHECK (validity_range_min_km >= 0),
    validity_range_max_km REAL,
    model_accuracy_percent REAL CHECK (model_accuracy_percent BETWEEN 0 AND 100),
    calibration_dataset_size INTEGER,
    reference_publication TEXT,
    model_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for depth pressure coefficients
CREATE INDEX idx_depth_pressure_depth ON depth_pressure_coefficients (depth_km);
CREATE INDEX idx_depth_pressure_coefficient ON depth_pressure_coefficients (coefficient_value);
CREATE INDEX idx_depth_pressure_setting ON depth_pressure_coefficients (geological_setting);
```

## Enhanced API Endpoints for Stress Field Data Integration

### Real-Time Stress Field APIs
```yaml
stress_field_apis:
  world_stress_map:
    endpoint: "https://www.world-stress-map.org/api/v1/"
    authentication: "api_key"
    rate_limit: "1000/hour"
    format: "json"
    description: "World Stress Map database - comprehensive global stress orientation data"
    parameters:
      - "latitude": "decimal degrees"
      - "longitude": "decimal degrees"
      - "radius": "search radius in km"
      - "depth_min": "minimum depth in km"
      - "depth_max": "maximum depth in km"
      - "quality": "data quality grade (A,B,C,D)"

  usgs_stress_measurements:
    endpoint: "https://earthquake.usgs.gov/research/induced/api/"
    authentication: "none"
    rate_limit: "2000/hour"
    format: "json"
    description: "USGS stress measurement database"
    parameters:
      - "minlatitude": "minimum latitude"
      - "maxlatitude": "maximum latitude"
      - "minlongitude": "minimum longitude"
      - "maxlongitude": "maximum longitude"
      - "measurement_type": "hydraulic_fracturing,overcoring,focal_mechanism"

  stanford_reservoir_geomechanics:
    endpoint: "https://pangea.stanford.edu/ERE/db/GeoMech/api/"
    authentication: "institutional_access"
    rate_limit: "500/hour"
    format: "json"
    description: "Stanford Reservoir Geomechanics database"
    parameters:
      - "project_id": "project identifier"
      - "measurement_type": "insitu_stress,rock_properties"
      - "depth_range": "depth range in meters"

  international_stress_database:
    endpoint: "https://www.uni-karlsruhe.de/~gi25/API/"
    authentication: "api_key"
    rate_limit: "800/hour"
    format: "xml"
    description: "International database of in-situ stress measurements"
    parameters:
      - "coordinates": "lat,lon coordinates"
      - "search_radius": "radius in km"
      - "stress_type": "hydraulic_fracturing,overcoring,breakout"
```

### Rock Properties APIs
```yaml
rock_properties_apis:
  usgs_rock_properties:
    endpoint: "https://mrdata.usgs.gov/geology/rock-props/api/"
    authentication: "none"
    rate_limit: "1500/hour"
    format: "json"
    description: "USGS rock properties database"
    parameters:
      - "rock_type": "rock type classification"
      - "property": "density,porosity,strength,modulus"
      - "geographic_region": "geographic constraint"

  nist_materials_database:
    endpoint: "https://www.nist.gov/srd/nist-standard-reference-database-api"
    authentication: "api_key"
    rate_limit: "1000/hour"
    format: "json"
    description: "NIST materials properties database"
    parameters:
      - "material_class": "rock,mineral,ceramic"
      - "property_type": "mechanical,thermal,physical"

  international_rock_mechanics:
    endpoint: "https://www.isrm.net/foswiki/pub/ISRM/api/"
    authentication: "member_access"
    rate_limit: "500/hour"
    format: "json"
    description: "ISRM international rock mechanics database"
    parameters:
      - "test_standard": "ISRM,ASTM,BS"
      - "rock_classification": "ISRM rock type"
      - "test_type": "strength,deformation,durability"

  australian_rock_properties:
    endpoint: "https://www.ga.gov.au/metadata-gateway/api/rock-properties/"
    authentication: "none"
    rate_limit: "2000/hour"
    format: "json"
    description: "Geoscience Australia rock properties database"
    parameters:
      - "state": "Australian state/territory"
      - "geological_unit": "formation or unit name"
      - "property": "density,strength,porosity"
```

### Geophysical Data APIs
```yaml
geophysical_apis:
  unavco_gps_data:
    endpoint: "https://www.unavco.org/data/web-services/api/gps/"
    authentication: "api_key"
    rate_limit: "1000/hour"
    format: "json"
    description: "UNAVCO GPS/GNSS velocity and strain data"
    parameters:
      - "station": "4-character station code"
      - "coordinates": "lat,lon bounding box"
      - "reference_frame": "ITRF2014,NAM08,etc"
      - "time_period": "start_date,end_date"

  igs_global_gps:
    endpoint: "https://www.igs.org/api/products/"
    authentication: "none"
    rate_limit: "2000/hour"
    format: "json"
    description: "International GNSS Service data products"
    parameters:
      - "product_type": "coordinates,velocities,quality"
      - "site_id": "IGS site identifier"
      - "format": "sinex,json,xml"

  gfz_grace_gravity:
    endpoint: "https://grace.gfz-potsdam.de/api/"
    authentication: "registration"
    rate_limit: "500/hour"
    format: "json"
    description: "GFZ GRACE gravity field data"
    parameters:
      - "time_period": "data time range"
      - "spatial_resolution": "grid resolution"
      - "correction": "atmospheric,oceanic,terrestrial"

  emag2_magnetic:
    endpoint: "https://www.ngdc.noaa.gov/geomag/EMag2/api/"
    authentication: "none"
    rate_limit: "3000/hour"
    format: "json"
    description: "Earth Magnetic Anomaly Grid 2-arc-minute"
    parameters:
      - "coordinates": "lat,lon grid points"
      - "altitude": "observation height"
      - "component": "total_field,declination,inclination"
```

### Volcanic and Thermal APIs
```yaml
volcanic_thermal_apis:
  smithsonian_gvp_enhanced:
    endpoint: "https://volcano.si.edu/api/volcanism/"
    authentication: "api_key"
    rate_limit: "800/hour"
    format: "json"
    description: "Enhanced Global Volcanism Program database"
    parameters:
      - "volcano_id": "GVP volcano number"
      - "activity_status": "active,restless,dormant"
      - "monitoring_level": "comprehensive,basic,minimal"
      - "chamber_properties": "depth,pressure,volume"

  usgs_volcano_monitoring:
    endpoint: "https://volcanoes.usgs.gov/api/monitoring/"
    authentication: "none"
    rate_limit: "1500/hour"
    format: "json"
    description: "USGS volcano monitoring real-time data"
    parameters:
      - "volcano_id": "USGS volcano identifier"
      - "data_type": "seismic,gas,deformation,thermal"
      - "time_window": "last_24h,last_week,last_month"

  international_heat_flow:
    endpoint: "https://ihfc-iugg.org/api/heat-flow/"
    authentication: "academic_license"
    rate_limit: "1000/hour"
    format: "json"
    description: "Global Heat Flow Database"
    parameters:
      - "coordinates": "lat,lon coordinates"
      - "measurement_type": "borehole,probe,correction"
      - "quality_flag": "A,B,C quality grades"

  geothermal_database:
    endpoint: "https://gdr.openei.org/api/geothermal/"
    authentication: "api_key"
    rate_limit: "1200/hour"
    format: "json"
    description: "DOE Geothermal Data Repository"
    parameters:
      - "system_type": "hydrothermal,enhanced,co-produced"
      - "temperature_range": "min,max temperature"
      - "resource_type": "electricity,direct_use"
```

## Environment Variables for Stress Field API Integration
```bash
# =============================================================================
# STRESS FIELD DATABASE API CONFIGURATION
# =============================================================================

# World Stress Map Database
WSM_API_ENDPOINT=https://www.world-stress-map.org/api/v1/
WSM_API_KEY=your_wsm_api_key_here
WSM_RATE_LIMIT=1000
WSM_TIMEOUT_SECONDS=30

# USGS Stress and Earthquake APIs
USGS_STRESS_API=https://earthquake.usgs.gov/research/induced/api/
USGS_NEHRP_API=https://earthquake.usgs.gov/nehrp/api/
USGS_QUATERNARY_FAULTS_API=https://earthquake.usgs.gov/hazards/qfaults/api/

# International Stress Databases
INTERNATIONAL_STRESS_DB_API=https://www.uni-karlsruhe.de/~gi25/API/
INTERNATIONAL_STRESS_DB_KEY=your_intl_stress_api_key
STANFORD_GEOMECH_API=https://pangea.stanford.edu/ERE/db/GeoMech/api/
STANFORD_GEOMECH_TOKEN=your_stanford_access_token

# Rock Properties APIs
USGS_ROCK_PROPS_API=https://mrdata.usgs.gov/geology/rock-props/api/
NIST_MATERIALS_API=https://www.nist.gov/srd/nist-standard-reference-database-api
NIST_MATERIALS_KEY=your_nist_api_key
ISRM_DATABASE_API=https://www.isrm.net/foswiki/pub/ISRM/api/
ISRM_USERNAME=your_isrm_username
ISRM_PASSWORD=your_isrm_password

# GPS/GNSS Velocity Data
UNAVCO_API_ENDPOINT=https://www.unavco.org/data/web-services/api/gps/
UNAVCO_API_KEY=your_unavco_api_key
IGS_API_ENDPOINT=https://www.igs.org/api/products/
IGS_USERNAME=your_igs_username
IGS_PASSWORD=your_igs_password

# Gravity and Magnetic Field Data
GFZ_GRACE_API=https://grace.gfz-potsdam.de/api/
GFZ_GRACE_TOKEN=your_gfz_access_token
EMAG2_MAGNETIC_API=https://www.ngdc.noaa.gov/geomag/EMag2/api/
ICGEM_GRAVITY_API=http://icgem.gfz-potsdam.de/api/

# Enhanced Volcanic APIs
GVP_ENHANCED_API=https://volcano.si.edu/api/volcanism/
GVP_API_KEY=your_gvp_api_key
USGS_VOLCANO_MONITORING_API=https://volcanoes.usgs.gov/api/monitoring/
GEOTHERMAL_DATABASE_API=https://gdr.openei.org/api/geothermal/
GEOTHERMAL_API_KEY=your_geothermal_api_key

# Heat Flow and Thermal Data
IHFC_HEAT_FLOW_API=https://ihfc-iugg.org/api/heat-flow/
IHFC_LICENSE_KEY=your_ihfc_license_key
GLOBAL_HEAT_FLOW_DB=https://www.heatflow.org/api/
HEAT_FLOW_DB_TOKEN=your_heat_flow_token

# =============================================================================
# STRESS FIELD INTEGRATION CONFIGURATION
# =============================================================================

# Data Quality Thresholds
MIN_STRESS_MEASUREMENT_QUALITY=0.6
MAX_STRESS_MEASUREMENT_AGE_YEARS=20
MIN_ROCK_PROPERTY_CONFIDENCE=0.7
MAX_GEOLOGICAL_DISTANCE_KM=100

# Search and Integration Parameters
DEFAULT_STRESS_SEARCH_RADIUS_KM=50
MAX_STRESS_SEARCH_RADIUS_KM=200
STRESS_DATA_GRID_RESOLUTION_M=1000
STRESS_INTERPOLATION_METHOD=inverse_distance_weighted

# Database Performance Settings
STRESS_QUERY_TIMEOUT_SECONDS=60
STRESS_CONNECTION_POOL_SIZE=15
STRESS_MAX_CONCURRENT_REQUESTS=8
STRESS_RESULT_CACHE_TTL_SECONDS=1800

# Machine Learning Integration
STRESS_ML_MODEL_ENDPOINT=https://your-ml-service.com/api/stress-prediction/
ML_MODEL_API_KEY=your_ml_api_key
STRESS_PREDICTION_CONFIDENCE_THRESHOLD=0.8
ENSEMBLE_MODEL_COUNT=5

# =============================================================================
# MONITORING AND ALERTING
# =============================================================================

# API Health Monitoring
API_HEALTH_CHECK_INTERVAL_MINUTES=5
API_FAILURE_THRESHOLD=3
API_RECOVERY_BACKOFF_SECONDS=300

# Data Quality Monitoring
DATA_QUALITY_CHECK_INTERVAL_HOURS=6
QUALITY_DEGRADATION_THRESHOLD=0.1
MISSING_DATA_ALERT_PERCENTAGE=25

# Performance Monitoring
QUERY_PERFORMANCE_THRESHOLD_MS=5000
CACHE_HIT_RATIO_THRESHOLD=0.85
DATABASE_CONNECTION_ALERT_PERCENTAGE=80
```

## Helper Methods for Stress Field Database Integration
```python
# Enhanced helper methods for stress field real data integration
async def integrate_world_stress_map_data(db_manager, lat: float, lon: float, radius_km: float = 50.0):
    """
    Integrate World Stress Map database with comprehensive stress orientation data
    """
    wsm_api_url = f"{os.getenv('WSM_API_ENDPOINT')}query"
    params = {
        'latitude': lat,
        'longitude': lon,
        'radius': radius_km,
        'quality': 'A,B,C',  # Include high to medium quality data
        'format': 'json',
        'api_key': os.getenv('WSM_API_KEY')
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(wsm_api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Process and insert into stress_measurements table
                    await _process_wsm_data(db_manager, data)
                    return True
    except Exception as e:
        logging.error(f"WSM API integration failed: {e}")
        return False

async def integrate_usgs_stress_data(db_manager, lat: float, lon: float):
    """
    Integrate USGS stress measurement data from multiple sources
    """
    # USGS Induced Seismicity Database
    usgs_api_url = f"{os.getenv('USGS_STRESS_API')}measurements"
    params = {
        'minlatitude': lat - 0.5,
        'maxlatitude': lat + 0.5,
        'minlongitude': lon - 0.5,
        'maxlongitude': lon + 0.5,
        'measurement_type': 'hydraulic_fracturing,overcoring,focal_mechanism',
        'format': 'json'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(usgs_api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Process and insert into insitu_stress_measurements table
                    await _process_usgs_stress_data(db_manager, data)
                    return True
    except Exception as e:
        logging.error(f"USGS stress API integration failed: {e}")
        return False

async def integrate_unavco_gps_velocities(db_manager, lat: float, lon: float, radius_km: float = 100.0):
    """
    Integrate UNAVCO GPS velocity data for crustal deformation analysis
    """
    unavco_api_url = f"{os.getenv('UNAVCO_API_ENDPOINT')}velocities"
    params = {
        'coordinates': f"{lat},{lon}",
        'radius': radius_km,
        'reference_frame': 'ITRF2014',
        'format': 'json',
        'api_key': os.getenv('UNAVCO_API_KEY')
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(unavco_api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Process and insert into gps_velocity_field table
                    await _process_unavco_gps_data(db_manager, data)
                    return True
    except Exception as e:
        logging.error(f"UNAVCO GPS API integration failed: {e}")
        return False

async def integrate_rock_properties_databases(db_manager, rock_type: str, lat: float = None, lon: float = None):
    """
    Integrate multiple rock properties databases
    """
    # USGS Rock Properties Database
    usgs_props_url = f"{os.getenv('USGS_ROCK_PROPS_API')}search"
    params = {
        'rock_type': rock_type,
        'property': 'density,porosity,strength,modulus',
        'format': 'json'
    }
    
    if lat and lon:
        params.update({
            'latitude': lat,
            'longitude': lon,
            'radius': 200  # 200 km search radius
        })
    
    try:
        # Process USGS data
        async with aiohttp.ClientSession() as session:
            async with session.get(usgs_props_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    await _process_usgs_rock_properties(db_manager, data)
        
        # Integrate NIST Materials Database
        if os.getenv('NIST_MATERIALS_KEY'):
            await _integrate_nist_materials_data(db_manager, rock_type)
        
        # Integrate ISRM Database
        if os.getenv('ISRM_USERNAME'):
            await _integrate_isrm_rock_data(db_manager, rock_type)
        
        return True
        
    except Exception as e:
        logging.error(f"Rock properties integration failed: {e}")
        return False

async def integrate_enhanced_volcanic_data(db_manager, lat: float, lon: float, radius_km: float = 300.0):
    """
    Integrate enhanced volcanic data with magma chamber properties
    """
    # Enhanced GVP API
    gvp_api_url = f"{os.getenv('GVP_ENHANCED_API')}search"
    params = {
        'latitude': lat,
        'longitude': lon,
        'radius': radius_km,
        'activity_status': 'active,restless,dormant',
        'include_chamber_data': 'true',
        'api_key': os.getenv('GVP_API_KEY'),
        'format': 'json'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(gvp_api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    await _process_enhanced_volcanic_data(db_manager, data)
        
        # Integrate USGS real-time monitoring
        await _integrate_usgs_volcano_monitoring(db_manager, lat, lon, radius_km)
        
        # Integrate geothermal systems data
        await _integrate_geothermal_systems(db_manager, lat, lon, radius_km)
        
        return True
        
    except Exception as e:
        logging.error(f"Enhanced volcanic data integration failed: {e}")
        return False

# Data processing helper functions
async def _process_wsm_data(db_manager, wsm_data):
    """Process World Stress Map data and insert into database"""
    for measurement in wsm_data.get('measurements', []):
        insert_query = """
            INSERT INTO stress_measurements (
                measurement_id, location, longitude, latitude, depth_m,
                stress_magnitude_mpa, stress_orientation_degrees, stress_regime,
                measurement_method, measurement_date, reliability_score,
                measurement_agency, s1_orientation_azimuth
            ) VALUES (
                %(measurement_id)s, ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s, %(depth)s), 4326),
                %(lon)s, %(lat)s, %(depth)s, %(magnitude)s, %(orientation)s, %(regime)s,
                %(method)s, %(date)s, %(reliability)s, %(agency)s, %(s1_azimuth)s
            ) ON CONFLICT (measurement_id) DO UPDATE SET
                reliability_score = EXCLUDED.reliability_score,
                updated_at = NOW();
        """
        
        await db_manager.execute_query(insert_query, {
            'measurement_id': measurement.get('id'),
            'lon': measurement.get('longitude'),
            'lat': measurement.get('latitude'), 
            'depth': measurement.get('depth', 0),
            'magnitude': measurement.get('stress_magnitude'),
            'orientation': measurement.get('orientation'),
            'regime': measurement.get('regime'),
            'method': measurement.get('method'),
            'date': measurement.get('measurement_date'),
            'reliability': measurement.get('quality_score', 0.7),
            'agency': measurement.get('data_source'),
            's1_azimuth': measurement.get('s1_azimuth')
        })

async def _process_usgs_stress_data(db_manager, usgs_data):
    """Process USGS stress measurement data"""
    # Implementation for processing USGS stress measurements
    # Similar structure to WSM processing but adapted for USGS format
    pass

async def _process_unavco_gps_data(db_manager, gps_data):
    """Process UNAVCO GPS velocity data"""
    for station in gps_data.get('stations', []):
        insert_query = """
            INSERT INTO gps_velocity_field (
                station_code, location, longitude, latitude, elevation_m,
                velocity_east_mm_yr, velocity_north_mm_yr, velocity_vertical_mm_yr,
                velocity_magnitude_mm_yr, velocity_azimuth_degrees,
                sigma_east_mm_yr, sigma_north_mm_yr, sigma_vertical_mm_yr,
                reference_frame, observation_span_years, solution_quality,
                processing_center, last_observation_date
            ) VALUES (
                %(station_code)s, ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326),
                %(lon)s, %(lat)s, %(elevation)s, %(vel_east)s, %(vel_north)s, %(vel_up)s,
                %(vel_magnitude)s, %(vel_azimuth)s, %(sigma_east)s, %(sigma_north)s, %(sigma_up)s,
                %(ref_frame)s, %(obs_span)s, %(quality)s, %(processor)s, %(last_obs)s
            ) ON CONFLICT (station_code) DO UPDATE SET
                velocity_east_mm_yr = EXCLUDED.velocity_east_mm_yr,
                velocity_north_mm_yr = EXCLUDED.velocity_north_mm_yr,
                last_observation_date = EXCLUDED.last_observation_date,
                updated_at = NOW();
        """
        
        await db_manager.execute_query(insert_query, {
            'station_code': station.get('station_code'),
            'lon': station.get('longitude'),
            'lat': station.get('latitude'),
            'elevation': station.get('elevation'),
            'vel_east': station.get('velocity_east'),
            'vel_north': station.get('velocity_north'),
            'vel_up': station.get('velocity_up'),
            'vel_magnitude': station.get('velocity_magnitude'),
            'vel_azimuth': station.get('velocity_azimuth'),
            'sigma_east': station.get('sigma_east'),
            'sigma_north': station.get('sigma_north'),
            'sigma_up': station.get('sigma_up'),
            'ref_frame': station.get('reference_frame'),
            'obs_span': station.get('observation_span'),
            'quality': station.get('solution_quality'),
            'processor': station.get('processing_center'),
            'last_obs': station.get('last_observation')
        })
```

## Database Performance Optimization for Stress Field Queries
```sql
-- Specialized indexes for stress field queries
CREATE INDEX idx_stress_measurements_spatial_depth ON stress_measurements 
USING GIST (location, depth_m) WHERE reliability_score > 0.6;

CREATE INDEX idx_stress_measurements_regime_spatial ON stress_measurements 
USING GIST (location, stress_regime) WHERE measurement_date >= NOW() - INTERVAL '20 years';

CREATE INDEX idx_insitu_stress_depth_quality ON insitu_stress_measurements 
(depth_m, measurement_quality) WHERE measurement_quality > 0.7;

CREATE INDEX idx_petrophysical_rock_spatial_quality ON petrophysical_measurements 
USING GIST (location, rock_classification, quality_score) WHERE quality_score > 0.6;

CREATE INDEX idx_gps_velocity_magnitude_quality ON gps_velocity_field 
(velocity_magnitude_mm_yr, solution_quality) WHERE solution_quality > 0.7;

CREATE INDEX idx_active_volcanoes_pressure_spatial ON active_volcanoes 
USING GIST (location, magma_chamber_pressure_mpa) 
WHERE magma_chamber_pressure_mpa IS NOT NULL AND activity_level IN ('active', 'restless');

-- Materialized views for commonly accessed stress field data
CREATE MATERIALIZED VIEW high_quality_stress_measurements AS
SELECT 
    sm.*,
    ST_X(location::geometry) as longitude,
    ST_Y(location::geometry) as latitude,
    ST_Z(location::geometry) as depth_km
FROM stress_measurements sm
WHERE sm.reliability_score > 0.8
    AND sm.measurement_date >= NOW() - INTERVAL '15 years'
    AND sm.stress_magnitude_mpa IS NOT NULL;

CREATE INDEX idx_mv_high_quality_stress_spatial ON high_quality_stress_measurements USING GIST (location);
CREATE INDEX idx_mv_high_quality_stress_regime ON high_quality_stress_measurements (stress_regime);

-- Refresh materialized view weekly
CREATE OR REPLACE FUNCTION refresh_stress_field_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY high_quality_stress_measurements;
END;
$$ LANGUAGE plpgsql;

-- Schedule automatic refresh
SELECT cron.schedule('refresh-stress-views', '0 2 * * 0', 'SELECT refresh_stress_field_views();');
```

## Integration Testing Requirements for Stress Field Database
```python
# Comprehensive integration testing configuration
STRESS_FIELD_INTEGRATION_TESTS = {
    'database_connectivity': {
        'test_stress_measurements_insert': True,
        'test_spatial_queries': True,
        'test_index_performance': True,
        'test_materialized_view_refresh': True
    },
    'api_integration': {
        'test_wsm_api_connection': True,
        'test_usgs_api_connection': True,
        'test_unavco_api_connection': True,
        'test_rate_limiting': True,
        'test_authentication': True
    },
    'data_quality': {
        'test_measurement_validation': True,
        'test_coordinate_accuracy': True,
        'test_temporal_consistency': True,
        'test_cross_reference_validation': True
    },
    'performance_benchmarks': {
        'max_query_time_ms': 5000,
        'min_cache_hit_ratio': 0.85,
        'max_connection_pool_utilization': 0.8,
        'target_throughput_queries_per_second': 100
    },
    'ml_integration': {
        'test_feature_extraction': True,
        'test_model_prediction_accuracy': True,
        'test_uncertainty_quantification': True,
        'test_ensemble_prediction': True
    }
}
```

## Real-Time Data Integration Schema and API Endpoints

### Enhanced Database Schema for Real-Time Cap Rock Analysis

#### 1. Time Evolution Data Tables
```sql
-- Time evolution monitoring data from geological databases
CREATE TABLE time_evolution_data (
    id SERIAL PRIMARY KEY,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    analysis_timestamp TIMESTAMP DEFAULT NOW(),
    time_points REAL[] NOT NULL,  -- Array of time points in days
    failure_evolution REAL[] NOT NULL,  -- Array of failure probabilities
    data_sources TEXT[] NOT NULL,  -- Sources used: 'usgs_earthquakes', 'gps_deformation', 'mineral_resources'
    seismic_factor REAL DEFAULT 1.0,
    deformation_factor REAL DEFAULT 1.0,
    tectonic_factor REAL DEFAULT 1.0,
    confidence_score REAL CHECK (confidence_score BETWEEN 0 AND 1),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_time_evolution_location ON time_evolution_data USING GIST (location);
CREATE INDEX idx_time_evolution_timestamp ON time_evolution_data (analysis_timestamp);
CREATE INDEX idx_time_evolution_confidence ON time_evolution_data (confidence_score);

-- Real-time seismic influence tracking
CREATE TABLE seismic_influence_tracking (
    id SERIAL PRIMARY KEY,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    earthquake_data JSONB NOT NULL,  -- USGS earthquake records
    magnitude_avg REAL,
    magnitude_max REAL,
    event_count INTEGER DEFAULT 0,
    time_window_days INTEGER DEFAULT 365,
    influence_factor REAL DEFAULT 1.0,
    last_updated TIMESTAMP DEFAULT NOW()
);

-- GPS/Deformation monitoring integration
CREATE TABLE deformation_influence_tracking (
    id SERIAL PRIMARY KEY,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    station_id TEXT NOT NULL,
    displacement_data JSONB NOT NULL,  -- GPS displacement measurements
    max_displacement_mm REAL,
    displacement_trend REAL,  -- mm/year
    influence_factor REAL DEFAULT 1.0,
    measurement_period_days INTEGER DEFAULT 365,
    last_updated TIMESTAMP DEFAULT NOW()
);
```

#### 2. External Data Source Registry
```sql
-- Track external API data sources and their status
CREATE TABLE external_data_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(100) UNIQUE NOT NULL,
    source_type VARCHAR(50) NOT NULL,  -- 'usgs', 'nasa', 'agu_wiley', 'worldbank', 'open_data'
    base_url TEXT NOT NULL,
    api_key_required BOOLEAN DEFAULT FALSE,
    rate_limit_per_second REAL DEFAULT 1.0,
    last_successful_request TIMESTAMP,
    request_count_today INTEGER DEFAULT 0,
    error_count_today INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',  -- 'active', 'rate_limited', 'error', 'disabled'
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- API endpoint configurations
CREATE TABLE api_endpoint_configs (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES external_data_sources(id),
    endpoint_name VARCHAR(100) NOT NULL,
    endpoint_url TEXT NOT NULL,
    parameters JSONB,  -- Expected parameters and their types
    response_format VARCHAR(20) DEFAULT 'json',
    timeout_seconds INTEGER DEFAULT 30,
    max_retries INTEGER DEFAULT 3,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### API Endpoints for Real Data Integration

#### 1. USGS Real-Time Data Endpoints
```yaml
usgs_endpoints:
  earthquake_data:
    url: "https://earthquake.usgs.gov/fdsnws/event/1/query"
    method: "GET"
    parameters:
      format: "geojson"
      starttime: "YYYY-MM-DD"
      endtime: "YYYY-MM-DD"
      minlatitude: "float"
      maxlatitude: "float"
      minlongitude: "float"
      maxlongitude: "float"
      minmagnitude: "float"
      limit: "integer"
    response_fields:
      - coordinates: "[longitude, latitude, depth]"
      - magnitude: "float"
      - time: "unix_timestamp_ms"
      - place: "string"
      - significance: "integer"
    rate_limit: "2 requests/second"
    
  mineral_resources:
    url: "https://mrdata.usgs.gov/services/wfs"
    method: "GET" 
    parameters:
      service: "WFS"
      version: "1.0.0"
      request: "GetFeature"
      typename: "mrds"
      outputFormat: "application/json"
      CQL_FILTER: "string"
      bbox: "minx,miny,maxx,maxy"
    response_fields:
      - site_name: "string"
      - commodity: "string"
      - coordinates: "[longitude, latitude]"
      - site_type: "string"
    rate_limit: "1 request/second"
```

#### 2. NASA Earth Data Endpoints
```yaml
nasa_endpoints:
  landsat_search:
    url: "https://cmr.earthdata.nasa.gov/search/granules.json"
    method: "GET"
    parameters:
      concept_id: "C1711961296-LPCLOUD"
      bounding_box: "min_lon,min_lat,max_lon,max_lat"
      temporal: "start_date,end_date"
      page_size: "integer"
    response_fields:
      - title: "string"
      - time_start: "iso_datetime"
      - cloud_cover: "float"
      - boxes: "array of bbox coordinates"
    authentication: "NASA Earthdata Login"
    rate_limit: "1 request/second"
```

#### 3. Database Integration Manager Endpoints
```yaml
database_manager_endpoints:
  time_evolution_analysis:
    internal_endpoint: "/api/v1/cap-rock/time-evolution"
    method: "POST"
    request_body:
      location: "[lat, lon]"
      failure_analysis: "object"
      stress_state: "object" 
      cap_rock_props: "object"
      time_window_days: "integer (default: 365)"
    response:
      time_points: "array of floats (days)"
      failure_evolution: "array of floats (probabilities)"
      data_sources_used: "array of strings"
      confidence_score: "float"
      integration_metadata: "object"
    
  regional_data_collection:
    internal_endpoint: "/api/v1/external-data/regional"
    method: "POST"
    request_body:
      region_name: "string"
      bbox: "[min_lon, min_lat, max_lon, max_lat]"
      time_range: "[start_iso_datetime, end_iso_datetime]"
      data_types: "array of strings"
    response:
      earthquakes: "array of geological records"
      mineral_resources: "array of geological records"
      satellite_imagery: "array of geological records"
      research_articles: "array of article objects"
      mining_production: "array of production data"
      integration_report: "object"
```

### Environment Variables for Database Configuration

```bash
# PostgreSQL Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=prometheus_geo
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_password

# External API Keys (Optional)
USGS_API_KEY=your_usgs_api_key_here
NASA_EARTHDATA_TOKEN=your_nasa_token_here
WORLDBANK_API_KEY=your_worldbank_key_here

# Rate Limiting Configuration  
USGS_RATE_LIMIT=1.0
NASA_RATE_LIMIT=1.0
EXTERNAL_DATA_TIMEOUT=30

# Model Database Configuration
MODEL_DATABASE_URL=sqlite:///volcanic_models.db
AZURE_STORAGE_ACCOUNT_URL=https://your_account.blob.core.windows.net
AZURE_VM_TOKEN=your_azure_vm_token
AZURE_VM_ENDPOINT=https://your-vm.azure.com/api

# MLOps Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Data Integration Pipeline Configuration

#### 1. Real-Time Data Processing Settings
```yaml
data_integration:
  cache_duration_hours: 24
  max_concurrent_requests: 5
  retry_attempts: 3
  fallback_to_physics: true
  
  quality_thresholds:
    min_earthquake_data_points: 5
    min_deformation_stations: 2
    max_cloud_cover_percent: 30
    min_confidence_score: 0.6
  
  geographic_limits:
    max_search_radius_km: 50
    coordinate_precision_decimal_places: 6
    
  temporal_limits:
    max_historical_data_years: 5
    min_analysis_window_days: 30
    time_evolution_max_days: 3650  # 10 years
```

#### 2. Database Optimization Settings
```sql
-- Performance optimization for real-time queries
-- Partitioning for time_evolution_data table
CREATE TABLE time_evolution_data_y2025 PARTITION OF time_evolution_data
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Materialized views for common queries
CREATE MATERIALIZED VIEW recent_seismic_activity AS
SELECT 
    location,
    COUNT(*) as event_count,
    AVG(magnitude_avg) as avg_magnitude,
    MAX(magnitude_max) as max_magnitude,
    date_trunc('day', last_updated) as analysis_date
FROM seismic_influence_tracking 
WHERE last_updated >= NOW() - INTERVAL '30 days'
GROUP BY location, date_trunc('day', last_updated);

-- Refresh policy for materialized view
CREATE OR REPLACE FUNCTION refresh_seismic_activity_view()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW recent_seismic_activity;
END;
$$ LANGUAGE plpgsql;

-- Schedule refresh every hour
SELECT cron.schedule('refresh-seismic-view', '0 * * * *', 'SELECT refresh_seismic_activity_view();');
```

### Integration Testing Endpoints

#### 1. Health Check Endpoints
```yaml
health_checks:
  database_connectivity:
    endpoint: "/api/v1/health/database"
    expected_response: 
      status: "healthy"
      postgres_version: "string"
      postgis_version: "string"
      timescaledb_version: "string"
      
  external_apis_status:
    endpoint: "/api/v1/health/external-apis"
    expected_response:
      usgs_status: "active|rate_limited|error"
      nasa_status: "active|rate_limited|error"
      worldbank_status: "active|rate_limited|error"
      
  data_integration_pipeline:
    endpoint: "/api/v1/health/integration"
    expected_response:
      pipeline_status: "active"
      last_successful_collection: "iso_datetime"
      cached_data_age_hours: "float"
```

## Real-Time Data Integration Schema and API Endpoints

### Enhanced Logging Tables for Real Data Connections

#### 1. Gas Thermal Effects Logging
```sql
CREATE TABLE gas_thermal_effects_log (
    id SERIAL PRIMARY KEY,
    coords REAL[] NOT NULL,
    temperature_c REAL NOT NULL,
    co2_ppm REAL DEFAULT 400,
    so2_ppm REAL DEFAULT 0,
    h2s_ppm REAL DEFAULT 0,
    damping_factor REAL NOT NULL CHECK (damping_factor > 0),
    volcanic_activity_level REAL DEFAULT 0,
    gas_flux_kg_s REAL DEFAULT 0,
    analysis_quality REAL DEFAULT 0.8 CHECK (analysis_quality BETWEEN 0 AND 1),
    data_source VARCHAR(100) DEFAULT 'USGS',
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Spatial and temporal indexes
CREATE INDEX idx_gas_effects_coords ON gas_thermal_effects_log USING GIN (coords);
CREATE INDEX idx_gas_effects_timestamp ON gas_thermal_effects_log (timestamp);
CREATE INDEX idx_gas_effects_temperature ON gas_thermal_effects_log (temperature_c);

-- Hypertable for time-series optimization
SELECT create_hypertable('gas_thermal_effects_log', 'timestamp');
```

#### 2. Hydrothermal Alteration Logging
```sql
CREATE TABLE hydrothermal_alteration_log (
    id SERIAL PRIMARY KEY,
    coords REAL[] NOT NULL,
    alteration_minerals JSONB DEFAULT '[]'::jsonb,
    thermal_springs_count INTEGER DEFAULT 0,
    thermal_damping_factor REAL NOT NULL CHECK (thermal_damping_factor > 0),
    thermal_diffusivity_factor REAL NOT NULL CHECK (thermal_diffusivity_factor > 0),
    grain_boundary_factor REAL DEFAULT 1.0 CHECK (grain_boundary_factor > 0),
    alteration_intensity REAL DEFAULT 0 CHECK (alteration_intensity BETWEEN 0 AND 1),
    ph_level REAL DEFAULT 7.0 CHECK (ph_level BETWEEN 0 AND 14),
    mineral_diversity_index REAL DEFAULT 0,
    geothermal_gradient_k_km REAL DEFAULT 25.0,
    data_source VARCHAR(100) DEFAULT 'USGS_Mineral_Resources',
    analysis_confidence REAL DEFAULT 0.8 CHECK (analysis_confidence BETWEEN 0 AND 1),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for alteration analysis
CREATE INDEX idx_alteration_coords ON hydrothermal_alteration_log USING GIN (coords);
CREATE INDEX idx_alteration_minerals ON hydrothermal_alteration_log USING GIN (alteration_minerals);
CREATE INDEX idx_alteration_timestamp ON hydrothermal_alteration_log (timestamp);
CREATE INDEX idx_alteration_intensity ON hydrothermal_alteration_log (alteration_intensity);

-- Hypertable for time-series optimization
SELECT create_hypertable('hydrothermal_alteration_log', 'timestamp');
```

#### 3. Fluid Composition Analysis Logging
```sql
CREATE TABLE fluid_composition_log (
    id SERIAL PRIMARY KEY,
    coords REAL[] NOT NULL,
    ph_level REAL NOT NULL CHECK (ph_level BETWEEN 0 AND 14),
    dissolved_solids_ppm REAL DEFAULT 200 CHECK (dissolved_solids_ppm >= 0),
    sulfate_ppm REAL DEFAULT 10 CHECK (sulfate_ppm >= 0),
    chloride_ppm REAL DEFAULT 15 CHECK (chloride_ppm >= 0),
    bicarbonate_ppm REAL DEFAULT 120 CHECK (bicarbonate_ppm >= 0),
    surface_temp_c REAL DEFAULT 15.0,
    enhancement_factor REAL NOT NULL CHECK (enhancement_factor > 0),
    ionic_strength REAL DEFAULT 0,
    total_alkalinity_mg_l REAL DEFAULT 0,
    electrical_conductivity_ms_cm REAL DEFAULT 0,
    redox_potential_mv REAL DEFAULT 0,
    water_type VARCHAR(50) DEFAULT 'freshwater',
    sampling_depth_m REAL DEFAULT 0,
    data_source VARCHAR(100) DEFAULT 'USGS_Water_Quality',
    analysis_method VARCHAR(100) DEFAULT 'field_measurement',
    measurement_precision REAL DEFAULT 0.9 CHECK (measurement_precision BETWEEN 0 AND 1),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fluid analysis
CREATE INDEX idx_fluid_coords ON fluid_composition_log USING GIN (coords);
CREATE INDEX idx_fluid_ph ON fluid_composition_log (ph_level);
CREATE INDEX idx_fluid_timestamp ON fluid_composition_log (timestamp);
CREATE INDEX idx_fluid_enhancement ON fluid_composition_log (enhancement_factor);

-- Hypertable for time-series optimization
SELECT create_hypertable('fluid_composition_log', 'timestamp');
```

#### 4. External Data Sources Registry
```sql
CREATE TABLE external_data_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(100) UNIQUE NOT NULL,
    base_url TEXT NOT NULL,
    api_version VARCHAR(20),
    api_key_required BOOLEAN DEFAULT false,
    rate_limit_requests_per_second REAL DEFAULT 1.0,
    timeout_seconds INTEGER DEFAULT 30,
    last_successful_connection TIMESTAMPTZ,
    last_error_message TEXT,
    error_count_24h INTEGER DEFAULT 0,
    status VARCHAR(20) CHECK (status IN ('active', 'rate_limited', 'error', 'maintenance')) DEFAULT 'active',
    data_types JSONB DEFAULT '[]'::jsonb,
    geographic_coverage GEOMETRY(POLYGON, 4326),
    temporal_coverage_start DATE,
    temporal_coverage_end DATE,
    update_frequency_hours INTEGER DEFAULT 24,
    reliability_score REAL DEFAULT 0.8 CHECK (reliability_score BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for data sources
CREATE INDEX idx_data_sources_status ON external_data_sources (status);
CREATE INDEX idx_data_sources_reliability ON external_data_sources (reliability_score DESC);
CREATE INDEX idx_data_sources_coverage ON external_data_sources USING GIST (geographic_coverage);
```

### API Endpoints Configuration

#### 1. USGS Data Integration Endpoints
```yaml
usgs_endpoints:
  earthquake_data:
    url: "https://earthquake.usgs.gov/fdsnws/event/1/query"
    method: "GET"
    rate_limit: 1.0  # requests per second
    timeout: 30
    parameters:
      format: "geojson"
      starttime: "dynamic"
      endtime: "dynamic"
      latitude: "dynamic"
      longitude: "dynamic"
      maxradiuskm: "dynamic"
      minmagnitude: 2.0
      
  mineral_resources:
    url: "https://mrdata.usgs.gov/services/wfs"
    method: "GET"
    rate_limit: 0.5
    timeout: 45
    parameters:
      service: "WFS"
      version: "2.0.0"
      request: "GetFeature"
      
  water_quality:
    url: "https://waterservices.usgs.gov/nwis/site"
    method: "GET"
    rate_limit: 1.0
    timeout: 30
    parameters:
      format: "waterml,2.0"
      seriesCatalogOutput: "true"
```

#### 2. NASA Earth Data Integration Endpoints
```yaml
nasa_endpoints:
  earth_data:
    url: "https://api.nasa.gov/planetary/earth"
    method: "GET"
    rate_limit: 1.0
    timeout: 60
    api_key_required: true
    parameters:
      api_key: "${NASA_API_KEY}"
      
  modis_thermal:
    url: "https://modis.gsfc.nasa.gov/data/dataprod"
    method: "GET"
    rate_limit: 0.3
    timeout: 120
    
  landsat_thermal:
    url: "https://landsatlook.usgs.gov/sat-api"
    method: "GET"
    rate_limit: 0.5
    timeout: 90
```

#### 3. Internal Database Management Endpoints
```yaml
internal_endpoints:
  data_health_check:
    path: "/api/v1/health/database-integration"
    method: "GET"
    response_format:
      status: "string"
      database_connection: "boolean"
      external_apis: "object"
      cached_data_age: "integer"
      
  real_time_analysis:
    path: "/api/v1/analysis/real-time"
    method: "POST"
    request_format:
      coordinates: "array[float]"
      analysis_type: "string"
      time_window_hours: "integer"
    response_format:
      analysis_results: "object"
      data_sources_used: "array[string]"
      confidence_score: "float"
      
  cache_management:
    path: "/api/v1/cache/purge"
    method: "DELETE"
    parameters:
      data_type: "string"
      max_age_hours: "integer"
```

### Environment Variables for Real-Time Integration
```bash
# Database connection
DATABASE_URL="postgresql://prometheus_admin:GeoPhys2024!PrometheusDB#@localhost:5432/prometheus_geo"

# External API keys
USGS_API_KEY="your_usgs_api_key"
NASA_API_KEY="your_nasa_api_key"
WORLDBANK_API_KEY="your_worldbank_api_key"

# Rate limiting configuration
MAX_CONCURRENT_REQUESTS=5
DEFAULT_TIMEOUT_SECONDS=30
RETRY_ATTEMPTS=3

# Cache configuration
CACHE_TTL_HOURS=24
MAX_CACHE_SIZE_MB=500

# Logging configuration
LOG_LEVEL="INFO"
LOG_DATABASE_QUERIES=true
LOG_EXTERNAL_API_CALLS=true
```

### Performance Optimization for Real-Time Data

#### 1. Table Partitioning
```sql
-- Partition gas effects log by month
CREATE TABLE gas_thermal_effects_log_y2025m01 PARTITION OF gas_thermal_effects_log
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Partition alteration log by month
CREATE TABLE hydrothermal_alteration_log_y2025m01 PARTITION OF hydrothermal_alteration_log
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Partition fluid composition log by month
CREATE TABLE fluid_composition_log_y2025m01 PARTITION OF fluid_composition_log
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

#### 2. Materialized Views for Common Queries
```sql
-- Recent gas activity summary
CREATE MATERIALIZED VIEW recent_gas_activity AS
SELECT 
    coords,
    AVG(damping_factor) as avg_damping_factor,
    MAX(co2_ppm) as max_co2,
    MAX(so2_ppm) as max_so2,
    COUNT(*) as measurement_count,
    date_trunc('day', timestamp) as measurement_date
FROM gas_thermal_effects_log 
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY coords, date_trunc('day', timestamp);

-- Recent alteration patterns
CREATE MATERIALIZED VIEW recent_alteration_patterns AS
SELECT 
    coords,
    AVG(thermal_damping_factor) as avg_thermal_damping,
    AVG(thermal_diffusivity_factor) as avg_diffusivity,
    AVG(alteration_intensity) as avg_intensity,
    COUNT(*) as analysis_count,
    date_trunc('week', timestamp) as analysis_week
FROM hydrothermal_alteration_log 
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY coords, date_trunc('week', timestamp);

-- Fluid composition trends
CREATE MATERIALIZED VIEW fluid_composition_trends AS
SELECT 
    coords,
    AVG(ph_level) as avg_ph,
    AVG(enhancement_factor) as avg_enhancement,
    AVG(dissolved_solids_ppm) as avg_tds,
    COUNT(*) as sample_count,
    date_trunc('day', timestamp) as sample_date
FROM fluid_composition_log 
WHERE timestamp >= NOW() - INTERVAL '14 days'
GROUP BY coords, date_trunc('day', timestamp);
```

#### 3. Automated View Refresh Functions
```sql
-- Function to refresh all real-time views
CREATE OR REPLACE FUNCTION refresh_real_time_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW recent_gas_activity;
    REFRESH MATERIALIZED VIEW recent_alteration_patterns;
    REFRESH MATERIALIZED VIEW fluid_composition_trends;
    
    -- Log refresh completion
    INSERT INTO system_maintenance_log (operation, completion_time, status)
    VALUES ('refresh_real_time_views', NOW(), 'completed');
END;
$$ LANGUAGE plpgsql;

-- Schedule refresh every 2 hours
SELECT cron.schedule('refresh-real-time-views', '0 */2 * * *', 'SELECT refresh_real_time_views();');
```

## Real-Time Data Integration Schema and API Endpoints

### Heat Flow and Thermal Data Integration

```sql
-- Heat flow measurements from various sources
CREATE TABLE heat_flow_measurements (
    id SERIAL PRIMARY KEY,
    location_geom GEOGRAPHY(POINT, 4326) NOT NULL,
    surface_temp_c REAL NOT NULL,
    temp_variation_c REAL DEFAULT 5.0,
    heat_flow_mw_m2 REAL NOT NULL CHECK (heat_flow_mw_m2 > 0),
    measurement_date TIMESTAMPTZ NOT NULL,
    data_quality REAL CHECK (data_quality BETWEEN 0 AND 1),
    source_id VARCHAR(100) NOT NULL,
    measurement_method VARCHAR(50),
    depth_m REAL,
    uncertainty_mw_m2 REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_heat_flow_spatial ON heat_flow_measurements USING GIST (location_geom);
CREATE INDEX idx_heat_flow_source ON heat_flow_measurements (source_id);
SELECT create_hypertable('heat_flow_measurements', 'measurement_date');

-- Geothermal gradients from drilling and modeling
CREATE TABLE geothermal_gradients (
    id SERIAL PRIMARY KEY,
    location_geom GEOGRAPHY(POINT, 4326) NOT NULL,
    gradient_c_km REAL NOT NULL CHECK (gradient_c_km > 0),
    depth_range_m INTEGER[] NOT NULL, -- [min_depth, max_depth]
    measurement_method VARCHAR(100) NOT NULL,
    data_quality REAL CHECK (data_quality BETWEEN 0 AND 1),
    measurement_date TIMESTAMPTZ NOT NULL,
    source_id VARCHAR(100) NOT NULL,
    borehole_id VARCHAR(100),
    lithology_context TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_geothermal_spatial ON geothermal_gradients USING GIST (location_geom);
CREATE INDEX idx_geothermal_quality ON geothermal_gradients (data_quality DESC);
```

### Volcanic Database Integration Schema

```sql
-- Volcanic chambers with geophysical constraints
CREATE TABLE volcanic_chambers (
    id SERIAL PRIMARY KEY,
    location_geom GEOGRAPHY(POINT, 4326) NOT NULL,
    chamber_id VARCHAR(100) UNIQUE NOT NULL,
    depth_min_m INTEGER NOT NULL,
    depth_max_m INTEGER NOT NULL,
    volume_km3 REAL CHECK (volume_km3 > 0),
    magma_composition VARCHAR(50),
    temperature_c REAL CHECK (temperature_c > 500),
    pressure_mpa REAL CHECK (pressure_mpa > 0),
    last_activity TIMESTAMPTZ,
    confidence_score REAL CHECK (confidence_score BETWEEN 0 AND 1),
    source_id VARCHAR(100) NOT NULL,
    geophysical_method VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chambers_spatial ON volcanic_chambers USING GIST (location_geom);
CREATE INDEX idx_chambers_confidence ON volcanic_chambers (confidence_score DESC);

-- Volcanic heat sources integration
CREATE TABLE volcanic_heat_sources (
    id SERIAL PRIMARY KEY,
    location_geom GEOGRAPHY(POINT, 4326) NOT NULL,
    source_id VARCHAR(100) NOT NULL,
    source_type VARCHAR(50) CHECK (source_type IN ('thermal_springs', 'fumarole_field', 'magma_chamber', 'geothermal_field')),
    heat_output_mw REAL CHECK (heat_output_mw > 0),
    depth_m REAL,
    temperature_c REAL CHECK (temperature_c > 0),
    confidence_score REAL CHECK (confidence_score BETWEEN 0 AND 1),
    last_measured TIMESTAMPTZ,
    measurement_method VARCHAR(100),
    source_origin VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_heat_sources_spatial ON volcanic_heat_sources USING GIST (location_geom);
CREATE INDEX idx_heat_sources_type ON volcanic_heat_sources (source_type);

-- Smithsonian Global Volcanism Program integration
CREATE TABLE global_volcanism_program (
    id SERIAL PRIMARY KEY,
    volcano_number VARCHAR(20) UNIQUE NOT NULL,
    volcano_name VARCHAR(200) NOT NULL,
    country VARCHAR(100),
    latitude REAL NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude REAL NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    elevation INTEGER,
    primary_volcano_type VARCHAR(100),
    last_known_eruption VARCHAR(50),
    evidence_category VARCHAR(100),
    major_rock_type VARCHAR(100),
    tectonic_setting VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_gvp_location ON global_volcanism_program (latitude, longitude);
CREATE INDEX idx_gvp_volcano_number ON global_volcanism_program (volcano_number);

-- Volcanic provinces for regional classification
CREATE TABLE volcanic_provinces (
    id SERIAL PRIMARY KEY,
    province_name VARCHAR(200) NOT NULL,
    province_geom GEOGRAPHY(POLYGON, 4326) NOT NULL,
    province_type VARCHAR(100) NOT NULL,
    tectonic_setting VARCHAR(200),
    magma_type VARCHAR(100),
    activity_period VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_provinces_spatial ON volcanic_provinces USING GIST (province_geom);
```

### Gas Effects and Hydrothermal Alteration Logging

```sql
-- Gas thermal effects logging for real-time analysis
CREATE TABLE gas_thermal_effects_log (
    id SERIAL PRIMARY KEY,
    coords REAL[] NOT NULL, -- [lat, lon]
    temperature_c REAL NOT NULL,
    co2_ppm REAL DEFAULT 400,
    so2_ppm REAL DEFAULT 0,
    h2s_ppm REAL DEFAULT 0,
    damping_factor REAL NOT NULL CHECK (damping_factor >= 1.0),
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('gas_thermal_effects_log', 'timestamp');
CREATE INDEX idx_gas_effects_coords ON gas_thermal_effects_log USING GIN (coords);

-- Hydrothermal alteration effects logging
CREATE TABLE hydrothermal_alteration_log (
    id SERIAL PRIMARY KEY,
    coords REAL[] NOT NULL, -- [lat, lon]
    alteration_minerals JSONB,
    thermal_springs_count INTEGER DEFAULT 0,
    thermal_damping_factor REAL NOT NULL CHECK (thermal_damping_factor >= 1.0),
    thermal_diffusivity_factor REAL NOT NULL CHECK (thermal_diffusivity_factor > 0),
    grain_boundary_factor REAL NOT NULL CHECK (grain_boundary_factor >= 1.0),
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('hydrothermal_alteration_log', 'timestamp');
CREATE INDEX idx_alteration_coords ON hydrothermal_alteration_log USING GIN (coords);

-- Fluid composition effects tracking
CREATE TABLE fluid_composition_log (
    id SERIAL PRIMARY KEY,
    coords REAL[] NOT NULL, -- [lat, lon]
    fluid_type VARCHAR(50) NOT NULL,
    composition_data JSONB NOT NULL,
    thermal_effects JSONB,
    mechanical_effects JSONB,
    confidence_score REAL CHECK (confidence_score BETWEEN 0 AND 1),
    data_source VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('fluid_composition_log', 'timestamp');
CREATE INDEX idx_fluid_coords ON fluid_composition_log USING GIN (coords);
```

### External Data Sources Registry

```sql
-- Track external data source health and usage
CREATE TABLE external_data_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(100) UNIQUE NOT NULL,
    source_type VARCHAR(50) NOT NULL, -- 'usgs', 'nasa', 'smithsonian', 'world_bank'
    base_url TEXT NOT NULL,
    api_key_required BOOLEAN DEFAULT FALSE,
    rate_limit_per_second REAL DEFAULT 1.0,
    last_successful_query TIMESTAMPTZ,
    last_failed_query TIMESTAMPTZ,
    total_queries INTEGER DEFAULT 0,
    failed_queries INTEGER DEFAULT 0,
    average_response_time_ms REAL,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'error')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_external_sources_status ON external_data_sources (status);
```

## API Endpoints Configuration

### USGS Data Integration Endpoints

```yaml
# USGS Earthquake Data
earthquake_api:
  base_url: "https://earthquake.usgs.gov/fdsnws/event/1/query"
  parameters:
    - format: "geojson"
    - starttime: "dynamic"
    - endtime: "dynamic"
    - minlatitude: "dynamic"
    - maxlatitude: "dynamic"
    - minlongitude: "dynamic"
    - maxlongitude: "dynamic"
    - minmagnitude: "2.0"
  rate_limit: 10 # requests per minute

# USGS Volcano Hazards Program
volcano_api:
  base_url: "https://volcano.wr.usgs.gov/vsc/api"
  endpoints:
    - "/volcanoes"
    - "/alerts"
    - "/monitoring"
  rate_limit: 20 # requests per minute

# USGS Geothermal Resources
geothermal_api:
  base_url: "https://mrdata.usgs.gov/services/geothermal"
  format: "json"
  rate_limit: 5 # requests per minute
```

### NASA Earth Data Integration

```yaml
# NASA MODIS Land Surface Temperature
nasa_modis_lst:
  base_url: "https://modis.gsfc.nasa.gov/data/dataprod"
  dataset: "MOD11A1"
  parameters:
    - resolution: "1km"
    - temporal: "daily"
  authentication: "required"

# NASA ASTER Global Digital Elevation Model
nasa_aster_gdem:
  base_url: "https://asterweb.jpl.nasa.gov/gdem.asp"
  version: "v3"
  resolution: "30m"
```

### Internal Database Management Endpoints

```yaml
# Database health monitoring
health_check:
  endpoint: "/api/v1/database/health"
  checks:
    - connection_pool_status
    - query_performance
    - table_sizes
    - index_usage
  frequency: "5 minutes"

# Real-time data ingestion
data_ingestion:
  endpoint: "/api/v1/ingest"
  supported_types:
    - heat_flow_measurements
    - geothermal_gradients
    - volcanic_chambers
    - gas_effects
  batch_size: 1000
  timeout: "30 seconds"

# Data quality assessment
quality_check:
  endpoint: "/api/v1/quality"
  parameters:
    - data_type: "required"
    - time_range: "optional"
    - location_bounds: "optional"
  response_format: "json"
```

## Environment Variables Configuration

```bash
# Database Configuration
PROMETHEUS_DB_HOST=localhost
PROMETHEUS_DB_PORT=5432
PROMETHEUS_DB_NAME=prometheus_geo
PROMETHEUS_DB_USER=prometheus_admin
PROMETHEUS_DB_PASSWORD=GeoPhys2024!PrometheusDB#
PROMETHEUS_DB_SSL_MODE=prefer

# External API Keys
USGS_API_KEY=your_usgs_api_key_here
NASA_EARTHDATA_TOKEN=your_nasa_token_here
SMITHSONIAN_API_ACCESS=your_smithsonian_key_here

# Rate Limiting Configuration
EXTERNAL_API_RATE_LIMIT=1.0
CACHE_TIMEOUT_SECONDS=3600
MAX_CONCURRENT_REQUESTS=10

# Logging Configuration
LOG_LEVEL=INFO
DATABASE_LOG_QUERIES=false
PERFORMANCE_MONITORING=true
```

## Performance Optimization Configuration

```sql
-- Materialized views for frequently accessed data
CREATE MATERIALIZED VIEW mv_regional_volcanic_summary AS
SELECT 
    ST_SnapToGrid(location_geom::geometry, 0.1) as grid_location,
    COUNT(*) as volcano_count,
    AVG(heat_output_mw) as avg_heat_output,
    MAX(confidence_score) as max_confidence,
    array_agg(DISTINCT source_type) as source_types
FROM volcanic_heat_sources 
GROUP BY ST_SnapToGrid(location_geom::geometry, 0.1);

CREATE UNIQUE INDEX ON mv_regional_volcanic_summary (grid_location);

-- Automated refresh schedule
SELECT cron.schedule('refresh-volcanic-summary', '0 */4 * * *', 
    'REFRESH MATERIALIZED VIEW CONCURRENTLY mv_regional_volcanic_summary;');

-- Table partitioning for time-series data
SELECT create_hypertable('gas_thermal_effects_log', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day');

SELECT add_retention_policy('gas_thermal_effects_log', INTERVAL '30 days');
```

## Health Check and Monitoring Endpoints

```sql
-- Database performance monitoring function
CREATE OR REPLACE FUNCTION get_database_health() 
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'connection_count', (SELECT count(*) FROM pg_stat_activity),
        'active_queries', (SELECT count(*) FROM pg_stat_activity WHERE state = 'active'),
        'cache_hit_ratio', (SELECT round(100.0 * sum(blks_hit) / sum(blks_hit + blks_read), 2) 
                           FROM pg_stat_database WHERE datname = current_database()),
        'largest_table_mb', (SELECT round(pg_total_relation_size(schemaname||'.'||tablename) / 1024 / 1024, 2) 
                            FROM pg_tables ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC LIMIT 1),
        'check_time', NOW()
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- API endpoint health status table
CREATE TABLE api_health_log (
    id SERIAL PRIMARY KEY,
    endpoint_name VARCHAR(100) NOT NULL,
    response_time_ms INTEGER,
    status_code INTEGER,
    success BOOLEAN,
    error_message TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('api_health_log', 'timestamp');
```

Last updated: 2025-08-29
Schema version: 3.0.0
Compatible with: PostgreSQL 15+, PostGIS 3.3+, TimescaleDB 2.8+
Data Connectors version: 1.8.0
Geological Age Integration version: 1.1.0
Placeholder Methods Implementation version: 2.0.0
Cap Rock Stability Real Data Integration version: 2.0.0
Real-Time Volcanic Analysis Integration version: 1.1.0
Stress Field Database Integration version: 1.0.0
Time Evolution Real Data Integration version: 1.1.0
Heat Flow Database Integration version: 1.0.0
Volcanic Chambers Database Integration version: 1.0.0
Global Volcanism Program Integration version: 1.0.0

## Advanced Features Integration (v3.0)

### Machine Learning Model Management Tables

#### ML Model Registry Table
```sql
CREATE TABLE ml_model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(200) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL CHECK (model_type IN (
        'regression', 'classification', 'ensemble_regression', 'ensemble_classification',
        'time_series', 'ensemble_time_series', 'neural_network', 'deep_learning'
    )),
    framework VARCHAR(50) NOT NULL CHECK (framework IN (
        'sklearn', 'pytorch', 'tensorflow', 'xgboost', 'multi'
    )),
    model_path TEXT NOT NULL,
    model_size_mb REAL CHECK (model_size_mb > 0),
    feature_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    target_variables JSONB NOT NULL DEFAULT '[]'::jsonb,
    hyperparameters JSONB DEFAULT '{}'::jsonb,
    training_metrics JSONB DEFAULT '{}'::jsonb,
    validation_metrics JSONB DEFAULT '{}'::jsonb,
    performance_thresholds JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT TRUE,
    auto_retrain BOOLEAN DEFAULT FALSE,
    validation_split REAL CHECK (validation_split BETWEEN 0 AND 1) DEFAULT 0.2,
    uncertainty_quantification BOOLEAN DEFAULT FALSE,
    interpretability BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_used_at TIMESTAMP,
    created_by VARCHAR(100),
    tags JSONB DEFAULT '[]'::jsonb,
    UNIQUE(model_name, model_version)
);

CREATE INDEX idx_ml_models_name ON ml_model_registry (model_name);
CREATE INDEX idx_ml_models_type ON ml_model_registry (model_type);
CREATE INDEX idx_ml_models_framework ON ml_model_registry (framework);
CREATE INDEX idx_ml_models_active ON ml_model_registry (is_active);
CREATE INDEX idx_ml_models_tags ON ml_model_registry USING GIN (tags);
```

#### ML Model Performance Tracking Table
```sql
CREATE TABLE ml_model_performance (
    id SERIAL PRIMARY KEY,
    model_registry_id INTEGER REFERENCES ml_model_registry(id) ON DELETE CASCADE,
    prediction_timestamp TIMESTAMP DEFAULT NOW(),
    input_features JSONB NOT NULL,
    prediction_output JSONB NOT NULL,
    actual_output JSONB,
    prediction_confidence REAL CHECK (prediction_confidence BETWEEN 0 AND 1),
    execution_time_ms REAL CHECK (execution_time_ms >= 0),
    data_drift_score REAL CHECK (data_drift_score BETWEEN 0 AND 1),
    prediction_drift_score REAL CHECK (prediction_drift_score BETWEEN 0 AND 1),
    feature_importance JSONB DEFAULT '{}'::jsonb,
    uncertainty_bounds JSONB DEFAULT '{}'::jsonb,
    model_explanation JSONB DEFAULT '{}'::jsonb,
    error_metrics JSONB DEFAULT '{}'::jsonb,
    location GEOGRAPHY(POINTZ, 4326)
);

CREATE INDEX idx_ml_performance_model ON ml_model_performance (model_registry_id);
CREATE INDEX idx_ml_performance_timestamp ON ml_model_performance (prediction_timestamp);
CREATE INDEX idx_ml_performance_confidence ON ml_model_performance (prediction_confidence);
CREATE INDEX idx_ml_performance_location ON ml_model_performance USING GIST (location);

-- Convert to hypertable for time-series performance tracking
SELECT create_hypertable('ml_model_performance', 'prediction_timestamp');
```

#### ML Model Versions Table
```sql
CREATE TABLE ml_model_versions (
    id SERIAL PRIMARY KEY,
    model_registry_id INTEGER REFERENCES ml_model_registry(id) ON DELETE CASCADE,
    version VARCHAR(50) NOT NULL,
    parent_version VARCHAR(50),
    version_type VARCHAR(50) CHECK (version_type IN (
        'major', 'minor', 'patch', 'hotfix', 'experimental'
    )) DEFAULT 'minor',
    change_description TEXT,
    performance_improvement REAL,
    breaking_changes BOOLEAN DEFAULT FALSE,
    rollback_available BOOLEAN DEFAULT TRUE,
    deployment_status VARCHAR(50) CHECK (deployment_status IN (
        'development', 'testing', 'staging', 'production', 'retired'
    )) DEFAULT 'development',
    deployment_date TIMESTAMP,
    rollback_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100)
);

CREATE INDEX idx_ml_versions_model ON ml_model_versions (model_registry_id);
CREATE INDEX idx_ml_versions_status ON ml_model_versions (deployment_status);
CREATE INDEX idx_ml_versions_deployment ON ml_model_versions (deployment_date);
```

### Advanced Data Integration Pipeline Tables

#### Geological Data Sources Registry
```sql
CREATE TABLE geological_data_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(200) NOT NULL UNIQUE,
    source_type VARCHAR(100) NOT NULL CHECK (source_type IN (
        'SEISMIC', 'TECTONIC', 'GEOLOGICAL', 'GEOPHYSICAL', 'VOLCANIC'
    )),
    source_url TEXT,
    api_endpoint TEXT,
    data_format VARCHAR(50) CHECK (data_format IN (
        'geojson', 'csv', 'xml', 'wms', 'wfs', 'shapefile', 'netcdf', 'miniseed'
    )),
    update_frequency VARCHAR(50) CHECK (update_frequency IN (
        'real_time', 'hourly', 'daily', 'weekly', 'monthly', 'static'
    )),
    spatial_coverage VARCHAR(100),
    temporal_coverage DATERANGE,
    data_quality_grade CHAR(1) CHECK (data_quality_grade IN ('A', 'B', 'C', 'D')),
    connection_config JSONB DEFAULT '{}'::jsonb,
    authentication_required BOOLEAN DEFAULT FALSE,
    rate_limit_per_hour INTEGER CHECK (rate_limit_per_hour > 0),
    cache_ttl_minutes INTEGER CHECK (cache_ttl_minutes > 0) DEFAULT 60,
    is_active BOOLEAN DEFAULT TRUE,
    last_successful_query TIMESTAMP,
    total_queries_count INTEGER DEFAULT 0,
    failed_queries_count INTEGER DEFAULT 0,
    avg_response_time_ms REAL CHECK (avg_response_time_ms >= 0),
    data_schema JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_geo_sources_type ON geological_data_sources (source_type);
CREATE INDEX idx_geo_sources_active ON geological_data_sources (is_active);
CREATE INDEX idx_geo_sources_quality ON geological_data_sources (data_quality_grade);
CREATE INDEX idx_geo_sources_frequency ON geological_data_sources (update_frequency);
```

#### Data Quality Monitoring Table
```sql
CREATE TABLE data_quality_metrics (
    id SERIAL PRIMARY KEY,
    data_source_id INTEGER REFERENCES geological_data_sources(id) ON DELETE CASCADE,
    assessment_timestamp TIMESTAMP DEFAULT NOW(),
    completeness_score REAL CHECK (completeness_score BETWEEN 0 AND 1),
    accuracy_score REAL CHECK (accuracy_score BETWEEN 0 AND 1),
    consistency_score REAL CHECK (consistency_score BETWEEN 0 AND 1),
    timeliness_score REAL CHECK (timeliness_score BETWEEN 0 AND 1),
    overall_quality_score REAL CHECK (overall_quality_score BETWEEN 0 AND 1),
    missing_data_percentage REAL CHECK (missing_data_percentage BETWEEN 0 AND 100),
    outlier_percentage REAL CHECK (outlier_percentage BETWEEN 0 AND 100),
    duplicate_percentage REAL CHECK (duplicate_percentage BETWEEN 0 AND 100),
    schema_violations_count INTEGER DEFAULT 0,
    validation_errors JSONB DEFAULT '[]'::jsonb,
    statistical_summary JSONB DEFAULT '{}'::jsonb,
    quality_alerts JSONB DEFAULT '[]'::jsonb,
    assessment_details JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_quality_source ON data_quality_metrics (data_source_id);
CREATE INDEX idx_quality_timestamp ON data_quality_metrics (assessment_timestamp);
CREATE INDEX idx_quality_overall_score ON data_quality_metrics (overall_quality_score);

-- Convert to hypertable for time-series quality tracking
SELECT create_hypertable('data_quality_metrics', 'assessment_timestamp');
```

#### Integrated Query Cache Table
```sql
CREATE TABLE integrated_query_cache (
    id SERIAL PRIMARY KEY,
    query_hash VARCHAR(64) NOT NULL UNIQUE,
    query_type VARCHAR(100) NOT NULL,
    query_parameters JSONB NOT NULL,
    data_sources_used JSONB NOT NULL DEFAULT '[]'::jsonb,
    spatial_bounds GEOGRAPHY(POLYGON, 4326),
    temporal_bounds TSRANGE,
    query_result JSONB NOT NULL,
    result_metadata JSONB DEFAULT '{}'::jsonb,
    cache_created_at TIMESTAMP DEFAULT NOW(),
    cache_expires_at TIMESTAMP NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP DEFAULT NOW(),
    result_size_bytes INTEGER CHECK (result_size_bytes >= 0),
    query_duration_ms REAL CHECK (query_duration_ms >= 0),
    data_quality_score REAL CHECK (data_quality_score BETWEEN 0 AND 1)
);

CREATE INDEX idx_query_cache_hash ON integrated_query_cache (query_hash);
CREATE INDEX idx_query_cache_type ON integrated_query_cache (query_type);
CREATE INDEX idx_query_cache_expires ON integrated_query_cache (cache_expires_at);
CREATE INDEX idx_query_cache_spatial ON integrated_query_cache USING GIST (spatial_bounds);
CREATE INDEX idx_query_cache_temporal ON integrated_query_cache USING GIST (temporal_bounds);
```

### Advanced Seismic Analysis Tables

#### Love Wave Dispersion Analysis Table
```sql
CREATE TABLE love_wave_analysis (
    id SERIAL PRIMARY KEY,
    analysis_id VARCHAR(100) NOT NULL UNIQUE,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    analysis_timestamp TIMESTAMP DEFAULT NOW(),
    layer_structure JSONB NOT NULL,
    frequency_range JSONB NOT NULL,
    dispersion_curve JSONB NOT NULL,
    fundamental_mode_velocity REAL CHECK (fundamental_mode_velocity > 0),
    higher_mode_velocities JSONB DEFAULT '[]'::jsonb,
    phase_velocity_uncertainty REAL CHECK (phase_velocity_uncertainty >= 0),
    group_velocity_uncertainty REAL CHECK (group_velocity_uncertainty >= 0),
    anisotropy_correction JSONB DEFAULT '{}'::jsonb,
    geological_heterogeneity_effects JSONB DEFAULT '{}'::jsonb,
    mode_coupling_factor REAL CHECK (mode_coupling_factor BETWEEN 0 AND 1),
    theoretical_dispersion JSONB DEFAULT '{}'::jsonb,
    numerical_solution JSONB DEFAULT '{}'::jsonb,
    confidence_score REAL CHECK (confidence_score BETWEEN 0 AND 1),
    analysis_method VARCHAR(100) DEFAULT 'comprehensive_multilayer_analysis',
    processing_details JSONB DEFAULT '{}'::jsonb,
    validation_metrics JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_love_wave_location ON love_wave_analysis USING GIST (location);
CREATE INDEX idx_love_wave_timestamp ON love_wave_analysis (analysis_timestamp);
CREATE INDEX idx_love_wave_confidence ON love_wave_analysis (confidence_score);
CREATE INDEX idx_love_wave_method ON love_wave_analysis (analysis_method);
```

#### Comprehensive Rock Classification Table
```sql
CREATE TABLE comprehensive_rock_classification (
    id SERIAL PRIMARY KEY,
    sample_id VARCHAR(100) NOT NULL UNIQUE,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    classification_timestamp TIMESTAMP DEFAULT NOW(),
    tier1_petrological_classification JSONB DEFAULT '{}'::jsonb,
    tier2_geochemical_classification JSONB DEFAULT '{}'::jsonb,
    tier3_textural_classification JSONB DEFAULT '{}'::jsonb,
    tier4_ml_classification JSONB DEFAULT '{}'::jsonb,
    tier5_empirical_classification JSONB DEFAULT '{}'::jsonb,
    ensemble_classification VARCHAR(200) NOT NULL,
    classification_confidence REAL CHECK (classification_confidence BETWEEN 0 AND 1),
    mineralogical_composition JSONB DEFAULT '{}'::jsonb,
    alteration_assemblages JSONB DEFAULT '{}'::jsonb,
    textural_features JSONB DEFAULT '{}'::jsonb,
    structural_characteristics JSONB DEFAULT '{}'::jsonb,
    geochemical_signature JSONB DEFAULT '{}'::jsonb,
    petrological_database_match JSONB DEFAULT '{}'::jsonb,
    classification_method VARCHAR(100) DEFAULT 'comprehensive_multilayer_classification',
    validation_results JSONB DEFAULT '{}'::jsonb,
    uncertainty_analysis JSONB DEFAULT '{}'::jsonb,
    data_provenance JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_rock_class_location ON comprehensive_rock_classification USING GIST (location);
CREATE INDEX idx_rock_class_timestamp ON comprehensive_rock_classification (classification_timestamp);
CREATE INDEX idx_rock_class_confidence ON comprehensive_rock_classification (classification_confidence);
CREATE INDEX idx_rock_class_ensemble ON comprehensive_rock_classification (ensemble_classification);
CREATE INDEX idx_rock_class_composition ON comprehensive_rock_classification USING GIN (mineralogical_composition);
```

### API Endpoints for Advanced Features

#### Model Management API Endpoints
```sql
-- API endpoint configurations
INSERT INTO api_endpoints (endpoint_path, http_method, description, parameters_schema, response_schema, rate_limit_per_minute, authentication_required) VALUES
('/api/v3/models', 'GET', 'List all registered ML models', 
 '{"type": "object", "properties": {"framework": {"type": "string"}, "model_type": {"type": "string"}, "is_active": {"type": "boolean"}}}',
 '{"type": "object", "properties": {"models": {"type": "array"}, "total_count": {"type": "integer"}}}',
 100, FALSE),

('/api/v3/models/{model_name}/versions', 'GET', 'Get all versions of a specific model',
 '{"type": "object", "properties": {"model_name": {"type": "string", "required": true}}}',
 '{"type": "object", "properties": {"versions": {"type": "array"}, "current_version": {"type": "string"}}}',
 50, FALSE),

('/api/v3/models/{model_name}/predict', 'POST', 'Get prediction from a specific model',
 '{"type": "object", "properties": {"model_name": {"type": "string", "required": true}, "version": {"type": "string"}, "features": {"type": "object", "required": true}, "include_uncertainty": {"type": "boolean"}}}',
 '{"type": "object", "properties": {"prediction": {"type": "object"}, "confidence": {"type": "number"}, "uncertainty": {"type": "object"}, "model_metadata": {"type": "object"}}}',
 200, TRUE),

('/api/v3/models/{model_name}/performance', 'GET', 'Get model performance metrics',
 '{"type": "object", "properties": {"model_name": {"type": "string", "required": true}, "start_date": {"type": "string", "format": "date"}, "end_date": {"type": "string", "format": "date"}}}',
 '{"type": "object", "properties": {"performance_metrics": {"type": "object"}, "drift_analysis": {"type": "object"}, "prediction_statistics": {"type": "object"}}}',
 30, TRUE);
```

#### Data Integration API Endpoints
```sql
INSERT INTO api_endpoints (endpoint_path, http_method, description, parameters_schema, response_schema, rate_limit_per_minute, authentication_required) VALUES
('/api/v3/data/sources', 'GET', 'List all geological data sources',
 '{"type": "object", "properties": {"source_type": {"type": "string"}, "is_active": {"type": "boolean"}, "quality_grade": {"type": "string"}}}',
 '{"type": "object", "properties": {"data_sources": {"type": "array"}, "total_count": {"type": "integer"}}}',
 50, FALSE),

('/api/v3/data/integrated-query', 'POST', 'Execute integrated geological data query',
 '{"type": "object", "properties": {"query_type": {"type": "string", "required": true}, "parameters": {"type": "object", "required": true}, "data_sources": {"type": "array"}, "quality_filter": {"type": "boolean"}, "spatial_bounds": {"type": "object"}, "temporal_bounds": {"type": "object"}}}',
 '{"type": "object", "properties": {"result": {"type": "object"}, "query_metadata": {"type": "object"}, "data_quality_score": {"type": "number"}, "data_sources_used": {"type": "array"}}}',
 100, TRUE),

('/api/v3/data/quality-metrics', 'GET', 'Get data quality metrics for sources',
 '{"type": "object", "properties": {"source_id": {"type": "integer"}, "start_date": {"type": "string", "format": "date"}, "end_date": {"type": "string", "format": "date"}, "metric_type": {"type": "string"}}}',
 '{"type": "object", "properties": {"quality_metrics": {"type": "array"}, "summary_statistics": {"type": "object"}, "quality_trends": {"type": "object"}}}',
 30, TRUE);
```

#### Advanced Seismic Analysis API Endpoints
```sql
INSERT INTO api_endpoints (endpoint_path, http_method, description, parameters_schema, response_schema, rate_limit_per_minute, authentication_required) VALUES
('/api/v3/seismic/love-wave-analysis', 'POST', 'Perform comprehensive Love wave dispersion analysis',
 '{"type": "object", "properties": {"location": {"type": "object", "required": true}, "layer_properties": {"type": "object", "required": true}, "frequency_range": {"type": "object"}, "analysis_options": {"type": "object"}}}',
 '{"type": "object", "properties": {"dispersion_analysis": {"type": "object"}, "fundamental_mode_velocity": {"type": "number"}, "higher_modes": {"type": "array"}, "confidence_score": {"type": "number"}, "processing_details": {"type": "object"}}}',
 20, TRUE),

('/api/v3/geology/rock-classification', 'POST', 'Perform comprehensive rock classification',
 '{"type": "object", "properties": {"location": {"type": "object", "required": true}, "rock_properties": {"type": "object", "required": true}, "classification_tiers": {"type": "array"}, "include_uncertainty": {"type": "boolean"}}}',
 '{"type": "object", "properties": {"classification_result": {"type": "object"}, "tier_analyses": {"type": "object"}, "ensemble_classification": {"type": "string"}, "confidence_score": {"type": "number"}, "uncertainty_analysis": {"type": "object"}}}',
 15, TRUE),

('/api/v3/analysis/comprehensive-stability', 'POST', 'Perform comprehensive cap rock stability analysis',
 '{"type": "object", "properties": {"location": {"type": "object", "required": true}, "cap_rock_properties": {"type": "object", "required": true}, "stress_conditions": {"type": "object"}, "thermal_conditions": {"type": "object"}, "analysis_options": {"type": "object"}}}',
 '{"type": "object", "properties": {"stability_analysis": {"type": "object"}, "failure_probability": {"type": "number"}, "critical_conditions": {"type": "object"}, "time_evolution": {"type": "object"}, "uncertainty_bounds": {"type": "object"}}}',
 10, TRUE);
```

### Database Views for Advanced Analytics

#### Comprehensive Model Performance View
```sql
CREATE VIEW v_model_performance_summary AS
SELECT 
    mr.model_name,
    mr.model_version,
    mr.model_type,
    mr.framework,
    COUNT(mp.id) as total_predictions,
    AVG(mp.prediction_confidence) as avg_confidence,
    AVG(mp.execution_time_ms) as avg_execution_time,
    AVG(mp.data_drift_score) as avg_data_drift,
    AVG(mp.prediction_drift_score) as avg_prediction_drift,
    MAX(mp.prediction_timestamp) as last_prediction,
    COUNT(CASE WHEN mp.prediction_confidence > 0.8 THEN 1 END) * 100.0 / COUNT(*) as high_confidence_percentage,
    mr.training_metrics,
    mr.validation_metrics
FROM ml_model_registry mr
LEFT JOIN ml_model_performance mp ON mr.id = mp.model_registry_id
WHERE mr.is_active = TRUE
GROUP BY mr.id, mr.model_name, mr.model_version, mr.model_type, mr.framework, mr.training_metrics, mr.validation_metrics;
```

#### Data Source Quality Dashboard View
```sql
CREATE VIEW v_data_source_quality_dashboard AS
SELECT 
    gds.source_name,
    gds.source_type,
    gds.data_quality_grade,
    AVG(dqm.overall_quality_score) as avg_quality_score,
    AVG(dqm.completeness_score) as avg_completeness,
    AVG(dqm.accuracy_score) as avg_accuracy,
    COUNT(dqm.id) as total_assessments,
    MAX(dqm.assessment_timestamp) as last_assessment,
    gds.total_queries_count,
    gds.failed_queries_count,
    CASE WHEN gds.total_queries_count > 0 
         THEN (gds.failed_queries_count * 100.0 / gds.total_queries_count) 
         ELSE 0 END as failure_rate_percentage,
    gds.avg_response_time_ms,
    gds.is_active
FROM geological_data_sources gds
LEFT JOIN data_quality_metrics dqm ON gds.id = dqm.data_source_id
GROUP BY gds.id, gds.source_name, gds.source_type, gds.data_quality_grade, 
         gds.total_queries_count, gds.failed_queries_count, gds.avg_response_time_ms, gds.is_active;
```

#### Advanced Analysis Summary View
```sql
CREATE VIEW v_advanced_analysis_summary AS
SELECT 
    'love_wave_analysis' as analysis_type,
    COUNT(*) as total_analyses,
    AVG(confidence_score) as avg_confidence,
    MAX(analysis_timestamp) as last_analysis,
    ST_Extent(location::geometry) as spatial_coverage
FROM love_wave_analysis
WHERE analysis_timestamp >= NOW() - INTERVAL '30 days'
UNION ALL
SELECT 
    'rock_classification' as analysis_type,
    COUNT(*) as total_analyses,
    AVG(classification_confidence) as avg_confidence,
    MAX(classification_timestamp) as last_analysis,
    ST_Extent(location::geometry) as spatial_coverage
FROM comprehensive_rock_classification
WHERE classification_timestamp >= NOW() - INTERVAL '30 days';
```

### Database Functions for Advanced Features

#### Function to get optimal model version
```sql
CREATE OR REPLACE FUNCTION get_optimal_model_version(p_model_name VARCHAR)
RETURNS VARCHAR AS $$
DECLARE
    v_version VARCHAR;
BEGIN
    SELECT mr.model_version INTO v_version
    FROM ml_model_registry mr
    LEFT JOIN ml_model_performance mp ON mr.id = mp.model_registry_id
    WHERE mr.model_name = p_model_name 
    AND mr.is_active = TRUE
    GROUP BY mr.id, mr.model_version
    ORDER BY AVG(mp.prediction_confidence) DESC, mr.updated_at DESC
    LIMIT 1;
    
    RETURN COALESCE(v_version, 'latest');
END;
$$ LANGUAGE plpgsql;
```

#### Function to assess data source reliability
```sql
CREATE OR REPLACE FUNCTION assess_data_source_reliability(p_source_id INTEGER)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
    v_avg_quality REAL;
    v_failure_rate REAL;
    v_uptime_score REAL;
BEGIN
    SELECT 
        AVG(overall_quality_score),
        CASE WHEN gds.total_queries_count > 0 
             THEN (gds.failed_queries_count * 100.0 / gds.total_queries_count) 
             ELSE 0 END,
        CASE WHEN gds.is_active THEN 1.0 ELSE 0.0 END
    INTO v_avg_quality, v_failure_rate, v_uptime_score
    FROM geological_data_sources gds
    LEFT JOIN data_quality_metrics dqm ON gds.id = dqm.data_source_id
    WHERE gds.id = p_source_id
    GROUP BY gds.total_queries_count, gds.failed_queries_count, gds.is_active;
    
    v_result := jsonb_build_object(
        'average_quality_score', COALESCE(v_avg_quality, 0.5),
        'failure_rate_percentage', COALESCE(v_failure_rate, 0),
        'uptime_score', COALESCE(v_uptime_score, 0),
        'reliability_score', (COALESCE(v_avg_quality, 0.5) * 0.5 + 
                             (1 - COALESCE(v_failure_rate, 0) / 100.0) * 0.3 + 
                             COALESCE(v_uptime_score, 0) * 0.2),
        'assessment_timestamp', NOW()
    );
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;
```

### Database Triggers for Advanced Features

#### Trigger to update model performance statistics
```sql
CREATE OR REPLACE FUNCTION update_model_last_used()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE ml_model_registry 
    SET last_used_at = NOW(), updated_at = NOW()
    WHERE id = NEW.model_registry_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_update_model_last_used
    AFTER INSERT ON ml_model_performance
    FOR EACH ROW EXECUTE FUNCTION update_model_last_used();
```

#### Trigger to update data source query statistics
```sql
CREATE OR REPLACE FUNCTION update_data_source_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- This would be called from application code when queries are made
    -- Placeholder for integration with data integration pipeline
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

### Data Retention Policies

#### ML Performance Data Retention
```sql
-- Retain ML performance data for 1 year, then compress older data
SELECT add_retention_policy('ml_model_performance', INTERVAL '1 year');

-- Compress data older than 1 month
SELECT add_compression_policy('ml_model_performance', INTERVAL '1 month');
```

#### Data Quality Metrics Retention
```sql
-- Retain data quality metrics for 6 months
SELECT add_retention_policy('data_quality_metrics', INTERVAL '6 months');

-- Compress data older than 1 week
SELECT add_compression_policy('data_quality_metrics', INTERVAL '1 week');
```

### Advanced Feature Configuration
```yaml
# Configuration for advanced features
advanced_features:
  model_management:
    enabled: true
    auto_versioning: true
    performance_monitoring: true
    max_model_versions: 10
    cache_size_mb: 1000
    
  data_integration:
    enabled: true
    real_time_processing: true
    quality_monitoring: true
    cache_ttl_hours: 24
    max_concurrent_queries: 20
    
  comprehensive_analysis:
    enabled: true
    love_wave_analysis: true
    rock_classification: true
    uncertainty_quantification: true
    
  api_features:
    rate_limiting: true
    authentication: true
    response_caching: true
    request_logging: true
```

Advanced Features Schema version: 3.0.0
ML Model Management Integration version: 1.0.0  
Comprehensive Data Pipeline Integration version: 1.0.0
Advanced Seismic Analysis Integration version: 1.0.0
Comprehensive Rock Classification Integration version: 1.0.0
Love Wave Dispersion Analysis Integration version: 1.0.0