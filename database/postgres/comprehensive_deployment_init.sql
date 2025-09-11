-- =============================================================================
-- COMPREHENSIVE DATABASE DEPLOYMENT CONFIGURATION
-- Prometheus AI Platform - Advanced Geological Database Setup
-- =============================================================================

-- Enable required extensions for advanced geological analysis
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS hstore;
CREATE EXTENSION IF NOT EXISTS ltree;

-- =============================================================================
-- SPECIALIZED SCHEMAS FOR GEOLOGICAL DATA ORGANIZATION
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS geological_age;
CREATE SCHEMA IF NOT EXISTS model_management;
CREATE SCHEMA IF NOT EXISTS mlops;
CREATE SCHEMA IF NOT EXISTS external_data;
CREATE SCHEMA IF NOT EXISTS backup_archive;
CREATE SCHEMA IF NOT EXISTS performance_monitoring;

-- =============================================================================
-- GEOLOGICAL AGE DATABASE INTEGRATION TABLES
-- =============================================================================

-- Comprehensive geological age reference data
CREATE TABLE geological_age.age_references (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    reference_id VARCHAR(100) UNIQUE NOT NULL,
    formation_name VARCHAR(200) NOT NULL,
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    
    -- Multiple age determination methods
    stratigraphic_age_ma REAL CHECK (stratigraphic_age_ma >= 0 AND stratigraphic_age_ma <= 4500),
    radiometric_age_ma REAL CHECK (radiometric_age_ma >= 0 AND radiometric_age_ma <= 4500),
    biostratigraphic_age_ma REAL CHECK (biostratigraphic_age_ma >= 0 AND biostratigraphic_age_ma <= 4500),
    magnetostratigraphic_age_ma REAL CHECK (magnetostratigraphic_age_ma >= 0 AND magnetostratigraphic_age_ma <= 4500),
    
    -- Age uncertainties and quality metrics
    age_uncertainty_ma REAL CHECK (age_uncertainty_ma >= 0),
    dating_method VARCHAR(100) NOT NULL,
    isotope_system VARCHAR(50),
    material_dated VARCHAR(100),
    analytical_precision_ma REAL,
    
    -- Data quality and confidence
    data_quality_grade CHAR(1) CHECK (data_quality_grade IN ('A', 'B', 'C', 'D')),
    confidence_level REAL CHECK (confidence_level BETWEEN 0 AND 1),
    reliability_score REAL CHECK (reliability_score BETWEEN 0 AND 1),
    
    -- External data source integration
    earthchem_sample_id VARCHAR(100),
    georoc_sample_id VARCHAR(100),
    navdat_sample_id VARCHAR(100),
    petdb_sample_id VARCHAR(100),
    
    -- Geological context
    lithology VARCHAR(200),
    geological_unit VARCHAR(200),
    tectonic_setting VARCHAR(100),
    geological_context JSONB DEFAULT '{}'::jsonb,
    
    -- Literature and provenance
    citation TEXT,
    doi VARCHAR(200),
    publication_year INTEGER,
    author_list TEXT,
    laboratory VARCHAR(200),
    
    -- Temporal tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_validated TIMESTAMPTZ
);

-- Comprehensive indexing for geological age data
CREATE INDEX idx_age_references_spatial ON geological_age.age_references USING GIST (location);
CREATE INDEX idx_age_references_stratigraphic_age ON geological_age.age_references (stratigraphic_age_ma);
CREATE INDEX idx_age_references_radiometric_age ON geological_age.age_references (radiometric_age_ma);
CREATE INDEX idx_age_references_quality ON geological_age.age_references (data_quality_grade, confidence_level);
CREATE INDEX idx_age_references_method ON geological_age.age_references (dating_method);
CREATE INDEX idx_age_references_lithology ON geological_age.age_references (lithology);
CREATE INDEX idx_age_references_external_ids ON geological_age.age_references (earthchem_sample_id, georoc_sample_id, navdat_sample_id, petdb_sample_id);
CREATE INDEX idx_age_references_context ON geological_age.age_references USING GIN (geological_context);

-- External data source synchronization tracking
CREATE TABLE geological_age.external_sync_status (
    id SERIAL PRIMARY KEY,
    data_source VARCHAR(50) NOT NULL,
    last_sync_timestamp TIMESTAMPTZ DEFAULT NOW(),
    sync_status VARCHAR(20) DEFAULT 'pending',
    records_synchronized INTEGER DEFAULT 0,
    sync_errors JSONB DEFAULT '[]'::jsonb,
    api_endpoint VARCHAR(500),
    api_key_hash VARCHAR(64),
    sync_configuration JSONB DEFAULT '{}'::jsonb
);

-- Age prediction model training data
CREATE TABLE geological_age.training_datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name VARCHAR(200) NOT NULL,
    dataset_version VARCHAR(20),
    feature_columns TEXT[],
    target_column VARCHAR(100),
    spatial_extent GEOMETRY(POLYGON, 4326),
    temporal_range TSTZRANGE,
    sample_count INTEGER,
    data_quality_score REAL CHECK (data_quality_score BETWEEN 0 AND 1),
    preprocessing_steps JSONB DEFAULT '[]'::jsonb,
    validation_metrics JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- MODEL MANAGEMENT SYSTEM
-- =============================================================================

-- ML Model metadata and versioning
CREATE TABLE model_management.model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(200) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL, -- 'age_prediction', 'eruption_forecasting', etc.
    
    -- Model architecture and parameters
    architecture_type VARCHAR(100), -- 'neural_network', 'random_forest', 'svm', etc.
    hyperparameters JSONB DEFAULT '{}'::jsonb,
    feature_importance JSONB DEFAULT '{}'::jsonb,
    
    -- Performance metrics
    training_accuracy REAL,
    validation_accuracy REAL,
    test_accuracy REAL,
    cross_validation_score REAL,
    performance_metrics JSONB DEFAULT '{}'::jsonb,
    
    -- Model artifacts and storage
    model_file_path TEXT,
    model_size_bytes BIGINT,
    storage_backend VARCHAR(50), -- 'local', 'azure_blob', 'aws_s3', 'gcp_storage'
    storage_uri TEXT,
    checksum_md5 VARCHAR(32),
    
    -- Training configuration
    training_dataset_id UUID REFERENCES geological_age.training_datasets(id),
    training_start_time TIMESTAMPTZ,
    training_end_time TIMESTAMPTZ,
    training_duration_seconds INTEGER,
    compute_resources JSONB DEFAULT '{}'::jsonb,
    
    -- Deployment and lifecycle
    deployment_status VARCHAR(20) DEFAULT 'trained', -- 'trained', 'deployed', 'retired'
    deployment_environment VARCHAR(50), -- 'development', 'staging', 'production'
    deployment_timestamp TIMESTAMPTZ,
    retirement_timestamp TIMESTAMPTZ,
    
    -- Model lineage and relationships
    parent_model_id UUID REFERENCES model_management.model_registry(id),
    model_lineage JSONB DEFAULT '[]'::jsonb,
    
    -- Metadata and documentation
    description TEXT,
    tags TEXT[],
    created_by VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(model_name, model_version)
);

-- Model performance monitoring
CREATE TABLE model_management.model_performance_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES model_management.model_registry(id),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Prediction metrics
    prediction_count INTEGER,
    average_prediction_time_ms REAL,
    prediction_accuracy REAL,
    confidence_scores REAL[],
    
    -- Data drift detection
    feature_drift_score REAL,
    target_drift_score REAL,
    drift_detection_method VARCHAR(100),
    
    -- System performance
    memory_usage_mb REAL,
    cpu_usage_percent REAL,
    gpu_usage_percent REAL,
    
    -- Error tracking
    error_count INTEGER DEFAULT 0,
    error_types JSONB DEFAULT '[]'::jsonb,
    
    -- Additional metrics
    custom_metrics JSONB DEFAULT '{}'::jsonb
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('model_management.model_performance_logs', 'timestamp');

-- =============================================================================
-- MLOPS PIPELINE CONFIGURATION
-- =============================================================================

-- MLflow experiment tracking
CREATE TABLE mlops.experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_name VARCHAR(200) UNIQUE NOT NULL,
    experiment_id VARCHAR(100) UNIQUE NOT NULL, -- MLflow experiment ID
    description TEXT,
    tags JSONB DEFAULT '{}'::jsonb,
    artifact_location TEXT,
    lifecycle_stage VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- MLflow run tracking
CREATE TABLE mlops.runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id VARCHAR(100) UNIQUE NOT NULL, -- MLflow run ID
    experiment_id UUID REFERENCES mlops.experiments(id),
    run_name VARCHAR(200),
    
    -- Run metadata
    status VARCHAR(20) DEFAULT 'RUNNING', -- 'RUNNING', 'FINISHED', 'FAILED', 'KILLED'
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    
    -- Parameters and metrics
    parameters JSONB DEFAULT '{}'::jsonb,
    metrics JSONB DEFAULT '{}'::jsonb,
    tags JSONB DEFAULT '{}'::jsonb,
    
    -- Artifacts and outputs
    artifact_uri TEXT,
    model_artifacts JSONB DEFAULT '[]'::jsonb,
    
    -- System information
    source_type VARCHAR(50),
    source_name TEXT,
    user_id VARCHAR(100),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Pipeline execution tracking
CREATE TABLE mlops.pipeline_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_name VARCHAR(200) NOT NULL,
    execution_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Execution metadata
    status VARCHAR(20) DEFAULT 'RUNNING',
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    duration_seconds INTEGER,
    
    -- Pipeline configuration
    pipeline_config JSONB DEFAULT '{}'::jsonb,
    input_datasets JSONB DEFAULT '[]'::jsonb,
    output_artifacts JSONB DEFAULT '[]'::jsonb,
    
    -- Resource usage
    compute_resources JSONB DEFAULT '{}'::jsonb,
    cost_estimate_usd REAL,
    
    -- Error handling
    error_message TEXT,
    error_stack_trace TEXT,
    retry_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- EXTERNAL DATA SOURCE CONFIGURATION
-- =============================================================================

-- EarthChem data integration
CREATE TABLE external_data.earthchem_samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    earthchem_sample_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOGRAPHY(POINTZ, 4326),
    
    -- Sample metadata
    sample_name VARCHAR(200),
    rock_type VARCHAR(100),
    mineral VARCHAR(100),
    
    -- Geochemical data
    major_elements JSONB DEFAULT '{}'::jsonb,
    trace_elements JSONB DEFAULT '{}'::jsonb,
    isotope_ratios JSONB DEFAULT '{}'::jsonb,
    
    -- Age information
    age_ma REAL,
    age_method VARCHAR(100),
    age_uncertainty REAL,
    
    -- Data provenance
    publication_doi VARCHAR(200),
    data_quality VARCHAR(20),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    sync_status VARCHAR(20) DEFAULT 'pending'
);

-- GEOROC data integration
CREATE TABLE external_data.georoc_samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    georoc_sample_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOGRAPHY(POINTZ, 4326),
    
    -- Sample classification
    rock_class VARCHAR(100),
    rock_type VARCHAR(100),
    tectonic_setting VARCHAR(100),
    
    -- Analytical data
    analytical_data JSONB DEFAULT '{}'::jsonb,
    analytical_methods JSONB DEFAULT '[]'::jsonb,
    
    -- Quality control
    data_quality_flags JSONB DEFAULT '[]'::jsonb,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    sync_status VARCHAR(20) DEFAULT 'pending'
);

-- NAVDAT data integration
CREATE TABLE external_data.navdat_samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    navdat_sample_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOGRAPHY(POINTZ, 4326),
    
    -- Volcanic arc context
    arc_name VARCHAR(200),
    volcano_name VARCHAR(200),
    eruption_age_ma REAL,
    
    -- Geochemical composition
    composition_data JSONB DEFAULT '{}'::jsonb,
    volatile_content JSONB DEFAULT '{}'::jsonb,
    
    -- Metadata
    collection_method VARCHAR(100),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    sync_status VARCHAR(20) DEFAULT 'pending'
);

-- PetDB data integration
CREATE TABLE external_data.petdb_samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    petdb_sample_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOGRAPHY(POINTZ, 4326),
    
    -- Petrological data
    mineral_assemblage JSONB DEFAULT '[]'::jsonb,
    texture_description TEXT,
    metamorphic_grade VARCHAR(100),
    
    -- P-T conditions
    pressure_gpa REAL,
    temperature_c REAL,
    
    -- Analytical results
    mineral_chemistry JSONB DEFAULT '{}'::jsonb,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    sync_status VARCHAR(20) DEFAULT 'pending'
);

-- =============================================================================
-- HIGH-PERFORMANCE COMPUTING CONFIGURATION
-- =============================================================================

-- Parallel processing job tracking
CREATE TABLE performance_monitoring.compute_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_name VARCHAR(200) NOT NULL,
    job_type VARCHAR(100) NOT NULL, -- 'model_training', 'data_processing', 'simulation'
    
    -- Resource allocation
    cpu_cores_requested INTEGER,
    cpu_cores_allocated INTEGER,
    memory_gb_requested REAL,
    memory_gb_allocated REAL,
    gpu_count_requested INTEGER,
    gpu_count_allocated INTEGER,
    gpu_type VARCHAR(100),
    
    -- Execution details
    status VARCHAR(20) DEFAULT 'QUEUED', -- 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    duration_seconds INTEGER,
    
    -- Performance metrics
    cpu_utilization_percent REAL,
    memory_utilization_percent REAL,
    gpu_utilization_percent REAL,
    io_throughput_mbps REAL,
    
    -- Cost tracking
    compute_cost_usd REAL,
    storage_cost_usd REAL,
    
    -- Job configuration
    job_parameters JSONB DEFAULT '{}'::jsonb,
    input_data_size_gb REAL,
    output_data_size_gb REAL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Redis cache performance monitoring
CREATE TABLE performance_monitoring.cache_performance (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Cache metrics
    cache_hit_rate REAL,
    cache_miss_rate REAL,
    total_requests INTEGER,
    
    -- Memory usage
    memory_used_mb REAL,
    memory_available_mb REAL,
    memory_fragmentation_ratio REAL,
    
    -- Performance metrics
    average_response_time_ms REAL,
    operations_per_second REAL,
    
    -- Connection metrics
    connected_clients INTEGER,
    blocked_clients INTEGER,
    
    -- Additional metrics
    evicted_keys INTEGER,
    expired_keys INTEGER
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('performance_monitoring.cache_performance', 'timestamp');

-- =============================================================================
-- BACKUP AND ARCHIVAL CONFIGURATION
-- =============================================================================

-- Backup job tracking
CREATE TABLE backup_archive.backup_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    backup_name VARCHAR(200) NOT NULL,
    backup_type VARCHAR(50) NOT NULL, -- 'full', 'incremental', 'differential'
    
    -- Backup scope
    schemas_included TEXT[],
    tables_included TEXT[],
    data_size_gb REAL,
    
    -- Backup locations
    primary_location VARCHAR(500), -- S3, Azure Blob, etc.
    secondary_location VARCHAR(500),
    tertiary_location VARCHAR(500),
    
    -- Backup metadata
    backup_start_time TIMESTAMPTZ DEFAULT NOW(),
    backup_end_time TIMESTAMPTZ,
    backup_duration_seconds INTEGER,
    
    -- Verification and integrity
    checksum_md5 VARCHAR(32),
    verification_status VARCHAR(20) DEFAULT 'pending',
    verification_timestamp TIMESTAMPTZ,
    
    -- Retention and lifecycle
    retention_days INTEGER DEFAULT 730,
    expiration_date TIMESTAMPTZ,
    archive_status VARCHAR(20) DEFAULT 'active',
    
    -- Performance metrics
    backup_throughput_mbps REAL,
    compression_ratio REAL,
    storage_cost_usd REAL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Archive storage tracking
CREATE TABLE backup_archive.archive_storage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    storage_provider VARCHAR(50) NOT NULL, -- 'aws_s3', 'azure_blob', 'gcp_storage'
    storage_region VARCHAR(100),
    bucket_name VARCHAR(200),
    
    -- Storage configuration
    storage_class VARCHAR(50), -- 'standard', 'cold', 'archive'
    encryption_enabled BOOLEAN DEFAULT TRUE,
    encryption_key_id VARCHAR(200),
    
    -- Capacity and usage
    total_capacity_gb REAL,
    used_capacity_gb REAL,
    available_capacity_gb REAL,
    
    -- Cost tracking
    monthly_cost_usd REAL,
    cost_per_gb_usd REAL,
    
    -- Performance metrics
    upload_throughput_mbps REAL,
    download_throughput_mbps REAL,
    
    -- Status and health
    status VARCHAR(20) DEFAULT 'active',
    last_health_check TIMESTAMPTZ DEFAULT NOW(),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- ADVANCED INDEXING AND OPTIMIZATION
-- =============================================================================

-- Spatial indexes for external data
CREATE INDEX idx_earthchem_samples_spatial ON external_data.earthchem_samples USING GIST (location);
CREATE INDEX idx_georoc_samples_spatial ON external_data.georoc_samples USING GIST (location);
CREATE INDEX idx_navdat_samples_spatial ON external_data.navdat_samples USING GIST (location);
CREATE INDEX idx_petdb_samples_spatial ON external_data.petdb_samples USING GIST (location);

-- Performance indexes for model management
CREATE INDEX idx_model_registry_type_version ON model_management.model_registry (model_type, model_version);
CREATE INDEX idx_model_registry_deployment ON model_management.model_registry (deployment_status, deployment_environment);
CREATE INDEX idx_model_performance_model_time ON model_management.model_performance_logs (model_id, timestamp);

-- MLOps indexes
CREATE INDEX idx_experiments_name ON mlops.experiments (experiment_name);
CREATE INDEX idx_runs_experiment_status ON mlops.runs (experiment_id, status);
CREATE INDEX idx_pipeline_executions_status_time ON mlops.pipeline_executions (status, start_time);

-- Compute job indexes
CREATE INDEX idx_compute_jobs_status_type ON performance_monitoring.compute_jobs (status, job_type);
CREATE INDEX idx_compute_jobs_time_range ON performance_monitoring.compute_jobs (start_time, end_time);

-- Backup indexes
CREATE INDEX idx_backup_jobs_type_time ON backup_archive.backup_jobs (backup_type, backup_start_time);
CREATE INDEX idx_backup_jobs_retention ON backup_archive.backup_jobs (expiration_date, archive_status);

-- =============================================================================
-- TRIGGERS AND FUNCTIONS FOR AUTOMATED MAINTENANCE
-- =============================================================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$ language 'plpgsql';

-- Triggers for timestamp updates
CREATE TRIGGER update_age_references_updated_at 
    BEFORE UPDATE ON geological_age.age_references
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_registry_updated_at 
    BEFORE UPDATE ON model_management.model_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_experiments_updated_at 
    BEFORE UPDATE ON mlops.experiments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_archive_storage_updated_at 
    BEFORE UPDATE ON backup_archive.archive_storage
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function for automatic data quality validation
CREATE OR REPLACE FUNCTION validate_geological_age_data()
RETURNS TRIGGER AS $
BEGIN
    -- Validate age ranges
    IF NEW.stratigraphic_age_ma IS NOT NULL AND (NEW.stratigraphic_age_ma < 0 OR NEW.stratigraphic_age_ma > 4500) THEN
        RAISE EXCEPTION 'Stratigraphic age must be between 0 and 4500 Ma';
    END IF;
    
    -- Validate confidence levels
    IF NEW.confidence_level IS NOT NULL AND (NEW.confidence_level < 0 OR NEW.confidence_level > 1) THEN
        RAISE EXCEPTION 'Confidence level must be between 0 and 1';
    END IF;
    
    -- Auto-assign data quality grade based on confidence and uncertainty
    IF NEW.confidence_level >= 0.9 AND NEW.age_uncertainty_ma <= 1.0 THEN
        NEW.data_quality_grade = 'A';
    ELSIF NEW.confidence_level >= 0.7 AND NEW.age_uncertainty_ma <= 5.0 THEN
        NEW.data_quality_grade = 'B';
    ELSIF NEW.confidence_level >= 0.5 AND NEW.age_uncertainty_ma <= 10.0 THEN
        NEW.data_quality_grade = 'C';
    ELSE
        NEW.data_quality_grade = 'D';
    END IF;
    
    RETURN NEW;
END;
$ language 'plpgsql';

-- Trigger for data quality validation
CREATE TRIGGER validate_age_data_quality 
    BEFORE INSERT OR UPDATE ON geological_age.age_references
    FOR EACH ROW EXECUTE FUNCTION validate_geological_age_data();

-- =============================================================================
-- PERFORMANCE MONITORING VIEWS
-- =============================================================================

-- View for model performance summary
CREATE OR REPLACE VIEW model_management.model_performance_summary AS
SELECT 
    mr.model_name,
    mr.model_version,
    mr.model_type,
    mr.deployment_status,
    AVG(mpl.prediction_accuracy) as avg_accuracy,
    AVG(mpl.average_prediction_time_ms) as avg_prediction_time_ms,
    COUNT(mpl.id) as monitoring_records,
    MAX(mpl.timestamp) as last_monitored
FROM model_management.model_registry mr
LEFT JOIN model_management.model_performance_logs mpl ON mr.id = mpl.model_id
GROUP BY mr.id, mr.model_name, mr.model_version, mr.model_type, mr.deployment_status;

-- View for external data sync status
CREATE OR REPLACE VIEW external_data.sync_status_summary AS
SELECT 
    'earthchem' as data_source,
    COUNT(*) as total_records,
    COUNT(CASE WHEN sync_status = 'completed' THEN 1 END) as synced_records,
    MAX(last_updated) as last_sync_time
FROM external_data.earthchem_samples
UNION ALL
SELECT 
    'georoc' as data_source,
    COUNT(*) as total_records,
    COUNT(CASE WHEN sync_status = 'completed' THEN 1 END) as synced_records,
    MAX(last_updated) as last_sync_time
FROM external_data.georoc_samples
UNION ALL
SELECT 
    'navdat' as data_source,
    COUNT(*) as total_records,
    COUNT(CASE WHEN sync_status = 'completed' THEN 1 END) as synced_records,
    MAX(last_updated) as last_sync_time
FROM external_data.navdat_samples
UNION ALL
SELECT 
    'petdb' as data_source,
    COUNT(*) as total_records,
    COUNT(CASE WHEN sync_status = 'completed' THEN 1 END) as synced_records,
    MAX(last_updated) as last_sync_time
FROM external_data.petdb_samples;

-- =============================================================================
-- INITIAL CONFIGURATION DATA
-- =============================================================================

-- Insert default MLOps configuration
INSERT INTO mlops.experiments (experiment_name, experiment_id, description) VALUES
    ('geological_age_prediction', 'exp_001', 'Geological age prediction using multi-source data'),
    ('eruption_forecasting', 'exp_002', 'Volcanic eruption forecasting models'),
    ('resource_prospectivity', 'exp_003', 'Mineral resource prospectivity mapping'),
    ('seismic_analysis', 'exp_004', 'Seismic event detection and classification')
ON CONFLICT (experiment_name) DO NOTHING;

-- Insert default external data sync configuration
INSERT INTO geological_age.external_sync_status (data_source, api_endpoint, sync_configuration) VALUES
    ('earthchem', 'https://www.earthchem.org/portal/api/', '{"sync_interval_hours": 24, "batch_size": 1000}'),
    ('georoc', 'https://georoc.mpch-mainz.gwdg.de/georoc/api/', '{"sync_interval_hours": 24, "batch_size": 500}'),
    ('navdat', 'https://navdat.org/api/v1/', '{"sync_interval_hours": 12, "batch_size": 200}'),
    ('petdb', 'https://petdb.ldeo.columbia.edu/api/', '{"sync_interval_hours": 24, "batch_size": 300}')
ON CONFLICT (data_source) DO NOTHING;

-- Insert default backup storage configuration
INSERT INTO backup_archive.archive_storage (storage_provider, storage_region, bucket_name, storage_class) VALUES
    ('aws_s3', 'us-west-2', 'prometheus-geological-backups', 'standard'),
    ('azure_blob', 'westus2', 'prometheus-age-data-archive', 'cool'),
    ('gcp_storage', 'us-central1', 'prometheus-ml-models', 'standard')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- PERFORMANCE OPTIMIZATION SETTINGS
-- =============================================================================

-- Optimize PostgreSQL settings for geological data processing
ALTER SYSTEM SET shared_preload_libraries = 'timescaledb,pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Enable query performance monitoring
ALTER SYSTEM SET pg_stat_statements.track = 'all';
ALTER SYSTEM SET pg_stat_statements.max = 10000;
ALTER SYSTEM SET log_statement = 'mod';
ALTER SYSTEM SET log_min_duration_statement = 1000;

-- Reload configuration
SELECT pg_reload_conf();

-- =============================================================================
-- COMPLETION MESSAGE
-- =============================================================================

DO $
BEGIN
    RAISE NOTICE 'Comprehensive database deployment configuration completed successfully!';
    RAISE NOTICE 'Created schemas: geological_age, model_management, mlops, external_data, backup_archive, performance_monitoring';
    RAISE NOTICE 'Configured external data integration for: EarthChem, GEOROC, NAVDAT, PetDB';
    RAISE NOTICE 'Set up MLOps pipeline with MLflow integration';
    RAISE NOTICE 'Configured high-performance computing job tracking';
    RAISE NOTICE 'Implemented multi-location backup and archival system';
    RAISE NOTICE 'Database is ready for advanced geological AI operations!';
END;
$;