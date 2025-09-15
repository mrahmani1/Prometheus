-- PostgreSQL initialization script for Prometheus AI Platform
-- This script sets up the database with required extensions and initial schema

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS hstore;

-- Create schemas for different data domains
CREATE SCHEMA IF NOT EXISTS geological_data;
CREATE SCHEMA IF NOT EXISTS seismic_data;
CREATE SCHEMA IF NOT EXISTS volcanic_data;
CREATE SCHEMA IF NOT EXISTS geochemical_data;
CREATE SCHEMA IF NOT EXISTS monitoring_networks;
CREATE SCHEMA IF NOT EXISTS causal_relationships;
CREATE SCHEMA IF NOT EXISTS metadata;

-- Set up proper permissions
GRANT USAGE ON SCHEMA geological_data TO prometheus;
GRANT USAGE ON SCHEMA seismic_data TO prometheus;
GRANT USAGE ON SCHEMA volcanic_data TO prometheus;
GRANT USAGE ON SCHEMA geochemical_data TO prometheus;
GRANT USAGE ON SCHEMA monitoring_networks TO prometheus;
GRANT USAGE ON SCHEMA causal_relationships TO prometheus;
GRANT USAGE ON SCHEMA metadata TO prometheus;

GRANT CREATE ON SCHEMA geological_data TO prometheus;
GRANT CREATE ON SCHEMA seismic_data TO prometheus;
GRANT CREATE ON SCHEMA volcanic_data TO prometheus;
GRANT CREATE ON SCHEMA geochemical_data TO prometheus;
GRANT CREATE ON SCHEMA monitoring_networks TO prometheus;
GRANT CREATE ON SCHEMA causal_relationships TO prometheus;
GRANT CREATE ON SCHEMA metadata TO prometheus;

-- Create spatial reference system for geological coordinates
INSERT INTO spatial_ref_sys (srid, auth_name, auth_srid, proj4text, srtext) 
VALUES (
    4326,
    'EPSG',
    4326,
    '+proj=longlat +datum=WGS84 +no_defs',
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
) ON CONFLICT (srid) DO NOTHING;

-- Create audit log table for tracking data changes
CREATE TABLE IF NOT EXISTS metadata.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(255) NOT NULL,
    operation VARCHAR(10) NOT NULL,
    old_values JSONB,
    new_values JSONB,
    user_name VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_id VARCHAR(255)
);

-- Create data quality tracking table
CREATE TABLE IF NOT EXISTS metadata.data_quality_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    data_source VARCHAR(255) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    completeness_score FLOAT CHECK (completeness_score >= 0 AND completeness_score <= 1),
    accuracy_score FLOAT CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
    consistency_score FLOAT CHECK (consistency_score >= 0 AND consistency_score <= 1),
    timeliness_score FLOAT CHECK (timeliness_score >= 0 AND timeliness_score <= 1),
    validation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    validation_details JSONB
);

-- Create data lineage tracking table
CREATE TABLE IF NOT EXISTS metadata.data_lineage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_table VARCHAR(255) NOT NULL,
    target_table VARCHAR(255) NOT NULL,
    transformation_type VARCHAR(100) NOT NULL,
    transformation_details JSONB,
    processing_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processor_id VARCHAR(255),
    data_volume BIGINT,
    processing_duration INTERVAL
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_audit_log_table_timestamp ON metadata.audit_log(table_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_data_quality_source_timestamp ON metadata.data_quality_metrics(data_source, validation_timestamp);
CREATE INDEX IF NOT EXISTS idx_data_lineage_source_target ON metadata.data_lineage(source_table, target_table);

-- Create function for automatic audit logging
CREATE OR REPLACE FUNCTION metadata.audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO metadata.audit_log (table_name, operation, old_values, user_name, session_id)
        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(OLD), current_user, current_setting('application_name', true));
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO metadata.audit_log (table_name, operation, old_values, new_values, user_name, session_id)
        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(OLD), row_to_json(NEW), current_user, current_setting('application_name', true));
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO metadata.audit_log (table_name, operation, new_values, user_name, session_id)
        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(NEW), current_user, current_setting('application_name', true));
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Set up connection pooling parameters
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Reload configuration
SELECT pg_reload_conf();