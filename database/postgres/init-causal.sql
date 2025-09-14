-- Causal Analysis Database Initialization
-- Production-grade PostgreSQL setup with PostGIS for geological causal analysis

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create schemas for organized data management
CREATE SCHEMA IF NOT EXISTS causal_analysis;
CREATE SCHEMA IF NOT EXISTS geological_data;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set search path
SET search_path TO causal_analysis, geological_data, monitoring, public;

-- Create custom types
CREATE TYPE geological_process_type AS ENUM (
    'seismic', 'volcanic', 'deformation', 'thermal', 'geochemical',
    'hydrological', 'atmospheric', 'tectonic', 'magmatic', 'hydrothermal',
    'gravitational', 'magnetic', 'electromagnetic', 'unknown'
);

CREATE TYPE spatial_scale AS ENUM (
    'microscopic', 'mineral', 'rock', 'outcrop', 'local', 
    'regional', 'continental', 'global'
);

CREATE TYPE temporal_scale AS ENUM (
    'milliseconds', 'seconds', 'minutes', 'hours', 'days',
    'weeks', 'months', 'years', 'decades', 'centuries',
    'millennia', 'geological'
);

CREATE TYPE causal_discovery_method AS ENUM (
    'pc', 'ges', 'lingam', 'causal_forest', 'notears',
    'granger_causality', 'transfer_entropy', 'ccm', 'pcmci'
);

CREATE TYPE validation_status AS ENUM (
    'validated', 'pending', 'rejected', 'uncertain',
    'ensemble_validated', 'domain_validated'
);

CREATE TYPE data_quality_level AS ENUM (
    'excellent', 'good', 'fair', 'poor', 'unknown'
);

-- Create audit function for tracking changes
CREATE OR REPLACE FUNCTION audit.audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit.audit_log (
            table_name, operation, new_data, changed_by, changed_at
        ) VALUES (
            TG_TABLE_NAME, TG_OP, row_to_json(NEW), current_user, now()
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit.audit_log (
            table_name, operation, old_data, new_data, changed_by, changed_at
        ) VALUES (
            TG_TABLE_NAME, TG_OP, row_to_json(OLD), row_to_json(NEW), current_user, now()
        );
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit.audit_log (
            table_name, operation, old_data, changed_by, changed_at
        ) VALUES (
            TG_TABLE_NAME, TG_OP, row_to_json(OLD), current_user, now()
        );
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit.audit_log (
    id BIGSERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    old_data JSONB,
    new_data JSONB,
    changed_by TEXT NOT NULL DEFAULT current_user,
    changed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create indexes for audit log
CREATE INDEX IF NOT EXISTS idx_audit_log_table_name ON audit.audit_log(table_name);
CREATE INDEX IF NOT EXISTS idx_audit_log_changed_at ON audit.audit_log(changed_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_operation ON audit.audit_log(operation);

COMMENT ON SCHEMA causal_analysis IS 'Schema for causal analysis and graph storage';
COMMENT ON SCHEMA geological_data IS 'Schema for real geological data from various sources';
COMMENT ON SCHEMA monitoring IS 'Schema for system monitoring and performance metrics';
COMMENT ON SCHEMA audit IS 'Schema for audit logging and change tracking';