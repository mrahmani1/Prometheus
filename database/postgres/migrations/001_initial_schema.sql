-- =============================================================================
-- DATABASE MIGRATION 001: Initial Comprehensive Geological Schema
-- Version: 1.0.0
-- Description: Initial setup of comprehensive geological database schema
-- =============================================================================

-- Migration metadata
INSERT INTO schema_migrations (version, description, applied_at) VALUES 
    ('001', 'Initial comprehensive geological schema setup', NOW())
ON CONFLICT (version) DO NOTHING;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create specialized schemas
CREATE SCHEMA IF NOT EXISTS geological_core;
CREATE SCHEMA IF NOT EXISTS rock_mechanics;
CREATE SCHEMA IF NOT EXISTS volcanic_systems;
CREATE SCHEMA IF NOT EXISTS seismic_networks;
CREATE SCHEMA IF NOT EXISTS geodetic_data;
CREATE SCHEMA IF NOT EXISTS geophysical_data;
CREATE SCHEMA IF NOT EXISTS temporal_analysis;
CREATE SCHEMA IF NOT EXISTS spatial_indexing;
CREATE SCHEMA IF NOT EXISTS geological_age;
CREATE SCHEMA IF NOT EXISTS model_management;
CREATE SCHEMA IF NOT EXISTS mlops;
CREATE SCHEMA IF NOT EXISTS external_data;
CREATE SCHEMA IF NOT EXISTS backup_archive;
CREATE SCHEMA IF NOT EXISTS performance_monitoring;

-- Log migration completion
INSERT INTO migration_log (migration_version, status, executed_at, execution_time_ms) 
VALUES ('001', 'completed', NOW(), 0);