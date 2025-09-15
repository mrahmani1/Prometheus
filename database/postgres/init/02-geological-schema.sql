-- Geological data schema for Prometheus AI Platform
-- This script creates the core geological data structures

-- Volcanic systems table
CREATE TABLE IF NOT EXISTS geological_data.volcanic_systems (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    volcano_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation FLOAT,
    volcano_type VARCHAR(100),
    tectonic_setting VARCHAR(100),
    last_eruption_date DATE,
    vei_max INTEGER CHECK (vei_max >= 0 AND vei_max <= 8),
    hazard_level VARCHAR(50),
    monitoring_status VARCHAR(50),
    population_exposure INTEGER,
    country VARCHAR(100),
    region VARCHAR(100),
    data_source VARCHAR(255),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Seismic events table
CREATE TABLE IF NOT EXISTS seismic_data.earthquake_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(50) UNIQUE NOT NULL,
    origin_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    depth_km FLOAT NOT NULL,
    magnitude FLOAT NOT NULL,
    magnitude_type VARCHAR(10),
    agency VARCHAR(50),
    event_type VARCHAR(50),
    quality_grade VARCHAR(10),
    uncertainty_km FLOAT,
    gap_degrees FLOAT,
    stations_used INTEGER,
    data_source VARCHAR(255),
    processing_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    waveform_data JSONB
);

-- Seismic stations table
CREATE TABLE IF NOT EXISTS monitoring_networks.seismic_stations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    station_code VARCHAR(20) UNIQUE NOT NULL,
    network_code VARCHAR(10) NOT NULL,
    station_name VARCHAR(255),
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation_m FLOAT,
    installation_date DATE,
    decommission_date DATE,
    instrument_type VARCHAR(100),
    sampling_rate_hz FLOAT,
    data_availability JSONB,
    contact_info JSONB,
    metadata JSONB
);

-- Geochemical samples table
CREATE TABLE IF NOT EXISTS geochemical_data.rock_samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sample_id VARCHAR(50) UNIQUE NOT NULL,
    collection_date DATE,
    location GEOMETRY(POINT, 4326) NOT NULL,
    rock_type VARCHAR(100),
    formation VARCHAR(255),
    age_ma FLOAT,
    collector VARCHAR(255),
    laboratory VARCHAR(255),
    analysis_date DATE,
    major_elements JSONB,
    trace_elements JSONB,
    isotope_ratios JSONB,
    analytical_methods JSONB,
    data_source VARCHAR(255),
    quality_flags TEXT[],
    metadata JSONB
);

-- Causal relationships table
CREATE TABLE IF NOT EXISTS causal_relationships.geological_causality (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_variable VARCHAR(255) NOT NULL,
    target_variable VARCHAR(255) NOT NULL,
    causal_strength FLOAT NOT NULL CHECK (causal_strength >= -1 AND causal_strength <= 1),
    confidence_lower FLOAT NOT NULL,
    confidence_upper FLOAT NOT NULL,
    temporal_lag INTERVAL,
    spatial_scale VARCHAR(100),
    statistical_significance FLOAT,
    discovery_method VARCHAR(100),
    validation_status VARCHAR(50),
    geological_context JSONB,
    data_sources TEXT[],
    discovery_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    validation_timestamp TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- Deformation monitoring table (MT-SAR and GPS)
CREATE TABLE IF NOT EXISTS geological_data.deformation_measurements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    measurement_id VARCHAR(50) UNIQUE NOT NULL,
    measurement_type VARCHAR(50) NOT NULL, -- 'GPS', 'InSAR', 'Tilt', etc.
    station_id VARCHAR(50),
    measurement_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    displacement_east_mm FLOAT,
    displacement_north_mm FLOAT,
    displacement_up_mm FLOAT,
    uncertainty_east_mm FLOAT,
    uncertainty_north_mm FLOAT,
    uncertainty_up_mm FLOAT,
    coherence FLOAT, -- For InSAR data
    satellite_platform VARCHAR(50), -- For InSAR data
    orbit_direction VARCHAR(20), -- For InSAR data
    baseline_days INTEGER, -- For InSAR data
    processing_parameters JSONB,
    data_source VARCHAR(255),
    quality_flags TEXT[],
    metadata JSONB
);

-- Gas emission monitoring table
CREATE TABLE IF NOT EXISTS geological_data.gas_emissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    measurement_id VARCHAR(50) UNIQUE NOT NULL,
    station_id VARCHAR(50),
    measurement_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    co2_ppm FLOAT,
    so2_ppm FLOAT,
    h2s_ppm FLOAT,
    temperature_celsius FLOAT,
    flux_tons_per_day FLOAT,
    wind_speed_ms FLOAT,
    wind_direction_degrees FLOAT,
    measurement_method VARCHAR(100),
    instrument_type VARCHAR(100),
    data_source VARCHAR(255),
    quality_flags TEXT[],
    metadata JSONB
);

-- Thermal monitoring table
CREATE TABLE IF NOT EXISTS geological_data.thermal_observations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    observation_id VARCHAR(50) UNIQUE NOT NULL,
    observation_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    temperature_celsius FLOAT,
    thermal_anomaly_area_m2 FLOAT,
    radiant_flux_mw FLOAT,
    satellite_platform VARCHAR(50),
    sensor_type VARCHAR(50),
    pixel_size_m FLOAT,
    cloud_cover_percent FLOAT,
    atmospheric_correction BOOLEAN,
    data_source VARCHAR(255),
    processing_level VARCHAR(20),
    quality_flags TEXT[],
    metadata JSONB
);

-- Create spatial indexes for performance
CREATE INDEX IF NOT EXISTS idx_volcanic_systems_location ON geological_data.volcanic_systems USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_earthquake_events_location ON seismic_data.earthquake_events USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_seismic_stations_location ON monitoring_networks.seismic_stations USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_rock_samples_location ON geochemical_data.rock_samples USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_deformation_location ON geological_data.deformation_measurements USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_gas_emissions_location ON geological_data.gas_emissions USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_thermal_observations_location ON geological_data.thermal_observations USING GIST (location);

-- Create temporal indexes
CREATE INDEX IF NOT EXISTS idx_earthquake_events_time ON seismic_data.earthquake_events (origin_time);
CREATE INDEX IF NOT EXISTS idx_deformation_time ON geological_data.deformation_measurements (measurement_time);
CREATE INDEX IF NOT EXISTS idx_gas_emissions_time ON geological_data.gas_emissions (measurement_time);
CREATE INDEX IF NOT EXISTS idx_thermal_observations_time ON geological_data.thermal_observations (observation_time);

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_earthquake_events_mag_time ON seismic_data.earthquake_events (magnitude DESC, origin_time DESC);
CREATE INDEX IF NOT EXISTS idx_causal_relationships_strength ON causal_relationships.geological_causality (causal_strength DESC);
CREATE INDEX IF NOT EXISTS idx_causal_relationships_variables ON causal_relationships.geological_causality (source_variable, target_variable);

-- Create GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_volcanic_systems_metadata ON geological_data.volcanic_systems USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_earthquake_events_waveform ON seismic_data.earthquake_events USING GIN (waveform_data);
CREATE INDEX IF NOT EXISTS idx_rock_samples_elements ON geochemical_data.rock_samples USING GIN (major_elements, trace_elements);
CREATE INDEX IF NOT EXISTS idx_causal_relationships_context ON causal_relationships.geological_causality USING GIN (geological_context);

-- Set up audit triggers for all tables
CREATE TRIGGER audit_volcanic_systems
    AFTER INSERT OR UPDATE OR DELETE ON geological_data.volcanic_systems
    FOR EACH ROW EXECUTE FUNCTION metadata.audit_trigger_function();

CREATE TRIGGER audit_earthquake_events
    AFTER INSERT OR UPDATE OR DELETE ON seismic_data.earthquake_events
    FOR EACH ROW EXECUTE FUNCTION metadata.audit_trigger_function();

CREATE TRIGGER audit_seismic_stations
    AFTER INSERT OR UPDATE OR DELETE ON monitoring_networks.seismic_stations
    FOR EACH ROW EXECUTE FUNCTION metadata.audit_trigger_function();

CREATE TRIGGER audit_rock_samples
    AFTER INSERT OR UPDATE OR DELETE ON geochemical_data.rock_samples
    FOR EACH ROW EXECUTE FUNCTION metadata.audit_trigger_function();

CREATE TRIGGER audit_geological_causality
    AFTER INSERT OR UPDATE OR DELETE ON causal_relationships.geological_causality
    FOR EACH ROW EXECUTE FUNCTION metadata.audit_trigger_function();

-- Grant appropriate permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA geological_data TO prometheus;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA seismic_data TO prometheus;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA volcanic_data TO prometheus;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA geochemical_data TO prometheus;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA monitoring_networks TO prometheus;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA causal_relationships TO prometheus;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA metadata TO prometheus;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA geological_data TO prometheus;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA seismic_data TO prometheus;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA volcanic_data TO prometheus;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA geochemical_data TO prometheus;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA monitoring_networks TO prometheus;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA causal_relationships TO prometheus;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA metadata TO prometheus;