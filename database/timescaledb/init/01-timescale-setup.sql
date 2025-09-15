-- TimescaleDB initialization script for Prometheus AI Platform
-- This script sets up TimescaleDB for high-performance time-series geological data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schemas for time-series data
CREATE SCHEMA IF NOT EXISTS real_time_monitoring;
CREATE SCHEMA IF NOT EXISTS historical_data;
CREATE SCHEMA IF NOT EXISTS aggregated_metrics;

-- Grant permissions
GRANT USAGE ON SCHEMA real_time_monitoring TO timescale;
GRANT USAGE ON SCHEMA historical_data TO timescale;
GRANT USAGE ON SCHEMA aggregated_metrics TO timescale;

GRANT CREATE ON SCHEMA real_time_monitoring TO timescale;
GRANT CREATE ON SCHEMA historical_data TO timescale;
GRANT CREATE ON SCHEMA aggregated_metrics TO timescale;

-- Real-time seismic monitoring table
CREATE TABLE IF NOT EXISTS real_time_monitoring.seismic_streams (
    time TIMESTAMPTZ NOT NULL,
    station_id VARCHAR(20) NOT NULL,
    network_code VARCHAR(10) NOT NULL,
    channel VARCHAR(10) NOT NULL,
    sample_rate FLOAT NOT NULL,
    amplitude FLOAT NOT NULL,
    frequency_hz FLOAT,
    signal_to_noise FLOAT,
    location GEOMETRY(POINT, 4326),
    data_quality VARCHAR(10),
    processing_flags INTEGER,
    metadata JSONB
);

-- Convert to hypertable
SELECT create_hypertable('real_time_monitoring.seismic_streams', 'time', 
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Real-time deformation monitoring table
CREATE TABLE IF NOT EXISTS real_time_monitoring.deformation_streams (
    time TIMESTAMPTZ NOT NULL,
    station_id VARCHAR(20) NOT NULL,
    measurement_type VARCHAR(20) NOT NULL, -- GPS, InSAR, Tilt
    displacement_east_mm FLOAT,
    displacement_north_mm FLOAT,
    displacement_up_mm FLOAT,
    velocity_east_mm_day FLOAT,
    velocity_north_mm_day FLOAT,
    velocity_up_mm_day FLOAT,
    uncertainty_mm FLOAT,
    location GEOMETRY(POINT, 4326),
    data_quality VARCHAR(10),
    processing_flags INTEGER,
    metadata JSONB
);

-- Convert to hypertable
SELECT create_hypertable('real_time_monitoring.deformation_streams', 'time',
    chunk_time_interval => INTERVAL '6 hours',
    if_not_exists => TRUE);

-- Real-time gas monitoring table
CREATE TABLE IF NOT EXISTS real_time_monitoring.gas_streams (
    time TIMESTAMPTZ NOT NULL,
    station_id VARCHAR(20) NOT NULL,
    co2_ppm FLOAT,
    so2_ppm FLOAT,
    h2s_ppm FLOAT,
    temperature_celsius FLOAT,
    humidity_percent FLOAT,
    pressure_hpa FLOAT,
    wind_speed_ms FLOAT,
    wind_direction_degrees FLOAT,
    flux_estimate_tons_day FLOAT,
    location GEOMETRY(POINT, 4326),
    data_quality VARCHAR(10),
    instrument_status INTEGER,
    metadata JSONB
);

-- Convert to hypertable
SELECT create_hypertable('real_time_monitoring.gas_streams', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Real-time thermal monitoring table
CREATE TABLE IF NOT EXISTS real_time_monitoring.thermal_streams (
    time TIMESTAMPTZ NOT NULL,
    pixel_id VARCHAR(50) NOT NULL,
    satellite_platform VARCHAR(20),
    sensor_type VARCHAR(20),
    temperature_celsius FLOAT,
    radiant_flux_mw FLOAT,
    thermal_anomaly BOOLEAN,
    cloud_cover_percent FLOAT,
    location GEOMETRY(POINT, 4326),
    pixel_size_m FLOAT,
    data_quality VARCHAR(10),
    processing_level VARCHAR(10),
    metadata JSONB
);

-- Convert to hypertable
SELECT create_hypertable('real_time_monitoring.thermal_streams', 'time',
    chunk_time_interval => INTERVAL '12 hours',
    if_not_exists => TRUE);

-- Causal relationship evolution table
CREATE TABLE IF NOT EXISTS real_time_monitoring.causal_evolution (
    time TIMESTAMPTZ NOT NULL,
    relationship_id UUID NOT NULL,
    source_variable VARCHAR(255) NOT NULL,
    target_variable VARCHAR(255) NOT NULL,
    causal_strength FLOAT NOT NULL,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    temporal_lag INTERVAL,
    statistical_significance FLOAT,
    data_window_hours INTEGER,
    discovery_method VARCHAR(100),
    validation_score FLOAT,
    metadata JSONB
);

-- Convert to hypertable
SELECT create_hypertable('real_time_monitoring.causal_evolution', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Historical earthquake catalog
CREATE TABLE IF NOT EXISTS historical_data.earthquake_catalog (
    time TIMESTAMPTZ NOT NULL,
    event_id VARCHAR(50) NOT NULL,
    magnitude FLOAT NOT NULL,
    depth_km FLOAT NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    magnitude_type VARCHAR(10),
    agency VARCHAR(50),
    region VARCHAR(100),
    tsunami_flag BOOLEAN,
    felt_reports INTEGER,
    damage_level VARCHAR(20),
    fatalities INTEGER,
    economic_loss_usd BIGINT,
    data_source VARCHAR(100),
    metadata JSONB
);

-- Convert to hypertable with longer chunk intervals for historical data
SELECT create_hypertable('historical_data.earthquake_catalog', 'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);

-- Aggregated daily metrics table
CREATE TABLE IF NOT EXISTS aggregated_metrics.daily_volcanic_metrics (
    time DATE NOT NULL,
    volcano_id VARCHAR(50) NOT NULL,
    seismic_events_count INTEGER DEFAULT 0,
    max_magnitude FLOAT,
    avg_magnitude FLOAT,
    total_seismic_energy FLOAT,
    deformation_rate_mm_day FLOAT,
    max_temperature_celsius FLOAT,
    avg_gas_flux_tons_day FLOAT,
    thermal_anomaly_area_km2 FLOAT,
    alert_level VARCHAR(20),
    activity_index FLOAT,
    data_completeness_percent FLOAT,
    metadata JSONB
);

-- Convert to hypertable
SELECT create_hypertable('aggregated_metrics.daily_volcanic_metrics', 'time',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_seismic_streams_station_time ON real_time_monitoring.seismic_streams (station_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_deformation_streams_station_time ON real_time_monitoring.deformation_streams (station_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_gas_streams_station_time ON real_time_monitoring.gas_streams (station_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_thermal_streams_pixel_time ON real_time_monitoring.thermal_streams (pixel_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_causal_evolution_relationship_time ON real_time_monitoring.causal_evolution (relationship_id, time DESC);

-- Create spatial indexes
CREATE INDEX IF NOT EXISTS idx_seismic_streams_location ON real_time_monitoring.seismic_streams USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_deformation_streams_location ON real_time_monitoring.deformation_streams USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_gas_streams_location ON real_time_monitoring.gas_streams USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_thermal_streams_location ON real_time_monitoring.thermal_streams USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_earthquake_catalog_location ON historical_data.earthquake_catalog USING GIST (location);

-- Create compression policies for older data
SELECT add_compression_policy('real_time_monitoring.seismic_streams', INTERVAL '7 days');
SELECT add_compression_policy('real_time_monitoring.deformation_streams', INTERVAL '30 days');
SELECT add_compression_policy('real_time_monitoring.gas_streams', INTERVAL '7 days');
SELECT add_compression_policy('real_time_monitoring.thermal_streams', INTERVAL '30 days');
SELECT add_compression_policy('real_time_monitoring.causal_evolution', INTERVAL '90 days');
SELECT add_compression_policy('historical_data.earthquake_catalog', INTERVAL '1 year');
SELECT add_compression_policy('aggregated_metrics.daily_volcanic_metrics', INTERVAL '2 years');

-- Create retention policies for data lifecycle management
SELECT add_retention_policy('real_time_monitoring.seismic_streams', INTERVAL '2 years');
SELECT add_retention_policy('real_time_monitoring.deformation_streams', INTERVAL '5 years');
SELECT add_retention_policy('real_time_monitoring.gas_streams', INTERVAL '2 years');
SELECT add_retention_policy('real_time_monitoring.thermal_streams', INTERVAL '3 years');
SELECT add_retention_policy('real_time_monitoring.causal_evolution', INTERVAL '10 years');

-- Create continuous aggregates for real-time dashboards
CREATE MATERIALIZED VIEW IF NOT EXISTS aggregated_metrics.hourly_seismic_summary
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    station_id,
    network_code,
    COUNT(*) as sample_count,
    AVG(amplitude) as avg_amplitude,
    MAX(amplitude) as max_amplitude,
    AVG(frequency_hz) as avg_frequency,
    AVG(signal_to_noise) as avg_snr
FROM real_time_monitoring.seismic_streams
GROUP BY hour, station_id, network_code;

CREATE MATERIALIZED VIEW IF NOT EXISTS aggregated_metrics.daily_deformation_summary
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS day,
    station_id,
    measurement_type,
    AVG(displacement_east_mm) as avg_displacement_east,
    AVG(displacement_north_mm) as avg_displacement_north,
    AVG(displacement_up_mm) as avg_displacement_up,
    AVG(velocity_east_mm_day) as avg_velocity_east,
    AVG(velocity_north_mm_day) as avg_velocity_north,
    AVG(velocity_up_mm_day) as avg_velocity_up
FROM real_time_monitoring.deformation_streams
GROUP BY day, station_id, measurement_type;

-- Add refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('aggregated_metrics.hourly_seismic_summary',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('aggregated_metrics.daily_deformation_summary',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA real_time_monitoring TO timescale;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA historical_data TO timescale;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA aggregated_metrics TO timescale;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA real_time_monitoring TO timescale;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA historical_data TO timescale;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA aggregated_metrics TO timescale;