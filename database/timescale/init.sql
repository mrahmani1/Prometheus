-- TimescaleDB initialization script for Prometheus AI Platform
-- This script sets up the TimescaleDB database with optimized schemas for geological data

-- Create database if it doesn't exist
CREATE DATABASE prometheus_timeseries;

-- Connect to the database
\c prometheus_timeseries;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS postgis CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements CASCADE;
CREATE EXTENSION IF NOT EXISTS btree_gist CASCADE;

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS sensors;
CREATE SCHEMA IF NOT EXISTS seismic;
CREATE SCHEMA IF NOT EXISTS volcanic;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set search path
SET search_path TO public, sensors, seismic, volcanic, analytics;

-- Create custom types for geological data
CREATE TYPE sensor_quality AS ENUM ('good', 'questionable', 'bad', 'missing');
CREATE TYPE seismic_event_type AS ENUM ('earthquake', 'volcanic', 'explosion', 'landslide', 'unknown');
CREATE TYPE alert_level AS ENUM ('green', 'yellow', 'orange', 'red');

-- Create sensor readings hypertable
CREATE TABLE sensors.readings (
    time TIMESTAMPTZ NOT NULL,
    sensor_id TEXT NOT NULL,
    measurement_type TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit TEXT NOT NULL,
    quality_flag sensor_quality DEFAULT 'good',
    location GEOMETRY(POINT, 4326),
    elevation DOUBLE PRECISION,
    metadata JSONB,
    processing_flags TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable with 1-day chunks
SELECT create_hypertable('sensors.readings', 'time', chunk_time_interval => INTERVAL '1 day');

-- Create indexes for efficient queries
CREATE INDEX idx_readings_sensor_id_time ON sensors.readings (sensor_id, time DESC);
CREATE INDEX idx_readings_measurement_type_time ON sensors.readings (measurement_type, time DESC);
CREATE INDEX idx_readings_location ON sensors.readings USING GIST (location);
CREATE INDEX idx_readings_quality ON sensors.readings (quality_flag, time DESC);
CREATE INDEX idx_readings_metadata ON sensors.readings USING GIN (metadata);

-- Create seismic events hypertable
CREATE TABLE seismic.events (
    time TIMESTAMPTZ NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    magnitude DOUBLE PRECISION,
    magnitude_type TEXT,
    depth DOUBLE PRECISION,
    location GEOMETRY(POINT, 4326),
    event_type seismic_event_type DEFAULT 'earthquake',
    confidence DOUBLE PRECISION CHECK (confidence >= 0 AND confidence <= 1),
    waveform_data BYTEA,
    phase_picks JSONB,
    moment_tensor JSONB,
    focal_mechanism JSONB,
    network_stations TEXT[],
    processing_info JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable with 1-week chunks for seismic events
SELECT create_hypertable('seismic.events', 'time', chunk_time_interval => INTERVAL '1 week');

-- Create indexes for seismic events
CREATE INDEX idx_events_event_id ON seismic.events (event_id);
CREATE INDEX idx_events_magnitude_time ON seismic.events (magnitude DESC, time DESC);
CREATE INDEX idx_events_location ON seismic.events USING GIST (location);
CREATE INDEX idx_events_type_time ON seismic.events (event_type, time DESC);
CREATE INDEX idx_events_depth_time ON seismic.events (depth, time DESC);

-- Create volcanic monitoring hypertable
CREATE TABLE volcanic.monitoring (
    time TIMESTAMPTZ NOT NULL,
    volcano_id TEXT NOT NULL,
    alert_level alert_level DEFAULT 'green',
    deformation_rate DOUBLE PRECISION,
    gas_emission_rate DOUBLE PRECISION,
    gas_composition JSONB,
    thermal_anomaly DOUBLE PRECISION,
    seismic_activity_count INTEGER DEFAULT 0,
    tremor_amplitude DOUBLE PRECISION,
    infrasound_activity DOUBLE PRECISION,
    ground_temperature DOUBLE PRECISION,
    spring_temperature DOUBLE PRECISION,
    ph_level DOUBLE PRECISION,
    risk_assessment JSONB,
    automated_alerts JSONB,
    observatory_notes TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable with 1-day chunks for volcanic monitoring
SELECT create_hypertable('volcanic.monitoring', 'time', chunk_time_interval => INTERVAL '1 day');

-- Create indexes for volcanic monitoring
CREATE INDEX idx_monitoring_volcano_time ON volcanic.monitoring (volcano_id, time DESC);
CREATE INDEX idx_monitoring_alert_level ON volcanic.monitoring (alert_level, time DESC);
CREATE INDEX idx_monitoring_deformation ON volcanic.monitoring (deformation_rate, time DESC) WHERE deformation_rate IS NOT NULL;
CREATE INDEX idx_monitoring_gas_emission ON volcanic.monitoring (gas_emission_rate, time DESC) WHERE gas_emission_rate IS NOT NULL;
CREATE INDEX idx_monitoring_thermal ON volcanic.monitoring (thermal_anomaly, time DESC) WHERE thermal_anomaly IS NOT NULL;

-- Create geochemical samples table
CREATE TABLE sensors.geochemical_samples (
    time TIMESTAMPTZ NOT NULL,
    sample_id TEXT NOT NULL UNIQUE,
    location GEOMETRY(POINT, 4326),
    sample_type TEXT NOT NULL,
    temperature DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    ph_level DOUBLE PRECISION,
    composition JSONB NOT NULL,
    isotope_ratios JSONB,
    collection_method TEXT,
    laboratory TEXT,
    analysis_date TIMESTAMPTZ,
    quality_control JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for geochemical samples
SELECT create_hypertable('sensors.geochemical_samples', 'time', chunk_time_interval => INTERVAL '1 month');

-- Create indexes for geochemical samples
CREATE INDEX idx_geochemical_sample_id ON sensors.geochemical_samples (sample_id);
CREATE INDEX idx_geochemical_location ON sensors.geochemical_samples USING GIST (location);
CREATE INDEX idx_geochemical_type_time ON sensors.geochemical_samples (sample_type, time DESC);
CREATE INDEX idx_geochemical_composition ON sensors.geochemical_samples USING GIN (composition);

-- Set up compression policies
SELECT add_compression_policy('sensors.readings', INTERVAL '7 days');
SELECT add_compression_policy('seismic.events', INTERVAL '30 days');
SELECT add_compression_policy('volcanic.monitoring', INTERVAL '14 days');
SELECT add_compression_policy('sensors.geochemical_samples', INTERVAL '90 days');

-- Set up retention policies
SELECT add_retention_policy('sensors.readings', INTERVAL '5 years');
SELECT add_retention_policy('seismic.events', INTERVAL '10 years');
SELECT add_retention_policy('volcanic.monitoring', INTERVAL '7 years');
SELECT add_retention_policy('sensors.geochemical_samples', INTERVAL '20 years');

-- Create continuous aggregates for real-time monitoring
CREATE MATERIALIZED VIEW analytics.sensor_readings_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    sensor_id,
    measurement_type,
    AVG(value) AS avg_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    STDDEV(value) AS stddev_value,
    COUNT(*) AS reading_count,
    COUNT(*) FILTER (WHERE quality_flag = 'good') AS good_readings,
    COUNT(*) FILTER (WHERE quality_flag = 'questionable') AS questionable_readings,
    COUNT(*) FILTER (WHERE quality_flag = 'bad') AS bad_readings
FROM sensors.readings
GROUP BY bucket, sensor_id, measurement_type;

CREATE MATERIALIZED VIEW analytics.sensor_readings_daily
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS bucket,
    sensor_id,
    measurement_type,
    AVG(value) AS avg_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    STDDEV(value) AS stddev_value,
    COUNT(*) AS reading_count,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) AS median_value,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value) AS p95_value,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY value) AS p05_value
FROM sensors.readings
WHERE quality_flag = 'good'
GROUP BY bucket, sensor_id, measurement_type;

CREATE MATERIALIZED VIEW analytics.volcanic_activity_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    volcano_id,
    AVG(CASE WHEN alert_level = 'green' THEN 1 
             WHEN alert_level = 'yellow' THEN 2 
             WHEN alert_level = 'orange' THEN 3 
             WHEN alert_level = 'red' THEN 4 END) AS avg_alert_level,
    MAX(CASE WHEN alert_level = 'green' THEN 1 
             WHEN alert_level = 'yellow' THEN 2 
             WHEN alert_level = 'orange' THEN 3 
             WHEN alert_level = 'red' THEN 4 END) AS max_alert_level,
    AVG(deformation_rate) AS avg_deformation_rate,
    MAX(deformation_rate) AS max_deformation_rate,
    AVG(gas_emission_rate) AS avg_gas_emission_rate,
    MAX(gas_emission_rate) AS max_gas_emission_rate,
    AVG(thermal_anomaly) AS avg_thermal_anomaly,
    MAX(thermal_anomaly) AS max_thermal_anomaly,
    SUM(seismic_activity_count) AS total_seismic_events,
    AVG(tremor_amplitude) AS avg_tremor_amplitude,
    MAX(tremor_amplitude) AS max_tremor_amplitude
FROM volcanic.monitoring
GROUP BY bucket, volcano_id;

CREATE MATERIALIZED VIEW analytics.seismic_activity_daily
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS bucket,
    event_type,
    COUNT(*) AS event_count,
    AVG(magnitude) AS avg_magnitude,
    MAX(magnitude) AS max_magnitude,
    MIN(magnitude) AS min_magnitude,
    AVG(depth) AS avg_depth,
    MAX(depth) AS max_depth,
    MIN(depth) AS min_depth,
    AVG(confidence) AS avg_confidence
FROM seismic.events
WHERE magnitude IS NOT NULL
GROUP BY bucket, event_type;

-- Set up refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('analytics.sensor_readings_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('analytics.sensor_readings_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

SELECT add_continuous_aggregate_policy('analytics.volcanic_activity_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('analytics.seismic_activity_daily',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- Create functions for geological trend analysis
CREATE OR REPLACE FUNCTION analytics.calculate_trend_statistics(
    p_sensor_id TEXT,
    p_measurement_type TEXT,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE (
    slope DOUBLE PRECISION,
    intercept DOUBLE PRECISION,
    r_squared DOUBLE PRECISION,
    data_points BIGINT,
    mean_value DOUBLE PRECISION,
    stddev_value DOUBLE PRECISION,
    trend_significance TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH trend_data AS (
        SELECT 
            time,
            value,
            EXTRACT(EPOCH FROM time) AS time_epoch
        FROM sensors.readings
        WHERE sensor_id = p_sensor_id 
            AND measurement_type = p_measurement_type
            AND time >= p_start_time 
            AND time <= p_end_time
            AND quality_flag = 'good'
        ORDER BY time
    ),
    regression AS (
        SELECT 
            regr_slope(value, time_epoch) AS slope,
            regr_intercept(value, time_epoch) AS intercept,
            regr_r2(value, time_epoch) AS r_squared,
            COUNT(*) AS data_points,
            AVG(value) AS mean_value,
            STDDEV(value) AS stddev_value
        FROM trend_data
    )
    SELECT 
        r.slope,
        r.intercept,
        r.r_squared,
        r.data_points,
        r.mean_value,
        r.stddev_value,
        CASE 
            WHEN r.r_squared > 0.7 THEN 'strong'
            WHEN r.r_squared > 0.3 THEN 'moderate'
            ELSE 'weak'
        END AS trend_significance
    FROM regression r;
END;
$$ LANGUAGE plpgsql;

-- Create function for anomaly detection
CREATE OR REPLACE FUNCTION analytics.detect_anomalies(
    p_sensor_id TEXT,
    p_measurement_type TEXT,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ,
    p_z_threshold DOUBLE PRECISION DEFAULT 3.0
)
RETURNS TABLE (
    time TIMESTAMPTZ,
    value DOUBLE PRECISION,
    z_score DOUBLE PRECISION,
    is_anomaly BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH windowed_data AS (
        SELECT 
            r.time,
            r.value,
            AVG(r.value) OVER (
                ORDER BY r.time 
                ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
            ) AS moving_avg,
            STDDEV(r.value) OVER (
                ORDER BY r.time 
                ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
            ) AS moving_stddev
        FROM sensors.readings r
        WHERE r.sensor_id = p_sensor_id 
            AND r.measurement_type = p_measurement_type
            AND r.time >= p_start_time 
            AND r.time <= p_end_time
            AND r.quality_flag = 'good'
        ORDER BY r.time
    )
    SELECT 
        wd.time,
        wd.value,
        ABS(wd.value - wd.moving_avg) / NULLIF(wd.moving_stddev, 0) AS z_score,
        (ABS(wd.value - wd.moving_avg) / NULLIF(wd.moving_stddev, 0)) > p_z_threshold AS is_anomaly
    FROM windowed_data wd
    WHERE wd.moving_stddev IS NOT NULL AND wd.moving_stddev > 0;
END;
$$ LANGUAGE plpgsql;

-- Create function for volcanic risk assessment
CREATE OR REPLACE FUNCTION analytics.assess_volcanic_risk(
    p_volcano_id TEXT,
    p_assessment_time TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE (
    risk_level TEXT,
    risk_score DOUBLE PRECISION,
    contributing_factors JSONB,
    recommendations TEXT[]
) AS $$
DECLARE
    recent_deformation DOUBLE PRECISION;
    recent_gas_emission DOUBLE PRECISION;
    recent_seismic_count INTEGER;
    recent_thermal_anomaly DOUBLE PRECISION;
    alert_level_numeric INTEGER;
    risk_score_calc DOUBLE PRECISION := 0;
    risk_level_result TEXT := 'low';
    factors JSONB := '{}';
    recommendations_result TEXT[] := ARRAY[]::TEXT[];
BEGIN
    -- Get recent volcanic monitoring data (last 24 hours)
    SELECT 
        AVG(deformation_rate),
        AVG(gas_emission_rate),
        SUM(seismic_activity_count),
        AVG(thermal_anomaly),
        AVG(CASE WHEN alert_level = 'green' THEN 1 
                 WHEN alert_level = 'yellow' THEN 2 
                 WHEN alert_level = 'orange' THEN 3 
                 WHEN alert_level = 'red' THEN 4 END)
    INTO recent_deformation, recent_gas_emission, recent_seismic_count, 
         recent_thermal_anomaly, alert_level_numeric
    FROM volcanic.monitoring
    WHERE volcano_id = p_volcano_id 
        AND time >= p_assessment_time - INTERVAL '24 hours'
        AND time <= p_assessment_time;
    
    -- Calculate risk score based on multiple factors
    IF recent_deformation IS NOT NULL AND recent_deformation > 0.001 THEN
        risk_score_calc := risk_score_calc + (recent_deformation * 1000);
        factors := factors || jsonb_build_object('deformation_rate', recent_deformation);
    END IF;
    
    IF recent_gas_emission IS NOT NULL AND recent_gas_emission > 100 THEN
        risk_score_calc := risk_score_calc + (recent_gas_emission / 100);
        factors := factors || jsonb_build_object('gas_emission_rate', recent_gas_emission);
    END IF;
    
    IF recent_seismic_count IS NOT NULL AND recent_seismic_count > 10 THEN
        risk_score_calc := risk_score_calc + (recent_seismic_count / 10);
        factors := factors || jsonb_build_object('seismic_activity_count', recent_seismic_count);
    END IF;
    
    IF recent_thermal_anomaly IS NOT NULL AND recent_thermal_anomaly > 5 THEN
        risk_score_calc := risk_score_calc + recent_thermal_anomaly;
        factors := factors || jsonb_build_object('thermal_anomaly', recent_thermal_anomaly);
    END IF;
    
    IF alert_level_numeric IS NOT NULL THEN
        risk_score_calc := risk_score_calc + (alert_level_numeric * 2);
        factors := factors || jsonb_build_object('alert_level', alert_level_numeric);
    END IF;
    
    -- Determine risk level and recommendations
    IF risk_score_calc > 20 THEN
        risk_level_result := 'critical';
        recommendations_result := ARRAY['Immediate evacuation planning', 'Continuous monitoring', 'Emergency response activation'];
    ELSIF risk_score_calc > 10 THEN
        risk_level_result := 'high';
        recommendations_result := ARRAY['Enhanced monitoring', 'Prepare evacuation plans', 'Increase observation frequency'];
    ELSIF risk_score_calc > 5 THEN
        risk_level_result := 'moderate';
        recommendations_result := ARRAY['Increased monitoring', 'Review emergency procedures', 'Monitor trends closely'];
    ELSE
        risk_level_result := 'low';
        recommendations_result := ARRAY['Continue routine monitoring', 'Maintain baseline observations'];
    END IF;
    
    RETURN QUERY SELECT risk_level_result, risk_score_calc, factors, recommendations_result;
END;
$$ LANGUAGE plpgsql;

-- Create indexes on continuous aggregates
CREATE INDEX idx_sensor_readings_hourly_bucket ON analytics.sensor_readings_hourly (bucket DESC);
CREATE INDEX idx_sensor_readings_daily_bucket ON analytics.sensor_readings_daily (bucket DESC);
CREATE INDEX idx_volcanic_activity_hourly_bucket ON analytics.volcanic_activity_hourly (bucket DESC);
CREATE INDEX idx_seismic_activity_daily_bucket ON analytics.seismic_activity_daily (bucket DESC);

-- Grant permissions
GRANT USAGE ON SCHEMA sensors, seismic, volcanic, analytics TO prometheus;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA sensors, seismic, volcanic, analytics TO prometheus;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA analytics TO prometheus;

-- Create sample data for testing (optional)
-- INSERT INTO sensors.readings (time, sensor_id, measurement_type, value, unit, location)
-- VALUES 
--     (NOW() - INTERVAL '1 hour', 'TEMP_001', 'temperature', 25.5, 'celsius', ST_GeomFromText('POINT(-122.4194 37.7749)', 4326)),
--     (NOW() - INTERVAL '2 hours', 'TEMP_001', 'temperature', 26.1, 'celsius', ST_GeomFromText('POINT(-122.4194 37.7749)', 4326)),
--     (NOW() - INTERVAL '3 hours', 'TEMP_001', 'temperature', 24.8, 'celsius', ST_GeomFromText('POINT(-122.4194 37.7749)', 4326));

COMMIT;