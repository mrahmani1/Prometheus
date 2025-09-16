-- Advanced Eruption Trigger Analysis Database Schema
-- Comprehensive schema for multi-chamber volcanic system analysis

-- Enable PostGIS for geospatial operations
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Pressure threshold analysis tables
CREATE TABLE pressure_thresholds (
    id SERIAL PRIMARY KEY,
    chamber_id VARCHAR(50) NOT NULL,
    critical_pressure DOUBLE PRECISION NOT NULL,
    pressure_gradient DOUBLE PRECISION NOT NULL,
    threshold_uncertainty DOUBLE PRECISION NOT NULL,
    time_to_failure DOUBLE PRECISION NOT NULL,
    failure_mechanism VARCHAR(50) NOT NULL,
    depth DOUBLE PRECISION NOT NULL,
    stress_tensor DOUBLE PRECISION[6], -- [σxx, σyy, σzz, σxy, σxz, σyz]
    principal_stresses DOUBLE PRECISION[3], -- [σ1, σ2, σ3]
    failure_probability DOUBLE PRECISION,
    subcritical_crack_parameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create hypertable for time-series pressure data
SELECT create_hypertable('pressure_thresholds', 'created_at');

-- Volatile oversaturation analysis tables
CREATE TABLE volatile_oversaturation (
    id SERIAL PRIMARY KEY,
    chamber_id VARCHAR(50) NOT NULL,
    co2_saturation DOUBLE PRECISION NOT NULL,
    h2o_saturation DOUBLE PRECISION NOT NULL,
    so2_saturation DOUBLE PRECISION NOT NULL,
    bubble_nucleation_rate DOUBLE PRECISION NOT NULL,
    bubble_growth_rate DOUBLE PRECISION NOT NULL,
    coalescence_efficiency DOUBLE PRECISION NOT NULL,
    gas_volume_fraction DOUBLE PRECISION NOT NULL,
    degassing_rate DOUBLE PRECISION NOT NULL,
    solubility_models JSONB, -- Store model parameters and results
    thermodynamic_state JSONB, -- P-T-X conditions
    bubble_size_distribution DOUBLE PRECISION[], -- Size distribution parameters
    degassing_pathways JSONB, -- Gas migration pathway analysis
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

SELECT create_hypertable('volatile_oversaturation', 'created_at');

-- External trigger analysis tables
CREATE TABLE external_triggers (
    id SERIAL PRIMARY KEY,
    trigger_type VARCHAR(50) NOT NULL,
    magnitude DOUBLE PRECISION NOT NULL,
    location GEOMETRY(POINTZ, 4326) NOT NULL, -- 3D point with SRID 4326
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    stress_change DOUBLE PRECISION NOT NULL,
    strain_rate DOUBLE PRECISION NOT NULL,
    influence_radius DOUBLE PRECISION NOT NULL,
    coupling_strength DOUBLE PRECISION NOT NULL,
    coulomb_stress_change DOUBLE PRECISION,
    dynamic_stress DOUBLE PRECISION,
    tidal_stress_components DOUBLE PRECISION[6],
    seismic_parameters JSONB, -- Earthquake-specific parameters
    tidal_parameters JSONB, -- Tidal force parameters
    atmospheric_parameters JSONB, -- Atmospheric loading parameters
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

SELECT create_hypertable('external_triggers', 'timestamp');

-- Create spatial index for trigger locations
CREATE INDEX idx_external_triggers_location ON external_triggers USING GIST (location);
CREATE INDEX idx_external_triggers_type_time ON external_triggers (trigger_type, timestamp);

-- Cascading failure analysis tables
CREATE TABLE cascading_failures (
    id SERIAL PRIMARY KEY,
    initial_trigger_id INTEGER REFERENCES external_triggers(id),
    failure_sequence VARCHAR(50)[] NOT NULL,
    failure_times DOUBLE PRECISION[] NOT NULL,
    stress_transfer_matrix DOUBLE PRECISION[][] NOT NULL,
    failure_probabilities DOUBLE PRECISION[] NOT NULL,
    total_system_failure_time DOUBLE PRECISION NOT NULL,
    critical_path VARCHAR(50)[] NOT NULL,
    network_connectivity JSONB, -- Chamber network connectivity data
    failure_mechanisms JSONB, -- Detailed failure mechanism analysis
    propagation_velocities DOUBLE PRECISION[], -- Failure propagation speeds
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Eruption forecasting tables
CREATE TABLE eruption_forecasts (
    id SERIAL PRIMARY KEY,
    chamber_id VARCHAR(50) NOT NULL,
    eruption_probability DOUBLE PRECISION NOT NULL,
    time_window_start DOUBLE PRECISION NOT NULL,
    time_window_end DOUBLE PRECISION NOT NULL,
    confidence_interval_lower DOUBLE PRECISION NOT NULL,
    confidence_interval_upper DOUBLE PRECISION NOT NULL,
    magnitude_estimate DOUBLE PRECISION NOT NULL,
    style_prediction VARCHAR(100) NOT NULL,
    uncertainty_bounds JSONB NOT NULL,
    forecast_skill DOUBLE PRECISION NOT NULL,
    forecast_method VARCHAR(50) NOT NULL, -- bayesian, monte_carlo, ml, physics
    model_parameters JSONB, -- Method-specific parameters
    validation_metrics JSONB, -- Forecast validation results
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    forecast_horizon DOUBLE PRECISION NOT NULL,
    issued_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

SELECT create_hypertable('eruption_forecasts', 'issued_at');

-- Eruption characteristics prediction tables
CREATE TABLE eruption_characteristics (
    id SERIAL PRIMARY KEY,
    chamber_id VARCHAR(50) NOT NULL,
    forecast_id INTEGER REFERENCES eruption_forecasts(id),
    vei_estimate DOUBLE PRECISION NOT NULL,
    discharge_rate DOUBLE PRECISION NOT NULL,
    eruption_duration DOUBLE PRECISION NOT NULL,
    explosivity_index DOUBLE PRECISION NOT NULL,
    column_height DOUBLE PRECISION NOT NULL,
    pyroclastic_flow_potential DOUBLE PRECISION NOT NULL,
    lava_flow_potential DOUBLE PRECISION NOT NULL,
    ash_dispersal_range DOUBLE PRECISION NOT NULL,
    volume_estimates JSONB, -- Different volume calculation methods
    energy_estimates JSONB, -- Energy-based calculations
    hazard_zones GEOMETRY(MULTIPOLYGON, 4326), -- Spatial hazard zones
    dispersion_modeling JSONB, -- Ash and gas dispersion results
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chamber network definition tables
CREATE TABLE volcanic_chambers (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    location GEOMETRY(POINTZ, 4326) NOT NULL,
    radius DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    depth DOUBLE PRECISION NOT NULL,
    pressure DOUBLE PRECISION,
    temperature DOUBLE PRECISION,
    composition JSONB, -- Chemical composition
    volatile_content JSONB, -- H2O, CO2, SO2, etc.
    physical_properties JSONB, -- Density, viscosity, etc.
    geometry_parameters JSONB, -- Detailed 3D geometry
    monitoring_status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_chambers_location ON volcanic_chambers USING GIST (location);

-- Chamber connectivity and relationships
CREATE TABLE chamber_connections (
    id SERIAL PRIMARY KEY,
    source_chamber_id VARCHAR(50) REFERENCES volcanic_chambers(id),
    target_chamber_id VARCHAR(50) REFERENCES volcanic_chambers(id),
    connection_type VARCHAR(50) NOT NULL, -- conduit, fracture, dike
    connectivity_strength DOUBLE PRECISION NOT NULL,
    distance DOUBLE PRECISION NOT NULL,
    pathway_geometry GEOMETRY(LINESTRING, 4326),
    flow_parameters JSONB, -- Permeability, flow rates, etc.
    structural_controls JSONB, -- Geological controls on connectivity
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Stress field analysis tables
CREATE TABLE stress_field_analysis (
    id SERIAL PRIMARY KEY,
    chamber_id VARCHAR(50) REFERENCES volcanic_chambers(id),
    analysis_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    stress_tensor DOUBLE PRECISION[6] NOT NULL, -- Full stress tensor
    principal_stresses DOUBLE PRECISION[3] NOT NULL,
    principal_directions DOUBLE PRECISION[9], -- 3x3 direction matrix
    tectonic_regime VARCHAR(50), -- normal, reverse, strike-slip
    stress_sources JSONB, -- Regional, local, chamber-induced
    uncertainty_analysis JSONB, -- Stress field uncertainties
    inversion_parameters JSONB, -- Stress inversion method parameters
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

SELECT create_hypertable('stress_field_analysis', 'analysis_timestamp');

-- Seismic data integration tables
CREATE TABLE seismic_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(100) UNIQUE NOT NULL,
    magnitude DOUBLE PRECISION NOT NULL,
    location GEOMETRY(POINTZ, 4326) NOT NULL,
    depth DOUBLE PRECISION NOT NULL,
    origin_time TIMESTAMP WITH TIME ZONE NOT NULL,
    focal_mechanism JSONB, -- Strike, dip, rake
    moment_tensor DOUBLE PRECISION[6], -- Moment tensor components
    stress_drop DOUBLE PRECISION,
    corner_frequency DOUBLE PRECISION,
    source_parameters JSONB, -- Detailed source parameters
    data_source VARCHAR(100), -- USGS, local network, etc.
    quality_metrics JSONB, -- Data quality indicators
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

SELECT create_hypertable('seismic_events', 'origin_time');
CREATE INDEX idx_seismic_location ON seismic_events USING GIST (location);
CREATE INDEX idx_seismic_magnitude ON seismic_events (magnitude, origin_time);

-- Tidal analysis tables
CREATE TABLE tidal_analysis (
    id SERIAL PRIMARY KEY,
    chamber_id VARCHAR(50) REFERENCES volcanic_chambers(id),
    analysis_time TIMESTAMP WITH TIME ZONE NOT NULL,
    tidal_potential DOUBLE PRECISION NOT NULL,
    tidal_stress_tensor DOUBLE PRECISION[6] NOT NULL,
    lunar_position JSONB, -- Lunar ephemeris data
    solar_position JSONB, -- Solar ephemeris data
    earth_tide_components JSONB, -- Detailed tidal components
    love_numbers JSONB, -- Earth deformation parameters
    strain_rate DOUBLE PRECISION,
    stress_rate DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

SELECT create_hypertable('tidal_analysis', 'analysis_time');

-- Monitoring data integration tables
CREATE TABLE monitoring_parameters (
    id SERIAL PRIMARY KEY,
    chamber_id VARCHAR(50) REFERENCES volcanic_chambers(id),
    parameter_type VARCHAR(100) NOT NULL, -- pressure, temperature, deformation, gas
    measurement_time TIMESTAMP WITH TIME ZONE NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    uncertainty DOUBLE PRECISION,
    units VARCHAR(50) NOT NULL,
    instrument_id VARCHAR(100),
    location GEOMETRY(POINTZ, 4326),
    quality_flag INTEGER DEFAULT 1, -- 1=good, 0=questionable, -1=bad
    processing_level VARCHAR(50), -- raw, calibrated, processed
    metadata JSONB, -- Instrument-specific metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

SELECT create_hypertable('monitoring_parameters', 'measurement_time');
CREATE INDEX idx_monitoring_chamber_param ON monitoring_parameters (chamber_id, parameter_type, measurement_time);

-- Analysis results aggregation tables
CREATE TABLE analysis_summaries (
    id SERIAL PRIMARY KEY,
    analysis_id UUID UNIQUE NOT NULL,
    analysis_type VARCHAR(100) NOT NULL,
    chamber_network JSONB NOT NULL, -- Analyzed chamber network
    analysis_timeframe TSTZRANGE NOT NULL, -- Time range of analysis
    total_chambers_analyzed INTEGER NOT NULL,
    chambers_with_forecasts INTEGER NOT NULL,
    highest_eruption_probability DOUBLE PRECISION,
    most_critical_chamber VARCHAR(50),
    earliest_forecast_time DOUBLE PRECISION,
    maximum_predicted_vei DOUBLE PRECISION,
    total_external_triggers INTEGER,
    cascading_failure_potential BOOLEAN,
    analysis_parameters JSONB, -- Input parameters
    computational_metrics JSONB, -- Performance metrics
    validation_results JSONB, -- Validation against observations
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Historical eruption database for validation
CREATE TABLE historical_eruptions (
    id SERIAL PRIMARY KEY,
    eruption_id VARCHAR(100) UNIQUE NOT NULL,
    volcano_name VARCHAR(200) NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    vei INTEGER,
    erupted_volume DOUBLE PRECISION, -- m³ DRE
    column_height DOUBLE PRECISION, -- m
    eruption_style VARCHAR(100),
    precursor_signals JSONB, -- Documented precursors
    trigger_events JSONB, -- Known triggers
    casualties INTEGER,
    economic_impact DOUBLE PRECISION,
    data_sources JSONB, -- References and data sources
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_historical_location ON historical_eruptions USING GIST (location);
CREATE INDEX idx_historical_date ON historical_eruptions (start_date);
CREATE INDEX idx_historical_vei ON historical_eruptions (vei);

-- Model performance tracking
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    evaluation_date TIMESTAMP WITH TIME ZONE NOT NULL,
    performance_metrics JSONB NOT NULL, -- Accuracy, precision, recall, etc.
    validation_dataset VARCHAR(200),
    training_parameters JSONB,
    feature_importance JSONB,
    confusion_matrix JSONB,
    roc_curves JSONB,
    calibration_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Real-time alert system tables
CREATE TABLE eruption_alerts (
    id SERIAL PRIMARY KEY,
    alert_id UUID UNIQUE NOT NULL,
    chamber_id VARCHAR(50) REFERENCES volcanic_chambers(id),
    alert_level INTEGER NOT NULL, -- 1=advisory, 2=watch, 3=warning, 4=critical
    alert_type VARCHAR(100) NOT NULL,
    probability_threshold DOUBLE PRECISION NOT NULL,
    current_probability DOUBLE PRECISION NOT NULL,
    forecast_time_window TSTZRANGE,
    predicted_characteristics JSONB,
    affected_areas GEOMETRY(MULTIPOLYGON, 4326),
    population_at_risk INTEGER,
    recommended_actions JSONB,
    issued_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'active', -- active, expired, cancelled
    issuing_authority VARCHAR(200)
);

CREATE INDEX idx_alerts_chamber ON eruption_alerts (chamber_id, issued_at);
CREATE INDEX idx_alerts_level ON eruption_alerts (alert_level, status);

-- Advanced analytical functions and views

-- View for current chamber status
CREATE VIEW current_chamber_status AS
SELECT 
    c.id,
    c.name,
    c.location,
    c.monitoring_status,
    pt.critical_pressure,
    pt.time_to_failure,
    vo.gas_volume_fraction,
    vo.degassing_rate,
    ef.eruption_probability,
    ef.magnitude_estimate,
    ec.vei_estimate,
    ec.column_height
FROM volcanic_chambers c
LEFT JOIN LATERAL (
    SELECT * FROM pressure_thresholds 
    WHERE chamber_id = c.id 
    ORDER BY created_at DESC 
    LIMIT 1
) pt ON true
LEFT JOIN LATERAL (
    SELECT * FROM volatile_oversaturation 
    WHERE chamber_id = c.id 
    ORDER BY created_at DESC 
    LIMIT 1
) vo ON true
LEFT JOIN LATERAL (
    SELECT * FROM eruption_forecasts 
    WHERE chamber_id = c.id 
    ORDER BY issued_at DESC 
    LIMIT 1
) ef ON true
LEFT JOIN LATERAL (
    SELECT * FROM eruption_characteristics 
    WHERE chamber_id = c.id 
    ORDER BY created_at DESC 
    LIMIT 1
) ec ON true;

-- Function to calculate chamber network connectivity
CREATE OR REPLACE FUNCTION calculate_network_connectivity(
    network_chambers VARCHAR(50)[]
) RETURNS TABLE(
    source_id VARCHAR(50),
    target_id VARCHAR(50),
    connectivity_score DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cc.source_chamber_id,
        cc.target_chamber_id,
        cc.connectivity_strength * 
        EXP(-cc.distance / 5000.0) * -- Distance decay
        CASE 
            WHEN ABS(c1.depth - c2.depth) < 1000 THEN 1.0
            ELSE EXP(-ABS(c1.depth - c2.depth) / 2000.0)
        END as connectivity_score
    FROM chamber_connections cc
    JOIN volcanic_chambers c1 ON cc.source_chamber_id = c1.id
    JOIN volcanic_chambers c2 ON cc.target_chamber_id = c2.id
    WHERE cc.source_chamber_id = ANY(network_chambers)
      AND cc.target_chamber_id = ANY(network_chambers);
END;
$$ LANGUAGE plpgsql;

-- Function to get recent trigger activity
CREATE OR REPLACE FUNCTION get_recent_triggers(
    chamber_location GEOMETRY,
    radius_km DOUBLE PRECISION DEFAULT 50.0,
    time_window_hours INTEGER DEFAULT 168 -- 1 week
) RETURNS TABLE(
    trigger_id INTEGER,
    trigger_type VARCHAR(50),
    magnitude DOUBLE PRECISION,
    distance_km DOUBLE PRECISION,
    stress_change DOUBLE PRECISION,
    hours_ago DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        et.id,
        et.trigger_type,
        et.magnitude,
        ST_Distance(chamber_location, et.location) / 1000.0 as distance_km,
        et.stress_change,
        EXTRACT(EPOCH FROM (NOW() - et.timestamp)) / 3600.0 as hours_ago
    FROM external_triggers et
    WHERE ST_DWithin(chamber_location, et.location, radius_km * 1000.0)
      AND et.timestamp > NOW() - INTERVAL '1 hour' * time_window_hours
    ORDER BY et.timestamp DESC;
END;
$$ LANGUAGE plpgsql;

-- Function for probabilistic hazard assessment
CREATE OR REPLACE FUNCTION calculate_hazard_probability(
    chamber_location GEOMETRY,
    forecast_probability DOUBLE PRECISION,
    vei_estimate DOUBLE PRECISION,
    distance_km DOUBLE PRECISION
) RETURNS DOUBLE PRECISION AS $$
DECLARE
    base_hazard DOUBLE PRECISION;
    distance_factor DOUBLE PRECISION;
    vei_factor DOUBLE PRECISION;
BEGIN
    -- Base hazard from eruption probability
    base_hazard := forecast_probability;
    
    -- Distance attenuation (exponential decay)
    distance_factor := EXP(-distance_km / (10.0 * POWER(10, vei_estimate * 0.5)));
    
    -- VEI scaling factor
    vei_factor := POWER(10, vei_estimate - 2.0);
    
    -- Combined hazard probability
    RETURN LEAST(1.0, base_hazard * distance_factor * vei_factor);
END;
$$ LANGUAGE plpgsql;

-- Materialized view for performance optimization
CREATE MATERIALIZED VIEW chamber_risk_assessment AS
SELECT 
    c.id,
    c.name,
    c.location,
    COALESCE(ef.eruption_probability, 0.0) as eruption_probability,
    COALESCE(ec.vei_estimate, 0.0) as vei_estimate,
    COALESCE(pt.time_to_failure, 1e9) as time_to_failure,
    COALESCE(vo.gas_volume_fraction, 0.0) as gas_volume_fraction,
    -- Risk score calculation
    (COALESCE(ef.eruption_probability, 0.0) * 
     COALESCE(ec.vei_estimate, 0.0) * 
     (1.0 / GREATEST(COALESCE(pt.time_to_failure, 1e9), 1.0))) as risk_score,
    NOW() as last_updated
FROM volcanic_chambers c
LEFT JOIN LATERAL (
    SELECT * FROM eruption_forecasts 
    WHERE chamber_id = c.id 
    ORDER BY issued_at DESC 
    LIMIT 1
) ef ON true
LEFT JOIN LATERAL (
    SELECT * FROM eruption_characteristics 
    WHERE chamber_id = c.id 
    ORDER BY created_at DESC 
    LIMIT 1
) ec ON true
LEFT JOIN LATERAL (
    SELECT * FROM pressure_thresholds 
    WHERE chamber_id = c.id 
    ORDER BY created_at DESC 
    LIMIT 1
) pt ON true
LEFT JOIN LATERAL (
    SELECT * FROM volatile_oversaturation 
    WHERE chamber_id = c.id 
    ORDER BY created_at DESC 
    LIMIT 1
) vo ON true;

-- Create indexes for performance
CREATE INDEX idx_pressure_thresholds_chamber_time ON pressure_thresholds (chamber_id, created_at DESC);
CREATE INDEX idx_volatile_oversaturation_chamber_time ON volatile_oversaturation (chamber_id, created_at DESC);
CREATE INDEX idx_eruption_forecasts_chamber_time ON eruption_forecasts (chamber_id, issued_at DESC);
CREATE INDEX idx_eruption_characteristics_chamber_time ON eruption_characteristics (chamber_id, created_at DESC);

-- Refresh materialized view function
CREATE OR REPLACE FUNCTION refresh_risk_assessment() RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW chamber_risk_assessment;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-refresh materialized view
CREATE OR REPLACE FUNCTION trigger_refresh_risk_assessment() RETURNS TRIGGER AS $$
BEGIN
    PERFORM refresh_risk_assessment();
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic refresh
CREATE TRIGGER refresh_on_forecast_update
    AFTER INSERT OR UPDATE ON eruption_forecasts
    FOR EACH STATEMENT
    EXECUTE FUNCTION trigger_refresh_risk_assessment();

CREATE TRIGGER refresh_on_characteristics_update
    AFTER INSERT OR UPDATE ON eruption_characteristics
    FOR EACH STATEMENT
    EXECUTE FUNCTION trigger_refresh_risk_assessment();

-- Grant permissions (adjust as needed for your security model)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO volcanic_analysis_role;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO volcanic_analysis_role;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO volcanic_analysis_role;