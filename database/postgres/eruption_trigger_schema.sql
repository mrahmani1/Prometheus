-- Eruption Trigger Analysis Framework Database Schema
-- Advanced database schema for comprehensive eruption trigger analysis

-- Pressure Thresholds Table
CREATE TABLE IF NOT EXISTS pressure_thresholds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(100) NOT NULL,
    critical_pressure DOUBLE PRECISION NOT NULL,
    pressure_gradient DOUBLE PRECISION NOT NULL,
    threshold_uncertainty DOUBLE PRECISION NOT NULL,
    time_to_failure DOUBLE PRECISION NOT NULL,
    failure_mechanism VARCHAR(50) NOT NULL,
    depth DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_pressure_thresholds_chamber ON pressure_thresholds(chamber_id);
CREATE INDEX idx_pressure_thresholds_mechanism ON pressure_thresholds(failure_mechanism);
CREATE INDEX idx_pressure_thresholds_created ON pressure_thresholds(created_at);

-- Volatile Oversaturation Table
CREATE TABLE IF NOT EXISTS volatile_oversaturation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(100) NOT NULL,
    co2_saturation DOUBLE PRECISION NOT NULL,
    h2o_saturation DOUBLE PRECISION NOT NULL,
    so2_saturation DOUBLE PRECISION NOT NULL,
    bubble_nucleation_rate DOUBLE PRECISION NOT NULL,
    bubble_growth_rate DOUBLE PRECISION NOT NULL,
    coalescence_efficiency DOUBLE PRECISION NOT NULL,
    gas_volume_fraction DOUBLE PRECISION NOT NULL,
    degassing_rate DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_volatile_oversaturation_chamber ON volatile_oversaturation(chamber_id);
CREATE INDEX idx_volatile_oversaturation_created ON volatile_oversaturation(created_at);

-- External Triggers Table
CREATE TABLE IF NOT EXISTS external_triggers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trigger_type VARCHAR(50) NOT NULL,
    magnitude DOUBLE PRECISION NOT NULL,
    location_lat DOUBLE PRECISION NOT NULL,
    location_lon DOUBLE PRECISION NOT NULL,
    location_depth DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    stress_change DOUBLE PRECISION NOT NULL,
    strain_rate DOUBLE PRECISION NOT NULL,
    influence_radius DOUBLE PRECISION NOT NULL,
    coupling_strength DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_external_triggers_type ON external_triggers(trigger_type);
CREATE INDEX idx_external_triggers_timestamp ON external_triggers(timestamp);
CREATE INDEX idx_external_triggers_location ON external_triggers(location_lat, location_lon);
CREATE INDEX idx_external_triggers_created ON external_triggers(created_at);

-- Cascading Failures Table
CREATE TABLE IF NOT EXISTS cascading_failures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    failure_sequence JSONB NOT NULL,
    failure_times JSONB NOT NULL,
    failure_probabilities JSONB NOT NULL,
    total_system_failure_time DOUBLE PRECISION NOT NULL,
    critical_path JSONB NOT NULL,
    stress_transfer_matrix JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_cascading_failures_created ON cascading_failures(created_at);
CREATE INDEX idx_cascading_failures_failure_time ON cascading_failures(total_system_failure_time);

-- Eruption Forecasts Table
CREATE TABLE IF NOT EXISTS eruption_forecasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id VARCHAR(100) NOT NULL,
    eruption_probability DOUBLE PRECISION NOT NULL,
    time_window_start DOUBLE PRECISION NOT NULL,
    time_window_end DOUBLE PRECISION NOT NULL,
    confidence_lower DOUBLE PRECISION NOT NULL,
    confidence_upper DOUBLE PRECISION NOT NULL,
    magnitude_estimate DOUBLE PRECISION NOT NULL,
    style_prediction VARCHAR(50) NOT NULL,
    uncertainty_bounds JSONB NOT NULL,
    forecast_skill DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_eruption_forecasts_system ON eruption_forecasts(system_id);
CREATE INDEX idx_eruption_forecasts_probability ON eruption_forecasts(eruption_probability);
CREATE INDEX idx_eruption_forecasts_created ON eruption_forecasts(created_at);

-- Eruption Characteristics Table
CREATE TABLE IF NOT EXISTS eruption_characteristics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id VARCHAR(100) NOT NULL,
    vei_estimate DOUBLE PRECISION NOT NULL,
    discharge_rate DOUBLE PRECISION NOT NULL,
    eruption_duration DOUBLE PRECISION NOT NULL,
    explosivity_index DOUBLE PRECISION NOT NULL,
    column_height DOUBLE PRECISION NOT NULL,
    pyroclastic_flow_potential DOUBLE PRECISION NOT NULL,
    lava_flow_potential DOUBLE PRECISION NOT NULL,
    ash_dispersal_range DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_eruption_characteristics_system ON eruption_characteristics(system_id);
CREATE INDEX idx_eruption_characteristics_vei ON eruption_characteristics(vei_estimate);
CREATE INDEX idx_eruption_characteristics_created ON eruption_characteristics(created_at);

-- Temporal Eruption Models Table
CREATE TABLE IF NOT EXISTS temporal_eruption_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id VARCHAR(100) NOT NULL,
    sequence_model JSONB NOT NULL,
    temporal_patterns JSONB NOT NULL,
    repose_intervals JSONB NOT NULL,
    cumulative_hazard JSONB NOT NULL,
    time_horizon DOUBLE PRECISION NOT NULL,
    n_simulations INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_temporal_models_system ON temporal_eruption_models(system_id);
CREATE INDEX idx_temporal_models_created ON temporal_eruption_models(created_at);

-- Eruption Model Validation Table
CREATE TABLE IF NOT EXISTS eruption_model_validation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_validation JSONB NOT NULL,
    magnitude_validation JSONB NOT NULL,
    temporal_validation JSONB NOT NULL,
    model_performance JSONB NOT NULL,
    n_historical_eruptions INTEGER NOT NULL,
    validation_timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_model_validation_timestamp ON eruption_model_validation(validation_timestamp);
CREATE INDEX idx_model_validation_created ON eruption_model_validation(created_at);

-- Seismic Events Table (for external trigger analysis)
CREATE TABLE IF NOT EXISTS seismic_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id VARCHAR(100) UNIQUE NOT NULL,
    magnitude DOUBLE PRECISION NOT NULL,
    location_lat DOUBLE PRECISION NOT NULL,
    location_lon DOUBLE PRECISION NOT NULL,
    depth DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    stress_drop DOUBLE PRECISION,
    moment_tensor JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_seismic_events_magnitude ON seismic_events(magnitude);
CREATE INDEX idx_seismic_events_timestamp ON seismic_events(timestamp);
CREATE INDEX idx_seismic_events_location ON seismic_events(location_lat, location_lon);
CREATE INDEX idx_seismic_events_type ON seismic_events(event_type);

-- Deformation Data Table
CREATE TABLE IF NOT EXISTS deformation_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id VARCHAR(100) NOT NULL,
    measurement_type VARCHAR(50) NOT NULL, -- 'insar', 'gps', 'tilt', 'strain'
    displacement_x DOUBLE PRECISION,
    displacement_y DOUBLE PRECISION,
    displacement_z DOUBLE PRECISION,
    displacement_magnitude DOUBLE PRECISION,
    measurement_uncertainty DOUBLE PRECISION,
    timestamp TIMESTAMP NOT NULL,
    location_lat DOUBLE PRECISION NOT NULL,
    location_lon DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_deformation_station ON deformation_data(station_id);
CREATE INDEX idx_deformation_type ON deformation_data(measurement_type);
CREATE INDEX idx_deformation_timestamp ON deformation_data(timestamp);
CREATE INDEX idx_deformation_location ON deformation_data(location_lat, location_lon);

-- Gas Emission Data Table
CREATE TABLE IF NOT EXISTS gas_emission_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id VARCHAR(100) NOT NULL,
    gas_species VARCHAR(20) NOT NULL, -- 'CO2', 'SO2', 'H2S', etc.
    concentration DOUBLE PRECISION NOT NULL, -- ppm or other units
    flux_rate DOUBLE PRECISION, -- kg/s or tons/day
    measurement_method VARCHAR(50), -- 'doas', 'ftir', 'multigas', etc.
    measurement_uncertainty DOUBLE PRECISION,
    timestamp TIMESTAMP NOT NULL,
    location_lat DOUBLE PRECISION NOT NULL,
    location_lon DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_gas_emission_station ON gas_emission_data(station_id);
CREATE INDEX idx_gas_emission_species ON gas_emission_data(gas_species);
CREATE INDEX idx_gas_emission_timestamp ON gas_emission_data(timestamp);
CREATE INDEX idx_gas_emission_location ON gas_emission_data(location_lat, location_lon);

-- Thermal Data Table
CREATE TABLE IF NOT EXISTS thermal_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id VARCHAR(100) NOT NULL,
    measurement_type VARCHAR(50) NOT NULL, -- 'satellite', 'thermal_camera', 'thermocouple'
    temperature DOUBLE PRECISION NOT NULL, -- Celsius or Kelvin
    temperature_anomaly DOUBLE PRECISION, -- Relative to baseline
    pixel_size DOUBLE PRECISION, -- For satellite data
    measurement_uncertainty DOUBLE PRECISION,
    timestamp TIMESTAMP NOT NULL,
    location_lat DOUBLE PRECISION NOT NULL,
    location_lon DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_thermal_data_type ON thermal_data(measurement_type);
CREATE INDEX idx_thermal_data_timestamp ON thermal_data(timestamp);
CREATE INDEX idx_thermal_data_location ON thermal_data(location_lat, location_lon);
CREATE INDEX idx_thermal_data_temperature ON thermal_data(temperature);

-- Magma Chamber Network Table
CREATE TABLE IF NOT EXISTS magma_chamber_networks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id VARCHAR(100) UNIQUE NOT NULL,
    volcano_name VARCHAR(200) NOT NULL,
    center_lat DOUBLE PRECISION NOT NULL,
    center_lon DOUBLE PRECISION NOT NULL,
    network_geometry JSONB NOT NULL, -- GeoJSON of network extent
    chamber_count INTEGER NOT NULL,
    total_volume DOUBLE PRECISION NOT NULL,
    network_depth_range JSONB NOT NULL, -- [min_depth, max_depth]
    monitoring_status VARCHAR(50) NOT NULL,
    last_activity TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_chamber_networks_volcano ON magma_chamber_networks(volcano_name);
CREATE INDEX idx_chamber_networks_location ON magma_chamber_networks(center_lat, center_lon);
CREATE INDEX idx_chamber_networks_status ON magma_chamber_networks(monitoring_status);

-- Individual Magma Chambers Table
CREATE TABLE IF NOT EXISTS magma_chambers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(100) UNIQUE NOT NULL,
    network_id VARCHAR(100) NOT NULL REFERENCES magma_chamber_networks(network_id),
    chamber_name VARCHAR(200),
    location_lat DOUBLE PRECISION NOT NULL,
    location_lon DOUBLE PRECISION NOT NULL,
    depth_min DOUBLE PRECISION NOT NULL,
    depth_max DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    geometry JSONB NOT NULL, -- 3D geometry description
    composition JSONB NOT NULL, -- Chemical composition
    temperature DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    crystallinity DOUBLE PRECISION,
    volatile_content JSONB,
    rheological_properties JSONB,
    connection_strength JSONB, -- Connections to other chambers
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_magma_chambers_network ON magma_chambers(network_id);
CREATE INDEX idx_magma_chambers_location ON magma_chambers(location_lat, location_lon);
CREATE INDEX idx_magma_chambers_depth ON magma_chambers(depth_min, depth_max);
CREATE INDEX idx_magma_chambers_volume ON magma_chambers(volume);

-- Historical Eruptions Table
CREATE TABLE IF NOT EXISTS historical_eruptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eruption_id VARCHAR(100) UNIQUE NOT NULL,
    volcano_name VARCHAR(200) NOT NULL,
    eruption_date DATE NOT NULL,
    eruption_timestamp TIMESTAMP,
    vei INTEGER,
    magnitude DOUBLE PRECISION,
    volume DOUBLE PRECISION, -- m³ of erupted material
    duration DOUBLE PRECISION, -- seconds
    style VARCHAR(100), -- eruption style
    explosivity_index DOUBLE PRECISION,
    column_height DOUBLE PRECISION, -- meters
    pyroclastic_flows BOOLEAN DEFAULT FALSE,
    lava_flows BOOLEAN DEFAULT FALSE,
    ash_fall_extent DOUBLE PRECISION, -- km
    fatalities INTEGER DEFAULT 0,
    economic_impact DOUBLE PRECISION, -- USD
    chambers_involved JSONB, -- List of chamber IDs involved
    precursor_duration DOUBLE PRECISION, -- seconds of precursory activity
    precursor_types JSONB, -- Types of precursory activity
    sequence_position INTEGER, -- Position in eruption sequence
    repose_time DOUBLE PRECISION, -- Time since previous eruption (seconds)
    location_lat DOUBLE PRECISION NOT NULL,
    location_lon DOUBLE PRECISION NOT NULL,
    data_quality VARCHAR(20) DEFAULT 'unknown', -- 'excellent', 'good', 'fair', 'poor', 'unknown'
    data_sources JSONB, -- References to data sources
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_historical_eruptions_volcano ON historical_eruptions(volcano_name);
CREATE INDEX idx_historical_eruptions_date ON historical_eruptions(eruption_date);
CREATE INDEX idx_historical_eruptions_vei ON historical_eruptions(vei);
CREATE INDEX idx_historical_eruptions_magnitude ON historical_eruptions(magnitude);
CREATE INDEX idx_historical_eruptions_location ON historical_eruptions(location_lat, location_lon);

-- Trigger Analysis Results Table
CREATE TABLE IF NOT EXISTS trigger_analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id VARCHAR(100) UNIQUE NOT NULL,
    chamber_network_id VARCHAR(100) NOT NULL,
    analysis_timestamp TIMESTAMP NOT NULL,
    analysis_type VARCHAR(50) NOT NULL, -- 'complete', 'pressure_only', 'volatile_only', etc.
    input_parameters JSONB NOT NULL,
    pressure_analysis_results JSONB,
    volatile_analysis_results JSONB,
    external_trigger_results JSONB,
    cascading_failure_results JSONB,
    eruption_forecast_results JSONB,
    characteristics_prediction_results JSONB,
    temporal_model_results JSONB,
    validation_results JSONB,
    overall_risk_assessment JSONB,
    confidence_metrics JSONB,
    computational_performance JSONB, -- Runtime, memory usage, etc.
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_trigger_analysis_network ON trigger_analysis_results(chamber_network_id);
CREATE INDEX idx_trigger_analysis_timestamp ON trigger_analysis_results(analysis_timestamp);
CREATE INDEX idx_trigger_analysis_type ON trigger_analysis_results(analysis_type);
CREATE INDEX idx_trigger_analysis_created ON trigger_analysis_results(created_at);

-- Performance Metrics Table
CREATE TABLE IF NOT EXISTS analysis_performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id VARCHAR(100) NOT NULL,
    component_name VARCHAR(100) NOT NULL,
    execution_time_ms DOUBLE PRECISION NOT NULL,
    memory_usage_mb DOUBLE PRECISION,
    cpu_usage_percent DOUBLE PRECISION,
    gpu_usage_percent DOUBLE PRECISION,
    database_queries INTEGER DEFAULT 0,
    cache_hit_rate DOUBLE PRECISION,
    error_count INTEGER DEFAULT 0,
    warning_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_performance_analysis ON analysis_performance_metrics(analysis_id);
CREATE INDEX idx_performance_component ON analysis_performance_metrics(component_name);
CREATE INDEX idx_performance_execution_time ON analysis_performance_metrics(execution_time_ms);

-- Create views for common queries

-- View for latest pressure thresholds by chamber
CREATE OR REPLACE VIEW latest_pressure_thresholds AS
SELECT DISTINCT ON (chamber_id) 
    chamber_id,
    critical_pressure,
    pressure_gradient,
    threshold_uncertainty,
    time_to_failure,
    failure_mechanism,
    depth,
    created_at
FROM pressure_thresholds
ORDER BY chamber_id, created_at DESC;

-- View for recent external triggers
CREATE OR REPLACE VIEW recent_external_triggers AS
SELECT *
FROM external_triggers
WHERE timestamp >= NOW() - INTERVAL '30 days'
ORDER BY timestamp DESC;

-- View for eruption forecast summary
CREATE OR REPLACE VIEW eruption_forecast_summary AS
SELECT 
    system_id,
    AVG(eruption_probability) as avg_probability,
    MAX(eruption_probability) as max_probability,
    AVG(magnitude_estimate) as avg_magnitude,
    COUNT(*) as forecast_count,
    MAX(created_at) as latest_forecast
FROM eruption_forecasts
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY system_id;

-- View for chamber network status
CREATE OR REPLACE VIEW chamber_network_status AS
SELECT 
    mcn.network_id,
    mcn.volcano_name,
    mcn.chamber_count,
    mcn.total_volume,
    mcn.monitoring_status,
    COUNT(pt.id) as pressure_threshold_count,
    COUNT(vo.id) as volatile_analysis_count,
    COUNT(ef.id) as forecast_count,
    MAX(ef.created_at) as latest_forecast
FROM magma_chamber_networks mcn
LEFT JOIN magma_chambers mc ON mcn.network_id = mc.network_id
LEFT JOIN pressure_thresholds pt ON mc.chamber_id = pt.chamber_id
LEFT JOIN volatile_oversaturation vo ON mc.chamber_id = vo.chamber_id
LEFT JOIN eruption_forecasts ef ON mcn.network_id = ef.system_id
GROUP BY mcn.network_id, mcn.volcano_name, mcn.chamber_count, 
         mcn.total_volume, mcn.monitoring_status;

-- Functions for data analysis

-- Function to calculate eruption probability trend
CREATE OR REPLACE FUNCTION calculate_probability_trend(
    p_system_id VARCHAR(100),
    p_days INTEGER DEFAULT 30
) RETURNS TABLE (
    date_bucket DATE,
    avg_probability DOUBLE PRECISION,
    max_probability DOUBLE PRECISION,
    forecast_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        DATE(ef.created_at) as date_bucket,
        AVG(ef.eruption_probability) as avg_probability,
        MAX(ef.eruption_probability) as max_probability,
        COUNT(*) as forecast_count
    FROM eruption_forecasts ef
    WHERE ef.system_id = p_system_id
      AND ef.created_at >= NOW() - (p_days || ' days')::INTERVAL
    GROUP BY DATE(ef.created_at)
    ORDER BY date_bucket;
END;
$$ LANGUAGE plpgsql;

-- Function to get chamber network risk assessment
CREATE OR REPLACE FUNCTION get_network_risk_assessment(
    p_network_id VARCHAR(100)
) RETURNS TABLE (
    risk_level VARCHAR(20),
    risk_score DOUBLE PRECISION,
    primary_risk_factors JSONB,
    recommendation TEXT
) AS $$
DECLARE
    v_max_probability DOUBLE PRECISION;
    v_avg_magnitude DOUBLE PRECISION;
    v_trigger_count INTEGER;
    v_risk_score DOUBLE PRECISION;
    v_risk_level VARCHAR(20);
BEGIN
    -- Get latest forecast data
    SELECT 
        COALESCE(MAX(ef.eruption_probability), 0.0),
        COALESCE(AVG(ec.vei_estimate), 0.0)
    INTO v_max_probability, v_avg_magnitude
    FROM eruption_forecasts ef
    LEFT JOIN eruption_characteristics ec ON ef.system_id = ec.system_id
    WHERE ef.system_id = p_network_id
      AND ef.created_at >= NOW() - INTERVAL '7 days';
    
    -- Count recent external triggers
    SELECT COUNT(*)
    INTO v_trigger_count
    FROM external_triggers et
    WHERE et.timestamp >= NOW() - INTERVAL '7 days';
    
    -- Calculate risk score
    v_risk_score := (v_max_probability * 0.5) + 
                    (LEAST(v_avg_magnitude / 8.0, 1.0) * 0.3) + 
                    (LEAST(v_trigger_count / 10.0, 1.0) * 0.2);
    
    -- Determine risk level
    IF v_risk_score >= 0.8 THEN
        v_risk_level := 'CRITICAL';
    ELSIF v_risk_score >= 0.6 THEN
        v_risk_level := 'HIGH';
    ELSIF v_risk_score >= 0.4 THEN
        v_risk_level := 'MODERATE';
    ELSIF v_risk_score >= 0.2 THEN
        v_risk_level := 'LOW';
    ELSE
        v_risk_level := 'MINIMAL';
    END IF;
    
    RETURN QUERY
    SELECT 
        v_risk_level,
        v_risk_score,
        jsonb_build_object(
            'eruption_probability', v_max_probability,
            'average_magnitude', v_avg_magnitude,
            'trigger_count', v_trigger_count
        ),
        CASE 
            WHEN v_risk_level = 'CRITICAL' THEN 'Immediate evacuation and emergency response activation recommended'
            WHEN v_risk_level = 'HIGH' THEN 'Enhanced monitoring and emergency preparedness required'
            WHEN v_risk_level = 'MODERATE' THEN 'Increased monitoring frequency and public awareness'
            WHEN v_risk_level = 'LOW' THEN 'Continue routine monitoring'
            ELSE 'Standard monitoring protocols sufficient'
        END;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic updates

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update triggers to relevant tables
CREATE TRIGGER update_pressure_thresholds_updated_at 
    BEFORE UPDATE ON pressure_thresholds 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_volatile_oversaturation_updated_at 
    BEFORE UPDATE ON volatile_oversaturation 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_external_triggers_updated_at 
    BEFORE UPDATE ON external_triggers 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cascading_failures_updated_at 
    BEFORE UPDATE ON cascading_failures 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_eruption_forecasts_updated_at 
    BEFORE UPDATE ON eruption_forecasts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_eruption_characteristics_updated_at 
    BEFORE UPDATE ON eruption_characteristics 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_temporal_eruption_models_updated_at 
    BEFORE UPDATE ON temporal_eruption_models 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_magma_chamber_networks_updated_at 
    BEFORE UPDATE ON magma_chamber_networks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_magma_chambers_updated_at 
    BEFORE UPDATE ON magma_chambers 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE pressure_thresholds IS 'Pressure threshold analysis results for eruption initiation';
COMMENT ON TABLE volatile_oversaturation IS 'Volatile oversaturation and bubble nucleation modeling results';
COMMENT ON TABLE external_triggers IS 'External trigger events affecting volcanic systems';
COMMENT ON TABLE cascading_failures IS 'Cascading failure analysis across chamber networks';
COMMENT ON TABLE eruption_forecasts IS 'Probabilistic eruption forecasting results';
COMMENT ON TABLE eruption_characteristics IS 'Predicted eruption magnitude and style characteristics';
COMMENT ON TABLE temporal_eruption_models IS 'Temporal eruption sequence modeling results';
COMMENT ON TABLE eruption_model_validation IS 'Validation results against historical eruptions';
COMMENT ON TABLE magma_chamber_networks IS 'Multi-chamber volcanic system definitions';
COMMENT ON TABLE magma_chambers IS 'Individual magma chamber properties and characteristics';
COMMENT ON TABLE historical_eruptions IS 'Historical eruption database for model validation';
COMMENT ON TABLE trigger_analysis_results IS 'Complete eruption trigger analysis results';
COMMENT ON TABLE analysis_performance_metrics IS 'Performance metrics for analysis components';