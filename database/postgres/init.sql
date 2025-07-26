-- Initialize PostgreSQL database for Prometheus AI Platform
-- Enable PostGIS extension for geospatial data
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;

-- Create schemas for different data types
CREATE SCHEMA IF NOT EXISTS geological;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS simulations;
CREATE SCHEMA IF NOT EXISTS models;

-- Geological data tables
CREATE TABLE geological.locations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    elevation DOUBLE PRECISION,
    depth DOUBLE PRECISION,
    coordinate_system VARCHAR(50) DEFAULT 'WGS84',
    geom GEOMETRY(POINT, 4326),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create spatial index
CREATE INDEX idx_locations_geom ON geological.locations USING GIST (geom);

-- Seismic events table
CREATE TABLE geological.seismic_events (
    id SERIAL PRIMARY KEY,
    event_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location_id INTEGER REFERENCES geological.locations(id),
    magnitude DOUBLE PRECISION NOT NULL,
    depth DOUBLE PRECISION NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    confidence DOUBLE PRECISION DEFAULT 1.0,
    waveform_data BYTEA,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for seismic events
CREATE INDEX idx_seismic_events_time ON geological.seismic_events (event_time);
CREATE INDEX idx_seismic_events_magnitude ON geological.seismic_events (magnitude);
CREATE INDEX idx_seismic_events_type ON geological.seismic_events (event_type);

-- Deformation measurements table
CREATE TABLE geological.deformation_measurements (
    id SERIAL PRIMARY KEY,
    measurement_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location_id INTEGER REFERENCES geological.locations(id),
    displacement_x DOUBLE PRECISION NOT NULL,
    displacement_y DOUBLE PRECISION NOT NULL,
    displacement_z DOUBLE PRECISION NOT NULL,
    uncertainty_x DOUBLE PRECISION,
    uncertainty_y DOUBLE PRECISION,
    uncertainty_z DOUBLE PRECISION,
    measurement_type VARCHAR(50) NOT NULL,
    reference_frame VARCHAR(50) DEFAULT 'ITRF2014',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for deformation measurements
CREATE INDEX idx_deformation_time ON geological.deformation_measurements (measurement_time);
CREATE INDEX idx_deformation_type ON geological.deformation_measurements (measurement_type);

-- Geochemical samples table
CREATE TABLE geological.geochemical_samples (
    id SERIAL PRIMARY KEY,
    sample_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location_id INTEGER REFERENCES geological.locations(id),
    sample_type VARCHAR(50) NOT NULL,
    composition JSONB NOT NULL,
    isotope_ratios JSONB,
    temperature DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for geochemical samples
CREATE INDEX idx_geochemical_time ON geological.geochemical_samples (sample_time);
CREATE INDEX idx_geochemical_type ON geological.geochemical_samples (sample_type);
CREATE INDEX idx_geochemical_composition ON geological.geochemical_samples USING GIN (composition);

-- Thermal measurements table
CREATE TABLE geological.thermal_measurements (
    id SERIAL PRIMARY KEY,
    measurement_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location_id INTEGER REFERENCES geological.locations(id),
    temperature DOUBLE PRECISION NOT NULL,
    measurement_method VARCHAR(100) NOT NULL,
    spatial_resolution DOUBLE PRECISION,
    uncertainty DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for thermal measurements
CREATE INDEX idx_thermal_time ON geological.thermal_measurements (measurement_time);
CREATE INDEX idx_thermal_method ON geological.thermal_measurements (measurement_method);

-- Monitoring data tables
CREATE TABLE monitoring.sensors (
    id SERIAL PRIMARY KEY,
    sensor_id VARCHAR(100) UNIQUE NOT NULL,
    location_id INTEGER REFERENCES geological.locations(id),
    sensor_type VARCHAR(50) NOT NULL,
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    installation_date DATE,
    calibration_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sensor readings table (for real-time monitoring)
CREATE TABLE monitoring.sensor_readings (
    id BIGSERIAL PRIMARY KEY,
    sensor_id VARCHAR(100) REFERENCES monitoring.sensors(sensor_id),
    reading_time TIMESTAMP WITH TIME ZONE NOT NULL,
    measurement_type VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(20) NOT NULL,
    quality_flag VARCHAR(20) DEFAULT 'good',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Partition sensor readings by month for better performance
CREATE TABLE monitoring.sensor_readings_y2024m01 PARTITION OF monitoring.sensor_readings
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Create indexes for sensor readings
CREATE INDEX idx_sensor_readings_time ON monitoring.sensor_readings (reading_time);
CREATE INDEX idx_sensor_readings_sensor ON monitoring.sensor_readings (sensor_id);
CREATE INDEX idx_sensor_readings_type ON monitoring.sensor_readings (measurement_type);

-- Alert levels table
CREATE TABLE monitoring.alert_levels (
    id SERIAL PRIMARY KEY,
    level INTEGER NOT NULL CHECK (level >= 0 AND level <= 5),
    description TEXT NOT NULL,
    recommended_actions TEXT[],
    validity_start TIMESTAMP WITH TIME ZONE NOT NULL,
    validity_end TIMESTAMP WITH TIME ZONE,
    confidence DOUBLE PRECISION DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Volcanic risk assessments table
CREATE TABLE monitoring.volcanic_risk_assessments (
    id SERIAL PRIMARY KEY,
    volcano_id VARCHAR(100) NOT NULL,
    assessment_time TIMESTAMP WITH TIME ZONE NOT NULL,
    alert_level_id INTEGER REFERENCES monitoring.alert_levels(id),
    eruption_probabilities JSONB NOT NULL,
    hazard_zones JSONB,
    contributing_factors TEXT[],
    model_ensemble_agreement DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Simulation tables
CREATE TABLE simulations.simulation_runs (
    id SERIAL PRIMARY KEY,
    run_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    simulation_type VARCHAR(50) NOT NULL,
    initial_conditions JSONB NOT NULL,
    parameters JSONB NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    spatial_domain JSONB NOT NULL,
    resolution JSONB NOT NULL,
    physics_models TEXT[],
    status VARCHAR(20) DEFAULT 'pending',
    progress DOUBLE PRECISION DEFAULT 0.0,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Simulation results table
CREATE TABLE simulations.simulation_results (
    id BIGSERIAL PRIMARY KEY,
    simulation_id INTEGER REFERENCES simulations.simulation_runs(id),
    result_time TIMESTAMP WITH TIME ZONE NOT NULL,
    result_type VARCHAR(50) NOT NULL,
    spatial_location GEOMETRY(POINT, 4326),
    values JSONB NOT NULL,
    uncertainty JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create spatial index for simulation results
CREATE INDEX idx_simulation_results_location ON simulations.simulation_results USING GIST (spatial_location);
CREATE INDEX idx_simulation_results_time ON simulations.simulation_results (result_time);
CREATE INDEX idx_simulation_results_type ON simulations.simulation_results (result_type);

-- Model tables
CREATE TABLE models.model_versions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    description TEXT,
    architecture JSONB,
    training_data_hash VARCHAR(64),
    performance_metrics JSONB,
    model_file_path TEXT,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(model_name, version)
);

-- Model predictions table
CREATE TABLE models.model_predictions (
    id BIGSERIAL PRIMARY KEY,
    model_version_id INTEGER REFERENCES models.model_versions(id),
    prediction_time TIMESTAMP WITH TIME ZONE NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    input_data_hash VARCHAR(64),
    values JSONB NOT NULL,
    uncertainty JSONB,
    confidence_interval JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for model predictions
CREATE INDEX idx_model_predictions_time ON models.model_predictions (prediction_time);
CREATE INDEX idx_model_predictions_type ON models.model_predictions (prediction_type);
CREATE INDEX idx_model_predictions_model ON models.model_predictions (model_version_id);

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_locations_updated_at BEFORE UPDATE ON geological.locations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_simulation_runs_updated_at BEFORE UPDATE ON simulations.simulation_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE VIEW geological.recent_seismic_activity AS
SELECT 
    se.id,
    se.event_time,
    se.magnitude,
    se.depth,
    se.event_type,
    l.name as location_name,
    l.latitude,
    l.longitude,
    l.geom
FROM geological.seismic_events se
JOIN geological.locations l ON se.location_id = l.id
WHERE se.event_time >= NOW() - INTERVAL '30 days'
ORDER BY se.event_time DESC;

CREATE VIEW monitoring.active_sensors AS
SELECT 
    s.sensor_id,
    s.sensor_type,
    s.manufacturer,
    s.model,
    l.name as location_name,
    l.latitude,
    l.longitude,
    s.calibration_date,
    s.status
FROM monitoring.sensors s
JOIN geological.locations l ON s.location_id = l.id
WHERE s.status = 'active';

-- Grant permissions
GRANT USAGE ON SCHEMA geological TO prometheus;
GRANT USAGE ON SCHEMA monitoring TO prometheus;
GRANT USAGE ON SCHEMA simulations TO prometheus;
GRANT USAGE ON SCHEMA models TO prometheus;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA geological TO prometheus;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO prometheus;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA simulations TO prometheus;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA models TO prometheus;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA geological TO prometheus;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO prometheus;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA simulations TO prometheus;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA models TO prometheus;