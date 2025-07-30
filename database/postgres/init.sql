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

-- ============================================================================
-- ADVANCED SPATIAL INDEXING FOR GEOLOGICAL COORDINATE SYSTEMS
-- ============================================================================

-- Create custom spatial reference systems for geological coordinate systems
INSERT INTO spatial_ref_sys (srid, auth_name, auth_srid, proj4text, srtext) VALUES (
    9001, 'PROMETHEUS', 9001,
    '+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs',
    'PROJCS["UTM Zone 10N WGS84 Geological",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-123],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]'
) ON CONFLICT (srid) DO NOTHING;

-- Add geological coordinate system columns to locations table
ALTER TABLE geological.locations ADD COLUMN IF NOT EXISTS geom_utm GEOMETRY(POINT, 9001);
ALTER TABLE geological.locations ADD COLUMN IF NOT EXISTS geom_3d GEOMETRY(POINTZ, 4326);

-- Create advanced spatial indexes with different coordinate systems
CREATE INDEX IF NOT EXISTS idx_locations_geom_utm ON geological.locations USING GIST (geom_utm);
CREATE INDEX IF NOT EXISTS idx_locations_geom_3d ON geological.locations USING GIST (geom_3d);

-- Create spatial-temporal composite indexes
CREATE INDEX IF NOT EXISTS idx_seismic_events_spatio_temporal ON geological.seismic_events 
USING GIST (
    (SELECT geom FROM geological.locations WHERE id = location_id), 
    event_time
);

CREATE INDEX IF NOT EXISTS idx_deformation_spatio_temporal ON geological.deformation_measurements 
USING GIST (
    (SELECT geom FROM geological.locations WHERE id = location_id), 
    measurement_time
);

-- Create specialized geological indexes
CREATE INDEX IF NOT EXISTS idx_seismic_magnitude_depth ON geological.seismic_events (magnitude, depth);
CREATE INDEX IF NOT EXISTS idx_thermal_temp_time ON geological.thermal_measurements (temperature, measurement_time);

-- Create multi-dimensional spatial indexes for geological analysis
CREATE INDEX IF NOT EXISTS idx_locations_elevation_depth ON geological.locations (elevation, depth) WHERE elevation IS NOT NULL AND depth IS NOT NULL;

-- ============================================================================
-- MATERIALIZED VIEWS FOR HIGH-PERFORMANCE GEOLOGICAL ANALYSIS
-- ============================================================================

-- Materialized view for seismic activity analysis
CREATE MATERIALIZED VIEW geological.mv_seismic_activity_summary AS
SELECT 
    DATE_TRUNC('day', se.event_time) as activity_date,
    se.event_type,
    COUNT(*) as event_count,
    AVG(se.magnitude) as avg_magnitude,
    MAX(se.magnitude) as max_magnitude,
    MIN(se.magnitude) as min_magnitude,
    STDDEV(se.magnitude) as magnitude_stddev,
    AVG(se.depth) as avg_depth,
    ST_Centroid(ST_Collect(l.geom)) as centroid_location,
    ST_ConvexHull(ST_Collect(l.geom)) as activity_envelope
FROM geological.seismic_events se
JOIN geological.locations l ON se.location_id = l.id
WHERE se.event_time >= NOW() - INTERVAL '1 year'
GROUP BY DATE_TRUNC('day', se.event_time), se.event_type;

CREATE UNIQUE INDEX ON geological.mv_seismic_activity_summary (activity_date, event_type);
CREATE INDEX ON geological.mv_seismic_activity_summary USING GIST (centroid_location);
CREATE INDEX ON geological.mv_seismic_activity_summary USING GIST (activity_envelope);

-- Materialized view for deformation analysis
CREATE MATERIALIZED VIEW geological.mv_deformation_analysis AS
SELECT 
    dm.location_id,
    l.name as location_name,
    l.geom,
    DATE_TRUNC('month', dm.measurement_time) as analysis_month,
    dm.measurement_type,
    COUNT(*) as measurement_count,
    AVG(SQRT(dm.displacement_x^2 + dm.displacement_y^2 + dm.displacement_z^2)) as avg_total_displacement,
    MAX(SQRT(dm.displacement_x^2 + dm.displacement_y^2 + dm.displacement_z^2)) as max_total_displacement,
    AVG(dm.displacement_x) as avg_displacement_x,
    AVG(dm.displacement_y) as avg_displacement_y,
    AVG(dm.displacement_z) as avg_displacement_z,
    STDDEV(dm.displacement_x) as stddev_displacement_x,
    STDDEV(dm.displacement_y) as stddev_displacement_y,
    STDDEV(dm.displacement_z) as stddev_displacement_z
FROM geological.deformation_measurements dm
JOIN geological.locations l ON dm.location_id = l.id
WHERE dm.measurement_time >= NOW() - INTERVAL '2 years'
GROUP BY dm.location_id, l.name, l.geom, DATE_TRUNC('month', dm.measurement_time), dm.measurement_type;

CREATE UNIQUE INDEX ON geological.mv_deformation_analysis (location_id, analysis_month, measurement_type);
CREATE INDEX ON geological.mv_deformation_analysis USING GIST (geom);

-- Materialized view for thermal anomaly detection
CREATE MATERIALIZED VIEW geological.mv_thermal_anomalies AS
SELECT 
    tm.location_id,
    l.name as location_name,
    l.geom,
    DATE_TRUNC('week', tm.measurement_time) as analysis_week,
    tm.measurement_method,
    COUNT(*) as measurement_count,
    AVG(tm.temperature) as avg_temperature,
    MAX(tm.temperature) as max_temperature,
    MIN(tm.temperature) as min_temperature,
    STDDEV(tm.temperature) as temperature_stddev,
    CASE 
        WHEN AVG(tm.temperature) > (
            SELECT AVG(temperature) + 2 * STDDEV(temperature) 
            FROM geological.thermal_measurements 
            WHERE measurement_time >= NOW() - INTERVAL '1 year'
        ) THEN 'HIGH_ANOMALY'
        WHEN AVG(tm.temperature) > (
            SELECT AVG(temperature) + STDDEV(temperature) 
            FROM geological.thermal_measurements 
            WHERE measurement_time >= NOW() - INTERVAL '1 year'
        ) THEN 'MODERATE_ANOMALY'
        ELSE 'NORMAL'
    END as anomaly_level
FROM geological.thermal_measurements tm
JOIN geological.locations l ON tm.location_id = l.id
WHERE tm.measurement_time >= NOW() - INTERVAL '1 year'
GROUP BY tm.location_id, l.name, l.geom, DATE_TRUNC('week', tm.measurement_time), tm.measurement_method;

CREATE UNIQUE INDEX ON geological.mv_thermal_anomalies (location_id, analysis_week, measurement_method);
CREATE INDEX ON geological.mv_thermal_anomalies USING GIST (geom);
CREATE INDEX ON geological.mv_thermal_anomalies (anomaly_level);

-- Materialized view for multi-parameter geological correlation
CREATE MATERIALIZED VIEW geological.mv_geological_correlation AS
SELECT 
    l.id as location_id,
    l.name as location_name,
    l.geom,
    l.elevation,
    l.depth,
    DATE_TRUNC('month', COALESCE(se.event_time, dm.measurement_time, tm.measurement_time)) as analysis_month,
    COUNT(DISTINCT se.id) as seismic_event_count,
    AVG(se.magnitude) as avg_seismic_magnitude,
    COUNT(DISTINCT dm.id) as deformation_measurement_count,
    AVG(SQRT(dm.displacement_x^2 + dm.displacement_y^2 + dm.displacement_z^2)) as avg_total_deformation,
    COUNT(DISTINCT tm.id) as thermal_measurement_count,
    AVG(tm.temperature) as avg_temperature,
    COUNT(DISTINCT gs.id) as geochemical_sample_count
FROM geological.locations l
LEFT JOIN geological.seismic_events se ON l.id = se.location_id 
    AND se.event_time >= NOW() - INTERVAL '2 years'
LEFT JOIN geological.deformation_measurements dm ON l.id = dm.location_id 
    AND dm.measurement_time >= NOW() - INTERVAL '2 years'
LEFT JOIN geological.thermal_measurements tm ON l.id = tm.location_id 
    AND tm.measurement_time >= NOW() - INTERVAL '2 years'
LEFT JOIN geological.geochemical_samples gs ON l.id = gs.location_id 
    AND gs.sample_time >= NOW() - INTERVAL '2 years'
WHERE COALESCE(se.event_time, dm.measurement_time, tm.measurement_time, gs.sample_time) IS NOT NULL
GROUP BY l.id, l.name, l.geom, l.elevation, l.depth, 
         DATE_TRUNC('month', COALESCE(se.event_time, dm.measurement_time, tm.measurement_time));

CREATE UNIQUE INDEX ON geological.mv_geological_correlation (location_id, analysis_month);
CREATE INDEX ON geological.mv_geological_correlation USING GIST (geom);

-- ============================================================================
-- STORED PROCEDURES FOR COMMON GEOLOGICAL CALCULATIONS
-- ============================================================================

-- Function to calculate distance between geological locations
CREATE OR REPLACE FUNCTION geological.calculate_distance(
    loc1_id INTEGER,
    loc2_id INTEGER,
    distance_type TEXT DEFAULT 'euclidean'
) RETURNS DOUBLE PRECISION AS $$
DECLARE
    loc1_geom GEOMETRY;
    loc2_geom GEOMETRY;
    distance_result DOUBLE PRECISION;
BEGIN
    -- Get geometries for both locations
    SELECT geom INTO loc1_geom FROM geological.locations WHERE id = loc1_id;
    SELECT geom INTO loc2_geom FROM geological.locations WHERE id = loc2_id;
    
    IF loc1_geom IS NULL OR loc2_geom IS NULL THEN
        RETURN NULL;
    END IF;
    
    CASE distance_type
        WHEN 'euclidean' THEN
            distance_result := ST_Distance(loc1_geom, loc2_geom);
        WHEN 'spheroidal' THEN
            distance_result := ST_DistanceSpheroid(loc1_geom, loc2_geom, 'SPHEROID["WGS 84",6378137,298.257223563]');
        WHEN 'great_circle' THEN
            distance_result := ST_Distance_Sphere(loc1_geom, loc2_geom);
        ELSE
            distance_result := ST_Distance(loc1_geom, loc2_geom);
    END CASE;
    
    RETURN distance_result;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate seismic moment from magnitude
CREATE OR REPLACE FUNCTION geological.calculate_seismic_moment(
    magnitude DOUBLE PRECISION,
    magnitude_type TEXT DEFAULT 'Mw'
) RETURNS DOUBLE PRECISION AS $$
DECLARE
    moment DOUBLE PRECISION;
BEGIN
    -- Calculate seismic moment using Hanks-Kanamori relation
    -- M0 = 10^(1.5 * Mw + 9.1) for moment magnitude
    CASE magnitude_type
        WHEN 'Mw' THEN
            moment := POWER(10, 1.5 * magnitude + 9.1);
        WHEN 'Ms' THEN
            -- Convert surface wave magnitude to moment magnitude first
            moment := POWER(10, 1.5 * (0.67 * magnitude + 2.07) + 9.1);
        WHEN 'Mb' THEN
            -- Convert body wave magnitude to moment magnitude first
            moment := POWER(10, 1.5 * (0.85 * magnitude + 1.03) + 9.1);
        ELSE
            moment := POWER(10, 1.5 * magnitude + 9.1);
    END CASE;
    
    RETURN moment;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate volcanic explosivity index (VEI) estimation
CREATE OR REPLACE FUNCTION geological.estimate_vei(
    eruption_volume DOUBLE PRECISION, -- in cubic meters
    eruption_height DOUBLE PRECISION DEFAULT NULL -- in meters
) RETURNS INTEGER AS $$
DECLARE
    vei INTEGER;
BEGIN
    -- VEI calculation based on erupted volume
    IF eruption_volume < 1e4 THEN
        vei := 0;
    ELSIF eruption_volume < 1e6 THEN
        vei := 1;
    ELSIF eruption_volume < 1e7 THEN
        vei := 2;
    ELSIF eruption_volume < 1e8 THEN
        vei := 3;
    ELSIF eruption_volume < 1e9 THEN
        vei := 4;
    ELSIF eruption_volume < 1e10 THEN
        vei := 5;
    ELSIF eruption_volume < 1e11 THEN
        vei := 6;
    ELSIF eruption_volume < 1e12 THEN
        vei := 7;
    ELSE
        vei := 8;
    END IF;
    
    -- Adjust based on eruption height if provided
    IF eruption_height IS NOT NULL THEN
        IF eruption_height > 35000 AND vei < 6 THEN
            vei := 6;
        ELSIF eruption_height > 25000 AND vei < 5 THEN
            vei := 5;
        ELSIF eruption_height > 15000 AND vei < 4 THEN
            vei := 4;
        END IF;
    END IF;
    
    RETURN vei;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate magma ascent rate
CREATE OR REPLACE FUNCTION geological.calculate_magma_ascent_rate(
    depth_change DOUBLE PRECISION, -- in meters
    time_interval INTERVAL
) RETURNS DOUBLE PRECISION AS $$
DECLARE
    time_seconds DOUBLE PRECISION;
    ascent_rate DOUBLE PRECISION;
BEGIN
    -- Convert interval to seconds
    time_seconds := EXTRACT(EPOCH FROM time_interval);
    
    IF time_seconds = 0 THEN
        RETURN NULL;
    END IF;
    
    -- Calculate ascent rate in m/s
    ascent_rate := ABS(depth_change) / time_seconds;
    
    RETURN ascent_rate;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate stress drop from seismic data
CREATE OR REPLACE FUNCTION geological.calculate_stress_drop(
    seismic_moment DOUBLE PRECISION, -- in N⋅m
    rupture_radius DOUBLE PRECISION -- in meters
) RETURNS DOUBLE PRECISION AS $$
DECLARE
    stress_drop DOUBLE PRECISION;
    shear_modulus DOUBLE PRECISION := 3e10; -- Pa, typical for crustal rocks
BEGIN
    -- Brune stress drop formula: Δσ = (7/16) * (M0 / r^3) * (1/μ)
    -- where μ is shear modulus
    stress_drop := (7.0/16.0) * seismic_moment / (POWER(rupture_radius, 3) * shear_modulus);
    
    RETURN stress_drop;
END;
$$ LANGUAGE plpgsql;

-- Function to find geological events within spatial-temporal window
CREATE OR REPLACE FUNCTION geological.find_events_in_spacetime(
    center_location GEOMETRY,
    spatial_radius DOUBLE PRECISION, -- in meters
    time_center TIMESTAMP WITH TIME ZONE,
    time_window INTERVAL,
    event_types TEXT[] DEFAULT ARRAY['earthquake', 'volcanic']
) RETURNS TABLE(
    event_id INTEGER,
    event_type TEXT,
    event_time TIMESTAMP WITH TIME ZONE,
    distance_from_center DOUBLE PRECISION,
    time_diff_seconds DOUBLE PRECISION,
    magnitude DOUBLE PRECISION,
    depth DOUBLE PRECISION,
    location_name TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        se.id as event_id,
        se.event_type,
        se.event_time,
        ST_Distance(center_location, l.geom) as distance_from_center,
        EXTRACT(EPOCH FROM (se.event_time - time_center)) as time_diff_seconds,
        se.magnitude,
        se.depth,
        l.name as location_name
    FROM geological.seismic_events se
    JOIN geological.locations l ON se.location_id = l.id
    WHERE ST_DWithin(center_location, l.geom, spatial_radius)
        AND se.event_time BETWEEN (time_center - time_window) AND (time_center + time_window)
        AND se.event_type = ANY(event_types)
    ORDER BY se.event_time;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate geological hazard probability
CREATE OR REPLACE FUNCTION geological.calculate_hazard_probability(
    location_geom GEOMETRY,
    hazard_type TEXT,
    time_horizon INTERVAL DEFAULT '1 year'::INTERVAL
) RETURNS DOUBLE PRECISION AS $$
DECLARE
    historical_count INTEGER;
    total_time_years DOUBLE PRECISION;
    probability DOUBLE PRECISION;
    search_radius DOUBLE PRECISION := 50000; -- 50km radius
BEGIN
    -- Get historical event count within radius
    SELECT COUNT(*) INTO historical_count
    FROM geological.seismic_events se
    JOIN geological.locations l ON se.location_id = l.id
    WHERE ST_DWithin(location_geom, l.geom, search_radius)
        AND se.event_type = hazard_type
        AND se.event_time >= NOW() - INTERVAL '50 years'; -- 50 years of historical data
    
    -- Calculate total observation time in years
    total_time_years := 50.0;
    
    -- Calculate probability using Poisson distribution
    -- P = 1 - exp(-λt) where λ = historical_count / total_time_years
    IF historical_count = 0 THEN
        probability := 0.0;
    ELSE
        probability := 1.0 - EXP(-(historical_count::DOUBLE PRECISION / total_time_years) * 
                                EXTRACT(EPOCH FROM time_horizon) / (365.25 * 24 * 3600));
    END IF;
    
    RETURN LEAST(probability, 1.0); -- Cap at 100%
END;
$$ LANGUAGE plpgsql;

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION geological.refresh_all_materialized_views()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY geological.mv_seismic_activity_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY geological.mv_deformation_analysis;
    REFRESH MATERIALIZED VIEW CONCURRENTLY geological.mv_thermal_anomalies;
    REFRESH MATERIALIZED VIEW CONCURRENTLY geological.mv_geological_correlation;
    
    -- Log the refresh
    INSERT INTO geological.system_log (log_time, log_level, message)
    VALUES (NOW(), 'INFO', 'All geological materialized views refreshed');
END;
$$ LANGUAGE plpgsql;

-- Create system log table for tracking operations
CREATE TABLE IF NOT EXISTS geological.system_log (
    id SERIAL PRIMARY KEY,
    log_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    log_level VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB
);

-- ============================================================================
-- COMPLEX SPATIAL-TEMPORAL QUERIES FOR GEOLOGICAL PROCESSES
-- ============================================================================

-- Create view for complex spatial-temporal seismic analysis
CREATE OR REPLACE VIEW geological.v_seismic_spatiotemporal_analysis AS
SELECT 
    se.id,
    se.event_time,
    se.magnitude,
    se.depth,
    se.event_type,
    l.name as location_name,
    l.geom,
    -- Spatial clustering analysis
    ST_ClusterDBSCAN(l.geom, 10000, 3) OVER() as spatial_cluster_id,
    -- Temporal clustering (events within 24 hours)
    CASE 
        WHEN LAG(se.event_time) OVER (ORDER BY se.event_time) IS NULL THEN 1
        WHEN se.event_time - LAG(se.event_time) OVER (ORDER BY se.event_time) <= INTERVAL '24 hours' THEN 0
        ELSE 1
    END as temporal_cluster_start,
    -- Distance to previous event
    ST_Distance(
        l.geom, 
        LAG(l.geom) OVER (ORDER BY se.event_time)
    ) as distance_to_previous,
    -- Time since previous event
    EXTRACT(EPOCH FROM (
        se.event_time - LAG(se.event_time) OVER (ORDER BY se.event_time)
    )) / 3600.0 as hours_since_previous,
    -- Magnitude difference from previous
    se.magnitude - LAG(se.magnitude) OVER (ORDER BY se.event_time) as magnitude_diff
FROM geological.seismic_events se
JOIN geological.locations l ON se.location_id = l.id
WHERE se.event_time >= NOW() - INTERVAL '1 year';

-- Create view for deformation velocity analysis
CREATE OR REPLACE VIEW geological.v_deformation_velocity_analysis AS
SELECT 
    dm1.location_id,
    l.name as location_name,
    l.geom,
    dm1.measurement_time as time1,
    dm2.measurement_time as time2,
    EXTRACT(EPOCH FROM (dm2.measurement_time - dm1.measurement_time)) / (24.0 * 3600.0) as days_between,
    -- Velocity calculations (mm/day)
    (dm2.displacement_x - dm1.displacement_x) * 1000 / 
        EXTRACT(EPOCH FROM (dm2.measurement_time - dm1.measurement_time)) * 86400 as velocity_x_mm_day,
    (dm2.displacement_y - dm1.displacement_y) * 1000 / 
        EXTRACT(EPOCH FROM (dm2.measurement_time - dm1.measurement_time)) * 86400 as velocity_y_mm_day,
    (dm2.displacement_z - dm1.displacement_z) * 1000 / 
        EXTRACT(EPOCH FROM (dm2.measurement_time - dm1.measurement_time)) * 86400 as velocity_z_mm_day,
    -- Total velocity magnitude
    SQRT(
        POWER((dm2.displacement_x - dm1.displacement_x), 2) +
        POWER((dm2.displacement_y - dm1.displacement_y), 2) +
        POWER((dm2.displacement_z - dm1.displacement_z), 2)
    ) * 1000 / EXTRACT(EPOCH FROM (dm2.measurement_time - dm1.measurement_time)) * 86400 as total_velocity_mm_day
FROM geological.deformation_measurements dm1
JOIN geological.deformation_measurements dm2 ON dm1.location_id = dm2.location_id
    AND dm2.measurement_time > dm1.measurement_time
    AND dm2.measurement_time <= dm1.measurement_time + INTERVAL '30 days'
JOIN geological.locations l ON dm1.location_id = l.id
WHERE dm1.measurement_time >= NOW() - INTERVAL '1 year';

-- Create automated refresh schedule for materialized views
CREATE OR REPLACE FUNCTION geological.schedule_materialized_view_refresh()
RETURNS VOID AS $$
BEGIN
    -- This would typically be called by a cron job or scheduler
    -- Refresh views during low-usage hours
    PERFORM geological.refresh_all_materialized_views();
END;
$$ LANGUAGE plpgsql;

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

-- Grant execute permissions on functions
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA geological TO prometheus;