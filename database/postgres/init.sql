-- PostgreSQL initialization script for Prometheus AI Platform
-- Creates the basic schema for geological data storage

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS geological;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS users;

-- Volcanoes table
CREATE TABLE IF NOT EXISTS geological.volcanoes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation FLOAT,
    volcano_type VARCHAR(100),
    last_eruption DATE,
    hazard_level INTEGER DEFAULT 1,
    country VARCHAR(100),
    region VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Geological units table
CREATE TABLE IF NOT EXISTS geological.geological_units (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    age_min FLOAT,
    age_max FLOAT,
    lithology VARCHAR(100),
    geometry GEOMETRY(MULTIPOLYGON, 4326),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seismic events table
CREATE TABLE IF NOT EXISTS monitoring.seismic_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_time TIMESTAMP NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    depth FLOAT NOT NULL,
    magnitude FLOAT NOT NULL,
    magnitude_type VARCHAR(10),
    event_type VARCHAR(50),
    volcano_id UUID REFERENCES geological.volcanoes(id),
    network VARCHAR(20),
    station_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- GPS stations table
CREATE TABLE IF NOT EXISTS monitoring.gps_stations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    station_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(255),
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation FLOAT,
    volcano_id UUID REFERENCES geological.volcanoes(id),
    installation_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users table
CREATE TABLE IF NOT EXISTS users.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    organization VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_volcanoes_location ON geological.volcanoes USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_geological_units_geometry ON geological.geological_units USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_seismic_events_location ON monitoring.seismic_events USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_seismic_events_time ON monitoring.seismic_events (event_time);
CREATE INDEX IF NOT EXISTS idx_seismic_events_volcano ON monitoring.seismic_events (volcano_id);
CREATE INDEX IF NOT EXISTS idx_gps_stations_location ON monitoring.gps_stations USING GIST (location);

-- Insert sample data
INSERT INTO geological.volcanoes (name, location, elevation, volcano_type, country, region) VALUES
    ('Mount St. Helens', ST_SetSRID(ST_MakePoint(-122.1944, 46.1912), 4326), 2549, 'Stratovolcano', 'USA', 'Washington'),
    ('Kilauea', ST_SetSRID(ST_MakePoint(-155.2920, 19.4069), 4326), 1247, 'Shield', 'USA', 'Hawaii'),
    ('Yellowstone', ST_SetSRID(ST_MakePoint(-110.5885, 44.4280), 4326), 2805, 'Caldera', 'USA', 'Wyoming'),
    ('Mount Vesuvius', ST_SetSRID(ST_MakePoint(14.4261, 40.8210), 4326), 1281, 'Stratovolcano', 'Italy', 'Campania'),
    ('Mount Fuji', ST_SetSRID(ST_MakePoint(138.7274, 35.3606), 4326), 3776, 'Stratovolcano', 'Japan', 'Honshu')
ON CONFLICT DO NOTHING;

-- Insert sample user
INSERT INTO users.users (username, email, password_hash, role, organization) VALUES
    ('admin', 'admin@prometheus.ai', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.s5uO.G', 'admin', 'Prometheus AI Platform')
ON CONFLICT DO NOTHING;

-- Create functions for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_volcanoes_updated_at BEFORE UPDATE ON geological.volcanoes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
-- Digit
al Elevation Models table for DEM caching
CREATE TABLE IF NOT EXISTS geological.digital_elevation_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    center_point GEOMETRY(POINT, 4326) NOT NULL,
    radius_km FLOAT NOT NULL,
    dem_data BYTEA NOT NULL,
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    source_combination TEXT,
    processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolution_m FLOAT,
    bbox GEOMETRY(POLYGON, 4326),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint to prevent duplicate cache entries
    UNIQUE(center_point, radius_km)
);

-- Indexes for DEM table
CREATE INDEX IF NOT EXISTS idx_dem_center_point ON geological.digital_elevation_models USING GIST (center_point);
CREATE INDEX IF NOT EXISTS idx_dem_bbox ON geological.digital_elevation_models USING GIST (bbox);
CREATE INDEX IF NOT EXISTS idx_dem_quality_score ON geological.digital_elevation_models (quality_score);
CREATE INDEX IF NOT EXISTS idx_dem_processing_timestamp ON geological.digital_elevation_models (processing_timestamp);
CREATE INDEX IF NOT EXISTS idx_dem_source_combination ON geological.digital_elevation_models (source_combination);

-- Trigger for updating timestamps
CREATE TRIGGER update_dem_updated_at BEFORE UPDATE ON geological.digital_elevation_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();