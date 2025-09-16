-- Multi-Chamber Geometry Database Schema
-- Advanced database schema for storing multi-chamber volcanic system geometries,
-- connectivity analysis, and temporal evolution data

-- Enable PostGIS extension for spatial data
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- ============================================================================
-- Chamber Geometries Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS chamber_geometries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(100) NOT NULL,
    volcanic_system_id UUID,
    
    -- Spatial coordinates (in km from reference point)
    center_coordinates GEOMETRY(POINT Z, 4326) NOT NULL,
    center_x DOUBLE PRECISION NOT NULL,
    center_y DOUBLE PRECISION NOT NULL,
    center_z DOUBLE PRECISION NOT NULL, -- Depth (positive downward)
    
    -- Geometric properties
    volume DOUBLE PRECISION NOT NULL CHECK (volume > 0), -- km³
    shape_type VARCHAR(50) NOT NULL DEFAULT 'ellipsoid',
    shape_parameters JSONB NOT NULL DEFAULT '{}',
    orientation JSONB DEFAULT '{}', -- Strike, dip, plunge angles
    aspect_ratios JSONB DEFAULT '{}', -- a/b, b/c ratios
    
    -- Surface mesh data (stored as binary)
    surface_mesh BYTEA,
    mesh_format VARCHAR(20) DEFAULT 'numpy_array',
    
    -- Uncertainty and validation
    uncertainty_bounds JSONB,
    confidence_level DOUBLE PRECISION DEFAULT 0.95 CHECK (confidence_level BETWEEN 0 AND 1),
    reconstruction_method VARCHAR(100) NOT NULL,
    validation_score DOUBLE PRECISION DEFAULT 0.0 CHECK (validation_score BETWEEN 0 AND 1),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    data_source VARCHAR(100),
    processing_version VARCHAR(20),
    
    -- Constraints
    UNIQUE(chamber_id, created_at)
);

-- Indexes for performance
CREATE INDEX idx_chamber_geometries_chamber_id ON chamber_geometries(chamber_id);
CREATE INDEX idx_chamber_geometries_system_id ON chamber_geometries(volcanic_system_id);
CREATE INDEX idx_chamber_geometries_spatial ON chamber_geometries USING GIST(center_coordinates);
CREATE INDEX idx_chamber_geometries_depth ON chamber_geometries(center_z);
CREATE INDEX idx_chamber_geometries_volume ON chamber_geometries(volume);
CREATE INDEX idx_chamber_geometries_created_at ON chamber_geometries(created_at);

-- ============================================================================
-- Seismic Tomography Data Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS seismic_tomography_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id VARCHAR(100) NOT NULL,
    
    -- Spatial extent
    spatial_bounds GEOMETRY(POLYGON, 4326) NOT NULL,
    depth_range NUMRANGE NOT NULL, -- Min and max depth in km
    
    -- Data properties
    velocity_model_path TEXT NOT NULL, -- Path to velocity model file
    coordinates_path TEXT NOT NULL, -- Path to coordinate arrays
    uncertainty_path TEXT, -- Path to uncertainty data
    
    -- Resolution information
    spatial_resolution JSONB NOT NULL, -- x, y, z resolution in km
    temporal_resolution INTERVAL, -- For time-lapse data
    
    -- Acquisition parameters
    acquisition_params JSONB DEFAULT '{}',
    quality_metrics JSONB DEFAULT '{}',
    
    -- Processing information
    processing_params JSONB DEFAULT '{}',
    processing_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    data_format VARCHAR(50) DEFAULT 'numpy',
    file_size_mb DOUBLE PRECISION,
    
    UNIQUE(dataset_id)
);

CREATE INDEX idx_seismic_data_spatial ON seismic_tomography_data USING GIST(spatial_bounds);
CREATE INDEX idx_seismic_data_depth ON seismic_tomography_data USING GIST(depth_range);

-- ============================================================================
-- Gravity and Magnetic Data Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS gravity_magnetic_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id VARCHAR(100) NOT NULL,
    
    -- Spatial coverage
    spatial_bounds GEOMETRY(POLYGON, 4326) NOT NULL,
    station_locations GEOMETRY(MULTIPOINT, 4326),
    
    -- Data arrays (stored as file paths)
    gravity_anomaly_path TEXT,
    magnetic_anomaly_path TEXT,
    coordinates_path TEXT NOT NULL,
    elevation_path TEXT,
    uncertainty_path TEXT,
    
    -- Regional field parameters
    regional_field JSONB DEFAULT '{}',
    
    -- Processing parameters
    processing_params JSONB DEFAULT '{}',
    corrections_applied TEXT[], -- Array of correction types
    
    -- Data quality
    data_quality JSONB DEFAULT '{}',
    survey_date DATE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    data_source VARCHAR(100),
    survey_type VARCHAR(50), -- 'ground', 'airborne', 'satellite'
    
    UNIQUE(dataset_id)
);

CREATE INDEX idx_gravity_magnetic_spatial ON gravity_magnetic_data USING GIST(spatial_bounds);
CREATE INDEX idx_gravity_magnetic_stations ON gravity_magnetic_data USING GIST(station_locations);

-- ============================================================================
-- Deformation Data Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS deformation_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id VARCHAR(100) NOT NULL,
    
    -- Spatial coverage
    spatial_bounds GEOMETRY(POLYGON, 4326) NOT NULL,
    station_locations GEOMETRY(MULTIPOINT, 4326),
    
    -- Displacement data (file paths)
    displacement_x_path TEXT,
    displacement_y_path TEXT,
    displacement_z_path TEXT,
    coordinates_path TEXT NOT NULL,
    time_series_path TEXT NOT NULL,
    uncertainty_path TEXT,
    
    -- Data characteristics
    data_type VARCHAR(50) NOT NULL, -- 'InSAR', 'GPS', 'tilt', 'strain'
    processing_level VARCHAR(50) NOT NULL, -- 'raw', 'processed', 'corrected'
    temporal_span TSRANGE, -- Time range of observations
    
    -- Processing information
    atmospheric_correction BOOLEAN DEFAULT FALSE,
    reference_frame VARCHAR(20) DEFAULT 'ITRF2014',
    processing_params JSONB DEFAULT '{}',
    
    -- Quality metrics
    coherence_threshold DOUBLE PRECISION,
    unwrapping_errors INTEGER DEFAULT 0,
    data_gaps INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    satellite_mission VARCHAR(50), -- For InSAR data
    processing_software VARCHAR(100),
    
    UNIQUE(dataset_id)
);

CREATE INDEX idx_deformation_spatial ON deformation_data USING GIST(spatial_bounds);
CREATE INDEX idx_deformation_temporal ON deformation_data USING GIST(temporal_span);
CREATE INDEX idx_deformation_type ON deformation_data(data_type);

-- ============================================================================
-- Chamber Connectivity Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS chamber_connectivity (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connection_id VARCHAR(100) NOT NULL,
    
    -- Connected chambers
    from_chamber_id VARCHAR(100) NOT NULL,
    to_chamber_id VARCHAR(100) NOT NULL,
    
    -- Pathway geometry
    pathway_geometry GEOMETRY(LINESTRING Z, 4326),
    pathway_coordinates BYTEA, -- Detailed 3D pathway as binary array
    
    -- Physical properties
    conductivity DOUBLE PRECISION NOT NULL CHECK (conductivity > 0), -- m²/(Pa·s)
    cross_sectional_area DOUBLE PRECISION NOT NULL CHECK (cross_sectional_area > 0), -- m²
    tortuosity DOUBLE PRECISION DEFAULT 1.0 CHECK (tortuosity >= 1.0),
    pressure_gradient DOUBLE PRECISION, -- Pa/m
    flow_resistance DOUBLE PRECISION, -- Pa·s/m³
    
    -- Connectivity metrics
    connectivity_strength DOUBLE PRECISION NOT NULL CHECK (connectivity_strength BETWEEN 0 AND 1),
    pathway_type VARCHAR(50) DEFAULT 'direct', -- 'direct', 'tortuous', 'network'
    
    -- Uncertainty quantification
    uncertainty JSONB DEFAULT '{}',
    confidence_level DOUBLE PRECISION DEFAULT 0.95,
    
    -- Analysis metadata
    analysis_method VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(connection_id),
    CHECK(from_chamber_id != to_chamber_id)
);

CREATE INDEX idx_connectivity_from_chamber ON chamber_connectivity(from_chamber_id);
CREATE INDEX idx_connectivity_to_chamber ON chamber_connectivity(to_chamber_id);
CREATE INDEX idx_connectivity_strength ON chamber_connectivity(connectivity_strength DESC);
CREATE INDEX idx_connectivity_pathway ON chamber_connectivity USING GIST(pathway_geometry);

-- ============================================================================
-- Geometry Evolution Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS geometry_evolution (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    evolution_id VARCHAR(100) NOT NULL,
    chamber_id VARCHAR(100) NOT NULL,
    
    -- Time series information
    time_series TIMESTAMP WITH TIME ZONE[] NOT NULL,
    observation_count INTEGER NOT NULL CHECK (observation_count > 1),
    time_span INTERVAL NOT NULL,
    
    -- Evolution data (stored as file paths for large arrays)
    geometry_history_path TEXT NOT NULL,
    volume_evolution_path TEXT NOT NULL,
    shape_evolution_path TEXT,
    migration_path_path TEXT,
    
    -- Evolution metrics
    growth_rate DOUBLE PRECISION, -- km³/year
    deformation_rate JSONB DEFAULT '{}', -- Shape change rates
    migration_velocity DOUBLE PRECISION, -- km/year
    
    -- Evolution drivers and patterns
    evolution_drivers TEXT[] DEFAULT '{}',
    dominant_pattern VARCHAR(100), -- 'growth', 'migration', 'deformation'
    
    -- Predictions
    prediction_horizon INTERVAL,
    confidence_intervals JSONB DEFAULT '{}',
    prediction_accuracy DOUBLE PRECISION, -- For validated predictions
    
    -- Analysis metadata
    analysis_method VARCHAR(100),
    model_version VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(evolution_id)
);

CREATE INDEX idx_evolution_chamber_id ON geometry_evolution(chamber_id);
CREATE INDEX idx_evolution_time_span ON geometry_evolution(time_span);
CREATE INDEX idx_evolution_growth_rate ON geometry_evolution(growth_rate);

-- ============================================================================
-- Inversion Results Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS inversion_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    inversion_id VARCHAR(100) NOT NULL,
    
    -- Input data references
    seismic_data_id UUID REFERENCES seismic_tomography_data(id),
    gravity_data_id UUID REFERENCES gravity_magnetic_data(id),
    deformation_data_id UUID REFERENCES deformation_data(id),
    
    -- Inversion parameters
    method VARCHAR(100) NOT NULL, -- 'least_squares', 'bayesian', 'ml', 'joint'
    parameters JSONB NOT NULL DEFAULT '{}',
    convergence_criteria JSONB DEFAULT '{}',
    
    -- Results
    chamber_count INTEGER NOT NULL CHECK (chamber_count >= 0),
    total_volume DOUBLE PRECISION, -- Total chamber volume
    misfit_value DOUBLE PRECISION,
    convergence_achieved BOOLEAN DEFAULT FALSE,
    iterations_used INTEGER,
    
    -- Quality metrics
    data_fit_quality DOUBLE PRECISION CHECK (data_fit_quality BETWEEN 0 AND 1),
    model_complexity DOUBLE PRECISION,
    uncertainty_level DOUBLE PRECISION,
    
    -- Computational metrics
    computation_time INTERVAL,
    memory_usage_mb DOUBLE PRECISION,
    cpu_cores_used INTEGER,
    
    -- Metadata
    software_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    analyst VARCHAR(100),
    
    UNIQUE(inversion_id)
);

CREATE INDEX idx_inversion_method ON inversion_results(method);
CREATE INDEX idx_inversion_quality ON inversion_results(data_fit_quality DESC);
CREATE INDEX idx_inversion_created_at ON inversion_results(created_at);

-- ============================================================================
-- Validation Results Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS validation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    validation_id VARCHAR(100) NOT NULL,
    
    -- Validation setup
    chamber_geometry_id UUID REFERENCES chamber_geometries(id),
    reference_system VARCHAR(100), -- Known volcanic system name
    validation_type VARCHAR(50), -- 'synthetic', 'known_system', 'cross_validation'
    
    -- Validation metrics
    volume_accuracy DOUBLE PRECISION CHECK (volume_accuracy BETWEEN 0 AND 1),
    depth_accuracy DOUBLE PRECISION CHECK (depth_accuracy BETWEEN 0 AND 1),
    position_accuracy DOUBLE PRECISION CHECK (position_accuracy BETWEEN 0 AND 1),
    shape_accuracy DOUBLE PRECISION CHECK (shape_accuracy BETWEEN 0 AND 1),
    
    -- Overall scores
    overall_accuracy DOUBLE PRECISION CHECK (overall_accuracy BETWEEN 0 AND 1),
    confidence_score DOUBLE PRECISION CHECK (confidence_score BETWEEN 0 AND 1),
    
    -- Detailed metrics
    validation_metrics JSONB DEFAULT '{}',
    error_analysis JSONB DEFAULT '{}',
    
    -- Reference data
    reference_parameters JSONB DEFAULT '{}',
    literature_source TEXT,
    
    -- Validation metadata
    validation_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    validator VARCHAR(100),
    notes TEXT,
    
    UNIQUE(validation_id)
);

CREATE INDEX idx_validation_system ON validation_results(reference_system);
CREATE INDEX idx_validation_accuracy ON validation_results(overall_accuracy DESC);
CREATE INDEX idx_validation_chamber ON validation_results(chamber_geometry_id);

-- ============================================================================
-- Processing Jobs Table (for tracking long-running computations)
-- ============================================================================

CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id VARCHAR(100) NOT NULL,
    
    -- Job information
    job_type VARCHAR(100) NOT NULL, -- 'reconstruction', 'connectivity', 'evolution'
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    
    -- Input/output references
    input_data_ids UUID[] DEFAULT '{}',
    output_data_ids UUID[] DEFAULT '{}',
    
    -- Processing parameters
    parameters JSONB DEFAULT '{}',
    resource_requirements JSONB DEFAULT '{}',
    
    -- Execution tracking
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration INTERVAL,
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage BETWEEN 0 AND 100),
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Resource usage
    cpu_time INTERVAL,
    memory_peak_mb DOUBLE PRECISION,
    disk_usage_mb DOUBLE PRECISION,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100),
    worker_node VARCHAR(100),
    
    UNIQUE(job_id)
);

CREATE INDEX idx_processing_jobs_status ON processing_jobs(status);
CREATE INDEX idx_processing_jobs_type ON processing_jobs(job_type);
CREATE INDEX idx_processing_jobs_priority ON processing_jobs(priority DESC);
CREATE INDEX idx_processing_jobs_created_at ON processing_jobs(created_at);

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- View for chamber summary with latest geometries
CREATE OR REPLACE VIEW chamber_summary AS
SELECT DISTINCT ON (chamber_id)
    chamber_id,
    center_x,
    center_y,
    center_z,
    volume,
    shape_type,
    confidence_level,
    validation_score,
    reconstruction_method,
    created_at
FROM chamber_geometries
ORDER BY chamber_id, created_at DESC;

-- View for active chamber connections
CREATE OR REPLACE VIEW active_connections AS
SELECT 
    cc.*,
    cg1.center_x as from_x,
    cg1.center_y as from_y,
    cg1.center_z as from_z,
    cg2.center_x as to_x,
    cg2.center_y as to_y,
    cg2.center_z as to_z
FROM chamber_connectivity cc
JOIN chamber_summary cg1 ON cc.from_chamber_id = cg1.chamber_id
JOIN chamber_summary cg2 ON cc.to_chamber_id = cg2.chamber_id
WHERE cc.connectivity_strength > 0.1;

-- View for recent processing activity
CREATE OR REPLACE VIEW recent_processing AS
SELECT 
    job_type,
    status,
    COUNT(*) as job_count,
    AVG(EXTRACT(EPOCH FROM duration)) as avg_duration_seconds,
    MAX(completed_at) as last_completed
FROM processing_jobs
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY job_type, status
ORDER BY job_type, status;

-- ============================================================================
-- Functions for Data Management
-- ============================================================================

-- Function to update chamber geometry timestamp
CREATE OR REPLACE FUNCTION update_chamber_geometry_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for automatic timestamp updates
CREATE TRIGGER trigger_update_chamber_geometry_timestamp
    BEFORE UPDATE ON chamber_geometries
    FOR EACH ROW
    EXECUTE FUNCTION update_chamber_geometry_timestamp();

-- Function to calculate chamber network statistics
CREATE OR REPLACE FUNCTION calculate_network_statistics(system_id UUID DEFAULT NULL)
RETURNS TABLE(
    total_chambers INTEGER,
    total_volume DOUBLE PRECISION,
    avg_depth DOUBLE PRECISION,
    connection_count INTEGER,
    network_density DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    WITH chamber_stats AS (
        SELECT 
            COUNT(*) as chamber_count,
            SUM(volume) as total_vol,
            AVG(center_z) as avg_depth_val
        FROM chamber_summary cs
        WHERE system_id IS NULL OR cs.volcanic_system_id = system_id
    ),
    connection_stats AS (
        SELECT COUNT(*) as conn_count
        FROM chamber_connectivity cc
        WHERE connectivity_strength > 0.1
    )
    SELECT 
        cs.chamber_count::INTEGER,
        cs.total_vol,
        cs.avg_depth_val,
        conn.conn_count::INTEGER,
        CASE 
            WHEN cs.chamber_count > 1 THEN 
                conn.conn_count::DOUBLE PRECISION / (cs.chamber_count * (cs.chamber_count - 1) / 2)
            ELSE 0.0
        END as density
    FROM chamber_stats cs, connection_stats conn;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old processing jobs
CREATE OR REPLACE FUNCTION cleanup_old_jobs(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM processing_jobs
    WHERE status IN ('completed', 'failed')
    AND completed_at < NOW() - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Initial Data and Configuration
-- ============================================================================

-- Insert default configuration
INSERT INTO chamber_geometries (chamber_id, center_x, center_y, center_z, volume, shape_type, reconstruction_method, validation_score)
VALUES ('example_chamber', 0.0, 0.0, 8.0, 100.0, 'ellipsoid', 'synthetic', 1.0)
ON CONFLICT (chamber_id, created_at) DO NOTHING;

-- Create indexes for performance optimization
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chamber_geometries_composite 
ON chamber_geometries(chamber_id, created_at DESC, validation_score DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_connectivity_composite
ON chamber_connectivity(connectivity_strength DESC, pathway_type, created_at DESC);

-- Grant permissions (adjust as needed for your security model)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO volcanic_analysis_role;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO volcanic_analysis_role;

COMMENT ON TABLE chamber_geometries IS 'Stores 3D chamber geometries with uncertainty quantification';
COMMENT ON TABLE chamber_connectivity IS 'Stores connectivity analysis between chambers';
COMMENT ON TABLE geometry_evolution IS 'Tracks temporal evolution of chamber geometries';
COMMENT ON TABLE validation_results IS 'Stores validation results against known systems';
COMMENT ON TABLE processing_jobs IS 'Tracks long-running computational jobs';