-- Surface Deformation Correlation System Database Schema
-- This schema supports the advanced surface deformation correlation capabilities
-- for sill emplacement analysis including InSAR, GPS, tilt, and strain data

-- Enable PostGIS extension for geospatial data
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- InSAR-Sill Correlation Results Table
CREATE TABLE IF NOT EXISTS insar_sill_correlations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_id VARCHAR(100) NOT NULL,
    dataset_key VARCHAR(200) NOT NULL,
    satellite VARCHAR(50),
    track INTEGER,
    frame INTEGER,
    acquisition_date TIMESTAMP,
    look_angle FLOAT,
    heading_angle FLOAT,
    spatial_resolution FLOAT,
    coherence_threshold FLOAT,
    correlation_coefficient FLOAT NOT NULL,
    weighted_correlation FLOAT,
    rms_misfit FLOAT,
    weighted_rms FLOAT,
    reduced_chi_squared FLOAT,
    valid_pixels INTEGER,
    statistical_significance FLOAT,
    processing_parameters JSONB,
    quality_metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(sill_id, dataset_key)
);

-- GPS-Sill Correlation Results Table
CREATE TABLE IF NOT EXISTS gps_sill_correlations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_id VARCHAR(100) NOT NULL,
    component VARCHAR(10) NOT NULL CHECK (component IN ('east', 'north', 'up')),
    num_stations INTEGER,
    analysis_radius FLOAT,
    correlation_coefficient FLOAT NOT NULL,
    weighted_correlation FLOAT,
    rms_misfit FLOAT,
    weighted_rms FLOAT,
    reduced_chi_squared FLOAT,
    statistical_significance FLOAT,
    distance_correlation JSONB,
    velocity_correlation JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(sill_id, component)
);

-- GPS Station Data Table
CREATE TABLE IF NOT EXISTS gps_stations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id VARCHAR(50) UNIQUE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation FLOAT,
    monument_type VARCHAR(50),
    antenna_type VARCHAR(100),
    installation_date DATE,
    operational_status VARCHAR(20) DEFAULT 'active',
    data_quality_flags TEXT[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- GPS Deformation Measurements Table
CREATE TABLE IF NOT EXISTS gps_deformation_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id VARCHAR(50) REFERENCES gps_stations(station_id),
    measurement_date TIMESTAMP NOT NULL,
    displacement_east FLOAT,
    displacement_north FLOAT,
    displacement_up FLOAT,
    velocity_east FLOAT,
    velocity_north FLOAT,
    velocity_up FLOAT,
    uncertainty_east FLOAT,
    uncertainty_north FLOAT,
    uncertainty_up FLOAT,
    correlation_matrix FLOAT[],
    measurement_type VARCHAR(20) CHECK (measurement_type IN ('campaign', 'continuous')),
    processing_strategy VARCHAR(50),
    reference_frame VARCHAR(20),
    environmental_corrections JSONB,
    data_quality JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tilt/Strain Instruments Table
CREATE TABLE IF NOT EXISTS tilt_strain_instruments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instrument_id VARCHAR(50) UNIQUE NOT NULL,
    instrument_type VARCHAR(30) CHECK (instrument_type IN ('tiltmeter', 'strainmeter', 'borehole_strainmeter')),
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation FLOAT,
    installation_depth FLOAT,
    installation_date DATE,
    calibration_date DATE,
    operational_status VARCHAR(20) DEFAULT 'active',
    sensitivity FLOAT,
    measurement_range FLOAT,
    temperature_compensation BOOLEAN DEFAULT false,
    pressure_compensation BOOLEAN DEFAULT false,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tilt/Strain Measurements Table
CREATE TABLE IF NOT EXISTS tilt_strain_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instrument_id VARCHAR(50) REFERENCES tilt_strain_instruments(instrument_id),
    measurement_date TIMESTAMP NOT NULL,
    tilt_x FLOAT, -- microradians
    tilt_y FLOAT, -- microradians
    strain_tensor FLOAT[], -- 2x2 tensor flattened
    shear_strain FLOAT,
    volumetric_strain FLOAT,
    uncertainty_tilt_x FLOAT,
    uncertainty_tilt_y FLOAT,
    uncertainty_strain FLOAT[],
    temperature_correction FLOAT,
    pressure_correction FLOAT,
    drift_correction FLOAT,
    data_quality_flags TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tilt/Strain-Sill Correlation Results Table
CREATE TABLE IF NOT EXISTS tilt_strain_sill_correlations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_id VARCHAR(100) NOT NULL,
    data_type VARCHAR(10) CHECK (data_type IN ('tilt', 'strain', 'combined')),
    component VARCHAR(20),
    num_instruments INTEGER,
    analysis_radius FLOAT,
    correlation_coefficient FLOAT NOT NULL,
    weighted_correlation FLOAT,
    rms_misfit FLOAT,
    depth_weighting_applied BOOLEAN DEFAULT false,
    statistical_significance FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(sill_id, data_type, component)
);

-- Sill Forward Model Results Table
CREATE TABLE IF NOT EXISTS sill_forward_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_id VARCHAR(100) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    observation_points INTEGER,
    elastic_parameters JSONB,
    model_parameters JSONB,
    deformation_components TEXT[] DEFAULT ARRAY['east', 'north', 'up'],
    max_displacement FLOAT,
    model_extent GEOMETRY(POLYGON, 4326),
    computational_time FLOAT, -- seconds
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Sill Parameter Inversion Results Table
CREATE TABLE IF NOT EXISTS sill_parameter_inversions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_id VARCHAR(100) UNIQUE NOT NULL,
    center_lon FLOAT NOT NULL,
    center_lat FLOAT NOT NULL,
    depth FLOAT NOT NULL,
    length FLOAT NOT NULL,
    width FLOAT NOT NULL,
    thickness FLOAT,
    strike FLOAT NOT NULL,
    dip FLOAT NOT NULL,
    opening FLOAT NOT NULL,
    pressure_change FLOAT,
    optimization_method VARCHAR(50),
    max_iterations INTEGER,
    convergence_tolerance FLOAT,
    rms_misfit FLOAT,
    reduced_chi_squared FLOAT,
    aic FLOAT, -- Akaike Information Criterion
    bic FLOAT, -- Bayesian Information Criterion
    variance_reduction FLOAT,
    success BOOLEAN NOT NULL,
    iterations_used INTEGER,
    function_evaluations INTEGER,
    computational_time FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Parameter Uncertainty Estimates Table
CREATE TABLE IF NOT EXISTS sill_parameter_uncertainties (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_id VARCHAR(100) NOT NULL,
    parameter_name VARCHAR(50) NOT NULL,
    best_estimate FLOAT,
    lower_bound FLOAT,
    upper_bound FLOAT,
    confidence_level FLOAT DEFAULT 0.95,
    uncertainty_method VARCHAR(50), -- 'hessian', 'bootstrap', 'monte_carlo'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(sill_id, parameter_name, uncertainty_method)
);

-- Temporal Deformation Analysis Results Table
CREATE TABLE IF NOT EXISTS temporal_deformation_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_id VARCHAR(100) NOT NULL,
    data_type VARCHAR(20) CHECK (data_type IN ('insar', 'gps', 'tilt_strain')),
    component VARCHAR(20),
    time_window_days INTEGER,
    num_measurements INTEGER,
    trend_slope FLOAT, -- units per day
    trend_acceleration FLOAT, -- units per day^2
    trend_significance FLOAT,
    num_change_points INTEGER,
    change_point_times FLOAT[], -- days since start
    velocity_mean FLOAT,
    velocity_std FLOAT,
    growth_correlation FLOAT,
    temporal_correlation_with_sill FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(sill_id, data_type, component)
);

-- Correlation Validation Results Table
CREATE TABLE IF NOT EXISTS sill_correlation_validations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id VARCHAR(100) UNIQUE NOT NULL,
    sill_id VARCHAR(100),
    num_events INTEGER,
    mean_validation_score FLOAT,
    spatial_accuracy FLOAT,
    temporal_accuracy FLOAT,
    magnitude_accuracy FLOAT,
    pattern_accuracy FLOAT,
    overall_validation_score FLOAT,
    validation_method VARCHAR(50),
    validation_criteria JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Monitored Sill Intrusion Events Table
CREATE TABLE IF NOT EXISTS monitored_sill_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id VARCHAR(100) UNIQUE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    depth FLOAT,
    event_time TIMESTAMP NOT NULL,
    duration_hours FLOAT,
    magnitude FLOAT,
    event_type VARCHAR(50),
    detection_method VARCHAR(100),
    confidence_level FLOAT,
    associated_sill_id VARCHAR(100),
    deformation_signature JSONB,
    seismic_signature JSONB,
    geochemical_signature JSONB,
    validation_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Deformation Data Quality Metrics Table
CREATE TABLE IF NOT EXISTS deformation_data_quality (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    data_source VARCHAR(50) NOT NULL, -- 'insar', 'gps', 'tilt', 'strain'
    dataset_id VARCHAR(200) NOT NULL,
    sill_id VARCHAR(100),
    spatial_coverage FLOAT, -- percentage
    temporal_coverage FLOAT, -- percentage
    data_completeness FLOAT, -- percentage
    measurement_precision FLOAT,
    systematic_error_estimate FLOAT,
    noise_level FLOAT,
    coherence_quality FLOAT, -- for InSAR
    atmospheric_correction_quality FLOAT,
    outlier_percentage FLOAT,
    quality_score FLOAT, -- overall 0-1 score
    quality_flags TEXT[],
    assessment_date TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(data_source, dataset_id, sill_id)
);

-- Cross-Correlation Analysis Results Table
CREATE TABLE IF NOT EXISTS cross_correlation_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_id VARCHAR(100) NOT NULL,
    data_type_1 VARCHAR(20) NOT NULL,
    data_type_2 VARCHAR(20) NOT NULL,
    component_1 VARCHAR(20),
    component_2 VARCHAR(20),
    cross_correlation_coefficient FLOAT,
    time_lag_days FLOAT,
    significance_level FLOAT,
    coherence_frequency_bands JSONB,
    phase_relationship FLOAT, -- radians
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(sill_id, data_type_1, data_type_2, component_1, component_2)
);

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_insar_correlations_sill_id ON insar_sill_correlations(sill_id);
CREATE INDEX IF NOT EXISTS idx_insar_correlations_correlation ON insar_sill_correlations(correlation_coefficient DESC);
CREATE INDEX IF NOT EXISTS idx_insar_correlations_date ON insar_sill_correlations(acquisition_date);

CREATE INDEX IF NOT EXISTS idx_gps_correlations_sill_id ON gps_sill_correlations(sill_id);
CREATE INDEX IF NOT EXISTS idx_gps_correlations_component ON gps_sill_correlations(component);
CREATE INDEX IF NOT EXISTS idx_gps_correlations_correlation ON gps_sill_correlations(correlation_coefficient DESC);

CREATE INDEX IF NOT EXISTS idx_gps_stations_location ON gps_stations USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_gps_stations_status ON gps_stations(operational_status);

CREATE INDEX IF NOT EXISTS idx_gps_measurements_station_date ON gps_deformation_measurements(station_id, measurement_date);
CREATE INDEX IF NOT EXISTS idx_gps_measurements_date ON gps_deformation_measurements(measurement_date);

CREATE INDEX IF NOT EXISTS idx_tilt_strain_instruments_location ON tilt_strain_instruments USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_tilt_strain_instruments_type ON tilt_strain_instruments(instrument_type);

CREATE INDEX IF NOT EXISTS idx_tilt_strain_measurements_instrument_date ON tilt_strain_measurements(instrument_id, measurement_date);
CREATE INDEX IF NOT EXISTS idx_tilt_strain_measurements_date ON tilt_strain_measurements(measurement_date);

CREATE INDEX IF NOT EXISTS idx_tilt_strain_correlations_sill_id ON tilt_strain_sill_correlations(sill_id);
CREATE INDEX IF NOT EXISTS idx_tilt_strain_correlations_type ON tilt_strain_sill_correlations(data_type);

CREATE INDEX IF NOT EXISTS idx_forward_models_sill_id ON sill_forward_models(sill_id);
CREATE INDEX IF NOT EXISTS idx_forward_models_type ON sill_forward_models(model_type);

CREATE INDEX IF NOT EXISTS idx_inversions_sill_id ON sill_parameter_inversions(sill_id);
CREATE INDEX IF NOT EXISTS idx_inversions_success ON sill_parameter_inversions(success);
CREATE INDEX IF NOT EXISTS idx_inversions_misfit ON sill_parameter_inversions(rms_misfit);

CREATE INDEX IF NOT EXISTS idx_uncertainties_sill_id ON sill_parameter_uncertainties(sill_id);
CREATE INDEX IF NOT EXISTS idx_uncertainties_parameter ON sill_parameter_uncertainties(parameter_name);

CREATE INDEX IF NOT EXISTS idx_temporal_analysis_sill_id ON temporal_deformation_analysis(sill_id);
CREATE INDEX IF NOT EXISTS idx_temporal_analysis_type ON temporal_deformation_analysis(data_type);

CREATE INDEX IF NOT EXISTS idx_validations_correlation_id ON sill_correlation_validations(correlation_id);
CREATE INDEX IF NOT EXISTS idx_validations_score ON sill_correlation_validations(overall_validation_score DESC);

CREATE INDEX IF NOT EXISTS idx_monitored_events_location ON monitored_sill_events USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_monitored_events_time ON monitored_sill_events(event_time);
CREATE INDEX IF NOT EXISTS idx_monitored_events_sill_id ON monitored_sill_events(associated_sill_id);

CREATE INDEX IF NOT EXISTS idx_data_quality_source ON deformation_data_quality(data_source);
CREATE INDEX IF NOT EXISTS idx_data_quality_sill_id ON deformation_data_quality(sill_id);
CREATE INDEX IF NOT EXISTS idx_data_quality_score ON deformation_data_quality(quality_score DESC);

CREATE INDEX IF NOT EXISTS idx_cross_correlation_sill_id ON cross_correlation_analysis(sill_id);
CREATE INDEX IF NOT EXISTS idx_cross_correlation_types ON cross_correlation_analysis(data_type_1, data_type_2);

-- Create spatial indexes for geographic queries
CREATE INDEX IF NOT EXISTS idx_gps_stations_spatial ON gps_stations USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_tilt_strain_instruments_spatial ON tilt_strain_instruments USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_monitored_events_spatial ON monitored_sill_events USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_forward_models_spatial ON sill_forward_models USING GIST(model_extent);

-- Create composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_insar_sill_date_correlation ON insar_sill_correlations(sill_id, acquisition_date, correlation_coefficient DESC);
CREATE INDEX IF NOT EXISTS idx_gps_station_component_correlation ON gps_sill_correlations(sill_id, component, correlation_coefficient DESC);
CREATE INDEX IF NOT EXISTS idx_temporal_sill_type_component ON temporal_deformation_analysis(sill_id, data_type, component);

-- Add comments for documentation
COMMENT ON TABLE insar_sill_correlations IS 'Correlation results between InSAR deformation measurements and sill emplacement models';
COMMENT ON TABLE gps_sill_correlations IS 'Correlation results between GPS deformation measurements and sill emplacement models';
COMMENT ON TABLE gps_stations IS 'GPS station metadata and configuration information';
COMMENT ON TABLE gps_deformation_measurements IS 'GPS deformation measurements with uncertainties and quality metrics';
COMMENT ON TABLE tilt_strain_instruments IS 'Tilt and strain instrument metadata and configuration';
COMMENT ON TABLE tilt_strain_measurements IS 'Tilt and strain measurements with uncertainties and corrections';
COMMENT ON TABLE tilt_strain_sill_correlations IS 'Correlation results between tilt/strain measurements and sill models';
COMMENT ON TABLE sill_forward_models IS 'Forward model results for surface deformation from sill sources';
COMMENT ON TABLE sill_parameter_inversions IS 'Parameter inversion results for sill geometry and properties';
COMMENT ON TABLE sill_parameter_uncertainties IS 'Uncertainty estimates for inverted sill parameters';
COMMENT ON TABLE temporal_deformation_analysis IS 'Temporal analysis results for deformation time series';
COMMENT ON TABLE sill_correlation_validations IS 'Validation results against monitored sill intrusion events';
COMMENT ON TABLE monitored_sill_events IS 'Catalog of monitored sill intrusion events for validation';
COMMENT ON TABLE deformation_data_quality IS 'Quality metrics and assessments for deformation datasets';
COMMENT ON TABLE cross_correlation_analysis IS 'Cross-correlation analysis between different deformation data types';

-- Create views for common queries
CREATE OR REPLACE VIEW sill_correlation_summary AS
SELECT 
    s.sill_id,
    s.center_lon,
    s.center_lat,
    s.depth,
    s.length,
    s.width,
    s.success as inversion_success,
    s.rms_misfit,
    COALESCE(i.avg_insar_correlation, 0) as avg_insar_correlation,
    COALESCE(g.avg_gps_correlation, 0) as avg_gps_correlation,
    COALESCE(t.avg_tilt_strain_correlation, 0) as avg_tilt_strain_correlation,
    COALESCE(v.overall_validation_score, 0) as validation_score
FROM sill_parameter_inversions s
LEFT JOIN (
    SELECT sill_id, AVG(correlation_coefficient) as avg_insar_correlation
    FROM insar_sill_correlations
    GROUP BY sill_id
) i ON s.sill_id = i.sill_id
LEFT JOIN (
    SELECT sill_id, AVG(correlation_coefficient) as avg_gps_correlation
    FROM gps_sill_correlations
    GROUP BY sill_id
) g ON s.sill_id = g.sill_id
LEFT JOIN (
    SELECT sill_id, AVG(correlation_coefficient) as avg_tilt_strain_correlation
    FROM tilt_strain_sill_correlations
    GROUP BY sill_id
) t ON s.sill_id = t.sill_id
LEFT JOIN sill_correlation_validations v ON s.sill_id = v.sill_id;

COMMENT ON VIEW sill_correlation_summary IS 'Summary view of sill correlation results across all data types';

-- Create materialized view for performance-critical queries
CREATE MATERIALIZED VIEW IF NOT EXISTS sill_deformation_statistics AS
SELECT 
    sill_id,
    COUNT(DISTINCT CASE WHEN data_source = 'insar' THEN dataset_id END) as num_insar_datasets,
    COUNT(DISTINCT CASE WHEN data_source = 'gps' THEN dataset_id END) as num_gps_stations,
    COUNT(DISTINCT CASE WHEN data_source = 'tilt' THEN dataset_id END) as num_tilt_instruments,
    COUNT(DISTINCT CASE WHEN data_source = 'strain' THEN dataset_id END) as num_strain_instruments,
    AVG(CASE WHEN data_source = 'insar' THEN quality_score END) as avg_insar_quality,
    AVG(CASE WHEN data_source = 'gps' THEN quality_score END) as avg_gps_quality,
    AVG(CASE WHEN data_source IN ('tilt', 'strain') THEN quality_score END) as avg_tilt_strain_quality,
    MIN(assessment_date) as first_assessment,
    MAX(assessment_date) as last_assessment
FROM deformation_data_quality
WHERE sill_id IS NOT NULL
GROUP BY sill_id;

CREATE UNIQUE INDEX ON sill_deformation_statistics (sill_id);
COMMENT ON MATERIALIZED VIEW sill_deformation_statistics IS 'Aggregated statistics for deformation data quality by sill';

-- Refresh the materialized view
REFRESH MATERIALIZED VIEW sill_deformation_statistics;