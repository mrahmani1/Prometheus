-- Database schema for Eruption Location Prediction System
-- Advanced volcanic dashboard - sill complex eruption prediction

-- Enable PostGIS extension for spatial data
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Stress field analyses table
CREATE TABLE IF NOT EXISTS stress_field_analyses (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_complex_id UUID,
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    grid_resolution JSONB NOT NULL, -- Grid parameters and resolution
    regional_stress JSONB NOT NULL, -- Regional stress field parameters
    stress_tensor_data JSONB NOT NULL, -- 3D stress tensor results
    principal_stresses JSONB NOT NULL, -- Principal stress calculations
    von_mises_stress JSONB NOT NULL, -- Von Mises stress distribution
    stress_gradients JSONB NOT NULL, -- Stress gradient calculations
    analysis_parameters JSONB NOT NULL, -- Analysis configuration
    computational_metrics JSONB, -- Performance and accuracy metrics
    validation_score FLOAT CHECK (validation_score >= 0 AND validation_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Failure criterion analyses table
CREATE TABLE IF NOT EXISTS failure_criterion_analyses (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stress_field_id UUID REFERENCES stress_field_analyses(analysis_id),
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    rock_properties JSONB NOT NULL, -- Rock mechanical properties
    mohr_coulomb_results JSONB NOT NULL, -- Mohr-Coulomb failure analysis
    griffith_results JSONB NOT NULL, -- Griffith failure criterion
    hoek_brown_results JSONB NOT NULL, -- Hoek-Brown failure criterion
    drucker_prager_results JSONB NOT NULL, -- Drucker-Prager criterion
    combined_failure_probability JSONB NOT NULL, -- Combined failure probability
    failure_gradients JSONB NOT NULL, -- Failure gradient calculations
    eruption_pathways JSONB NOT NULL, -- Identified eruption pathways
    critical_stress_locations JSONB NOT NULL, -- Critical stress concentration points
    analysis_parameters JSONB NOT NULL,
    validation_score FLOAT CHECK (validation_score >= 0 AND validation_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Topographic and structural controls table
CREATE TABLE IF NOT EXISTS topographic_structural_controls (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_area GEOMETRY(POLYGON, 4326) NOT NULL, -- Analysis area boundary
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    dem_resolution FLOAT NOT NULL, -- DEM resolution in meters
    elevation_data JSONB NOT NULL, -- Digital elevation model data
    slope_data JSONB NOT NULL, -- Slope calculations
    aspect_data JSONB NOT NULL, -- Aspect calculations
    curvature_data JSONB NOT NULL, -- Surface curvature
    roughness_data JSONB NOT NULL, -- Surface roughness index
    fault_density JSONB NOT NULL, -- Fault density mapping
    fracture_density JSONB NOT NULL, -- Fracture density mapping
    lineament_density JSONB NOT NULL, -- Lineament density mapping
    lineament_orientations JSONB NOT NULL, -- Structural lineament orientations
    drainage_network JSONB NOT NULL, -- Drainage network analysis
    drainage_density JSONB NOT NULL, -- Drainage density calculations
    topographic_stress JSONB NOT NULL, -- Topography-induced stress
    structural_control_index JSONB NOT NULL, -- Composite structural control
    topographic_favorability JSONB NOT NULL, -- Eruption location favorability
    analysis_parameters JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sill geometries table
CREATE TABLE IF NOT EXISTS sill_geometries (
    sill_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_complex_id UUID,
    sill_name VARCHAR(255),
    center_location GEOMETRY(POINTZ, 4326) NOT NULL, -- 3D center coordinates
    geometry_3d GEOMETRY(POLYHEDRALSURFACEZ, 4326), -- 3D sill geometry
    length_m FLOAT NOT NULL CHECK (length_m > 0),
    width_m FLOAT NOT NULL CHECK (width_m > 0),
    thickness_m FLOAT NOT NULL CHECK (thickness_m > 0),
    dip_degrees FLOAT CHECK (dip_degrees >= 0 AND dip_degrees <= 90),
    strike_degrees FLOAT CHECK (strike_degrees >= 0 AND strike_degrees < 360),
    volume_m3 FLOAT NOT NULL CHECK (volume_m3 > 0),
    aspect_ratio FLOAT NOT NULL CHECK (aspect_ratio > 0),
    shape_factor FLOAT CHECK (shape_factor >= 0 AND shape_factor <= 1),
    connectivity_index FLOAT CHECK (connectivity_index >= 0 AND connectivity_index <= 1),
    emplacement_age TIMESTAMP WITH TIME ZONE,
    magma_composition JSONB, -- Magma composition data
    crystallinity FLOAT CHECK (crystallinity >= 0 AND crystallinity <= 1),
    temperature_c FLOAT,
    pressure_pa FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Probabilistic eruption mapping table
CREATE TABLE IF NOT EXISTS probabilistic_eruption_mapping (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stress_field_id UUID REFERENCES stress_field_analyses(analysis_id),
    failure_analysis_id UUID REFERENCES failure_criterion_analyses(analysis_id),
    topographic_analysis_id UUID REFERENCES topographic_structural_controls(analysis_id),
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    surface_coordinates JSONB NOT NULL, -- Surface analysis grid coordinates
    eruption_probability JSONB NOT NULL, -- Eruption probability distribution
    uncertainty_bounds JSONB NOT NULL, -- Uncertainty quantification
    contributing_factors JSONB NOT NULL, -- Individual factor contributions
    high_probability_zones JSONB NOT NULL, -- High probability zone identification
    probability_contours JSONB NOT NULL, -- Probability contour lines
    analysis_weights JSONB NOT NULL, -- Factor weighting scheme
    confidence_level FLOAT NOT NULL CHECK (confidence_level > 0 AND confidence_level < 1),
    ml_enhancement_applied BOOLEAN DEFAULT FALSE,
    model_version VARCHAR(50),
    validation_score FLOAT CHECK (validation_score >= 0 AND validation_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Temporal evolution modeling table
CREATE TABLE IF NOT EXISTS temporal_evolution_modeling (
    evolution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_complex_id UUID,
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    time_horizon_days FLOAT NOT NULL CHECK (time_horizon_days > 0),
    time_steps JSONB NOT NULL, -- Time step array
    sill_evolution JSONB NOT NULL, -- Individual sill evolution data
    stress_evolution JSONB NOT NULL, -- Stress field evolution over time
    failure_evolution JSONB NOT NULL, -- Failure probability evolution
    eruption_potential_evolution JSONB NOT NULL, -- Overall eruption potential
    critical_events JSONB NOT NULL, -- Identified critical events and thresholds
    eruption_forecasts JSONB NOT NULL, -- Probabilistic eruption forecasts
    growth_rates JSONB NOT NULL, -- Sill growth rate parameters
    model_parameters JSONB NOT NULL,
    uncertainty_bounds JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Eruption style predictions table
CREATE TABLE IF NOT EXISTS eruption_style_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_id UUID REFERENCES sill_geometries(sill_id),
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    eruption_style VARCHAR(100) NOT NULL, -- Predicted eruption style
    vei_estimate INTEGER CHECK (vei_estimate >= 0 AND vei_estimate <= 8),
    explosive_potential FLOAT CHECK (explosive_potential >= 0 AND explosive_potential <= 1),
    discharge_rate_m3_per_s FLOAT CHECK (discharge_rate_m3_per_s >= 0),
    column_height_m FLOAT CHECK (column_height_m >= 0),
    duration_estimate_hours FLOAT CHECK (duration_estimate_hours >= 0),
    contributing_factors JSONB NOT NULL, -- Factor breakdown
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    uncertainty_range JSONB NOT NULL, -- VEI uncertainty bounds
    magma_properties JSONB NOT NULL, -- Input magma properties
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Hazard zones table
CREATE TABLE IF NOT EXISTS hazard_zones (
    zone_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mapping_id UUID REFERENCES probabilistic_eruption_mapping(mapping_id),
    zone_geometry GEOMETRY(POLYGON, 4326) NOT NULL, -- Hazard zone boundary
    hazard_level VARCHAR(50) NOT NULL CHECK (hazard_level IN ('Low', 'Moderate', 'High', 'Extreme')),
    hazard_type VARCHAR(100) NOT NULL, -- Type of volcanic hazard
    occurrence_probability FLOAT CHECK (occurrence_probability >= 0 AND occurrence_probability <= 1),
    impact_radius_m FLOAT CHECK (impact_radius_m >= 0),
    population_at_risk INTEGER CHECK (population_at_risk >= 0),
    infrastructure_at_risk JSONB, -- Critical infrastructure list
    evacuation_time_hours FLOAT CHECK (evacuation_time_hours >= 0),
    monitoring_priority INTEGER CHECK (monitoring_priority >= 1 AND monitoring_priority <= 5),
    risk_assessment JSONB NOT NULL, -- Detailed risk assessment
    mitigation_measures JSONB, -- Recommended mitigation measures
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Historical eruptions validation table
CREATE TABLE IF NOT EXISTS historical_eruptions (
    eruption_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    volcano_name VARCHAR(255) NOT NULL,
    eruption_location GEOMETRY(POINT, 4326) NOT NULL,
    eruption_date DATE,
    eruption_end_date DATE,
    vei INTEGER CHECK (vei >= 0 AND vei <= 8),
    eruption_style VARCHAR(100),
    sill_related BOOLEAN DEFAULT FALSE,
    sill_depth_m FLOAT,
    sill_volume_m3 FLOAT,
    magma_composition JSONB,
    eruption_duration_hours FLOAT,
    discharge_rate_m3_per_s FLOAT,
    column_height_m FLOAT,
    affected_area_km2 FLOAT,
    casualties INTEGER DEFAULT 0,
    economic_impact_usd BIGINT,
    data_source VARCHAR(255),
    data_quality_score FLOAT CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model validation results table
CREATE TABLE IF NOT EXISTS model_validation_results (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID REFERENCES probabilistic_eruption_mapping(mapping_id),
    validation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    historical_eruptions_count INTEGER NOT NULL,
    spatial_accuracy JSONB NOT NULL, -- Spatial prediction accuracy metrics
    probability_calibration JSONB NOT NULL, -- Probability calibration metrics
    temporal_accuracy JSONB, -- Temporal prediction accuracy (if available)
    style_prediction_accuracy JSONB, -- Eruption style prediction accuracy
    overall_performance JSONB NOT NULL, -- Overall model performance metrics
    model_performance_score FLOAT CHECK (model_performance_score >= 0 AND model_performance_score <= 1),
    validation_report TEXT,
    improvement_recommendations JSONB,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance optimization
CREATE INDEX idx_stress_field_timestamp ON stress_field_analyses(analysis_timestamp);
CREATE INDEX idx_stress_field_sill_complex ON stress_field_analyses(sill_complex_id);
CREATE INDEX idx_failure_analysis_stress_field ON failure_criterion_analyses(stress_field_id);
CREATE INDEX idx_topographic_analysis_area ON topographic_structural_controls USING GIST(analysis_area);
CREATE INDEX idx_sill_geometries_location ON sill_geometries USING GIST(center_location);
CREATE INDEX idx_sill_geometries_complex ON sill_geometries(sill_complex_id);
CREATE INDEX idx_probabilistic_mapping_timestamp ON probabilistic_eruption_mapping(analysis_timestamp);
CREATE INDEX idx_temporal_evolution_complex ON temporal_evolution_modeling(sill_complex_id);
CREATE INDEX idx_eruption_style_sill ON eruption_style_predictions(sill_id);
CREATE INDEX idx_hazard_zones_geometry ON hazard_zones USING GIST(zone_geometry);
CREATE INDEX idx_hazard_zones_mapping ON hazard_zones(mapping_id);
CREATE INDEX idx_historical_eruptions_location ON historical_eruptions USING GIST(eruption_location);
CREATE INDEX idx_historical_eruptions_date ON historical_eruptions(eruption_date);
CREATE INDEX idx_historical_eruptions_sill_related ON historical_eruptions(sill_related);
CREATE INDEX idx_validation_results_prediction ON model_validation_results(prediction_id);

-- Create views for common queries
CREATE OR REPLACE VIEW v_active_eruption_predictions AS
SELECT 
    pem.mapping_id,
    pem.analysis_timestamp,
    pem.eruption_probability,
    pem.high_probability_zones,
    pem.confidence_level,
    pem.validation_score,
    COUNT(hz.zone_id) as hazard_zones_count,
    MAX(hz.monitoring_priority) as max_monitoring_priority
FROM probabilistic_eruption_mapping pem
LEFT JOIN hazard_zones hz ON pem.mapping_id = hz.mapping_id
WHERE pem.analysis_timestamp > NOW() - INTERVAL '30 days'
GROUP BY pem.mapping_id, pem.analysis_timestamp, pem.eruption_probability, 
         pem.high_probability_zones, pem.confidence_level, pem.validation_score
ORDER BY pem.analysis_timestamp DESC;

CREATE OR REPLACE VIEW v_high_risk_hazard_zones AS
SELECT 
    hz.zone_id,
    hz.zone_geometry,
    hz.hazard_level,
    hz.hazard_type,
    hz.occurrence_probability,
    hz.population_at_risk,
    hz.monitoring_priority,
    pem.analysis_timestamp
FROM hazard_zones hz
JOIN probabilistic_eruption_mapping pem ON hz.mapping_id = pem.mapping_id
WHERE hz.hazard_level IN ('High', 'Extreme') 
  AND hz.occurrence_probability > 0.3
  AND pem.analysis_timestamp > NOW() - INTERVAL '7 days'
ORDER BY hz.occurrence_probability DESC, hz.population_at_risk DESC;

-- Insert sample data for testing
INSERT INTO historical_eruptions (
    volcano_name, eruption_location, eruption_date, vei, eruption_style, 
    sill_related, sill_depth_m, data_source, data_quality_score
) VALUES 
('Mount Vesuvius', ST_SetSRID(ST_MakePoint(14.426, 40.821), 4326), '1944-03-18', 3, 'Vulcanian', true, 1200, 'GVP', 0.9),
('Kilauea', ST_SetSRID(ST_MakePoint(-155.292, 19.421), 4326), '2018-05-03', 4, 'Effusive', true, 800, 'USGS', 0.95),
('Eyjafjallajokull', ST_SetSRID(ST_MakePoint(-19.633, 63.633), 4326), '2010-04-14', 4, 'Explosive', true, 1500, 'IMO', 0.85);

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO prometheus;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO prometheus;