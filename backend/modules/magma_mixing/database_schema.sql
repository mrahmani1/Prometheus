-- Database schema for magma mixing visualization and analysis

-- Magma chambers table
CREATE TABLE IF NOT EXISTS magma_chambers (
    chamber_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location GEOMETRY(POINT, 4326) NOT NULL,
    geometry JSONB NOT NULL,
    volume FLOAT NOT NULL,
    depth_min FLOAT NOT NULL,
    depth_max FLOAT NOT NULL,
    composition JSONB NOT NULL,
    temperature FLOAT,
    pressure FLOAT,
    crystallinity FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Magma mixing events table
CREATE TABLE IF NOT EXISTS magma_mixing_events (
    mixing_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id UUID REFERENCES magma_chambers(chamber_id),
    timestamp TIMESTAMP NOT NULL,
    mixing_type VARCHAR(50) NOT NULL, -- 'injection', 'convection', 'external_trigger'
    composition_data JSONB NOT NULL,
    temperature_data JSONB NOT NULL,
    rheological_data JSONB NOT NULL,
    mixing_efficiency FLOAT,
    trigger_mechanism JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Magma mixing visualizations table
CREATE TABLE IF NOT EXISTS magma_mixing_visualizations (
    visualization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id UUID REFERENCES magma_chambers(chamber_id),
    visualization_type VARCHAR(100) NOT NULL,
    visualization_data JSONB NOT NULL,
    mixing_indices JSONB NOT NULL,
    grid_resolution JSONB,
    time_range JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Thermal monitoring data table
CREATE TABLE IF NOT EXISTS thermal_monitoring_data (
    monitoring_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id UUID REFERENCES magma_chambers(chamber_id),
    timestamp TIMESTAMP NOT NULL,
    temperature_field JSONB NOT NULL,
    thermal_properties JSONB NOT NULL,
    heat_flux JSONB,
    thermal_gradients JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Rheological monitoring data table
CREATE TABLE IF NOT EXISTS rheological_monitoring_data (
    monitoring_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id UUID REFERENCES magma_chambers(chamber_id),
    timestamp TIMESTAMP NOT NULL,
    viscosity_field JSONB NOT NULL,
    rheological_properties JSONB NOT NULL,
    flow_patterns JSONB,
    mixing_parameters JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Convection monitoring data table
CREATE TABLE IF NOT EXISTS convection_monitoring_data (
    monitoring_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id UUID REFERENCES magma_chambers(chamber_id),
    timestamp TIMESTAMP NOT NULL,
    velocity_field JSONB NOT NULL,
    convection_parameters JSONB NOT NULL,
    rayleigh_number FLOAT,
    peclet_number FLOAT,
    mixing_efficiency FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Particle tracking data table
CREATE TABLE IF NOT EXISTS particle_tracking_data (
    tracking_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id UUID REFERENCES magma_chambers(chamber_id),
    timestamp TIMESTAMP NOT NULL,
    particle_positions JSONB NOT NULL,
    particle_trajectories JSONB NOT NULL,
    mixing_statistics JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Compositional evolution table
CREATE TABLE IF NOT EXISTS compositional_evolution (
    evolution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id UUID REFERENCES magma_chambers(chamber_id),
    timestamp TIMESTAMP NOT NULL,
    major_elements JSONB NOT NULL,
    trace_elements JSONB,
    isotope_ratios JSONB,
    volatile_content JSONB,
    crystallization_sequence JSONB,
    thermodynamic_state JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Mixing trigger analysis table
CREATE TABLE IF NOT EXISTS mixing_trigger_analysis (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id UUID REFERENCES magma_chambers(chamber_id),
    timestamp TIMESTAMP NOT NULL,
    trigger_type VARCHAR(100) NOT NULL,
    trigger_parameters JSONB NOT NULL,
    density_contrast FLOAT,
    temperature_difference FLOAT,
    volatile_effects JSONB,
    statistical_analysis JSONB,
    prediction_results JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Eruption prediction results table
CREATE TABLE IF NOT EXISTS eruption_prediction_results (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id UUID REFERENCES magma_chambers(chamber_id),
    timestamp TIMESTAMP NOT NULL,
    mixing_degree FLOAT NOT NULL,
    eruption_intensity_prediction FLOAT,
    explosivity_forecast FLOAT,
    discharge_rate_prediction FLOAT,
    eruption_style_prediction VARCHAR(100),
    hazard_assessment JSONB,
    uncertainty_bounds JSONB,
    validation_metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_magma_chambers_location ON magma_chambers USING GIST(location);
CREATE INDEX idx_mixing_events_chamber_timestamp ON magma_mixing_events(chamber_id, timestamp);
CREATE INDEX idx_thermal_monitoring_chamber_timestamp ON thermal_monitoring_data(chamber_id, timestamp);
CREATE INDEX idx_rheological_monitoring_chamber_timestamp ON rheological_monitoring_data(chamber_id, timestamp);
CREATE INDEX idx_convection_monitoring_chamber_timestamp ON convection_monitoring_data(chamber_id, timestamp);
CREATE INDEX idx_compositional_evolution_chamber_timestamp ON compositional_evolution(chamber_id, timestamp);
CREATE INDEX idx_mixing_trigger_chamber_timestamp ON mixing_trigger_analysis(chamber_id, timestamp);
CREATE INDEX idx_eruption_prediction_chamber_timestamp ON eruption_prediction_results(chamber_id, timestamp);

-- Create views for common queries
CREATE OR REPLACE VIEW mixing_chamber_summary AS
SELECT 
    mc.chamber_id,
    mc.location,
    mc.volume,
    mc.depth_min,
    mc.depth_max,
    COUNT(mme.mixing_event_id) as total_mixing_events,
    AVG(mme.mixing_efficiency) as avg_mixing_efficiency,
    MAX(mme.timestamp) as last_mixing_event,
    COUNT(mv.visualization_id) as total_visualizations
FROM magma_chambers mc
LEFT JOIN magma_mixing_events mme ON mc.chamber_id = mme.chamber_id
LEFT JOIN magma_mixing_visualizations mv ON mc.chamber_id = mv.chamber_id
GROUP BY mc.chamber_id, mc.location, mc.volume, mc.depth_min, mc.depth_max;

CREATE OR REPLACE VIEW recent_mixing_activity AS
SELECT 
    mme.chamber_id,
    mme.timestamp,
    mme.mixing_type,
    mme.mixing_efficiency,
    mc.location,
    mc.volume
FROM magma_mixing_events mme
JOIN magma_chambers mc ON mme.chamber_id = mc.chamber_id
WHERE mme.timestamp >= NOW() - INTERVAL '30 days'
ORDER BY mme.timestamp DESC;
-- A
dditional tables for Mixing Trigger Analysis System (Task 5.2)

-- Magma injection events table
CREATE TABLE IF NOT EXISTS magma_injection_events (
    event_id VARCHAR(255) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    chamber_id VARCHAR(255) NOT NULL,
    injection_location JSONB NOT NULL,
    injection_volume FLOAT NOT NULL,
    injection_rate FLOAT NOT NULL,
    injection_temperature FLOAT NOT NULL,
    injection_pressure FLOAT NOT NULL,
    duration FLOAT NOT NULL,
    trigger_strength FLOAT NOT NULL,
    mixing_efficiency FLOAT NOT NULL,
    injected_composition JSONB NOT NULL,
    injection_velocity JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Convection initiation events table
CREATE TABLE IF NOT EXISTS convection_initiation_events (
    initiation_id VARCHAR(255) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    chamber_id VARCHAR(255) NOT NULL,
    rayleigh_number FLOAT NOT NULL,
    critical_rayleigh FLOAT NOT NULL,
    temperature_gradient FLOAT NOT NULL,
    density_gradient FLOAT NOT NULL,
    viscosity_contrast FLOAT NOT NULL,
    onset_depth FLOAT NOT NULL,
    cell_size FLOAT NOT NULL,
    convection_velocity FLOAT NOT NULL,
    mixing_potential FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- External mixing triggers table
CREATE TABLE IF NOT EXISTS external_mixing_triggers (
    trigger_id VARCHAR(255) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    trigger_type VARCHAR(100) NOT NULL,
    magnitude FLOAT NOT NULL,
    location JSONB NOT NULL,
    frequency FLOAT NOT NULL,
    duration FLOAT NOT NULL,
    energy FLOAT NOT NULL,
    stress_change FLOAT NOT NULL,
    pressure_change FLOAT NOT NULL,
    mixing_response FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Density contrast analysis table
CREATE TABLE IF NOT EXISTS density_contrast_analysis (
    analysis_id VARCHAR(255) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    chamber_id VARCHAR(255) NOT NULL,
    density_field JSONB NOT NULL,
    density_gradient JSONB NOT NULL,
    buoyancy_force JSONB NOT NULL,
    rayleigh_taylor_instability FLOAT NOT NULL,
    mixing_potential FLOAT NOT NULL,
    critical_density_contrast FLOAT NOT NULL,
    instability_growth_rate FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Temperature difference analysis table
CREATE TABLE IF NOT EXISTS temperature_difference_analysis (
    analysis_id VARCHAR(255) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    chamber_id VARCHAR(255) NOT NULL,
    temperature_field JSONB NOT NULL,
    temperature_gradient JSONB NOT NULL,
    thermal_buoyancy JSONB NOT NULL,
    thermal_diffusivity FLOAT NOT NULL,
    peclet_number FLOAT NOT NULL,
    thermal_mixing_efficiency FLOAT NOT NULL,
    convection_strength FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Volatile content analysis table
CREATE TABLE IF NOT EXISTS volatile_content_analysis (
    analysis_id VARCHAR(255) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    chamber_id VARCHAR(255) NOT NULL,
    volatile_distribution JSONB NOT NULL,
    saturation_pressure JSONB NOT NULL,
    bubble_nucleation_rate FLOAT NOT NULL,
    degassing_rate FLOAT NOT NULL,
    volatile_induced_convection FLOAT NOT NULL,
    mixing_enhancement_factor FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Mixing trigger predictions table
CREATE TABLE IF NOT EXISTS mixing_trigger_predictions (
    prediction_id VARCHAR(255) PRIMARY KEY,
    prediction_timestamp TIMESTAMP NOT NULL,
    predicted_trigger_time TIMESTAMP NOT NULL,
    trigger_probability FLOAT NOT NULL,
    trigger_type VARCHAR(100) NOT NULL,
    expected_magnitude FLOAT NOT NULL,
    mixing_intensity_prediction FLOAT NOT NULL,
    confidence_interval JSONB NOT NULL,
    uncertainty_bounds JSONB NOT NULL,
    chamber_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Statistical analysis results table
CREATE TABLE IF NOT EXISTS trigger_statistical_analysis (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(255) NOT NULL,
    analysis_timestamp TIMESTAMP NOT NULL,
    trigger_frequencies JSONB NOT NULL,
    trigger_correlations JSONB NOT NULL,
    temporal_patterns JSONB NOT NULL,
    magnitude_distributions JSONB NOT NULL,
    trigger_clusters JSONB NOT NULL,
    pca_results JSONB NOT NULL,
    significance_tests JSONB NOT NULL,
    total_triggers_analyzed INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Seismic monitoring data for trigger analysis
CREATE TABLE IF NOT EXISTS seismic_trigger_monitoring (
    monitoring_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    seismic_data JSONB NOT NULL,
    event_magnitude FLOAT,
    event_location JSONB,
    dominant_frequency FLOAT,
    event_duration FLOAT,
    distance_km FLOAT,
    stress_change FLOAT,
    mixing_trigger_potential FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Pressure monitoring data for trigger analysis
CREATE TABLE IF NOT EXISTS pressure_trigger_monitoring (
    monitoring_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    pressure_data JSONB NOT NULL,
    pressure_change_pa FLOAT NOT NULL,
    pressure_gradient JSONB,
    source_location JSONB,
    event_duration FLOAT,
    mixing_trigger_potential FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Multi-parameter monitoring signatures
CREATE TABLE IF NOT EXISTS multi_parameter_signatures (
    signature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    seismic_signature JSONB,
    deformation_signature JSONB,
    thermal_signature JSONB,
    gas_signature JSONB,
    correlation_score FLOAT NOT NULL,
    signature_strength FLOAT NOT NULL,
    signature_duration FLOAT NOT NULL,
    injection_probability FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for the new tables
CREATE INDEX idx_injection_events_chamber_timestamp ON magma_injection_events(chamber_id, timestamp);
CREATE INDEX idx_injection_events_timestamp ON magma_injection_events(timestamp);
CREATE INDEX idx_convection_events_chamber_timestamp ON convection_initiation_events(chamber_id, timestamp);
CREATE INDEX idx_convection_events_rayleigh ON convection_initiation_events(rayleigh_number);
CREATE INDEX idx_external_triggers_timestamp ON external_mixing_triggers(timestamp);
CREATE INDEX idx_external_triggers_type ON external_mixing_triggers(trigger_type);
CREATE INDEX idx_density_analysis_chamber_timestamp ON density_contrast_analysis(chamber_id, timestamp);
CREATE INDEX idx_temperature_analysis_chamber_timestamp ON temperature_difference_analysis(chamber_id, timestamp);
CREATE INDEX idx_volatile_analysis_chamber_timestamp ON volatile_content_analysis(chamber_id, timestamp);
CREATE INDEX idx_trigger_predictions_timestamp ON mixing_trigger_predictions(prediction_timestamp);
CREATE INDEX idx_trigger_predictions_trigger_time ON mixing_trigger_predictions(predicted_trigger_time);
CREATE INDEX idx_statistical_analysis_chamber ON trigger_statistical_analysis(chamber_id);
CREATE INDEX idx_seismic_monitoring_chamber_timestamp ON seismic_trigger_monitoring(chamber_id, timestamp);
CREATE INDEX idx_pressure_monitoring_chamber_timestamp ON pressure_trigger_monitoring(chamber_id, timestamp);
CREATE INDEX idx_signatures_chamber_timestamp ON multi_parameter_signatures(chamber_id, timestamp);
CREATE INDEX idx_signatures_correlation ON multi_parameter_signatures(correlation_score);

-- Create views for trigger analysis
CREATE OR REPLACE VIEW trigger_analysis_summary AS
SELECT 
    chamber_id,
    COUNT(CASE WHEN trigger_type = 'magma_injection' THEN 1 END) as injection_triggers,
    COUNT(CASE WHEN trigger_type = 'seismic_event' THEN 1 END) as seismic_triggers,
    COUNT(CASE WHEN trigger_type = 'pressure_change' THEN 1 END) as pressure_triggers,
    COUNT(CASE WHEN trigger_type = 'external_perturbation' THEN 1 END) as external_triggers,
    AVG(magnitude) as avg_trigger_magnitude,
    AVG(mixing_response) as avg_mixing_response,
    MAX(timestamp) as last_trigger_event,
    COUNT(*) as total_triggers
FROM external_mixing_triggers
GROUP BY chamber_id;

CREATE OR REPLACE VIEW recent_injection_events AS
SELECT 
    ie.chamber_id,
    ie.timestamp,
    ie.injection_volume,
    ie.injection_rate,
    ie.injection_temperature,
    ie.mixing_efficiency,
    ie.trigger_strength
FROM magma_injection_events ie
WHERE ie.timestamp >= NOW() - INTERVAL '7 days'
ORDER BY ie.timestamp DESC;

CREATE OR REPLACE VIEW convection_activity_summary AS
SELECT 
    chamber_id,
    COUNT(*) as convection_events,
    AVG(rayleigh_number) as avg_rayleigh_number,
    AVG(mixing_potential) as avg_mixing_potential,
    MAX(convection_velocity) as max_convection_velocity,
    MAX(timestamp) as last_convection_event
FROM convection_initiation_events
GROUP BY chamber_id;

CREATE OR REPLACE VIEW trigger_prediction_accuracy AS
SELECT 
    trigger_type,
    COUNT(*) as total_predictions,
    AVG(trigger_probability) as avg_predicted_probability,
    AVG(mixing_intensity_prediction) as avg_predicted_intensity,
    COUNT(CASE WHEN predicted_trigger_time <= NOW() THEN 1 END) as expired_predictions,
    COUNT(CASE WHEN predicted_trigger_time > NOW() THEN 1 END) as active_predictions
FROM mixing_trigger_predictions
GROUP BY trigger_type;
-- Addi
tional tables for Geochemical Evolution Tracking System (Task 5.3)

-- Geochemical predictions table
CREATE TABLE IF NOT EXISTS geochemical_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(255) NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    target_timestamp TIMESTAMP NOT NULL,
    predictions JSONB NOT NULL,
    uncertainty_bounds JSONB,
    confidence_intervals JSONB,
    model_version VARCHAR(100),
    prediction_accuracy FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Geochemical validation results table
CREATE TABLE IF NOT EXISTS geochemical_validation_results (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(255) NOT NULL,
    validation_timestamp TIMESTAMP NOT NULL,
    validation_metrics JSONB NOT NULL,
    natural_composition_match VARCHAR(255),
    similarity_score FLOAT,
    statistical_significance FLOAT,
    validation_method VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Thermodynamic equilibrium calculations table
CREATE TABLE IF NOT EXISTS thermodynamic_equilibrium (
    equilibrium_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    temperature FLOAT NOT NULL,
    pressure FLOAT NOT NULL,
    oxygen_fugacity FLOAT,
    water_activity FLOAT,
    silica_activity FLOAT,
    gibbs_free_energy FLOAT,
    enthalpy FLOAT,
    entropy FLOAT,
    heat_capacity FLOAT,
    compressibility FLOAT,
    thermal_expansion FLOAT,
    equilibrium_assemblage JSONB,
    activity_coefficients JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Partition coefficient database table
CREATE TABLE IF NOT EXISTS partition_coefficients (
    coefficient_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mineral_phase VARCHAR(100) NOT NULL,
    element VARCHAR(20) NOT NULL,
    temperature FLOAT NOT NULL,
    pressure FLOAT NOT NULL,
    partition_coefficient FLOAT NOT NULL,
    temperature_dependence FLOAT,
    pressure_dependence FLOAT,
    composition_dependence JSONB,
    experimental_conditions JSONB,
    reference_study VARCHAR(255),
    uncertainty FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Isotopic fractionation factors table
CREATE TABLE IF NOT EXISTS isotopic_fractionation_factors (
    fractionation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    isotope_system VARCHAR(20) NOT NULL,
    isotope_ratio VARCHAR(20) NOT NULL,
    mineral_phase VARCHAR(100) NOT NULL,
    temperature FLOAT NOT NULL,
    pressure FLOAT NOT NULL,
    fractionation_factor FLOAT NOT NULL,
    temperature_dependence FLOAT,
    kinetic_effects JSONB,
    experimental_conditions JSONB,
    reference_study VARCHAR(255),
    uncertainty FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Volatile solubility data table
CREATE TABLE IF NOT EXISTS volatile_solubility_data (
    solubility_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    volatile_species VARCHAR(20) NOT NULL,
    temperature FLOAT NOT NULL,
    pressure FLOAT NOT NULL,
    melt_composition JSONB NOT NULL,
    solubility FLOAT NOT NULL,
    solubility_law VARCHAR(50) NOT NULL,
    solubility_constants JSONB,
    henry_constant FLOAT,
    diffusivity FLOAT,
    experimental_conditions JSONB,
    reference_study VARCHAR(255),
    uncertainty FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Natural magma compositions database table
CREATE TABLE IF NOT EXISTS natural_magma_compositions (
    composition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    volcano_name VARCHAR(255) NOT NULL,
    eruption_date DATE,
    location GEOMETRY(POINT, 4326),
    composition_type VARCHAR(50) NOT NULL, -- 'mafic', 'felsic', 'mixed'
    major_elements JSONB NOT NULL,
    trace_elements JSONB,
    isotope_ratios JSONB,
    volatile_content JSONB,
    mixing_evidence VARCHAR(255),
    mixing_proportion FLOAT,
    temperature FLOAT,
    pressure FLOAT,
    crystallinity FLOAT,
    reference_study VARCHAR(255),
    sample_id VARCHAR(100),
    analytical_methods JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Machine learning model performance table
CREATE TABLE IF NOT EXISTS ml_model_performance (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'major_elements', 'trace_elements', 'isotopes', etc.
    model_version VARCHAR(50) NOT NULL,
    training_data_size INTEGER,
    validation_accuracy FLOAT,
    test_accuracy FLOAT,
    rmse FLOAT,
    r2_score FLOAT,
    feature_importance JSONB,
    hyperparameters JSONB,
    training_timestamp TIMESTAMP,
    model_file_path TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Geochemical evolution time series table
CREATE TABLE IF NOT EXISTS geochemical_evolution_timeseries (
    timeseries_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(255) NOT NULL,
    evolution_type VARCHAR(50) NOT NULL, -- 'major_elements', 'trace_elements', 'isotopes', 'volatiles'
    start_timestamp TIMESTAMP NOT NULL,
    end_timestamp TIMESTAMP NOT NULL,
    time_resolution FLOAT NOT NULL, -- seconds
    evolution_data JSONB NOT NULL,
    mixing_parameters JSONB,
    thermodynamic_conditions JSONB,
    uncertainty_bounds JSONB,
    validation_metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Mineral crystallization kinetics table
CREATE TABLE IF NOT EXISTS mineral_crystallization_kinetics (
    kinetics_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(255) NOT NULL,
    mineral_phase VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    temperature FLOAT NOT NULL,
    pressure FLOAT NOT NULL,
    supersaturation FLOAT,
    nucleation_rate FLOAT,
    growth_rate FLOAT,
    dissolution_rate FLOAT,
    modal_abundance FLOAT,
    crystal_size_distribution JSONB,
    mineral_composition JSONB,
    trace_element_content JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Volatile degassing kinetics table
CREATE TABLE IF NOT EXISTS volatile_degassing_kinetics (
    degassing_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chamber_id VARCHAR(255) NOT NULL,
    volatile_species VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    temperature FLOAT NOT NULL,
    pressure FLOAT NOT NULL,
    dissolved_concentration FLOAT,
    gas_phase_concentration FLOAT,
    saturation_pressure FLOAT,
    supersaturation FLOAT,
    nucleation_rate FLOAT,
    degassing_rate FLOAT,
    bubble_volume_fraction FLOAT,
    bubble_size_distribution JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_geochemical_predictions_chamber_timestamp ON geochemical_predictions(chamber_id, prediction_timestamp);
CREATE INDEX idx_geochemical_predictions_target_time ON geochemical_predictions(target_timestamp);
CREATE INDEX idx_validation_results_chamber ON geochemical_validation_results(chamber_id);
CREATE INDEX idx_validation_results_similarity ON geochemical_validation_results(similarity_score DESC);
CREATE INDEX idx_thermodynamic_equilibrium_chamber_timestamp ON thermodynamic_equilibrium(chamber_id, timestamp);
CREATE INDEX idx_partition_coefficients_mineral_element ON partition_coefficients(mineral_phase, element);
CREATE INDEX idx_partition_coefficients_temperature ON partition_coefficients(temperature);
CREATE INDEX idx_isotopic_fractionation_system_mineral ON isotopic_fractionation_factors(isotope_system, mineral_phase);
CREATE INDEX idx_volatile_solubility_species_temp ON volatile_solubility_data(volatile_species, temperature);
CREATE INDEX idx_natural_compositions_volcano ON natural_magma_compositions(volcano_name);
CREATE INDEX idx_natural_compositions_location ON natural_magma_compositions USING GIST(location);
CREATE INDEX idx_natural_compositions_type ON natural_magma_compositions(composition_type);
CREATE INDEX idx_ml_model_performance_name_version ON ml_model_performance(model_name, model_version);
CREATE INDEX idx_evolution_timeseries_chamber_type ON geochemical_evolution_timeseries(chamber_id, evolution_type);
CREATE INDEX idx_evolution_timeseries_time_range ON geochemical_evolution_timeseries(start_timestamp, end_timestamp);
CREATE INDEX idx_crystallization_kinetics_chamber_mineral ON mineral_crystallization_kinetics(chamber_id, mineral_phase);
CREATE INDEX idx_crystallization_kinetics_timestamp ON mineral_crystallization_kinetics(timestamp);
CREATE INDEX idx_degassing_kinetics_chamber_species ON volatile_degassing_kinetics(chamber_id, volatile_species);
CREATE INDEX idx_degassing_kinetics_timestamp ON volatile_degassing_kinetics(timestamp);

-- Create views for geochemical evolution analysis
CREATE OR REPLACE VIEW geochemical_evolution_summary AS
SELECT 
    chamber_id,
    COUNT(DISTINCT evolution_type) as evolution_types_tracked,
    MIN(start_timestamp) as earliest_evolution,
    MAX(end_timestamp) as latest_evolution,
    AVG(time_resolution) as avg_time_resolution,
    COUNT(*) as total_evolution_records
FROM geochemical_evolution_timeseries
GROUP BY chamber_id;

CREATE OR REPLACE VIEW recent_geochemical_predictions AS
SELECT 
    gp.chamber_id,
    gp.prediction_timestamp,
    gp.target_timestamp,
    gp.predictions,
    gp.uncertainty_bounds,
    mp.validation_accuracy,
    mp.model_type
FROM geochemical_predictions gp
LEFT JOIN ml_model_performance mp ON mp.model_name = 'geochemical_predictor'
WHERE gp.prediction_timestamp >= NOW() - INTERVAL '7 days'
ORDER BY gp.prediction_timestamp DESC;

CREATE OR REPLACE VIEW validation_accuracy_summary AS
SELECT 
    chamber_id,
    COUNT(*) as total_validations,
    AVG(similarity_score) as avg_similarity_score,
    AVG(statistical_significance) as avg_statistical_significance,
    MAX(similarity_score) as best_similarity_score,
    natural_composition_match as best_match_volcano
FROM geochemical_validation_results
GROUP BY chamber_id, natural_composition_match
ORDER BY avg_similarity_score DESC;

CREATE OR REPLACE VIEW mineral_crystallization_summary AS
SELECT 
    chamber_id,
    mineral_phase,
    COUNT(*) as crystallization_events,
    AVG(nucleation_rate) as avg_nucleation_rate,
    AVG(growth_rate) as avg_growth_rate,
    MAX(modal_abundance) as max_modal_abundance,
    AVG(supersaturation) as avg_supersaturation
FROM mineral_crystallization_kinetics
GROUP BY chamber_id, mineral_phase;

CREATE OR REPLACE VIEW volatile_degassing_summary AS
SELECT 
    chamber_id,
    volatile_species,
    COUNT(*) as degassing_events,
    AVG(degassing_rate) as avg_degassing_rate,
    MAX(supersaturation) as max_supersaturation,
    AVG(bubble_volume_fraction) as avg_bubble_fraction,
    SUM(gas_phase_concentration) as total_degassed
FROM volatile_degassing_kinetics
GROUP BY chamber_id, volatile_species;

-- Create materialized views for performance
CREATE MATERIALIZED VIEW mv_chamber_geochemical_status AS
SELECT 
    chamber_id,
    MAX(timestamp) as last_update,
    COUNT(DISTINCT evolution_type) as active_evolution_types,
    AVG(CASE WHEN evolution_type = 'major_elements' THEN 1 ELSE 0 END) as major_element_tracking,
    AVG(CASE WHEN evolution_type = 'trace_elements' THEN 1 ELSE 0 END) as trace_element_tracking,
    AVG(CASE WHEN evolution_type = 'isotopes' THEN 1 ELSE 0 END) as isotope_tracking,
    AVG(CASE WHEN evolution_type = 'volatiles' THEN 1 ELSE 0 END) as volatile_tracking
FROM geochemical_evolution_timeseries
GROUP BY chamber_id;

-- Create unique indexes to prevent duplicates
CREATE UNIQUE INDEX idx_partition_coefficients_unique ON partition_coefficients(mineral_phase, element, temperature, pressure);
CREATE UNIQUE INDEX idx_isotopic_fractionation_unique ON isotopic_fractionation_factors(isotope_system, isotope_ratio, mineral_phase, temperature);
CREATE UNIQUE INDEX idx_volatile_solubility_unique ON volatile_solubility_data(volatile_species, temperature, pressure, md5(melt_composition::text));
CREATE UNIQUE INDEX idx_natural_compositions_unique ON natural_magma_compositions(volcano_name, eruption_date, composition_type, sample_id);

-- Refresh materialized view function
CREATE OR REPLACE FUNCTION refresh_geochemical_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW mv_chamber_geochemical_status;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to refresh materialized views
CREATE OR REPLACE FUNCTION trigger_refresh_geochemical_views()
RETURNS trigger AS $$
BEGIN
    PERFORM refresh_geochemical_materialized_views();
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_geochemical_evolution_update
    AFTER INSERT OR UPDATE OR DELETE ON geochemical_evolution_timeseries
    FOR EACH STATEMENT
    EXECUTE FUNCTION trigger_refresh_geochemical_views();