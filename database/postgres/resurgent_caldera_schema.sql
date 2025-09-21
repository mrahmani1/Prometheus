-- Resurgent Caldera Analysis Database Schema
-- Advanced database schema for storing resurgent caldera analysis results

-- Enable PostGIS extension for spatial data
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Uplift mechanism modeling results
CREATE TABLE IF NOT EXISTS uplift_mechanism_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    caldera_id VARCHAR(100) NOT NULL,
    mechanism_type VARCHAR(50) NOT NULL,
    simulation_timestamp TIMESTAMP DEFAULT NOW(),
    intrusion_properties JSONB,
    deformation_results JSONB,
    pressure_evolution JSONB,
    thermal_evolution JSONB,
    validation_metrics JSONB,
    uncertainty_bounds JSONB,
    spatial_extent GEOMETRY(POLYGON, 4326),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Caldera geometry and properties
CREATE TABLE IF NOT EXISTS caldera_geometries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    caldera_id VARCHAR(100) UNIQUE NOT NULL,
    caldera_name VARCHAR(200),
    location GEOMETRY(POINT, 4326) NOT NULL,
    rim_geometry GEOMETRY(POLYGON, 4326),
    rim_diameter FLOAT, -- km
    floor_area FLOAT, -- km²
    depth_to_floor FLOAT, -- km
    wall_angle FLOAT, -- degrees
    basement_depth FLOAT, -- km
    crustal_thickness FLOAT, -- km
    topography_data JSONB, -- Digital elevation model
    structural_features JSONB,
    geological_setting VARCHAR(100),
    last_eruption TIMESTAMP,
    monitoring_status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Magma intrusion properties and evolution
CREATE TABLE IF NOT EXISTS magma_intrusions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    caldera_id VARCHAR(100) REFERENCES caldera_geometries(caldera_id),
    intrusion_id VARCHAR(100) NOT NULL,
    depth FLOAT NOT NULL, -- km
    volume FLOAT NOT NULL, -- km³
    pressure FLOAT, -- MPa
    temperature FLOAT, -- °C
    composition JSONB, -- Major element composition
    volatile_content JSONB, -- H2O, CO2, SO2 content
    viscosity FLOAT, -- Pa·s
    density FLOAT, -- kg/m³
    bulk_modulus FLOAT, -- GPa
    thermal_expansion FLOAT, -- 1/K
    emplacement_date TIMESTAMP,
    geometry GEOMETRY(POLYGON, 4326),
    evolution_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Rock properties for different units
CREATE TABLE IF NOT EXISTS rock_properties (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    caldera_id VARCHAR(100) REFERENCES caldera_geometries(caldera_id),
    rock_unit VARCHAR(100) NOT NULL,
    youngs_modulus FLOAT, -- GPa
    poissons_ratio FLOAT,
    density FLOAT, -- kg/m³
    thermal_conductivity FLOAT, -- W/m/K
    specific_heat FLOAT, -- J/kg/K
    thermal_expansion FLOAT, -- 1/K
    permeability FLOAT, -- m²
    porosity FLOAT,
    compressibility FLOAT, -- 1/Pa
    viscosity FLOAT, -- Pa·s (for viscoelastic modeling)
    depth_range JSONB, -- min/max depth range
    spatial_extent GEOMETRY(POLYGON, 4326),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Hydrothermal system properties
CREATE TABLE IF NOT EXISTS hydrothermal_systems (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    caldera_id VARCHAR(100) REFERENCES caldera_geometries(caldera_id),
    system_id VARCHAR(100) NOT NULL,
    fluid_temperature FLOAT, -- °C
    fluid_pressure FLOAT, -- MPa
    fluid_composition JSONB,
    flow_rate FLOAT, -- kg/s
    heat_flux FLOAT, -- W/m²
    permeability_field JSONB,
    porosity_field JSONB,
    fluid_pathways JSONB,
    system_geometry GEOMETRY(MULTIPOLYGON, 4326),
    activity_level VARCHAR(50),
    monitoring_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Deformation monitoring data
CREATE TABLE IF NOT EXISTS deformation_measurements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    caldera_id VARCHAR(100) REFERENCES caldera_geometries(caldera_id),
    measurement_type VARCHAR(50) NOT NULL, -- InSAR, GPS, tilt, strain
    measurement_timestamp TIMESTAMP NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    vertical_displacement FLOAT, -- mm
    horizontal_displacement_east FLOAT, -- mm
    horizontal_displacement_north FLOAT, -- mm
    displacement_rate FLOAT, -- mm/year
    measurement_uncertainty FLOAT, -- mm
    processing_method VARCHAR(100),
    data_quality VARCHAR(20),
    raw_data JSONB,
    processed_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Uplift-subsidence cycle analysis
CREATE TABLE IF NOT EXISTS uplift_subsidence_cycles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    caldera_id VARCHAR(100) REFERENCES caldera_geometries(caldera_id),
    cycle_id VARCHAR(100) NOT NULL,
    cycle_start_date TIMESTAMP,
    cycle_end_date TIMESTAMP,
    cycle_duration FLOAT, -- years
    maximum_uplift FLOAT, -- mm
    maximum_subsidence FLOAT, -- mm
    uplift_rate FLOAT, -- mm/year
    subsidence_rate FLOAT, -- mm/year
    cycle_amplitude FLOAT, -- mm
    cycle_pattern VARCHAR(50),
    driving_mechanism VARCHAR(100),
    statistical_metrics JSONB,
    cycle_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Validation data from monitored calderas
CREATE TABLE IF NOT EXISTS caldera_validation_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    caldera_id VARCHAR(100) REFERENCES caldera_geometries(caldera_id),
    validation_type VARCHAR(50) NOT NULL,
    observation_period_start TIMESTAMP,
    observation_period_end TIMESTAMP,
    observed_deformation JSONB,
    observed_mechanisms JSONB,
    monitoring_network JSONB,
    data_quality_metrics JSONB,
    reference_studies JSONB,
    validation_notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Model validation results
CREATE TABLE IF NOT EXISTS model_validation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_run_id UUID REFERENCES uplift_mechanism_results(id),
    validation_data_id UUID REFERENCES caldera_validation_data(id),
    validation_timestamp TIMESTAMP DEFAULT NOW(),
    fit_metrics JSONB, -- R², RMSE, MAE, etc.
    residual_analysis JSONB,
    mechanism_attribution JSONB,
    prediction_accuracy JSONB,
    validation_score FLOAT,
    validation_status VARCHAR(50),
    validation_notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Uncertainty quantification results
CREATE TABLE IF NOT EXISTS uncertainty_quantification (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_run_id UUID REFERENCES uplift_mechanism_results(id),
    parameter_uncertainties JSONB,
    model_uncertainties JSONB,
    prediction_intervals JSONB,
    sensitivity_analysis JSONB,
    monte_carlo_results JSONB,
    confidence_levels JSONB,
    uncertainty_sources JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance optimization
CREATE INDEX idx_uplift_mechanism_caldera ON uplift_mechanism_results(caldera_id);
CREATE INDEX idx_uplift_mechanism_type ON uplift_mechanism_results(mechanism_type);
CREATE INDEX idx_uplift_mechanism_timestamp ON uplift_mechanism_results(simulation_timestamp);

CREATE INDEX idx_caldera_location ON caldera_geometries USING GIST(location);
CREATE INDEX idx_caldera_rim ON caldera_geometries USING GIST(rim_geometry);

CREATE INDEX idx_deformation_caldera ON deformation_measurements(caldera_id);
CREATE INDEX idx_deformation_timestamp ON deformation_measurements(measurement_timestamp);
CREATE INDEX idx_deformation_location ON deformation_measurements USING GIST(location);
CREATE INDEX idx_deformation_type ON deformation_measurements(measurement_type);

CREATE INDEX idx_cycles_caldera ON uplift_subsidence_cycles(caldera_id);
CREATE INDEX idx_cycles_dates ON uplift_subsidence_cycles(cycle_start_date, cycle_end_date);

CREATE INDEX idx_validation_caldera ON caldera_validation_data(caldera_id);
CREATE INDEX idx_validation_period ON caldera_validation_data(observation_period_start, observation_period_end);

-- Views for common queries
CREATE OR REPLACE VIEW active_resurgent_calderas AS
SELECT 
    cg.*,
    COUNT(dm.id) as measurement_count,
    MAX(dm.measurement_timestamp) as last_measurement,
    AVG(dm.vertical_displacement) as avg_vertical_displacement,
    MAX(dm.vertical_displacement) as max_vertical_displacement
FROM caldera_geometries cg
LEFT JOIN deformation_measurements dm ON cg.caldera_id = dm.caldera_id
WHERE cg.monitoring_status = 'active'
GROUP BY cg.id;

CREATE OR REPLACE VIEW recent_uplift_mechanisms AS
SELECT 
    umr.*,
    cg.caldera_name,
    cg.location
FROM uplift_mechanism_results umr
JOIN caldera_geometries cg ON umr.caldera_id = cg.caldera_id
WHERE umr.simulation_timestamp > NOW() - INTERVAL '30 days'
ORDER BY umr.simulation_timestamp DESC;

-- Functions for data analysis
CREATE OR REPLACE FUNCTION calculate_deformation_rate(
    caldera_id_param VARCHAR(100),
    start_date TIMESTAMP,
    end_date TIMESTAMP
) RETURNS TABLE(
    location GEOMETRY,
    vertical_rate FLOAT,
    horizontal_rate FLOAT,
    measurement_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        dm.location,
        (MAX(dm.vertical_displacement) - MIN(dm.vertical_displacement)) / 
        EXTRACT(EPOCH FROM (end_date - start_date)) * 365.25 * 24 * 3600 as vertical_rate,
        SQRT(POWER(MAX(dm.horizontal_displacement_east) - MIN(dm.horizontal_displacement_east), 2) +
             POWER(MAX(dm.horizontal_displacement_north) - MIN(dm.horizontal_displacement_north), 2)) /
        EXTRACT(EPOCH FROM (end_date - start_date)) * 365.25 * 24 * 3600 as horizontal_rate,
        COUNT(*)::INTEGER as measurement_count
    FROM deformation_measurements dm
    WHERE dm.caldera_id = caldera_id_param
    AND dm.measurement_timestamp BETWEEN start_date AND end_date
    GROUP BY dm.location;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_uplift_mechanism_results_updated_at
    BEFORE UPDATE ON uplift_mechanism_results
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE uplift_mechanism_results IS 'Results from uplift mechanism modeling including magma intrusion, volatile exsolution, thermal expansion, and coupled processes';
COMMENT ON TABLE caldera_geometries IS 'Geometric properties and characteristics of resurgent caldera systems';
COMMENT ON TABLE magma_intrusions IS 'Properties and evolution of magma intrusions driving caldera resurgence';
COMMENT ON TABLE rock_properties IS 'Mechanical and thermal properties of host rocks in different geological units';
COMMENT ON TABLE hydrothermal_systems IS 'Properties and characteristics of hydrothermal systems affecting caldera deformation';
COMMENT ON TABLE deformation_measurements IS 'Time-series deformation measurements from various monitoring techniques';
COMMENT ON TABLE uplift_subsidence_cycles IS 'Analysis results for long-term uplift-subsidence cycles in resurgent calderas';
COMMENT ON TABLE caldera_validation_data IS 'Observational data from well-monitored calderas for model validation';
COMMENT ON TABLE model_validation_results IS 'Results from validating uplift mechanism models against observational data';
COMMENT ON TABLE uncertainty_quantification IS 'Uncertainty analysis results for model parameters and predictions';