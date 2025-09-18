-- Stacked-Sill Complex Database Schema
-- Advanced database schema for stacked-sill intrusion complexes

-- Create sill-specific schema
CREATE SCHEMA IF NOT EXISTS sill_complex;

-- Sill complexes table
CREATE TABLE IF NOT EXISTS sill_complex.sill_complexes (
    complex_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    total_volume FLOAT, -- km³
    sill_count INTEGER,
    depth_range_min FLOAT, -- km
    depth_range_max FLOAT, -- km
    age_min FLOAT, -- Ma
    age_max FLOAT, -- Ma
    complex_type VARCHAR(100),
    host_rock_types TEXT[],
    emplacement_sequence JSONB,
    structural_controls JSONB,
    thermal_evolution JSONB,
    mechanical_evolution JSONB,
    hazard_assessment JSONB,
    uncertainty_analysis JSONB,
    validation_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual sills table
CREATE TABLE IF NOT EXISTS sill_complex.sills (
    sill_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    complex_id UUID REFERENCES sill_complex.sill_complexes(complex_id) ON DELETE CASCADE,
    name VARCHAR(255),
    center_coordinates GEOMETRY(POINT, 4326) NOT NULL,
    depth FLOAT NOT NULL, -- km
    lateral_extent JSONB, -- length, width in km
    thickness_distribution BYTEA, -- compressed numpy array
    volume FLOAT, -- km³
    surface_mesh BYTEA, -- compressed 3D surface mesh
    shape_parameters JSONB,
    orientation JSONB, -- strike, dip angles
    aspect_ratios JSONB, -- length/width, thickness/width
    irregularity_index FLOAT,
    emplacement_depth FLOAT, -- km
    host_rock_contact BYTEA, -- compressed contact surface
    internal_structure JSONB,
    uncertainty_bounds JSONB,
    confidence_level FLOAT DEFAULT 0.95,
    reconstruction_method VARCHAR(100),
    validation_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sill interconnections table
CREATE TABLE IF NOT EXISTS sill_complex.sill_interconnections (
    connection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_sill_id UUID REFERENCES sill_complex.sills(sill_id) ON DELETE CASCADE,
    child_sill_id UUID REFERENCES sill_complex.sills(sill_id) ON DELETE CASCADE,
    connection_type VARCHAR(50), -- 'direct', 'indirect', 'feeder'
    connection_geometry BYTEA, -- compressed 3D pathway
    flow_capacity FLOAT, -- m³/s
    pressure_drop FLOAT, -- Pa
    geometric_complexity FLOAT,
    activation_threshold FLOAT, -- Pa
    temporal_activity JSONB,
    uncertainty JSONB,
    validation_evidence TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Structural data table
CREATE TABLE IF NOT EXISTS sill_complex.structural_data (
    data_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    complex_id UUID REFERENCES sill_complex.sill_complexes(complex_id) ON DELETE CASCADE,
    location GEOMETRY(POINT, 4326) NOT NULL,
    fracture_networks BYTEA, -- compressed array
    bedding_planes BYTEA, -- compressed array
    fault_systems BYTEA, -- compressed array
    joint_sets BYTEA, -- compressed array
    lithological_contacts BYTEA, -- compressed array
    structural_measurements JSONB,
    coordinates JSONB,
    uncertainty JSONB,
    data_quality JSONB,
    measurement_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seismic structural data table
CREATE TABLE IF NOT EXISTS sill_complex.seismic_structural_data (
    data_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    complex_id UUID REFERENCES sill_complex.sill_complexes(complex_id) ON DELETE CASCADE,
    survey_name VARCHAR(255),
    location GEOMETRY(POLYGON, 4326) NOT NULL,
    reflection_data BYTEA, -- compressed 3D array
    coherence_attributes BYTEA, -- compressed array
    amplitude_attributes BYTEA, -- compressed array
    frequency_attributes BYTEA, -- compressed array
    coordinates JSONB,
    acquisition_geometry JSONB,
    processing_parameters JSONB,
    resolution JSONB,
    signal_to_noise FLOAT,
    acquisition_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Host rock properties table
CREATE TABLE IF NOT EXISTS sill_complex.host_rock_properties (
    property_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    complex_id UUID REFERENCES sill_complex.sill_complexes(complex_id) ON DELETE CASCADE,
    location GEOMETRY(POINT, 4326) NOT NULL,
    rock_type VARCHAR(100),
    mechanical_properties JSONB,
    thermal_properties JSONB,
    permeability FLOAT, -- m²
    porosity FLOAT, -- fraction
    density FLOAT, -- kg/m³
    elastic_moduli JSONB,
    failure_criteria JSONB,
    anisotropy_tensor BYTEA, -- compressed array
    layering_properties JSONB,
    alteration_zones JSONB,
    measurement_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sill evolution table
CREATE TABLE IF NOT EXISTS sill_complex.sill_evolution (
    evolution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sill_id UUID REFERENCES sill_complex.sills(sill_id) ON DELETE CASCADE,
    time_series TIMESTAMP[],
    geometry_evolution BYTEA, -- compressed evolution data
    thickness_evolution BYTEA, -- compressed array
    lateral_growth BYTEA, -- compressed array
    volume_evolution BYTEA, -- compressed array
    growth_mechanisms TEXT[],
    growth_rates JSONB,
    cooling_history BYTEA, -- compressed array
    crystallization_progress BYTEA, -- compressed array
    stress_evolution JSONB,
    prediction_horizon INTERVAL,
    confidence_intervals JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Validation complexes table
CREATE TABLE IF NOT EXISTS sill_complex.validation_complexes (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    complex_name VARCHAR(255) UNIQUE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    complex_type VARCHAR(100),
    total_volume FLOAT, -- km³
    sill_count INTEGER,
    depth_range JSONB, -- min, max in km
    age_range JSONB, -- min, max in Ma
    host_rocks TEXT[],
    reference TEXT,
    sill_geometries JSONB,
    interconnection_network JSONB,
    structural_controls JSONB,
    emplacement_mechanisms JSONB,
    thermal_evolution JSONB,
    validation_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create spatial indexes
CREATE INDEX IF NOT EXISTS idx_sill_complexes_location ON sill_complex.sill_complexes USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_sills_center_coordinates ON sill_complex.sills USING GIST (center_coordinates);
CREATE INDEX IF NOT EXISTS idx_sills_complex_id ON sill_complex.sills (complex_id);
CREATE INDEX IF NOT EXISTS idx_sill_interconnections_parent ON sill_complex.sill_interconnections (parent_sill_id);
CREATE INDEX IF NOT EXISTS idx_sill_interconnections_child ON sill_complex.sill_interconnections (child_sill_id);
CREATE INDEX IF NOT EXISTS idx_structural_data_location ON sill_complex.structural_data USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_seismic_data_location ON sill_complex.seismic_structural_data USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_host_rock_location ON sill_complex.host_rock_properties USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_validation_complexes_location ON sill_complex.validation_complexes USING GIST (location);

-- Create performance indexes
CREATE INDEX IF NOT EXISTS idx_sills_depth ON sill_complex.sills (depth);
CREATE INDEX IF NOT EXISTS idx_sills_volume ON sill_complex.sills (volume);
CREATE INDEX IF NOT EXISTS idx_sills_validation_score ON sill_complex.sills (validation_score);
CREATE INDEX IF NOT EXISTS idx_complexes_total_volume ON sill_complex.sill_complexes (total_volume);
CREATE INDEX IF NOT EXISTS idx_complexes_sill_count ON sill_complex.sill_complexes (sill_count);

-- Insert validation complex data
INSERT INTO sill_complex.validation_complexes (
    complex_name, location, complex_type, total_volume, sill_count, 
    depth_range, age_range, host_rocks, reference
) VALUES
    ('karoo_basin', ST_SetSRID(ST_MakePoint(24.0, -30.0), 4326), 'continental_flood_basalt', 
     2500.0, 150, '{"min": 0.5, "max": 8.0}', '{"min": 177, "max": 183}',
     ARRAY['sandstone', 'shale', 'mudstone'], 'Svensen et al. (2012), Galerne et al. (2008)'),
    ('ferrar_dolerite', ST_SetSRID(ST_MakePoint(160.0, -77.0), 4326), 'continental_flood_basalt',
     1800.0, 200, '{"min": 0.2, "max": 6.0}', '{"min": 182, "max": 184}',
     ARRAY['sandstone', 'coal_measures', 'shale'], 'Elliot & Fleming (2004), Ross et al. (2008)'),
    ('whin_sill', ST_SetSRID(ST_MakePoint(-2.5, 55.0), 4326), 'regional_sill_complex',
     45.0, 25, '{"min": 0.1, "max": 2.5}', '{"min": 295, "max": 295}',
     ARRAY['limestone', 'sandstone', 'shale'], 'Liss et al. (2004), Woodcock & Strachan (2012)'),
    ('palisades_sill', ST_SetSRID(ST_MakePoint(-74.0, 40.8), 4326), 'rift_related_sill',
     12.0, 8, '{"min": 0.05, "max": 1.0}', '{"min": 199, "max": 201}',
     ARRAY['sandstone', 'shale', 'conglomerate'], 'Puffer (2001), Husch (1990)'),
    ('midland_valley', ST_SetSRID(ST_MakePoint(-4.0, 56.0), 4326), 'rift_related_complex',
     180.0, 75, '{"min": 0.1, "max": 3.5}', '{"min": 330, "max": 340}',
     ARRAY['sandstone', 'limestone', 'volcanic_rocks'], 'Stephenson et al. (2003), Monaghan & Pringle (2004)')
ON CONFLICT (complex_name) DO NOTHING;