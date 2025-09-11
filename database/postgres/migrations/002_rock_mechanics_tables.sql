-- =============================================================================
-- DATABASE MIGRATION 002: Rock Mechanics Tables
-- Version: 1.1.0
-- Description: Advanced rock mechanical properties with tensor calculations
-- =============================================================================

-- Migration metadata
INSERT INTO schema_migrations (version, description, applied_at) VALUES 
    ('002', 'Rock mechanics tables with anisotropic modeling', NOW())
ON CONFLICT (version) DO NOTHING;

-- Create comprehensive rock mechanical properties table
CREATE TABLE IF NOT EXISTS rock_mechanics.rock_mechanical_properties (
    id SERIAL PRIMARY KEY,
    formation_id VARCHAR(100) NOT NULL,
    lithology VARCHAR(100) NOT NULL,
    
    -- Basic mechanical properties
    unconfined_compressive_strength_mpa REAL CHECK (unconfined_compressive_strength_mpa > 0),
    tensile_strength_mpa REAL CHECK (tensile_strength_mpa > 0),
    cohesion_mpa REAL CHECK (cohesion_mpa >= 0),
    friction_angle_deg REAL CHECK (friction_angle_deg BETWEEN 0 AND 90),
    
    -- Elastic properties with anisotropic modeling
    elastic_modulus_gpa REAL CHECK (elastic_modulus_gpa > 0),
    poisson_ratio REAL CHECK (poisson_ratio BETWEEN -1 AND 0.5),
    shear_modulus_gpa REAL CHECK (shear_modulus_gpa > 0),
    bulk_modulus_gpa REAL CHECK (bulk_modulus_gpa > 0),
    
    -- Anisotropic elastic tensor components (Voigt notation)
    c11_gpa REAL, c12_gpa REAL, c13_gpa REAL, c14_gpa REAL, c15_gpa REAL, c16_gpa REAL,
    c22_gpa REAL, c23_gpa REAL, c24_gpa REAL, c25_gpa REAL, c26_gpa REAL,
    c33_gpa REAL, c34_gpa REAL, c35_gpa REAL, c36_gpa REAL,
    c44_gpa REAL, c45_gpa REAL, c46_gpa REAL,
    c55_gpa REAL, c56_gpa REAL,
    c66_gpa REAL,
    
    -- Hydraulic properties
    permeability_m2 REAL CHECK (permeability_m2 > 0),
    porosity REAL CHECK (porosity BETWEEN 0 AND 1),
    hydraulic_conductivity_m_s REAL,
    specific_storage_m REAL,
    
    -- Thermal properties
    thermal_conductivity_w_m_k REAL CHECK (thermal_conductivity_w_m_k > 0),
    thermal_expansion_per_k REAL CHECK (thermal_expansion_per_k > 0),
    specific_heat_capacity_j_kg_k REAL,
    thermal_diffusivity_m2_s REAL,
    
    -- Fracture and damage properties
    fracture_density_per_m3 REAL CHECK (fracture_density_per_m3 >= 0),
    fracture_orientation_tensor JSONB,
    damage_parameter REAL CHECK (damage_parameter BETWEEN 0 AND 1),
    crack_closure_stress_mpa REAL,
    
    -- Quality and metadata
    data_quality_grade CHAR(1) CHECK (data_quality_grade IN ('A', 'B', 'C', 'D')),
    test_standard VARCHAR(20) CHECK (test_standard IN ('ASTM', 'ISRM', 'BS', 'DIN', 'ISO')),
    sample_location GEOGRAPHY(POINTZ, 4326),
    collected_date DATE,
    laboratory VARCHAR(100),
    test_conditions JSONB,
    
    -- Temporal tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_rock_mech_lithology ON rock_mechanics.rock_mechanical_properties (lithology);
CREATE INDEX IF NOT EXISTS idx_rock_mech_location ON rock_mechanics.rock_mechanical_properties USING GIST (sample_location);
CREATE INDEX IF NOT EXISTS idx_rock_mech_quality ON rock_mechanics.rock_mechanical_properties (data_quality_grade);

-- Log migration completion
INSERT INTO migration_log (migration_version, status, executed_at, execution_time_ms) 
VALUES ('002', 'completed', NOW(), 0);