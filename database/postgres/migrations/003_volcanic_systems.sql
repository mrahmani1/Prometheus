-- =============================================================================
-- DATABASE MIGRATION 003: Volcanic Systems Tables
-- Version: 1.2.0
-- Description: Volcanic chambers and critical failure zones with 3D spatial indexing
-- =============================================================================

-- Migration metadata
INSERT INTO schema_migrations (version, description, applied_at) VALUES 
    ('003', 'Volcanic systems with 3D spatial indexing and temporal evolution', NOW())
ON CONFLICT (version) DO NOTHING;

-- Volcanic chambers with 3D spatial indexing and temporal evolution
CREATE TABLE IF NOT EXISTS volcanic_systems.volcanic_chambers (
    id SERIAL PRIMARY KEY,
    chamber_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- 3D spatial geometry
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    chamber_geometry GEOMETRY(POLYHEDRALSURFACEZ, 4326),
    
    -- Physical dimensions
    chamber_volume_km3 REAL CHECK (chamber_volume_km3 > 0),
    chamber_depth_km REAL CHECK (chamber_depth_km > 0),
    chamber_width_km REAL CHECK (chamber_width_km > 0),
    chamber_length_km REAL CHECK (chamber_length_km > 0),
    chamber_height_km REAL CHECK (chamber_height_km > 0),
    roof_depth_km REAL CHECK (roof_depth_km > 0),
    
    -- Chamber characteristics
    chamber_shape VARCHAR(50) CHECK (chamber_shape IN ('spherical', 'ellipsoid', 'cylindrical', 'irregular', 'sill-like', 'dike-like')),
    connectivity_factor REAL CHECK (connectivity_factor BETWEEN 0 AND 1),
    melt_fraction REAL CHECK (melt_fraction BETWEEN 0 AND 1),
    crystal_fraction REAL CHECK (crystal_fraction BETWEEN 0 AND 1),
    volatile_fraction REAL CHECK (volatile_fraction BETWEEN 0 AND 1),
    
    -- Thermodynamic properties
    temperature_c REAL CHECK (temperature_c > 0),
    pressure_mpa REAL CHECK (pressure_mpa > 0),
    density_kg_m3 REAL CHECK (density_kg_m3 > 0),
    viscosity_pa_s REAL CHECK (viscosity_pa_s > 0),
    
    -- Volatile content
    volatile_content JSONB DEFAULT '{
        "h2o_wt_percent": null,
        "co2_ppm": null,
        "so2_ppm": null,
        "h2s_ppm": null,
        "hf_ppm": null,
        "hcl_ppm": null,
        "total_volatiles_wt_percent": null
    }'::jsonb,
    
    -- Temporal evolution tracking
    formation_age_ma REAL,
    last_recharge_event TIMESTAMPTZ,
    evolution_history JSONB,
    
    -- Data quality and provenance
    data_source VARCHAR(100),
    confidence_level REAL CHECK (confidence_level BETWEEN 0 AND 1) DEFAULT 0.5,
    detection_method VARCHAR(100),
    
    -- Temporal tracking
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Critical failure zones with probabilistic risk assessment
CREATE TABLE IF NOT EXISTS volcanic_systems.critical_failure_zones (
    id SERIAL PRIMARY KEY,
    zone_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Spatial definition
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    critical_point GEOGRAPHY(POINTZ, 4326) NOT NULL,
    failure_zone_geometry GEOMETRY(POLYGONZ, 4326),
    
    -- Failure mechanics
    depth_km REAL CHECK (depth_km > 0),
    failure_mechanism VARCHAR(50) CHECK (failure_mechanism IN ('mohr_coulomb', 'tensile', 'thermal', 'combined', 'hydraulic_fracturing', 'fatigue')),
    stress_ratio REAL CHECK (stress_ratio >= 0),
    safety_factor REAL CHECK (safety_factor > 0),
    
    -- Stress state
    principal_stress_1_mpa REAL,
    principal_stress_2_mpa REAL,
    principal_stress_3_mpa REAL,
    shear_stress_mpa REAL,
    normal_stress_mpa REAL,
    
    -- Probabilistic assessment
    failure_probability REAL CHECK (failure_probability BETWEEN 0 AND 1),
    confidence_level REAL CHECK (confidence_level BETWEEN 0 AND 1),
    uncertainty_bounds JSONB,
    
    -- Historical data
    historical_failures INTEGER DEFAULT 0,
    last_failure_date DATE,
    failure_recurrence_interval_years REAL,
    
    -- Real-time monitoring integration
    monitoring_data JSONB DEFAULT '{
        "seismic_activity": null,
        "deformation_rate": null,
        "temperature_anomaly": null,
        "gas_emissions": null,
        "ground_tilt": null
    }'::jsonb,
    
    -- Temporal tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_volcanic_chambers_location ON volcanic_systems.volcanic_chambers USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_volcanic_chambers_geometry ON volcanic_systems.volcanic_chambers USING GIST (chamber_geometry);
CREATE INDEX IF NOT EXISTS idx_critical_zones_location ON volcanic_systems.critical_failure_zones USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_critical_zones_geometry ON volcanic_systems.critical_failure_zones USING GIST (failure_zone_geometry);

-- Log migration completion
INSERT INTO migration_log (migration_version, status, executed_at, execution_time_ms) 
VALUES ('003', 'completed', NOW(), 0);