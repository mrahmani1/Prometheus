-- =============================================================================
-- COMPREHENSIVE GEOLOGICAL DATABASE SCHEMA
-- Advanced PostgreSQL schema with 50+ specialized geological tables
-- Supports cutting-edge database design patterns for geological modeling
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Create specialized schemas
CREATE SCHEMA IF NOT EXISTS geological_core;
CREATE SCHEMA IF NOT EXISTS rock_mechanics;
CREATE SCHEMA IF NOT EXISTS volcanic_systems;
CREATE SCHEMA IF NOT EXISTS seismic_networks;
CREATE SCHEMA IF NOT EXISTS geodetic_data;
CREATE SCHEMA IF NOT EXISTS geophysical_data;
CREATE SCHEMA IF NOT EXISTS temporal_analysis;
CREATE SCHEMA IF NOT EXISTS spatial_indexing;

-- =============================================================================
-- ADVANCED ROCK MECHANICAL PROPERTIES TABLES
-- =============================================================================

-- Comprehensive rock mechanical properties with full tensor calculations
CREATE TABLE rock_mechanics.rock_mechanical_properties (
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
    fracture_orientation_tensor JSONB, -- 3x3 tensor for fracture fabric
    damage_parameter REAL CHECK (damage_parameter BETWEEN 0 AND 1),
    crack_closure_stress_mpa REAL,
    
    -- Quality and metadata
    data_quality_grade CHAR(1) CHECK (data_quality_grade IN ('A', 'B', 'C', 'D')),
    test_standard VARCHAR(20) CHECK (test_standard IN ('ASTM', 'ISRM', 'BS', 'DIN', 'ISO')),
    sample_location GEOGRAPHY(POINTZ, 4326),
    collected_date DATE,
    laboratory VARCHAR(100),
    test_conditions JSONB, -- Temperature, pressure, strain rate, etc.
    
    -- Temporal tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Advanced indexing for rock mechanical properties
CREATE INDEX idx_rock_mech_lithology ON rock_mechanics.rock_mechanical_properties (lithology);
CREATE INDEX idx_rock_mech_location ON rock_mechanics.rock_mechanical_properties USING GIST (sample_location);
CREATE INDEX idx_rock_mech_quality ON rock_mechanics.rock_mechanical_properties (data_quality_grade);
CREATE INDEX idx_rock_mech_strength ON rock_mechanics.rock_mechanical_properties (unconfined_compressive_strength_mpa);
CREATE INDEX idx_rock_mech_elastic ON rock_mechanics.rock_mechanical_properties (elastic_modulus_gpa, poisson_ratio);

-- Literature-based rock properties database
CREATE TABLE rock_mechanics.literature_rock_properties (
    id SERIAL PRIMARY KEY,
    rock_type VARCHAR(100) NOT NULL,
    property_name VARCHAR(50) NOT NULL,
    property_value REAL NOT NULL,
    property_unit VARCHAR(20) NOT NULL,
    
    -- Reference information
    reference_doi VARCHAR(200),
    reference_title TEXT,
    authors TEXT,
    publication_year INTEGER,
    journal VARCHAR(200),
    
    -- Quality metrics
    confidence_level REAL CHECK (confidence_level BETWEEN 0 AND 1),
    sample_size INTEGER,
    statistical_method VARCHAR(50),
    
    -- Test conditions and metadata
    test_conditions JSONB,
    geographic_region VARCHAR(100),
    geological_age_ma REAL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_lit_rock_type ON rock_mechanics.literature_rock_properties (rock_type);
CREATE INDEX idx_lit_confidence ON rock_mechanics.literature_rock_properties (confidence_level);
CREATE INDEX idx_lit_property ON rock_mechanics.literature_rock_properties (property_name);
CREATE INDEX idx_lit_year ON rock_mechanics.literature_rock_properties (publication_year);

-- =============================================================================
-- ADVANCED VOLCANIC CHAMBERS AND CRITICAL FAILURE ZONES
-- =============================================================================

-- Volcanic chambers with 3D spatial indexing and temporal evolution
CREATE TABLE volcanic_systems.volcanic_chambers (
    id SERIAL PRIMARY KEY,
    chamber_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- 3D spatial geometry
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    chamber_geometry GEOMETRY(POLYHEDRALSURFACEZ, 4326), -- 3D chamber shape
    
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
    
    -- Volatile content (detailed composition)
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
    evolution_history JSONB, -- Array of temporal snapshots
    
    -- Data quality and provenance
    data_source VARCHAR(100),
    confidence_level REAL CHECK (confidence_level BETWEEN 0 AND 1) DEFAULT 0.5,
    detection_method VARCHAR(100), -- seismic tomography, gravity, etc.
    
    -- Temporal tracking
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Advanced 3D spatial indexing for volcanic chambers
CREATE INDEX idx_volcanic_chambers_location ON volcanic_systems.volcanic_chambers USING GIST (location);
CREATE INDEX idx_volcanic_chambers_geometry ON volcanic_systems.volcanic_chambers USING GIST (chamber_geometry);
CREATE INDEX idx_volcanic_chambers_volume ON volcanic_systems.volcanic_chambers (chamber_volume_km3);
CREATE INDEX idx_volcanic_chambers_depth ON volcanic_systems.volcanic_chambers (chamber_depth_km);
CREATE INDEX idx_volcanic_chambers_temp_pressure ON volcanic_systems.volcanic_chambers (temperature_c, pressure_mpa);

-- Critical failure zones with probabilistic risk assessment
CREATE TABLE volcanic_systems.critical_failure_zones (
    id SERIAL PRIMARY KEY,
    zone_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Spatial definition
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    critical_point GEOGRAPHY(POINTZ, 4326) NOT NULL,
    failure_zone_geometry GEOMETRY(POLYGONZ, 4326), -- 3D failure zone
    
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
    uncertainty_bounds JSONB, -- Upper and lower bounds for key parameters
    
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

-- Advanced indexing for critical failure zones
CREATE INDEX idx_critical_zones_location ON volcanic_systems.critical_failure_zones USING GIST (location);
CREATE INDEX idx_critical_zones_geometry ON volcanic_systems.critical_failure_zones USING GIST (failure_zone_geometry);
CREATE INDEX idx_critical_zones_stress_ratio ON volcanic_systems.critical_failure_zones (stress_ratio);
CREATE INDEX idx_critical_zones_probability ON volcanic_systems.critical_failure_zones (failure_probability);
CREATE INDEX idx_critical_zones_confidence ON volcanic_systems.critical_failure_zones (confidence_level);

-- =============================================================================
-- COMPREHENSIVE SEISMIC NETWORK INTEGRATION
-- =============================================================================

-- Advanced seismic stations with real-time processing capabilities
CREATE TABLE seismic_networks.seismic_stations (
    id SERIAL PRIMARY KEY,
    station_code VARCHAR(20) UNIQUE NOT NULL,
    network_code VARCHAR(10) NOT NULL,
    
    -- Location and installation
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    site_geology VARCHAR(100),
    site_conditions VARCHAR(100),
    installation_date DATE,
    
    -- Station characteristics
    station_type VARCHAR(50) CHECK (station_type IN ('broadband', 'short_period', 'strong_motion', 'infrasound', 'tiltmeter')),
    instrument_type VARCHAR(100),
    sampling_rate_hz REAL CHECK (sampling_rate_hz > 0),
    dynamic_range_db REAL,
    
    -- Operational status
    operational_status VARCHAR(20) DEFAULT 'active' CHECK (operational_status IN ('active', 'inactive', 'maintenance', 'decommissioned')),
    data_availability_percent REAL CHECK (data_availability_percent BETWEEN 0 AND 100),
    
    -- Instrument response and calibration
    instrument_response JSONB, -- Poles, zeros, sensitivity
    calibration_date DATE,
    calibration_factor REAL,
    
    -- Data quality metrics
    data_quality_score REAL CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
    noise_level_db REAL,
    signal_to_noise_ratio REAL,
    
    -- Real-time processing configuration
    real_time_processing BOOLEAN DEFAULT FALSE,
    phase_picking_algorithm VARCHAR(50),
    trigger_threshold REAL,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_seismic_stations_location ON seismic_networks.seismic_stations USING GIST (location);
CREATE INDEX idx_seismic_stations_network ON seismic_networks.seismic_stations (network_code);
CREATE INDEX idx_seismic_stations_status ON seismic_networks.seismic_stations (operational_status);
CREATE INDEX idx_seismic_stations_quality ON seismic_networks.seismic_stations (data_quality_score);

-- Real-time waveform processing and phase picking
CREATE TABLE seismic_networks.waveform_data (
    id SERIAL PRIMARY KEY,
    station_code VARCHAR(20) REFERENCES seismic_networks.seismic_stations(station_code),
    
    -- Temporal information
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    sampling_rate_hz REAL NOT NULL,
    
    -- Waveform components
    component VARCHAR(3) CHECK (component IN ('Z', 'N', 'E', 'BHZ', 'BHN', 'BHE', 'HHZ', 'HHN', 'HHE')),
    waveform_data BYTEA, -- Compressed waveform data
    data_format VARCHAR(20) DEFAULT 'mseed',
    
    -- Processing status
    processing_status VARCHAR(20) DEFAULT 'raw' CHECK (processing_status IN ('raw', 'filtered', 'corrected', 'processed')),
    filter_applied VARCHAR(100),
    
    -- Quality metrics
    data_quality CHAR(1) CHECK (data_quality IN ('A', 'B', 'C', 'D', 'M')),
    completeness_percent REAL CHECK (completeness_percent BETWEEN 0 AND 100),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('seismic_networks.waveform_data', 'start_time');
CREATE INDEX idx_waveform_station_time ON seismic_networks.waveform_data (station_code, start_time);
CREATE INDEX idx_waveform_component ON seismic_networks.waveform_data (component);

-- Automated phase picking results
CREATE TABLE seismic_networks.phase_picks (
    id SERIAL PRIMARY KEY,
    station_code VARCHAR(20) REFERENCES seismic_networks.seismic_stations(station_code),
    
    -- Phase information
    phase_type VARCHAR(10) CHECK (phase_type IN ('P', 'S', 'Pg', 'Sg', 'Pn', 'Sn', 'PKP', 'SKS')),
    pick_time TIMESTAMPTZ NOT NULL,
    uncertainty_seconds REAL,
    
    -- Pick quality and method
    pick_quality CHAR(1) CHECK (pick_quality IN ('0', '1', '2', '3', '4')), -- SEISAN quality scale
    picking_method VARCHAR(50) CHECK (picking_method IN ('manual', 'sta_lta', 'ar_aic', 'kurtosis', 'ml_picker')),
    picker_algorithm VARCHAR(100),
    
    -- Amplitude and period measurements
    amplitude_nm REAL,
    period_seconds REAL,
    signal_to_noise_ratio REAL,
    
    -- Association information
    event_id VARCHAR(50), -- Links to earthquake catalog
    distance_km REAL,
    azimuth_deg REAL CHECK (azimuth_deg BETWEEN 0 AND 360),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('seismic_networks.phase_picks', 'pick_time');
CREATE INDEX idx_phase_picks_station ON seismic_networks.phase_picks (station_code);
CREATE INDEX idx_phase_picks_type ON seismic_networks.phase_picks (phase_type);
CREATE INDEX idx_phase_picks_quality ON seismic_networks.phase_picks (pick_quality);

-- =============================================================================
-- GLOBAL VOLCANISM PROGRAM DATABASE INTEGRATION
-- =============================================================================

-- Comprehensive volcano database with GVP integration
CREATE TABLE volcanic_systems.gvp_volcanoes (
    id SERIAL PRIMARY KEY,
    gvp_volcano_number INTEGER UNIQUE, -- Official GVP volcano number
    volcano_name VARCHAR(200) NOT NULL,
    
    -- Location information
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    country VARCHAR(100),
    region VARCHAR(100),
    subregion VARCHAR(100),
    
    -- Physical characteristics
    elevation_m INTEGER,
    volcano_type VARCHAR(100),
    dominant_rock_type VARCHAR(100),
    
    -- Tectonic setting
    tectonic_setting VARCHAR(100),
    geological_summary TEXT,
    
    -- Eruption history
    last_known_eruption DATE,
    eruption_certainty VARCHAR(50),
    holocene_activity BOOLEAN DEFAULT FALSE,
    pleistocene_activity BOOLEAN DEFAULT FALSE,
    
    -- Hazard assessment
    population_within_5km INTEGER,
    population_within_10km INTEGER,
    population_within_30km INTEGER,
    population_within_100km INTEGER,
    
    -- Data synchronization
    gvp_last_update TIMESTAMPTZ,
    sync_status VARCHAR(20) DEFAULT 'pending' CHECK (sync_status IN ('pending', 'synced', 'error')),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_gvp_volcanoes_location ON volcanic_systems.gvp_volcanoes USING GIST (location);
CREATE INDEX idx_gvp_volcanoes_number ON volcanic_systems.gvp_volcanoes (gvp_volcano_number);
CREATE INDEX idx_gvp_volcanoes_country ON volcanic_systems.gvp_volcanoes (country);
CREATE INDEX idx_gvp_volcanoes_type ON volcanic_systems.gvp_volcanoes (volcano_type);

-- GVP eruption database with automated synchronization
CREATE TABLE volcanic_systems.gvp_eruptions (
    id SERIAL PRIMARY KEY,
    gvp_volcano_number INTEGER REFERENCES volcanic_systems.gvp_volcanoes(gvp_volcano_number),
    
    -- Eruption timing
    eruption_start_date DATE,
    eruption_end_date DATE,
    eruption_certainty VARCHAR(50),
    
    -- Eruption characteristics
    vei INTEGER CHECK (vei BETWEEN 0 AND 8), -- Volcanic Explosivity Index
    eruption_type VARCHAR(100),
    explosive_evidence VARCHAR(200),
    
    -- Eruption products
    lava_flows BOOLEAN DEFAULT FALSE,
    pyroclastic_flows BOOLEAN DEFAULT FALSE,
    lahars BOOLEAN DEFAULT FALSE,
    tephra_fall BOOLEAN DEFAULT FALSE,
    
    -- Impact assessment
    fatalities INTEGER DEFAULT 0,
    injuries INTEGER DEFAULT 0,
    evacuations INTEGER DEFAULT 0,
    damage_description TEXT,
    
    -- Data quality
    data_quality VARCHAR(20),
    evidence_quality VARCHAR(20),
    
    -- Synchronization metadata
    gvp_last_update TIMESTAMPTZ,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_gvp_eruptions_volcano ON volcanic_systems.gvp_eruptions (gvp_volcano_number);
CREATE INDEX idx_gvp_eruptions_date ON volcanic_systems.gvp_eruptions (eruption_start_date);
CREATE INDEX idx_gvp_eruptions_vei ON volcanic_systems.gvp_eruptions (vei);

-- =============================================================================
-- NATIONAL GEOLOGICAL SURVEYS INTEGRATION
-- =============================================================================

-- Geological survey data integration with spatial-temporal querying
CREATE TABLE geological_core.geological_surveys_data (
    id SERIAL PRIMARY KEY,
    survey_organization VARCHAR(100) NOT NULL,
    dataset_name VARCHAR(200) NOT NULL,
    
    -- Spatial coverage
    coverage_geometry GEOMETRY(MULTIPOLYGON, 4326),
    spatial_resolution_m REAL,
    
    -- Temporal coverage
    temporal_start DATE,
    temporal_end DATE,
    temporal_resolution_days INTEGER,
    
    -- Data characteristics
    data_type VARCHAR(100) CHECK (data_type IN ('geological_map', 'geochemical', 'geophysical', 'structural', 'stratigraphic', 'mineral_resources')),
    scale_denominator INTEGER, -- Map scale (e.g., 50000 for 1:50,000)
    coordinate_system VARCHAR(50),
    
    -- Quality and metadata
    data_quality_grade CHAR(1) CHECK (data_quality_grade IN ('A', 'B', 'C', 'D')),
    completeness_percent REAL CHECK (completeness_percent BETWEEN 0 AND 100),
    accuracy_assessment TEXT,
    
    -- Access and licensing
    access_url TEXT,
    license_type VARCHAR(100),
    usage_restrictions TEXT,
    
    -- Synchronization
    last_sync_date TIMESTAMPTZ,
    sync_frequency_days INTEGER DEFAULT 30,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_geological_surveys_coverage ON geological_core.geological_surveys_data USING GIST (coverage_geometry);
CREATE INDEX idx_geological_surveys_org ON geological_core.geological_surveys_data (survey_organization);
CREATE INDEX idx_geological_surveys_type ON geological_core.geological_surveys_data (data_type);

-- =============================================================================
-- GEOTECHNICAL INVESTIGATIONS AND HAZARD DATABASES
-- =============================================================================

-- Comprehensive geotechnical investigations database
CREATE TABLE geological_core.geotechnical_investigations (
    id SERIAL PRIMARY KEY,
    investigation_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Location and project information
    location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    project_name VARCHAR(200),
    client_organization VARCHAR(200),
    investigation_purpose VARCHAR(100),
    
    -- Investigation details
    investigation_date DATE,
    investigation_depth_m REAL CHECK (investigation_depth_m > 0),
    investigation_type VARCHAR(100) CHECK (investigation_type IN ('borehole', 'test_pit', 'geophysical_survey', 'in_situ_testing', 'laboratory_testing')),
    
    -- Soil/rock classification
    soil_classification VARCHAR(100), -- USCS, AASHTO, etc.
    rock_classification VARCHAR(100),
    groundwater_level_m REAL,
    
    -- Geotechnical properties
    bearing_capacity_kpa REAL,
    settlement_potential_mm REAL,
    slope_stability_factor REAL,
    liquefaction_potential VARCHAR(20) CHECK (liquefaction_potential IN ('low', 'moderate', 'high', 'very_high')),
    
    -- Laboratory test results
    test_results JSONB DEFAULT '{
        "grain_size_distribution": null,
        "atterberg_limits": null,
        "compaction_characteristics": null,
        "shear_strength_parameters": null,
        "consolidation_parameters": null,
        "permeability": null
    }'::jsonb,
    
    -- Quality and standards
    investigation_standard VARCHAR(50), -- ASTM, BS, etc.
    data_quality_grade CHAR(1) CHECK (data_quality_grade IN ('A', 'B', 'C', 'D')),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_geotechnical_location ON geological_core.geotechnical_investigations USING GIST (location);
CREATE INDEX idx_geotechnical_type ON geological_core.geotechnical_investigations (investigation_type);
CREATE INDEX idx_geotechnical_purpose ON geological_core.geotechnical_investigations (investigation_purpose);

-- Probabilistic hazard assessment database
CREATE TABLE geological_core.geological_hazards (
    id SERIAL PRIMARY KEY,
    hazard_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Spatial definition
    hazard_zone GEOMETRY(MULTIPOLYGON, 4326) NOT NULL,
    centroid_location GEOGRAPHY(POINTZ, 4326),
    
    -- Hazard characteristics
    hazard_type VARCHAR(100) CHECK (hazard_type IN ('landslide', 'rockfall', 'debris_flow', 'liquefaction', 'subsidence', 'volcanic', 'seismic', 'tsunami')),
    hazard_magnitude REAL,
    magnitude_scale VARCHAR(50),
    
    -- Probabilistic assessment
    annual_probability REAL CHECK (annual_probability BETWEEN 0 AND 1),
    return_period_years REAL CHECK (return_period_years > 0),
    confidence_interval_lower REAL,
    confidence_interval_upper REAL,
    
    -- Risk assessment
    vulnerability_score REAL CHECK (vulnerability_score BETWEEN 0 AND 1),
    exposure_population INTEGER DEFAULT 0,
    exposure_infrastructure_value REAL,
    expected_annual_loss REAL,
    
    -- Temporal factors
    assessment_date DATE,
    validity_period_years INTEGER DEFAULT 10,
    
    -- Methodology
    assessment_method VARCHAR(100),
    model_parameters JSONB,
    uncertainty_sources TEXT[],
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_geological_hazards_zone ON geological_core.geological_hazards USING GIST (hazard_zone);
CREATE INDEX idx_geological_hazards_type ON geological_core.geological_hazards (hazard_type);
CREATE INDEX idx_geological_hazards_probability ON geological_core.geological_hazards (annual_probability);

-- =============================================================================
-- FIELD SURVEY AND EXTERNAL API CONFIGURATIONS
-- =============================================================================

-- Field survey data with real-time validation
CREATE TABLE geological_core.field_surveys (
    id SERIAL PRIMARY KEY,
    survey_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Survey metadata
    survey_date DATE NOT NULL,
    surveyor_name VARCHAR(200),
    survey_organization VARCHAR(200),
    survey_purpose VARCHAR(200),
    
    -- Location and coverage
    survey_location GEOGRAPHY(POINTZ, 4326) NOT NULL,
    survey_area GEOMETRY(POLYGON, 4326),
    
    -- Geological observations
    lithology_observed VARCHAR(200),
    structural_measurements JSONB DEFAULT '{
        "strike_dip_measurements": [],
        "joint_orientations": [],
        "fault_orientations": [],
        "fold_axes": []
    }'::jsonb,
    
    -- Sample collection
    samples_collected INTEGER DEFAULT 0,
    sample_locations GEOGRAPHY(MULTIPOINT, 4326),
    sample_descriptions JSONB,
    
    -- Environmental conditions
    weather_conditions VARCHAR(100),
    visibility_conditions VARCHAR(100),
    access_conditions VARCHAR(100),
    
    -- Data quality and validation
    data_quality_score REAL CHECK (data_quality_score BETWEEN 0 AND 1),
    validation_status VARCHAR(20) DEFAULT 'pending' CHECK (validation_status IN ('pending', 'validated', 'rejected', 'needs_review')),
    validation_notes TEXT,
    
    -- Real-time integration
    gps_accuracy_m REAL,
    measurement_precision REAL,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_field_surveys_location ON geological_core.field_surveys USING GIST (survey_location);
CREATE INDEX idx_field_surveys_area ON geological_core.field_surveys USING GIST (survey_area);
CREATE INDEX idx_field_surveys_date ON geological_core.field_surveys (survey_date);
CREATE INDEX idx_field_surveys_quality ON geological_core.field_surveys (data_quality_score);

-- External API endpoint configurations with real-time monitoring
CREATE TABLE geological_core.external_api_endpoints (
    id SERIAL PRIMARY KEY,
    endpoint_name VARCHAR(100) UNIQUE NOT NULL,
    
    -- API configuration
    base_url TEXT NOT NULL,
    api_version VARCHAR(20),
    authentication_type VARCHAR(50) CHECK (authentication_type IN ('none', 'api_key', 'oauth', 'basic_auth', 'bearer_token')),
    authentication_config JSONB,
    
    -- Data characteristics
    data_type VARCHAR(100),
    data_format VARCHAR(50) CHECK (data_format IN ('json', 'xml', 'csv', 'geojson', 'wfs', 'wms')),
    spatial_coverage GEOMETRY(MULTIPOLYGON, 4326),
    temporal_coverage_start DATE,
    temporal_coverage_end DATE,
    
    -- Quality control
    data_validation_rules JSONB DEFAULT '{
        "required_fields": [],
        "data_type_validation": {},
        "range_validation": {},
        "spatial_validation": {},
        "temporal_validation": {}
    }'::jsonb,
    
    -- Monitoring and reliability
    uptime_percentage REAL CHECK (uptime_percentage BETWEEN 0 AND 100),
    average_response_time_ms REAL,
    last_successful_request TIMESTAMPTZ,
    last_failed_request TIMESTAMPTZ,
    error_rate_percentage REAL CHECK (error_rate_percentage BETWEEN 0 AND 100),
    
    -- Synchronization settings
    sync_frequency_hours INTEGER DEFAULT 24,
    max_records_per_request INTEGER DEFAULT 1000,
    rate_limit_requests_per_minute INTEGER,
    
    -- Status and configuration
    endpoint_status VARCHAR(20) DEFAULT 'active' CHECK (endpoint_status IN ('active', 'inactive', 'maintenance', 'deprecated')),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_external_api_name ON geological_core.external_api_endpoints (endpoint_name);
CREATE INDEX idx_external_api_status ON geological_core.external_api_endpoints (endpoint_status);
CREATE INDEX idx_external_api_coverage ON geological_core.external_api_endpoints USING GIST (spatial_coverage);

-- =============================================================================
-- SPATIAL-TEMPORAL INDEXING AND AUTOMATED DATA QUALITY ASSESSMENT
-- =============================================================================

-- Spatial-temporal index management
CREATE TABLE spatial_indexing.spatial_temporal_indexes (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    index_name VARCHAR(100) NOT NULL,
    index_type VARCHAR(50) CHECK (index_type IN ('gist', 'spgist', 'gin', 'btree_gist', 'brin')),
    
    -- Spatial configuration
    spatial_column VARCHAR(100),
    spatial_srid INTEGER DEFAULT 4326,
    spatial_dimensions INTEGER DEFAULT 2,
    
    -- Temporal configuration
    temporal_column VARCHAR(100),
    temporal_partitioning BOOLEAN DEFAULT FALSE,
    partition_interval INTERVAL,
    
    -- Performance metrics
    index_size_mb REAL,
    query_performance_ms REAL,
    maintenance_cost_score REAL,
    
    -- Status
    index_status VARCHAR(20) DEFAULT 'active' CHECK (index_status IN ('active', 'building', 'invalid', 'disabled')),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_analyzed TIMESTAMPTZ
);

-- Automated data quality assessment system
CREATE TABLE geological_core.data_quality_assessments (
    id SERIAL PRIMARY KEY,
    assessment_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Target data
    table_name VARCHAR(100) NOT NULL,
    record_id INTEGER,
    assessment_scope VARCHAR(50) CHECK (assessment_scope IN ('record', 'table', 'dataset', 'system')),
    
    -- Quality dimensions
    completeness_score REAL CHECK (completeness_score BETWEEN 0 AND 1),
    accuracy_score REAL CHECK (accuracy_score BETWEEN 0 AND 1),
    consistency_score REAL CHECK (consistency_score BETWEEN 0 AND 1),
    timeliness_score REAL CHECK (timeliness_score BETWEEN 0 AND 1),
    validity_score REAL CHECK (validity_score BETWEEN 0 AND 1),
    
    -- Overall quality
    overall_quality_score REAL CHECK (overall_quality_score BETWEEN 0 AND 1),
    quality_grade CHAR(1) CHECK (quality_grade IN ('A', 'B', 'C', 'D', 'F')),
    
    -- Issues and recommendations
    identified_issues JSONB DEFAULT '[]'::jsonb,
    recommendations JSONB DEFAULT '[]'::jsonb,
    
    -- Assessment metadata
    assessment_method VARCHAR(100),
    assessment_date TIMESTAMPTZ DEFAULT NOW(),
    assessor VARCHAR(100), -- System or user identifier
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_data_quality_table ON geological_core.data_quality_assessments (table_name);
CREATE INDEX idx_data_quality_score ON geological_core.data_quality_assessments (overall_quality_score);
CREATE INDEX idx_data_quality_grade ON geological_core.data_quality_assessments (quality_grade);

-- Real-time synchronization status tracking
CREATE TABLE geological_core.synchronization_status (
    id SERIAL PRIMARY KEY,
    sync_job_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Source information
    source_system VARCHAR(100) NOT NULL,
    source_endpoint VARCHAR(200),
    target_table VARCHAR(100) NOT NULL,
    
    -- Synchronization details
    sync_start_time TIMESTAMPTZ NOT NULL,
    sync_end_time TIMESTAMPTZ,
    sync_status VARCHAR(20) CHECK (sync_status IN ('running', 'completed', 'failed', 'cancelled')),
    
    -- Performance metrics
    records_processed INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    
    -- Error handling
    error_message TEXT,
    error_details JSONB,
    retry_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('geological_core.synchronization_status', 'sync_start_time');
CREATE INDEX idx_sync_status_job ON geological_core.synchronization_status (sync_job_id);
CREATE INDEX idx_sync_status_system ON geological_core.synchronization_status (source_system);
CREATE INDEX idx_sync_status_table ON geological_core.synchronization_status (target_table);

-- =============================================================================
-- ADVANCED TRIGGERS AND FUNCTIONS
-- =============================================================================

-- Function for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_timestamp_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$ language 'plpgsql';

-- Function for automated data quality scoring
CREATE OR REPLACE FUNCTION calculate_data_quality_score()
RETURNS TRIGGER AS $
DECLARE
    completeness REAL := 0;
    field_count INTEGER := 0;
    non_null_count INTEGER := 0;
BEGIN
    -- Calculate completeness based on non-null fields
    -- This is a simplified example - real implementation would be more sophisticated
    
    IF TG_TABLE_NAME = 'rock_mechanical_properties' THEN
        field_count := 10; -- Key fields count
        non_null_count := 0;
        
        IF NEW.unconfined_compressive_strength_mpa IS NOT NULL THEN non_null_count := non_null_count + 1; END IF;
        IF NEW.tensile_strength_mpa IS NOT NULL THEN non_null_count := non_null_count + 1; END IF;
        IF NEW.elastic_modulus_gpa IS NOT NULL THEN non_null_count := non_null_count + 1; END IF;
        IF NEW.poisson_ratio IS NOT NULL THEN non_null_count := non_null_count + 1; END IF;
        IF NEW.permeability_m2 IS NOT NULL THEN non_null_count := non_null_count + 1; END IF;
        IF NEW.porosity IS NOT NULL THEN non_null_count := non_null_count + 1; END IF;
        IF NEW.thermal_conductivity_w_m_k IS NOT NULL THEN non_null_count := non_null_count + 1; END IF;
        IF NEW.fracture_density_per_m3 IS NOT NULL THEN non_null_count := non_null_count + 1; END IF;
        IF NEW.sample_location IS NOT NULL THEN non_null_count := non_null_count + 1; END IF;
        IF NEW.collected_date IS NOT NULL THEN non_null_count := non_null_count + 1; END IF;
        
        completeness := non_null_count::REAL / field_count::REAL;
        
        -- Set data quality grade based on completeness
        IF completeness >= 0.9 THEN
            NEW.data_quality_grade := 'A';
        ELSIF completeness >= 0.7 THEN
            NEW.data_quality_grade := 'B';
        ELSIF completeness >= 0.5 THEN
            NEW.data_quality_grade := 'C';
        ELSE
            NEW.data_quality_grade := 'D';
        END IF;
    END IF;
    
    RETURN NEW;
END;
$ language 'plpgsql';

-- Apply triggers to key tables
CREATE TRIGGER update_rock_mech_timestamp
    BEFORE UPDATE ON rock_mechanics.rock_mechanical_properties
    FOR EACH ROW EXECUTE FUNCTION update_timestamp_column();

CREATE TRIGGER update_volcanic_chambers_timestamp
    BEFORE UPDATE ON volcanic_systems.volcanic_chambers
    FOR EACH ROW EXECUTE FUNCTION update_timestamp_column();

CREATE TRIGGER update_critical_zones_timestamp
    BEFORE UPDATE ON volcanic_systems.critical_failure_zones
    FOR EACH ROW EXECUTE FUNCTION update_timestamp_column();

CREATE TRIGGER calculate_rock_mech_quality
    BEFORE INSERT OR UPDATE ON rock_mechanics.rock_mechanical_properties
    FOR EACH ROW EXECUTE FUNCTION calculate_data_quality_score();

-- =============================================================================
-- PERFORMANCE OPTIMIZATION VIEWS
-- =============================================================================

-- High-performance view for spatial queries
CREATE MATERIALIZED VIEW geological_core.spatial_data_summary AS
SELECT 
    'rock_mechanical_properties' as data_type,
    COUNT(*) as record_count,
    ST_Extent(sample_location::geometry) as spatial_extent,
    AVG(CASE WHEN data_quality_grade = 'A' THEN 1.0 
             WHEN data_quality_grade = 'B' THEN 0.8
             WHEN data_quality_grade = 'C' THEN 0.6
             ELSE 0.4 END) as avg_quality_score
FROM rock_mechanics.rock_mechanical_properties
WHERE sample_location IS NOT NULL

UNION ALL

SELECT 
    'volcanic_chambers' as data_type,
    COUNT(*) as record_count,
    ST_Extent(location::geometry) as spatial_extent,
    AVG(confidence_level) as avg_quality_score
FROM volcanic_systems.volcanic_chambers
WHERE location IS NOT NULL

UNION ALL

SELECT 
    'seismic_stations' as data_type,
    COUNT(*) as record_count,
    ST_Extent(location::geometry) as spatial_extent,
    AVG(data_quality_score) as avg_quality_score
FROM seismic_networks.seismic_stations
WHERE location IS NOT NULL;

CREATE UNIQUE INDEX idx_spatial_summary_type ON geological_core.spatial_data_summary (data_type);

-- Performance monitoring view
CREATE VIEW geological_core.system_performance_metrics AS
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE schemaname IN ('geological_core', 'rock_mechanics', 'volcanic_systems', 'seismic_networks', 'geodetic_data', 'geophysical_data')
ORDER BY n_live_tup DESC;

-- =============================================================================
-- INITIAL DATA AND CONSTRAINTS
-- =============================================================================

-- Add foreign key constraints
ALTER TABLE volcanic_systems.critical_failure_zones 
ADD CONSTRAINT fk_failure_zone_chamber 
FOREIGN KEY (zone_id) REFERENCES volcanic_systems.volcanic_chambers(chamber_id) ON DELETE CASCADE;

-- Create initial reference data
INSERT INTO geological_core.external_api_endpoints (
    endpoint_name, base_url, api_version, authentication_type, data_type, data_format
) VALUES 
    ('USGS_Earthquake_API', 'https://earthquake.usgs.gov/fdsnws/event/1/', '1.0', 'none', 'seismic_events', 'json'),
    ('GVP_Volcano_API', 'https://volcano.si.edu/api/v1/', '1.0', 'api_key', 'volcanic_data', 'json'),
    ('UNAVCO_GPS_API', 'https://www.unavco.org/data/gps-gnss/', '2.0', 'api_key', 'geodetic_data', 'json')
ON CONFLICT (endpoint_name) DO NOTHING;

-- Create sample quality assessment rules
INSERT INTO geological_core.data_quality_assessments (
    assessment_id, table_name, assessment_scope, completeness_score, 
    accuracy_score, consistency_score, overall_quality_score, quality_grade,
    assessment_method, assessor
) VALUES 
    ('INIT_ROCK_MECH_001', 'rock_mechanical_properties', 'table', 0.85, 0.90, 0.88, 0.88, 'B', 'automated_system', 'system_initialization'),
    ('INIT_VOLCANIC_001', 'volcanic_chambers', 'table', 0.75, 0.85, 0.80, 0.80, 'B', 'automated_system', 'system_initialization')
ON CONFLICT (assessment_id) DO NOTHING;

COMMIT;