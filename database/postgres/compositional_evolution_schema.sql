-- Compositional Evolution Tracking Database Schema
-- Advanced geochemical modeling and temporal evolution tracking

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- Main table for compositional evolution results
CREATE TABLE IF NOT EXISTS compositional_evolution_results (
    evolution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chamber_system_id UUID NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    evolution_data JSONB NOT NULL,
    mixing_parameters JSONB,
    crystallization_data JSONB,
    volatile_evolution JSONB,
    validation_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Magma composition snapshots for detailed tracking
CREATE TABLE IF NOT EXISTS magma_compositions (
    composition_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evolution_id UUID REFERENCES compositional_evolution_results(evolution_id) ON DELETE CASCADE,
    chamber_id VARCHAR(100) NOT NULL,
    time_step INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Major elements (wt%)
    sio2 FLOAT NOT NULL,
    tio2 FLOAT NOT NULL,
    al2o3 FLOAT NOT NULL,
    feo FLOAT NOT NULL,
    mno FLOAT NOT NULL,
    mgo FLOAT NOT NULL,
    cao FLOAT NOT NULL,
    na2o FLOAT NOT NULL,
    k2o FLOAT NOT NULL,
    p2o5 FLOAT NOT NULL,
    
    -- Volatile content (wt%)
    h2o FLOAT DEFAULT 0.0,
    co2 FLOAT DEFAULT 0.0,
    so2 FLOAT DEFAULT 0.0,
    cl FLOAT DEFAULT 0.0,
    f FLOAT DEFAULT 0.0,
    
    -- Physical conditions
    temperature FLOAT NOT NULL,
    pressure FLOAT NOT NULL,
    
    -- Trace elements and isotopes (JSONB for flexibility)
    trace_elements JSONB,
    isotopic_ratios JSONB,
    
    -- Calculated properties
    mg_number FLOAT,
    differentiation_index FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Mineral crystallization tracking
CREATE TABLE IF NOT EXISTS crystallization_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    composition_id UUID REFERENCES magma_compositions(composition_id) ON DELETE CASCADE,
    mineral_name VARCHAR(50) NOT NULL,
    nucleation_rate FLOAT,
    growth_rate FLOAT,
    crystal_fraction FLOAT,
    liquidus_temperature FLOAT,
    undercooling FLOAT,
    gibbs_energy FLOAT,
    stability_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Volatile degassing tracking
CREATE TABLE IF NOT EXISTS degassing_events (
    degassing_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    composition_id UUID REFERENCES magma_compositions(composition_id) ON DELETE CASCADE,
    volatile_species VARCHAR(20) NOT NULL,
    initial_concentration FLOAT NOT NULL,
    final_concentration FLOAT NOT NULL,
    saturation_concentration FLOAT,
    degassing_rate FLOAT,
    degassing_efficiency FLOAT,
    ascent_rate FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Magma mixing events
CREATE TABLE IF NOT EXISTS mixing_events (
    mixing_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evolution_id UUID REFERENCES compositional_evolution_results(evolution_id) ON DELETE CASCADE,
    chamber1_id VARCHAR(100) NOT NULL,
    chamber2_id VARCHAR(100) NOT NULL,
    mixing_ratio FLOAT NOT NULL CHECK (mixing_ratio >= 0 AND mixing_ratio <= 1),
    mixing_rate FLOAT,
    thermal_equilibration_temp FLOAT,
    mixing_entropy FLOAT,
    time_step INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Thermodynamic calculations cache
CREATE TABLE IF NOT EXISTS thermodynamic_calculations (
    calc_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    composition_hash VARCHAR(64) NOT NULL, -- MD5 hash of composition for caching
    temperature FLOAT NOT NULL,
    pressure FLOAT NOT NULL,
    mineral_stabilities JSONB,
    volatile_saturations JSONB,
    phase_equilibria JSONB,
    gibbs_energies JSONB,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Validation against natural compositions
CREATE TABLE IF NOT EXISTS composition_validation (
    validation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evolution_id UUID REFERENCES compositional_evolution_results(evolution_id) ON DELETE CASCADE,
    chamber_id VARCHAR(100) NOT NULL,
    reference_rock_type VARCHAR(50),
    rms_difference FLOAT,
    validity_score FLOAT CHECK (validity_score >= 0 AND validity_score <= 1),
    best_match_confidence FLOAT,
    validation_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Compositional gradients within chambers
CREATE TABLE IF NOT EXISTS compositional_gradients (
    gradient_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    composition_id UUID REFERENCES magma_compositions(composition_id) ON DELETE CASCADE,
    spatial_dimension VARCHAR(20) NOT NULL, -- 'height', 'radius', etc.
    gradient_profile JSONB NOT NULL, -- Array of position-composition pairs
    diffusion_coefficients JSONB,
    stratification_strength FLOAT,
    mixing_efficiency FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trace element partition coefficients
CREATE TABLE IF NOT EXISTS partition_coefficients (
    partition_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    element VARCHAR(10) NOT NULL,
    mineral_phase VARCHAR(50) NOT NULL,
    melt_phase VARCHAR(50) NOT NULL,
    coefficient_value FLOAT NOT NULL,
    temperature FLOAT NOT NULL,
    pressure FLOAT NOT NULL,
    composition_hash VARCHAR(64),
    temperature_dependence FLOAT DEFAULT 0.0,
    pressure_dependence FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_compositional_evolution_chamber_system ON compositional_evolution_results(chamber_system_id);
CREATE INDEX IF NOT EXISTS idx_compositional_evolution_timestamp ON compositional_evolution_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_magma_compositions_evolution ON magma_compositions(evolution_id);
CREATE INDEX IF NOT EXISTS idx_magma_compositions_chamber_time ON magma_compositions(chamber_id, time_step);
CREATE INDEX IF NOT EXISTS idx_crystallization_composition ON crystallization_events(composition_id);
CREATE INDEX IF NOT EXISTS idx_degassing_composition ON degassing_events(composition_id);
CREATE INDEX IF NOT EXISTS idx_mixing_evolution ON mixing_events(evolution_id);
CREATE INDEX IF NOT EXISTS idx_thermodynamic_hash ON thermodynamic_calculations(composition_hash);
CREATE INDEX IF NOT EXISTS idx_validation_evolution ON composition_validation(evolution_id);
CREATE INDEX IF NOT EXISTS idx_gradients_composition ON compositional_gradients(composition_id);
CREATE INDEX IF NOT EXISTS idx_partition_element_mineral ON partition_coefficients(element, mineral_phase);

-- GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_evolution_data_gin ON compositional_evolution_results USING GIN (evolution_data);
CREATE INDEX IF NOT EXISTS idx_trace_elements_gin ON magma_compositions USING GIN (trace_elements);
CREATE INDEX IF NOT EXISTS idx_isotopic_ratios_gin ON magma_compositions USING GIN (isotopic_ratios);
CREATE INDEX IF NOT EXISTS idx_mineral_stabilities_gin ON thermodynamic_calculations USING GIN (mineral_stabilities);

-- Views for common queries
CREATE OR REPLACE VIEW composition_evolution_summary AS
SELECT 
    cer.evolution_id,
    cer.chamber_system_id,
    cer.timestamp,
    COUNT(mc.composition_id) as total_compositions,
    COUNT(DISTINCT mc.chamber_id) as chamber_count,
    MAX(mc.time_step) as max_time_step,
    AVG(cv.validity_score) as avg_validity_score
FROM compositional_evolution_results cer
LEFT JOIN magma_compositions mc ON cer.evolution_id = mc.evolution_id
LEFT JOIN composition_validation cv ON cer.evolution_id = cv.evolution_id
GROUP BY cer.evolution_id, cer.chamber_system_id, cer.timestamp;

CREATE OR REPLACE VIEW chamber_composition_trends AS
SELECT 
    mc.chamber_id,
    mc.time_step,
    mc.sio2,
    mc.mgo,
    mc.cao,
    mc.na2o + mc.k2o as alkalis,
    mc.h2o + mc.co2 as total_volatiles,
    mc.temperature,
    mc.pressure,
    mc.mg_number,
    COUNT(ce.event_id) as crystallization_events,
    COUNT(de.degassing_id) as degassing_events
FROM magma_compositions mc
LEFT JOIN crystallization_events ce ON mc.composition_id = ce.composition_id
LEFT JOIN degassing_events de ON mc.composition_id = de.composition_id
GROUP BY mc.composition_id, mc.chamber_id, mc.time_step, mc.sio2, mc.mgo, 
         mc.cao, mc.na2o, mc.k2o, mc.h2o, mc.co2, mc.temperature, 
         mc.pressure, mc.mg_number;

-- Functions for data analysis
CREATE OR REPLACE FUNCTION calculate_differentiation_index(
    sio2 FLOAT, na2o FLOAT, k2o FLOAT
) RETURNS FLOAT AS $$
BEGIN
    -- Thornton & Tuttle (1960) Differentiation Index
    RETURN sio2 + na2o + k2o;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION calculate_mg_number(
    mgo FLOAT, feo FLOAT
) RETURNS FLOAT AS $$
BEGIN
    -- Mg# = Mg/(Mg + Fe) molar ratio
    IF (mgo + feo * 0.9) > 0 THEN
        RETURN mgo / (mgo + feo * 0.9);
    ELSE
        RETURN NULL;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Trigger to automatically calculate derived properties
CREATE OR REPLACE FUNCTION update_composition_properties()
RETURNS TRIGGER AS $$
BEGIN
    NEW.mg_number := calculate_mg_number(NEW.mgo, NEW.feo);
    NEW.differentiation_index := calculate_differentiation_index(NEW.sio2, NEW.na2o, NEW.k2o);
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_composition_properties
    BEFORE INSERT OR UPDATE ON magma_compositions
    FOR EACH ROW
    EXECUTE FUNCTION update_composition_properties();

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO prometheus_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO prometheus_user;