-- Causal Analysis Database Schema
-- Comprehensive schema for storing causal graphs, geological data, and analysis results

SET search_path TO causal_analysis, geological_data, monitoring, public;

-- ============================================================================
-- CAUSAL ANALYSIS SCHEMA
-- ============================================================================

-- Causal graphs metadata
CREATE TABLE causal_analysis.causal_graphs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    algorithms_used causal_discovery_method[] NOT NULL,
    ensemble_method VARCHAR(100),
    confidence_threshold DECIMAL(5,4) NOT NULL CHECK (confidence_threshold > 0 AND confidence_threshold < 1),
    total_nodes INTEGER NOT NULL CHECK (total_nodes >= 0),
    total_edges INTEGER NOT NULL CHECK (total_edges >= 0),
    graph_density DECIMAL(5,4) CHECK (graph_density >= 0 AND graph_density <= 1),
    data_sources_used TEXT[] NOT NULL,
    processing_time_seconds DECIMAL(10,3),
    spatial_bounds GEOMETRY(POLYGON, 4326),
    temporal_start TIMESTAMP WITH TIME ZONE,
    temporal_end TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    created_by VARCHAR(100) NOT NULL DEFAULT current_user,
    
    CONSTRAINT valid_time_range CHECK (temporal_end > temporal_start)
);

-- Causal nodes (variables in the causal graph)
CREATE TABLE causal_analysis.causal_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    graph_id UUID NOT NULL REFERENCES causal_analysis.causal_graphs(id) ON DELETE CASCADE,
    node_identifier VARCHAR(255) NOT NULL, -- Unique within graph
    label VARCHAR(255) NOT NULL,
    process_type geological_process_type NOT NULL,
    spatial_scale spatial_scale NOT NULL,
    temporal_scale temporal_scale NOT NULL,
    data_source VARCHAR(255) NOT NULL,
    location GEOMETRY(POINT, 4326),
    spatial_extent GEOMETRY(POLYGON, 4326),
    time_range TSTZRANGE,
    sampling_rate_hz DECIMAL(10,3),
    uncertainty DECIMAL(5,4) CHECK (uncertainty >= 0 AND uncertainty <= 1),
    confidence DECIMAL(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    data_quality data_quality_level DEFAULT 'unknown',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    
    UNIQUE(graph_id, node_identifier)
);

-- Causal edges (relationships between nodes)
CREATE TABLE causal_analysis.causal_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    graph_id UUID NOT NULL REFERENCES causal_analysis.causal_graphs(id) ON DELETE CASCADE,
    edge_identifier VARCHAR(255) NOT NULL, -- Unique within graph
    source_node_id UUID NOT NULL REFERENCES causal_analysis.causal_nodes(id) ON DELETE CASCADE,
    target_node_id UUID NOT NULL REFERENCES causal_analysis.causal_nodes(id) ON DELETE CASCADE,
    causal_strength DECIMAL(5,4) NOT NULL CHECK (causal_strength >= 0 AND causal_strength <= 1),
    confidence_lower DECIMAL(5,4) NOT NULL CHECK (confidence_lower >= 0 AND confidence_lower <= 1),
    confidence_upper DECIMAL(5,4) NOT NULL CHECK (confidence_upper >= 0 AND confidence_upper <= 1),
    temporal_lag_hours DECIMAL(10,3),
    lag_uncertainty_hours DECIMAL(10,3),
    statistical_significance DECIMAL(5,4) CHECK (statistical_significance >= 0 AND statistical_significance <= 1),
    p_value DECIMAL(10,8) CHECK (p_value >= 0 AND p_value <= 1),
    effect_size DECIMAL(5,4),
    discovery_method causal_discovery_method NOT NULL,
    validation_status validation_status NOT NULL DEFAULT 'pending',
    temporal_stability DECIMAL(5,4) CHECK (temporal_stability >= 0 AND temporal_stability <= 1),
    geological_context JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    
    UNIQUE(graph_id, edge_identifier),
    CONSTRAINT valid_confidence_interval CHECK (confidence_lower <= confidence_upper),
    CONSTRAINT no_self_loops CHECK (source_node_id != target_node_id)
);

-- Hypothesis tests and validation results
CREATE TABLE causal_analysis.hypothesis_tests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    graph_id UUID NOT NULL REFERENCES causal_analysis.causal_graphs(id) ON DELETE CASCADE,
    hypothesis_name VARCHAR(255) NOT NULL,
    hypothesis_description TEXT,
    test_methods TEXT[] NOT NULL,
    significance_level DECIMAL(5,4) NOT NULL CHECK (significance_level > 0 AND significance_level < 1),
    bootstrap_p_value DECIMAL(10,8) CHECK (bootstrap_p_value >= 0 AND bootstrap_p_value <= 1),
    permutation_p_value DECIMAL(10,8) CHECK (permutation_p_value >= 0 AND permutation_p_value <= 1),
    confidence_interval_lower DECIMAL(5,4),
    confidence_interval_upper DECIMAL(5,4),
    effect_size DECIMAL(5,4),
    conclusion TEXT NOT NULL,
    recommendations TEXT[],
    test_data JSONB DEFAULT '{}',
    results JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    created_by VARCHAR(100) NOT NULL DEFAULT current_user
);

-- Temporal evolution of causal relationships
CREATE TABLE causal_analysis.temporal_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    graph_id UUID NOT NULL REFERENCES causal_analysis.causal_graphs(id) ON DELETE CASCADE,
    snapshot_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    snapshot_data JSONB NOT NULL,
    changes_from_previous JSONB DEFAULT '{}',
    confidence DECIMAL(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- ============================================================================
-- GEOLOGICAL DATA SCHEMA
-- ============================================================================

-- Seismic events and waveform data
CREATE TABLE geological_data.seismic_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(100) UNIQUE NOT NULL,
    origin_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    depth_km DECIMAL(8,3) NOT NULL,
    magnitude DECIMAL(4,2),
    magnitude_type VARCHAR(10),
    event_type VARCHAR(50),
    network VARCHAR(10) NOT NULL,
    station VARCHAR(10),
    data_source VARCHAR(100) NOT NULL,
    quality_flags TEXT[],
    uncertainty_km DECIMAL(8,3),
    processing_info JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    
    CONSTRAINT valid_depth CHECK (depth_km >= 0),
    CONSTRAINT valid_magnitude CHECK (magnitude >= -2 AND magnitude <= 10)
);

-- MT-SAR deformation observations
CREATE TABLE geological_data.mtsar_observations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    observation_id VARCHAR(100) UNIQUE NOT NULL,
    acquisition_date TIMESTAMP WITH TIME ZONE NOT NULL,
    satellite_platform VARCHAR(50) NOT NULL,
    orbit_direction VARCHAR(20) NOT NULL CHECK (orbit_direction IN ('ascending', 'descending')),
    reference_point GEOMETRY(POINT, 4326) NOT NULL,
    los_displacement_mm DECIMAL(10,3) NOT NULL,
    coherence DECIMAL(4,3) CHECK (coherence >= 0 AND coherence <= 1),
    uncertainty_mm DECIMAL(8,3),
    baseline_perpendicular_m DECIMAL(10,3),
    baseline_temporal_days INTEGER,
    atmospheric_correction BOOLEAN DEFAULT false,
    processing_parameters JSONB DEFAULT '{}',
    data_source VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- GPS displacement measurements
CREATE TABLE geological_data.gps_displacements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    measurement_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    station_id VARCHAR(50) NOT NULL,
    station_location GEOMETRY(POINT, 4326) NOT NULL,
    displacement_east_mm DECIMAL(10,3) NOT NULL,
    displacement_north_mm DECIMAL(10,3) NOT NULL,
    displacement_up_mm DECIMAL(10,3) NOT NULL,
    uncertainty_east_mm DECIMAL(8,3),
    uncertainty_north_mm DECIMAL(8,3),
    uncertainty_up_mm DECIMAL(8,3),
    processing_mode VARCHAR(50),
    data_source VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Geochemical analyses from EarthChem/GEOROC
CREATE TABLE geological_data.geochemical_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sample_id VARCHAR(100) UNIQUE NOT NULL,
    sample_date TIMESTAMP WITH TIME ZONE,
    sample_location GEOMETRY(POINT, 4326) NOT NULL,
    sample_type VARCHAR(100),
    rock_type VARCHAR(100),
    tectonic_setting VARCHAR(100),
    
    -- Major elements (wt%)
    sio2 DECIMAL(6,3) CHECK (sio2 >= 0 AND sio2 <= 100),
    tio2 DECIMAL(6,3) CHECK (tio2 >= 0 AND tio2 <= 100),
    al2o3 DECIMAL(6,3) CHECK (al2o3 >= 0 AND al2o3 <= 100),
    fe2o3 DECIMAL(6,3) CHECK (fe2o3 >= 0 AND fe2o3 <= 100),
    feo DECIMAL(6,3) CHECK (feo >= 0 AND feo <= 100),
    mno DECIMAL(6,3) CHECK (mno >= 0 AND mno <= 100),
    mgo DECIMAL(6,3) CHECK (mgo >= 0 AND mgo <= 100),
    cao DECIMAL(6,3) CHECK (cao >= 0 AND cao <= 100),
    na2o DECIMAL(6,3) CHECK (na2o >= 0 AND na2o <= 100),
    k2o DECIMAL(6,3) CHECK (k2o >= 0 AND k2o <= 100),
    p2o5 DECIMAL(6,3) CHECK (p2o5 >= 0 AND p2o5 <= 100),
    
    -- Trace elements (ppm)
    rb DECIMAL(10,3) CHECK (rb >= 0),
    sr DECIMAL(10,3) CHECK (sr >= 0),
    y DECIMAL(10,3) CHECK (y >= 0),
    zr DECIMAL(10,3) CHECK (zr >= 0),
    nb DECIMAL(10,3) CHECK (nb >= 0),
    ba DECIMAL(10,3) CHECK (ba >= 0),
    
    -- REE elements (ppm)
    la DECIMAL(10,3) CHECK (la >= 0),
    ce DECIMAL(10,3) CHECK (ce >= 0),
    pr DECIMAL(10,3) CHECK (pr >= 0),
    nd DECIMAL(10,3) CHECK (nd >= 0),
    sm DECIMAL(10,3) CHECK (sm >= 0),
    eu DECIMAL(10,3) CHECK (eu >= 0),
    gd DECIMAL(10,3) CHECK (gd >= 0),
    tb DECIMAL(10,3) CHECK (tb >= 0),
    dy DECIMAL(10,3) CHECK (dy >= 0),
    ho DECIMAL(10,3) CHECK (ho >= 0),
    er DECIMAL(10,3) CHECK (er >= 0),
    tm DECIMAL(10,3) CHECK (tm >= 0),
    yb DECIMAL(10,3) CHECK (yb >= 0),
    lu DECIMAL(10,3) CHECK (lu >= 0),
    
    analytical_method VARCHAR(100),
    laboratory VARCHAR(100),
    database_source VARCHAR(100) NOT NULL,
    quality_flags TEXT[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Thermal anomalies from NASA FIRMS and ground sensors
CREATE TABLE geological_data.thermal_anomalies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id VARCHAR(100) UNIQUE NOT NULL,
    detection_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    brightness_temp_k DECIMAL(7,2) CHECK (brightness_temp_k > 0),
    fire_radiative_power DECIMAL(10,3) CHECK (fire_radiative_power >= 0),
    confidence INTEGER CHECK (confidence >= 0 AND confidence <= 100),
    satellite VARCHAR(50),
    instrument VARCHAR(50),
    data_source VARCHAR(100) NOT NULL,
    processing_info JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Ground-based thermal measurements
CREATE TABLE geological_data.thermal_measurements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    measurement_id VARCHAR(100) UNIQUE NOT NULL,
    measurement_time TIMESTAMP WITH TIME ZONE NOT NULL,
    sensor_location GEOMETRY(POINT, 4326) NOT NULL,
    temperature_celsius DECIMAL(7,2),
    heat_flux_w_m2 DECIMAL(10,3),
    sensor_type VARCHAR(100),
    station_id VARCHAR(50),
    data_source VARCHAR(100) NOT NULL,
    quality_flags TEXT[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Gas emission measurements
CREATE TABLE geological_data.gas_measurements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    measurement_id VARCHAR(100) UNIQUE NOT NULL,
    measurement_time TIMESTAMP WITH TIME ZONE NOT NULL,
    station_location GEOMETRY(POINT, 4326) NOT NULL,
    
    -- Gas concentrations (ppm)
    so2_ppm DECIMAL(10,3) CHECK (so2_ppm >= 0),
    co2_ppm DECIMAL(10,3) CHECK (co2_ppm >= 0),
    h2s_ppm DECIMAL(10,3) CHECK (h2s_ppm >= 0),
    co_ppm DECIMAL(10,3) CHECK (co_ppm >= 0),
    h2_ppm DECIMAL(10,3) CHECK (h2_ppm >= 0),
    
    -- Gas fluxes (tons/day)
    so2_flux_tons_day DECIMAL(10,3) CHECK (so2_flux_tons_day >= 0),
    co2_flux_tons_day DECIMAL(10,3) CHECK (co2_flux_tons_day >= 0),
    
    -- Environmental conditions
    wind_speed_ms DECIMAL(6,2) CHECK (wind_speed_ms >= 0),
    wind_direction_deg DECIMAL(5,2) CHECK (wind_direction_deg >= 0 AND wind_direction_deg < 360),
    temperature_celsius DECIMAL(6,2),
    pressure_hpa DECIMAL(7,2) CHECK (pressure_hpa > 0),
    humidity_percent DECIMAL(5,2) CHECK (humidity_percent >= 0 AND humidity_percent <= 100),
    
    measurement_method VARCHAR(100),
    station_id VARCHAR(50),
    observatory VARCHAR(100),
    data_source VARCHAR(100) NOT NULL,
    quality_flags TEXT[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- ============================================================================
-- MONITORING SCHEMA
-- ============================================================================

-- System performance metrics
CREATE TABLE monitoring.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    service_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(50),
    tags JSONB DEFAULT '{}',
    
    UNIQUE(timestamp, service_name, metric_name)
);

-- Causal discovery job tracking
CREATE TABLE monitoring.discovery_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(100) UNIQUE NOT NULL,
    graph_id UUID REFERENCES causal_analysis.causal_graphs(id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    data_sources TEXT[] NOT NULL,
    algorithms causal_discovery_method[] NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    end_time TIMESTAMP WITH TIME ZONE,
    duration_seconds DECIMAL(10,3),
    nodes_discovered INTEGER,
    edges_discovered INTEGER,
    error_message TEXT,
    progress_percent INTEGER DEFAULT 0 CHECK (progress_percent >= 0 AND progress_percent <= 100),
    metadata JSONB DEFAULT '{}',
    created_by VARCHAR(100) NOT NULL DEFAULT current_user
);

-- Real-time alerts and notifications
CREATE TABLE monitoring.causal_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id VARCHAR(100) UNIQUE NOT NULL,
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    graph_id UUID REFERENCES causal_analysis.causal_graphs(id),
    affected_nodes UUID[],
    affected_edges UUID[],
    spatial_location GEOMETRY(POINT, 4326),
    temporal_context TSTZRANGE,
    geological_process geological_process_type,
    resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    
    CONSTRAINT valid_severity CHECK (severity IN ('info', 'warning', 'error', 'critical'))
);

-- Add audit triggers to all main tables
CREATE TRIGGER audit_causal_graphs 
    AFTER INSERT OR UPDATE OR DELETE ON causal_analysis.causal_graphs
    FOR EACH ROW EXECUTE FUNCTION audit.audit_trigger_function();

CREATE TRIGGER audit_causal_nodes 
    AFTER INSERT OR UPDATE OR DELETE ON causal_analysis.causal_nodes
    FOR EACH ROW EXECUTE FUNCTION audit.audit_trigger_function();

CREATE TRIGGER audit_causal_edges 
    AFTER INSERT OR UPDATE OR DELETE ON causal_analysis.causal_edges
    FOR EACH ROW EXECUTE FUNCTION audit.audit_trigger_function();

-- Add update timestamp triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_causal_graphs_updated_at 
    BEFORE UPDATE ON causal_analysis.causal_graphs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_causal_nodes_updated_at 
    BEFORE UPDATE ON causal_analysis.causal_nodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_causal_edges_updated_at 
    BEFORE UPDATE ON causal_analysis.causal_edges
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();