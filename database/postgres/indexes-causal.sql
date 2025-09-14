-- Comprehensive Indexes for Causal Analysis Database
-- Optimized indexes for high-performance queries and spatial operations

SET search_path TO causal_analysis, geological_data, monitoring, public;

-- ============================================================================
-- CAUSAL ANALYSIS INDEXES
-- ============================================================================

-- Causal graphs indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_graphs_created_at 
    ON causal_analysis.causal_graphs(created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_graphs_algorithms 
    ON causal_analysis.causal_graphs USING GIN(algorithms_used);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_graphs_data_sources 
    ON causal_analysis.causal_graphs USING GIN(data_sources_used);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_graphs_temporal 
    ON causal_analysis.causal_graphs(temporal_start, temporal_end);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_graphs_spatial 
    ON causal_analysis.causal_graphs USING GIST(spatial_bounds);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_graphs_confidence 
    ON causal_analysis.causal_graphs(confidence_threshold);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_graphs_size 
    ON causal_analysis.causal_graphs(total_nodes, total_edges);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_graphs_metadata 
    ON causal_analysis.causal_graphs USING GIN(metadata);

-- Causal nodes indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_graph_id 
    ON causal_analysis.causal_nodes(graph_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_identifier 
    ON causal_analysis.causal_nodes(graph_id, node_identifier);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_process_type 
    ON causal_analysis.causal_nodes(process_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_spatial_scale 
    ON causal_analysis.causal_nodes(spatial_scale);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_temporal_scale 
    ON causal_analysis.causal_nodes(temporal_scale);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_location 
    ON causal_analysis.causal_nodes USING GIST(location);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_spatial_extent 
    ON causal_analysis.causal_nodes USING GIST(spatial_extent);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_time_range 
    ON causal_analysis.causal_nodes USING GIST(time_range);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_data_source 
    ON causal_analysis.causal_nodes(data_source);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_quality 
    ON causal_analysis.causal_nodes(data_quality, confidence);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_metadata 
    ON causal_analysis.causal_nodes USING GIN(metadata);

-- Causal edges indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_graph_id 
    ON causal_analysis.causal_edges(graph_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_identifier 
    ON causal_analysis.causal_edges(graph_id, edge_identifier);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_source_target 
    ON causal_analysis.causal_edges(source_node_id, target_node_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_strength 
    ON causal_analysis.causal_edges(causal_strength DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_confidence 
    ON causal_analysis.causal_edges(confidence_lower, confidence_upper);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_temporal_lag 
    ON causal_analysis.causal_edges(temporal_lag_hours) WHERE temporal_lag_hours IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_significance 
    ON causal_analysis.causal_edges(statistical_significance DESC, p_value);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_method 
    ON causal_analysis.causal_edges(discovery_method);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_validation 
    ON causal_analysis.causal_edges(validation_status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_stability 
    ON causal_analysis.causal_edges(temporal_stability DESC) WHERE temporal_stability IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_geological_context 
    ON causal_analysis.causal_edges USING GIN(geological_context);

-- Composite indexes for common queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_strength_validation 
    ON causal_analysis.causal_edges(causal_strength DESC, validation_status) 
    WHERE validation_status IN ('validated', 'ensemble_validated', 'domain_validated');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_process_location 
    ON causal_analysis.causal_nodes(process_type, spatial_scale) 
    WHERE location IS NOT NULL;

-- Hypothesis tests indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hypothesis_tests_graph_id 
    ON causal_analysis.hypothesis_tests(graph_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hypothesis_tests_created_at 
    ON causal_analysis.hypothesis_tests(created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hypothesis_tests_p_values 
    ON causal_analysis.hypothesis_tests(bootstrap_p_value, permutation_p_value);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hypothesis_tests_methods 
    ON causal_analysis.hypothesis_tests USING GIN(test_methods);

-- Temporal snapshots indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_temporal_snapshots_graph_timestamp 
    ON causal_analysis.temporal_snapshots(graph_id, snapshot_timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_temporal_snapshots_confidence 
    ON causal_analysis.temporal_snapshots(confidence DESC) WHERE confidence IS NOT NULL;

-- ============================================================================
-- GEOLOGICAL DATA INDEXES
-- ============================================================================

-- Seismic events indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_seismic_events_origin_time 
    ON geological_data.seismic_events(origin_time DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_seismic_events_location 
    ON geological_data.seismic_events USING GIST(location);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_seismic_events_magnitude 
    ON geological_data.seismic_events(magnitude DESC) WHERE magnitude IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_seismic_events_depth 
    ON geological_data.seismic_events(depth_km);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_seismic_events_network_station 
    ON geological_data.seismic_events(network, station);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_seismic_events_event_type 
    ON geological_data.seismic_events(event_type) WHERE event_type IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_seismic_events_data_source 
    ON geological_data.seismic_events(data_source);

-- Composite spatial-temporal index for seismic events
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_seismic_events_spatiotemporal 
    ON geological_data.seismic_events USING GIST(location, origin_time);

-- MT-SAR observations indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtsar_acquisition_date 
    ON geological_data.mtsar_observations(acquisition_date DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtsar_reference_point 
    ON geological_data.mtsar_observations USING GIST(reference_point);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtsar_satellite_orbit 
    ON geological_data.mtsar_observations(satellite_platform, orbit_direction);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtsar_displacement 
    ON geological_data.mtsar_observations(los_displacement_mm) WHERE los_displacement_mm IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtsar_coherence 
    ON geological_data.mtsar_observations(coherence DESC) WHERE coherence IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mtsar_baseline 
    ON geological_data.mtsar_observations(baseline_perpendicular_m, baseline_temporal_days);

-- GPS displacements indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gps_timestamp 
    ON geological_data.gps_displacements(timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gps_station_location 
    ON geological_data.gps_displacements USING GIST(station_location);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gps_station_id 
    ON geological_data.gps_displacements(station_id, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gps_displacement_magnitude 
    ON geological_data.gps_displacements(
        sqrt(displacement_east_mm^2 + displacement_north_mm^2 + displacement_up_mm^2)
    );

-- Geochemical analyses indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_geochem_sample_date 
    ON geological_data.geochemical_analyses(sample_date DESC) WHERE sample_date IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_geochem_location 
    ON geological_data.geochemical_analyses USING GIST(sample_location);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_geochem_rock_type 
    ON geological_data.geochemical_analyses(rock_type) WHERE rock_type IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_geochem_tectonic_setting 
    ON geological_data.geochemical_analyses(tectonic_setting) WHERE tectonic_setting IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_geochem_database_source 
    ON geological_data.geochemical_analyses(database_source);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_geochem_sio2 
    ON geological_data.geochemical_analyses(sio2) WHERE sio2 IS NOT NULL;

-- Composite geochemical classification index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_geochem_classification 
    ON geological_data.geochemical_analyses(sio2, k2o + na2o) 
    WHERE sio2 IS NOT NULL AND k2o IS NOT NULL AND na2o IS NOT NULL;

-- Thermal anomalies indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thermal_anomalies_detection_time 
    ON geological_data.thermal_anomalies(detection_time DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thermal_anomalies_location 
    ON geological_data.thermal_anomalies USING GIST(location);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thermal_anomalies_brightness 
    ON geological_data.thermal_anomalies(brightness_temp_k DESC) WHERE brightness_temp_k IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thermal_anomalies_power 
    ON geological_data.thermal_anomalies(fire_radiative_power DESC) WHERE fire_radiative_power IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thermal_anomalies_satellite 
    ON geological_data.thermal_anomalies(satellite, instrument);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thermal_anomalies_confidence 
    ON geological_data.thermal_anomalies(confidence DESC) WHERE confidence IS NOT NULL;

-- Thermal measurements indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thermal_measurements_time 
    ON geological_data.thermal_measurements(measurement_time DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thermal_measurements_location 
    ON geological_data.thermal_measurements USING GIST(sensor_location);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thermal_measurements_station 
    ON geological_data.thermal_measurements(station_id, measurement_time DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thermal_measurements_temperature 
    ON geological_data.thermal_measurements(temperature_celsius DESC) WHERE temperature_celsius IS NOT NULL;

-- Gas measurements indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gas_measurements_time 
    ON geological_data.gas_measurements(measurement_time DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gas_measurements_location 
    ON geological_data.gas_measurements USING GIST(station_location);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gas_measurements_station 
    ON geological_data.gas_measurements(station_id, measurement_time DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gas_measurements_so2 
    ON geological_data.gas_measurements(so2_ppm DESC) WHERE so2_ppm IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gas_measurements_co2 
    ON geological_data.gas_measurements(co2_ppm DESC) WHERE co2_ppm IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gas_measurements_flux 
    ON geological_data.gas_measurements(so2_flux_tons_day DESC, co2_flux_tons_day DESC) 
    WHERE so2_flux_tons_day IS NOT NULL OR co2_flux_tons_day IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gas_measurements_observatory 
    ON geological_data.gas_measurements(observatory) WHERE observatory IS NOT NULL;

-- ============================================================================
-- MONITORING INDEXES
-- ============================================================================

-- Performance metrics indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_timestamp 
    ON monitoring.performance_metrics(timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_service_metric 
    ON monitoring.performance_metrics(service_name, metric_name, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_tags 
    ON monitoring.performance_metrics USING GIN(tags);

-- Discovery jobs indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_discovery_jobs_status 
    ON monitoring.discovery_jobs(status, start_time DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_discovery_jobs_graph_id 
    ON monitoring.discovery_jobs(graph_id) WHERE graph_id IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_discovery_jobs_duration 
    ON monitoring.discovery_jobs(duration_seconds DESC) WHERE duration_seconds IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_discovery_jobs_algorithms 
    ON monitoring.discovery_jobs USING GIN(algorithms);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_discovery_jobs_data_sources 
    ON monitoring.discovery_jobs USING GIN(data_sources);

-- Causal alerts indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_alerts_created_at 
    ON monitoring.causal_alerts(created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_alerts_severity 
    ON monitoring.causal_alerts(severity, created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_alerts_type 
    ON monitoring.causal_alerts(alert_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_alerts_graph_id 
    ON monitoring.causal_alerts(graph_id) WHERE graph_id IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_alerts_resolved 
    ON monitoring.causal_alerts(resolved, created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_alerts_location 
    ON monitoring.causal_alerts USING GIST(spatial_location) WHERE spatial_location IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_alerts_temporal 
    ON monitoring.causal_alerts USING GIST(temporal_context) WHERE temporal_context IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_alerts_process 
    ON monitoring.causal_alerts(geological_process) WHERE geological_process IS NOT NULL;

-- ============================================================================
-- SPECIALIZED INDEXES FOR COMPLEX QUERIES
-- ============================================================================

-- Multi-column indexes for common filter combinations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_strength_method_validation 
    ON causal_analysis.causal_edges(causal_strength DESC, discovery_method, validation_status)
    WHERE causal_strength > 0.1 AND validation_status != 'rejected';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_nodes_process_scale_quality 
    ON causal_analysis.causal_nodes(process_type, spatial_scale, temporal_scale, data_quality)
    WHERE data_quality IN ('excellent', 'good');

-- Partial indexes for active/recent data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recent_seismic_events 
    ON geological_data.seismic_events(origin_time DESC, magnitude DESC)
    WHERE origin_time > (now() - INTERVAL '30 days');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recent_mtsar_observations 
    ON geological_data.mtsar_observations(acquisition_date DESC, coherence DESC)
    WHERE acquisition_date > (now() - INTERVAL '90 days') AND coherence > 0.3;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recent_gas_measurements 
    ON geological_data.gas_measurements(measurement_time DESC, so2_flux_tons_day DESC)
    WHERE measurement_time > (now() - INTERVAL '7 days') AND so2_flux_tons_day IS NOT NULL;

-- Expression indexes for calculated values
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_edges_confidence_width 
    ON causal_analysis.causal_edges((confidence_upper - confidence_lower))
    WHERE confidence_upper IS NOT NULL AND confidence_lower IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_gps_total_displacement 
    ON geological_data.gps_displacements(
        sqrt(displacement_east_mm^2 + displacement_north_mm^2 + displacement_up_mm^2)
    ) WHERE displacement_east_mm IS NOT NULL AND displacement_north_mm IS NOT NULL AND displacement_up_mm IS NOT NULL;

-- ============================================================================
-- STATISTICS AND MAINTENANCE
-- ============================================================================

-- Update table statistics for query optimization
ANALYZE causal_analysis.causal_graphs;
ANALYZE causal_analysis.causal_nodes;
ANALYZE causal_analysis.causal_edges;
ANALYZE causal_analysis.hypothesis_tests;
ANALYZE causal_analysis.temporal_snapshots;

ANALYZE geological_data.seismic_events;
ANALYZE geological_data.mtsar_observations;
ANALYZE geological_data.gps_displacements;
ANALYZE geological_data.geochemical_analyses;
ANALYZE geological_data.thermal_anomalies;
ANALYZE geological_data.thermal_measurements;
ANALYZE geological_data.gas_measurements;

ANALYZE monitoring.performance_metrics;
ANALYZE monitoring.discovery_jobs;
ANALYZE monitoring.causal_alerts;

-- Create maintenance functions for index health monitoring
CREATE OR REPLACE FUNCTION monitoring.check_index_usage()
RETURNS TABLE(
    schemaname TEXT,
    tablename TEXT,
    indexname TEXT,
    idx_scan BIGINT,
    idx_tup_read BIGINT,
    idx_tup_fetch BIGINT,
    usage_ratio DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.schemaname::TEXT,
        s.tablename::TEXT,
        s.indexrelname::TEXT,
        s.idx_scan,
        s.idx_tup_read,
        s.idx_tup_fetch,
        CASE 
            WHEN s.idx_scan = 0 THEN 0.00
            ELSE ROUND((s.idx_tup_fetch::DECIMAL / s.idx_scan), 2)
        END as usage_ratio
    FROM pg_stat_user_indexes s
    JOIN pg_index i ON s.indexrelid = i.indexrelid
    WHERE s.schemaname IN ('causal_analysis', 'geological_data', 'monitoring')
    ORDER BY s.idx_scan DESC;
END;
$$ LANGUAGE plpgsql;

-- Create function to identify unused indexes
CREATE OR REPLACE FUNCTION monitoring.find_unused_indexes()
RETURNS TABLE(
    schemaname TEXT,
    tablename TEXT,
    indexname TEXT,
    index_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.schemaname::TEXT,
        s.tablename::TEXT,
        s.indexrelname::TEXT,
        pg_size_pretty(pg_relation_size(s.indexrelid))::TEXT as index_size
    FROM pg_stat_user_indexes s
    JOIN pg_index i ON s.indexrelid = i.indexrelid
    WHERE s.idx_scan = 0
    AND s.schemaname IN ('causal_analysis', 'geological_data', 'monitoring')
    AND NOT i.indisunique
    ORDER BY pg_relation_size(s.indexrelid) DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION monitoring.check_index_usage() IS 'Monitor index usage statistics for performance optimization';
COMMENT ON FUNCTION monitoring.find_unused_indexes() IS 'Identify unused indexes that may be candidates for removal';