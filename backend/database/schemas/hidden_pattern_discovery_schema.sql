-- Hidden Pattern Discovery Framework Database Schema
-- Advanced schema for storing discovered patterns, causal relationships, and validation results

-- Main discoveries table
CREATE TABLE IF NOT EXISTS hidden_pattern_discoveries (
    discovery_id VARCHAR(255) PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    data_source_id VARCHAR(255) NOT NULL,
    configuration JSONB NOT NULL,
    summary_metrics JSONB,
    processing_time_seconds FLOAT,
    algorithms_used TEXT[],
    total_patterns_found INTEGER DEFAULT 0,
    significant_patterns_found INTEGER DEFAULT 0,
    validation_success_rate FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Individual discovered patterns
CREATE TABLE IF NOT EXISTS discovered_patterns (
    pattern_id VARCHAR(255) PRIMARY KEY,
    discovery_id VARCHAR(255) REFERENCES hidden_pattern_discoveries(discovery_id) ON DELETE CASCADE,
    pattern_type VARCHAR(100) NOT NULL,
    dimensions_involved INTEGER[],
    pattern_strength FLOAT NOT NULL,
    statistical_significance FLOAT NOT NULL,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    p_value FLOAT NOT NULL,
    effect_size FLOAT,
    physical_interpretation TEXT,
    discovery_metadata JSONB,
    validation_results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Causal relationships discovered
CREATE TABLE IF NOT EXISTS causal_relationships (
    relationship_id SERIAL PRIMARY KEY,
    discovery_id VARCHAR(255) REFERENCES hidden_pattern_discoveries(discovery_id) ON DELETE CASCADE,
    cause_variable VARCHAR(255) NOT NULL,
    effect_variable VARCHAR(255) NOT NULL,
    causal_strength FLOAT NOT NULL,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    p_value FLOAT NOT NULL,
    causal_mechanism VARCHAR(100),
    temporal_lag FLOAT,
    confounding_variables TEXT[],
    validation_score FLOAT,
    algorithm_used VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(discovery_id, cause_variable, effect_variable, algorithm_used)
);

-- Network patterns and structures
CREATE TABLE IF NOT EXISTS network_patterns (
    network_id SERIAL PRIMARY KEY,
    discovery_id VARCHAR(255) REFERENCES hidden_pattern_discoveries(discovery_id) ON DELETE CASCADE,
    network_type VARCHAR(100) NOT NULL,
    nodes TEXT[] NOT NULL,
    edges JSONB NOT NULL, -- Store as JSON array of edge objects
    centrality_measures JSONB,
    community_structure JSONB,
    motif_counts JSONB,
    network_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pattern validation results
CREATE TABLE IF NOT EXISTS pattern_validations (
    validation_id VARCHAR(255) PRIMARY KEY,
    pattern_id VARCHAR(255) REFERENCES discovered_patterns(pattern_id) ON DELETE CASCADE,
    validation_dataset VARCHAR(255) NOT NULL,
    replication_success BOOLEAN NOT NULL,
    replication_strength FLOAT,
    statistical_tests JSONB,
    cross_validation_scores FLOAT[],
    bootstrap_confidence_lower FLOAT,
    bootstrap_confidence_upper FLOAT,
    robustness_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Anomaly detection results
CREATE TABLE IF NOT EXISTS anomaly_detections (
    anomaly_id SERIAL PRIMARY KEY,
    discovery_id VARCHAR(255) REFERENCES hidden_pattern_discoveries(discovery_id) ON DELETE CASCADE,
    algorithm_used VARCHAR(100) NOT NULL,
    anomaly_indices INTEGER[],
    anomaly_scores FLOAT[],
    contamination_rate FLOAT,
    feature_importance FLOAT[],
    statistical_significance FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Machine learning pattern results
CREATE TABLE IF NOT EXISTS ml_pattern_results (
    ml_result_id SERIAL PRIMARY KEY,
    discovery_id VARCHAR(255) REFERENCES hidden_pattern_discoveries(discovery_id) ON DELETE CASCADE,
    algorithm_name VARCHAR(100) NOT NULL,
    hyperparameters JSONB,
    performance_metrics JSONB,
    feature_importance JSONB,
    model_artifacts_path TEXT,
    training_time_seconds FLOAT,
    validation_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Ensemble results combining multiple algorithms
CREATE TABLE IF NOT EXISTS ensemble_results (
    ensemble_id SERIAL PRIMARY KEY,
    discovery_id VARCHAR(255) REFERENCES hidden_pattern_discoveries(discovery_id) ON DELETE CASCADE,
    algorithms_combined TEXT[] NOT NULL,
    ensemble_method VARCHAR(100) NOT NULL,
    consensus_patterns JSONB,
    agreement_scores JSONB,
    ensemble_performance JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Physical interpretations of discovered patterns
CREATE TABLE IF NOT EXISTS physical_interpretations (
    interpretation_id SERIAL PRIMARY KEY,
    discovery_id VARCHAR(255) REFERENCES hidden_pattern_discoveries(discovery_id) ON DELETE CASCADE,
    pattern_id VARCHAR(255) REFERENCES discovered_patterns(pattern_id) ON DELETE CASCADE,
    volcanic_process_type VARCHAR(255),
    interpretation_text TEXT NOT NULL,
    confidence_score FLOAT,
    supporting_evidence JSONB,
    predictive_insights JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cross-dataset validation results
CREATE TABLE IF NOT EXISTS cross_dataset_validations (
    cross_validation_id SERIAL PRIMARY KEY,
    discovery_id VARCHAR(255) REFERENCES hidden_pattern_discoveries(discovery_id) ON DELETE CASCADE,
    source_dataset VARCHAR(255) NOT NULL,
    target_dataset VARCHAR(255) NOT NULL,
    patterns_tested INTEGER,
    patterns_replicated INTEGER,
    replication_rate FLOAT,
    generalization_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Statistical significance tests
CREATE TABLE IF NOT EXISTS significance_tests (
    test_id SERIAL PRIMARY KEY,
    discovery_id VARCHAR(255) REFERENCES hidden_pattern_discoveries(discovery_id) ON DELETE CASCADE,
    pattern_id VARCHAR(255) REFERENCES discovered_patterns(pattern_id) ON DELETE CASCADE,
    test_name VARCHAR(100) NOT NULL,
    test_statistic FLOAT,
    p_value FLOAT NOT NULL,
    degrees_of_freedom INTEGER,
    effect_size FLOAT,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    multiple_testing_corrected BOOLEAN DEFAULT FALSE,
    corrected_p_value FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Temporal analysis results for time-series patterns
CREATE TABLE IF NOT EXISTS temporal_pattern_analysis (
    temporal_id SERIAL PRIMARY KEY,
    discovery_id VARCHAR(255) REFERENCES hidden_pattern_discoveries(discovery_id) ON DELETE CASCADE,
    time_lag INTEGER NOT NULL,
    temporal_correlation FLOAT,
    granger_causality_p_value FLOAT,
    transfer_entropy FLOAT,
    temporal_significance FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_discoveries_timestamp ON hidden_pattern_discoveries(timestamp);
CREATE INDEX IF NOT EXISTS idx_discoveries_data_source ON hidden_pattern_discoveries(data_source_id);
CREATE INDEX IF NOT EXISTS idx_patterns_discovery_id ON discovered_patterns(discovery_id);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON discovered_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_significance ON discovered_patterns(statistical_significance);
CREATE INDEX IF NOT EXISTS idx_causal_discovery_id ON causal_relationships(discovery_id);
CREATE INDEX IF NOT EXISTS idx_causal_strength ON causal_relationships(causal_strength);
CREATE INDEX IF NOT EXISTS idx_network_discovery_id ON network_patterns(discovery_id);
CREATE INDEX IF NOT EXISTS idx_validations_pattern_id ON pattern_validations(pattern_id);
CREATE INDEX IF NOT EXISTS idx_validations_success ON pattern_validations(replication_success);
CREATE INDEX IF NOT EXISTS idx_anomalies_discovery_id ON anomaly_detections(discovery_id);
CREATE INDEX IF NOT EXISTS idx_ml_results_discovery_id ON ml_pattern_results(discovery_id);
CREATE INDEX IF NOT EXISTS idx_ensemble_discovery_id ON ensemble_results(discovery_id);
CREATE INDEX IF NOT EXISTS idx_interpretations_discovery_id ON physical_interpretations(discovery_id);
CREATE INDEX IF NOT EXISTS idx_cross_validation_discovery_id ON cross_dataset_validations(discovery_id);
CREATE INDEX IF NOT EXISTS idx_significance_discovery_id ON significance_tests(discovery_id);
CREATE INDEX IF NOT EXISTS idx_temporal_discovery_id ON temporal_pattern_analysis(discovery_id);

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_patterns_discovery_significance ON discovered_patterns(discovery_id, statistical_significance);
CREATE INDEX IF NOT EXISTS idx_causal_discovery_strength ON causal_relationships(discovery_id, causal_strength);
CREATE INDEX IF NOT EXISTS idx_validations_pattern_success ON pattern_validations(pattern_id, replication_success);

-- Add comments for documentation
COMMENT ON TABLE hidden_pattern_discoveries IS 'Main table storing discovery sessions and their metadata';
COMMENT ON TABLE discovered_patterns IS 'Individual patterns discovered by machine learning algorithms';
COMMENT ON TABLE causal_relationships IS 'Causal relationships discovered between volcanic variables';
COMMENT ON TABLE network_patterns IS 'Network-based patterns and structures in volcanic data';
COMMENT ON TABLE pattern_validations IS 'Validation results for discovered patterns against independent datasets';
COMMENT ON TABLE anomaly_detections IS 'Anomalous patterns detected in volcanic data';
COMMENT ON TABLE ml_pattern_results IS 'Results from machine learning pattern discovery algorithms';
COMMENT ON TABLE ensemble_results IS 'Combined results from multiple discovery algorithms';
COMMENT ON TABLE physical_interpretations IS 'Physical interpretations and meanings of discovered patterns';
COMMENT ON TABLE cross_dataset_validations IS 'Cross-dataset validation results for pattern generalization';
COMMENT ON TABLE significance_tests IS 'Statistical significance tests for discovered patterns';
COMMENT ON TABLE temporal_pattern_analysis IS 'Temporal analysis results for time-series patterns';