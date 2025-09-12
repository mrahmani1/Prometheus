-- Advanced Threat Detection and Response Database Schema
-- Initialize all required tables for SIEM, SOAR, Vulnerability Management, and Penetration Testing

-- SIEM Engine Tables
CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    source_ip INET NOT NULL,
    user_id VARCHAR(255),
    resource VARCHAR(500) NOT NULL,
    action VARCHAR(255) NOT NULL,
    outcome VARCHAR(50) NOT NULL,
    severity INTEGER NOT NULL,
    raw_data JSONB,
    geological_context JSONB,
    threat_indicators TEXT[],
    correlation_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS security_incidents (
    id UUID PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    threat_level VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    assigned_to VARCHAR(255),
    geological_impact JSONB,
    response_actions JSONB,
    timeline JSONB
);

CREATE TABLE IF NOT EXISTS threat_indicators (
    id UUID PRIMARY KEY,
    type VARCHAR(100) NOT NULL,
    value VARCHAR(500) NOT NULL,
    confidence FLOAT NOT NULL,
    source VARCHAR(100) NOT NULL,
    first_seen TIMESTAMP WITH TIME ZONE NOT NULL,
    last_seen TIMESTAMP WITH TIME ZONE NOT NULL,
    geological_relevance BOOLEAN DEFAULT FALSE,
    context JSONB
);

-- SOAR Engine Tables
CREATE TABLE IF NOT EXISTS incident_responses (
    id UUID PRIMARY KEY,
    incident_id UUID NOT NULL,
    playbook_id VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    approval_required BOOLEAN DEFAULT TRUE,
    approved_by VARCHAR(255),
    approved_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS response_tasks (
    id VARCHAR(255) PRIMARY KEY,
    response_id UUID NOT NULL REFERENCES incident_responses(id),
    action VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    result JSONB,
    dependencies TEXT[],
    timeout_seconds INTEGER DEFAULT 300
);

CREATE TABLE IF NOT EXISTS playbook_executions (
    id UUID PRIMARY KEY,
    playbook_id VARCHAR(255) NOT NULL,
    incident_id UUID NOT NULL,
    execution_time_ms INTEGER,
    success_rate FLOAT,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Vulnerability Management Tables
CREATE TABLE IF NOT EXISTS scan_targets (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    target_type VARCHAR(50) NOT NULL,
    target_value VARCHAR(500) NOT NULL,
    scan_types TEXT[] NOT NULL,
    geological_service BOOLEAN DEFAULT FALSE,
    scan_frequency VARCHAR(20) DEFAULT 'weekly',
    last_scan TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vulnerability_scans (
    id UUID PRIMARY KEY,
    target_id UUID NOT NULL REFERENCES scan_targets(id),
    scan_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    vulnerabilities_found INTEGER DEFAULT 0,
    scanner_version VARCHAR(50),
    scan_config JSONB,
    error_message TEXT,
    results_path VARCHAR(500)
);

CREATE TABLE IF NOT EXISTS vulnerabilities (
    id UUID PRIMARY KEY,
    cve_id VARCHAR(50),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    severity VARCHAR(20) NOT NULL,
    cvss_score FLOAT NOT NULL,
    affected_component VARCHAR(500) NOT NULL,
    affected_version VARCHAR(100),
    discovery_date TIMESTAMP WITH TIME ZONE NOT NULL,
    last_seen TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(20) NOT NULL,
    remediation_advice TEXT,
    geological_context JSONB,
    exploit_available BOOLEAN DEFAULT FALSE,
    patch_available BOOLEAN DEFAULT FALSE,
    references TEXT[],
    tags TEXT[]
);

-- Penetration Testing Tables
CREATE TABLE IF NOT EXISTS pentest_targets (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    target_type VARCHAR(50) NOT NULL,
    target_scope TEXT[] NOT NULL,
    geological_context BOOLEAN DEFAULT FALSE,
    authorized_by VARCHAR(255) NOT NULL,
    authorization_date TIMESTAMP WITH TIME ZONE,
    restrictions TEXT[],
    contact_info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pentest_sessions (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    target_ids UUID[] NOT NULL,
    phases TEXT[] NOT NULL,
    status VARCHAR(20) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    tester_id VARCHAR(255) NOT NULL,
    authorization_ref VARCHAR(255) NOT NULL,
    geological_focus BOOLEAN DEFAULT FALSE,
    report_path VARCHAR(500)
);

CREATE TABLE IF NOT EXISTS exploit_attempts (
    id UUID PRIMARY KEY,
    target_id UUID NOT NULL,
    exploit_name VARCHAR(255) NOT NULL,
    attack_vector VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    attempted_at TIMESTAMP WITH TIME ZONE NOT NULL,
    success BOOLEAN NOT NULL,
    evidence JSONB,
    geological_impact JSONB,
    remediation_priority INTEGER DEFAULT 1
);

-- Create Indexes for Performance
-- SIEM Indexes
CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_events_user_id ON security_events(user_id);
CREATE INDEX IF NOT EXISTS idx_security_events_source_ip ON security_events(source_ip);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security_events(severity);
CREATE INDEX IF NOT EXISTS idx_security_events_event_type ON security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_incidents_threat_level ON security_incidents(threat_level);
CREATE INDEX IF NOT EXISTS idx_security_incidents_status ON security_incidents(status);
CREATE INDEX IF NOT EXISTS idx_threat_indicators_type ON threat_indicators(type);
CREATE INDEX IF NOT EXISTS idx_threat_indicators_value ON threat_indicators(value);

-- SOAR Indexes
CREATE INDEX IF NOT EXISTS idx_incident_responses_incident_id ON incident_responses(incident_id);
CREATE INDEX IF NOT EXISTS idx_incident_responses_status ON incident_responses(status);
CREATE INDEX IF NOT EXISTS idx_response_tasks_response_id ON response_tasks(response_id);
CREATE INDEX IF NOT EXISTS idx_response_tasks_status ON response_tasks(status);

-- Vulnerability Management Indexes
CREATE INDEX IF NOT EXISTS idx_vulnerability_scans_target_id ON vulnerability_scans(target_id);
CREATE INDEX IF NOT EXISTS idx_vulnerability_scans_status ON vulnerability_scans(status);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_severity ON vulnerabilities(severity);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_status ON vulnerabilities(status);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_cve_id ON vulnerabilities(cve_id);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_affected_component ON vulnerabilities(affected_component);

-- Penetration Testing Indexes
CREATE INDEX IF NOT EXISTS idx_pentest_sessions_status ON pentest_sessions(status);
CREATE INDEX IF NOT EXISTS idx_pentest_sessions_tester_id ON pentest_sessions(tester_id);
CREATE INDEX IF NOT EXISTS idx_exploit_attempts_target_id ON exploit_attempts(target_id);
CREATE INDEX IF NOT EXISTS idx_exploit_attempts_severity ON exploit_attempts(severity);
CREATE INDEX IF NOT EXISTS idx_exploit_attempts_success ON exploit_attempts(success);

-- Insert Sample Data for Testing
INSERT INTO scan_targets (id, name, target_type, target_value, scan_types, geological_service) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'Geological API Server', 'domain', 'api.geological-platform.com', ARRAY['web_application', 'network_scan'], true),
('550e8400-e29b-41d4-a716-446655440002', 'Seismic Data Service', 'ip', '192.168.1.100', ARRAY['network_scan', 'container_scan'], true),
('550e8400-e29b-41d4-a716-446655440003', 'Volcanic Monitoring System', 'domain', 'volcano.monitoring.com', ARRAY['web_application'], true);

INSERT INTO pentest_targets (id, name, target_type, target_scope, geological_context, authorized_by) VALUES
('660e8400-e29b-41d4-a716-446655440001', 'Geological Platform Infrastructure', 'network', ARRAY['192.168.1.0/24', 'geological-platform.com'], true, 'security@geological-platform.com'),
('660e8400-e29b-41d4-a716-446655440002', 'Seismic Analysis API', 'web_app', ARRAY['api.seismic.com', 'seismic.geological-platform.com'], true, 'security@geological-platform.com');

-- Create Views for Dashboard Queries
CREATE OR REPLACE VIEW security_dashboard_summary AS
SELECT 
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '24 hours') as incidents_24h,
    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days') as incidents_7d,
    COUNT(*) FILTER (WHERE status = 'open') as open_incidents,
    COUNT(*) FILTER (WHERE threat_level = 'critical') as critical_incidents,
    COUNT(*) FILTER (WHERE geological_impact IS NOT NULL) as geological_incidents
FROM security_incidents;

CREATE OR REPLACE VIEW vulnerability_dashboard_summary AS
SELECT 
    COUNT(*) FILTER (WHERE status IN ('open', 'confirmed')) as open_vulnerabilities,
    COUNT(*) FILTER (WHERE severity = 'critical') as critical_vulnerabilities,
    COUNT(*) FILTER (WHERE geological_context IS NOT NULL) as geological_vulnerabilities,
    COUNT(*) FILTER (WHERE discovery_date >= NOW() - INTERVAL '7 days') as new_vulnerabilities_7d
FROM vulnerabilities;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO threat_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO threat_user;