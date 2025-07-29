-- System Monitoring Database Schema
-- Creates tables for storing monitoring data, alerts, and incident responses

-- Create monitoring schema
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Health check results table
CREATE TABLE IF NOT EXISTS monitoring.health_checks (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('healthy', 'degraded', 'unhealthy', 'down')),
    response_time FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    details JSONB,
    dependencies TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for efficient querying
CREATE INDEX IF NOT EXISTS idx_health_checks_service_timestamp 
ON monitoring.health_checks(service_name, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_health_checks_status 
ON monitoring.health_checks(status);

-- System metrics table
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id SERIAL PRIMARY KEY,
    cpu_usage FLOAT NOT NULL,
    memory_usage FLOAT NOT NULL,
    disk_usage FLOAT NOT NULL,
    network_io JSONB,
    active_connections INTEGER NOT NULL,
    queue_depth INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for time-series queries
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp 
ON monitoring.system_metrics(timestamp DESC);

-- Alerts table
CREATE TABLE IF NOT EXISTS monitoring.alerts (
    id VARCHAR(255) PRIMARY KEY,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical', 'emergency')),
    service VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    escalated BOOLEAN DEFAULT FALSE,
    escalated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for alerts
CREATE INDEX IF NOT EXISTS idx_alerts_service_timestamp 
ON monitoring.alerts(service, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_severity 
ON monitoring.alerts(severity);

CREATE INDEX IF NOT EXISTS idx_alerts_resolved 
ON monitoring.alerts(resolved);

-- Incidents table
CREATE TABLE IF NOT EXISTS monitoring.incidents (
    id VARCHAR(255) PRIMARY KEY,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical', 'emergency')),
    service VARCHAR(100) NOT NULL,
    actions TEXT[],
    executed_actions TEXT[],
    escalation_level INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_action_at TIMESTAMP WITH TIME ZONE,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for incidents
CREATE INDEX IF NOT EXISTS idx_incidents_service_created 
ON monitoring.incidents(service, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_incidents_severity 
ON monitoring.incidents(severity);

CREATE INDEX IF NOT EXISTS idx_incidents_resolved 
ON monitoring.incidents(resolved);

-- Scaling actions table
CREATE TABLE IF NOT EXISTS monitoring.scaling_actions (
    id SERIAL PRIMARY KEY,
    service VARCHAR(100) NOT NULL,
    action VARCHAR(20) NOT NULL CHECK (action IN ('scale_up', 'scale_down')),
    from_replicas INTEGER NOT NULL,
    to_replicas INTEGER NOT NULL,
    reason TEXT,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    success BOOLEAN NOT NULL,
    error_message TEXT
);

-- Create index for scaling actions
CREATE INDEX IF NOT EXISTS idx_scaling_actions_service_timestamp 
ON monitoring.scaling_actions(service, timestamp DESC);

-- Performance metrics table for detailed service metrics
CREATE TABLE IF NOT EXISTS monitoring.performance_metrics (
    id SERIAL PRIMARY KEY,
    service VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(20),
    labels JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for performance metrics
CREATE INDEX IF NOT EXISTS idx_performance_metrics_service_metric_timestamp 
ON monitoring.performance_metrics(service, metric_name, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_labels 
ON monitoring.performance_metrics USING GIN(labels);

-- Service dependencies table
CREATE TABLE IF NOT EXISTS monitoring.service_dependencies (
    id SERIAL PRIMARY KEY,
    service VARCHAR(100) NOT NULL,
    dependency VARCHAR(100) NOT NULL,
    dependency_type VARCHAR(50) NOT NULL, -- 'database', 'cache', 'message_queue', 'service'
    critical BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(service, dependency)
);

-- Create index for service dependencies
CREATE INDEX IF NOT EXISTS idx_service_dependencies_service 
ON monitoring.service_dependencies(service);

-- Notification log table
CREATE TABLE IF NOT EXISTS monitoring.notifications (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(255) REFERENCES monitoring.alerts(id),
    incident_id VARCHAR(255) REFERENCES monitoring.incidents(id),
    notification_type VARCHAR(50) NOT NULL, -- 'email', 'slack', 'pagerduty', 'webhook'
    recipient VARCHAR(255) NOT NULL,
    subject TEXT,
    message TEXT NOT NULL,
    sent_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    success BOOLEAN NOT NULL,
    error_message TEXT
);

-- Create indexes for notifications
CREATE INDEX IF NOT EXISTS idx_notifications_alert_id 
ON monitoring.notifications(alert_id);

CREATE INDEX IF NOT EXISTS idx_notifications_incident_id 
ON monitoring.notifications(incident_id);

CREATE INDEX IF NOT EXISTS idx_notifications_sent_at 
ON monitoring.notifications(sent_at DESC);

-- Views for common queries

-- Current system status view
CREATE OR REPLACE VIEW monitoring.current_system_status AS
SELECT 
    service_name,
    status,
    response_time,
    timestamp,
    details,
    ROW_NUMBER() OVER (PARTITION BY service_name ORDER BY timestamp DESC) as rn
FROM monitoring.health_checks
WHERE timestamp > NOW() - INTERVAL '5 minutes';

-- Active alerts view
CREATE OR REPLACE VIEW monitoring.active_alerts AS
SELECT 
    id,
    severity,
    service,
    message,
    details,
    timestamp,
    escalated,
    EXTRACT(EPOCH FROM (NOW() - timestamp))/60 as age_minutes
FROM monitoring.alerts
WHERE resolved = FALSE
ORDER BY 
    CASE severity 
        WHEN 'emergency' THEN 1
        WHEN 'critical' THEN 2
        WHEN 'high' THEN 3
        WHEN 'medium' THEN 4
        WHEN 'low' THEN 5
    END,
    timestamp DESC;

-- Service health summary view
CREATE OR REPLACE VIEW monitoring.service_health_summary AS
SELECT 
    service_name,
    COUNT(*) as total_checks,
    COUNT(CASE WHEN status = 'healthy' THEN 1 END) as healthy_checks,
    COUNT(CASE WHEN status = 'degraded' THEN 1 END) as degraded_checks,
    COUNT(CASE WHEN status = 'unhealthy' THEN 1 END) as unhealthy_checks,
    COUNT(CASE WHEN status = 'down' THEN 1 END) as down_checks,
    AVG(response_time) as avg_response_time,
    MAX(response_time) as max_response_time,
    MIN(timestamp) as first_check,
    MAX(timestamp) as last_check
FROM monitoring.health_checks
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY service_name;

-- Recent incidents view
CREATE OR REPLACE VIEW monitoring.recent_incidents AS
SELECT 
    i.id,
    i.severity,
    i.service,
    i.escalation_level,
    i.created_at,
    i.resolved,
    i.resolved_at,
    COALESCE(array_length(i.executed_actions, 1), 0) as actions_taken,
    EXTRACT(EPOCH FROM (COALESCE(i.resolved_at, NOW()) - i.created_at))/60 as duration_minutes
FROM monitoring.incidents i
WHERE i.created_at > NOW() - INTERVAL '7 days'
ORDER BY i.created_at DESC;

-- Functions for common operations

-- Function to get service health status
CREATE OR REPLACE FUNCTION monitoring.get_service_health(service_name_param VARCHAR)
RETURNS TABLE(
    service VARCHAR,
    current_status VARCHAR,
    last_check TIMESTAMP WITH TIME ZONE,
    response_time FLOAT,
    uptime_percentage FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        hc.service_name::VARCHAR,
        hc.status::VARCHAR,
        hc.timestamp,
        hc.response_time,
        (COUNT(CASE WHEN hc2.status = 'healthy' THEN 1 END) * 100.0 / COUNT(*))::FLOAT
    FROM monitoring.health_checks hc
    LEFT JOIN monitoring.health_checks hc2 ON hc2.service_name = hc.service_name 
        AND hc2.timestamp > NOW() - INTERVAL '24 hours'
    WHERE hc.service_name = service_name_param
        AND hc.timestamp = (
            SELECT MAX(timestamp) 
            FROM monitoring.health_checks 
            WHERE service_name = service_name_param
        )
    GROUP BY hc.service_name, hc.status, hc.timestamp, hc.response_time;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate service availability
CREATE OR REPLACE FUNCTION monitoring.calculate_availability(
    service_name_param VARCHAR,
    hours_back INTEGER DEFAULT 24
)
RETURNS FLOAT AS $$
DECLARE
    availability FLOAT;
BEGIN
    SELECT 
        (COUNT(CASE WHEN status IN ('healthy', 'degraded') THEN 1 END) * 100.0 / COUNT(*))
    INTO availability
    FROM monitoring.health_checks
    WHERE service_name = service_name_param
        AND timestamp > NOW() - (hours_back || ' hours')::INTERVAL;
    
    RETURN COALESCE(availability, 0.0);
END;
$$ LANGUAGE plpgsql;

-- Function to get alert statistics
CREATE OR REPLACE FUNCTION monitoring.get_alert_stats(days_back INTEGER DEFAULT 7)
RETURNS TABLE(
    severity VARCHAR,
    total_alerts BIGINT,
    resolved_alerts BIGINT,
    avg_resolution_time_minutes FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        a.severity::VARCHAR,
        COUNT(*)::BIGINT,
        COUNT(CASE WHEN a.resolved THEN 1 END)::BIGINT,
        AVG(EXTRACT(EPOCH FROM (a.resolved_at - a.timestamp))/60)::FLOAT
    FROM monitoring.alerts a
    WHERE a.timestamp > NOW() - (days_back || ' days')::INTERVAL
    GROUP BY a.severity
    ORDER BY 
        CASE a.severity 
            WHEN 'emergency' THEN 1
            WHEN 'critical' THEN 2
            WHEN 'high' THEN 3
            WHEN 'medium' THEN 4
            WHEN 'low' THEN 5
        END;
END;
$$ LANGUAGE plpgsql;

-- Insert default service dependencies
INSERT INTO monitoring.service_dependencies (service, dependency, dependency_type, critical) VALUES
('api-gateway', 'redis', 'cache', true),
('api-gateway', 'consul', 'service_discovery', true),
('helios-core', 'postgresql', 'database', true),
('helios-core', 'redis', 'cache', true),
('helios-core', 'influxdb', 'database', true),
('volcano-watchtower', 'kafka', 'message_queue', true),
('volcano-watchtower', 'influxdb', 'database', true),
('volcano-watchtower', 'redis', 'cache', true),
('data-ingestion', 'kafka', 'message_queue', true),
('data-ingestion', 'postgresql', 'database', true),
('data-ingestion', 'influxdb', 'database', true),
('nlp-service', 'redis', 'cache', false)
ON CONFLICT (service, dependency) DO NOTHING;

-- Create partitioning for large tables (optional, for high-volume environments)
-- Partition health_checks by month
-- CREATE TABLE monitoring.health_checks_y2024m01 PARTITION OF monitoring.health_checks
-- FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Grant permissions
GRANT USAGE ON SCHEMA monitoring TO prometheus;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA monitoring TO prometheus;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA monitoring TO prometheus;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA monitoring TO prometheus;