# System Monitoring and Alerting Infrastructure

This service provides comprehensive monitoring, health checking, automated scaling, and incident response for the Prometheus AI Geodynamic Platform.

## Features

### 🔍 Comprehensive Monitoring
- **Service Health Monitoring**: Real-time health checks for all microservices
- **Dependency Tracking**: Monitor service dependencies (databases, caches, message queues)
- **Performance Metrics**: CPU, memory, response times, and custom application metrics
- **System-wide Status**: Overall platform health assessment

### 🚨 Intelligent Alerting
- **Multi-channel Notifications**: Email, Slack, PagerDuty, webhooks
- **Severity-based Routing**: Different notification channels based on alert severity
- **Rate Limiting**: Prevent notification spam with intelligent rate limiting
- **Escalation Procedures**: Automatic escalation based on time and severity

### 🔄 Automated Incident Response
- **Service Recovery**: Automatic service restarts and failover
- **Intelligent Scaling**: Dynamic scaling based on load and performance metrics
- **Emergency Procedures**: Special handling for critical services (volcanic monitoring)
- **Cooldown Management**: Prevent excessive automated actions

### 📊 Advanced Scaling
- **Custom Metrics Scaling**: Scale based on application-specific metrics
- **Emergency Scaling**: Rapid scaling for critical volcanic monitoring
- **Resource Optimization**: Vertical Pod Autoscaling for resource efficiency
- **Pod Disruption Budgets**: Ensure service availability during updates

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │    Incident     │    │  Notification   │
│    Service      │────│    Response     │────│    Service      │
│                 │    │     System      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Kubernetes    │    │   Prometheus    │    │   AlertManager  │
│      API        │    │    Metrics      │    │   & External    │
│                 │    │                 │    │    Systems      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Components

### 1. Monitoring Service (`main.py`)
- **Health Checks**: Performs regular health checks on all services
- **Metrics Collection**: Gathers system and application metrics
- **Status Assessment**: Determines overall system health
- **Scaling Decisions**: Makes intelligent scaling recommendations

### 2. Incident Response System (`incident_response.py`)
- **Automated Actions**: Executes predefined responses to incidents
- **Rule Engine**: Matches incidents to appropriate response actions
- **Cooldown Management**: Prevents excessive automated interventions
- **Escalation Logic**: Escalates unresolved incidents

### 3. Notification Service (`notification_service.py`)
- **Multi-channel Support**: Email, Slack, PagerDuty, webhooks
- **Template Engine**: Customizable notification templates
- **Rate Limiting**: Intelligent notification throttling
- **Escalation Handling**: Manages notification escalation chains

## Configuration

### Service Configuration
Services are configured in the `MonitoringService._load_service_configs()` method:

```python
{
    "service-name": {
        "url": "http://service-name:port",
        "health_endpoint": "/health",
        "metrics_endpoint": "/metrics",
        "critical": True,  # Critical service flag
        "dependencies": ["redis", "postgresql"],
        "scaling": {
            "min_replicas": 2,
            "max_replicas": 10,
            "cpu_threshold": 70,
            "memory_threshold": 80
        }
    }
}
```

### Notification Channels
Configure notification channels in `NotificationService._load_notification_channels()`:

```python
{
    "channel_name": NotificationChannel(
        name="Display Name",
        type=NotificationType.EMAIL,
        config={
            "smtp_server": "smtp.example.com",
            "username": "alerts@example.com",
            "recipients": ["team@example.com"]
        },
        rate_limit_per_hour=50
    )
}
```

### Escalation Rules
Define escalation rules in `NotificationService._load_escalation_rules()`:

```python
EscalationRule(
    service="volcano-watchtower",
    severity="emergency",
    level=EscalationLevel.LEVEL_4,
    timeout_minutes=0,  # Immediate
    channels=["emergency_email", "pagerduty_emergency"],
    conditions={"immediate": True}
)
```

## Deployment

### Prerequisites
- Kubernetes cluster with RBAC enabled
- PostgreSQL database for storing monitoring data
- Redis for caching and queuing
- Prometheus and Grafana (deployed via monitoring.yaml)

### Deploy the Monitoring Infrastructure

1. **Build and Deploy**:
   ```bash
   # Build the Docker image
   cd services/system-monitor
   docker build -t prometheus-ai/system-monitor:latest .
   
   # Deploy the database schema
   kubectl exec -n prometheus-ai deployment/postgres -- psql -U prometheus -d prometheus -f - < ../../database/postgres/monitoring_schema.sql
   
   # Deploy the monitoring infrastructure
   kubectl apply -f ../../kubernetes/monitoring.yaml
   kubectl apply -f ../../kubernetes/system-monitor-deployment.yaml
   kubectl apply -f ../../kubernetes/enhanced-autoscaling.yaml
   ```

2. **Verify Deployment**:
   ```bash
   kubectl get pods -n prometheus-ai
   kubectl get svc -n prometheus-ai
   kubectl get hpa -n prometheus-ai
   ```

3. **Access Services**:
   ```bash
   # Port forward to access services locally
   kubectl port-forward -n prometheus-ai svc/system-monitor-service 8000:8000
   kubectl port-forward -n prometheus-ai svc/prometheus-service 9090:9090
   kubectl port-forward -n prometheus-ai svc/grafana 3000:3000
   ```

## API Endpoints

### Health and Status
- `GET /health` - Service health check
- `GET /system/status` - Overall system status
- `GET /scaling/status` - Current scaling status

### Alert Management
- `POST /alerts/{alert_id}/resolve` - Manually resolve an alert

### Metrics
- `GET /metrics` - Prometheus metrics endpoint

## Monitoring Metrics

### Custom Metrics
- `health_checks_total` - Total health checks performed
- `service_response_time_seconds` - Service response times
- `system_load_average` - System load metrics
- `incidents_total` - Total incidents detected
- `scaling_actions_total` - Scaling actions performed

### Application Metrics
- `ai_model_inference_queue_depth` - AI model processing queue
- `seismic_events_processing_rate` - Volcanic monitoring throughput
- `kafka_consumer_lag` - Data ingestion lag
- `volcanic_alert_queue_depth` - Emergency alert queue

## Emergency Procedures

### Volcanic Monitoring Emergency
The volcanic monitoring system (`volcano-watchtower`) has special emergency handling:

1. **Immediate Response**: Zero-delay escalation for emergency alerts
2. **Automatic Failover**: Backup systems activated automatically
3. **Emergency Scaling**: Rapid scaling to maximum capacity
4. **Executive Notification**: Immediate notification to executive team

### Critical Service Failures
For critical service failures:

1. **Automatic Restart**: Service restart attempted automatically
2. **Scaling Adjustment**: Resources scaled up if needed
3. **Dependency Check**: Dependencies verified and restored
4. **Escalation**: On-call team notified if restart fails

## Testing

Run the test suite:

```bash
cd services/system-monitor
pip install -r requirements.txt
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Service interaction testing
- **Performance Tests**: Load and performance validation
- **Emergency Response Tests**: Critical path validation

## Troubleshooting

### Common Issues

1. **Service Health Checks Failing**:
   ```bash
   kubectl logs -f deployment/system-monitor-deployment -n prometheus-ai
   kubectl describe pod <pod-name> -n prometheus-ai
   ```

2. **Scaling Not Working**:
   ```bash
   kubectl get hpa -n prometheus-ai
   kubectl describe hpa <hpa-name> -n prometheus-ai
   kubectl top pods -n prometheus-ai
   ```

3. **Notifications Not Sending**:
   - Check notification service logs
   - Verify SMTP/Slack/PagerDuty credentials
   - Check rate limiting status

4. **Database Connection Issues**:
   ```bash
   kubectl exec -n prometheus-ai deployment/postgres -- psql -U prometheus -d prometheus -c "SELECT 1;"
   ```

### Debug Commands

```bash
# Check all monitoring components
kubectl get all -n prometheus-ai -l component=monitoring

# View system monitor logs
kubectl logs -f deployment/system-monitor-deployment -n prometheus-ai

# Check Prometheus targets
kubectl port-forward -n prometheus-ai svc/prometheus-service 9090:9090
# Visit http://localhost:9090/targets

# View current alerts
kubectl port-forward -n prometheus-ai svc/alertmanager 9093:9093
# Visit http://localhost:9093
```

## Security Considerations

- **RBAC**: Service account with minimal required permissions
- **Secrets**: Database and API credentials stored in Kubernetes secrets
- **Network Policies**: Restrict pod-to-pod communication
- **TLS**: Enable TLS for external communications
- **Rate Limiting**: Prevent abuse of notification channels

## Performance Tuning

### Resource Limits
Adjust resource limits based on cluster size:

```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### Scaling Configuration
Tune HPA settings for your workload:

```yaml
behavior:
  scaleUp:
    stabilizationWindowSeconds: 60
    policies:
    - type: Percent
      value: 100
      periodSeconds: 15
```

### Database Optimization
For high-volume environments:
- Enable table partitioning for large tables
- Configure appropriate indexes
- Set up read replicas for reporting

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for configuration changes
4. Test emergency scenarios thoroughly
5. Ensure backward compatibility

## License

This monitoring system is part of the Prometheus AI Geodynamic Platform.