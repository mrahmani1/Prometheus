#!/bin/bash
# Deploy System Monitoring and Alerting Infrastructure
# This script deploys the comprehensive monitoring system for Prometheus AI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="prometheus-ai"
DOCKER_REGISTRY="prometheus-ai"
MONITORING_VERSION="latest"

echo -e "${BLUE}🚀 Deploying Prometheus AI System Monitoring Infrastructure${NC}"
echo "=================================================="

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    print_error "docker is not installed or not in PATH"
    exit 1
fi

print_status "Prerequisites check passed"

# Check if namespace exists
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    print_warning "Namespace $NAMESPACE does not exist, creating it..."
    kubectl create namespace $NAMESPACE
    print_status "Namespace $NAMESPACE created"
else
    print_status "Namespace $NAMESPACE already exists"
fi

# Build and push system monitor Docker image
echo -e "${BLUE}Building system monitor Docker image...${NC}"
cd services/system-monitor

if docker build -t $DOCKER_REGISTRY/system-monitor:$MONITORING_VERSION .; then
    print_status "System monitor image built successfully"
else
    print_error "Failed to build system monitor image"
    exit 1
fi

# Push image (in production, you'd push to a registry)
print_warning "In production, push image to registry: docker push $DOCKER_REGISTRY/system-monitor:$MONITORING_VERSION"

cd ../..

# Create database schema
echo -e "${BLUE}Setting up monitoring database schema...${NC}"

# Check if PostgreSQL is running
if kubectl get pods -n $NAMESPACE | grep postgres | grep Running &> /dev/null; then
    print_status "PostgreSQL is running, applying monitoring schema..."
    
    # Apply the monitoring schema
    kubectl exec -n $NAMESPACE deployment/postgres -- psql -U prometheus -d prometheus -f - < database/postgres/monitoring_schema.sql
    
    if [ $? -eq 0 ]; then
        print_status "Monitoring database schema applied successfully"
    else
        print_warning "Failed to apply monitoring schema, continuing anyway..."
    fi
else
    print_warning "PostgreSQL is not running, skipping schema setup"
fi

# Deploy monitoring infrastructure
echo -e "${BLUE}Deploying monitoring infrastructure...${NC}"

# Deploy enhanced monitoring configuration
if kubectl apply -f kubernetes/monitoring.yaml; then
    print_status "Prometheus and Grafana monitoring deployed"
else
    print_error "Failed to deploy monitoring infrastructure"
    exit 1
fi

# Deploy system monitor service
if kubectl apply -f kubernetes/system-monitor-deployment.yaml; then
    print_status "System monitor service deployed"
else
    print_error "Failed to deploy system monitor service"
    exit 1
fi

# Deploy enhanced autoscaling
if kubectl apply -f kubernetes/enhanced-autoscaling.yaml; then
    print_status "Enhanced autoscaling configuration deployed"
else
    print_warning "Failed to deploy enhanced autoscaling, using basic configuration"
    kubectl apply -f kubernetes/autoscaling.yaml
fi

# Wait for deployments to be ready
echo -e "${BLUE}Waiting for deployments to be ready...${NC}"

deployments=("prometheus" "alertmanager" "grafana" "system-monitor-deployment")

for deployment in "${deployments[@]}"; do
    echo "Waiting for $deployment to be ready..."
    if kubectl wait --for=condition=available --timeout=300s deployment/$deployment -n $NAMESPACE; then
        print_status "$deployment is ready"
    else
        print_warning "$deployment is not ready within timeout, continuing..."
    fi
done

# Verify monitoring services
echo -e "${BLUE}Verifying monitoring services...${NC}"

# Check Prometheus
if kubectl get pods -n $NAMESPACE | grep prometheus | grep Running &> /dev/null; then
    print_status "Prometheus is running"
    
    # Get Prometheus service URL
    PROMETHEUS_PORT=$(kubectl get svc prometheus-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].port}')
    print_status "Prometheus available at: http://localhost:$PROMETHEUS_PORT (use kubectl port-forward)"
else
    print_error "Prometheus is not running"
fi

# Check Grafana
if kubectl get pods -n $NAMESPACE | grep grafana | grep Running &> /dev/null; then
    print_status "Grafana is running"
    
    # Get Grafana service URL
    GRAFANA_PORT=$(kubectl get svc grafana -n $NAMESPACE -o jsonpath='{.spec.ports[0].port}')
    print_status "Grafana available at: http://localhost:$GRAFANA_PORT (use kubectl port-forward)"
    print_status "Default Grafana credentials: admin/admin123"
else
    print_error "Grafana is not running"
fi

# Check AlertManager
if kubectl get pods -n $NAMESPACE | grep alertmanager | grep Running &> /dev/null; then
    print_status "AlertManager is running"
else
    print_error "AlertManager is not running"
fi

# Check System Monitor
if kubectl get pods -n $NAMESPACE | grep system-monitor | grep Running &> /dev/null; then
    print_status "System Monitor is running"
    
    # Get System Monitor service URL
    MONITOR_PORT=$(kubectl get svc system-monitor-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].port}')
    print_status "System Monitor API available at: http://localhost:$MONITOR_PORT (use kubectl port-forward)"
else
    print_error "System Monitor is not running"
fi

# Setup port forwarding commands
echo -e "${BLUE}Setting up port forwarding...${NC}"

# Create port forwarding script
cat > scripts/port-forward-monitoring.sh << 'EOF'
#!/bin/bash
# Port forwarding script for monitoring services

NAMESPACE="prometheus-ai"

echo "Setting up port forwarding for monitoring services..."
echo "Press Ctrl+C to stop all port forwards"

# Start port forwards in background
kubectl port-forward -n $NAMESPACE svc/prometheus-service 9090:9090 &
PROMETHEUS_PID=$!

kubectl port-forward -n $NAMESPACE svc/grafana 3000:3000 &
GRAFANA_PID=$!

kubectl port-forward -n $NAMESPACE svc/alertmanager 9093:9093 &
ALERTMANAGER_PID=$!

kubectl port-forward -n $NAMESPACE svc/system-monitor-service 8000:8000 &
MONITOR_PID=$!

echo "Port forwards started:"
echo "  Prometheus:    http://localhost:9090"
echo "  Grafana:       http://localhost:3000 (admin/admin123)"
echo "  AlertManager:  http://localhost:9093"
echo "  System Monitor: http://localhost:8000"

# Wait for interrupt
trap "kill $PROMETHEUS_PID $GRAFANA_PID $ALERTMANAGER_PID $MONITOR_PID; exit" INT
wait
EOF

chmod +x scripts/port-forward-monitoring.sh
print_status "Port forwarding script created: scripts/port-forward-monitoring.sh"

# Create monitoring dashboard URLs
echo -e "${BLUE}Creating monitoring dashboard access...${NC}"

cat > monitoring-access.md << 'EOF'
# Monitoring Dashboard Access

## Quick Start
Run the port forwarding script to access all monitoring services:
```bash
./scripts/port-forward-monitoring.sh
```

## Individual Services

### Prometheus (Metrics & Alerts)
- URL: http://localhost:9090
- Port Forward: `kubectl port-forward -n prometheus-ai svc/prometheus-service 9090:9090`

### Grafana (Dashboards & Visualization)
- URL: http://localhost:3000
- Credentials: admin/admin123
- Port Forward: `kubectl port-forward -n prometheus-ai svc/grafana 3000:3000`

### AlertManager (Alert Management)
- URL: http://localhost:9093
- Port Forward: `kubectl port-forward -n prometheus-ai svc/alertmanager 9093:9093`

### System Monitor (Health & Incidents)
- URL: http://localhost:8000
- API Endpoints:
  - `/system/status` - Overall system status
  - `/scaling/status` - Scaling status
  - `/health` - Service health
  - `/metrics` - Prometheus metrics
- Port Forward: `kubectl port-forward -n prometheus-ai svc/system-monitor-service 8000:8000`

## Key Monitoring Features

### Automated Scaling
- Intelligent HPA based on custom metrics
- Emergency scaling for volcanic monitoring
- Resource optimization with VPA

### Incident Response
- Automated service restart
- Intelligent failover
- Escalation procedures
- Rate-limited notifications

### Alerting Channels
- Email notifications
- Slack integration
- PagerDuty integration
- Webhook notifications

### Health Monitoring
- Service dependency tracking
- Performance metrics
- Real-time status dashboard
- Historical trend analysis

## Emergency Procedures

### Volcanic Monitoring Emergency
If the volcano-watchtower service fails:
1. Automatic failover is triggered
2. Emergency notifications sent
3. Backup systems activated
4. Immediate escalation to management

### Critical Service Failures
For critical service failures:
1. Automatic restart attempts
2. Scaling adjustments
3. On-call notifications
4. Incident tracking

## Troubleshooting

### Check Service Status
```bash
kubectl get pods -n prometheus-ai
kubectl get svc -n prometheus-ai
kubectl logs -f deployment/system-monitor-deployment -n prometheus-ai
```

### View Recent Alerts
```bash
kubectl logs -f deployment/alertmanager -n prometheus-ai
```

### Check Scaling Status
```bash
kubectl get hpa -n prometheus-ai
kubectl describe hpa volcano-watchtower-enhanced-hpa -n prometheus-ai
```
EOF

print_status "Monitoring access guide created: monitoring-access.md"

# Run basic health checks
echo -e "${BLUE}Running basic health checks...${NC}"

# Check if all pods are running
TOTAL_PODS=$(kubectl get pods -n $NAMESPACE --no-headers | wc -l)
RUNNING_PODS=$(kubectl get pods -n $NAMESPACE --no-headers | grep Running | wc -l)

echo "Pod Status: $RUNNING_PODS/$TOTAL_PODS running"

if [ $RUNNING_PODS -eq $TOTAL_PODS ]; then
    print_status "All pods are running successfully"
else
    print_warning "Some pods are not running. Check with: kubectl get pods -n $NAMESPACE"
fi

# Check services
SERVICES=$(kubectl get svc -n $NAMESPACE --no-headers | wc -l)
print_status "$SERVICES services deployed"

# Check HPAs
HPAS=$(kubectl get hpa -n $NAMESPACE --no-headers | wc -l)
print_status "$HPAS horizontal pod autoscalers configured"

echo ""
echo -e "${GREEN}🎉 System Monitoring Infrastructure Deployment Complete!${NC}"
echo "=================================================="
echo ""
echo "Next Steps:"
echo "1. Run: ./scripts/port-forward-monitoring.sh"
echo "2. Open Grafana at http://localhost:3000 (admin/admin123)"
echo "3. Open Prometheus at http://localhost:9090"
echo "4. Check System Monitor at http://localhost:8000/system/status"
echo "5. Review monitoring-access.md for detailed information"
echo ""
echo "The system is now monitoring:"
echo "  ✓ Service health and dependencies"
echo "  ✓ Automated scaling and resource optimization"
echo "  ✓ Incident response and escalation"
echo "  ✓ Multi-channel alerting (email, Slack, PagerDuty)"
echo "  ✓ Performance metrics and trends"
echo ""
echo -e "${YELLOW}Important: Configure notification channels in the system-monitor service${NC}"
echo -e "${YELLOW}Update SMTP, Slack, and PagerDuty credentials in the deployment${NC}"