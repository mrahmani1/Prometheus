#!/bin/bash

# Multi-Cloud Monitoring Setup Script
# Configures monitoring integration with Google Cloud, Azure, and AWS

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if required environment variables are set
check_environment() {
    log "Checking environment variables..."
    
    local required_vars=(
        "GOOGLE_CLOUD_PROJECT"
        "AZURE_SUBSCRIPTION_ID"
        "AWS_REGION"
        "PROMETHEUS_NAMESPACE"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error "Missing required environment variables:"
        printf '%s\n' "${missing_vars[@]}"
        exit 1
    fi
    
    success "All required environment variables are set"
}

# Setup Google Cloud Monitoring
setup_google_cloud_monitoring() {
    log "Setting up Google Cloud Monitoring integration..."
    
    # Enable required APIs
    gcloud services enable monitoring.googleapis.com --project="${GOOGLE_CLOUD_PROJECT}"
    gcloud services enable logging.googleapis.com --project="${GOOGLE_CLOUD_PROJECT}"
    gcloud services enable cloudtrace.googleapis.com --project="${GOOGLE_CLOUD_PROJECT}"
    
    # Create service account for monitoring
    local sa_name="prometheus-ai-monitoring"
    local sa_email="${sa_name}@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com"
    
    if ! gcloud iam service-accounts describe "${sa_email}" --project="${GOOGLE_CLOUD_PROJECT}" >/dev/null 2>&1; then
        gcloud iam service-accounts create "${sa_name}" \
            --display-name="Prometheus AI Monitoring" \
            --description="Service account for Prometheus AI monitoring integration" \
            --project="${GOOGLE_CLOUD_PROJECT}"
    fi
    
    # Grant necessary roles
    local roles=(
        "roles/monitoring.metricWriter"
        "roles/logging.logWriter"
        "roles/cloudtrace.agent"
        "roles/monitoring.viewer"
    )
    
    for role in "${roles[@]}"; do
        gcloud projects add-iam-policy-binding "${GOOGLE_CLOUD_PROJECT}" \
            --member="serviceAccount:${sa_email}" \
            --role="${role}"
    done
    
    # Create and download service account key
    local key_file="/tmp/gcp-monitoring-key.json"
    gcloud iam service-accounts keys create "${key_file}" \
        --iam-account="${sa_email}" \
        --project="${GOOGLE_CLOUD_PROJECT}"
    
    # Create Kubernetes secret
    kubectl create secret generic gcp-monitoring-credentials \
        --from-file=key.json="${key_file}" \
        --namespace="${PROMETHEUS_NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Clean up temporary key file
    rm -f "${key_file}"
    
    success "Google Cloud Monitoring setup completed"
}

# Setup Azure Monitor
setup_azure_monitor() {
    log "Setting up Azure Monitor integration..."
    
    # Create Azure Monitor workspace
    local workspace_name="prometheus-ai-workspace"
    local resource_group="prometheus-ai-monitoring"
    
    # Create resource group if it doesn't exist
    if ! az group show --name "${resource_group}" >/dev/null 2>&1; then
        az group create --name "${resource_group}" --location "East US"
    fi
    
    # Create Log Analytics workspace
    if ! az monitor log-analytics workspace show \
        --resource-group "${resource_group}" \
        --workspace-name "${workspace_name}" >/dev/null 2>&1; then
        az monitor log-analytics workspace create \
            --resource-group "${resource_group}" \
            --workspace-name "${workspace_name}" \
            --location "East US"
    fi
    
    # Get workspace ID and key
    local workspace_id
    workspace_id=$(az monitor log-analytics workspace show \
        --resource-group "${resource_group}" \
        --workspace-name "${workspace_name}" \
        --query customerId -o tsv)
    
    local workspace_key
    workspace_key=$(az monitor log-analytics workspace get-shared-keys \
        --resource-group "${resource_group}" \
        --workspace-name "${workspace_name}" \
        --query primarySharedKey -o tsv)
    
    # Create Kubernetes secret
    kubectl create secret generic azure-monitoring-credentials \
        --from-literal=workspace-id="${workspace_id}" \
        --from-literal=workspace-key="${workspace_key}" \
        --namespace="${PROMETHEUS_NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    success "Azure Monitor setup completed"
}

# Setup AWS CloudWatch
setup_aws_cloudwatch() {
    log "Setting up AWS CloudWatch integration..."
    
    # Create IAM policy for CloudWatch access
    local policy_name="PrometheusAICloudWatchPolicy"
    local policy_document='{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "cloudwatch:PutMetricData",
                    "cloudwatch:GetMetricStatistics",
                    "cloudwatch:ListMetrics",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams"
                ],
                "Resource": "*"
            }
        ]
    }'
    
    # Create policy if it doesn't exist
    if ! aws iam get-policy --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/${policy_name}" >/dev/null 2>&1; then
        aws iam create-policy \
            --policy-name "${policy_name}" \
            --policy-document "${policy_document}"
    fi
    
    # Create IAM user for monitoring
    local user_name="prometheus-ai-monitoring"
    if ! aws iam get-user --user-name "${user_name}" >/dev/null 2>&1; then
        aws iam create-user --user-name "${user_name}"
    fi
    
    # Attach policy to user
    aws iam attach-user-policy \
        --user-name "${user_name}" \
        --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/${policy_name}"
    
    # Create access keys
    local access_keys
    access_keys=$(aws iam create-access-key --user-name "${user_name}" --output json)
    
    local access_key_id
    access_key_id=$(echo "${access_keys}" | jq -r '.AccessKey.AccessKeyId')
    
    local secret_access_key
    secret_access_key=$(echo "${access_keys}" | jq -r '.AccessKey.SecretAccessKey')
    
    # Create Kubernetes secret
    kubectl create secret generic aws-monitoring-credentials \
        --from-literal=access-key-id="${access_key_id}" \
        --from-literal=secret-access-key="${secret_access_key}" \
        --from-literal=region="${AWS_REGION}" \
        --namespace="${PROMETHEUS_NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    success "AWS CloudWatch setup completed"
}

# Setup Prometheus remote write configuration
setup_prometheus_remote_write() {
    log "Configuring Prometheus remote write endpoints..."
    
    # Create ConfigMap with remote write configuration
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-remote-write-config
  namespace: ${PROMETHEUS_NAMESPACE}
data:
  remote-write.yml: |
    remote_write:
      # Google Cloud Monitoring
      - url: "https://monitoring.googleapis.com/v1/projects/${GOOGLE_CLOUD_PROJECT}/location/global/prometheus/api/v1/write"
        queue_config:
          capacity: 10000
          max_samples_per_send: 5000
          batch_send_deadline: 5s
        metadata_config:
          send: true
        oauth2:
          client_id: "\${GOOGLE_CLIENT_ID}"
          client_secret: "\${GOOGLE_CLIENT_SECRET}"
          token_url: "https://oauth2.googleapis.com/token"
          scopes:
            - "https://www.googleapis.com/auth/monitoring.write"
      
      # Azure Monitor (via remote write adapter)
      - url: "https://\${AZURE_WORKSPACE_ID}.ods.opinsights.azure.com/api/logs?api-version=2016-04-01"
        queue_config:
          capacity: 10000
          max_samples_per_send: 5000
        headers:
          "Authorization": "SharedKey \${AZURE_WORKSPACE_ID}:\${AZURE_WORKSPACE_KEY}"
          "Log-Type": "PrometheusMetrics"
      
      # AWS CloudWatch (via remote write adapter)
      - url: "https://aps-workspaces.\${AWS_REGION}.amazonaws.com/workspaces/\${AWS_WORKSPACE_ID}/api/v1/remote_write"
        queue_config:
          capacity: 10000
          max_samples_per_send: 5000
        sigv4:
          region: "\${AWS_REGION}"
          access_key: "\${AWS_ACCESS_KEY_ID}"
          secret_key: "\${AWS_SECRET_ACCESS_KEY}"
EOF
    
    success "Prometheus remote write configuration created"
}

# Setup alerting integrations
setup_alerting_integrations() {
    log "Setting up multi-cloud alerting integrations..."
    
    # PagerDuty integration
    if [[ -n "${PAGERDUTY_ROUTING_KEY:-}" ]]; then
        kubectl create secret generic pagerduty-config \
            --from-literal=routing-key="${PAGERDUTY_ROUTING_KEY}" \
            --namespace="${PROMETHEUS_NAMESPACE}" \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    # Slack integration
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        kubectl create secret generic slack-config \
            --from-literal=webhook-url="${SLACK_WEBHOOK_URL}" \
            --namespace="${PROMETHEUS_NAMESPACE}" \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    # Email SMTP configuration
    if [[ -n "${SMTP_HOST:-}" && -n "${SMTP_USERNAME:-}" && -n "${SMTP_PASSWORD:-}" ]]; then
        kubectl create secret generic smtp-config \
            --from-literal=host="${SMTP_HOST}" \
            --from-literal=username="${SMTP_USERNAME}" \
            --from-literal=password="${SMTP_PASSWORD}" \
            --namespace="${PROMETHEUS_NAMESPACE}" \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    success "Alerting integrations configured"
}

# Deploy monitoring stack
deploy_monitoring_stack() {
    log "Deploying comprehensive monitoring stack..."
    
    # Apply monitoring configurations
    kubectl apply -f kubernetes/advanced-monitoring.yaml
    kubectl apply -f kubernetes/monitoring-configs.yaml
    kubectl apply -f kubernetes/grafana-dashboards.yaml
    
    # Wait for deployments to be ready
    log "Waiting for monitoring components to be ready..."
    
    kubectl wait --for=condition=available --timeout=300s \
        deployment/prometheus-deployment \
        deployment/grafana-deployment \
        deployment/otel-collector \
        deployment/jaeger-all-in-one \
        deployment/alertmanager \
        -n "${PROMETHEUS_NAMESPACE}"
    
    kubectl wait --for=condition=ready --timeout=300s \
        statefulset/elasticsearch \
        -n "${PROMETHEUS_NAMESPACE}"
    
    success "Monitoring stack deployed successfully"
}

# Setup monitoring dashboards
setup_monitoring_dashboards() {
    log "Setting up monitoring dashboards..."
    
    # Import Grafana dashboards
    kubectl apply -f kubernetes/grafana-dashboards.yaml
    
    # Create dashboard provisioning
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-provisioning
  namespace: ${PROMETHEUS_NAMESPACE}
data:
  dashboards.yml: |
    apiVersion: 1
    providers:
      - name: 'geological-monitoring'
        orgId: 1
        folder: 'Geological Monitoring'
        type: file
        disableDeletion: false
        updateIntervalSeconds: 30
        allowUiUpdates: true
        options:
          path: /var/lib/grafana/dashboards/geological
      - name: 'system-monitoring'
        orgId: 1
        folder: 'System Monitoring'
        type: file
        disableDeletion: false
        updateIntervalSeconds: 30
        allowUiUpdates: true
        options:
          path: /var/lib/grafana/dashboards/system
      - name: 'business-metrics'
        orgId: 1
        folder: 'Business Metrics'
        type: file
        disableDeletion: false
        updateIntervalSeconds: 30
        allowUiUpdates: true
        options:
          path: /var/lib/grafana/dashboards/business
EOF
    
    success "Monitoring dashboards configured"
}

# Validate monitoring setup
validate_monitoring_setup() {
    log "Validating monitoring setup..."
    
    # Check if all services are running
    local services=(
        "prometheus-service"
        "grafana-service"
        "otel-collector-service"
        "jaeger-service"
        "elasticsearch-service"
        "alertmanager-service"
    )
    
    for service in "${services[@]}"; do
        if kubectl get service "${service}" -n "${PROMETHEUS_NAMESPACE}" >/dev/null 2>&1; then
            success "Service ${service} is running"
        else
            error "Service ${service} is not running"
            return 1
        fi
    done
    
    # Test Prometheus connectivity
    log "Testing Prometheus connectivity..."
    kubectl port-forward -n "${PROMETHEUS_NAMESPACE}" service/prometheus-service 9090:9090 &
    local port_forward_pid=$!
    sleep 5
    
    if curl -s http://localhost:9090/-/healthy >/dev/null; then
        success "Prometheus is healthy"
    else
        error "Prometheus health check failed"
        kill $port_forward_pid 2>/dev/null || true
        return 1
    fi
    
    kill $port_forward_pid 2>/dev/null || true
    
    # Test Grafana connectivity
    log "Testing Grafana connectivity..."
    kubectl port-forward -n "${PROMETHEUS_NAMESPACE}" service/grafana-service 3001:3001 &
    port_forward_pid=$!
    sleep 5
    
    if curl -s http://localhost:3001/api/health >/dev/null; then
        success "Grafana is healthy"
    else
        error "Grafana health check failed"
        kill $port_forward_pid 2>/dev/null || true
        return 1
    fi
    
    kill $port_forward_pid 2>/dev/null || true
    
    success "Monitoring setup validation completed successfully"
}

# Print access information
print_access_info() {
    log "Monitoring setup completed successfully!"
    echo
    echo "Access Information:"
    echo "=================="
    echo
    echo "Prometheus:"
    echo "  kubectl port-forward -n ${PROMETHEUS_NAMESPACE} service/prometheus-service 9090:9090"
    echo "  Then access: http://localhost:9090"
    echo
    echo "Grafana:"
    echo "  kubectl port-forward -n ${PROMETHEUS_NAMESPACE} service/grafana-service 3001:3001"
    echo "  Then access: http://localhost:3001"
    echo "  Default credentials: admin/prometheus123"
    echo
    echo "Jaeger:"
    echo "  kubectl port-forward -n ${PROMETHEUS_NAMESPACE} service/jaeger-service 16686:16686"
    echo "  Then access: http://localhost:16686"
    echo
    echo "Kibana:"
    echo "  kubectl port-forward -n ${PROMETHEUS_NAMESPACE} service/kibana-service 5601:5601"
    echo "  Then access: http://localhost:5601/kibana"
    echo
    echo "AlertManager:"
    echo "  kubectl port-forward -n ${PROMETHEUS_NAMESPACE} service/alertmanager-service 9093:9093"
    echo "  Then access: http://localhost:9093"
    echo
}

# Main execution
main() {
    log "Starting multi-cloud monitoring setup..."
    
    # Set default values
    PROMETHEUS_NAMESPACE="${PROMETHEUS_NAMESPACE:-prometheus-ai}"
    
    check_environment
    
    # Setup cloud integrations
    if [[ "${SETUP_GCP:-true}" == "true" ]]; then
        setup_google_cloud_monitoring
    fi
    
    if [[ "${SETUP_AZURE:-true}" == "true" ]]; then
        setup_azure_monitor
    fi
    
    if [[ "${SETUP_AWS:-true}" == "true" ]]; then
        setup_aws_cloudwatch
    fi
    
    setup_prometheus_remote_write
    setup_alerting_integrations
    deploy_monitoring_stack
    setup_monitoring_dashboards
    validate_monitoring_setup
    print_access_info
    
    success "Multi-cloud monitoring setup completed successfully!"
}

# Run main function
main "$@"