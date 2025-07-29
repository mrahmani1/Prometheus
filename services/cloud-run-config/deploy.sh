#!/bin/bash

# Cloud Run Multi-Region Deployment Script
# Deploys Prometheus AI services to Cloud Run with cost optimization

set -e

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"prometheus-ai-platform"}
REGIONS=("us-central1" "europe-west1" "asia-northeast1")
SERVICES=("helios-core" "api-gateway" "data-ingestion" "volcano-watchtower")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not authenticated with gcloud. Please run 'gcloud auth login'"
        exit 1
    fi
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    log_success "Prerequisites check passed"
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required Google Cloud APIs..."
    
    gcloud services enable \
        run.googleapis.com \
        cloudbuild.googleapis.com \
        containerregistry.googleapis.com \
        monitoring.googleapis.com \
        logging.googleapis.com \
        dns.googleapis.com \
        compute.googleapis.com
    
    log_success "APIs enabled"
}

# Build and push container images
build_and_push_images() {
    log_info "Building and pushing container images..."
    
    for service in "${SERVICES[@]}"; do
        log_info "Building $service..."
        
        # Determine which Dockerfile to use
        if [ -f "services/cloud-run-config/Dockerfile.${service}-cloudrun" ]; then
            dockerfile="services/cloud-run-config/Dockerfile.${service}-cloudrun"
        else
            dockerfile="services/$service/Dockerfile"
        fi
        
        # Build image
        docker build -t gcr.io/$PROJECT_ID/$service:latest \
            -f $dockerfile \
            services/$service/
        
        # Push image
        docker push gcr.io/$PROJECT_ID/$service:latest
        
        log_success "Built and pushed $service"
    done
}

# Deploy service to a specific region
deploy_service_to_region() {
    local service=$1
    local region=$2
    
    log_info "Deploying $service to $region..."
    
    # Get service-specific configuration
    case $service in
        "helios-core")
            cpu="4"
            memory="8Gi"
            concurrency="10"
            timeout="300s"
            min_instances="0"
            max_instances="50"
            ;;
        "api-gateway")
            cpu="2"
            memory="4Gi"
            concurrency="1000"
            timeout="60s"
            min_instances="1"
            max_instances="100"
            ;;
        "data-ingestion")
            cpu="2"
            memory="4Gi"
            concurrency="100"
            timeout="600s"
            min_instances="0"
            max_instances="20"
            ;;
        "volcano-watchtower")
            cpu="3"
            memory="6Gi"
            concurrency="50"
            timeout="180s"
            min_instances="1"
            max_instances="30"
            ;;
        *)
            cpu="1"
            memory="2Gi"
            concurrency="100"
            timeout="60s"
            min_instances="0"
            max_instances="10"
            ;;
    esac
    
    # Deploy to Cloud Run
    gcloud run deploy $service \
        --image gcr.io/$PROJECT_ID/$service:latest \
        --platform managed \
        --region $region \
        --allow-unauthenticated \
        --cpu $cpu \
        --memory $memory \
        --concurrency $concurrency \
        --timeout $timeout \
        --min-instances $min_instances \
        --max-instances $max_instances \
        --execution-environment gen2 \
        --cpu-throttling \
        --set-env-vars "REGION=$region,ENVIRONMENT=production" \
        --quiet
    
    if [ $? -eq 0 ]; then
        log_success "Deployed $service to $region"
        
        # Get service URL
        service_url=$(gcloud run services describe $service --region $region --format="value(status.url)")
        log_info "Service URL: $service_url"
        
        return 0
    else
        log_error "Failed to deploy $service to $region"
        return 1
    fi
}

# Deploy all services to all regions
deploy_all_services() {
    log_info "Starting multi-region deployment..."
    
    local total_deployments=0
    local successful_deployments=0
    
    for service in "${SERVICES[@]}"; do
        for region in "${REGIONS[@]}"; do
            total_deployments=$((total_deployments + 1))
            
            if deploy_service_to_region $service $region; then
                successful_deployments=$((successful_deployments + 1))
            fi
        done
    done
    
    log_info "Deployment Summary:"
    log_info "Total deployments: $total_deployments"
    log_info "Successful deployments: $successful_deployments"
    log_info "Failed deployments: $((total_deployments - successful_deployments))"
    
    if [ $successful_deployments -eq $total_deployments ]; then
        log_success "All deployments successful!"
    else
        log_warning "Some deployments failed. Check logs above."
    fi
}

# Setup global load balancing
setup_load_balancing() {
    log_info "Setting up global load balancing..."
    
    # Create global load balancer (simplified example)
    # In a real implementation, this would involve:
    # 1. Creating a global load balancer
    # 2. Setting up backend services for each region
    # 3. Configuring health checks
    # 4. Setting up URL maps and forwarding rules
    
    log_info "Creating load balancer configuration..."
    
    # Create a simple load balancer configuration file
    cat > load-balancer-config.yaml << EOF
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: prometheus-ai-ssl-cert
spec:
  domains:
    - api.prometheus-ai.com
    - helios.prometheus-ai.com
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: prometheus-ai-ingress
  annotations:
    kubernetes.io/ingress.global-static-ip-name: "prometheus-ai-ip"
    networking.gke.io/managed-certificates: "prometheus-ai-ssl-cert"
    kubernetes.io/ingress.class: "gce"
spec:
  rules:
  - host: api.prometheus-ai.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: api-gateway
            port:
              number: 80
  - host: helios.prometheus-ai.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: helios-core
            port:
              number: 80
EOF
    
    log_success "Load balancer configuration created"
}

# Setup monitoring and alerting
setup_monitoring() {
    log_info "Setting up monitoring and alerting..."
    
    # Create monitoring dashboard
    cat > monitoring-dashboard.json << EOF
{
  "displayName": "Prometheus AI Cloud Run Dashboard",
  "mosaicLayout": {
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Request Count",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE",
                      "crossSeriesReducer": "REDUCE_SUM"
                    }
                  }
                }
              }
            ]
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "xPos": 6,
        "widget": {
          "title": "Response Latency",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN"
                    }
                  }
                }
              }
            ]
          }
        }
      }
    ]
  }
}
EOF
    
    # Create the dashboard
    gcloud monitoring dashboards create --config-from-file=monitoring-dashboard.json
    
    log_success "Monitoring dashboard created"
}

# Run cost optimization
run_cost_optimization() {
    log_info "Running cost optimization..."
    
    # Build and run the cost optimizer
    docker build -t cost-optimizer -f services/cloud-run-config/Dockerfile.cloud-run-base services/cloud-run-config/
    
    # Run cost optimization analysis
    docker run --rm \
        -e GOOGLE_CLOUD_PROJECT=$PROJECT_ID \
        -v ~/.config/gcloud:/root/.config/gcloud \
        cost-optimizer python cost-optimization.py
    
    log_success "Cost optimization completed"
}

# Health check for deployed services
health_check() {
    log_info "Performing health checks..."
    
    for service in "${SERVICES[@]}"; do
        for region in "${REGIONS[@]}"; do
            service_url=$(gcloud run services describe $service --region $region --format="value(status.url)" 2>/dev/null)
            
            if [ -n "$service_url" ]; then
                log_info "Checking health of $service in $region..."
                
                if curl -f -s "$service_url/health" > /dev/null; then
                    log_success "$service in $region is healthy"
                else
                    log_warning "$service in $region health check failed"
                fi
            fi
        done
    done
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f load-balancer-config.yaml monitoring-dashboard.json
}

# Main deployment function
main() {
    log_info "Starting Cloud Run multi-region deployment for Prometheus AI Platform"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    enable_apis
    build_and_push_images
    deploy_all_services
    setup_load_balancing
    setup_monitoring
    run_cost_optimization
    health_check
    
    log_success "Cloud Run deployment completed successfully!"
    log_info "Services are now running across multiple regions with cost optimization enabled."
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "build")
        check_prerequisites
        build_and_push_images
        ;;
    "health-check")
        health_check
        ;;
    "cost-optimize")
        run_cost_optimization
        ;;
    "cleanup")
        cleanup
        ;;
    *)
        echo "Usage: $0 [deploy|build|health-check|cost-optimize|cleanup]"
        echo "  deploy       - Full deployment (default)"
        echo "  build        - Build and push images only"
        echo "  health-check - Check service health"
        echo "  cost-optimize - Run cost optimization"
        echo "  cleanup      - Clean up temporary files"
        exit 1
        ;;
esac