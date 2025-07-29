#!/bin/bash

# Multi-Cloud Deployment Script for Prometheus AI Platform
# Supports AWS EKS, Google GKE, and Azure AKS

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NAMESPACE="prometheus-ai"

# Default values
CLOUD_PROVIDER=""
ENVIRONMENT="production"
REGION=""
CLUSTER_NAME="prometheus-ai-cluster"
NODE_COUNT=3
MACHINE_TYPE=""
ENABLE_AUTOSCALING=true
MIN_NODES=1
MAX_NODES=10
ENABLE_MONITORING=true
ENABLE_BACKUP=true
DRY_RUN=false

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

# Help function
show_help() {
    cat << EOF
Multi-Cloud Deployment Script for Prometheus AI Platform

Usage: $0 [OPTIONS]

OPTIONS:
    -c, --cloud PROVIDER        Cloud provider (aws, gcp, azure)
    -e, --environment ENV       Environment (development, staging, production)
    -r, --region REGION         Cloud region
    -n, --cluster-name NAME     Kubernetes cluster name
    --node-count COUNT          Number of nodes (default: 3)
    --machine-type TYPE         Machine/instance type
    --min-nodes COUNT           Minimum nodes for autoscaling (default: 1)
    --max-nodes COUNT           Maximum nodes for autoscaling (default: 10)
    --no-autoscaling           Disable cluster autoscaling
    --no-monitoring            Disable monitoring setup
    --no-backup                Disable backup setup
    --dry-run                  Show what would be deployed without executing
    -h, --help                 Show this help message

EXAMPLES:
    # Deploy to AWS EKS
    $0 --cloud aws --region us-west-2 --machine-type m5.large

    # Deploy to Google GKE
    $0 --cloud gcp --region us-central1 --machine-type n1-standard-2

    # Deploy to Azure AKS
    $0 --cloud azure --region eastus --machine-type Standard_D2s_v3

    # Development deployment with minimal resources
    $0 --cloud aws --environment development --node-count 1 --no-backup

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--cloud)
                CLOUD_PROVIDER="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -n|--cluster-name)
                CLUSTER_NAME="$2"
                shift 2
                ;;
            --node-count)
                NODE_COUNT="$2"
                shift 2
                ;;
            --machine-type)
                MACHINE_TYPE="$2"
                shift 2
                ;;
            --min-nodes)
                MIN_NODES="$2"
                shift 2
                ;;
            --max-nodes)
                MAX_NODES="$2"
                shift 2
                ;;
            --no-autoscaling)
                ENABLE_AUTOSCALING=false
                shift
                ;;
            --no-monitoring)
                ENABLE_MONITORING=false
                shift
                ;;
            --no-backup)
                ENABLE_BACKUP=false
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."

    # Check required tools
    local required_tools=("kubectl" "helm")
    
    case $CLOUD_PROVIDER in
        aws)
            required_tools+=("aws" "eksctl")
            ;;
        gcp)
            required_tools+=("gcloud")
            ;;
        azure)
            required_tools+=("az")
            ;;
    esac

    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done

    # Validate cloud provider
    if [[ ! "$CLOUD_PROVIDER" =~ ^(aws|gcp|azure)$ ]]; then
        log_error "Cloud provider must be one of: aws, gcp, azure"
        exit 1
    fi

    # Set default machine types if not specified
    if [[ -z "$MACHINE_TYPE" ]]; then
        case $CLOUD_PROVIDER in
            aws)
                MACHINE_TYPE="m5.large"
                ;;
            gcp)
                MACHINE_TYPE="n1-standard-2"
                ;;
            azure)
                MACHINE_TYPE="Standard_D2s_v3"
                ;;
        esac
    fi

    # Set default regions if not specified
    if [[ -z "$REGION" ]]; then
        case $CLOUD_PROVIDER in
            aws)
                REGION="us-west-2"
                ;;
            gcp)
                REGION="us-central1"
                ;;
            azure)
                REGION="eastus"
                ;;
        esac
    fi

    log_success "Prerequisites validated"
}

# Create or update Kubernetes cluster
create_cluster() {
    log_info "Creating/updating Kubernetes cluster: $CLUSTER_NAME"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create cluster with:"
        log_info "  Cloud: $CLOUD_PROVIDER"
        log_info "  Region: $REGION"
        log_info "  Machine Type: $MACHINE_TYPE"
        log_info "  Node Count: $NODE_COUNT"
        log_info "  Autoscaling: $ENABLE_AUTOSCALING ($MIN_NODES-$MAX_NODES)"
        return
    fi

    case $CLOUD_PROVIDER in
        aws)
            create_eks_cluster
            ;;
        gcp)
            create_gke_cluster
            ;;
        azure)
            create_aks_cluster
            ;;
    esac

    log_success "Cluster created/updated successfully"
}

# Create AWS EKS cluster
create_eks_cluster() {
    log_info "Creating AWS EKS cluster..."

    # Check if cluster exists
    if aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" &>/dev/null; then
        log_info "EKS cluster $CLUSTER_NAME already exists, updating..."
        # Update node group if needed
        eksctl scale nodegroup --cluster="$CLUSTER_NAME" --region="$REGION" \
            --nodes="$NODE_COUNT" --nodes-min="$MIN_NODES" --nodes-max="$MAX_NODES" \
            --name="prometheus-ai-nodes" || true
    else
        log_info "Creating new EKS cluster..."
        
        # Create cluster configuration
        cat > /tmp/eks-cluster.yaml << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: $CLUSTER_NAME
  region: $REGION
  version: "1.28"

iam:
  withOIDC: true

nodeGroups:
  - name: prometheus-ai-nodes
    instanceType: $MACHINE_TYPE
    desiredCapacity: $NODE_COUNT
    minSize: $MIN_NODES
    maxSize: $MAX_NODES
    volumeSize: 100
    volumeType: gp3
    amiFamily: AmazonLinux2
    ssh:
      allow: false
    iam:
      attachPolicyARNs:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
        - arn:aws:iam::aws:policy/AmazonS3FullAccess
    labels:
      environment: $ENVIRONMENT
      application: prometheus-ai
    tags:
      Environment: $ENVIRONMENT
      Application: prometheus-ai
      ManagedBy: eksctl

managedNodeGroups:
  - name: prometheus-ai-spot
    instanceTypes:
      - m5.large
      - m5.xlarge
      - c5.large
      - c5.xlarge
    spot: true
    desiredCapacity: 2
    minSize: 0
    maxSize: 5
    volumeSize: 50
    volumeType: gp3
    labels:
      environment: $ENVIRONMENT
      application: prometheus-ai
      node-type: spot
    tags:
      Environment: $ENVIRONMENT
      Application: prometheus-ai
      NodeType: spot

addons:
  - name: vpc-cni
    version: latest
  - name: coredns
    version: latest
  - name: kube-proxy
    version: latest
  - name: aws-ebs-csi-driver
    version: latest
EOF

        eksctl create cluster -f /tmp/eks-cluster.yaml
        rm /tmp/eks-cluster.yaml
    fi

    # Update kubeconfig
    aws eks update-kubeconfig --region "$REGION" --name "$CLUSTER_NAME"
}

# Create Google GKE cluster
create_gke_cluster() {
    log_info "Creating Google GKE cluster..."

    # Check if cluster exists
    if gcloud container clusters describe "$CLUSTER_NAME" --region="$REGION" &>/dev/null; then
        log_info "GKE cluster $CLUSTER_NAME already exists, updating..."
        # Update cluster if needed
        gcloud container clusters resize "$CLUSTER_NAME" --region="$REGION" \
            --num-nodes="$NODE_COUNT" --quiet || true
    else
        log_info "Creating new GKE cluster..."
        
        local autoscaling_flags=""
        if [[ "$ENABLE_AUTOSCALING" == "true" ]]; then
            autoscaling_flags="--enable-autoscaling --min-nodes=$MIN_NODES --max-nodes=$MAX_NODES"
        fi

        gcloud container clusters create "$CLUSTER_NAME" \
            --region="$REGION" \
            --machine-type="$MACHINE_TYPE" \
            --num-nodes="$NODE_COUNT" \
            $autoscaling_flags \
            --disk-size=100GB \
            --disk-type=pd-ssd \
            --enable-network-policy \
            --enable-ip-alias \
            --enable-autorepair \
            --enable-autoupgrade \
            --maintenance-window-start="2023-01-01T09:00:00Z" \
            --maintenance-window-end="2023-01-01T17:00:00Z" \
            --maintenance-window-recurrence="FREQ=WEEKLY;BYDAY=SA" \
            --labels="environment=$ENVIRONMENT,application=prometheus-ai" \
            --addons=HorizontalPodAutoscaling,HttpLoadBalancing,NetworkPolicy \
            --enable-shielded-nodes \
            --enable-autorepair \
            --enable-autoupgrade \
            --workload-pool="$(gcloud config get-value project).svc.id.goog"

        # Create spot instance node pool
        gcloud container node-pools create "prometheus-ai-spot" \
            --cluster="$CLUSTER_NAME" \
            --region="$REGION" \
            --machine-type="$MACHINE_TYPE" \
            --num-nodes=1 \
            --enable-autoscaling \
            --min-nodes=0 \
            --max-nodes=5 \
            --spot \
            --disk-size=50GB \
            --disk-type=pd-ssd \
            --node-labels="environment=$ENVIRONMENT,application=prometheus-ai,node-type=spot"
    fi

    # Get credentials
    gcloud container clusters get-credentials "$CLUSTER_NAME" --region="$REGION"
}

# Create Azure AKS cluster
create_aks_cluster() {
    log_info "Creating Azure AKS cluster..."

    local resource_group="prometheus-ai-$ENVIRONMENT"
    
    # Create resource group if it doesn't exist
    az group create --name "$resource_group" --location "$REGION" || true

    # Check if cluster exists
    if az aks show --name "$CLUSTER_NAME" --resource-group "$resource_group" &>/dev/null; then
        log_info "AKS cluster $CLUSTER_NAME already exists, updating..."
        # Scale cluster if needed
        az aks scale --name "$CLUSTER_NAME" --resource-group "$resource_group" \
            --node-count "$NODE_COUNT" || true
    else
        log_info "Creating new AKS cluster..."
        
        local autoscaling_flags=""
        if [[ "$ENABLE_AUTOSCALING" == "true" ]]; then
            autoscaling_flags="--enable-cluster-autoscaler --min-count=$MIN_NODES --max-count=$MAX_NODES"
        fi

        az aks create \
            --name "$CLUSTER_NAME" \
            --resource-group "$resource_group" \
            --location "$REGION" \
            --node-count "$NODE_COUNT" \
            --node-vm-size "$MACHINE_TYPE" \
            $autoscaling_flags \
            --node-osdisk-size 100 \
            --node-osdisk-type Managed \
            --network-plugin azure \
            --network-policy azure \
            --enable-addons monitoring,azure-policy \
            --enable-managed-identity \
            --generate-ssh-keys \
            --tags "Environment=$ENVIRONMENT" "Application=prometheus-ai"

        # Add spot node pool
        az aks nodepool add \
            --cluster-name "$CLUSTER_NAME" \
            --resource-group "$resource_group" \
            --name "spotpool" \
            --node-count 1 \
            --node-vm-size "$MACHINE_TYPE" \
            --enable-cluster-autoscaler \
            --min-count 0 \
            --max-count 5 \
            --priority Spot \
            --eviction-policy Delete \
            --spot-max-price -1 \
            --node-osdisk-size 50 \
            --node-taints "kubernetes.azure.com/scalesetpriority=spot:NoSchedule" \
            --labels "environment=$ENVIRONMENT" "application=prometheus-ai" "node-type=spot"
    fi

    # Get credentials
    az aks get-credentials --name "$CLUSTER_NAME" --resource-group "$resource_group"
}

# Deploy Kubernetes manifests
deploy_manifests() {
    log_info "Deploying Kubernetes manifests..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy the following manifests:"
        find "$PROJECT_ROOT/kubernetes" -name "*.yaml" | sort
        return
    fi

    # Create namespace
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    # Apply manifests in order
    local manifests=(
        "namespace.yaml"
        "secrets.yaml"
        "configmap.yaml"
        "multi-cloud-config.yaml"
        "persistent-volumes.yaml"
        "database-deployments.yaml"
        "api-gateway-deployment.yaml"
    )

    # Add optional manifests based on configuration
    if [[ "$ENABLE_AUTOSCALING" == "true" ]]; then
        manifests+=("autoscaling.yaml")
    fi

    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        manifests+=("monitoring.yaml")
    fi

    if [[ "$ENABLE_BACKUP" == "true" ]]; then
        manifests+=("disaster-recovery.yaml")
    fi

    # Apply each manifest
    for manifest in "${manifests[@]}"; do
        local manifest_path="$PROJECT_ROOT/kubernetes/$manifest"
        if [[ -f "$manifest_path" ]]; then
            log_info "Applying $manifest..."
            kubectl apply -f "$manifest_path"
        else
            log_warning "Manifest $manifest not found, skipping..."
        fi
    done

    log_success "Kubernetes manifests deployed successfully"
}

# Wait for deployments to be ready
wait_for_deployments() {
    log_info "Waiting for deployments to be ready..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would wait for deployments to be ready"
        return
    fi

    local deployments=(
        "postgres"
        "influxdb"
        "neo4j"
        "redis"
        "api-gateway"
    )

    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        deployments+=("prometheus" "grafana")
    fi

    for deployment in "${deployments[@]}"; do
        log_info "Waiting for deployment $deployment..."
        kubectl wait --for=condition=available --timeout=600s deployment/"$deployment" -n "$NAMESPACE" || {
            log_error "Deployment $deployment failed to become ready"
            kubectl describe deployment "$deployment" -n "$NAMESPACE"
            exit 1
        }
    done

    log_success "All deployments are ready"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would verify deployment"
        return
    fi

    # Check pod status
    log_info "Pod status:"
    kubectl get pods -n "$NAMESPACE"

    # Check service status
    log_info "Service status:"
    kubectl get services -n "$NAMESPACE"

    # Check ingress/load balancer
    log_info "External access:"
    kubectl get ingress,service -n "$NAMESPACE" -o wide

    # Run basic health checks
    log_info "Running health checks..."
    
    # Check API Gateway health
    local api_gateway_ip
    api_gateway_ip=$(kubectl get service api-gateway-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [[ -n "$api_gateway_ip" ]]; then
        if curl -f "http://$api_gateway_ip/health" &>/dev/null; then
            log_success "API Gateway health check passed"
        else
            log_warning "API Gateway health check failed"
        fi
    else
        log_info "API Gateway external IP not yet available"
    fi

    log_success "Deployment verification completed"
}

# Cleanup function
cleanup() {
    if [[ -f "/tmp/eks-cluster.yaml" ]]; then
        rm -f /tmp/eks-cluster.yaml
    fi
}

# Main execution
main() {
    trap cleanup EXIT

    log_info "Starting Prometheus AI Platform deployment..."
    log_info "Cloud Provider: $CLOUD_PROVIDER"
    log_info "Environment: $ENVIRONMENT"
    log_info "Region: $REGION"
    log_info "Cluster: $CLUSTER_NAME"

    parse_args "$@"
    validate_prerequisites
    create_cluster
    deploy_manifests
    wait_for_deployments
    verify_deployment

    log_success "Deployment completed successfully!"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        log_info "Access your deployment:"
        log_info "  kubectl get services -n $NAMESPACE"
        log_info "  kubectl port-forward service/api-gateway-service 8080:80 -n $NAMESPACE"
        
        if [[ "$ENABLE_MONITORING" == "true" ]]; then
            log_info "  kubectl port-forward service/grafana 3000:3000 -n $NAMESPACE"
            log_info "  kubectl port-forward service/prometheus-service 9090:9090 -n $NAMESPACE"
        fi
    fi
}

# Run main function with all arguments
main "$@"