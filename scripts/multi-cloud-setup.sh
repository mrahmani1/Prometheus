#!/bin/bash

# Multi-Cloud Setup Script for Prometheus AI Platform
# Configures GCP, Azure, and AWS environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking multi-cloud prerequisites..."
    
    # Check required tools
    command -v gcloud >/dev/null 2>&1 || error "gcloud CLI is required but not installed"
    command -v az >/dev/null 2>&1 || error "Azure CLI is required but not installed"
    command -v aws >/dev/null 2>&1 || error "AWS CLI is required but not installed"
    command -v kubectl >/dev/null 2>&1 || error "kubectl is required but not installed"
    command -v helm >/dev/null 2>&1 || error "helm is required but not installed"
    command -v terraform >/dev/null 2>&1 || error "terraform is required but not installed"
    
    success "Prerequisites check passed"
}

# Setup Google Cloud Platform
setup_gcp() {
    log "Setting up Google Cloud Platform..."
    
    # Authenticate with service account
    if [[ -n "${GCP_SERVICE_ACCOUNT_KEY:-}" ]]; then
        echo "${GCP_SERVICE_ACCOUNT_KEY}" | base64 -d > /tmp/gcp-key.json
        gcloud auth activate-service-account --key-file /tmp/gcp-key.json
        rm /tmp/gcp-key.json
    fi
    
    # Set project
    gcloud config set project "${GCP_PROJECT_ID}"
    
    # Enable required APIs
    log "Enabling GCP APIs..."
    gcloud services enable container.googleapis.com
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable cloudresourcemanager.googleapis.com
    gcloud services enable compute.googleapis.com
    gcloud services enable iam.googleapis.com
    gcloud services enable monitoring.googleapis.com
    gcloud services enable logging.googleapis.com
    
    # Create GKE cluster if it doesn't exist
    if ! gcloud container clusters describe prometheus-production --zone=us-central1-a >/dev/null 2>&1; then
        log "Creating GKE cluster..."
        gcloud container clusters create prometheus-production \
            --zone=us-central1-a \
            --num-nodes=3 \
            --machine-type=e2-standard-4 \
            --enable-autoscaling \
            --min-nodes=1 \
            --max-nodes=10 \
            --enable-autorepair \
            --enable-autoupgrade \
            --enable-network-policy \
            --enable-ip-alias \
            --addons=HorizontalPodAutoscaling,HttpLoadBalancing,NetworkPolicy
    fi
    
    # Get cluster credentials
    gcloud container clusters get-credentials prometheus-production --zone=us-central1-a
    
    success "GCP setup completed"
}

# Setup Microsoft Azure
setup_azure() {
    log "Setting up Microsoft Azure..."
    
    # Login with service principal
    if [[ -n "${AZURE_CLIENT_ID:-}" && -n "${AZURE_CLIENT_SECRET:-}" && -n "${AZURE_TENANT_ID:-}" ]]; then
        az login --service-principal \
            -u "${AZURE_CLIENT_ID}" \
            -p "${AZURE_CLIENT_SECRET}" \
            --tenant "${AZURE_TENANT_ID}"
    fi
    
    # Set subscription
    az account set --subscription "${AZURE_SUBSCRIPTION_ID}"
    
    # Create resource group
    if ! az group show --name prometheus-ai-rg >/dev/null 2>&1; then
        log "Creating Azure resource group..."
        az group create --name prometheus-ai-rg --location eastus
    fi
    
    # Create AKS cluster if it doesn't exist
    if ! az aks show --resource-group prometheus-ai-rg --name prometheus-production >/dev/null 2>&1; then
        log "Creating AKS cluster..."
        az aks create \
            --resource-group prometheus-ai-rg \
            --name prometheus-production \
            --node-count 3 \
            --node-vm-size Standard_D4s_v3 \
            --enable-cluster-autoscaler \
            --min-count 1 \
            --max-count 10 \
            --enable-addons monitoring \
            --generate-ssh-keys \
            --network-plugin azure \
            --network-policy azure
    fi
    
    # Get cluster credentials
    az aks get-credentials --resource-group prometheus-ai-rg --name prometheus-production
    
    success "Azure setup completed"
}

# Setup Amazon Web Services
setup_aws() {
    log "Setting up Amazon Web Services..."
    
    # Configure AWS credentials
    if [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
        aws configure set aws_access_key_id "${AWS_ACCESS_KEY_ID}"
        aws configure set aws_secret_access_key "${AWS_SECRET_ACCESS_KEY}"
        aws configure set region us-west-2
    fi
    
    # Create EKS cluster if it doesn't exist
    if ! aws eks describe-cluster --name prometheus-production >/dev/null 2>&1; then
        log "Creating EKS cluster..."
        
        # Create cluster role
        cat > /tmp/cluster-role-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "eks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
        
        aws iam create-role \
            --role-name prometheus-cluster-role \
            --assume-role-policy-document file:///tmp/cluster-role-trust-policy.json || true
        
        aws iam attach-role-policy \
            --role-name prometheus-cluster-role \
            --policy-arn arn:aws:iam::aws:policy/AmazonEKSClusterPolicy || true
        
        # Create node group role
        cat > /tmp/node-role-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
        
        aws iam create-role \
            --role-name prometheus-node-role \
            --assume-role-policy-document file:///tmp/node-role-trust-policy.json || true
        
        aws iam attach-role-policy \
            --role-name prometheus-node-role \
            --policy-arn arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy || true
        
        aws iam attach-role-policy \
            --role-name prometheus-node-role \
            --policy-arn arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy || true
        
        aws iam attach-role-policy \
            --role-name prometheus-node-role \
            --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly || true
        
        # Get default VPC and subnets
        VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text)
        SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=${VPC_ID}" --query 'Subnets[*].SubnetId' --output text | tr '\t' ',')
        
        # Create EKS cluster
        aws eks create-cluster \
            --name prometheus-production \
            --version 1.28 \
            --role-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/prometheus-cluster-role" \
            --resources-vpc-config subnetIds="${SUBNET_IDS}"
        
        # Wait for cluster to be active
        aws eks wait cluster-active --name prometheus-production
        
        # Create node group
        aws eks create-nodegroup \
            --cluster-name prometheus-production \
            --nodegroup-name prometheus-nodes \
            --subnets $(echo "${SUBNET_IDS}" | tr ',' ' ') \
            --node-role "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/prometheus-node-role" \
            --instance-types t3.large \
            --scaling-config minSize=1,maxSize=10,desiredSize=3
        
        # Wait for node group to be active
        aws eks wait nodegroup-active --cluster-name prometheus-production --nodegroup-name prometheus-nodes
    fi
    
    # Update kubeconfig
    aws eks update-kubeconfig --name prometheus-production --region us-west-2
    
    success "AWS setup completed"
}

# Setup Terraform infrastructure
setup_terraform() {
    log "Setting up Terraform infrastructure..."
    
    # Create Terraform configuration directory
    mkdir -p terraform/multi-cloud
    
    # Create main Terraform configuration
    cat > terraform/multi-cloud/main.tf <<EOF
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "gcs" {
    bucket = "prometheus-ai-terraform-state"
    prefix = "multi-cloud"
  }
}

# Google Cloud Provider
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Azure Provider
provider "azurerm" {
  features {}
  subscription_id = var.azure_subscription_id
}

# AWS Provider
provider "aws" {
  region = var.aws_region
}

# Variables
variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "azure_subscription_id" {
  description = "Azure Subscription ID"
  type        = string
}

variable "aws_region" {
  description = "AWS Region"
  type        = string
  default     = "us-west-2"
}

# GCP Resources
resource "google_container_cluster" "prometheus_gcp" {
  name     = "prometheus-production-gcp"
  location = var.gcp_region
  
  initial_node_count = 3
  
  node_config {
    machine_type = "e2-standard-4"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
  
  network_policy {
    enabled = true
  }
}

# Azure Resources
resource "azurerm_resource_group" "prometheus_azure" {
  name     = "prometheus-ai-rg"
  location = "East US"
}

resource "azurerm_kubernetes_cluster" "prometheus_azure" {
  name                = "prometheus-production-azure"
  location            = azurerm_resource_group.prometheus_azure.location
  resource_group_name = azurerm_resource_group.prometheus_azure.name
  dns_prefix          = "prometheus-azure"
  
  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D4s_v3"
  }
  
  identity {
    type = "SystemAssigned"
  }
  
  network_profile {
    network_plugin = "azure"
    network_policy = "azure"
  }
}

# AWS Resources
resource "aws_eks_cluster" "prometheus_aws" {
  name     = "prometheus-production-aws"
  role_arn = aws_iam_role.prometheus_cluster_role.arn
  version  = "1.28"
  
  vpc_config {
    subnet_ids = data.aws_subnets.default.ids
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.prometheus_cluster_policy,
  ]
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_iam_role" "prometheus_cluster_role" {
  name = "prometheus-cluster-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "prometheus_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.prometheus_cluster_role.name
}

# Outputs
output "gcp_cluster_endpoint" {
  value = google_container_cluster.prometheus_gcp.endpoint
}

output "azure_cluster_fqdn" {
  value = azurerm_kubernetes_cluster.prometheus_azure.fqdn
}

output "aws_cluster_endpoint" {
  value = aws_eks_cluster.prometheus_aws.endpoint
}
EOF
    
    # Create variables file
    cat > terraform/multi-cloud/terraform.tfvars <<EOF
gcp_project_id        = "${GCP_PROJECT_ID}"
gcp_region           = "us-central1"
azure_subscription_id = "${AZURE_SUBSCRIPTION_ID}"
aws_region           = "us-west-2"
EOF
    
    # Initialize and apply Terraform
    cd terraform/multi-cloud
    terraform init
    terraform plan
    terraform apply -auto-approve
    cd ../..
    
    success "Terraform infrastructure setup completed"
}

# Setup monitoring and observability
setup_monitoring() {
    log "Setting up multi-cloud monitoring..."
    
    # Install Prometheus Operator
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    helm upgrade --install prometheus-operator prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
        --set grafana.adminPassword=prometheus123
    
    # Install Jaeger for distributed tracing
    kubectl create namespace observability || true
    kubectl apply -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.49.0/jaeger-operator.yaml -n observability
    
    # Create Jaeger instance
    cat > /tmp/jaeger.yaml <<EOF
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: prometheus-jaeger
  namespace: observability
spec:
  strategy: production
  storage:
    type: elasticsearch
    elasticsearch:
      nodeCount: 3
      storage:
        size: 50Gi
EOF
    
    kubectl apply -f /tmp/jaeger.yaml
    
    success "Monitoring setup completed"
}

# Main setup function
main() {
    log "Starting multi-cloud setup for Prometheus AI Platform..."
    
    check_prerequisites
    
    # Setup cloud providers
    setup_gcp
    setup_azure
    setup_aws
    
    # Setup infrastructure as code
    setup_terraform
    
    # Setup monitoring
    setup_monitoring
    
    success "Multi-cloud setup completed successfully!"
    log "Clusters created:"
    log "  - GCP: prometheus-production (us-central1-a)"
    log "  - Azure: prometheus-production (eastus)"
    log "  - AWS: prometheus-production (us-west-2)"
}

# Trap errors
trap 'error "Multi-cloud setup failed"' ERR

# Run main function
main "$@"