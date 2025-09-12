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
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
  
  backend "gcs" {
    bucket = "prometheus-ai-terraform-state"
    prefix = "infrastructure"
  }
}

# Local variables
locals {
  project_name = "prometheus-ai-platform"
  environment  = var.environment
  
  common_tags = {
    Project     = local.project_name
    Environment = local.environment
    ManagedBy   = "terraform"
    Owner       = "prometheus-ai-team"
  }
  
  # Multi-cloud configuration
  gcp_enabled   = var.enable_gcp
  azure_enabled = var.enable_azure
  aws_enabled   = var.enable_aws
}

# Google Cloud Provider
provider "google" {
  count   = local.gcp_enabled ? 1 : 0
  project = var.gcp_project_id
  region  = var.gcp_region
}

provider "google-beta" {
  count   = local.gcp_enabled ? 1 : 0
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Azure Provider
provider "azurerm" {
  count = local.azure_enabled ? 1 : 0
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
  subscription_id = var.azure_subscription_id
}

# AWS Provider
provider "aws" {
  count  = local.aws_enabled ? 1 : 0
  region = var.aws_region
  
  default_tags {
    tags = local.common_tags
  }
}

# Data sources for existing infrastructure
data "google_client_config" "default" {
  count = local.gcp_enabled ? 1 : 0
}

data "azurerm_client_config" "current" {
  count = local.azure_enabled ? 1 : 0
}

data "aws_caller_identity" "current" {
  count = local.aws_enabled ? 1 : 0
}

# GCP Infrastructure
module "gcp_infrastructure" {
  count  = local.gcp_enabled ? 1 : 0
  source = "./modules/gcp"
  
  project_id   = var.gcp_project_id
  region       = var.gcp_region
  zone         = var.gcp_zone
  environment  = local.environment
  project_name = local.project_name
  
  # GKE Configuration
  gke_cluster_name    = "${local.project_name}-${local.environment}"
  gke_node_count      = var.gcp_node_count
  gke_machine_type    = var.gcp_machine_type
  gke_disk_size       = var.gcp_disk_size
  gke_enable_gpu      = var.gcp_enable_gpu
  gke_gpu_type        = var.gcp_gpu_type
  gke_gpu_count       = var.gcp_gpu_count
  
  # Network Configuration
  vpc_name           = "${local.project_name}-vpc"
  subnet_name        = "${local.project_name}-subnet"
  subnet_cidr        = "10.0.0.0/16"
  pod_cidr           = "10.1.0.0/16"
  service_cidr       = "10.2.0.0/16"
  
  # Security Configuration
  enable_network_policy     = true
  enable_private_nodes      = true
  enable_authorized_networks = true
  authorized_networks = [
    {
      cidr_block   = "0.0.0.0/0"
      display_name = "All"
    }
  ]
  
  tags = local.common_tags
}

# Azure Infrastructure
module "azure_infrastructure" {
  count  = local.azure_enabled ? 1 : 0
  source = "./modules/azure"
  
  resource_group_name = "${local.project_name}-${local.environment}-rg"
  location           = var.azure_location
  environment        = local.environment
  project_name       = local.project_name
  
  # AKS Configuration
  aks_cluster_name     = "${local.project_name}-${local.environment}"
  aks_node_count       = var.azure_node_count
  aks_vm_size          = var.azure_vm_size
  aks_disk_size        = var.azure_disk_size
  aks_enable_gpu       = var.azure_enable_gpu
  
  # Network Configuration
  vnet_name          = "${local.project_name}-vnet"
  vnet_address_space = ["10.10.0.0/16"]
  subnet_name        = "${local.project_name}-subnet"
  subnet_address_prefix = "10.10.1.0/24"
  
  # Security Configuration
  enable_network_policy = true
  enable_rbac          = true
  
  tags = local.common_tags
}

# AWS Infrastructure
module "aws_infrastructure" {
  count  = local.aws_enabled ? 1 : 0
  source = "./modules/aws"
  
  region       = var.aws_region
  environment  = local.environment
  project_name = local.project_name
  
  # EKS Configuration
  eks_cluster_name    = "${local.project_name}-${local.environment}"
  eks_node_count      = var.aws_node_count
  eks_instance_type   = var.aws_instance_type
  eks_disk_size       = var.aws_disk_size
  eks_enable_gpu      = var.aws_enable_gpu
  
  # Network Configuration
  vpc_name           = "${local.project_name}-vpc"
  vpc_cidr           = "10.20.0.0/16"
  availability_zones = var.aws_availability_zones
  
  # Security Configuration
  enable_network_policy = true
  enable_irsa          = true
  
  tags = local.common_tags
}

# Kubernetes provider configuration
provider "kubernetes" {
  alias = "gcp"
  count = local.gcp_enabled ? 1 : 0
  
  host                   = local.gcp_enabled ? module.gcp_infrastructure[0].cluster_endpoint : ""
  token                  = local.gcp_enabled ? data.google_client_config.default[0].access_token : ""
  cluster_ca_certificate = local.gcp_enabled ? base64decode(module.gcp_infrastructure[0].cluster_ca_certificate) : ""
}

provider "kubernetes" {
  alias = "azure"
  count = local.azure_enabled ? 1 : 0
  
  host                   = local.azure_enabled ? module.azure_infrastructure[0].cluster_endpoint : ""
  client_certificate     = local.azure_enabled ? base64decode(module.azure_infrastructure[0].client_certificate) : ""
  client_key             = local.azure_enabled ? base64decode(module.azure_infrastructure[0].client_key) : ""
  cluster_ca_certificate = local.azure_enabled ? base64decode(module.azure_infrastructure[0].cluster_ca_certificate) : ""
}

provider "kubernetes" {
  alias = "aws"
  count = local.aws_enabled ? 1 : 0
  
  host                   = local.aws_enabled ? module.aws_infrastructure[0].cluster_endpoint : ""
  cluster_ca_certificate = local.aws_enabled ? base64decode(module.aws_infrastructure[0].cluster_ca_certificate) : ""
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", local.aws_enabled ? module.aws_infrastructure[0].cluster_name : ""]
  }
}

# Helm provider configuration
provider "helm" {
  alias = "gcp"
  count = local.gcp_enabled ? 1 : 0
  
  kubernetes {
    host                   = local.gcp_enabled ? module.gcp_infrastructure[0].cluster_endpoint : ""
    token                  = local.gcp_enabled ? data.google_client_config.default[0].access_token : ""
    cluster_ca_certificate = local.gcp_enabled ? base64decode(module.gcp_infrastructure[0].cluster_ca_certificate) : ""
  }
}

provider "helm" {
  alias = "azure"
  count = local.azure_enabled ? 1 : 0
  
  kubernetes {
    host                   = local.azure_enabled ? module.azure_infrastructure[0].cluster_endpoint : ""
    client_certificate     = local.azure_enabled ? base64decode(module.azure_infrastructure[0].client_certificate) : ""
    client_key             = local.azure_enabled ? base64decode(module.azure_infrastructure[0].client_key) : ""
    cluster_ca_certificate = local.azure_enabled ? base64decode(module.azure_infrastructure[0].cluster_ca_certificate) : ""
  }
}

provider "helm" {
  alias = "aws"
  count = local.aws_enabled ? 1 : 0
  
  kubernetes {
    host                   = local.aws_enabled ? module.aws_infrastructure[0].cluster_endpoint : ""
    cluster_ca_certificate = local.aws_enabled ? base64decode(module.aws_infrastructure[0].cluster_ca_certificate) : ""
    
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", local.aws_enabled ? module.aws_infrastructure[0].cluster_name : ""]
    }
  }
}

# Monitoring and Observability
module "monitoring" {
  source = "./modules/monitoring"
  
  # Deploy to all enabled clouds
  gcp_enabled   = local.gcp_enabled
  azure_enabled = local.azure_enabled
  aws_enabled   = local.aws_enabled
  
  environment  = local.environment
  project_name = local.project_name
  
  # Prometheus configuration
  prometheus_retention = "30d"
  prometheus_storage   = "100Gi"
  
  # Grafana configuration
  grafana_admin_password = var.grafana_admin_password
  
  # Jaeger configuration
  jaeger_storage_type = "elasticsearch"
  
  depends_on = [
    module.gcp_infrastructure,
    module.azure_infrastructure,
    module.aws_infrastructure
  ]
}

# Security and Compliance
module "security" {
  source = "./modules/security"
  
  # Deploy to all enabled clouds
  gcp_enabled   = local.gcp_enabled
  azure_enabled = local.azure_enabled
  aws_enabled   = local.aws_enabled
  
  environment  = local.environment
  project_name = local.project_name
  
  # Security configuration
  enable_pod_security_policy = true
  enable_network_policy      = true
  enable_rbac               = true
  
  # Compliance configuration
  enable_falco              = true
  enable_opa_gatekeeper     = true
  
  depends_on = [
    module.gcp_infrastructure,
    module.azure_infrastructure,
    module.aws_infrastructure
  ]
}

# Backup and Disaster Recovery
module "backup" {
  source = "./modules/backup"
  
  # Deploy to all enabled clouds
  gcp_enabled   = local.gcp_enabled
  azure_enabled = local.azure_enabled
  aws_enabled   = local.aws_enabled
  
  environment  = local.environment
  project_name = local.project_name
  
  # Backup configuration
  backup_schedule    = "0 2 * * *"  # Daily at 2 AM
  backup_retention   = 30           # 30 days
  
  # Cross-region replication
  enable_cross_region_backup = true
  
  depends_on = [
    module.gcp_infrastructure,
    module.azure_infrastructure,
    module.aws_infrastructure
  ]
}