# Environment Configuration
variable "environment" {
  description = "Environment name (staging, production)"
  type        = string
  default     = "staging"
  
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be either 'staging' or 'production'."
  }
}

# Multi-cloud enablement
variable "enable_gcp" {
  description = "Enable Google Cloud Platform deployment"
  type        = bool
  default     = true
}

variable "enable_azure" {
  description = "Enable Microsoft Azure deployment"
  type        = bool
  default     = false
}

variable "enable_aws" {
  description = "Enable Amazon Web Services deployment"
  type        = bool
  default     = false
}

# Google Cloud Platform Variables
variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
  default     = ""
}

variable "gcp_region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "gcp_zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "gcp_node_count" {
  description = "Number of nodes in GKE cluster"
  type        = number
  default     = 3
  
  validation {
    condition     = var.gcp_node_count >= 1 && var.gcp_node_count <= 100
    error_message = "Node count must be between 1 and 100."
  }
}

variable "gcp_machine_type" {
  description = "GKE node machine type"
  type        = string
  default     = "e2-standard-4"
}

variable "gcp_disk_size" {
  description = "GKE node disk size in GB"
  type        = number
  default     = 100
}

variable "gcp_enable_gpu" {
  description = "Enable GPU nodes in GKE cluster"
  type        = bool
  default     = true
}

variable "gcp_gpu_type" {
  description = "GPU type for GKE nodes"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gcp_gpu_count" {
  description = "Number of GPUs per node"
  type        = number
  default     = 1
}

# Microsoft Azure Variables
variable "azure_subscription_id" {
  description = "Azure Subscription ID"
  type        = string
  default     = ""
}

variable "azure_location" {
  description = "Azure Location"
  type        = string
  default     = "East US"
}

variable "azure_node_count" {
  description = "Number of nodes in AKS cluster"
  type        = number
  default     = 3
  
  validation {
    condition     = var.azure_node_count >= 1 && var.azure_node_count <= 100
    error_message = "Node count must be between 1 and 100."
  }
}

variable "azure_vm_size" {
  description = "AKS node VM size"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "azure_disk_size" {
  description = "AKS node disk size in GB"
  type        = number
  default     = 100
}

variable "azure_enable_gpu" {
  description = "Enable GPU nodes in AKS cluster"
  type        = bool
  default     = true
}

# Amazon Web Services Variables
variable "aws_region" {
  description = "AWS Region"
  type        = string
  default     = "us-west-2"
}

variable "aws_availability_zones" {
  description = "AWS Availability Zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "aws_node_count" {
  description = "Number of nodes in EKS cluster"
  type        = number
  default     = 3
  
  validation {
    condition     = var.aws_node_count >= 1 && var.aws_node_count <= 100
    error_message = "Node count must be between 1 and 100."
  }
}

variable "aws_instance_type" {
  description = "EKS node instance type"
  type        = string
  default     = "m5.xlarge"
}

variable "aws_disk_size" {
  description = "EKS node disk size in GB"
  type        = number
  default     = 100
}

variable "aws_enable_gpu" {
  description = "Enable GPU nodes in EKS cluster"
  type        = bool
  default     = true
}

# Application Configuration
variable "application_image_tag" {
  description = "Docker image tag for application deployment"
  type        = string
  default     = "latest"
}

variable "application_replicas" {
  description = "Number of application replicas"
  type        = number
  default     = 3
}

variable "enable_autoscaling" {
  description = "Enable horizontal pod autoscaling"
  type        = bool
  default     = true
}

variable "min_replicas" {
  description = "Minimum number of replicas for autoscaling"
  type        = number
  default     = 2
}

variable "max_replicas" {
  description = "Maximum number of replicas for autoscaling"
  type        = number
  default     = 10
}

# Database Configuration
variable "enable_postgresql" {
  description = "Enable PostgreSQL deployment"
  type        = bool
  default     = true
}

variable "postgresql_storage_size" {
  description = "PostgreSQL storage size"
  type        = string
  default     = "100Gi"
}

variable "enable_redis" {
  description = "Enable Redis deployment"
  type        = bool
  default     = true
}

variable "redis_storage_size" {
  description = "Redis storage size"
  type        = string
  default     = "20Gi"
}

variable "enable_influxdb" {
  description = "Enable InfluxDB deployment"
  type        = bool
  default     = true
}

variable "influxdb_storage_size" {
  description = "InfluxDB storage size"
  type        = string
  default     = "200Gi"
}

variable "enable_neo4j" {
  description = "Enable Neo4j deployment"
  type        = bool
  default     = true
}

variable "neo4j_storage_size" {
  description = "Neo4j storage size"
  type        = string
  default     = "50Gi"
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable monitoring stack (Prometheus, Grafana, Jaeger)"
  type        = bool
  default     = true
}

variable "prometheus_retention" {
  description = "Prometheus data retention period"
  type        = string
  default     = "30d"
}

variable "prometheus_storage_size" {
  description = "Prometheus storage size"
  type        = string
  default     = "100Gi"
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = "prometheus123"
  sensitive   = true
}

# Security Configuration
variable "enable_network_policy" {
  description = "Enable Kubernetes network policies"
  type        = bool
  default     = true
}

variable "enable_pod_security_policy" {
  description = "Enable pod security policies"
  type        = bool
  default     = true
}

variable "enable_rbac" {
  description = "Enable role-based access control"
  type        = bool
  default     = true
}

# Backup Configuration
variable "enable_backup" {
  description = "Enable backup and disaster recovery"
  type        = bool
  default     = true
}

variable "backup_schedule" {
  description = "Backup schedule in cron format"
  type        = string
  default     = "0 2 * * *"  # Daily at 2 AM
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

# Feature Flags Configuration
variable "enable_feature_flags" {
  description = "Enable feature flags service"
  type        = bool
  default     = true
}

variable "feature_flags_provider" {
  description = "Feature flags provider (launchdarkly, configcat, split)"
  type        = string
  default     = "launchdarkly"
  
  validation {
    condition     = contains(["launchdarkly", "configcat", "split"], var.feature_flags_provider)
    error_message = "Feature flags provider must be one of: launchdarkly, configcat, split."
  }
}

# SSL/TLS Configuration
variable "enable_tls" {
  description = "Enable TLS/SSL certificates"
  type        = bool
  default     = true
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "prometheus-ai.example.com"
}

variable "tls_issuer" {
  description = "TLS certificate issuer (letsencrypt-staging, letsencrypt-prod)"
  type        = string
  default     = "letsencrypt-prod"
}

# Resource Limits
variable "default_cpu_request" {
  description = "Default CPU request for containers"
  type        = string
  default     = "100m"
}

variable "default_memory_request" {
  description = "Default memory request for containers"
  type        = string
  default     = "128Mi"
}

variable "default_cpu_limit" {
  description = "Default CPU limit for containers"
  type        = string
  default     = "1000m"
}

variable "default_memory_limit" {
  description = "Default memory limit for containers"
  type        = string
  default     = "1Gi"
}

# Logging Configuration
variable "enable_centralized_logging" {
  description = "Enable centralized logging (ELK stack)"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Log retention period in days"
  type        = number
  default     = 30
}

# Service Mesh Configuration
variable "enable_service_mesh" {
  description = "Enable service mesh (Istio)"
  type        = bool
  default     = true
}

variable "service_mesh_type" {
  description = "Service mesh type (istio, consul, linkerd)"
  type        = string
  default     = "istio"
  
  validation {
    condition     = contains(["istio", "consul", "linkerd"], var.service_mesh_type)
    error_message = "Service mesh type must be one of: istio, consul, linkerd."
  }
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot/preemptible instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_percentage" {
  description = "Percentage of spot instances in the cluster"
  type        = number
  default     = 50
  
  validation {
    condition     = var.spot_instance_percentage >= 0 && var.spot_instance_percentage <= 100
    error_message = "Spot instance percentage must be between 0 and 100."
  }
}