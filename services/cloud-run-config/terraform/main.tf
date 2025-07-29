# Terraform configuration for Cloud Run multi-region deployment
# with cost optimization and global load balancing

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.0"
    }
  }
}

# Variables
variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
  default     = "prometheus-ai-platform"
}

variable "regions" {
  description = "List of regions for multi-region deployment"
  type        = list(string)
  default     = ["us-central1", "europe-west1", "asia-northeast1"]
}

variable "services" {
  description = "Map of services with their configurations"
  type = map(object({
    image_name      = string
    cpu            = string
    memory         = string
    concurrency    = number
    timeout        = string
    min_instances  = number
    max_instances  = number
  }))
  default = {
    helios-core = {
      image_name    = "helios-core"
      cpu          = "4000m"
      memory       = "8Gi"
      concurrency  = 10
      timeout      = "300s"
      min_instances = 0
      max_instances = 50
    }
    api-gateway = {
      image_name    = "api-gateway"
      cpu          = "2000m"
      memory       = "4Gi"
      concurrency  = 1000
      timeout      = "60s"
      min_instances = 1
      max_instances = 100
    }
    data-ingestion = {
      image_name    = "data-ingestion"
      cpu          = "2000m"
      memory       = "4Gi"
      concurrency  = 100
      timeout      = "600s"
      min_instances = 0
      max_instances = 20
    }
    volcano-watchtower = {
      image_name    = "volcano-watchtower"
      cpu          = "3000m"
      memory       = "6Gi"
      concurrency  = 50
      timeout      = "180s"
      min_instances = 1
      max_instances = 30
    }
  }
}

# Provider configuration
provider "google" {
  project = var.project_id
}

provider "google-beta" {
  project = var.project_id
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "containerregistry.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "dns.googleapis.com",
    "compute.googleapis.com"
  ])
  
  service = each.value
  
  disable_dependent_services = true
  disable_on_destroy        = false
}

# Cloud Run services deployed to multiple regions
resource "google_cloud_run_service" "services" {
  for_each = {
    for combination in setproduct(keys(var.services), var.regions) :
    "${combination[0]}-${combination[1]}" => {
      service = combination[0]
      region  = combination[1]
    }
  }
  
  name     = each.value.service
  location = each.value.region
  
  template {
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale"        = var.services[each.value.service].min_instances
        "autoscaling.knative.dev/maxScale"        = var.services[each.value.service].max_instances
        "run.googleapis.com/execution-environment" = "gen2"
        "run.googleapis.com/cpu-throttling"       = each.value.service == "helios-core" ? "false" : "true"
        "run.googleapis.com/startup-cpu-boost"    = "true"
      }
    }
    
    spec {
      container_concurrency = var.services[each.value.service].concurrency
      timeout_seconds      = tonumber(replace(var.services[each.value.service].timeout, "s", ""))
      
      containers {
        image = "gcr.io/${var.project_id}/${var.services[each.value.service].image_name}:latest"
        
        ports {
          container_port = 8080
        }
        
        env {
          name  = "REGION"
          value = each.value.region
        }
        
        env {
          name  = "ENVIRONMENT"
          value = "production"
        }
        
        env {
          name  = "PORT"
          value = "8080"
        }
        
        resources {
          limits = {
            cpu    = var.services[each.value.service].cpu
            memory = var.services[each.value.service].memory
          }
          requests = {
            cpu    = "500m"
            memory = "1Gi"
          }
        }
        
        liveness_probe {
          http_get {
            path = "/health"
            port = 8080
          }
          initial_delay_seconds = 30
          period_seconds       = 30
          timeout_seconds      = 10
        }
        
        readiness_probe {
          http_get {
            path = "/ready"
            port = 8080
          }
          initial_delay_seconds = 10
          period_seconds       = 10
          timeout_seconds      = 5
        }
      }
    }
  }
  
  traffic {
    percent         = 100
    latest_revision = true
  }
  
  depends_on = [google_project_service.required_apis]
}

# IAM policy for public access (adjust as needed for production)
resource "google_cloud_run_service_iam_member" "public_access" {
  for_each = {
    for combination in setproduct(keys(var.services), var.regions) :
    "${combination[0]}-${combination[1]}" => {
      service = combination[0]
      region  = combination[1]
    }
  }
  
  service  = google_cloud_run_service.services[each.key].name
  location = google_cloud_run_service.services[each.key].location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Global static IP for load balancer
resource "google_compute_global_address" "prometheus_ai_ip" {
  name = "prometheus-ai-global-ip"
}

# SSL certificate for HTTPS
resource "google_compute_managed_ssl_certificate" "prometheus_ai_ssl" {
  name = "prometheus-ai-ssl-cert"
  
  managed {
    domains = [
      "api.prometheus-ai.com",
      "helios.prometheus-ai.com",
      "data.prometheus-ai.com",
      "volcano.prometheus-ai.com"
    ]
  }
}

# URL map for load balancer
resource "google_compute_url_map" "prometheus_ai_lb" {
  name            = "prometheus-ai-load-balancer"
  default_service = google_compute_backend_service.api_gateway_backend.id
  
  host_rule {
    hosts        = ["api.prometheus-ai.com"]
    path_matcher = "api-gateway"
  }
  
  host_rule {
    hosts        = ["helios.prometheus-ai.com"]
    path_matcher = "helios-core"
  }
  
  host_rule {
    hosts        = ["data.prometheus-ai.com"]
    path_matcher = "data-ingestion"
  }
  
  host_rule {
    hosts        = ["volcano.prometheus-ai.com"]
    path_matcher = "volcano-watchtower"
  }
  
  path_matcher {
    name            = "api-gateway"
    default_service = google_compute_backend_service.api_gateway_backend.id
  }
  
  path_matcher {
    name            = "helios-core"
    default_service = google_compute_backend_service.helios_core_backend.id
  }
  
  path_matcher {
    name            = "data-ingestion"
    default_service = google_compute_backend_service.data_ingestion_backend.id
  }
  
  path_matcher {
    name            = "volcano-watchtower"
    default_service = google_compute_backend_service.volcano_watchtower_backend.id
  }
}

# Backend services for each microservice
resource "google_compute_backend_service" "api_gateway_backend" {
  name                  = "api-gateway-backend"
  protocol              = "HTTP"
  timeout_sec           = 60
  enable_cdn           = true
  load_balancing_scheme = "EXTERNAL"
  
  dynamic "backend" {
    for_each = var.regions
    content {
      group = google_compute_region_network_endpoint_group.api_gateway_neg[backend.value].id
    }
  }
  
  health_checks = [google_compute_health_check.api_gateway_health.id]
}

resource "google_compute_backend_service" "helios_core_backend" {
  name                  = "helios-core-backend"
  protocol              = "HTTP"
  timeout_sec           = 300
  load_balancing_scheme = "EXTERNAL"
  
  dynamic "backend" {
    for_each = var.regions
    content {
      group = google_compute_region_network_endpoint_group.helios_core_neg[backend.value].id
    }
  }
  
  health_checks = [google_compute_health_check.helios_core_health.id]
}

resource "google_compute_backend_service" "data_ingestion_backend" {
  name                  = "data-ingestion-backend"
  protocol              = "HTTP"
  timeout_sec           = 600
  load_balancing_scheme = "EXTERNAL"
  
  dynamic "backend" {
    for_each = var.regions
    content {
      group = google_compute_region_network_endpoint_group.data_ingestion_neg[backend.value].id
    }
  }
  
  health_checks = [google_compute_health_check.data_ingestion_health.id]
}

resource "google_compute_backend_service" "volcano_watchtower_backend" {
  name                  = "volcano-watchtower-backend"
  protocol              = "HTTP"
  timeout_sec           = 180
  load_balancing_scheme = "EXTERNAL"
  
  dynamic "backend" {
    for_each = var.regions
    content {
      group = google_compute_region_network_endpoint_group.volcano_watchtower_neg[backend.value].id
    }
  }
  
  health_checks = [google_compute_health_check.volcano_watchtower_health.id]
}

# Network Endpoint Groups for Cloud Run services
resource "google_compute_region_network_endpoint_group" "api_gateway_neg" {
  for_each = toset(var.regions)
  
  name                  = "api-gateway-neg-${each.value}"
  network_endpoint_type = "SERVERLESS"
  region                = each.value
  
  cloud_run {
    service = google_cloud_run_service.services["api-gateway-${each.value}"].name
  }
}

resource "google_compute_region_network_endpoint_group" "helios_core_neg" {
  for_each = toset(var.regions)
  
  name                  = "helios-core-neg-${each.value}"
  network_endpoint_type = "SERVERLESS"
  region                = each.value
  
  cloud_run {
    service = google_cloud_run_service.services["helios-core-${each.value}"].name
  }
}

resource "google_compute_region_network_endpoint_group" "data_ingestion_neg" {
  for_each = toset(var.regions)
  
  name                  = "data-ingestion-neg-${each.value}"
  network_endpoint_type = "SERVERLESS"
  region                = each.value
  
  cloud_run {
    service = google_cloud_run_service.services["data-ingestion-${each.value}"].name
  }
}

resource "google_compute_region_network_endpoint_group" "volcano_watchtower_neg" {
  for_each = toset(var.regions)
  
  name                  = "volcano-watchtower-neg-${each.value}"
  network_endpoint_type = "SERVERLESS"
  region                = each.value
  
  cloud_run {
    service = google_cloud_run_service.services["volcano-watchtower-${each.value}"].name
  }
}

# Health checks
resource "google_compute_health_check" "api_gateway_health" {
  name = "api-gateway-health-check"
  
  http_health_check {
    request_path = "/health"
    port         = "8080"
  }
  
  check_interval_sec  = 30
  timeout_sec         = 10
  healthy_threshold   = 2
  unhealthy_threshold = 3
}

resource "google_compute_health_check" "helios_core_health" {
  name = "helios-core-health-check"
  
  http_health_check {
    request_path = "/health"
    port         = "8080"
  }
  
  check_interval_sec  = 30
  timeout_sec         = 10
  healthy_threshold   = 2
  unhealthy_threshold = 3
}

resource "google_compute_health_check" "data_ingestion_health" {
  name = "data-ingestion-health-check"
  
  http_health_check {
    request_path = "/health"
    port         = "8080"
  }
  
  check_interval_sec  = 30
  timeout_sec         = 10
  healthy_threshold   = 2
  unhealthy_threshold = 3
}

resource "google_compute_health_check" "volcano_watchtower_health" {
  name = "volcano-watchtower-health-check"
  
  http_health_check {
    request_path = "/health"
    port         = "8080"
  }
  
  check_interval_sec  = 30
  timeout_sec         = 10
  healthy_threshold   = 2
  unhealthy_threshold = 3
}

# HTTPS proxy
resource "google_compute_target_https_proxy" "prometheus_ai_https_proxy" {
  name             = "prometheus-ai-https-proxy"
  url_map          = google_compute_url_map.prometheus_ai_lb.id
  ssl_certificates = [google_compute_managed_ssl_certificate.prometheus_ai_ssl.id]
}

# Global forwarding rule
resource "google_compute_global_forwarding_rule" "prometheus_ai_forwarding_rule" {
  name       = "prometheus-ai-forwarding-rule"
  target     = google_compute_target_https_proxy.prometheus_ai_https_proxy.id
  port_range = "443"
  ip_address = google_compute_global_address.prometheus_ai_ip.address
}

# Monitoring dashboard
resource "google_monitoring_dashboard" "prometheus_ai_dashboard" {
  dashboard_json = jsonencode({
    displayName = "Prometheus AI Cloud Run Dashboard"
    mosaicLayout = {
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "Request Count by Service"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\""
                      aggregation = {
                        alignmentPeriod     = "60s"
                        perSeriesAligner    = "ALIGN_RATE"
                        crossSeriesReducer  = "REDUCE_SUM"
                        groupByFields       = ["resource.label.service_name"]
                      }
                    }
                  }
                }
              ]
            }
          }
        },
        {
          width  = 6
          height = 4
          xPos   = 6
          widget = {
            title = "Response Latency"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_MEAN"
                        crossSeriesReducer = "REDUCE_MEAN"
                        groupByFields      = ["resource.label.service_name"]
                      }
                    }
                  }
                }
              ]
            }
          }
        },
        {
          width  = 12
          height = 4
          yPos   = 4
          widget = {
            title = "Instance Count by Region"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_MEAN"
                        crossSeriesReducer = "REDUCE_SUM"
                        groupByFields      = ["resource.label.location", "resource.label.service_name"]
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
  })
}

# Outputs
output "service_urls" {
  description = "URLs of deployed Cloud Run services"
  value = {
    for key, service in google_cloud_run_service.services :
    key => service.status[0].url
  }
}

output "load_balancer_ip" {
  description = "Global load balancer IP address"
  value       = google_compute_global_address.prometheus_ai_ip.address
}

output "dashboard_url" {
  description = "Monitoring dashboard URL"
  value       = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.prometheus_ai_dashboard.id}?project=${var.project_id}"
}