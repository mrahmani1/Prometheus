#!/usr/bin/env python3
"""
Multi-Cloud Disaster Recovery Configuration
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class Environment(str, Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class CloudProviderConfig:
    """Cloud provider configuration"""
    enabled: bool
    project_id: Optional[str] = None
    subscription_id: Optional[str] = None
    account_id: Optional[str] = None
    regions: List[str] = None
    credentials_path: Optional[str] = None
    
    def __post_init__(self):
        if self.regions is None:
            self.regions = []

@dataclass
class ReplicationConfig:
    """Replication configuration"""
    enabled: bool = True
    sync_interval_seconds: int = 300
    max_concurrent_jobs: int = 10
    retry_attempts: int = 3
    compression_enabled: bool = True
    encryption_enabled: bool = True
    bandwidth_limit_mbps: Optional[int] = None

@dataclass
class FailoverConfig:
    """Failover configuration"""
    enabled: bool = True
    automatic_failover: bool = False
    validation_timeout_seconds: int = 300
    rollback_timeout_seconds: int = 600
    health_check_interval_seconds: int = 30
    max_failover_attempts: int = 3

@dataclass
class BackupConfig:
    """Backup configuration"""
    enabled: bool = True
    default_retention_days: int = 30
    max_retention_days: int = 365
    encryption_enabled: bool = True
    compression_enabled: bool = True
    cross_region_backup: bool = True
    backup_schedule_cron: str = "0 2 * * *"  # Daily at 2 AM

@dataclass
class SecurityConfig:
    """Security configuration"""
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    mfa_required: bool = True
    audit_logging: bool = True
    vulnerability_scanning: bool = True
    compliance_monitoring: bool = True
    security_scan_interval_hours: int = 24

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    metrics_interval_seconds: int = 60
    health_check_interval_seconds: int = 30
    alert_thresholds: Dict[str, float] = None
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0,
                "network_latency_ms": 100.0,
                "error_rate": 5.0
            }
        if self.notification_channels is None:
            self.notification_channels = []

@dataclass
class CostOptimizationConfig:
    """Cost optimization configuration"""
    enabled: bool = True
    analysis_interval_hours: int = 24
    auto_optimization: bool = False
    cost_alert_threshold_percent: float = 20.0
    reserved_instance_recommendations: bool = True
    spot_instance_recommendations: bool = True

class DisasterRecoverySettings:
    """Main disaster recovery settings"""
    
    def __init__(self):
        self.environment = Environment(os.getenv("ENVIRONMENT", "development"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Service configuration
        self.service_name = "multi-cloud-disaster-recovery"
        self.service_version = "1.0.0"
        self.api_prefix = "/api/v1"
        
        # Server configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "4"))
        
        # Cloud provider configurations
        self.gcp = CloudProviderConfig(
            enabled=os.getenv("GCP_ENABLED", "true").lower() == "true",
            project_id=os.getenv("GCP_PROJECT_ID"),
            regions=os.getenv("GCP_REGIONS", "us-central1,europe-west1").split(","),
            credentials_path=os.getenv("GCP_CREDENTIALS_PATH")
        )
        
        self.azure = CloudProviderConfig(
            enabled=os.getenv("AZURE_ENABLED", "true").lower() == "true",
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            regions=os.getenv("AZURE_REGIONS", "eastus,westeurope").split(",")
        )
        
        self.aws = CloudProviderConfig(
            enabled=os.getenv("AWS_ENABLED", "true").lower() == "true",
            account_id=os.getenv("AWS_ACCOUNT_ID"),
            regions=os.getenv("AWS_REGIONS", "us-east-1,eu-west-1").split(",")
        )
        
        # Feature configurations
        self.replication = ReplicationConfig(
            enabled=os.getenv("REPLICATION_ENABLED", "true").lower() == "true",
            sync_interval_seconds=int(os.getenv("REPLICATION_SYNC_INTERVAL", "300")),
            max_concurrent_jobs=int(os.getenv("REPLICATION_MAX_JOBS", "10"))
        )
        
        self.failover = FailoverConfig(
            enabled=os.getenv("FAILOVER_ENABLED", "true").lower() == "true",
            automatic_failover=os.getenv("AUTOMATIC_FAILOVER", "false").lower() == "true",
            validation_timeout_seconds=int(os.getenv("FAILOVER_VALIDATION_TIMEOUT", "300"))
        )
        
        self.backup = BackupConfig(
            enabled=os.getenv("BACKUP_ENABLED", "true").lower() == "true",
            default_retention_days=int(os.getenv("BACKUP_RETENTION_DAYS", "30")),
            cross_region_backup=os.getenv("CROSS_REGION_BACKUP", "true").lower() == "true"
        )
        
        self.security = SecurityConfig(
            encryption_at_rest=os.getenv("ENCRYPTION_AT_REST", "true").lower() == "true",
            encryption_in_transit=os.getenv("ENCRYPTION_IN_TRANSIT", "true").lower() == "true",
            mfa_required=os.getenv("MFA_REQUIRED", "true").lower() == "true"
        )
        
        self.monitoring = MonitoringConfig(
            enabled=os.getenv("MONITORING_ENABLED", "true").lower() == "true",
            metrics_interval_seconds=int(os.getenv("MONITORING_INTERVAL", "60"))
        )
        
        self.cost_optimization = CostOptimizationConfig(
            enabled=os.getenv("COST_OPTIMIZATION_ENABLED", "true").lower() == "true",
            analysis_interval_hours=int(os.getenv("COST_ANALYSIS_INTERVAL", "24")),
            auto_optimization=os.getenv("AUTO_COST_OPTIMIZATION", "false").lower() == "true"
        )
        
        # Database configuration
        self.database_url = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/prometheus_dr")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # External service URLs
        self.prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
        self.grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
        self.consul_url = os.getenv("CONSUL_URL", "http://localhost:8500")
        
        # Authentication and authorization
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.jwt_expiration_hours = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
        
        # Rate limiting
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window_seconds = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        
        # CORS configuration
        self.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        self.cors_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.cors_headers = ["*"]
    
    def get_provider_configs(self) -> Dict[str, CloudProviderConfig]:
        """Get enabled provider configurations"""
        configs = {}
        
        if self.gcp.enabled:
            configs["gcp"] = self.gcp
        if self.azure.enabled:
            configs["azure"] = self.azure
        if self.aws.enabled:
            configs["aws"] = self.aws
            
        return configs
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check if at least one provider is enabled
        enabled_providers = [
            self.gcp.enabled,
            self.azure.enabled,
            self.aws.enabled
        ]
        
        if not any(enabled_providers):
            issues.append("At least one cloud provider must be enabled")
        
        # Validate GCP configuration
        if self.gcp.enabled and not self.gcp.project_id:
            issues.append("GCP project ID is required when GCP is enabled")
        
        # Validate Azure configuration
        if self.azure.enabled and not self.azure.subscription_id:
            issues.append("Azure subscription ID is required when Azure is enabled")
        
        # Validate AWS configuration
        if self.aws.enabled and not self.aws.account_id:
            issues.append("AWS account ID is required when AWS is enabled")
        
        # Validate JWT secret in production
        if self.environment == Environment.PRODUCTION and self.jwt_secret_key == "your-secret-key-change-in-production":
            issues.append("JWT secret key must be changed in production")
        
        # Validate database URL
        if not self.database_url:
            issues.append("Database URL is required")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "log_level": self.log_level,
            "service": {
                "name": self.service_name,
                "version": self.service_version,
                "host": self.host,
                "port": self.port,
                "workers": self.workers
            },
            "providers": {
                "gcp": {
                    "enabled": self.gcp.enabled,
                    "project_id": self.gcp.project_id,
                    "regions": self.gcp.regions
                },
                "azure": {
                    "enabled": self.azure.enabled,
                    "subscription_id": self.azure.subscription_id,
                    "regions": self.azure.regions
                },
                "aws": {
                    "enabled": self.aws.enabled,
                    "account_id": self.aws.account_id,
                    "regions": self.aws.regions
                }
            },
            "features": {
                "replication": {
                    "enabled": self.replication.enabled,
                    "sync_interval": self.replication.sync_interval_seconds,
                    "max_jobs": self.replication.max_concurrent_jobs
                },
                "failover": {
                    "enabled": self.failover.enabled,
                    "automatic": self.failover.automatic_failover
                },
                "backup": {
                    "enabled": self.backup.enabled,
                    "retention_days": self.backup.default_retention_days
                },
                "security": {
                    "encryption_at_rest": self.security.encryption_at_rest,
                    "encryption_in_transit": self.security.encryption_in_transit,
                    "mfa_required": self.security.mfa_required
                },
                "monitoring": {
                    "enabled": self.monitoring.enabled,
                    "interval": self.monitoring.metrics_interval_seconds
                },
                "cost_optimization": {
                    "enabled": self.cost_optimization.enabled,
                    "auto_optimization": self.cost_optimization.auto_optimization
                }
            }
        }

# Global settings instance
settings = DisasterRecoverySettings()