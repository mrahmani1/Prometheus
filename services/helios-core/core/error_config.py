"""
Configuration module for hierarchical error management system

This module provides configuration settings and customization options
for the Prometheus AI error management system.

Requirements: 12.3, 12.4
"""

import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    pass
except ImportError:
    pass

@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling behavior"""

    # Global settings
    max_error_history: int = 1000
    error_analysis_window_hours: int = 24
    enable_automatic_degradation: bool = True
    enable_error_alerts: bool = True

    # Retry settings
    default_max_retries: int = 3
    default_retry_delay: float = 1.0
    exponential_backoff: bool = True

    # Degradation thresholds
    critical_error_threshold: int = 5
    high_error_rate_threshold: float = 50.0
    medium_error_rate_threshold: float = 20.0
    low_error_rate_threshold: float = 10.0

    # Recovery strategy settings
    enable_data_quality_recovery: bool = True
    enable_model_convergence_recovery: bool = True
    enable_physics_violation_recovery: bool = True
    enable_system_resource_recovery: bool = True
    enable_network_connectivity_recovery: bool = True
    enable_security_recovery: bool = True

    # Logging settings
    log_level: str = "INFO"
    structured_logging: bool = True
    log_error_details: bool = True
    log_recovery_attempts: bool = True

    # Alert settings
    alert_critical_errors: bool = True
    alert_high_error_rates: bool = True
    alert_degradation_changes: bool = True
    alert_recovery_failures: bool = True

    # Service-specific settings
    service_specific_configs: Dict[str, Any] = None
    service_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class ServiceSpecificConfig:
    """Service-specific error handling configuration"""

    service_name: str
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_fallback: bool = True
    fallback_timeout: float = 30.0
    custom_recovery_strategies: List[str] = field(default_factory=list)
    priority_operations: List[str] = field(default_factory=list)
    degradation_exempt: bool = False

class ErrorConfigManager:
    """Manages error handling configuration"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.getenv(
            "ERROR_CONFIG_FILE", "error_config.json"
        )
        self.config = self._load_default_config()
        self._load_config_from_file()
        self._load_config_from_env()

    def _load_default_config(self) -> ErrorHandlingConfig:
        """Load default configuration"""
        return ErrorHandlingConfig(
            service_configs={
                "helios_core": {
                    "max_retries": 2,
                    "retry_delay": 2.0,
                    "enable_physics_recovery": True,
                    "model_complexity_fallback": True,
                },
                "volcano_watchtower": {
                    "max_retries": 1,
                    "retry_delay": 0.5,
                    "real_time_priority": True,
                    "sensor_data_fallback": True,
                },
                "data_ingestion": {
                    "max_retries": 5,
                    "retry_delay": 1.0,
                    "external_service_timeout": 30.0,
                    "data_quality_checks": True,
                },
                "api_gateway": {
                    "max_retries": 3,
                    "retry_delay": 0.5,
                    "security_priority": True,
                    "rate_limiting_recovery": True,
                },
            }
        )

    def _load_config_from_file(self):
        """Load configuration from file if it exists"""
        import json

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    file_config = json.load(f)
                    self._merge_config(file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")

    def _load_config_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            "ERROR_MAX_RETRIES": ("default_max_retries", int),
            "ERROR_RETRY_DELAY": ("default_retry_delay", float),
            "ERROR_CRITICAL_THRESHOLD": ("critical_error_threshold", int),
            "ERROR_HIGH_RATE_THRESHOLD": ("high_error_rate_threshold", float),
            "ERROR_ENABLE_DEGRADATION": ("enable_automatic_degradation", bool),
            "ERROR_ENABLE_ALERTS": ("enable_error_alerts", bool),
            "ERROR_LOG_LEVEL": ("log_level", str),
            "ERROR_MAX_HISTORY": ("max_error_history", int),
        }

        for env_var, (config_attr, type_func) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if type_func == bool:
                        value = env_value.lower() in ("true", "1", "yes", "on")
                    else:
                        value = type_func(env_value)
                    setattr(self.config, config_attr, value)
                except ValueError as e:
                    print(f"Warning: Invalid value for {env_var}: {env_value} ({e})")

    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing config"""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                if isinstance(value, dict) and isinstance(
                    getattr(self.config, key), dict
                ):
                    # Merge dictionaries
                    getattr(self.config, key).update(value)
                else:
                    setattr(self.config, key, value)

    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a specific service"""
        return self.config.service_configs.get(service_name, {})

    def update_service_config(self, service_name: str, config_updates: Dict[str, Any]):
        """Update configuration for a specific service"""
        if service_name not in self.config.service_configs:
            self.config.service_configs[service_name] = {}

        self.config.service_configs[service_name].update(config_updates)

    def get_degradation_thresholds(self) -> Dict[str, Any]:
        """Get service degradation thresholds"""
        return {
            "critical_errors": self.config.critical_error_threshold,
            "high_error_rate": self.config.high_error_rate_threshold,
            "medium_error_rate": self.config.medium_error_rate_threshold,
            "low_error_rate": self.config.low_error_rate_threshold,
        }

    def get_recovery_strategy_config(self) -> Dict[str, bool]:
        """Get recovery strategy enablement configuration"""
        return {
            "data_quality": self.config.enable_data_quality_recovery,
            "model_convergence": self.config.enable_model_convergence_recovery,
            "physics_violation": self.config.enable_physics_violation_recovery,
            "system_resource": self.config.enable_system_resource_recovery,
            "network_connectivity": self.config.enable_network_connectivity_recovery,
            "security": self.config.enable_security_recovery,
        }

    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file"""
        from dataclasses import asdict

        save_path = file_path or self.config_file

        try:
            config_dict = asdict(self.config)
            with open(save_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving config to {save_path}: {e}")

# Predefined configuration profiles
class ConfigProfiles:
    """Predefined configuration profiles for different deployment scenarios"""

    @staticmethod
    def development_profile() -> ErrorHandlingConfig:
        """Configuration profile for development environment"""
        return ErrorHandlingConfig(
            max_error_history=100,
            default_max_retries=1,
            default_retry_delay=0.5,
            enable_automatic_degradation=False,
            enable_error_alerts=False,
            log_level="DEBUG",
            critical_error_threshold=10,
            high_error_rate_threshold=100.0,
        )

    @staticmethod
    def testing_profile() -> ErrorHandlingConfig:
        """Configuration profile for testing environment"""
        return ErrorHandlingConfig(
            max_error_history=500,
            default_max_retries=2,
            default_retry_delay=0.1,
            enable_automatic_degradation=True,
            enable_error_alerts=False,
            log_level="INFO",
            critical_error_threshold=5,
            high_error_rate_threshold=50.0,
        )

    @staticmethod
    def production_profile() -> ErrorHandlingConfig:
        """Configuration profile for production environment"""
        return ErrorHandlingConfig(
            max_error_history=2000,
            default_max_retries=3,
            default_retry_delay=1.0,
            enable_automatic_degradation=True,
            enable_error_alerts=True,
            log_level="WARNING",
            critical_error_threshold=3,
            high_error_rate_threshold=20.0,
            service_configs={
                "helios_core": {
                    "max_retries": 2,
                    "retry_delay": 2.0,
                    "enable_physics_recovery": True,
                    "model_complexity_fallback": True,
                    "production_optimized": True,
                },
                "volcano_watchtower": {
                    "max_retries": 1,
                    "retry_delay": 0.5,
                    "real_time_priority": True,
                    "sensor_data_fallback": True,
                    "alert_system_priority": True,
                },
            },
        )

    @staticmethod
    def high_availability_profile() -> ErrorHandlingConfig:
        """Configuration profile for high availability deployment"""
        return ErrorHandlingConfig(
            max_error_history=5000,
            default_max_retries=5,
            default_retry_delay=0.5,
            enable_automatic_degradation=True,
            enable_error_alerts=True,
            log_level="ERROR",
            critical_error_threshold=1,
            high_error_rate_threshold=10.0,
            exponential_backoff=True,
        )

# Environment-specific configuration loader
def load_config_for_environment(environment: str = None) -> ErrorHandlingConfig:
    """Load configuration based on environment"""
    env = environment or os.getenv("PROMETHEUS_ENV", "development").lower()

    profile_map = {
        "development": ConfigProfiles.development_profile,
        "dev": ConfigProfiles.development_profile,
        "testing": ConfigProfiles.testing_profile,
        "test": ConfigProfiles.testing_profile,
        "production": ConfigProfiles.production_profile,
        "prod": ConfigProfiles.production_profile,
        "high_availability": ConfigProfiles.high_availability_profile,
        "ha": ConfigProfiles.high_availability_profile,
    }

    profile_func = profile_map.get(env, ConfigProfiles.development_profile)
    return profile_func()

# Global configuration manager instance
config_manager = ErrorConfigManager()

# Convenience functions
def get_config() -> ErrorHandlingConfig:
    """Get current error handling configuration"""
    return config_manager.config

def get_service_config(service_name: str) -> Dict[str, Any]:
    """Get configuration for a specific service"""
    return config_manager.get_service_config(service_name)

def update_config(**kwargs):
    """Update global configuration"""
    for key, value in kwargs.items():
        if hasattr(config_manager.config, key):
            setattr(config_manager.config, key, value)

def reload_config():
    """Reload configuration from file and environment"""
    global config_manager
    config_manager = ErrorConfigManager(config_manager.config_file)

# Configuration validation
def validate_config(config: ErrorHandlingConfig) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []

    if config.max_error_history < 10:
        issues.append("max_error_history should be at least 10")

    if config.default_max_retries < 0:
        issues.append("default_max_retries cannot be negative")

    if config.default_retry_delay < 0:
        issues.append("default_retry_delay cannot be negative")

    if config.critical_error_threshold < 1:
        issues.append("critical_error_threshold should be at least 1")

    if config.high_error_rate_threshold <= config.medium_error_rate_threshold:
        issues.append(
            "high_error_rate_threshold should be greater than medium_error_rate_threshold"
        )

    if config.medium_error_rate_threshold <= config.low_error_rate_threshold:
        issues.append(
            "medium_error_rate_threshold should be greater than low_error_rate_threshold"
        )

    return issues

# Example configuration file template
CONFIG_TEMPLATE = """
{
  "max_error_history": 1000,
  "error_analysis_window_hours": 24,
  "enable_automatic_degradation": true,
  "enable_error_alerts": true,
  "default_max_retries": 3,
  "default_retry_delay": 1.0,
  "exponential_backo": true,
  "critical_error_threshold": 5,
  "high_error_rate_threshold": 50.0,
  "medium_error_rate_threshold": 20.0,
  "low_error_rate_threshold": 10.0,
  "enable_data_quality_recovery": true,
  "enable_model_convergence_recovery": true,
  "enable_physics_violation_recovery": true,
  "enable_system_resource_recovery": true,
  "enable_network_connectivity_recovery": true,
  "enable_security_recovery": true,
  "log_level": "INFO",
  "structured_logging": true,
  "log_error_details": true,
  "log_recovery_attempts": true,
  "alert_critical_errors": true,
  "alert_high_error_rates": true,
  "alert_degradation_changes": true,
  "alert_recovery_failures": true,
  "service_configs": {
    "helios_core": {
      "max_retries": 2,
      "retry_delay": 2.0,
      "enable_physics_recovery": true,
      "model_complexity_fallback": true
    },
    "volcano_watchtower": {
      "max_retries": 1,
      "retry_delay": 0.5,
      "real_time_priority": true,
      "sensor_data_fallback": true
    },
    "data_ingestion": {
      "max_retries": 5,
      "retry_delay": 1.0,
      "external_service_timeout": 30.0,
      "data_quality_checks": true
    },
    "api_gateway": {
      "max_retries": 3,
      "retry_delay": 0.5,
      "security_priority": true,
      "rate_limiting_recovery": true
    }
  }
}
"""

def create_config_template(file_path: str = "error_config.json"):
    """Create a configuration template file"""
    try:
        with open(file_path, "w") as f:
            f.write(CONFIG_TEMPLATE)
        print(f"Configuration template created at {file_path}")
    except Exception as e:
        print(f"Error creating config template: {e}")

if __name__ == "__main__":
    # Example usage
    print("Error Management Configuration")
    print("=" * 40)

    # Load configuration
    config = get_config()
    print(f"Current environment: {os.getenv('PROMETHEUS_ENV', 'development')}")
    print(f"Max retries: {config.default_max_retries}")
    print(f"Critical error threshold: {config.critical_error_threshold}")
    print(f"Automatic degradation: {config.enable_automatic_degradation}")

    # Validate configuration
    issues = validate_config(config)
    if issues:
        print("\nConfiguration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration is valid")

    # Show service-specific configs
    print("\nService-specific configurations:")
    for service_name, service_config in config.service_configs.items():
        print(f"  {service_name}: {service_config}")
