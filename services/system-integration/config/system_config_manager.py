#!/usr/bin/env python3
"""
System-wide Configuration Management for Complex Deployments
Handles configuration for all advanced system components
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import redis
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Configuration for individual services"""
    name: str
    url: str
    port: int
    timeout: int
    retry_attempts: int
    health_check_path: str
    authentication_required: bool
    resource_limits: Dict[str, str]
    environment_variables: Dict[str, str]

@dataclass
class CloudResourceConfig:
    """Configuration for cloud computing resources"""
    provider: str
    region: str
    resource_type: str  # tpu, gpu, serverless
    instance_type: str
    auto_scaling: bool
    min_instances: int
    max_instances: int
    cost_optimization: bool
    monitoring_enabled: bool

@dataclass
class DatabaseConfig:
    """Configuration for database systems"""
    database_type: str  # postgres, neo4j, influxdb, redis, mongodb
    host: str
    port: int
    database_name: str
    connection_pool_size: int
    timeout: int
    ssl_enabled: bool
    backup_enabled: bool
    replication_enabled: bool

@dataclass
class PipelineConfig:
    """Configuration for data processing pipelines"""
    pipeline_name: str
    input_services: List[str]
    processing_services: List[str]
    output_services: List[str]
    processing_interval: int
    batch_size: int
    parallel_processing: bool
    error_handling_strategy: str
    monitoring_enabled: bool

@dataclass
class IntegrationFeatureConfig:
    """Configuration for advanced integration features"""
    feature_name: str
    enabled: bool
    dependencies: List[str]
    configuration_parameters: Dict[str, Any]
    resource_requirements: Dict[str, str]
    monitoring_metrics: List[str]

class SystemConfigManager:
    """Manages system-wide configuration for complex deployments"""
    
    def __init__(self, config_path: str = "/app/config", redis_host: str = "redis"):
        self.config_path = Path(config_path)
        self.redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        self.config_cache = {}
        self.config_version = "1.0.0"
        
        # Initialize configuration
        self._load_base_configuration()
        
    def _load_base_configuration(self):
        """Load base configuration from files"""
        try:
            # Load main configuration file
            main_config_file = self.config_path / "integration.yaml"
            if main_config_file.exists():
                with open(main_config_file, 'r') as f:
                    self.base_config = yaml.safe_load(f)
            else:
                self.base_config = self._get_default_configuration()
                
            logger.info("Base configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load base configuration: {e}")
            self.base_config = self._get_default_configuration()
    
    def _get_default_configuration(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
            'system': {
                'name': 'Prometheus AI System Integration',
                'version': self.config_version,
                'environment': os.getenv('ENVIRONMENT', 'development')
            },
            'services': {},
            'cloud_resources': {},
            'databases': {},
            'pipelines': {},
            'integration_features': {},
            'monitoring': {
                'enabled': True,
                'health_check_interval': 30,
                'performance_metrics': True,
                'error_tracking': True
            }
        }
    
    def register_service(self, service_config: ServiceConfig):
        """Register a new service configuration"""
        try:
            service_dict = asdict(service_config)
            
            # Store in local cache
            if 'services' not in self.config_cache:
                self.config_cache['services'] = {}
            self.config_cache['services'][service_config.name] = service_dict
            
            # Store in Redis for distributed access
            self.redis_client.hset(
                'service_configs', 
                service_config.name, 
                json.dumps(service_dict)
            )
            
            logger.info(f"Service {service_config.name} registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register service {service_config.name}: {e}")
            raise
    
    def register_cloud_resource(self, resource_config: CloudResourceConfig):
        """Register cloud resource configuration"""
        try:
            resource_dict = asdict(resource_config)
            resource_key = f"{resource_config.provider}_{resource_config.resource_type}"
            
            # Store in local cache
            if 'cloud_resources' not in self.config_cache:
                self.config_cache['cloud_resources'] = {}
            self.config_cache['cloud_resources'][resource_key] = resource_dict
            
            # Store in Redis
            self.redis_client.hset(
                'cloud_resource_configs',
                resource_key,
                json.dumps(resource_dict)
            )
            
            logger.info(f"Cloud resource {resource_key} registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register cloud resource: {e}")
            raise
    
    def register_database(self, db_config: DatabaseConfig):
        """Register database configuration"""
        try:
            db_dict = asdict(db_config)
            
            # Store in local cache
            if 'databases' not in self.config_cache:
                self.config_cache['databases'] = {}
            self.config_cache['databases'][db_config.database_type] = db_dict
            
            # Store in Redis
            self.redis_client.hset(
                'database_configs',
                db_config.database_type,
                json.dumps(db_dict)
            )
            
            logger.info(f"Database {db_config.database_type} registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register database: {e}")
            raise
    
    def register_pipeline(self, pipeline_config: PipelineConfig):
        """Register data pipeline configuration"""
        try:
            pipeline_dict = asdict(pipeline_config)
            
            # Store in local cache
            if 'pipelines' not in self.config_cache:
                self.config_cache['pipelines'] = {}
            self.config_cache['pipelines'][pipeline_config.pipeline_name] = pipeline_dict
            
            # Store in Redis
            self.redis_client.hset(
                'pipeline_configs',
                pipeline_config.pipeline_name,
                json.dumps(pipeline_dict)
            )
            
            logger.info(f"Pipeline {pipeline_config.pipeline_name} registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register pipeline: {e}")
            raise
    
    def register_integration_feature(self, feature_config: IntegrationFeatureConfig):
        """Register integration feature configuration"""
        try:
            feature_dict = asdict(feature_config)
            
            # Store in local cache
            if 'integration_features' not in self.config_cache:
                self.config_cache['integration_features'] = {}
            self.config_cache['integration_features'][feature_config.feature_name] = feature_dict
            
            # Store in Redis
            self.redis_client.hset(
                'integration_feature_configs',
                feature_config.feature_name,
                json.dumps(feature_dict)
            )
            
            logger.info(f"Integration feature {feature_config.feature_name} registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register integration feature: {e}")
            raise
    
    def get_service_config(self, service_name: str) -> Optional[ServiceConfig]:
        """Get service configuration"""
        try:
            # Try local cache first
            if 'services' in self.config_cache and service_name in self.config_cache['services']:
                config_dict = self.config_cache['services'][service_name]
                return ServiceConfig(**config_dict)
            
            # Try Redis
            config_json = self.redis_client.hget('service_configs', service_name)
            if config_json:
                config_dict = json.loads(config_json)
                return ServiceConfig(**config_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get service config for {service_name}: {e}")
            return None
    
    def get_cloud_resource_config(self, resource_key: str) -> Optional[CloudResourceConfig]:
        """Get cloud resource configuration"""
        try:
            # Try local cache first
            if 'cloud_resources' in self.config_cache and resource_key in self.config_cache['cloud_resources']:
                config_dict = self.config_cache['cloud_resources'][resource_key]
                return CloudResourceConfig(**config_dict)
            
            # Try Redis
            config_json = self.redis_client.hget('cloud_resource_configs', resource_key)
            if config_json:
                config_dict = json.loads(config_json)
                return CloudResourceConfig(**config_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cloud resource config for {resource_key}: {e}")
            return None
    
    def get_database_config(self, database_type: str) -> Optional[DatabaseConfig]:
        """Get database configuration"""
        try:
            # Try local cache first
            if 'databases' in self.config_cache and database_type in self.config_cache['databases']:
                config_dict = self.config_cache['databases'][database_type]
                return DatabaseConfig(**config_dict)
            
            # Try Redis
            config_json = self.redis_client.hget('database_configs', database_type)
            if config_json:
                config_dict = json.loads(config_json)
                return DatabaseConfig(**config_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get database config for {database_type}: {e}")
            return None
    
    def get_pipeline_config(self, pipeline_name: str) -> Optional[PipelineConfig]:
        """Get pipeline configuration"""
        try:
            # Try local cache first
            if 'pipelines' in self.config_cache and pipeline_name in self.config_cache['pipelines']:
                config_dict = self.config_cache['pipelines'][pipeline_name]
                return PipelineConfig(**config_dict)
            
            # Try Redis
            config_json = self.redis_client.hget('pipeline_configs', pipeline_name)
            if config_json:
                config_dict = json.loads(config_json)
                return PipelineConfig(**config_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get pipeline config for {pipeline_name}: {e}")
            return None
    
    def get_integration_feature_config(self, feature_name: str) -> Optional[IntegrationFeatureConfig]:
        """Get integration feature configuration"""
        try:
            # Try local cache first
            if 'integration_features' in self.config_cache and feature_name in self.config_cache['integration_features']:
                config_dict = self.config_cache['integration_features'][feature_name]
                return IntegrationFeatureConfig(**config_dict)
            
            # Try Redis
            config_json = self.redis_client.hget('integration_feature_configs', feature_name)
            if config_json:
                config_dict = json.loads(config_json)
                return IntegrationFeatureConfig(**config_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get integration feature config for {feature_name}: {e}")
            return None
    
    def get_all_service_configs(self) -> Dict[str, ServiceConfig]:
        """Get all service configurations"""
        try:
            configs = {}
            
            # Get all from Redis
            all_configs = self.redis_client.hgetall('service_configs')
            for service_name, config_json in all_configs.items():
                config_dict = json.loads(config_json)
                configs[service_name] = ServiceConfig(**config_dict)
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to get all service configs: {e}")
            return {}
    
    def update_service_config(self, service_name: str, updates: Dict[str, Any]):
        """Update service configuration"""
        try:
            current_config = self.get_service_config(service_name)
            if not current_config:
                raise ValueError(f"Service {service_name} not found")
            
            # Apply updates
            config_dict = asdict(current_config)
            config_dict.update(updates)
            
            # Create updated config
            updated_config = ServiceConfig(**config_dict)
            
            # Re-register with updates
            self.register_service(updated_config)
            
            logger.info(f"Service {service_name} configuration updated")
            
        except Exception as e:
            logger.error(f"Failed to update service config for {service_name}: {e}")
            raise
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate system configuration"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Validate service configurations
            service_configs = self.get_all_service_configs()
            for service_name, config in service_configs.items():
                if not config.url or not config.port:
                    validation_results['errors'].append(
                        f"Service {service_name} missing required URL or port"
                    )
                    validation_results['valid'] = False
                
                if config.timeout <= 0:
                    validation_results['warnings'].append(
                        f"Service {service_name} has invalid timeout value"
                    )
            
            # Validate cloud resource configurations
            cloud_configs = self.redis_client.hgetall('cloud_resource_configs')
            for resource_key, config_json in cloud_configs.items():
                config_dict = json.loads(config_json)
                if not config_dict.get('provider') or not config_dict.get('region'):
                    validation_results['errors'].append(
                        f"Cloud resource {resource_key} missing provider or region"
                    )
                    validation_results['valid'] = False
            
            # Validate database configurations
            db_configs = self.redis_client.hgetall('database_configs')
            for db_type, config_json in db_configs.items():
                config_dict = json.loads(config_json)
                if not config_dict.get('host') or not config_dict.get('port'):
                    validation_results['errors'].append(
                        f"Database {db_type} missing host or port"
                    )
                    validation_results['valid'] = False
            
            logger.info(f"Configuration validation completed: {validation_results['valid']}")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def export_configuration(self, export_path: str):
        """Export complete system configuration"""
        try:
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'config_version': self.config_version,
                    'system_name': 'Prometheus AI System Integration'
                },
                'services': {},
                'cloud_resources': {},
                'databases': {},
                'pipelines': {},
                'integration_features': {}
            }
            
            # Export service configurations
            service_configs = self.redis_client.hgetall('service_configs')
            for service_name, config_json in service_configs.items():
                export_data['services'][service_name] = json.loads(config_json)
            
            # Export cloud resource configurations
            cloud_configs = self.redis_client.hgetall('cloud_resource_configs')
            for resource_key, config_json in cloud_configs.items():
                export_data['cloud_resources'][resource_key] = json.loads(config_json)
            
            # Export database configurations
            db_configs = self.redis_client.hgetall('database_configs')
            for db_type, config_json in db_configs.items():
                export_data['databases'][db_type] = json.loads(config_json)
            
            # Export pipeline configurations
            pipeline_configs = self.redis_client.hgetall('pipeline_configs')
            for pipeline_name, config_json in pipeline_configs.items():
                export_data['pipelines'][pipeline_name] = json.loads(config_json)
            
            # Export integration feature configurations
            feature_configs = self.redis_client.hgetall('integration_feature_configs')
            for feature_name, config_json in feature_configs.items():
                export_data['integration_features'][feature_name] = json.loads(config_json)
            
            # Write to file
            with open(export_path, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            raise
    
    def import_configuration(self, import_path: str):
        """Import system configuration from file"""
        try:
            with open(import_path, 'r') as f:
                import_data = yaml.safe_load(f)
            
            # Import service configurations
            if 'services' in import_data:
                for service_name, config_dict in import_data['services'].items():
                    service_config = ServiceConfig(**config_dict)
                    self.register_service(service_config)
            
            # Import cloud resource configurations
            if 'cloud_resources' in import_data:
                for resource_key, config_dict in import_data['cloud_resources'].items():
                    resource_config = CloudResourceConfig(**config_dict)
                    self.register_cloud_resource(resource_config)
            
            # Import database configurations
            if 'databases' in import_data:
                for db_type, config_dict in import_data['databases'].items():
                    db_config = DatabaseConfig(**config_dict)
                    self.register_database(db_config)
            
            # Import pipeline configurations
            if 'pipelines' in import_data:
                for pipeline_name, config_dict in import_data['pipelines'].items():
                    pipeline_config = PipelineConfig(**config_dict)
                    self.register_pipeline(pipeline_config)
            
            # Import integration feature configurations
            if 'integration_features' in import_data:
                for feature_name, config_dict in import_data['integration_features'].items():
                    feature_config = IntegrationFeatureConfig(**config_dict)
                    self.register_integration_feature(feature_config)
            
            logger.info(f"Configuration imported from {import_path}")
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            raise

# Global configuration manager instance
config_manager = SystemConfigManager()

def initialize_default_configurations():
    """Initialize default configurations for all system components"""
    try:
        # Register default service configurations
        services = [
            ServiceConfig(
                name="helios-core",
                url="http://helios-core",
                port=8000,
                timeout=30,
                retry_attempts=3,
                health_check_path="/health",
                authentication_required=True,
                resource_limits={"memory": "2Gi", "cpu": "1000m"},
                environment_variables={"LOG_LEVEL": "INFO"}
            ),
            ServiceConfig(
                name="volcano-watchtower",
                url="http://volcano-watchtower",
                port=8001,
                timeout=30,
                retry_attempts=3,
                health_check_path="/health",
                authentication_required=True,
                resource_limits={"memory": "1Gi", "cpu": "500m"},
                environment_variables={"LOG_LEVEL": "INFO"}
            ),
            ServiceConfig(
                name="vulcan-ui",
                url="http://vulcan-ui",
                port=3000,
                timeout=30,
                retry_attempts=3,
                health_check_path="/health",
                authentication_required=False,
                resource_limits={"memory": "512Mi", "cpu": "250m"},
                environment_variables={"NODE_ENV": "production"}
            )
        ]
        
        for service in services:
            config_manager.register_service(service)
        
        # Register default cloud resource configurations
        cloud_resources = [
            CloudResourceConfig(
                provider="google_cloud",
                region="us-central1",
                resource_type="tpu",
                instance_type="v4-8",
                auto_scaling=True,
                min_instances=0,
                max_instances=10,
                cost_optimization=True,
                monitoring_enabled=True
            ),
            CloudResourceConfig(
                provider="google_cloud",
                region="us-central1",
                resource_type="gpu",
                instance_type="a100-40gb",
                auto_scaling=True,
                min_instances=0,
                max_instances=5,
                cost_optimization=True,
                monitoring_enabled=True
            )
        ]
        
        for resource in cloud_resources:
            config_manager.register_cloud_resource(resource)
        
        # Register default database configurations
        databases = [
            DatabaseConfig(
                database_type="postgres",
                host="postgres",
                port=5432,
                database_name="prometheus",
                connection_pool_size=20,
                timeout=30,
                ssl_enabled=True,
                backup_enabled=True,
                replication_enabled=True
            ),
            DatabaseConfig(
                database_type="redis",
                host="redis",
                port=6379,
                database_name="0",
                connection_pool_size=10,
                timeout=5,
                ssl_enabled=False,
                backup_enabled=True,
                replication_enabled=False
            )
        ]
        
        for db in databases:
            config_manager.register_database(db)
        
        logger.info("Default configurations initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize default configurations: {e}")
        raise

if __name__ == "__main__":
    initialize_default_configurations()