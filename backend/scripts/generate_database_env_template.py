#!/usr/bin/env python3
"""
Database Environment Variable Template Generator
Generates comprehensive environment variable templates for all database configurations
across different environments with proper documentation and validation.
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.database_config_manager import (
    DatabaseConfigManager, DatabaseType, Environment, create_config_manager
)
from backend.core.database_config_replacer import DatabaseConfigReplacer


class DatabaseEnvTemplateGenerator:
    """
    Comprehensive environment variable template generator for database configurations.
    Supports multiple environments, validation, and documentation generation.
    """
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.environments = [Environment.DEVELOPMENT, Environment.TESTING, Environment.STAGING, Environment.PRODUCTION]
        
    async def generate_comprehensive_template(self, output_dir: str = None) -> Dict[str, str]:
        """Generate comprehensive environment templates for all environments"""
        if output_dir is None:
            output_dir = self.root_dir / "config" / "database" / "env_templates"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        templates = {}
        
        for env in self.environments:
            template_content = await self._generate_environment_template(env)
            template_file = output_dir / f"{env.value}.env"
            
            with open(template_file, 'w') as f:
                f.write(template_content)
            
            templates[env.value] = str(template_file)
            print(f"Generated template for {env.value}: {template_file}")
        
        # Generate comprehensive documentation
        docs_content = self._generate_documentation()
        docs_file = output_dir / "README.md"
        
        with open(docs_file, 'w') as f:
            f.write(docs_content)
        
        templates['documentation'] = str(docs_file)
        print(f"Generated documentation: {docs_file}")
        
        # Generate validation script
        validation_script = self._generate_validation_script()
        validation_file = output_dir / "validate_env.py"
        
        with open(validation_file, 'w') as f:
            f.write(validation_script)
        
        templates['validation_script'] = str(validation_file)
        print(f"Generated validation script: {validation_file}")
        
        return templates
    
    async def _generate_environment_template(self, environment: Environment) -> str:
        """Generate environment variable template for specific environment"""
        template = f"""# Database Configuration Environment Variables
# Environment: {environment.value.upper()}
# Generated: {datetime.now().isoformat()}
# 
# This file contains all database configuration environment variables
# for the {environment.value} environment. Copy this file to .env and
# update the values according to your setup.

# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================

# Environment setting
ENVIRONMENT={environment.value}

# Database configuration directory
DB_CONFIG_DIR=./config/database

# Encryption settings for sensitive data
DB_ENCRYPTION_KEY=
DB_ENCRYPTION_PASSWORD=prometheus-{environment.value}-key
DB_ENCRYPTION_SALT=prometheus-{environment.value}-salt

"""
        
        # Add database-specific configurations
        db_configs = self._get_database_configurations(environment)
        
        for db_type, config in db_configs.items():
            template += f"""
# =============================================================================
# {db_type.upper()} CONFIGURATION
# =============================================================================

"""
            template += config
        
        # Add service discovery configuration
        template += self._get_service_discovery_config(environment)
        
        # Add credential management configuration
        template += self._get_credential_management_config(environment)
        
        # Add monitoring configuration
        template += self._get_monitoring_config(environment)
        
        # Add security configuration
        template += self._get_security_config(environment)
        
        return template
    
    def _get_database_configurations(self, environment: Environment) -> Dict[str, str]:
        """Get database-specific environment variable configurations"""
        configs = {}
        
        # PostgreSQL Configuration
        if environment == Environment.DEVELOPMENT:
            configs['postgresql'] = """# PostgreSQL Primary Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=prometheus_geo
POSTGRES_USER=prometheus_admin
POSTGRES_PASSWORD=GeoPhys2024!PrometheusDB#
POSTGRES_URL=postgresql://prometheus_admin:GeoPhys2024!PrometheusDB#@localhost:5432/prometheus_geo

# PostgreSQL Connection Pool Settings
POSTGRES_POOL_SIZE=10
POSTGRES_MAX_OVERFLOW=20
POSTGRES_POOL_TIMEOUT=30
POSTGRES_POOL_RECYCLE=3600

# PostgreSQL SSL Settings (disabled for development)
POSTGRES_SSL_ENABLED=false
POSTGRES_SSL_VERIFY=false
POSTGRES_SSL_CA_PATH=
POSTGRES_SSL_CERT_PATH=
POSTGRES_SSL_KEY_PATH=
"""
        elif environment == Environment.PRODUCTION:
            configs['postgresql'] = """# PostgreSQL Primary Database
POSTGRES_PRIMARY_HOST=
POSTGRES_PRIMARY_PORT=5432
POSTGRES_DB=prometheus_geo
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_URL=

# PostgreSQL Read Replica
POSTGRES_REPLICA_HOST=
POSTGRES_REPLICA_PORT=5432

# PostgreSQL Connection Pool Settings
POSTGRES_POOL_SIZE=50
POSTGRES_MAX_OVERFLOW=100
POSTGRES_POOL_TIMEOUT=30
POSTGRES_POOL_RECYCLE=1800

# PostgreSQL SSL Settings (required for production)
POSTGRES_SSL_ENABLED=true
POSTGRES_SSL_VERIFY=true
POSTGRES_SSL_CA_PATH=/etc/ssl/certs/postgres-ca.crt
POSTGRES_SSL_CERT_PATH=/etc/ssl/certs/postgres-client.crt
POSTGRES_SSL_KEY_PATH=/etc/ssl/private/postgres-client.key
"""
        else:
            configs['postgresql'] = f"""# PostgreSQL Database
POSTGRES_HOST=
POSTGRES_PORT=5432
POSTGRES_DB=prometheus_geo_{environment.value}
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_URL=

# PostgreSQL Connection Pool Settings
POSTGRES_POOL_SIZE=25
POSTGRES_MAX_OVERFLOW=50
POSTGRES_POOL_TIMEOUT=30
POSTGRES_POOL_RECYCLE=3600

# PostgreSQL SSL Settings
POSTGRES_SSL_ENABLED=true
POSTGRES_SSL_VERIFY=true
POSTGRES_SSL_CA_PATH=/etc/ssl/certs/postgres-ca.crt
POSTGRES_SSL_CERT_PATH=/etc/ssl/certs/postgres-client.crt
POSTGRES_SSL_KEY_PATH=/etc/ssl/private/postgres-client.key
"""
        
        # Redis Configuration
        if environment == Environment.DEVELOPMENT:
            configs['redis'] = """# Redis Cache Database
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_URL=redis://localhost:6379/0

# Redis Connection Pool Settings
REDIS_POOL_SIZE=20
REDIS_MAX_CONNECTIONS=100
REDIS_SOCKET_TIMEOUT=10
REDIS_HEALTH_CHECK_INTERVAL=60

# Redis SSL Settings (disabled for development)
REDIS_SSL_ENABLED=false
REDIS_SSL_CA_PATH=
REDIS_SSL_CERT_PATH=
REDIS_SSL_KEY_PATH=
"""
        elif environment == Environment.PRODUCTION:
            configs['redis'] = """# Redis Cluster Configuration
REDIS_CLUSTER_HOST=
REDIS_CLUSTER_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_URL=
REDIS_CLUSTER_NODES=3

# Redis Connection Pool Settings
REDIS_POOL_SIZE=100
REDIS_MAX_CONNECTIONS=200
REDIS_SOCKET_TIMEOUT=10
REDIS_HEALTH_CHECK_INTERVAL=30

# Redis SSL Settings (required for production)
REDIS_SSL_ENABLED=true
REDIS_SSL_CA_PATH=/etc/ssl/certs/redis-ca.crt
REDIS_SSL_CERT_PATH=/etc/ssl/certs/redis-client.crt
REDIS_SSL_KEY_PATH=/etc/ssl/private/redis-client.key
"""
        else:
            configs['redis'] = f"""# Redis Database
REDIS_HOST=
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_URL=

# Redis Connection Pool Settings
REDIS_POOL_SIZE=50
REDIS_MAX_CONNECTIONS=100
REDIS_SOCKET_TIMEOUT=15
REDIS_HEALTH_CHECK_INTERVAL=60

# Redis SSL Settings
REDIS_SSL_ENABLED=true
REDIS_SSL_CA_PATH=/etc/ssl/certs/redis-ca.crt
REDIS_SSL_CERT_PATH=/etc/ssl/certs/redis-client.crt
REDIS_SSL_KEY_PATH=/etc/ssl/private/redis-client.key
"""
        
        # InfluxDB Configuration
        if environment == Environment.DEVELOPMENT:
            configs['influxdb'] = """# InfluxDB Time-Series Database
INFLUXDB_HOST=localhost
INFLUXDB_PORT=8086
INFLUXDB_ORG=prometheus
INFLUXDB_BUCKET=geological_data
INFLUXDB_TOKEN=
INFLUXDB_URL=http://localhost:8086
INFLUXDB_RETENTION=30d

# InfluxDB SSL Settings (disabled for development)
INFLUXDB_SSL_ENABLED=false
INFLUXDB_SSL_CA_PATH=
INFLUXDB_SSL_CERT_PATH=
INFLUXDB_SSL_KEY_PATH=
"""
        elif environment == Environment.PRODUCTION:
            configs['influxdb'] = """# InfluxDB Cluster Configuration
INFLUXDB_CLUSTER_HOST=
INFLUXDB_CLUSTER_PORT=8086
INFLUXDB_ORG=prometheus
INFLUXDB_BUCKET=geological_data
INFLUXDB_TOKEN=
INFLUXDB_URL=
INFLUXDB_RETENTION=90d

# InfluxDB SSL Settings (required for production)
INFLUXDB_SSL_ENABLED=true
INFLUXDB_SSL_CA_PATH=/etc/ssl/certs/influxdb-ca.crt
INFLUXDB_SSL_CERT_PATH=/etc/ssl/certs/influxdb-client.crt
INFLUXDB_SSL_KEY_PATH=/etc/ssl/private/influxdb-client.key
"""
        else:
            configs['influxdb'] = f"""# InfluxDB Database
INFLUXDB_HOST=
INFLUXDB_PORT=8086
INFLUXDB_ORG=prometheus_{environment.value}
INFLUXDB_BUCKET=geological_data_{environment.value}
INFLUXDB_TOKEN=
INFLUXDB_URL=
INFLUXDB_RETENTION=7d

# InfluxDB SSL Settings
INFLUXDB_SSL_ENABLED=true
INFLUXDB_SSL_CA_PATH=/etc/ssl/certs/influxdb-ca.crt
INFLUXDB_SSL_CERT_PATH=/etc/ssl/certs/influxdb-client.crt
INFLUXDB_SSL_KEY_PATH=/etc/ssl/private/influxdb-client.key
"""
        
        # Neo4j Configuration
        if environment == Environment.DEVELOPMENT:
            configs['neo4j'] = """# Neo4j Graph Database
NEO4J_HOST=localhost
NEO4J_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=prometheus123
NEO4J_URL=bolt://localhost:7687

# Neo4j Connection Pool Settings
NEO4J_POOL_SIZE=10
NEO4J_MAX_CONNECTIONS=25
NEO4J_CONNECTION_TIMEOUT=30
NEO4J_MAX_TRANSACTION_RETRY_TIME=15

# Neo4j SSL Settings (disabled for development)
NEO4J_SSL_ENABLED=false
NEO4J_SSL_CA_PATH=
NEO4J_SSL_CERT_PATH=
NEO4J_SSL_KEY_PATH=
"""
        elif environment == Environment.PRODUCTION:
            configs['neo4j'] = """# Neo4j Cluster Configuration
NEO4J_CLUSTER_HOST=
NEO4J_CLUSTER_PORT=7687
NEO4J_USER=
NEO4J_PASSWORD=
NEO4J_URL=
NEO4J_CLUSTER_NODES=3

# Neo4j Connection Pool Settings
NEO4J_POOL_SIZE=25
NEO4J_MAX_CONNECTIONS=50
NEO4J_CONNECTION_TIMEOUT=30
NEO4J_MAX_TRANSACTION_RETRY_TIME=15

# Neo4j SSL Settings (required for production)
NEO4J_SSL_ENABLED=true
NEO4J_SSL_CA_PATH=/etc/ssl/certs/neo4j-ca.crt
NEO4J_SSL_CERT_PATH=/etc/ssl/certs/neo4j-client.crt
NEO4J_SSL_KEY_PATH=/etc/ssl/private/neo4j-client.key
"""
        else:
            configs['neo4j'] = f"""# Neo4j Database
NEO4J_HOST=
NEO4J_PORT=7687
NEO4J_USER=
NEO4J_PASSWORD=
NEO4J_URL=

# Neo4j Connection Pool Settings
NEO4J_POOL_SIZE=15
NEO4J_MAX_CONNECTIONS=25
NEO4J_CONNECTION_TIMEOUT=30
NEO4J_MAX_TRANSACTION_RETRY_TIME=15

# Neo4j SSL Settings
NEO4J_SSL_ENABLED=true
NEO4J_SSL_CA_PATH=/etc/ssl/certs/neo4j-ca.crt
NEO4J_SSL_CERT_PATH=/etc/ssl/certs/neo4j-client.crt
NEO4J_SSL_KEY_PATH=/etc/ssl/private/neo4j-client.key
"""
        
        # TimescaleDB Configuration
        if environment == Environment.DEVELOPMENT:
            configs['timescaledb'] = """# TimescaleDB Time-Series Database
TIMESCALEDB_HOST=localhost
TIMESCALEDB_PORT=5432
TIMESCALEDB_DB=prometheus_timeseries
TIMESCALEDB_USER=prometheus_admin
TIMESCALEDB_PASSWORD=GeoPhys2024!PrometheusDB#
TIMESCALEDB_URL=postgresql://prometheus_admin:GeoPhys2024!PrometheusDB#@localhost:5432/prometheus_timeseries

# TimescaleDB Connection Pool Settings
TIMESCALEDB_POOL_SIZE=15
TIMESCALEDB_MAX_OVERFLOW=25
TIMESCALEDB_POOL_TIMEOUT=30
TIMESCALEDB_POOL_RECYCLE=3600

# TimescaleDB SSL Settings (disabled for development)
TIMESCALEDB_SSL_ENABLED=false
TIMESCALEDB_SSL_CA_PATH=
TIMESCALEDB_SSL_CERT_PATH=
TIMESCALEDB_SSL_KEY_PATH=
"""
        elif environment == Environment.PRODUCTION:
            configs['timescaledb'] = """# TimescaleDB Cluster Configuration
TIMESCALEDB_CLUSTER_HOST=
TIMESCALEDB_CLUSTER_PORT=5432
TIMESCALEDB_DB=prometheus_timeseries
TIMESCALEDB_USER=
TIMESCALEDB_PASSWORD=
TIMESCALEDB_URL=

# TimescaleDB Connection Pool Settings
TIMESCALEDB_POOL_SIZE=40
TIMESCALEDB_MAX_OVERFLOW=80
TIMESCALEDB_POOL_TIMEOUT=30
TIMESCALEDB_POOL_RECYCLE=1800

# TimescaleDB SSL Settings (required for production)
TIMESCALEDB_SSL_ENABLED=true
TIMESCALEDB_SSL_CA_PATH=/etc/ssl/certs/timescaledb-ca.crt
TIMESCALEDB_SSL_CERT_PATH=/etc/ssl/certs/timescaledb-client.crt
TIMESCALEDB_SSL_KEY_PATH=/etc/ssl/private/timescaledb-client.key
"""
        else:
            configs['timescaledb'] = f"""# TimescaleDB Database
TIMESCALEDB_HOST=
TIMESCALEDB_PORT=5432
TIMESCALEDB_DB=prometheus_timeseries_{environment.value}
TIMESCALEDB_USER=
TIMESCALEDB_PASSWORD=
TIMESCALEDB_URL=

# TimescaleDB Connection Pool Settings
TIMESCALEDB_POOL_SIZE=20
TIMESCALEDB_MAX_OVERFLOW=40
TIMESCALEDB_POOL_TIMEOUT=30
TIMESCALEDB_POOL_RECYCLE=3600

# TimescaleDB SSL Settings
TIMESCALEDB_SSL_ENABLED=true
TIMESCALEDB_SSL_CA_PATH=/etc/ssl/certs/timescaledb-ca.crt
TIMESCALEDB_SSL_CERT_PATH=/etc/ssl/certs/timescaledb-client.crt
TIMESCALEDB_SSL_KEY_PATH=/etc/ssl/private/timescaledb-client.key
"""
        
        return configs
    
    def _get_service_discovery_config(self, environment: Environment) -> str:
        """Get service discovery configuration"""
        if environment == Environment.DEVELOPMENT:
            return """
# =============================================================================
# SERVICE DISCOVERY CONFIGURATION (CONSUL)
# =============================================================================

# Service discovery settings (disabled for development)
CONSUL_ENABLED=false
CONSUL_HOST=localhost
CONSUL_PORT=8500
CONSUL_TOKEN=
CONSUL_DATACENTER=dc1
CONSUL_ENCRYPTION_ENABLED=false
CONSUL_ACL_ENABLED=false

# Service registration settings
SERVICE_NAME_PREFIX=prometheus-db
HEALTH_CHECK_INTERVAL=30
DEREGISTER_CRITICAL_AFTER=300
"""
        elif environment == Environment.PRODUCTION:
            return """
# =============================================================================
# SERVICE DISCOVERY CONFIGURATION (CONSUL)
# =============================================================================

# Service discovery settings (enabled for production)
CONSUL_ENABLED=true
CONSUL_HOST=
CONSUL_PORT=8500
CONSUL_TOKEN=
CONSUL_DATACENTER=dc1
CONSUL_ENCRYPTION_ENABLED=true
CONSUL_ACL_ENABLED=true

# Service registration settings
SERVICE_NAME_PREFIX=prometheus-db
HEALTH_CHECK_INTERVAL=10
DEREGISTER_CRITICAL_AFTER=60
"""
        else:
            return f"""
# =============================================================================
# SERVICE DISCOVERY CONFIGURATION (CONSUL)
# =============================================================================

# Service discovery settings
CONSUL_ENABLED=true
CONSUL_{environment.value.upper()}_HOST=
CONSUL_{environment.value.upper()}_PORT=8500
CONSUL_{environment.value.upper()}_TOKEN=
CONSUL_DATACENTER={environment.value}
CONSUL_ENCRYPTION_ENABLED=true
CONSUL_ACL_ENABLED=false

# Service registration settings
SERVICE_NAME_PREFIX=prometheus-db-{environment.value}
HEALTH_CHECK_INTERVAL=30
DEREGISTER_CRITICAL_AFTER=180
"""
    
    def _get_credential_management_config(self, environment: Environment) -> str:
        """Get credential management configuration"""
        if environment == Environment.DEVELOPMENT:
            return """
# =============================================================================
# CREDENTIAL MANAGEMENT CONFIGURATION
# =============================================================================

# Credential sources (environment variables for development)
CREDENTIAL_SOURCE=environment
VAULT_ENABLED=false
VAULT_URL=
VAULT_TOKEN=
VAULT_NAMESPACE=
VAULT_MOUNT_PATH=database

# Kubernetes secrets (disabled for development)
KUBERNETES_SECRETS_ENABLED=false
KUBERNETES_NAMESPACE=default

# Secret rotation (disabled for development)
SECRET_ROTATION_ENABLED=false
SECRET_ROTATION_INTERVAL=24h
"""
        elif environment == Environment.PRODUCTION:
            return """
# =============================================================================
# CREDENTIAL MANAGEMENT CONFIGURATION
# =============================================================================

# Credential sources (Vault for production)
CREDENTIAL_SOURCE=vault
VAULT_ENABLED=true
VAULT_URL=
VAULT_TOKEN=
VAULT_NAMESPACE=prometheus
VAULT_MOUNT_PATH=database

# Kubernetes secrets (enabled for production)
KUBERNETES_SECRETS_ENABLED=true
KUBERNETES_NAMESPACE=prometheus-ai

# Secret rotation (enabled for production)
SECRET_ROTATION_ENABLED=true
SECRET_ROTATION_INTERVAL=24h
"""
        else:
            return f"""
# =============================================================================
# CREDENTIAL MANAGEMENT CONFIGURATION
# =============================================================================

# Credential sources (Kubernetes secrets for {environment.value})
CREDENTIAL_SOURCE=kubernetes_secret
VAULT_ENABLED=false
VAULT_{environment.value.upper()}_URL=
VAULT_{environment.value.upper()}_TOKEN=
VAULT_NAMESPACE=prometheus-{environment.value}

# Kubernetes secrets
KUBERNETES_SECRETS_ENABLED=true
KUBERNETES_NAMESPACE=prometheus-ai-{environment.value}

# Secret rotation
SECRET_ROTATION_ENABLED=false
SECRET_ROTATION_INTERVAL=168h
"""
    
    def _get_monitoring_config(self, environment: Environment) -> str:
        """Get monitoring configuration"""
        if environment == Environment.DEVELOPMENT:
            return """
# =============================================================================
# MONITORING AND LOGGING CONFIGURATION
# =============================================================================

# Monitoring settings
MONITORING_ENABLED=true
METRICS_COLLECTION=true
METRICS_ENDPOINT=/metrics
HEALTH_CHECKS=true
HEALTH_CHECK_ENDPOINT=/health

# Performance monitoring
PERFORMANCE_MONITORING=true
SLOW_QUERY_LOGGING=true
SLOW_QUERY_THRESHOLD_MS=1000
CONNECTION_POOL_MONITORING=true

# Logging settings
LOG_LEVEL=DEBUG
STRUCTURED_LOGGING=true
LOG_RETENTION_DAYS=7
ERROR_TRACKING=true
DISTRIBUTED_TRACING=false

# Alerting (disabled for development)
ALERTING_ENABLED=false
ALERT_MANAGER_URL=
"""
        elif environment == Environment.PRODUCTION:
            return """
# =============================================================================
# MONITORING AND LOGGING CONFIGURATION
# =============================================================================

# Monitoring settings
MONITORING_ENABLED=true
METRICS_COLLECTION=true
METRICS_ENDPOINT=/metrics
HEALTH_CHECKS=true
HEALTH_CHECK_ENDPOINT=/health

# Performance monitoring
PERFORMANCE_MONITORING=true
SLOW_QUERY_LOGGING=true
SLOW_QUERY_THRESHOLD_MS=5000
CONNECTION_POOL_MONITORING=true

# Logging settings
LOG_LEVEL=INFO
STRUCTURED_LOGGING=true
LOG_RETENTION_DAYS=30
ERROR_TRACKING=true
DISTRIBUTED_TRACING=true

# Alerting (enabled for production)
ALERTING_ENABLED=true
ALERT_MANAGER_URL=
"""
        else:
            return f"""
# =============================================================================
# MONITORING AND LOGGING CONFIGURATION
# =============================================================================

# Monitoring settings
MONITORING_ENABLED=true
METRICS_COLLECTION=true
METRICS_ENDPOINT=/metrics
HEALTH_CHECKS=true
HEALTH_CHECK_ENDPOINT=/health

# Performance monitoring
PERFORMANCE_MONITORING=true
SLOW_QUERY_LOGGING=true
SLOW_QUERY_THRESHOLD_MS=2000
CONNECTION_POOL_MONITORING=true

# Logging settings
LOG_LEVEL=DEBUG
STRUCTURED_LOGGING=true
LOG_RETENTION_DAYS=14
ERROR_TRACKING=true
DISTRIBUTED_TRACING=true

# Alerting
ALERTING_ENABLED=true
ALERT_MANAGER_{environment.value.upper()}_URL=
"""
    
    def _get_security_config(self, environment: Environment) -> str:
        """Get security configuration"""
        if environment == Environment.DEVELOPMENT:
            return """
# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# SSL/TLS settings (disabled for development)
SSL_ENFORCEMENT=false
CERTIFICATE_VALIDATION=false
CONNECTION_ENCRYPTION=false

# Authentication and authorization (basic for development)
AUTHENTICATION_REQUIRED=true
AUTHORIZATION_REQUIRED=false
ACCESS_CONTROL=basic

# Audit logging (minimal for development)
AUDIT_LOGGING=true
AUDIT_LOG_RETENTION_DAYS=7

# Connection limits (relaxed for development)
MAX_CONNECTIONS_PER_IP=100
MAX_CONNECTIONS_PER_USER=50
RATE_LIMITING=false

# Security scanning (disabled for development)
VULNERABILITY_SCANNING=false
SECURITY_PATCHES=manual
COMPLIANCE_STANDARDS=
"""
        elif environment == Environment.PRODUCTION:
            return """
# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# SSL/TLS settings (required for production)
SSL_ENFORCEMENT=true
CERTIFICATE_VALIDATION=true
CONNECTION_ENCRYPTION=true

# Authentication and authorization (RBAC for production)
AUTHENTICATION_REQUIRED=true
AUTHORIZATION_REQUIRED=true
ACCESS_CONTROL=rbac

# Audit logging (comprehensive for production)
AUDIT_LOGGING=true
AUDIT_LOG_RETENTION_DAYS=90

# Connection limits (strict for production)
MAX_CONNECTIONS_PER_IP=50
MAX_CONNECTIONS_PER_USER=25
RATE_LIMITING=true
RATE_LIMIT_REQUESTS_PER_MINUTE=1000

# Security scanning (enabled for production)
VULNERABILITY_SCANNING=true
SECURITY_PATCHES=auto
COMPLIANCE_STANDARDS=SOC2,GDPR,HIPAA

# IP whitelist (configure as needed)
DATABASE_IP_WHITELIST=
"""
        else:
            return f"""
# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# SSL/TLS settings
SSL_ENFORCEMENT=true
CERTIFICATE_VALIDATION=true
CONNECTION_ENCRYPTION=true

# Authentication and authorization
AUTHENTICATION_REQUIRED=true
AUTHORIZATION_REQUIRED=false
ACCESS_CONTROL=basic

# Audit logging
AUDIT_LOGGING=true
AUDIT_LOG_RETENTION_DAYS=30

# Connection limits
MAX_CONNECTIONS_PER_IP=100
MAX_CONNECTIONS_PER_USER=50
RATE_LIMITING=false
RATE_LIMIT_REQUESTS_PER_MINUTE=2000

# Security scanning
VULNERABILITY_SCANNING=false
SECURITY_PATCHES=manual
COMPLIANCE_STANDARDS=

# IP whitelist
DATABASE_IP_WHITELIST=
"""
    
    def _generate_documentation(self) -> str:
        """Generate comprehensive documentation for environment variables"""
        return """# Database Configuration Environment Variables

This directory contains environment variable templates for database configuration across different environments.

## Files

- `development.env` - Development environment configuration
- `testing.env` - Testing environment configuration  
- `staging.env` - Staging environment configuration
- `production.env` - Production environment configuration
- `validate_env.py` - Environment validation script

## Usage

1. Copy the appropriate template file to `.env` in your project root:
   ```bash
   cp config/database/env_templates/development.env .env
   ```

2. Update the values in `.env` according to your setup

3. Validate your configuration:
   ```bash
   python config/database/env_templates/validate_env.py
   ```

## Environment Variables Reference

### General Configuration

- `ENVIRONMENT` - Current environment (development, testing, staging, production)
- `DB_CONFIG_DIR` - Database configuration directory path
- `DB_ENCRYPTION_KEY` - Encryption key for sensitive data (base64 encoded)
- `DB_ENCRYPTION_PASSWORD` - Password for generating encryption key
- `DB_ENCRYPTION_SALT` - Salt for key derivation

### PostgreSQL Configuration

- `POSTGRES_HOST` - PostgreSQL server hostname
- `POSTGRES_PORT` - PostgreSQL server port (default: 5432)
- `POSTGRES_DB` - Database name
- `POSTGRES_USER` - Database username
- `POSTGRES_PASSWORD` - Database password
- `POSTGRES_URL` - Complete connection URL
- `POSTGRES_POOL_SIZE` - Connection pool size
- `POSTGRES_MAX_OVERFLOW` - Maximum pool overflow
- `POSTGRES_SSL_ENABLED` - Enable SSL connections
- `POSTGRES_SSL_CA_PATH` - CA certificate path
- `POSTGRES_SSL_CERT_PATH` - Client certificate path
- `POSTGRES_SSL_KEY_PATH` - Client private key path

### Redis Configuration

- `REDIS_HOST` - Redis server hostname
- `REDIS_PORT` - Redis server port (default: 6379)
- `REDIS_DB` - Redis database number
- `REDIS_PASSWORD` - Redis password
- `REDIS_URL` - Complete connection URL
- `REDIS_POOL_SIZE` - Connection pool size
- `REDIS_SSL_ENABLED` - Enable SSL connections

### InfluxDB Configuration

- `INFLUXDB_HOST` - InfluxDB server hostname
- `INFLUXDB_PORT` - InfluxDB server port (default: 8086)
- `INFLUXDB_ORG` - InfluxDB organization
- `INFLUXDB_BUCKET` - InfluxDB bucket name
- `INFLUXDB_TOKEN` - InfluxDB authentication token
- `INFLUXDB_URL` - Complete connection URL
- `INFLUXDB_RETENTION` - Data retention policy

### Neo4j Configuration

- `NEO4J_HOST` - Neo4j server hostname
- `NEO4J_PORT` - Neo4j server port (default: 7687)
- `NEO4J_USER` - Neo4j username
- `NEO4J_PASSWORD` - Neo4j password
- `NEO4J_URL` - Complete connection URL
- `NEO4J_POOL_SIZE` - Connection pool size

### TimescaleDB Configuration

- `TIMESCALEDB_HOST` - TimescaleDB server hostname
- `TIMESCALEDB_PORT` - TimescaleDB server port (default: 5432)
- `TIMESCALEDB_DB` - Database name
- `TIMESCALEDB_USER` - Database username
- `TIMESCALEDB_PASSWORD` - Database password
- `TIMESCALEDB_URL` - Complete connection URL

### Service Discovery (Consul)

- `CONSUL_ENABLED` - Enable Consul service discovery
- `CONSUL_HOST` - Consul server hostname
- `CONSUL_PORT` - Consul server port (default: 8500)
- `CONSUL_TOKEN` - Consul authentication token
- `CONSUL_DATACENTER` - Consul datacenter name

### Credential Management

- `CREDENTIAL_SOURCE` - Credential source (environment, kubernetes_secret, vault)
- `VAULT_ENABLED` - Enable HashiCorp Vault integration
- `VAULT_URL` - Vault server URL
- `VAULT_TOKEN` - Vault authentication token
- `KUBERNETES_SECRETS_ENABLED` - Enable Kubernetes secrets
- `KUBERNETES_NAMESPACE` - Kubernetes namespace

### Monitoring and Logging

- `MONITORING_ENABLED` - Enable monitoring
- `METRICS_COLLECTION` - Enable metrics collection
- `HEALTH_CHECKS` - Enable health checks
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `SLOW_QUERY_THRESHOLD_MS` - Slow query threshold in milliseconds

### Security

- `SSL_ENFORCEMENT` - Enforce SSL connections
- `AUTHENTICATION_REQUIRED` - Require authentication
- `AUTHORIZATION_REQUIRED` - Require authorization
- `AUDIT_LOGGING` - Enable audit logging
- `MAX_CONNECTIONS_PER_IP` - Maximum connections per IP address

## Environment-Specific Notes

### Development
- SSL is typically disabled for easier local development
- Connection pools are smaller
- Verbose logging is enabled
- Security restrictions are relaxed

### Testing
- Isolated test databases with separate ports
- Fast setup/teardown configurations
- Minimal resource usage
- No SSL or encryption

### Staging
- Production-like configuration with reduced resources
- SSL enabled with proper certificates
- Enhanced logging for debugging
- Backup and monitoring enabled

### Production
- High availability configurations
- SSL/TLS required for all connections
- Large connection pools for performance
- Comprehensive monitoring and alerting
- Strict security controls
- Automated backups and disaster recovery

## Security Best Practices

1. **Never commit `.env` files to version control**
2. **Use strong, unique passwords for each environment**
3. **Enable SSL/TLS for all production connections**
4. **Rotate credentials regularly**
5. **Use secrets management systems (Vault, Kubernetes secrets)**
6. **Limit connection access by IP address**
7. **Enable audit logging for compliance**
8. **Monitor for suspicious activity**

## Troubleshooting

### Connection Issues
- Verify hostname and port are correct
- Check firewall rules and network connectivity
- Validate credentials
- Review SSL certificate configuration

### Performance Issues
- Adjust connection pool sizes
- Monitor slow query logs
- Check resource utilization
- Review query optimization

### Security Issues
- Verify SSL/TLS configuration
- Check certificate validity
- Review access control settings
- Monitor audit logs

## Support

For additional help with database configuration:
1. Check the application logs for detailed error messages
2. Use the validation script to verify configuration
3. Review the database-specific documentation
4. Contact the development team for environment-specific issues
"""
    
    def _generate_validation_script(self) -> str:
        """Generate environment validation script"""
        return '''#!/usr/bin/env python3
"""
Environment Variable Validation Script
Validates database configuration environment variables for completeness and correctness.
"""

import os
import sys
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class EnvironmentValidator:
    """Validates database environment configuration"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.required_vars = self._get_required_variables()
        self.validation_rules = self._get_validation_rules()
    
    def _get_required_variables(self) -> Dict[str, List[str]]:
        """Get required environment variables by category"""
        return {
            'general': [
                'ENVIRONMENT',
                'DB_CONFIG_DIR'
            ],
            'postgresql': [
                'POSTGRES_HOST',
                'POSTGRES_PORT',
                'POSTGRES_DB',
                'POSTGRES_USER',
                'POSTGRES_PASSWORD'
            ],
            'redis': [
                'REDIS_HOST',
                'REDIS_PORT'
            ],
            'influxdb': [
                'INFLUXDB_HOST',
                'INFLUXDB_PORT',
                'INFLUXDB_ORG',
                'INFLUXDB_BUCKET',
                'INFLUXDB_TOKEN'
            ],
            'neo4j': [
                'NEO4J_HOST',
                'NEO4J_PORT',
                'NEO4J_USER',
                'NEO4J_PASSWORD'
            ],
            'timescaledb': [
                'TIMESCALEDB_HOST',
                'TIMESCALEDB_PORT',
                'TIMESCALEDB_DB',
                'TIMESCALEDB_USER',
                'TIMESCALEDB_PASSWORD'
            ]
        }
    
    def _get_validation_rules(self) -> Dict[str, callable]:
        """Get validation rules for environment variables"""
        return {
            'ENVIRONMENT': lambda x: x in ['development', 'testing', 'staging', 'production'],
            'POSTGRES_PORT': lambda x: self._validate_port(x),
            'REDIS_PORT': lambda x: self._validate_port(x),
            'INFLUXDB_PORT': lambda x: self._validate_port(x),
            'NEO4J_PORT': lambda x: self._validate_port(x),
            'TIMESCALEDB_PORT': lambda x: self._validate_port(x),
            'CONSUL_PORT': lambda x: self._validate_port(x),
            'POSTGRES_URL': lambda x: self._validate_postgres_url(x),
            'REDIS_URL': lambda x: self._validate_redis_url(x),
            'INFLUXDB_URL': lambda x: self._validate_influxdb_url(x),
            'NEO4J_URL': lambda x: self._validate_neo4j_url(x),
            'TIMESCALEDB_URL': lambda x: self._validate_postgres_url(x),
            'SSL_ENFORCEMENT': lambda x: x.lower() in ['true', 'false'],
            'MONITORING_ENABLED': lambda x: x.lower() in ['true', 'false'],
            'CONSUL_ENABLED': lambda x: x.lower() in ['true', 'false'],
        }
    
    def _validate_port(self, port_str: str) -> bool:
        """Validate port number"""
        try:
            port = int(port_str)
            return 1 <= port <= 65535
        except ValueError:
            return False
    
    def _validate_postgres_url(self, url: str) -> bool:
        """Validate PostgreSQL URL format"""
        pattern = r'^postgresql://[^:]+:[^@]+@[^:/]+:\d+/[^/]+$'
        return bool(re.match(pattern, url))
    
    def _validate_redis_url(self, url: str) -> bool:
        """Validate Redis URL format"""
        pattern = r'^redis://(?::[^@]+@)?[^:/]+:\d+(?:/\d+)?$'
        return bool(re.match(pattern, url))
    
    def _validate_influxdb_url(self, url: str) -> bool:
        """Validate InfluxDB URL format"""
        pattern = r'^https?://[^:/]+:\d+$'
        return bool(re.match(pattern, url))
    
    def _validate_neo4j_url(self, url: str) -> bool:
        """Validate Neo4j URL format"""
        pattern = r'^bolt(?:\+s)?://[^:/]+:\d+$'
        return bool(re.match(pattern, url))
    
    def validate_environment(self) -> Tuple[bool, List[str], List[str]]:
        """Validate all environment variables"""
        self.errors = []
        self.warnings = []
        
        environment = os.getenv('ENVIRONMENT', 'development')
        
        # Check required variables
        self._check_required_variables(environment)
        
        # Validate variable formats
        self._validate_variable_formats()
        
        # Check SSL configuration
        self._check_ssl_configuration(environment)
        
        # Check security settings
        self._check_security_settings(environment)
        
        # Check monitoring configuration
        self._check_monitoring_configuration()
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _check_required_variables(self, environment: str):
        """Check if required variables are set"""
        # Always check general variables
        for var in self.required_vars['general']:
            if not os.getenv(var):
                self.errors.append(f"Required variable {var} is not set")
        
        # Check database-specific variables based on what's configured
        for db_type, variables in self.required_vars.items():
            if db_type == 'general':
                continue
            
            # Check if this database type is configured
            host_var = f"{db_type.upper()}_HOST"
            if os.getenv(host_var):
                for var in variables:
                    if not os.getenv(var):
                        self.errors.append(f"Required {db_type} variable {var} is not set")
    
    def _validate_variable_formats(self):
        """Validate variable formats using validation rules"""
        for var_name, validator in self.validation_rules.items():
            value = os.getenv(var_name)
            if value and not validator(value):
                self.errors.append(f"Invalid format for {var_name}: {value}")
    
    def _check_ssl_configuration(self, environment: str):
        """Check SSL configuration consistency"""
        ssl_enforcement = os.getenv('SSL_ENFORCEMENT', 'false').lower() == 'true'
        
        if environment == 'production' and not ssl_enforcement:
            self.errors.append("SSL enforcement is required for production environment")
        
        # Check SSL certificate paths if SSL is enabled
        if ssl_enforcement:
            ssl_vars = [
                'POSTGRES_SSL_CA_PATH',
                'REDIS_SSL_CA_PATH',
                'INFLUXDB_SSL_CA_PATH',
                'NEO4J_SSL_CA_PATH',
                'TIMESCALEDB_SSL_CA_PATH'
            ]
            
            for var in ssl_vars:
                db_type = var.split('_')[0].lower()
                host_var = f"{db_type.upper()}_HOST"
                
                if os.getenv(host_var) and os.getenv(f"{db_type.upper()}_SSL_ENABLED", 'false').lower() == 'true':
                    ca_path = os.getenv(var)
                    if ca_path and not Path(ca_path).exists():
                        self.warnings.append(f"SSL CA certificate not found: {ca_path}")
    
    def _check_security_settings(self, environment: str):
        """Check security settings"""
        if environment == 'production':
            # Check authentication
            auth_required = os.getenv('AUTHENTICATION_REQUIRED', 'false').lower() == 'true'
            if not auth_required:
                self.errors.append("Authentication is required for production environment")
            
            # Check audit logging
            audit_logging = os.getenv('AUDIT_LOGGING', 'false').lower() == 'true'
            if not audit_logging:
                self.warnings.append("Audit logging should be enabled for production")
            
            # Check connection limits
            max_conn_per_ip = os.getenv('MAX_CONNECTIONS_PER_IP')
            if max_conn_per_ip and int(max_conn_per_ip) > 100:
                self.warnings.append("High connection limit per IP may pose security risk")
    
    def _check_monitoring_configuration(self):
        """Check monitoring configuration"""
        monitoring_enabled = os.getenv('MONITORING_ENABLED', 'false').lower() == 'true'
        
        if monitoring_enabled:
            # Check required monitoring variables
            monitoring_vars = [
                'METRICS_ENDPOINT',
                'HEALTH_CHECK_ENDPOINT'
            ]
            
            for var in monitoring_vars:
                if not os.getenv(var):
                    self.warnings.append(f"Monitoring variable {var} should be set when monitoring is enabled")


def main():
    """Main validation function"""
    print("Database Environment Configuration Validator")
    print("=" * 50)
    
    # Load .env file if it exists
    env_file = Path('.env')
    if env_file.exists():
        print(f"Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    else:
        print("No .env file found, using system environment variables")
    
    # Validate environment
    validator = EnvironmentValidator()
    is_valid, errors, warnings = validator.validate_environment()
    
    # Print results
    print(f"\\nValidation Results:")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'not set')}")
    
    if errors:
        print(f"\\n❌ Errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print(f"\\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if is_valid and not warnings:
        print("\\n✅ Configuration is valid!")
    elif is_valid:
        print("\\n✅ Configuration is valid with warnings")
    else:
        print("\\n❌ Configuration has errors")
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()
'''
    
    async def scan_and_generate_from_codebase(self, output_dir: str = None) -> Dict[str, str]:
        """Scan codebase for hardcoded configurations and generate templates"""
        print("Scanning codebase for hardcoded database configurations...")
        
        # Initialize replacer
        replacer = DatabaseConfigReplacer(root_dir=str(self.root_dir))
        
        # Scan for hardcoded configurations
        hardcoded_configs = await replacer.scan_codebase()
        
        print(f"Found {len(hardcoded_configs)} hardcoded configurations")
        
        # Generate templates based on findings
        templates = await self.generate_comprehensive_template(output_dir)
        
        # Generate additional template based on scan results
        if hardcoded_configs:
            scan_based_template = replacer.generate_environment_template(
                hardcoded_configs, Environment.DEVELOPMENT
            )
            
            scan_template_file = Path(output_dir or self.root_dir / "config" / "database" / "env_templates") / "scan_based.env"
            with open(scan_template_file, 'w') as f:
                f.write(scan_based_template)
            
            templates['scan_based'] = str(scan_template_file)
            print(f"Generated scan-based template: {scan_template_file}")
        
        return templates


async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Generate database environment variable templates")
    parser.add_argument("--root-dir", default=".", help="Root directory of the project")
    parser.add_argument("--output-dir", help="Output directory for templates")
    parser.add_argument("--scan-codebase", action="store_true", help="Scan codebase for hardcoded configurations")
    
    args = parser.parse_args()
    
    generator = DatabaseEnvTemplateGenerator(args.root_dir)
    
    if args.scan_codebase:
        templates = await generator.scan_and_generate_from_codebase(args.output_dir)
    else:
        templates = await generator.generate_comprehensive_template(args.output_dir)
    
    print("\\nGenerated templates:")
    for env, path in templates.items():
        print(f"  {env}: {path}")
    
    print("\\nNext steps:")
    print("1. Copy the appropriate template to .env")
    print("2. Update the values according to your setup")
    print("3. Run the validation script to verify configuration")


if __name__ == "__main__":
    asyncio.run(main())