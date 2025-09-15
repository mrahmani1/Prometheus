# Database Configuration Management

## Overview

The Prometheus AI Platform uses a centralized database configuration management system that supports multiple database types across different environments with secure credential management, automatic service discovery, and configuration versioning.

## Supported Database Types

- **PostgreSQL + PostGIS**: Primary relational database with geospatial extensions
- **InfluxDB**: Time-series data for real-time sensor readings and monitoring
- **Neo4j**: Graph database for geological process relationships and causal networks
- **Redis**: Caching and session management
- **TimescaleDB**: High-performance time-series data with SQL interface
- **MongoDB**: Document store for flexible geological data structures
- **Elasticsearch**: Search and analytics engine for geological data
- **Kafka**: Message streaming and event processing

## Environment Configuration

### Development Environment
- Local database instances with minimal security
- Small connection pools for resource efficiency
- Simplified SSL configuration
- Local file-based credential storage

### Staging Environment
- Production-like configuration with reduced scale
- SSL enabled with certificate validation
- Moderate connection pools
- Kubernetes secrets for credential management

### Production Environment
- Full SSL/TLS encryption with mutual authentication
- Large connection pools with overflow handling
- Comprehensive monitoring and alerting
- HashiCorp Vault or Kubernetes secrets for credentials
- Automatic service discovery via Consul

## Configuration Structure

### Basic Configuration Template

```yaml
postgresql:
  host: "${POSTGRES_HOST}"
  port: "${POSTGRES_PORT:5432}"
  database: "${POSTGRES_DB}"
  ssl_enabled: true
  ssl_verify: true
  ssl_ca_path: "${POSTGRES_SSL_CA_PATH}"
  timeout: 30
  read_timeout: 120
  write_timeout: 120
  pool_config:
    min_connections: 10
    max_connections: 50
    max_overflow: 100
    pool_timeout: 60
    pool_recycle: 1800
    pool_pre_ping: true
    retry_attempts: 5
    retry_delay: 2.0
    health_check_interval: 60
  tags:
    environment: production
    purpose: primary_database
    backup_enabled: true
    monitoring_enabled: true
  metadata:
    description: "Primary PostgreSQL database for production"
    maintainer: "prometheus-ops-team"
    backup_schedule: "0 2 * * *"
    monitoring_alerts: true
```

### Environment Variable Substitution

Configuration templates support environment variable substitution with default values:

- `${VAR_NAME}`: Required environment variable
- `${VAR_NAME:default_value}`: Optional with default value

### Connection Pool Configuration

```yaml
pool_config:
  min_connections: 5        # Minimum connections to maintain
  max_connections: 20       # Maximum connections in pool
  max_overflow: 30          # Additional connections beyond max
  pool_timeout: 30          # Seconds to wait for connection
  pool_recycle: 3600        # Seconds before recycling connection
  pool_pre_ping: true       # Test connections before use
  retry_attempts: 3         # Connection retry attempts
  retry_delay: 1.0          # Delay between retries (seconds)
  health_check_interval: 300 # Health check frequency (seconds)
```

## Credential Management

### Environment Variables (Development)

```bash
# PostgreSQL
POSTGRES_USERNAME=prometheus_user
POSTGRES_PASSWORD=secure_password

# InfluxDB
INFLUXDB_TOKEN=your_influxdb_token

# Neo4j
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j_password
```

### Kubernetes Secrets (Staging/Production)

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: postgresql-credentials
  namespace: prometheus-ai
type: Opaque
data:
  username: <base64-encoded-username>
  password: <base64-encoded-password>
```

### HashiCorp Vault (Production)

```bash
# Store credentials in Vault
vault kv put database/production/postgresql \
  username=prometheus_user \
  password=secure_password

# Configure Vault authentication
export VAULT_URL=https://vault.example.com
export VAULT_TOKEN=your_vault_token
```

## SSL/TLS Configuration

### Certificate Paths

```yaml
ssl_enabled: true
ssl_verify: true
ssl_ca_path: "/etc/ssl/certs/ca-certificate.crt"
ssl_cert_path: "/etc/ssl/certs/client-certificate.crt"
ssl_key_path: "/etc/ssl/private/client-private-key.key"
```

### Certificate Management

1. **Development**: Self-signed certificates or SSL disabled
2. **Staging**: CA-signed certificates with verification
3. **Production**: Mutual TLS with certificate rotation

## Service Discovery

### Consul Integration

```yaml
service_discovery:
  enabled: true
  consul_host: "consul.service.consul"
  consul_port: 8500
  consul_token: "${CONSUL_TOKEN}"
  service_name_prefix: "prometheus-db"
  health_check_interval: 30
  deregister_critical_after: 300
```

### Automatic Service Registration

Services are automatically registered with Consul including:
- Service name and tags
- Health check endpoints
- Connection metadata
- Environment information

## Configuration Versioning

### Version Management

- Semantic versioning (major.minor.patch)
- Automatic version increment on updates
- Configuration history tracking
- Rollback capabilities

### Version Operations

```python
# Create new configuration
config = await db_manager.create_database_config(
    name="geological_data",
    db_type=DatabaseType.POSTGRESQL,
    host="postgres.example.com",
    port=5432,
    database="geological_data_prod"
)

# Update configuration (creates new version)
await db_manager.update_config("geological_data", {
    "pool_config": {"max_connections": 100}
})

# Rollback to previous version
await db_manager.rollback_config("geological_data")

# Rollback to specific version
await db_manager.rollback_config("geological_data", version="1.2.3")

# Get configuration history
history = db_manager.get_config_history("geological_data")

# Compare versions
diff = await db_manager.get_configuration_diff("geological_data", "1.2.3", "1.2.4")
```

## Best Practices

### Security

1. **Never store credentials in configuration files**
2. **Use environment variables or secure secret management**
3. **Enable SSL/TLS in staging and production**
4. **Implement certificate rotation**
5. **Use least-privilege database users**
6. **Enable audit logging**

### Performance

1. **Configure appropriate connection pools**
2. **Monitor connection usage**
3. **Implement connection health checks**
4. **Use read replicas for read-heavy workloads**
5. **Enable connection pooling at application level**

### Reliability

1. **Test configurations before deployment**
2. **Implement graceful degradation**
3. **Use circuit breakers for external services**
4. **Monitor database health continuously**
5. **Implement automatic failover**

### Monitoring

1. **Enable comprehensive logging**
2. **Set up alerting for connection failures**
3. **Monitor connection pool utilization**
4. **Track query performance**
5. **Implement health check endpoints**

## Configuration Examples

### Development Environment

```yaml
# config/database/templates/development.yaml
postgresql:
  host: "localhost"
  port: 5432
  database: "prometheus_dev"
  ssl_enabled: false
  pool_config:
    min_connections: 2
    max_connections: 10
```

### Production Environment

```yaml
# config/database/templates/production.yaml
postgresql:
  host: "${POSTGRES_HOST}"
  port: "${POSTGRES_PORT:5432}"
  database: "${POSTGRES_DB}"
  ssl_enabled: true
  ssl_verify: true
  ssl_ca_path: "${POSTGRES_SSL_CA_PATH}"
  pool_config:
    min_connections: 10
    max_connections: 50
    max_overflow: 100
```

## Troubleshooting

### Common Issues

1. **Connection Timeouts**
   - Check network connectivity
   - Verify firewall rules
   - Increase timeout values

2. **SSL Certificate Errors**
   - Verify certificate paths
   - Check certificate validity
   - Ensure CA certificate is trusted

3. **Pool Exhaustion**
   - Increase max_connections
   - Check for connection leaks
   - Monitor connection usage

4. **Authentication Failures**
   - Verify credentials in secret store
   - Check user permissions
   - Validate credential source configuration

### Debugging

```python
# Test specific configuration
success, error = await db_manager.test_connection("geological_data")
if not success:
    print(f"Connection failed: {error}")

# Validate all configurations
results = await db_manager.validate_all_configurations()
for name, (is_valid, errors) in results.items():
    if not is_valid:
        print(f"{name}: {errors}")

# Export configuration for debugging
config_yaml = await db_manager.export_configuration("geological_data", "yaml")
print(config_yaml)
```

## Migration Guide

### Upgrading from Legacy Configuration

1. **Audit existing configurations**
2. **Create configuration templates**
3. **Migrate credentials to secure storage**
4. **Test new configurations**
5. **Deploy with rollback plan**

### Environment Migration

1. **Export configurations from source environment**
2. **Update environment-specific values**
3. **Import to target environment**
4. **Validate and test**
5. **Update service discovery**

## API Reference

### DatabaseConfigManager Methods

- `initialize()`: Initialize configuration manager
- `create_database_config()`: Create new database configuration
- `update_config()`: Update existing configuration
- `rollback_config()`: Rollback to previous version
- `get_config_history()`: Get configuration version history
- `validate_all_configurations()`: Validate all configurations
- `test_all_connections()`: Test all database connections
- `export_configuration()`: Export configuration to file
- `import_configuration()`: Import configuration from file

### Configuration Classes

- `DatabaseConfig`: Main configuration class
- `DatabaseCredentials`: Credential management
- `ConnectionPoolConfig`: Connection pool settings
- `ServiceDiscoveryConfig`: Service discovery settings

## Support

For configuration issues or questions:
1. Check the troubleshooting section
2. Review logs for error details
3. Validate configuration syntax
4. Test connections manually
5. Contact the platform team for assistance