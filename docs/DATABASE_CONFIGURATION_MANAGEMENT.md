# Database Configuration Management System

## Overview

The Database Configuration Management System provides centralized, secure, and environment-aware configuration management for all database connections in the Prometheus AI Platform. This system eliminates hardcoded database configurations and provides production-grade features including secure credential management, automatic service discovery, configuration versioning, and comprehensive validation.

## Features

### Core Features
- **Environment-based Configuration**: Separate configurations for development, testing, staging, and production
- **Secure Credential Management**: Integration with Kubernetes Secrets and HashiCorp Vault
- **Automatic Service Discovery**: Consul integration for dynamic database service discovery
- **Configuration Versioning**: Full versioning and rollback capabilities for all configurations
- **Connection Pool Management**: Advanced connection pooling with health monitoring
- **Multi-Database Support**: PostgreSQL, InfluxDB, Neo4j, Redis, TimescaleDB, MongoDB, Elasticsearch, Kafka

### Advanced Features
- **Configuration Templates**: Environment-specific templates with variable substitution
- **SSL/TLS Support**: Comprehensive SSL configuration with certificate management
- **Health Monitoring**: Continuous health checks and automatic failover
- **Configuration Validation**: Comprehensive validation with detailed error reporting
- **Audit Logging**: Complete audit trail of all configuration changes
- **Documentation Generation**: Automatic generation of configuration documentation

## Architecture

```mermaid
graph TB
    subgraph "Configuration Management Layer"
        A[DatabaseConfigManager]
        B[Configuration Templates]
        C[Environment Variables]
        D[Credential Sources]
    end
    
    subgraph "Credential Sources"
        E[Environment Variables]
        F[Kubernetes Secrets]
        G[HashiCorp Vault]
        H[Configuration Files]
    end
    
    subgraph "Service Layer"
        I[DatabaseService]
        J[Connection Pools]
        K[Health Monitoring]
        L[Legacy Integration]
    end
    
    subgraph "Database Types"
        M[PostgreSQL]
        N[InfluxDB]
        O[Neo4j]
        P[Redis]
        Q[TimescaleDB]
        R[MongoDB]
        S[Elasticsearch]
        T[Kafka]
    end
    
    A --> B
    A --> C
    A --> D
    D --> E
    D --> F
    D --> G
    D --> H
    
    I --> A
    I --> J
    I --> K
    I --> L
    
    J --> M
    J --> N
    J --> O
    J --> P
    J --> Q
    J --> R
    J --> S
    J --> T
```

## Quick Start

### 1. Basic Setup

```python
from backend.core.database_service import get_database_service

# Initialize database service
service = await get_database_service(environment="development")

# Get database connection
async with service.get_database_connection("postgresql") as conn:
    # Use connection
    result = await conn.execute("SELECT 1")
```

### 2. Environment Configuration

Create environment-specific configuration files:

```yaml
# config/database/development.yaml
environment: development
updated_at: "2025-01-14T10:00:00Z"
databases:
  postgresql:
    name: postgresql
    type: postgresql
    host: localhost
    port: 5432
    database: prometheus_dev
    ssl_enabled: false
    pool_config:
      min_connections: 2
      max_connections: 10
```

### 3. Environment Variables

Set up environment variables for sensitive data:

```bash
# Database hosts
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Credentials
export POSTGRES_USERNAME=prometheus
export POSTGRES_PASSWORD=secure_password
export REDIS_PASSWORD=redis_password

# SSL Configuration
export SSL_ENABLED=true
export POSTGRES_SSL_CA_PATH=/etc/ssl/certs/postgres-ca.crt
```

## Configuration Management

### Creating Database Configurations

```python
from backend.core.database_config_manager import (
    DatabaseConfigManager, DatabaseType, Environment
)

# Initialize configuration manager
manager = DatabaseConfigManager(environment=Environment.PRODUCTION)
await manager.initialize()

# Create PostgreSQL configuration
config = await manager.create_database_config(
    name="main_postgres",
    db_type=DatabaseType.POSTGRESQL,
    host="postgres.example.com",
    port=5432,
    database="prometheus_prod",
    ssl_enabled=True,
    ssl_verify=True
)

# Create Redis configuration
redis_config = await manager.create_database_config(
    name="cache_redis",
    db_type=DatabaseType.REDIS,
    host="redis.example.com",
    port=6379,
    ssl_enabled=True
)
```

### Configuration Templates

Templates provide environment-specific defaults:

```yaml
# config/database/templates/production.yaml
postgresql:
  host: "${POSTGRES_HOST}"
  port: "${POSTGRES_PORT:5432}"
  database: "${POSTGRES_DB:prometheus_prod}"
  ssl_enabled: true
  ssl_verify: true
  ssl_ca_path: "${POSTGRES_SSL_CA_PATH:/etc/ssl/certs/postgres-ca.crt}"
  pool_config:
    min_connections: 10
    max_connections: 50
    max_overflow: 100
```

### Environment Variable Substitution

The system supports flexible environment variable substitution:

- `${VAR_NAME}` - Required variable (fails if not set)
- `${VAR_NAME:default}` - Optional variable with default value
- `${VAR_NAME:}` - Optional variable with empty default

## Credential Management

### Environment Variables

```python
# Credentials are automatically loaded from environment variables
# Pattern: {DATABASE_NAME}_{CREDENTIAL_TYPE}
export POSTGRES_USERNAME=prometheus_user
export POSTGRES_PASSWORD=secure_password
export INFLUXDB_TOKEN=influx_auth_token
export NEO4J_USERNAME=neo4j_user
export NEO4J_PASSWORD=neo4j_password
```

### Kubernetes Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: postgres-credentials
  namespace: prometheus-ai
type: Opaque
data:
  username: cHJvbWV0aGV1cw==  # base64 encoded
  password: c2VjdXJlX3Bhc3N3b3Jk  # base64 encoded
```

### HashiCorp Vault

```python
# Vault integration (automatic if configured)
export VAULT_URL=https://vault.example.com
export VAULT_TOKEN=vault_auth_token

# Credentials stored at: database/{environment}/{database_name}
```

## Service Discovery

### Consul Integration

```python
# Enable service discovery
export CONSUL_HOST=consul.example.com
export CONSUL_PORT=8500
export CONSUL_TOKEN=consul_token

# Services are automatically discovered and registered
# Service naming: prometheus-db-{database_name}
```

### Manual Service Registration

```python
# Register database service manually
await manager._register_service(config)

# Services include health checks and metadata
```

## Connection Pool Management

### Pool Configuration

```python
from backend.core.database_config_manager import ConnectionPoolConfig

pool_config = ConnectionPoolConfig(
    min_connections=10,
    max_connections=50,
    max_overflow=100,
    pool_timeout=60,
    pool_recycle=1800,
    pool_pre_ping=True,
    retry_attempts=5,
    retry_delay=2.0,
    health_check_interval=60
)
```

### Health Monitoring

```python
# Get service status
service = await get_database_service()
status = service.get_service_status()

print(f"Total databases: {status['total_databases']}")
print(f"Healthy databases: {status['healthy_databases']}")

# Test specific connection
success, error = await manager.test_connection("postgresql")
if not success:
    print(f"Connection failed: {error}")
```

## SSL/TLS Configuration

### Certificate Management

```python
config = DatabaseConfig(
    name="secure_postgres",
    type=DatabaseType.POSTGRESQL,
    host="secure.postgres.com",
    port=5432,
    ssl_enabled=True,
    ssl_verify=True,
    ssl_ca_path="/etc/ssl/certs/postgres-ca.crt",
    ssl_cert_path="/etc/ssl/certs/postgres-client.crt",
    ssl_key_path="/etc/ssl/private/postgres-client.key"
)
```

### Environment-based SSL

```bash
# SSL Configuration
export SSL_ENABLED=true
export SSL_VERIFY=true

# Certificate paths
export POSTGRES_SSL_CA_PATH=/etc/ssl/certs/postgres-ca.crt
export POSTGRES_SSL_CERT_PATH=/etc/ssl/certs/postgres-client.crt
export POSTGRES_SSL_KEY_PATH=/etc/ssl/private/postgres-client.key
```

## Configuration Versioning

### Version Management

```python
# Update configuration (creates new version)
updated_config = await manager.update_database_config(
    "postgresql",
    port=5433,
    pool_config={'max_connections': 75}
)

print(f"New version: {updated_config.version}")

# View configuration history
history = manager.get_config_history("postgresql")
for config in history:
    print(f"Version {config.version}: {config.updated_at}")

# Rollback to previous version
rolled_back = await manager.rollback_config("postgresql")
print(f"Rolled back to version: {rolled_back.version}")

# Rollback to specific version
specific_rollback = await manager.rollback_config("postgresql", version="1.0.5")
```

## Database-Specific Configuration

### PostgreSQL

```python
postgres_config = await manager.create_database_config(
    name="main_postgres",
    db_type=DatabaseType.POSTGRESQL,
    host="postgres.example.com",
    port=5432,
    database="prometheus",
    schema="public",
    ssl_enabled=True,
    pool_config=ConnectionPoolConfig(
        min_connections=10,
        max_connections=50
    )
)
```

### InfluxDB

```python
influx_config = await manager.create_database_config(
    name="timeseries_influx",
    db_type=DatabaseType.INFLUXDB,
    host="influxdb.example.com",
    port=8086,
    database="geological_data",  # bucket name
    ssl_enabled=True,
    metadata={
        'org': 'prometheus_org',
        'retention_policy': '1y'
    }
)
```

### Neo4j

```python
neo4j_config = await manager.create_database_config(
    name="graph_neo4j",
    db_type=DatabaseType.NEO4J,
    host="neo4j.example.com",
    port=7687,
    database="neo4j",
    ssl_enabled=True,
    pool_config=ConnectionPoolConfig(
        min_connections=5,
        max_connections=25
    )
)
```

### Redis

```python
redis_config = await manager.create_database_config(
    name="cache_redis",
    db_type=DatabaseType.REDIS,
    host="redis.example.com",
    port=6379,
    database="0",
    ssl_enabled=True,
    metadata={
        'persistence_enabled': True
    }
)
```

## Migration from Hardcoded Configurations

### Automated Migration Script

```bash
# Scan for hardcoded configurations
python scripts/replace_hardcoded_db_configs.py \
    --root-dir . \
    --output-report hardcoded_configs.json \
    --output-env-template .env.template

# Dry run to see what would be changed
python scripts/replace_hardcoded_db_configs.py \
    --root-dir . \
    --dry-run \
    --verbose

# Perform actual replacement
python scripts/replace_hardcoded_db_configs.py \
    --root-dir . \
    --output-report replacement_report.json
```

### Manual Migration Steps

1. **Identify Hardcoded Configurations**
   ```bash
   grep -r "localhost:5432" backend/
   grep -r "redis://localhost" backend/
   ```

2. **Replace with Environment Variables**
   ```python
   # Before
   DATABASE_URL = "postgresql://user:pass@localhost:5432/db"
   
   # After
   DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/db')
   ```

3. **Use Configuration Manager**
   ```python
   # Before
   conn = await asyncpg.connect("postgresql://localhost:5432/db")
   
   # After
   service = await get_database_service()
   async with service.get_database_connection("postgresql") as conn:
       # use connection
   ```

## Kubernetes Integration

### Deployment Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus-api
spec:
  template:
    spec:
      containers:
      - name: api
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DB_CONFIG_DIR
          value: "/etc/database-config"
        - name: POSTGRES_HOST
          valueFrom:
            configMapKeyRef:
              name: prometheus-database-config
              key: postgres-host
        - name: POSTGRES_USERNAME
          valueFrom:
            secretKeyRef:
              name: prometheus-database-credentials
              key: postgres-username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: prometheus-database-credentials
              key: postgres-password
        volumeMounts:
        - name: database-config
          mountPath: /etc/database-config
        - name: ssl-certs
          mountPath: /etc/ssl/certs
      volumes:
      - name: database-config
        configMap:
          name: prometheus-database-config
      - name: ssl-certs
        secret:
          secretName: prometheus-database-ssl-certs
```

### ConfigMap and Secrets

```yaml
# ConfigMap for non-sensitive configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-database-config
data:
  environment: production
  postgres-host: postgres.prometheus.svc.cluster.local
  postgres-port: "5432"
  redis-host: redis.prometheus.svc.cluster.local
  redis-port: "6379"

---
# Secret for sensitive credentials
apiVersion: v1
kind: Secret
metadata:
  name: prometheus-database-credentials
type: Opaque
data:
  postgres-username: cHJvbWV0aGV1cw==
  postgres-password: c2VjdXJlX3Bhc3N3b3Jk
  redis-password: cmVkaXNfcGFzcw==
```

## Monitoring and Observability

### Health Checks

```python
# Service-level health check
service = await get_database_service()
health_status = await service.test_all_connections()

for db_name, is_healthy in health_status.items():
    if not is_healthy:
        logger.warning(f"Database {db_name} is unhealthy")

# Individual connection health check
pool = service.get_connection_pool("postgresql")
if pool:
    is_healthy = await pool.health_check()
    print(f"PostgreSQL health: {pool.health_status}")
```

### Metrics Collection

```python
# Get database metrics
service = await get_database_service()
if service.legacy_integrator:
    metrics = service.legacy_integrator.get_database_metrics()
    
    print(f"Total queries: {metrics['query_count']}")
    print(f"Average query time: {metrics['average_query_time']}")
    print(f"Error count: {metrics['error_count']}")
    print(f"Cache hit rate: {metrics['cache_hit_rate']}")
```

### Logging Configuration

```python
import logging

# Configure logging for database operations
logging.getLogger('backend.core.database_config_manager').setLevel(logging.INFO)
logging.getLogger('backend.core.database_service').setLevel(logging.INFO)

# Enable SQL query logging (development only)
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

## Best Practices

### Security

1. **Never store credentials in code or configuration files**
2. **Use environment variables or secure credential stores**
3. **Enable SSL/TLS for all production connections**
4. **Regularly rotate database credentials**
5. **Use least-privilege access principles**

### Performance

1. **Configure appropriate connection pool sizes**
2. **Enable connection pre-ping for reliability**
3. **Set reasonable timeout values**
4. **Monitor connection pool utilization**
5. **Use read replicas for read-heavy workloads**

### Reliability

1. **Implement health checks and monitoring**
2. **Configure automatic failover mechanisms**
3. **Use circuit breakers for external dependencies**
4. **Implement retry logic with exponential backoff**
5. **Maintain configuration backups and versioning**

### Development

1. **Use separate configurations for each environment**
2. **Test configuration changes in staging first**
3. **Document all configuration changes**
4. **Use configuration templates for consistency**
5. **Validate configurations before deployment**

## Troubleshooting

### Common Issues

#### Connection Failures

```python
# Debug connection issues
success, error = await manager.test_connection("postgresql")
if not success:
    print(f"Connection failed: {error}")
    
    # Check configuration
    config = manager.get_database_config("postgresql")
    print(f"Host: {config.host}:{config.port}")
    print(f"Database: {config.database}")
    print(f"SSL enabled: {config.ssl_enabled}")
```

#### SSL Certificate Issues

```bash
# Verify certificate files exist and are readable
ls -la /etc/ssl/certs/postgres-ca.crt
ls -la /etc/ssl/certs/postgres-client.crt
ls -la /etc/ssl/private/postgres-client.key

# Check certificate validity
openssl x509 -in /etc/ssl/certs/postgres-client.crt -text -noout
```

#### Environment Variable Issues

```python
# Debug environment variable resolution
import os

required_vars = [
    'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_USERNAME', 
    'POSTGRES_PASSWORD', 'POSTGRES_DATABASE'
]

for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f"{var}: {value}")
    else:
        print(f"{var}: NOT SET")
```

#### Configuration Validation Errors

```python
# Validate all configurations
results = await manager.validate_all_configs()

for db_name, (is_valid, errors) in results.items():
    if not is_valid:
        print(f"Configuration errors for {db_name}:")
        for error in errors:
            print(f"  - {error}")
```

### Debugging Tools

#### Configuration Inspection

```python
# List all configurations
configs = manager.list_database_configs()
for config in configs:
    print(f"{config.name}: {config.type.value} at {config.host}:{config.port}")

# Get configuration history
history = manager.get_config_history("postgresql")
for config in history:
    print(f"Version {config.version}: {config.updated_at}")
```

#### Service Status

```python
# Get detailed service status
service = await get_database_service()
status = service.get_service_status()

print(json.dumps(status, indent=2, default=str))
```

#### Generate Documentation

```python
# Generate configuration documentation
docs = manager.generate_documentation()
print(docs)

# Save to file
with open('database_config_docs.md', 'w') as f:
    f.write(docs)
```

## API Reference

### DatabaseConfigManager

#### Methods

- `initialize() -> bool`: Initialize the configuration manager
- `create_database_config(name, db_type, host, port, **kwargs) -> DatabaseConfig`: Create new database configuration
- `get_database_config(name) -> Optional[DatabaseConfig]`: Get configuration by name
- `update_database_config(name, **updates) -> Optional[DatabaseConfig]`: Update existing configuration
- `rollback_config(name, version=None) -> Optional[DatabaseConfig]`: Rollback configuration
- `test_connection(name) -> Tuple[bool, Optional[str]]`: Test database connection
- `validate_all_configs() -> Dict[str, Tuple[bool, List[str]]]`: Validate all configurations
- `generate_documentation() -> str`: Generate configuration documentation

### DatabaseService

#### Methods

- `initialize() -> bool`: Initialize the database service
- `get_database_connection(name) -> AsyncContextManager`: Get database connection
- `add_database(name, db_type, host, port, **kwargs) -> bool`: Add new database
- `remove_database(name) -> bool`: Remove database
- `test_all_connections() -> Dict[str, bool]`: Test all connections
- `get_service_status() -> Dict[str, Any]`: Get service status
- `reload_configurations() -> bool`: Reload configurations

### DatabaseConfig

#### Properties

- `name: str`: Database name
- `type: DatabaseType`: Database type
- `host: str`: Database host
- `port: int`: Database port
- `database: Optional[str]`: Database name
- `credentials: Optional[DatabaseCredentials]`: Database credentials
- `ssl_enabled: bool`: SSL enabled flag
- `pool_config: ConnectionPoolConfig`: Connection pool configuration

#### Methods

- `get_connection_string(include_credentials=True) -> str`: Generate connection string
- `validate() -> Tuple[bool, List[str]]`: Validate configuration
- `to_dict() -> Dict[str, Any]`: Convert to dictionary
- `from_dict(data) -> DatabaseConfig`: Create from dictionary

## Environment Variables Reference

### Core Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | `development` | No |
| `DB_CONFIG_DIR` | Configuration directory | `./config/database` | No |
| `DB_ENCRYPTION_KEY` | Encryption key for credentials | Generated | No |

### Database Hosts

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POSTGRES_HOST` | PostgreSQL host | `localhost` | No |
| `POSTGRES_PORT` | PostgreSQL port | `5432` | No |
| `REDIS_HOST` | Redis host | `localhost` | No |
| `REDIS_PORT` | Redis port | `6379` | No |
| `INFLUXDB_HOST` | InfluxDB host | `localhost` | No |
| `INFLUXDB_PORT` | InfluxDB port | `8086` | No |
| `NEO4J_HOST` | Neo4j host | `localhost` | No |
| `NEO4J_PORT` | Neo4j port | `7687` | No |

### Database Credentials

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POSTGRES_USERNAME` | PostgreSQL username | - | Yes* |
| `POSTGRES_PASSWORD` | PostgreSQL password | - | Yes* |
| `POSTGRES_DATABASE` | PostgreSQL database | - | Yes* |
| `REDIS_PASSWORD` | Redis password | - | No |
| `INFLUXDB_TOKEN` | InfluxDB token | - | Yes* |
| `INFLUXDB_ORG` | InfluxDB organization | - | Yes* |
| `NEO4J_USERNAME` | Neo4j username | - | Yes* |
| `NEO4J_PASSWORD` | Neo4j password | - | Yes* |

*Required for production environments

### SSL Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SSL_ENABLED` | Enable SSL globally | `false` | No |
| `SSL_VERIFY` | Verify SSL certificates | `true` | No |
| `POSTGRES_SSL_CA_PATH` | PostgreSQL CA certificate | - | No |
| `POSTGRES_SSL_CERT_PATH` | PostgreSQL client certificate | - | No |
| `POSTGRES_SSL_KEY_PATH` | PostgreSQL client key | - | No |

### Service Discovery

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `CONSUL_HOST` | Consul host | `localhost` | No |
| `CONSUL_PORT` | Consul port | `8500` | No |
| `CONSUL_TOKEN` | Consul token | - | No |

### Vault Integration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `VAULT_URL` | Vault URL | - | No |
| `VAULT_TOKEN` | Vault token | - | No |
| `VAULT_PATH` | Vault secret path | `database` | No |

## Conclusion

The Database Configuration Management System provides a comprehensive, secure, and scalable solution for managing database configurations across all environments. By centralizing configuration management, implementing secure credential handling, and providing advanced features like service discovery and health monitoring, this system ensures reliable and maintainable database connectivity for the Prometheus AI Platform.

For additional support or questions, please refer to the API documentation or contact the development team.