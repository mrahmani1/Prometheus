# Database Configuration Management Best Practices

## Overview

This document outlines best practices for managing database configurations in the Prometheus AI Platform using the standardized configuration management system.

## Configuration Organization

### Environment Separation

1. **Use Environment-Specific Templates**
   ```yaml
   # Development: Minimal security, local resources
   # Staging: Production-like with reduced scale
   # Production: Full security, high availability
   ```

2. **Consistent Naming Conventions**
   ```
   Format: {service}_{environment}_{purpose}
   Examples:
   - geological_data_prod_primary
   - time_series_staging_replica
   - cache_dev_session
   ```

3. **Template Inheritance**
   ```yaml
   # Base template with common settings
   base: &base
     timeout: 30
     pool_config:
       retry_attempts: 3
   
   # Environment-specific overrides
   production:
     <<: *base
     ssl_enabled: true
     pool_config:
       max_connections: 100
   ```

## Security Best Practices

### Credential Management

1. **Never Store Credentials in Configuration Files**
   ```yaml
   # ❌ Wrong
   postgresql:
     username: "myuser"
     password: "mypassword"
   
   # ✅ Correct
   postgresql:
     username: "${POSTGRES_USERNAME}"
     password: "${POSTGRES_PASSWORD}"
   ```

2. **Use Secure Secret Stores**
   - **Development**: Environment variables
   - **Staging/Production**: Kubernetes Secrets or HashiCorp Vault
   - **CI/CD**: Encrypted secret management

3. **Implement Credential Rotation**
   ```python
   # Automated credential rotation
   async def rotate_credentials(config_name: str):
       new_password = generate_secure_password()
       await vault_client.update_secret(config_name, password=new_password)
       await config_manager.reload_credentials(config_name)
   ```

### SSL/TLS Configuration

1. **Enable SSL in All Non-Development Environments**
   ```yaml
   production:
     ssl_enabled: true
     ssl_verify: true
     ssl_ca_path: "/etc/ssl/certs/ca-certificate.crt"
   ```

2. **Use Mutual TLS for High-Security Environments**
   ```yaml
   ssl_cert_path: "/etc/ssl/certs/client-certificate.crt"
   ssl_key_path: "/etc/ssl/private/client-private-key.key"
   ```

3. **Implement Certificate Rotation**
   - Automated certificate renewal
   - Zero-downtime certificate updates
   - Certificate expiration monitoring

## Performance Optimization

### Connection Pool Configuration

1. **Size Pools Based on Workload**
   ```yaml
   # High-throughput service
   pool_config:
     min_connections: 20
     max_connections: 100
     max_overflow: 50
   
   # Low-throughput service
   pool_config:
     min_connections: 2
     max_connections: 10
     max_overflow: 5
   ```

2. **Configure Appropriate Timeouts**
   ```yaml
   # Real-time applications
   timeout: 5
   read_timeout: 10
   write_timeout: 15
   
   # Batch processing
   timeout: 60
   read_timeout: 300
   write_timeout: 300
   ```

3. **Enable Connection Health Checks**
   ```yaml
   pool_config:
     pool_pre_ping: true
     health_check_interval: 60
     pool_recycle: 3600  # 1 hour
   ```

### Database-Specific Optimizations

1. **PostgreSQL**
   ```yaml
   postgresql:
     pool_config:
       min_connections: 10
       max_connections: 50
     # Enable prepared statements
     prepared_statements: true
     # Connection-level settings
     options: "-c shared_preload_libraries=pg_stat_statements"
   ```

2. **InfluxDB**
   ```yaml
   influxdb:
     # Batch writes for better performance
     batch_size: 5000
     flush_interval: 10
     # Retention policy
     retention_policy: "30d"
   ```

3. **Redis**
   ```yaml
   redis:
     # Connection pooling
     pool_config:
       max_connections: 20
     # Pipeline commands
     pipeline_enabled: true
   ```

## Monitoring and Observability

### Health Monitoring

1. **Implement Comprehensive Health Checks**
   ```python
   async def health_check_all_databases():
       results = await config_manager.test_all_connections()
       for name, (success, error) in results.items():
           if not success:
               await alert_manager.send_alert(
                   f"Database {name} health check failed: {error}"
               )
   ```

2. **Monitor Connection Pool Metrics**
   ```yaml
   monitoring:
     metrics:
       - connection_pool_size
       - connection_pool_utilization
       - connection_failures
       - query_latency
   ```

3. **Set Up Alerting**
   ```yaml
   alerts:
     - name: "Database Connection Failure"
       condition: "connection_success_rate < 0.95"
       severity: "critical"
     
     - name: "High Connection Pool Utilization"
       condition: "pool_utilization > 0.80"
       severity: "warning"
   ```

### Logging Best Practices

1. **Structured Logging**
   ```python
   logger.info(
       "Database connection established",
       extra={
           "database_name": config.name,
           "database_type": config.type.value,
           "host": config.host,
           "port": config.port,
           "connection_time_ms": connection_time
       }
   )
   ```

2. **Log Sensitive Data Carefully**
   ```python
   # ❌ Don't log credentials
   logger.info(f"Connecting with password: {password}")
   
   # ✅ Log connection attempts without credentials
   logger.info(f"Connecting to {host}:{port} as {username}")
   ```

## Version Management

### Configuration Versioning

1. **Use Semantic Versioning**
   ```
   Format: MAJOR.MINOR.PATCH
   - MAJOR: Breaking changes
   - MINOR: New features, backward compatible
   - PATCH: Bug fixes, backward compatible
   ```

2. **Maintain Configuration History**
   ```python
   # Keep reasonable history
   await config_manager.cleanup_old_versions(
       config_name, 
       keep_versions=10
   )
   ```

3. **Document Changes**
   ```yaml
   metadata:
     changelog:
       - version: "1.2.0"
         date: "2024-01-15"
         changes: "Added SSL configuration"
         author: "ops-team"
   ```

### Rollback Strategies

1. **Test Before Rollback**
   ```python
   # Validate rollback target
   rollback_config = get_version(target_version)
   is_valid, errors = rollback_config.validate()
   
   if is_valid:
       success, error = await test_connection(rollback_config)
       if success:
           await perform_rollback(target_version)
   ```

2. **Implement Gradual Rollback**
   ```python
   # Canary rollback - test with subset of traffic
   await rollback_canary_instances(target_version)
   await monitor_canary_health()
   await rollback_remaining_instances(target_version)
   ```

## Deployment Strategies

### Blue-Green Deployment

1. **Configuration Validation**
   ```bash
   # Validate new configuration
   python database_config_cli.py --env production validate --name new_config
   
   # Test connections
   python database_config_cli.py --env production test --name new_config
   ```

2. **Staged Deployment**
   ```bash
   # Deploy to blue environment
   kubectl apply -f blue-environment-configs.yaml
   
   # Validate blue environment
   ./scripts/validate-blue-environment.sh
   
   # Switch traffic to blue
   kubectl patch service app-service -p '{"spec":{"selector":{"version":"blue"}}}'
   ```

### Canary Deployment

1. **Gradual Traffic Shift**
   ```yaml
   # Start with 10% traffic
   canary_weight: 0.1
   
   # Monitor metrics and gradually increase
   # 10% -> 25% -> 50% -> 100%
   ```

2. **Automated Rollback on Failure**
   ```python
   async def canary_deployment_monitor():
       while canary_active:
           metrics = await get_canary_metrics()
           if metrics.error_rate > threshold:
               await rollback_canary()
               break
   ```

## Disaster Recovery

### Backup Strategies

1. **Configuration Backup**
   ```bash
   # Export all configurations
   for config in $(python database_config_cli.py list --format json | jq -r '.[].name'); do
       python database_config_cli.py export --name $config --output "backup/${config}.yaml"
   done
   ```

2. **Cross-Region Replication**
   ```yaml
   backup:
     regions:
       - us-west-2
       - us-east-1
     schedule: "0 2 * * *"  # Daily at 2 AM
     retention: "30d"
   ```

### Recovery Procedures

1. **Configuration Recovery**
   ```bash
   # Restore from backup
   python database_config_cli.py import --name restored_config --input backup/config.yaml
   
   # Validate restored configuration
   python database_config_cli.py validate --name restored_config
   ```

2. **Database Failover**
   ```python
   async def database_failover(primary_config: str, replica_config: str):
       # Promote replica to primary
       await promote_replica(replica_config)
       
       # Update configuration to point to new primary
       await config_manager.update_config(
           primary_config,
           {"host": get_replica_host(replica_config)}
       )
   ```

## Testing Strategies

### Configuration Testing

1. **Unit Tests**
   ```python
   def test_configuration_validation():
       config = DatabaseConfig(...)
       is_valid, errors = config.validate()
       assert is_valid
   ```

2. **Integration Tests**
   ```python
   async def test_database_connection():
       config = await config_manager.get_config("test_db")
       success, error = await test_connection(config)
       assert success
   ```

3. **Load Testing**
   ```python
   async def test_connection_pool_under_load():
       tasks = []
       for _ in range(100):
           tasks.append(test_database_query())
       
       results = await asyncio.gather(*tasks)
       success_rate = sum(results) / len(results)
       assert success_rate > 0.95
   ```

### Chaos Engineering

1. **Network Partition Testing**
   ```python
   async def test_network_partition_resilience():
       # Simulate network partition
       await simulate_network_partition()
       
       # Verify graceful degradation
       success, error = await test_connection_with_retry()
       assert success or "timeout" in error.lower()
   ```

2. **Database Failure Simulation**
   ```python
   async def test_database_failure_recovery():
       # Stop database
       await stop_database_container()
       
       # Verify circuit breaker activation
       assert circuit_breaker.is_open()
       
       # Restart database
       await start_database_container()
       
       # Verify recovery
       await wait_for_circuit_breaker_close()
   ```

## Compliance and Governance

### Audit Requirements

1. **Configuration Change Auditing**
   ```python
   audit_log = {
       "timestamp": datetime.utcnow().isoformat(),
       "user": get_current_user(),
       "action": "configuration_update",
       "resource": config_name,
       "changes": configuration_diff,
       "approval_id": approval_ticket_id
   }
   ```

2. **Access Control**
   ```yaml
   rbac:
     roles:
       - name: "config-admin"
         permissions: ["create", "update", "delete", "rollback"]
       - name: "config-viewer"
         permissions: ["read", "export"]
       - name: "config-operator"
         permissions: ["read", "test", "validate"]
   ```

### Compliance Frameworks

1. **SOC 2 Compliance**
   - Encrypted credential storage
   - Access logging and monitoring
   - Change management processes
   - Regular security assessments

2. **GDPR Compliance**
   - Data encryption at rest and in transit
   - Access controls and audit trails
   - Data retention policies
   - Right to erasure procedures

## Troubleshooting Guide

### Common Issues

1. **Connection Pool Exhaustion**
   ```
   Symptoms: "No available connections" errors
   Solution: Increase max_connections or investigate connection leaks
   ```

2. **SSL Certificate Errors**
   ```
   Symptoms: "Certificate verification failed"
   Solution: Check certificate paths and validity
   ```

3. **Authentication Failures**
   ```
   Symptoms: "Access denied" or "Invalid credentials"
   Solution: Verify credentials in secret store
   ```

### Debugging Tools

1. **Configuration Validation**
   ```bash
   python database_config_cli.py validate --name problematic_config
   ```

2. **Connection Testing**
   ```bash
   python database_config_cli.py test --name problematic_config
   ```

3. **Configuration Export for Analysis**
   ```bash
   python database_config_cli.py show --name problematic_config --format yaml
   ```

## Performance Benchmarks

### Expected Performance Metrics

1. **Configuration Operations**
   - Create configuration: < 100ms
   - Update configuration: < 50ms
   - Validate configuration: < 10ms
   - Export configuration: < 20ms

2. **Connection Operations**
   - Connection establishment: < 1s
   - Connection pool checkout: < 10ms
   - Health check: < 5s

3. **Discovery Operations**
   - Network scan (Class C): < 30s
   - Docker discovery: < 5s
   - Kubernetes discovery: < 10s

## Migration Strategies

### Legacy System Migration

1. **Assessment Phase**
   ```bash
   # Audit existing configurations
   python scripts/audit_legacy_configs.py --output assessment.json
   ```

2. **Migration Phase**
   ```bash
   # Convert legacy configurations
   python scripts/migrate_legacy_configs.py --input legacy/ --output standardized/
   ```

3. **Validation Phase**
   ```bash
   # Validate migrated configurations
   python database_config_cli.py validate-all --env production
   ```

### Zero-Downtime Migration

1. **Parallel Configuration**
   ```python
   # Run old and new configurations in parallel
   await deploy_new_configuration()
   await validate_new_configuration()
   await switch_traffic_gradually()
   await decommission_old_configuration()
   ```

2. **Rollback Plan**
   ```python
   # Always have a rollback plan
   if migration_fails():
       await rollback_to_previous_configuration()
       await notify_operations_team()
   ```

This comprehensive best practices guide ensures that database configurations are managed securely, efficiently, and reliably across all environments in the Prometheus AI Platform.