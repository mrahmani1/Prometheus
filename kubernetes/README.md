# Kubernetes Deployment for Prometheus AI Platform

This directory contains all the Kubernetes manifests and configuration files needed to deploy the Prometheus AI Platform in a production Kubernetes environment.

## Architecture Overview

The deployment includes:

- **Namespace Isolation**: All resources deployed in `prometheus-ai` namespace
- **Multi-Database Support**: PostgreSQL+PostGIS, InfluxDB, Neo4j, Redis, Kafka
- **Microservices**: API Gateway, Helios Core, Data Ingestion, Volcano Watchtower
- **Service Mesh**: Consul Connect for service-to-service communication
- **Monitoring**: Prometheus metrics collection and Grafana dashboards
- **Autoscaling**: Horizontal Pod Autoscalers for all services
- **Ingress**: External access with SSL termination and rate limiting

## Prerequisites

1. **Kubernetes Cluster**: Version 1.24+ with at least 4 nodes
2. **kubectl**: Configured to access your cluster
3. **Ingress Controller**: NGINX ingress controller installed
4. **Cert Manager**: For SSL certificate management (optional)
5. **Storage**: Persistent volume support for databases

## Resource Requirements

### Minimum Cluster Resources
- **CPU**: 20 cores (40 cores with limits)
- **Memory**: 40GB (80GB with limits)
- **Storage**: 140GB persistent storage
- **Nodes**: 4+ nodes for proper distribution

### Per-Service Resources
- **API Gateway**: 250m CPU, 256Mi RAM (scales 2-10 pods)
- **Helios Core**: 1 CPU, 2Gi RAM (scales 3-20 pods)
- **Data Ingestion**: 500m CPU, 1Gi RAM (scales 2-15 pods)
- **Volcano Watchtower**: 500m CPU, 1Gi RAM (scales 2-12 pods)
- **Databases**: 8 CPU, 20Gi RAM total

## Deployment Instructions

### 1. Quick Deployment

```bash
# Clone the repository and navigate to kubernetes directory
cd kubernetes/

# Run the deployment script (Linux/Mac)
./deploy.sh

# Or deploy manually step by step
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f configmaps.yaml
kubectl apply -f postgres-init-configmap.yaml
kubectl apply -f persistent-volumes.yaml
kubectl apply -f database-deployments.yaml
kubectl apply -f consul.yaml
kubectl apply -f monitoring.yaml
kubectl apply -f service-deployments.yaml
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml
```

### 2. Manual Step-by-Step Deployment

#### Step 1: Create Namespace and Basic Resources
```bash
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f configmaps.yaml
kubectl apply -f postgres-init-configmap.yaml
```

#### Step 2: Create Storage
```bash
kubectl apply -f persistent-volumes.yaml

# Wait for PVs to be available
kubectl get pv
```

#### Step 3: Deploy Databases
```bash
kubectl apply -f database-deployments.yaml

# Wait for databases to be ready
kubectl wait --for=condition=available --timeout=300s deployment/postgres-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/influxdb-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/neo4j-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/redis-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/kafka-deployment -n prometheus-ai
```

#### Step 4: Deploy Service Mesh
```bash
kubectl apply -f consul.yaml

# Wait for Consul to be ready
kubectl wait --for=condition=available --timeout=180s deployment/consul-deployment -n prometheus-ai
```

#### Step 5: Deploy Monitoring
```bash
kubectl apply -f monitoring.yaml

# Wait for monitoring to be ready
kubectl wait --for=condition=available --timeout=180s deployment/prometheus-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=180s deployment/grafana-deployment -n prometheus-ai
```

#### Step 6: Deploy Application Services
```bash
kubectl apply -f service-deployments.yaml

# Wait for services to be ready
kubectl wait --for=condition=available --timeout=300s deployment/api-gateway-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/helios-core-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/data-ingestion-deployment -n prometheus-ai
kubectl wait --for=condition=available --timeout=300s deployment/volcano-watchtower-deployment -n prometheus-ai
```

#### Step 7: Configure Autoscaling and Ingress
```bash
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml
```

## Configuration

### Environment Variables

All services are configured through ConfigMaps and Secrets:

- **Database Configuration**: `database-config` ConfigMap
- **Service Configuration**: `service-config` ConfigMap  
- **Monitoring Configuration**: `monitoring-config` ConfigMap
- **Database Secrets**: `database-secrets` Secret
- **JWT Secret**: `jwt-secret` Secret
- **API Keys**: `api-keys` Secret

### Customization

1. **Update Secrets**: Replace base64-encoded values in `secrets.yaml`
2. **Modify Resources**: Adjust CPU/memory limits in deployment files
3. **Scale Configuration**: Update replica counts and HPA settings
4. **Storage**: Modify persistent volume sizes in `persistent-volumes.yaml`
5. **Ingress**: Update hostnames in `ingress.yaml`

## Monitoring and Observability

### Prometheus Metrics

All services expose metrics on `/metrics` endpoint:
- **API Gateway**: Request rates, response times, error rates
- **Helios Core**: Model inference metrics, GPU utilization
- **Data Ingestion**: Data processing rates, queue depths
- **Volcano Watchtower**: Alert frequencies, monitoring status

### Grafana Dashboards

Access Grafana at `https://monitoring.prometheus-ai.com/grafana`:
- **Username**: admin
- **Password**: prometheus123 (change in production!)

Pre-configured dashboards:
- Service health and availability
- Request rates and response times
- Error rates and alerts
- Resource utilization

### Health Checks

All services implement health check endpoints:
- `/health`: Liveness probe (service is running)
- `/ready`: Readiness probe (service is ready to accept traffic)

## Security

### Network Policies

Network policies are configured to:
- Isolate ingress traffic to gateway tier only
- Allow backend services to communicate with databases
- Restrict cross-namespace communication

### Secrets Management

- All passwords and API keys stored in Kubernetes Secrets
- Base64 encoding (replace with proper secret management in production)
- JWT tokens for service authentication
- Database credentials rotation supported

### Service Mesh Security

Consul Connect provides:
- Mutual TLS between services
- Service identity and authorization
- Traffic encryption in transit
- Service discovery with health checking

## Troubleshooting

### Common Issues

1. **Pods Stuck in Pending**: Check resource availability and persistent volumes
2. **Database Connection Errors**: Verify database pods are running and services are accessible
3. **Service Discovery Issues**: Check Consul deployment and service registration
4. **Ingress Not Working**: Verify ingress controller and DNS configuration

### Debugging Commands

```bash
# Check overall status
kubectl get all -n prometheus-ai

# Check pod logs
kubectl logs -f deployment/api-gateway-deployment -n prometheus-ai
kubectl logs -f deployment/helios-core-deployment -n prometheus-ai

# Check service endpoints
kubectl get endpoints -n prometheus-ai

# Check persistent volumes
kubectl get pv,pvc -n prometheus-ai

# Check autoscaling
kubectl get hpa -n prometheus-ai

# Port forward for local access
kubectl port-forward svc/api-gateway-service 3000:3000 -n prometheus-ai
kubectl port-forward svc/grafana-service 3001:3001 -n prometheus-ai
kubectl port-forward svc/prometheus-service 9090:9090 -n prometheus-ai
```

### Performance Tuning

1. **Database Optimization**: Adjust memory settings in database deployments
2. **Service Scaling**: Modify HPA settings based on load patterns
3. **Resource Limits**: Fine-tune CPU/memory requests and limits
4. **Storage Performance**: Use SSD-backed storage classes for databases

## Production Considerations

### Before Production Deployment

1. **Security Hardening**:
   - Replace default passwords and secrets
   - Configure proper RBAC
   - Enable network policies
   - Set up proper certificate management

2. **Monitoring Enhancement**:
   - Configure alerting rules
   - Set up log aggregation
   - Enable distributed tracing
   - Configure backup monitoring

3. **High Availability**:
   - Deploy across multiple availability zones
   - Configure database replication
   - Set up disaster recovery procedures
   - Implement proper backup strategies

4. **Performance Optimization**:
   - Load test all services
   - Optimize database queries
   - Configure caching strategies
   - Tune garbage collection settings

### Maintenance

- **Regular Updates**: Keep all images updated with security patches
- **Backup Strategy**: Regular backups of persistent data
- **Monitoring**: Continuous monitoring of resource usage and performance
- **Scaling**: Monitor and adjust scaling policies based on usage patterns

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review pod logs and events
3. Verify configuration against requirements
4. Check resource availability and limits

This deployment provides a production-ready foundation for the Prometheus AI Platform with proper security, monitoring, and scalability features.