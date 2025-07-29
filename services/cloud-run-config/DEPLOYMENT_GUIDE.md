# Cloud Run Serverless Scaling Deployment Guide

This guide provides comprehensive instructions for deploying the Prometheus AI Geodynamic Platform using Cloud Run with serverless scaling, cost optimization, and multi-region deployment.

## Overview

The Cloud Run serverless scaling implementation provides:

- **Containerized Microservices**: Optimized for Cloud Run deployment with fast cold starts
- **Automatic Scaling**: CPU, memory, and request-based scaling with custom metrics
- **Cost Optimization**: Preemptible instances, scale-to-zero, and intelligent resource allocation
- **Multi-Region Deployment**: Global distribution with traffic routing and load balancing
- **Real-time Monitoring**: Comprehensive monitoring, alerting, and performance tracking

## Architecture Components

### 1. Cloud Run Services
- **helios-core**: AI/ML geological modeling service
- **api-gateway**: High-concurrency API gateway
- **data-ingestion**: Batch data processing service
- **volcano-watchtower**: Real-time volcanic monitoring

### 2. Optimization Components
- **Cloud Run Optimizer**: Intelligent scaling based on workload patterns
- **Cost Optimizer**: Cost reduction through preemptible instances and resource optimization
- **Multi-Region Manager**: Global deployment and traffic routing

### 3. Infrastructure
- **Global Load Balancer**: HTTPS load balancing with SSL termination
- **Monitoring Dashboard**: Real-time metrics and alerting
- **DNS Management**: Global DNS with geographic routing

## Prerequisites

### Required Tools
```bash
# Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Docker
# Install Docker Desktop or Docker Engine

# Terraform (optional)
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Python dependencies
pip install -r services/cloud-run-config/requirements.txt
```

### Google Cloud Setup
```bash
# Set project
export GOOGLE_CLOUD_PROJECT="prometheus-ai-platform"
gcloud config set project $GOOGLE_CLOUD_PROJECT

# Enable APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  dns.googleapis.com \
  compute.googleapis.com

# Create service account for deployment
gcloud iam service-accounts create cloud-run-deployer \
  --display-name="Cloud Run Deployer"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
  --member="serviceAccount:cloud-run-deployer@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
  --member="serviceAccount:cloud-run-deployer@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

## Deployment Methods

### Method 1: Automated Deployment Script

The fastest way to deploy is using the automated deployment script:

```bash
# Make script executable (Linux/Mac)
chmod +x services/cloud-run-config/deploy.sh

# Run full deployment
./services/cloud-run-config/deploy.sh deploy

# Or run specific steps
./services/cloud-run-config/deploy.sh build          # Build images only
./services/cloud-run-config/deploy.sh health-check  # Check service health
./services/cloud-run-config/deploy.sh cost-optimize # Run cost optimization
```

### Method 2: Manual Deployment

#### Step 1: Build and Push Container Images

```bash
# Build optimized images for each service
docker build -t gcr.io/$GOOGLE_CLOUD_PROJECT/helios-core:latest \
  -f services/cloud-run-config/Dockerfile.helios-core-cloudrun \
  services/helios-core/

docker build -t gcr.io/$GOOGLE_CLOUD_PROJECT/api-gateway:latest \
  -f services/cloud-run-config/Dockerfile.api-gateway-cloudrun \
  services/api-gateway/

docker build -t gcr.io/$GOOGLE_CLOUD_PROJECT/data-ingestion:latest \
  -f services/cloud-run-config/Dockerfile.cloud-run-base \
  services/data-ingestion/

docker build -t gcr.io/$GOOGLE_CLOUD_PROJECT/volcano-watchtower:latest \
  -f services/cloud-run-config/Dockerfile.cloud-run-base \
  services/volcano-watchtower/

# Push images to Container Registry
docker push gcr.io/$GOOGLE_CLOUD_PROJECT/helios-core:latest
docker push gcr.io/$GOOGLE_CLOUD_PROJECT/api-gateway:latest
docker push gcr.io/$GOOGLE_CLOUD_PROJECT/data-ingestion:latest
docker push gcr.io/$GOOGLE_CLOUD_PROJECT/volcano-watchtower:latest
```

#### Step 2: Deploy Services to Multiple Regions

```bash
# Deploy Helios Core (AI/ML service)
for region in us-central1 europe-west1 asia-northeast1; do
  gcloud run deploy helios-core \
    --image gcr.io/$GOOGLE_CLOUD_PROJECT/helios-core:latest \
    --platform managed \
    --region $region \
    --allow-unauthenticated \
    --cpu 4 \
    --memory 8Gi \
    --concurrency 10 \
    --timeout 300s \
    --min-instances 0 \
    --max-instances 50 \
    --execution-environment gen2 \
    --no-cpu-throttling \
    --set-env-vars "REGION=$region,ENVIRONMENT=production"
done

# Deploy API Gateway (high-concurrency service)
for region in us-central1 europe-west1 asia-northeast1; do
  gcloud run deploy api-gateway \
    --image gcr.io/$GOOGLE_CLOUD_PROJECT/api-gateway:latest \
    --platform managed \
    --region $region \
    --allow-unauthenticated \
    --cpu 2 \
    --memory 4Gi \
    --concurrency 1000 \
    --timeout 60s \
    --min-instances 1 \
    --max-instances 100 \
    --execution-environment gen2 \
    --cpu-throttling \
    --set-env-vars "REGION=$region,ENVIRONMENT=production"
done

# Deploy Data Ingestion (batch processing)
for region in us-central1 europe-west1; do
  gcloud run deploy data-ingestion \
    --image gcr.io/$GOOGLE_CLOUD_PROJECT/data-ingestion:latest \
    --platform managed \
    --region $region \
    --allow-unauthenticated \
    --cpu 2 \
    --memory 4Gi \
    --concurrency 100 \
    --timeout 600s \
    --min-instances 0 \
    --max-instances 20 \
    --execution-environment gen2 \
    --cpu-throttling \
    --set-env-vars "REGION=$region,ENVIRONMENT=production"
done

# Deploy Volcano Watchtower (real-time monitoring)
for region in us-central1 europe-west1 asia-northeast1; do
  gcloud run deploy volcano-watchtower \
    --image gcr.io/$GOOGLE_CLOUD_PROJECT/volcano-watchtower:latest \
    --platform managed \
    --region $region \
    --allow-unauthenticated \
    --cpu 3 \
    --memory 6Gi \
    --concurrency 50 \
    --timeout 180s \
    --min-instances 1 \
    --max-instances 30 \
    --execution-environment gen2 \
    --cpu-throttling \
    --set-env-vars "REGION=$region,ENVIRONMENT=production"
done
```

#### Step 3: Setup Global Load Balancing

```bash
# Create global static IP
gcloud compute addresses create prometheus-ai-global-ip --global

# Create SSL certificate
gcloud compute ssl-certificates create prometheus-ai-ssl-cert \
  --domains=api.prometheus-ai.com,helios.prometheus-ai.com,data.prometheus-ai.com,volcano.prometheus-ai.com \
  --global

# Create backend services and URL map (detailed configuration in Terraform)
# See services/cloud-run-config/terraform/main.tf for complete setup
```

### Method 3: Terraform Deployment

For infrastructure-as-code deployment:

```bash
cd services/cloud-run-config/terraform

# Initialize Terraform
terraform init

# Copy and customize variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project settings

# Plan deployment
terraform plan

# Apply deployment
terraform apply
```

## Cost Optimization Configuration

### Automatic Cost Optimization

The cost optimizer runs automatically and provides:

```python
# Run cost optimization analysis
python services/cloud-run-config/cost-optimization.py

# Key optimization strategies:
# 1. Preemptible instances for batch workloads (up to 80% for data-ingestion)
# 2. Scale-to-zero for non-latency-sensitive services
# 3. Right-sizing based on actual resource usage
# 4. Regional cost arbitrage
# 5. Concurrency optimization
```

### Manual Cost Configuration

```bash
# Enable preemptible pricing for specific services
gcloud run services update data-ingestion \
  --region us-central1 \
  --execution-environment gen1 \
  --cpu-throttling

# Configure scale-to-zero timeout
gcloud run services update helios-core \
  --region us-central1 \
  --min-instances 0 \
  --max-instances 50
```

## Monitoring and Alerting

### Setup Monitoring Dashboard

```bash
# Create monitoring dashboard
gcloud monitoring dashboards create \
  --config-from-file=services/cloud-run-config/monitoring/alerts.yaml
```

### Key Metrics to Monitor

1. **Performance Metrics**
   - Request latency (target: <2s for real-time services)
   - Error rate (target: <1%)
   - Cold start frequency
   - Instance utilization

2. **Cost Metrics**
   - Hourly/daily spend
   - Cost per request
   - Preemptible instance usage
   - Resource utilization efficiency

3. **Availability Metrics**
   - Service uptime (target: 99.9%)
   - Regional availability
   - Load balancer health

### Alerting Configuration

Key alerts are configured for:
- High response latency (>5s)
- High error rate (>5%)
- High CPU/memory utilization (>80%/90%)
- Cost spikes (>50% increase)
- Service unavailability

## Scaling Configuration

### Service-Specific Scaling Profiles

#### Helios Core (AI/ML Service)
```yaml
cpu: 4000m
memory: 8Gi
concurrency: 10
min_instances: 0
max_instances: 50
scaling_strategy: predictive
timeout: 300s
```

#### API Gateway (High Concurrency)
```yaml
cpu: 2000m
memory: 4Gi
concurrency: 1000
min_instances: 1
max_instances: 100
scaling_strategy: reactive
timeout: 60s
```

#### Data Ingestion (Batch Processing)
```yaml
cpu: 2000m
memory: 4Gi
concurrency: 100
min_instances: 0
max_instances: 20
scaling_strategy: cost_optimized
timeout: 600s
preemptible_percentage: 80%
```

#### Volcano Watchtower (Real-time Monitoring)
```yaml
cpu: 3000m
memory: 6Gi
concurrency: 50
min_instances: 1
max_instances: 30
scaling_strategy: hybrid
timeout: 180s
```

## Testing and Validation

### Health Checks

```bash
# Check service health across all regions
for service in helios-core api-gateway data-ingestion volcano-watchtower; do
  for region in us-central1 europe-west1 asia-northeast1; do
    url=$(gcloud run services describe $service --region $region --format="value(status.url)" 2>/dev/null)
    if [ -n "$url" ]; then
      echo "Testing $service in $region: $url/health"
      curl -f "$url/health" && echo " ✅" || echo " ❌"
    fi
  done
done
```

### Load Testing

```bash
# Install k6 for load testing
curl https://github.com/grafana/k6/releases/download/v0.46.0/k6-v0.46.0-linux-amd64.tar.gz -L | tar xvz --strip-components 1

# Run load test
./k6 run --vus 100 --duration 5m - <<EOF
import http from 'k6/http';
import { check } from 'k6';

export default function() {
  const response = http.get('https://api.prometheus-ai.com/health');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 2s': (r) => r.timings.duration < 2000,
  });
}
EOF
```

### Cost Analysis

```bash
# Run cost optimization analysis
python services/cloud-run-config/cost-optimization.py

# Expected output:
# - Current monthly costs by service and region
# - Optimization recommendations
# - Potential savings (typically 20-40%)
# - Risk assessment for each recommendation
```

## Troubleshooting

### Common Issues

1. **Cold Start Latency**
   ```bash
   # Solution: Increase min-instances for latency-sensitive services
   gcloud run services update volcano-watchtower \
     --region us-central1 \
     --min-instances 2
   ```

2. **High Costs**
   ```bash
   # Solution: Enable more aggressive cost optimization
   python -c "
   from cost_optimization import CloudRunCostOptimizer, CostOptimizationStrategy
   optimizer = CloudRunCostOptimizer('$GOOGLE_CLOUD_PROJECT', 'billing-account')
   optimizer.optimize_all_services(CostOptimizationStrategy.AGGRESSIVE)
   "
   ```

3. **Regional Failures**
   ```bash
   # Solution: Check regional health and redistribute traffic
   gcloud run services describe helios-core --region europe-west1
   ```

### Debugging Commands

```bash
# View service logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=helios-core" --limit 50

# Check service configuration
gcloud run services describe helios-core --region us-central1

# Monitor real-time metrics
gcloud monitoring metrics list --filter="resource.type=cloud_run_revision"
```

## Security Considerations

### IAM and Access Control

```bash
# Create service-specific service accounts
gcloud iam service-accounts create helios-core-sa
gcloud iam service-accounts create volcano-watchtower-sa

# Grant minimal required permissions
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
  --member="serviceAccount:helios-core-sa@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
  --role="roles/monitoring.metricWriter"
```

### Network Security

```bash
# Create VPC connector for private networking
gcloud compute networks vpc-access connectors create prometheus-connector \
  --region us-central1 \
  --subnet prometheus-subnet \
  --subnet-project $GOOGLE_CLOUD_PROJECT

# Update services to use VPC connector
gcloud run services update helios-core \
  --region us-central1 \
  --vpc-connector prometheus-connector \
  --vpc-egress private-ranges-only
```

## Performance Optimization

### Container Optimization

1. **Multi-stage builds** for smaller images
2. **Layer caching** for faster builds
3. **Startup optimization** for reduced cold starts
4. **Resource right-sizing** based on actual usage

### Application Optimization

1. **Connection pooling** for database connections
2. **Caching strategies** for frequently accessed data
3. **Async processing** for I/O operations
4. **Graceful shutdown** handling

## Maintenance and Updates

### Rolling Updates

```bash
# Deploy new version with traffic splitting
gcloud run services update helios-core \
  --image gcr.io/$GOOGLE_CLOUD_PROJECT/helios-core:v2.0 \
  --region us-central1 \
  --tag=v2 \
  --no-traffic

# Gradually shift traffic to new version
gcloud run services update-traffic helios-core \
  --region us-central1 \
  --to-tags=v2=10

# Monitor and complete rollout
gcloud run services update-traffic helios-core \
  --region us-central1 \
  --to-latest
```

### Backup and Recovery

```bash
# Export service configurations
gcloud run services describe helios-core --region us-central1 --format=export > helios-core-backup.yaml

# Restore from backup
gcloud run services replace helios-core-backup.yaml --region us-central1
```

## Cost Estimation

### Expected Monthly Costs (USD)

| Service | Region | Instances | CPU | Memory | Est. Cost |
|---------|--------|-----------|-----|--------|-----------|
| helios-core | us-central1 | 0-50 | 4 vCPU | 8 GB | $200-500 |
| api-gateway | us-central1 | 1-100 | 2 vCPU | 4 GB | $150-400 |
| data-ingestion | us-central1 | 0-20 | 2 vCPU | 4 GB | $50-200 |
| volcano-watchtower | us-central1 | 1-30 | 3 vCPU | 6 GB | $100-300 |

**Total Estimated Range**: $500-1,400/month per region

**With Optimization**: $350-980/month per region (30-40% savings)

## Support and Resources

### Documentation
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Container Optimization Guide](https://cloud.google.com/run/docs/tips/general)
- [Cost Optimization Best Practices](https://cloud.google.com/run/docs/tips/costs)

### Monitoring
- Cloud Run Dashboard: `https://console.cloud.google.com/run`
- Monitoring Dashboard: `https://console.cloud.google.com/monitoring`
- Cost Analysis: `https://console.cloud.google.com/billing`

### Support Channels
- Google Cloud Support (if applicable)
- Community forums and Stack Overflow
- Internal team documentation and runbooks

This deployment guide provides comprehensive instructions for implementing the Cloud Run serverless scaling solution for the Prometheus AI Geodynamic Platform, ensuring optimal performance, cost efficiency, and global availability.