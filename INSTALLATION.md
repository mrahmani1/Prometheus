# Prometheus AI Platform - Installation Guide
## Complete Setup Instructions for Development and Production

---

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+ with WSL2
- **Memory**: 16GB RAM minimum, 32GB+ recommended
- **Storage**: 100GB+ available space
- **CPU**: 8+ cores recommended
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional but recommended for ML workloads)

### Required Software
- **Python**: 3.11 or higher
- **Node.js**: 18.0 or higher
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **Git**: 2.30 or higher

### Optional Tools
- **Kubernetes**: kubectl and minikube/kind for local development
- **Azure CLI**: For cloud deployment
- **NVIDIA Docker**: For GPU support in containers

---

## 🚀 Quick Start (Development)

### 1. Clone Repository
```bash
git clone https://github.com/your-org/prometheus-ai-platform.git
cd prometheus-ai-platform
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (see Configuration section below)
nano .env
```

### 3. Start Development Environment
```bash
# Start all services with Docker Compose
docker-compose up -d

# Wait for services to be ready (2-3 minutes)
docker-compose logs -f

# Verify services are running
docker-compose ps
```

### 4. Access the Platform
- **Main Dashboard**: http://localhost:8080
- **API Documentation**: http://localhost:8080/api/docs
- **InfluxDB UI**: http://localhost:8086
- **Neo4j Browser**: http://localhost:7474
- **Consul UI**: http://localhost:8500

---

## 🔧 Detailed Installation

### Python Backend Setup

#### 1. Create Virtual Environment
```bash
cd backend
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

#### 2. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install multi-cloud ML dependencies for Model Manager
pip install azure-ai-ml azure-storage-blob azure-identity
pip install google-cloud-aiplatform google-cloud-storage
pip install boto3 sagemaker
pip install mlflow
```

#### 3. Database Setup
```bash
# Initialize PostgreSQL database
python scripts/init_database.py

# Run database migrations
python scripts/migrate_database.py

# Load sample data (optional)
python scripts/load_sample_data.py
```

#### 4. Start Backend Services
```bash
# Start main API server
python api.py

# In separate terminals, start additional services:
python -m modules.data_ingestion
python -m modules.volcano_watchtower
python -m modules.ml_processor
```

### Frontend Setup

#### 1. Install Node.js Dependencies
```bash
cd frontend
npm install

# Install development tools
npm install -g @angular/cli  # If using Angular
npm install -g create-react-app  # If using React
```

#### 2. Build Frontend Assets
```bash
# Development build
npm run build:dev

# Production build
npm run build:prod

# Start development server
npm run dev
```

### Database Configuration

#### PostgreSQL + PostGIS
```bash
# Install PostgreSQL and PostGIS
sudo apt-get install postgresql postgresql-contrib postgis

# Create database and user
sudo -u postgres psql
CREATE DATABASE prometheus;
CREATE USER prometheus WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE prometheus TO prometheus;

# Enable PostGIS extension
\c prometheus
CREATE EXTENSION postgis;
CREATE EXTENSION postgis_topology;
```

#### InfluxDB
```bash
# Install InfluxDB 2.x
wget https://dl.influxdata.com/influxdb/releases/influxdb2-2.7.4-linux-amd64.tar.gz
tar xvfz influxdb2-2.7.4-linux-amd64.tar.gz
sudo cp influxdb2-2.7.4-linux-amd64/influxd /usr/local/bin/

# Start InfluxDB
influxd

# Setup initial configuration
influx setup --username admin --password your_password --org prometheus --bucket geological_data
```

#### Neo4j
```bash
# Install Neo4j
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 4.4' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j

# Configure Neo4j
sudo nano /etc/neo4j/neo4j.conf
# Uncomment: dbms.default_listen_address=0.0.0.0

# Start Neo4j
sudo systemctl start neo4j
sudo systemctl enable neo4j
```

#### Redis
```bash
# Install Redis
sudo apt-get install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
# Set: bind 127.0.0.1 ::1
# Set: requirepass your_password

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

---

## 🐳 Docker Installation

### Development Environment
```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose logs -f helios-core
docker-compose logs -f api-gateway

# Stop services
docker-compose down
```

### Production Environment
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start production stack
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale helios-core=3
```

### Custom Docker Build
```bash
# Build backend image
cd backend
docker build -t prometheus/helios-core:latest .

# Build frontend image
cd frontend
docker build -t prometheus/vulcan-ui:latest .

# Run with custom images
docker run -d -p 8000:8000 prometheus/helios-core:latest
docker run -d -p 3000:3000 prometheus/vulcan-ui:latest
```

---

## ☸️ Kubernetes Deployment

### Local Development (minikube)
```bash
# Start minikube with enhanced resources
minikube start --memory=16384 --cpus=8 --disk-size=50g

# Enable addons
minikube addons enable ingress
minikube addons enable dashboard
minikube addons enable metrics-server

# Deploy application with enhanced pipeline
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmaps.yaml
kubectl apply -f kubernetes/secrets.yaml
kubectl apply -f kubernetes/persistent-volumes.yaml
kubectl apply -f kubernetes/database-deployments.yaml
kubectl apply -f kubernetes/consul.yaml
kubectl apply -f kubernetes/monitoring.yaml
kubectl apply -f kubernetes/service-deployments.yaml
kubectl apply -f kubernetes/hpa.yaml
kubectl apply -f kubernetes/ingress.yaml

# Check deployment status
kubectl get pods -n prometheus-ai
kubectl get services -n prometheus-ai
kubectl get hpa -n prometheus-ai
```

### Production Kubernetes with Advanced CI/CD
```bash
# Enhanced deployment script
cd kubernetes/
./deploy.sh  # Automated deployment with health checks

# Or use Helm for advanced deployment
helm upgrade --install prometheus-ai ./helm/prometheus-ai \
  --namespace prometheus-ai-production \
  --create-namespace \
  --set environment=production \
  --set monitoring.enabled=true \
  --set autoscaling.enabled=true \
  --wait --timeout=15m

# Monitor deployment with comprehensive observability
kubectl rollout status deployment/helios-core -n prometheus-ai
kubectl rollout status deployment/api-gateway -n prometheus-ai
kubectl get hpa -n prometheus-ai

# Access services with ingress
# API: https://api.prometheus-ai.com
# Monitoring: https://monitoring.prometheus-ai.com/grafana
# Or port-forward for local access
kubectl port-forward service/api-gateway 8080:3000 -n prometheus-ai
kubectl port-forward service/grafana-service 3000:3000 -n prometheus-ai
```

### Multi-Cloud Deployment
```bash
# Deploy across multiple cloud providers
./scripts/deploy-multi-cloud.sh \
  --gcp-project prometheus-ai-project \
  --azure-resource-group prometheus-ai-rg \
  --aws-region us-west-2

# Configure cross-cloud networking
./scripts/setup-multi-cloud-networking.sh

# Enable disaster recovery
./scripts/setup-disaster-recovery.sh \
  --primary-region us-central1 \
  --secondary-region us-east1
```

---

## ☁️ Azure Deployment

### Azure VM Setup
```bash
# Create resource group
az group create --name prometheus-rg --location eastus

# Create VM
az vm create \
  --resource-group prometheus-rg \
  --name prometheus-vm \
  --image Ubuntu2004 \
  --size Standard_D8s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# Open ports
az vm open-port --resource-group prometheus-rg --name prometheus-vm --port 80
az vm open-port --resource-group prometheus-rg --name prometheus-vm --port 443
az vm open-port --resource-group prometheus-rg --name prometheus-vm --port 8080
```

### Azure Container Instances
```bash
# Create container registry
az acr create --resource-group prometheus-rg --name prometheusregistry --sku Basic

# Build and push images
az acr build --registry prometheusregistry --image helios-core:latest ./backend
az acr build --registry prometheusregistry --image vulcan-ui:latest ./frontend

# Deploy container group
az container create \
  --resource-group prometheus-rg \
  --name prometheus-containers \
  --image prometheusregistry.azurecr.io/helios-core:latest \
  --cpu 4 \
  --memory 8 \
  --ports 8080
```

### Azure Kubernetes Service (AKS)
```bash
# Create AKS cluster
az aks create \
  --resource-group prometheus-rg \
  --name prometheus-aks \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring

# Get credentials
az aks get-credentials --resource-group prometheus-rg --name prometheus-aks

# Deploy application
kubectl apply -f kubernetes/azure/
```

---

## ⚙️ Configuration

### Environment Variables
Create `.env` file in the root directory:

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=prometheus
POSTGRES_USER=prometheus
POSTGRES_PASSWORD=your_password

INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_influx_token
INFLUXDB_ORG=prometheus
INFLUXDB_BUCKET=geological_data

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
API_SECRET_KEY=your_secret_key
JWT_SECRET_KEY=your_jwt_secret

# ML Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# External APIs
USGS_API_KEY=your_usgs_key
NASA_API_KEY=your_nasa_key
COPERNICUS_USERNAME=your_username
COPERNICUS_PASSWORD=your_password

# Monitoring
PROMETHEUS_METRICS_PORT=9090
GRAFANA_ADMIN_PASSWORD=your_grafana_password

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Service Configuration Files

#### `config/helios-core.yaml`
```yaml
model:
  physics_informed: true
  uncertainty_quantification: true
  gpu_acceleration: true
  
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  
data:
  cache_size: 1000
  preprocessing: true
  augmentation: true
```

#### `config/volcano-watchtower.yaml`
```yaml
monitoring:
  update_interval: 60  # seconds
  alert_threshold: 0.8
  data_sources:
    - iris
    - usgs
    - copernicus
    
alerts:
  email_notifications: true
  webhook_url: https://your-webhook-url.com
  
processing:
  real_time: true
  ml_enhancement: true
```

---

## 🧪 Testing

### Unit Tests
```bash
# Backend tests
cd backend
pytest tests/ -v --cov=modules --cov-report=html

# Frontend tests
cd frontend
npm test
npm run test:coverage
```

### Integration Tests
```bash
# Database integration
python -m pytest tests/test_database_integration.py -v

# API integration
python -m pytest tests/test_api_integration.py -v

# End-to-end tests
npm run test:e2e
```

### Performance Tests
```bash
# Load testing with k6
k6 run tests/performance/load_test.js

# Stress testing
k6 run tests/performance/stress_test.js

# Database performance
python tests/performance/db_benchmark.py
```

---

## 🔍 Troubleshooting

### Common Issues

#### Database Connection Errors
```bash
# Check database status
sudo systemctl status postgresql
sudo systemctl status redis-server

# Test connections
psql -h localhost -U prometheus -d prometheus
redis-cli ping

# Reset passwords
sudo -u postgres psql
ALTER USER prometheus PASSWORD 'new_password';
```

#### Docker Issues
```bash
# Clean up Docker
docker system prune -a
docker volume prune

# Rebuild images
docker-compose build --no-cache

# Check logs
docker-compose logs -f service_name
```

#### Memory Issues
```bash
# Check memory usage
free -h
docker stats

# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### GPU Issues
```bash
# Check NVIDIA drivers
nvidia-smi

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Performance Optimization

#### Database Tuning
```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
SELECT pg_reload_conf();
```

#### Python Optimization
```bash
# Install optimized packages
pip install numpy[mkl] scipy[mkl]

# Set environment variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
```

---

## 📊 Advanced Monitoring and Observability

### Multi-Cloud Monitoring Setup
```bash
# Deploy comprehensive monitoring stack
kubectl apply -f kubernetes/monitoring.yaml

# Access monitoring services
kubectl port-forward svc/prometheus-service 9090:9090 -n prometheus-ai
kubectl port-forward svc/grafana-service 3000:3000 -n prometheus-ai

# Grafana credentials: admin/prometheus123 (change in production!)
```

### Prometheus Metrics with Advanced Features
```bash
# Custom geological metrics
curl http://localhost:8080/metrics | grep geological
curl http://localhost:8080/metrics | grep ml_model

# Service mesh metrics (if using Istio)
kubectl port-forward svc/kiali 20001:20001 -n istio-system
```

### Intelligent Alerting with PagerDuty
```bash
# Configure alerting
kubectl apply -f monitoring/alertmanager/

# Test alert routing
curl -X POST http://localhost:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{
    "labels": {
      "alertname": "TestAlert",
      "severity": "critical"
    },
    "annotations": {
      "summary": "Test alert for validation"
    }
  }]'
```

### Distributed Tracing with Jaeger
```bash
# Deploy Jaeger
kubectl apply -f monitoring/jaeger/

# Access Jaeger UI
kubectl port-forward svc/jaeger-query 16686:16686 -n prometheus-ai

# View traces at http://localhost:16686
```

### Centralized Logging with ELK Stack
```bash
# Deploy logging infrastructure
kubectl apply -f monitoring/logging/

# Access Kibana
kubectl port-forward svc/kibana 5601:5601 -n prometheus-ai

# View logs at http://localhost:5601
```

### Cost Optimization Monitoring
```bash
# Deploy cost monitoring
kubectl apply -f monitoring/cost-optimization/

# View cost dashboards in Grafana
# Navigate to "Cost Optimization" dashboard
```

---

## 🔐 Security Configuration

### SSL/TLS Setup
```bash
# Generate certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout prometheus.key -out prometheus.crt

# Configure nginx
sudo cp config/nginx/prometheus.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/prometheus.conf /etc/nginx/sites-enabled/
sudo systemctl reload nginx
```

### Firewall Configuration
```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8080/tcp
sudo ufw enable

# iptables
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
```

---

## 📚 Next Steps

### After Installation
1. **Configure Data Sources**: Set up connections to IRIS, USGS, and other data providers
2. **Load Sample Data**: Import geological datasets for testing
3. **Train ML Models**: Initialize and train physics-informed neural networks
4. **Set Up Monitoring**: Configure alerts and dashboards
5. **User Management**: Create user accounts and set permissions

### Development Workflow
1. **Create Feature Branch**: `git checkout -b feature/new-feature`
2. **Write Tests**: Add unit and integration tests
3. **Run Tests**: `pytest` and `npm test`
4. **Submit PR**: Create pull request for review
5. **Deploy**: Merge to main and deploy to staging/production

### Production Checklist
- [ ] SSL certificates configured
- [ ] Database backups scheduled
- [ ] Monitoring alerts configured
- [ ] Log rotation set up
- [ ] Security scanning enabled
- [ ] Performance testing completed
- [ ] Disaster recovery plan documented
- [ ] User training completed

---

## 📞 Support

### Getting Help
- **Documentation**: https://docs.prometheus-ai.org
- **GitHub Issues**: https://github.com/your-org/prometheus-ai-platform/issues
- **Community Forum**: https://community.prometheus-ai.org
- **Email Support**: support@prometheus-ai.org

### Professional Services
- **Installation Support**: Available for enterprise customers
- **Custom Configuration**: Tailored setup for specific requirements
- **Training Programs**: On-site and remote training available
- **Consulting Services**: Architecture and optimization consulting

---

*Last Updated: January 10, 2025*
*Installation Guide Version: 1.0*

---

## 🔄 Current Development Status

**Task 1: Core Module Integration Framework** is currently **In Progress** with active development. The installation instructions above represent the target architecture for the microservices implementation.

### Current Implementation Status
- **Backend**: Monolithic Flask API in `backend/api.py` with 100+ geological modules in `backend/modules/`
- **Frontend**: Basic HTML/JavaScript components in `frontend/` directory
- **Databases**: Multi-database integration layer implemented in `backend/core/database_integration.py`
- **Security**: Authentication framework in `backend/core/security_integration.py`
- **Data Integration Pipeline**: Advanced data integration pipeline (`backend/modules/data_integration_pipeline.py`) currently in active development

### Active Development Focus
The current development is actively refining:
- **Module Integration**: Unified module loader and dependency management system (In Progress)
- **Advanced Data Integration Pipeline (Task 2.3)**: Multi-source geological data processing with real-time volcanic data enhancement and automated feature engineering (Active Development)
- **Microservices Development**: Active transition to distributed services architecture
- **Container Configuration**: Docker and Kubernetes deployment preparation
- **Performance Optimization**: Enhancing response times and resource utilization

For current development setup, use the existing Flask API:
```bash
cd backend
python api.py  # Start current implementation
```

The Docker Compose and Kubernetes configurations described above are being actively developed for the upcoming microservices architecture.