# Deployment Guide - Prometheus AI Platform

## Overview

This comprehensive deployment guide covers the installation, configuration, and deployment of the Prometheus AI Platform across different environments. The platform supports multiple deployment scenarios from single-node development setups to large-scale production clusters.

## 🏗️ Architecture Overview

### System Components

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[NGINX/HAProxy]
    end
    
    subgraph "API Gateway"
        GW[Node.js Express Gateway]
    end
    
    subgraph "Core Services"
        HC[Helios Core - FastAPI]
        VW[Volcano Watchtower - Python]
        DI[Data Ingestion - Python]
        MM[Model Manager - Python]
    end
    
    subgraph "Databases"
        PG[(PostgreSQL + PostGIS)]
        IF[(InfluxDB)]
        NEO[(Neo4j)]
        RD[(Redis)]
    end
    
    subgraph "Message Queue"
        KF[Apache Kafka]
    end
    
    subgraph "Monitoring"
        PR[Prometheus]
        GR[Grafana]
        JG[Jaeger]
    end
    
    LB --> GW
    GW --> HC
    GW --> VW
    GW --> DI
    GW --> MM
    
    HC --> PG
    HC --> IF
    HC --> NEO
    HC --> RD
    
    VW --> KF
    DI --> KF
    
    PR --> GR
    HC --> PR
    VW --> PR
```

### Deployment Patterns

1. **Development Environment**
   - Single-node deployment with Docker Compose
   - Local databases and services
   - Simplified configuration

2. **Staging Environment**
   - Multi-node Kubernetes cluster
   - Shared databases
   - Production-like configuration

3. **Production Environment**
   - High-availability Kubernetes cluster
   - Distributed databases
   - Full monitoring and security

---

## 🐳 Docker Deployment

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 16GB RAM minimum (32GB recommended)
- 100GB available disk space

### Quick Start with Docker Compose

1. **Clone the Repository**
```bash
git clone https://github.com/prometheus-ai/platform.git
cd platform
```

2. **Configure Environment Variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start the Platform**
```bash
docker-compose up -d
```

4. **Verify Deployment**
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Test API endpoints
curl http://localhost:3000/health
```

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  # API Gateway
  api-gateway:
    build: ./services/api-gateway
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - CONSUL_HOST=consul
      - REDIS_URL=redis://redis:6379
    depends_on:
      - consul
      - redis
    networks:
      - prometheus-network

  # Helios Core Service
  helios-core:
    build: ./services/helios-core
    ports:
      - "8001:8001"
    environment:
      - CONSUL_HOST=consul
      - POSTGRES_URL=postgresql://prometheus:password@postgres:5432/prometheus
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
      - consul
    volumes:
      - ./data/models:/app/models
    networks:
      - prometheus-network

  # Volcano Watchtower Service
  volcano-watchtower:
    build: ./services/volcano-watchtower
    ports:
      - "8002:8002"
    environment:
      - CONSUL_HOST=consul
      - INFLUXDB_URL=http://influxdb:8086
      - KAFKA_BROKERS=kafka:9092
    depends_on:
      - influxdb
      - kafka
      - consul
    networks:
      - prometheus-network

  # Data Ingestion Service
  data-ingestion:
    build: ./services/data-ingestion
    ports:
      - "8003:8003"
    environment:
      - CONSUL_HOST=consul
      - KAFKA_BROKERS=kafka:9092
      - POSTGRES_URL=postgresql://prometheus:password@postgres:5432/prometheus
    depends_on:
      - postgres
      - kafka
      - consul
    volumes:
      - ./data/ingestion:/app/data
    networks:
      - prometheus-network

  # Databases
  postgres:
    image: postgis/postgis:15-3.3
    environment:
      - POSTGRES_DB=prometheus
      - POSTGRES_USER=prometheus
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - prometheus-network

  influxdb:
    image: influxdb:2.7
    environment:
      - INFLUXDB_DB=prometheus
      - INFLUXDB_ADMIN_USER=admin
      - INFLUXDB_ADMIN_PASSWORD=password
    volumes:
      - influxdb_data:/var/lib/influxdb2
    ports:
      - "8086:8086"
    networks:
      - prometheus-network

  neo4j:
    image: neo4j:5.12
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
    ports:
      - "7474:7474"
      - "7687:7687"
    networks:
      - prometheus-network

  redis:
    image: redis:7.2-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - prometheus-network

  # Message Queue
  kafka:
    image: confluentinc/cp-kafka:7.4.0
    environment:
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
      - KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1
    depends_on:
      - zookeeper
    volumes:
      - kafka_data:/var/lib/kafka/data
    ports:
      - "9092:9092"
    networks:
      - prometheus-network

  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      - ZOOKEEPER_CLIENT_PORT=2181
      - ZOOKEEPER_TICK_TIME=2000
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
    networks:
      - prometheus-network

  # Service Discovery
  consul:
    image: consul:1.16
    command: consul agent -dev -client=0.0.0.0
    ports:
      - "8500:8500"
    networks:
      - prometheus-network

  # Monitoring
  prometheus:
    image: prom/prometheus:v2.45.0
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - prometheus-network

  grafana:
    image: grafana/grafana:10.0.0
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3001:3000"
    networks:
      - prometheus-network

volumes:
  postgres_data:
  influxdb_data:
  neo4j_data:
  redis_data:
  kafka_data:
  zookeeper_data:
  prometheus_data:
  grafana_data:

networks:
  prometheus-network:
    driver: bridge
```

### Environment Configuration

Create a `.env` file with the following configuration:

```bash
# Environment
NODE_ENV=production
LOG_LEVEL=info

# Security
JWT_SECRET=your-super-secret-jwt-key-change-this
API_KEY_SECRET=your-api-key-secret-change-this

# Database Passwords
POSTGRES_PASSWORD=your-secure-postgres-password
INFLUXDB_ADMIN_PASSWORD=your-secure-influxdb-password
NEO4J_PASSWORD=your-secure-neo4j-password
REDIS_PASSWORD=your-secure-redis-password

# External Services
CONSUL_HOST=consul
CONSUL_PORT=8500

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000

# Data Storage
DATA_VOLUME_PATH=./data
MODEL_STORAGE_PATH=./data/models
BACKUP_PATH=./backups

# Performance Tuning
MAX_WORKERS=4
BATCH_SIZE=1000
CACHE_TTL=3600

# Feature Flags
ENABLE_GPU_ACCELERATION=false
ENABLE_DISTRIBUTED_TRAINING=false
ENABLE_REAL_TIME_PROCESSING=true
```

---

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster 1.25+
- kubectl configured
- Helm 3.0+
- 64GB RAM across cluster (minimum)
- 500GB available storage

### Cluster Requirements

#### Minimum Cluster Specification
- **Master Nodes:** 3 nodes, 4 CPU, 8GB RAM each
- **Worker Nodes:** 5 nodes, 8 CPU, 16GB RAM each
- **Storage:** 500GB SSD storage with ReadWriteMany support
- **Network:** CNI plugin (Calico, Flannel, or Weave)

#### Production Cluster Specification
- **Master Nodes:** 3 nodes, 8 CPU, 16GB RAM each
- **Worker Nodes:** 10+ nodes, 16 CPU, 64GB RAM each
- **Storage:** 2TB+ SSD storage with backup
- **Network:** High-performance CNI with network policies

### Namespace Setup

```bash
# Create namespace
kubectl create namespace prometheus-ai

# Set default namespace
kubectl config set-context --current --namespace=prometheus-ai

# Create service account
kubectl create serviceaccount prometheus-ai-sa

# Apply RBAC
kubectl apply -f kubernetes/rbac.yaml
```

### Storage Configuration

```yaml
# kubernetes/persistent-volumes.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: prometheus-postgres-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: fast-ssd
  hostPath:
    path: /data/postgres

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-postgres-pvc
  namespace: prometheus-ai
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd

---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: prometheus-influxdb-pv
spec:
  capacity:
    storage: 200Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: fast-ssd
  hostPath:
    path: /data/influxdb

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-influxdb-pvc
  namespace: prometheus-ai
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 200Gi
  storageClassName: fast-ssd
```

### ConfigMaps and Secrets

```yaml
# kubernetes/configmaps.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: prometheus-ai
data:
  NODE_ENV: "production"
  LOG_LEVEL: "info"
  CONSUL_HOST: "consul"
  CONSUL_PORT: "8500"
  PROMETHEUS_URL: "http://prometheus:9090"
  GRAFANA_URL: "http://grafana:3000"
  MAX_WORKERS: "8"
  BATCH_SIZE: "2000"
  CACHE_TTL: "3600"
  ENABLE_REAL_TIME_PROCESSING: "true"

---
apiVersion: v1
kind: Secret
metadata:
  name: prometheus-secrets
  namespace: prometheus-ai
type: Opaque
data:
  JWT_SECRET: <base64-encoded-jwt-secret>
  API_KEY_SECRET: <base64-encoded-api-key-secret>
  POSTGRES_PASSWORD: <base64-encoded-postgres-password>
  INFLUXDB_ADMIN_PASSWORD: <base64-encoded-influxdb-password>
  NEO4J_PASSWORD: <base64-encoded-neo4j-password>
  REDIS_PASSWORD: <base64-encoded-redis-password>
```

### Database Deployments

```yaml
# kubernetes/database-deployments.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: prometheus-ai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgis/postgis:15-3.3
        env:
        - name: POSTGRES_DB
          value: "prometheus"
        - name: POSTGRES_USER
          value: "prometheus"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: prometheus-secrets
              key: POSTGRES_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: init-scripts
          mountPath: /docker-entrypoint-initdb.d
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: prometheus-postgres-pvc
      - name: init-scripts
        configMap:
          name: postgres-init-scripts

---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: prometheus-ai
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: influxdb
  namespace: prometheus-ai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: influxdb
  template:
    metadata:
      labels:
        app: influxdb
    spec:
      containers:
      - name: influxdb
        image: influxdb:2.7
        env:
        - name: INFLUXDB_DB
          value: "prometheus"
        - name: INFLUXDB_ADMIN_USER
          value: "admin"
        - name: INFLUXDB_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: prometheus-secrets
              key: INFLUXDB_ADMIN_PASSWORD
        ports:
        - containerPort: 8086
        volumeMounts:
        - name: influxdb-storage
          mountPath: /var/lib/influxdb2
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "16Gi"
            cpu: "8000m"
      volumes:
      - name: influxdb-storage
        persistentVolumeClaim:
          claimName: prometheus-influxdb-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: influxdb
  namespace: prometheus-ai
spec:
  selector:
    app: influxdb
  ports:
  - port: 8086
    targetPort: 8086
  type: ClusterIP
```

### Service Deployments

```yaml
# kubernetes/service-deployments.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: prometheus-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: prometheus-ai/api-gateway:latest
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          valueFrom:
            configMapKeyRef:
              name: prometheus-config
              key: NODE_ENV
        - name: CONSUL_HOST
          valueFrom:
            configMapKeyRef:
              name: prometheus-config
              key: CONSUL_HOST
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: prometheus-secrets
              key: JWT_SECRET
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway
  namespace: prometheus-ai
spec:
  selector:
    app: api-gateway
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: helios-core
  namespace: prometheus-ai
spec:
  replicas: 5
  selector:
    matchLabels:
      app: helios-core
  template:
    metadata:
      labels:
        app: helios-core
    spec:
      containers:
      - name: helios-core
        image: prometheus-ai/helios-core:latest
        ports:
        - containerPort: 8001
        env:
        - name: CONSUL_HOST
          valueFrom:
            configMapKeyRef:
              name: prometheus-config
              key: CONSUL_HOST
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: prometheus-secrets
              key: POSTGRES_PASSWORD
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "16Gi"
            cpu: "8000m"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: prometheus-model-storage-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: helios-core
  namespace: prometheus-ai
spec:
  selector:
    app: helios-core
  ports:
  - port: 8001
    targetPort: 8001
  type: ClusterIP
```

### Horizontal Pod Autoscaling

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
  namespace: prometheus-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: helios-core-hpa
  namespace: prometheus-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: helios-core
  minReplicas: 5
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 75
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 85
```

### Ingress Configuration

```yaml
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: prometheus-ingress
  namespace: prometheus-ai
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.prometheus-ai.com
    secretName: prometheus-tls
  rules:
  - host: api.prometheus-ai.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-gateway
            port:
              number: 80
```

---

## 🚀 Deployment Scripts

### Automated Deployment Script

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

ENVIRONMENT=${1:-development}
NAMESPACE=${2:-prometheus-ai}

echo "🚀 Deploying Prometheus AI Platform to $ENVIRONMENT environment..."

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    echo "❌ Invalid environment. Use: development, staging, or production"
    exit 1
fi

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl."
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo "❌ Helm not found. Please install Helm 3.0+."
    exit 1
fi

# Check cluster connectivity
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create namespace if it doesn't exist
echo "📦 Setting up namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Set context
kubectl config set-context --current --namespace=$NAMESPACE

# Apply RBAC
echo "🔐 Applying RBAC configuration..."
kubectl apply -f kubernetes/rbac.yaml

# Create secrets
echo "🔑 Creating secrets..."
if [[ "$ENVIRONMENT" == "production" ]]; then
    # Production secrets should be created manually or via CI/CD
    echo "⚠️  Production secrets must be created manually"
    echo "   Run: kubectl apply -f kubernetes/secrets-production.yaml"
else
    kubectl apply -f kubernetes/secrets-$ENVIRONMENT.yaml
fi

# Apply ConfigMaps
echo "⚙️  Applying configuration..."
kubectl apply -f kubernetes/configmaps-$ENVIRONMENT.yaml

# Create persistent volumes
echo "💾 Setting up storage..."
kubectl apply -f kubernetes/persistent-volumes.yaml

# Deploy databases
echo "🗄️  Deploying databases..."
kubectl apply -f kubernetes/database-deployments.yaml

# Wait for databases to be ready
echo "⏳ Waiting for databases to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/postgres
kubectl wait --for=condition=available --timeout=300s deployment/influxdb
kubectl wait --for=condition=available --timeout=300s deployment/neo4j
kubectl wait --for=condition=available --timeout=300s deployment/redis

# Deploy services
echo "🔧 Deploying services..."
kubectl apply -f kubernetes/service-deployments.yaml

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
kubectl wait --for=condition=available --timeout=600s deployment/api-gateway
kubectl wait --for=condition=available --timeout=600s deployment/helios-core
kubectl wait --for=condition=available --timeout=600s deployment/volcano-watchtower
kubectl wait --for=condition=available --timeout=600s deployment/data-ingestion

# Apply HPA
if [[ "$ENVIRONMENT" == "production" ]]; then
    echo "📈 Setting up autoscaling..."
    kubectl apply -f kubernetes/hpa.yaml
fi

# Apply ingress
if [[ "$ENVIRONMENT" != "development" ]]; then
    echo "🌐 Setting up ingress..."
    kubectl apply -f kubernetes/ingress-$ENVIRONMENT.yaml
fi

# Deploy monitoring
echo "📊 Deploying monitoring..."
kubectl apply -f kubernetes/monitoring.yaml

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods
kubectl get services
kubectl get ingress

# Run health checks
echo "🏥 Running health checks..."
./scripts/health-checks.sh

echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Deployment Summary:"
echo "   Environment: $ENVIRONMENT"
echo "   Namespace: $NAMESPACE"
echo "   Services: $(kubectl get deployments --no-headers | wc -l)"
echo "   Pods: $(kubectl get pods --no-headers | wc -l)"
echo ""

if [[ "$ENVIRONMENT" == "development" ]]; then
    echo "🔗 Access URLs:"
    echo "   API Gateway: http://localhost:3000"
    echo "   Grafana: http://localhost:3001"
    echo "   Prometheus: http://localhost:9090"
else
    echo "🔗 Access URLs:"
    echo "   API Gateway: https://api.prometheus-ai.com"
    echo "   Grafana: https://grafana.prometheus-ai.com"
    echo "   Prometheus: https://prometheus.prometheus-ai.com"
fi

echo ""
echo "📚 Next steps:"
echo "   1. Configure monitoring alerts"
echo "   2. Set up backup procedures"
echo "   3. Run integration tests"
echo "   4. Configure SSL certificates (if not using cert-manager)"
```

### Health Check Script

```bash
#!/bin/bash
# scripts/health-checks.sh

set -e

NAMESPACE=${1:-prometheus-ai}

echo "🏥 Running health checks for Prometheus AI Platform..."

# Function to check service health
check_service_health() {
    local service=$1
    local port=$2
    local path=${3:-/health}
    
    echo "🔍 Checking $service health..."
    
    # Port forward to service
    kubectl port-forward service/$service $port:$port --namespace=$NAMESPACE &
    PF_PID=$!
    
    # Wait for port forward to establish
    sleep 5
    
    # Check health endpoint
    if curl -f -s http://localhost:$port$path > /dev/null; then
        echo "✅ $service is healthy"
        HEALTH_STATUS=0
    else
        echo "❌ $service health check failed"
        HEALTH_STATUS=1
    fi
    
    # Clean up port forward
    kill $PF_PID 2>/dev/null || true
    
    return $HEALTH_STATUS
}

# Check database connectivity
echo "🗄️  Checking database connectivity..."

# PostgreSQL
if kubectl exec deployment/postgres --namespace=$NAMESPACE -- pg_isready -U prometheus; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL connection failed"
    exit 1
fi

# InfluxDB
if kubectl exec deployment/influxdb --namespace=$NAMESPACE -- influx ping; then
    echo "✅ InfluxDB is ready"
else
    echo "❌ InfluxDB connection failed"
    exit 1
fi

# Redis
if kubectl exec deployment/redis --namespace=$NAMESPACE -- redis-cli ping | grep -q PONG; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis connection failed"
    exit 1
fi

# Check service health endpoints
check_service_health "api-gateway" 3000 "/health"
check_service_health "helios-core" 8001 "/health"
check_service_health "volcano-watchtower" 8002 "/health"
check_service_health "data-ingestion" 8003 "/health"

# Check service discovery
echo "🔍 Checking service discovery..."
if kubectl exec deployment/api-gateway --namespace=$NAMESPACE -- curl -f -s http://consul:8500/v1/catalog/services > /dev/null; then
    echo "✅ Service discovery is working"
else
    echo "❌ Service discovery check failed"
    exit 1
fi

# Check monitoring
echo "📊 Checking monitoring..."
if kubectl exec deployment/prometheus --namespace=$NAMESPACE -- wget -q --spider http://localhost:9090/-/healthy; then
    echo "✅ Prometheus is healthy"
else
    echo "❌ Prometheus health check failed"
    exit 1
fi

# Run integration tests
echo "🧪 Running integration tests..."
kubectl apply -f tests/integration/test-job.yaml --namespace=$NAMESPACE

# Wait for test job to complete
kubectl wait --for=condition=complete --timeout=300s job/integration-tests --namespace=$NAMESPACE

# Check test results
if kubectl logs job/integration-tests --namespace=$NAMESPACE | grep -q "All tests passed"; then
    echo "✅ Integration tests passed"
else
    echo "❌ Integration tests failed"
    kubectl logs job/integration-tests --namespace=$NAMESPACE
    exit 1
fi

# Clean up test job
kubectl delete job/integration-tests --namespace=$NAMESPACE

echo "🎉 All health checks passed!"
```

---

## 🔧 Configuration Management

### Environment-Specific Configurations

#### Development Configuration
```yaml
# kubernetes/configmaps-development.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: prometheus-ai
data:
  NODE_ENV: "development"
  LOG_LEVEL: "debug"
  ENABLE_DEBUG_LOGGING: "true"
  ENABLE_PROFILING: "true"
  MAX_WORKERS: "2"
  BATCH_SIZE: "100"
  CACHE_TTL: "300"
  ENABLE_GPU_ACCELERATION: "false"
  ENABLE_DISTRIBUTED_TRAINING: "false"
```

#### Staging Configuration
```yaml
# kubernetes/configmaps-staging.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: prometheus-ai
data:
  NODE_ENV: "staging"
  LOG_LEVEL: "info"
  ENABLE_DEBUG_LOGGING: "false"
  ENABLE_PROFILING: "true"
  MAX_WORKERS: "4"
  BATCH_SIZE: "500"
  CACHE_TTL: "1800"
  ENABLE_GPU_ACCELERATION: "true"
  ENABLE_DISTRIBUTED_TRAINING: "false"
```

#### Production Configuration
```yaml
# kubernetes/configmaps-production.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: prometheus-ai
data:
  NODE_ENV: "production"
  LOG_LEVEL: "warn"
  ENABLE_DEBUG_LOGGING: "false"
  ENABLE_PROFILING: "false"
  MAX_WORKERS: "8"
  BATCH_SIZE: "2000"
  CACHE_TTL: "3600"
  ENABLE_GPU_ACCELERATION: "true"
  ENABLE_DISTRIBUTED_TRAINING: "true"
  ENABLE_HIGH_AVAILABILITY: "true"
```

### Resource Limits by Environment

#### Development Resources
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

#### Staging Resources
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

#### Production Resources
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "16Gi"
    cpu: "8000m"
```

---

## 🔒 Security Configuration

### Network Policies

```yaml
# kubernetes/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: prometheus-network-policy
  namespace: prometheus-ai
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: prometheus-ai
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: prometheus-ai
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 443
```

### Pod Security Standards

```yaml
# kubernetes/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: prometheus-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

---

## 📊 Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "prometheus_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api-gateway'
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names:
            - prometheus-ai
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        action: keep
        regex: api-gateway

  - job_name: 'helios-core'
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names:
            - prometheus-ai
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        action: keep
        regex: helios-core

  - job_name: 'volcano-watchtower'
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names:
            - prometheus-ai
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        action: keep
        regex: volcano-watchtower

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "Prometheus AI Platform Overview",
    "tags": ["prometheus-ai"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=~\"api-gateway|helios-core|volcano-watchtower|data-ingestion\"}",
            "legendFormat": "{{job}}"
          }
        ]
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{job}} - {{method}}"
          }
        ]
      },
      {
        "id": 3,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

---

## 🚨 Troubleshooting

### Common Deployment Issues

#### Pod Startup Failures
```bash
# Check pod status
kubectl get pods -n prometheus-ai

# Describe problematic pod
kubectl describe pod <pod-name> -n prometheus-ai

# Check logs
kubectl logs <pod-name> -n prometheus-ai --previous

# Check events
kubectl get events -n prometheus-ai --sort-by='.lastTimestamp'
```

#### Service Discovery Issues
```bash
# Check Consul health
kubectl exec deployment/consul -n prometheus-ai -- consul members

# Check service registration
kubectl exec deployment/consul -n prometheus-ai -- consul catalog services

# Test service connectivity
kubectl exec deployment/api-gateway -n prometheus-ai -- curl http://helios-core:8001/health
```

#### Database Connection Issues
```bash
# Test PostgreSQL connection
kubectl exec deployment/postgres -n prometheus-ai -- pg_isready -U prometheus

# Check InfluxDB status
kubectl exec deployment/influxdb -n prometheus-ai -- influx ping

# Test Redis connection
kubectl exec deployment/redis -n prometheus-ai -- redis-cli ping
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n prometheus-ai
kubectl top nodes

# Check HPA status
kubectl get hpa -n prometheus-ai

# Check metrics
kubectl exec deployment/prometheus -n prometheus-ai -- wget -qO- http://localhost:9090/api/v1/query?query=up
```

### Debugging Commands

```bash
# Get all resources
kubectl get all -n prometheus-ai

# Check resource quotas
kubectl describe quota -n prometheus-ai

# Check network policies
kubectl get networkpolicy -n prometheus-ai

# Check persistent volumes
kubectl get pv,pvc -n prometheus-ai

# Check secrets and configmaps
kubectl get secrets,configmaps -n prometheus-ai

# Port forward for debugging
kubectl port-forward service/api-gateway 3000:80 -n prometheus-ai
kubectl port-forward service/grafana 3001:3000 -n prometheus-ai
kubectl port-forward service/prometheus 9090:9090 -n prometheus-ai
```

---

## 📚 Additional Resources

### Documentation Links
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Helm Documentation](https://helm.sh/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)

### Support Channels
- **Technical Support:** support@prometheus-ai.com
- **Deployment Issues:** deployment@prometheus-ai.com
- **Community Forum:** https://community.prometheus-ai.com
- **Emergency Support:** +1-800-PROMETHEUS

### Best Practices
- Always use specific image tags, never `latest` in production
- Implement proper resource limits and requests
- Use health checks and readiness probes
- Implement proper logging and monitoring
- Regular backup and disaster recovery testing
- Security scanning of container images
- Regular updates and patch management

This deployment guide provides comprehensive instructions for deploying the Prometheus AI Platform across different environments. For specific deployment scenarios or custom requirements, please contact our deployment team.