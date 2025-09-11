# CI/CD Pipeline and Advanced Deployment Guide
## Prometheus AI Platform - Production-Grade DevOps Infrastructure

---

## 🚀 Overview

The Prometheus AI Platform implements a comprehensive CI/CD pipeline with advanced deployment strategies, multi-cloud support, and enterprise-grade monitoring. This guide covers the complete DevOps infrastructure from development to production deployment.

## 🏗️ CI/CD Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SOURCE CODE MANAGEMENT                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
│  │   GitHub    │ │   GitLab    │ │  Bitbucket  │ │   Azure     │  │
│  │  Repository │ │  Repository │ │  Repository │ │   DevOps    │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                    CI/CD PIPELINE ORCHESTRATION                │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
    │  │   GitHub    │  │   GitLab    │  │   Google    │           │
    │  │   Actions   │  │     CI      │  │ Cloud Build │           │
    │  └─────────────┘  └─────────────┘  └─────────────┘           │
    └───────────────────────────────────┼───────────────────────────┘
                                        │
┌───────────────────────────────────────┼───────────────────────────────┐
│                    TESTING & QUALITY GATES                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │    Unit     │  │ Integration │  │  Security   │  │    Code     │ │
│  │    Tests    │  │    Tests    │  │  Scanning   │  │   Quality   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└───────────────────────────────────────┼───────────────────────────────┘
                                        │
┌───────────────────────────────────────┼───────────────────────────────┐
│                  CONTAINER REGISTRY & ARTIFACTS                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Docker    │  │   Google    │  │    Azure    │  │     AWS     │ │
│  │     Hub     │  │ Artifact    │  │  Container  │  │     ECR     │ │
│  │             │  │  Registry   │  │  Registry   │  │             │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└───────────────────────────────────────┼───────────────────────────────┘
                                        │
┌───────────────────────────────────────┼───────────────────────────────┐
│                    DEPLOYMENT TARGETS                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Google    │  │    Azure    │  │     AWS     │  │ On-Premise  │ │
│  │     GKE     │  │     AKS     │  │     EKS     │  │ Kubernetes  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Advanced CI/CD Pipeline Configuration

### 1. Multi-Stage Pipeline Architecture

#### GitHub Actions Workflow
```yaml
# .github/workflows/ci-cd-pipeline.yml
name: Prometheus AI Platform CI/CD

on:
  push:
    branches: [main, develop, feature/*]
  pull_request:
    branches: [main, develop]
  release:
    types: [published]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Stage 1: Code Quality and Security
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: SonarQube Scan
        uses: sonarqube-quality-gate-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
      
      - name: Run Snyk Security Scan
        uses: snyk/actions/python@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high
      
      - name: OWASP ZAP Security Scan
        uses: zaproxy/action-full-scan@v0.7.0
        with:
          target: 'http://localhost:8080'

  # Stage 2: Unit and Integration Testing
  test:
    runs-on: ubuntu-latest
    needs: code-quality
    strategy:
      matrix:
        python-version: [3.11, 3.12]
        node-version: [18, 20]
    
    services:
      postgres:
        image: postgis/postgis:15-3.3
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Set up Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      
      - name: Install Python dependencies
        run: |
          pip install -r backend/requirements.txt
          pip install -r backend/requirements-dev.txt
      
      - name: Install Node.js dependencies
        run: |
          cd frontend
          npm ci
      
      - name: Run Python tests
        run: |
          cd backend
          pytest tests/ -v --cov=. --cov-report=xml --cov-report=html
      
      - name: Run Node.js tests
        run: |
          cd frontend
          npm test -- --coverage --watchAll=false
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./backend/coverage.xml,./frontend/coverage/lcov.info

  # Stage 3: Build and Push Container Images
  build:
    runs-on: ubuntu-latest
    needs: test
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-
      
      - name: Build and push Backend image
        id: build-backend
        uses: docker/build-push-action@v5
        with:
          context: ./backend
          push: true
          tags: ${{ steps.meta.outputs.tags }}-backend
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
      
      - name: Build and push Frontend image
        id: build-frontend
        uses: docker/build-push-action@v5
        with:
          context: ./frontend
          push: true
          tags: ${{ steps.meta.outputs.tags }}-frontend
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
      
      - name: Sign container images
        run: |
          cosign sign --yes ${{ steps.meta.outputs.tags }}-backend
          cosign sign --yes ${{ steps.meta.outputs.tags }}-frontend

  # Stage 4: Infrastructure as Code
  infrastructure:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.6.0
      
      - name: Terraform Init
        run: terraform init
        working-directory: ./infrastructure
      
      - name: Terraform Plan
        run: terraform plan -out=tfplan
        working-directory: ./infrastructure
        env:
          TF_VAR_image_tag: ${{ needs.build.outputs.image-tag }}
      
      - name: Terraform Apply
        if: github.event_name == 'push'
        run: terraform apply -auto-approve tfplan
        working-directory: ./infrastructure

  # Stage 5: Deploy to Staging
  deploy-staging:
    runs-on: ubuntu-latest
    needs: [build, infrastructure]
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}
      
      - name: Deploy to staging
        run: |
          helm upgrade --install prometheus-ai ./helm/prometheus-ai \
            --namespace prometheus-ai-staging \
            --create-namespace \
            --set image.tag=${{ needs.build.outputs.image-tag }} \
            --set environment=staging \
            --wait --timeout=10m
      
      - name: Run smoke tests
        run: |
          kubectl wait --for=condition=available --timeout=300s \
            deployment/api-gateway -n prometheus-ai-staging
          ./scripts/smoke-tests.sh staging

  # Stage 6: Deploy to Production
  deploy-production:
    runs-on: ubuntu-latest
    needs: [build, deploy-staging]
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG_PRODUCTION }}
      
      - name: Blue-Green Deployment
        run: |
          # Deploy to green environment
          helm upgrade --install prometheus-ai-green ./helm/prometheus-ai \
            --namespace prometheus-ai-production \
            --set image.tag=${{ needs.build.outputs.image-tag }} \
            --set environment=production \
            --set deployment.color=green \
            --wait --timeout=15m
          
          # Run production validation tests
          ./scripts/production-validation.sh green
          
          # Switch traffic to green
          kubectl patch service api-gateway-service \
            -n prometheus-ai-production \
            -p '{"spec":{"selector":{"deployment":"green"}}}'
          
          # Wait and validate
          sleep 60
          ./scripts/production-validation.sh green
          
          # Clean up blue environment
          helm uninstall prometheus-ai-blue -n prometheus-ai-production || true

  # Stage 7: Post-Deployment Monitoring
  post-deployment:
    runs-on: ubuntu-latest
    needs: deploy-production
    if: always()
    
    steps:
      - name: Update deployment status
        uses: actions/github-script@v7
        with:
          script: |
            const { data: deployment } = await github.rest.repos.createDeploymentStatus({
              owner: context.repo.owner,
              repo: context.repo.repo,
              deployment_id: context.payload.deployment.id,
              state: '${{ job.status }}',
              environment_url: 'https://api.prometheus-ai.com',
              description: 'Deployment completed'
            });
      
      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          channel: '#deployments'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### 2. GitLab CI Configuration
```yaml
# .gitlab-ci.yml
stages:
  - validate
  - test
  - build
  - security
  - deploy-staging
  - deploy-production
  - monitor

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  REGISTRY: $CI_REGISTRY
  IMAGE_NAME: $CI_PROJECT_PATH

# Code Quality and Validation
code-quality:
  stage: validate
  image: sonarqube/sonar-scanner-cli:latest
  script:
    - sonar-scanner -Dsonar.projectKey=$CI_PROJECT_NAME
  only:
    - merge_requests
    - main
    - develop

# Comprehensive Testing
test-backend:
  stage: test
  image: python:3.11
  services:
    - postgres:15
    - redis:7-alpine
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: test_user
    POSTGRES_PASSWORD: test_pass
  script:
    - cd backend
    - pip install -r requirements.txt -r requirements-dev.txt
    - pytest tests/ -v --cov=. --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: backend/coverage.xml

test-frontend:
  stage: test
  image: node:18
  script:
    - cd frontend
    - npm ci
    - npm run test:coverage
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: frontend/coverage/cobertura-coverage.xml

# Multi-stage Docker Build
build-images:
  stage: build
  image: docker:24-dind
  services:
    - docker:24-dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - |
      # Build backend image
      docker build -t $REGISTRY/$IMAGE_NAME/backend:$CI_COMMIT_SHA ./backend
      docker push $REGISTRY/$IMAGE_NAME/backend:$CI_COMMIT_SHA
      
      # Build frontend image
      docker build -t $REGISTRY/$IMAGE_NAME/frontend:$CI_COMMIT_SHA ./frontend
      docker push $REGISTRY/$IMAGE_NAME/frontend:$CI_COMMIT_SHA
      
      # Tag latest for main branch
      if [ "$CI_COMMIT_REF_NAME" = "main" ]; then
        docker tag $REGISTRY/$IMAGE_NAME/backend:$CI_COMMIT_SHA $REGISTRY/$IMAGE_NAME/backend:latest
        docker tag $REGISTRY/$IMAGE_NAME/frontend:$CI_COMMIT_SHA $REGISTRY/$IMAGE_NAME/frontend:latest
        docker push $REGISTRY/$IMAGE_NAME/backend:latest
        docker push $REGISTRY/$IMAGE_NAME/frontend:latest
      fi

# Security Scanning
security-scan:
  stage: security
  image: owasp/zap2docker-stable
  script:
    - zap-baseline.py -t http://localhost:8080 -J zap-report.json
  artifacts:
    reports:
      sast: zap-report.json
  allow_failure: true

# Staging Deployment
deploy-staging:
  stage: deploy-staging
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: https://staging.prometheus-ai.com
  script:
    - kubectl config use-context staging
    - |
      helm upgrade --install prometheus-ai-staging ./helm/prometheus-ai \
        --namespace prometheus-ai-staging \
        --create-namespace \
        --set image.backend.tag=$CI_COMMIT_SHA \
        --set image.frontend.tag=$CI_COMMIT_SHA \
        --set environment=staging \
        --wait --timeout=10m
  only:
    - develop

# Production Deployment with Canary
deploy-production:
  stage: deploy-production
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://api.prometheus-ai.com
  script:
    - kubectl config use-context production
    - |
      # Canary deployment (10% traffic)
      helm upgrade --install prometheus-ai-canary ./helm/prometheus-ai \
        --namespace prometheus-ai-production \
        --set image.backend.tag=$CI_COMMIT_SHA \
        --set image.frontend.tag=$CI_COMMIT_SHA \
        --set environment=production \
        --set deployment.strategy=canary \
        --set deployment.canaryWeight=10 \
        --wait --timeout=15m
      
      # Wait for validation
      sleep 300
      
      # Full deployment if canary succeeds
      helm upgrade prometheus-ai-canary ./helm/prometheus-ai \
        --namespace prometheus-ai-production \
        --set deployment.canaryWeight=100 \
        --wait --timeout=10m
  when: manual
  only:
    - main
```

---

## 🔍 Comprehensive Monitoring and Observability

### 1. Multi-Cloud Monitoring Setup

#### Prometheus Configuration
```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'prometheus-ai-production'
    region: 'multi-cloud'

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Kubernetes API Server
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
    - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
      action: keep
      regex: default;kubernetes;https

  # Application Services
  - job_name: 'prometheus-ai-services'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - prometheus-ai-production
        - prometheus-ai-staging
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)

  # Google Cloud Monitoring
  - job_name: 'gcp-monitoring'
    gce_sd_configs:
    - project: 'prometheus-ai-project'
      zone: 'us-central1-a'
    relabel_configs:
    - source_labels: [__meta_gce_label_monitoring]
      action: keep
      regex: enabled

  # Azure Monitor Integration
  - job_name: 'azure-monitoring'
    azure_sd_configs:
    - subscription_id: 'your-subscription-id'
      resource_group: 'prometheus-ai-rg'
    relabel_configs:
    - source_labels: [__meta_azure_machine_tag_monitoring]
      action: keep
      regex: enabled

  # AWS CloudWatch Integration
  - job_name: 'aws-monitoring'
    ec2_sd_configs:
    - region: us-west-2
      port: 9100
    relabel_configs:
    - source_labels: [__meta_ec2_tag_Monitoring]
      action: keep
      regex: enabled
```

#### Advanced Alerting Rules
```yaml
# monitoring/prometheus/rules/geological-alerts.yml
groups:
- name: geological-system-alerts
  rules:
  - alert: HighSeismicActivity
    expr: increase(seismic_events_total[5m]) > 10
    for: 2m
    labels:
      severity: warning
      service: volcano-watchtower
    annotations:
      summary: "High seismic activity detected"
      description: "{{ $value }} seismic events detected in the last 5 minutes"

  - alert: MLModelDrift
    expr: ml_model_accuracy < 0.85
    for: 10m
    labels:
      severity: critical
      service: helios-core
    annotations:
      summary: "ML model performance degradation"
      description: "Model accuracy dropped to {{ $value }}"

  - alert: DataIngestionFailure
    expr: rate(data_ingestion_errors_total[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
      service: data-ingestion
    annotations:
      summary: "Data ingestion failure rate high"
      description: "Error rate: {{ $value }} errors per second"

- name: infrastructure-alerts
  rules:
  - alert: HighCPUUsage
    expr: (100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is {{ $value }}% on {{ $labels.instance }}"

  - alert: HighMemoryUsage
    expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value }}% on {{ $labels.instance }}"

  - alert: DatabaseConnectionFailure
    expr: up{job="postgres-exporter"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failure"
      description: "Cannot connect to PostgreSQL database"
```

### 2. Intelligent Alerting with PagerDuty Integration
```yaml
# monitoring/alertmanager/alertmanager.yml
global:
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default-receiver'
  routes:
  - match:
      severity: critical
    receiver: 'pagerduty-critical'
    group_wait: 0s
    repeat_interval: 5m
  - match:
      service: volcano-watchtower
    receiver: 'geological-team'
  - match:
      service: helios-core
    receiver: 'ml-team'

receivers:
- name: 'default-receiver'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#prometheus-alerts'
    title: 'Prometheus AI Platform Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'pagerduty-critical'
  pagerduty_configs:
  - routing_key: 'YOUR_PAGERDUTY_INTEGRATION_KEY'
    description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'
    severity: '{{ .CommonLabels.severity }}'
    details:
      firing: '{{ .Alerts.Firing | len }}'
      resolved: '{{ .Alerts.Resolved | len }}'

- name: 'geological-team'
  email_configs:
  - to: 'geological-team@prometheus-ai.com'
    subject: 'Geological System Alert: {{ .GroupLabels.alertname }}'
    body: |
      Alert Details:
      {{ range .Alerts }}
      - Alert: {{ .Annotations.summary }}
      - Description: {{ .Annotations.description }}
      - Severity: {{ .Labels.severity }}
      - Time: {{ .StartsAt }}
      {{ end }}

- name: 'ml-team'
  webhook_configs:
  - url: 'https://hooks.slack.com/services/YOUR/ML/TEAM/WEBHOOK'
    send_resolved: true
    title: 'ML System Alert'
    text: '{{ .CommonAnnotations.summary }}'
```

---

## 🌐 Service Mesh and Advanced Communication

### 1. Istio Service Mesh Configuration
```yaml
# service-mesh/istio/gateway.yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: prometheus-ai-gateway
  namespace: prometheus-ai-production
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: prometheus-ai-tls
    hosts:
    - api.prometheus-ai.com
    - monitoring.prometheus-ai.com
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - api.prometheus-ai.com
    - monitoring.prometheus-ai.com
    redirect:
      httpsRedirect: true

---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: prometheus-ai-vs
  namespace: prometheus-ai-production
spec:
  hosts:
  - api.prometheus-ai.com
  gateways:
  - prometheus-ai-gateway
  http:
  - match:
    - uri:
        prefix: /api/v1/
    route:
    - destination:
        host: api-gateway-service
        port:
          number: 3000
      weight: 90
    - destination:
        host: api-gateway-service-canary
        port:
          number: 3000
      weight: 10
    fault:
      delay:
        percentage:
          value: 0.1
        fixedDelay: 5s
    retries:
      attempts: 3
      perTryTimeout: 10s
```

### 2. Circuit Breaker and Bulkhead Patterns
```yaml
# service-mesh/istio/destination-rules.yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: helios-core-circuit-breaker
  namespace: prometheus-ai-production
spec:
  host: helios-core-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 10
        maxRetries: 3
        consecutiveGatewayErrors: 5
        interval: 30s
        baseEjectionTime: 30s
        maxEjectionPercent: 50
    outlierDetection:
      consecutiveGatewayErrors: 5
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 50
  portLevelSettings:
  - port:
      number: 8000
    connectionPool:
      tcp:
        maxConnections: 50
```

---

## 🔐 Advanced Security Framework

### 1. Zero-Trust Security Architecture
```yaml
# security/network-policies/zero-trust.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: zero-trust-policy
  namespace: prometheus-ai-production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
    - namespaceSelector:
        matchLabels:
          name: prometheus-ai-production
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to:
    - namespaceSelector:
        matchLabels:
          name: prometheus-ai-production
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-gateway-policy
  namespace: prometheus-ai-production
spec:
  podSelector:
    matchLabels:
      app: api-gateway
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
    ports:
    - protocol: TCP
      port: 3000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: helios-core
    ports:
    - protocol: TCP
      port: 8000
  - to:
    - podSelector:
        matchLabels:
          app: consul
    ports:
    - protocol: TCP
      port: 8500
```

### 2. Advanced Authentication and Authorization
```yaml
# security/rbac/service-accounts.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: api-gateway-sa
  namespace: prometheus-ai-production
  annotations:
    iam.gke.io/gcp-service-account: api-gateway@prometheus-ai-project.iam.gserviceaccount.com

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: prometheus-ai-production
  name: api-gateway-role
rules:
- apiGroups: [""]
  resources: ["services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: api-gateway-binding
  namespace: prometheus-ai-production
subjects:
- kind: ServiceAccount
  name: api-gateway-sa
  namespace: prometheus-ai-production
roleRef:
  kind: Role
  name: api-gateway-role
  apiGroup: rbac.authorization.k8s.io
```

---

## 🔄 Disaster Recovery and Business Continuity

### 1. Automated Backup Strategies
```yaml
# disaster-recovery/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
  namespace: prometheus-ai-production
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:15
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: database-secrets
                  key: postgres-password
            command:
            - /bin/bash
            - -c
            - |
              BACKUP_FILE="prometheus-ai-$(date +%Y%m%d-%H%M%S).sql"
              pg_dump -h postgres-service -U prometheus -d prometheus > /backup/$BACKUP_FILE
              
              # Upload to multiple cloud storage
              gsutil cp /backup/$BACKUP_FILE gs://prometheus-ai-backups/postgres/
              aws s3 cp /backup/$BACKUP_FILE s3://prometheus-ai-backups/postgres/
              az storage blob upload --file /backup/$BACKUP_FILE --container-name backups --name postgres/$BACKUP_FILE
              
              # Verify backup integrity
              pg_restore --list /backup/$BACKUP_FILE > /dev/null
              if [ $? -eq 0 ]; then
                echo "Backup verification successful"
              else
                echo "Backup verification failed" >&2
                exit 1
              fi
              
              # Clean up old backups (keep 30 days)
              find /backup -name "prometheus-ai-*.sql" -mtime +30 -delete
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

### 2. Cross-Region Failover Configuration
```yaml
# disaster-recovery/failover-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: failover-config
  namespace: prometheus-ai-production
data:
  primary-region: "us-central1"
  secondary-region: "us-east1"
  tertiary-region: "europe-west1"
  rto-minutes: "15"  # Recovery Time Objective
  rpo-minutes: "5"   # Recovery Point Objective
  
  failover-script: |
    #!/bin/bash
    set -e
    
    PRIMARY_CLUSTER="gke_prometheus-ai-project_us-central1_primary"
    SECONDARY_CLUSTER="gke_prometheus-ai-project_us-east1_secondary"
    
    # Check primary cluster health
    kubectl --context=$PRIMARY_CLUSTER get nodes > /dev/null 2>&1
    PRIMARY_HEALTHY=$?
    
    if [ $PRIMARY_HEALTHY -ne 0 ]; then
      echo "Primary cluster unhealthy, initiating failover..."
      
      # Update DNS to point to secondary region
      gcloud dns record-sets transaction start --zone=prometheus-ai-zone
      gcloud dns record-sets transaction remove --zone=prometheus-ai-zone \
        --name=api.prometheus-ai.com. --type=A --ttl=300 \
        --rrdatas=$(kubectl --context=$PRIMARY_CLUSTER get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
      gcloud dns record-sets transaction add --zone=prometheus-ai-zone \
        --name=api.prometheus-ai.com. --type=A --ttl=60 \
        --rrdatas=$(kubectl --context=$SECONDARY_CLUSTER get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
      gcloud dns record-sets transaction execute --zone=prometheus-ai-zone
      
      # Scale up secondary cluster
      kubectl --context=$SECONDARY_CLUSTER scale deployment api-gateway --replicas=5 -n prometheus-ai-production
      kubectl --context=$SECONDARY_CLUSTER scale deployment helios-core --replicas=10 -n prometheus-ai-production
      
      # Restore latest backup
      kubectl --context=$SECONDARY_CLUSTER create job --from=cronjob/database-restore restore-$(date +%s) -n prometheus-ai-production
      
      echo "Failover completed successfully"
    fi
```

---

## 📊 Performance Optimization and Scaling

### 1. Predictive Auto-scaling with Google Cloud AI
```yaml
# scaling/predictive-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: helios-core-hpa
  namespace: prometheus-ai-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: helios-core
  minReplicas: 3
  maxReplicas: 50
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
  - type: Pods
    pods:
      metric:
        name: ml_inference_queue_length
      target:
        type: AverageValue
        averageValue: "10"
  - type: External
    external:
      metric:
        name: pubsub.googleapis.com|subscription|num_undelivered_messages
        selector:
          matchLabels:
            resource.labels.subscription_id: geological-data-processing
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 5
        periodSeconds: 60
      selectPolicy: Max
```

### 2. Cost Optimization Monitoring
```yaml
# monitoring/cost-optimization/cost-alerts.yml
groups:
- name: cost-optimization
  rules:
  - alert: HighCloudCosts
    expr: increase(cloud_billing_cost_usd[1h]) > 100
    for: 5m
    labels:
      severity: warning
      team: platform
    annotations:
      summary: "High cloud costs detected"
      description: "Cloud costs increased by ${{ $value }} in the last hour"

  - alert: UnderutilizedResources
    expr: avg_over_time(node_cpu_utilization[24h]) < 0.3
    for: 1h
    labels:
      severity: info
      team: platform
    annotations:
      summary: "Underutilized resources detected"
      description: "Node {{ $labels.instance }} has been underutilized for 24 hours"

  - alert: IdleGPUResources
    expr: nvidia_gpu_utilization < 0.2
    for: 30m
    labels:
      severity: warning
      team: ml
    annotations:
      summary: "Idle GPU resources detected"
      description: "GPU {{ $labels.gpu }} has been idle for 30 minutes"
```

---

## 🚀 Deployment Commands and Usage

### Quick Start Commands
```bash
# Initialize CI/CD pipeline
git clone https://github.com/your-org/prometheus-ai-platform.git
cd prometheus-ai-platform

# Set up development environment
./scripts/setup-dev-environment.sh

# Run local CI pipeline
./scripts/run-local-ci.sh

# Deploy to staging
./scripts/deploy-staging.sh

# Deploy to production (requires approval)
./scripts/deploy-production.sh

# Monitor deployment
./scripts/monitor-deployment.sh production

# Rollback if needed
./scripts/rollback-deployment.sh production v1.2.3
```

### Advanced Operations
```bash
# Multi-cloud deployment
./scripts/deploy-multi-cloud.sh --regions us-central1,us-east1,europe-west1

# Disaster recovery test
./scripts/test-disaster-recovery.sh --scenario region-failure

# Performance testing
./scripts/run-performance-tests.sh --load-profile production

# Security audit
./scripts/run-security-audit.sh --full-scan

# Cost optimization analysis
./scripts/analyze-costs.sh --period 30d --recommendations
```

This comprehensive CI/CD pipeline and deployment infrastructure provides enterprise-grade DevOps capabilities with advanced monitoring, security, and disaster recovery features for the Prometheus AI Platform.