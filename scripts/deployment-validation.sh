#!/bin/bash

# Deployment validation script for Prometheus AI Platform
# Usage: ./deployment-validation.sh <environment>

set -e

ENVIRONMENT=$1
NAMESPACE="prometheus-$ENVIRONMENT"

if [ -z "$ENVIRONMENT" ]; then
    echo "Usage: $0 <environment>"
    exit 1
fi

echo "Validating deployment for $ENVIRONMENT environment..."

# Check all pods are running
echo "Checking pod status..."
kubectl get pods -n $NAMESPACE
kubectl wait --for=condition=ready pod -l app=prometheus-ai-platform \
    --namespace $NAMESPACE --timeout=300s

# Check service endpoints
echo "Checking service endpoints..."
kubectl get endpoints -n $NAMESPACE

# Validate database connections
echo "Validating database connections..."
kubectl exec -n $NAMESPACE deployment/helios-core -- python -c "
import asyncio
from core.database import test_connections
asyncio.run(test_connections())
"

# Check resource usage
echo "Checking resource usage..."
kubectl top pods -n $NAMESPACE

echo "Deployment validation completed successfully!"