#!/bin/bash

# Rollback deployment script for Prometheus AI Platform
# Usage: ./rollback-deployment.sh <environment>

set -e

ENVIRONMENT=$1
NAMESPACE="prometheus-$ENVIRONMENT"

if [ -z "$ENVIRONMENT" ]; then
    echo "Usage: $0 <environment>"
    exit 1
fi

echo "Rolling back deployment for $ENVIRONMENT environment..."

# Get previous revision
PREVIOUS_REVISION=$(helm history prometheus-$ENVIRONMENT -n $NAMESPACE --max 2 -o json | jq -r '.[1].revision')

if [ "$PREVIOUS_REVISION" == "null" ] || [ -z "$PREVIOUS_REVISION" ]; then
    echo "No previous revision found for rollback"
    exit 1
fi

echo "Rolling back to revision $PREVIOUS_REVISION..."

# Perform rollback
helm rollback prometheus-$ENVIRONMENT $PREVIOUS_REVISION -n $NAMESPACE --wait --timeout=10m

# Verify rollback
echo "Verifying rollback..."
kubectl wait --for=condition=ready pod -l app=prometheus-ai-platform \
    --namespace $NAMESPACE --timeout=300s

# Run health checks
echo "Running post-rollback health checks..."
./scripts/health-checks.sh $ENVIRONMENT

echo "Rollback completed successfully!"