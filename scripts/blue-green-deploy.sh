#!/bin/bash

# Blue-Green deployment script for Prometheus AI Platform
# Usage: ./blue-green-deploy.sh <environment> <image_tag>

set -e

ENVIRONMENT=$1
IMAGE_TAG=$2
NAMESPACE="prometheus-$ENVIRONMENT"

if [ -z "$ENVIRONMENT" ] || [ -z "$IMAGE_TAG" ]; then
    echo "Usage: $0 <environment> <image_tag>"
    exit 1
fi

echo "Starting blue-green deployment for $ENVIRONMENT with image tag $IMAGE_TAG"

# Deploy to green environment
echo "Deploying to green environment..."
helm upgrade --install prometheus-$ENVIRONMENT-green ./helm/prometheus-ai-platform \
    --namespace $NAMESPACE-green \
    --create-namespace \
    --set image.tag=$IMAGE_TAG \
    --set environment=$ENVIRONMENT \
    --set deployment.color=green \
    --wait --timeout=15m

# Run health checks on green environment
echo "Running health checks on green environment..."
kubectl wait --for=condition=ready pod -l app=prometheus-ai-platform,color=green \
    --namespace $NAMESPACE-green --timeout=300s

# Switch traffic to green
echo "Switching traffic to green environment..."
kubectl patch service prometheus-ai-platform \
    --namespace $NAMESPACE \
    --patch '{"spec":{"selector":{"color":"green"}}}'

# Wait for traffic switch
sleep 30

# Cleanup blue environment
echo "Cleaning up blue environment..."
helm uninstall prometheus-$ENVIRONMENT-blue --namespace $NAMESPACE-blue || true

echo "Blue-green deployment completed successfully!"