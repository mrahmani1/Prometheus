#!/bin/bash

# Canary deployment script for Prometheus AI Platform
# Usage: ./canary-deploy.sh <environment> <image_tag>

set -e

ENVIRONMENT=$1
IMAGE_TAG=$2
NAMESPACE="prometheus-$ENVIRONMENT"

if [ -z "$ENVIRONMENT" ] || [ -z "$IMAGE_TAG" ]; then
    echo "Usage: $0 <environment> <image_tag>"
    exit 1
fi

echo "Starting canary deployment for $ENVIRONMENT with image tag $IMAGE_TAG"

# Deploy canary version (10% traffic)
echo "Deploying canary version..."
helm upgrade --install prometheus-$ENVIRONMENT-canary ./helm/prometheus-ai-platform \
    --namespace $NAMESPACE \
    --set image.tag=$IMAGE_TAG \
    --set environment=$ENVIRONMENT \
    --set deployment.type=canary \
    --set deployment.canaryWeight=10 \
    --wait --timeout=15m

# Monitor canary for 5 minutes
echo "Monitoring canary deployment..."
sleep 300

# Check error rates and performance metrics
echo "Checking canary metrics..."
# Add your monitoring checks here

# If successful, increase to 50%
echo "Increasing canary traffic to 50%..."
helm upgrade prometheus-$ENVIRONMENT-canary ./helm/prometheus-ai-platform \
    --namespace $NAMESPACE \
    --set deployment.canaryWeight=50 \
    --reuse-values

sleep 300

# Final promotion to 100%
echo "Promoting canary to 100%..."
helm upgrade prometheus-$ENVIRONMENT ./helm/prometheus-ai-platform \
    --namespace $NAMESPACE \
    --set image.tag=$IMAGE_TAG \
    --set environment=$ENVIRONMENT \
    --set deployment.type=stable

echo "Canary deployment completed successfully!"