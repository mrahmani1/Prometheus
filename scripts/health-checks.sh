#!/bin/bash

# Health checks script for Prometheus AI Platform
# Usage: ./health-checks.sh <environment>

set -e

ENVIRONMENT=$1
BASE_URL=""

case $ENVIRONMENT in
  "staging")
    BASE_URL="https://staging.prometheus-ai.com"
    ;;
  "production")
    BASE_URL="https://prometheus-ai.com"
    ;;
  *)
    echo "Unknown environment: $ENVIRONMENT"
    exit 1
    ;;
esac

echo "Running comprehensive health checks for $ENVIRONMENT..."

# API Gateway health
echo "Checking API Gateway..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
if [ "$response" != "200" ]; then
    echo "API Gateway health check failed: $response"
    exit 1
fi

# Service health checks
services=("helios-core" "data-ingestion" "volcano-watchtower")
for service in "${services[@]}"; do
    echo "Checking $service service..."
    response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/v1/$service/health")
    if [ "$response" != "200" ]; then
        echo "$service health check failed: $response"
        exit 1
    fi
done

# Database connectivity checks
echo "Checking database connectivity..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/v1/health/databases")
if [ "$response" != "200" ]; then
    echo "Database connectivity check failed: $response"
    exit 1
fi

# Performance checks
echo "Running performance checks..."
response_time=$(curl -s -o /dev/null -w "%{time_total}" "$BASE_URL/api/v1/helios/status")
if (( $(echo "$response_time > 2.0" | bc -l) )); then
    echo "Performance check failed: response time $response_time seconds"
    exit 1
fi

echo "All health checks passed!"