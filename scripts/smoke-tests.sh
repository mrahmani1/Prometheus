#!/bin/bash

# Smoke tests for Prometheus AI Platform
# Usage: ./smoke-tests.sh <environment>

set -e

ENVIRONMENT=${1:-staging}
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

echo "Running smoke tests for $ENVIRONMENT environment..."

# Test API Gateway health
echo "Testing API Gateway health..."
curl -f "$BASE_URL/health" || exit 1

# Test Helios Core service
echo "Testing Helios Core service..."
curl -f "$BASE_URL/api/v1/helios/health" || exit 1

# Test Data Ingestion service
echo "Testing Data Ingestion service..."
curl -f "$BASE_URL/api/v1/data-ingestion/health" || exit 1

# Test Volcano Watchtower service
echo "Testing Volcano Watchtower service..."
curl -f "$BASE_URL/api/v1/volcano-watchtower/health" || exit 1

echo "All smoke tests passed!"