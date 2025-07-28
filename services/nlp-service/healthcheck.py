#!/usr/bin/env python3
"""
Health check script for NLP service
"""

import sys
import requests
import json

def check_health():
    """Check if the NLP service is healthy"""
    try:
        # Check health endpoint
        health_response = requests.get("http://localhost:8005/health", timeout=5)
        if health_response.status_code != 200:
            print(f"Health check failed: {health_response.status_code}")
            return False
        
        health_data = health_response.json()
        if health_data.get("status") != "healthy":
            print(f"Service not healthy: {health_data}")
            return False
        
        # Check readiness endpoint
        ready_response = requests.get("http://localhost:8005/ready", timeout=5)
        if ready_response.status_code != 200:
            print(f"Readiness check failed: {ready_response.status_code}")
            return False
        
        ready_data = ready_response.json()
        if ready_data.get("status") not in ["ready", "not_ready"]:
            print(f"Unexpected readiness status: {ready_data}")
            return False
        
        print("NLP service is healthy and responding")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Health check request failed: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during health check: {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)