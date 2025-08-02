#!/usr/bin/env python3
"""
Health check script for System Integration Service
"""

import sys

import requests
import torch.nn.functional as F


def check_health():
    """Check if the service is healthy"""
    try:
        response = requests.get("http://localhost:8010/health", timeout=5)
        if response.status_code == 200:
            print("System Integration Service is healthy")
            return True
        else:
            print(f"Health check failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
