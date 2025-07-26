#!/usr/bin/env python3
"""Health check script for data ingestion service"""

import sys
import httpx
import asyncio

async def check_health():
    """Check if the service is healthy"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/health", timeout=10.0)
            if response.status_code == 200:
                print("Service is healthy")
                return True
            else:
                print(f"Service unhealthy: {response.status_code}")
                return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    is_healthy = asyncio.run(check_health())
    sys.exit(0 if is_healthy else 1)