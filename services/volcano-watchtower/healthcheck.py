#!/usr/bin/env python3
"""
Health check script for Volcano Watchtower service
"""

import asyncio
import os
import sys

import httpx
import torch.nn.functional as F


async def check_health():
    """Check service health"""
    try:
        port = int(os.getenv("PORT", 8080))
        url = f"http://localhost:{port}/health"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)

            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    print("Service is healthy")
                    return True
                else:
                    print(f"Service reports unhealthy status: {health_data}")
                    return False
            else:
                print(f"Health check failed with status code: {response.status_code}")
                return False

    except Exception as e:
        print(f"Health check failed with error: {e}")
        return False


if __name__ == "__main__":
    is_healthy = asyncio.run(check_health())
    sys.exit(0 if is_healthy else 1)
