#!/usr/bin/env python3
"""Health check script for Helios Core Service"""

import asyncio
import os
import sys

import aiohttp
import torch.nn.functional as F


async def check_health():
    """Check if the service is healthy"""
    try:
        port = os.getenv("PORT", "8000")
        url = f"http://localhost:{port}/health"

        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "healthy":
                        print("Health check passed")
                        return True

                print(f"Health check failed: HTTP {response.status}")
                return False

    except Exception as e:
        print(f"Health check failed: {e}")
        return False


if __name__ == "__main__":
    is_healthy = asyncio.run(check_health())
    sys.exit(0 if is_healthy else 1)
