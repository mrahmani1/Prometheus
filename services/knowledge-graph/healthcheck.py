#!/usr/bin/env python3
"""
Health check script for the Scientific Knowledge Graph Service
"""

import asyncio
import os
import sys

import aiohttp
import torch.nn.functional as F


async def health_check():
    """Perform health check"""
    try:
        port = os.getenv("PORT", 8007)
        url = f"http://localhost:{port}/health"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
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
    result = asyncio.run(health_check())
    sys.exit(0 if result else 1)
