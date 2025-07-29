#!/usr/bin/env python3
"""
Health check endpoint for the System Monitoring Service
"""

import asyncio
import sys
import aiohttp

async def health_check():
    """Perform health check"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health', timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == 'healthy':
                        print("Health check passed")
                        return 0
                    else:
                        print("Health check failed: unhealthy status")
                        return 1
                else:
                    print(f"Health check failed: HTTP {response.status}")
                    return 1
    except Exception as e:
        print(f"Health check failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(health_check())
    sys.exit(exit_code)