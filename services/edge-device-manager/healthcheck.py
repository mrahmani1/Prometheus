#!/usr/bin/env python3
"""
Health check script for Edge Device Manager service
"""

import asyncio
import logging
import os
import sys

import aiohttp
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_health():
    """Perform health check on the service"""
    try:
        port = int(os.getenv("PORT", 8080))
        health_url = f"http://localhost:{port}/health"
        ready_url = f"http://localhost:{port}/ready"

        timeout = aiohttp.ClientTimeout(total=10)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Check basic health endpoint
            async with session.get(health_url) as response:
                if response.status != 200:
                    logger.error(f"Health check failed with status {response.status}")
                    return False

                health_data = await response.json()
                if health_data.get("status") != "healthy":
                    logger.error(f"Service reports unhealthy status: {health_data}")
                    return False

            # Check readiness endpoint
            async with session.get(ready_url) as response:
                if response.status != 200:
                    logger.warning(
                        f"Readiness check failed with status {response.status}"
                    )
                    # Don't fail health check if service is starting up
                    return True

                ready_data = await response.json()
                if ready_data.get("status") != "ready":
                    logger.warning(f"Service not ready: {ready_data}")
                    # Don't fail health check if service is starting up
                    return True

        logger.info("Health check passed")
        return True

    except aiohttp.ClientError as e:
        logger.error(f"Health check failed with client error: {e}")
        return False
    except Exception as e:
        logger.error(f"Health check failed with unexpected error: {e}")
        return False


async def main():
    """Main health check function"""
    try:
        is_healthy = await check_health()

        if is_healthy:
            logger.info("Edge Device Manager is healthy")
            sys.exit(0)
        else:
            logger.error("Edge Device Manager is unhealthy")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Health check script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
