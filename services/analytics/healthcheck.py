"""
Health check script for Advanced Analytics Service
Used by Docker and Kubernetes for service health monitoring
"""

import asyncio
import aiohttp
import sys
import os
from datetime import datetime

async def check_health():
    """Check service health"""
    try:
        port = os.getenv('PORT', '8004')
        health_url = f"http://localhost:{port}/health"
        
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(health_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check if all critical components are healthy
                    components = data.get('components', {})
                    critical_components = [
                        'geological_modeler',
                        'magmatic_modeler',
                        'gnn_processor'
                    ]
                    
                    unhealthy_critical = [
                        name for name in critical_components 
                        if components.get(name) != 'healthy'
                    ]
                    
                    if unhealthy_critical:
                        print(f"Critical components unhealthy: {unhealthy_critical}")
                        return False
                    
                    # Check for excessive active jobs (potential overload)
                    active_jobs = data.get('active_jobs', 0)
                    if active_jobs > 50:  # Configurable threshold
                        print(f"Too many active jobs: {active_jobs}")
                        return False
                    
                    print(f"Service healthy - {active_jobs} active jobs, GPU: {data.get('gpu_available', False)}")
                    return True
                else:
                    print(f"Health check failed with status {response.status}")
                    return False
                    
    except aiohttp.ClientError as e:
        print(f"Health check connection failed: {e}")
        return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

async def check_readiness():
    """Check service readiness"""
    try:
        port = os.getenv('PORT', '8004')
        ready_url = f"http://localhost:{port}/ready"
        
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(ready_url) as response:
                if response.status == 200:
                    print("Service ready")
                    return True
                else:
                    print(f"Service not ready - status {response.status}")
                    return False
                    
    except Exception as e:
        print(f"Readiness check failed: {e}")
        return False

def main():
    """Main health check function"""
    check_type = sys.argv[1] if len(sys.argv) > 1 else "health"
    
    if check_type == "ready":
        result = asyncio.run(check_readiness())
    else:
        result = asyncio.run(check_health())
    
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()