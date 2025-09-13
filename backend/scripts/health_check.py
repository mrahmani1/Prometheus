#!/usr/bin/env python3
"""
Health check script for Real-Time Process Monitor

This script performs comprehensive health checks for the monitoring system
including database connectivity, WebSocket server status, and system performance.
"""

import asyncio
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import aiohttp
import asyncpg
import redis
from influxdb_client import InfluxDBClient
import websockets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checker for the monitoring system"""
    
    def __init__(self):
        self.health_status = {
            'status': 'unknown',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {},
            'overall_health': False
        }
    
    async def check_redis_connectivity(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        try:
            redis_client = redis.Redis(
                host='redis-master',
                port=6379,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            start_time = time.time()
            
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            redis_client.set(test_key, "test_value", ex=60)
            value = redis_client.get(test_key)
            redis_client.delete(test_key)
            
            latency = (time.time() - start_time) * 1000  # ms
            
            # Get Redis info
            info = redis_client.info()
            
            return {
                'status': 'healthy',
                'latency_ms': round(latency, 2),
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', 'unknown'),
                'uptime_in_seconds': info.get('uptime_in_seconds', 0)
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_influxdb_connectivity(self) -> Dict[str, Any]:
        """Check InfluxDB connectivity and performance"""
        try:
            client = InfluxDBClient(
                url="http://influxdb:8086",
                timeout=10000
            )
            
            start_time = time.time()
            
            # Test query
            query_api = client.query_api()
            query = 'buckets() |> limit(1)'
            
            result = query_api.query(query)
            latency = (time.time() - start_time) * 1000  # ms
            
            # Check health endpoint
            health = client.health()
            
            client.close()
            
            return {
                'status': 'healthy',
                'latency_ms': round(latency, 2),
                'health_status': health.status,
                'version': health.version
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_postgresql_connectivity(self) -> Dict[str, Any]:
        """Check PostgreSQL connectivity and performance"""
        try:
            start_time = time.time()
            
            conn = await asyncpg.connect(
                host='postgresql',
                port=5432,
                database='prometheus',
                user='postgres',
                timeout=10
            )
            
            # Test query
            result = await conn.fetchval('SELECT version()')
            latency = (time.time() - start_time) * 1000  # ms
            
            # Get database stats
            stats = await conn.fetchrow("""
                SELECT 
                    pg_database_size(current_database()) as db_size,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections
            """)
            
            await conn.close()
            
            return {
                'status': 'healthy',
                'latency_ms': round(latency, 2),
                'version': result.split()[1] if result else 'unknown',
                'database_size_bytes': stats['db_size'] if stats else 0,
                'active_connections': stats['active_connections'] if stats else 0
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_websocket_server(self) -> Dict[str, Any]:
        """Check WebSocket server status"""
        try:
            start_time = time.time()
            
            # Test WebSocket connection
            uri = "ws://localhost:8765"
            async with websockets.connect(uri, timeout=5) as websocket:
                # Send ping
                await websocket.send(json.dumps({'type': 'ping'}))
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    latency = (time.time() - start_time) * 1000  # ms
                    
                    return {
                        'status': 'healthy',
                        'latency_ms': round(latency, 2),
                        'response_received': True
                    }
                except asyncio.TimeoutError:
                    return {
                        'status': 'degraded',
                        'error': 'WebSocket response timeout'
                    }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Determine health based on thresholds
            status = 'healthy'
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = 'degraded'
            if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
                status = 'unhealthy'
            
            return {
                'status': status,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'process_count': process_count
            }
            
        except ImportError:
            # psutil not available, basic check
            return {
                'status': 'unknown',
                'error': 'psutil not available for detailed system monitoring'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def check_application_health(self) -> Dict[str, Any]:
        """Check application-specific health metrics"""
        try:
            # Check if the main application is responding
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get('http://localhost:8080/health') as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': 'healthy',
                            'http_status': response.status,
                            'response_data': data
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'http_status': response.status
                        }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status"""
        logger.info("Starting comprehensive health checks...")
        
        # Run all checks concurrently
        checks = await asyncio.gather(
            self.check_redis_connectivity(),
            self.check_influxdb_connectivity(),
            self.check_postgresql_connectivity(),
            self.check_websocket_server(),
            self.check_system_resources(),
            self.check_application_health(),
            return_exceptions=True
        )
        
        # Map results to check names
        check_names = [
            'redis',
            'influxdb',
            'postgresql',
            'websocket',
            'system_resources',
            'application'
        ]
        
        self.health_status['checks'] = {}
        healthy_checks = 0
        total_checks = len(check_names)
        
        for name, result in zip(check_names, checks):
            if isinstance(result, Exception):
                self.health_status['checks'][name] = {
                    'status': 'error',
                    'error': str(result)
                }
            else:
                self.health_status['checks'][name] = result
                if result.get('status') == 'healthy':
                    healthy_checks += 1
        
        # Determine overall health
        health_ratio = healthy_checks / total_checks
        
        if health_ratio >= 0.8:
            self.health_status['status'] = 'healthy'
            self.health_status['overall_health'] = True
        elif health_ratio >= 0.6:
            self.health_status['status'] = 'degraded'
            self.health_status['overall_health'] = True
        else:
            self.health_status['status'] = 'unhealthy'
            self.health_status['overall_health'] = False
        
        self.health_status['healthy_checks'] = healthy_checks
        self.health_status['total_checks'] = total_checks
        self.health_status['health_ratio'] = health_ratio
        self.health_status['timestamp'] = datetime.utcnow().isoformat()
        
        logger.info(f"Health check completed: {self.health_status['status']} "
                   f"({healthy_checks}/{total_checks} checks passed)")
        
        return self.health_status

async def main():
    """Main health check function"""
    checker = HealthChecker()
    
    try:
        health_status = await checker.run_all_checks()
        
        # Print results
        print(json.dumps(health_status, indent=2))
        
        # Exit with appropriate code
        if health_status['overall_health']:
            sys.exit(0)  # Healthy
        else:
            sys.exit(1)  # Unhealthy
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        print(json.dumps({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())