#!/usr/bin/env python3
"""
Health check for Prometheus Model Training Orchestrator
"""

import asyncio
import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any

import httpx
import redis
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text
from influxdb_client import InfluxDBClient

logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checker for the orchestrator"""
    
    def __init__(self):
        self.checks = {
            'api_server': self.check_api_server,
            'redis': self.check_redis,
            'neo4j': self.check_neo4j,
            'postgres': self.check_postgres,
            'influxdb': self.check_influxdb,
            'gpu_availability': self.check_gpu_availability,
            'disk_space': self.check_disk_space,
            'memory_usage': self.check_memory_usage,
            'data_sources': self.check_data_sources,
            'model_registry': self.check_model_registry
        }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {},
            'summary': {
                'total_checks': len(self.checks),
                'passed': 0,
                'failed': 0,
                'warnings': 0
            }
        }
        
        for check_name, check_func in self.checks.items():
            try:
                check_result = await check_func()
                results['checks'][check_name] = check_result
                
                if check_result['status'] == 'healthy':
                    results['summary']['passed'] += 1
                elif check_result['status'] == 'warning':
                    results['summary']['warnings'] += 1
                else:
                    results['summary']['failed'] += 1
                    results['overall_status'] = 'unhealthy'
                    
            except Exception as e:
                results['checks'][check_name] = {
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results['summary']['failed'] += 1
                results['overall_status'] = 'unhealthy'
        
        return results
    
    async def check_api_server(self) -> Dict[str, Any]:
        """Check API server health"""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:8000/health",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    return {
                        'status': 'healthy',
                        'message': 'API server responding',
                        'response_time': response.elapsed.total_seconds(),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'message': f'API server returned status {response.status_code}',
                        'timestamp': datetime.now().isoformat()
                    }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'API server unreachable: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connection"""
        
        try:
            redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True,
                socket_connect_timeout=5
            )
            
            # Test connection
            redis_client.ping()
            
            # Test basic operations
            test_key = "health_check_test"
            redis_client.set(test_key, "test_value", ex=60)
            value = redis_client.get(test_key)
            redis_client.delete(test_key)
            
            if value == "test_value":
                # Get Redis info
                info = redis_client.info()
                
                return {
                    'status': 'healthy',
                    'message': 'Redis connection successful',
                    'version': info.get('redis_version'),
                    'memory_usage': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'message': 'Redis read/write test failed',
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Redis connection failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_neo4j(self) -> Dict[str, Any]:
        """Check Neo4j connection"""
        
        try:
            driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                auth=(os.getenv("NEO4J_USER", "neo4j"), 
                      os.getenv("NEO4J_PASSWORD", "password"))
            )
            
            with driver.session() as session:
                # Test connection
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
                if test_value == 1:
                    # Get database info
                    db_info = session.run("CALL dbms.components()").data()
                    
                    return {
                        'status': 'healthy',
                        'message': 'Neo4j connection successful',
                        'version': db_info[0]['versions'][0] if db_info else 'unknown',
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'message': 'Neo4j query test failed',
                        'timestamp': datetime.now().isoformat()
                    }
            
            driver.close()
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Neo4j connection failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_postgres(self) -> Dict[str, Any]:
        """Check PostgreSQL connection"""
        
        try:
            engine = create_engine(
                os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost/prometheus"),
                pool_timeout=5
            )
            
            with engine.connect() as connection:
                # Test connection
                result = connection.execute(text("SELECT 1 as test"))
                test_value = result.fetchone()[0]
                
                if test_value == 1:
                    # Get database info
                    version_result = connection.execute(text("SELECT version()"))
                    version = version_result.fetchone()[0]
                    
                    return {
                        'status': 'healthy',
                        'message': 'PostgreSQL connection successful',
                        'version': version.split()[1] if version else 'unknown',
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'message': 'PostgreSQL query test failed',
                        'timestamp': datetime.now().isoformat()
                    }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'PostgreSQL connection failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_influxdb(self) -> Dict[str, Any]:
        """Check InfluxDB connection"""
        
        try:
            client = InfluxDBClient(
                url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
                token=os.getenv("INFLUXDB_TOKEN", "token"),
                org=os.getenv("INFLUXDB_ORG", "prometheus"),
                timeout=5000
            )
            
            # Test connection
            health = client.health()
            
            if health.status == "pass":
                return {
                    'status': 'healthy',
                    'message': 'InfluxDB connection successful',
                    'version': health.version,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'message': f'InfluxDB health check failed: {health.message}',
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'InfluxDB connection failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability"""
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = []
                
                for i in range(gpu_count):
                    gpu_props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_cached = torch.cuda.memory_reserved(i)
                    
                    gpu_info.append({
                        'device_id': i,
                        'name': gpu_props.name,
                        'memory_total': gpu_props.total_memory,
                        'memory_allocated': memory_allocated,
                        'memory_cached': memory_cached,
                        'memory_free': gpu_props.total_memory - memory_allocated,
                        'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                    })
                
                return {
                    'status': 'healthy',
                    'message': f'{gpu_count} GPU(s) available',
                    'gpu_count': gpu_count,
                    'gpu_info': gpu_info,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'No GPUs available, using CPU only',
                    'gpu_count': 0,
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'GPU check failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_disk_space(self) -> Dict[str, Any]:
        """Check disk space"""
        
        try:
            import shutil
            
            # Check main application directory
            total, used, free = shutil.disk_usage("/app")
            
            # Convert to GB
            total_gb = total / (1024**3)
            used_gb = used / (1024**3)
            free_gb = free / (1024**3)
            usage_percent = (used / total) * 100
            
            if usage_percent > 90:
                status = 'unhealthy'
                message = f'Disk usage critical: {usage_percent:.1f}%'
            elif usage_percent > 80:
                status = 'warning'
                message = f'Disk usage high: {usage_percent:.1f}%'
            else:
                status = 'healthy'
                message = f'Disk usage normal: {usage_percent:.1f}%'
            
            return {
                'status': status,
                'message': message,
                'total_gb': round(total_gb, 2),
                'used_gb': round(used_gb, 2),
                'free_gb': round(free_gb, 2),
                'usage_percent': round(usage_percent, 1),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Disk space check failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                status = 'unhealthy'
                message = f'Memory usage critical: {memory.percent:.1f}%'
            elif memory.percent > 80:
                status = 'warning'
                message = f'Memory usage high: {memory.percent:.1f}%'
            else:
                status = 'healthy'
                message = f'Memory usage normal: {memory.percent:.1f}%'
            
            return {
                'status': status,
                'message': message,
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'usage_percent': round(memory.percent, 1),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Memory check failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_data_sources(self) -> Dict[str, Any]:
        """Check external data sources availability"""
        
        data_sources = [
            "https://service.iris.edu/fdsnws/",
            "https://api.daac.asf.alaska.edu",
            "https://earthquake.usgs.gov/fdsnws/",
            "https://scihub.copernicus.eu/"
        ]
        
        available_sources = 0
        source_status = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for source in data_sources:
                try:
                    response = await client.head(source)
                    if response.status_code < 400:
                        source_status[source] = 'available'
                        available_sources += 1
                    else:
                        source_status[source] = f'error_{response.status_code}'
                except Exception as e:
                    source_status[source] = f'unreachable_{str(e)[:50]}'
        
        availability_percent = (available_sources / len(data_sources)) * 100
        
        if availability_percent >= 80:
            status = 'healthy'
            message = f'{available_sources}/{len(data_sources)} data sources available'
        elif availability_percent >= 50:
            status = 'warning'
            message = f'Only {available_sources}/{len(data_sources)} data sources available'
        else:
            status = 'unhealthy'
            message = f'Critical: Only {available_sources}/{len(data_sources)} data sources available'
        
        return {
            'status': status,
            'message': message,
            'available_sources': available_sources,
            'total_sources': len(data_sources),
            'availability_percent': round(availability_percent, 1),
            'source_status': source_status,
            'timestamp': datetime.now().isoformat()
        }
    
    async def check_model_registry(self) -> Dict[str, Any]:
        """Check model registry and storage"""
        
        try:
            models_dir = os.getenv("MODELS_DIR", "/app/models")
            
            if not os.path.exists(models_dir):
                os.makedirs(models_dir, exist_ok=True)
            
            # Check if directory is writable
            test_file = os.path.join(models_dir, "health_check_test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            
            os.remove(test_file)
            
            # Count existing models
            model_files = [f for f in os.listdir(models_dir) 
                          if f.endswith(('.pth', '.pkl', '.joblib', '.h5'))]
            
            return {
                'status': 'healthy',
                'message': f'Model registry accessible with {len(model_files)} models',
                'models_directory': models_dir,
                'model_count': len(model_files),
                'writable': True,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Model registry check failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

async def main():
    """Main health check function"""
    
    checker = HealthChecker()
    results = await checker.run_all_checks()
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    if results['overall_status'] == 'healthy':
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())