#!/usr/bin/env python3
"""
Health Check Module for Collaboration Hub
Comprehensive health monitoring and diagnostics
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import text
import structlog

logger = structlog.get_logger()

@dataclass
class HealthStatus:
    """Health status information"""
    service: str
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any]

class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.redis_client = None
        self.db_engine = None
        
    async def initialize(self):
        """Initialize health checker"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
            
            # Initialize Database connection
            self.db_engine = create_async_engine("postgresql+asyncpg://prometheus:prometheus123@localhost/prometheus")
            
            logger.info("Health checker initialized")
            
        except Exception as e:
            logger.error("Failed to initialize health checker", error=str(e))
            raise
    
    async def check_overall_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        start_time = time.time()
        
        health_checks = [
            self.check_redis_health(),
            self.check_database_health(),
            self.check_websocket_health(),
            self.check_memory_usage(),
            self.check_collaboration_metrics()
        ]
        
        results = await asyncio.gather(*health_checks, return_exceptions=True)
        
        # Process results
        health_statuses = []
        overall_status = "healthy"
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                health_statuses.append(HealthStatus(
                    service=f"check_{i}",
                    status="unhealthy",
                    timestamp=datetime.utcnow(),
                    response_time_ms=0,
                    details={"error": str(result)}
                ))
                overall_status = "unhealthy"
            else:
                health_statuses.append(result)
                if result.status == "degraded" and overall_status == "healthy":
                    overall_status = "degraded"
                elif result.status == "unhealthy":
                    overall_status = "unhealthy"
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "response_time_ms": total_time,
            "version": "1.0.0",
            "service": "collaboration-hub",
            "checks": [
                {
                    "service": status.service,
                    "status": status.status,
                    "response_time_ms": status.response_time_ms,
                    "details": status.details
                }
                for status in health_statuses
            ]
        }
    
    async def check_redis_health(self) -> HealthStatus:
        """Check Redis connectivity and performance"""
        start_time = time.time()
        
        try:
            # Test basic connectivity
            await self.redis_client.ping()
            
            # Test read/write operations
            test_key = "health_check_test"
            test_value = f"test_{int(time.time())}"
            
            await self.redis_client.set(test_key, test_value, ex=60)
            retrieved_value = await self.redis_client.get(test_key)
            
            if retrieved_value != test_value:
                raise Exception("Redis read/write test failed")
            
            # Get Redis info
            info = await self.redis_client.info()
            
            response_time = (time.time() - start_time) * 1000
            
            # Check if response time is acceptable
            status = "healthy"
            if response_time > 100:
                status = "degraded"
            if response_time > 500:
                status = "unhealthy"
            
            return HealthStatus(
                service="redis",
                status=status,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "redis_version": info.get("redis_version", "unknown")
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                service="redis",
                status="unhealthy",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    async def check_database_health(self) -> HealthStatus:
        """Check database connectivity and performance"""
        start_time = time.time()
        
        try:
            async with self.db_engine.begin() as conn:
                # Test basic connectivity
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()
                
                # Test table access
                result = await conn.execute(text("SELECT COUNT(*) FROM collaboration_sessions"))
                session_count = result.scalar()
                
                # Check database performance
                result = await conn.execute(text("SELECT version()"))
                db_version = result.scalar()
            
            response_time = (time.time() - start_time) * 1000
            
            # Check if response time is acceptable
            status = "healthy"
            if response_time > 200:
                status = "degraded"
            if response_time > 1000:
                status = "unhealthy"
            
            return HealthStatus(
                service="database",
                status=status,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={
                    "session_count": session_count,
                    "database_version": db_version.split()[0] if db_version else "unknown"
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                service="database",
                status="unhealthy",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    async def check_websocket_health(self) -> HealthStatus:
        """Check WebSocket server health"""
        start_time = time.time()
        
        try:
            # Check if WebSocket server is responsive
            # This is a simplified check - in production, you might want to test actual WebSocket connections
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                service="websocket",
                status="healthy",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={
                    "active_connections": 0,  # TODO: Get from connection manager
                    "message_queue_size": 0
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                service="websocket",
                status="unhealthy",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    async def check_memory_usage(self) -> HealthStatus:
        """Check memory usage and system resources"""
        start_time = time.time()
        
        try:
            import psutil
            
            # Get memory usage
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on memory usage
            status = "healthy"
            if memory.percent > 80:
                status = "degraded"
            if memory.percent > 95:
                status = "unhealthy"
            
            return HealthStatus(
                service="memory",
                status=status,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={
                    "system_memory_percent": memory.percent,
                    "system_memory_available_gb": round(memory.available / (1024**3), 2),
                    "process_memory_mb": round(process_memory.rss / (1024**2), 2),
                    "cpu_percent": psutil.cpu_percent()
                }
            )
            
        except ImportError:
            # psutil not available, return basic check
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                service="memory",
                status="healthy",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={"note": "psutil not available for detailed metrics"}
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                service="memory",
                status="unhealthy",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    async def check_collaboration_metrics(self) -> HealthStatus:
        """Check collaboration-specific metrics"""
        start_time = time.time()
        
        try:
            # Get active sessions from Redis
            active_sessions_keys = await self.redis_client.keys("operations:*")
            active_sessions_count = len(active_sessions_keys)
            
            # Get recent operations count
            recent_operations = 0
            for key in active_sessions_keys[:10]:  # Check first 10 sessions
                operations = await self.redis_client.llen(key)
                recent_operations += operations
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on activity
            status = "healthy"
            if active_sessions_count > 100:
                status = "degraded"
            if active_sessions_count > 500:
                status = "unhealthy"
            
            return HealthStatus(
                service="collaboration",
                status=status,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={
                    "active_sessions": active_sessions_count,
                    "recent_operations": recent_operations,
                    "avg_operations_per_session": round(recent_operations / max(1, active_sessions_count), 2)
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                service="collaboration",
                status="unhealthy",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.db_engine:
                await self.db_engine.dispose()
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))

# Global health checker instance
health_checker = HealthChecker()

async def get_health_status() -> Dict[str, Any]:
    """Get current health status"""
    try:
        await health_checker.initialize()
        return await health_checker.check_overall_health()
    finally:
        await health_checker.cleanup()

if __name__ == "__main__":
    # CLI health check
    async def main():
        status = await get_health_status()
        print(json.dumps(status, indent=2))
        
        # Exit with appropriate code
        if status["status"] == "healthy":
            exit(0)
        elif status["status"] == "degraded":
            exit(1)
        else:
            exit(2)
    
    asyncio.run(main())