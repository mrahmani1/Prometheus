#!/usr/bin/env python3
"""
Health check script for TimescaleDB service
"""

import asyncio
import asyncpg
import sys
import os
from datetime import datetime

async def check_database_health():
    """Check database connectivity and TimescaleDB functionality"""
    try:
        database_url = os.getenv(
            "TIMESCALE_DATABASE_URL",
            "postgresql://prometheus:prometheus123@localhost:5432/prometheus_timeseries"
        )
        
        conn = await asyncpg.connect(database_url)
        
        # Test basic connectivity
        result = await conn.fetchval("SELECT 1")
        if result != 1:
            return False, "Basic connectivity test failed"
        
        # Test TimescaleDB extension
        timescale_version = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
        )
        if not timescale_version:
            return False, "TimescaleDB extension not found"
        
        # Test hypertables exist
        hypertables = await conn.fetch(
            "SELECT hypertable_name FROM timescaledb_information.hypertables"
        )
        if len(hypertables) == 0:
            return False, "No hypertables found"
        
        await conn.close()
        return True, f"Healthy - TimescaleDB {timescale_version}, {len(hypertables)} hypertables"
        
    except Exception as e:
        return False, f"Health check failed: {str(e)}"

async def main():
    """Main health check function"""
    is_healthy, message = await check_database_health()
    
    print(f"[{datetime.utcnow().isoformat()}] TimescaleDB Health Check: {message}")
    
    if is_healthy:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())