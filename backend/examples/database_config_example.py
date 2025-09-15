#!/usr/bin/env python3
"""
Database Configuration Management Example
Demonstrates the standardized database configuration management system
"""

import os
import asyncio
import logging
from pathlib import Path

# Set up the configuration directory
os.environ['DB_CONFIG_DIR'] = str(Path(__file__).parent.parent.parent / 'config' / 'database')
os.environ['ENVIRONMENT'] = 'development'

from backend.core.database_service import get_database_service
from backend.core.database_config_manager import DatabaseType, Environment


async def main():
    """Demonstrate database configuration management"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Database Configuration Management Example")
    
    try:
        # Get database service (automatically initializes with environment-based config)
        service = await get_database_service(environment='development')
        
        # Display service status
        status = service.get_service_status()
        logger.info(f"Database Service Status:")
        logger.info(f"  Environment: {status['environment']}")
        logger.info(f"  Total Databases: {status['total_databases']}")
        logger.info(f"  Healthy Databases: {status['healthy_databases']}")
        
        # Add a new database configuration
        logger.info("Adding new Redis database configuration...")
        success = await service.add_database(
            name="example_redis",
            db_type=DatabaseType.REDIS,
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            database="0"
        )
        
        if success:
            logger.info("Successfully added Redis configuration")
            
            # Test the connection (will fail if Redis not running, but that's OK)
            try:
                async with service.get_database_connection("example_redis") as conn:
                    logger.info("Successfully connected to Redis")
            except Exception as e:
                logger.warning(f"Redis connection failed (expected if not running): {e}")
        
        # Add PostgreSQL configuration
        logger.info("Adding PostgreSQL database configuration...")
        success = await service.add_database(
            name="example_postgres",
            db_type=DatabaseType.POSTGRESQL,
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', '5432')),
            database=os.getenv('POSTGRES_DATABASE', 'prometheus_dev')
        )
        
        if success:
            logger.info("Successfully added PostgreSQL configuration")
        
        # Display updated status
        status = service.get_service_status()
        logger.info(f"Updated Database Service Status:")
        logger.info(f"  Total Databases: {status['total_databases']}")
        
        for db_name, db_status in status['database_status'].items():
            logger.info(f"  {db_name}: {db_status['type']} at {db_status['host']}:{db_status['port']} - {db_status['health']}")
        
        # Test all connections
        logger.info("Testing all database connections...")
        connection_results = await service.test_all_connections()
        
        for db_name, is_healthy in connection_results.items():
            status_msg = "✓ Healthy" if is_healthy else "✗ Unhealthy"
            logger.info(f"  {db_name}: {status_msg}")
        
        # Demonstrate configuration manager features
        if service.config_manager:
            logger.info("Generating configuration documentation...")
            docs = service.config_manager.generate_documentation()
            
            # Save documentation to file
            docs_file = Path(__file__).parent / 'database_config_docs.md'
            with open(docs_file, 'w') as f:
                f.write(docs)
            logger.info(f"Documentation saved to: {docs_file}")
        
        logger.info("Database Configuration Management Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
    
    finally:
        # Cleanup
        if 'service' in locals():
            await service.cleanup()


if __name__ == '__main__':
    asyncio.run(main())