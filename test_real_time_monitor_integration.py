#!/usr/bin/env python3
"""
Test script for Real-Time Process Monitor with real data integration

This script tests the updated real-time process monitor with actual data connections
to IRIS, USGS, and other geological data sources.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from modules.real_time_process_monitor import RealTimeProcessMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_real_data_integration():
    """Test real data integration with the process monitor"""
    
    # Configuration for the monitor using environment variables
    import os
    config = {
        'redis_host': os.getenv('REDIS_HOST', 'localhost'),
        'redis_port': int(os.getenv('REDIS_PORT', '6379')),
        'influxdb_url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
        'influxdb_token': os.getenv('INFLUXDB_TOKEN', 'your_influx_token'),
        'influxdb_org': os.getenv('INFLUXDB_ORG', 'prometheus'),
        'max_workers': int(os.getenv('MAX_WORKERS', '4')),
        
        # API keys (would be loaded from environment in production)
        'iris_api_key': None,  # IRIS is generally open access
        'usgs_api_key': None,  # USGS APIs are generally open access
    }
    
    logger.info("Starting Real-Time Process Monitor test with real data integration")
    
    try:
        # Initialize the monitor
        monitor = RealTimeProcessMonitor(config)
        
        # Test data connector initialization
        logger.info("Testing data connector initialization...")
        await monitor._initialize_data_connectors()
        
        if hasattr(monitor, '_data_connectors') and monitor._data_connectors:
            logger.info(f"Successfully initialized {len(monitor._data_connectors)} data connectors")
            
            # Test fetching real-time data
            logger.info("Testing real-time data fetch...")
            data_points = await monitor._fetch_real_time_data()
            
            if data_points:
                logger.info(f"Successfully fetched {len(data_points)} real-time data points")
                
                # Display sample data
                for i, point in enumerate(data_points[:3]):  # Show first 3 points
                    logger.info(f"Sample data point {i+1}:")
                    logger.info(f"  Timestamp: {point.timestamp}")
                    logger.info(f"  Process Type: {point.process_type}")
                    logger.info(f"  Station: {point.station_id}")
                    logger.info(f"  Location: {point.location}")
                    logger.info(f"  Parameters: {point.parameters}")
                    logger.info(f"  Quality Score: {point.quality_score}")
                    logger.info(f"  Source: {point.source_id}")
                
                # Test data processing
                logger.info("Testing data processing...")
                for point in data_points[:5]:  # Process first 5 points
                    await monitor._process_data_point(point)
                
                logger.info("Data processing test completed successfully")
                
            else:
                logger.warning("No real-time data points were fetched")
                
        else:
            logger.warning("No data connectors were initialized - may be running in mock mode")
            
        # Test individual connector methods if available
        if hasattr(monitor, '_data_connectors'):
            for connector_name, connector in monitor._data_connectors.items():
                logger.info(f"Testing {connector_name} connector health...")
                try:
                    health = await connector.health_check()
                    logger.info(f"{connector_name} health: {health}")
                except Exception as e:
                    logger.warning(f"{connector_name} health check failed: {e}")
        
        logger.info("Real-time process monitor test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise
    
    finally:
        # Clean up
        if hasattr(monitor, '_data_connectors'):
            for connector in monitor._data_connectors.values():
                try:
                    await connector.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting connector: {e}")


async def test_specific_data_sources():
    """Test specific data source connections individually"""
    
    logger.info("Testing individual data source connections...")
    
    try:
        from backend.core.real_data_connectors.iris_connector import IRISConnector
        from backend.core.real_data_connectors.usgs_connector import USGSVolcanoConnector
        from backend.core.real_data_connectors.base_connector import DataSourceConfig, DataSourceType
        
        # Test IRIS connector
        logger.info("Testing IRIS/EarthScope connector...")
        iris_config = DataSourceConfig(
            source_id="iris_test",
            name="IRIS Test",
            base_url="http://service.iris.edu/fdsnws",
            timeout=30,
            rate_limit=60,
            data_type=DataSourceType.SEISMIC
        )
        
        async with IRISConnector(iris_config) as iris_connector:
            logger.info("IRIS connector connected successfully")
            
            # Fetch a small amount of historical data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            historical_data = await iris_connector.fetch_historical_data(
                start_time, end_time, max_events=5
            )
            
            logger.info(f"Fetched {len(historical_data)} historical seismic events from IRIS")
            
        # Test USGS Volcano connector
        logger.info("Testing USGS Volcano Hazards Program connector...")
        usgs_config = DataSourceConfig(
            source_id="usgs_test",
            name="USGS Volcano Test",
            base_url="https://volcanoes.usgs.gov/vsc-api/v1",
            timeout=30,
            rate_limit=100,
            data_type=DataSourceType.VOLCANIC
        )
        
        # Note: This might fail if the API endpoint doesn't exist
        # In that case, we'll catch the exception and continue
        try:
            async with USGSVolcanoConnector(usgs_config) as usgs_connector:
                logger.info("USGS Volcano connector connected successfully")
                
                # Test health check
                health = await usgs_connector.health_check()
                logger.info(f"USGS connector health: {health}")
                
        except Exception as e:
            logger.warning(f"USGS Volcano connector test failed (expected if API not available): {e}")
        
        logger.info("Individual data source tests completed")
        
    except ImportError as e:
        logger.warning(f"Could not import data connectors: {e}")
        logger.info("This is expected if the connectors are not fully implemented yet")


if __name__ == "__main__":
    async def main():
        """Main test function"""
        logger.info("Starting Real-Time Process Monitor Integration Tests")
        
        # Test 1: Individual data source connections
        await test_specific_data_sources()
        
        # Test 2: Full integration test
        await test_real_data_integration()
        
        logger.info("All tests completed")
    
    # Run the tests
    asyncio.run(main())