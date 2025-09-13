"""
Real Data Connectors Framework Usage Example

This example demonstrates how to use the Real Data Connectors Framework
to fetch authentic geological data from multiple sources with health
monitoring and failover capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the real data connectors framework
from backend.core.real_data_connectors import (
    ConnectorFactory,
    DataSourceConfig,
    DataSourceType,
    HealthStatus
)


async def main():
    """Main example function demonstrating the Real Data Connectors Framework"""
    
    logger.info("Starting Real Data Connectors Framework Example")
    
    # Initialize the connector factory
    factory = ConnectorFactory("backend/config/data_sources.yaml")
    
    try:
        # Initialize all connectors from configuration
        await factory.initialize()
        logger.info("Connector factory initialized successfully")
        
        # List available connectors
        connectors = factory.list_connectors()
        logger.info(f"Available connectors: {connectors}")
        
        # Test all connectors
        logger.info("Testing all connectors...")
        test_results = await factory.test_all_connectors()
        
        for source_id, result in test_results.items():
            if result['status'] == 'success':
                logger.info(f"✓ {source_id}: Connection successful")
            else:
                logger.error(f"✗ {source_id}: {result.get('message', 'Connection failed')}")
        
        # Example 1: Fetch real-time seismic data from IRIS
        await example_iris_real_time_data(factory)
        
        # Example 2: Fetch volcanic alerts from USGS
        await example_usgs_volcanic_alerts(factory)
        
        # Example 3: Fetch satellite data from NASA EarthData
        await example_nasa_satellite_data(factory)
        
        # Example 4: Fetch InSAR deformation data from ASF-Hyp3
        await example_asf_insar_data(factory)
        
        # Example 5: Fetch geochemical data from multiple databases
        await example_geochemical_data(factory)
        
        # Example 6: Health monitoring and failover demonstration
        await example_health_monitoring(factory)
        
        # Example 7: Historical data fetching
        await example_historical_data_fetch(factory)
        
        # Generate comprehensive health report
        health_report = factory.get_health_report()
        logger.info("=== Health Report ===")
        logger.info(f"Total sources: {health_report['summary']['total_sources']}")
        logger.info(f"Healthy sources: {health_report['summary']['healthy_sources']}")
        logger.info(f"Degraded sources: {health_report['summary']['degraded_sources']}")
        logger.info(f"Unhealthy sources: {health_report['summary']['unhealthy_sources']}")
        
    except Exception as e:
        logger.error(f"Error in main example: {e}")
        
    finally:
        # Shutdown all connectors
        await factory.shutdown()
        logger.info("Connector factory shutdown complete")


async def example_iris_real_time_data(factory: ConnectorFactory):
    """Example: Fetch real-time seismic data from IRIS/EarthScope"""
    
    logger.info("=== Example 1: IRIS Real-Time Seismic Data ===")
    
    try:
        # Get the active IRIS connector
        iris_connector = factory.get_active_connector('seismic_data')
        
        if not iris_connector:
            logger.warning("IRIS connector not available")
            return
        
        # Fetch real-time seismic data for 30 seconds
        logger.info("Fetching real-time seismic data...")
        
        observation_count = 0
        timeout = asyncio.create_task(asyncio.sleep(30))  # 30 second timeout
        
        async for observation in iris_connector.fetch_real_time_data(
            networks=['IU', 'II'],  # Global networks
            min_magnitude=4.0,
            fetch_interval=10  # Check every 10 seconds
        ):
            logger.info(f"Seismic Event: M{observation.value.get('magnitude', 'N/A')} "
                       f"at {observation.latitude:.2f}, {observation.longitude:.2f} "
                       f"({observation.timestamp})")
            
            observation_count += 1
            
            if timeout.done():
                break
        
        timeout.cancel()
        logger.info(f"Fetched {observation_count} seismic observations")
        
    except Exception as e:
        logger.error(f"Error fetching IRIS data: {e}")


async def example_usgs_volcanic_alerts(factory: ConnectorFactory):
    """Example: Fetch volcanic alerts from USGS Volcano Hazards Program"""
    
    logger.info("=== Example 2: USGS Volcanic Alerts ===")
    
    try:
        # Get the active USGS volcano connector
        usgs_connector = factory.get_active_connector('volcanic_data')
        
        if not usgs_connector:
            logger.warning("USGS volcano connector not available")
            return
        
        # Fetch current volcanic alerts for 20 seconds
        logger.info("Fetching volcanic alerts...")
        
        alert_count = 0
        timeout = asyncio.create_task(asyncio.sleep(20))
        
        async for observation in usgs_connector.fetch_real_time_data(
            fetch_interval=15  # Check every 15 seconds
        ):
            if observation.observation_type == "volcanic_alert":
                alert_level = observation.value.get('alert_level', 'unknown')
                volcano_name = observation.value.get('volcano_name', 'Unknown')
                
                logger.info(f"Volcanic Alert: {volcano_name} - {alert_level.upper()} "
                           f"at {observation.latitude:.2f}, {observation.longitude:.2f}")
                
                alert_count += 1
            
            if timeout.done():
                break
        
        timeout.cancel()
        logger.info(f"Processed {alert_count} volcanic alerts")
        
    except Exception as e:
        logger.error(f"Error fetching USGS volcanic data: {e}")


async def example_nasa_satellite_data(factory: ConnectorFactory):
    """Example: Fetch satellite observations from NASA EarthData"""
    
    logger.info("=== Example 3: NASA EarthData Satellite Observations ===")
    
    try:
        # Get the active NASA EarthData connector
        nasa_connector = factory.get_active_connector('satellite_data')
        
        if not nasa_connector:
            logger.warning("NASA EarthData connector not available")
            return
        
        # Fetch recent satellite observations
        logger.info("Fetching satellite thermal anomalies...")
        
        observation_count = 0
        timeout = asyncio.create_task(asyncio.sleep(25))
        
        async for observation in nasa_connector.fetch_real_time_data(
            instruments=['MODIS', 'VIIRS'],
            bbox=[-125, 40, -110, 50],  # Western US
            fetch_interval=20
        ):
            if observation.observation_type in ["thermal_anomaly", "fire_detection"]:
                temp_k = observation.value.get('brightness_temperature_k', 0)
                instrument = observation.value.get('instrument', 'Unknown')
                
                logger.info(f"Thermal Anomaly: {temp_k:.1f}K detected by {instrument} "
                           f"at {observation.latitude:.3f}, {observation.longitude:.3f}")
                
                observation_count += 1
            
            if timeout.done():
                break
        
        timeout.cancel()
        logger.info(f"Detected {observation_count} thermal anomalies")
        
    except Exception as e:
        logger.error(f"Error fetching NASA satellite data: {e}")


async def example_asf_insar_data(factory: ConnectorFactory):
    """Example: Fetch InSAR deformation data from ASF-Hyp3"""
    
    logger.info("=== Example 4: ASF-Hyp3 InSAR Deformation Data ===")
    
    try:
        # Get the active ASF-Hyp3 connector
        asf_connector = factory.get_active_connector('insar_data')
        
        if not asf_connector:
            logger.warning("ASF-Hyp3 connector not available")
            return
        
        # Fetch recent InSAR processing results
        logger.info("Fetching InSAR deformation measurements...")
        
        deformation_count = 0
        timeout = asyncio.create_task(asyncio.sleep(15))
        
        async for observation in asf_connector.fetch_real_time_data(
            processing_types=['INSAR_GAMMA'],
            bbox=[-155, 19, -154, 20],  # Hawaii (volcanic region)
            fetch_interval=30
        ):
            if observation.observation_type == "ground_deformation":
                displacement = observation.value.get('displacement_mm', 0)
                coherence = observation.value.get('coherence', 0)
                
                logger.info(f"Ground Deformation: {displacement:.1f}mm displacement "
                           f"(coherence: {coherence:.2f}) at "
                           f"{observation.latitude:.4f}, {observation.longitude:.4f}")
                
                deformation_count += 1
            
            if timeout.done():
                break
        
        timeout.cancel()
        logger.info(f"Processed {deformation_count} deformation measurements")
        
    except Exception as e:
        logger.error(f"Error fetching ASF InSAR data: {e}")


async def example_geochemical_data(factory: ConnectorFactory):
    """Example: Fetch geochemical data from multiple databases"""
    
    logger.info("=== Example 5: Geochemical Database Integration ===")
    
    try:
        # Get the active geochemical connector
        geochem_connector = factory.get_active_connector('geochemical_data')
        
        if not geochem_connector:
            logger.warning("Geochemical connector not available")
            return
        
        # Fetch recent geochemical analyses
        logger.info("Fetching geochemical analyses...")
        
        analysis_count = 0
        timeout = asyncio.create_task(asyncio.sleep(10))
        
        async for observation in geochem_connector.fetch_real_time_data(
            databases=['earthchem', 'georoc'],
            sample_types=['volcanic', 'igneous'],
            fetch_interval=60  # Check every minute
        ):
            if observation.observation_type == "geochemical_analysis":
                sio2 = observation.value.get('SiO2_wt_percent', 0)
                rock_type = observation.value.get('rock_type', 'Unknown')
                
                logger.info(f"Geochemical Analysis: {rock_type} with {sio2:.1f}% SiO2 "
                           f"from {observation.data_source} at "
                           f"{observation.latitude:.3f}, {observation.longitude:.3f}")
                
                analysis_count += 1
            
            if timeout.done():
                break
        
        timeout.cancel()
        logger.info(f"Processed {analysis_count} geochemical analyses")
        
    except Exception as e:
        logger.error(f"Error fetching geochemical data: {e}")


async def example_health_monitoring(factory: ConnectorFactory):
    """Example: Demonstrate health monitoring and failover"""
    
    logger.info("=== Example 6: Health Monitoring and Failover ===")
    
    try:
        # Get health status for all sources
        health_status = factory.health_monitor.get_all_health_status()
        
        logger.info("Current health status:")
        for source_id, metrics in health_status.items():
            status_emoji = {
                'healthy': '✅',
                'degraded': '⚠️',
                'unhealthy': '❌',
                'critical': '🚨',
                'unknown': '❓'
            }.get(metrics.status.value, '❓')
            
            logger.info(f"{status_emoji} {source_id}: {metrics.status.value} "
                       f"(Response: {metrics.response_time_ms:.1f}ms, "
                       f"Success Rate: {metrics.success_rate:.1%}, "
                       f"Errors: {metrics.error_count})")
        
        # Demonstrate manual failover (if fallback sources exist)
        active_sources = factory.health_monitor.active_sources
        logger.info(f"Active sources: {active_sources}")
        
        # Add alert callback for demonstration
        async def demo_alert_handler(source_id: str, status: HealthStatus, alert_data: dict):
            logger.warning(f"🚨 ALERT: {source_id} status changed to {status.value}")
            logger.warning(f"   Details: {alert_data}")
        
        factory.health_monitor.add_alert_callback(demo_alert_handler)
        
        # Wait a bit to see if any alerts are generated
        await asyncio.sleep(5)
        
    except Exception as e:
        logger.error(f"Error in health monitoring example: {e}")


async def example_historical_data_fetch(factory: ConnectorFactory):
    """Example: Fetch historical data from multiple sources"""
    
    logger.info("=== Example 7: Historical Data Fetching ===")
    
    try:
        # Define time range for historical data (last 7 days)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        logger.info(f"Fetching historical data from {start_time} to {end_time}")
        
        # Fetch historical seismic data
        iris_connector = factory.get_active_connector('seismic_data')
        if iris_connector:
            logger.info("Fetching historical seismic events...")
            seismic_data = await iris_connector.fetch_historical_data(
                start_time, end_time,
                min_magnitude=5.0,
                max_events=50
            )
            logger.info(f"Retrieved {len(seismic_data)} historical seismic events")
        
        # Fetch historical volcanic activity
        usgs_connector = factory.get_active_connector('volcanic_data')
        if usgs_connector:
            logger.info("Fetching historical volcanic activity...")
            volcanic_data = await usgs_connector.fetch_historical_data(
                start_time, end_time,
                max_records=100
            )
            logger.info(f"Retrieved {len(volcanic_data)} historical volcanic observations")
        
        # Fetch historical geochemical data
        geochem_connector = factory.get_active_connector('geochemical_data')
        if geochem_connector:
            logger.info("Fetching historical geochemical data...")
            geochem_data = await geochem_connector.fetch_historical_data(
                start_time, end_time,
                databases=['earthchem'],
                max_results=50
            )
            logger.info(f"Retrieved {len(geochem_data)} historical geochemical samples")
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")


async def example_dynamic_connector_management(factory: ConnectorFactory):
    """Example: Dynamic connector addition and removal"""
    
    logger.info("=== Example 8: Dynamic Connector Management ===")
    
    try:
        # Add a new connector dynamically
        new_config = {
            'name': 'Custom Test Source',
            'base_url': 'https://api.custom-test.com',
            'timeout': 30,
            'max_retries': 3,
            'rate_limit': 50,
            'data_type': 'seismic'
        }
        
        success = await factory.add_connector('custom_test_source', 'iris', new_config)
        if success:
            logger.info("✅ Successfully added custom connector")
        else:
            logger.error("❌ Failed to add custom connector")
        
        # List connectors after addition
        connectors = factory.list_connectors()
        logger.info(f"Connectors after addition: {connectors}")
        
        # Test the new connector
        if 'custom_test_source' in connectors:
            test_result = await factory.test_connector('custom_test_source')
            logger.info(f"Custom connector test result: {test_result}")
        
        # Remove the connector
        removed = await factory.remove_connector('custom_test_source')
        if removed:
            logger.info("✅ Successfully removed custom connector")
        else:
            logger.error("❌ Failed to remove custom connector")
        
    except Exception as e:
        logger.error(f"Error in dynamic connector management: {e}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())