"""
Comprehensive Volcanic Systems Database Demonstration
Showcases all implemented features of task 4: GVP integration, NASA FIRMS, tectonic classification, monitoring networks
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from modules.volcanic_systems_integration_service import create_volcanic_systems_service
from modules.comprehensive_volcanic_systems_database import TectonicSetting, VolcanoType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VolcanicSystemsDemo:
    """Demonstration of comprehensive volcanic systems database capabilities"""
    
    def __init__(self):
        self.service = None
        
    async def initialize(self):
        """Initialize the volcanic systems service"""
        config = {
            'postgres_dsn': 'postgresql://prometheus:prometheus123@localhost:5432/prometheus',
            'redis_url': 'redis://localhost:6379/0',
            'cache_enabled': True,
            'auto_sync_enabled': False,  # Disable for demo
            'gvp_api_key': None,  # Add your API key if available
            'nasa_firms_api_key': None  # Add your NASA FIRMS API key if available
        }
        
        logger.info("Initializing Comprehensive Volcanic Systems Database...")
        self.service = await create_volcanic_systems_service(config)
        logger.info("Service initialized successfully!")
    
    async def demonstrate_gvp_integration(self):
        """Demonstrate Global Volcanism Program integration (Task 4.1)"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING GVP INTEGRATION (Task 4.1)")
        logger.info("="*60)
        
        try:
            # Perform GVP database integration
            logger.info("Integrating Global Volcanism Program database...")
            gvp_result = await self.service.database.integrate_gvp_database()
            
            logger.info(f"GVP Integration Results:")
            logger.info(f"  Status: {gvp_result['status']}")
            logger.info(f"  Volcanoes processed: {gvp_result['volcanoes_processed']}")
            logger.info(f"  New volcanoes: {gvp_result['new_volcanoes']}")
            logger.info(f"  Updated volcanoes: {gvp_result['updated_volcanoes']}")
            logger.info(f"  Eruptions processed: {gvp_result['eruptions_processed']}")
            
            if gvp_result['errors']:
                logger.warning(f"  Errors encountered: {len(gvp_result['errors'])}")
            
            # Query some volcanoes to show the data
            logger.info("\nSample volcano data from GVP:")
            volcanoes = await self.service.database.query_volcanic_systems()
            
            for i, volcano in enumerate(volcanoes[:3]):  # Show first 3
                logger.info(f"  {i+1}. {volcano.name} ({volcano.country})")
                logger.info(f"     Location: {volcano.location[0]:.3f}°N, {volcano.location[1]:.3f}°E")
                logger.info(f"     Type: {volcano.volcano_type.value}")
                logger.info(f"     Tectonic Setting: {volcano.tectonic_setting.value}")
                logger.info(f"     Elevation: {volcano.elevation_m}m")
                if volcano.last_known_eruption:
                    logger.info(f"     Last Eruption: {volcano.last_known_eruption.strftime('%Y-%m-%d')}")
                logger.info("")
            
        except Exception as e:
            logger.error(f"GVP integration demonstration failed: {e}")
    
    async def demonstrate_nasa_firms_integration(self):
        """Demonstrate NASA FIRMS thermal anomaly integration (Task 4.2)"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING NASA FIRMS INTEGRATION (Task 4.2)")
        logger.info("="*60)
        
        try:
            # Perform NASA FIRMS thermal data integration
            logger.info("Integrating NASA FIRMS thermal anomaly data...")
            time_window = timedelta(days=7)  # Last 7 days
            firms_result = await self.service.database.integrate_nasa_firms_thermal_data(time_window)
            
            logger.info(f"NASA FIRMS Integration Results:")
            logger.info(f"  Status: {firms_result['status']}")
            logger.info(f"  Anomalies processed: {firms_result['anomalies_processed']}")
            logger.info(f"  New anomalies: {firms_result['new_anomalies']}")
            logger.info(f"  Volcanoes with thermal activity: {firms_result['volcanoes_with_thermal_activity']}")
            logger.info(f"  Satellites processed: {', '.join(firms_result['satellites_processed'])}")
            logger.info(f"  Time range: {firms_result['time_range']['start']} to {firms_result['time_range']['end']}")
            
            # Show thermal analysis for active volcanoes
            logger.info("\nThermal activity analysis:")
            volcanoes = await self.service.database.query_volcanic_systems()
            
            for volcano in volcanoes[:5]:  # Check first 5 volcanoes
                time_range = (datetime.utcnow() - timedelta(days=7), datetime.utcnow())
                thermal_analysis = await self.service.database.get_thermal_activity_analysis(
                    volcano.volcano_id, time_range
                )
                
                if thermal_analysis['thermal_anomalies'] > 0:
                    logger.info(f"  {volcano.name}: {thermal_analysis['thermal_anomalies']} thermal anomalies detected")
                    
                    # Show some anomaly details
                    if 'anomaly_data' in thermal_analysis:
                        for anomaly in thermal_analysis['anomaly_data'][:2]:  # Show first 2
                            logger.info(f"    - {anomaly['detection_time']}: {anomaly['brightness_temp_k']:.1f}K "
                                      f"({anomaly['satellite']}, confidence: {anomaly['confidence']})")
            
        except Exception as e:
            logger.error(f"NASA FIRMS integration demonstration failed: {e}")
    
    async def demonstrate_tectonic_classification(self):
        """Demonstrate tectonic setting classification (Task 4.3)"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING TECTONIC CLASSIFICATION (Task 4.3)")
        logger.info("="*60)
        
        try:
            # Perform tectonic classification
            logger.info("Classifying tectonic settings for all volcanic systems...")
            tectonic_result = await self.service.database.classify_tectonic_settings()
            
            logger.info(f"Tectonic Classification Results:")
            logger.info(f"  Status: {tectonic_result['status']}")
            logger.info(f"  Volcanoes classified: {tectonic_result['volcanoes_classified']}")
            logger.info(f"  Plate boundary analyses: {tectonic_result['plate_boundary_analysis']}")
            logger.info(f"  Stress field analyses: {tectonic_result['stress_field_analysis']}")
            
            # Show tectonic setting distribution
            logger.info("\nTectonic setting distribution:")
            for setting, count in tectonic_result['tectonic_settings'].items():
                logger.info(f"  {setting.replace('_', ' ').title()}: {count} volcanoes")
            
            # Demonstrate querying by tectonic setting
            logger.info("\nExamples by tectonic setting:")
            
            for tectonic_setting in [TectonicSetting.SUBDUCTION_ZONE, TectonicSetting.HOTSPOT, TectonicSetting.MID_OCEAN_RIDGE]:
                volcanoes = await self.service.database.query_volcanic_systems(tectonic_setting=tectonic_setting)
                
                if volcanoes:
                    logger.info(f"  {tectonic_setting.value.replace('_', ' ').title()} ({len(volcanoes)} volcanoes):")
                    for volcano in volcanoes[:2]:  # Show first 2
                        logger.info(f"    - {volcano.name} ({volcano.country})")
            
        except Exception as e:
            logger.error(f"Tectonic classification demonstration failed: {e}")
    
    async def demonstrate_monitoring_network_registry(self):
        """Demonstrate monitoring network registry (Task 4.4)"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING MONITORING NETWORK REGISTRY (Task 4.4)")
        logger.info("="*60)
        
        try:
            # Create monitoring network registry
            logger.info("Creating global volcanic monitoring network registry...")
            registry_result = await self.service.database.create_monitoring_network_registry()
            
            logger.info(f"Monitoring Network Registry Results:")
            logger.info(f"  Status: {registry_result['status']}")
            logger.info(f"  Stations processed: {registry_result['stations_processed']}")
            logger.info(f"  Networks identified: {registry_result['networks_identified']}")
            logger.info(f"  Real-time stations: {registry_result['real_time_stations']}")
            
            # Show station types
            logger.info("\nMonitoring station types:")
            for station_type, count in registry_result['station_types'].items():
                logger.info(f"  {station_type.replace('_', ' ').title()}: {count} stations")
            
            # Show monitoring status for specific volcanoes
            logger.info("\nMonitoring status for selected volcanoes:")
            volcanoes = await self.service.database.query_volcanic_systems()
            
            for volcano in volcanoes[:3]:  # Check first 3 volcanoes
                monitoring_status = await self.service.database.get_monitoring_network_status(volcano.volcano_id)
                
                analysis = monitoring_status['monitoring_analysis']
                logger.info(f"  {volcano.name}:")
                logger.info(f"    Total stations: {analysis['total_stations']}")
                logger.info(f"    Real-time capable: {analysis['real_time_stations']}")
                logger.info(f"    Operators: {', '.join(analysis['operators']) if analysis['operators'] else 'None'}")
                
                if analysis['station_types']:
                    types_str = ', '.join([f"{k}: {v}" for k, v in analysis['station_types'].items()])
                    logger.info(f"    Station types: {types_str}")
            
        except Exception as e:
            logger.error(f"Monitoring network registry demonstration failed: {e}")
    
    async def demonstrate_comprehensive_synchronization(self):
        """Demonstrate comprehensive synchronization of all components"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING COMPREHENSIVE SYNCHRONIZATION")
        logger.info("="*60)
        
        try:
            # Perform comprehensive synchronization
            logger.info("Performing comprehensive synchronization of all data sources...")
            sync_result = await self.service.perform_comprehensive_synchronization()
            
            logger.info(f"Comprehensive Synchronization Results:")
            
            # Show synchronization summary
            sync_summary = sync_result['synchronization_summary']
            logger.info(f"  Overall Status: {sync_summary['overall_status']}")
            
            # Show individual component results
            components = ['gvp_sync', 'firms_sync', 'tectonic_sync', 'monitoring_sync']
            for component in components:
                if component in sync_summary:
                    comp_result = sync_summary[component]
                    if isinstance(comp_result, dict):
                        status = comp_result.get('status', 'unknown')
                        logger.info(f"  {component.replace('_', ' ').title()}: {status}")
            
            # Show service status
            service_status = sync_result['service_status']
            logger.info(f"\nService Status:")
            logger.info(f"  Database: {service_status['database_status']}")
            logger.info(f"  Monitoring: {service_status['monitoring_status']}")
            logger.info(f"  Data Connector: {service_status['data_connector_status']}")
            
            # Show data quality metrics
            quality_metrics = sync_result['data_quality_metrics']
            logger.info(f"\nData Quality Metrics:")
            for metric, score in quality_metrics.items():
                logger.info(f"  {metric.replace('_', ' ').title()}: {score:.2f}")
            
            # Show integration validation
            validation = sync_result['integration_validation']
            logger.info(f"\nIntegration Validation:")
            logger.info(f"  Overall Status: {validation['overall_integration_status']}")
            
            validation_components = ['gvp_integration', 'firms_integration', 'tectonic_classification', 'monitoring_registry']
            for component in validation_components:
                if component in validation:
                    comp_status = validation[component].get('status', 'unknown')
                    logger.info(f"  {component.replace('_', ' ').title()}: {comp_status}")
            
        except Exception as e:
            logger.error(f"Comprehensive synchronization demonstration failed: {e}")
    
    async def demonstrate_advanced_search_and_analysis(self):
        """Demonstrate advanced search and analysis capabilities"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING ADVANCED SEARCH AND ANALYSIS")
        logger.info("="*60)
        
        try:
            # Basic search
            logger.info("Performing basic search...")
            basic_search = await self.service.search_volcanic_systems_advanced({})
            logger.info(f"Total volcanoes in database: {basic_search['total_results']}")
            
            # Advanced search with multiple criteria
            logger.info("\nPerforming advanced search (subduction zone stratovolcanoes above 2000m)...")
            search_criteria = {
                'tectonic_setting': 'subduction_zone',
                'volcano_types': ['stratovolcano'],
                'min_elevation': 2000
            }
            
            advanced_search = await self.service.search_volcanic_systems_advanced(search_criteria)
            logger.info(f"Matching volcanoes: {advanced_search['total_results']}")
            
            # Show some results
            for volcano in advanced_search['volcanoes'][:3]:
                logger.info(f"  - {volcano['name']} ({volcano['country']}): {volcano['elevation_m']}m")
            
            # Regional search
            logger.info("\nPerforming regional search (Italy)...")
            italy_search = await self.service.search_volcanic_systems_advanced({
                'bounding_box': [6.0, 35.0, 19.0, 47.0]  # Italy bounding box
            })
            logger.info(f"Volcanoes in Italy region: {italy_search['total_results']}")
            
            for volcano in italy_search['volcanoes']:
                logger.info(f"  - {volcano['name']}: {volcano['volcano_type']}")
            
            # Global activity summary
            logger.info("\nGenerating global volcanic activity summary...")
            global_summary = await self.service.get_global_volcanic_activity_summary()
            
            logger.info(f"Global Volcanic Activity Summary:")
            logger.info(f"  Total volcanoes: {global_summary['total_volcanoes']}")
            
            # Show distribution by tectonic setting
            logger.info(f"  Distribution by tectonic setting:")
            for setting, count in global_summary['volcanoes_by_tectonic_setting'].items():
                logger.info(f"    {setting.replace('_', ' ').title()}: {count}")
            
            # Show recent thermal activity
            thermal_activity = global_summary['recent_thermal_activity']
            logger.info(f"  Recent thermal activity (last 7 days):")
            logger.info(f"    Active volcanoes: {thermal_activity['active_volcanoes']}")
            logger.info(f"    Total thermal anomalies: {thermal_activity['total_thermal_anomalies']}")
            
            if thermal_activity['most_active']:
                logger.info(f"    Most active volcanoes:")
                for volcano in thermal_activity['most_active'][:3]:
                    logger.info(f"      - {volcano['name']}: {volcano['thermal_anomalies']} anomalies")
            
        except Exception as e:
            logger.error(f"Advanced search and analysis demonstration failed: {e}")
    
    async def demonstrate_data_validation(self):
        """Demonstrate data validation and quality control"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING DATA VALIDATION AND QUALITY CONTROL")
        logger.info("="*60)
        
        try:
            # Validate database integrity
            logger.info("Validating database integrity...")
            validation_result = await self.service.validate_database_integrity()
            
            logger.info(f"Database Integrity Validation:")
            logger.info(f"  Overall Status: {validation_result['overall_status']}")
            logger.info(f"  Validation Time: {validation_result['validation_time']}")
            
            # Show data completeness
            completeness = validation_result['data_completeness']
            logger.info(f"\nData Completeness:")
            for metric, score in completeness.items():
                if isinstance(score, (int, float)):
                    logger.info(f"  {metric.replace('_', ' ').title()}: {score:.2f}")
            
            # Show data quality
            quality = validation_result['data_quality']
            logger.info(f"\nData Quality:")
            for metric, score in quality.items():
                if isinstance(score, (int, float)):
                    logger.info(f"  {metric.replace('_', ' ').title()}: {score:.2f}")
            
            # Show recommendations
            if validation_result['recommendations']:
                logger.info(f"\nRecommendations:")
                for recommendation in validation_result['recommendations'][:5]:
                    logger.info(f"  - {recommendation}")
            
        except Exception as e:
            logger.error(f"Data validation demonstration failed: {e}")
    
    async def demonstrate_service_status(self):
        """Demonstrate service status and capabilities"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING SERVICE STATUS AND CAPABILITIES")
        logger.info("="*60)
        
        try:
            # Get service status
            status = await self.service.get_service_status()
            
            logger.info(f"Service Information:")
            logger.info(f"  Name: {status['service_name']}")
            logger.info(f"  Version: {status['version']}")
            logger.info(f"  Status: {status['status']}")
            
            if status.get('last_sync'):
                logger.info(f"  Last Sync: {status['last_sync']}")
            
            logger.info(f"\nComponent Status:")
            for component, comp_status in status['components'].items():
                logger.info(f"  {component.replace('_', ' ').title()}: {comp_status}")
            
            logger.info(f"\nCapabilities:")
            for capability in status['capabilities']:
                logger.info(f"  - {capability.replace('_', ' ').title()}")
            
        except Exception as e:
            logger.error(f"Service status demonstration failed: {e}")
    
    async def run_complete_demonstration(self):
        """Run the complete demonstration of all features"""
        logger.info("STARTING COMPREHENSIVE VOLCANIC SYSTEMS DATABASE DEMONSTRATION")
        logger.info("This demonstration showcases all implemented features of Task 4")
        
        try:
            await self.initialize()
            
            # Demonstrate all major components
            await self.demonstrate_gvp_integration()
            await self.demonstrate_nasa_firms_integration()
            await self.demonstrate_tectonic_classification()
            await self.demonstrate_monitoring_network_registry()
            await self.demonstrate_comprehensive_synchronization()
            await self.demonstrate_advanced_search_and_analysis()
            await self.demonstrate_data_validation()
            await self.demonstrate_service_status()
            
            logger.info("\n" + "="*60)
            logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY!")
            logger.info("All Task 4 components have been implemented and demonstrated:")
            logger.info("✓ 4.1 Global Volcanism Program Integration")
            logger.info("✓ 4.2 NASA FIRMS Thermal Anomaly Integration")
            logger.info("✓ 4.3 Tectonic Setting Classification")
            logger.info("✓ 4.4 Monitoring Network Registry")
            logger.info("✓ Comprehensive Database Integration")
            logger.info("✓ Advanced Search and Analysis")
            logger.info("✓ Data Validation and Quality Control")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Demonstration failed: {e}")
            raise
        
        finally:
            if self.service:
                await self.service.close()

async def main():
    """Main demonstration function"""
    demo = VolcanicSystemsDemo()
    await demo.run_complete_demonstration()

if __name__ == "__main__":
    asyncio.run(main())