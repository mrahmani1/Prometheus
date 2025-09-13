"""
Monitoring Network Registry Demonstration

This script demonstrates the comprehensive functionality of the Monitoring Network Registry
system, including real-world scenarios for volcanic observatory management, monitoring
station inventory, coverage analysis, and collaborative monitoring coordination.

This example showcases production-grade features including:
- Global volcanic observatory database management
- Monitoring station inventory with detailed instrumentation
- Real-time data stream integration
- Network coverage analysis and gap identification
- Performance assessment and quality metrics
- Collaborative monitoring network coordination

Author: Prometheus AI Platform
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Import the monitoring network registry
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from monitoring_network_registry import (
    MonitoringNetworkRegistry,
    MonitoringNetworkType,
    InstrumentType,
    DataStreamStatus,
    NetworkStatus,
    GeographicBounds,
    InstrumentSpecification,
    DataStreamInfo,
    NetworkCoverageMetrics,
    PerformanceMetrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringNetworkDemo:
    """
    Comprehensive demonstration of the Monitoring Network Registry system.
    
    This class provides realistic examples of how the registry would be used
    in production environments for managing global volcanic monitoring networks.
    """
    
    def __init__(self):
        """Initialize the demonstration system"""
        self.registry = None
        self.demo_data = {}
        self.results = {}
    
    async def initialize_registry(self):
        """Initialize the monitoring network registry"""
        logger.info("Initializing Monitoring Network Registry...")
        
        # Use SQLite for demonstration (would be PostgreSQL in production)
        database_url = "sqlite:///monitoring_network_demo.db"
        
        self.registry = MonitoringNetworkRegistry(
            database_url=database_url,
            redis_url=None,  # Would use Redis in production
            kafka_config=None  # Would use Kafka in production
        )
        
        logger.info("Registry initialized successfully")
    
    async def populate_global_observatories(self):
        """Populate the registry with real-world volcanic observatories"""
        logger.info("Populating global volcanic observatories...")
        
        # Real volcanic observatories from around the world
        observatories = [
            {
                'name': 'Yellowstone Volcano Observatory',
                'acronym': 'YVO',
                'country': 'United States',
                'region': 'Wyoming',
                'latitude': 44.428,
                'longitude': -110.588,
                'elevation': 2400,
                'established_date': datetime(1996, 1, 1),
                'parent_organization': 'USGS',
                'contact_email': 'yvo@usgs.gov',
                'website_url': 'https://www.usgs.gov/observatories/yellowstone-volcano-observatory',
                'primary_volcanoes': ['yellowstone'],
                'network_types': ['seismic', 'gps_gnss', 'thermal', 'deformation'],
                'status': 'operational',
                'capabilities': {
                    'seismic_monitoring': True,
                    'deformation_monitoring': True,
                    'gas_monitoring': True,
                    'thermal_monitoring': True,
                    'real_time_processing': True,
                    'eruption_forecasting': True
                },
                'staff_count': 25,
                'annual_budget': 5000000.0,
                'data_sharing_policy': 'Open access with attribution',
                'real_time_capability': True,
                'alert_system': True
            },
            {
                'name': 'Cascades Volcano Observatory',
                'acronym': 'CVO',
                'country': 'United States',
                'region': 'Washington',
                'latitude': 46.1951,
                'longitude': -122.1906,
                'elevation': 165,
                'established_date': datetime(1982, 1, 1),
                'parent_organization': 'USGS',
                'contact_email': 'cvo@usgs.gov',
                'website_url': 'https://www.usgs.gov/observatories/cascades-volcano-observatory',
                'primary_volcanoes': ['mount_st_helens', 'mount_rainier', 'mount_baker', 'glacier_peak'],
                'network_types': ['seismic', 'gps_gnss', 'gas_geochemical', 'deformation'],
                'status': 'operational',
                'capabilities': {
                    'seismic_monitoring': True,
                    'deformation_monitoring': True,
                    'gas_monitoring': True,
                    'lahar_detection': True,
                    'real_time_processing': True,
                    'eruption_forecasting': True
                },
                'staff_count': 30,
                'annual_budget': 6500000.0,
                'data_sharing_policy': 'Open access with attribution',
                'real_time_capability': True,
                'alert_system': True
            },
            {
                'name': 'Alaska Volcano Observatory',
                'acronym': 'AVO',
                'country': 'United States',
                'region': 'Alaska',
                'latitude': 61.2181,
                'longitude': -149.9003,
                'elevation': 102,
                'established_date': datetime(1988, 1, 1),
                'parent_organization': 'USGS/UAF/ADGGS',
                'contact_email': 'avo@usgs.gov',
                'website_url': 'https://www.avo.alaska.edu',
                'primary_volcanoes': ['redoubt', 'augustine', 'spurr', 'cleveland'],
                'network_types': ['seismic', 'infrasound', 'thermal', 'deformation'],
                'status': 'operational',
                'capabilities': {
                    'seismic_monitoring': True,
                    'infrasound_monitoring': True,
                    'satellite_monitoring': True,
                    'ash_tracking': True,
                    'aviation_alerts': True
                },
                'staff_count': 20,
                'annual_budget': 4000000.0,
                'data_sharing_policy': 'Open access with attribution',
                'real_time_capability': True,
                'alert_system': True
            },
            {
                'name': 'Istituto Nazionale di Geofisica e Vulcanologia - Catania',
                'acronym': 'INGV-CT',
                'country': 'Italy',
                'region': 'Sicily',
                'latitude': 37.5079,
                'longitude': 15.0830,
                'elevation': 300,
                'established_date': datetime(1999, 1, 1),
                'parent_organization': 'INGV',
                'contact_email': 'info.ct@ingv.it',
                'website_url': 'https://www.ct.ingv.it',
                'primary_volcanoes': ['etna', 'stromboli', 'vulcano'],
                'network_types': ['seismic', 'deformation', 'gas_geochemical', 'thermal'],
                'status': 'operational',
                'capabilities': {
                    'seismic_monitoring': True,
                    'deformation_monitoring': True,
                    'gas_monitoring': True,
                    'thermal_monitoring': True,
                    'lava_flow_modeling': True
                },
                'staff_count': 45,
                'annual_budget': 8000000.0,
                'data_sharing_policy': 'Open access with registration',
                'real_time_capability': True,
                'alert_system': True
            },
            {
                'name': 'Japan Meteorological Agency - Volcanic Division',
                'acronym': 'JMA-VD',
                'country': 'Japan',
                'region': 'Tokyo',
                'latitude': 35.6762,
                'longitude': 139.6503,
                'elevation': 40,
                'established_date': datetime(1956, 1, 1),
                'parent_organization': 'JMA',
                'contact_email': 'volcano@jma.go.jp',
                'website_url': 'https://www.jma.go.jp/jma/en/Activities/volcano.html',
                'primary_volcanoes': ['fuji', 'sakurajima', 'aso', 'ontake'],
                'network_types': ['seismic', 'deformation', 'gas_geochemical', 'thermal'],
                'status': 'operational',
                'capabilities': {
                    'seismic_monitoring': True,
                    'deformation_monitoring': True,
                    'gas_monitoring': True,
                    'thermal_monitoring': True,
                    'national_alert_system': True
                },
                'staff_count': 60,
                'annual_budget': 12000000.0,
                'data_sharing_policy': 'Restricted access',
                'real_time_capability': True,
                'alert_system': True
            }
        ]
        
        # Register observatories
        observatory_ids = {}
        for obs_data in observatories:
            try:
                obs_id = await self.registry.register_volcanic_observatory(obs_data)
                observatory_ids[obs_data['acronym']] = obs_id
                logger.info(f"Registered observatory: {obs_data['name']} ({obs_data['acronym']})")
            except Exception as e:
                logger.error(f"Failed to register {obs_data['name']}: {e}")
        
        self.demo_data['observatory_ids'] = observatory_ids
        logger.info(f"Successfully registered {len(observatory_ids)} observatories")
    
    async def populate_monitoring_stations(self):
        """Populate monitoring stations for each observatory"""
        logger.info("Populating monitoring stations...")
        
        # Yellowstone monitoring stations
        yellowstone_stations = [
            {
                'station_code': 'YNP.LKWY',
                'name': 'Lake Yellowstone Seismic Station',
                'observatory_id': self.demo_data['observatory_ids']['YVO'],
                'volcano_id': 'yellowstone',
                'latitude': 44.428,
                'longitude': -110.588,
                'elevation': 2357,
                'installation_date': datetime(2000, 6, 15),
                'station_type': 'seismic',
                'network_code': 'YNP',
                'site_conditions': 'Bedrock installation near lake shore',
                'access_difficulty': 'moderate',
                'power_source': 'Solar panel with battery backup',
                'communication_method': 'Satellite telemetry',
                'maintenance_schedule': 'Annual summer maintenance',
                'status': 'operational',
                'instruments': [
                    {
                        'type': 'seismometer',
                        'manufacturer': 'Nanometrics',
                        'model': 'Trillium Compact',
                        'serial_number': 'TC120-001',
                        'sampling_rate': 100.0,
                        'sensitivity': 1200.0,
                        'frequency_range': [0.0083, 50.0],
                        'installation_date': '2000-06-15'
                    }
                ],
                'data_availability': 95.5,
                'uptime': 98.2
            },
            {
                'station_code': 'YNP.NRWY',
                'name': 'Norris Geyser Basin GPS Station',
                'observatory_id': self.demo_data['observatory_ids']['YVO'],
                'volcano_id': 'yellowstone',
                'latitude': 44.7255,
                'longitude': -110.7031,
                'elevation': 2275,
                'installation_date': datetime(2005, 8, 20),
                'station_type': 'gps_gnss',
                'network_code': 'YNP',
                'site_conditions': 'Concrete monument in thermal area',
                'access_difficulty': 'easy',
                'power_source': 'Solar panel with battery backup',
                'communication_method': 'Cellular modem',
                'maintenance_schedule': 'Quarterly inspection',
                'status': 'operational',
                'instruments': [
                    {
                        'type': 'gps_receiver',
                        'manufacturer': 'Trimble',
                        'model': 'NetR9',
                        'serial_number': 'NR9-YNP-002',
                        'sampling_rate': 1.0,
                        'accuracy': 0.003,
                        'installation_date': '2005-08-20'
                    }
                ],
                'data_availability': 97.8,
                'uptime': 99.1
            }
        ]
        
        # Mount St. Helens monitoring stations
        msh_stations = [
            {
                'station_code': 'MSH.JRO',
                'name': 'Johnston Ridge Observatory Seismic Station',
                'observatory_id': self.demo_data['observatory_ids']['CVO'],
                'volcano_id': 'mount_st_helens',
                'latitude': 46.2756,
                'longitude': -122.2197,
                'elevation': 1314,
                'installation_date': datetime(1997, 5, 18),
                'station_type': 'seismic',
                'network_code': 'UW',
                'site_conditions': 'Bedrock installation with blast-resistant housing',
                'access_difficulty': 'easy',
                'power_source': 'AC power with UPS backup',
                'communication_method': 'Microwave telemetry',
                'maintenance_schedule': 'Monthly inspection',
                'status': 'operational',
                'instruments': [
                    {
                        'type': 'seismometer',
                        'manufacturer': 'Guralp',
                        'model': 'CMG-40T',
                        'serial_number': 'CMG40T-MSH-001',
                        'sampling_rate': 100.0,
                        'sensitivity': 800.0,
                        'frequency_range': [0.033, 50.0],
                        'installation_date': '1997-05-18'
                    }
                ],
                'data_availability': 98.9,
                'uptime': 99.5
            },
            {
                'station_code': 'MSH.YEL',
                'name': 'Yelm Tiltmeter Station',
                'observatory_id': self.demo_data['observatory_ids']['CVO'],
                'volcano_id': 'mount_st_helens',
                'latitude': 46.2833,
                'longitude': -122.2167,
                'elevation': 1280,
                'installation_date': datetime(2004, 9, 10),
                'station_type': 'deformation',
                'network_code': 'UW',
                'site_conditions': 'Shallow borehole installation',
                'access_difficulty': 'moderate',
                'power_source': 'Solar panel with battery backup',
                'communication_method': 'Radio telemetry',
                'maintenance_schedule': 'Bi-annual calibration',
                'status': 'operational',
                'instruments': [
                    {
                        'type': 'tiltmeter',
                        'manufacturer': 'Applied Geomechanics',
                        'model': 'Model 722',
                        'serial_number': 'AGI722-MSH-003',
                        'sampling_rate': 0.1,
                        'sensitivity': 0.001,
                        'installation_date': '2004-09-10'
                    }
                ],
                'data_availability': 94.2,
                'uptime': 96.8
            }
        ]
        
        # Etna monitoring stations
        etna_stations = [
            {
                'station_code': 'ESLN',
                'name': 'Etna South - La Montagnola Seismic Station',
                'observatory_id': self.demo_data['observatory_ids']['INGV-CT'],
                'volcano_id': 'etna',
                'latitude': 37.7047,
                'longitude': 15.0014,
                'elevation': 2900,
                'installation_date': datetime(1999, 3, 12),
                'station_type': 'seismic',
                'network_code': 'IV',
                'site_conditions': 'Volcanic bedrock at high altitude',
                'access_difficulty': 'difficult',
                'power_source': 'Wind generator with battery backup',
                'communication_method': 'Radio link to base station',
                'maintenance_schedule': 'Weather-dependent access',
                'status': 'operational',
                'instruments': [
                    {
                        'type': 'seismometer',
                        'manufacturer': 'Lennartz',
                        'model': 'LE-3D/20s',
                        'serial_number': 'LE3D20-ETNA-005',
                        'sampling_rate': 125.0,
                        'sensitivity': 400.0,
                        'frequency_range': [0.05, 50.0],
                        'installation_date': '1999-03-12'
                    }
                ],
                'data_availability': 92.1,
                'uptime': 94.7
            }
        ]
        
        # Combine all stations
        all_stations = yellowstone_stations + msh_stations + etna_stations
        
        # Register stations
        station_ids = []
        for station_data in all_stations:
            try:
                station_id = await self.registry.register_monitoring_station(station_data)
                station_ids.append(station_id)
                logger.info(f"Registered station: {station_data['station_code']} - {station_data['name']}")
            except Exception as e:
                logger.error(f"Failed to register {station_data['station_code']}: {e}")
        
        self.demo_data['station_ids'] = station_ids
        logger.info(f"Successfully registered {len(station_ids)} monitoring stations")
    
    async def setup_data_streams(self):
        """Set up data streams for monitoring stations"""
        logger.info("Setting up data streams...")
        
        # Example data streams
        data_streams = [
            {
                'stream_id': 'YNP.LKWY.BHZ',
                'station_id': self.demo_data['station_ids'][0],  # Yellowstone seismic
                'observatory_id': self.demo_data['observatory_ids']['YVO'],
                'parameter_name': 'vertical_velocity',
                'unit': 'm/s',
                'instrument_type': 'seismometer',
                'sampling_rate': 100.0,
                'data_format': 'miniSEED',
                'endpoint_url': 'https://service.iris.edu/fdsnws/dataselect/1/query',
                'authentication_method': 'none',
                'real_time': True,
                'latency': 2.5,
                'availability': 95.5,
                'quality_score': 0.92,
                'last_update': datetime.utcnow(),
                'status': 'active',
                'access_restrictions': 'Open access',
                'data_policy': 'CC BY 4.0',
                'contact_info': 'yvo-data@usgs.gov'
            },
            {
                'stream_id': 'YNP.NRWY.LLZ',
                'station_id': self.demo_data['station_ids'][1],  # Yellowstone GPS
                'observatory_id': self.demo_data['observatory_ids']['YVO'],
                'parameter_name': 'vertical_displacement',
                'unit': 'mm',
                'instrument_type': 'gps_receiver',
                'sampling_rate': 1.0,
                'data_format': 'RINEX',
                'endpoint_url': 'https://www.unavco.org/data/gps-gnss/data-access-methods/dai2/app/dai2.html',
                'authentication_method': 'registration',
                'real_time': False,
                'latency': 3600.0,  # 1 hour
                'availability': 97.8,
                'quality_score': 0.95,
                'last_update': datetime.utcnow(),
                'status': 'active',
                'access_restrictions': 'Registration required',
                'data_policy': 'Open access with registration',
                'contact_info': 'yvo-gps@usgs.gov'
            }
        ]
        
        # Register data streams
        stream_ids = []
        for stream_data in data_streams:
            try:
                stream_id = await self.registry.register_data_stream(stream_data)
                stream_ids.append(stream_id)
                logger.info(f"Registered data stream: {stream_data['stream_id']}")
            except Exception as e:
                logger.error(f"Failed to register stream {stream_data['stream_id']}: {e}")
        
        self.demo_data['stream_ids'] = stream_ids
        logger.info(f"Successfully registered {len(stream_ids)} data streams")
    
    async def demonstrate_coverage_analysis(self):
        """Demonstrate network coverage analysis"""
        logger.info("Demonstrating network coverage analysis...")
        
        # Define regions for analysis
        regions = {
            'Yellowstone': GeographicBounds(44.0, 45.0, -111.0, -110.0),
            'Cascades': GeographicBounds(45.5, 47.0, -123.0, -121.0),
            'Mediterranean': GeographicBounds(37.0, 38.5, 14.5, 16.0)
        }
        
        coverage_results = {}
        
        for region_name, bounds in regions.items():
            logger.info(f"Analyzing coverage for {region_name} region...")
            
            # Analyze coverage for different network types
            region_coverage = {}
            for network_type in [MonitoringNetworkType.SEISMIC, 
                               MonitoringNetworkType.GPS_GNSS, 
                               MonitoringNetworkType.DEFORMATION]:
                try:
                    coverage = await self.registry.analyze_network_coverage(bounds, network_type)
                    region_coverage[network_type.value] = {
                        'coverage_percentage': coverage.coverage_percentage,
                        'station_density': coverage.station_density,
                        'average_distance': coverage.average_inter_station_distance,
                        'redundancy_score': coverage.redundancy_score,
                        'temporal_coverage': coverage.temporal_coverage
                    }
                    logger.info(f"  {network_type.value}: {coverage.coverage_percentage:.1f}% coverage")
                except Exception as e:
                    logger.error(f"Coverage analysis failed for {network_type.value}: {e}")
            
            coverage_results[region_name] = region_coverage
        
        self.results['coverage_analysis'] = coverage_results
        return coverage_results
    
    async def demonstrate_performance_assessment(self):
        """Demonstrate network performance assessment"""
        logger.info("Demonstrating network performance assessment...")
        
        try:
            # Assess overall network performance
            performance = await self.registry.assess_network_performance()
            
            performance_metrics = {
                'uptime_percentage': performance.uptime_percentage,
                'data_completeness': performance.data_completeness,
                'latency_p95': performance.latency_p95,
                'error_rate': performance.error_rate,
                'false_alarm_rate': performance.false_alarm_rate,
                'detection_capability': performance.detection_capability,
                'response_time': performance.response_time,
                'maintenance_frequency': performance.maintenance_frequency
            }
            
            logger.info("Network Performance Metrics:")
            for metric, value in performance_metrics.items():
                logger.info(f"  {metric}: {value:.2f}")
            
            self.results['performance_metrics'] = performance_metrics
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Performance assessment failed: {e}")
            return None
    
    async def demonstrate_gap_identification(self):
        """Demonstrate monitoring gap identification"""
        logger.info("Demonstrating monitoring gap identification...")
        
        # Analyze gaps for Yellowstone region
        yellowstone_bounds = GeographicBounds(44.0, 45.0, -111.0, -110.0)
        
        try:
            gaps = await self.registry.identify_monitoring_gaps(
                yellowstone_bounds, 
                target_coverage=95.0
            )
            
            logger.info(f"Identified {len(gaps)} monitoring gaps:")
            for gap in gaps:
                logger.info(f"  Network: {gap['network_type']}")
                logger.info(f"    Current coverage: {gap['current_coverage']:.1f}%")
                logger.info(f"    Gap: {gap['gap_percentage']:.1f}%")
                logger.info(f"    Additional stations needed: {gap['additional_stations_needed']}")
                logger.info(f"    Priority: {gap['priority']}")
                logger.info(f"    Estimated cost: ${gap['estimated_cost']['total_first_year']:,.0f}")
            
            self.results['monitoring_gaps'] = gaps
            return gaps
            
        except Exception as e:
            logger.error(f"Gap identification failed: {e}")
            return []
    
    async def demonstrate_real_time_integration(self):
        """Demonstrate real-time data integration setup"""
        logger.info("Demonstrating real-time data integration...")
        
        # Example stream configurations
        stream_configs = [
            {
                'stream_id': 'YNP.LKWY.BHZ.RT',
                'endpoint_url': 'wss://rtserve.iris.washington.edu/seedlink',
                'parameter_name': 'seismic_velocity',
                'station_id': self.demo_data['station_ids'][0] if self.demo_data['station_ids'] else 'demo-station'
            },
            {
                'stream_id': 'MSH.JRO.BHZ.RT',
                'endpoint_url': 'wss://pnsn.org/seedlink',
                'parameter_name': 'seismic_velocity',
                'station_id': self.demo_data['station_ids'][2] if len(self.demo_data['station_ids']) > 2 else 'demo-station'
            }
        ]
        
        try:
            setup_report = await self.registry.setup_real_time_data_integration(stream_configs)
            
            logger.info("Real-time Integration Setup Report:")
            logger.info(f"  Streams configured: {setup_report['streams_configured']}")
            logger.info(f"  Streams active: {setup_report['streams_active']}")
            logger.info(f"  Failed streams: {len(setup_report['failed_streams'])}")
            logger.info(f"  Success: {setup_report['success']}")
            
            self.results['real_time_setup'] = setup_report
            return setup_report
            
        except Exception as e:
            logger.error(f"Real-time integration setup failed: {e}")
            return None
    
    async def generate_comprehensive_report(self):
        """Generate comprehensive monitoring network report"""
        logger.info("Generating comprehensive monitoring report...")
        
        try:
            # Generate global report
            global_report = await self.registry.generate_monitoring_report()
            
            # Generate regional reports
            regional_reports = {}
            regions = {
                'North_America': GeographicBounds(25.0, 70.0, -170.0, -50.0),
                'Europe_Mediterranean': GeographicBounds(35.0, 70.0, -10.0, 50.0),
                'Pacific_Ring_of_Fire': GeographicBounds(-60.0, 70.0, 120.0, -70.0)
            }
            
            for region_name, bounds in regions.items():
                try:
                    regional_report = await self.registry.generate_monitoring_report(
                        region_bounds=bounds
                    )
                    regional_reports[region_name] = regional_report
                    logger.info(f"Generated report for {region_name}")
                except Exception as e:
                    logger.error(f"Failed to generate report for {region_name}: {e}")
            
            # Combine reports
            comprehensive_report = {
                'global_summary': global_report,
                'regional_reports': regional_reports,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'total_observatories': global_report['summary']['total_observatories'],
                'total_countries': global_report['summary']['countries']
            }
            
            self.results['comprehensive_report'] = comprehensive_report
            
            logger.info("Comprehensive Report Summary:")
            logger.info(f"  Total observatories: {comprehensive_report['total_observatories']}")
            logger.info(f"  Countries represented: {comprehensive_report['total_countries']}")
            logger.info(f"  Regional reports: {len(regional_reports)}")
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None
    
    def create_visualizations(self):
        """Create visualizations of the monitoring network data"""
        logger.info("Creating visualizations...")
        
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Global Volcanic Monitoring Network Analysis', fontsize=16, fontweight='bold')
            
            # 1. Coverage Analysis by Region
            if 'coverage_analysis' in self.results:
                ax1 = axes[0, 0]
                coverage_data = self.results['coverage_analysis']
                
                regions = list(coverage_data.keys())
                seismic_coverage = [coverage_data[r].get('seismic', {}).get('coverage_percentage', 0) 
                                  for r in regions]
                gps_coverage = [coverage_data[r].get('gps_gnss', {}).get('coverage_percentage', 0) 
                               for r in regions]
                
                x = np.arange(len(regions))
                width = 0.35
                
                ax1.bar(x - width/2, seismic_coverage, width, label='Seismic', alpha=0.8)
                ax1.bar(x + width/2, gps_coverage, width, label='GPS/GNSS', alpha=0.8)
                
                ax1.set_xlabel('Region')
                ax1.set_ylabel('Coverage Percentage (%)')
                ax1.set_title('Network Coverage by Region')
                ax1.set_xticks(x)
                ax1.set_xticklabels(regions, rotation=45)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Performance Metrics
            if 'performance_metrics' in self.results:
                ax2 = axes[0, 1]
                perf_data = self.results['performance_metrics']
                
                metrics = ['uptime_percentage', 'data_completeness', 'detection_capability']
                values = [perf_data.get(m, 0) for m in metrics]
                colors = ['#2E8B57', '#4169E1', '#DC143C']
                
                bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
                ax2.set_ylabel('Percentage (%)')
                ax2.set_title('Network Performance Metrics')
                ax2.set_xticklabels(['Uptime', 'Data\nCompleteness', 'Detection\nCapability'])
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{value:.1f}%', ha='center', va='bottom')
                
                ax2.set_ylim(0, 105)
                ax2.grid(True, alpha=0.3)
            
            # 3. Monitoring Gaps Priority
            if 'monitoring_gaps' in self.results:
                ax3 = axes[1, 0]
                gaps_data = self.results['monitoring_gaps']
                
                if gaps_data:
                    network_types = [gap['network_type'] for gap in gaps_data]
                    gap_percentages = [gap['gap_percentage'] for gap in gaps_data]
                    priorities = [gap['priority'] for gap in gaps_data]
                    
                    # Color by priority
                    priority_colors = {'high': '#FF4500', 'medium': '#FFA500', 'low': '#32CD32'}
                    colors = [priority_colors.get(p, '#808080') for p in priorities]
                    
                    bars = ax3.bar(network_types, gap_percentages, color=colors, alpha=0.7)
                    ax3.set_xlabel('Network Type')
                    ax3.set_ylabel('Coverage Gap (%)')
                    ax3.set_title('Monitoring Gaps by Network Type')
                    ax3.tick_params(axis='x', rotation=45)
                    
                    # Add priority legend
                    from matplotlib.patches import Patch
                    legend_elements = [Patch(facecolor=priority_colors[p], label=p.title()) 
                                     for p in priority_colors.keys()]
                    ax3.legend(handles=legend_elements, title='Priority')
                    ax3.grid(True, alpha=0.3)
            
            # 4. Observatory Distribution
            ax4 = axes[1, 1]
            if 'comprehensive_report' in self.results:
                # Create a simple pie chart of observatory distribution by region
                # This is a placeholder - would use actual geographic data
                regions = ['North America', 'Europe', 'Asia-Pacific', 'Other']
                counts = [2, 1, 1, 1]  # Based on our demo observatories
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                
                wedges, texts, autotexts = ax4.pie(counts, labels=regions, colors=colors, 
                                                  autopct='%1.1f%%', startangle=90)
                ax4.set_title('Observatory Distribution by Region')
            
            plt.tight_layout()
            
            # Save the plot
            output_dir = Path('monitoring_network_analysis')
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / 'monitoring_network_analysis.png', dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_dir / 'monitoring_network_analysis.png'}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
    
    def save_results(self):
        """Save demonstration results to files"""
        logger.info("Saving demonstration results...")
        
        try:
            output_dir = Path('monitoring_network_analysis')
            output_dir.mkdir(exist_ok=True)
            
            # Save results as JSON
            results_file = output_dir / 'demo_results.json'
            with open(results_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                json_results = json.loads(json.dumps(self.results, default=str))
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
            
            # Save summary report
            summary_file = output_dir / 'summary_report.txt'
            with open(summary_file, 'w') as f:
                f.write("MONITORING NETWORK REGISTRY DEMONSTRATION SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Demonstration completed at: {datetime.utcnow().isoformat()}\n\n")
                
                if 'comprehensive_report' in self.results:
                    report = self.results['comprehensive_report']
                    f.write(f"Total observatories registered: {report['total_observatories']}\n")
                    f.write(f"Countries represented: {report['total_countries']}\n\n")
                
                if 'performance_metrics' in self.results:
                    f.write("PERFORMANCE METRICS:\n")
                    for metric, value in self.results['performance_metrics'].items():
                        f.write(f"  {metric}: {value:.2f}\n")
                    f.write("\n")
                
                if 'monitoring_gaps' in self.results:
                    f.write(f"MONITORING GAPS IDENTIFIED: {len(self.results['monitoring_gaps'])}\n")
                    for gap in self.results['monitoring_gaps']:
                        f.write(f"  {gap['network_type']}: {gap['gap_percentage']:.1f}% gap, "
                               f"{gap['priority']} priority\n")
                    f.write("\n")
                
                f.write("For detailed results, see demo_results.json\n")
            
            logger.info(f"Summary report saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    async def run_complete_demonstration(self):
        """Run the complete monitoring network registry demonstration"""
        logger.info("Starting complete Monitoring Network Registry demonstration...")
        
        try:
            # Initialize system
            await self.initialize_registry()
            
            # Populate data
            await self.populate_global_observatories()
            await self.populate_monitoring_stations()
            await self.setup_data_streams()
            
            # Demonstrate analysis capabilities
            await self.demonstrate_coverage_analysis()
            await self.demonstrate_performance_assessment()
            await self.demonstrate_gap_identification()
            await self.demonstrate_real_time_integration()
            
            # Generate comprehensive report
            await self.generate_comprehensive_report()
            
            # Create visualizations
            self.create_visualizations()
            
            # Save results
            self.save_results()
            
            logger.info("Demonstration completed successfully!")
            
            # Print summary
            print("\n" + "="*60)
            print("MONITORING NETWORK REGISTRY DEMONSTRATION COMPLETE")
            print("="*60)
            
            if 'comprehensive_report' in self.results:
                report = self.results['comprehensive_report']
                print(f"Observatories registered: {report['total_observatories']}")
                print(f"Countries represented: {report['total_countries']}")
            
            if 'performance_metrics' in self.results:
                perf = self.results['performance_metrics']
                print(f"Network uptime: {perf['uptime_percentage']:.1f}%")
                print(f"Data completeness: {perf['data_completeness']:.1f}%")
            
            if 'monitoring_gaps' in self.results:
                print(f"Monitoring gaps identified: {len(self.results['monitoring_gaps'])}")
            
            print("\nResults saved to: monitoring_network_analysis/")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Demonstration failed: {e}")
            raise
        finally:
            # Cleanup
            if self.registry and hasattr(self.registry, 'executor'):
                self.registry.executor.shutdown(wait=True)


async def main():
    """Main function to run the demonstration"""
    demo = MonitoringNetworkDemo()
    await demo.run_complete_demonstration()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())