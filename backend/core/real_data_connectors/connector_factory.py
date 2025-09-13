"""
Connector Factory

Factory class for creating and managing real data connectors with
configuration management and health monitoring integration.
"""

import asyncio
from typing import Dict, Any, Optional, List, Type
import logging
from pathlib import Path
import yaml
import json

from .base_connector import BaseDataConnector, DataSourceConfig, DataSourceType
from .iris_connector import IRISConnector
from .usgs_connector import USGSVolcanoConnector
from .nasa_connector import NASAEarthDataConnector
from .copernicus_connector import CopernicusConnector
from .asf_connector import ASFHyp3Connector
from .geochem_connector import GeochemConnector
from .geological_survey_connector import GeologicalSurveyConnector
from .health_monitor import DataSourceHealthMonitor, FailoverConfig

logger = logging.getLogger(__name__)


class ConnectorFactory:
    """Factory for creating and managing real data connectors"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "backend/config/data_sources.yaml"
        self.connectors: Dict[str, BaseDataConnector] = {}
        self.health_monitor = DataSourceHealthMonitor()
        
        # Connector type mapping
        self.connector_types: Dict[str, Type[BaseDataConnector]] = {
            'iris': IRISConnector,
            'usgs_volcano': USGSVolcanoConnector,
            'nasa_earthdata': NASAEarthDataConnector,
            'copernicus': CopernicusConnector,
            'asf_hyp3': ASFHyp3Connector,
            'geochem': GeochemConnector,
            'geological_survey': GeologicalSurveyConnector
        }
        
        # Default configurations
        self.default_configs = {
            'iris': {
                'name': 'IRIS/EarthScope Seismic Data',
                'base_url': 'http://service.iris.edu',
                'timeout': 30,
                'max_retries': 3,
                'rate_limit': 100,
                'data_type': DataSourceType.SEISMIC
            },
            'usgs_volcano': {
                'name': 'USGS Volcano Hazards Program',
                'base_url': 'https://api.usgs.gov/volcano',
                'timeout': 30,
                'max_retries': 3,
                'rate_limit': 60,
                'data_type': DataSourceType.VOLCANIC
            },
            'nasa_earthdata': {
                'name': 'NASA EarthData',
                'base_url': 'https://api.earthdata.nasa.gov',
                'timeout': 60,
                'max_retries': 3,
                'rate_limit': 50,
                'data_type': DataSourceType.SATELLITE
            },
            'copernicus': {
                'name': 'Copernicus Data Space Ecosystem',
                'base_url': 'https://catalogue.dataspace.copernicus.eu',
                'timeout': 60,
                'max_retries': 3,
                'rate_limit': 40,
                'data_type': DataSourceType.SATELLITE
            },
            'asf_hyp3': {
                'name': 'ASF-Hyp3 InSAR Processing',
                'base_url': 'https://hyp3-api.asf.alaska.edu',
                'timeout': 120,
                'max_retries': 3,
                'rate_limit': 30,
                'data_type': DataSourceType.INSAR
            },
            'geochem': {
                'name': 'Geochemical Databases',
                'base_url': 'https://api.earthchem.org',
                'timeout': 30,
                'max_retries': 3,
                'rate_limit': 60,
                'data_type': DataSourceType.GEOCHEMICAL
            },
            'geological_survey': {
                'name': 'National Geological Surveys',
                'base_url': 'https://api.usgs.gov',
                'timeout': 30,
                'max_retries': 3,
                'rate_limit': 60,
                'data_type': DataSourceType.GEOLOGICAL_SURVEY
            }
        }
    
    async def initialize(self):
        """Initialize the connector factory and load configurations"""
        try:
            # Load configuration file
            config = await self._load_config()
            
            # Create connectors from configuration
            for source_id, source_config in config.get('data_sources', {}).items():
                await self._create_connector(source_id, source_config)
            
            # Setup health monitoring
            await self._setup_health_monitoring(config.get('health_monitoring', {}))
            
            # Setup failover configurations
            await self._setup_failover_configs(config.get('failover', {}))
            
            logger.info(f"Initialized {len(self.connectors)} data source connectors")
            
        except Exception as e:
            logger.error(f"Failed to initialize connector factory: {e}")
            raise
    
    async def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    if config_file.suffix.lower() == '.yaml':
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                
                logger.info(f"Loaded configuration from {config_file}")
                return config
                
            except Exception as e:
                logger.error(f"Failed to load config file {config_file}: {e}")
                
        # Create default configuration
        config = await self._create_default_config()
        await self._save_config(config)
        return config
    
    async def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        config = {
            'data_sources': {},
            'health_monitoring': {
                'check_interval': 60,
                'alert_thresholds': {
                    'response_time_ms': 5000,
                    'success_rate': 0.95,
                    'consecutive_failures': 3
                }
            },
            'failover': {}
        }
        
        # Add default data source configurations
        for connector_type, default_config in self.default_configs.items():
            source_id = f"{connector_type}_primary"
            config['data_sources'][source_id] = {
                'type': connector_type,
                'enabled': True,
                **default_config
            }
            
            # Add failover configuration
            config['failover'][connector_type] = {
                'primary_source': source_id,
                'fallback_sources': [],
                'max_failures': 3,
                'failure_window_minutes': 15,
                'auto_failback': True
            }
        
        return config
    
    async def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                if config_file.suffix.lower() == '.yaml':
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    json.dump(config, f, indent=2)
            
            logger.info(f"Saved configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")
    
    async def _create_connector(self, source_id: str, source_config: Dict[str, Any]):
        """Create a data source connector"""
        try:
            if not source_config.get('enabled', True):
                logger.info(f"Skipping disabled data source: {source_id}")
                return
            
            connector_type = source_config['type']
            if connector_type not in self.connector_types:
                logger.error(f"Unknown connector type: {connector_type}")
                return
            
            # Create data source configuration
            config = DataSourceConfig(
                source_id=source_id,
                name=source_config['name'],
                base_url=source_config['base_url'],
                api_key=source_config.get('api_key'),
                username=source_config.get('username'),
                password=source_config.get('password'),
                timeout=source_config.get('timeout', 30),
                max_retries=source_config.get('max_retries', 3),
                rate_limit=source_config.get('rate_limit', 100),
                data_type=DataSourceType(source_config.get('data_type', 'seismic'))
            )
            
            # Create connector instance
            connector_class = self.connector_types[connector_type]
            connector = connector_class(config)
            
            # Test connection
            connection_status = await connector.connect()
            if connection_status.name == 'CONNECTED':
                self.connectors[source_id] = connector
                logger.info(f"Created connector for {source_id}: {config.name}")
            else:
                logger.error(f"Failed to connect to {source_id}: {connection_status}")
                await connector.disconnect()
                
        except Exception as e:
            logger.error(f"Failed to create connector {source_id}: {e}")
    
    async def _setup_health_monitoring(self, health_config: Dict[str, Any]):
        """Setup health monitoring for all connectors"""
        try:
            # Configure health monitor
            if 'check_interval' in health_config:
                self.health_monitor.check_interval = health_config['check_interval']
            
            if 'alert_thresholds' in health_config:
                self.health_monitor.alert_thresholds.update(health_config['alert_thresholds'])
            
            # Register all connectors for monitoring
            for source_id, connector in self.connectors.items():
                self.health_monitor.register_connector(source_id, connector)
            
            # Add default alert callback
            self.health_monitor.add_alert_callback(self._default_alert_handler)
            
            logger.info("Health monitoring setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup health monitoring: {e}")
    
    async def _setup_failover_configs(self, failover_config: Dict[str, Any]):
        """Setup failover configurations"""
        try:
            for service_name, config in failover_config.items():
                failover_cfg = FailoverConfig(
                    primary_source=config['primary_source'],
                    fallback_sources=config.get('fallback_sources', []),
                    max_failures=config.get('max_failures', 3),
                    failure_window_minutes=config.get('failure_window_minutes', 15),
                    recovery_check_interval=config.get('recovery_check_interval', 300),
                    auto_failback=config.get('auto_failback', True)
                )
                
                # Register failover config with health monitor
                primary_source = config['primary_source']
                if primary_source in self.connectors:
                    self.health_monitor.failover_configs[primary_source] = failover_cfg
                    self.health_monitor.active_sources[service_name] = primary_source
            
            logger.info("Failover configurations setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup failover configs: {e}")
    
    async def _default_alert_handler(self, source_id: str, status, alert_data: Dict[str, Any]):
        """Default alert handler for health monitoring"""
        logger.warning(f"Health Alert - {source_id}: {alert_data}")
        
        # Here you could integrate with external alerting systems
        # like Slack, PagerDuty, email notifications, etc.
    
    def get_connector(self, source_id: str) -> Optional[BaseDataConnector]:
        """Get a specific connector by ID"""
        return self.connectors.get(source_id)
    
    def get_active_connector(self, service_name: str) -> Optional[BaseDataConnector]:
        """Get the currently active connector for a service"""
        return self.health_monitor.get_active_connector(service_name)
    
    def list_connectors(self) -> List[str]:
        """List all available connector IDs"""
        return list(self.connectors.keys())
    
    def list_connector_types(self) -> List[str]:
        """List all available connector types"""
        return list(self.connector_types.keys())
    
    async def add_connector(self, source_id: str, connector_type: str, config: Dict[str, Any]) -> bool:
        """Dynamically add a new connector"""
        try:
            if connector_type not in self.connector_types:
                raise ValueError(f"Unknown connector type: {connector_type}")
            
            # Create connector configuration
            source_config = {
                'type': connector_type,
                'enabled': True,
                **config
            }
            
            # Create and register connector
            await self._create_connector(source_id, source_config)
            
            if source_id in self.connectors:
                # Register with health monitor
                self.health_monitor.register_connector(source_id, self.connectors[source_id])
                
                # Update configuration file
                current_config = await self._load_config()
                current_config['data_sources'][source_id] = source_config
                await self._save_config(current_config)
                
                logger.info(f"Successfully added connector: {source_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to add connector {source_id}: {e}")
            return False
    
    async def remove_connector(self, source_id: str) -> bool:
        """Remove a connector"""
        try:
            if source_id in self.connectors:
                # Disconnect and remove connector
                await self.connectors[source_id].disconnect()
                del self.connectors[source_id]
                
                # Remove from health monitoring
                if source_id in self.health_monitor.connectors:
                    del self.health_monitor.connectors[source_id]
                    if source_id in self.health_monitor.monitoring_tasks:
                        self.health_monitor.monitoring_tasks[source_id].cancel()
                        del self.health_monitor.monitoring_tasks[source_id]
                
                # Update configuration
                current_config = await self._load_config()
                if source_id in current_config.get('data_sources', {}):
                    del current_config['data_sources'][source_id]
                    await self._save_config(current_config)
                
                logger.info(f"Successfully removed connector: {source_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove connector {source_id}: {e}")
            return False
    
    async def test_connector(self, source_id: str) -> Dict[str, Any]:
        """Test a specific connector"""
        if source_id not in self.connectors:
            return {'status': 'error', 'message': f'Connector {source_id} not found'}
        
        try:
            connector = self.connectors[source_id]
            health_result = await connector.health_check()
            
            return {
                'status': 'success',
                'source_id': source_id,
                'health_check': health_result,
                'connection_status': connector.status.value
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'source_id': source_id,
                'message': str(e)
            }
    
    async def test_all_connectors(self) -> Dict[str, Dict[str, Any]]:
        """Test all connectors"""
        results = {}
        
        for source_id in self.connectors:
            results[source_id] = await self.test_connector(source_id)
        
        return results
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        return self.health_monitor.export_health_report()
    
    async def shutdown(self):
        """Shutdown all connectors and monitoring"""
        try:
            # Shutdown health monitoring
            await self.health_monitor.shutdown()
            
            # Disconnect all connectors
            for connector in self.connectors.values():
                await connector.disconnect()
            
            self.connectors.clear()
            logger.info("Connector factory shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def reload_config(self):
        """Reload configuration and restart connectors"""
        try:
            # Shutdown existing connectors
            await self.shutdown()
            
            # Reinitialize with new configuration
            await self.initialize()
            
            logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            raise