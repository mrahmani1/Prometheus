"""
ExternalDataConnectors Wrapper Class
Comprehensive wrapper for external data connectors with real-time streaming capabilities
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable, Optional
from .external_data_connectors import (
    USGSConnector, AGUWileyConnector, NASAEarthDataConnector,
    WorldBankDataConnector, OpenDataPortalConnector, 
    DataIntegrationManager, GeologicalRecord, RateLimiter
)
import requests
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class ExternalDataConnectors:
    """
    Comprehensive external data connectors wrapper for geological AI systems
    Provides unified access to multiple geological data sources with real-time streaming capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize all external data connectors with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize rate limiting
        self.rate_limit = config.get('rate_limit', 1.0)
        
        # Initialize all connectors
        self._initialize_connectors()
        
        # Set up data streaming capabilities
        self._initialize_streaming()
        
        # Initialize real-time monitoring
        self._initialize_monitoring()
    
    def _initialize_connectors(self):
        """Initialize all geological data source connectors"""
        try:
            # Primary geological data sources
            self.usgs_connector = USGSConnector(rate_limit=self.rate_limit)
            self.agu_wiley_connector = AGUWileyConnector(
                api_key=self.config.get('wiley_api_key', '')
            )
            self.nasa_earthdata_connector = NASAEarthDataConnector(
                token=self.config.get('nasa_earthdata_token', '')
            )
            
            # Economic and government data
            self.world_bank_connector = WorldBankDataConnector()
            self.open_data_portal_connector = OpenDataPortalConnector()
            
            # Additional specialized connectors
            self.gns_connector = self._initialize_gns_connector()
            self.gvp_connector = self._initialize_gvp_connector()
            self.usgs_volcano_connector = self._initialize_volcano_connector()
            self.emsc_connector = self._initialize_emsc_connector()
            
            # Database and integration manager
            self.integration_manager = DataIntegrationManager()
            
            self.logger.info("Successfully initialized all external data connectors")
            
        except Exception as e:
            self.logger.error(f"Error initializing connectors: {e}")
            raise
    
    def _initialize_gns_connector(self):
        """Initialize GNS Science (New Zealand) connector"""
        class GNSConnector:
            def __init__(self):
                self.base_url = "https://api.gns.cri.nz/geonet"
                self.rate_limiter = RateLimiter(1.0)
            
            def get_earthquake_data(self, **kwargs) -> List[GeologicalRecord]:
                """Fetch earthquake data from GNS GeoNet"""
                self.rate_limiter.wait_if_needed()
                
                try:
                    # Build API query
                    params = {
                        'format': 'json',
                        'limit': kwargs.get('limit', 1000)
                    }
                    
                    if kwargs.get('min_magnitude'):
                        params['minmag'] = kwargs['min_magnitude']
                    if kwargs.get('start_time'):
                        params['startdate'] = kwargs['start_time'].strftime('%Y-%m-%d')
                    if kwargs.get('end_time'):
                        params['enddate'] = kwargs['end_time'].strftime('%Y-%m-%d')
                    
                    url = f"{self.base_url}/earthquake?" + urlencode(params)
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    records = []
                    
                    for event in data.get('features', []):
                        props = event.get('properties', {})
                        coords = event.get('geometry', {}).get('coordinates', [0, 0, 0])
                        
                        record = GeologicalRecord(
                            record_id=f"gns_{props.get('publicID', '')}",
                            source="GNS_GeoNet",
                            data_type="earthquake",
                            coordinates=(coords[1], coords[0], -coords[2]),  # lat, lon, depth
                            timestamp=datetime.fromisoformat(props.get('time', '').replace('Z', '+00:00')),
                            properties={
                                'magnitude': props.get('magnitude'),
                                'depth': props.get('depth'),
                                'location': props.get('locality'),
                                'event_type': props.get('magnitudeType')
                            },
                            quality_score=0.9,  # GNS has high quality data
                            metadata={'network': 'GNS_GeoNet', 'country': 'New Zealand'}
                        )
                        records.append(record)
                    
                    return records
                    
                except Exception as e:
                    logger.error(f"Error fetching GNS earthquake data: {e}")
                    return []
        
        return GNSConnector()
    
    def _initialize_gvp_connector(self):
        """Initialize Global Volcanism Program connector"""
        class GVPConnector:
            def __init__(self):
                self.base_url = "https://volcano.si.edu/api"
                self.rate_limiter = RateLimiter(0.5)  # Be conservative with Smithsonian API
            
            def get_volcano_data(self, **kwargs) -> List[GeologicalRecord]:
                """Fetch volcano data from Smithsonian GVP"""
                self.rate_limiter.wait_if_needed()
                
                try:
                    # Use web scraping approach for GVP data (they don't have a public API)
                    url = "https://volcano.si.edu/database/search_volcano.cfm"
                    
                    params = {
                        'searchvol': '1',
                        'format': 'csv'
                    }
                    
                    # Add geographic filters
                    if kwargs.get('min_latitude'):
                        params['lat_min'] = kwargs['min_latitude']
                    if kwargs.get('max_latitude'):
                        params['lat_max'] = kwargs['max_latitude']
                    if kwargs.get('min_longitude'):
                        params['lon_min'] = kwargs['min_longitude']
                    if kwargs.get('max_longitude'):
                        params['lon_max'] = kwargs['max_longitude']
                    
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    # Parse CSV response
                    from io import StringIO
                    import csv
                    
                    csv_data = StringIO(response.text)
                    reader = csv.DictReader(csv_data)
                    
                    records = []
                    for row in reader:
                        try:
                            lat = float(row.get('Latitude', 0))
                            lon = float(row.get('Longitude', 0))
                            elevation = float(row.get('Elevation (m)', 0))
                            
                            record = GeologicalRecord(
                                record_id=f"gvp_{row.get('Volcano Number', '')}",
                                source="GVP_Smithsonian",
                                data_type="volcano",
                                coordinates=(lat, lon, elevation),
                                timestamp=datetime.now(),  # GVP doesn't provide timestamps for all data
                                properties={
                                    'name': row.get('Volcano Name'),
                                    'country': row.get('Country'),
                                    'region': row.get('Region'),
                                    'type': row.get('Primary Volcano Type'),
                                    'last_eruption': row.get('Last Known Eruption'),
                                    'tectonic_setting': row.get('Tectonic Setting'),
                                    'dominant_rock_type': row.get('Dominant Rock Type')
                                },
                                quality_score=0.95,  # GVP is authoritative for volcano data
                                metadata={'database': 'Global_Volcanism_Program'}
                            )
                            records.append(record)
                            
                        except (ValueError, TypeError) as e:
                            continue  # Skip malformed records
                    
                    return records[:kwargs.get('limit', 1000)]
                    
                except Exception as e:
                    logger.error(f"Error fetching GVP volcano data: {e}")
                    return []
        
        return GVPConnector()
    
    def _initialize_volcano_connector(self):
        """Initialize USGS Volcano Observatory connector"""
        class USGSVolcanoConnector:
            def __init__(self):
                self.base_url = "https://www.usgs.gov/observatories/volcano-observatories"
                self.rate_limiter = RateLimiter(1.0)
            
            def get_volcano_alerts(self, **kwargs) -> List[GeologicalRecord]:
                """Fetch current volcano alerts from USGS observatories"""
                self.rate_limiter.wait_if_needed()
                
                try:
                    # Use USGS VADS (Volcano Alert-Location System) data
                    url = "https://volcanoes.usgs.gov/observatories/alerts.json"
                    
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    records = []
                    
                    for alert in data.get('alerts', []):
                        try:
                            lat = float(alert.get('latitude', 0))
                            lon = float(alert.get('longitude', 0))
                            
                            record = GeologicalRecord(
                                record_id=f"usgs_volcano_{alert.get('id', '')}",
                                source="USGS_Volcano_Observatories",
                                data_type="volcano_alert",
                                coordinates=(lat, lon, alert.get('elevation', 0)),
                                timestamp=datetime.fromisoformat(alert.get('updated', '').replace('Z', '+00:00')),
                                properties={
                                    'name': alert.get('name'),
                                    'alert_level': alert.get('alertLevel'),
                                    'aviation_color': alert.get('aviationColor'),
                                    'activity_summary': alert.get('activitySummary'),
                                    'hazards_summary': alert.get('hazardsSummary'),
                                    'recent_activity': alert.get('recentActivity')
                                },
                                quality_score=0.98,  # USGS observatory data is very high quality
                                metadata={'observatory': alert.get('observatory')}
                            )
                            records.append(record)
                            
                        except (ValueError, TypeError, KeyError):
                            continue
                    
                    return records
                    
                except Exception as e:
                    logger.error(f"Error fetching USGS volcano alerts: {e}")
                    return []
        
        return USGSVolcanoConnector()
    
    def _initialize_emsc_connector(self):
        """Initialize European-Mediterranean Seismological Centre connector"""
        class EMSCConnector:
            def __init__(self):
                self.base_url = "https://www.seismicportal.eu/fdsnws/event/1"
                self.rate_limiter = RateLimiter(0.5)  # Be conservative
            
            def get_earthquake_data(self, **kwargs) -> List[GeologicalRecord]:
                """Fetch earthquake data from EMSC"""
                self.rate_limiter.wait_if_needed()
                
                try:
                    params = {
                        'format': 'json',
                        'limit': kwargs.get('limit', 1000)
                    }
                    
                    # Add filters
                    if kwargs.get('start_time'):
                        params['starttime'] = kwargs['start_time'].strftime('%Y-%m-%dT%H:%M:%S')
                    if kwargs.get('end_time'):
                        params['endtime'] = kwargs['end_time'].strftime('%Y-%m-%dT%H:%M:%S')
                    if kwargs.get('min_magnitude'):
                        params['minmag'] = kwargs['min_magnitude']
                    if kwargs.get('min_latitude'):
                        params['minlat'] = kwargs['min_latitude']
                    if kwargs.get('max_latitude'):
                        params['maxlat'] = kwargs['max_latitude']
                    if kwargs.get('min_longitude'):
                        params['minlon'] = kwargs['min_longitude']
                    if kwargs.get('max_longitude'):
                        params['maxlon'] = kwargs['max_longitude']
                    
                    url = f"{self.base_url}/query"
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    records = []
                    
                    for event in data.get('features', []):
                        props = event.get('properties', {})
                        coords = event.get('geometry', {}).get('coordinates', [0, 0, 0])
                        
                        record = GeologicalRecord(
                            record_id=f"emsc_{props.get('id', '')}",
                            source="EMSC",
                            data_type="earthquake",
                            coordinates=(coords[1], coords[0], -coords[2]),  # lat, lon, depth
                            timestamp=datetime.fromtimestamp(props.get('time', 0) / 1000),
                            properties={
                                'magnitude': props.get('mag'),
                                'depth': props.get('depth'),
                                'place': props.get('place'),
                                'magnitude_type': props.get('magType'),
                                'event_type': props.get('type')
                            },
                            quality_score=0.85,
                            metadata={'network': 'EMSC', 'region': 'Euro-Mediterranean'}
                        )
                        records.append(record)
                    
                    return records
                    
                except Exception as e:
                    logger.error(f"Error fetching EMSC earthquake data: {e}")
                    return []
        
        return EMSCConnector()
    
    def _initialize_streaming(self):
        """Initialize real-time data streaming capabilities"""
        self.streaming_active = False
        self.stream_handlers = {}
        self.websocket_connections = {}
        
        # Set up WebSocket connections for real-time feeds
        self._setup_realtime_feeds()
    
    def _setup_realtime_feeds(self):
        """Set up real-time data feeds from various sources"""
        try:
            # USGS Earthquake real-time feed
            self.realtime_feeds = {
                'usgs_earthquakes': {
                    'url': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson',
                    'update_interval': 300,  # 5 minutes
                    'processor': self._process_usgs_realtime
                },
                'emsc_earthquakes': {
                    'url': 'https://www.seismicportal.eu/fdsnws/event/1/query?format=geojson&limit=100&orderby=time',
                    'update_interval': 600,  # 10 minutes
                    'processor': self._process_emsc_realtime
                },
                'usgs_volcano_alerts': {
                    'url': 'https://volcanoes.usgs.gov/observatories/alerts.json',
                    'update_interval': 1800,  # 30 minutes
                    'processor': self._process_volcano_alerts
                }
            }
            
            self.logger.info("Real-time data feeds configured")
            
        except Exception as e:
            self.logger.error(f"Error setting up real-time feeds: {e}")
    
    def _initialize_monitoring(self):
        """Initialize real-time monitoring and alerting"""
        self.monitoring_thresholds = {
            'earthquake_magnitude': 5.0,
            'volcano_alert_level': 'WATCH',
            'data_quality_minimum': 0.7,
            'update_frequency_minimum': 3600  # 1 hour
        }
        
        self.alert_callbacks = []
        
        # Set up monitoring loop
        self.monitoring_active = False
    
    async def start_realtime_monitoring(self):
        """Start real-time monitoring and data collection"""
        self.monitoring_active = True
        self.streaming_active = True
        
        # Start monitoring tasks
        tasks = [
            self._monitor_earthquake_feeds(),
            self._monitor_volcano_feeds(),
            self._monitor_data_quality(),
            self._update_realtime_feeds()
        ]
        
        await asyncio.gather(*tasks)
    
    async def _monitor_earthquake_feeds(self):
        """Monitor earthquake feeds for significant events"""
        while self.monitoring_active:
            try:
                # Check USGS feed
                recent_earthquakes = await self._fetch_recent_earthquakes()
                
                for earthquake in recent_earthquakes:
                    magnitude = earthquake.properties.get('magnitude', 0)
                    if magnitude >= self.monitoring_thresholds['earthquake_magnitude']:
                        await self._trigger_earthquake_alert(earthquake)
                
                await asyncio.sleep(self.realtime_feeds['usgs_earthquakes']['update_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in earthquake monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _monitor_volcano_feeds(self):
        """Monitor volcano feeds for alert level changes"""
        while self.monitoring_active:
            try:
                # Check volcano alert levels
                volcano_alerts = await self._fetch_volcano_alerts()
                
                for alert in volcano_alerts:
                    alert_level = alert.properties.get('alert_level', '')
                    if alert_level in ['WATCH', 'WARNING']:
                        await self._trigger_volcano_alert(alert)
                
                await asyncio.sleep(self.realtime_feeds['usgs_volcano_alerts']['update_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in volcano monitoring: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _monitor_data_quality(self):
        """Monitor overall data quality and connectivity"""
        while self.monitoring_active:
            try:
                # Test connectivity to all data sources
                connectivity_status = await self._test_all_connections()
                
                # Log any connectivity issues
                for source, status in connectivity_status.items():
                    if not status['connected']:
                        self.logger.warning(f"Connection issue with {source}: {status.get('error', 'Unknown error')}")
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in data quality monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _update_realtime_feeds(self):
        """Update real-time data feeds periodically"""
        while self.streaming_active:
            try:
                for feed_name, feed_config in self.realtime_feeds.items():
                    try:
                        # Fetch latest data
                        async with aiohttp.ClientSession() as session:
                            async with session.get(feed_config['url']) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    
                                    # Process data using feed-specific processor
                                    processed_data = feed_config['processor'](data)
                                    
                                    # Store in integration manager
                                    if processed_data:
                                        await self._store_realtime_data(feed_name, processed_data)
                                    
                                    self.logger.info(f"Updated {feed_name}: {len(processed_data)} records")
                                
                    except Exception as e:
                        self.logger.error(f"Error updating {feed_name}: {e}")
                
                # Wait for next update cycle (use minimum interval)
                min_interval = min(feed['update_interval'] for feed in self.realtime_feeds.values())
                await asyncio.sleep(min_interval)
                
            except Exception as e:
                self.logger.error(f"Error in real-time feed updates: {e}")
                await asyncio.sleep(300)
    
    def _process_usgs_realtime(self, data: Dict) -> List[GeologicalRecord]:
        """Process USGS real-time earthquake data"""
        records = []
        
        for feature in data.get('features', []):
            try:
                props = feature.get('properties', {})
                coords = feature.get('geometry', {}).get('coordinates', [0, 0, 0])
                
                record = GeologicalRecord(
                    record_id=f"usgs_rt_{props.get('id', '')}",
                    source="USGS_Realtime",
                    data_type="earthquake",
                    coordinates=(coords[1], coords[0], -coords[2]),
                    timestamp=datetime.fromtimestamp(props.get('time', 0) / 1000),
                    properties={
                        'magnitude': props.get('mag'),
                        'depth': props.get('depth'),
                        'place': props.get('place'),
                        'magnitude_type': props.get('magType'),
                        'significance': props.get('sig'),
                        'alert': props.get('alert'),
                        'tsunami': props.get('tsunami')
                    },
                    quality_score=0.9,
                    metadata={'realtime': True, 'feed': 'USGS'}
                )
                records.append(record)
                
            except Exception as e:
                self.logger.warning(f"Error processing USGS realtime record: {e}")
                continue
        
        return records
    
    def _process_emsc_realtime(self, data: Dict) -> List[GeologicalRecord]:
        """Process EMSC real-time earthquake data"""
        return self.emsc_connector.get_earthquake_data()  # Use existing EMSC processing
    
    def _process_volcano_alerts(self, data: Dict) -> List[GeologicalRecord]:
        """Process volcano alert data"""
        return self.usgs_volcano_connector.get_volcano_alerts()  # Use existing processing
    
    async def _store_realtime_data(self, feed_name: str, data: List[GeologicalRecord]):
        """Store real-time data in the integration manager"""
        try:
            for record in data:
                self.integration_manager.db_manager.store_record(record)
            
            self.logger.debug(f"Stored {len(data)} records from {feed_name}")
            
        except Exception as e:
            self.logger.error(f"Error storing real-time data: {e}")
    
    async def _fetch_recent_earthquakes(self) -> List[GeologicalRecord]:
        """Fetch recent earthquakes for monitoring"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        return self.usgs_connector.get_earthquake_data(
            start_time=start_time,
            end_time=end_time,
            min_magnitude=4.0,
            limit=100
        )
    
    async def _fetch_volcano_alerts(self) -> List[GeologicalRecord]:
        """Fetch current volcano alerts"""
        return self.usgs_volcano_connector.get_volcano_alerts()
    
    async def _trigger_earthquake_alert(self, earthquake: GeologicalRecord):
        """Trigger alert for significant earthquake"""
        alert_data = {
            'type': 'earthquake',
            'magnitude': earthquake.properties.get('magnitude'),
            'location': earthquake.properties.get('place'),
            'coordinates': earthquake.coordinates,
            'timestamp': earthquake.timestamp,
            'source': earthquake.source
        }
        
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    async def _trigger_volcano_alert(self, volcano_alert: GeologicalRecord):
        """Trigger alert for volcano status change"""
        alert_data = {
            'type': 'volcano',
            'name': volcano_alert.properties.get('name'),
            'alert_level': volcano_alert.properties.get('alert_level'),
            'coordinates': volcano_alert.coordinates,
            'timestamp': volcano_alert.timestamp,
            'activity': volcano_alert.properties.get('activity_summary')
        }
        
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                self.logger.error(f"Error in volcano alert callback: {e}")
    
    async def _test_all_connections(self) -> Dict[str, Dict]:
        """Test connectivity to all data sources"""
        connectivity_status = {}
        
        # Test each connector
        connectors_to_test = [
            ('USGS', self.usgs_connector),
            ('GNS', self.gns_connector),
            ('GVP', self.gvp_connector),
            ('EMSC', self.emsc_connector),
            ('NASA_EarthData', self.nasa_earthdata_connector)
        ]
        
        for name, connector in connectors_to_test:
            try:
                # Attempt a simple data fetch
                if hasattr(connector, 'get_earthquake_data'):
                    test_data = connector.get_earthquake_data(limit=1)
                elif hasattr(connector, 'get_volcano_data'):
                    test_data = connector.get_volcano_data(limit=1)
                else:
                    test_data = []
                
                connectivity_status[name] = {
                    'connected': True,
                    'records_fetched': len(test_data),
                    'last_test': datetime.now()
                }
                
            except Exception as e:
                connectivity_status[name] = {
                    'connected': False,
                    'error': str(e),
                    'last_test': datetime.now()
                }
        
        return connectivity_status
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for real-time alerts"""
        self.alert_callbacks.append(callback)
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        self.streaming_active = False
    
    def get_all_available_data(self, **kwargs) -> Dict[str, List[GeologicalRecord]]:
        """Get data from all available connectors"""
        all_data = {}
        
        # Earthquake data from multiple sources
        all_data['usgs_earthquakes'] = self.usgs_connector.get_earthquake_data(**kwargs)
        all_data['gns_earthquakes'] = self.gns_connector.get_earthquake_data(**kwargs)
        all_data['emsc_earthquakes'] = self.emsc_connector.get_earthquake_data(**kwargs)
        
        # Volcano data
        all_data['volcano_alerts'] = self.usgs_volcano_connector.get_volcano_alerts(**kwargs)
        all_data['global_volcanoes'] = self.gvp_connector.get_volcano_data(**kwargs)
        
        # Research and satellite data
        if 'research_keywords' in kwargs:
            all_data['research_papers'] = self.agu_wiley_connector.search_papers(
                keywords=kwargs['research_keywords'],
                limit=kwargs.get('limit', 100)
            )
        
        if 'satellite_collection' in kwargs:
            all_data['satellite_imagery'] = self.nasa_earthdata_connector.search_datasets(
                collection=kwargs['satellite_collection'],
                bbox=kwargs.get('bbox'),
                time_range=kwargs.get('time_range')
            )
        
        return all_data