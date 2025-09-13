"""
USGS Volcano Hazards Program API Connector

Provides authenticated access to USGS Volcano Hazards Program data
with real-time alert processing and volcanic monitoring capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncIterator, List, Optional
import logging
import json

from .base_connector import BaseDataConnector, GeologicalObservation, DataSourceConfig

logger = logging.getLogger(__name__)


class USGSVolcanoConnector(BaseDataConnector):
    """USGS Volcano Hazards Program connector with real-time alert processing"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        
        # USGS Volcano Hazards Program endpoints
        self.endpoints = {
            'volcanoes': '/volcanoes',
            'alerts': '/alerts',
            'activity': '/activity',
            'monitoring': '/monitoring',
            'webcams': '/webcams',
            'reports': '/reports',
            'earthquakes': '/earthquakes',
            'deformation': '/deformation',
            'gas': '/gas',
            'thermal': '/thermal'
        }
        
        # Alert level mappings
        self.alert_levels = {
            'green': 0,
            'yellow': 1, 
            'orange': 2,
            'red': 3
        }
    
    async def _authenticate(self) -> bool:
        """Authenticate with USGS Volcano Hazards Program API"""
        try:
            # Test connection with volcano list endpoint
            url = f"{self.config.base_url}{self.endpoints['volcanoes']}"
            response = await self._make_request('GET', url)
            
            if response and 'volcanoes' in response:
                logger.info("Successfully authenticated with USGS Volcano Hazards Program")
                return True
            else:
                logger.error("Invalid response from USGS API")
                return False
                
        except Exception as e:
            logger.error(f"USGS authentication failed: {e}")
            return False
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for USGS API"""
        headers = {
            'User-Agent': 'Prometheus-AI-Platform/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.config.api_key:
            headers['X-API-Key'] = self.config.api_key
            
        return headers
    
    async def fetch_real_time_data(self, **kwargs) -> AsyncIterator[GeologicalObservation]:
        """Fetch real-time volcanic monitoring data and alerts"""
        try:
            while True:
                try:
                    # Fetch current volcanic alerts
                    alerts_url = f"{self.config.base_url}{self.endpoints['alerts']}"
                    alerts_response = await self._make_request('GET', alerts_url)
                    
                    if alerts_response and 'alerts' in alerts_response:
                        for alert in alerts_response['alerts']:
                            observation = GeologicalObservation(
                                observation_id=f"usgs_alert_{alert['volcano_id']}_{alert['alert_id']}",
                                timestamp=datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00')),
                                latitude=alert['latitude'],
                                longitude=alert['longitude'],
                                elevation=alert.get('elevation'),
                                observation_type="volcanic_alert",
                                value={
                                    "alert_level": alert['alert_level'],
                                    "alert_level_numeric": self.alert_levels.get(alert['alert_level'].lower(), 0),
                                    "aviation_color": alert.get('aviation_color'),
                                    "volcano_name": alert['volcano_name'],
                                    "alert_description": alert.get('description', '')
                                },
                                data_source="USGS Volcano Hazards Program",
                                quality_flags=self._assess_alert_quality(alert),
                                metadata={
                                    "volcano_id": alert['volcano_id'],
                                    "alert_id": alert['alert_id'],
                                    "issuing_agency": alert.get('agency', 'USGS'),
                                    "alert_type": alert.get('type', 'status_update')
                                }
                            )
                            
                            if await self.validate_data_quality(observation):
                                yield observation
                    
                    # Fetch recent volcanic activity
                    activity_url = f"{self.config.base_url}{self.endpoints['activity']}"
                    activity_params = {
                        'starttime': (datetime.now() - timedelta(hours=1)).isoformat(),
                        'endtime': datetime.now().isoformat()
                    }
                    
                    activity_response = await self._make_request('GET', activity_url, params=activity_params)
                    
                    if activity_response and 'activities' in activity_response:
                        for activity in activity_response['activities']:
                            observation = GeologicalObservation(
                                observation_id=f"usgs_activity_{activity['volcano_id']}_{activity['activity_id']}",
                                timestamp=datetime.fromisoformat(activity['timestamp'].replace('Z', '+00:00')),
                                latitude=activity['latitude'],
                                longitude=activity['longitude'],
                                elevation=activity.get('elevation'),
                                observation_type="volcanic_activity",
                                value={
                                    "activity_type": activity['activity_type'],
                                    "intensity": activity.get('intensity'),
                                    "duration_minutes": activity.get('duration_minutes'),
                                    "description": activity.get('description', ''),
                                    "vei": activity.get('vei')  # Volcanic Explosivity Index
                                },
                                data_source="USGS Volcano Hazards Program",
                                quality_flags=self._assess_activity_quality(activity),
                                metadata={
                                    "volcano_id": activity['volcano_id'],
                                    "activity_id": activity['activity_id'],
                                    "observer": activity.get('observer'),
                                    "confirmation_status": activity.get('confirmation_status')
                                }
                            )
                            
                            if await self.validate_data_quality(observation):
                                yield observation
                    
                    # Fetch volcanic earthquake data
                    eq_url = f"{self.config.base_url}{self.endpoints['earthquakes']}"
                    eq_params = {
                        'starttime': (datetime.now() - timedelta(hours=1)).isoformat(),
                        'endtime': datetime.now().isoformat(),
                        'minmagnitude': kwargs.get('min_magnitude', 1.0)
                    }
                    
                    eq_response = await self._make_request('GET', eq_url, params=eq_params)
                    
                    if eq_response and 'earthquakes' in eq_response:
                        for eq in eq_response['earthquakes']:
                            observation = GeologicalObservation(
                                observation_id=f"usgs_volcanic_eq_{eq['event_id']}",
                                timestamp=datetime.fromisoformat(eq['timestamp'].replace('Z', '+00:00')),
                                latitude=eq['latitude'],
                                longitude=eq['longitude'],
                                elevation=eq.get('depth_km', 0) * -1,  # Convert depth to elevation
                                observation_type="volcanic_earthquake",
                                value={
                                    "magnitude": eq['magnitude'],
                                    "magnitude_type": eq.get('magnitude_type', 'ML'),
                                    "depth_km": eq.get('depth_km'),
                                    "distance_from_volcano_km": eq.get('distance_from_volcano_km'),
                                    "associated_volcano": eq.get('associated_volcano')
                                },
                                uncertainty=eq.get('magnitude_uncertainty'),
                                data_source="USGS Volcano Hazards Program",
                                quality_flags=self._assess_earthquake_quality(eq),
                                metadata={
                                    "event_id": eq['event_id'],
                                    "volcano_id": eq.get('volcano_id'),
                                    "network": eq.get('network', 'USGS'),
                                    "location_method": eq.get('location_method')
                                }
                            )
                            
                            if await self.validate_data_quality(observation):
                                yield observation
                    
                    # Fetch gas emission data
                    gas_url = f"{self.config.base_url}{self.endpoints['gas']}"
                    gas_params = {
                        'starttime': (datetime.now() - timedelta(hours=6)).isoformat(),
                        'endtime': datetime.now().isoformat()
                    }
                    
                    gas_response = await self._make_request('GET', gas_url, params=gas_params)
                    
                    if gas_response and 'measurements' in gas_response:
                        for measurement in gas_response['measurements']:
                            observation = GeologicalObservation(
                                observation_id=f"usgs_gas_{measurement['volcano_id']}_{measurement['measurement_id']}",
                                timestamp=datetime.fromisoformat(measurement['timestamp'].replace('Z', '+00:00')),
                                latitude=measurement['latitude'],
                                longitude=measurement['longitude'],
                                elevation=measurement.get('elevation'),
                                observation_type="gas_emission",
                                value={
                                    "so2_tonnes_per_day": measurement.get('so2_tonnes_per_day'),
                                    "co2_tonnes_per_day": measurement.get('co2_tonnes_per_day'),
                                    "h2s_ppm": measurement.get('h2s_ppm'),
                                    "temperature_celsius": measurement.get('temperature_celsius'),
                                    "measurement_method": measurement.get('method')
                                },
                                uncertainty=measurement.get('uncertainty_percent'),
                                data_source="USGS Volcano Hazards Program",
                                quality_flags=self._assess_gas_quality(measurement),
                                metadata={
                                    "volcano_id": measurement['volcano_id'],
                                    "measurement_id": measurement['measurement_id'],
                                    "instrument": measurement.get('instrument'),
                                    "weather_conditions": measurement.get('weather_conditions')
                                }
                            )
                            
                            if await self.validate_data_quality(observation):
                                yield observation
                    
                    # Wait before next fetch
                    await asyncio.sleep(kwargs.get('fetch_interval', 300))  # 5 minutes default
                    
                except Exception as e:
                    logger.error(f"Error in USGS real-time data fetch: {e}")
                    await asyncio.sleep(60)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Fatal error in USGS real-time streaming: {e}")
            raise
    
    async def fetch_historical_data(self, start_time: datetime, end_time: datetime, **kwargs) -> List[GeologicalObservation]:
        """Fetch historical volcanic data"""
        observations = []
        
        try:
            # Fetch historical volcanic activity
            activity_url = f"{self.config.base_url}{self.endpoints['activity']}"
            activity_params = {
                'starttime': start_time.isoformat(),
                'endtime': end_time.isoformat(),
                'limit': kwargs.get('max_records', 10000)
            }
            
            activity_response = await self._make_request('GET', activity_url, params=activity_params)
            
            if activity_response and 'activities' in activity_response:
                for activity in activity_response['activities']:
                    observation = GeologicalObservation(
                        observation_id=f"usgs_historical_activity_{activity['activity_id']}",
                        timestamp=datetime.fromisoformat(activity['timestamp'].replace('Z', '+00:00')),
                        latitude=activity['latitude'],
                        longitude=activity['longitude'],
                        elevation=activity.get('elevation'),
                        observation_type="volcanic_activity",
                        value={
                            "activity_type": activity['activity_type'],
                            "intensity": activity.get('intensity'),
                            "vei": activity.get('vei'),
                            "description": activity.get('description', '')
                        },
                        data_source="USGS Volcano Hazards Program",
                        quality_flags=self._assess_activity_quality(activity),
                        metadata={
                            "volcano_id": activity['volcano_id'],
                            "activity_id": activity['activity_id']
                        }
                    )
                    
                    if await self.validate_data_quality(observation):
                        observations.append(observation)
            
            logger.info(f"Fetched {len(observations)} historical volcanic observations from USGS")
            
        except Exception as e:
            logger.error(f"Error fetching USGS historical data: {e}")
            raise
            
        return observations
    
    def _assess_alert_quality(self, alert: Dict[str, Any]) -> List[str]:
        """Assess quality flags for volcanic alerts"""
        flags = []
        
        if not alert.get('alert_level'):
            flags.append("missing_alert_level")
        
        if not alert.get('description'):
            flags.append("missing_description")
            
        if alert.get('confidence_level', 'high') == 'low':
            flags.append("low_confidence")
            
        return flags
    
    def _assess_activity_quality(self, activity: Dict[str, Any]) -> List[str]:
        """Assess quality flags for volcanic activity"""
        flags = []
        
        if not activity.get('activity_type'):
            flags.append("missing_activity_type")
            
        if activity.get('confirmation_status') == 'unconfirmed':
            flags.append("unconfirmed")
            
        return flags
    
    def _assess_earthquake_quality(self, earthquake: Dict[str, Any]) -> List[str]:
        """Assess quality flags for volcanic earthquakes"""
        flags = []
        
        if earthquake.get('magnitude_uncertainty', 0) > 0.5:
            flags.append("high_magnitude_uncertainty")
            
        if earthquake.get('location_uncertainty_km', 0) > 5:
            flags.append("high_location_uncertainty")
            
        return flags
    
    def _assess_gas_quality(self, measurement: Dict[str, Any]) -> List[str]:
        """Assess quality flags for gas measurements"""
        flags = []
        
        if measurement.get('uncertainty_percent', 0) > 30:
            flags.append("high_uncertainty")
            
        if measurement.get('weather_conditions') == 'poor':
            flags.append("poor_weather_conditions")
            
        return flags
    
    async def validate_data_quality(self, observation: GeologicalObservation) -> bool:
        """Validate volcanic data quality"""
        # Basic validation checks
        if observation.latitude < -90 or observation.latitude > 90:
            return False
        if observation.longitude < -180 or observation.longitude > 180:
            return False
        
        if observation.observation_type == "volcanic_alert":
            alert_level = observation.value.get("alert_level")
            if alert_level not in ['green', 'yellow', 'orange', 'red']:
                return False
        
        elif observation.observation_type == "volcanic_earthquake":
            magnitude = observation.value.get("magnitude")
            if magnitude is None or magnitude < 0 or magnitude > 10:
                return False
        
        elif observation.observation_type == "gas_emission":
            so2 = observation.value.get("so2_tonnes_per_day")
            if so2 is not None and so2 < 0:
                return False
        
        # Check for excessive quality flags
        if len(observation.quality_flags) > 4:
            return False
            
        return True