"""
NASA EarthData Connector

Provides authenticated access to NASA EarthData for satellite observations
and remote sensing data with automated processing capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncIterator, List, Optional
import logging
import base64
import json

from .base_connector import BaseDataConnector, GeologicalObservation, DataSourceConfig

logger = logging.getLogger(__name__)


class NASAEarthDataConnector(BaseDataConnector):
    """NASA EarthData connector for satellite observations and remote sensing"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.access_token: Optional[str] = None
        
        # NASA EarthData endpoints
        self.endpoints = {
            'token': '/oauth/token',
            'search': '/search/granules',
            'modis': '/api/v1/modis',
            'viirs': '/api/v1/viirs',
            'landsat': '/api/v1/landsat',
            'aster': '/api/v1/aster',
            'thermal': '/api/v1/thermal',
            'fire': '/api/v1/fire',
            'vegetation': '/api/v1/vegetation',
            'elevation': '/api/v1/elevation'
        }
        
        # Satellite instruments and their capabilities
        self.instruments = {
            'MODIS': ['thermal_anomalies', 'vegetation_indices', 'land_surface_temperature'],
            'VIIRS': ['fire_detection', 'nighttime_lights', 'vegetation_health'],
            'LANDSAT': ['multispectral_imagery', 'thermal_infrared', 'surface_reflectance'],
            'ASTER': ['thermal_infrared', 'digital_elevation', 'mineral_mapping'],
            'SRTM': ['digital_elevation_model', 'topography']
        }
    
    async def _authenticate(self) -> bool:
        """Authenticate with NASA EarthData using OAuth2"""
        try:
            # Prepare OAuth2 credentials
            if not self.config.username or not self.config.password:
                logger.error("NASA EarthData requires username and password")
                return False
            
            # Create basic auth header
            credentials = f"{self.config.username}:{self.config.password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            # Request access token
            token_url = f"{self.config.base_url}{self.endpoints['token']}"
            headers = {
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {
                'grant_type': 'client_credentials',
                'scope': 'read'
            }
            
            response = await self._make_request('POST', token_url, headers=headers, data=data)
            
            if response and 'access_token' in response:
                self.access_token = response['access_token']
                logger.info("Successfully authenticated with NASA EarthData")
                return True
            else:
                logger.error("Failed to obtain NASA EarthData access token")
                return False
                
        except Exception as e:
            logger.error(f"NASA EarthData authentication failed: {e}")
            return False
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for NASA EarthData API"""
        headers = {
            'User-Agent': 'Prometheus-AI-Platform/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
        elif self.config.api_key:
            headers['X-API-Key'] = self.config.api_key
            
        return headers
    
    async def fetch_real_time_data(self, **kwargs) -> AsyncIterator[GeologicalObservation]:
        """Fetch real-time satellite observations"""
        try:
            # Parameters for satellite data
            instruments = kwargs.get('instruments', ['MODIS', 'VIIRS'])
            bbox = kwargs.get('bbox', [-180, -90, 180, 90])  # Global by default
            
            while True:
                try:
                    # Fetch MODIS thermal anomalies (fires/volcanic activity)
                    if 'MODIS' in instruments:
                        async for obs in self._fetch_modis_thermal_anomalies(bbox):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Fetch VIIRS fire detection
                    if 'VIIRS' in instruments:
                        async for obs in self._fetch_viirs_fire_detection(bbox):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Fetch Landsat surface temperature
                    if 'LANDSAT' in instruments:
                        async for obs in self._fetch_landsat_thermal(bbox):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Wait before next fetch
                    await asyncio.sleep(kwargs.get('fetch_interval', 3600))  # 1 hour default
                    
                except Exception as e:
                    logger.error(f"Error in NASA EarthData real-time fetch: {e}")
                    await asyncio.sleep(300)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Fatal error in NASA EarthData streaming: {e}")
            raise
    
    async def _fetch_modis_thermal_anomalies(self, bbox: List[float]) -> AsyncIterator[GeologicalObservation]:
        """Fetch MODIS thermal anomaly data"""
        try:
            modis_url = f"{self.config.base_url}{self.endpoints['modis']}/thermal_anomalies"
            params = {
                'bbox': ','.join(map(str, bbox)),
                'time_range': 'recent_24h',
                'confidence': 'high'
            }
            
            response = await self._make_request('GET', modis_url, params=params)
            
            if response and 'features' in response:
                for feature in response['features']:
                    properties = feature['properties']
                    geometry = feature['geometry']
                    
                    observation = GeologicalObservation(
                        observation_id=f"nasa_modis_thermal_{properties['id']}",
                        timestamp=datetime.fromisoformat(properties['acquisition_time'].replace('Z', '+00:00')),
                        latitude=geometry['coordinates'][1],
                        longitude=geometry['coordinates'][0],
                        observation_type="thermal_anomaly",
                        value={
                            "brightness_temperature_k": properties['brightness_temp_k'],
                            "fire_radiative_power_mw": properties.get('frp_mw'),
                            "confidence_percent": properties['confidence'],
                            "satellite": properties['satellite'],
                            "instrument": "MODIS"
                        },
                        uncertainty=properties.get('uncertainty_k'),
                        data_source="NASA EarthData MODIS",
                        quality_flags=self._assess_modis_quality(properties),
                        metadata={
                            "granule_id": properties.get('granule_id'),
                            "orbit_number": properties.get('orbit'),
                            "processing_level": properties.get('processing_level', 'L2'),
                            "cloud_cover_percent": properties.get('cloud_cover')
                        }
                    )
                    
                    yield observation
                    
        except Exception as e:
            logger.error(f"Error fetching MODIS thermal anomalies: {e}")
    
    async def _fetch_viirs_fire_detection(self, bbox: List[float]) -> AsyncIterator[GeologicalObservation]:
        """Fetch VIIRS fire detection data"""
        try:
            viirs_url = f"{self.config.base_url}{self.endpoints['viirs']}/fire_detection"
            params = {
                'bbox': ','.join(map(str, bbox)),
                'time_range': 'recent_12h',
                'min_confidence': 75
            }
            
            response = await self._make_request('GET', viirs_url, params=params)
            
            if response and 'detections' in response:
                for detection in response['detections']:
                    observation = GeologicalObservation(
                        observation_id=f"nasa_viirs_fire_{detection['detection_id']}",
                        timestamp=datetime.fromisoformat(detection['acquisition_time'].replace('Z', '+00:00')),
                        latitude=detection['latitude'],
                        longitude=detection['longitude'],
                        observation_type="fire_detection",
                        value={
                            "brightness_temperature_k": detection['brightness_temp'],
                            "fire_radiative_power_mw": detection['frp'],
                            "confidence_percent": detection['confidence'],
                            "satellite": detection['satellite'],
                            "instrument": "VIIRS"
                        },
                        uncertainty=detection.get('uncertainty'),
                        data_source="NASA EarthData VIIRS",
                        quality_flags=self._assess_viirs_quality(detection),
                        metadata={
                            "scan_angle": detection.get('scan_angle'),
                            "track": detection.get('track'),
                            "processing_time": detection.get('processing_time')
                        }
                    )
                    
                    yield observation
                    
        except Exception as e:
            logger.error(f"Error fetching VIIRS fire detection: {e}")
    
    async def _fetch_landsat_thermal(self, bbox: List[float]) -> AsyncIterator[GeologicalObservation]:
        """Fetch Landsat thermal infrared data"""
        try:
            landsat_url = f"{self.config.base_url}{self.endpoints['landsat']}/thermal"
            params = {
                'bbox': ','.join(map(str, bbox)),
                'time_range': 'recent_week',
                'cloud_cover_max': 30
            }
            
            response = await self._make_request('GET', landsat_url, params=params)
            
            if response and 'scenes' in response:
                for scene in response['scenes']:
                    # Process thermal band statistics
                    thermal_stats = scene.get('thermal_statistics', {})
                    
                    observation = GeologicalObservation(
                        observation_id=f"nasa_landsat_thermal_{scene['scene_id']}",
                        timestamp=datetime.fromisoformat(scene['acquisition_date'].replace('Z', '+00:00')),
                        latitude=scene['center_lat'],
                        longitude=scene['center_lon'],
                        observation_type="thermal_infrared",
                        value={
                            "mean_temperature_k": thermal_stats.get('mean_temp_k'),
                            "max_temperature_k": thermal_stats.get('max_temp_k'),
                            "min_temperature_k": thermal_stats.get('min_temp_k'),
                            "std_temperature_k": thermal_stats.get('std_temp_k'),
                            "satellite": scene['satellite'],
                            "instrument": "Landsat_TIRS"
                        },
                        data_source="NASA EarthData Landsat",
                        quality_flags=self._assess_landsat_quality(scene),
                        metadata={
                            "scene_id": scene['scene_id'],
                            "path": scene['path'],
                            "row": scene['row'],
                            "cloud_cover_percent": scene['cloud_cover'],
                            "processing_level": scene.get('processing_level', 'L1')
                        }
                    )
                    
                    yield observation
                    
        except Exception as e:
            logger.error(f"Error fetching Landsat thermal data: {e}")
    
    async def fetch_historical_data(self, start_time: datetime, end_time: datetime, **kwargs) -> List[GeologicalObservation]:
        """Fetch historical satellite data"""
        observations = []
        
        try:
            # Search for historical granules
            search_url = f"{self.config.base_url}{self.endpoints['search']}"
            params = {
                'temporal': f"{start_time.isoformat()},{end_time.isoformat()}",
                'bounding_box': kwargs.get('bbox', [-180, -90, 180, 90]),
                'short_name': kwargs.get('dataset', 'MOD14A1'),  # MODIS fire product
                'page_size': kwargs.get('page_size', 100)
            }
            
            response = await self._make_request('GET', search_url, params=params)
            
            if response and 'feed' in response and 'entry' in response['feed']:
                for entry in response['feed']['entry']:
                    # Extract metadata from granule
                    granule_id = entry['id']
                    time_start = datetime.fromisoformat(entry['time_start'].replace('Z', '+00:00'))
                    
                    # Get bounding box
                    bbox = entry.get('boxes', [''])[0].split() if entry.get('boxes') else []
                    if len(bbox) >= 4:
                        lat = (float(bbox[1]) + float(bbox[3])) / 2
                        lon = (float(bbox[0]) + float(bbox[2])) / 2
                    else:
                        continue
                    
                    observation = GeologicalObservation(
                        observation_id=f"nasa_historical_{granule_id}",
                        timestamp=time_start,
                        latitude=lat,
                        longitude=lon,
                        observation_type="satellite_granule",
                        value={
                            "dataset": params['short_name'],
                            "granule_size_mb": entry.get('granule_size', 0),
                            "processing_level": entry.get('processing_level', 'L2')
                        },
                        data_source="NASA EarthData",
                        metadata={
                            "granule_id": granule_id,
                            "producer_granule_id": entry.get('producer_granule_id'),
                            "collection_concept_id": entry.get('collection_concept_id'),
                            "data_center": entry.get('data_center')
                        }
                    )
                    
                    if await self.validate_data_quality(observation):
                        observations.append(observation)
            
            logger.info(f"Fetched {len(observations)} historical satellite observations from NASA EarthData")
            
        except Exception as e:
            logger.error(f"Error fetching NASA EarthData historical data: {e}")
            raise
            
        return observations
    
    def _assess_modis_quality(self, properties: Dict[str, Any]) -> List[str]:
        """Assess quality flags for MODIS data"""
        flags = []
        
        confidence = properties.get('confidence', 0)
        if confidence < 75:
            flags.append("low_confidence")
        
        cloud_cover = properties.get('cloud_cover', 0)
        if cloud_cover > 50:
            flags.append("high_cloud_cover")
        
        if properties.get('processing_level') != 'L2':
            flags.append("non_standard_processing")
            
        return flags
    
    def _assess_viirs_quality(self, detection: Dict[str, Any]) -> List[str]:
        """Assess quality flags for VIIRS data"""
        flags = []
        
        confidence = detection.get('confidence', 0)
        if confidence < 80:
            flags.append("low_confidence")
        
        scan_angle = detection.get('scan_angle', 0)
        if abs(scan_angle) > 45:
            flags.append("high_scan_angle")
            
        return flags
    
    def _assess_landsat_quality(self, scene: Dict[str, Any]) -> List[str]:
        """Assess quality flags for Landsat data"""
        flags = []
        
        cloud_cover = scene.get('cloud_cover', 0)
        if cloud_cover > 30:
            flags.append("high_cloud_cover")
        
        if scene.get('processing_level') != 'L1':
            flags.append("non_standard_processing")
            
        return flags
    
    async def validate_data_quality(self, observation: GeologicalObservation) -> bool:
        """Validate satellite data quality"""
        # Basic validation checks
        if observation.latitude < -90 or observation.latitude > 90:
            return False
        if observation.longitude < -180 or observation.longitude > 180:
            return False
        
        if observation.observation_type in ["thermal_anomaly", "fire_detection"]:
            temp = observation.value.get("brightness_temperature_k")
            if temp is not None and (temp < 200 or temp > 400):  # Reasonable temperature range
                return False
        
        # Check for excessive quality flags
        if len(observation.quality_flags) > 3:
            return False
            
        return True