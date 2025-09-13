"""
Copernicus Satellite Data Connector

Provides authenticated access to Copernicus satellite data with automated
processing pipelines for Sentinel missions and other European satellite data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncIterator, List, Optional
import logging
import json

from .base_connector import BaseDataConnector, GeologicalObservation, DataSourceConfig

logger = logging.getLogger(__name__)


class CopernicusConnector(BaseDataConnector):
    """Copernicus satellite data connector with automated processing pipelines"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.access_token: Optional[str] = None
        
        # Copernicus services endpoints
        self.endpoints = {
            'auth': '/oauth/token',
            'search': '/search',
            'download': '/odata/v1/Products',
            'sentinel1': '/sentinel-1',
            'sentinel2': '/sentinel-2', 
            'sentinel3': '/sentinel-3',
            'sentinel5p': '/sentinel-5p',
            'processing': '/processing-api/v1'
        }
        
        # Sentinel mission capabilities
        self.missions = {
            'Sentinel-1': ['sar_interferometry', 'ground_motion', 'deformation'],
            'Sentinel-2': ['multispectral_imagery', 'vegetation_monitoring', 'land_cover'],
            'Sentinel-3': ['ocean_land_monitoring', 'thermal_infrared', 'altimetry'],
            'Sentinel-5P': ['atmospheric_monitoring', 'trace_gases', 'air_quality']
        }
    
    async def _authenticate(self) -> bool:
        """Authenticate with Copernicus Data Space Ecosystem"""
        try:
            if not self.config.username or not self.config.password:
                logger.error("Copernicus requires username and password")
                return False
            
            # Request access token
            auth_url = f"{self.config.base_url}{self.endpoints['auth']}"
            data = {
                'grant_type': 'password',
                'username': self.config.username,
                'password': self.config.password,
                'client_id': 'cdse-public'
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = await self._make_request('POST', auth_url, data=data, headers=headers)
            
            if response and 'access_token' in response:
                self.access_token = response['access_token']
                logger.info("Successfully authenticated with Copernicus")
                return True
            else:
                logger.error("Failed to obtain Copernicus access token")
                return False
                
        except Exception as e:
            logger.error(f"Copernicus authentication failed: {e}")
            return False
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Copernicus API"""
        headers = {
            'User-Agent': 'Prometheus-AI-Platform/1.0',
            'Accept': 'application/json'
        }
        
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
            
        return headers
    
    async def fetch_real_time_data(self, **kwargs) -> AsyncIterator[GeologicalObservation]:
        """Fetch real-time Copernicus satellite data"""
        try:
            # Parameters for satellite data
            missions = kwargs.get('missions', ['Sentinel-1', 'Sentinel-2', 'Sentinel-3'])
            bbox = kwargs.get('bbox', [-180, -90, 180, 90])
            
            while True:
                try:
                    # Fetch Sentinel-1 SAR data for deformation monitoring
                    if 'Sentinel-1' in missions:
                        async for obs in self._fetch_sentinel1_data(bbox):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Fetch Sentinel-2 optical data
                    if 'Sentinel-2' in missions:
                        async for obs in self._fetch_sentinel2_data(bbox):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Fetch Sentinel-3 thermal data
                    if 'Sentinel-3' in missions:
                        async for obs in self._fetch_sentinel3_data(bbox):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Fetch Sentinel-5P atmospheric data
                    if 'Sentinel-5P' in missions:
                        async for obs in self._fetch_sentinel5p_data(bbox):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Wait before next fetch
                    await asyncio.sleep(kwargs.get('fetch_interval', 3600))  # 1 hour default
                    
                except Exception as e:
                    logger.error(f"Error in Copernicus real-time fetch: {e}")
                    await asyncio.sleep(300)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Fatal error in Copernicus streaming: {e}")
            raise
    
    async def _fetch_sentinel1_data(self, bbox: List[float]) -> AsyncIterator[GeologicalObservation]:
        """Fetch Sentinel-1 SAR data for deformation monitoring"""
        try:
            search_url = f"{self.config.base_url}{self.endpoints['search']}"
            
            # Search for recent Sentinel-1 products
            params = {
                'platformname': 'Sentinel-1',
                'producttype': 'SLC',  # Single Look Complex for interferometry
                'sensoroperationalmode': 'IW',  # Interferometric Wide swath
                'beginposition': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'endposition': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'footprint': f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))"
            }
            
            response = await self._make_request('GET', search_url, params=params)
            
            if response and 'feed' in response and 'entry' in response['feed']:
                for entry in response['feed']['entry']:
                    if isinstance(entry, dict):
                        # Extract product metadata
                        product_id = entry['id']
                        sensing_time = datetime.fromisoformat(entry['date'][0]['content'].replace('Z', '+00:00'))
                        
                        # Get footprint coordinates
                        footprint = entry.get('str', [])
                        coordinates = self._parse_footprint(footprint)
                        
                        if coordinates:
                            center_lat = sum(coord[1] for coord in coordinates) / len(coordinates)
                            center_lon = sum(coord[0] for coord in coordinates) / len(coordinates)
                            
                            observation = GeologicalObservation(
                                observation_id=f"copernicus_s1_{product_id}",
                                timestamp=sensing_time,
                                latitude=center_lat,
                                longitude=center_lon,
                                observation_type="sar_acquisition",
                                value={
                                    "mission": "Sentinel-1",
                                    "product_type": "SLC",
                                    "sensor_mode": "IW",
                                    "polarization": entry.get('polarisationmode', 'VV VH'),
                                    "orbit_direction": entry.get('orbitdirection', 'ASCENDING'),
                                    "relative_orbit": entry.get('relativeorbitnumber')
                                },
                                data_source="Copernicus Sentinel-1",
                                quality_flags=self._assess_sentinel1_quality(entry),
                                metadata={
                                    "product_id": product_id,
                                    "satellite": entry.get('platformname'),
                                    "instrument": entry.get('instrumentname', 'C-SAR'),
                                    "processing_level": entry.get('processinglevel', 'L1'),
                                    "footprint": footprint
                                }
                            )
                            
                            yield observation
                            
        except Exception as e:
            logger.error(f"Error fetching Sentinel-1 data: {e}")
    
    async def _fetch_sentinel2_data(self, bbox: List[float]) -> AsyncIterator[GeologicalObservation]:
        """Fetch Sentinel-2 optical data"""
        try:
            search_url = f"{self.config.base_url}{self.endpoints['search']}"
            
            params = {
                'platformname': 'Sentinel-2',
                'producttype': 'S2MSI2A',  # Level-2A product
                'cloudcoverpercentage': '[0 TO 30]',  # Low cloud cover
                'beginposition': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'endposition': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'footprint': f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))"
            }
            
            response = await self._make_request('GET', search_url, params=params)
            
            if response and 'feed' in response and 'entry' in response['feed']:
                for entry in response['feed']['entry']:
                    if isinstance(entry, dict):
                        product_id = entry['id']
                        sensing_time = datetime.fromisoformat(entry['date'][0]['content'].replace('Z', '+00:00'))
                        
                        footprint = entry.get('str', [])
                        coordinates = self._parse_footprint(footprint)
                        
                        if coordinates:
                            center_lat = sum(coord[1] for coord in coordinates) / len(coordinates)
                            center_lon = sum(coord[0] for coord in coordinates) / len(coordinates)
                            
                            observation = GeologicalObservation(
                                observation_id=f"copernicus_s2_{product_id}",
                                timestamp=sensing_time,
                                latitude=center_lat,
                                longitude=center_lon,
                                observation_type="optical_imagery",
                                value={
                                    "mission": "Sentinel-2",
                                    "product_type": "S2MSI2A",
                                    "cloud_cover_percent": entry.get('cloudcoverpercentage', 0),
                                    "processing_level": "L2A",
                                    "tile_id": entry.get('tileid')
                                },
                                data_source="Copernicus Sentinel-2",
                                quality_flags=self._assess_sentinel2_quality(entry),
                                metadata={
                                    "product_id": product_id,
                                    "satellite": entry.get('platformname'),
                                    "instrument": "MSI",
                                    "mgrs_tile": entry.get('tileid'),
                                    "footprint": footprint
                                }
                            )
                            
                            yield observation
                            
        except Exception as e:
            logger.error(f"Error fetching Sentinel-2 data: {e}")
    
    async def _fetch_sentinel3_data(self, bbox: List[float]) -> AsyncIterator[GeologicalObservation]:
        """Fetch Sentinel-3 thermal and ocean data"""
        try:
            search_url = f"{self.config.base_url}{self.endpoints['search']}"
            
            params = {
                'platformname': 'Sentinel-3',
                'producttype': 'SL_2_LST___',  # Land Surface Temperature
                'beginposition': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'endposition': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'footprint': f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))"
            }
            
            response = await self._make_request('GET', search_url, params=params)
            
            if response and 'feed' in response and 'entry' in response['feed']:
                for entry in response['feed']['entry']:
                    if isinstance(entry, dict):
                        product_id = entry['id']
                        sensing_time = datetime.fromisoformat(entry['date'][0]['content'].replace('Z', '+00:00'))
                        
                        footprint = entry.get('str', [])
                        coordinates = self._parse_footprint(footprint)
                        
                        if coordinates:
                            center_lat = sum(coord[1] for coord in coordinates) / len(coordinates)
                            center_lon = sum(coord[0] for coord in coordinates) / len(coordinates)
                            
                            observation = GeologicalObservation(
                                observation_id=f"copernicus_s3_{product_id}",
                                timestamp=sensing_time,
                                latitude=center_lat,
                                longitude=center_lon,
                                observation_type="land_surface_temperature",
                                value={
                                    "mission": "Sentinel-3",
                                    "product_type": "SL_2_LST___",
                                    "instrument": "SLSTR",
                                    "processing_level": "L2"
                                },
                                data_source="Copernicus Sentinel-3",
                                quality_flags=self._assess_sentinel3_quality(entry),
                                metadata={
                                    "product_id": product_id,
                                    "satellite": entry.get('platformname'),
                                    "instrument": "SLSTR",
                                    "footprint": footprint
                                }
                            )
                            
                            yield observation
                            
        except Exception as e:
            logger.error(f"Error fetching Sentinel-3 data: {e}")
    
    async def _fetch_sentinel5p_data(self, bbox: List[float]) -> AsyncIterator[GeologicalObservation]:
        """Fetch Sentinel-5P atmospheric data"""
        try:
            search_url = f"{self.config.base_url}{self.endpoints['search']}"
            
            params = {
                'platformname': 'Sentinel-5P',
                'producttype': 'L2__SO2___',  # Sulfur Dioxide (volcanic emissions)
                'beginposition': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'endposition': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'footprint': f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))"
            }
            
            response = await self._make_request('GET', search_url, params=params)
            
            if response and 'feed' in response and 'entry' in response['feed']:
                for entry in response['feed']['entry']:
                    if isinstance(entry, dict):
                        product_id = entry['id']
                        sensing_time = datetime.fromisoformat(entry['date'][0]['content'].replace('Z', '+00:00'))
                        
                        footprint = entry.get('str', [])
                        coordinates = self._parse_footprint(footprint)
                        
                        if coordinates:
                            center_lat = sum(coord[1] for coord in coordinates) / len(coordinates)
                            center_lon = sum(coord[0] for coord in coordinates) / len(coordinates)
                            
                            observation = GeologicalObservation(
                                observation_id=f"copernicus_s5p_{product_id}",
                                timestamp=sensing_time,
                                latitude=center_lat,
                                longitude=center_lon,
                                observation_type="atmospheric_so2",
                                value={
                                    "mission": "Sentinel-5P",
                                    "product_type": "L2__SO2___",
                                    "instrument": "TROPOMI",
                                    "processing_level": "L2"
                                },
                                data_source="Copernicus Sentinel-5P",
                                quality_flags=self._assess_sentinel5p_quality(entry),
                                metadata={
                                    "product_id": product_id,
                                    "satellite": entry.get('platformname'),
                                    "instrument": "TROPOMI",
                                    "footprint": footprint
                                }
                            )
                            
                            yield observation
                            
        except Exception as e:
            logger.error(f"Error fetching Sentinel-5P data: {e}")
    
    def _parse_footprint(self, footprint_data: List[Dict]) -> Optional[List[tuple]]:
        """Parse footprint coordinates from Copernicus metadata"""
        try:
            for item in footprint_data:
                if item.get('name') == 'footprint':
                    footprint_str = item.get('content', '')
                    if 'POLYGON' in footprint_str:
                        # Extract coordinates from POLYGON string
                        coords_str = footprint_str.split('((')[1].split('))')[0]
                        coord_pairs = coords_str.split(',')
                        coordinates = []
                        for pair in coord_pairs:
                            lon, lat = map(float, pair.strip().split())
                            coordinates.append((lon, lat))
                        return coordinates
        except Exception as e:
            logger.warning(f"Failed to parse footprint: {e}")
        return None
    
    async def fetch_historical_data(self, start_time: datetime, end_time: datetime, **kwargs) -> List[GeologicalObservation]:
        """Fetch historical Copernicus satellite data"""
        observations = []
        
        try:
            search_url = f"{self.config.base_url}{self.endpoints['search']}"
            
            # Search parameters
            params = {
                'platformname': kwargs.get('mission', 'Sentinel-1'),
                'beginposition': start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'endposition': end_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'rows': kwargs.get('max_results', 100)
            }
            
            if 'bbox' in kwargs:
                bbox = kwargs['bbox']
                params['footprint'] = f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))"
            
            response = await self._make_request('GET', search_url, params=params)
            
            if response and 'feed' in response and 'entry' in response['feed']:
                for entry in response['feed']['entry']:
                    if isinstance(entry, dict):
                        product_id = entry['id']
                        sensing_time = datetime.fromisoformat(entry['date'][0]['content'].replace('Z', '+00:00'))
                        
                        observation = GeologicalObservation(
                            observation_id=f"copernicus_historical_{product_id}",
                            timestamp=sensing_time,
                            latitude=0.0,  # Will be updated with actual coordinates
                            longitude=0.0,
                            observation_type="satellite_product",
                            value={
                                "mission": params['platformname'],
                                "product_id": product_id
                            },
                            data_source="Copernicus",
                            metadata={
                                "product_id": product_id,
                                "satellite": entry.get('platformname')
                            }
                        )
                        
                        if await self.validate_data_quality(observation):
                            observations.append(observation)
            
            logger.info(f"Fetched {len(observations)} historical Copernicus observations")
            
        except Exception as e:
            logger.error(f"Error fetching Copernicus historical data: {e}")
            raise
            
        return observations
    
    def _assess_sentinel1_quality(self, entry: Dict[str, Any]) -> List[str]:
        """Assess quality flags for Sentinel-1 data"""
        flags = []
        
        if entry.get('processinglevel') != 'L1':
            flags.append("non_standard_processing")
            
        return flags
    
    def _assess_sentinel2_quality(self, entry: Dict[str, Any]) -> List[str]:
        """Assess quality flags for Sentinel-2 data"""
        flags = []
        
        cloud_cover = entry.get('cloudcoverpercentage', 0)
        if cloud_cover > 30:
            flags.append("high_cloud_cover")
            
        return flags
    
    def _assess_sentinel3_quality(self, entry: Dict[str, Any]) -> List[str]:
        """Assess quality flags for Sentinel-3 data"""
        flags = []
        
        if entry.get('processinglevel') != 'L2':
            flags.append("non_standard_processing")
            
        return flags
    
    def _assess_sentinel5p_quality(self, entry: Dict[str, Any]) -> List[str]:
        """Assess quality flags for Sentinel-5P data"""
        flags = []
        
        if entry.get('processinglevel') != 'L2':
            flags.append("non_standard_processing")
            
        return flags
    
    async def validate_data_quality(self, observation: GeologicalObservation) -> bool:
        """Validate Copernicus satellite data quality"""
        # Basic validation checks
        if observation.latitude < -90 or observation.latitude > 90:
            return False
        if observation.longitude < -180 or observation.longitude > 180:
            return False
        
        # Check for excessive quality flags
        if len(observation.quality_flags) > 2:
            return False
            
        return True