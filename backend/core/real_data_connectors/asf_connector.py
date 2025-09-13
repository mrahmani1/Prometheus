"""
ASF-Hyp3 InSAR Data Connector

Provides authenticated access to Alaska Satellite Facility (ASF) Hyp3 
InSAR processing service for deformation analysis and ground motion monitoring.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncIterator, List, Optional
import logging
import json

from .base_connector import BaseDataConnector, GeologicalObservation, DataSourceConfig

logger = logging.getLogger(__name__)


class ASFHyp3Connector(BaseDataConnector):
    """ASF-Hyp3 InSAR connector for deformation analysis"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.access_token: Optional[str] = None
        
        # ASF-Hyp3 endpoints
        self.endpoints = {
            'auth': '/auth',
            'jobs': '/jobs',
            'products': '/products',
            'search': '/search',
            'insar': '/insar',
            'rtc': '/rtc',
            'autorift': '/autorift'
        }
        
        # Processing types available
        self.processing_types = {
            'INSAR_GAMMA': 'interferometric_sar',
            'RTC_GAMMA': 'radiometric_terrain_correction',
            'AUTORIFT': 'feature_tracking'
        }
    
    async def _authenticate(self) -> bool:
        """Authenticate with ASF-Hyp3 using NASA Earthdata credentials"""
        try:
            if not self.config.username or not self.config.password:
                logger.error("ASF-Hyp3 requires NASA Earthdata username and password")
                return False
            
            # ASF uses NASA Earthdata authentication
            auth_url = f"{self.config.base_url}{self.endpoints['auth']}"
            
            auth_data = {
                'username': self.config.username,
                'password': self.config.password
            }
            
            response = await self._make_request('POST', auth_url, json=auth_data)
            
            if response and 'access_token' in response:
                self.access_token = response['access_token']
                logger.info("Successfully authenticated with ASF-Hyp3")
                return True
            else:
                logger.error("Failed to obtain ASF-Hyp3 access token")
                return False
                
        except Exception as e:
            logger.error(f"ASF-Hyp3 authentication failed: {e}")
            return False
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for ASF-Hyp3 API"""
        headers = {
            'User-Agent': 'Prometheus-AI-Platform/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
            
        return headers
    
    async def fetch_real_time_data(self, **kwargs) -> AsyncIterator[GeologicalObservation]:
        """Fetch real-time InSAR processing results and deformation data"""
        try:
            # Parameters for InSAR monitoring
            processing_types = kwargs.get('processing_types', ['INSAR_GAMMA'])
            bbox = kwargs.get('bbox', [-180, -90, 180, 90])
            
            while True:
                try:
                    # Check for completed InSAR processing jobs
                    jobs_url = f"{self.config.base_url}{self.endpoints['jobs']}"
                    jobs_params = {
                        'status': 'SUCCEEDED',
                        'start': (datetime.now() - timedelta(hours=24)).isoformat(),
                        'end': datetime.now().isoformat()
                    }
                    
                    jobs_response = await self._make_request('GET', jobs_url, params=jobs_params)
                    
                    if jobs_response and 'jobs' in jobs_response:
                        for job in jobs_response['jobs']:
                            if job['job_type'] in processing_types:
                                # Fetch InSAR products for this job
                                async for obs in self._process_insar_job(job):
                                    if await self.validate_data_quality(obs):
                                        yield obs
                    
                    # Check for new deformation measurements
                    async for obs in self._fetch_deformation_measurements(bbox):
                        if await self.validate_data_quality(obs):
                            yield obs
                    
                    # Wait before next fetch
                    await asyncio.sleep(kwargs.get('fetch_interval', 1800))  # 30 minutes default
                    
                except Exception as e:
                    logger.error(f"Error in ASF-Hyp3 real-time fetch: {e}")
                    await asyncio.sleep(300)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Fatal error in ASF-Hyp3 streaming: {e}")
            raise
    
    async def _process_insar_job(self, job: Dict[str, Any]) -> AsyncIterator[GeologicalObservation]:
        """Process completed InSAR job and extract deformation measurements"""
        try:
            job_id = job['job_id']
            
            # Get job products
            products_url = f"{self.config.base_url}{self.endpoints['products']}/{job_id}"
            products_response = await self._make_request('GET', products_url)
            
            if products_response and 'products' in products_response:
                for product in products_response['products']:
                    # Extract interferogram metadata
                    metadata = product.get('metadata', {})
                    
                    # Get reference and secondary image dates
                    ref_date = datetime.fromisoformat(metadata.get('reference_date', '').replace('Z', '+00:00'))
                    sec_date = datetime.fromisoformat(metadata.get('secondary_date', '').replace('Z', '+00:00'))
                    
                    # Calculate temporal baseline
                    temporal_baseline = (sec_date - ref_date).days
                    
                    # Extract spatial information
                    bbox = metadata.get('bounding_box', {})
                    center_lat = (bbox.get('north', 0) + bbox.get('south', 0)) / 2
                    center_lon = (bbox.get('east', 0) + bbox.get('west', 0)) / 2
                    
                    # Extract deformation statistics
                    deformation_stats = metadata.get('deformation_statistics', {})
                    
                    observation = GeologicalObservation(
                        observation_id=f"asf_insar_{job_id}_{product['product_id']}",
                        timestamp=sec_date,  # Use secondary image date
                        latitude=center_lat,
                        longitude=center_lon,
                        observation_type="insar_interferogram",
                        value={
                            "temporal_baseline_days": temporal_baseline,
                            "perpendicular_baseline_m": metadata.get('perpendicular_baseline'),
                            "coherence_mean": deformation_stats.get('mean_coherence'),
                            "phase_unwrapping_quality": deformation_stats.get('unwrapping_quality'),
                            "max_deformation_cm": deformation_stats.get('max_deformation_cm'),
                            "min_deformation_cm": deformation_stats.get('min_deformation_cm'),
                            "std_deformation_cm": deformation_stats.get('std_deformation_cm'),
                            "processing_type": job['job_type']
                        },
                        uncertainty=deformation_stats.get('deformation_uncertainty_cm'),
                        data_source="ASF-Hyp3 InSAR",
                        quality_flags=self._assess_insar_quality(metadata, deformation_stats),
                        metadata={
                            "job_id": job_id,
                            "product_id": product['product_id'],
                            "reference_granule": metadata.get('reference_granule'),
                            "secondary_granule": metadata.get('secondary_granule'),
                            "satellite": metadata.get('satellite', 'Sentinel-1'),
                            "orbit_direction": metadata.get('orbit_direction'),
                            "processing_level": "L2",
                            "dem_source": metadata.get('dem_source', 'Copernicus GLO-30')
                        }
                    )
                    
                    yield observation
                    
        except Exception as e:
            logger.error(f"Error processing InSAR job {job.get('job_id', 'unknown')}: {e}")
    
    async def _fetch_deformation_measurements(self, bbox: List[float]) -> AsyncIterator[GeologicalObservation]:
        """Fetch processed deformation measurements from InSAR time series"""
        try:
            # Search for time series products in the specified area
            search_url = f"{self.config.base_url}{self.endpoints['search']}"
            
            search_params = {
                'product_type': 'INSAR_GAMMA',
                'bbox': ','.join(map(str, bbox)),
                'start_date': (datetime.now() - timedelta(days=30)).isoformat(),
                'end_date': datetime.now().isoformat(),
                'processing_level': 'L2'
            }
            
            search_response = await self._make_request('GET', search_url, params=search_params)
            
            if search_response and 'results' in search_response:
                for result in search_response['results']:
                    # Extract time series deformation data
                    time_series = result.get('time_series_data', [])
                    
                    for measurement in time_series:
                        measurement_date = datetime.fromisoformat(measurement['date'].replace('Z', '+00:00'))
                        
                        observation = GeologicalObservation(
                            observation_id=f"asf_deformation_{result['product_id']}_{measurement['point_id']}",
                            timestamp=measurement_date,
                            latitude=measurement['latitude'],
                            longitude=measurement['longitude'],
                            observation_type="ground_deformation",
                            value={
                                "displacement_mm": measurement['displacement_mm'],
                                "velocity_mm_year": measurement.get('velocity_mm_year'),
                                "coherence": measurement.get('coherence'),
                                "point_id": measurement['point_id'],
                                "reference_date": result.get('reference_date')
                            },
                            uncertainty=measurement.get('displacement_uncertainty_mm'),
                            data_source="ASF-Hyp3 InSAR",
                            quality_flags=self._assess_deformation_quality(measurement),
                            metadata={
                                "product_id": result['product_id'],
                                "point_id": measurement['point_id'],
                                "satellite": result.get('satellite', 'Sentinel-1'),
                                "processing_method": "SBAS",  # Small Baseline Subset
                                "reference_point": result.get('reference_point')
                            }
                        )
                        
                        yield observation
                        
        except Exception as e:
            logger.error(f"Error fetching deformation measurements: {e}")
    
    async def fetch_historical_data(self, start_time: datetime, end_time: datetime, **kwargs) -> List[GeologicalObservation]:
        """Fetch historical InSAR processing results"""
        observations = []
        
        try:
            # Search for historical InSAR products
            search_url = f"{self.config.base_url}{self.endpoints['search']}"
            
            search_params = {
                'product_type': kwargs.get('product_type', 'INSAR_GAMMA'),
                'start_date': start_time.isoformat(),
                'end_date': end_time.isoformat(),
                'limit': kwargs.get('max_results', 1000)
            }
            
            if 'bbox' in kwargs:
                search_params['bbox'] = ','.join(map(str, kwargs['bbox']))
            
            search_response = await self._make_request('GET', search_url, params=search_params)
            
            if search_response and 'results' in search_response:
                for result in search_response['results']:
                    product_date = datetime.fromisoformat(result['acquisition_date'].replace('Z', '+00:00'))
                    
                    # Extract bounding box center
                    bbox = result.get('bounding_box', {})
                    center_lat = (bbox.get('north', 0) + bbox.get('south', 0)) / 2
                    center_lon = (bbox.get('east', 0) + bbox.get('west', 0)) / 2
                    
                    observation = GeologicalObservation(
                        observation_id=f"asf_historical_{result['product_id']}",
                        timestamp=product_date,
                        latitude=center_lat,
                        longitude=center_lon,
                        observation_type="insar_product",
                        value={
                            "product_type": result['product_type'],
                            "processing_level": result.get('processing_level', 'L2'),
                            "temporal_baseline_days": result.get('temporal_baseline_days'),
                            "perpendicular_baseline_m": result.get('perpendicular_baseline_m')
                        },
                        data_source="ASF-Hyp3 InSAR",
                        quality_flags=self._assess_product_quality(result),
                        metadata={
                            "product_id": result['product_id'],
                            "satellite": result.get('satellite', 'Sentinel-1'),
                            "orbit_direction": result.get('orbit_direction'),
                            "processing_date": result.get('processing_date')
                        }
                    )
                    
                    if await self.validate_data_quality(observation):
                        observations.append(observation)
            
            logger.info(f"Fetched {len(observations)} historical InSAR observations from ASF-Hyp3")
            
        except Exception as e:
            logger.error(f"Error fetching ASF-Hyp3 historical data: {e}")
            raise
            
        return observations
    
    def _assess_insar_quality(self, metadata: Dict[str, Any], deformation_stats: Dict[str, Any]) -> List[str]:
        """Assess quality flags for InSAR interferograms"""
        flags = []
        
        # Check temporal baseline
        temporal_baseline = metadata.get('temporal_baseline_days', 0)
        if temporal_baseline > 48:  # More than 48 days
            flags.append("long_temporal_baseline")
        
        # Check perpendicular baseline
        perp_baseline = metadata.get('perpendicular_baseline', 0)
        if abs(perp_baseline) > 300:  # More than 300m
            flags.append("large_perpendicular_baseline")
        
        # Check coherence
        mean_coherence = deformation_stats.get('mean_coherence', 1.0)
        if mean_coherence < 0.3:
            flags.append("low_coherence")
        
        # Check unwrapping quality
        unwrapping_quality = deformation_stats.get('unwrapping_quality', 1.0)
        if unwrapping_quality < 0.7:
            flags.append("poor_unwrapping")
            
        return flags
    
    def _assess_deformation_quality(self, measurement: Dict[str, Any]) -> List[str]:
        """Assess quality flags for deformation measurements"""
        flags = []
        
        coherence = measurement.get('coherence', 1.0)
        if coherence < 0.4:
            flags.append("low_coherence")
        
        uncertainty = measurement.get('displacement_uncertainty_mm', 0)
        displacement = abs(measurement.get('displacement_mm', 0))
        
        if uncertainty > 0 and displacement > 0:
            uncertainty_ratio = uncertainty / displacement
            if uncertainty_ratio > 0.5:  # Uncertainty > 50% of signal
                flags.append("high_uncertainty")
                
        return flags
    
    def _assess_product_quality(self, product: Dict[str, Any]) -> List[str]:
        """Assess quality flags for InSAR products"""
        flags = []
        
        processing_level = product.get('processing_level', 'L2')
        if processing_level != 'L2':
            flags.append("non_standard_processing")
            
        return flags
    
    async def validate_data_quality(self, observation: GeologicalObservation) -> bool:
        """Validate InSAR data quality"""
        # Basic validation checks
        if observation.latitude < -90 or observation.latitude > 90:
            return False
        if observation.longitude < -180 or observation.longitude > 180:
            return False
        
        if observation.observation_type == "ground_deformation":
            displacement = observation.value.get("displacement_mm")
            if displacement is not None and abs(displacement) > 1000:  # > 1 meter seems unrealistic for most cases
                return False
        
        elif observation.observation_type == "insar_interferogram":
            temporal_baseline = observation.value.get("temporal_baseline_days")
            if temporal_baseline is not None and (temporal_baseline < 1 or temporal_baseline > 365):
                return False
        
        # Check for excessive quality flags
        if len(observation.quality_flags) > 4:
            return False
            
        return True