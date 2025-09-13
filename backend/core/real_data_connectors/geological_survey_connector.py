"""
National Geological Survey Connector

Provides authenticated access to national geological survey data from
USGS, BGS, GSC, GA and other geological survey organizations worldwide.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncIterator, List, Optional
import logging
import json

from .base_connector import BaseDataConnector, GeologicalObservation, DataSourceConfig

logger = logging.getLogger(__name__)


class GeologicalSurveyConnector(BaseDataConnector):
    """National geological survey connector for USGS, BGS, GSC, GA"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        
        # National geological survey endpoints
        self.surveys = {
            'usgs': {
                'base_url': 'https://api.usgs.gov',
                'endpoints': {
                    'geology': '/geology/v1',
                    'minerals': '/minerals/v1',
                    'hazards': '/hazards/v1',
                    'water': '/water/v1',
                    'maps': '/maps/v1'
                }
            },
            'bgs': {
                'base_url': 'https://api.bgs.ac.uk',
                'endpoints': {
                    'geology': '/geology/v1',
                    'boreholes': '/boreholes/v1',
                    'minerals': '/minerals/v1',
                    'hazards': '/hazards/v1'
                }
            },
            'gsc': {
                'base_url': 'https://api.nrcan.gc.ca/gsc',
                'endpoints': {
                    'geology': '/geology/v1',
                    'minerals': '/minerals/v1',
                    'fossils': '/fossils/v1',
                    'geochronology': '/geochronology/v1'
                }
            },
            'ga': {
                'base_url': 'https://api.ga.gov.au',
                'endpoints': {
                    'geology': '/geology/v1',
                    'minerals': '/minerals/v1',
                    'geophysics': '/geophysics/v1',
                    'hazards': '/hazards/v1'
                }
            }
        }
    
    async def _authenticate(self) -> bool:
        """Authenticate with geological survey APIs"""
        try:
            auth_success = True
            
            # Test USGS connection
            try:
                usgs_url = f"{self.surveys['usgs']['base_url']}/geology/v1/formations"
                usgs_params = {'limit': 1}
                await self._make_request('GET', usgs_url, params=usgs_params)
                logger.info("Successfully connected to USGS")
            except Exception as e:
                logger.warning(f"USGS connection failed: {e}")
                auth_success = False
            
            # Test BGS connection
            try:
                bgs_url = f"{self.surveys['bgs']['base_url']}/geology/v1/formations"
                bgs_params = {'limit': 1}
                await self._make_request('GET', bgs_url, params=bgs_params)
                logger.info("Successfully connected to BGS")
            except Exception as e:
                logger.warning(f"BGS connection failed: {e}")
            
            # Test GSC connection
            try:
                gsc_url = f"{self.surveys['gsc']['base_url']}/geology/v1/formations"
                gsc_params = {'limit': 1}
                await self._make_request('GET', gsc_url, params=gsc_params)
                logger.info("Successfully connected to GSC")
            except Exception as e:
                logger.warning(f"GSC connection failed: {e}")
            
            # Test GA connection
            try:
                ga_url = f"{self.surveys['ga']['base_url']}/geology/v1/formations"
                ga_params = {'limit': 1}
                await self._make_request('GET', ga_url, params=ga_params)
                logger.info("Successfully connected to GA")
            except Exception as e:
                logger.warning(f"GA connection failed: {e}")
            
            return auth_success
            
        except Exception as e:
            logger.error(f"Geological survey authentication failed: {e}")
            return False
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for geological survey APIs"""
        headers = {
            'User-Agent': 'Prometheus-AI-Platform/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.config.api_key:
            headers['X-API-Key'] = self.config.api_key
            
        return headers
    
    async def fetch_real_time_data(self, **kwargs) -> AsyncIterator[GeologicalObservation]:
        """Fetch recent geological survey data and updates"""
        try:
            # Parameters for geological survey data
            surveys = kwargs.get('surveys', ['usgs', 'bgs', 'gsc', 'ga'])
            data_types = kwargs.get('data_types', ['geology', 'minerals', 'hazards'])
            
            while True:
                try:
                    # Fetch from USGS
                    if 'usgs' in surveys:
                        async for obs in self._fetch_usgs_data(data_types):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Fetch from BGS
                    if 'bgs' in surveys:
                        async for obs in self._fetch_bgs_data(data_types):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Fetch from GSC
                    if 'gsc' in surveys:
                        async for obs in self._fetch_gsc_data(data_types):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Fetch from GA
                    if 'ga' in surveys:
                        async for obs in self._fetch_ga_data(data_types):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Wait before next fetch (geological survey data updates less frequently)
                    await asyncio.sleep(kwargs.get('fetch_interval', 86400))  # 24 hours default
                    
                except Exception as e:
                    logger.error(f"Error in geological survey real-time fetch: {e}")
                    await asyncio.sleep(3600)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Fatal error in geological survey streaming: {e}")
            raise
    
    async def _fetch_usgs_data(self, data_types: List[str]) -> AsyncIterator[GeologicalObservation]:
        """Fetch data from USGS"""
        try:
            base_url = self.surveys['usgs']['base_url']
            
            # Fetch geological formations
            if 'geology' in data_types:
                geology_url = f"{base_url}/geology/v1/formations"
                geology_params = {
                    'updated_since': (datetime.now() - timedelta(days=30)).isoformat(),
                    'limit': 100
                }
                
                response = await self._make_request('GET', geology_url, params=geology_params)
                
                if response and 'formations' in response:
                    for formation in response['formations']:
                        observation = GeologicalObservation(
                            observation_id=f"usgs_formation_{formation['formation_id']}",
                            timestamp=datetime.fromisoformat(formation.get('last_updated', '2000-01-01').replace('Z', '+00:00')),
                            latitude=formation.get('center_latitude', 0),
                            longitude=formation.get('center_longitude', 0),
                            observation_type="geological_formation",
                            value={
                                "formation_name": formation['name'],
                                "age": formation.get('age'),
                                "lithology": formation.get('lithology'),
                                "thickness_m": formation.get('thickness_meters'),
                                "area_km2": formation.get('area_km2')
                            },
                            data_source="USGS",
                            quality_flags=self._assess_usgs_quality(formation),
                            metadata={
                                "formation_id": formation['formation_id'],
                                "state": formation.get('state'),
                                "quadrangle": formation.get('quadrangle'),
                                "map_scale": formation.get('map_scale'),
                                "reference": formation.get('reference')
                            }
                        )
                        
                        yield observation
            
            # Fetch mineral resources
            if 'minerals' in data_types:
                minerals_url = f"{base_url}/minerals/v1/deposits"
                minerals_params = {
                    'updated_since': (datetime.now() - timedelta(days=30)).isoformat(),
                    'limit': 100
                }
                
                response = await self._make_request('GET', minerals_url, params=minerals_params)
                
                if response and 'deposits' in response:
                    for deposit in response['deposits']:
                        observation = GeologicalObservation(
                            observation_id=f"usgs_mineral_{deposit['deposit_id']}",
                            timestamp=datetime.fromisoformat(deposit.get('last_updated', '2000-01-01').replace('Z', '+00:00')),
                            latitude=deposit['latitude'],
                            longitude=deposit['longitude'],
                            observation_type="mineral_deposit",
                            value={
                                "deposit_name": deposit['name'],
                                "commodity": deposit.get('primary_commodity'),
                                "deposit_type": deposit.get('deposit_type'),
                                "status": deposit.get('development_status'),
                                "tonnage": deposit.get('tonnage'),
                                "grade": deposit.get('grade')
                            },
                            data_source="USGS",
                            quality_flags=self._assess_mineral_quality(deposit),
                            metadata={
                                "deposit_id": deposit['deposit_id'],
                                "country": deposit.get('country'),
                                "state": deposit.get('state'),
                                "operator": deposit.get('operator'),
                                "reference": deposit.get('reference')
                            }
                        )
                        
                        yield observation
                        
        except Exception as e:
            logger.error(f"Error fetching USGS data: {e}")
    
    async def _fetch_bgs_data(self, data_types: List[str]) -> AsyncIterator[GeologicalObservation]:
        """Fetch data from British Geological Survey"""
        try:
            base_url = self.surveys['bgs']['base_url']
            
            # Fetch borehole data
            if 'geology' in data_types:
                boreholes_url = f"{base_url}/boreholes/v1/boreholes"
                boreholes_params = {
                    'updated_since': (datetime.now() - timedelta(days=30)).isoformat(),
                    'limit': 100
                }
                
                response = await self._make_request('GET', boreholes_url, params=boreholes_params)
                
                if response and 'boreholes' in response:
                    for borehole in response['boreholes']:
                        observation = GeologicalObservation(
                            observation_id=f"bgs_borehole_{borehole['borehole_id']}",
                            timestamp=datetime.fromisoformat(borehole.get('completion_date', '2000-01-01').replace('Z', '+00:00')),
                            latitude=borehole['latitude'],
                            longitude=borehole['longitude'],
                            elevation=borehole.get('ground_level_elevation'),
                            observation_type="borehole",
                            value={
                                "borehole_name": borehole['name'],
                                "total_depth_m": borehole.get('total_depth'),
                                "purpose": borehole.get('purpose'),
                                "drilling_method": borehole.get('drilling_method'),
                                "status": borehole.get('status')
                            },
                            data_source="BGS",
                            quality_flags=self._assess_bgs_quality(borehole),
                            metadata={
                                "borehole_id": borehole['borehole_id'],
                                "operator": borehole.get('operator'),
                                "completion_date": borehole.get('completion_date'),
                                "reference": borehole.get('reference')
                            }
                        )
                        
                        yield observation
                        
        except Exception as e:
            logger.error(f"Error fetching BGS data: {e}")
    
    async def _fetch_gsc_data(self, data_types: List[str]) -> AsyncIterator[GeologicalObservation]:
        """Fetch data from Geological Survey of Canada"""
        try:
            base_url = self.surveys['gsc']['base_url']
            
            # Fetch geochronology data
            if 'geology' in data_types:
                geochronology_url = f"{base_url}/geochronology/v1/ages"
                geochronology_params = {
                    'updated_since': (datetime.now() - timedelta(days=30)).isoformat(),
                    'limit': 100
                }
                
                response = await self._make_request('GET', geochronology_url, params=geochronology_params)
                
                if response and 'ages' in response:
                    for age_data in response['ages']:
                        observation = GeologicalObservation(
                            observation_id=f"gsc_geochronology_{age_data['sample_id']}",
                            timestamp=datetime.fromisoformat(age_data.get('analysis_date', '2000-01-01').replace('Z', '+00:00')),
                            latitude=age_data['latitude'],
                            longitude=age_data['longitude'],
                            observation_type="geochronology",
                            value={
                                "age_ma": age_data.get('age_ma'),
                                "age_uncertainty_ma": age_data.get('age_uncertainty'),
                                "dating_method": age_data.get('method'),
                                "mineral": age_data.get('mineral'),
                                "rock_type": age_data.get('rock_type')
                            },
                            uncertainty=age_data.get('age_uncertainty'),
                            data_source="GSC",
                            quality_flags=self._assess_gsc_quality(age_data),
                            metadata={
                                "sample_id": age_data['sample_id'],
                                "laboratory": age_data.get('laboratory'),
                                "reference": age_data.get('reference')
                            }
                        )
                        
                        yield observation
                        
        except Exception as e:
            logger.error(f"Error fetching GSC data: {e}")
    
    async def _fetch_ga_data(self, data_types: List[str]) -> AsyncIterator[GeologicalObservation]:
        """Fetch data from Geoscience Australia"""
        try:
            base_url = self.surveys['ga']['base_url']
            
            # Fetch geophysical data
            if 'geophysics' in data_types:
                geophysics_url = f"{base_url}/geophysics/v1/surveys"
                geophysics_params = {
                    'updated_since': (datetime.now() - timedelta(days=30)).isoformat(),
                    'limit': 100
                }
                
                response = await self._make_request('GET', geophysics_url, params=geophysics_params)
                
                if response and 'surveys' in response:
                    for survey in response['surveys']:
                        observation = GeologicalObservation(
                            observation_id=f"ga_geophysics_{survey['survey_id']}",
                            timestamp=datetime.fromisoformat(survey.get('completion_date', '2000-01-01').replace('Z', '+00:00')),
                            latitude=survey.get('center_latitude', 0),
                            longitude=survey.get('center_longitude', 0),
                            observation_type="geophysical_survey",
                            value={
                                "survey_name": survey['name'],
                                "survey_type": survey.get('survey_type'),
                                "method": survey.get('method'),
                                "area_km2": survey.get('area_km2'),
                                "line_km": survey.get('total_line_km')
                            },
                            data_source="GA",
                            quality_flags=self._assess_ga_quality(survey),
                            metadata={
                                "survey_id": survey['survey_id'],
                                "operator": survey.get('operator'),
                                "completion_date": survey.get('completion_date'),
                                "reference": survey.get('reference')
                            }
                        )
                        
                        yield observation
                        
        except Exception as e:
            logger.error(f"Error fetching GA data: {e}")
    
    async def fetch_historical_data(self, start_time: datetime, end_time: datetime, **kwargs) -> List[GeologicalObservation]:
        """Fetch historical geological survey data"""
        observations = []
        
        try:
            surveys = kwargs.get('surveys', ['usgs', 'bgs'])
            
            for survey in surveys:
                if survey == 'usgs':
                    # Fetch USGS historical formations
                    geology_url = f"{self.surveys['usgs']['base_url']}/geology/v1/formations"
                    params = {
                        'created_start': start_time.isoformat(),
                        'created_end': end_time.isoformat(),
                        'limit': kwargs.get('max_results', 1000)
                    }
                    
                    response = await self._make_request('GET', geology_url, params=params)
                    
                    if response and 'formations' in response:
                        for formation in response['formations']:
                            observation = GeologicalObservation(
                                observation_id=f"usgs_historical_{formation['formation_id']}",
                                timestamp=datetime.fromisoformat(formation.get('created_date', '2000-01-01').replace('Z', '+00:00')),
                                latitude=formation.get('center_latitude', 0),
                                longitude=formation.get('center_longitude', 0),
                                observation_type="geological_formation",
                                value={
                                    "formation_name": formation['name'],
                                    "age": formation.get('age'),
                                    "lithology": formation.get('lithology')
                                },
                                data_source="USGS",
                                metadata={
                                    "formation_id": formation['formation_id'],
                                    "reference": formation.get('reference')
                                }
                            )
                            
                            if await self.validate_data_quality(observation):
                                observations.append(observation)
            
            logger.info(f"Fetched {len(observations)} historical geological survey observations")
            
        except Exception as e:
            logger.error(f"Error fetching geological survey historical data: {e}")
            raise
            
        return observations
    
    def _assess_usgs_quality(self, formation: Dict[str, Any]) -> List[str]:
        """Assess quality flags for USGS data"""
        flags = []
        
        if not formation.get('lithology'):
            flags.append("missing_lithology")
        
        if not formation.get('age'):
            flags.append("missing_age")
            
        return flags
    
    def _assess_mineral_quality(self, deposit: Dict[str, Any]) -> List[str]:
        """Assess quality flags for mineral deposit data"""
        flags = []
        
        if not deposit.get('primary_commodity'):
            flags.append("missing_commodity")
            
        return flags
    
    def _assess_bgs_quality(self, borehole: Dict[str, Any]) -> List[str]:
        """Assess quality flags for BGS data"""
        flags = []
        
        if not borehole.get('total_depth'):
            flags.append("missing_depth")
            
        return flags
    
    def _assess_gsc_quality(self, age_data: Dict[str, Any]) -> List[str]:
        """Assess quality flags for GSC data"""
        flags = []
        
        uncertainty = age_data.get('age_uncertainty', 0)
        age = age_data.get('age_ma', 0)
        
        if uncertainty > 0 and age > 0:
            uncertainty_ratio = uncertainty / age
            if uncertainty_ratio > 0.1:  # > 10% uncertainty
                flags.append("high_age_uncertainty")
                
        return flags
    
    def _assess_ga_quality(self, survey: Dict[str, Any]) -> List[str]:
        """Assess quality flags for GA data"""
        flags = []
        
        if not survey.get('survey_type'):
            flags.append("missing_survey_type")
            
        return flags
    
    async def validate_data_quality(self, observation: GeologicalObservation) -> bool:
        """Validate geological survey data quality"""
        # Basic validation checks
        if observation.latitude < -90 or observation.latitude > 90:
            return False
        if observation.longitude < -180 or observation.longitude > 180:
            return False
        
        if observation.observation_type == "geochronology":
            age = observation.value.get("age_ma")
            if age is not None and (age < 0 or age > 4600):  # Earth's age
                return False
        
        # Check for excessive quality flags
        if len(observation.quality_flags) > 3:
            return False
            
        return True