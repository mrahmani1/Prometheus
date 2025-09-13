"""
Geochemical Database Connector

Provides authenticated access to EarthChem, GEOROC, NAVDAT, and PetDB
geochemical databases for authentic rock and mineral composition data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncIterator, List, Optional
import logging
import json

from .base_connector import BaseDataConnector, GeologicalObservation, DataSourceConfig

logger = logging.getLogger(__name__)


class GeochemConnector(BaseDataConnector):
    """Geochemical database connector for EarthChem, GEOROC, NAVDAT, and PetDB"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        
        # Geochemical database endpoints
        self.endpoints = {
            'earthchem': {
                'samples': '/samples',
                'analyses': '/analyses',
                'search': '/search',
                'chemistry': '/chemistry'
            },
            'georoc': {
                'samples': '/api/v1/samples',
                'compositions': '/api/v1/compositions',
                'locations': '/api/v1/locations'
            },
            'navdat': {
                'samples': '/navdat/samples',
                'analyses': '/navdat/analyses',
                'volcanoes': '/navdat/volcanoes'
            },
            'petdb': {
                'samples': '/petdb/samples',
                'compositions': '/petdb/compositions',
                'metadata': '/petdb/metadata'
            }
        }
        
        # Database-specific configurations
        self.databases = {
            'earthchem': 'https://api.earthchem.org',
            'georoc': 'https://georoc.eu',
            'navdat': 'https://navdat.org',
            'petdb': 'https://petdb.org'
        }
    
    async def _authenticate(self) -> bool:
        """Authenticate with geochemical databases"""
        try:
            # Test connections to all databases
            auth_success = True
            
            # EarthChem authentication
            try:
                earthchem_url = f"{self.databases['earthchem']}/samples"
                earthchem_params = {'limit': 1}
                await self._make_request('GET', earthchem_url, params=earthchem_params)
                logger.info("Successfully connected to EarthChem")
            except Exception as e:
                logger.warning(f"EarthChem connection failed: {e}")
                auth_success = False
            
            # GEOROC authentication
            try:
                georoc_url = f"{self.databases['georoc']}/api/v1/samples"
                georoc_params = {'limit': 1}
                await self._make_request('GET', georoc_url, params=georoc_params)
                logger.info("Successfully connected to GEOROC")
            except Exception as e:
                logger.warning(f"GEOROC connection failed: {e}")
            
            # NAVDAT authentication
            try:
                navdat_url = f"{self.databases['navdat']}/navdat/samples"
                navdat_params = {'limit': 1}
                await self._make_request('GET', navdat_url, params=navdat_params)
                logger.info("Successfully connected to NAVDAT")
            except Exception as e:
                logger.warning(f"NAVDAT connection failed: {e}")
            
            # PetDB authentication
            try:
                petdb_url = f"{self.databases['petdb']}/petdb/samples"
                petdb_params = {'limit': 1}
                await self._make_request('GET', petdb_url, params=petdb_params)
                logger.info("Successfully connected to PetDB")
            except Exception as e:
                logger.warning(f"PetDB connection failed: {e}")
            
            return auth_success
            
        except Exception as e:
            logger.error(f"Geochemical database authentication failed: {e}")
            return False
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for geochemical databases"""
        headers = {
            'User-Agent': 'Prometheus-AI-Platform/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.config.api_key:
            headers['X-API-Key'] = self.config.api_key
            
        return headers
    
    async def fetch_real_time_data(self, **kwargs) -> AsyncIterator[GeologicalObservation]:
        """Fetch recent geochemical analyses and sample data"""
        try:
            # Parameters for geochemical data
            databases = kwargs.get('databases', ['earthchem', 'georoc', 'navdat', 'petdb'])
            sample_types = kwargs.get('sample_types', ['volcanic', 'igneous', 'metamorphic'])
            
            while True:
                try:
                    # Fetch from EarthChem
                    if 'earthchem' in databases:
                        async for obs in self._fetch_earthchem_data(sample_types):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Fetch from GEOROC
                    if 'georoc' in databases:
                        async for obs in self._fetch_georoc_data(sample_types):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Fetch from NAVDAT
                    if 'navdat' in databases:
                        async for obs in self._fetch_navdat_data():
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Fetch from PetDB
                    if 'petdb' in databases:
                        async for obs in self._fetch_petdb_data(sample_types):
                            if await self.validate_data_quality(obs):
                                yield obs
                    
                    # Wait before next fetch (geochemical data updates less frequently)
                    await asyncio.sleep(kwargs.get('fetch_interval', 86400))  # 24 hours default
                    
                except Exception as e:
                    logger.error(f"Error in geochemical real-time fetch: {e}")
                    await asyncio.sleep(3600)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Fatal error in geochemical streaming: {e}")
            raise
    
    async def _fetch_earthchem_data(self, sample_types: List[str]) -> AsyncIterator[GeologicalObservation]:
        """Fetch data from EarthChem database"""
        try:
            # Search for recent samples
            search_url = f"{self.databases['earthchem']}/samples"
            
            for sample_type in sample_types:
                params = {
                    'rock_type': sample_type,
                    'limit': 100,
                    'updated_since': (datetime.now() - timedelta(days=30)).isoformat()
                }
                
                response = await self._make_request('GET', search_url, params=params)
                
                if response and 'samples' in response:
                    for sample in response['samples']:
                        # Get detailed chemistry for this sample
                        chemistry_url = f"{self.databases['earthchem']}/chemistry/{sample['sample_id']}"
                        chemistry_response = await self._make_request('GET', chemistry_url)
                        
                        if chemistry_response and 'analyses' in chemistry_response:
                            for analysis in chemistry_response['analyses']:
                                observation = GeologicalObservation(
                                    observation_id=f"earthchem_{sample['sample_id']}_{analysis['analysis_id']}",
                                    timestamp=datetime.fromisoformat(analysis.get('analysis_date', sample['collection_date']).replace('Z', '+00:00')),
                                    latitude=sample['latitude'],
                                    longitude=sample['longitude'],
                                    elevation=sample.get('elevation'),
                                    observation_type="geochemical_analysis",
                                    value={
                                        "SiO2_wt_percent": analysis.get('SiO2'),
                                        "TiO2_wt_percent": analysis.get('TiO2'),
                                        "Al2O3_wt_percent": analysis.get('Al2O3'),
                                        "FeO_wt_percent": analysis.get('FeO'),
                                        "MnO_wt_percent": analysis.get('MnO'),
                                        "MgO_wt_percent": analysis.get('MgO'),
                                        "CaO_wt_percent": analysis.get('CaO'),
                                        "Na2O_wt_percent": analysis.get('Na2O'),
                                        "K2O_wt_percent": analysis.get('K2O'),
                                        "P2O5_wt_percent": analysis.get('P2O5'),
                                        "rock_type": sample['rock_type'],
                                        "sample_type": sample.get('sample_type')
                                    },
                                    data_source="EarthChem",
                                    quality_flags=self._assess_earthchem_quality(sample, analysis),
                                    metadata={
                                        "sample_id": sample['sample_id'],
                                        "analysis_id": analysis['analysis_id'],
                                        "analytical_method": analysis.get('method'),
                                        "laboratory": analysis.get('laboratory'),
                                        "reference": sample.get('reference'),
                                        "collector": sample.get('collector'),
                                        "collection_date": sample.get('collection_date')
                                    }
                                )
                                
                                yield observation
                                
        except Exception as e:
            logger.error(f"Error fetching EarthChem data: {e}")
    
    async def _fetch_georoc_data(self, sample_types: List[str]) -> AsyncIterator[GeologicalObservation]:
        """Fetch data from GEOROC database"""
        try:
            samples_url = f"{self.databases['georoc']}/api/v1/samples"
            
            for sample_type in sample_types:
                params = {
                    'rock_type': sample_type,
                    'limit': 100,
                    'tectonic_setting': 'volcanic_arc,mid_ocean_ridge,hotspot'
                }
                
                response = await self._make_request('GET', samples_url, params=params)
                
                if response and 'data' in response:
                    for sample in response['data']:
                        # Get compositions for this sample
                        comp_url = f"{self.databases['georoc']}/api/v1/compositions"
                        comp_params = {'sample_id': sample['sample_id']}
                        
                        comp_response = await self._make_request('GET', comp_url, params=comp_params)
                        
                        if comp_response and 'compositions' in comp_response:
                            for composition in comp_response['compositions']:
                                observation = GeologicalObservation(
                                    observation_id=f"georoc_{sample['sample_id']}_{composition['composition_id']}",
                                    timestamp=datetime.fromisoformat(sample.get('collection_date', '2000-01-01').replace('Z', '+00:00')),
                                    latitude=sample['latitude'],
                                    longitude=sample['longitude'],
                                    elevation=sample.get('elevation'),
                                    observation_type="volcanic_geochemistry",
                                    value={
                                        "SiO2_wt_percent": composition.get('SiO2'),
                                        "TiO2_wt_percent": composition.get('TiO2'),
                                        "Al2O3_wt_percent": composition.get('Al2O3'),
                                        "Fe2O3_wt_percent": composition.get('Fe2O3'),
                                        "MgO_wt_percent": composition.get('MgO'),
                                        "CaO_wt_percent": composition.get('CaO'),
                                        "Na2O_wt_percent": composition.get('Na2O'),
                                        "K2O_wt_percent": composition.get('K2O'),
                                        "rock_type": sample['rock_type'],
                                        "tectonic_setting": sample.get('tectonic_setting')
                                    },
                                    data_source="GEOROC",
                                    quality_flags=self._assess_georoc_quality(sample, composition),
                                    metadata={
                                        "sample_id": sample['sample_id'],
                                        "composition_id": composition['composition_id'],
                                        "volcano_name": sample.get('volcano_name'),
                                        "eruption_age": sample.get('eruption_age'),
                                        "reference": sample.get('reference')
                                    }
                                )
                                
                                yield observation
                                
        except Exception as e:
            logger.error(f"Error fetching GEOROC data: {e}")
    
    async def _fetch_navdat_data(self) -> AsyncIterator[GeologicalObservation]:
        """Fetch data from NAVDAT (North American Volcanic Database)"""
        try:
            samples_url = f"{self.databases['navdat']}/navdat/samples"
            
            params = {
                'region': 'north_america',
                'limit': 100,
                'updated_since': (datetime.now() - timedelta(days=90)).isoformat()
            }
            
            response = await self._make_request('GET', samples_url, params=params)
            
            if response and 'samples' in response:
                for sample in response['samples']:
                    # Get analyses for this sample
                    analyses_url = f"{self.databases['navdat']}/navdat/analyses"
                    analyses_params = {'sample_id': sample['sample_id']}
                    
                    analyses_response = await self._make_request('GET', analyses_url, params=analyses_params)
                    
                    if analyses_response and 'analyses' in analyses_response:
                        for analysis in analyses_response['analyses']:
                            observation = GeologicalObservation(
                                observation_id=f"navdat_{sample['sample_id']}_{analysis['analysis_id']}",
                                timestamp=datetime.fromisoformat(sample.get('collection_date', '2000-01-01').replace('Z', '+00:00')),
                                latitude=sample['latitude'],
                                longitude=sample['longitude'],
                                elevation=sample.get('elevation'),
                                observation_type="north_american_volcanic_geochemistry",
                                value={
                                    "SiO2_wt_percent": analysis.get('SiO2'),
                                    "TiO2_wt_percent": analysis.get('TiO2'),
                                    "Al2O3_wt_percent": analysis.get('Al2O3'),
                                    "FeO_total_wt_percent": analysis.get('FeO_total'),
                                    "MgO_wt_percent": analysis.get('MgO'),
                                    "CaO_wt_percent": analysis.get('CaO'),
                                    "Na2O_wt_percent": analysis.get('Na2O'),
                                    "K2O_wt_percent": analysis.get('K2O'),
                                    "volcanic_field": sample.get('volcanic_field'),
                                    "eruption_age_ma": sample.get('eruption_age_ma')
                                },
                                data_source="NAVDAT",
                                quality_flags=self._assess_navdat_quality(sample, analysis),
                                metadata={
                                    "sample_id": sample['sample_id'],
                                    "analysis_id": analysis['analysis_id'],
                                    "volcanic_field": sample.get('volcanic_field'),
                                    "state_province": sample.get('state_province'),
                                    "reference": sample.get('reference')
                                }
                            )
                            
                            yield observation
                            
        except Exception as e:
            logger.error(f"Error fetching NAVDAT data: {e}")
    
    async def _fetch_petdb_data(self, sample_types: List[str]) -> AsyncIterator[GeologicalObservation]:
        """Fetch data from PetDB (Petrological Database)"""
        try:
            samples_url = f"{self.databases['petdb']}/petdb/samples"
            
            for sample_type in sample_types:
                params = {
                    'rock_type': sample_type,
                    'limit': 100,
                    'ocean_basin': 'all'
                }
                
                response = await self._make_request('GET', samples_url, params=params)
                
                if response and 'samples' in response:
                    for sample in response['samples']:
                        # Get compositions for this sample
                        comp_url = f"{self.databases['petdb']}/petdb/compositions"
                        comp_params = {'sample_id': sample['sample_id']}
                        
                        comp_response = await self._make_request('GET', comp_url, params=comp_params)
                        
                        if comp_response and 'compositions' in comp_response:
                            for composition in comp_response['compositions']:
                                observation = GeologicalObservation(
                                    observation_id=f"petdb_{sample['sample_id']}_{composition['composition_id']}",
                                    timestamp=datetime.fromisoformat(sample.get('collection_date', '2000-01-01').replace('Z', '+00:00')),
                                    latitude=sample['latitude'],
                                    longitude=sample['longitude'],
                                    elevation=sample.get('water_depth', 0) * -1,  # Convert depth to elevation
                                    observation_type="marine_igneous_geochemistry",
                                    value={
                                        "SiO2_wt_percent": composition.get('SiO2'),
                                        "TiO2_wt_percent": composition.get('TiO2'),
                                        "Al2O3_wt_percent": composition.get('Al2O3'),
                                        "FeO_total_wt_percent": composition.get('FeO_total'),
                                        "MgO_wt_percent": composition.get('MgO'),
                                        "CaO_wt_percent": composition.get('CaO'),
                                        "Na2O_wt_percent": composition.get('Na2O'),
                                        "K2O_wt_percent": composition.get('K2O'),
                                        "ocean_basin": sample.get('ocean_basin'),
                                        "tectonic_setting": sample.get('tectonic_setting')
                                    },
                                    data_source="PetDB",
                                    quality_flags=self._assess_petdb_quality(sample, composition),
                                    metadata={
                                        "sample_id": sample['sample_id'],
                                        "composition_id": composition['composition_id'],
                                        "cruise": sample.get('cruise'),
                                        "dive": sample.get('dive'),
                                        "water_depth_m": sample.get('water_depth'),
                                        "reference": sample.get('reference')
                                    }
                                )
                                
                                yield observation
                                
        except Exception as e:
            logger.error(f"Error fetching PetDB data: {e}")
    
    async def fetch_historical_data(self, start_time: datetime, end_time: datetime, **kwargs) -> List[GeologicalObservation]:
        """Fetch historical geochemical data"""
        observations = []
        
        try:
            databases = kwargs.get('databases', ['earthchem', 'georoc'])
            
            # Fetch from each database
            for database in databases:
                if database == 'earthchem':
                    search_url = f"{self.databases['earthchem']}/samples"
                    params = {
                        'collection_date_start': start_time.isoformat(),
                        'collection_date_end': end_time.isoformat(),
                        'limit': kwargs.get('max_results', 1000)
                    }
                    
                elif database == 'georoc':
                    search_url = f"{self.databases['georoc']}/api/v1/samples"
                    params = {
                        'collection_date_start': start_time.isoformat(),
                        'collection_date_end': end_time.isoformat(),
                        'limit': kwargs.get('max_results', 1000)
                    }
                
                else:
                    continue
                
                response = await self._make_request('GET', search_url, params=params)
                
                if response and ('samples' in response or 'data' in response):
                    samples = response.get('samples', response.get('data', []))
                    
                    for sample in samples:
                        observation = GeologicalObservation(
                            observation_id=f"{database}_historical_{sample['sample_id']}",
                            timestamp=datetime.fromisoformat(sample.get('collection_date', '2000-01-01').replace('Z', '+00:00')),
                            latitude=sample['latitude'],
                            longitude=sample['longitude'],
                            elevation=sample.get('elevation'),
                            observation_type="geochemical_sample",
                            value={
                                "rock_type": sample.get('rock_type'),
                                "sample_type": sample.get('sample_type')
                            },
                            data_source=database.upper(),
                            metadata={
                                "sample_id": sample['sample_id'],
                                "reference": sample.get('reference')
                            }
                        )
                        
                        if await self.validate_data_quality(observation):
                            observations.append(observation)
            
            logger.info(f"Fetched {len(observations)} historical geochemical observations")
            
        except Exception as e:
            logger.error(f"Error fetching geochemical historical data: {e}")
            raise
            
        return observations
    
    def _assess_earthchem_quality(self, sample: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        """Assess quality flags for EarthChem data"""
        flags = []
        
        if not analysis.get('method'):
            flags.append("missing_analytical_method")
        
        # Check for major element sum
        major_elements = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O']
        total = sum(analysis.get(element, 0) for element in major_elements)
        
        if total < 95 or total > 105:  # Should be close to 100%
            flags.append("abnormal_major_element_sum")
            
        return flags
    
    def _assess_georoc_quality(self, sample: Dict[str, Any], composition: Dict[str, Any]) -> List[str]:
        """Assess quality flags for GEOROC data"""
        flags = []
        
        if not sample.get('tectonic_setting'):
            flags.append("missing_tectonic_setting")
            
        return flags
    
    def _assess_navdat_quality(self, sample: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        """Assess quality flags for NAVDAT data"""
        flags = []
        
        if not sample.get('volcanic_field'):
            flags.append("missing_volcanic_field")
            
        return flags
    
    def _assess_petdb_quality(self, sample: Dict[str, Any], composition: Dict[str, Any]) -> List[str]:
        """Assess quality flags for PetDB data"""
        flags = []
        
        if not sample.get('ocean_basin'):
            flags.append("missing_ocean_basin")
            
        return flags
    
    async def validate_data_quality(self, observation: GeologicalObservation) -> bool:
        """Validate geochemical data quality"""
        # Basic validation checks
        if observation.latitude < -90 or observation.latitude > 90:
            return False
        if observation.longitude < -180 or observation.longitude > 180:
            return False
        
        # Check for reasonable oxide values
        for key, value in observation.value.items():
            if key.endswith('_wt_percent') and value is not None:
                if value < 0 or value > 100:
                    return False
        
        # Check for excessive quality flags
        if len(observation.quality_flags) > 3:
            return False
            
        return True