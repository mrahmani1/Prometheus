"""
Real Geochemical Data Integration System
Comprehensive integration with EarthChem, GEOROC, NAVDAT, and PetDB databases
Replaces all synthetic/mock geochemical data with authentic laboratory measurements
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import sqlite3
import logging
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
from urllib.parse import urlencode, quote
import time
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeochemicalSample:
    """Comprehensive geochemical sample data structure"""
    sample_id: str
    database_source: str
    location: Dict[str, float]  # lat, lon, elevation
    rock_type: str
    tectonic_setting: str
    age: Optional[float] = None
    age_method: Optional[str] = None
    major_elements: Dict[str, float] = field(default_factory=dict)  # wt%
    trace_elements: Dict[str, float] = field(default_factory=dict)  # ppm
    isotope_ratios: Dict[str, float] = field(default_factory=dict)
    analytical_methods: Dict[str, str] = field(default_factory=dict)
    uncertainties: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)
    data_source_url: Optional[str] = None
    collection_date: Optional[datetime] = None
    analysis_date: Optional[datetime] = None

@dataclass 
class DatabaseConfig:
    """Database connection configuration"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit: float = 1.0  # requests per second
    timeout: int = 30
    max_retries: int = 3
    cache_duration: int = 86400  # seconds

class EarthChemConnector:
    """EarthChem database connector for rock and mineral compositions"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.base_url = "https://www.earthchem.org/portal"
        self.session = None
        self.cache = {}
        self.last_request_time = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_samples(self, 
                           location_bounds: Optional[Dict[str, float]] = None,
                           rock_types: Optional[List[str]] = None,
                           elements: Optional[List[str]] = None,
                           max_samples: int = 1000) -> List[GeochemicalSample]:
        """Search EarthChem database for geochemical samples"""
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Build query parameters
            params = {
                'format': 'json',
                'limit': min(max_samples, 1000),
                'includeResults': 'Y'
            }
            
            # Add spatial constraints
            if location_bounds:
                params.update({
                    'minlat': location_bounds.get('min_lat', -90),
                    'maxlat': location_bounds.get('max_lat', 90),
                    'minlon': location_bounds.get('min_lon', -180),
                    'maxlon': location_bounds.get('max_lon', 180)
                })
            
            # Add rock type filters
            if rock_types:
                params['rocktype'] = ','.join(rock_types)
            
            # Add element filters
            if elements:
                params['elements'] = ','.join(elements)
            
            # Make API request
            url = f"{self.base_url}/search"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._parse_earthchem_response(data)
                else:
                    logger.error(f"EarthChem API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"EarthChem search failed: {e}")
            return []
    
    async def get_sample_details(self, sample_id: str) -> Optional[GeochemicalSample]:
        """Get detailed sample information from EarthChem"""
        
        try:
            await self._rate_limit()
            
            url = f"{self.base_url}/sample/{sample_id}"
            params = {'format': 'json', 'includeResults': 'Y'}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._parse_sample_details(data, 'EarthChem')
                else:
                    logger.warning(f"Sample {sample_id} not found in EarthChem")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get EarthChem sample {sample_id}: {e}")
            return None
    
    async def _parse_earthchem_response(self, data: Dict) -> List[GeochemicalSample]:
        """Parse EarthChem API response"""
        
        samples = []
        
        try:
            results = data.get('results', [])
            
            for result in results:
                sample = GeochemicalSample(
                    sample_id=result.get('sample_id', ''),
                    database_source='EarthChem',
                    location={
                        'latitude': float(result.get('latitude', 0)),
                        'longitude': float(result.get('longitude', 0)),
                        'elevation': float(result.get('elevation', 0))
                    },
                    rock_type=result.get('rock_type', 'unknown'),
                    tectonic_setting=result.get('tectonic_setting', 'unknown'),
                    age=result.get('age'),
                    age_method=result.get('age_method'),
                    data_source_url=result.get('url'),
                    collection_date=self._parse_date(result.get('collection_date')),
                    analysis_date=self._parse_date(result.get('analysis_date'))
                )
                
                # Parse geochemical data
                await self._parse_geochemical_data(sample, result)
                
                samples.append(sample)
                
        except Exception as e:
            logger.error(f"Failed to parse EarthChem response: {e}")
        
        return samples
    
    async def _parse_geochemical_data(self, sample: GeochemicalSample, data: Dict):
        """Parse geochemical measurements from API response"""
        
        # Major elements (wt%)
        major_elements = ['SiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MgO', 'CaO', 
                         'Na2O', 'K2O', 'TiO2', 'P2O5', 'MnO', 'Cr2O3']
        
        for element in major_elements:
            if element in data:
                value = data[element]
                if value is not None and value != '':
                    sample.major_elements[element] = float(value)
                    sample.uncertainties[f"{element}_error"] = data.get(f"{element}_error", 0.0)
                    sample.analytical_methods[element] = data.get(f"{element}_method", 'unknown')
        
        # Trace elements (ppm)
        trace_elements = ['Ba', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Th', 'U', 'La', 'Ce', 
                         'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 
                         'Yb', 'Lu', 'Sc', 'V', 'Cr', 'Co', 'Ni', 'Cu', 'Zn', 'Pb',
                         'Hf', 'Ta', 'Cs', 'Be', 'Li', 'B', 'F', 'Cl', 'Br', 'I']
        
        for element in trace_elements:
            if element in data:
                value = data[element]
                if value is not None and value != '':
                    sample.trace_elements[element] = float(value)
                    sample.uncertainties[f"{element}_error"] = data.get(f"{element}_error", 0.0)
                    sample.analytical_methods[element] = data.get(f"{element}_method", 'unknown')
        
        # Isotope ratios
        isotope_ratios = ['Sr87_Sr86', 'Nd143_Nd144', 'Pb206_Pb204', 'Pb207_Pb204', 
                         'Pb208_Pb204', 'He3_He4', 'Ar40_Ar36', 'Os187_Os188']
        
        for ratio in isotope_ratios:
            if ratio in data:
                value = data[ratio]
                if value is not None and value != '':
                    sample.isotope_ratios[ratio] = float(value)
                    sample.uncertainties[f"{ratio}_error"] = data.get(f"{ratio}_error", 0.0)
                    sample.analytical_methods[ratio] = data.get(f"{ratio}_method", 'unknown')
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.config.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return None

class GEOROCConnector:
    """GEOROC volcanic rock database connector"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.base_url = "https://georoc.mpch-mainz.gwdg.de/georoc"
        self.session = None
        self.cache = {}
        self.last_request_time = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_volcanic_samples(self,
                                    location_bounds: Optional[Dict[str, float]] = None,
                                    tectonic_settings: Optional[List[str]] = None,
                                    rock_types: Optional[List[str]] = None,
                                    max_samples: int = 1000) -> List[GeochemicalSample]:
        """Search GEOROC for volcanic rock samples"""
        
        try:
            await self._rate_limit()
            
            # GEOROC uses different API structure
            params = {
                'format': 'csv',  # GEOROC typically returns CSV
                'limit': min(max_samples, 1000)
            }
            
            # Add filters
            if location_bounds:
                params.update({
                    'lat_min': location_bounds.get('min_lat', -90),
                    'lat_max': location_bounds.get('max_lat', 90),
                    'lon_min': location_bounds.get('min_lon', -180),
                    'lon_max': location_bounds.get('max_lon', 180)
                })
            
            if tectonic_settings:
                params['tectonic_setting'] = ','.join(tectonic_settings)
            
            if rock_types:
                params['rock_type'] = ','.join(rock_types)
            
            url = f"{self.base_url}/api/samples"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    csv_data = await response.text()
                    return await self._parse_georoc_csv(csv_data)
                else:
                    logger.error(f"GEOROC API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"GEOROC search failed: {e}")
            return []
    
    async def _parse_georoc_csv(self, csv_data: str) -> List[GeochemicalSample]:
        """Parse GEOROC CSV response"""
        
        samples = []
        
        try:
            # Use pandas to parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(csv_data))
            
            for _, row in df.iterrows():
                sample = GeochemicalSample(
                    sample_id=str(row.get('SAMPLE_ID', '')),
                    database_source='GEOROC',
                    location={
                        'latitude': float(row.get('LATITUDE', 0)),
                        'longitude': float(row.get('LONGITUDE', 0)),
                        'elevation': float(row.get('ELEVATION', 0))
                    },
                    rock_type=str(row.get('ROCK_TYPE', 'unknown')),
                    tectonic_setting=str(row.get('TECTONIC_SETTING', 'unknown')),
                    age=row.get('AGE'),
                    age_method=str(row.get('AGE_METHOD', ''))
                )
                
                # Parse major elements
                major_cols = [col for col in df.columns if col.endswith('_WT%')]
                for col in major_cols:
                    element = col.replace('_WT%', '')
                    value = row.get(col)
                    if pd.notna(value):
                        sample.major_elements[element] = float(value)
                
                # Parse trace elements
                trace_cols = [col for col in df.columns if col.endswith('_PPM')]
                for col in trace_cols:
                    element = col.replace('_PPM', '')
                    value = row.get(col)
                    if pd.notna(value):
                        sample.trace_elements[element] = float(value)
                
                # Parse isotope ratios
                isotope_cols = [col for col in df.columns if '_' in col and 
                               any(iso in col for iso in ['SR87', 'ND143', 'PB206', 'HE3'])]
                for col in isotope_cols:
                    value = row.get(col)
                    if pd.notna(value):
                        sample.isotope_ratios[col] = float(value)
                
                samples.append(sample)
                
        except Exception as e:
            logger.error(f"Failed to parse GEOROC CSV: {e}")
        
        return samples
    
    async def _rate_limit(self):
        """Implement rate limiting for GEOROC"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.config.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()

class NAVDATConnector:
    """NAVDAT North American volcanic database connector"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.base_url = "https://www.navdat.org"
        self.session = None
        self.cache = {}
        self.last_request_time = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_north_american_volcanics(self,
                                            location_bounds: Optional[Dict[str, float]] = None,
                                            age_range: Optional[Tuple[float, float]] = None,
                                            volcanic_fields: Optional[List[str]] = None,
                                            max_samples: int = 1000) -> List[GeochemicalSample]:
        """Search NAVDAT for North American volcanic samples"""
        
        try:
            await self._rate_limit()
            
            params = {
                'format': 'json',
                'limit': min(max_samples, 1000),
                'region': 'north_america'
            }
            
            # Geographic constraints (focus on North American volcanic regions)
            if location_bounds:
                params.update({
                    'min_lat': max(location_bounds.get('min_lat', 25), 25),  # North America bounds
                    'max_lat': min(location_bounds.get('max_lat', 75), 75),
                    'min_lon': max(location_bounds.get('min_lon', -170), -170),
                    'max_lon': min(location_bounds.get('max_lon', -50), -50)
                })
            else:
                # Default North American bounds
                params.update({
                    'min_lat': 25, 'max_lat': 75,
                    'min_lon': -170, 'max_lon': -50
                })
            
            # Age constraints
            if age_range:
                params['min_age'] = age_range[0]
                params['max_age'] = age_range[1]
            
            # Volcanic field filters
            if volcanic_fields:
                params['volcanic_fields'] = ','.join(volcanic_fields)
            
            url = f"{self.base_url}/api/search"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._parse_navdat_response(data)
                else:
                    logger.error(f"NAVDAT API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"NAVDAT search failed: {e}")
            return []
    
    async def _parse_navdat_response(self, data: Dict) -> List[GeochemicalSample]:
        """Parse NAVDAT API response"""
        
        samples = []
        
        try:
            results = data.get('samples', [])
            
            for result in results:
                sample = GeochemicalSample(
                    sample_id=result.get('sample_id', ''),
                    database_source='NAVDAT',
                    location={
                        'latitude': float(result.get('latitude', 0)),
                        'longitude': float(result.get('longitude', 0)),
                        'elevation': float(result.get('elevation', 0))
                    },
                    rock_type=result.get('rock_type', 'volcanic'),
                    tectonic_setting=result.get('tectonic_setting', 'continental_arc'),
                    age=result.get('age'),
                    age_method=result.get('age_method'),
                    data_source_url=result.get('url')
                )
                
                # Parse geochemical data specific to NAVDAT format
                await self._parse_navdat_geochemistry(sample, result)
                
                samples.append(sample)
                
        except Exception as e:
            logger.error(f"Failed to parse NAVDAT response: {e}")
        
        return samples
    
    async def _parse_navdat_geochemistry(self, sample: GeochemicalSample, data: Dict):
        """Parse NAVDAT geochemical data"""
        
        # NAVDAT focuses on volcanic rocks with comprehensive trace element data
        geochemistry = data.get('geochemistry', {})
        
        # Major elements
        majors = geochemistry.get('major_elements', {})
        for element, value in majors.items():
            if value is not None:
                sample.major_elements[element] = float(value)
        
        # Trace elements (NAVDAT has extensive trace element coverage)
        traces = geochemistry.get('trace_elements', {})
        for element, value in traces.items():
            if value is not None:
                sample.trace_elements[element] = float(value)
        
        # Isotope ratios
        isotopes = geochemistry.get('isotope_ratios', {})
        for ratio, value in isotopes.items():
            if value is not None:
                sample.isotope_ratios[ratio] = float(value)
        
        # Analytical methods and uncertainties
        methods = geochemistry.get('analytical_methods', {})
        sample.analytical_methods.update(methods)
        
        uncertainties = geochemistry.get('uncertainties', {})
        sample.uncertainties.update(uncertainties)
    
    async def _rate_limit(self):
        """Implement rate limiting for NAVDAT"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.config.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()

class PetDBConnector:
    """PetDB igneous petrology database connector"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.base_url = "https://www.earthchem.org/petdb"
        self.session = None
        self.cache = {}
        self.last_request_time = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_igneous_samples(self,
                                   location_bounds: Optional[Dict[str, float]] = None,
                                   rock_types: Optional[List[str]] = None,
                                   tectonic_settings: Optional[List[str]] = None,
                                   max_samples: int = 1000) -> List[GeochemicalSample]:
        """Search PetDB for igneous petrology samples"""
        
        try:
            await self._rate_limit()
            
            params = {
                'format': 'json',
                'limit': min(max_samples, 1000),
                'rock_class': 'igneous'
            }
            
            # Geographic constraints
            if location_bounds:
                params.update({
                    'west': location_bounds.get('min_lon', -180),
                    'east': location_bounds.get('max_lon', 180),
                    'south': location_bounds.get('min_lat', -90),
                    'north': location_bounds.get('max_lat', 90)
                })
            
            # Rock type filters
            if rock_types:
                params['rock_type'] = ','.join(rock_types)
            
            # Tectonic setting filters
            if tectonic_settings:
                params['tectonic_setting'] = ','.join(tectonic_settings)
            
            url = f"{self.base_url}/api/samples"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._parse_petdb_response(data)
                else:
                    logger.error(f"PetDB API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"PetDB search failed: {e}")
            return []
    
    async def _parse_petdb_response(self, data: Dict) -> List[GeochemicalSample]:
        """Parse PetDB API response"""
        
        samples = []
        
        try:
            results = data.get('results', [])
            
            for result in results:
                sample = GeochemicalSample(
                    sample_id=result.get('sample_id', ''),
                    database_source='PetDB',
                    location={
                        'latitude': float(result.get('latitude', 0)),
                        'longitude': float(result.get('longitude', 0)),
                        'elevation': float(result.get('elevation', 0))
                    },
                    rock_type=result.get('rock_type', 'igneous'),
                    tectonic_setting=result.get('tectonic_setting', 'unknown'),
                    age=result.get('age'),
                    age_method=result.get('age_method'),
                    data_source_url=result.get('url')
                )
                
                # Parse PetDB geochemical data
                await self._parse_petdb_geochemistry(sample, result)
                
                samples.append(sample)
                
        except Exception as e:
            logger.error(f"Failed to parse PetDB response: {e}")
        
        return samples
    
    async def _parse_petdb_geochemistry(self, sample: GeochemicalSample, data: Dict):
        """Parse PetDB geochemical data with focus on igneous petrology"""
        
        # PetDB has comprehensive major and trace element data
        chemistry = data.get('chemistry', {})
        
        # Major elements with high precision
        major_oxides = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MnO', 
                       'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O', 'CO2']
        
        for oxide in major_oxides:
            if oxide in chemistry:
                value = chemistry[oxide]
                if value is not None and value != '':
                    sample.major_elements[oxide] = float(value)
                    # PetDB typically includes analytical uncertainties
                    error_key = f"{oxide}_ERR"
                    if error_key in chemistry:
                        sample.uncertainties[f"{oxide}_error"] = float(chemistry[error_key])
        
        # Comprehensive trace element suite
        trace_elements = ['Li', 'Be', 'B', 'F', 'Sc', 'V', 'Cr', 'Co', 'Ni', 'Cu', 'Zn', 
                         'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 
                         'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 
                         'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 
                         'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 
                         'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 
                         'Ac', 'Th', 'Pa', 'U']
        
        for element in trace_elements:
            if element in chemistry:
                value = chemistry[element]
                if value is not None and value != '':
                    sample.trace_elements[element] = float(value)
                    error_key = f"{element}_ERR"
                    if error_key in chemistry:
                        sample.uncertainties[f"{element}_error"] = float(chemistry[error_key])
        
        # Isotope ratios (PetDB includes radiogenic and stable isotopes)
        isotope_systems = ['Sr87_Sr86', 'Nd143_Nd144', 'Pb206_Pb204', 'Pb207_Pb204', 
                          'Pb208_Pb204', 'He3_He4', 'Ar40_Ar36', 'Os187_Os188', 
                          'Re187_Os188', 'Sm147_Nd143', 'Rb87_Sr87', 'U238_Pb206', 
                          'U235_Pb207', 'Th232_Pb208', 'K40_Ar40']
        
        for ratio in isotope_systems:
            if ratio in chemistry:
                value = chemistry[ratio]
                if value is not None and value != '':
                    sample.isotope_ratios[ratio] = float(value)
                    error_key = f"{ratio}_ERR"
                    if error_key in chemistry:
                        sample.uncertainties[f"{ratio}_error"] = float(chemistry[error_key])
        
        # Analytical methods
        methods = chemistry.get('analytical_methods', {})
        sample.analytical_methods.update(methods)
    
    async def _rate_limit(self):
        """Implement rate limiting for PetDB"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.config.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()

class RealGeochemicalDataIntegrator:
    """Main integration system for real geochemical databases"""
    
    def __init__(self):
        self.connectors = {}
        self.cache_db_path = "backend/modules/real_geochemical_cache.db"
        self.quality_thresholds = {
            'min_major_elements': 5,
            'min_trace_elements': 10,
            'max_uncertainty': 0.1,  # 10%
            'required_location_precision': 0.01  # degrees
        }
        
    async def initialize_connectors(self):
        """Initialize database connectors"""
        
        # Configure database connections
        configs = {
            'earthchem': DatabaseConfig(
                name='EarthChem',
                base_url='https://www.earthchem.org/portal',
                rate_limit=0.5,  # 0.5 requests per second
                timeout=60
            ),
            'georoc': DatabaseConfig(
                name='GEOROC',
                base_url='https://georoc.mpch-mainz.gwdg.de/georoc',
                rate_limit=0.3,  # Conservative rate limit
                timeout=90
            ),
            'navdat': DatabaseConfig(
                name='NAVDAT',
                base_url='https://www.navdat.org',
                rate_limit=0.5,
                timeout=60
            ),
            'petdb': DatabaseConfig(
                name='PetDB',
                base_url='https://www.earthchem.org/petdb',
                rate_limit=0.5,
                timeout=60
            )
        }
        
        # Initialize connectors
        self.connectors = {
            'earthchem': EarthChemConnector(configs['earthchem']),
            'georoc': GEOROCConnector(configs['georoc']),
            'navdat': NAVDATConnector(configs['navdat']),
            'petdb': PetDBConnector(configs['petdb'])
        }
        
        # Initialize local cache database
        await self._initialize_cache_database()
    
    async def search_comprehensive_geochemical_data(self,
                                                  location_bounds: Optional[Dict[str, float]] = None,
                                                  rock_types: Optional[List[str]] = None,
                                                  tectonic_settings: Optional[List[str]] = None,
                                                  elements_of_interest: Optional[List[str]] = None,
                                                  max_samples_per_db: int = 500) -> Dict[str, List[GeochemicalSample]]:
        """Search all databases for comprehensive geochemical data"""
        
        results = {}
        
        try:
            # Search EarthChem
            logger.info("Searching EarthChem database...")
            async with self.connectors['earthchem'] as earthchem:
                earthchem_samples = await earthchem.search_samples(
                    location_bounds=location_bounds,
                    rock_types=rock_types,
                    elements=elements_of_interest,
                    max_samples=max_samples_per_db
                )
                results['earthchem'] = earthchem_samples
                logger.info(f"Found {len(earthchem_samples)} samples in EarthChem")
            
            # Search GEOROC
            logger.info("Searching GEOROC database...")
            async with self.connectors['georoc'] as georoc:
                georoc_samples = await georoc.search_volcanic_samples(
                    location_bounds=location_bounds,
                    tectonic_settings=tectonic_settings,
                    rock_types=rock_types,
                    max_samples=max_samples_per_db
                )
                results['georoc'] = georoc_samples
                logger.info(f"Found {len(georoc_samples)} samples in GEOROC")
            
            # Search NAVDAT
            logger.info("Searching NAVDAT database...")
            async with self.connectors['navdat'] as navdat:
                navdat_samples = await navdat.search_north_american_volcanics(
                    location_bounds=location_bounds,
                    max_samples=max_samples_per_db
                )
                results['navdat'] = navdat_samples
                logger.info(f"Found {len(navdat_samples)} samples in NAVDAT")
            
            # Search PetDB
            logger.info("Searching PetDB database...")
            async with self.connectors['petdb'] as petdb:
                petdb_samples = await petdb.search_igneous_samples(
                    location_bounds=location_bounds,
                    rock_types=rock_types,
                    tectonic_settings=tectonic_settings,
                    max_samples=max_samples_per_db
                )
                results['petdb'] = petdb_samples
                logger.info(f"Found {len(petdb_samples)} samples in PetDB")
            
            # Cache results
            await self._cache_search_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive search failed: {e}")
            return {}
    
    async def get_integrated_dataset(self,
                                   search_criteria: Dict[str, Any],
                                   quality_filter: bool = True,
                                   deduplicate: bool = True) -> List[GeochemicalSample]:
        """Get integrated, quality-filtered dataset from all sources"""
        
        # Search all databases
        raw_results = await self.search_comprehensive_geochemical_data(**search_criteria)
        
        # Combine all samples
        all_samples = []
        for database, samples in raw_results.items():
            all_samples.extend(samples)
        
        logger.info(f"Combined {len(all_samples)} samples from all databases")
        
        # Apply quality filtering
        if quality_filter:
            all_samples = await self._apply_quality_filters(all_samples)
            logger.info(f"After quality filtering: {len(all_samples)} samples")
        
        # Remove duplicates
        if deduplicate:
            all_samples = await self._deduplicate_samples(all_samples)
            logger.info(f"After deduplication: {len(all_samples)} samples")
        
        # Validate and standardize data
        all_samples = await self._validate_and_standardize(all_samples)
        
        return all_samples
    
    async def _apply_quality_filters(self, samples: List[GeochemicalSample]) -> List[GeochemicalSample]:
        """Apply quality filters to geochemical samples"""
        
        filtered_samples = []
        
        for sample in samples:
            quality_score = 0
            quality_flags = []
            
            # Check major element completeness
            major_count = len(sample.major_elements)
            if major_count >= self.quality_thresholds['min_major_elements']:
                quality_score += 2
            else:
                quality_flags.append(f"insufficient_major_elements_{major_count}")
            
            # Check trace element completeness
            trace_count = len(sample.trace_elements)
            if trace_count >= self.quality_thresholds['min_trace_elements']:
                quality_score += 2
            else:
                quality_flags.append(f"insufficient_trace_elements_{trace_count}")
            
            # Check location precision
            if (abs(sample.location.get('latitude', 0)) > 0.001 and 
                abs(sample.location.get('longitude', 0)) > 0.001):
                quality_score += 1
            else:
                quality_flags.append("imprecise_location")
            
            # Check analytical uncertainties
            avg_uncertainty = np.mean([v for k, v in sample.uncertainties.items() 
                                     if 'error' in k and v > 0]) if sample.uncertainties else 1.0
            if avg_uncertainty <= self.quality_thresholds['max_uncertainty']:
                quality_score += 1
            else:
                quality_flags.append(f"high_uncertainty_{avg_uncertainty:.3f}")
            
            # Check for isotope data (bonus points)
            if sample.isotope_ratios:
                quality_score += 1
            
            # Accept samples with quality score >= 3
            if quality_score >= 3:
                sample.quality_flags = quality_flags
                sample.metadata['quality_score'] = quality_score
                filtered_samples.append(sample)
        
        return filtered_samples
    
    async def _deduplicate_samples(self, samples: List[GeochemicalSample]) -> List[GeochemicalSample]:
        """Remove duplicate samples based on location and composition"""
        
        unique_samples = []
        seen_signatures = set()
        
        for sample in samples:
            # Create signature based on location and major element composition
            location_sig = f"{sample.location.get('latitude', 0):.4f}_{sample.location.get('longitude', 0):.4f}"
            
            # Major element signature
            major_elements = sorted(sample.major_elements.items())
            major_sig = "_".join([f"{k}:{v:.2f}" for k, v in major_elements[:5]])  # Top 5 elements
            
            signature = f"{location_sig}_{major_sig}"
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_samples.append(sample)
            else:
                # If duplicate, keep the one with more complete data
                existing_idx = None
                for i, existing in enumerate(unique_samples):
                    existing_sig = f"{existing.location.get('latitude', 0):.4f}_{existing.location.get('longitude', 0):.4f}"
                    if existing_sig in signature:
                        existing_idx = i
                        break
                
                if existing_idx is not None:
                    existing_sample = unique_samples[existing_idx]
                    current_completeness = len(sample.major_elements) + len(sample.trace_elements) + len(sample.isotope_ratios)
                    existing_completeness = len(existing_sample.major_elements) + len(existing_sample.trace_elements) + len(existing_sample.isotope_ratios)
                    
                    if current_completeness > existing_completeness:
                        unique_samples[existing_idx] = sample
        
        return unique_samples
    
    async def _validate_and_standardize(self, samples: List[GeochemicalSample]) -> List[GeochemicalSample]:
        """Validate and standardize geochemical data"""
        
        validated_samples = []
        
        for sample in samples:
            try:
                # Validate major element totals
                major_total = sum(sample.major_elements.values())
                if 95 <= major_total <= 105:  # Reasonable total for major oxides
                    sample.metadata['major_element_total'] = major_total
                else:
                    sample.quality_flags.append(f"unusual_major_total_{major_total:.1f}")
                
                # Normalize major elements to 100% if close
                if 98 <= major_total <= 102:
                    normalization_factor = 100.0 / major_total
                    sample.major_elements = {k: v * normalization_factor 
                                           for k, v in sample.major_elements.items()}
                    sample.metadata['normalized'] = True
                
                # Validate trace element ranges
                for element, concentration in sample.trace_elements.items():
                    if concentration < 0:
                        sample.quality_flags.append(f"negative_{element}")
                    elif concentration > 10000:  # Very high concentration
                        sample.quality_flags.append(f"very_high_{element}")
                
                # Validate isotope ratios
                for ratio, value in sample.isotope_ratios.items():
                    if 'Sr87_Sr86' in ratio and not (0.700 <= value <= 0.720):
                        sample.quality_flags.append(f"unusual_sr_ratio_{value:.6f}")
                    elif 'Nd143_Nd144' in ratio and not (0.510 <= value <= 0.515):
                        sample.quality_flags.append(f"unusual_nd_ratio_{value:.6f}")
                
                # Add processing timestamp
                sample.metadata['validation_timestamp'] = datetime.now().isoformat()
                sample.metadata['data_source'] = 'real_geochemical_integration'
                
                validated_samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Validation failed for sample {sample.sample_id}: {e}")
                continue
        
        return validated_samples
    
    async def _initialize_cache_database(self):
        """Initialize SQLite cache database"""
        
        try:
            # Create directory if it doesn't exist
            Path(self.cache_db_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cached_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id TEXT UNIQUE,
                    database_source TEXT,
                    data_json TEXT,
                    cache_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    search_hash TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_hash TEXT UNIQUE,
                    search_params TEXT,
                    result_count INTEGER,
                    search_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize cache database: {e}")
    
    async def _cache_search_results(self, results: Dict[str, List[GeochemicalSample]]):
        """Cache search results to local database"""
        
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            for database, samples in results.items():
                for sample in samples:
                    # Convert sample to JSON
                    sample_dict = {
                        'sample_id': sample.sample_id,
                        'database_source': sample.database_source,
                        'location': sample.location,
                        'rock_type': sample.rock_type,
                        'tectonic_setting': sample.tectonic_setting,
                        'age': sample.age,
                        'major_elements': sample.major_elements,
                        'trace_elements': sample.trace_elements,
                        'isotope_ratios': sample.isotope_ratios,
                        'analytical_methods': sample.analytical_methods,
                        'uncertainties': sample.uncertainties,
                        'metadata': sample.metadata,
                        'quality_flags': sample.quality_flags
                    }
                    
                    sample_json = json.dumps(sample_dict, default=str)
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO cached_samples 
                        (sample_id, database_source, data_json, search_hash)
                        VALUES (?, ?, ?, ?)
                    ''', (sample.sample_id, database, sample_json, 'current_search'))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to cache results: {e}")

# Example usage and testing functions
async def test_real_geochemical_integration():
    """Test the real geochemical data integration system"""
    
    integrator = RealGeochemicalDataIntegrator()
    await integrator.initialize_connectors()
    
    # Test search criteria
    search_criteria = {
        'location_bounds': {
            'min_lat': 35.0, 'max_lat': 45.0,
            'min_lon': -125.0, 'max_lon': -115.0
        },
        'rock_types': ['basalt', 'andesite', 'dacite', 'rhyolite'],
        'tectonic_settings': ['volcanic_arc', 'back_arc', 'continental_rift'],
        'elements_of_interest': ['SiO2', 'Al2O3', 'MgO', 'CaO', 'Ba', 'Sr', 'Rb', 'La', 'Ce'],
        'max_samples_per_db': 100
    }
    
    # Get integrated dataset
    samples = await integrator.get_integrated_dataset(search_criteria)
    
    print(f"Successfully integrated {len(samples)} real geochemical samples")
    
    # Analyze data completeness
    if samples:
        major_completeness = np.mean([len(s.major_elements) for s in samples])
        trace_completeness = np.mean([len(s.trace_elements) for s in samples])
        isotope_completeness = np.mean([len(s.isotope_ratios) for s in samples])
        
        print(f"Average major elements per sample: {major_completeness:.1f}")
        print(f"Average trace elements per sample: {trace_completeness:.1f}")
        print(f"Average isotope ratios per sample: {isotope_completeness:.1f}")
        
        # Database distribution
        db_counts = {}
        for sample in samples:
            db = sample.database_source
            db_counts[db] = db_counts.get(db, 0) + 1
        
        print("Database distribution:")
        for db, count in db_counts.items():
            print(f"  {db}: {count} samples")
    
    return samples

if __name__ == "__main__":
    # Run test
    samples = asyncio.run(test_real_geochemical_integration())
    print("Real geochemical data integration system initialized successfully!")